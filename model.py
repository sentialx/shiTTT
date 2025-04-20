from torch import nn
import torch
from einops import rearrange, repeat
from torch.func import functional_call, grad, vmap
import torch.nn.functional as F

from agi.models.blocks.attention import apply_rotary_pos_emb
from agi.models.blocks.rotary import RotaryEmbedding

@torch.compile()
def causal_linear_attention(q, k, v, eps=1e-6):
    # q,k,v: (B,H,N,D)
    φ_q = F.softmax(q, dim=-1) * (q.shape[-1]**-0.5)
    φ_k = torch.exp(k)

    K_cum  = φ_k.cumsum(dim=2)  # (B,H,N,D)
    KV_cum = (φ_k.unsqueeze(-1) * v.unsqueeze(-2)).cumsum(dim=2) # (B,H,N,D,D)

    num = torch.einsum('b h n d, b h n d e -> b h n e', φ_q,  KV_cum)
    den = torch.einsum('b h n d, b h n d   -> b h n  ', φ_q,  K_cum).clamp(min=eps)

    D_inv = 1.0 / den.unsqueeze(-1)   # (B,H,N,1)
    attn  = num * D_inv # (B,H,N,D)
    return attn

class MultiHeadLinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64, eps=1e-6):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.eps = eps
        inner_dim = heads * dim_head

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        
        self.rotary_emb = RotaryEmbedding(dim_head, base=10_000, max_position_embeddings=2048)

    @torch.compile()
    def forward(self, x):
        """
        x: (batch, seq_len, dim)
        returns: (batch, seq_len, dim)
        """
        b, n, _ = x.shape
        h = self.heads
        d = self.dim_head
        
        pos_ids = torch.arange(n, device=x.device).unsqueeze(0).expand(b, n)
        rotary_emb = self.rotary_emb(x, pos_ids)

        qkv = self.to_qkv(x)  # (B, N, 3*H*D)
        q, k, v = qkv.chunk(3, dim=-1)

        # -> (B, H, N, D)
        q = q.view(b, n, h, d).transpose(1, 2)
        k = k.view(b, n, h, d).transpose(1, 2)
        v = v.view(b, n, h, d).transpose(1, 2)
        
        cos, sin = rotary_emb
        q, k = apply_rotary_pos_emb(
            q, k, cos.to(x.dtype), sin.to(x.dtype)
        )

        attn = causal_linear_attention(q, k, v, eps=self.eps)

        attn = attn.transpose(1, 2).contiguous().view(b, n, h * d)

        return self.to_out(attn)  # (B, N, dim)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult), nn.GELU(), nn.Linear(dim * mult, dim)
        )

    @torch.compile()
    def forward(self, x):
        return self.net(x)


class InnerModel(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64, eps=1e-6):
        super().__init__()
        self.attn = MultiHeadLinearAttention(
            dim, heads=heads, dim_head=dim_head, eps=eps
        )
        self.ff = FeedForward(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    @torch.compile()
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class TTT2Block(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64, eps=1e-6, chunk_size=8, max_lr=0.1):
        super().__init__()
        self.inner_model = InnerModel(
            dim_head, heads=heads, dim_head=dim_head // heads, eps=eps
        )
        self.chunk_size = chunk_size
        self.heads = heads
        self.max_lr = max_lr

        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.to_lr = nn.Linear(dim, heads, bias=True)
        self.to_out = nn.Linear(dim, dim, bias=False)
        self.post_norm = nn.LayerNorm(dim)

        def model_fwd_loss(weights, pred, target):
            output = functional_call(self.inner_model, weights, pred.unsqueeze(0))
            loss = F.mse_loss(output, target.unsqueeze(0))
            return loss

        self.grad_fn = torch.compile(
            vmap(
                grad(model_fwd_loss, 0),
                in_dims=({k: 0 for k in self.inner_model.state_dict().keys()}, 0, 0),
            )
        )

        def model_forward(x, weights):
            return functional_call(self.inner_model, weights, x.unsqueeze(0))

        self.model_forward = torch.compile(
            vmap(
                model_forward,
                in_dims=(0, {k: 0 for k in self.inner_model.state_dict().keys()}),
            )
        )

    def get_initial_weights(self, batch_size):
        return {
                    name: repeat(param, "... -> b ...", b=batch_size * self.heads)
                    for name, param in self.inner_model.named_parameters()
                }

    @torch.compile()
    def get_inputs(self, x):
      x = rearrange(x, "b (n c) d -> b n c d", c=self.chunk_size)

      q = self.q(x)
      k = self.k(x)
      v = self.v(x)
      lr = self.to_lr(x).sigmoid()
      
      q = rearrange(q, "b n c (h d) -> (b h) n c d", h=self.heads)
      k = rearrange(k, "b n c (h d) -> (b h) n c d", h=self.heads)
      v = rearrange(v, "b n c (h d) -> (b h) n c d", h=self.heads)
      lr = rearrange(lr, "b n c h -> (b h) n c", h=self.heads)
      lr = lr.mean(dim=2)
      
      lr = lr * self.max_lr
      
      return q, k, v, lr, x

    def forward(self, x):
        q, k, v, lr, x = self.get_inputs(x)

        weights = self.get_initial_weights(x.shape[0])

        outs = []
        
        recon_target = v - k

        for n in range(x.shape[1]):
            kn = k[:, n, :]
            qn = q[:, n, :]
            lrn = lr[:, n]

            target = recon_target[:, n, :]
        
            grads = self.grad_fn(weights, kn, target)
            state = self.model_forward(qn, weights).squeeze(1)

            weights = {
                name: param - lrn.view(-1, *[1]*(grads[name].ndim-1)) * grads[name].detach()
                for (name, param) in weights.items()
            }
            
            state = state + qn
            
            outs.append(state)

        outs = torch.stack(outs, dim=0)
        outs = rearrange(outs, "n (b h) c d -> b (n c) (h d)", h=self.heads, n=x.shape[1])

        x = self.post_norm(outs)
        x = self.to_out(outs)

        return x
