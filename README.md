# TTT implementation

An attempt to implement the [TTT paper](https://arxiv.org/abs/2407.04620) but by making the inner model linear attention + MLP instead of doing linear attention over grads, i.e. let torch autograd do the thing.
