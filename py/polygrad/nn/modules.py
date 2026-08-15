"""nn.modules — Stateful neural network layers (tinygrad-compatible)."""

import math
from ..dtype import dtypes
from ..tensor import Tensor


def _mark_param(t):
    """Mark a tensor as a model parameter without changing its lazy graph."""
    t._is_param = True
    return t


class Linear:
    """y = x @ weight.T + bias"""
    def __init__(self, in_features, out_features, bias=True):
        bound = 1 / math.sqrt(in_features)
        self.weight = Tensor.uniform(out_features, in_features, low=-bound, high=bound)
        self.weight.requires_grad = True
        _mark_param(self.weight)
        self.bias = None
        if bias:
            self.bias = Tensor.uniform(out_features, low=-bound, high=bound)
            self.bias.requires_grad = True
            _mark_param(self.bias)

    def __call__(self, x):
        # Pinned nn/__init__.py:156-174 stores (out,in) and transposes at the
        # module boundary; Tensor.linear itself consumes (in,out).
        return x.linear(self.weight.transpose(), self.bias)


class LayerNorm:
    """Layer normalization."""
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.weight = None
        self.bias = None
        if elementwise_affine:
            self.weight = Tensor.ones(*normalized_shape).realize()
            self.weight.requires_grad = True
            _mark_param(self.weight)
            self.bias = Tensor.zeros(*normalized_shape).realize()
            self.bias.requires_grad = True
            _mark_param(self.bias)

    def __call__(self, x):
        axis = -1
        result = x.layernorm(axis=axis, eps=self.eps)
        if self.weight is not None:
            result = result * self.weight + self.bias
        return result


class LayerNorm2d(LayerNorm):
    """Channel-first 2D LayerNorm through the pinned NHWC composition."""

    def __call__(self, x):
        # Pinned tinygrad/nn/__init__.py:263-278.
        return super().__call__(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class GroupNorm:
    """Group normalization."""
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        if num_channels % num_groups != 0:
            raise ValueError('num_channels must be divisible by num_groups')
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.weight = None
        self.bias = None
        if affine:
            self.weight = Tensor.ones(num_channels)
            self.weight.requires_grad = True
            _mark_param(self.weight)
            self.bias = Tensor.zeros(num_channels)
            self.bias.requires_grad = True
            _mark_param(self.bias)

    def __call__(self, x):
        # Literal pinned tinygrad/nn/__init__.py:200-207 composition.
        shape = x.shape
        if len(shape) < 2:
            raise ValueError('GroupNorm expects input with at least 2 dimensions')
        if shape[1] != self.num_channels:
            raise ValueError(f'GroupNorm expected C={self.num_channels}, got C={shape[1]}')
        result = x.reshape(shape[0], self.num_groups, -1).layernorm(
            eps=self.eps
        ).reshape(*shape)
        if self.weight is None or self.bias is None:
            return result
        affine_shape = [1, -1] + [1] * (len(shape) - 2)
        return result * self.weight.reshape(*affine_shape) + self.bias.reshape(*affine_shape)


class RMSNorm:
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim, eps=1e-6, elementwise_affine=True):
        self.eps = eps
        self.weight = Tensor.ones(dim) if elementwise_affine else None
        if self.weight is not None:
            self.weight.requires_grad = True
            _mark_param(self.weight)

    def _norm(self, x):
        # Literal pinned tinygrad/nn/__init__.py:301 expression.
        return x * (x.square().mean(axis=-1, keepdim=True) + self.eps).rsqrt()

    def __call__(self, x):
        normalized = self._norm(x.float()).cast(x.dtype)
        return normalized if self.weight is None else normalized * self.weight


class Embedding:
    """Lookup table embedding (pure tensor ops, autograd-compatible).

    Port of tinygrad's Tensor._embedding_fwd:
      arange(vocab) == idx.unsqueeze(-1) → selector mask
      mask.unsqueeze(-1).where(weight, 0).sum(-2) → gathered rows
    """
    def __init__(self, vocab_size, embed_dim):
        # Pinned nn.Embedding uses this exact initializer
        # (tinygrad/nn/__init__.py:384-385; mixin/rand.py:191-204).
        self.weight = Tensor.glorot_uniform(vocab_size, embed_dim)
        self.weight.requires_grad = True
        _mark_param(self.weight)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def __call__(self, idx):
        # Pinned nn.Embedding rejects non-integer indices before building the
        # selector graph (nn/__init__.py:388-392).
        if not dtypes.is_int(idx.dtype):
            raise TypeError(f'Expected integer dtype for index in embedding, got {idx.dtype}')
        arange = Tensor.arange(self.vocab_size)
        # Preserve pinned ordered CMPNE topology: arange == idx.unsqueeze(-1)
        # (tinygrad/nn/__init__.py:368-369).
        mask = arange.eq(idx.unsqueeze(-1))
        return mask.unsqueeze(-1).where(self.weight, 0).sum(
            axis=-2, dtype=self.weight.dtype
        )


class Dropout:
    """Dropout layer (training only)."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        if not Tensor.training or self.p == 0:
            return x
        mask = Tensor.rand(*x.shape).gt(self.p)
        return x * mask / (1.0 - self.p)


class Conv2d:
    """2D convolution."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(padding, str):
            if padding.lower() != 'same':
                raise ValueError(f"Invalid padding string {padding!r}, only 'same' is supported")
            if stride != 1:
                raise ValueError("padding='same' is not supported for strided convolutions")
            dilation_tuple = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
            padding = tuple(v for d, k in zip(dilation_tuple, kernel_size[::-1])
                            for v in (d * (k - 1) // 2, d * (k - 1) - d * (k - 1) // 2))
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding = padding
        bound = 1 / math.sqrt(in_channels * kernel_size[0] * kernel_size[1])
        self.weight = Tensor.uniform(
            out_channels, in_channels // groups, *kernel_size, low=-bound, high=bound
        )
        self.weight.requires_grad = True
        _mark_param(self.weight)
        self.bias = None
        if bias:
            self.bias = Tensor.uniform(out_channels, low=-bound, high=bound)
            self.bias.requires_grad = True
            _mark_param(self.bias)

    def __call__(self, x):
        return x.conv2d(self.weight, self.bias, self.groups, self.stride, self.dilation, self.padding)


class BatchNorm:
    """Batch normalization."""
    # Pinned tinygrad/nn/__init__.py:35-60.
    def __init__(self, num_features, eps=1e-5, affine=True, track_running_stats=True, momentum=0.1):
        self.eps, self.track_running_stats, self.momentum = eps, track_running_stats, momentum
        self.weight = Tensor.ones(num_features) if affine else None
        self.bias = Tensor.zeros(num_features) if affine else None
        self.num_batches_tracked = Tensor.zeros(dtype='long').is_param_(False)
        if track_running_stats:
            self.running_mean = Tensor.zeros(num_features).is_param_(False)
            self.running_var = Tensor.ones(num_features).is_param_(False)
        if self.weight is not None:
            self.weight.requires_grad = True
            _mark_param(self.weight)
        if self.bias is not None:
            self.bias.requires_grad = True
            _mark_param(self.bias)

    def calc_stats(self, x):
        shape_mask = [1, -1, *([1] * (x.ndim - 2))]
        if self.track_running_stats and not Tensor.training:
            return self.running_mean, self.running_var.reshape(shape=shape_mask).expand(x.shape)
        reduce_axes = tuple(axis for axis in range(x.ndim) if axis != 1)
        batch_mean = x.mean(axis=reduce_axes)
        y = x - batch_mean.detach().reshape(shape=shape_mask)
        batch_var = (y * y).mean(axis=reduce_axes)
        return batch_mean, batch_var

    def __call__(self, x):
        batch_mean, batch_var = self.calc_stats(x)
        if self.track_running_stats and Tensor.training:
            self.running_mean.assign(
                (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.detach()
            )
            self.running_var.assign(
                (1 - self.momentum) * self.running_var
                + self.momentum * x.numel() / (x.numel() - x.shape[1]) * batch_var.detach()
            )
            # Pinned Tensor.__iadd__ lowers this spelling to assign(add(...)).
            self.num_batches_tracked.assign(self.num_batches_tracked + 1)
        return x.batchnorm(
            self.weight, self.bias, batch_mean, batch_var.add(self.eps).rsqrt()
        )
