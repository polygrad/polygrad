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
        return x.linear(self.weight, self.bias)


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
            self.weight = Tensor.ones(num_channels).realize()
            self.weight.requires_grad = True
            _mark_param(self.weight)
            self.bias = Tensor.zeros(num_channels).realize()
            self.bias.requires_grad = True
            _mark_param(self.bias)

    def __call__(self, x):
        # x: (N, C, *) → reshape to (N, G, C//G, *) → normalize over (C//G, *)
        shape = x.shape
        if len(shape) < 2:
            raise ValueError('GroupNorm expects input with at least 2 dimensions')
        N = shape[0]
        G = self.num_groups
        C = self.num_channels
        if shape[1] != C:
            raise ValueError(f'GroupNorm expected C={C}, got C={shape[1]}')
        x = x.reshape(N, G, C // G, *shape[2:])
        # Normalize over all dims after G using flattened tail.
        flat = x.reshape(N, G, -1)
        # Normalization statistics are graph nodes. Realizing them here makes
        # GroupNorm a hidden materialization boundary unlike tinygrad.
        m = flat.mean(axis=-1, keepdim=True)
        v = flat.var(axis=-1, keepdim=True, correction=0)
        flat = (flat - m) / (v + self.eps).sqrt()
        result = flat.reshape(*shape)
        if self.weight is not None:
            # Broadcast weight (C,) over spatial dims
            w_shape = [1, C] + [1] * (len(shape) - 2)
            result = result * self.weight.reshape(*w_shape) + self.bias.reshape(*w_shape)
        return result


class RMSNorm:
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.weight = Tensor.ones(dim).realize()
        self.weight.requires_grad = True
        _mark_param(self.weight)

    def __call__(self, x):
        # RMSNorm should stay lazy until the user/backend materializes the
        # enclosing graph; the previous realize cut gradients through x.
        rms = (x * x).mean(axis=-1, keepdim=True)
        x_norm = x / (rms + self.eps).sqrt()
        return x_norm * self.weight


class Embedding:
    """Lookup table embedding (pure tensor ops, autograd-compatible).

    Port of tinygrad's Tensor._embedding_fwd:
      arange(vocab) == idx.unsqueeze(-1) → selector mask
      mask.unsqueeze(-1).where(weight, 0).sum(-2) → gathered rows
    """
    def __init__(self, vocab_size, embed_dim):
        self.weight = (Tensor.randn(vocab_size, embed_dim) * 0.02).realize()
        self.weight.requires_grad = True
        _mark_param(self.weight)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def __call__(self, idx):
        # Pinned nn.Embedding rejects non-integer indices before building the
        # selector graph (nn/__init__.py:388-392).
        if not dtypes.is_int(idx.dtype):
            raise TypeError(f'Expected integer dtype for index in embedding, got {idx.dtype}')
        # idx: (*batch_dims,) integer tensor
        # arange: (vocab_size,)
        arange = Tensor.arange(self.vocab_size)
        # idx.unsqueeze(-1) == arange → (*batch_dims, vocab_size) boolean mask
        mask = idx.unsqueeze(-1).eq(arange)
        # mask.unsqueeze(-1) → (*batch_dims, vocab_size, 1)
        # where(weight, 0) → (*batch_dims, vocab_size, embed_dim)
        # sum(-2) → (*batch_dims, embed_dim)
        selected = mask.unsqueeze(-1).where(self.weight, Tensor(0.0))
        return selected.sum(axis=-2)


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
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        self.weight = Tensor.ones(num_features) if affine else None
        self.bias = Tensor.zeros(num_features) if affine else None
        self.running_mean = Tensor.zeros(num_features).is_param_(False) if track_running_stats else None
        self.running_var = Tensor.ones(num_features).is_param_(False) if track_running_stats else None
        if self.weight is not None:
            self.weight.requires_grad = True
            _mark_param(self.weight)
        if self.bias is not None:
            self.bias.requires_grad = True
            _mark_param(self.bias)

    def __call__(self, x):
        if len(x.shape) < 2:
            raise ValueError('BatchNorm expects input with at least 2 dimensions')
        C = self.num_features
        if x.shape[1] != C:
            raise ValueError(f'BatchNorm expected C={C}, got C={x.shape[1]}')

        def _channel_stats(inp):
            # Move channel axis first, flatten remaining dims: (C, -1).
            perm = (1, 0) + tuple(range(2, len(inp.shape)))
            flat = inp.permute(*perm).reshape(C, -1)
            mean = flat.mean(axis=1)
            centered = flat - mean.detach().reshape(C, 1)
            var = (centered * centered).mean(axis=1)
            return mean, var

        if Tensor.training or not self.track_running_stats:
            mean_c, var_c = _channel_stats(x)
            if self.track_running_stats:
                mom = self.momentum
                self.running_mean = (
                    (1.0 - mom) * self.running_mean + mom * mean_c.detach()
                ).is_param_(False).realize()
                denom = x.numel() - x.shape[1]
                corr = (x.numel() / denom) if denom > 0 else 1.0
                self.running_var = (
                    (1.0 - mom) * self.running_var + mom * corr * var_c.detach()
                ).is_param_(False).realize()
        else:
            mean_c, var_c = self.running_mean, self.running_var

        bshape = (1, C) + (1,) * (len(x.shape) - 2)
        mean = mean_c.reshape(*bshape)
        var = var_c.reshape(*bshape)
        out = (x - mean) / (var + self.eps).sqrt()
        if self.weight is not None:
            out = out * self.weight.reshape(*bshape) + self.bias.reshape(*bshape)
        return out
