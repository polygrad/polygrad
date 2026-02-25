"""nn.modules — Stateful neural network layers (tinygrad-compatible)."""

import math
from ..tensor import Tensor


def _mark_param(t):
    """Mark a tensor as a model parameter (for backward graph stitching)."""
    t._is_param = True
    return t


class Linear:
    """y = x @ weight.T + bias"""
    def __init__(self, in_features, out_features, bias=True):
        bound = 1 / math.sqrt(in_features)
        self.weight = (Tensor.rand(out_features, in_features) * (2 * bound) - bound).realize()
        self.weight.requires_grad = True
        _mark_param(self.weight)
        self.bias = None
        if bias:
            self.bias = (Tensor.rand(out_features) * (2 * bound) - bound).realize()
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
        m = flat.mean(axis=-1, keepdim=True).realize()
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
        rms = (x * x).mean(axis=-1, keepdim=True).realize()
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
        # idx: (*batch_dims,) integer tensor (realized to float — indices as floats)
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
    """2D convolution (tensor-op implementation without cat-chains)."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        self.stride = stride
        self.padding = padding
        bound = 1 / math.sqrt(in_channels * kernel_size[0] * kernel_size[1])
        self.weight = (Tensor.rand(out_channels, in_channels, *kernel_size) * (2 * bound) - bound).realize()
        self.weight.requires_grad = True
        _mark_param(self.weight)
        self.bias = None
        if bias:
            self.bias = (Tensor.rand(out_channels) * (2 * bound) - bound).realize()
            self.bias.requires_grad = True
            _mark_param(self.bias)

    def __call__(self, x):
        if len(x.shape) != 4:
            raise ValueError('Conv2d expects input shape (N, C, H, W)')
        N, C, H, W = x.shape
        OC, IC, KH, KW = self.weight.shape
        if C != IC:
            raise ValueError(f'Conv2d expected input channels {IC}, got {C}')

        sh, sw = self.stride
        ph, pw = self.padding
        if sh <= 0 or sw <= 0:
            raise ValueError('stride must be positive')

        if ph or pw:
            x = x.pad(((0, 0), (0, 0), (ph, ph), (pw, pw)))

        Hp, Wp = x.shape[2], x.shape[3]
        OH = (Hp - KH) // sh + 1
        OW = (Wp - KW) // sw + 1
        if OH <= 0 or OW <= 0:
            raise ValueError('kernel size/stride/padding produce non-positive output shape')

        # Sum per-channel/kernel-offset contributions:
        # out[n,o,y,x] = sum_{c,kh,kw} x[n,c,y*sh+kh,x*sw+kw] * w[o,c,kh,kw]
        out = None
        for c in range(IC):
            for kh in range(KH):
                hslice = x[:, c:c + 1, kh:kh + OH * sh, :]  # (N,1,OH*sh,Wp)
                if sh > 1:
                    hslice = hslice.reshape(N, 1, OH, sh, Wp)[:, :, :, 0, :]
                for kw in range(KW):
                    wslice = hslice[:, :, :, kw:kw + OW * sw]  # (N,1,OH,OW*sw)
                    if sw > 1:
                        wslice = wslice.reshape(N, 1, OH, OW, sw)[:, :, :, :, 0]
                    wterm = self.weight[:, c, kh, kw].reshape(1, OC, 1, 1)  # (1,OC,1,1)
                    term = wslice * wterm  # (N,OC,OH,OW)
                    out = term if out is None else (out + term)

        if self.bias is not None:
            out = out + self.bias.reshape(1, OC, 1, 1)
        return out


class BatchNorm:
    """Batch normalization."""
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        self.weight = Tensor.ones(num_features).realize() if affine else None
        self.bias = Tensor.zeros(num_features).realize() if affine else None
        self.running_mean = Tensor.zeros(num_features).realize() if track_running_stats else None
        self.running_var = Tensor.ones(num_features).realize() if track_running_stats else None
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
                self.running_mean = ((1.0 - mom) * self.running_mean + mom * mean_c.detach()).realize()
                denom = x.numel() - x.shape[1]
                corr = (x.numel() / denom) if denom > 0 else 1.0
                self.running_var = ((1.0 - mom) * self.running_var + mom * corr * var_c.detach()).realize()
        else:
            mean_c, var_c = self.running_mean, self.running_var

        bshape = (1, C) + (1,) * (len(x.shape) - 2)
        mean = mean_c.reshape(*bshape)
        var = var_c.reshape(*bshape)
        out = (x - mean) / (var + self.eps).sqrt()
        if self.weight is not None:
            out = out * self.weight.reshape(*bshape) + self.bias.reshape(*bshape)
        return out
