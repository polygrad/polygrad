"""nn.modules — Stateful neural network layers (tinygrad-compatible)."""

import math
import ctypes
from .. import _ffi
from ..dtype import dtypes
from ..helpers import TRAINING
from ..tensor import Tensor


class Linear:
    """y = x @ weight.T + bias"""
    def __init__(self, in_features, out_features, bias=True):
        bound = 1 / math.sqrt(in_features)
        self.weight = Tensor.uniform(out_features, in_features, low=-bound, high=bound)
        self.bias = None
        if bias:
            self.bias = Tensor.uniform(out_features, low=-bound, high=bound)

    def __call__(self, x):
        core = _ffi._lib.poly_tensor_linear_apply(x._ctx, x._tensor, self.weight._tensor,
                                                  self.bias._tensor if self.bias is not None else None)
        return x._make_result_from_core(core, None)


class LayerNorm:
    """Layer normalization."""
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.axis = tuple(-1 - i for i in range(len(self.normalized_shape)))
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.weight = None
        self.bias = None
        if elementwise_affine:
            self.weight = Tensor.ones(*normalized_shape)
            self.bias = Tensor.zeros(*normalized_shape)

    def __call__(self, x):
        # Pinned nn.LayerNorm normalizes the entire declared trailing shape.
        if x.shape[-len(self.normalized_shape):] != self.normalized_shape:
            raise ValueError(f"last dimensions of {x.shape} must match {self.normalized_shape}")
        axes = (ctypes.c_int64 * len(self.axis))(*self.axis)
        core = _ffi._lib.poly_tensor_layernorm_axes_apply(x._ctx, x._tensor,
            self.weight._tensor if self.weight is not None else None,
            self.bias._tensor if self.bias is not None else None, axes, len(self.axis), self.eps)
        return x._make_result_from_core(core, x.shape)


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
            self.bias = Tensor.zeros(num_channels)

    def __call__(self, x):
        shape = x.shape
        if len(shape) < 2:
            raise ValueError('GroupNorm expects input with at least 2 dimensions')
        if shape[1] != self.num_channels:
            raise ValueError(f'GroupNorm expected C={self.num_channels}, got C={shape[1]}')
        core = _ffi._lib.poly_tensor_groupnorm_apply(x._ctx, x._tensor,
            self.weight._tensor if self.weight is not None else None,
            self.bias._tensor if self.bias is not None else None, self.num_groups, self.eps)
        return x._make_result_from_core(core, x.shape)


class RMSNorm:
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim, eps=1e-6, elementwise_affine=True):
        self.eps = eps
        self.weight = Tensor.ones(dim) if elementwise_affine else None

    def __call__(self, x):
        core = _ffi._lib.poly_tensor_rmsnorm_apply(x._ctx, x._tensor,
            self.weight._tensor if self.weight is not None else None, self.eps)
        return x._make_result_from_core(core, x.shape)


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
        # Polygrad's module convenience delegates the pinned Tensor contract,
        # including probability admission, p=1 and the shared training mode.
        return x.dropout(self.p)


class Conv2d:
    """2D convolution."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = tuple(kernel_size)
        if isinstance(padding, str):
            if padding.lower() != 'same':
                raise ValueError(f"Invalid padding string {padding!r}, only 'same' is supported")
            if stride != 1:
                raise ValueError("padding='same' is not supported for strided convolutions")
            dilation_tuple = (dilation,) * len(kernel_size) if isinstance(dilation, int) else tuple(dilation)
            padding = tuple(v for d, k in zip(dilation_tuple, kernel_size[::-1])
                            for v in (d * (k - 1) // 2, d * (k - 1) - d * (k - 1) // 2))
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding = padding
        bound = 1 / math.sqrt(in_channels * math.prod(kernel_size))
        self.weight = Tensor.uniform(
            out_channels, in_channels // groups, *kernel_size, low=-bound, high=bound
        )
        self.bias = None
        if bias:
            self.bias = Tensor.uniform(out_channels, low=-bound, high=bound)

    def __call__(self, x):
        return x.conv2d(self.weight, self.bias, self.groups, self.stride, self.dilation, self.padding)


def Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return Conv2d(in_channels, out_channels, (kernel_size,), stride, padding, dilation, groups, bias)


class ConvTranspose2d(Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0,
                 dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        # nn.ConvTranspose2d reverses the channel layout, not the bias layout.
        bound = 1 / math.sqrt(in_channels * math.prod(self.kernel_size))
        self.weight = Tensor.uniform(in_channels, out_channels // groups, *self.kernel_size, low=-bound, high=bound)
        self.output_padding = output_padding

    def __call__(self, x):
        return x.conv_transpose2d(self.weight, self.bias, self.groups, self.stride, self.dilation,
                                  self.padding, self.output_padding)


def ConvTranspose1d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0,
                    dilation=1, groups=1, bias=True):
    return ConvTranspose2d(in_channels, out_channels, (kernel_size,), stride, padding, output_padding, dilation, groups, bias)


class InstanceNorm:
    def __init__(self, num_features, eps=1e-5, affine=True):
        self.num_features, self.eps = num_features, eps
        self.weight = Tensor.ones(num_features) if affine else None
        self.bias = Tensor.zeros(num_features) if affine else None

    def __call__(self, x):
        core = _ffi._lib.poly_tensor_instancenorm_apply(x._ctx, x._tensor,
            self.weight._tensor if self.weight is not None else None,
            self.bias._tensor if self.bias is not None else None, self.num_features, self.eps)
        return x._make_result_from_core(core, x.shape)


class LSTMCell:
    def __init__(self, input_size, hidden_size, bias=True):
        bound = 1 / math.sqrt(hidden_size)
        self.weight_ih = Tensor.uniform(hidden_size * 4, input_size, low=-bound, high=bound)
        self.weight_hh = Tensor.uniform(hidden_size * 4, hidden_size, low=-bound, high=bound)
        self.bias_ih = Tensor.zeros(hidden_size * 4) if bias else None
        self.bias_hh = Tensor.zeros(hidden_size * 4) if bias else None

    def __call__(self, x, hc=None):
        h, c = (None, None) if hc is None else hc
        inputs = (x, h, c, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)
        if any(t is not None and t._ctx != x._ctx for t in inputs):
            raise ValueError("LSTMCell inputs must share the same context")
        new_h, new_c = _ffi._ptr(), _ffi._ptr()
        if _ffi._lib.poly_tensor_lstm_cell(x._ctx, *(t._tensor if t is not None else None for t in inputs),
                                          ctypes.byref(new_h), ctypes.byref(new_c)) != 0:
            raise RuntimeError("poly_tensor_lstm_cell failed")
        shape = (x.shape[0], self.weight_hh.shape[1])
        return x._make_result_from_core(new_h, shape), x._make_result_from_core(new_c, shape)


class BatchNorm:
    """Batch normalization."""
    # Pinned tinygrad/nn/__init__.py:35-60.
    def __init__(self, sz, eps=1e-5, affine=True, track_running_stats=True, momentum=0.1):
        self.eps, self.track_running_stats, self.momentum = eps, track_running_stats, momentum
        self.weight = Tensor.ones(sz) if affine else None
        self.bias = Tensor.zeros(sz) if affine else None
        self.num_batches_tracked = Tensor.zeros(dtype='long').is_param_(False)
        if track_running_stats:
            self.running_mean = Tensor.zeros(sz).is_param_(False)
            self.running_var = Tensor.ones(sz).is_param_(False)

    def calc_stats(self, x):
        mean, var = _ffi._ptr(), _ffi._ptr()
        if _ffi._lib.poly_tensor_batchnorm_stats(x._ctx, x._tensor,
                self.running_mean._tensor if self.track_running_stats else None,
                self.running_var._tensor if self.track_running_stats else None,
                bool(TRAINING), ctypes.byref(mean), ctypes.byref(var)) != 0:
            raise RuntimeError('poly_tensor_batchnorm_stats failed')
        return x._make_result_from_core(mean, None), x._make_result_from_core(var, None)

    def __call__(self, x):
        core = _ffi._lib.poly_tensor_batchnorm_apply(x._ctx, x._tensor,
            self.weight._tensor if self.weight is not None else None,
            self.bias._tensor if self.bias is not None else None,
            self.running_mean._tensor if self.track_running_stats else None,
            self.running_var._tensor if self.track_running_stats else None,
            self.num_batches_tracked._tensor, bool(TRAINING), self.eps, self.momentum)
        return x._make_result_from_core(core, x.shape)
