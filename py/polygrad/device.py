"""Device -- tinygrad-compatible device."""

import ctypes
import numpy as np

from . import _ffi
from .dtype import _to_np_dtype


class Buffer:
    """Lightweight Python wrapper around a realized PolyBuffer. Mirrors
    tinygrad's Buffer surface at the minimum needed for Tensor.numpy():
    exposes a .numpy() method that copies the buffer bytes into a fresh
    NumPy array of the requested dtype and size."""

    __slots__ = ('ctx', 'uop', 'dtype_name', 'numel')

    def __init__(self, ctx, uop, dtype_name, numel):
        # ctx: PolyCtx*, uop: UOp (terminal BUFFER/PARAM),
        # dtype_name: polygrad dtype string, numel: int element count.
        self.ctx = ctx
        self.uop = uop
        self.dtype_name = dtype_name
        self.numel = numel

    def numpy(self):
        """Read the buffer bytes as a flat numpy array (copy)."""
        ptr = _ffi._lib.poly_buffer_get_ptr(self.ctx, self.uop.raw)
        if not ptr:
            raise RuntimeError('Buffer.numpy: buffer not allocated in ctx->buffers')
        np_dt = _to_np_dtype(self.dtype_name)
        itemsize = np.dtype(np_dt).itemsize
        nbytes = self.numel * itemsize
        # Copy C-owned bytes into a fresh NumPy array. ctypes.string_at copies
        # once; np.frombuffer then wraps without extra copy.
        raw = ctypes.string_at(ptr, nbytes)
        return np.frombuffer(raw, dtype=np_dt)


class Device:
    DEFAULT = 'CPU'

    def __class_getitem__(cls, key):
        return cls.canonicalize(key)

    @staticmethod
    def canonicalize(device):
        if device is None:
            return Device.DEFAULT
        dev = str(device).upper()
        if dev not in {'CPU', 'CUDA'}:
            raise ValueError(f'Unsupported device: {device!r}')
        return dev

    @staticmethod
    def set_default(device):
        Device.DEFAULT = Device.canonicalize(device)

    @staticmethod
    def cuda_available():
        """Check if CUDA is available."""
        return _ffi._has_cuda_ffi and _ffi.get_lib().poly_cuda_available()
