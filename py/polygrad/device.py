"""Device -- tinygrad-compatible device."""

import ctypes
import functools
import numpy as np

from . import _ffi
from .dtype import _to_np_dtype, dtypes


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
        np_dt = _to_np_dtype(self.dtype_name)
        return np.frombuffer(self.as_memoryview(), dtype=np_dt)

    def as_memoryview(self):
        """Copy realized bytes into a fresh memoryview, matching tinygrad."""
        itemsize = np.dtype(_to_np_dtype(self.dtype_name)).itemsize
        nbytes = self.numel * itemsize
        out = bytearray(nbytes)
        if nbytes:
            raw = (ctypes.c_uint8 * nbytes).from_buffer(out)
            rc = _ffi._lib.poly_buffer_read(self.ctx, self.uop.raw, raw, nbytes)
            if rc != 0:
                raise RuntimeError('Buffer.as_memoryview: buffer readback failed')
        return memoryview(out)


class Renderer:
    """Selected backend renderer capability surface."""

    def __init__(self, device):
        self.device = device

    @functools.cached_property
    def _supported_dtypes(self):
        from . import can_run
        return frozenset(
            dtype for dtype in dtypes.all
            if dtype is not dtypes.weakint and
            can_run('add', dtype=dtype, shape=(1,), device=self.device)
        )

    def supported_dtypes(self):
        # Pinned Renderer.supported_dtypes returns a mutable set.
        return set(self._supported_dtypes)


class Compiled:
    """Opened executable device, matching tinygrad's selected-device shape."""

    def __init__(self, device):
        self.device = device
        self.renderer = Renderer(device)

    def __repr__(self):
        return f"<Compiled device:{self.device}>"

    def __str__(self):
        return self.device


class _Device:
    def __init__(self):
        self._default = 'CPU'

    @property
    def DEFAULT(self):
        from .helpers import DEV
        return self.canonicalize(DEV.value) if DEV.value else self._default

    @functools.cache
    def __getitem__(self, key):
        return Compiled(self.canonicalize(key))

    def canonicalize(self, device):
        if device is None:
            return self.DEFAULT
        if isinstance(device, Compiled):
            return device.device
        lib = _ffi.get_lib()
        value = str(device)
        if not value.upper().startswith('DISK:') and value.endswith(':0'):
            value = value[:-2]
        device_id = int(lib.poly_device_by_name(value.encode('utf-8')))
        canonical = lib.poly_device_name(device_id).decode('utf-8').upper()
        if canonical == 'DISK':
            if ':' not in value or not value.split(':', 1)[1]:
                raise ValueError(f'Unsupported device: {device!r}')
            return f"DISK:{value.split(':', 1)[1]}"
        if not lib.poly_device_can_execute(device_id):
            raise ValueError(f'Unsupported device: {device!r}')
        return canonical

    def set_default(self, device):
        from .helpers import DEV
        DEV.value = self.canonicalize(device)

    def cuda_available(self):
        """Check if CUDA is available."""
        return _ffi._has_cuda_ffi and _ffi.get_lib().poly_cuda_available()


Device = _Device()


def _device_id(device=None):
    canonical = Device.canonicalize(device)
    return int(_ffi.get_lib().poly_device_by_name(canonical.lower().encode('utf-8')))
