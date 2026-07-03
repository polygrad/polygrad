"""polygrad -- Python frontend for the polygrad C11 tensor compiler."""

import atexit
from importlib.metadata import PackageNotFoundError, version as _pkg_version

from . import _ffi

# Module-level default context (triggers lazy library load)
_default_ctx = _ffi.get_lib().poly_ctx_new()
atexit.register(lambda: _ffi.get_lib().poly_ctx_destroy(_default_ctx))

from .tensor import Tensor, Variable, BoundVariable
from .dtype import dtypes
from .device import Device
from .instance import Instance
from .jit import CompiledCallable, Jit, JitError, compile, jit


def stats():
    """Return monotonically accumulated counters for the module default context."""
    s = _ffi.PolyCtxStats()
    rc = _ffi.get_lib().poly_ctx_stats(_default_ctx, s)
    if rc != 0:
        raise RuntimeError('poly_ctx_stats failed')
    return {name: getattr(s, name) for name, _ in s._fields_}


__all__ = [
    'Tensor', 'Variable', 'BoundVariable', 'dtypes', 'Device', 'Instance',
    'CompiledCallable', 'Jit', 'JitError', 'compile', 'jit', 'stats',
]

try:
    __version__ = _pkg_version('polygrad')
except PackageNotFoundError:
    __version__ = '0+unknown'
