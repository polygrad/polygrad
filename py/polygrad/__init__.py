"""polygrad -- Python frontend for the polygrad C11 tensor compiler."""

import atexit

from . import _ffi

# Module-level default context (triggers lazy library load)
_default_ctx = _ffi.get_lib().poly_ctx_new()
atexit.register(lambda: _ffi.get_lib().poly_ctx_destroy(_default_ctx))

from .tensor import Tensor, Variable, BoundVariable
from .dtype import dtypes
from .device import Device

__all__ = ['Tensor', 'Variable', 'BoundVariable', 'dtypes', 'Device']
__version__ = '0.2.0'
