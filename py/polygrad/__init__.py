"""polygrad — Python frontend for the polygrad C11 tensor compiler."""

import atexit

from . import _ffi

# Module-level default context
_default_ctx = _ffi._lib.poly_ctx_new()
atexit.register(lambda: _ffi._lib.poly_ctx_destroy(_default_ctx))

from .tensor import Tensor, Variable, BoundVariable
from .dtype import dtypes
from .device import Device

__all__ = ['Tensor', 'Variable', 'BoundVariable', 'dtypes', 'Device']
__version__ = '0.0.1'
