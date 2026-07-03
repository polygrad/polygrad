"""polygrad -- Python frontend for the polygrad C11 tensor compiler."""

import atexit
import ctypes
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


def _can_run_dtype(dtype):
    if dtype is None:
        return 'float32'
    name = str(dtype).lower()
    if name == 'half':
        return 'float16'
    if name == 'double':
        return 'float64'
    return name


def _can_run_op(op):
    s = str(op)
    out = []
    for ch in s:
        if ch.isupper():
            out.extend(['_', ch.lower()])
        elif ch == '-':
            out.append('_')
        else:
            out.append(ch)
    return ''.join(out).lower()


def _shape_array(shape, label):
    if shape is None:
        raise ValueError(f'can_run {label} is required')
    vals = [int(x) for x in shape]
    if any(x < 0 for x in vals):
        raise ValueError(f'can_run {label} contains a negative dimension')
    return vals


def _can_run_shape(op, shape, shapes):
    if shapes is not None:
        ss = [_shape_array(s, f'shapes[{i}]') for i, s in enumerate(shapes)]
        if not ss:
            raise ValueError('can_run shapes must be non-empty')
        if op in ('matmul', 'dot'):
            if len(ss) < 2 or len(ss[0]) < 2 or len(ss[1]) < 2:
                raise ValueError('can_run matmul shapes must be [[m,k],[k,n]]')
            return [ss[0][-2], ss[0][-1], ss[1][-1]]
        if op in ('triangular_solve', 'solve', 'lstsq'):
            if len(ss) < 2 or len(ss[0]) < 2 or len(ss[1]) < 1:
                raise ValueError(f'can_run {op} shapes must be [matrix_shape, rhs_shape]')
            if len(ss[1]) >= 2:
                return [ss[0][-2], ss[0][-1], ss[1][-1]]
            return [ss[0][-2], ss[0][-1]]
        return ss[0]
    return _shape_array(shape, 'shape')


def can_run(op=None, *, dtype='float32', shape=None, shapes=None, device='auto'):
    """Return whether the default context can lower a representative op/shape."""
    if op is None and (shape is not None or shapes is not None):
        raise ValueError('can_run shape queries require an op')
    lib = _ffi.get_lib()
    dtype_name = _can_run_dtype(dtype)
    dtype_id = lib.poly_dtype_id_by_name(dtype_name.encode('utf-8'))
    if dtype_id < 0:
        return False
    dev_id = lib.poly_device_by_name(str(device).encode('utf-8'))
    op_name = _can_run_op(op) if op is not None else 'add'
    dims = _can_run_shape(op_name, shape, shapes) if op is not None else [1]
    arr_t = ctypes.c_int64 * len(dims)
    arr = arr_t(*dims) if dims else None
    rc = lib.poly_can_run_op(_default_ctx, dev_id, op_name.encode('utf-8'), dtype_id, arr, len(dims))
    if rc < 0:
        raise RuntimeError('can_run cannot prove this op/shape query')
    return rc == 1


__all__ = [
    'Tensor', 'Variable', 'BoundVariable', 'dtypes', 'Device', 'Instance',
    'CompiledCallable', 'Jit', 'JitError', 'compile', 'jit', 'stats', 'can_run',
]

try:
    __version__ = _pkg_version('polygrad')
except PackageNotFoundError:
    __version__ = '0+unknown'
