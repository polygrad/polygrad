"""polygrad -- Python frontend for the polygrad C11 tensor compiler."""

import atexit
import ctypes
from collections import defaultdict
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
from . import nn as nn

TinyJit = Jit


class _GlobalCountersMeta(type):
    @property
    def global_ops(cls):
        return _stats_for_ctx(cls._ctx)['global_ops']

    @property
    def global_mem(cls):
        return _stats_for_ctx(cls._ctx)['global_mem']

    @property
    def time_sum_s(cls):
        return _stats_for_ctx(cls._ctx)['time_sum_s']

    @property
    def kernel_count(cls):
        return _stats_for_ctx(cls._ctx)['kernel_count']

    @property
    def mem_used(cls):
        return _stats_for_ctx(cls._ctx)['mem_used']

    @property
    def mem_used_per_device(cls):
        out = defaultdict(int)
        lib = _ffi.get_lib()
        for device_id in range(1, 9):
            used = int(lib.poly_ctx_mem_used_for_device(cls._ctx, device_id))
            if used:
                name = lib.poly_device_name(device_id).decode('utf-8').upper()
                out[name] = used
        return out


def _global_counters_class(ctx):
    class BoundGlobalCounters(metaclass=_GlobalCountersMeta):
        _ctx = ctx

        @classmethod
        def reset(cls):
            _ffi.get_lib().poly_ctx_reset_counters(cls._ctx)

    BoundGlobalCounters.__name__ = 'GlobalCounters'
    BoundGlobalCounters.__qualname__ = 'GlobalCounters'
    return BoundGlobalCounters


GlobalCounters = _global_counters_class(_default_ctx)

# tinygrad exposes its canonical counter class from helpers. Keep one counter
# owner and bind that exact object after the default PolyCtx exists.
from . import helpers as _helpers
_helpers.GlobalCounters = GlobalCounters


def _stats_for_ctx(ctx):
    s = _ffi.PolyCtxStats()
    rc = _ffi.get_lib().poly_ctx_stats(ctx, s)
    if rc != 0:
        raise RuntimeError('poly_ctx_stats failed')
    return {name: getattr(s, name) for name, _ in s._fields_}


def stats():
    """Return monotonically accumulated counters for the module default context."""
    return _stats_for_ctx(_default_ctx)


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
    return _can_run_ctx(_default_ctx, op, dtype=dtype, shape=shape, shapes=shapes, device=device)


def _can_run_ctx(ctx, op=None, *, dtype='float32', shape=None, shapes=None, device='auto'):
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


def _bound_tensor_class(ctx):
    class RuntimeTensor(Tensor):
        def __init__(self, data=None, *args, **kwargs):
            kwargs.setdefault('_ctx', ctx)
            super().__init__(data, *args, **kwargs)

        @staticmethod
        def zeros(*shape, **kwargs):
            kwargs.setdefault('_ctx', ctx)
            return Tensor.zeros(*shape, **kwargs)

        @staticmethod
        def ones(*shape, **kwargs):
            kwargs.setdefault('_ctx', ctx)
            return Tensor.ones(*shape, **kwargs)

        @staticmethod
        def full(shape, fill_value, **kwargs):
            kwargs.setdefault('_ctx', ctx)
            return Tensor.full(shape, fill_value, **kwargs)

        @staticmethod
        def arange(start, stop=None, step=1, **kwargs):
            kwargs.setdefault('_ctx', ctx)
            return Tensor.arange(start, stop, step, **kwargs)

        @staticmethod
        def rand(*shape, **kwargs):
            kwargs.setdefault('_ctx', ctx)
            return Tensor.rand(*shape, **kwargs)

        @staticmethod
        def randn(*shape, **kwargs):
            kwargs.setdefault('_ctx', ctx)
            return Tensor.randn(*shape, **kwargs)

        @staticmethod
        def kaiming_uniform(*shape, **kwargs):
            kwargs.setdefault('_ctx', ctx)
            return Tensor.kaiming_uniform(*shape, **kwargs)

        @staticmethod
        def randint(*shape, **kwargs):
            kwargs.setdefault('_ctx', ctx)
            return Tensor.randint(*shape, **kwargs)

        @staticmethod
        def randperm(n, **kwargs):
            kwargs.setdefault('_ctx', ctx)
            return Tensor.randperm(n, **kwargs)

        @staticmethod
        def linspace(start, stop, steps, **kwargs):
            kwargs.setdefault('_ctx', ctx)
            return Tensor.linspace(start, stop, steps, **kwargs)

        @staticmethod
        def eye(n, m=None, **kwargs):
            kwargs.setdefault('_ctx', ctx)
            return Tensor.eye(n, m, **kwargs)

        @staticmethod
        def empty(*shape, **kwargs):
            kwargs.setdefault('_ctx', ctx)
            return Tensor.empty(*shape, **kwargs)

    RuntimeTensor.__name__ = 'Tensor'
    RuntimeTensor.__qualname__ = 'Tensor'
    return RuntimeTensor


class Runtime:
    """Explicit PolyCtx owner for device/context-scoped Python code."""

    def __init__(self, *, device='auto'):
      lib = _ffi.get_lib()
      self._ctx = lib.poly_ctx_new()
      self._disposed = False
      dev_id = lib.poly_device_by_name(str(device).lower().encode('utf-8'))
      if dev_id >= 0 and hasattr(lib, 'poly_ctx_set_preferred_device'):
          lib.poly_ctx_set_preferred_device(self._ctx, dev_id)
      self.Tensor = _bound_tensor_class(self._ctx)
      self.Variable = lambda name, min_val, max_val: Variable(name, min_val, max_val, _ctx=self._ctx)
      self.Instance = Instance
      self.GlobalCounters = _global_counters_class(self._ctx)
      self.jit = jit
      self.compile = compile

    def stats(self):
      self._check_live()
      return _stats_for_ctx(self._ctx)

    def can_run(self, op=None, *, dtype='float32', shape=None, shapes=None, device='auto'):
      self._check_live()
      return _can_run_ctx(self._ctx, op, dtype=dtype, shape=shape, shapes=shapes, device=device)

    def dispose(self):
      if not self._disposed:
          _ffi.get_lib().poly_ctx_destroy(self._ctx)
          self._disposed = True
          self._ctx = None

    def _check_live(self):
      if self._disposed:
          raise RuntimeError('polygrad runtime has been disposed')

    def __enter__(self):
      self._check_live()
      return self

    def __exit__(self, exc_type, exc, tb):
      self.dispose()
      return False


def create(*, device='auto'):
    """Create an explicit Polygrad runtime/context."""
    return Runtime(device=device)


__all__ = [
    'Tensor', 'Variable', 'BoundVariable', 'dtypes', 'Device', 'Instance', 'nn',
    'GlobalCounters',
    'CompiledCallable', 'Jit', 'TinyJit', 'JitError', 'Runtime', 'create',
    'compile', 'jit', 'stats', 'can_run',
]

try:
    __version__ = _pkg_version('polygrad')
except PackageNotFoundError:
    __version__ = '0+unknown'
