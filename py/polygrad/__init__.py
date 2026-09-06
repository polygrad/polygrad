"""polygrad -- Python frontend for the polygrad C11 tensor compiler."""

import atexit
import ctypes
from collections import defaultdict
from importlib.metadata import PackageNotFoundError, version as _pkg_version

from . import _ffi

# Module-level default context (triggers lazy library load)
_default_ctx = _ffi.get_lib().poly_ctx_new()
if not _default_ctx:
    raise RuntimeError('invalid POLY_LOGICAL: expected 0, 1, or 2')

from .tensor import Tensor, Variable, BoundVariable, _dispose_tensors_for_ctx
from .dtype import DType, INVERSE_DTYPES_DICT, dtypes
from .device import Device
from .model import Model, _dispose_models_for_ctx
from .jit import CompiledCallable, Jit, JitError, TinyJit, _dispose_jits_for_ctx, compile, jit
from .function import function
from .uop.ops import UOp, _dispose_uops_for_ctx
from .helpers import Context, LOGICAL, _normalize_logical_policy, fetch, getenv
from . import nn as nn

def _dispose_default_ctx():
    global _default_ctx
    if _default_ctx:
        _dispose_models_for_ctx(_default_ctx)
        _dispose_jits_for_ctx(_default_ctx)
        _dispose_tensors_for_ctx(_default_ctx)
        _dispose_uops_for_ctx(_default_ctx)
        _ffi.get_lib().poly_ctx_destroy(_default_ctx)
        _default_ctx = None


atexit.register(_dispose_default_ctx)


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


def collect():
    """Collect C graph storage retired by frontend finalizers."""
    if _ffi.get_lib().poly_ctx_collect(_default_ctx) != 0:
        raise RuntimeError('poly_ctx_collect failed')


def _can_run_dtype(dtype):
    if dtype is None:
        return 'float32'
    if isinstance(dtype, DType):
        name = INVERSE_DTYPES_DICT.get(dtype.name, dtype.name)
    else:
        name = str(dtype).lower()
    return {
        'half': 'float16', 'float': 'float32', 'double': 'float64',
        'char': 'int8', 'uchar': 'uint8', 'short': 'int16', 'ushort': 'uint16',
        'int': 'int32', 'uint': 'uint32', 'long': 'int64', 'ulong': 'uint64',
    }.get(name, name)


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
    rc = lib.poly_can_run_op(ctx, dev_id, op_name.encode('utf-8'), dtype_id, arr, len(dims))
    if rc < 0:
        raise RuntimeError('can_run cannot prove this op/shape query')
    return rc == 1


def _bound_tensor_class(ctx, runtime):
    def live_ctx():
        runtime._check_live()
        return ctx

    class RuntimeTensor(Tensor):
        def __init__(self, data=None, *args, **kwargs):
            kwargs.setdefault('_ctx', live_ctx())
            super().__init__(data, *args, **kwargs)

        @staticmethod
        def from_url(url, gunzip=False, **kwargs):
            kwargs.setdefault('_ctx', live_ctx())
            return Tensor.from_url(url, gunzip=gunzip, **kwargs)

        @staticmethod
        def zeros(*shape, **kwargs):
            kwargs.setdefault('_ctx', live_ctx())
            return Tensor.zeros(*shape, **kwargs)

        @staticmethod
        def ones(*shape, **kwargs):
            kwargs.setdefault('_ctx', live_ctx())
            return Tensor.ones(*shape, **kwargs)

        @staticmethod
        def full(shape, fill_value, **kwargs):
            kwargs.setdefault('_ctx', live_ctx())
            return Tensor.full(shape, fill_value, **kwargs)

        @staticmethod
        def arange(start, stop=None, step=1, **kwargs):
            kwargs.setdefault('_ctx', live_ctx())
            return Tensor.arange(start, stop, step, **kwargs)

        @staticmethod
        def rand(*shape, **kwargs):
            kwargs.setdefault('_ctx', live_ctx())
            return Tensor.rand(*shape, **kwargs)

        @staticmethod
        def randn(*shape, **kwargs):
            kwargs.setdefault('_ctx', live_ctx())
            return Tensor.randn(*shape, **kwargs)

        @staticmethod
        def kaiming_uniform(*shape, **kwargs):
            kwargs.setdefault('_ctx', live_ctx())
            return Tensor.kaiming_uniform(*shape, **kwargs)

        @staticmethod
        def randint(*shape, **kwargs):
            kwargs.setdefault('_ctx', live_ctx())
            return Tensor.randint(*shape, **kwargs)

        @staticmethod
        def randperm(n, **kwargs):
            kwargs.setdefault('_ctx', live_ctx())
            return Tensor.randperm(n, **kwargs)

        @staticmethod
        def linspace(start, stop, steps, **kwargs):
            kwargs.setdefault('_ctx', live_ctx())
            return Tensor.linspace(start, stop, steps, **kwargs)

        @staticmethod
        def eye(n, m=None, **kwargs):
            kwargs.setdefault('_ctx', live_ctx())
            return Tensor.eye(n, m, **kwargs)

        @staticmethod
        def empty(*shape, **kwargs):
            kwargs.setdefault('_ctx', live_ctx())
            return Tensor.empty(*shape, **kwargs)

        @staticmethod
        def manual_seed(seed=0):
            _ffi.get_lib().poly_tensor_manual_seed(live_ctx(), int(seed))

    RuntimeTensor.__name__ = 'Tensor'
    RuntimeTensor.__qualname__ = 'Tensor'
    return RuntimeTensor


class Runtime:
    """Explicit PolyCtx owner for device/context-scoped Python code."""

    def __init__(self, *, device='auto', logical=None):
      lib = _ffi.get_lib()
      self._ctx = lib.poly_ctx_new()
      if not self._ctx:
          raise RuntimeError('poly_ctx_new failed; check POLY_LOGICAL')
      self._disposed = False
      policy = _normalize_logical_policy(logical)
      if policy is not None and lib.poly_ctx_set_logical_policy(self._ctx, policy) != 0:
          lib.poly_ctx_destroy(self._ctx)
          self._ctx = None
          self._disposed = True
          raise ValueError(f'invalid logical policy {logical!r}')
      dev_id = lib.poly_device_by_name(str(device).lower().encode('utf-8'))
      if dev_id >= 0 and hasattr(lib, 'poly_ctx_set_preferred_device'):
          lib.poly_ctx_set_preferred_device(self._ctx, dev_id)
      self.Tensor = _bound_tensor_class(self._ctx, self)
      def runtime_variable(name, min_val, max_val):
          self._check_live()
          return Variable(name, min_val, max_val, _ctx=self._ctx)
      self.Variable = runtime_variable
      self.Model = Model
      self.GlobalCounters = _global_counters_class(self._ctx)
      self.jit = jit
      self.compile = compile

    def context(self, **kwargs):
      if set(kwargs) != {'LOGICAL'}:
          raise KeyError(next(iter(set(kwargs) - {'LOGICAL'}), 'LOGICAL'))
      runtime = self
      policy = _normalize_logical_policy(kwargs['LOGICAL'])

      class RuntimeContext:
          def __enter__(self):
              runtime._check_live()
              lib = _ffi.get_lib()
              self.old_policy = int(lib.poly_ctx_get_logical_policy(runtime._ctx))
              if policy is not None and lib.poly_ctx_set_logical_policy(runtime._ctx, policy) != 0:
                  raise ValueError(f"invalid logical policy {kwargs['LOGICAL']!r}")

          def __exit__(self, *_):
              if _ffi.get_lib().poly_ctx_set_logical_policy(runtime._ctx, self.old_policy) != 0:
                  raise RuntimeError('failed to restore logical policy')

      return RuntimeContext()

    def logical(self, mode):
      return self.context(LOGICAL=mode)

    def stats(self):
      self._check_live()
      return _stats_for_ctx(self._ctx)

    def collect(self):
      self._check_live()
      if _ffi.get_lib().poly_ctx_collect(self._ctx) != 0:
          raise RuntimeError('poly_ctx_collect failed')

    def can_run(self, op=None, *, dtype='float32', shape=None, shapes=None, device='auto'):
      self._check_live()
      return _can_run_ctx(self._ctx, op, dtype=dtype, shape=shape, shapes=shapes, device=device)

    def dispose(self):
      if not self._disposed:
          _dispose_models_for_ctx(self._ctx)
          _dispose_jits_for_ctx(self._ctx)
          _dispose_tensors_for_ctx(self._ctx)
          _dispose_uops_for_ctx(self._ctx)
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


def create(*, device='auto', logical=None):
    """Create an explicit Polygrad runtime/context."""
    return Runtime(device=device, logical=logical)


__all__ = [
    'Tensor', 'Variable', 'BoundVariable', 'UOp', 'dtypes', 'Device', 'Model', 'nn',
    'GlobalCounters', 'Context', 'LOGICAL', 'fetch', 'getenv', 'function',
    'CompiledCallable', 'Jit', 'TinyJit', 'JitError', 'Runtime', 'create',
    'compile', 'jit', 'stats', 'collect', 'can_run',
]

try:
    __version__ = _pkg_version('polygrad')
except PackageNotFoundError:
    __version__ = '0+unknown'
