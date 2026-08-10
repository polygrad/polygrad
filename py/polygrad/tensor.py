"""
Tensor class for polygrad, lazy evaluation backed by the C compiler core.
Supports float32 (default) and float64 dtypes.
"""

import ctypes
import functools
import hashlib
import math
import pathlib
import sys
import weakref

import numpy as np

from . import _ffi
from .dtype import INVERSE_DTYPES_DICT, _from_np_dtype, _to_np_dtype, dtypes, least_upper_dtype, least_upper_float, to_dtype
from polygrad.uop.ops import UOp
from polygrad.device import Buffer


# Global registry of live Tensors. Keys are weakrefs so GC'd tensors vanish
# automatically. Used for post-realize retargeting and backward graph
# discovery, matching tinygrad's live-tensor registry.
all_tensors: dict[weakref.ref, None] = {}
_custom_kernel_grad_records = []

# Strong frontend owner registry keyed by the C-side PolyBuffer* address value.
# This is intentionally keyed by the retired/imported residency object, not by
# BUFFER UOp identity. The core explicitly calls frontend_buffer_release(buffer)
# when a HOST PolyBuffer is retired, and only then do we drop the owner entry.
_host_buffers: dict[int, np.ndarray] = {}
_frontend_buffer_release_cb = None

_POLY_TENSOR_VALUE = 0


def frontend_buffer_release(buffer_key):
    """Drop the strong owner for a retired HOST PolyBuffer."""
    _host_buffers.pop(int(buffer_key), None)


def _ensure_frontend_buffer_release_registered(ctx=None):
    global _frontend_buffer_release_cb
    if _frontend_buffer_release_cb is None:
        _frontend_buffer_release_cb = _ffi.PolyFrontendBufferReleaseFn(frontend_buffer_release)
    if ctx is not None and hasattr(_ffi._lib, 'poly_ctx_set_frontend_buffer_release'):
        _ffi._lib.poly_ctx_set_frontend_buffer_release(ctx, _frontend_buffer_release_cb)
    else:
        _ffi._lib.poly_set_frontend_buffer_release(_frontend_buffer_release_cb)


def _buffer_key(ctx, uop):
    raw = uop.raw if isinstance(uop, UOp) else uop
    return int(_ffi._lib.poly_buffer_get_key(ctx, raw))


def _uop_raw(uop):
    return uop.raw if isinstance(uop, UOp) else uop


def _uop_wrap(ctx, raw):
    return UOp(ctx, raw) if raw else None


def _ptr_value(ptr):
    raw = _uop_raw(ptr)
    if isinstance(raw, ctypes.c_void_p):
        return 0 if raw.value is None else int(raw.value)
    return int(raw)


def _device_name_from_id(device_id):
    raw = _ffi._lib.poly_device_name(int(device_id))
    if not raw:
        return 'CPU'
    return raw.decode('utf-8').upper()


def _device_id(device):
    from .device import Device

    dev = Device.canonicalize(device).lower().encode('utf-8')
    return int(_ffi._lib.poly_device_by_name(dev))


def _int64_array(vals):
    """Convert a Python sequence to a ctypes int64 array."""
    n = len(vals)
    arr = (ctypes.c_int64 * n)(*vals)
    return arr, n


def _pair_array(pairs):
    """Convert a sequence of (a, b) pairs to a contiguous int64 array."""
    n = len(pairs)
    flat = (ctypes.c_int64 * (n * 2))()
    for i, (a, b) in enumerate(pairs):
        flat[i * 2] = a
        flat[i * 2 + 1] = b
    return flat, n


def _shape_from_uop(ctx, uop):
    """Read tinygrad-style shape tuple from a UOp.

    Static dimensions are returned as ints. Symbolic dimensions are returned as
    UOp wrappers around DEFINE_VAR/BIND expressions, matching tinygrad's
    `uop.shape`. `poly_uop_max_shape_dims` exposes max_shape storage.
    """
    ndim = _ffi._lib.poly_uop_ndim(ctx, uop)
    if ndim <= 0:
        return ()
    dims = _ffi._lib.poly_uop_max_shape_dims(ctx, uop)
    out = []
    for i in range(ndim):
        dim_raw = _ffi._lib.poly_uop_shape_dim(ctx, uop, i)
        if dim_raw:
            value = ctypes.c_int64()
            if _ffi._lib.poly_uop_const_i64(dim_raw, ctypes.byref(value)) == 0:
                out.append(int(value.value))
            else:
                out.append(UOp(ctx, dim_raw))
        else:
            out.append(dims[i])
    return tuple(out)


def _slice_indices_size(ctx, uop, dim, size):
    if isinstance(size, UOp):
        dims = _ffi._lib.poly_uop_max_shape_dims(ctx, uop)
        return int(dims[dim])
    return int(size)


I64_MIN = -(1 << 63)
I64_MAX = (1 << 63) - 1
_KNOWN_DTYPE_NAMES = (
    'bool', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32',
    'int64', 'uint64', 'float16', 'bfloat16', 'float32', 'float64',
)
_DTYPE_ID_CACHE = {}
_DTYPE_NAME_BY_ID = {}
_U64_MASK = (1 << 64) - 1


def _dtype_name(dtype, default='float32'):
    # Normalize through the shared dtype helpers first so Python and the C
    # frontend talk about the same scalar names before ids enter the picture.
    target = default if dtype is None else dtype
    if isinstance(target, str):
        dt = to_dtype(target)
    else:
        try:
            dt = to_dtype(target)
        except AttributeError:
            dt = _from_np_dtype(np.dtype(target))
    sdt = dt.scalar()
    if sdt == dtypes.bool:
        return 'bool'
    if sdt == dtypes.int8:
        return 'int8'
    if sdt == dtypes.uint8:
        return 'uint8'
    if sdt == dtypes.int16:
        return 'int16'
    if sdt == dtypes.uint16:
        return 'uint16'
    if sdt == dtypes.int32:
        return 'int32'
    if sdt == dtypes.uint32:
        return 'uint32'
    if sdt == dtypes.int64:
        return 'int64'
    if sdt == dtypes.uint64:
        return 'uint64'
    if sdt == dtypes.float16:
        return 'float16'
    if sdt == dtypes.bfloat16:
        return 'bfloat16'
    if sdt == dtypes.float32:
        return 'float32'
    if sdt == dtypes.float64:
        return 'float64'
    return INVERSE_DTYPES_DICT.get(sdt.name, sdt.name)


def _dtype_id(dtype):
    # JS already asks the core for dtype ids by name; Python needs the same
    # rule so frontend wrappers do not hard-code a second dtype-id table.
    name = _dtype_name(dtype, default='float32')
    cached = _DTYPE_ID_CACHE.get(name)
    if cached is not None:
        return cached
    dtype_id = _ffi._lib.poly_dtype_id_by_name(name.encode('utf-8'))
    if dtype_id < 0:
        raise ValueError(f'unsupported dtype: {name}')
    _DTYPE_ID_CACHE[name] = dtype_id
    _DTYPE_NAME_BY_ID[dtype_id] = name
    return dtype_id


def _dtype_name_from_id(dtype_id, default='float32'):
    if dtype_id in _DTYPE_NAME_BY_ID:
        return _DTYPE_NAME_BY_ID[dtype_id]
    for name in _KNOWN_DTYPE_NAMES:
        if _dtype_id(name) == dtype_id:
            return name
    return default


def _shape_tuple(*shape):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return tuple(shape)


def _normalize_expand_shape(current_shape, requested_shape):
    ndim = max(len(current_shape), len(requested_shape))
    cur = (1,) * (ndim - len(current_shape)) + tuple(current_shape)
    req = (1,) * (ndim - len(requested_shape)) + tuple(requested_shape)
    return tuple(c if r == -1 or r is None else r for c, r in zip(cur, req))


def _make_tuple(value, count):
    if isinstance(value, int):
        return (int(value),) * count
    return tuple(int(x) for x in value)


def _resolve_pool_pads(padding, dims):
    if isinstance(padding, int):
        return [int(padding)] * (2 * dims)
    padding = tuple(int(x) for x in padding)
    if len(padding) == 2 * dims:
        return list(padding)
    if len(padding) == dims:
        return [p for p in padding for _ in range(2)][::-1]
    raise ValueError(
        f"Padding must be an int or a sequence of length {dims} or {2 * dims}, "
        f"but got padding={padding!r}"
    )


def _normalize_pad_arg(padding, ndim):
    padding = tuple(padding)
    if not any(isinstance(p, (tuple, list, type(None))) for p in padding):
        if len(padding) % 2 != 0:
            raise ValueError("Flat padding must have even number of pads")
        grouped = tuple(zip(padding[-2::-2], padding[::-2]))
        padding = ((0, 0),) * (ndim - len(grouped)) + grouped
    else:
        padding = tuple((0, 0) if p is None else tuple(p) for p in padding)
    if len(padding) != ndim:
        raise ValueError(f"padding length is improper, padding={padding!r} ndim={ndim}")
    return padding


def _flat_to_grouped(padding):
    return tuple(zip(padding[-2::-2], padding[::-2]))


def _prod(vals):
    out = 1
    for v in vals:
        out *= int(v)
    return out


def _dtype_min_value(dtype_name):
    dt = to_dtype(dtype_name).scalar()
    if dtypes.is_float(dt):
        return -math.inf
    if dt == dtypes.bool:
        return 0.0
    try:
        return float(np.iinfo(_to_np_dtype(dt)).min)
    except Exception:
        return -math.inf


def _sum_acc_dtype(dtype):
    """Pinned tinygrad dtype.py:274-278 default sum accumulation dtype."""
    dt = to_dtype(dtype).scalar()
    floor = dtypes.uint32 if dtypes.is_unsigned(dt) else (
        dtypes.int32 if dtypes.is_int(dt) or dtypes.is_bool(dt) else dtypes.float32
    )
    return least_upper_dtype(dt, floor)


def _uop_dtype_name(ctx, uop, default='float32'):
    # Ask the realized UOp for its dtype because Python-side float heuristics
    # were hiding valid int and bool results from the frontend.
    raw = uop.raw if isinstance(uop, UOp) else uop
    dtype_id = _ffi._lib.poly_uop_dtype_id(ctx, raw) if raw else -1
    return _dtype_name_from_id(dtype_id, default)


def _creation_meta(kwargs):
    # Static constructors do not have input tensors to inherit ctx/device from,
    # so centralize the frontend defaults instead of re-encoding them per op.
    from . import _default_ctx
    from .device import Device
    ctx = kwargs.get('_ctx') or _default_ctx
    dev = Device.canonicalize(kwargs.get('_device') if '_device' in kwargs else kwargs.get('device'))
    requires_grad = kwargs.get('requires_grad', False)
    return ctx, dev, requires_grad


def _shape_arg(shape):
    # Reuse one normalization path so every constructor feeds the C helpers the
    # same shape encoding and errors on bad dimensions the same way.
    if any(_is_symbolic_dim(s) for s in shape):
        raise TypeError('this constructor does not yet accept symbolic dimensions')
    shape = tuple(int(s) for s in shape)
    if not shape:
        return None, 0, shape
    dims, ndim = _int64_array(shape)
    return dims, ndim, shape


def _is_symbolic_dim(value):
    return isinstance(value, (Variable, BoundVariable, UOp))


def _shape_all_int(shape):
    return all(isinstance(s, int) for s in shape)


def _shape_has_symbolic(shape):
    return any(_is_symbolic_dim(s) for s in shape)


def _symbolic_dim_raw(value):
    if isinstance(value, BoundVariable):
        return value.uop.raw
    if isinstance(value, Variable):
        return value.uop.raw
    if isinstance(value, UOp):
        return value.raw
    return None


def _shape_dim_uop_raw(ctx, value):
    raw = _symbolic_dim_raw(value)
    if raw is not None:
        return raw
    return UOp.const(ctx, int(value)).raw


def _shape_uop_array(ctx, shape):
    return (_ffi._ptr * len(shape))(*(_shape_dim_uop_raw(ctx, s) for s in shape))


def _is_symbolic_bound(value):
    return isinstance(value, (BoundVariable, UOp))


def _bound_to_uop(ctx, value):
    if isinstance(value, BoundVariable):
        return value.uop
    if isinstance(value, UOp):
        return value
    if isinstance(value, int):
        return UOp.const(ctx, int(value))
    raise TypeError(f'unsupported symbolic slice bound {type(value).__name__}')


def _bound_to_int(ctx, value):
    if isinstance(value, BoundVariable):
        return int(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, UOp):
        out = ctypes.c_int64()
        if _ffi._lib.poly_uop_bind_value(value.raw, ctypes.byref(out)) == 0:
            return int(out.value)
    raise TypeError(f'symbolic slice bound {value!r} is not bound to an integer')


def _uop_const_i64(value):
    if not isinstance(value, UOp):
        return None
    out = ctypes.c_int64()
    if _ffi._lib.poly_uop_const_i64(value.raw, ctypes.byref(out)) == 0:
        return int(out.value)
    return None


def _uop_add_const_delta(value, base):
    if not isinstance(value, UOp) or not isinstance(base, UOp):
        return None
    if value.op != _ffi.OPS.get('ADD') or len(value.src) != 2:
        return None
    lhs, rhs = value.src
    if lhs == base:
        return _uop_const_i64(rhs)
    if rhs == base:
        return _uop_const_i64(lhs)
    return None


def _symbolic_slice_size_uop(ctx, start_obj, stop_obj, start_u, stop_u):
    if isinstance(start_obj, int) and start_obj == 0 and isinstance(stop_obj, (BoundVariable, UOp)):
        return stop_u
    delta = _uop_add_const_delta(stop_u, start_u)
    if delta is not None:
        return UOp.const(ctx, delta)
    return stop_u - start_u


def _py_scalar(value):
    # Collapse NumPy scalar wrappers early so constructor validation matches the
    # behavior of plain Python literals and tinygrad-style call sites.
    return value.item() if isinstance(value, np.generic) else value


def _require_i64(value, what):
    # Typed constructor helpers accept int64 payloads, so fail before the FFI
    # boundary instead of silently truncating Python integers.
    ivalue = int(value)
    if ivalue < I64_MIN or ivalue > I64_MAX:
        raise ValueError(f'{what} {ivalue} is out of int64 range')
    return ivalue


def _created_tensor(ctx, tensor, dtype_name, device, requires_grad, op_name):
    # Constructors should trust the core for final shape metadata so Python
    # does not grow a second copy of shape or dual-root ownership logic.
    if not tensor:
        raise RuntimeError(f'{op_name} failed')
    uop = _ffi._lib.poly_tensor_uop(tensor)
    if not uop:
        raise RuntimeError(f'{op_name} returned a Tensor without a current UOp')
    return Tensor(
        _ctx=ctx, _tensor=tensor, _shape=_shape_from_uop(ctx, uop),
        requires_grad=requires_grad, _dtype=dtype_name, _device=device,
    )


class Variable:
    """Symbolic integer variable for dynamic tensor dimensions.

    Usage:
        N = Variable("N", 1, 128)
        bound = N.bind(32)          # bind concrete value
        int(bound)                  # => 32
    """
    def __init__(self, name, min_val, max_val, *, _ctx=None):
        from . import _default_ctx
        self._ctx = _ctx or _default_ctx
        self.name = name
        self.min_val = min_val
        self.max_val = max_val
        self.uop = UOp.variable(self._ctx, name, min_val, max_val)

    def bind(self, value):
        """Bind a concrete value, returning a BoundVariable."""
        assert self.min_val <= value <= self.max_val, \
            f"value {value} out of bounds [{self.min_val}, {self.max_val}]"
        return BoundVariable(self, value)

    def __repr__(self):
        return f"Variable({self.name!r}, {self.min_val}, {self.max_val})"


class BoundVariable:
    """A Variable bound to a concrete value."""
    def __init__(self, variable, value):
        self.variable = variable
        self.value = value
        self._ctx = variable._ctx
        self.uop = variable.uop.bind(value)

    def __int__(self):
        return self.value

    def __add__(self, other):
        return self.uop + other

    def __radd__(self, other):
        return other + self.uop

    def __sub__(self, other):
        return self.uop - other

    def __rsub__(self, other):
        return other - self.uop

    def __mul__(self, other):
        return self.uop * other

    def __rmul__(self, other):
        return other * self.uop

    def __repr__(self):
        return f"BoundVariable({self.variable.name!r}, {self.value})"


def _broadcast_shapes(a, b):
    """Compute NumPy-style broadcast shape of two tuples."""
    if not a: return b
    if not b: return a
    ndim = max(len(a), len(b))
    a = (1,) * (ndim - len(a)) + a
    b = (1,) * (ndim - len(b)) + b
    result = []
    for x, y in zip(a, b):
        if x == y: result.append(x)
        elif x == 1: result.append(y)
        elif y == 1: result.append(x)
        else: raise ValueError(f'Cannot broadcast shapes {a} and {b}')
    return tuple(result)


class Tensor:
    """Lazy tensor backed by polygrad's C11 compiler core.

    Operations build a UOp graph. Computation happens only when
    .numpy() or .item() is called.
    """

    training = False  # tinygrad compat
    _seed = 0
    _device_seeds = {}
    _device_rng_counters = {}

    @classmethod
    def train(cls, mode=True):
        """tinygrad-compatible training-mode context manager."""
        class _TrainCtx:
            def __enter__(self_nonlocal):
                self_nonlocal.prev = cls.training
                cls.training = bool(mode)
                return cls

            def __exit__(self_nonlocal, exc_type, exc, tb):
                cls.training = self_nonlocal.prev
                return False

        return _TrainCtx()

    def __init__(self, data=None, requires_grad=False, *, dtype=None, device=None, _ctx=None, _uop=None,
                 _data=None, _shape=None, _dtype=None, _device=None, _tensor=None):
        """Create a tensor from a list, numpy array, or scalar."""
        from . import _default_ctx
        from .device import Device
        if isinstance(data, UOp) and _uop is None:
            _uop = data
            data = None
            if _ctx is None:
                _ctx = _uop.ctx
        self._ctx = _ctx or _default_ctx
        path_data = isinstance(data, pathlib.Path)
        resolved_path = data.resolve() if path_data else None
        disk_device = f'DISK:{resolved_path}' if resolved_path is not None else None
        requested_device = _device if _device is not None else device
        if requested_device is None and disk_device is not None:
            requested_device = disk_device
        self._device = Device.canonicalize(requested_device)
        self._tensor = _tensor
        self._shape_override = tuple(_shape) if _shape is not None else None
        current_uop = None
        imported_tensor_from_host = False
        imported_from_disk = False
        if self._tensor is not None and _uop is None:
            _uop = self._core_uop_raw(self._tensor)

        if _uop is not None:
            # Internal construction (from ops) -- shape is on the UOp.
            # Accept either a UOp instance or a raw ctypes pointer from FFI.
            current_uop = _uop if isinstance(_uop, UOp) else UOp(self._ctx, _uop)
            self._data = _data
            if _dtype is not None:
                self._dtype_str = _dtype
            elif dtype is not None:
                self._dtype_str = _dtype_name(dtype)
            elif current_uop.raw is not None:
                self._dtype_str = _dtype_name_from_id(
                    int(_ffi._lib.poly_uop_dtype_id(self._ctx, current_uop.raw))
                )
            else:
                self._dtype_str = 'float32'
        elif path_data:
            imported_from_disk = True
            self._data = None
            self._dtype_str = _dtype_name(dtype, default='uint8')
            # Match tinygrad's Path constructor boundary: stat determines the
            # logical file-backed buffer size and preserves FileNotFoundError.
            resolved_path.stat()
            raw = _ffi._lib.poly_buffer_from_file(
                self._ctx,
                str(resolved_path).encode('utf-8'),
                _dtype_id(self._dtype_str),
            )
            if not raw:
                raise RuntimeError(f'poly_buffer_from_file failed for {data}')
            current_uop = UOp(self._ctx, raw)
        else:
            # User construction from data
            numpy_scalar = isinstance(data, np.ndarray) and data.shape == ()
            scalar_value = data.item() if numpy_scalar else data
            python_scalar = numpy_scalar or isinstance(data, (bool, int, float))
            if python_scalar:
                if numpy_scalar:
                    default_dt = _dtype_name(data.dtype, default='float32')
                else:
                    default_dt = (
                        'bool' if isinstance(data, bool)
                        else 'int32' if isinstance(data, int)
                        else 'float32'
                    )
                dt = _dtype_name(dtype, default=default_dt)
                scalar_dt = to_dtype(dt)
                normalized = scalar_dt.const(scalar_value)
                dtype_id = _dtype_id(dt)
                target_device_id = _device_id(self._device)
                if dtypes.is_bool(scalar_dt) or dtypes.is_int(scalar_dt):
                    value = int(bool(normalized)) if dtypes.is_bool(scalar_dt) else int(normalized)
                    if value < I64_MIN or value > I64_MAX:
                        raise ValueError(f'scalar {value} is out of int64 range')
                    self._tensor = _ffi._lib.poly_tensor_const_int_by_id(
                        self._ctx, value, dtype_id, target_device_id
                    )
                else:
                    self._tensor = _ffi._lib.poly_tensor_const_float_by_id(
                        self._ctx, float(normalized), dtype_id, target_device_id
                    )
                if not self._tensor:
                    raise RuntimeError('C-owned scalar Tensor construction failed')
                current_raw = self._core_uop_physical_raw(self._tensor)
                if not current_raw:
                    raise RuntimeError('scalar Tensor has no physical root')
                current_uop = UOp(self._ctx, current_raw)
                self._data = None
                self._dtype_str = dt
            else:
                _ensure_frontend_buffer_release_registered(self._ctx)
                if dtype is None and isinstance(data, np.ndarray):
                    dt = _dtype_name(data.dtype, default='float32')
                elif dtype is None and isinstance(data, (list, tuple)):
                    # Pinned Tensor.__init__ infers bool/default-int/default-float
                    # from flattened list contents (tensor.py:96-100).
                    dt = _dtype_name(dtypes.from_py(data), default='float32')
                else:
                    dt = _dtype_name(dtype, default='float32')
                import_dt = dt
                post_cast_dt = None
                np_dt = _to_np_dtype(dt)
                if dt == 'bfloat16':
                    import_dt = 'float32'
                    post_cast_dt = dt
                    np_dt = np.float32
                arr = np.ascontiguousarray(data, dtype=np_dt)
                self._data = arr.ravel()
                self._dtype_str = dt
                # UOp.from_host creates the BUFFER UOp, registers a PolyBuffer
                # wrapping the NumPy host bytes in ctx->buffers, and wraps in
                # RESHAPE when ndim>1.
                # The frontend keeps a second strong owner entry keyed by the
                # C-side PolyBuffer* address value, not by the BUFFER UOp.
                dtype_id = _dtype_id(import_dt)
                if len(arr.shape) > 0:
                    dims, ndim = _int64_array(arr.shape)
                else:
                    dims, ndim = None, 0
                owner_uop = None
                if arr.ndim > 0:
                    # Pinned UOp._frompy builds a deviceful PYTHON source before
                    # casting BF16/FP8 staging bytes and adding COPY to the
                    # requested device (uop/ops.py:752-764). Keep the byte owner
                    # on that exact C-owned physical source.
                    self._tensor = _ffi._lib.poly_tensor_from_host_by_id(
                        self._ctx, self._data.ctypes.data, self._data.nbytes,
                        dtype_id, dims, ndim,
                    )
                    if not self._tensor:
                        raise RuntimeError('poly_tensor_from_host_by_id failed')
                    current_raw = self._core_uop_physical_raw(self._tensor)
                    if not current_raw:
                        raise RuntimeError('host Tensor source has no physical root')
                    current_uop = UOp(self._ctx, current_raw)
                    owner_uop = current_uop
                    if post_cast_dt is not None:
                        self._tensor = _ffi._lib.poly_tensor_cast_by_id(
                            self._ctx, self._tensor, _dtype_id(post_cast_dt)
                        )
                        if not self._tensor:
                            raise RuntimeError(
                                f'poly_tensor_cast_by_id failed for dtype {post_cast_dt}'
                            )
                        current_raw = self._core_uop_physical_raw(self._tensor)
                        if not current_raw:
                            raise RuntimeError('cast host Tensor has no physical root')
                        current_uop = UOp(self._ctx, current_raw)
                    imported_tensor_from_host = True
                else:
                    imported = UOp.from_host(
                        self._ctx, self._data.ctypes.data, self._data.nbytes,
                        dtype_id, dims, ndim,
                    )
                    current_uop = imported
                    if post_cast_dt is not None:
                        cast_uop = _ffi._lib.poly_cast_by_id(
                            self._ctx, imported.raw, _dtype_id(post_cast_dt)
                        )
                        if not cast_uop:
                            raise RuntimeError(f'poly_cast_by_id failed for dtype {post_cast_dt}')
                        current_uop = UOp(self._ctx, cast_uop)
                    owner_uop = imported
                key_uop = owner_uop.buffer or owner_uop
                key = _buffer_key(self._ctx, key_uop)
                if key:
                    _host_buffers[key] = self._data

        self._requires_grad = bool(requires_grad)
        if imported_tensor_from_host:
            source_device_id = int(_ffi._lib.poly_tensor_device(self._tensor))
            target_device_id = _device_id(self._device)
            if target_device_id != source_device_id:
                self._tensor = _ffi._lib.poly_tensor_to_device(
                    self._ctx, self._tensor, target_device_id
                )
                if not self._tensor:
                    raise RuntimeError(f'poly_tensor_to_device failed for {self._device}')
        elif self._tensor is None and current_uop is not None:
            source_device = disk_device if imported_from_disk else 'CPU'
            target_device_id = _device_id(self._device)
            source_device_id = _device_id(source_device)
            if imported_from_disk and target_device_id != source_device_id:
                source = self._core_create(current_uop, _POLY_TENSOR_VALUE, source_device)
                self._tensor = _ffi._lib.poly_tensor_to_device(
                    self._ctx, source, target_device_id
                )
                if not self._tensor:
                    raise RuntimeError(f'poly_tensor_to_device failed for {self._device}')
            else:
                self._tensor = self._core_create(current_uop, _POLY_TENSOR_VALUE, self._device)
        if self._requires_grad:
            self._sync_core_requires_grad(force=True)

        self._grad = None
        # Match tinygrad's public parameter marker.  This is independent from
        # Polygrad's private requires_grad switch, which controls which live
        # logical roots the C autograd bridge targets.
        self._is_param = True
        all_tensors[weakref.ref(self)] = None

    # --- Core PolyTensor bridge ---

    def _core_create(self, uop, role=_POLY_TENSOR_VALUE, device=None):
        # Pinned Tensor.__init__ stores a supplied UOp directly
        # (tensor.py:76-121). The C boundary makes that exact current root explicit;
        # the legacy raw-C constructor remains available to logical/import callers.
        return Tensor._core_create_with_roots_for(
            self._ctx, uop, uop, role, self._device if device is None else device
        )

    @staticmethod
    def _core_create_with_roots_for(ctx, logical, physical, role, device):
        logical_raw = _uop_raw(logical)
        if not logical_raw:
            return None
        physical_raw = _uop_raw(physical) if physical is not None else None
        return _ffi._lib.poly_tensor_create_with_roots(
            ctx, logical_raw, physical_raw, int(role), _device_id(device)
        )

    def _core_create_with_roots(self, logical, physical=None, role=_POLY_TENSOR_VALUE, device=None):
        return Tensor._core_create_with_roots_for(
            self._ctx, logical, physical, role, self._device if device is None else device
        )

    def _sync_core_requires_grad(self, *, force=False):
        if self._tensor and (force or self._requires_grad):
            _ffi._lib.poly_tensor_set_requires_grad(self._tensor, bool(self._requires_grad))

    @staticmethod
    def _core_uop_raw(tensor):
        return _ffi._lib.poly_tensor_uop(tensor) if tensor else None

    @staticmethod
    def _core_uop_logical_raw(tensor):
        return _ffi._lib.poly_tensor_uop_logical(tensor) if tensor else None

    @staticmethod
    def _core_uop_physical_raw(tensor):
        return _ffi._lib.poly_tensor_uop_physical(tensor) if tensor else None

    def _core_to_device(self, device):
        if self._tensor is None:
            raise RuntimeError("Tensor has no core PolyTensor")
        return _ffi._lib.poly_tensor_to_device(self._ctx, self._tensor, _device_id(device))

    def _core_assign(self, value):
        if self._tensor is None or value._tensor is None:
            raise RuntimeError("Tensor.assign requires core PolyTensor handles")
        return _ffi._lib.poly_tensor_assign(self._ctx, self._tensor, value._tensor)

    @staticmethod
    def _core_realize_batch(ctx, targets, *, update_stats=True):
        n = len(targets)
        in_arr = (_ffi._ptr * n)(*[t._tensor for t in targets])
        out_arr = (_ffi._ptr * n)()
        rc = _ffi._lib.poly_realize_tensors_ex(ctx, in_arr, n, out_arr, bool(update_stats))
        if rc != 0:
            return rc, []
        return 0, [out_arr[i] for i in range(n)]

    # --- Properties ---

    @property
    def uop(self):
        raw = self._core_uop_raw(self._tensor)
        return _uop_wrap(self._ctx, raw)

    @property
    def uop_logical(self):
        raw = self._core_uop_logical_raw(self._tensor)
        return _uop_wrap(self._ctx, raw)

    @property
    def uop_physical(self):
        raw = self._core_uop_physical_raw(self._tensor)
        return _uop_wrap(self._ctx, raw)

    def _graph_uop_raw(self):
        logical = self._core_uop_logical_raw(self._tensor)
        physical = self._core_uop_physical_raw(self._tensor)
        if logical and physical and _ffi._lib.poly_uop_op(logical) == _ffi.OPS.get('AFTER'):
            return physical
        return logical or self._core_uop_raw(self._tensor)

    @property
    def _graph_uop(self):
        """Root used to construct new lazy value graphs.

        Normally this is the logical/export root. After realized mutation
        effects, tinygrad's tensor root is the current BUFFER again; Polygrad
        mirrors that by using the physical root when the logical root is AFTER.
        """
        return _uop_wrap(self._ctx, self._graph_uop_raw())

    @staticmethod
    def _physicalize_result_for(ctx, logical, inputs):
        from_roots = []
        to_roots = []
        seen = set()
        for t in inputs:
            if not isinstance(t, Tensor):
                continue
            logical_root = t._graph_uop_raw()
            current_root = Tensor._core_uop_raw(t._tensor)
            if not logical_root or not current_root or logical_root == current_root:
                continue
            if logical_root in seen:
                continue
            seen.add(logical_root)
            from_roots.append(logical_root)
            to_roots.append(current_root)
        if not from_roots:
            return None
        n = len(from_roots)
        from_arr = (_ffi._ptr * n)(*from_roots)
        to_arr = (_ffi._ptr * n)(*to_roots)
        physical = _ffi._lib.poly_uop_substitute(ctx, _uop_raw(logical), from_arr, to_arr, n)
        return _uop_wrap(ctx, physical) if physical and physical != _uop_raw(logical) else None

    def _physicalize_result(self, logical, inputs):
        return Tensor._physicalize_result_for(self._ctx, logical, inputs)

    @property
    def shape(self):
        """Read shape from cached UOp fields (O(1), no allocation)."""
        if getattr(self, '_shape_override', None) is not None:
            return self._shape_override
        if self._tensor is None:
            return ()
        raw = self._core_uop_logical_raw(self._tensor) or self._core_uop_raw(self._tensor)
        if not raw:
            return ()
        return _shape_from_uop(self._ctx, raw)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def dtype(self):
        return self._dtype_str

    @property
    def device(self):
        if self._tensor is None:
            raise RuntimeError("Tensor has no core PolyTensor")
        raw = _ffi._lib.poly_device_name(
            int(_ffi._lib.poly_tensor_device(self._tensor))
        )
        if not raw:
            raise RuntimeError("core returned unknown tensor device")
        canonical = raw.decode("utf-8").upper()
        if canonical == 'DISK' and str(self._device).upper().startswith('DISK:'):
            return self._device
        return canonical

    @property
    def requires_grad(self):
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, val):
        self._requires_grad = bool(val)
        self._sync_core_requires_grad(force=True)

    def requires_grad_(self, val=True):
        self.requires_grad = val
        return self

    @property
    def is_param(self):
        return self._is_param

    @is_param.setter
    def is_param(self, val):
        self._is_param = bool(val)

    def is_param_(self, is_param=True):
        self._is_param = bool(is_param)
        return self

    @property
    def grad(self):
        return self._grad

    @property
    def T(self):
        return self.transpose()

    def numel(self):
        shape = self.shape
        if not _shape_all_int(shape):
            raise AssertionError(f'no data if shape is symbolic, self.shape={shape}')
        n = 1
        for s in shape:
            n *= s
        return n

    def size(self, dim=None):
        if dim is None:
            return self.shape
        if dim < 0:
            dim += len(self.shape)
        return self.shape[dim]

    def _apply_uop(self, fxn, *x, extra_args=(), **kwargs):
        srcs = (self,) + x
        new_uop = fxn(*[t._graph_uop for t in srcs], *extra_args, **kwargs)
        needs_input_grad = [t._requires_grad for t in srcs]
        ret = Tensor.__new__(Tensor)
        ret._grad = None
        ret._requires_grad = True if any(needs_input_grad) else None if None in needs_input_grad else False
        ret._ctx = self._ctx
        ret._dtype_str = self._dtype_str
        ret._device = self._device
        ret._data = None
        ret._is_param = True
        ret._shape_override = self.shape
        ret._tensor = ret._core_create_with_roots(
            new_uop, self._physicalize_result(new_uop, srcs), _POLY_TENSOR_VALUE, ret._device
        )
        if ret._requires_grad:
            ret._sync_core_requires_grad(force=True)
        all_tensors[weakref.ref(ret)] = None
        return ret

    def custom_kernel(self, *lst, fxn, grad_fxn=None):
        """Call a custom SINK kernel written in UOps.

        Mirrors tinygrad's alpha `Tensor.custom_kernel`: inputs are made
        contiguous, placeholder PARAM UOps are passed to `fxn`, the returned
        SINK body is wrapped in CALL, and every source tensor is returned as
        `AFTER(source, call)`.
        """
        srcs = (self,) + tuple(lst)
        for t in srcs:
            if not isinstance(t, Tensor):
                raise TypeError('custom_kernel expects Tensor arguments')
            if t._ctx != self._ctx:
                raise ValueError('custom_kernel tensors must share a context')
        contig = tuple(
            t if t._graph_uop and t._graph_uop.op == _ffi.OPS.get('AFTER') else t.contiguous()
            for t in srcs
        )
        placeholders = [UOp.placeholder_like(t.uop, slot=i) for i, t in enumerate(contig)]
        body = fxn(*placeholders)
        if not isinstance(body, UOp):
            raise TypeError('custom_kernel fxn must return a UOp SINK body')
        input_arr = (_ffi._ptr * len(contig))(*[t._tensor for t in contig])
        output_arr = (_ffi._ptr * len(contig))()
        if _ffi._lib.poly_tensor_custom_kernel(
            self._ctx, body.raw, input_arr, len(contig), output_arr
        ) != 0:
            raise RuntimeError('poly_tensor_custom_kernel failed')
        outs = []
        afters = []
        physical_afters = []
        for t, core in zip(contig, output_arr):
            logical = _uop_wrap(t._ctx, self._core_uop_logical_raw(core))
            physical = _uop_wrap(t._ctx, self._core_uop_physical_raw(core))
            afters.append(logical)
            physical_afters.append(physical)
            out = Tensor(
                _ctx=t._ctx,
                _tensor=core,
                _dtype=t._dtype_str,
                _device=t._device,
            )
            if t._requires_grad:
                out.requires_grad = True
            outs.append(out)
        call = afters[0].src[1]
        _custom_kernel_grad_records.append({
            'ctx': _ptr_value(self._ctx),
            'call': call,
            'args': tuple(call.src[1:]),
            'afters': tuple(afters),
            'physical_afters': tuple(physical_afters),
            'grad_fxn': grad_fxn,
        })
        return outs

    # --- Realization ---

    def _live_grad_targets(self):
        """Find gradient targets the same way tinygrad does.

        tinygrad has no per-tensor input graph. Polygrad accepts reachability
        through either half of a logical/current alias pair, but the reverse
        pass must prefer the current executable root. The logical root is
        retained for export/provenance and can describe a stale pre-realization
        value (notably an earlier RNG counter version).
        """
        targets = []
        stale_refs = []
        root_raw = Tensor._core_uop_raw(self._tensor)
        root = _uop_wrap(self._ctx, root_raw)
        ctx_key = _ptr_value(self._ctx)
        for tref in list(all_tensors):
            t = tref()
            if t is None:
                stale_refs.append(tref)
                continue
            if _ptr_value(t._ctx) != ctx_key or not t._requires_grad:
                continue
            target = t._graph_uop
            current_raw = Tensor._core_uop_raw(t._tensor)
            target_raw = _uop_raw(target)
            grad_root = None
            if current_raw and _ffi._lib.poly_uop_reachable(self._ctx, root, current_raw):
                grad_root = current_raw
            elif (target_raw and current_raw != target_raw and
                  _ffi._lib.poly_uop_reachable(self._ctx, root, target_raw)):
                grad_root = target_raw
            if grad_root:
                targets.append((t, grad_root))
        for tref in stale_refs:
            all_tensors.pop(tref, None)
        return targets

    @staticmethod
    def _grad_many_raw(ctx, root, initial_grad, wrts, *, return_present=False):
        wrts = tuple(_uop_raw(w) for w in wrts if _uop_raw(w))
        if not wrts:
            return ((), ()) if return_present else ()
        wrt_arr = (_ffi._ptr * len(wrts))(*wrts)
        out_arr = (_ffi._ptr * len(wrts))()
        present_arr = (ctypes.c_uint8 * len(wrts))()
        rc = _ffi._lib.poly_grad_many_ex(
            ctx, _uop_raw(root), _uop_raw(initial_grad), wrt_arr, len(wrts),
            out_arr, present_arr,
        )
        if rc != 0:
            raise RuntimeError('poly_grad_many failed')
        grads = tuple(out_arr)
        if return_present:
            return grads, tuple(bool(v) for v in present_arr)
        return grads

    @staticmethod
    def _custom_grad_records_for(ctx, root):
        root_raw = _uop_raw(root)
        if not root_raw:
            return []
        ctx_key = _ptr_value(ctx)
        out = []
        for rec in list(_custom_kernel_grad_records):
            if rec.get('ctx') != ctx_key:
                continue
            active = []
            active_seen = set()
            for i, after in enumerate(rec['afters']):
                active_after = None
                if _ffi._lib.poly_uop_reachable(ctx, root_raw, _uop_raw(after)):
                    active_after = _uop_raw(after)
                else:
                    physical_afters = rec.get('physical_afters', ())
                    physical_after = physical_afters[i] if i < len(physical_afters) else None
                    if physical_after and _ffi._lib.poly_uop_reachable(
                        ctx, root_raw, _uop_raw(physical_after)
                    ):
                        active_after = _uop_raw(physical_after)
                active_key = _ptr_value(active_after) if active_after else 0
                # UOps are hash-consed. If one source appears more than once,
                # tinygrad's k.src.index(data) assigns the accumulated AFTER
                # gradient to its first CALL slot only.
                if active_key and active_key not in active_seen:
                    after_src = UOp(ctx, active_after).src
                    if (len(after_src) != 2 or
                            after_src[1].op != _ffi.OPS.get('CALL')):
                        raise RuntimeError('custom_kernel active output is not AFTER(data, CALL)')
                    call_src = after_src[1].src
                    if len(call_src) != len(rec['args']) + 1:
                        raise RuntimeError('custom_kernel active CALL argument count changed')
                    active_seen.add(active_key)
                    # tinygrad mixin/gradient.py:90-91 and :25-31 passes the
                    # exact reachable CALL and its matching argument slot to
                    # call_gradient. Select at Polygrad's logical/physical
                    # boundary from that same active AFTER, never from the
                    # record's other graph representation.
                    active.append((active_after, call_src[i + 1].raw, after_src[1].raw))
            if active:
                out.append((rec, tuple(active)))
        return out

    @staticmethod
    def _grad_result_raw(grad):
        if grad is None:
            return None
        if isinstance(grad, Tensor):
            return _uop_raw(grad._graph_uop)
        if isinstance(grad, UOp):
            return grad.raw
        raw = _uop_raw(grad)
        return raw if raw else None

    @staticmethod
    def _accumulate_grad_raw(ctx, current, new_grad):
        cur = _uop_raw(current)
        nxt = _uop_raw(new_grad)
        if not cur:
            return nxt
        if not nxt:
            return cur
        return _ffi._lib.poly_alu2(ctx, _ffi.OPS['ADD'], cur, nxt)

    def assign(self, x):
        """In-place assignment: self's buffer will be overwritten with x's values.
        Must be realized before use. Returns self for chaining."""
        if not isinstance(x, Tensor):
            x = Tensor(x, dtype=self._dtype_str, device=self._device)
        if self.shape != x.shape:
            x = Tensor(
                _ctx=x._ctx, _uop=x._broadcast_uop(self.shape),
                requires_grad=x._requires_grad, _dtype=x.dtype, _device=x._device,
            )
        if self.device != x.device:
            raise RuntimeError(f'assign device mismatch {self.device} != {x.device}')
        if self.dtype != x.dtype:
            raise RuntimeError(f'assign dtype mismatch {self.dtype} != {x.dtype}')

        assigned = self._core_assign(x)
        if not assigned:
            raise RuntimeError('poly_tensor_assign failed')
        self._tensor = assigned
        if self._requires_grad:
            self._sync_core_requires_grad(force=True)
        self._data = None
        return self

    def copy_from(self, data):
        """Update this tensor's existing buffer from host data without changing
        its BUFFER identity. This is the explicit Polygrad update API for
        JIT/replay loops; tinygrad's closest public mutation API is assign()."""
        logical = self.uop_logical or self.uop
        buf = logical.buffer if logical is not None else None
        if buf is None:
            raise RuntimeError('copy_from requires a tensor backed by a BUFFER UOp')
        np_dt = _to_np_dtype(to_dtype(self._dtype_str))
        arr = np.asarray(data, dtype=np_dt)
        if arr.size != self.numel():
            raise ValueError(f'copy_from size mismatch {arr.size} != {self.numel()}')
        arr = np.ascontiguousarray(arr.reshape(self.shape))
        ptr = ctypes.c_void_p(arr.ctypes.data)
        physical = self.uop_physical
        physical_raw = physical.raw if physical is not None else None
        write_buf = buf
        if physical is not None:
            physical_buf = physical.buffer
            if physical_buf is None:
                physical_raw = None
            elif physical_buf.raw != buf.raw:
                write_buf = physical_buf
        target_device = _device_id(self._device)
        rc = _ffi._lib.poly_buffer_ensure_device_allocated(
            self._ctx, write_buf.raw, target_device
        )
        if rc != 0:
            raise RuntimeError('poly_buffer_ensure_device_allocated failed')
        rc = _ffi._lib.poly_buffer_write(self._ctx, write_buf.raw, ptr, arr.nbytes)
        if rc != 0:
            raise RuntimeError('poly_buffer_write failed')
        if logical is not None:
            rc = _ffi._lib.poly_tensor_replace_roots(
                self._ctx,
                self._tensor,
                logical.raw,
                physical_raw,
                _POLY_TENSOR_VALUE,
                _device_id(self._device),
            )
            if rc != 0:
                raise RuntimeError('poly_tensor_replace_roots failed during copy_from')
        self._data = None
        return self

    def update_from(self, data):
        return self.copy_from(data)

    def realize(self, *lst, do_update_stats=True):
        """Triggers the computation needed to create these Tensor(s).
        Batches tensors into the shared core realization path, which publishes
        the becomes-map to every live PolyTensor before scheduling."""
        targets = []
        seen = set()
        for x in (self,) + lst:
            current_raw = Tensor._core_uop_raw(x._tensor)
            device_id = int(_ffi._lib.poly_tensor_device(x._tensor))
            key = (_ptr_value(current_raw), device_id)
            if key in seen:
                continue
            seen.add(key)
            targets.append(x)
        if not targets:
            return self
        rc, _ = Tensor._core_realize_batch(
            self._ctx, targets, update_stats=do_update_stats
        )
        if rc != 0:
            devices = ', '.join(sorted({str(t._device) for t in targets}))
            raise RuntimeError(f'poly_realize_tensors failed for device(s): {devices}')
        return self

    def _buffer(self) -> Buffer:
        """Return the runtime Buffer backing this tensor.
        Matches tinygrad's temporary base-dtype/contiguous readback path."""
        if str(self._device).upper().startswith('DISK:'):
            self.realize()
            identity = self.uop.buffer
            if identity is not None:
                return Buffer(self._ctx, identity, self._dtype_str, self.numel())
        # Pinned tensor.py:259-266 clones a device-free source (or MULTI
        # source) to CPU for readback. Default execution must not recover the old implicit
        # realize-time placement merely because wrapper metadata names a
        # preferred execution backend.
        x = self.cast(to_dtype(self._dtype_str).base).contiguous()
        if int(_ffi._lib.poly_uop_device(self.uop.raw)) == 0 or isinstance(self._device, tuple):
            x = x.clone("CPU")
        x.realize()
        return Buffer(self._ctx, x.uop.buffer, x._dtype_str, x.numel())

    def data(self):
        """Return tensor contents as a shaped memoryview, matching tinygrad."""
        shape = self.shape
        dtype = to_dtype(self._dtype_str).base
        if 0 in shape:
            return memoryview(bytearray(0)).cast(dtype.fmt)
        assert _shape_all_int(shape), f'no data if shape is symbolic, self.shape={shape}'
        assert dtype.fmt is not None, f'no fmt dtype for {dtype}'
        assert dtype.fmt != 'e' or sys.version_info >= (3, 12)
        return self._buffer().as_memoryview().cast(dtype.fmt, shape)

    def numpy(self):
        """Return the value of this tensor as a numpy.ndarray.
        Matches tinygrad's Tensor.numpy signature."""
        from .dtype import _to_np_dtype
        shape = self.shape
        dtype = to_dtype(self._dtype_str)
        if dtype.base in {dtypes.bfloat16, *dtypes.fp8s}:
            return self.float().numpy()
        np_dt = _to_np_dtype(dtype)
        if 0 in shape:
            return np.empty(shape, dtype=np_dt)
        assert _shape_all_int(shape), f'no data if shape is symbolic, self.shape={shape}'
        return self._buffer().numpy().reshape(shape)

    @staticmethod
    def from_url(url, gunzip=False, **kwargs):
        from .helpers import fetch
        return Tensor(fetch(url, gunzip=gunzip), **kwargs)

    @staticmethod
    def numpy_many(*tensors):
        """Return multiple tensors as numpy arrays after one batched realize.

        This is a Polygrad embedding/readback helper. tinygrad exposes batched
        realization through Tensor.realize(*lst), but readback remains per
        tensor via numpy()/tolist().
        """
        if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
            tensors = tuple(tensors[0])
        if not tensors:
            return tuple()
        for t in tensors:
            if not isinstance(t, Tensor):
                raise TypeError('Tensor.numpy_many expects Tensor arguments')
        tensors[0].realize(*tensors[1:])
        return tuple(t.numpy() for t in tensors)

    def item(self):
        """Return scalar value."""
        arr = self.numpy()
        if arr.size != 1:
            raise ValueError(f'item() requires scalar tensor, got shape {self.shape}')
        return arr.flat[0].item()

    def tolist(self):
        # Pinned tensor.py:285-300 widens half through the graph before
        # returning Python values. This is distinct from widening values after
        # reading rounded float16 storage through numpy().
        if to_dtype(self._dtype_str).base == dtypes.half:
            return self.float().tolist()
        return self.numpy().tolist()

    def detach(self):
        # Pinned mixin/elementwise.py:33-37 is one DETACH Tensor ALU.
        core = _ffi._lib.poly_tensor_detach(self._ctx, self._tensor)
        ret = self._make_result_from_core(core, self.shape, [self])
        ret.requires_grad = False
        return ret

    def clone(self, device=None):
        from .device import Device

        dev = self._device if device is None else Device.canonicalize(device)
        ret = Tensor.empty(
            self.shape,
            _ctx=self._ctx,
            requires_grad=self._requires_grad,
            dtype=self._dtype_str,
            device=dev,
        )
        cloned = _ffi._lib.poly_tensor_clone_into(self._ctx, ret._tensor, self._tensor)
        if not cloned:
            raise RuntimeError('poly_tensor_clone_into failed')
        ret._tensor = cloned
        if self._grad is not None:
            ret._grad = self._grad.clone(device=dev)
        ret._is_param = self._is_param
        return ret

    def to(self, device):
        from .device import Device

        dev = Device.canonicalize(device)
        if dev == self._device:
            return self

        core_tensor = self._core_to_device(dev)

        out = Tensor(
            _ctx=self._ctx,
            _tensor=core_tensor,
            _data=self._data,
            _dtype=self._dtype_str,
            _device=dev,
            requires_grad=self._requires_grad,
        )
        out._grad = self._grad.to(dev) if self._grad is not None else None
        out._is_param = self._is_param
        return out

    def to_(self, device):
        moved = self.to(device)
        if moved is self:
            return self
        self._tensor = moved._tensor
        self._data = moved._data
        self._dtype_str = moved._dtype_str
        self._device = moved._device
        self._requires_grad = moved._requires_grad
        self._grad = moved._grad
        self._is_param = moved._is_param
        return self

    def shard_(self, devices, axis=None):
        if isinstance(devices, str):
            return self.to_(devices)
        devices = tuple(devices)
        if len(devices) == 1:
            return self.to_(devices[0])
        raise NotImplementedError('Polygrad Python does not yet support multi-device shard_')

    def cpu(self):
        return self.to('cpu')

    def cuda(self):
        return self.to('cuda')

    def contiguous(self, *args, **kwargs):
        """Returns a contiguous tensor."""
        if args or kwargs:
            return self._apply_uop(UOp.contiguous, extra_args=args, **kwargs)
        # Pinned Tensor.contiguous -> UOp.contiguous (tensor.py:742-746,
        # uop/ops.py:587-591). C owns both retained/current roots.
        core = _ffi._lib.poly_tensor_contiguous(self._ctx, self._tensor)
        return self._make_result_from_core(core, self.shape, [self])

    # --- Dtype casting ---

    def cast(self, dtype):
        """Cast tensor to the given dtype. No-op if already that dtype."""
        target_name = _dtype_name(dtype, default=self._dtype_str)
        if target_name == self._dtype_str:
            return self
        dtype_id = _dtype_id(target_name)
        core = _ffi._lib.poly_tensor_cast_by_id(self._ctx, self._tensor, dtype_id)
        if not core:
            raise RuntimeError(f'poly_tensor_cast_by_id failed for dtype {target_name}')
        return self._make_result_from_core(core, self.shape, [self])

    def bitcast(self, dtype):
        """Bit reinterpretation matching tinygrad Tensor.bitcast."""
        target = to_dtype(dtype)
        current = to_dtype(self._dtype_str)
        if current == target:
            return self
        old_size, new_size = current.itemsize, target.itemsize
        if new_size != old_size:
            if not self.shape or (self.shape[-1] * old_size) % new_size != 0:
                raise RuntimeError('unsupported size in bitcast')
            old_uint = to_dtype(f'uint{8 * old_size}')
            new_uint = to_dtype(f'uint{8 * new_size}')
            tmp = self.bitcast(old_uint)
            if new_size > old_size:
                rate = new_size // old_size
                tmp = tmp.reshape(self.shape[:-1] + (self.shape[-1] // rate, rate))
                parts = [tmp[..., i:i + 1].cast(new_uint).lshift(8 * i * old_size)
                         for i in range(rate)]
                combined = functools.reduce(lambda a, b: a + b, parts).squeeze(-1)
                return combined.bitcast(target)
            parts = [tmp.rshift(8 * i * new_size) for i in range(old_size // new_size)]
            return Tensor.stack(*parts, dim=-1).flatten(-2).cast(new_uint).bitcast(target)
        core = _ffi._lib.poly_tensor_bitcast_by_id(
            self._ctx, self._tensor, _dtype_id(target)
        )
        if not core:
            raise RuntimeError(f'poly_tensor_bitcast_by_id failed for dtype {target}')
        return self._make_result_from_core(core, self.shape, [self])

    def half(self):
        """Cast to float16."""
        return self.cast('float16')

    def float(self):
        """Cast to float32."""
        return self.cast('float32')

    def double(self):
        """Cast to float64."""
        return self.cast('float64')

    def int(self):
        """Cast to int32."""
        return self.cast('int32')

    def long(self):
        """Cast to int64."""
        return self.cast('int64')

    def short(self):
        """Cast to int16."""
        return self.cast('int16')

    def bool(self):
        """Cast to bool."""
        return self.cast('bool')

    def bfloat16(self):
        """Cast to bfloat16."""
        return self.cast('bfloat16')

    # --- Internal helpers ---

    def _make_result(self, uop, shape, inputs):
        dev = self._infer_device(inputs)
        # Result dtype comes from the core graph now; inheriting from Python
        # inputs was the stale behavior that made bool/int ops look floaty.
        dt = _uop_dtype_name(self._ctx, uop, self._dtype_str)
        physical = self._physicalize_result(uop, inputs)
        ret = Tensor.__new__(Tensor)
        ret._ctx = self._ctx
        ret._device = dev
        ret._tensor = self._core_create_with_roots(uop, physical, _POLY_TENSOR_VALUE, dev)
        ret._shape_override = tuple(shape) if shape is not None else None
        ret._data = None
        ret._dtype_str = dt
        ret._requires_grad = any(t._requires_grad for t in inputs)
        ret._grad = None
        ret._is_param = True
        if ret._requires_grad:
            ret._sync_core_requires_grad(force=True)
        all_tensors[weakref.ref(ret)] = None
        return ret

    def _make_result_from_core(self, core_tensor, shape, inputs):
        if not core_tensor:
            raise RuntimeError('core Tensor operation failed')
        current = self._core_uop_raw(core_tensor)
        if not current:
            raise RuntimeError('core Tensor operation returned no current UOp')
        dev = self._infer_device(inputs)
        ret = Tensor.__new__(Tensor)
        ret._ctx = self._ctx
        ret._device = dev
        ret._tensor = core_tensor
        ret._shape_override = tuple(shape) if shape is not None else None
        ret._data = None
        ret._dtype_str = _uop_dtype_name(self._ctx, current, self._dtype_str)
        ret._requires_grad = any(t._requires_grad for t in inputs)
        ret._grad = None
        ret._is_param = True
        if ret._requires_grad:
            ret._sync_core_requires_grad(force=True)
        all_tensors[weakref.ref(ret)] = None
        return ret

    def _infer_device(self, inputs):
        from .device import Device

        devices = {
            t._device
            for t in inputs
            if hasattr(t, '_device')
        }
        if not devices:
            return self._device
        if len(devices) != 1:
            raise RuntimeError(f'Mixed devices are not supported: {sorted(devices)}')
        return Device.canonicalize(next(iter(devices)))

    def _resolve_dim(self, dim, *, extra=False):
        total = self.ndim + int(extra)
        lo = -max(1, total)
        hi = max(1, total) - 1
        if not lo <= dim <= hi:
            raise IndexError(f'dim={dim} out of range {[lo, hi]}')
        return dim + total if dim < 0 else dim

    def _ensure_tensor(self, other):
        if isinstance(other, Tensor):
            return other
        if isinstance(other, np.generic):
            other = other.item()
        if isinstance(other, (bool, int, float)):
            self_dt = to_dtype(self._dtype_str)
            if dtypes.is_float(self_dt) or (dtypes.is_int(self_dt) and isinstance(other, int) and not isinstance(other, bool)):
                const_dtype = self_dt
            else:
                const_dtype = dtypes.from_py(other)
            const_dtype_name = _dtype_name(const_dtype, default='float32')
            dt = to_dtype(const_dtype_name)
            normalized = dt.const(other)
            dtype_id = _dtype_id(const_dtype_name)
            if dtypes.is_bool(dt) or dtypes.is_int(dt):
                int_value = int(bool(normalized)) if dtypes.is_bool(dt) else int(normalized)
                if int_value < I64_MIN or int_value > I64_MAX:
                    raise ValueError(f'scalar {int_value} is out of int64 range')
                tensor = _ffi._lib.poly_tensor_const_int_by_id(
                    self._ctx, int_value, dtype_id, _device_id(self._device)
                )
            else:
                tensor = _ffi._lib.poly_tensor_const_float_by_id(
                    self._ctx, float(normalized), dtype_id, _device_id(self._device)
                )
            return _created_tensor(
                self._ctx, tensor, const_dtype_name, self._device, False,
                'C-owned internal scalar Tensor construction',
            )
        raise TypeError(f'Cannot convert {type(other)} to Tensor')

    def const_like(self, value):
        """Pinned CreationMixin.const_like: typed CONST broadcast to this shape."""
        return Tensor(
            value, dtype=self.dtype, device=self.device, _ctx=self._ctx
        )._broadcast_to_tensor(self.shape)

    def _broadcast_shape(self, other_shape):
        """Compute broadcast shape between self.shape and other_shape."""
        a, b = self.shape, other_shape
        if not a:
            return b
        if not b:
            return a
        ndim = max(len(a), len(b))
        a = (1,) * (ndim - len(a)) + a
        b = (1,) * (ndim - len(b)) + b
        result = []
        for x, y in zip(a, b):
            if x == y:
                result.append(x)
            elif x == 1:
                result.append(y)
            elif y == 1:
                result.append(x)
            else:
                raise IndexError(f'shape mismatch: objects cannot be broadcast to a single shape {(self.shape, other_shape)}')
        return tuple(result)

    def _broadcast_uop(self, target_shape):
        """Return a UOp that broadcasts self to target_shape via RESHAPE+EXPAND.

        Matches tinygrad's _broadcast_to: explicit shape ops so the scheduler
        sees EXPAND UOps instead of implicit ALU broadcasting.
        """
        target_shape = tuple(target_shape)
        if self.shape == target_shape:
            return self._graph_uop
        uop = self._graph_uop
        cur_shape = self.shape
        target_nd = len(target_shape)
        if len(cur_shape) > target_nd:
            raise ValueError(f"cannot broadcast tensor to fewer dimensions. shape={cur_shape} to new_shape={target_shape}")
        aligned_shape = (1,) * (target_nd - len(cur_shape)) + tuple(cur_shape)
        for s, ns in zip(aligned_shape, target_shape):
            if not _is_symbolic_dim(ns) and int(ns) < 0:
                raise ValueError(f"negative dimensions are not allowed: {target_shape}")
            if not (s == ns or s == 1):
                raise ValueError(f"cannot broadcast {cur_shape} to new_shape={target_shape}")
        # Scalar tensor or lower-rank: pad left with 1s
        if len(cur_shape) < target_nd:
            cur_shape = aligned_shape
            if not _shape_all_int(cur_shape):
                raise NotImplementedError('symbolic movement reshape is not implemented')
            dims, n = _int64_array(cur_shape)
            uop = _ffi._lib.poly_reshape(self._ctx, uop, dims, n)
        # Expand any dimensions where size 1 → target size
        if cur_shape != target_shape:
            if _shape_has_symbolic(target_shape):
                dims = _shape_uop_array(self._ctx, target_shape)
                uop = _ffi._lib.poly_expand_uop(self._ctx, uop, dims, len(target_shape))
            else:
                dims, n = _int64_array(target_shape)
                uop = _ffi._lib.poly_expand(self._ctx, uop, dims, n)
            if not uop:
                raise RuntimeError('poly_expand failed')
        return uop

    def _broadcast_to_tensor(self, target_shape):
        """Return the ordered Tensor occurrence used by an elementwise ALU.

        Pinned tinygrad's _broadcasted/_broadcast_to first constructs movement
        Tensor occurrences, then Tensor.alu consumes their current UOps
        (mixin/__init__.py:439-449, mixin/movement.py:116-128).
        """
        target_shape = tuple(target_shape)
        if self.shape == target_shape:
            return self
        if self.ndim > len(target_shape):
            raise ValueError(
                f"cannot broadcast tensor to fewer dimensions. shape={self.shape} "
                f"to new_shape={target_shape}"
            )
        aligned_shape = (1,) * (len(target_shape) - self.ndim) + tuple(self.shape)
        if not all(s == ns or s == 1 for s, ns in zip(aligned_shape, target_shape)):
            raise ValueError(f"cannot broadcast {self.shape} to new_shape={target_shape}")
        reshaped = self.reshape(aligned_shape)
        expanded = reshaped.expand(target_shape)
        return reshaped if expanded.shape == reshaped.shape else expanded

    # --- Element-wise arithmetic ---

    def _broadcasted(self, other, reverse=False):
        other = self._ensure_tensor(other)
        x, y = (self, other) if not reverse else (other, self)
        out_shape = x._broadcast_shape(y.shape)
        x, y = x._broadcast_to_tensor(out_shape), y._broadcast_to_tensor(out_shape)
        if x.dtype != y.dtype:
            out_dtype = least_upper_dtype(to_dtype(x.dtype), to_dtype(y.dtype))
            x, y = x.cast(out_dtype), y.cast(out_dtype)
        return x, y, out_shape

    def _binop(self, other, op_name, reverse=False):
        """Build a binary Tensor ALU from ordered current operand occurrences."""
        x, y, out_shape = self._broadcasted(other, reverse)
        core = _ffi._lib.poly_tensor_alu2(
            self._ctx, _ffi.OPS[op_name], x._tensor, y._tensor
        )
        return self._make_result_from_core(core, out_shape, [x, y])

    def bitwise_and(self, other, reverse=False):
        if not (dtypes.is_int(to_dtype(self.dtype)) or dtypes.is_bool(to_dtype(self.dtype))):
            raise RuntimeError(f'bitwise ops require integer or bool dtype, got {self.dtype}')
        return self._binop(other, 'AND', reverse)

    def bitwise_or(self, other, reverse=False):
        if not (dtypes.is_int(to_dtype(self.dtype)) or dtypes.is_bool(to_dtype(self.dtype))):
            raise RuntimeError(f'bitwise ops require integer or bool dtype, got {self.dtype}')
        return self._binop(other, 'OR', reverse)

    def bitwise_xor(self, other, reverse=False):
        if not (dtypes.is_int(to_dtype(self.dtype)) or dtypes.is_bool(to_dtype(self.dtype))):
            raise RuntimeError(f'bitwise ops require integer or bool dtype, got {self.dtype}')
        return self._binop(other, 'XOR', reverse)

    def __and__(self, other):
        return self.bitwise_and(other)

    def __rand__(self, other):
        return self.bitwise_and(other, True)

    def __or__(self, other):
        return self.bitwise_or(other)

    def __ror__(self, other):
        return self.bitwise_or(other, True)

    def __xor__(self, other):
        return self.bitwise_xor(other)

    def __rxor__(self, other):
        return self.bitwise_xor(other, True)

    def lshift(self, other, reverse=False):
        return self._binop(other, 'SHL', reverse)

    def rshift(self, other, reverse=False):
        return self._binop(other, 'SHR', reverse)

    def __lshift__(self, other):
        return self.lshift(other)

    def __rlshift__(self, other):
        return self.lshift(other, True)

    def __rshift__(self, other):
        return self.rshift(other)

    def __rrshift__(self, other):
        return self.rshift(other, True)

    def threefry(self, seed):
        return self._binop(seed, 'THREEFRY')

    def __add__(self, other):
        return self._binop(other, 'ADD')

    def __radd__(self, other):
        # Pinned __radd__ preserves scalar-first source order through
        # add(..., reverse=True) (mixin/elementwise.py:72-88,267-268).
        return self.add(other, True)

    def add(self, other, reverse=False):
        return self._binop(other, 'ADD', reverse)

    def __sub__(self, other):
        return self.sub(other)

    def __rsub__(self, other):
        return self.sub(other, reverse=True)

    def sub(self, other, reverse=False):
        # C owns tinygrad's `a + (-b)` topology so every frontend receives
        # the same ordered physical graph (mixin/elementwise.py:90-109).
        return self._binop(other, 'SUB', reverse)

    def __mul__(self, other):
        return self._binop(other, 'MUL')

    def __rmul__(self, other):
        # Pinned __rmul__ preserves scalar-first source order through
        # mul(..., reverse=True) (mixin/elementwise.py:110-126,273-274).
        return self.mul(other, True)

    def mul(self, other, reverse=False):
        return self._binop(other, 'MUL', reverse)

    def __truediv__(self, other):
        return self.div(other)

    def __rtruediv__(self, other):
        return self.div(other, reverse=True)

    def div(self, other, reverse=False, rounding_mode=None):
        if rounding_mode is not None:
            raise NotImplementedError(f"rounding_mode={rounding_mode!r} is not supported")
        other = self._ensure_tensor(other)
        dividend, divisor = (other, self) if reverse else (self, other)
        out_shape = dividend._broadcast_shape(divisor.shape)
        core = _ffi._lib.poly_tensor_div(
            self._ctx, dividend._tensor, divisor._tensor
        )
        return self._make_result_from_core(core, out_shape, [dividend, divisor])

    def __neg__(self):
        if dtypes.is_bool(to_dtype(self.dtype)):
            return self.logical_not()
        return self * -1

    def logical_not(self):
        return self.cast('bool') != True

    def neg(self):
        return self.__neg__()

    def pow(self, other, reverse=False):
        if isinstance(other, np.generic):
            other = other.item()
        base, exponent, out_shape = self._broadcasted(other, reverse)
        if (not dtypes.is_float(base.dtype) and
                not isinstance(other, Tensor) and not (isinstance(other, int) and other >= 0)):
            raise RuntimeError("base needs to be float")
        core = _ffi._lib.poly_tensor_alu2(
            self._ctx, _ffi.OPS['POW'], base._tensor, exponent._tensor
        )
        ret = self._make_result_from_core(core, out_shape, [base, exponent])
        if not reverse and not dtypes.is_float(self.dtype) and dtypes.is_float(exponent.dtype):
            return ret.round().cast(self.dtype)
        return ret

    def __pow__(self, other):
        return self.pow(other)

    def __rpow__(self, other):
        return self.pow(other, reverse=True)

    # --- Comparisons (C core) ---

    def __lt__(self, other):
        return self._binop(other, 'CMPLT')

    def __eq__(self, other):
        # Pinned mixin/elementwise.py:315-322: equality is promoted CMPNE
        # followed by logical_not, not a raw mixed-dtype UOp helper.
        return self._binop(other, 'CMPNE').logical_not()

    def __ne__(self, other):
        return self._binop(other, 'CMPNE')

    def __gt__(self, other):
        # Pinned mixin/elementwise.py:312-313 swaps promoted operands and uses
        # CMPLT because tinygrad has no distinct greater-than UOp.
        return self._binop(other, 'CMPLT', reverse=True)

    def __ge__(self, other):
        return (self < other).logical_not()

    def __le__(self, other):
        return (self > other).logical_not()

    def ne(self, other):
        return self.__ne__(other)

    def eq(self, other):
        return self.__eq__(other)

    def where(self, x, y):
        """self is condition: where(cond, x, y)."""
        # Pinned tensor.py:750-772 anchors branch promotion on an existing
        # branch Tensor, not on the boolean condition.
        if isinstance(x, Tensor):
            x, y, branch_shape = x._broadcasted(y)
        elif isinstance(y, Tensor):
            y, x, branch_shape = y._broadcasted(x)
        else:
            x, y, branch_shape = self._ensure_tensor(x)._broadcasted(y)
        out_shape = _broadcast_shapes(self.shape, branch_shape)
        cond = self.cast('bool')._broadcast_to_tensor(out_shape)
        x = x._broadcast_to_tensor(out_shape)
        y = y._broadcast_to_tensor(out_shape)
        core = _ffi._lib.poly_tensor_alu3(
            self._ctx, _ffi.OPS['WHERE'], cond._tensor, x._tensor, y._tensor
        )
        return self._make_result_from_core(core, out_shape, [cond, x, y])

    def maximum(self, other):
        return self._binop(other, 'MAX')

    def minimum(self, other):
        # Pinned _broadcasted constructs ordered movement/cast Tensor
        # occurrences before minimum's inverse/maximum/inverse program
        # (mixin/__init__.py:439-449; mixin/elementwise.py:381-393).
        x, y, out_shape = self._broadcasted(other)
        core = _ffi._lib.poly_tensor_minimum(self._ctx, x._tensor, y._tensor)
        return self._make_result_from_core(core, out_shape, [x, y])

    def clamp(self, min_=None, max_=None):
        if min_ is None and max_ is None:
            raise RuntimeError("at least one of 'min_' or 'max_' must not be None")
        # Pinned clamp conditionally composes comparison/WHERE Tensor
        # operations; an omitted bound is not a finite sentinel
        # (mixin/elementwise.py:569-580).
        ret = (self < min_).where(min_, self) if min_ is not None else self
        return (ret > max_).where(max_, ret) if max_ is not None else ret

    # --- Unary math (C core composed ops) ---

    def exp2(self):
        # Pinned _ensure_float().alu(EXP2) (mixin/elementwise.py:517-527).
        base = self if dtypes.is_float(self.dtype) else self.cast(least_upper_float(to_dtype(self.dtype)))
        core = _ffi._lib.poly_tensor_alu1(
            base._ctx, _ffi.OPS['EXP2'], base._tensor
        )
        return base._make_result_from_core(core, base.shape, [base])

    def log2(self):
        # Pinned _ensure_float().alu(LOG2) (mixin/elementwise.py:505-515).
        base = self if dtypes.is_float(self.dtype) else self.cast(least_upper_float(to_dtype(self.dtype)))
        core = _ffi._lib.poly_tensor_alu1(
            base._ctx, _ffi.OPS['LOG2'], base._tensor
        )
        return base._make_result_from_core(core, base.shape, [base])

    def sqrt(self):
        # Pinned _ensure_float().alu(SQRT) (mixin/elementwise.py:460-468).
        base = self if dtypes.is_float(self.dtype) else self.cast(least_upper_float(to_dtype(self.dtype)))
        core = _ffi._lib.poly_tensor_alu1(
            base._ctx, _ffi.OPS['SQRT'], base._tensor
        )
        return base._make_result_from_core(core, base.shape, [base])

    def reciprocal(self):
        base = self if dtypes.is_float(self.dtype) else self.cast(least_upper_float(to_dtype(self.dtype)))
        core = _ffi._lib.poly_tensor_alu1(
            base._ctx, _ffi.OPS['RECIPROCAL'], base._tensor
        )
        return base._make_result_from_core(core, base.shape, [base])

    def trunc(self):
        core = _ffi._lib.poly_tensor_alu1(
            self._ctx, _ffi.OPS['TRUNC'], self._tensor
        )
        return self._make_result_from_core(core, self.shape, [self])

    def exp(self):
        core = _ffi._lib.poly_tensor_exp(self._ctx, self._tensor)
        return self._make_result_from_core(core, self.shape, [self])

    def log(self):
        core = _ffi._lib.poly_tensor_log(self._ctx, self._tensor)
        return self._make_result_from_core(core, self.shape, [self])

    def log1p(self):
        core = _ffi._lib.poly_tensor_log1p(self._ctx, self._tensor)
        return self._make_result_from_core(core, self.shape, [self])

    def expm1(self):
        core = _ffi._lib.poly_tensor_expm1(self._ctx, self._tensor)
        return self._make_result_from_core(core, self.shape, [self])

    def sin(self):
        # Pinned _ensure_float().alu(SIN), consuming the exact current Tensor
        # occurrence (mixin/elementwise.py:437-478; tensor.py:128-140).
        base = self if dtypes.is_float(self.dtype) else self.cast(least_upper_float(to_dtype(self.dtype)))
        core = _ffi._lib.poly_tensor_alu1(
            base._ctx, _ffi.OPS['SIN'], base._tensor
        )
        return base._make_result_from_core(core, base.shape, [base])

    def cos(self):
        # Pinned floating COS promotes against float32, subtracts from pi/2,
        # applies SIN, then casts back (mixin/elementwise.py:480-489).
        if dtypes.is_float(self.dtype):
            work = self.cast(least_upper_dtype(to_dtype(self.dtype), dtypes.float32))
            return (math.pi / 2 - work).sin().cast(self.dtype)
        return (math.pi / 2 - self).sin()

    def tan(self):
        # Pinned tan is the high-level SIN/COS quotient
        # (mixin/elementwise.py:902-910).
        return self.sin() / self.cos()

    def sigmoid(self):
        # Pinned mixin/elementwise.py:667-677.
        return (1 + (self * (-1 / math.log(2))).exp2()).reciprocal()

    def tanh(self):
        # Pinned mixin/elementwise.py:739-749.
        return 2.0 * (2.0 * self).sigmoid() - 1.0

    def abs(self):
        # Pinned mixin/elementwise.py:892-900.
        return self * self.sign()

    def sign(self):
        # Pinned mixin/elementwise.py:882-890 uses typed const_like branches.
        return self.ne(0).where(
            (self < 0).where(self.const_like(-1), self.const_like(1)),
            self.const_like(0),
        )

    def square(self):
        # Pinned mixin/elementwise.py:558-566.
        return self * self

    def rsqrt(self):
        # Pinned mixin/elementwise.py:802-810.
        return self.sqrt().reciprocal()

    def ceil(self):
        # Pinned mixin/elementwise.py:636-644.
        b = self.trunc()
        return (self > b).where(b + 1, b)

    def floor(self):
        # Pinned mixin/elementwise.py:646-654.
        b = self.trunc()
        return (self < b).where(b - 1, b)

    def round(self):
        # Pinned mixin/elementwise.py:872-880 implements round-half-to-even
        # through ordinary Tensor primitives and their scalar broadcast rules.
        b = self.trunc() / 2.0
        return ((self > 0).eq(b.trunc().eq(b))).where(
            (self - 0.5).ceil(), (self + 0.5).floor()
        )

    def isinf(self, detect_positive=True, detect_negative=True):
        # Pinned mixin/elementwise.py:596-604 independently gates positive
        # and negative infinity before adding the boolean results.
        return (
            self.eq(float("inf")) * detect_positive
            + self.eq(float("-inf")) * detect_negative
        )

    def isnan(self):
        # Pinned mixin/elementwise.py:586-594.
        return self != self

    # --- Activations (C core composed ops) ---

    def relu(self):
        # Pinned mixin/elementwise.py:656-666. Tensor comparison/where keeps
        # the scalar-zero RESHAPE/EXPAND topology and ordered physical roots.
        return (self > 0).where(self, 0)

    def relu6(self):
        # Pinned mixin/elementwise.py:679-689.
        return self.relu() - (self - 6).relu()

    def leaky_relu(self, neg_slope=0.01):
        # Pinned mixin/elementwise.py:726-737.
        return (self < 0).where(neg_slope * self, self)

    def gelu(self):
        # Pinned mixin/elementwise.py:761-776. C applies the exact formula
        # independently to retained/current roots.
        core = _ffi._lib.poly_tensor_gelu(self._ctx, self._tensor)
        return self._make_result_from_core(core, self.shape, [self])

    def quick_gelu(self):
        # Pinned mixin/elementwise.py:751-759 builds the broadcasted scalar
        # formula from the one current Tensor.uop. The C Tensor boundary does
        # the same independently for retained/current roots.
        core = _ffi._lib.poly_tensor_quick_gelu(self._ctx, self._tensor)
        return self._make_result_from_core(core, self.shape, [self])

    def silu(self):
        # Pinned mixin/elementwise.py:790-800.
        return self.swish()

    def swish(self):
        # Pinned mixin/elementwise.py:778-788.
        return self * self.sigmoid()

    def elu(self, alpha=1.0):
        # Pinned mixin/elementwise.py:945-955.
        return self.relu() - alpha * (1 - self.exp()).relu()

    def logaddexp(self, other):
        # Pinned mixin/elementwise.py:403-410.
        a, b, _ = self._broadcasted(other)
        m = a.maximum(b)
        return ((a - m).exp() + (b - m).exp()).log() + m

    def softplus(self, beta=1.0):
        # Pinned mixin/elementwise.py:981-989.
        return (1 / beta) * (self * beta).logaddexp(0.0)

    def mish(self):
        # Pinned mixin/elementwise.py:991-1001.
        return self * self.softplus().tanh()

    def hardtanh(self, min_val=-1, max_val=1):
        # Pinned mixin/elementwise.py:716-724.
        return self.clamp(min_val, max_val)

    def hardswish(self):
        # Pinned mixin/elementwise.py:691-701.
        return self * (self + 3).relu6() * (1 / 6)

    def hardsigmoid(self, alpha=1 / 6, beta=0.5):
        # Pinned mixin/elementwise.py:703-714.
        return (alpha * self + beta).relu() - (alpha * self + beta - 1).relu()

    # --- Softmax ---

    def softmax(self, axis=-1):
        core = _ffi._lib.poly_tensor_softmax(self._ctx, self._tensor, int(axis))
        return self._make_result_from_core(core, self.shape, [self])

    def log_softmax(self, axis=-1):
        core = _ffi._lib.poly_tensor_log_softmax(self._ctx, self._tensor, int(axis))
        return self._make_result_from_core(core, self.shape, [self])

    # --- Movement ops ---

    def reshape(self, shape, *args):
        shape = (shape,) + args
        if shape[0].__class__ in (tuple, list):
            if len(shape) != 1:
                raise ValueError(f"bad arg {shape}")
            shape = tuple(shape[0])
        shape = tuple(s if s is not None else self.shape[i] for i, s in enumerate(shape))
        if (inferred := shape.count(-1)) > 1:
            raise RuntimeError(
                f"only one dimension can be inferred using -1, getting {shape}"
            )
        if inferred:
            shape = tuple(
                -self.numel() // _prod(shape) if s == -1 else s for s in shape
            )
        if self.numel() != _prod(shape):
            raise ValueError(f"size mismatch, can't reshape ({self.shape}) -> ({shape})")
        if shape == self.shape:
            return self
        arr, n = _int64_array(shape)
        core = _ffi._lib.poly_tensor_reshape(self._ctx, self._tensor, arr, n)
        return self._make_result_from_core(core, shape, [self])

    def permute(self, *order):
        if len(order) == 1 and isinstance(order[0], (tuple, list)):
            order = tuple(order[0])
        arr, n = _int64_array(order)
        new_shape = tuple(self.shape[i] for i in order)
        core = _ffi._lib.poly_tensor_permute(self._ctx, self._tensor, arr, n)
        return self._make_result_from_core(core, new_shape, [self])

    def expand(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        shape = _normalize_expand_shape(self.shape, shape)
        if self.shape == shape:
            return self
        if _shape_has_symbolic(shape):
            aligned = (1,) * (len(shape) - self.ndim) + tuple(self.shape)
            if aligned != self.shape:
                if not _shape_all_int(aligned):
                    raise NotImplementedError('symbolic movement reshape is not implemented')
                reshaped = self.reshape(aligned)
            else:
                reshaped = self
            dims = _shape_uop_array(self._ctx, shape)
            core = _ffi._lib.poly_tensor_expand_uop(
                self._ctx, reshaped._tensor, dims, len(shape)
            )
            if not core:
                raise RuntimeError('poly_tensor_expand_uop failed')
            current = self._core_uop_raw(core)
            return reshaped._make_result_from_core(
                core, _shape_from_uop(self._ctx, current), [reshaped]
            )
        aligned = (1,) * (len(shape) - self.ndim) + tuple(self.shape)
        reshaped = self.reshape(aligned)
        dims, n = _int64_array(shape)
        core = _ffi._lib.poly_tensor_expand(reshaped._ctx, reshaped._tensor, dims, n)
        return reshaped._make_result_from_core(core, shape, [reshaped])

    def shrink(self, arg):
        """arg is tuple of (start, end) pairs per dimension."""
        flat, n = _pair_array(arg)
        new_shape = tuple(e - s for s, e in arg)
        core = _ffi._lib.poly_tensor_shrink(self._ctx, self._tensor, flat, n)
        return self._make_result_from_core(core, new_shape, [self])

    def pad(self, arg, mode="constant", value=0.0):
        """Pad using tinygrad-compatible flat or grouped padding."""
        if mode != "constant":
            raise NotImplementedError(f"mode={mode!r} is not supported")
        arg = _normalize_pad_arg(arg, self.ndim)
        return self._pad_constant(arg, value)

    def _pad_constant(self, arg, value):
        arg = tuple((0, 0) if p is None else tuple(p) for p in arg)
        flat, n = _pair_array(arg)
        # Pinned _pad_constant shrinks negative pads before emitting a
        # non-negative PAD (mixin/__init__.py:359-368). Keep that policy in
        # the shared C boundary for both zero and nonzero fill values.
        core = _ffi._lib.poly_tensor_pad_value(
            self._ctx, self._tensor, flat, n, float(value)
        )
        new_shape = tuple(s + b + a for s, (b, a) in zip(self.shape, arg))
        return self._make_result_from_core(core, new_shape, [self])

    def flip(self, axis, *args):
        axes = tuple(axis) if isinstance(axis, (tuple, list)) else (axis,)
        axes = axes + tuple(args)
        axes = tuple(self._resolve_dim(int(a)) for a in axes)
        if len(set(axes)) != len(axes):
            raise RuntimeError(f"dim can appear at most once, getting {axes}")
        arr, n = _int64_array(axes)
        core = _ffi._lib.poly_tensor_flip(self._ctx, self._tensor, arr, n)
        return self._make_result_from_core(core, self.shape, [self])

    def _pool(self, kernel_size, stride=1, dilation=1):
        k = _make_tuple(kernel_size, len(kernel_size) if not isinstance(kernel_size, int) else 2)
        s = _make_tuple(stride, len(k))
        d = _make_tuple(dilation, len(k))
        if len(self.shape) < len(k):
            raise AssertionError(f"can't pool {self.shape} with {k}")
        k_arr, nk = _int64_array(k)
        s_arr, _ = _int64_array(s)
        d_arr, _ = _int64_array(d)
        core = _ffi._lib.poly_tensor_pool(
            self._ctx, self._tensor, k_arr, nk, s_arr, d_arr
        )
        if not core:
            raise RuntimeError(f'poly_pool failed for shape={self.shape}, kernel={k}, stride={s}, dilation={d}')
        current = self._core_uop_raw(core)
        return self._make_result_from_core(
            core, _shape_from_uop(self._ctx, current), [self]
        )

    def transpose(self, dim0=-2, dim1=-1):
        nd = len(self.shape)
        if nd < 2:
            return self
        if dim0 < 0:
            dim0 += nd
        if dim1 < 0:
            dim1 += nd
        order = list(range(nd))
        order[dim0], order[dim1] = order[dim1], order[dim0]
        return self.permute(*order)

    @staticmethod
    def _tri(r, c, diagonal=0, device=None):
        # Pinned mixin/__init__.py:310-311. Polygrad's optional device is
        # wrapper placement metadata only; arange remains a deviceless UOp.
        opts = {} if device is None else {'device': device}
        return (
            Tensor.arange(r, **opts).unsqueeze(-1) + diagonal
        ) <= Tensor.arange(c, **opts)

    def triu(self, diagonal=0):
        r, c = self.shape[-2], self.shape[-1]
        mask = Tensor._tri(r, c, diagonal=diagonal, device=self.device)
        return mask.where(self, self.const_like(0))

    def tril(self, diagonal=0):
        r, c = self.shape[-2], self.shape[-1]
        mask = Tensor._tri(r, c, diagonal=diagonal + 1, device=self.device)
        return mask.where(self.const_like(0), self)

    def squeeze(self, dim=None):
        if dim is not None:
            if dim < 0:
                dim += len(self.shape)
            if self.shape[dim] != 1:
                return self
            new_shape = tuple(s for i, s in enumerate(self.shape) if i != dim)
            return self.reshape(new_shape)
        new_shape = tuple(s for s in self.shape if s != 1)
        if new_shape == self.shape:
            return self
        return self.reshape(new_shape)

    def unsqueeze(self, dim):
        if dim < 0:
            dim += len(self.shape) + 1
        new_shape = list(self.shape)
        new_shape.insert(dim, 1)
        return self.reshape(tuple(new_shape))

    def flatten(self, start_dim=0, end_dim=-1):
        if end_dim < 0:
            end_dim += len(self.shape)
        new_shape = list(self.shape[:start_dim])
        flat_dim = 1
        for i in range(start_dim, end_dim + 1):
            flat_dim *= self.shape[i]
        new_shape.append(flat_dim)
        new_shape.extend(self.shape[end_dim + 1:])
        return self.reshape(tuple(new_shape))

    def unflatten(self, dim, sizes):
        if dim < 0:
            dim += len(self.shape)
        new_shape = list(self.shape[:dim]) + list(sizes) + list(self.shape[dim + 1:])
        return self.reshape(tuple(new_shape))

    def view(self, *shape):
        return self.reshape(*shape)

    def repeat(self, *repeats):
        if len(repeats) == 1 and isinstance(repeats[0], (tuple, list)):
            repeats = tuple(repeats[0])
        # Pad shape if needed
        nd = max(len(self.shape), len(repeats))
        shape = (1,) * (nd - len(self.shape)) + self.shape
        repeats = (1,) * (nd - len(repeats)) + repeats
        # Interleave: reshape to (1, s0, 1, s1, ...), expand to (r0, s0, r1, s1, ...), flatten pairs
        new_shape = []
        exp_shape = []
        for s, r in zip(shape, repeats):
            new_shape.extend([1, s])
            exp_shape.extend([r, s])
        result = self.reshape(tuple(new_shape)).expand(tuple(exp_shape))
        final_shape = tuple(s * r for s, r in zip(shape, repeats))
        return result.reshape(final_shape)

    def roll(self, shifts, dims=None):
        if dims is None:
            return self.flatten().roll(shifts, 0).reshape(self.shape)

        dims = (int(dims),) if isinstance(dims, (int, np.integer)) else tuple(dims)
        shifts = (int(shifts),) if isinstance(shifts, (int, np.integer)) else tuple(shifts)
        dims = tuple(self._resolve_dim(d) for d in dims)
        if len(dims) != len(shifts):
            raise RuntimeError(f"len(dims)={len(dims)} != len(shifts)={len(shifts)}")

        shrink_arg = [(0, s) for s in self.shape]
        for d, s in zip(dims, shifts):
            size = self.shape[d]
            delta = size - int(s) % size
            shrink_arg[d] = (delta, delta + size)
        repeats = tuple(2 if i in dims else 1 for i in range(self.ndim))
        return self.repeat(*repeats).shrink(tuple(shrink_arg))

    # --- Reduction ops (C core) ---

    def sum(self, axis=None, keepdim=False):
        if axis is None:
            axis = tuple(range(self.ndim))
        elif isinstance(axis, int):
            axis = (axis,)
        axis = tuple(self._resolve_dim(int(a)) for a in axis)
        arr, n = _int64_array(axis)
        core = _ffi._lib.poly_tensor_sum(
            self._ctx, self._tensor, arr, n, bool(keepdim)
        )
        new_shape = (
            tuple(1 if i in axis else s for i, s in enumerate(self.shape))
            if keepdim else
            tuple(s for i, s in enumerate(self.shape) if i not in axis)
        )
        return self._make_result_from_core(core, new_shape, [self])

    def max(self, axis=None, keepdim=False):
        # Pinned ReduceMixin._reduce emits one REDUCE over the complete
        # normalized axis tuple (mixin/reduce.py:12-17, uop/ops.py:567-569).
        axes = tuple(range(self.ndim)) if axis is None else (
            (self._resolve_dim(int(axis)),) if isinstance(axis, int) else
            tuple(self._resolve_dim(int(a)) for a in axis)
        )
        arr, n = _int64_array(axes)
        core = _ffi._lib.poly_tensor_max(
            self._ctx, self._tensor, arr, n, bool(keepdim)
        )
        new_shape = (
            tuple(1 if i in axes else s for i, s in enumerate(self.shape))
            if keepdim else
            tuple(s for i, s in enumerate(self.shape) if i not in axes)
        )
        return self._make_result_from_core(core, new_shape, [self])

    def argmax(self, axis=None, keepdim=False):
        if axis is None:
            return self.flatten().argmax(0)
        axis = self._resolve_dim(int(axis))
        core = _ffi._lib.poly_tensor_argmax(
            self._ctx, self._tensor, axis, bool(keepdim)
        )
        shape = (
            tuple(1 if i == axis else s for i, s in enumerate(self.shape))
            if keepdim else
            tuple(s for i, s in enumerate(self.shape) if i != axis)
        )
        result = self._make_result_from_core(core, shape, [self])
        result.requires_grad = False
        return result

    def sort(self, dim=-1, descending=False):
        dim = self._resolve_dim(int(dim))
        values = _ffi._ptr()
        indices = _ffi._ptr()
        rc = _ffi._lib.poly_tensor_sort(
            self._ctx, self._tensor, dim, int(bool(descending)),
            ctypes.byref(values), ctypes.byref(indices)
        )
        if rc != 0 or not values or not indices:
            raise RuntimeError('poly_tensor_sort failed')
        vals = self._make_result_from_core(values, self.shape, [self])
        idx = self._make_result_from_core(indices, self.shape, [self])
        idx.requires_grad = False
        return vals, idx

    def argsort(self, dim=-1, descending=False):
        return self.sort(dim, descending)[1]

    def topk(self, k, dim=-1, largest=True, sorted_=True):
        if not sorted_:
            raise NotImplementedError('topk with sorted_=False is not supported')
        dim = self._resolve_dim(int(dim))
        if int(k) > self.shape[dim]:
            raise ValueError(f'selected index k={int(k)} is out of range')
        values = _ffi._ptr()
        indices = _ffi._ptr()
        rc = _ffi._lib.poly_tensor_topk(
            self._ctx, self._tensor, int(k), dim, int(bool(largest)), int(bool(sorted_)),
            ctypes.byref(values), ctypes.byref(indices)
        )
        if rc != 0 or not values or not indices:
            raise RuntimeError('poly_tensor_topk failed')
        out_shape = list(self.shape)
        out_shape[dim] = int(k)
        vals = self._make_result_from_core(values, tuple(out_shape), [self])
        idx = self._make_result_from_core(indices, tuple(out_shape), [self])
        idx.requires_grad = False
        return vals, idx

    def min(self, axis=None, keepdim=False):
        return (-self).max(axis=axis, keepdim=keepdim).__neg__()

    def mean(self, axis=None, keepdim=False):
        axes = tuple(range(self.ndim)) if axis is None else (
            (self._resolve_dim(int(axis)),) if isinstance(axis, int) else
            tuple(self._resolve_dim(int(a)) for a in axis)
        )
        # Pinned mixin/__init__.py:581-599 casts before sum, divides through
        # Tensor reciprocal/multiply, then casts to the public output dtype.
        numerator = self.cast(_sum_acc_dtype(self.dtype)).sum(
            axis=axes, keepdim=keepdim
        )
        denominator = _prod(self.shape[a] for a in axes)
        output_dtype = self.dtype if dtypes.is_float(self.dtype) else dtypes.float32
        return numerator.div(denominator).cast(output_dtype)

    def var(self, axis=None, keepdim=False, correction=1):
        # Pinned Tensor.var is this exact lazy Tensor expression
        # (mixin/__init__.py:608-635), including tuple axes and RELU on the
        # denominator. Every constituent operation is C-owned.
        squares = (self - self.mean(axis=axis, keepdim=True)).square()
        reduced_shape = squares.sum(axis=axis, keepdim=True).shape
        n = _prod(
            si for si, so in zip(self.shape, reduced_shape) if int(si) != int(so)
        )
        reduced = squares.sum(axis=axis, keepdim=keepdim)
        denominator = reduced.const_like(n) - correction
        return reduced.div(denominator.relu())

    def std(self, axis=None, keepdim=False, correction=1):
        return self.var(axis=axis, keepdim=keepdim, correction=correction).sqrt()

    def gather(self, dim, index):
        if not isinstance(index, Tensor):
            index = Tensor(index, dtype='int32', device=self._device)
        if index.device != self.device:
            raise RuntimeError(
                f"expected index and self on the same device, index.device={index.device}, self.device={self.device}"
            )
        if index.ndim != self.ndim:
            raise RuntimeError(f"self.ndim must equal index.ndim, self.ndim={self.ndim}, index.ndim={index.ndim}")
        dim = self._resolve_dim(int(dim))
        for d, (s, i) in enumerate(zip(self.shape, index.shape)):
            if d != dim and s < i:
                raise AssertionError('requires self.shape[d] >= index.shape[d] for all d != dim')

        core = _ffi._lib.poly_tensor_gather_dim(
            self._ctx, self._tensor, dim, index._tensor
        )
        if not core:
            raise RuntimeError('poly_tensor_gather_dim failed')
        current = self._core_uop_raw(core)
        return self._make_result_from_core(
            core, _shape_from_uop(self._ctx, current), [self, index]
        )

    def take_along_axis(self, index, axis):
        return self.gather(axis, index)

    def one_hot(self, num_classes):
        core = _ffi._lib.poly_tensor_one_hot(
            self._ctx, self._tensor, int(num_classes)
        )
        if not core:
            raise RuntimeError('poly_tensor_one_hot failed')
        current = self._core_uop_raw(core)
        return self._make_result_from_core(
            core, _shape_from_uop(self._ctx, current), [self]
        )

    def _one_hot_along_dim(self, num_classes, dim=-1):
        # Pinned tinygrad compares the integer index directly with a
        # right-aligned arange (mixin/__init__.py:1086-1091).
        if not dtypes.is_int(to_dtype(self.dtype)):
            raise RuntimeError(
                f"_one_hot_along_dim expects int index tensor, getting {self.dtype}"
            )
        dim = self._resolve_dim(int(dim))
        offset = self.ndim - dim - 1
        dtype = dtypes.int64 if int(num_classes) > np.iinfo(np.int32).max else dtypes.int32
        classes = Tensor.arange(
            int(num_classes), dtype=dtype, _ctx=self._ctx
        ).reshape((int(num_classes),) + (1,) * offset)
        return self.eq(classes)

    def _pre_scatter_validate(self, dim, index, src):
        if not isinstance(index, Tensor):
            index = Tensor(index, dtype='int32', device=self._device)
        if not isinstance(src, Tensor):
            src = Tensor.full(index.shape, _py_scalar(src), dtype=self._dtype_str, device=self._device)
        if index.device != self.device:
            raise RuntimeError(
                f"expected index and self on the same device, index.device={index.device}, self.device={self.device}"
            )
        if src.device != self.device:
            raise RuntimeError(
                f"expected src and self on the same device, src.device={src.device}, self.device={self.device}"
            )
        dim = self._resolve_dim(int(dim))
        if index.ndim != self.ndim or src.ndim != self.ndim:
            raise RuntimeError(
                f"index.ndim, self.ndim and src.ndim must all match, index.ndim={index.ndim}, "
                f"self.ndim={self.ndim}, src.ndim={src.ndim}"
            )
        for d, (self_d, index_d, src_d) in enumerate(zip(self.shape, index.shape, src.shape)):
            if ((d != dim and self_d < index_d) or src_d < index_d):
                raise AssertionError(
                    'requires self.shape[d] >= index.shape[d] for all d != dim and '
                    'src.shape[d] >= index.shape[d] for all d'
                )
        if self.dtype != src.dtype:
            raise RuntimeError(f"expected self and src to have the same dtype, self.dtype={self.dtype}, src.dtype={src.dtype}")
        return dim, index, src

    def scatter_reduce(self, dim, index, src, reduce, include_self=True):
        if reduce not in {'sum', 'prod', 'mean', 'amax', 'amin'}:
            raise RuntimeError(f"reduce={reduce!r} must be one of 'sum', 'prod', 'mean', 'amax', 'amin'")
        if not isinstance(src, Tensor):
            src = Tensor(src, dtype=self._dtype_str, device=self._device)
        dim, index, src = self._pre_scatter_validate(dim, index, src)
        core = _ffi._lib.poly_tensor_scatter_reduce(
            self._ctx, self._tensor, dim, index._tensor, src._tensor,
            reduce.encode('utf-8'), int(bool(include_self))
        )
        if not core:
            raise RuntimeError('poly_tensor_scatter_reduce failed')
        current = self._core_uop_raw(core)
        return self._make_result_from_core(
            core, _shape_from_uop(self._ctx, current), [self, index, src]
        )

    def scatter(self, dim, index, src, reduce=None):
        if reduce not in {None, 'add', 'multiply'}:
            raise TypeError(f"reduce={reduce!r} must be one of None, 'multiply', or 'add'")
        src_is_tensor = isinstance(src, Tensor)
        if not src_is_tensor:
            src = Tensor.full(index.shape if isinstance(index, Tensor) else np.asarray(index).shape,
                              _py_scalar(src), dtype=self._dtype_str, device=self._device)
        elif reduce is not None:
            raise TypeError('non-scalar src is not supported with reduce arg. use scatter_reduce')
        dim, index, src = self._pre_scatter_validate(dim, index, src)
        reduce_arg = b'' if reduce is None else reduce.encode('utf-8')
        core = _ffi._lib.poly_tensor_scatter(
            self._ctx, self._tensor, dim, index._tensor, src._tensor, reduce_arg
        )
        if not core:
            raise RuntimeError('poly_tensor_scatter failed')
        current = self._core_uop_raw(core)
        return self._make_result_from_core(
            core, _shape_from_uop(self._ctx, current), [self, index, src]
        )

    # --- Matmul (C core dot) ---

    def dot(self, w):
        if not isinstance(w, Tensor):
            raise TypeError(f'Expected Tensor, got {type(w)}')
        core = _ffi._lib.poly_tensor_dot(self._ctx, self._tensor, w._tensor)
        if not core:
            raise ValueError(f'cannot dot {self.shape} and {w.shape}')
        current = self._core_uop_raw(core)
        return self._make_result_from_core(
            core, _shape_from_uop(self._ctx, current), [self, w]
        )

    def matmul(self, other):
        return self.dot(other)

    def qr(self, mode='complete'):
        mode_id = {'complete': 0, 'reduced': 1, 'r': 2}.get(mode)
        if mode_id is None:
            raise ValueError("qr mode must be 'complete', 'reduced', or 'r'")
        q = _ffi._ptr()
        r = _ffi._ptr()
        rc = _ffi._lib.poly_tensor_qr_ex(
            self._ctx, self._tensor, mode_id, ctypes.byref(q), ctypes.byref(r)
        )
        if rc != 0 or not r or (mode_id != 2 and not q):
            raise RuntimeError('poly_tensor_qr_ex failed')
        if mode_id == 2:
            r_current = self._core_uop_raw(r)
            return self._make_result_from_core(
                r, _shape_from_uop(self._ctx, r_current), [self]
            )
        q_current = self._core_uop_raw(q)
        r_current = self._core_uop_raw(r)
        return (
            self._make_result_from_core(
                q, _shape_from_uop(self._ctx, q_current), [self]
            ),
            self._make_result_from_core(
                r, _shape_from_uop(self._ctx, r_current), [self]
            ),
        )

    def triangular_solve(self, b, upper=False, transpose_a=False, unit_diagonal=False):
        if not isinstance(b, Tensor):
            b = self._ensure_tensor(b)
        core = _ffi._lib.poly_tensor_triangular_solve(
            self._ctx, self._tensor, b._tensor,
            int(bool(upper)), int(bool(transpose_a)), int(bool(unit_diagonal))
        )
        if not core:
            raise ValueError(
                f'cannot triangular_solve A.shape={self.shape} and b.shape={b.shape}'
            )
        current = self._core_uop_raw(core)
        return self._make_result_from_core(
            core, _shape_from_uop(self._ctx, current), [self, b]
        )

    def solve_triangular(self, b, upper=False, transpose_a=False, unit_diagonal=False):
        return self.triangular_solve(b, upper, transpose_a, unit_diagonal)

    def cholesky(self, upper=False):
        core = _ffi._lib.poly_tensor_cholesky(
            self._ctx, self._tensor, int(bool(upper))
        )
        if not core:
            raise ValueError(f'cannot cholesky shape={self.shape}')
        current = self._core_uop_raw(core)
        return self._make_result_from_core(
            core, _shape_from_uop(self._ctx, current), [self]
        )

    def cholesky_solve(self, b, upper=False):
        if not isinstance(b, Tensor):
            b = self._ensure_tensor(b)
        core = _ffi._lib.poly_tensor_cholesky_solve(
            self._ctx, self._tensor, b._tensor, int(bool(upper))
        )
        if not core:
            raise ValueError(
                f'cannot cholesky_solve factor.shape={self.shape} and b.shape={b.shape}'
            )
        current = self._core_uop_raw(core)
        return self._make_result_from_core(
            core, _shape_from_uop(self._ctx, current), [self, b]
        )

    def solve(self, b):
        if not isinstance(b, Tensor):
            b = self._ensure_tensor(b)
        core = _ffi._lib.poly_tensor_solve(self._ctx, self._tensor, b._tensor)
        if not core:
            raise ValueError(f'cannot solve A.shape={self.shape} and b.shape={b.shape}')
        current = self._core_uop_raw(core)
        return self._make_result_from_core(
            core, _shape_from_uop(self._ctx, current), [self, b]
        )

    def lstsq(self, b):
        if not isinstance(b, Tensor):
            b = self._ensure_tensor(b)
        core = _ffi._lib.poly_tensor_lstsq(self._ctx, self._tensor, b._tensor)
        if not core:
            raise ValueError(f'cannot lstsq A.shape={self.shape} and b.shape={b.shape}')
        current = self._core_uop_raw(core)
        return self._make_result_from_core(
            core, _shape_from_uop(self._ctx, current), [self, b]
        )

    def __matmul__(self, other):
        return self.dot(other)

    def __rmatmul__(self, other):
        other = self._ensure_tensor(other)
        return other.dot(self)

    def linear(self, weight, bias=None):
        """linear(x, w, bias) = x @ w.T + bias"""
        result = self.dot(weight.transpose(-1, -2))
        if bias is not None:
            result = result + bias
        return result

    def sequential(self, ll):
        return functools.reduce(lambda x, f: f(x), ll, self)

    def max_pool2d(self, kernel_size=(2, 2), stride=None, dilation=1, padding=0,
                   ceil_mode=False, return_indices=False):
        if ceil_mode:
            raise NotImplementedError('max_pool2d ceil_mode is not implemented in Polygrad yet')
        if return_indices:
            raise NotImplementedError('max_pool2d return_indices is not implemented in Polygrad yet')
        k = _make_tuple(kernel_size, 2)
        s = k if stride is None else _make_tuple(stride, len(k))
        d = _make_tuple(dilation, len(k))
        pads = _resolve_pool_pads(padding, len(k))
        k_arr, nk = _int64_array(k)
        s_arr, _ = _int64_array(s)
        d_arr, _ = _int64_array(d)
        p_arr, npad = _int64_array(pads)
        core = _ffi._lib.poly_tensor_max_pool2d(
            self._ctx, self._tensor, k_arr, nk, s_arr, d_arr, p_arr, npad
        )
        if not core:
            raise RuntimeError('poly_max_pool2d failed')
        current = self._core_uop_raw(core)
        return self._make_result_from_core(
            core, _shape_from_uop(self._ctx, current), [self]
        )

    def conv2d(self, weight, bias=None, groups=1, stride=1, dilation=1, padding=0, dtype=None):
        if not isinstance(weight, Tensor):
            weight = self._ensure_tensor(weight)
        if bias is not None and not isinstance(bias, Tensor):
            bias = self._ensure_tensor(bias)

        hw = weight.shape[2:]
        s = _make_tuple(stride, len(hw))
        d = _make_tuple(dilation, len(hw))
        pads = _resolve_pool_pads(padding, len(hw))
        s_arr, _ = _int64_array(s)
        d_arr, _ = _int64_array(d)
        p_arr, npad = _int64_array(pads)
        core = _ffi._lib.poly_tensor_conv2d(
            self._ctx, self._tensor, weight._tensor,
            bias._tensor if bias is not None else None,
            int(groups), s_arr, d_arr, p_arr, npad,
        )
        if not core:
            raise RuntimeError('poly_conv2d failed')
        inputs = [self, weight] + ([bias] if bias is not None else [])
        current = self._core_uop_raw(core)
        return self._make_result_from_core(
            core, _shape_from_uop(self._ctx, current), inputs
        )

    def batchnorm(self, weight, bias, mean, invstd, axis=1):
        axis_ = tuple(axis) if isinstance(axis, (tuple, list)) else (axis,)
        axis_ = tuple(self._resolve_dim(int(a)) for a in axis_)
        axes, n_axes = _int64_array(axis_)
        core = _ffi._lib.poly_tensor_batchnorm(
            self._ctx, self._tensor,
            weight._tensor if weight is not None else None,
            bias._tensor if bias is not None else None,
            mean._tensor, invstd._tensor, axes, n_axes,
        )
        if not core:
            raise RuntimeError('poly_batchnorm failed')
        inputs = [self, mean, invstd]
        if weight is not None:
            inputs.append(weight)
        if bias is not None:
            inputs.append(bias)
        current = self._core_uop_raw(core)
        return self._make_result_from_core(
            core, _shape_from_uop(self._ctx, current), inputs
        )

    # --- Loss functions ---

    def cross_entropy(
        self, target, reduction="mean", label_smoothing=0.0, *, axis=None
    ):
        """Cross-entropy loss with tinygrad's API and graph spelling."""
        assert 0.0 <= label_smoothing <= 1.0, (
            "label_smoothing must be in [0.0, 1.0]"
        )
        if not isinstance(target, Tensor):
            target = Tensor(target, device=self._device)
        classes_dim = (
            self._resolve_dim(int(axis))
            if axis is not None
            else (0 if self.ndim == 1 else 1)
        )
        if self.shape != target.shape:
            expected = self.shape[:classes_dim] + self.shape[classes_dim + 1:]
            if expected != target.shape:
                raise RuntimeError(
                    f"shape mismatch: self.shape={self.shape}, target.shape={target.shape}"
                )
            target = target.unsqueeze(classes_dim)._one_hot_along_dim(
                num_classes=self.shape[classes_dim], dim=classes_dim
            )
        target = (
            (1 - label_smoothing) * target
            + label_smoothing / int(target.shape[classes_dim])
        )
        reduced = self.log_softmax(classes_dim).mul(target).sum(classes_dim)
        if reduction == "none":
            return -reduced
        if reduction == "sum":
            return -reduced.sum()
        if reduction == "mean":
            return -reduced.mean()
        raise ValueError(
            f"reduction={reduction!r} must be one of ('none', 'sum', 'mean')"
        )

    def sparse_categorical_crossentropy(self, target, axis=None):
        """tinygrad name for class-index cross entropy."""
        return self.cross_entropy(target, axis=axis)

    def binary_crossentropy(self, target):
        return -(target * self.log() + (1.0 - target) * (1.0 - self).log()).mean()

    def layernorm(self, axis=-1, eps=1e-5):
        # Layernorm is graph construction, not a materialization boundary.
        # Explicit realize calls here severed gradients in cases where tinygrad
        # keeps the full UOp DAG lazy until the user calls realize/backward.
        m = self.mean(axis=axis, keepdim=True)
        v = self.var(axis=axis, keepdim=True, correction=0)
        return (self - m) / (v + eps).sqrt()

    # --- Indexing ---

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            idx = (idx,)

        def _apply_uop_shrink(tensor, starts, sizes):
            start_arr = (_ffi._ptr * len(starts))(*starts)
            size_arr = (_ffi._ptr * len(sizes))(*sizes)
            core = _ffi._lib.poly_tensor_shrink_uop(
                tensor._ctx, tensor._tensor, start_arr, size_arr, len(starts)
            )
            if not core:
                raise RuntimeError('poly_tensor_shrink_uop failed')
            current = tensor._core_uop_raw(core)
            return tensor._make_result_from_core(
                core, _shape_from_uop(tensor._ctx, current), [tensor]
            )

        # Expand Ellipsis
        n_ellipsis = sum(1 for i in idx if i is Ellipsis)
        if n_ellipsis > 1:
            raise IndexError('Only one Ellipsis allowed')
        if n_ellipsis == 1:
            eidx = idx.index(Ellipsis)
            n_none = sum(1 for i in idx if i is None)
            n_real = len(idx) - 1 - n_none
            n_fill = len(self.shape) - n_real
            idx = idx[:eidx] + (slice(None),) * n_fill + idx[eidx + 1:]

        result = self
        dim = 0
        for i in idx:
            if i is None:
                result = result.unsqueeze(dim)
                dim += 1
            elif isinstance(i, int):
                if i < 0:
                    i += result.shape[dim]
                result = result.shrink(
                    tuple((i, i + 1) if d == dim else (0, s)
                          for d, s in enumerate(result.shape)))
                result = result.squeeze(dim)
            elif isinstance(i, Tensor):
                if not dtypes.is_int(to_dtype(i.dtype)):
                    raise IndexError(f"index dtype {i.dtype} is not supported")
                if i.device != result.device:
                    raise RuntimeError(
                        f"expected index and self on the same device, index.device={i.device}, self.device={result.device}"
                    )
                core = _ffi._lib.poly_tensor_index_select(
                    result._ctx, result._tensor, dim, i._tensor
                )
                if not core:
                    raise RuntimeError('poly_tensor_index_select failed')
                current = result._core_uop_raw(core)
                result = result._make_result_from_core(
                    core, _shape_from_uop(result._ctx, current), [result, i]
                )
                dim += len(i.shape)
            elif isinstance(i, slice):
                size = result.shape[dim]
                start_obj = 0 if i.start is None else i.start
                stop_obj = size if i.stop is None else i.stop
                step = 1 if i.step is None else i.step
                if _is_symbolic_bound(start_obj) or _is_symbolic_bound(stop_obj) or _is_symbolic_bound(step):
                    if step == 0:
                        raise ValueError('slice step cannot be zero')
                    if step != 1:
                        raise TypeError(f'slice {i!r} is not supported for symbolic shape')
                    start_u = _bound_to_uop(result._ctx, start_obj)
                    stop_u = _bound_to_uop(result._ctx, stop_obj)
                    size_u = _symbolic_slice_size_uop(result._ctx, start_obj, stop_obj, start_u, stop_u)
                    start_val = _bound_to_int(result._ctx, start_obj)
                    stop_val = _bound_to_int(result._ctx, stop_obj)
                    try:
                        size_val = _bound_to_int(result._ctx, size)
                    except TypeError:
                        size_val = None
                    if start_val < 0 or stop_val < start_val or (size_val is not None and stop_val > size_val):
                        raise IndexError(f'symbolic slice {i!r} is out of bounds for size {size}')
                    starts = []
                    sizes = []
                    for d, s in enumerate(result.shape):
                        if d == dim:
                            starts.append(start_u.raw)
                            sizes.append(size_u.raw)
                        else:
                            starts.append(UOp.const(result._ctx, 0).raw)
                            size_raw = _symbolic_dim_raw(s)
                            sizes.append(size_raw if size_raw is not None else UOp.const(result._ctx, int(s)).raw)
                    result = _apply_uop_shrink(result, starts, sizes)
                    dim += 1
                    continue
                if i.start is None and i.stop is None and (i.step is None or i.step == 1):
                    dim += 1
                    continue
                start, stop, step = i.indices(_slice_indices_size(result._ctx, result._graph_uop, dim, size))
                if step == 0:
                    raise ValueError('slice step cannot be zero')
                # Normalize boundary and stride (matching tinygrad _getitem)
                boundary = [start, stop]
                stride = step
                if stride * (boundary[1] - boundary[0]) < 0:
                    boundary = [0, 0]
                elif stride < 0:
                    boundary = [boundary[1] + 1, boundary[0] + 1]
                new_size = -(-abs(boundary[1] - boundary[0]) // abs(stride))  # ceildiv
                # shrink to boundary
                if _shape_all_int(result.shape):
                    result = result.shrink(
                        tuple(tuple(boundary) if d == dim else (0, s)
                              for d, s in enumerate(result.shape)))
                else:
                    starts = []
                    sizes = []
                    symbolic_out = False
                    for d, s in enumerate(result.shape):
                        starts.append(UOp.const(result._ctx, boundary[0] if d == dim else 0).raw)
                        if d == dim:
                            sizes.append(UOp.const(result._ctx, boundary[1] - boundary[0]).raw)
                        else:
                            size_raw = _symbolic_dim_raw(s)
                            symbolic_out = symbolic_out or size_raw is not None
                            sizes.append(size_raw if size_raw is not None else UOp.const(result._ctx, int(s)).raw)
                    if abs(stride) != 1 and symbolic_out:
                        raise RuntimeError('symbolic shape not supported')
                    result = _apply_uop_shrink(result, starts, sizes)
                # flip if negative stride
                if stride < 0:
                    result = result.flip(dim)
                abs_stride = abs(stride)
                # apply stride via pad+reshape+shrink+reshape
                if abs_stride != 1:
                    if not _shape_all_int(result.shape):
                        raise RuntimeError('symbolic shape not supported')
                    sh = list(result.shape)
                    # pad to multiple of stride
                    rem = sh[dim] % abs_stride
                    if rem != 0:
                        pad_amt = abs_stride - rem
                        padding = tuple((0, pad_amt) if d == dim else (0, 0)
                                        for d in range(len(sh)))
                        result = result.pad(padding)
                        sh[dim] += pad_amt
                    # reshape: split dim into (n_groups, stride)
                    new_sh = sh[:dim] + [sh[dim] // abs_stride, abs_stride] + sh[dim+1:]
                    result = result.reshape(*new_sh)
                    # shrink to first element of each stride group
                    result = result.shrink(
                        tuple((0, 1) if d == dim + 1 else (0, s)
                              for d, s in enumerate(result.shape)))
                    # reshape back, collapsing the stride dim
                    final_sh = list(result.shape)
                    final_sh = final_sh[:dim] + [final_sh[dim]] + final_sh[dim+2:]
                    result = result.reshape(*final_sh)
                dim += 1
            else:
                raise IndexError(f'Unsupported index type: {type(i)}')
        return result

    # --- Einsum (C core) ---

    @staticmethod
    def einsum(formula, *operands):
        """Einstein summation convention. E.g. Tensor.einsum('ij,jk->ik', a, b)."""
        if len(operands) == 1 and isinstance(operands[0], (list, tuple)):
            operands = tuple(operands[0])
        if not operands:
            raise ValueError('einsum requires at least one operand')
        if any(not isinstance(t, Tensor) for t in operands):
            raise TypeError('einsum operands must be Tensors')
        ctx = operands[0]._ctx
        if any(t._ctx != ctx for t in operands):
            raise ValueError('einsum operands must belong to the same Polygrad context')
        n = len(operands)

        tensor_arr = (_ffi._ptr * n)(*[t._tensor for t in operands])
        core = _ffi._lib.poly_tensor_einsum(
            ctx, formula.encode('utf-8'),
            tensor_arr, n)
        if not core:
            raise ValueError(f'poly_einsum failed for formula: {formula}')
        current = operands[0]._core_uop_raw(core)
        return operands[0]._make_result_from_core(
            core, _shape_from_uop(ctx, current), list(operands)
        )

    # --- Rearrange (C core, einops-style) ---

    def rearrange(self, formula, **kwargs):
        """Einops-style rearrange. E.g. t.rearrange('b c h w -> b (c h) w')."""
        names = list(kwargs.keys())
        values = [kwargs[k] for k in names]
        n = len(names)

        axis_names = ' '.join(names).encode('utf-8') if names else None
        axis_values = (ctypes.c_int64 * n)(*values) if n > 0 else None

        core = _ffi._lib.poly_tensor_rearrange(
            self._ctx, formula.encode('utf-8'),
            self._tensor,
            axis_names, axis_values, n)
        if not core:
            raise ValueError(f'poly_rearrange failed for formula: {formula}')
        current = self._core_uop_raw(core)
        return self._make_result_from_core(
            core, _shape_from_uop(self._ctx, current), [self]
        )

    # --- Static constructors ---

    @staticmethod
    def _resolve_np_dtype(kwargs):
        dt = kwargs.get('dtype', 'float32')
        return np.float64 if dt == 'float64' else np.float32

    @staticmethod
    def zeros(*shape, **kwargs):
        return Tensor.full(_shape_tuple(*shape), 0.0, **kwargs)

    @staticmethod
    def ones(*shape, **kwargs):
        return Tensor.full(_shape_tuple(*shape), 1.0, **kwargs)

    @staticmethod
    def full(shape, fill_value, **kwargs):
        buffer = bool(kwargs.pop('buffer', True))
        ctx, dev, requires_grad = _creation_meta(kwargs)
        shape = (shape,) if isinstance(shape, int) else _shape_tuple(shape)
        fill_value = _py_scalar(fill_value)
        inferred = kwargs.get('dtype', dtypes.from_py(fill_value))
        dtype_name = _dtype_name(inferred, default='float32')
        dtype_id = _dtype_id(dtype_name)
        dims, ndim, _ = _shape_arg(shape)

        dt = to_dtype(dtype_name)
        if dtypes.is_float(dt):
            tensor = _ffi._lib.poly_tensor_full_float_by_id(
                ctx, dims, ndim, float(fill_value), dtype_id, _device_id(dev)
            )
        else:
            tensor = _ffi._lib.poly_tensor_full_int_by_id(
                ctx, dims, ndim, _require_i64(fill_value, 'fill_value'), dtype_id, _device_id(dev)
            )
        value = _created_tensor(
            ctx, tensor, dtype_name, dev, requires_grad, 'poly_tensor_full_*_by_id'
        )
        return value.clone(device=dev) if buffer else value

    @staticmethod
    def arange(start, stop=None, step=1, **kwargs):
        ctx, dev, requires_grad = _creation_meta(kwargs)
        start = _py_scalar(start)
        stop = _py_scalar(stop) if stop is not None else None
        step = _py_scalar(step)
        if stop is None:
            stop, start = start, 0

        inferred = kwargs.get(
            'dtype',
            dtypes.default_float if any(isinstance(x, float) for x in (start, stop, step)) else dtypes.default_int,
        )
        dtype_name = _dtype_name(inferred, default='float32')
        dtype_id = _dtype_id(dtype_name)
        dt = to_dtype(dtype_name)

        if dtypes.is_float(dt):
            tensor = _ffi._lib.poly_tensor_arange_float_by_id(
                ctx, float(start), float(stop), float(step), dtype_id, _device_id(dev)
            )
        else:
            tensor = _ffi._lib.poly_tensor_arange_int_by_id(
                ctx,
                _require_i64(start, 'start'),
                _require_i64(stop, 'stop'),
                _require_i64(step, 'step'),
                dtype_id,
                _device_id(dev),
            )
        return _created_tensor(
            ctx, tensor, dtype_name, dev, requires_grad, 'poly_tensor_arange_*_by_id'
        )

    @staticmethod
    def rand(*shape, **kwargs):
        _ctx, dev, requires_grad = _creation_meta(kwargs)
        shape = _shape_tuple(*shape)
        dtype_name = _dtype_name(kwargs.get('dtype', dtypes.default_float), default='float32')
        dt = to_dtype(dtype_name)
        if not dtypes.is_float(dt):
            raise ValueError(f'rand only supports float dtypes, got {dt}')
        if any(not isinstance(s, int) or s < 0 for s in shape):
            raise ValueError(f'invalid input shape={shape}')
        num = -(-(_prod(shape) * dt.itemsize) // 4)
        key, counter = Tensor._next_counter(dev, num)
        out = Tensor._rand(key, counter, shape, dt, contiguous=kwargs.get('contiguous', True))
        if requires_grad:
            out.requires_grad = True
        return out

    @staticmethod
    def randn(*shape, **kwargs):
        _ctx, dev, requires_grad = _creation_meta(kwargs)
        shape = _shape_tuple(*shape)
        dtype_name = _dtype_name(kwargs.get('dtype', dtypes.default_float), default='float32')
        dt = to_dtype(dtype_name)
        if not dtypes.is_float(dt):
            raise ValueError(f'randn only supports float dtypes, got {dt}')
        src = Tensor.rand(2, *shape, device=dev, dtype=dtypes.float32)
        out = src[0].mul(2 * math.pi).cos().mul((1 - src[1]).log().mul(-2).sqrt()).cast(dt)
        if requires_grad:
            out.requires_grad = True
        return out

    @staticmethod
    def uniform(*shape, low=0.0, high=1.0, **kwargs):
        """Create uniform values using pinned tinygrad's lazy rand/scale/add graph."""
        shape = _shape_tuple(*shape)
        if any(not isinstance(s, int) or s < 0 for s in shape):
            raise ValueError(f'invalid input shape={shape}')
        if low >= high:
            raise ValueError(f'Tensor.uniform requires low < high, got low={low}, high={high}')
        dtype = kwargs.get('dtype', dtypes.default_float)
        return ((high - low) * Tensor.rand(*shape, **kwargs)).cast(dtype) + low

    @staticmethod
    def kaiming_uniform(*shape, **kwargs):
        """tinygrad-compatible Kaiming uniform initializer."""
        shape = _shape_tuple(*shape)
        if not shape:
            raise ValueError("kaiming_uniform requires a non-scalar shape")
        fan_in = shape[0] if len(shape) == 1 else math.prod(shape[1:])
        bound = math.sqrt(6.0 / fan_in)
        return Tensor.rand(*shape, **kwargs) * (2.0 * bound) - bound

    @staticmethod
    def randint(*shape, low=0, high=10, dtype=dtypes.int32, **kwargs):
        if 'shape' in kwargs:
            kw_shape = kwargs.pop('shape')
            if len(shape) == 2 and low == 0 and high == 10:
                # Previous Polygrad spelling: Tensor.randint(low, high, shape=(...)).
                low, high = int(shape[0]), int(shape[1])
                shape = (kw_shape,) if isinstance(kw_shape, int) else tuple(kw_shape)
            elif len(shape) == 0:
                shape = (kw_shape,) if isinstance(kw_shape, int) else tuple(kw_shape)
            else:
                raise TypeError('Tensor.randint got both positional shape and shape= keyword')
        else:
            shape = _shape_tuple(*shape)
        if not isinstance(low, int) or not isinstance(high, int):
            raise TypeError(f'low={low!r} and high={high!r} must be integers')
        rand_kwargs = dict(kwargs)
        dtype_name = _dtype_name(rand_kwargs.pop('dtype', dtype), default='int32')
        dt = to_dtype(dtype_name)
        if not dtypes.is_int(dt):
            raise TypeError(f'dtype={dt!r} must be int')
        if high <= low:
            raise ValueError(f'high must be greater than low, got {low=} {high=}')
        return (Tensor.rand(*shape, dtype='float32', **rand_kwargs) * (high - low) + low).cast(dtype_name)

    @staticmethod
    def randperm(n, device=None, dtype=dtypes.int32, **kwargs):
        return Tensor.rand(int(n), device=device, **kwargs).argsort().cast(dtype)

    @staticmethod
    def linspace(start, stop, steps, **kwargs):
        ctx, dev, requires_grad = _creation_meta(kwargs)
        dtype_name = _dtype_name(kwargs.get('dtype', dtypes.default_float), default='float32')
        tensor = _ffi._lib.poly_tensor_linspace_by_id(
            ctx,
            float(_py_scalar(start)),
            float(_py_scalar(stop)),
            _require_i64(_py_scalar(steps), 'steps'),
            _dtype_id(dtype_name),
            _device_id(dev),
        )
        return _created_tensor(
            ctx, tensor, dtype_name, dev, requires_grad, 'poly_tensor_linspace_by_id'
        )

    @staticmethod
    def eye(n, m=None, **kwargs):
        ctx, dev, requires_grad = _creation_meta(kwargs)
        dtype_name = _dtype_name(kwargs.get('dtype', dtypes.default_float), default='float32')
        rows = _require_i64(_py_scalar(n), 'n')
        cols = rows if m is None else _require_i64(_py_scalar(m), 'm')
        tensor = _ffi._lib.poly_tensor_eye_by_id(
            ctx, rows, cols, _dtype_id(dtype_name), _device_id(dev)
        )
        return _created_tensor(
            ctx, tensor, dtype_name, dev, requires_grad, 'poly_tensor_eye_by_id'
        )

    @staticmethod
    def empty(*shape, **kwargs):
        if 'name' in kwargs:
            raise TypeError('Tensor.empty does not accept name; pass names to Instance.from_tensors')
        ctx, dev, requires_grad = _creation_meta(kwargs)
        dtype_name = _dtype_name(kwargs.get('dtype', dtypes.default_float), default='float32')
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        shape = tuple(_py_scalar(x) for x in shape)
        if any(_is_symbolic_dim(x) for x in shape):
            if not shape or not _is_symbolic_dim(shape[0]):
                raise NotImplementedError('symbolic Tensor.empty currently requires the leading dimension to be symbolic')
            if any(_is_symbolic_dim(x) for x in shape[1:]):
                raise NotImplementedError('symbolic Tensor.empty currently supports one leading symbolic dimension')
            batch_raw = _symbolic_dim_raw(shape[0])
            if not batch_raw or not _ffi._lib.poly_uop_unbind_var(batch_raw):
                raise ValueError('symbolic Tensor.empty dimension must be a Variable, BoundVariable, or variable UOp')
            inner = tuple(_require_i64(_py_scalar(x), 'shape') for x in shape[1:])
            if any(dim < 0 for dim in inner):
                raise ValueError(f'negative dimensions are not allowed: {shape}')
            dims, n_inner = _int64_array(inner) if inner else (None, 0)
            dtype_id = _dtype_id(dtype_name)
            logical = _ffi._lib.poly_buffer_var_by_id(
                ctx, dtype_id, batch_raw, dims, n_inner, 0
            )
            physical = _ffi._lib.poly_buffer_var_by_id(
                ctx, dtype_id, batch_raw, dims, n_inner, _device_id(dev)
            )
            if not logical or not physical:
                raise RuntimeError('poly_buffer_var_by_id failed')
            tensor = _ffi._lib.poly_tensor_create_with_roots(
                ctx, logical, physical, _POLY_TENSOR_VALUE, _device_id(dev)
            )
            if not tensor:
                raise RuntimeError('poly_tensor_create_with_roots failed')
            return Tensor(
                _ctx=ctx, _uop=logical, _tensor=tensor, _shape=shape,
                requires_grad=requires_grad, _dtype=dtype_name, _device=dev,
            )

        shape = tuple(_require_i64(x, 'shape') for x in shape)
        if any(dim < 0 for dim in shape):
            raise ValueError(f'negative dimensions are not allowed: {shape}')
        dtype_id = _dtype_id(dtype_name)
        dims, ndim = _int64_array(shape)
        tensor = _ffi._lib.poly_tensor_empty_by_id(
            ctx, dtype_id, dims, ndim, _device_id(dev)
        )
        if not tensor:
            raise RuntimeError('poly_tensor_empty_by_id failed')
        return Tensor(
            _ctx=ctx, _tensor=tensor, _shape=shape,
            requires_grad=requires_grad, _dtype=dtype_name, _device=dev,
        )

    @staticmethod
    def manual_seed(seed=0):
        Tensor._seed = int(seed)
        Tensor._device_seeds = {}
        Tensor._device_rng_counters = {}

    @staticmethod
    def _next_counter(device, num):
        if device not in Tensor._device_seeds:
            device_index = len(Tensor._device_seeds)
            device_seed = int.from_bytes(
                hashlib.sha256(device_index.to_bytes(4, 'big')).digest(), 'big'
            )
            Tensor._device_seeds[device] = Tensor(
                [device_seed & 0xffffffff, Tensor._seed & 0xffffffff],
                device=device,
                dtype=dtypes.uint32,
            )
            Tensor._device_rng_counters[device] = Tensor(
                [0, 0], device=device, dtype=dtypes.uint32
            )
        counter = Tensor._device_rng_counters[device]
        new_low = counter[0:1] + (num & 0xffffffff)
        new_high = counter[1:2] + (num >> 32) + (new_low < counter[0])
        counter.assign(new_low.cat(new_high))
        low = counter[0:1] - (num & 0xffffffff)
        high = counter[1:2] - (num >> 32) - (counter[0] < (num & 0xffffffff))
        return Tensor._device_seeds[device], low.cat(high)

    @staticmethod
    def _threefry_random_bits(key, counts0, counts1):
        x = counts1.cast(dtypes.uint64).lshift(32).bitwise_or(counts0.cast(dtypes.uint64))
        key_low = key[0]._broadcast_to_tensor(x.shape).cast(dtypes.uint64)
        key_high = key[1]._broadcast_to_tensor(x.shape).cast(dtypes.uint64).lshift(32)
        x = x.threefry(key_high.bitwise_or(key_low))
        mask = 0xffffffff
        return x.bitwise_and(mask).cast(dtypes.uint32).cat(
            x.rshift(32).bitwise_and(mask).cast(dtypes.uint32)
        )

    @staticmethod
    def random_bits(key, counter, num):
        low, high = counter[0:1], counter[1:2]
        bits = []
        for i in range(0, num, dtypes.uint32.max):
            chunk_num = min(num - i, dtypes.uint32.max)
            c_low = low + (i & 0xffffffff)
            c_high = high + (i >> 32) + (c_low < low).cast(dtypes.uint32)
            new_key = Tensor._threefry_random_bits(key, c_low, c_high)
            half = -(-chunk_num // 2)
            counts0 = Tensor.arange(half, device=key.device, dtype=dtypes.uint32)
            counts1 = counts0 + half
            bits.append(Tensor._threefry_random_bits(new_key, counts0, counts1)[:chunk_num])
        return bits[0].cat(*bits[1:]) if bits else counter[0:0]

    @staticmethod
    def _bits_to_rand(bits, shape, dtype):
        _, nmant = dtypes.finfo(dtype)
        uint_dtype = {
            1: dtypes.uint8,
            2: dtypes.uint16,
            4: dtypes.uint32,
            8: dtypes.uint64,
        }[dtype.itemsize]
        uint_bits = bits.bitcast(uint_dtype)
        float_one_bits = (
            uint_bits._ensure_tensor(1)
            ._broadcast_to_tensor(uint_bits.shape)
            .cast(dtype)
            .bitcast(uint_dtype)
        )
        return uint_bits.rshift(dtype.bitsize - nmant).bitwise_or(float_one_bits).bitcast(dtype)[:_prod(shape)].sub(1).reshape(shape)

    @staticmethod
    def _rand(key, counter, shape, dtype, contiguous=True):
        bits = Tensor.random_bits(key, counter, -(-(_prod(shape) * dtype.itemsize) // 4))
        out = Tensor._bits_to_rand(bits, shape, dtype)
        return out.contiguous() if contiguous else out

    def cat(self, *tensors, dim=0):
        if isinstance(self, (list, tuple)) and not tensors:
            tensors = tuple(self)
        elif isinstance(self, Tensor):
            tensors = (self,) + tensors
        else:
            tensors = (self,) + tensors
        if not tensors:
            raise ValueError('cat requires at least one tensor')
        # Implementation: pad each tensor, then sum
        shapes = [t.shape for t in tensors]
        ndim = len(shapes[0])
        if dim < 0:
            dim += ndim

        # Compute output shape
        out_shape = list(shapes[0])
        out_shape[dim] = sum(s[dim] for s in shapes)

        # Pad each tensor to output shape and sum
        offset = 0
        result = None
        for t in tensors:
            pad_before = [0] * ndim
            pad_after = [0] * ndim
            pad_before[dim] = offset
            pad_after[dim] = out_shape[dim] - offset - t.shape[dim]
            padded = t.pad(tuple((pad_before[i], pad_after[i]) for i in range(ndim)))
            if result is None:
                result = padded
            else:
                result = result + padded
            offset += t.shape[dim]
        return result

    def stack(self, *tensors, dim=0):
        if isinstance(self, (list, tuple)) and not tensors:
            tensors = tuple(self)
        elif isinstance(self, Tensor):
            tensors = (self,) + tensors
        else:
            tensors = (self,) + tensors
        return Tensor.cat(*[t.unsqueeze(dim) for t in tensors], dim=dim)

    def split(self, sizes, dim=0):
        if dim < 0:
            dim += len(self.shape)
        if isinstance(sizes, int):
            # Split into chunks of given size
            total = self.shape[dim]
            sizes = [sizes] * (total // sizes)
            if total % sizes[0]:
                sizes.append(total % sizes[0])
        results = []
        offset = 0
        for sz in sizes:
            arg = tuple(
                (offset, offset + sz) if d == dim else (0, s)
                for d, s in enumerate(self.shape)
            )
            results.append(self.shrink(arg))
            offset += sz
        return results

    def chunk(self, n, dim=0):
        if dim < 0:
            dim += len(self.shape)
        total = self.shape[dim]
        chunk_size = math.ceil(total / n)
        return self.split(chunk_size, dim)

    # --- Autograd ---

    def backward(self):
        """Compute gradients for live tensors reachable from this loss UOp."""
        target_entries = self._live_grad_targets()
        if not target_entries:
            raise RuntimeError('No leaf tensors require grad')

        root = _uop_wrap(self._ctx, Tensor._core_uop_raw(self._tensor))
        targets = tuple(t for t, _ in target_entries)
        target_roots = tuple(grad_root for _, grad_root in target_entries)
        custom_records = Tensor._custom_grad_records_for(self._ctx, root)
        if custom_records:
            # Pinned tinygrad differentiates AFTER(data, CALL) along two
            # independent edges: ctx flows directly to data, and the same exact
            # ctx is passed into CALL.grad_fxn. The core handles the data edge;
            # keep every active AFTER as an explicit WRT for the callback edge.
            temp_wrts = []
            seen = set()

            def add_wrt(raw):
                raw = _uop_raw(raw)
                key = _ptr_value(raw) if raw else 0
                if key and key not in seen:
                    seen.add(key)
                    temp_wrts.append(raw)

            for target_root in target_roots:
                add_wrt(target_root)
            active_by_call = []
            for rec, active in custom_records:
                active_aliases = []
                active_call = None
                for after_root, arg_root, call_root in active:
                    after_raw = _uop_raw(after_root)
                    arg_raw = _uop_raw(arg_root)
                    call_raw = _uop_raw(call_root)
                    if active_call is not None and active_call != call_raw:
                        raise RuntimeError('custom_kernel outputs resolve to different CALLs')
                    active_call = call_raw
                    active_aliases.append((after_raw, arg_raw))
                    add_wrt(after_raw)
                active_by_call.append((rec, tuple(active_aliases), active_call))

            temp_grads, temp_present = Tensor._grad_many_raw(
                self._ctx, root, None, temp_wrts, return_present=True
            )
            grad_by_root = {
                _ptr_value(raw): grad for raw, grad in zip(temp_wrts, temp_grads)
            }
            present_by_root = {
                _ptr_value(raw): present for raw, present in zip(temp_wrts, temp_present)
            }

            # Custom kernels are registered in forward construction order.
            # Reverse it so a downstream callback can contribute to an upstream
            # custom AFTER before that earlier callback is invoked.
            for rec, active_aliases, active_call in reversed(active_by_call):
                upstreams = [
                    UOp(self._ctx, grad_by_root.get(_ptr_value(after_raw)))
                    for after_raw, _arg_raw in active_aliases
                    if present_by_root.get(_ptr_value(after_raw), False)
                ]
                if not upstreams:
                    continue
                call = UOp(self._ctx, active_call)
                call_args = call.src[1:]
                if rec['grad_fxn'] is None:
                    needs_call_grad = any(
                        _ffi._lib.poly_uop_reachable(
                            self._ctx, _uop_raw(arg), target_root
                        )
                        for arg in call_args for target_root in target_roots
                    )
                    if needs_call_grad:
                        body_op = call.src[0].op_name
                        raise AssertionError(
                            f'expected TUPLE body for gradient, got Ops.{body_op}'
                        )
                    continue
                if len(upstreams) > 1:
                    returned = rec['grad_fxn'](*upstreams, call=call)
                else:
                    returned = rec['grad_fxn'](upstreams[0], call)
                if returned is None:
                    continue
                if isinstance(returned, (Tensor, UOp)):
                    returned = (returned,)
                if len(returned) != len(call_args):
                    raise RuntimeError(
                        f"custom_kernel grad_fxn returned {len(returned)} grads, expected {len(call_args)}"
                    )
                for arg, arg_grad in zip(call_args, returned):
                    arg_grad_raw = Tensor._grad_result_raw(arg_grad)
                    if not arg_grad_raw:
                        continue
                    propagated, propagated_present = Tensor._grad_many_raw(
                        self._ctx, _uop_raw(arg), arg_grad_raw, temp_wrts,
                        return_present=True,
                    )
                    for wrt_raw, contrib, is_present in zip(
                        temp_wrts, propagated, propagated_present
                    ):
                        if not is_present:
                            continue
                        key = _ptr_value(wrt_raw)
                        grad_by_root[key] = Tensor._accumulate_grad_raw(
                            self._ctx, grad_by_root.get(key), contrib
                        )
                        present_by_root[key] = True
            out_grads = tuple(grad_by_root.get(_ptr_value(raw)) for raw in target_roots)
        else:
            # tinygrad calls gradient(*targets), so every live target is handled by
            # one reverse pass. Calling poly_grad repeatedly can observe frontend
            # retargeting side effects between targets.
            out_grads = Tensor._grad_many_raw(self._ctx, root, None, target_roots)

        for target, grad_uop in zip(targets, out_grads):
            if not grad_uop:
                raise RuntimeError('poly_grad_many returned NULL for a live target')
            grad_handle = Tensor._core_create_with_roots_for(
                self._ctx, grad_uop, grad_uop, _POLY_TENSOR_VALUE, target._device
            )
            if not grad_handle:
                raise RuntimeError('failed to store backward gradient roots')
            grad_tensor = Tensor(
                _ctx=self._ctx, _tensor=grad_handle, _shape=target.shape,
                _dtype=target._dtype_str, _device=target._device,
            )
            if int(_ffi._lib.poly_uop_device(grad_uop)) == 0:
                grad_tensor = grad_tensor.clone(device=target._device)
            if target._grad is not None:
                target._grad.assign(target._grad + grad_tensor.to(target._grad.device))
            else:
                target._grad = grad_tensor
        return self

    # --- Representation ---

    def __repr__(self):
        if self.uop.has_buffer_identity() and self._data is not None and self.numel() <= 16:
            data_str = str(self._data.reshape(self.shape).tolist())
            return f'Tensor({data_str}, shape={self.shape}, dtype={self.dtype})'
        return f'Tensor(shape={self.shape}, dtype={self.dtype})'

    def __len__(self):
        if not self.shape:
            raise TypeError('len() of scalar tensor')
        return self.shape[0]

    def __hash__(self):
        return id(self)

    def __bool__(self):
        raise TypeError("__bool__ on Tensor is not defined")
