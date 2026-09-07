"""
Tensor class for polygrad, lazy evaluation backed by the C compiler core.
Supports float32 (default) and float64 dtypes.
"""

import ctypes
import functools
import math
import pathlib
import sys
import weakref
from typing import cast as cast

import numpy as np

from . import _ffi
from .dtype import INVERSE_DTYPES_DICT, Invalid, _from_np_dtype, _to_np_dtype, dtypes, least_upper_dtype, least_upper_float, strong_dtype, to_dtype
from polygrad.uop.ops import UOp
from polygrad.device import Buffer
from polygrad.helpers import TRAINING, _logical_policy_name, _logical_state_name, _normalize_logical_policy


# Global registry of live Tensors. Keys are weakrefs so GC'd tensors vanish
# automatically. Used for post-realize retargeting and backward graph
# discovery, matching tinygrad's live-tensor registry.
all_tensors: dict[weakref.ref, None] = {}
_custom_kernel_grad_keys = weakref.WeakKeyDictionary()
_custom_kernel_grad_fxns = {}
_next_custom_kernel_grad_key = 1


def _custom_kernel_grad_key(ctx, grad_fxn):
    global _next_custom_kernel_grad_key
    if grad_fxn is None:
        return 0
    key = _custom_kernel_grad_keys.get(grad_fxn)
    if key is None:
        if _next_custom_kernel_grad_key > 0xFFFFFFFF:
            raise OverflowError('custom kernel gradient key space exhausted')
        key = _next_custom_kernel_grad_key
        _next_custom_kernel_grad_key += 1
        _custom_kernel_grad_keys[grad_fxn] = key
    _custom_kernel_grad_fxns[(_ptr_value(ctx), key)] = grad_fxn
    return key


def _dispose_tensors_for_ctx(ctx):
    ctx_key = _ptr_value(ctx)
    for ref in list(all_tensors):
        tensor = ref()
        if tensor is None:
            all_tensors.pop(ref, None)
        elif _ptr_value(tensor._ctx) == ctx_key:
            tensor.dispose()
            tensor._ctx = None
    for key in [key for key in _custom_kernel_grad_fxns if key[0] == ctx_key]:
        _custom_kernel_grad_fxns.pop(key, None)

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
    return int(raw) if raw else 0


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


def _argfix(*args):
    # Pinned helpers.argfix: a sequence and positional dimensions cannot mix.
    if args and args[0].__class__ in (tuple, list):
        if len(args) != 1:
            raise ValueError(f'bad arg {args}')
        return tuple(args[0])
    return args


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
    UOp wrappers around ALU BUFFER variables and AFTER/STORE bindings, matching
    tinygrad's `uop.shape`. `poly_uop_max_shape_dims` exposes max_shape storage.
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
    'fp8e4m3', 'fp8e5m2', 'fp8e4m3fnuz', 'fp8e5m2fnuz', 'weakint', 'weakfloat',
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
    sdt = dt
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
        out *= v
    return out


def _dtype_min_value(dtype_name):
    dt = to_dtype(dtype_name)
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
    dt = to_dtype(dtype)
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
    return ctx, dev


def _shape_arg(shape):
    # Reuse one normalization path so every constructor feeds the C helpers the
    # same shape encoding and errors on bad dimensions the same way.
    if any(_is_symbolic_dim(s) for s in shape):
        raise TypeError('this constructor does not yet accept symbolic dimensions')
    shape = tuple(int(s) for s in shape)
    if any(s < 0 for s in shape):
        raise ValueError(f'invalid input shape={shape}')
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
    return UOp.const(int(value), ctx=ctx).raw


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
        return UOp.const(int(value), ctx=ctx)
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
    if isinstance(start_obj, int) and start_obj == 0:
        return stop_u
    delta = _uop_add_const_delta(stop_u, start_u)
    if delta is not None:
        return UOp.const(delta, ctx=ctx)
    # Negative symbolic slice starts use base + offset. Preserve the constant
    # extent exposed by Tinygrad's ssimplify(base - (base + offset)).
    delta = _uop_add_const_delta(start_u, stop_u)
    if delta is not None:
        return UOp.const(-delta, ctx=ctx)
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


def _created_tensor(ctx, tensor, dtype_name, device, op_name):
    # Constructors should trust the core for final shape metadata so Python
    # does not grow a second copy of shape or dual-root ownership logic.
    if not tensor:
        raise RuntimeError(f'{op_name} failed')
    uop = _ffi._lib.poly_tensor_uop(tensor)
    if not uop:
        raise RuntimeError(f'{op_name} returned a Tensor without a current UOp')
    dtype_name = _dtype_name_from_id(int(_ffi._lib.poly_uop_dtype_id(ctx, uop)))
    return Tensor(
        _ctx=ctx, _tensor=tensor, _shape=_shape_from_uop(ctx, uop),
        _dtype=dtype_name, _device=device,
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
        self.uop = UOp.variable(name, min_val, max_val, ctx=self._ctx)

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

    def __init__(self, data=None, *, dtype=None, device=None, logical=None, _ctx=None, _uop=None,
                 _data=None, _shape=None, _dtype=None, _device=None, _tensor=None):
        from . import _default_ctx
        ctx = _ctx or (data.ctx if isinstance(data, UOp) else None) or _default_ctx
        policy = _normalize_logical_policy(logical)
        if policy is None:
            self._init(data, dtype=dtype, device=device, _ctx=_ctx, _uop=_uop, _data=_data,
                       _shape=_shape, _dtype=_dtype, _device=_device, _tensor=_tensor)
            return
        lib = _ffi.get_lib()
        old_policy = int(lib.poly_ctx_get_logical_policy(ctx))
        if lib.poly_ctx_set_logical_policy(ctx, policy) != 0:
            raise ValueError(f"invalid logical policy {logical!r}")
        try:
            self._init(data, dtype=dtype, device=device, _ctx=_ctx, _uop=_uop, _data=_data,
                       _shape=_shape, _dtype=_dtype, _device=_device, _tensor=_tensor)
        finally:
            if lib.poly_ctx_set_logical_policy(ctx, old_policy) != 0:
                raise RuntimeError("failed to restore logical policy")

    def _init(self, data=None, *, dtype=None, device=None, _ctx=None, _uop=None,
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
            if data is None:
                data = 0.0
            numpy_scalar = isinstance(data, np.ndarray) and data.shape == ()
            scalar_value = data.item() if numpy_scalar else data
            python_scalar = numpy_scalar or isinstance(data, (bool, int, float))
            if python_scalar:
                if numpy_scalar:
                    default_dt = _dtype_name(data.dtype, default='float32')
                else:
                    default_dt = (
                        'bool' if isinstance(data, bool)
                        else 'weakint' if isinstance(data, int)
                        else 'weakfloat'
                    )
                dt = _dtype_name(dtype, default=default_dt)
                scalar_dt = to_dtype(dt)
                normalized = scalar_dt.const(scalar_value)
                dtype_id = _dtype_id(dt)
                target_device_id = _device_id(self._device)
                if dtypes.is_bool(scalar_dt) or dtypes.is_int(scalar_dt):
                    value = int(bool(normalized)) if dtypes.is_bool(scalar_dt) else int(normalized)
                    if 0 <= value < 2**64 and value > I64_MAX:
                        factory = _ffi._lib.poly_tensor_const_uint_by_id
                    elif value < I64_MIN or value > I64_MAX:
                        raise ValueError(f'scalar {value} is out of int64 range')
                    else:
                        factory = _ffi._lib.poly_tensor_const_int_by_id
                    self._tensor = factory(
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
                    dt = _dtype_name(dtype, default='uint8' if isinstance(data, bytes) else 'float32')
                import_dt = dt
                post_cast_dt = None
                storage_dtype = to_dtype(dt)
                if storage_dtype in dtypes.weaks:
                    raise RuntimeError(f'cannot create storage for weak dtype {storage_dtype}')
                if isinstance(data, bytes):
                    # _frompy(bytes) owns writable encoded storage. A void view
                    # preserves bits (including BF16/FP8), validates whole
                    # elements, and gives the existing host importer its shape.
                    arr = np.frombuffer(data, dtype=f'V{storage_dtype.itemsize}').copy()
                else:
                    np_dt = _to_np_dtype(dt)
                    if to_dtype(dt) in {dtypes.bfloat16, *dtypes.fp8s}:
                        import_dt = 'float32'
                        post_cast_dt = dt
                        np_dt = np.float32
                    try:
                        arr = np.ascontiguousarray(data, dtype=np_dt)
                    except OverflowError:
                        if not isinstance(data, (list, tuple)) or not dtypes.is_int(storage_dtype):
                            raise
                        # UOp._frompy applies dtype.const then truncate before
                        # packing. NumPy2 rejects overflowing Python integers;
                        # object storage preserves them until the ctypes cast.
                        # Keep normal inputs and ragged-shape validation on
                        # NumPy's fast path, rather than boxing every input.
                        values = np.asarray(data, dtype=object)
                        ctype = getattr(ctypes, f'c_{dt}')
                        arr = np.fromiter((ctype(int(x)).value for x in values.flat),
                                          dtype=np_dt, count=values.size).reshape(values.shape)
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
                        cast_tensor = _ffi._lib.poly_tensor_cast_by_id(
                            self._ctx, self._tensor, _dtype_id(post_cast_dt)
                        )
                        if not cast_tensor:
                            raise RuntimeError(
                                f'poly_tensor_cast_by_id failed for dtype {post_cast_dt}'
                            )
                        self._replace_core_tensor(cast_tensor)
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

        if imported_tensor_from_host:
            source_device_id = int(_ffi._lib.poly_tensor_device(self._tensor))
            target_device_id = _device_id(self._device)
            if target_device_id != source_device_id:
                moved_tensor = _ffi._lib.poly_tensor_to_device(
                    self._ctx, self._tensor, target_device_id
                )
                if not moved_tensor:
                    raise RuntimeError(f'poly_tensor_to_device failed for {self._device}')
                self._replace_core_tensor(moved_tensor)
        elif self._tensor is None and current_uop is not None:
            source_device = disk_device if imported_from_disk else 'CPU'
            target_device_id = _device_id(self._device)
            source_device_id = _device_id(source_device)
            if imported_from_disk and target_device_id != source_device_id:
                source = self._core_create(current_uop, _POLY_TENSOR_VALUE, source_device)
                try:
                    self._tensor = _ffi._lib.poly_tensor_to_device(
                        self._ctx, source, target_device_id
                    )
                    if not self._tensor:
                        raise RuntimeError(f'poly_tensor_to_device failed for {self._device}')
                finally:
                    if source:
                        _ffi._lib.poly_tensor_release(source)
            else:
                self._tensor = self._core_create(current_uop, _POLY_TENSOR_VALUE, self._device)
        self._grad = None
        self._is_param = True
        all_tensors[weakref.ref(self)] = None

    def _replace_core_tensor(self, tensor):
        old = getattr(self, '_tensor', None)
        if _ptr_value(old) == _ptr_value(tensor):
            self._tensor = tensor
            return tensor
        self._tensor = tensor
        if old:
            _ffi._lib.poly_tensor_release(old)
        return tensor

    def _take_core_tensor(self):
        tensor = getattr(self, '_tensor', None)
        self._tensor = None
        return tensor

    def dispose(self):
        tensor = self._take_core_tensor()
        if tensor:
            _ffi._lib.poly_tensor_release(tensor)

    def __del__(self):
        try:
            self.dispose()
        except Exception:
            pass

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

    def _core_create_result_like(self, logical, physical, role=_POLY_TENSOR_VALUE, device=None):
        logical_raw = _uop_raw(logical) if logical is not None else None
        physical_raw = _uop_raw(physical)
        if not physical_raw:
            return None
        return _ffi._lib.poly_tensor_create_result_like(
            self._ctx, self._tensor, logical_raw, physical_raw, int(role),
            _device_id(self._device if device is None else device),
        )

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
        if self._ctx is None:
            raise RuntimeError('polygrad runtime has been disposed')
        raw = self._core_uop_raw(self._tensor)
        return _uop_wrap(self._ctx, raw)

    @property
    def uop_logical(self):
        if self._ctx is None:
            raise RuntimeError('polygrad runtime has been disposed')
        raw = self._core_uop_logical_raw(self._tensor)
        return _uop_wrap(self._ctx, raw)

    @property
    def uop_physical(self):
        if self._ctx is None:
            raise RuntimeError('polygrad runtime has been disposed')
        raw = self._core_uop_physical_raw(self._tensor)
        return _uop_wrap(self._ctx, raw)

    @property
    def logical_policy(self):
        return _logical_policy_name(_ffi._lib.poly_tensor_logical_policy(self._tensor))

    @property
    def logical_state(self):
        return _logical_state_name(_ffi._lib.poly_tensor_logical_state(self._tensor))

    def set_logical_policy(self, policy):
        policy_id = _normalize_logical_policy(policy)
        if policy_id is None:
            return True
        return _ffi._lib.poly_tensor_set_logical_policy(
            self._ctx, self._tensor, policy_id
        ) == 0

    def preserve_logical(self):
        if not self.set_logical_policy("always"):
            raise RuntimeError("logical producer is no longer available")
        return self

    @property
    def shape(self):
        """Read concrete and symbolic extents from the current physical UOp."""
        if self._ctx is None:
            raise RuntimeError('polygrad runtime has been disposed')
        if getattr(self, '_shape_override', None) is not None:
            return self._shape_override
        if self._tensor is None:
            return ()
        # Pinned Tensor.shape is derived from the one current Tensor.uop. On
        # Polygrad's split boundary that parity surface is uop_physical;
        # uop_logical is reserved for export/re-placement (tensor.py:76-121).
        raw = self._core_uop_raw(self._tensor)
        if not raw:
            return ()
        return _shape_from_uop(self._ctx, raw)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def dtype(self):
        return to_dtype(self._dtype_str)

    @property
    def device(self):
        if self._ctx is None:
            raise RuntimeError('polygrad runtime has been disposed')
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

    @grad.setter
    def grad(self, value):
        # Like Tensor.grad in the pin, this owns a Tensor reference. Its C
        # handle already retains the gradient graph; resetting releases ours.
        self._grad = value

    @property
    def T(self):
        return self.transpose()

    def numel(self):
        # Pinned MovementMixin.numel returns the symbolic product and keeps
        # the all-static assertion at data/readback boundaries
        # (mixin/movement.py:38-47, tensor.py:278-280,311-313).
        return _prod(self.shape)

    @property
    def max_shape(self):
        """Maximum extents without replacing symbolic dimensions in the graph."""
        if self._ctx is None:
            raise RuntimeError('polygrad runtime has been disposed')
        raw = self._core_uop_raw(self._tensor)
        ndim = _ffi._lib.poly_uop_ndim(self._ctx, raw)
        dims = _ffi._lib.poly_uop_max_shape_dims(self._ctx, raw)
        return tuple(int(dims[i]) for i in range(ndim))

    def max_numel(self):
        return _prod(self.max_shape)

    def size(self, dim=None):
        if dim is None:
            return self.shape
        if dim < 0:
            dim += len(self.shape)
        return self.shape[dim]

    def custom_kernel(self, *lst, fxn, grad_fxn=None):
        """Call a custom ``SINK(..., arg=KernelInfo(...))`` kernel written in UOps.

        Mirrors tinygrad's alpha `Tensor.custom_kernel`: inputs are made
        contiguous, placeholder PARAM UOps are passed to `fxn`, the returned
        compiler-ready SINK body is wrapped in CALL, and every source tensor
        is returned as `AFTER(source, call)`.
        """
        srcs = (self,) + tuple(lst)
        for t in srcs:
            if not isinstance(t, Tensor):
                raise TypeError('custom_kernel expects Tensor arguments')
            if t._ctx != self._ctx:
                raise ValueError('custom_kernel tensors must share a context')
        contig = tuple(
            t if t.uop and t.uop.op == _ffi.OPS.get('AFTER') else t.contiguous()
            for t in srcs
        )
        placeholders = [UOp.placeholder_like(t.uop, slot=i) for i, t in enumerate(contig)]
        body = fxn(*placeholders)
        if not isinstance(body, UOp):
            raise TypeError('custom_kernel fxn must return a UOp SINK body')
        input_arr = (_ffi._ptr * len(contig))(*[t._tensor for t in contig])
        output_arr = (_ffi._ptr * len(contig))()
        grad_fxn_key = _custom_kernel_grad_key(self._ctx, grad_fxn)
        if _ffi._lib.poly_tensor_custom_kernel(
            self._ctx, body.raw, input_arr, len(contig), grad_fxn_key, output_arr
        ) != 0:
            raise RuntimeError('poly_tensor_custom_kernel failed')
        outs = []
        physical_afters = []
        for t, core in zip(contig, output_arr):
            physical = self._core_uop_physical_raw(core)
            physical_afters.append(_ptr_value(physical))
            out = Tensor(
                _ctx=t._ctx,
                _tensor=core,
                _dtype=t._dtype_str,
                _device=t._device,
            )
            outs.append(out)
        call = _ffi._lib.poly_uop_src(physical_afters[0], 1)
        if not call or _ffi._lib.poly_uop_op(call) != _ffi.OPS.get('CALL'):
            raise RuntimeError('custom_kernel physical output is not AFTER(data, CALL)')
        if _ffi._lib.poly_uop_call_grad_fxn_key(call) != grad_fxn_key:
            raise RuntimeError('custom_kernel CALL lost its gradient identity')
        return outs

    # --- Realization ---

    def _live_grad_targets(self):
        """Find gradient targets the same way tinygrad does.

        Current tinygrad discovers targets only through each current
        ``Tensor.uop`` (tensor.py:657-677). Polygrad's corresponding root is
        the mandatory physical/current root; retained logical provenance is
        never an execution fallback.
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
            if _ptr_value(t._ctx) != ctx_key or not dtypes.is_float(t.dtype):
                continue
            current_raw = Tensor._core_uop_raw(t._tensor)
            if (current_raw and int(_ffi._lib.poly_uop_device(current_raw)) != 0 and
                    _ffi._lib.poly_uop_reachable(self._ctx, root, current_raw)):
                targets.append((t, current_raw))
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
        lib = _ffi._lib
        topo, seen, stack = [], set(), [(root_raw, False)]
        while stack:
            node, expanded = stack.pop()
            node_key = _ptr_value(node)
            if expanded:
                topo.append(node)
                continue
            if not node_key or node_key in seen:
                continue
            seen.add(node_key)
            stack.append((node, True))
            for i in range(lib.poly_uop_n_src(node) - 1, -1, -1):
                src = lib.poly_uop_src(node, i)
                if src:
                    stack.append((src, False))

        by_call = {}
        for after in topo:
            if lib.poly_uop_op(after) != _ffi.OPS.get('AFTER') or lib.poly_uop_n_src(after) != 2:
                continue
            data, call = lib.poly_uop_src(after, 0), lib.poly_uop_src(after, 1)
            if not call or lib.poly_uop_op(call) != _ffi.OPS.get('CALL'):
                continue
            body = lib.poly_uop_src(call, 0)
            if not body or lib.poly_uop_op(body) != _ffi.OPS.get('SINK'):
                continue
            arg = None
            for i in range(1, lib.poly_uop_n_src(call)):
                candidate = lib.poly_uop_src(call, i)
                if _ptr_value(candidate) == _ptr_value(data):
                    arg = candidate
                    break
            if not arg:
                continue
            call_key = _ptr_value(call)
            rec_active = by_call.get(call_key)
            if rec_active is None:
                grad_key = int(lib.poly_uop_call_grad_fxn_key(call))
                grad_fxn = _custom_kernel_grad_fxns.get((ctx_key, grad_key)) if grad_key else None
                if grad_key and grad_fxn is None:
                    raise RuntimeError('custom_kernel gradient callback is unavailable')
                rec_active = ({'grad_fxn': grad_fxn}, [])
                by_call[call_key] = rec_active
            active = rec_active[1]
            if all(_ptr_value(existing[0]) != _ptr_value(after) for existing in active):
                active.append((after, arg, call))
        return [(rec, tuple(active)) for rec, active in by_call.values()]

    @staticmethod
    def _grad_result_raw(grad):
        if grad is None:
            return None
        if isinstance(grad, Tensor):
            return Tensor._core_uop_raw(grad._tensor)
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

    def replace(self, x):
        """Replace this Tensor's value graph while preserving its wrapper identity.

        Pinned tinygrad tensor.py:221-228 assigns ``self.uop = x.uop`` after
        checking shape. Polygrad adapts that operation at its documented
        logical/physical boundary by replacing both roots atomically.
        """
        if not isinstance(x, Tensor):
            raise TypeError('replace expects a Tensor')
        if self.shape != x.shape:
            raise AssertionError(f'replace shape mismatch {self.shape} != {x.shape}')
        if _ptr_value(self._ctx) != _ptr_value(x._ctx):
            raise RuntimeError('replace requires Tensors owned by the same PolyCtx')
        logical, physical = x.uop_logical, x.uop_physical
        if logical is None or physical is None:
            raise RuntimeError('replace source must have logical and physical roots')
        rc = _ffi._lib.poly_tensor_replace_roots(
            self._ctx,
            self._tensor,
            logical.raw,
            physical.raw,
            _POLY_TENSOR_VALUE,
            _device_id(x._device),
        )
        if rc != 0:
            raise RuntimeError('poly_tensor_replace_roots failed during replace')
        self._data = x._data
        self._dtype_str = x._dtype_str
        self._device = x._device
        return self

    def assign(self, x):
        """In-place assignment: self's buffer will be overwritten with x's values.
        Must be realized before use. Returns self for chaining."""
        if not isinstance(x, Tensor):
            x = Tensor(x, dtype=self._dtype_str, device=self._device)
        if self.shape != x.shape:
            if x._broadcast_shape(self.shape) != self.shape:
                raise ValueError(f'assign shape mismatch {self.shape} != {x.shape}')
        if self.device != x.device:
            raise RuntimeError(f'assign device mismatch {self.device} != {x.device}')
        if self.dtype != x.dtype:
            raise RuntimeError(f'assign dtype mismatch {self.dtype} != {x.dtype}')

        assigned = self._core_assign(x)
        if not assigned:
            raise RuntimeError('poly_tensor_assign failed')
        self._replace_core_tensor(assigned)
        self._data = None
        return self

    def copy_from(self, data):
        """Materialize pending work, then update current storage from host data.
        Existing BUFFER identity is preserved for JIT/replay loops. This is
        Polygrad's host-write API; ordinary graph mutation uses assign()."""
        np_dt = _to_np_dtype(to_dtype(self._dtype_str))
        arr = np.asarray(data, dtype=np_dt)
        if arr.size != self.numel():
            raise ValueError(f'copy_from size mismatch {arr.size} != {self.numel()}')
        arr = np.ascontiguousarray(arr.reshape(self.shape))
        ptr = ctypes.c_void_p(arr.ctypes.data)
        physical = self.uop_physical
        if physical is None:
            raise RuntimeError('copy_from requires a physical Tensor root')
        # Tensor._buffer -> Buffer.copy_from: finish COPY/AFTER effects before
        # writing, then use current storage only. UOp.buffer alone is not a
        # materialization test: it can look through an unexecuted AFTER.
        if not _ffi._lib.poly_uop_has_buffer_identity(physical.raw):
            self.realize()
            physical = self.uop_physical
        write_buf = physical.buffer
        if write_buf is None:
            raise RuntimeError('copy_from requires a tensor backed by a BUFFER UOp')
        target_device = _device_id(self._device)
        rc = _ffi._lib.poly_buffer_ensure_device_allocated(
            self._ctx, write_buf.raw, target_device
        )
        if rc != 0:
            raise RuntimeError('poly_buffer_ensure_device_allocated failed')
        rc = _ffi._lib.poly_buffer_write(self._ctx, write_buf.raw, ptr, arr.nbytes)
        if rc != 0:
            raise RuntimeError('poly_buffer_write failed')
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
        x = self.cast(to_dtype(self._dtype_str)).contiguous()
        if int(_ffi._lib.poly_uop_device(self.uop.raw)) == 0 or isinstance(self._device, tuple):
            x = x.clone("CPU")
        x.realize()
        return Buffer(self._ctx, x.uop.buffer, x._dtype_str, x.numel())

    def data(self):
        """Return tensor contents as a shaped memoryview, matching tinygrad."""
        dtype = to_dtype(self._dtype_str)
        if dtype in dtypes.weaks:
            return self.cast(strong_dtype(dtype)).data()
        shape = self.shape
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
        if dtype in dtypes.weaks:
            return self.cast(strong_dtype(dtype)).numpy()
        if dtype in {dtypes.bfloat16, *dtypes.fp8s}:
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
        if to_dtype(self._dtype_str) == dtypes.half:
            return self.float().tolist()
        return self.numpy().tolist()

    def detach(self):
        # Pinned mixin/elementwise.py:33-37 is one DETACH Tensor ALU.
        core = _ffi._lib.poly_tensor_detach(self._ctx, self._tensor)
        return self._make_result_from_core(core, self.shape, [self])

    def contiguous_backward(self):
        """Insert a contiguous operation in the backward pass."""
        # Pinned mixin/elementwise.py:51-55 is one CONTIGUOUS_BACKWARD UOp;
        # the C Tensor bridge owns both retained and executable roots.
        core = _ffi._lib.poly_tensor_contiguous_backward(self._ctx, self._tensor)
        return self._make_result_from_core(core, self.shape, [self])

    def clone(self, device=None):
        from .device import Device

        dev = self._device if device is None else Device.canonicalize(device)
        cloned = _ffi._lib.poly_tensor_clone(
            self._ctx, self._tensor, _device_id(dev),
        )
        if not cloned:
            raise RuntimeError('poly_tensor_clone failed')
        ret = Tensor(
            _ctx=self._ctx,
            _tensor=cloned,
            _device=dev,
        )
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
        if _ptr_value(core_tensor) == _ptr_value(self._tensor):
            return self

        out = Tensor(
            _ctx=self._ctx,
            _tensor=core_tensor,
            _data=self._data,
            _dtype=self._dtype_str,
            _device=dev,
        )
        out._grad = self._grad.to(dev) if self._grad is not None else None
        out._is_param = self._is_param
        return out

    def to_(self, device):
        moved = self.to(device)
        if moved is self:
            return self
        self._replace_core_tensor(moved._take_core_tensor())
        self._data = moved._data
        self._dtype_str = moved._dtype_str
        self._device = moved._device
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
            raise NotImplementedError('contiguous optimization options are not yet exposed by the C Tensor API')
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
        if current in (dtypes.weakint, dtypes.weakfloat) or target in (dtypes.weakint, dtypes.weakfloat):
            raise RuntimeError(f'bitcast requires concrete dtypes, got {current} -> {target}')
        if current == target:
            return self
        # Current DTypeMixin.bitcast constructs one raw BITCAST and leaves
        # last-axis validation/inference to UOp._shape (mixin/dtype.py:35-50,
        # uop/ops.py:404-411). The C core owns both roots and that shape rule.
        core = _ffi._lib.poly_tensor_bitcast_by_id(
            self._ctx, self._tensor, _dtype_id(target)
        )
        if not core:
            raise RuntimeError('unsupported size in bitcast')
        # Unequal-width BITCAST changes the final dimension. Derive the
        # wrapper shape from the returned UOp instead of copying source shape.
        result_raw = self._core_uop_raw(core)
        return self._make_result_from_core(
            core, _shape_from_uop(self._ctx, result_raw), [self]
        )

    def element_size(self):
        """Storage bytes per element; weak dtypes have no storage width."""
        if self.dtype in dtypes.weaks:
            raise RuntimeError(f'element_size requires a concrete dtype, got {self.dtype}')
        return self.dtype.itemsize

    def is_floating_point(self):
        return dtypes.is_float(self.dtype)

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
        ret._grad = None
        ret._is_param = True
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
            if _ptr_value(other._ctx) != _ptr_value(self._ctx):
                raise ValueError('Tensor operands must belong to the same Polygrad context')
            return other
        if isinstance(other, np.generic):
            other = other.item()
        if isinstance(other, (bool, int, float)):
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
                self._ctx, tensor, const_dtype_name, self._device,
                'C-owned internal scalar Tensor construction',
            )
        raise TypeError(f'Cannot convert {type(other)} to Tensor')

    def const_like(self, value):
        """Pinned CreationMixin.const_like: typed CONST broadcast to this shape."""
        if isinstance(value, np.generic):
            value = value.item()
        if not isinstance(value, (bool, int, float)):
            raise TypeError(f'const_like value must be numeric, got {type(value)}')
        if isinstance(value, (bool, int)):
            normalized = int(bool(value)) if isinstance(value, bool) else int(value)
            if normalized < I64_MIN or normalized > I64_MAX:
                raise ValueError(f'scalar {normalized} is out of int64 range')
            core = _ffi._lib.poly_tensor_const_like_int(
                self._ctx, self._tensor, normalized
            )
        else:
            core = _ffi._lib.poly_tensor_const_like_float(
                self._ctx, self._tensor, float(value)
            )
        return self._make_result_from_core(core, self.shape, [self])

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

    def _broadcast_to_tensor(self, target_shape):
        """Return the ordered Tensor occurrence used by an elementwise ALU.

        Pinned tinygrad's _broadcasted/_broadcast_to first constructs movement
        Tensor occurrences, then Tensor.alu consumes their current UOps
        (mixin/__init__.py:439-449, mixin/movement.py:116-128).
        """
        return self.expand(tuple(target_shape))

    # --- Element-wise arithmetic ---

    def _broadcasted(self, other, reverse=False):
        other = self._ensure_tensor(other)
        x, y = (self, other) if not reverse else (other, self)
        out_shape = x._broadcast_shape(y.shape)
        return x, y, out_shape

    def _binop(self, other, op_name, reverse=False):
        """Build a binary Tensor ALU from ordered current operand occurrences."""
        x, y, out_shape = self._broadcasted(other, reverse)
        core = _ffi._lib.poly_tensor_alu2(
            self._ctx, _ffi.OPS[op_name], x._tensor, y._tensor
        )
        return self._make_result_from_core(core, None, [x, y])

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

    def __floordiv__(self, other):
        return self.div(other, rounding_mode='floor')

    def __rfloordiv__(self, other):
        return self.div(other, reverse=True, rounding_mode='floor')

    def bitwise_not(self):
        core = _ffi._lib.poly_tensor_bitwise_not(self._ctx, self._tensor)
        return self._make_result_from_core(core, None, [self])

    def __invert__(self):
        return self.bitwise_not()

    def mod(self, other, reverse=False):
        a, b, shape = self._broadcasted(other, reverse)
        if dtypes.is_int(a.dtype) and dtypes.is_int(b.dtype):
            core = _ffi._lib.poly_tensor_alu2(self._ctx, _ffi.OPS['FLOORMOD'], a._tensor, b._tensor)
            return self._make_result_from_core(core, shape, [a, b])
        return a - a.div(b, rounding_mode='floor') * b

    def __mod__(self, other):
        return self.mod(other)

    def __rmod__(self, other):
        return self.mod(other, reverse=True)

    def fmod(self, other):
        a, b, shape = self._broadcasted(other)
        if dtypes.is_int(a.dtype) and dtypes.is_int(b.dtype):
            core = _ffi._lib.poly_tensor_alu2(self._ctx, _ffi.OPS['CMOD'], a._tensor, b._tensor)
            return self._make_result_from_core(core, shape, [a, b])
        return a - a.div(b, rounding_mode='trunc') * b

    def masked_fill(self, mask, value):
        return self._ensure_tensor(mask).where(value, self)

    def __rtruediv__(self, other):
        return self.div(other, reverse=True)

    def div(self, other, reverse=False, rounding_mode=None):
        # Pinned ElementwiseMixin.div selects integer CDIV/FLOORDIV after
        # broadcasting and promotion, otherwise rounds true division
        # (mixin/elementwise.py:219-247).
        dividend, divisor, out_shape = self._broadcasted(other, reverse)
        if rounding_mode not in (None, 'trunc', 'floor'):
            raise RuntimeError(f"rounding_mode={rounding_mode!r} is not supported")
        core = _ffi._lib.poly_tensor_div(
            self._ctx, dividend._tensor, divisor._tensor, (None, 'trunc', 'floor').index(rounding_mode)
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
        # Tinygrad 2026-08-22/a9069c177a9d mixin/elementwise.py:545-564
        # validates the promoted scalar pair and returns the raw POW.
        common = least_upper_dtype(to_dtype(base.dtype), to_dtype(exponent.dtype))
        if (not dtypes.is_float(common) and
                not isinstance(other, Tensor) and not (isinstance(other, int) and other >= 0)):
            raise RuntimeError("base needs to be float")
        core = _ffi._lib.poly_tensor_alu2(
            self._ctx, _ffi.OPS['POW'], base._tensor, exponent._tensor
        )
        return self._make_result_from_core(core, out_shape, [base, exponent])

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
        # Current ElementwiseMixin.where selects a branch Tensor only to own
        # host-scalar conversion. C owns branch promotion and ALU shape
        # inference for both frontends (mixin/elementwise.py:422-434).
        ref = x if isinstance(x, Tensor) else y if isinstance(y, Tensor) else self
        x = x if isinstance(x, Tensor) else ref._ensure_tensor(x)
        y = y if isinstance(y, Tensor) else ref._ensure_tensor(y)
        core = _ffi._lib.poly_tensor_alu3(
            self._ctx, _ffi.OPS['WHERE'], self._tensor, x._tensor, y._tensor
        )
        return self._make_result_from_core(core, None, [self, x, y])

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

    def clip(self, min_=None, max_=None):
        """Alias for :meth:`Tensor.clamp` (mixin/elementwise.py:582-584)."""
        return self.clamp(min_, max_)

    # --- Unary math (C core composed ops) ---

    def exp2(self):
        # Current Tinygrad emits the raw ALU op and lets UOp.dtype own
        # least_upper_float (mixin/elementwise.py:513-531, uop/ops.py:144-145).
        core = _ffi._lib.poly_tensor_alu1(
            self._ctx, _ffi.OPS['EXP2'], self._tensor
        )
        return self._make_result_from_core(core, self.shape, [self])

    def log2(self):
        # Current Tinygrad emits the raw ALU op; result dtype is C-owned.
        core = _ffi._lib.poly_tensor_alu1(
            self._ctx, _ffi.OPS['LOG2'], self._tensor
        )
        return self._make_result_from_core(core, self.shape, [self])

    def sqrt(self):
        # Current Tinygrad emits the raw ALU op; result dtype is C-owned.
        core = _ffi._lib.poly_tensor_alu1(
            self._ctx, _ffi.OPS['SQRT'], self._tensor
        )
        return self._make_result_from_core(core, self.shape, [self])

    def reciprocal(self):
        core = _ffi._lib.poly_tensor_alu1(
            self._ctx, _ffi.OPS['RECIPROCAL'], self._tensor
        )
        return self._make_result_from_core(core, self.shape, [self])

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

    def log10(self):
        core = _ffi._lib.poly_tensor_log10(self._ctx, self._tensor)
        return self._make_result_from_core(core, self.shape, [self])

    def atanh(self):
        core = _ffi._lib.poly_tensor_atanh(self._ctx, self._tensor)
        return self._make_result_from_core(core, self.shape, [self])

    def asinh(self):
        core = _ffi._lib.poly_tensor_asinh(self._ctx, self._tensor)
        return self._make_result_from_core(core, self.shape, [self])

    def acosh(self):
        core = _ffi._lib.poly_tensor_acosh(self._ctx, self._tensor)
        return self._make_result_from_core(core, self.shape, [self])

    def asin(self):
        core = _ffi._lib.poly_tensor_asin(self._ctx, self._tensor)
        return self._make_result_from_core(core, self.shape, [self])

    def acos(self):
        core = _ffi._lib.poly_tensor_acos(self._ctx, self._tensor)
        return self._make_result_from_core(core, self.shape, [self])

    def atan(self):
        core = _ffi._lib.poly_tensor_atan(self._ctx, self._tensor)
        return self._make_result_from_core(core, self.shape, [self])

    def logsigmoid(self):
        core = _ffi._lib.poly_tensor_logsigmoid(self._ctx, self._tensor)
        return self._make_result_from_core(core, self.shape, [self])

    def sinh(self):
        core = _ffi._lib.poly_tensor_sinh(self._ctx, self._tensor)
        return self._make_result_from_core(core, self.shape, [self])

    def cosh(self):
        core = _ffi._lib.poly_tensor_cosh(self._ctx, self._tensor)
        return self._make_result_from_core(core, self.shape, [self])

    def erf(self):
        core = _ffi._lib.poly_tensor_erf(self._ctx, self._tensor)
        return self._make_result_from_core(core, self.shape, [self])

    def softsign(self):
        core = _ffi._lib.poly_tensor_softsign(self._ctx, self._tensor)
        return self._make_result_from_core(core, self.shape, [self])

    def isfinite(self):
        core = _ffi._lib.poly_tensor_isfinite(self._ctx, self._tensor)
        return self._make_result_from_core(core, self.shape, [self])

    def celu(self, alpha=1.0):
        alpha = self._ensure_tensor(alpha)
        core = _ffi._lib.poly_tensor_celu(self._ctx, self._tensor, alpha._tensor)
        return self._make_result_from_core(core, None, [self, alpha])

    def selu(self, alpha=1.67326, gamma=1.0507):
        alpha, gamma = self._ensure_tensor(alpha), self._ensure_tensor(gamma)
        core = _ffi._lib.poly_tensor_selu(self._ctx, self._tensor, alpha._tensor, gamma._tensor)
        return self._make_result_from_core(core, None, [self, alpha, gamma])

    def isclose(self, other, rtol=1e-5, atol=1e-8, equal_nan=False):
        other, rtol, atol = (self._ensure_tensor(v) for v in (other, rtol, atol))
        core = _ffi._lib.poly_tensor_isclose(self._ctx, self._tensor, other._tensor, rtol._tensor, atol._tensor, bool(equal_nan))
        return self._make_result_from_core(core, None, [self, other, rtol, atol])

    def copysign(self, other):
        other = self._ensure_tensor(other)
        core = _ffi._lib.poly_tensor_copysign(self._ctx, self._tensor, other._tensor)
        return self._make_result_from_core(core, None, [self, other])

    def lerp(self, end, weight):
        scalar_weight = not isinstance(weight, Tensor)
        end, weight = self._ensure_tensor(end), self._ensure_tensor(weight)
        core = _ffi._lib.poly_tensor_lerp(self._ctx, self._tensor, end._tensor, weight._tensor, scalar_weight)
        return self._make_result_from_core(core, None, [self, end, weight])

    @staticmethod
    def _loss_reduction_id(reduction):
        try:
            return ('none', 'sum', 'mean').index(reduction)
        except ValueError:
            raise ValueError("reduction must be 'none', 'sum', or 'mean'") from None

    def binary_crossentropy_logits(self, Y, reduction='mean', pos_weight=None):
        reduction_id = self._loss_reduction_id(reduction)
        Y = self._ensure_tensor(Y)
        weight = self._ensure_tensor(pos_weight) if pos_weight is not None else None
        core = _ffi._lib.poly_tensor_binary_crossentropy_logits(
            self._ctx, self._tensor, Y._tensor, weight._tensor if weight is not None else None, reduction_id)
        return self._make_result_from_core(core, None, [self, Y] + ([weight] if weight is not None else []))

    def nll_loss(self, Y, weight=None, ignore_index=None, reduction='mean'):
        reduction_id = self._loss_reduction_id(reduction)
        Y = self._ensure_tensor(Y)
        weight = self._ensure_tensor(weight) if weight is not None else None
        ignore = self._ensure_tensor(ignore_index) if ignore_index is not None else None
        core = _ffi._lib.poly_tensor_nll_loss(
            self._ctx, self._tensor, Y._tensor, weight._tensor if weight is not None else None,
            ignore._tensor if ignore is not None else None, reduction_id)
        return self._make_result_from_core(core, None, [self, Y] + [v for v in (weight, ignore) if v is not None])

    def log1p(self):
        core = _ffi._lib.poly_tensor_log1p(self._ctx, self._tensor)
        return self._make_result_from_core(core, self.shape, [self])

    def expm1(self):
        core = _ffi._lib.poly_tensor_expm1(self._ctx, self._tensor)
        return self._make_result_from_core(core, self.shape, [self])

    def sin(self):
        # Current Tinygrad emits SIN over the exact Tensor occurrence and lets
        # UOp.dtype promote its result (mixin/elementwise.py:468-478).
        core = _ffi._lib.poly_tensor_alu1(
            self._ctx, _ffi.OPS['SIN'], self._tensor
        )
        return self._make_result_from_core(core, self.shape, [self])

    def cos(self):
        # Current least_upper_float/float32 composition is shared in C
        # (mixin/elementwise.py:480-489).
        core = _ffi._lib.poly_tensor_cos(self._ctx, self._tensor)
        return self._make_result_from_core(core, self.shape, [self])

    def tan(self):
        # Current self.sin()/self.cos() composition is shared in C
        # (mixin/elementwise.py:917-927).
        core = _ffi._lib.poly_tensor_tan(self._ctx, self._tensor)
        return self._make_result_from_core(core, self.shape, [self])

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

    def gelu(self, approximate='tanh'):
        # Pinned mixin/elementwise.py:761-776. C applies the exact formula
        # independently to retained/current roots.
        if approximate not in ('tanh', 'none'):
            raise RuntimeError(f'unknown GELU approximation: {approximate}')
        fn = _ffi._lib.poly_tensor_gelu if approximate == 'tanh' else _ffi._lib.poly_tensor_gelu_exact
        core = fn(self._ctx, self._tensor)
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

    def dropout(self, p=0.5):
        # Direct port of pinned tensor.py:809-829.
        if not 0 <= p <= 1:
            raise ValueError(f'p={p} is out of range [0, 1]')
        if not TRAINING or p == 0:
            return self
        if p == 1:
            return self.const_like(0)
        return (Tensor.rand_like(self, dtype=dtypes.default_float, contiguous=False) >= p).contiguous().where(self, 0) / (1.0 - p)

    def scaled_dot_product_attention(
        self, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, enable_gqa=False,
    ):
        # Direct port of pinned tensor.py:831-858.
        if enable_gqa:
            key = key.repeat_interleave(int(self.shape[-3] // key.shape[-3]), dim=-3)
            value = value.repeat_interleave(int(self.shape[-3] // value.shape[-3]), dim=-3)

        qk = self.matmul(
            key.transpose(-2, -1),
            dtype=least_upper_dtype(to_dtype(self.dtype), to_dtype(key.dtype), dtypes.float32),
        ) / math.sqrt(self.shape[-1])
        if is_causal:
            if attn_mask is not None:
                raise RuntimeError('cannot set attn_mask when is_causal=True')
            attn_mask = qk.const_like(1).cast(dtypes.bool).tril()
        if attn_mask is not None:
            if dtypes.is_bool(to_dtype(attn_mask.dtype)):
                attn_mask = attn_mask.where(0, -float('inf'))
            qk = qk + attn_mask
        return qk.cast(self.dtype).softmax(-1).dropout(dropout_p) @ value

    # --- Movement ops ---

    def reshape(self, shape, *args):
        shape = _argfix(shape, *args)
        shape = tuple(s if s is not None else self.shape[i] for i, s in enumerate(shape))
        if (inferred := shape.count(-1)) > 1:
            raise RuntimeError(
                f"only one dimension can be inferred using -1, getting {shape}"
            )
        if inferred:
            shape = tuple(
                -self.numel() // _prod(shape) if s == -1 else s for s in shape
            )
        symbolic = _shape_has_symbolic(self.shape) or _shape_has_symbolic(shape)
        if not symbolic and self.numel() != _prod(shape):
            raise ValueError(f"size mismatch, can't reshape ({self.shape}) -> ({shape})")
        if shape == self.shape:
            return self
        if symbolic:
            dims = _shape_uop_array(self._ctx, shape)
            core = _ffi._lib.poly_tensor_reshape_uop(
                self._ctx, self._tensor, dims, len(shape)
            )
            if not core:
                raise ValueError(f"size mismatch, can't reshape ({self.shape}) -> ({shape})")
            current = self._core_uop_raw(core)
            return self._make_result_from_core(
                core, _shape_from_uop(self._ctx, current), [self]
            )
        arr, n = _int64_array(shape)
        core = _ffi._lib.poly_tensor_reshape(self._ctx, self._tensor, arr, n)
        return self._make_result_from_core(core, shape, [self])

    def permute(self, order, *args):
        # Direct port of pinned mixin/movement.py:195-211.
        order = tuple(self._resolve_dim(int(axis)) for axis in (
            tuple(order) if isinstance(order, (tuple, list)) and not args else (order, *args)
        ))
        if sorted(order) != list(range(self.ndim)):
            raise RuntimeError(f'order is not a valid permutation, getting {order}')
        if order == tuple(range(self.ndim)):
            return self
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
            dims = _shape_uop_array(self._ctx, shape)
            core = _ffi._lib.poly_tensor_expand_uop(
                self._ctx, self._tensor, dims, len(shape)
            )
            if not core:
                raise RuntimeError('poly_tensor_expand_uop failed')
            current = self._core_uop_raw(core)
            return self._make_result_from_core(
                core, _shape_from_uop(self._ctx, current), [self]
            )
        dims, n = _int64_array(shape)
        core = _ffi._lib.poly_tensor_expand(self._ctx, self._tensor, dims, n)
        return self._make_result_from_core(core, shape, [self])

    def shrink(self, arg):
        """Shrink by (start, end) pairs; None leaves that axis unchanged."""
        shape = self.shape
        if len(arg) != len(shape):
            raise ValueError(f'ndim={len(shape)} != len(arg)={len(arg)}')
        if any(p is None for p in arg):
            arg = tuple((0, s) if p is None else p for p, s in zip(arg, shape))
        if all(start == 0 and end == size for (start, end), size in zip(arg, shape)):
            return self
        symbolic = any(isinstance(v, (BoundVariable, UOp)) for pair in arg for v in pair)
        if symbolic:
            # Keep symbolic source identities; max_shape is only an allocation bound.
            starts = [_bound_to_uop(self._ctx, start) for start, _ in arg]
            ends = [_bound_to_uop(self._ctx, end) for _, end in arg]
            sizes = [_symbolic_slice_size_uop(self._ctx, a, b, start, end)
                     for (a, b), start, end in zip(arg, starts, ends)]
            core = _ffi._lib.poly_tensor_shrink_uop(
                self._ctx, self._tensor, _shape_uop_array(self._ctx, starts),
                _shape_uop_array(self._ctx, sizes), len(arg)
            )
        else:
            flat, n = _pair_array(arg)
            core = _ffi._lib.poly_tensor_shrink(self._ctx, self._tensor, flat, n)
        if not core:
            raise ValueError(f'invalid shrink {arg} for {shape}')
        new_shape = (_shape_from_uop(self._ctx, self._core_uop_raw(core)) if symbolic
                     else tuple(end - start for start, end in arg))
        return self._make_result_from_core(core, new_shape, [self])

    def shrink_to(self, shape, *args):
        shape = _argfix(shape, *args)
        return self.shrink(tuple(None if s is None else (0, s) for s in shape))

    def pad_to(self, shape, *args, value=0):
        shape = _argfix(shape, *args)
        current = self.shape
        if len(shape) != len(current):
            raise ValueError(f'ndim={len(current)} != len(shape)={len(shape)}')
        shape = tuple(s if ns is None else ns for s, ns in zip(current, shape))
        if shape == current:
            return self
        if not _shape_all_int(current) or not _shape_all_int(shape):
            raise NotImplementedError('symbolic padding is not supported')
        if any(ns < s for s, ns in zip(current, shape)):
            raise ValueError(f'invalid pad_to {shape} for {current}')
        return self._pad_constant(tuple((0, ns - s) for s, ns in zip(current, shape)), value)

    def pad(self, padding, mode="constant", value=0.0):
        """Pad using tinygrad-compatible flat or grouped padding."""
        arg = _normalize_pad_arg(padding, self.ndim)
        if mode == 'constant':
            return self._pad_constant(arg, value)
        if mode not in ('circular', 'reflect', 'replicate'):
            raise NotImplementedError(f"mode={mode!r} is not supported")
        pairs = tuple((0, 0) if p is None else p for p in arg)
        flat, n = _pair_array(pairs)
        core = _ffi._lib.poly_tensor_pad_mode(self._ctx, self._tensor, flat, n, {'circular': 1, 'reflect': 2, 'replicate': 3}[mode])
        if not core:
            raise ValueError(f'invalid {mode} padding {pairs} for {self.shape}')
        return self._make_result_from_core(core, None, [self])

    def _pad_constant(self, arg, value):
        arg = tuple((0, 0) if p is None else tuple(p) for p in arg)
        flat, n = _pair_array(arg)
        # Pinned _pad_constant shrinks negative pads before emitting a
        # non-negative PAD (mixin/__init__.py:359-368). Keep that policy in
        # the shared C boundary for both zero and nonzero fill values.
        if isinstance(value, bool):
            fn, scalar = _ffi._lib.poly_tensor_pad_value_bool, value
        elif isinstance(value, int):
            fn, scalar = _ffi._lib.poly_tensor_pad_value_int, value
        elif isinstance(value, float):
            fn, scalar = _ffi._lib.poly_tensor_pad_value_float, value
        else:
            raise TypeError(f"pad value must be bool, int, or float, got {type(value).__name__}")
        core = fn(self._ctx, self._tensor, flat, n, scalar)
        new_shape = tuple(s + b + a for s, (b, a) in zip(self.shape, arg))
        return self._make_result_from_core(core, new_shape, [self])

    def flip(self, axis, *args):
        axes = tuple(axis) if isinstance(axis, (tuple, list)) else (axis,)
        axes = axes + tuple(args)
        axes = tuple(self._resolve_dim(int(a)) for a in axes)
        if len(set(axes)) != len(axes):
            raise RuntimeError(f"dim can appear at most once, getting {axes}")
        if not axes:
            return self
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
            dim = self._resolve_dim(dim)
            if not self.ndim or self.shape[dim] != 1:
                return self
            new_shape = tuple(s for i, s in enumerate(self.shape) if i != dim)
            return self.reshape(new_shape)
        new_shape = tuple(s for s in self.shape if s != 1)
        if new_shape == self.shape:
            return self
        return self.reshape(new_shape)

    def unsqueeze(self, dim):
        dim = self._resolve_dim(dim, extra=True)
        new_shape = list(self.shape)
        new_shape.insert(dim, 1)
        return self.reshape(tuple(new_shape))

    def flatten(self, start_dim=0, end_dim=-1):
        # Pinned tinygrad/mixin/movement.py:319-321 resolves both ends before
        # multiplying the selected dimensions, including negative start_dim.
        start_dim = self._resolve_dim(int(start_dim))
        end_dim = self._resolve_dim(int(end_dim))
        return self.reshape(
            self.shape[:start_dim]
            + (math.prod(self.shape[start_dim:end_dim + 1]),)
            + self.shape[end_dim + 1:]
        )

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
        # Literal pinned MovementMixin.repeat: repeat-one axes remain single
        # lanes instead of introducing no-op RESHAPE/EXPAND dimensions
        # (mixin/movement.py:536-554).
        base_shape = (1,) * (max(len(self.shape), len(repeats)) - len(self.shape)) + self.shape
        unsqueezed_shape = tuple(
            v for r, s in zip(repeats, base_shape) for v in ((s,) if r == 1 else (1, s))
        )
        expanded_shape = tuple(
            v for r, s in zip(repeats, base_shape) for v in ((s,) if r == 1 else (r, s))
        )
        final_shape = tuple(r * s for r, s in zip(repeats, base_shape))
        return self.reshape(unsqueezed_shape).expand(expanded_shape).reshape(final_shape)

    def repeat_interleave(self, repeats, dim=None):
        # Direct port of pinned mixin/movement.py:520-534.
        x, dim = (self.flatten(), 0) if dim is None else (self, self._resolve_dim(dim))
        shp = x.shape
        x = x.reshape(*shp[:dim + 1], 1, *shp[dim + 1:])
        x = x.expand(*shp[:dim + 1], repeats, *shp[dim + 1:])
        return x.reshape(*shp[:dim], shp[dim] * repeats, *shp[dim + 1:])

    def roll(self, shifts, dims=None):
        if dims is None:
            return self.flatten().roll(shifts, 0).reshape(self.shape)

        dims = (int(dims),) if isinstance(dims, (int, np.integer)) else tuple(dims)
        shifts = (int(shifts),) if isinstance(shifts, (int, np.integer)) else tuple(shifts)
        dims = tuple(self._resolve_dim(d) for d in dims)
        if len(dims) != len(shifts):
            raise RuntimeError(f"len(dims)={len(dims)} != len(shifts)={len(shifts)}")

        if 0 in self.shape:
            return self

        shrink_arg = [(0, s) for s in self.shape]
        for d, s in zip(dims, shifts):
            size = self.shape[d]
            delta = size - int(s) % size
            shrink_arg[d] = (delta, delta + size)
        repeats = tuple(2 if i in dims else 1 for i in range(self.ndim))
        return self.repeat(*repeats).shrink(tuple(shrink_arg))

    # --- Reduction ops (C core) ---

    def sum(self, axis=None, keepdim=False, dtype=None):
        if axis is None:
            axis = tuple(range(self.ndim))
        elif isinstance(axis, int):
            axis = (axis,)
        axis = tuple(self._resolve_dim(int(a)) for a in axis)
        arr, n = _int64_array(axis)
        core = (
            _ffi._lib.poly_tensor_sum(self._ctx, self._tensor, arr, n, bool(keepdim))
            if dtype is None else
            _ffi._lib.poly_tensor_sum_dtype_by_id(
                self._ctx, self._tensor, arr, n, bool(keepdim), _dtype_id(_dtype_name(dtype))
            )
        )
        new_shape = (
            tuple(1 if i in axis else s for i, s in enumerate(self.shape))
            if keepdim else
            tuple(s for i, s in enumerate(self.shape) if i not in axis)
        )
        return self._make_result_from_core(core, new_shape, [self])

    def prod(self, axis=None, keepdim=False, dtype=None):
        x = self if dtype is None else self.cast(dtype)
        return x._extremum(_ffi._lib.poly_tensor_prod, axis, keepdim)

    def logsumexp(self, axis=None, keepdim=False):
        return self._extremum(_ffi._lib.poly_tensor_logsumexp, axis, keepdim)

    def logcumsumexp(self, axis=0):
        core = _ffi._lib.poly_tensor_logcumsumexp(self._ctx, self._tensor, self._resolve_dim(axis))
        return self._make_result_from_core(core, None, [self])

    def normalize(self, p=2.0, dim=1, eps=1e-12):
        core = _ffi._lib.poly_tensor_normalize(self._ctx, self._tensor, float(p), self._resolve_dim(dim), float(eps))
        return self._make_result_from_core(core, None, [self])

    def softmin(self, axis=-1, dtype=None):
        x = -self
        return (x if dtype is None else x.cast(dtype)).softmax(axis)

    def std_mean(self, axis=None, keepdim=False, correction=1):
        return self.std(axis, keepdim, correction), self.mean(axis, keepdim)

    def argmin(self, axis=None, keepdim=False):
        if axis is None:
            return self.flatten().argmin(0)
        core = _ffi._lib.poly_tensor_argmin(self._ctx, self._tensor, self._resolve_dim(axis), bool(keepdim))
        return self._make_result_from_core(core, None, [self])

    def diag(self):
        if self.ndim != 1:
            raise ValueError('diag requires a vector')
        core = _ffi._lib.poly_tensor_diag(self._ctx, self._tensor)
        return self._make_result_from_core(core, None, [self])

    def diagonal(self, offset=0, dim1=0, dim2=1):
        dim1, dim2 = self._resolve_dim(dim1), self._resolve_dim(dim2)
        if dim1 == dim2:
            raise RuntimeError('diagonal dimensions must differ')
        core = _ffi._lib.poly_tensor_diagonal(self._ctx, self._tensor, int(offset), dim1, dim2)
        return self._make_result_from_core(core, None, [self])

    def unfold(self, dim, size, step):
        dim = self._resolve_dim(dim)
        if size < 0 or step <= 0 or size > self.shape[dim]:
            raise RuntimeError('invalid unfold size or step')
        core = _ffi._lib.poly_tensor_unfold(self._ctx, self._tensor, dim, int(size), int(step))
        return self._make_result_from_core(core, None, [self])

    def meshgrid(self, *args, indexing='ij'):
        if indexing not in ('ij', 'xy'):
            raise RuntimeError('indexing must be ij or xy')
        tensors = (self,) + args
        if len(tensors) == 1:
            return tensors
        basis = tuple(range(len(tensors))) if indexing == 'ij' else (1, 0) + tuple(range(2, len(tensors)))
        tensors = tuple(t.reshape((-1,) + (1,) * (len(args)-i)) for i, t in zip(basis, tensors))
        shape = ()
        for t in tensors:
            shape = _broadcast_shapes(shape, t.shape)
        return tuple(t.expand(shape) for t in tensors)

    def max(self, axis=None, keepdim=False):
        return self._extremum(_ffi._lib.poly_tensor_max, axis, keepdim)

    def all(self, axis=None, keepdim=False):
        return self._extremum(_ffi._lib.poly_tensor_all, axis, keepdim)

    def any(self, axis=None, keepdim=False):
        return self._extremum(_ffi._lib.poly_tensor_any, axis, keepdim)

    def cumsum(self, axis=0):
        core = _ffi._lib.poly_tensor_cumsum(self._ctx, self._tensor, self._resolve_dim(axis))
        return self._make_result_from_core(core, None, [self])

    def cumprod(self, axis):
        core = _ffi._lib.poly_tensor_cumprod(self._ctx, self._tensor, self._resolve_dim(axis))
        return self._make_result_from_core(core, None, [self])

    def _cum_extremum(self, operation, axis):
        values, indices = _ffi._ptr(), _ffi._ptr()
        rc = operation(self._ctx, self._tensor, self._resolve_dim(axis),
                       ctypes.byref(values), ctypes.byref(indices))
        if rc != 0 or not values or not indices:
            raise RuntimeError('core cumulative extremum failed')
        return (self._make_result_from_core(values, None, [self]),
                self._make_result_from_core(indices, None, [self]))

    def cummax(self, axis=0):
        return self._cum_extremum(_ffi._lib.poly_tensor_cummax, axis)

    def cummin(self, axis=0):
        return self._cum_extremum(_ffi._lib.poly_tensor_cummin, axis)

    def _extremum(self, operation, axis, keepdim):
        # Pinned ReduceMixin._reduce emits one REDUCE over the complete
        # normalized axis tuple (mixin/reduce.py:12-17, uop/ops.py:567-569).
        axes = tuple(range(self.ndim)) if axis is None else (
            (self._resolve_dim(int(axis)),) if isinstance(axis, int) else
            tuple(self._resolve_dim(int(a)) for a in axis)
        )
        arr, n = _int64_array(axes)
        core = operation(
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
        return vals, idx

    def min(self, axis=None, keepdim=False):
        return self._extremum(_ffi._lib.poly_tensor_min, axis, keepdim)

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

    def dot(self, w, dtype=None):
        if not isinstance(w, Tensor):
            raise TypeError(f'Expected Tensor, got {type(w)}')
        core = (
            _ffi._lib.poly_tensor_dot(self._ctx, self._tensor, w._tensor)
            if dtype is None else
            _ffi._lib.poly_tensor_dot_dtype_by_id(
                self._ctx, self._tensor, w._tensor, _dtype_id(_dtype_name(dtype))
            )
        )
        if not core:
            raise ValueError(f'cannot dot {self.shape} and {w.shape}')
        current = self._core_uop_raw(core)
        return self._make_result_from_core(
            core, _shape_from_uop(self._ctx, current), [self, w]
        )

    def matmul(self, other, reverse=False, dtype=None):
        return other.dot(self, dtype=dtype) if reverse else self.dot(other, dtype=dtype)

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
        return self.matmul(other)

    def __rmatmul__(self, other):
        other = self._ensure_tensor(other)
        return self.matmul(other, reverse=True)

    def linear(self, weight, bias=None, dtype=None):
        # Direct port of pinned mixin/__init__.py:1335-1350. Stateful
        # nn.Linear owns its stored-weight transpose at the module boundary.
        if dtype is not None:
            dt = to_dtype(dtype)
            return self.cast(dt).linear(
                weight.cast(dt), bias.cast(dt) if bias is not None else None
            )
        result = self.mul(weight) if len(weight.shape) == 1 else self.dot(weight)
        return result.add(bias) if bias is not None else result

    def sequential(self, ll):
        return functools.reduce(lambda x, f: f(x), ll, self)

    def _pool2d_args(self, kernel_size, stride, dilation, padding):
        k = _make_tuple(kernel_size, 2)
        s = k if stride is None else _make_tuple(stride, len(k))
        d = _make_tuple(dilation, len(k))
        assert len(k) == len(s) == len(d), f'stride/dilation mismatch kernel:{k} stride:{s} dilation:{d}'
        pads = _resolve_pool_pads(padding, len(k))
        k_arr, nk = _int64_array(k)
        s_arr, _ = _int64_array(s)
        d_arr, _ = _int64_array(d)
        p_arr, npad = _int64_array(pads)
        return self._ctx, self._tensor, k_arr, nk, s_arr, d_arr, p_arr, npad

    def max_pool2d(self, kernel_size=(2, 2), stride=None, dilation=1, padding=0,
                   ceil_mode=False, return_indices=False):
        indices = ctypes.c_void_p()
        core = _ffi._lib.poly_tensor_max_pool2d(
            *self._pool2d_args(kernel_size, stride, dilation, padding),
            bool(ceil_mode), ctypes.byref(indices) if return_indices else None)
        if not core:
            raise RuntimeError('poly_max_pool2d failed')
        current = self._core_uop_raw(core)
        shape = _shape_from_uop(self._ctx, current)
        out = self._make_result_from_core(core, shape, [self])
        return (out, self._make_result_from_core(indices.value, shape, [self])) if return_indices else out

    def avg_pool2d(self, kernel_size=(2, 2), stride=None, dilation=1, padding=0,
                   ceil_mode=False, count_include_pad=True):
        core = _ffi._lib.poly_tensor_avg_pool2d(
            *self._pool2d_args(kernel_size, stride, dilation, padding), bool(ceil_mode), bool(count_include_pad))
        if not core:
            raise RuntimeError('poly_avg_pool2d failed')
        return self._make_result_from_core(core, _shape_from_uop(self._ctx, self._core_uop_raw(core)), [self])

    def interpolate(self, size, mode='linear', align_corners=False):
        assert isinstance(size, (tuple, list)) and all(isinstance(x, int) for x in size) and 0 < len(size) <= self.ndim, f'invalid size={size}'
        assert mode in ('linear', 'nearest', 'nearest-exact'), 'only supports linear, nearest or nearest-exact interpolate'
        assert not (align_corners and mode != 'linear'), 'align_corners option can only be set with the interpolating mode linear'
        sizes, n = _int64_array(size)
        core = _ffi._lib.poly_tensor_interpolate(self._ctx, self._tensor, sizes, n, mode.encode(), bool(align_corners))
        if not core:
            raise RuntimeError('poly_interpolate failed')
        return self._make_result_from_core(core, _shape_from_uop(self._ctx, self._core_uop_raw(core)), [self])

    def max_unpool2d(self, indices, kernel_size=(2, 2), stride=None, dilation=1, padding=0, output_size=None):
        ctx, tensor, k, nk, s, d, p, npad = self._pool2d_args(kernel_size, stride, dilation, padding)
        output, n = _int64_array(()) if output_size is None else _int64_array(output_size)
        core = _ffi._lib.poly_tensor_max_unpool2d(ctx, tensor, indices._tensor, k, nk, s, d, p, npad, output, n)
        if not core: raise RuntimeError('poly_max_unpool2d failed')
        return self._make_result_from_core(core, _shape_from_uop(self._ctx, self._core_uop_raw(core)), [self, indices])

    def conv_transpose2d(self, weight, bias=None, groups=1, stride=1, dilation=1, padding=0, output_padding=0):
        if not isinstance(weight, Tensor): weight = self._ensure_tensor(weight)
        if bias is not None and not isinstance(bias, Tensor): bias = self._ensure_tensor(bias)
        nk = len(weight.shape) - 2
        strides, dilations = _make_tuple(stride, nk), _make_tuple(dilation, nk)
        assert len(dilations) == nk, 'stride/dilation mismatch'
        # The pin only consumes stride when inserting spaces. C always needs
        # nk readable entries; unused short/extra tuples normalize to ones.
        if any(s > 1 for s in strides):
            if len(strides) != nk: raise ValueError('stride length mismatch')
        else: strides = (1,) * nk
        output = _make_tuple(output_padding, nk)[:nk]
        if not output: raise ValueError('output_padding must not be empty')
        s, _ = _int64_array(strides)
        d, _ = _int64_array(dilations)
        p, npad = _int64_array(_resolve_pool_pads(padding, nk))
        op, nop = _int64_array(output)
        core = _ffi._lib.poly_tensor_conv_transpose2d(self._ctx, self._tensor, weight._tensor,
                bias._tensor if bias is not None else None, int(groups), s, d, p, npad, op, nop)
        if not core: raise RuntimeError('poly_conv_transpose2d failed')
        return self._make_result_from_core(core, _shape_from_uop(self._ctx, self._core_uop_raw(core)),
                                          [self, weight] + ([] if bias is None else [bias]))

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
        args = (
            self._ctx, self._tensor, weight._tensor, bias._tensor if bias is not None else None,
            int(groups), s_arr, d_arr, p_arr, npad,
        )
        core = (
            _ffi._lib.poly_tensor_conv2d(*args)
            if dtype is None else
            _ffi._lib.poly_tensor_conv2d_dtype_by_id(*args, _dtype_id(_dtype_name(dtype)))
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

    def sparse_categorical_crossentropy(self, target, ignore_index=-1, label_smoothing=0.0, reduction='mean'):
        """Pinned sparse loss uses the last class axis, unlike cross_entropy."""
        assert 0 <= label_smoothing <= 1, 'label_smoothing must be in [0, 1]'
        target = self._ensure_tensor(target)
        if target.device != self.device:
            raise RuntimeError('loss inputs must be on the same device')
        core = _ffi._lib.poly_tensor_sparse_categorical_crossentropy(
            self._ctx, self._tensor, target._tensor, _require_i64(ignore_index, 'ignore_index'),
            float(label_smoothing), self._loss_reduction_id(reduction))
        return self._make_result_from_core(core, None, [self, target])

    def binary_crossentropy(self, target, reduction='mean'):
        reduction_id = self._loss_reduction_id(reduction)
        target = self._ensure_tensor(target)
        core = _ffi._lib.poly_tensor_binary_crossentropy(self._ctx, self._tensor, target._tensor, reduction_id)
        return self._make_result_from_core(core, None, [self, target])

    def layernorm(self, axis=-1, eps=1e-5):
        # Pinned mixin/__init__.py:1548-1564. Keep the centered value shared
        # and use the exact mul/rsqrt spelling; this is ordinary lazy graph
        # construction and never a materialization boundary.
        y = self - self.mean(axis=axis, keepdim=True)
        return y.mul((y * y).mean(axis=axis, keepdim=True).add(eps).rsqrt())

    # --- Indexing ---

    def _index_args(self, indices):
        # Normalize host syntax only. OpMixin._getitem's view, mask, reduction
        # and write graphs are built once in C for both frontends.
        if (isinstance(indices, list) and all(isinstance(i, int) for i in indices)) or not isinstance(indices, (tuple, list)):
            indices = [indices]
        indices = list(indices)
        ell = [j for j, i in enumerate(indices) if i is Ellipsis]
        if len(ell) > 1:
            raise IndexError('indices can only have a single ellipsis')
        real = len(indices) - len(ell) - sum(i is None for i in indices)
        if real > self.ndim:
            raise IndexError(f'too many indices ({real}) for {self.ndim}D')
        at = ell[0] if ell else len(indices)
        indices[at:at + 1] = [slice(None)] * (self.ndim - real)
        kinds, starts, sizes, steps, tensors, owners, view_shape = [], [], [], [], [], [], []
        dim = 0
        for index in indices:
            kind, tensor, step = 2, None, 1
            if index is None:
                kinds.append(0); starts.append(None); sizes.append(None); steps.append(1); tensors.append(None)
                continue
            size = self.shape[dim]
            dim += 1
            start, extent = 0, size
            if isinstance(index, (list, tuple)):
                def flatten(v):
                    return [z for a in v for z in flatten(a)] if isinstance(v, (list, tuple)) else [v]
                flat = flatten(index)
                # OpMixin._getitem infers bool only when every entry is bool;
                # a mixed bool/int list is an integer index list (all_int).
                if not flat or any(not isinstance(v, int) for v in flat) or all(isinstance(v, bool) for v in flat):
                    raise IndexError(f'index={index!r} contains non-int element')
                if not isinstance(size, int):
                    raise AssertionError('size must be an int')
                # The pin adjusts list entries before _frompy; doing this as
                # a Tensor WHERE would produce a different physical graph.
                def normalize(v):
                    return [normalize(a) for a in v] if isinstance(v, (tuple, list)) else v + size if v < 0 else v
                tensor = Tensor(normalize(index), dtype=dtypes.default_int, device=self.device, _ctx=self._ctx)
                kind = 4
            elif isinstance(index, Tensor):
                if not dtypes.is_int(index.dtype):
                    raise IndexError(f'index dtype {index.dtype} is not supported')
                if index._ctx != self._ctx or index.device != self.device:
                    raise RuntimeError('expected index and self on the same device and context')
                tensor, kind = index, 3
            elif isinstance(index, int) or _is_symbolic_bound(index):
                if isinstance(size, int) and isinstance(index, int) and not -size <= index < size:
                    raise IndexError(f'index={index} is out of bounds with size={size}')
                start = index if index >= 0 else size + index
                extent, kind = 1, 1
            elif isinstance(index, slice):
                if not all(v is None or isinstance(v, int) or _is_symbolic_bound(v) for v in (index.start, index.stop, index.step)):
                    raise TypeError(f'slice index={index!r} is not supported')
                step = 1 if index.step is None else index.step
                if step == 0:
                    raise ValueError(f'index={index!r} cannot have 0 as step')
                begin = 0 if index.start is None else index.start
                end = size if index.stop is None else index.stop
                if isinstance(begin, int) and begin < 0: begin += size
                if isinstance(end, int) and end < 0: end += size
                if all(isinstance(v, int) for v in (begin, end, step)):
                    lo, hi, step = index.indices(_slice_indices_size(self._ctx, self.uop, dim-1, size))
                    if step * (hi - lo) < 0: lo, hi = 0, 0
                    elif step < 0: lo, hi = hi + 1, lo + 1
                    start, extent = lo, hi - lo
                elif step == 1:
                    start_u, end_u = _bound_to_uop(self._ctx, begin), _bound_to_uop(self._ctx, end)
                    start = begin
                    extent = _symbolic_slice_size_uop(self._ctx, begin, end, start_u, end_u)
                else:
                    raise TypeError(f'slice index={index!r} is not supported for symbolic shape')
            else:
                raise IndexError(f'{type(index).__name__} indexing is not supported')
            start_u, size_u = _bound_to_uop(self._ctx, start), _bound_to_uop(self._ctx, extent)
            view_shape.append(extent)
            owners.extend((start_u, size_u))
            if tensor is not None: owners.append(tensor)
            kinds.append(kind); starts.append(start_u.raw); sizes.append(size_u.raw); steps.append(step)
            tensors.append(tensor._tensor if tensor is not None else None)
        if any(step != 1 and step != -1 for step in steps) and _shape_has_symbolic(view_shape):
            raise RuntimeError('symbolic shape not supported for strided indexing')
        n = len(kinds)
        args = ((ctypes.c_int * n)(*kinds), (_ffi._ptr * n)(*starts), (_ffi._ptr * n)(*sizes),
                (ctypes.c_int64 * n)(*steps), (_ffi._ptr * n)(*tensors), n)
        return args, owners

    def __getitem__(self, indices):
        args, owners = self._index_args(indices)
        core = _ffi._lib.poly_tensor_getitem(self._ctx, self._tensor, *args)
        if not core:
            raise IndexError('cannot broadcast indices or unsupported indexing shape')
        if core == self._tensor:
            _ffi._lib.poly_tensor_release(core)
            return self
        return self._make_result_from_core(core, None, [self] + [x for x in owners if isinstance(x, Tensor)])

    def __setitem__(self, indices, value):
        if self.dtype in dtypes.weaks:
            raise RuntimeError('cannot setitem into a weak tensor; it has no storage')
        if not isinstance(value, Tensor):
            value = Tensor(value, dtype=self.dtype, device=self.device, _ctx=self._ctx)
        args, owners = self._index_args(indices)
        rc = _ffi._lib.poly_tensor_setitem(self._ctx, self._tensor, *args, value._tensor)
        if rc == -6:
            raise IndexError('cannot broadcast indices')
        if rc:
            raise RuntimeError({-2: "can't setitem on a tensor with other uses",
                                -3: 'setitem dtype mismatch',
                                -4: 'cannot setitem into a weak tensor; it has no storage',
                                -5: 'advanced setitem is not supported for DISK tensors'}.get(rc, 'cannot broadcast assigned value or unsupported indexing shape'))
        self._data = None

    def __delitem__(self, indices):
        raise TypeError('Tensor does not support deleting items')

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

    def full_like(self, fill_value, dtype=None, device=None, buffer=True):
        return Tensor.full(self.shape, fill_value, dtype=dtype or self.dtype,
                           device=self.device if device is None else device, buffer=buffer, _ctx=self._ctx)

    def zeros_like(self, **kwargs):
        return self.full_like(0, **kwargs)

    def ones_like(self, **kwargs):
        return self.full_like(1, **kwargs)

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
        ctx, dev = _creation_meta(kwargs)
        shape = (shape,) if isinstance(shape, int) else _shape_tuple(shape)
        fill_value = _py_scalar(fill_value)
        dtype_explicit = 'dtype' in kwargs
        inferred = kwargs.get('dtype', dtypes.from_py(fill_value))
        dtype_name = _dtype_name(inferred, default='float32')
        dtype_id = _dtype_id(dtype_name)
        dims, ndim, _ = _shape_arg(shape)

        dt = to_dtype(dtype_name)
        if fill_value is Invalid:
            tensor = _ffi._lib.poly_tensor_full_invalid_by_id(
                ctx, dims, ndim, dtype_id, _device_id(dev), buffer)
        elif dtypes.is_float(dt):
            tensor = _ffi._lib.poly_tensor_full_float_by_id(
                ctx, dims, ndim, float(fill_value), dtype_id, _device_id(dev),
                dtype_explicit, buffer,
            )
        else:
            factory = _ffi._lib.poly_tensor_full_uint_by_id if I64_MAX < fill_value < 2**64 else _ffi._lib.poly_tensor_full_int_by_id
            integer = fill_value if I64_MAX < fill_value < 2**64 else _require_i64(fill_value, 'fill_value')
            tensor = factory(
                ctx, dims, ndim, integer, dtype_id,
                _device_id(dev), dtype_explicit, buffer,
            )
        return _created_tensor(
            ctx, tensor, dtype_name, dev, 'poly_tensor_full_*_by_id'
        )

    @staticmethod
    def arange(start, stop=None, step=1, **kwargs):
        ctx, dev = _creation_meta(kwargs)
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
            ctx, tensor, dtype_name, dev, 'poly_tensor_arange_*_by_id'
        )

    @staticmethod
    def rand(*shape, **kwargs):
        ctx, dev = _creation_meta(kwargs)
        shape = _shape_tuple(*shape)
        dtype_name = _dtype_name(kwargs.get('dtype', dtypes.default_float), default='float32')
        dt = to_dtype(dtype_name)
        if not dtypes.is_float(dt):
            raise ValueError(f'rand only supports float dtypes, got {dt}')
        if any(not isinstance(s, int) or s < 0 for s in shape):
            raise ValueError(f'invalid input shape={shape}')
        dims, ndim, _ = _shape_arg(shape)
        tensor = _ffi._lib.poly_tensor_rand_by_id(
            ctx, dims, ndim, _dtype_id(dtype_name), _device_id(dev),
            int(bool(kwargs.get('contiguous', True))),
        )
        return _created_tensor(
            ctx, tensor, dtype_name, dev, 'poly_tensor_rand_by_id'
        )

    def rand_like(self, **kwargs):
        # Direct single-device port of pinned mixin/rand.py:70-86. Polygrad's
        # public Tensor wrapper currently exposes one device string per Tensor.
        return type(self).rand(
            *self.shape,
            device=kwargs.pop('device', self.device),
            dtype=kwargs.pop('dtype', self.dtype),
            **kwargs,
        )

    @staticmethod
    def randn(*shape, **kwargs):
        ctx, dev = _creation_meta(kwargs)
        shape = _shape_tuple(*shape)
        dtype_name = _dtype_name(kwargs.get('dtype', dtypes.default_float), default='float32')
        dt = to_dtype(dtype_name)
        if not dtypes.is_float(dt):
            raise ValueError(f'randn only supports float dtypes, got {dt}')
        if any(not isinstance(s, int) or s < 0 for s in shape):
            raise ValueError(f'invalid input shape={shape}')
        dims, ndim, _ = _shape_arg(shape)
        tensor = _ffi._lib.poly_tensor_randn_by_id(
            ctx, dims, ndim, _dtype_id(dtype_name), _device_id(dev)
        )
        return _created_tensor(
            ctx, tensor, dtype_name, dev, 'poly_tensor_randn_by_id'
        )

    @classmethod
    def normal(cls, *shape, mean=0.0, std=1.0, **kwargs):
        if std < 0:
            raise ValueError('std must be nonnegative')
        return std * cls.randn(*shape, **kwargs) + mean

    @classmethod
    def kaiming_normal(cls, *shape, a=0.01, **kwargs):
        shape = _shape_tuple(*shape)
        std = (2 / (1 + a**2) / _prod(shape[1:]))**0.5
        return cls.normal(*shape, mean=0.0, std=std, **kwargs)

    @classmethod
    def uniform(cls, *shape, low=0.0, high=1.0, **kwargs):
        """Create uniform values using pinned tinygrad's lazy rand/scale/add graph."""
        shape = _shape_tuple(*shape)
        if any(not isinstance(s, int) or s < 0 for s in shape):
            raise ValueError(f'invalid input shape={shape}')
        if low >= high:
            raise ValueError(f'Tensor.uniform requires low < high, got low={low}, high={high}')
        dtype = kwargs.get('dtype', dtypes.default_float)
        return ((high - low) * cls.rand(*shape, **kwargs)).cast(dtype) + low

    @classmethod
    def scaled_uniform(cls, *shape, **kwargs):
        """Create pinned tinygrad's product-scaled uniform initializer."""
        shape = _shape_tuple(*shape)
        return cls.uniform(*shape, low=-1.0, high=1.0, **kwargs).mul(_prod(shape) ** -0.5)

    @classmethod
    def glorot_uniform(cls, *shape, **kwargs):
        """Create pinned tinygrad's Glorot-uniform initializer."""
        shape = _shape_tuple(*shape)
        bound = math.sqrt(6.0 / (shape[0] + _prod(shape[1:])))
        return cls.uniform(*shape, low=-bound, high=bound, **kwargs)

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
        ctx, dev = _creation_meta(kwargs)
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
            ctx, tensor, dtype_name, dev, 'poly_tensor_linspace_by_id'
        )

    @staticmethod
    def eye(n, m=None, **kwargs):
        ctx, dev = _creation_meta(kwargs)
        dtype_name = _dtype_name(kwargs.get('dtype', dtypes.default_float), default='float32')
        rows = _require_i64(_py_scalar(n), 'n')
        cols = rows if m is None else _require_i64(_py_scalar(m), 'm')
        tensor = _ffi._lib.poly_tensor_eye_by_id(
            ctx, rows, cols, _dtype_id(dtype_name), _device_id(dev)
        )
        return _created_tensor(
            ctx, tensor, dtype_name, dev, 'poly_tensor_eye_by_id'
        )

    @staticmethod
    def invalids(*shape, **kwargs):
        return Tensor.full(_shape_tuple(*shape), Invalid, **kwargs)

    @staticmethod
    def empty(*shape, **kwargs):
        if 'name' in kwargs:
            raise TypeError('Tensor.empty does not accept name; pass names to Model.from_tensors')
        ctx, dev = _creation_meta(kwargs)
        dtype_name = _dtype_name(kwargs.get('dtype', dtypes.default_float), default='float32')
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        shape = tuple(_py_scalar(x) for x in shape)
        if any(_is_symbolic_dim(x) for x in shape):
            dims = _shape_uop_array(ctx, shape)
            tensor = _ffi._lib.poly_tensor_empty_uop_by_id(
                ctx, _dtype_id(dtype_name), dims, len(shape), _device_id(dev)
            )
            if not tensor:
                raise RuntimeError('poly_tensor_empty_uop_by_id failed')
            return Tensor(
                _ctx=ctx, _tensor=tensor, _shape=shape,
                _dtype=dtype_name, _device=dev,
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
            _dtype=dtype_name, _device=dev,
        )

    @staticmethod
    def manual_seed(seed=0):
        from . import _default_ctx
        _ffi._lib.poly_tensor_manual_seed(_default_ctx, _require_i64(int(seed), 'seed'))

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
        if isinstance(self, (list, tuple)) and tensors:
            raise ValueError('pass either a sequence or individual tensors')
        if isinstance(self, (list, tuple)) and not tensors:
            tensors = tuple(self)
        elif isinstance(self, Tensor):
            tensors = (self,) + tensors
        else:
            tensors = (self,) + tensors
        dim = tensors[0]._resolve_dim(dim, extra=True)
        assert all(t.shape == tensors[0].shape for t in tensors), 'stack shape mismatch'
        cores = (_ffi._ptr * len(tensors))(*(t._tensor for t in tensors))
        core = _ffi._lib.poly_tensor_stack(tensors[0]._ctx, cores, len(tensors), dim)
        return tensors[0]._make_result_from_core(core, None, tensors)

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

    def gradient(self, *targets, gradient=None):
        """Compute gradients of ``targets`` with respect to this Tensor."""
        if gradient is None:
            assert self.shape == (), (
                'when no gradient is provided, backward must be called on a scalar tensor'
            )
            initial_grad = None
        else:
            if not isinstance(gradient, Tensor):
                raise TypeError('gradient must be a Tensor')
            if _ptr_value(gradient._ctx) != _ptr_value(self._ctx):
                raise RuntimeError('gradient must share the differentiated Tensor context')
            initial_grad = Tensor._core_uop_raw(gradient._tensor)
        if not dtypes.is_float(self.dtype) or any(
            not isinstance(target, Tensor) or not dtypes.is_float(target.dtype)
            for target in targets
        ):
            raise RuntimeError('only float Tensors have gradient')
        if any(_ptr_value(target._ctx) != _ptr_value(self._ctx) for target in targets):
            raise RuntimeError('gradient targets must share the differentiated Tensor context')

        root = _uop_wrap(self._ctx, Tensor._core_uop_raw(self._tensor))
        target_roots = tuple(Tensor._core_uop_raw(target._tensor) for target in targets)
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
                self._ctx, root, initial_grad, temp_wrts, return_present=True
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
            out_present = tuple(
                present_by_root.get(_ptr_value(raw), False) for raw in target_roots
            )
        else:
            # tinygrad calls gradient(*targets), so every live target is handled by
            # one reverse pass. Calling poly_grad repeatedly can observe frontend
            # retargeting side effects between targets.
            out_grads, out_present = Tensor._grad_many_raw(
                self._ctx, root, initial_grad, target_roots, return_present=True
            )

        grads = []
        for target, grad_uop, present in zip(targets, out_grads, out_present):
            if not present:
                grads.append(target.const_like(0))
                continue
            if not grad_uop:
                raise RuntimeError('poly_grad_many returned NULL for a present target')
            grad_handle = target._core_create_result_like(
                grad_uop if target.uop_logical is not None else None,
                grad_uop, _POLY_TENSOR_VALUE, target._device,
            )
            if not grad_handle:
                raise RuntimeError('failed to store backward gradient roots')
            grad_tensor = Tensor(
                _ctx=self._ctx, _tensor=grad_handle, _shape=target.shape,
                _dtype=target._dtype_str, _device=target._device,
            )
            if int(_ffi._lib.poly_uop_device(grad_uop)) == 0:
                grad_tensor = grad_tensor.clone(device=target._device)
            grads.append(grad_tensor)
        return grads

    def backward(self, gradient=None):
        """Populate gradients for every live reachable floating Tensor."""
        target_entries = self._live_grad_targets()
        targets = tuple(target for target, _ in target_entries)
        for target, grad_tensor in zip(
            targets, self.gradient(*targets, gradient=gradient)
        ):
            assert grad_tensor.shape == target.shape, (
                f'grad shape must match tensor shape, {grad_tensor.shape!r} != {target.shape!r}'
            )
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
