"""
Tensor class for polygrad, lazy evaluation backed by the C compiler core.
Supports float32 (default) and float64 dtypes.
"""

import ctypes
import hashlib
import math
import weakref

import numpy as np

from . import _ffi
from .dtype import INVERSE_DTYPES_DICT, _from_np_dtype, _to_np_dtype, dtypes, to_dtype
from polygrad.uop.ops import UOp
from polygrad.device import Buffer


# Global registry of live Tensors. Keys are weakrefs so GC'd tensors vanish
# automatically. Used for post-realize retargeting and backward graph
# discovery, matching tinygrad's live-tensor registry.
all_tensors: dict[weakref.ref, None] = {}

# Strong frontend owner registry keyed by the C-side PolyBuffer* address value.
# This is intentionally keyed by the retired/imported residency object, not by
# BUFFER UOp identity. The core explicitly calls frontend_buffer_release(buffer)
# when a HOST PolyBuffer is retired, and only then do we drop the owner entry.
_host_buffers: dict[int, np.ndarray] = {}
_frontend_buffer_release_cb = None

_POLY_TENSOR_VALUE = 0
_POLY_TENSOR_PLACE = 1


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


def _symbolic_dim_raw(value):
    if isinstance(value, BoundVariable):
        return value.uop.raw
    if isinstance(value, Variable):
        return value.uop.raw
    if isinstance(value, UOp):
        return value.raw
    return None


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


def _created_tensor(ctx, uop, dtype_name, device, requires_grad, op_name):
    # Constructors should trust the core for final shape metadata so Python
    # does not grow a second copy of shape logic for the same helper.
    if not uop:
        raise RuntimeError(f'{op_name} failed')
    return Tensor(
        _ctx=ctx, _uop=uop, _shape=_shape_from_uop(ctx, uop),
        requires_grad=requires_grad, _dtype=dtype_name, _device=device,
    )


def _apply_map_to_tensors(ctx, replacements):
    """Apply realized-root substitutions to live frontend tensors.

    This mirrors tinygrad's _apply_map_to_tensors: after a realize call maps a
    lazy root to a realized buffer root, every live Tensor whose current UOp
    graph contains that root must be rewritten. Polygrad filters by core device
    because placement is stored on PolyTensor rather than inside the UOp.
    """
    if not replacements:
        return

    by_device = {}
    for old_raw, new_raw, device_id, realized_tensor in replacements:
        if not old_raw or not new_raw:
            continue
        by_device.setdefault(int(device_id), []).append(
            (old_raw, new_raw, realized_tensor)
        )
    if not by_device:
        return

    ctx_key = _ptr_value(ctx)
    stale_refs = []

    for tref in list(all_tensors):
        t = tref()
        if t is None:
            stale_refs.append(tref)
            continue
        if getattr(t, '_tensor', None) is None or _ptr_value(t._ctx) != ctx_key:
            continue

        device_id = int(_ffi._lib.poly_tensor_device(t._tensor))
        entries = by_device.get(device_id)
        if not entries:
            continue

        current_raw = Tensor._core_uop_raw(t._tensor)
        if not current_raw:
            continue

        n = len(entries)
        from_arr = (_ffi._ptr * n)(*[old for old, _, _ in entries])
        to_arr = (_ffi._ptr * n)(*[new for _, new, _ in entries])
        new_current = _ffi._lib.poly_uop_substitute(
            t._ctx, current_raw, from_arr, to_arr, n
        )
        if not new_current or _ptr_value(new_current) == _ptr_value(current_raw):
            continue

        rc = _ffi._lib.poly_tensor_update(
            t._ctx,
            t._tensor,
            None,
            new_current,
            _POLY_TENSOR_VALUE,
            device_id,
        )
        if rc != 0:
            raise RuntimeError('poly_tensor_update failed during live UOp retarget')
        t._data = None
        t._device = _device_name_from_id(device_id)

    for tref in stale_refs:
        all_tensors.pop(tref, None)


def _rng_device_tag(device):
    if not isinstance(device, str):
        raise ValueError(f'random constructors support only a single device, got {device!r}')
    return int.from_bytes(hashlib.sha256(device.encode('utf-8')).digest()[:8], 'big')


def _next_rng_seed(device):
    counter = Tensor._device_rng_counters.get(device, 0)
    Tensor._device_rng_counters[device] = counter + 1
    return (Tensor._seed + _rng_device_tag(device) + counter * 0x9E3779B97F4A7C15) & _U64_MASK


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
        self._ctx = _ctx or _default_ctx
        self._device = Device.canonicalize(_device if _device is not None else device)
        self._tensor = _tensor
        current_uop = None
        if self._tensor is not None and _uop is None:
            _uop = self._core_uop_raw(self._tensor)

        if _uop is not None:
            # Internal construction (from ops) -- shape is on the UOp.
            # Accept either a UOp instance or a raw ctypes pointer from FFI.
            current_uop = _uop if isinstance(_uop, UOp) else UOp(self._ctx, _uop)
            self._data = _data
            self._dtype_str = _dtype or 'float32'
        else:
            # User construction from data
            _ensure_frontend_buffer_release_registered(self._ctx)
            if isinstance(data, (int, float)):
                data = [data]
            if dtype is None and isinstance(data, np.ndarray):
                dt = _dtype_name(data.dtype, default='float32')
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
            if len(arr.shape) > 1:
                dims, ndim = _int64_array(arr.shape)
            else:
                dims, ndim = None, 0
            imported = UOp.from_host(
                self._ctx, self._data.ctypes.data, self._data.nbytes,
                dtype_id, dims, ndim,
            )
            current_uop = imported
            if post_cast_dt is not None:
                cast_uop = _ffi._lib.poly_cast_by_id(self._ctx, imported.raw, _dtype_id(post_cast_dt))
                if not cast_uop:
                    raise RuntimeError(f'poly_cast_by_id failed for dtype {post_cast_dt}')
                current_uop = UOp(self._ctx, cast_uop)
            key_uop = current_uop.buffer or current_uop
            key = _buffer_key(self._ctx, key_uop)
            if key:
                _host_buffers[key] = self._data

        self._requires_grad = bool(requires_grad)
        if self._tensor is None and current_uop is not None:
            self._tensor = self._core_create(current_uop, _POLY_TENSOR_VALUE, self._device)
        if self._requires_grad:
            self._sync_core_requires_grad(force=True)

        self._grad = None
        self._is_param = False     # True for model parameters (set by nn modules)
        all_tensors[weakref.ref(self)] = None

    # --- Core PolyTensor bridge ---

    @staticmethod
    def _core_create_for(ctx, uop, role, device):
        raw = _uop_raw(uop)
        if not raw:
            return None
        return _ffi._lib.poly_tensor_create(ctx, raw, int(role), _device_id(device))

    def _core_create(self, uop, role=_POLY_TENSOR_VALUE, device=None):
        return Tensor._core_create_for(
            self._ctx, uop, role, self._device if device is None else device
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
    def _core_realize_batch(ctx, targets):
        n = len(targets)
        in_arr = (_ffi._ptr * n)(*[t._tensor for t in targets])
        out_arr = (_ffi._ptr * n)()
        rc = _ffi._lib.poly_realize_tensors(ctx, in_arr, n, out_arr)
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

    @property
    def _graph_uop(self):
        """Root used to construct new lazy value graphs.

        Normally this is the logical/export root. After realized mutation
        effects, tinygrad's tensor root is the current BUFFER again; Polygrad
        mirrors that by using the physical root when the logical root is AFTER.
        """
        logical = self.uop_logical
        physical = self.uop_physical
        if logical and physical and logical.op == _ffi.OPS.get('AFTER'):
            return physical
        return logical or self.uop

    @staticmethod
    def _physicalize_result_for(ctx, logical, inputs):
        from_roots = []
        to_roots = []
        seen = set()
        for t in inputs:
            if not isinstance(t, Tensor):
                continue
            logical_root = _uop_raw(t._graph_uop)
            current_root = _uop_raw(t.uop)
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
        return raw.decode("utf-8").upper()

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
        ret._is_param = False
        ret._tensor = ret._core_create_with_roots(
            new_uop, self._physicalize_result(new_uop, srcs), _POLY_TENSOR_VALUE, ret._device
        )
        if ret._requires_grad:
            ret._sync_core_requires_grad(force=True)
        all_tensors[weakref.ref(ret)] = None
        return ret

    # --- Realization ---

    def _live_grad_targets(self):
        """Find gradient targets the same way tinygrad does.

        tinygrad has no per-tensor input graph. Polygrad scans the portable
        logical graph so realized/placed tensors remain valid gradient leaves.
        """
        targets = []
        stale_refs = []
        root = self._graph_uop
        ctx_key = _ptr_value(self._ctx)
        for tref in list(all_tensors):
            t = tref()
            if t is None:
                stale_refs.append(tref)
                continue
            if _ptr_value(t._ctx) != ctx_key or not t._requires_grad:
                continue
            target = t._graph_uop
            if target and _ffi._lib.poly_uop_reachable(self._ctx, root, target):
                targets.append(t)
        for tref in stale_refs:
            all_tensors.pop(tref, None)
        return targets

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
        buf = self.uop.buffer
        if buf is None:
            raise RuntimeError('copy_from requires a tensor backed by a BUFFER UOp')
        np_dt = _to_np_dtype(to_dtype(self._dtype_str))
        arr = np.asarray(data, dtype=np_dt)
        if arr.size != self.numel():
            raise ValueError(f'copy_from size mismatch {arr.size} != {self.numel()}')
        arr = np.ascontiguousarray(arr.reshape(self.shape))
        ptr = ctypes.c_void_p(arr.ctypes.data)
        rc = _ffi._lib.poly_buffer_write(self._ctx, buf.raw, ptr, arr.nbytes)
        if rc != 0:
            raise RuntimeError('poly_buffer_write failed')
        self._data = None
        return self

    def update_from(self, data):
        return self.copy_from(data)

    def realize(self, *lst, do_update_stats=True):
        """Triggers the computation needed to create these Tensor(s).
        Batches tensors into a single graph-side realize call, and retargets
        every live tensor whose current UOp graph contains a realized root."""
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
        old_roots = [Tensor._core_uop_raw(t._tensor) for t in targets]
        device_ids = [int(_ffi._lib.poly_tensor_device(t._tensor)) for t in targets]
        rc, realized_tensors = Tensor._core_realize_batch(self._ctx, targets)
        if rc != 0:
            devices = ', '.join(sorted({str(t._device) for t in targets}))
            raise RuntimeError(f'poly_realize_tensors failed for device(s): {devices}')
        replacements = []
        for old_raw, realized, device_id in zip(old_roots, realized_tensors, device_ids):
            new_raw = Tensor._core_uop_raw(realized)
            replacements.append((old_raw, new_raw, device_id, realized))
        _apply_map_to_tensors(self._ctx, replacements)
        return self

    def _buffer(self) -> Buffer:
        """Return the runtime Buffer backing this tensor.
        Materializes self via cast -> contiguous -> (to CPU if sharded) ->
        realize, then wraps the realized UOp's terminal buffer."""
        x = self.cast(self._dtype_str).contiguous()
        if isinstance(self._device, tuple):
            x = x.to("CPU")
        x.realize()
        return Buffer(self._ctx, x.uop.buffer, x._dtype_str, x.numel())

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
        return self.numpy().tolist()

    def detach(self):
        # Match tinygrad's graph-level detach: preserve the current value UOp,
        # but cut autograd flow without forcing a host readback/materialization.
        uop = _ffi._lib.poly_detach(self._ctx, self._graph_uop)
        if not uop:
            raise RuntimeError('poly_detach failed')
        return Tensor(
            _ctx=self._ctx,
            _tensor=self._core_create_with_roots(
                uop, self._physicalize_result(uop, [self]), _POLY_TENSOR_VALUE, self._device
            ),
            _shape=self.shape,
            _dtype=self._dtype_str,
            _device=self._device,
            requires_grad=False,
        )

    def clone(self):
        return Tensor(
            self.numpy(),
            requires_grad=self._requires_grad,
            dtype=self._dtype_str,
            device=self._device,
        )

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

    def cpu(self):
        return self.to('cpu')

    def cuda(self):
        return self.to('cuda')

    def contiguous(self, *args, **kwargs):
        """Returns a contiguous tensor."""
        return self._apply_uop(UOp.contiguous, extra_args=args, **kwargs)

    # --- Dtype casting ---

    def cast(self, dtype):
        """Cast tensor to the given dtype. No-op if already that dtype."""
        target_name = _dtype_name(dtype, default=self._dtype_str)
        if target_name == self._dtype_str:
            return self
        dtype_id = _dtype_id(target_name)
        uop = _ffi._lib.poly_cast_by_id(self._ctx, self._graph_uop, dtype_id)
        if not uop:
            raise RuntimeError(f'poly_cast_by_id failed for dtype {target_name}')
        return Tensor(
            _ctx=self._ctx,
            _tensor=self._core_create_with_roots(
                uop, self._physicalize_result(uop, [self]), _POLY_TENSOR_VALUE, self._device
            ),
            _shape=self.shape,
            _dtype=target_name,
            _device=self._device,
            requires_grad=self._requires_grad,
        )

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
        return Tensor(
            _ctx=self._ctx,
            _tensor=self._core_create_with_roots(uop, physical, _POLY_TENSOR_VALUE, dev),
            _shape=shape,
            requires_grad=any(t._requires_grad for t in inputs),
            _dtype=dt,
            _device=dev,
        )

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
            dt = to_dtype(self._dtype_str)
            # Normalize the Python scalar through the active tensor dtype first
            # so implicit constants stop bypassing the same dtype rules as core
            # constructors and comparisons.
            normalized = dt.const(other)
            dtype_id = _dtype_id(self._dtype_str)
            if dtypes.is_bool(dt) or dtypes.is_int(dt):
                int_value = int(bool(normalized)) if dtypes.is_bool(dt) else int(normalized)
                if int_value < I64_MIN or int_value > I64_MAX:
                    raise ValueError(f'scalar {int_value} is out of int64 range')
                c = _ffi._lib.poly_const_int_by_id(self._ctx, int_value, dtype_id)
            else:
                c = _ffi._lib.poly_const_float_by_id(self._ctx, float(normalized), dtype_id)
            return Tensor(_ctx=self._ctx, _uop=c, _shape=(),
                          _dtype=self._dtype_str, _device=self._device)
        raise TypeError(f'Cannot convert {type(other)} to Tensor')

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
                raise ValueError(f'Cannot broadcast shapes {self.shape} and {other_shape}')
        return tuple(result)

    def _broadcast_uop(self, target_shape):
        """Return a UOp that broadcasts self to target_shape via RESHAPE+EXPAND.

        Matches tinygrad's _broadcast_to: explicit shape ops so the scheduler
        sees EXPAND UOps instead of implicit ALU broadcasting.
        """
        if self.shape == target_shape:
            return self._graph_uop
        uop = self._graph_uop
        cur_shape = self.shape
        # CONST scalars (from _ensure_tensor) auto-broadcast — no EXPAND needed
        if not cur_shape and self._graph_uop.buffer is None:
            return uop
        if not _shape_all_int(target_shape) or not _shape_all_int(cur_shape):
            raise NotImplementedError('symbolic movement broadcast is not implemented')
        # Scalar tensor or lower-rank: pad left with 1s
        target_nd = len(target_shape)
        if len(cur_shape) < target_nd:
            cur_shape = (1,) * (target_nd - len(cur_shape)) + cur_shape
            dims, n = _int64_array(cur_shape)
            uop = _ffi._lib.poly_reshape(self._ctx, uop, dims, n)
        # Expand any dimensions where size 1 → target size
        if cur_shape != target_shape:
            dims, n = _int64_array(target_shape)
            uop = _ffi._lib.poly_expand(self._ctx, uop, dims, n)
        return uop

    # --- Element-wise arithmetic ---

    def _binop(self, other, op_name):
        """Binary op with explicit EXPAND broadcasting (matches tinygrad _broadcasted)."""
        other = self._ensure_tensor(other)
        out_shape = self._broadcast_shape(other.shape)
        x_uop = self._broadcast_uop(out_shape)
        y_uop = other._broadcast_uop(out_shape)
        uop = _ffi._lib.poly_alu2(self._ctx, _ffi.OPS[op_name], x_uop, y_uop)
        return self._make_result(uop, out_shape, [self, other])

    def __add__(self, other):
        return self._binop(other, 'ADD')

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._binop(other, 'SUB')

    def __rsub__(self, other):
        other = self._ensure_tensor(other)
        return other.__sub__(self)

    def __mul__(self, other):
        return self._binop(other, 'MUL')

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._binop(other, 'FDIV')

    def __rtruediv__(self, other):
        other = self._ensure_tensor(other)
        return other.__truediv__(self)

    def __neg__(self):
        uop = _ffi._lib.poly_alu1(self._ctx, _ffi.OPS['NEG'], self._graph_uop)
        return self._make_result(uop, self.shape, [self])

    def __pow__(self, other):
        return self._binop(other, 'POW')

    # --- Comparisons (C core) ---

    def __lt__(self, other):
        return self._binop(other, 'CMPLT')

    def __eq__(self, other):
        other = self._ensure_tensor(other)
        out_shape = self._broadcast_shape(other.shape)
        uop = _ffi._lib.poly_eq(self._ctx, self._broadcast_uop(out_shape),
                                 other._broadcast_uop(out_shape))
        return self._make_result(uop, out_shape, [self, other])

    def __ne__(self, other):
        other = self._ensure_tensor(other)
        out_shape = self._broadcast_shape(other.shape)
        uop = _ffi._lib.poly_ne(self._ctx, self._broadcast_uop(out_shape),
                                 other._broadcast_uop(out_shape))
        return self._make_result(uop, out_shape, [self, other])

    def __gt__(self, other):
        other = self._ensure_tensor(other)
        out_shape = self._broadcast_shape(other.shape)
        uop = _ffi._lib.poly_gt(self._ctx, self._broadcast_uop(out_shape),
                                 other._broadcast_uop(out_shape))
        return self._make_result(uop, out_shape, [self, other])

    def __ge__(self, other):
        other = self._ensure_tensor(other)
        out_shape = self._broadcast_shape(other.shape)
        uop = _ffi._lib.poly_ge(self._ctx, self._broadcast_uop(out_shape),
                                 other._broadcast_uop(out_shape))
        return self._make_result(uop, out_shape, [self, other])

    def __le__(self, other):
        other = self._ensure_tensor(other)
        out_shape = self._broadcast_shape(other.shape)
        uop = _ffi._lib.poly_le(self._ctx, self._broadcast_uop(out_shape),
                                 other._broadcast_uop(out_shape))
        return self._make_result(uop, out_shape, [self, other])

    def ne(self, other):
        return self.__ne__(other)

    def eq(self, other):
        return self.__eq__(other)

    def where(self, x, y):
        """self is condition: where(cond, x, y)."""
        x = self._ensure_tensor(x)
        y = self._ensure_tensor(y)
        out_shape = _broadcast_shapes(_broadcast_shapes(self.shape, x.shape), y.shape)
        c_uop = self._broadcast_uop(out_shape)
        x_uop = x._broadcast_uop(out_shape)
        y_uop = y._broadcast_uop(out_shape)
        uop = _ffi._lib.poly_where_op(self._ctx, c_uop, x_uop, y_uop)
        return self._make_result(uop, out_shape, [self, x, y])

    def maximum(self, other):
        other = self._ensure_tensor(other)
        out_shape = self._broadcast_shape(other.shape)
        uop = _ffi._lib.poly_maximum(self._ctx, self._broadcast_uop(out_shape),
                                      other._broadcast_uop(out_shape))
        return self._make_result(uop, out_shape, [self, other])

    def minimum(self, other):
        other = self._ensure_tensor(other)
        out_shape = self._broadcast_shape(other.shape)
        uop = _ffi._lib.poly_minimum(self._ctx, self._broadcast_uop(out_shape),
                                      other._broadcast_uop(out_shape))
        return self._make_result(uop, out_shape, [self, other])

    def clamp(self, min_=None, max_=None):
        if min_ is None and max_ is None:
            raise RuntimeError("at least one of 'min_' or 'max_' must not be None")
        lo = min_ if min_ is not None else -1e38
        hi = max_ if max_ is not None else 1e38
        uop = _ffi._lib.poly_clamp(self._ctx, self._graph_uop, float(lo), float(hi))
        return self._make_result(uop, self.shape, [self])

    # --- Unary math (C core composed ops) ---

    def exp2(self):
        uop = _ffi._lib.poly_alu1(self._ctx, _ffi.OPS['EXP2'], self._graph_uop)
        return self._make_result(uop, self.shape, [self])

    def log2(self):
        uop = _ffi._lib.poly_alu1(self._ctx, _ffi.OPS['LOG2'], self._graph_uop)
        return self._make_result(uop, self.shape, [self])

    def sqrt(self):
        uop = _ffi._lib.poly_alu1(self._ctx, _ffi.OPS['SQRT'], self._graph_uop)
        return self._make_result(uop, self.shape, [self])

    def reciprocal(self):
        uop = _ffi._lib.poly_alu1(self._ctx, _ffi.OPS['RECIPROCAL'], self._graph_uop)
        return self._make_result(uop, self.shape, [self])

    def trunc(self):
        uop = _ffi._lib.poly_alu1(self._ctx, _ffi.OPS['TRUNC'], self._graph_uop)
        return self._make_result(uop, self.shape, [self])

    def exp(self):
        uop = _ffi._lib.poly_exp(self._ctx, self._graph_uop)
        return self._make_result(uop, self.shape, [self])

    def log(self):
        uop = _ffi._lib.poly_log(self._ctx, self._graph_uop)
        return self._make_result(uop, self.shape, [self])

    def sin(self):
        uop = _ffi._lib.poly_sin(self._ctx, self._graph_uop)
        return self._make_result(uop, self.shape, [self])

    def cos(self):
        uop = _ffi._lib.poly_cos(self._ctx, self._graph_uop)
        return self._make_result(uop, self.shape, [self])

    def tan(self):
        uop = _ffi._lib.poly_tan(self._ctx, self._graph_uop)
        return self._make_result(uop, self.shape, [self])

    def sigmoid(self):
        uop = _ffi._lib.poly_sigmoid(self._ctx, self._graph_uop)
        return self._make_result(uop, self.shape, [self])

    def tanh(self):
        uop = _ffi._lib.poly_tanh_act(self._ctx, self._graph_uop)
        return self._make_result(uop, self.shape, [self])

    def abs(self):
        uop = _ffi._lib.poly_abs(self._ctx, self._graph_uop)
        return self._make_result(uop, self.shape, [self])

    def sign(self):
        uop = _ffi._lib.poly_sign(self._ctx, self._graph_uop)
        return self._make_result(uop, self.shape, [self])

    def square(self):
        uop = _ffi._lib.poly_square(self._ctx, self._graph_uop)
        return self._make_result(uop, self.shape, [self])

    def rsqrt(self):
        uop = _ffi._lib.poly_rsqrt(self._ctx, self._graph_uop)
        return self._make_result(uop, self.shape, [self])

    def ceil(self):
        uop = _ffi._lib.poly_ceil(self._ctx, self._graph_uop)
        return self._make_result(uop, self.shape, [self])

    def floor(self):
        uop = _ffi._lib.poly_floor(self._ctx, self._graph_uop)
        return self._make_result(uop, self.shape, [self])

    def round(self):
        uop = _ffi._lib.poly_round_f(self._ctx, self._graph_uop)
        return self._make_result(uop, self.shape, [self])

    def isinf(self):
        uop = _ffi._lib.poly_isinf(self._ctx, self._graph_uop)
        return self._make_result(uop, self.shape, [self])

    def isnan(self):
        uop = _ffi._lib.poly_isnan(self._ctx, self._graph_uop)
        return self._make_result(uop, self.shape, [self])

    # --- Activations (C core composed ops) ---

    def relu(self):
        uop = _ffi._lib.poly_relu(self._ctx, self._graph_uop)
        return self._make_result(uop, self.shape, [self])

    def relu6(self):
        uop = _ffi._lib.poly_relu6(self._ctx, self._graph_uop)
        return self._make_result(uop, self.shape, [self])

    def leaky_relu(self, neg_slope=0.01):
        uop = _ffi._lib.poly_leaky_relu(self._ctx, self._graph_uop, neg_slope)
        return self._make_result(uop, self.shape, [self])

    def gelu(self):
        uop = _ffi._lib.poly_gelu(self._ctx, self._graph_uop)
        return self._make_result(uop, self.shape, [self])

    def quick_gelu(self):
        uop = _ffi._lib.poly_quick_gelu(self._ctx, self._graph_uop)
        return self._make_result(uop, self.shape, [self])

    def silu(self):
        uop = _ffi._lib.poly_silu(self._ctx, self._graph_uop)
        return self._make_result(uop, self.shape, [self])

    def swish(self):
        return self.silu()

    def elu(self, alpha=1.0):
        uop = _ffi._lib.poly_elu(self._ctx, self._graph_uop, alpha)
        return self._make_result(uop, self.shape, [self])

    def softplus(self, beta=1.0):
        uop = _ffi._lib.poly_softplus(self._ctx, self._graph_uop, beta)
        return self._make_result(uop, self.shape, [self])

    def mish(self):
        uop = _ffi._lib.poly_mish(self._ctx, self._graph_uop)
        return self._make_result(uop, self.shape, [self])

    def hardtanh(self, min_val=-1, max_val=1):
        uop = _ffi._lib.poly_hardtanh(self._ctx, self._graph_uop, min_val, max_val)
        return self._make_result(uop, self.shape, [self])

    def hardswish(self):
        uop = _ffi._lib.poly_hardswish(self._ctx, self._graph_uop)
        return self._make_result(uop, self.shape, [self])

    def hardsigmoid(self):
        uop = _ffi._lib.poly_hardsigmoid(self._ctx, self._graph_uop)
        return self._make_result(uop, self.shape, [self])

    # --- Softmax ---

    def softmax(self, axis=-1):
        # Keep softmax lazy so backward sees the same current UOp graph as
        # tinygrad; scheduling decides where kernel boundaries belong.
        uop = _ffi._lib.poly_softmax(self._ctx, self._graph_uop, int(axis))
        if not uop:
            raise RuntimeError('poly_softmax failed')
        return self._make_result(uop, self.shape, [self])

    def log_softmax(self, axis=-1):
        uop = _ffi._lib.poly_log_softmax(self._ctx, self._graph_uop, int(axis))
        if not uop:
            raise RuntimeError('poly_log_softmax failed')
        return self._make_result(uop, self.shape, [self])

    # --- Movement ops ---

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        # Support -1 dimension inference
        if -1 in shape:
            total = self.numel()
            neg_idx = shape.index(-1)
            known = 1
            for i, s in enumerate(shape):
                if i != neg_idx:
                    known *= s
            shape = tuple(total // known if i == neg_idx else s for i, s in enumerate(shape))
        arr, n = _int64_array(shape)
        uop = _ffi._lib.poly_reshape(self._ctx, self._graph_uop, arr, n)
        return self._make_result(uop, shape, [self])

    def permute(self, *order):
        if len(order) == 1 and isinstance(order[0], (tuple, list)):
            order = tuple(order[0])
        arr, n = _int64_array(order)
        uop = _ffi._lib.poly_permute(self._ctx, self._graph_uop, arr, n)
        new_shape = tuple(self.shape[i] for i in order)
        return self._make_result(uop, new_shape, [self])

    def expand(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        arr, n = _int64_array(shape)
        uop = _ffi._lib.poly_expand(self._ctx, self._graph_uop, arr, n)
        return self._make_result(uop, shape, [self])

    def shrink(self, arg):
        """arg is tuple of (start, end) pairs per dimension."""
        flat, n = _pair_array(arg)
        uop = _ffi._lib.poly_shrink(self._ctx, self._graph_uop, flat, n)
        new_shape = tuple(e - s for s, e in arg)
        return self._make_result(uop, new_shape, [self])

    def pad(self, arg):
        """arg is tuple of (before, after) pairs per dimension."""
        flat, n = _pair_array(arg)
        uop = _ffi._lib.poly_pad(self._ctx, self._graph_uop, flat, n)
        new_shape = tuple(s + b + a for s, (b, a) in zip(self.shape, arg))
        return self._make_result(uop, new_shape, [self])

    def flip(self, axis):
        if isinstance(axis, int):
            axis = (axis,)
        arr, n = _int64_array(axis)
        uop = _ffi._lib.poly_flip(self._ctx, self._graph_uop, arr, n)
        return self._make_result(uop, self.shape, [self])

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
    def _tri(r, c, diagonal=0):
        return (Tensor.arange(r).unsqueeze(-1) + diagonal) <= Tensor.arange(c)

    def triu(self, diagonal=0):
        r, c = self.shape[-2], self.shape[-1]
        mask = Tensor._tri(r, c, diagonal=diagonal)
        return mask.where(self, Tensor(0.0))

    def tril(self, diagonal=0):
        r, c = self.shape[-2], self.shape[-1]
        mask = Tensor._tri(r, c, diagonal=diagonal + 1)
        return mask.where(Tensor(0.0), self)

    def squeeze(self, dim=None):
        if dim is not None:
            if dim < 0:
                dim += len(self.shape)
            if self.shape[dim] != 1:
                return self
            new_shape = tuple(s for i, s in enumerate(self.shape) if i != dim)
            if not new_shape:
                new_shape = (1,)
            return self.reshape(new_shape)
        new_shape = tuple(s for s in self.shape if s != 1)
        if not new_shape:
            new_shape = (1,)
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
        # Normalize negative axes
        nd = self.ndim
        axis = tuple(a + nd if a < 0 else a for a in axis)

        arr, n = _int64_array(axis)
        uop = _ffi._lib.poly_reduce_axis(self._ctx, _ffi.OPS['ADD'], self._graph_uop, arr, n)

        # REDUCE_AXIS keeps all dims (reduced→1). If keepdim=False, reshape to squeeze.
        if not keepdim and axis:
            new_shape = tuple(s for i, s in enumerate(self.shape) if i not in axis)
            if not new_shape:
                new_shape = ()
            dims, ndim = _int64_array(new_shape) if new_shape else (None, 0)
            uop = _ffi._lib.poly_reshape(self._ctx, uop, dims, ndim)
            return self._make_result(uop, new_shape, [self])

        # keepdim=True: shape has 1 at reduced axes
        new_shape = tuple(1 if i in axis else s for i, s in enumerate(self.shape))
        return self._make_result(uop, new_shape, [self])

    def max(self, axis=None, keepdim=False):
        if axis is None:
            # Reduce all
            result = self
            for i in range(len(self.shape) - 1, -1, -1):
                uop = _ffi._lib.poly_max_reduce(self._ctx, result._graph_uop,
                                                  i, int(keepdim))
                new_shape = _shape_from_uop(self._ctx, uop)
                result = self._make_result(uop, new_shape, [result])
            return result
        # Normalize negative axis
        nd = len(self.shape)
        if axis < 0:
            axis = axis + nd
        uop = _ffi._lib.poly_max_reduce(self._ctx, self._graph_uop,
                                          axis, int(keepdim))
        new_shape = _shape_from_uop(self._ctx, uop)
        result = self._make_result(uop, new_shape, [self])
        return result

    def argmax(self, axis=None, keepdim=False):
        if axis is None:
            return self.flatten().argmax(0)
        axis = self._resolve_dim(int(axis))
        uop = _ffi._lib.poly_argmax(self._ctx, self._graph_uop, axis)
        if not uop:
            raise RuntimeError('poly_argmax failed')
        shape = _shape_from_uop(self._ctx, uop)
        result = self._make_result(uop, shape, [self])
        if keepdim:
            keep_shape = list(self.shape)
            keep_shape[axis] = 1
            result = result.reshape(tuple(keep_shape))
        return result

    def sort(self, dim=-1, descending=False):
        dim = self._resolve_dim(int(dim))
        values = _ffi._ptr()
        indices = _ffi._ptr()
        rc = _ffi._lib.poly_sort(
            self._ctx, self._graph_uop, dim, int(bool(descending)),
            ctypes.byref(values), ctypes.byref(indices)
        )
        if rc != 0 or not values or not indices:
            raise RuntimeError('poly_sort failed')
        val_shape = _shape_from_uop(self._ctx, values)
        idx_shape = _shape_from_uop(self._ctx, indices)
        vals = self._make_result(values, val_shape, [self])
        idx = Tensor(
            _ctx=self._ctx,
            _tensor=self._core_create_with_roots(
                indices, self._physicalize_result(indices, [self]), _POLY_TENSOR_VALUE, self._device
            ),
            _shape=idx_shape,
            _dtype=_uop_dtype_name(self._ctx, indices, 'int32'),
            _device=self._device,
            requires_grad=False,
        )
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
        rc = _ffi._lib.poly_topk(
            self._ctx, self._graph_uop, int(k), dim, int(bool(largest)), int(bool(sorted_)),
            ctypes.byref(values), ctypes.byref(indices)
        )
        if rc != 0 or not values or not indices:
            raise RuntimeError('poly_topk failed')
        val_shape = _shape_from_uop(self._ctx, values)
        idx_shape = _shape_from_uop(self._ctx, indices)
        vals = self._make_result(values, val_shape, [self])
        idx = Tensor(
            _ctx=self._ctx,
            _tensor=self._core_create_with_roots(
                indices, self._physicalize_result(indices, [self]), _POLY_TENSOR_VALUE, self._device
            ),
            _shape=idx_shape,
            _dtype=_uop_dtype_name(self._ctx, indices, 'int32'),
            _device=self._device,
            requires_grad=False,
        )
        return vals, idx

    def min(self, axis=None, keepdim=False):
        return (-self).max(axis=axis, keepdim=keepdim).__neg__()

    def mean(self, axis=None, keepdim=False):
        if axis is None:
            return self.sum(keepdim=keepdim) / self.numel()
        uop = _ffi._lib.poly_mean_reduce(self._ctx, self._graph_uop,
                                           axis, int(keepdim))
        new_shape = _shape_from_uop(self._ctx, uop)
        return self._make_result(uop, new_shape, [self])

    def var(self, axis=None, keepdim=False, correction=1):
        if axis is None:
            m = self.mean()
            diff = self - m
            sq = diff * diff
            return sq.sum() / (self.numel() - correction)
        if isinstance(axis, int):
            if axis < 0:
                axis += len(self.shape)
        # Keep variance construction lazy like tinygrad. The old frontend
        # realized the mean as a scheduler workaround, which made var/layernorm
        # invisible to backward across that forced materialization boundary.
        uop = _ffi._lib.poly_var_reduce(
            self._ctx, self._graph_uop, int(axis), int(keepdim), int(correction)
        )
        if not uop:
            raise RuntimeError('poly_var_reduce failed')
        return self._make_result(uop, _shape_from_uop(self._ctx, uop), [self])

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

        uop = _ffi._lib.poly_gather_dim(self._ctx, self._graph_uop, dim, index._graph_uop)
        if not uop:
            raise RuntimeError('poly_gather_dim failed')
        return self._make_result(uop, _shape_from_uop(self._ctx, uop), [self, index])

    def take_along_axis(self, index, axis):
        return self.gather(axis, index)

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
        uop = _ffi._lib.poly_scatter_reduce(
            self._ctx, self._graph_uop, dim, index._graph_uop, src._graph_uop,
            reduce.encode('utf-8'), int(bool(include_self))
        )
        if not uop:
            raise RuntimeError('poly_scatter_reduce failed')
        return self._make_result(uop, _shape_from_uop(self._ctx, uop), [self, index, src])

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
        uop = _ffi._lib.poly_scatter(self._ctx, self._graph_uop, dim, index._graph_uop, src._graph_uop, reduce_arg)
        if not uop:
            raise RuntimeError('poly_scatter failed')
        return self._make_result(uop, _shape_from_uop(self._ctx, uop), [self, index, src])

    # --- Matmul (C core dot) ---

    def dot(self, w):
        if not isinstance(w, Tensor):
            raise TypeError(f'Expected Tensor, got {type(w)}')
        uop = _ffi._lib.poly_dot(self._ctx, self._graph_uop, w._graph_uop)
        if not uop:
            raise ValueError(f'cannot dot {self.shape} and {w.shape}')
        new_shape = _shape_from_uop(self._ctx, uop)
        return self._make_result(uop, new_shape, [self, w])

    def matmul(self, other):
        return self.dot(other)

    def qr(self, mode='complete'):
        mode_id = {'complete': 0, 'reduced': 1, 'r': 2}.get(mode)
        if mode_id is None:
            raise ValueError("qr mode must be 'complete', 'reduced', or 'r'")
        q = _ffi._ptr()
        r = _ffi._ptr()
        rc = _ffi._lib.poly_qr_ex(self._ctx, self._graph_uop, mode_id, ctypes.byref(q), ctypes.byref(r))
        if rc != 0 or not r or (mode_id != 2 and not q):
            raise RuntimeError('poly_qr_ex failed')
        if mode_id == 2:
            return self._make_result(r, _shape_from_uop(self._ctx, r), [self])
        return (
            self._make_result(q, _shape_from_uop(self._ctx, q), [self]),
            self._make_result(r, _shape_from_uop(self._ctx, r), [self]),
        )

    def triangular_solve(self, b, upper=False, transpose_a=False, unit_diagonal=False):
        if not isinstance(b, Tensor):
            b = self._ensure_tensor(b)
        uop = _ffi._lib.poly_triangular_solve(
            self._ctx, self._graph_uop, b._graph_uop,
            int(bool(upper)), int(bool(transpose_a)), int(bool(unit_diagonal))
        )
        if not uop:
            raise ValueError(
                f'cannot triangular_solve A.shape={self.shape} and b.shape={b.shape}'
            )
        return self._make_result(uop, _shape_from_uop(self._ctx, uop), [self, b])

    def solve_triangular(self, b, upper=False, transpose_a=False, unit_diagonal=False):
        return self.triangular_solve(b, upper, transpose_a, unit_diagonal)

    def cholesky(self, upper=False):
        uop = _ffi._lib.poly_cholesky(self._ctx, self._graph_uop, int(bool(upper)))
        if not uop:
            raise ValueError(f'cannot cholesky shape={self.shape}')
        return self._make_result(uop, _shape_from_uop(self._ctx, uop), [self])

    def cholesky_solve(self, b, upper=False):
        if not isinstance(b, Tensor):
            b = self._ensure_tensor(b)
        uop = _ffi._lib.poly_cholesky_solve(self._ctx, self._graph_uop, b._graph_uop, int(bool(upper)))
        if not uop:
            raise ValueError(
                f'cannot cholesky_solve factor.shape={self.shape} and b.shape={b.shape}'
            )
        return self._make_result(uop, _shape_from_uop(self._ctx, uop), [self, b])

    def solve(self, b):
        if not isinstance(b, Tensor):
            b = self._ensure_tensor(b)
        uop = _ffi._lib.poly_solve(self._ctx, self._graph_uop, b._graph_uop)
        if not uop:
            raise ValueError(f'cannot solve A.shape={self.shape} and b.shape={b.shape}')
        return self._make_result(uop, _shape_from_uop(self._ctx, uop), [self, b])

    def lstsq(self, b):
        if not isinstance(b, Tensor):
            b = self._ensure_tensor(b)
        uop = _ffi._lib.poly_lstsq(self._ctx, self._graph_uop, b._graph_uop)
        if not uop:
            raise ValueError(f'cannot lstsq A.shape={self.shape} and b.shape={b.shape}')
        return self._make_result(uop, _shape_from_uop(self._ctx, uop), [self, b])

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

    # --- Loss functions ---

    def cross_entropy(self, target, axis=None):
        """Cross-entropy loss for class-index or same-shape probability targets."""
        if not isinstance(target, Tensor):
            target = Tensor(target, device=self._device)
        target = target.to(self._device)
        if axis is None:
            axis = 0 if self.ndim == 1 else 1
        uop = _ffi._lib.poly_cross_entropy(
            self._ctx, self._graph_uop, target._graph_uop, axis
        )
        if not uop:
            raise ValueError(f'shape mismatch: self.shape={self.shape}, target.shape={target.shape}')
        new_shape = _shape_from_uop(self._ctx, uop)
        return self._make_result(uop, new_shape, [self, target])

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
            elif isinstance(i, slice):
                size = result.shape[dim]
                start, stop, step = i.indices(size)
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
                result = result.shrink(
                    tuple(tuple(boundary) if d == dim else (0, s)
                          for d, s in enumerate(result.shape)))
                # flip if negative stride
                if stride < 0:
                    result = result.flip(dim)
                abs_stride = abs(stride)
                # apply stride via pad+reshape+shrink+reshape
                if abs_stride != 1:
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
        ctx = operands[0]._ctx
        n = len(operands)

        tensor_arr = (_ffi._ptr * n)(*[_uop_raw(t._graph_uop) for t in operands])

        uop = _ffi._lib.poly_einsum(
            ctx, formula.encode('utf-8'),
            tensor_arr, n)
        if not uop:
            raise ValueError(f'poly_einsum failed for formula: {formula}')
        shape = _shape_from_uop(ctx, uop)
        dev = operands[0]._infer_device(list(operands))
        physical = Tensor._physicalize_result_for(ctx, uop, operands)
        return Tensor(
            _ctx=ctx,
            _tensor=Tensor._core_create_with_roots_for(
                ctx, uop, physical, _POLY_TENSOR_VALUE, dev
            ),
            _shape=shape, _device=dev,
            requires_grad=any(t._requires_grad for t in operands),
            _dtype=_uop_dtype_name(ctx, uop, operands[0]._dtype_str),
        )

    # --- Rearrange (C core, einops-style) ---

    def rearrange(self, formula, **kwargs):
        """Einops-style rearrange. E.g. t.rearrange('b c h w -> b (c h) w')."""
        names = list(kwargs.keys())
        values = [kwargs[k] for k in names]
        n = len(names)

        axis_names = ' '.join(names).encode('utf-8') if names else None
        axis_values = (ctypes.c_int64 * n)(*values) if n > 0 else None

        uop = _ffi._lib.poly_rearrange(
            self._ctx, formula.encode('utf-8'),
            self._graph_uop,
            axis_names, axis_values, n)
        if not uop:
            raise ValueError(f'poly_rearrange failed for formula: {formula}')
        shape = _shape_from_uop(self._ctx, uop)
        return self._make_result(uop, shape, [self])

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
        ctx, dev, requires_grad = _creation_meta(kwargs)
        shape = (shape,) if isinstance(shape, int) else _shape_tuple(shape)
        fill_value = _py_scalar(fill_value)
        inferred = kwargs.get('dtype', dtypes.from_py(fill_value))
        dtype_name = _dtype_name(inferred, default='float32')
        dtype_id = _dtype_id(dtype_name)
        dims, ndim, _ = _shape_arg(shape)

        dt = to_dtype(dtype_name)
        if dtypes.is_float(dt):
            uop = _ffi._lib.poly_full_float_by_id(ctx, dims, ndim, float(fill_value), dtype_id)
        else:
            uop = _ffi._lib.poly_full_int_by_id(
                ctx, dims, ndim, _require_i64(fill_value, 'fill_value'), dtype_id
            )
        return _created_tensor(ctx, uop, dtype_name, dev, requires_grad, 'poly_full_*_by_id')

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
            uop = _ffi._lib.poly_arange_float_by_id(
                ctx, float(start), float(stop), float(step), dtype_id
            )
        else:
            uop = _ffi._lib.poly_arange_int_by_id(
                ctx,
                _require_i64(start, 'start'),
                _require_i64(stop, 'stop'),
                _require_i64(step, 'step'),
                dtype_id,
            )
        return _created_tensor(ctx, uop, dtype_name, dev, requires_grad, 'poly_arange_*_by_id')

    @staticmethod
    def rand(*shape, **kwargs):
        ctx, dev, requires_grad = _creation_meta(kwargs)
        shape = _shape_tuple(*shape)
        dtype_name = _dtype_name(kwargs.get('dtype', dtypes.default_float), default='float32')
        dt = to_dtype(dtype_name)
        if not dtypes.is_float(dt):
            raise ValueError(f'rand only supports float dtypes, got {dt}')
        dims, ndim, _ = _shape_arg(shape)
        uop = _ffi._lib.poly_rand_by_id(ctx, dims, ndim, _next_rng_seed(dev), _dtype_id(dtype_name))
        return _created_tensor(ctx, uop, dtype_name, dev, requires_grad, 'poly_rand_by_id')

    @staticmethod
    def randn(*shape, **kwargs):
        ctx, dev, requires_grad = _creation_meta(kwargs)
        shape = _shape_tuple(*shape)
        dtype_name = _dtype_name(kwargs.get('dtype', dtypes.default_float), default='float32')
        dt = to_dtype(dtype_name)
        if not dtypes.is_float(dt):
            raise ValueError(f'randn only supports float dtypes, got {dt}')
        dims, ndim, _ = _shape_arg(shape)
        uop = _ffi._lib.poly_randn_by_id(ctx, dims, ndim, _next_rng_seed(dev), _dtype_id(dtype_name))
        return _created_tensor(ctx, uop, dtype_name, dev, requires_grad, 'poly_randn_by_id')

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
    def randint(low, high=None, shape=(1,), **kwargs):
        if high is None:
            high = low
            low = 0
        if isinstance(shape, int):
            shape = (shape,)
        if not isinstance(low, int) or not isinstance(high, int):
            raise TypeError(f'low={low!r} and high={high!r} must be integers')
        rand_kwargs = dict(kwargs)
        dtype_name = _dtype_name(rand_kwargs.pop('dtype', dtypes.int32), default='int32')
        dt = to_dtype(dtype_name)
        if not dtypes.is_int(dt):
            raise TypeError(f'dtype={dt!r} must be int')
        if high <= low:
            raise ValueError(f'high must be greater than low, got {low=} {high=}')
        return (Tensor.rand(*shape, dtype='float32', **rand_kwargs) * (high - low) + low).cast(dtype_name)

    @staticmethod
    def linspace(start, stop, steps, **kwargs):
        ctx, dev, requires_grad = _creation_meta(kwargs)
        dtype_name = _dtype_name(kwargs.get('dtype', dtypes.default_float), default='float32')
        uop = _ffi._lib.poly_linspace_by_id(
            ctx,
            float(_py_scalar(start)),
            float(_py_scalar(stop)),
            _require_i64(_py_scalar(steps), 'steps'),
            _dtype_id(dtype_name),
        )
        return _created_tensor(ctx, uop, dtype_name, dev, requires_grad, 'poly_linspace_by_id')

    @staticmethod
    def eye(n, m=None, **kwargs):
        ctx, dev, requires_grad = _creation_meta(kwargs)
        dtype_name = _dtype_name(kwargs.get('dtype', dtypes.default_float), default='float32')
        rows = _require_i64(_py_scalar(n), 'n')
        cols = rows if m is None else _require_i64(_py_scalar(m), 'm')
        uop = _ffi._lib.poly_eye_by_id(ctx, rows, cols, _dtype_id(dtype_name))
        return _created_tensor(ctx, uop, dtype_name, dev, requires_grad, 'poly_eye_by_id')

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
            uop = _ffi._lib.poly_buffer_var_by_id(ctx, _dtype_id(dtype_name), batch_raw, dims, n_inner)
            return _created_tensor(ctx, uop, dtype_name, dev, requires_grad, 'poly_buffer_var_by_id')

        shape = tuple(_require_i64(x, 'shape') for x in shape)
        if any(dim < 0 for dim in shape):
            raise ValueError(f'negative dimensions are not allowed: {shape}')
        numel = 1
        for dim in shape:
            numel *= dim
        uop = _ffi._lib.poly_buffer_by_id(ctx, _dtype_id(dtype_name), numel)
        if not uop:
            raise RuntimeError('poly_buffer_by_id failed')
        if len(shape) != 1 or (shape and shape[0] != numel):
            dims, ndim = _int64_array(shape)
            uop = _ffi._lib.poly_reshape(ctx, uop, dims, ndim)
            if not uop:
                raise RuntimeError('poly_reshape failed')
        return _created_tensor(ctx, uop, dtype_name, dev, requires_grad, 'poly_buffer_by_id')

    @staticmethod
    def manual_seed(seed):
        Tensor._seed = int(seed) & _U64_MASK
        Tensor._device_rng_counters = {}

    @staticmethod
    def cat(*tensors, dim=0):
        if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
            tensors = tensors[0]
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

    @staticmethod
    def stack(*tensors, dim=0):
        if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
            tensors = tensors[0]
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
        targets = self._live_grad_targets()
        if not targets:
            raise RuntimeError('No leaf tensors require grad')

        wrts = (_ffi._ptr * len(targets))(*[_uop_raw(t._graph_uop) for t in targets])
        out_grads = (_ffi._ptr * len(targets))()
        # tinygrad calls gradient(*targets), so every live target is handled by
        # one reverse pass. Calling poly_grad repeatedly can observe frontend
        # retargeting side effects between targets.
        rc = _ffi._lib.poly_grad_many(
            self._ctx, _uop_raw(self._graph_uop), None, wrts, len(targets), out_grads
        )
        if rc != 0:
            raise RuntimeError('poly_grad_many failed')

        for target, grad_uop in zip(targets, out_grads):
            if not grad_uop:
                raise RuntimeError('poly_grad_many returned NULL for a live target')
            grad_tensor = Tensor(
                _ctx=self._ctx, _uop=grad_uop, _shape=target.shape,
                _dtype=target._dtype_str, _device=target._device,
            )
            if target._grad is not None:
                target._grad = target._grad + grad_tensor
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
