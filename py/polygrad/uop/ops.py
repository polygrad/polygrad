"""Thin Python wrapper around polygrad's C UOp nodes.

Mirrors tinygrad's uop/ops.py UOp class at a minimal surface: hash-consed
identity in C means pointer equality is semantic equality. `_as_parameter_`
makes UOp instances transparently usable wherever a raw `PolyUOp*` is
expected by ctypes (argtype `c_void_p`), so existing FFI call sites need no
change when `Tensor.uop` is migrated to hold a `UOp` instance.
"""

import ctypes
import weakref

from .. import _ffi
from ..dtype import DTYPES_DICT, INVERSE_DTYPES_DICT, dtypes, to_dtype
from enum import IntEnum


class AxisType(IntEnum):
    DEVICE = 0
    GLOBAL = 1
    WARP = 2
    LOCAL = 3
    WEAK = 4
    GROUP_REDUCE = 5
    REDUCE = 6
    UPCAST = 7
    UNROLL = 8
    THREAD = 9
    PLACEHOLDER = 10
    LOOP = 11


POLY_AXIS_DEVICE = int(AxisType.DEVICE)
POLY_AXIS_GLOBAL = int(AxisType.GLOBAL)
POLY_AXIS_WARP = int(AxisType.WARP)
POLY_AXIS_LOCAL = int(AxisType.LOCAL)
POLY_AXIS_WEAK = int(AxisType.WEAK)
POLY_AXIS_LOOP = int(AxisType.LOOP)
POLY_AXIS_GROUP_REDUCE = int(AxisType.GROUP_REDUCE)
POLY_AXIS_REDUCE = int(AxisType.REDUCE)
POLY_AXIS_UPCAST = int(AxisType.UPCAST)
POLY_AXIS_UNROLL = int(AxisType.UNROLL)
POLY_AXIS_THREAD = int(AxisType.THREAD)
POLY_AXIS_PLACEHOLDER = int(AxisType.PLACEHOLDER)

_BASE_OPS = frozenset(
    _ffi.OPS[name]
    for name in ('RESHAPE', 'EXPAND', 'PERMUTE', 'PAD', 'SHRINK', 'FLIP', 'UNSHARD', 'DETACH')
)
_DIRECT_REALIZED_OPS = frozenset((_ffi.OPS['BUFFER'],))


def resolve(x, default=True):
    """Return a proven symbolic boolean, otherwise ``default``.

    Mirrors pinned tinygrad ``uop/ops.py:50-54``: symbolic simplification
    precedes the vmin/vmax proof rather than querying raw intervals.
    """
    if isinstance(x, bool):
        return x
    assert isinstance(x, UOp) and x.dtype == dtypes.bool, 'UOp in resolve must be bool'
    value = _ffi._lib.poly_uop_resolve(x.ctx, x.raw, int(bool(default)))
    if value < 0:
        raise RuntimeError('poly_uop_resolve failed')
    return bool(value)


class KernelInfo:
    """Minimal tinygrad KernelInfo carrier for custom-kernel SINK metadata."""

    __slots__ = ('name', 'opts_to_apply')

    def __init__(self, name='test', opts_to_apply=None):
        self.name = name
        self.opts_to_apply = opts_to_apply


all_uops: dict[int, weakref.ReferenceType] = {}


def _remove_uop_ref(identity, ref):
    if all_uops.get(identity) is ref:
        all_uops.pop(identity, None)


def _ptr_value(ptr):
    if isinstance(ptr, ctypes.c_void_p):
        return 0 if ptr.value is None else int(ptr.value)
    return int(ptr) if ptr else 0


def _dispose_uops_for_ctx(ctx):
    ctx_key = _ptr_value(ctx)
    for identity, ref in list(all_uops.items()):
        uop = ref()
        if uop is None:
            all_uops.pop(identity, None)
        elif _ptr_value(uop.ctx) == ctx_key:
            uop._dispose()


class UOp:
    __slots__ = ('ctx', 'raw', '_owned', '__weakref__')

    def __init__(self, ctx, raw):
        # raw: int returned by ctypes for a c_void_p restype (or None).
        # ctypes returns 0/None interchangeably; normalize to None for falsy.
        self.ctx = ctx
        self.raw = raw if raw else None
        self._owned = False
        if self.raw is not None:
            if _ffi._lib.poly_uop_retain(self.ctx, self.raw) != 0:
                self.raw = None
                raise RuntimeError('poly_uop_retain failed')
            self._owned = True
        identity = id(self)
        ref = weakref.ref(self, lambda dead, identity=identity: _remove_uop_ref(identity, dead))
        all_uops[identity] = ref

    def _dispose(self):
        raw = getattr(self, 'raw', None)
        ctx = getattr(self, 'ctx', None)
        owned = getattr(self, '_owned', False)
        self.raw = None
        self._owned = False
        self.ctx = None
        if raw is not None and owned and ctx is not None:
            _ffi._lib.poly_uop_release(ctx, raw)

    def __del__(self):
        try:
            self._dispose()
        except Exception:
            pass

    # ctypes hook: when a UOp instance is passed as an argument to a ctypes
    # function declared with argtype c_void_p, ctypes looks up
    # `_as_parameter_` and passes that instead. This lets us drop raw
    # pointers into FFI without changing the call sites.
    @property
    def _as_parameter_(self):
        return self.raw

    def __bool__(self):
        return self.raw is not None

    def __eq__(self, other):
        return isinstance(other, UOp) and self.raw == other.raw

    def __hash__(self):
        return hash(self.raw) if self.raw is not None else 0

    def __repr__(self):
        return f"UOp(0x{self.raw:x})" if self.raw else "UOp(null)"

    @property
    def op(self):
        """Integer Ops value, mirroring tinygrad's UOp.op."""
        if self.ctx is None:
            raise RuntimeError('polygrad runtime has been disposed')
        return _ffi._lib.poly_uop_op(self.raw) if self.raw is not None else 0

    @property
    def op_name(self):
        name = _ffi._lib.poly_op_name(self.op)
        return name.decode('utf-8') if name else 'UNKNOWN'

    @property
    def dtype(self):
        if self.ctx is None:
            raise RuntimeError('polygrad runtime has been disposed')
        dtype_id = _ffi._lib.poly_uop_dtype_id(self.ctx, self.raw) if self.raw is not None else -1
        if _ffi._lib.poly_dtype_id_by_name(b'void') == dtype_id:
            return dtypes.void
        if _ffi._lib.poly_dtype_id_by_name(b'weakint') == dtype_id:
            return dtypes.weakint
        if _ffi._lib.poly_dtype_id_by_name(b'weakfloat') == dtype_id:
            return dtypes.weakfloat
        for name, dtype in DTYPES_DICT.items():
            if _ffi._lib.poly_dtype_id_by_name(name.encode('utf-8')) == dtype_id:
                return dtype
        raise RuntimeError(f'unknown UOp dtype id {dtype_id}')

    @property
    def src(self):
        if self.ctx is None:
            raise RuntimeError('polygrad runtime has been disposed')
        if self.raw is None:
            return ()
        n = _ffi._lib.poly_uop_n_src(self.raw)
        return tuple(UOp(self.ctx, _ffi._lib.poly_uop_src(self.raw, i)) for i in range(n))

    @property
    def is_variable(self):
        return bool(self.raw and _ffi._lib.poly_uop_is_variable(self.raw))

    @property
    def is_bound_var(self):
        return bool(self.raw and _ffi._lib.poly_uop_is_bound_var(self.raw))

    @property
    def device(self):
        if self.ctx is None or not self.raw:
            raise RuntimeError('polygrad UOp has been disposed')
        names = ctypes.POINTER(ctypes.c_char_p)()
        is_tuple = ctypes.c_bool()
        count = _ffi._lib.poly_uop_device_names(self.ctx, self.raw, ctypes.byref(names), ctypes.byref(is_tuple))
        if count < 0:
            raise RuntimeError('poly_uop_device_names failed')
        # Copy borrowed metadata before any collection safe point.
        values = tuple(names[i].decode('utf-8') for i in range(count))
        return values if is_tuple.value else values[0] if count else None

    # --- Factories ---

    @staticmethod
    def from_host(ctx, ptr, nbytes, dtype_id, dims, ndim):
        """Polygrad equivalent of tinygrad's _fromnp: create a BUFFER UOp
        wrapping frontend-owned host bytes, with RESHAPE on top if ndim>1.
        Caller must keep the host memory alive."""
        raw = _ffi._lib.poly_buffer_from_host(ctx, ptr, nbytes, dtype_id, dims, ndim)
        return UOp(ctx, raw) if raw else None

    @staticmethod
    def variable(name, min_val, max_val, dtype=dtypes.weakint, multiple_of=1, param=False, *, ctx=None):
        """Create Tinygrad's ALU-address-space BUFFER/PARAM variable."""
        if ctx is None:
            from .. import _default_ctx
            ctx = _default_ctx
        dtype = to_dtype(dtype)
        dtype_name = INVERSE_DTYPES_DICT.get(dtype.name, dtype.name)
        dtype_id = _ffi._lib.poly_dtype_id_by_name(dtype_name.encode('utf-8'))
        if dtype_id < 0:
            raise ValueError(f'unknown dtype {dtype}')
        # Endpoint types are independent of the variable dtype (ParamArg's PyConst).
        bounds = [UOp.const(v, dtype=dtypes.float64 if isinstance(v, float) else None, ctx=ctx)
                  for v in (min_val, max_val)]
        raw = _ffi._lib.poly_uop_variable_by_id(
            ctx, name.encode() if isinstance(name, str) else name,
            bounds[0].raw, bounds[1].raw, dtype_id, multiple_of, param,
        )
        return UOp(ctx, raw) if raw else None

    def bind(self, value):
        """Bind an integer through AFTER(variable, STORE(variable, CONST))."""
        if not isinstance(value, int):
            raise TypeError('binding value must be an integer')
        if not -(1 << 63) <= value < (1 << 63):
            raise OverflowError('binding value exceeds the signed64 C runtime domain')
        raw = _ffi._lib.poly_uop_bind(self.ctx, self.raw, value)
        return UOp(self.ctx, raw) if raw else None

    # --- UOp-level op constructors (mirror tinygrad's UOp.contiguous / etc.) ---

    def contiguous(self):
        raw = _ffi._lib.poly_contiguous(self.ctx, self.raw)
        return UOp(self.ctx, raw) if raw else None

    @staticmethod
    def placeholder_like(uop, slot=0):
        """Mirrors tinygrad's UOp.placeholder_like for custom-kernel bodies."""
        if not isinstance(uop, UOp):
            raise TypeError('placeholder_like expects a UOp')
        raw = _ffi._lib.poly_uop_placeholder_like(uop.ctx, uop.raw, int(slot))
        return UOp(uop.ctx, raw) if raw else None

    @staticmethod
    def range(ctx, bound, axis_id=0, axis_type=AxisType.WEAK):
        raw = _ffi._lib.poly_uop_range(ctx, int(bound), int(axis_id), int(axis_type))
        return UOp(ctx, raw) if raw else None

    def numel(self):
        n = _ffi._lib.poly_uop_numel(self.ctx, self.raw)
        if n < 0:
            raise RuntimeError('poly_uop_numel failed')
        return int(n)

    def flatten(self):
        raw = _ffi._lib.poly_uop_flatten(self.ctx, self.raw)
        return UOp(self.ctx, raw) if raw else None

    def index(self, *idx):
        if len(idx) == 1 and isinstance(idx[0], (tuple, list)):
            idx = tuple(idx[0])
        indices = []
        for x in idx:
            if isinstance(x, UOp):
                indices.append(x.raw)
            elif isinstance(x, int):
                indices.append(_ffi._lib.poly_const_int(self.ctx, int(x)))
            else:
                raise TypeError(f'unsupported index type {type(x).__name__}')
        arr = (_ffi._ptr * len(indices))(*indices) if indices else None
        raw = _ffi._lib.poly_uop_index(self.ctx, self.raw, arr, len(indices))
        return UOp(self.ctx, raw) if raw else None

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            idx = (idx,)
        return self.index(*idx)

    def load(self):
        raw = _ffi._lib.poly_uop_load(self.ctx, self.raw)
        return UOp(self.ctx, raw) if raw else None

    def store(self, value):
        value = self._coerce(value)
        raw = _ffi._lib.poly_uop_store(self.ctx, self.raw, value.raw)
        return UOp(self.ctx, raw) if raw else None

    def set(self, value, end=()):
        """Mirrors tinygrad's UOp.set: store value and return AFTER(base, effect)."""
        value = self._coerce(value)
        if isinstance(end, UOp):
            ranges = (end,)
        elif end is None:
            ranges = ()
        else:
            ranges = tuple(end)
        arr = (_ffi._ptr * len(ranges))(*[r.raw if isinstance(r, UOp) else r for r in ranges]) if ranges else None
        raw = _ffi._lib.poly_uop_set(self.ctx, self.raw, value.raw, arr, len(ranges))
        return UOp(self.ctx, raw) if raw else None

    def group(self, *srcs):
        all_srcs = (self,) + tuple(s for s in srcs if s is not None)
        arr = (_ffi._ptr * len(all_srcs))(*[s.raw if isinstance(s, UOp) else s for s in all_srcs])
        raw = _ffi._lib.poly_uop_group(self.ctx, arr, len(all_srcs))
        return UOp(self.ctx, raw) if raw else None

    def end(self, *ranges):
        arr = (_ffi._ptr * len(ranges))(*[r.raw if isinstance(r, UOp) else r for r in ranges]) if ranges else None
        raw = _ffi._lib.poly_uop_end(self.ctx, self.raw, arr, len(ranges))
        return UOp(self.ctx, raw) if raw else None

    def sink(self, *srcs, arg=None):
        all_srcs = (self,) + tuple(s for s in srcs if s is not None)
        arr = (_ffi._ptr * len(all_srcs))(*[s.raw if isinstance(s, UOp) else s for s in all_srcs])
        if isinstance(arg, KernelInfo):
            name = arg.name.encode('utf-8') if arg.name else None
            optimize = 0 if arg.opts_to_apply is not None and len(arg.opts_to_apply) == 0 else 1
            raw = _ffi._lib.poly_uop_sink_ex(self.ctx, arr, len(all_srcs), name, optimize)
        elif arg is None:
            raw = _ffi._lib.poly_uop_sink(self.ctx, arr, len(all_srcs))
        else:
            raise TypeError('sink arg must be KernelInfo or None')
        return UOp(self.ctx, raw) if raw else None

    def call(self, *srcs):
        arr = (_ffi._ptr * len(srcs))(*[s.raw if isinstance(s, UOp) else s for s in srcs]) if srcs else None
        raw = _ffi._lib.poly_uop_call(self.ctx, self.raw, arr, len(srcs))
        return UOp(self.ctx, raw) if raw else None

    def after(self, *effects):
        out = self
        for effect in effects:
            effect_raw = effect.raw if isinstance(effect, UOp) else effect
            raw = _ffi._lib.poly_uop_after(self.ctx, out.raw, effect_raw)
            out = UOp(self.ctx, raw) if raw else None
            if out is None:
                return None
        return out

    def _coerce(self, value):
        if isinstance(value, UOp):
            return value
        if isinstance(value, bool):
            dtype_id = _ffi._lib.poly_dtype_id_by_name(b'bool')
            return UOp(self.ctx, _ffi._lib.poly_const_int_by_id(self.ctx, int(value), dtype_id))
        if isinstance(value, int):
            return UOp(self.ctx, _ffi._lib.poly_const_int(self.ctx, value))
        if isinstance(value, float):
            return UOp(self.ctx, _ffi._lib.poly_const_float(self.ctx, value))
        raise TypeError(f'cannot convert {type(value).__name__} to UOp')

    def _coerce_like(self, value, ref):
        if isinstance(value, UOp):
            return value
        if not isinstance(value, (int, float)):
            raise TypeError(f'cannot convert {type(value).__name__} to UOp')
        dtype_id = _ffi._lib.poly_uop_dtype_id(ref.ctx, ref.raw) if ref and ref.raw else -1
        if dtype_id >= 0:
            raw = _ffi._lib.poly_const_float_by_id(self.ctx, float(value), dtype_id)
            if raw:
                return UOp(self.ctx, raw)
            if isinstance(value, int):
                raw = _ffi._lib.poly_const_int_by_id(self.ctx, value, dtype_id)
                if raw:
                    return UOp(self.ctx, raw)
        return self._coerce(value)

    @staticmethod
    def const(value, dtype=None, *, ctx=None):
        """Create a scalar in Tensor's default context, or the explicit owner."""
        if isinstance(value, UOp):
            if ctx is not None and _ptr_value(ctx) != _ptr_value(value.ctx):
                raise ValueError('UOp.const context mismatch')
            return value if dtype is None else value.cast(dtype)
        if ctx is None:
            from .. import _default_ctx
            ctx = _default_ctx
        if dtype is not None:
            dtype = to_dtype(dtype)
            dtype_name = INVERSE_DTYPES_DICT.get(dtype.name, dtype.name)
            dtype_id = _ffi._lib.poly_dtype_id_by_name(dtype_name.encode('utf-8'))
            raw = (_ffi._lib.poly_const_int_by_id(ctx, value, dtype_id)
                   if isinstance(value, int)
                   else _ffi._lib.poly_const_float_by_id(ctx, float(value), dtype_id))
            return UOp(ctx, raw) if raw else None
        if isinstance(value, bool):
            dtype_id = _ffi._lib.poly_dtype_id_by_name(b'bool')
            return UOp(ctx, _ffi._lib.poly_const_int_by_id(ctx, int(value), dtype_id))
        if isinstance(value, int):
            raw = (_ffi._lib.poly_const_int(ctx, value) if -(1 << 63) <= value < (1 << 63)
                   else _ffi._lib.poly_const_int_decimal(ctx, str(value).encode()))
            return UOp(ctx, raw) if raw else None
        if isinstance(value, float):
            return UOp(ctx, _ffi._lib.poly_const_float(ctx, value))
        raise TypeError(f'cannot convert {type(value).__name__} to UOp')

    def _alu1(self, op_name):
        raw = _ffi._lib.poly_alu1(self.ctx, _ffi.OPS[op_name], self.raw)
        return UOp(self.ctx, raw) if raw else None

    def _alu2(self, op_name, other):
        other = self._coerce(other)
        raw = _ffi._lib.poly_binop(self.ctx, _ffi.OPS[op_name], self.raw, other.raw)
        return UOp(self.ctx, raw) if raw else None

    def _alu3(self, op_name, b, c):
        b = self._coerce_like(b, self)
        c = self._coerce_like(c, self)
        raw = _ffi._lib.poly_alu3(self.ctx, _ffi.OPS[op_name], self.raw, b.raw, c.raw)
        return UOp(self.ctx, raw) if raw else None

    def cast(self, dtype):
        dtype = to_dtype(dtype)
        name = INVERSE_DTYPES_DICT.get(dtype.name, dtype.name)
        dtype_id = _ffi._lib.poly_dtype_id_by_name(name.encode('utf-8'))
        if dtype_id < 0:
            raise ValueError(f'unknown dtype {dtype}')
        raw = _ffi._lib.poly_cast_by_id(self.ctx, self.raw, dtype_id)
        return UOp(self.ctx, raw) if raw else None

    def __add__(self, other):
        return self._alu2('ADD', other)

    def __radd__(self, other):
        other = self._coerce(other)
        raw = _ffi._lib.poly_binop(self.ctx, _ffi.OPS['ADD'], other.raw, self.raw)
        return UOp(self.ctx, raw) if raw else None

    def __sub__(self, other):
        return self._alu2('SUB', other)

    def __rsub__(self, other):
        other = self._coerce(other)
        raw = _ffi._lib.poly_binop(self.ctx, _ffi.OPS['SUB'], other.raw, self.raw)
        return UOp(self.ctx, raw) if raw else None

    def __mul__(self, other):
        return self._alu2('MUL', other)

    def __rmul__(self, other):
        other = self._coerce(other)
        raw = _ffi._lib.poly_binop(self.ctx, _ffi.OPS['MUL'], other.raw, self.raw)
        return UOp(self.ctx, raw) if raw else None

    def __truediv__(self, other):
        return self._alu2('FDIV', other)

    def __rtruediv__(self, other):
        other = self._coerce(other)
        raw = _ffi._lib.poly_binop(self.ctx, _ffi.OPS['FDIV'], other.raw, self.raw)
        return UOp(self.ctx, raw) if raw else None

    def __neg__(self):
        return self._alu1('NEG')

    def cdiv(self, other):
        return self._alu2('CDIV', other)

    def cmod(self, other):
        return self._alu2('CMOD', other)

    def floordiv(self, other):
        return self._alu2('FLOORDIV', other)

    def __floordiv__(self, other):
        return self.floordiv(other)

    def __rfloordiv__(self, other):
        other = self._coerce(other)
        raw = _ffi._lib.poly_binop(self.ctx, _ffi.OPS['FLOORDIV'], other.raw, self.raw)
        return UOp(self.ctx, raw) if raw else None

    def floormod(self, other):
        return self._alu2('FLOORMOD', other)

    def __mod__(self, other):
        return self.floormod(other)

    def __rmod__(self, other):
        other = self._coerce(other)
        raw = _ffi._lib.poly_binop(self.ctx, _ffi.OPS['FLOORMOD'], other.raw, self.raw)
        return UOp(self.ctx, raw) if raw else None

    def mod(self, other):
        return self.floormod(other)

    def maximum(self, other):
        return self._alu2('MAX', other)

    def max(self, other):
        return self.maximum(other)

    def bitwise_and(self, other):
        return self._alu2('AND', other)

    def bitwise_or(self, other):
        return self._alu2('OR', other)

    def bitwise_xor(self, other):
        return self._alu2('XOR', other)

    def shl(self, other):
        return self._alu2('SHL', other)

    def shr(self, other):
        return self._alu2('SHR', other)

    def pow(self, other):
        return self._alu2('POW', other)

    def lt(self, other):
        return self._alu2('CMPLT', other)

    def cmplt(self, other):
        return self.lt(other)

    def eq(self, other):
        # Pinned tinygrad mixin/elementwise.py:321-325 spells public equality
        # as logical-not of CMPNE. CMPEQ remains a lower-level UOp available
        # through alu; using it here changes symbolic simplification topology.
        return self.ne(other).logical_not()

    def cmpeq(self, other):
        return self.eq(other)

    def ne(self, other):
        return self._alu2('CMPNE', other)

    def cmpne(self, other):
        return self.ne(other)

    def logical_not(self):
        # Pinned tinygrad mixin/elementwise.py:39-49 casts to bool first, then
        # compares against True. poly_cast returns self for an exact dtype.
        return self.cast(dtypes.bool).ne(True)

    def sqrt(self):
        return self._alu1('SQRT')

    def exp2(self):
        return self._alu1('EXP2')

    def log2(self):
        return self._alu1('LOG2')

    def sin(self):
        return self._alu1('SIN')

    def reciprocal(self):
        return self._alu1('RECIPROCAL')

    def trunc(self):
        return self._alu1('TRUNC')

    def where(self, yes, no):
        yes, no = self._coerce(yes), self._coerce(no)
        raw = _ffi._lib.poly_where_op(self.ctx, self.raw, yes.raw, no.raw)
        return UOp(self.ctx, raw) if raw else None

    def mulacc(self, mul, acc):
        return self._alu3('MULACC', mul, acc)

    def reduce(self, *ranges, op='ADD'):
        if not isinstance(op, str):
            raise TypeError('UOp.reduce expects op as a string name')
        arr = (_ffi._ptr * len(ranges))(*[r.raw if isinstance(r, UOp) else r for r in ranges]) if ranges else None
        raw = _ffi._lib.poly_uop_reduce(self.ctx, _ffi.OPS[op], self.raw, arr, len(ranges))
        return UOp(self.ctx, raw) if raw else None

    def sum(self, *ranges):
        return self.reduce(*ranges, op='ADD')

    def max_reduce(self, *ranges):
        return self.reduce(*ranges, op='MAX')

    # --- Buffer identity ---

    @property
    def base(self):
        """Recursive storage base, matching tinygrad UOp.base."""
        if self.op in _BASE_OPS:
            sources = self.src
            return sources[0].base if sources else self
        return self

    @property
    def buffer(self):
        """tinygrad-style runtime buffer for this UOp.

        Direct identities are returned unchanged; a statically contiguous
        movement over realized storage returns its exact movement UOp with a
        C-owned zero-copy runtime view, without rewriting graph topology.
        """
        if self.raw is None:
            return None
        raw = _ffi._lib.poly_uop_buffer(self.ctx, self.raw)
        return UOp(self.ctx, raw) if raw else None

    def has_buffer_identity(self):
        if self.raw is None:
            return False
        return bool(_ffi._lib.poly_uop_has_buffer_identity(self.raw))

    # --- Realization state ---

    @property
    def realized(self):
        """Runtime buffer for a directly realized storage UOp, otherwise None.

        Current Tinygrad restricts direct scalar storage to BUFFER; movement
        wrappers are handled only by ``is_realized`` through ``base``.
        """
        if self.op not in _DIRECT_REALIZED_OPS:
            return None
        buf = self.buffer
        if buf is None:
            return None
        if not _ffi._lib.poly_buffer_is_allocated(self.ctx, buf.raw):
            return None
        return _ffi._lib.poly_buffer_get(self.ctx, buf.raw)

    @property
    def is_realized(self):
        # tinygrad/uop/ops.py:881-891: movement views are realized when their
        # recursive base buffer is allocated.
        return self.base.realized is not None
