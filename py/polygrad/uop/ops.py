"""Thin Python wrapper around polygrad's C UOp nodes.

Mirrors tinygrad's uop/ops.py UOp class at a minimal surface: hash-consed
identity in C means pointer equality is semantic equality. `_as_parameter_`
makes UOp instances transparently usable wherever a raw `PolyUOp*` is
expected by ctypes (argtype `c_void_p`), so existing FFI call sites need no
change when `Tensor.uop` is migrated to hold a `UOp` instance.
"""

from .. import _ffi

POLY_AXIS_LOOP = 3


class UOp:
    __slots__ = ('ctx', 'raw')

    def __init__(self, ctx, raw):
        # raw: int returned by ctypes for a c_void_p restype (or None).
        # ctypes returns 0/None interchangeably; normalize to None for falsy.
        self.ctx = ctx
        self.raw = raw if raw else None

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
        return _ffi._lib.poly_uop_op(self.raw) if self.raw is not None else 0

    @property
    def op_name(self):
        name = _ffi._lib.poly_op_name(self.op)
        return name.decode('utf-8') if name else 'UNKNOWN'

    # --- Factories ---

    @staticmethod
    def from_host(ctx, ptr, nbytes, dtype_id, dims, ndim):
        """Polygrad equivalent of tinygrad's _fromnp: create a BUFFER UOp
        wrapping frontend-owned host bytes, with RESHAPE on top if ndim>1.
        Caller must keep the host memory alive."""
        raw = _ffi._lib.poly_buffer_from_host(ctx, ptr, nbytes, dtype_id, dims, ndim)
        return UOp(ctx, raw) if raw else None

    @staticmethod
    def variable(ctx, name, min_val, max_val):
        """Mirrors tinygrad's UOp.variable: creates a DEFINE_VAR UOp."""
        raw = _ffi._lib.poly_define_var(ctx, name.encode() if isinstance(name, str) else name, min_val, max_val)
        return UOp(ctx, raw) if raw else None

    def bind(self, value):
        """Mirrors tinygrad's UOp.bind: binds a DEFINE_VAR to a concrete value."""
        raw = _ffi._lib.poly_bind_var(self.ctx, self.raw, value)
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
    def range(ctx, bound, axis_id=0, axis_type=POLY_AXIS_LOOP):
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

    def index(self, *idx, ptr=False):
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
        raw = _ffi._lib.poly_uop_index(self.ctx, self.raw, arr, len(indices), int(bool(ptr)))
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

    def end(self, *ranges):
        arr = (_ffi._ptr * len(ranges))(*[r.raw if isinstance(r, UOp) else r for r in ranges]) if ranges else None
        raw = _ffi._lib.poly_uop_end(self.ctx, self.raw, arr, len(ranges))
        return UOp(self.ctx, raw) if raw else None

    def sink(self, *srcs):
        all_srcs = (self,) + tuple(s for s in srcs if s is not None)
        arr = (_ffi._ptr * len(all_srcs))(*[s.raw if isinstance(s, UOp) else s for s in all_srcs])
        raw = _ffi._lib.poly_uop_sink(self.ctx, arr, len(all_srcs))
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
        if isinstance(value, int):
            return UOp(self.ctx, _ffi._lib.poly_const_int(self.ctx, value))
        if isinstance(value, float):
            return UOp(self.ctx, _ffi._lib.poly_const_float(self.ctx, value))
        raise TypeError(f'cannot convert {type(value).__name__} to UOp')

    def _alu2(self, op_name, other):
        other = self._coerce(other)
        raw = _ffi._lib.poly_alu2(self.ctx, _ffi.OPS[op_name], self.raw, other.raw)
        return UOp(self.ctx, raw) if raw else None

    def __add__(self, other):
        return self._alu2('ADD', other)

    def __radd__(self, other):
        return self._coerce(other)._alu2('ADD', self)

    def __sub__(self, other):
        return self._alu2('SUB', other)

    def __rsub__(self, other):
        return self._coerce(other)._alu2('SUB', self)

    def __mul__(self, other):
        return self._alu2('MUL', other)

    def __rmul__(self, other):
        return self._coerce(other)._alu2('MUL', self)

    def __truediv__(self, other):
        return self._alu2('FDIV', other)

    # --- Buffer identity ---

    @property
    def buffer(self):
        """Terminal buffer-identity UOp (BUFFER/BUFFER_VIEW/PARAM) after
        unwrapping RESHAPE/MULTI. None for expression UOps."""
        if self.raw is None:
            return None
        raw = _ffi._lib.poly_uop_get_buffer_identity(self.raw)
        return UOp(self.ctx, raw) if raw else None

    def has_buffer_identity(self):
        if self.raw is None:
            return False
        return bool(_ffi._lib.poly_uop_has_buffer_identity(self.raw))

    # --- Realization state ---

    @property
    def realized(self):
        """Mirrors tinygrad's UOp.realized: returns the runtime PolyBuffer
        pointer if this UOp has a buffer identity AND that buffer is
        allocated (ptr != NULL in ctx->buffers). Otherwise None.

        Callers typically only need the truthiness (is it realized?); the
        returned pointer can be passed back to the C side for data access.
        """
        buf = self.buffer
        if buf is None:
            return None
        if not _ffi._lib.poly_buffer_is_allocated(self.ctx, buf.raw):
            return None
        return _ffi._lib.poly_buffer_get(self.ctx, buf.raw)

    @property
    def is_realized(self):
        return self.realized is not None
