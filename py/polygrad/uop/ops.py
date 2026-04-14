"""Thin Python wrapper around polygrad's C UOp nodes.

Mirrors tinygrad's uop/ops.py UOp class at a minimal surface: hash-consed
identity in C means pointer equality is semantic equality. `_as_parameter_`
makes UOp instances transparently usable wherever a raw `PolyUOp*` is
expected by ctypes (argtype `c_void_p`), so existing FFI call sites need no
change when `Tensor.uop` is migrated to hold a `UOp` instance.
"""

from .. import _ffi


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
