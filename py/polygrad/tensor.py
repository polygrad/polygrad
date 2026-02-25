"""
Tensor class for polygrad — lazy evaluation backed by C compiler core.
CPU-only, float32-only for v0.
"""

import ctypes
import math

import numpy as np

from . import _ffi


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


def _shape_array(shape):
    """Return (ctypes array, int ndim) from a shape tuple."""
    return (ctypes.c_int64 * len(shape))(*shape), len(shape)


def _out_shape():
    """Allocate out_shape and out_ndim for C calls."""
    return (ctypes.c_int64 * 8)(), ctypes.c_int(0)


def _read_shape(out_shape, out_ndim):
    """Read shape tuple from C out params."""
    return tuple(out_shape[i] for i in range(out_ndim.value))


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
        self._uop = _ffi._lib.poly_define_var(self._ctx, name.encode(), min_val, max_val)

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
        self._uop = _ffi._lib.poly_bind_var(self._ctx, variable._uop, value)

    def __int__(self):
        return self.value

    def __repr__(self):
        return f"BoundVariable({self.variable.name!r}, {self.value})"


class Tensor:
    """Lazy tensor backed by polygrad's C11 compiler core.

    Operations build a UOp graph. Computation happens only when
    .numpy() or .item() is called.
    """

    training = False  # tinygrad compat
    _compile_mode = False  # suppress realize() for compile_step tracing
    _compile_assigns_ordered = []  # class-level: ASSIGN UOps in program order

    def __init__(self, data=None, requires_grad=False, *, _ctx=None, _uop=None,
                 _buffer=None, _data=None, _shape=None, _inputs=None):
        """Create a tensor from a list, numpy array, or scalar."""
        from . import _default_ctx
        self._ctx = _ctx or _default_ctx

        if _uop is not None:
            # Internal construction (from ops)
            self._uop = _uop
            self._buffer = _buffer
            self._data = _data
            self._shape = tuple(_shape) if _shape is not None else ()
            self._inputs = _inputs or []
        else:
            # User construction from data
            if isinstance(data, (int, float)):
                data = [data]
            arr = np.asarray(data, dtype=np.float32)
            self._shape = arr.shape
            self._data = arr.flatten().copy()
            self._buffer = _ffi._lib.poly_buffer_f32(self._ctx, len(self._data))
            # For multi-dimensional tensors, insert a RESHAPE UOp so the
            # scheduler knows the shape (BUFFER alone is flat/1D).
            if len(self._shape) > 1:
                dims, ndim = _int64_array(self._shape)
                self._uop = _ffi._lib.poly_reshape(self._ctx, self._buffer, dims, ndim)
            else:
                self._uop = self._buffer
            self._inputs = []

        self._requires_grad = requires_grad
        self._grad = None
        self._is_param = False     # True for model parameters (set by nn modules)
        # Saved for gradient graph stitching after realize
        self._saved_uop = None     # pre-realize UOp (computation graph)
        self._saved_inputs = None  # pre-realize inputs list

    # --- Properties ---

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def dtype(self):
        return 'float32'

    @property
    def device(self):
        from .device import Device
        return Device.DEFAULT

    @property
    def requires_grad(self):
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, val):
        self._requires_grad = val

    @property
    def grad(self):
        return self._grad

    @property
    def T(self):
        return self.transpose()

    def numel(self):
        n = 1
        for s in self._shape:
            n *= s
        return n

    def size(self, dim=None):
        if dim is None:
            return self._shape
        if dim < 0:
            dim += len(self._shape)
        return self._shape[dim]

    # --- Realization ---

    def _is_leaf(self):
        return self._data is not None and self._buffer is not None

    def _collect_leaves(self, seen=None):
        """Walk the Python tensor graph to find all leaf tensors with data."""
        if seen is None:
            seen = set()
        tid = id(self)
        if tid in seen:
            return []
        seen.add(tid)

        if self._is_leaf():
            return [self]

        leaves = []
        for inp in self._inputs:
            leaves.extend(inp._collect_leaves(seen))
        return leaves

    def _collect_deep_leaves(self, seen=None):
        """Walk through realized intermediates to find original parameter leaves.

        Like _collect_leaves, but follows _saved_inputs through realized
        non-parameter intermediates. Stops at requires_grad tensors (_requires_grad) and
        tensors that are original leaves (input data).
        """
        if seen is None:
            seen = set()
        tid = id(self)
        if tid in seen:
            return []
        seen.add(tid)

        # If this is a parameter → stop here
        if self._is_leaf() and self._requires_grad:
            return [self]

        # If this is a realized intermediate → follow saved_inputs
        if self._is_leaf() and self._saved_inputs is not None:
            leaves = []
            for inp in self._saved_inputs:
                leaves.extend(inp._collect_deep_leaves(seen))
            return leaves

        # If this is a plain leaf (input data, no saved_inputs) → stop here
        if self._is_leaf():
            return [self]

        # Not a leaf: walk regular inputs
        leaves = []
        for inp in self._inputs:
            leaves.extend(inp._collect_deep_leaves(seen))
        return leaves

    def _realize(self):
        """Execute the lazy graph to produce concrete data."""
        if self._is_leaf():
            return
        # Already realized (value cached by backward's _cache_value)
        if self._data is not None:
            return

        from .device import Device
        if Device.DEFAULT == 'CUDA':
            return self._realize_cuda()

        # Collect all leaf tensor bindings
        leaves = self._collect_leaves()

        # Build output buffer
        numel = int(np.prod(self._shape)) if self._shape else 1
        out_buf = _ffi._lib.poly_buffer_f32(self._ctx, numel)
        out_data = np.zeros(numel, dtype=np.float32)

        # Build STORE + SINK
        store = _ffi._lib.poly_store_val(self._ctx, out_buf, self._uop)
        sink = _ffi._lib.poly_sink1(self._ctx, store)

        # Build bindings array
        all_bindings = []
        for leaf in leaves:
            all_bindings.append((leaf._buffer, leaf._data.ctypes.data))
        all_bindings.append((out_buf, out_data.ctypes.data))

        n = len(all_bindings)
        c_bindings = (_ffi.PolyBufferBinding * n)()
        for i, (buf, data) in enumerate(all_bindings):
            c_bindings[i].buffer = buf
            c_bindings[i].data = data

        ret = _ffi._lib.poly_realize(self._ctx, sink, c_bindings, n)
        if ret != 0:
            raise RuntimeError(f'poly_realize failed {ret}')

        self._data = out_data
        self._buffer = out_buf
        # Save original UOp and inputs for backward graph stitching
        self._saved_uop = self._uop
        self._saved_inputs = self._inputs[:]
        # Preserve shape in UOp graph so subsequent ops see correct dimensions
        if len(self._shape) > 1:
            dims, ndim = _int64_array(self._shape)
            self._uop = _ffi._lib.poly_reshape(self._ctx, out_buf, dims, ndim)
        else:
            self._uop = out_buf
        self._inputs = []

    def _realize_cuda(self):
        """Execute the lazy graph on GPU via CUDA backend."""
        # Collect all leaf tensor bindings
        leaves = self._collect_leaves()

        # Build output buffer
        numel = int(np.prod(self._shape)) if self._shape else 1
        out_buf = _ffi._lib.poly_buffer_f32(self._ctx, numel)
        out_data = np.zeros(numel, dtype=np.float32)

        # Build STORE + SINK
        store = _ffi._lib.poly_store_val(self._ctx, out_buf, self._uop)
        sink = _ffi._lib.poly_sink1(self._ctx, store)

        # Build bindings array
        all_bindings = []
        for leaf in leaves:
            all_bindings.append((leaf._buffer, leaf._data.ctypes.data))
        all_bindings.append((out_buf, out_data.ctypes.data))

        n = len(all_bindings)
        c_bindings = (_ffi.PolyBufferBinding * n)()
        for i, (buf, data) in enumerate(all_bindings):
            c_bindings[i].buffer = buf
            c_bindings[i].data = data

        # Run on GPU
        ret = _ffi._lib.poly_realize_cuda(self._ctx, sink, c_bindings, n)
        if ret != 0:
            raise RuntimeError('poly_realize_cuda failed')

        # Copy results back from GPU to host
        _ffi._lib.poly_cuda_copyback(c_bindings, n)

        self._data = out_data
        self._buffer = out_buf
        # Save original UOp and inputs for backward graph stitching
        self._saved_uop = self._uop
        self._saved_inputs = self._inputs[:]
        # Preserve shape in UOp graph so subsequent ops see correct dimensions
        if len(self._shape) > 1:
            dims, ndim = _int64_array(self._shape)
            self._uop = _ffi._lib.poly_reshape(self._ctx, out_buf, dims, ndim)
        else:
            self._uop = out_buf
        self._inputs = []

    def assign(self, x):
        """In-place assignment: self's buffer will be overwritten with x's values.
        Must be realized before use. Returns self for chaining."""
        if not isinstance(x, Tensor):
            x = Tensor(x)
        if self._shape != x._shape:
            x = x._broadcast_to(self._shape)
        assert self._buffer is not None, "assign target must be a realized tensor"
        # Get the base buffer UOp for the ASSIGN target.
        # self._uop may be RESHAPE(BUFFER) for multi-dim tensors.
        target_uop = self._uop
        assign_uop = _ffi._lib.poly_assign(self._ctx, target_uop, x._uop)
        # Save the data array (kernel writes to it in-place) and buffer
        self._assign_data = self._data
        self._assign_buffer = self._buffer
        # Mark as unrealized so _realize() processes the ASSIGN
        self._uop = assign_uop
        self._data = None
        self._inputs = [self._make_leaf_ref(), x]
        return self

    def _make_leaf_ref(self):
        """Create a lightweight leaf reference for ASSIGN's self-binding."""
        ref = object.__new__(Tensor)
        ref._ctx = self._ctx
        ref._buffer = self._assign_buffer
        ref._data = self._assign_data
        ref._shape = self._shape
        ref._uop = self._assign_buffer
        ref._inputs = []
        ref._requires_grad = False
        ref._grad = None
        ref._is_param = False
        ref._saved_uop = None
        ref._saved_inputs = None
        return ref

    def _realize_assign(self):
        """Realize an ASSIGN: in-place update to existing buffer."""
        assign_uop = self._uop
        target_data = self._assign_data
        target_buffer = self._assign_buffer

        # Collect leaf bindings from the value expression
        leaves = self._collect_leaves()

        # Build SINK(ASSIGN) -- ASSIGN goes directly in SINK
        sink = _ffi._lib.poly_sink1(self._ctx, assign_uop)

        # Build bindings: all leaves (which includes our self-ref with target buffer)
        all_bindings = []
        seen_bufs = set()
        for leaf in leaves:
            buf_id = leaf._buffer
            if buf_id in seen_bufs:
                continue
            seen_bufs.add(buf_id)
            all_bindings.append((leaf._buffer, leaf._data.ctypes.data))

        n = len(all_bindings)
        c_bindings = (_ffi.PolyBufferBinding * n)()
        for i, (buf, data) in enumerate(all_bindings):
            c_bindings[i].buffer = buf
            c_bindings[i].data = data

        ret = _ffi._lib.poly_realize(self._ctx, sink, c_bindings, n)
        if ret != 0:
            raise RuntimeError(f'poly_realize failed (assign) {ret}')

        # Restore: data was updated in-place, point UOp back to buffer
        self._data = target_data
        self._buffer = target_buffer
        if len(self._shape) > 1:
            dims, ndim = _int64_array(self._shape)
            self._uop = _ffi._lib.poly_reshape(self._ctx, self._buffer, dims, ndim)
        else:
            self._uop = self._buffer
        self._inputs = []
        # Clean up assign temporaries
        del self._assign_data
        del self._assign_buffer

    def realize(self):
        """Realize the computation (in-place). Returns self for chaining."""
        if Tensor._compile_mode:
            if hasattr(self, '_assign_data'):
                # Pseudo-realize: save ASSIGN UOp for compile_step SINK,
                # then restore tensor to buffer state (as if assign ran).
                # This is needed because Adam's step() does:
                #   m.assign(new_m).realize()
                #   m_hat = m * bc1   <-- must use m_buffer, not ASSIGN UOp
                Tensor._compile_assigns_ordered.append(self._uop)
                self._data = self._assign_data
                self._buffer = self._assign_buffer
                if len(self._shape) > 1:
                    dims, ndim = _int64_array(self._shape)
                    self._uop = _ffi._lib.poly_reshape(self._ctx, self._buffer, dims, ndim)
                else:
                    self._uop = self._buffer
                self._inputs = []
                del self._assign_data
                del self._assign_buffer
            return self  # noop for non-assign: keep lazy graph intact
        if hasattr(self, '_assign_data'):
            self._realize_assign()
        else:
            self._realize()
        return self

    def numpy(self):
        """Realize the computation and return as numpy array."""
        if self._data is None:
            self._realize()
        return self._data.reshape(self._shape).copy()

    def item(self):
        """Return scalar value."""
        arr = self.numpy()
        if arr.size != 1:
            raise ValueError(f'item() requires scalar tensor, got shape {self._shape}')
        return float(arr.flat[0])

    def tolist(self):
        return self.numpy().tolist()

    def detach(self):
        return Tensor(self.numpy())

    def clone(self):
        return Tensor(self.numpy(), requires_grad=self._requires_grad)

    def contiguous(self):
        return self  # no-op for now

    # --- Internal helpers ---

    def _make_result(self, uop, shape, inputs):
        return Tensor(
            _ctx=self._ctx, _uop=uop, _shape=shape, _inputs=inputs,
            requires_grad=any(t._requires_grad for t in inputs),
        )

    def _ensure_tensor(self, other):
        if isinstance(other, Tensor):
            return other
        if isinstance(other, (int, float)):
            # Scalar constant — create a CONST UOp (broadcasts automatically)
            c = _ffi._lib.poly_const_float(self._ctx, float(other))
            return Tensor(_ctx=self._ctx, _uop=c, _shape=(), _inputs=[])
        raise TypeError(f'Cannot convert {type(other)} to Tensor')

    def _broadcast_shape(self, other_shape):
        """Compute broadcast shape between self._shape and other_shape."""
        a, b = self._shape, other_shape
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
                raise ValueError(f'Cannot broadcast shapes {self._shape} and {other_shape}')
        return tuple(result)

    def _broadcast_uop(self, target_shape):
        """Return a UOp that broadcasts self to target_shape via RESHAPE+EXPAND.

        Matches tinygrad's _broadcast_to: explicit shape ops so the scheduler
        sees EXPAND UOps instead of implicit ALU broadcasting.
        """
        if self._shape == target_shape:
            return self._uop
        uop = self._uop
        cur_shape = self._shape
        # CONST scalars (from _ensure_tensor) auto-broadcast — no EXPAND needed
        if not cur_shape and self._buffer is None:
            return uop
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
        out_shape = self._broadcast_shape(other._shape)
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
        uop = _ffi._lib.poly_alu1(self._ctx, _ffi.OPS['NEG'], self._uop)
        return self._make_result(uop, self._shape, [self])

    def __pow__(self, other):
        return self._binop(other, 'POW')

    # --- Comparisons (C core) ---

    def __lt__(self, other):
        return self._binop(other, 'CMPLT')

    def __eq__(self, other):
        other = self._ensure_tensor(other)
        out_shape = self._broadcast_shape(other._shape)
        uop = _ffi._lib.poly_eq(self._ctx, self._broadcast_uop(out_shape),
                                 other._broadcast_uop(out_shape))
        return self._make_result(uop, out_shape, [self, other])

    def __ne__(self, other):
        other = self._ensure_tensor(other)
        out_shape = self._broadcast_shape(other._shape)
        uop = _ffi._lib.poly_ne(self._ctx, self._broadcast_uop(out_shape),
                                 other._broadcast_uop(out_shape))
        return self._make_result(uop, out_shape, [self, other])

    def __gt__(self, other):
        other = self._ensure_tensor(other)
        out_shape = self._broadcast_shape(other._shape)
        uop = _ffi._lib.poly_gt(self._ctx, self._broadcast_uop(out_shape),
                                 other._broadcast_uop(out_shape))
        return self._make_result(uop, out_shape, [self, other])

    def __ge__(self, other):
        other = self._ensure_tensor(other)
        out_shape = self._broadcast_shape(other._shape)
        uop = _ffi._lib.poly_ge(self._ctx, self._broadcast_uop(out_shape),
                                 other._broadcast_uop(out_shape))
        return self._make_result(uop, out_shape, [self, other])

    def __le__(self, other):
        other = self._ensure_tensor(other)
        out_shape = self._broadcast_shape(other._shape)
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
        out_shape = self._broadcast_shape(x._shape)
        out_shape = Tensor(_ctx=self._ctx, _uop=self._uop, _shape=out_shape, _inputs=[])._broadcast_shape(y._shape)
        c_uop = self._broadcast_uop(out_shape)
        x_uop = x._broadcast_uop(out_shape)
        y_uop = y._broadcast_uop(out_shape)
        uop = _ffi._lib.poly_where_op(self._ctx, c_uop, x_uop, y_uop)
        return self._make_result(uop, out_shape, [self, x, y])

    def maximum(self, other):
        other = self._ensure_tensor(other)
        out_shape = self._broadcast_shape(other._shape)
        uop = _ffi._lib.poly_maximum(self._ctx, self._broadcast_uop(out_shape),
                                      other._broadcast_uop(out_shape))
        return self._make_result(uop, out_shape, [self, other])

    def minimum(self, other):
        other = self._ensure_tensor(other)
        out_shape = self._broadcast_shape(other._shape)
        uop = _ffi._lib.poly_minimum(self._ctx, self._broadcast_uop(out_shape),
                                      other._broadcast_uop(out_shape))
        return self._make_result(uop, out_shape, [self, other])

    def clamp(self, min_=None, max_=None):
        if min_ is None and max_ is None:
            raise RuntimeError("at least one of 'min_' or 'max_' must not be None")
        lo = min_ if min_ is not None else -1e38
        hi = max_ if max_ is not None else 1e38
        uop = _ffi._lib.poly_clamp(self._ctx, self._uop, float(lo), float(hi))
        return self._make_result(uop, self._shape, [self])

    # --- Unary math (C core composed ops) ---

    def exp2(self):
        uop = _ffi._lib.poly_alu1(self._ctx, _ffi.OPS['EXP2'], self._uop)
        return self._make_result(uop, self._shape, [self])

    def log2(self):
        uop = _ffi._lib.poly_alu1(self._ctx, _ffi.OPS['LOG2'], self._uop)
        return self._make_result(uop, self._shape, [self])

    def sqrt(self):
        uop = _ffi._lib.poly_alu1(self._ctx, _ffi.OPS['SQRT'], self._uop)
        return self._make_result(uop, self._shape, [self])

    def reciprocal(self):
        uop = _ffi._lib.poly_alu1(self._ctx, _ffi.OPS['RECIPROCAL'], self._uop)
        return self._make_result(uop, self._shape, [self])

    def trunc(self):
        uop = _ffi._lib.poly_alu1(self._ctx, _ffi.OPS['TRUNC'], self._uop)
        return self._make_result(uop, self._shape, [self])

    def exp(self):
        uop = _ffi._lib.poly_exp(self._ctx, self._uop)
        return self._make_result(uop, self._shape, [self])

    def log(self):
        uop = _ffi._lib.poly_log(self._ctx, self._uop)
        return self._make_result(uop, self._shape, [self])

    def sin(self):
        uop = _ffi._lib.poly_sin(self._ctx, self._uop)
        return self._make_result(uop, self._shape, [self])

    def cos(self):
        uop = _ffi._lib.poly_cos(self._ctx, self._uop)
        return self._make_result(uop, self._shape, [self])

    def tan(self):
        uop = _ffi._lib.poly_tan(self._ctx, self._uop)
        return self._make_result(uop, self._shape, [self])

    def sigmoid(self):
        uop = _ffi._lib.poly_sigmoid(self._ctx, self._uop)
        return self._make_result(uop, self._shape, [self])

    def tanh(self):
        uop = _ffi._lib.poly_tanh_act(self._ctx, self._uop)
        return self._make_result(uop, self._shape, [self])

    def abs(self):
        uop = _ffi._lib.poly_abs(self._ctx, self._uop)
        return self._make_result(uop, self._shape, [self])

    def sign(self):
        uop = _ffi._lib.poly_sign(self._ctx, self._uop)
        return self._make_result(uop, self._shape, [self])

    def square(self):
        uop = _ffi._lib.poly_square(self._ctx, self._uop)
        return self._make_result(uop, self._shape, [self])

    def rsqrt(self):
        uop = _ffi._lib.poly_rsqrt(self._ctx, self._uop)
        return self._make_result(uop, self._shape, [self])

    def ceil(self):
        uop = _ffi._lib.poly_ceil(self._ctx, self._uop)
        return self._make_result(uop, self._shape, [self])

    def floor(self):
        uop = _ffi._lib.poly_floor(self._ctx, self._uop)
        return self._make_result(uop, self._shape, [self])

    def round(self):
        uop = _ffi._lib.poly_round_f(self._ctx, self._uop)
        return self._make_result(uop, self._shape, [self])

    def isinf(self):
        uop = _ffi._lib.poly_isinf(self._ctx, self._uop)
        return self._make_result(uop, self._shape, [self])

    def isnan(self):
        uop = _ffi._lib.poly_isnan(self._ctx, self._uop)
        return self._make_result(uop, self._shape, [self])

    # --- Activations (C core composed ops) ---

    def relu(self):
        uop = _ffi._lib.poly_relu(self._ctx, self._uop)
        return self._make_result(uop, self._shape, [self])

    def relu6(self):
        uop = _ffi._lib.poly_relu6(self._ctx, self._uop)
        return self._make_result(uop, self._shape, [self])

    def leaky_relu(self, neg_slope=0.01):
        uop = _ffi._lib.poly_leaky_relu(self._ctx, self._uop, neg_slope)
        return self._make_result(uop, self._shape, [self])

    def gelu(self):
        uop = _ffi._lib.poly_gelu(self._ctx, self._uop)
        return self._make_result(uop, self._shape, [self])

    def quick_gelu(self):
        uop = _ffi._lib.poly_quick_gelu(self._ctx, self._uop)
        return self._make_result(uop, self._shape, [self])

    def silu(self):
        uop = _ffi._lib.poly_silu(self._ctx, self._uop)
        return self._make_result(uop, self._shape, [self])

    def swish(self):
        return self.silu()

    def elu(self, alpha=1.0):
        uop = _ffi._lib.poly_elu(self._ctx, self._uop, alpha)
        return self._make_result(uop, self._shape, [self])

    def softplus(self, beta=1.0):
        uop = _ffi._lib.poly_softplus(self._ctx, self._uop, beta)
        return self._make_result(uop, self._shape, [self])

    def mish(self):
        uop = _ffi._lib.poly_mish(self._ctx, self._uop)
        return self._make_result(uop, self._shape, [self])

    def hardtanh(self, min_val=-1, max_val=1):
        uop = _ffi._lib.poly_hardtanh(self._ctx, self._uop, min_val, max_val)
        return self._make_result(uop, self._shape, [self])

    def hardswish(self):
        uop = _ffi._lib.poly_hardswish(self._ctx, self._uop)
        return self._make_result(uop, self._shape, [self])

    def hardsigmoid(self):
        uop = _ffi._lib.poly_hardsigmoid(self._ctx, self._uop)
        return self._make_result(uop, self._shape, [self])

    # --- Softmax (Tensor-level composition with realize boundaries) ---
    # Composed at Tensor level (not C core) because the reduce→expand→alu
    # pattern requires separate kernels. Each .realize() forces a kernel boundary.

    def softmax(self, axis=-1):
        m = self.max(axis=axis, keepdim=True).realize()
        e = (self - m).exp().realize()
        s = e.sum(axis=axis, keepdim=True).realize()
        return e / s

    def log_softmax(self, axis=-1):
        m = self.max(axis=axis, keepdim=True).realize()
        shifted = (self - m).realize()
        e = shifted.exp().realize()
        s = e.sum(axis=axis, keepdim=True).realize()
        return shifted - s.log()

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
        uop = _ffi._lib.poly_reshape(self._ctx, self._uop, arr, n)
        return self._make_result(uop, shape, [self])

    def permute(self, *order):
        if len(order) == 1 and isinstance(order[0], (tuple, list)):
            order = tuple(order[0])
        arr, n = _int64_array(order)
        uop = _ffi._lib.poly_permute(self._ctx, self._uop, arr, n)
        new_shape = tuple(self._shape[i] for i in order)
        return self._make_result(uop, new_shape, [self])

    def expand(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        arr, n = _int64_array(shape)
        uop = _ffi._lib.poly_expand(self._ctx, self._uop, arr, n)
        return self._make_result(uop, shape, [self])

    def shrink(self, arg):
        """arg is tuple of (start, end) pairs per dimension."""
        flat, n = _pair_array(arg)
        uop = _ffi._lib.poly_shrink(self._ctx, self._uop, flat, n)
        new_shape = tuple(e - s for s, e in arg)
        return self._make_result(uop, new_shape, [self])

    def pad(self, arg):
        """arg is tuple of (before, after) pairs per dimension."""
        flat, n = _pair_array(arg)
        uop = _ffi._lib.poly_pad(self._ctx, self._uop, flat, n)
        new_shape = tuple(s + b + a for s, (b, a) in zip(self._shape, arg))
        return self._make_result(uop, new_shape, [self])

    def flip(self, axis):
        if isinstance(axis, int):
            axis = (axis,)
        arr, n = _int64_array(axis)
        uop = _ffi._lib.poly_flip(self._ctx, self._uop, arr, n)
        return self._make_result(uop, self._shape, [self])

    def transpose(self, dim0=-2, dim1=-1):
        nd = len(self._shape)
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
        r, c = self._shape[-2], self._shape[-1]
        mask = Tensor._tri(r, c, diagonal=diagonal)
        return mask.where(self, Tensor(0.0))

    def tril(self, diagonal=0):
        r, c = self._shape[-2], self._shape[-1]
        mask = Tensor._tri(r, c, diagonal=diagonal + 1)
        return mask.where(Tensor(0.0), self)

    def squeeze(self, dim=None):
        if dim is not None:
            if dim < 0:
                dim += len(self._shape)
            if self._shape[dim] != 1:
                return self
            new_shape = tuple(s for i, s in enumerate(self._shape) if i != dim)
            if not new_shape:
                new_shape = (1,)
            return self.reshape(new_shape)
        new_shape = tuple(s for s in self._shape if s != 1)
        if not new_shape:
            new_shape = (1,)
        if new_shape == self._shape:
            return self
        return self.reshape(new_shape)

    def unsqueeze(self, dim):
        if dim < 0:
            dim += len(self._shape) + 1
        new_shape = list(self._shape)
        new_shape.insert(dim, 1)
        return self.reshape(tuple(new_shape))

    def flatten(self, start_dim=0, end_dim=-1):
        if end_dim < 0:
            end_dim += len(self._shape)
        new_shape = list(self._shape[:start_dim])
        flat_dim = 1
        for i in range(start_dim, end_dim + 1):
            flat_dim *= self._shape[i]
        new_shape.append(flat_dim)
        new_shape.extend(self._shape[end_dim + 1:])
        return self.reshape(tuple(new_shape))

    def unflatten(self, dim, sizes):
        if dim < 0:
            dim += len(self._shape)
        new_shape = list(self._shape[:dim]) + list(sizes) + list(self._shape[dim + 1:])
        return self.reshape(tuple(new_shape))

    def view(self, *shape):
        return self.reshape(*shape)

    def repeat(self, *repeats):
        if len(repeats) == 1 and isinstance(repeats[0], (tuple, list)):
            repeats = tuple(repeats[0])
        # Pad shape if needed
        nd = max(len(self._shape), len(repeats))
        shape = (1,) * (nd - len(self._shape)) + self._shape
        repeats = (1,) * (nd - len(repeats)) + repeats
        # Interleave: reshape to (s0, 1, s1, 1, ...), expand to (s0, r0, s1, r1, ...), flatten pairs
        new_shape = []
        exp_shape = []
        for s, r in zip(shape, repeats):
            new_shape.extend([s, 1])
            exp_shape.extend([s, r])
        result = self.reshape(tuple(new_shape)).expand(tuple(exp_shape))
        final_shape = tuple(s * r for s, r in zip(shape, repeats))
        return result.reshape(final_shape)

    # --- Reduction ops (C core) ---

    def sum(self, axis=None, keepdim=False):
        if axis is None:
            axis = tuple(range(len(self._shape)))
        elif isinstance(axis, int):
            axis = (axis,)
        # Normalize negative axes
        nd = len(self._shape)
        axis = tuple(a + nd if a < 0 else a for a in axis)

        arr, n = _int64_array(axis)
        uop = _ffi._lib.poly_reduce_axis(self._ctx, _ffi.OPS['ADD'], self._uop, arr, n)

        # Compute output shape
        new_shape = []
        for i, s in enumerate(self._shape):
            if i in axis:
                if keepdim:
                    new_shape.append(1)
            else:
                new_shape.append(s)
        if not new_shape:
            new_shape = ()
        new_shape = tuple(new_shape)

        return self._make_result(uop, new_shape, [self])

    def max(self, axis=None, keepdim=False):
        if axis is None:
            # Reduce all
            result = self
            for i in range(len(self._shape) - 1, -1, -1):
                sh, ndim = _shape_array(result._shape)
                out_shape, out_ndim = _out_shape()
                uop = _ffi._lib.poly_max_reduce(self._ctx, result._uop,
                                                  sh, ndim, i, int(keepdim),
                                                  out_shape, ctypes.byref(out_ndim))
                result = self._make_result(uop, _read_shape(out_shape, out_ndim), [result])
            return result
        # Normalize negative axis
        nd = len(self._shape)
        if axis < 0:
            axis = axis + nd
        sh, ndim = _shape_array(self._shape)
        out_shape, out_ndim = _out_shape()
        uop = _ffi._lib.poly_max_reduce(self._ctx, self._uop, sh, ndim,
                                          axis, int(keepdim),
                                          out_shape, ctypes.byref(out_ndim))
        result = self._make_result(uop, _read_shape(out_shape, out_ndim), [self])
        return result

    def min(self, axis=None, keepdim=False):
        return (-self).max(axis=axis, keepdim=keepdim).__neg__()

    def mean(self, axis=None, keepdim=False):
        if axis is None:
            return self.sum(keepdim=keepdim) / self.numel()
        sh, ndim = _shape_array(self._shape)
        out_shape, out_ndim = _out_shape()
        uop = _ffi._lib.poly_mean_reduce(self._ctx, self._uop, sh, ndim,
                                           axis, int(keepdim),
                                           out_shape, ctypes.byref(out_ndim))
        return self._make_result(uop, _read_shape(out_shape, out_ndim), [self])

    def var(self, axis=None, keepdim=False, correction=1):
        if axis is None:
            m = self.mean()
            diff = self - m
            sq = diff * diff
            return sq.sum() / (self.numel() - correction)
        if isinstance(axis, int):
            if axis < 0:
                axis += len(self._shape)
        # Tensor-level: mean→realize→subtract→square→sum (avoid reduce→expand→alu)
        m = self.mean(axis=axis, keepdim=True).realize()
        diff = self - m
        sq = diff * diff
        dim_size = self._shape[axis]
        return sq.sum(axis=axis, keepdim=keepdim) / (dim_size - correction)

    def std(self, axis=None, keepdim=False, correction=1):
        return self.var(axis=axis, keepdim=keepdim, correction=correction).sqrt()

    # --- Matmul (C core dot) ---

    def dot(self, w):
        if not isinstance(w, Tensor):
            raise TypeError(f'Expected Tensor, got {type(w)}')
        x_sh, x_n = _shape_array(self._shape)
        w_sh, w_n = _shape_array(w._shape)
        out_shape, out_ndim = _out_shape()
        uop = _ffi._lib.poly_dot(self._ctx,
            self._uop, x_sh, x_n, w._uop, w_sh, w_n,
            out_shape, ctypes.byref(out_ndim))
        return self._make_result(uop, _read_shape(out_shape, out_ndim), [self, w])

    def matmul(self, other):
        return self.dot(other)

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

    def cross_entropy(self, target, axis=-1):
        """Cross-entropy loss: -log_softmax(self, axis)[target]."""
        log_probs = self.log_softmax(axis=axis)
        # Simple implementation: -(target * log_probs).sum(-1).mean()
        return -(target * log_probs).sum(axis=axis).mean()

    def binary_crossentropy(self, target):
        return -(target * self.log() + (1.0 - target) * (1.0 - self).log()).mean()

    def layernorm(self, axis=-1, eps=1e-5):
        m = self.mean(axis=axis, keepdim=True).realize()
        v = self.var(axis=axis, keepdim=True, correction=0).realize()
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
            n_fill = len(self._shape) - n_real
            idx = idx[:eidx] + (slice(None),) * n_fill + idx[eidx + 1:]

        result = self
        dim = 0
        for i in idx:
            if i is None:
                result = result.unsqueeze(dim)
                dim += 1
            elif isinstance(i, int):
                if i < 0:
                    i += result._shape[dim]
                result = result.shrink(
                    tuple((i, i + 1) if d == dim else (0, s)
                          for d, s in enumerate(result._shape)))
                result = result.squeeze(dim)
            elif isinstance(i, slice):
                start, stop, step = i.indices(result._shape[dim])
                if step != 1:
                    raise IndexError('Step != 1 not supported')
                result = result.shrink(
                    tuple((start, stop) if d == dim else (0, s)
                          for d, s in enumerate(result._shape)))
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

        # Build C arrays
        tensor_arr = (_ffi._ptr * n)(*[t._uop for t in operands])
        shape_arrays = []
        for t in operands:
            sa = (ctypes.c_int64 * len(t._shape))(*t._shape)
            shape_arrays.append(sa)
        shape_ptrs = (ctypes.POINTER(ctypes.c_int64) * n)(
            *[ctypes.cast(sa, ctypes.POINTER(ctypes.c_int64)) for sa in shape_arrays])
        ndim_arr = (ctypes.c_int * n)(*[len(t._shape) for t in operands])

        out_shape = (ctypes.c_int64 * 8)()
        out_ndim = ctypes.c_int(0)

        uop = _ffi._lib.poly_einsum(
            ctx, formula.encode('utf-8'),
            tensor_arr, shape_ptrs, ndim_arr, n,
            out_shape, ctypes.byref(out_ndim))
        if not uop:
            raise ValueError(f'poly_einsum failed for formula: {formula}')
        shape = tuple(out_shape[i] for i in range(out_ndim.value))
        return Tensor(_ctx=ctx, _uop=uop, _shape=shape, _inputs=list(operands))

    # --- Rearrange (C core, einops-style) ---

    def rearrange(self, formula, **kwargs):
        """Einops-style rearrange. E.g. t.rearrange('b c h w -> b (c h) w')."""
        # Build axis_names and axis_values from kwargs
        names = list(kwargs.keys())
        values = [kwargs[k] for k in names]
        n = len(names)

        axis_names = ' '.join(names).encode('utf-8') if names else None
        axis_values = (ctypes.c_int64 * n)(*values) if n > 0 else None

        sh, ndim = _int64_array(self._shape)
        out_shape = (ctypes.c_int64 * 8)()
        out_ndim = ctypes.c_int(0)

        uop = _ffi._lib.poly_rearrange(
            self._ctx, formula.encode('utf-8'),
            self._uop, sh, ndim,
            axis_names, axis_values, n,
            out_shape, ctypes.byref(out_ndim))
        if not uop:
            raise ValueError(f'poly_rearrange failed for formula: {formula}')
        shape = tuple(out_shape[i] for i in range(out_ndim.value))
        return self._make_result(uop, shape, [self])

    # --- Static constructors ---

    @staticmethod
    def zeros(*shape, **kwargs):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return Tensor(np.zeros(shape, dtype=np.float32), **kwargs)

    @staticmethod
    def ones(*shape, **kwargs):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return Tensor(np.ones(shape, dtype=np.float32), **kwargs)

    @staticmethod
    def full(shape, fill_value, **kwargs):
        if isinstance(shape, int):
            shape = (shape,)
        return Tensor(np.full(shape, fill_value, dtype=np.float32), **kwargs)

    @staticmethod
    def arange(stop, start=0, step=1, **kwargs):
        return Tensor(np.arange(start, stop, step, dtype=np.float32), **kwargs)

    @staticmethod
    def rand(*shape, **kwargs):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return Tensor(np.random.rand(*shape).astype(np.float32), **kwargs)

    @staticmethod
    def randn(*shape, **kwargs):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return Tensor(np.random.randn(*shape).astype(np.float32), **kwargs)

    @staticmethod
    def randint(low, high=None, shape=(1,), **kwargs):
        if high is None:
            high = low
            low = 0
        if isinstance(shape, int):
            shape = (shape,)
        return Tensor(np.random.randint(low, high, size=shape).astype(np.float32), **kwargs)

    @staticmethod
    def linspace(start, stop, steps, **kwargs):
        return Tensor(np.linspace(start, stop, steps, dtype=np.float32), **kwargs)

    @staticmethod
    def eye(n, **kwargs):
        return Tensor(np.eye(n, dtype=np.float32), **kwargs)

    @staticmethod
    def empty(*shape, **kwargs):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return Tensor(np.empty(shape, dtype=np.float32), **kwargs)

    @staticmethod
    def manual_seed(seed):
        np.random.seed(seed)

    @staticmethod
    def cat(*tensors, dim=0):
        if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
            tensors = tensors[0]
        if not tensors:
            raise ValueError('cat requires at least one tensor')
        # Implementation: pad each tensor, then sum
        shapes = [t._shape for t in tensors]
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
            pad_after[dim] = out_shape[dim] - offset - t._shape[dim]
            padded = t.pad(tuple((pad_before[i], pad_after[i]) for i in range(ndim)))
            if result is None:
                result = padded
            else:
                result = result + padded
            offset += t._shape[dim]
        return result

    @staticmethod
    def stack(*tensors, dim=0):
        if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
            tensors = tensors[0]
        return Tensor.cat(*[t.unsqueeze(dim) for t in tensors], dim=dim)

    def split(self, sizes, dim=0):
        if dim < 0:
            dim += len(self._shape)
        if isinstance(sizes, int):
            # Split into chunks of given size
            total = self._shape[dim]
            sizes = [sizes] * (total // sizes)
            if total % sizes[0]:
                sizes.append(total % sizes[0])
        results = []
        offset = 0
        for sz in sizes:
            arg = tuple(
                (offset, offset + sz) if d == dim else (0, s)
                for d, s in enumerate(self._shape)
            )
            results.append(self.shrink(arg))
            offset += sz
        return results

    def chunk(self, n, dim=0):
        if dim < 0:
            dim += len(self._shape)
        total = self._shape[dim]
        chunk_size = math.ceil(total / n)
        return self.split(chunk_size, dim)

    # --- Autograd ---

    def _cache_value(self):
        """Compute and cache tensor value without modifying the UOp graph.

        This is called by backward() to cache the loss value before optimizer
        step potentially mutates parameter buffers.
        """
        if Tensor._compile_mode:
            return  # noop: keep lazy graph intact for compile_step tracing
        if self._data is not None:
            return
        if self._is_leaf():
            return

        leaves = self._collect_leaves()
        numel = int(np.prod(self._shape)) if self._shape else 1
        out_buf = _ffi._lib.poly_buffer_f32(self._ctx, numel)
        out_data = np.zeros(numel, dtype=np.float32)

        store = _ffi._lib.poly_store_val(self._ctx, out_buf, self._uop)
        sink = _ffi._lib.poly_sink1(self._ctx, store)

        all_bindings = []
        for leaf in leaves:
            all_bindings.append((leaf._buffer, leaf._data.ctypes.data))
        all_bindings.append((out_buf, out_data.ctypes.data))

        n = len(all_bindings)
        c_bindings = (_ffi.PolyBufferBinding * n)()
        for i, (buf, data) in enumerate(all_bindings):
            c_bindings[i].buffer = buf
            c_bindings[i].data = data

        ret = _ffi._lib.poly_realize(self._ctx, sink, c_bindings, n)
        if ret == 0:
            self._data = out_data

    def _collect_realized_intermediates(self, seen=None):
        """Walk the tensor graph (via _saved_inputs) to find all realized
        intermediate tensors that need UOp substitution for backward.

        A realized tensor is an intermediate if it has computation history
        (_saved_inputs). Leaf tensors (params, user-created) have _saved_uop
        but no _saved_inputs — they are terminal leaves, not intermediates.
        """
        if seen is None:
            seen = set()
        tid = id(self)
        if tid in seen:
            return []
        seen.add(tid)

        result = []
        if self._saved_uop is not None and self._saved_inputs:
            # Realized tensor with computation history → intermediate
            result.append(self)
            for inp in self._saved_inputs:
                result.extend(inp._collect_realized_intermediates(seen))
        elif self._saved_uop is not None:
            # Realized leaf (param, user-created) → stop here
            pass
        else:
            # Walk regular inputs
            for inp in self._inputs:
                result.extend(inp._collect_realized_intermediates(seen))
        return result

    def backward(self):
        """Compute gradients via reverse-mode autodiff.

        Uses segment-wise backward: each realize() boundary defines a segment.
        Segments are processed in reverse topological order. For each segment,
        a local VJP (vector-Jacobian product) is computed using poly_grad_many,
        and all gradient outputs are realized in a single sink.

        For simple graphs (no intermediate realizes), falls back to the direct
        poly_grad approach.
        """
        if not self._inputs and self._is_leaf() and self._saved_uop is None:
            raise RuntimeError('backward() called on a realized tensor — '
                               'call backward() before item()/numpy()')

        # In compile mode, always use simple backward: the full graph is lazy
        # (no intermediate realizes during tracing), so segment-wise backward
        # would incorrectly reference _saved_uop from pre-compile realizes
        # (model param initialization), bringing in stale BUFFER UOps.
        if Tensor._compile_mode:
            return self._backward_simple()

        # Collect realized intermediates (segment boundaries)
        intermediates = self._collect_realized_intermediates()

        if not intermediates:
            # Simple case: no intermediate realizes, use direct poly_grad
            return self._backward_simple()

        # Segment-wise backward
        return self._backward_segmented(intermediates)

    def _backward_simple(self):
        """Direct backward for simple graphs (no intermediate realizes)."""
        all_leaves = self._collect_leaves()
        leaves = [t for t in all_leaves if t._requires_grad]
        if not leaves:
            raise RuntimeError('No leaf tensors require grad')

        # Deduplicate
        seen_bufs = set()
        unique_leaves = []
        for l in leaves:
            buf_id = id(l._buffer) if l._buffer else id(l)
            if buf_id not in seen_bufs:
                seen_bufs.add(buf_id)
                unique_leaves.append(l)
        leaves = unique_leaves

        self._cache_value()

        for leaf in leaves:
            grad_uop = _ffi._lib.poly_grad(self._ctx, self._uop, leaf._uop)
            if not grad_uop:
                raise RuntimeError('poly_grad returned NULL for a leaf tensor')

            snap_leaves = []
            for l in all_leaves:
                snap_leaves.append(Tensor(_ctx=self._ctx, _uop=l._uop, _buffer=l._buffer, _data=l._data, _shape=l._shape))
            grad_tensor = Tensor(_ctx=self._ctx, _uop=grad_uop, _shape=leaf._shape, _inputs=snap_leaves)
            if leaf._grad is not None:
                leaf._grad = leaf._grad + grad_tensor
            else:
                leaf._grad = grad_tensor

    def _backward_segmented(self, intermediates):
        """Segment-wise backward through realized intermediate boundaries.

        Each realize() defines a segment boundary. We process segments using
        Kahn's algorithm (topological sort) so that shared intermediates
        (e.g. qkv split into q, k, v) receive ALL upstream contributions
        before being processed. This keeps each poly_grad_many call operating
        on a small local graph with correct accumulated gradients.
        """
        from collections import deque

        self._cache_value()

        inter_set = set(id(t) for t in intermediates)
        inter_map = {id(t): t for t in intermediates}

        # Map: tensor id → (grad_numpy_data, grad_buffer_uop)
        upstream_grads = {}

        # ── Pre-compute segment info and contribution counts ─────
        # For each segment, find its leaves and targets (params + intermediates).
        # Count how many segments will contribute upstream to each intermediate.
        seg_info = {}  # id(seg) → (seg_leaves, seg_targets)
        contribution_count = {id(t): 0 for t in intermediates}

        for seg_tensor in intermediates:
            seg_leaves = []
            seen = set()
            for inp in (seg_tensor._saved_inputs or []):
                seg_leaves.extend(inp._collect_leaves(seen))
            seg_targets = [l for l in seg_leaves
                           if l._requires_grad or id(l) in inter_set]
            # Dedupe targets by tensor id to avoid over-counting contributions
            seen_target_ids = set()
            unique_targets = []
            for t in seg_targets:
                if id(t) not in seen_target_ids:
                    seen_target_ids.add(id(t))
                    unique_targets.append(t)
            seg_targets = unique_targets
            seg_info[id(seg_tensor)] = (seg_leaves, seg_targets)
            for t in seg_targets:
                if id(t) in inter_set:
                    contribution_count[id(t)] += 1

        # ── Phase 1: Loss segment ────────────────────────────────
        # From the loss tensor back to its immediate leaves (shallow walk).
        # These include both params and realized intermediates.
        loss_leaves = self._collect_leaves()
        loss_targets = [l for l in loss_leaves
                        if l._requires_grad or id(l) in inter_set]

        if loss_targets:
            self._realize_segment_grads(
                self._uop, None, loss_targets, loss_leaves,
                upstream_grads, inter_set=inter_set)

        # ── Phase 2: Kahn's algorithm (topological sort) ─────────
        # Process each intermediate only after ALL segments that contribute
        # to its upstream gradient have been processed.
        received = {id(t): 0 for t in intermediates}
        processed = set()
        ready = deque()

        # Seed: intermediates that received upstream from Phase 1 and have
        # no pending contributions (contribution_count == 0), OR received
        # upstream and all contributions are already in.
        for seg_tensor in intermediates:
            tid = id(seg_tensor)
            if tid in upstream_grads and contribution_count[tid] == received[tid]:
                ready.append(seg_tensor)

        while ready:
            seg_tensor = ready.popleft()
            tid = id(seg_tensor)
            if tid in processed:
                continue
            processed.add(tid)

            if tid not in upstream_grads:
                continue

            up_data, up_buf = upstream_grads[tid]
            seg_leaves, seg_targets = seg_info[tid]

            if not seg_targets:
                continue

            # ── Generic VJP approach ──────────────────────────────
            # Build VJP loss = sum(upstream_grad * segment_output)
            up_uop = up_buf
            if len(seg_tensor._shape) > 1:
                dims, ndim = _int64_array(seg_tensor._shape)
                up_uop = _ffi._lib.poly_reshape(self._ctx, up_buf, dims, ndim)

            vjp_prod = _ffi._lib.poly_alu2(
                self._ctx, _ffi.OPS['MUL'], up_uop, seg_tensor._saved_uop)

            # Reduce all axes to scalar
            n_dims = len(seg_tensor._shape)
            if n_dims > 0:
                axes_arr, n_axes = _int64_array(list(range(n_dims)))
                vjp_loss = _ffi._lib.poly_reduce_axis(
                    self._ctx, _ffi.OPS['ADD'], vjp_prod, axes_arr, n_axes)
            else:
                vjp_loss = vjp_prod

            self._realize_segment_grads(
                vjp_loss, None, seg_targets, seg_leaves,
                upstream_grads,
                extra_bufs=[(up_buf, up_data)],
                inter_set=inter_set)

            # Check if any intermediate targets are now ready
            for t in seg_targets:
                t_id = id(t)
                if t_id in inter_set and t_id not in processed:
                    received[t_id] += 1
                    if received[t_id] == contribution_count[t_id]:
                        ready.append(inter_map[t_id])

        # ── Consistency check ─────────────────────────────────────
        for t in intermediates:
            if id(t) in upstream_grads and id(t) not in processed:
                raise RuntimeError(
                    f'segment backward: intermediate received upstream '
                    f'gradient but was never processed '
                    f'(shape={t._shape}, contribution_count='
                    f'{contribution_count[id(t)]}, '
                    f'received={received[id(t)]})'
                )

    def _realize_segment_grads(self, loss_uop, initial_grad_uop,
                                targets, all_leaves, upstream_grads,
                                extra_bufs=None, inter_set=None):
        """Compute and realize gradients for one segment.

        loss_uop:   local loss UOp for this segment
        targets:    tensors to diff w.r.t. (params + intermediates)
        all_leaves: all leaf tensors in this segment (for bindings)
        upstream_grads: dict to store intermediate gradient data
        extra_bufs: additional (buffer, data) pairs for bindings
        inter_set:  set of id(t) for intermediate tensors (must be realized)
        """
        if inter_set is None:
            inter_set = set()
        n = len(targets)
        if n == 0:
            return

        # poly_grad_many: single reverse pass, all targets
        wrts = (_ffi._ptr * n)()
        for i, t in enumerate(targets):
            wrts[i] = t._uop
        out_grads = (_ffi._ptr * n)()

        ret = _ffi._lib.poly_grad_many(
            self._ctx, loss_uop, initial_grad_uop, wrts, n, out_grads)
        if ret != 0:
            raise RuntimeError('poly_grad_many failed')

        # Realize each gradient individually (poly_sink_n has multi-store
        # scheduling issues, so we realize one at a time for correctness)
        base_bindings = []
        seen_bufs = set()
        for l in all_leaves:
            bid = id(l._buffer)
            if bid not in seen_bufs:
                base_bindings.append((l._buffer, l._data.ctypes.data))
                seen_bufs.add(bid)
        for buf, data in (extra_bufs or []):
            bid = id(buf)
            if bid not in seen_bufs:
                base_bindings.append((buf, data.ctypes.data))
                seen_bufs.add(bid)

        for i, t in enumerate(targets):
            grad_val = out_grads[i]
            is_intermediate = id(t) in inter_set

            if is_intermediate:
                # Intermediates MUST be realized — their numpy data is needed
                # for upstream segment chaining via Kahn's algorithm.
                numel = int(np.prod(t._shape)) if t._shape else 1
                g_buf = _ffi._lib.poly_buffer_f32(self._ctx, numel)
                g_data = np.zeros(numel, dtype=np.float32)

                store = _ffi._lib.poly_store_val(self._ctx, g_buf, grad_val)
                sink = _ffi._lib.poly_sink1(self._ctx, store)

                bindings = base_bindings + [(g_buf, g_data.ctypes.data)]
                nb = len(bindings)
                c_bindings = (_ffi.PolyBufferBinding * nb)()
                for j, (buf, data) in enumerate(bindings):
                    c_bindings[j].buffer = buf
                    c_bindings[j].data = data

                ret = _ffi._lib.poly_realize(self._ctx, sink, c_bindings, nb)
                if ret != 0:
                    raise RuntimeError('Failed to realize segment gradient')

                # Accumulate upstream for intermediates (may receive from multiple segments)
                if id(t) in upstream_grads:
                    old_data, _ = upstream_grads[id(t)]
                    g_data = old_data + g_data
                # Create a fresh buffer for the accumulated data
                acc_buf = _ffi._lib.poly_buffer_f32(self._ctx, len(g_data))
                upstream_grads[id(t)] = (g_data, acc_buf)

                # Also set _grad if this intermediate wants gradients
                if t._requires_grad:
                    t._grad = Tensor(g_data.reshape(t._shape), _ctx=self._ctx)

            elif t._requires_grad:
                # Non-intermediate params/tensors: lazy gradient (UOp graph
                # wraps the gradient computation, realized when consumed).
                inputs = []
                for l in all_leaves:
                    inputs.append(Tensor(_ctx=self._ctx, _uop=l._uop, _buffer=l._buffer, _data=l._data, _shape=l._shape))
                if extra_bufs is not None:
                    for buf, data in extra_bufs:
                        inputs.append(Tensor(_ctx=self._ctx, _uop=buf, _buffer=buf, _data=data, _shape=(len(data),)))
                grad_tensor = Tensor(_ctx=self._ctx, _uop=grad_val, _shape=t._shape, _inputs=inputs)
                if t._grad is not None:
                    t._grad = t._grad + grad_tensor
                else:
                    t._grad = grad_tensor

    # --- Representation ---

    def __repr__(self):
        if self._is_leaf() and self._data is not None and self.numel() <= 16:
            data_str = str(self._data.reshape(self._shape).tolist())
            return f'Tensor({data_str}, shape={self._shape}, dtype={self.dtype})'
        return f'Tensor(shape={self._shape}, dtype={self.dtype})'

    def __len__(self):
        if not self._shape:
            raise TypeError('len() of scalar tensor')
        return self._shape[0]

    def __hash__(self):
        return id(self)

    def __bool__(self):
        return bool(self.item())


class CompiledStep:
    """Pre-compiled multi-kernel execution step.

    Wraps a C PolyStep*. The step owns compiled kernel programs and
    pre-allocated intermediate buffers. Call run() to execute with
    current buffer data. Not thread-safe (shared intermediates).
    """
    def __init__(self, step_ptr, buf_bindings):
        self._step = step_ptr
        self._buf_bindings = buf_bindings  # list of (buf_uop_ptr, tensor_or_data)

    @property
    def n_kernels(self):
        return _ffi._lib.poly_step_n_kernels(self._step)

    @property
    def n_intermediates(self):
        return _ffi._lib.poly_step_n_intermediates(self._step)

    def run(self):
        """Execute the step with current buffer data."""
        n = len(self._buf_bindings)
        c_bindings = (_ffi.PolyBufferBinding * n)()
        for i, (buf_uop, holder) in enumerate(self._buf_bindings):
            c_bindings[i].buffer = buf_uop
            if isinstance(holder, Tensor):
                c_bindings[i].data = ctypes.cast(
                    holder._data.ctypes.data, ctypes.c_void_p)
            else:
                c_bindings[i].data = ctypes.cast(
                    holder.ctypes.data, ctypes.c_void_p)
        ret = _ffi._lib.poly_step_run(self._step, c_bindings, n)
        if ret != 0:
            raise RuntimeError(f'poly_step_run failed ({ret})')

    def __del__(self):
        if hasattr(self, '_step') and self._step:
            _ffi._lib.poly_step_destroy(self._step)
            self._step = None
