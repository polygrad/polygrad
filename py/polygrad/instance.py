"""PolyInstance -- Python wrapper for the C PolyInstance runtime.

Provides forward pass, training, and weight I/O for models created from
IR bytes or family builders (MLP, etc.). This is the "product layer" API,
independent of the Tensor class used in polygrad's tinygrad-compatible frontend.
"""

import ctypes
import ctypes.util
import json
import numpy as np
from . import _ffi

_lib = _ffi._lib

# libc free for caller-frees byte arrays
_libc = ctypes.CDLL(ctypes.util.find_library('c'))
_libc.free.restype = None
_libc.free.argtypes = [ctypes.c_void_p]

# Role constants
ROLE_PARAM = 0
ROLE_INPUT = 1
ROLE_TARGET = 2
ROLE_OUTPUT = 3
ROLE_AUX = 4

# Optimizer constants
OPTIM_NONE = 0
OPTIM_SGD = 1
OPTIM_ADAM = 2
OPTIM_ADAMW = 3


class Instance:
    """Opaque model instance with forward, train, and weight I/O."""

    def __init__(self, ptr):
        if not ptr:
            raise RuntimeError('Failed to create PolyInstance (NULL pointer)')
        self._ptr = ptr

    def free(self):
        if self._ptr:
            _lib.poly_instance_free(self._ptr)
            self._ptr = None

    def __del__(self):
        if hasattr(self, '_ptr') and self._ptr:
            self.free()

    # ── Lifecycle ────────────────────────────────────────────────────

    @staticmethod
    def from_ir(ir_bytes, weights_bytes=None):
        """Create from IR binary + optional safetensors weights."""
        ir_buf = (ctypes.c_uint8 * len(ir_bytes)).from_buffer_copy(ir_bytes)
        w_buf = None
        w_len = 0
        if weights_bytes:
            w_buf = (ctypes.c_uint8 * len(weights_bytes)).from_buffer_copy(weights_bytes)
            w_len = len(weights_bytes)
        ptr = _lib.poly_instance_from_ir(ir_buf, len(ir_bytes), w_buf, w_len)
        return Instance(ptr)

    @staticmethod
    def mlp(spec):
        """Create an MLP from a spec dict or JSON string."""
        if isinstance(spec, dict):
            spec = json.dumps(spec)
        if isinstance(spec, str):
            spec = spec.encode('utf-8')
        ptr = _lib.poly_mlp_instance(spec, len(spec))
        return Instance(ptr)

    # ── Param Enumeration ────────────────────────────────────────────

    @property
    def param_count(self):
        return _lib.poly_instance_param_count(self._ptr)

    def param_name(self, i):
        name = _lib.poly_instance_param_name(self._ptr, i)
        return name.decode('utf-8') if name else None

    def param_shape(self, i):
        shape_buf = (ctypes.c_int64 * 8)()
        ndim = _lib.poly_instance_param_shape(self._ptr, i, shape_buf, 8)
        return tuple(shape_buf[d] for d in range(ndim))

    def param_data(self, i):
        """Return a numpy view of param data (mutable, zero-copy)."""
        numel = ctypes.c_int64(0)
        ptr = _lib.poly_instance_param_data(self._ptr, i, ctypes.byref(numel))
        if not ptr:
            return None
        return np.ctypeslib.as_array(ptr, shape=(numel.value,))

    def params(self):
        """Iterate (name, shape, data) for all params."""
        for i in range(self.param_count):
            yield self.param_name(i), self.param_shape(i), self.param_data(i)

    # ── Buffer Enumeration ───────────────────────────────────────────

    @property
    def buf_count(self):
        return _lib.poly_instance_buf_count(self._ptr)

    def buf_name(self, i):
        name = _lib.poly_instance_buf_name(self._ptr, i)
        return name.decode('utf-8') if name else None

    def buf_role(self, i):
        return _lib.poly_instance_buf_role(self._ptr, i)

    def buf_shape(self, i):
        shape_buf = (ctypes.c_int64 * 8)()
        ndim = _lib.poly_instance_buf_shape(self._ptr, i, shape_buf, 8)
        return tuple(shape_buf[d] for d in range(ndim))

    def buf_data(self, i):
        """Return a numpy view of buffer data (mutable, zero-copy)."""
        numel = ctypes.c_int64(0)
        ptr = _lib.poly_instance_buf_data(self._ptr, i, ctypes.byref(numel))
        if not ptr:
            return None
        return np.ctypeslib.as_array(ptr, shape=(numel.value,))

    def find_buf(self, name):
        """Find buffer index by name, or -1."""
        for i in range(self.buf_count):
            if self.buf_name(i) == name:
                return i
        return -1

    # ── Weight I/O ───────────────────────────────────────────────────

    def export_weights(self):
        """Export param weights as safetensors bytes."""
        out_len = ctypes.c_int(0)
        ptr = _lib.poly_instance_export_weights(self._ptr, ctypes.byref(out_len))
        if not ptr:
            return None
        data = bytes(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8 * out_len.value)).contents)
        _libc.free(ptr)
        return data

    def import_weights(self, data):
        """Import weights from safetensors bytes."""
        buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
        ret = _lib.poly_instance_import_weights(self._ptr, buf, len(data))
        if ret != 0:
            raise RuntimeError(f'import_weights failed (ret={ret})')

    def export_ir(self):
        """Export IR graph as binary bytes."""
        out_len = ctypes.c_int(0)
        ptr = _lib.poly_instance_export_ir(self._ptr, ctypes.byref(out_len))
        if not ptr:
            return None
        data = bytes(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8 * out_len.value)).contents)
        _libc.free(ptr)
        return data

    # ── Execution ────────────────────────────────────────────────────

    def set_optimizer(self, kind, lr=0.01, beta1=0.9, beta2=0.999,
                      eps=1e-8, weight_decay=0.0):
        """Configure optimizer before first train_step."""
        ret = _lib.poly_instance_set_optimizer(
            self._ptr, kind,
            ctypes.c_float(lr), ctypes.c_float(beta1), ctypes.c_float(beta2),
            ctypes.c_float(eps), ctypes.c_float(weight_decay))
        if ret != 0:
            raise RuntimeError(f'set_optimizer failed (ret={ret})')

    def forward(self, **inputs):
        """Run forward pass. Pass input arrays as keyword args (name=array).

        Returns dict of output buffer names to numpy arrays.
        """
        bindings, n = self._make_bindings(inputs)
        ret = _lib.poly_instance_forward(self._ptr, bindings, n)
        if ret != 0:
            raise RuntimeError(f'forward failed (ret={ret})')
        return self._collect_outputs()

    def train_step(self, **io):
        """Run one training step. Pass input+target arrays as kwargs.

        Returns the loss value (float).
        """
        bindings, n = self._make_bindings(io)
        loss = ctypes.c_float(0.0)
        ret = _lib.poly_instance_train_step(
            self._ptr, bindings, n, ctypes.byref(loss))
        if ret != 0:
            raise RuntimeError(f'train_step failed (ret={ret})')
        return float(loss.value)

    # ── Internals ────────────────────────────────────────────────────

    def _make_bindings(self, io_dict):
        """Convert {name: array} dict to PolyIOBinding array."""
        n = len(io_dict)
        arr = (_ffi.PolyIOBinding * n)()
        for i, (name, data) in enumerate(io_dict.items()):
            if isinstance(data, np.ndarray):
                data = np.ascontiguousarray(data, dtype=np.float32)
            else:
                data = np.ascontiguousarray(data, dtype=np.float32)
            arr[i].name = name.encode('utf-8')
            arr[i].data = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        return arr, n

    def _collect_outputs(self):
        """Read all output buffers into a dict."""
        result = {}
        for i in range(self.buf_count):
            if self.buf_role(i) == ROLE_OUTPUT:
                name = self.buf_name(i)
                data = self.buf_data(i)
                if data is not None:
                    result[name] = data.copy()
        return result
