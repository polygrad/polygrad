"""
ctypes bindings to libpolygrad.so — uses only the frontend.h surface
to avoid passing PolyArg/PolyDType structs across FFI.
"""

import ctypes
import os
import pathlib

# Find libpolygrad.so
_default_path = pathlib.Path(__file__).resolve().parents[2] / 'build' / 'libpolygrad.so'
_lib_path = os.environ.get('POLYGRAD_LIB', str(_default_path))

_lib = ctypes.CDLL(_lib_path)

# --- Opaque pointer type ---
_ptr = ctypes.c_void_p
_i64p = ctypes.POINTER(ctypes.c_int64)
_ip = ctypes.POINTER(ctypes.c_int)

# --- Context ---
_lib.poly_ctx_new.restype = _ptr
_lib.poly_ctx_new.argtypes = []

_lib.poly_ctx_destroy.restype = None
_lib.poly_ctx_destroy.argtypes = [_ptr]

# --- Op helpers ---
_lib.poly_op_count.restype = ctypes.c_int
_lib.poly_op_count.argtypes = []

_lib.poly_op_name.restype = ctypes.c_char_p
_lib.poly_op_name.argtypes = [ctypes.c_int]

# Build op name -> int mapping at import time
_op_count = _lib.poly_op_count()
OPS = {}
for _i in range(_op_count):
    _name = _lib.poly_op_name(_i)
    if _name:
        OPS[_name.decode()] = _i

# --- Frontend helpers (frontend.h) ---
_lib.poly_const_float.restype = _ptr
_lib.poly_const_float.argtypes = [_ptr, ctypes.c_double]

_lib.poly_const_int.restype = _ptr
_lib.poly_const_int.argtypes = [_ptr, ctypes.c_int64]

_lib.poly_alu1.restype = _ptr
_lib.poly_alu1.argtypes = [_ptr, ctypes.c_int, _ptr]

_lib.poly_alu2.restype = _ptr
_lib.poly_alu2.argtypes = [_ptr, ctypes.c_int, _ptr, _ptr]

_lib.poly_alu3.restype = _ptr
_lib.poly_alu3.argtypes = [_ptr, ctypes.c_int, _ptr, _ptr, _ptr]

_lib.poly_store_val.restype = _ptr
_lib.poly_store_val.argtypes = [_ptr, _ptr, _ptr]

_lib.poly_sink1.restype = _ptr
_lib.poly_sink1.argtypes = [_ptr, _ptr]

_lib.poly_sink_n.restype = _ptr
_lib.poly_sink_n.argtypes = [_ptr, ctypes.POINTER(_ptr), ctypes.c_int]

_lib.poly_buffer_f32.restype = _ptr
_lib.poly_buffer_f32.argtypes = [_ptr, ctypes.c_int64]

# --- Composed elementwise ops (shape-free) ---

def _unary(name):
    fn = getattr(_lib, name)
    fn.restype = _ptr
    fn.argtypes = [_ptr, _ptr]

def _unary_d(name):
    fn = getattr(_lib, name)
    fn.restype = _ptr
    fn.argtypes = [_ptr, _ptr, ctypes.c_double]

def _unary_dd(name):
    fn = getattr(_lib, name)
    fn.restype = _ptr
    fn.argtypes = [_ptr, _ptr, ctypes.c_double, ctypes.c_double]

def _binary(name):
    fn = getattr(_lib, name)
    fn.restype = _ptr
    fn.argtypes = [_ptr, _ptr, _ptr]

# Math
for _n in ['poly_exp', 'poly_log', 'poly_sin', 'poly_cos', 'poly_tan',
           'poly_sigmoid', 'poly_tanh_act', 'poly_abs', 'poly_sign',
           'poly_square', 'poly_rsqrt', 'poly_ceil', 'poly_floor',
           'poly_round_f', 'poly_isinf', 'poly_isnan',
           'poly_relu', 'poly_relu6', 'poly_gelu', 'poly_quick_gelu',
           'poly_silu', 'poly_mish', 'poly_hardswish', 'poly_hardsigmoid']:
    _unary(_n)

for _n in ['poly_leaky_relu', 'poly_elu', 'poly_softplus']:
    _unary_d(_n)

_unary_dd('poly_hardtanh')

# Comparisons
for _n in ['poly_eq', 'poly_ne', 'poly_gt', 'poly_ge', 'poly_le',
           'poly_maximum', 'poly_minimum']:
    _binary(_n)

_lib.poly_where_op.restype = _ptr
_lib.poly_where_op.argtypes = [_ptr, _ptr, _ptr, _ptr]

_lib.poly_clamp.restype = _ptr
_lib.poly_clamp.argtypes = [_ptr, _ptr, ctypes.c_double, ctypes.c_double]

# --- Shape-aware composed ops ---

_lib.poly_sum_reduce.restype = _ptr
_lib.poly_sum_reduce.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int,
                                  ctypes.c_int, ctypes.c_int, _i64p, _ip]

_lib.poly_max_reduce.restype = _ptr
_lib.poly_max_reduce.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int,
                                  ctypes.c_int, ctypes.c_int, _i64p, _ip]

_lib.poly_mean_reduce.restype = _ptr
_lib.poly_mean_reduce.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_int, _i64p, _ip]

_lib.poly_var_reduce.restype = _ptr
_lib.poly_var_reduce.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int,
                                  ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                  _i64p, _ip]

_lib.poly_dot.restype = _ptr
_lib.poly_dot.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int,
                           _ptr, _i64p, ctypes.c_int, _i64p, _ip]

_lib.poly_softmax.restype = _ptr
_lib.poly_softmax.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int, ctypes.c_int]

_lib.poly_log_softmax.restype = _ptr
_lib.poly_log_softmax.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int, ctypes.c_int]

# --- ASSIGN ---
_lib.poly_assign.restype = _ptr
_lib.poly_assign.argtypes = [_ptr, _ptr, _ptr]

# --- Dynamic shapes (DEFINE_VAR / BIND) ---
_lib.poly_define_var.restype = _ptr
_lib.poly_define_var.argtypes = [_ptr, ctypes.c_char_p, ctypes.c_int64, ctypes.c_int64]

_lib.poly_bind_var.restype = _ptr
_lib.poly_bind_var.argtypes = [_ptr, _ptr, ctypes.c_int64]

_lib.poly_buffer_var.restype = _ptr
_lib.poly_buffer_var.argtypes = [_ptr, ctypes.c_int, _ptr, _i64p, ctypes.c_int]

class PolyVarBinding(ctypes.Structure):
    _fields_ = [('var', _ptr), ('value', ctypes.c_int32)]

# Note: poly_realize_ex argtypes defined after PolyBufferBinding (Realize section)

# --- Sched helpers (sched.h) ---
_lib.poly_reshape.restype = _ptr
_lib.poly_reshape.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int]

_lib.poly_expand.restype = _ptr
_lib.poly_expand.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int]

_lib.poly_reduce_axis.restype = _ptr
_lib.poly_reduce_axis.argtypes = [_ptr, ctypes.c_int, _ptr, _i64p, ctypes.c_int]

_lib.poly_permute.restype = _ptr
_lib.poly_permute.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int]

_lib.poly_shrink.restype = _ptr
_lib.poly_shrink.argtypes = [_ptr, _ptr, ctypes.c_void_p, ctypes.c_int]

_lib.poly_flip.restype = _ptr
_lib.poly_flip.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int]

_lib.poly_pad.restype = _ptr
_lib.poly_pad.argtypes = [_ptr, _ptr, ctypes.c_void_p, ctypes.c_int]

# --- Autograd ---
_lib.poly_grad.restype = _ptr
_lib.poly_grad.argtypes = [_ptr, _ptr, _ptr]

_lib.poly_uop_substitute.restype = _ptr
_lib.poly_uop_substitute.argtypes = [_ptr, _ptr, ctypes.POINTER(_ptr), ctypes.POINTER(_ptr), ctypes.c_int]

_lib.poly_grad_many.restype = ctypes.c_int
_lib.poly_grad_many.argtypes = [_ptr, _ptr, _ptr, ctypes.POINTER(_ptr), ctypes.c_int, ctypes.POINTER(_ptr)]

# --- Realize ---
class PolyBufferBinding(ctypes.Structure):
    _fields_ = [('buffer', _ptr), ('data', _ptr)]

_lib.poly_realize.restype = ctypes.c_int
_lib.poly_realize.argtypes = [_ptr, _ptr, ctypes.POINTER(PolyBufferBinding), ctypes.c_int]

_lib.poly_realize_ex.restype = ctypes.c_int
_lib.poly_realize_ex.argtypes = [_ptr, _ptr,
    ctypes.POINTER(PolyBufferBinding), ctypes.c_int,
    ctypes.POINTER(PolyVarBinding), ctypes.c_int]

# --- CUDA realize ---
# These are conditionally available (only if libpolygrad.so was built with POLY_HAS_CUDA)
_has_cuda_ffi = hasattr(_lib, 'poly_realize_cuda')
if _has_cuda_ffi:
    _lib.poly_realize_cuda.restype = ctypes.c_int
    _lib.poly_realize_cuda.argtypes = [_ptr, _ptr, ctypes.POINTER(PolyBufferBinding), ctypes.c_int]

    _lib.poly_cuda_copyback.restype = ctypes.c_int
    _lib.poly_cuda_copyback.argtypes = [ctypes.POINTER(PolyBufferBinding), ctypes.c_int]

    _lib.poly_cuda_flush_buffers.restype = None
    _lib.poly_cuda_flush_buffers.argtypes = []

    _lib.poly_cuda_prog_cache_flush.restype = None
    _lib.poly_cuda_prog_cache_flush.argtypes = []

    _lib.poly_cuda_available.restype = ctypes.c_bool
    _lib.poly_cuda_available.argtypes = []

# --- Einsum ---
_lib.poly_einsum.restype = _ptr
_lib.poly_einsum.argtypes = [
    _ptr,                                  # ctx
    ctypes.c_char_p,                       # formula
    ctypes.POINTER(_ptr),                  # tensors[]
    ctypes.POINTER(ctypes.POINTER(ctypes.c_int64)),  # shapes[]
    ctypes.POINTER(ctypes.c_int),          # ndims[]
    ctypes.c_int,                          # n_tensors
    ctypes.POINTER(ctypes.c_int64),        # out_shape
    _ip,                                   # out_ndim
]

# --- Rearrange ---
_lib.poly_rearrange.restype = _ptr
_lib.poly_rearrange.argtypes = [
    _ptr,                                  # ctx
    ctypes.c_char_p,                       # formula
    _ptr,                                  # x
    ctypes.POINTER(ctypes.c_int64),        # shape
    ctypes.c_int,                          # ndim
    ctypes.c_char_p,                       # axis_names
    ctypes.POINTER(ctypes.c_int64),        # axis_values
    ctypes.c_int,                          # n_axis_sizes
    ctypes.POINTER(ctypes.c_int64),        # out_shape
    _ip,                                   # out_ndim
]

# --- Compiled Step ---
_lib.poly_compile_step.restype = _ptr
_lib.poly_compile_step.argtypes = [_ptr, _ptr]

_lib.poly_step_run.restype = ctypes.c_int
_lib.poly_step_run.argtypes = [_ptr, ctypes.POINTER(PolyBufferBinding), ctypes.c_int]

_lib.poly_step_run_ex.restype = ctypes.c_int
_lib.poly_step_run_ex.argtypes = [_ptr,
    ctypes.POINTER(PolyBufferBinding), ctypes.c_int,
    ctypes.POINTER(PolyVarBinding), ctypes.c_int]

_lib.poly_step_destroy.restype = None
_lib.poly_step_destroy.argtypes = [_ptr]

_lib.poly_step_n_kernels.restype = ctypes.c_int
_lib.poly_step_n_kernels.argtypes = [_ptr]

_lib.poly_step_n_intermediates.restype = ctypes.c_int
_lib.poly_step_n_intermediates.argtypes = [_ptr]
