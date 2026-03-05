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

_lib.poly_const_double.restype = _ptr
_lib.poly_const_double.argtypes = [_ptr, ctypes.c_double]

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

_lib.poly_buffer_f64.restype = _ptr
_lib.poly_buffer_f64.argtypes = [_ptr, ctypes.c_int64]

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
for _n in ['poly_exp', 'poly_log', 'poly_log1p', 'poly_expm1',
           'poly_sin', 'poly_cos', 'poly_tan',
           'poly_erf', 'poly_erfc', 'poly_erfinv', 'poly_ndtri',
           'poly_digamma', 'poly_lgamma',
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

_lib.poly_detach.restype = _ptr
_lib.poly_detach.argtypes = [_ptr, _ptr]

_lib.poly_rand.restype = _ptr
_lib.poly_rand.argtypes = [_ptr, _i64p, ctypes.c_int, ctypes.c_uint64]

_lib.poly_randn.restype = _ptr
_lib.poly_randn.argtypes = [_ptr, _i64p, ctypes.c_int, ctypes.c_uint64]

_lib.poly_arange.restype = _ptr
_lib.poly_arange.argtypes = [_ptr, ctypes.c_double, ctypes.c_double, ctypes.c_double]

_lib.poly_eye.restype = _ptr
_lib.poly_eye.argtypes = [_ptr, ctypes.c_int64]

_lib.poly_linspace.restype = _ptr
_lib.poly_linspace.argtypes = [_ptr, ctypes.c_double, ctypes.c_double, ctypes.c_int64]

_lib.poly_full.restype = _ptr
_lib.poly_full.argtypes = [_ptr, _i64p, ctypes.c_int, ctypes.c_double]

_lib.poly_tril.restype = _ptr
_lib.poly_tril.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int, ctypes.c_int]

_lib.poly_triu.restype = _ptr
_lib.poly_triu.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int, ctypes.c_int]

_lib.poly_cholesky.restype = _ptr
_lib.poly_cholesky.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int, ctypes.c_int]

_lib.poly_triangular_solve.restype = _ptr
_lib.poly_triangular_solve.argtypes = [
    _ptr, _ptr, _i64p, ctypes.c_int,
    _ptr, _i64p, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    _i64p, _ip
]

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

_lib.poly_logsumexp.restype = _ptr
_lib.poly_logsumexp.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int,
                                 ctypes.c_int, ctypes.c_int, _i64p, _ip]

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

_lib.poly_compile_value_and_grad.restype = _ptr
_lib.poly_compile_value_and_grad.argtypes = [
    _ptr,                              # ctx
    _ptr,                              # loss
    ctypes.POINTER(_ptr),              # params[]
    ctypes.c_int,                      # n_params
    _ip,                               # out_loss_buf_idx
    _ip,                               # out_grad_buf_idxs[]
]

_lib.poly_step_run.restype = ctypes.c_int
_lib.poly_step_run.argtypes = [_ptr, ctypes.POINTER(PolyBufferBinding), ctypes.c_int]

_lib.poly_step_run_ex.restype = ctypes.c_int
_lib.poly_step_run_ex.argtypes = [_ptr,
    ctypes.POINTER(PolyBufferBinding), ctypes.c_int,
    ctypes.POINTER(PolyVarBinding), ctypes.c_int]

_lib.poly_step_run_indexed.restype = ctypes.c_int
_lib.poly_step_run_indexed.argtypes = [_ptr, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]

_lib.poly_step_run_indexed_ex.restype = ctypes.c_int
_lib.poly_step_run_indexed_ex.argtypes = [_ptr, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
    ctypes.POINTER(PolyVarBinding), ctypes.c_int]

_lib.poly_step_destroy.restype = None
_lib.poly_step_destroy.argtypes = [_ptr]

_lib.poly_step_n_kernels.restype = ctypes.c_int
_lib.poly_step_n_kernels.argtypes = [_ptr]

_lib.poly_step_n_intermediates.restype = ctypes.c_int
_lib.poly_step_n_intermediates.argtypes = [_ptr]

_lib.poly_step_n_buffers.restype = ctypes.c_int
_lib.poly_step_n_buffers.argtypes = [_ptr]

_lib.poly_step_n_bindable_buffers.restype = ctypes.c_int
_lib.poly_step_n_bindable_buffers.argtypes = [_ptr]

class PolyDType(ctypes.Structure):
    _fields_ = [
        ('priority', ctypes.c_int8),
        ('bitsize', ctypes.c_uint16),
        ('name', ctypes.c_char_p),
        ('fmt', ctypes.c_char),
        ('count', ctypes.c_uint16),
        ('is_ptr', ctypes.c_bool),
        ('addrspace', ctypes.c_int),
        ('vcount', ctypes.c_uint16),
        ('ptr_size', ctypes.c_int64),
    ]

POLY_STEP_BUF_INPUT = 0
POLY_STEP_BUF_OUTPUT = 1
POLY_STEP_BUF_TEMP = 2
POLY_STEP_BUF_CONSTANT = 3

class PolyStepBufferInfo(ctypes.Structure):
    _fields_ = [
        ('version', ctypes.c_int),
        ('index', ctypes.c_int),
        ('role', ctypes.c_int),
        ('dtype', PolyDType),
        ('numel', ctypes.c_int64),
        ('nbytes', ctypes.c_int64),
    ]

_lib.poly_step_buffer_info.restype = ctypes.c_int
_lib.poly_step_buffer_info.argtypes = [_ptr, ctypes.c_int, ctypes.POINTER(PolyStepBufferInfo)]

# --- WASM step plan ---
_lib.poly_render_step_wasm_plan.restype = _ptr
_lib.poly_render_step_wasm_plan.argtypes = [_ptr, _ptr]

_lib.poly_wasm_stepplan_n_kernels.restype = ctypes.c_int
_lib.poly_wasm_stepplan_n_kernels.argtypes = [_ptr]

_lib.poly_wasm_stepplan_kernel_bytes.restype = ctypes.POINTER(ctypes.c_uint8)
_lib.poly_wasm_stepplan_kernel_bytes.argtypes = [_ptr, ctypes.c_int, _ip]

_lib.poly_wasm_stepplan_kernel_n_params.restype = ctypes.c_int
_lib.poly_wasm_stepplan_kernel_n_params.argtypes = [_ptr, ctypes.c_int]

_lib.poly_wasm_stepplan_n_buffers.restype = ctypes.c_int
_lib.poly_wasm_stepplan_n_buffers.argtypes = [_ptr]

_lib.poly_wasm_stepplan_n_bindable_buffers.restype = ctypes.c_int
_lib.poly_wasm_stepplan_n_bindable_buffers.argtypes = [_ptr]

_lib.poly_wasm_stepplan_kernel_param_buf_index.restype = ctypes.c_int
_lib.poly_wasm_stepplan_kernel_param_buf_index.argtypes = [_ptr, ctypes.c_int, ctypes.c_int]

_lib.poly_wasm_stepplan_exec_order.restype = _ip
_lib.poly_wasm_stepplan_exec_order.argtypes = [_ptr, _ip]

_lib.poly_wasm_stepplan_destroy.restype = None
_lib.poly_wasm_stepplan_destroy.argtypes = [_ptr]

# --- PolyInstance (instance.h) ---

class PolyIOBinding(ctypes.Structure):
    _fields_ = [('name', ctypes.c_char_p),
                ('data', ctypes.POINTER(ctypes.c_float))]

_u8p = ctypes.POINTER(ctypes.c_uint8)
_fp = ctypes.POINTER(ctypes.c_float)

# Lifecycle
_lib.poly_instance_from_ir.restype = _ptr
_lib.poly_instance_from_ir.argtypes = [_u8p, ctypes.c_int, _u8p, ctypes.c_int]

_lib.poly_instance_free.restype = None
_lib.poly_instance_free.argtypes = [_ptr]

# Param enumeration
_lib.poly_instance_param_count.restype = ctypes.c_int
_lib.poly_instance_param_count.argtypes = [_ptr]

_lib.poly_instance_param_name.restype = ctypes.c_char_p
_lib.poly_instance_param_name.argtypes = [_ptr, ctypes.c_int]

_lib.poly_instance_param_shape.restype = ctypes.c_int
_lib.poly_instance_param_shape.argtypes = [_ptr, ctypes.c_int, _i64p, ctypes.c_int]

_lib.poly_instance_param_data.restype = _fp
_lib.poly_instance_param_data.argtypes = [_ptr, ctypes.c_int, _i64p]

# Buffer enumeration
_lib.poly_instance_buf_count.restype = ctypes.c_int
_lib.poly_instance_buf_count.argtypes = [_ptr]

_lib.poly_instance_buf_name.restype = ctypes.c_char_p
_lib.poly_instance_buf_name.argtypes = [_ptr, ctypes.c_int]

_lib.poly_instance_buf_role.restype = ctypes.c_int
_lib.poly_instance_buf_role.argtypes = [_ptr, ctypes.c_int]

_lib.poly_instance_buf_shape.restype = ctypes.c_int
_lib.poly_instance_buf_shape.argtypes = [_ptr, ctypes.c_int, _i64p, ctypes.c_int]

_lib.poly_instance_buf_data.restype = _fp
_lib.poly_instance_buf_data.argtypes = [_ptr, ctypes.c_int, _i64p]

# Weight I/O
_lib.poly_instance_export_weights.restype = _u8p
_lib.poly_instance_export_weights.argtypes = [_ptr, _ip]

_lib.poly_instance_import_weights.restype = ctypes.c_int
_lib.poly_instance_import_weights.argtypes = [_ptr, _u8p, ctypes.c_int]

# IR export
_lib.poly_instance_export_ir.restype = _u8p
_lib.poly_instance_export_ir.argtypes = [_ptr, _ip]

# Execution
_lib.poly_instance_forward.restype = ctypes.c_int
_lib.poly_instance_forward.argtypes = [_ptr, ctypes.POINTER(PolyIOBinding), ctypes.c_int]

_lib.poly_instance_train_step.restype = ctypes.c_int
_lib.poly_instance_train_step.argtypes = [_ptr, ctypes.POINTER(PolyIOBinding), ctypes.c_int, _fp]

_lib.poly_instance_set_optimizer.restype = ctypes.c_int
_lib.poly_instance_set_optimizer.argtypes = [_ptr, ctypes.c_int,
    ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ctypes.c_float, ctypes.c_float]

# MLP family builder (model_mlp.h)
_lib.poly_mlp_instance.restype = _ptr
_lib.poly_mlp_instance.argtypes = [ctypes.c_char_p, ctypes.c_int]

# TabM family builder (model_tabm.h)
_lib.poly_tabm_instance.restype = _ptr
_lib.poly_tabm_instance.argtypes = [ctypes.c_char_p, ctypes.c_int]

# NAM family builder (model_nam.h)
_lib.poly_nam_instance.restype = _ptr
_lib.poly_nam_instance.argtypes = [ctypes.c_char_p, ctypes.c_int]

# HF loader (modelzoo/hf_loader.c)
_lib.poly_hf_load.restype = _ptr
_lib.poly_hf_load.argtypes = [
    ctypes.c_char_p,                       # config_json
    ctypes.c_int,                          # config_len
    ctypes.POINTER(_u8p),                  # weight_files[]
    ctypes.POINTER(ctypes.c_int64),        # weight_lens[]
    ctypes.c_int,                          # n_weight_files
    ctypes.c_int,                          # max_batch
    ctypes.c_int,                          # max_seq_len
]
