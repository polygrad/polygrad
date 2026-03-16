"""
ctypes bindings to libpolygrad.so -- uses only the frontend.h surface
to avoid passing PolyArg/PolyDType structs across FFI.

Library is loaded lazily on first call to get_lib(). Module-level globals
(_lib, OPS, _has_cuda_ffi) are populated atomically by get_lib().
"""

import ctypes
import ctypes.util
import os
import platform
import sys

# --- Module-level state (populated by get_lib()) ---
_lib = None
OPS = {}
_has_cuda_ffi = False

# --- Opaque pointer type (always available) ---
_ptr = ctypes.c_void_p
_i64p = ctypes.POINTER(ctypes.c_int64)
_ip = ctypes.POINTER(ctypes.c_int)


# --- Structures (always available, no _lib dependency) ---

class PolyVarBinding(ctypes.Structure):
    _fields_ = [('var', _ptr), ('value', ctypes.c_int32)]

class PolyBufferHandle(ctypes.Structure):
    _fields_ = [
        ('ptr', _ptr),
        ('nbytes', ctypes.c_size_t),
        ('domain', ctypes.c_int),    # PolyDeviceId
        ('owned', ctypes.c_bool),
    ]

class PolyBufferBinding(ctypes.Structure):
    _fields_ = [('buffer', _ptr), ('handle', PolyBufferHandle)]

class PolyIOBinding(ctypes.Structure):
    _fields_ = [('name', ctypes.c_char_p),
                ('data', ctypes.POINTER(ctypes.c_float))]

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


# --- Library discovery ---

def _find_lib():
    """Find libpolygrad shared library. Returns path or None."""
    # 1. POLYGRAD_LIB env var (explicit override)
    env_path = os.environ.get('POLYGRAD_LIB')
    if env_path and os.path.isfile(env_path):
        return env_path

    # 2. Installed _native extension module (pip install)
    # Scan package directory directly to avoid circular import
    # (find_spec('polygrad._native') would trigger polygrad.__init__)
    import importlib.machinery
    this_dir = os.path.dirname(os.path.abspath(__file__))
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        native_path = os.path.join(this_dir, '_native' + suffix)
        if os.path.isfile(native_path):
            return native_path

    # 3. Development layout (relative to this file)
    this_dir = os.path.dirname(os.path.abspath(__file__))
    system = platform.system()

    if system == 'Darwin':
        lib_name = 'libpolygrad.dylib'
    else:
        lib_name = 'libpolygrad.so'

    search_paths = [
        os.path.join(this_dir, lib_name),
        os.path.join(this_dir, '..', '..', 'build', lib_name),
    ]

    for path in search_paths:
        resolved = os.path.abspath(path)
        if os.path.isfile(resolved):
            return resolved

    # 4. System library path
    found = ctypes.util.find_library('polygrad')
    if found:
        return found

    return None


# --- Signature declaration helpers ---

def _unary(lib, name):
    fn = getattr(lib, name)
    fn.restype = _ptr
    fn.argtypes = [_ptr, _ptr]

def _unary_d(lib, name):
    fn = getattr(lib, name)
    fn.restype = _ptr
    fn.argtypes = [_ptr, _ptr, ctypes.c_double]

def _unary_dd(lib, name):
    fn = getattr(lib, name)
    fn.restype = _ptr
    fn.argtypes = [_ptr, _ptr, ctypes.c_double, ctypes.c_double]

def _binary(lib, name):
    fn = getattr(lib, name)
    fn.restype = _ptr
    fn.argtypes = [_ptr, _ptr, _ptr]


def _declare_signatures(lib):
    """Declare all ctypes argtypes/restype on the loaded library."""

    _u8p = ctypes.POINTER(ctypes.c_uint8)
    _fp = ctypes.POINTER(ctypes.c_float)

    # --- Context ---
    lib.poly_ctx_new.restype = _ptr
    lib.poly_ctx_new.argtypes = []

    lib.poly_ctx_destroy.restype = None
    lib.poly_ctx_destroy.argtypes = [_ptr]

    # --- Op helpers ---
    lib.poly_op_count.restype = ctypes.c_int
    lib.poly_op_count.argtypes = []

    lib.poly_op_name.restype = ctypes.c_char_p
    lib.poly_op_name.argtypes = [ctypes.c_int]

    # --- Frontend helpers (frontend.h) ---
    lib.poly_const_float.restype = _ptr
    lib.poly_const_float.argtypes = [_ptr, ctypes.c_double]

    lib.poly_const_double.restype = _ptr
    lib.poly_const_double.argtypes = [_ptr, ctypes.c_double]

    lib.poly_const_int.restype = _ptr
    lib.poly_const_int.argtypes = [_ptr, ctypes.c_int64]

    lib.poly_alu1.restype = _ptr
    lib.poly_alu1.argtypes = [_ptr, ctypes.c_int, _ptr]

    lib.poly_alu2.restype = _ptr
    lib.poly_alu2.argtypes = [_ptr, ctypes.c_int, _ptr, _ptr]

    lib.poly_alu3.restype = _ptr
    lib.poly_alu3.argtypes = [_ptr, ctypes.c_int, _ptr, _ptr, _ptr]

    lib.poly_store_val.restype = _ptr
    lib.poly_store_val.argtypes = [_ptr, _ptr, _ptr]

    lib.poly_sink1.restype = _ptr
    lib.poly_sink1.argtypes = [_ptr, _ptr]

    lib.poly_sink_n.restype = _ptr
    lib.poly_sink_n.argtypes = [_ptr, ctypes.POINTER(_ptr), ctypes.c_int]

    lib.poly_buffer_f32.restype = _ptr
    lib.poly_buffer_f32.argtypes = [_ptr, ctypes.c_int64]

    lib.poly_buffer_f64.restype = _ptr
    lib.poly_buffer_f64.argtypes = [_ptr, ctypes.c_int64]

    lib.poly_buffer_by_id.restype = _ptr
    lib.poly_buffer_by_id.argtypes = [_ptr, ctypes.c_int64, ctypes.c_int]

    lib.poly_cast_by_id.restype = _ptr
    lib.poly_cast_by_id.argtypes = [_ptr, _ptr, ctypes.c_int]

    # --- Composed elementwise ops (shape-free) ---
    for n in ['poly_exp', 'poly_log', 'poly_log1p', 'poly_expm1',
              'poly_sin', 'poly_cos', 'poly_tan',
              'poly_erf', 'poly_erfc', 'poly_erfinv', 'poly_ndtri',
              'poly_digamma', 'poly_lgamma',
              'poly_sigmoid', 'poly_tanh_act', 'poly_abs', 'poly_sign',
              'poly_square', 'poly_rsqrt', 'poly_ceil', 'poly_floor',
              'poly_round_f', 'poly_isinf', 'poly_isnan',
              'poly_relu', 'poly_relu6', 'poly_gelu', 'poly_quick_gelu',
              'poly_silu', 'poly_mish', 'poly_hardswish', 'poly_hardsigmoid']:
        _unary(lib, n)

    for n in ['poly_leaky_relu', 'poly_elu', 'poly_softplus']:
        _unary_d(lib, n)

    _unary_dd(lib, 'poly_hardtanh')

    # Comparisons
    for n in ['poly_eq', 'poly_ne', 'poly_gt', 'poly_ge', 'poly_le',
              'poly_maximum', 'poly_minimum']:
        _binary(lib, n)

    lib.poly_where_op.restype = _ptr
    lib.poly_where_op.argtypes = [_ptr, _ptr, _ptr, _ptr]

    lib.poly_clamp.restype = _ptr
    lib.poly_clamp.argtypes = [_ptr, _ptr, ctypes.c_double, ctypes.c_double]

    lib.poly_detach.restype = _ptr
    lib.poly_detach.argtypes = [_ptr, _ptr]

    lib.poly_rand.restype = _ptr
    lib.poly_rand.argtypes = [_ptr, _i64p, ctypes.c_int, ctypes.c_uint64]

    lib.poly_randn.restype = _ptr
    lib.poly_randn.argtypes = [_ptr, _i64p, ctypes.c_int, ctypes.c_uint64]

    lib.poly_arange.restype = _ptr
    lib.poly_arange.argtypes = [_ptr, ctypes.c_double, ctypes.c_double, ctypes.c_double]

    lib.poly_eye.restype = _ptr
    lib.poly_eye.argtypes = [_ptr, ctypes.c_int64]

    lib.poly_linspace.restype = _ptr
    lib.poly_linspace.argtypes = [_ptr, ctypes.c_double, ctypes.c_double, ctypes.c_int64]

    lib.poly_full.restype = _ptr
    lib.poly_full.argtypes = [_ptr, _i64p, ctypes.c_int, ctypes.c_double]

    lib.poly_tril.restype = _ptr
    lib.poly_tril.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int, ctypes.c_int]

    lib.poly_triu.restype = _ptr
    lib.poly_triu.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int, ctypes.c_int]

    lib.poly_cholesky.restype = _ptr
    lib.poly_cholesky.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int, ctypes.c_int]

    lib.poly_triangular_solve.restype = _ptr
    lib.poly_triangular_solve.argtypes = [
        _ptr, _ptr, _i64p, ctypes.c_int,
        _ptr, _i64p, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        _i64p, _ip
    ]

    # --- Shape-aware composed ops ---
    lib.poly_sum_reduce.restype = _ptr
    lib.poly_sum_reduce.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int,
                                     ctypes.c_int, ctypes.c_int, _i64p, _ip]

    lib.poly_max_reduce.restype = _ptr
    lib.poly_max_reduce.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int,
                                     ctypes.c_int, ctypes.c_int, _i64p, _ip]

    lib.poly_mean_reduce.restype = _ptr
    lib.poly_mean_reduce.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int,
                                      ctypes.c_int, ctypes.c_int, _i64p, _ip]

    lib.poly_var_reduce.restype = _ptr
    lib.poly_var_reduce.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int,
                                     ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                     _i64p, _ip]

    lib.poly_logsumexp.restype = _ptr
    lib.poly_logsumexp.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int, _i64p, _ip]

    lib.poly_dot.restype = _ptr
    lib.poly_dot.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int,
                              _ptr, _i64p, ctypes.c_int, _i64p, _ip]

    lib.poly_cross_entropy.restype = _ptr
    lib.poly_cross_entropy.argtypes = [
        _ptr, _ptr, _i64p, ctypes.c_int,
        _ptr, _i64p, ctypes.c_int,
        ctypes.c_int, _i64p, _ip
    ]

    lib.poly_softmax.restype = _ptr
    lib.poly_softmax.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int, ctypes.c_int]

    lib.poly_log_softmax.restype = _ptr
    lib.poly_log_softmax.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int, ctypes.c_int]

    # --- ASSIGN ---
    lib.poly_assign.restype = _ptr
    lib.poly_assign.argtypes = [_ptr, _ptr, _ptr]

    # --- Dynamic shapes (DEFINE_VAR / BIND) ---
    lib.poly_define_var.restype = _ptr
    lib.poly_define_var.argtypes = [_ptr, ctypes.c_char_p, ctypes.c_int64, ctypes.c_int64]

    lib.poly_bind_var.restype = _ptr
    lib.poly_bind_var.argtypes = [_ptr, _ptr, ctypes.c_int64]

    lib.poly_buffer_var.restype = _ptr
    lib.poly_buffer_var.argtypes = [_ptr, ctypes.c_int, _ptr, _i64p, ctypes.c_int]

    # --- Sched helpers (sched.h) ---
    lib.poly_reshape.restype = _ptr
    lib.poly_reshape.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int]

    lib.poly_expand.restype = _ptr
    lib.poly_expand.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int]

    lib.poly_reduce_axis.restype = _ptr
    lib.poly_reduce_axis.argtypes = [_ptr, ctypes.c_int, _ptr, _i64p, ctypes.c_int]

    lib.poly_permute.restype = _ptr
    lib.poly_permute.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int]

    lib.poly_shrink.restype = _ptr
    lib.poly_shrink.argtypes = [_ptr, _ptr, ctypes.c_void_p, ctypes.c_int]

    lib.poly_flip.restype = _ptr
    lib.poly_flip.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int]

    lib.poly_pad.restype = _ptr
    lib.poly_pad.argtypes = [_ptr, _ptr, ctypes.c_void_p, ctypes.c_int]

    # --- Autograd ---
    lib.poly_grad.restype = _ptr
    lib.poly_grad.argtypes = [_ptr, _ptr, _ptr]

    lib.poly_uop_substitute.restype = _ptr
    lib.poly_uop_substitute.argtypes = [_ptr, _ptr, ctypes.POINTER(_ptr), ctypes.POINTER(_ptr), ctypes.c_int]

    lib.poly_grad_many.restype = ctypes.c_int
    lib.poly_grad_many.argtypes = [_ptr, _ptr, _ptr, ctypes.POINTER(_ptr), ctypes.c_int, ctypes.POINTER(_ptr)]

    # --- Realize ---
    lib.poly_realize.restype = ctypes.c_int
    lib.poly_realize.argtypes = [_ptr, _ptr, ctypes.POINTER(PolyBufferBinding), ctypes.c_int]

    lib.poly_realize_ex.restype = ctypes.c_int
    lib.poly_realize_ex.argtypes = [_ptr, _ptr,
        ctypes.POINTER(PolyBufferBinding), ctypes.c_int,
        ctypes.POINTER(PolyVarBinding), ctypes.c_int]

    # --- Einsum ---
    lib.poly_einsum.restype = _ptr
    lib.poly_einsum.argtypes = [
        _ptr, ctypes.c_char_p,
        ctypes.POINTER(_ptr),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_int64)),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int64),
        _ip,
    ]

    # --- Rearrange ---
    lib.poly_rearrange.restype = _ptr
    lib.poly_rearrange.argtypes = [
        _ptr, ctypes.c_char_p, _ptr,
        ctypes.POINTER(ctypes.c_int64), ctypes.c_int,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_int64), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int64), _ip,
    ]

    # --- Compiled Step ---
    lib.poly_compile_step.restype = _ptr
    lib.poly_compile_step.argtypes = [_ptr, _ptr]

    lib.poly_compile_value_and_grad.restype = _ptr
    lib.poly_compile_value_and_grad.argtypes = [
        _ptr, _ptr, ctypes.POINTER(_ptr), ctypes.c_int, _ip, _ip,
    ]

    lib.poly_step_run.restype = ctypes.c_int
    lib.poly_step_run.argtypes = [_ptr, ctypes.POINTER(PolyBufferBinding), ctypes.c_int]

    lib.poly_step_run_ex.restype = ctypes.c_int
    lib.poly_step_run_ex.argtypes = [_ptr,
        ctypes.POINTER(PolyBufferBinding), ctypes.c_int,
        ctypes.POINTER(PolyVarBinding), ctypes.c_int]

    lib.poly_step_run_indexed.restype = ctypes.c_int
    lib.poly_step_run_indexed.argtypes = [_ptr, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]

    lib.poly_step_run_indexed_ex.restype = ctypes.c_int
    lib.poly_step_run_indexed_ex.argtypes = [_ptr, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
        ctypes.POINTER(PolyVarBinding), ctypes.c_int]

    lib.poly_step_destroy.restype = None
    lib.poly_step_destroy.argtypes = [_ptr]

    lib.poly_step_n_kernels.restype = ctypes.c_int
    lib.poly_step_n_kernels.argtypes = [_ptr]

    lib.poly_step_n_intermediates.restype = ctypes.c_int
    lib.poly_step_n_intermediates.argtypes = [_ptr]

    lib.poly_step_n_buffers.restype = ctypes.c_int
    lib.poly_step_n_buffers.argtypes = [_ptr]

    lib.poly_step_n_bindable_buffers.restype = ctypes.c_int
    lib.poly_step_n_bindable_buffers.argtypes = [_ptr]

    lib.poly_step_buffer_info.restype = ctypes.c_int
    lib.poly_step_buffer_info.argtypes = [_ptr, ctypes.c_int, ctypes.POINTER(PolyStepBufferInfo)]

    # --- WASM step plan ---
    lib.poly_render_step_wasm_plan.restype = _ptr
    lib.poly_render_step_wasm_plan.argtypes = [_ptr, _ptr]

    lib.poly_wasm_stepplan_n_kernels.restype = ctypes.c_int
    lib.poly_wasm_stepplan_n_kernels.argtypes = [_ptr]

    lib.poly_wasm_stepplan_kernel_bytes.restype = ctypes.POINTER(ctypes.c_uint8)
    lib.poly_wasm_stepplan_kernel_bytes.argtypes = [_ptr, ctypes.c_int, _ip]

    lib.poly_wasm_stepplan_kernel_n_params.restype = ctypes.c_int
    lib.poly_wasm_stepplan_kernel_n_params.argtypes = [_ptr, ctypes.c_int]

    lib.poly_wasm_stepplan_n_buffers.restype = ctypes.c_int
    lib.poly_wasm_stepplan_n_buffers.argtypes = [_ptr]

    lib.poly_wasm_stepplan_n_bindable_buffers.restype = ctypes.c_int
    lib.poly_wasm_stepplan_n_bindable_buffers.argtypes = [_ptr]

    lib.poly_wasm_stepplan_kernel_param_buf_index.restype = ctypes.c_int
    lib.poly_wasm_stepplan_kernel_param_buf_index.argtypes = [_ptr, ctypes.c_int, ctypes.c_int]

    lib.poly_wasm_stepplan_exec_order.restype = _ip
    lib.poly_wasm_stepplan_exec_order.argtypes = [_ptr, _ip]

    lib.poly_wasm_stepplan_destroy.restype = None
    lib.poly_wasm_stepplan_destroy.argtypes = [_ptr]

    # --- PolyInstance (instance.h) ---
    lib.poly_instance_from_ir.restype = _ptr
    lib.poly_instance_from_ir.argtypes = [_u8p, ctypes.c_int, _u8p, ctypes.c_int]

    lib.poly_instance_free.restype = None
    lib.poly_instance_free.argtypes = [_ptr]

    lib.poly_instance_param_count.restype = ctypes.c_int
    lib.poly_instance_param_count.argtypes = [_ptr]

    lib.poly_instance_param_name.restype = ctypes.c_char_p
    lib.poly_instance_param_name.argtypes = [_ptr, ctypes.c_int]

    lib.poly_instance_param_shape.restype = ctypes.c_int
    lib.poly_instance_param_shape.argtypes = [_ptr, ctypes.c_int, _i64p, ctypes.c_int]

    lib.poly_instance_param_data.restype = _fp
    lib.poly_instance_param_data.argtypes = [_ptr, ctypes.c_int, _i64p]

    lib.poly_instance_buf_count.restype = ctypes.c_int
    lib.poly_instance_buf_count.argtypes = [_ptr]

    lib.poly_instance_buf_name.restype = ctypes.c_char_p
    lib.poly_instance_buf_name.argtypes = [_ptr, ctypes.c_int]

    lib.poly_instance_buf_role.restype = ctypes.c_int
    lib.poly_instance_buf_role.argtypes = [_ptr, ctypes.c_int]

    lib.poly_instance_buf_shape.restype = ctypes.c_int
    lib.poly_instance_buf_shape.argtypes = [_ptr, ctypes.c_int, _i64p, ctypes.c_int]

    lib.poly_instance_buf_data.restype = _fp
    lib.poly_instance_buf_data.argtypes = [_ptr, ctypes.c_int, _i64p]

    lib.poly_instance_export_weights.restype = _u8p
    lib.poly_instance_export_weights.argtypes = [_ptr, _ip]

    lib.poly_instance_import_weights.restype = ctypes.c_int
    lib.poly_instance_import_weights.argtypes = [_ptr, _u8p, ctypes.c_int]

    lib.poly_instance_export_ir.restype = _u8p
    lib.poly_instance_export_ir.argtypes = [_ptr, _ip]

    lib.poly_instance_forward.restype = ctypes.c_int
    lib.poly_instance_forward.argtypes = [_ptr, ctypes.POINTER(PolyIOBinding), ctypes.c_int]

    lib.poly_instance_train_step.restype = ctypes.c_int
    lib.poly_instance_train_step.argtypes = [_ptr, ctypes.POINTER(PolyIOBinding), ctypes.c_int, _fp]

    lib.poly_instance_set_optimizer.restype = ctypes.c_int
    lib.poly_instance_set_optimizer.argtypes = [_ptr, ctypes.c_int,
        ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_float, ctypes.c_float]

    # MLP family builder (model_mlp.h)
    lib.poly_mlp_instance.restype = _ptr
    lib.poly_mlp_instance.argtypes = [ctypes.c_char_p, ctypes.c_int]

    # TabM family builder (model_tabm.h)
    lib.poly_tabm_instance.restype = _ptr
    lib.poly_tabm_instance.argtypes = [ctypes.c_char_p, ctypes.c_int]

    # NAM family builder (model_nam.h)
    lib.poly_nam_instance.restype = _ptr
    lib.poly_nam_instance.argtypes = [ctypes.c_char_p, ctypes.c_int]

    # HF loader (modelzoo/hf_loader.c)
    lib.poly_hf_load.restype = _ptr
    lib.poly_hf_load.argtypes = [
        ctypes.c_char_p, ctypes.c_int,
        ctypes.POINTER(_u8p),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]

    # --- CUDA realize (conditional) ---
    has_cuda = hasattr(lib, 'poly_realize_cuda')
    if has_cuda:
        lib.poly_realize_cuda.restype = ctypes.c_int
        lib.poly_realize_cuda.argtypes = [_ptr, _ptr, ctypes.POINTER(PolyBufferBinding), ctypes.c_int]

        lib.poly_cuda_copyback.restype = ctypes.c_int
        lib.poly_cuda_copyback.argtypes = [ctypes.POINTER(PolyBufferBinding), ctypes.c_int]

        lib.poly_cuda_flush_buffers.restype = None
        lib.poly_cuda_flush_buffers.argtypes = []

        lib.poly_cuda_prog_cache_flush.restype = None
        lib.poly_cuda_prog_cache_flush.argtypes = []

        lib.poly_cuda_available.restype = ctypes.c_bool
        lib.poly_cuda_available.argtypes = []

    return has_cuda


# --- Public API ---

def get_lib():
    """Load the library and initialize all module-level state.

    Returns the ctypes CDLL handle. Also populates module globals:
    _lib, OPS, _has_cuda_ffi.
    """
    global _lib, OPS, _has_cuda_ffi

    if _lib is not None:
        return _lib

    lib_path = _find_lib()
    if not lib_path:
        raise RuntimeError(
            'Could not find libpolygrad shared library.\n'
            'Install: pip install polygrad\n'
            'Or build from source: make\n'
            'Or set POLYGRAD_LIB=/path/to/libpolygrad.so'
        )

    lib = ctypes.CDLL(lib_path)
    has_cuda = _declare_signatures(lib)

    # Build op name -> int mapping
    op_count = lib.poly_op_count()
    ops = {}
    for i in range(op_count):
        name = lib.poly_op_name(i)
        if name:
            ops[name.decode()] = i

    # Atomically populate module globals
    _lib = lib
    OPS = ops
    _has_cuda_ffi = has_cuda

    return _lib
