"""
ctypes bindings to libpolygrad.so. The bound surface is the public C ABI:
frontend helpers, tensor/UOp helpers, schedulers, and instance/model entrypoints.

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
POLYGRAD_ABI_VERSION = 57

# --- Opaque pointer type (always available) ---
_ptr = ctypes.c_void_p
_ptrp = ctypes.POINTER(_ptr)
_i64p = ctypes.POINTER(ctypes.c_int64)
_ip = ctypes.POINTER(ctypes.c_int)
_uintptr = ctypes.c_size_t


# --- Structures (always available, no _lib dependency) ---

class PolyVarBinding(ctypes.Structure):
    _fields_ = [('var', _ptr), ('value', ctypes.c_int32)]

class PolyBuffer(ctypes.Structure):
    _fields_ = [
        ('ptr', _ptr),
        ('nbytes', ctypes.c_size_t),
        ('device', ctypes.c_int),    # PolyDevice
        ('owned', ctypes.c_bool),
        ('allocator', _ptr),
        ('src', _ptr),
        ('valid', ctypes.c_bool),
        ('frontend_release', _ptr),
        ('memory_accounted', ctypes.c_bool),
        ('memory_device', ctypes.c_int),
        ('device_uop', _ptr),
        ('memory_device_uop', _ptr),
    ]

class PolyIOBinding(ctypes.Structure):
    _fields_ = [('name', ctypes.c_char_p),
                ('data', ctypes.c_void_p),
                ('nbytes', ctypes.c_size_t),
                ('dtype_id', ctypes.c_int)]

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

class PolyOptimConfig(ctypes.Structure):
    _fields_ = [
        ('kind', ctypes.c_int),
        ('beta1', ctypes.c_double),
        ('beta2', ctypes.c_double),
        ('eps', ctypes.c_double),
        ('weight_decay', ctypes.c_double),
        ('momentum', ctypes.c_double),
        ('nesterov', ctypes.c_bool),
        ('classic', ctypes.c_bool),
    ]

class PolyCtxStats(ctypes.Structure):
    _fields_ = [
        ('arena_bytes', ctypes.c_size_t),
        ('arena_high_water', ctypes.c_size_t),
        ('scratch_bytes', ctypes.c_size_t),
        ('scratch_high_water', ctypes.c_size_t),
        ('cse_entries', ctypes.c_size_t),
        ('schedule_cache_entries', ctypes.c_size_t),
        ('to_program_cache_entries', ctypes.c_size_t),
        ('runtime_cache_entries', ctypes.c_size_t),
        ('program_cache_entries', ctypes.c_size_t),
        ('shape_cache_entries', ctypes.c_size_t),
        ('buffer_entries', ctypes.c_size_t),
        ('buffer_owned_bytes', ctypes.c_size_t),
        ('buffer_owned_current_bytes', ctypes.c_size_t),
        ('buffer_owned_source_bytes', ctypes.c_size_t),
        ('tensor_records', ctypes.c_size_t),
        ('registry_entries', ctypes.c_size_t),
        ('entrypoint_entries', ctypes.c_size_t),
        ('compiled_artifact_bytes', ctypes.c_size_t),
        ('runtime_artifact_entries', ctypes.c_size_t),
        ('launch_count', ctypes.c_size_t),
        ('schedule_cache_hits', ctypes.c_size_t),
        ('schedule_cache_misses', ctypes.c_size_t),
        ('runtime_cache_hits', ctypes.c_size_t),
        ('runtime_cache_misses', ctypes.c_size_t),
        ('buffer_read_count', ctypes.c_size_t),
        ('buffer_read_bytes', ctypes.c_size_t),
        ('buffer_write_count', ctypes.c_size_t),
        ('buffer_write_bytes', ctypes.c_size_t),
        ('buffer_copy_count', ctypes.c_size_t),
        ('buffer_copy_bytes', ctypes.c_size_t),
        ('global_ops', ctypes.c_uint64),
        ('global_mem', ctypes.c_uint64),
        ('time_sum_s', ctypes.c_double),
        ('kernel_count', ctypes.c_uint64),
        ('mem_used', ctypes.c_uint64),
    ]

class PolyInstanceOptions(ctypes.Structure):
    _fields_ = [
        ('own_ctx_on_success', ctypes.c_bool),
        ('own_ctx_on_failure', ctypes.c_bool),
    ]

class PolyInstanceError(ctypes.Structure):
    _fields_ = [
        ('code', ctypes.c_int),
        ('func', ctypes.c_char_p),
        ('message', ctypes.c_char * 256),
    ]

class PolyBindingSpec(ctypes.Structure):
    _fields_ = [
        ('name', ctypes.c_char_p),
        ('role', ctypes.c_int),
        ('tensor', _ptr),
        ('flags', ctypes.c_uint32),
    ]

class PolyEntrypointSpec(ctypes.Structure):
    _fields_ = [
        ('name', ctypes.c_char_p),
        ('inputs', ctypes.POINTER(ctypes.c_char_p)),
        ('n_inputs', ctypes.c_int),
        ('outputs', ctypes.POINTER(ctypes.c_char_p)),
        ('n_outputs', ctypes.c_int),
        ('objective', ctypes.c_char_p),
        ('flags', ctypes.c_uint32),
    ]

PolyFrontendBufferReleaseFn = ctypes.CFUNCTYPE(None, _uintptr)


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

    lib.poly_ctx_set_preferred_device.restype = None
    lib.poly_ctx_set_preferred_device.argtypes = [_ptr, ctypes.c_int]

    try:
        lib.poly_ctx_set_frontend_buffer_release.restype = None
        lib.poly_ctx_set_frontend_buffer_release.argtypes = [_ptr, PolyFrontendBufferReleaseFn]
    except AttributeError:
        pass

    lib.poly_device_by_name.restype = ctypes.c_int
    lib.poly_device_by_name.argtypes = [ctypes.c_char_p]

    lib.poly_device_name.restype = ctypes.c_char_p
    lib.poly_device_name.argtypes = [ctypes.c_int]

    lib.poly_uop_device_name.restype = ctypes.c_char_p
    lib.poly_uop_device_name.argtypes = [_ptr, _ptr]

    lib.poly_uop_resolve.restype = ctypes.c_int
    lib.poly_uop_resolve.argtypes = [_ptr, _ptr, ctypes.c_int]

    lib.poly_device_can_execute.restype = ctypes.c_bool
    lib.poly_device_can_execute.argtypes = [ctypes.c_int]

    lib.poly_device_is_host_addressable.restype = ctypes.c_bool
    lib.poly_device_is_host_addressable.argtypes = [ctypes.c_int]

    lib.poly_ctx_named_count.restype = ctypes.c_int
    lib.poly_ctx_named_count.argtypes = [_ptr]

    lib.poly_ctx_stats.restype = ctypes.c_int
    lib.poly_ctx_stats.argtypes = [_ptr, ctypes.POINTER(PolyCtxStats)]

    lib.poly_ctx_reset_counters.restype = None
    lib.poly_ctx_reset_counters.argtypes = [_ptr]

    lib.poly_ctx_mem_used_for_device.restype = ctypes.c_uint64
    lib.poly_ctx_mem_used_for_device.argtypes = [_ptr, ctypes.c_int]

    lib.poly_can_run_op.restype = ctypes.c_int
    lib.poly_can_run_op.argtypes = [_ptr, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int64), ctypes.c_int]

    # --- Op helpers ---
    lib.poly_op_count.restype = ctypes.c_int
    lib.poly_op_count.argtypes = []

    lib.poly_op_name.restype = ctypes.c_char_p
    lib.poly_op_name.argtypes = [ctypes.c_int]

    lib.poly_dtype_count.restype = ctypes.c_int
    lib.poly_dtype_count.argtypes = []

    lib.poly_dtype_id_by_name.restype = ctypes.c_int
    lib.poly_dtype_id_by_name.argtypes = [ctypes.c_char_p]

    # --- Frontend helpers (frontend.h) ---
    lib.poly_const_float.restype = _ptr
    lib.poly_const_float.argtypes = [_ptr, ctypes.c_double]

    lib.poly_const_double.restype = _ptr
    lib.poly_const_double.argtypes = [_ptr, ctypes.c_double]

    lib.poly_const_int.restype = _ptr
    lib.poly_const_int.argtypes = [_ptr, ctypes.c_int64]

    lib.poly_contiguous.restype = _ptr
    lib.poly_contiguous.argtypes = [_ptr, _ptr]

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

    lib.poly_uop_placeholder_like.restype = _ptr
    lib.poly_uop_placeholder_like.argtypes = [_ptr, _ptr, ctypes.c_int]

    lib.poly_uop_range.restype = _ptr
    lib.poly_uop_range.argtypes = [_ptr, ctypes.c_int64, ctypes.c_int64, ctypes.c_int]

    lib.poly_uop_index.restype = _ptr
    lib.poly_uop_index.argtypes = [_ptr, _ptr, ctypes.POINTER(_ptr), ctypes.c_int, ctypes.c_int]

    lib.poly_uop_load.restype = _ptr
    lib.poly_uop_load.argtypes = [_ptr, _ptr]

    lib.poly_uop_store.restype = _ptr
    lib.poly_uop_store.argtypes = [_ptr, _ptr, _ptr]

    lib.poly_uop_set.restype = _ptr
    lib.poly_uop_set.argtypes = [_ptr, _ptr, _ptr, ctypes.POINTER(_ptr), ctypes.c_int]

    lib.poly_uop_group.restype = _ptr
    lib.poly_uop_group.argtypes = [_ptr, ctypes.POINTER(_ptr), ctypes.c_int]

    lib.poly_uop_end.restype = _ptr
    lib.poly_uop_end.argtypes = [_ptr, _ptr, ctypes.POINTER(_ptr), ctypes.c_int]

    lib.poly_uop_sink.restype = _ptr
    lib.poly_uop_sink.argtypes = [_ptr, ctypes.POINTER(_ptr), ctypes.c_int]

    lib.poly_uop_sink_ex.restype = _ptr
    lib.poly_uop_sink_ex.argtypes = [_ptr, ctypes.POINTER(_ptr), ctypes.c_int, ctypes.c_char_p, ctypes.c_int]

    lib.poly_uop_call.restype = _ptr
    lib.poly_uop_call.argtypes = [_ptr, _ptr, ctypes.POINTER(_ptr), ctypes.c_int]

    lib.poly_uop_after.restype = _ptr
    lib.poly_uop_after.argtypes = [_ptr, _ptr, _ptr]

    lib.poly_uop_reduce.restype = _ptr
    lib.poly_uop_reduce.argtypes = [_ptr, ctypes.c_int, _ptr, ctypes.POINTER(_ptr), ctypes.c_int]

    lib.poly_uop_flatten.restype = _ptr
    lib.poly_uop_flatten.argtypes = [_ptr, _ptr]

    lib.poly_uop_numel.restype = ctypes.c_int64
    lib.poly_uop_numel.argtypes = [_ptr, _ptr]

    lib.poly_register_buffer_by_id.restype = _ptr
    lib.poly_register_buffer_by_id.argtypes = [
        _ptr, ctypes.c_int, ctypes.c_int, _i64p, ctypes.c_int, ctypes.c_char_p
    ]

    lib.poly_register_existing_buffer.restype = _ptr
    lib.poly_register_existing_buffer.argtypes = [
        _ptr, ctypes.c_int, _ptr, _i64p, ctypes.c_int, ctypes.c_char_p, ctypes.c_bool
    ]

    lib.poly_buffer_by_id.restype = _ptr
    lib.poly_buffer_by_id.argtypes = [_ptr, ctypes.c_int, ctypes.c_int64]

    lib.poly_buffer_on_device_by_id.restype = _ptr
    lib.poly_buffer_on_device_by_id.argtypes = [
        _ptr, ctypes.c_int, ctypes.c_int64, ctypes.c_int,
    ]

    lib.poly_buffer_var_by_id.restype = _ptr
    lib.poly_buffer_var_by_id.argtypes = [
        _ptr, ctypes.c_int, _ptr, _i64p, ctypes.c_int, ctypes.c_int
    ]

    lib.poly_buffer_f32.restype = _ptr
    lib.poly_buffer_f32.argtypes = [_ptr, ctypes.c_int64]

    lib.poly_buffer_f64.restype = _ptr
    lib.poly_buffer_f64.argtypes = [_ptr, ctypes.c_int64]

    lib.poly_tensor_empty_by_id.restype = _ptr
    lib.poly_tensor_empty_by_id.argtypes = [
        _ptr, ctypes.c_int, _i64p, ctypes.c_int, ctypes.c_int,
    ]

    lib.poly_tensor_from_host_by_id.restype = _ptr
    lib.poly_tensor_from_host_by_id.argtypes = [
        _ptr, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, _i64p, ctypes.c_int,
    ]

    lib.poly_tensor_const_int_by_id.restype = _ptr
    lib.poly_tensor_const_int_by_id.argtypes = [
        _ptr, ctypes.c_int64, ctypes.c_int, ctypes.c_int,
    ]

    lib.poly_tensor_const_float_by_id.restype = _ptr
    lib.poly_tensor_const_float_by_id.argtypes = [
        _ptr, ctypes.c_double, ctypes.c_int, ctypes.c_int,
    ]

    lib.poly_tensor_full_int_by_id.restype = _ptr
    lib.poly_tensor_full_int_by_id.argtypes = [
        _ptr, _i64p, ctypes.c_int, ctypes.c_int64, ctypes.c_int, ctypes.c_int,
    ]

    lib.poly_tensor_full_float_by_id.restype = _ptr
    lib.poly_tensor_full_float_by_id.argtypes = [
        _ptr, _i64p, ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.c_int,
    ]

    lib.poly_tensor_arange_int_by_id.restype = _ptr
    lib.poly_tensor_arange_int_by_id.argtypes = [
        _ptr, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int, ctypes.c_int,
    ]

    lib.poly_tensor_arange_float_by_id.restype = _ptr
    lib.poly_tensor_arange_float_by_id.argtypes = [
        _ptr, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int,
    ]

    lib.poly_tensor_linspace_by_id.restype = _ptr
    lib.poly_tensor_linspace_by_id.argtypes = [
        _ptr, ctypes.c_double, ctypes.c_double, ctypes.c_int64, ctypes.c_int, ctypes.c_int,
    ]

    lib.poly_tensor_eye_by_id.restype = _ptr
    lib.poly_tensor_eye_by_id.argtypes = [
        _ptr, ctypes.c_int64, ctypes.c_int64, ctypes.c_int, ctypes.c_int,
    ]

    lib.poly_tensor_manual_seed.restype = None
    lib.poly_tensor_manual_seed.argtypes = [_ptr, ctypes.c_int64]

    lib.poly_tensor_rand_by_id.restype = _ptr
    lib.poly_tensor_rand_by_id.argtypes = [
        _ptr, _i64p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]

    lib.poly_tensor_randn_by_id.restype = _ptr
    lib.poly_tensor_randn_by_id.argtypes = [
        _ptr, _i64p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]

    lib.poly_uop_op.restype = ctypes.c_int
    lib.poly_uop_op.argtypes = [_ptr]

    lib.poly_uop_device.restype = ctypes.c_int
    lib.poly_uop_device.argtypes = [_ptr]

    lib.poly_uop_dtype_id.restype = ctypes.c_int
    lib.poly_uop_dtype_id.argtypes = [_ptr, _ptr]

    lib.poly_cast_by_id.restype = _ptr
    lib.poly_cast_by_id.argtypes = [_ptr, _ptr, ctypes.c_int]

    lib.poly_bitcast_by_id.restype = _ptr
    lib.poly_bitcast_by_id.argtypes = [_ptr, _ptr, ctypes.c_int]

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

    lib.poly_rand_by_id.restype = _ptr
    lib.poly_rand_by_id.argtypes = [_ptr, _i64p, ctypes.c_int, ctypes.c_uint64, ctypes.c_int]

    lib.poly_randn_by_id.restype = _ptr
    lib.poly_randn_by_id.argtypes = [_ptr, _i64p, ctypes.c_int, ctypes.c_uint64, ctypes.c_int]

    lib.poly_arange.restype = _ptr
    lib.poly_arange.argtypes = [_ptr, ctypes.c_double, ctypes.c_double, ctypes.c_double]

    lib.poly_arange_int_by_id.restype = _ptr
    lib.poly_arange_int_by_id.argtypes = [_ptr, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int]

    lib.poly_arange_float_by_id.restype = _ptr
    lib.poly_arange_float_by_id.argtypes = [_ptr, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int]

    lib.poly_eye.restype = _ptr
    lib.poly_eye.argtypes = [_ptr, ctypes.c_int64]

    lib.poly_eye_by_id.restype = _ptr
    lib.poly_eye_by_id.argtypes = [_ptr, ctypes.c_int64, ctypes.c_int64, ctypes.c_int]

    lib.poly_linspace.restype = _ptr
    lib.poly_linspace.argtypes = [_ptr, ctypes.c_double, ctypes.c_double, ctypes.c_int64]

    lib.poly_linspace_by_id.restype = _ptr
    lib.poly_linspace_by_id.argtypes = [_ptr, ctypes.c_double, ctypes.c_double, ctypes.c_int64, ctypes.c_int]

    lib.poly_full.restype = _ptr
    lib.poly_full.argtypes = [_ptr, _i64p, ctypes.c_int, ctypes.c_double]

    lib.poly_const_int_by_id.restype = _ptr
    lib.poly_const_int_by_id.argtypes = [_ptr, ctypes.c_int64, ctypes.c_int]

    lib.poly_const_float_by_id.restype = _ptr
    lib.poly_const_float_by_id.argtypes = [_ptr, ctypes.c_double, ctypes.c_int]

    lib.poly_full_int_by_id.restype = _ptr
    lib.poly_full_int_by_id.argtypes = [_ptr, _i64p, ctypes.c_int, ctypes.c_int64, ctypes.c_int]

    lib.poly_full_float_by_id.restype = _ptr
    lib.poly_full_float_by_id.argtypes = [_ptr, _i64p, ctypes.c_int, ctypes.c_double, ctypes.c_int]

    lib.poly_tril.restype = _ptr
    lib.poly_tril.argtypes = [_ptr, _ptr, ctypes.c_int]

    lib.poly_triu.restype = _ptr
    lib.poly_triu.argtypes = [_ptr, _ptr, ctypes.c_int]

    lib.poly_cholesky.restype = _ptr
    lib.poly_cholesky.argtypes = [_ptr, _ptr, ctypes.c_int]

    lib.poly_cholesky_solve.restype = _ptr
    lib.poly_cholesky_solve.argtypes = [_ptr, _ptr, _ptr, ctypes.c_int]

    lib.poly_triangular_solve.restype = _ptr
    lib.poly_triangular_solve.argtypes = [
        _ptr, _ptr, _ptr,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]

    # --- Shape-aware composed ops (shape read from UOp) ---
    lib.poly_sum_reduce.restype = _ptr
    lib.poly_sum_reduce.argtypes = [_ptr, _ptr, ctypes.c_int, ctypes.c_int]

    lib.poly_max_reduce.restype = _ptr
    lib.poly_max_reduce.argtypes = [_ptr, _ptr, ctypes.c_int, ctypes.c_int]

    lib.poly_mean_reduce.restype = _ptr
    lib.poly_mean_reduce.argtypes = [_ptr, _ptr, ctypes.c_int, ctypes.c_int]

    lib.poly_var_reduce.restype = _ptr
    lib.poly_var_reduce.argtypes = [_ptr, _ptr, ctypes.c_int, ctypes.c_int, ctypes.c_int]

    lib.poly_logsumexp.restype = _ptr
    lib.poly_logsumexp.argtypes = [_ptr, _ptr, ctypes.c_int, ctypes.c_int]

    lib.poly_dot.restype = _ptr
    lib.poly_dot.argtypes = [_ptr, _ptr, _ptr]

    lib.poly_qr.restype = ctypes.c_int
    lib.poly_qr.argtypes = [_ptr, _ptr, _ptrp, _ptrp]

    lib.poly_qr_ex.restype = ctypes.c_int
    lib.poly_qr_ex.argtypes = [_ptr, _ptr, ctypes.c_int, _ptrp, _ptrp]

    lib.poly_solve.restype = _ptr
    lib.poly_solve.argtypes = [_ptr, _ptr, _ptr]

    lib.poly_lstsq.restype = _ptr
    lib.poly_lstsq.argtypes = [_ptr, _ptr, _ptr]

    lib.poly_cross_entropy.restype = _ptr
    lib.poly_cross_entropy.argtypes = [_ptr, _ptr, _ptr, ctypes.c_int]

    lib.poly_softmax.restype = _ptr
    lib.poly_softmax.argtypes = [_ptr, _ptr, ctypes.c_int]

    lib.poly_log_softmax.restype = _ptr
    lib.poly_log_softmax.argtypes = [_ptr, _ptr, ctypes.c_int]

    lib.poly_gather.restype = _ptr
    lib.poly_gather.argtypes = [_ptr, _ptr, _ptr]

    lib.poly_gather_dim.restype = _ptr
    lib.poly_gather_dim.argtypes = [_ptr, _ptr, ctypes.c_int, _ptr]

    lib.poly_scatter.restype = _ptr
    lib.poly_scatter.argtypes = [_ptr, _ptr, ctypes.c_int, _ptr, _ptr, ctypes.c_char_p]

    lib.poly_scatter_reduce.restype = _ptr
    lib.poly_scatter_reduce.argtypes = [
        _ptr, _ptr, ctypes.c_int, _ptr, _ptr, ctypes.c_char_p, ctypes.c_int
    ]

    # --- Dynamic shapes (DEFINE_VAR / BIND) ---
    lib.poly_define_var.restype = _ptr
    lib.poly_define_var.argtypes = [_ptr, ctypes.c_char_p, ctypes.c_int64, ctypes.c_int64]

    lib.poly_bind_var.restype = _ptr
    lib.poly_bind_var.argtypes = [_ptr, _ptr, ctypes.c_int64]

    # --- Sched helpers (sched.h) ---
    lib.poly_reshape.restype = _ptr
    lib.poly_reshape.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int]

    lib.poly_expand.restype = _ptr
    lib.poly_expand.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int]
    lib.poly_expand_uop.restype = _ptr
    lib.poly_expand_uop.argtypes = [_ptr, _ptr, ctypes.POINTER(_ptr), ctypes.c_int]

    lib.poly_reduce_axis.restype = _ptr
    lib.poly_reduce_axis.argtypes = [_ptr, ctypes.c_int, _ptr, _i64p, ctypes.c_int]

    lib.poly_permute.restype = _ptr
    lib.poly_permute.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int]

    lib.poly_shrink.restype = _ptr
    lib.poly_shrink.argtypes = [_ptr, _ptr, ctypes.c_void_p, ctypes.c_int]
    lib.poly_shrink_uop.restype = _ptr
    lib.poly_shrink_uop.argtypes = [_ptr, _ptr, ctypes.POINTER(_ptr), ctypes.POINTER(_ptr), ctypes.c_int]

    lib.poly_flip.restype = _ptr
    lib.poly_flip.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int]

    lib.poly_pad.restype = _ptr
    lib.poly_pad.argtypes = [_ptr, _ptr, ctypes.c_void_p, ctypes.c_int]

    lib.poly_pad_value.restype = _ptr
    lib.poly_pad_value.argtypes = [_ptr, _ptr, ctypes.c_void_p, ctypes.c_int, ctypes.c_double]

    lib.poly_pool.restype = _ptr
    lib.poly_pool.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int, _i64p, _i64p]

    lib.poly_max_pool2d.restype = _ptr
    lib.poly_max_pool2d.argtypes = [
        _ptr, _ptr, _i64p, ctypes.c_int, _i64p, _i64p, _i64p, ctypes.c_int,
    ]

    lib.poly_conv2d.restype = _ptr
    lib.poly_conv2d.argtypes = [
        _ptr, _ptr, _ptr, _ptr, ctypes.c_int, _i64p, _i64p, _i64p, ctypes.c_int,
    ]

    lib.poly_batchnorm.restype = _ptr
    lib.poly_batchnorm.argtypes = [
        _ptr, _ptr, _ptr, _ptr, _ptr, _ptr, _i64p, ctypes.c_int,
    ]

    lib.poly_one_hot.restype = _ptr
    lib.poly_one_hot.argtypes = [_ptr, _ptr, ctypes.c_int64]

    lib.poly_index_select.restype = _ptr
    lib.poly_index_select.argtypes = [_ptr, _ptr, ctypes.c_int, _ptr]

    # --- Autograd ---
    lib.poly_grad.restype = _ptr
    lib.poly_grad.argtypes = [_ptr, _ptr, _ptr]

    lib.poly_uop_substitute.restype = _ptr
    lib.poly_uop_substitute.argtypes = [_ptr, _ptr, ctypes.POINTER(_ptr), ctypes.POINTER(_ptr), ctypes.c_int]

    lib.poly_grad_many.restype = ctypes.c_int
    lib.poly_grad_many.argtypes = [_ptr, _ptr, _ptr, ctypes.POINTER(_ptr), ctypes.c_int, ctypes.POINTER(_ptr)]

    lib.poly_grad_many_ex.restype = ctypes.c_int
    lib.poly_grad_many_ex.argtypes = [
        _ptr, _ptr, _ptr, ctypes.POINTER(_ptr), ctypes.c_int,
        ctypes.POINTER(_ptr), ctypes.POINTER(ctypes.c_uint8),
    ]

    # --- UOp identity helpers ---
    lib.poly_uop_has_buffer_identity.restype = ctypes.c_bool
    lib.poly_uop_has_buffer_identity.argtypes = [_ptr]

    lib.poly_uop_get_buffer_identity.restype = _ptr
    lib.poly_uop_get_buffer_identity.argtypes = [_ptr]

    lib.poly_uop_buffer.restype = _ptr
    lib.poly_uop_buffer.argtypes = [_ptr, _ptr]

    lib.poly_uop_reachable.restype = ctypes.c_bool
    lib.poly_uop_reachable.argtypes = [_ptr, _ptr, _ptr]

    lib.poly_uop_n_src.restype = ctypes.c_int
    lib.poly_uop_n_src.argtypes = [_ptr]

    lib.poly_uop_src.restype = _ptr
    lib.poly_uop_src.argtypes = [_ptr, ctypes.c_int]

    # --- Side-table buffer API (device.h) ---
    lib.poly_buffer_set.restype = None
    lib.poly_buffer_set.argtypes = [_ptr, _ptr, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]

    lib.poly_buffer_get_ptr.restype = ctypes.c_void_p
    lib.poly_buffer_get_ptr.argtypes = [_ptr, _ptr]

    lib.poly_buffer_get_key.restype = ctypes.c_uint64
    lib.poly_buffer_get_key.argtypes = [_ptr, _ptr]

    lib.poly_buffer_ensure_device_allocated.restype = ctypes.c_int
    lib.poly_buffer_ensure_device_allocated.argtypes = [_ptr, _ptr, ctypes.c_int]

    lib.poly_buffer_read.restype = ctypes.c_int
    lib.poly_buffer_read.argtypes = [_ptr, _ptr, ctypes.c_void_p, ctypes.c_size_t]

    lib.poly_buffer_write.restype = ctypes.c_int
    lib.poly_buffer_write.argtypes = [_ptr, _ptr, ctypes.c_void_p, ctypes.c_size_t]

    lib.poly_buffer_from_host.restype = _ptr
    lib.poly_buffer_from_host.argtypes = [
        _ptr, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, _i64p, ctypes.c_int,
    ]

    lib.poly_buffer_from_file.restype = _ptr
    lib.poly_buffer_from_file.argtypes = [_ptr, ctypes.c_char_p, ctypes.c_int]

    lib.poly_set_frontend_buffer_release.restype = None
    lib.poly_set_frontend_buffer_release.argtypes = [PolyFrontendBufferReleaseFn]

    lib.poly_buffer_get.restype = _ptr
    lib.poly_buffer_get.argtypes = [_ptr, _ptr]

    lib.poly_buffer_is_allocated.restype = ctypes.c_bool
    lib.poly_buffer_is_allocated.argtypes = [_ptr, _ptr]

    # --- Graph-driven realization and core frontend tensors ---
    lib.poly_realize_uops.restype = ctypes.c_int
    lib.poly_realize_uops.argtypes = [_ptr, ctypes.POINTER(_ptr), ctypes.c_int, ctypes.POINTER(_ptr)]

    lib.poly_tensor_create_with_roots.restype = _ptr
    lib.poly_tensor_create_with_roots.argtypes = [_ptr, _ptr, _ptr, ctypes.c_int, ctypes.c_int]

    lib.poly_tensor_replace_roots.restype = ctypes.c_int
    lib.poly_tensor_replace_roots.argtypes = [
        _ptr, _ptr, _ptr, _ptr, ctypes.c_int, ctypes.c_int,
    ]

    lib.poly_tensor_to_device.restype = _ptr
    lib.poly_tensor_to_device.argtypes = [_ptr, _ptr, ctypes.c_int]

    lib.poly_tensor_assign.restype = _ptr
    lib.poly_tensor_assign.argtypes = [_ptr, _ptr, _ptr]

    lib.poly_tensor_alu1.restype = _ptr
    lib.poly_tensor_alu1.argtypes = [_ptr, ctypes.c_int, _ptr]

    lib.poly_tensor_alu2.restype = _ptr
    lib.poly_tensor_alu2.argtypes = [_ptr, ctypes.c_int, _ptr, _ptr]

    lib.poly_tensor_alu3.restype = _ptr
    lib.poly_tensor_alu3.argtypes = [_ptr, ctypes.c_int, _ptr, _ptr, _ptr]

    lib.poly_tensor_div.restype = _ptr
    lib.poly_tensor_div.argtypes = [_ptr, _ptr, _ptr]

    lib.poly_tensor_exp.restype = _ptr
    lib.poly_tensor_exp.argtypes = [_ptr, _ptr]

    lib.poly_tensor_log.restype = _ptr
    lib.poly_tensor_log.argtypes = [_ptr, _ptr]

    lib.poly_tensor_log1p.restype = _ptr
    lib.poly_tensor_log1p.argtypes = [_ptr, _ptr]

    lib.poly_tensor_expm1.restype = _ptr
    lib.poly_tensor_expm1.argtypes = [_ptr, _ptr]

    lib.poly_tensor_gelu.restype = _ptr
    lib.poly_tensor_gelu.argtypes = [_ptr, _ptr]

    lib.poly_tensor_quick_gelu.restype = _ptr
    lib.poly_tensor_quick_gelu.argtypes = [_ptr, _ptr]

    lib.poly_tensor_detach.restype = _ptr
    lib.poly_tensor_detach.argtypes = [_ptr, _ptr]

    lib.poly_tensor_contiguous_backward.restype = _ptr
    lib.poly_tensor_contiguous_backward.argtypes = [_ptr, _ptr]

    lib.poly_tensor_custom_kernel.restype = ctypes.c_int
    lib.poly_tensor_custom_kernel.argtypes = [
        _ptr, _ptr, ctypes.POINTER(_ptr), ctypes.c_int, ctypes.POINTER(_ptr),
    ]

    lib.poly_tensor_function.restype = ctypes.c_int
    lib.poly_tensor_function.argtypes = [
        _ptr, ctypes.POINTER(_ptr), ctypes.c_int,
        ctypes.POINTER(_ptr), ctypes.c_int,
        ctypes.c_char_p, ctypes.c_bool, ctypes.c_bool, ctypes.c_bool,
        ctypes.POINTER(_ptr),
    ]

    lib.poly_tensor_sum.restype = _ptr
    lib.poly_tensor_sum.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int, ctypes.c_bool]

    lib.poly_tensor_sum_dtype_by_id.restype = _ptr
    lib.poly_tensor_sum_dtype_by_id.argtypes = [
        _ptr, _ptr, _i64p, ctypes.c_int, ctypes.c_bool, ctypes.c_int,
    ]

    lib.poly_tensor_max.restype = _ptr
    lib.poly_tensor_max.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int, ctypes.c_bool]

    lib.poly_tensor_argmax.restype = _ptr
    lib.poly_tensor_argmax.argtypes = [_ptr, _ptr, ctypes.c_int, ctypes.c_bool]

    lib.poly_tensor_minimum.restype = _ptr
    lib.poly_tensor_minimum.argtypes = [_ptr, _ptr, _ptr]

    lib.poly_tensor_dot.restype = _ptr
    lib.poly_tensor_dot.argtypes = [_ptr, _ptr, _ptr]

    lib.poly_tensor_dot_dtype_by_id.restype = _ptr
    lib.poly_tensor_dot_dtype_by_id.argtypes = [_ptr, _ptr, _ptr, ctypes.c_int]

    lib.poly_tensor_qr_ex.restype = ctypes.c_int
    lib.poly_tensor_qr_ex.argtypes = [_ptr, _ptr, ctypes.c_int, _ptrp, _ptrp]

    lib.poly_tensor_triangular_solve.restype = _ptr
    lib.poly_tensor_triangular_solve.argtypes = [
        _ptr, _ptr, _ptr,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]

    lib.poly_tensor_cholesky.restype = _ptr
    lib.poly_tensor_cholesky.argtypes = [_ptr, _ptr, ctypes.c_int]

    lib.poly_tensor_cholesky_solve.restype = _ptr
    lib.poly_tensor_cholesky_solve.argtypes = [_ptr, _ptr, _ptr, ctypes.c_int]

    lib.poly_tensor_solve.restype = _ptr
    lib.poly_tensor_solve.argtypes = [_ptr, _ptr, _ptr]

    lib.poly_tensor_lstsq.restype = _ptr
    lib.poly_tensor_lstsq.argtypes = [_ptr, _ptr, _ptr]

    lib.poly_tensor_scatter.restype = _ptr
    lib.poly_tensor_scatter.argtypes = [
        _ptr, _ptr, ctypes.c_int, _ptr, _ptr, ctypes.c_char_p,
    ]

    lib.poly_tensor_scatter_reduce.restype = _ptr
    lib.poly_tensor_scatter_reduce.argtypes = [
        _ptr, _ptr, ctypes.c_int, _ptr, _ptr, ctypes.c_char_p, ctypes.c_int,
    ]

    lib.poly_tensor_einsum.restype = _ptr
    lib.poly_tensor_einsum.argtypes = [
        _ptr, ctypes.c_char_p, ctypes.POINTER(_ptr), ctypes.c_int,
    ]

    lib.poly_tensor_sort.restype = ctypes.c_int
    lib.poly_tensor_sort.argtypes = [
        _ptr, _ptr, ctypes.c_int, ctypes.c_int, _ptrp, _ptrp
    ]

    lib.poly_tensor_topk.restype = ctypes.c_int
    lib.poly_tensor_topk.argtypes = [
        _ptr, _ptr, ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        _ptrp, _ptrp
    ]

    lib.poly_tensor_softmax.restype = _ptr
    lib.poly_tensor_softmax.argtypes = [_ptr, _ptr, ctypes.c_int]

    lib.poly_tensor_log_softmax.restype = _ptr
    lib.poly_tensor_log_softmax.argtypes = [_ptr, _ptr, ctypes.c_int]

    lib.poly_tensor_cast_by_id.restype = _ptr
    lib.poly_tensor_cast_by_id.argtypes = [_ptr, _ptr, ctypes.c_int]

    lib.poly_tensor_bitcast_by_id.restype = _ptr
    lib.poly_tensor_bitcast_by_id.argtypes = [_ptr, _ptr, ctypes.c_int]

    lib.poly_tensor_contiguous.restype = _ptr
    lib.poly_tensor_contiguous.argtypes = [_ptr, _ptr]

    lib.poly_tensor_reshape.restype = _ptr
    lib.poly_tensor_reshape.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int]

    lib.poly_tensor_reshape_uop.restype = _ptr
    lib.poly_tensor_reshape_uop.argtypes = [
        _ptr, _ptr, ctypes.POINTER(_ptr), ctypes.c_int,
    ]

    lib.poly_tensor_expand.restype = _ptr
    lib.poly_tensor_expand.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int]

    lib.poly_tensor_expand_uop.restype = _ptr
    lib.poly_tensor_expand_uop.argtypes = [
        _ptr, _ptr, ctypes.POINTER(_ptr), ctypes.c_int,
    ]

    lib.poly_tensor_permute.restype = _ptr
    lib.poly_tensor_permute.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int]

    lib.poly_tensor_shrink.restype = _ptr
    lib.poly_tensor_shrink.argtypes = [_ptr, _ptr, ctypes.c_void_p, ctypes.c_int]

    lib.poly_tensor_shrink_uop.restype = _ptr
    lib.poly_tensor_shrink_uop.argtypes = [
        _ptr, _ptr, ctypes.POINTER(_ptr), ctypes.POINTER(_ptr), ctypes.c_int,
    ]

    lib.poly_tensor_flip.restype = _ptr
    lib.poly_tensor_flip.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int]

    lib.poly_tensor_pad_value.restype = _ptr
    lib.poly_tensor_pad_value.argtypes = [
        _ptr, _ptr, ctypes.c_void_p, ctypes.c_int, ctypes.c_double,
    ]

    lib.poly_tensor_pool.restype = _ptr
    lib.poly_tensor_pool.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int, _i64p, _i64p]

    lib.poly_tensor_max_pool2d.restype = _ptr
    lib.poly_tensor_max_pool2d.argtypes = [
        _ptr, _ptr, _i64p, ctypes.c_int, _i64p, _i64p, _i64p, ctypes.c_int,
    ]

    lib.poly_tensor_conv2d.restype = _ptr
    lib.poly_tensor_conv2d.argtypes = [
        _ptr, _ptr, _ptr, _ptr, ctypes.c_int, _i64p, _i64p, _i64p, ctypes.c_int,
    ]
    lib.poly_tensor_conv2d_dtype_by_id.restype = _ptr
    lib.poly_tensor_conv2d_dtype_by_id.argtypes = [
        _ptr, _ptr, _ptr, _ptr, ctypes.c_int, _i64p, _i64p, _i64p, ctypes.c_int,
        ctypes.c_int,
    ]

    lib.poly_tensor_batchnorm.restype = _ptr
    lib.poly_tensor_batchnorm.argtypes = [
        _ptr, _ptr, _ptr, _ptr, _ptr, _ptr, _i64p, ctypes.c_int,
    ]

    lib.poly_tensor_one_hot.restype = _ptr
    lib.poly_tensor_one_hot.argtypes = [_ptr, _ptr, ctypes.c_int64]

    lib.poly_tensor_gather_dim.restype = _ptr
    lib.poly_tensor_gather_dim.argtypes = [_ptr, _ptr, ctypes.c_int, _ptr]

    lib.poly_tensor_index_select.restype = _ptr
    lib.poly_tensor_index_select.argtypes = [_ptr, _ptr, ctypes.c_int, _ptr]

    lib.poly_tensor_clone_into.restype = _ptr
    lib.poly_tensor_clone_into.argtypes = [_ptr, _ptr, _ptr]

    lib.poly_tensor_uop.restype = _ptr
    lib.poly_tensor_uop.argtypes = [_ptr]

    lib.poly_tensor_uop_logical.restype = _ptr
    lib.poly_tensor_uop_logical.argtypes = [_ptr]

    lib.poly_tensor_uop_physical.restype = _ptr
    lib.poly_tensor_uop_physical.argtypes = [_ptr]

    lib.poly_tensor_device.restype = ctypes.c_int
    lib.poly_tensor_device.argtypes = [_ptr]

    lib.poly_tensor_requires_grad.restype = ctypes.c_bool
    lib.poly_tensor_requires_grad.argtypes = [_ptr]

    lib.poly_tensor_requires_grad_is_set.restype = ctypes.c_bool
    lib.poly_tensor_requires_grad_is_set.argtypes = [_ptr]

    lib.poly_tensor_set_requires_grad.restype = None
    lib.poly_tensor_set_requires_grad.argtypes = [_ptr, ctypes.c_bool]

    lib.poly_realize_tensors.restype = ctypes.c_int
    lib.poly_realize_tensors.argtypes = [
        _ptr, ctypes.POINTER(_ptr), ctypes.c_int, ctypes.POINTER(_ptr)
    ]
    lib.poly_realize_tensors_ex.restype = ctypes.c_int
    lib.poly_realize_tensors_ex.argtypes = [
        _ptr, ctypes.POINTER(_ptr), ctypes.c_int, ctypes.POINTER(_ptr), ctypes.c_bool
    ]

    # --- Raw Tensor JIT capture/replay ---
    lib.poly_jit_new.restype = _ptr
    lib.poly_jit_new.argtypes = [_ptr]

    lib.poly_jit_free.restype = None
    lib.poly_jit_free.argtypes = [_ptr]

    lib.poly_jit_set_prune.restype = ctypes.c_int
    lib.poly_jit_set_prune.argtypes = [_ptr, ctypes.c_bool]

    lib.poly_jit_begin_capture.restype = ctypes.c_int
    lib.poly_jit_begin_capture.argtypes = [_ptr, ctypes.POINTER(_ptr), ctypes.c_int]

    lib.poly_jit_end_capture.restype = ctypes.c_int
    lib.poly_jit_end_capture.argtypes = [_ptr]

    lib.poly_jit_cancel_capture.restype = None
    lib.poly_jit_cancel_capture.argtypes = [_ptr]

    lib.poly_jit_is_captured.restype = ctypes.c_bool
    lib.poly_jit_is_captured.argtypes = [_ptr]

    lib.poly_jit_schedule_count.restype = ctypes.c_int
    lib.poly_jit_schedule_count.argtypes = [_ptr]

    lib.poly_jit_run.restype = ctypes.c_int
    lib.poly_jit_run.argtypes = [_ptr, ctypes.POINTER(_ptr), ctypes.c_int]

    lib.poly_jit_run_with_vars.restype = ctypes.c_int
    lib.poly_jit_run_with_vars.argtypes = [
        _ptr, ctypes.POINTER(_ptr), ctypes.c_int, ctypes.POINTER(PolyVarBinding), ctypes.c_int
    ]

    # --- Optimizer graph builders (optim.h) ---
    lib.poly_optim_build_step.restype = ctypes.c_int
    lib.poly_optim_build_step.argtypes = [
        _ptr,
        ctypes.POINTER(PolyOptimConfig),
        _ptr,
        ctypes.POINTER(_ptr),
        ctypes.POINTER(_ptr),
        ctypes.c_int,
        ctypes.POINTER(_ptr),
        ctypes.POINTER(_ptr),
        _ptr,
        _ptr,
        ctypes.POINTER(_ptr),
        ctypes.c_int,
    ]

    # --- Einsum ---
    lib.poly_einsum.restype = _ptr
    lib.poly_einsum.argtypes = [
        _ptr, ctypes.c_char_p,
        ctypes.POINTER(_ptr),
        ctypes.c_int,
    ]

    # --- Rearrange ---
    lib.poly_rearrange.restype = _ptr
    lib.poly_rearrange.argtypes = [
        _ptr, ctypes.c_char_p, _ptr,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_int64), ctypes.c_int,
    ]
    lib.poly_tensor_rearrange.restype = _ptr
    lib.poly_tensor_rearrange.argtypes = [
        _ptr, ctypes.c_char_p, _ptr,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_int64), ctypes.c_int,
    ]

    # --- PolyInstance (instance.h) ---
    lib.poly_instance_from_ir.restype = _ptr
    lib.poly_instance_from_ir.argtypes = [_u8p, ctypes.c_int, _u8p, ctypes.c_int]
    lib.poly_instance_from_program.restype = _ptr
    lib.poly_instance_from_program.argtypes = [_u8p, ctypes.c_int, _u8p, ctypes.c_int]

    lib.poly_instance_from_sinks.restype = _ptr
    lib.poly_instance_from_sinks.argtypes = [
        _ptr, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(_ptr), ctypes.c_int
    ]

    lib.poly_instance_from_bindings.restype = _ptr
    lib.poly_instance_from_bindings.argtypes = [
        _ptr,
        ctypes.POINTER(PolyBindingSpec), ctypes.c_int,
        ctypes.POINTER(PolyEntrypointSpec), ctypes.c_int,
        ctypes.POINTER(PolyInstanceOptions),
        ctypes.POINTER(PolyInstanceError),
    ]

    lib.poly_instance_define_module_arrays.restype = ctypes.c_int
    lib.poly_instance_define_module_arrays.argtypes = [
        _ptr,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(_ptr),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(_ptr),
        ctypes.c_int,
    ]

    lib.poly_instance_set_device_map_arrays.restype = ctypes.c_int
    lib.poly_instance_set_device_map_arrays.argtypes = [
        _ptr,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.c_int,
    ]

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
    lib.poly_instance_param_data_raw.restype = _ptr
    lib.poly_instance_param_data_raw.argtypes = [_ptr, ctypes.c_int, _i64p]
    lib.poly_instance_param_dtype_id.restype = ctypes.c_int
    lib.poly_instance_param_dtype_id.argtypes = [_ptr, ctypes.c_int]
    lib.poly_instance_param_nbytes.restype = ctypes.c_size_t
    lib.poly_instance_param_nbytes.argtypes = [_ptr, ctypes.c_int]

    lib.poly_instance_param_trainable.restype = ctypes.c_bool
    lib.poly_instance_param_trainable.argtypes = [_ptr, ctypes.c_int]

    lib.poly_instance_set_param_trainable.restype = ctypes.c_int
    lib.poly_instance_set_param_trainable.argtypes = [_ptr, ctypes.c_int, ctypes.c_bool]

    lib.poly_instance_buf_count.restype = ctypes.c_int
    lib.poly_instance_buf_count.argtypes = [_ptr]

    lib.poly_instance_buf_name.restype = ctypes.c_char_p
    lib.poly_instance_buf_name.argtypes = [_ptr, ctypes.c_int]

    lib.poly_instance_buf_role.restype = ctypes.c_int
    lib.poly_instance_buf_role.argtypes = [_ptr, ctypes.c_int]

    lib.poly_instance_buf_trainable.restype = ctypes.c_bool
    lib.poly_instance_buf_trainable.argtypes = [_ptr, ctypes.c_int]

    lib.poly_instance_set_buf_trainable.restype = ctypes.c_int
    lib.poly_instance_set_buf_trainable.argtypes = [_ptr, ctypes.c_int, ctypes.c_bool]

    lib.poly_instance_buf_shape.restype = ctypes.c_int
    lib.poly_instance_buf_shape.argtypes = [_ptr, ctypes.c_int, _i64p, ctypes.c_int]

    lib.poly_instance_buf_data.restype = _fp
    lib.poly_instance_buf_data.argtypes = [_ptr, ctypes.c_int, _i64p]
    lib.poly_instance_buf_data_raw.restype = _ptr
    lib.poly_instance_buf_data_raw.argtypes = [_ptr, ctypes.c_int, _i64p]
    lib.poly_instance_buf_dtype_id.restype = ctypes.c_int
    lib.poly_instance_buf_dtype_id.argtypes = [_ptr, ctypes.c_int]
    lib.poly_instance_buf_nbytes.restype = ctypes.c_size_t
    lib.poly_instance_buf_nbytes.argtypes = [_ptr, ctypes.c_int]

    lib.poly_instance_export_weights.restype = _u8p
    lib.poly_instance_export_weights.argtypes = [_ptr, _ip]
    lib.poly_instance_export_weights_ex.restype = _u8p
    lib.poly_instance_export_weights_ex.argtypes = [_ptr, _ip, ctypes.c_uint32]

    lib.poly_instance_import_weights.restype = ctypes.c_int
    lib.poly_instance_import_weights.argtypes = [_ptr, _u8p, ctypes.c_int]

    lib.poly_instance_export_ir.restype = _u8p
    lib.poly_instance_export_ir.argtypes = [_ptr, _ip]
    lib.poly_instance_export_program.restype = _u8p
    lib.poly_instance_export_program.argtypes = [_ptr, _ip]

    # Bundle format
    lib.poly_instance_save_bundle.restype = _u8p
    lib.poly_instance_save_bundle.argtypes = [_ptr, _ip]
    lib.poly_instance_save_bundle_ex.restype = _u8p
    lib.poly_instance_save_bundle_ex.argtypes = [_ptr, _ip, ctypes.c_uint32]

    lib.poly_instance_from_bundle.restype = _ptr
    lib.poly_instance_from_bundle.argtypes = [_u8p, ctypes.c_int]

    lib.poly_instance_forward.restype = ctypes.c_int
    lib.poly_instance_forward.argtypes = [_ptr, ctypes.POINTER(PolyIOBinding), ctypes.c_int]

    lib.poly_instance_call.restype = ctypes.c_int
    lib.poly_instance_call.argtypes = [
        _ptr, ctypes.c_char_p, ctypes.POINTER(PolyIOBinding), ctypes.c_int]

    lib.poly_instance_entrypoint_count.restype = ctypes.c_int
    lib.poly_instance_entrypoint_count.argtypes = [_ptr]
    lib.poly_instance_entrypoint_name.restype = ctypes.c_char_p
    lib.poly_instance_entrypoint_name.argtypes = [_ptr, ctypes.c_int]
    lib.poly_instance_entrypoint_input_count.restype = ctypes.c_int
    lib.poly_instance_entrypoint_input_count.argtypes = [_ptr, ctypes.c_char_p]
    lib.poly_instance_entrypoint_input_name.restype = ctypes.c_char_p
    lib.poly_instance_entrypoint_input_name.argtypes = [_ptr, ctypes.c_char_p, ctypes.c_int]
    lib.poly_instance_entrypoint_output_count.restype = ctypes.c_int
    lib.poly_instance_entrypoint_output_count.argtypes = [_ptr, ctypes.c_char_p]
    lib.poly_instance_entrypoint_output_name.restype = ctypes.c_char_p
    lib.poly_instance_entrypoint_output_name.argtypes = [_ptr, ctypes.c_char_p, ctypes.c_int]

    lib.poly_instance_train_step.restype = ctypes.c_int
    lib.poly_instance_train_step.argtypes = [_ptr, ctypes.POINTER(PolyIOBinding), ctypes.c_int, _fp]

    lib.poly_instance_set_optimizer.restype = ctypes.c_int
    lib.poly_instance_set_optimizer.argtypes = [_ptr, ctypes.c_int,
        ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_float, ctypes.c_float]
    lib.poly_instance_set_optimizer_ex.restype = ctypes.c_int
    lib.poly_instance_set_optimizer_ex.argtypes = [_ptr, ctypes.c_int,
        ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_bool, ctypes.c_bool]

    # MLP family builder (model_mlp.h)
    lib.poly_mlp_from_json.restype = _ptr
    lib.poly_mlp_from_json.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]

    # TabM family builder (model_tabm.h)
    lib.poly_tabm_instance.restype = _ptr
    lib.poly_tabm_instance.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]

    # NAM family builder (model_nam.h)
    lib.poly_nam_instance.restype = _ptr
    lib.poly_nam_instance.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]

    # HF/model loaders (src/models/*.c)
    lib.poly_hf_load.restype = _ptr
    lib.poly_hf_load.argtypes = [
        ctypes.c_char_p, ctypes.c_int,
        ctypes.POINTER(_u8p),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]

    lib.poly_gguf_load.restype = _ptr
    lib.poly_gguf_load.argtypes = [
        _u8p, ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]

    # --- Shape-on-UOp accessors ---
    lib.poly_uop_ndim.restype = ctypes.c_int
    lib.poly_uop_ndim.argtypes = [_ptr, _ptr]

    lib.poly_uop_max_shape_dims.restype = _i64p
    lib.poly_uop_max_shape_dims.argtypes = [_ptr, _ptr]

    lib.poly_uop_shape_dim.restype = _ptr
    lib.poly_uop_shape_dim.argtypes = [_ptr, _ptr, ctypes.c_int]

    lib.poly_uop_const_i64.restype = ctypes.c_int
    lib.poly_uop_const_i64.argtypes = [_ptr, ctypes.POINTER(ctypes.c_int64)]

    lib.poly_uop_unbind_var.restype = _ptr
    lib.poly_uop_unbind_var.argtypes = [_ptr]

    lib.poly_uop_bind_value.restype = ctypes.c_int
    lib.poly_uop_bind_value.argtypes = [_ptr, ctypes.POINTER(ctypes.c_int64)]

    # --- Additional composed ops (shape read from UOp) ---
    lib.poly_rmsnorm_apply.restype = _ptr
    lib.poly_rmsnorm_apply.argtypes = [_ptr, _ptr, _ptr, ctypes.c_double]

    lib.poly_sdpa.restype = _ptr
    lib.poly_sdpa.argtypes = [_ptr, _ptr, _ptr, _ptr, _ptr, ctypes.c_int]

    lib.poly_rope.restype = _ptr
    lib.poly_rope.argtypes = [_ptr, _ptr, _ptr, _ptr]

    lib.poly_repeat_interleave.restype = _ptr
    lib.poly_repeat_interleave.argtypes = [_ptr, _ptr, ctypes.c_int, ctypes.c_int]

    lib.poly_argmax.restype = _ptr
    lib.poly_argmax.argtypes = [_ptr, _ptr, ctypes.c_int, ctypes.c_int]

    lib.poly_sort.restype = ctypes.c_int
    lib.poly_sort.argtypes = [_ptr, _ptr, ctypes.c_int, ctypes.c_int, _ptrp, _ptrp]

    lib.poly_argsort.restype = _ptr
    lib.poly_argsort.argtypes = [_ptr, _ptr, ctypes.c_int, ctypes.c_int]

    lib.poly_topk.restype = ctypes.c_int
    lib.poly_topk.argtypes = [_ptr, _ptr, ctypes.c_int64, ctypes.c_int, ctypes.c_int,
                              ctypes.c_int, _ptrp, _ptrp]

    lib.poly_mse_loss.restype = _ptr
    lib.poly_mse_loss.argtypes = [_ptr, _ptr, _ptr]

    lib.poly_mae_loss.restype = _ptr
    lib.poly_mae_loss.argtypes = [_ptr, _ptr, _ptr]

    # --- CUDA helpers (conditional) ---
    has_cuda = hasattr(lib, 'poly_cuda_available')
    if has_cuda:
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
    try:
        lib.poly_abi_version.restype = ctypes.c_int
        lib.poly_abi_version.argtypes = []
        abi = int(lib.poly_abi_version())
    except AttributeError as exc:
        raise RuntimeError(
            f'Polygrad ABI mismatch: expected version {POLYGRAD_ABI_VERSION}, '
            'but the loaded library has no poly_abi_version symbol. Rebuild the library.'
        ) from exc
    if abi != POLYGRAD_ABI_VERSION:
        raise RuntimeError(
            f'Polygrad ABI mismatch: expected version {POLYGRAD_ABI_VERSION}, got {abi}. '
            'Rebuild the library or install a matching polygrad package.'
        )
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
