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
POLYGRAD_ABI_VERSION = 108

# --- Opaque pointer type (always available) ---
_ptr = ctypes.c_void_p
_ptrp = ctypes.POINTER(_ptr)
_i64p = ctypes.POINTER(ctypes.c_int64)
_ip = ctypes.POINTER(ctypes.c_int)
_uintptr = ctypes.c_size_t

# --- Structures (always available, no _lib dependency) ---

class PolyVarBinding(ctypes.Structure):
    _fields_ = [('var', _ptr), ('value', ctypes.c_int64)]

class PolyControlBinding(ctypes.Structure):
    _fields_ = [('name', ctypes.c_char_p), ('value', ctypes.c_int64)]

class PolyOnnxOptions(ctypes.Structure):
    _fields_ = [('dimensions_json', ctypes.c_char_p),
                ('external_names', ctypes.POINTER(ctypes.c_char_p)),
                ('external_data', ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8))),
                ('external_lengths', ctypes.POINTER(ctypes.c_int64)),
                ('n_external', ctypes.c_int)]

class PolyIOBinding(ctypes.Structure):
    _fields_ = [('name', ctypes.c_char_p),
                ('data', ctypes.c_void_p),
                ('nbytes', ctypes.c_size_t),
                ('dtype_id', ctypes.c_int),
                ('tensor', _ptr),
                ('shape', ctypes.POINTER(ctypes.c_int64)),
                ('ndim', ctypes.c_int)]

class PolyDType(ctypes.Structure):
    _fields_ = [
        ('priority', ctypes.c_int8),
        ('bitsize', ctypes.c_uint16),
        ('name', ctypes.c_char_p),
        ('fmt', ctypes.c_char),
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
        ('tcoef', ctypes.c_double),
        ('ns_steps', ctypes.c_int),
        ('n_ns_coefficients', ctypes.c_int),
        ('ns_coefficients', ctypes.POINTER(ctypes.c_double)),
        ('pre_wd', ctypes.c_bool),
    ]

class PolyCtxStats(ctypes.Structure):
    _fields_ = [
        ('arena_bytes', ctypes.c_size_t),
        ('arena_high_water', ctypes.c_size_t),
        ('scratch_bytes', ctypes.c_size_t),
        ('scratch_high_water', ctypes.c_size_t),
        ('cse_entries', ctypes.c_size_t),
        ('to_program_cache_entries', ctypes.c_size_t),
        ('runtime_cache_entries', ctypes.c_size_t),
        ('shape_cache_entries', ctypes.c_size_t),
        ('buffer_entries', ctypes.c_size_t),
        ('buffer_owned_bytes', ctypes.c_size_t),
        ('buffer_owned_current_bytes', ctypes.c_size_t),
        ('buffer_owned_source_bytes', ctypes.c_size_t),
        ('tensor_records', ctypes.c_size_t),
        ('compiled_artifact_bytes', ctypes.c_size_t),
        ('runtime_artifact_entries', ctypes.c_size_t),
        ('launch_count', ctypes.c_size_t),
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

class PolyModelOptions(ctypes.Structure):
    _fields_ = [
        ('own_ctx_on_success', ctypes.c_bool),
        ('own_ctx_on_failure', ctypes.c_bool),
    ]

class PolyModelError(ctypes.Structure):
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
    # 1. POLY_LIB env var (explicit override)
    env_path = os.environ.get('POLY_LIB')
    if env_path:
        if not os.path.isfile(env_path):
            raise RuntimeError(f'POLY_LIB does not name a library file: {env_path}')
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

def _declare_signatures(lib):
    """Declare all ctypes argtypes/restype on the loaded library."""
    from ._generated_ffi import bind
    bind(lib)

    _u8p = ctypes.POINTER(ctypes.c_uint8)
    _fp = ctypes.POINTER(ctypes.c_float)

    # --- Context ---
    lib.poly_get_noopt.restype = ctypes.c_int
    lib.poly_get_noopt.argtypes = []
    lib.poly_set_noopt.restype = None
    lib.poly_set_noopt.argtypes = [ctypes.c_int]
    lib.poly_get_beam.restype = ctypes.c_int
    lib.poly_get_beam.argtypes = []
    lib.poly_set_beam.restype = None
    lib.poly_set_beam.argtypes = [ctypes.c_int]
    lib.poly_get_ignore_beam_cache.restype = ctypes.c_int
    lib.poly_get_ignore_beam_cache.argtypes = []
    lib.poly_set_ignore_beam_cache.restype = None
    lib.poly_set_ignore_beam_cache.argtypes = [ctypes.c_int]
    lib.poly_get_beam_compare.restype = ctypes.c_int
    lib.poly_get_beam_compare.argtypes = []
    lib.poly_set_beam_compare.restype = None
    lib.poly_set_beam_compare.argtypes = [ctypes.c_int]
    for kind in ('float', 'int'):
        getter = getattr(lib, f'poly_get_default_{kind}')
        getter.restype, getter.argtypes = ctypes.c_int, []
        setter = getattr(lib, f'poly_set_default_{kind}')
        setter.restype, setter.argtypes = ctypes.c_int, [ctypes.c_int]
    lib.poly_ctx_new.restype = _ptr
    lib.poly_ctx_new.argtypes = []

    lib.poly_ctx_destroy.restype = None
    lib.poly_ctx_destroy.argtypes = [_ptr]

    lib.poly_ctx_set_preferred_device.restype = None
    lib.poly_ctx_set_preferred_device.argtypes = [_ptr, ctypes.c_int]

    lib.poly_ctx_set_logical_policy.restype = ctypes.c_int
    lib.poly_ctx_set_logical_policy.argtypes = [_ptr, ctypes.c_int]
    lib.poly_ctx_set_kernel_policy.restype = ctypes.c_int
    lib.poly_ctx_set_kernel_policy.argtypes = [_ptr, ctypes.c_int]

    lib.poly_ctx_get_logical_policy.restype = ctypes.c_int
    lib.poly_ctx_get_logical_policy.argtypes = [_ptr]

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
    lib.poly_uop_device_names.restype = ctypes.c_int
    lib.poly_uop_device_names.argtypes = [
        _ptr, _ptr, ctypes.POINTER(ctypes.POINTER(ctypes.c_char_p)), ctypes.POINTER(ctypes.c_bool),
    ]

    lib.poly_uop_resolve.restype = ctypes.c_int
    lib.poly_uop_resolve.argtypes = [_ptr, _ptr, ctypes.c_int]

    lib.poly_device_can_execute.restype = ctypes.c_bool
    lib.poly_device_can_execute.argtypes = [ctypes.c_int]

    lib.poly_device_is_host_addressable.restype = ctypes.c_bool
    lib.poly_device_is_host_addressable.argtypes = [ctypes.c_int]

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
    lib.poly_dtype_by_id.restype = ctypes.c_bool
    lib.poly_dtype_by_id.argtypes = [ctypes.c_int, ctypes.POINTER(PolyDType)]

    lib.poly_dtype_id_by_name.restype = ctypes.c_int
    lib.poly_dtype_id_by_name.argtypes = [ctypes.c_char_p]

    lib.poly_dtype_can_lossless_cast.restype = ctypes.c_bool
    lib.poly_dtype_can_lossless_cast.argtypes = [PolyDType, PolyDType]

    # --- Frontend helpers (frontend.h) ---

    lib.poly_uop_const_int_decimal.restype = _ptr
    lib.poly_uop_const_int_decimal.argtypes = [_ptr, ctypes.c_char_p]

    lib.poly_uop_sink_n.restype = _ptr
    lib.poly_uop_sink_n.argtypes = [_ptr, ctypes.POINTER(_ptr), ctypes.c_int]

    lib.poly_uop_index.restype = _ptr
    lib.poly_uop_index.argtypes = [_ptr, _ptr, ctypes.POINTER(_ptr), ctypes.c_int]

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

    lib.poly_uop_reduce.restype = _ptr
    lib.poly_uop_reduce.argtypes = [_ptr, ctypes.c_int, _ptr, ctypes.POINTER(_ptr), ctypes.c_int]

    lib.poly_uop_numel.restype = ctypes.c_int64
    lib.poly_uop_numel.argtypes = [_ptr, _ptr]

    lib.poly_tensor_empty_uop_by_id.restype = _ptr
    lib.poly_tensor_empty_uop_by_id.argtypes = [
        _ptr, ctypes.c_int, ctypes.POINTER(_ptr), ctypes.c_int, ctypes.c_int,
    ]
    lib.poly_tensor_empty_uop_name_by_id.restype = _ptr
    lib.poly_tensor_empty_uop_name_by_id.argtypes = [
        _ptr, ctypes.c_int, ctypes.POINTER(_ptr), ctypes.c_int, ctypes.c_char_p,
    ]

    lib.poly_tensor_from_host_by_id.restype = _ptr
    lib.poly_tensor_from_host_by_id.argtypes = [
        _ptr, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, _i64p, ctypes.c_int,
    ]

    lib.poly_tensor_full_int_by_id.restype = _ptr
    lib.poly_tensor_full_uint_by_id.restype = _ptr
    lib.poly_tensor_full_uint_by_id.argtypes = [_ptr, _i64p, ctypes.c_int, ctypes.c_uint64, ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.c_bool]
    lib.poly_tensor_full_int_by_id.argtypes = [
        _ptr, _i64p, ctypes.c_int, ctypes.c_int64, ctypes.c_int, ctypes.c_int,
        ctypes.c_bool, ctypes.c_bool,
    ]

    lib.poly_tensor_full_float_by_id.restype = _ptr
    lib.poly_tensor_full_float_by_id.argtypes = [
        _ptr, _i64p, ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.c_int,
        ctypes.c_bool, ctypes.c_bool,
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

    # --- Composed elementwise ops (shape-free) ---

    # Comparisons

    lib.poly_uop_rand.restype = _ptr
    lib.poly_uop_rand.argtypes = [_ptr, _i64p, ctypes.c_int, ctypes.c_uint64]

    lib.poly_uop_randn.restype = _ptr
    lib.poly_uop_randn.argtypes = [_ptr, _i64p, ctypes.c_int, ctypes.c_uint64]

    lib.poly_uop_rand_by_id.restype = _ptr
    lib.poly_uop_rand_by_id.argtypes = [_ptr, _i64p, ctypes.c_int, ctypes.c_uint64, ctypes.c_int]

    lib.poly_uop_randn_by_id.restype = _ptr
    lib.poly_uop_randn_by_id.argtypes = [_ptr, _i64p, ctypes.c_int, ctypes.c_uint64, ctypes.c_int]

    lib.poly_uop_causal_mask_offset.restype = _ptr
    lib.poly_uop_causal_mask_offset.argtypes = [_ptr, _ptr, _ptr]

    lib.poly_uop_full.restype = _ptr
    lib.poly_uop_full.argtypes = [_ptr, _i64p, ctypes.c_int, ctypes.c_double]

    lib.poly_uop_full_int_by_id.restype = _ptr
    lib.poly_uop_full_int_by_id.argtypes = [_ptr, _i64p, ctypes.c_int, ctypes.c_int64, ctypes.c_int]

    lib.poly_uop_full_float_by_id.restype = _ptr
    lib.poly_uop_full_float_by_id.argtypes = [_ptr, _i64p, ctypes.c_int, ctypes.c_double, ctypes.c_int]

    # --- Shape-aware composed ops (shape read from UOp) ---


    lib.poly_uop_qr.restype = ctypes.c_int
    lib.poly_uop_qr.argtypes = [_ptr, _ptr, _ptrp, _ptrp]

    lib.poly_uop_qr_ex.restype = ctypes.c_int
    lib.poly_uop_qr_ex.argtypes = [_ptr, _ptr, ctypes.c_int, _ptrp, _ptrp]

    lib.poly_uop_scatter.restype = _ptr
    lib.poly_uop_scatter.argtypes = [_ptr, _ptr, ctypes.c_int, _ptr, _ptr, ctypes.c_char_p]

    lib.poly_uop_scatter_reduce.restype = _ptr
    lib.poly_uop_scatter_reduce.argtypes = [
        _ptr, _ptr, ctypes.c_int, _ptr, _ptr, ctypes.c_char_p, ctypes.c_int
    ]

    # --- Current UOp.variable / UOp.bind storage topology ---
    lib.poly_uop_variable_by_id.restype = _ptr
    lib.poly_uop_variable_by_id.argtypes = [
        _ptr, ctypes.c_char_p, _ptr, _ptr,
        ctypes.c_int, ctypes.c_int64, ctypes.c_bool,
    ]

    lib.poly_uop_is_variable.restype = ctypes.c_bool
    lib.poly_uop_is_variable.argtypes = [_ptr]
    lib.poly_uop_is_bound_var.restype = ctypes.c_bool
    lib.poly_uop_is_bound_var.argtypes = [_ptr]

    # --- Sched helpers (sched.h) ---
    lib.poly_uop_reshape.restype = _ptr
    lib.poly_uop_reshape.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int]

    lib.poly_uop_expand.restype = _ptr
    lib.poly_uop_expand.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int]
    lib.poly_uop_expand_symbolic.restype = _ptr
    lib.poly_uop_expand_symbolic.argtypes = [_ptr, _ptr, ctypes.POINTER(_ptr), ctypes.c_int]

    lib.poly_uop_reduce_axis.restype = _ptr
    lib.poly_uop_reduce_axis.argtypes = [_ptr, ctypes.c_int, _ptr, _i64p, ctypes.c_int]

    lib.poly_uop_permute.restype = _ptr
    lib.poly_uop_permute.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int]

    lib.poly_uop_shrink.restype = _ptr
    lib.poly_uop_shrink.argtypes = [_ptr, _ptr, ctypes.c_void_p, ctypes.c_int]
    lib.poly_uop_shrink_symbolic.restype = _ptr
    lib.poly_uop_shrink_symbolic.argtypes = [_ptr, _ptr, ctypes.POINTER(_ptr), ctypes.POINTER(_ptr), ctypes.c_int]

    lib.poly_uop_flip.restype = _ptr
    lib.poly_uop_flip.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int]

    lib.poly_uop_pad.restype = _ptr
    lib.poly_uop_pad.argtypes = [_ptr, _ptr, ctypes.c_void_p, ctypes.c_int]

    lib.poly_uop_pad_value.restype = _ptr
    lib.poly_uop_pad_value.argtypes = [_ptr, _ptr, ctypes.c_void_p, ctypes.c_int, ctypes.c_double]

    lib.poly_uop_pool.restype = _ptr
    lib.poly_uop_pool.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int, _i64p, _i64p]

    lib.poly_uop_max_pool2d.restype = _ptr
    lib.poly_uop_max_pool2d.argtypes = [
        _ptr, _ptr, _i64p, ctypes.c_int, _i64p, _i64p, _i64p, ctypes.c_int,
    ]

    lib.poly_uop_conv2d.restype = _ptr
    lib.poly_uop_conv2d.argtypes = [
        _ptr, _ptr, _ptr, _ptr, ctypes.c_int, _i64p, _i64p, _i64p, ctypes.c_int,
    ]

    lib.poly_uop_batchnorm.restype = _ptr
    lib.poly_uop_batchnorm.argtypes = [
        _ptr, _ptr, _ptr, _ptr, _ptr, _ptr, _i64p, ctypes.c_int,
    ]

    # --- Autograd ---

    lib.poly_uop_substitute.restype = _ptr
    lib.poly_uop_substitute.argtypes = [_ptr, _ptr, ctypes.POINTER(_ptr), ctypes.POINTER(_ptr), ctypes.c_int]

    lib.poly_uop_grad_many.restype = ctypes.c_int
    lib.poly_uop_grad_many.argtypes = [_ptr, _ptr, _ptr, ctypes.POINTER(_ptr), ctypes.c_int, ctypes.POINTER(_ptr)]

    lib.poly_uop_grad_many_ex.restype = ctypes.c_int
    lib.poly_uop_grad_many_ex.argtypes = [
        _ptr, _ptr, _ptr, ctypes.POINTER(_ptr), ctypes.c_int,
        ctypes.POINTER(_ptr), ctypes.POINTER(ctypes.c_uint8),
    ]

    # --- UOp identity helpers ---
    lib.poly_uop_has_buffer_identity.restype = ctypes.c_bool
    lib.poly_uop_has_buffer_identity.argtypes = [_ptr]

    lib.poly_uop_get_buffer_identity.restype = _ptr
    lib.poly_uop_get_buffer_identity.argtypes = [_ptr]

    lib.poly_uop_reachable.restype = ctypes.c_bool
    lib.poly_uop_reachable.argtypes = [_ptr, _ptr, _ptr]

    lib.poly_uop_n_src.restype = ctypes.c_int
    lib.poly_uop_n_src.argtypes = [_ptr]

    lib.poly_uop_call_grad_fxn_key.restype = ctypes.c_uint32
    lib.poly_uop_call_grad_fxn_key.argtypes = [_ptr]

    # --- Side-table buffer API (device.h) ---
    lib.poly_buffer_set.restype = ctypes.c_int
    lib.poly_buffer_set.argtypes = [_ptr, _ptr, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]

    lib.poly_buffer_get_ptr.restype = ctypes.c_void_p
    lib.poly_buffer_get_ptr.argtypes = [_ptr, _ptr]

    lib.poly_buffer_get_key.restype = ctypes.c_uint64
    lib.poly_buffer_get_key.argtypes = [_ptr, _ptr]

    lib.poly_buffer_ensure_device_allocated.restype = ctypes.c_int
    lib.poly_buffer_ensure_device_allocated.argtypes = [_ptr, _ptr, ctypes.c_int]
    lib.poly_buffer_ensure_allocated.restype = ctypes.c_int
    lib.poly_buffer_ensure_allocated.argtypes = [_ptr, _ptr, ctypes.c_int]

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
    lib.poly_tensor_create_result_like.restype = _ptr
    lib.poly_tensor_create_result_like.argtypes = [
        _ptr, _ptr, _ptr, _ptr, ctypes.c_int, ctypes.c_int,
    ]

    lib.poly_tensor_release.restype = None
    lib.poly_tensor_release.argtypes = [_ptr]

    lib.poly_uop_retain.restype = ctypes.c_int
    lib.poly_uop_retain.argtypes = [_ptr, _ptr]

    lib.poly_uop_release.restype = None
    lib.poly_uop_release.argtypes = [_ptr, _ptr]

    lib.poly_ctx_collect.restype = ctypes.c_int
    lib.poly_ctx_collect.argtypes = [_ptr]
    lib.poly_schedule_cache_clear.restype = ctypes.c_int
    lib.poly_schedule_cache_clear.argtypes = [_ptr]

    lib.poly_tensor_replace_roots.restype = ctypes.c_int
    lib.poly_tensor_replace_roots.argtypes = [
        _ptr, _ptr, _ptr, _ptr, ctypes.c_int, ctypes.c_int,
    ]
    lib.poly_tensor_capture_begin.restype = _ptr
    lib.poly_tensor_capture_begin.argtypes = [_ptr]
    lib.poly_tensor_capture_wrap.restype = ctypes.c_int
    lib.poly_tensor_capture_wrap.argtypes = [_ptr, ctypes.POINTER(_ptr), ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                                           ctypes.POINTER(_ptr), ctypes.c_int, ctypes.POINTER(_ptr)]
    lib.poly_tensor_capture_rng.restype = ctypes.c_int
    lib.poly_tensor_capture_rng.argtypes = [_ptr, ctypes.c_int, ctypes.POINTER(_ptr), ctypes.POINTER(_ptr)]
    lib.poly_tensor_capture_end.restype = None
    lib.poly_tensor_capture_end.argtypes = [_ptr]

    lib.poly_tensor_to_device_name.restype = _ptr
    lib.poly_tensor_to_device_name.argtypes = [_ptr, _ptr, ctypes.c_char_p]

    lib.poly_tensor_prod.restype = _ptr
    lib.poly_tensor_prod.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int, ctypes.c_bool]
    lib.poly_tensor_logsumexp.restype = _ptr
    lib.poly_tensor_logsumexp.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int, ctypes.c_bool]
    lib.poly_tensor_pad_mode.restype = _ptr
    lib.poly_tensor_pad_mode.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int, ctypes.c_int]
    for name in ('relu', 'ceil', 'floor', 'mish', 'hardswish', 'sigmoid', 'abs', 'sign',
                 'round', 'isnan', 'silu', 'square', 'rsqrt', 'relu6', 'logical_not', 'neg'):
        fn = getattr(lib, 'poly_tensor_' + name)
        fn.restype = _ptr
        fn.argtypes = [_ptr, _ptr]
    for name in ('elu_tensor', 'logaddexp', 'eq', 'ge', 'le', 'triu_tensor', 'tril_tensor'):
        fn = getattr(lib, 'poly_tensor_' + name)
        fn.restype, fn.argtypes = _ptr, [_ptr, _ptr, _ptr]
    for name in ('softplus_tensor', 'clamp', 'hardsigmoid'):
        fn = getattr(lib, 'poly_tensor_' + name)
        fn.restype, fn.argtypes = _ptr, [_ptr, _ptr, _ptr, _ptr]
    lib.poly_tensor_repeat.restype = _ptr
    lib.poly_tensor_repeat.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int]
    for name in ('triu', 'tril'):
        fn = getattr(lib, 'poly_tensor_' + name)
        fn.restype, fn.argtypes = _ptr, [_ptr, _ptr, ctypes.c_int]
    lib.poly_tensor_stack.restype = _ptr
    lib.poly_tensor_stack.argtypes = [_ptr, ctypes.POINTER(_ptr), ctypes.c_int, ctypes.c_int]
    lib.poly_tensor_cat.restype = _ptr
    lib.poly_tensor_cat.argtypes = [_ptr, ctypes.POINTER(_ptr), ctypes.c_int, ctypes.c_int]

    lib.poly_tensor_custom_kernel.restype = ctypes.c_int
    lib.poly_tensor_custom_kernel.argtypes = [
        _ptr, _ptr, ctypes.POINTER(_ptr), ctypes.c_int, ctypes.c_uint32,
        ctypes.POINTER(_ptr),
    ]

    lib.poly_tensor_function.restype = ctypes.c_int
    lib.poly_tensor_function.argtypes = [
        _ptr, ctypes.POINTER(_ptr), ctypes.c_int,
        ctypes.POINTER(_ptr), ctypes.POINTER(_ptr), ctypes.c_int,
        ctypes.c_char_p, ctypes.c_bool, ctypes.c_bool, ctypes.c_bool,
        ctypes.POINTER(_ptr),
    ]

    lib.poly_gguf_decode.restype = ctypes.c_int
    lib.poly_gguf_decode.argtypes = [ctypes.c_char_p, ctypes.c_int64, _ptrp]
    lib.poly_gguf_decoded_free.restype = None
    lib.poly_gguf_decoded_free.argtypes = [_ptr]
    lib.poly_tokenizer_from_gguf.restype = _ptr
    lib.poly_tokenizer_from_gguf.argtypes = [_ptr]
    lib.poly_tokenizer_from_json_ex.restype = _ptr
    lib.poly_tokenizer_from_json_ex.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
                                               ctypes.c_char_p, ctypes.c_int]
    lib.poly_tokenizer_free.restype = None
    lib.poly_tokenizer_free.argtypes = [_ptr]
    lib.poly_tokenize.restype = ctypes.c_int
    lib.poly_tokenize.argtypes = [_ptr, ctypes.c_char_p, _ip, ctypes.c_int]
    lib.poly_detokenize.restype = ctypes.c_int
    lib.poly_detokenize.argtypes = [_ptr, _ip, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    for name in ('vocab_size', 'bos_id', 'eos_id'):
        fn = getattr(lib, 'poly_tokenizer_' + name)
        fn.restype = ctypes.c_int
        fn.argtypes = [_ptr]

    lib.poly_tensor_sum_dtype_by_id.restype = _ptr
    lib.poly_tensor_sum_dtype_by_id.argtypes = [
        _ptr, _ptr, _i64p, ctypes.c_int, ctypes.c_bool, ctypes.c_int,
    ]

    for name in ('all', 'any'):
        fn = getattr(lib, 'poly_tensor_' + name)
        fn.restype = _ptr
        fn.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int, ctypes.c_bool]
    for name in ('cumsum', 'cumprod'):
        fn = getattr(lib, 'poly_tensor_' + name)
        fn.restype = _ptr
        fn.argtypes = [_ptr, _ptr, ctypes.c_int]
    for name in ('cummax', 'cummin'):
        fn = getattr(lib, 'poly_tensor_' + name)
        fn.restype = ctypes.c_int
        fn.argtypes = [_ptr, _ptr, ctypes.c_int, _ptrp, _ptrp]

    lib.poly_tensor_qr_ex.restype = ctypes.c_int
    lib.poly_tensor_qr_ex.argtypes = [_ptr, _ptr, ctypes.c_int, _ptrp, _ptrp]

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
    lib.poly_tensor_lstm_cell.restype = ctypes.c_int
    lib.poly_tensor_lstm_cell.argtypes = [_ptr] * 8 + [_ptrp, _ptrp]
    lib.poly_tensor_linear_apply.restype = _ptr
    lib.poly_tensor_linear_apply.argtypes = [_ptr] * 4
    lib.poly_tensor_rmsnorm_apply.restype = _ptr
    lib.poly_tensor_rmsnorm_apply.argtypes = [_ptr] * 3 + [ctypes.c_double]
    lib.poly_tensor_instancenorm_apply.restype = _ptr
    lib.poly_tensor_instancenorm_apply.argtypes = [_ptr] * 4 + [ctypes.c_int, ctypes.c_double]
    lib.poly_tensor_groupnorm_apply.restype = _ptr
    lib.poly_tensor_groupnorm_apply.argtypes = [_ptr] * 4 + [ctypes.c_int, ctypes.c_double]
    lib.poly_tensor_layernorm_axes_apply.restype = _ptr
    lib.poly_tensor_layernorm_axes_apply.argtypes = [_ptr] * 4 + [ctypes.POINTER(ctypes.c_int64), ctypes.c_int, ctypes.c_double]
    lib.poly_tensor_batchnorm_stats.restype = ctypes.c_int
    lib.poly_tensor_batchnorm_stats.argtypes = [_ptr] * 4 + [ctypes.c_bool, _ptrp, _ptrp]
    lib.poly_tensor_batchnorm_apply.restype = _ptr
    lib.poly_tensor_batchnorm_apply.argtypes = [_ptr] * 7 + [ctypes.c_bool, ctypes.c_double, ctypes.c_double]
    lib.poly_tensor_topk.argtypes = [
        _ptr, _ptr, ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        _ptrp, _ptrp
    ]

    lib.poly_tensor_dropout.restype = _ptr
    lib.poly_tensor_sample_logits.restype = _ptr
    lib.poly_tensor_sample_logits.argtypes = [_ptr, _ptr, _ptr, ctypes.c_int, ctypes.c_int]
    lib.poly_tensor_dropout.argtypes = [_ptr, _ptr, ctypes.c_double, ctypes.c_int]
    lib.poly_tensor_sdpa.restype = _ptr
    lib.poly_tensor_sdpa.argtypes = [_ptr] * 5 + [ctypes.c_double] + [ctypes.c_int] * 3

    lib.poly_tensor_reshape_uop.restype = _ptr
    lib.poly_tensor_reshape_uop.argtypes = [
        _ptr, _ptr, ctypes.POINTER(_ptr), ctypes.c_int,
    ]

    lib.poly_tensor_expand_uop.restype = _ptr
    lib.poly_tensor_expand_uop.argtypes = [
        _ptr, _ptr, ctypes.POINTER(_ptr), ctypes.c_int,
    ]

    lib.poly_tensor_shrink.restype = _ptr
    lib.poly_tensor_shrink.argtypes = [_ptr, _ptr, ctypes.c_void_p, ctypes.c_int]

    lib.poly_tensor_shrink_uop.restype = _ptr
    lib.poly_tensor_shrink_uop.argtypes = [
        _ptr, _ptr, ctypes.POINTER(_ptr), ctypes.POINTER(_ptr), ctypes.c_int,
    ]

    lib.poly_tensor_pad_value_bool.restype = _ptr
    lib.poly_tensor_pad_value_bool.argtypes = [
        _ptr, _ptr, ctypes.c_void_p, ctypes.c_int, ctypes.c_bool,
    ]
    lib.poly_tensor_pad_value_int.restype = _ptr
    lib.poly_tensor_pad_value_int.argtypes = [
        _ptr, _ptr, ctypes.c_void_p, ctypes.c_int, ctypes.c_int64,
    ]
    lib.poly_tensor_pad_value_float.restype = _ptr
    lib.poly_tensor_pad_value_float.argtypes = [
        _ptr, _ptr, ctypes.c_void_p, ctypes.c_int, ctypes.c_double,
    ]

    lib.poly_tensor_pool.restype = _ptr
    lib.poly_tensor_pool.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int, _i64p, _i64p]

    lib.poly_tensor_max_pool2d.restype = _ptr
    lib.poly_tensor_max_pool2d.argtypes = [
        _ptr, _ptr, _i64p, ctypes.c_int, _i64p, _i64p, _i64p, ctypes.c_int, ctypes.c_bool, ctypes.POINTER(_ptr),
    ]
    lib.poly_tensor_avg_pool2d.restype = _ptr
    lib.poly_tensor_avg_pool2d.argtypes = [
        _ptr, _ptr, _i64p, ctypes.c_int, _i64p, _i64p, _i64p, ctypes.c_int, ctypes.c_bool, ctypes.c_bool,
    ]
    lib.poly_tensor_interpolate.restype = _ptr
    lib.poly_tensor_max_unpool2d.restype = _ptr
    lib.poly_tensor_max_unpool2d.argtypes = [
        _ptr, _ptr, _ptr, _i64p, ctypes.c_int, _i64p, _i64p, _i64p, ctypes.c_int, _i64p, ctypes.c_int,
    ]
    lib.poly_tensor_interpolate.argtypes = [_ptr, _ptr, _i64p, ctypes.c_int, ctypes.c_char_p, ctypes.c_bool]
    lib.poly_tensor_conv_transpose2d.restype = _ptr
    lib.poly_tensor_conv_transpose2d.argtypes = [
        _ptr, _ptr, _ptr, _ptr, ctypes.c_int, _i64p, _i64p, _i64p, ctypes.c_int, _i64p, ctypes.c_int,
    ]
    lib.poly_tensor_full_invalid_by_id.restype = _ptr
    lib.poly_tensor_full_invalid_by_id.argtypes = [_ptr, _i64p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_bool]

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

    _index_args = [_ptr, _ptr, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(_ptr),
                   ctypes.POINTER(_ptr), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(_ptr), ctypes.c_int]
    lib.poly_tensor_getitem.restype = _ptr
    lib.poly_tensor_getitem.argtypes = _index_args
    lib.poly_tensor_setitem.restype = ctypes.c_int
    lib.poly_tensor_setitem.argtypes = _index_args + [_ptr]

    lib.poly_tensor_logical_policy.restype = ctypes.c_int
    lib.poly_tensor_logical_policy.argtypes = [_ptr]

    lib.poly_tensor_logical_state.restype = ctypes.c_int
    lib.poly_tensor_logical_state.argtypes = [_ptr]

    lib.poly_tensor_set_logical_policy.restype = ctypes.c_int
    lib.poly_tensor_set_logical_policy.argtypes = [_ptr, _ptr, ctypes.c_int]

    lib.poly_tensor_device.restype = ctypes.c_int
    lib.poly_tensor_device.argtypes = [_ptr]

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
    lib.poly_jit_end_capture.argtypes = [_ptr, ctypes.POINTER(_ptr), ctypes.c_int]

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
    lib.poly_uop_einsum.restype = _ptr
    lib.poly_uop_einsum.argtypes = [
        _ptr, ctypes.c_char_p,
        ctypes.POINTER(_ptr),
        ctypes.c_int,
    ]

    # --- Rearrange ---
    lib.poly_uop_rearrange.restype = _ptr
    lib.poly_uop_rearrange.argtypes = [
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

    lib.poly_import_last_error_message.restype = ctypes.c_char_p
    lib.poly_import_last_error_message.argtypes = []

    # --- PolyModel (model.h) ---
    lib.poly_model_from_config.restype = _ptr
    lib.poly_model_from_config.argtypes = [_ptr, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(PolyModelError)]
    lib.poly_model_type_name.restype = ctypes.c_char_p
    lib.poly_model_type_name.argtypes = [ctypes.c_int]
    lib.poly_model_type_capabilities.restype = ctypes.c_int
    lib.poly_model_type_capabilities.argtypes = [ctypes.c_int]
    lib.poly_model_last_error.restype = ctypes.POINTER(PolyModelError)
    lib.poly_model_last_error.argtypes = [_ptr]
    lib.poly_model_from_ir.restype = _ptr
    lib.poly_model_from_ir.argtypes = [_u8p, ctypes.c_int, _u8p, ctypes.c_int]
    lib.poly_model_from_ir_into.restype = _ptr
    lib.poly_model_from_ir_into.argtypes = [_ptr, _u8p, ctypes.c_int, _u8p, ctypes.c_int, ctypes.c_int]
    lib.poly_model_from_program.restype = _ptr
    lib.poly_model_from_program.argtypes = [_u8p, ctypes.c_int, _u8p, ctypes.c_int]

    lib.poly_model_from_bindings.restype = _ptr
    lib.poly_model_from_bindings.argtypes = [
        _ptr,
        ctypes.POINTER(PolyBindingSpec), ctypes.c_int,
        ctypes.POINTER(PolyEntrypointSpec), ctypes.c_int,
        ctypes.POINTER(PolyModelOptions),
        ctypes.POINTER(PolyModelError),
    ]

    lib.poly_model_define_module_arrays.restype = ctypes.c_int
    lib.poly_model_define_module_arrays.argtypes = [
        _ptr,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(_ptr),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(_ptr),
        ctypes.c_int,
    ]

    lib.poly_model_set_device_map_arrays.restype = ctypes.c_int
    lib.poly_model_set_device.restype = ctypes.c_int
    lib.poly_model_set_device.argtypes = [_ptr, ctypes.c_int]
    lib.poly_model_set_device_name.restype = ctypes.c_int
    lib.poly_model_set_device_name.argtypes = [_ptr, ctypes.c_char_p]
    lib.poly_model_set_device_map_arrays.argtypes = [
        _ptr,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.c_int,
    ]

    lib.poly_model_free.restype = None
    lib.poly_model_free.argtypes = [_ptr]

    lib.poly_model_param_count.restype = ctypes.c_int
    lib.poly_model_param_count.argtypes = [_ptr]

    lib.poly_model_param_name.restype = ctypes.c_char_p
    lib.poly_model_param_name.argtypes = [_ptr, ctypes.c_int]

    lib.poly_model_param_shape.restype = ctypes.c_int
    lib.poly_model_param_shape.argtypes = [_ptr, ctypes.c_int, _i64p, ctypes.c_int]

    lib.poly_model_param_data.restype = _fp
    lib.poly_model_param_data.argtypes = [_ptr, ctypes.c_int, _i64p]
    lib.poly_model_param_data_raw.restype = _ptr
    lib.poly_model_param_data_raw.argtypes = [_ptr, ctypes.c_int, _i64p]
    lib.poly_model_param_dtype_id.restype = ctypes.c_int
    lib.poly_model_param_dtype_id.argtypes = [_ptr, ctypes.c_int]
    lib.poly_model_param_nbytes.restype = ctypes.c_size_t
    lib.poly_model_param_nbytes.argtypes = [_ptr, ctypes.c_int]

    lib.poly_model_param_trainable.restype = ctypes.c_bool
    lib.poly_model_param_trainable.argtypes = [_ptr, ctypes.c_int]

    lib.poly_model_set_param_trainable.restype = ctypes.c_int
    lib.poly_model_set_param_trainable.argtypes = [_ptr, ctypes.c_int, ctypes.c_bool]

    lib.poly_model_buf_count.restype = ctypes.c_int
    lib.poly_model_buf_count.argtypes = [_ptr]

    lib.poly_model_buf_name.restype = ctypes.c_char_p
    lib.poly_model_buf_name.argtypes = [_ptr, ctypes.c_int]

    lib.poly_model_find_buf.restype = ctypes.c_int
    lib.poly_model_find_buf.argtypes = [_ptr, ctypes.c_char_p]

    lib.poly_model_buf_role.restype = ctypes.c_int
    lib.poly_model_buf_role.argtypes = [_ptr, ctypes.c_int]

    lib.poly_model_buf_trainable.restype = ctypes.c_bool
    lib.poly_model_buf_trainable.argtypes = [_ptr, ctypes.c_int]

    lib.poly_model_set_buf_trainable.restype = ctypes.c_int
    lib.poly_model_set_buf_trainable.argtypes = [_ptr, ctypes.c_int, ctypes.c_bool]

    lib.poly_model_buf_shape.restype = ctypes.c_int
    lib.poly_model_buf_shape.argtypes = [_ptr, ctypes.c_int, _i64p, ctypes.c_int]
    lib.poly_model_buf_current_shape.restype = ctypes.c_int
    lib.poly_model_buf_current_shape.argtypes = [_ptr, ctypes.c_int, _i64p, ctypes.c_int]
    lib.poly_model_buf_shape_bounds.restype = ctypes.c_int
    lib.poly_model_buf_shape_bounds.argtypes = [_ptr, ctypes.c_int, _i64p, _i64p, ctypes.c_int]

    lib.poly_model_buf_data.restype = _fp
    lib.poly_model_buf_data.argtypes = [_ptr, ctypes.c_int, _i64p]
    lib.poly_model_buf_data_raw.restype = _ptr
    lib.poly_model_buf_data_raw.argtypes = [_ptr, ctypes.c_int, _i64p]
    lib.poly_model_buf_dtype_id.restype = ctypes.c_int
    lib.poly_model_buf_dtype_id.argtypes = [_ptr, ctypes.c_int]
    lib.poly_model_buf_nbytes.restype = ctypes.c_size_t
    lib.poly_model_buf_nbytes.argtypes = [_ptr, ctypes.c_int]

    lib.poly_model_read_buf.restype = ctypes.c_int
    lib.poly_model_read_buf.argtypes = [_ptr, ctypes.c_int, _ptr, ctypes.c_size_t]
    lib.poly_model_write_buf.restype = ctypes.c_int
    lib.poly_model_write_buf.argtypes = [_ptr, ctypes.c_int, _ptr, ctypes.c_size_t]

    lib.poly_model_export_weights.restype = _u8p
    lib.poly_model_export_weights.argtypes = [_ptr, _ip]
    lib.poly_model_export_weights_ex.restype = _u8p
    lib.poly_model_export_weights_ex.argtypes = [_ptr, _ip, ctypes.c_uint32]

    lib.poly_model_import_weights.restype = ctypes.c_int
    lib.poly_model_import_weights.argtypes = [_ptr, _u8p, ctypes.c_int]

    lib.poly_model_export_ir.restype = _u8p
    lib.poly_model_export_ir.argtypes = [_ptr, _ip]
    lib.poly_model_export_program.restype = _u8p
    lib.poly_model_export_program.argtypes = [_ptr, _ip]

    # Bundle format
    lib.poly_model_save_bundle.restype = _u8p
    lib.poly_model_save_bundle.argtypes = [_ptr, _ip]
    lib.poly_model_save_bundle_ex.restype = _u8p
    lib.poly_model_save_bundle_ex.argtypes = [_ptr, _ip, ctypes.c_uint32]

    lib.poly_model_from_bundle.restype = _ptr
    lib.poly_model_from_bundle.argtypes = [_u8p, ctypes.c_int]
    lib.poly_model_from_bundle_into.restype = _ptr
    lib.poly_model_from_bundle_into.argtypes = [_ptr, _u8p, ctypes.c_int, ctypes.c_int]

    lib.poly_model_forward.restype = ctypes.c_int
    lib.poly_model_forward.argtypes = [_ptr, ctypes.POINTER(PolyIOBinding), ctypes.c_int]

    lib.poly_model_call.restype = ctypes.c_int
    lib.poly_model_call.argtypes = [
        _ptr, ctypes.c_char_p, ctypes.POINTER(PolyIOBinding), ctypes.c_int]
    lib.poly_model_control.restype = ctypes.c_int
    lib.poly_model_reset_transient.restype = ctypes.c_int
    lib.poly_model_reset_transient.argtypes = [_ptr]
    lib.poly_transformer_from_model.restype = _ptr
    lib.poly_transformer_from_model.argtypes = [_ptr, ctypes.POINTER(PolyModelError)]
    lib.poly_transformer_available.restype = ctypes.c_bool
    lib.poly_transformer_available.argtypes = [_ptr]
    lib.poly_transformer_last_error.restype = ctypes.POINTER(PolyModelError)
    lib.poly_transformer_last_error.argtypes = [_ptr]
    lib.poly_transformer_reset.restype = ctypes.c_int
    lib.poly_transformer_reset.argtypes = [_ptr]
    lib.poly_transformer_start.restype = ctypes.c_int
    lib.poly_transformer_start.argtypes = [_ptr, ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
    lib.poly_transformer_next.restype = ctypes.c_int
    lib.poly_transformer_next.argtypes = [_ptr, ctypes.c_float, ctypes.POINTER(ctypes.c_int32)]
    lib.poly_transformer_free.restype = None
    lib.poly_transformer_free.argtypes = [_ptr]
    lib.poly_transformer_vocab.restype = ctypes.c_int
    lib.poly_transformer_vocab.argtypes = [_ptr]
    lib.poly_transformer_position.restype = ctypes.c_int
    lib.poly_transformer_position.argtypes = [_ptr]
    lib.poly_transformer_rewind.restype = ctypes.c_int
    lib.poly_transformer_rewind.argtypes = [_ptr, ctypes.c_int]
    lib.poly_transformer_prefill.restype = ctypes.c_int
    lib.poly_transformer_prefill.argtypes = [_ptr, ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
                                             ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    lib.poly_transformer_append.restype = ctypes.c_int
    lib.poly_transformer_append.argtypes = [_ptr, ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
                                            ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    lib.poly_model_control.argtypes = [_ptr, ctypes.c_char_p, ctypes.c_char_p, _ptr]
    lib.poly_model_call_with_controls.restype = ctypes.c_int
    lib.poly_model_call_with_controls.argtypes = [
        _ptr, ctypes.c_char_p, ctypes.POINTER(PolyIOBinding), ctypes.c_int,
        ctypes.POINTER(PolyControlBinding), ctypes.c_int]
    lib.poly_model_call_tensors_with_controls.restype = ctypes.c_int
    lib.poly_model_call_tensors_with_controls.argtypes = [
        _ptr, ctypes.c_char_p, ctypes.POINTER(PolyIOBinding), ctypes.c_int,
        ctypes.POINTER(PolyControlBinding), ctypes.c_int, ctypes.POINTER(_ptr), ctypes.c_int]
    lib.poly_model_call_tensors.restype = ctypes.c_int
    lib.poly_model_call_tensors.argtypes = [
        _ptr, ctypes.c_char_p, ctypes.POINTER(PolyIOBinding), ctypes.c_int,
        ctypes.POINTER(_ptr), ctypes.c_int]

    lib.poly_model_entrypoint_count.restype = ctypes.c_int
    lib.poly_model_entrypoint_count.argtypes = [_ptr]
    lib.poly_model_entrypoint_name.restype = ctypes.c_char_p
    lib.poly_model_entrypoint_name.argtypes = [_ptr, ctypes.c_int]
    lib.poly_model_entrypoint_objective.restype = ctypes.c_char_p
    lib.poly_model_entrypoint_objective.argtypes = [_ptr, ctypes.c_char_p]
    lib.poly_model_entrypoint_input_count.restype = ctypes.c_int
    lib.poly_model_entrypoint_input_count.argtypes = [_ptr, ctypes.c_char_p]
    lib.poly_model_entrypoint_input_name.restype = ctypes.c_char_p
    lib.poly_model_entrypoint_input_name.argtypes = [_ptr, ctypes.c_char_p, ctypes.c_int]
    lib.poly_model_entrypoint_output_count.restype = ctypes.c_int
    lib.poly_model_entrypoint_output_count.argtypes = [_ptr, ctypes.c_char_p]
    lib.poly_model_entrypoint_output_name.restype = ctypes.c_char_p
    lib.poly_model_entrypoint_output_name.argtypes = [_ptr, ctypes.c_char_p, ctypes.c_int]

    lib.poly_model_train_step.restype = ctypes.c_int
    lib.poly_model_train_step.argtypes = [_ptr, ctypes.c_char_p, ctypes.POINTER(PolyIOBinding), ctypes.c_int, _fp]

    lib.poly_model_set_optimizer.restype = ctypes.c_int
    lib.poly_model_set_learning_rate.restype = ctypes.c_int
    lib.poly_model_set_learning_rate.argtypes = [_ptr, ctypes.c_float]
    lib.poly_model_set_optimizer.argtypes = [_ptr, ctypes.c_int,
        ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_float, ctypes.c_float]
    lib.poly_model_set_optimizer_ex.restype = ctypes.c_int
    lib.poly_model_set_optimizer_ex.argtypes = [_ptr, ctypes.c_int,
        ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_bool, ctypes.c_bool]

    # HF/model loaders (src/models/*.c)
    lib.poly_hf_plan.restype = ctypes.c_int
    lib.poly_hf_plan.argtypes = [ctypes.c_char_p, ctypes.c_char_p, _ptr, ctypes.c_int]
    lib.poly_hf_load_into.restype = _ptr
    lib.poly_hf_load_into.argtypes = [
        _ptr, ctypes.c_char_p, ctypes.c_int,
        ctypes.POINTER(_u8p),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]

    lib.poly_gguf_load_into.restype = _ptr
    lib.poly_gguf_load_into.argtypes = [
        _ptr, _u8p, ctypes.c_int64, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    lib.poly_onnx_load_into.restype = _ptr
    lib.poly_onnx_load_into.argtypes = [
        _ptr, _u8p, ctypes.c_int64, ctypes.POINTER(PolyOnnxOptions), ctypes.c_int
    ]

    # --- Shape-on-UOp accessors ---
    lib.poly_uop_ndim.restype = ctypes.c_int
    lib.poly_uop_ndim.argtypes = [_ptr, _ptr]

    lib.poly_uop_max_shape_dims.restype = _i64p
    lib.poly_uop_max_shape_dims.argtypes = [_ptr, _ptr]

    lib.poly_uop_const_i64.restype = ctypes.c_int
    lib.poly_uop_const_i64.argtypes = [_ptr, ctypes.POINTER(ctypes.c_int64)]

    lib.poly_uop_bind_value.restype = ctypes.c_int
    lib.poly_uop_bind_value.argtypes = [_ptr, ctypes.POINTER(ctypes.c_int64)]

    # --- Additional composed ops (shape read from UOp) ---
    lib.poly_uop_rmsnorm_apply.restype = _ptr
    lib.poly_uop_rmsnorm_apply.argtypes = [_ptr, _ptr, _ptr, ctypes.c_double]

    lib.poly_uop_sdpa.restype = _ptr
    lib.poly_uop_sdpa.argtypes = [_ptr] * 5 + [ctypes.c_int] * 2

    lib.poly_uop_sort.restype = ctypes.c_int
    lib.poly_uop_sort.argtypes = [_ptr, _ptr, ctypes.c_int, ctypes.c_int, _ptrp, _ptrp]

    lib.poly_uop_topk.restype = ctypes.c_int
    lib.poly_uop_topk.argtypes = [_ptr, _ptr, ctypes.c_int64, ctypes.c_int, ctypes.c_int,
                              ctypes.c_int, _ptrp, _ptrp]

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
            'Or set POLY_LIB=/path/to/libpolygrad.so'
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
