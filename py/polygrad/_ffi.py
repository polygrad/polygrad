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
    """Bind the header-derived ABI, including structured and conditional calls."""
    from ._generated_ffi import bind
    from ._generated_special_ffi import bind as bind_special
    bind(lib)
    bind_special(lib, globals())
    return hasattr(lib, 'poly_cuda_available')

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
        from ._generated_special_ffi import bind as bind_special
        bind_special(lib, globals(), only="poly_abi_version")
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
