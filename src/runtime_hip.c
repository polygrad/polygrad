/*
 * runtime_hip.c -- HIP/ROCm runtime via dlopen (no link-time -lhip or -lhiprtc)
 *
 * Lazy-loads libamdhip64.so and libhiprtc.so at runtime, resolves all symbols
 * via dlsym. Provides compilation (hiprtc -> HSACO) and execution.
 *
 * Mirrors runtime_cuda.c with HIP-specific API names and types.
 * Key differences from CUDA:
 *   - hipDeviceptr_t is void* (not unsigned long long)
 *   - hiprtc produces HSACO ELF directly (no PTX intermediate)
 *   - Architecture string comes from hipGetDeviceProperties().gcnArchName
 *   - Device math library (__ocml_*) is linked by hiprtc by default;
 *     if compilation fails on __ocml_* calls, pass
 *     -hip-device-lib-path=/opt/rocm/amdgcn/bitcode to hiprtc options.
 */

#ifdef POLY_HAS_HIP

#include "codegen.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <dlfcn.h>

/* ── HIP API types ─────────────────────────────────────────────────── */

typedef int hipError_t;
typedef int hipDevice_t;
typedef void *hipModule_t;
typedef void *hipFunction_t;
typedef void *hipDeviceptr_t;

#define hipSuccess 0

/* hipMemcpyKind */
#define hipMemcpyHostToDevice 1
#define hipMemcpyDeviceToHost 2

/* hipDeviceProperties_t: we only need gcnArchName and warpSize.
 * Full struct is ~700 bytes; define a large enough buffer and access
 * the fields at known offsets. Instead, use a minimal struct that
 * covers the fields we read, then pad generously. */
typedef struct {
  char name[256];
  size_t totalGlobalMem;
  size_t sharedMemPerBlock;
  int regsPerBlock;
  int warpSize;
  int _padding1[64]; /* skip to gcnArchName at a safe offset */
  char gcnArchName[256];
  char _padding2[4096]; /* ensure we don't undersize the struct */
} HipDevicePropsPartial;

/* ── HIPRTC types ──────────────────────────────────────────────────── */

typedef int hiprtcResult;
typedef void *hiprtcProgram;

#define HIPRTC_SUCCESS 0

/* ── PolyHipProgram struct (forward-declared in codegen.h) ────────── */

struct PolyHipProgram {
  void *module;    /* hipModule_t */
  void *function;  /* hipFunction_t */
};

/* ── Function pointer typedefs ─────────────────────────────────────── */

/* HIP runtime */
typedef hipError_t (*hipInit_fn)(unsigned int);
typedef hipError_t (*hipGetDeviceCount_fn)(int *);
typedef hipError_t (*hipSetDevice_fn)(int);
typedef hipError_t (*hipGetDeviceProperties_fn)(void *, int);
typedef hipError_t (*hipMalloc_fn)(void **, size_t);
typedef hipError_t (*hipFree_fn)(void *);
typedef hipError_t (*hipMemcpy_fn)(void *, const void *, size_t, int);
typedef hipError_t (*hipMemset_fn)(void *, int, size_t);
typedef hipError_t (*hipModuleLoadData_fn)(hipModule_t *, const void *);
typedef hipError_t (*hipModuleGetFunction_fn)(hipFunction_t *, hipModule_t, const char *);
typedef hipError_t (*hipModuleLaunchKernel_fn)(hipFunction_t, unsigned int, unsigned int,
                                               unsigned int, unsigned int, unsigned int,
                                               unsigned int, unsigned int, void *,
                                               void **, void **);
typedef hipError_t (*hipDeviceSynchronize_fn)(void);
typedef hipError_t (*hipModuleUnload_fn)(hipModule_t);

/* HIPRTC */
typedef hiprtcResult (*hiprtcCreateProgram_fn)(hiprtcProgram *, const char *, const char *,
                                               int, const char *const *, const char *const *);
typedef hiprtcResult (*hiprtcCompileProgram_fn)(hiprtcProgram, int, const char *const *);
typedef hiprtcResult (*hiprtcGetProgramLogSize_fn)(hiprtcProgram, size_t *);
typedef hiprtcResult (*hiprtcGetProgramLog_fn)(hiprtcProgram, char *);
typedef hiprtcResult (*hiprtcGetCodeSize_fn)(hiprtcProgram, size_t *);
typedef hiprtcResult (*hiprtcGetCode_fn)(hiprtcProgram, char *);
typedef hiprtcResult (*hiprtcDestroyProgram_fn)(hiprtcProgram *);

/* ── Loaded symbols ────────────────────────────────────────────────── */

static struct {
  void *libhip;
  void *libhiprtc;

  /* runtime */
  hipInit_fn hipInit;
  hipGetDeviceCount_fn hipGetDeviceCount;
  hipSetDevice_fn hipSetDevice;
  hipGetDeviceProperties_fn hipGetDeviceProperties;
  hipMalloc_fn hipMalloc;
  hipFree_fn hipFree;
  hipMemcpy_fn hipMemcpy;
  hipMemset_fn hipMemset;
  hipModuleLoadData_fn hipModuleLoadData;
  hipModuleGetFunction_fn hipModuleGetFunction;
  hipModuleLaunchKernel_fn hipModuleLaunchKernel;
  hipDeviceSynchronize_fn hipDeviceSynchronize;
  hipModuleUnload_fn hipModuleUnload;

  /* hiprtc */
  hiprtcCreateProgram_fn hiprtcCreateProgram;
  hiprtcCompileProgram_fn hiprtcCompileProgram;
  hiprtcGetProgramLogSize_fn hiprtcGetProgramLogSize;
  hiprtcGetProgramLog_fn hiprtcGetProgramLog;
  hiprtcGetCodeSize_fn hiprtcGetCodeSize;
  hiprtcGetCode_fn hiprtcGetCode;
  hiprtcDestroyProgram_fn hiprtcDestroyProgram;
} hip_api = {0};

/* ── Lazy singleton state ──────────────────────────────────────────── */

static enum { HIP_NOT_TRIED, HIP_INIT_OK, HIP_INIT_FAIL } hip_state = HIP_NOT_TRIED;
static char hip_arch_name[256] = {0};
static int hip_warp_size = 64; /* default for CDNA */

/* ── dlsym helper ──────────────────────────────────────────────────── */

static void *hip_load_sym(void *lib, const char *name) {
  void *sym = dlsym(lib, name);
  if (!sym) {
    fprintf(stderr, "polygrad: hip: dlsym(%s) failed: %s\n", name, dlerror());
  }
  return sym;
}

/* ── Load libraries + resolve all symbols ──────────────────────────── */

static bool load_hip_libs(void) {
  /* Try common library names */
  hip_api.libhip = dlopen("libamdhip64.so.6", RTLD_LAZY);
  if (!hip_api.libhip)
    hip_api.libhip = dlopen("libamdhip64.so", RTLD_LAZY);
  if (!hip_api.libhip) {
    fprintf(stderr, "polygrad: hip: cannot load libamdhip64.so: %s\n", dlerror());
    return false;
  }

  hip_api.libhiprtc = dlopen("libhiprtc.so.6", RTLD_LAZY);
  if (!hip_api.libhiprtc)
    hip_api.libhiprtc = dlopen("libhiprtc.so", RTLD_LAZY);
  if (!hip_api.libhiprtc) {
    fprintf(stderr, "polygrad: hip: cannot load libhiprtc.so: %s\n", dlerror());
    dlclose(hip_api.libhip);
    hip_api.libhip = NULL;
    return false;
  }

  /* Resolve HIP runtime symbols */
#define LOAD_HIP(name) do { \
    *(void **)&hip_api.name = hip_load_sym(hip_api.libhip, #name); \
    if (!hip_api.name) return false; \
  } while (0)

  LOAD_HIP(hipInit);
  LOAD_HIP(hipGetDeviceCount);
  LOAD_HIP(hipSetDevice);
  LOAD_HIP(hipGetDeviceProperties);
  LOAD_HIP(hipMalloc);
  LOAD_HIP(hipFree);
  LOAD_HIP(hipMemcpy);
  LOAD_HIP(hipMemset);
  LOAD_HIP(hipModuleLoadData);
  LOAD_HIP(hipModuleGetFunction);
  LOAD_HIP(hipModuleLaunchKernel);
  LOAD_HIP(hipDeviceSynchronize);
  LOAD_HIP(hipModuleUnload);

#undef LOAD_HIP

  /* Resolve HIPRTC symbols */
#define LOAD_HIPRTC(name) do { \
    *(void **)&hip_api.name = hip_load_sym(hip_api.libhiprtc, #name); \
    if (!hip_api.name) return false; \
  } while (0)

  LOAD_HIPRTC(hiprtcCreateProgram);
  LOAD_HIPRTC(hiprtcCompileProgram);
  LOAD_HIPRTC(hiprtcGetProgramLogSize);
  LOAD_HIPRTC(hiprtcGetProgramLog);
  LOAD_HIPRTC(hiprtcGetCodeSize);
  LOAD_HIPRTC(hiprtcGetCode);
  LOAD_HIPRTC(hiprtcDestroyProgram);

#undef LOAD_HIPRTC

  return true;
}

/* ── Public API ────────────────────────────────────────────────────── */

int poly_hip_init(void) {
  if (hip_state == HIP_INIT_OK) return 0;
  if (hip_state == HIP_INIT_FAIL) return -1;

  /* First attempt */
  hip_state = HIP_INIT_FAIL; /* assume failure until success */

  if (!load_hip_libs()) return -1;

  hipError_t err;

  err = hip_api.hipInit(0);
  if (err != hipSuccess) {
    fprintf(stderr, "polygrad: hip: hipInit failed (hipError_t=%d)\n", err);
    return -1;
  }

  int dev_count = 0;
  err = hip_api.hipGetDeviceCount(&dev_count);
  if (err != hipSuccess || dev_count == 0) {
    fprintf(stderr, "polygrad: hip: no HIP devices found (hipError_t=%d, count=%d)\n",
            err, dev_count);
    return -1;
  }

  err = hip_api.hipSetDevice(0);
  if (err != hipSuccess) {
    fprintf(stderr, "polygrad: hip: hipSetDevice(0) failed (hipError_t=%d)\n", err);
    return -1;
  }

  /* Query device properties for architecture name and warp size */
  HipDevicePropsPartial props;
  memset(&props, 0, sizeof(props));
  err = hip_api.hipGetDeviceProperties(&props, 0);
  if (err != hipSuccess) {
    fprintf(stderr, "polygrad: hip: hipGetDeviceProperties failed (hipError_t=%d)\n", err);
    return -1;
  }

  /* Cache architecture name and warp size */
  strncpy(hip_arch_name, props.gcnArchName, sizeof(hip_arch_name) - 1);
  hip_arch_name[sizeof(hip_arch_name) - 1] = '\0';
  /* Strip any trailing colon-separated flags (e.g. "gfx90a:sramecc+:xnack-") */
  char *colon = strchr(hip_arch_name, ':');
  if (colon) *colon = '\0';

  hip_warp_size = props.warpSize > 0 ? props.warpSize : 64;

  hip_state = HIP_INIT_OK;
  return 0;
}

bool poly_hip_available(void) {
  if (hip_state == HIP_NOT_TRIED) poly_hip_init();
  return hip_state == HIP_INIT_OK;
}

int poly_hip_wave_size(void) {
  return hip_warp_size;
}

const char *poly_hip_arch(void) {
  return hip_arch_name;
}

void *poly_hip_alloc(size_t bytes) {
  if (hip_state != HIP_INIT_OK) {
    fprintf(stderr, "polygrad: hip: alloc called but HIP not initialized\n");
    return NULL;
  }
  void *ptr = NULL;
  hipError_t err = hip_api.hipMalloc(&ptr, bytes);
  if (err != hipSuccess) {
    fprintf(stderr, "polygrad: hip: hipMalloc(%zu) failed (hipError_t=%d)\n", bytes, err);
    return NULL;
  }
  return ptr;
}

void poly_hip_free(void *ptr) {
  if (hip_state != HIP_INIT_OK) return;
  if (!ptr) return;
  hipError_t err = hip_api.hipFree(ptr);
  if (err != hipSuccess) {
    fprintf(stderr, "polygrad: hip: hipFree failed (hipError_t=%d)\n", err);
  }
}

int poly_hip_copy_htod(void *dst, const void *src, size_t bytes) {
  if (hip_state != HIP_INIT_OK) return -1;
  hipError_t err = hip_api.hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice);
  if (err != hipSuccess) {
    fprintf(stderr, "polygrad: hip: hipMemcpy(HtoD) failed (hipError_t=%d)\n", err);
    return -1;
  }
  return 0;
}

int poly_hip_copy_dtoh(void *dst, const void *src, size_t bytes) {
  if (hip_state != HIP_INIT_OK) return -1;
  hipError_t err = hip_api.hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost);
  if (err != hipSuccess) {
    fprintf(stderr, "polygrad: hip: hipMemcpy(DtoH) failed (hipError_t=%d)\n", err);
    return -1;
  }
  return 0;
}

PolyHipProgram *poly_compile_hip(const char *source, const char *fn_name) {
  if (hip_state != HIP_INIT_OK) {
    fprintf(stderr, "polygrad: hip: compile called but HIP not initialized\n");
    return NULL;
  }

  hiprtcResult rtc_err;
  hipError_t hip_err;

  /* ── HIPRTC: source -> HSACO ──────────────────────────────────────── */

  hiprtcProgram prog = NULL;
  rtc_err = hip_api.hiprtcCreateProgram(&prog, source, fn_name, 0, NULL, NULL);
  if (rtc_err != HIPRTC_SUCCESS) {
    fprintf(stderr, "polygrad: hip: hiprtcCreateProgram failed (hiprtcResult=%d)\n", rtc_err);
    return NULL;
  }

  /* Build --gpu-architecture option from detected gcnArchName */
  char arch_opt[320];
  snprintf(arch_opt, sizeof(arch_opt), "--gpu-architecture=%s", hip_arch_name);

  const char *opts[] = { arch_opt };
  rtc_err = hip_api.hiprtcCompileProgram(prog, 1, opts);
  if (rtc_err != HIPRTC_SUCCESS) {
    /* Print compilation log */
    size_t log_size = 0;
    hip_api.hiprtcGetProgramLogSize(prog, &log_size);
    if (log_size > 1) {
      char *log = malloc(log_size);
      if (log) {
        hip_api.hiprtcGetProgramLog(prog, log);
        fprintf(stderr, "polygrad: hip: HIPRTC compilation failed:\n%s\n", log);
        free(log);
      }
    } else {
      fprintf(stderr, "polygrad: hip: hiprtcCompileProgram failed (hiprtcResult=%d)\n", rtc_err);
    }
    hip_api.hiprtcDestroyProgram(&prog);
    return NULL;
  }

  /* Extract HSACO binary (not PTX like NVRTC) */
  size_t code_size = 0;
  rtc_err = hip_api.hiprtcGetCodeSize(prog, &code_size);
  if (rtc_err != HIPRTC_SUCCESS || code_size == 0) {
    fprintf(stderr, "polygrad: hip: hiprtcGetCodeSize failed (hiprtcResult=%d)\n", rtc_err);
    hip_api.hiprtcDestroyProgram(&prog);
    return NULL;
  }

  char *code = malloc(code_size);
  if (!code) {
    fprintf(stderr, "polygrad: hip: malloc(%zu) for HSACO failed\n", code_size);
    hip_api.hiprtcDestroyProgram(&prog);
    return NULL;
  }

  rtc_err = hip_api.hiprtcGetCode(prog, code);
  hip_api.hiprtcDestroyProgram(&prog);
  if (rtc_err != HIPRTC_SUCCESS) {
    fprintf(stderr, "polygrad: hip: hiprtcGetCode failed (hiprtcResult=%d)\n", rtc_err);
    free(code);
    return NULL;
  }

  /* ── HIP: HSACO -> module -> function ─────────────────────────────── */

  hipModule_t module = NULL;
  hip_err = hip_api.hipModuleLoadData(&module, code);
  free(code);
  if (hip_err != hipSuccess) {
    fprintf(stderr, "polygrad: hip: hipModuleLoadData failed (hipError_t=%d)\n", hip_err);
    return NULL;
  }

  hipFunction_t function = NULL;
  hip_err = hip_api.hipModuleGetFunction(&function, module, fn_name);
  if (hip_err != hipSuccess) {
    fprintf(stderr, "polygrad: hip: hipModuleGetFunction('%s') failed (hipError_t=%d)\n"
            "  Ensure the kernel is wrapped in extern \"C\" to prevent name mangling.\n",
            fn_name, hip_err);
    hip_api.hipModuleUnload(module);
    return NULL;
  }

  /* ── Package result ──────────────────────────────────────────────── */

  PolyHipProgram *result = malloc(sizeof(PolyHipProgram));
  if (!result) {
    hip_api.hipModuleUnload(module);
    return NULL;
  }
  result->module = module;
  result->function = function;
  return result;
}

int poly_hip_launch(PolyHipProgram *prog, void **args, int n_args,
                    int gx, int gy, int gz, int bx, int by, int bz) {
  (void)n_args; /* kernel knows its own param count */

  if (!prog || hip_state != HIP_INIT_OK) return -1;

  hipError_t err = hip_api.hipModuleLaunchKernel(
    (hipFunction_t)prog->function,
    (unsigned int)gx, (unsigned int)gy, (unsigned int)gz,  /* grid */
    (unsigned int)bx, (unsigned int)by, (unsigned int)bz,  /* block */
    0,      /* shared memory bytes */
    NULL,   /* stream (0 = default) */
    args,   /* kernel arguments */
    NULL    /* extra (unused) */
  );
  if (err != hipSuccess) {
    fprintf(stderr, "polygrad: hip: hipModuleLaunchKernel failed (hipError_t=%d)\n", err);
    return -1;
  }
  return 0;
}

int poly_hip_sync(void) {
  if (hip_state != HIP_INIT_OK) return -1;
  hipError_t err = hip_api.hipDeviceSynchronize();
  if (err != hipSuccess) {
    fprintf(stderr, "polygrad: hip: hipDeviceSynchronize failed (hipError_t=%d)\n", err);
    return -1;
  }
  return 0;
}

void poly_hip_program_destroy(PolyHipProgram *prog) {
  if (!prog) return;
  if (prog->module && hip_state == HIP_INIT_OK) {
    hip_api.hipModuleUnload((hipModule_t)prog->module);
  }
  free(prog);
}

int poly_hip_memset(void *ptr, unsigned char val, size_t bytes) {
  if (hip_state != HIP_INIT_OK) return -1;
  if (!ptr || bytes == 0) return 0;
  hipError_t err = hip_api.hipMemset(ptr, val, bytes);
  if (err != hipSuccess) {
    fprintf(stderr, "polygrad: hip: hipMemset failed (hipError_t=%d)\n", err);
    return -1;
  }
  return 0;
}

#endif /* POLY_HAS_HIP */
