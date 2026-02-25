/*
 * runtime_cuda.c — CUDA runtime via dlopen (no link-time -lcuda or -lnvrtc)
 *
 * Lazy-loads libcuda.so and libnvrtc.so at runtime, resolves all symbols
 * via dlsym. Provides compilation (NVRTC → PTX → cuModule) and execution.
 */

#ifdef POLY_HAS_CUDA

#include "codegen.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <dlfcn.h>

/* ── CUDA driver API types ───────────────────────────────────────────── */

typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
typedef void *CUmodule;
typedef void *CUfunction;
typedef unsigned long long CUdeviceptr;

/* CUresult codes we check */
#define CUDA_SUCCESS 0

/* cuDeviceGetAttribute IDs (used instead of deprecated cuDeviceComputeCapability) */
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR 75
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR 76

/* ── NVRTC types ─────────────────────────────────────────────────────── */

typedef int nvrtcResult;
typedef void *nvrtcProgram;

#define NVRTC_SUCCESS 0

/* ── PolyCudaProgram struct (forward-declared in codegen.h) ───────────── */

struct PolyCudaProgram {
  void *module;    /* CUmodule */
  void *function;  /* CUfunction */
};

/* ── Function pointer typedefs ───────────────────────────────────────── */

/* CUDA driver */
typedef CUresult (*cuInit_fn)(unsigned int);
typedef CUresult (*cuDeviceGet_fn)(CUdevice *, int);
typedef CUresult (*cuCtxCreate_v2_fn)(CUcontext *, unsigned int, CUdevice);
typedef CUresult (*cuDeviceGetAttribute_fn)(int *, int, CUdevice);
typedef CUresult (*cuMemAlloc_v2_fn)(CUdeviceptr *, size_t);
typedef CUresult (*cuMemFree_v2_fn)(CUdeviceptr);
typedef CUresult (*cuMemcpyHtoD_v2_fn)(CUdeviceptr, const void *, size_t);
typedef CUresult (*cuMemcpyDtoH_v2_fn)(void *, CUdeviceptr, size_t);
typedef CUresult (*cuModuleLoadData_fn)(CUmodule *, const void *);
typedef CUresult (*cuModuleGetFunction_fn)(CUfunction *, CUmodule, const char *);
typedef CUresult (*cuLaunchKernel_fn)(CUfunction, unsigned int, unsigned int, unsigned int,
                                       unsigned int, unsigned int, unsigned int,
                                       unsigned int, void *, void **, void **);
typedef CUresult (*cuCtxSynchronize_fn)(void);
typedef CUresult (*cuModuleUnload_fn)(CUmodule);

/* NVRTC */
typedef nvrtcResult (*nvrtcCreateProgram_fn)(nvrtcProgram *, const char *, const char *,
                                              int, const char *const *, const char *const *);
typedef nvrtcResult (*nvrtcCompileProgram_fn)(nvrtcProgram, int, const char *const *);
typedef nvrtcResult (*nvrtcGetProgramLogSize_fn)(nvrtcProgram, size_t *);
typedef nvrtcResult (*nvrtcGetProgramLog_fn)(nvrtcProgram, char *);
typedef nvrtcResult (*nvrtcGetPTXSize_fn)(nvrtcProgram, size_t *);
typedef nvrtcResult (*nvrtcGetPTX_fn)(nvrtcProgram, char *);
typedef nvrtcResult (*nvrtcDestroyProgram_fn)(nvrtcProgram *);

/* ── Loaded symbols ──────────────────────────────────────────────────── */

static struct {
  void *libcuda;
  void *libnvrtc;

  /* driver */
  cuInit_fn cuInit;
  cuDeviceGet_fn cuDeviceGet;
  cuCtxCreate_v2_fn cuCtxCreate_v2;
  cuDeviceGetAttribute_fn cuDeviceGetAttribute;
  cuMemAlloc_v2_fn cuMemAlloc_v2;
  cuMemFree_v2_fn cuMemFree_v2;
  cuMemcpyHtoD_v2_fn cuMemcpyHtoD_v2;
  cuMemcpyDtoH_v2_fn cuMemcpyDtoH_v2;
  cuModuleLoadData_fn cuModuleLoadData;
  cuModuleGetFunction_fn cuModuleGetFunction;
  cuLaunchKernel_fn cuLaunchKernel;
  cuCtxSynchronize_fn cuCtxSynchronize;
  cuModuleUnload_fn cuModuleUnload;

  /* nvrtc */
  nvrtcCreateProgram_fn nvrtcCreateProgram;
  nvrtcCompileProgram_fn nvrtcCompileProgram;
  nvrtcGetProgramLogSize_fn nvrtcGetProgramLogSize;
  nvrtcGetProgramLog_fn nvrtcGetProgramLog;
  nvrtcGetPTXSize_fn nvrtcGetPTXSize;
  nvrtcGetPTX_fn nvrtcGetPTX;
  nvrtcDestroyProgram_fn nvrtcDestroyProgram;
} cuda_api = {0};

/* ── Lazy singleton state ────────────────────────────────────────────── */

static enum { CUDA_NOT_TRIED, CUDA_INIT_OK, CUDA_INIT_FAIL } cuda_state = CUDA_NOT_TRIED;
static CUcontext cuda_ctx = NULL;
static int cuda_arch_major = 0;
static int cuda_arch_minor = 0;

/* ── dlsym helper ────────────────────────────────────────────────────── */

static void *load_sym(void *lib, const char *name) {
  void *sym = dlsym(lib, name);
  if (!sym) {
    fprintf(stderr, "polygrad: cuda: dlsym(%s) failed: %s\n", name, dlerror());
  }
  return sym;
}

/* ── Load libraries + resolve all symbols ────────────────────────────── */

static bool load_cuda_libs(void) {
  /* Try common library names */
  cuda_api.libcuda = dlopen("libcuda.so.1", RTLD_LAZY);
  if (!cuda_api.libcuda)
    cuda_api.libcuda = dlopen("libcuda.so", RTLD_LAZY);
  if (!cuda_api.libcuda) {
    fprintf(stderr, "polygrad: cuda: cannot load libcuda.so: %s\n", dlerror());
    return false;
  }

  cuda_api.libnvrtc = dlopen("libnvrtc.so.12", RTLD_LAZY);
  if (!cuda_api.libnvrtc)
    cuda_api.libnvrtc = dlopen("libnvrtc.so", RTLD_LAZY);
  if (!cuda_api.libnvrtc) {
    fprintf(stderr, "polygrad: cuda: cannot load libnvrtc.so: %s\n", dlerror());
    dlclose(cuda_api.libcuda);
    cuda_api.libcuda = NULL;
    return false;
  }

  /* Resolve CUDA driver symbols */
#define LOAD_CUDA(name) do { \
    *(void **)&cuda_api.name = load_sym(cuda_api.libcuda, #name); \
    if (!cuda_api.name) return false; \
  } while (0)

  LOAD_CUDA(cuInit);
  LOAD_CUDA(cuDeviceGet);
  LOAD_CUDA(cuCtxCreate_v2);
  LOAD_CUDA(cuDeviceGetAttribute);
  LOAD_CUDA(cuMemAlloc_v2);
  LOAD_CUDA(cuMemFree_v2);
  LOAD_CUDA(cuMemcpyHtoD_v2);
  LOAD_CUDA(cuMemcpyDtoH_v2);
  LOAD_CUDA(cuModuleLoadData);
  LOAD_CUDA(cuModuleGetFunction);
  LOAD_CUDA(cuLaunchKernel);
  LOAD_CUDA(cuCtxSynchronize);
  LOAD_CUDA(cuModuleUnload);

#undef LOAD_CUDA

  /* Resolve NVRTC symbols */
#define LOAD_NVRTC(name) do { \
    *(void **)&cuda_api.name = load_sym(cuda_api.libnvrtc, #name); \
    if (!cuda_api.name) return false; \
  } while (0)

  LOAD_NVRTC(nvrtcCreateProgram);
  LOAD_NVRTC(nvrtcCompileProgram);
  LOAD_NVRTC(nvrtcGetProgramLogSize);
  LOAD_NVRTC(nvrtcGetProgramLog);
  LOAD_NVRTC(nvrtcGetPTXSize);
  LOAD_NVRTC(nvrtcGetPTX);
  LOAD_NVRTC(nvrtcDestroyProgram);

#undef LOAD_NVRTC

  return true;
}

/* ── Public API ──────────────────────────────────────────────────────── */

int poly_cuda_init(void) {
  if (cuda_state == CUDA_INIT_OK) return 0;
  if (cuda_state == CUDA_INIT_FAIL) return -1;

  /* First attempt */
  cuda_state = CUDA_INIT_FAIL; /* assume failure until success */

  if (!load_cuda_libs()) return -1;

  CUresult err;

  err = cuda_api.cuInit(0);
  if (err != CUDA_SUCCESS) {
    fprintf(stderr, "polygrad: cuda: cuInit failed (CUresult=%d)\n", err);
    return -1;
  }

  CUdevice dev;
  err = cuda_api.cuDeviceGet(&dev, 0);
  if (err != CUDA_SUCCESS) {
    fprintf(stderr, "polygrad: cuda: cuDeviceGet failed (CUresult=%d)\n", err);
    return -1;
  }

  /* Query compute capability via cuDeviceGetAttribute */
  err = cuda_api.cuDeviceGetAttribute(&cuda_arch_major,
                                       CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
  if (err != CUDA_SUCCESS) {
    fprintf(stderr, "polygrad: cuda: cuDeviceGetAttribute(MAJOR) failed (CUresult=%d)\n", err);
    return -1;
  }
  err = cuda_api.cuDeviceGetAttribute(&cuda_arch_minor,
                                       CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
  if (err != CUDA_SUCCESS) {
    fprintf(stderr, "polygrad: cuda: cuDeviceGetAttribute(MINOR) failed (CUresult=%d)\n", err);
    return -1;
  }

  err = cuda_api.cuCtxCreate_v2(&cuda_ctx, 0, dev);
  if (err != CUDA_SUCCESS) {
    fprintf(stderr, "polygrad: cuda: cuCtxCreate_v2 failed (CUresult=%d)\n", err);
    return -1;
  }

  cuda_state = CUDA_INIT_OK;
  return 0;
}

bool poly_cuda_available(void) {
  if (cuda_state == CUDA_NOT_TRIED) poly_cuda_init();
  return cuda_state == CUDA_INIT_OK;
}

unsigned long long poly_cuda_alloc(size_t bytes) {
  if (cuda_state != CUDA_INIT_OK) {
    fprintf(stderr, "polygrad: cuda: alloc called but CUDA not initialized\n");
    return 0;
  }
  CUdeviceptr ptr = 0;
  CUresult err = cuda_api.cuMemAlloc_v2(&ptr, bytes);
  if (err != CUDA_SUCCESS) {
    fprintf(stderr, "polygrad: cuda: cuMemAlloc_v2(%zu) failed (CUresult=%d)\n", bytes, err);
    return 0;
  }
  return (unsigned long long)ptr;
}

void poly_cuda_free(unsigned long long ptr) {
  if (cuda_state != CUDA_INIT_OK) return;
  if (ptr == 0) return;
  CUresult err = cuda_api.cuMemFree_v2((CUdeviceptr)ptr);
  if (err != CUDA_SUCCESS) {
    fprintf(stderr, "polygrad: cuda: cuMemFree_v2 failed (CUresult=%d)\n", err);
  }
}

int poly_cuda_copy_htod(unsigned long long dst, const void *src, size_t bytes) {
  if (cuda_state != CUDA_INIT_OK) return -1;
  CUresult err = cuda_api.cuMemcpyHtoD_v2((CUdeviceptr)dst, src, bytes);
  if (err != CUDA_SUCCESS) {
    fprintf(stderr, "polygrad: cuda: cuMemcpyHtoD_v2 failed (CUresult=%d)\n", err);
    return -1;
  }
  return 0;
}

int poly_cuda_copy_dtoh(void *dst, unsigned long long src, size_t bytes) {
  if (cuda_state != CUDA_INIT_OK) return -1;
  CUresult err = cuda_api.cuMemcpyDtoH_v2(dst, (CUdeviceptr)src, bytes);
  if (err != CUDA_SUCCESS) {
    fprintf(stderr, "polygrad: cuda: cuMemcpyDtoH_v2 failed (CUresult=%d)\n", err);
    return -1;
  }
  return 0;
}

PolyCudaProgram *poly_compile_cuda(const char *source, const char *fn_name) {
  if (cuda_state != CUDA_INIT_OK) {
    fprintf(stderr, "polygrad: cuda: compile called but CUDA not initialized\n");
    return NULL;
  }

  nvrtcResult nv_err;
  CUresult cu_err;

  /* ── NVRTC: source → PTX ──────────────────────────────────────────── */

  nvrtcProgram prog = NULL;
  nv_err = cuda_api.nvrtcCreateProgram(&prog, source, fn_name, 0, NULL, NULL);
  if (nv_err != NVRTC_SUCCESS) {
    fprintf(stderr, "polygrad: cuda: nvrtcCreateProgram failed (nvrtcResult=%d)\n", nv_err);
    return NULL;
  }

  /* Build --gpu-architecture option from detected compute capability */
  char arch_opt[64];
  snprintf(arch_opt, sizeof(arch_opt), "--gpu-architecture=compute_%d%d",
           cuda_arch_major, cuda_arch_minor);

  const char *opts[] = { arch_opt };
  nv_err = cuda_api.nvrtcCompileProgram(prog, 1, opts);
  if (nv_err != NVRTC_SUCCESS) {
    /* Print compilation log */
    size_t log_size = 0;
    cuda_api.nvrtcGetProgramLogSize(prog, &log_size);
    if (log_size > 1) {
      char *log = malloc(log_size);
      if (log) {
        cuda_api.nvrtcGetProgramLog(prog, log);
        fprintf(stderr, "polygrad: cuda: NVRTC compilation failed:\n%s\n", log);
        free(log);
      }
    } else {
      fprintf(stderr, "polygrad: cuda: nvrtcCompileProgram failed (nvrtcResult=%d)\n", nv_err);
    }
    cuda_api.nvrtcDestroyProgram(&prog);
    return NULL;
  }

  /* Extract PTX */
  size_t ptx_size = 0;
  nv_err = cuda_api.nvrtcGetPTXSize(prog, &ptx_size);
  if (nv_err != NVRTC_SUCCESS || ptx_size == 0) {
    fprintf(stderr, "polygrad: cuda: nvrtcGetPTXSize failed (nvrtcResult=%d)\n", nv_err);
    cuda_api.nvrtcDestroyProgram(&prog);
    return NULL;
  }

  char *ptx = malloc(ptx_size);
  if (!ptx) {
    fprintf(stderr, "polygrad: cuda: malloc(%zu) for PTX failed\n", ptx_size);
    cuda_api.nvrtcDestroyProgram(&prog);
    return NULL;
  }

  nv_err = cuda_api.nvrtcGetPTX(prog, ptx);
  cuda_api.nvrtcDestroyProgram(&prog);
  if (nv_err != NVRTC_SUCCESS) {
    fprintf(stderr, "polygrad: cuda: nvrtcGetPTX failed (nvrtcResult=%d)\n", nv_err);
    free(ptx);
    return NULL;
  }

  /* ── CUDA driver: PTX → module → function ─────────────────────────── */

  CUmodule module = NULL;
  cu_err = cuda_api.cuModuleLoadData(&module, ptx);
  free(ptx);
  if (cu_err != CUDA_SUCCESS) {
    fprintf(stderr, "polygrad: cuda: cuModuleLoadData failed (CUresult=%d)\n", cu_err);
    return NULL;
  }

  CUfunction function = NULL;
  cu_err = cuda_api.cuModuleGetFunction(&function, module, fn_name);
  if (cu_err != CUDA_SUCCESS) {
    fprintf(stderr, "polygrad: cuda: cuModuleGetFunction(%s) failed (CUresult=%d)\n",
            fn_name, cu_err);
    cuda_api.cuModuleUnload(module);
    return NULL;
  }

  /* ── Package result ────────────────────────────────────────────────── */

  PolyCudaProgram *result = malloc(sizeof(PolyCudaProgram));
  if (!result) {
    cuda_api.cuModuleUnload(module);
    return NULL;
  }
  result->module = module;
  result->function = function;
  return result;
}

int poly_cuda_launch(PolyCudaProgram *prog, void **args, int n_args,
                     int gx, int gy, int gz, int bx, int by, int bz) {
  (void)n_args; /* kernel knows its own param count */

  if (!prog || cuda_state != CUDA_INIT_OK) return -1;

  CUresult err = cuda_api.cuLaunchKernel(
    (CUfunction)prog->function,
    (unsigned int)gx, (unsigned int)gy, (unsigned int)gz,   /* grid */
    (unsigned int)bx, (unsigned int)by, (unsigned int)bz,   /* block */
    0,      /* shared memory bytes */
    NULL,   /* stream (0 = default) */
    args,   /* kernel arguments */
    NULL    /* extra (unused) */
  );
  if (err != CUDA_SUCCESS) {
    fprintf(stderr, "polygrad: cuda: cuLaunchKernel failed (CUresult=%d)\n", err);
    return -1;
  }
  return 0;
}

int poly_cuda_sync(void) {
  if (cuda_state != CUDA_INIT_OK) return -1;
  CUresult err = cuda_api.cuCtxSynchronize();
  if (err != CUDA_SUCCESS) {
    fprintf(stderr, "polygrad: cuda: cuCtxSynchronize failed (CUresult=%d)\n", err);
    return -1;
  }
  return 0;
}

void poly_cuda_program_destroy(PolyCudaProgram *prog) {
  if (!prog) return;
  if (prog->module && cuda_state == CUDA_INIT_OK) {
    cuda_api.cuModuleUnload((CUmodule)prog->module);
  }
  free(prog);
}

#endif /* POLY_HAS_CUDA */
