/*
 * runtime_cuda.c — CUDA runtime via dlopen (no link-time -lcuda or -lnvrtc)
 *
 * Lazy-loads libcuda.so and libnvrtc.so at runtime, resolves all symbols
 * via dlsym. Provides compilation (NVRTC → PTX → cuModule) and execution.
 */

#ifdef POLY_HAS_CUDA

#include "codegen/codegen.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <unistd.h>
#include <dlfcn.h>
#include <limits.h>

/* CUDA driver API types */

typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
typedef void *CUmodule;
typedef void *CUfunction;
typedef unsigned long long CUdeviceptr;
typedef void *CUgraph;
typedef void *CUgraphNode;
typedef void *CUgraphExec;
typedef void *CUstream;
typedef void *CUevent;

/* Legacy CUDA driver graph ABI used by pinned tinygrad's generated bindings.
 * Keep this local so HAS_CUDA remains a dlopen-only build with no CUDA-header
 * or link-time libcuda requirement. */
typedef struct {
  CUfunction func;
  unsigned int gridDimX;
  unsigned int gridDimY;
  unsigned int gridDimZ;
  unsigned int blockDimX;
  unsigned int blockDimY;
  unsigned int blockDimZ;
  unsigned int sharedMemBytes;
  void **kernelParams;
  void **extra;
} PolyCudaKernelNodeParams;

/* Current CUDA_MEMCPY3D_v2 layout used by runtime/graph/cuda.py. */
typedef struct {
  size_t srcXInBytes, srcY, srcZ, srcLOD;
  int srcMemoryType;
  void *srcHost;
  CUdeviceptr srcDevice;
  void *srcArray;
  void *reserved0;
  size_t srcPitch, srcHeight;
  size_t dstXInBytes, dstY, dstZ, dstLOD;
  int dstMemoryType;
  void *dstHost;
  CUdeviceptr dstDevice;
  void *dstArray;
  void *reserved1;
  size_t dstPitch, dstHeight, WidthInBytes, Height, Depth;
} PolyCudaMemcpy3D;

#define CU_MEMORYTYPE_DEVICE 2

/* CUresult codes we check */
#define CUDA_SUCCESS 0

/* cuDeviceGetAttribute IDs (used instead of deprecated cuDeviceComputeCapability) */
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR 75
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR 76

/* NVRTC types */

typedef int nvrtcResult;
typedef void *nvrtcProgram;

#define NVRTC_SUCCESS 0

/* PolyCudaProgram struct (forward-declared in codegen.h) */

struct PolyCudaProgram {
  void *module; /* CUmodule */
  void *function; /* CUfunction */
};

/* Function pointer typedefs */

/* CUDA driver */
typedef CUresult (*cuInit_fn)(unsigned int);
typedef CUresult (*cuDeviceGet_fn)(CUdevice *, int);
typedef CUresult (*cuCtxCreate_v2_fn)(CUcontext *, unsigned int, CUdevice);
typedef CUresult (*cuDeviceGetAttribute_fn)(int *, int, CUdevice);
typedef CUresult (*cuMemAlloc_v2_fn)(CUdeviceptr *, size_t);
typedef CUresult (*cuMemFree_v2_fn)(CUdeviceptr);
typedef CUresult (*cuMemcpyHtoD_v2_fn)(CUdeviceptr, const void *, size_t);
typedef CUresult (*cuMemcpyDtoH_v2_fn)(void *, CUdeviceptr, size_t);
typedef CUresult (*cuMemcpyDtoD_v2_fn)(CUdeviceptr, CUdeviceptr, size_t);
typedef CUresult (*cuModuleLoadData_fn)(CUmodule *, const void *);
typedef CUresult (*cuModuleGetFunction_fn)(CUfunction *, CUmodule, const char *);
typedef CUresult (*cuLaunchKernel_fn
)(CUfunction,
  unsigned int,
  unsigned int,
  unsigned int,
  unsigned int,
  unsigned int,
  unsigned int,
  unsigned int,
  void *,
  void **,
  void **);
typedef CUresult (*cuCtxSynchronize_fn)(void);
typedef CUresult (*cuMemsetD8_v2_fn)(CUdeviceptr, unsigned char, size_t);
typedef CUresult (*cuModuleUnload_fn)(CUmodule);
typedef CUresult (*cuEventCreate_fn)(CUevent *, unsigned int);
typedef CUresult (*cuEventRecord_fn)(CUevent, CUstream);
typedef CUresult (*cuEventSynchronize_fn)(CUevent);
typedef CUresult (*cuEventElapsedTime_fn)(float *, CUevent, CUevent);
typedef CUresult (*cuEventDestroy_v2_fn)(CUevent);
typedef CUresult (*cuGraphCreate_fn)(CUgraph *, unsigned int);
typedef CUresult (*cuGraphAddKernelNode_fn
)(CUgraphNode *, CUgraph, const CUgraphNode *, size_t, const PolyCudaKernelNodeParams *);
typedef CUresult (*cuGraphAddMemcpyNode_fn
)(CUgraphNode *, CUgraph, const CUgraphNode *, size_t, const PolyCudaMemcpy3D *, CUcontext);
typedef CUresult (*cuGraphInstantiate_v2_fn)(CUgraphExec *, CUgraph, CUgraphNode *, char *, size_t);
typedef CUresult (*cuGraphExecKernelNodeSetParams_fn
)(CUgraphExec, CUgraphNode, const PolyCudaKernelNodeParams *);
typedef CUresult (*cuGraphExecMemcpyNodeSetParams_fn
)(CUgraphExec, CUgraphNode, const PolyCudaMemcpy3D *, CUcontext);
typedef CUresult (*cuGraphLaunch_fn)(CUgraphExec, CUstream);
typedef CUresult (*cuGraphDestroy_fn)(CUgraph);
typedef CUresult (*cuGraphExecDestroy_fn)(CUgraphExec);

/* NVRTC */
typedef nvrtcResult (*nvrtcCreateProgram_fn
)(nvrtcProgram *, const char *, const char *, int, const char *const *, const char *const *);
typedef nvrtcResult (*nvrtcCompileProgram_fn)(nvrtcProgram, int, const char *const *);
typedef nvrtcResult (*nvrtcGetProgramLogSize_fn)(nvrtcProgram, size_t *);
typedef nvrtcResult (*nvrtcGetProgramLog_fn)(nvrtcProgram, char *);
typedef nvrtcResult (*nvrtcGetPTXSize_fn)(nvrtcProgram, size_t *);
typedef nvrtcResult (*nvrtcGetPTX_fn)(nvrtcProgram, char *);
typedef nvrtcResult (*nvrtcDestroyProgram_fn)(nvrtcProgram *);

/* Loaded symbols */

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
  cuMemcpyDtoD_v2_fn cuMemcpyDtoD_v2;
  cuModuleLoadData_fn cuModuleLoadData;
  cuModuleGetFunction_fn cuModuleGetFunction;
  cuLaunchKernel_fn cuLaunchKernel;
  cuCtxSynchronize_fn cuCtxSynchronize;
  cuMemsetD8_v2_fn cuMemsetD8_v2;
  cuModuleUnload_fn cuModuleUnload;
  cuEventCreate_fn cuEventCreate;
  cuEventRecord_fn cuEventRecord;
  cuEventSynchronize_fn cuEventSynchronize;
  cuEventElapsedTime_fn cuEventElapsedTime;
  cuEventDestroy_v2_fn cuEventDestroy_v2;
  cuGraphCreate_fn cuGraphCreate;
  cuGraphAddKernelNode_fn cuGraphAddKernelNode;
  cuGraphAddMemcpyNode_fn cuGraphAddMemcpyNode;
  cuGraphInstantiate_v2_fn cuGraphInstantiate_v2;
  cuGraphExecKernelNodeSetParams_fn cuGraphExecKernelNodeSetParams;
  cuGraphExecMemcpyNodeSetParams_fn cuGraphExecMemcpyNodeSetParams;
  cuGraphLaunch_fn cuGraphLaunch;
  cuGraphDestroy_fn cuGraphDestroy;
  cuGraphExecDestroy_fn cuGraphExecDestroy;

  /* nvrtc */
  nvrtcCreateProgram_fn nvrtcCreateProgram;
  nvrtcCompileProgram_fn nvrtcCompileProgram;
  nvrtcGetProgramLogSize_fn nvrtcGetProgramLogSize;
  nvrtcGetProgramLog_fn nvrtcGetProgramLog;
  nvrtcGetPTXSize_fn nvrtcGetPTXSize;
  nvrtcGetPTX_fn nvrtcGetPTX;
  nvrtcDestroyProgram_fn nvrtcDestroyProgram;
} cuda_api = {0};

/* Lazy singleton state */

static enum { CUDA_NOT_TRIED, CUDA_INIT_OK, CUDA_INIT_FAIL } cuda_state = CUDA_NOT_TRIED;
static CUcontext cuda_ctx = NULL;
static int cuda_arch_major = 0;
static int cuda_arch_minor = 0;

/* dlsym helper */

static void *load_sym(void *lib, const char *name) {
  void *sym = dlsym(lib, name);
  if (!sym) {
    fprintf(stderr, "polygrad: cuda: dlsym(%s) failed: %s\n", name, dlerror());
  }
  return sym;
}

/* Load libraries + resolve all symbols */

static bool load_cuda_libs(void) {
  /* Try common library names */
  cuda_api.libcuda = dlopen("libcuda.so.1", RTLD_LAZY);
  if (!cuda_api.libcuda) cuda_api.libcuda = dlopen("libcuda.so", RTLD_LAZY);
  if (!cuda_api.libcuda) {
    fprintf(stderr, "polygrad: cuda: cannot load libcuda.so: %s\n", dlerror());
    return false;
  }

  cuda_api.libnvrtc = dlopen("libnvrtc.so.12", RTLD_LAZY);
  if (!cuda_api.libnvrtc) cuda_api.libnvrtc = dlopen("libnvrtc.so", RTLD_LAZY);
  if (!cuda_api.libnvrtc) {
    fprintf(stderr, "polygrad: cuda: cannot load libnvrtc.so: %s\n", dlerror());
    dlclose(cuda_api.libcuda);
    cuda_api.libcuda = NULL;
    return false;
  }

  /* Resolve CUDA driver symbols */
#define LOAD_CUDA(name)                                                                            \
  do {                                                                                             \
    *(void **)&cuda_api.name = load_sym(cuda_api.libcuda, #name);                                  \
    if (!cuda_api.name) return false;                                                              \
  } while (0)

  LOAD_CUDA(cuInit);
  LOAD_CUDA(cuDeviceGet);
  LOAD_CUDA(cuCtxCreate_v2);
  LOAD_CUDA(cuDeviceGetAttribute);
  LOAD_CUDA(cuMemAlloc_v2);
  LOAD_CUDA(cuMemFree_v2);
  LOAD_CUDA(cuMemcpyHtoD_v2);
  LOAD_CUDA(cuMemcpyDtoH_v2);
  LOAD_CUDA(cuMemcpyDtoD_v2);
  LOAD_CUDA(cuModuleLoadData);
  LOAD_CUDA(cuModuleGetFunction);
  LOAD_CUDA(cuLaunchKernel);
  LOAD_CUDA(cuCtxSynchronize);
  LOAD_CUDA(cuMemsetD8_v2);
  LOAD_CUDA(cuModuleUnload);
  LOAD_CUDA(cuEventCreate);
  LOAD_CUDA(cuEventRecord);
  LOAD_CUDA(cuEventSynchronize);
  LOAD_CUDA(cuEventElapsedTime);
  LOAD_CUDA(cuEventDestroy_v2);

#undef LOAD_CUDA

  /* Pinned graph_split_rewrite only batches when the selected device exposes
   * a graph runtime. Keep these optional so a driver without graph symbols
   * still supports ordinary CUDA PROGRAM launches. */
#define LOAD_CUDA_GRAPH(name) *(void **)&cuda_api.name = dlsym(cuda_api.libcuda, #name)
  LOAD_CUDA_GRAPH(cuGraphCreate);
  LOAD_CUDA_GRAPH(cuGraphAddKernelNode);
  LOAD_CUDA_GRAPH(cuGraphAddMemcpyNode);
  LOAD_CUDA_GRAPH(cuGraphInstantiate_v2);
  LOAD_CUDA_GRAPH(cuGraphExecKernelNodeSetParams);
  LOAD_CUDA_GRAPH(cuGraphExecMemcpyNodeSetParams);
  LOAD_CUDA_GRAPH(cuGraphLaunch);
  LOAD_CUDA_GRAPH(cuGraphDestroy);
  LOAD_CUDA_GRAPH(cuGraphExecDestroy);
#undef LOAD_CUDA_GRAPH

  /* Resolve NVRTC symbols */
#define LOAD_NVRTC(name)                                                                           \
  do {                                                                                             \
    *(void **)&cuda_api.name = load_sym(cuda_api.libnvrtc, #name);                                 \
    if (!cuda_api.name) return false;                                                              \
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

/* Public API */

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
  err = cuda_api.cuDeviceGetAttribute(
      &cuda_arch_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev
  );
  if (err != CUDA_SUCCESS) {
    fprintf(stderr, "polygrad: cuda: cuDeviceGetAttribute(MAJOR) failed (CUresult=%d)\n", err);
    return -1;
  }
  err = cuda_api.cuDeviceGetAttribute(
      &cuda_arch_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev
  );
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

int poly_cuda_arch_major(void) {
  if (cuda_state == CUDA_NOT_TRIED) poly_cuda_init();
  return cuda_arch_major;
}

int poly_cuda_arch_minor(void) {
  if (cuda_state == CUDA_NOT_TRIED) poly_cuda_init();
  return cuda_arch_minor;
}

unsigned long long poly_cuda_alloc(size_t bytes) {
  if (cuda_state == CUDA_NOT_TRIED && poly_cuda_init() != 0) return 0;
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
  /* Match tinygrad CUDAAllocator._copyout: readback is an explicit completion
   * boundary for all previously enqueued work in this CUDA context. */
  if (poly_cuda_sync() != 0) return -1;
  CUresult err = cuda_api.cuMemcpyDtoH_v2(dst, (CUdeviceptr)src, bytes);
  if (err != CUDA_SUCCESS) {
    fprintf(stderr, "polygrad: cuda: cuMemcpyDtoH_v2 failed (CUresult=%d)\n", err);
    return -1;
  }
  return 0;
}

int poly_cuda_copy_dtod(unsigned long long dst, unsigned long long src, size_t bytes) {
  if (cuda_state != CUDA_INIT_OK) return -1;
  CUresult err = cuda_api.cuMemcpyDtoD_v2((CUdeviceptr)dst, (CUdeviceptr)src, bytes);
  if (err != CUDA_SUCCESS) {
    fprintf(stderr, "polygrad: cuda: cuMemcpyDtoD_v2 failed (CUresult=%d)\n", err);
    return -1;
  }
  return 0;
}

PolyCudaProgram *poly_compile_cuda_with_binary(
    const char *source,
    const char *fn_name,
    uint8_t **binary,
    int *binary_size,
    int *status
) {
  if (status) *status = -1;
  if (binary) *binary = NULL;
  if (binary_size) *binary_size = 0;
  if ((binary == NULL) != (binary_size == NULL)) return NULL;
  if (cuda_state != CUDA_INIT_OK) {
    fprintf(stderr, "polygrad: cuda: compile called but CUDA not initialized\n");
    return NULL;
  }

  nvrtcResult nv_err;
  CUresult cu_err;

  /* NVRTC: source → PTX */

  /* compiler_cuda.nvrtc_check raises CompileError, not RuntimeError. Keep
   * that distinction across the C boundary for BEAM_STRICT_MODE. */
  if (status) *status = -2;

  nvrtcProgram prog = NULL;
  nv_err = cuda_api.nvrtcCreateProgram(&prog, source, fn_name, 0, NULL, NULL);
  if (nv_err != NVRTC_SUCCESS) {
    fprintf(stderr, "polygrad: cuda: nvrtcCreateProgram failed (nvrtcResult=%d)\n", nv_err);
    return NULL;
  }

  /* Build --gpu-architecture option from detected compute capability */
  char arch_opt[64];
  snprintf(
      arch_opt, sizeof(arch_opt), "--gpu-architecture=compute_%d%d", cuda_arch_major,
      cuda_arch_minor
  );

  /* Include path for cuda_fp16.h / cuda_bf16.h.
   * NVRTC doesn't search system include dirs by default. */
  char inc_opt[256] = "";
  const char *cuda_path = getenv("CUDA_PATH");
  if (cuda_path)
    snprintf(inc_opt, sizeof(inc_opt), "-I%s/include", cuda_path);
  else if (access("/usr/local/cuda/include/cuda_fp16.h", 0) == 0)
    snprintf(inc_opt, sizeof(inc_opt), "-I/usr/local/cuda/include");
  else
    snprintf(inc_opt, sizeof(inc_opt), "-I/usr/include");

  const char *opts[] = {arch_opt, inc_opt};
  nv_err = cuda_api.nvrtcCompileProgram(prog, 2, opts);
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
  if (nv_err != NVRTC_SUCCESS || ptx_size == 0 || ptx_size > INT_MAX) {
    fprintf(stderr, "polygrad: cuda: nvrtcGetPTXSize failed (nvrtcResult=%d)\n", nv_err);
    cuda_api.nvrtcDestroyProgram(&prog);
    return NULL;
  }

  char *ptx = malloc(ptx_size);
  if (!ptx) {
    if (status) *status = -1;
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

  /* CUDA driver: PTX → module → function */

  if (status) *status = -1;

  CUmodule module = NULL;
  cu_err = cuda_api.cuModuleLoadData(&module, ptx);
  if (cu_err != CUDA_SUCCESS) {
    fprintf(stderr, "polygrad: cuda: cuModuleLoadData failed (CUresult=%d)\n", cu_err);
    free(ptx);
    return NULL;
  }

  CUfunction function = NULL;
  cu_err = cuda_api.cuModuleGetFunction(&function, module, fn_name);
  if (cu_err != CUDA_SUCCESS) {
    fprintf(
        stderr, "polygrad: cuda: cuModuleGetFunction(%s) failed (CUresult=%d)\n", fn_name, cu_err
    );
    cuda_api.cuModuleUnload(module);
    free(ptx);
    return NULL;
  }

  /* Package result */

  PolyCudaProgram *result = malloc(sizeof(PolyCudaProgram));
  if (!result) {
    cuda_api.cuModuleUnload(module);
    free(ptx);
    return NULL;
  }
  result->module = module;
  result->function = function;
  /* search.beam_search compares compiler output, not source or module handles.
   * Transfer the PTX only after the complete runnable candidate exists. */
  if (binary) {
    *binary = (uint8_t *)ptx;
    *binary_size = (int)ptx_size;
  } else {
    free(ptx);
  }
  if (status) *status = 0;
  return result;
}

PolyCudaProgram *poly_compile_cuda(const char *source, const char *fn_name) {
  return poly_compile_cuda_with_binary(source, fn_name, NULL, NULL, NULL);
}

int poly_cuda_launch_timed(
    PolyCudaProgram *prog,
    void **args,
    int n_args,
    int gx,
    int gy,
    int gz,
    int bx,
    int by,
    int bz,
    double *elapsed_us
) {
  if (!elapsed_us || cuda_state != CUDA_INIT_OK) return -1;
  *elapsed_us = 0;
  CUevent start = NULL, end = NULL;
  int rc = -1;
  /* ops_cuda.cu_time_execution: record around the launch on the same stream.
   * Host compilation, argument packing and synchronization are not GPU time. */
  if (cuda_api.cuEventCreate(&start, 0) != CUDA_SUCCESS ||
      cuda_api.cuEventCreate(&end, 0) != CUDA_SUCCESS ||
      cuda_api.cuEventRecord(start, NULL) != CUDA_SUCCESS)
    goto done;
  if (poly_cuda_launch(prog, args, n_args, gx, gy, gz, bx, by, bz) != 0 ||
      cuda_api.cuEventRecord(end, NULL) != CUDA_SUCCESS ||
      cuda_api.cuEventSynchronize(end) != CUDA_SUCCESS)
    goto done;
  float ms = 0;
  if (cuda_api.cuEventElapsedTime(&ms, start, end) != CUDA_SUCCESS) goto done;
  *elapsed_us = (double)ms * 1000;
  rc = 0;
done:
  if (end && cuda_api.cuEventDestroy_v2(end) != CUDA_SUCCESS) rc = -1;
  if (start && cuda_api.cuEventDestroy_v2(start) != CUDA_SUCCESS) rc = -1;
  return rc;
}

int poly_cuda_launch(
    PolyCudaProgram *prog,
    void **args,
    int n_args,
    int gx,
    int gy,
    int gz,
    int bx,
    int by,
    int bz
) {
  (void)n_args; /* kernel knows its own param count */

  if (!prog || cuda_state != CUDA_INIT_OK) return -1;

  CUresult err = cuda_api.cuLaunchKernel(
      (CUfunction)prog->function, (unsigned int)gx, (unsigned int)gy, (unsigned int)gz, /* grid */
      (unsigned int)bx, (unsigned int)by, (unsigned int)bz, /* block */
      0, /* shared memory bytes */
      NULL, /* stream (0 = default) */
      args, /* kernel arguments */
      NULL /* extra (unused) */
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

typedef struct {
  CUgraphNode node;
  PolyCudaGraphCallKind kind;
  PolyCudaKernelNodeParams params;
  PolyCudaMemcpy3D copy;
  CUdeviceptr *buffer_values;
  int64_t *scalar_values;
  void **kernel_params;
  int n_buffer_args;
  int n_args;
} PolyCudaGraphNode;

struct PolyCudaGraph {
  CUgraph graph;
  CUgraphExec instance;
  PolyCudaGraphNode *nodes;
  int n_nodes;
};

bool poly_cuda_graph_available(void) {
  if (cuda_state == CUDA_NOT_TRIED && poly_cuda_init() != 0) return false;
  return cuda_state == CUDA_INIT_OK && cuda_api.cuGraphCreate && cuda_api.cuGraphAddKernelNode &&
         cuda_api.cuGraphAddMemcpyNode && cuda_api.cuGraphInstantiate_v2 &&
         cuda_api.cuGraphExecKernelNodeSetParams && cuda_api.cuGraphExecMemcpyNodeSetParams &&
         cuda_api.cuGraphLaunch && cuda_api.cuGraphDestroy && cuda_api.cuGraphExecDestroy;
}

static int poly_cuda_graph_node_update(PolyCudaGraphNode *node, const PolyCudaGraphCallSpec *spec) {
  if (!node || !spec || node->kind != spec->kind) return -1;
  if (spec->kind == POLY_CUDA_GRAPH_COPY) {
    if (!spec->value.copy.dst || !spec->value.copy.src || spec->value.copy.nbytes == 0) return -1;
    node->copy = (PolyCudaMemcpy3D){
        .srcMemoryType = CU_MEMORYTYPE_DEVICE,
        .srcDevice = (CUdeviceptr)(uintptr_t)spec->value.copy.src,
        .srcPitch = spec->value.copy.nbytes,
        .srcHeight = 1,
        .dstMemoryType = CU_MEMORYTYPE_DEVICE,
        .dstDevice = (CUdeviceptr)(uintptr_t)spec->value.copy.dst,
        .dstPitch = spec->value.copy.nbytes,
        .dstHeight = 1,
        .WidthInBytes = spec->value.copy.nbytes,
        .Height = 1,
        .Depth = 1,
    };
    return 0;
  }

  PolyCudaProgram *program = spec->value.program.program;
  void **args = spec->value.program.args;
  int n_buffer_args = spec->value.program.n_buffer_args;
  int n_args = spec->value.program.n_args;
  if (spec->kind != POLY_CUDA_GRAPH_PROGRAM || !program || !args || n_buffer_args < 0 ||
      n_args < n_buffer_args || node->n_buffer_args != n_buffer_args || node->n_args != n_args)
    return -1;

  for (int i = 0; i < n_buffer_args; i++) {
    node->buffer_values[i] = (CUdeviceptr)(uintptr_t)args[i];
    node->kernel_params[i] = &node->buffer_values[i];
  }
  for (int i = n_buffer_args; i < n_args; i++) {
    int scalar_idx = i - n_buffer_args;
    if (!args[i]) return -1;
    /* CUDAGraph uses the same width-aware integer encoding as ordinary
     * launches. Keep eight bytes alive even for an int32 host binding. */
    node->scalar_values[scalar_idx] = (int64_t) * (int32_t *)args[i];
    node->kernel_params[i] = &node->scalar_values[scalar_idx];
  }

  node->params.func = (CUfunction)program->function;
  node->params.gridDimX =
      (unsigned int)(spec->value.program.grid[0] > 0 ? spec->value.program.grid[0] : 1);
  node->params.gridDimY =
      (unsigned int)(spec->value.program.grid[1] > 0 ? spec->value.program.grid[1] : 1);
  node->params.gridDimZ =
      (unsigned int)(spec->value.program.grid[2] > 0 ? spec->value.program.grid[2] : 1);
  node->params.blockDimX =
      (unsigned int)(spec->value.program.block[0] > 0 ? spec->value.program.block[0] : 1);
  node->params.blockDimY =
      (unsigned int)(spec->value.program.block[1] > 0 ? spec->value.program.block[1] : 1);
  node->params.blockDimZ =
      (unsigned int)(spec->value.program.block[2] > 0 ? spec->value.program.block[2] : 1);
  node->params.sharedMemBytes = 0;
  node->params.kernelParams = node->kernel_params;
  node->params.extra = NULL;
  return 0;
}

void poly_cuda_graph_destroy(PolyCudaGraph *graph) {
  if (!graph) return;
  if (graph->instance && cuda_api.cuGraphExecDestroy) cuda_api.cuGraphExecDestroy(graph->instance);
  if (graph->graph && cuda_api.cuGraphDestroy) cuda_api.cuGraphDestroy(graph->graph);
  for (int i = 0; i < graph->n_nodes; i++) {
    free(graph->nodes[i].buffer_values);
    free(graph->nodes[i].scalar_values);
    free(graph->nodes[i].kernel_params);
  }
  free(graph->nodes);
  free(graph);
}

PolyCudaGraph *poly_cuda_graph_create(const PolyCudaGraphCallSpec *specs, int n_specs) {
  if (!specs || n_specs <= 0 || !poly_cuda_graph_available()) return NULL;
  PolyCudaGraph *graph = calloc(1, sizeof(*graph));
  if (!graph) return NULL;
  graph->n_nodes = n_specs;
  graph->nodes = calloc((size_t)n_specs, sizeof(*graph->nodes));
  if (!graph->nodes) goto fail;

  CUresult err = cuda_api.cuGraphCreate(&graph->graph, 0);
  if (err != CUDA_SUCCESS) {
    fprintf(stderr, "polygrad: cuda: cuGraphCreate failed (CUresult=%d)\n", err);
    goto fail;
  }

  for (int i = 0; i < n_specs; i++) {
    const PolyCudaGraphCallSpec *spec = &specs[i];
    PolyCudaGraphNode *node = &graph->nodes[i];
    if (spec->n_dependencies < 0) goto fail;
    node->kind = spec->kind;
    if (spec->kind == POLY_CUDA_GRAPH_PROGRAM) {
      int n_buffer_args = spec->value.program.n_buffer_args;
      int n_args = spec->value.program.n_args;
      if (!spec->value.program.program || !spec->value.program.args || n_buffer_args < 0 ||
          n_args < n_buffer_args)
        goto fail;
      node->n_buffer_args = n_buffer_args;
      node->n_args = n_args;
      int n_scalars = n_args - n_buffer_args;
      node->buffer_values =
          calloc((size_t)(n_buffer_args > 0 ? n_buffer_args : 1), sizeof(*node->buffer_values));
      node->scalar_values =
          calloc((size_t)(n_scalars > 0 ? n_scalars : 1), sizeof(*node->scalar_values));
      node->kernel_params = calloc((size_t)(n_args > 0 ? n_args : 1), sizeof(*node->kernel_params));
      if (!node->buffer_values || !node->scalar_values || !node->kernel_params) goto fail;
    } else if (spec->kind != POLY_CUDA_GRAPH_COPY) {
      goto fail;
    }
    if (poly_cuda_graph_node_update(node, spec) != 0) goto fail;

    CUgraphNode deps_inline[32];
    CUgraphNode *deps = deps_inline;
    if (spec->n_dependencies > (int)(sizeof(deps_inline) / sizeof(deps_inline[0]))) {
      deps = malloc((size_t)spec->n_dependencies * sizeof(*deps));
      if (!deps) goto fail;
    }
    bool deps_ok = true;
    for (int d = 0; d < spec->n_dependencies; d++) {
      int dep = spec->dependencies ? spec->dependencies[d] : -1;
      if (dep < 0 || dep >= i || !graph->nodes[dep].node) {
        deps_ok = false;
        break;
      }
      deps[d] = graph->nodes[dep].node;
    }
    if (!deps_ok) {
      if (deps != deps_inline) free(deps);
      goto fail;
    }
    err = spec->kind == POLY_CUDA_GRAPH_PROGRAM
              ? cuda_api.cuGraphAddKernelNode(
                    &node->node, graph->graph, spec->n_dependencies ? deps : NULL,
                    (size_t)spec->n_dependencies, &node->params
                )
              : cuda_api.cuGraphAddMemcpyNode(
                    &node->node, graph->graph, spec->n_dependencies ? deps : NULL,
                    (size_t)spec->n_dependencies, &node->copy, cuda_ctx
                );
    if (deps != deps_inline) free(deps);
    if (err != CUDA_SUCCESS) {
      fprintf(stderr, "polygrad: cuda: cuGraphAddKernelNode failed (CUresult=%d)\n", err);
      goto fail;
    }
  }

  err = cuda_api.cuGraphInstantiate_v2(&graph->instance, graph->graph, NULL, NULL, 0);
  if (err != CUDA_SUCCESS) {
    fprintf(stderr, "polygrad: cuda: cuGraphInstantiate_v2 failed (CUresult=%d)\n", err);
    goto fail;
  }
  return graph;

fail:
  poly_cuda_graph_destroy(graph);
  return NULL;
}

int poly_cuda_graph_update(PolyCudaGraph *graph, const PolyCudaGraphCallSpec *specs, int n_specs) {
  if (!graph || !graph->instance || !specs || n_specs != graph->n_nodes) return -1;
  for (int i = 0; i < n_specs; i++) {
    PolyCudaGraphNode *node = &graph->nodes[i];
    if (poly_cuda_graph_node_update(node, &specs[i]) != 0) return -1;
    CUresult err =
        specs[i].kind == POLY_CUDA_GRAPH_PROGRAM
            ? cuda_api.cuGraphExecKernelNodeSetParams(graph->instance, node->node, &node->params)
            : cuda_api.cuGraphExecMemcpyNodeSetParams(
                  graph->instance, node->node, &node->copy, cuda_ctx
              );
    if (err != CUDA_SUCCESS) {
      fprintf(stderr, "polygrad: cuda: cuGraphExecKernelNodeSetParams failed (CUresult=%d)\n", err);
      return -1;
    }
  }
  return 0;
}

int poly_cuda_graph_launch(PolyCudaGraph *graph) {
  if (!graph || !graph->instance || !cuda_api.cuGraphLaunch) return -1;
  CUresult err = cuda_api.cuGraphLaunch(graph->instance, NULL);
  if (err != CUDA_SUCCESS) {
    fprintf(stderr, "polygrad: cuda: cuGraphLaunch failed (CUresult=%d)\n", err);
    return -1;
  }
  return 0;
}

int poly_cuda_memset(unsigned long long ptr, unsigned char val, size_t bytes) {
  if (cuda_state != CUDA_INIT_OK) return -1;
  if (ptr == 0 || bytes == 0) return 0;
  CUresult err = cuda_api.cuMemsetD8_v2((CUdeviceptr)ptr, val, bytes);
  if (err != CUDA_SUCCESS) {
    fprintf(stderr, "polygrad: cuda: cuMemsetD8_v2 failed (CUresult=%d)\n", err);
    return -1;
  }
  return 0;
}

#endif /* POLY_HAS_CUDA */
