/*
 * runtime_hip.c -- HIP/ROCm runtime via dlopen (no link-time dependencies)
 *
 * Lazy-loads libamdhip64.so and libamd_comgr.so at runtime, resolves all
 * symbols via dlsym. Compilation uses AMD comgr (same as tinygrad) with
 * -nogpuinc so the renderer controls all device function declarations.
 *
 * Pipeline: HIP C++ source -> comgr BC -> relocatable -> executable (HSACO)
 *           -> hipModuleLoadData -> hipModuleGetFunction -> hipModuleLaunchKernel
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
typedef void *hipModule_t;
typedef void *hipFunction_t;

#define hipSuccess 0
#define hipMemcpyHostToDevice 1
#define hipMemcpyDeviceToHost 2

/* hipDeviceProp_tR0600 layout from tinygrad autogen (ROCm 6.x):
 *   SIZE = 1472, warpSize @ 308, gcnArchName @ 1160 (char[256]) */
#define HIP_DEVICE_PROP_SIZE 1472
#define HIP_PROP_WARP_SIZE_OFFSET 308
#define HIP_PROP_GCN_ARCH_NAME_OFFSET 1160
#define HIP_PROP_GCN_ARCH_NAME_LEN 256

/* ── comgr types and constants (from tinygrad autogen) ─────────────── */
/* All handles are opaque uint64. Enum values match ROCm 6.x headers. */

typedef int amd_comgr_status_t;
typedef struct { uint64_t handle; } amd_comgr_data_t;
typedef struct { uint64_t handle; } amd_comgr_data_set_t;
typedef struct { uint64_t handle; } amd_comgr_action_info_t;

#define AMD_COMGR_STATUS_SUCCESS 0
/* Data kind enums (same across comgr 2 and 3) */
#define AMD_COMGR_DATA_KIND_SOURCE 1
#define AMD_COMGR_DATA_KIND_LOG 5
#define AMD_COMGR_DATA_KIND_EXECUTABLE 8
/* Action/language enums differ between comgr 2 and 3.
 * Detected at runtime via amd_comgr_get_version. */
static int COMGR_LANGUAGE_HIP;
static int COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC;
static int COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE;
static int COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE;

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

/* comgr */
typedef amd_comgr_status_t (*comgr_create_action_info_fn)(amd_comgr_action_info_t *);
typedef amd_comgr_status_t (*comgr_destroy_action_info_fn)(amd_comgr_action_info_t);
typedef amd_comgr_status_t (*comgr_set_language_fn)(amd_comgr_action_info_t, int);
typedef amd_comgr_status_t (*comgr_set_isa_name_fn)(amd_comgr_action_info_t, const char *);
typedef amd_comgr_status_t (*comgr_set_logging_fn)(amd_comgr_action_info_t, bool);
typedef amd_comgr_status_t (*comgr_set_option_list_fn)(amd_comgr_action_info_t, const char *const *, size_t);
typedef amd_comgr_status_t (*comgr_create_data_set_fn)(amd_comgr_data_set_t *);
typedef amd_comgr_status_t (*comgr_destroy_data_set_fn)(amd_comgr_data_set_t);
typedef amd_comgr_status_t (*comgr_create_data_fn)(int kind, amd_comgr_data_t *);
typedef amd_comgr_status_t (*comgr_release_data_fn)(amd_comgr_data_t);
typedef amd_comgr_status_t (*comgr_set_data_fn)(amd_comgr_data_t, size_t, const char *);
typedef amd_comgr_status_t (*comgr_set_data_name_fn)(amd_comgr_data_t, const char *);
typedef amd_comgr_status_t (*comgr_get_data_fn)(amd_comgr_data_t, size_t *, char *);
typedef amd_comgr_status_t (*comgr_data_set_add_fn)(amd_comgr_data_set_t, amd_comgr_data_t);
typedef amd_comgr_status_t (*comgr_do_action_fn)(int action, amd_comgr_action_info_t,
                                                  amd_comgr_data_set_t, amd_comgr_data_set_t);
typedef amd_comgr_status_t (*comgr_action_data_get_data_fn)(amd_comgr_data_set_t, int kind,
                                                             size_t index, amd_comgr_data_t *);
typedef amd_comgr_status_t (*comgr_get_version_fn)(uint64_t *, uint64_t *);

/* ── Loaded symbols ────────────────────────────────────────────────── */

static struct {
  void *libhip;
  void *libcomgr;

  /* HIP runtime */
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

  /* comgr */
  comgr_create_action_info_fn create_action_info;
  comgr_destroy_action_info_fn destroy_action_info;
  comgr_set_language_fn set_language;
  comgr_set_isa_name_fn set_isa_name;
  comgr_set_logging_fn set_logging;
  comgr_set_option_list_fn set_option_list;
  comgr_create_data_set_fn create_data_set;
  comgr_destroy_data_set_fn destroy_data_set;
  comgr_create_data_fn create_data;
  comgr_release_data_fn release_data;
  comgr_set_data_fn set_data;
  comgr_set_data_name_fn set_data_name;
  comgr_get_data_fn get_data;
  comgr_data_set_add_fn data_set_add;
  comgr_do_action_fn do_action;
  comgr_action_data_get_data_fn action_data_get_data;
} hip_api = {0};

/* ── Lazy singleton state ──────────────────────────────────────────── */

static enum { HIP_NOT_TRIED, HIP_INIT_OK, HIP_INIT_FAIL } hip_state = HIP_NOT_TRIED;
static char hip_arch_name[256] = {0};
static char hip_isa_name[320] = {0}; /* "amdgcn-amd-amdhsa--gfx90a" */
static int hip_warp_size = 64;

/* ── dlsym helper ──────────────────────────────────────────────────── */

static void *hip_load_sym(void *lib, const char *name) {
  void *sym = dlsym(lib, name);
  if (!sym)
    fprintf(stderr, "polygrad: hip: dlsym(%s) failed: %s\n", name, dlerror());
  return sym;
}

/* ── Load libraries + resolve all symbols ──────────────────────────── */

static bool load_hip_libs(void) {
  hip_api.libhip = dlopen("libamdhip64.so.6", RTLD_LAZY);
  if (!hip_api.libhip)
    hip_api.libhip = dlopen("libamdhip64.so", RTLD_LAZY);
  if (!hip_api.libhip) {
    fprintf(stderr, "polygrad: hip: cannot load libamdhip64.so: %s\n", dlerror());
    return false;
  }

  hip_api.libcomgr = dlopen("libamd_comgr.so.2", RTLD_LAZY);
  if (!hip_api.libcomgr)
    hip_api.libcomgr = dlopen("libamd_comgr.so.3", RTLD_LAZY);
  if (!hip_api.libcomgr)
    hip_api.libcomgr = dlopen("libamd_comgr.so", RTLD_LAZY);
  if (!hip_api.libcomgr) {
    fprintf(stderr, "polygrad: hip: cannot load libamd_comgr.so: %s\n", dlerror());
    dlclose(hip_api.libhip);
    hip_api.libhip = NULL;
    return false;
  }

  /* HIP runtime symbols */
#define LOAD_HIP(name) do { \
    *(void **)&hip_api.name = hip_load_sym(hip_api.libhip, #name); \
    if (!hip_api.name) return false; \
  } while (0)

  LOAD_HIP(hipInit);
  LOAD_HIP(hipGetDeviceCount);
  LOAD_HIP(hipSetDevice);
  /* Versioned API first (ROCm 6.x), then unversioned */
  *(void **)&hip_api.hipGetDeviceProperties =
    hip_load_sym(hip_api.libhip, "hipGetDevicePropertiesR0600");
  if (!hip_api.hipGetDeviceProperties)
    *(void **)&hip_api.hipGetDeviceProperties =
      hip_load_sym(hip_api.libhip, "hipGetDeviceProperties");
  if (!hip_api.hipGetDeviceProperties) return false;
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

  /* comgr symbols */
#define LOAD_COMGR(field, sym) do { \
    *(void **)&hip_api.field = hip_load_sym(hip_api.libcomgr, sym); \
    if (!hip_api.field) return false; \
  } while (0)

  LOAD_COMGR(create_action_info,    "amd_comgr_create_action_info");
  LOAD_COMGR(destroy_action_info,   "amd_comgr_destroy_action_info");
  LOAD_COMGR(set_language,          "amd_comgr_action_info_set_language");
  LOAD_COMGR(set_isa_name,         "amd_comgr_action_info_set_isa_name");
  LOAD_COMGR(set_logging,          "amd_comgr_action_info_set_logging");
  LOAD_COMGR(set_option_list,      "amd_comgr_action_info_set_option_list");
  LOAD_COMGR(create_data_set,      "amd_comgr_create_data_set");
  LOAD_COMGR(destroy_data_set,     "amd_comgr_destroy_data_set");
  LOAD_COMGR(create_data,          "amd_comgr_create_data");
  LOAD_COMGR(release_data,         "amd_comgr_release_data");
  LOAD_COMGR(set_data,             "amd_comgr_set_data");
  LOAD_COMGR(set_data_name,        "amd_comgr_set_data_name");
  LOAD_COMGR(get_data,             "amd_comgr_get_data");
  LOAD_COMGR(data_set_add,         "amd_comgr_data_set_add");
  LOAD_COMGR(do_action,            "amd_comgr_do_action");
  LOAD_COMGR(action_data_get_data, "amd_comgr_action_data_get_data");
#undef LOAD_COMGR

  /* Detect comgr version and set enum values accordingly.
   * comgr 3 renumbered several enums: https://github.com/ROCm/llvm-project/issues/272 */
  {
    comgr_get_version_fn get_ver = (comgr_get_version_fn)hip_load_sym(
      hip_api.libcomgr, "amd_comgr_get_version");
    uint64_t major = 2, minor = 0;
    if (get_ver) get_ver(&major, &minor);

    if (major >= 3) {
      /* comgr 3.x enum values */
      COMGR_LANGUAGE_HIP = 3;
      COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC = 12;
      COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE = 4;
      COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE = 7;
    } else {
      /* comgr 2.x enum values */
      COMGR_LANGUAGE_HIP = 4;
      COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC = 15;
      COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE = 6;
      COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE = 9;
    }
  }

  return true;
}

/* ── comgr helpers ─────────────────────────────────────────────────── */

/* Extract result data from a comgr data set. Returns malloc'd bytes. */
static char *comgr_get_data_bytes(amd_comgr_data_set_t ds, int kind, size_t *out_size) {
  amd_comgr_data_t data = {0};
  if (hip_api.action_data_get_data(ds, kind, 0, &data) != AMD_COMGR_STATUS_SUCCESS)
    return NULL;
  size_t sz = 0;
  hip_api.get_data(data, &sz, NULL);
  char *buf = malloc(sz);
  if (buf) hip_api.get_data(data, &sz, buf);
  hip_api.release_data(data);
  if (out_size) *out_size = sz;
  return buf;
}

/* ── Public API ────────────────────────────────────────────────────── */

int poly_hip_init(void) {
  if (hip_state == HIP_INIT_OK) return 0;
  if (hip_state == HIP_INIT_FAIL) return -1;
  hip_state = HIP_INIT_FAIL;

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

  /* Query device properties via raw byte buffer with known offsets
   * (from tinygrad autogen for ROCm 6.x hipDeviceProp_tR0600). */
  char props[HIP_DEVICE_PROP_SIZE];
  memset(props, 0, sizeof(props));
  err = hip_api.hipGetDeviceProperties(props, 0);
  if (err != hipSuccess) {
    fprintf(stderr, "polygrad: hip: hipGetDeviceProperties failed (hipError_t=%d)\n", err);
    return -1;
  }

  int warp_size_val;
  memcpy(&warp_size_val, props + HIP_PROP_WARP_SIZE_OFFSET, sizeof(int));
  hip_warp_size = warp_size_val > 0 ? warp_size_val : 64;

  memcpy(hip_arch_name, props + HIP_PROP_GCN_ARCH_NAME_OFFSET, HIP_PROP_GCN_ARCH_NAME_LEN);
  hip_arch_name[HIP_PROP_GCN_ARCH_NAME_LEN - 1] = '\0';
  char *colon = strchr(hip_arch_name, ':');
  if (colon) *colon = '\0';

  /* Build ISA name for comgr: "amdgcn-amd-amdhsa--gfx90a" */
  snprintf(hip_isa_name, sizeof(hip_isa_name), "amdgcn-amd-amdhsa--%s", hip_arch_name);

  hip_state = HIP_INIT_OK;
  return 0;
}

bool poly_hip_available(void) {
  if (hip_state == HIP_NOT_TRIED) poly_hip_init();
  return hip_state == HIP_INIT_OK;
}

int poly_hip_wave_size(void) { return hip_warp_size; }
const char *poly_hip_arch(void) { return hip_arch_name; }

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
  if (hip_state != HIP_INIT_OK || !ptr) return;
  hipError_t err = hip_api.hipFree(ptr);
  if (err != hipSuccess)
    fprintf(stderr, "polygrad: hip: hipFree failed (hipError_t=%d)\n", err);
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

/* ── Compilation via comgr (matches tinygrad's compile_hip) ────────── */

PolyHipProgram *poly_compile_hip(const char *source, const char *fn_name) {
  if (hip_state != HIP_INIT_OK) {
    fprintf(stderr, "polygrad: hip: compile called but HIP not initialized\n");
    return NULL;
  }

  amd_comgr_status_t st;
  amd_comgr_action_info_t action_info = {0};
  amd_comgr_data_set_t ds_src = {0}, ds_bc = {0}, ds_reloc = {0}, ds_exec = {0};
  amd_comgr_data_t data_src = {0};
  char *hsaco = NULL;
  size_t hsaco_size = 0;

  /* Create action info */
  st = hip_api.create_action_info(&action_info);
  if (st != AMD_COMGR_STATUS_SUCCESS) goto fail_early;
  hip_api.set_language(action_info, COMGR_LANGUAGE_HIP);
  hip_api.set_isa_name(action_info, hip_isa_name);
  hip_api.set_logging(action_info, true);

  /* Create data sets */
  hip_api.create_data_set(&ds_src);
  hip_api.create_data_set(&ds_bc);
  hip_api.create_data_set(&ds_reloc);
  hip_api.create_data_set(&ds_exec);

  /* Add source */
  hip_api.create_data(AMD_COMGR_DATA_KIND_SOURCE, &data_src);
  hip_api.set_data(data_src, strlen(source), source);
  hip_api.set_data_name(data_src, "<null>");
  hip_api.data_set_add(ds_src, data_src);

  /* Step 1: source -> BC (with device libs, matching tinygrad options) */
  {
    const char *opts[] = {
      "-O3", "-mcumode",
      "--hip-version=6.0.32830",
      "-DHIP_VERSION_MAJOR=6", "-DHIP_VERSION_MINOR=0", "-DHIP_VERSION_PATCH=32830",
      "-D__HIPCC_RTC__", "-std=c++14", "-nogpuinc",
      "-Wno-gnu-line-marker", "-Wno-missing-prototypes",
    };
    int n_opts = (int)(sizeof(opts) / sizeof(opts[0]));

    /* Append --offload-arch=<arch> */
    char arch_opt[320];
    snprintf(arch_opt, sizeof(arch_opt), "--offload-arch=%s", hip_arch_name);
    const char *all_opts[16];
    for (int i = 0; i < n_opts && i < 15; i++) all_opts[i] = opts[i];
    all_opts[n_opts] = arch_opt;
    n_opts++;

    hip_api.set_option_list(action_info, all_opts, n_opts);
    st = hip_api.do_action(COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC,
                            action_info, ds_src, ds_bc);
    if (st != AMD_COMGR_STATUS_SUCCESS) {
      size_t log_size;
      char *log = comgr_get_data_bytes(ds_bc, AMD_COMGR_DATA_KIND_LOG, &log_size);
      fprintf(stderr, "polygrad: hip: comgr compile failed (status=%d):\n%s\n",
              st, log ? log : "(no log)");
      free(log);
      goto fail;
    }
  }

  /* Step 2: BC -> relocatable */
  {
    const char *opts[] = { "-O3", "-mllvm", "-amdgpu-internalize-symbols" };
    hip_api.set_option_list(action_info, opts, 3);
    st = hip_api.do_action(COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE,
                            action_info, ds_bc, ds_reloc);
    if (st != AMD_COMGR_STATUS_SUCCESS) {
      fprintf(stderr, "polygrad: hip: comgr codegen failed (status=%d)\n", st);
      goto fail;
    }
  }

  /* Step 3: relocatable -> executable (HSACO) */
  {
    const char *no_opts[] = { NULL };
    hip_api.set_option_list(action_info, no_opts, 0);
    st = hip_api.do_action(COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE,
                            action_info, ds_reloc, ds_exec);
    if (st != AMD_COMGR_STATUS_SUCCESS) {
      fprintf(stderr, "polygrad: hip: comgr link failed (status=%d)\n", st);
      goto fail;
    }
  }

  /* Extract HSACO */
  hsaco = comgr_get_data_bytes(ds_exec, AMD_COMGR_DATA_KIND_EXECUTABLE, &hsaco_size);
  if (!hsaco || hsaco_size == 0) {
    fprintf(stderr, "polygrad: hip: comgr produced empty executable\n");
    goto fail;
  }

  /* Cleanup comgr resources */
  hip_api.release_data(data_src);
  hip_api.destroy_data_set(ds_src);
  hip_api.destroy_data_set(ds_bc);
  hip_api.destroy_data_set(ds_reloc);
  hip_api.destroy_data_set(ds_exec);
  hip_api.destroy_action_info(action_info);

  /* Load HSACO into HIP module */
  hipModule_t module = NULL;
  hipError_t hip_err = hip_api.hipModuleLoadData(&module, hsaco);
  free(hsaco);
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

  PolyHipProgram *result = malloc(sizeof(PolyHipProgram));
  if (!result) { hip_api.hipModuleUnload(module); return NULL; }
  result->module = module;
  result->function = function;
  return result;

fail:
  hip_api.release_data(data_src);
  hip_api.destroy_data_set(ds_src);
  hip_api.destroy_data_set(ds_bc);
  hip_api.destroy_data_set(ds_reloc);
  hip_api.destroy_data_set(ds_exec);
  hip_api.destroy_action_info(action_info);
  return NULL;

fail_early:
  fprintf(stderr, "polygrad: hip: comgr_create_action_info failed (status=%d)\n", st);
  return NULL;
}

int poly_hip_launch(PolyHipProgram *prog, void **args, int n_args,
                    int gx, int gy, int gz, int bx, int by, int bz) {
  (void)n_args;
  if (!prog || hip_state != HIP_INIT_OK) return -1;

  hipError_t err = hip_api.hipModuleLaunchKernel(
    (hipFunction_t)prog->function,
    (unsigned int)gx, (unsigned int)gy, (unsigned int)gz,
    (unsigned int)bx, (unsigned int)by, (unsigned int)bz,
    0, NULL, args, NULL);
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
  if (prog->module && hip_state == HIP_INIT_OK)
    hip_api.hipModuleUnload((hipModule_t)prog->module);
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
