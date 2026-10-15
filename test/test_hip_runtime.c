/* Isolated COMGR fault injection: include the owner to replace its private
 * driver table without exporting test hooks or requiring a GPU. */
#include "../src/runtime_hip.c"
#include <assert.h>

static int step, fail_step, live, invalid_release, loads, mode;
static bool handles[64];
static unsigned next_handle;
static int status(void) {
  return ++step == fail_step ? 1 : 0;
}
static int create(uint64_t *handle) {
  int rc = status();
  if (!rc) {
    *handle = ++next_handle;
    handles[*handle] = true;
    live++;
  }
  return rc;
}
static int release(uint64_t handle) {
  if (!handle || handle >= 64 || !handles[handle])
    invalid_release++;
  else {
    handles[handle] = false;
    live--;
  }
  return 0;
}
static int action_create(amd_comgr_action_info_t *x) {
  return create(&x->handle);
}
static int action_destroy(amd_comgr_action_info_t x) {
  return release(x.handle);
}
static int language(amd_comgr_action_info_t x, int v) {
  return status();
}
static int isa(amd_comgr_action_info_t x, const char *v) {
  return status();
}
static int logging(amd_comgr_action_info_t x, bool v) {
  return status();
}
static int options(amd_comgr_action_info_t x, const char *const *v, size_t n) {
  return status();
}
static int set_create(amd_comgr_data_set_t *x) {
  return create(&x->handle);
}
static int set_destroy(amd_comgr_data_set_t x) {
  return release(x.handle);
}
static int data_create(int kind, amd_comgr_data_t *x) {
  return create(&x->handle);
}
static int data_release(amd_comgr_data_t x) {
  return release(x.handle);
}
static int data_set(amd_comgr_data_t x, size_t n, const char *v) {
  return status();
}
static int data_name(amd_comgr_data_t x, const char *v) {
  return status();
}
static int data_add(amd_comgr_data_set_t x, amd_comgr_data_t y) {
  return status();
}
static int action(
    int kind,
    amd_comgr_action_info_t x,
    amd_comgr_data_set_t a,
    amd_comgr_data_set_t b
) {
  return status();
}
static int get(amd_comgr_data_set_t x, int kind, size_t idx, amd_comgr_data_t *data) {
  return create(&data->handle);
}
static int read_data(amd_comgr_data_t x, size_t *n, char *buf) {
  int rc = status();
  if (rc) return rc;
  if (!buf)
    *n = mode == 1 ? 0 : mode == 2 ? SIZE_MAX : 4;
  else if (mode == 3)
    *n = 5; /* Driver reports a changed size without writing past capacity. */
  else if (*n)
    memcpy(buf, "ELF!", 4);
  return 0;
}
static int load(hipModule_t *m, const void *p) {
  loads++;
  *m = (void *)1;
  return 0;
}
static int function(hipFunction_t *f, hipModule_t m, const char *n) {
  *f = (void *)2;
  return 0;
}
static int unload(hipModule_t m) {
  return 0;
}

int main(void) {
  hip_state = HIP_INIT_OK;
  hip_api.create_action_info = action_create;
  hip_api.destroy_action_info = action_destroy;
  hip_api.set_language = language;
  hip_api.set_isa_name = isa;
  hip_api.set_logging = logging;
  hip_api.set_option_list = options;
  hip_api.create_data_set = set_create;
  hip_api.destroy_data_set = set_destroy;
  hip_api.create_data = data_create;
  hip_api.release_data = data_release;
  hip_api.set_data = data_set;
  hip_api.set_data_name = data_name;
  hip_api.data_set_add = data_add;
  hip_api.do_action = action;
  hip_api.action_data_get_data = get;
  hip_api.get_data = read_data;
  hip_api.hipModuleLoadData = load;
  hip_api.hipModuleGetFunction = function;
  hip_api.hipModuleUnload = unload;
  PolyHipProgram *p = poly_compile_hip("source", "kernel");
  assert(p && loads == 1 && live == 0 && invalid_release == 0);
  poly_hip_program_destroy(p);
  int calls = step, failures = 0;
  for (int i = 1; i <= calls + 3; i++) {
    step = live = invalid_release = loads = 0;
    next_handle = 0;
    memset(handles, 0, sizeof(handles));
    fail_step = i <= calls ? i : 0;
    mode = i <= calls ? 0 : i - calls;
    p = poly_compile_hip("source", "kernel");
    bool ok = !p && !loads && !live && !invalid_release;
    poly_hip_program_destroy(p);
    if (!ok) {
      fprintf(
          stderr, "FAIL injection=%d mode=%d loads=%d live=%d invalid=%d\n", fail_step, mode, loads,
          live, invalid_release
      );
      failures++;
    }
  }
  printf("COMGR cases: %d passed, %d failed\n", calls + 4 - failures, failures);
  return failures != 0;
}
