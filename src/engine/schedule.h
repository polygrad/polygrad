/*
 * schedule.h -- Tinygrad-aligned engine scheduling and compiled execution.
 *
 * Mirrors the role split in tinygrad's engine/schedule.py:
 *   kernel graph creation happens in schedule/rangeify.c
 *   backend-neutral schedule construction happens here
 *   backend-specific lowering and schedule execution happen here
 */

#ifndef POLY_ENGINE_SCHEDULE_H
#define POLY_ENGINE_SCHEDULE_H

#include "polygrad.h"
#include "device.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PolyVarBinding {
  PolyUOp *var; /* DEFINE_VAR UOp */
  int32_t value; /* concrete runtime value */
} PolyVarBinding;

/* Compilation mode mirrors the entrypoint intent used by tinygrad call sites. */
typedef enum {
  POLY_MODE_CALL = 0,
  POLY_MODE_VALUE_AND_GRAD,
} PolyCompileMode;

typedef enum {
  POLY_RUNNER_COMPILED,
  POLY_RUNNER_COPY,
  POLY_RUNNER_VIEW,
  POLY_RUNNER_ENCDEC,
  POLY_RUNNER_INTERP,
} PolyRunnerKind;

typedef struct {
  PolyRunnerKind kind;

  void *handle;
  int handle_size;

  int (*execute)(void *self, void **args, int n_args);
  void (*free_handle)(void *self);
  bool borrowed_handle;

  int grid[3];
  int block[3];
  PolyUOp *grid_exprs[3];
  PolyUOp *block_exprs[3];

  int *param_to_slot;
  int n_params;

  int *var_indices;
  int n_vars;
} PolyRunner;

typedef struct {
  PolyDType dtype;
  int64_t numel;
  int64_t nbytes;
  PolyUOp *buf_uop;
  PolyDevice device;
  bool is_intermediate;
  bool needs_zero;
  int external_buf_idx;
  bool is_memory_arena;
  bool has_memory_parent;
  int memory_parent_slot;
  int64_t memory_offset;
  bool has_zero_before_call;
  int zero_before_call;
} PolyScheduleBufSlot;

typedef struct {
  int n_args;
  bool *outs;
  bool *ins;
  int *read_args;
  int n_read_args;
  int *write_args;
  int n_write_args;
  int *active_args;
  int n_active_args;
} PolyCallAccess;

typedef struct {
  int n_args;
  int *arg_to_slot;
  const PolyCallAccess *access;
} PolyCallIO;

struct PolyProgramInfo {
  const char *name;

  int global_size[3];
  int local_size[3];
  PolyUOp *global_exprs[3];
  PolyUOp *local_exprs[3];
  bool has_local_size;

  PolyUOp **vars;
  int n_vars;

  int *globals;
  int n_globals;

  int *outs;
  int n_outs;

  int *ins;
  int n_ins;
};

typedef struct PolyRuntimeCacheEntry PolyRuntimeCacheEntry;
typedef struct PolyScheduleBlueprint PolyScheduleBlueprint;

typedef struct {
  PolyUOp *call; /* LINEAR source: CALL(body, buffer args..., DEFINE_VAR args...) */

  /* Runtime cache parallel to LINEAR.src[]. Mutable backend state stays here,
   * never in the CALL UOp itself. */
  PolyRunner prg;
  PolyRuntimeCacheEntry *runtime_program;
  PolyDevice lowered_device;
  uint32_t lowered_env_stamp;
  bool prg_valid;
} PolyCallRuntime;

typedef struct {
  int refcount;

  const char *entrypoint_name;
  PolyCompileMode mode;
  uint32_t graph_hash;

  PolyScheduleBufSlot *buf_slots;
  int n_buf_slots;

  /* Optional retained source blueprint. This is a ctx-cache object containing
   * parameterized LINEAR/static CALL metadata; this concrete template owns
   * resolved slots and fresh intermediates. */
  PolyScheduleBlueprint *blueprint;

  PolyUOp *linear; /* POLY_OP_LINEAR; srcs are ordered POLY_OP_CALL UOps */
  PolyCallAccess *call_access;
  int n_calls;

  PolyVarBinding *default_vars;
  int n_default_vars;

  int loss_buf_slot;
  int *grad_buf_slots;
} PolyScheduleTemplate;

typedef struct {
  PolyCallRuntime *calls;
  PolyCallIO *call_io;
  PolyDevice device;
  const PolyAllocator *allocator;
  PolyBuffer *intermediates;
  int n_intermediates;
  void ***kernel_args;
  void **slot_to_data;
  int n_slot_to_data;
  PolyBuffer *slot_views;
  int n_slot_views;
  PolyVarBinding *merged_vars;
  int merged_vars_cap;
  int *var_int_storage;
  int var_int_cap;
} PolyScheduleRuntime;

typedef struct {
  /* LINEAR/CALL schedule metadata. This is structurally immutable after
   * schedule construction and is separated from mutable runtime state. */
  PolyScheduleTemplate *template;

  /* Direct run_schedule runtime state. Mutable backend runners, intermediates,
   * argument arrays, and merged variables live here, not in the template. */
  PolyScheduleRuntime *run;
} PolySchedule;

typedef struct {
  PolyCtx *ctx;
  PolyScheduleTemplate *template;
  PolyDevice device;
  const PolyAllocator *allocator;

  /* Compiled/replay runtime workspace. This has the same shape as direct
   * schedule execution; the compiled plan owns the workspace and borrows the
   * schedule template. */
  PolyScheduleRuntime *run;
} PolyCompiledSchedule;

typedef struct {
  const char *name;
  PolyDevice id;
  bool host_executed;

  /* Lower a PROGRAM-rooted scheduled compute call body. Backends may unwrap the
   * PROGRAM to a renderer-specific kernel body internally, but the vtable
   * boundary follows tinygrad's runtime cache boundary. */
  PolyUOp *(*rewrite_program)(PolyCtx *ctx, PolyUOp *sink);
  char *(*render_source)(PolyCtx *ctx, PolyUOp *program, const char *fn_name);
  int (*lower_item)(PolyCtx *ctx, PolyUOp *program, const char *fn_name, PolyRunner *runner_out);
  int (*execute)(PolyRunner *runner, void **args, int n_args);
  void (*free_runner)(PolyRunner *runner);
  int (*ensure_open)(void);
  const PolyAllocator *(*get_allocator)(void);
} PolyBackendDesc;

typedef struct {
  int code;
  char message[256];
  int failed_item_index;
} PolyExecStatus;

/* Tinygrad engine/schedule.py analogue: schedule construction from a tensor sink. */
PolySchedule *poly_complete_create_schedule_with_vars(PolyCtx *ctx, PolyUOp *sink, PolyCompileMode mode);

/* Tinygrad engine/schedule.py analogue: create a schedule from a kernel graph. */
PolySchedule *poly_create_schedule(PolyCtx *ctx, PolyUOp *kernel_graph);

/* Tinygrad engine/schedule.py analogue: lower a sink into LINEAR form. */
PolyUOp *poly_lower_sink_to_linear(PolyCtx *ctx, PolyUOp *sink, PolyCompileMode mode);

size_t poly_schedule_cache_len(PolyCtx *ctx);
void poly_schedule_cache_clear(PolyCtx *ctx);
/* Tinygrad runtime_cache analogue: count/clear backend runtime handles cached
 * below PROGRAM UOps. Clearing evicts future lookups; refcounted live
 * schedules/plans remain usable until they release their cached runners.
 * poly_program_cache_* names are compatibility aliases. */
size_t poly_runtime_cache_len(PolyCtx *ctx);
void poly_runtime_cache_clear(PolyCtx *ctx);
/* Includes live refcounted runtime payloads retained by schedules/plans plus
 * cache-map wrapper bytes for entries still available for future lookup. */
size_t poly_runtime_cache_artifact_bytes(PolyCtx *ctx);
size_t poly_program_cache_len(PolyCtx *ctx);
void poly_program_cache_clear(PolyCtx *ctx);
size_t poly_program_cache_artifact_bytes(PolyCtx *ctx);
size_t poly_to_program_cache_len(PolyCtx *ctx);
void poly_to_program_cache_clear(PolyCtx *ctx);
int poly_program_source_render_count(void);
void poly_program_source_render_count_reset(void);

void poly_schedule_free(PolySchedule *schedule);
size_t poly_schedule_runtime_intermediate_bytes(const PolySchedule *schedule);

PolyUOp *poly_schedule_call(const PolySchedule *schedule, int call_index);
PolyUOp *poly_schedule_call_body(const PolySchedule *schedule, int call_index);
bool poly_schedule_call_is_copy(const PolySchedule *schedule, int call_index);
int poly_schedule_call_n_buffer_args(const PolySchedule *schedule, int call_index);
int poly_schedule_call_buffer_slot(const PolySchedule *schedule, int call_index, int arg_index);

/* Tinygrad ProgramInfo analogue for PROGRAM CALL bodies. Indices are over
 * filtered CALL buffer arguments, excluding DEFINE_VAR/BIND-like arguments. */
PolyUOp *poly_program_from_call(PolyCtx *ctx, PolyUOp *call, const char *name);
const PolyProgramInfo *poly_program_info(PolyCtx *ctx, PolyUOp *program);
/* Return PROGRAM's cached POLY_OP_LINEAR child when present. This mirrors
 * tinygrad's PROGRAM(SINK, DEVICE, LINEAR, SOURCE, BINARY...) staging.
 * Polygrad currently stores SOURCE as a PROGRAM child and keeps compiled
 * BINARY/runtime handles in backend caches. */
PolyUOp *poly_program_linear(PolyUOp *program);
PolyUOp *poly_schedule_call_to_program(
    PolyCtx *ctx,
    PolySchedule *schedule,
    int call_index,
    PolyDevice device
);

int poly_schedule_call_lower(
    PolyCtx *ctx,
    PolySchedule *schedule,
    int call_index,
    PolyDevice device
);
int poly_schedule_call_run(
    PolyCtx *ctx,
    PolySchedule *schedule,
    int call_index,
    PolyVarBinding *var_bindings,
    int n_var_bindings
);

/* Backend lowering from backend-neutral schedule to compiled runners. */
PolyCompiledSchedule *poly_lower_schedule(PolyCtx *ctx, PolySchedule *schedule, PolyDevice device);

/* Internal execution-device selection shared by direct schedule execution and
 * cached entrypoint replay. Preferred executable device wins, otherwise slot
 * annotations/current non-host residencies are used, then the default device. */
PolyDevice poly_schedule_infer_device(PolyCtx *ctx, const PolySchedule *schedule);

/* Tinygrad engine/realize.py analogue: lower and run a backend-neutral
 * schedule using ctx->buffers as the runtime data source. */
int poly_run_schedule(
    PolyCtx *ctx,
    PolySchedule *schedule,
    PolyVarBinding *var_bindings,
    int n_var_bindings
);

/* Low-level runner for pre-lowered compiled schedules. */
/* Low-level compiled schedule replay helper.
 *
 * This is an internal/runtime API, not the public Instance execution surface.
 * Passing slot_data preserves the old raw-slot compatibility path used by
 * backend tests. Passing NULL or omitting a slot resolves external buffers
 * through ctx->buffers and still performs the normal prepare/execute/commit
 * residency protocol.
 */
int poly_run_compiled_schedule(
    PolyCompiledSchedule *schedule,
    void **slot_data,
    int n_slots,
    PolyVarBinding *var_bindings,
    int n_var_bindings
);

void poly_compiled_schedule_free(PolyCompiledSchedule *schedule);
size_t poly_compiled_schedule_runtime_intermediate_bytes(const PolyCompiledSchedule *schedule);
void poly_sched_cache_flush(void);

extern const PolyAllocator POLY_CPU_ALLOCATOR;

#ifdef POLY_HAS_CUDA
extern const PolyAllocator POLY_CUDA_ALLOCATOR;
#endif

#ifdef POLY_HAS_HIP
extern const PolyAllocator POLY_HIP_ALLOCATOR;
#endif

const PolyBackendDesc *poly_backend_get(PolyDevice device);
int poly_backend_ensure_open(PolyDevice device);
bool poly_device_is_host_addressable(PolyDevice device);

#ifdef __cplusplus
}
#endif

#endif /* POLY_ENGINE_SCHEDULE_H */
