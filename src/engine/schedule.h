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
  POLY_EXEC_COMPUTE,
  POLY_EXEC_COPY,
  POLY_EXEC_VIEW,
  POLY_EXEC_ENCDEC,
} PolyExecItemKind;

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
  int external_buf_idx;
} PolyScheduleBufSlot;

typedef struct {
  PolyExecItemKind kind;
  PolyUOp *root; /* scheduled kernel root before backend lowering */

  int *buf_slot_indices;
  int n_buf_slots;

  PolyVarBinding *fixedvars;
  int n_fixedvars;

  PolyUOp **var_uops;
  int n_var_uops;

  /* Tinygrad ExecItem analogue for the direct run_schedule path:
   * cache the lowered runner on the item itself. */
  PolyRunner prg;
  PolyDevice lowered_device;
  uint32_t lowered_env_stamp;
  bool prg_valid;
} PolyExecItem;

typedef struct {
  const char *entrypoint_name;
  PolyCompileMode mode;
  uint32_t graph_hash;

  PolyScheduleBufSlot *buf_slots;
  int n_buf_slots;

  PolyExecItem *items;
  int n_items;

  int *exec_order;

  PolyVarBinding *default_vars;
  int n_default_vars;

  int loss_buf_slot;
  int *grad_buf_slots;

  /* Direct run_schedule runtime state. This moves the main execution path
   * closer to tinygrad's Schedule + ExecItem ownership while compiled-plan
   * wrappers remain as a compatibility layer during migration. */
  PolyDevice run_device;
  const PolyAllocator *run_allocator;
  PolyBuffer *run_intermediates;
  int n_run_intermediates;
  void ***run_kernel_args;
  void **run_slot_to_data;
  int n_run_slot_to_data;
  PolyVarBinding *run_merged_vars;
  int run_merged_vars_cap;
  int *run_var_int_storage;
  int run_var_int_cap;
} PolySchedule;

typedef struct {
  PolySchedule *schedule;
  PolyDevice device;
  const PolyAllocator *allocator;

  PolyRunner *runners;
  int n_runners;

  PolyBuffer *intermediates;
  int n_intermediates;

  void ***kernel_args;

  void **slot_to_data;
  int n_slot_to_data;

  PolyVarBinding *merged_vars;
  int merged_vars_cap;

  int *var_int_storage;
  int var_int_cap;
} PolyCompiledSchedule;

typedef struct {
  const char *name;
  PolyDevice id;
  bool host_executed;

  int (*lower_item)(PolyCtx *ctx, PolyUOp *scheduled_root, const char *fn_name, PolyRunner *runner_out);
  int (*execute)(PolyRunner *runner, void **args, int n_args);
  void (*free_runner)(PolyRunner *runner);
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
/* Tinygrad to_program/runtime-cache analogue: count/clear backend program
 * handles cached below scheduling. Clearing is intended for tests and must not
 * race an in-flight PolySchedule that is borrowing cached runners. */
size_t poly_program_cache_len(PolyCtx *ctx);
void poly_program_cache_clear(PolyCtx *ctx);

void poly_schedule_free(PolySchedule *schedule);

/* Tinygrad ExecItem analogues for the direct schedule runner. */
int poly_exec_item_lower(PolyCtx *ctx, PolySchedule *schedule, int item_index, PolyDevice device);
int poly_exec_item_run(
    PolyCtx *ctx,
    PolySchedule *schedule,
    int item_index,
    PolyVarBinding *var_bindings,
    int n_var_bindings
);

/* Backend lowering from backend-neutral schedule to compiled runners. */
PolyCompiledSchedule *poly_lower_schedule(PolyCtx *ctx, PolySchedule *schedule, PolyDevice device);

/* Tinygrad engine/realize.py analogue: lower and run a backend-neutral
 * schedule using ctx->buffers as the runtime data source. */
int poly_run_schedule(
    PolyCtx *ctx,
    PolySchedule *schedule,
    PolyVarBinding *var_bindings,
    int n_var_bindings
);

/* Low-level runner for pre-lowered compiled schedules. */
int poly_run_compiled_schedule(
    PolyCompiledSchedule *schedule,
    void **slot_data,
    int n_slots,
    PolyVarBinding *var_bindings,
    int n_var_bindings
);

void poly_compiled_schedule_free(PolyCompiledSchedule *schedule);
void poly_sched_cache_flush(void);

extern const PolyAllocator POLY_CPU_ALLOCATOR;

#ifdef POLY_HAS_CUDA
extern const PolyAllocator POLY_CUDA_ALLOCATOR;
#endif

#ifdef POLY_HAS_HIP
extern const PolyAllocator POLY_HIP_ALLOCATOR;
#endif

const PolyBackendDesc *poly_backend_get(PolyDevice device);
bool poly_device_is_host_addressable(PolyDevice device);

#ifdef __cplusplus
}
#endif

#endif /* POLY_ENGINE_SCHEDULE_H */
