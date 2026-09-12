/* engine/schedule.h -- C runtime mechanics for Tinygrad LINEAR execution. */

#ifndef POLY_ENGINE_SCHEDULE_H
#define POLY_ENGINE_SCHEDULE_H

#include "polygrad.h"
#include "device.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  POLY_RUNNER_COMPILED,
  POLY_RUNNER_INTERP,
} PolyRunnerKind;

/* C handle below Tinygrad's PROGRAM/runtime_cache boundary. */
typedef struct {
  PolyRunnerKind kind;
  PolyUOp *program;
  void *handle;
  int handle_size;
  int (*execute)(void *self, void **args, int n_args);
  void (*free_handle)(void *self);
  bool borrowed_handle;
  /* Uncached search compilation may retain compiler output for binary dedup.
   * Ordinary runtime-cache runners do not request or retain this extra root. */
  bool capture_binary;
  PolyUOp *compiled_binary;
  bool wait;
  double elapsed_us; /* Device timestamps when wait is requested; otherwise unused. */
  int grid[3];
  int block[3];
  PolyUOp *grid_exprs[3];
  PolyUOp *block_exprs[3];
  int n_params;
  int *var_indices;
  int n_vars;
} PolyRunner;

int poly_estimates_from_uops(
    PolyCtx *ctx,
    PolyUOp **uops,
    int n_uops,
    bool ignore_indexing,
    PolyEstimates *out
);
int poly_estimates_infer(
    const PolyEstimates *estimates,
    const PolyVarBinding *bindings,
    int n_bindings,
    uint64_t *ops,
    uint64_t *lds,
    uint64_t *mem
);

/* tinygrad.engine.realize CALL/PROGRAM helpers. */
int poly_call_n_buffer_args(PolyUOp *call);
PolyUOp *poly_call_buffer_arg(PolyUOp *call, int param_idx);
int poly_call_get_outs_ins(PolyCtx *ctx, PolyUOp *call, bool *outs, bool *ins, int n_args);
PolyUOp *poly_program_from_call(PolyCtx *ctx, PolyUOp *call, const char *name);
const PolyProgramInfo *poly_program_info(PolyCtx *ctx, PolyUOp *program);
PolyUOp *poly_program_linear(PolyUOp *program);
PolyUOp *poly_compile_linear(PolyCtx *ctx, PolyUOp *linear, int beam);

/* Uncached, waited time_call execution used by codegen/opt/search. The caller
 * owns scratch buffers; prepare/finish balance the compiled runner and roots.
 * prepare returns -1 for ordinary candidate rejection, -2 for an unexpected
 * backend failure that search._try_compile propagates under strict mode. */
int poly_time_call_prepare(PolyCtx *ctx, PolyUOp *call, PolyDevice device, PolyRunner *runner);
double poly_time_call(
    PolyRunner *runner,
    PolyDevice device,
    void **args,
    int n_args,
    const PolyVarBinding *bindings,
    int n_bindings,
    int count,
    double early_stop_us,
    int max_global_size
);
void poly_time_call_finish(PolyCtx *ctx, PolyRunner *runner, PolyDevice device);

/* tinygrad@2026-08-22/a9069c177a9d to_program_cache and runtime_cache. */
size_t poly_runtime_cache_len(PolyCtx *ctx);
void poly_runtime_cache_clear(PolyCtx *ctx);
size_t poly_runtime_cache_artifact_bytes(PolyCtx *ctx);
size_t poly_to_program_cache_len(PolyCtx *ctx);
void poly_to_program_cache_clear(PolyCtx *ctx);
int poly_program_source_render_count(void);
void poly_program_source_render_count_reset(void);

typedef struct {
  const char *name;
  PolyDevice id;
  bool host_executed;
  PolyUOp *(*rewrite_program)(PolyCtx *ctx, PolyUOp *sink);
  char *(*render_source)(PolyCtx *ctx, PolyUOp *program, const char *fn_name);
  int (*lower_item)(PolyCtx *ctx, PolyUOp *program, const char *fn_name, PolyRunner *runner_out);
  int (*execute)(PolyRunner *runner, void **args, int n_args);
  void (*free_runner)(PolyRunner *runner);
  int (*ensure_open)(void);
  const PolyAllocator *(*get_allocator)(void);
} PolyBackendDesc;

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

#ifdef POLY_TESTING
PolyUOp *poly_test_resolve_linear_param(PolyCtx *ctx, PolyUOp *uop, PolyUOp **inputs, int n_inputs);
int poly_test_linear_lane_vars(
    PolyVarBinding *bindings,
    int n_bindings,
    PolyUOp *device_num,
    int lane,
    PolyVarBinding **out,
    int *n_out
);
#endif

/* Structural hashing/equality (for cache keying by graph shape) */

uint32_t poly_structural_hash(PolyUOp *u);
bool poly_structural_eq(const void *a, const void *b);

#ifdef __cplusplus
}
#endif

#endif /* POLY_ENGINE_SCHEDULE_H */
