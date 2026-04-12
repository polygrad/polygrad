/*
 * frontend.h — FFI-friendly helpers for language bindings
 *
 * Provides a simplified C surface that avoids passing PolyArg (tagged union)
 * and PolyDType (struct) across FFI boundaries. All functions take only
 * opaque pointers, integers, and doubles.
 *
 * Also provides poly_realize() which wraps the full
 * schedule → linearize → render → compile → execute pipeline.
 */

#ifndef POLY_FRONTEND_H
#define POLY_FRONTEND_H

#include "polygrad.h"
#include "tensor.h"
#include "exec_plan.h" /* PolyBuffer, PolyDeviceId */

#define POLYGRAD_ABI_VERSION 1

#ifdef __cplusplus
extern "C" {
#endif

/* In-place assignment */

/* Create ASSIGN(target, value): in-place write of value into target's buffer.
 * Target must be a BUFFER-rooted UOp (full-buffer ASSIGN only).
 * The ASSIGN is realized as a kernel that writes to the existing buffer
 * (no intermediate allocation). WAR edges ensure readers complete first. */
PolyUOp *poly_assign(PolyCtx *ctx, PolyUOp *target, PolyUOp *value);

/* Buffer shortcuts */

PolyUOp *poly_buffer_f32(PolyCtx *ctx, int64_t size);
PolyUOp *poly_buffer_f64(PolyCtx *ctx, int64_t size);
PolyUOp *poly_buffer_by_id(PolyCtx *ctx, int64_t size, int dtype_id);

/* Dynamic shapes (DEFINE_VAR / BIND) */

/* Create a symbolic integer variable with bounds [min_val, max_val]. */
PolyUOp *poly_define_var(PolyCtx *ctx, const char *name, int64_t min_val, int64_t max_val);

/* Bind a concrete value to a DEFINE_VAR (creates BIND UOp). */
PolyUOp *poly_bind_var(PolyCtx *ctx, PolyUOp *var, int64_t value);

/* Create a dynamic buffer with variable dim 0 and fixed inner dims.
 * 1D: poly_buffer_var(ctx, dt, var, NULL, 0)  -> shape (max_B,)
 * 2D: poly_buffer_var(ctx, dt, var, &K, 1)    -> shape (max_B, K)
 * Allocation size = max_val * product(inner_dims). */
PolyUOp *poly_buffer_var(
    PolyCtx *ctx,
    PolyDType dt,
    PolyUOp *batch_var,
    const int64_t *inner_dims,
    int n_inner_dims
);

/* Realize: full pipeline in one call */

typedef struct PolyBufferBinding {
  PolyUOp *buffer; /* tensor-level BUFFER UOp */
  PolyBuffer handle; /* ptr + domain + nbytes (device-aware) */
} PolyBufferBinding;

/* Convenience: build a CPU host-memory binding.
 * POLY_BIND_HOST(buf, ptr) sets domain=CPU, nbytes=0 (inferred from buf). */
#define POLY_BIND_HOST(buf, ptr) ((PolyBufferBinding){(buf), {(ptr), 0, POLY_DEVICE_CPU, false}})

typedef struct PolyVarBinding {
  PolyUOp *var; /* DEFINE_VAR UOp */
  int32_t value; /* concrete runtime value */
} PolyVarBinding;

/* Schedule, compile, and execute a tensor-level SINK.
 * bindings[] maps each BUFFER UOp in the graph to its host data.
 * Returns 0 on success, -1 on error. */
int poly_realize(PolyCtx *ctx, PolyUOp *tensor_sink, PolyBufferBinding *bindings, int n_bindings);

/* Extended realize with dynamic shape variable bindings.
 * var_bindings[] provides concrete values for DEFINE_VAR UOps.
 * BIND nodes in the graph are auto-extracted if var_bindings is NULL. */
int poly_realize_ex(
    PolyCtx *ctx,
    PolyUOp *tensor_sink,
    PolyBufferBinding *bindings,
    int n_bindings,
    PolyVarBinding *var_bindings,
    int n_var_bindings
);

/* FFI-friendlier variant: separate arrays of buffer pointers and data pointers.
 * buffers[i] is a BUFFER UOp, datas[i] is the corresponding host pointer.
 * Device is inferred from POLY_DEVICE env var or defaults to CPU/WASM_JIT. */
int poly_realize_flat(PolyCtx *ctx, PolyUOp *tensor_sink, PolyUOp **buffers, void **datas, int n);

/* Same as poly_realize_flat but with explicit device selection.
 * Use POLY_DEVICE_AUTO for default, or POLY_DEVICE_INTERP etc. */
int poly_realize_flat_device(
    PolyCtx *ctx,
    PolyUOp *tensor_sink,
    PolyUOp **buffers,
    void **datas,
    int n,
    PolyDeviceId device
);

/* Stateful realize builder — simplest FFI surface (one pointer pair per call).
 * Usage: poly_realize_begin(ctx) → N× poly_realize_bind(ctx, buf, data) → poly_realize_exec(ctx,
 * sink) */
void poly_realize_begin(PolyCtx *ctx);
void poly_realize_bind(PolyCtx *ctx, PolyUOp *buffer, void *data);
int poly_realize_exec(PolyCtx *ctx, PolyUOp *tensor_sink);

/* Get (or create) the compiled plan for a tensor-level SINK on a device.
 * Does schedule cache lookup + plan cache lookup, creating on miss.
 * The returned plan is cache-owned; caller must NOT free it.
 * Returns NULL on error. */
PolyCompiledPlan *poly_get_plan(PolyCtx *ctx, PolyUOp *tensor_sink, PolyDeviceId device);

/* WASM kernel rendering (for browser execution) */

/* Render a tensor SINK to a WASM binary module.
 * Does: schedule → linearize → render_wasm.
 * Returns malloc'd WASM bytes (caller must free). Sets *wasm_len.
 * Also stores buffer ordering internally — use poly_kernel_buf() to query.
 * *n_bufs_out receives the number of buffer parameters in the kernel. */
uint8_t *poly_render_kernel_wasm(
    PolyCtx *ctx,
    PolyUOp *tensor_sink,
    int *wasm_len,
    int *n_bufs_out
);

/* After poly_render_kernel_wasm(), get the i-th buffer UOp in PARAM order. */
PolyUOp *poly_kernel_buf(PolyCtx *ctx, int index);

/* Step-level WASM render plan for multi-kernel compiled steps.
 * Browser hosts can compile/instantiate each kernel once, then reuse.
 * kernel bytes pointers are owned by the plan and valid until destroy(). */
typedef struct PolyWasmStepPlan PolyWasmStepPlan;
PolyWasmStepPlan *poly_render_step_wasm_plan(PolyCtx *ctx, PolyUOp *tensor_sink);
int poly_wasm_stepplan_n_kernels(const PolyWasmStepPlan *p);
const uint8_t *poly_wasm_stepplan_kernel_bytes(const PolyWasmStepPlan *p, int k, int *len);
int poly_wasm_stepplan_kernel_n_params(const PolyWasmStepPlan *p, int k);
int poly_wasm_stepplan_n_buffers(const PolyWasmStepPlan *p);
int poly_wasm_stepplan_n_bindable_buffers(const PolyWasmStepPlan *p);
int poly_wasm_stepplan_kernel_param_buf_index(const PolyWasmStepPlan *p, int k, int param_idx);
const int *poly_wasm_stepplan_exec_order(const PolyWasmStepPlan *p, int *n);
void poly_wasm_stepplan_destroy(PolyWasmStepPlan *p);

int64_t poly_wasm_stepplan_buf_size(const PolyWasmStepPlan *p, int buf_idx);
int64_t poly_wasm_stepplan_buf_nbytes(const PolyWasmStepPlan *p, int buf_idx);
int poly_wasm_stepplan_bindable_buf_index(const PolyWasmStepPlan *p, int bi);

/* WebGPU step plan */

/* WebGPU step plan: schedule + linearize + render WGSL per kernel.
 * Each kernel has WGSL source, grid/local dispatch dimensions, and
 * param-to-buffer mappings. JS hosts compile WGSL → GPUShaderModule,
 * create compute pipelines, and dispatch workgroups.
 * WGSL source pointers are owned by the plan and valid until destroy(). */
typedef struct PolyWebGpuStepPlan PolyWebGpuStepPlan;
PolyWebGpuStepPlan *poly_render_step_webgpu_plan(PolyCtx *ctx, PolyUOp *tensor_sink);
int poly_webgpu_stepplan_n_kernels(const PolyWebGpuStepPlan *p);
const char *poly_webgpu_stepplan_kernel_wgsl(const PolyWebGpuStepPlan *p, int k, int *len);
int poly_webgpu_stepplan_kernel_n_params(const PolyWebGpuStepPlan *p, int k);
int poly_webgpu_stepplan_kernel_grid(const PolyWebGpuStepPlan *p, int k, int dim);
int poly_webgpu_stepplan_kernel_local(const PolyWebGpuStepPlan *p, int k, int dim);
int poly_webgpu_stepplan_n_buffers(const PolyWebGpuStepPlan *p);
int poly_webgpu_stepplan_n_bindable_buffers(const PolyWebGpuStepPlan *p);
int poly_webgpu_stepplan_bindable_buf_index(const PolyWebGpuStepPlan *p, int bi);
int poly_webgpu_stepplan_kernel_param_buf_index(const PolyWebGpuStepPlan *p, int k, int param_idx);
const int *poly_webgpu_stepplan_exec_order(const PolyWebGpuStepPlan *p, int *n);
int64_t poly_webgpu_stepplan_buf_size(const PolyWebGpuStepPlan *p, int buf_idx);
int64_t poly_webgpu_stepplan_buf_nbytes(const PolyWebGpuStepPlan *p, int buf_idx);
void poly_webgpu_stepplan_destroy(PolyWebGpuStepPlan *p);

/* ABI version (callers check at load time for compatibility). */
int poly_abi_version(void);

/* Debug: print UOp info to stderr */
void poly_debug_uop(PolyCtx *ctx, PolyUOp *u);
void poly_debug_opsets(void);

/* Compiled step (persistent sched-cache entry) */

typedef struct PolyStep PolyStep;

typedef enum {
  POLY_STEP_BUF_INPUT = 0,
  POLY_STEP_BUF_OUTPUT = 1,
  POLY_STEP_BUF_TEMP = 2,
  /* Phase E: POLY_STEP_BUF_CONSTANT = 3 was removed along with the
   * const-registry. Pure-UOp creation helpers (arange/eye/full/...) never
   * produce constant-role buffers any more. */
} PolyStepBufRole;

#define POLY_STEP_BUFFER_INFO_VERSION 1
typedef struct {
  int version;
  int index;
  PolyStepBufRole role;
  PolyDType dtype;
  int64_t numel;
  int64_t nbytes;
} PolyStepBufferInfo;

/* Compile a tensor-level SINK into a reusable execution step.
 * Schedules, compiles all kernels, pre-allocates intermediates.
 * BIND values are extracted as compile-time defaults for DEFINE_VAR params.
 * Caller must keep ctx alive while step exists (UOp pointers are references).
 * Not thread-safe: concurrent poly_step_run on the same step is undefined.
 * Returns NULL on failure. */
PolyStep *poly_compile_step(PolyCtx *ctx, PolyUOp *tensor_sink);

/* Compile a scalar loss + parameter gradients into a reusable step.
 * The compiled step writes:
 *   output buffer[0]   -> loss (1 element)
 *   output buffer[i+1] -> grad for params[i] (flattened)
 * out_loss_buf_idx receives the loss output buffer index.
 * out_grad_buf_idxs must point to caller-allocated array of n_params ints. */
PolyStep *poly_compile_value_and_grad(
    PolyCtx *ctx,
    PolyUOp *loss,
    PolyUOp **params,
    int n_params,
    int *out_loss_buf_idx,
    int *out_grad_buf_idxs
);

/* Execute a compiled step with buffer bindings.
 * Uses compile-time BIND defaults for DEFINE_VAR params. */
int poly_step_run(PolyStep *step, PolyBufferBinding *bindings, int n_bindings);

/* Execute with explicit var bindings (overrides compile-time BIND defaults). */
int poly_step_run_ex(
    PolyStep *step,
    PolyBufferBinding *bindings,
    int n_bindings,
    PolyVarBinding *var_bindings,
    int n_var_bindings
);

/* Execute using index-based buffer pointers (no PolyUOp exposure).
 * buffer_data[idx] maps to PolyStepBufferInfo.index.
 * Bindable slots are [0 .. poly_step_n_bindable_buffers(step)-1] (external buffers).
 * TEMP/CONSTANT metadata entries beyond that range are informational. */
int poly_step_run_indexed(PolyStep *step, void **buffer_data, int n_buffers);
int poly_step_run_indexed_ex(
    PolyStep *step,
    void **buffer_data,
    int n_buffers,
    PolyVarBinding *var_bindings,
    int n_var_bindings
);

/* Free a compiled step (programs, intermediates, all allocations). */
void poly_step_destroy(PolyStep *step);

/* Metadata queries. */
int poly_step_n_kernels(const PolyStep *step);
int poly_step_n_intermediates(const PolyStep *step);
int poly_step_n_buffers(const PolyStep *step);
int poly_step_n_bindable_buffers(const PolyStep *step);
int poly_step_buffer_info(const PolyStep *step, int idx, PolyStepBufferInfo *out);

/* Return the BUFFER UOp at position idx in the step's DFS buffer ordering.
 * Valid range: [0, poly_step_n_bindable_buffers(step)). Returns NULL on error. */
PolyUOp *poly_step_buf_uop(const PolyStep *step, int idx);

/* Cache cleanup (for leak-free shutdown) */

/* Free all cached compiled CPU programs (dlclose + free). */
void poly_cpu_cache_flush(void);

/* Free cached schedule results (param-to-binding mappings). */
void poly_sched_cache_flush(void);

/* CUDA realize */

#ifdef POLY_HAS_CUDA

/* DEPRECATED: use poly_realize() with CUDA-domain PolyBuffer bindings.
 * Legacy standalone CUDA realize path. Retained for Python/JS frontend compat. */
int poly_realize_cuda(
    PolyCtx *ctx,
    PolyUOp *tensor_sink,
    PolyBufferBinding *bindings,
    int n_bindings
);
void poly_cuda_flush_buffers(void);
void poly_cuda_prog_cache_flush(void);
int poly_cuda_copyback(PolyBufferBinding *bindings, int n_bindings);

#endif /* POLY_HAS_CUDA */

#ifdef __cplusplus
}
#endif

#endif /* POLY_FRONTEND_H */
