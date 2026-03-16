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
#include "exec_plan.h"  /* PolyBufferHandle, PolyDeviceId */

#define POLYGRAD_ABI_VERSION 1

#ifdef __cplusplus
extern "C" {
#endif

/* ── Op enum helpers ──────────────────────────────────────────────────── */

int poly_op_count(void);

/* ── Constants ────────────────────────────────────────────────────────── */

PolyUOp *poly_const_float(PolyCtx *ctx, double value);
PolyUOp *poly_const_double(PolyCtx *ctx, double value);
PolyUOp *poly_const_int(PolyCtx *ctx, int64_t value);
PolyUOp *poly_const_typed(PolyCtx *ctx, PolyDType dt, double value);

/* ── ALU ops (no PolyArg needed) ───────────────────────────────────────── */

PolyUOp *poly_alu1(PolyCtx *ctx, PolyOps op, PolyUOp *src);
PolyUOp *poly_alu2(PolyCtx *ctx, PolyOps op, PolyUOp *a, PolyUOp *b);
PolyUOp *poly_alu3(PolyCtx *ctx, PolyOps op, PolyUOp *a, PolyUOp *b, PolyUOp *c);

/* ── Graph construction ───────────────────────────────────────────────── */

PolyUOp *poly_store_val(PolyCtx *ctx, PolyUOp *buf, PolyUOp *value);
PolyUOp *poly_sink1(PolyCtx *ctx, PolyUOp *store);
PolyUOp *poly_sink_n(PolyCtx *ctx, PolyUOp **stores, int n);

/* ── In-place assignment ──────────────────────────────────────────────── */

/* Create ASSIGN(target, value): in-place write of value into target's buffer.
 * Target must be a BUFFER-rooted UOp (full-buffer ASSIGN only).
 * The ASSIGN is realized as a kernel that writes to the existing buffer
 * (no intermediate allocation). WAR edges ensure readers complete first. */
PolyUOp *poly_assign(PolyCtx *ctx, PolyUOp *target, PolyUOp *value);

/* ── Buffer shortcuts ─────────────────────────────────────────────────── */

PolyUOp *poly_buffer_f32(PolyCtx *ctx, int64_t size);
PolyUOp *poly_buffer_f64(PolyCtx *ctx, int64_t size);
PolyUOp *poly_buffer_by_id(PolyCtx *ctx, int64_t size, int dtype_id);

/* ── Dynamic shapes (DEFINE_VAR / BIND) ──────────────────────────────── */

/* Create a symbolic integer variable with bounds [min_val, max_val]. */
PolyUOp *poly_define_var(PolyCtx *ctx, const char *name, int64_t min_val, int64_t max_val);

/* Bind a concrete value to a DEFINE_VAR (creates BIND UOp). */
PolyUOp *poly_bind_var(PolyCtx *ctx, PolyUOp *var, int64_t value);

/* Create a dynamic buffer with variable dim 0 and fixed inner dims.
 * 1D: poly_buffer_var(ctx, dt, var, NULL, 0)  -> shape (max_B,)
 * 2D: poly_buffer_var(ctx, dt, var, &K, 1)    -> shape (max_B, K)
 * Allocation size = max_val * product(inner_dims). */
PolyUOp *poly_buffer_var(PolyCtx *ctx, PolyDType dt, PolyUOp *batch_var,
                          const int64_t *inner_dims, int n_inner_dims);

/* ── Realize: full pipeline in one call ───────────────────────────────── */

typedef struct PolyBufferBinding {
  PolyUOp *buffer;       /* tensor-level BUFFER UOp */
  PolyBufferHandle handle; /* ptr + domain + nbytes (device-aware) */
} PolyBufferBinding;

/* Convenience: build a CPU host-memory binding.
 * POLY_BIND_HOST(buf, ptr) sets domain=CPU, nbytes=0 (inferred from buf). */
#define POLY_BIND_HOST(buf, ptr) \
  ((PolyBufferBinding){ (buf), { (ptr), 0, POLY_DEVICE_CPU, false } })

typedef struct PolyVarBinding {
  PolyUOp *var;     /* DEFINE_VAR UOp */
  int32_t value;    /* concrete runtime value */
} PolyVarBinding;

/* Schedule, compile, and execute a tensor-level SINK.
 * bindings[] maps each BUFFER UOp in the graph to its host data.
 * Returns 0 on success, -1 on error. */
int poly_realize(PolyCtx *ctx, PolyUOp *tensor_sink,
                PolyBufferBinding *bindings, int n_bindings);

/* Extended realize with dynamic shape variable bindings.
 * var_bindings[] provides concrete values for DEFINE_VAR UOps.
 * BIND nodes in the graph are auto-extracted if var_bindings is NULL. */
int poly_realize_ex(PolyCtx *ctx, PolyUOp *tensor_sink,
                    PolyBufferBinding *bindings, int n_bindings,
                    PolyVarBinding *var_bindings, int n_var_bindings);

/* FFI-friendlier variant: separate arrays of buffer pointers and data pointers.
 * buffers[i] is a BUFFER UOp, datas[i] is the corresponding host pointer. */
int poly_realize_flat(PolyCtx *ctx, PolyUOp *tensor_sink,
                     PolyUOp **buffers, void **datas, int n);

/* Stateful realize builder — simplest FFI surface (one pointer pair per call).
 * Usage: poly_realize_begin(ctx) → N× poly_realize_bind(ctx, buf, data) → poly_realize_exec(ctx, sink) */
void poly_realize_begin(PolyCtx *ctx);
void poly_realize_bind(PolyCtx *ctx, PolyUOp *buffer, void *data);
int  poly_realize_exec(PolyCtx *ctx, PolyUOp *tensor_sink);

/* ── WASM kernel rendering (for browser execution) ───────────────────── */

/* Render a tensor SINK to a WASM binary module.
 * Does: schedule → linearize → render_wasm.
 * Returns malloc'd WASM bytes (caller must free). Sets *wasm_len.
 * Also stores buffer ordering internally — use poly_kernel_buf() to query.
 * *n_bufs_out receives the number of buffer parameters in the kernel. */
uint8_t *poly_render_kernel_wasm(PolyCtx *ctx, PolyUOp *tensor_sink,
                                 int *wasm_len, int *n_bufs_out);

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

/* Query constant buffer data. Returns NULL if buf is not a registered constant. */
const void *poly_const_buffer_data(PolyCtx *ctx, PolyUOp *buf);

/* ABI version (callers check at load time for compatibility). */
int poly_abi_version(void);

/* ── Composed elementwise ops (shape-free, UOp-level) ────────────────── */

/* Math */
PolyUOp *poly_exp(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_log(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_log1p(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_expm1(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_sin(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_cos(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_tan(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_erf(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_erfc(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_erfinv(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_ndtri(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_digamma(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_lgamma(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_sigmoid(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_tanh_act(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_abs(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_sign(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_square(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_rsqrt(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_ceil(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_floor(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_round_f(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_isinf(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_isnan(PolyCtx *ctx, PolyUOp *x);

/* Activations */
PolyUOp *poly_relu(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_relu6(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_leaky_relu(PolyCtx *ctx, PolyUOp *x, double neg_slope);
PolyUOp *poly_gelu(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_quick_gelu(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_silu(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_elu(PolyCtx *ctx, PolyUOp *x, double alpha);
PolyUOp *poly_softplus(PolyCtx *ctx, PolyUOp *x, double beta);
PolyUOp *poly_mish(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_hardtanh(PolyCtx *ctx, PolyUOp *x, double min_val, double max_val);
PolyUOp *poly_hardswish(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_hardsigmoid(PolyCtx *ctx, PolyUOp *x);

/* Comparisons */
PolyUOp *poly_eq(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);
PolyUOp *poly_ne(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);
PolyUOp *poly_gt(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);
PolyUOp *poly_ge(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);
PolyUOp *poly_le(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);
PolyUOp *poly_cast(PolyCtx *ctx, PolyUOp *x, PolyDType target);
PolyUOp *poly_cast_by_id(PolyCtx *ctx, PolyUOp *x, int dtype_id);
PolyUOp *poly_where_op(PolyCtx *ctx, PolyUOp *cond, PolyUOp *x, PolyUOp *y);
PolyUOp *poly_maximum(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);
PolyUOp *poly_minimum(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);
PolyUOp *poly_clamp(PolyCtx *ctx, PolyUOp *x, double lo, double hi);
PolyUOp *poly_detach(PolyCtx *ctx, PolyUOp *x);

/* Deterministic RNG helpers (stateless seed -> tensor). */
PolyUOp *poly_rand(PolyCtx *ctx, const int64_t *shape, int ndim, uint64_t seed);
PolyUOp *poly_randn(PolyCtx *ctx, const int64_t *shape, int ndim, uint64_t seed);

/* Creation helpers (constant-backed tensors). */
PolyUOp *poly_arange(PolyCtx *ctx, double start, double stop, double step);
PolyUOp *poly_eye(PolyCtx *ctx, int64_t n);
PolyUOp *poly_linspace(PolyCtx *ctx, double start, double stop, int64_t steps);
PolyUOp *poly_full(PolyCtx *ctx, const int64_t *shape, int ndim, double fill_value);
PolyUOp *poly_tril(PolyCtx *ctx, PolyUOp *x, const int64_t *shape, int ndim, int diagonal);
PolyUOp *poly_triu(PolyCtx *ctx, PolyUOp *x, const int64_t *shape, int ndim, int diagonal);
PolyUOp *poly_cholesky(PolyCtx *ctx, PolyUOp *x, const int64_t *shape, int ndim, int upper);
PolyUOp *poly_triangular_solve(PolyCtx *ctx,
                               PolyUOp *a, const int64_t *a_shape, int a_ndim,
                               PolyUOp *b, const int64_t *b_shape, int b_ndim,
                               int upper, int transpose_a, int unit_diagonal,
                               int64_t *out_shape, int *out_ndim);

/* ── Shape-aware composed ops ────────────────────────────────────────── */

PolyUOp *poly_sum_reduce(PolyCtx *ctx, PolyUOp *x,
                         const int64_t *shape, int ndim,
                         int axis, int keepdim,
                         int64_t *out_shape, int *out_ndim);
PolyUOp *poly_max_reduce(PolyCtx *ctx, PolyUOp *x,
                         const int64_t *shape, int ndim,
                         int axis, int keepdim,
                         int64_t *out_shape, int *out_ndim);
PolyUOp *poly_mean_reduce(PolyCtx *ctx, PolyUOp *x,
                          const int64_t *shape, int ndim,
                          int axis, int keepdim,
                          int64_t *out_shape, int *out_ndim);
PolyUOp *poly_var_reduce(PolyCtx *ctx, PolyUOp *x,
                         const int64_t *shape, int ndim,
                         int axis, int keepdim, int correction,
                         int64_t *out_shape, int *out_ndim);
PolyUOp *poly_logsumexp(PolyCtx *ctx, PolyUOp *x,
                        const int64_t *shape, int ndim,
                        int axis, int keepdim,
                        int64_t *out_shape, int *out_ndim);

PolyUOp *poly_dot(PolyCtx *ctx,
                  PolyUOp *x, const int64_t *x_shape, int x_ndim,
                  PolyUOp *w, const int64_t *w_shape, int w_ndim,
                  int64_t *out_shape, int *out_ndim);

PolyUOp *poly_softmax(PolyCtx *ctx, PolyUOp *x,
                      const int64_t *shape, int ndim, int axis);
PolyUOp *poly_log_softmax(PolyCtx *ctx, PolyUOp *x,
                          const int64_t *shape, int ndim, int axis);
PolyUOp *poly_cross_entropy(PolyCtx *ctx,
                            PolyUOp *logits, const int64_t *logits_shape, int logits_ndim,
                            PolyUOp *target, const int64_t *target_shape, int target_ndim,
                            int axis, int64_t *out_shape, int *out_ndim);

/* ── Einsum ────────────────────────────────────────────────────────── */

/* Einstein summation: parse subscript formula, align+mul+sum+permute.
 * formula: e.g. "ij,jk->ik"   (lowercase a-z indices only, no ellipsis)
 * tensors/shapes/ndims: parallel arrays for each input operand
 * out_shape/out_ndim: written with result shape
 * Returns the result UOp, or NULL on parse/shape error. */
PolyUOp *poly_einsum(PolyCtx *ctx, const char *formula,
                     PolyUOp **tensors, const int64_t **shapes, const int *ndims,
                     int n_tensors, int64_t *out_shape, int *out_ndim);

/* ── Rearrange (einops) ───────────────────────────────────────────── */

/* Einops-style rearrange: parse formula, unflatten→permute→flatten.
 * formula: e.g. "b c h w -> b (c h) w"
 * x/shape/ndim: input tensor
 * axis_sizes: name→size pairs for unflatten. Format: names as space-separated
 *             string in axis_names, values in axis_values, n_axis_sizes count.
 * out_shape/out_ndim: written with result shape
 * Returns the result UOp, or NULL on error. */
PolyUOp *poly_rearrange(PolyCtx *ctx, const char *formula,
                        PolyUOp *x, const int64_t *shape, int ndim,
                        const char *axis_names, const int64_t *axis_values,
                        int n_axis_sizes,
                        int64_t *out_shape, int *out_ndim);

/* ── Transformer building blocks ───────────────────────────────────── */

/* Gather rows from table by integer indices (embedding lookup).
 * table: (N, D) weight matrix
 * indices: (...) integer indices (as floats, cast internally)
 * Returns: (..., D) gathered rows
 * Lowered as: out[..., j] = table[int(indices[...]), j]
 * Avoids O(vocab) materialization of one-hot mask. */
PolyUOp *poly_gather(PolyCtx *ctx,
                      PolyUOp *table, const int64_t *table_shape, int table_ndim,
                      PolyUOp *indices, const int64_t *idx_shape, int idx_ndim,
                      int64_t *out_shape, int *out_ndim);

/* Layer normalization: (x - mean) / sqrt(var + eps).
 * Normalizes over the last axis. No affine (caller applies weight/bias). */
PolyUOp *poly_layernorm(PolyCtx *ctx, PolyUOp *x,
                         const int64_t *shape, int ndim,
                         int axis, double eps,
                         int64_t *out_shape, int *out_ndim);

/* Build causal attention mask for sequence length T.
 * Returns (T, T) mask: 0 where allowed, -1e9 where masked (upper triangle).
 * T_val: const UOp (value = sequence length). */
PolyUOp *poly_causal_mask(PolyCtx *ctx, int64_t T,
                           int64_t *out_shape, int *out_ndim);

/* Linear projection: x @ weight.T + bias.
 * x: (..., in_features), weight: (out_features, in_features)
 * bias: (out_features,) or NULL for no bias.
 * Returns: (..., out_features). */
PolyUOp *poly_linear(PolyCtx *ctx,
                      PolyUOp *x, const int64_t *x_shape, int x_ndim,
                      PolyUOp *weight, const int64_t *w_shape, int w_ndim,
                      PolyUOp *bias, const int64_t *bias_shape, int bias_ndim,
                      int64_t *out_shape, int *out_ndim);

/* Debug: print UOp info to stderr */
void poly_debug_uop(PolyCtx *ctx, PolyUOp *u);
void poly_debug_opsets(void);

/* ── Compiled step (persistent sched-cache entry) ──────────────────────── */

typedef struct PolyStep PolyStep;

typedef enum {
  POLY_STEP_BUF_INPUT = 0,
  POLY_STEP_BUF_OUTPUT = 1,
  POLY_STEP_BUF_TEMP = 2,
  POLY_STEP_BUF_CONSTANT = 3,
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
PolyStep *poly_compile_value_and_grad(PolyCtx *ctx, PolyUOp *loss,
                                      PolyUOp **params, int n_params,
                                      int *out_loss_buf_idx,
                                      int *out_grad_buf_idxs);

/* Execute a compiled step with buffer bindings.
 * Uses compile-time BIND defaults for DEFINE_VAR params. */
int poly_step_run(PolyStep *step,
                  PolyBufferBinding *bindings, int n_bindings);

/* Execute with explicit var bindings (overrides compile-time BIND defaults). */
int poly_step_run_ex(PolyStep *step,
                     PolyBufferBinding *bindings, int n_bindings,
                     PolyVarBinding *var_bindings, int n_var_bindings);

/* Execute using index-based buffer pointers (no PolyUOp exposure).
 * buffer_data[idx] maps to PolyStepBufferInfo.index.
 * Bindable slots are [0 .. poly_step_n_bindable_buffers(step)-1] (external buffers).
 * TEMP/CONSTANT metadata entries beyond that range are informational. */
int poly_step_run_indexed(PolyStep *step, void **buffer_data, int n_buffers);
int poly_step_run_indexed_ex(PolyStep *step, void **buffer_data, int n_buffers,
                             PolyVarBinding *var_bindings, int n_var_bindings);

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

/* ── Cache cleanup (for leak-free shutdown) ───────────────────────────── */

/* Free all cached compiled CPU programs (dlclose + free). */
void poly_cpu_cache_flush(void);

/* Free cached schedule results (param-to-binding mappings). */
void poly_sched_cache_flush(void);

/* ── CUDA realize ────────────────────────────────────────────────────── */

#ifdef POLY_HAS_CUDA

/* GPU version of poly_realize(). Same interface: schedule → linearize_cuda →
 * render_cuda → compile_cuda → GPU alloc/copy/launch/copy-back.
 * Returns 0 on success, -1 on error. */
int poly_realize_cuda(PolyCtx *ctx, PolyUOp *tensor_sink,
                     PolyBufferBinding *bindings, int n_bindings);

/* Free all cached GPU buffer allocations */
void poly_cuda_flush_buffers(void);

/* Free all cached compiled CUDA programs. */
void poly_cuda_prog_cache_flush(void);

/* Copy GPU-resident output buffers back to host memory.
 * Call after poly_realize_cuda() to read results. */
int poly_cuda_copyback(PolyBufferBinding *bindings, int n_bindings);

#endif /* POLY_HAS_CUDA */

#ifdef __cplusplus
}
#endif

#endif /* POLY_FRONTEND_H */
