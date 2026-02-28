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

#ifdef __cplusplus
extern "C" {
#endif

/* ── Op enum helpers ──────────────────────────────────────────────────── */

int poly_op_count(void);

/* ── Constants ────────────────────────────────────────────────────────── */

PolyUOp *poly_const_float(PolyCtx *ctx, double value);
PolyUOp *poly_const_int(PolyCtx *ctx, int64_t value);

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

/* ── Float32 buffer shortcut ──────────────────────────────────────────── */

PolyUOp *poly_buffer_f32(PolyCtx *ctx, int64_t size);

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

typedef struct {
  PolyUOp *buffer;   /* tensor-level BUFFER UOp */
  void *data;       /* pointer to host memory (float* for f32) */
} PolyBufferBinding;

typedef struct {
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

/* ── Composed elementwise ops (shape-free, UOp-level) ────────────────── */

/* Math */
PolyUOp *poly_exp(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_log(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_sin(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_cos(PolyCtx *ctx, PolyUOp *x);
PolyUOp *poly_tan(PolyCtx *ctx, PolyUOp *x);
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
PolyUOp *poly_where_op(PolyCtx *ctx, PolyUOp *cond, PolyUOp *x, PolyUOp *y);
PolyUOp *poly_maximum(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);
PolyUOp *poly_minimum(PolyCtx *ctx, PolyUOp *a, PolyUOp *b);
PolyUOp *poly_clamp(PolyCtx *ctx, PolyUOp *x, double lo, double hi);

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

PolyUOp *poly_dot(PolyCtx *ctx,
                  PolyUOp *x, const int64_t *x_shape, int x_ndim,
                  PolyUOp *w, const int64_t *w_shape, int w_ndim,
                  int64_t *out_shape, int *out_ndim);

PolyUOp *poly_softmax(PolyCtx *ctx, PolyUOp *x,
                      const int64_t *shape, int ndim, int axis);
PolyUOp *poly_log_softmax(PolyCtx *ctx, PolyUOp *x,
                          const int64_t *shape, int ndim, int axis);

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

/* Compile a tensor-level SINK into a reusable execution step.
 * Schedules, compiles all kernels, pre-allocates intermediates.
 * BIND values are extracted as compile-time defaults for DEFINE_VAR params.
 * Caller must keep ctx alive while step exists (UOp pointers are references).
 * Not thread-safe: concurrent poly_step_run on the same step is undefined.
 * Returns NULL on failure. */
PolyStep *poly_compile_step(PolyCtx *ctx, PolyUOp *tensor_sink);

/* Execute a compiled step with buffer bindings.
 * Uses compile-time BIND defaults for DEFINE_VAR params. */
int poly_step_run(PolyStep *step,
                  PolyBufferBinding *bindings, int n_bindings);

/* Execute with explicit var bindings (overrides compile-time BIND defaults). */
int poly_step_run_ex(PolyStep *step,
                     PolyBufferBinding *bindings, int n_bindings,
                     PolyVarBinding *var_bindings, int n_var_bindings);

/* Free a compiled step (programs, intermediates, all allocations). */
void poly_step_destroy(PolyStep *step);

/* Metadata queries. */
int poly_step_n_kernels(const PolyStep *step);
int poly_step_n_intermediates(const PolyStep *step);

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
