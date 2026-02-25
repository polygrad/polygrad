/*
 * codegen.h — Linearizer, renderers, and runtimes (CPU + CUDA)
 *
 * Linearizer: priority-based toposort (port of tinygrad's linearizer.py)
 * Renderers: C source (render_c.c), WASM binary (render_wasm.c), CUDA (render_cuda.c)
 * Runtimes: CPU (runtime_cpu.c), CUDA (runtime_cuda.c)
 */

#ifndef POLY_CODEGEN_H
#define POLY_CODEGEN_H

#include "polygrad.h"
#include "pat.h"
#include <stdbool.h>

typedef struct {
  bool optimize;     /* tinygrad optimize path (UPCAST/UNROLL + late pipeline) */
  int devectorize;   /* tinygrad DEVECTORIZE level (0/1/2) */
} PolyRewriteOpts;

/* Linearize: full codegen pipeline + priority-based toposort.
 * Returns malloc'd array of UOp pointers in execution order.
 * Caller must free() the returned array. */
PolyUOp **poly_linearize(PolyCtx *ctx, PolyUOp *sink, int *n_out);

/* Linearize an already-rewritten sink (skip full_rewrite_to_sink).
 * Used by GPU backends that insert passes between rewrite and linearization. */
PolyUOp **poly_linearize_rewritten(PolyCtx *ctx, PolyUOp *sink, int *n_out);

/* GPU dims: replace outermost RANGE with SPECIAL thread indices.
 * Returns rewritten sink (or original if no suitable RANGE found). */
PolyUOp *poly_add_gpudims(PolyCtx *ctx, PolyUOp *sink);

/* Port of tinygrad's pm_add_control_flow: inject predecessor edges as
 * real RANGE sources so loop nesting is structural in the DAG.
 * Applied after full_rewrite_to_sink, before linearize. */
PolyUOp *poly_apply_control_flow(PolyCtx *ctx, PolyUOp *sink);

/* Codegen pipeline: full rewrite to sink (sym → reduce → decomp → transcendental) */
PolyUOp *poly_full_rewrite_to_sink(PolyCtx *ctx, PolyUOp *sink);
PolyUOp *poly_full_rewrite_to_sink_ex(PolyCtx *ctx, PolyUOp *sink, PolyRewriteOpts opts);

/* Individual codegen pass getters (for GPU linearizer to insert passes between them).
 * poly_symbolic_simple() is declared in pat.h. */
PolyPatternMatcher *poly_pm_reduce_pass(void);
PolyPatternMatcher *poly_pm_decomp_pass(void);
PolyPatternMatcher *poly_pm_transcendental_pass(void);
/* Apply pm_reduce with pass-local state (preferred over manual graph_rewrite). */
PolyUOp *poly_apply_pm_reduce(PolyCtx *ctx, PolyUOp *sink);

/* Legacy compatibility no-op.
 * pm_reduce state is pass-local; use poly_apply_pm_reduce(). */
void poly_reset_acc_num(void);

/* GPU parallel reduction: split large REDUCE ops into block-level parallel
 * reduction using shared memory (DEFINE_LOCAL + BARRIER).
 * Only affects REDUCE ops with range > block_size * 2.
 * Returns rewritten sink. */
PolyUOp *poly_group_for_reduce(PolyCtx *ctx, PolyUOp *sink, int block_size);

/* Render linearized UOps to C source code.
 * Includes #include <math.h>, the kernel function, and a _call wrapper.
 * Returns malloc'd string. Caller must free(). */
char *poly_render_c(PolyUOp **uops, int n, const char *fn_name);

/* Render linearized UOps to a WGSL compute shader string.
 * Returns malloc'd string. Caller must free(). */
char *poly_render_wgsl(PolyUOp **uops, int n, const char *fn_name);

/* Render linearized UOps to a WASM binary module.
 * Returns malloc'd byte array containing a valid WASM module.
 * Caller must free(). *size_out receives the byte count.
 * If use_simd is true, emits f32x4 SIMD ops for the main loop body
 * with a scalar epilogue for remainder elements. */
uint8_t *poly_render_wasm(PolyUOp **uops, int n, int *size_out, bool use_simd);

/* CPU Runtime: compile C source, load, execute */
typedef struct PolyProgram PolyProgram;

/* Compile C source string into a loadable program.
 * fn_name is the kernel function name (wrapper is fn_name_call).
 * Returns NULL on failure. */
PolyProgram *poly_compile_c(const char *source, const char *fn_name);

/* Execute a compiled program. args is an array of buffer pointers. */
void poly_program_call(PolyProgram *prog, void **args, int n_args);

/* Free a compiled program (dlclose + cleanup). */
void poly_program_destroy(PolyProgram *prog);

/* ── CUDA support (conditional on POLY_HAS_CUDA) ───────────────────── */

#ifdef POLY_HAS_CUDA

/* CUDA linearizer: rewrite + gpudims + linearize. */
PolyUOp **poly_linearize_cuda(PolyCtx *ctx, PolyUOp *sink, int *n_out);

/* Render linearized UOps to CUDA C source code.
 * Returns malloc'd string. Caller must free(). */
char *poly_render_cuda(PolyUOp **uops, int n, const char *fn_name, int launch_bounds);

/* CUDA Runtime */
typedef struct PolyCudaProgram PolyCudaProgram;

int poly_cuda_init(void);
bool poly_cuda_available(void);
unsigned long long poly_cuda_alloc(size_t bytes);
void poly_cuda_free(unsigned long long ptr);
int poly_cuda_copy_htod(unsigned long long dst, const void *src, size_t bytes);
int poly_cuda_copy_dtoh(void *dst, unsigned long long src, size_t bytes);
PolyCudaProgram *poly_compile_cuda(const char *source, const char *fn_name);
int poly_cuda_launch(PolyCudaProgram *prog, void **args, int n_args,
                     int gx, int gy, int gz, int bx, int by, int bz);
int poly_cuda_sync(void);
void poly_cuda_program_destroy(PolyCudaProgram *prog);

#endif /* POLY_HAS_CUDA */

#endif /* POLY_CODEGEN_H */
