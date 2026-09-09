/*
 * codegen/codegen.h — current Tinygrad codegen pipeline declarations
 *
 * Linearizer: priority-based toposort (port of tinygrad's linearizer.py)
 * Renderers live under src/renderer; runtimes live below src/runtime.
 * Runtimes: CPU (runtime_cpu.c), CUDA (runtime_cuda.c)
 */

#ifndef POLY_CODEGEN_H
#define POLY_CODEGEN_H

#include "polygrad.h"
#include "uop/upat.h"
#include "uop/ops.h"
#include "codegen/late/linearizer.h"
#include "renderer/renderer.h"
#include <stdbool.h>

/* Optimization policy: explicit discriminator for what the heuristic does.
 * Wrappers set this directly instead of inferring from caps. */
typedef enum {
  POLY_OPT_HEURISTIC = 0, /* CPU: full heuristic (masked upcast, stride upcast, reduce unroll) */
  POLY_OPT_TC_ONLY, /* GPU: TC detection only, no CPU-oriented scheduling */
} PolyOptPolicy;

typedef struct {
  bool optimize; /* tinygrad optimize path (UPCAST/UNROLL + late pipeline) */
  int beam_width; /* BEAM search width (0 = heuristic, >0 = BEAM search) */
  PolyRendererCaps caps; /* renderer capabilities (zero-init = conservative/unsupported) */
  /* Renderer config for unified pipeline (Phase 4) */
  int device; /* PolyDevice from engine/schedule.h (0 = CPU) */
  PolyOptPolicy opt_policy; /* explicit optimization strategy */
  PolyPatternMatcher *extra_matcher; /* renderer-specific final rewrite (NULL = none) */
} PolyRewriteOpts;

/* Tinygrad's library-wide NOOPT compilation ContextVar. */
int poly_get_noopt(void);
void poly_set_noopt(int value);

static inline bool poly_kernel_optimize_enabled(PolyUOp *sink) {
  return !(sink && sink->op == POLY_OP_SINK && sink->tag != 0);
}

static inline const char *poly_kernel_name(PolyUOp *sink, const char *fallback) {
  if (sink && sink->op == POLY_OP_SINK && sink->arg.kind == POLY_ARG_KERNEL_INFO &&
      sink->arg.kernel_info && sink->arg.kernel_info->name && sink->arg.kernel_info->name[0])
    return sink->arg.kernel_info->name;
  return fallback ? fallback : "test";
}

static inline int poly_kernel_beam(PolyUOp *sink) {
  return sink && sink->op == POLY_OP_SINK && sink->arg.kind == POLY_ARG_KERNEL_INFO &&
                 sink->arg.kernel_info
             ? sink->arg.kernel_info->beam
             : 0;
}

/* Current Tinygrad 2026-08-22/a9069c177a9d codegen/__init__.py:do_linearize.
 * The input is a full_rewrite_to_sink result; the output is malloc-owned. */
PolyUOp **poly_do_linearize(PolyCtx *ctx, PolyUOp *sink, int *n_out);
/* WASM linearizer: use the shared late pipeline, preserving only the native
 * renderer-capability vector subset (currently f32x4 ALU/compare/compare-mask
 * WHERE) and scalarizing unsupported semantic vectors before rendering. */
PolyUOp *poly_rewrite_wasm(PolyCtx *ctx, PolyUOp *sink);
PolyUOp **poly_linearize_wasm(PolyCtx *ctx, PolyUOp *sink, int *n_out);

/* Default CPU/C renderer capabilities, matching tinygrad's ClangRenderer:
 * float4-capable, no local memory scheduling, no native MULACC/THREEFRY. */
PolyRendererCaps poly_c_renderer_caps(void);

/* GPU dims: replace outermost RANGE with SPECIAL thread indices.
 * Returns rewritten sink (or original if no suitable RANGE found). */
PolyUOp *poly_add_gpudims(PolyCtx *ctx, PolyUOp *sink);
PolyUOp *poly_add_gpudims_ex(PolyCtx *ctx, PolyUOp *sink, PolyRendererCaps caps);

/* Port of tinygrad's pm_add_control_flow: inject predecessor edges as
 * real RANGE sources so loop nesting is structural in the DAG.
 * Applied after full_rewrite_to_sink, before linearize. */
PolyUOp *poly_apply_control_flow(PolyCtx *ctx, PolyUOp *sink);

/* Apply heuristic optimizer only, before movement-shaped expander2. */
PolyUOp *poly_apply_opts_heuristic_ex(PolyCtx *ctx, PolyUOp *sink, PolyRendererCaps caps);

/* Apply only the TC detection part of the heuristic. Does not apply CPU-oriented
 * upcast/unroll heuristics. For GPU linearizers that handle their own scheduling. */
PolyUOp *poly_apply_tc_opt(PolyCtx *ctx, PolyUOp *sink, PolyRendererCaps caps);

/* TensorCore helpers (internal, exposed for testing) */

/* Current Tinygrad memory addresses are INDEX UOps; dtype carries values. */
static inline PolyUOp *poly_as_index(PolyUOp *u) {
  return (u && u->op == POLY_OP_INDEX) ? u : NULL;
}

/* Memory-addressing form accepted after August memory coalescing. A three-src
 * SHRINK is a coalesced memory slice, not a frontend movement. */
static inline PolyUOp *poly_as_memory_slice(PolyUOp *u) {
  return (u && (u->op == POLY_OP_INDEX || u->op == POLY_OP_SHRINK)) ? u : NULL;
}

static inline bool poly_is_program_memory_base(const PolyUOp *u) {
  if (!u) return false;
  return u->op == POLY_OP_PARAM || u->op == POLY_OP_BUFFER || u->op == POLY_OP_AFTER;
}

/* tinygrad@2026-08-22/a9069c177a9d uop/ops.py:384,786-799 and
 * renderer/cstyle.py:154-178. Program memory stores address space and slot in
 * ParamArg and the storage extent in src[0]. */
static inline PolyAddrSpace poly_program_memory_addrspace(const PolyUOp *u) {
  PolyAddrSpace addrspace = POLY_ADDR_GLOBAL;
  return poly_uop_addrspace(u, &addrspace) ? addrspace : POLY_ADDR_GLOBAL;
}

static inline bool poly_program_memory_is(const PolyUOp *u, PolyAddrSpace addrspace) {
  return u && poly_program_memory_addrspace(u) == addrspace;
}

static inline int64_t poly_program_buffer_slot(const PolyUOp *u) {
  if (!u) return -1;
  if (u->arg.kind == POLY_ARG_PARAM && u->arg.param) return u->arg.param->slot;
  return u->arg.kind == POLY_ARG_INT ? u->arg.i : -1;
}

static inline int64_t poly_program_buffer_size(const PolyUOp *u) {
  if (!u) return 1;
  const PolyUOp *size = u->n_src == 1 ? u->src[0] : NULL;
  /* pm_casted_consts spells a concrete shape as CAST(int, CONST(weakint)). */
  if (size && size->op == POLY_OP_CAST && size->n_src == 1) size = size->src[0];
  if (size && size->op == POLY_OP_CONST && size->arg.kind == POLY_ARG_INT && size->arg.i > 0)
    return size->arg.i;
  return 1;
}

static inline PolyDType poly_program_buffer_dtype(const PolyUOp *u) {
  return u ? u->dtype : POLY_VOID;
}

/* Codegen pipeline: full rewrite to sink (sym → reduce → decomp → transcendental) */
PolyUOp *poly_full_rewrite_to_sink(PolyCtx *ctx, PolyUOp *sink);
PolyUOp *poly_full_rewrite_to_sink_ex(PolyCtx *ctx, PolyUOp *sink, PolyRewriteOpts opts);

#ifdef POLY_TESTING
/* Owner-local reduction scratch allocation only; -1 disables injection. */
void poly_test_reduce_alloc_fail_after(int count);
#endif

#if defined(POLY_TESTING) && !defined(__EMSCRIPTEN__)
int poly_test_beam_actions(PolyOpt *out, int capacity);
int poly_test_beam_kernel_action(PolyCtx *ctx, PolyUOp *sink, PolyRendererCaps caps, PolyOpt opt);
int poly_test_beam_last_device(void);
uint64_t poly_test_beam_cache_key(PolyCtx *ctx, PolyUOp *sink, int width, PolyDevice device);
int poly_test_beam_cache_write(
    PolyCtx *ctx,
    PolyUOp *sink,
    PolyUOp *result,
    int width,
    char *path,
    size_t capacity
);
PolyUOp *poly_test_beam_cache_read(PolyCtx *ctx, PolyUOp *sink, int width);
PolyUOp *poly_test_apply_opt(PolyCtx *ctx, PolyUOp *sink, PolyRendererCaps caps, PolyOpt opt);
bool poly_test_scheduler_copy_rollback(PolyCtx *ctx, PolyUOp *sink, PolyOpt opt);
bool poly_test_scheduler_copy_lifetime(PolyCtx *ctx, PolyUOp *sink);
bool poly_test_scheduler_reaches(PolyCtx *ctx, PolyUOp *sink, PolyUOp *index, PolyUOp *range);
PolyUOp *poly_test_convert_loop_to_global(PolyCtx *ctx, PolyUOp *sink);
void **poly_test_beam_args_from_ast(PolyCtx *ctx, PolyUOp *sink, int *n_args);
double poly_test_beam_compile_and_time(
    PolyCtx *ctx,
    PolyUOp *sink,
    PolyRewriteOpts opts,
    int reps,
    int *n_args
);
#endif

/* Individual codegen pass getters (for GPU linearizer to insert passes between them).
 * poly_symbolic_simple()/poly_symbolic() are declared in upat.h. */
/* Current tinygrad/codegen/decomp/op.py matcher boundaries. */
PolyPatternMatcher *poly_get_simplifying_rewrite_patterns(PolyRendererCaps caps);
PolyPatternMatcher *poly_get_late_rewrite_patterns(PolyRendererCaps caps);
PolyPatternMatcher *poly_get_transcendental_patterns(PolyRendererCaps caps);
PolyUOp *poly_apply_expander2(PolyCtx *ctx, PolyUOp *sink);
/* Tinygrad-aligned late codegen stage helpers used by probes/tests.
 * These apply the same stage slices as full_rewrite_to_sink_ex instead of
 * exposing a single internal matcher under a misleading boundary name. */
PolyUOp *poly_apply_devectorizer2_stage(PolyCtx *ctx, PolyUOp *sink, PolyRendererCaps caps);
PolyUOp *poly_apply_expand_broadcast_stage(PolyCtx *ctx, PolyUOp *sink);
PolyUOp *poly_apply_post_index_symbolic_stage(PolyCtx *ctx, PolyUOp *sink);
/* Apply pm_reduce with pass-local state (preferred over manual graph_rewrite). */
PolyUOp *poly_apply_pm_reduce(PolyCtx *ctx, PolyUOp *sink);

/* Render linearized UOps to C source code.
 * Includes #include <math.h>, the kernel function, and a _call wrapper.
 * Returns malloc'd string. Caller must free(). */
char *poly_render_c(PolyCtx *ctx, PolyUOp **uops, int n, const char *fn_name);

/* Render linearized UOps to a WGSL compute shader string.
 * Returns malloc'd string. Caller must free(). */
char *poly_render_wgsl(PolyCtx *ctx, PolyUOp **uops, int n, const char *fn_name);

/* WGSL extra matcher: shift u32 normalization, bool CMPLT/XOR (tinygrad wgsl_matcher). */
PolyPatternMatcher *poly_pm_wgsl_extra(void);

/* Linearize a kernel for WebGPU execution.
 * Full codegen pipeline with GPU dims (SPECIAL, BARRIER, shared memory).
 * WebGPU constraints: supports_float4=false, local_max=256, no tensor cores. */
PolyUOp *poly_rewrite_webgpu(PolyCtx *ctx, PolyUOp *sink);
PolyUOp **poly_linearize_webgpu(PolyCtx *ctx, PolyUOp *sink, int *n_out);

/* Render linearized UOps to a WASM binary module.
 * Returns malloc'd byte array containing a valid WASM module.
 * Caller must free(). *size_out receives the byte count.
 * If use_simd is true, emits f32x4 SIMD ops for the main loop body
 * with a scalar epilogue for remainder elements. */
uint8_t *poly_render_wasm(PolyCtx *ctx, PolyUOp **uops, int n, int *size_out, bool use_simd);
bool poly_wasm_can_render_matmul(PolyUOp *sink);
uint8_t *poly_render_wasm_matmul(PolyUOp *sink, int *size_out, bool use_relaxed_madd);
bool poly_wasm_can_render_reduce(PolyUOp *sink);
uint8_t *poly_render_wasm_reduce(PolyUOp *sink, int *size_out);

/* CPU Runtime: compile C source, load, execute */
typedef struct PolyProgram PolyProgram;

/* Compile C source string into a loadable program.
 * fn_name is the kernel function name (wrapper is fn_name_call).
 * Returns NULL on failure. */
PolyProgram *poly_compile_c(const char *source, const char *fn_name);

/* Execute a compiled program. args is an array of buffer pointers. */
void poly_program_call(PolyProgram *prog, void **args, int n_args);
void poly_program_call_threaded(PolyProgram *prog, void **args, int n_args, int threads);

/* Approximate C runtime artifact bytes retained by a compiled program.
 * Includes the PolyProgram wrapper and backing shared object when available. */
size_t poly_program_estimated_size(const PolyProgram *prog);
/* Compiler output copy for search.seen_libs; caller frees the returned bytes. */
uint8_t *poly_program_read_binary(const PolyProgram *prog, int *size);

/* Free a compiled program (dlclose + cleanup). */
void poly_program_destroy(PolyProgram *prog);

/* CUDA support (conditional on POLY_HAS_CUDA) */

#ifdef POLY_HAS_CUDA

#include "runtime/graph/cuda.h"

/* CUDA linearizer: rewrite + gpudims + linearize. */
PolyUOp *poly_rewrite_cuda(PolyCtx *ctx, PolyUOp *sink);
PolyUOp **poly_linearize_cuda(PolyCtx *ctx, PolyUOp *sink, int *n_out);

/* Render linearized UOps to CUDA C source code.
 * Returns malloc'd string. Caller must free(). */
char *poly_render_cuda(PolyCtx *ctx, PolyUOp **uops, int n, const char *fn_name, int launch_bounds);

/* CUDA Runtime */
typedef struct PolyCudaProgram PolyCudaProgram;

int poly_cuda_init(void);
bool poly_cuda_available(void);
int poly_cuda_arch_major(void);
int poly_cuda_arch_minor(void);
unsigned long long poly_cuda_alloc(size_t bytes);
void poly_cuda_free(unsigned long long ptr);
int poly_cuda_copy_htod(unsigned long long dst, const void *src, size_t bytes);
int poly_cuda_copy_dtoh(void *dst, unsigned long long src, size_t bytes);
int poly_cuda_copy_dtod(unsigned long long dst, unsigned long long src, size_t bytes);
PolyCudaProgram *poly_compile_cuda(const char *source, const char *fn_name);
/* Optional status distinguishes CompileError (-2) from driver/runtime
 * rejection (-1), matching strict BEAM's exception boundary; success is0. */
PolyCudaProgram *poly_compile_cuda_with_binary(
    const char *source,
    const char *fn_name,
    uint8_t **binary,
    int *size,
    int *status
);
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
);
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
);
int poly_cuda_sync(void);
void poly_cuda_program_destroy(PolyCudaProgram *prog);
int poly_cuda_memset(unsigned long long ptr, unsigned char val, size_t bytes);
#endif /* POLY_HAS_CUDA */

/* HIP/ROCm support (conditional on POLY_HAS_HIP) */

#ifdef POLY_HAS_HIP

/* HIP linearizer: rewrite + gpudims + linearize (same pipeline as CUDA). */
PolyUOp *poly_rewrite_hip(PolyCtx *ctx, PolyUOp *sink);
PolyUOp **poly_linearize_hip(PolyCtx *ctx, PolyUOp *sink, int *n_out);

/* Render linearized UOps to HIP C++ source code.
 * Returns malloc'd string. Caller must free(). */
char *poly_render_hip(
    PolyCtx *ctx,
    PolyUOp **uops,
    int n,
    const char *fn_name,
    int launch_bounds,
    const char *arch
);

/* HIP Runtime */
typedef struct PolyHipProgram PolyHipProgram;

int poly_hip_init(void);
bool poly_hip_available(void);
int poly_hip_wave_size(void);
const char *poly_hip_arch(void);
void *poly_hip_alloc(size_t bytes);
void poly_hip_free(void *ptr);
int poly_hip_copy_htod(void *dst, const void *src, size_t bytes);
int poly_hip_copy_dtoh(void *dst, const void *src, size_t bytes);
PolyHipProgram *poly_compile_hip(const char *source, const char *fn_name);
int poly_hip_launch(
    PolyHipProgram *prog,
    void **args,
    int n_args,
    int gx,
    int gy,
    int gz,
    int bx,
    int by,
    int bz
);
int poly_hip_sync(void);
void poly_hip_program_destroy(PolyHipProgram *prog);
int poly_hip_memset(void *ptr, unsigned char val, size_t bytes);

#endif /* POLY_HAS_HIP */

#endif /* POLY_CODEGEN_H */
