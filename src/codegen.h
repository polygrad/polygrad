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

/* Tensor core spec -- port of tinygrad tc.py TensorCore dataclass.
 * Describes a hardware matrix multiply-accumulate instruction.
 * D(M,N) = A(M,K) * B(K,N) + C(M,N) across a warp of threads. */
#define POLY_TC_MAX_OPTS 8
#define POLY_TC_MAX_SWIZZLE 8

typedef struct {
  int dims[3]; /* N, M, K */
  int threads; /* warp size (64 for CDNA, 32 for CUDA/RDNA) */
  int elements_per_thread[3]; /* per-thread elements for A, B, C */
  PolyDType dtype_in; /* A and B input dtype */
  PolyDType dtype_out; /* C and D output dtype */
  struct {
    char type;
    int dim;
  } opts[POLY_TC_MAX_OPTS]; /* 'l'=LOCAL, 'u'=UPCAST */
  int n_opts;
  const char *swizzle[2][3][POLY_TC_MAX_SWIZZLE]; /* [input][local/upcast/reduce][axis] */
  int swizzle_len[2][3];
  const char *intrinsic_name; /* e.g. "mfma_f32_16x16x16f16" */
} PolyTensorCore;

/* Renderer capability flags -- determines which ops survive into rendered code.
 * Mirrors tinygrad's code_for_op / supported_ops gating in get_late_rewrite_patterns.
 * max_vec_width is a polygrad extension (tinygrad uses boolean supports_float4). */
typedef struct {
  bool has_mulacc; /* Backend supports fused multiply-add (MULACC -> fmaf/fma) */
  bool has_max; /* Backend renders MAX directly */
  bool has_threefry; /* Backend supports native THREEFRY op without decomposition */
  bool has_exp2; /* Backend renders EXP2 directly */
  bool has_log2; /* Backend renders LOG2 directly */
  bool has_sin; /* Backend renders SIN directly */
  bool has_fdiv; /* Backend renders FDIV and therefore lowers RECIPROCAL to it */
  bool has_int64; /* Backend renders signed/unsigned 64-bit integer values directly */
  bool has_local; /* Backend supports local/workgroup scheduling */
  bool has_threads; /* Backend supports CPU-style core_id runtime threading */
  bool has_simd_int; /* Backend supports packed integer ops in vector regs (vpaddd etc) */
  bool has_simd_float; /* Backend supports packed f32 vector ops in native vector regs */
  int max_vec_width; /* Max elements in VECTORIZE (0=scalar-only, 4=SSE, 8=AVX2) */
  int max_threads; /* Max CPU worker threads for THREAD axes (0=disabled) */
  int global_max[3]; /* GPU SPECIAL global/workgroup caps, 0 = unbounded */
  int local_max[3]; /* GPU SPECIAL local/workgroup caps, 0 = unbounded */
  const PolyTensorCore *tensor_cores; /* array of available TC specs (NULL if none) */
  int n_tensor_cores; /* number of TC specs */
} PolyRendererCaps;

/* Optimization policy: explicit discriminator for what the heuristic does.
 * Wrappers set this directly instead of inferring from caps. */
typedef enum {
  POLY_OPT_HEURISTIC = 0, /* CPU: full heuristic (masked upcast, stride upcast, reduce unroll) */
  POLY_OPT_TC_ONLY, /* GPU: TC detection only, no CPU-oriented scheduling */
} PolyOptPolicy;

typedef struct {
  bool optimize; /* tinygrad optimize path (UPCAST/UNROLL + late pipeline) */
  int devectorize; /* tinygrad DEVECTORIZE level (0/1/2) */
  int beam_width; /* BEAM search width (0 = heuristic, >0 = BEAM search) */
  PolyRendererCaps caps; /* renderer capabilities (zero-init = conservative/unsupported) */
  /* Renderer config for unified pipeline (Phase 4) */
  int device; /* PolyDevice from engine/schedule.h (0 = CPU) */
  PolyOptPolicy opt_policy; /* explicit optimization strategy */
  /* Pinned tinygrad pm_dtype_decomps: unsupported dtype legalization runs
   * bottom-up before transcendental and renderer-final rewrites. */
  PolyPatternMatcher *dtype_matcher;
  PolyPatternMatcher *extra_matcher; /* renderer-specific final rewrite (NULL = none) */
  int gpu_block_size; /* group_for_reduce block size (0 = skip) */
} PolyRewriteOpts;

static inline bool poly_kernel_optimize_enabled(PolyUOp *sink) {
  return !(sink && sink->op == POLY_OP_SINK && sink->tag_arg.kind == POLY_ARG_BOOL &&
           sink->tag_arg.b == false);
}

static inline const char *poly_kernel_name(PolyUOp *sink, const char *fallback) {
  if (sink && sink->op == POLY_OP_SINK && sink->arg.kind == POLY_ARG_STRING && sink->arg.str &&
      sink->arg.str[0])
    return sink->arg.str;
  return fallback ? fallback : "test";
}

/* Linearize: full codegen pipeline + priority-based toposort.
 * Returns malloc'd array of UOp pointers in execution order.
 * Caller must free() the returned array. */
PolyUOp **poly_linearize(PolyCtx *ctx, PolyUOp *sink, int *n_out);
PolyUOp **poly_linearize_ex(PolyCtx *ctx, PolyUOp *sink, PolyRewriteOpts opts, int *n_out);
/* Like poly_linearize but reads POLY_OPTIMIZE/POLY_DEVECTORIZE from env. */
PolyUOp **poly_linearize_env(PolyCtx *ctx, PolyUOp *sink, int *n_out);
/* WASM linearizer: use the shared late pipeline, preserving only the native
 * renderer-capability vector subset (currently f32x4 ALU/compare/compare-mask
 * WHERE) and scalarizing unsupported semantic vectors before rendering. */
PolyUOp *poly_rewrite_wasm(PolyCtx *ctx, PolyUOp *sink);
PolyUOp *poly_rewrite_wasm_env(PolyCtx *ctx, PolyUOp *sink);
PolyUOp **poly_linearize_wasm(PolyCtx *ctx, PolyUOp *sink, int *n_out);
PolyUOp **poly_linearize_wasm_env(PolyCtx *ctx, PolyUOp *sink, int *n_out);

/* Default CPU/C renderer capabilities, matching tinygrad's ClangRenderer:
 * float4-capable, no local memory scheduling, no native MULACC/THREEFRY. */
PolyRendererCaps poly_c_renderer_caps(void);

/* Linearize an already-rewritten sink (skip full_rewrite_to_sink).
 * Used by GPU backends that insert passes between rewrite and linearization. */
PolyUOp **poly_linearize_rewritten(PolyCtx *ctx, PolyUOp *sink, int *n_out);

/* GPU dims: replace outermost RANGE with SPECIAL thread indices.
 * Returns rewritten sink (or original if no suitable RANGE found). */
PolyUOp *poly_add_gpudims(PolyCtx *ctx, PolyUOp *sink);
PolyUOp *poly_add_gpudims_ex(PolyCtx *ctx, PolyUOp *sink, PolyRendererCaps caps);

/* Port of tinygrad's pm_add_control_flow: inject predecessor edges as
 * real RANGE sources so loop nesting is structural in the DAG.
 * Applied after full_rewrite_to_sink, before linearize. */
PolyUOp *poly_apply_control_flow(PolyCtx *ctx, PolyUOp *sink);

/* Apply heuristic optimizer only (no expander/decomp). For testing TC detection
 * at the pre-expander stage where CONTRACT/UNROLL/WMMA are still visible. */
PolyUOp *poly_apply_opts_heuristic_ex(PolyCtx *ctx, PolyUOp *sink, PolyRendererCaps caps);

/* Apply only the TC detection part of the heuristic. Does not apply CPU-oriented
 * upcast/unroll heuristics. For GPU linearizers that handle their own scheduling. */
PolyUOp *poly_apply_tc_opt(PolyCtx *ctx, PolyUOp *sink, PolyRendererCaps caps);

/* TensorCore helpers (internal, exposed for testing) */
int poly_tc_get_reduce_axes(const PolyTensorCore *tc, int out[][2]);
int poly_tc_count_local(const PolyTensorCore *tc);
int poly_tc_count_upcast(const PolyTensorCore *tc);
int poly_tc_base_shape_str(const PolyTensorCore *tc, const char *out[], int max_n);
int poly_tc_base_upcast_axes(const PolyTensorCore *tc, const char *out[], int max_n);
void poly_tc_permute_for_shape_str(
    const PolyTensorCore *tc,
    int swz_idx,
    const char *shape_str[],
    int n_shape,
    int perm[],
    int max_n
);

/* Walk through transparent pointer casts to find an INDEX. Do not return
 * late-codegen SHRINK: it has distinct width semantics. */
static inline PolyUOp *poly_find_index_through_cast(PolyUOp *u) {
  while (u && (u->op == POLY_OP_CAST || u->op == POLY_OP_BITCAST) && u->n_src > 0 && u->dtype.is_ptr
  )
    u = u->src[0];
  return (u && u->op == POLY_OP_INDEX) ? u : NULL;
}

/* Memory-addressing form accepted by renderers after tinygrad-style
 * pm_index_is_shrink. SHRINK(src, idx, width) is a late codegen slice, not a
 * frontend movement op. */
static inline PolyUOp *poly_find_memory_slice_through_cast(PolyUOp *u) {
  while (u && (u->op == POLY_OP_CAST || u->op == POLY_OP_BITCAST) && u->n_src > 0 && u->dtype.is_ptr
  )
    u = u->src[0];
  return (u && (u->op == POLY_OP_INDEX || u->op == POLY_OP_SHRINK)) ? u : NULL;
}

static inline bool poly_is_program_memory_base(const PolyUOp *u) {
  if (!u) return false;
  return u->dtype.is_ptr || u->op == POLY_OP_PARAM || u->op == POLY_OP_BUFFER ||
         u->op == POLY_OP_DEFINE_LOCAL || u->op == POLY_OP_DEFINE_REG || u->op == POLY_OP_AFTER;
}

/* Pinned tinygrad UOp.addrspace/max_numel and ParamArg.slot program-IR
 * accessors (uop/ops.py:384,786-799; renderer/cstyle.py:154-178). The final
 * pm_remove_vec_dtypes representation stores address space and slot in
 * ParamArg and the storage extent in src[0], after pointer dtype metadata has
 * been removed. Legacy DEFINE_* and pointer forms remain readable until their
 * independent vocabulary migrations close. */
static inline PolyAddrSpace poly_program_memory_addrspace(const PolyUOp *u) {
  while (u) {
    if (u->op == POLY_OP_DEFINE_LOCAL) return POLY_ADDR_LOCAL;
    if (u->op == POLY_OP_DEFINE_REG) return POLY_ADDR_REG;
    if (u->op == POLY_OP_PARAM || u->op == POLY_OP_BUFFER) {
      if (u->arg.kind == POLY_ARG_PARAM && u->arg.param) return u->arg.param->addrspace;
      return u->dtype.is_ptr ? u->dtype.addrspace : POLY_ADDR_GLOBAL;
    }
    if ((u->op == POLY_OP_INDEX || u->op == POLY_OP_SHRINK || u->op == POLY_OP_AFTER ||
         u->op == POLY_OP_CAST || u->op == POLY_OP_BITCAST || u->op == POLY_OP_GEP ||
         u->op == POLY_OP_STORE) &&
        u->n_src > 0) {
      u = u->src[0];
      continue;
    }
    return POLY_ADDR_GLOBAL;
  }
  return POLY_ADDR_GLOBAL;
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
  if (u->n_src == 1 && u->src[0] && u->src[0]->op == POLY_OP_CONST &&
      u->src[0]->arg.kind == POLY_ARG_INT && u->src[0]->arg.i > 0)
    return u->src[0]->arg.i;
  return u->dtype.is_ptr && u->dtype.ptr_size > 0 ? u->dtype.ptr_size : 1;
}

static inline PolyDType poly_program_buffer_dtype(const PolyUOp *u) {
  PolyDType dtype = u ? u->dtype : POLY_VOID;
  dtype.is_ptr = false;
  dtype.addrspace = POLY_ADDR_GLOBAL;
  dtype.vcount = 0;
  dtype.ptr_size = 0;
  return dtype;
}

/* Codegen pipeline: full rewrite to sink (sym → reduce → decomp → transcendental) */
PolyUOp *poly_full_rewrite_to_sink(PolyCtx *ctx, PolyUOp *sink);
PolyUOp *poly_full_rewrite_to_sink_ex(PolyCtx *ctx, PolyUOp *sink, PolyRewriteOpts opts);

/* Range start offset for ops that have trailing RANGE sources.
 * Returns index of first RANGE source, or -1 if op has no range sources. */
int range_start_for_op(PolyOps op);

/* Individual codegen pass getters (for GPU linearizer to insert passes between them).
 * poly_symbolic_simple()/poly_symbolic() are declared in pat.h. */
PolyPatternMatcher *poly_pm_reduce_pass(void);
PolyPatternMatcher *poly_pm_decomp_pass(void);
PolyPatternMatcher *poly_pm_decomp_pass_caps(PolyRendererCaps caps);
PolyPatternMatcher *poly_pm_transcendental_pass(void);
PolyPatternMatcher *poly_pm_move_where_on_load_pass(void);
/* Late codegen stage parity hook: tinygrad pm_add_loads from
 * codegen/late/devectorizer.py. Exposed so focused tests/probes can isolate the
 * post-gpudims add-loads boundary before render/devectorize. */
PolyPatternMatcher *poly_pm_add_loads_pass(void);
PolyPatternMatcher *poly_pm_bf16_non_native(void);
PolyPatternMatcher *poly_pm_f16_non_native(void);
PolyPatternMatcher *poly_pm_bf16_renderer_extra(void);
PolyPatternMatcher *poly_pm_c_renderer_extra(void);
PolyPatternMatcher *poly_pm_pre_expander_pass(void);
PolyPatternMatcher *poly_pm_expander_pass(void);
PolyPatternMatcher *poly_pm_devectorize_pass(void);
PolyPatternMatcher *poly_pm_render_subset_pass(void);
PolyPatternMatcher *poly_pm_split_ends_pass(void);
/* Tinygrad-aligned late codegen stage helpers used by probes/tests.
 * These apply the same stage slices as full_rewrite_to_sink_ex instead of
 * exposing a single internal matcher under a misleading boundary name. */
PolyUOp *poly_apply_devectorize_stage(
    PolyCtx *ctx,
    PolyUOp *sink,
    int devectorize,
    PolyRendererCaps caps
);
PolyUOp *poly_apply_post_index_symbolic_stage(PolyCtx *ctx, PolyUOp *sink, int devectorize);
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
uint8_t *poly_render_wasm(PolyUOp **uops, int n, int *size_out, bool use_simd);
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

/* Free a compiled program (dlclose + cleanup). */
void poly_program_destroy(PolyProgram *prog);

/* CUDA support (conditional on POLY_HAS_CUDA) */

#ifdef POLY_HAS_CUDA

/* CUDA linearizer: rewrite + gpudims + linearize. */
PolyUOp *poly_rewrite_cuda(PolyCtx *ctx, PolyUOp *sink);
PolyUOp **poly_linearize_cuda(PolyCtx *ctx, PolyUOp *sink, int *n_out);

/* Render linearized UOps to CUDA C source code.
 * Returns malloc'd string. Caller must free(). */
char *poly_render_cuda(PolyUOp **uops, int n, const char *fn_name, int launch_bounds);

/* CUDA Runtime */
typedef struct PolyCudaProgram PolyCudaProgram;
typedef struct PolyCudaGraph PolyCudaGraph;

/* Pinned tinygrad runtime/graph/cuda.py CUDAGraph kernel-node boundary.
 * `args` uses PolyRunner's prepared ABI: raw device handles first, followed
 * by pointers to scalar values. Dependency indices address earlier specs. */
typedef struct PolyCudaGraphKernelSpec {
  PolyCudaProgram *program;
  void **args;
  int n_buffer_args;
  int n_args;
  int grid[3];
  int block[3];
  const int *dependencies;
  int n_dependencies;
} PolyCudaGraphKernelSpec;

int poly_cuda_init(void);
bool poly_cuda_available(void);
int poly_cuda_arch_major(void);
unsigned long long poly_cuda_alloc(size_t bytes);
void poly_cuda_free(unsigned long long ptr);
int poly_cuda_copy_htod(unsigned long long dst, const void *src, size_t bytes);
int poly_cuda_copy_dtoh(void *dst, unsigned long long src, size_t bytes);
int poly_cuda_copy_dtod(unsigned long long dst, unsigned long long src, size_t bytes);
PolyCudaProgram *poly_compile_cuda(const char *source, const char *fn_name);
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
bool poly_cuda_graph_available(void);
PolyCudaGraph *poly_cuda_graph_create(const PolyCudaGraphKernelSpec *specs, int n_specs);
int poly_cuda_graph_update(PolyCudaGraph *graph, const PolyCudaGraphKernelSpec *specs, int n_specs);
int poly_cuda_graph_launch(PolyCudaGraph *graph);
void poly_cuda_graph_destroy(PolyCudaGraph *graph);

#endif /* POLY_HAS_CUDA */

/* HIP/ROCm support (conditional on POLY_HAS_HIP) */

#ifdef POLY_HAS_HIP

/* HIP linearizer: rewrite + gpudims + linearize (same pipeline as CUDA). */
PolyUOp *poly_rewrite_hip(PolyCtx *ctx, PolyUOp *sink);
PolyUOp **poly_linearize_hip(PolyCtx *ctx, PolyUOp *sink, int *n_out);

/* Render linearized UOps to HIP C++ source code.
 * Returns malloc'd string. Caller must free(). */
char *poly_render_hip(PolyUOp **uops, int n, const char *fn_name, int launch_bounds);

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

/* x86 ISA backend (tinygrad X86Renderer parity, conditional on POLY_HAS_X86) */

#ifdef POLY_HAS_X86

PolyUOp *poly_rewrite_x86(PolyCtx *ctx, PolyUOp *sink);
PolyUOp **poly_linearize_x86_rewritten(PolyCtx *ctx, PolyUOp *sink, int *n_out);
PolyUOp **poly_linearize_x86(PolyCtx *ctx, PolyUOp *sink, int *n_out);
uint8_t *poly_render_x86(PolyUOp **uops, int n, int *size_out);
char *poly_render_x86_source(PolyUOp **uops, int n);
uint32_t poly_x86_feature_stamp(void);

typedef struct PolyX86Program PolyX86Program;

PolyX86Program *poly_compile_x86(const uint8_t *code, int code_size);
PolyX86Program *poly_compile_x86_source(const char *source);
int poly_x86_program_call(PolyX86Program *prog, void **args, int n_args);
int poly_x86_program_call_core(PolyX86Program *prog, void **args, int n_args, int core_id);
int poly_x86_program_call_threaded(PolyX86Program *prog, void **args, int n_args, int threads);
void poly_x86_program_destroy(PolyX86Program *prog);

#endif /* POLY_HAS_X86 */

#endif /* POLY_CODEGEN_H */
