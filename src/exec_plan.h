/*
 * exec_plan.h -- Backend-neutral and backend-specific execution plan types
 *
 * Defines the type system for the cross-platform execution architecture:
 *
 *   Artifact (bundle file)
 *     -> PolyInstance (loaded model + mutable execution state)
 *       -> PolyPreparedStep (backend-neutral scheduled exec items)
 *         -> PolyExecutableStep (backend-specific compiled runners)
 *
 * Phase 1: types only, no constructors or behavior.
 */

#ifndef POLY_EXEC_PLAN_H
#define POLY_EXEC_PLAN_H

#include "polygrad.h" /* PolyDType (by value), PolyCtx/PolyUOp (forward-declared) */

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declaration -- defined in frontend.h, pointer-only usage here */
struct PolyVarBinding;

/* ── Device identity ─────────────────────────────────────────────────── */

typedef enum {
  POLY_DEVICE_AUTO = 0, /* resolved at set_device time, never stored */
  POLY_DEVICE_CPU,      /* host C compiler (fork + clang/gcc + dlopen) */
  POLY_DEVICE_INTERP,   /* interpreter: walks scheduled graph, no codegen */
  POLY_DEVICE_CUDA,     /* NVIDIA GPU (PTX via cuModuleLoad) */
  POLY_DEVICE_WASM_JIT, /* WASM: C renders bytes, host compiles+executes */
  POLY_DEVICE_WEBGPU,   /* WebGPU: C renders WGSL, host compiles+executes */
} PolyDeviceId;

/* ── Compilation mode ────────────────────────────────────────────────── */

typedef enum {
  POLY_MODE_CALL = 0,       /* forward-only execution */
  POLY_MODE_VALUE_AND_GRAD, /* forward + backward (value + gradient) */
} PolyCompileMode;

/* ── Exec item classification ────────────────────────────────────────── */
/*
 * Only POLY_EXEC_COMPUTE is emitted by the scheduler today.
 * COPY, VIEW, ENCDEC are defined for forward compatibility -- they become
 * live when the scheduler learns to emit them (separate workstream).
 */

typedef enum {
  POLY_EXEC_COMPUTE, /* scheduled kernel: lower -> render -> compile -> run */
  POLY_EXEC_COPY,    /* buffer transfer across memory domains */
  POLY_EXEC_VIEW,    /* buffer alias with offset (noop or alias setup) */
  POLY_EXEC_ENCDEC,  /* encode/decode (HEVC, etc.) */
} PolyExecItemKind;

/* ── Runner classification ───────────────────────────────────────────── */

typedef enum {
  POLY_RUNNER_COMPILED, /* lowered + rendered + compiled compute kernel */
  POLY_RUNNER_COPY,     /* buffer copy between memory domains */
  POLY_RUNNER_VIEW,     /* buffer view (noop or alias setup) */
  POLY_RUNNER_ENCDEC,   /* encode/decode */
  POLY_RUNNER_INTERP,   /* interpreter: walks scheduled graph directly */
} PolyRunnerKind;

/* ── Buffer handle ───────────────────────────────────────────────────── */
/*
 * Device-specific buffer reference. The ptr field is opaque:
 *   CPU/INTERP: host malloc'd pointer
 *   CUDA:       CUdeviceptr (cast to void*)
 *   WASM_JIT:   offset into Emscripten heap
 *   WEBGPU:     GPUBuffer (host-managed, wrapped)
 */

typedef struct {
  void *ptr;            /* device-specific handle */
  size_t nbytes;        /* physical byte size (handles quantized packing) */
  PolyDeviceId domain;  /* which memory domain this lives in */
  bool owned;           /* whether the instance should free this on cleanup */
} PolyBufferHandle;

/* ── Allocator interface ─────────────────────────────────────────────── */
/*
 * Per-device memory operations. Every executable step has exactly one
 * allocator corresponding to its device's memory domain.
 */

typedef struct {
  void *(*alloc)(size_t nbytes, void *dev_ctx);
  void (*free)(void *handle, void *dev_ctx);
  int (*copy_in)(void *dst_handle, const void *host_src, size_t nbytes,
                 void *dev_ctx);
  int (*copy_out)(void *host_dst, const void *src_handle, size_t nbytes,
                  void *dev_ctx);
  int (*copy_between)(void *dst_handle, const void *src_handle, size_t nbytes,
                      void *dev_ctx);
  void *dev_ctx; /* NULL for CPU, CUcontext* for CUDA, etc. */
} PolyAllocator;

/* ── Prepared buffer slot ────────────────────────────────────────────── */
/*
 * One logical buffer in a prepared step. Merges external (named/bound)
 * and intermediate buffers into a flat indexed array.
 */

typedef struct {
  PolyDType dtype;
  int64_t numel;
  int64_t nbytes;
  PolyUOp *buf_uop;       /* pointer into PolyCtx arena */
  bool is_intermediate;
  int external_buf_idx;    /* index into PolyInstance.bufs[], or -1 */
} PolyPreparedBufSlot;

/* ── Exec item spec ──────────────────────────────────────────────────── */
/*
 * One item in a prepared step. Mirrors tinygrad's ExecItem before lowering.
 *
 * For COMPUTE items, root is the *scheduled kernel SINK* -- output of
 * poly_schedule_v2(), not the output of linearization. Lowering
 * (linearize -> render -> compile) is the first backend-specific
 * operation and happens when producing an executable step.
 */

typedef struct {
  PolyExecItemKind kind;
  PolyUOp *root; /* scheduled root UOp (pre-lowering) */

  /* Buffer slot references (indices into PolyPreparedStep.buf_slots) */
  int *buf_slot_indices;
  int n_buf_slots;

  /* Fixed vars for this item (from BIND stripping) */
  struct PolyVarBinding *fixedvars;
  int n_fixedvars;
} PolyExecItemSpec;

/* ── Prepared step ───────────────────────────────────────────────────── */
/*
 * Backend-neutral execution plan. Internal only, not serialized.
 * Produced by scheduling a tensor-level SINK. Survives device changes.
 */

typedef struct {
  const char *entrypoint_name;
  PolyCompileMode mode;
  uint32_t graph_hash; /* structural hash of the entrypoint SINK */

  /* Logical buffer slots (external + intermediate) */
  PolyPreparedBufSlot *buf_slots;
  int n_buf_slots;

  /* Ordered exec items */
  PolyExecItemSpec *items;
  int n_items;

  /* Execution order (topological sort of item dependency graph) */
  int *exec_order; /* exec_order[step] = item index */

  /* Default variable bindings (from BIND nodes) */
  struct PolyVarBinding *default_vars;
  int n_default_vars;

  /* Mode-specific metadata (VALUE_AND_GRAD only) */
  int loss_buf_slot;  /* which slot is the loss output, or -1 */
  int *grad_buf_slots; /* [n_params] grad output slots, or NULL */
} PolyPreparedStep;

/* ── Runner ──────────────────────────────────────────────────────────── */
/*
 * Backend-specific compiled execution unit. One per exec item after lowering.
 *
 * handle is a backend-owned opaque payload:
 *   COMPILED (CPU):  PolyProgram*
 *   COMPILED (CUDA): PolyCudaProgram*
 *   COMPILED (WASM): WASM bytes (malloc'd, handle_size = byte length)
 *   INTERP:          PolyUOp* (scheduled root, lives in PolyCtx arena)
 *   COPY/VIEW:       NULL
 */

typedef struct {
  PolyRunnerKind kind;

  void *handle;
  int handle_size; /* byte length for rendered-only handles (WASM, WGSL) */

  /* GPU launch metadata (ignored by non-GPU backends) */
  int grid[3];
  int block[3];

  /* Maps kernel parameter index -> prepared step buf_slot index */
  int *param_to_slot;
  int n_params;

  /* Maps kernel var index -> default_vars index in prepared step */
  int *var_indices;
  int n_vars;
} PolyRunner;

/* ── Executable step ─────────────────────────────────────────────────── */
/*
 * Backend-specific lowered execution plan. Produced by lowering a
 * prepared step for a specific device. Contains compiled runners
 * and allocated intermediate buffer handles.
 */

typedef struct {
  PolyPreparedStep *prepared; /* retained reference, not owned */
  PolyDeviceId device;
  const PolyAllocator *allocator;

  PolyRunner *runners; /* [prepared->n_items], one per exec item */
  int n_runners;

  /* Runtime buffer handles for intermediates */
  PolyBufferHandle *intermediate_handles;
  int n_intermediates;
} PolyExecutableStep;

/* ── Backend descriptor ──────────────────────────────────────────────── */
/*
 * Each backend provides a runner construction hook. The seam is at
 * lowering: for each COMPUTE item, lower_item does backend-specific
 * linearize -> render -> compile (or skips all three for INTERP).
 *
 *   CPU:      poly_linearize -> poly_render_c -> poly_compile_c
 *   CUDA:     poly_linearize_cuda -> poly_render_cuda -> poly_compile_cuda
 *   WASM_JIT: poly_linearize -> poly_render_wasm -> store bytes (no compile)
 *   INTERP:   store scheduled root UOp (no linearize, no render, no compile)
 *   WEBGPU:   linearize -> poly_render_wgsl -> store source (no compile)
 */

typedef struct {
  const char *name;
  PolyDeviceId id;
  bool host_executed; /* true for WASM_JIT and WEBGPU */

  /* Lower a scheduled root into a runner. ctx needed for linearization.
   * fn_name is for debug output and cache keying.
   * Returns 0 on success, <0 on error. */
  int (*lower_item)(PolyCtx *ctx, PolyUOp *scheduled_root,
                    const char *fn_name, PolyRunner *runner_out);

  /* Execute a lowered runner. NULL for host-executed backends. */
  int (*execute)(PolyRunner *runner, void **args, int n_args);

  /* Free backend-owned resources in a runner. */
  void (*free_runner)(PolyRunner *runner);

  /* Return the allocator for this backend's memory domain. */
  const PolyAllocator *(*get_allocator)(void);
} PolyBackendDesc;

/* ── Execution status ────────────────────────────────────────────────── */
/*
 * Structured error reporting for host-executed backends.
 * Replaces sentinel return codes.
 */

typedef struct {
  int code;              /* 0 = success, <0 = error */
  char message[256];     /* human-readable error description */
  int failed_item_index; /* which exec item failed, -1 if N/A */
} PolyExecStatus;

/* ── Prepared step construction ───────────────────────────────────────── */
/*
 * Build a backend-neutral execution plan from a tensor-level SINK.
 * Calls poly_schedule_v2() internally, populates exec items from the
 * schedule result, and builds the buffer slot table.
 *
 * All items are POLY_EXEC_COMPUTE in the current implementation.
 * The prepared step stores scheduled kernel roots (pre-lowering).
 *
 * Returns NULL on error.
 */
PolyPreparedStep *poly_prepare_step(PolyCtx *ctx, PolyUOp *sink,
                                    PolyCompileMode mode);

void poly_prepared_step_free(PolyPreparedStep *step);

/* ── Executable step construction ────────────────────────────────────── */
/*
 * Lower a prepared step into a backend-specific executable for a device.
 * For each COMPUTE item, calls the backend's linearize -> render -> compile
 * pipeline and populates a PolyRunner. Allocates intermediate buffers.
 *
 * ctx is needed for linearization (arena allocation of rewritten UOps).
 * Currently only POLY_DEVICE_CPU is implemented.
 *
 * Returns NULL on error.
 */
PolyExecutableStep *poly_lower_step(PolyCtx *ctx, PolyPreparedStep *prepared,
                                    PolyDeviceId device);

/*
 * Execute a lowered step with bound buffer data.
 *
 * slot_data is indexed by prepared step buf_slot index. External slots
 * must have non-NULL data pointers. Intermediate slots are ignored
 * (the executable step owns its own intermediate buffers).
 *
 * var_bindings override the prepared step's default_vars (same var -> last wins).
 *
 * Returns 0 on success, <0 on error.
 */
int poly_executable_step_run(PolyExecutableStep *step,
                             void **slot_data, int n_slots,
                             struct PolyVarBinding *var_bindings, int n_var_bindings);

void poly_executable_step_free(PolyExecutableStep *step);

/* ── CPU allocator ───────────────────────────────────────────────────── */

extern const PolyAllocator POLY_CPU_ALLOCATOR;

#ifdef __cplusplus
}
#endif

#endif /* POLY_EXEC_PLAN_H */
