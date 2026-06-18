/*
 * polygrad.h — Public header for the polygrad C11 tensor compiler
 *
 * A C11 port of tinygrad's compiler core: UOp IR, pattern matcher,
 * scheduler, codegen, and CPU runtime.
 */

#ifndef POLYGRAD_H
#define POLYGRAD_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Ops enum mirrors tinygrad_latest's Ops ordering for shared ops.
 * The numeric order is observable through exported IR and is also used by
 * tinygrad's linearizer tuplize tiebreak, so Polygrad-only compatibility ops
 * live at the tail instead of shifting current tinygrad values. */

typedef enum {
  /* 1 — defines/special */
  POLY_OP_DEFINE_VAR = 1,
  POLY_OP_BIND,
  POLY_OP_SPECIAL,
  POLY_OP_DEFINE_LOCAL,
  POLY_OP_DEFINE_REG,

  /* 2 — non-op uops */
  POLY_OP_NOOP,
  POLY_OP_REWRITE_ERROR,
  POLY_OP_PARAM,
  POLY_OP_FUNCTION,
  POLY_OP_CALL,
  POLY_OP_PROGRAM,
  POLY_OP_LINEAR,
  POLY_OP_SOURCE,
  POLY_OP_BINARY,
  POLY_OP_SINK,
  POLY_OP_AFTER,
  POLY_OP_GROUP,
  POLY_OP_GEP,
  POLY_OP_STACK,
  POLY_OP_VECTORIZE = POLY_OP_STACK, /* source-compatible old name */
  POLY_OP_TUPLE,
  POLY_OP_GETTUPLE,

  /* 3 — load/store */
  POLY_OP_INDEX,
  POLY_OP_LOAD,
  POLY_OP_STORE,

  /* 4 — math */
  POLY_OP_WMMA,
  POLY_OP_SHAPED_WMMA,
  /* unary */
  POLY_OP_CAST,
  POLY_OP_BITCAST,
  POLY_OP_EXP2,
  POLY_OP_LOG2,
  POLY_OP_SIN,
  POLY_OP_SQRT,
  POLY_OP_RECIPROCAL,
  POLY_OP_NEG,
  POLY_OP_TRUNC,
  /* binary */
  POLY_OP_ADD,
  POLY_OP_MUL,
  POLY_OP_SHL,
  POLY_OP_SHR,
  POLY_OP_IDIV,
  POLY_OP_MAX,
  POLY_OP_MOD,
  POLY_OP_CMPLT,
  POLY_OP_CMPNE,
  POLY_OP_CMPEQ,
  POLY_OP_XOR,
  POLY_OP_OR,
  POLY_OP_AND,
  POLY_OP_THREEFRY,
  POLY_OP_SUB,
  POLY_OP_FDIV,
  POLY_OP_POW,
  /* ternary */
  POLY_OP_WHERE,
  POLY_OP_MULACC,

  /* 5 — control flow / consts / custom */
  POLY_OP_BARRIER,
  POLY_OP_RANGE,
  POLY_OP_IF,
  POLY_OP_END,
  POLY_OP_ENDIF,
  POLY_OP_VCONST,
  POLY_OP_CONST,
  POLY_OP_CUSTOM,
  POLY_OP_CUSTOMI,
  POLY_OP_INS,

  /* 6 — ops that don't exist in programs */
  POLY_OP_UNIQUE,
  POLY_OP_DEVICE,
  POLY_OP_LUNIQUE,
  POLY_OP_CONTIGUOUS,
  POLY_OP_CONTIGUOUS_BACKWARD,
  POLY_OP_DETACH,
  POLY_OP_BUFFERIZE,
  POLY_OP_COPY,
  POLY_OP_BUFFER,
  POLY_OP_BUFFER_VIEW,
  POLY_OP_MSELECT,
  POLY_OP_MSTACK,
  POLY_OP_CUSTOM_FUNCTION,
  POLY_OP_RESHAPE,
  POLY_OP_PERMUTE,
  POLY_OP_EXPAND,
  POLY_OP_PAD,
  POLY_OP_SHRINK,
  POLY_OP_FLIP,
  POLY_OP_MULTI,
  POLY_OP_REDUCE,
  POLY_OP_ALLREDUCE,
  POLY_OP_UNROLL,
  POLY_OP_CONTRACT,
  POLY_OP_VCAT,
  POLY_OP_PTRCAT,

  /* Polygrad-only compatibility ops. Current tinygrad spells assign as
   * AFTER(target, STORE(target, value)); REDUCE_AXIS is frontend/scheduler
   * sugar lowered before program IR; ENCDEC is not present upstream. */
  POLY_OP_ASSIGN,
  POLY_OP_ENCDEC,
  POLY_OP_REDUCE_AXIS,

  POLY_OP_COUNT /* sentinel — total number of ops */
} PolyOps;

/* GroupOp bitmask sets */
/* Each set is a uint64_t[2] bitmask (128 bits, enough for current ops). */

typedef struct {
  uint64_t bits[2];
} PolyOpSet;

extern PolyOpSet POLY_GROUP_UNARY;
extern PolyOpSet POLY_GROUP_BINARY;
extern PolyOpSet POLY_GROUP_TERNARY;
extern PolyOpSet POLY_GROUP_ALU;
extern PolyOpSet POLY_GROUP_ELEMENTWISE;
extern PolyOpSet POLY_GROUP_MOVEMENT;
extern PolyOpSet POLY_GROUP_COMMUTATIVE;
extern PolyOpSet POLY_GROUP_ASSOCIATIVE;
extern PolyOpSet POLY_GROUP_IDEMPOTENT;
extern PolyOpSet POLY_GROUP_COMPARISON;
extern PolyOpSet POLY_GROUP_UNSAFEPAD;
extern PolyOpSet POLY_GROUP_BUFFER;
extern PolyOpSet POLY_GROUP_IRREDUCIBLE;

/* Must be called before using any opset globals.
   Called automatically from poly_ctx_new(); idempotent. */
void poly_init_group_ops(void);

static inline bool poly_opset_has(PolyOpSet set, PolyOps op) {
  return (set.bits[op / 64] >> (op % 64)) & 1;
}

static inline PolyOpSet poly_opset_add(PolyOpSet set, PolyOps op) {
  set.bits[op / 64] |= (uint64_t)1 << (op % 64);
  return set;
}

static inline PolyOpSet poly_opset_union(PolyOpSet a, PolyOpSet b) {
  return (PolyOpSet){{a.bits[0] | b.bits[0], a.bits[1] | b.bits[1]}};
}

static inline bool poly_opset_empty(PolyOpSet set) {
  return set.bits[0] == 0 && set.bits[1] == 0;
}

static inline PolyOpSet poly_opset_intersect(PolyOpSet a, PolyOpSet b) {
  return (PolyOpSet){{a.bits[0] & b.bits[0], a.bits[1] & b.bits[1]}};
}

static inline bool poly_opset_subset(PolyOpSet sub, PolyOpSet super) {
  return (sub.bits[0] & ~super.bits[0]) == 0 && (sub.bits[1] & ~super.bits[1]) == 0;
}

const char *poly_op_name(PolyOps op);

/* DType */

typedef enum {
  POLY_ADDR_GLOBAL = 0,
  POLY_ADDR_LOCAL,
  POLY_ADDR_REG,
} PolyAddrSpace;

typedef struct {
  int8_t priority;
  uint16_t bitsize;
  const char *name; /* C type name, e.g. "float", "int" */
  char fmt; /* struct pack format char, 0 if none */
  uint16_t count; /* vector width, 1 for scalars */
  /* pointer fields (0/defaults for non-pointer dtypes) */
  bool is_ptr;
  PolyAddrSpace addrspace;
  uint16_t vcount; /* pointer vector count */
  int64_t ptr_size; /* -1 = unlimited */
} PolyDType;

/* Predefined scalar dtypes */
extern const PolyDType POLY_VOID;
extern const PolyDType POLY_INDEX;
extern const PolyDType POLY_BOOL;
extern const PolyDType POLY_INT8;
extern const PolyDType POLY_UINT8;
extern const PolyDType POLY_INT16;
extern const PolyDType POLY_UINT16;
extern const PolyDType POLY_INT32;
extern const PolyDType POLY_UINT32;
extern const PolyDType POLY_INT64;
extern const PolyDType POLY_UINT64;
extern const PolyDType POLY_FLOAT16;
extern const PolyDType POLY_BFLOAT16;
extern const PolyDType POLY_FLOAT32;
extern const PolyDType POLY_FLOAT64;

bool poly_dtype_eq(PolyDType a, PolyDType b);
bool poly_dtype_is_float(PolyDType dt);
bool poly_dtype_is_int(PolyDType dt);
bool poly_dtype_is_index(PolyDType dt);
bool poly_dtype_is_unsigned(PolyDType dt);
bool poly_dtype_is_bool(PolyDType dt);
PolyDType poly_dtype_scalar(PolyDType dt);
PolyDType poly_dtype_vec(PolyDType dt, int sz);
PolyDType poly_dtype_ptr(PolyDType dt, int64_t size, PolyAddrSpace addrspace);
int poly_dtype_itemsize(PolyDType dt);
const char *poly_dtype_name(PolyDType dt);
int poly_dtype_count(void);
bool poly_dtype_by_id(int id, PolyDType *out);
int poly_dtype_id_by_name(const char *name);

/* Axis metadata for RANGE args (tinygrad AxisType parity) */

typedef enum {
  POLY_AXIS_GLOBAL = 0,
  POLY_AXIS_WARP,
  POLY_AXIS_LOCAL,
  POLY_AXIS_LOOP,
  POLY_AXIS_GROUP_REDUCE,
  POLY_AXIS_REDUCE,
  POLY_AXIS_UPCAST,
  POLY_AXIS_UNROLL,
  POLY_AXIS_THREAD,
  POLY_AXIS_PLACEHOLDER,
} PolyAxisType;

/* PolyArg — tagged union for UOp arg field */

typedef enum {
  POLY_ARG_NONE = 0,
  POLY_ARG_INT,
  POLY_ARG_FLOAT,
  POLY_ARG_BOOL,
  POLY_ARG_INT_TUPLE,
  POLY_ARG_PAIR_TUPLE, /* array of (int64_t, int64_t) pairs */
  POLY_ARG_STRING,
  POLY_ARG_OPS,
  POLY_ARG_REDUCE_AXIS, /* (PolyOps, int64_t[], n) */
  POLY_ARG_RANGE, /* (axis_id, axis_type, extra...) */
  POLY_ARG_DEFINE_VAR, /* (name, min_val, max_val) */
  POLY_ARG_BUFFERIZE_OPTS, /* (device, addrspace, removable) */
  POLY_ARG_INVALID,
} PolyArgKind;

typedef struct {
  PolyArgKind kind;
  union {
    int64_t i;
    double f;
    bool b;
    struct {
      int64_t *vals;
      int n;
    } int_tuple;
    struct {
      int64_t (*pairs)[2];
      int n;
    } pair_tuple;
    const char *str;
    PolyOps ops;
    struct {
      PolyOps op;
      int64_t *axes;
      int n;
    } reduce_axis;
    struct {
      int64_t axis_id;
      PolyAxisType axis_type;
      int64_t *extra;
      int n_extra;
    } range;
    struct {
      const char *name;
      int64_t min_val;
      int64_t max_val;
    } define_var;
    struct {
      int32_t device; /* PolyDevice, kept int32_t because PolyDevice is declared later. */
      PolyAddrSpace addrspace;
      bool removable;
    } bufferize_opts;
  };
} PolyArg;

static inline PolyArg poly_arg_none(void) {
  return (PolyArg){.kind = POLY_ARG_NONE};
}
static inline PolyArg poly_arg_int(int64_t v) {
  return (PolyArg){.kind = POLY_ARG_INT, .i = v};
}
static inline PolyArg poly_arg_float(double v) {
  return (PolyArg){.kind = POLY_ARG_FLOAT, .f = v};
}
static inline PolyArg poly_arg_bool(bool v) {
  return (PolyArg){.kind = POLY_ARG_BOOL, .b = v};
}
static inline PolyArg poly_arg_ops(PolyOps op) {
  return (PolyArg){.kind = POLY_ARG_OPS, .ops = op};
}
static inline PolyArg poly_arg_invalid(void) {
  return (PolyArg){.kind = POLY_ARG_INVALID};
}
static inline PolyArg poly_arg_str(const char *s) {
  return (PolyArg){.kind = POLY_ARG_STRING, .str = s};
}
static inline PolyArg poly_arg_range(int64_t axis_id, PolyAxisType axis_type) {
  return (PolyArg
  ){.kind = POLY_ARG_RANGE,
    .range = {.axis_id = axis_id, .axis_type = axis_type, .extra = NULL, .n_extra = 0}};
}
static inline PolyArg poly_arg_range_ex(
    int64_t axis_id,
    PolyAxisType axis_type,
    int64_t *extra,
    int n_extra
) {
  return (PolyArg
  ){.kind = POLY_ARG_RANGE,
    .range = {.axis_id = axis_id, .axis_type = axis_type, .extra = extra, .n_extra = n_extra}};
}

static inline PolyArg poly_arg_define_var(const char *name, int64_t min_val, int64_t max_val) {
  return (PolyArg
  ){.kind = POLY_ARG_DEFINE_VAR,
    .define_var = {.name = name, .min_val = min_val, .max_val = max_val}};
}
/* tinygrad BufferizeOpts equivalent.
 *
 * device is the already-derived physical device for the intermediate buffer
 * (or POLY_DEVICE_AUTO when unknown), addrspace selects global/local storage,
 * and removable preserves the rangeify optimization contract for eliminating
 * redundant temporary buffers.
 */
static inline PolyArg poly_arg_bufferize_opts(
    int32_t device,
    PolyAddrSpace addrspace,
    bool removable
) {
  return (PolyArg
  ){.kind = POLY_ARG_BUFFERIZE_OPTS,
    .bufferize_opts = {.device = device, .addrspace = addrspace, .removable = removable}};
}

/* RANGE metadata helpers. New RANGE UOps store POLY_ARG_RANGE; poly_uop()
 * canonicalizes legacy POLY_ARG_INT range ids to LOOP ranges at creation. */
static inline int64_t poly_range_axis_id(PolyArg a) {
  if (a.kind == POLY_ARG_RANGE) return a.range.axis_id;
  if (a.kind == POLY_ARG_INT) return a.i;
  return -1;
}
static inline PolyAxisType poly_range_axis_type(PolyArg a) {
  if (a.kind == POLY_ARG_RANGE) return a.range.axis_type;
  return POLY_AXIS_LOOP;
}
static inline int poly_range_n_extra(PolyArg a) {
  return (a.kind == POLY_ARG_RANGE) ? a.range.n_extra : 0;
}
static inline int64_t *poly_range_extra(PolyArg a) {
  return (a.kind == POLY_ARG_RANGE) ? a.range.extra : NULL;
}
static inline bool poly_arg_is_range(PolyArg a) {
  return a.kind == POLY_ARG_RANGE || a.kind == POLY_ARG_INT;
}

/* Compatibility helpers: legacy BUFFERIZE arg kind was POLY_ARG_INT with
 * 1=removable and 0=must materialize. New code should use
 * POLY_ARG_BUFFERIZE_OPTS to match tinygrad's BufferizeOpts carrier. */
static inline bool poly_bufferize_arg_removable(PolyArg a) {
  if (a.kind == POLY_ARG_BUFFERIZE_OPTS) return a.bufferize_opts.removable;
  if (a.kind == POLY_ARG_INT) return a.i == 1;
  return false;
}
static inline PolyAddrSpace poly_bufferize_arg_addrspace(PolyArg a) {
  if (a.kind == POLY_ARG_BUFFERIZE_OPTS) return a.bufferize_opts.addrspace;
  return POLY_ADDR_GLOBAL;
}
static inline int32_t poly_bufferize_arg_device(PolyArg a) {
  if (a.kind == POLY_ARG_BUFFERIZE_OPTS) return a.bufferize_opts.device;
  return 0; /* POLY_DEVICE_AUTO */
}

bool poly_arg_eq(PolyArg a, PolyArg b);
uint32_t poly_arg_hash(PolyArg a);

/* Arena allocator */

typedef struct PolyArena PolyArena;

PolyArena *poly_arena_new(size_t initial_cap);
void *poly_arena_alloc(PolyArena *a, size_t size, size_t align);
void poly_arena_reset(PolyArena *a);
void poly_arena_destroy(PolyArena *a);
size_t poly_arena_used(PolyArena *a);

/* Hash map (for CSE) */

typedef struct PolyMap PolyMap;

PolyMap *poly_map_new(size_t initial_cap);
void poly_map_destroy(PolyMap *m);
void *poly_map_get(
    PolyMap *m,
    uint32_t hash,
    const void *key,
    bool (*eq)(const void *a, const void *b)
);
void poly_map_set(
    PolyMap *m,
    uint32_t hash,
    const void *key,
    void *value,
    bool (*eq)(const void *a, const void *b)
);
void poly_map_remove(
    PolyMap *m,
    uint32_t hash,
    const void *key,
    bool (*eq)(const void *a, const void *b)
);
size_t poly_map_len(PolyMap *m);
void poly_map_clear(PolyMap *m);

/* Iterate over all entries in the map.
 * Callback receives (key, value, userdata) for each occupied slot. */
typedef void (*PolyMapIterFn)(const void *key, void *value, void *userdata);
void poly_map_foreach(PolyMap *m, PolyMapIterFn fn, void *userdata);

/* Device identity */

typedef enum {
  POLY_DEVICE_AUTO = 0,
  POLY_DEVICE_HOST, /* imported frontend data, never a kernel target */
  POLY_DEVICE_CPU, /* native compiled CPU backend */
  POLY_DEVICE_INTERP, /* interpreter (shares CPU/WASM storage) */
  POLY_DEVICE_WASM, /* WASM JIT backend */
  POLY_DEVICE_WEBGPU, /* WebGPU GPU backend */
  POLY_DEVICE_CUDA,
  POLY_DEVICE_HIP,
  POLY_DEVICE_X64_JIT,
} PolyDevice;

/* Default compute backend for the current build */
PolyDevice poly_device_default(void);

/* Can this device execute kernels? false for HOST and AUTO */
bool poly_device_can_execute(PolyDevice dev);

/* Is ptr directly dereferenceable by the compiled core? */
bool poly_device_is_host_addressable(PolyDevice dev);

/* Do two devices use the same underlying storage domain? */
bool poly_devices_share_storage(PolyDevice a, PolyDevice b);

/* Look up device id by name. Returns POLY_DEVICE_AUTO if unknown. */
PolyDevice poly_device_by_name(const char *name);
const char *poly_device_name(PolyDevice device);

typedef struct PolyCtx PolyCtx;
typedef struct PolyUOp PolyUOp;

/* Core frontend tensor handle.
 *
 * PolyUOp stays the pure logical value graph. PolyTensor is the C-side value
 * reference used by frontends: it points at a logical UOp and carries the
 * current non-portable realization intent. The tensor carries two roots:
 * logical is the exportable/provenance expression, physical is the realized
 * execution/readback root. physical is NULL for lazy tensors. poly_tensor_uop()
 * returns the frontend convenience root: physical when present, otherwise
 * logical. Portable export/autograd provenance should use uop_logical.
 */
typedef struct PolyTensor PolyTensor;

typedef enum {
  POLY_TENSOR_VALUE = 0,
  POLY_TENSOR_PLACE = 1,
  POLY_TENSOR_BARRIER = 2,
} PolyTensorRole;

struct PolyTensor {
  PolyUOp *uop_logical;
  PolyUOp *uop_physical;
  PolyTensorRole role;
  PolyDevice device;
  uint64_t order;
  PolyTensor *source;
};

PolyTensor *poly_tensor_create(PolyCtx *ctx, PolyUOp *uop, PolyTensorRole role, PolyDevice device);
PolyTensor *poly_tensor_create_with_roots(
    PolyCtx *ctx,
    PolyUOp *uop_logical,
    PolyUOp *uop_physical,
    PolyTensorRole role,
    PolyDevice device
);
int poly_tensor_update(
    PolyCtx *ctx,
    PolyTensor *tensor,
    PolyUOp *uop_logical,
    PolyUOp *uop_physical,
    PolyTensorRole role,
    PolyDevice device
);
PolyTensor *poly_tensor_to_device(PolyCtx *ctx, PolyTensor *tensor, PolyDevice device);
PolyTensor *poly_tensor_assign(PolyCtx *ctx, PolyTensor *target, PolyTensor *value);
PolyUOp *poly_tensor_uop(PolyTensor *tensor);
PolyUOp *poly_tensor_uop_logical(PolyTensor *tensor);
PolyUOp *poly_tensor_uop_physical(PolyTensor *tensor);
PolyDevice poly_tensor_device(PolyTensor *tensor);
PolyUOp *poly_tensor_physicalize(PolyCtx *ctx, PolyTensor *tensor);
int poly_realize_tensors(PolyCtx *ctx, PolyTensor **inputs, int n, PolyTensor **outputs);

PolyDevice poly_device_from_device_uop(PolyUOp *device);
PolyDevice poly_uop_device(PolyUOp *u);

/* Frontend host-buffer lifetime hook.
 * Frontends keep strong maps keyed by the C-side PolyBuffer* address value.
 * When the core retires an imported HOST residency, it calls the registered
 * release function with that key so the frontend can drop its owner entry. */
typedef void (*PolyFrontendBufferReleaseFn)(uintptr_t buffer_key);
void poly_set_frontend_buffer_release(PolyFrontendBufferReleaseFn fn);

/* PolyBuffer is defined in device.h (needs PolyAllocator pointer) */

/* UOp */

struct PolyUOp {
  PolyOps op;
  PolyDType dtype;
  PolyUOp **src;
  uint16_t n_src;
  PolyArg arg;
  int32_t tag;
  uint32_t hash;
};

/* Cached rendered kernel (used by kernel_cache in PolyCtx) */
#define POLY_MAX_KERNEL_BUFS 64
typedef struct {
  uint8_t *bytes; /* malloc'd rendered bytes (WASM/C/etc) */
  int len; /* byte length */
  int n_bufs; /* number of buffer params */
  PolyUOp *bufs[POLY_MAX_KERNEL_BUFS]; /* ordered buffer UOps */
} PolyCachedKernel;

/* Context owns the arena, CSE cache, kernel cache, and all UOps */
PolyCtx *poly_ctx_new(void);
void poly_ctx_destroy(PolyCtx *ctx);
void poly_ctx_set_preferred_device(PolyCtx *ctx, PolyDevice device);
PolyDevice poly_ctx_get_preferred_device(PolyCtx *ctx);
bool poly_ctx_owns_ptr(PolyCtx *ctx, const void *p);
PolyMap *poly_ctx_kernel_cache(PolyCtx *ctx);

/* Return the current ctx->buffers entry address for a BUFFER UOp as an opaque
 * frontend key. Returns 0 when no PolyBuffer is attached. */
uint64_t poly_buffer_get_key(PolyCtx *ctx, PolyUOp *buf);
int poly_buffer_read(PolyCtx *ctx, PolyUOp *buf, void *dst, size_t nbytes);
PolyArena *poly_ctx_arena(PolyCtx *ctx);

/* Create a UOp (with CSE deduplication) */
PolyUOp *poly_uop(PolyCtx *ctx, PolyOps op, PolyDType dtype, PolyUOp **src, int n_src, PolyArg arg);

/* Create a UOp with a non-zero tag. Tag is part of the CSE key,
 * so a tagged node is distinct from an untagged node with the same
 * (op, dtype, src, arg). Matches tinygrad's UOp.replace(tag=...). */
PolyUOp *poly_uop_tagged(
    PolyCtx *ctx,
    PolyOps op,
    PolyDType dtype,
    PolyUOp **src,
    int n_src,
    PolyArg arg,
    int32_t tag
);

/* Convenience: create a UOp with 0, 1, 2, or 3 sources */
PolyUOp *poly_uop0(PolyCtx *ctx, PolyOps op, PolyDType dtype, PolyArg arg);
PolyUOp *poly_uop1(PolyCtx *ctx, PolyOps op, PolyDType dtype, PolyUOp *s0, PolyArg arg);
PolyUOp *poly_uop2(
    PolyCtx *ctx,
    PolyOps op,
    PolyDType dtype,
    PolyUOp *s0,
    PolyUOp *s1,
    PolyArg arg
);
PolyUOp *poly_uop3(
    PolyCtx *ctx,
    PolyOps op,
    PolyDType dtype,
    PolyUOp *s0,
    PolyUOp *s1,
    PolyUOp *s2,
    PolyArg arg
);

/* Toposort: returns arena-allocated array of UOp pointers, sets *n_out.
 * _ex variant: gate callback (NULL=visit all, return false to skip subtree),
 * enter_calls (false = skip CALL src[0], process src[1:] only).
 * _ex_user variant: gate carries user_data (closure-style, mirrors tinygrad's
 * `u.toposort(gate=lambda x: r in x.ranges)` where r is captured). */
PolyUOp **poly_toposort(PolyCtx *ctx, PolyUOp *root, int *n_out);
PolyUOp **poly_toposort_ex(
    PolyCtx *ctx,
    PolyUOp *root,
    int *n_out,
    bool (*gate)(PolyUOp *),
    bool enter_calls
);
PolyUOp **poly_toposort_ex_user(
    PolyCtx *ctx,
    PolyUOp *root,
    int *n_out,
    bool (*gate)(PolyUOp *, void *),
    void *user_data,
    bool enter_calls
);

/* Per-pass cache for UOp queries (ranges, vmin/vmax) *
 * Tinygrad caches every queryable UOp property as @functools.cached_property
 * on the immutable UOp instance, which gives per-UOp-lifetime memoization
 * for free. Polygrad's UOps are also immutable (arena-allocated, hash-consed)
 * but we keep the cache external so it can be scoped to one rewrite pass
 * and thrown away cleanly.
 *
 * PolyUOpCache unifies two per-UOp caches that Phase D's reduce_collapse
 * driver queries together:
 *   - minmax: UOp -> (int64_t vmin, int64_t vmax) per tinygrad _min_max
 *   - ranges: UOp -> set of active RANGE ancestors per tinygrad u.ranges
 *
 * Without caching, the minmax computation is exponential on diamond DAGs
 * (MUL alone is a 4-corner recurrence; hash-consed graphs like
 * `arange(n).reshape(n,1) - arange(n).reshape(1,n)` compound that with
 * shared subexpressions). The same holds for the ranges-set walk on any
 * graph where many ancestors query the same subtree.
 *
 * Usage:
 *   PolyUOpCache *c = poly_uop_cache_new();
 *   poly_uop_minmax_ex(ctx, u, c, &lo, &hi);
 *   bool ok = poly_no_range_ex(ctx, u2, c);
 *   ... more queries reusing c ...
 *   poly_uop_cache_destroy(c);
 *
 * Lifetime: cache entries are allocated from the PolyCtx arena and live
 * until the ctx is destroyed. poly_uop_cache_destroy frees the two PolyMap
 * wrappers only; the arena-backed entries are reclaimed at ctx teardown.
 * Cache invalidation is NOT automatic — if the graph is mutated via
 * poly_uop_substitute between queries, destroy and recreate the cache. */

typedef struct PolyUOpCache PolyUOpCache;

PolyUOpCache *poly_uop_cache_new(void);
void poly_uop_cache_destroy(PolyUOpCache *c);

/* Range helpers. `poly_no_range` matches tinygrad codegen/simplify.py:75:
 *   def no_range(u): return not any(x.op is Ops.RANGE for x in u.backward_slice_with_self)
 *
 * `poly_uop_ranges` / `poly_uop_in_ranges` mirror tinygrad uop/ops.py:362-378:
 *   ranges(u) = union(ranges(s) for s in u.src) - ended_ranges(u) + ({u} if RANGE)
 *
 * where ended_ranges() matches ops.py:351-358 (trailing srcs past range_start,
 * AFTER: recursive flatten, CONTRACT: filter by axis_id). See src/uop.c.
 *
 * Every helper has a public one-off entry point (allocates a throwaway cache
 * per call, destroys on return) and an `_ex` variant that takes a caller-
 * owned PolyUOpCache for batch queries. Use the `_ex` form in any hot loop. */
/* Returns the terminal buffer-identity UOp (BUFFER / BUFFER_VIEW / PARAM)
 * after unwrapping RESHAPE/MULTI, or NULL if `u` has no buffer identity. */
const PolyUOp *poly_uop_get_buffer_identity(const PolyUOp *u);

/* True iff `u` is a buffer view (RESHAPE/MULTI over BUFFER/BUFFER_VIEW/PARAM).
 * Frontends use this to decide whether a tensor needs materialization
 * (matches tinygrad's UOp.has_buffer_identity). */
bool poly_uop_has_buffer_identity(const PolyUOp *u);

/* True when target appears in root's source graph. Frontends use this to match
 * tinygrad's backward discovery rule: live tensors with t.uop in loss.toposort. */
bool poly_uop_reachable(PolyCtx *ctx, PolyUOp *root, PolyUOp *target);

bool poly_no_range(PolyCtx *ctx, PolyUOp *u);
bool poly_no_range_ex(PolyCtx *ctx, PolyUOp *u, PolyUOpCache *cache);
bool poly_uop_in_ranges(PolyCtx *ctx, PolyUOp *u, PolyUOp *r);
bool poly_uop_in_ranges_ex(PolyCtx *ctx, PolyUOp *u, PolyUOp *r, PolyUOpCache *cache);
int poly_uop_ranges(PolyCtx *ctx, PolyUOp *u, PolyUOp **out, int max_out);
int poly_uop_ranges_ex(PolyCtx *ctx, PolyUOp *u, PolyUOp **out, int max_out, PolyUOpCache *cache);

/* vmin/vmax interval arithmetic. Full port of tinygrad uop/ops.py:856-897
 * (UOp._min_max) with per-pass memoization via PolyUOpCache.
 *
 * Integer-only: polygrad tracks bounds as int64 and uses an (INT64_MIN,
 * INT64_MAX) sentinel for any float-dtype UOp. Phase D's rules only query
 * bounds on integer operands (range counts, comparison cuts) so this is
 * sufficient. Callers that need float bounds must check poly_dtype_is_float
 * first and handle the sentinel explicitly.
 *
 * Overflow: corner multiplications and shifts use __builtin_*_overflow
 * detection and fall through to dtype bounds on overflow. This can produce
 * loose (but conservative and correct) intervals for pathological inputs;
 * practical Phase D workloads stay far below int64 saturation.
 *
 * Parity: verified against test/parity_scripts/tg_minmax_gt.py. */
void poly_uop_minmax(PolyCtx *ctx, PolyUOp *u, int64_t *vmin, int64_t *vmax);
void poly_uop_minmax_ex(
    PolyCtx *ctx,
    PolyUOp *u,
    PolyUOpCache *cache,
    int64_t *vmin,
    int64_t *vmax
);

/* Pretty-print a UOp graph to a buffer (returns malloc'd string, caller frees) */
char *poly_uop_str(PolyUOp *u);
char *poly_graph_str(PolyUOp *root);

/* Recursive indented IR tree dump to a FILE*. Used by passes (rangeify,
 * codegen, reduce_simplify) to print pre/post-rewrite IR for diagnostics.
 * Caps recursion at max_depth (suggested: 12-16 for full graphs). */
#include <stdio.h>
void poly_uop_dump_tree(FILE *fp, PolyUOp *u, int depth, int max_depth);

/* Shape */
/* ndim == -1 means "no tensor shape" (kernel-level ops like RANGE, LOAD) */

typedef struct {
  int64_t *dims; /* arena-allocated array of dimension sizes */
  int ndim; /* -1 = no shape, 0 = scalar, >0 = tensor */
} PolyShape;

#define POLY_SHAPE_NONE ((PolyShape){NULL, -1})
#define POLY_MAX_DIMS 16

PolyShape poly_uop_shape(PolyCtx *ctx, PolyUOp *u);
int64_t poly_shape_numel(PolyShape s);
bool poly_shape_eq(PolyShape a, PolyShape b);

/* Lazy cached shape accessors -- computes on first access, O(1) thereafter.
 * Returns arena-owned dims, do NOT free. */
int poly_uop_ndim(PolyCtx *ctx, const PolyUOp *u);
const int64_t *poly_uop_dims(PolyCtx *ctx, const PolyUOp *u);
PolyShape poly_uop_shape_cached(PolyCtx *ctx, const PolyUOp *u);
PolyArena *poly_ctx_arena(PolyCtx *ctx);

/* Named buffer registry */

typedef enum {
  POLY_ROLE_PARAM = 0,
  POLY_ROLE_INPUT = 1,
  POLY_ROLE_TARGET = 2,
  POLY_ROLE_OUTPUT = 3,
  POLY_ROLE_AUX = 4,
} PolyBufRole;

typedef struct {
  const char *name; /* arena-allocated */
  PolyBufRole role;
  PolyUOp *buffer; /* BUFFER UOp */
  int64_t shape[8];
  int ndim;
  bool is_alias;
  bool trainable; /* PARAMs default true; frozen imported params set false */
} PolyRegEntry;

/* Register a named BUFFER on ctx. Returns the BUFFER UOp.
 * Re-registration with same (name, dtype, shape) returns existing buffer.
 * Mismatch (same name, different dtype or shape) returns NULL. */
PolyUOp *poly_param(
    PolyCtx *ctx,
    PolyDType dt,
    const int64_t *shape,
    int ndim,
    const char *fmt,
    ...
) __attribute__((format(printf, 5, 6)));
PolyUOp *poly_input(
    PolyCtx *ctx,
    PolyDType dt,
    const int64_t *shape,
    int ndim,
    const char *fmt,
    ...
) __attribute__((format(printf, 5, 6)));
PolyUOp *poly_output(
    PolyCtx *ctx,
    PolyDType dt,
    const int64_t *shape,
    int ndim,
    const char *fmt,
    ...
) __attribute__((format(printf, 5, 6)));
PolyUOp *poly_target(
    PolyCtx *ctx,
    PolyDType dt,
    const int64_t *shape,
    int ndim,
    const char *fmt,
    ...
) __attribute__((format(printf, 5, 6)));
PolyUOp *poly_aux(PolyCtx *ctx, PolyDType dt, const int64_t *shape, int ndim, const char *fmt, ...)
    __attribute__((format(printf, 5, 6)));

/* Non-variadic named-buffer registration for FFI/frontends. These mirror
 * poly_param/poly_input/poly_output/poly_target/poly_aux, but accept an exact
 * string and dtype id so Python/JS/WASM do not need to call variadic C APIs. */
PolyUOp *poly_register_buffer_by_id(
    PolyCtx *ctx,
    int role,
    int dtype_id,
    const int64_t *shape,
    int ndim,
    const char *name
);

/* Register an existing BUFFER-like UOp as a named ABI buffer. This is the
 * export path for tinygrad-style lazy model objects whose parameters already
 * exist before the model graph is traced. */
PolyUOp *poly_register_existing_buffer(
    PolyCtx *ctx,
    int role,
    PolyUOp *buffer,
    const int64_t *shape,
    int ndim,
    const char *name,
    bool trainable
);

/* Create an alias: alias_name resolves to the same buffer as existing_name.
 * Returns 0 on success, -1 on error (existing_name not found, or alias_name
 * already taken by a different buffer). */
int poly_alias(PolyCtx *ctx, const char *alias_name, const char *existing_name);

/* Lookup a named buffer by name. Returns the BUFFER UOp, or NULL. */
PolyUOp *poly_ctx_get(PolyCtx *ctx, const char *fmt, ...) __attribute__((format(printf, 2, 3)));

/* Lookup a registry entry by name. Returns NULL if not found. */
const PolyRegEntry *poly_ctx_get_entry(PolyCtx *ctx, const char *name);

/* Mark a named PARAM trainable/frozen. Non-PARAM buffers ignore optimizer
 * trainability but still carry the bit for import/export round-trips. */
int poly_ctx_set_trainable(PolyCtx *ctx, const char *name, bool trainable);
bool poly_ctx_is_trainable(PolyCtx *ctx, const char *name);

/* Enumeration of all named entries (including aliases). */
int poly_ctx_named_count(PolyCtx *ctx);
const PolyRegEntry *poly_ctx_named_entry(PolyCtx *ctx, int i);

/* Register a named entrypoint (SINK UOp). Returns 0 on success, -1 on error. */
int poly_register_entrypoint(PolyCtx *ctx, const char *name, PolyUOp *sink);

/* Entrypoint enumeration. */
int poly_ctx_entrypoint_count(PolyCtx *ctx);
const char *poly_ctx_entrypoint_name(PolyCtx *ctx, int i);
PolyUOp *poly_ctx_entrypoint_sink(PolyCtx *ctx, int i);

/* Autograd */
/* Reverse-mode gradient of loss w.r.t. wrt.
 * Returns a UOp expression for d(loss)/d(wrt), or NULL on unsupported path. */
PolyUOp *poly_grad(PolyCtx *ctx, PolyUOp *loss, PolyUOp *wrt);

/* Compute gradients for multiple targets in a single reverse pass.
 * initial_grad: the upstream gradient (NULL = ones_like(loss)).
 * wrts[0..n-1]: target UOps to differentiate w.r.t.
 * out_grads[0..n-1]: receives gradient UOps (zero if no path).
 * Returns 0 on success, -1 on failure. */
int poly_grad_many(
    PolyCtx *ctx,
    PolyUOp *loss,
    PolyUOp *initial_grad,
    PolyUOp **wrts,
    int n,
    PolyUOp **out_grads
);

/* Substitute UOps in a graph: replace from[i] with to[i] for i in [0,n).
 * Returns a new root UOp with substitutions applied. Used to reconnect
 * realized intermediate buffers back to their original computation graphs
 * before calling poly_grad. */
PolyUOp *poly_uop_substitute(PolyCtx *ctx, PolyUOp *root, PolyUOp **from, PolyUOp **to, int n);

/* UOp construction helpers */

int poly_op_count(void);

PolyUOp *poly_const_float(PolyCtx *ctx, double value);
PolyUOp *poly_const_double(PolyCtx *ctx, double value);
PolyUOp *poly_const_int(PolyCtx *ctx, int64_t value);
PolyUOp *poly_const_typed(PolyCtx *ctx, PolyDType dt, double value);

PolyUOp *poly_alu1(PolyCtx *ctx, PolyOps op, PolyUOp *src);
PolyUOp *poly_alu2(PolyCtx *ctx, PolyOps op, PolyUOp *a, PolyUOp *b);
PolyUOp *poly_alu3(PolyCtx *ctx, PolyOps op, PolyUOp *a, PolyUOp *b, PolyUOp *c);

PolyUOp *poly_store_val(PolyCtx *ctx, PolyUOp *buf, PolyUOp *value);
PolyUOp *poly_sink1(PolyCtx *ctx, PolyUOp *store);
PolyUOp *poly_sink_n(PolyCtx *ctx, PolyUOp **stores, int n);

PolyUOp *poly_buffer_on_device(
    PolyCtx *ctx,
    PolyDType scalar_dtype,
    int64_t size,
    PolyDevice device
);
PolyUOp *poly_buffer(PolyCtx *ctx, PolyDType scalar_dtype, int64_t size);
PolyUOp *poly_reshape(PolyCtx *ctx, PolyUOp *src, int64_t *dims, int ndim);
PolyUOp *poly_expand(PolyCtx *ctx, PolyUOp *src, int64_t *dims, int ndim);
PolyUOp *poly_permute(PolyCtx *ctx, PolyUOp *src, int64_t *perm, int ndim);
PolyUOp *poly_shrink(PolyCtx *ctx, PolyUOp *src, int64_t (*pairs)[2], int ndim);
PolyUOp *poly_flip(PolyCtx *ctx, PolyUOp *src, int64_t *axes, int n_axes);
PolyUOp *poly_pad(PolyCtx *ctx, PolyUOp *src, int64_t (*pairs)[2], int ndim);
PolyUOp *poly_reduce_axis(PolyCtx *ctx, PolyOps reduce_op, PolyUOp *src, int64_t *axes, int n_axes);

#ifdef __cplusplus
}
#endif

#endif /* POLYGRAD_H */
