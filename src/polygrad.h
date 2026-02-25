/*
 * polygrad.h — Public header for the polygrad C11 tensor compiler
 *
 * A C11 port of tinygrad's compiler core: UOp IR, pattern matcher,
 * scheduler, codegen, and CPU runtime.
 *
 * Reference: tinygrad commit c2be31e75b366638965337b96f2c66c2ba8c4068
 */

#ifndef POLYGRAD_H
#define POLYGRAD_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Ops enum (mirrors tinygrad Ops, 78 members) ──────────────────────── */
/* The order controls toposort priority (lower = earlier).                  */

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
  POLY_OP_CALL,
  POLY_OP_PROGRAM,
  POLY_OP_LINEAR,
  POLY_OP_SOURCE,
  POLY_OP_BINARY,
  POLY_OP_SINK,
  POLY_OP_AFTER,
  POLY_OP_GROUP,
  POLY_OP_GEP,
  POLY_OP_VECTORIZE,

  /* 3 — load/store */
  POLY_OP_INDEX,
  POLY_OP_LOAD,
  POLY_OP_STORE,

  /* 4 — math */
  POLY_OP_WMMA,
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
  POLY_OP_ASSIGN,
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
  POLY_OP_ENCDEC,
  POLY_OP_RESHAPE,
  POLY_OP_PERMUTE,
  POLY_OP_EXPAND,
  POLY_OP_PAD,
  POLY_OP_SHRINK,
  POLY_OP_FLIP,
  POLY_OP_MULTI,
  POLY_OP_REDUCE_AXIS,
  POLY_OP_REDUCE,
  POLY_OP_ALLREDUCE,
  POLY_OP_UNROLL,
  POLY_OP_CONTRACT,
  POLY_OP_CAT,
  POLY_OP_PTRCAT,

  POLY_OP_COUNT  /* sentinel — total number of ops */
} PolyOps;

/* ── GroupOp bitmask sets ─────────────────────────────────────────────── */
/* Each set is a uint64_t[2] bitmask (128 bits, enough for 78 ops).        */

typedef struct { uint64_t bits[2]; } PolyOpSet;

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
  return (PolyOpSet){{ a.bits[0] | b.bits[0], a.bits[1] | b.bits[1] }};
}

static inline bool poly_opset_empty(PolyOpSet set) {
  return set.bits[0] == 0 && set.bits[1] == 0;
}

static inline PolyOpSet poly_opset_intersect(PolyOpSet a, PolyOpSet b) {
  return (PolyOpSet){{ a.bits[0] & b.bits[0], a.bits[1] & b.bits[1] }};
}

static inline bool poly_opset_subset(PolyOpSet sub, PolyOpSet super) {
  return (sub.bits[0] & ~super.bits[0]) == 0 && (sub.bits[1] & ~super.bits[1]) == 0;
}

const char *poly_op_name(PolyOps op);

/* ── DType ────────────────────────────────────────────────────────────── */

typedef enum {
  POLY_ADDR_GLOBAL = 0,
  POLY_ADDR_LOCAL,
  POLY_ADDR_REG,
} PolyAddrSpace;

typedef struct {
  int8_t priority;
  uint16_t bitsize;
  const char *name;   /* C type name, e.g. "float", "int" */
  char fmt;           /* struct pack format char, 0 if none */
  uint16_t count;     /* vector width, 1 for scalars */
  /* pointer fields (0/defaults for non-pointer dtypes) */
  bool is_ptr;
  PolyAddrSpace addrspace;
  uint16_t vcount;    /* pointer vector count */
  int64_t ptr_size;   /* -1 = unlimited */
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
bool poly_dtype_is_unsigned(PolyDType dt);
bool poly_dtype_is_bool(PolyDType dt);
PolyDType poly_dtype_scalar(PolyDType dt);
PolyDType poly_dtype_vec(PolyDType dt, int sz);
PolyDType poly_dtype_ptr(PolyDType dt, int64_t size, PolyAddrSpace addrspace);
int poly_dtype_itemsize(PolyDType dt);
const char *poly_dtype_name(PolyDType dt);

/* ── Axis metadata for RANGE args (tinygrad AxisType parity) ─────────── */

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
  POLY_AXIS_OUTER,
  POLY_AXIS_PLACEHOLDER,
} PolyAxisType;

/* ── PolyArg — tagged union for UOp arg field ──────────────────────────── */

typedef enum {
  POLY_ARG_NONE = 0,
  POLY_ARG_INT,
  POLY_ARG_FLOAT,
  POLY_ARG_BOOL,
  POLY_ARG_INT_TUPLE,
  POLY_ARG_PAIR_TUPLE,   /* array of (int64_t, int64_t) pairs */
  POLY_ARG_STRING,
  POLY_ARG_OPS,
  POLY_ARG_REDUCE_AXIS,  /* (PolyOps, int64_t[], n) */
  POLY_ARG_RANGE,        /* (axis_id, axis_type, extra...) */
  POLY_ARG_DEFINE_VAR,   /* (name, min_val, max_val) */
  POLY_ARG_INVALID,
} PolyArgKind;

typedef struct {
  PolyArgKind kind;
  union {
    int64_t i;
    double f;
    bool b;
    struct { int64_t *vals; int n; } int_tuple;
    struct { int64_t (*pairs)[2]; int n; } pair_tuple;
    const char *str;
    PolyOps ops;
    struct { PolyOps op; int64_t *axes; int n; } reduce_axis;
    struct { int64_t axis_id; PolyAxisType axis_type; int64_t *extra; int n_extra; } range;
    struct { const char *name; int64_t min_val; int64_t max_val; } define_var;
  };
} PolyArg;

static inline PolyArg poly_arg_none(void) { return (PolyArg){ .kind = POLY_ARG_NONE }; }
static inline PolyArg poly_arg_int(int64_t v) { return (PolyArg){ .kind = POLY_ARG_INT, .i = v }; }
static inline PolyArg poly_arg_float(double v) { return (PolyArg){ .kind = POLY_ARG_FLOAT, .f = v }; }
static inline PolyArg poly_arg_bool(bool v) { return (PolyArg){ .kind = POLY_ARG_BOOL, .b = v }; }
static inline PolyArg poly_arg_ops(PolyOps op) { return (PolyArg){ .kind = POLY_ARG_OPS, .ops = op }; }
static inline PolyArg poly_arg_invalid(void) { return (PolyArg){ .kind = POLY_ARG_INVALID }; }
static inline PolyArg poly_arg_str(const char *s) { return (PolyArg){ .kind = POLY_ARG_STRING, .str = s }; }
static inline PolyArg poly_arg_range(int64_t axis_id, PolyAxisType axis_type) {
  return (PolyArg){
    .kind = POLY_ARG_RANGE,
    .range = { .axis_id = axis_id, .axis_type = axis_type, .extra = NULL, .n_extra = 0 }
  };
}
static inline PolyArg poly_arg_range_ex(int64_t axis_id, PolyAxisType axis_type,
                                        int64_t *extra, int n_extra) {
  return (PolyArg){
    .kind = POLY_ARG_RANGE,
    .range = { .axis_id = axis_id, .axis_type = axis_type, .extra = extra, .n_extra = n_extra }
  };
}

static inline PolyArg poly_arg_define_var(const char *name, int64_t min_val, int64_t max_val) {
  return (PolyArg){
    .kind = POLY_ARG_DEFINE_VAR,
    .define_var = { .name = name, .min_val = min_val, .max_val = max_val }
  };
}

/* Compatibility helpers: legacy RANGE arg kind may still be POLY_ARG_INT
 * during migration; treat that as LOOP axis with id=arg.i. */
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

bool poly_arg_eq(PolyArg a, PolyArg b);
uint32_t poly_arg_hash(PolyArg a);

/* ── Arena allocator ──────────────────────────────────────────────────── */

typedef struct PolyArena PolyArena;

PolyArena *poly_arena_new(size_t initial_cap);
void *poly_arena_alloc(PolyArena *a, size_t size, size_t align);
void poly_arena_reset(PolyArena *a);
void poly_arena_destroy(PolyArena *a);
size_t poly_arena_used(PolyArena *a);

/* ── Hash map (for CSE) ───────────────────────────────────────────────── */

typedef struct PolyMap PolyMap;

PolyMap *poly_map_new(size_t initial_cap);
void poly_map_destroy(PolyMap *m);
void *poly_map_get(PolyMap *m, uint32_t hash, const void *key,
                   bool (*eq)(const void *a, const void *b));
void poly_map_set(PolyMap *m, uint32_t hash, const void *key, void *value,
                  bool (*eq)(const void *a, const void *b));
void poly_map_remove(PolyMap *m, uint32_t hash, const void *key,
                     bool (*eq)(const void *a, const void *b));
size_t poly_map_len(PolyMap *m);
void poly_map_clear(PolyMap *m);

/* Iterate over all entries in the map.
 * Callback receives (key, value, userdata) for each occupied slot. */
typedef void (*PolyMapIterFn)(const void *key, void *value, void *userdata);
void poly_map_foreach(PolyMap *m, PolyMapIterFn fn, void *userdata);

/* ── UOp ──────────────────────────────────────────────────────────────── */

typedef struct PolyUOp PolyUOp;

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
  uint8_t *bytes;                         /* malloc'd rendered bytes (WASM/C/etc) */
  int len;                                /* byte length */
  int n_bufs;                             /* number of buffer params */
  PolyUOp *bufs[POLY_MAX_KERNEL_BUFS];     /* ordered buffer UOps */
} PolyCachedKernel;

/* Context owns the arena, CSE cache, kernel cache, and all UOps */
typedef struct PolyCtx PolyCtx;

PolyCtx *poly_ctx_new(void);
void poly_ctx_destroy(PolyCtx *ctx);
bool poly_ctx_owns_ptr(PolyCtx *ctx, const void *p);
PolyMap *poly_ctx_kernel_cache(PolyCtx *ctx);

/* Create a UOp (with CSE deduplication) */
PolyUOp *poly_uop(PolyCtx *ctx, PolyOps op, PolyDType dtype,
                 PolyUOp **src, int n_src, PolyArg arg);

/* Convenience: create a UOp with 0, 1, 2, or 3 sources */
PolyUOp *poly_uop0(PolyCtx *ctx, PolyOps op, PolyDType dtype, PolyArg arg);
PolyUOp *poly_uop1(PolyCtx *ctx, PolyOps op, PolyDType dtype, PolyUOp *s0, PolyArg arg);
PolyUOp *poly_uop2(PolyCtx *ctx, PolyOps op, PolyDType dtype, PolyUOp *s0, PolyUOp *s1, PolyArg arg);
PolyUOp *poly_uop3(PolyCtx *ctx, PolyOps op, PolyDType dtype, PolyUOp *s0, PolyUOp *s1, PolyUOp *s2, PolyArg arg);

/* Toposort: returns arena-allocated array of UOp pointers, sets *n_out */
PolyUOp **poly_toposort(PolyCtx *ctx, PolyUOp *root, int *n_out);

/* Pretty-print a UOp graph to a buffer (returns malloc'd string, caller frees) */
char *poly_uop_str(PolyUOp *u);
char *poly_graph_str(PolyUOp *root);

/* ── Shape ────────────────────────────────────────────────────────────── */
/* ndim == -1 means "no tensor shape" (kernel-level ops like RANGE, LOAD) */

typedef struct {
  int64_t *dims;   /* arena-allocated array of dimension sizes */
  int ndim;        /* -1 = no shape, 0 = scalar, >0 = tensor */
} PolyShape;

#define POLY_SHAPE_NONE ((PolyShape){ NULL, -1 })
#define POLY_MAX_DIMS 16

PolyShape poly_uop_shape(PolyCtx *ctx, PolyUOp *u);
int64_t  poly_shape_numel(PolyShape s);
bool     poly_shape_eq(PolyShape a, PolyShape b);

/* ── Autograd ─────────────────────────────────────────────────────────── */
/* Reverse-mode gradient of loss w.r.t. wrt.
 * Returns a UOp expression for d(loss)/d(wrt), or NULL on unsupported path. */
PolyUOp *poly_grad(PolyCtx *ctx, PolyUOp *loss, PolyUOp *wrt);

/* Compute gradients for multiple targets in a single reverse pass.
 * initial_grad: the upstream gradient (NULL = ones_like(loss)).
 * wrts[0..n-1]: target UOps to differentiate w.r.t.
 * out_grads[0..n-1]: receives gradient UOps (zero if no path).
 * Returns 0 on success, -1 on failure. */
int poly_grad_many(PolyCtx *ctx, PolyUOp *loss, PolyUOp *initial_grad,
                    PolyUOp **wrts, int n, PolyUOp **out_grads);

/* Substitute UOps in a graph: replace from[i] with to[i] for i in [0,n).
 * Returns a new root UOp with substitutions applied. Used to reconnect
 * realized intermediate buffers back to their original computation graphs
 * before calling poly_grad. */
PolyUOp *poly_uop_substitute(PolyCtx *ctx, PolyUOp *root,
                               PolyUOp **from, PolyUOp **to, int n);

#ifdef __cplusplus
}
#endif

#endif /* POLYGRAD_H */
