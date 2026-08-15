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

#ifndef POLY_DEPRECATED
/* Define POLY_ENABLE_DEPRECATED_WARNINGS before including this header to make
 * compatibility-only APIs produce compiler diagnostics. The default is quiet so
 * Polygrad's own compatibility tests and legacy shims do not flood normal
 * builds with expected warnings. */
#if defined(POLY_ENABLE_DEPRECATED_WARNINGS) && (defined(__GNUC__) || defined(__clang__))
#define POLY_DEPRECATED(msg) __attribute__((deprecated(msg)))
#else
#define POLY_DEPRECATED(msg)
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Free heap memory returned by Polygrad public APIs.
 * This keeps bindings from having to match Polygrad's C runtime allocator. */
void poly_free(void *ptr);

/* Polygrad's public op enum keeps a few C-side compatibility ops. Use
 * poly_op_value(op) where tinygrad's Ops.value ordering is required. */

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
  POLY_OP_CDIV,
  POLY_OP_IDIV = POLY_OP_CDIV, /* compatibility alias: tinygrad name is CDIV */
  POLY_OP_MAX,
  POLY_OP_CMOD,
  POLY_OP_MOD = POLY_OP_CMOD, /* compatibility alias: tinygrad name is CMOD */
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
  POLY_OP_FLOORDIV,
  POLY_OP_FLOORMOD,
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
  POLY_OP_STAGE,
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
   * AFTER(target, STORE(target, value)); REDUCE_AXIS is retained for legacy
   * raw/import graphs while normal Tensor construction uses REDUCE; ENCDEC is
   * not present upstream. */
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
int poly_op_value(PolyOps op);

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
bool poly_dtype_least_upper(PolyDType a, PolyDType b, PolyDType *out);
bool poly_dtype_can_lossless_cast(PolyDType dt0, PolyDType dt1);
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

typedef struct PolyProgramInfo PolyProgramInfo;

/* Pinned tinygrad ParamArg metadata. Value-level call PARAMs carry shape in
 * src[0] and keep slot/device here; rangeify later lowers them to the existing
 * pointer PARAM form whose arg is the kernel-local integer slot. */
typedef struct {
  int64_t slot;
  const char *name;
  int64_t min_val;
  int64_t max_val;
  bool has_minmax;
  PolyAddrSpace addrspace;
  int32_t axis;
  bool has_axis;
  /* Pinned ParamArg.device is str | tuple[str, ...] | None
   * (tinygrad/uop/ops.py:1071-1076). Scalar identities use device;
   * ordered tuple identities use devices/n_devices. Exactly one arm is set. */
  const char *device;
  const char **devices;
  int32_t n_devices;
  bool device_is_tuple;
} PolyParamArg;

/* Pinned tinygrad CallInfo metadata for CALL/FUNCTION UOps
 * (tinygrad/uop/ops.py:1158-1170). The current C value-function boundary
 * supports the serializable default subset: no Python grad callback, empty
 * metadata, optional name, no aux, and non-precompiled forward/backward. The
 * presence bits keep unsupported variants distinguishable and fail-closed. */
typedef struct {
  const char *name;
  bool precompile;
  bool precompile_backward;
  bool has_grad_fxn;
  bool has_metadata;
  bool has_aux;
} PolyCallInfo;

/* Exact Python-int-compatible CONST argument. Limbs are little-endian base
 * 2**32 magnitude; sign is -1 or +1 and zero remains POLY_ARG_INT(0).
 * poly_uop copies limbs into the owning context arena. */
typedef struct {
  int8_t sign;
  uint32_t n_limbs;
  const uint32_t *limbs;
} PolyBigInt;

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
  POLY_ARG_BUFFERIZE_OPTS, /* (exact scalar/tuple device, addrspace, removable) */
  POLY_ARG_TENSOR_CORE, /* tinygrad WMMA metadata: (name, dims, threads) */
  POLY_ARG_PROGRAM_INFO, /* PolyProgramInfo* value metadata for PROGRAM */
  POLY_ARG_BYTES, /* immutable runtime bytes for BINARY UOps */
  POLY_ARG_INVALID,
  POLY_ARG_PARAM, /* pinned tinygrad ParamArg* for shaped value PARAMs */
  POLY_ARG_BIGINT, /* exact signed arbitrary-precision integer CONST */
  POLY_ARG_STRING_TUPLE, /* ordered immutable string tuple (for DEVICE.arg) */
  POLY_ARG_CALL_INFO, /* pinned tinygrad CallInfo* for CALL/FUNCTION */
} PolyArgKind;

typedef struct {
  PolyArgKind kind;
  union {
    int64_t i;
    PolyBigInt bigint;
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
    struct {
      const char **vals;
      int n;
    } string_tuple;
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
      /* Pinned BufferizeOpts.device is str | tuple[str, ...] | int | None
       * (tinygrad/schedule/indexing.py:38-43). Global physical scheduling uses
       * the scalar/tuple string arms; local integer ids remain addrspace-local. */
      const char *device;
      const char **devices;
      int32_t n_devices;
      bool device_is_tuple;
      PolyAddrSpace addrspace;
      bool removable;
    } bufferize_opts;
    struct {
      const char *name;
      int dims[3];
      int threads;
    } tensor_core;
    const PolyProgramInfo *program_info;
    const PolyParamArg *param;
    const PolyCallInfo *call_info;
    struct {
      const uint8_t *data;
      int n;
    } bytes;
  };
} PolyArg;

static inline PolyArg poly_arg_none(void) {
  return (PolyArg){.kind = POLY_ARG_NONE};
}
static inline PolyArg poly_arg_int(int64_t v) {
  return (PolyArg){.kind = POLY_ARG_INT, .i = v};
}
static inline PolyArg poly_arg_bigint(int sign, const uint32_t *limbs, uint32_t n_limbs) {
  return (PolyArg
  ){.kind = POLY_ARG_BIGINT,
    .bigint = {.sign = sign < 0 ? -1 : 1, .n_limbs = n_limbs, .limbs = limbs}};
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
static inline PolyArg poly_arg_string_tuple(const char **vals, int n) {
  return (PolyArg){.kind = POLY_ARG_STRING_TUPLE, .string_tuple = {.vals = vals, .n = n}};
}
static inline PolyArg poly_arg_call_info(const PolyCallInfo *info) {
  return (PolyArg){.kind = POLY_ARG_CALL_INFO, .call_info = info};
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
 * device is the exact canonical physical identity for the intermediate buffer
 * (or NULL when unknown/local), addrspace selects global/local storage, and
 * removable preserves the rangeify optimization contract for eliminating
 * redundant temporary buffers. This matches pinned BufferizeOpts.device.
 */
static inline PolyArg poly_arg_bufferize_opts(
    const char *device,
    PolyAddrSpace addrspace,
    bool removable
) {
  return (PolyArg
  ){.kind = POLY_ARG_BUFFERIZE_OPTS,
    .bufferize_opts = {
        .device = device,
        .devices = NULL,
        .n_devices = 0,
        .device_is_tuple = false,
        .addrspace = addrspace,
        .removable = removable,
    }};
}
static inline PolyArg poly_arg_bufferize_opts_tuple(
    const char **devices,
    int32_t n_devices,
    PolyAddrSpace addrspace,
    bool removable
) {
  return (PolyArg
  ){.kind = POLY_ARG_BUFFERIZE_OPTS,
    .bufferize_opts = {
        .device = NULL,
        .devices = devices,
        .n_devices = n_devices,
        .device_is_tuple = true,
        .addrspace = addrspace,
        .removable = removable,
    }};
}
static inline PolyArg poly_arg_tensor_core(const char *name, const int dims[3], int threads) {
  return (PolyArg
  ){.kind = POLY_ARG_TENSOR_CORE,
    .tensor_core = {
        .name = name,
        .dims = {dims ? dims[0] : 0, dims ? dims[1] : 0, dims ? dims[2] : 0},
        .threads = threads,
    }};
}
static inline PolyArg poly_arg_program_info(const PolyProgramInfo *info) {
  return (PolyArg){.kind = POLY_ARG_PROGRAM_INFO, .program_info = info};
}
static inline PolyArg poly_arg_bytes(const uint8_t *data, int n) {
  return (PolyArg){.kind = POLY_ARG_BYTES, .bytes = {.data = data, .n = n}};
}
static inline PolyArg poly_arg_param(const PolyParamArg *param) {
  return (PolyArg){.kind = POLY_ARG_PARAM, .param = param};
}

bool poly_program_info_eq(const PolyProgramInfo *a, const PolyProgramInfo *b);
uint32_t poly_program_info_hash(const PolyProgramInfo *info);

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
static inline const char *poly_bufferize_arg_device(PolyArg a) {
  if (a.kind == POLY_ARG_BUFFERIZE_OPTS && !a.bufferize_opts.device_is_tuple)
    return a.bufferize_opts.device;
  return NULL;
}
static inline bool poly_bufferize_arg_device_is_tuple(PolyArg a) {
  return a.kind == POLY_ARG_BUFFERIZE_OPTS && a.bufferize_opts.device_is_tuple;
}
static inline const char **poly_bufferize_arg_devices(PolyArg a) {
  return poly_bufferize_arg_device_is_tuple(a) ? a.bufferize_opts.devices : NULL;
}
static inline int32_t poly_bufferize_arg_n_devices(PolyArg a) {
  return poly_bufferize_arg_device_is_tuple(a) ? a.bufferize_opts.n_devices : 0;
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
size_t poly_arena_high_water(PolyArena *a);

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

/* Backend kind.
 *
 * Concrete DEVICE UOp identity follows pinned tinygrad and is a canonical
 * string (for example "CUDA" or "CUDA:1").  This enum still selects the
 * current ordinal-zero backend/runtime and must not be used to collapse a
 * nonzero device identity. */

typedef enum {
  POLY_DEVICE_AUTO = 0,
  POLY_DEVICE_HOST, /* imported frontend data, never a kernel target */
  POLY_DEVICE_CPU, /* native compiled CPU backend */
  POLY_DEVICE_INTERP, /* interpreter (shares CPU/WASM storage) */
  POLY_DEVICE_WASM, /* WASM JIT backend */
  POLY_DEVICE_WEBGPU, /* WebGPU GPU backend */
  POLY_DEVICE_CUDA,
  POLY_DEVICE_HIP,
  POLY_DEVICE_X86,
  POLY_DEVICE_DISK, /* mmap-backed storage device, never a kernel target */
} PolyDevice;

/* Default compute backend for the current build */
PolyDevice poly_device_default(void);

/* Can this device execute kernels? false for HOST and AUTO */
bool poly_device_can_execute(PolyDevice dev);

/* Is ptr directly dereferenceable by the compiled core? */
bool poly_device_is_host_addressable(PolyDevice dev);

/* Do two devices use the same underlying storage domain? */
bool poly_devices_share_storage(PolyDevice a, PolyDevice b);

/* Look up the currently executable ordinal-zero backend by name. `:0` is
 * canonicalized away; nonzero ordinals return AUTO until runtime identity is
 * migrated. */
PolyDevice poly_device_by_name(const char *name);
const char *poly_device_name(PolyDevice device);

/* Run a small embeddable sanity check for ALU folding plus schedule/runtime
 * execution through ctx->buffers. `poly_selftest()` uses the interpreter so it
 * does not require a host C compiler. `poly_selftest_device()` lets embedders
 * explicitly validate a backend/runtime device. Returns 0 on success. */
int poly_selftest(void);
int poly_selftest_device(PolyDevice device);

typedef struct PolyCtx PolyCtx;
typedef struct PolyUOp PolyUOp;
typedef struct PolyJit PolyJit;

/* Core frontend tensor handle.
 *
 * PolyUOp stays the pure logical value graph. PolyTensor is the C-side value
 * reference used by frontends. The tensor carries two roots: logical is the
 * exportable/re-placement expression, while physical is the mandatory current
 * tinygrad-shaped execution/readback root. poly_tensor_uop() returns physical;
 * portable export/provenance uses uop_logical explicitly.
 */
typedef struct PolyTensor PolyTensor;

typedef enum {
  POLY_TENSOR_VALUE = 0,
  POLY_TENSOR_PLACE = 1,
  POLY_TENSOR_BARRIER = 2,
} PolyTensorRole;

typedef enum {
  POLY_TENSOR_PROVENANCE_UNKNOWN = 0,
  POLY_TENSOR_PROVENANCE_USER_INPUT = 1,
  POLY_TENSOR_PROVENANCE_PARAM_INIT = 2,
  POLY_TENSOR_PROVENANCE_STATE_LOADED = 3,
  POLY_TENSOR_PROVENANCE_CONST_INIT = 4,
  POLY_TENSOR_PROVENANCE_COMPUTED = 5,
} PolyTensorProvenance;

struct PolyTensor {
  PolyUOp *uop_logical;
  PolyUOp *uop_physical;
  PolyTensorRole role;
  PolyDevice device;
  uint64_t order;
  PolyTensor *source;
  bool requires_grad;
  bool requires_grad_set;
  PolyTensorProvenance provenance;
};

PolyTensor *poly_tensor_create_with_roots(
    PolyCtx *ctx,
    PolyUOp *uop_logical,
    PolyUOp *uop_physical,
    PolyTensorRole role,
    PolyDevice device
);
PolyTensor *poly_tensor_empty(
    PolyCtx *ctx,
    PolyDType scalar_dtype,
    const int64_t *dims,
    int ndim,
    PolyDevice device
);
PolyTensor *poly_tensor_from_host(
    PolyCtx *ctx,
    void *ptr,
    size_t nbytes,
    PolyDType scalar_dtype,
    const int64_t *dims,
    int ndim
);
int poly_tensor_replace_roots(
    PolyCtx *ctx,
    PolyTensor *tensor,
    PolyUOp *uop_logical,
    PolyUOp *uop_physical,
    PolyTensorRole role,
    PolyDevice device
);
PolyTensor *poly_tensor_to_device(PolyCtx *ctx, PolyTensor *tensor, PolyDevice device);
PolyTensor *poly_tensor_assign(PolyCtx *ctx, PolyTensor *target, PolyTensor *value);
PolyTensor *poly_tensor_clone_into(PolyCtx *ctx, PolyTensor *target, PolyTensor *source);
int poly_tensor_custom_kernel(
    PolyCtx *ctx,
    PolyUOp *body,
    PolyTensor **inputs,
    int n_inputs,
    PolyTensor **outputs
);
/* Build pinned value-producing FUNCTION roots from already-constructed Tensor
 * results and ordered argument/state Tensors. The Python decorator owns only
 * object traversal; all UOp composition and implicit-input discovery stays in
 * the C core (tinygrad/function.py:39-94, uop/ops.py:1077-1092). */
int poly_tensor_function(
    PolyCtx *ctx,
    PolyTensor **results,
    int n_results,
    PolyTensor **inputs,
    int n_inputs,
    const char *name,
    bool allow_implicit,
    PolyTensor **outputs
);
PolyTensor *poly_tensor_alu1(PolyCtx *ctx, PolyOps op, PolyTensor *src);
PolyTensor *poly_tensor_alu2(PolyCtx *ctx, PolyOps op, PolyTensor *a, PolyTensor *b);
PolyTensor *poly_tensor_alu3(PolyCtx *ctx, PolyOps op, PolyTensor *a, PolyTensor *b, PolyTensor *c);
PolyTensor *poly_tensor_div(PolyCtx *ctx, PolyTensor *dividend, PolyTensor *divisor);
PolyTensor *poly_tensor_exp(PolyCtx *ctx, PolyTensor *src);
PolyTensor *poly_tensor_log(PolyCtx *ctx, PolyTensor *src);
PolyTensor *poly_tensor_log1p(PolyCtx *ctx, PolyTensor *src);
PolyTensor *poly_tensor_expm1(PolyCtx *ctx, PolyTensor *src);
PolyTensor *poly_tensor_gelu(PolyCtx *ctx, PolyTensor *src);
PolyTensor *poly_tensor_quick_gelu(PolyCtx *ctx, PolyTensor *src);
PolyTensor *poly_tensor_detach(PolyCtx *ctx, PolyTensor *src);
PolyTensor *poly_tensor_contiguous_backward(PolyCtx *ctx, PolyTensor *src);
PolyTensor *poly_tensor_sum(PolyCtx *ctx, PolyTensor *src, int64_t *axes, int n_axes, bool keepdim);
PolyTensor *poly_tensor_sum_dtype_by_id(
    PolyCtx *ctx, PolyTensor *src, int64_t *axes, int n_axes, bool keepdim, int dtype_id
);
PolyTensor *poly_tensor_max(PolyCtx *ctx, PolyTensor *src, int64_t *axes, int n_axes, bool keepdim);
PolyTensor *poly_tensor_argmax(PolyCtx *ctx, PolyTensor *src, int axis, bool keepdim);
PolyTensor *poly_tensor_minimum(PolyCtx *ctx, PolyTensor *a, PolyTensor *b);
PolyTensor *poly_tensor_dot(PolyCtx *ctx, PolyTensor *src, PolyTensor *weight);
PolyTensor *poly_tensor_dot_dtype_by_id(
    PolyCtx *ctx, PolyTensor *src, PolyTensor *weight, int dtype_id
);
int poly_tensor_qr_ex(
    PolyCtx *ctx,
    PolyTensor *src,
    int mode,
    PolyTensor **out_q,
    PolyTensor **out_r
);
PolyTensor *poly_tensor_triangular_solve(
    PolyCtx *ctx,
    PolyTensor *a,
    PolyTensor *b,
    int upper,
    int transpose_a,
    int unit_diagonal
);
PolyTensor *poly_tensor_cholesky(PolyCtx *ctx, PolyTensor *src, int upper);
PolyTensor *poly_tensor_cholesky_solve(PolyCtx *ctx, PolyTensor *chol, PolyTensor *b, int upper);
PolyTensor *poly_tensor_solve(PolyCtx *ctx, PolyTensor *a, PolyTensor *b);
PolyTensor *poly_tensor_lstsq(PolyCtx *ctx, PolyTensor *a, PolyTensor *b);
int poly_tensor_sort(
    PolyCtx *ctx,
    PolyTensor *src,
    int dim,
    int descending,
    PolyTensor **out_values,
    PolyTensor **out_indices
);
int poly_tensor_topk(
    PolyCtx *ctx,
    PolyTensor *src,
    int64_t k,
    int dim,
    int largest,
    int sorted,
    PolyTensor **out_values,
    PolyTensor **out_indices
);
PolyTensor *poly_tensor_softmax(PolyCtx *ctx, PolyTensor *src, int axis);
PolyTensor *poly_tensor_log_softmax(PolyCtx *ctx, PolyTensor *src, int axis);
PolyTensor *poly_tensor_rope(
    PolyCtx *ctx,
    PolyTensor *x,
    PolyTensor *freqs_cos,
    PolyTensor *freqs_sin
);
PolyTensor *poly_tensor_cast_by_id(PolyCtx *ctx, PolyTensor *src, int dtype_id);
PolyTensor *poly_tensor_bitcast_by_id(PolyCtx *ctx, PolyTensor *src, int dtype_id);
PolyTensor *poly_tensor_contiguous(PolyCtx *ctx, PolyTensor *src);
PolyTensor *poly_tensor_reshape(PolyCtx *ctx, PolyTensor *src, int64_t *dims, int ndim);
PolyTensor *poly_tensor_reshape_uop(PolyCtx *ctx, PolyTensor *src, PolyUOp **dims, int ndim);
PolyTensor *poly_tensor_expand(PolyCtx *ctx, PolyTensor *src, int64_t *dims, int ndim);
PolyTensor *poly_tensor_expand_uop(PolyCtx *ctx, PolyTensor *src, PolyUOp **dims, int ndim);
PolyTensor *poly_tensor_permute(PolyCtx *ctx, PolyTensor *src, int64_t *perm, int ndim);
PolyTensor *poly_tensor_shrink(PolyCtx *ctx, PolyTensor *src, int64_t (*pairs)[2], int ndim);
PolyTensor *poly_tensor_shrink_uop(
    PolyCtx *ctx,
    PolyTensor *src,
    PolyUOp **starts,
    PolyUOp **sizes,
    int ndim
);
PolyTensor *poly_tensor_flip(PolyCtx *ctx, PolyTensor *src, int64_t *axes, int n_axes);
PolyTensor *poly_tensor_pad_value(
    PolyCtx *ctx,
    PolyTensor *src,
    int64_t (*pairs)[2],
    int ndim,
    double value
);
PolyTensor *poly_tensor_pool(
    PolyCtx *ctx,
    PolyTensor *src,
    const int64_t *kernel,
    int n_kernel,
    const int64_t *stride,
    const int64_t *dilation
);
PolyTensor *poly_tensor_max_pool2d(
    PolyCtx *ctx,
    PolyTensor *src,
    const int64_t *kernel,
    int n_kernel,
    const int64_t *stride,
    const int64_t *dilation,
    const int64_t *padding,
    int n_padding
);
PolyTensor *poly_tensor_conv2d(
    PolyCtx *ctx,
    PolyTensor *src,
    PolyTensor *weight,
    PolyTensor *bias,
    int groups,
    const int64_t *stride,
    const int64_t *dilation,
    const int64_t *padding,
    int n_padding
);
PolyTensor *poly_tensor_conv2d_dtype_by_id(
    PolyCtx *ctx,
    PolyTensor *src,
    PolyTensor *weight,
    PolyTensor *bias,
    int groups,
    const int64_t *stride,
    const int64_t *dilation,
    const int64_t *padding,
    int n_padding,
    int dtype_id
);
PolyTensor *poly_tensor_batchnorm(
    PolyCtx *ctx,
    PolyTensor *src,
    PolyTensor *weight,
    PolyTensor *bias,
    PolyTensor *mean,
    PolyTensor *invstd,
    const int64_t *axes,
    int n_axes
);
PolyTensor *poly_tensor_one_hot(PolyCtx *ctx, PolyTensor *x, int64_t num_classes);
PolyTensor *poly_tensor_gather_dim(PolyCtx *ctx, PolyTensor *x, int dim, PolyTensor *index);
PolyTensor *poly_tensor_index_select(PolyCtx *ctx, PolyTensor *x, int dim, PolyTensor *index);
PolyTensor *poly_tensor_scatter(
    PolyCtx *ctx,
    PolyTensor *self,
    int dim,
    PolyTensor *index,
    PolyTensor *src,
    const char *reduce
);
PolyTensor *poly_tensor_scatter_reduce(
    PolyCtx *ctx,
    PolyTensor *self,
    int dim,
    PolyTensor *index,
    PolyTensor *src,
    const char *reduce,
    int include_self
);
PolyTensor *poly_tensor_einsum(
    PolyCtx *ctx,
    const char *formula,
    PolyTensor **tensors,
    int n_tensors
);
PolyTensor *poly_tensor_rearrange(
    PolyCtx *ctx,
    const char *formula,
    PolyTensor *tensor,
    const char *axis_names,
    const int64_t *axis_values,
    int n_axis_sizes
);
PolyUOp *poly_tensor_uop(PolyTensor *tensor);
PolyUOp *poly_tensor_uop_logical(PolyTensor *tensor);
PolyUOp *poly_tensor_uop_physical(PolyTensor *tensor);
PolyDevice poly_tensor_device(PolyTensor *tensor);
bool poly_tensor_requires_grad(PolyTensor *tensor);
bool poly_tensor_requires_grad_is_set(PolyTensor *tensor);
void poly_tensor_set_requires_grad(PolyTensor *tensor, bool requires_grad);
PolyTensorProvenance poly_tensor_provenance(PolyTensor *tensor);
void poly_tensor_set_provenance(PolyTensor *tensor, PolyTensorProvenance provenance);
PolyUOp *poly_tensor_physicalize(PolyCtx *ctx, PolyTensor *tensor);
int poly_realize_tensors(PolyCtx *ctx, PolyTensor **inputs, int n, PolyTensor **outputs);
int poly_realize_tensors_ex(
    PolyCtx *ctx,
    PolyTensor **inputs,
    int n,
    PolyTensor **outputs,
    bool update_stats
);

typedef struct PolyVarBinding {
  PolyUOp *var; /* DEFINE_VAR UOp */
  int32_t value; /* concrete runtime value */
} PolyVarBinding;

/* Tinygrad-style raw Tensor JIT capture/replay.
 *
 * This is deliberately a tensor/schedule-layer object, not an Instance
 * entrypoint plan. Capture records the LINEAR schedules produced by normal
 * poly_realize_tensors calls. The retained physical LINEAR substitutes only
 * JIT input BUFFERs with shaped PARAMs and replays the same compiled plan with
 * current input identities, matching tinygrad's CapturedJit input_uops
 * substitution boundary. Input
 * compatibility and runtime variable values follow
 * tinygrad/engine/jit.py:_prepare_jit_inputs: the physical base is replaced by
 * NOOP, the remaining view is unbound, and current BIND values override the
 * captured schedule defaults. Polygrad's logical provenance root is not part
 * of this execution-layer signature.
 */
PolyJit *poly_jit_new(PolyCtx *ctx);
void poly_jit_free(PolyJit *jit);
int poly_jit_set_prune(PolyJit *jit, bool prune);
int poly_jit_begin_capture(PolyJit *jit, PolyTensor **inputs, int n_inputs);
int poly_jit_end_capture(PolyJit *jit);
void poly_jit_cancel_capture(PolyJit *jit);
bool poly_jit_is_captured(PolyJit *jit);
int poly_jit_schedule_count(PolyJit *jit);
int poly_jit_run(PolyJit *jit, PolyTensor **inputs, int n_inputs);
int poly_jit_run_with_vars(
    PolyJit *jit,
    PolyTensor **inputs,
    int n_inputs,
    PolyVarBinding *var_bindings,
    int n_var_bindings
);

/* Derive the backend implementation from an exact DEVICE identity.  This does
 * not imply that every identity of that backend is executable; schedule
 * ingress separately rejects unsupported runtime instances. */
PolyDevice poly_device_from_device_uop(PolyUOp *device);
PolyDevice poly_uop_device(PolyUOp *u);
/* Exact concrete scalar device identity carried by the physical UOp graph.
 * Returns the canonical DEVICE string (for example "CPU:1") or NULL when the
 * graph has no concrete DEVICE identity.  Backend dispatch remains the
 * separate PolyDevice-valued poly_uop_device() compatibility query. */
const char *poly_uop_device_name(PolyCtx *ctx, PolyUOp *u);

/* Frontend host-buffer lifetime hook.
 * Frontends keep strong maps keyed by the C-side PolyBuffer* address value.
 * When the core retires an imported HOST residency, it calls the registered
 * release function with that key so the frontend can drop its owner entry. */
typedef void (*PolyFrontendBufferReleaseFn)(uintptr_t buffer_key);
void poly_set_frontend_buffer_release(PolyFrontendBufferReleaseFn fn);
void poly_ctx_set_frontend_buffer_release(PolyCtx *ctx, PolyFrontendBufferReleaseFn fn);

/* PolyBuffer is defined in device.h (needs PolyAllocator pointer) */

/* UOp */

struct PolyUOp {
  PolyOps op;
  PolyDType dtype;
  PolyUOp **src;
  uint16_t n_src;
  PolyArg arg;
  int32_t tag;
  PolyArg tag_arg;
  uint32_t hash;
  bool minmax_cached;
  int64_t minmax_vmin;
  int64_t minmax_vmax;
  void *ranges_cache;
  void *ended_ranges_cache;
};

/* Context owns the arena, CSE cache, schedule/to-program/runtime caches, and all UOps */
PolyCtx *poly_ctx_new(void);
void poly_ctx_destroy(PolyCtx *ctx);
void poly_ctx_set_preferred_device(PolyCtx *ctx, PolyDevice device);
PolyDevice poly_ctx_get_preferred_device(PolyCtx *ctx);
bool poly_ctx_owns_ptr(PolyCtx *ctx, const void *p);

typedef struct {
  size_t arena_bytes;
  size_t arena_high_water;
  size_t scratch_bytes;
  size_t scratch_high_water;
  size_t cse_entries;
  size_t schedule_cache_entries;
  size_t to_program_cache_entries;
  size_t runtime_cache_entries;
  /* Compatibility alias for runtime_cache_entries. */
  size_t program_cache_entries;
  size_t shape_cache_entries;
  size_t buffer_entries;
  size_t buffer_owned_bytes;
  size_t buffer_owned_current_bytes;
  size_t buffer_owned_source_bytes;
  size_t tensor_records;
  size_t registry_entries;
  size_t entrypoint_entries;
  size_t compiled_artifact_bytes;
  size_t runtime_artifact_entries;
  size_t launch_count;
  size_t schedule_cache_hits;
  size_t schedule_cache_misses;
  size_t runtime_cache_hits;
  size_t runtime_cache_misses;
  size_t buffer_read_count;
  size_t buffer_read_bytes;
  size_t buffer_write_count;
  size_t buffer_write_bytes;
  size_t buffer_copy_count;
  size_t buffer_copy_bytes;
  uint64_t global_ops;
  uint64_t global_mem;
  double time_sum_s;
  uint64_t kernel_count;
  uint64_t mem_used;
} PolyCtxStats;

int poly_ctx_stats(PolyCtx *ctx, PolyCtxStats *out);
void poly_ctx_reset_counters(PolyCtx *ctx);
uint64_t poly_ctx_mem_used_for_device(PolyCtx *ctx, PolyDevice device);

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
PolyUOp *poly_uop_tagged_arg(
    PolyCtx *ctx,
    PolyOps op,
    PolyDType dtype,
    PolyUOp **src,
    int n_src,
    PolyArg arg,
    int32_t tag,
    PolyArg tag_arg
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
 * enter_calls (false = skip CALL/FUNCTION src[0], process src[1:] only).
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

/* Owned toposort variants for local scans. These mirror tinygrad's temporary
 * `u.toposort()` result lifetime: the returned array is heap-owned and must be
 * released with poly_toposort_free(). UOp nodes themselves remain ctx-owned.
 * `ctx` may be NULL because owned traversal allocates no arena/scratch data. */
PolyUOp **poly_toposort_alloc(PolyCtx *ctx, PolyUOp *root, int *n_out);
PolyUOp **poly_toposort_ex_alloc(
    PolyCtx *ctx,
    PolyUOp *root,
    int *n_out,
    bool (*gate)(PolyUOp *),
    bool enter_calls
);
PolyUOp **poly_toposort_ex_user_alloc(
    PolyCtx *ctx,
    PolyUOp *root,
    int *n_out,
    bool (*gate)(PolyUOp *, void *),
    void *user_data,
    bool enter_calls
);
void poly_toposort_free(PolyUOp **topo);

/* Per-pass cache for UOp queries (ranges, vmin/vmax) *
 * Tinygrad caches every queryable UOp property as @functools.cached_property
 * on the immutable UOp instance, which gives per-UOp-lifetime memoization
 * for free. Polygrad's UOps are also immutable (arena-allocated, hash-consed),
 * and the default min/max query caches directly on the UOp. Callers may still
 * pass their own PolyUOpCache to scope batch/rewrite-local range queries and
 * throw those maps away cleanly.
 *
 * PolyUOpCache unifies the query maps that Phase D's reduce_collapse
 * driver uses together:
 *   - minmax: UOp inline cached (int64_t vmin, int64_t vmax), matching
 *     tinygrad's cached UOp._min_max property
 *   - ranges: UOp -> set of active RANGE ancestors per tinygrad u.ranges
 *   - ended_ranges: UOp -> set of ended RANGE ancestors per tinygrad u.ended_ranges
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
 * Lifetime: minmax values live inline on immutable UOps. Range-set values
 * are allocated from the PolyCtx arena and live until the ctx is destroyed,
 * matching tinygrad's UOp-lifetime cached properties. poly_uop_cache_destroy
 * frees the range-query PolyMap wrappers only. The minmax `_ex` entry accepts
 * a cache argument for API symmetry but does not allocate cache-owned values.
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
 * Every helper has a public entry point and an `_ex` variant that takes a
 * caller-owned PolyUOpCache for batch queries. Use the `_ex` form in hot loops
 * that need a pass-local cache distinct from UOp-local cached properties. */
/* Returns the terminal buffer-identity UOp (BUFFER / BUFFER_VIEW / PARAM)
 * after unwrapping RESHAPE/MULTI, or NULL if `u` has no buffer identity. */
const PolyUOp *poly_uop_get_buffer_identity(const PolyUOp *u);

/* True iff `u` is a buffer view (RESHAPE/MULTI over BUFFER/BUFFER_VIEW/PARAM).
 * Frontends use this to decide whether a tensor needs materialization
 * (matches tinygrad's UOp.has_buffer_identity). */
bool poly_uop_has_buffer_identity(const PolyUOp *u);

/* tinygrad UOp.buffer analogue. Returns a direct buffer identity, or the exact
 * movement UOp with an attached runtime view when a contiguous movement over
 * realized storage is provable. It does not create/rewrite graph topology or
 * change a Tensor root. */
PolyUOp *poly_uop_buffer(PolyCtx *ctx, PolyUOp *u);

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

PolyShape poly_uop_max_shape(PolyCtx *ctx, PolyUOp *u);
int64_t poly_shape_numel(PolyShape s);
bool poly_shape_eq(PolyShape a, PolyShape b);

/* Lazy cached shape accessors -- computes on first access, O(1) thereafter.
 * Returns arena-owned dims, do NOT free. */
int poly_uop_ndim(PolyCtx *ctx, const PolyUOp *u);
const int64_t *poly_uop_max_shape_dims(PolyCtx *ctx, const PolyUOp *u);
PolyUOp *poly_uop_shape_dim(PolyCtx *ctx, const PolyUOp *u, int dim);
/* Pinned UOp.axis query for multi-device shard propagation
 * (tinygrad/uop/ops.py:623-651). Returns false when the value is unsharded. */
bool poly_uop_axis(PolyCtx *ctx, const PolyUOp *u, int *out_axis);
int poly_uop_const_i64(const PolyUOp *u, int64_t *out);
PolyUOp *poly_uop_unbind_var(PolyUOp *u);
int poly_uop_bind_value(PolyUOp *u, int64_t *out);
PolyShape poly_uop_max_shape_cached(PolyCtx *ctx, const PolyUOp *u);
PolyArena *poly_ctx_arena(PolyCtx *ctx);

/* Named buffer registry.
 *
 * Compatibility API for old ctx-global instance construction. New C model
 * builders should use staged PolyInstance declarations from instance.h:
 * poly_instance_input/param/state/output/entrypoint/build. */

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
) POLY_DEPRECATED("use staged poly_instance_param/poly_instance_state")
    __attribute__((format(printf, 5, 6)));
PolyUOp *poly_input(
    PolyCtx *ctx,
    PolyDType dt,
    const int64_t *shape,
    int ndim,
    const char *fmt,
    ...
) POLY_DEPRECATED("use staged poly_instance_input") __attribute__((format(printf, 5, 6)));
PolyUOp *poly_output(
    PolyCtx *ctx,
    PolyDType dt,
    const int64_t *shape,
    int ndim,
    const char *fmt,
    ...
) POLY_DEPRECATED("use staged poly_instance_output") __attribute__((format(printf, 5, 6)));
PolyUOp *poly_target(
    PolyCtx *ctx,
    PolyDType dt,
    const int64_t *shape,
    int ndim,
    const char *fmt,
    ...
) POLY_DEPRECATED("use staged poly_instance_target") __attribute__((format(printf, 5, 6)));
PolyUOp *poly_aux(PolyCtx *ctx, PolyDType dt, const int64_t *shape, int ndim, const char *fmt, ...)
    POLY_DEPRECATED("use staged poly_instance_aux") __attribute__((format(printf, 5, 6)));

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
) POLY_DEPRECATED("use poly_instance_from_binding_arrays");

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
) POLY_DEPRECATED("use poly_instance_from_bindings/poly_instance_from_binding_arrays");

/* Create an alias: alias_name resolves to the same buffer as existing_name.
 * Returns 0 on success, -1 on error (existing_name not found, or alias_name
 * already taken by a different buffer). */
int poly_alias(PolyCtx *ctx, const char *alias_name, const char *existing_name)
    POLY_DEPRECATED("declare multiple state bindings for the same PolyTensor");

/* Lookup a named buffer by name. Returns the BUFFER UOp, or NULL. */
PolyUOp *poly_ctx_get(PolyCtx *ctx, const char *fmt, ...)
    POLY_DEPRECATED("ctx-global registry lookup is compatibility-only")
        __attribute__((format(printf, 2, 3)));

/* Lookup a registry entry by name. Returns NULL if not found. */
const PolyRegEntry *poly_ctx_get_entry(PolyCtx *ctx, const char *name)
    POLY_DEPRECATED("ctx-global registry lookup is compatibility-only");

/* Mark a named PARAM trainable/frozen. Non-PARAM buffers ignore optimizer
 * trainability but still carry the bit for import/export round-trips. */
int poly_ctx_set_trainable(PolyCtx *ctx, const char *name, bool trainable)
    POLY_DEPRECATED("set PolyTensor requires_grad before staged instance build");
bool poly_ctx_is_trainable(PolyCtx *ctx, const char *name)
    POLY_DEPRECATED("read trainability from PolyTensor or built PolyInstance");

/* Enumeration of all named entries (including aliases). */
int poly_ctx_named_count(PolyCtx *ctx) POLY_DEPRECATED("ctx-global registry is compatibility-only");
const PolyRegEntry *poly_ctx_named_entry(PolyCtx *ctx, int i)
    POLY_DEPRECATED("ctx-global registry is compatibility-only");

/* Register a named entrypoint (SINK UOp). Returns 0 on success, -1 on error. */
int poly_register_entrypoint(PolyCtx *ctx, const char *name, PolyUOp *sink)
    POLY_DEPRECATED("use staged poly_instance_entrypoint");

/* Entrypoint enumeration. */
int poly_ctx_entrypoint_count(PolyCtx *ctx)
    POLY_DEPRECATED("ctx-global entrypoints are compatibility-only");
const char *poly_ctx_entrypoint_name(PolyCtx *ctx, int i)
    POLY_DEPRECATED("ctx-global entrypoints are compatibility-only");
PolyUOp *poly_ctx_entrypoint_sink(PolyCtx *ctx, int i)
    POLY_DEPRECATED("ctx-global entrypoints are compatibility-only");

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

/* Extended multi-target gradient result. out_present is optional; when
 * supplied, each slot records whether reverse-mode produced a gradient before
 * the public zero-for-no-path fallback. This preserves tinygrad's distinction
 * between an absent/NOOP gradient and a numerically zero gradient. */
int poly_grad_many_ex(
    PolyCtx *ctx,
    PolyUOp *loss,
    PolyUOp *initial_grad,
    PolyUOp **wrts,
    int n,
    PolyUOp **out_grads,
    uint8_t *out_present
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
PolyUOp *poly_identity_element(PolyCtx *ctx, PolyOps op, PolyDType dtype);

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
PolyUOp *poly_expand_uop(PolyCtx *ctx, PolyUOp *src, PolyUOp **dims, int ndim);
PolyUOp *poly_permute(PolyCtx *ctx, PolyUOp *src, int64_t *perm, int ndim);
PolyUOp *poly_shrink(PolyCtx *ctx, PolyUOp *src, int64_t (*pairs)[2], int ndim);
PolyUOp *poly_shrink_uop(PolyCtx *ctx, PolyUOp *src, PolyUOp **starts, PolyUOp **sizes, int ndim);
PolyUOp *poly_pad_uop(PolyCtx *ctx, PolyUOp *src, PolyUOp **offsets, PolyUOp **sizes, int ndim);
PolyUOp *poly_flip(PolyCtx *ctx, PolyUOp *src, int64_t *axes, int n_axes);
PolyUOp *poly_pad(PolyCtx *ctx, PolyUOp *src, int64_t (*pairs)[2], int ndim);
PolyUOp *poly_reduce_axis(PolyCtx *ctx, PolyOps reduce_op, PolyUOp *src, int64_t *axes, int n_axes);
PolyUOp *poly_pad_value(PolyCtx *ctx, PolyUOp *x, int64_t (*pads)[2], int ndim, double value);
PolyUOp *poly_pool(
    PolyCtx *ctx,
    PolyUOp *x,
    const int64_t *k_,
    int nk,
    const int64_t *stride_,
    const int64_t *dilation_
);
PolyUOp *poly_max_pool2d(
    PolyCtx *ctx,
    PolyUOp *x,
    const int64_t *kernel,
    int n_kernel,
    const int64_t *stride,
    const int64_t *dilation,
    const int64_t *padding,
    int n_padding
);
PolyUOp *poly_conv2d(
    PolyCtx *ctx,
    PolyUOp *x,
    PolyUOp *weight,
    PolyUOp *bias,
    int groups,
    const int64_t *stride,
    const int64_t *dilation,
    const int64_t *padding,
    int n_padding
);
PolyUOp *poly_batchnorm(
    PolyCtx *ctx,
    PolyUOp *x,
    PolyUOp *weight,
    PolyUOp *bias,
    PolyUOp *mean,
    PolyUOp *invstd,
    const int64_t *axes,
    int n_axes
);
PolyUOp *poly_one_hot(PolyCtx *ctx, PolyUOp *x, int64_t num_classes);
PolyUOp *poly_index_select(PolyCtx *ctx, PolyUOp *x, int dim, PolyUOp *index);

#ifdef __cplusplus
}
#endif

#endif /* POLYGRAD_H */
