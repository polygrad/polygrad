/* core.h -- Shared ABI types, context and foundational utilities. */

#ifndef POLY_CORE_H
#define POLY_CORE_H

/* Public C/frontend ABI version. Bump when exported symbols or public struct
 * layouts used by frontends change. */
#define POLYGRAD_ABI_VERSION 100

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

/* Use poly_op_value(op) where tinygrad's Ops.value ordering is required. */

typedef enum {
  /* 1 — defines/special */
  POLY_OP_SPECIAL = 1,

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
  POLY_OP_STACK,
  POLY_OP_TUPLE,
  POLY_OP_GETTUPLE,

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
  POLY_OP_CONST,
  POLY_OP_CUSTOM,
  POLY_OP_CUSTOMI,
  POLY_OP_INS,

  /* 6 — ops that don't exist in programs */
  POLY_OP_UNIQUE,
  POLY_OP_DEVICE,
  POLY_OP_CONTIGUOUS,
  POLY_OP_CONTIGUOUS_BACKWARD,
  POLY_OP_DETACH,
  POLY_OP_STAGE,
  POLY_OP_COPY,
  POLY_OP_BUFFER,
  POLY_OP_MSELECT,
  POLY_OP_MSTACK,
  POLY_OP_CUSTOM_FUNCTION,
  POLY_OP_RESHAPE,
  POLY_OP_PERMUTE,
  POLY_OP_EXPAND,
  POLY_OP_PAD,
  POLY_OP_SHRINK,
  POLY_OP_FLIP,
  POLY_OP_UNSHARD,
  POLY_OP_REDUCE,
  POLY_OP_ALLREDUCE,

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
extern PolyOpSet POLY_GROUP_BROADCASTABLE;
extern PolyOpSet POLY_GROUP_ALU;
extern PolyOpSet POLY_GROUP_ELEMENTWISE;
extern PolyOpSet POLY_GROUP_MOVEMENT;
extern PolyOpSet POLY_GROUP_COMMUTATIVE;
extern PolyOpSet POLY_GROUP_ASSOCIATIVE;
extern PolyOpSet POLY_GROUP_IDEMPOTENT;
extern PolyOpSet POLY_GROUP_COMPARISON;
extern PolyOpSet POLY_GROUP_UNSAFEPAD;
extern PolyOpSet POLY_GROUP_DEFINES;
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
int poly_range_start(PolyOps op);

/* DType */

typedef enum {
  POLY_ADDR_GLOBAL = 0,
  POLY_ADDR_LOCAL,
  POLY_ADDR_REG,
  POLY_ADDR_ALU,
} PolyAddrSpace;

typedef struct {
  int8_t priority;
  uint16_t bitsize;
  const char *name; /* C type name, e.g. "float", "int" */
  char fmt; /* struct pack format char, 0 if none */
} PolyDType;

/* Predefined scalar dtypes */
extern const PolyDType POLY_VOID;
extern const PolyDType POLY_WEAKINT;
extern const PolyDType POLY_WEAKFLOAT;
extern const PolyDType POLY_BOOL;
extern const PolyDType POLY_INT8;
extern const PolyDType POLY_UINT8;
extern const PolyDType POLY_INT16;
extern const PolyDType POLY_UINT16;
extern const PolyDType POLY_INT32;
extern const PolyDType POLY_UINT32;
extern const PolyDType POLY_INT64;
extern const PolyDType POLY_UINT64;
extern const PolyDType POLY_FP8E4M3;
extern const PolyDType POLY_FP8E5M2;
extern const PolyDType POLY_FP8E4M3FNUZ;
extern const PolyDType POLY_FP8E5M2FNUZ;
extern const PolyDType POLY_FLOAT16;
extern const PolyDType POLY_BFLOAT16;
extern const PolyDType POLY_FLOAT32;
extern const PolyDType POLY_FLOAT64;

bool poly_dtype_eq(PolyDType a, PolyDType b);
bool poly_dtype_is_float(PolyDType dt);
bool poly_dtype_is_fp8(PolyDType dt);
bool poly_dtype_is_fp8_fnuz(PolyDType dt);
bool poly_dtype_is_int(PolyDType dt);
bool poly_dtype_is_index(PolyDType dt);
bool poly_dtype_is_weak(PolyDType dt);
bool poly_dtype_is_unsigned(PolyDType dt);
bool poly_dtype_is_bool(PolyDType dt);
PolyDType poly_dtype_strong(PolyDType dt);
/* Library-wide Tinygrad DEFAULT_FLOAT/INT dtype IDs. A setter returns -1 for
 * an unknown ID without changing the prior value. Configure while idle. */
int poly_get_default_float(void);
int poly_get_default_int(void);
int poly_set_default_float(int dtype_id);
int poly_set_default_int(int dtype_id);
PolyDType poly_dtype_weak(PolyDType dt);
bool poly_dtype_least_upper(PolyDType a, PolyDType b, PolyDType *out);
bool poly_dtype_least_upper_float(PolyDType dt, PolyDType *out);
bool poly_sum_acc_dtype(PolyDType dt, PolyDType *out);
bool poly_dtype_can_lossless_cast(PolyDType dt0, PolyDType dt1);
int poly_dtype_itemsize(PolyDType dt);
const char *poly_dtype_name(PolyDType dt);
int poly_dtype_count(void);
bool poly_dtype_by_id(int id, PolyDType *out);
int poly_dtype_id_by_name(const char *name);
uint8_t poly_float_to_fp8(double x, PolyDType dtype);
double poly_fp8_to_float(uint8_t x, PolyDType dtype);

/* Axis metadata for RANGE args (tinygrad AxisType parity) */

typedef enum {
  POLY_AXIS_DEVICE = 0,
  POLY_AXIS_GLOBAL,
  POLY_AXIS_WARP,
  POLY_AXIS_LOCAL,
  POLY_AXIS_WEAK,
  POLY_AXIS_GROUP_REDUCE,
  POLY_AXIS_REDUCE,
  POLY_AXIS_UPCAST,
  POLY_AXIS_UNROLL,
  POLY_AXIS_THREAD,
  POLY_AXIS_PLACEHOLDER,
  POLY_AXIS_LOOP,
} PolyAxisType;

/* PolyArg — tagged union for UOp arg field */

typedef struct PolyUOp PolyUOp;

/* Tinygrad 2026-08-22/a9069c177a9d uop/ops.py:1109-1152 ProgramInfo. */
typedef struct PolyProgramInfo {
  const char *name;
  const char *target;
  int global_size[3];
  int local_size[3];
  PolyUOp *global_exprs[3];
  PolyUOp *local_exprs[3];
  bool has_local_size;
  PolyUOp **vars;
  int n_vars;
  int *globals;
  int n_globals;
  int *outs;
  int n_outs;
  int *ins;
  int n_ins;
} PolyProgramInfo;

/* Current Tinygrad codegen/opt/__init__.py:OptOps and Opt. */
typedef enum {
  POLY_OPT_TC = 1,
  POLY_OPT_UPCAST,
  POLY_OPT_UNROLL,
  POLY_OPT_LOCAL,
  POLY_OPT_THREAD,
  POLY_OPT_GROUP,
  POLY_OPT_GROUPTOP,
  POLY_OPT_NOLOCALS,
  POLY_OPT_PADTO,
  POLY_OPT_SWAP,
} PolyOptOps;

typedef enum {
  POLY_OPT_ARG_NONE = 0,
  POLY_OPT_ARG_INT,
  POLY_OPT_ARG_INT_TUPLE,
} PolyOptArgKind;

typedef struct {
  PolyOptOps op;
  bool has_axis;
  int axis;
  PolyOptArgKind arg_kind;
  int64_t arg;
  const int64_t *arg_tuple;
  int n_arg_tuple;
} PolyOpt;

/* tinygrad.renderer.Estimates; expressions remain symbolic until execution. */
typedef struct {
  PolyUOp *ops;
  PolyUOp *lds;
  PolyUOp *mem;
} PolyEstimates;

/* Current Tinygrad uop/ops.py:KernelInfo. */
typedef struct {
  const char *name;
  const PolyAxisType *axis_types;
  int n_axis_types;
  bool dont_use_locals;
  const PolyOpt *applied_opts;
  int n_applied_opts;
  const PolyOpt *opts_to_apply;
  int n_opts_to_apply;
  bool has_opts_to_apply;
  const PolyEstimates *estimates;
  int beam;
} PolyKernelInfo;

/* Pinned tinygrad ParamArg metadata. Value-level call PARAMs carry shape in
 * src[0] and keep slot/device here; rangeify later lowers them to the existing
 * pointer PARAM form whose arg is the kernel-local integer slot. */
typedef struct PolyParamArg PolyParamArg;

/* Tinygrad 2026-08-22/a9069c177a9d CallInfo for CALL/FUNCTION UOps
 * (uop/ops.py:1261-1271). C stores the serializable fields and fail-closed
 * presence bits for Python callback/aux values. */
typedef struct {
  const char *name;
  bool precompile;
  bool precompile_backward;
  bool has_grad_fxn;
  uint32_t grad_fxn_key;
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
  POLY_ARG_STRING,
  POLY_ARG_OPS,
  POLY_ARG_REDUCE, /* current tinygrad tensor/lowered REDUCE: (PolyOps, num_axes) */
  POLY_ARG_ALLREDUCE, /* current tinygrad ALLREDUCE: (PolyOps, str|tuple[str,...]) */
  POLY_ARG_RANGE, /* (axis_id, axis_type, extra...) */
  POLY_ARG_BUFFERIZE_OPTS, /* (exact scalar/tuple device, addrspace, removable) */
  POLY_ARG_TENSOR_CORE, /* tinygrad WMMA metadata: (dims,dtype_in,device,threads,upcast_axes) */
  POLY_ARG_PROGRAM_INFO, /* PolyProgramInfo* value metadata for PROGRAM */
  POLY_ARG_BYTES, /* immutable runtime bytes for BINARY UOps */
  POLY_ARG_INVALID,
  POLY_ARG_PARAM, /* pinned tinygrad ParamArg* for shaped value PARAMs */
  POLY_ARG_BIGINT, /* exact signed arbitrary-precision integer CONST */
  POLY_ARG_STRING_TUPLE, /* ordered immutable string tuple (for DEVICE.arg) */
  POLY_ARG_CALL_INFO, /* pinned tinygrad CallInfo* for CALL/FUNCTION */
  POLY_ARG_DTYPE, /* pinned DType arg for CAST/BITCAST */
  POLY_ARG_KERNEL_INFO, /* pinned tinygrad KernelInfo* for compiler SINK */
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
      const char **vals;
      int n;
    } string_tuple;
    const char *str;
    PolyOps ops;
    struct {
      PolyOps op;
      int num_axes;
    } reduce;
    struct {
      PolyOps op;
      const char *device;
      const char **devices;
      int32_t n_devices;
      bool device_is_tuple;
    } allreduce;
    struct {
      int64_t axis_id;
      PolyAxisType axis_type;
      int64_t *extra;
      int n_extra;
    } range;
    struct {
      /* Pinned BufferizeOpts.device is str | tuple[str, ...] | int | None
       * (tinygrad/schedule/indexing.py:57-62). Global physical scheduling uses
       * the scalar/tuple string arms; local integer ids remain addrspace-local. */
      const char *device;
      const char **devices;
      int32_t n_devices;
      bool device_is_tuple;
      PolyAddrSpace addrspace;
      bool removable;
      bool device_is_int;
      int64_t device_int;
    } bufferize_opts;
    struct {
      int dims[3];
      PolyDType dtype_in;
      const char *device;
      int threads;
      int64_t (*upcast_axes[3])[2];
      int n_upcast_axes[3];
      bool has_upcast_axes;
    } tensor_core;
    const PolyProgramInfo *program_info;
    const PolyKernelInfo *kernel_info;
    const PolyParamArg *param;
    const PolyCallInfo *call_info;
    PolyDType dtype;
    struct {
      const uint8_t *data;
      int n;
    } bytes;
  };
} PolyArg;

struct PolyParamArg {
  int64_t slot;
  PolyDType dtype;
  const char *name;
  /* ParamArg.vmin_vmax: ordered scalar INT/FLOAT/BOOL/BIGINT values,
   * independent of dtype. UOp storage owns copies, including integer limbs. */
  PolyArg min_val;
  PolyArg max_val;
  bool has_minmax;
  int64_t multiple_of;
  bool has_multiple_of;
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
  bool volatile_;
};

static inline PolyArg poly_arg_none(void) {
  return (PolyArg){.kind = POLY_ARG_NONE};
}
static inline PolyArg poly_arg_int(int64_t v) {
  return (PolyArg){.kind = POLY_ARG_INT, .i = v};
}
static inline PolyArg poly_arg_bigint(int sign, const uint32_t *limbs, uint32_t n_limbs) {
  return (PolyArg
  ){.kind = POLY_ARG_BIGINT,
    .bigint = {.sign = (int8_t)(sign < 0 ? -1 : 1), .n_limbs = n_limbs, .limbs = limbs}};
}
static inline PolyArg poly_arg_float(double v) {
  return (PolyArg){.kind = POLY_ARG_FLOAT, .f = v};
}
static inline PolyArg poly_arg_bool(bool v) {
  return (PolyArg){.kind = POLY_ARG_BOOL, .b = v};
}
static inline PolyArg poly_arg_int_tuple(int64_t *vals, int n) {
  return (PolyArg){.kind = POLY_ARG_INT_TUPLE, .int_tuple = {.vals = vals, .n = n}};
}
static inline PolyArg poly_arg_ops(PolyOps op) {
  return (PolyArg){.kind = POLY_ARG_OPS, .ops = op};
}
static inline PolyArg poly_arg_reduce(PolyOps op, int num_axes) {
  return (PolyArg){.kind = POLY_ARG_REDUCE, .reduce = {.op = op, .num_axes = num_axes}};
}
static inline PolyArg poly_arg_allreduce(
    PolyOps op,
    const char *device,
    const char **devices,
    int32_t n_devices
) {
  return (PolyArg){
      .kind = POLY_ARG_ALLREDUCE,
      .allreduce =
          {
              .op = op,
              .device = device,
              .devices = devices,
              .n_devices = n_devices,
              .device_is_tuple = devices != NULL,
          },
  };
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
static inline PolyArg poly_arg_dtype(PolyDType dtype) {
  return (PolyArg){.kind = POLY_ARG_DTYPE, .dtype = dtype};
}
static inline PolyArg poly_arg_kernel_info(const PolyKernelInfo *info) {
  return (PolyArg){.kind = POLY_ARG_KERNEL_INFO, .kernel_info = info};
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

/* tinygrad BufferizeOpts equivalent.
 *
 * device is the exact canonical physical identity for the intermediate buffer
 * (or NULL when unknown), addrspace selects global/local storage, and
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
/* BufferizeOpts.device on LOCAL stages names a reduction, not a backend. */
static inline PolyArg poly_arg_bufferize_opts_int(
    int64_t device,
    PolyAddrSpace addrspace,
    bool removable
) {
  return (PolyArg
  ){.kind = POLY_ARG_BUFFERIZE_OPTS,
    .bufferize_opts = {
        .addrspace = addrspace,
        .removable = removable,
        .device_is_int = true,
        .device_int = device}};
}
static inline PolyArg poly_arg_tensor_core(
    const int dims[3],
    PolyDType dtype_in,
    const char *device,
    int threads,
    int64_t (*upcast_axes[3])[2],
    const int n_upcast_axes[3],
    bool has_upcast_axes
) {
  return (PolyArg
  ){.kind = POLY_ARG_TENSOR_CORE,
    .tensor_core = {
        .dims = {dims ? dims[0] : 0, dims ? dims[1] : 0, dims ? dims[2] : 0},
        .dtype_in = dtype_in,
        .device = device,
        .threads = threads,
        .upcast_axes =
            {
                upcast_axes ? upcast_axes[0] : NULL,
                upcast_axes ? upcast_axes[1] : NULL,
                upcast_axes ? upcast_axes[2] : NULL,
            },
        .n_upcast_axes =
            {
                n_upcast_axes ? n_upcast_axes[0] : 0,
                n_upcast_axes ? n_upcast_axes[1] : 0,
                n_upcast_axes ? n_upcast_axes[2] : 0,
            },
        .has_upcast_axes = has_upcast_axes,
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
bool poly_kernel_info_eq(const PolyKernelInfo *a, const PolyKernelInfo *b);
uint32_t poly_kernel_info_hash(const PolyKernelInfo *info);

/* RANGE metadata helpers. New RANGE UOps store POLY_ARG_RANGE; poly_uop()
 * canonicalizes legacy POLY_ARG_INT range ids to WEAK ranges at creation. */
static inline int64_t poly_range_axis_id(PolyArg a) {
  if (a.kind == POLY_ARG_RANGE) return a.range.axis_id;
  if (a.kind == POLY_ARG_INT) return a.i;
  return -1;
}
static inline PolyAxisType poly_range_axis_type(PolyArg a) {
  if (a.kind == POLY_ARG_RANGE) return a.range.axis_type;
  return POLY_AXIS_WEAK;
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
  if (a.kind == POLY_ARG_BUFFERIZE_OPTS && !a.bufferize_opts.device_is_tuple &&
      !a.bufferize_opts.device_is_int)
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
  POLY_DEVICE_HOST, /* Tinygrad PYTHON storage/runtime; may own frontend bytes */
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
/* POLY_DEV > DEV > platform default; AUTO means an unsupported target. */
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
typedef struct PolyJit PolyJit;
typedef struct PolyTensor PolyTensor;

/* Polygrad portable-graph lifetime policy. The physical parity graph is
 * mandatory and independent of this approved logical/placement boundary. */
typedef enum {
  POLY_LOGICAL_NEVER = 0,
  POLY_LOGICAL_ALWAYS = 1,
  POLY_LOGICAL_UNTIL_REALIZE = 2,
} PolyLogicalPolicy;

typedef enum {
  POLY_LOGICAL_AVAILABLE = 0,
  POLY_LOGICAL_NEVER_CONSTRUCTED = 1,
  POLY_LOGICAL_RETIRED = 2,
  POLY_LOGICAL_UNSUPPORTED_RESOURCE = 3,
} PolyLogicalState;

typedef struct PolyVarBinding {
  PolyUOp *var; /* ALU BUFFER variable */
  int64_t value; /* Numeric vals, not PARAM-dtype bits; same domain as poly_uop_bind. */
} PolyVarBinding;

/* Frontend host-buffer lifetime hook.
 * Frontends keep strong maps keyed by the C-side PolyBuffer* address value.
 * When the core retires an imported HOST residency, it calls the registered
 * release function with that key so the frontend can drop its owner entry. */
typedef void (*PolyFrontendBufferReleaseFn)(uintptr_t buffer_key);
void poly_set_frontend_buffer_release(PolyFrontendBufferReleaseFn fn);
void poly_ctx_set_frontend_buffer_release(PolyCtx *ctx, PolyFrontendBufferReleaseFn fn);

/* PolyBuffer is defined in device.h (needs PolyAllocator pointer) */

/* Every PolyUOp pointer returned by this C API is borrowed until the next
 * collection safe point. Call poly_uop_retain() before storing it beyond its
 * Tensor/Model/JIT owner, then pair it with poly_uop_release(). Collection
 * may reclaim both residency and an unretained UOp record. */

/* Context owns the arena, CSE, to_program/runtime caches, and all UOps. */
/* Version query is public to native C and language bindings alike. */
int poly_abi_version(void);
PolyCtx *poly_ctx_new(void);
void poly_ctx_destroy(PolyCtx *ctx);
/* Explicit safe-point collection for retired owners. */
int poly_ctx_collect(PolyCtx *ctx);
/* C observability for current Tinygrad's module-level schedule_cache. */
size_t poly_schedule_cache_len(PolyCtx *ctx);
/* Explicit schedule_cache.clear(): release cache ownership only. The caller
 * must serialize access and finish queued work. Returns -1 during execution,
 * JIT capture or collection; otherwise 0, including an empty cache. Borrowed
 * UOps need an independent owner before a subsequent poly_ctx_collect(). */
int poly_schedule_cache_clear(PolyCtx *ctx);
void poly_ctx_set_preferred_device(PolyCtx *ctx, PolyDevice device);
PolyDevice poly_ctx_get_preferred_device(PolyCtx *ctx);
int poly_ctx_set_logical_policy(PolyCtx *ctx, PolyLogicalPolicy policy);
PolyLogicalPolicy poly_ctx_get_logical_policy(const PolyCtx *ctx);
bool poly_ctx_owns_ptr(PolyCtx *ctx, const void *p);

typedef struct {
  size_t arena_bytes;
  size_t arena_high_water;
  size_t scratch_bytes;
  size_t scratch_high_water;
  size_t cse_entries;
  size_t to_program_cache_entries;
  size_t runtime_cache_entries;
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

/* Named buffer registry.
 *
 * Compatibility API for old ctx-global instance construction. New C model
 * builders should use staged PolyModel declarations from model.h:
 * poly_model_input/param/state/output/entrypoint/build. */

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
) POLY_DEPRECATED("use staged poly_model_param/poly_model_state")
    __attribute__((format(printf, 5, 6)));
PolyUOp *poly_input(
    PolyCtx *ctx,
    PolyDType dt,
    const int64_t *shape,
    int ndim,
    const char *fmt,
    ...
) POLY_DEPRECATED("use staged poly_model_input") __attribute__((format(printf, 5, 6)));
PolyUOp *poly_output(
    PolyCtx *ctx,
    PolyDType dt,
    const int64_t *shape,
    int ndim,
    const char *fmt,
    ...
) POLY_DEPRECATED("use staged poly_model_output") __attribute__((format(printf, 5, 6)));
PolyUOp *poly_target(
    PolyCtx *ctx,
    PolyDType dt,
    const int64_t *shape,
    int ndim,
    const char *fmt,
    ...
) POLY_DEPRECATED("use staged poly_model_target") __attribute__((format(printf, 5, 6)));
PolyUOp *poly_aux(PolyCtx *ctx, PolyDType dt, const int64_t *shape, int ndim, const char *fmt, ...)
    POLY_DEPRECATED("use staged poly_model_aux") __attribute__((format(printf, 5, 6)));

/* Non-variadic typed registration for the deprecated context registry. */
PolyUOp *poly_register_buffer(
    PolyCtx *ctx,
    int role,
    PolyDType dtype,
    const int64_t *shape,
    int ndim,
    const char *name
) POLY_DEPRECATED("use poly_model_from_binding_arrays");

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
) POLY_DEPRECATED("use poly_model_from_bindings/poly_model_from_binding_arrays");

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
    POLY_DEPRECATED("use PolyModel trainability metadata");
bool poly_ctx_is_trainable(PolyCtx *ctx, const char *name)
    POLY_DEPRECATED("read trainability from the built PolyModel");

/* Enumeration of all named entries (including aliases). */
int poly_ctx_named_count(PolyCtx *ctx) POLY_DEPRECATED("ctx-global registry is compatibility-only");
const PolyRegEntry *poly_ctx_named_entry(PolyCtx *ctx, int i)
    POLY_DEPRECATED("ctx-global registry is compatibility-only");

/* Register a named entrypoint (SINK UOp). Returns 0 on success, -1 on error. */
int poly_register_entrypoint(PolyCtx *ctx, const char *name, PolyUOp *sink)
    POLY_DEPRECATED("use staged poly_model_entrypoint");

/* Entrypoint enumeration. */
int poly_ctx_entrypoint_count(PolyCtx *ctx)
    POLY_DEPRECATED("ctx-global entrypoints are compatibility-only");
const char *poly_ctx_entrypoint_name(PolyCtx *ctx, int i)
    POLY_DEPRECATED("ctx-global entrypoints are compatibility-only");
PolyUOp *poly_ctx_entrypoint_sink(PolyCtx *ctx, int i)
    POLY_DEPRECATED("ctx-global entrypoints are compatibility-only");

/* Tinygrad helpers.NOOPT: library-wide compilation policy, initialized from
 * the environment once, not graph state. Existing PROGRAMs are unchanged. */
int poly_get_noopt(void);
void poly_set_noopt(int value);

/* Tinygrad helpers.BEAM; the same library/module ownership as NOOPT. */
int poly_get_beam(void);
void poly_set_beam(int value);
int poly_get_ignore_beam_cache(void);
void poly_set_ignore_beam_cache(int value);

#ifdef __cplusplus
}
#endif

#endif /* POLY_CORE_H */
