/*
 * uop.c — UOp creation with CSE, toposort, pretty-print
 *
 * Mirrors tinygrad's UOp class and UOpMetaClass hash-consing cache.
 * All UOps are allocated from the context's arena.
 */

#include "polygrad.h"
#include "bigint.h"
#include "utils.h"
#include "ctx.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* PolyArg equality and hashing */

bool poly_arg_eq(PolyArg a, PolyArg b) {
  if (a.kind != b.kind) return false;
  switch (a.kind) {
  case POLY_ARG_NONE:
    return true;
  case POLY_ARG_INVALID:
    return true;
  case POLY_ARG_INT:
    return a.i == b.i;
  case POLY_ARG_BIGINT:
    return a.bigint.sign == b.bigint.sign && a.bigint.n_limbs == b.bigint.n_limbs &&
           (a.bigint.n_limbs == 0 ||
            (a.bigint.limbs && b.bigint.limbs &&
             memcmp(
                 a.bigint.limbs, b.bigint.limbs,
                 (size_t)a.bigint.n_limbs * sizeof(uint32_t)
             ) == 0));
  case POLY_ARG_FLOAT: /* bitwise compare to distinguish -0.0 from 0.0 */
  {
    uint64_t ba, bb;
    memcpy(&ba, &a.f, 8);
    memcpy(&bb, &b.f, 8);
    return ba == bb;
  }
  case POLY_ARG_BOOL:
    return a.b == b.b;
  case POLY_ARG_STRING:
    return a.str == b.str || (a.str && b.str && strcmp(a.str, b.str) == 0);
  case POLY_ARG_STRING_TUPLE:
    if (a.string_tuple.n != b.string_tuple.n) return false;
    for (int i = 0; i < a.string_tuple.n; i++) {
      const char *av = a.string_tuple.vals ? a.string_tuple.vals[i] : NULL;
      const char *bv = b.string_tuple.vals ? b.string_tuple.vals[i] : NULL;
      if (av != bv && (!av || !bv || strcmp(av, bv) != 0)) return false;
    }
    return true;
  case POLY_ARG_OPS:
    return a.ops == b.ops;
  case POLY_ARG_INT_TUPLE:
    if (a.int_tuple.n != b.int_tuple.n) return false;
    if (a.int_tuple.n == 0) return true;
    return memcmp(a.int_tuple.vals, b.int_tuple.vals, a.int_tuple.n * sizeof(int64_t)) == 0;
  case POLY_ARG_PAIR_TUPLE:
    if (a.pair_tuple.n != b.pair_tuple.n) return false;
    if (a.pair_tuple.n == 0) return true;
    return memcmp(a.pair_tuple.pairs, b.pair_tuple.pairs, a.pair_tuple.n * 2 * sizeof(int64_t)) ==
           0;
  case POLY_ARG_REDUCE_AXIS:
    if (a.reduce_axis.op != b.reduce_axis.op) return false;
    if (a.reduce_axis.n != b.reduce_axis.n) return false;
    if (a.reduce_axis.n == 0) return true;
    return memcmp(a.reduce_axis.axes, b.reduce_axis.axes, a.reduce_axis.n * sizeof(int64_t)) == 0;
  case POLY_ARG_RANGE:
    if (a.range.axis_id != b.range.axis_id) return false;
    if (a.range.axis_type != b.range.axis_type) return false;
    if (a.range.n_extra != b.range.n_extra) return false;
    if (a.range.n_extra == 0) return true;
    return memcmp(a.range.extra, b.range.extra, (size_t)a.range.n_extra * sizeof(int64_t)) == 0;
  case POLY_ARG_DEFINE_VAR:
    if (a.define_var.min_val != b.define_var.min_val) return false;
    if (a.define_var.max_val != b.define_var.max_val) return false;
    if (a.define_var.name == b.define_var.name) return true;
    if (!a.define_var.name || !b.define_var.name) return false;
    return strcmp(a.define_var.name, b.define_var.name) == 0;
  case POLY_ARG_BUFFERIZE_OPTS:
    return (a.bufferize_opts.device == b.bufferize_opts.device ||
            (a.bufferize_opts.device && b.bufferize_opts.device &&
             strcmp(a.bufferize_opts.device, b.bufferize_opts.device) == 0)) &&
           a.bufferize_opts.addrspace == b.bufferize_opts.addrspace &&
           a.bufferize_opts.removable == b.bufferize_opts.removable;
  case POLY_ARG_TENSOR_CORE:
    if (a.tensor_core.threads != b.tensor_core.threads ||
        memcmp(a.tensor_core.dims, b.tensor_core.dims, sizeof(a.tensor_core.dims)) != 0)
      return false;
    if (a.tensor_core.name == b.tensor_core.name) return true;
    return a.tensor_core.name && b.tensor_core.name &&
           strcmp(a.tensor_core.name, b.tensor_core.name) == 0;
  case POLY_ARG_PROGRAM_INFO:
    return poly_program_info_eq(a.program_info, b.program_info);
  case POLY_ARG_BYTES:
    if (a.bytes.n != b.bytes.n) return false;
    if (a.bytes.n == 0) return true;
    if (!a.bytes.data || !b.bytes.data) return false;
    return memcmp(a.bytes.data, b.bytes.data, (size_t)a.bytes.n) == 0;
  case POLY_ARG_PARAM:
    if (a.param == b.param) return true;
    if (!a.param || !b.param) return false;
    if (a.param->slot != b.param->slot || a.param->min_val != b.param->min_val ||
        a.param->max_val != b.param->max_val || a.param->has_minmax != b.param->has_minmax ||
        a.param->addrspace != b.param->addrspace || a.param->axis != b.param->axis ||
        a.param->has_axis != b.param->has_axis ||
        a.param->device_is_tuple != b.param->device_is_tuple ||
        a.param->n_devices != b.param->n_devices)
      return false;
    if (a.param->device != b.param->device &&
        (!a.param->device || !b.param->device || strcmp(a.param->device, b.param->device) != 0))
      return false;
    for (int i = 0; i < a.param->n_devices; i++) {
      const char *av = a.param->devices ? a.param->devices[i] : NULL;
      const char *bv = b.param->devices ? b.param->devices[i] : NULL;
      if (av != bv && (!av || !bv || strcmp(av, bv) != 0)) return false;
    }
    if (a.param->name == b.param->name) return true;
    return a.param->name && b.param->name && strcmp(a.param->name, b.param->name) == 0;
  }
  return false;
}

static uint32_t hash_mix(uint32_t h, uint32_t v) {
  h ^= v;
  h *= 0x9e3779b9;
  h ^= h >> 16;
  return h;
}

uint32_t poly_arg_hash(PolyArg a) {
  uint32_t h = (uint32_t)a.kind;
  switch (a.kind) {
  case POLY_ARG_NONE:
  case POLY_ARG_INVALID:
    break;
  case POLY_ARG_INT:
    h = hash_mix(h, (uint32_t)(a.i ^ (a.i >> 32)));
    break;
  case POLY_ARG_BIGINT:
    h = hash_mix(h, (uint32_t)(int32_t)a.bigint.sign);
    h = hash_mix(h, a.bigint.n_limbs);
    for (uint32_t i = 0; i < a.bigint.n_limbs; i++)
      h = hash_mix(h, a.bigint.limbs ? a.bigint.limbs[i] : 0);
    break;
  case POLY_ARG_FLOAT: {
    uint64_t bits;
    memcpy(&bits, &a.f, 8);
    h = hash_mix(h, (uint32_t)(bits ^ (bits >> 32)));
  } break;
  case POLY_ARG_BOOL:
    h = hash_mix(h, a.b ? 1 : 0);
    break;
  case POLY_ARG_STRING:
    if (a.str) {
      for (const char *p = a.str; *p; p++)
        h = hash_mix(h, (uint32_t)*p);
    }
    break;
  case POLY_ARG_STRING_TUPLE:
    h = hash_mix(h, (uint32_t)a.string_tuple.n);
    for (int i = 0; i < a.string_tuple.n; i++) {
      const char *value = a.string_tuple.vals ? a.string_tuple.vals[i] : NULL;
      if (value)
        for (const char *p = value; *p; p++)
          h = hash_mix(h, (uint32_t)*p);
      h = hash_mix(h, UINT32_C(0xff));
    }
    break;
  case POLY_ARG_OPS:
    h = hash_mix(h, (uint32_t)a.ops);
    break;
  case POLY_ARG_INT_TUPLE:
    for (int i = 0; i < a.int_tuple.n; i++)
      h = hash_mix(h, (uint32_t)(a.int_tuple.vals[i] ^ (a.int_tuple.vals[i] >> 32)));
    break;
  case POLY_ARG_PAIR_TUPLE:
    for (int i = 0; i < a.pair_tuple.n; i++) {
      h = hash_mix(h, (uint32_t)(a.pair_tuple.pairs[i][0] ^ (a.pair_tuple.pairs[i][0] >> 32)));
      h = hash_mix(h, (uint32_t)(a.pair_tuple.pairs[i][1] ^ (a.pair_tuple.pairs[i][1] >> 32)));
    }
    break;
  case POLY_ARG_REDUCE_AXIS:
    h = hash_mix(h, (uint32_t)a.reduce_axis.op);
    for (int i = 0; i < a.reduce_axis.n; i++)
      h = hash_mix(h, (uint32_t)(a.reduce_axis.axes[i] ^ (a.reduce_axis.axes[i] >> 32)));
    break;
  case POLY_ARG_RANGE:
    h = hash_mix(h, (uint32_t)(a.range.axis_id ^ (a.range.axis_id >> 32)));
    h = hash_mix(h, (uint32_t)a.range.axis_type);
    for (int i = 0; i < a.range.n_extra; i++)
      h = hash_mix(h, (uint32_t)(a.range.extra[i] ^ (a.range.extra[i] >> 32)));
    break;
  case POLY_ARG_DEFINE_VAR:
    if (a.define_var.name) {
      for (const char *p = a.define_var.name; *p; p++)
        h = hash_mix(h, (uint32_t)*p);
    }
    h = hash_mix(h, (uint32_t)(a.define_var.min_val ^ (a.define_var.min_val >> 32)));
    h = hash_mix(h, (uint32_t)(a.define_var.max_val ^ (a.define_var.max_val >> 32)));
    break;
  case POLY_ARG_BUFFERIZE_OPTS:
    if (a.bufferize_opts.device)
      for (const char *p = a.bufferize_opts.device; *p; p++)
        h = hash_mix(h, (uint32_t)*p);
    h = hash_mix(h, (uint32_t)a.bufferize_opts.addrspace);
    h = hash_mix(h, a.bufferize_opts.removable ? 1u : 0u);
    break;
  case POLY_ARG_TENSOR_CORE:
    if (a.tensor_core.name) {
      for (const char *p = a.tensor_core.name; *p; p++)
        h = hash_mix(h, (uint32_t)*p);
    }
    for (int i = 0; i < 3; i++) h = hash_mix(h, (uint32_t)a.tensor_core.dims[i]);
    h = hash_mix(h, (uint32_t)a.tensor_core.threads);
    break;
  case POLY_ARG_PROGRAM_INFO:
    h = hash_mix(h, poly_program_info_hash(a.program_info));
    break;
  case POLY_ARG_BYTES:
    for (int i = 0; i < a.bytes.n; i++)
      h = hash_mix(h, a.bytes.data ? a.bytes.data[i] : 0);
    break;
  case POLY_ARG_PARAM:
    if (a.param) {
      h = hash_mix(h, (uint32_t)(a.param->slot ^ (a.param->slot >> 32)));
      if (a.param->device)
        for (const char *p = a.param->device; *p; p++)
          h = hash_mix(h, (uint32_t)*p);
      h = hash_mix(h, a.param->device_is_tuple ? 1u : 0u);
      h = hash_mix(h, (uint32_t)a.param->n_devices);
      for (int i = 0; i < a.param->n_devices; i++) {
        const char *device = a.param->devices ? a.param->devices[i] : NULL;
        if (device)
          for (const char *p = device; *p; p++) h = hash_mix(h, (uint32_t)*p);
        h = hash_mix(h, UINT32_C(0xff));
      }
      h = hash_mix(h, (uint32_t)a.param->addrspace);
      h = hash_mix(h, (uint32_t)a.param->axis);
      h = hash_mix(h, a.param->has_axis ? 1u : 0u);
      h = hash_mix(h, a.param->has_minmax ? 1u : 0u);
      h = hash_mix(h, (uint32_t)(a.param->min_val ^ (a.param->min_val >> 32)));
      h = hash_mix(h, (uint32_t)(a.param->max_val ^ (a.param->max_val >> 32)));
      if (a.param->name) {
        for (const char *p = a.param->name; *p; p++) h = hash_mix(h, (uint32_t)*p);
      }
    }
    break;
  }
  return h;
}

/* CSE key: (op, dtype, src[], arg, tag) */

typedef struct {
  PolyOps op;
  PolyDType dtype;
  PolyUOp **src;
  uint16_t n_src;
  PolyArg arg;
  int32_t tag;
  PolyArg tag_arg;
} CseKey;

static uint32_t cse_hash(const CseKey *k) {
  uint32_t h = (uint32_t)k->op;
  h = hash_mix(h, (uint32_t)k->dtype.priority);
  h = hash_mix(h, (uint32_t)k->dtype.bitsize);
  h = hash_mix(h, (uint32_t)k->dtype.count);
  h = hash_mix(h, k->dtype.is_ptr ? 1u : 0u);
  h = hash_mix(h, (uint32_t)k->dtype.addrspace);
  h = hash_mix(h, (uint32_t)k->dtype.vcount);
  h = hash_mix(h, (uint32_t)(k->dtype.ptr_size ^ (k->dtype.ptr_size >> 32)));
  if (k->dtype.name) {
    const unsigned char *p = (const unsigned char *)k->dtype.name;
    while (*p)
      h = hash_mix(h, (uint32_t)*p++);
  }
  for (int i = 0; i < k->n_src; i++) {
    /* hash pointer as integer — identity-based like tinygrad */
    uintptr_t p = (uintptr_t)k->src[i];
    h = hash_mix(h, (uint32_t)(p ^ (sizeof(p) > 4 ? (uint32_t)(p >> 32) : 0)));
  }
  h = hash_mix(h, poly_arg_hash(k->arg));
  h = hash_mix(h, (uint32_t)k->tag);
  h = hash_mix(h, poly_arg_hash(k->tag_arg));
  return h;
}

static bool cse_eq(const void *a, const void *b) {
  const CseKey *ka = a, *kb = b;
  if (ka->op != kb->op) return false;
  if (!poly_dtype_eq(ka->dtype, kb->dtype)) return false;
  if (ka->n_src != kb->n_src) return false;
  for (int i = 0; i < ka->n_src; i++)
    if (ka->src[i] != kb->src[i]) return false;
  if (!poly_arg_eq(ka->arg, kb->arg)) return false;
  if (ka->tag != kb->tag) return false;
  if (!poly_arg_eq(ka->tag_arg, kb->tag_arg)) return false;
  return true;
}

/* struct PolyCtx and lifecycle are in ctx.h / ctx.c */

/* UOp creation with CSE */

static bool rank_tuple_valid(const void *data, int n) {
  return n >= 0 && n <= POLY_MAX_DIMS && (n == 0 || data != NULL);
}

static bool shape_value_rank_valid(PolyUOp *shape) {
  if (!shape) return false;
  int rank = shape->op == POLY_OP_STACK
                 ? shape->n_src
                 : (shape->op == POLY_OP_CONST ? shape->dtype.count : 1);
  return rank >= 0 && rank <= POLY_MAX_DIMS;
}

static bool string_tuple_valid(const char **vals, int n) {
  if (n < 0 || n > UINT16_MAX || (n > 0 && !vals)) return false;
  for (int i = 0; i < n; i++)
    if (!vals[i] || !vals[i][0]) return false;
  return true;
}

static bool uop_rank_arg_valid(PolyOps op, PolyUOp **src, int n_src, PolyArg arg) {
  switch (op) {
  case POLY_OP_DEVICE:
    return arg.kind == POLY_ARG_NONE || (arg.kind == POLY_ARG_STRING && arg.str && arg.str[0]) ||
           (arg.kind == POLY_ARG_STRING_TUPLE &&
            string_tuple_valid(arg.string_tuple.vals, arg.string_tuple.n));
  case POLY_OP_RESHAPE:
    /* Pinned spec.py accepts any shape-value UOp in src[1]. UOp.as_shape
     * decodes CONST, STACK, and scalar symbolic expressions (ops.py:697-700). */
    return arg.kind == POLY_ARG_NONE && n_src == 2 && src &&
           shape_value_rank_valid(src[1]);
  case POLY_OP_PERMUTE:
    return arg.kind == POLY_ARG_INT_TUPLE && rank_tuple_valid(arg.int_tuple.vals, arg.int_tuple.n);
  case POLY_OP_FLIP:
    if (arg.kind != POLY_ARG_INT_TUPLE || !rank_tuple_valid(arg.int_tuple.vals, arg.int_tuple.n))
      return false;
    for (int i = 0; i < arg.int_tuple.n; i++)
      if (arg.int_tuple.vals[i] != 0 && arg.int_tuple.vals[i] != 1) return false;
    return true;
  case POLY_OP_EXPAND:
    return arg.kind == POLY_ARG_NONE && n_src == 2 && src &&
           shape_value_rank_valid(src[1]);
  case POLY_OP_SHRINK:
    if (arg.kind == POLY_ARG_NONE) return true;
    return arg.kind == POLY_ARG_PAIR_TUPLE &&
           rank_tuple_valid(arg.pair_tuple.pairs, arg.pair_tuple.n);
  case POLY_OP_PAD:
    if (arg.kind == POLY_ARG_NONE)
      return n_src == 3 && src && shape_value_rank_valid(src[1]) && shape_value_rank_valid(src[2]);
    return arg.kind == POLY_ARG_PAIR_TUPLE &&
           rank_tuple_valid(arg.pair_tuple.pairs, arg.pair_tuple.n);
  case POLY_OP_REDUCE:
    if (arg.kind == POLY_ARG_OPS) return n_src >= 1;
    return arg.kind == POLY_ARG_REDUCE_AXIS && n_src == 1 &&
           rank_tuple_valid(arg.reduce_axis.axes, arg.reduce_axis.n);
  case POLY_OP_REDUCE_AXIS:
    return arg.kind == POLY_ARG_REDUCE_AXIS &&
           rank_tuple_valid(arg.reduce_axis.axes, arg.reduce_axis.n);
  case POLY_OP_ASSIGN:
    return arg.kind != POLY_ARG_INT_TUPLE || rank_tuple_valid(arg.int_tuple.vals, arg.int_tuple.n);
  case POLY_OP_PARAM:
    if (arg.kind != POLY_ARG_PARAM) return true;
    if (!arg.param || arg.param->n_devices < 0) return false;
    if (arg.param->device_is_tuple)
      return !arg.param->device &&
             string_tuple_valid(arg.param->devices, arg.param->n_devices);
    return arg.param->n_devices == 0 && !arg.param->devices;
  default:
    return true;
  }
}

static bool poly_arg_canonicalize_bigint(PolyArg *arg) {
  if (!arg || arg->kind != POLY_ARG_BIGINT) return true;
  if (arg->bigint.n_limbs > 0 && !arg->bigint.limbs) return false;
  while (arg->bigint.n_limbs > 0 &&
         arg->bigint.limbs[arg->bigint.n_limbs - 1] == 0)
    arg->bigint.n_limbs--;
  if (arg->bigint.n_limbs == 0) {
    *arg = poly_arg_int(0);
    return true;
  }
  arg->bigint.sign = arg->bigint.sign < 0 ? -1 : 1;
  if (arg->bigint.n_limbs > 2) return true;
  uint64_t magnitude = arg->bigint.limbs[0];
  if (arg->bigint.n_limbs == 2)
    magnitude |= (uint64_t)arg->bigint.limbs[1] << 32;
  if (arg->bigint.sign > 0 && magnitude <= INT64_MAX) {
    *arg = poly_arg_int((int64_t)magnitude);
  } else if (arg->bigint.sign < 0 && magnitude <= (UINT64_C(1) << 63)) {
    *arg = poly_arg_int(
        magnitude == (UINT64_C(1) << 63) ? INT64_MIN : -(int64_t)magnitude
    );
  }
  return true;
}

static void poly_arg_copy_to_arena(PolyArena *arena, PolyArg *dst) {
  if (!arena || !dst) return;
  if (dst->kind == POLY_ARG_INT_TUPLE && dst->int_tuple.n > 0) {
    int64_t *vals =
        poly_arena_alloc(arena, dst->int_tuple.n * sizeof(int64_t), _Alignof(int64_t));
    memcpy(vals, dst->int_tuple.vals, dst->int_tuple.n * sizeof(int64_t));
    dst->int_tuple.vals = vals;
  } else if (dst->kind == POLY_ARG_BIGINT && dst->bigint.n_limbs > 0 &&
             dst->bigint.limbs) {
    uint32_t *limbs = poly_arena_alloc(
        arena, (size_t)dst->bigint.n_limbs * sizeof(uint32_t), _Alignof(uint32_t)
    );
    memcpy(
        limbs, dst->bigint.limbs, (size_t)dst->bigint.n_limbs * sizeof(uint32_t)
    );
    dst->bigint.limbs = limbs;
  } else if (dst->kind == POLY_ARG_PAIR_TUPLE && dst->pair_tuple.n > 0) {
    int64_t(*pairs)[2] =
        poly_arena_alloc(arena, dst->pair_tuple.n * 2 * sizeof(int64_t), _Alignof(int64_t));
    memcpy(pairs, dst->pair_tuple.pairs, dst->pair_tuple.n * 2 * sizeof(int64_t));
    dst->pair_tuple.pairs = pairs;
  } else if (dst->kind == POLY_ARG_REDUCE_AXIS && dst->reduce_axis.n > 0) {
    int64_t *axes =
        poly_arena_alloc(arena, dst->reduce_axis.n * sizeof(int64_t), _Alignof(int64_t));
    memcpy(axes, dst->reduce_axis.axes, dst->reduce_axis.n * sizeof(int64_t));
    dst->reduce_axis.axes = axes;
  } else if (dst->kind == POLY_ARG_RANGE && dst->range.n_extra > 0) {
    int64_t *extra = poly_arena_alloc(
        arena, (size_t)dst->range.n_extra * sizeof(int64_t), _Alignof(int64_t)
    );
    memcpy(extra, dst->range.extra, (size_t)dst->range.n_extra * sizeof(int64_t));
    dst->range.extra = extra;
  } else if (dst->kind == POLY_ARG_STRING && dst->str) {
    size_t len = strlen(dst->str);
    char *s = poly_arena_alloc(arena, len + 1, 1);
    memcpy(s, dst->str, len + 1);
    dst->str = s;
  } else if (dst->kind == POLY_ARG_STRING_TUPLE && dst->string_tuple.n > 0) {
    const char **vals = poly_arena_alloc(
        arena, (size_t)dst->string_tuple.n * sizeof(*vals), _Alignof(const char *)
    );
    for (int i = 0; i < dst->string_tuple.n; i++) {
      size_t len = strlen(dst->string_tuple.vals[i]);
      char *value = poly_arena_alloc(arena, len + 1, 1);
      memcpy(value, dst->string_tuple.vals[i], len + 1);
      vals[i] = value;
    }
    dst->string_tuple.vals = vals;
  } else if (dst->kind == POLY_ARG_DEFINE_VAR && dst->define_var.name) {
    size_t len = strlen(dst->define_var.name);
    char *s = poly_arena_alloc(arena, len + 1, 1);
    memcpy(s, dst->define_var.name, len + 1);
    dst->define_var.name = s;
  } else if (dst->kind == POLY_ARG_BUFFERIZE_OPTS && dst->bufferize_opts.device) {
    size_t len = strlen(dst->bufferize_opts.device);
    char *s = poly_arena_alloc(arena, len + 1, 1);
    memcpy(s, dst->bufferize_opts.device, len + 1);
    dst->bufferize_opts.device = s;
  } else if (dst->kind == POLY_ARG_TENSOR_CORE && dst->tensor_core.name) {
    size_t len = strlen(dst->tensor_core.name);
    char *s = poly_arena_alloc(arena, len + 1, 1);
    memcpy(s, dst->tensor_core.name, len + 1);
    dst->tensor_core.name = s;
  } else if (dst->kind == POLY_ARG_BYTES && dst->bytes.n > 0 && dst->bytes.data) {
    uint8_t *data = poly_arena_alloc(arena, (size_t)dst->bytes.n, 1);
    memcpy(data, dst->bytes.data, (size_t)dst->bytes.n);
    dst->bytes.data = data;
  } else if (dst->kind == POLY_ARG_PARAM && dst->param) {
    PolyParamArg *param = poly_arena_alloc(arena, sizeof(*param), _Alignof(PolyParamArg));
    *param = *dst->param;
    if (param->name) {
      size_t len = strlen(param->name);
      char *name = poly_arena_alloc(arena, len + 1, 1);
      memcpy(name, param->name, len + 1);
      param->name = name;
    }
    if (param->device) {
      size_t len = strlen(param->device);
      char *device = poly_arena_alloc(arena, len + 1, 1);
      memcpy(device, param->device, len + 1);
      param->device = device;
    }
    if (param->n_devices > 0) {
      const char **devices = poly_arena_alloc(
          arena, (size_t)param->n_devices * sizeof(*devices), _Alignof(const char *)
      );
      for (int i = 0; i < param->n_devices; i++) {
        size_t len = strlen(param->devices[i]);
        char *device = poly_arena_alloc(arena, len + 1, 1);
        memcpy(device, param->devices[i], len + 1);
        devices[i] = device;
      }
      param->devices = devices;
    }
    dst->param = param;
  }
}

static PolyUOp *poly_uop_internal(
    PolyCtx *ctx,
    PolyOps op,
    PolyDType dtype,
    PolyUOp **src,
    int n_src,
    PolyArg arg,
    int32_t tag,
    PolyArg tag_arg
) {
  if (!ctx || n_src < 0 || n_src > UINT16_MAX || (n_src > 0 && !src)) return NULL;
  /* Python ints have one value identity regardless of construction history.
   * Canonicalize the C carrier before hashing so signed-64 values cannot
   * acquire a second BIGINT CSE identity. */
  if (!poly_arg_canonicalize_bigint(&arg) ||
      !poly_arg_canonicalize_bigint(&tag_arg))
    return NULL;
  /* Canonicalize legacy RANGE(axis_id as INT) at the UOp boundary. tinygrad
   * stores RANGE args as (axis_id, AxisType, ...); keeping the legacy form out
   * of CSE lets PolyArg equality stay exact by kind. */
  if (op == POLY_OP_RANGE && arg.kind == POLY_ARG_INT) {
    arg = poly_arg_range(arg.i, POLY_AXIS_LOOP);
  }

  if (!uop_rank_arg_valid(op, src, n_src, arg)) return NULL;

  /* Build a CSE key on the stack */
  CseKey key = {op, dtype, src, (uint16_t)n_src, arg, tag, tag_arg};
  uint32_t h = cse_hash(&key);

  /* DEFINE_LOCAL represents mutable accumulators — each REDUCE needs its
   * own unique accumulator, so skip CSE dedup for this op.  Without this,
   * two reductions in a multi-store kernel that happen to share identity
   * value and outer-range deps would get merged into a single acc variable,
   * corrupting both computations. */
  bool cse_lookup = op != POLY_OP_DEFINE_LOCAL;
  if (op == POLY_OP_INS)
    cse_lookup = n_src == 0 && tag == 0 && tag_arg.kind == POLY_ARG_NONE;
  if (cse_lookup) {
    PolyUOp *existing = poly_map_get(ctx->cse, h, &key, cse_eq);
    if (existing) return existing;
  }

  /* Allocate new UOp in arena */
  PolyUOp *u = poly_arena_alloc(ctx->arena, sizeof(PolyUOp), _Alignof(PolyUOp));
  u->op = op;
  u->dtype = dtype;
  u->n_src = (uint16_t)n_src;
  u->arg = arg;
  u->tag = tag;
  u->tag_arg = tag_arg;
  u->hash = h;
  u->minmax_cached = false;
  u->minmax_vmin = 0;
  u->minmax_vmax = 0;
  u->ranges_cache = NULL;
  u->ended_ranges_cache = NULL;

  /* Copy src pointers into arena */
  if (n_src > 0) {
    u->src = poly_arena_alloc(ctx->arena, n_src * sizeof(PolyUOp *), _Alignof(PolyUOp *));
    memcpy(u->src, src, n_src * sizeof(PolyUOp *));
  } else {
    u->src = NULL;
  }

  /* Copy arg/tag data that needs arena allocation */
  poly_arg_copy_to_arena(ctx->arena, &u->arg);
  poly_arg_copy_to_arena(ctx->arena, &u->tag_arg);

  /* Also store the CSE key in the arena so it persists for hash map lookups */
  CseKey *stored_key = poly_arena_alloc(ctx->arena, sizeof(CseKey), _Alignof(CseKey));
  *stored_key = (CseKey){op, dtype, u->src, (uint16_t)n_src, u->arg, tag, u->tag_arg};

  poly_map_set(ctx->cse, h, stored_key, u, cse_eq);
  return u;
}

PolyUOp *poly_uop(
    PolyCtx *ctx,
    PolyOps op,
    PolyDType dtype,
    PolyUOp **src,
    int n_src,
    PolyArg arg
) {
  return poly_uop_internal(ctx, op, dtype, src, n_src, arg, 0, poly_arg_none());
}

PolyUOp *poly_uop_tagged(
    PolyCtx *ctx,
    PolyOps op,
    PolyDType dtype,
    PolyUOp **src,
    int n_src,
    PolyArg arg,
    int32_t tag
) {
  return poly_uop_internal(ctx, op, dtype, src, n_src, arg, tag, poly_arg_none());
}

PolyUOp *poly_uop_tagged_arg(
    PolyCtx *ctx,
    PolyOps op,
    PolyDType dtype,
    PolyUOp **src,
    int n_src,
    PolyArg arg,
    int32_t tag,
    PolyArg tag_arg
) {
  return poly_uop_internal(ctx, op, dtype, src, n_src, arg, tag, tag_arg);
}

PolyUOp *poly_uop0(PolyCtx *ctx, PolyOps op, PolyDType dtype, PolyArg arg) {
  return poly_uop(ctx, op, dtype, NULL, 0, arg);
}

PolyUOp *poly_uop1(PolyCtx *ctx, PolyOps op, PolyDType dtype, PolyUOp *s0, PolyArg arg) {
  PolyUOp *src[] = {s0};
  return poly_uop(ctx, op, dtype, src, 1, arg);
}

PolyUOp *poly_uop2(
    PolyCtx *ctx,
    PolyOps op,
    PolyDType dtype,
    PolyUOp *s0,
    PolyUOp *s1,
    PolyArg arg
) {
  PolyUOp *src[] = {s0, s1};
  return poly_uop(ctx, op, dtype, src, 2, arg);
}

PolyUOp *poly_uop3(
    PolyCtx *ctx,
    PolyOps op,
    PolyDType dtype,
    PolyUOp *s0,
    PolyUOp *s1,
    PolyUOp *s2,
    PolyArg arg
) {
  PolyUOp *src[] = {s0, s1, s2};
  return poly_uop(ctx, op, dtype, src, 3, arg);
}

/* Toposort (iterative DFS, mirrors tinygrad's toposort) */

typedef enum {
  POLY_TOPO_RESULT_ARENA,
  POLY_TOPO_RESULT_OWNED,
  POLY_TOPO_RESULT_SCRATCH,
} PolyTopoResultStorage;

static PolyUOp **toposort_result_alloc(PolyCtx *ctx, int cap, PolyTopoResultStorage storage) {
  size_t nbytes = (size_t)cap * sizeof(PolyUOp *);
  switch (storage) {
  case POLY_TOPO_RESULT_OWNED:
    return malloc(nbytes);
  case POLY_TOPO_RESULT_SCRATCH:
    return ctx ? poly_ctx_scratch_alloc(ctx, nbytes, _Alignof(PolyUOp *)) : NULL;
  case POLY_TOPO_RESULT_ARENA:
  default:
    return ctx ? poly_arena_alloc(ctx->arena, nbytes, _Alignof(PolyUOp *)) : NULL;
  }
}

/* Shared iterative DFS worker. Either `gate_simple` or `gate_user` may be
 * non-NULL (never both). The gate signature difference is bridged here so
 * callers can use closure-style gating without another whole copy. */
static PolyUOp **toposort_worker(
    PolyCtx *ctx,
    PolyUOp *root,
    int *n_out,
    bool (*gate_simple)(PolyUOp *),
    bool (*gate_user)(PolyUOp *, void *),
    void *user_data,
    bool enter_calls,
    PolyTopoResultStorage storage
) {
  if (n_out) *n_out = 0;
  /* Owned temporary toposorts only traverse immutable UOps and need no arena.
   * This permits tinygrad-style topological symbolic evaluation in helpers
   * whose complete input is already the root UOp. */
  if (!root || !n_out || (!ctx && storage != POLY_TOPO_RESULT_OWNED)) return NULL;
  int cap = 256;
  int n = 0;
  PolyUOp **result = toposort_result_alloc(ctx, cap, storage);
  if (!result) return NULL;

  /* Visited set — pointer identity map */
  PolyMap *visited = poly_map_new(256);
  if (!visited) {
    if (storage == POLY_TOPO_RESULT_OWNED) free(result);
    return NULL;
  }

  /* DFS stack: pairs of (UOp*, state) where state 0=first visit, 1=children pushed */
  int stack_cap = 256;
  int stack_top = 0;
  PolyUOp **stack = malloc(stack_cap * sizeof(PolyUOp *));
  int *state = malloc(stack_cap * sizeof(int));
  if (!stack || !state) {
    free(stack);
    free(state);
    poly_map_destroy(visited);
    if (storage == POLY_TOPO_RESULT_OWNED) free(result);
    return NULL;
  }

  stack[stack_top] = root;
  state[stack_top] = 0;
  stack_top++;

  while (stack_top > 0) {
    PolyUOp *u = stack[stack_top - 1];
    int s = state[stack_top - 1];

    uint32_t vh = poly_ptr_hash(u);
    if (s == 0 && poly_map_get(visited, vh, u, poly_ptr_eq) != NULL) {
      stack_top--;
      continue;
    }

    if (s == 0) {
      /* Gate check: if gate returns false, skip this subtree entirely */
      bool pass = true;
      if (gate_simple && !gate_simple(u)) pass = false;
      if (gate_user && !gate_user(u, user_data)) pass = false;
      if (!pass) {
        stack_top--;
        continue;
      }
      /* First visit: push children in reverse order */
      state[stack_top - 1] = 1;
      /* CALL/FUNCTION src[0] is the opaque callee body. External arguments
       * remain ordinary graph inputs when enter_calls=false. */
      bool opaque_body = u->op == POLY_OP_CALL || u->op == POLY_OP_FUNCTION;
      int start = (!enter_calls && opaque_body && u->n_src > 0) ? 1 : 0;
      for (int i = u->n_src - 1; i >= start; i--) {
        uint32_t ch = poly_ptr_hash(u->src[i]);
        if (poly_map_get(visited, ch, u->src[i], poly_ptr_eq) != NULL) continue;
        if (stack_top >= stack_cap) {
          stack_cap *= 2;
          PolyUOp **new_stack = realloc(stack, (size_t)stack_cap * sizeof(PolyUOp *));
          int *new_state = realloc(state, (size_t)stack_cap * sizeof(int));
          if (!new_stack || !new_state) {
            free(new_stack ? new_stack : stack);
            free(new_state ? new_state : state);
            poly_map_destroy(visited);
            if (storage == POLY_TOPO_RESULT_OWNED) free(result);
            *n_out = 0;
            return NULL;
          }
          stack = new_stack;
          state = new_state;
        }
        stack[stack_top] = u->src[i];
        state[stack_top] = 0;
        stack_top++;
      }
    } else {
      /* Post-order: all children done, emit this node */
      stack_top--;
      if (poly_map_get(visited, vh, u, poly_ptr_eq) != NULL) continue;
      /* Use a non-NULL sentinel as value */
      poly_map_set(visited, vh, u, (void *)(uintptr_t)1, poly_ptr_eq);

      if (n >= cap) {
        int new_cap = cap * 2;
        PolyUOp **new_result =
            (storage == POLY_TOPO_RESULT_OWNED) ?
                realloc(result, (size_t)new_cap * sizeof(PolyUOp *)) :
                toposort_result_alloc(ctx, new_cap, storage);
        if (!new_result) {
          free(stack);
          free(state);
          poly_map_destroy(visited);
          if (storage == POLY_TOPO_RESULT_OWNED) free(result);
          *n_out = 0;
          return NULL;
        }
        if (storage != POLY_TOPO_RESULT_OWNED)
          memcpy(new_result, result, (size_t)n * sizeof(PolyUOp *));
        result = new_result;
        cap = new_cap;
      }
      result[n++] = u;
    }
  }

  free(stack);
  free(state);
  poly_map_destroy(visited);
  *n_out = n;
  return result;
}

PolyUOp **poly_toposort_ex(
    PolyCtx *ctx,
    PolyUOp *root,
    int *n_out,
    bool (*gate)(PolyUOp *),
    bool enter_calls
) {
  return toposort_worker(ctx, root, n_out, gate, NULL, NULL, enter_calls, POLY_TOPO_RESULT_ARENA);
}

PolyUOp **poly_toposort_ex_user(
    PolyCtx *ctx,
    PolyUOp *root,
    int *n_out,
    bool (*gate)(PolyUOp *, void *),
    void *user_data,
    bool enter_calls
) {
  return toposort_worker(ctx, root, n_out, NULL, gate, user_data, enter_calls, POLY_TOPO_RESULT_ARENA);
}

PolyUOp **poly_toposort(PolyCtx *ctx, PolyUOp *root, int *n_out) {
  return toposort_worker(ctx, root, n_out, NULL, NULL, NULL, true, POLY_TOPO_RESULT_ARENA);
}

PolyUOp **poly_toposort_ex_alloc(
    PolyCtx *ctx,
    PolyUOp *root,
    int *n_out,
    bool (*gate)(PolyUOp *),
    bool enter_calls
) {
  return toposort_worker(ctx, root, n_out, gate, NULL, NULL, enter_calls, POLY_TOPO_RESULT_OWNED);
}

PolyUOp **poly_toposort_ex_user_alloc(
    PolyCtx *ctx,
    PolyUOp *root,
    int *n_out,
    bool (*gate)(PolyUOp *, void *),
    void *user_data,
    bool enter_calls
) {
  return toposort_worker(ctx, root, n_out, NULL, gate, user_data, enter_calls, POLY_TOPO_RESULT_OWNED);
}

PolyUOp **poly_toposort_alloc(PolyCtx *ctx, PolyUOp *root, int *n_out) {
  return toposort_worker(ctx, root, n_out, NULL, NULL, NULL, true, POLY_TOPO_RESULT_OWNED);
}

PolyUOp **poly_toposort_scratch(PolyCtx *ctx, PolyUOp *root, int *n_out) {
  return toposort_worker(ctx, root, n_out, NULL, NULL, NULL, true, POLY_TOPO_RESULT_SCRATCH);
}

PolyUOp **poly_toposort_ex_user_scratch(
    PolyCtx *ctx,
    PolyUOp *root,
    int *n_out,
    bool (*gate)(PolyUOp *, void *),
    void *user_data,
    bool enter_calls
) {
  return toposort_worker(
      ctx, root, n_out, NULL, gate, user_data, enter_calls, POLY_TOPO_RESULT_SCRATCH
  );
}

void poly_toposort_free(PolyUOp **topo) {
  free(topo);
}

/* Local duplicate of codegen.c's range_start_for_op table so uop.c can
 * compute ended_ranges without pulling codegen.h (which depends on uop).
 * The two definitions must stay in sync — both mirror tinygrad's
 * `range_start` dict (uop/ops.py:29). */
static int uop_range_start_for_op(PolyOps op) {
  switch (op) {
  case POLY_OP_STAGE:
    return 1;
  case POLY_OP_REDUCE:
    return 1;
  case POLY_OP_STORE:
    return 2;
  case POLY_OP_WMMA:
    return 3;
  case POLY_OP_END:
    return 1;
  case POLY_OP_CALL:
    return 1;
  case POLY_OP_COPY:
    return 2;
  case POLY_OP_BUFFER_VIEW:
    return 1;
  default:
    return -1;
  }
}

/* Range helpers *
 * Ports tinygrad's `no_range` (codegen/simplify.py:75) and the
 * `_ranges`/`ranges`/`ended_ranges` triple (uop/ops.py:351-378).
 *
 * `poly_no_range(u)` = not any RANGE in u's backward slice (simple, exact).
 *
 * `poly_uop_ranges(u)` = the set of *active* RANGE UOps at u's position:
 *   ranges(u) = union(ranges(src) for src in u.src)
 *               minus ended_ranges(u)
 *               plus ({u} if u.op == RANGE else {})
 *
 * `ended_ranges(u)` follows tinygrad's semantics:
 *   - Ops with range_start_for_op(op) in {BUFFERIZE:1, REDUCE:1, STORE:2,
 *     WMMA:3, END:1, CALL:1, COPY:2, BUFFER_VIEW:1}: trailing srcs past
 *     range_start are the ended ranges.
 *   - AFTER: flatten(ended_ranges(src) for src in u.src[1:]) — recursive.
 *   - CONTRACT: the RANGE elements of ranges(src[0]) whose axis_id appears
 *     in u.arg.pair_tuple (first element of each pair).
 *   - otherwise: empty.
 *
 * When an ended entry is not itself a RANGE (e.g. a bound expression like
 * DEFINE_VAR, a symbolic bound, an AFTER value), tinygrad's `_ranges`
 * removes every range in that entry's `ranges` set from the result — not
 * just the entry itself. We mirror that exactly.
 *
 * Sets are represented as arena-allocated PolyUOp* arrays with linear-time
 * dedup. Typical range set sizes in realistic kernels are 0-8 elements, so
 * linear ops beat a hashmap. Computation is memoized per-UOp in a PolyMap
 * cache so a single pass-wide walk is O(N * avg_set_size). */

typedef struct PolyRangeSet {
  PolyUOp **items;
  int n;
  int cap;
} PolyRangeSet;

static PolyRangeSet *range_set_new(PolyCtx *ctx, int cap) {
  PolyArena *arena = poly_ctx_arena(ctx);
  PolyRangeSet *s = poly_arena_alloc(arena, sizeof(PolyRangeSet), _Alignof(PolyRangeSet));
  if (cap < 4) cap = 4;
  s->items = poly_arena_alloc(arena, (size_t)cap * sizeof(PolyUOp *), _Alignof(PolyUOp *));
  s->n = 0;
  s->cap = cap;
  return s;
}

static void range_set_grow(PolyCtx *ctx, PolyRangeSet *s, int need) {
  if (need <= s->cap) return;
  int new_cap = s->cap * 2;
  while (new_cap < need)
    new_cap *= 2;
  PolyUOp **new_items = poly_arena_alloc(
      poly_ctx_arena(ctx), (size_t)new_cap * sizeof(PolyUOp *), _Alignof(PolyUOp *)
  );
  memcpy(new_items, s->items, (size_t)s->n * sizeof(PolyUOp *));
  s->items = new_items;
  s->cap = new_cap;
}

static bool range_set_contains(const PolyRangeSet *s, PolyUOp *r) {
  for (int i = 0; i < s->n; i++)
    if (s->items[i] == r) return true;
  return false;
}

static void range_set_add(PolyCtx *ctx, PolyRangeSet *s, PolyUOp *r) {
  if (range_set_contains(s, r)) return;
  range_set_grow(ctx, s, s->n + 1);
  s->items[s->n++] = r;
}

static void range_set_remove(PolyRangeSet *s, PolyUOp *r) {
  for (int i = 0; i < s->n; i++) {
    if (s->items[i] == r) {
      s->items[i] = s->items[s->n - 1];
      s->n--;
      return;
    }
  }
}

static void range_set_union(PolyCtx *ctx, PolyRangeSet *dst, const PolyRangeSet *src) {
  for (int i = 0; i < src->n; i++)
    range_set_add(ctx, dst, src->items[i]);
}

static void range_set_subtract(PolyRangeSet *dst, const PolyRangeSet *src) {
  for (int i = 0; i < src->n; i++)
    range_set_remove(dst, src->items[i]);
}

static const PolyRangeSet *ranges_memo_get(PolyMap *memo, PolyUOp *u) {
  if (!u) return NULL;
  if (u->ranges_cache) return (const PolyRangeSet *)u->ranges_cache;
  return memo ? (const PolyRangeSet *)poly_map_get(memo, poly_ptr_hash(u), u, poly_ptr_eq) : NULL;
}

static const PolyRangeSet *ended_memo_get(PolyMap *memo, PolyUOp *u) {
  if (!u) return NULL;
  if (u->ended_ranges_cache) return (const PolyRangeSet *)u->ended_ranges_cache;
  return memo ? (const PolyRangeSet *)poly_map_get(memo, poly_ptr_hash(u), u, poly_ptr_eq) : NULL;
}

static void ranges_memo_set(PolyMap *memo, PolyUOp *u, PolyRangeSet *ranges) {
  if (!u || !ranges) return;
  u->ranges_cache = ranges;
  if (memo) poly_map_set(memo, poly_ptr_hash(u), u, ranges, poly_ptr_eq);
}

static void ended_memo_set(PolyMap *memo, PolyUOp *u, PolyRangeSet *ranges) {
  if (!u || !ranges) return;
  u->ended_ranges_cache = ranges;
  if (memo) poly_map_set(memo, poly_ptr_hash(u), u, ranges, poly_ptr_eq);
}

/* ended_ranges per tinygrad ops.py:351-358. Writes into `out` (caller-owned).
 * For entries that are themselves RANGE nodes we just add them; for non-RANGE
 * entries we union in their full ranges set so the caller's subtract does the
 * right thing regardless of entry shape.
 *
 * This helper assumes `ranges_memo` and `ended_memo` already contain every
 * source node, which is true because compute_ranges() processes topological
 * order. */
static bool compute_ended_ranges_node(
    PolyCtx *ctx,
    PolyUOp *u,
    PolyRangeSet *out,
    PolyMap *ranges_memo,
    PolyMap *ended_memo
) {
  int rs = uop_range_start_for_op(u->op);
  if (rs >= 0) {
    if (rs > u->n_src) rs = u->n_src;
    for (int i = rs; i < u->n_src; i++) {
      PolyUOp *er = u->src[i];
      if (er->op == POLY_OP_RANGE) {
        range_set_add(ctx, out, er);
      } else {
        const PolyRangeSet *er_r = ranges_memo_get(ranges_memo, er);
        if (!er_r) return false;
        range_set_union(ctx, out, er_r);
      }
    }
    return true;
  }
  if (u->op == POLY_OP_AFTER) {
    /* flatten(ended_ranges(x) for x in src[1:]) */
    for (int i = 1; i < u->n_src; i++) {
      const PolyRangeSet *src_ended = ended_memo_get(ended_memo, u->src[i]);
      if (!src_ended) return false;
      range_set_union(ctx, out, src_ended);
    }
    return true;
  }
  if (u->op == POLY_OP_CONTRACT && u->n_src >= 1 && u->arg.kind == POLY_ARG_PAIR_TUPLE) {
    const PolyRangeSet *s0 = ranges_memo_get(ranges_memo, u->src[0]);
    if (!s0) return false;
    int n_pairs = u->arg.pair_tuple.n;
    for (int i = 0; i < s0->n; i++) {
      PolyUOp *r = s0->items[i];
      if (r->op != POLY_OP_RANGE) continue;
      int64_t axis_id = poly_range_axis_id(r->arg);
      for (int j = 0; j < n_pairs; j++) {
        if (u->arg.pair_tuple.pairs[j][0] == axis_id) {
          range_set_add(ctx, out, r);
          break;
        }
      }
    }
    return true;
  }
  /* No ended ranges for other ops. */
  return true;
}

typedef struct {
  PolyMap *ranges;
  PolyMap *ended;
} RangeComputeGateCtx;

static bool range_compute_gate(PolyUOp *u, void *user_data) {
  RangeComputeGateCtx *g = (RangeComputeGateCtx *)user_data;
  if (!g || !u) return false;
  return !ranges_memo_get(g->ranges, u) || !ended_memo_get(g->ended, u);
}

static PolyRangeSet *compute_ranges_with_ended(
    PolyCtx *ctx,
    PolyUOp *u,
    PolyMap *ranges_memo,
    PolyMap *ended_memo
) {
  PolyRangeSet *cached = (PolyRangeSet *)ranges_memo_get(ranges_memo, u);
  if (cached) return cached;

  bool own_ended = false;
  if (!ended_memo) {
    ended_memo = poly_map_new(64);
    if (!ended_memo) return NULL;
    own_ended = true;
  }

  int n_topo = 0;
  RangeComputeGateCtx gate = {ranges_memo, ended_memo};
  PolyScratchMark scratch = poly_ctx_scratch_mark(ctx);
  PolyUOp **topo = poly_toposort_ex_user_scratch(
      ctx, u, &n_topo, range_compute_gate, &gate, true
  );
  if (!topo) {
    poly_ctx_scratch_rewind(ctx, scratch);
    if (own_ended) poly_map_destroy(ended_memo);
    return NULL;
  }

  bool ok = true;
  for (int ti = 0; ok && ti < n_topo; ti++) {
    PolyUOp *cur = topo[ti];

    PolyRangeSet *ended = (PolyRangeSet *)ended_memo_get(ended_memo, cur);
    if (!ended) {
      ended = range_set_new(ctx, 4);
      ok = compute_ended_ranges_node(ctx, cur, ended, ranges_memo, ended_memo);
      if (!ok) break;
      ended_memo_set(ended_memo, cur, ended);
    }

    if (ranges_memo_get(ranges_memo, cur)) continue;

    PolyRangeSet *ret = range_set_new(ctx, 4);
    for (int i = 0; i < cur->n_src; i++) {
      const PolyRangeSet *src_ranges = ranges_memo_get(ranges_memo, cur->src[i]);
      if (!src_ranges) {
        ok = false;
        break;
      }
      range_set_union(ctx, ret, src_ranges);
    }
    if (!ok) break;

    range_set_subtract(ret, ended);
    if (cur->op == POLY_OP_RANGE) range_set_add(ctx, ret, cur);
    ranges_memo_set(ranges_memo, cur, ret);
  }

  cached = ok ? (PolyRangeSet *)ranges_memo_get(ranges_memo, u) : NULL;
  poly_ctx_scratch_rewind(ctx, scratch);
  if (own_ended) poly_map_destroy(ended_memo);
  return cached;
}

/* PolyUOpCache: per-pass query maps for minmax + ranges *
 * Owns PolyMaps keyed by PolyUOp*. The struct is opaque in the public
 * header; callers get it via poly_uop_cache_new and pass it to any `_ex`
 * query. Range-set values are UOp-lifetime cached in the ctx arena. Minmax
 * values live inline on the UOp. Destroying the cache only tears down the map
 * wrappers. */

struct PolyUOpCache {
  PolyMap *ranges; /* PolyUOp* -> PolyRangeSet* */
  PolyMap *ended; /* PolyUOp* -> PolyRangeSet* for ended_ranges */
};

PolyUOpCache *poly_uop_cache_new(void) {
  PolyUOpCache *c = malloc(sizeof(PolyUOpCache));
  if (!c) return NULL;
  c->ranges = poly_map_new(64);
  c->ended = poly_map_new(64);
  if (!c->ranges || !c->ended) {
    if (c->ranges) poly_map_destroy(c->ranges);
    if (c->ended) poly_map_destroy(c->ended);
    free(c);
    return NULL;
  }
  return c;
}

void poly_uop_cache_destroy(PolyUOpCache *c) {
  if (!c) return;
  if (c->ranges) poly_map_destroy(c->ranges);
  if (c->ended) poly_map_destroy(c->ended);
  free(c);
}

/* Walks RESHAPE/MULTI wrappers and returns the terminal buffer-identity UOp
 * (BUFFER / BUFFER_VIEW / PARAM), or NULL if `u` has no buffer identity. */
const PolyUOp *poly_uop_get_buffer_identity(const PolyUOp *u) {
  while (u) {
    if (u->op == POLY_OP_RESHAPE || u->op == POLY_OP_MULTI) {
      if (u->n_src < 1) return NULL;
      u = u->src[0];
      continue;
    }
    if (u->op == POLY_OP_BUFFER || u->op == POLY_OP_BUFFER_VIEW || u->op == POLY_OP_PARAM) {
      return u;
    }
    return NULL;
  }
  return NULL;
}

/* Port of tinygrad's UOp.has_buffer_identity.
 * Unwraps RESHAPE/MULTI via src[0], then returns true iff the terminus is
 * BUFFER / BUFFER_VIEW / PARAM. tinygrad also handles GETTUPLE(TUPLE(...));
 * polygrad does not yet have those ops, so that part is intentionally
 * omitted. */
bool poly_uop_has_buffer_identity(const PolyUOp *u) {
  return poly_uop_get_buffer_identity(u) != NULL;
}

bool poly_uop_reachable(PolyCtx *ctx, PolyUOp *root, PolyUOp *target) {
  if (!ctx || !root || !target) return false;
  if (root == target) return true;
  int n = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, root, &n);
  if (!topo) return false;
  /* Mirrors tinygrad's `t.uop in loss.uop.toposort()` frontend test. */
  bool found = false;
  for (int i = 0; i < n; i++) {
    if (topo[i] == target) {
      found = true;
      break;
    }
  }
  poly_toposort_free(topo);
  return found;
}

bool poly_no_range(PolyCtx *ctx, PolyUOp *u) {
  return poly_no_range_ex(ctx, u, NULL);
}

bool poly_no_range_ex(PolyCtx *ctx, PolyUOp *u, PolyUOpCache *cache) {
  (void)cache; /* no_range is a single backward-slice walk; caching wouldn't
                * help because we short-circuit on the first RANGE hit. */
  if (!u) return true;
  if (u->op == POLY_OP_RANGE) return false;
  int n = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, u, &n);
  if (!topo) return false;
  bool no_range = true;
  for (int i = 0; i < n; i++) {
    if (topo[i]->op == POLY_OP_RANGE) {
      no_range = false;
      break;
    }
  }
  poly_toposort_free(topo);
  return no_range;
}

bool poly_uop_in_ranges(PolyCtx *ctx, PolyUOp *u, PolyUOp *r) {
  return poly_uop_in_ranges_ex(ctx, u, r, NULL);
}

bool poly_uop_in_ranges_ex(PolyCtx *ctx, PolyUOp *u, PolyUOp *r, PolyUOpCache *cache) {
  if (!ctx || !u || !r || r->op != POLY_OP_RANGE) return false;
  if (!cache && u->ranges_cache) return range_set_contains((const PolyRangeSet *)u->ranges_cache, r);
  PolyMap *memo = cache ? cache->ranges : poly_map_new(64);
  PolyMap *ended = cache ? cache->ended : poly_map_new(64);
  if (!memo) return false;
  if (!ended) {
    if (!cache) poly_map_destroy(memo);
    return false;
  }
  const PolyRangeSet *s = compute_ranges_with_ended(ctx, u, memo, ended);
  bool found = s && range_set_contains(s, r);
  if (!cache) {
    poly_map_destroy(memo);
    poly_map_destroy(ended);
  }
  return found;
}

int poly_uop_ranges(PolyCtx *ctx, PolyUOp *u, PolyUOp **out, int max_out) {
  return poly_uop_ranges_ex(ctx, u, out, max_out, NULL);
}

int poly_uop_ranges_ex(PolyCtx *ctx, PolyUOp *u, PolyUOp **out, int max_out, PolyUOpCache *cache) {
  if (!ctx || !u || !out || max_out <= 0) return 0;
  if (!cache && u->ranges_cache) {
    const PolyRangeSet *s = (const PolyRangeSet *)u->ranges_cache;
    int n_out = s->n < max_out ? s->n : max_out;
    memcpy(out, s->items, (size_t)n_out * sizeof(PolyUOp *));
    return n_out;
  }
  PolyMap *memo = cache ? cache->ranges : poly_map_new(64);
  PolyMap *ended = cache ? cache->ended : poly_map_new(64);
  if (!memo) return 0;
  if (!ended) {
    if (!cache) poly_map_destroy(memo);
    return 0;
  }
  const PolyRangeSet *s = compute_ranges_with_ended(ctx, u, memo, ended);
  if (!s) {
    if (!cache) {
      poly_map_destroy(memo);
      poly_map_destroy(ended);
    }
    return 0;
  }
  int n_out = s->n < max_out ? s->n : max_out;
  memcpy(out, s->items, (size_t)n_out * sizeof(PolyUOp *));
  if (!cache) {
    poly_map_destroy(memo);
    poly_map_destroy(ended);
  }
  return n_out;
}

/* Pretty-print */

static void uop_print_one(PolyUOp *u, char *buf, int *pos, int cap) {
  int written = snprintf(buf + *pos, cap - *pos, "UOp(%s", poly_op_name(u->op));
  if (written > 0) *pos += written;

  /* dtype */
  if (!poly_dtype_eq(u->dtype, POLY_VOID)) {
    written = snprintf(buf + *pos, cap - *pos, ", %s", poly_dtype_name(u->dtype));
    if (written > 0) *pos += written;
  }

  /* arg */
  switch (u->arg.kind) {
  case POLY_ARG_NONE:
    break;
  case POLY_ARG_INT:
    written = snprintf(buf + *pos, cap - *pos, ", %ld", (long)u->arg.i);
    if (written > 0) *pos += written;
    break;
  case POLY_ARG_BIGINT: {
    char *decimal = poly_arg_integer_to_decimal(u->arg);
    written = snprintf(buf + *pos, cap - *pos, ", %s", decimal ? decimal : "0");
    free(decimal);
    if (written > 0) *pos += written;
  } break;
  case POLY_ARG_PAIR_TUPLE:
    written = snprintf(buf + *pos, cap - *pos, ", (");
    if (written > 0) *pos += written;
    for (int i = 0; i < u->arg.pair_tuple.n; i++) {
      written = snprintf(
          buf + *pos, cap - *pos, "%s(%ld,%ld)", i ? "," : "",
          (long)u->arg.pair_tuple.pairs[i][0], (long)u->arg.pair_tuple.pairs[i][1]
      );
      if (written > 0) *pos += written;
    }
    written = snprintf(buf + *pos, cap - *pos, ")");
    if (written > 0) *pos += written;
    break;
  case POLY_ARG_REDUCE_AXIS:
    written = snprintf(
        buf + *pos, cap - *pos, ", (%s,(", poly_op_name(u->arg.reduce_axis.op)
    );
    if (written > 0) *pos += written;
    for (int i = 0; i < u->arg.reduce_axis.n; i++) {
      written = snprintf(
          buf + *pos, cap - *pos, "%s%ld", i ? "," : "",
          (long)u->arg.reduce_axis.axes[i]
      );
      if (written > 0) *pos += written;
    }
    written = snprintf(buf + *pos, cap - *pos, "))");
    if (written > 0) *pos += written;
    break;
  case POLY_ARG_FLOAT:
    /* Pinned UOp.argstr uses Python's round-trippable float repr
     * (uop/ops.py:166-171). Seventeen significant decimal digits preserve a
     * C double exactly for graph diagnostics and parity tooling. */
    written = snprintf(buf + *pos, cap - *pos, ", %.17g", u->arg.f);
    if (written > 0) *pos += written;
    break;
  case POLY_ARG_BOOL:
    written = snprintf(buf + *pos, cap - *pos, ", %s", u->arg.b ? "True" : "False");
    if (written > 0) *pos += written;
    break;
  case POLY_ARG_STRING:
    written = snprintf(buf + *pos, cap - *pos, ", \"%s\"", u->arg.str ? u->arg.str : "");
    if (written > 0) *pos += written;
    break;
  case POLY_ARG_STRING_TUPLE:
    written = snprintf(buf + *pos, cap - *pos, ", (");
    if (written > 0) *pos += written;
    for (int i = 0; i < u->arg.string_tuple.n; i++) {
      written =
          snprintf(buf + *pos, cap - *pos, "%s\"%s\"", i ? "," : "", u->arg.string_tuple.vals[i]);
      if (written > 0) *pos += written;
    }
    written = snprintf(buf + *pos, cap - *pos, ")");
    if (written > 0) *pos += written;
    break;
  case POLY_ARG_OPS:
    written = snprintf(buf + *pos, cap - *pos, ", %s", poly_op_name(u->arg.ops));
    if (written > 0) *pos += written;
    break;
  case POLY_ARG_INVALID:
    written = snprintf(buf + *pos, cap - *pos, ", Invalid");
    if (written > 0) *pos += written;
    break;
  case POLY_ARG_INT_TUPLE:
    written = snprintf(buf + *pos, cap - *pos, ", (");
    if (written > 0) *pos += written;
    for (int i = 0; i < u->arg.int_tuple.n; i++) {
      if (u->op == POLY_OP_FLIP)
        written = snprintf(
            buf + *pos, cap - *pos, "%s%s", i ? "," : "",
            u->arg.int_tuple.vals[i] ? "True" : "False"
        );
      else
        written =
            snprintf(buf + *pos, cap - *pos, "%s%ld", i ? "," : "", (long)u->arg.int_tuple.vals[i]);
      if (written > 0) *pos += written;
    }
    written = snprintf(buf + *pos, cap - *pos, ")");
    if (written > 0) *pos += written;
    break;
  case POLY_ARG_RANGE:
    written = snprintf(
        buf + *pos, cap - *pos, ", (%ld,%d", (long)u->arg.range.axis_id, (int)u->arg.range.axis_type
    );
    if (written > 0) *pos += written;
    for (int i = 0; i < u->arg.range.n_extra; i++) {
      written = snprintf(buf + *pos, cap - *pos, ",%ld", (long)u->arg.range.extra[i]);
      if (written > 0) *pos += written;
    }
    written = snprintf(buf + *pos, cap - *pos, ")");
    if (written > 0) *pos += written;
    break;
  case POLY_ARG_DEFINE_VAR:
    written = snprintf(
        buf + *pos, cap - *pos, ", (\"%s\",%ld,%ld)",
        u->arg.define_var.name ? u->arg.define_var.name : "?", (long)u->arg.define_var.min_val,
        (long)u->arg.define_var.max_val
    );
    if (written > 0) *pos += written;
    break;
  case POLY_ARG_BUFFERIZE_OPTS:
    written = snprintf(
        buf + *pos, cap - *pos, ", BufferizeOpts(device=%s,addrspace=%d,removable=%d)",
        u->arg.bufferize_opts.device ? u->arg.bufferize_opts.device : "None",
        (int)u->arg.bufferize_opts.addrspace, (int)u->arg.bufferize_opts.removable
    );
    if (written > 0) *pos += written;
    break;
  case POLY_ARG_TENSOR_CORE:
    written = snprintf(
        buf + *pos, cap - *pos, ", TensorCore(\"%s\",(%d,%d,%d),threads=%d)",
        u->arg.tensor_core.name ? u->arg.tensor_core.name : "?", u->arg.tensor_core.dims[0],
        u->arg.tensor_core.dims[1], u->arg.tensor_core.dims[2], u->arg.tensor_core.threads
    );
    if (written > 0) *pos += written;
    break;
  case POLY_ARG_PARAM:
    written = snprintf(
        buf + *pos, cap - *pos, ", ParamArg(slot=%ld,device=%s,addrspace=%d%s%s)",
        u->arg.param ? (long)u->arg.param->slot : -1L,
        u->arg.param && u->arg.param->device ? u->arg.param->device : "None",
        u->arg.param ? (int)u->arg.param->addrspace : 0,
        u->arg.param && u->arg.param->has_axis ? ",axis" : "",
        u->arg.param && u->arg.param->name ? ",name" : ""
    );
    if (written > 0) *pos += written;
    break;
  default:
    break;
  }

  /* n_src */
  if (u->n_src > 0) {
    written = snprintf(buf + *pos, cap - *pos, ", src=%d", u->n_src);
    if (written > 0) *pos += written;
  }

  written = snprintf(buf + *pos, cap - *pos, ")");
  if (written > 0) *pos += written;
}

char *poly_uop_str(PolyUOp *u) {
  int cap = 256;
  char *buf = malloc(cap);
  int pos = 0;
  uop_print_one(u, buf, &pos, cap);
  buf[pos] = '\0';
  return buf;
}

char *poly_graph_str(PolyUOp *root) {
  /* Simple: just print the root node for now */
  /* A full graph print would need toposort, but that needs a ctx */
  return poly_uop_str(root);
}

/* Recursive indented tree dump to a FILE*. Used by debug paths in
 * reduce_simplify.c, codegen.c, and any future pass that needs to inspect
 * IR shape. Caps recursion at max_depth to keep output bounded on cyclic
 * or very deep graphs (though hash-consed UOps cannot actually cycle). */
void poly_uop_dump_tree(FILE *fp, PolyUOp *u, int depth, int max_depth) {
  if (!u || !fp) return;
  for (int i = 0; i < depth; i++)
    fputc(' ', fp);
  fprintf(
      fp, "%s dt=%s n_src=%d", poly_op_name(u->op), u->dtype.name ? u->dtype.name : "?",
      (int)u->n_src
  );
  switch (u->arg.kind) {
  case POLY_ARG_INT:
    fprintf(fp, " i=%lld", (long long)u->arg.i);
    break;
  case POLY_ARG_BIGINT: {
    char *decimal = poly_arg_integer_to_decimal(u->arg);
    fprintf(fp, " i=%s", decimal ? decimal : "0");
    free(decimal);
  } break;
  case POLY_ARG_FLOAT:
    fprintf(fp, " f=%g", u->arg.f);
    break;
  case POLY_ARG_BOOL:
    fprintf(fp, " b=%d", (int)u->arg.b);
    break;
  case POLY_ARG_OPS:
    fprintf(fp, " op=%s", poly_op_name(u->arg.ops));
    break;
  case POLY_ARG_STRING_TUPLE:
    fprintf(fp, " strings=(");
    for (int i = 0; i < u->arg.string_tuple.n; i++)
      fprintf(fp, "%s\"%s\"", i ? "," : "", u->arg.string_tuple.vals[i]);
    fprintf(fp, ")");
    break;
  case POLY_ARG_PAIR_TUPLE:
    fprintf(fp, " pairs=(");
    for (int i = 0; i < u->arg.pair_tuple.n; i++)
      fprintf(
          fp, "%s(%lld,%lld)", i ? "," : "",
          (long long)u->arg.pair_tuple.pairs[i][0],
          (long long)u->arg.pair_tuple.pairs[i][1]
      );
    fprintf(fp, ")");
    break;
  case POLY_ARG_REDUCE_AXIS:
    fprintf(fp, " reduce=(%s,(", poly_op_name(u->arg.reduce_axis.op));
    for (int i = 0; i < u->arg.reduce_axis.n; i++)
      fprintf(
          fp, "%s%lld", i ? "," : "", (long long)u->arg.reduce_axis.axes[i]
      );
    fprintf(fp, "))");
    break;
  case POLY_ARG_RANGE:
    fprintf(fp, " axis=%lld", (long long)u->arg.range.axis_id);
    break;
  case POLY_ARG_DEFINE_VAR:
    fprintf(
        fp, " var=%s[%lld,%lld]", u->arg.define_var.name ? u->arg.define_var.name : "?",
        (long long)u->arg.define_var.min_val, (long long)u->arg.define_var.max_val
    );
    break;
  case POLY_ARG_BUFFERIZE_OPTS:
    fprintf(
        fp, " bufferize_opts=(device=%s,addrspace=%d,removable=%d)",
        u->arg.bufferize_opts.device ? u->arg.bufferize_opts.device : "None",
        (int)u->arg.bufferize_opts.addrspace, (int)u->arg.bufferize_opts.removable
    );
    break;
  case POLY_ARG_TENSOR_CORE:
    fprintf(
        fp, " tensor_core=%s[(%d,%d,%d),threads=%d]",
        u->arg.tensor_core.name ? u->arg.tensor_core.name : "?", u->arg.tensor_core.dims[0],
        u->arg.tensor_core.dims[1], u->arg.tensor_core.dims[2], u->arg.tensor_core.threads
    );
    break;
  case POLY_ARG_PARAM:
    fprintf(
        fp, " param=(slot=%lld,device=%s,addrspace=%d%s%s)",
        u->arg.param ? (long long)u->arg.param->slot : -1LL,
        u->arg.param && u->arg.param->device ? u->arg.param->device : "None",
        u->arg.param ? (int)u->arg.param->addrspace : 0,
        u->arg.param && u->arg.param->has_axis ? ",axis" : "",
        u->arg.param && u->arg.param->name ? ",name" : ""
    );
    break;
  default:
    break;
  }
  fputc('\n', fp);
  if (depth >= max_depth) return;
  for (uint16_t i = 0; i < u->n_src; i++)
    poly_uop_dump_tree(fp, u->src[i], depth + 2, max_depth);
}

/* Named buffer registry */

#include <stdarg.h>

static uint32_t reg_str_hash(const char *s) {
  uint32_t h = 2166136261u;
  for (; *s; s++)
    h = (h ^ (uint8_t)*s) * 16777619u;
  return h;
}

static bool reg_str_eq(const void *a, const void *b) {
  return strcmp((const char *)a, (const char *)b) == 0;
}

static char *arena_strdup(PolyArena *a, const char *s) {
  size_t len = strlen(s) + 1;
  char *p = poly_arena_alloc(a, len, 1);
  if (p) memcpy(p, s, len);
  return p;
}

static char *arena_vsprintf(PolyArena *a, const char *fmt, va_list ap) {
  va_list ap2;
  va_copy(ap2, ap);
  int len = vsnprintf(NULL, 0, fmt, ap2);
  va_end(ap2);
  if (len < 0) return NULL;
  char *p = poly_arena_alloc(a, (size_t)len + 1, 1);
  if (p) vsnprintf(p, (size_t)len + 1, fmt, ap);
  return p;
}

/* Internal: grow the entries array */
static int reg_grow_entries(PolyCtx *ctx) {
  int new_cap = ctx->entries_cap ? ctx->entries_cap * 2 : 16;
  PolyRegEntry **p = realloc(ctx->entries, (size_t)new_cap * sizeof(PolyRegEntry *));
  if (!p) return -1;
  ctx->entries = p;
  ctx->entries_cap = new_cap;
  return 0;
}

/* Internal: register a named buffer with given role */
static PolyUOp *register_named(
    PolyCtx *ctx,
    PolyBufRole role,
    PolyDType dt,
    const int64_t *shape,
    int ndim,
    const char *name
) {
  if (!ctx || !name || ndim < 0 || ndim > 8) return NULL;

  /* Check for re-registration */
  uint32_t h = reg_str_hash(name);
  PolyRegEntry *existing = poly_map_get(ctx->name_map, h, name, reg_str_eq);
  if (existing) {
    /* Validate dtype: compare by priority+bitsize (works for both scalar and ptr dtypes) */
    if (existing->buffer->dtype.priority != dt.priority ||
        existing->buffer->dtype.bitsize != dt.bitsize) {
      fprintf(stderr, "poly_register: '%s' already registered with different dtype\n", name);
      return NULL;
    }
    /* Validate shape */
    if (existing->ndim != ndim) {
      fprintf(
          stderr, "poly_register: '%s' already registered with different ndim (%d vs %d)\n", name,
          existing->ndim, ndim
      );
      return NULL;
    }
    for (int i = 0; i < ndim; i++) {
      if (existing->shape[i] != shape[i]) {
        fprintf(stderr, "poly_register: '%s' already registered with different shape\n", name);
        return NULL;
      }
    }
    return existing->buffer;
  }

  /* Compute numel */
  int64_t numel = 1;
  for (int i = 0; i < ndim; i++)
    numel *= shape[i];

  /* Create BUFFER UOp with unique tag to avoid CSE dedup.
   * Use scalar dtype (not PtrDType) — matches poly_buffer() in the retired
   * single-kernel scheduler path.
   * The scheduling pipeline creates proper PARAM ptrs during lowering. */
  PolyUOp *buf =
      poly_uop_tagged(ctx, POLY_OP_BUFFER, dt, NULL, 0, poly_arg_int(numel), ctx->next_buf_tag++);

  /* Arena-alloc entry */
  PolyRegEntry *entry = poly_arena_alloc(ctx->arena, sizeof(PolyRegEntry), _Alignof(PolyRegEntry));
  if (!entry) return NULL;
  entry->name = arena_strdup(ctx->arena, name);
  entry->role = role;
  entry->buffer = buf;
  entry->ndim = ndim;
  entry->is_alias = false;
  entry->trainable = (role == POLY_ROLE_PARAM);
  for (int i = 0; i < ndim; i++)
    entry->shape[i] = shape[i];

  /* Append to entries array */
  if (ctx->n_entries >= ctx->entries_cap && reg_grow_entries(ctx) < 0) return NULL;
  ctx->entries[ctx->n_entries++] = entry;

  /* Insert into name map */
  poly_map_set(ctx->name_map, h, entry->name, entry, reg_str_eq);

  return buf;
}

static PolyUOp *register_existing_named(
    PolyCtx *ctx,
    PolyBufRole role,
    PolyUOp *buffer,
    const int64_t *shape,
    int ndim,
    const char *name,
    bool trainable
) {
  if (!ctx || !name || !buffer || ndim < 0 || ndim > 8) return NULL;
  const PolyUOp *identity = poly_uop_get_buffer_identity(buffer);
  if (!identity || identity != buffer) {
    fprintf(stderr, "poly_register_existing_buffer: '%s' is not a BUFFER identity\n", name);
    return NULL;
  }

  uint32_t h = reg_str_hash(name);
  PolyRegEntry *existing = poly_map_get(ctx->name_map, h, name, reg_str_eq);
  if (existing) {
    if (existing->buffer != buffer) {
      fprintf(stderr, "poly_register_existing_buffer: '%s' already names another buffer\n", name);
      return NULL;
    }
    if (existing->ndim != ndim) {
      fprintf(stderr, "poly_register_existing_buffer: '%s' registered with different ndim\n", name);
      return NULL;
    }
    for (int i = 0; i < ndim; i++) {
      if (existing->shape[i] != shape[i]) {
        fprintf(
            stderr, "poly_register_existing_buffer: '%s' registered with different shape\n", name
        );
        return NULL;
      }
    }
    existing->role = role;
    existing->trainable = trainable;
    return existing->buffer;
  }

  PolyRegEntry *entry = poly_arena_alloc(ctx->arena, sizeof(PolyRegEntry), _Alignof(PolyRegEntry));
  if (!entry) return NULL;
  entry->name = arena_strdup(ctx->arena, name);
  entry->role = role;
  entry->buffer = buffer;
  entry->ndim = ndim;
  entry->is_alias = false;
  entry->trainable = trainable;
  for (int i = 0; i < ndim; i++)
    entry->shape[i] = shape[i];

  if (ctx->n_entries >= ctx->entries_cap && reg_grow_entries(ctx) < 0) return NULL;
  ctx->entries[ctx->n_entries++] = entry;
  poly_map_set(ctx->name_map, h, entry->name, entry, reg_str_eq);
  return buffer;
}

/* Public registration wrappers */

PolyUOp *poly_param(
    PolyCtx *ctx,
    PolyDType dt,
    const int64_t *shape,
    int ndim,
    const char *fmt,
    ...
) {
  va_list ap;
  va_start(ap, fmt);
  char *name = arena_vsprintf(ctx->arena, fmt, ap);
  va_end(ap);
  if (!name) return NULL;
  return register_named(ctx, POLY_ROLE_PARAM, dt, shape, ndim, name);
}

PolyUOp *poly_input(
    PolyCtx *ctx,
    PolyDType dt,
    const int64_t *shape,
    int ndim,
    const char *fmt,
    ...
) {
  va_list ap;
  va_start(ap, fmt);
  char *name = arena_vsprintf(ctx->arena, fmt, ap);
  va_end(ap);
  if (!name) return NULL;
  return register_named(ctx, POLY_ROLE_INPUT, dt, shape, ndim, name);
}

PolyUOp *poly_output(
    PolyCtx *ctx,
    PolyDType dt,
    const int64_t *shape,
    int ndim,
    const char *fmt,
    ...
) {
  va_list ap;
  va_start(ap, fmt);
  char *name = arena_vsprintf(ctx->arena, fmt, ap);
  va_end(ap);
  if (!name) return NULL;
  return register_named(ctx, POLY_ROLE_OUTPUT, dt, shape, ndim, name);
}

PolyUOp *poly_target(
    PolyCtx *ctx,
    PolyDType dt,
    const int64_t *shape,
    int ndim,
    const char *fmt,
    ...
) {
  va_list ap;
  va_start(ap, fmt);
  char *name = arena_vsprintf(ctx->arena, fmt, ap);
  va_end(ap);
  if (!name) return NULL;
  return register_named(ctx, POLY_ROLE_TARGET, dt, shape, ndim, name);
}

PolyUOp *poly_aux(
    PolyCtx *ctx,
    PolyDType dt,
    const int64_t *shape,
    int ndim,
    const char *fmt,
    ...
) {
  va_list ap;
  va_start(ap, fmt);
  char *name = arena_vsprintf(ctx->arena, fmt, ap);
  va_end(ap);
  if (!name) return NULL;
  return register_named(ctx, POLY_ROLE_AUX, dt, shape, ndim, name);
}

PolyUOp *poly_register_buffer_by_id(
    PolyCtx *ctx,
    int role,
    int dtype_id,
    const int64_t *shape,
    int ndim,
    const char *name
) {
  PolyDType dt;
  if (!poly_dtype_by_id(dtype_id, &dt)) return NULL;
  if (role < POLY_ROLE_PARAM || role > POLY_ROLE_AUX) return NULL;
  return register_named(ctx, (PolyBufRole)role, dt, shape, ndim, name);
}

PolyUOp *poly_register_existing_buffer(
    PolyCtx *ctx,
    int role,
    PolyUOp *buffer,
    const int64_t *shape,
    int ndim,
    const char *name,
    bool trainable
) {
  if (role < POLY_ROLE_PARAM || role > POLY_ROLE_AUX) return NULL;
  return register_existing_named(ctx, (PolyBufRole)role, buffer, shape, ndim, name, trainable);
}

/* Alias */

int poly_alias(PolyCtx *ctx, const char *alias_name, const char *existing_name) {
  if (!ctx || !alias_name || !existing_name) return -1;

  uint32_t eh = reg_str_hash(existing_name);
  PolyRegEntry *existing = poly_map_get(ctx->name_map, eh, existing_name, reg_str_eq);
  if (!existing) return -1;

  uint32_t ah = reg_str_hash(alias_name);
  PolyRegEntry *check = poly_map_get(ctx->name_map, ah, alias_name, reg_str_eq);
  if (check) {
    return (check->buffer == existing->buffer) ? 0 : -1;
  }

  PolyRegEntry *entry = poly_arena_alloc(ctx->arena, sizeof(PolyRegEntry), _Alignof(PolyRegEntry));
  if (!entry) return -1;
  entry->name = arena_strdup(ctx->arena, alias_name);
  entry->role = existing->role;
  entry->buffer = existing->buffer;
  entry->ndim = existing->ndim;
  entry->is_alias = true;
  entry->trainable = existing->trainable;
  for (int i = 0; i < existing->ndim; i++)
    entry->shape[i] = existing->shape[i];

  if (ctx->n_entries >= ctx->entries_cap && reg_grow_entries(ctx) < 0) return -1;
  ctx->entries[ctx->n_entries++] = entry;
  poly_map_set(ctx->name_map, ah, entry->name, entry, reg_str_eq);
  return 0;
}

/* Lookup */

PolyUOp *poly_ctx_get(PolyCtx *ctx, const char *fmt, ...) {
  if (!ctx || !fmt) return NULL;
  char buf[256];
  va_list ap;
  va_start(ap, fmt);
  int n = vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  if (n < 0 || n >= (int)sizeof(buf)) return NULL;
  PolyRegEntry *entry = poly_map_get(ctx->name_map, reg_str_hash(buf), buf, reg_str_eq);
  return entry ? entry->buffer : NULL;
}

const PolyRegEntry *poly_ctx_get_entry(PolyCtx *ctx, const char *name) {
  if (!ctx || !name) return NULL;
  return poly_map_get(ctx->name_map, reg_str_hash(name), name, reg_str_eq);
}

int poly_ctx_set_trainable(PolyCtx *ctx, const char *name, bool trainable) {
  if (!ctx || !name) return -1;
  PolyRegEntry *entry = poly_map_get(ctx->name_map, reg_str_hash(name), name, reg_str_eq);
  if (!entry) return -1;
  entry->trainable = trainable;
  for (int i = 0; i < ctx->n_entries; i++) {
    PolyRegEntry *other = ctx->entries[i];
    if (other && other->buffer == entry->buffer) other->trainable = trainable;
  }
  return 0;
}

bool poly_ctx_is_trainable(PolyCtx *ctx, const char *name) {
  if (!ctx || !name) return false;
  PolyRegEntry *entry = poly_map_get(ctx->name_map, reg_str_hash(name), name, reg_str_eq);
  return entry ? entry->trainable : false;
}

/* Enumeration */

int poly_ctx_named_count(PolyCtx *ctx) {
  return ctx ? ctx->n_entries : 0;
}

const PolyRegEntry *poly_ctx_named_entry(PolyCtx *ctx, int i) {
  if (!ctx || i < 0 || i >= ctx->n_entries) return NULL;
  return ctx->entries[i];
}

/* Entrypoints */

int poly_register_entrypoint(PolyCtx *ctx, const char *name, PolyUOp *sink) {
  if (!ctx || !name || !sink) return -1;
  if (ctx->n_ep >= ctx->ep_cap) {
    int new_cap = ctx->ep_cap ? ctx->ep_cap * 2 : 4;
    void *p = realloc(ctx->ep, (size_t)new_cap * sizeof(ctx->ep[0]));
    if (!p) return -1;
    ctx->ep = p;
    ctx->ep_cap = new_cap;
  }
  ctx->ep[ctx->n_ep].name = arena_strdup(ctx->arena, name);
  ctx->ep[ctx->n_ep].sink = sink;
  ctx->n_ep++;
  return 0;
}

int poly_ctx_entrypoint_count(PolyCtx *ctx) {
  return ctx ? ctx->n_ep : 0;
}

const char *poly_ctx_entrypoint_name(PolyCtx *ctx, int i) {
  if (!ctx || i < 0 || i >= ctx->n_ep) return NULL;
  return ctx->ep[i].name;
}

PolyUOp *poly_ctx_entrypoint_sink(PolyCtx *ctx, int i) {
  if (!ctx || i < 0 || i >= ctx->n_ep) return NULL;
  return ctx->ep[i].sink;
}

/* UOp construction helpers (moved from frontend.c) */

int poly_op_count(void) {
  return (int)POLY_OP_COUNT;
}

PolyUOp *poly_const_float(PolyCtx *ctx, double value) {
  return poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(value));
}

PolyUOp *poly_const_double(PolyCtx *ctx, double value) {
  return poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT64, poly_arg_float(value));
}

PolyUOp *poly_const_int(PolyCtx *ctx, int64_t value) {
  return poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(value));
}

PolyUOp *poly_const_typed(PolyCtx *ctx, PolyDType dt, double value) {
  if (poly_dtype_is_float(dt)) return poly_uop0(ctx, POLY_OP_CONST, dt, poly_arg_float(value));
  /* tinygrad/uop/ops.py:553-561: dtype.const owns the canonical scalar
   * representation, so a bool CONST has one identity regardless of caller. */
  if (poly_dtype_is_bool(dt)) return poly_uop0(ctx, POLY_OP_CONST, dt, poly_arg_bool(value != 0.0));
  return poly_uop0(ctx, POLY_OP_CONST, dt, poly_arg_int((int64_t)value));
}

PolyUOp *poly_identity_element(PolyCtx *ctx, PolyOps op, PolyDType dtype) {
  /* Pinned identity_element returns dtype.const({ADD:0, MUL:1,
   * MAX:dtype.min}) (uop/ops.py:47, dtype.py:82-92). Keep the identity typed;
   * routing floating -infinity through an integer cast is not equivalent. */
  if (!ctx || (op != POLY_OP_ADD && op != POLY_OP_MUL && op != POLY_OP_MAX) ||
      dtype.is_ptr)
    return NULL;
  PolyDType scalar = poly_dtype_scalar(dtype);
  if (poly_dtype_is_float(scalar)) {
    double value = op == POLY_OP_ADD ? 0.0 : op == POLY_OP_MUL ? 1.0 : -INFINITY;
    return poly_uop0(ctx, POLY_OP_CONST, dtype, poly_arg_float(value));
  }
  if (poly_dtype_is_bool(scalar)) {
    bool value = op == POLY_OP_MUL;
    return poly_uop0(ctx, POLY_OP_CONST, dtype, poly_arg_bool(value));
  }
  if (!poly_dtype_is_int(scalar)) return NULL;
  int64_t value = op == POLY_OP_MUL ? 1 : 0;
  if (op == POLY_OP_MAX && !poly_dtype_is_unsigned(scalar)) {
    int bits = (int)scalar.bitsize;
    value = bits >= 64 ? INT64_MIN : -(INT64_C(1) << (bits - 1));
  }
  return poly_uop0(ctx, POLY_OP_CONST, dtype, poly_arg_int(value));
}

PolyUOp *poly_alu1(PolyCtx *ctx, PolyOps op, PolyUOp *src) {
  return poly_uop1(ctx, op, src->dtype, src, poly_arg_none());
}

PolyUOp *poly_alu2(PolyCtx *ctx, PolyOps op, PolyUOp *a, PolyUOp *b) {
  PolyDType dt;
  if (op == POLY_OP_CMPLT || op == POLY_OP_CMPNE || op == POLY_OP_CMPEQ) {
    dt = POLY_BOOL;
  } else if (poly_dtype_is_float(a->dtype)) {
    dt = a->dtype;
  } else if (poly_dtype_is_float(b->dtype)) {
    dt = b->dtype;
  } else {
    dt = a->dtype;
  }
  return poly_uop2(ctx, op, dt, a, b, poly_arg_none());
}

PolyUOp *poly_alu3(PolyCtx *ctx, PolyOps op, PolyUOp *a, PolyUOp *b, PolyUOp *c) {
  PolyDType dt = (op == POLY_OP_WHERE) ? b->dtype : a->dtype;
  return poly_uop3(ctx, op, dt, a, b, c, poly_arg_none());
}

PolyUOp *poly_store_val(PolyCtx *ctx, PolyUOp *buf, PolyUOp *value) {
  return poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, buf, value, poly_arg_none());
}

PolyUOp *poly_sink1(PolyCtx *ctx, PolyUOp *store) {
  return poly_uop(ctx, POLY_OP_SINK, POLY_VOID, &store, 1, poly_arg_none());
}

PolyUOp *poly_sink_n(PolyCtx *ctx, PolyUOp **stores, int n) {
  return poly_uop(ctx, POLY_OP_SINK, POLY_VOID, stores, n, poly_arg_none());
}
