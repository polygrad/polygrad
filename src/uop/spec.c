/* Current Tinygrad uop/spec.py kernel-graph validation. */

#include "uop/spec.h"

#include "device.h"

#include <stdio.h>

static bool matches_dtype(PolyUOp *u, PolyDType dtype) {
  if (!u) return false;
  PolyUOp *base = poly_uop_base(u);
  return poly_dtype_eq(u->dtype, dtype) ||
         (base && base->op == POLY_OP_CONST && base->arg.kind == POLY_ARG_INVALID);
}

static bool dtype_matches_or_weak(PolyUOp *u, PolyDType dtype) {
  return u && (matches_dtype(u, dtype) || poly_dtype_is_weak(u->dtype));
}

static bool same_shape(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  int na = poly_uop_ndim(ctx, a), nb = poly_uop_ndim(ctx, b);
  if (na != nb || na < 0) return false;
  for (int i = 0; i < na; i++) {
    PolyUOp *ad = poly_uop_shape_dim(ctx, a, i);
    PolyUOp *bd = poly_uop_shape_dim(ctx, b, i);
    if (ad != bd) {
      int64_t av = 0, bv = 0;
      if (poly_uop_const_i64(ad, &av) != 0 || poly_uop_const_i64(bd, &bv) != 0 || av != bv)
        return false;
    }
  }
  return true;
}

static bool is_memory_index(PolyUOp *u) {
  while (u && (u->op == POLY_OP_CAST || u->op == POLY_OP_BITCAST) && u->n_src == 1)
    u = u->src[0];
  return u && (u->op == POLY_OP_INDEX || u->op == POLY_OP_SHRINK);
}

/* tinygrad@2026-08-22/a9069c177a9d uop/spec.py:50-131 `spec_shared`. */
static bool verify_shared_uop(PolyCtx *ctx, PolyUOp *u) {
  switch (u->op) {
  case POLY_OP_SINK:
    return poly_dtype_eq(u->dtype, POLY_VOID);
  case POLY_OP_NOOP:
    return true;
  case POLY_OP_CONST:
    return u->n_src == 0;
  case POLY_OP_STACK:
    if (u->n_src == 0) return poly_dtype_eq(u->dtype, POLY_VOID);
    for (int i = 0; i < u->n_src; i++)
      if (!same_shape(ctx, u->src[0], u->src[i]) ||
          !dtype_matches_or_weak(u->src[i], u->dtype))
        return false;
    return true;
  case POLY_OP_WHERE:
    return u->n_src == 3 && poly_dtype_eq(u->src[0]->dtype, POLY_BOOL) &&
           dtype_matches_or_weak(u->src[1], u->dtype) &&
           dtype_matches_or_weak(u->src[2], u->dtype);
  case POLY_OP_CAST:
  case POLY_OP_BITCAST:
    return u->n_src == 1 && u->arg.kind == POLY_ARG_DTYPE;
  case POLY_OP_RANGE:
    return u->n_src >= 1 && matches_dtype(u->src[0], u->dtype) && poly_arg_is_range(u->arg);
  case POLY_OP_INDEX:
    if (u->n_src < 1) return false;
    for (int i = 1; i < u->n_src; i++)
      if (!poly_dtype_is_int(u->src[i]->dtype)) return false;
    return true;
  case POLY_OP_END:
    if (u->n_src < 1) return false;
    {
      bool all_ranges = true;
      for (int i = 1; i < u->n_src; i++)
        if (u->src[i]->op != POLY_OP_RANGE) all_ranges = false;
      if (all_ranges) return true;
      return u->n_src == 3 && u->src[1]->op == POLY_OP_RANGE &&
             poly_dtype_eq(u->src[1]->dtype, POLY_VOID) &&
             poly_dtype_eq(u->src[2]->dtype, POLY_BOOL);
    }
  case POLY_OP_PARAM:
    return u->arg.kind == POLY_ARG_PARAM && u->arg.param;
  case POLY_OP_BUFFER:
    return u->arg.kind == POLY_ARG_PARAM && u->arg.param &&
           (u->arg.param->addrspace == POLY_ADDR_REG ||
            u->arg.param->addrspace == POLY_ADDR_LOCAL);
  case POLY_OP_GROUP:
    if (!poly_dtype_eq(u->dtype, POLY_VOID)) return false;
    for (int i = 0; i < u->n_src; i++)
      if (u->src[i]->op != POLY_OP_GROUP && u->src[i]->op != POLY_OP_STORE &&
          u->src[i]->op != POLY_OP_NOOP && u->src[i]->op != POLY_OP_INS &&
          u->src[i]->op != POLY_OP_END)
        return false;
    return true;
  case POLY_OP_AFTER:
    if (u->n_src < 1 || !matches_dtype(u->src[0], u->dtype)) return false;
    return poly_opset_has(POLY_GROUP_MOVEMENT, u->src[0]->op) ||
           u->src[0]->op == POLY_OP_PARAM || u->src[0]->op == POLY_OP_BUFFER ||
           u->src[0]->op == POLY_OP_CONTIGUOUS || u->src[0]->op == POLY_OP_INDEX ||
           u->src[0]->op == POLY_OP_AFTER || u->src[0]->op == POLY_OP_UNSHARD ||
           u->src[0]->op == POLY_OP_BITCAST || u->src[0]->op == POLY_OP_INS;
  case POLY_OP_CUSTOM:
  case POLY_OP_CUSTOMI:
  case POLY_OP_INS:
    return true;
  case POLY_OP_BARRIER:
    return poly_dtype_eq(u->dtype, POLY_VOID);
  case POLY_OP_CALL:
    return u->n_src >= 1 && !poly_dtype_eq(u->src[0]->dtype, POLY_VOID) &&
           matches_dtype(u->src[0], POLY_UINT64);
  case POLY_OP_LOAD:
    if (u->n_src == 1) return is_memory_index(u->src[0]);
    return u->n_src == 3 && is_memory_index(u->src[0]) &&
           matches_dtype(u->src[1], u->dtype) &&
           poly_dtype_eq(u->src[2]->dtype, POLY_BOOL);
  case POLY_OP_STORE:
    if (!poly_dtype_eq(u->dtype, POLY_VOID)) return false;
    if (u->n_src == 2) return true;
    return u->n_src == 3 && is_memory_index(u->src[0]) &&
           poly_dtype_eq(u->src[2]->dtype, POLY_BOOL);
  case POLY_OP_WMMA:
    return u->n_src == 3 && u->arg.kind == POLY_ARG_TENSOR_CORE;
  default:
    break;
  }

  if (poly_opset_has(POLY_GROUP_COMPARISON, u->op)) {
    return u->n_src == 2 && poly_dtype_eq(u->dtype, POLY_BOOL) &&
           (matches_dtype(u->src[0], u->src[1]->dtype) ||
            matches_dtype(u->src[1], u->src[0]->dtype) ||
            poly_dtype_is_weak(u->src[0]->dtype) || poly_dtype_is_weak(u->src[1]->dtype));
  }
  if (u->op == POLY_OP_SHL || u->op == POLY_OP_SHR) {
    if (u->n_src != 2 || poly_dtype_is_float(u->dtype)) return false;
    bool value_ok = matches_dtype(u->src[0], u->dtype) ||
                    poly_dtype_eq(u->src[0]->dtype, POLY_WEAKINT);
    bool count_ok = matches_dtype(u->src[1], u->dtype) ||
                    poly_dtype_eq(u->src[1]->dtype, POLY_UINT32) ||
                    poly_dtype_eq(u->src[1]->dtype, POLY_WEAKINT);
    return value_ok && count_ok;
  }
  if (u->op == POLY_OP_CDIV || u->op == POLY_OP_CMOD ||
      u->op == POLY_OP_FLOORDIV || u->op == POLY_OP_FLOORMOD) {
    bool invalid = false;
    for (int i = 0; i < u->n_src; i++) {
      PolyUOp *base = poly_uop_base(u->src[i]);
      if (base && base->op == POLY_OP_CONST && base->arg.kind == POLY_ARG_INVALID)
        invalid = true;
    }
    if (!poly_dtype_is_int(u->dtype) && !invalid) return false;
  }
  if (poly_opset_has(POLY_GROUP_ALU, u->op)) {
    for (int i = 0; i < u->n_src; i++)
      if (!dtype_matches_or_weak(u->src[i], u->dtype)) return false;
    return u->n_src > 0;
  }
  return false;
}

/* Current Tinygrad uop/spec.py:187-224 `spec_program`. */
static bool verify_program_uop(PolyCtx *ctx, PolyUOp *u) {
  if (u->op == POLY_OP_CONST && poly_dtype_is_weak(u->dtype))
    return u->n_src == 0 &&
           ((poly_dtype_eq(u->dtype, POLY_WEAKINT) &&
             (u->arg.kind == POLY_ARG_INT || u->arg.kind == POLY_ARG_BIGINT)) ||
            (poly_dtype_eq(u->dtype, POLY_WEAKFLOAT) && u->arg.kind == POLY_ARG_FLOAT));
  if (poly_dtype_is_weak(u->dtype)) return false;

  if (u->op == POLY_OP_SHRINK && u->n_src == 3 &&
      (u->src[0]->op == POLY_OP_PARAM || u->src[0]->op == POLY_OP_BUFFER ||
       u->src[0]->op == POLY_OP_AFTER)) {
    PolyUOp *extent = u->src[2];
    while (extent && (extent->op == POLY_OP_CAST || extent->op == POLY_OP_BITCAST) &&
           extent->n_src == 1)
      extent = extent->src[0];
    if (extent && extent->op == POLY_OP_CONST) return true;
  }
  if (poly_opset_has(POLY_GROUP_MOVEMENT, u->op)) return false;
  if (u->op == POLY_OP_BUFFER)
    return u->arg.kind == POLY_ARG_PARAM && u->arg.param &&
           (u->arg.param->addrspace == POLY_ADDR_REG ||
            u->arg.param->addrspace == POLY_ADDR_LOCAL);
  if (u->op == POLY_OP_CONST && u->arg.kind == POLY_ARG_INVALID) return false;
  if (u->op == POLY_OP_IF)
    return poly_dtype_eq(u->dtype, POLY_VOID) && u->n_src == 2 &&
           poly_dtype_eq(u->src[0]->dtype, POLY_BOOL) &&
           (u->src[1]->op == POLY_OP_CAST || u->src[1]->op == POLY_OP_INDEX ||
            u->src[1]->op == POLY_OP_SHRINK);
  if (u->op == POLY_OP_ENDIF)
    return poly_dtype_eq(u->dtype, POLY_VOID) && u->n_src == 1 &&
           u->src[0]->op == POLY_OP_IF;
  if (u->op == POLY_OP_SPECIAL)
    return poly_dtype_eq(u->dtype, POLY_INT32) && u->n_src == 1 &&
           matches_dtype(u->src[0], u->dtype) && u->arg.kind == POLY_ARG_STRING;
  return verify_shared_uop(ctx, u);
}

/* Tinygrad@2026-08-22/a9069c177a9d uop/spec.py:35-44 type_verify. */
static bool type_verify(PolyCtx *ctx, PolyUOp *root, bool (*verify)(PolyCtx *, PolyUOp *)) {
  if (!ctx || !root || !verify) return false;
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, root, &n_topo);
  if (!topo) return false;
  for (int i = 0; i < n_topo; i++) {
    if (poly_ctx_owns_ptr(ctx, topo[i]) && verify(ctx, topo[i])) continue;
    fprintf(
        stderr, "polygrad: UOp verification failed at %d on %s %s %d arg=%d\n", i,
        poly_op_name(topo[i]->op), poly_dtype_name(topo[i]->dtype), topo[i]->n_src,
        (int)topo[i]->arg.kind
    );
    for (int j = 0; j < topo[i]->n_src; j++)
      fprintf(
          stderr, "  src[%d]=%s %s dtype=(%d,%u)\n", j,
          poly_op_name(topo[i]->src[j]->op), poly_dtype_name(topo[i]->src[j]->dtype),
          topo[i]->src[j]->dtype.priority, topo[i]->src[j]->dtype.bitsize
      );
    fprintf(
        stderr, "  self dtype=(%d,%u)\n", topo[i]->dtype.priority,
        topo[i]->dtype.bitsize
    );
    poly_toposort_free(topo);
    return false;
  }
  poly_toposort_free(topo);
  return true;
}

bool poly_type_verify_program(PolyCtx *ctx, PolyUOp *root) {
  return type_verify(ctx, root, verify_program_uop);
}

static bool stack_source_is_kernel_value(PolyUOp *u) {
  return u && (u->op == POLY_OP_CONST || u->op == POLY_OP_PARAM ||
               poly_uop_is_variable(u) || poly_uop_is_bound_var(u));
}

static bool mstack_is_kernel_value(PolyCtx *ctx, PolyUOp *u) {
  bool all_scalar_devices = true;
  for (int i = 0; i < u->n_src; i++) {
    PolyUOp *device = poly_uop_device_uop_cached(ctx, u->src[i], NULL);
    if (!device || device->arg.kind != POLY_ARG_STRING) {
      all_scalar_devices = false;
      break;
    }
  }
  if (all_scalar_devices) return true;
  if (u->n_src == 0) return true;
  for (int i = 1; i < u->n_src; i++)
    if (u->src[i] != u->src[0]) return false;
  return poly_uop_device_uop_cached(ctx, u->src[0], NULL) == NULL;
}

static bool mselect_is_kernel_value(PolyCtx *ctx, PolyUOp *u) {
  if (u->n_src != 1 || u->arg.kind != POLY_ARG_INT) return false;
  PolyUOp *device = poly_uop_device_uop_cached(ctx, u->src[0], NULL);
  return device && device->arg.kind == POLY_ARG_STRING_TUPLE &&
         u->arg.i < device->arg.string_tuple.n;
}

static bool is_device_arg(PolyArg arg) {
  if (arg.kind == POLY_ARG_STRING) return arg.str && arg.str[0];
  if (arg.kind != POLY_ARG_STRING_TUPLE || arg.string_tuple.n < 0) return false;
  for (int i = 0; i < arg.string_tuple.n; i++)
    if (!arg.string_tuple.vals || !arg.string_tuple.vals[i] || !arg.string_tuple.vals[i][0])
      return false;
  return true;
}

static bool param_has_device(const PolyParamArg *arg) {
  if (!arg) return false;
  if (!arg->device_is_tuple) return arg->device && arg->device[0];
  if (arg->device || arg->n_devices < 0) return false;
  for (int i = 0; i < arg->n_devices; i++)
    if (!arg->devices || !arg->devices[i] || !arg->devices[i][0]) return false;
  return true;
}

static bool is_reduce_op(PolyOps op) {
  return op == POLY_OP_ADD || op == POLY_OP_MUL || op == POLY_OP_MAX;
}

/* tinygrad@2026-08-22/a9069c177a9d uop/spec.py:137-202 `spec_tensor`. */
static bool verify_tensor_uop(PolyCtx *ctx, PolyUOp *u) {
  switch (u->op) {
  case POLY_OP_SIN:
  case POLY_OP_LOG2:
  case POLY_OP_EXP2:
  case POLY_OP_SQRT:
  case POLY_OP_RECIPROCAL:
    return u->n_src == 1 &&
           (poly_dtype_is_float(u->dtype) ||
            (poly_uop_base(u->src[0]) &&
             poly_uop_base(u->src[0])->arg.kind == POLY_ARG_INVALID));
  case POLY_OP_BUFFER:
    if (u->arg.kind != POLY_ARG_PARAM || !u->arg.param) return false;
    if (u->arg.param->addrspace == POLY_ADDR_GLOBAL)
      return u->n_src == 1 && matches_dtype(u->src[0], POLY_WEAKINT) &&
             param_has_device(u->arg.param);
    if (poly_uop_is_variable(u))
      return !u->arg.param->device && !u->arg.param->device_is_tuple;
    return verify_shared_uop(ctx, u);
  case POLY_OP_CUSTOM_FUNCTION:
    return u->arg.kind == POLY_ARG_STRING && u->arg.str;
  case POLY_OP_CALL:
    if (poly_dtype_eq(u->dtype, POLY_VOID) && u->n_src >= 1) {
      PolyOps op = u->src[0]->op;
      if (op == POLY_OP_SINK || op == POLY_OP_LINEAR || op == POLY_OP_PROGRAM ||
          op == POLY_OP_COPY || op == POLY_OP_CUSTOM_FUNCTION)
        return true;
    }
    return verify_shared_uop(ctx, u);
  case POLY_OP_FUNCTION:
    return poly_dtype_eq(u->dtype, POLY_VOID) && u->n_src >= 1 &&
           u->src[0]->op == POLY_OP_TUPLE;
  case POLY_OP_TUPLE:
    return poly_dtype_eq(u->dtype, POLY_VOID);
  case POLY_OP_GETTUPLE: {
    if (u->n_src != 1 || u->arg.kind != POLY_ARG_INT) return false;
    PolyUOp *tuple = u->src[0];
    if (tuple->op == POLY_OP_FUNCTION && tuple->n_src >= 1 &&
        tuple->src[0]->op == POLY_OP_TUPLE)
      tuple = tuple->src[0];
    if (tuple->op != POLY_OP_TUPLE || u->arg.i < 0 || u->arg.i >= tuple->n_src) return false;
    return matches_dtype(tuple->src[u->arg.i], u->dtype);
  }
  case POLY_OP_SPECIAL:
    return u->n_src == 1 && poly_dtype_eq(u->src[0]->dtype, POLY_WEAKINT) &&
           matches_dtype(u->src[0], u->dtype) && u->arg.kind == POLY_ARG_STRING;
  case POLY_OP_RESHAPE:
  case POLY_OP_EXPAND:
    return u->n_src == 2;
  case POLY_OP_PAD:
  case POLY_OP_SHRINK:
    return u->n_src == 3 && same_shape(ctx, u->src[1], u->src[2]);
  case POLY_OP_PERMUTE:
  case POLY_OP_FLIP:
    return u->n_src == 1 && u->arg.kind == POLY_ARG_INT_TUPLE;
  case POLY_OP_REDUCE:
    if (u->n_src < 1 || u->arg.kind != POLY_ARG_REDUCE ||
        !is_reduce_op(u->arg.reduce.op))
      return false;
    for (int i = 1; i < u->n_src; i++)
      if (!poly_dtype_eq(u->src[i]->dtype, POLY_WEAKINT) &&
          !poly_dtype_eq(u->src[i]->dtype, POLY_INT32))
        return false;
    return true;
  case POLY_OP_COPY:
    return u->n_src == 1 && matches_dtype(u->src[0], u->dtype) && is_device_arg(u->arg);
  case POLY_OP_ALLREDUCE:
    return u->n_src == 1 && matches_dtype(u->src[0], u->dtype) &&
           u->arg.kind == POLY_ARG_ALLREDUCE &&
           is_reduce_op(u->arg.allreduce.op) &&
           (u->arg.allreduce.device_is_tuple
                ? is_device_arg(poly_arg_string_tuple(
                      u->arg.allreduce.devices, u->arg.allreduce.n_devices
                  ))
                : u->arg.allreduce.device && u->arg.allreduce.device[0]);
  case POLY_OP_UNSHARD:
    if (u->arg.kind != POLY_ARG_INT_TUPLE || u->n_src != 1 + u->arg.int_tuple.n ||
        !matches_dtype(u->src[0], u->dtype))
      return false;
    for (int i = 0; i < u->arg.int_tuple.n; i++)
      if (!poly_dtype_is_weak(u->src[i + 1]->dtype))
        return false;
    return true;
  case POLY_OP_MSELECT:
    return mselect_is_kernel_value(ctx, u);
  case POLY_OP_MSTACK:
    return mstack_is_kernel_value(ctx, u);
  case POLY_OP_DETACH:
  case POLY_OP_CONTIGUOUS:
  case POLY_OP_CONTIGUOUS_BACKWARD:
    return u->n_src == 1 && u->arg.kind == POLY_ARG_NONE &&
           matches_dtype(u->src[0], u->dtype);
  case POLY_OP_STAGE:
    return u->n_src >= 1;
  case POLY_OP_LINEAR:
    return poly_dtype_eq(u->dtype, POLY_VOID);
  case POLY_OP_SOURCE:
    return poly_dtype_eq(u->dtype, POLY_VOID) && u->n_src == 0;
  case POLY_OP_BINARY:
    return poly_dtype_eq(u->dtype, POLY_UINT8) && u->n_src == 0 &&
           u->arg.kind == POLY_ARG_BYTES;
  case POLY_OP_PROGRAM:
    if (!poly_dtype_eq(u->dtype, POLY_VOID) || u->n_src < 1 || u->n_src > 4 ||
        u->src[0]->op != POLY_OP_SINK)
      return false;
    if (u->n_src >= 2 && u->src[1]->op != POLY_OP_LINEAR) return false;
    if (u->n_src >= 3 && u->src[2]->op != POLY_OP_SOURCE) return false;
    return u->n_src < 4 || u->src[3]->op == POLY_OP_BINARY;
  default:
    return verify_shared_uop(ctx, u);
  }
}

bool poly_type_verify_tensor(PolyCtx *ctx, PolyUOp *root) {
  return type_verify(ctx, root, verify_tensor_uop);
}

/* Current Tinygrad uop/spec.py:252-273 `spec_kernel_graph`. */
static bool verify_kernel_graph_uop(PolyCtx *ctx, PolyUOp *u) {
  if (!u) return false;
  switch (u->op) {
  case POLY_OP_SINK:
    return poly_dtype_eq(u->dtype, POLY_VOID);
  case POLY_OP_STORE:
    return poly_dtype_eq(u->dtype, POLY_VOID) && u->n_src == 2 &&
           u->src[0]->op == POLY_OP_BUFFER && poly_uop_is_variable(u->src[0]) &&
           u->src[1]->op == POLY_OP_CONST;
  case POLY_OP_CONST:
    return u->n_src == 0;
  case POLY_OP_STACK:
    for (int i = 0; i < u->n_src; i++)
      if (!stack_source_is_kernel_value(u->src[i])) return false;
    return true;
  case POLY_OP_PARAM:
    return u->arg.kind == POLY_ARG_PARAM && u->arg.param;
  case POLY_OP_BUFFER:
    return u->arg.kind == POLY_ARG_PARAM && u->arg.param &&
           (u->arg.param->addrspace == POLY_ADDR_GLOBAL ||
            u->arg.param->addrspace == POLY_ADDR_ALU);
  case POLY_OP_RESHAPE:
  case POLY_OP_BITCAST:
    return true;
  case POLY_OP_MSTACK:
    return mstack_is_kernel_value(ctx, u);
  case POLY_OP_MSELECT:
    return mselect_is_kernel_value(ctx, u);
  case POLY_OP_CALL:
    return u->n_src >= 1 &&
           (u->src[0]->op == POLY_OP_SINK || u->src[0]->op == POLY_OP_LINEAR ||
            u->src[0]->op == POLY_OP_PROGRAM ||
            u->src[0]->op == POLY_OP_CUSTOM_FUNCTION);
  case POLY_OP_AFTER: {
    if (u->n_src < 1 || !matches_dtype(u->src[0], u->dtype)) return false;
    PolyOps op = u->src[0]->op;
    return poly_opset_has(POLY_GROUP_MOVEMENT, op) || op == POLY_OP_PARAM ||
           op == POLY_OP_AFTER || op == POLY_OP_BUFFER || op == POLY_OP_MSTACK ||
           op == POLY_OP_MSELECT || op == POLY_OP_BITCAST || op == POLY_OP_RESHAPE;
  }
  default:
    return false;
  }
}

/* Current Tinygrad uop/spec.py:35-44 `type_verify`, with
 * `enter_calls=False` as required by get_kernel_graph. */
bool poly_type_verify_kernel_graph(PolyCtx *ctx, PolyUOp *root) {
  if (!ctx || !root) return false;
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_ex_alloc(ctx, root, &n_topo, NULL, false);
  if (!topo) return false;
  for (int i = 0; i < n_topo; i++) {
    if (verify_kernel_graph_uop(ctx, topo[i])) continue;
    fprintf(
        stderr, "polygrad: UOp verification failed at %d on %s %s %d\n", i,
        poly_op_name(topo[i]->op), poly_dtype_name(topo[i]->dtype), topo[i]->n_src
    );
    poly_toposort_free(topo);
    return false;
  }
  poly_toposort_free(topo);
  return true;
}
