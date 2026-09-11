/*
 * autograd.c — Reverse-mode autodiff for tensor-level UOp graphs
 *
 * Provides poly_grad(loss, wrt): builds a gradient expression graph for
 * d(loss)/d(wrt) using reverse-mode accumulation on the UOp DAG.
 *
 * Current scope:
 * - Core ALU: ADD, SUB, MUL, FDIV, NEG, EXP2, LOG2, SQRT, RECIPROCAL, SIN, POW
 * - Binary: MAX (elementwise)
 * - Movement: RESHAPE, EXPAND, PERMUTE, PAD, SHRINK, FLIP
 * - Reductions: tensor REDUCE with ADD, MAX and MUL
 * - Utility: CAST pass-through, CONTIGUOUS/COPY/BUFFERIZE pass-through
 * - Stop gradient: DETACH, CMPLT, CMPNE, BITCAST
 * - Target-pruned reverse pass (port of tinygrad's _deepwalk)
 */

#include "ctx.h"
#include "uop/ops.h"
#include "uop/upat.h"
#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"

/* Local helpers */

/* Current pm_gradient writes ordinary Python literals into ElementwiseMixin
 * expressions. UOp.ufix creates a weak CONST and _broadcasted keeps the
 * promoted literal weak (mixin/gradient.py:49-62,
 * mixin/elementwise.py:18-29). */
static PolyUOp *ufix_like(PolyCtx *ctx, PolyUOp *ref, double value) {
  return ref ? poly_const_typed(ctx, poly_dtype_weak(ref->dtype), value) : NULL;
}

static PolyUOp *cast_to(PolyCtx *ctx, PolyUOp *u, PolyDType dt) {
  if (poly_dtype_eq(u->dtype, dt)) return u;
  return poly_uop1(ctx, POLY_OP_CAST, dt, u, poly_arg_none());
}

static PolyUOp *shape_dim_node(PolyCtx *ctx, PolyUOp *u, PolyShape shape, int idx) {
  PolyUOp *dim = poly_uop_shape_dim(ctx, u, idx);
  if (dim) return dim;
  if (idx < 0 || idx >= shape.ndim) return NULL;
  return poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(shape.dims[idx]));
}

static PolyUOp *broadcast_to_uop_shape(PolyCtx *ctx, PolyUOp *value, PolyUOp *target) {
  int ndim = poly_uop_ndim(ctx, target);
  if (ndim < 0 || ndim > POLY_MAX_DIMS) return NULL;
  PolyUOp *shape[POLY_MAX_DIMS];
  for (int i = 0; i < ndim; i++)
    if (!(shape[i] = poly_uop_shape_dim(ctx, target, i))) return NULL;
  return poly_expand_uop(ctx, value, shape, ndim);
}

static PolyUOp *grad_get(PolyMap *grads, PolyUOp *u) {
  return poly_map_get(grads, poly_ptr_hash(u), u, poly_ptr_eq);
}

static bool shape_dim_equal(PolyUOp *a, PolyUOp *b) {
  if (a == b) return true;
  int64_t av = 0, bv = 0;
  return a && b && poly_uop_const_i64(a, &av) == 0 && poly_uop_const_i64(b, &bv) == 0 && av == bv;
}

static bool shaped_edge_equal(PolyCtx *ctx, PolyUOp *source, PolyUOp *grad) {
  int source_ndim = poly_uop_ndim(ctx, source), grad_ndim = poly_uop_ndim(ctx, grad);
  if (source_ndim < 0 || grad_ndim < 0) return true;
  if (source_ndim != grad_ndim) return false;
  for (int i = 0; i < source_ndim; i++)
    if (!shape_dim_equal(poly_uop_shape_dim(ctx, source, i), poly_uop_shape_dim(ctx, grad, i)))
      return false;
  return true;
}

/* Current tinygrad mixin/gradient.py:121-124 repairs every shaped gradient
 * edge once, after the local derivative rule and before accumulation. */
static PolyUOp *shape_gradient_edge(PolyCtx *ctx, PolyUOp *source, PolyUOp *grad) {
  int source_ndim = poly_uop_ndim(ctx, source), grad_ndim = poly_uop_ndim(ctx, grad);
  if (source_ndim < 0 || grad_ndim < 0 || shaped_edge_equal(ctx, source, grad)) return grad;
  PolyDType original_dtype = grad->dtype, acc_dtype;
  if (!poly_sum_acc_dtype(original_dtype, &acc_dtype)) return NULL;
  grad = cast_to(ctx, grad, acc_dtype);
  int axes[POLY_MAX_DIMS];
  int n_axes = poly_broadcast_axes(ctx, source, grad, axes, POLY_MAX_DIMS);
  if (n_axes < 0) return NULL;
  int64_t reduce_axes[POLY_MAX_DIMS];
  for (int i = 0; i < n_axes; i++)
    reduce_axes[i] = axes[i];
  grad = poly_reduce_axis(ctx, POLY_OP_ADD, grad, reduce_axes, n_axes);
  if (!grad) return NULL;
  if (!shaped_edge_equal(ctx, source, grad)) {
    PolyUOp *target_shape[POLY_MAX_DIMS];
    for (int i = 0; i < source_ndim; i++)
      if (!(target_shape[i] = poly_uop_shape_dim(ctx, source, i))) return NULL;
    grad = poly_reshape_uop(ctx, grad, target_shape, source_ndim);
  }
  return grad ? cast_to(ctx, grad, original_dtype) : NULL;
}

static bool grad_add(PolyCtx *ctx, PolyMap *grads, PolyUOp *u, PolyUOp *g) {
  if (!u || !g || !(g = shape_gradient_edge(ctx, u, g))) return false;
  PolyUOp *old = grad_get(grads, u);
  if (!old) {
    poly_map_set(grads, poly_ptr_hash(u), u, g, poly_ptr_eq);
    return true;
  }
  PolyUOp *rhs = cast_to(ctx, g, old->dtype);
  PolyUOp *sum = poly_uop2(ctx, POLY_OP_ADD, old->dtype, old, rhs, poly_arg_none());
  poly_map_set(grads, poly_ptr_hash(u), u, sum, poly_ptr_eq);
  return true;
}

static PolyUOp *const_like(PolyCtx *ctx, PolyUOp *u, double value) {
  /* Current UOp.const_like is shared core behavior, not an autograd-specific
   * constructor (uop/ops.py:581-583). */
  return poly_const_like_float(ctx, u, value);
}

static PolyUOp *ones_like(PolyCtx *ctx, PolyUOp *u) {
  return const_like(ctx, u, 1.0);
}

static PolyUOp *zeros_like(PolyCtx *ctx, PolyUOp *u) {
  return const_like(ctx, u, 0.0);
}

/* Pinned gradient.py constructs every product explicitly, including products
 * by an all-one upstream (lines 51-65). Tensor-graph parity depends on
 * preserving that construction order; later compiler rewrites remain free to
 * simplify after scheduling reaches the appropriate stage. */
static PolyUOp *mul_grad(PolyCtx *ctx, PolyUOp *g, PolyUOp *x, PolyDType dt) {
  return poly_uop2(ctx, POLY_OP_MUL, dt, x, g, poly_arg_none());
}

/* Current ElementwiseMixin._broadcasted promotes operand dtypes but leaves
 * shape broadcasting implicit in UOp shape inference (elementwise.py:19-29).
 * POW has no public raw-UOp helper, so retain the exact local spelling here. */
static PolyUOp *pow_grad(PolyCtx *ctx, PolyUOp *base, PolyUOp *exponent) {
  if (!poly_broadcasted_pair(ctx, &base, &exponent)) return NULL;
  return poly_alu2(ctx, POLY_OP_POW, base, exponent);
}

/* Target-pruned walk (port of tinygrad _deepwalk) */

/* Compute the subset of topo[] that lies on paths from any target to root.
 * Skips DETACH nodes. Returns filtered array (arena-allocated). */
static PolyUOp **target_walk(
    PolyCtx *ctx,
    PolyUOp **topo,
    int n_topo,
    PolyUOp **targets,
    int n_targets,
    int *n_out
) {
  PolyMap *target_set = poly_map_new((size_t)n_targets * 2 + 16);
  for (int i = 0; i < n_targets; i++)
    poly_map_set(target_set, poly_ptr_hash(targets[i]), targets[i], targets[i], poly_ptr_eq);

  /* Forward pass: mark nodes whose sources lead to any target */
  PolyMap *in_path = poly_map_new((size_t)n_topo * 2 + 16);
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (u->op == POLY_OP_DETACH) continue;
    bool on_path = poly_map_get(target_set, poly_ptr_hash(u), u, poly_ptr_eq) != NULL;
    if (!on_path) {
      for (int j = 0; j < u->n_src; j++) {
        PolyUOp *s = u->src[j];
        if (poly_map_get(target_set, poly_ptr_hash(s), s, poly_ptr_eq) ||
            poly_map_get(in_path, poly_ptr_hash(s), s, poly_ptr_eq)) {
          on_path = true;
          break;
        }
      }
    }
    if (on_path) poly_map_set(in_path, poly_ptr_hash(u), u, u, poly_ptr_eq);
  }

  PolyUOp **result = malloc((size_t)n_topo * sizeof(PolyUOp *));
  if (!result) {
    poly_map_destroy(target_set);
    poly_map_destroy(in_path);
    *n_out = 0;
    return NULL;
  }
  int count = 0;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (poly_map_get(in_path, poly_ptr_hash(u), u, poly_ptr_eq)) result[count++] = u;
  }

  poly_map_destroy(target_set);
  poly_map_destroy(in_path);
  *n_out = count;
  return result;
}

/* Pinned UOp.base strips only movement, MULTI and DETACH
 * (tinygrad/uop/ops.py:675-680). FUNCTION backward keeps shaped constant
 * output gradients inline and parameterizes every other output gradient. */
static PolyUOp *grad_uop_base(PolyUOp *u) {
  while (u && u->n_src > 0 &&
         (poly_opset_has(POLY_GROUP_MOVEMENT, u->op) || u->op == POLY_OP_UNSHARD ||
          u->op == POLY_OP_DETACH))
    u = u->src[0];
  return u;
}

static bool grad_walk_contains(PolyUOp **walk, int n_walk, PolyUOp *u) {
  for (int i = 0; i < n_walk; i++)
    if (walk[i] == u) return true;
  return false;
}

static PolyUOp *grad_param_replace_slot(PolyCtx *ctx, PolyUOp *param, int64_t slot) {
  if (!ctx || !param || param->op != POLY_OP_PARAM || param->n_src != 1 ||
      param->arg.kind != POLY_ARG_PARAM || !param->arg.param)
    return NULL;
  PolyParamArg arg = *param->arg.param;
  arg.slot = slot;
  return poly_uop1(ctx, POLY_OP_PARAM, param->dtype, param->src[0], poly_arg_param(&arg));
}

/* Pinned _compact_params (tinygrad/mixin/gradient.py:17-21): sort the PARAMs
 * actually reachable from the backward body by their old slots, renumber them
 * densely, and retain only the corresponding call arguments. */
static bool grad_compact_params(
    PolyCtx *ctx,
    PolyUOp *body,
    PolyUOp **all_args,
    int n_all_args,
    PolyUOp **out_body,
    PolyUOp ***out_args,
    int *out_n_args
) {
  if (!ctx || !body || !out_body || !out_args || !out_n_args || n_all_args < 0 ||
      (n_all_args > 0 && !all_args))
    return false;
  *out_body = NULL;
  *out_args = NULL;
  *out_n_args = 0;

  int n_topo = 0;
  PolyUOp **topo = poly_toposort_ex_alloc(ctx, body, &n_topo, NULL, false);
  PolyUOp **by_slot = n_all_args > 0 ? calloc((size_t)n_all_args, sizeof(*by_slot)) : NULL;
  if (!topo || (n_all_args > 0 && !by_slot)) {
    poly_toposort_free(topo);
    free(by_slot);
    return false;
  }
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (!u || u->op != POLY_OP_PARAM || u->arg.kind != POLY_ARG_PARAM || !u->arg.param) continue;
    int64_t slot = u->arg.param->slot;
    if (slot < 0 || slot >= n_all_args) {
      poly_toposort_free(topo);
      free(by_slot);
      return false;
    }
    by_slot[slot] = u;
  }

  int n_used = 0;
  for (int i = 0; i < n_all_args; i++)
    if (by_slot[i]) n_used++;
  PolyUOp **from = n_used > 0 ? malloc((size_t)n_used * sizeof(*from)) : NULL;
  PolyUOp **temporary = n_used > 0 ? malloc((size_t)n_used * sizeof(*temporary)) : NULL;
  PolyUOp **to = n_used > 0 ? malloc((size_t)n_used * sizeof(*to)) : NULL;
  PolyUOp **args = n_used > 0 ? malloc((size_t)n_used * sizeof(*args)) : NULL;
  if (n_used > 0 && (!from || !temporary || !to || !args)) {
    poly_toposort_free(topo);
    free(by_slot);
    free(from);
    free(temporary);
    free(to);
    free(args);
    return false;
  }
  int at = 0;
  for (int i = 0; i < n_all_args; i++) {
    if (!by_slot[i]) continue;
    from[at] = by_slot[i];
    temporary[at] = grad_param_replace_slot(ctx, by_slot[i], n_all_args + at);
    to[at] = grad_param_replace_slot(ctx, by_slot[i], at);
    args[at] = all_args[i];
    if (!temporary[at] || !to[at] || !args[at]) {
      poly_toposort_free(topo);
      free(by_slot);
      free(from);
      free(temporary);
      free(to);
      free(args);
      return false;
    }
    at++;
  }
  /* The generic substitution follows replacement nodes recursively. A final
   * dense PARAM can be identical to another old PARAM, so direct renaming can
   * chain two intended terminal replacements. Pinned graph_rewrite does not.
   * Rename through disjoint slots, then densify in a second pass. */
  PolyUOp *staged = n_used > 0 ? poly_uop_substitute(ctx, body, from, temporary, n_used) : body;
  PolyUOp *compacted =
      n_used > 0 ? poly_uop_substitute(ctx, staged, temporary, to, n_used) : staged;
  poly_toposort_free(topo);
  free(by_slot);
  free(from);
  free(temporary);
  free(to);
  if (!compacted) {
    free(args);
    return false;
  }
  *out_body = compacted;
  *out_args = args;
  *out_n_args = n_used;
  return true;
}

/* Public API */

/* Core gradient reverse pass. Builds the gradient map from loss backward.
 * When targets/n_targets are provided, only walks nodes on paths to targets
 * (port of tinygrad's _deepwalk). This prevents unsupported ops on
 * unrelated branches from crashing the backward pass.
 * Returns the gradient map on success, NULL on failure.
 * Caller must call poly_map_destroy on the returned map. */
static PolyMap *grad_reverse_pass(
    PolyCtx *ctx,
    PolyUOp *loss,
    PolyUOp *initial_grad,
    PolyUOp **targets,
    int n_targets
) {
  int n_topo = 0;
  PolyScratchMark scratch = poly_ctx_scratch_mark(ctx);
  PolyUOp **topo = poly_toposort_scratch(ctx, loss, &n_topo);
  if (!topo || n_topo <= 0) {
    poly_ctx_scratch_rewind(ctx, scratch);
    return NULL;
  }

  /* Target-prune: only walk nodes on paths to targets */
  PolyUOp **walk = topo;
  int n_walk = n_topo;
  bool walk_owned = false;
  if (targets && n_targets > 0) {
    walk = target_walk(ctx, topo, n_topo, targets, n_targets, &n_walk);
    walk_owned = true;
    if (!walk || n_walk <= 0) {
      /* No path from loss to any target — return empty gradient map */
      free(walk);
      PolyMap *grads = poly_map_new(16);
      poly_ctx_scratch_rewind(ctx, scratch);
      return grads;
    }
  }

  PolyMap *grads = poly_map_new((size_t)n_walk * 2 + 16);
  if (!grads) {
    if (walk_owned) free(walk);
    poly_ctx_scratch_rewind(ctx, scratch);
    return NULL;
  }

#define GRAD_REVERSE_FAIL()                                                                        \
  do {                                                                                             \
    if (walk_owned) free(walk);                                                                    \
    poly_map_destroy(grads);                                                                       \
    poly_ctx_scratch_rewind(ctx, scratch);                                                         \
    return NULL;                                                                                   \
  } while (0)

#define GRAD_ADD(source, value)                                                                    \
  do {                                                                                             \
    if (!grad_add(ctx, grads, (source), (value))) GRAD_REVERSE_FAIL();                             \
  } while (0)

  GRAD_ADD(loss, initial_grad);

  for (int i = n_walk - 1; i >= 0; i--) {
    PolyUOp *u = walk[i];
    PolyUOp *g = grad_get(grads, u);
    if (!g) continue;

    switch (u->op) {
    /* leaf / no-parent cases */
    case POLY_OP_CONST:
    case POLY_OP_BUFFER:
    case POLY_OP_PARAM:
    case POLY_OP_UNIQUE:
    case POLY_OP_DEVICE:
      break;

    /* Pinned compute_gradient accumulates each GETTUPLE contribution into a
     * TUPLE gradient on its owning FUNCTION, filling unused outputs with NOOP
     * (tinygrad/mixin/gradient.py:101-109). */
    case POLY_OP_GETTUPLE: {
      if (u->n_src != 1 || u->arg.kind != POLY_ARG_INT || !u->src[0] ||
          u->src[0]->op != POLY_OP_FUNCTION || u->src[0]->n_src < 1 || !u->src[0]->src[0] ||
          u->src[0]->src[0]->op != POLY_OP_TUPLE) {
        fprintf(stderr, "polygrad: autograd: malformed GETTUPLE/FUNCTION\n");
        GRAD_REVERSE_FAIL();
      }
      PolyUOp *function = u->src[0];
      int n_outputs = function->src[0]->n_src;
      int64_t selected = u->arg.i;
      if (selected < 0 || selected >= n_outputs) {
        fprintf(stderr, "polygrad: autograd: GETTUPLE index out of range\n");
        GRAD_REVERSE_FAIL();
      }
      PolyUOp *old = grad_get(grads, function);
      if (old && (old->op != POLY_OP_TUPLE || old->n_src != n_outputs)) {
        fprintf(stderr, "polygrad: autograd: FUNCTION gradient is not a TUPLE\n");
        GRAD_REVERSE_FAIL();
      }
      PolyUOp **parts = malloc((size_t)n_outputs * sizeof(*parts));
      if (!parts) GRAD_REVERSE_FAIL();
      for (int j = 0; j < n_outputs; j++) {
        PolyUOp *prev = old ? old->src[j] : NULL;
        if (j != selected) {
          parts[j] = prev ? prev : poly_uop0(ctx, POLY_OP_NOOP, POLY_VOID, poly_arg_none());
        } else if (prev && prev->op != POLY_OP_NOOP) {
          PolyUOp *rhs = cast_to(ctx, g, prev->dtype);
          parts[j] = poly_uop2(ctx, POLY_OP_ADD, prev->dtype, prev, rhs, poly_arg_none());
        } else {
          parts[j] = g;
        }
        if (!parts[j]) {
          free(parts);
          GRAD_REVERSE_FAIL();
        }
      }
      PolyUOp *tuple = poly_uop(ctx, POLY_OP_TUPLE, POLY_VOID, parts, n_outputs, poly_arg_none());
      free(parts);
      if (!tuple) GRAD_REVERSE_FAIL();
      poly_map_set(grads, poly_ptr_hash(function), function, tuple, poly_ptr_eq);
    } break;

    /* Pinned pm_gradient maps a TUPLE gradient directly to its ordered sources
     * (tinygrad/mixin/gradient.py:79). */
    case POLY_OP_TUPLE:
      if (g->op != POLY_OP_TUPLE || g->n_src != u->n_src) {
        fprintf(stderr, "polygrad: autograd: malformed TUPLE gradient\n");
        GRAD_REVERSE_FAIL();
      }
      for (int j = 0; j < u->n_src; j++)
        if (g->src[j] && g->src[j]->op != POLY_OP_NOOP) GRAD_ADD(u->src[j], g->src[j]);
      break;

    /* Pinned call_gradient differentiates the value-producing TUPLE body
     * against only needed PARAM slots, compacts its arguments, and emits one
     * backward FUNCTION (tinygrad/mixin/gradient.py:23-47). Custom grad_fxn and
     * precompiled execution remain fail-closed until those behaviors land. */
    case POLY_OP_FUNCTION: {
      if (u->n_src < 1 || !u->src[0] || u->src[0]->op != POLY_OP_TUPLE || !g ||
          g->op != POLY_OP_TUPLE || g->n_src != u->src[0]->n_src) {
        fprintf(stderr, "polygrad: autograd: malformed FUNCTION gradient\n");
        GRAD_REVERSE_FAIL();
      }
      if (u->arg.kind != POLY_ARG_CALL_INFO || !u->arg.call_info ||
          u->arg.call_info->has_grad_fxn || u->arg.call_info->has_aux ||
          u->arg.call_info->precompile || u->arg.call_info->precompile_backward) {
        fprintf(stderr, "polygrad: autograd: unsupported FUNCTION CallInfo\n");
        GRAD_REVERSE_FAIL();
      }
      int n_args = u->n_src - 1;
      int n_outputs = g->n_src;
      PolyUOp **params_by_slot =
          n_args > 0 ? calloc((size_t)n_args, sizeof(*params_by_slot)) : NULL;
      int n_body_topo = 0;
      PolyUOp **body_topo = poly_toposort_ex_alloc(ctx, u->src[0], &n_body_topo, NULL, false);
      if (!body_topo || (n_args > 0 && !params_by_slot)) {
        poly_toposort_free(body_topo);
        free(params_by_slot);
        GRAD_REVERSE_FAIL();
      }
      for (int j = 0; j < n_body_topo; j++) {
        PolyUOp *p = body_topo[j];
        if (!p || p->op != POLY_OP_PARAM || p->arg.kind != POLY_ARG_PARAM || !p->arg.param)
          continue;
        int64_t slot = p->arg.param->slot;
        if (slot >= 0 && slot < n_args) params_by_slot[slot] = p;
      }
      poly_toposort_free(body_topo);

      PolyUOp **needed_params = n_args > 0 ? malloc((size_t)n_args * sizeof(*needed_params)) : NULL;
      int *needed_slots = n_args > 0 ? malloc((size_t)n_args * sizeof(*needed_slots)) : NULL;
      PolyUOp **root_grad_parts =
          n_outputs > 0 ? malloc((size_t)n_outputs * sizeof(*root_grad_parts)) : NULL;
      PolyUOp **all_args = (n_args + n_outputs) > 0
                               ? malloc((size_t)(n_args + n_outputs) * sizeof(*all_args))
                               : NULL;
      if ((n_args > 0 && (!needed_params || !needed_slots)) ||
          (n_outputs > 0 && !root_grad_parts) || (n_args + n_outputs > 0 && !all_args)) {
        free(params_by_slot);
        free(needed_params);
        free(needed_slots);
        free(root_grad_parts);
        free(all_args);
        GRAD_REVERSE_FAIL();
      }

      int n_needed = 0;
      for (int j = 0; j < n_args; j++) {
        all_args[j] = u->src[j + 1];
        if (params_by_slot[j] && grad_walk_contains(walk, n_walk, u->src[j + 1])) {
          needed_params[n_needed] = params_by_slot[j];
          needed_slots[n_needed++] = j;
        }
      }
      for (int j = 0; j < n_outputs; j++) {
        PolyUOp *out_grad = g->src[j];
        all_args[n_args + j] = out_grad;
        if (out_grad->op == POLY_OP_NOOP || grad_uop_base(out_grad)->op == POLY_OP_CONST) {
          root_grad_parts[j] = out_grad;
        } else {
          root_grad_parts[j] = poly_uop_param(ctx, n_args + j, out_grad);
        }
        if (!root_grad_parts[j]) {
          free(params_by_slot);
          free(needed_params);
          free(needed_slots);
          free(root_grad_parts);
          free(all_args);
          GRAD_REVERSE_FAIL();
        }
      }
      PolyUOp *root_grad =
          poly_uop(ctx, POLY_OP_TUPLE, POLY_VOID, root_grad_parts, n_outputs, poly_arg_none());
      free(root_grad_parts);
      if (!root_grad) {
        free(params_by_slot);
        free(needed_params);
        free(needed_slots);
        free(all_args);
        GRAD_REVERSE_FAIL();
      }
      PolyMap *body_grads =
          n_needed > 0 ? grad_reverse_pass(ctx, u->src[0], root_grad, needed_params, n_needed)
                       : poly_map_new(16);
      free(needed_params);
      if (!body_grads) {
        free(params_by_slot);
        free(needed_slots);
        free(all_args);
        GRAD_REVERSE_FAIL();
      }

      PolyUOp **grad_bodies = n_needed > 0 ? malloc((size_t)n_needed * sizeof(*grad_bodies)) : NULL;
      int *grad_slots = n_needed > 0 ? malloc((size_t)n_needed * sizeof(*grad_slots)) : NULL;
      if (n_needed > 0 && (!grad_bodies || !grad_slots)) {
        poly_map_destroy(body_grads);
        free(params_by_slot);
        free(needed_slots);
        free(all_args);
        free(grad_bodies);
        free(grad_slots);
        GRAD_REVERSE_FAIL();
      }
      int n_grad_bodies = 0;
      for (int j = 0; j < n_needed; j++) {
        PolyUOp *body_grad = grad_get(body_grads, params_by_slot[needed_slots[j]]);
        if (!body_grad) continue;
        grad_bodies[n_grad_bodies] = body_grad;
        grad_slots[n_grad_bodies++] = needed_slots[j];
      }
      poly_map_destroy(body_grads);
      free(params_by_slot);
      free(needed_slots);

      if (n_grad_bodies > 0) {
        PolyUOp *backward_body =
            poly_uop(ctx, POLY_OP_TUPLE, POLY_VOID, grad_bodies, n_grad_bodies, poly_arg_none());
        PolyUOp *compact_body = NULL;
        PolyUOp **compact_args = NULL;
        int n_compact_args = 0;
        if (!backward_body || !grad_compact_params(
                                  ctx, backward_body, all_args, n_args + n_outputs, &compact_body,
                                  &compact_args, &n_compact_args
                              )) {
          free(grad_bodies);
          free(grad_slots);
          free(all_args);
          GRAD_REVERSE_FAIL();
        }
        PolyUOp **function_src = malloc((size_t)(n_compact_args + 1) * sizeof(*function_src));
        if (!function_src) {
          free(compact_args);
          free(grad_bodies);
          free(grad_slots);
          free(all_args);
          GRAD_REVERSE_FAIL();
        }
        function_src[0] = compact_body;
        for (int j = 0; j < n_compact_args; j++)
          function_src[j + 1] = compact_args[j];
        const char *forward_name = u->arg.call_info->name ? u->arg.call_info->name : "";
        size_t name_len = strlen(forward_name);
        char *backward_name = malloc(name_len + sizeof("_backward"));
        if (!backward_name) {
          free(function_src);
          free(compact_args);
          free(grad_bodies);
          free(grad_slots);
          free(all_args);
          GRAD_REVERSE_FAIL();
        }
        memcpy(backward_name, forward_name, name_len);
        memcpy(backward_name + name_len, "_backward", sizeof("_backward"));
        PolyCallInfo backward_info = {
            .name = backward_name,
            .precompile = false,
            .precompile_backward = false,
            .has_grad_fxn = false,
            .grad_fxn_key = 0,
            .has_aux = false,
        };
        PolyUOp *backward_function = poly_uop(
            ctx, POLY_OP_FUNCTION, POLY_VOID, function_src, n_compact_args + 1,
            poly_arg_call_info(&backward_info)
        );
        free(backward_name);
        free(function_src);
        free(compact_args);
        if (!backward_function) {
          free(grad_bodies);
          free(grad_slots);
          free(all_args);
          GRAD_REVERSE_FAIL();
        }
        for (int j = 0; j < n_grad_bodies; j++) {
          PolyUOp *input_grad = poly_uop1(
              ctx, POLY_OP_GETTUPLE, grad_bodies[j]->dtype, backward_function, poly_arg_int(j)
          );
          if (!input_grad) {
            free(grad_bodies);
            free(grad_slots);
            free(all_args);
            GRAD_REVERSE_FAIL();
          }
          GRAD_ADD(u->src[grad_slots[j] + 1], input_grad);
        }
      }
      free(grad_bodies);
      free(grad_slots);
      free(all_args);
    } break;

    /* explicitly stop gradient flow */
    case POLY_OP_DETACH:
      break;

    /* non-differentiable ops: stop gradient (tinygrad returns (None,...)) */
    case POLY_OP_CMPLT:
    case POLY_OP_CMPNE:
    case POLY_OP_BITCAST:
      break;

    /* tinygrad AFTER gradients keep the value edge separate from the effect.
     * For an opaque CALL, the shared core routes ctx directly to data while
     * the language frontend invokes the registered custom gradient callback
     * for the CALL edge. For clone/assign, AFTER routes through STORE and
     * STORE routes through the stored value. */
    case POLY_OP_AFTER:
      if (u->n_src == 2 && u->src[1]->op == POLY_OP_CALL) {
        GRAD_ADD(u->src[0], g);
        break;
      }
      if (u->n_src == 2 && u->src[1]->op == POLY_OP_STORE) {
        GRAD_ADD(u->src[1], g);
        break;
      }
      fprintf(stderr, "polygrad: autograd: missing gradient rule for AFTER effect\n");
      GRAD_REVERSE_FAIL();

    case POLY_OP_STORE:
      if (u->n_src == 2) {
        GRAD_ADD(u->src[1], g);
        break;
      }
      fprintf(stderr, "polygrad: autograd: malformed STORE\n");
      GRAD_REVERSE_FAIL();

    /* pass-through (no realize barrier on gradient) */
    case POLY_OP_CONTIGUOUS:
    case POLY_OP_COPY:
    case POLY_OP_NOOP:
    case POLY_OP_STAGE:
      GRAD_ADD(u->src[0], g);
      break;

    /* CONTIGUOUS_BACKWARD: wrap gradient in CONTIGUOUS (realize barrier).
     * Matches tinygrad gradient.py:44: (ctx.contiguous(),) */
    case POLY_OP_CONTIGUOUS_BACKWARD: {
      PolyUOp *gx = poly_uop1(ctx, POLY_OP_CONTIGUOUS, g->dtype, g, poly_arg_none());
      GRAD_ADD(u->src[0], gx);

    } break;

    case POLY_OP_CAST: {
      PolyUOp *gx = cast_to(ctx, g, u->src[0]->dtype);
      GRAD_ADD(u->src[0], gx);

    } break;

    case POLY_OP_TRUNC:
      /* Pinned tinygrad/mixin/gradient.py:57 returns ctx.const_like(0).
       * TRUNC is shape/dtype preserving, so the upstream-shaped zero is the
       * complete local gradient. */
      GRAD_ADD(u->src[0], const_like(ctx, g, 0.0));
      break;

    case POLY_OP_NEG: {
      PolyUOp *gx = poly_uop1(ctx, POLY_OP_NEG, g->dtype, g, poly_arg_none());
      GRAD_ADD(u->src[0], gx);

    } break;

    case POLY_OP_ADD:
    case POLY_OP_SUB: {
      GRAD_ADD(u->src[0], g);

      PolyUOp *gb =
          (u->op == POLY_OP_SUB) ? poly_uop1(ctx, POLY_OP_NEG, g->dtype, g, poly_arg_none()) : g;
      GRAD_ADD(u->src[1], gb);

    } break;

    case POLY_OP_MUL: {
      PolyUOp *g0 = cast_to(ctx, g, u->dtype);
      PolyUOp *ga = mul_grad(ctx, g0, u->src[1], u->dtype);
      GRAD_ADD(u->src[0], ga);

      PolyUOp *gb = mul_grad(ctx, g0, u->src[0], u->dtype);
      GRAD_ADD(u->src[1], gb);

    } break;

    case POLY_OP_FDIV: {
      PolyUOp *g0 = cast_to(ctx, g, u->dtype);

      /* d(a/b)/da = g/b */
      PolyUOp *ga = poly_uop2(ctx, POLY_OP_FDIV, u->dtype, g0, u->src[1], poly_arg_none());
      GRAD_ADD(u->src[0], ga);

      /* d(a/b)/db = -(g*u)/b  where u = a/b (reuses forward result) */
      PolyUOp *gu = mul_grad(ctx, g0, u, u->dtype);
      PolyUOp *gb = poly_uop2(ctx, POLY_OP_FDIV, u->dtype, gu, u->src[1], poly_arg_none());
      gb = poly_uop1(ctx, POLY_OP_NEG, u->dtype, gb, poly_arg_none());
      GRAD_ADD(u->src[1], gb);

    } break;

    case POLY_OP_EXP2: {
      /* Current pm_gradient: ret * ctx * math.log(2), left-associated. */
      PolyUOp *g0 = cast_to(ctx, g, u->dtype);
      PolyUOp *gx = mul_grad(ctx, g0, u, u->dtype);
      PolyUOp *ln2 = ufix_like(ctx, gx, 0.69314718055994530942);
      gx = ln2 ? poly_mul(ctx, gx, ln2) : NULL;
      if (!gx) GRAD_REVERSE_FAIL();
      GRAD_ADD(u->src[0], gx);

    } break;

    case POLY_OP_LOG2: {
      /* Current pm_gradient: ctx / (ret.src[0] * math.log(2)). */
      PolyUOp *g0 = cast_to(ctx, g, u->dtype);
      PolyUOp *ln2 = ufix_like(ctx, u->src[0], 0.69314718055994530942);
      PolyUOp *denominator = ln2 ? poly_mul(ctx, u->src[0], ln2) : NULL;
      PolyUOp *gx = denominator ? poly_div(ctx, g0, denominator) : NULL;
      if (!gx) GRAD_REVERSE_FAIL();
      GRAD_ADD(u->src[0], gx);

    } break;

    case POLY_OP_SQRT: {
      /* Current pm_gradient: ctx / (ret * 2). */
      PolyUOp *g0 = cast_to(ctx, g, u->dtype);
      PolyUOp *two = ufix_like(ctx, u, 2.0);
      PolyUOp *denominator = two ? poly_mul(ctx, u, two) : NULL;
      PolyUOp *gx = denominator ? poly_div(ctx, g0, denominator) : NULL;
      if (!gx) GRAD_REVERSE_FAIL();
      GRAD_ADD(u->src[0], gx);

    } break;

    case POLY_OP_RECIPROCAL: {
      /* Current pm_gradient constructs (-ctx * ret) * ret,
       * left-associated; ElementwiseMixin.neg is ctx * weak(-1). */
      PolyUOp *g0 = cast_to(ctx, g, u->dtype);
      PolyUOp *neg_one = ufix_like(ctx, g0, -1.0);
      PolyUOp *gx = neg_one ? poly_mul(ctx, g0, neg_one) : NULL;
      gx = gx ? poly_mul(ctx, gx, u) : NULL;
      gx = gx ? poly_mul(ctx, gx, u) : NULL;
      if (!gx) GRAD_REVERSE_FAIL();
      GRAD_ADD(u->src[0], gx);

    } break;

    case POLY_OP_SIN: {
      /* d/dx sin(x) = cos(x) = sin(π/2 - x)
       * tinygrad: (math.pi/2 - ret.src[0]).sin() * ctx */
      PolyUOp *g0 = cast_to(ctx, g, u->dtype);
      PolyUOp *half_pi = ufix_like(ctx, u->src[0], 1.5707963267948966);
      PolyUOp *shifted = half_pi ? poly_sub(ctx, half_pi, u->src[0]) : NULL;
      PolyUOp *cos_x =
          shifted ? poly_uop1(ctx, POLY_OP_SIN, u->dtype, shifted, poly_arg_none()) : NULL;
      PolyUOp *gx = cos_x ? poly_mul(ctx, cos_x, g0) : NULL;
      if (!gx) GRAD_REVERSE_FAIL();
      GRAD_ADD(u->src[0], gx);

    } break;

    case POLY_OP_MAX: {
      /* d/dx max(x, y): gradient goes to x when x>y, to y when x<y,
       * split 50/50 when x==y.
       * tinygrad: (x>y).where(ctx, (x.eq(y)).where(ctx*0.5, 0)) */
      PolyDType bdt = POLY_BOOL;
      PolyUOp *g0 = cast_to(ctx, g, u->dtype);
      PolyUOp *half = ufix_like(ctx, g0, 0.5);
      PolyUOp *zero = ufix_like(ctx, g0, 0.0);
      PolyUOp *half_g = half ? poly_mul(ctx, g0, half) : NULL;
      if (!half_g || !zero) GRAD_REVERSE_FAIL();

      /* x > y is CMPLT(y, x) */
      PolyUOp *x_gt_y = poly_uop2(ctx, POLY_OP_CMPLT, bdt, u->src[1], u->src[0], poly_arg_none());
      /* x < y is CMPLT(x, y) */
      PolyUOp *x_lt_y = poly_uop2(ctx, POLY_OP_CMPLT, bdt, u->src[0], u->src[1], poly_arg_none());
      /* x == y via double negation: eq = !CMPNE(x, y) */
      PolyUOp *neq = poly_uop2(ctx, POLY_OP_CMPNE, bdt, u->src[0], u->src[1], poly_arg_none());
      PolyUOp *true_c = poly_uop0(ctx, POLY_OP_CONST, bdt, poly_arg_bool(true));
      PolyUOp *eq_mask = poly_uop2(ctx, POLY_OP_CMPNE, bdt, neq, true_c, poly_arg_none());

      PolyUOp *eq_part = poly_where_op(ctx, eq_mask, half_g, zero);
      /* ga = WHERE(x>y, g, eq_part) */
      PolyUOp *ga = poly_where_op(ctx, x_gt_y, g0, eq_part);
      /* gb = WHERE(x<y, g, eq_part) */
      PolyUOp *gb = poly_where_op(ctx, x_lt_y, g0, eq_part);
      if (!eq_part || !ga || !gb) GRAD_REVERSE_FAIL();

      GRAD_ADD(u->src[0], ga);
      GRAD_ADD(u->src[1], gb);

    } break;

    case POLY_OP_POW: {
      /* Current gradient.py:60-61:
       *   ctx * e.eq(0).where(e, e*b.pow(e-1))
       *   ctx * b.eq(0).where((e<0).where(ret.const_like(-inf), 0),
       *                       ret*b.log2()*log(2))
       * Python integer/float literals enter through ufix as weak scalar
       * CONSTs, while ret.const_like(-inf) is deliberately shaped. */
      PolyUOp *g0 = cast_to(ctx, g, u->dtype);
      PolyUOp *b = u->src[0], *e = u->src[1];
      PolyDType dt = u->dtype;

      PolyUOp *zero = poly_const_typed(ctx, POLY_WEAKINT, 0.0);
      PolyUOp *b_eq_0 = zero ? poly_eq(ctx, b, zero) : NULL;
      PolyUOp *e_eq_0 = zero ? poly_eq(ctx, e, zero) : NULL;

      PolyUOp *one = poly_const_typed(ctx, POLY_WEAKINT, 1.0);
      PolyUOp *em1 = one ? poly_sub(ctx, e, one) : NULL;
      PolyUOp *bpem1 = em1 ? pow_grad(ctx, b, em1) : NULL;
      PolyUOp *normal_db = bpem1 ? poly_mul(ctx, e, bpem1) : NULL;
      PolyUOp *db = e_eq_0 && normal_db ? poly_where_op(ctx, e_eq_0, e, normal_db) : NULL;
      PolyUOp *ga = db ? poly_mul(ctx, g0, db) : NULL;
      if (!ga) GRAD_REVERSE_FAIL();
      GRAD_ADD(b, ga);

      PolyUOp *log2_b = poly_uop1(ctx, POLY_OP_LOG2, dt, b, poly_arg_none());
      PolyUOp *ret_log2_b = log2_b ? poly_mul(ctx, u, log2_b) : NULL;
      PolyUOp *ln2 =
          ret_log2_b ? poly_const_typed(ctx, POLY_WEAKFLOAT, 0.69314718055994530942) : NULL;
      PolyUOp *normal_de = ln2 ? poly_mul(ctx, ret_log2_b, ln2) : NULL;
      PolyUOp *e_cmp = e, *lt_zero = zero;
      PolyUOp *e_lt_0 = zero && poly_broadcasted_pair(ctx, &e_cmp, &lt_zero)
                            ? poly_alu2(ctx, POLY_OP_CMPLT, e_cmp, lt_zero)
                            : NULL;
      PolyUOp *neg_inf = const_like(ctx, u, -1.0 / 0.0);
      PolyUOp *b_zero_case =
          e_lt_0 && neg_inf && zero ? poly_where_op(ctx, e_lt_0, neg_inf, zero) : NULL;
      PolyUOp *de = b_eq_0 && b_zero_case && normal_de
                        ? poly_where_op(ctx, b_eq_0, b_zero_case, normal_de)
                        : NULL;
      PolyUOp *gb = de ? poly_mul(ctx, g0, de) : NULL;
      if (!gb) GRAD_REVERSE_FAIL();
      GRAD_ADD(e, gb);

    } break;

    case POLY_OP_WHERE: {
      PolyUOp *g0 = cast_to(ctx, g, u->dtype);
      /* Pinned tinygrad/mixin/gradient.py:65 uses ctx.const_like(0) for
       * both branches, preserving one shared shaped zero. */
      PolyUOp *zero = const_like(ctx, g0, 0.0);
      PolyUOp *gt = poly_uop3(ctx, POLY_OP_WHERE, u->dtype, u->src[0], g0, zero, poly_arg_none());
      PolyUOp *gf = poly_uop3(ctx, POLY_OP_WHERE, u->dtype, u->src[0], zero, g0, poly_arg_none());
      GRAD_ADD(u->src[1], gt);
      GRAD_ADD(u->src[2], gf);

    } break;

    case POLY_OP_STACK: {
      /* Pinned mixin/gradient.py:76 takes ctx[i] for each ordered source.
       * A value index is SHRINK followed by dropping its singleton axis. */
      int ndim = poly_uop_ndim(ctx, g);
      if (ndim < 1 || ndim > POLY_MAX_DIMS) GRAD_REVERSE_FAIL();
      PolyUOp *starts[POLY_MAX_DIMS], *sizes[POLY_MAX_DIMS];
      PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
      for (int d = 0; d < ndim; d++) {
        starts[d] = zero;
        sizes[d] = poly_uop_shape_dim(ctx, g, d);
        if (!sizes[d]) GRAD_REVERSE_FAIL();
      }
      sizes[0] = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
      for (int i = 0; i < u->n_src; i++) {
        starts[0] = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(i));
        PolyUOp *slice = poly_shrink_uop(ctx, g, starts, sizes, ndim);
        PolyUOp *gx = slice ? poly_reshape_uop(ctx, slice, sizes + 1, ndim - 1) : NULL;
        GRAD_ADD(u->src[i], gx);
      }
    } break;

    case POLY_OP_RESHAPE: {
      PolyShape s0 = poly_uop_max_shape_cached(ctx, u->src[0]);
      PolyUOp *gx = poly_reshape(ctx, g, s0.dims, s0.ndim);
      GRAD_ADD(u->src[0], gx);

    } break;

    case POLY_OP_EXPAND:
      GRAD_ADD(u->src[0], g);
      break;

    case POLY_OP_PERMUTE: {
      if (u->arg.kind != POLY_ARG_INT_TUPLE) {
        fprintf(stderr, "polygrad: autograd: PERMUTE missing int tuple arg\n");
        GRAD_REVERSE_FAIL();
      }
      int n = u->arg.int_tuple.n;
      if (n < 0 || n > POLY_MAX_DIMS) {
        fprintf(stderr, "polygrad: autograd: PERMUTE rank %d out of bounds\n", n);
        GRAD_REVERSE_FAIL();
      }
      int64_t inv[POLY_MAX_DIMS];
      for (int i = 0; i < n; i++)
        inv[i] = i;
      for (int i = 0; i < n; i++)
        inv[u->arg.int_tuple.vals[i]] = i;
      PolyUOp *gx = poly_permute(ctx, g, inv, n);
      GRAD_ADD(u->src[0], gx);
    } break;

    case POLY_OP_PAD: {
      PolyShape in_shape = poly_uop_max_shape_cached(ctx, u->src[0]);
      int n = in_shape.ndim;
      if (n < 0 || n > POLY_MAX_DIMS) {
        fprintf(stderr, "polygrad: autograd: PAD rank %d out of bounds\n", n);
        GRAD_REVERSE_FAIL();
      }
      PolyUOp *gx = NULL;
      if (u->arg.kind == POLY_ARG_NONE && u->n_src == 3) {
        PolyUOp *offsets[POLY_MAX_DIMS], *sizes[POLY_MAX_DIMS];
        if (poly_uop_as_shape(ctx, u->src[1], offsets, POLY_MAX_DIMS) != n) {
          fprintf(stderr, "polygrad: autograd: PAD offset rank mismatch\n");
          GRAD_REVERSE_FAIL();
        }
        for (int i = 0; i < n; i++) {
          sizes[i] = shape_dim_node(ctx, u->src[0], in_shape, i);
          if (!sizes[i]) GRAD_REVERSE_FAIL();
        }
        gx = poly_shrink_uop(ctx, g, offsets, sizes, n);
      } else {
        fprintf(stderr, "polygrad: autograd: PAD unsupported form\n");
        GRAD_REVERSE_FAIL();
      }
      GRAD_ADD(u->src[0], gx);

    } break;

    case POLY_OP_SHRINK: {
      PolyShape in_shape = poly_uop_max_shape_cached(ctx, u->src[0]);
      int n = in_shape.ndim;
      if (n < 0 || n > POLY_MAX_DIMS) {
        fprintf(stderr, "polygrad: autograd: SHRINK rank %d out of bounds\n", n);
        GRAD_REVERSE_FAIL();
      }
      PolyUOp *gx = NULL;
      if (u->arg.kind == POLY_ARG_NONE && u->n_src >= 3) {
        PolyUOp *starts[POLY_MAX_DIMS];
        if (poly_uop_as_shape(ctx, u->src[1], starts, POLY_MAX_DIMS) != n) {
          fprintf(stderr, "polygrad: autograd: SHRINK start rank mismatch\n");
          GRAD_REVERSE_FAIL();
        }
        PolyUOp *sizes[POLY_MAX_DIMS];
        for (int i = 0; i < n; i++) {
          sizes[i] = shape_dim_node(ctx, u->src[0], in_shape, i);
          if (!sizes[i]) GRAD_REVERSE_FAIL();
        }
        gx = poly_pad_uop(ctx, g, starts, sizes, n);
      } else {
        fprintf(stderr, "polygrad: autograd: SHRINK unsupported form\n");
        GRAD_REVERSE_FAIL();
      }
      GRAD_ADD(u->src[0], gx);

    } break;

    case POLY_OP_FLIP: {
      if (u->arg.kind != POLY_ARG_INT_TUPLE) {
        fprintf(stderr, "polygrad: autograd: FLIP missing int tuple arg\n");
        GRAD_REVERSE_FAIL();
      }
      PolyUOp *gx = poly_uop1(ctx, POLY_OP_FLIP, g->dtype, g, u->arg);
      GRAD_ADD(u->src[0], gx);
    } break;

    case POLY_OP_REDUCE: {
      if (u->arg.kind != POLY_ARG_REDUCE) {
        fprintf(stderr, "polygrad: autograd: tensor REDUCE missing current reduce arg\n");
        GRAD_REVERSE_FAIL();
      }
      PolyOps reduce_op = u->arg.reduce.op;

      if (reduce_op == POLY_OP_ADD) {
        /* Current reduce_gradient: ctx._broadcast_to(ret.src[0].shape). */
        PolyUOp *gx = broadcast_to_uop_shape(ctx, g, u->src[0]);
        if (!gx) GRAD_REVERSE_FAIL();
        GRAD_ADD(u->src[0], gx);
      } else if (reduce_op == POLY_OP_MAX) {
        /* Current reduce_gradient uses ordinary implicit broadcasting around
         * the same prefix reduction (mixin/gradient.py:7-9). */
        int n_axes = u->arg.reduce.num_axes;
        int64_t axes[POLY_MAX_DIMS];
        for (int i = 0; i < n_axes; i++)
          axes[i] = i;
        PolyUOp *mask = poly_eq(ctx, u->src[0], u);
        PolyUOp *fmask = mask ? cast_to(ctx, mask, g->dtype) : NULL;
        PolyUOp *count = fmask ? poly_reduce_axis(ctx, POLY_OP_ADD, fmask, axes, n_axes) : NULL;
        PolyUOp *reciprocal =
            count ? poly_uop1(ctx, POLY_OP_RECIPROCAL, count->dtype, count, poly_arg_none()) : NULL;
        PolyUOp *scaled_mask =
            fmask && reciprocal
                ? poly_uop2(ctx, POLY_OP_MUL, g->dtype, fmask, reciprocal, poly_arg_none())
                : NULL;
        PolyUOp *gx = scaled_mask
                          ? poly_uop2(ctx, POLY_OP_MUL, g->dtype, scaled_mask, g, poly_arg_none())
                          : NULL;
        if (!gx) GRAD_REVERSE_FAIL();
        GRAD_ADD(u->src[0], gx);
      } else if (reduce_op == POLY_OP_MUL) {
        /* Pinned reduce_gradient (mixin/gradient.py:10-14): divide by a
         * nonzero replacement, and use the product of other inputs only
         * when this is the sole zero. This also keeps zero gradients finite. */
        int n_axes = u->arg.reduce.num_axes;
        int64_t axes[POLY_MAX_DIMS];
        for (int i = 0; i < n_axes; i++)
          axes[i] = i;
        PolyUOp *x = u->src[0];
        PolyUOp *is_zero = poly_eq(ctx, x, ufix_like(ctx, x, 0));
        PolyUOp *safe_x = is_zero ? poly_where_op(ctx, is_zero, ufix_like(ctx, x, 1), x) : NULL;
        PolyDType count_dtype;
        if (!safe_x || !poly_sum_acc_dtype(POLY_BOOL, &count_dtype)) GRAD_REVERSE_FAIL();
        PolyUOp *count = cast_to(ctx, is_zero, count_dtype);
        count = count ? poly_reduce_axis(ctx, POLY_OP_ADD, count, axes, n_axes) : NULL;
        PolyUOp *single_zero = count ? poly_eq(ctx, count, ufix_like(ctx, count, 1)) : NULL;
        PolyUOp *others = poly_reduce_axis(ctx, POLY_OP_MUL, safe_x, axes, n_axes);
        PolyUOp *at_zero = single_zero && others
                               ? poly_where_op(ctx, single_zero, others, ufix_like(ctx, others, 0))
                               : NULL;
        PolyUOp *nonzero = poly_div(ctx, u, safe_x);
        PolyUOp *local = at_zero && nonzero ? poly_where_op(ctx, is_zero, at_zero, nonzero) : NULL;
        PolyUOp *gx = local ? poly_mul(ctx, g, local) : NULL;
        if (!gx) GRAD_REVERSE_FAIL();
        GRAD_ADD(x, gx);
      } else {
        fprintf(
            stderr, "polygrad: autograd: unsupported tensor REDUCE op: %s\n",
            poly_op_name(reduce_op)
        );

        GRAD_REVERSE_FAIL();
      }

    } break;

    default:
      /* With target pruning, any op on the gradient path was selected
       * because it lies on a path to a target. Silent skip hides bugs.
       * Fail hard so callers get NULL and can report the error. */
      fprintf(stderr, "polygrad: autograd: missing gradient rule for %s\n", poly_op_name(u->op));
      GRAD_REVERSE_FAIL();
    }
  }

  if (walk_owned) free(walk);
  poly_ctx_scratch_rewind(ctx, scratch);
#undef GRAD_ADD
#undef GRAD_REVERSE_FAIL
  return grads;
}

PolyUOp *poly_grad(PolyCtx *ctx, PolyUOp *loss, PolyUOp *wrt) {
  if (!ctx || !loss || !wrt) return NULL;

  PolyMap *grads = grad_reverse_pass(ctx, loss, ones_like(ctx, loss), &wrt, 1);
  if (!grads) return NULL;

  PolyUOp *out = grad_get(grads, wrt);
  if (!out) out = zeros_like(ctx, wrt);

  poly_map_destroy(grads);
  return out;
}

int poly_grad_many_ex(
    PolyCtx *ctx,
    PolyUOp *loss,
    PolyUOp *initial_grad,
    PolyUOp **wrts,
    int n,
    PolyUOp **out_grads,
    uint8_t *out_present
) {
  if (!ctx || !loss || !wrts || !out_grads || n <= 0) return -1;

  PolyMap *grads =
      grad_reverse_pass(ctx, loss, initial_grad ? initial_grad : ones_like(ctx, loss), wrts, n);
  if (!grads) return -1;

  for (int i = 0; i < n; i++) {
    PolyUOp *g = grad_get(grads, wrts[i]);
    if (out_present) out_present[i] = g ? 1 : 0;
    if (!g) g = zeros_like(ctx, wrts[i]);
    out_grads[i] = g;
  }

  poly_map_destroy(grads);
  return 0;
}

int poly_grad_many(
    PolyCtx *ctx,
    PolyUOp *loss,
    PolyUOp *initial_grad,
    PolyUOp **wrts,
    int n,
    PolyUOp **out_grads
) {
  return poly_grad_many_ex(ctx, loss, initial_grad, wrts, n, out_grads, NULL);
}

/* UOp graph substitution */

typedef struct {
  PolyUOp *u;
  PolyUOp *link;
  int stage;
} SubFrame;

static bool sub_stack_push(SubFrame **stack, int *n, int *cap, SubFrame f) {
  if (*n >= *cap) {
    int new_cap = (*cap > 0) ? (*cap * 2) : 256;
    SubFrame *new_stack = realloc(*stack, (size_t)new_cap * sizeof(SubFrame));
    if (!new_stack) return false;
    *stack = new_stack;
    *cap = new_cap;
  }
  (*stack)[(*n)++] = f;
  return true;
}

static bool substitute_opaque_body_op(PolyOps op) {
  return op == POLY_OP_CALL || op == POLY_OP_FUNCTION;
}

/* Discover only the identity pins that pinned RewriteContext will eventually
 * install while traversing these roots in order. This dry traversal follows
 * substitutions and shared-memo visibility but creates no UOps. Preinstalling
 * the final pins prevents an earlier standalone body root from allocating a
 * rebuilt UOp that a later CALL/FUNCTION owner overwrites with body->body.
 *
 * This is deliberately ordered rather than a whole-graph opaque-body scan: if
 * an outer owner pins its body before that body is exposed as a later root,
 * nested owners inside it are not visited, matching the pinned probe. */
static bool substitute_collect_opaque_pins(
    PolyUOp **roots,
    int n_roots,
    PolyMap *sub_map,
    PolyUOp ***out_pins,
    int *out_n_pins
) {
  if (!roots || n_roots < 0 || !sub_map || !out_pins || !out_n_pins) return false;
  *out_pins = NULL;
  *out_n_pins = 0;

  PolyMap *done = poly_map_new(256);
  PolyMap *visiting = poly_map_new(256);
  PolyMap *pin_seen = poly_map_new(64);
  SubFrame *stack = NULL;
  PolyUOp **pins = NULL;
  int n_stack = 0, cap_stack = 0, n_pins = 0, pins_cap = 0;
  bool ok = done && visiting && pin_seen;

  for (int root_idx = 0; ok && root_idx < n_roots; root_idx++) {
    PolyUOp *root = roots[root_idx];
    if (!root) {
      ok = false;
      break;
    }
    if (poly_map_get(done, poly_ptr_hash(root), root, poly_ptr_eq)) continue;
    ok = sub_stack_push(&stack, &n_stack, &cap_stack, (SubFrame){root, NULL, 0});

    while (ok && n_stack > 0) {
      SubFrame f = stack[--n_stack];
      PolyUOp *u = f.u;
      if (!u) {
        ok = false;
        break;
      }
      uint32_t h = poly_ptr_hash(u);

      if (f.stage == 0) {
        if (poly_map_get(done, h, u, poly_ptr_eq)) continue;
        if (poly_map_get(visiting, h, u, poly_ptr_eq)) {
          ok = false;
          break;
        }
        poly_map_set(visiting, h, u, (void *)(uintptr_t)1, poly_ptr_eq);

        PolyUOp *sub = poly_map_get(sub_map, h, u, poly_ptr_eq);
        if (sub) {
          ok = sub_stack_push(&stack, &n_stack, &cap_stack, (SubFrame){u, sub, 2}) &&
               sub_stack_push(&stack, &n_stack, &cap_stack, (SubFrame){sub, NULL, 0});
          continue;
        }

        int first_src = 0;
        if (substitute_opaque_body_op(u->op) && u->n_src > 0) {
          PolyUOp *body = u->src[0];
          if (!body || (poly_map_get(visiting, poly_ptr_hash(body), body, poly_ptr_eq) &&
                        !poly_map_get(done, poly_ptr_hash(body), body, poly_ptr_eq))) {
            ok = false;
            break;
          }
          if (!poly_map_get(pin_seen, poly_ptr_hash(body), body, poly_ptr_eq)) {
            if (n_pins >= pins_cap) {
              int new_cap = pins_cap ? pins_cap * 2 : 16;
              PolyUOp **new_pins = realloc(pins, (size_t)new_cap * sizeof(*new_pins));
              if (!new_pins) {
                ok = false;
                break;
              }
              pins = new_pins;
              pins_cap = new_cap;
            }
            pins[n_pins++] = body;
            poly_map_set(pin_seen, poly_ptr_hash(body), body, (void *)(uintptr_t)1, poly_ptr_eq);
          }
          /* A later owner overwrites any earlier rewritten result for body. */
          poly_map_set(done, poly_ptr_hash(body), body, (void *)(uintptr_t)1, poly_ptr_eq);
          first_src = 1;
        }

        ok = sub_stack_push(&stack, &n_stack, &cap_stack, (SubFrame){u, NULL, 1});
        for (int i = u->n_src - 1; ok && i >= first_src; i--) {
          PolyUOp *src = u->src[i];
          if (!src || poly_map_get(done, poly_ptr_hash(src), src, poly_ptr_eq)) continue;
          ok = sub_stack_push(&stack, &n_stack, &cap_stack, (SubFrame){src, NULL, 0});
        }
        continue;
      }

      if (f.stage == 2 &&
          (!f.link || !poly_map_get(done, poly_ptr_hash(f.link), f.link, poly_ptr_eq))) {
        ok = false;
        break;
      }
      poly_map_set(done, h, u, (void *)(uintptr_t)1, poly_ptr_eq);
      poly_map_remove(visiting, h, u, poly_ptr_eq);
    }
  }

  poly_map_destroy(pin_seen);
  poly_map_destroy(visiting);
  poly_map_destroy(done);
  free(stack);
  if (!ok) {
    free(pins);
    return false;
  }
  *out_pins = pins;
  *out_n_pins = n_pins;
  return true;
}

static bool substitute_iter(
    PolyCtx *ctx,
    PolyUOp *root,
    PolyMap *sub_map,
    PolyMap *memo,
    PolyUOp **out
) {
  if (!ctx || !root || !sub_map || !memo || !out) return false;

  PolyMap *visiting = poly_map_new(256);
  SubFrame *stack = NULL;
  int n_stack = 0, cap_stack = 0;
  bool ok = visiting && sub_stack_push(&stack, &n_stack, &cap_stack, (SubFrame){root, NULL, 0});

  while (ok && n_stack > 0) {
    SubFrame f = stack[--n_stack];
    PolyUOp *u = f.u;
    if (!u) {
      ok = false;
      break;
    }
    uint32_t h = poly_ptr_hash(u);

    if (f.stage == 0) {
      if (poly_map_get(memo, h, u, poly_ptr_eq)) continue;
      if (poly_map_get(visiting, h, u, poly_ptr_eq)) {
        ok = false;
        break;
      }
      poly_map_set(visiting, h, u, (void *)(uintptr_t)1, poly_ptr_eq);

      PolyUOp *sub = poly_map_get(sub_map, h, u, poly_ptr_eq);
      if (sub) {
        PolyUOp *cached = poly_map_get(memo, poly_ptr_hash(sub), sub, poly_ptr_eq);
        if (cached) {
          poly_map_set(memo, h, u, cached, poly_ptr_eq);
          poly_map_remove(visiting, h, u, poly_ptr_eq);
          continue;
        }
        ok = sub_stack_push(&stack, &n_stack, &cap_stack, (SubFrame){u, sub, 2}) &&
             sub_stack_push(&stack, &n_stack, &cap_stack, (SubFrame){sub, NULL, 0});
        continue;
      }

      ok = sub_stack_push(&stack, &n_stack, &cap_stack, (SubFrame){u, NULL, 1});
      int first_src = 0;
      if (substitute_opaque_body_op(u->op) && u->n_src > 0) {
        /* Pinned graph_rewrite keeps a CALL/FUNCTION's code body opaque unless
         * enter_calls is explicitly enabled. Scope discovery still enters the
         * body, but default substitution preserves source zero exactly and
         * rewrites only the external arguments. */
        PolyUOp *body = u->src[0];
        if (!body) {
          ok = false;
          break;
        }
        poly_map_set(memo, poly_ptr_hash(body), body, body, poly_ptr_eq);
        first_src = 1;
      }
      for (int i = u->n_src - 1; ok && i >= first_src; i--) {
        PolyUOp *src = u->src[i];
        if (!src || poly_map_get(memo, poly_ptr_hash(src), src, poly_ptr_eq)) continue;
        ok = sub_stack_push(&stack, &n_stack, &cap_stack, (SubFrame){src, NULL, 0});
      }
      continue;
    }

    if (f.stage == 2) {
      PolyUOp *mapped =
          f.link ? poly_map_get(memo, poly_ptr_hash(f.link), f.link, poly_ptr_eq) : NULL;
      if (!mapped) {
        ok = false;
        break;
      }
      poly_map_set(memo, h, u, mapped, poly_ptr_eq);
      poly_map_remove(visiting, h, u, poly_ptr_eq);
      continue;
    }

    PolyUOp *src_buf[POLY_MAX_DIMS + 2];
    PolyUOp **new_srcs = src_buf;
    if ((size_t)u->n_src > sizeof(src_buf) / sizeof(src_buf[0])) {
      new_srcs = malloc((size_t)u->n_src * sizeof(PolyUOp *));
      if (!new_srcs) {
        ok = false;
        break;
      }
    }

    bool changed = false;
    for (int i = 0; i < u->n_src; i++) {
      PolyUOp *src = u->src[i];
      PolyUOp *mapped = src ? poly_map_get(memo, poly_ptr_hash(src), src, poly_ptr_eq) : NULL;
      if (!mapped) {
        if (new_srcs != src_buf) free(new_srcs);
        ok = false;
        break;
      }
      new_srcs[i] = mapped;
      if (mapped != src) changed = true;
    }
    if (!ok) break;

    PolyUOp *result = u;
    if (changed) {
      result = (u->tag != 0 || u->tag_arg.kind != POLY_ARG_NONE)
                   ? poly_uop_tagged_arg(
                         ctx, u->op, u->dtype, new_srcs, u->n_src, u->arg, u->tag, u->tag_arg
                     )
                   : poly_uop(ctx, u->op, u->dtype, new_srcs, u->n_src, u->arg);
    }
    if (new_srcs != src_buf) free(new_srcs);
    poly_map_set(memo, h, u, result, poly_ptr_eq);
    poly_map_remove(visiting, h, u, poly_ptr_eq);
  }

  if (ok) {
    *out = poly_map_get(memo, poly_ptr_hash(root), root, poly_ptr_eq);
    if (!*out) ok = false;
  }
  if (visiting) poly_map_destroy(visiting);
  free(stack);
  return ok;
}

int poly_uop_substitute_many(
    PolyCtx *ctx,
    PolyUOp **roots,
    int n_roots,
    PolyUOp **from,
    PolyUOp **to,
    int n,
    PolyUOp **out
) {
  if (!ctx || n_roots < 0 || n < 0 || (n_roots > 0 && (!roots || !out)) ||
      (n > 0 && (!from || !to)))
    return -1;
  if (n_roots == 0) return 0;
  if (n == 0) {
    for (int i = 0; i < n_roots; i++) {
      if (!roots[i]) return -1;
      out[i] = roots[i];
    }
    return 0;
  }

  PolyMap *sub_map = poly_map_new((size_t)n * 2 + 16);
  PolyMap *memo = poly_map_new(256);
  if (!sub_map || !memo) {
    if (sub_map) poly_map_destroy(sub_map);
    if (memo) poly_map_destroy(memo);
    return -1;
  }

  for (int i = 0; i < n; i++) {
    if (!from[i] || !to[i] || from[i] == to[i]) continue;
    poly_map_set(sub_map, poly_ptr_hash(from[i]), from[i], to[i], poly_ptr_eq);
  }

  int rc = 0;
  if (poly_map_len(sub_map) == 0) {
    for (int i = 0; i < n_roots; i++) {
      if (!roots[i]) {
        rc = -1;
        break;
      }
      out[i] = roots[i];
    }
  } else {
    PolyUOp **opaque_pins = NULL;
    int n_opaque_pins = 0;
    if (!substitute_collect_opaque_pins(roots, n_roots, sub_map, &opaque_pins, &n_opaque_pins)) {
      rc = -1;
    }
    for (int i = 0; rc == 0 && i < n_opaque_pins; i++) {
      PolyUOp *body = opaque_pins[i];
      poly_map_set(memo, poly_ptr_hash(body), body, body, poly_ptr_eq);
    }
    free(opaque_pins);

    for (int i = 0; rc == 0 && i < n_roots; i++) {
      PolyUOp *root = roots[i];
      if (!root) {
        rc = -1;
        break;
      }
      PolyUOp *rewritten = poly_map_get(memo, poly_ptr_hash(root), root, poly_ptr_eq);
      if (!rewritten && (!substitute_iter(ctx, root, sub_map, memo, &rewritten) || !rewritten)) {
        rc = -1;
        break;
      }
    }
    /* Pinned UOp.sink(*roots).substitute(...) owns one shared RewriteContext
     * and reads every SINK source only after the complete ordered traversal.
     * A later CALL/FUNCTION can therefore identity-pin a body that was also an
     * earlier root. Read outputs from that final shared memo. */
    for (int i = 0; rc == 0 && i < n_roots; i++) {
      out[i] = poly_map_get(memo, poly_ptr_hash(roots[i]), roots[i], poly_ptr_eq);
      if (!out[i]) rc = -1;
    }
  }

  poly_map_destroy(sub_map);
  poly_map_destroy(memo);
  return rc;
}

PolyUOp *poly_uop_substitute(PolyCtx *ctx, PolyUOp *root, PolyUOp **from, PolyUOp **to, int n) {
  if (!ctx || !root || n <= 0) return root;
  PolyUOp *result = root;
  if (poly_uop_substitute_many(ctx, &root, 1, from, to, n, &result) != 0) return root;
  return result;
}
