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
 * - Reductions: REDUCE_AXIS with ADD and MAX (CONTIGUOUS barrier for MAX)
 * - Utility: CAST pass-through, CONTIGUOUS/COPY/BUFFERIZE pass-through
 * - Stop gradient: DETACH, CMPLT, CMPNE, BITCAST
 * - Target-pruned reverse pass (port of tinygrad's _deepwalk)
 */

#include "sched.h"
#include "pat.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Local helpers ────────────────────────────────────────────────────── */

static bool ptr_eq(const void *a, const void *b) { return a == b; }

static uint32_t ptr_hash(const void *p) {
  uintptr_t v = (uintptr_t)p;
  return (uint32_t)(v ^ (v >> 16) ^ (sizeof(v) > 4 ? (uint32_t)(v >> 32) : 0));
}

static void shape_free(PolyShape s) {
  if (s.ndim > 0 && s.dims) free(s.dims);
}

static PolyUOp *const_scalar(PolyCtx *ctx, PolyDType dt, double v) {
  if (poly_dtype_is_float(dt)) return poly_uop0(ctx, POLY_OP_CONST, dt, poly_arg_float(v));
  if (poly_dtype_is_bool(dt)) return poly_uop0(ctx, POLY_OP_CONST, dt, poly_arg_bool(v != 0.0));
  return poly_uop0(ctx, POLY_OP_CONST, dt, poly_arg_int((int64_t)v));
}

static PolyUOp *cast_to(PolyCtx *ctx, PolyUOp *u, PolyDType dt) {
  if (poly_dtype_eq(u->dtype, dt)) return u;
  return poly_uop1(ctx, POLY_OP_CAST, dt, u, poly_arg_none());
}

static PolyUOp *grad_get(PolyMap *grads, PolyUOp *u) {
  return poly_map_get(grads, ptr_hash(u), u, ptr_eq);
}

static void grad_add(PolyCtx *ctx, PolyMap *grads, PolyUOp *u, PolyUOp *g) {
  if (!u || !g) return;
  PolyUOp *old = grad_get(grads, u);
  if (!old) {
    poly_map_set(grads, ptr_hash(u), u, g, ptr_eq);
    return;
  }
  PolyUOp *rhs = cast_to(ctx, g, old->dtype);
  PolyUOp *sum = poly_uop2(ctx, POLY_OP_ADD, old->dtype, old, rhs, poly_arg_none());
  poly_map_set(grads, ptr_hash(u), u, sum, ptr_eq);
}

static PolyUOp *reduce_sum_axes(PolyCtx *ctx, PolyUOp *u, int64_t *axes, int n_axes) {
  if (n_axes <= 0) return u;
  return poly_reduce_axis(ctx, POLY_OP_ADD, u, axes, n_axes);
}

/* Match a gradient tensor to target shape by reducing broadcasted axes and/or
 * expanding singleton axes as needed. */
static PolyUOp *reduce_to_shape(PolyCtx *ctx, PolyUOp *grad, PolyShape target) {
  if (target.ndim < 0) return grad;

  PolyShape gshape = poly_uop_shape(ctx, grad);
  if (gshape.ndim < 0) {
    shape_free(gshape);
    return grad;
  }

  if (target.ndim == 0) {
    if (gshape.ndim > 0) {
      int64_t axes[POLY_MAX_DIMS];
      int n_axes = 0;
      for (int i = 0; i < gshape.ndim && i < POLY_MAX_DIMS; i++) axes[n_axes++] = i;
      grad = reduce_sum_axes(ctx, grad, axes, n_axes);
      grad = poly_reshape(ctx, grad, NULL, 0);
    }
    shape_free(gshape);
    return grad;
  }

  if (gshape.ndim == 0) {
    grad = poly_expand(ctx, grad, target.dims, target.ndim);
    shape_free(gshape);
    return grad;
  }

  if (gshape.ndim > target.ndim) {
    int lead = gshape.ndim - target.ndim;
    int64_t axes[POLY_MAX_DIMS];
    int n_axes = 0;
    for (int i = 0; i < lead && i < POLY_MAX_DIMS; i++) axes[n_axes++] = i;
    grad = reduce_sum_axes(ctx, grad, axes, n_axes);
    shape_free(gshape);
    gshape = poly_uop_shape(ctx, grad);
    if (gshape.ndim < 0) return grad;
  } else if (gshape.ndim < target.ndim) {
    grad = poly_expand(ctx, grad, target.dims, target.ndim);
    shape_free(gshape);
    return grad;
  }

  int64_t axes[POLY_MAX_DIMS];
  int n_axes = 0;
  for (int i = 0; i < target.ndim && i < gshape.ndim && i < POLY_MAX_DIMS; i++) {
    if (target.dims[i] == 1 && gshape.dims[i] != 1) axes[n_axes++] = i;
  }
  grad = reduce_sum_axes(ctx, grad, axes, n_axes);

  shape_free(gshape);
  gshape = poly_uop_shape(ctx, grad);
  if (gshape.ndim == 0) {
    grad = poly_expand(ctx, grad, target.dims, target.ndim);
  } else if (gshape.ndim == target.ndim) {
    bool need_expand = false;
    for (int i = 0; i < target.ndim; i++) {
      if (gshape.dims[i] == target.dims[i]) continue;
      if (gshape.dims[i] == 1 && target.dims[i] > 1) {
        need_expand = true;
      } else {
        need_expand = false;
        break;
      }
    }
    if (need_expand) grad = poly_expand(ctx, grad, target.dims, target.ndim);
  }

  shape_free(gshape);
  return grad;
}

static PolyUOp *ones_like(PolyCtx *ctx, PolyUOp *u) {
  PolyUOp *one = const_scalar(ctx, u->dtype, 1.0);
  PolyShape s = poly_uop_shape(ctx, u);
  if (s.ndim > 0) one = poly_expand(ctx, one, s.dims, s.ndim);
  shape_free(s);
  return one;
}

static PolyUOp *zeros_like(PolyCtx *ctx, PolyUOp *u) {
  PolyUOp *zero = const_scalar(ctx, u->dtype, 0.0);
  PolyShape s = poly_uop_shape(ctx, u);
  if (s.ndim > 0) zero = poly_expand(ctx, zero, s.dims, s.ndim);
  shape_free(s);
  return zero;
}

/* Check if a UOp is effectively constant 1 (possibly through EXPAND/RESHAPE) */
static bool is_const_one(PolyUOp *u) {
  while (u->op == POLY_OP_EXPAND || u->op == POLY_OP_RESHAPE)
    u = u->src[0];
  if (u->op != POLY_OP_CONST) return false;
  if (u->arg.kind == POLY_ARG_FLOAT) return u->arg.f == 1.0;
  if (u->arg.kind == POLY_ARG_INT)   return u->arg.i == 1;
  return false;
}

/* Multiply value by gradient, skipping when gradient is effectively 1.
 * This matches tinygrad's symbolic simplification of g*x when g=1. */
static PolyUOp *mul_grad(PolyCtx *ctx, PolyUOp *g, PolyUOp *x, PolyDType dt) {
  if (is_const_one(g)) return x;
  return poly_uop2(ctx, POLY_OP_MUL, dt, g, x, poly_arg_none());
}

/* ── Target-pruned walk (port of tinygrad _deepwalk) ─────────────────── */

/* Compute the subset of topo[] that lies on paths from any target to root.
 * Skips DETACH/ASSIGN nodes. Returns filtered array (arena-allocated). */
static PolyUOp **target_walk(PolyCtx *ctx, PolyUOp **topo, int n_topo,
                              PolyUOp **targets, int n_targets, int *n_out) {
  PolyMap *target_set = poly_map_new((size_t)n_targets * 2 + 16);
  for (int i = 0; i < n_targets; i++)
    poly_map_set(target_set, ptr_hash(targets[i]), targets[i], targets[i], ptr_eq);

  /* Forward pass: mark nodes whose sources lead to any target */
  PolyMap *in_path = poly_map_new((size_t)n_topo * 2 + 16);
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (u->op == POLY_OP_DETACH || u->op == POLY_OP_ASSIGN) continue;
    bool on_path = poly_map_get(target_set, ptr_hash(u), u, ptr_eq) != NULL;
    if (!on_path) {
      for (int j = 0; j < u->n_src; j++) {
        PolyUOp *s = u->src[j];
        if (poly_map_get(target_set, ptr_hash(s), s, ptr_eq) ||
            poly_map_get(in_path, ptr_hash(s), s, ptr_eq)) {
          on_path = true;
          break;
        }
      }
    }
    if (on_path)
      poly_map_set(in_path, ptr_hash(u), u, u, ptr_eq);
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
    if (poly_map_get(in_path, ptr_hash(u), u, ptr_eq))
      result[count++] = u;
  }

  poly_map_destroy(target_set);
  poly_map_destroy(in_path);
  *n_out = count;
  return result;
}

/* ── Public API ───────────────────────────────────────────────────────── */

/* Core gradient reverse pass. Builds the gradient map from loss backward.
 * When targets/n_targets are provided, only walks nodes on paths to targets
 * (port of tinygrad's _deepwalk). This prevents unsupported ops on
 * unrelated branches from crashing the backward pass.
 * Returns the gradient map on success, NULL on failure.
 * Caller must call poly_map_destroy on the returned map. */
static PolyMap *grad_reverse_pass(PolyCtx *ctx, PolyUOp *loss,
                                   PolyUOp *initial_grad,
                                   PolyUOp **targets, int n_targets) {
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, loss, &n_topo);
  if (!topo || n_topo <= 0) return NULL;

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
      return grads;
    }
  }

  PolyMap *grads = poly_map_new((size_t)n_walk * 2 + 16);
  if (!grads) return NULL;

  grad_add(ctx, grads, loss, initial_grad);

  for (int i = n_walk - 1; i >= 0; i--) {
    PolyUOp *u = walk[i];
    PolyUOp *g = grad_get(grads, u);
    if (!g) continue;

    switch (u->op) {
      /* leaf / no-parent cases */
      case POLY_OP_CONST:
      case POLY_OP_BUFFER:
      case POLY_OP_DEFINE_VAR:
      case POLY_OP_BIND:
      case POLY_OP_PARAM:
      case POLY_OP_UNIQUE:
      case POLY_OP_DEVICE:
        break;

      /* explicitly stop gradient flow */
      case POLY_OP_DETACH:
        break;

      /* non-differentiable ops: stop gradient (tinygrad returns (None,...)) */
      case POLY_OP_CMPLT:
      case POLY_OP_CMPNE:
      case POLY_OP_BITCAST:
        break;

      /* pass-through (no realize barrier on gradient) */
      case POLY_OP_CONTIGUOUS:
      case POLY_OP_COPY:
      case POLY_OP_NOOP:
      case POLY_OP_BUFFERIZE: {
        PolyShape s0 = poly_uop_shape(ctx, u->src[0]);
        PolyUOp *gx = reduce_to_shape(ctx, g, s0);
        gx = cast_to(ctx, gx, u->src[0]->dtype);
        grad_add(ctx, grads, u->src[0], gx);
        shape_free(s0);
      } break;

      /* CONTIGUOUS_BACKWARD: wrap gradient in CONTIGUOUS (realize barrier).
       * Matches tinygrad gradient.py:44: (ctx.contiguous(),) */
      case POLY_OP_CONTIGUOUS_BACKWARD: {
        PolyShape s0 = poly_uop_shape(ctx, u->src[0]);
        PolyUOp *gx = reduce_to_shape(ctx, g, s0);
        gx = cast_to(ctx, gx, u->src[0]->dtype);
        gx = poly_uop1(ctx, POLY_OP_CONTIGUOUS, gx->dtype, gx, poly_arg_none());
        grad_add(ctx, grads, u->src[0], gx);
        shape_free(s0);
      } break;

      case POLY_OP_CAST: {
        PolyShape s0 = poly_uop_shape(ctx, u->src[0]);
        PolyUOp *gx = reduce_to_shape(ctx, g, s0);
        gx = cast_to(ctx, gx, u->src[0]->dtype);
        grad_add(ctx, grads, u->src[0], gx);
        shape_free(s0);
      } break;

      case POLY_OP_NEG: {
        PolyShape s0 = poly_uop_shape(ctx, u->src[0]);
        PolyUOp *gx = poly_uop1(ctx, POLY_OP_NEG, g->dtype, g, poly_arg_none());
        gx = reduce_to_shape(ctx, gx, s0);
        gx = cast_to(ctx, gx, u->src[0]->dtype);
        grad_add(ctx, grads, u->src[0], gx);
        shape_free(s0);
      } break;

      case POLY_OP_ADD:
      case POLY_OP_SUB: {
        PolyShape s0 = poly_uop_shape(ctx, u->src[0]);
        PolyShape s1 = poly_uop_shape(ctx, u->src[1]);

        PolyUOp *ga = reduce_to_shape(ctx, g, s0);
        ga = cast_to(ctx, ga, u->src[0]->dtype);
        grad_add(ctx, grads, u->src[0], ga);

        PolyUOp *gb = (u->op == POLY_OP_SUB)
          ? poly_uop1(ctx, POLY_OP_NEG, g->dtype, g, poly_arg_none())
          : g;
        gb = reduce_to_shape(ctx, gb, s1);
        gb = cast_to(ctx, gb, u->src[1]->dtype);
        grad_add(ctx, grads, u->src[1], gb);

        shape_free(s0);
        shape_free(s1);
      } break;

      case POLY_OP_MUL: {
        PolyShape s0 = poly_uop_shape(ctx, u->src[0]);
        PolyShape s1 = poly_uop_shape(ctx, u->src[1]);

        PolyUOp *g0 = cast_to(ctx, g, u->dtype);
        PolyUOp *ga = mul_grad(ctx, g0, u->src[1], u->dtype);
        ga = reduce_to_shape(ctx, ga, s0);
        ga = cast_to(ctx, ga, u->src[0]->dtype);
        grad_add(ctx, grads, u->src[0], ga);

        PolyUOp *gb = mul_grad(ctx, g0, u->src[0], u->dtype);
        gb = reduce_to_shape(ctx, gb, s1);
        gb = cast_to(ctx, gb, u->src[1]->dtype);
        grad_add(ctx, grads, u->src[1], gb);

        shape_free(s0);
        shape_free(s1);
      } break;

      case POLY_OP_FDIV: {
        PolyShape s0 = poly_uop_shape(ctx, u->src[0]);
        PolyShape s1 = poly_uop_shape(ctx, u->src[1]);
        PolyUOp *g0 = cast_to(ctx, g, u->dtype);

        /* d(a/b)/da = g/b */
        PolyUOp *ga = poly_uop2(ctx, POLY_OP_FDIV, u->dtype, g0, u->src[1], poly_arg_none());
        ga = reduce_to_shape(ctx, ga, s0);
        ga = cast_to(ctx, ga, u->src[0]->dtype);
        grad_add(ctx, grads, u->src[0], ga);

        /* d(a/b)/db = -(g*u)/b  where u = a/b (reuses forward result) */
        PolyUOp *gu = mul_grad(ctx, g0, u, u->dtype);
        PolyUOp *gb = poly_uop2(ctx, POLY_OP_FDIV, u->dtype, gu, u->src[1], poly_arg_none());
        gb = poly_uop1(ctx, POLY_OP_NEG, u->dtype, gb, poly_arg_none());
        gb = reduce_to_shape(ctx, gb, s1);
        gb = cast_to(ctx, gb, u->src[1]->dtype);
        grad_add(ctx, grads, u->src[1], gb);

        shape_free(s0);
        shape_free(s1);
      } break;

      case POLY_OP_EXP2: {
        /* d/dx exp2(x) = exp2(x) * ln2
         * tinygrad: ret * ctx * math.log(2) → exp2(x) * g * ln2 */
        PolyShape s0 = poly_uop_shape(ctx, u->src[0]);
        PolyUOp *g0 = cast_to(ctx, g, u->dtype);
        PolyUOp *ln2 = const_scalar(ctx, u->dtype, 0.69314718055994530942);
        PolyUOp *tmp = poly_uop2(ctx, POLY_OP_MUL, u->dtype, u, ln2, poly_arg_none());
        PolyUOp *gx = mul_grad(ctx, g0, tmp, u->dtype);
        gx = reduce_to_shape(ctx, gx, s0);
        gx = cast_to(ctx, gx, u->src[0]->dtype);
        grad_add(ctx, grads, u->src[0], gx);
        shape_free(s0);
      } break;

      case POLY_OP_LOG2: {
        /* d/dx log2(x) = 1/(x*ln2) = (1/ln2)/x
         * Pre-fold 1/ln2 to match tinygrad's symbolic simplification.
         * tinygrad: ctx / (ret.src[0] * math.log(2)) → (g*(1/ln2))/x */
        PolyShape s0 = poly_uop_shape(ctx, u->src[0]);
        PolyUOp *g0 = cast_to(ctx, g, u->dtype);
        PolyUOp *rln2 = const_scalar(ctx, u->dtype, 1.0 / 0.69314718055994530942);
        PolyUOp *gx = mul_grad(ctx, g0, rln2, u->dtype);
        gx = poly_uop2(ctx, POLY_OP_FDIV, u->dtype, gx, u->src[0], poly_arg_none());
        gx = reduce_to_shape(ctx, gx, s0);
        gx = cast_to(ctx, gx, u->src[0]->dtype);
        grad_add(ctx, grads, u->src[0], gx);
        shape_free(s0);
      } break;

      case POLY_OP_SQRT: {
        /* d/dx sqrt(x) = 1/(2*sqrt(x)) = 0.5/sqrt(x)
         * tinygrad: ctx / (ret*2) → g / (2*sqrt(x))
         * Pre-fold as (g*0.5)/sqrt(x) to match tinygrad output. */
        PolyShape s0 = poly_uop_shape(ctx, u->src[0]);
        PolyUOp *g0 = cast_to(ctx, g, u->dtype);
        PolyUOp *half = const_scalar(ctx, u->dtype, 0.5);
        PolyUOp *gx = mul_grad(ctx, g0, half, u->dtype);
        gx = poly_uop2(ctx, POLY_OP_FDIV, u->dtype, gx, u, poly_arg_none());
        gx = reduce_to_shape(ctx, gx, s0);
        gx = cast_to(ctx, gx, u->src[0]->dtype);
        grad_add(ctx, grads, u->src[0], gx);
        shape_free(s0);
      } break;

      case POLY_OP_RECIPROCAL: {
        /* d/dx (1/x) = -1/x^2 = -(1/x)^2 = -ret^2
         * tinygrad: -ctx * ret * ret */
        PolyShape s0 = poly_uop_shape(ctx, u->src[0]);
        PolyUOp *g0 = cast_to(ctx, g, u->dtype);
        PolyUOp *sq = poly_uop2(ctx, POLY_OP_MUL, u->dtype, u, u, poly_arg_none());
        PolyUOp *gx = mul_grad(ctx, g0, sq, u->dtype);
        gx = poly_uop1(ctx, POLY_OP_NEG, u->dtype, gx, poly_arg_none());
        gx = reduce_to_shape(ctx, gx, s0);
        gx = cast_to(ctx, gx, u->src[0]->dtype);
        grad_add(ctx, grads, u->src[0], gx);
        shape_free(s0);
      } break;

      case POLY_OP_SIN: {
        /* d/dx sin(x) = cos(x) = sin(π/2 - x)
         * tinygrad: (math.pi/2 - ret.src[0]).sin() * ctx */
        PolyShape s0 = poly_uop_shape(ctx, u->src[0]);
        PolyUOp *g0 = cast_to(ctx, g, u->dtype);
        PolyUOp *half_pi = const_scalar(ctx, u->dtype, 1.5707963267948966);
        PolyUOp *shifted = poly_uop2(ctx, POLY_OP_SUB, u->dtype,
            half_pi, u->src[0], poly_arg_none());
        PolyUOp *cos_x = poly_uop1(ctx, POLY_OP_SIN, u->dtype, shifted, poly_arg_none());
        PolyUOp *gx = mul_grad(ctx, g0, cos_x, u->dtype);
        gx = reduce_to_shape(ctx, gx, s0);
        gx = cast_to(ctx, gx, u->src[0]->dtype);
        grad_add(ctx, grads, u->src[0], gx);
        shape_free(s0);
      } break;

      case POLY_OP_MAX: {
        /* d/dx max(x, y): gradient goes to x when x>y, to y when x<y,
         * split 50/50 when x==y.
         * tinygrad: (x>y).where(ctx, (x.eq(y)).where(ctx*0.5, 0)) */
        PolyShape s0 = poly_uop_shape(ctx, u->src[0]);
        PolyShape s1 = poly_uop_shape(ctx, u->src[1]);
        PolyDType bdt = POLY_BOOL;
        PolyUOp *g0 = cast_to(ctx, g, u->dtype);
        PolyUOp *half = const_scalar(ctx, u->dtype, 0.5);
        PolyUOp *zero = const_scalar(ctx, u->dtype, 0.0);
        PolyUOp *half_g = poly_uop2(ctx, POLY_OP_MUL, u->dtype, g0, half, poly_arg_none());

        /* x > y is CMPLT(y, x) */
        PolyUOp *x_gt_y = poly_uop2(ctx, POLY_OP_CMPLT, bdt,
            u->src[1], u->src[0], poly_arg_none());
        /* x < y is CMPLT(x, y) */
        PolyUOp *x_lt_y = poly_uop2(ctx, POLY_OP_CMPLT, bdt,
            u->src[0], u->src[1], poly_arg_none());
        /* x == y via double negation: eq = !CMPNE(x, y) */
        PolyUOp *neq = poly_uop2(ctx, POLY_OP_CMPNE, bdt,
            u->src[0], u->src[1], poly_arg_none());
        PolyUOp *true_c = poly_uop0(ctx, POLY_OP_CONST, bdt, poly_arg_bool(true));
        PolyUOp *eq_mask = poly_uop2(ctx, POLY_OP_CMPNE, bdt,
            neq, true_c, poly_arg_none());

        PolyUOp *eq_part = poly_uop3(ctx, POLY_OP_WHERE, u->dtype,
            eq_mask, half_g, zero, poly_arg_none());
        /* ga = WHERE(x>y, g, eq_part) */
        PolyUOp *ga = poly_uop3(ctx, POLY_OP_WHERE, u->dtype,
            x_gt_y, g0, eq_part, poly_arg_none());
        /* gb = WHERE(x<y, g, eq_part) */
        PolyUOp *gb = poly_uop3(ctx, POLY_OP_WHERE, u->dtype,
            x_lt_y, g0, eq_part, poly_arg_none());

        ga = reduce_to_shape(ctx, ga, s0);
        ga = cast_to(ctx, ga, u->src[0]->dtype);
        gb = reduce_to_shape(ctx, gb, s1);
        gb = cast_to(ctx, gb, u->src[1]->dtype);
        grad_add(ctx, grads, u->src[0], ga);
        grad_add(ctx, grads, u->src[1], gb);
        shape_free(s0);
        shape_free(s1);
      } break;

      case POLY_OP_POW: {
        /* d/db b^e = e * b^(e-1), with edge case: if b==0 && e==0, use e
         * d/de b^e = b^e * ln(b), with edge case: if b==0, use -inf when e<0 else 0
         * tinygrad gradient.py:35-36 */
        PolyShape s0 = poly_uop_shape(ctx, u->src[0]);  /* base */
        PolyShape s1 = poly_uop_shape(ctx, u->src[1]);  /* exponent */
        PolyUOp *g0 = cast_to(ctx, g, u->dtype);
        PolyUOp *b = u->src[0], *e = u->src[1];
        PolyDType dt = u->dtype;
        PolyDType bdt = POLY_BOOL;
        PolyUOp *zero = const_scalar(ctx, dt, 0.0);
        PolyUOp *one = const_scalar(ctx, dt, 1.0);
        PolyUOp *true_c = poly_uop0(ctx, POLY_OP_CONST, bdt, poly_arg_bool(true));

        /* b_eq_0 = !(b CMPNE 0) */
        PolyUOp *b_neq_0 = poly_uop2(ctx, POLY_OP_CMPNE, bdt, b, zero, poly_arg_none());
        PolyUOp *b_eq_0 = poly_uop2(ctx, POLY_OP_CMPNE, bdt, b_neq_0, true_c, poly_arg_none());
        /* e_eq_0 = !(e CMPNE 0) */
        PolyUOp *e_neq_0 = poly_uop2(ctx, POLY_OP_CMPNE, bdt, e, zero, poly_arg_none());
        PolyUOp *e_eq_0 = poly_uop2(ctx, POLY_OP_CMPNE, bdt, e_neq_0, true_c, poly_arg_none());

        /* --- d/db: ctx * WHERE(b==0 & e==0, e, e * b^(e-1)) --- */
        PolyUOp *em1 = poly_uop2(ctx, POLY_OP_SUB, dt, e, one, poly_arg_none());
        PolyUOp *bpem1 = poly_uop2(ctx, POLY_OP_POW, dt, b, em1, poly_arg_none());
        PolyUOp *normal_db = poly_uop2(ctx, POLY_OP_MUL, dt, e, bpem1, poly_arg_none());
        /* both_zero = b_eq_0 AND e_eq_0 — use MUL on bools (AND semantics) */
        PolyUOp *b_eq_0_f = poly_uop1(ctx, POLY_OP_CAST, dt, b_eq_0, poly_arg_none());
        PolyUOp *e_eq_0_f = poly_uop1(ctx, POLY_OP_CAST, dt, e_eq_0, poly_arg_none());
        PolyUOp *both_zero_f = poly_uop2(ctx, POLY_OP_MUL, dt, b_eq_0_f, e_eq_0_f, poly_arg_none());
        PolyUOp *both_zero_neq = poly_uop2(ctx, POLY_OP_CMPNE, bdt, both_zero_f, zero, poly_arg_none());
        PolyUOp *db = poly_uop3(ctx, POLY_OP_WHERE, dt,
            both_zero_neq, e, normal_db, poly_arg_none());
        PolyUOp *ga = mul_grad(ctx, g0, db, dt);
        ga = reduce_to_shape(ctx, ga, s0);
        ga = cast_to(ctx, ga, b->dtype);
        grad_add(ctx, grads, b, ga);

        /* --- d/de: ctx * WHERE(b==0, WHERE(e<0, -inf, 0), ret * ln(b)) --- */
        PolyUOp *ln2 = const_scalar(ctx, dt, 0.69314718055994530942);
        PolyUOp *log2_b = poly_uop1(ctx, POLY_OP_LOG2, dt, b, poly_arg_none());
        PolyUOp *ln_b = poly_uop2(ctx, POLY_OP_MUL, dt, log2_b, ln2, poly_arg_none());
        PolyUOp *ret_ln_b = poly_uop2(ctx, POLY_OP_MUL, dt, u, ln_b, poly_arg_none());
        PolyUOp *neg_inf = const_scalar(ctx, dt, -1.0/0.0);
        PolyUOp *e_lt_0 = poly_uop2(ctx, POLY_OP_CMPLT, bdt, e, zero, poly_arg_none());
        PolyUOp *b_zero_case = poly_uop3(ctx, POLY_OP_WHERE, dt,
            e_lt_0, neg_inf, zero, poly_arg_none());
        PolyUOp *de = poly_uop3(ctx, POLY_OP_WHERE, dt,
            b_eq_0, b_zero_case, ret_ln_b, poly_arg_none());
        PolyUOp *gb = mul_grad(ctx, g0, de, dt);
        gb = reduce_to_shape(ctx, gb, s1);
        gb = cast_to(ctx, gb, e->dtype);
        grad_add(ctx, grads, e, gb);

        shape_free(s0);
        shape_free(s1);
      } break;

      case POLY_OP_WHERE: {
        PolyShape s1 = poly_uop_shape(ctx, u->src[1]);
        PolyShape s2 = poly_uop_shape(ctx, u->src[2]);
        PolyUOp *zero = const_scalar(ctx, u->dtype, 0.0);
        PolyUOp *g0 = cast_to(ctx, g, u->dtype);
        PolyUOp *gt = poly_uop3(ctx, POLY_OP_WHERE, u->dtype, u->src[0], g0, zero, poly_arg_none());
        PolyUOp *gf = poly_uop3(ctx, POLY_OP_WHERE, u->dtype, u->src[0], zero, g0, poly_arg_none());
        gt = reduce_to_shape(ctx, gt, s1);
        gf = reduce_to_shape(ctx, gf, s2);
        gt = cast_to(ctx, gt, u->src[1]->dtype);
        gf = cast_to(ctx, gf, u->src[2]->dtype);
        grad_add(ctx, grads, u->src[1], gt);
        grad_add(ctx, grads, u->src[2], gf);
        shape_free(s1);
        shape_free(s2);
      } break;

      case POLY_OP_RESHAPE: {
        PolyShape s0 = poly_uop_shape(ctx, u->src[0]);
        PolyUOp *gx = poly_reshape(ctx, g, s0.dims, s0.ndim);
        gx = cast_to(ctx, gx, u->src[0]->dtype);
        grad_add(ctx, grads, u->src[0], gx);
        shape_free(s0);
      } break;

      case POLY_OP_EXPAND: {
        PolyShape s0 = poly_uop_shape(ctx, u->src[0]);
        PolyUOp *gx = reduce_to_shape(ctx, g, s0);
        gx = cast_to(ctx, gx, u->src[0]->dtype);
        grad_add(ctx, grads, u->src[0], gx);
        shape_free(s0);
      } break;

      case POLY_OP_PERMUTE: {
        if (u->arg.kind != POLY_ARG_INT_TUPLE) {
          fprintf(stderr, "polygrad: autograd: PERMUTE missing int tuple arg\n");
          if (walk_owned) free(walk);
          poly_map_destroy(grads);
          return NULL;
        }
        int n = u->arg.int_tuple.n;
        if (n < 0 || n > POLY_MAX_DIMS) {
          fprintf(stderr, "polygrad: autograd: PERMUTE rank %d out of bounds\n", n);
          if (walk_owned) free(walk);
          poly_map_destroy(grads);
          return NULL;
        }
        int64_t inv[POLY_MAX_DIMS];
        for (int i = 0; i < n; i++) inv[i] = i;
        for (int i = 0; i < n; i++) inv[u->arg.int_tuple.vals[i]] = i;
        PolyUOp *gx = poly_permute(ctx, g, inv, n);
        gx = cast_to(ctx, gx, u->src[0]->dtype);
        grad_add(ctx, grads, u->src[0], gx);
      } break;

      case POLY_OP_PAD: {
        if (u->arg.kind != POLY_ARG_PAIR_TUPLE) {
          fprintf(stderr, "polygrad: autograd: PAD missing pair tuple arg\n");
          if (walk_owned) free(walk);
          poly_map_destroy(grads);
          return NULL;
        }
        int n = u->arg.pair_tuple.n;
        if (n < 0 || n > POLY_MAX_DIMS) {
          fprintf(stderr, "polygrad: autograd: PAD rank %d out of bounds\n", n);
          if (walk_owned) free(walk);
          poly_map_destroy(grads);
          return NULL;
        }
        PolyShape in_shape = poly_uop_shape(ctx, u->src[0]);
        int64_t shrink_pairs[POLY_MAX_DIMS][2];
        for (int i = 0; i < n; i++) {
          int64_t before = u->arg.pair_tuple.pairs[i][0];
          shrink_pairs[i][0] = before;
          shrink_pairs[i][1] = before + in_shape.dims[i];
        }
        PolyUOp *gx = poly_shrink(ctx, g, shrink_pairs, n);
        gx = cast_to(ctx, gx, u->src[0]->dtype);
        grad_add(ctx, grads, u->src[0], gx);
        shape_free(in_shape);
      } break;

      case POLY_OP_SHRINK: {
        if (u->arg.kind != POLY_ARG_PAIR_TUPLE) {
          fprintf(stderr, "polygrad: autograd: SHRINK missing pair tuple arg\n");
          if (walk_owned) free(walk);
          poly_map_destroy(grads);
          return NULL;
        }
        int n = u->arg.pair_tuple.n;
        if (n < 0 || n > POLY_MAX_DIMS) {
          fprintf(stderr, "polygrad: autograd: SHRINK rank %d out of bounds\n", n);
          if (walk_owned) free(walk);
          poly_map_destroy(grads);
          return NULL;
        }
        PolyShape in_shape = poly_uop_shape(ctx, u->src[0]);
        int64_t pad_pairs[POLY_MAX_DIMS][2];
        for (int i = 0; i < n; i++) {
          int64_t start = u->arg.pair_tuple.pairs[i][0];
          int64_t end = u->arg.pair_tuple.pairs[i][1];
          pad_pairs[i][0] = start;
          pad_pairs[i][1] = in_shape.dims[i] - end;
        }
        PolyUOp *gx = poly_pad(ctx, g, pad_pairs, n);
        gx = cast_to(ctx, gx, u->src[0]->dtype);
        grad_add(ctx, grads, u->src[0], gx);
        shape_free(in_shape);
      } break;

      case POLY_OP_FLIP: {
        if (u->arg.kind != POLY_ARG_INT_TUPLE) {
          fprintf(stderr, "polygrad: autograd: FLIP missing int tuple arg\n");
          if (walk_owned) free(walk);
          poly_map_destroy(grads);
          return NULL;
        }
        PolyUOp *gx = poly_flip(ctx, g, u->arg.int_tuple.vals, u->arg.int_tuple.n);
        gx = cast_to(ctx, gx, u->src[0]->dtype);
        grad_add(ctx, grads, u->src[0], gx);
      } break;

      case POLY_OP_REDUCE_AXIS: {
        if (u->arg.kind != POLY_ARG_REDUCE_AXIS) {
          fprintf(stderr, "polygrad: autograd: REDUCE_AXIS missing reduce_axis arg\n");
          if (walk_owned) free(walk);
          poly_map_destroy(grads);
          return NULL;
        }
        PolyOps reduce_op = u->arg.reduce_axis.op;
        PolyShape in_shape = poly_uop_shape(ctx, u->src[0]);

        if (reduce_op == POLY_OP_ADD) {
          /* d/dx sum(x) = expand(g, input_shape) */
          PolyUOp *gx = reduce_to_shape(ctx, g, in_shape);
          gx = cast_to(ctx, gx, u->src[0]->dtype);
          grad_add(ctx, grads, u->src[0], gx);
        } else if (reduce_op == POLY_OP_MAX) {
          /* d/dx max(x, axis) = (x == expand(max_val)) / count * expand(g)
           * Matches tinygrad: mask = input.eq(broadcast(ret)), count = mask.sum(axis),
           * grad = (mask / broadcast(count)) * broadcast(upstream) */
          int n_axes = u->arg.reduce_axis.n;
          int64_t *axes = u->arg.reduce_axis.axes;

          /* CONTIGUOUS barrier on the max result: forces rangeify to realize
           * the forward REDUCE_AXIS(MAX) as a separate kernel. Without this,
           * the consumer kernel would recompute max, creating a
           * reduce→expand→alu pattern that rangeify can't handle. */
          PolyUOp *max_contig = poly_uop1(ctx, POLY_OP_CONTIGUOUS, u->dtype, u,
                                           poly_arg_none());
          /* Broadcast max output back to input shape */
          PolyUOp *max_bcast = poly_expand(ctx, max_contig, in_shape.dims,
                                            in_shape.ndim);

          /* mask = (input == max_val).cast(float32) */
          PolyDType bool_dt = POLY_BOOL;
          PolyUOp *eq = poly_uop2(ctx, POLY_OP_CMPNE, bool_dt,
              u->src[0], max_bcast, poly_arg_none());
          /* CMPNE gives true where not equal; negate to get true where equal */
          PolyUOp *true_val = poly_uop0(ctx, POLY_OP_CONST, bool_dt, poly_arg_bool(true));
          true_val = poly_expand(ctx, true_val, in_shape.dims, in_shape.ndim);
          PolyUOp *mask = poly_uop2(ctx, POLY_OP_CMPNE, bool_dt, eq, true_val, poly_arg_none());
          /* Cast mask to float */
          PolyUOp *fmask = poly_uop1(ctx, POLY_OP_CAST, u->dtype, mask, poly_arg_none());

          /* count = mask.sum(axis) — how many elements equal the max */
          PolyUOp *count = poly_reduce_axis(ctx, POLY_OP_ADD, fmask, axes, n_axes);
          /* CONTIGUOUS barrier: forces rangeify to realize count as a separate
           * kernel, breaking the reduce→expand→alu pattern into two realizable
           * kernels. Without this, the scheduler can't handle the fused pattern. */
          count = poly_uop1(ctx, POLY_OP_CONTIGUOUS, count->dtype, count, poly_arg_none());

          /* Broadcast count and upstream grad to input shape */
          PolyUOp *count_bcast = poly_expand(ctx, count, in_shape.dims, in_shape.ndim);
          PolyUOp *g_bcast = reduce_to_shape(ctx, g, in_shape);
          g_bcast = cast_to(ctx, g_bcast, u->dtype);

          /* gx = (mask / count) * upstream_grad */
          PolyUOp *scaled_mask = poly_uop2(ctx, POLY_OP_FDIV, u->dtype,
              fmask, count_bcast, poly_arg_none());
          PolyUOp *gx = poly_uop2(ctx, POLY_OP_MUL, u->dtype,
              scaled_mask, g_bcast, poly_arg_none());
          gx = cast_to(ctx, gx, u->src[0]->dtype);
          grad_add(ctx, grads, u->src[0], gx);
        } else {
          fprintf(stderr, "polygrad: autograd: unsupported REDUCE_AXIS op: %s\n",
                  poly_op_name(reduce_op));
          shape_free(in_shape);
          if (walk_owned) free(walk);
          poly_map_destroy(grads);
          return NULL;
        }
        shape_free(in_shape);
      } break;

      default:
        /* With target pruning, any op on the gradient path was selected
         * because it lies on a path to a target. Silent skip hides bugs.
         * Fail hard so callers get NULL and can report the error. */
        fprintf(stderr, "polygrad: autograd: missing gradient rule for %s\n",
                poly_op_name(u->op));
        poly_map_destroy(grads);
        if (walk_owned) free(walk);
        return NULL;
    }
  }

  if (walk_owned) free(walk);
  return grads;
}

PolyUOp *poly_grad(PolyCtx *ctx, PolyUOp *loss, PolyUOp *wrt) {
  if (!ctx || !loss || !wrt) return NULL;

  PolyMap *grads = grad_reverse_pass(ctx, loss, ones_like(ctx, loss), &wrt, 1);
  if (!grads) return NULL;

  PolyUOp *out = grad_get(grads, wrt);
  if (!out) out = zeros_like(ctx, wrt);

  out = poly_graph_rewrite(ctx, out, poly_symbolic_simple());

  poly_map_destroy(grads);
  return out;
}

int poly_grad_many(PolyCtx *ctx, PolyUOp *loss, PolyUOp *initial_grad,
                    PolyUOp **wrts, int n, PolyUOp **out_grads) {
  if (!ctx || !loss || !wrts || !out_grads || n <= 0) return -1;

  PolyMap *grads = grad_reverse_pass(ctx, loss,
      initial_grad ? initial_grad : ones_like(ctx, loss), wrts, n);
  if (!grads) return -1;

  for (int i = 0; i < n; i++) {
    PolyUOp *g = grad_get(grads, wrts[i]);
    if (!g) g = zeros_like(ctx, wrts[i]);
    out_grads[i] = poly_graph_rewrite(ctx, g, poly_symbolic_simple());
  }

  poly_map_destroy(grads);
  return 0;
}

/* ── UOp graph substitution ─────────────────────────────────────────── */

static PolyUOp *substitute_rec(PolyCtx *ctx, PolyUOp *u,
                                PolyMap *sub_map, PolyMap *memo) {
  /* Check substitution map first — and recurse into the replacement
   * to handle nested intermediates (e.g. var's internal mean realize
   * inside layernorm's var realize). */
  void *sub = poly_map_get(sub_map, ptr_hash(u), u, ptr_eq);
  if (sub) return substitute_rec(ctx, (PolyUOp *)sub, sub_map, memo);

  /* Check memo */
  void *cached = poly_map_get(memo, ptr_hash(u), u, ptr_eq);
  if (cached) return (PolyUOp *)cached;

  /* Leaf node: no sources to recurse into */
  if (u->n_src == 0) {
    poly_map_set(memo, ptr_hash(u), u, u, ptr_eq);
    return u;
  }

  /* Recursively substitute sources */
  PolyUOp *new_srcs[16]; /* enough for any UOp */
  bool changed = false;
  for (int i = 0; i < u->n_src && i < 16; i++) {
    new_srcs[i] = substitute_rec(ctx, u->src[i], sub_map, memo);
    if (new_srcs[i] != u->src[i]) changed = true;
  }

  PolyUOp *result;
  if (!changed) {
    result = u;
  } else {
    result = poly_uop(ctx, u->op, u->dtype, new_srcs, u->n_src, u->arg);
  }
  poly_map_set(memo, ptr_hash(u), u, result, ptr_eq);
  return result;
}

PolyUOp *poly_uop_substitute(PolyCtx *ctx, PolyUOp *root,
                               PolyUOp **from, PolyUOp **to, int n) {
  if (!ctx || !root || n <= 0) return root;

  PolyMap *sub_map = poly_map_new((size_t)n * 2 + 16);
  for (int i = 0; i < n; i++) {
    poly_map_set(sub_map, ptr_hash(from[i]), from[i], to[i], ptr_eq);
  }

  PolyMap *memo = poly_map_new(256);
  PolyUOp *result = substitute_rec(ctx, root, sub_map, memo);

  poly_map_destroy(sub_map);
  poly_map_destroy(memo);
  return result;
}
