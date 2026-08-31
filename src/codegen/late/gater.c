/* Current Tinygrad 2026-08-22/a9069c177a9d codegen/late/gater.py. */

#include "codegen/late/gater.h"
#include "uop/ops.h"

#include <stdlib.h>

/* Preserves Polygrad's C-only metadata while applying Tinygrad UOp.replace. */
static PolyUOp *gater_rebuild(PolyCtx *ctx, PolyUOp *u, PolyUOp **src, int n_src) {
  return (u->tag != 0 || u->tag_arg.kind != POLY_ARG_NONE)
             ? poly_uop_tagged_arg(ctx, u->op, u->dtype, src, n_src, u->arg, u->tag, u->tag_arg)
             : poly_uop(ctx, u->op, u->dtype, src, n_src, u->arg);
}

/* C matcher for `gate.where(idx, Invalid)`. */
static bool gater_invalid_where(PolyUOp *u, PolyUOp **gate, PolyUOp **index) {
  if (!u || u->op != POLY_OP_WHERE || u->n_src != 3 || !u->src[2] ||
      u->src[2]->op != POLY_OP_CONST || u->src[2]->arg.kind != POLY_ARG_INVALID ||
      !poly_dtype_is_bool(u->src[0]->dtype) || !poly_dtype_is_int(u->src[1]->dtype))
    return false;
  if (gate) *gate = u->src[0];
  if (index) *index = u->src[1];
  return true;
}

/* Tinygrad gater.py:5-12 has a distinct two-coordinate image-shaped rule. */
static PolyUOp *ungated_image_index(PolyCtx *ctx, PolyUOp *mop, PolyUOp **gate_out) {
  if (!mop || mop->op != POLY_OP_INDEX || mop->n_src != 3) return NULL;
  PolyUOp *gate_y = NULL, *gate_x = NULL, *idx_y = NULL, *idx_x = NULL;
  if (!gater_invalid_where(mop->src[1], &gate_y, &idx_y) ||
      !gater_invalid_where(mop->src[2], &gate_x, &idx_x) || gate_y != gate_x)
    return NULL;
  PolyUOp *src[3] = {mop->src[0], idx_y, idx_x};
  if (gate_out) *gate_out = gate_y;
  return gater_rebuild(ctx, mop, src, 3);
}

/* Tinygrad gater.py:14-17 matches only src[1] and retains src[2:]. */
static PolyUOp *ungated_first_index(PolyCtx *ctx, PolyUOp *mop, PolyUOp **gate_out) {
  if (!mop || (mop->op != POLY_OP_INDEX && mop->op != POLY_OP_SHRINK) || mop->n_src < 2)
    return NULL;
  PolyUOp *gate = NULL, *idx = NULL;
  if (!gater_invalid_where(mop->src[1], &gate, &idx)) return NULL;
  PolyUOp **src = malloc((size_t)mop->n_src * sizeof(*src));
  if (!src) return NULL;
  src[0] = mop->src[0];
  src[1] = idx;
  for (int i = 2; i < mop->n_src; i++)
    src[i] = mop->src[i];
  PolyUOp *ret = gater_rebuild(ctx, mop, src, mop->n_src);
  free(src);
  if (ret && gate_out) *gate_out = gate;
  return ret;
}

static PolyUOp *move_gated_image_index_to_load(PolyCtx *ctx, PolyUOp *load, const PolyBindings *b) {
  (void)b;
  if (!load || load->op != POLY_OP_LOAD || load->n_src < 1 || load->n_src >= 3) return NULL;
  PolyUOp *gate = NULL;
  PolyUOp *ungated = ungated_image_index(ctx, load->src[0], &gate);
  if (!ungated || !gate) return NULL;
  PolyUOp *alt = load->n_src >= 2 ? load->src[1] : poly_const_like_int(ctx, load, 0);
  PolyUOp *src[3] = {ungated, alt, gate};
  return gater_rebuild(ctx, load, src, 3);
}

static PolyUOp *move_gated_image_index_to_store(
    PolyCtx *ctx,
    PolyUOp *store,
    const PolyBindings *b
) {
  (void)b;
  if (!store || store->op != POLY_OP_STORE || store->n_src != 2) return NULL;
  PolyUOp *gate = NULL;
  PolyUOp *ungated = ungated_image_index(ctx, store->src[0], &gate);
  if (!ungated || !gate) return NULL;
  PolyUOp *src[3] = {ungated, store->src[1], gate};
  return gater_rebuild(ctx, store, src, 3);
}

/* Tinygrad gater.py:14-16 moves one gated coordinate to LOAD's gate. */
static PolyUOp *move_gated_index_to_load(PolyCtx *ctx, PolyUOp *load, const PolyBindings *b) {
  (void)b;
  if (!load || load->op != POLY_OP_LOAD || load->n_src < 1 || load->n_src >= 3) return NULL;
  PolyUOp *gate = NULL;
  PolyUOp *ungated = ungated_first_index(ctx, load->src[0], &gate);
  if (!ungated || !gate) return NULL;
  PolyUOp *alt = load->n_src >= 2 ? load->src[1] : poly_const_like_int(ctx, load, 0);
  PolyUOp *src[3] = {ungated, alt, gate};
  return gater_rebuild(ctx, load, src, 3);
}

/* Tinygrad gater.py:17 moves one gated coordinate to STORE's gate. */
static PolyUOp *move_gated_index_to_store(PolyCtx *ctx, PolyUOp *store, const PolyBindings *b) {
  (void)b;
  if (!store || store->op != POLY_OP_STORE || store->n_src != 2) return NULL;
  PolyUOp *gate = NULL;
  PolyUOp *ungated = ungated_first_index(ctx, store->src[0], &gate);
  if (!ungated || !gate) return NULL;
  PolyUOp *src[3] = {ungated, store->src[1], gate};
  return gater_rebuild(ctx, store, src, 3);
}

static PolyUOp *unwrap_casted_load(PolyUOp *u) {
  if (u && u->op == POLY_OP_LOAD) return u;
  return u && u->op == POLY_OP_CAST && u->n_src == 1 && u->src[0] && u->src[0]->op == POLY_OP_LOAD
             ? u->src[0]
             : NULL;
}

static bool is_logical_not(PolyUOp *u, PolyUOp *gate) {
  return u && u->op == POLY_OP_CMPNE && u->n_src == 2 && u->src[0] == gate && u->src[1] &&
         u->src[1]->op == POLY_OP_CONST && u->src[1]->arg.kind == POLY_ARG_BOOL && u->src[1]->arg.b;
}

/* Current Tinygrad gater.py:move_where_load. */
static PolyUOp *move_where_load(PolyCtx *ctx, PolyUOp *where, PolyUOp *load, PolyUOp *alt) {
  PolyUOp *load_alt = NULL;
  if (alt->op == POLY_OP_CONST && alt->arg.kind == POLY_ARG_INVALID)
    load_alt = poly_const_like_int(ctx, load, 0);
  else if (alt->op == POLY_OP_CONST)
    load_alt = poly_const_like(ctx, load, alt->arg);
  else if (alt->op == POLY_OP_CAST && alt->n_src == 1 && poly_dtype_eq(alt->src[0]->dtype, load->dtype))
    load_alt = alt->src[0];
  else
    load_alt = poly_cast(ctx, alt, load->dtype);
  if (!load_alt) return NULL;
  PolyUOp *src[3] = {load->src[0], load_alt, load->src[2]};
  PolyUOp *ret = gater_rebuild(ctx, load, src, 3);
  return ret ? poly_cast(ctx, ret, where->dtype) : NULL;
}

static PolyUOp *move_where_gated_load(PolyCtx *ctx, PolyUOp *where, const PolyBindings *b) {
  (void)b;
  if (!where || where->op != POLY_OP_WHERE || where->n_src != 3) return NULL;
  PolyUOp *load = unwrap_casted_load(where->src[1]);
  if (!load || load->n_src != 3 || load->src[2] != where->src[0]) return NULL;
  return move_where_load(ctx, where, load, where->src[2]);
}

static PolyUOp *move_where_reverse_gated_load(PolyCtx *ctx, PolyUOp *where, const PolyBindings *b) {
  (void)b;
  if (!where || where->op != POLY_OP_WHERE || where->n_src != 3) return NULL;
  PolyUOp *load = unwrap_casted_load(where->src[2]);
  if (!load || load->n_src != 3 || !is_logical_not(load->src[2], where->src[0])) return NULL;
  return move_where_load(ctx, where, load, where->src[1]);
}

static _Thread_local PolyPatternMatcher *g_pm_move_gates_from_index = NULL;
PolyPatternMatcher *poly_pm_move_gates_from_index(void) {
  if (g_pm_move_gates_from_index) return g_pm_move_gates_from_index;
  PolyRule rules[] = {
      {poly_upat_allow_any_len(poly_upat_op(POLY_OP_LOAD, NULL, 0, "load")),
       move_gated_image_index_to_load},
      {poly_upat_allow_any_len(poly_upat_op(POLY_OP_STORE, NULL, 0, "store")),
       move_gated_image_index_to_store},
      {poly_upat_allow_any_len(poly_upat_op(POLY_OP_LOAD, NULL, 0, "load")),
       move_gated_index_to_load},
      {poly_upat_allow_any_len(poly_upat_op(POLY_OP_STORE, NULL, 0, "store")),
       move_gated_index_to_store},
      {poly_upat_op(POLY_OP_WHERE, NULL, 0, "w"), move_where_gated_load},
      {poly_upat_op(POLY_OP_WHERE, NULL, 0, "w"), move_where_reverse_gated_load},
  };
  g_pm_move_gates_from_index =
      poly_pm_thread_cache(poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0]))));
  return g_pm_move_gates_from_index;
}
