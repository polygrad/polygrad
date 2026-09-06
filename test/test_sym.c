/*
 * test_sym.c — Tests for symbolic simplification + ALU constant folding
 */

#include "test_harness.h"
#include "../src/bigint.h"
#include "../src/ctx.h"
#include "../src/frontend.h"
#include "../src/uop/upat.h"
#include "../src/tensor.h"
#include "../src/uop/weak.h"
#include <limits.h>
#include <math.h>
#include <stdlib.h>

/* Helper: apply symbolic_simple via graph_rewrite */

static PolyUOp *simplify(PolyCtx *ctx, PolyUOp *root) {
  return poly_graph_rewrite(ctx, root, poly_symbolic_simple());
}

/* Matches test/null/test_uop_symbolic.py's compiler-side Variable helper. */
static PolyUOp *Variable(
    PolyCtx *ctx,
    const char *name,
    int64_t min_val,
    int64_t max_val,
    PolyDType dtype
) {
  return poly_uop_variable(ctx, name, min_val, max_val, dtype, 1, true);
}

static bool integer_const_eq(PolyUOp *u, const char *expected);

TEST(sym, current_symbolic_simple_identity_batch) {
  /* tinygrad@2026-08-22/a9069c177a9d uop/symbolic.py:111-186. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *x = Variable(ctx, "x", 0, 255, POLY_WEAKINT);
  PolyUOp *y = Variable(ctx, "y", 0, 255, POLY_WEAKINT);
  PolyUOp *b = Variable(ctx, "b", 0, 1, POLY_BOOL);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *three = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(3));
  PolyUOp *seven = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(7));
  PolyUOp *eight = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(8));
  PolyUOp *neg_eight = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(-8));
  PolyUOp *false_uop = poly_uop0(ctx, POLY_OP_CONST, POLY_BOOL, poly_arg_bool(false));
  PolyUOp *true_uop = poly_uop0(ctx, POLY_OP_CONST, POLY_BOOL, poly_arg_bool(true));

  ASSERT_PTR_EQ(
      simplify(ctx, poly_uop2(ctx, POLY_OP_OR, POLY_WEAKINT, x, zero, poly_arg_none())), x
  );
  ASSERT_PTR_EQ(
      simplify(ctx, poly_uop2(ctx, POLY_OP_SHL, POLY_WEAKINT, x, zero, poly_arg_none())), x
  );
  ASSERT_PTR_EQ(
      simplify(ctx, poly_uop2(ctx, POLY_OP_SHR, POLY_WEAKINT, x, zero, poly_arg_none())), x
  );

  PolyUOp *xy = poly_uop2(ctx, POLY_OP_XOR, POLY_WEAKINT, x, y, poly_arg_none());
  ASSERT_PTR_EQ(
      simplify(ctx, poly_uop2(ctx, POLY_OP_XOR, POLY_WEAKINT, xy, y, poly_arg_none())), x
  );
  PolyUOp *mod = poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_WEAKINT, x, seven, poly_arg_none());
  ASSERT_PTR_EQ(
      simplify(ctx, poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_WEAKINT, mod, seven, poly_arg_none())),
      mod
  );

  ASSERT_PTR_EQ(
      simplify(ctx, poly_uop2(ctx, POLY_OP_CMPNE, POLY_BOOL, b, false_uop, poly_arg_none())), b
  );
  ASSERT_PTR_EQ(simplify(ctx, poly_logical_not(ctx, poly_logical_not(ctx, b))), b);

  PolyUOp *b_int = poly_uop1(ctx, POLY_OP_CAST, POLY_WEAKINT, b, poly_arg_none());
  ASSERT_PTR_EQ(
      simplify(ctx, poly_uop2(ctx, POLY_OP_CMPNE, POLY_BOOL, b_int, zero, poly_arg_none())), b
  );
  PolyUOp *not_b = poly_logical_not(ctx, b);
  ASSERT_PTR_EQ(
      simplify(
          ctx, poly_uop2(
                   ctx, POLY_OP_CMPNE, POLY_BOOL, b_int,
                   poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1)), poly_arg_none()
               )
      ),
      not_b
  );
  ASSERT_PTR_EQ(
      simplify(
          ctx, poly_uop2(
                   ctx, POLY_OP_CMPNE, POLY_BOOL, b_int,
                   poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(2)), poly_arg_none()
               )
      ),
      true_uop
  );

  PolyUOp *masked = poly_uop2(ctx, POLY_OP_AND, POLY_WEAKINT, x, neg_eight, poly_arg_none());
  ASSERT_PTR_EQ(
      simplify(ctx, poly_uop2(ctx, POLY_OP_SHR, POLY_WEAKINT, masked, three, poly_arg_none())),
      poly_uop2(ctx, POLY_OP_SHR, POLY_WEAKINT, x, three, poly_arg_none())
  );
  ASSERT_PTR_EQ(
      simplify(ctx, poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_WEAKINT, masked, eight, poly_arg_none())),
      poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_WEAKINT, x, eight, poly_arg_none())
  );
  ASSERT_PTR_EQ(
      simplify(ctx, poly_uop2(ctx, POLY_OP_MAX, POLY_BOOL, b, false_uop, poly_arg_none())), b
  );

  PolyUOp *as_uint = poly_uop1(ctx, POLY_OP_CAST, POLY_UINT32, x, poly_arg_none());
  PolyUOp *as_float = poly_uop1(ctx, POLY_OP_BITCAST, POLY_FLOAT32, as_uint, poly_arg_none());
  ASSERT_PTR_EQ(
      simplify(ctx, poly_uop1(ctx, POLY_OP_BITCAST, POLY_UINT32, as_float, poly_arg_none())),
      as_uint
  );

  PolyUOp *weak_seven = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(7));
  PolyUOp *strong_nine = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(9));
  PolyUOp *strong_where =
      poly_uop3(ctx, POLY_OP_WHERE, POLY_INT32, true_uop, weak_seven, strong_nine, poly_arg_none());
  PolyUOp *strong_seven = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(7));
  ASSERT_PTR_EQ(simplify(ctx, strong_where), strong_seven);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, current_symbolic_completion_batch) {
  /* Tinygrad 2026-08-22/a9069c177a9d uop/symbolic.py:231-308. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *x = Variable(ctx, "x", 0, 15, POLY_WEAKINT);
  PolyUOp *y = Variable(ctx, "y", 0, 15, POLY_WEAKINT);
  PolyUOp *a = Variable(ctx, "a", 0, 15, POLY_WEAKINT);
  PolyUOp *b = Variable(ctx, "b", 0, 15, POLY_WEAKINT);
  PolyUOp *cond = Variable(ctx, "cond", 0, 1, POLY_BOOL);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(2));
  PolyUOp *three = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(3));
  PolyUOp *ten = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(10));
  PolyUOp *truth = poly_uop0(ctx, POLY_OP_CONST, POLY_BOOL, poly_arg_bool(true));

  PolyUOp *not_cond = poly_logical_not(ctx, cond);
  PolyUOp *out = poly_graph_rewrite(
      ctx, poly_uop2(ctx, POLY_OP_OR, POLY_BOOL, cond, not_cond, poly_arg_none()), poly_symbolic()
  );
  ASSERT_INT_EQ(out->op, POLY_OP_CONST);
  ASSERT_TRUE(out->arg.kind == POLY_ARG_BOOL && out->arg.b);

  PolyUOp *x2 = poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, x, two, poly_arg_none());
  PolyUOp *x3 = poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, x, three, poly_arg_none());
  PolyUOp *combine[] = {
      poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, x2, x3, poly_arg_none()),
      poly_uop2(
          ctx, POLY_OP_ADD, POLY_WEAKINT,
          poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, y, x2, poly_arg_none()), x3, poly_arg_none()
      ),
      poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, x, x2, poly_arg_none()),
      poly_uop2(
          ctx, POLY_OP_ADD, POLY_WEAKINT,
          poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, y, x, poly_arg_none()), x2, poly_arg_none()
      ),
      poly_uop2(
          ctx, POLY_OP_ADD, POLY_WEAKINT,
          poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, y, x2, poly_arg_none()), x, poly_arg_none()
      ),
  };
  for (int i = 0; i < 5; i++) {
    out = poly_graph_rewrite(ctx, combine[i], poly_symbolic());
    PolyUOp *mul = (i == 0 || i == 2) ? out : out->src[1];
    ASSERT_INT_EQ(mul->op, POLY_OP_MUL);
    ASSERT_PTR_EQ(mul->src[0], x);
    ASSERT_TRUE(integer_const_eq(mul->src[1], i < 2 ? "5" : "3"));
    if (i != 0 && i != 2) {
      ASSERT_INT_EQ(out->op, POLY_OP_ADD);
      ASSERT_PTR_EQ(out->src[0], y);
    }
  }

  PolyUOp *wa = poly_uop3(ctx, POLY_OP_WHERE, POLY_WEAKINT, cond, two, a, poly_arg_none());
  PolyUOp *wb = poly_uop3(ctx, POLY_OP_WHERE, POLY_WEAKINT, cond, three, b, poly_arg_none());
  out = poly_graph_rewrite(
      ctx, poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, wa, wb, poly_arg_none()), poly_symbolic()
  );
  ASSERT_INT_EQ(out->op, POLY_OP_WHERE);
  ASSERT_PTR_EQ(out->src[0], cond);
  ASSERT_TRUE(integer_const_eq(out->src[1], "5"));
  ASSERT_INT_EQ(out->src[2]->op, POLY_OP_ADD);
  ASSERT_PTR_EQ(out->src[2]->src[0], a);
  ASSERT_PTR_EQ(out->src[2]->src[1], b);

  PolyUOp *assoc_where = poly_uop2(
      ctx, POLY_OP_ADD, POLY_WEAKINT,
      poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, y, wa, poly_arg_none()), wb, poly_arg_none()
  );
  out = poly_graph_rewrite(ctx, assoc_where, poly_symbolic());
  ASSERT_INT_EQ(out->op, POLY_OP_ADD);
  ASSERT_PTR_EQ(out->src[0], y);
  ASSERT_INT_EQ(out->src[1]->op, POLY_OP_WHERE);
  ASSERT_TRUE(integer_const_eq(out->src[1]->src[1], "5"));

  PolyUOp *wx = poly_uop3(ctx, POLY_OP_WHERE, POLY_WEAKINT, cond, x, zero, poly_arg_none());
  PolyUOp *wy = poly_uop3(ctx, POLY_OP_WHERE, POLY_WEAKINT, cond, zero, y, poly_arg_none());
  out = poly_graph_rewrite(
      ctx, poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, wx, wy, poly_arg_none()), poly_symbolic()
  );
  ASSERT_INT_EQ(out->op, POLY_OP_WHERE);
  ASSERT_PTR_EQ(out->src[0], cond);
  ASSERT_PTR_EQ(out->src[1], x);
  ASSERT_PTR_EQ(out->src[2], y);

  PolyUOp *two_lt_x = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, two, x, poly_arg_none());
  out = poly_graph_rewrite(
      ctx, poly_uop3(ctx, POLY_OP_WHERE, POLY_WEAKINT, two_lt_x, x, two, poly_arg_none()),
      poly_symbolic()
  );
  ASSERT_INT_EQ(out->op, POLY_OP_MAX);
  ASSERT_PTR_EQ(out->src[0], x);
  ASSERT_PTR_EQ(out->src[1], two);
  PolyUOp *x_lt_two = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, x, two, poly_arg_none());
  out = poly_graph_rewrite(
      ctx, poly_uop3(ctx, POLY_OP_WHERE, POLY_WEAKINT, x_lt_two, two, x, poly_arg_none()),
      poly_symbolic()
  );
  ASSERT_INT_EQ(out->op, POLY_OP_MAX);
  ASSERT_PTR_EQ(out->src[0], x);
  ASSERT_PTR_EQ(out->src[1], two);

  PolyUOp *three_x = poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, three, x, poly_arg_none());
  out = poly_graph_rewrite(
      ctx, poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, three_x, ten, poly_arg_none()), poly_symbolic()
  );
  ASSERT_INT_EQ(out->op, POLY_OP_CMPLT);
  ASSERT_PTR_EQ(out->src[0], x);
  ASSERT_TRUE(integer_const_eq(out->src[1], "4"));
  PolyUOp *x_div_three = poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_WEAKINT, x, three, poly_arg_none());
  out = poly_graph_rewrite(
      ctx, poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, x_div_three, two, poly_arg_none()),
      poly_symbolic()
  );
  ASSERT_PTR_EQ(out->src[0], x);
  ASSERT_TRUE(integer_const_eq(out->src[1], "6"));

  PolyUOp *neg_one = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(-1));
  PolyUOp *neg_x = poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, x, neg_one, poly_arg_none());
  PolyUOp *neg_y = poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, y, neg_one, poly_arg_none());
  out = poly_graph_rewrite(
      ctx, poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, neg_x, neg_y, poly_arg_none()), poly_symbolic()
  );
  ASSERT_PTR_EQ(out->src[0], y);
  ASSERT_PTR_EQ(out->src[1], x);

  PolyUOp *weighted = poly_uop2(
      ctx, POLY_OP_ADD, POLY_WEAKINT,
      poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, two, x, poly_arg_none()),
      poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, three, y, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *simplex = poly_uop2(
      ctx, POLY_OP_CMPNE, POLY_BOOL,
      poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, weighted, one, poly_arg_none()), truth,
      poly_arg_none()
  );
  out = poly_graph_rewrite(ctx, simplex, poly_symbolic());
  ASSERT_INT_EQ(out->op, POLY_OP_CMPNE);
  ASSERT_INT_EQ(out->src[0]->op, POLY_OP_CMPLT);
  ASSERT_INT_EQ(out->src[0]->src[0]->op, POLY_OP_ADD);
  ASSERT_PTR_EQ(out->src[0]->src[0]->src[0], x);
  ASSERT_PTR_EQ(out->src[0]->src[0]->src[1], y);

  PolyUOp *sixteen = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(16));
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, sixteen, poly_arg_range(0, POLY_AXIS_LOOP));
  ASSERT_PTR_EQ(
      poly_graph_rewrite(
          ctx, poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_WEAKINT, range, sixteen, poly_arg_none()),
          poly_symbolic()
      ),
      range
  );
  out = poly_graph_rewrite(
      ctx, poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_WEAKINT, range, sixteen, poly_arg_none()),
      poly_symbolic()
  );
  ASSERT_TRUE(integer_const_eq(out, "0"));

  PolyUOp *xf = Variable(ctx, "xf", 1, 4, POLY_FLOAT32);
  PolyUOp *yf = Variable(ctx, "yf", 1, 4, POLY_FLOAT32);
  PolyUOp *zf = Variable(ctx, "zf", 1, 4, POLY_FLOAT32);
  PolyUOp *ry = poly_uop1(ctx, POLY_OP_RECIPROCAL, POLY_FLOAT32, yf, poly_arg_none());
  PolyUOp *rz = poly_uop1(ctx, POLY_OP_RECIPROCAL, POLY_FLOAT32, zf, poly_arg_none());
  PolyUOp *nested_div = poly_uop2(
      ctx, POLY_OP_MUL, POLY_FLOAT32,
      poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, xf, ry, poly_arg_none()), rz, poly_arg_none()
  );
  out = poly_graph_rewrite(ctx, nested_div, poly_symbolic());
  ASSERT_INT_EQ(out->op, POLY_OP_MUL);
  ASSERT_PTR_EQ(out->src[0], xf);
  ASSERT_INT_EQ(out->src[1]->op, POLY_OP_RECIPROCAL);
  ASSERT_INT_EQ(out->src[1]->src[0]->op, POLY_OP_MUL);
  ASSERT_PTR_EQ(out->src[1]->src[0]->src[0], yf);
  ASSERT_PTR_EQ(out->src[1]->src[0]->src[1], zf);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, complementary_float_where_selects_directly) {
  /* tinygrad 2026-08-22/a9069c177a9d uop/symbolic.py:254-255. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *cond = Variable(ctx, "cond", 0, 1, POLY_BOOL);
  PolyUOp *t = Variable(ctx, "t", -100, 100, POLY_FLOAT32);
  PolyUOp *f = Variable(ctx, "f", -100, 100, POLY_FLOAT32);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(0.0));
  PolyUOp *true_leg = poly_uop3(ctx, POLY_OP_WHERE, POLY_FLOAT32, cond, t, zero, poly_arg_none());
  PolyUOp *false_leg = poly_uop3(ctx, POLY_OP_WHERE, POLY_FLOAT32, cond, zero, f, poly_arg_none());
  PolyUOp *out = poly_graph_rewrite(
      ctx, poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, true_leg, false_leg, poly_arg_none()),
      poly_symbolic()
  );

  ASSERT_INT_EQ(out->op, POLY_OP_WHERE);
  ASSERT_PTR_EQ(out->src[0], cond);
  ASSERT_PTR_EQ(out->src[1], t);
  ASSERT_PTR_EQ(out->src[2], f);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, bool_const_rules_do_not_rewrite_integer_bitwise_ops) {
  /* tinygrad 2026-08-22/a9069c177a9d uop/symbolic.py:120-121. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *x = Variable(ctx, "x", 0, INT32_MAX, POLY_UINT32);
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT32, poly_arg_int(1));
  PolyUOp *integer_and = poly_uop2(ctx, POLY_OP_AND, POLY_UINT32, x, one, poly_arg_none());
  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, integer_and, poly_symbolic_simple()), integer_and);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, current_sym_completion_batch) {
  /* Tinygrad 2026-08-22/a9069c177a9d uop/symbolic.py:441-475. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *xi = Variable(ctx, "xi", 0, 7, POLY_WEAKINT);
  PolyUOp *yi = Variable(ctx, "yi", 0, 7, POLY_WEAKINT);
  PolyUOp *x = Variable(ctx, "x", 1, 8, POLY_FLOAT32);
  PolyUOp *y = Variable(ctx, "y", 1, 8, POLY_FLOAT32);
  PolyUOp *cond = Variable(ctx, "cond", 0, 1, POLY_BOOL);

  PolyUOp *xs_src[2] = {xi, xi};
  PolyUOp *ys_src[2] = {yi, yi};
  PolyUOp *xs = poly_uop(ctx, POLY_OP_STACK, POLY_WEAKINT, xs_src, 2, poly_arg_none());
  PolyUOp *ys = poly_uop(ctx, POLY_OP_STACK, POLY_WEAKINT, ys_src, 2, poly_arg_none());
  PolyUOp *stack_add = poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, xs, ys, poly_arg_none());
  PolyUOp *lane_add = poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, xi, yi, poly_arg_none());
  PolyUOp *stack_expected = poly_uop1(ctx, POLY_OP_STACK, POLY_WEAKINT, lane_add, poly_arg_none());
  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, stack_add, poly_sym()), stack_expected);

  PolyUOp *weak_two = poly_uop_const(ctx, poly_arg_float(2.0), POLY_WEAKFLOAT);
  PolyUOp *weak_three = poly_uop_const(ctx, poly_arg_float(3.0), POLY_WEAKFLOAT);
  PolyUOp *raw_where =
      poly_uop3(ctx, POLY_OP_WHERE, POLY_WEAKFLOAT, cond, weak_two, weak_three, poly_arg_none());
  PolyUOp *cast_where = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, raw_where, poly_arg_none());
  PolyUOp *cast_expected = poly_uop3(
      ctx, POLY_OP_WHERE, POLY_FLOAT32, cond,
      poly_uop_const(ctx, poly_arg_float(2.0), POLY_FLOAT32),
      poly_uop_const(ctx, poly_arg_float(3.0), POLY_FLOAT32), poly_arg_none()
  );
  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, cast_where, poly_sym()), cast_expected);

  PolyUOp *pow = poly_uop2(ctx, POLY_OP_POW, POLY_FLOAT32, x, y, poly_arg_none());
  PolyUOp *pow_out = poly_graph_rewrite(ctx, pow, poly_sym());
  ASSERT_INT_EQ(pow_out->op, POLY_OP_EXP2);
  ASSERT_INT_EQ(pow_out->src[0]->op, POLY_OP_MUL);
  ASSERT_INT_EQ(pow_out->src[0]->src[0]->op, POLY_OP_LOG2);
  ASSERT_PTR_EQ(pow_out->src[0]->src[0]->src[0], x);
  ASSERT_PTR_EQ(pow_out->src[0]->src[1], y);

  PolyUOp *one = poly_uop_const(ctx, poly_arg_int(1), POLY_WEAKINT);
  PolyUOp *zero = poly_uop_const(ctx, poly_arg_int(0), POLY_WEAKINT);
  PolyUOp *seven = poly_uop_const(ctx, poly_arg_int(7), POLY_WEAKINT);
  PolyParamArg buf_arg = {
      .slot = 0,
      .dtype = POLY_INT32,
      .addrspace = POLY_ADDR_GLOBAL,
  };
  PolyUOp *buf = poly_uop1(ctx, POLY_OP_PARAM, POLY_INT32, one, poly_arg_param(&buf_arg));
  PolyUOp *index = poly_uop2(ctx, POLY_OP_INDEX, POLY_INT32, buf, zero, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_INT32, index, poly_arg_none());
  PolyUOp *noop = poly_uop0(ctx, POLY_OP_NOOP, POLY_VOID, poly_arg_none());
  ASSERT_PTR_EQ(
      poly_graph_rewrite(
          ctx, poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, index, load, poly_arg_none()), poly_sym()
      ),
      noop
  );

  PolyUOp *invalid = poly_uop_const(ctx, poly_arg_invalid(), POLY_BOOL);
  ASSERT_PTR_EQ(
      poly_graph_rewrite(
          ctx, poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, index, invalid, poly_arg_none()), poly_sym()
      ),
      noop
  );
  PolyUOp *valid_index =
      poly_uop3(ctx, POLY_OP_WHERE, POLY_WEAKINT, cond, zero, invalid, poly_arg_none());
  PolyUOp *gated_index =
      poly_uop2(ctx, POLY_OP_INDEX, POLY_INT32, buf, valid_index, poly_arg_none());
  PolyUOp *gated_expected =
      poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, gated_index, seven, poly_arg_none());
  PolyUOp *gated_load =
      poly_uop3(ctx, POLY_OP_WHERE, POLY_INT32, cond, seven, load, poly_arg_none());
  ASSERT_PTR_EQ(
      poly_graph_rewrite(
          ctx, poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, index, gated_load, poly_arg_none()),
          poly_sym()
      ),
      gated_expected
  );
  PolyUOp *gated_invalid =
      poly_uop3(ctx, POLY_OP_WHERE, POLY_WEAKINT, cond, seven, invalid, poly_arg_none());
  ASSERT_PTR_EQ(
      poly_graph_rewrite(
          ctx, poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, index, gated_invalid, poly_arg_none()),
          poly_sym()
      ),
      gated_expected
  );

  PolyUOp *neg_one = poly_uop_const(ctx, poly_arg_int(-1), POLY_WEAKINT);
  PolyUOp *negated = poly_uop2(
      ctx, POLY_OP_MUL, POLY_WEAKINT,
      poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, xi, yi, poly_arg_none()), neg_one, poly_arg_none()
  );
  PolyUOp *negated_expected = poly_uop2(
      ctx, POLY_OP_ADD, POLY_WEAKINT,
      poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, xi, neg_one, poly_arg_none()),
      poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, yi, neg_one, poly_arg_none()), poly_arg_none()
  );
  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, negated, poly_sym()), negated_expected);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, symbolic_simple_excludes_removed_june_rows) {
  /* Tinygrad 2026-08-22/a9069c177a9d uop/symbolic.py:106-178. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *x = Variable(ctx, "x", 1, 4, POLY_FLOAT32);
  PolyUOp *y = Variable(ctx, "y", 1, 4, POLY_FLOAT32);

  PolyUOp *double_neg = poly_uop1(
      ctx, POLY_OP_NEG, POLY_FLOAT32, poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, x, poly_arg_none()),
      poly_arg_none()
  );
  PolyUOp *raw_fdiv_self = poly_uop2(ctx, POLY_OP_FDIV, POLY_FLOAT32, x, x, poly_arg_none());
  PolyUOp *xy = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, x, y, poly_arg_none());
  PolyUOp *raw_fdiv_cancel = poly_uop2(ctx, POLY_OP_FDIV, POLY_FLOAT32, xy, y, poly_arg_none());
  PolyUOp *one = poly_uop_const(ctx, poly_arg_int(1), POLY_UINT32);
  PolyUOp *two = poly_uop_const(ctx, poly_arg_int(2), POLY_UINT32);
  PolyUOp *threefry = poly_uop2(ctx, POLY_OP_THREEFRY, POLY_UINT32, one, two, poly_arg_none());

  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, double_neg, poly_symbolic_simple()), double_neg);
  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, raw_fdiv_self, poly_symbolic_simple()), raw_fdiv_self);
  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, raw_fdiv_cancel, poly_symbolic_simple()), raw_fdiv_cancel);
  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, threefry, poly_symbolic_simple()), threefry);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, add_self_keeps_python_literal_weak) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = Variable(ctx, "x", 1, 8, POLY_INT32);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, x, x, poly_arg_none());
  PolyUOp *result = poly_graph_rewrite(ctx, add, poly_symbolic());

  ASSERT_NOT_NULL(result);
  ASSERT_INT_EQ(result->op, POLY_OP_MUL);
  ASSERT_PTR_EQ(result->src[0], x);
  ASSERT_INT_EQ(result->src[1]->op, POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(result->src[1]->dtype, POLY_WEAKINT));
  ASSERT_INT_EQ(result->src[1]->arg.i, 2);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, where_shared_true_branch_combines_conditions_with_or) {
  /* Current tinygrad/uop/symbolic.py:189-190 rewrites
   * a.where(c, b.where(c, d)) to (a | b).where(c, d). */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = Variable(ctx, "a", 0, 1, POLY_BOOL);
  PolyUOp *b = Variable(ctx, "b", 0, 1, POLY_BOOL);
  PolyUOp *c = Variable(ctx, "c", -4, 4, POLY_FLOAT32);
  PolyUOp *d = Variable(ctx, "d", -4, 4, POLY_FLOAT32);
  PolyUOp *inner = poly_uop3(ctx, POLY_OP_WHERE, POLY_FLOAT32, b, c, d, poly_arg_none());
  PolyUOp *root = poly_uop3(ctx, POLY_OP_WHERE, POLY_FLOAT32, a, c, inner, poly_arg_none());
  PolyUOp *out = simplify(ctx, root);

  ASSERT_INT_EQ(out->op, POLY_OP_WHERE);
  ASSERT_INT_EQ(out->src[0]->op, POLY_OP_OR);
  ASSERT_TRUE(out->src[0]->src[0] == a);
  ASSERT_TRUE(out->src[0]->src[1] == b);
  ASSERT_TRUE(out->src[1] == c);
  ASSERT_TRUE(out->src[2] == d);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, where_closure_folds_condition_inside_true_branch) {
  /* tinygrad 2026-08-22/a9069c177a9d uop/symbolic.py:218-223 treats the
   * WHERE condition as True in its true branch and preserves Invalid. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *cond = Variable(ctx, "cond", 0, 1, POLY_BOOL);
  PolyUOp *extra = Variable(ctx, "extra", 0, 1, POLY_BOOL);
  PolyUOp *both = poly_uop2(ctx, POLY_OP_AND, POLY_BOOL, cond, extra, poly_arg_none());
  PolyUOp *invalid = poly_uop_const(ctx, poly_arg_invalid(), POLY_BOOL);
  PolyUOp *root = poly_uop3(ctx, POLY_OP_WHERE, POLY_BOOL, cond, both, invalid, poly_arg_none());

  PolyUOp *out = poly_graph_rewrite(ctx, root, poly_symbolic());
  ASSERT_INT_EQ(out->op, POLY_OP_WHERE);
  ASSERT_PTR_EQ(out->src[0], cond);
  ASSERT_PTR_EQ(out->src[1], extra);
  ASSERT_PTR_EQ(out->src[2], invalid);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, variable_and_bind_use_current_alu_storage_topology) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *n = poly_uop_variable(ctx, "n", 1, 8, POLY_WEAKINT, 1, false);
  ASSERT_NOT_NULL(n);
  ASSERT_INT_EQ(n->op, POLY_OP_BUFFER);
  ASSERT_TRUE(poly_dtype_eq(n->dtype, POLY_WEAKINT));
  ASSERT_INT_EQ(n->n_src, 1);
  ASSERT_INT_EQ(n->src[0]->op, POLY_OP_STACK);
  ASSERT_INT_EQ(n->src[0]->n_src, 0);
  ASSERT_INT_EQ(n->arg.kind, POLY_ARG_PARAM);
  ASSERT_NOT_NULL(n->arg.param);
  ASSERT_INT_EQ(n->arg.param->slot, -1);
  ASSERT_TRUE(poly_dtype_eq(n->arg.param->dtype, POLY_WEAKINT));
  ASSERT_TRUE(n->arg.param->has_minmax);
  ASSERT_INT_EQ(n->arg.param->min_val, 1);
  ASSERT_INT_EQ(n->arg.param->max_val, 8);
  ASSERT_TRUE(n->arg.param->has_multiple_of);
  ASSERT_INT_EQ(n->arg.param->multiple_of, 1);
  ASSERT_INT_EQ(n->arg.param->addrspace, POLY_ADDR_ALU);
  ASSERT_TRUE(n->arg.param->name && strcmp(n->arg.param->name, "n") == 0);

  PolyUOp *bound = poly_uop_bind(ctx, n, 4);
  ASSERT_NOT_NULL(bound);
  ASSERT_INT_EQ(bound->op, POLY_OP_AFTER);
  ASSERT_PTR_EQ(bound->src[0], n);
  ASSERT_INT_EQ(bound->src[1]->op, POLY_OP_STORE);
  ASSERT_PTR_EQ(bound->src[1]->src[0], n);
  ASSERT_INT_EQ(bound->src[1]->src[1]->op, POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(bound->src[1]->src[1]->dtype, POLY_WEAKINT));
  ASSERT_INT_EQ(bound->src[1]->src[1]->arg.i, 4);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, lower_weak_commits_current_alu_param_metadata) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *param = poly_uop_variable(ctx, "n", 1, 8, POLY_WEAKINT, 1, true);
  PolyUOp *lowered = poly_graph_rewrite(ctx, param, poly_pm_lower_weak());
  ASSERT_NOT_NULL(lowered);
  ASSERT_INT_EQ(lowered->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(lowered->dtype, POLY_WEAKINT));
  ASSERT_INT_EQ(lowered->src[0]->op, POLY_OP_PARAM);
  ASSERT_TRUE(poly_dtype_eq(lowered->src[0]->dtype, POLY_INT32));
  ASSERT_TRUE(poly_dtype_eq(lowered->src[0]->arg.param->dtype, POLY_INT32));
  ASSERT_INT_EQ(lowered->src[0]->arg.param->addrspace, POLY_ADDR_ALU);
  ASSERT_TRUE(
      lowered->src[0]->arg.param->name && strcmp(lowered->src[0]->arg.param->name, "n") == 0
  );
  ASSERT_INT_EQ(lowered->src[0]->arg.param->min_val, 1);
  ASSERT_INT_EQ(lowered->src[0]->arg.param->max_val, 8);
  poly_ctx_destroy(ctx);
  PASS();
}

/* ALU constant fold tests */

TEST(alu, fold_add_int) {
  PolyArg ops[2] = {poly_arg_int(2), poly_arg_int(3)};
  PolyArg r = poly_exec_alu(POLY_OP_ADD, POLY_INT32, ops, 2, true);
  ASSERT_INT_EQ(r.i, 5);
  PASS();
}

TEST(alu, fold_mul_int) {
  PolyArg ops[2] = {poly_arg_int(4), poly_arg_int(7)};
  PolyArg r = poly_exec_alu(POLY_OP_MUL, POLY_INT32, ops, 2, true);
  ASSERT_INT_EQ(r.i, 28);
  PASS();
}

TEST(alu, fold_neg_float) {
  PolyArg ops[1] = {poly_arg_float(3.14)};
  PolyArg r = poly_exec_alu(POLY_OP_NEG, POLY_FLOAT32, ops, 1, true);
  ASSERT_FLOAT_EQ(r.f, -3.14, 1e-6);
  PASS();
}

TEST(alu, fold_add_float) {
  PolyArg ops[2] = {poly_arg_float(1.5), poly_arg_float(2.5)};
  PolyArg r = poly_exec_alu(POLY_OP_ADD, POLY_FLOAT32, ops, 2, true);
  ASSERT_FLOAT_EQ(r.f, 4.0, 1e-6);
  PASS();
}

TEST(alu, cast_without_output_truncation_matches_host_symbolic_conversion) {
  /* Pinned tinygrad UOp._sym_fxn renders CAST as a host-language conversion
   * (ops.py:1021-1035), so inference does not narrow to the destination width. */
  PolyArg operand = poly_arg_int(260);
  PolyArg host = poly_exec_alu(POLY_OP_CAST, POLY_UINT8, &operand, 1, false);
  PolyArg typed = poly_exec_alu(POLY_OP_CAST, POLY_UINT8, &operand, 1, true);
  ASSERT_TRUE(host.kind == POLY_ARG_INT);
  ASSERT_INT_EQ(host.i, 260);
  ASSERT_TRUE(typed.kind == POLY_ARG_INT);
  ASSERT_INT_EQ(typed.i, 4);
  PASS();
}

TEST(sym, bitcast_const_reinterprets_equal_width_storage_like_tinygrad) {
  /* Pinned tinygrad/uop/symbolic.py:19-24,150 uses struct pack/unpack rather
   * than a numeric cast. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *bits = simplify(ctx, poly_uop1(ctx, POLY_OP_BITCAST, POLY_UINT32, one, poly_arg_none()));
  ASSERT_NOT_NULL(bits);
  ASSERT_INT_EQ(bits->op, POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(bits->dtype, POLY_UINT32));
  ASSERT_INT_EQ(bits->arg.i, 1065353216);

  PolyUOp *negative_bits = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT32, poly_arg_int(0xbf800000));
  PolyUOp *negative =
      simplify(ctx, poly_uop1(ctx, POLY_OP_BITCAST, POLY_FLOAT32, negative_bits, poly_arg_none()));
  ASSERT_NOT_NULL(negative);
  ASSERT_INT_EQ(negative->op, POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(negative->dtype, POLY_FLOAT32));
  ASSERT_FLOAT_EQ(negative->arg.f, -1.0, 0.0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, threefry_pack_unpack_identities_match_tinygrad) {
  /* Tinygrad 2026-08-22/a9069c177a9d uop/symbolic.py:179-182 keeps only
   * the SHL/OR low- and high-lane extraction identities. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *high = poly_uop0(ctx, POLY_OP_NOOP, POLY_UINT32, poly_arg_none());
  PolyUOp *low = poly_uop0(ctx, POLY_OP_NOOP, POLY_UINT32, poly_arg_int(1));
  PolyUOp *high64 = poly_uop1(ctx, POLY_OP_CAST, POLY_UINT64, high, poly_arg_none());
  PolyUOp *mask = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT64, poly_arg_int(0xffffffff));
  PolyUOp *masked = poly_uop1(
      ctx, POLY_OP_CAST, POLY_UINT32,
      poly_uop2(ctx, POLY_OP_AND, POLY_UINT64, high64, mask, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *masked_rewritten = simplify(ctx, masked);
  ASSERT_INT_EQ(masked_rewritten->op, POLY_OP_CAST);
  ASSERT_INT_EQ(masked_rewritten->src[0]->op, POLY_OP_AND);

  PolyUOp *low64 = poly_uop1(ctx, POLY_OP_CAST, POLY_UINT64, low, poly_arg_none());
  PolyUOp *c32 = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT64, poly_arg_int(32));
  PolyUOp *shift_pack = poly_uop2(
      ctx, POLY_OP_OR, POLY_UINT64,
      poly_uop2(ctx, POLY_OP_SHL, POLY_UINT64, high64, c32, poly_arg_none()), low64, poly_arg_none()
  );
  ASSERT_PTR_EQ(
      simplify(ctx, poly_uop1(ctx, POLY_OP_CAST, POLY_UINT32, shift_pack, poly_arg_none())), low
  );
  ASSERT_PTR_EQ(
      simplify(ctx, poly_uop2(ctx, POLY_OP_SHR, POLY_UINT64, shift_pack, c32, poly_arg_none())),
      high64
  );

  /* A different mask must remain explicit. */
  PolyUOp *small_mask = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT64, poly_arg_int(0xffff));
  PolyUOp *not_lane = poly_uop1(
      ctx, POLY_OP_CAST, POLY_UINT32,
      poly_uop2(ctx, POLY_OP_AND, POLY_UINT64, high64, small_mask, poly_arg_none()), poly_arg_none()
  );
  ASSERT_INT_EQ(simplify(ctx, not_lane)->src[0]->op, POLY_OP_AND);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(alu, fold_idiv) {
  PolyArg ops[2] = {poly_arg_int(7), poly_arg_int(3)};
  PolyArg r = poly_exec_alu(POLY_OP_IDIV, POLY_INT32, ops, 2, true);
  ASSERT_INT_EQ(r.i, 2);
  PASS();
}

TEST(alu, fold_mod) {
  PolyArg ops[2] = {poly_arg_int(7), poly_arg_int(3)};
  PolyArg r = poly_exec_alu(POLY_OP_MOD, POLY_INT32, ops, 2, true);
  ASSERT_INT_EQ(r.i, 1);
  PASS();
}

TEST(alu, fold_floordiv_floormod_signed_like_tinygrad) {
  const int64_t vals[][4] = {
      {-7, 3, -3, 2}, {-7, -3, 2, -1}, {7, -3, -3, -2}, {7, 3, 2, 1},
      {-1, 4, -1, 3}, {1, -4, -1, -3}, {0, 3, 0, 0},
  };
  for (int i = 0; i < (int)(sizeof(vals) / sizeof(vals[0])); i++) {
    PolyArg ops[2] = {poly_arg_int(vals[i][0]), poly_arg_int(vals[i][1])};
    PolyArg q = poly_exec_alu(POLY_OP_FLOORDIV, POLY_INT32, ops, 2, true);
    PolyArg r = poly_exec_alu(POLY_OP_FLOORMOD, POLY_INT32, ops, 2, true);
    ASSERT_INT_EQ(q.i, vals[i][2]);
    ASSERT_INT_EQ(r.i, vals[i][3]);
  }
  PASS();
}

TEST(alu, fold_cmplt) {
  PolyArg ops[2] = {poly_arg_int(2), poly_arg_int(5)};
  PolyArg r = poly_exec_alu(POLY_OP_CMPLT, POLY_INT32, ops, 2, true);
  ASSERT_TRUE(r.b == true);
  ops[0] = poly_arg_int(5);
  r = poly_exec_alu(POLY_OP_CMPLT, POLY_INT32, ops, 2, true);
  ASSERT_TRUE(r.b == false);
  PASS();
}

TEST(alu, fold_where) {
  PolyArg ops[3] = {poly_arg_bool(true), poly_arg_int(10), poly_arg_int(20)};
  PolyArg r = poly_exec_alu(POLY_OP_WHERE, POLY_INT32, ops, 3, true);
  ASSERT_INT_EQ(r.i, 10);
  ops[0] = poly_arg_bool(false);
  r = poly_exec_alu(POLY_OP_WHERE, POLY_INT32, ops, 3, true);
  ASSERT_INT_EQ(r.i, 20);
  PASS();
}

TEST(alu, raw_bool_neg_matches_pinned_arithmetic_then_bool_truncation) {
  /* Pinned uop/ops.py:1182-1197 maps NEG to operator.neg, then applies
   * bool() only when truncate_output is true. Raw bool NEG is not logical NOT. */
  PolyArg f[1] = {poly_arg_bool(false)};
  PolyArg t[1] = {poly_arg_bool(true)};
  PolyArg f_raw = poly_exec_alu(POLY_OP_NEG, POLY_BOOL, f, 1, false);
  PolyArg t_raw = poly_exec_alu(POLY_OP_NEG, POLY_BOOL, t, 1, false);
  PolyArg f_typed = poly_exec_alu(POLY_OP_NEG, POLY_BOOL, f, 1, true);
  PolyArg t_typed = poly_exec_alu(POLY_OP_NEG, POLY_BOOL, t, 1, true);
  ASSERT_TRUE(f_raw.kind == POLY_ARG_INT && f_raw.i == 0);
  ASSERT_TRUE(t_raw.kind == POLY_ARG_INT && t_raw.i == -1);
  ASSERT_TRUE(f_typed.kind == POLY_ARG_BOOL && !f_typed.b);
  ASSERT_TRUE(t_typed.kind == POLY_ARG_BOOL && t_typed.b);
  PASS();
}

TEST(alu, divmod_zero_and_extreme_floor_remainder_match_pinned_helpers) {
  /* Pinned helpers.py:69-74 defines remainder through x-div(x,y)*y. */
  PolyArg zero_divisor[2] = {poly_arg_int(7), poly_arg_int(0)};
  ASSERT_INT_EQ(poly_exec_alu(POLY_OP_IDIV, POLY_INT64, zero_divisor, 2, false).i, 0);
  ASSERT_INT_EQ(poly_exec_alu(POLY_OP_MOD, POLY_INT64, zero_divisor, 2, false).i, 7);
  ASSERT_INT_EQ(poly_exec_alu(POLY_OP_FLOORDIV, POLY_INT64, zero_divisor, 2, false).i, 0);
  ASSERT_INT_EQ(poly_exec_alu(POLY_OP_FLOORMOD, POLY_INT64, zero_divisor, 2, false).i, 7);

  PolyArg extreme[2] = {poly_arg_int(1), poly_arg_int(INT64_MIN)};
  ASSERT_INT_EQ(poly_exec_alu(POLY_OP_FLOORDIV, POLY_INT64, extreme, 2, false).i, -1);
  ASSERT_TRUE(poly_exec_alu(POLY_OP_FLOORMOD, POLY_INT64, extreme, 2, false).i == -INT64_MAX);
  PASS();
}

TEST(alu, weak_index_where_preserves_invalid_with_truncation_enabled) {
  /* weakint is absent from pinned dtype.truncate (dtype.py:351-355). */
  PolyArg operands[3] = {poly_arg_bool(false), poly_arg_int(7), poly_arg_invalid()};
  PolyArg raw = poly_exec_alu(POLY_OP_WHERE, POLY_WEAKINT, operands, 3, false);
  PolyArg typed = poly_exec_alu(POLY_OP_WHERE, POLY_WEAKINT, operands, 3, true);
  ASSERT_TRUE(raw.kind == POLY_ARG_INVALID);
  ASSERT_TRUE(typed.kind == POLY_ARG_INVALID);
  PASS();
}

TEST(alu, negative_integer_pow_matches_pinned_raw_and_refuses_fixed_width_truncation) {
  /* Pinned safe_pow returns a float before ctypes fixed-width truncation
   * rejects it (uop/ops.py:1178-1197). weakint has no truncation entry. */
  PolyArg finite[2] = {poly_arg_int(2), poly_arg_int(-1)};
  PolyArg raw = poly_exec_alu(POLY_OP_POW, POLY_INT32, finite, 2, false);
  PolyArg typed = poly_exec_alu(POLY_OP_POW, POLY_INT32, finite, 2, true);
  PolyArg weak_typed = poly_exec_alu(POLY_OP_POW, POLY_WEAKINT, finite, 2, true);
  ASSERT_TRUE(raw.kind == POLY_ARG_FLOAT);
  ASSERT_FLOAT_EQ(raw.f, 0.5, 0.0);
  ASSERT_TRUE(typed.kind == POLY_ARG_INVALID);
  ASSERT_TRUE(weak_typed.kind == POLY_ARG_FLOAT);
  ASSERT_FLOAT_EQ(weak_typed.f, 0.5, 0.0);

  PolyArg infinite[2] = {poly_arg_int(0), poly_arg_int(-1)};
  PolyArg inf_raw = poly_exec_alu(POLY_OP_POW, POLY_INT32, infinite, 2, false);
  ASSERT_TRUE(inf_raw.kind == POLY_ARG_FLOAT && isinf(inf_raw.f) && inf_raw.f > 0.0);
  PASS();
}

TEST(alu, raw_integer_operands_match_pinned_python_alu_before_output_truncation) {
  /* Pinned tinygrad uop/ops.py:1182-1197 applies python_alu to stored CONST
   * values first. Nominal signedness/width only truncates the result. */
  PolyArg pow_args[2] = {poly_arg_int(2), poly_arg_int(3)};
  PolyArg cmp_args[2] = {poly_arg_int(130), poly_arg_int(0)};
  ASSERT_INT_EQ(poly_exec_alu(POLY_OP_POW, POLY_INT32, pow_args, 2, false).i, 8);
  ASSERT_TRUE(!poly_exec_alu(POLY_OP_CMPLT, POLY_INT8, cmp_args, 2, false).b);

  const PolyOps divmod_ops[] = {
      POLY_OP_CDIV,
      POLY_OP_CMOD,
      POLY_OP_FLOORDIV,
      POLY_OP_FLOORMOD,
  };
  const int64_t raw_expected[] = {-1, -1, -2, 1};
  const int64_t u8_expected[] = {255, 255, 254, 1};
  PolyArg div_args[2] = {poly_arg_int(-3), poly_arg_int(2)};
  for (int i = 0; i < 4; i++) {
    PolyArg raw = poly_exec_alu(divmod_ops[i], POLY_UINT8, div_args, 2, false);
    PolyArg truncated = poly_exec_alu(divmod_ops[i], POLY_UINT8, div_args, 2, true);
    ASSERT_TRUE(raw.kind == POLY_ARG_INT);
    ASSERT_TRUE(truncated.kind == POLY_ARG_INT);
    ASSERT_INT_EQ(raw.i, raw_expected[i]);
    ASSERT_INT_EQ(truncated.i, u8_expected[i]);
  }

  PolyArg where_args[3] = {poly_arg_bool(true), poly_arg_int(-3), poly_arg_int(2)};
  ASSERT_INT_EQ(poly_exec_alu(POLY_OP_WHERE, POLY_UINT8, where_args, 3, false).i, -3);
  ASSERT_INT_EQ(poly_exec_alu(POLY_OP_WHERE, POLY_UINT8, where_args, 3, true).i, 253);
  PASS();
}

TEST(sym, raw_integer_const_folds_match_pinned_python_alu) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *two_i32 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(2));
  PolyUOp *three_i32 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(3));
  PolyUOp *pow =
      simplify(ctx, poly_uop2(ctx, POLY_OP_POW, POLY_INT32, two_i32, three_i32, poly_arg_none()));
  ASSERT_NOT_NULL(pow);
  ASSERT_INT_EQ(pow->op, POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(pow->dtype, POLY_INT32));
  ASSERT_INT_EQ(pow->arg.i, 8);

  PolyUOp *raw_i8 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT8, poly_arg_int(130));
  PolyUOp *zero_i8 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT8, poly_arg_int(0));
  PolyUOp *cmp =
      simplify(ctx, poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, raw_i8, zero_i8, poly_arg_none()));
  ASSERT_NOT_NULL(cmp);
  ASSERT_INT_EQ(cmp->op, POLY_OP_CONST);
  ASSERT_TRUE(cmp->arg.kind == POLY_ARG_BOOL);
  ASSERT_TRUE(!cmp->arg.b);

  PolyUOp *neg_three_u8 = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT8, poly_arg_int(-3));
  PolyUOp *two_u8 = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT8, poly_arg_int(2));
  PolyUOp *floordiv = simplify(
      ctx, poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_UINT8, neg_three_u8, two_u8, poly_arg_none())
  );
  ASSERT_NOT_NULL(floordiv);
  ASSERT_INT_EQ(floordiv->op, POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(floordiv->dtype, POLY_UINT8));
  ASSERT_INT_EQ(floordiv->arg.i, -2);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(alu, compare_int64_preserves_precision) {
  PolyArg ops[2] = {poly_arg_int(9007199254740992LL), poly_arg_int(9007199254740993LL)};
  PolyArg r = poly_exec_alu(POLY_OP_CMPNE, POLY_INT64, ops, 2, true);
  ASSERT_TRUE(r.b == true);
  r = poly_exec_alu(POLY_OP_CMPEQ, POLY_INT64, ops, 2, true);
  ASSERT_TRUE(r.b == false);
  PASS();
}

TEST(alu, compare_uint64_const_args_remain_raw_before_result_truncation) {
  /* A genuinely positive uint64 above INT64_MAX is PG-PARITY-020. Do not
   * reinterpret a stored negative CONST as that missing representation:
   * pinned exec_alu compares the raw Python values (uop/ops.py:1192-1197). */
  PolyArg ops[2] = {poly_arg_int(-3), poly_arg_int(2)};
  PolyArg r = poly_exec_alu(POLY_OP_CMPLT, POLY_UINT64, ops, 2, true);
  ASSERT_TRUE(r.b == true);
  PolyArg rev[2] = {ops[1], ops[0]};
  r = poly_exec_alu(POLY_OP_CMPLT, POLY_UINT64, rev, 2, true);
  ASSERT_TRUE(r.b == false);
  PASS();
}

TEST(sym, shaped_stack_const_fold_broadcasts_scalar_like_current_tinygrad) {
  /* Current symbolic.py:27-40 treats STACK(CONST, ...) as tuple data even
   * though its dtype is scalar. exec_alu broadcasts the scalar operand and
   * returns one folded CONST per shaped lane. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *lanes[3] = {
      poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0)),
      poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1)),
      poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(2)),
  };
  PolyUOp *stack = poly_uop_stack(ctx, lanes, 3);
  PolyUOp *four = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(4));
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, stack, four, poly_arg_none());
  PolyUOp *folded = simplify(ctx, mul);

  ASSERT_NOT_NULL(folded);
  ASSERT_INT_EQ(folded->op, POLY_OP_STACK);
  ASSERT_INT_EQ(folded->n_src, 3);
  for (int i = 0; i < 3; i++) {
    ASSERT_INT_EQ(folded->src[i]->op, POLY_OP_CONST);
    ASSERT_TRUE(poly_dtype_eq(folded->src[i]->dtype, POLY_WEAKINT));
    ASSERT_INT_EQ(folded->src[i]->arg.i, i * 4);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, nonconstant_stack_is_not_constant_folded_like_current_tinygrad) {
  /* Current symbolic.py:31-38 returns None unless every STACK child is CONST. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *lhs_lanes[2] = {
      poly_uop0(ctx, POLY_OP_PARAM, POLY_FLOAT32, poly_arg_int(0)),
      poly_uop0(ctx, POLY_OP_PARAM, POLY_FLOAT32, poly_arg_int(1)),
  };
  PolyUOp *rhs_lanes[2] = {
      poly_uop0(ctx, POLY_OP_PARAM, POLY_FLOAT32, poly_arg_int(2)),
      poly_uop0(ctx, POLY_OP_PARAM, POLY_FLOAT32, poly_arg_int(3)),
  };
  PolyUOp *lhs = poly_uop_stack(ctx, lhs_lanes, 2);
  PolyUOp *rhs = poly_uop_stack(ctx, rhs_lanes, 2);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, lhs, rhs, poly_arg_none());

  PolyUOp *rewritten = poly_graph_rewrite(ctx, add, poly_symbolic_simple());
  ASSERT_PTR_EQ(rewritten, add);
  ASSERT_INT_EQ(rewritten->op, POLY_OP_ADD);
  ASSERT_PTR_EQ(rewritten->src[0], lhs);
  ASSERT_PTR_EQ(rewritten->src[1], rhs);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, vector_compare_uint64_const_fold_uses_raw_args) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a0 = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT64, poly_arg_int(-3));
  PolyUOp *a1 = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT64, poly_arg_int(4));
  PolyUOp *b0 = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT64, poly_arg_int(2));
  PolyUOp *b1 = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT64, poly_arg_int(-5));
  PolyUOp *avec_src[2] = {a0, a1};
  PolyUOp *bvec_src[2] = {b0, b1};
  PolyUOp *avec = poly_uop_stack(ctx, avec_src, 2);
  PolyUOp *bvec = poly_uop_stack(ctx, bvec_src, 2);
  PolyUOp *cmp = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, avec, bvec, poly_arg_none());

  PolyUOp *r = poly_graph_rewrite(ctx, cmp, poly_symbolic());
  ASSERT_TRUE(r != NULL);
  ASSERT_TRUE(r->op == POLY_OP_STACK);
  ASSERT_INT_EQ(r->n_src, 2);
  ASSERT_TRUE(r->src[0]->op == POLY_OP_CONST && r->src[0]->arg.kind == POLY_ARG_BOOL);
  ASSERT_TRUE(r->src[1]->op == POLY_OP_CONST && r->src[1]->arg.kind == POLY_ARG_BOOL);
  ASSERT_TRUE(r->src[0]->arg.b == true);
  ASSERT_TRUE(r->src[1]->arg.b == false);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, negative_integer_pow_const_folding_normalizes_or_refuses_like_pinned) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *two_i32 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(2));
  PolyUOp *zero_i32 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *neg_one_i32 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(-1));

  /* symbolic.py:30-32 evaluates 0.5, then UOp.const_like/DType.const turns it
   * into the integer CONST 0. */
  PolyUOp *finite =
      simplify(ctx, poly_uop2(ctx, POLY_OP_POW, POLY_INT32, two_i32, neg_one_i32, poly_arg_none()));
  ASSERT_NOT_NULL(finite);
  ASSERT_TRUE(finite->op == POLY_OP_CONST && finite->arg.kind == POLY_ARG_INT);
  ASSERT_INT_EQ(finite->arg.i, 0);

  /* Pinned DType.const raises on inf. C has no exception carrier, so retain
   * the POW graph instead of manufacturing an Invalid CONST. */
  PolyUOp *infinite_root =
      poly_uop2(ctx, POLY_OP_POW, POLY_INT32, zero_i32, neg_one_i32, poly_arg_none());
  PolyUOp *infinite = simplify(ctx, infinite_root);
  ASSERT_NOT_NULL(infinite);
  ASSERT_TRUE(infinite->op == POLY_OP_POW);

  PolyUOp *two_src[2] = {two_i32, two_i32};
  PolyUOp *neg_src[2] = {neg_one_i32, neg_one_i32};
  PolyUOp *vtwo = poly_uop_stack(ctx, two_src, 2);
  PolyUOp *vneg = poly_uop_stack(ctx, neg_src, 2);
  PolyUOp *fixed_root = poly_uop2(ctx, POLY_OP_POW, POLY_INT32, vtwo, vneg, poly_arg_none());
  PolyUOp *fixed = simplify(ctx, fixed_root);
  ASSERT_NOT_NULL(fixed);
  ASSERT_TRUE(fixed->op == POLY_OP_POW);

  PolyUOp *two_idx = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(2));
  PolyUOp *neg_one_idx = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(-1));
  PolyUOp *two_idx_src[2] = {two_idx, two_idx};
  PolyUOp *neg_idx_src[2] = {neg_one_idx, neg_one_idx};
  PolyUOp *weak_two = poly_uop_stack(ctx, two_idx_src, 2);
  PolyUOp *weak_neg = poly_uop_stack(ctx, neg_idx_src, 2);
  PolyUOp *weak =
      simplify(ctx, poly_uop2(ctx, POLY_OP_POW, POLY_WEAKINT, weak_two, weak_neg, poly_arg_none()));
  ASSERT_NOT_NULL(weak);
  ASSERT_TRUE(weak->op == POLY_OP_STACK && weak->n_src == 2);
  for (int i = 0; i < 2; i++) {
    ASSERT_TRUE(weak->src[i]->op == POLY_OP_CONST);
    ASSERT_TRUE(poly_dtype_is_index(weak->src[i]->dtype));
    ASSERT_TRUE(weak->src[i]->arg.kind == POLY_ARG_INT);
    ASSERT_INT_EQ(weak->src[i]->arg.i, 0);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, uint64_vector_high_bit_intermediate_matches_pinned_exact_fold) {
  /* Pinned uop/ops.py:1192-1197 truncates each vector lane recursively:
   * (INT64_MAX+1)>>1 stays the positive uint64 value 2**62. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *max = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT64, poly_arg_int(INT64_MAX));
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT64, poly_arg_int(1));
  PolyUOp *max_src[2] = {max, max};
  PolyUOp *one_src[2] = {one, one};
  PolyUOp *vmax = poly_uop_stack(ctx, max_src, 2);
  PolyUOp *vone = poly_uop_stack(ctx, one_src, 2);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_UINT64, vmax, vone, poly_arg_none());
  PolyUOp *root = poly_uop2(ctx, POLY_OP_SHR, POLY_UINT64, add, vone, poly_arg_none());
  PolyUOp *folded = simplify(ctx, root);
  ASSERT_NOT_NULL(folded);
  ASSERT_TRUE(folded->op == POLY_OP_STACK);
  ASSERT_INT_EQ(folded->n_src, 2);
  ASSERT_TRUE(folded->src[0] == folded->src[1]);
  ASSERT_TRUE(integer_const_eq(folded->src[0], "4611686018427387904"));
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, negative_shift_count_does_not_fold_to_invalid_const) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(1));
  PolyUOp *neg = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(-1));
  PolyUOp *shl = poly_uop2(ctx, POLY_OP_SHL, POLY_INT32, one, neg, poly_arg_none());

  PolyUOp *r = poly_graph_rewrite(ctx, shl, poly_symbolic());
  ASSERT_TRUE(r != NULL);
  ASSERT_TRUE(r->op == POLY_OP_SHL);
  ASSERT_TRUE(!(r->op == POLY_OP_CONST && r->arg.kind == POLY_ARG_INVALID));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, vector_negative_shift_count_does_not_fold_to_invalid_stack) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(1));
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(2));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *neg = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(-1));
  PolyUOp *lhs_src[2] = {one, two};
  PolyUOp *rhs_src[2] = {zero, neg};
  PolyUOp *lhs = poly_uop_stack(ctx, lhs_src, 2);
  PolyUOp *rhs = poly_uop_stack(ctx, rhs_src, 2);
  PolyUOp *shl = poly_uop2(ctx, POLY_OP_SHL, POLY_INT32, lhs, rhs, poly_arg_none());

  PolyUOp *r = poly_graph_rewrite(ctx, shl, poly_symbolic());
  ASSERT_TRUE(r != NULL);
  ASSERT_TRUE(r->op == POLY_OP_SHL);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, vector_float16_const_fold_uses_pinned_lane_truncation) {
  /* tinygrad ops.py:1192-1197 does not forward truncate_output through vector
   * recursion, so fixed-width lanes use the default truncating execution. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT16, poly_arg_float(1.0));
  PolyUOp *small = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT16, poly_arg_float(0.0001));
  PolyUOp *lhs_src[2] = {one, one};
  PolyUOp *rhs_src[2] = {small, small};
  PolyUOp *lhs = poly_uop_stack(ctx, lhs_src, 2);
  PolyUOp *rhs = poly_uop_stack(ctx, rhs_src, 2);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT16, lhs, rhs, poly_arg_none());

  PolyUOp *folded = poly_graph_rewrite(ctx, add, poly_symbolic_simple());
  ASSERT_NOT_NULL(folded);
  ASSERT_INT_EQ(folded->op, POLY_OP_STACK);
  ASSERT_INT_EQ(folded->n_src, 2);
  for (int i = 0; i < folded->n_src; i++) {
    ASSERT_INT_EQ(folded->src[i]->op, POLY_OP_CONST);
    ASSERT_TRUE(poly_dtype_eq(folded->src[i]->dtype, POLY_FLOAT16));
    ASSERT_TRUE(folded->src[i]->arg.kind == POLY_ARG_FLOAT);
    ASSERT_FLOAT_EQ(folded->src[i]->arg.f, 1.0, 0.0);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

static bool integer_const_eq(PolyUOp *u, const char *expected) {
  if (!u || u->op != POLY_OP_CONST ||
      (u->arg.kind != POLY_ARG_INT && u->arg.kind != POLY_ARG_BIGINT))
    return false;
  char *actual = poly_arg_integer_to_decimal(u->arg);
  bool equal = actual && strcmp(actual, expected) == 0;
  free(actual);
  return equal;
}

TEST(sym, exact_symbolic_integer_results_match_pinned_python_topology) {
  /* Pinned tinygrad uop/symbolic.py:30-32,252,260-262 evaluates these
   * constants as exact Python integers. The approved PolyBigInt carrier must
   * preserve both their values and the resulting rewrite topology. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *max = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(INT64_MAX));
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(2));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *var = Variable(ctx, "x", 0, 1, POLY_WEAKINT);

  PolyUOp *direct =
      simplify(ctx, poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, max, one, poly_arg_none()));
  ASSERT_NOT_NULL(direct);
  ASSERT_TRUE(integer_const_eq(direct, "9223372036854775808"));

  PolyUOp *assoc = poly_graph_rewrite(
      ctx,
      poly_uop2(
          ctx, POLY_OP_ADD, POLY_WEAKINT,
          poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, var, max, poly_arg_none()), one, poly_arg_none()
      ),
      poly_symbolic()
  );
  ASSERT_NOT_NULL(assoc);
  ASSERT_INT_EQ(assoc->op, POLY_OP_ADD);
  ASSERT_TRUE(assoc->src[0] == var);
  ASSERT_TRUE(integer_const_eq(assoc->src[1], "9223372036854775808"));

  PolyUOp *distributed = poly_graph_rewrite(
      ctx,
      poly_uop2(
          ctx, POLY_OP_MUL, POLY_WEAKINT, two,
          poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, var, max, poly_arg_none()), poly_arg_none()
      ),
      poly_symbolic()
  );
  ASSERT_NOT_NULL(distributed);
  ASSERT_INT_EQ(distributed->op, POLY_OP_ADD);
  ASSERT_INT_EQ(distributed->src[0]->op, POLY_OP_MUL);
  ASSERT_TRUE(distributed->src[0]->src[0] == var);
  ASSERT_TRUE(integer_const_eq(distributed->src[0]->src[1], "2"));
  ASSERT_TRUE(integer_const_eq(distributed->src[1], "18446744073709551614"));

  PolyUOp *max_src[2] = {max, max};
  PolyUOp *one_src[2] = {one, one};
  PolyUOp *vmax = poly_uop_stack(ctx, max_src, 2);
  PolyUOp *vone = poly_uop_stack(ctx, one_src, 2);
  PolyUOp *vector =
      simplify(ctx, poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, vmax, vone, poly_arg_none()));
  ASSERT_NOT_NULL(vector);
  ASSERT_INT_EQ(vector->op, POLY_OP_STACK);
  ASSERT_INT_EQ(vector->n_src, 2);
  ASSERT_TRUE(vector->src[0] == vector->src[1]);
  ASSERT_TRUE(integer_const_eq(vector->src[0], "9223372036854775808"));

  PolyUOp *min = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(INT64_MIN));
  PolyUOp *neg = simplify(ctx, poly_uop1(ctx, POLY_OP_NEG, POLY_WEAKINT, min, poly_arg_none()));
  ASSERT_NOT_NULL(neg);
  ASSERT_TRUE(integer_const_eq(neg, "9223372036854775808"));

  PolyUOp *mulacc =
      simplify(ctx, poly_uop3(ctx, POLY_OP_MULACC, POLY_WEAKINT, max, two, zero, poly_arg_none()));
  ASSERT_NOT_NULL(mulacc);
  ASSERT_TRUE(integer_const_eq(mulacc, "18446744073709551614"));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, codegen_sym_distributes_weak_index_add_like_tinygrad) {
  /* Pinned tinygrad/uop/symbolic.py:454,488 keeps this rule in `sym`, not
   * `symbolic`. It exposes every split RANGE stride before expansion while
   * leaving concrete integer and floating-point association unchanged. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *c64 = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(64));
  PolyUOp *c16 = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(16));
  PolyUOp *c3 = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(3));
  PolyUOp *c1024 = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1024));
  PolyUOp *global =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, c64, poly_arg_range(2, POLY_AXIS_GLOBAL));
  PolyUOp *local =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, c16, poly_arg_range(5, POLY_AXIS_LOCAL));
  PolyUOp *upcast =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, c3, poly_arg_range(3, POLY_AXIS_UPCAST));
  PolyUOp *global16 = poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, global, c16, poly_arg_none());
  PolyUOp *global_local =
      poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, global16, local, poly_arg_none());
  PolyUOp *scaled = poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, global_local, c3, poly_arg_none());
  PolyUOp *with_upcast = poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, scaled, upcast, poly_arg_none());
  PolyUOp *root = poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, with_upcast, c1024, poly_arg_none());

  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, root, poly_symbolic()), root);
  PolyUOp *rewritten = poly_graph_rewrite(ctx, root, poly_sym());
  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(rewritten->op, POLY_OP_ADD);
  ASSERT_INT_EQ(rewritten->n_src, 2);
  ASSERT_INT_EQ(rewritten->src[0]->op, POLY_OP_ADD);
  ASSERT_INT_EQ(rewritten->src[0]->n_src, 2);
  ASSERT_INT_EQ(rewritten->src[0]->src[0]->op, POLY_OP_MUL);
  ASSERT_PTR_EQ(rewritten->src[0]->src[0]->src[0], global);
  ASSERT_TRUE(integer_const_eq(rewritten->src[0]->src[0]->src[1], "49152"));
  ASSERT_INT_EQ(rewritten->src[0]->src[1]->op, POLY_OP_MUL);
  ASSERT_PTR_EQ(rewritten->src[0]->src[1]->src[0], local);
  ASSERT_TRUE(integer_const_eq(rewritten->src[0]->src[1]->src[1], "3072"));
  ASSERT_INT_EQ(rewritten->src[1]->op, POLY_OP_MUL);
  ASSERT_PTR_EQ(rewritten->src[1]->src[0], upcast);
  ASSERT_TRUE(integer_const_eq(rewritten->src[1]->src[1], "1024"));

  PolyUOp *i32_x = poly_uop0(ctx, POLY_OP_PARAM, POLY_INT32, poly_arg_int(0));
  PolyUOp *i32_y = poly_uop0(ctx, POLY_OP_PARAM, POLY_INT32, poly_arg_int(1));
  PolyUOp *i32_four = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(4));
  PolyUOp *i32_root = poly_uop2(
      ctx, POLY_OP_MUL, POLY_INT32,
      poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, i32_x, i32_y, poly_arg_none()), i32_four,
      poly_arg_none()
  );
  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, i32_root, poly_sym()), i32_root);

  PolyUOp *f_x = poly_uop0(ctx, POLY_OP_PARAM, POLY_FLOAT32, poly_arg_int(2));
  PolyUOp *f_y = poly_uop0(ctx, POLY_OP_PARAM, POLY_FLOAT32, poly_arg_int(3));
  PolyUOp *f_four = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(4.0));
  PolyUOp *f_root = poly_uop2(
      ctx, POLY_OP_MUL, POLY_FLOAT32,
      poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, f_x, f_y, poly_arg_none()), f_four, poly_arg_none()
  );
  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, f_root, poly_sym()), f_root);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, codegen_sym_flattens_group_under_sink) {
  /* tinygrad@2026-08-22/a9069c177a9d uop/symbolic.py:431-439. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *left = poly_uop0(ctx, POLY_OP_CUSTOM, POLY_VOID, poly_arg_str("left"));
  PolyUOp *right = poly_uop0(ctx, POLY_OP_CUSTOM, POLY_VOID, poly_arg_str("right"));
  PolyUOp *group_src[2] = {left, right};
  PolyUOp *group = poly_uop(ctx, POLY_OP_GROUP, POLY_VOID, group_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, group, poly_arg_none());

  PolyUOp *rewritten = poly_graph_rewrite(ctx, sink, poly_sym());
  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(rewritten->op, POLY_OP_SINK);
  ASSERT_INT_EQ(rewritten->n_src, 2);
  ASSERT_PTR_EQ(rewritten->src[0], left);
  ASSERT_PTR_EQ(rewritten->src[1], right);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, singleton_end_is_its_value) {
  /* tinygrad@2026-08-22/a9069c177a9d uop/symbolic.py:306-308. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *value = poly_uop0(ctx, POLY_OP_CUSTOM, POLY_VOID, poly_arg_str("value"));
  PolyUOp *end = poly_uop1(ctx, POLY_OP_END, POLY_VOID, value, poly_arg_none());

  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, end, poly_symbolic()), value);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, symbolic_negated_add_distributes_like_tinygrad) {
  /* Pinned tinygrad/uop/symbolic.py:250 distributes the exact -1
   * coefficient for concrete ints and floats. Line 251 separately restricts
   * arbitrary coefficients to weakint, so concrete (x+1)*2 must remain. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *x_i = Variable(ctx, "x_i", 0, 1186, POLY_INT32);
  PolyUOp *one_i = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(1));
  PolyUOp *neg_i = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(-1));
  PolyUOp *two_i = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(2));
  PolyUOp *add_i = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, x_i, one_i, poly_arg_none());
  PolyUOp *negated_i = poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, add_i, neg_i, poly_arg_none());
  PolyUOp *scaled_i = poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, add_i, two_i, poly_arg_none());

  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, negated_i, poly_symbolic_simple()), negated_i);
  PolyUOp *rewritten_i = poly_graph_rewrite(ctx, negated_i, poly_symbolic());
  ASSERT_NOT_NULL(rewritten_i);
  ASSERT_INT_EQ(rewritten_i->op, POLY_OP_ADD);
  ASSERT_INT_EQ(rewritten_i->n_src, 2);
  ASSERT_INT_EQ(rewritten_i->src[0]->op, POLY_OP_MUL);
  ASSERT_PTR_EQ(rewritten_i->src[0]->src[0], x_i);
  ASSERT_PTR_EQ(rewritten_i->src[0]->src[1], neg_i);
  ASSERT_PTR_EQ(rewritten_i->src[1], neg_i);
  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, scaled_i, poly_symbolic()), scaled_i);

  PolyUOp *x_f = Variable(ctx, "x_f", 0, 1186, POLY_FLOAT32);
  PolyUOp *one_f = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *neg_f = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(-1.0));
  PolyUOp *add_f = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, x_f, one_f, poly_arg_none());
  PolyUOp *negated_f = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, add_f, neg_f, poly_arg_none());
  PolyUOp *rewritten_f = poly_graph_rewrite(ctx, negated_f, poly_symbolic());
  ASSERT_NOT_NULL(rewritten_f);
  ASSERT_INT_EQ(rewritten_f->op, POLY_OP_ADD);
  ASSERT_INT_EQ(rewritten_f->n_src, 2);
  ASSERT_INT_EQ(rewritten_f->src[0]->op, POLY_OP_MUL);
  ASSERT_PTR_EQ(rewritten_f->src[0]->src[0], x_f);
  ASSERT_PTR_EQ(rewritten_f->src[0]->src[1], neg_f);
  ASSERT_PTR_EQ(rewritten_f->src[1], neg_f);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, codegen_sym_moves_range_independent_reduce_factors_like_tinygrad) {
  /* Pinned tinygrad/uop/symbolic.py:386-395,483-484 partitions a MUL chain by
   * exact REDUCE-range reachability. Only dependent factors stay inside. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *c8 = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(8));
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, c8, poly_arg_range(0, POLY_AXIS_REDUCE));
  PolyUOp *varying = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, range, poly_arg_none());
  PolyUOp *outside = poly_uop0(ctx, POLY_OP_PARAM, POLY_FLOAT32, poly_arg_int(0));
  PolyUOp *value = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, varying, outside, poly_arg_none());
  PolyUOp *reduce_src[2] = {value, range};
  PolyArg tag_arg = poly_arg_str("reduce-parity");
  PolyUOp *reduce = poly_uop_tagged_arg(
      ctx, POLY_OP_REDUCE, POLY_FLOAT32, reduce_src, 2, poly_arg_reduce(POLY_OP_ADD, 0), 17, tag_arg
  );

  PolyUOp *rewritten = poly_graph_rewrite(ctx, reduce, poly_sym());
  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(rewritten->op, POLY_OP_MUL);
  ASSERT_INT_EQ(rewritten->n_src, 2);
  ASSERT_INT_EQ(rewritten->src[0]->op, POLY_OP_REDUCE);
  ASSERT_PTR_EQ(rewritten->src[0]->src[0], varying);
  ASSERT_PTR_EQ(rewritten->src[0]->src[1], range);
  ASSERT_INT_EQ(rewritten->src[0]->tag, 17);
  ASSERT_TRUE(poly_arg_eq(rewritten->src[0]->tag_arg, tag_arg));
  ASSERT_PTR_EQ(rewritten->src[1], outside);

  /* A factor that reaches the reduction RANGE is not movable. */
  PolyUOp *dependent_value =
      poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, varying, varying, poly_arg_none());
  PolyUOp *dependent_src[2] = {dependent_value, range};
  PolyUOp *dependent = poly_uop(
      ctx, POLY_OP_REDUCE, POLY_FLOAT32, dependent_src, 2, poly_arg_reduce(POLY_OP_ADD, 0)
  );
  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, dependent, poly_sym()), dependent);

  /* MAX only admits a proved non-negative invariant factor. */
  PolyUOp *negative = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(-2.0));
  PolyUOp *negative_value =
      poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, varying, negative, poly_arg_none());
  PolyUOp *negative_src[2] = {negative_value, range};
  PolyUOp *max_negative =
      poly_uop(ctx, POLY_OP_REDUCE, POLY_FLOAT32, negative_src, 2, poly_arg_reduce(POLY_OP_MAX, 0));
  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, max_negative, poly_sym()), max_negative);
  PolyUOp *positive = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(2.0));
  PolyUOp *positive_src[2] = {
      poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, varying, positive, poly_arg_none()), range};
  PolyUOp *max_positive =
      poly_uop(ctx, POLY_OP_REDUCE, POLY_FLOAT32, positive_src, 2, poly_arg_reduce(POLY_OP_MAX, 0));
  PolyUOp *max_rewritten = poly_graph_rewrite(ctx, max_positive, poly_sym());
  ASSERT_INT_EQ(max_rewritten->op, POLY_OP_MUL);
  ASSERT_INT_EQ(max_rewritten->src[0]->op, POLY_OP_REDUCE);
  ASSERT_PTR_EQ(max_rewritten->src[1], positive);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, integral_trunc_is_identity_before_rendering) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyDType integral_dtypes[] = {
      POLY_INT32,   POLY_UINT32, POLY_INT64, POLY_BOOL,
      POLY_WEAKINT, POLY_INT32,  POLY_BOOL,  POLY_WEAKINT,
  };
  for (int i = 0; i < (int)(sizeof(integral_dtypes) / sizeof(integral_dtypes[0])); i++) {
    PolyUOp *x = poly_uop0(ctx, POLY_OP_PARAM, integral_dtypes[i], poly_arg_int(i));
    PolyUOp *trunc = poly_uop1(ctx, POLY_OP_TRUNC, integral_dtypes[i], x, poly_arg_none());
    ASSERT_TRUE(simplify(ctx, trunc) == x);
  }

  PolyDType float_dtypes[] = {
      POLY_FLOAT32,
      POLY_FLOAT64,
      POLY_FLOAT32,
  };
  for (int i = 0; i < (int)(sizeof(float_dtypes) / sizeof(float_dtypes[0])); i++) {
    PolyUOp *x = poly_uop0(ctx, POLY_OP_PARAM, float_dtypes[i], poly_arg_int(32 + i));
    PolyUOp *trunc = poly_uop1(ctx, POLY_OP_TRUNC, float_dtypes[i], x, poly_arg_none());
    PolyUOp *rewritten = simplify(ctx, trunc);
    ASSERT_NOT_NULL(rewritten);
    ASSERT_INT_EQ(rewritten->op, POLY_OP_TRUNC);
    ASSERT_TRUE(rewritten->src[0] == x);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, integral_trunc_preserves_invalid_gate_ordering) {
  /* Pinned tinygrad/uop/symbolic.py:60-86 propagates Invalid before the
   * line-114 integral TRUNC identity. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *gate = poly_uop0(ctx, POLY_OP_PARAM, POLY_BOOL, poly_arg_int(0));
  PolyUOp *x = poly_uop0(ctx, POLY_OP_PARAM, POLY_WEAKINT, poly_arg_int(1));
  PolyUOp *invalid = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_invalid());
  PolyUOp *masked = poly_uop3(ctx, POLY_OP_WHERE, POLY_WEAKINT, gate, x, invalid, poly_arg_none());
  PolyUOp *trunc = poly_uop1(ctx, POLY_OP_TRUNC, POLY_WEAKINT, masked, poly_arg_none());

  PolyUOp *rewritten = simplify(ctx, trunc);
  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(rewritten->op, POLY_OP_WHERE);
  ASSERT_TRUE(rewritten->src[0] == gate);
  ASSERT_TRUE(rewritten->src[1] == x);
  ASSERT_TRUE(rewritten->src[2] == invalid);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, cast_and_bitcast_preserve_invalid_gate) {
  /* Tinygrad 2026-08-22/a9069c177a9d uop/symbolic.py:74-78 keeps
   * Invalid outside CAST/BITCAST so late gater can recover the predicate. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *gate = poly_uop0(ctx, POLY_OP_PARAM, POLY_BOOL, poly_arg_int(0));

  PolyUOp *weak_value = poly_uop0(ctx, POLY_OP_PARAM, POLY_WEAKINT, poly_arg_int(1));
  PolyUOp *weak_invalid = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_invalid());
  PolyUOp *weak_masked =
      poly_uop3(ctx, POLY_OP_WHERE, POLY_WEAKINT, gate, weak_value, weak_invalid, poly_arg_none());
  PolyUOp *casted =
      simplify(ctx, poly_uop1(ctx, POLY_OP_CAST, POLY_INT32, weak_masked, poly_arg_none()));
  ASSERT_NOT_NULL(casted);
  ASSERT_INT_EQ(casted->op, POLY_OP_WHERE);
  ASSERT_TRUE(casted->src[0] == gate);
  ASSERT_INT_EQ(casted->src[1]->op, POLY_OP_CAST);
  ASSERT_TRUE(casted->src[1]->src[0] == weak_value);
  ASSERT_TRUE(casted->src[2] == weak_invalid);
  ASSERT_TRUE(
      simplify(ctx, poly_uop1(ctx, POLY_OP_CAST, POLY_INT32, weak_invalid, poly_arg_none())) ==
      weak_invalid
  );

  PolyUOp *bits = poly_uop0(ctx, POLY_OP_PARAM, POLY_INT32, poly_arg_int(2));
  PolyUOp *int_invalid = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_invalid());
  PolyUOp *bits_masked =
      poly_uop3(ctx, POLY_OP_WHERE, POLY_INT32, gate, bits, int_invalid, poly_arg_none());
  PolyUOp *bitcasted =
      simplify(ctx, poly_uop1(ctx, POLY_OP_BITCAST, POLY_FLOAT32, bits_masked, poly_arg_none()));
  ASSERT_NOT_NULL(bitcasted);
  ASSERT_INT_EQ(bitcasted->op, POLY_OP_WHERE);
  ASSERT_TRUE(bitcasted->src[0] == gate);
  ASSERT_INT_EQ(bitcasted->src[1]->op, POLY_OP_BITCAST);
  ASSERT_TRUE(bitcasted->src[1]->src[0] == bits);
  ASSERT_TRUE(bitcasted->src[2] == int_invalid);
  ASSERT_TRUE(
      simplify(ctx, poly_uop1(ctx, POLY_OP_BITCAST, POLY_FLOAT32, int_invalid, poly_arg_none())) ==
      int_invalid
  );

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, invalid_where_propagates_through_binary_before_zero_folding) {
  /*
   * Pinned tinygrad uop/symbolic.py:60-66 keeps Invalid as an index-validity
   * sentinel by lifting it through ALU before ordinary identities such as
   * x*0 -> 0.  This exact topology is consumed later by
   * codegen/late/gater.py:5-17.
   */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *gate = poly_uop0(ctx, POLY_OP_PARAM, POLY_BOOL, poly_arg_int(0));
  PolyUOp *x = poly_uop0(ctx, POLY_OP_PARAM, POLY_WEAKINT, poly_arg_int(1));
  PolyUOp *invalid = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_invalid());
  PolyUOp *masked = poly_uop3(ctx, POLY_OP_WHERE, POLY_WEAKINT, gate, x, invalid, poly_arg_none());

  const int64_t factors[] = {2, 0};
  for (int i = 0; i < 2; i++) {
    PolyUOp *factor = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(factors[i]));
    PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, masked, factor, poly_arg_none());
    PolyUOp *r = poly_graph_rewrite(ctx, mul, poly_symbolic());
    ASSERT_NOT_NULL(r);
    ASSERT_INT_EQ(r->op, POLY_OP_WHERE);
    ASSERT_TRUE(r->src[0] == gate);
    ASSERT_INT_EQ(r->src[2]->op, POLY_OP_CONST);
    ASSERT_INT_EQ(r->src[2]->arg.kind, POLY_ARG_INVALID);
    if (factors[i] == 2) {
      ASSERT_INT_EQ(r->src[1]->op, POLY_OP_MUL);
      ASSERT_TRUE(r->src[1]->src[0] == x);
    } else {
      ASSERT_INT_EQ(r->src[1]->op, POLY_OP_CONST);
      ASSERT_INT_EQ(r->src[1]->arg.i, 0);
    }
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, nested_where_lifts_invalid_before_zero_branch_merge) {
  /* Tinygrad 2026-08-22/a9069c177a9d uop/symbolic.py:83-91 keeps the
   * Invalid branch outside nested WHEREs so late gater can recover validity. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = Variable(ctx, "a", 0, 1, POLY_BOOL);
  PolyUOp *b = Variable(ctx, "b", 0, 1, POLY_BOOL);
  PolyUOp *x = Variable(ctx, "x", 0, 31, POLY_WEAKINT);
  PolyUOp *invalid = poly_uop_const(ctx, poly_arg_invalid(), POLY_WEAKINT);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *inner = poly_uop3(ctx, POLY_OP_WHERE, POLY_WEAKINT, b, x, invalid, poly_arg_none());
  PolyUOp *root = poly_uop3(ctx, POLY_OP_WHERE, POLY_WEAKINT, a, inner, zero, poly_arg_none());

  PolyUOp *out = poly_graph_rewrite(ctx, root, poly_symbolic());
  ASSERT_NOT_NULL(out);
  ASSERT_TRUE(poly_dtype_eq(invalid->dtype, POLY_BOOL));
  ASSERT_INT_EQ(out->op, POLY_OP_WHERE);
  ASSERT_INT_EQ(out->src[0]->op, POLY_OP_OR);
  ASSERT_INT_EQ(out->src[0]->src[0]->op, POLY_OP_CMPNE);
  ASSERT_TRUE(out->src[0]->src[0]->src[0] == a);
  ASSERT_TRUE(out->src[0]->src[1] == b);
  ASSERT_INT_EQ(out->src[1]->op, POLY_OP_WHERE);
  ASSERT_TRUE(out->src[1]->src[0] == a);
  ASSERT_TRUE(out->src[1]->src[1] == x);
  ASSERT_TRUE(out->src[1]->src[2] == zero);
  ASSERT_INT_EQ(out->src[2]->arg.kind, POLY_ARG_INVALID);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, comparison_preserves_invalid_gate) {
  /* Tinygrad 2026-08-22/a9069c177a9d uop/symbolic.py:79-82 applies the
   * GroupOp.Binary rule to comparisons and retains the Invalid false lane. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *gate = Variable(ctx, "gate", 0, 1, POLY_BOOL);
  PolyUOp *x = Variable(ctx, "x", 0, 31, POLY_WEAKINT);
  PolyUOp *invalid = poly_uop_const(ctx, poly_arg_invalid(), POLY_WEAKINT);
  PolyUOp *masked = poly_uop3(ctx, POLY_OP_WHERE, POLY_WEAKINT, gate, x, invalid, poly_arg_none());
  PolyUOp *five = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(5));
  PolyUOp *comparison = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, masked, five, poly_arg_none());

  PolyUOp *out = poly_graph_rewrite(ctx, comparison, poly_symbolic());
  ASSERT_NOT_NULL(out);
  ASSERT_TRUE(poly_dtype_eq(invalid->dtype, POLY_BOOL));
  ASSERT_INT_EQ(out->op, POLY_OP_WHERE);
  ASSERT_TRUE(out->src[0] == gate);
  ASSERT_INT_EQ(out->src[1]->op, POLY_OP_CMPLT);
  ASSERT_TRUE(out->src[1]->src[0] == x);
  ASSERT_TRUE(out->src[1]->src[1] == five);
  ASSERT_TRUE(out->src[2] == invalid);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, invalid_condition_and_broadcast_poison_value) {
  /* Tinygrad 2026-08-22/a9069c177a9d uop/symbolic.py:73,84 makes a direct
   * Invalid condition or broadcast collapse to the scalar sentinel. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *invalid = poly_uop_const(ctx, poly_arg_invalid(), POLY_WEAKINT);
  PolyUOp *x = Variable(ctx, "x", 0, 31, POLY_WEAKINT);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *condition =
      poly_uop3(ctx, POLY_OP_WHERE, POLY_WEAKINT, invalid, x, zero, poly_arg_none());
  ASSERT_TRUE(poly_graph_rewrite(ctx, condition, poly_symbolic()) == invalid);

  PolyUOp *src[4] = {invalid, invalid, invalid, invalid};
  PolyUOp *broadcast = poly_uop(ctx, POLY_OP_STACK, POLY_BOOL, src, 4, poly_arg_none());
  ASSERT_TRUE(poly_graph_rewrite(ctx, broadcast, poly_symbolic()) == invalid);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, gated_invalid_condition_lifts_outer_gate) {
  /* Tinygrad 2026-08-22/a9069c177a9d uop/symbolic.py:85 lifts an Invalid
   * condition before late gater turns masked coordinates into LOAD gates. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *a = Variable(ctx, "a", 0, 1, POLY_BOOL);
  PolyUOp *b = Variable(ctx, "b", 0, 1, POLY_BOOL);
  PolyUOp *x = Variable(ctx, "x", 0, 31, POLY_WEAKINT);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *invalid = poly_uop_const(ctx, poly_arg_invalid(), POLY_BOOL);
  PolyUOp *masked_condition =
      poly_uop3(ctx, POLY_OP_WHERE, POLY_BOOL, b, a, invalid, poly_arg_none());
  PolyUOp *root =
      poly_uop3(ctx, POLY_OP_WHERE, POLY_WEAKINT, masked_condition, x, zero, poly_arg_none());

  PolyUOp *out = poly_graph_rewrite(ctx, root, poly_symbolic());
  ASSERT_NOT_NULL(out);
  ASSERT_INT_EQ(out->op, POLY_OP_WHERE);
  ASSERT_TRUE(out->src[0] == b);
  ASSERT_INT_EQ(out->src[1]->op, POLY_OP_WHERE);
  ASSERT_TRUE(out->src[1]->src[0] == a);
  ASSERT_TRUE(out->src[1]->src[1] == x);
  ASSERT_TRUE(out->src[1]->src[2] == zero);
  ASSERT_TRUE(out->src[2] == invalid);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, repeated_gate_eliminates_nested_invalid) {
  /* Tinygrad 2026-08-22/a9069c177a9d uop/symbolic.py:89 reduces
   * a.where(a.where(x, Invalid), zero) to a.where(x, zero). */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *a = Variable(ctx, "a", 0, 1, POLY_BOOL);
  PolyUOp *x = Variable(ctx, "x", 0, 31, POLY_WEAKINT);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *invalid = poly_uop_const(ctx, poly_arg_invalid(), POLY_BOOL);
  PolyUOp *inner = poly_uop3(ctx, POLY_OP_WHERE, POLY_WEAKINT, a, x, invalid, poly_arg_none());
  PolyUOp *root = poly_uop3(ctx, POLY_OP_WHERE, POLY_WEAKINT, a, inner, zero, poly_arg_none());

  PolyUOp *out = poly_graph_rewrite(ctx, root, poly_symbolic());
  ASSERT_NOT_NULL(out);
  ASSERT_INT_EQ(out->op, POLY_OP_WHERE);
  ASSERT_TRUE(out->src[0] == a);
  ASSERT_TRUE(out->src[1] == x);
  ASSERT_TRUE(out->src[2] == zero);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, given_valid_parses_constant_left_lower_bound) {
  /* Tinygrad 2026-08-22/a9069c177a9d uop/symbolic.py:316-319 parses
   * `0 < x` as x >= 1, making x < 1 false under that validity. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *x = Variable(ctx, "x", 0, 31, POLY_WEAKINT);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
  PolyUOp *lower = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, zero, x, poly_arg_none());
  PolyUOp *upper = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, x, one, poly_arg_none());

  PolyUOp *out = poly_uop_given_valid(ctx, lower, upper, true);
  ASSERT_NOT_NULL(out);
  ASSERT_INT_EQ(out->op, POLY_OP_CONST);
  ASSERT_INT_EQ(out->arg.kind, POLY_ARG_BOOL);
  ASSERT_FALSE(out->arg.b);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, given_valid_removes_modulo_from_bounded_hlb_coordinate) {
  /* Tinygrad 2026-08-22/a9069c177a9d uop/symbolic.py:327-363 uses all
   * parsed validity bounds together.  Under `(row % 10) < 8`, the HLB
   * coordinate `row % 10` becomes `row`. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *three = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(3));
  PolyUOp *eight = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(8));
  PolyUOp *nine = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(9));
  PolyUOp *ten = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(10));
  PolyUOp *twenty = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(20));
  PolyUOp *gidx0 = poly_uop1(ctx, POLY_OP_SPECIAL, POLY_WEAKINT, nine, poly_arg_str("gidx0"));
  PolyUOp *lidx1 = poly_uop1(ctx, POLY_OP_SPECIAL, POLY_WEAKINT, three, poly_arg_str("lidx1"));
  PolyUOp *lidx2 = poly_uop1(ctx, POLY_OP_SPECIAL, POLY_WEAKINT, three, poly_arg_str("lidx2"));
  PolyUOp *gmod = poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_WEAKINT, gidx0, three, poly_arg_none());
  PolyUOp *gdiv = poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_WEAKINT, gidx0, three, poly_arg_none());
  PolyUOp *row = poly_uop2(
      ctx, POLY_OP_ADD, POLY_WEAKINT,
      poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, gmod, three, poly_arg_none()), lidx1,
      poly_arg_none()
  );
  PolyUOp *col = poly_uop2(
      ctx, POLY_OP_ADD, POLY_WEAKINT,
      poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, gdiv, three, poly_arg_none()), lidx2,
      poly_arg_none()
  );
  PolyUOp *row_mod = poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_WEAKINT, row, ten, poly_arg_none());
  PolyUOp *col_mod = poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_WEAKINT, col, ten, poly_arg_none());
  PolyUOp *clauses[4] = {
      poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, col, twenty, poly_arg_none()),
      poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, row, twenty, poly_arg_none()),
      poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, col_mod, eight, poly_arg_none()),
      poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, row_mod, eight, poly_arg_none()),
  };
  PolyUOp *valid = clauses[0];
  for (int i = 1; i < 4; i++)
    valid = poly_uop2(ctx, POLY_OP_AND, POLY_BOOL, valid, clauses[i], poly_arg_none());

  PolyUOp *out = poly_uop_given_valid(ctx, valid, row_mod, true);
  PolyUOp *expected = poly_graph_rewrite(ctx, row, poly_symbolic());
  ASSERT_PTR_EQ(out, expected);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, given_valid_drops_repeated_known_bounds_from_gated_value) {
  /* tinygrad 2026-08-22/a9069c177a9d uop/symbolic.py:326-363 constrains
   * every parsed expression together, so known upper bounds fold to True. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = Variable(ctx, "x", 0, 39, POLY_WEAKINT);
  PolyUOp *y = Variable(ctx, "y", 0, 3, POLY_WEAKINT);
  PolyUOp *ten = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(10));
  PolyUOp *twenty = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(20));
  PolyUOp *eight = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(8));
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
  PolyUOp *a = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, x, twenty, poly_arg_none());
  PolyUOp *mod = poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_WEAKINT, x, ten, poly_arg_none());
  PolyUOp *b = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, mod, eight, poly_arg_none());
  PolyUOp *e = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, y, one, poly_arg_none());
  PolyUOp *known = poly_uop2(ctx, POLY_OP_AND, POLY_BOOL, a, b, poly_arg_none());
  PolyUOp *value = poly_uop2(
      ctx, POLY_OP_AND, POLY_BOOL,
      poly_uop2(
          ctx, POLY_OP_AND, POLY_BOOL,
          poly_uop2(ctx, POLY_OP_AND, POLY_BOOL, a, b, poly_arg_none()), b, poly_arg_none()
      ),
      e, poly_arg_none()
  );

  PolyUOp *out = poly_uop_given_valid(ctx, known, value, false);
  ASSERT_PTR_EQ(out, e);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, given_valid_false_substitutes_overlapping_bounds_together) {
  /* Tinygrad 2026-08-22/a9069c177a9d uop/symbolic.py:327-363 performs
   * only the final all-candidate substitution when try_simplex is false. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *x = Variable(ctx, "x", 0, 15, POLY_WEAKINT);
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(2));
  PolyUOp *value = poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, x, two, poly_arg_none());
  PolyUOp *x_bound = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, x, one, poly_arg_none());
  PolyUOp *value_bound = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, value, two, poly_arg_none());
  PolyUOp *valid = poly_uop2(ctx, POLY_OP_AND, POLY_BOOL, x_bound, value_bound, poly_arg_none());

  ASSERT_PTR_EQ(poly_uop_given_valid(ctx, valid, value, false), value);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, given_valid_has_no_clause_ceiling) {
  /* Tinygrad 2026-08-22/a9069c177a9d UOp.split_uop and uop_given_valid use
   * unbounded Python iterators/lists; the 129th bound must participate. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
  PolyUOp *valid = NULL;
  PolyUOp *last = NULL;
  for (int i = 0; i < 129; i++) {
    char name[16];
    snprintf(name, sizeof(name), "x%d", i);
    PolyUOp *x = Variable(ctx, name, 0, 3, POLY_WEAKINT);
    PolyUOp *bound = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, x, one, poly_arg_none());
    valid = valid ? poly_uop2(ctx, POLY_OP_AND, POLY_BOOL, valid, bound, poly_arg_none()) : bound;
    last = x;
  }

  PolyUOp *out = poly_uop_given_valid(ctx, valid, last, false);
  ASSERT_INT_EQ(out->op, POLY_OP_CONST);
  ASSERT_INT_EQ(out->arg.kind, POLY_ARG_INT);
  ASSERT_INT_EQ(out->arg.i, 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, gated_valid_does_not_assume_through_index) {
  /* Tinygrad 2026-08-22/a9069c177a9d uop/symbolic.py:418-423 keeps an
   * INDEX-bearing value unchanged: its validity must survive the gate. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *r = Variable(ctx, "r", 0, 7, POLY_WEAKINT);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *four = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(4));
  PolyUOp *cond = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, r, four, poly_arg_none());
  PolyUOp *index_src[2] = {zero, zero};
  PolyUOp *opaque_index = poly_uop(ctx, POLY_OP_INDEX, POLY_WEAKINT, index_src, 2, poly_arg_none());
  PolyUOp *mod = poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_WEAKINT, r, four, poly_arg_none());
  PolyUOp *value = poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, opaque_index, mod, poly_arg_none());
  PolyUOp *invalid = poly_uop_const(ctx, poly_arg_invalid(), POLY_WEAKINT);
  PolyUOp *root =
      poly_uop3(ctx, POLY_OP_WHERE, POLY_WEAKINT, cond, value, invalid, poly_arg_none());

  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, root, poly_pm_simplify_valid()), root);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(alu, fdiv_zero_zero_is_nan) {
  PolyArg ops[2] = {poly_arg_float(0.0), poly_arg_float(0.0)};
  PolyArg r = poly_exec_alu(POLY_OP_FDIV, POLY_FLOAT32, ops, 2, true);
  ASSERT_TRUE(isnan(r.f));
  PASS();
}

TEST(alu, pow_domain_values_match_safe_pow) {
  /* Pinned uop/ops.py:safe_pow maps complex results to NaN and either
   * signed zero raised to a negative exponent to positive infinity. */
  const double cases[][3] = {
      {-28.0, 0.2, NAN},
      {-28.0, 1.2, NAN},
      {-28.0, -0.2, NAN},
      {-2.0, 3.0, -8.0},
      {-2.0, -3.0, -0.125},
      {0.0, -3.0, INFINITY},
      {-0.0, -3.0, INFINITY},
      {-INFINITY, 0.2, INFINITY},
      {-INFINITY, 3.0, -INFINITY},
      {NAN, 2.0, NAN},
      {2.0, NAN, NAN},
      {NAN, 0.0, 1.0},
      {1.0, NAN, 1.0},
      {0.0, 1.1, 0.0},
  };
  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
    PolyArg args[] = {poly_arg_float(cases[i][0]), poly_arg_float(cases[i][1])};
    for (int truncate = 0; truncate <= 1; truncate++) {
      PolyArg out = poly_exec_alu(POLY_OP_POW, POLY_FLOAT64, args, 2, truncate);
      ASSERT_EQ(out.kind, POLY_ARG_FLOAT);
      ASSERT_TRUE(isnan(cases[i][2]) ? isnan(out.f) : out.f == cases[i][2]);
    }
  }
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *base = poly_const_typed(ctx, POLY_FLOAT32, -28.0);
  PolyUOp *exponent = poly_const_typed(ctx, POLY_FLOAT32, 0.2);
  PolyUOp *out = simplify(ctx, poly_alu2(ctx, POLY_OP_POW, base, exponent));
  ASSERT_NOT_NULL(out);
  ASSERT_EQ(out->op, POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(out->dtype, POLY_FLOAT32));
  ASSERT_TRUE(isnan(out->arg.f));
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(alu, fold_float16_truncates_output) {
  PolyArg ops[2] = {poly_arg_float(1.0), poly_arg_float(0.0001)};
  PolyArg r = poly_exec_alu(POLY_OP_ADD, POLY_FLOAT16, ops, 2, true);
  ASSERT_FLOAT_EQ(r.f, 1.0, 0.0);
  PASS();
}

TEST(alu, fold_bfloat16_truncates_output) {
  PolyArg ops[2] = {poly_arg_float(1.0), poly_arg_float(0.001)};
  PolyArg r = poly_exec_alu(POLY_OP_ADD, POLY_BFLOAT16, ops, 2, true);
  ASSERT_FLOAT_EQ(r.f, 1.0, 0.0);
  PASS();
}

TEST(sym, associative_float16_const_fold_rounds_through_dtype_const) {
  /* Current symbolic.py:31-33 computes ALU values without output truncation,
   * then const_like -> UOp.const -> DType.const rounds the stored half CONST. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *x = poly_uop0(ctx, POLY_OP_PARAM, POLY_FLOAT16, poly_arg_int(0));
  PolyUOp *c1 = poly_uop_const(ctx, poly_arg_float(1.702), POLY_FLOAT16);
  PolyUOp *c2 = poly_uop_const(ctx, poly_arg_float(-1.0 / 0.693147180559945309417), POLY_FLOAT16);
  PolyUOp *expression = poly_uop2(
      ctx, POLY_OP_MUL, POLY_FLOAT16,
      poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT16, x, c1, poly_arg_none()), c2, poly_arg_none()
  );
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(c1);
  ASSERT_NOT_NULL(c2);
  ASSERT_NOT_NULL(expression);

  PolyUOp *rewritten = poly_graph_rewrite(ctx, expression, poly_symbolic());
  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(rewritten->op, POLY_OP_MUL);
  ASSERT_PTR_EQ(rewritten->src[0], x);
  ASSERT_INT_EQ(rewritten->src[1]->op, POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(rewritten->src[1]->dtype, POLY_FLOAT16));
  PolyArg operands[2] = {c1->arg, c2->arg};
  PolyArg untruncated = poly_exec_alu(POLY_OP_MUL, POLY_FLOAT16, operands, 2, false);
  PolyUOp *expected = poly_uop_const(ctx, untruncated, POLY_FLOAT16);
  ASSERT_PTR_EQ(rewritten->src[1], expected);
  ASSERT_DOUBLE_ULP(rewritten->src[1]->arg.f, -2.455078125, 0);
  poly_ctx_destroy(ctx);
  PASS();
}

/* Symbolic simplification tests */

TEST(sym, add_zero) {
  /* x + 0 -> x */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(5));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, x, zero, poly_arg_none());

  PolyUOp *r = simplify(ctx, add);
  ASSERT_PTR_EQ(r, x);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, add_zero_reversed) {
  /* 0 + x -> x (commutative) */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(5));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, zero, x, poly_arg_none());

  PolyUOp *r = simplify(ctx, add);
  ASSERT_PTR_EQ(r, x);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, mul_one) {
  /* x * 1 -> x */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(42));
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(1));
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, x, one, poly_arg_none());

  PolyUOp *r = simplify(ctx, mul);
  ASSERT_PTR_EQ(r, x);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, mul_zero) {
  /* x * 0 -> 0 */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(42));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, x, zero, poly_arg_none());

  PolyUOp *r = simplify(ctx, mul);
  ASSERT_TRUE(r->op == POLY_OP_CONST);
  ASSERT_INT_EQ(r->arg.i, 0);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, const_fold_add) {
  /* 2 + 3 -> 5 */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(2));
  PolyUOp *b = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(3));
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, a, b, poly_arg_none());

  PolyUOp *r = simplify(ctx, add);
  ASSERT_TRUE(r->op == POLY_OP_CONST);
  ASSERT_INT_EQ(r->arg.i, 5);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, const_fold_neg) {
  /* NEG(3.0) -> -3.0 */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(3.0));
  PolyUOp *neg = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, c, poly_arg_none());

  PolyUOp *r = simplify(ctx, neg);
  ASSERT_TRUE(r->op == POLY_OP_CONST);
  ASSERT_FLOAT_EQ(r->arg.f, -3.0, 1e-6);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, div_self) {
  /* x // x -> 1 */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(7));
  PolyUOp *div = poly_uop2(ctx, POLY_OP_IDIV, POLY_INT32, x, x, poly_arg_none());

  PolyUOp *r = simplify(ctx, div);
  ASSERT_TRUE(r->op == POLY_OP_CONST);
  ASSERT_INT_EQ(r->arg.i, 1);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, mod_self) {
  /* x % x -> 0 */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(5));
  PolyUOp *mod = poly_uop2(ctx, POLY_OP_MOD, POLY_INT32, x, x, poly_arg_none());

  PolyUOp *r = simplify(ctx, mod);
  ASSERT_TRUE(r->op == POLY_OP_CONST);
  ASSERT_INT_EQ(r->arg.i, 0);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, combined_rewrite) {
  /* (a + 0) * 1 -> a  (requires two rules, cascading) */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(42));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(1));
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, a, zero, poly_arg_none());
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, add, one, poly_arg_none());

  PolyUOp *r = simplify(ctx, mul);
  /* Should simplify to just 'a' */
  ASSERT_PTR_EQ(r, a);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, shifted_cmplt_bound_matches_pinned_symbolic) {
  /* Pinned tinygrad/uop/symbolic.py:261 rewrites
   *   (x + -2) < 3  ->  x < 5
   * through the commutative ADD pattern. Assert exact source identity and
   * constant topology, not only equivalent boolean values. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = Variable(ctx, "shifted_bound", 0, 40, POLY_WEAKINT);
  PolyUOp *minus_two = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(-2));
  PolyUOp *three = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(3));
  PolyUOp *shifted = poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, x, minus_two, poly_arg_none());
  PolyUOp *cmp = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, shifted, three, poly_arg_none());

  PolyUOp *rewritten = poly_graph_rewrite(ctx, cmp, poly_symbolic());
  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(rewritten->op, POLY_OP_CMPLT);
  ASSERT_INT_EQ(rewritten->n_src, 2);
  ASSERT_PTR_EQ(rewritten->src[0], x);
  ASSERT_INT_EQ(rewritten->src[1]->op, POLY_OP_CONST);
  ASSERT_TRUE(integer_const_eq(rewritten->src[1], "5"));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, weak_index_lt_folding_matches_pinned_topology) {
  /* Pinned tinygrad/uop/symbolic.py:174-178,290 rewrites
   *   RANGE_reduce*2 + RANGE_loop < 2  ->  RANGE_reduce < 1
   * because the unit-factor remainder lies exactly in [0,2). This is the
   * validity coordinate reached by ResNet18 conv1.weight backward CALL 99. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(2));
  PolyUOp *reduce =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, two, poly_arg_range(1, POLY_AXIS_REDUCE));
  PolyUOp *loop =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, two, poly_arg_range(8, POLY_AXIS_LOOP));
  PolyUOp *scaled = poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, reduce, two, poly_arg_none());
  PolyUOp *lhs = poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, scaled, loop, poly_arg_none());
  PolyUOp *cmp = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, lhs, two, poly_arg_none());

  PolyUOp *rewritten = poly_graph_rewrite(ctx, cmp, poly_symbolic());
  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(rewritten->op, POLY_OP_CMPLT);
  ASSERT_INT_EQ(rewritten->n_src, 2);
  ASSERT_PTR_EQ(rewritten->src[0], reduce);
  ASSERT_TRUE(integer_const_eq(rewritten->src[1], "1"));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, weak_index_lt_folding_requires_pinned_range_and_gcd_proofs) {
  /* Pinned lt_folding returns None when the unit remainder reaches d or the
   * common divisor is one. Assert the original root survives in both cases. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(2));
  PolyUOp *three = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(3));
  PolyUOp *reduce =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, two, poly_arg_range(1, POLY_AXIS_REDUCE));
  PolyUOp *wide_remainder =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, three, poly_arg_range(8, POLY_AXIS_LOOP));
  PolyUOp *mul_two = poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, reduce, two, poly_arg_none());
  PolyUOp *range_lhs =
      poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, mul_two, wide_remainder, poly_arg_none());
  PolyUOp *range_cmp = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, range_lhs, two, poly_arg_none());
  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, range_cmp, poly_symbolic()), range_cmp);

  PolyUOp *loop =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, two, poly_arg_range(9, POLY_AXIS_LOOP));
  PolyUOp *mul_three = poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, reduce, three, poly_arg_none());
  PolyUOp *gcd_lhs = poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, mul_three, loop, poly_arg_none());
  PolyUOp *gcd_cmp = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, gcd_lhs, two, poly_arg_none());
  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, gcd_cmp, poly_symbolic()), gcd_cmp);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, weak_index_lt_folding_preserves_arbitrary_precision_factors) {
  /* Python ints are unbounded. Prove the same topology above int64 using
   * d=2**64, rather than truncating the factor or comparison bound. */
  PolyCtx *ctx = poly_ctx_new();
  const uint32_t two_to_64_limbs[] = {0, 0, 1};
  PolyUOp *two_to_64 =
      poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_bigint(1, two_to_64_limbs, 3));
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(2));
  PolyUOp *reduce =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, two, poly_arg_range(1, POLY_AXIS_REDUCE));
  PolyUOp *loop =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, two, poly_arg_range(8, POLY_AXIS_LOOP));
  PolyUOp *scaled = poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, reduce, two_to_64, poly_arg_none());
  PolyUOp *lhs = poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, scaled, loop, poly_arg_none());
  PolyUOp *cmp = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, lhs, two_to_64, poly_arg_none());

  PolyUOp *rewritten = poly_graph_rewrite(ctx, cmp, poly_symbolic());
  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(rewritten->op, POLY_OP_CMPLT);
  ASSERT_INT_EQ(rewritten->n_src, 2);
  ASSERT_PTR_EQ(rewritten->src[0], reduce);
  ASSERT_TRUE(integer_const_eq(rewritten->src[1], "1"));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, validity_priority_sort_is_not_output_topology) {
  /* Pinned tinygrad/uop/symbolic.py:374-383 sorts only the working clause
   * order. If no clause is deduplicated or simplified, simplify_valid returns
   * None and preserves this grouped condition exactly. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *end18 = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(18));
  PolyUOp *end7 = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(7));
  PolyUOp *r =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, end18, poly_arg_range(3, POLY_AXIS_PLACEHOLDER));
  PolyUOp *q =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, end7, poly_arg_range(4, POLY_AXIS_PLACEHOLDER));
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(2));
  PolyUOp *x = poly_uop2(
      ctx, POLY_OP_ADD, POLY_WEAKINT,
      poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, r, two, poly_arg_none()), q, poly_arg_none()
  );
  PolyUOp *truth = poly_uop0(ctx, POLY_OP_CONST, POLY_BOOL, poly_arg_bool(true));
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
  PolyUOp *five = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(5));
  PolyUOp *seventeen = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(17));
  PolyUOp *thirty_seven = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(37));
  PolyUOp *x_lo = poly_uop2(
      ctx, POLY_OP_CMPNE, POLY_BOOL,
      poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, x, five, poly_arg_none()), truth, poly_arg_none()
  );
  PolyUOp *x_hi = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, x, thirty_seven, poly_arg_none());
  PolyUOp *r_lo = poly_uop2(
      ctx, POLY_OP_CMPNE, POLY_BOOL,
      poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, r, one, poly_arg_none()), truth, poly_arg_none()
  );
  PolyUOp *r_hi = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, r, seventeen, poly_arg_none());
  PolyUOp *x_valid = poly_uop2(ctx, POLY_OP_AND, POLY_BOOL, x_lo, x_hi, poly_arg_none());
  PolyUOp *r_valid = poly_uop2(ctx, POLY_OP_AND, POLY_BOOL, r_lo, r_hi, poly_arg_none());
  PolyUOp *valid = poly_uop2(ctx, POLY_OP_AND, POLY_BOOL, x_valid, r_valid, poly_arg_none());

  ASSERT_TRUE(poly_pm_rewrite(poly_pm_simplify_valid(), ctx, valid) == NULL);
  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, valid, poly_pm_simplify_valid()), valid);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, const_fold_mul_float) {
  /* 2.0 * 3.5 -> 7.0 */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(2.0));
  PolyUOp *b = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(3.5));
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, a, b, poly_arg_none());

  PolyUOp *r = simplify(ctx, mul);
  ASSERT_TRUE(r->op == POLY_OP_CONST);
  ASSERT_FLOAT_EQ(r->arg.f, 7.0, 1e-6);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, where_same_branches) {
  /* WHERE(cond, v, v) -> v */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *cond = poly_uop0(ctx, POLY_OP_CONST, POLY_BOOL, poly_arg_bool(true));
  PolyUOp *v = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(99));
  PolyUOp *wh = poly_uop3(ctx, POLY_OP_WHERE, POLY_INT32, cond, v, v, poly_arg_none());

  PolyUOp *r = simplify(ctx, wh);
  ASSERT_PTR_EQ(r, v);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, where_logical_not_swaps_branches) {
  /* Tinygrad symbolic.py:230-231:
   *   cond.logical_not().where(t, f) -> cond.where(f, t) */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *cond = Variable(ctx, "c", 0, 1, POLY_BOOL);
  PolyUOp *not_cond = poly_logical_not(ctx, cond);
  PolyUOp *t = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(7));
  PolyUOp *f = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(3));
  PolyUOp *wh = poly_uop3(ctx, POLY_OP_WHERE, POLY_INT32, not_cond, t, f, poly_arg_none());

  PolyUOp *r = poly_graph_rewrite(ctx, wh, poly_symbolic());
  ASSERT_TRUE(r->op == POLY_OP_WHERE);
  ASSERT_TRUE(poly_uop_is_alu_param(r->src[0]));
  ASSERT_TRUE(r->src[0]->dtype.priority == POLY_BOOL.priority);
  ASSERT_TRUE(r->src[0]->dtype.bitsize == POLY_BOOL.bitsize);
  ASSERT_TRUE(r->src[0]->arg.kind == POLY_ARG_PARAM);
  ASSERT_STR_EQ(r->src[0]->arg.param->name, "c");
  ASSERT_EQ(r->src[0]->arg.param->min_val, 0);
  ASSERT_EQ(r->src[0]->arg.param->max_val, 1);
  ASSERT_PTR_EQ(r->src[1], f);
  ASSERT_PTR_EQ(r->src[2], t);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, where_logical_not_is_not_symbolic_simple) {
  /* Current tinygrad keeps this rule in symbolic, after symbolic_simple. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *cond = Variable(ctx, "c", 0, 1, POLY_BOOL);
  PolyUOp *not_cond = poly_logical_not(ctx, cond);
  PolyUOp *t = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(7));
  PolyUOp *f = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(3));
  PolyUOp *wh = poly_uop3(ctx, POLY_OP_WHERE, POLY_INT32, not_cond, t, f, poly_arg_none());

  PolyUOp *r = poly_graph_rewrite(ctx, wh, poly_symbolic_simple());
  ASSERT_PTR_EQ(r, wh);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, after_flattens_non_effect_dependencies_like_tinygrad) {
  /* Pinned symbolic.py:306-311 keeps effect dependencies, replaces every
   * other dependency by its direct sources, deduplicates that flattened list
   * in order, and eliminates a one-source AFTER. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *value = poly_uop0(ctx, POLY_OP_NOOP, POLY_INT32, poly_arg_none());
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(4));
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_range(7, POLY_AXIS_LOOP));
  PolyUOp *wrapped = poly_uop1(ctx, POLY_OP_CAST, POLY_WEAKINT, range, poly_arg_none());
  PolyUOp *target = poly_uop0(ctx, POLY_OP_NOOP, POLY_INT32, poly_arg_none());
  PolyUOp *stored = poly_uop2(
      ctx, POLY_OP_STORE, POLY_VOID, target,
      poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(1)), poly_arg_none()
  );
  PolyUOp *barrier = poly_uop1(ctx, POLY_OP_BARRIER, POLY_VOID, stored, poly_arg_none());
  PolyUOp *end_srcs[2] = {stored, range};
  PolyUOp *ended = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_srcs, 2, poly_arg_none());
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, range, bound, poly_arg_none());
  PolyUOp *srcs[7] = {value, wrapped, stored, add, barrier, ended, wrapped};
  PolyUOp *root = poly_uop_tagged_arg(
      ctx, POLY_OP_AFTER, POLY_INT32, srcs, 7, poly_arg_none(), 19, poly_arg_str("after-canonical")
  );

  PolyUOp *direct = poly_pm_rewrite(poly_symbolic(), ctx, root);
  ASSERT_NOT_NULL(direct);
  ASSERT_INT_EQ(direct->op, POLY_OP_AFTER);
  ASSERT_INT_EQ(direct->n_src, 6);
  PolyUOp *direct_expected[6] = {value, range, stored, bound, barrier, ended};
  for (int i = 0; i < 6; i++)
    ASSERT_PTR_EQ(direct->src[i], direct_expected[i]);
  ASSERT_INT_EQ(direct->tag, 19);
  ASSERT_TRUE(direct->tag_arg.kind == POLY_ARG_STRING);
  ASSERT_STR_EQ(direct->tag_arg.str, "after-canonical");

  /* Recursive graph rewrite first folds ADD(range, bound), so pinned and
   * Polygrad both omit bound from the parent's flattened dependency list. */
  PolyUOp *rewritten = poly_graph_rewrite(ctx, root, poly_sym());
  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(rewritten->op, POLY_OP_AFTER);
  ASSERT_INT_EQ(rewritten->n_src, 5);
  PolyUOp *recursive_expected[5] = {value, range, stored, barrier, ended};
  for (int i = 0; i < 5; i++)
    ASSERT_PTR_EQ(rewritten->src[i], recursive_expected[i]);
  ASSERT_INT_EQ(rewritten->tag, 19);
  ASSERT_TRUE(rewritten->tag_arg.kind == POLY_ARG_STRING);
  ASSERT_STR_EQ(rewritten->tag_arg.str, "after-canonical");

  PolyUOp *singleton =
      poly_uop_tagged(ctx, POLY_OP_AFTER, POLY_INT32, &value, 1, poly_arg_none(), 20);
  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, singleton, poly_symbolic()), value);

  poly_ctx_destroy(ctx);
  PASS();
}

/* poly_uop_minmax tinygrad parity tests *
 * Every assertion below corresponds to one row of
 * test/parity_scripts/tg_minmax_gt.py output, captured against
 * the pinned tinygrad UOp._min_max. To re-verify after a tinygrad bump:
 *
 *   PYTHONPATH=references/tinygrad_latest \
 *     references/.venv-tinygrad-py311/bin/python test/parity_scripts/tg_minmax_gt.py
 *
 * Then update the literal values below if tinygrad's semantics changed. */

static PolyUOp *mk_const(PolyCtx *ctx, int64_t v) {
  return poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(v));
}
static PolyUOp *mk_dvar(PolyCtx *ctx, const char *name, int64_t lo, int64_t hi) {
  return Variable(ctx, name, lo, hi, POLY_INT32);
}
static PolyUOp *mk_range(PolyCtx *ctx, int64_t n, int64_t axis_id) {
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(n));
  return poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_range(axis_id, POLY_AXIS_LOOP));
}

typedef struct {
  int64_t ranges[8];
  int n_ranges;
  const char *var_names[8];
  int64_t var_values[8];
  int n_vars;
} SymEvalEnv;

static bool sym_eval_i64(PolyUOp *u, const SymEvalEnv *env, int64_t *out) {
  if (!u || !out) return false;
  if (poly_uop_is_variable(u) || poly_uop_is_alu_param(u)) {
    const char *name = poly_uop_expr(u);
    for (int i = 0; name && i < env->n_vars; i++) {
      if (strcmp(env->var_names[i], name) == 0) {
        *out = env->var_values[i];
        return true;
      }
    }
    return false;
  }
  switch (u->op) {
  case POLY_OP_CONST:
    if (u->arg.kind == POLY_ARG_BOOL) {
      *out = u->arg.b ? 1 : 0;
      return true;
    }
    if (u->arg.kind != POLY_ARG_INT) return false;
    *out = u->arg.i;
    return true;
  case POLY_OP_RANGE: {
    int64_t axis = poly_range_axis_id(u->arg);
    if (axis < 0 || axis >= env->n_ranges) return false;
    *out = env->ranges[axis];
    return true;
  }
  default:
    break;
  }

  int64_t a = 0, b = 0, c = 0;
  if (u->n_src > 0 && !sym_eval_i64(u->src[0], env, &a)) return false;
  if (u->n_src > 1 && !sym_eval_i64(u->src[1], env, &b)) return false;
  if (u->n_src > 2 && !sym_eval_i64(u->src[2], env, &c)) return false;

  switch (u->op) {
  case POLY_OP_NEG:
    *out = -a;
    return true;
  case POLY_OP_ADD:
    *out = a + b;
    return true;
  case POLY_OP_SUB:
    *out = a - b;
    return true;
  case POLY_OP_MUL:
    *out = a * b;
    return true;
  case POLY_OP_IDIV:
    if (b == 0) return false;
    *out = a / b;
    return true;
  case POLY_OP_MOD:
    if (b == 0) return false;
    *out = a % b;
    return true;
  case POLY_OP_FLOORDIV:
  case POLY_OP_FLOORMOD: {
    if (b == 0) return false;
    PolyArg args[2] = {poly_arg_int(a), poly_arg_int(b)};
    PolyArg result = poly_exec_alu(u->op, u->dtype, args, 2, true);
    if (result.kind != POLY_ARG_INT) return false;
    *out = result.i;
    return true;
  }
  case POLY_OP_CMPLT:
    *out = a < b;
    return true;
  case POLY_OP_CMPNE:
    *out = a != b;
    return true;
  case POLY_OP_CMPEQ:
    *out = a == b;
    return true;
  case POLY_OP_WHERE:
    *out = a ? b : c;
    return true;
  case POLY_OP_MAX:
    *out = a > b ? a : b;
    return true;
  case POLY_OP_AND:
    *out = a & b;
    return true;
  case POLY_OP_OR:
    *out = a | b;
    return true;
  case POLY_OP_XOR:
    *out = a ^ b;
    return true;
  default:
    return false;
  }
}

static int sym_count_ops_in_root(PolyCtx *ctx, PolyUOp *root, PolyOps op) {
  int n = 0;
  PolyUOp **topo = poly_toposort(ctx, root, &n);
  int count = 0;
  for (int i = 0; i < n; i++)
    if (topo[i]->op == op) count++;
  return count;
}

TEST(sym, integer_pow_uses_current_xpow_float_graph) {
  /* Tinygrad 2026-08-22/a9069c177a9d
   * codegen/decomp/transcendental.py:257-267 rewrites every remaining POW,
   * including integer operands, through the float xpow graph. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *base = Variable(ctx, "pow_base", -8, 8, POLY_INT32);
  PolyUOp *exponent = Variable(ctx, "pow_exponent", -8, 8, POLY_INT32);
  PolyUOp *root = poly_uop2(ctx, POLY_OP_POW, POLY_INT32, base, exponent, poly_arg_none());
  PolyUOp *rewritten = poly_graph_rewrite(ctx, root, poly_sym());
  ASSERT_NOT_NULL(rewritten);
  ASSERT_EQ(rewritten->op, POLY_OP_WHERE);
  ASSERT_TRUE(poly_dtype_eq(rewritten->dtype, POLY_FLOAT32));
  ASSERT_INT_EQ(sym_count_ops_in_root(ctx, rewritten, POLY_OP_POW), 0);
  ASSERT_INT_EQ(sym_count_ops_in_root(ctx, rewritten, POLY_OP_EXP2), 1);
  ASSERT_INT_EQ(sym_count_ops_in_root(ctx, rewritten, POLY_OP_LOG2), 1);
  ASSERT_INT_EQ(sym_count_ops_in_root(ctx, rewritten, POLY_OP_FLOORMOD), 1);
  ASSERT_INT_EQ(sym_count_ops_in_root(ctx, rewritten, POLY_OP_CAST), 1);
  ASSERT_INT_EQ(sym_count_ops_in_root(ctx, rewritten, POLY_OP_WHERE), 4);
  ASSERT_INT_EQ(sym_count_ops_in_root(ctx, rewritten, POLY_OP_CMPLT), 2);
  ASSERT_INT_EQ(sym_count_ops_in_root(ctx, rewritten, POLY_OP_CMPNE), 2);

  poly_ctx_destroy(ctx);
  PASS();
}

static uint32_t sym_fuzz_next(uint32_t *state) {
  *state = *state * 1664525u + 1013904223u;
  return *state;
}

static int64_t sym_fuzz_i64(uint32_t *state, int64_t lo, int64_t hi) {
  return lo + (int64_t)(sym_fuzz_next(state) % (uint32_t)(hi - lo + 1));
}

static PolyUOp *sym_mul_const(PolyCtx *ctx, PolyUOp *x, int64_t c) {
  if (c == 1) return x;
  PolyUOp *cc = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(c));
  return poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, x, cc, poly_arg_none());
}

static PolyUOp *sym_add(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  return poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, a, b, poly_arg_none());
}

static PolyUOp *sym_fuzz_linear(PolyCtx *ctx, PolyUOp **vars, int n_vars, uint32_t *state) {
  PolyUOp *ret =
      poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(sym_fuzz_i64(state, -15, 15)));
  int n_terms = 1 + (int)(sym_fuzz_next(state) % 4);
  for (int i = 0; i < n_terms; i++) {
    PolyUOp *v = vars[sym_fuzz_next(state) % (uint32_t)n_vars];
    int64_t f = sym_fuzz_i64(state, -12, 12);
    if (f == 0) f = 1;
    ret = sym_add(ctx, ret, sym_mul_const(ctx, v, f));
  }
  return ret;
}

static PolyUOp *sym_fuzz_expr(PolyCtx *ctx, PolyUOp **vars, int n_vars, uint32_t *state) {
  PolyUOp *x = sym_fuzz_linear(ctx, vars, n_vars, state);
  int64_t d = sym_fuzz_i64(state, 2, 19);
  PolyUOp *den = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(d));
  switch (sym_fuzz_next(state) % 7) {
  case 0:
    return poly_uop2(ctx, POLY_OP_MOD, POLY_INT32, x, den, poly_arg_none());
  case 1:
    return poly_uop2(ctx, POLY_OP_IDIV, POLY_INT32, x, den, poly_arg_none());
  case 2: {
    int64_t k = sym_fuzz_i64(state, 2, 6);
    PolyUOp *wide_den = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(d * k));
    PolyUOp *inner = poly_uop2(ctx, POLY_OP_MOD, POLY_INT32, x, wide_den, poly_arg_none());
    return poly_uop2(ctx, POLY_OP_IDIV, POLY_INT32, inner, den, poly_arg_none());
  }
  case 3: {
    int64_t k = sym_fuzz_i64(state, 2, 6);
    PolyUOp *wide_den = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(d * k));
    PolyUOp *inner = poly_uop2(ctx, POLY_OP_MOD, POLY_INT32, x, wide_den, poly_arg_none());
    return poly_uop2(ctx, POLY_OP_MOD, POLY_INT32, inner, den, poly_arg_none());
  }
  case 4: {
    PolyUOp *y = sym_fuzz_linear(ctx, vars, n_vars, state);
    PolyUOp *cond =
        poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, vars[0], mk_const(ctx, 4), poly_arg_none());
    return poly_uop3(ctx, POLY_OP_WHERE, POLY_INT32, cond, x, y, poly_arg_none());
  }
  case 5:
    return poly_uop2(
        ctx, POLY_OP_MOD, POLY_INT32, sym_add(ctx, x, mk_const(ctx, 31)), den, poly_arg_none()
    );
  default:
    return x;
  }
}

TEST(sym, symbolic_fuzzer_integer_rewrite_equivalence) {
  uint32_t seed = 0xC0FFEEu;
  for (int trial = 0; trial < 512; trial++) {
    PolyCtx *ctx = poly_ctx_new();
    PolyUOp *vars[4] = {
        mk_range(ctx, 9, 0),
        mk_range(ctx, 7, 1),
        mk_dvar(ctx, "a", 4, 11),
        mk_dvar(ctx, "b", -3, 5),
    };
    PolyUOp *root = sym_fuzz_expr(ctx, vars, 4, &seed);
    PolyUOp *rewritten = poly_graph_rewrite(ctx, root, poly_symbolic());
    ASSERT_TRUE(rewritten != NULL);

    for (int r0 = 0; r0 < 9; r0 += 2) {
      for (int r1 = 0; r1 < 7; r1 += 3) {
        for (int a = 4; a <= 11; a += 3) {
          for (int b = -3; b <= 5; b += 4) {
            SymEvalEnv env = {0};
            env.ranges[0] = r0;
            env.ranges[1] = r1;
            env.n_ranges = 2;
            env.var_names[0] = "a";
            env.var_values[0] = a;
            env.var_names[1] = "b";
            env.var_values[1] = b;
            env.n_vars = 2;
            int64_t got = 0, want = 0;
            ASSERT_TRUE(sym_eval_i64(root, &env, &want));
            ASSERT_TRUE(sym_eval_i64(rewritten, &env, &got));
            if (got != want) {
              char *root_s = poly_uop_str(root);
              char *rewritten_s = poly_uop_str(rewritten);
              fprintf(stderr, "    root=%s\n    rewritten=%s\n", root_s, rewritten_s);
              fprintf(stderr, "    root tree:\n");
              poly_uop_dump_tree(stderr, root, 0, 10);
              fprintf(stderr, "    rewritten tree:\n");
              poly_uop_dump_tree(stderr, rewritten, 0, 10);
              free(root_s);
              free(rewritten_s);
              FAIL(
                  "trial %d sample r0=%d r1=%d a=%d b=%d got %lld want %lld", trial, r0, r1, a, b,
                  (long long)got, (long long)want
              );
            }
          }
        }
      }
    }
    poly_ctx_destroy(ctx);
  }
  PASS();
}

TEST(sym, add_divmod_recombine_preserves_terms_past_old_cap) {
  /* tinygrad symbolic.fold_add_divmod_recombine uses list(x.split_uop(ADD))
   * with no fixed scratch cap. Polygrad used to collect only the first 32
   * terms, then rebuild after a match, which could drop tail terms. Shape the
   * tree so the div/mod pair appears together only at the root. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *base_bound = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(100));
  PolyUOp *tail_bound = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(11));
  PolyUOp *base =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, base_bound, poly_arg_range(0, POLY_AXIS_LOOP));
  PolyUOp *tail =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, tail_bound, poly_arg_range(1, POLY_AXIS_LOOP));
  PolyUOp *c4 = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(4));

  PolyUOp *mod = poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_WEAKINT, base, c4, poly_arg_none());
  PolyUOp *div = poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_WEAKINT, base, c4, poly_arg_none());
  PolyUOp *divmul = poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, div, c4, poly_arg_none());

  PolyUOp *left = mod;
  for (int i = 0; i < 30; i++)
    left = poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, left, tail, poly_arg_none());

  PolyUOp *right = divmul;
  for (int i = 0; i < 20; i++)
    right = poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, right, tail, poly_arg_none());

  PolyUOp *root = poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, left, right, poly_arg_none());
  PolyUOp *rewritten = poly_graph_rewrite(ctx, root, poly_symbolic());
  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(sym_count_ops_in_root(ctx, rewritten, POLY_OP_FLOORMOD), 0);
  ASSERT_INT_EQ(sym_count_ops_in_root(ctx, rewritten, POLY_OP_FLOORDIV), 0);

  for (int r0 = 0; r0 < 100; r0 += 17) {
    for (int r1 = 0; r1 < 11; r1 += 5) {
      SymEvalEnv env = {0};
      env.ranges[0] = r0;
      env.ranges[1] = r1;
      env.n_ranges = 2;
      int64_t got = 0, want = 0;
      ASSERT_TRUE(sym_eval_i64(root, &env, &want));
      ASSERT_TRUE(sym_eval_i64(rewritten, &env, &got));
      if (got != want) {
        char *root_s = poly_uop_str(root);
        char *rewritten_s = poly_uop_str(rewritten);
        fprintf(stderr, "    root=%s\n    rewritten=%s\n", root_s, rewritten_s);
        free(root_s);
        free(rewritten_s);
        FAIL("sample r0=%d r1=%d got %lld want %lld", r0, r1, (long long)got, (long long)want);
      }
    }
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, add_divmod_recombine_canonicalizes_nested_quotient) {
  /* tinygrad@2026-08-22/a9069c177a9d uop/symbolic.py:_quotient_base
   * recognizes ((r//6)//6) as r//36 before recombining r%36. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *r = poly_range(ctx, 432, 0, POLY_AXIS_LOOP);
  PolyUOp *six = poly_const_int(ctx, 6);
  PolyUOp *thirty_six = poly_const_int(ctx, 36);
  PolyUOp *q0 = poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_WEAKINT, r, six, poly_arg_none());
  PolyUOp *q1 = poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_WEAKINT, q0, six, poly_arg_none());
  PolyUOp *lhs = poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, q1, thirty_six, poly_arg_none());
  PolyUOp *rhs = poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_WEAKINT, r, thirty_six, poly_arg_none());
  PolyUOp *root = poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, lhs, rhs, poly_arg_none());

  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, root, poly_symbolic()), r);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, nested_floordiv_constants_match_current_tinygrad) {
  /* tinygrad@2026-08-22/a9069c177a9d uop/symbolic.py:269-270. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *r = poly_range(ctx, 432, 0, POLY_AXIS_LOOP);
  PolyUOp *six = poly_const_int(ctx, 6);
  PolyUOp *q0 = poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_WEAKINT, r, six, poly_arg_none());
  PolyUOp *q1 = poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_WEAKINT, q0, six, poly_arg_none());

  PolyUOp *out = poly_graph_rewrite(ctx, q1, poly_symbolic());
  ASSERT_NOT_NULL(out);
  ASSERT_INT_EQ(out->op, POLY_OP_FLOORDIV);
  ASSERT_PTR_EQ(out->src[0], r);
  ASSERT_TRUE(integer_const_eq(out->src[1], "36"));
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, divmod_splits_constant_for_signed_nonzero_divisor) {
  /* tinygrad@2026-08-22/a9069c177a9d uop/divandmod.py:102-105. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *r = poly_range(ctx, 432, 0, POLY_AXIS_LOOP);
  PolyUOp *thirteen = poly_const_int(ctx, 13);
  PolyUOp *numerator = poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, r, thirteen, poly_arg_none());

  for (int divisor = 6; divisor >= -6; divisor -= 12) {
    PolyUOp *d = poly_const_int(ctx, divisor);
    PolyUOp *div = poly_graph_rewrite(
        ctx, poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_WEAKINT, numerator, d, poly_arg_none()),
        poly_symbolic()
    );
    PolyUOp *mod = poly_graph_rewrite(
        ctx, poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_WEAKINT, numerator, d, poly_arg_none()),
        poly_symbolic()
    );
    const char *remainder = divisor > 0 ? "1" : "-5";
    const char *quotient = divisor > 0 ? "2" : "-3";

    ASSERT_INT_EQ(div->op, POLY_OP_ADD);
    ASSERT_TRUE(integer_const_eq(div->src[1], quotient));
    ASSERT_INT_EQ(div->src[0]->op, POLY_OP_FLOORDIV);
    ASSERT_PTR_EQ(div->src[0]->src[1], d);
    ASSERT_INT_EQ(div->src[0]->src[0]->op, POLY_OP_ADD);
    ASSERT_PTR_EQ(div->src[0]->src[0]->src[0], r);
    ASSERT_TRUE(integer_const_eq(div->src[0]->src[0]->src[1], remainder));

    ASSERT_INT_EQ(mod->op, POLY_OP_FLOORMOD);
    ASSERT_PTR_EQ(mod->src[1], d);
    ASSERT_INT_EQ(mod->src[0]->op, POLY_OP_ADD);
    ASSERT_PTR_EQ(mod->src[0]->src[0], r);
    ASSERT_TRUE(integer_const_eq(mod->src[0]->src[1], remainder));
  }
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, add_divmod_recombine_multiplier_product_overflow_declines) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *base = Variable(ctx, "base", INT64_MIN, INT64_MAX, POLY_WEAKINT);
  PolyUOp *q = poly_uop0(ctx, POLY_OP_PARAM, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *max = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(INT64_MAX));
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(2));
  PolyUOp *three = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(3));
  PolyUOp *mod = poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_WEAKINT, base, max, poly_arg_none());
  PolyUOp *mod_scaled = poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, mod, two, poly_arg_none());
  PolyUOp *q_scaled = poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, q, three, poly_arg_none());
  PolyUOp *root = poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, mod_scaled, q_scaled, poly_arg_none());

  PolyUOp *rewritten = simplify(ctx, root);
  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(rewritten->op, POLY_OP_ADD);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, add_divmod_recombine_nested_divisor_product_overflow_declines) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *base = Variable(ctx, "base", INT64_MIN, INT64_MAX, POLY_WEAKINT);
  PolyUOp *max = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(INT64_MAX));
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(2));
  PolyUOp *three = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(3));
  PolyUOp *base_div_max =
      poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_WEAKINT, base, max, poly_arg_none());
  PolyUOp *nested_mod =
      poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_WEAKINT, base_div_max, two, poly_arg_none());
  PolyUOp *base_div_three =
      poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_WEAKINT, base, three, poly_arg_none());
  PolyUOp *scaled_div =
      poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, base_div_three, two, poly_arg_none());
  PolyUOp *root =
      poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, nested_mod, scaled_div, poly_arg_none());

  PolyUOp *rewritten = simplify(ctx, root);
  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(rewritten->op, POLY_OP_ADD);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, add_divmod_recombine_builds_bigint_modulus) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *base = Variable(ctx, "base", INT64_MIN, INT64_MAX, POLY_WEAKINT);
  PolyUOp *max = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(INT64_MAX));
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(2));
  PolyUOp *base_mod = poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_WEAKINT, base, max, poly_arg_none());
  PolyUOp *base_div = poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_WEAKINT, base, max, poly_arg_none());
  PolyUOp *nested_mod =
      poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_WEAKINT, base_div, two, poly_arg_none());
  PolyUOp *scaled_nested =
      poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, nested_mod, max, poly_arg_none());
  PolyUOp *root =
      poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, base_mod, scaled_nested, poly_arg_none());

  PolyUOp *rewritten = simplify(ctx, root);
  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(rewritten->op, POLY_OP_FLOORMOD);
  ASSERT_PTR_EQ(rewritten->src[0], base);
  ASSERT_TRUE(integer_const_eq(rewritten->src[1], "18446744073709551614"));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, add_divmod_recombine_negative_divisor_stays_add) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = Variable(ctx, "x", -12, 12, POLY_WEAKINT);
  PolyUOp *three = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(3));
  PolyUOp *negative_two = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(-2));
  PolyUOp *negative_six = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(-6));
  PolyUOp *base = poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_WEAKINT, x, three, poly_arg_none());
  PolyUOp *left =
      poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_WEAKINT, base, negative_two, poly_arg_none());
  PolyUOp *right_div =
      poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_WEAKINT, x, negative_six, poly_arg_none());
  PolyUOp *right =
      poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, right_div, negative_two, poly_arg_none());
  PolyUOp *root = poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, left, right, poly_arg_none());

  PolyUOp *rewritten = simplify(ctx, root);
  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(rewritten->op, POLY_OP_ADD);
  ASSERT_INT_EQ(sym_count_ops_in_root(ctx, rewritten, POLY_OP_FLOORMOD), 1);
  ASSERT_INT_EQ(sym_count_ops_in_root(ctx, rewritten, POLY_OP_FLOORDIV), 2);

  poly_ctx_destroy(ctx);
  PASS();
}
/* check_mm queries minmax and asserts both bounds match the tinygrad
 * ground truth. Implemented as a macro because ASSERT_INT_EQ touches the
 * test-local _passed/_failed counters that only exist inside a TEST. */
#define check_mm(ctx, u, want_lo, want_hi, label)                                                  \
  do {                                                                                             \
    int64_t _mm_lo, _mm_hi;                                                                        \
    poly_uop_minmax((ctx), (u), &_mm_lo, &_mm_hi);                                                 \
    if (_mm_lo != (want_lo) || _mm_hi != (want_hi)) {                                              \
      fprintf(                                                                                     \
          stderr, "%s: got [%lld..%lld], want [%lld..%lld]\n", (label), (long long)_mm_lo,         \
          (long long)_mm_hi, (long long)(want_lo), (long long)(want_hi)                            \
      );                                                                                           \
    }                                                                                              \
    ASSERT_INT_EQ(_mm_lo, (want_lo));                                                              \
    ASSERT_INT_EQ(_mm_hi, (want_hi));                                                              \
  } while (0)

TEST(sym, minmax_const_pos) {
  PolyCtx *ctx = poly_ctx_new();
  check_mm(ctx, mk_const(ctx, 5), 5, 5, "CONST 5");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_const_neg) {
  PolyCtx *ctx = poly_ctx_new();
  check_mm(ctx, mk_const(ctx, -2), -2, -2, "CONST -2");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_define_var_pos) {
  PolyCtx *ctx = poly_ctx_new();
  check_mm(ctx, mk_dvar(ctx, "x", 2, 7), 2, 7, "VARIABLE[2..7]");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_define_var_mixed_sign) {
  PolyCtx *ctx = poly_ctx_new();
  check_mm(ctx, mk_dvar(ctx, "y", -3, 4), -3, 4, "VARIABLE[-3..4]");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_bounded_param_matches_tinygrad) {
  /* Pinned tinygrad UOp._min_max returns ParamArg.vmin_vmax directly
   * (tinygrad/uop/ops.py:1010). */
  PolyCtx *ctx = poly_ctx_new();
  PolyParamArg arg = {
      .slot = 1,
      .name = "size",
      .min_val = 1,
      .max_val = 8,
      .has_minmax = true,
      .addrspace = POLY_ADDR_GLOBAL};
  PolyUOp *shape = poly_uop(ctx, POLY_OP_STACK, POLY_VOID, NULL, 0, poly_arg_none());
  PolyUOp *param = poly_uop1(ctx, POLY_OP_PARAM, POLY_WEAKINT, shape, poly_arg_param(&arg));
  check_mm(ctx, param, 1, 8, "PARAM[1..8]");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_range_10) {
  PolyCtx *ctx = poly_ctx_new();
  check_mm(ctx, mk_range(ctx, 10, 0), 0, 9, "RANGE(10)");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_range_1_degenerate) {
  PolyCtx *ctx = poly_ctx_new();
  check_mm(ctx, mk_range(ctx, 1, 1), 0, 0, "RANGE(1)");
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, minmax_cast_const_reads_stored_arg_like_tinygrad) {
  /* Pinned tinygrad ops.py:1010-1017 returns CONST.arg directly even when a
   * preceding CAST(CONST) produced a value outside the nominal dtype range. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *source = poly_uop0(ctx, POLY_OP_CONST, POLY_INT64, poly_arg_int(260));
  PolyUOp *cast = simplify(ctx, poly_uop1(ctx, POLY_OP_CAST, POLY_UINT8, source, poly_arg_none()));
  ASSERT_NOT_NULL(cast);
  ASSERT_INT_EQ(cast->op, POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(cast->dtype, POLY_UINT8));
  ASSERT_TRUE(cast->arg.kind == POLY_ARG_INT);
  ASSERT_INT_EQ(cast->arg.i, 260);
  int64_t lo = 0, hi = 0;
  poly_uop_minmax(ctx, cast, &lo, &hi);
  ASSERT_INT_EQ(lo, 260);
  ASSERT_INT_EQ(hi, 260);
  poly_ctx_destroy(ctx);
  PASS();
}

/* ADD / SUB */
TEST(sym, minmax_add_r_const) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(
      ctx, POLY_OP_ADD, POLY_INT32, mk_range(ctx, 10, 0), mk_const(ctx, 3), poly_arg_none()
  );
  check_mm(ctx, u, 3, 12, "r+3");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_add_r_dvar) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(
      ctx, POLY_OP_ADD, POLY_INT32, mk_range(ctx, 10, 0), mk_dvar(ctx, "x", 2, 7), poly_arg_none()
  );
  check_mm(ctx, u, 2, 16, "r+dv");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_sub_r_const) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(
      ctx, POLY_OP_SUB, POLY_INT32, mk_range(ctx, 10, 0), mk_const(ctx, 3), poly_arg_none()
  );
  check_mm(ctx, u, -3, 6, "r-3");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_sub_const_r) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(
      ctx, POLY_OP_SUB, POLY_INT32, mk_const(ctx, 3), mk_range(ctx, 10, 0), poly_arg_none()
  );
  check_mm(ctx, u, -6, 3, "3-r");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_sub_dv_dvn) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *dv = mk_dvar(ctx, "x", 2, 7);
  PolyUOp *dvn = mk_dvar(ctx, "y", -3, 4);
  PolyUOp *u = poly_uop2(ctx, POLY_OP_SUB, POLY_INT32, dv, dvn, poly_arg_none());
  check_mm(ctx, u, -2, 10, "dv-dvn");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_add_int64_overflow_falls_back_to_dtype) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = Variable(ctx, "x", INT64_MAX - 1, INT64_MAX, POLY_INT64);
  PolyUOp *y = Variable(ctx, "y", 1, INT32_MAX, POLY_INT64);
  PolyUOp *u = poly_uop2(ctx, POLY_OP_ADD, POLY_INT64, x, y, poly_arg_none());
  check_mm(ctx, u, INT64_MIN, INT64_MAX, "i64 add overflow fallback");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_sub_int64_overflow_falls_back_to_dtype) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = Variable(ctx, "x", INT64_MIN, INT64_MIN + 1, POLY_INT64);
  PolyUOp *y = Variable(ctx, "y", 1, INT32_MAX, POLY_INT64);
  PolyUOp *u = poly_uop2(ctx, POLY_OP_SUB, POLY_INT64, x, y, poly_arg_none());
  check_mm(ctx, u, INT64_MIN, INT64_MAX, "i64 sub overflow fallback");
  poly_ctx_destroy(ctx);
  PASS();
}

/* Pinned tinygrad/uop/ops.py:858-865 tracks mathematical integer endpoints
 * even when they exceed a narrow storage dtype. Runtime ALU still wraps at
 * the backend boundary; min/max is a symbolic proof surface, not emulation. */
TEST(sym, minmax_narrow_integer_uses_mathematical_bounds_like_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *u8 = Variable(ctx, "u8", 250, 251, POLY_UINT8);
  PolyUOp *u8_add = poly_uop2(
      ctx, POLY_OP_ADD, POLY_UINT8, u8, poly_uop0(ctx, POLY_OP_CONST, POLY_UINT8, poly_arg_int(10)),
      poly_arg_none()
  );
  check_mm(ctx, u8_add, 260, 261, "uint8 mathematical add");
  PolyUOp *u8_cmp = poly_uop2(
      ctx, POLY_OP_CMPLT, POLY_BOOL, u8_add,
      poly_uop0(ctx, POLY_OP_CONST, POLY_UINT8, poly_arg_int(5)), poly_arg_none()
  );
  PolyUOp *u8_cmp_simplified = poly_graph_rewrite(ctx, u8_cmp, poly_symbolic());
  ASSERT_INT_EQ(u8_cmp_simplified->op, POLY_OP_CONST);
  ASSERT_FALSE(u8_cmp_simplified->arg.b);

  PolyUOp *i8 = Variable(ctx, "i8", 120, 121, POLY_INT8);
  PolyUOp *i8_add = poly_uop2(
      ctx, POLY_OP_ADD, POLY_INT8, i8, poly_uop0(ctx, POLY_OP_CONST, POLY_INT8, poly_arg_int(10)),
      poly_arg_none()
  );
  check_mm(ctx, i8_add, 130, 131, "int8 mathematical add");
  PolyUOp *i8_cmp = poly_uop2(
      ctx, POLY_OP_CMPLT, POLY_BOOL, i8_add,
      poly_uop0(ctx, POLY_OP_CONST, POLY_INT8, poly_arg_int(0)), poly_arg_none()
  );
  PolyUOp *i8_cmp_simplified = poly_graph_rewrite(ctx, i8_cmp, poly_symbolic());
  ASSERT_INT_EQ(i8_cmp_simplified->op, POLY_OP_CONST);
  ASSERT_FALSE(i8_cmp_simplified->arg.b);

  PolyUOp *u8_sub = poly_uop2(
      ctx, POLY_OP_SUB, POLY_UINT8, Variable(ctx, "u8s", 0, 1, POLY_UINT8),
      poly_uop0(ctx, POLY_OP_CONST, POLY_UINT8, poly_arg_int(2)), poly_arg_none()
  );
  check_mm(ctx, u8_sub, -2, -1, "uint8 mathematical sub");

  PolyUOp *i8_mul = poly_uop2(
      ctx, POLY_OP_MUL, POLY_INT8, Variable(ctx, "i8m", 64, 65, POLY_INT8),
      poly_uop0(ctx, POLY_OP_CONST, POLY_INT8, poly_arg_int(2)), poly_arg_none()
  );
  check_mm(ctx, i8_mul, 128, 130, "int8 mathematical mul");

  PolyUOp *u8_shl = poly_uop2(
      ctx, POLY_OP_SHL, POLY_UINT8, Variable(ctx, "u8l", 128, 129, POLY_UINT8),
      poly_uop0(ctx, POLY_OP_CONST, POLY_UINT8, poly_arg_int(1)), poly_arg_none()
  );
  check_mm(ctx, u8_shl, 256, 258, "uint8 mathematical shl");

  PolyArg u8_add_args[2] = {poly_arg_int(250), poly_arg_int(10)};
  PolyArg i8_add_args[2] = {poly_arg_int(120), poly_arg_int(10)};
  ASSERT_INT_EQ(poly_exec_alu(POLY_OP_ADD, POLY_UINT8, u8_add_args, 2, true).i, 4);
  ASSERT_INT_EQ(poly_exec_alu(POLY_OP_ADD, POLY_INT8, i8_add_args, 2, true).i, -126);

  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_int64_division_overflow_falls_back_to_dtype) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = Variable(ctx, "x", INT64_MIN, INT64_MIN, POLY_INT64);
  PolyUOp *neg_one = poly_uop0(ctx, POLY_OP_CONST, POLY_INT64, poly_arg_int(-1));
  PolyUOp *cdiv_u = poly_uop2(ctx, POLY_OP_IDIV, POLY_INT64, x, neg_one, poly_arg_none());
  PolyUOp *floordiv_u = poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_INT64, x, neg_one, poly_arg_none());
  check_mm(ctx, cdiv_u, INT64_MIN, INT64_MAX, "i64 cdiv overflow fallback");
  check_mm(ctx, floordiv_u, INT64_MIN, INT64_MAX, "i64 floordiv overflow fallback");
  poly_ctx_destroy(ctx);
  PASS();
}

/* MUL (4-corner) */
TEST(sym, minmax_mul_r_3) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(
      ctx, POLY_OP_MUL, POLY_INT32, mk_range(ctx, 10, 0), mk_const(ctx, 3), poly_arg_none()
  );
  check_mm(ctx, u, 0, 27, "r*3");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_mul_r_neg2) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(
      ctx, POLY_OP_MUL, POLY_INT32, mk_range(ctx, 10, 0), mk_const(ctx, -2), poly_arg_none()
  );
  check_mm(ctx, u, -18, 0, "r*-2");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_mul_dv_dvn_mixed_signs) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *dv = mk_dvar(ctx, "x", 2, 7);
  PolyUOp *dvn = mk_dvar(ctx, "y", -3, 4);
  PolyUOp *u = poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, dv, dvn, poly_arg_none());
  /* corners: 2*-3=-6, 2*4=8, 7*-3=-21, 7*4=28 -> [-21, 28] */
  check_mm(ctx, u, -21, 28, "dv*dvn");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_mul_dvn_dvn_squared) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *dvn = mk_dvar(ctx, "y", -3, 4);
  PolyUOp *u = poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, dvn, dvn, poly_arg_none());
  /* corners: -3*-3=9, -3*4=-12, 4*-3=-12, 4*4=16 -> [-12, 16] */
  check_mm(ctx, u, -12, 16, "dvn*dvn");
  poly_ctx_destroy(ctx);
  PASS();
}

/* IDIV */
TEST(sym, minmax_idiv_r_3) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(
      ctx, POLY_OP_IDIV, POLY_INT32, mk_range(ctx, 10, 0), mk_const(ctx, 3), poly_arg_none()
  );
  check_mm(ctx, u, 0, 3, "r//3");
  poly_ctx_destroy(ctx);
  PASS();
}

/* MOD */
TEST(sym, minmax_mod_r_3) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(
      ctx, POLY_OP_MOD, POLY_INT32, mk_range(ctx, 10, 0), mk_const(ctx, 3), poly_arg_none()
  );
  check_mm(ctx, u, 0, 2, "r%3");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_mod_dvn_3_mixed) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *dvn = mk_dvar(ctx, "y", -3, 4);
  PolyUOp *u = poly_uop2(ctx, POLY_OP_MOD, POLY_INT32, dvn, mk_const(ctx, 3), poly_arg_none());
  /* tinygrad: dvn vmin=-3 vmax=4, mod 3 -> middle branch -> [-2, 2] */
  check_mm(ctx, u, -2, 2, "dvn%3");
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, minmax_floordiv_floormod_dvn_3_mixed) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *dvn = mk_dvar(ctx, "y", -3, 4);
  PolyUOp *q = poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_INT32, dvn, mk_const(ctx, 3), poly_arg_none());
  PolyUOp *r = poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_INT32, dvn, mk_const(ctx, 3), poly_arg_none());
  check_mm(ctx, q, -1, 1, "floor(dvn/3)");
  check_mm(ctx, r, 0, 2, "floor_mod(dvn,3)");
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, minmax_empty_range_divmod_is_zero_like_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *empty = mk_range(ctx, 0, 0);
  PolyUOp *three = mk_const(ctx, 3);
  PolyUOp *cdiv = poly_uop2(ctx, POLY_OP_IDIV, POLY_INT32, empty, three, poly_arg_none());
  PolyUOp *floordiv = poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_INT32, empty, three, poly_arg_none());
  PolyUOp *floormod = poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_INT32, empty, three, poly_arg_none());
  check_mm(ctx, cdiv, 0, 0, "empty CDIV 3");
  check_mm(ctx, floordiv, 0, 0, "empty FLOORDIV 3");
  check_mm(ctx, floormod, 0, 0, "empty FLOORMOD 3");
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, minmax_negative_int64_divisor_bounds_are_overflow_safe) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *numerator = Variable(ctx, "numerator", 0, 1, POLY_INT64);
  PolyUOp *divisor = Variable(ctx, "negative_divisor", INT64_MIN, -1, POLY_INT64);
  PolyUOp *cmod = poly_uop2(ctx, POLY_OP_MOD, POLY_INT64, numerator, divisor, poly_arg_none());
  PolyUOp *floormod =
      poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_INT64, numerator, divisor, poly_arg_none());
  check_mm(ctx, cmod, 0, INT64_MAX, "CMOD([0,1], [INT64_MIN,-1])");
  check_mm(ctx, floormod, -INT64_MAX, 0, "FLOORMOD([0,1], [INT64_MIN,-1])");
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, fold_divmod_uint64_additive_overflow_declines_without_ub) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *max_i64 = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT64, poly_arg_int(INT64_MAX));
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT64, poly_arg_int(1));
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT64, poly_arg_int(2));
  PolyUOp *sum = poly_uop2(ctx, POLY_OP_ADD, POLY_UINT64, max_i64, one, poly_arg_none());
  PolyUOp *div = poly_uop2(ctx, POLY_OP_IDIV, POLY_UINT64, sum, two, poly_arg_none());
  PolyUOp *rewritten = simplify(ctx, div);
  ASSERT_NOT_NULL(rewritten);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, fold_divmod_uint64_congruence_overflow_declines_without_ub) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *base = Variable(ctx, "base", 0, 4, POLY_UINT64);
  PolyUOp *factor = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT64, poly_arg_int(INT64_MAX / 3));
  PolyUOp *denominator = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT64, poly_arg_int(INT64_MAX));
  PolyUOp *numerator = poly_uop2(ctx, POLY_OP_MUL, POLY_UINT64, base, factor, poly_arg_none());
  PolyUOp *div = poly_uop2(ctx, POLY_OP_IDIV, POLY_UINT64, numerator, denominator, poly_arg_none());
  PolyUOp *rewritten = simplify(ctx, div);
  ASSERT_NOT_NULL(rewritten);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, uint64_minmax_point_does_not_fold_signed_surrogate_bounds) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = Variable(ctx, "x", INT64_MIN, INT64_MIN + 1, POLY_UINT64);
  PolyUOp *denominator = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT64, poly_arg_int(INT64_MAX));
  PolyUOp *div = poly_uop2(ctx, POLY_OP_IDIV, POLY_UINT64, x, denominator, poly_arg_none());
  PolyUOp *rewritten = simplify(ctx, div);
  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(rewritten->op, POLY_OP_IDIV);

  PolyArg lower_args[2] = {poly_arg_int(INT64_MIN), poly_arg_int(INT64_MAX)};
  PolyArg upper_args[2] = {poly_arg_int(INT64_MIN + 1), poly_arg_int(INT64_MAX)};
  /* These are raw negative CONST args, not representable positive uint64
   * values. Pinned CDIV returns -1, then uint64 truncation is stored as the
   * signed all-ones surrogate until PG-PARITY-020 closes. */
  ASSERT_INT_EQ(poly_exec_alu(POLY_OP_IDIV, POLY_UINT64, lower_args, 2, true).i, -1);
  ASSERT_INT_EQ(poly_exec_alu(POLY_OP_IDIV, POLY_UINT64, upper_args, 2, true).i, -1);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, unsigned_div_all_ones_does_not_rewrite_to_negation) {
  const PolyDType unsigned_dtypes[] = {POLY_UINT8, POLY_UINT16, POLY_UINT32, POLY_UINT64};
  const int64_t truncated_neg_one[] = {UINT8_MAX, UINT16_MAX, UINT32_MAX, -1};
  for (int i = 0; i < (int)(sizeof(unsigned_dtypes) / sizeof(unsigned_dtypes[0])); i++) {
    PolyCtx *ctx = poly_ctx_new();
    PolyDType dtype = unsigned_dtypes[i];
    PolyUOp *x = Variable(ctx, "x", 2, 3, dtype);
    PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, dtype, poly_arg_int(1));
    PolyUOp *numerator = poly_uop2(ctx, POLY_OP_SHR, dtype, x, one, poly_arg_none());
    PolyUOp *all_ones = poly_uop0(ctx, POLY_OP_CONST, dtype, poly_arg_int(-1));
    PolyUOp *div = poly_uop2(ctx, POLY_OP_IDIV, dtype, numerator, all_ones, poly_arg_none());
    PolyUOp *mod = poly_uop2(ctx, POLY_OP_MOD, dtype, numerator, all_ones, poly_arg_none());
    PolyUOp *rewritten_div = simplify(ctx, div);
    PolyUOp *rewritten_mod = simplify(ctx, mod);
    ASSERT_NOT_NULL(rewritten_div);
    ASSERT_NOT_NULL(rewritten_mod);
    ASSERT_INT_EQ(rewritten_div->op, POLY_OP_IDIV);
    ASSERT_INT_EQ(rewritten_mod->op, POLY_OP_MOD);

    PolyArg endpoint[2] = {poly_arg_int(1), poly_arg_int(-1)};
    ASSERT_INT_EQ(poly_exec_alu(POLY_OP_IDIV, dtype, endpoint, 2, true).i, truncated_neg_one[i]);
    ASSERT_INT_EQ(poly_exec_alu(POLY_OP_MOD, dtype, endpoint, 2, true).i, 0);
    poly_ctx_destroy(ctx);
  }
  PASS();
}

TEST(sym, cdiv_stays_raw_while_floordiv_uses_floor_identities) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = Variable(ctx, "x", 2, 3, POLY_INT32);
  PolyUOp *neg_one = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(-1));
  PolyUOp *four = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(4));

  PolyUOp *cdiv_neg = poly_uop2(ctx, POLY_OP_CDIV, POLY_INT32, x, neg_one, poly_arg_none());
  PolyUOp *floordiv_neg = poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_INT32, x, neg_one, poly_arg_none());
  PolyUOp *cdiv_four = poly_uop2(ctx, POLY_OP_CDIV, POLY_INT32, x, four, poly_arg_none());
  PolyUOp *floordiv_four = poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_INT32, x, four, poly_arg_none());

  PolyUOp *rewritten_cdiv_neg = poly_graph_rewrite(ctx, cdiv_neg, poly_symbolic());
  PolyUOp *rewritten_floordiv_neg = poly_graph_rewrite(ctx, floordiv_neg, poly_symbolic());
  PolyUOp *rewritten_cdiv_four = poly_graph_rewrite(ctx, cdiv_four, poly_symbolic());
  PolyUOp *rewritten_floordiv_four = poly_graph_rewrite(ctx, floordiv_four, poly_symbolic());
  ASSERT_INT_EQ(rewritten_cdiv_neg->op, POLY_OP_CDIV);
  ASSERT_INT_EQ(rewritten_floordiv_neg->op, POLY_OP_NEG);
  ASSERT_PTR_EQ(rewritten_floordiv_neg->src[0], x);
  ASSERT_INT_EQ(rewritten_cdiv_four->op, POLY_OP_CDIV);
  ASSERT_INT_EQ(rewritten_floordiv_four->op, POLY_OP_CONST);
  ASSERT_INT_EQ(rewritten_floordiv_four->arg.i, 0);

  PolyUOp *min_x = Variable(ctx, "min_x", INT64_MIN, INT64_MIN + 1, POLY_INT64);
  PolyUOp *min_neg_one = poly_uop0(ctx, POLY_OP_CONST, POLY_INT64, poly_arg_int(-1));
  PolyUOp *min_floordiv =
      poly_uop2(ctx, POLY_OP_FLOORDIV, POLY_INT64, min_x, min_neg_one, poly_arg_none());
  ASSERT_INT_EQ(simplify(ctx, min_floordiv)->op, POLY_OP_FLOORDIV);

  poly_ctx_destroy(ctx);
  PASS();
}

/* SHL / SHR */
TEST(sym, minmax_shl_r_2) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(
      ctx, POLY_OP_SHL, POLY_INT32, mk_range(ctx, 10, 0), mk_const(ctx, 2), poly_arg_none()
  );
  check_mm(ctx, u, 0, 36, "r<<2");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_shl_int64_overflow_falls_back_to_dtype) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = Variable(ctx, "x", INT64_MAX / 2, INT64_MAX, POLY_INT64);
  PolyUOp *u = poly_uop2(
      ctx, POLY_OP_SHL, POLY_INT64, x, poly_uop0(ctx, POLY_OP_CONST, POLY_INT64, poly_arg_int(2)),
      poly_arg_none()
  );
  check_mm(ctx, u, INT64_MIN, INT64_MAX, "i64 shl overflow fallback");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_shr_r_1) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(
      ctx, POLY_OP_SHR, POLY_INT32, mk_range(ctx, 10, 0), mk_const(ctx, 1), poly_arg_none()
  );
  check_mm(ctx, u, 0, 4, "r>>1");
  poly_ctx_destroy(ctx);
  PASS();
}

/* XOR with -1 (bitwise NOT) */
TEST(sym, minmax_xor_r_neg1) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(
      ctx, POLY_OP_XOR, POLY_INT32, mk_range(ctx, 10, 0), mk_const(ctx, -1), poly_arg_none()
  );
  /* ~r where r in [0,9] -> [~9, ~0] = [-10, -1] */
  check_mm(ctx, u, -10, -1, "r xor -1");
  poly_ctx_destroy(ctx);
  PASS();
}

/* AND (int with non-negative const) */
TEST(sym, minmax_and_r_5) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(
      ctx, POLY_OP_AND, POLY_INT32, mk_range(ctx, 10, 0), mk_const(ctx, 5), poly_arg_none()
  );
  /* tinygrad: r vmin=0 (>=0), vmax=min(9,5)=5 -> [0, 5] */
  check_mm(ctx, u, 0, 5, "r & 5");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_and_negative_var_nonnegative_mask) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = mk_dvar(ctx, "x", -100, 100);

  /* Port of tinygrad test_uop_vmin_vmax.py:
   * when the mask has no sign bit, x & mask is known non-negative even if x
   * spans negative and positive values. A negative mask falls back to dtype
   * bounds because the sign bit can survive. */
  PolyUOp *mask511 =
      poly_uop2(ctx, POLY_OP_AND, POLY_INT32, x, mk_const(ctx, 511), poly_arg_none());
  check_mm(ctx, mask511, 0, 511, "[-100..100] & 511");

  PolyUOp *mask_all =
      poly_uop2(ctx, POLY_OP_AND, POLY_INT32, x, mk_const(ctx, -1), poly_arg_none());
  check_mm(ctx, mask_all, INT32_MIN, INT32_MAX, "[-100..100] & -1");

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, minmax_special_with_variable_source) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *dv = Variable(ctx, "i", 1, 10, POLY_INT32);
  PolyUOp *special = poly_uop1(ctx, POLY_OP_SPECIAL, POLY_INT32, dv, poly_arg_str("gidx0"));

  /* tinygrad SPECIAL uses the source bound as an extent, so [1..10] becomes
   * an index-style range [0..9]. */
  check_mm(ctx, special, 0, 9, "SPECIAL(Variable[1..10])");
  poly_ctx_destroy(ctx);
  PASS();
}

/* MAX */
TEST(sym, minmax_max_r_5) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(
      ctx, POLY_OP_MAX, POLY_INT32, mk_range(ctx, 10, 0), mk_const(ctx, 5), poly_arg_none()
  );
  check_mm(ctx, u, 5, 9, "max(r,5)");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_nested_max_min_clamp) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = mk_dvar(ctx, "x", 0, 10);
  PolyUOp *lo = poly_uop2(ctx, POLY_OP_MAX, POLY_INT32, x, mk_const(ctx, 5), poly_arg_none());
  PolyUOp *neg_one = mk_const(ctx, -1);
  PolyUOp *not_lo = poly_uop2(ctx, POLY_OP_XOR, POLY_INT32, lo, neg_one, poly_arg_none());
  PolyUOp *not_8 =
      poly_uop2(ctx, POLY_OP_XOR, POLY_INT32, mk_const(ctx, 8), neg_one, poly_arg_none());
  PolyUOp *inner = poly_uop2(ctx, POLY_OP_MAX, POLY_INT32, not_lo, not_8, poly_arg_none());
  PolyUOp *clamped = poly_uop2(ctx, POLY_OP_XOR, POLY_INT32, inner, neg_one, poly_arg_none());

  /* Port of tinygrad test_vmin_vmax_nested_min_max:
   * x.maximum(5).minimum(8) renders as (max((max(x, 5)^-1), -9)^-1)
   * at the UOp level, then narrows to [5..8]. */
  check_mm(ctx, clamped, 5, 8, "min(max(x,5),8)");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_max_dv_5) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(
      ctx, POLY_OP_MAX, POLY_INT32, mk_dvar(ctx, "x", 2, 7), mk_const(ctx, 5), poly_arg_none()
  );
  check_mm(ctx, u, 5, 7, "max(dv,5)");
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, minmax_idiv_negative_constant) {
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *pos = mk_dvar(ctx, "pos", 10, 20);
  PolyUOp *pos_div_neg =
      poly_uop2(ctx, POLY_OP_IDIV, POLY_INT32, pos, mk_const(ctx, -2), poly_arg_none());
  check_mm(ctx, pos_div_neg, -10, -5, "[10..20]//-2");

  PolyUOp *neg = mk_dvar(ctx, "neg", -20, -10);
  PolyUOp *neg_div_neg =
      poly_uop2(ctx, POLY_OP_IDIV, POLY_INT32, neg, mk_const(ctx, -3), poly_arg_none());
  check_mm(ctx, neg_div_neg, 3, 6, "[-20..-10]//-3");

  poly_ctx_destroy(ctx);
  PASS();
}

/* CMPLT (bool bounds) */
TEST(sym, minmax_cmplt_r_5) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(
      ctx, POLY_OP_CMPLT, POLY_BOOL, mk_range(ctx, 10, 0), mk_const(ctx, 5), poly_arg_none()
  );
  /* (a1<b0) = (9<5)=false=0, (a0<b1) = (0<5)=true=1 -> [0,1] */
  check_mm(ctx, u, 0, 1, "r<5");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_cmplt_r_neg1_static_false) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(
      ctx, POLY_OP_CMPLT, POLY_BOOL, mk_range(ctx, 10, 0), mk_const(ctx, -1), poly_arg_none()
  );
  /* (a1<b0) = (9<-1)=false=0, (a0<b1) = (0<-1)=false=0 -> [0,0] */
  check_mm(ctx, u, 0, 0, "r<-1");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_cmplt_r_20_static_true) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(
      ctx, POLY_OP_CMPLT, POLY_BOOL, mk_range(ctx, 10, 0), mk_const(ctx, 20), poly_arg_none()
  );
  /* (a1<b0) = (9<20)=true=1, (a0<b1) = (0<20)=true=1 -> [1,1] */
  check_mm(ctx, u, 1, 1, "r<20");
  poly_ctx_destroy(ctx);
  PASS();
}

/* CMPNE */
TEST(sym, minmax_cmpne_r_neg1) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop2(
      ctx, POLY_OP_CMPNE, POLY_BOOL, mk_range(ctx, 10, 0), mk_const(ctx, -1), poly_arg_none()
  );
  /* def_ne: (a1<b0)||(b1<a0) = (9<-1)||(-1<0) = false||true = true
   * all_eq: false (a0=0 != a1=9). vmin=1, vmax=1 */
  check_mm(ctx, u, 1, 1, "r!=-1");
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, minmax_stack_uses_lane_bounds_like_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *lanes[4] = {
      mk_const(ctx, -2),
      mk_const(ctx, 0),
      mk_const(ctx, 5),
      mk_const(ctx, 10),
  };
  PolyUOp *stack = poly_uop_stack(ctx, lanes, 4);
  check_mm(ctx, stack, -2, 10, "STACK(-2,0,5,10)");
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, minmax_vector_bool_index_and_gate_stays_dynamic) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *base = poly_test_uop_param(ctx, POLY_FLOAT32, 4, 0, POLY_ADDR_GLOBAL);
  PolyUOp *exponent = poly_test_uop_param(ctx, POLY_FLOAT32, 4, 1, POLY_ADDR_GLOBAL);
  PolyUOp *zero_lanes[4] = {
      poly_const_float(ctx, 0.0),
      poly_const_float(ctx, 0.0),
      poly_const_float(ctx, 0.0),
      poly_const_float(ctx, 0.0),
  };
  PolyUOp *zero_vec = poly_uop_stack(ctx, zero_lanes, 4);
  PolyUOp *base_eq = poly_uop2(ctx, POLY_OP_CMPEQ, POLY_BOOL, base, zero_vec, poly_arg_none());
  PolyUOp *exp_eq = poly_uop2(ctx, POLY_OP_CMPEQ, POLY_BOOL, exponent, zero_vec, poly_arg_none());
  PolyUOp *lane_idx = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *base_lane = poly_uop_index(ctx, base_eq, &lane_idx, 1);
  PolyUOp *exp_lane = poly_uop_index(ctx, exp_eq, &lane_idx, 1);
  PolyUOp *both = poly_uop2(ctx, POLY_OP_AND, POLY_BOOL, base_lane, exp_lane, poly_arg_none());
  PolyUOp *as_int = poly_uop1(ctx, POLY_OP_CAST, POLY_INT32, both, poly_arg_none());
  PolyUOp *as_float = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, as_int, poly_arg_none());
  PolyUOp *gate = poly_uop2(
      ctx, POLY_OP_CMPNE, POLY_BOOL, as_float, poly_const_float(ctx, 0.0), poly_arg_none()
  );

  check_mm(ctx, base_eq, 0, 1, "vector CMPEQ");
  check_mm(ctx, base_lane, 0, 1, "INDEX(vector CMPEQ)");
  check_mm(ctx, both, 0, 1, "AND of vector comparison lanes");
  check_mm(ctx, gate, 0, 1, "casted bool gate");

  PolyUOp *where = poly_uop3(
      ctx, POLY_OP_WHERE, POLY_FLOAT32, gate, poly_const_float(ctx, 1.0),
      poly_const_float(ctx, 8.0), poly_arg_none()
  );
  PolyUOp *rewritten = simplify(ctx, where);
  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(rewritten->op, POLY_OP_WHERE);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, minmax_float_comparisons_with_infinity_stay_dynamic) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_uop0(ctx, POLY_OP_PARAM, POLY_FLOAT32, poly_arg_int(0));
  PolyUOp *negative_inf = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(-INFINITY));
  PolyUOp *positive_inf = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(INFINITY));
  PolyUOp *reciprocal = poly_uop1(ctx, POLY_OP_RECIPROCAL, POLY_FLOAT32, x, poly_arg_none());
  PolyUOp *not_negative_inf =
      poly_uop2(ctx, POLY_OP_CMPNE, POLY_BOOL, reciprocal, negative_inf, poly_arg_none());
  PolyUOp *below_positive_inf =
      poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, x, positive_inf, poly_arg_none());

  PolyUOp *rewritten_ne = simplify(ctx, not_negative_inf);
  PolyUOp *rewritten_lt = simplify(ctx, below_positive_inf);
  ASSERT_NOT_NULL(rewritten_ne);
  ASSERT_NOT_NULL(rewritten_lt);
  ASSERT_INT_EQ(rewritten_ne->op, POLY_OP_CMPNE);
  ASSERT_INT_EQ(rewritten_lt->op, POLY_OP_CMPLT);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, reciprocal_self_product_matches_pinned_division_spelling) {
  /* Pinned ElementwiseMixin.div constructs x/x as
   * MUL(x, RECIPROCAL(x)); symbolic.py:136-145 rewrites it to const_like(1). */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *x = poly_uop0(ctx, POLY_OP_PARAM, POLY_FLOAT16, poly_arg_int(0));
  PolyUOp *reciprocal = poly_uop1(ctx, POLY_OP_RECIPROCAL, POLY_FLOAT16, x, poly_arg_none());
  PolyUOp *division = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT16, x, reciprocal, poly_arg_none());

  PolyUOp *rewritten = simplify(ctx, division);
  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(rewritten->op, POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(rewritten->dtype, POLY_FLOAT16));
  ASSERT_TRUE(rewritten->arg.kind == POLY_ARG_FLOAT);
  ASSERT_FLOAT_EQ(rewritten->arg.f, 1.0, 0.0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, zero_division_rules_precede_self_and_zero_folds) {
  /* Tinygrad 2026-08-22/a9069c177a9d uop/symbolic.py:157-166. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(0.0));
  PolyUOp *reciprocal_zero =
      poly_uop1(ctx, POLY_OP_RECIPROCAL, POLY_FLOAT32, zero, poly_arg_none());
  PolyUOp *zero_div_zero =
      poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, zero, reciprocal_zero, poly_arg_none());
  PolyUOp *x = Variable(ctx, "x", -4, 4, POLY_FLOAT32);
  PolyUOp *x_times_zero = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, x, zero, poly_arg_none());
  PolyUOp *zero_product_div_zero =
      poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, x_times_zero, reciprocal_zero, poly_arg_none());

  PolyUOp *direct = simplify(ctx, zero_div_zero);
  PolyUOp *product = simplify(ctx, zero_product_div_zero);
  ASSERT_NOT_NULL(direct);
  ASSERT_NOT_NULL(product);
  ASSERT_INT_EQ(direct->op, POLY_OP_CONST);
  ASSERT_INT_EQ(product->op, POLY_OP_CONST);
  ASSERT_INT_EQ(direct->arg.kind, POLY_ARG_FLOAT);
  ASSERT_INT_EQ(product->arg.kind, POLY_ARG_FLOAT);
  ASSERT_TRUE(isnan(direct->arg.f));
  ASSERT_TRUE(isnan(product->arg.f));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, reciprocal_factorization_matches_tinygrad) {
  /* tinygrad@2026-08-22/a9069c177a9d uop/symbolic.py:461-463. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *x = poly_uop0(ctx, POLY_OP_PARAM, POLY_FLOAT32, poly_arg_int(0));
  PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(2.0));
  PolyUOp *rx = poly_uop1(ctx, POLY_OP_RECIPROCAL, POLY_FLOAT32, x, poly_arg_none());
  PolyUOp *half = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(0.5));

  PolyUOp *xx = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, x, x, poly_arg_none());
  PolyUOp *xxx = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, xx, x, poly_arg_none());
  PolyUOp *xc = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, x, c, poly_arg_none());
  PolyUOp *pair = poly_uop1(ctx, POLY_OP_RECIPROCAL, POLY_FLOAT32, xx, poly_arg_none());
  PolyUOp *triple = poly_uop1(ctx, POLY_OP_RECIPROCAL, POLY_FLOAT32, xxx, poly_arg_none());
  PolyUOp *constant = poly_uop1(ctx, POLY_OP_RECIPROCAL, POLY_FLOAT32, xc, poly_arg_none());

  PolyUOp *pair_expected = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, rx, rx, poly_arg_none());
  PolyUOp *triple_expected =
      poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, pair_expected, rx, poly_arg_none());
  PolyUOp *constant_expected = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, rx, half, poly_arg_none());

  ASSERT_TRUE(poly_pm_rewrite(poly_symbolic(), ctx, pair) == NULL);
  ASSERT_TRUE(poly_pm_rewrite(poly_symbolic(), ctx, triple) == NULL);
  ASSERT_TRUE(poly_pm_rewrite(poly_symbolic(), ctx, constant) == NULL);
  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, pair, poly_sym()), pair_expected);
  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, triple, poly_sym()), triple_expected);
  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, constant, poly_sym()), constant_expected);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, maximum_uses_float_constant_bounds) {
  /* tinygrad@2026-08-22/a9069c177a9d uop/ops.py:1046-1109 and
   * uop/symbolic.py:263 preserve float constants in vmin/vmax comparisons. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *x = poly_uop0(ctx, POLY_OP_PARAM, POLY_FLOAT32, poly_arg_int(0));
  PolyUOp *negative_inf = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(-INFINITY));
  PolyUOp *positive_inf = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(INFINITY));

  PolyUOp *lower = poly_uop2(ctx, POLY_OP_MAX, POLY_FLOAT32, negative_inf, x, poly_arg_none());
  PolyUOp *upper = poly_uop2(ctx, POLY_OP_MAX, POLY_FLOAT32, x, positive_inf, poly_arg_none());
  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, lower, poly_symbolic_simple()), lower);
  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, upper, poly_symbolic_simple()), upper);
  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, lower, poly_symbolic()), x);
  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, upper, poly_symbolic()), positive_inf);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, reciprocal_product_rules_match_pinned_topology) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* Pinned tinygrad/uop/symbolic.py:478-480:
   *   x * d       -> 1-d
   *   x * (d*y)   -> y*(1-d)
   *   x * (d+y)   -> (1-d)+x*y
   * where d = reciprocal(1+x). */
  PolyUOp *x = poly_uop0(ctx, POLY_OP_PARAM, POLY_FLOAT16, poly_arg_int(0));
  PolyUOp *y = poly_uop0(ctx, POLY_OP_PARAM, POLY_FLOAT16, poly_arg_int(1));
  PolyUOp *one = poly_const_like_float(ctx, x, 1.0);
  PolyUOp *neg_one = poly_const_like_float(ctx, x, -1.0);
  PolyUOp *den = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT16, one, x, poly_arg_none());
  PolyUOp *d = poly_uop1(ctx, POLY_OP_RECIPROCAL, POLY_FLOAT16, den, poly_arg_none());
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(y);
  ASSERT_NOT_NULL(one);
  ASSERT_NOT_NULL(neg_one);
  ASSERT_NOT_NULL(den);
  ASSERT_NOT_NULL(d);

  PolyUOp *one_minus_d = poly_uop2(
      ctx, POLY_OP_ADD, POLY_FLOAT16, one,
      poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT16, d, neg_one, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *expressions[3] = {
      poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT16, x, d, poly_arg_none()),
      poly_uop2(
          ctx, POLY_OP_MUL, POLY_FLOAT16, x,
          poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT16, d, y, poly_arg_none()), poly_arg_none()
      ),
      poly_uop2(
          ctx, POLY_OP_MUL, POLY_FLOAT16, x,
          poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT16, d, y, poly_arg_none()), poly_arg_none()
      ),
  };
  PolyUOp *expected[3] = {
      one_minus_d,
      poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT16, y, one_minus_d, poly_arg_none()),
      poly_uop2(
          ctx, POLY_OP_ADD, POLY_FLOAT16, one_minus_d,
          poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT16, x, y, poly_arg_none()), poly_arg_none()
      ),
  };

  for (int i = 0; i < 3; i++) {
    ASSERT_PTR_EQ(poly_graph_rewrite(ctx, expressions[i], poly_symbolic()), expressions[i]);
    PolyUOp *rewritten = poly_graph_rewrite(ctx, expressions[i], poly_sym());
    PolyUOp *canonical_expected = poly_graph_rewrite(ctx, expected[i], poly_sym());
    ASSERT_NOT_NULL(rewritten);
    ASSERT_NOT_NULL(canonical_expected);
    ASSERT_PTR_EQ(rewritten, canonical_expected);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, move_const_to_end_matches_commutative_outer_orientation) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* Pinned tinygrad/uop/symbolic.py:286-287 uses commutative UPat matching:
   *   y op (x op c) -> (x op y) op c
   * The nested same-op subtree can therefore be either outer operand. */
  PolyUOp *x = poly_uop0(ctx, POLY_OP_PARAM, POLY_FLOAT16, poly_arg_int(0));
  PolyUOp *y = poly_uop0(ctx, POLY_OP_PARAM, POLY_FLOAT16, poly_arg_int(1));
  PolyUOp *c = poly_const_like_float(ctx, x, -1.0);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(y);
  ASSERT_NOT_NULL(c);

  PolyOps ops[] = {POLY_OP_ADD, POLY_OP_MUL};
  for (int i = 0; i < 2; i++) {
    PolyUOp *inner = poly_uop2(ctx, ops[i], POLY_FLOAT16, x, c, poly_arg_none());
    PolyUOp *expression = poly_uop2(ctx, ops[i], POLY_FLOAT16, y, inner, poly_arg_none());
    PolyUOp *expected = poly_uop2(
        ctx, ops[i], POLY_FLOAT16, poly_uop2(ctx, ops[i], POLY_FLOAT16, x, y, poly_arg_none()), c,
        poly_arg_none()
    );
    PolyUOp *rewritten = poly_graph_rewrite(ctx, expression, poly_symbolic());
    PolyUOp *canonical_expected = poly_graph_rewrite(ctx, expected, poly_symbolic());
    ASSERT_NOT_NULL(rewritten);
    ASSERT_NOT_NULL(canonical_expected);
    ASSERT_PTR_EQ(rewritten, canonical_expected);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, reciprocal_product_stabilizes_after_commutative_const_move) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* Exact half-GELU spelling at the proven first divergence:
   *   x * (((v*d)*d)*-1), d=1/(1+x)
   * Pinned symbolic.py:286-287 first moves -1 to the end; :478-480 then
   * replaces x*(d*y) with y*(1-d). */
  PolyUOp *z = poly_uop0(ctx, POLY_OP_PARAM, POLY_FLOAT16, poly_arg_int(0));
  PolyUOp *v = poly_uop0(ctx, POLY_OP_PARAM, POLY_FLOAT16, poly_arg_int(1));
  PolyUOp *x = poly_uop1(ctx, POLY_OP_EXP2, POLY_FLOAT16, z, poly_arg_none());
  PolyUOp *one = poly_const_like_float(ctx, x, 1.0);
  PolyUOp *neg_one = poly_const_like_float(ctx, x, -1.0);
  PolyUOp *d = poly_uop1(
      ctx, POLY_OP_RECIPROCAL, POLY_FLOAT16,
      poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT16, one, x, poly_arg_none()), poly_arg_none()
  );
  ASSERT_NOT_NULL(z);
  ASSERT_NOT_NULL(v);
  ASSERT_NOT_NULL(x);
  ASSERT_NOT_NULL(one);
  ASSERT_NOT_NULL(neg_one);
  ASSERT_NOT_NULL(d);

  PolyUOp *vd = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT16, v, d, poly_arg_none());
  PolyUOp *expression = poly_uop2(
      ctx, POLY_OP_MUL, POLY_FLOAT16, x,
      poly_uop2(
          ctx, POLY_OP_MUL, POLY_FLOAT16,
          poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT16, vd, d, poly_arg_none()), neg_one,
          poly_arg_none()
      ),
      poly_arg_none()
  );
  PolyUOp *one_minus_d = poly_uop2(
      ctx, POLY_OP_ADD, POLY_FLOAT16, one,
      poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT16, d, neg_one, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *expected = poly_uop2(
      ctx, POLY_OP_MUL, POLY_FLOAT16,
      poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT16, vd, one_minus_d, poly_arg_none()), neg_one,
      poly_arg_none()
  );

  PolyUOp *rewritten = poly_graph_rewrite(ctx, expression, poly_sym());
  PolyUOp *canonical_expected = poly_graph_rewrite(ctx, expected, poly_sym());
  ASSERT_NOT_NULL(rewritten);
  ASSERT_NOT_NULL(canonical_expected);
  ASSERT_PTR_EQ(rewritten, canonical_expected);

  poly_ctx_destroy(ctx);
  PASS();
}

/* WHERE (int branches) */
TEST(sym, minmax_where_int) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *cond = poly_uop2(
      ctx, POLY_OP_CMPLT, POLY_BOOL, mk_range(ctx, 10, 0), mk_const(ctx, 5), poly_arg_none()
  );
  PolyUOp *u = poly_uop3(
      ctx, POLY_OP_WHERE, POLY_INT32, cond, mk_const(ctx, 3), mk_const(ctx, 5), poly_arg_none()
  );
  check_mm(ctx, u, 3, 5, "WHERE(r<5,3,5)");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_where_dv_dvn) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *cond = poly_uop2(
      ctx, POLY_OP_CMPLT, POLY_BOOL, mk_range(ctx, 10, 0), mk_const(ctx, 5), poly_arg_none()
  );
  PolyUOp *dv = mk_dvar(ctx, "x", 2, 7);
  PolyUOp *dvn = mk_dvar(ctx, "y", -3, 4);
  PolyUOp *u = poly_uop3(ctx, POLY_OP_WHERE, POLY_INT32, cond, dv, dvn, poly_arg_none());
  /* min(2,-3)=-3, max(7,4)=7 -> [-3, 7] */
  check_mm(ctx, u, -3, 7, "WHERE(r<5,dv,dvn)");
  poly_ctx_destroy(ctx);
  PASS();
}

/* CAST (monotone narrowing to signed) */
TEST(sym, minmax_cast_dv_to_int16) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop1(ctx, POLY_OP_CAST, POLY_INT16, mk_dvar(ctx, "x", 2, 7), poly_arg_none());
  check_mm(ctx, u, 2, 7, "CAST dv i32->i16");
  poly_ctx_destroy(ctx);
  PASS();
}
TEST(sym, minmax_cast_r_to_int8) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *u = poly_uop1(ctx, POLY_OP_CAST, POLY_INT8, mk_range(ctx, 10, 0), poly_arg_none());
  check_mm(ctx, u, 0, 9, "CAST r i32->i8");
  poly_ctx_destroy(ctx);
  PASS();
}

/* Diamond / chained (memoization stress) */
TEST(sym, minmax_diamond_r3_mul2) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *r = mk_range(ctx, 10, 0);
  PolyUOp *r3 = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, r, mk_const(ctx, 3), poly_arg_none());
  PolyUOp *u = poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, r3, mk_const(ctx, 2), poly_arg_none());
  /* (r+3)*2 with r in [0,9]: r+3 in [3,12], *2 in [6,24] */
  check_mm(ctx, u, 6, 24, "(r+3)*2");
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, minmax_deep_chain_is_iterative) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *expr = mk_range(ctx, 10, 0);
  PolyUOp *one = mk_const(ctx, 1);
  for (int i = 0; i < 12000; i++) {
    expr = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, expr, one, poly_arg_none());
  }
  check_mm(ctx, expr, 12000, 12009, "deep add chain");
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, minmax_backward_slice_score_uses_rewound_scratch_toposort) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *expr = mk_range(ctx, 10, 0);
  PolyUOp *one = mk_const(ctx, 1);
  for (int i = 0; i < 64; i++)
    expr = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, expr, one, poly_arg_none());

  size_t scratch_before = poly_arena_used(ctx->scratch);
  int64_t lo = 0, hi = 0;
  poly_uop_minmax(ctx, expr, &lo, &hi);
  ASSERT_INT_EQ(lo, 64);
  ASSERT_INT_EQ(hi, 73);
  ASSERT_INT_EQ(poly_arena_used(ctx->scratch), scratch_before);

  poly_ctx_destroy(ctx);
  PASS();
}

/* Phase D end-to-end gate * poly_arange returns a *Tensor-level* UOp graph composed of movement ops
 * (RESHAPE / EXPAND / PAD / PERMUTE / SHRINK) over a CONST + REDUCE_AXIS
 * + ADD. Tinygrad's _min_max has no special case for any of these (they
 * live in GroupOp.Movement, not GroupOp.Binary), so the bounds at this
 * level fall through to the dtype default. Tight bounds emerge only after
 * rangeify replaces movement ops with RANGE/INDEX — which is exactly where
 * Phase D's reduce_collapse driver runs.
 *
 * This test is therefore a smoke check: poly_uop_minmax must not crash
 * on a real arange graph and must respect the dtype-bounds fallback. */
TEST(sym, minmax_arange_0_to_5_smoke) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *ar = poly_arange(ctx, 0.0, 5.0, 1.0);
  ASSERT_NOT_NULL(ar);
  int64_t lo, hi;
  poly_uop_minmax(ctx, ar, &lo, &hi);
  /* Loose bounds: Tensor-level graph hasn't been rangeified yet. */
  ASSERT_TRUE(lo <= 0);
  ASSERT_TRUE(hi >= 4);
  poly_ctx_destroy(ctx);
  PASS();
}

/* PolyUOpCache lifetime + reuse */
TEST(sym, minmax_cache_reuse_yields_same) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOpCache *c = poly_uop_cache_new();
  ASSERT_NOT_NULL(c);
  PolyUOp *u = poly_uop2(
      ctx, POLY_OP_MUL, POLY_INT32, mk_dvar(ctx, "x", 2, 7), mk_dvar(ctx, "y", -3, 4),
      poly_arg_none()
  );
  int64_t lo1, hi1, lo2, hi2;
  poly_uop_minmax_ex(ctx, u, c, &lo1, &hi1);
  poly_uop_minmax_ex(ctx, u, c, &lo2, &hi2); /* second query should hit cache */
  ASSERT_INT_EQ(lo1, lo2);
  ASSERT_INT_EQ(hi1, hi2);
  ASSERT_INT_EQ(lo1, -21);
  ASSERT_INT_EQ(hi1, 28);
  poly_uop_cache_destroy(c);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, minmax_explicit_cache_does_not_allocate_arena_values) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOpCache *c = poly_uop_cache_new();
  ASSERT_NOT_NULL(c);

  PolyUOp *x = mk_dvar(ctx, "x", 0, 1023);
  PolyUOp *y = mk_dvar(ctx, "y", -7, 11);
  PolyUOp *u = x;
  for (int i = 0; i < 8; i++) {
    PolyUOp *k = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(i + 1));
    PolyUOp *a = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, u, k, poly_arg_none());
    PolyUOp *b = poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, y, k, poly_arg_none());
    u = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, a, b, poly_arg_none());
  }

  PolyCtxStats before = {0}, after = {0};
  ASSERT_INT_EQ(poly_ctx_stats(ctx, &before), 0);

  int64_t lo = 0, hi = 0;
  poly_uop_minmax_ex(ctx, u, c, &lo, &hi);

  ASSERT_INT_EQ(poly_ctx_stats(ctx, &after), 0);
  ASSERT_INT_EQ(after.arena_bytes, before.arena_bytes);
  ASSERT_INT_EQ(lo, -216);
  ASSERT_INT_EQ(hi, 1455);

  poly_uop_cache_destroy(c);
  poly_ctx_destroy(ctx);
  PASS();
}

/* poly_logical_not / bool comparison parity (Phase A P5) *
 * Tinygrad's `logical_not()` (mixin/elementwise.py:25-33) lowers to
 *   CMPNE(CAST(x, bool), CONST(true))
 * which symbolic.py:126 collapses to
 *   CMPNE(x, CONST(true))           when x is already bool.
 *
 * Polygrad's poly_logical_not is the same form (CMPNE-with-true). All
 * comparison helpers (poly_le, poly_ge, poly_eq) return bool via this
 * canonical NOT, mirroring tinygrad's mixin/elementwise.py:240-250.
 *
 * Captured against tinygrad in test/parity_scripts/tg_logical_not_gt.py. */
TEST(sym, logical_not_canonical_form) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *r = mk_range(ctx, 5, 0);
  PolyUOp *c3 = mk_const(ctx, 3);
  PolyUOp *lt = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, r, c3, poly_arg_none());
  PolyUOp *not_lt = poly_logical_not(ctx, lt);
  /* Shape: CMPNE(CMPLT(...), CONST(true)) */
  ASSERT_INT_EQ(not_lt->op, POLY_OP_CMPNE);
  ASSERT_TRUE(poly_dtype_eq(not_lt->dtype, POLY_BOOL));
  ASSERT_INT_EQ(not_lt->n_src, 2);
  ASSERT_PTR_EQ(not_lt->src[0], lt);
  ASSERT_INT_EQ(not_lt->src[1]->op, POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(not_lt->src[1]->dtype, POLY_BOOL));
  ASSERT_TRUE(not_lt->src[1]->arg.b == true);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, poly_le_returns_bool) {
  /* poly_le must return bool (was previously float WHERE(0,1) before P5). */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = mk_range(ctx, 5, 0);
  PolyUOp *b = mk_const(ctx, 3);
  PolyUOp *le = poly_le(ctx, a, b);
  ASSERT_TRUE(poly_dtype_eq(le->dtype, POLY_BOOL));
  /* Structure: CMPNE(CMPLT(b, a), CONST(true)) per the (b<a).logical_not() form */
  ASSERT_INT_EQ(le->op, POLY_OP_CMPNE);
  ASSERT_INT_EQ(le->src[0]->op, POLY_OP_CMPLT);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, poly_ge_returns_bool) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = mk_range(ctx, 5, 0);
  PolyUOp *b = mk_const(ctx, 3);
  PolyUOp *ge = poly_ge(ctx, a, b);
  ASSERT_TRUE(poly_dtype_eq(ge->dtype, POLY_BOOL));
  ASSERT_INT_EQ(ge->op, POLY_OP_CMPNE);
  ASSERT_INT_EQ(ge->src[0]->op, POLY_OP_CMPLT);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, poly_eq_returns_bool) {
  /* poly_eq = (a != b).logical_not() = CMPNE(CMPNE(a,b), true). */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = mk_range(ctx, 5, 0);
  PolyUOp *b = mk_const(ctx, 3);
  PolyUOp *eq = poly_eq(ctx, a, b);
  ASSERT_TRUE(poly_dtype_eq(eq->dtype, POLY_BOOL));
  ASSERT_INT_EQ(eq->op, POLY_OP_CMPNE);
  ASSERT_INT_EQ(eq->src[0]->op, POLY_OP_CMPNE);
  poly_ctx_destroy(ctx);
  PASS();
}

/* CAST(CONST) constant fold parity (Phase D regression coverage) *
 * Mirrors test/parity_scripts/tg_cast_const_fold_gt.py cases A-E.
 *
 * History: Phase D's pm_reduce_unparented produces
 *   MUL(CONST_float, CAST(CONST_int -> float))
 * which symbolic_simple folds via rule_cast_const + rule_const_fold_binary.
 * The original poly_const_like (src/uop/upat.c) blindly copied the source
 * CONST's PolyArg into a CONST tagged with the new dtype, producing a
 * CONST(dtype=float, arg.kind=INT) — a tagged-union mismatch that the
 * codegen misread as a denormal/zero. The fix: poly_const_like now
 * normalizes the value through poly_arg_float / _int / _bool dispatch,
 * matching tinygrad's DType.const(b) at dtype.py:92-100. These tests
 * lock the fix in place. */

TEST(sym, cast_const_int_to_float) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(5));
  PolyUOp *cast = poly_cast(ctx, c, POLY_FLOAT32);
  PolyUOp *folded = simplify(ctx, cast);
  ASSERT_TRUE(folded->op == POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(folded->dtype, POLY_FLOAT32));
  ASSERT_TRUE(folded->arg.kind == POLY_ARG_FLOAT);
  ASSERT_TRUE(folded->arg.f == 5.0);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, vector_stack_cast_remains_explicit_like_tinygrad) {
  /* Pinned symbolic.py:148 folds CAST only over UPat.cvar (scalar CONST).
   * A vector STACK is not a cvar and retains its explicit CAST. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(1));
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(2));
  PolyUOp *lanes[2] = {one, two};
  PolyUOp *stack = poly_uop(ctx, POLY_OP_STACK, POLY_INT32, lanes, 2, poly_arg_none());
  PolyUOp *cast = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, stack, poly_arg_none());
  PolyUOp *rewritten = simplify(ctx, cast);
  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(rewritten->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(rewritten->dtype, POLY_FLOAT32));
  ASSERT_PTR_EQ(rewritten->src[0], stack);
  ASSERT_TRUE(poly_dtype_eq(stack->dtype, POLY_INT32));
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, cast_const_float_to_int) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(3.7));
  PolyUOp *cast = poly_cast(ctx, c, POLY_INT32);
  PolyUOp *folded = simplify(ctx, cast);
  ASSERT_TRUE(folded->op == POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(folded->dtype, POLY_INT32));
  ASSERT_TRUE(folded->arg.kind == POLY_ARG_INT);
  ASSERT_INT_EQ((int)folded->arg.i, 3); /* truncates to 3 */
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, cast_const_large_int_to_uint64_preserves_integer_bits) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t big = 9007199254740993LL;
  PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, POLY_INT64, poly_arg_int(big));
  PolyUOp *cast = poly_cast(ctx, c, POLY_UINT64);
  PolyUOp *folded = simplify(ctx, cast);
  ASSERT_TRUE(folded->op == POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(folded->dtype, POLY_UINT64));
  ASSERT_TRUE(folded->arg.kind == POLY_ARG_INT);
  ASSERT_TRUE(folded->arg.i == big);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, cast_const_bool_to_float) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, POLY_BOOL, poly_arg_bool(true));
  PolyUOp *cast = poly_cast(ctx, c, POLY_FLOAT32);
  PolyUOp *folded = simplify(ctx, cast);
  ASSERT_TRUE(folded->op == POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(folded->dtype, POLY_FLOAT32));
  ASSERT_TRUE(folded->arg.kind == POLY_ARG_FLOAT);
  ASSERT_TRUE(folded->arg.f == 1.0);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, cast_const_zero_int_to_bool) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *c = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *cast = poly_cast(ctx, c, POLY_BOOL);
  PolyUOp *folded = simplify(ctx, cast);
  ASSERT_TRUE(folded->op == POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(folded->dtype, POLY_BOOL));
  ASSERT_TRUE(folded->arg.kind == POLY_ARG_BOOL);
  ASSERT_TRUE(folded->arg.b == false);
  poly_ctx_destroy(ctx);
  PASS();
}

/* The exact MUL(CONST_float, CAST(CONST_int)) shape that pm_reduce_unparented
 * produces and that triggered the original expand_reduce_e2e regression. */
TEST(sym, mul_float_cast_int_const_fold) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *one_f = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *five_i = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(5));
  PolyUOp *cast = poly_cast(ctx, five_i, POLY_FLOAT32);
  PolyUOp *mul = poly_alu2(ctx, POLY_OP_MUL, one_f, cast);
  PolyUOp *folded = simplify(ctx, mul);
  ASSERT_TRUE(folded->op == POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(folded->dtype, POLY_FLOAT32));
  ASSERT_TRUE(folded->arg.kind == POLY_ARG_FLOAT);
  ASSERT_TRUE(folded->arg.f == 5.0);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, double_logical_not_idempotent_via_minmax) {
  /* Tinygrad symbolic.py:91: x.logical_not().logical_not() -> x.
   * Polygrad still doesn't carry a dedicated double-not rewrite, but the
   * bound semantics must match even without structural folding. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *r = mk_range(ctx, 5, 0);
  PolyUOp *lt = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, r, mk_const(ctx, 3), poly_arg_none());
  PolyUOp *not_lt = poly_logical_not(ctx, lt);
  PolyUOp *not_not_lt = poly_logical_not(ctx, not_lt);
  int64_t lo_a, hi_a, lo_b, hi_b;
  poly_uop_minmax(ctx, lt, &lo_a, &hi_a);
  poly_uop_minmax(ctx, not_not_lt, &lo_b, &hi_b);
  ASSERT_INT_EQ(lo_a, lo_b);
  ASSERT_INT_EQ(hi_a, hi_b);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, weak_index_combines_same_base_terms_with_pinned_dtype_topology) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* Pinned tinygrad/uop/symbolic.py:242,247 canonicalizes both x+x and
   * x*c0+x*c1 while retaining the weak-index result dtype. */
  PolyUOp *n = Variable(ctx, "n", 1, 8, POLY_WEAKINT);
  PolyUOp *two = poly_const_int(ctx, 2);
  PolyUOp *three = poly_const_int(ctx, 3);
  PolyUOp *five = poly_const_int(ctx, 5);
  ASSERT_NOT_NULL(n);
  ASSERT_NOT_NULL(two);
  ASSERT_NOT_NULL(three);
  ASSERT_NOT_NULL(five);

  PolyUOp *add_self = poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, n, n, poly_arg_none());
  PolyUOp *expected_two = poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, n, two, poly_arg_none());
  PolyUOp *rewritten_self = poly_graph_rewrite(ctx, add_self, poly_symbolic());
  ASSERT_PTR_EQ(rewritten_self, expected_two);
  ASSERT_TRUE(poly_dtype_eq(rewritten_self->dtype, POLY_WEAKINT));

  PolyUOp *term_two = poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, n, two, poly_arg_none());
  PolyUOp *term_three = poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, n, three, poly_arg_none());
  PolyUOp *sum = poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, term_two, term_three, poly_arg_none());
  PolyUOp *expected_five = poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, n, five, poly_arg_none());
  PolyUOp *rewritten_sum = poly_graph_rewrite(ctx, sum, poly_symbolic());
  ASSERT_PTR_EQ(rewritten_sum, expected_five);
  ASSERT_EQ(rewritten_sum->op, POLY_OP_MUL);
  ASSERT_INT_EQ(rewritten_sum->n_src, 2);
  ASSERT_PTR_EQ(rewritten_sum->src[0], n);
  ASSERT_PTR_EQ(rewritten_sum->src[1], five);
  ASSERT_TRUE(poly_dtype_eq(rewritten_sum->dtype, POLY_WEAKINT));

  PolyUOp *y = Variable(ctx, "y", 1, 8, POLY_WEAKINT);
  PolyUOp *assoc_source = poly_uop2(
      ctx, POLY_OP_ADD, POLY_WEAKINT,
      poly_uop2(ctx, POLY_OP_ADD, POLY_WEAKINT, y, n, poly_arg_none()), n, poly_arg_none()
  );
  PolyUOp *assoc_expected = poly_uop2(
      ctx, POLY_OP_ADD, POLY_WEAKINT, y,
      poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, n, two, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *assoc_rewritten = poly_graph_rewrite(ctx, assoc_source, poly_symbolic());
  ASSERT_PTR_EQ(assoc_rewritten, assoc_expected);
  ASSERT_INT_EQ(assoc_rewritten->op, POLY_OP_ADD);
  ASSERT_INT_EQ(assoc_rewritten->src[1]->op, POLY_OP_MUL);
  ASSERT_TRUE(poly_dtype_eq(assoc_rewritten->src[1]->dtype, POLY_WEAKINT));

  for (int64_t value = 1; value <= 8; value = value == 1 ? 4 : 8) {
    PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(value));
    PolyUOp *bound_y = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(value + 1));
    PolyUOp *from[] = {n, y};
    PolyUOp *to[] = {bound, bound_y};
    PolyUOp *original_value =
        poly_graph_rewrite(ctx, poly_uop_substitute(ctx, sum, from, to, 1), poly_symbolic());
    PolyUOp *rewritten_value = poly_graph_rewrite(
        ctx, poly_uop_substitute(ctx, rewritten_sum, from, to, 1), poly_symbolic()
    );
    int64_t original_i = 0, rewritten_i = 0;
    ASSERT_INT_EQ(poly_uop_const_i64(original_value, &original_i), 0);
    ASSERT_INT_EQ(poly_uop_const_i64(rewritten_value, &rewritten_i), 0);
    ASSERT_INT_EQ(original_i, value * 5);
    ASSERT_INT_EQ(rewritten_i, original_i);

    PolyUOp *assoc_original_value = poly_graph_rewrite(
        ctx, poly_uop_substitute(ctx, assoc_source, from, to, 2), poly_symbolic()
    );
    PolyUOp *assoc_rewritten_value = poly_graph_rewrite(
        ctx, poly_uop_substitute(ctx, assoc_rewritten, from, to, 2), poly_symbolic()
    );
    int64_t assoc_original_i = 0, assoc_rewritten_i = 0;
    ASSERT_INT_EQ(poly_uop_const_i64(assoc_original_value, &assoc_original_i), 0);
    ASSERT_INT_EQ(poly_uop_const_i64(assoc_rewritten_value, &assoc_rewritten_i), 0);
    ASSERT_INT_EQ(assoc_original_i, 3 * value + 1);
    ASSERT_INT_EQ(assoc_rewritten_i, assoc_original_i);
    if (value == 8) break;
  }

  PolyUOp *max_coeff = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(INT64_MAX));
  PolyUOp *index_one = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
  PolyUOp *overflow_sum = poly_uop2(
      ctx, POLY_OP_ADD, POLY_WEAKINT,
      poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, n, max_coeff, poly_arg_none()),
      poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, n, index_one, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *overflow_rewritten = poly_graph_rewrite(ctx, overflow_sum, poly_symbolic());
  ASSERT_NOT_NULL(overflow_rewritten);
  ASSERT_INT_EQ(overflow_rewritten->op, POLY_OP_MUL);
  ASSERT_PTR_EQ(overflow_rewritten->src[0], n);
  ASSERT_TRUE(integer_const_eq(overflow_rewritten->src[1], "9223372036854775808"));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, symbolic_simple_defers_add_self_combination_to_full_symbolic) {
  /* Pinned symbolic.py:93-228 versus 237-248: x+x combination is a full
   * symbolic rule and must not run inside the simple matcher used by late
   * renderer decomposition. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *x = poly_uop0(ctx, POLY_OP_PARAM, POLY_UINT32, poly_arg_int(0));
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT32, poly_arg_int(1));
  PolyUOp *pair = poly_uop2(ctx, POLY_OP_ADD, POLY_UINT32, x, x, poly_arg_none());
  PolyUOp *nested = poly_uop2(
      ctx, POLY_OP_ADD, POLY_UINT32,
      poly_uop2(ctx, POLY_OP_ADD, POLY_UINT32, x, one, poly_arg_none()), one, poly_arg_none()
  );

  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, pair, poly_symbolic_simple()), pair);
  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, nested, poly_symbolic_simple()), nested);

  PolyUOp *pair_full = poly_graph_rewrite(ctx, pair, poly_symbolic());
  ASSERT_EQ(pair_full->op, POLY_OP_MUL);
  ASSERT_PTR_EQ(pair_full->src[0], x);
  ASSERT_TRUE(pair_full->src[1]->op == POLY_OP_CONST && pair_full->src[1]->arg.i == 2);

  PolyUOp *nested_full = poly_graph_rewrite(ctx, nested, poly_symbolic());
  ASSERT_EQ(nested_full->op, POLY_OP_ADD);
  ASSERT_PTR_EQ(nested_full->src[0], x);
  ASSERT_TRUE(nested_full->src[1]->op == POLY_OP_CONST && nested_full->src[1]->arg.i == 2);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, lossless_nested_cast_roundtrip_matches_tinygrad) {
  /* Pinned symbolic.py:151-152 removes b.cast(a).cast(b) only when a
   * preserves every value representable by b. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *x = Variable(ctx, "cast_x", INT32_MIN, INT32_MAX, POLY_INT32);

  PolyUOp *weak_roundtrip = poly_uop1(
      ctx, POLY_OP_CAST, POLY_INT32, poly_uop1(ctx, POLY_OP_CAST, POLY_WEAKINT, x, poly_arg_none()),
      poly_arg_none()
  );
  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, weak_roundtrip, poly_symbolic()), x);

  PolyUOp *x64 = poly_uop1(ctx, POLY_OP_CAST, POLY_INT64, x, poly_arg_none());
  PolyUOp *wide_weak_roundtrip = poly_uop1(
      ctx, POLY_OP_CAST, POLY_INT64,
      poly_uop1(ctx, POLY_OP_CAST, POLY_WEAKINT, x64, poly_arg_none()), poly_arg_none()
  );
  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, wide_weak_roundtrip, poly_symbolic()), x64);

  PolyUOp *narrow = poly_uop1(ctx, POLY_OP_CAST, POLY_INT16, x, poly_arg_none());
  PolyUOp *lossy_roundtrip = poly_uop1(ctx, POLY_OP_CAST, POLY_INT32, narrow, poly_arg_none());
  PolyUOp *lossy_lowered = poly_graph_rewrite(ctx, lossy_roundtrip, poly_symbolic());
  ASSERT_INT_EQ(lossy_lowered->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(lossy_lowered->dtype, POLY_INT32));
  ASSERT_INT_EQ(lossy_lowered->src[0]->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(lossy_lowered->src[0]->dtype, POLY_INT16));

  PolyDType int2 = POLY_INT32;
  PolyDType weak2 = POLY_WEAKINT;
  PolyUOp *vector_lanes[2] = {x, x};
  PolyUOp *vector_x = poly_uop(ctx, POLY_OP_STACK, int2, vector_lanes, 2, poly_arg_none());
  PolyUOp *vector_roundtrip = poly_uop1(
      ctx, POLY_OP_CAST, int2, poly_uop1(ctx, POLY_OP_CAST, weak2, vector_x, poly_arg_none()),
      poly_arg_none()
  );
  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, vector_roundtrip, poly_symbolic()), vector_x);

  PolyUOp *bool_x = Variable(ctx, "cast_bool", 0, 1, POLY_BOOL);
  PolyUOp *bool_vector_roundtrip = poly_uop1(
      ctx, POLY_OP_CAST, POLY_BOOL, poly_uop1(ctx, POLY_OP_CAST, int2, bool_x, poly_arg_none()),
      poly_arg_none()
  );
  ASSERT_PTR_EQ(poly_graph_rewrite(ctx, bool_vector_roundtrip, poly_symbolic()), bool_x);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, proven_long_binary_narrows_only_with_exact_i32_bounds) {
  /* Pinned tinygrad uop/symbolic.py:297-299 computes signed-long Binary
   * operations in int32 only when the result and both inputs provably fit,
   * then casts the result back to long. This is shared symbolic behavior, not
   * a WebGPU or gather legalization. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *bounded = Variable(ctx, "bounded_long", 0, 1, POLY_INT64);
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_INT64, poly_arg_int(2));
  PolyUOp *sum = poly_uop2(ctx, POLY_OP_ADD, POLY_INT64, bounded, two, poly_arg_none());
  PolyUOp *lowered = poly_graph_rewrite(ctx, sum, poly_symbolic());
  ASSERT_NOT_NULL(lowered);
  ASSERT_INT_EQ(lowered->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(lowered->dtype, POLY_INT64));
  ASSERT_INT_EQ(lowered->n_src, 1);
  ASSERT_INT_EQ(lowered->src[0]->op, POLY_OP_ADD);
  ASSERT_TRUE(poly_dtype_eq(lowered->src[0]->dtype, POLY_INT32));
  ASSERT_INT_EQ(lowered->src[0]->n_src, 2);
  ASSERT_INT_EQ(lowered->src[0]->src[0]->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(lowered->src[0]->src[0]->dtype, POLY_INT32));
  ASSERT_PTR_EQ(lowered->src[0]->src[0]->src[0], bounded);
  ASSERT_INT_EQ(lowered->src[0]->src[1]->op, POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(lowered->src[0]->src[1]->dtype, POLY_INT32));
  ASSERT_INT_EQ(lowered->src[0]->src[1]->arg.i, 2);

  PolyUOp *wide = Variable(ctx, "wide_long", INT32_MAX - 1, INT32_MAX, POLY_INT64);
  PolyUOp *wide_sum = poly_uop2(ctx, POLY_OP_ADD, POLY_INT64, wide, two, poly_arg_none());
  PolyUOp *wide_lowered = poly_graph_rewrite(ctx, wide_sum, poly_symbolic());
  ASSERT_PTR_EQ(wide_lowered, wide_sum);
  ASSERT_TRUE(poly_dtype_eq(wide_lowered->dtype, POLY_INT64));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, self_comparisons_keep_boolean_result_dtype) {
  /* Pinned symbolic.py:107-108,120-121 casts the operand-like false constant
   * to scalar/vector bool. The result must not retain the integer operand
   * dtype after constant folding. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *scalar = Variable(ctx, "cmp_scalar", 0, 7, POLY_UINT16);
  PolyUOp *vector_lane = Variable(ctx, "cmp_vector", 0, 7, POLY_UINT16);
  PolyUOp *vector_src[] = {vector_lane, vector_lane};
  PolyUOp *vector = poly_uop_stack(ctx, vector_src, 2);
  const PolyOps ops[] = {POLY_OP_CMPLT, POLY_OP_CMPNE};
  for (int i = 0; i < 2; i++) {
    PolyUOp *scalar_cmp = poly_uop2(ctx, ops[i], POLY_BOOL, scalar, scalar, poly_arg_none());
    PolyUOp *scalar_out = poly_graph_rewrite(ctx, scalar_cmp, poly_symbolic());
    ASSERT_NOT_NULL(scalar_out);
    ASSERT_INT_EQ(scalar_out->op, POLY_OP_CONST);
    ASSERT_TRUE(poly_dtype_eq(scalar_out->dtype, POLY_BOOL));
    ASSERT_INT_EQ(scalar_out->arg.kind, POLY_ARG_BOOL);
    ASSERT_TRUE(!scalar_out->arg.b);

    PolyUOp *vector_cmp = poly_uop2(ctx, ops[i], POLY_BOOL, vector, vector, poly_arg_none());
    PolyUOp *vector_out = poly_graph_rewrite(ctx, vector_cmp, poly_symbolic());
    ASSERT_NOT_NULL(vector_out);
    ASSERT_INT_EQ(vector_out->op, POLY_OP_EXPAND);
    ASSERT_TRUE(poly_dtype_eq(vector_out->dtype, POLY_BOOL));
    ASSERT_INT_EQ(poly_uop_max_numel(ctx, vector_out), 2);
    ASSERT_INT_EQ(vector_out->src[0]->op, POLY_OP_CONST);
    ASSERT_INT_EQ(vector_out->src[0]->arg.kind, POLY_ARG_BOOL);
    ASSERT_TRUE(!vector_out->src[0]->arg.b);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(sym, general_nested_cast_composition_matches_tinygrad) {
  /* Pinned symbolic.py:297-301 composes through an intermediate CAST when
   * either can_lossless_cast proves it globally or integer bounds prove this
   * value is not narrowed. Unsafe narrowing must retain both CASTs. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *wide = Variable(ctx, "compose_wide", -(INT64_C(1) << 30), INT64_C(1) << 30, POLY_INT32);
  wide = poly_uop2(
      ctx, POLY_OP_ADD, POLY_INT32, wide,
      poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(1)), poly_arg_none()
  );
  PolyUOp *weak_to_uint = poly_uop1(
      ctx, POLY_OP_CAST, POLY_UINT32,
      poly_uop1(ctx, POLY_OP_CAST, POLY_WEAKINT, wide, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *weak_to_uint_lowered = poly_graph_rewrite(ctx, weak_to_uint, poly_symbolic());
  ASSERT_INT_EQ(weak_to_uint_lowered->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(weak_to_uint_lowered->dtype, POLY_UINT32));
  ASSERT_PTR_EQ(weak_to_uint_lowered->src[0], wide);

  PolyUOp *small = Variable(ctx, "compose_small", 0, 255, POLY_INT32);
  PolyUOp *safe_narrow = poly_uop1(
      ctx, POLY_OP_CAST, POLY_FLOAT32,
      poly_uop1(ctx, POLY_OP_CAST, POLY_UINT8, small, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *safe_narrow_lowered = poly_graph_rewrite(ctx, safe_narrow, poly_symbolic());
  ASSERT_INT_EQ(safe_narrow_lowered->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(safe_narrow_lowered->dtype, POLY_FLOAT32));
  ASSERT_PTR_EQ(safe_narrow_lowered->src[0], small);

  PolyUOp *unsafe = Variable(ctx, "compose_unsafe", -1, 255, POLY_INT32);
  PolyUOp *unsafe_narrow = poly_uop1(
      ctx, POLY_OP_CAST, POLY_FLOAT32,
      poly_uop1(ctx, POLY_OP_CAST, POLY_UINT8, unsafe, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *unsafe_narrow_lowered = poly_graph_rewrite(ctx, unsafe_narrow, poly_symbolic());
  ASSERT_INT_EQ(unsafe_narrow_lowered->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(unsafe_narrow_lowered->dtype, POLY_FLOAT32));
  ASSERT_INT_EQ(unsafe_narrow_lowered->src[0]->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(unsafe_narrow_lowered->src[0]->dtype, POLY_UINT8));
  ASSERT_PTR_EQ(unsafe_narrow_lowered->src[0]->src[0], unsafe);

  /* Pinned Python bounds retain narrowing intermediates for full uint64 and
   * weakint ranges. Polygrad's int64 cache endpoints must not act as proof. */
  PolyUOp *unbounded_u64 = poly_uop0(ctx, POLY_OP_NOOP, POLY_UINT64, poly_arg_none());
  PolyUOp *u64_narrow = poly_uop1(
      ctx, POLY_OP_CAST, POLY_FLOAT64,
      poly_uop1(ctx, POLY_OP_CAST, POLY_INT64, unbounded_u64, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *u64_narrow_lowered = poly_graph_rewrite(ctx, u64_narrow, poly_symbolic());
  ASSERT_INT_EQ(u64_narrow_lowered->src[0]->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(u64_narrow_lowered->src[0]->dtype, POLY_INT64));

  PolyUOp *unbounded_weak = poly_uop0(ctx, POLY_OP_NOOP, POLY_WEAKINT, poly_arg_none());
  PolyUOp *weak_narrow = poly_uop1(
      ctx, POLY_OP_CAST, POLY_FLOAT64,
      poly_uop1(ctx, POLY_OP_CAST, POLY_INT64, unbounded_weak, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *weak_narrow_lowered = poly_graph_rewrite(ctx, weak_narrow, poly_symbolic());
  ASSERT_INT_EQ(weak_narrow_lowered->src[0]->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(weak_narrow_lowered->src[0]->dtype, POLY_INT64));

  /* Pinned `_min_max` keeps arbitrary-precision endpoints through derived
   * uint64/weakint expressions. Sentinel arithmetic must not prove a
   * narrowing intermediate safe. */
  PolyUOp *weak_shr = poly_uop2(
      ctx, POLY_OP_SHR, POLY_WEAKINT, unbounded_weak,
      poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1)), poly_arg_none()
  );
  PolyUOp *weak_shr_narrow = poly_uop1(
      ctx, POLY_OP_CAST, POLY_FLOAT64,
      poly_uop1(ctx, POLY_OP_CAST, POLY_INT64, weak_shr, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *weak_shr_lowered = poly_graph_rewrite(ctx, weak_shr_narrow, poly_symbolic());
  ASSERT_INT_EQ(weak_shr_lowered->src[0]->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(weak_shr_lowered->src[0]->dtype, POLY_INT64));

  PolyUOp *u64_shr = poly_uop2(
      ctx, POLY_OP_SHR, POLY_UINT64, unbounded_u64,
      poly_uop0(ctx, POLY_OP_CONST, POLY_UINT64, poly_arg_int(32)), poly_arg_none()
  );
  PolyUOp *u64_shr_i32 = poly_uop1(
      ctx, POLY_OP_CAST, POLY_FLOAT64,
      poly_uop1(ctx, POLY_OP_CAST, POLY_INT32, u64_shr, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *u64_shr_i32_lowered = poly_graph_rewrite(ctx, u64_shr_i32, poly_symbolic());
  ASSERT_INT_EQ(u64_shr_i32_lowered->src[0]->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(u64_shr_i32_lowered->src[0]->dtype, POLY_INT32));

  /* The same exact [0, 2**32-1] range does fit int64, so pinned removes
   * that wider intermediate. */
  PolyUOp *u64_shr_i64 = poly_uop1(
      ctx, POLY_OP_CAST, POLY_FLOAT64,
      poly_uop1(ctx, POLY_OP_CAST, POLY_INT64, u64_shr, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *u64_shr_i64_lowered = poly_graph_rewrite(ctx, u64_shr_i64, poly_symbolic());
  ASSERT_INT_EQ(u64_shr_i64_lowered->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(u64_shr_i64_lowered->dtype, POLY_FLOAT64));
  ASSERT_PTR_EQ(u64_shr_i64_lowered->src[0], u64_shr);

  PolyUOp *u64_mask = poly_uop2(
      ctx, POLY_OP_AND, POLY_UINT64, unbounded_u64,
      poly_uop0(ctx, POLY_OP_CONST, POLY_UINT64, poly_arg_int(INT64_MAX)), poly_arg_none()
  );
  PolyUOp *u64_mask_i64 = poly_uop1(
      ctx, POLY_OP_CAST, POLY_FLOAT64,
      poly_uop1(ctx, POLY_OP_CAST, POLY_INT64, u64_mask, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *u64_mask_lowered = poly_graph_rewrite(ctx, u64_mask_i64, poly_symbolic());
  ASSERT_INT_EQ(u64_mask_lowered->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(u64_mask_lowered->dtype, POLY_FLOAT64));
  ASSERT_PTR_EQ(u64_mask_lowered->src[0], u64_mask);

  /* Explicit bounded uint64/weakint variables remain exact and do compose. */
  PolyDType bounded_types[2] = {POLY_UINT64, POLY_WEAKINT};
  for (int i = 0; i < 2; i++) {
    PolyUOp *bounded = Variable(
        ctx, i == 0 ? "compose_bounded_u64" : "compose_bounded_weak", 0, 255, bounded_types[i]
    );
    PolyUOp *bounded_cast = poly_uop1(
        ctx, POLY_OP_CAST, POLY_FLOAT64,
        poly_uop1(ctx, POLY_OP_CAST, POLY_INT64, bounded, poly_arg_none()), poly_arg_none()
    );
    PolyUOp *bounded_lowered = poly_graph_rewrite(ctx, bounded_cast, poly_symbolic());
    ASSERT_INT_EQ(bounded_lowered->op, POLY_OP_CAST);
    ASSERT_TRUE(poly_dtype_eq(bounded_lowered->dtype, POLY_FLOAT64));
    ASSERT_PTR_EQ(bounded_lowered->src[0], bounded);
  }

  PolyUOp *vector_lane = Variable(ctx, "compose_vector", 0, 255, POLY_INT32);
  PolyUOp *vector_src[] = {vector_lane, vector_lane};
  PolyUOp *vector = poly_uop_stack(ctx, vector_src, 2);
  PolyUOp *vector_cast = poly_uop1(
      ctx, POLY_OP_CAST, POLY_UINT32,
      poly_uop1(ctx, POLY_OP_CAST, POLY_WEAKINT, vector, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *vector_lowered = poly_graph_rewrite(ctx, vector_cast, poly_symbolic());
  ASSERT_INT_EQ(vector_lowered->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(vector_lowered->dtype, POLY_UINT32));
  ASSERT_INT_EQ(poly_uop_max_numel(ctx, vector_lowered), 2);
  ASSERT_PTR_EQ(vector_lowered->src[0], vector);

  /* Tinygrad 2026-08-22/a9069c177a9d symbolic.py:294-296 applies scalar
   * dtype bounds to shaped CASTs and removes the intermediate uint16 CAST. */
  PolyUOp *wide_lane = Variable(ctx, "compose_wide_vector", 0, 100000, POLY_INT32);
  PolyUOp *wide_src[] = {wide_lane, wide_lane};
  PolyUOp *wide_vector = poly_uop_stack(ctx, wide_src, 2);
  PolyUOp *short_vector = poly_uop1(ctx, POLY_OP_CAST, POLY_INT16, wide_vector, poly_arg_none());
  PolyUOp *vector_unsigned = poly_uop1(
      ctx, POLY_OP_CAST, POLY_FLOAT32,
      poly_uop1(ctx, POLY_OP_CAST, POLY_UINT16, short_vector, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *vector_unsigned_lowered = poly_graph_rewrite(ctx, vector_unsigned, poly_symbolic());
  ASSERT_TRUE(poly_dtype_eq(vector_unsigned_lowered->dtype, POLY_FLOAT32));
  ASSERT_INT_EQ(poly_uop_max_numel(ctx, vector_unsigned_lowered), 2);
  ASSERT_INT_EQ(vector_unsigned_lowered->src[0]->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(vector_unsigned_lowered->src[0]->dtype, POLY_INT16));
  ASSERT_PTR_EQ(vector_unsigned_lowered->src[0], short_vector);

  /* Pinned ops.py:980-999 assigns [0,0] directly only to empty floor
   * div/mod numerators. CDIV still uses its corner formula. */
  PolyUOp *empty_range = poly_uop1(
      ctx, POLY_OP_RANGE, POLY_INT32, poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(-4)),
      poly_arg_range(0, POLY_AXIS_LOOP)
  );
  PolyOps empty_ops[4] = {POLY_OP_CDIV, POLY_OP_CMOD, POLY_OP_FLOORDIV, POLY_OP_FLOORMOD};
  bool empty_composes[4] = {false, true, true, true};
  for (int i = 0; i < 4; i++) {
    PolyUOp *empty_div = poly_uop2(
        ctx, empty_ops[i], POLY_INT32, empty_range,
        poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(3)), poly_arg_none()
    );
    PolyUOp *empty_cast = poly_uop1(
        ctx, POLY_OP_CAST, POLY_FLOAT32,
        poly_uop1(ctx, POLY_OP_CAST, POLY_UINT8, empty_div, poly_arg_none()), poly_arg_none()
    );
    PolyUOp *empty_direct = poly_pm_rewrite(poly_symbolic(), ctx, empty_cast);
    if (!empty_composes[i]) {
      ASSERT_TRUE(empty_direct == NULL);
      continue;
    }
    ASSERT_NOT_NULL(empty_direct);
    ASSERT_INT_EQ(empty_direct->op, POLY_OP_CAST);
    ASSERT_TRUE(poly_dtype_eq(empty_direct->dtype, POLY_FLOAT32));
    ASSERT_PTR_EQ(empty_direct->src[0], empty_div);
  }

  poly_ctx_destroy(ctx);
  PASS();
}
