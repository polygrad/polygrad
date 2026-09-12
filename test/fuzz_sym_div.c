/*
 * fuzz_sym_div.c -- div/mod-focused symbolic simplification fuzzer.
 *
 * Mirrors tinygrad's test/external/fuzz_symbolic_symbolic_div.py shape:
 * build sums of signed range terms, divide/mod by positive symbolic factors,
 * simplify, then check sampled equivalence over valid variable assignments.
 */

#include "polygrad.h"
#include "uop/upat.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  const uint8_t *data;
  size_t size;
  size_t pos;
} FuzzReader;

typedef struct {
  int64_t ranges[4];
  int n_ranges;
  const char *var_names[3];
  int64_t var_values[3];
  int n_vars;
} SymEnv;

static uint8_t fuzz_byte(FuzzReader *r) {
  if (!r || r->size == 0) return 0;
  if (r->pos >= r->size) return (uint8_t)(r->pos++ * 131u);
  return r->data[r->pos++];
}

static int64_t fuzz_i64(FuzzReader *r, int64_t lo, int64_t hi) {
  return lo + (int64_t)(fuzz_byte(r) % (uint8_t)(hi - lo + 1));
}

static PolyUOp *mk_const(PolyCtx *ctx, int64_t v) {
  return poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(v));
}

static PolyUOp *mk_dvar(PolyCtx *ctx, const char *name, int64_t lo, int64_t hi) {
  return poly_uop_variable(ctx, name, poly_arg_int(lo), poly_arg_int(hi), POLY_INT32, 1, true);
}

static PolyUOp *mk_range(PolyCtx *ctx, PolyUOp *bound, int64_t axis_id) {
  return poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_range(axis_id, POLY_AXIS_LOOP));
}

static bool safe_add_i64(int64_t a, int64_t b, int64_t *out) {
  return !__builtin_add_overflow(a, b, out);
}

static bool safe_sub_i64(int64_t a, int64_t b, int64_t *out) {
  return !__builtin_sub_overflow(a, b, out);
}

static bool safe_mul_i64(int64_t a, int64_t b, int64_t *out) {
  return !__builtin_mul_overflow(a, b, out);
}

static bool sym_eval_i64(PolyUOp *u, const SymEnv *env, int64_t *out) {
  if (!u || !env || !out) return false;
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

  int64_t a = 0, b = 0;
  if (u->n_src > 0 && !sym_eval_i64(u->src[0], env, &a)) return false;
  if (u->n_src > 1 && !sym_eval_i64(u->src[1], env, &b)) return false;

  switch (u->op) {
  case POLY_OP_NEG:
    return safe_sub_i64(0, a, out);
  case POLY_OP_ADD:
    return safe_add_i64(a, b, out);
  case POLY_OP_SUB:
    return safe_sub_i64(a, b, out);
  case POLY_OP_MUL:
    return safe_mul_i64(a, b, out);
  case POLY_OP_IDIV:
    if (b == 0 || (a == INT64_MIN && b == -1)) return false;
    *out = a / b;
    return true;
  case POLY_OP_MOD:
    if (b == 0 || (a == INT64_MIN && b == -1)) return false;
    *out = a % b;
    return true;
  default:
    return false;
  }
}

static PolyUOp *random_factor(PolyCtx *ctx, FuzzReader *r, PolyUOp **factors, int n_factors) {
  PolyUOp *base = factors[fuzz_byte(r) % (uint8_t)n_factors];
  switch (fuzz_byte(r) % 4) {
  case 0:
    return base;
  case 1:
    return poly_uop2(
        ctx, POLY_OP_MUL, POLY_INT32, base, mk_const(ctx, fuzz_i64(r, 2, 7)), poly_arg_none()
    );
  case 2:
    return poly_uop2(
        ctx, POLY_OP_ADD, POLY_INT32, base, factors[fuzz_byte(r) % (uint8_t)n_factors],
        poly_arg_none()
    );
  default:
    return mk_const(ctx, fuzz_i64(r, 1, 17));
  }
}

static PolyUOp *random_term(
    PolyCtx *ctx,
    FuzzReader *r,
    PolyUOp **ranges,
    int n_ranges,
    PolyUOp **factors,
    int n_factors
) {
  PolyUOp *base = ranges[fuzz_byte(r) % (uint8_t)n_ranges];
  PolyUOp *factor = random_factor(ctx, r, factors, n_factors);
  PolyUOp *term = poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, base, factor, poly_arg_none());
  if (fuzz_byte(r) & 1) term = poly_uop1(ctx, POLY_OP_NEG, POLY_INT32, term, poly_arg_none());
  return term;
}

static PolyUOp *random_sum(
    PolyCtx *ctx,
    FuzzReader *r,
    PolyUOp **ranges,
    int n_ranges,
    PolyUOp **factors,
    int n_factors
) {
  int n_terms = 2 + (int)(fuzz_byte(r) % 4);
  PolyUOp *acc = random_term(ctx, r, ranges, n_ranges, factors, n_factors);
  for (int i = 1; i < n_terms; i++) {
    PolyUOp *term = random_term(ctx, r, ranges, n_ranges, factors, n_factors);
    acc = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, acc, term, poly_arg_none());
  }
  return acc;
}

static PolyUOp *random_div_expr(
    PolyCtx *ctx,
    FuzzReader *r,
    PolyUOp **ranges,
    int n_ranges,
    PolyUOp **factors,
    int n_factors
) {
  PolyUOp *num = random_sum(ctx, r, ranges, n_ranges, factors, n_factors);
  if ((fuzz_byte(r) % 10) == 0) {
    PolyUOp *nested_den = random_factor(ctx, r, factors, n_factors);
    num = poly_uop2(
        ctx, fuzz_byte(r) & 1 ? POLY_OP_IDIV : POLY_OP_MOD, POLY_INT32, num, nested_den,
        poly_arg_none()
    );
  }
  PolyUOp *den = random_factor(ctx, r, factors, n_factors);
  if (fuzz_byte(r) & 1) den = poly_uop1(ctx, POLY_OP_NEG, POLY_INT32, den, poly_arg_none());
  return poly_uop2(
      ctx, fuzz_byte(r) & 1 ? POLY_OP_IDIV : POLY_OP_MOD, POLY_INT32, num, den, poly_arg_none()
  );
}

static void check_sample(
    PolyUOp *root,
    PolyUOp *rewritten,
    int r0,
    int r1,
    int r2,
    int r3,
    int i,
    int j,
    int k
) {
  SymEnv env = {0};
  env.ranges[0] = r0;
  env.ranges[1] = r1;
  env.ranges[2] = r2;
  env.ranges[3] = r3;
  env.n_ranges = 4;
  env.var_names[0] = "i";
  env.var_values[0] = i;
  env.var_names[1] = "j";
  env.var_values[1] = j;
  env.var_names[2] = "k";
  env.var_values[2] = k;
  env.n_vars = 3;

  int64_t want = 0, got = 0;
  if (!sym_eval_i64(root, &env, &want) || !sym_eval_i64(rewritten, &env, &got)) return;
  if (got == want) return;

  char *root_s = poly_uop_str(root);
  char *rewritten_s = poly_uop_str(rewritten);
  fprintf(
      stderr, "symbolic-div fuzz mismatch r=[%d,%d,%d,%d] vars=[%d,%d,%d] got=%lld want=%lld\n", r0,
      r1, r2, r3, i, j, k, (long long)got, (long long)want
  );
  fprintf(
      stderr, "root=%s\nrewritten=%s\n", root_s ? root_s : "<oom>",
      rewritten_s ? rewritten_s : "<oom>"
  );
  fprintf(stderr, "root tree:\n");
  poly_uop_dump_tree(stderr, root, 0, 10);
  fprintf(stderr, "rewritten tree:\n");
  poly_uop_dump_tree(stderr, rewritten, 0, 10);
  free(root_s);
  free(rewritten_s);
  abort();
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  if (size > 4096) return 0;

  PolyCtx *ctx = poly_ctx_new();
  if (!ctx) return 0;

  FuzzReader r = {.data = data, .size = size, .pos = 0};
  PolyUOp *i = mk_dvar(ctx, "i", 1, 9);
  PolyUOp *j = mk_dvar(ctx, "j", 1, 7);
  PolyUOp *k = mk_dvar(ctx, "k", 1, 5);

  PolyUOp *factors[10];
  int nf = 0;
  factors[nf++] = i;
  factors[nf++] = j;
  factors[nf++] = k;
  factors[nf++] = mk_const(ctx, 2 + (fuzz_byte(&r) % 7));
  factors[nf++] = mk_const(ctx, 16);
  factors[nf++] = mk_const(ctx, 33);
  factors[nf++] = poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, i, j, poly_arg_none());
  factors[nf++] = poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, j, k, poly_arg_none());
  factors[nf++] = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, i, factors[3], poly_arg_none());
  factors[nf++] = poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, j, k, poly_arg_none());

  PolyUOp *ranges[4] = {
      mk_range(ctx, mk_const(ctx, 4), 0),
      mk_range(ctx, mk_const(ctx, 5), 1),
      mk_range(ctx, mk_const(ctx, 7), 2),
      mk_range(ctx, mk_const(ctx, 9), 3),
  };

  PolyUOp *root = random_div_expr(ctx, &r, ranges, 4, factors, nf);
  PolyUOp *rewritten = poly_graph_rewrite(ctx, root, poly_symbolic());
  if (rewritten) {
    for (int vi = 1; vi <= 9; vi += 4) {
      for (int vj = 1; vj <= 7; vj += 3) {
        for (int vk = 1; vk <= 5; vk += 2) {
          check_sample(root, rewritten, 0, 0, 0, 0, vi, vj, vk);
          check_sample(root, rewritten, 1, 2, 3, 4, vi, vj, vk);
          check_sample(root, rewritten, 3, 4, 6, 8, vi, vj, vk);
        }
      }
    }
  }

  poly_ctx_destroy(ctx);
  return 0;
}
