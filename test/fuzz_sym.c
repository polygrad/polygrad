/*
 * fuzz_sym.c -- libFuzzer harness for Polygrad symbolic simplification.
 *
 * Builds small integer UOp graphs from arbitrary bytes, rewrites with
 * poly_symbolic(), and samples both graphs over bounded RANGE/ALU PARAM
 * assignments. Mismatches, sanitizer findings, and leaks are failures.
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
  const char *var_names[4];
  int64_t var_values[4];
  int n_vars;
} SymEnv;

static uint8_t fuzz_byte(FuzzReader *r) {
  if (!r || r->size == 0) return 0;
  if (r->pos >= r->size) return (uint8_t)(r->pos++ * 131u);
  return r->data[r->pos++];
}

static int64_t fuzz_small_i64(FuzzReader *r, int64_t lo, int64_t hi) {
  return lo + (int64_t)(fuzz_byte(r) % (uint8_t)(hi - lo + 1));
}

static int64_t fuzz_nonzero_i64(FuzzReader *r) {
  int64_t v = fuzz_small_i64(r, -9, 9);
  return v == 0 ? 1 : v;
}

static PolyUOp *mk_const(PolyCtx *ctx, int64_t v) {
  return poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(v));
}

static PolyUOp *mk_dvar(PolyCtx *ctx, const char *name, int64_t lo, int64_t hi) {
  return poly_uop_variable(ctx, name, poly_arg_int(lo), poly_arg_int(hi), POLY_INT32, 1, true);
}

static PolyUOp *mk_range(PolyCtx *ctx, int64_t n, int64_t axis_id) {
  PolyUOp *bound = mk_const(ctx, n);
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

  int64_t a = 0, b = 0, c = 0;
  if (u->n_src > 0 && !sym_eval_i64(u->src[0], env, &a)) return false;
  if (u->n_src > 1 && !sym_eval_i64(u->src[1], env, &b)) return false;
  if (u->n_src > 2 && !sym_eval_i64(u->src[2], env, &c)) return false;

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

static PolyUOp *fuzz_expr(PolyCtx *ctx, FuzzReader *r, PolyUOp **vars, int depth) {
  if (depth <= 0) {
    switch (fuzz_byte(r) % 5) {
    case 0:
      return vars[0];
    case 1:
      return vars[1];
    case 2:
      return vars[2];
    case 3:
      return vars[3];
    default:
      return mk_const(ctx, fuzz_small_i64(r, -16, 16));
    }
  }

  PolyUOp *a = fuzz_expr(ctx, r, vars, depth - 1);
  uint8_t op = fuzz_byte(r) % 14;
  if (op <= 7) {
    PolyUOp *b = fuzz_expr(ctx, r, vars, depth - 1);
    switch (op) {
    case 0:
      return poly_uop2(ctx, POLY_OP_ADD, POLY_INT32, a, b, poly_arg_none());
    case 1:
      return poly_uop2(ctx, POLY_OP_SUB, POLY_INT32, a, b, poly_arg_none());
    case 2:
      return poly_uop2(ctx, POLY_OP_MUL, POLY_INT32, a, b, poly_arg_none());
    case 3:
      return poly_uop2(ctx, POLY_OP_MAX, POLY_INT32, a, b, poly_arg_none());
    case 4:
      return poly_uop2(ctx, POLY_OP_AND, POLY_INT32, a, b, poly_arg_none());
    case 5:
      return poly_uop2(ctx, POLY_OP_OR, POLY_INT32, a, b, poly_arg_none());
    case 6:
      return poly_uop2(ctx, POLY_OP_XOR, POLY_INT32, a, b, poly_arg_none());
    default: {
      PolyUOp *c = fuzz_expr(ctx, r, vars, depth - 1);
      PolyUOp *cond = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, a, b, poly_arg_none());
      return poly_uop3(ctx, POLY_OP_WHERE, POLY_INT32, cond, b, c, poly_arg_none());
    }
    }
  }

  int64_t d = fuzz_nonzero_i64(r);
  PolyUOp *den = mk_const(ctx, d);
  switch (op) {
  case 8:
    return poly_uop2(ctx, POLY_OP_IDIV, POLY_INT32, a, den, poly_arg_none());
  case 9:
    return poly_uop2(ctx, POLY_OP_MOD, POLY_INT32, a, den, poly_arg_none());
  case 10:
    return poly_uop2(
        ctx, POLY_OP_ADD, POLY_INT32, a, mk_const(ctx, fuzz_small_i64(r, -31, 31)), poly_arg_none()
    );
  case 11:
    return poly_uop2(
        ctx, POLY_OP_MUL, POLY_INT32, a, mk_const(ctx, fuzz_nonzero_i64(r)), poly_arg_none()
    );
  case 12: {
    int64_t k = fuzz_small_i64(r, 2, 6);
    PolyUOp *wide = mk_const(ctx, d * k);
    PolyUOp *inner = poly_uop2(ctx, POLY_OP_MOD, POLY_INT32, a, wide, poly_arg_none());
    return poly_uop2(ctx, POLY_OP_IDIV, POLY_INT32, inner, den, poly_arg_none());
  }
  default: {
    int64_t k = fuzz_small_i64(r, 2, 6);
    PolyUOp *wide = mk_const(ctx, d * k);
    PolyUOp *inner = poly_uop2(ctx, POLY_OP_MOD, POLY_INT32, a, wide, poly_arg_none());
    return poly_uop2(ctx, POLY_OP_MOD, POLY_INT32, inner, den, poly_arg_none());
  }
  }
}

static void check_sample(PolyUOp *root, PolyUOp *rewritten, int r0, int r1, int a, int b) {
  SymEnv env = {0};
  env.ranges[0] = r0;
  env.ranges[1] = r1;
  env.n_ranges = 2;
  env.var_names[0] = "a";
  env.var_values[0] = a;
  env.var_names[1] = "b";
  env.var_values[1] = b;
  env.n_vars = 2;

  int64_t want = 0, got = 0;
  if (!sym_eval_i64(root, &env, &want) || !sym_eval_i64(rewritten, &env, &got)) return;
  if (got == want) return;

  char *root_s = poly_uop_str(root);
  char *rewritten_s = poly_uop_str(rewritten);
  fprintf(
      stderr, "symbolic fuzz mismatch r0=%d r1=%d a=%d b=%d got=%lld want=%lld\n", r0, r1, a, b,
      (long long)got, (long long)want
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
  PolyUOp *vars[4] = {
      mk_range(ctx, 9, 0),
      mk_range(ctx, 7, 1),
      mk_dvar(ctx, "a", 4, 11),
      mk_dvar(ctx, "b", -3, 5),
  };

  int depth = 1 + (int)(fuzz_byte(&r) % 5);
  PolyUOp *root = fuzz_expr(ctx, &r, vars, depth);
  PolyUOp *rewritten = poly_graph_rewrite(ctx, root, poly_symbolic());
  if (rewritten) {
    for (int r0 = 0; r0 < 9; r0 += 2) {
      for (int r1 = 0; r1 < 7; r1 += 3) {
        for (int a = 4; a <= 11; a += 3) {
          for (int b = -3; b <= 5; b += 4)
            check_sample(root, rewritten, r0, r1, a, b);
        }
      }
    }
  }

  poly_ctx_destroy(ctx);
  return 0;
}
