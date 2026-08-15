/*
 * test_codegen.c — Tests for linearizer, C renderer, and CPU runtime
 */

#include "test_harness.h"
#include "../src/codegen.h"
#include "../src/bigint.h"
#include "../src/frontend.h"
#include "../src/frontend_internal.h"
#include "../src/interp.h"
#include "../src/tensor.h" /* poly_sum_reduce */

#include <inttypes.h>
#include <time.h>

static uint64_t topology_fnv_bytes(uint64_t h, const void *data, size_t n) {
  const uint8_t *bytes = (const uint8_t *)data;
  for (size_t i = 0; i < n; i++) {
    h ^= bytes[i];
    h *= UINT64_C(1099511628211);
  }
  return h;
}

static uint64_t topology_fnv_u32(uint64_t h, uint32_t value) {
  for (int i = 0; i < 4; i++) {
    uint8_t byte = (uint8_t)(value >> (i * 8));
    h = topology_fnv_bytes(h, &byte, 1);
  }
  return h;
}

static uint64_t topology_fnv_u64(uint64_t h, uint64_t value) {
  for (int i = 0; i < 8; i++) {
    uint8_t byte = (uint8_t)(value >> (i * 8));
    h = topology_fnv_bytes(h, &byte, 1);
  }
  return h;
}

/* Cross-language structural fingerprint used by the pinned direct-UOp
 * transcendental probes. PARAM shape/metadata is intentionally excluded:
 * PG-PARITY-002 tracks that independent vocabulary migration. */
static uint64_t normalized_topology_fingerprint(
    PolyUOp **topo,
    int n_topo,
    PolyUOp *root
) {
  uint64_t *hashes = calloc((size_t)n_topo, sizeof(*hashes));
  if (!hashes) return 0;
  uint64_t root_hash = 0;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    PolyDType scalar = poly_dtype_scalar(u->dtype);
    uint8_t category =
        poly_dtype_eq(scalar, POLY_BOOL) ? 1
        : poly_dtype_is_unsigned(scalar) ? 3
        : poly_dtype_is_int(scalar)      ? 2
        : poly_dtype_is_float(scalar)    ? 4
        : poly_dtype_eq(scalar, POLY_VOID) ? 5
                                          : 0;
    uint64_t h = UINT64_C(1469598103934665603);
    const char *op_name = poly_op_name(u->op);
    h = topology_fnv_bytes(h, op_name, strlen(op_name) + 1);
    h = topology_fnv_bytes(h, &category, 1);
    uint8_t bits = (uint8_t)scalar.bitsize;
    h = topology_fnv_bytes(h, &bits, 1);
    h = topology_fnv_u32(h, (uint32_t)u->dtype.count);

    uint8_t arg_tag = 0;
    uint64_t arg_value = 0;
    if (u->op == POLY_OP_CONST) {
      if (category == 4 && u->arg.kind == POLY_ARG_FLOAT) {
        arg_tag = 2;
        if (isnan(u->arg.f)) {
          arg_value = scalar.bitsize == 64 ? UINT64_C(0x7ff8000000000000)
                                           : UINT64_C(0x7fc00000);
        } else if (scalar.bitsize == 64) {
          memcpy(&arg_value, &u->arg.f, sizeof(arg_value));
        } else {
          float value = (float)u->arg.f;
          uint32_t value_bits = 0;
          memcpy(&value_bits, &value, sizeof(value_bits));
          arg_value = value_bits;
        }
      } else if (category == 1 && u->arg.kind == POLY_ARG_BOOL) {
        arg_tag = 3;
        arg_value = u->arg.b ? 1 : 0;
      } else if (u->arg.kind == POLY_ARG_INT || u->arg.kind == POLY_ARG_BIGINT) {
        arg_tag = 1;
        arg_value = poly_arg_integer_to_u64_mod(u->arg);
        if (scalar.bitsize > 0 && scalar.bitsize < 64)
          arg_value &= (UINT64_C(1) << scalar.bitsize) - 1;
      }
    }
    h = topology_fnv_bytes(h, &arg_tag, 1);
    h = topology_fnv_u64(h, arg_value);

    int n_src = u->op == POLY_OP_PARAM ? 0 : u->n_src;
    h = topology_fnv_u32(h, (uint32_t)n_src);
    for (int s = 0; s < n_src; s++) {
      uint64_t child_hash = 0;
      for (int j = 0; j < i; j++) {
        if (topo[j] == u->src[s]) {
          child_hash = hashes[j];
          break;
        }
      }
      h = topology_fnv_u64(h, child_hash);
    }
    hashes[i] = h;
    if (u == root) root_hash = h;
  }
  free(hashes);
  return root_hash;
}

/* Helper: build c[i] = a[i] OP b[i] kernel IR */

typedef struct {
  PolyCtx *ctx;
  PolyUOp *sink;
  int n; /* loop bound */
} VecKernel;

static VecKernel make_vec_binop(PolyOps alu_op, int n) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);

  /* PARAM: buffer pointers */
  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(1));
  PolyUOp *p2 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(2));

  /* loop: for ridx0 in range(n) */
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(n));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));

  /* index into buffers */
  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p0, range, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p1, range, poly_arg_none());
  PolyUOp *idx2 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p2, range, poly_arg_none());

  /* load */
  PolyUOp *load0 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx0, poly_arg_none());
  PolyUOp *load1 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx1, poly_arg_none());

  /* ALU */
  PolyUOp *alu = poly_uop2(ctx, alu_op, POLY_FLOAT32, load0, load1, poly_arg_none());

  /* store */
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx2, alu, poly_arg_none());

  /* end loop + sink */
  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  return (VecKernel){ctx, sink, n};
}

static int count_lin_ops(PolyUOp **lin, int n, PolyOps op) {
  int c = 0;
  for (int i = 0; i < n; i++)
    if (lin[i]->op == op) c++;
  return c;
}

static int count_reg_storage_ops(PolyUOp **lin, int n) {
  int c = 0;
  for (int i = 0; i < n; i++) {
    PolyUOp *u = lin[i];
    if (u->op == POLY_OP_DEFINE_REG ||
        (u->op == POLY_OP_BUFFER && u->dtype.is_ptr && u->dtype.addrspace == POLY_ADDR_REG))
      c++;
  }
  return c;
}

static int count_special_named(PolyUOp **lin, int n, const char *name) {
  int c = 0;
  for (int i = 0; i < n; i++) {
    if (lin[i]->op == POLY_OP_SPECIAL && lin[i]->arg.str && strcmp(lin[i]->arg.str, name) == 0) c++;
  }
  return c;
}

static int64_t special_bound_hi_named(PolyUOp **lin, int n, const char *name) {
  for (int i = 0; i < n; i++) {
    if (lin[i]->op != POLY_OP_SPECIAL || !lin[i]->arg.str || strcmp(lin[i]->arg.str, name) != 0 ||
        lin[i]->n_src <= 0)
      continue;
    int64_t lo = 0, hi = 0;
    poly_uop_minmax(NULL, lin[i]->src[0], &lo, &hi);
    return hi;
  }
  return -1;
}

static VecKernel make_vec_copy_with_weak_index_expr(int n) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);

  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(1));

  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(n));
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_INDEX, bound, poly_arg_range(0, POLY_AXIS_LOOP));

  /* tinygrad carries address expressions as weakint through most of codegen,
   * then pm_lower_index_dtype narrows them to a concrete integer dtype. This
   * test mirrors that path; explicit user casts to int64 are intentionally not
   * stripped by tinygrad and are not part of this invariant. */
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(0));
  PolyUOp *addr = poly_uop2(ctx, POLY_OP_ADD, POLY_INDEX, range, zero, poly_arg_none());

  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p0, addr, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p1, range, poly_arg_none());
  PolyUOp *load0 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx0, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx1, load0, poly_arg_none());

  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());
  return (VecKernel){ctx, sink, n};
}

static bool subtree_has_i64_dtype(PolyUOp *u) {
  if (!u) return false;
  PolyDType scalar = poly_dtype_scalar(u->dtype);
  if (poly_dtype_is_int(scalar) && scalar.bitsize == 64) return true;
  for (int i = 0; i < u->n_src; i++)
    if (subtree_has_i64_dtype(u->src[i])) return true;
  return false;
}

static int count_indexes_with_i64_addr(PolyUOp **nodes, int n) {
  int bad = 0;
  for (int i = 0; i < n; i++) {
    if (nodes[i]->op != POLY_OP_INDEX || nodes[i]->n_src < 2) continue;
    if (subtree_has_i64_dtype(nodes[i]->src[1])) bad++;
    if (nodes[i]->n_src >= 3 && subtree_has_i64_dtype(nodes[i]->src[2])) bad++;
  }
  return bad;
}

static int count_weakint_named_nodes(PolyUOp **nodes, int n) {
  int bad = 0;
  for (int i = 0; i < n; i++) {
    PolyDType s = poly_dtype_scalar(nodes[i]->dtype);
    if (s.name && strcmp(s.name, "weakint") == 0) bad++;
  }
  return bad;
}

static PolyUOp *make_shaped_f32_buf(PolyCtx *ctx, const int64_t *shape, int ndim) {
  int64_t numel = 1;
  for (int i = 0; i < ndim; i++)
    numel *= shape[i];
  PolyUOp *buf = poly_buffer_f32(ctx, numel);
  return ndim > 1 ? poly_reshape(ctx, buf, (int64_t *)shape, ndim) : buf;
}

static int wgsl_workgroup_product(const char *wgsl, int dims[3]) {
  dims[0] = dims[1] = dims[2] = 1;
  const char *p = strstr(wgsl, "@workgroup_size(");
  if (!p) return 1;
  p += strlen("@workgroup_size(");
  for (int i = 0; i < 3 && *p; i++) {
    dims[i] = atoi(p);
    const char *comma = strchr(p, ',');
    const char *close = strchr(p, ')');
    if (!comma || (close && close < comma)) break;
    p = comma + 1;
  }
  return dims[0] * dims[1] * dims[2];
}

/* Linearizer tests */

TEST(codegen, linearize_order) {
  VecKernel k = make_vec_binop(POLY_OP_ADD, 10);
  int n;
  PolyUOp **lin = poly_linearize(k.ctx, k.sink, &n);

  ASSERT_TRUE(n >= 6);
  ASSERT_INT_EQ(count_lin_ops(lin, n, POLY_OP_PARAM), 3);

  /* SINK should be last */
  ASSERT_TRUE(lin[n - 1]->op == POLY_OP_SINK);

  /* END should come just before SINK */
  ASSERT_TRUE(lin[n - 2]->op == POLY_OP_END);

  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(codegen, linearize_deps) {
  /* Verify: every UOp's sources appear before it in the linearized list */
  VecKernel k = make_vec_binop(POLY_OP_ADD, 10);
  int n;
  PolyUOp **lin = poly_linearize(k.ctx, k.sink, &n);

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < lin[i]->n_src; j++) {
      PolyUOp *src = lin[i]->src[j];
      bool found = false;
      for (int k = 0; k < i; k++) {
        if (lin[k] == src) {
          found = true;
          break;
        }
      }
      ASSERT_TRUE(found);
    }
  }

  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(codegen, split_ends_excludes_ranges_already_closed_by_nested_end) {
  /* Pinned tinygrad codegen/late/linearizer.py:88-90 queries
   * SINK(*end.src[1:]).ranges. An inner END removes its RANGE from that
   * active set, so the outer END disappears instead of closing it twice. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(8));
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_INDEX, bound, poly_arg_range(3, POLY_AXIS_LOOP));
  PolyUOp *body = poly_uop0(ctx, POLY_OP_NOOP, POLY_VOID, poly_arg_none());
  PolyUOp *outer_body = poly_uop0(ctx, POLY_OP_NOOP, POLY_VOID, poly_arg_str("outer"));
  PolyUOp *inner_srcs[2] = {body, range};
  PolyUOp *inner = poly_uop(ctx, POLY_OP_END, POLY_VOID, inner_srcs, 2, poly_arg_none());
  PolyUOp *outer_srcs[2] = {outer_body, inner};
  PolyUOp *outer = poly_uop(ctx, POLY_OP_END, POLY_VOID, outer_srcs, 2, poly_arg_none());

  PolyUOp *rewritten = poly_graph_rewrite(ctx, outer, poly_pm_split_ends_pass());
  ASSERT_TRUE(rewritten == outer_body);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, split_ends_retains_ranges_still_active_in_dependency) {
  /* The exclusion above is not a blanket END-subtree skip: an arithmetic
   * dependency on RANGE keeps it active and rebuilds exactly END(body, r). */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(8));
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_INDEX, bound, poly_arg_range(3, POLY_AXIS_LOOP));
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(1));
  PolyUOp *active = poly_uop2(ctx, POLY_OP_ADD, POLY_INDEX, range, one, poly_arg_none());
  PolyUOp *body = poly_uop0(ctx, POLY_OP_NOOP, POLY_VOID, poly_arg_str("outer"));
  PolyUOp *end_srcs[2] = {body, active};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_srcs, 2, poly_arg_none());

  PolyUOp *rewritten = poly_graph_rewrite(ctx, end, poly_pm_split_ends_pass());
  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(rewritten->op, POLY_OP_END);
  ASSERT_INT_EQ(rewritten->n_src, 2);
  ASSERT_TRUE(rewritten->src[0] == body);
  ASSERT_TRUE(rewritten->src[1] == range);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, reduce_merge_shared_end) {
  /* Two REDUCE ops over the same RANGE should share one merged END chain. */
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);

  PolyUOp *pin = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));
  PolyUOp *pout0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(1));
  PolyUOp *pout1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(2));

  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(8));
  PolyUOp *r0 = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));

  PolyUOp *in_idx = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, pin, r0, poly_arg_none());
  PolyUOp *in_ld = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, in_idx, poly_arg_none());

  PolyUOp *red0_srcs[2] = {in_ld, r0};
  PolyUOp *red1_srcs[2] = {in_ld, r0};
  PolyUOp *sum =
      poly_uop(ctx, POLY_OP_REDUCE, POLY_FLOAT32, red0_srcs, 2, poly_arg_ops(POLY_OP_ADD));
  PolyUOp *mx =
      poly_uop(ctx, POLY_OP_REDUCE, POLY_FLOAT32, red1_srcs, 2, poly_arg_ops(POLY_OP_MAX));

  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *out0_idx = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, pout0, zero, poly_arg_none());
  PolyUOp *out1_idx = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, pout1, zero, poly_arg_none());
  PolyUOp *st0 = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out0_idx, sum, poly_arg_none());
  PolyUOp *st1 = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out1_idx, mx, poly_arg_none());
  PolyUOp *stores[2] = {st0, st1};
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, stores, 2, poly_arg_none());

  int n = 0;
  PolyUOp **lin = poly_linearize(ctx, sink, &n);
  ASSERT_TRUE(n > 0);
  ASSERT_INT_EQ(count_reg_storage_ops(lin, n), 2);
  ASSERT_INT_EQ(count_lin_ops(lin, n, POLY_OP_END), 1);

  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

/* Renderer tests */

TEST(codegen, render_vecadd) {
  VecKernel k = make_vec_binop(POLY_OP_ADD, 10);
  const char *old_expand_ssa = getenv("EXPAND_SSA");
  char *saved_expand_ssa = old_expand_ssa ? strdup(old_expand_ssa) : NULL;
  unsetenv("EXPAND_SSA");

  int n;
  PolyUOp **lin = poly_linearize(k.ctx, k.sink, &n);
  char *src = poly_render_c(lin, n, "vecadd");

  /* Pinned cstyle.py:194,232-237 inlines a single-consumer ALU unless
   * EXPAND_SSA is enabled. */
  ASSERT_NOT_NULL(strstr(src, "void vecadd("));
  ASSERT_NOT_NULL(strstr(src, "float* restrict data0"));
  ASSERT_NOT_NULL(strstr(src, "float* restrict data1"));
  ASSERT_NOT_NULL(strstr(src, "float* restrict data2"));
  ASSERT_NOT_NULL(strstr(src, "for (int ridx0 = 0; ridx0 < 10; ridx0++)"));
  ASSERT_NOT_NULL(strstr(src, "float val0"));
  ASSERT_NOT_NULL(strstr(src, "float val1"));
  ASSERT_TRUE(strstr(src, "float alu0") == NULL);
  /* wrapper function */
  ASSERT_NOT_NULL(strstr(src, "void vecadd_call(void **args)"));

  free(src);

  setenv("EXPAND_SSA", "1", 1);
  src = poly_render_c(lin, n, "vecadd_expanded");
  ASSERT_NOT_NULL(src);
  ASSERT_NOT_NULL(strstr(src, "float alu0"));

  if (saved_expand_ssa)
    setenv("EXPAND_SSA", saved_expand_ssa, 1);
  else
    unsetenv("EXPAND_SSA");
  free(saved_expand_ssa);
  free(lin);
  free(src);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(codegen, render_half_single_consumer_chain_matches_pinned_c_expression) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  const char *old_expand_ssa = getenv("EXPAND_SSA");
  char *saved_expand_ssa = old_expand_ssa ? strdup(old_expand_ssa) : NULL;
  unsetenv("EXPAND_SSA");

  PolyDType ptr_f16 = poly_dtype_ptr(POLY_FLOAT16, -1, POLY_ADDR_GLOBAL);
  PolyUOp *a = poly_uop0(ctx, POLY_OP_PARAM, ptr_f16, poly_arg_int(0));
  PolyUOp *b = poly_uop0(ctx, POLY_OP_PARAM, ptr_f16, poly_arg_int(1));
  PolyUOp *out = poly_uop0(ctx, POLY_OP_PARAM, ptr_f16, poly_arg_int(2));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(0));
  PolyUOp *a_idx =
      poly_uop2(ctx, POLY_OP_INDEX, ptr_f16, a, zero, poly_arg_none());
  PolyUOp *b_idx =
      poly_uop2(ctx, POLY_OP_INDEX, ptr_f16, b, zero, poly_arg_none());
  PolyUOp *out_idx =
      poly_uop2(ctx, POLY_OP_INDEX, ptr_f16, out, zero, poly_arg_none());
  PolyUOp *a_load =
      poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT16, a_idx, poly_arg_none());
  PolyUOp *b_load =
      poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT16, b_idx, poly_arg_none());
  PolyUOp *scale =
      poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT16, poly_arg_float(1.702));
  PolyUOp *inner =
      poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT16, b_load, scale, poly_arg_none());
  PolyUOp *value =
      poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT16, a_load, inner, poly_arg_none());
  PolyUOp *store =
      poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_idx, value, poly_arg_none());
  PolyUOp *linear[] = {
      a, b, out, zero, a_idx, b_idx, out_idx,
      a_load, b_load, scale, inner, value, store,
  };

  char *src = poly_render_c(
      linear, (int)(sizeof(linear) / sizeof(linear[0])), "half_chain"
  );
  ASSERT_NOT_NULL(src);
  /* Pinned cstyle.py:40-43 casts a larger literal to half, :62-63 removes
   * same-MUL child parentheses, and :232-237 inlines both one-use MULs. */
  ASSERT_TRUE(strstr(src, "__fp16 alu") == NULL);
  ASSERT_NOT_NULL(strstr(src, "((__fp16)(1.702"));
  ASSERT_NOT_NULL(strstr(src, "(val0*val1*"));

  if (saved_expand_ssa)
    setenv("EXPAND_SSA", saved_expand_ssa, 1);
  else
    unsetenv("EXPAND_SSA");
  free(saved_expand_ssa);
  free(src);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, render_vecmul) {
  VecKernel k = make_vec_binop(POLY_OP_MUL, 8);
  int n;
  PolyUOp **lin = poly_linearize(k.ctx, k.sink, &n);
  char *src = poly_render_c(lin, n, "vecmul");

  /* Default C path follows tinygrad's ClangRenderer float4 upcast:
   * 8 elements become a 2-iteration loop with vec4 loads/stores. */
  ASSERT_NOT_NULL(strstr(src, "void vecmul("));
  ASSERT_NOT_NULL(strstr(src, "ridx0 < 2"));
  ASSERT_NOT_NULL(strstr(src, "__attribute__((vector_size(16)))"));
  ASSERT_NOT_NULL(strstr(src, "val0[0]"));
  ASSERT_NOT_NULL(strstr(src, "*"));

  free(src);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

/* End-to-end tests */

TEST(codegen, render_int64_min_literal_is_portable) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_i64 = poly_dtype_ptr(POLY_INT64, -1, POLY_ADDR_GLOBAL);

  PolyUOp *out = poly_uop0(ctx, POLY_OP_PARAM, ptr_i64, poly_arg_int(0));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(0));
  PolyUOp *idx = poly_uop2(ctx, POLY_OP_INDEX, ptr_i64, out, zero, poly_arg_none());
  PolyUOp *val = poly_uop0(ctx, POLY_OP_CONST, POLY_INT64, poly_arg_int(INT64_MIN));
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx, val, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  int n = 0;
  PolyUOp **lin = poly_linearize(ctx, sink, &n);
  ASSERT_NOT_NULL(lin);
  char *src = poly_render_c(lin, n, "store_i64_min");
  ASSERT_NOT_NULL(src);
  ASSERT_TRUE(strstr(src, "(-9223372036854775807ll - 1ll)") != NULL);
  ASSERT_TRUE(strstr(src, "-9223372036854775808ll") == NULL);

  PolyProgram *prog = poly_compile_c(src, "store_i64_min");
  ASSERT_NOT_NULL(prog);
  int64_t out_data = 0;
  void *args[1] = {&out_data};
  poly_program_call(prog, args, 1);
  ASSERT_TRUE(out_data == INT64_MIN);

  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, exact_uint64_bigint_const_executes_like_tinygrad) {
  /* Pinned CStyleLanguage truncates uint64 CONST args at rendering
   * (renderer/cstyle.py:37), while the UOp retains the exact Python int. */
  PolyCtx *ctx = poly_ctx_new();
  PolyInt value = {0};
  ASSERT_TRUE(poly_int_from_decimal(&value, "18446744073709550593"));
  PolyUOp *constant =
      poly_uop0(ctx, POLY_OP_CONST, POLY_UINT64, poly_int_as_arg(&value));
  poly_int_free(&value);

  PolyDType ptr = poly_dtype_ptr(POLY_UINT64, 1, POLY_ADDR_GLOBAL);
  PolyUOp *out = poly_uop0(ctx, POLY_OP_PARAM, ptr, poly_arg_int(0));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(0));
  PolyUOp *index = poly_uop2(ctx, POLY_OP_INDEX, ptr, out, zero, poly_arg_none());
  PolyUOp *sink = poly_sink1(
      ctx, poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, index, constant, poly_arg_none())
  );

  int n = 0;
  PolyUOp **lin = poly_linearize(ctx, sink, &n);
  ASSERT_NOT_NULL(lin);
  char *src = poly_render_c(lin, n, "store_exact_uint64");
  ASSERT_NOT_NULL(src);
  ASSERT_NOT_NULL(strstr(src, "18446744073709550593ull"));
  PolyProgram *prog = poly_compile_c(src, "store_exact_uint64");
  ASSERT_NOT_NULL(prog);

  uint64_t output = 0;
  void *args[1] = {&output};
  poly_program_call(prog, args, 1);
  ASSERT_TRUE(output == UINT64_C(18446744073709550593));
  output = 0;
  ASSERT_INT_EQ(poly_interp_eval(lin, n, args, 1), 0);
  ASSERT_TRUE(output == UINT64_C(18446744073709550593));

  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, raw_bool_neg_matches_pinned_arithmetic_typed_identity) {
  /* Pinned CStyleLanguage renders raw NEG as -x (renderer/cstyle.py:128-130).
   * Standard logical NOT remains CMPNE(x,true); raw bool NEG normalizes back
   * to bool and therefore preserves False/True. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyDType ptr_bool = poly_dtype_ptr(POLY_BOOL, 1, POLY_ADDR_GLOBAL);
  PolyUOp *out = poly_uop0(ctx, POLY_OP_PARAM, ptr_bool, poly_arg_int(0));
  PolyUOp *in = poly_uop0(ctx, POLY_OP_PARAM, ptr_bool, poly_arg_int(1));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(0));
  PolyUOp *out_idx = poly_uop2(ctx, POLY_OP_INDEX, ptr_bool, out, zero, poly_arg_none());
  PolyUOp *in_idx = poly_uop2(ctx, POLY_OP_INDEX, ptr_bool, in, zero, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_BOOL, in_idx, poly_arg_none());
  PolyUOp *neg = poly_uop1(ctx, POLY_OP_NEG, POLY_BOOL, load, poly_arg_none());
  PolyUOp *store =
      poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_idx, neg, poly_arg_none());
  PolyUOp *sink = poly_sink1(ctx, store);

  int n = 0;
  PolyUOp **lin = poly_linearize(ctx, sink, &n);
  ASSERT_NOT_NULL(lin);
  char *src = poly_render_c(lin, n, "raw_bool_neg");
  ASSERT_NOT_NULL(src);
  ASSERT_NOT_NULL(strstr(src, "-"));
  ASSERT_TRUE(strstr(src, "!") == NULL);

  PolyProgram *prog = poly_compile_c(src, "raw_bool_neg");
  ASSERT_NOT_NULL(prog);
  uint8_t in_false = 0, in_true = 1, out_value = 0;
  void *args[2] = {&out_value, &in_false};
  poly_program_call(prog, args, 2);
  ASSERT_INT_EQ(out_value, 0);
  args[1] = &in_true;
  out_value = 0;
  poly_program_call(prog, args, 2);
  ASSERT_INT_EQ(out_value, 1);

  args[1] = &in_false;
  out_value = 1;
  ASSERT_INT_EQ(poly_interp_eval(lin, n, args, 2), 0);
  ASSERT_INT_EQ(out_value, 0);
  args[1] = &in_true;
  out_value = 0;
  ASSERT_INT_EQ(poly_interp_eval(lin, n, args, 2), 0);
  ASSERT_INT_EQ(out_value, 1);

  char *wgsl = poly_render_wgsl(lin, n, "raw_bool_neg_wgsl");
  ASSERT_NOT_NULL(wgsl);
  ASSERT_TRUE(strstr(wgsl, "(!") == NULL);

  poly_program_destroy(prog);
  free(wgsl);
  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, e2e_vecadd) {
  /* c[i] = a[i] + b[i] for i in 0..9 */
  VecKernel k = make_vec_binop(POLY_OP_ADD, 10);
  int n;
  PolyUOp **lin = poly_linearize(k.ctx, k.sink, &n);
  char *src = poly_render_c(lin, n, "vecadd");

  PolyProgram *prog = poly_compile_c(src, "vecadd");
  ASSERT_NOT_NULL(prog);

  float a[10], b[10], c[10];
  for (int i = 0; i < 10; i++) {
    a[i] = (float)(i + 1); /* 1, 2, ..., 10 */
    b[i] = (float)((i + 1) * 10); /* 10, 20, ..., 100 */
    c[i] = 0.0f;
  }

  void *args[3] = {a, b, c};
  poly_program_call(prog, args, 3);

  for (int i = 0; i < 10; i++) {
    ASSERT_FLOAT_EQ(c[i], a[i] + b[i], 1e-6);
  }

  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(codegen, e2e_vecmul) {
  /* c[i] = a[i] * b[i] for i in 0..7 */
  VecKernel k = make_vec_binop(POLY_OP_MUL, 8);
  int n;
  PolyUOp **lin = poly_linearize(k.ctx, k.sink, &n);
  char *src = poly_render_c(lin, n, "vecmul");

  PolyProgram *prog = poly_compile_c(src, "vecmul");
  ASSERT_NOT_NULL(prog);

  float a[8], b[8], c[8];
  for (int i = 0; i < 8; i++) {
    a[i] = (float)(i + 1);
    b[i] = 0.5f;
    c[i] = 0.0f;
  }

  void *args[3] = {a, b, c};
  poly_program_call(prog, args, 3);

  for (int i = 0; i < 8; i++) {
    ASSERT_FLOAT_EQ(c[i], a[i] * 0.5f, 1e-6);
  }

  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(codegen, e2e_vecsub) {
  /* c[i] = a[i] - b[i] for i in 0..3 */
  VecKernel k = make_vec_binop(POLY_OP_SUB, 4);
  int n;
  PolyUOp **lin = poly_linearize(k.ctx, k.sink, &n);
  char *src = poly_render_c(lin, n, "vecsub");

  PolyProgram *prog = poly_compile_c(src, "vecsub");
  ASSERT_NOT_NULL(prog);

  float a[4] = {10, 20, 30, 40};
  float b[4] = {1, 2, 3, 4};
  float c[4] = {0};

  void *args[3] = {a, b, c};
  poly_program_call(prog, args, 3);

  for (int i = 0; i < 4; i++) {
    ASSERT_FLOAT_EQ(c[i], a[i] - b[i], 1e-6);
  }

  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

/* WGSL renderer tests */

TEST(codegen, render_wgsl_vecadd) {
  VecKernel k = make_vec_binop(POLY_OP_ADD, 10);
  int n;
  PolyUOp **lin = poly_linearize(k.ctx, k.sink, &n);
  char *src = poly_render_wgsl(lin, n, "vecadd");

  /* preamble: INFINITY uniform at binding(0) */
  ASSERT_NOT_NULL(strstr(src, "fn nan()"));
  ASSERT_NOT_NULL(strstr(src, "var<uniform> INFINITY"));
  ASSERT_NOT_NULL(strstr(src, "@group(0) @binding(0)"));

  /* buffer bindings: offset by +1 (binding 0 = INFINITY) */
  ASSERT_NOT_NULL(strstr(src, "@group(0) @binding(1)"));
  ASSERT_NOT_NULL(strstr(src, "var<storage,read_write> data0: array<f32>"));
  ASSERT_NOT_NULL(strstr(src, "@group(0) @binding(2)"));
  ASSERT_NOT_NULL(strstr(src, "var<storage,read_write> data1: array<f32>"));
  ASSERT_NOT_NULL(strstr(src, "@group(0) @binding(3)"));
  ASSERT_NOT_NULL(strstr(src, "var<storage,read_write> data2: array<f32>"));

  /* compute shader entry point: workgroup_id + local_invocation_id */
  ASSERT_NOT_NULL(strstr(src, "@compute @workgroup_size(1)"));
  ASSERT_NOT_NULL(strstr(src, "fn vecadd("));
  ASSERT_NOT_NULL(strstr(src, "@builtin(workgroup_id) gindex"));
  ASSERT_NOT_NULL(strstr(src, "@builtin(local_invocation_id) lindex"));

  /* loop */
  ASSERT_NOT_NULL(strstr(src, "for (var ridx0: i32 = 0; ridx0 < 10; ridx0++)"));

  /* array indexing (not pointer arithmetic) */
  ASSERT_NOT_NULL(strstr(src, "data0[ridx0]"));
  ASSERT_NOT_NULL(strstr(src, "data1[ridx0]"));
  ASSERT_NOT_NULL(strstr(src, "data2[ridx0]"));

  /* variable declarations with WGSL types */
  ASSERT_NOT_NULL(strstr(src, "var val0: f32"));
  ASSERT_NOT_NULL(strstr(src, "var val1: f32"));
  ASSERT_NOT_NULL(strstr(src, "var alu0: f32"));

  /* no C-style wrapper */
  ASSERT_TRUE(strstr(src, "_call") == NULL);

  free(src);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(codegen, render_wgsl_vecmul) {
  VecKernel k = make_vec_binop(POLY_OP_MUL, 8);
  int n;
  PolyUOp **lin = poly_linearize_webgpu(k.ctx, k.sink, &n);
  char *src = poly_render_wgsl(lin, n, "vecmul");

  ASSERT_NOT_NULL(strstr(src, "fn vecmul("));
  ASSERT_NOT_NULL(strstr(src, "@compute @workgroup_size(2,1,1)"));
  ASSERT_NOT_NULL(strstr(src, "var lidx0: i32 = i32(lindex.x);"));
  /* The tinygrad-style recursive tuplize tiebreak may order vector lanes as
   * offsets first and base lane last. The semantic contract is four scalar
   * stores to the four lanes, not a specific temporary-number pairing. */
  ASSERT_NOT_NULL(strstr(src, "data2[alu0] ="));
  ASSERT_NOT_NULL(strstr(src, "data2[alu1] ="));
  ASSERT_NOT_NULL(strstr(src, "data2[alu2] ="));
  ASSERT_NOT_NULL(strstr(src, "data2[alu3] ="));
  ASSERT_NOT_NULL(strstr(src, "*")); /* multiply operator */
  ASSERT_TRUE(strstr(src, "for (var ridx0") == NULL);
  ASSERT_TRUE(strstr(src, "vec4<f32>") == NULL);

  free(src);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(codegen, webgpu_preserves_native_sin_without_long_shift) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(1));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(4));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *in_idx = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p0, range, poly_arg_none());
  PolyUOp *out_idx = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p1, range, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, in_idx, poly_arg_none());
  PolyUOp *sin_uop = poly_uop1(ctx, POLY_OP_SIN, POLY_FLOAT32, load, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_idx, sin_uop, poly_arg_none());
  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  int n = 0;
  PolyUOp **lin = poly_linearize_webgpu(ctx, sink, &n);
  ASSERT_NOT_NULL(lin);
  /* The four-element loop may be scalarized into four native SINs.  The
   * renderer contract is that SIN survives lowering, not a particular
   * scalarization width or count. */
  ASSERT_TRUE(count_lin_ops(lin, n, POLY_OP_SIN) > 0);
  for (int i = 0; i < n; i++) {
    PolyDType scalar = poly_dtype_scalar(lin[i]->dtype);
    ASSERT_TRUE(!poly_dtype_is_int(scalar) || scalar.bitsize != 64);
  }

  char *src = poly_render_wgsl(lin, n, "native_sin");
  ASSERT_NOT_NULL(src);
  ASSERT_NOT_NULL(strstr(src, "sin("));
  ASSERT_TRUE(strstr(src, "<<32u") == NULL);
  ASSERT_TRUE(strstr(src, "<< 32u") == NULL);

  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, webgpu_keeps_reciprocal_without_fdiv_like_pinned_wgsl) {
  /* Pinned WGSLRenderer inherits RECIPROCAL from CStyleLanguage and does not
   * advertise FDIV (renderer/wgsl.py:56-66). */
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *in = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));
  PolyUOp *out = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(1));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(4));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *in_idx = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, in, range, poly_arg_none());
  PolyUOp *out_idx = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, out, range, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, in_idx, poly_arg_none());
  PolyUOp *reciprocal =
      poly_uop1(ctx, POLY_OP_RECIPROCAL, POLY_FLOAT32, load, poly_arg_none());
  PolyUOp *store =
      poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_idx, reciprocal, poly_arg_none());
  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_sink1(ctx, end);

  int n = 0;
  PolyUOp **lin = poly_linearize_webgpu(ctx, sink, &n);
  ASSERT_NOT_NULL(lin);
  ASSERT_TRUE(count_lin_ops(lin, n, POLY_OP_RECIPROCAL) > 0);
  ASSERT_INT_EQ(count_lin_ops(lin, n, POLY_OP_FDIV), 0);

  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, webgpu_decomposes_bf16_before_wgsl_render) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, 4, POLY_ADDR_GLOBAL);
  PolyDType ptr_bf16 = poly_dtype_ptr(POLY_BFLOAT16, 4, POLY_ADDR_GLOBAL);
  PolyUOp *out = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));
  PolyUOp *in = poly_uop0(ctx, POLY_OP_PARAM, ptr_bf16, poly_arg_int(1));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(4));
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_INDEX, bound, poly_arg_range(0, POLY_AXIS_LOOP));
  PolyUOp *out_idx = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, out, range, poly_arg_none());
  PolyUOp *in_idx = poly_uop2(ctx, POLY_OP_INDEX, ptr_bf16, in, range, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_BFLOAT16, in_idx, poly_arg_none());
  PolyUOp *value = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, load, poly_arg_none());
  PolyUOp *sink =
      poly_sink1(ctx, poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_idx, value, poly_arg_none()));

  PolyUOp *rewritten = poly_rewrite_webgpu(ctx, sink);
  ASSERT_NOT_NULL(rewritten);
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  ASSERT_NOT_NULL(topo);
  for (int i = 0; i < n_topo; i++) {
    PolyDType scalar = poly_dtype_scalar(topo[i]->dtype);
    ASSERT_FALSE(
        scalar.priority == POLY_BFLOAT16.priority && strcmp(scalar.name, POLY_BFLOAT16.name) == 0
    );
  }

  int n_lin = 0;
  PolyUOp **lin = poly_linearize_rewritten(ctx, rewritten, &n_lin);
  ASSERT_NOT_NULL(lin);
  char *src = poly_render_wgsl(lin, n_lin, "bf16_to_f32");
  ASSERT_NOT_NULL(src);
  ASSERT_NOT_NULL(strstr(src, "array<atomic<u32>>"));
  ASSERT_TRUE(strstr(src, "data1: array<f32>") == NULL);

  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, python311_f16_dtype_decomposition_keeps_native_transcendental) {
  /* Pinned PythonRenderer on Python 3.11 keeps EXP2 in code_for_op but
   * emulates unsupported f16 LOAD/ALU/STORE through f32 and uint16 storage
   * (ops_python.py:203-223; decompositions.py:388-429,532-564). */
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_f16 = poly_dtype_ptr(POLY_FLOAT16, 4, POLY_ADDR_GLOBAL);
  PolyUOp *out = poly_uop0(ctx, POLY_OP_PARAM, ptr_f16, poly_arg_int(0));
  PolyUOp *in = poly_uop0(ctx, POLY_OP_PARAM, ptr_f16, poly_arg_int(1));
  PolyUOp *idx = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(0));
  PolyUOp *out_idx = poly_uop2(ctx, POLY_OP_INDEX, ptr_f16, out, idx, poly_arg_none());
  PolyUOp *in_idx = poly_uop2(ctx, POLY_OP_INDEX, ptr_f16, in, idx, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT16, in_idx, poly_arg_none());
  PolyUOp *exp2 = poly_uop1(ctx, POLY_OP_EXP2, POLY_FLOAT16, load, poly_arg_none());
  PolyUOp *sink =
      poly_sink1(ctx, poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_idx, exp2, poly_arg_none()));

  PolyUOp *rewritten =
      poly_graph_rewrite_ex(ctx, sink, poly_pm_f16_non_native(), true);
  ASSERT_NOT_NULL(rewritten);
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  ASSERT_NOT_NULL(topo);
  ASSERT_INT_EQ(count_lin_ops(topo, n_topo, POLY_OP_EXP2), 1);
  int f32_exp2 = 0, u16_load = 0, u16_store_value = 0, residual_f16 = 0;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    PolyDType scalar = poly_dtype_scalar(u->dtype);
    if (u->op == POLY_OP_EXP2 && poly_dtype_eq(u->dtype, POLY_FLOAT32)) f32_exp2++;
    if (u->op == POLY_OP_LOAD && poly_dtype_eq(u->dtype, POLY_UINT16)) u16_load++;
    if (u->op == POLY_OP_STORE && u->n_src == 2 &&
        poly_dtype_eq(u->src[1]->dtype, POLY_UINT16))
      u16_store_value++;
    if (poly_dtype_is_float(scalar) && scalar.priority == POLY_FLOAT16.priority &&
        scalar.bitsize == 16)
      residual_f16++;
  }
  ASSERT_INT_EQ(f32_exp2, 1);
  ASSERT_INT_EQ(u16_load, 1);
  ASSERT_INT_EQ(u16_store_value, 1);
  ASSERT_INT_EQ(residual_f16, 0);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, hip_bf16_final_matcher_preserves_wmma_fragments) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyDType bf16x4 = poly_dtype_vec(POLY_BFLOAT16, 4);
  PolyDType f32x4 = poly_dtype_vec(POLY_FLOAT32, 4);

  PolyUOp *a = poly_uop0(ctx, POLY_OP_CONST, POLY_BFLOAT16, poly_arg_float(1.0));
  PolyUOp *b = poly_uop0(ctx, POLY_OP_CONST, POLY_BFLOAT16, poly_arg_float(2.0));
  PolyUOp *const_rewritten = poly_graph_rewrite(ctx, a, poly_pm_bf16_renderer_extra());
  ASSERT_NOT_NULL(const_rewritten);
  ASSERT_INT_EQ(const_rewritten->op, POLY_OP_BITCAST);
  ASSERT_TRUE(poly_dtype_eq(const_rewritten->dtype, POLY_BFLOAT16));
  ASSERT_INT_EQ(const_rewritten->n_src, 1);
  ASSERT_INT_EQ(const_rewritten->src[0]->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(const_rewritten->src[0]->dtype, POLY_UINT16));

  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_BFLOAT16, a, b, poly_arg_none());
  PolyUOp *add_rewritten = poly_graph_rewrite(ctx, add, poly_pm_bf16_renderer_extra());
  ASSERT_NOT_NULL(add_rewritten);
  ASSERT_INT_EQ(add_rewritten->op, POLY_OP_BITCAST);
  ASSERT_TRUE(poly_dtype_eq(add_rewritten->dtype, POLY_BFLOAT16));
  ASSERT_INT_EQ(add_rewritten->n_src, 1);
  ASSERT_INT_EQ(add_rewritten->src[0]->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(add_rewritten->src[0]->dtype, POLY_UINT16));

  PolyUOp *a_src[] = {a, a, a, a};
  PolyUOp *b_src[] = {b, b, b, b};
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(0.0));
  PolyUOp *c_src[] = {zero, zero, zero, zero};
  PolyUOp *a_vec = poly_uop(ctx, POLY_OP_STACK, bf16x4, a_src, 4, poly_arg_none());
  PolyUOp *b_vec = poly_uop(ctx, POLY_OP_STACK, bf16x4, b_src, 4, poly_arg_none());
  PolyUOp *c_vec = poly_uop(ctx, POLY_OP_STACK, f32x4, c_src, 4, poly_arg_none());
  PolyUOp *wmma_src[] = {a_vec, b_vec, c_vec};
  PolyUOp *wmma =
      poly_uop(ctx, POLY_OP_WMMA, f32x4, wmma_src, 3, poly_arg_str("mfma_f32_16x16x16bf16_1k"));
  PolyUOp *wmma_rewritten = poly_graph_rewrite(ctx, wmma, poly_pm_bf16_renderer_extra());
  /* Pinned graph_rewrite rebuilds the WMMA because its BF16 CONST leaves are
   * converted, but preserves the WMMA and BF16 fragment topology. */
  ASSERT_NOT_NULL(wmma_rewritten);
  ASSERT_INT_EQ(wmma_rewritten->op, POLY_OP_WMMA);
  ASSERT_FALSE(wmma_rewritten == wmma);
  ASSERT_FALSE(wmma_rewritten->src[0] == a_vec);
  ASSERT_FALSE(wmma_rewritten->src[1] == b_vec);
  ASSERT_INT_EQ(wmma_rewritten->src[0]->op, POLY_OP_STACK);
  ASSERT_INT_EQ(wmma_rewritten->src[1]->op, POLY_OP_STACK);
  ASSERT_TRUE(poly_dtype_eq(wmma_rewritten->src[0]->dtype, bf16x4));
  ASSERT_TRUE(poly_dtype_eq(wmma_rewritten->src[1]->dtype, bf16x4));
  for (int lane = 0; lane < 4; lane++) {
    ASSERT_INT_EQ(wmma_rewritten->src[0]->src[lane]->op, POLY_OP_BITCAST);
    ASSERT_INT_EQ(wmma_rewritten->src[1]->src[lane]->op, POLY_OP_BITCAST);
    ASSERT_TRUE(poly_dtype_eq(wmma_rewritten->src[0]->src[lane]->dtype, POLY_BFLOAT16));
    ASSERT_TRUE(poly_dtype_eq(wmma_rewritten->src[1]->src[lane]->dtype, POLY_BFLOAT16));
  }

  poly_ctx_destroy(ctx);
  PASS();
}

#ifdef POLY_HAS_HIP
TEST(codegen, hip_bf16_dynamic_vector_scalarizes_before_final_matcher) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyDType bf16x4 = poly_dtype_vec(POLY_BFLOAT16, 4);
  PolyDType ptr_bf16 = poly_dtype_ptr(POLY_BFLOAT16, 16, POLY_ADDR_GLOBAL);
  PolyDType ptr_bf16x4 = poly_dtype_ptr(bf16x4, 16, POLY_ADDR_GLOBAL);
  PolyDType f32x4 = poly_dtype_vec(POLY_FLOAT32, 4);

  PolyUOp *a_buf = poly_uop0(ctx, POLY_OP_PARAM, ptr_bf16, poly_arg_int(0));
  PolyUOp *b_buf = poly_uop0(ctx, POLY_OP_PARAM, ptr_bf16, poly_arg_int(1));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(0));
  PolyUOp *a_idx = poly_uop2(ctx, POLY_OP_INDEX, ptr_bf16, a_buf, zero, poly_arg_none());
  PolyUOp *b_idx = poly_uop2(ctx, POLY_OP_INDEX, ptr_bf16, b_buf, zero, poly_arg_none());
  PolyUOp *a_vec = poly_uop1(
      ctx, POLY_OP_LOAD, bf16x4, poly_uop1(ctx, POLY_OP_CAST, ptr_bf16x4, a_idx, poly_arg_none()),
      poly_arg_none()
  );
  PolyUOp *b_vec = poly_uop1(
      ctx, POLY_OP_LOAD, bf16x4, poly_uop1(ctx, POLY_OP_CAST, ptr_bf16x4, b_idx, poly_arg_none()),
      poly_arg_none()
  );
  PolyUOp *add_vec = poly_uop2(ctx, POLY_OP_ADD, bf16x4, a_vec, b_vec, poly_arg_none());
  PolyUOp *zero_vec = poly_uop0(ctx, POLY_OP_CONST, f32x4, poly_arg_float(0.0));
  PolyUOp *wmma_src[] = {add_vec, add_vec, zero_vec};
  PolyUOp *wmma =
      poly_uop(ctx, POLY_OP_WMMA, f32x4, wmma_src, 3, poly_arg_str("mfma_f32_16x16x16bf16_1k"));

  PolyUOp *rewritten = poly_rewrite_hip(ctx, poly_sink1(ctx, wmma));
  ASSERT_NOT_NULL(rewritten);
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  ASSERT_NOT_NULL(topo);
  int raw_bf16_consts = 0;
  int encoded_scalar_bf16_lanes = 0;
  int scalar_bf16_loads = 0;
  int vector_bf16_loads = 0;
  int scalar_f32_adds = 0;
  int vector_bf16_alu = 0;
  int vector_bf16_bitcasts = 0;
  int wmma_count = 0;
  PolyUOp *wmma_rewritten = NULL;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op == POLY_OP_CONST &&
        poly_dtype_eq(poly_dtype_scalar(topo[i]->dtype), POLY_BFLOAT16))
      raw_bf16_consts++;
    if (topo[i]->op == POLY_OP_BITCAST) {
      if (poly_dtype_eq(topo[i]->dtype, POLY_BFLOAT16))
        encoded_scalar_bf16_lanes++;
      else if (poly_dtype_eq(poly_dtype_scalar(topo[i]->dtype), POLY_BFLOAT16) && topo[i]->dtype.count > 1)
        vector_bf16_bitcasts++;
    }
    if (topo[i]->op == POLY_OP_LOAD &&
        poly_dtype_eq(poly_dtype_scalar(topo[i]->dtype), POLY_BFLOAT16)) {
      if (topo[i]->dtype.count == 1)
        scalar_bf16_loads++;
      else
        vector_bf16_loads++;
    }
    if (topo[i]->op == POLY_OP_ADD && poly_dtype_eq(topo[i]->dtype, POLY_FLOAT32))
      scalar_f32_adds++;
    if (poly_opset_has(POLY_GROUP_ALU, topo[i]->op) &&
        poly_dtype_eq(poly_dtype_scalar(topo[i]->dtype), POLY_BFLOAT16) && topo[i]->dtype.count > 1)
      vector_bf16_alu++;
    if (topo[i]->op == POLY_OP_WMMA) {
      wmma_count++;
      wmma_rewritten = topo[i];
    }
  }
  /* Pinned tinygrad codegen/__init__.py:105-107,125-137 scalarizes BF16
   * vector memory and ALU before HIPRenderer.extra_matcher encodes each lane
   * (devectorizer.py:155-177,241-245; renderer/cstyle.py:515-520). */
  ASSERT_INT_EQ(raw_bf16_consts, 0);
  ASSERT_INT_EQ(encoded_scalar_bf16_lanes, 4);
  ASSERT_INT_EQ(scalar_bf16_loads, 8);
  ASSERT_INT_EQ(vector_bf16_loads, 0);
  ASSERT_INT_EQ(scalar_f32_adds, 4);
  ASSERT_INT_EQ(vector_bf16_alu, 0);
  ASSERT_INT_EQ(vector_bf16_bitcasts, 0);
  ASSERT_INT_EQ(wmma_count, 1);
  ASSERT_NOT_NULL(wmma_rewritten);
  ASSERT_INT_EQ(wmma_rewritten->src[0]->op, POLY_OP_STACK);
  ASSERT_INT_EQ(wmma_rewritten->src[1]->op, POLY_OP_STACK);
  ASSERT_INT_EQ(wmma_rewritten->src[0]->n_src, 4);
  ASSERT_INT_EQ(wmma_rewritten->src[1]->n_src, 4);
  for (int lane = 0; lane < 4; lane++) {
    ASSERT_INT_EQ(wmma_rewritten->src[0]->src[lane]->op, POLY_OP_BITCAST);
    ASSERT_INT_EQ(wmma_rewritten->src[1]->src[lane]->op, POLY_OP_BITCAST);
    ASSERT_TRUE(poly_dtype_eq(wmma_rewritten->src[0]->src[lane]->dtype, POLY_BFLOAT16));
    ASSERT_TRUE(poly_dtype_eq(wmma_rewritten->src[1]->src[lane]->dtype, POLY_BFLOAT16));
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, hip_float_and_half4_memory_use_renderer_vector_width) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyDType scalar_dtypes[] = {POLY_FLOAT16, POLY_FLOAT32};
  for (int d = 0; d < 2; d++) {
    PolyDType scalar = scalar_dtypes[d];
    PolyDType vec4 = poly_dtype_vec(scalar, 4);
    PolyDType ptr = poly_dtype_ptr(scalar, 16, POLY_ADDR_GLOBAL);
    PolyDType ptr4 = poly_dtype_ptr(vec4, 16, POLY_ADDR_GLOBAL);
    PolyUOp *out_buf = poly_uop0(ctx, POLY_OP_PARAM, ptr, poly_arg_int(d * 3));
    PolyUOp *a_buf = poly_uop0(ctx, POLY_OP_PARAM, ptr, poly_arg_int(d * 3 + 1));
    PolyUOp *b_buf = poly_uop0(ctx, POLY_OP_PARAM, ptr, poly_arg_int(d * 3 + 2));
    PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(0));
    PolyUOp *out_idx = poly_uop2(ctx, POLY_OP_INDEX, ptr, out_buf, zero, poly_arg_none());
    PolyUOp *a_idx = poly_uop2(ctx, POLY_OP_INDEX, ptr, a_buf, zero, poly_arg_none());
    PolyUOp *b_idx = poly_uop2(ctx, POLY_OP_INDEX, ptr, b_buf, zero, poly_arg_none());
    PolyUOp *out_vptr = poly_uop1(ctx, POLY_OP_CAST, ptr4, out_idx, poly_arg_none());
    PolyUOp *a_vec = poly_uop1(
        ctx, POLY_OP_LOAD, vec4, poly_uop1(ctx, POLY_OP_CAST, ptr4, a_idx, poly_arg_none()),
        poly_arg_none()
    );
    PolyUOp *b_vec = poly_uop1(
        ctx, POLY_OP_LOAD, vec4, poly_uop1(ctx, POLY_OP_CAST, ptr4, b_idx, poly_arg_none()),
        poly_arg_none()
    );
    PolyUOp *add_vec = poly_uop2(ctx, POLY_OP_ADD, vec4, a_vec, b_vec, poly_arg_none());
    PolyUOp *sink = poly_sink1(
        ctx, poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_vptr, add_vec, poly_arg_none())
    );

    PolyUOp *rewritten = poly_rewrite_hip(ctx, sink);
    ASSERT_NOT_NULL(rewritten);
    int n_topo = 0;
    PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
    ASSERT_NOT_NULL(topo);
    int scalar_loads = 0, vector_loads = 0;
    int scalar_adds = 0, vector_adds = 0;
    int scalar_stores = 0, vector_stores = 0;
    for (int i = 0; i < n_topo; i++) {
      if (topo[i]->op == POLY_OP_LOAD && poly_dtype_eq(poly_dtype_scalar(topo[i]->dtype), scalar)) {
        if (topo[i]->dtype.count == 1)
          scalar_loads++;
        else if (topo[i]->dtype.count == 4)
          vector_loads++;
      }
      if (topo[i]->op == POLY_OP_ADD && poly_dtype_eq(poly_dtype_scalar(topo[i]->dtype), scalar)) {
        if (topo[i]->dtype.count == 1)
          scalar_adds++;
        else if (topo[i]->dtype.count == 4)
          vector_adds++;
      }
      if (topo[i]->op == POLY_OP_STORE && topo[i]->n_src >= 2 &&
          poly_dtype_eq(poly_dtype_scalar(topo[i]->src[1]->dtype), scalar)) {
        if (topo[i]->src[1]->dtype.count == 1)
          scalar_stores++;
        else if (topo[i]->src[1]->dtype.count == 4)
          vector_stores++;
      }
    }
    /* Pinned HIP inherits Renderer.supports_float4=True: aligned float16x4
     * and float32x4 memory operations survive while devectorize_alu scalarizes
     * the ADD
     * (renderer/__init__.py:58-63; devectorizer.py:155-177,241-245). */
    ASSERT_INT_EQ(scalar_loads, 0);
    ASSERT_INT_EQ(vector_loads, 2);
    ASSERT_INT_EQ(scalar_adds, 4);
    ASSERT_INT_EQ(vector_adds, 0);
    ASSERT_INT_EQ(scalar_stores, 0);
    ASSERT_INT_EQ(vector_stores, 1);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, hip_bf16_vector_memory_scalarizes) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyDType bf16x4 = poly_dtype_vec(POLY_BFLOAT16, 4);
  PolyDType ptr_bf16 = poly_dtype_ptr(POLY_BFLOAT16, 16, POLY_ADDR_GLOBAL);
  PolyDType ptr_bf16x4 = poly_dtype_ptr(bf16x4, 16, POLY_ADDR_GLOBAL);
  PolyUOp *out_buf = poly_uop0(ctx, POLY_OP_PARAM, ptr_bf16, poly_arg_int(0));
  PolyUOp *in_buf = poly_uop0(ctx, POLY_OP_PARAM, ptr_bf16, poly_arg_int(1));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(0));
  PolyUOp *out_idx = poly_uop2(ctx, POLY_OP_INDEX, ptr_bf16, out_buf, zero, poly_arg_none());
  PolyUOp *in_idx = poly_uop2(ctx, POLY_OP_INDEX, ptr_bf16, in_buf, zero, poly_arg_none());
  PolyUOp *out_vptr = poly_uop1(ctx, POLY_OP_CAST, ptr_bf16x4, out_idx, poly_arg_none());
  PolyUOp *value = poly_uop1(
      ctx, POLY_OP_LOAD, bf16x4, poly_uop1(ctx, POLY_OP_CAST, ptr_bf16x4, in_idx, poly_arg_none()),
      poly_arg_none()
  );
  PolyUOp *sink =
      poly_sink1(ctx, poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_vptr, value, poly_arg_none()));
  PolyUOp *rewritten = poly_rewrite_hip(ctx, sink);
  ASSERT_NOT_NULL(rewritten);

  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  ASSERT_NOT_NULL(topo);
  int scalar_loads = 0, vector_loads = 0;
  int scalar_stores = 0, vector_stores = 0;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op == POLY_OP_LOAD &&
        poly_dtype_eq(poly_dtype_scalar(topo[i]->dtype), POLY_BFLOAT16)) {
      if (topo[i]->dtype.count == 1)
        scalar_loads++;
      else if (topo[i]->dtype.count == 4)
        vector_loads++;
    }
    if (topo[i]->op == POLY_OP_STORE && topo[i]->n_src >= 2 &&
        poly_dtype_eq(poly_dtype_scalar(topo[i]->src[1]->dtype), POLY_BFLOAT16)) {
      if (topo[i]->src[1]->dtype.count == 1)
        scalar_stores++;
      else if (topo[i]->src[1]->dtype.count == 4)
        vector_stores++;
    }
  }
  /* Pinned split_load_store only retains vector memory for float/half/fp8;
   * BF16 takes the scalar fallback (devectorizer.py:155-177). */
  ASSERT_INT_EQ(scalar_loads, 4);
  ASSERT_INT_EQ(vector_loads, 0);
  ASSERT_INT_EQ(scalar_stores, 4);
  ASSERT_INT_EQ(vector_stores, 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, hip_gated_bf16_load_legalizes_late_zero_alternative) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyDType ptr_bf16 = poly_dtype_ptr(POLY_BFLOAT16, 1, POLY_ADDR_GLOBAL);
  PolyUOp *buf = poly_uop0(ctx, POLY_OP_PARAM, ptr_bf16, poly_arg_int(0));
  PolyUOp *gate = poly_uop0(ctx, POLY_OP_PARAM, POLY_BOOL, poly_arg_int(1));
  PolyUOp *offset = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(0));
  PolyUOp *invalid = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_invalid());
  PolyUOp *gated_offset =
      poly_uop3(ctx, POLY_OP_WHERE, POLY_INDEX, gate, offset, invalid, poly_arg_none());
  PolyUOp *index = poly_uop2(ctx, POLY_OP_INDEX, ptr_bf16, buf, gated_offset, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_BFLOAT16, index, poly_arg_none());
  PolyUOp *rewritten = poly_rewrite_hip(ctx, poly_sink1(ctx, load));
  ASSERT_NOT_NULL(rewritten);

  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  ASSERT_NOT_NULL(topo);
  PolyUOp *gated_load = NULL;
  int raw_bf16_consts = 0;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op == POLY_OP_LOAD && poly_dtype_eq(topo[i]->dtype, POLY_BFLOAT16) &&
        topo[i]->n_src == 3)
      gated_load = topo[i];
    if (topo[i]->op == POLY_OP_CONST && poly_dtype_eq(topo[i]->dtype, POLY_BFLOAT16))
      raw_bf16_consts++;
  }
  /* Pinned codegen/__init__.py:124-137 moves gates before the renderer-final
   * matcher. The BF16 zero alternative introduced by gater.py:14-18 is
   * therefore encoded by HIPRenderer.extra_matcher instead of remaining a raw
   * numeric CONST in uint16 storage. */
  ASSERT_NOT_NULL(gated_load);
  ASSERT_INT_EQ(gated_load->src[1]->op, POLY_OP_BITCAST);
  ASSERT_TRUE(poly_dtype_eq(gated_load->src[1]->dtype, POLY_BFLOAT16));
  ASSERT_INT_EQ(raw_bf16_consts, 0);

  poly_ctx_destroy(ctx);
  PASS();
}
#endif

TEST(codegen, webgpu_decomposes_u64_threefry_buffers_to_u32_lanes) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_u64 = poly_dtype_ptr(POLY_UINT64, 1, POLY_ADDR_GLOBAL);
  PolyUOp *out = poly_uop0(ctx, POLY_OP_PARAM, ptr_u64, poly_arg_int(0));
  PolyUOp *xbuf = poly_uop0(ctx, POLY_OP_PARAM, ptr_u64, poly_arg_int(1));
  PolyUOp *kbuf = poly_uop0(ctx, POLY_OP_PARAM, ptr_u64, poly_arg_int(2));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(0));
  PolyUOp *oidx = poly_uop2(ctx, POLY_OP_INDEX, ptr_u64, out, zero, poly_arg_none());
  PolyUOp *xidx = poly_uop2(ctx, POLY_OP_INDEX, ptr_u64, xbuf, zero, poly_arg_none());
  PolyUOp *kidx = poly_uop2(ctx, POLY_OP_INDEX, ptr_u64, kbuf, zero, poly_arg_none());
  PolyUOp *x = poly_uop1(ctx, POLY_OP_LOAD, POLY_UINT64, xidx, poly_arg_none());
  PolyUOp *key = poly_uop1(ctx, POLY_OP_LOAD, POLY_UINT64, kidx, poly_arg_none());
  PolyUOp *value = poly_uop2(ctx, POLY_OP_THREEFRY, POLY_UINT64, x, key, poly_arg_none());
  PolyUOp *sink =
      poly_sink1(ctx, poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, oidx, value, poly_arg_none()));

  PolyUOp *rewritten = poly_rewrite_webgpu(ctx, sink);
  ASSERT_NOT_NULL(rewritten);
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  ASSERT_NOT_NULL(topo);
  ASSERT_TRUE(n_topo > 0);
  ASSERT_INT_EQ(count_lin_ops(topo, n_topo, POLY_OP_THREEFRY), 0);
  ASSERT_INT_EQ(count_lin_ops(topo, n_topo, POLY_OP_LOAD), 4);
  ASSERT_INT_EQ(count_lin_ops(topo, n_topo, POLY_OP_STORE), 2);

  bool lane_seen[3][2] = {{false}};
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    PolyDType scalar = poly_dtype_scalar(u->dtype);
    ASSERT_TRUE(!poly_dtype_is_int(scalar) || scalar.bitsize != 64);
    if (u->op == POLY_OP_PARAM && u->dtype.is_ptr) {
      ASSERT_INT_EQ(poly_dtype_scalar(u->dtype).bitsize, 32);
      ASSERT_INT_EQ((int)u->dtype.ptr_size, 2);
    }
    if ((u->op != POLY_OP_LOAD && u->op != POLY_OP_STORE) || u->n_src < 1) continue;
    PolyUOp *idx = u->src[0];
    ASSERT_TRUE(idx->op == POLY_OP_INDEX && idx->n_src >= 2);
    PolyUOp *param = idx->src[0], *offset = idx->src[1];
    ASSERT_TRUE(param->op == POLY_OP_PARAM && param->arg.kind == POLY_ARG_INT);
    ASSERT_TRUE(offset->op == POLY_OP_CONST && offset->arg.kind == POLY_ARG_INT);
    ASSERT_TRUE(param->arg.i >= 0 && param->arg.i < 3);
    ASSERT_TRUE(offset->arg.i >= 0 && offset->arg.i < 2);
    lane_seen[param->arg.i][offset->arg.i] = true;
  }
  for (int p = 0; p < 3; p++)
    for (int lane = 0; lane < 2; lane++)
      ASSERT_TRUE(lane_seen[p][lane]);

  int n_lin = 0;
  PolyUOp **lin = poly_linearize_rewritten(ctx, rewritten, &n_lin);
  ASSERT_NOT_NULL(lin);
  char *src = poly_render_wgsl(lin, n_lin, "threefry_u64_buffers");
  ASSERT_NOT_NULL(src);
  ASSERT_NOT_NULL(strstr(src, "array<u32>"));
  ASSERT_TRUE(strstr(src, "<<32") == NULL && strstr(src, "<< 32") == NULL);
  ASSERT_TRUE(strstr(src, ">>32") == NULL && strstr(src, ">> 32") == NULL);

  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, webgpu_decomposes_exact_uint64_bigint_const_to_u32_lanes) {
  /* Pinned pm_long_decomp selects low/high uint32 lanes from the exact Python
   * CONST arg (uop/decompositions.py:528-529). */
  PolyCtx *ctx = poly_ctx_new();
  PolyInt value = {0};
  ASSERT_TRUE(poly_int_from_decimal(&value, "18446744073709550593"));
  PolyUOp *constant =
      poly_uop0(ctx, POLY_OP_CONST, POLY_UINT64, poly_int_as_arg(&value));
  poly_int_free(&value);

  PolyDType ptr = poly_dtype_ptr(POLY_UINT64, 1, POLY_ADDR_GLOBAL);
  PolyUOp *out = poly_uop0(ctx, POLY_OP_PARAM, ptr, poly_arg_int(0));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(0));
  PolyUOp *index = poly_uop2(ctx, POLY_OP_INDEX, ptr, out, zero, poly_arg_none());
  PolyUOp *sink = poly_sink1(
      ctx, poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, index, constant, poly_arg_none())
  );

  PolyUOp *rewritten = poly_rewrite_webgpu(ctx, sink);
  ASSERT_NOT_NULL(rewritten);
  int n = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, rewritten, &n);
  ASSERT_NOT_NULL(topo);
  bool saw_low = false, saw_high = false;
  for (int i = 0; i < n; i++) {
    PolyUOp *u = topo[i];
    if (u->op != POLY_OP_CONST || !poly_dtype_eq(u->dtype, POLY_UINT32))
      continue;
    uint64_t lane = poly_arg_integer_to_u64_mod(u->arg);
    if (lane == UINT64_C(4294966273)) saw_low = true;
    if (lane == UINT64_C(4294967295)) saw_high = true;
  }
  ASSERT_TRUE(saw_low);
  ASSERT_TRUE(saw_high);
  poly_toposort_free(topo);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, webgpu_decomposes_long_divmod_without_shift32) {
  struct {
    PolyDType dtype;
    PolyOps op;
  } cases[] = {
      {POLY_INT64, POLY_OP_CDIV},
      {POLY_INT64, POLY_OP_CMOD},
      {POLY_UINT64, POLY_OP_CDIV},
      {POLY_UINT64, POLY_OP_CMOD},
  };

  for (int ci = 0; ci < (int)(sizeof(cases) / sizeof(cases[0])); ci++) {
    PolyCtx *ctx = poly_ctx_new();
    PolyDType ptr = poly_dtype_ptr(cases[ci].dtype, 1, POLY_ADDR_GLOBAL);
    PolyUOp *out = poly_uop0(ctx, POLY_OP_PARAM, ptr, poly_arg_int(0));
    PolyUOp *abuf = poly_uop0(ctx, POLY_OP_PARAM, ptr, poly_arg_int(1));
    PolyUOp *bbuf = poly_uop0(ctx, POLY_OP_PARAM, ptr, poly_arg_int(2));
    PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(0));
    PolyUOp *oidx = poly_uop2(ctx, POLY_OP_INDEX, ptr, out, zero, poly_arg_none());
    PolyUOp *aidx = poly_uop2(ctx, POLY_OP_INDEX, ptr, abuf, zero, poly_arg_none());
    PolyUOp *bidx = poly_uop2(ctx, POLY_OP_INDEX, ptr, bbuf, zero, poly_arg_none());
    PolyUOp *a = poly_uop1(ctx, POLY_OP_LOAD, cases[ci].dtype, aidx, poly_arg_none());
    PolyUOp *b = poly_uop1(ctx, POLY_OP_LOAD, cases[ci].dtype, bidx, poly_arg_none());
    PolyUOp *value = poly_uop2(ctx, cases[ci].op, cases[ci].dtype, a, b, poly_arg_none());
    PolyUOp *sink =
        poly_sink1(ctx, poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, oidx, value, poly_arg_none()));

    PolyUOp *rewritten = poly_rewrite_webgpu(ctx, sink);
    ASSERT_NOT_NULL(rewritten);
    int n_topo = 0;
    PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
    ASSERT_NOT_NULL(topo);
    ASSERT_INT_EQ(count_lin_ops(topo, n_topo, POLY_OP_LOAD), 4);
    ASSERT_INT_EQ(count_lin_ops(topo, n_topo, POLY_OP_STORE), 2);
    for (int i = 0; i < n_topo; i++) {
      PolyUOp *u = topo[i];
      PolyDType scalar = poly_dtype_scalar(u->dtype);
      ASSERT_TRUE(!poly_dtype_is_int(scalar) || scalar.bitsize != 64);
      if ((u->op == POLY_OP_SHL || u->op == POLY_OP_SHR) && u->n_src >= 2 &&
          u->src[1]->op == POLY_OP_CONST && u->src[1]->arg.kind == POLY_ARG_INT)
        ASSERT_TRUE(u->src[1]->arg.i >= 0 && u->src[1]->arg.i < 32);
      if (u->op == POLY_OP_PARAM && u->dtype.is_ptr) {
        ASSERT_INT_EQ(poly_dtype_scalar(u->dtype).bitsize, 32);
        ASSERT_INT_EQ((int)u->dtype.ptr_size, 2);
      }
    }
    poly_ctx_destroy(ctx);
  }
  PASS();
}

TEST(codegen, webgpu_long_shift_preserves_arithmetic_semantics) {
  struct {
    PolyOps op;
    PolyDType dtype;
    uint64_t value;
    uint32_t shift;
    uint64_t expected;
  } cases[] = {
      {POLY_OP_SHR, POLY_INT64, UINT64_C(0x0000000080000000), 1, UINT64_C(0x0000000040000000)},
      {POLY_OP_SHR, POLY_INT64, UINT64_MAX, 32, UINT64_MAX},
      {POLY_OP_SHR, POLY_INT64, UINT64_C(0x8000000000000000), 64, UINT64_MAX},
      {POLY_OP_SHR, POLY_INT64, UINT64_C(0x7fffffffffffffff), 64, 0},
      {POLY_OP_SHL, POLY_INT64, UINT64_C(0x0000000080000000), 1, UINT64_C(0x0000000100000000)},
      {POLY_OP_SHL, POLY_INT64, UINT64_MAX, 64, 0},
      {POLY_OP_SHR, POLY_UINT64, UINT64_MAX, 1, UINT64_C(0x7fffffffffffffff)},
  };

  for (int ci = 0; ci < (int)(sizeof(cases) / sizeof(cases[0])); ci++) {
    PolyCtx *ctx = poly_ctx_new();
    PolyDType ptr = poly_dtype_ptr(cases[ci].dtype, 1, POLY_ADDR_GLOBAL);
    PolyDType shift_ptr = poly_dtype_ptr(POLY_UINT32, 1, POLY_ADDR_GLOBAL);
    PolyUOp *out = poly_uop0(ctx, POLY_OP_PARAM, ptr, poly_arg_int(0));
    PolyUOp *in = poly_uop0(ctx, POLY_OP_PARAM, ptr, poly_arg_int(1));
    PolyUOp *shift = poly_uop0(ctx, POLY_OP_PARAM, shift_ptr, poly_arg_int(2));
    PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(0));
    PolyUOp *oidx = poly_uop2(ctx, POLY_OP_INDEX, ptr, out, zero, poly_arg_none());
    PolyUOp *iidx = poly_uop2(ctx, POLY_OP_INDEX, ptr, in, zero, poly_arg_none());
    PolyUOp *sidx = poly_uop2(ctx, POLY_OP_INDEX, shift_ptr, shift, zero, poly_arg_none());
    PolyUOp *value = poly_uop2(
        ctx, cases[ci].op, cases[ci].dtype,
        poly_uop1(ctx, POLY_OP_LOAD, cases[ci].dtype, iidx, poly_arg_none()),
        poly_uop1(ctx, POLY_OP_LOAD, POLY_UINT32, sidx, poly_arg_none()), poly_arg_none()
    );
    PolyUOp *sink =
        poly_sink1(ctx, poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, oidx, value, poly_arg_none()));
    PolyUOp *rewritten = poly_rewrite_webgpu(ctx, sink);
    ASSERT_NOT_NULL(rewritten);

    int n_topo = 0;
    PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
    ASSERT_NOT_NULL(topo);
    for (int i = 0; i < n_topo; i++) {
      PolyDType scalar = poly_dtype_scalar(topo[i]->dtype);
      ASSERT_TRUE(!poly_dtype_is_int(scalar) || scalar.bitsize != 64);
    }

    int n_lin = 0;
    PolyUOp **lin = poly_linearize_rewritten(ctx, rewritten, &n_lin);
    ASSERT_NOT_NULL(lin);
    uint32_t out_words[2] = {0, 0};
    uint32_t in_words[2] = {(uint32_t)cases[ci].value, (uint32_t)(cases[ci].value >> 32)};
    uint32_t shift_value = cases[ci].shift;
    void *args[3] = {out_words, in_words, &shift_value};
    ASSERT_INT_EQ(poly_interp_eval(lin, n_lin, args, 3), 0);
    uint64_t got = (uint64_t)out_words[0] | ((uint64_t)out_words[1] << 32);
    if (got != cases[ci].expected) {
      free(lin);
      poly_ctx_destroy(ctx);
      FAIL(
          "case %d %s value=0x%016" PRIx64 " shift=%" PRIu32 " got=0x%016" PRIx64
          " expected=0x%016" PRIx64,
          ci, poly_op_name(cases[ci].op), cases[ci].value, cases[ci].shift, got, cases[ci].expected
      );
    }

    free(lin);
    poly_ctx_destroy(ctx);
  }
  PASS();
}

TEST(codegen, webgpu_long_shift_accepts_dynamic_long_count) {
  struct {
    uint64_t value;
    uint64_t shift;
    uint64_t expected;
  } cases[] = {
      {UINT64_C(0x0000000080000000), 1, UINT64_C(0x0000000040000000)},
      {UINT64_C(0x8000000000000000), UINT64_C(1) << 32, UINT64_MAX},
  };

  for (int ci = 0; ci < (int)(sizeof(cases) / sizeof(cases[0])); ci++) {
    PolyCtx *ctx = poly_ctx_new();
    PolyDType ptr = poly_dtype_ptr(POLY_INT64, 1, POLY_ADDR_GLOBAL);
    PolyUOp *out = poly_uop0(ctx, POLY_OP_PARAM, ptr, poly_arg_int(0));
    PolyUOp *in = poly_uop0(ctx, POLY_OP_PARAM, ptr, poly_arg_int(1));
    PolyUOp *shift = poly_uop0(ctx, POLY_OP_PARAM, ptr, poly_arg_int(2));
    PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(0));
    PolyUOp *oidx = poly_uop2(ctx, POLY_OP_INDEX, ptr, out, zero, poly_arg_none());
    PolyUOp *iidx = poly_uop2(ctx, POLY_OP_INDEX, ptr, in, zero, poly_arg_none());
    PolyUOp *sidx = poly_uop2(ctx, POLY_OP_INDEX, ptr, shift, zero, poly_arg_none());
    PolyUOp *value = poly_uop2(
        ctx, POLY_OP_SHR, POLY_INT64,
        poly_uop1(ctx, POLY_OP_LOAD, POLY_INT64, iidx, poly_arg_none()),
        poly_uop1(ctx, POLY_OP_LOAD, POLY_INT64, sidx, poly_arg_none()), poly_arg_none()
    );
    PolyUOp *sink =
        poly_sink1(ctx, poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, oidx, value, poly_arg_none()));
    PolyUOp *rewritten = poly_rewrite_webgpu(ctx, sink);
    ASSERT_NOT_NULL(rewritten);

    int n_topo = 0;
    PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
    ASSERT_NOT_NULL(topo);
    for (int i = 0; i < n_topo; i++) {
      PolyDType scalar = poly_dtype_scalar(topo[i]->dtype);
      ASSERT_TRUE(!poly_dtype_is_int(scalar) || scalar.bitsize != 64);
    }

    int n_lin = 0;
    PolyUOp **lin = poly_linearize_rewritten(ctx, rewritten, &n_lin);
    ASSERT_NOT_NULL(lin);
    uint32_t out_words[2] = {0, 0};
    uint32_t in_words[2] = {(uint32_t)cases[ci].value, (uint32_t)(cases[ci].value >> 32)};
    uint32_t shift_words[2] = {(uint32_t)cases[ci].shift, (uint32_t)(cases[ci].shift >> 32)};
    void *args[3] = {out_words, in_words, shift_words};
    ASSERT_INT_EQ(poly_interp_eval(lin, n_lin, args, 3), 0);
    uint64_t got = (uint64_t)out_words[0] | ((uint64_t)out_words[1] << 32);
    if (got != cases[ci].expected) {
      free(lin);
      poly_ctx_destroy(ctx);
      FAIL(
          "case %d value=0x%016" PRIx64 " shift=0x%016" PRIx64 " got=0x%016" PRIx64
          " expected=0x%016" PRIx64,
          ci, cases[ci].value, cases[ci].shift, got, cases[ci].expected
      );
    }

    free(lin);
    poly_ctx_destroy(ctx);
  }
  PASS();
}

TEST(codegen, webgpu_decomposes_raw_mixed_width_long_shift) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType out_ptr = poly_dtype_ptr(POLY_INT64, 1, POLY_ADDR_GLOBAL);
  PolyDType in_ptr = poly_dtype_ptr(POLY_INT32, 1, POLY_ADDR_GLOBAL);
  PolyUOp *out = poly_uop0(ctx, POLY_OP_PARAM, out_ptr, poly_arg_int(0));
  PolyUOp *in = poly_uop0(ctx, POLY_OP_PARAM, in_ptr, poly_arg_int(1));
  PolyUOp *shift = poly_uop0(ctx, POLY_OP_PARAM, in_ptr, poly_arg_int(2));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(0));
  PolyUOp *value = poly_uop2(
      ctx, POLY_OP_SHL, POLY_INT64,
      poly_uop1(
          ctx, POLY_OP_LOAD, POLY_INT32,
          poly_uop2(ctx, POLY_OP_INDEX, in_ptr, in, zero, poly_arg_none()), poly_arg_none()
      ),
      poly_uop1(
          ctx, POLY_OP_LOAD, POLY_INT32,
          poly_uop2(ctx, POLY_OP_INDEX, in_ptr, shift, zero, poly_arg_none()), poly_arg_none()
      ),
      poly_arg_none()
  );
  PolyUOp *sink = poly_sink1(
      ctx,
      poly_uop2(
          ctx, POLY_OP_STORE, POLY_VOID,
          poly_uop2(ctx, POLY_OP_INDEX, out_ptr, out, zero, poly_arg_none()), value, poly_arg_none()
      )
  );
  PolyUOp *rewritten = poly_rewrite_webgpu(ctx, sink);
  ASSERT_NOT_NULL(rewritten);

  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  ASSERT_NOT_NULL(topo);
  for (int i = 0; i < n_topo; i++) {
    PolyDType scalar = poly_dtype_scalar(topo[i]->dtype);
    ASSERT_TRUE(!poly_dtype_is_int(scalar) || scalar.bitsize != 64);
  }

  int n_lin = 0;
  PolyUOp **lin = poly_linearize_rewritten(ctx, rewritten, &n_lin);
  ASSERT_NOT_NULL(lin);
  uint32_t out_words[2] = {0, 0};
  int32_t input = 7, shift_value = 2;
  void *args[3] = {out_words, &input, &shift_value};
  ASSERT_INT_EQ(poly_interp_eval(lin, n_lin, args, 3), 0);
  ASSERT_TRUE(((uint64_t)out_words[0] | ((uint64_t)out_words[1] << 32)) == UINT64_C(28));

  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, webgpu_long_decomp_preserves_public_tag_metadata) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr = poly_dtype_ptr(POLY_UINT64, 1, POLY_ADDR_GLOBAL);
  PolyUOp *out = poly_uop0(ctx, POLY_OP_PARAM, ptr, poly_arg_int(0));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(0));
  PolyUOp *idx = poly_uop2(ctx, POLY_OP_INDEX, ptr, out, zero, poly_arg_none());
  int32_t public_tag = (int32_t)UINT32_C(0xa5000000);
  PolyUOp *value = poly_uop_tagged_arg(
      ctx, POLY_OP_CONST, POLY_UINT64, NULL, 0, poly_arg_int(INT64_C(0x1122334455667788)),
      public_tag, poly_arg_str("public-long-tag")
  );
  PolyUOp *sink =
      poly_sink1(ctx, poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx, value, poly_arg_none()));
  PolyUOp *rewritten = poly_rewrite_webgpu(ctx, sink);
  ASSERT_NOT_NULL(rewritten);

  int n_topo = 0, stores = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  ASSERT_NOT_NULL(topo);
  for (int i = 0; i < n_topo; i++) {
    PolyDType scalar = poly_dtype_scalar(topo[i]->dtype);
    ASSERT_TRUE(!poly_dtype_is_int(scalar) || scalar.bitsize != 64);
    if (topo[i]->op != POLY_OP_STORE) continue;
    stores++;
    ASSERT_TRUE(topo[i]->n_src >= 2);
    ASSERT_INT_EQ(topo[i]->src[1]->tag, public_tag);
    ASSERT_INT_EQ(topo[i]->src[1]->tag_arg.kind, POLY_ARG_STRING);
    ASSERT_STR_EQ(topo[i]->src[1]->tag_arg.str, "public-long-tag");
  }
  ASSERT_INT_EQ(stores, 2);

  int n_lin = 0;
  PolyUOp **lin = poly_linearize_rewritten(ctx, rewritten, &n_lin);
  ASSERT_NOT_NULL(lin);
  uint32_t out_words[2] = {0, 0};
  void *args[1] = {out_words};
  ASSERT_INT_EQ(poly_interp_eval(lin, n_lin, args, 1), 0);
  uint64_t got = (uint64_t)out_words[0] | ((uint64_t)out_words[1] << 32);
  ASSERT_TRUE(got == UINT64_C(0x1122334455667788));

  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, renderer_caps_redecompose_post_transcendental_int64) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr = poly_dtype_ptr(POLY_FLOAT32, 1, POLY_ADDR_GLOBAL);
  PolyUOp *out = poly_uop0(ctx, POLY_OP_PARAM, ptr, poly_arg_int(0));
  PolyUOp *in = poly_uop0(ctx, POLY_OP_PARAM, ptr, poly_arg_int(1));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(0));
  PolyUOp *oidx = poly_uop2(ctx, POLY_OP_INDEX, ptr, out, zero, poly_arg_none());
  PolyUOp *iidx = poly_uop2(ctx, POLY_OP_INDEX, ptr, in, zero, poly_arg_none());
  PolyUOp *value = poly_uop1(
      ctx, POLY_OP_SIN, POLY_FLOAT32,
      poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, iidx, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *sink =
      poly_sink1(ctx, poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, oidx, value, poly_arg_none()));
  PolyRewriteOpts opts = {
      .optimize = false,
      .devectorize = 0,
      .caps = poly_c_renderer_caps(),
      .device = POLY_DEVICE_CPU,
      .opt_policy = POLY_OPT_HEURISTIC,
  };
  opts.caps.has_int64 = false;
  opts.caps.has_sin = false;
  PolyUOp *rewritten = poly_full_rewrite_to_sink_ex(ctx, sink, opts);
  ASSERT_NOT_NULL(rewritten);

  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  ASSERT_NOT_NULL(topo);
  ASSERT_INT_EQ(count_lin_ops(topo, n_topo, POLY_OP_SIN), 0);
  for (int i = 0; i < n_topo; i++) {
    PolyDType scalar = poly_dtype_scalar(topo[i]->dtype);
    ASSERT_TRUE(!poly_dtype_is_int(scalar) || scalar.bitsize != 64);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, c_renderer_matches_pinned_clang_transcendental_caps) {
  /* Pinned ClangRenderer removes EXP2, LOG2, and SIN from code_for_op so
   * codegen applies the shared transcendental decompositions
   * (tinygrad/renderer/cstyle.py:246-269). */
  PolyRendererCaps caps = poly_c_renderer_caps();
  ASSERT_FALSE(caps.has_exp2);
  ASSERT_FALSE(caps.has_log2);
  ASSERT_FALSE(caps.has_sin);
  PASS();
}

TEST(codegen, late_fdiv_rewrites_follow_renderer_capability) {
  /* Pinned tinygrad uop/decompositions.py:500-503 gates both
   * RECIPROCAL(x) -> FDIV(1,x) and a*FDIV(1,b) -> FDIV(a,b) on FDIV being
   * present in the renderer op table. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_uop0(ctx, POLY_OP_PARAM, POLY_FLOAT32, poly_arg_int(0));
  PolyUOp *b = poly_uop0(ctx, POLY_OP_PARAM, POLY_FLOAT32, poly_arg_int(1));
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));
  PolyUOp *div = poly_uop2(ctx, POLY_OP_FDIV, POLY_FLOAT32, one, b, poly_arg_none());
  PolyUOp *raw = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, a, div, poly_arg_none());

  PolyRendererCaps without_fdiv = {0};
  PolyUOp *kept =
      poly_graph_rewrite(ctx, raw, poly_pm_decomp_pass_caps(without_fdiv));
  ASSERT_INT_EQ(kept->op, POLY_OP_MUL);
  ASSERT_INT_EQ(kept->n_src, 2);
  ASSERT_INT_EQ(kept->src[1]->op, POLY_OP_FDIV);

  PolyRendererCaps with_fdiv = {.has_fdiv = true};
  PolyUOp *folded =
      poly_graph_rewrite(ctx, raw, poly_pm_decomp_pass_caps(with_fdiv));
  ASSERT_INT_EQ(folded->op, POLY_OP_FDIV);
  ASSERT_INT_EQ(folded->n_src, 2);
  ASSERT_TRUE(folded->src[0] == a);
  ASSERT_TRUE(folded->src[1] == b);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, transcendental_pow2if_dtype_follows_integer_input) {
  /* Pinned tinygrad uop/decompositions.py:29-32 chooses pow2if's float result
   * from q.dtype: int32 -> float32 and int64 -> float64. Its f64
   * payne_hanek_reduction keeps f64 intermediates, while the int32 residual
   * exponent still intentionally creates a float32 pow2 value. */
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr = poly_dtype_ptr(POLY_FLOAT64, 1, POLY_ADDR_GLOBAL);
  PolyUOp *out = poly_uop0(ctx, POLY_OP_PARAM, ptr, poly_arg_int(0));
  PolyUOp *in = poly_uop0(ctx, POLY_OP_PARAM, ptr, poly_arg_int(1));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(0));
  PolyUOp *oidx = poly_uop2(ctx, POLY_OP_INDEX, ptr, out, zero, poly_arg_none());
  PolyUOp *iidx = poly_uop2(ctx, POLY_OP_INDEX, ptr, in, zero, poly_arg_none());
  PolyUOp *value = poly_uop1(
      ctx, POLY_OP_SIN, POLY_FLOAT64,
      poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT64, iidx, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *sink =
      poly_sink1(ctx, poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, oidx, value, poly_arg_none()));
  PolyRewriteOpts opts = {
      .optimize = false,
      .devectorize = 0,
      .caps = poly_c_renderer_caps(),
      .device = POLY_DEVICE_CPU,
      .opt_policy = POLY_OPT_HEURISTIC,
  };
  opts.caps.has_sin = false;
  PolyUOp *rewritten = poly_full_rewrite_to_sink_ex(ctx, sink, opts);
  ASSERT_NOT_NULL(rewritten);

  int n_topo = 0, i32_to_f32 = 0, u64_to_f64 = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  ASSERT_NOT_NULL(topo);
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (u->op != POLY_OP_BITCAST || u->n_src != 1) continue;
    PolyDType from = poly_dtype_scalar(u->src[0]->dtype);
    PolyDType to = poly_dtype_scalar(u->dtype);
    ASSERT_INT_EQ(from.bitsize, to.bitsize);
    if (poly_dtype_eq(from, POLY_INT32) && poly_dtype_eq(to, POLY_FLOAT32)) i32_to_f32++;
    if (poly_dtype_eq(from, POLY_UINT64) && poly_dtype_eq(to, POLY_FLOAT64)) u64_to_f64++;
  }
  ASSERT_TRUE(i32_to_f32 > 0);
  ASSERT_TRUE(u64_to_f64 > 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, bf16_transcendental_widens_to_f32_like_tinygrad) {
  /* Pinned tinygrad uop/decompositions.py:get_transcendental_patterns keeps
   * BF16 outside TRANSCENDENTAL_DTYPES and rewrites it as
   * CAST(BF16, EXP2(CAST(F32, input))). BF16 must never enter the binary16
   * mantissa=10/bias=15/int16 xexp2 branch. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *input =
      poly_uop0(ctx, POLY_OP_PARAM, POLY_BFLOAT16, poly_arg_int(0));
  PolyUOp *raw =
      poly_uop1(ctx, POLY_OP_EXP2, POLY_BFLOAT16, input, poly_arg_none());
  PolyUOp *rewritten =
      poly_graph_rewrite(ctx, raw, poly_pm_transcendental_pass());
  ASSERT_NOT_NULL(rewritten);
  ASSERT_EQ(rewritten->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(rewritten->dtype, POLY_BFLOAT16));
  ASSERT_INT_EQ(rewritten->n_src, 1);
  ASSERT_TRUE(poly_dtype_eq(rewritten->src[0]->dtype, POLY_FLOAT32));

  int n_topo = 0;
  int raw_exp2 = 0, bf16_nodes = 0, bf16_to_f32 = 0, f32_to_bf16 = 0;
  int f32_bitcasts = 0, bf16_bitcasts = 0;
  int floordiv_nodes = 0, cdiv_nodes = 0, cmod_nodes = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, rewritten, &n_topo);
  ASSERT_NOT_NULL(topo);
  ASSERT_INT_EQ(n_topo, 64);
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (u->op == POLY_OP_EXP2) raw_exp2++;
    if (u->op == POLY_OP_FLOORDIV) floordiv_nodes++;
    if (u->op == POLY_OP_CDIV) cdiv_nodes++;
    if (u->op == POLY_OP_CMOD) cmod_nodes++;
    if (poly_dtype_eq(u->dtype, POLY_BFLOAT16)) bf16_nodes++;
    if (u->op == POLY_OP_CAST && u->n_src == 1 &&
        poly_dtype_eq(u->src[0]->dtype, POLY_BFLOAT16) &&
        poly_dtype_eq(u->dtype, POLY_FLOAT32))
      bf16_to_f32++;
    if (u->op == POLY_OP_CAST && u->n_src == 1 &&
        poly_dtype_eq(u->src[0]->dtype, POLY_FLOAT32) &&
        poly_dtype_eq(u->dtype, POLY_BFLOAT16))
      f32_to_bf16++;
    if (u->op == POLY_OP_BITCAST && poly_dtype_eq(u->dtype, POLY_FLOAT32))
      f32_bitcasts++;
    if (u->op == POLY_OP_BITCAST && poly_dtype_eq(u->dtype, POLY_BFLOAT16))
      bf16_bitcasts++;
  }
  ASSERT_INT_EQ(raw_exp2, 0);
  ASSERT_INT_EQ(bf16_nodes, 2);
  ASSERT_INT_EQ(bf16_to_f32, 1);
  ASSERT_INT_EQ(f32_to_bf16, 1);
  ASSERT_INT_EQ(f32_bitcasts, 2);
  ASSERT_INT_EQ(bf16_bitcasts, 0);
  ASSERT_INT_EQ(floordiv_nodes, 1);
  ASSERT_INT_EQ(cdiv_nodes, 0);
  ASSERT_INT_EQ(cmod_nodes, 0);
  free(topo);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, bf16_scalar_and_vector_transcendentals_share_f32_widening) {
  /* Pinned tinygrad uop/decompositions.py:get_transcendental_patterns applies
   * the same BF16 -> float32 -> BF16 wrapper to EXP2, LOG2, and SIN while
   * preserving vector lanes. */
  PolyCtx *ctx = poly_ctx_new();
  PolyOps ops[] = {POLY_OP_EXP2, POLY_OP_LOG2, POLY_OP_SIN};
  PolyDType dts[] = {POLY_BFLOAT16, poly_dtype_vec(POLY_BFLOAT16, 2)};
  for (int d = 0; d < 2; d++) {
    PolyDType f32 =
        dts[d].count > 1 ? poly_dtype_vec(POLY_FLOAT32, dts[d].count) : POLY_FLOAT32;
    for (int o = 0; o < 3; o++) {
      PolyUOp *input =
          poly_uop0(ctx, POLY_OP_PARAM, dts[d], poly_arg_int(10 + d * 3 + o));
      PolyUOp *raw = poly_uop1(ctx, ops[o], dts[d], input, poly_arg_none());
      PolyUOp *rewritten =
          poly_graph_rewrite(ctx, raw, poly_pm_transcendental_pass());
      ASSERT_NOT_NULL(rewritten);
      ASSERT_EQ(rewritten->op, POLY_OP_CAST);
      ASSERT_TRUE(poly_dtype_eq(rewritten->dtype, dts[d]));
      ASSERT_INT_EQ(rewritten->n_src, 1);
      ASSERT_TRUE(poly_dtype_eq(rewritten->src[0]->dtype, f32));

      int n_topo = 0, raw_op = 0, bf16_bitcasts = 0;
      int in_casts = 0, out_casts = 0, scalar_controls = 0;
      int f32_poly_first = 0, f64_poly_first = 0;
      PolyUOp **topo = poly_toposort_alloc(ctx, rewritten, &n_topo);
      ASSERT_NOT_NULL(topo);
      for (int i = 0; i < n_topo; i++) {
        PolyUOp *u = topo[i];
        if (u->op == ops[o]) raw_op++;
        if (u->op == POLY_OP_BITCAST &&
            poly_dtype_eq(poly_dtype_scalar(u->dtype), POLY_BFLOAT16))
          bf16_bitcasts++;
        if (u->op == POLY_OP_CAST && u->n_src == 1 &&
            poly_dtype_eq(u->src[0]->dtype, dts[d]) && poly_dtype_eq(u->dtype, f32))
          in_casts++;
        if (u->op == POLY_OP_CAST && u->n_src == 1 &&
            poly_dtype_eq(u->src[0]->dtype, f32) && poly_dtype_eq(u->dtype, dts[d]))
          out_casts++;
        if (u->op == POLY_OP_CONST && u->arg.kind == POLY_ARG_FLOAT &&
            u->arg.f == 2.6083159809786593541503e-06)
          f32_poly_first++;
        if (u->op == POLY_OP_CONST && u->arg.kind == POLY_ARG_FLOAT &&
            u->arg.f == -7.97255955009037868891952e-18)
          f64_poly_first++;
        if (dts[d].count > 1 &&
            (u->op == POLY_OP_CMPNE || u->op == POLY_OP_WHERE || u->op == POLY_OP_CAST)) {
          if (u->dtype.count != dts[d].count)
            scalar_controls++;
          for (int s = 0; s < u->n_src; s++)
            if (u->src[s]->dtype.count != dts[d].count)
              scalar_controls++;
        }
      }
      ASSERT_INT_EQ(raw_op, 0);
      ASSERT_INT_EQ(bf16_bitcasts, 0);
      ASSERT_INT_EQ(in_casts, 1);
      ASSERT_INT_EQ(out_casts, 1);
      ASSERT_INT_EQ(scalar_controls, 0);
      if (ops[o] == POLY_OP_SIN) {
        ASSERT_INT_EQ(f32_poly_first, 1);
        ASSERT_INT_EQ(f64_poly_first, 0);
        if (dts[d].count == 2) {
          /* Pinned codegen/__init__.py:119 composes symbolic_simple with the
           * transcendental matcher. The exact normalized BF16x2 body is
           * recorded by the paired vector-sine fingerprint probe. */
          ASSERT_INT_EQ(n_topo, 207);
          ASSERT_TRUE(
              normalized_topology_fingerprint(topo, n_topo, rewritten) ==
              UINT64_C(0xc62d202c17fe7fbf)
          );
        }
      }
      poly_toposort_free(topo);
    }
  }
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, float64_vector_sin_preserves_cast_lanes_like_tinygrad) {
  /* Pinned tinygrad decompositions.py:148-149 and UOp.cast
   * (uop/ops.py:513-516) retain two lanes through the Cody-Waite int64
   * quotient. The f64 polynomial is selected from dtype.scalar(). */
  PolyCtx *ctx = poly_ctx_new();
  PolyDType f64x2 = poly_dtype_vec(POLY_FLOAT64, 2);
  PolyDType i64x2 = poly_dtype_vec(POLY_INT64, 2);
  PolyUOp *input = poly_uop0(ctx, POLY_OP_PARAM, f64x2, poly_arg_int(20));
  PolyUOp *raw = poly_uop1(ctx, POLY_OP_SIN, f64x2, input, poly_arg_none());
  PolyUOp *rewritten =
      poly_graph_rewrite(ctx, raw, poly_pm_transcendental_pass());
  ASSERT_NOT_NULL(rewritten);

  int n_topo = 0, raw_sin = 0, f32_poly_first = 0, f64_poly_first = 0;
  int scalar_i64_nodes = 0, vector_i64_nodes = 0, bad_cast_lanes = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, rewritten, &n_topo);
  ASSERT_NOT_NULL(topo);
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (u->op == POLY_OP_SIN) raw_sin++;
    if (poly_dtype_eq(u->dtype, POLY_INT64)) scalar_i64_nodes++;
    if (poly_dtype_eq(u->dtype, i64x2)) vector_i64_nodes++;
    if (u->op == POLY_OP_CONST && u->arg.kind == POLY_ARG_FLOAT &&
        u->arg.f == 2.6083159809786593541503e-06)
      f32_poly_first++;
    if (u->op == POLY_OP_CONST && u->arg.kind == POLY_ARG_FLOAT &&
        u->arg.f == -7.97255955009037868891952e-18)
      f64_poly_first++;
    if (u->op == POLY_OP_CAST &&
        (u->dtype.count != f64x2.count || u->src[0]->dtype.count != f64x2.count))
      bad_cast_lanes++;
  }
  ASSERT_INT_EQ(raw_sin, 0);
  ASSERT_INT_EQ(f32_poly_first, 0);
  ASSERT_INT_EQ(f64_poly_first, 1);
  ASSERT_INT_EQ(scalar_i64_nodes, 0);
  ASSERT_TRUE(vector_i64_nodes > 0);
  ASSERT_INT_EQ(bad_cast_lanes, 0);
  ASSERT_INT_EQ(n_topo, 244);
  ASSERT_TRUE(
      normalized_topology_fingerprint(topo, n_topo, rewritten) ==
      UINT64_C(0x6bc17d53178d9e43)
  );
  poly_toposort_free(topo);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, partial_reshape_index_matches_tinygrad_mop) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* Pinned schedule/rangeify.py:63-77 collapses
   * PARAM(3).RESHAPE(1,3).INDEX(0) to PARAM(3) because the unindexed output
   * suffix exactly matches the input shape. */
  PolyUOp *like = poly_reshape(ctx, poly_buffer_f32(ctx, 3), (int64_t[]){1, 3}, 2);
  PolyUOp *input = poly_uop_placeholder_like(ctx, like, 1);
  PolyUOp *output = poly_uop_placeholder_like(ctx, poly_buffer_f32(ctx, 1), 0);
  PolyUOp *zero = poly_const_int(ctx, 0);
  PolyUOp *two = poly_const_int(ctx, 2);
  PolyUOp *partial = poly_uop_index(ctx, input, &zero, 1, 1);
  PolyUOp *value = poly_uop_index(ctx, partial, &two, 1, 0);
  PolyUOp *address = poly_uop_index(ctx, output, &zero, 1, 1);
  PolyUOp *store = poly_uop_store(ctx, address, value);
  PolyUOp *sink = poly_uop_sink(ctx, &store, 1);
  ASSERT_NOT_NULL(sink);

  PolyRewriteOpts opts = {
      .optimize = false,
      .devectorize = 0,
      .caps = poly_c_renderer_caps(),
  };
  PolyUOp *rewritten = poly_full_rewrite_to_sink_ex(ctx, sink, opts);
  ASSERT_NOT_NULL(rewritten);
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  ASSERT_NOT_NULL(topo);
  ASSERT_INT_EQ(count_lin_ops(topo, n_topo, POLY_OP_RESHAPE), 0);

  int n_linear = 0;
  PolyUOp **linear = poly_linearize_ex(ctx, sink, opts, &n_linear);
  ASSERT_NOT_NULL(linear);
  char *source = poly_render_c(linear, n_linear, "partial_reshape_index");
  ASSERT_NOT_NULL(source);
  ASSERT_NOT_NULL(strstr(source, "data1+2"));
  PolyProgram *program = poly_compile_c(source, "partial_reshape_index");
  ASSERT_NOT_NULL(program);

  float out = 0.0f;
  float in[3] = {11.0f, 22.0f, 33.0f};
  void *args[2] = {&out, in};
  poly_program_call(program, args, 2);
  ASSERT_FLOAT_EQ(out, 33.0f, 1e-6);

  poly_program_destroy(program);
  free(source);
  free(linear);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, partial_reshape_index_maps_nonzero_input_prefix) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* Pinned schedule/rangeify.py:68-77 maps
   * (2,3,4)->(6,4)->INDEX(5) to source.INDEX(1,2), retaining shape (4). */
  PolyUOp *like = poly_reshape(ctx, poly_buffer_f32(ctx, 24), (int64_t[]){2, 3, 4}, 3);
  PolyUOp *input = poly_uop_placeholder_like(ctx, like, 1);
  PolyUOp *output = poly_uop_placeholder_like(ctx, poly_buffer_f32(ctx, 1), 0);
  PolyUOp *reshape = poly_reshape(ctx, input, (int64_t[]){6, 4}, 2);
  PolyUOp *five = poly_const_int(ctx, 5);
  PolyUOp *partial = poly_uop_index(ctx, reshape, &five, 1, 1);
  PolyUOp *zero = poly_const_int(ctx, 0);
  PolyUOp *value = poly_uop_index(ctx, partial, &zero, 1, 0);
  PolyUOp *address = poly_uop_index(ctx, output, &zero, 1, 1);
  PolyUOp *store = poly_uop_store(ctx, address, value);
  PolyUOp *sink = poly_uop_sink(ctx, &store, 1);
  ASSERT_NOT_NULL(sink);

  PolyRewriteOpts opts = {
      .optimize = false,
      .devectorize = 0,
      .caps = poly_c_renderer_caps(),
  };
  PolyUOp *rewritten = poly_full_rewrite_to_sink_ex(ctx, sink, opts);
  ASSERT_NOT_NULL(rewritten);
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  ASSERT_NOT_NULL(topo);
  /* Pinned codegen/__init__.py:61 runs pm_mops+pm_syntactic_sugar together:
   * partial (1,2), then scalar (0), becomes flat PARAM index 20. */
  ASSERT_FALSE(poly_uop_reachable(ctx, rewritten, reshape));
  ASSERT_INT_EQ(count_lin_ops(topo, n_topo, POLY_OP_RESHAPE), 0);
  ASSERT_INT_EQ(count_lin_ops(topo, n_topo, POLY_OP_INDEX), 2);
  PolyUOp *read_load = NULL;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op != POLY_OP_LOAD) continue;
    ASSERT_TRUE(read_load == NULL);
    read_load = topo[i];
  }
  ASSERT_NOT_NULL(read_load);
  ASSERT_INT_EQ(read_load->n_src, 1);
  PolyUOp *read_index = read_load->src[0];
  ASSERT_NOT_NULL(read_index);
  ASSERT_INT_EQ(read_index->op, POLY_OP_INDEX);
  ASSERT_INT_EQ(read_index->n_src, 2);
  ASSERT_INT_EQ(read_index->src[0]->op, POLY_OP_PARAM);
  ASSERT_INT_EQ(read_index->src[0]->arg.kind, POLY_ARG_INT);
  ASSERT_INT_EQ(read_index->src[0]->arg.i, 1);
  ASSERT_INT_EQ(read_index->src[1]->op, POLY_OP_CONST);
  ASSERT_INT_EQ(read_index->src[1]->arg.i, 20);

  int n_linear = 0;
  PolyUOp **linear = poly_linearize_ex(ctx, sink, opts, &n_linear);
  ASSERT_NOT_NULL(linear);
  char *source = poly_render_c(linear, n_linear, "partial_reshape_index_prefix");
  ASSERT_NOT_NULL(source);
  ASSERT_NOT_NULL(strstr(source, "data1+20"));
  PolyProgram *program = poly_compile_c(source, "partial_reshape_index_prefix");
  ASSERT_NOT_NULL(program);

  float out = 0.0f;
  float in[24];
  for (int i = 0; i < 24; i++)
    in[i] = (float)(100 + i);
  void *args[2] = {&out, in};
  poly_program_call(program, args, 2);
  ASSERT_FLOAT_EQ(out, 120.0f, 1e-6);

  poly_program_destroy(program);
  free(source);
  free(linear);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, validates_index_coordinates_before_pointer_concat) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* Pinned tinygrad/uop/spec.py:73-77 rejects a direct bool INDEX coordinate
   * before codegen preprocess, while integer WHERE(valid, idx, Invalid)
   * remains a valid coordinate for pm_syntactic_sugar. */
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, 16, POLY_ADDR_GLOBAL);
  PolyUOp *input = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));
  PolyUOp *output = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(1));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(0));
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(1));
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(2));
  PolyUOp *three = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(3));
  PolyUOp *gate = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, zero, one, poly_arg_none());
  ASSERT_NOT_NULL(gate);

  PolyUOp *bad_inner_indices[] = {zero, gate};
  PolyUOp *bad_inner = poly_uop_index(ctx, input, bad_inner_indices, 2, 1);
  ASSERT_NOT_NULL(bad_inner); /* Construction matches pinned UOp.index. */
  PolyUOp *bad_outer = poly_uop_index(ctx, bad_inner, &one, 1, 0);
  PolyUOp *bad_value = poly_uop_load(ctx, bad_outer);
  PolyUOp *out_index = poly_uop_index(ctx, output, &zero, 1, 1);
  PolyUOp *bad_store = poly_uop_store(ctx, out_index, bad_value);
  PolyUOp *bad_sink = poly_uop_sink(ctx, &bad_store, 1);
  ASSERT_NOT_NULL(bad_sink);
  ASSERT_FALSE(poly_validate_kernel_graph(ctx, bad_sink));

  PolyRewriteOpts opts = {
      .optimize = false,
      .devectorize = 0,
      .caps = poly_c_renderer_caps(),
  };
  ASSERT_TRUE(poly_full_rewrite_to_sink_ex(ctx, bad_sink, opts) == NULL);

  PolyUOp *invalid = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_invalid());
  PolyUOp *valid_coord =
      poly_uop3(ctx, POLY_OP_WHERE, POLY_INDEX, gate, two, invalid, poly_arg_none());
  PolyUOp *good_inner = poly_uop_index(ctx, input, &valid_coord, 1, 1);
  PolyUOp *good_outer = poly_uop_index(ctx, good_inner, &three, 1, 0);
  PolyUOp *good_value = poly_uop_load(ctx, good_outer);
  PolyUOp *good_store = poly_uop_store(ctx, out_index, good_value);
  PolyUOp *good_sink = poly_uop_sink(ctx, &good_store, 1);
  ASSERT_TRUE(poly_validate_kernel_graph(ctx, good_sink));
  PolyUOp *rewritten = poly_full_rewrite_to_sink_ex(ctx, good_sink, opts);
  ASSERT_NOT_NULL(rewritten);

  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  ASSERT_NOT_NULL(topo);
  PolyUOp *read_index = NULL;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op == POLY_OP_LOAD && topo[i]->n_src >= 1 && topo[i]->src[0]->op == POLY_OP_INDEX)
      read_index = topo[i]->src[0];
    if (topo[i]->op != POLY_OP_INDEX) continue;
    for (int j = 1; j < topo[i]->n_src; j++)
      ASSERT_TRUE(poly_dtype_is_int(topo[i]->src[j]->dtype));
  }
  /* Literal pm_syntactic_sugar concat preserves both coordinates in order.
   * The adjacent partial_reshape_index_maps_nonzero_input_prefix test executes
   * the same rule with a nontrivial multidimensional offset. */
  ASSERT_NOT_NULL(read_index);
  ASSERT_INT_EQ(read_index->n_src, 3);
  ASSERT_INT_EQ(read_index->src[0]->op, POLY_OP_PARAM);
  ASSERT_INT_EQ(read_index->src[0]->arg.kind, POLY_ARG_INT);
  ASSERT_INT_EQ(read_index->src[0]->arg.i, 0);
  ASSERT_INT_EQ(read_index->src[1]->op, POLY_OP_CONST);
  ASSERT_INT_EQ(read_index->src[1]->arg.kind, POLY_ARG_INT);
  ASSERT_INT_EQ(read_index->src[1]->arg.i, 2);
  ASSERT_INT_EQ(read_index->src[2]->op, POLY_OP_CONST);
  ASSERT_INT_EQ(read_index->src[2]->arg.kind, POLY_ARG_INT);
  ASSERT_INT_EQ(read_index->src[2]->arg.i, 3);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, render_wgsl_unary) {
  /* b[i] = -a[i] for i in 0..5 */
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);

  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(1));

  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(6));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));

  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p0, range, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p1, range, poly_arg_none());

  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx0, poly_arg_none());
  PolyUOp *neg = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, load, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx1, neg, poly_arg_none());

  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  int n;
  PolyUOp **lin = poly_linearize(ctx, sink, &n);
  char *src = poly_render_wgsl(lin, n, "vecneg");

  /* only 2 bindings */
  ASSERT_NOT_NULL(strstr(src, "data0: array<f32>"));
  ASSERT_NOT_NULL(strstr(src, "data1: array<f32>"));
  ASSERT_NOT_NULL(strstr(src, "fn vecneg("));
  /* NEG renders as (-val) */
  ASSERT_NOT_NULL(strstr(src, "(-val0)"));

  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, render_wgsl_where) {
  /* c[i] = cond ? a[i] : b[i], where cond = (a[i] < 5.0) */
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);

  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(1));
  PolyUOp *p2 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(2));

  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(4));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));

  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p0, range, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p1, range, poly_arg_none());
  PolyUOp *idx2 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p2, range, poly_arg_none());

  PolyUOp *load0 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx0, poly_arg_none());
  PolyUOp *load1 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx1, poly_arg_none());

  PolyUOp *five = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(5.0));
  PolyUOp *cond = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, load0, five, poly_arg_none());

  PolyUOp *where_src[3] = {cond, load0, load1};
  PolyUOp *where = poly_uop(ctx, POLY_OP_WHERE, POLY_FLOAT32, where_src, 3, poly_arg_none());

  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx2, where, poly_arg_none());

  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  int n;
  PolyUOp **lin = poly_linearize(ctx, sink, &n);
  char *src = poly_render_wgsl(lin, n, "vecwhere");

  /* WHERE maps to select(false_val, true_val, cond) */
  ASSERT_NOT_NULL(strstr(src, "select("));

  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, render_wgsl_uint32_ops) {
  /* out[i] = where(a[i] < b[i], (a[i] >> 1) + (a[i] % 31), a[i] // b[i]) */
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_u32 = poly_dtype_ptr(POLY_UINT32, -1, POLY_ADDR_GLOBAL);

  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_u32, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_u32, poly_arg_int(1));
  PolyUOp *p2 = poly_uop0(ctx, POLY_OP_PARAM, ptr_u32, poly_arg_int(2));

  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(8));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));

  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_u32, p0, range, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_u32, p1, range, poly_arg_none());
  PolyUOp *idx2 = poly_uop2(ctx, POLY_OP_INDEX, ptr_u32, p2, range, poly_arg_none());

  PolyUOp *la = poly_uop1(ctx, POLY_OP_LOAD, POLY_UINT32, idx0, poly_arg_none());
  PolyUOp *lb = poly_uop1(ctx, POLY_OP_LOAD, POLY_UINT32, idx1, poly_arg_none());
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT32, poly_arg_int(1));
  PolyUOp *thirty_one = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT32, poly_arg_int(31));

  PolyUOp *cond = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, la, lb, poly_arg_none());
  PolyUOp *rhs = poly_uop2(
      ctx, POLY_OP_ADD, POLY_UINT32,
      poly_uop2(ctx, POLY_OP_SHR, POLY_UINT32, la, one, poly_arg_none()),
      poly_uop2(ctx, POLY_OP_MOD, POLY_UINT32, la, thirty_one, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *lhs = poly_uop2(ctx, POLY_OP_IDIV, POLY_UINT32, la, lb, poly_arg_none());
  PolyUOp *sel_src[3] = {cond, rhs, lhs};
  PolyUOp *sel = poly_uop(ctx, POLY_OP_WHERE, POLY_UINT32, sel_src, 3, poly_arg_none());

  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx2, sel, poly_arg_none());
  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  int n;
  PolyUOp **lin = poly_linearize(ctx, sink, &n);
  char *src = poly_render_wgsl(lin, n, "vecu32");

  ASSERT_NOT_NULL(strstr(src, "array<u32>"));
  ASSERT_NOT_NULL(strstr(src, "var val0: u32"));
  ASSERT_NOT_NULL(strstr(src, "31u"));
  ASSERT_NOT_NULL(strstr(src, ">>"));
  ASSERT_NOT_NULL(strstr(src, "%"));
  ASSERT_NOT_NULL(strstr(src, "select("));

  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, render_wgsl_uint8_storage_is_packed_like_tinygrad) {
  /* out[i] = in[i] for uint8. WGSL has no byte-addressable storage buffer,
   * so tinygrad packs byte/short storage into atomic<u32> lanes. */
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_u8 = poly_dtype_ptr(POLY_UINT8, -1, POLY_ADDR_GLOBAL);

  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_u8, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_u8, poly_arg_int(1));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(5));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_u8, p0, range, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_u8, p1, range, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_UINT8, idx0, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx1, load, poly_arg_none());
  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  int n;
  PolyUOp **lin = poly_linearize(ctx, sink, &n);
  char *src = poly_render_wgsl(lin, n, "copy_u8");

  ASSERT_NOT_NULL(strstr(src, "data0: array<atomic<u32>>"));
  ASSERT_NOT_NULL(strstr(src, "data1: array<atomic<u32>>"));
  ASSERT_NOT_NULL(strstr(src, "atomicLoad(&data0[(ridx0/4)]"));
  ASSERT_NOT_NULL(strstr(src, "atomicAnd(&data1[(ridx0/4)]"));
  ASSERT_NOT_NULL(strstr(src, "atomicAdd(&data1[(ridx0/4)]"));
  ASSERT_TRUE(strstr(src, "data1[ridx0] =") == NULL);

  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, render_wgsl_bool_storage_is_packed_like_tinygrad) {
  /* WGSL permits scalar bool values but forbids bool in storage buffers.
   * tinygrad's WGSLRenderer packs bool output storage into atomic<u32> byte
   * lanes, while keeping the comparison itself as a scalar bool expression. */
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_bool = poly_dtype_ptr(POLY_BOOL, -1, POLY_ADDR_GLOBAL);
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);

  PolyUOp *out = poly_uop0(ctx, POLY_OP_PARAM, ptr_bool, poly_arg_int(0));
  PolyUOp *a = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(1));
  PolyUOp *b = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(2));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(3));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *idx_out = poly_uop2(ctx, POLY_OP_INDEX, ptr_bool, out, range, poly_arg_none());
  PolyUOp *idx_a = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, a, range, poly_arg_none());
  PolyUOp *idx_b = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, b, range, poly_arg_none());
  PolyUOp *load_a = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx_a, poly_arg_none());
  PolyUOp *load_b = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx_b, poly_arg_none());
  PolyUOp *eq = poly_uop2(ctx, POLY_OP_CMPEQ, POLY_BOOL, load_a, load_b, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx_out, eq, poly_arg_none());
  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  int n;
  PolyUOp **lin = poly_linearize(ctx, sink, &n);
  char *src = poly_render_wgsl(lin, n, "eq_bool");

  ASSERT_NOT_NULL(strstr(src, "data0: array<atomic<u32>>"));
  ASSERT_TRUE(strstr(src, "array<bool>") == NULL);
  ASSERT_NOT_NULL(strstr(src, "var alu0: bool"));
  ASSERT_NOT_NULL(strstr(src, "atomicAdd(&data0[(ridx0/4)]"));
  ASSERT_NOT_NULL(strstr(src, "select(0u, 1u, alu0)"));

  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, render_wgsl_reduce) {
  /* out[0] = sum(a[0..9]) — reduce with accumulator */
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);

  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(1));

  /* accumulator */
  PolyUOp *acc = poly_uop0(ctx, POLY_OP_DEFINE_LOCAL, POLY_FLOAT32, poly_arg_float(0.0));

  /* inner loop */
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(10));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));

  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p0, range, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx0, poly_arg_none());

  /* acc += load */
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, acc, load, poly_arg_none());
  PolyUOp *store_acc = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, acc, add, poly_arg_none());

  PolyUOp *end_src[2] = {store_acc, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());

  /* store result */
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p1, zero, poly_arg_none());
  PolyUOp *store_out_src[3] = {idx1, acc, end};
  PolyUOp *store_out = poly_uop(ctx, POLY_OP_STORE, POLY_VOID, store_out_src, 3, poly_arg_none());

  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store_out, poly_arg_none());

  int n_lin;
  PolyUOp **lin = poly_linearize(ctx, sink, &n_lin);
  char *src = poly_render_wgsl(lin, n_lin, "reduce_sum");

  /* accumulator declaration hoisted to function scope, init in body */
  ASSERT_NOT_NULL(strstr(src, "var acc0: f32;"));
  ASSERT_NOT_NULL(strstr(src, "acc0 = 0.0"));
  /* loop present */
  ASSERT_NOT_NULL(strstr(src, "for (var ridx0: i32 = 0;"));
  /* accumulator store (not array write) */
  ASSERT_NOT_NULL(strstr(src, "acc0 = "));
  /* output buffer store */
  ASSERT_NOT_NULL(strstr(src, "data1[0]"));

  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, render_wgsl_define_var) {
  /* out[i] = data[i] + cast(N) for i in 0..N, with N as DEFINE_VAR.
   * Verified against tinygrad: PARAM gets var<storage,read_write>,
   * DEFINE_VAR gets var<uniform>, sequential binding indices. */
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);

  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(1));
  PolyUOp *N = poly_uop0(ctx, POLY_OP_DEFINE_VAR, POLY_INT32, poly_arg_define_var("N", 1, 16));

  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, N, poly_arg_int(0));

  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p0, range, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p1, range, poly_arg_none());

  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx0, poly_arg_none());
  PolyUOp *cast_n = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, N, poly_arg_none());
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, load, cast_n, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx1, add, poly_arg_none());

  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  int n_lin;
  PolyUOp **lin = poly_linearize(ctx, sink, &n_lin);
  char *src = poly_render_wgsl(lin, n_lin, "var_kernel");

  /* binding(0) = INFINITY uniform (always) */
  ASSERT_NOT_NULL(strstr(src, "@group(0) @binding(0)\nvar<uniform> INFINITY"));

  /* binding(1) = data0 storage buffer */
  ASSERT_NOT_NULL(strstr(src, "@group(0) @binding(1)\nvar<storage,read_write> data0: array<f32>"));

  /* binding(2) = data1 storage buffer */
  ASSERT_NOT_NULL(strstr(src, "@group(0) @binding(2)\nvar<storage,read_write> data1: array<f32>"));

  /* binding(3) = N scalar uniform (DEFINE_VAR) */
  ASSERT_NOT_NULL(strstr(src, "@group(0) @binding(3)\nvar<uniform> N: i32"));

  /* the variable name N appears in the kernel body */
  ASSERT_NOT_NULL(strstr(src, "f32(N)"));

  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, render_wgsl_param_bindings_follow_encounter_order) {
  /* Match tinygrad WGSL binding assignment:
   * bindings are sequential in PARAM/DEFINE_VAR encounter order, not PARAM.arg.
   * Sparse PARAM ids used to leak into @binding(N), which mismatched runtime binding order. */
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);

  PolyUOp *p7 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(7));
  PolyUOp *p2 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(2));
  PolyUOp *p9 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(9));

  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(8));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));

  PolyUOp *idx7 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p7, range, poly_arg_none());
  PolyUOp *idx2 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p2, range, poly_arg_none());
  PolyUOp *idx9 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p9, range, poly_arg_none());

  PolyUOp *lhs = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx2, poly_arg_none());
  PolyUOp *rhs = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx9, poly_arg_none());
  PolyUOp *sum = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, lhs, rhs, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx7, sum, poly_arg_none());

  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  int n_lin = 0;
  PolyUOp **lin = poly_linearize(ctx, sink, &n_lin);
  char *src = poly_render_wgsl(lin, n_lin, "param_order");

  ASSERT_NOT_NULL(strstr(src, "@group(0) @binding(1)\nvar<storage,read_write>"));
  ASSERT_NOT_NULL(strstr(src, "@group(0) @binding(2)\nvar<storage,read_write>"));
  ASSERT_NOT_NULL(strstr(src, "@group(0) @binding(3)\nvar<storage,read_write>"));
  ASSERT_NOT_NULL(strstr(src, "var<storage,read_write> data7: array<f32>;"));
  ASSERT_NOT_NULL(strstr(src, "var<storage,read_write> data2: array<f32>;"));
  ASSERT_NOT_NULL(strstr(src, "var<storage,read_write> data9: array<f32>;"));
  ASSERT_TRUE(
      strstr(src, "@group(0) @binding(8)\nvar<storage,read_write> data7: array<f32>") == NULL
  );
  ASSERT_TRUE(
      strstr(src, "@group(0) @binding(10)\nvar<storage,read_write> data9: array<f32>") == NULL
  );

  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, render_wgsl_bindings_grow_past_old_fixed_cap) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  enum { N_BINDINGS = 70 };
  PolyUOp **uops = calloc(N_BINDINGS, sizeof(PolyUOp *));
  ASSERT_NOT_NULL(uops);

  for (int i = 0; i < N_BINDINGS; i++)
    uops[i] = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(i));

  char *src = poly_render_wgsl(uops, N_BINDINGS, "many_bindings");
  ASSERT_NOT_NULL(src);

  ASSERT_NOT_NULL(strstr(src, "@group(0) @binding(65)\nvar<storage,read_write> data64"));
  ASSERT_NOT_NULL(strstr(src, "@group(0) @binding(70)\nvar<storage,read_write> data69"));

  free(src);
  free(uops);
  poly_ctx_destroy(ctx);
  PASS();
}

/* WebGPU GPU linearizer output tests */

TEST(codegen, linearize_webgpu_vecadd_emits_gpudims) {
  /* Verify GPU linearizer produces SPECIAL ops and correct WGSL builtins */
  VecKernel k = make_vec_binop(POLY_OP_ADD, 1024);
  int n_lin = 0;
  PolyUOp **lin = poly_linearize_webgpu(k.ctx, k.sink, &n_lin);
  ASSERT_NOT_NULL(lin);
  ASSERT_TRUE(n_lin > 0);

  /* GPU linearizer should produce at least gidx0 SPECIAL op */
  ASSERT_TRUE(count_special_named(lin, n_lin, "gidx0") >= 1);

  /* Render to WGSL and verify GPU-specific patterns */
  char *wgsl = poly_render_wgsl(lin, n_lin, "vecadd_gpu");
  ASSERT_NOT_NULL(wgsl);
  ASSERT_NOT_NULL(strstr(wgsl, "i32(gindex."));
  ASSERT_NOT_NULL(strstr(wgsl, "@compute @workgroup_size("));
  /* Builtins: workgroup_id + local_invocation_id */
  ASSERT_NOT_NULL(strstr(wgsl, "workgroup_id"));
  ASSERT_NOT_NULL(strstr(wgsl, "local_invocation_id"));

  free(wgsl);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(codegen, linearize_webgpu_vecadd_splits_large_global_dispatch) {
  /* End-to-end through the WebGPU linearizer: the backend caps must flow into
   * add_gpudims, so oversized logical dispatches produce multiple hardware
   * workgroup_id SPECIALs instead of an illegal x-dimension. */
  VecKernel k = make_vec_binop(POLY_OP_ADD, 16777216);
  int n_lin = 0;
  PolyUOp **lin = poly_linearize_webgpu(k.ctx, k.sink, &n_lin);
  ASSERT_NOT_NULL(lin);

  ASSERT_INT_EQ(count_special_named(lin, n_lin, "gidx0"), 1);
  ASSERT_INT_EQ(count_special_named(lin, n_lin, "gidx1"), 1);
  ASSERT_INT_EQ((int)special_bound_hi_named(lin, n_lin, "gidx0"), 32768);
  ASSERT_INT_EQ((int)special_bound_hi_named(lin, n_lin, "gidx1"), 4);

  char *wgsl = poly_render_wgsl(lin, n_lin, "vecadd_split_gpu");
  ASSERT_NOT_NULL(wgsl);
  ASSERT_NOT_NULL(strstr(wgsl, "i32(gindex.x)"));
  ASSERT_NOT_NULL(strstr(wgsl, "i32(gindex.y)"));

  free(wgsl);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(codegen, add_gpudims_same_axis_ranges_share_special) {
  /* tinygrad gpudims groups ranges by axis tuple, not UOp identity. Distinct
   * RANGE nodes for the same axis, including weakint/int variants, must map to
   * one GPU builtin instead of rendering duplicate gidx/lidx declarations. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *b16_w = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(16));
  PolyUOp *b16_i = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(16));
  PolyUOp *rw =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_INDEX, b16_w, poly_arg_range(0, POLY_AXIS_GLOBAL));
  PolyUOp *ri =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, b16_i, poly_arg_range(0, POLY_AXIS_GLOBAL));
  PolyUOp *srcs[2] = {rw, ri};
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, srcs, 2, poly_arg_str("same_axis"));
  PolyUOp *rewritten = poly_add_gpudims(ctx, sink);

  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  ASSERT_INT_EQ(count_lin_ops(topo, n_topo, POLY_OP_SPECIAL), 1);
  ASSERT_INT_EQ(count_lin_ops(topo, n_topo, POLY_OP_RANGE), 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, add_gpudims_reverse_group_preserves_logical_axis_order) {
  /* Pinned gpudims.py:28-56 implements reverse=True by reconstructing the
   * reversed dimensions first, then reversing the result. For the HLB-shaped
   * (32,2,9,3) global grid this is exactly
   * (gidx2, gidx1, gidx0//3, gidx0%3). */
  PolyCtx *ctx = poly_ctx_new();
  const int64_t bounds[4] = {32, 2, 9, 3};
  PolyUOp *ranges[4];
  for (int i = 0; i < 4; i++) {
    PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(bounds[i]));
    ranges[i] =
        poly_uop1(ctx, POLY_OP_RANGE, POLY_INDEX, bound, poly_arg_range(3 + i, POLY_AXIS_GLOBAL));
  }
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, ranges, 4, poly_arg_str("reverse_group"));
  PolyRendererCaps caps = {.global_max = {INT32_MAX, 65535, 65535}};
  PolyUOp *rewritten = poly_add_gpudims_ex(ctx, sink, caps);

  ASSERT_NOT_NULL(rewritten);
  ASSERT_INT_EQ(rewritten->n_src, 4);
  ASSERT_INT_EQ(rewritten->src[0]->op, POLY_OP_SPECIAL);
  ASSERT_STR_EQ(rewritten->src[0]->arg.str, "gidx2");
  ASSERT_INT_EQ(rewritten->src[1]->op, POLY_OP_SPECIAL);
  ASSERT_STR_EQ(rewritten->src[1]->arg.str, "gidx1");

  ASSERT_INT_EQ(rewritten->src[2]->op, POLY_OP_FLOORDIV);
  ASSERT_INT_EQ(rewritten->src[2]->src[0]->op, POLY_OP_SPECIAL);
  ASSERT_STR_EQ(rewritten->src[2]->src[0]->arg.str, "gidx0");
  ASSERT_INT_EQ(rewritten->src[2]->src[1]->arg.i, 3);

  ASSERT_INT_EQ(rewritten->src[3]->op, POLY_OP_FLOORMOD);
  ASSERT_TRUE(rewritten->src[3]->src[0] == rewritten->src[2]->src[0]);
  ASSERT_INT_EQ(rewritten->src[3]->src[1]->arg.i, 3);

  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  ASSERT_INT_EQ((int)special_bound_hi_named(topo, n_topo, "gidx0"), 27);
  ASSERT_INT_EQ((int)special_bound_hi_named(topo, n_topo, "gidx1"), 2);
  ASSERT_INT_EQ((int)special_bound_hi_named(topo, n_topo, "gidx2"), 32);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, add_gpudims_reverse_group_handles_nonprefix_and_repeated_merges) {
  /* Pinned test/null/test_gpudims.py:73-79 covers reverse grouping when the
   * leftmost pair cannot merge. Keep the ordered-domain origins on that
   * non-prefix contraction as well. */
  PolyCtx *ctx = poly_ctx_new();
  const int64_t nonprefix_bounds[4] = {2, 3, 4, 5};
  PolyUOp *nonprefix_ranges[4];
  for (int i = 0; i < 4; i++) {
    PolyUOp *bound =
        poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(nonprefix_bounds[i]));
    nonprefix_ranges[i] =
        poly_uop1(ctx, POLY_OP_RANGE, POLY_INDEX, bound, poly_arg_range(i, POLY_AXIS_GLOBAL));
  }
  PolyUOp *nonprefix_sink =
      poly_uop(ctx, POLY_OP_SINK, POLY_VOID, nonprefix_ranges, 4, poly_arg_str("nonprefix"));
  PolyRendererCaps nonprefix_caps = {.global_max = {16, 16, 16}};
  PolyUOp *nonprefix = poly_add_gpudims_ex(ctx, nonprefix_sink, nonprefix_caps);
  ASSERT_INT_EQ(nonprefix->src[0]->op, POLY_OP_SPECIAL);
  ASSERT_STR_EQ(nonprefix->src[0]->arg.str, "gidx2");
  ASSERT_INT_EQ(nonprefix->src[1]->op, POLY_OP_FLOORDIV);
  ASSERT_INT_EQ(nonprefix->src[1]->src[0]->op, POLY_OP_SPECIAL);
  ASSERT_STR_EQ(nonprefix->src[1]->src[0]->arg.str, "gidx1");
  ASSERT_INT_EQ(nonprefix->src[1]->src[1]->arg.i, 4);
  ASSERT_INT_EQ(nonprefix->src[2]->op, POLY_OP_FLOORMOD);
  ASSERT_TRUE(nonprefix->src[2]->src[0] == nonprefix->src[1]->src[0]);
  ASSERT_INT_EQ(nonprefix->src[2]->src[1]->arg.i, 4);
  ASSERT_INT_EQ(nonprefix->src[3]->op, POLY_OP_SPECIAL);
  ASSERT_STR_EQ(nonprefix->src[3]->arg.str, "gidx0");
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, nonprefix, &n_topo);
  ASSERT_INT_EQ((int)special_bound_hi_named(topo, n_topo, "gidx0"), 5);
  ASSERT_INT_EQ((int)special_bound_hi_named(topo, n_topo, "gidx1"), 12);
  ASSERT_INT_EQ((int)special_bound_hi_named(topo, n_topo, "gidx2"), 2);
  poly_ctx_destroy(ctx);

  /* Two ordered-domain merges must preserve the complete contraction order,
   * not only the first adjacent pair used by the HLB regression above. */
  ctx = poly_ctx_new();
  const int64_t repeated_bounds[5] = {6, 5, 4, 3, 2};
  PolyUOp *repeated_ranges[5];
  for (int i = 0; i < 5; i++) {
    PolyUOp *bound =
        poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(repeated_bounds[i]));
    repeated_ranges[i] =
        poly_uop1(ctx, POLY_OP_RANGE, POLY_INDEX, bound, poly_arg_range(i, POLY_AXIS_GLOBAL));
  }
  PolyUOp *repeated_sink =
      poly_uop(ctx, POLY_OP_SINK, POLY_VOID, repeated_ranges, 5, poly_arg_str("repeated"));
  PolyRendererCaps repeated_caps = {.global_max = {100, 100, 100}};
  PolyUOp *repeated = poly_add_gpudims_ex(ctx, repeated_sink, repeated_caps);
  ASSERT_INT_EQ(repeated->src[0]->op, POLY_OP_SPECIAL);
  ASSERT_STR_EQ(repeated->src[0]->arg.str, "gidx2");
  ASSERT_INT_EQ(repeated->src[1]->op, POLY_OP_SPECIAL);
  ASSERT_STR_EQ(repeated->src[1]->arg.str, "gidx1");
  ASSERT_INT_EQ(repeated->src[2]->op, POLY_OP_FLOORDIV);
  ASSERT_INT_EQ(repeated->src[2]->src[0]->op, POLY_OP_FLOORDIV);
  ASSERT_INT_EQ(repeated->src[2]->src[0]->src[0]->op, POLY_OP_SPECIAL);
  ASSERT_STR_EQ(repeated->src[2]->src[0]->src[0]->arg.str, "gidx0");
  ASSERT_INT_EQ(repeated->src[2]->src[0]->src[1]->arg.i, 2);
  ASSERT_INT_EQ(repeated->src[2]->src[1]->arg.i, 3);
  ASSERT_INT_EQ(repeated->src[3]->op, POLY_OP_FLOORMOD);
  ASSERT_TRUE(repeated->src[3]->src[0] == repeated->src[2]->src[0]);
  ASSERT_INT_EQ(repeated->src[3]->src[1]->arg.i, 3);
  ASSERT_INT_EQ(repeated->src[4]->op, POLY_OP_FLOORMOD);
  ASSERT_TRUE(repeated->src[4]->src[0] == repeated->src[2]->src[0]->src[0]);
  ASSERT_INT_EQ(repeated->src[4]->src[1]->arg.i, 2);
  n_topo = 0;
  topo = poly_toposort(ctx, repeated, &n_topo);
  ASSERT_INT_EQ((int)special_bound_hi_named(topo, n_topo, "gidx0"), 24);
  ASSERT_INT_EQ((int)special_bound_hi_named(topo, n_topo, "gidx1"), 5);
  ASSERT_INT_EQ((int)special_bound_hi_named(topo, n_topo, "gidx2"), 6);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, add_gpudims_rewrites_large_source_nodes) {
  /* C port guard: tinygrad substitution handles arbitrary source tuple sizes.
   * Polygrad must not cap rewritten source arrays at the small stack scratch
   * size, or model-scale SINK/END nodes can read past initialized sources. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(16));
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_INDEX, bound, poly_arg_range(0, POLY_AXIS_GLOBAL));
  PolyUOp *srcs[72];
  srcs[0] = range;
  for (int i = 1; i < 72; i++)
    srcs[i] = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(i));
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, srcs, 72, poly_arg_str("large_srcs"));
  PolyUOp *rewritten = poly_add_gpudims(ctx, sink);

  ASSERT_INT_EQ(rewritten->n_src, 72);
  ASSERT_TRUE(rewritten->src[0]->op == POLY_OP_SPECIAL);
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  ASSERT_INT_EQ(count_lin_ops(topo, n_topo, POLY_OP_SPECIAL), 1);
  ASSERT_INT_EQ(count_lin_ops(topo, n_topo, POLY_OP_RANGE), 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, add_gpudims_webgpu_splits_oversized_global_dim) {
  /* tinygrad gpudims.py legalizes backend launch dimensions before rendering.
   * WebGPU caps workgroup_id.x at 65535, so a logical 1D launch of 73728
   * workgroups must split into two hardware SPECIAL dimensions while the
   * replacement expression reconstructs the original logical index. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(73728));
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_INDEX, bound, poly_arg_range(0, POLY_AXIS_GLOBAL));
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, range, poly_arg_str("split_global"));

  PolyRendererCaps caps = {.global_max = {65535, 65535, 65535}, .local_max = {256, 256, 64}};
  PolyUOp *rewritten = poly_add_gpudims_ex(ctx, sink, caps);

  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  ASSERT_INT_EQ(count_special_named(topo, n_topo, "gidx0"), 1);
  ASSERT_INT_EQ(count_special_named(topo, n_topo, "gidx1"), 1);
  ASSERT_INT_EQ((int)special_bound_hi_named(topo, n_topo, "gidx0"), 36864);
  ASSERT_INT_EQ((int)special_bound_hi_named(topo, n_topo, "gidx1"), 2);
  ASSERT_INT_EQ(count_lin_ops(topo, n_topo, POLY_OP_RANGE), 0);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, group_for_reduce_preserves_range_replacement_metadata_like_tinygrad) {
  /* Pinned expander.py:139 builds the final REDUCE loop with x.replace(arg=...),
   * so dtype, bound source, tag, and tag_arg stay identical to the original
   * GROUP_REDUCE range and the final INDEX consumes that exact occurrence. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(16));
  PolyUOp *group_srcs[1] = {bound};
  PolyUOp *group = poly_uop_tagged_arg(
      ctx, POLY_OP_RANGE, POLY_INDEX, group_srcs, 1, poly_arg_range(7, POLY_AXIS_GROUP_REDUCE), 23,
      poly_arg_str("group-range")
  );
  PolyUOp *value = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, group, poly_arg_none());
  PolyUOp *reduce_srcs[2] = {value, group};
  PolyUOp *reduce =
      poly_uop(ctx, POLY_OP_REDUCE, POLY_FLOAT32, reduce_srcs, 2, poly_arg_ops(POLY_OP_ADD));
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, reduce, poly_arg_none());
  PolyUOp *grouped = poly_group_for_reduce(ctx, sink, 256);

  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, grouped, &n_topo);
  PolyUOp *final_range = NULL;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (u->op == POLY_OP_RANGE && poly_range_axis_type(u->arg) == POLY_AXIS_REDUCE &&
        poly_range_axis_id(u->arg) == 107) {
      final_range = u;
      break;
    }
  }
  ASSERT_NOT_NULL(final_range);
  ASSERT_TRUE(poly_dtype_eq(final_range->dtype, POLY_INDEX));
  ASSERT_INT_EQ(final_range->n_src, 1);
  ASSERT_PTR_EQ(final_range->src[0], bound);
  ASSERT_INT_EQ(final_range->tag, 23);
  ASSERT_TRUE(final_range->tag_arg.kind == POLY_ARG_STRING);
  ASSERT_STR_EQ(final_range->tag_arg.str, "group-range");

  bool direct_index_coordinate = false;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (u->op == POLY_OP_INDEX && u->n_src >= 2 && u->src[1] == final_range) {
      direct_index_coordinate = true;
      break;
    }
  }
  ASSERT_TRUE(direct_index_coordinate);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, add_gpudims_group_reduce_after_source_becomes_lidx) {
  /* tinygrad gpudims.py substitutes GROUP_REDUCE axes as local workitem
   * indices. It only skips AxisType.REDUCE. Polygrad must not infer "serial
   * reduce" from accumulator AFTER source lists, because group-reduce ranges can
   * appear there as input-context ranges and still need to become lidxN. */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(256));
  PolyUOp *group_range = poly_uop1(
      ctx, POLY_OP_RANGE, POLY_INDEX, bound, poly_arg_range(1000, POLY_AXIS_GROUP_REDUCE)
  );
  PolyDType reg_f32 = poly_dtype_ptr(POLY_FLOAT32, 1, POLY_ADDR_REG);
  PolyUOp *acc = poly_uop0(ctx, POLY_OP_DEFINE_REG, reg_f32, poly_arg_int(0));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(0));
  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, reg_f32, acc, zero, poly_arg_none());
  PolyUOp *init = poly_uop2(
      ctx, POLY_OP_STORE, POLY_VOID, idx0,
      poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(0.0)), poly_arg_none()
  );
  PolyUOp *after_srcs[3] = {acc, init, group_range};
  PolyUOp *after = poly_uop(ctx, POLY_OP_AFTER, reg_f32, after_srcs, 3, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, reg_f32, after, zero, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx1, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, load, poly_arg_str("group_reduce"));

  PolyUOp *rewritten = poly_add_gpudims(ctx, sink);
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);

  ASSERT_INT_EQ(count_special_named(topo, n_topo, "lidx0"), 1);
  for (int i = 0; i < n_topo; i++) {
    ASSERT_FALSE(
        topo[i]->op == POLY_OP_RANGE && poly_arg_is_range(topo[i]->arg) &&
        poly_range_axis_type(topo[i]->arg) == POLY_AXIS_GROUP_REDUCE
    );
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, gpu_output_gate_matches_pinned_gpudims_gater_and_linear_cleanup) {
  /* Pinned tinygrad:
   *   gpudims.py:92-99       -> INDEX(buf, WHERE(gate, idx, Invalid))
   *   late/gater.py:15-17    -> STORE(INDEX(buf, idx), data, gate)
   *   codegen/__init__.py:152-174 -> IF / STORE / ENDIF lines. */
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, 1, POLY_ADDR_GLOBAL);
  PolyUOp *out = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(0));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(32));
  PolyUOp *local =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_INDEX, bound, poly_arg_range(0, POLY_AXIS_GROUP_REDUCE));
  PolyUOp *idx = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, out, zero, poly_arg_none());
  PolyUOp *value = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, local, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx, value, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_str("gated_store"));

  PolyRendererCaps caps = {
      .has_mulacc = true,
      .has_int64 = true,
      .has_local = true,
      .global_max = {2147483647, 65535, 65535},
      .local_max = {1024, 1024, 64},
  };
  PolyUOp *gpudims = poly_add_gpudims_ex(ctx, sink, caps);
  ASSERT_NOT_NULL(gpudims);
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, gpudims, &n_topo);
  PolyUOp *gpudims_store = NULL;
  for (int i = 0; i < n_topo; i++)
    if (topo[i]->op == POLY_OP_STORE) gpudims_store = topo[i];
  ASSERT_NOT_NULL(gpudims_store);
  ASSERT_INT_EQ(gpudims_store->n_src, 2);
  PolyUOp *gpudims_idx = poly_find_index_through_cast(gpudims_store->src[0]);
  ASSERT_NOT_NULL(gpudims_idx);
  ASSERT_INT_EQ(gpudims_idx->n_src, 2);
  ASSERT_INT_EQ(gpudims_idx->src[1]->op, POLY_OP_WHERE);
  ASSERT_TRUE(poly_dtype_is_int(gpudims_idx->src[1]->dtype));
  ASSERT_TRUE(poly_dtype_eq(gpudims_idx->src[1]->src[0]->dtype, POLY_BOOL));
  ASSERT_INT_EQ(gpudims_idx->src[1]->src[2]->arg.kind, POLY_ARG_INVALID);
  PolyUOp *gpudims_gate = gpudims_idx->src[1]->src[0];
  ASSERT_INT_EQ(gpudims_gate->op, POLY_OP_CMPNE);
  ASSERT_INT_EQ(gpudims_gate->n_src, 2);
  ASSERT_INT_EQ(gpudims_gate->src[0]->op, POLY_OP_CMPNE);
  ASSERT_TRUE(poly_dtype_eq(gpudims_gate->src[1]->dtype, POLY_BOOL));
  ASSERT_INT_EQ(gpudims_gate->src[1]->arg.kind, POLY_ARG_BOOL);
  ASSERT_TRUE(gpudims_gate->src[1]->arg.b);
  PolyUOp *gpudims_not_zero = gpudims_gate->src[0];
  ASSERT_INT_EQ(gpudims_not_zero->n_src, 2);
  ASSERT_TRUE(poly_dtype_is_index(gpudims_not_zero->src[0]->dtype));
  ASSERT_TRUE(poly_dtype_is_index(gpudims_not_zero->src[1]->dtype));
  ASSERT_INT_EQ(gpudims_not_zero->src[1]->op, POLY_OP_CONST);
  ASSERT_INT_EQ(gpudims_not_zero->src[1]->arg.kind, POLY_ARG_INT);
  ASSERT_INT_EQ(gpudims_not_zero->src[1]->arg.i, 0);
  ASSERT_TRUE(poly_validate_kernel_graph(ctx, gpudims));

  PolyRewriteOpts opts = {
      .optimize = false,
      .devectorize = 1,
      .caps = caps,
      .device = POLY_DEVICE_CUDA,
      .opt_policy = POLY_OPT_HEURISTIC,
  };
  PolyUOp *rewritten = poly_full_rewrite_to_sink_ex(ctx, sink, opts);
  ASSERT_NOT_NULL(rewritten);
  topo = poly_toposort(ctx, rewritten, &n_topo);
  PolyUOp *final_store = NULL;
  for (int i = 0; i < n_topo; i++) {
    ASSERT_FALSE(poly_dtype_is_index(topo[i]->dtype));
    ASSERT_FALSE(topo[i]->op == POLY_OP_CONST && topo[i]->arg.kind == POLY_ARG_INVALID);
    if (topo[i]->op == POLY_OP_STORE) final_store = topo[i];
    if (topo[i]->op != POLY_OP_INDEX) continue;
    for (int j = 1; j < topo[i]->n_src; j++)
      ASSERT_TRUE(poly_dtype_is_int(topo[i]->src[j]->dtype));
  }
  ASSERT_NOT_NULL(final_store);
  ASSERT_INT_EQ(final_store->n_src, 3);
  ASSERT_TRUE(poly_dtype_eq(final_store->src[2]->dtype, POLY_BOOL));
  PolyUOp *final_idx = poly_find_index_through_cast(final_store->src[0]);
  ASSERT_NOT_NULL(final_idx);
  ASSERT_INT_EQ(final_idx->n_src, 2);
  ASSERT_TRUE(poly_validate_kernel_graph(ctx, rewritten));

  int n_linear = 0;
  PolyUOp **linear = poly_linearize_rewritten(ctx, rewritten, &n_linear);
  ASSERT_NOT_NULL(linear);
  int if_count = 0, endif_count = 0, store_count = 0;
  for (int i = 0; i < n_linear; i++) {
    if (linear[i]->op == POLY_OP_IF) {
      if_count++;
      ASSERT_TRUE(i + 2 < n_linear);
      ASSERT_INT_EQ(linear[i + 1]->op, POLY_OP_STORE);
      ASSERT_INT_EQ(linear[i + 1]->n_src, 2);
      ASSERT_INT_EQ(linear[i + 2]->op, POLY_OP_ENDIF);
      ASSERT_PTR_EQ(linear[i + 2]->src[0], linear[i]);
    }
    if (linear[i]->op == POLY_OP_ENDIF) endif_count++;
    if (linear[i]->op == POLY_OP_STORE) {
      store_count++;
      ASSERT_INT_EQ(linear[i]->n_src, 2);
    }
  }
  ASSERT_INT_EQ(if_count, 1);
  ASSERT_INT_EQ(endif_count, 1);
  ASSERT_INT_EQ(store_count, 1);

  free(linear);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, linearize_webgpu_special_stays_int32_under_wide_index_use) {
  /* tinygrad pm_lower_index_dtype keeps SPECIAL itself int32 and casts widened
   * uses. Rebuilding SPECIAL as int64 creates duplicate WGSL declarations for
   * the same builtin name when another use keeps the int32 form. */
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *out = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(16));
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_INDEX, bound, poly_arg_range(0, POLY_AXIS_GLOBAL));
  PolyUOp *wide = poly_uop1(ctx, POLY_OP_CAST, POLY_INT64, range, poly_arg_none());
  PolyUOp *idx = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, out, wide, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx, load, poly_arg_none());
  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_str("special_lower"));

  int n_lin = 0;
  PolyUOp **lin = poly_linearize_webgpu(ctx, sink, &n_lin);
  ASSERT_NOT_NULL(lin);

  int n_special = 0;
  for (int i = 0; i < n_lin; i++) {
    if (lin[i]->op != POLY_OP_SPECIAL) continue;
    n_special++;
    ASSERT_TRUE(poly_dtype_eq(poly_dtype_scalar(lin[i]->dtype), POLY_INT32));
    ASSERT_TRUE(lin[i]->n_src == 1);
    ASSERT_TRUE(poly_dtype_eq(poly_dtype_scalar(lin[i]->src[0]->dtype), POLY_INT32));
  }
  ASSERT_INT_EQ(n_special, 1);

  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, lower_index_dtype_preserves_rank10_index_sources) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *buf = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));

  PolyUOp *idx_srcs[11];
  idx_srcs[0] = buf;
  for (int i = 1; i < 11; i++)
    idx_srcs[i] = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(0));

  PolyUOp *idx = poly_uop(ctx, POLY_OP_INDEX, ptr_f32, idx_srcs, 11, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, load, poly_arg_str("rank10_index"));

  PolyUOp *rewritten = poly_apply_post_index_symbolic_stage(ctx, sink, 1);
  ASSERT_NOT_NULL(rewritten);

  int n = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n);
  bool saw_rank10_index = false;
  for (int i = 0; i < n; i++) {
    if (topo[i]->op != POLY_OP_INDEX || topo[i]->n_src != 11) continue;
    saw_rank10_index = true;
    for (int j = 1; j < topo[i]->n_src; j++)
      ASSERT_TRUE(poly_dtype_eq(poly_dtype_scalar(topo[i]->src[j]->dtype), POLY_INT32));
  }
  ASSERT_TRUE(saw_rank10_index);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, lower_index_dtype_uses_one_graph_local_traversal) {
  /* Pinned tinygrad's pm_lower_index_dtype consists of local rewrite rules;
   * RewriteContext owns descendant traversal and memoization
   * (tinygrad/uop/ops.py:1655-1686). A callback that recursively lowers every
   * matched subtree turns this shared chain quadratic. Keep both the exact
   * concrete result and a generous scaling ceiling as regression evidence. */
  const int depths[2] = {8192, 16384};
  double elapsed_s[2] = {0.0, 0.0};
  for (int run = 0; run < 2; run++) {
    int depth = depths[run];
    PolyCtx *ctx = poly_ctx_new();
    ASSERT_NOT_NULL(ctx);

    PolyUOp *value = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(0));
    for (int i = 0; i < depth; i++) {
      PolyUOp *term = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(i + 1));
      value = poly_uop2(ctx, POLY_OP_ADD, POLY_INDEX, value, term, poly_arg_none());
    }
    PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, value, poly_arg_none());

    clock_t begin = clock();
    PolyUOp *lowered = poly_apply_post_index_symbolic_stage(ctx, sink, 1);
    elapsed_s[run] = (double)(clock() - begin) / (double)CLOCKS_PER_SEC;
    ASSERT_NOT_NULL(lowered);
    ASSERT_INT_EQ(lowered->op, POLY_OP_SINK);
    ASSERT_INT_EQ(lowered->n_src, 1);
    ASSERT_INT_EQ(lowered->src[0]->op, POLY_OP_CONST);
    ASSERT_TRUE(poly_dtype_eq(poly_dtype_scalar(lowered->src[0]->dtype), POLY_INT32));
    ASSERT_INT_EQ(lowered->src[0]->arg.kind, POLY_ARG_INT);
    ASSERT_INT_EQ(lowered->src[0]->arg.i, (int64_t)depth * (depth + 1) / 2);
    poly_ctx_destroy(ctx);
  }
  ASSERT_TRUE(elapsed_s[1] < elapsed_s[0] * 2.75 + 0.05);
  ASSERT_TRUE(elapsed_s[1] < 3.0);
  PASS();
}

TEST(codegen, lower_loaded_weak_index_uses_overflow_bounds) {
  /* Pinned tinygrad uop/ops.py:1655 lowers from u.overflows(int32), even when
   * the weak expression contains a LOAD. int32 bounds plus one require long;
   * forcing all loaded expressions to int32 silently wraps the address. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyDType ptr_i32 = poly_dtype_ptr(POLY_INT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *buf = poly_uop0(ctx, POLY_OP_PARAM, ptr_i32, poly_arg_int(0));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *index = poly_uop2(ctx, POLY_OP_INDEX, ptr_i32, buf, zero, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_INT32, index, poly_arg_none());
  PolyUOp *weak_load = poly_uop1(ctx, POLY_OP_CAST, POLY_INDEX, load, poly_arg_none());
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(1));
  PolyUOp *expr = poly_uop2(ctx, POLY_OP_ADD, POLY_INDEX, weak_load, one, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, expr, poly_arg_none());

  PolyUOp *lowered = poly_apply_post_index_symbolic_stage(ctx, sink, 1);
  ASSERT_NOT_NULL(lowered);
  ASSERT_INT_EQ(lowered->op, POLY_OP_SINK);
  ASSERT_INT_EQ(lowered->n_src, 1);
  PolyUOp *root = lowered->src[0];
  ASSERT_INT_EQ(root->op, POLY_OP_ADD);
  ASSERT_TRUE(poly_dtype_eq(root->dtype, POLY_INT64));
  ASSERT_INT_EQ(root->n_src, 2);
  ASSERT_INT_EQ(root->src[0]->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(root->src[0]->dtype, POLY_INT64));
  ASSERT_INT_EQ(root->src[0]->n_src, 1);
  ASSERT_INT_EQ(root->src[0]->src[0]->op, POLY_OP_LOAD);
  ASSERT_TRUE(poly_dtype_eq(root->src[0]->src[0]->dtype, POLY_INT32));
  ASSERT_INT_EQ(root->src[1]->op, POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(root->src[1]->dtype, POLY_INT64));
  ASSERT_INT_EQ(root->src[1]->arg.kind, POLY_ARG_INT);
  ASSERT_INT_EQ(root->src[1]->arg.i, 1);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, valid_gated_loaded_index_narrows_like_tinygrad) {
  /* Pinned devectorizer.py:39-42,54-58 applies uop_given_valid to the value
   * branch of an Invalid-bearing INDEX coordinate. symbolic.py:315-356 then
   * proves LOAD<int> in [0,1], so LOAD+2 is int32 even though the same
   * arithmetic without the gate must remain int64. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyDType ptr_i32 = poly_dtype_ptr(POLY_INT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *indices = poly_uop0(ctx, POLY_OP_PARAM, ptr_i32, poly_arg_int(0));
  PolyUOp *table = poly_uop0(ctx, POLY_OP_PARAM, ptr_i32, poly_arg_int(1));
  PolyUOp *out = poly_uop0(ctx, POLY_OP_PARAM, ptr_i32, poly_arg_int(2));
  PolyUOp *lane = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(2));
  PolyUOp *indices_at_lane =
      poly_uop2(ctx, POLY_OP_INDEX, ptr_i32, indices, lane, poly_arg_none());
  PolyUOp *loaded = poly_uop1(ctx, POLY_OP_LOAD, POLY_INT32, indices_at_lane, poly_arg_none());
  PolyUOp *weak_loaded = poly_uop1(ctx, POLY_OP_CAST, POLY_INDEX, loaded, poly_arg_none());
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(0));
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(2));
  PolyUOp *true_uop = poly_uop0(ctx, POLY_OP_CONST, POLY_BOOL, poly_arg_bool(true));
  PolyUOp *below_zero =
      poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, weak_loaded, zero, poly_arg_none());
  PolyUOp *at_least_zero =
      poly_uop2(ctx, POLY_OP_CMPNE, POLY_BOOL, below_zero, true_uop, poly_arg_none());
  PolyUOp *below_two =
      poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, weak_loaded, two, poly_arg_none());
  PolyUOp *gate =
      poly_uop2(ctx, POLY_OP_AND, POLY_BOOL, at_least_zero, below_two, poly_arg_none());
  PolyUOp *shifted =
      poly_uop2(ctx, POLY_OP_ADD, POLY_INDEX, weak_loaded, two, poly_arg_none());
  PolyUOp *invalid = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_invalid());
  PolyUOp *gated =
      poly_uop3(ctx, POLY_OP_WHERE, POLY_INDEX, gate, shifted, invalid, poly_arg_none());
  PolyUOp *table_index =
      poly_uop2(ctx, POLY_OP_INDEX, ptr_i32, table, gated, poly_arg_none());
  PolyUOp *value = poly_uop1(ctx, POLY_OP_LOAD, POLY_INT32, table_index, poly_arg_none());
  PolyUOp *out_zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *out_index =
      poly_uop2(ctx, POLY_OP_INDEX, ptr_i32, out, out_zero, poly_arg_none());
  PolyUOp *store =
      poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_index, value, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  PolyUOp *lowered = poly_apply_post_index_symbolic_stage(ctx, sink, 1);
  ASSERT_NOT_NULL(lowered);
  PolyUOp *lowered_coord = lowered->src[0]->src[1]->src[0]->src[1];
  ASSERT_INT_EQ(lowered_coord->op, POLY_OP_WHERE);
  ASSERT_TRUE(poly_dtype_eq(lowered_coord->dtype, POLY_INT32));
  ASSERT_INT_EQ(lowered_coord->n_src, 3);
  ASSERT_INT_EQ(lowered_coord->src[1]->op, POLY_OP_ADD);
  ASSERT_TRUE(poly_dtype_eq(lowered_coord->src[1]->dtype, POLY_INT32));
  ASSERT_INT_EQ(lowered_coord->src[1]->src[0]->op, POLY_OP_LOAD);
  ASSERT_TRUE(poly_dtype_eq(lowered_coord->src[1]->src[0]->dtype, POLY_INT32));
  ASSERT_INT_EQ(lowered_coord->src[1]->src[1]->op, POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(lowered_coord->src[1]->src[1]->dtype, POLY_INT32));
  ASSERT_INT_EQ(lowered_coord->src[1]->src[1]->arg.i, 2);
  int n_lowered = 0;
  PolyUOp **lowered_topo = poly_toposort(ctx, lowered, &n_lowered);
  ASSERT_NOT_NULL(lowered_topo);
  for (int i = 0; i < n_lowered; i++) {
    PolyDType scalar = poly_dtype_scalar(lowered_topo[i]->dtype);
    ASSERT_TRUE(!poly_dtype_is_int(scalar) || scalar.bitsize != 64);
    ASSERT_TRUE(lowered_topo[i]->op != POLY_OP_DEFINE_VAR);
  }

  PolyUOp *webgpu = poly_rewrite_webgpu(ctx, sink);
  ASSERT_NOT_NULL(webgpu);
  int n_webgpu = 0;
  PolyUOp **webgpu_topo = poly_toposort(ctx, webgpu, &n_webgpu);
  ASSERT_NOT_NULL(webgpu_topo);
  for (int i = 0; i < n_webgpu; i++) {
    PolyDType scalar = poly_dtype_scalar(webgpu_topo[i]->dtype);
    ASSERT_TRUE(!poly_dtype_is_int(scalar) || scalar.bitsize != 64);
  }
  ASSERT_TRUE(poly_validate_kernel_graph(ctx, webgpu));

  /* The validity proof must not over-narrow a branch whose shifted maximum
   * exceeds int32. `loaded < INT32_MAX` permits INT32_MAX-1, and +2 needs
   * signed long. */
  PolyUOp *i32max =
      poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(INT32_MAX));
  PolyUOp *wide_upper =
      poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, weak_loaded, i32max, poly_arg_none());
  PolyUOp *wide_gate =
      poly_uop2(ctx, POLY_OP_AND, POLY_BOOL, at_least_zero, wide_upper, poly_arg_none());
  PolyUOp *wide_gated =
      poly_uop3(ctx, POLY_OP_WHERE, POLY_INDEX, wide_gate, shifted, invalid, poly_arg_none());
  PolyUOp *wide_index =
      poly_uop2(ctx, POLY_OP_INDEX, ptr_i32, table, wide_gated, poly_arg_none());
  PolyUOp *wide_load = poly_uop1(ctx, POLY_OP_LOAD, POLY_INT32, wide_index, poly_arg_none());
  PolyUOp *wide_store =
      poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_index, wide_load, poly_arg_none());
  PolyUOp *wide_sink =
      poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, wide_store, poly_arg_none());
  PolyUOp *wide_lowered = poly_apply_post_index_symbolic_stage(ctx, wide_sink, 1);
  ASSERT_NOT_NULL(wide_lowered);
  PolyUOp *wide_coord = wide_lowered->src[0]->src[1]->src[0]->src[1];
  ASSERT_INT_EQ(wide_coord->op, POLY_OP_WHERE);
  ASSERT_TRUE(poly_dtype_eq(wide_coord->dtype, POLY_INT64));
  ASSERT_INT_EQ(wide_coord->src[1]->op, POLY_OP_ADD);
  ASSERT_TRUE(poly_dtype_eq(wide_coord->src[1]->dtype, POLY_INT64));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, valid_index_simplifies_inside_devectorizer_like_tinygrad) {
  /* Pinned tinygrad codegen/__init__.py:105-110 includes
   * load_store_indexing in both the devectorizer and lower-index rewrites.
   * devectorizer.py:39-42 therefore reduces x%8 to x under 0<=x<8 before
   * weak-index dtype lowering; checking only the final graph misses that
   * stage-order contract and can produce materially worse kernels. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyDType ptr_i32 = poly_dtype_ptr(POLY_INT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *buf = poly_uop0(ctx, POLY_OP_PARAM, ptr_i32, poly_arg_int(0));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(0));
  PolyUOp *input_index =
      poly_uop2(ctx, POLY_OP_INDEX, ptr_i32, buf, zero, poly_arg_none());
  PolyUOp *loaded = poly_uop1(ctx, POLY_OP_LOAD, POLY_INT32, input_index, poly_arg_none());
  PolyUOp *weak_loaded = poly_uop1(ctx, POLY_OP_CAST, POLY_INDEX, loaded, poly_arg_none());
  PolyUOp *width = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(8));
  PolyUOp *truth = poly_uop0(ctx, POLY_OP_CONST, POLY_BOOL, poly_arg_bool(true));
  PolyUOp *negative =
      poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, weak_loaded, zero, poly_arg_none());
  PolyUOp *nonnegative =
      poly_uop2(ctx, POLY_OP_CMPNE, POLY_BOOL, negative, truth, poly_arg_none());
  PolyUOp *below_width =
      poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, weak_loaded, width, poly_arg_none());
  PolyUOp *valid =
      poly_uop2(ctx, POLY_OP_AND, POLY_BOOL, nonnegative, below_width, poly_arg_none());
  PolyUOp *bounded_mod =
      poly_uop2(ctx, POLY_OP_FLOORMOD, POLY_INDEX, weak_loaded, width, poly_arg_none());
  PolyUOp *invalid = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_invalid());
  PolyUOp *coord =
      poly_uop3(ctx, POLY_OP_WHERE, POLY_INDEX, valid, bounded_mod, invalid, poly_arg_none());
  PolyUOp *root = poly_uop2(ctx, POLY_OP_INDEX, ptr_i32, buf, coord, poly_arg_none());

  PolyRendererCaps caps = {.max_vec_width = 1};
  PolyUOp *devec = poly_apply_devectorize_stage(ctx, root, 1, caps);
  ASSERT_NOT_NULL(devec);
  ASSERT_INT_EQ(devec->op, POLY_OP_INDEX);
  ASSERT_INT_EQ(devec->n_src, 2);
  PolyUOp *devec_coord = devec->src[1];
  ASSERT_INT_EQ(devec_coord->op, POLY_OP_WHERE);
  ASSERT_TRUE(poly_dtype_eq(devec_coord->dtype, POLY_INDEX));
  ASSERT_INT_EQ(devec_coord->n_src, 3);
  ASSERT_INT_EQ(devec_coord->src[1]->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(devec_coord->src[1]->dtype, POLY_INDEX));
  ASSERT_INT_EQ(devec_coord->src[1]->n_src, 1);
  ASSERT_PTR_EQ(devec_coord->src[1]->src[0], loaded);

  PolyUOp *lowered = poly_apply_post_index_symbolic_stage(ctx, devec, 1);
  ASSERT_NOT_NULL(lowered);
  ASSERT_INT_EQ(lowered->op, POLY_OP_INDEX);
  PolyUOp *lowered_coord = lowered->src[1];
  ASSERT_INT_EQ(lowered_coord->op, POLY_OP_WHERE);
  ASSERT_TRUE(poly_dtype_eq(lowered_coord->dtype, POLY_INT32));
  ASSERT_INT_EQ(lowered_coord->src[1]->op, POLY_OP_LOAD);
  ASSERT_TRUE(poly_dtype_eq(lowered_coord->src[1]->dtype, POLY_INT32));
  ASSERT_INT_EQ(lowered_coord->src[1]->n_src, 1);
  PolyUOp *lowered_input_index = lowered_coord->src[1]->src[0];
  ASSERT_INT_EQ(lowered_input_index->op, POLY_OP_INDEX);
  ASSERT_INT_EQ(lowered_input_index->n_src, 2);
  ASSERT_PTR_EQ(lowered_input_index->src[0], buf);
  ASSERT_INT_EQ(lowered_input_index->src[1]->op, POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(lowered_input_index->src[1]->dtype, POLY_INT32));
  ASSERT_INT_EQ(lowered_input_index->src[1]->arg.i, 0);

  int n = 0;
  PolyUOp **topo = poly_toposort(ctx, lowered, &n);
  ASSERT_NOT_NULL(topo);
  for (int i = 0; i < n; i++) {
    ASSERT_TRUE(topo[i]->op != POLY_OP_FLOORMOD);
    ASSERT_TRUE(topo[i]->op != POLY_OP_DEFINE_VAR);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, wide_after_cast_devectorization_preserves_every_effect) {
  /* Pinned devectorizer.py:268-270 uses allow_any_len=True and rebuilds
   * AFTER(CAST(x), *effects) as CAST(AFTER(x, *effects)). ResNet18 backward
   * reaches more than 128 effects, so assert exact arbitrary-arity topology. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_TRUE(ctx != NULL);
  const int n_effects = 129;
  PolyUOp **src = calloc((size_t)n_effects + 1, sizeof(*src));
  ASSERT_TRUE(src != NULL);
  PolyUOp *base = poly_uop0(ctx, POLY_OP_NOOP, POLY_FLOAT16, poly_arg_str("base"));
  src[0] = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, base, poly_arg_none());
  ASSERT_TRUE(base != NULL && src[0] != NULL);
  for (int i = 0; i < n_effects; i++) {
    src[i + 1] = poly_uop0(ctx, POLY_OP_NOOP, POLY_VOID, poly_arg_int(i));
    ASSERT_TRUE(src[i + 1] != NULL);
  }
  PolyUOp *root = poly_uop(
      ctx, POLY_OP_AFTER, POLY_FLOAT32, src, n_effects + 1, poly_arg_none());
  ASSERT_TRUE(root != NULL);

  PolyUOp *rewritten = poly_graph_rewrite(ctx, root, poly_pm_devectorize_pass());
  ASSERT_TRUE(rewritten != NULL && rewritten->op == POLY_OP_CAST);
  ASSERT_INT_EQ(rewritten->n_src, 1);
  ASSERT_TRUE(poly_dtype_eq(rewritten->dtype, POLY_FLOAT32));
  PolyUOp *inner = rewritten->src[0];
  ASSERT_TRUE(inner != NULL && inner->op == POLY_OP_AFTER);
  ASSERT_TRUE(poly_dtype_eq(inner->dtype, POLY_FLOAT16));
  ASSERT_INT_EQ(inner->n_src, n_effects + 1);
  ASSERT_TRUE(inner->src[0] == base);
  for (int i = 0; i < n_effects; i++)
    ASSERT_TRUE(inner->src[i + 1] == src[i + 1]);

  free(src);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, post_index_validity_does_not_specialize_effects_under_value_where) {
  /* Pinned codegen/__init__.py:105-111 runs only pm_lower_index_dtype,
   * load_store_indexing, gep_pushing, then symbolic at this boundary.
   * symbolic.py:422-428 scopes uop_given_valid to a weak-index WHERE value;
   * devectorizer.py:39-58 scopes it to an Invalid-bearing INDEX coordinate.
   * An outer float WHERE therefore must not recursively specialize a STORE
   * coordinate hidden under AFTER. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(2));
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_INDEX, two, poly_arg_range(0, POLY_AXIS_LOOP));
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(1));
  PolyUOp *cond = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, range, one, poly_arg_none());
  PolyUOp *invalid = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_invalid());
  PolyUOp *coord =
      poly_uop3(ctx, POLY_OP_WHERE, POLY_INDEX, cond, range, invalid, poly_arg_none());
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *buf = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));
  PolyUOp *index = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, buf, coord, poly_arg_none());
  PolyUOp *store = poly_uop2(
      ctx, POLY_OP_STORE, POLY_VOID, index,
      poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0)), poly_arg_none()
  );
  PolyUOp *after = poly_uop2(
      ctx, POLY_OP_AFTER, POLY_FLOAT32,
      poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(2.0)), store,
      poly_arg_none()
  );
  PolyUOp *outer = poly_uop3(
      ctx, POLY_OP_WHERE, POLY_FLOAT32, cond, after,
      poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(0.0)), poly_arg_none()
  );
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, outer, poly_arg_none());

  PolyUOp *lowered = poly_apply_post_index_symbolic_stage(ctx, sink, 1);
  ASSERT_NOT_NULL(lowered);
  int n = 0;
  PolyUOp **topo = poly_toposort(ctx, lowered, &n);
  ASSERT_NOT_NULL(topo);
  ASSERT_INT_EQ(n, 16);
  ASSERT_INT_EQ(count_lin_ops(topo, n, POLY_OP_WHERE), 2);
  ASSERT_INT_EQ(count_lin_ops(topo, n, POLY_OP_STORE), 1);
  ASSERT_INT_EQ(count_lin_ops(topo, n, POLY_OP_AFTER), 1);

  PolyUOp *lowered_store = NULL;
  for (int i = 0; i < n; i++)
    if (topo[i]->op == POLY_OP_STORE) lowered_store = topo[i];
  ASSERT_NOT_NULL(lowered_store);
  ASSERT_INT_EQ(lowered_store->n_src, 2);
  ASSERT_INT_EQ(lowered_store->src[0]->op, POLY_OP_INDEX);
  ASSERT_INT_EQ(lowered_store->src[0]->n_src, 2);
  PolyUOp *lowered_coord = lowered_store->src[0]->src[1];
  ASSERT_INT_EQ(lowered_coord->op, POLY_OP_WHERE);
  ASSERT_TRUE(poly_dtype_eq(lowered_coord->dtype, POLY_INT32));
  ASSERT_INT_EQ(lowered_coord->n_src, 3);
  ASSERT_INT_EQ(lowered_coord->src[2]->op, POLY_OP_CONST);
  ASSERT_INT_EQ(lowered_coord->src[2]->arg.kind, POLY_ARG_INVALID);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, lower_weak_comparison_legalizes_mixed_integer_operands) {
  /* Pinned tinygrad's GroupOp.Binary rule includes comparisons
   * (uop/ops.py:1657-1659): the result remains bool, but both weak-index
   * operands are cast to their least-upper concrete integer dtype. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyDType ptr_i32 = poly_dtype_ptr(POLY_INT32, -1, POLY_ADDR_GLOBAL);
  PolyDType ptr_i64 = poly_dtype_ptr(POLY_INT64, -1, POLY_ADDR_GLOBAL);
  PolyUOp *buf32 = poly_uop0(ctx, POLY_OP_PARAM, ptr_i32, poly_arg_int(0));
  PolyUOp *buf64 = poly_uop0(ctx, POLY_OP_PARAM, ptr_i64, poly_arg_int(1));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *idx32 = poly_uop2(ctx, POLY_OP_INDEX, ptr_i32, buf32, zero, poly_arg_none());
  PolyUOp *idx64 = poly_uop2(ctx, POLY_OP_INDEX, ptr_i64, buf64, zero, poly_arg_none());
  PolyUOp *load32 = poly_uop1(ctx, POLY_OP_LOAD, POLY_INT32, idx32, poly_arg_none());
  PolyUOp *load64 = poly_uop1(ctx, POLY_OP_LOAD, POLY_INT64, idx64, poly_arg_none());
  PolyUOp *weak32 = poly_uop1(ctx, POLY_OP_CAST, POLY_INDEX, load32, poly_arg_none());
  PolyUOp *weak64 = poly_uop1(ctx, POLY_OP_CAST, POLY_INDEX, load64, poly_arg_none());
  PolyUOp *compare = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, weak32, weak64, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, compare, poly_arg_none());

  PolyUOp *lowered = poly_apply_post_index_symbolic_stage(ctx, sink, 1);
  ASSERT_NOT_NULL(lowered);
  ASSERT_INT_EQ(lowered->op, POLY_OP_SINK);
  ASSERT_INT_EQ(lowered->n_src, 1);
  PolyUOp *root = lowered->src[0];
  ASSERT_INT_EQ(root->op, POLY_OP_CMPLT);
  ASSERT_TRUE(poly_dtype_eq(root->dtype, POLY_BOOL));
  ASSERT_INT_EQ(root->n_src, 2);
  ASSERT_INT_EQ(root->src[0]->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_eq(root->src[0]->dtype, POLY_INT64));
  ASSERT_PTR_EQ(root->src[0]->src[0], load32);
  ASSERT_PTR_EQ(root->src[1], load64);

  /* The weak wrappers are the provenance for this legalization. Pinned
   * pm_lower_index_dtype leaves a raw mixed concrete comparison untouched so
   * validation can reject it; do not normalize arbitrary malformed IR. */
  PolyUOp *raw_compare =
      poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, load32, load64, poly_arg_none());
  PolyUOp *raw_sink =
      poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, raw_compare, poly_arg_none());
  PolyUOp *raw_lowered = poly_apply_post_index_symbolic_stage(ctx, raw_sink, 1);
  ASSERT_NOT_NULL(raw_lowered);
  ASSERT_PTR_EQ(raw_lowered->src[0], raw_compare);
  ASSERT_PTR_EQ(raw_lowered->src[0]->src[0], load32);
  ASSERT_PTR_EQ(raw_lowered->src[0]->src[1], load64);

  PolyDType ptr_out = poly_dtype_ptr(POLY_INT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *out = poly_uop0(ctx, POLY_OP_PARAM, ptr_out, poly_arg_int(2));
  PolyUOp *out_idx = poly_uop2(ctx, POLY_OP_INDEX, ptr_out, out, zero, poly_arg_none());
  PolyUOp *value = poly_uop1(ctx, POLY_OP_CAST, POLY_INT32, compare, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_idx, value, poly_arg_none());
  PolyUOp *webgpu = poly_rewrite_webgpu(
      ctx, poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none())
  );
  ASSERT_NOT_NULL(webgpu);
  int n = 0;
  PolyUOp **topo = poly_toposort(ctx, webgpu, &n);
  ASSERT_NOT_NULL(topo);
  for (int i = 0; i < n; i++) {
    PolyDType scalar = poly_dtype_scalar(topo[i]->dtype);
    ASSERT_TRUE(!poly_dtype_is_int(scalar) || scalar.bitsize != 64);
  }
  ASSERT_TRUE(poly_validate_kernel_graph(ctx, webgpu));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, lower_index_dtype_matches_pinned_range_bound_patterns) {
  /* Pinned pm_lower_index_dtype matches a weak-wrapped RANGE bound regardless
   * of the RANGE's current dtype. Weak bounds choose int/long by overflow; an
   * explicitly concrete RANGE<long> with a concrete bound is unchanged. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *narrow_bound =
      poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(16));
  PolyUOp *boundary_bound =
      poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(INT64_C(1) << 31));
  PolyUOp *wide_bound =
      poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(INT64_C(1) << 40));
  PolyUOp *long_bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT64, poly_arg_int(16));
  PolyUOp *wrapped_int_bound = poly_uop1(
      ctx, POLY_OP_CAST, POLY_INDEX,
      poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(16)), poly_arg_none()
  );
  PolyUOp *ranges[5] = {
      poly_uop1(
          ctx, POLY_OP_RANGE, POLY_INDEX, narrow_bound,
          poly_arg_range(0, POLY_AXIS_LOOP)
      ),
      poly_uop1(
          ctx, POLY_OP_RANGE, POLY_INDEX, boundary_bound,
          poly_arg_range(1, POLY_AXIS_LOOP)
      ),
      poly_uop1(
          ctx, POLY_OP_RANGE, POLY_INDEX, wide_bound,
          poly_arg_range(2, POLY_AXIS_LOOP)
      ),
      poly_uop1(
          ctx, POLY_OP_RANGE, POLY_INT64, long_bound,
          poly_arg_range(3, POLY_AXIS_LOOP)
      ),
      poly_uop1(
          ctx, POLY_OP_RANGE, POLY_INT32, wrapped_int_bound,
          poly_arg_range(4, POLY_AXIS_LOOP)
      ),
  };
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, ranges, 5, poly_arg_none());

  PolyUOp *lowered = poly_apply_post_index_symbolic_stage(ctx, sink, 1);
  ASSERT_NOT_NULL(lowered);
  ASSERT_INT_EQ(lowered->op, POLY_OP_SINK);
  ASSERT_INT_EQ(lowered->n_src, 5);
  ASSERT_TRUE(poly_dtype_eq(lowered->src[0]->dtype, POLY_INT32));
  ASSERT_TRUE(poly_dtype_eq(lowered->src[0]->src[0]->dtype, POLY_INT32));
  ASSERT_TRUE(poly_dtype_eq(lowered->src[1]->dtype, POLY_INT64));
  ASSERT_TRUE(poly_dtype_eq(lowered->src[1]->src[0]->dtype, POLY_INT64));
  ASSERT_TRUE(poly_dtype_eq(lowered->src[2]->dtype, POLY_INT64));
  ASSERT_TRUE(poly_dtype_eq(lowered->src[2]->src[0]->dtype, POLY_INT64));
  ASSERT_PTR_EQ(lowered->src[3], ranges[3]);
  ASSERT_TRUE(poly_dtype_eq(lowered->src[3]->dtype, POLY_INT64));
  ASSERT_PTR_EQ(lowered->src[3]->src[0], long_bound);
  ASSERT_INT_EQ(lowered->src[4]->op, POLY_OP_RANGE);
  ASSERT_TRUE(poly_dtype_eq(lowered->src[4]->dtype, POLY_INT32));
  ASSERT_INT_EQ(lowered->src[4]->n_src, 1);
  ASSERT_INT_EQ(lowered->src[4]->src[0]->op, POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(lowered->src[4]->src[0]->dtype, POLY_INT32));
  ASSERT_INT_EQ(lowered->src[4]->src[0]->arg.kind, POLY_ARG_INT);
  ASSERT_INT_EQ(lowered->src[4]->src[0]->arg.i, 16);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, lower_index_dtype_matches_pinned_wrapper_metadata_lifecycle) {
  /* Pinned ops.py:1657-1686 uses fresh construction for Binary/WHERE/BIND and
   * INDEX (drop tags), but replace for CONST/RANGE (preserve tags). It also
   * strips only exact scalar weakint casts at roots and only lowers ALU PARAM;
   * Polygrad's current symbolic adaptation is DEFINE_VAR (PG-PARITY-002). */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *a = poly_uop0(
      ctx, POLY_OP_DEFINE_VAR, POLY_INDEX, poly_arg_define_var("idx_a", -4, 4)
  );
  PolyUOp *b = poly_uop0(
      ctx, POLY_OP_DEFINE_VAR, POLY_INDEX, poly_arg_define_var("idx_b", -8, 8)
  );
  PolyUOp *cond = poly_uop0(
      ctx, POLY_OP_DEFINE_VAR, POLY_BOOL, poly_arg_define_var("idx_cond", 0, 1)
  );
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(2));
  PolyUOp *binary_srcs[2] = {a, b};
  PolyUOp *where_srcs[3] = {cond, a, b};
  PolyUOp *bind_srcs[2] = {a, two};
  PolyUOp *binary = poly_uop_tagged_arg(
      ctx, POLY_OP_ADD, POLY_INDEX, binary_srcs, 2, poly_arg_none(), 17,
      poly_arg_str("drop-binary")
  );
  PolyUOp *where = poly_uop_tagged_arg(
      ctx, POLY_OP_WHERE, POLY_INDEX, where_srcs, 3, poly_arg_none(), 18,
      poly_arg_str("drop-where")
  );
  PolyUOp *bind = poly_uop_tagged_arg(
      ctx, POLY_OP_BIND, POLY_INDEX, bind_srcs, 2, poly_arg_none(), 19,
      poly_arg_str("drop-bind")
  );

  PolyDType ptr_i32 = poly_dtype_ptr(POLY_INT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *buf = poly_uop0(ctx, POLY_OP_PARAM, ptr_i32, poly_arg_int(0));
  PolyUOp *index_srcs[2] = {buf, two};
  PolyUOp *index = poly_uop_tagged_arg(
      ctx, POLY_OP_INDEX, ptr_i32, index_srcs, 2, poly_arg_none(), 20,
      poly_arg_str("drop-index")
  );
  PolyUOp *tagged_const = poly_uop_tagged_arg(
      ctx, POLY_OP_CONST, POLY_INDEX, NULL, 0, poly_arg_int(3), 21,
      poly_arg_str("keep-const")
  );
  PolyUOp *range_srcs[1] = {tagged_const};
  PolyUOp *range = poly_uop_tagged_arg(
      ctx, POLY_OP_RANGE, POLY_INDEX, range_srcs, 1,
      poly_arg_range(0, POLY_AXIS_LOOP), 22, poly_arg_str("keep-range")
  );
  PolyUOp *roots[6] = {binary, where, bind, index, tagged_const, range};
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, roots, 6, poly_arg_none());

  PolyUOp *lowered = poly_apply_post_index_symbolic_stage(ctx, sink, 1);
  ASSERT_NOT_NULL(lowered);
  ASSERT_INT_EQ(lowered->n_src, 6);
  for (int i = 0; i < 4; i++) {
    ASSERT_INT_EQ(lowered->src[i]->tag, 0);
    ASSERT_INT_EQ(lowered->src[i]->tag_arg.kind, POLY_ARG_NONE);
  }
  ASSERT_INT_EQ(lowered->src[4]->tag, 21);
  ASSERT_INT_EQ(lowered->src[4]->tag_arg.kind, POLY_ARG_STRING);
  ASSERT_STR_EQ(lowered->src[4]->tag_arg.str, "keep-const");
  ASSERT_INT_EQ(lowered->src[5]->tag, 22);
  ASSERT_INT_EQ(lowered->src[5]->tag_arg.kind, POLY_ARG_STRING);
  ASSERT_STR_EQ(lowered->src[5]->tag_arg.str, "keep-range");

  PolyUOp *raw_param = poly_uop0(ctx, POLY_OP_PARAM, POLY_INDEX, poly_arg_int(7));
  PolyUOp *param_sink =
      poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, raw_param, poly_arg_none());
  PolyUOp *param_lowered = poly_apply_post_index_symbolic_stage(ctx, param_sink, 1);
  ASSERT_NOT_NULL(param_lowered);
  ASSERT_PTR_EQ(param_lowered->src[0], raw_param);
  ASSERT_TRUE(poly_dtype_eq(param_lowered->src[0]->dtype, POLY_INDEX));

  PolyUOp *lane0 = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(1));
  PolyUOp *lane1 =
      poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(INT64_C(1) << 40));
  PolyUOp *lanes[2] = {lane0, lane1};
  PolyUOp *stack = poly_uop(
      ctx, POLY_OP_STACK, poly_dtype_vec(POLY_INDEX, 2), lanes, 2, poly_arg_none()
  );
  PolyUOp *vector_sink =
      poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, stack, poly_arg_none());
  PolyUOp *vector_lowered = poly_apply_post_index_symbolic_stage(ctx, vector_sink, 1);
  ASSERT_NOT_NULL(vector_lowered);
  ASSERT_INT_EQ(vector_lowered->src[0]->op, POLY_OP_CAST);
  ASSERT_TRUE(poly_dtype_is_index(poly_dtype_scalar(vector_lowered->src[0]->dtype)));
  ASSERT_INT_EQ(vector_lowered->src[0]->dtype.count, 2);
  ASSERT_INT_EQ(vector_lowered->src[0]->src[0]->op, POLY_OP_STACK);
  ASSERT_TRUE(poly_dtype_eq(
      vector_lowered->src[0]->src[0]->dtype, poly_dtype_vec(POLY_INT64, 2)
  ));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, lower_index_dtype_matches_pinned_param_special_and_index_cast_edges) {
  /* Pinned ops.py:1666-1674: ALU PARAM lowers to int unconditionally,
   * SPECIAL preserves its already-lowered bound while becoming int itself,
   * and INDEX strips only casts whose inner dtype is a concrete integer. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *wide_var = poly_uop0(
      ctx, POLY_OP_DEFINE_VAR, POLY_INDEX, poly_arg_define_var("wide", 0, INT64_C(1) << 40)
  );
  PolyUOp *wide_bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(INT64_C(1) << 40));
  PolyUOp *wide_special =
      poly_uop1(ctx, POLY_OP_SPECIAL, POLY_INDEX, wide_bound, poly_arg_str("gidx0"));

  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *buf = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));
  PolyUOp *weak_three = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(3));
  PolyUOp *long_three = poly_uop1(ctx, POLY_OP_CAST, POLY_INT64, weak_three, poly_arg_none());
  PolyUOp *explicit_long_index =
      poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, buf, long_three, poly_arg_none());

  PolyUOp *bind_var =
      poly_uop0(ctx, POLY_OP_DEFINE_VAR, POLY_INDEX, poly_arg_define_var("bind_var", 0, 10));
  PolyUOp *bind_value =
      poly_uop0(ctx, POLY_OP_DEFINE_VAR, POLY_INDEX, poly_arg_define_var("bind_value", 0, 10));
  PolyUOp *dynamic_bind =
      poly_uop2(ctx, POLY_OP_BIND, POLY_INDEX, bind_var, bind_value, poly_arg_none());
  PolyUOp *cond = poly_uop0(ctx, POLY_OP_DEFINE_VAR, POLY_BOOL, poly_arg_define_var("cond", 0, 1));
  PolyUOp *where_value = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(3));
  PolyUOp *invalid = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_invalid());
  PolyUOp *standalone_invalid_where =
      poly_uop3(ctx, POLY_OP_WHERE, POLY_INDEX, cond, where_value, invalid, poly_arg_none());
  PolyUOp *gated_index =
      poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, buf, standalone_invalid_where, poly_arg_none());

  PolyUOp *gate_b =
      poly_uop0(ctx, POLY_OP_DEFINE_VAR, POLY_BOOL, poly_arg_define_var("gate_b", 0, 1));
  PolyUOp *where_value_two = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(2));
  PolyUOp *same_gate_x =
      poly_uop3(ctx, POLY_OP_WHERE, POLY_INDEX, cond, where_value_two, invalid, poly_arg_none());
  PolyUOp *different_gate_x =
      poly_uop3(ctx, POLY_OP_WHERE, POLY_INDEX, gate_b, where_value_two, invalid, poly_arg_none());
  PolyUOp *same_gate_srcs[3] = {buf, standalone_invalid_where, same_gate_x};
  PolyUOp *different_gate_srcs[3] = {buf, standalone_invalid_where, different_gate_x};
  PolyUOp *mixed_srcs[3] = {buf, standalone_invalid_where, where_value_two};
  PolyUOp *same_gate_image =
      poly_uop(ctx, POLY_OP_INDEX, ptr_f32, same_gate_srcs, 3, poly_arg_none());
  PolyUOp *different_gate_image =
      poly_uop(ctx, POLY_OP_INDEX, ptr_f32, different_gate_srcs, 3, poly_arg_none());
  PolyUOp *mixed_image = poly_uop(ctx, POLY_OP_INDEX, ptr_f32, mixed_srcs, 3, poly_arg_none());

  PolyUOp *vector_lanes[2] = {
      poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(1)),
      poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(2)),
  };
  PolyUOp *weak_vector =
      poly_uop(ctx, POLY_OP_STACK, poly_dtype_vec(POLY_INDEX, 2), vector_lanes, 2, poly_arg_none());
  PolyUOp *vector_index = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, buf, weak_vector, poly_arg_none());
  PolyUOp *vector_var = poly_uop0(
      ctx, POLY_OP_DEFINE_VAR, poly_dtype_vec(POLY_INDEX, 2),
      poly_arg_define_var("vector_var", 0, 10)
  );
  PolyUOp *wide_bind_var = poly_uop0(
      ctx, POLY_OP_DEFINE_VAR, POLY_INDEX, poly_arg_define_var("wide_bind", 0, INT64_C(1) << 40)
  );
  PolyUOp *wide_bind_value =
      poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(INT64_C(1) << 40));
  PolyUOp *wide_bind =
      poly_uop2(ctx, POLY_OP_BIND, POLY_INDEX, wide_bind_var, wide_bind_value, poly_arg_none());
  PolyUOp *malformed_srcs[2] = {
      poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(4)),
      poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(5)),
  };
  PolyUOp *malformed_range = poly_uop(
      ctx, POLY_OP_RANGE, POLY_INDEX, malformed_srcs, 2, poly_arg_range(0, POLY_AXIS_LOOP)
  );
  PolyUOp *malformed_special =
      poly_uop(ctx, POLY_OP_SPECIAL, POLY_INDEX, malformed_srcs, 2, poly_arg_str("gidx0"));

  PolyUOp *roots[14] = {
      wide_var,
      wide_special,
      explicit_long_index,
      dynamic_bind,
      standalone_invalid_where,
      gated_index,
      same_gate_image,
      different_gate_image,
      mixed_image,
      vector_index,
      vector_var,
      wide_bind,
      malformed_range,
      malformed_special,
  };
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, roots, 14, poly_arg_none());
  PolyUOp *lowered = poly_apply_post_index_symbolic_stage(ctx, sink, 1);
  ASSERT_NOT_NULL(lowered);
  ASSERT_INT_EQ(lowered->n_src, 14);

  ASSERT_INT_EQ(lowered->src[0]->op, POLY_OP_DEFINE_VAR);
  ASSERT_TRUE(poly_dtype_eq(lowered->src[0]->dtype, POLY_INT32));

  ASSERT_INT_EQ(lowered->src[1]->op, POLY_OP_SPECIAL);
  ASSERT_TRUE(poly_dtype_eq(lowered->src[1]->dtype, POLY_INT32));
  ASSERT_INT_EQ(lowered->src[1]->n_src, 1);
  ASSERT_INT_EQ(lowered->src[1]->src[0]->op, POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(lowered->src[1]->src[0]->dtype, POLY_INT64));
  ASSERT_INT_EQ(lowered->src[1]->src[0]->arg.i, INT64_C(1) << 40);

  ASSERT_INT_EQ(lowered->src[2]->op, POLY_OP_INDEX);
  ASSERT_INT_EQ(lowered->src[2]->n_src, 2);
  ASSERT_INT_EQ(lowered->src[2]->src[1]->op, POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(lowered->src[2]->src[1]->dtype, POLY_INT64));
  ASSERT_INT_EQ(lowered->src[2]->src[1]->arg.i, 3);

  ASSERT_INT_EQ(lowered->src[3]->op, POLY_OP_BIND);
  ASSERT_TRUE(poly_dtype_eq(lowered->src[3]->dtype, POLY_INDEX));
  ASSERT_INT_EQ(lowered->src[3]->n_src, 2);
  for (int i = 0; i < 2; i++) {
    ASSERT_INT_EQ(lowered->src[3]->src[i]->op, POLY_OP_CAST);
    ASSERT_TRUE(poly_dtype_eq(lowered->src[3]->src[i]->dtype, POLY_INDEX));
    ASSERT_INT_EQ(lowered->src[3]->src[i]->src[0]->op, POLY_OP_DEFINE_VAR);
    ASSERT_TRUE(poly_dtype_eq(lowered->src[3]->src[i]->src[0]->dtype, POLY_INT32));
  }

  ASSERT_INT_EQ(lowered->src[4]->op, POLY_OP_WHERE);
  ASSERT_TRUE(poly_dtype_eq(lowered->src[4]->dtype, POLY_INDEX));
  ASSERT_TRUE(poly_dtype_eq(lowered->src[4]->src[1]->dtype, POLY_INDEX));
  ASSERT_TRUE(poly_dtype_eq(lowered->src[4]->src[2]->dtype, POLY_INDEX));
  ASSERT_INT_EQ(lowered->src[4]->src[2]->arg.kind, POLY_ARG_INVALID);

  ASSERT_INT_EQ(lowered->src[5]->op, POLY_OP_INDEX);
  ASSERT_INT_EQ(lowered->src[5]->src[1]->op, POLY_OP_WHERE);
  ASSERT_TRUE(poly_dtype_eq(lowered->src[5]->src[1]->dtype, POLY_INT32));
  ASSERT_TRUE(poly_dtype_eq(lowered->src[5]->src[1]->src[1]->dtype, POLY_INT32));
  ASSERT_TRUE(poly_dtype_eq(lowered->src[5]->src[1]->src[2]->dtype, POLY_INT32));
  ASSERT_INT_EQ(lowered->src[5]->src[1]->src[2]->arg.kind, POLY_ARG_INVALID);

  ASSERT_INT_EQ(lowered->src[6]->op, POLY_OP_INDEX);
  for (int i = 1; i <= 2; i++) {
    ASSERT_INT_EQ(lowered->src[6]->src[i]->op, POLY_OP_WHERE);
    ASSERT_TRUE(poly_dtype_eq(lowered->src[6]->src[i]->dtype, POLY_INT32));
  }
  ASSERT_TRUE(lowered->src[6]->src[1]->src[0] == lowered->src[6]->src[2]->src[0]);

  ASSERT_INT_EQ(lowered->src[7]->op, POLY_OP_INDEX);
  for (int i = 1; i <= 2; i++) {
    ASSERT_INT_EQ(lowered->src[7]->src[i]->op, POLY_OP_WHERE);
    ASSERT_TRUE(poly_dtype_eq(lowered->src[7]->src[i]->dtype, POLY_INDEX));
  }
  ASSERT_TRUE(lowered->src[7]->src[1]->src[0] != lowered->src[7]->src[2]->src[0]);

  ASSERT_INT_EQ(lowered->src[8]->op, POLY_OP_INDEX);
  ASSERT_INT_EQ(lowered->src[8]->src[1]->op, POLY_OP_WHERE);
  ASSERT_TRUE(poly_dtype_eq(lowered->src[8]->src[1]->dtype, POLY_INDEX));
  ASSERT_INT_EQ(lowered->src[8]->src[2]->op, POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(lowered->src[8]->src[2]->dtype, POLY_INDEX));

  ASSERT_INT_EQ(lowered->src[9]->op, POLY_OP_INDEX);
  ASSERT_INT_EQ(lowered->src[9]->src[1]->op, POLY_OP_STACK);
  ASSERT_TRUE(poly_dtype_eq(lowered->src[9]->src[1]->dtype, poly_dtype_vec(POLY_INT32, 2)));

  ASSERT_INT_EQ(lowered->src[10]->op, POLY_OP_DEFINE_VAR);
  ASSERT_TRUE(poly_dtype_eq(lowered->src[10]->dtype, POLY_INT32));

  ASSERT_INT_EQ(lowered->src[11]->op, POLY_OP_BIND);
  ASSERT_TRUE(poly_dtype_eq(lowered->src[11]->dtype, POLY_INT32));
  ASSERT_INT_EQ(lowered->src[11]->src[0]->op, POLY_OP_DEFINE_VAR);
  ASSERT_TRUE(poly_dtype_eq(lowered->src[11]->src[0]->dtype, POLY_INT32));
  ASSERT_INT_EQ(lowered->src[11]->src[1]->op, POLY_OP_CONST);
  ASSERT_TRUE(poly_dtype_eq(lowered->src[11]->src[1]->dtype, POLY_INT64));
  ASSERT_INT_EQ(lowered->src[11]->src[1]->arg.i, INT64_C(1) << 40);

  ASSERT_INT_EQ(lowered->src[12]->op, POLY_OP_RANGE);
  ASSERT_TRUE(poly_dtype_eq(lowered->src[12]->dtype, POLY_INDEX));
  ASSERT_INT_EQ(lowered->src[12]->n_src, 2);
  ASSERT_INT_EQ(lowered->src[13]->op, POLY_OP_SPECIAL);
  ASSERT_TRUE(poly_dtype_eq(lowered->src[13]->dtype, POLY_INDEX));
  ASSERT_INT_EQ(lowered->src[13]->n_src, 2);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, linearize_webgpu_reduce_emits_tinygrad_sized_shared_barrier) {
  /* tinygrad emits shared memory for this reduce, but with one local axis:
   * @workgroup_size(16). The regression is growing an extra local reduce axis
   * and producing @workgroup_size(16,256,1). */
  PolyCtx *ctx = poly_ctx_new();

  PolyUOp *a = poly_buffer_f32(ctx, 4096);
  PolyUOp *out = poly_buffer_f32(ctx, 1);
  PolyUOp *sum = poly_sum_reduce(ctx, a, 0, 0);
  PolyUOp *store = poly_store_val(ctx, out, sum);
  PolyUOp *sink = poly_sink1(ctx, store);

  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_NOT_NULL(sched);

  bool found = false;
  for (int i = 0; i < sched->template->n_calls; i++) {
    int n_lin = 0;
    PolyUOp **lin = poly_linearize_webgpu(ctx, poly_schedule_call_body(sched, i), &n_lin);
    ASSERT_NOT_NULL(lin);
    char *wgsl = poly_render_wgsl(lin, n_lin, "reduce_webgpu");
    ASSERT_NOT_NULL(wgsl);
    int dims[3];
    int prod = wgsl_workgroup_product(wgsl, dims);
    ASSERT_TRUE(prod <= 256);
    ASSERT_INT_EQ(dims[0], 16);
    ASSERT_INT_EQ(dims[1], 1);
    ASSERT_INT_EQ(dims[2], 1);
    if (strstr(wgsl, "var<workgroup>") && strstr(wgsl, "workgroupBarrier();")) found = true;
    free(wgsl);
    free(lin);
  }
  ASSERT_TRUE(found);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, linearize_webgpu_qwen_downproj_workgroup_matches_tinygrad) {
  /* Probe parity with tinygrad_latest:
   *   Tensor.empty((25,3072)).matmul(Tensor.empty((1024,3072)).T)
   * renders @workgroup_size(16), not @workgroup_size(16,256). */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = make_shaped_f32_buf(ctx, (int64_t[]){25, 3072}, 2);
  PolyUOp *w = make_shaped_f32_buf(ctx, (int64_t[]){1024, 3072}, 2);
  PolyUOp *wt = poly_permute(ctx, w, (int64_t[]){1, 0}, 2);
  PolyUOp *y = poly_dot(ctx, x, wt);
  PolyUOp *out = poly_buffer_f32(ctx, 25 * 1024);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, y));

  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_NOT_NULL(sched);
  ASSERT_INT_EQ(sched->template->n_calls, 1);

  int n_lin = 0;
  PolyUOp **lin = poly_linearize_webgpu(ctx, poly_schedule_call_body(sched, 0), &n_lin);
  ASSERT_NOT_NULL(lin);
  char *wgsl = poly_render_wgsl(lin, n_lin, "qwen_downproj");
  ASSERT_NOT_NULL(wgsl);

  int dims[3];
  int prod = wgsl_workgroup_product(wgsl, dims);
  ASSERT_TRUE(prod <= 256);
  ASSERT_INT_EQ(dims[0], 16);
  ASSERT_INT_EQ(dims[1], 1);
  ASSERT_INT_EQ(dims[2], 1);
  ASSERT_TRUE(strstr(wgsl, "@workgroup_size(16,256") == NULL);

  free(wgsl);
  free(lin);
  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, webgpu_vector_store_target_survives_add_loads_until_devectorize) {
  /* tinygrad pm_add_loads only adds LOAD around scalar value indexes and removes
   * STORE(LOAD(ptr), val). It must not turn STACK(ptr indexes) store targets
   * into STACK(LOAD(...)) value expressions; load_store_folding needs pointer
   * targets to split vector stores legally before WGSL rendering. */
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, 256, POLY_ADDR_LOCAL);
  PolyUOp *smem = poly_uop0(ctx, POLY_OP_DEFINE_LOCAL, ptr_f32, poly_arg_int(0));
  PolyUOp *idx = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(7));
  PolyUOp *ptr = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, smem, idx, poly_arg_none());

  PolyDType ptr_vec = ptr_f32;
  ptr_vec.vcount = 4;
  PolyUOp *ptrs[4] = {ptr, ptr, ptr, ptr};
  PolyUOp *target = poly_uop(ctx, POLY_OP_VECTORIZE, ptr_vec, ptrs, 4, poly_arg_none());

  PolyUOp *vals[4] = {
      poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0)),
      poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(2.0)),
      poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(3.0)),
      poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(4.0)),
  };
  PolyUOp *value =
      poly_uop(ctx, POLY_OP_VECTORIZE, poly_dtype_vec(POLY_FLOAT32, 4), vals, 4, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, target, value, poly_arg_none());

  PolyUOp *after_add_loads = poly_graph_rewrite(ctx, store, poly_pm_add_loads_pass());
  ASSERT_TRUE(after_add_loads->op == POLY_OP_STORE);
  ASSERT_TRUE(after_add_loads->src[0]->op == POLY_OP_VECTORIZE);
  ASSERT_TRUE(after_add_loads->src[0]->src[0]->op == POLY_OP_INDEX);

  PolyRendererCaps caps = {.max_vec_width = 1};
  PolyUOp *after_devec = poly_apply_devectorize_stage(ctx, after_add_loads, 1, caps);
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, after_devec, &n_topo);
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op != POLY_OP_STORE || topo[i]->n_src < 1) continue;
    ASSERT_TRUE(topo[i]->src[0]->op != POLY_OP_VECTORIZE);
    ASSERT_TRUE(topo[i]->src[0]->op != POLY_OP_VCONST);
  }

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, full_rewrite_post_index_lowering_narrows_i64_addressing) {
  VecKernel k = make_vec_copy_with_weak_index_expr(32);
  PolyRewriteOpts opts = {
      .optimize = false,
      .devectorize = 1,
      .caps = {.max_vec_width = 1},
      .device = POLY_DEVICE_WEBGPU,
  };

  PolyUOp *rewritten = poly_full_rewrite_to_sink_ex(k.ctx, k.sink, opts);
  ASSERT_NOT_NULL(rewritten);

  int n_topo = 0;
  PolyUOp **topo = poly_toposort(k.ctx, rewritten, &n_topo);
  ASSERT_TRUE(n_topo > 0);
  ASSERT_INT_EQ(count_indexes_with_i64_addr(topo, n_topo), 0);

  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(codegen, full_rewrite_lowers_weakint_like_address_dtype) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyDType weak_like = POLY_INDEX;
  weak_like.bitsize = 144;

  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(1));
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(32));
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_INDEX, bound, poly_arg_range(0, POLY_AXIS_LOOP));
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, weak_like, poly_arg_int(1));
  PolyUOp *addr = poly_uop2(ctx, POLY_OP_ADD, weak_like, range, one, poly_arg_none());
  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p0, addr, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p1, range, poly_arg_none());
  PolyUOp *load0 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx0, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx1, load0, poly_arg_none());
  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  int n_lin = 0;
  PolyUOp **lin = poly_linearize(ctx, sink, &n_lin);
  ASSERT_NOT_NULL(lin);
  ASSERT_INT_EQ(count_weakint_named_nodes(lin, n_lin), 0);

  char *src = poly_render_c(lin, n_lin, "index_like_addr");
  ASSERT_NOT_NULL(src);
  ASSERT_TRUE(strstr(src, " weakint ") == NULL);

  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(codegen, linearize_webgpu_narrows_i64_addressing) {
  VecKernel k = make_vec_copy_with_weak_index_expr(32);
  int n_lin = 0;
  PolyUOp **lin = poly_linearize_webgpu(k.ctx, k.sink, &n_lin);
  ASSERT_NOT_NULL(lin);
  ASSERT_TRUE(n_lin > 0);
  ASSERT_INT_EQ(count_indexes_with_i64_addr(lin, n_lin), 0);

  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

/* Unary op end-to-end */

TEST(codegen, e2e_neg) {
  /* b[i] = -a[i] for i in 0..5 */
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);

  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(1));

  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(6));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));

  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p0, range, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p1, range, poly_arg_none());

  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx0, poly_arg_none());
  PolyUOp *neg = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, load, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx1, neg, poly_arg_none());

  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  int n;
  PolyUOp **lin = poly_linearize(ctx, sink, &n);
  char *src = poly_render_c(lin, n, "vecneg");

  PolyProgram *prog = poly_compile_c(src, "vecneg");
  ASSERT_NOT_NULL(prog);

  float a[6] = {1.0f, -2.5f, 3.14f, 0.0f, -100.0f, 42.0f};
  float b[6] = {0};

  void *args[2] = {a, b};
  poly_program_call(prog, args, 2);

  for (int i = 0; i < 6; i++) {
    ASSERT_FLOAT_EQ(b[i], -a[i], 1e-6);
  }

  poly_program_destroy(prog);
  free(src);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}
