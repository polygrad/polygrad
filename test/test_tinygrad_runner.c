#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/sched.h"
#include "../src/codegen.h"
#include "../src/rangeify.h"
#include "../src/frontend.h"

/* ── JSON helpers ─────────────────────────────────────────────────────── */

static void json_float_array(const float *data, int n) {
  printf("[");
  for (int i = 0; i < n; i++) {
    if (i) printf(",");
    printf("%.9g", data[i]);
  }
  printf("]");
}

static void json_ops_array(PolyUOp **lin, int n_lin) {
  printf("[");
  for (int i = 0; i < n_lin; i++) {
    if (i) printf(",");
    printf("\"%s\"", poly_op_name(lin[i]->op));
  }
  printf("]");
}

/* ── run_and_report: schedule → linearize → render → execute → JSON ── */

typedef struct {
  PolyUOp *buffer;
  void *data;
} ParityBinding;

static int env_enabled(const char *name) {
  const char *v = getenv(name);
  return v && strcmp(v, "0") != 0 && strcmp(v, "false") != 0 && strcmp(v, "False") != 0;
}

/* tinygrad can schedule pure movement outputs as COPY-only work with no SINK
 * kernels. For full parity mode, optionally mirror that at report level while
 * still executing the generated kernels for value correctness. */
static int graph_has_compute_ops(PolyCtx *ctx, PolyUOp *tensor_sink) {
  int n = 0;
  PolyUOp **nodes = poly_toposort(ctx, tensor_sink, &n);
  for (int i = 0; i < n; i++) {
    PolyOps op = nodes[i]->op;
    if (poly_opset_has(POLY_GROUP_ALU, op)) return 1;
    if (op == POLY_OP_REDUCE_AXIS || op == POLY_OP_REDUCE || op == POLY_OP_ALLREDUCE) return 1;
    if (op == POLY_OP_WMMA || op == POLY_OP_MULACC) return 1;
  }
  return 0;
}

/* Schedules, linearizes, renders, executes, and emits JSON with both
 * kernel ops and output data. For kernel-only cases, pass NULL/0 for
 * out_data/out_n and NULL/0 for bindings/n_bindings. */
static int run_and_report(PolyCtx *ctx, PolyUOp *tensor_sink,
                          ParityBinding *bindings, int n_bindings,
                          float *out_data, int out_n) {
  /* 1. Schedule */
  PolyScheduleResult sr = poly_schedule_v2(ctx, tensor_sink);
  if (sr.n_kernels < 1) {
    fprintf(stderr, "parity: schedule produced 0 kernels\n");
    poly_schedule_result_free(&sr);
    return 0;
  }

  int use_cuda = env_enabled("POLY_PARITY_CUDA");

  /* 2. Per-kernel: linearize + collect op names */
  PolyUOp ***all_lin = calloc(sr.n_kernels, sizeof(PolyUOp **));
  int *all_n_lin = calloc(sr.n_kernels, sizeof(int));

  for (int k = 0; k < sr.n_kernels; k++) {
#ifdef POLY_HAS_CUDA
    if (use_cuda) {
      all_lin[k] = poly_linearize_cuda(ctx, sr.kernels[k], &all_n_lin[k]);
    } else
#endif
    {
      (void)use_cuda;
      all_lin[k] = poly_linearize(ctx, sr.kernels[k], &all_n_lin[k]);
    }
    if (!all_lin[k]) {
      fprintf(stderr, "parity: linearize failed for kernel %d\n", k);
      for (int j = 0; j < k; j++) free(all_lin[j]);
      free(all_lin);
      free(all_n_lin);
      poly_schedule_result_free(&sr);
      return 0;
    }
  }

  /* 3. Execute (only if bindings provided) */
  int ok = 1;
  if (bindings && n_bindings > 0) {
#ifdef POLY_HAS_CUDA
    if (use_cuda) {
      /* CUDA path: use poly_realize_cuda + poly_cuda_copyback */
      PolyBufferBinding *bb = malloc(n_bindings * sizeof(PolyBufferBinding));
      for (int j = 0; j < n_bindings; j++) {
        bb[j].buffer = bindings[j].buffer;
        bb[j].data = bindings[j].data;
      }
      ok = (poly_realize_cuda(ctx, tensor_sink, bb, n_bindings) == 0);
      if (ok) poly_cuda_copyback(bb, n_bindings);
      free(bb);
    } else
#endif
    {
      /* CPU path: per-kernel render_c → compile_c → program_call */
      int n_int = sr.n_intermediates;
      void **intermediates = NULL;
      if (n_int > 0) {
        intermediates = calloc(n_int, sizeof(void *));
        for (int b = 0; b < n_int; b++)
          intermediates[b] = calloc(sr.intermediate_sizes[b], sizeof(float));
      }

      for (int k = 0; k < sr.n_kernels && ok; k++) {
        int n_params = sr.kernel_n_params[k];
        void **args = calloc(n_params, sizeof(void *));

        for (int i = 0; i < n_params; i++) {
          PolyUOp *buf = sr.param_to_buf[k][i];
          if (buf->op == POLY_OP_BUFFER) {
            for (int j = 0; j < n_bindings; j++) {
              if (bindings[j].buffer == buf) {
                args[i] = bindings[j].data;
                break;
              }
            }
            /* New split path: intermediate buffers are BUFFER+LUNIQUE, not BUFFERIZE */
            if (!args[i]) {
              for (int b = 0; b < n_int; b++) {
                if (sr.param_to_buf[b] && sr.param_to_buf[b][0] == buf) {
                  args[i] = intermediates[b];
                  break;
                }
              }
            }
          } else if (buf->op == POLY_OP_BUFFERIZE) {
            for (int b = 0; b < n_int; b++) {
              if (sr.param_to_buf[b] && sr.param_to_buf[b][0] == buf) {
                args[i] = intermediates[b];
                break;
              }
            }
          }
          if (!args[i]) {
            fprintf(stderr, "parity: no binding for param %d in kernel %d\n", i, k);
            ok = 0;
          }
        }

        if (ok) {
          char fn_name[32];
          snprintf(fn_name, sizeof(fn_name), "par_k%d", k);
          char *src = poly_render_c(all_lin[k], all_n_lin[k], fn_name);
          if (!src) { ok = 0; free(args); continue; }
          PolyProgram *prog = poly_compile_c(src, fn_name);
          free(src);
          if (!prog) { ok = 0; free(args); continue; }
          poly_program_call(prog, args, n_params);
          poly_program_destroy(prog);
        }
        free(args);
      }

      if (intermediates) {
        for (int b = 0; b < n_int; b++) free(intermediates[b]);
        free(intermediates);
      }
    }
  }

  /* 4. Emit JSON */
  if (ok) {
    int movement_as_copy = env_enabled("POLY_PARITY_MOVEMENT_AS_COPY") &&
                           !graph_has_compute_ops(ctx, tensor_sink);
    if (movement_as_copy) {
      printf("{\"n_kernels\":0,\"kernels\":[]");
    } else {
      printf("{\"n_kernels\":%d,\"kernels\":[", sr.n_kernels);
      for (int k = 0; k < sr.n_kernels; k++) {
        if (k) printf(",");
        printf("{\"ops\":");
        json_ops_array(all_lin[k], all_n_lin[k]);
        printf("}");
      }
      printf("]");
    }
    if (out_data && out_n > 0) {
      printf(",\"n\":%d,\"data\":", out_n);
      json_float_array(out_data, out_n);
    }
    printf("}\n");
  }

  /* Cleanup */
  for (int k = 0; k < sr.n_kernels; k++) free(all_lin[k]);
  free(all_lin);
  free(all_n_lin);
  poly_schedule_result_free(&sr);
  return ok;
}

/* ── Original 16 test cases ───────────────────────────────────────── */

static int case_vecadd(void) {
  int N = 16;
  float a_d[16], b_d[16], c_d[16];
  for (int i = 0; i < N; i++) {
    a_d[i] = (float)(i + 1);
    b_d[i] = (float)(i + 1) * 0.5f;
    c_d[i] = 0.0f;
  }

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  ParityBinding bindings[] = {{c, c_d}, {a, a_d}, {b, b_d}};
  int ok = run_and_report(ctx, sink, bindings, 3, c_d, N);
  poly_ctx_destroy(ctx);
  return ok;
}

static int case_chain(void) {
  int N = 8;
  float a_d[8], b_d[8], c_d[8], d_d[8];
  for (int i = 0; i < N; i++) {
    a_d[i] = (float)(i + 1);
    b_d[i] = 2.0f;
    c_d[i] = 0.5f;
    d_d[i] = 0.0f;
  }

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *d = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, b, poly_arg_none());
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, add, c, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, d, mul, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  ParityBinding bindings[] = {{d, d_d}, {a, a_d}, {b, b_d}, {c, c_d}};
  int ok = run_and_report(ctx, sink, bindings, 4, d_d, N);
  poly_ctx_destroy(ctx);
  return ok;
}

static int case_broadcast_scalar(void) {
  int N = 8;
  float a_d[8], c_d[8];
  for (int i = 0; i < N; i++) {
    a_d[i] = (float)(i + 1);
    c_d[i] = 0.0f;
  }

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(2.0));
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, two, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  ParityBinding bindings[] = {{c, c_d}, {a, a_d}};
  int ok = run_and_report(ctx, sink, bindings, 2, c_d, N);
  poly_ctx_destroy(ctx);
  return ok;
}

static int case_reduce_sum_axis1(void) {
  float a_d[12], c_d[4] = {0};
  for (int i = 0; i < 12; i++) a_d[i] = (float)(i + 1);

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 12);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, 4);
  int64_t rdims[] = {4, 3};
  PolyUOp *a2d = poly_reshape(ctx, a, rdims, 2);
  int64_t axes[] = {1};
  PolyUOp *sum = poly_reduce_axis(ctx, POLY_OP_ADD, a2d, axes, 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, sum, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  ParityBinding bindings[] = {{c, c_d}, {a, a_d}};
  int ok = run_and_report(ctx, sink, bindings, 2, c_d, 4);
  poly_ctx_destroy(ctx);
  return ok;
}

static int case_reduce_scalar_chain(void) {
  int N = 8;
  float a_d[8], b_d[8], c_d[8];
  for (int i = 0; i < N; i++) {
    a_d[i] = (float)(i + 1);
    b_d[i] = (float)(10 + i);
    c_d[i] = 0.0f;
  }

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, N);
  int64_t axes[] = {0};
  PolyUOp *sum = poly_reduce_axis(ctx, POLY_OP_ADD, a, axes, 1);
  /* Match tinygrad broadcast graph shape: REDUCE_AXIS(1) -> RESHAPE(1) -> EXPAND(N) */
  int64_t one_shape[] = {1};
  PolyUOp *sum_1 = poly_reshape(ctx, sum, one_shape, 1);
  int64_t exp_shape[] = {N};
  PolyUOp *sum_exp = poly_expand(ctx, sum_1, exp_shape, 1);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, sum_exp, b, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  ParityBinding bindings[] = {{c, c_d}, {a, a_d}, {b, b_d}};
  int ok = run_and_report(ctx, sink, bindings, 3, c_d, N);
  poly_ctx_destroy(ctx);
  return ok;
}

static int case_reduce_vector_chain(void) {
  int N = 12;
  float a_d[12], b_d[12], c_d[12];
  for (int i = 0; i < N; i++) {
    a_d[i] = (float)(i + 1);
    b_d[i] = (float)(100 + i);
    c_d[i] = 0.0f;
  }

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, N);
  int64_t dims2d[] = {4, 3};
  PolyUOp *a2d = poly_reshape(ctx, a, dims2d, 2);
  PolyUOp *b2d = poly_reshape(ctx, b, dims2d, 2);
  int64_t axes[] = {1};
  PolyUOp *sum = poly_reduce_axis(ctx, POLY_OP_ADD, a2d, axes, 1);
  int64_t expd[] = {4, 3};
  PolyUOp *sum_exp = poly_expand(ctx, sum, expd, 2);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, sum_exp, b2d, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, add, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  ParityBinding bindings[] = {{c, c_d}, {a, a_d}, {b, b_d}};
  int ok = run_and_report(ctx, sink, bindings, 3, c_d, N);
  poly_ctx_destroy(ctx);
  return ok;
}

static int case_shared_scalar_reduce_branches(void) {
  int N = 8;
  float a_d[8], c0_d[8], e0_d[8], out_d[16];
  float *out_c = out_d;
  float *out_e = out_d + 8;
  for (int i = 0; i < N; i++) {
    a_d[i] = (float)(i + 1);
    c0_d[i] = (float)(10 + i);
    e0_d[i] = (float)(20 + i);
    out_c[i] = 0.0f;
    out_e[i] = 0.0f;
  }

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *c0 = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *e0 = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *oc = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *oe = poly_buffer(ctx, POLY_FLOAT32, N);
  int64_t axes[] = {0};
  PolyUOp *sum = poly_reduce_axis(ctx, POLY_OP_ADD, a, axes, 1);
  /* Match tinygrad: shared expanded scalar consumed by both branches. */
  int64_t one_shape[] = {1};
  PolyUOp *sum_1 = poly_reshape(ctx, sum, one_shape, 1);
  int64_t exp_shape[] = {N};
  PolyUOp *sum_exp = poly_expand(ctx, sum_1, exp_shape, 1);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, sum_exp, c0, poly_arg_none());
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, sum_exp, e0, poly_arg_none());
  PolyUOp *store_c = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, oc, add, poly_arg_none());
  PolyUOp *store_e = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, oe, mul, poly_arg_none());
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID,
      (PolyUOp *[]){store_c, store_e}, 2, poly_arg_none());

  ParityBinding bindings[] = {{oc, out_c}, {oe, out_e}, {a, a_d}, {c0, c0_d}, {e0, e0_d}};
  int ok = run_and_report(ctx, sink, bindings, 5, out_d, 16);
  poly_ctx_destroy(ctx);
  return ok;
}

static int case_permute_2d(void) {
  float a_d[12], c_d[12];
  for (int i = 0; i < 12; i++) {
    a_d[i] = (float)i;
    c_d[i] = -1.0f;
  }

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 12);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, 12);
  int64_t rdims[] = {3, 4};
  PolyUOp *a2d = poly_reshape(ctx, a, rdims, 2);
  int64_t perm[] = {1, 0};
  PolyUOp *t = poly_permute(ctx, a2d, perm, 2);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, t, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  ParityBinding bindings[] = {{c, c_d}, {a, a_d}};
  int ok = run_and_report(ctx, sink, bindings, 2, c_d, 12);
  poly_ctx_destroy(ctx);
  return ok;
}

static int case_shrink_2d(void) {
  float a_d[12], c_d[6];
  for (int i = 0; i < 12; i++) a_d[i] = (float)i;
  for (int i = 0; i < 6; i++) c_d[i] = -1.0f;

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 12);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, 6);
  int64_t rdims[] = {4, 3};
  PolyUOp *a2d = poly_reshape(ctx, a, rdims, 2);
  int64_t pairs[][2] = {{1, 3}, {0, 3}};
  PolyUOp *s = poly_shrink(ctx, a2d, pairs, 2);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, s, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  ParityBinding bindings[] = {{c, c_d}, {a, a_d}};
  int ok = run_and_report(ctx, sink, bindings, 2, c_d, 6);
  poly_ctx_destroy(ctx);
  return ok;
}

static int case_pad_2d(void) {
  float a_d[] = {1, 2, 3, 4, 5, 6};
  float c_d[20];
  for (int i = 0; i < 20; i++) c_d[i] = -1.0f;

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 6);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, 20);
  int64_t rdims[] = {2, 3};
  PolyUOp *a2d = poly_reshape(ctx, a, rdims, 2);
  int64_t pairs[][2] = {{1, 1}, {1, 1}};
  PolyUOp *p = poly_pad(ctx, a2d, pairs, 2);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, p, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  ParityBinding bindings[] = {{c, c_d}, {a, a_d}};
  int ok = run_and_report(ctx, sink, bindings, 2, c_d, 20);
  poly_ctx_destroy(ctx);
  return ok;
}

static int case_chain_pad_flip(void) {
  float a_d[] = {1, 2, 3};
  float c_d[5] = {-1, -1, -1, -1, -1};

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 3);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, 5);
  int64_t pad_pairs[][2] = {{1, 1}};
  PolyUOp *p = poly_pad(ctx, a, pad_pairs, 1);
  int64_t axes[] = {0};
  PolyUOp *f = poly_flip(ctx, p, axes, 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, f, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  ParityBinding bindings[] = {{c, c_d}, {a, a_d}};
  int ok = run_and_report(ctx, sink, bindings, 2, c_d, 5);
  poly_ctx_destroy(ctx);
  return ok;
}

/* ── Autograd cases (original) ────────────────────────────────────── */

static int case_grad_mul_sum(void) {
  int N = 8;
  float x_d[8], gx_d[8];
  for (int i = 0; i < N; i++) {
    x_d[i] = (float)(i - 3);
    gx_d[i] = 0.0f;
  }

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, x, x, poly_arg_none());
  int64_t axes[] = {0};
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, mul, axes, 1);
  PolyUOp *gx = poly_grad(ctx, loss, x);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, gx, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  ParityBinding bindings[] = {{out, gx_d}, {x, x_d}};
  int ok = run_and_report(ctx, sink, bindings, 2, gx_d, N);
  poly_ctx_destroy(ctx);
  return ok;
}

static int case_grad_exp2_sum(void) {
  int N = 6;
  float x_d[6] = {-2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f};
  float gx_d[6] = {0};

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *e = poly_uop1(ctx, POLY_OP_EXP2, POLY_FLOAT32, x, poly_arg_none());
  int64_t axes[] = {0};
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, e, axes, 1);
  PolyUOp *gx = poly_grad(ctx, loss, x);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, gx, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  ParityBinding bindings[] = {{out, gx_d}, {x, x_d}};
  int ok = run_and_report(ctx, sink, bindings, 2, gx_d, N);
  poly_ctx_destroy(ctx);
  return ok;
}

static int case_grad_fdiv_sum_x(void) {
  int N = 6;
  float x_d[6] = {1.0f, 2.0f, 3.0f, -1.0f, -2.0f, 4.0f};
  float y_d[6] = {2.0f, 4.0f, -2.0f, 5.0f, -3.0f, 8.0f};
  float gx_d[6] = {0};

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *y = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *q = poly_uop2(ctx, POLY_OP_FDIV, POLY_FLOAT32, x, y, poly_arg_none());
  int64_t axes[] = {0};
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, q, axes, 1);
  PolyUOp *gx = poly_grad(ctx, loss, x);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, gx, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  ParityBinding bindings[] = {{out, gx_d}, {x, x_d}, {y, y_d}};
  int ok = run_and_report(ctx, sink, bindings, 3, gx_d, N);
  poly_ctx_destroy(ctx);
  return ok;
}

static int case_grad_fdiv_sum_y(void) {
  int N = 6;
  float x_d[6] = {1.0f, 2.0f, 3.0f, -1.0f, -2.0f, 4.0f};
  float y_d[6] = {2.0f, 4.0f, -2.0f, 5.0f, -3.0f, 8.0f};
  float gy_d[6] = {0};

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *y = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *q = poly_uop2(ctx, POLY_OP_FDIV, POLY_FLOAT32, x, y, poly_arg_none());
  int64_t axes[] = {0};
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, q, axes, 1);
  PolyUOp *gy = poly_grad(ctx, loss, y);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, gy, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  ParityBinding bindings[] = {{out, gy_d}, {x, x_d}, {y, y_d}};
  int ok = run_and_report(ctx, sink, bindings, 3, gy_d, N);
  poly_ctx_destroy(ctx);
  return ok;
}

static int case_grad_chain_movement(void) {
  int N = 6;
  float x_d[6], gx_d[6];
  for (int i = 0; i < N; i++) {
    x_d[i] = (float)(i + 1);
    gx_d[i] = 0.0f;
  }

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer(ctx, POLY_FLOAT32, N);
  int64_t rshape[] = {2, 3};
  int64_t perm[] = {1, 0};
  int64_t axes[] = {0, 1};
  PolyUOp *xr = poly_reshape(ctx, x, rshape, 2);
  PolyUOp *xp = poly_permute(ctx, xr, perm, 2);
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, xp, axes, 2);
  PolyUOp *gx = poly_grad(ctx, loss, x);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, gx, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  ParityBinding bindings[] = {{out, gx_d}, {x, x_d}};
  int ok = run_and_report(ctx, sink, bindings, 2, gx_d, N);
  poly_ctx_destroy(ctx);
  return ok;
}

/* ── Tier 1: Simple elementwise ──────────────────────────────────── */

static int case_neg_1d(void) {
  int N = 8;
  float x_d[8], c_d[8];
  for (int i = 0; i < N; i++) {
    x_d[i] = (float)(i - 3);
    c_d[i] = 0.0f;
  }

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *neg = poly_uop1(ctx, POLY_OP_NEG, POLY_FLOAT32, x, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, neg, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  ParityBinding bindings[] = {{c, c_d}, {x, x_d}};
  int ok = run_and_report(ctx, sink, bindings, 2, c_d, N);
  poly_ctx_destroy(ctx);
  return ok;
}

static int case_exp2_1d(void) {
  int N = 6;
  float x_d[6] = {-2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f};
  float c_d[6] = {0};

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *e = poly_uop1(ctx, POLY_OP_EXP2, POLY_FLOAT32, x, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, e, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  ParityBinding bindings[] = {{c, c_d}, {x, x_d}};
  int ok = run_and_report(ctx, sink, bindings, 2, c_d, N);
  poly_ctx_destroy(ctx);
  return ok;
}

static int case_sqrt_1d(void) {
  int N = 6;
  float x_d[6] = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 0.25f};
  float c_d[6] = {0};

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *s = poly_uop1(ctx, POLY_OP_SQRT, POLY_FLOAT32, x, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, s, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  ParityBinding bindings[] = {{c, c_d}, {x, x_d}};
  int ok = run_and_report(ctx, sink, bindings, 2, c_d, N);
  poly_ctx_destroy(ctx);
  return ok;
}

static int case_mul_1d(void) {
  int N = 8;
  float x_d[8], y_d[8], c_d[8];
  for (int i = 0; i < N; i++) {
    x_d[i] = (float)(i + 1);
    y_d[i] = (float)(i + 1) * 0.5f;
    c_d[i] = 0.0f;
  }

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *y = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, x, y, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, mul, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  ParityBinding bindings[] = {{c, c_d}, {x, x_d}, {y, y_d}};
  int ok = run_and_report(ctx, sink, bindings, 3, c_d, N);
  poly_ctx_destroy(ctx);
  return ok;
}

static int case_where_1d(void) {
  int N = 8;
  float x_d[8], c_d[8];
  for (int i = 0; i < N; i++) {
    x_d[i] = (float)(i - 3);
    c_d[i] = 0.0f;
  }

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(0.0));
  PolyUOp *cmp = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, zero, x, poly_arg_none());
  PolyUOp *w = poly_uop3(ctx, POLY_OP_WHERE, POLY_FLOAT32, cmp, x, zero, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, w, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  ParityBinding bindings[] = {{c, c_d}, {x, x_d}};
  int ok = run_and_report(ctx, sink, bindings, 2, c_d, N);
  poly_ctx_destroy(ctx);
  return ok;
}

/* ── Tier 2: Reductions ──────────────────────────────────────────── */

static int case_reduce_sum_all(void) {
  int N = 12;
  float a_d[12], c_d[1] = {0};
  for (int i = 0; i < N; i++) a_d[i] = (float)(i + 1);

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, 1);
  int64_t axes[] = {0};
  PolyUOp *sum = poly_reduce_axis(ctx, POLY_OP_ADD, a, axes, 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, sum, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  ParityBinding bindings[] = {{c, c_d}, {a, a_d}};
  int ok = run_and_report(ctx, sink, bindings, 2, c_d, 1);
  poly_ctx_destroy(ctx);
  return ok;
}

static int case_reduce_max_1d(void) {
  int N = 8;
  float a_d[8], c_d[1] = {0};
  for (int i = 0; i < N; i++) a_d[i] = (float)(i * 2 - 7);

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, 1);
  int64_t axes[] = {0};
  PolyUOp *mx = poly_reduce_axis(ctx, POLY_OP_MAX, a, axes, 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, mx, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  ParityBinding bindings[] = {{c, c_d}, {a, a_d}};
  int ok = run_and_report(ctx, sink, bindings, 2, c_d, 1);
  poly_ctx_destroy(ctx);
  return ok;
}

/* ── Tier 3: Composed ────────────────────────────────────────────── */

static int case_reshape_reduce(void) {
  float a_d[12], c_d[4] = {0};
  for (int i = 0; i < 12; i++) a_d[i] = (float)(i + 1);

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 12);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, 4);
  int64_t rdims[] = {4, 3};
  PolyUOp *a2d = poly_reshape(ctx, a, rdims, 2);
  int64_t axes[] = {1};
  PolyUOp *sum = poly_reduce_axis(ctx, POLY_OP_ADD, a2d, axes, 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, sum, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  ParityBinding bindings[] = {{c, c_d}, {a, a_d}};
  int ok = run_and_report(ctx, sink, bindings, 2, c_d, 4);
  poly_ctx_destroy(ctx);
  return ok;
}

static int case_expand_alu_reduce(void) {
  float a_d[4] = {1, 2, 3, 4};
  float b_d[12], c_d[4] = {0};
  for (int i = 0; i < 12; i++) b_d[i] = (float)(i + 1) * 0.1f;

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 4);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, 12);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, 4);
  int64_t dims[] = {4, 3};
  int64_t a_dims[] = {4, 1};
  PolyUOp *a2d = poly_reshape(ctx, a, a_dims, 2);
  PolyUOp *a_exp = poly_expand(ctx, a2d, dims, 2);
  PolyUOp *b2d = poly_reshape(ctx, b, dims, 2);
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a_exp, b2d, poly_arg_none());
  int64_t axes[] = {1};
  PolyUOp *sum = poly_reduce_axis(ctx, POLY_OP_ADD, add, axes, 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, sum, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  ParityBinding bindings[] = {{c, c_d}, {a, a_d}, {b, b_d}};
  int ok = run_and_report(ctx, sink, bindings, 3, c_d, 4);
  poly_ctx_destroy(ctx);
  return ok;
}

static int case_multi_movement(void) {
  float a_d[12], c_d[6];
  for (int i = 0; i < 12; i++) a_d[i] = (float)(i + 1);
  for (int i = 0; i < 6; i++) c_d[i] = -1.0f;

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 12);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, 6);
  int64_t rdims[] = {3, 4};
  PolyUOp *a2d = poly_reshape(ctx, a, rdims, 2);
  int64_t perm[] = {1, 0};
  PolyUOp *p = poly_permute(ctx, a2d, perm, 2);
  int64_t pairs[][2] = {{0, 2}, {0, 3}};
  PolyUOp *s = poly_shrink(ctx, p, pairs, 2);
  int64_t faxes[] = {0};
  PolyUOp *f = poly_flip(ctx, s, faxes, 1);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, f, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  ParityBinding bindings[] = {{c, c_d}, {a, a_d}};
  int ok = run_and_report(ctx, sink, bindings, 2, c_d, 6);
  poly_ctx_destroy(ctx);
  return ok;
}

/* ── Tier 4: Autograd (new) ──────────────────────────────────────── */

static int case_grad_log2_sum(void) {
  int N = 6;
  float x_d[6] = {0.25f, 0.5f, 1.0f, 2.0f, 4.0f, 8.0f};
  float gx_d[6] = {0};

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *l = poly_uop1(ctx, POLY_OP_LOG2, POLY_FLOAT32, x, poly_arg_none());
  int64_t axes[] = {0};
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, l, axes, 1);
  PolyUOp *gx = poly_grad(ctx, loss, x);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, gx, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  ParityBinding bindings[] = {{out, gx_d}, {x, x_d}};
  int ok = run_and_report(ctx, sink, bindings, 2, gx_d, N);
  poly_ctx_destroy(ctx);
  return ok;
}

static int case_grad_sqrt_sum(void) {
  int N = 6;
  float x_d[6] = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 0.25f};
  float gx_d[6] = {0};

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *s = poly_uop1(ctx, POLY_OP_SQRT, POLY_FLOAT32, x, poly_arg_none());
  int64_t axes[] = {0};
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, s, axes, 1);
  PolyUOp *gx = poly_grad(ctx, loss, x);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, gx, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  ParityBinding bindings[] = {{out, gx_d}, {x, x_d}};
  int ok = run_and_report(ctx, sink, bindings, 2, gx_d, N);
  poly_ctx_destroy(ctx);
  return ok;
}

static int case_grad_where_sum(void) {
  int N = 8;
  float x_d[8], gx_d[8] = {0};
  for (int i = 0; i < N; i++) x_d[i] = (float)(i - 3);

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(0.0));
  PolyUOp *cmp = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, zero, x, poly_arg_none());
  PolyUOp *relu = poly_uop3(ctx, POLY_OP_WHERE, POLY_FLOAT32, cmp, x, zero, poly_arg_none());
  int64_t axes[] = {0};
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, relu, axes, 1);
  PolyUOp *gx = poly_grad(ctx, loss, x);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, gx, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  ParityBinding bindings[] = {{out, gx_d}, {x, x_d}};
  int ok = run_and_report(ctx, sink, bindings, 2, gx_d, N);
  poly_ctx_destroy(ctx);
  return ok;
}

static int case_grad_multi_use(void) {
  int N = 6;
  float x_d[6] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f};
  float gx_d[6] = {0};

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *x = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(2.0));
  PolyUOp *xx = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, x, x, poly_arg_none());
  PolyUOp *x2 = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, x, two, poly_arg_none());
  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, xx, x2, poly_arg_none());
  int64_t axes[] = {0};
  PolyUOp *loss = poly_reduce_axis(ctx, POLY_OP_ADD, add, axes, 1);
  PolyUOp *gx = poly_grad(ctx, loss, x);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, N);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out, gx, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  ParityBinding bindings[] = {{out, gx_d}, {x, x_d}};
  int ok = run_and_report(ctx, sink, bindings, 2, gx_d, N);
  poly_ctx_destroy(ctx);
  return ok;
}

/* ── Tier 5: NN patterns ─────────────────────────────────────────── */

static int case_matmul_small(void) {
  /* 2x3 @ 3x2 → 2x2 via reshape+expand+mul+reduce */
  float a_d[6] = {1, 2, 3, 4, 5, 6};
  float b_d[6] = {1, 4, 2, 5, 3, 6};
  float c_d[4] = {0};

  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *a = poly_buffer(ctx, POLY_FLOAT32, 6);
  PolyUOp *b = poly_buffer(ctx, POLY_FLOAT32, 6);
  PolyUOp *c = poly_buffer(ctx, POLY_FLOAT32, 4);

  int64_t a_shape[] = {2, 1, 3};
  PolyUOp *ar = poly_reshape(ctx, a, a_shape, 3);
  int64_t a_exp_shape[] = {2, 2, 3};
  PolyUOp *ae = poly_expand(ctx, ar, a_exp_shape, 3);

  int64_t b_2d[] = {3, 2};
  PolyUOp *br = poly_reshape(ctx, b, b_2d, 2);
  int64_t b_perm[] = {1, 0};
  PolyUOp *bp = poly_permute(ctx, br, b_perm, 2);
  int64_t b_3d[] = {1, 2, 3};
  PolyUOp *br2 = poly_reshape(ctx, bp, b_3d, 3);
  int64_t b_exp[] = {2, 2, 3};
  PolyUOp *be = poly_expand(ctx, br2, b_exp, 3);

  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, ae, be, poly_arg_none());
  int64_t red_axes[] = {2};
  PolyUOp *sum = poly_reduce_axis(ctx, POLY_OP_ADD, mul, red_axes, 1);

  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, c, sum, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  ParityBinding bindings[] = {{c, c_d}, {a, a_d}, {b, b_d}};
  int ok = run_and_report(ctx, sink, bindings, 3, c_d, 4);
  poly_ctx_destroy(ctx);
  return ok;
}

/* ── Case dispatch ────────────────────────────────────────────────── */

typedef int (*CaseFn)(void);
typedef struct {
  const char *name;
  CaseFn fn;
} CaseEntry;

static CaseEntry CASES[] = {
  /* Original 16 */
  {"vecadd", case_vecadd},
  {"chain", case_chain},
  {"broadcast_scalar", case_broadcast_scalar},
  {"reduce_sum_axis1", case_reduce_sum_axis1},
  {"reduce_scalar_chain", case_reduce_scalar_chain},
  {"reduce_vector_chain", case_reduce_vector_chain},
  {"shared_scalar_reduce_branches", case_shared_scalar_reduce_branches},
  {"permute_2d", case_permute_2d},
  {"shrink_2d", case_shrink_2d},
  {"pad_2d", case_pad_2d},
  {"chain_pad_flip", case_chain_pad_flip},
  {"grad_mul_sum", case_grad_mul_sum},
  {"grad_exp2_sum", case_grad_exp2_sum},
  {"grad_fdiv_sum_x", case_grad_fdiv_sum_x},
  {"grad_fdiv_sum_y", case_grad_fdiv_sum_y},
  {"grad_chain_movement", case_grad_chain_movement},
  /* Tier 1: elementwise */
  {"neg_1d", case_neg_1d},
  {"exp2_1d", case_exp2_1d},
  {"sqrt_1d", case_sqrt_1d},
  {"mul_1d", case_mul_1d},
  {"where_1d", case_where_1d},
  /* Tier 2: reductions */
  {"reduce_sum_all", case_reduce_sum_all},
  {"reduce_max_1d", case_reduce_max_1d},
  /* Tier 3: composed */
  {"reshape_reduce", case_reshape_reduce},
  {"expand_alu_reduce", case_expand_alu_reduce},
  {"multi_movement", case_multi_movement},
  /* Tier 4: autograd */
  {"grad_log2_sum", case_grad_log2_sum},
  {"grad_sqrt_sum", case_grad_sqrt_sum},
  {"grad_where_sum", case_grad_where_sum},
  {"grad_multi_use", case_grad_multi_use},
  /* Tier 5: NN */
  {"matmul_small", case_matmul_small},
};

static int run_case(const char *name) {
  int n_cases = (int)(sizeof(CASES) / sizeof(CASES[0]));
  for (int i = 0; i < n_cases; i++) {
    if (strcmp(CASES[i].name, name) == 0) return CASES[i].fn();
  }
  fprintf(stderr, "unknown case: %s\n", name);
  return 0;
}

static void print_cases(void) {
  int n_cases = (int)(sizeof(CASES) / sizeof(CASES[0]));
  for (int i = 0; i < n_cases; i++) printf("%s\n", CASES[i].name);
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s <case-name|--list>\n", argv[0]);
    return 2;
  }
  if (strcmp(argv[1], "--list") == 0) {
    print_cases();
    return 0;
  }
  return run_case(argv[1]) ? 0 : 1;
}
