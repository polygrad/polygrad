/*
 * schedule.c -- Tinygrad-aligned engine scheduling and compiled execution.
 *
 * Execution counterpart: tinygrad/engine/realize.py. C runtime preparation,
 * argument binding and backend resources sit below the same LINEAR/PROGRAM IR.
 * Tensor scheduling lives in schedule/schedule.c.
 */

#define _POSIX_C_SOURCE 200809L
#include "engine/schedule.h"
#include "bigint.h"
#include "ctx.h"
#include "utils.h"
#include "uop/upat.h"
#include "placer.h"
#include "codegen/codegen.h"
#include "renderer/cstyle.h"
#include "renderer/isa/x86.h"
#include "interp.h"
#include "schedule/rangeify.h"
#include "schedule/schedule.h"
#include "runtime_wasm.h"
#include "runtime_webgpu.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <math.h>

static PolyUOp *poly_program_source_identity(PolyUOp *program) {
  if (!program || program->op != POLY_OP_PROGRAM) return program;
  if (program->n_src >= 2 && program->src[1] && program->src[1]->op == POLY_OP_LINEAR)
    return program->src[1];
  if (program->n_src >= 1) return program->src[0];
  return program;
}

static void stable_kernel_fn_name(
    PolyCtx *ctx,
    char *out,
    size_t cap,
    PolyDevice device,
    PolyUOp *root
) {
  (void)ctx;
  uint32_t h = poly_structural_hash(poly_program_source_identity(root));
  /* The native CPU compiler hashes the full rendered source for its disk cache.
   * Counter-based function names make identical kernels render different source,
   * defeating tinygrad-style compile caching across eager realize calls. */
  snprintf(out, cap, "poly_k_%d_%08x", (int)device, h);
}

typedef struct PolyRuntimeCacheEntry {
  int refcount;
  bool in_cache;
  PolyCtx *ctx;
  size_t accounted_bytes;
  PolyUOp *program;
  PolyUOp *device_uop;
  PolyDevice device;
  uint32_t env_stamp;
  PolyRunner runner;
} PolyRuntimeCacheEntry;

typedef struct {
  PolyUOp *program;
  PolyUOp *device_uop;
  PolyDevice device;
  uint32_t env_stamp;
  PolyRuntimeCacheEntry *runtime_program;
} PolyRuntimeCacheMapEntry;

typedef struct {
  PolyUOp *program;
  PolyUOp *device_uop;
  PolyDevice device;
  uint32_t env_stamp;
  PolyUOp *prepared_program;
} PolyToProgramCacheEntry;

#ifdef POLY_HAS_CUDA
typedef struct {
  PolyUOp *function;
  PolyCudaGraph *graph;
  PolyRunner *runners;
  PolyRuntimeCacheEntry **runtime_entries;
  int n_nodes;
} PolyGraphCacheEntry;
#endif

static uint32_t poly_runtime_cache_env_stamp(void);
static int poly_launch_dim_upper_bound(PolyCtx *ctx, PolyUOp *expr);
static PolyUOp *poly_program_ensure_source(
    PolyCtx *ctx,
    const PolyBackendDesc *backend,
    PolyUOp *program,
    PolyDevice device,
    const char *name_override
);
static int g_program_source_render_count = 0;

int poly_program_source_render_count(void) {
  return g_program_source_render_count;
}

void poly_program_source_render_count_reset(void) {
  g_program_source_render_count = 0;
}

static char *program_info_strdup(const char *s) {
  if (!s) s = "";
  size_t len = strlen(s);
  char *out = malloc(len + 1);
  if (!out) return NULL;
  memcpy(out, s, len + 1);
  return out;
}

static PolyUOp *poly_call_raw_body(PolyUOp *call) {
  if (!call || call->op != POLY_OP_CALL || call->n_src < 1) return NULL;
  return call->src[0];
}

static PolyUOp *poly_program_body(PolyUOp *body) {
  if (body && body->op == POLY_OP_PROGRAM && body->n_src >= 1) return body->src[0];
  return body;
}

static PolyUOp *poly_program_kernel_body(PolyUOp *program) {
  if (!program || program->op != POLY_OP_PROGRAM || program->n_src < 1) return NULL;
  return program->src[0];
}

PolyUOp *poly_program_linear(PolyUOp *program) {
  if (!program || program->op != POLY_OP_PROGRAM || program->n_src < 2) return NULL;
  PolyUOp *linear = program->src[1];
  return (linear && linear->op == POLY_OP_LINEAR) ? linear : NULL;
}

static PolyUOp *poly_program_source(PolyUOp *program) {
  if (!program || program->op != POLY_OP_PROGRAM || program->n_src < 3) return NULL;
  PolyUOp *source = program->src[2];
  return (source && source->op == POLY_OP_SOURCE) ? source : NULL;
}

#ifdef POLY_HAS_X86
static PolyUOp *poly_program_binary(PolyUOp *program) {
  if (!program || program->op != POLY_OP_PROGRAM || program->n_src < 4) return NULL;
  PolyUOp *binary = program->src[3];
  return (binary && binary->op == POLY_OP_BINARY) ? binary : NULL;
}
#endif

#ifndef __EMSCRIPTEN__
static const char *poly_program_source_text(PolyUOp *program) {
  PolyUOp *source = poly_program_source(program);
  if (!source || source->arg.kind != POLY_ARG_STRING) return NULL;
  return source->arg.str;
}
#endif

static PolyUOp **poly_program_linear_uops(PolyUOp *program, int *n_out) {
  PolyUOp *linear = poly_program_linear(program);
  if (!linear) return NULL;
  if (n_out) *n_out = linear->n_src;
  return linear->src;
}

static bool poly_call_arg_is_var(PolyUOp *u) {
  return poly_uop_is_bound_var(u);
}

int poly_call_n_buffer_args(PolyUOp *call) {
  if (!call || call->op != POLY_OP_CALL || call->n_src < 1) return 0;
  int n = 0;
  for (int i = 1; i < call->n_src; i++)
    if (!poly_call_arg_is_var(call->src[i])) n++;
  return n;
}

PolyUOp *poly_call_buffer_arg(PolyUOp *call, int param_idx) {
  if (!call || call->op != POLY_OP_CALL || param_idx < 0) return NULL;
  int seen = 0;
  for (int i = 1; i < call->n_src; i++) {
    if (poly_call_arg_is_var(call->src[i])) continue;
    if (seen++ == param_idx) return call->src[i];
  }
  return NULL;
}

static int poly_call_param_index_for_identity(PolyUOp *call, const PolyUOp *identity) {
  if (!call || !identity) return -1;
  int n_args = poly_call_n_buffer_args(call);
  if (identity->op == POLY_OP_PARAM &&
      (identity->arg.kind == POLY_ARG_INT ||
       (identity->arg.kind == POLY_ARG_PARAM && identity->arg.param))) {
    int idx =
        identity->arg.kind == POLY_ARG_INT ? (int)identity->arg.i : (int)identity->arg.param->slot;
    return (idx >= 0 && idx < n_args) ? idx : -1;
  }
  for (int i = 0; i < n_args; i++) {
    PolyUOp *arg = poly_call_buffer_arg(call, i);
    if (poly_uop_get_buffer_identity(arg) == identity) return i;
  }
  return -1;
}

static void poly_call_mark_access_param(PolyUOp *call, PolyUOp *ptr, bool *mask, int n_args) {
  if (!ptr || !mask) return;
  /* Pinned ProgramInfo.from_sink reads ParamArg.slot for shaped function
   * PARAMs as well as the lowered integer-slot PARAM form
   * (tinygrad/uop/ops.py:1127-1152). */
  if (ptr->op == POLY_OP_CAST && ptr->n_src == 1 && ptr->src[0] && ptr->src[0]->op == POLY_OP_INDEX)
    ptr = ptr->src[0];
  if (ptr->op == POLY_OP_PARAM &&
      (ptr->arg.kind == POLY_ARG_INT || (ptr->arg.kind == POLY_ARG_PARAM && ptr->arg.param))) {
    int idx = ptr->arg.kind == POLY_ARG_INT ? (int)ptr->arg.i : (int)ptr->arg.param->slot;
    if (idx >= 0 && idx < n_args) mask[idx] = true;
    return;
  }
  if (ptr->op == POLY_OP_INDEX || ptr->op == POLY_OP_SHRINK) {
    if (ptr->n_src > 0) poly_call_mark_access_param(call, ptr->src[0], mask, n_args);
    return;
  }
  const PolyUOp *identity = poly_uop_get_buffer_identity(ptr);
  int idx = poly_call_param_index_for_identity(call, identity);
  if (idx >= 0 && idx < n_args) {
    mask[idx] = true;
    return;
  }
  if (ptr->op == POLY_OP_STACK || ptr->op == POLY_OP_TUPLE || ptr->op == POLY_OP_GROUP) {
    for (int i = 0; i < ptr->n_src; i++)
      poly_call_mark_access_param(call, ptr->src[i], mask, n_args);
  }
}

static int poly_call_mask_to_indices(const bool *mask, int n, int **out_items, int *out_n) {
  if (!mask || !out_items || !out_n || n < 0) return -1;
  int count = 0;
  for (int i = 0; i < n; i++)
    if (mask[i]) count++;

  int *items = NULL;
  if (count > 0) {
    items = malloc((size_t)count * sizeof(*items));
    if (!items) return -1;
    int w = 0;
    for (int i = 0; i < n; i++)
      if (mask[i]) items[w++] = i;
  }
  *out_items = items;
  *out_n = count;
  return 0;
}

const PolyProgramInfo *poly_program_info(PolyCtx *ctx, PolyUOp *program) {
  (void)ctx;
  if (!program || program->op != POLY_OP_PROGRAM) return NULL;
  if (program->arg.kind == POLY_ARG_PROGRAM_INFO) return program->arg.program_info;
  return NULL;
}

/* Current Tinygrad stores estimates on PROGRAM.src[0].arg KernelInfo. */
static const PolyEstimates *poly_program_estimates(PolyUOp *program) {
  PolyUOp *sink = poly_program_kernel_body(program);
  return sink && sink->op == POLY_OP_SINK && sink->arg.kind == POLY_ARG_KERNEL_INFO &&
                 sink->arg.kernel_info
             ? sink->arg.kernel_info->estimates
             : NULL;
}

static PolyUOp *estimate_const(PolyCtx *ctx, int64_t value) {
  return poly_uop0(ctx, POLY_OP_CONST, POLY_INT64, poly_arg_int(value));
}

static PolyUOp *estimate_i64(PolyCtx *ctx, PolyUOp *value) {
  if (!ctx || !value) return NULL;
  PolyDType scalar = value->dtype;
  if (poly_dtype_eq(scalar, POLY_INT64)) return value;
  return poly_uop1(ctx, POLY_OP_CAST, POLY_INT64, value, poly_arg_none());
}

static PolyUOp *estimate_binary(PolyCtx *ctx, PolyOps op, PolyUOp *a, PolyUOp *b) {
  if (!ctx || !a || !b) return NULL;
  return poly_uop2(ctx, op, POLY_INT64, a, b, poly_arg_none());
}

static PolyUOp *estimate_add(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  int64_t a_min = 0, a_max = 0, b_min = 0, b_max = 0, max_sum = 0;
  if (!ctx || !a || !b) return NULL;
  poly_uop_minmax(ctx, a, &a_min, &a_max);
  poly_uop_minmax(ctx, b, &b_min, &b_max);
  if (a_min < 0 || b_min < 0 || __builtin_add_overflow(a_max, b_max, &max_sum)) return NULL;
  return estimate_binary(ctx, POLY_OP_ADD, a, b);
}

static PolyUOp *estimate_mul(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  int64_t a_min = 0, a_max = 0, b_min = 0, b_max = 0, max_product = 0;
  if (!ctx || !a || !b) return NULL;
  poly_uop_minmax(ctx, a, &a_min, &a_max);
  poly_uop_minmax(ctx, b, &b_min, &b_max);
  if (a_min < 0 || b_min < 0 || __builtin_mul_overflow(a_max, b_max, &max_product)) return NULL;
  return estimate_binary(ctx, POLY_OP_MUL, a, b);
}

static bool estimate_i64_product(int64_t a, int64_t b, int64_t *out) {
  return out && a >= 0 && b >= 0 && !__builtin_mul_overflow(a, b, out);
}

static PolyUOp *estimate_min(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  if (!ctx || !a || !b) return NULL;
  PolyUOp *lt = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, a, b, poly_arg_none());
  return lt ? poly_uop3(ctx, POLY_OP_WHERE, POLY_INT64, lt, a, b, poly_arg_none()) : NULL;
}

static int64_t estimate_max_numel(PolyCtx *ctx, PolyUOp *u) {
  if (!ctx || !u) return 1;
  int64_t numel = 1;
  PolyShape shape = poly_uop_max_shape_cached(ctx, u);
  if (shape.ndim > 0) {
    for (int i = 0; i < shape.ndim; i++) {
      if (shape.dims[i] <= 0 || __builtin_mul_overflow(numel, shape.dims[i], &numel)) {
        numel = INT64_MAX;
        break;
      }
    }
  }
  int64_t lanes = poly_uop_max_numel(ctx, u);
  return numel > lanes ? numel : lanes;
}

static bool estimate_not_end(PolyUOp *u) {
  return u && u->op != POLY_OP_END;
}

static PolyUOp *estimate_without_specials(PolyCtx *ctx, PolyUOp *expr) {
  if (!ctx || !expr) return NULL;
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, expr, &n_topo);
  if (!topo) return NULL;
  int n_special = 0;
  for (int i = 0; i < n_topo; i++)
    if (topo[i]->op == POLY_OP_SPECIAL) n_special++;
  if (n_special == 0) {
    poly_toposort_free(topo);
    return expr;
  }
  PolyUOp **from = calloc((size_t)n_special, sizeof(*from));
  PolyUOp **to = calloc((size_t)n_special, sizeof(*to));
  PolyUOp *zero = estimate_const(ctx, 0);
  if (!from || !to || !zero) {
    free(from);
    free(to);
    poly_toposort_free(topo);
    return NULL;
  }
  int at = 0;
  for (int i = 0; i < n_topo; i++) {
    if (topo[i]->op != POLY_OP_SPECIAL) continue;
    from[at] = topo[i];
    to[at++] = zero;
  }
  PolyUOp *out = poly_uop_substitute(ctx, expr, from, to, n_special);
  free(from);
  free(to);
  poly_toposort_free(topo);
  return out;
}

typedef struct {
  PolyUOp *base;
  PolyOps op;
  PolyUOp *bytes;
} PolyEstimateMemoryEntry;

int poly_estimates_from_uops(
    PolyCtx *ctx,
    PolyUOp **uops,
    int n_uops,
    bool ignore_indexing,
    PolyEstimates *out
) {
  if (!ctx || !out || n_uops < 0 || (n_uops > 0 && !uops)) return -1;
  memset(out, 0, sizeof(*out));
  PolyUOp *zero = estimate_const(ctx, 0);
  PolyUOp *one = estimate_const(ctx, 1);
  if (!zero || !one) return -1;

  PolyMap *excluded = poly_map_new((size_t)(n_uops > 0 ? n_uops : 4));
  PolyUOp **mult_stack = calloc((size_t)(n_uops > 0 ? n_uops : 1), sizeof(*mult_stack));
  PolyEstimateMemoryEntry *memory = calloc((size_t)(n_uops > 0 ? n_uops : 1), sizeof(*memory));
  if (!excluded || !mult_stack || !memory) {
    if (excluded) poly_map_destroy(excluded);
    free(mult_stack);
    free(memory);
    return -1;
  }

  int rc = 0;
  if (ignore_indexing) {
    for (int i = 0; i < n_uops; i++) {
      PolyUOp *u = uops[i];
      if (!u || (u->op != POLY_OP_INDEX && u->op != POLY_OP_SHRINK)) continue;
      for (int j = 1; j < u->n_src; j++) {
        int n_topo = 0;
        PolyUOp **topo = poly_toposort_ex_alloc(ctx, u->src[j], &n_topo, estimate_not_end, true);
        if (!topo) {
          rc = -1;
          break;
        }
        for (int k = 0; topo && k < n_topo; k++)
          poly_map_set(excluded, poly_ptr_hash(topo[k]), topo[k], (void *)1, poly_ptr_eq);
        poly_toposort_free(topo);
      }
      if (rc != 0) break;
    }
  }

  PolyUOp *flops = zero;
  PolyUOp *lds = zero;
  PolyUOp *mults = one;
  int n_mult_stack = 0;
  int n_memory = 0;

  for (int i = 0; i < n_uops && rc == 0; i++) {
    PolyUOp *u = uops[i];
    if (!u) continue;

    if ((u->op == POLY_OP_LOAD || u->op == POLY_OP_STORE) && u->n_src > 0) {
      PolyUOp *base = u;
      while (base->n_src > 0 && base->op != POLY_OP_PARAM)
        base = base->src[0];
      if (base->op == POLY_OP_PARAM) {
        int entry = -1;
        for (int j = 0; j < n_memory; j++)
          if (memory[j].base == base && memory[j].op == u->op) {
            entry = j;
            break;
          }
        if (entry < 0) {
          entry = n_memory++;
          memory[entry] = (PolyEstimateMemoryEntry){.base = base, .op = u->op, .bytes = zero};
        }
        int64_t index_lanes = estimate_max_numel(ctx, u->src[0]);
        int64_t itemsize = poly_dtype_itemsize(u->src[0]->dtype);
        int64_t access_bytes = 0;
        PolyUOp *access_once = estimate_i64_product(index_lanes, itemsize, &access_bytes)
                                   ? estimate_const(ctx, access_bytes)
                                   : NULL;
        PolyUOp *accessed = access_once ? estimate_mul(ctx, access_once, mults) : NULL;
        accessed = accessed ? estimate_add(ctx, memory[entry].bytes, accessed) : NULL;
        /* tinygrad renderer/__init__.py::Estimates.from_uops caps repeated
         * traffic with buf.max_numel(), including scalar PARAM(CONST(size))
         * after codegen's pm_remove_vec_dtypes has removed the pointer dtype. */
        PolyShape cap_shape = poly_uop_max_shape_cached(ctx, base);
        int64_t cap_numel = cap_shape.ndim >= 0 ? poly_shape_numel(cap_shape) : -1;
        int64_t cap_itemsize = poly_dtype_itemsize(base->dtype);
        int64_t cap_bytes = 0;
        PolyUOp *cap = cap_numel >= 0 && estimate_i64_product(cap_numel, cap_itemsize, &cap_bytes)
                           ? estimate_const(ctx, cap_bytes)
                           : NULL;
        memory[entry].bytes = cap ? estimate_min(ctx, accessed, cap) : accessed;
        if (!memory[entry].bytes) rc = -1;
      }
    }

    if (u->op == POLY_OP_RANGE) {
      if (u->n_src < 1 || n_mult_stack >= n_uops) {
        rc = -1;
        break;
      }
      mult_stack[n_mult_stack++] = mults;
      /* Tinygrad 2026-08-22/a9069c177a9d renderer/__init__.py:40-46:
       * a void RANGE is an unbounded runtime loop with no known trip count. */
      if (!poly_dtype_eq(u->dtype, POLY_VOID)) {
        PolyUOp *bound = estimate_i64(ctx, u->src[0]);
        mults = bound ? estimate_mul(ctx, mults, bound) : NULL;
        mults = mults ? estimate_without_specials(ctx, mults) : NULL;
        if (!mults) rc = -1;
      }
      continue;
    }
    if (u->op == POLY_OP_END) {
      if (n_mult_stack <= 0) {
        rc = -1;
        break;
      }
      mults = mult_stack[--n_mult_stack];
      continue;
    }
    if (u->op == POLY_OP_SPECIAL) {
      if (u->n_src < 1) {
        rc = -1;
        break;
      }
      PolyUOp *bound = estimate_i64(ctx, u->src[0]);
      mults = bound ? estimate_mul(ctx, mults, bound) : NULL;
      if (!mults) rc = -1;
      continue;
    }
    if (poly_uop_is_alu_param(u) && poly_uop_expr(u) && strcmp(poly_uop_expr(u), "core_id") == 0) {
      int64_t cores_count = 0;
      if (!poly_arg_integer_to_i64(u->arg.param->max_val, &cores_count) ||
          __builtin_add_overflow(cores_count, INT64_C(1), &cores_count) || cores_count < 0) {
        rc = -1;
        break;
      }
      PolyUOp *cores = estimate_const(ctx, cores_count);
      mults = cores ? estimate_mul(ctx, mults, cores) : NULL;
      if (!mults) rc = -1;
      continue;
    }

    PolyAddrSpace addrspace = POLY_ADDR_GLOBAL;
    bool has_addrspace = u->n_src > 0 && poly_uop_addrspace(u->src[0], &addrspace);
    if (u->op == POLY_OP_LOAD && u->n_src > 0 && (!has_addrspace || addrspace != POLY_ADDR_REG)) {
      int64_t bytes = 0;
      PolyUOp *amount =
          estimate_i64_product(estimate_max_numel(ctx, u), poly_dtype_itemsize(u->dtype), &bytes)
              ? estimate_const(ctx, bytes)
              : NULL;
      amount = amount ? estimate_mul(ctx, amount, mults) : NULL;
      lds = amount ? estimate_add(ctx, lds, amount) : NULL;
      if (!lds) rc = -1;
    } else if (u->op == POLY_OP_STORE && u->n_src > 1 && (!has_addrspace || addrspace != POLY_ADDR_REG)) {
      int64_t bytes = 0;
      PolyUOp *amount =
          estimate_i64_product(
              estimate_max_numel(ctx, u), poly_dtype_itemsize(u->src[1]->dtype), &bytes
          )
              ? estimate_const(ctx, bytes)
              : NULL;
      amount = amount ? estimate_mul(ctx, amount, mults) : NULL;
      lds = amount ? estimate_add(ctx, lds, amount) : NULL;
      if (!lds) rc = -1;
    } else if (poly_opset_has(POLY_GROUP_ALU, u->op) && !poly_map_get(excluded, poly_ptr_hash(u), u, poly_ptr_eq)) {
      int64_t weight = 0;
      PolyUOp *amount =
          estimate_i64_product(u->op == POLY_OP_MULACC ? 2 : 1, estimate_max_numel(ctx, u), &weight)
              ? estimate_const(ctx, weight)
              : NULL;
      amount = amount ? estimate_mul(ctx, amount, mults) : NULL;
      flops = amount ? estimate_add(ctx, flops, amount) : NULL;
      if (!flops) rc = -1;
    } else if (u->op == POLY_OP_WMMA && u->arg.kind == POLY_ARG_TENSOR_CORE && u->arg.tensor_core.threads > 0 && !poly_map_get(excluded, poly_ptr_hash(u), u, poly_ptr_eq)) {
      int64_t per_thread = 2;
      for (int d = 0; d < 3; d++) {
        if (u->arg.tensor_core.dims[d] <= 0 ||
            __builtin_mul_overflow(per_thread, u->arg.tensor_core.dims[d], &per_thread)) {
          rc = -1;
          break;
        }
      }
      if (rc != 0) break;
      per_thread /= u->arg.tensor_core.threads;
      PolyUOp *amount = estimate_const(ctx, per_thread);
      amount = amount ? estimate_mul(ctx, amount, mults) : NULL;
      flops = amount ? estimate_add(ctx, flops, amount) : NULL;
      if (!flops) rc = -1;
    }
  }

  PolyUOp *mem = zero;
  for (int i = 0; i < n_memory && rc == 0; i++) {
    mem = estimate_add(ctx, mem, memory[i].bytes);
    if (!mem) rc = -1;
  }
  if (rc == 0) {
    out->ops = flops;
    out->lds = lds;
    out->mem = mem;
  }
  poly_map_destroy(excluded);
  free(mult_stack);
  free(memory);
  return rc;
}

static bool str_eq(const char *a, const char *b) {
  return a == b || (a && b && strcmp(a, b) == 0);
}

static bool int_arr_eq(const int *a, int na, const int *b, int nb) {
  if (na != nb) return false;
  if (na == 0) return true;
  return a && b && memcmp(a, b, (size_t)na * sizeof(int)) == 0;
}

static uint32_t program_info_hash_mix(uint32_t h, uint32_t v) {
  h ^= v;
  h *= 0x9e3779b9;
  h ^= h >> 16;
  return h;
}

static uint32_t program_info_hash_ptr(uint32_t h, const void *p) {
  uintptr_t v = (uintptr_t)p;
  uint32_t folded = (uint32_t)v;
  if (sizeof(v) > 4) folded ^= (uint32_t)(v >> 32);
  h = program_info_hash_mix(h, folded);
  return h;
}

bool poly_program_info_eq(const PolyProgramInfo *a, const PolyProgramInfo *b) {
  if (a == b) return true;
  if (!a || !b) return false;
  if (!str_eq(a->name, b->name)) return false;
  if (!str_eq(a->target, b->target)) return false;
  if (memcmp(a->global_size, b->global_size, sizeof(a->global_size)) != 0) return false;
  if (memcmp(a->local_size, b->local_size, sizeof(a->local_size)) != 0) return false;
  if (a->has_local_size != b->has_local_size) return false;
  if (a->n_vars != b->n_vars || a->n_globals != b->n_globals || a->n_outs != b->n_outs ||
      a->n_ins != b->n_ins)
    return false;
  for (int i = 0; i < 3; i++) {
    if (a->global_exprs[i] != b->global_exprs[i]) return false;
    if (a->local_exprs[i] != b->local_exprs[i]) return false;
  }
  for (int i = 0; i < a->n_vars; i++)
    if (a->vars[i] != b->vars[i]) return false;
  return int_arr_eq(a->globals, a->n_globals, b->globals, b->n_globals) &&
         int_arr_eq(a->outs, a->n_outs, b->outs, b->n_outs) &&
         int_arr_eq(a->ins, a->n_ins, b->ins, b->n_ins);
}

uint32_t poly_program_info_hash(const PolyProgramInfo *info) {
  uint32_t h = 0x811c9dc5u;
  if (!info) return h;
  if (info->name) {
    for (const unsigned char *p = (const unsigned char *)info->name; *p; p++)
      h = program_info_hash_mix(h, (uint32_t)*p);
  }
  if (info->target) {
    for (const unsigned char *p = (const unsigned char *)info->target; *p; p++)
      h = program_info_hash_mix(h, (uint32_t)*p);
  }
  for (int i = 0; i < 3; i++) {
    h = program_info_hash_mix(h, (uint32_t)info->global_size[i]);
    h = program_info_hash_mix(h, (uint32_t)info->local_size[i]);
    h = program_info_hash_ptr(h, info->global_exprs[i]);
    h = program_info_hash_ptr(h, info->local_exprs[i]);
  }
  h = program_info_hash_mix(h, info->has_local_size ? 1u : 0u);
  h = program_info_hash_mix(h, (uint32_t)info->n_vars);
  for (int i = 0; i < info->n_vars; i++)
    h = program_info_hash_ptr(h, info->vars[i]);
  h = program_info_hash_mix(h, (uint32_t)info->n_globals);
  for (int i = 0; i < info->n_globals; i++)
    h = program_info_hash_mix(h, (uint32_t)info->globals[i]);
  h = program_info_hash_mix(h, (uint32_t)info->n_outs);
  for (int i = 0; i < info->n_outs; i++)
    h = program_info_hash_mix(h, (uint32_t)info->outs[i]);
  h = program_info_hash_mix(h, (uint32_t)info->n_ins);
  for (int i = 0; i < info->n_ins; i++)
    h = program_info_hash_mix(h, (uint32_t)info->ins[i]);
  return h;
}

static int poly_call_get_outs_ins_from_body(
    PolyCtx *ctx,
    PolyUOp *call,
    PolyUOp *body,
    bool *globals,
    bool *outs,
    bool *ins,
    int n_args
) {
  if (!ctx || !call || !outs || !ins || n_args < 0) return -1;
  if (globals && n_args > 0) memset(globals, 0, (size_t)n_args * sizeof(bool));
  if (n_args > 0) {
    memset(outs, 0, (size_t)n_args * sizeof(bool));
    memset(ins, 0, (size_t)n_args * sizeof(bool));
  }

  if (!body) return -1;
  if (body->op == POLY_OP_COPY) {
    if (n_args < 2) return -1;
    if (globals) {
      globals[0] = true;
      globals[1] = true;
    }
    outs[0] = true;
    ins[1] = true;
    return 0;
  }
  int cap = 64, n_stack = 0;
  PolyUOp **stack = malloc((size_t)cap * sizeof(PolyUOp *));
  PolyMap *seen = poly_map_new(128);
  if (!stack || !seen) {
    free(stack);
    if (seen) poly_map_destroy(seen);
    return -1;
  }
  stack[n_stack++] = body;
  while (n_stack > 0) {
    PolyUOp *u = stack[--n_stack];
    if (!u) continue;
    if (poly_map_get(seen, poly_ptr_hash(u), u, poly_ptr_eq)) continue;
    poly_map_set(seen, poly_ptr_hash(u), u, u, poly_ptr_eq);
    if (globals) poly_call_mark_access_param(call, u, globals, n_args);

    if (u->op == POLY_OP_STORE && u->n_src >= 1) {
      if (globals) poly_call_mark_access_param(call, u->src[0], globals, n_args);
      poly_call_mark_access_param(call, u->src[0], outs, n_args);
      for (int s = 1; s < u->n_src; s++) {
        if (n_stack >= cap) {
          cap *= 2;
          PolyUOp **new_stack = realloc(stack, (size_t)cap * sizeof(PolyUOp *));
          if (!new_stack) {
            free(stack);
            poly_map_destroy(seen);
            return -1;
          }
          stack = new_stack;
        }
        stack[n_stack++] = u->src[s];
      }
      continue;
    }

    if (u->op == POLY_OP_LOAD && u->n_src >= 1) {
      if (globals) poly_call_mark_access_param(call, u->src[0], globals, n_args);
      poly_call_mark_access_param(call, u->src[0], ins, n_args);
    } else {
      const PolyUOp *identity = poly_uop_get_buffer_identity(u);
      int idx = poly_call_param_index_for_identity(call, identity);
      if (idx >= 0 && idx < n_args) ins[idx] = true;
    }

    for (int s = 0; s < u->n_src; s++) {
      if (n_stack >= cap) {
        cap *= 2;
        PolyUOp **new_stack = realloc(stack, (size_t)cap * sizeof(PolyUOp *));
        if (!new_stack) {
          free(stack);
          poly_map_destroy(seen);
          return -1;
        }
        stack = new_stack;
      }
      stack[n_stack++] = u->src[s];
    }
  }
  free(stack);
  poly_map_destroy(seen);
  return 0;
}

static int poly_call_apply_program_info(
    const PolyProgramInfo *info,
    bool *outs,
    bool *ins,
    int n_args
) {
  if (!info || !outs || !ins || n_args < 0) return -1;
  /* exec_kernel indexes all ProgramInfo.globals, not just read/write sets.
   * Validate here before either direct execution or graph preparation. */
  for (int i = 0; i < info->n_globals; i++)
    if (info->globals[i] < 0 || info->globals[i] >= n_args) return -1;
  memset(outs, 0, (size_t)n_args * sizeof(bool));
  memset(ins, 0, (size_t)n_args * sizeof(bool));
  for (int i = 0; i < info->n_outs; i++) {
    int idx = info->outs[i];
    if (idx < 0 || idx >= n_args) return -1;
    outs[idx] = true;
  }
  for (int i = 0; i < info->n_ins; i++) {
    int idx = info->ins[i];
    if (idx < 0 || idx >= n_args) return -1;
    ins[idx] = true;
  }
  return 0;
}

static bool poly_program_info_collect_launch(PolyCtx *ctx, PolyUOp *body, PolyProgramInfo *info) {
  if (!ctx || !body || !info) return false;
  info->global_size[0] = 1;
  info->global_size[1] = 1;
  info->global_size[2] = 1;
  info->local_size[0] = 1;
  info->local_size[1] = 1;
  info->local_size[2] = 1;
  info->has_local_size = true;

  int n_topo = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, body, &n_topo);
  if (!topo) return false;
  if (n_topo > 0) {
    info->vars = malloc((size_t)n_topo * sizeof(*info->vars));
    if (!info->vars) {
      poly_toposort_free(topo);
      return false;
    }
  }
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (poly_uop_is_alu_param(u)) {
      bool duplicate = false;
      for (int v = 0; v < info->n_vars; v++)
        duplicate |= info->vars[v] == u;
      if (!duplicate) info->vars[info->n_vars++] = u;
      if (u->arg.param->name && strcmp(u->arg.param->name, "core_id") == 0) {
        int64_t n = 0;
        if (!poly_arg_integer_to_i64(u->arg.param->max_val, &n) || n < 0 || n >= INT32_MAX) {
          poly_toposort_free(topo);
          return false;
        }
        info->global_size[0] = (int)(n + 1);
      }
    }
    if (!u || u->op != POLY_OP_SPECIAL || u->n_src <= 0 || u->arg.kind != POLY_ARG_STRING) continue;
    const char *name = u->arg.str;
    if (!name || !name[0]) continue;
    int len = (int)strlen(name);
    int dim = (len > 0) ? name[len - 1] - '0' : 0;
    if (dim < 0 || dim > 2) dim = 0;
    PolyUOp *bound = u->src[0];
    int size = poly_launch_dim_upper_bound(ctx, bound);
    if (name[0] == 'i') info->has_local_size = false;
    if (name[0] == 'l') {
      if (info->has_local_size) {
        info->local_size[dim] = size;
        info->local_exprs[dim] = bound;
      }
    } else {
      info->global_size[dim] = size;
      info->global_exprs[dim] = bound;
    }
  }
  /* ProgramInfo.from_sink uses stable slot ordering after pointer dedup. */
  for (int i = 1; i < info->n_vars; i++) {
    PolyUOp *value = info->vars[i];
    int64_t slot = value->arg.param->slot;
    int j = i;
    while (j > 0 && info->vars[j - 1]->arg.param->slot > slot) {
      info->vars[j] = info->vars[j - 1];
      j--;
    }
    info->vars[j] = value;
  }
  poly_toposort_free(topo);
  return true;
}

int poly_call_get_outs_ins(PolyCtx *ctx, PolyUOp *call, bool *outs, bool *ins, int n_args) {
  if (!ctx || !call || call->op != POLY_OP_CALL || call->n_src < 1 || n_args < 0 ||
      (n_args > 0 && (!outs || !ins)))
    return -1;
  PolyUOp *body = call->src[0];
  if (!body) return -1;
  if (body->op == POLY_OP_PROGRAM) {
    const PolyProgramInfo *info = poly_program_info(ctx, body);
    if (info) return poly_call_apply_program_info(info, outs, ins, n_args);
    return -1;
  }
  /* get_call_outs_ins classifies executable calls. The SINK walk belongs to
   * ProgramInfo construction, not arbitrary custom-function children. */
  if (n_args > 0) {
    memset(outs, 0, (size_t)n_args * sizeof(*outs));
    memset(ins, 0, (size_t)n_args * sizeof(*ins));
  }
  if (body->op == POLY_OP_COPY) {
    if (n_args < 2) return -1;
    outs[0] = ins[1] = true;
  } else if (body->op == POLY_OP_CUSTOM_FUNCTION && body->arg.kind == POLY_ARG_STRING &&
             body->arg.str && strcmp(body->arg.str, "encdec") == 0) {
    if (n_args < 1) return -1;
    outs[0] = true;
    for (int i = 1; i < n_args; i++)
      ins[i] = true;
  }
  return 0;
}

static void poly_program_info_destroy(PolyProgramInfo *info) {
  if (!info) return;
  free((char *)info->name);
  free((char *)info->target);
  free(info->vars);
  free(info->globals);
  free(info->outs);
  free(info->ins);
  free(info);
}

static PolyProgramInfo *poly_program_info_build(
    PolyCtx *ctx,
    PolyUOp *call,
    PolyUOp *body,
    const char *program_name,
    PolyDevice device
) {
  if (!ctx || !call || !body || !program_name) return NULL;
  int n_args = poly_call_n_buffer_args(call);
  bool *globals = n_args > 0 ? calloc((size_t)n_args, sizeof(bool)) : NULL;
  bool *outs = n_args > 0 ? calloc((size_t)n_args, sizeof(bool)) : NULL;
  bool *ins = n_args > 0 ? calloc((size_t)n_args, sizeof(bool)) : NULL;
  if (n_args > 0 && (!globals || !outs || !ins)) {
    free(globals);
    free(outs);
    free(ins);
    return NULL;
  }
  if (poly_call_get_outs_ins_from_body(ctx, call, body, globals, outs, ins, n_args) != 0) {
    free(globals);
    free(outs);
    free(ins);
    return NULL;
  }

  PolyProgramInfo *info = calloc(1, sizeof(*info));
  if (!info) {
    free(globals);
    free(outs);
    free(ins);
    return NULL;
  }
  info->name = program_info_strdup(program_name);
  info->target = program_info_strdup(poly_device_name(device));
  if (!info->name || !info->target || !poly_program_info_collect_launch(ctx, body, info)) {
    free(globals);
    free(outs);
    free(ins);
    poly_program_info_destroy(info);
    return NULL;
  }

  if (n_args > 0) {
    if (poly_call_mask_to_indices(globals, n_args, &info->globals, &info->n_globals) != 0) {
      free(globals);
      free(outs);
      free(ins);
      poly_program_info_destroy(info);
      return NULL;
    }
  }

  if (poly_call_mask_to_indices(outs, n_args, &info->outs, &info->n_outs) != 0 ||
      poly_call_mask_to_indices(ins, n_args, &info->ins, &info->n_ins) != 0) {
    free(globals);
    free(outs);
    free(ins);
    poly_program_info_destroy(info);
    return NULL;
  }
  free(globals);
  free(outs);
  free(ins);
  return info;
}

static PolyUOp *poly_program_from_call_body(
    PolyCtx *ctx,
    PolyUOp *call,
    PolyUOp *body,
    const char *name,
    PolyDevice device
) {
  if (!ctx || !call || call->op != POLY_OP_CALL || !body) return NULL;
  if (body->op == POLY_OP_PROGRAM) return body;

  const char *program_name = poly_kernel_name(body, name ? name : "test");
  if (device == POLY_DEVICE_AUTO) device = poly_uop_device(body);

  PolyProgramInfo *info = poly_program_info_build(ctx, call, body, program_name, device);
  if (!info) return NULL;

  PolyUOp *program = poly_uop1(ctx, POLY_OP_PROGRAM, POLY_VOID, body, poly_arg_program_info(info));
  poly_program_info_destroy(info);
  if (!program) return NULL;
  return program;
}

PolyUOp *poly_program_from_call(PolyCtx *ctx, PolyUOp *call, const char *name) {
  if (!ctx || !call || call->op != POLY_OP_CALL || call->n_src < 1) return NULL;
  PolyUOp *body = call->src[0];
  if (!body) return NULL;
  if (body->op == POLY_OP_PROGRAM) return body;
  return poly_program_from_call_body(ctx, call, body, name, poly_uop_device(body));
}

static PolyUOp *poly_program_with_linear(PolyCtx *ctx, PolyUOp *program, PolyUOp *linear) {
  if (!ctx || !program || program->op != POLY_OP_PROGRAM || program->n_src < 1 || !linear ||
      linear->op != POLY_OP_LINEAR)
    return NULL;
  PolyUOp *sink = program->src[0];
  if (!sink || sink->op != POLY_OP_SINK || sink->arg.kind != POLY_ARG_KERNEL_INFO ||
      !sink->arg.kernel_info)
    return NULL;
  PolyUOp *estimated_sink = sink;
  if (!sink->arg.kernel_info->estimates) {
    PolyEstimates estimates = {0};
    if (poly_estimates_from_uops(ctx, linear->src, linear->n_src, true, &estimates) != 0)
      return NULL;
    PolyKernelInfo kernel_info = *sink->arg.kernel_info;
    kernel_info.estimates = &estimates;
    estimated_sink = sink->tag || sink->tag_arg.kind != POLY_ARG_NONE
                         ? poly_uop_tagged_arg(
                               ctx, POLY_OP_SINK, sink->dtype, sink->src, sink->n_src,
                               poly_arg_kernel_info(&kernel_info), sink->tag, sink->tag_arg
                           )
                         : poly_uop(
                               ctx, POLY_OP_SINK, sink->dtype, sink->src, sink->n_src,
                               poly_arg_kernel_info(&kernel_info)
                           );
    if (!estimated_sink) return NULL;
  }

  if (poly_getenv_int("POLY_TRACE_PROGRAM_ESTIMATES", 0)) {
    static _Thread_local int trace_program = 0;
    const PolyEstimates *estimates = estimated_sink->arg.kernel_info->estimates;
    uint64_t ops = 0, lds = 0, mem = 0;
    int rc = estimates ? poly_estimates_infer(estimates, NULL, 0, &ops, &lds, &mem) : -1;
    fprintf(
        stderr, "POLY_TRACE_PROGRAM index=%d rc=%d ops=%llu lds=%llu mem=%llu\n", ++trace_program,
        rc, (unsigned long long)ops, (unsigned long long)lds, (unsigned long long)mem
    );
  }

  PolyUOp *src[2] = {estimated_sink, linear};
  return poly_uop(ctx, POLY_OP_PROGRAM, POLY_VOID, src, 2, program->arg);
}

static PolyUOp *poly_program_attach_linear(PolyCtx *ctx, PolyUOp *program) {
  if (!ctx || !program || program->op != POLY_OP_PROGRAM) return NULL;
  if (poly_program_linear(program)) return program;
  if (program->n_src < 1) return NULL;

  PolyUOp *body = poly_program_kernel_body(program);
  if (!body) return NULL;

  int n_lin = 0;
  PolyUOp **lin = poly_do_linearize(ctx, body, &n_lin);
  if (!lin) return NULL;

  PolyUOp *linear = poly_uop(ctx, POLY_OP_LINEAR, POLY_VOID, lin, n_lin, poly_arg_none());
  free(lin);
  if (!linear) return NULL;

  return poly_program_with_linear(ctx, program, linear);
}

static PolyUOp *poly_program_attach_source(
    PolyCtx *ctx,
    PolyUOp *program,
    const char *source_text
) {
  if (!ctx || !program || program->op != POLY_OP_PROGRAM) return NULL;
  if (poly_program_source(program)) return program;
  if (program->n_src != 2 || !poly_program_linear(program)) return NULL;
  PolyUOp *source = poly_uop0(ctx, POLY_OP_SOURCE, POLY_VOID, poly_arg_str(source_text));
  if (!source) return NULL;
  PolyUOp *src[3] = {program->src[0], program->src[1], source};
  return poly_uop(ctx, POLY_OP_PROGRAM, POLY_VOID, src, 3, program->arg);
}

#ifdef POLY_HAS_X86
static PolyUOp *poly_program_attach_binary(
    PolyCtx *ctx,
    PolyUOp *program,
    const uint8_t *bytes,
    int n_bytes
) {
  if (!ctx || !program || program->op != POLY_OP_PROGRAM || !bytes || n_bytes <= 0) return NULL;
  if (poly_program_binary(program)) return program;
  if (program->n_src != 3 || !poly_program_linear(program) || !poly_program_source(program))
    return NULL;
  PolyUOp *binary = poly_uop0(ctx, POLY_OP_BINARY, POLY_UINT8, poly_arg_bytes(bytes, n_bytes));
  if (!binary) return NULL;
  PolyUOp *src[4] = {program->src[0], program->src[1], program->src[2], binary};
  return poly_uop(ctx, POLY_OP_PROGRAM, POLY_VOID, src, 4, program->arg);
}
#endif

static bool schedule_is_alu_var(const PolyUOp *u) {
  return u && (u->op == POLY_OP_PARAM || u->op == POLY_OP_BUFFER) &&
         u->arg.kind == POLY_ARG_PARAM && u->arg.param && u->arg.param->name &&
         u->arg.param->has_minmax && u->arg.param->addrspace == POLY_ADDR_ALU;
}

static bool schedule_same_var(const PolyUOp *a, const PolyUOp *b) {
  if (a == b) return true;
  const char *an = poly_uop_expr(a), *bn = poly_uop_expr(b);
  return an && bn && strcmp(an, bn) == 0;
}

static bool poly_lookup_var_value(
    PolyUOp *var,
    const PolyVarBinding *bindings,
    int n_bindings,
    PolyArg *out
) {
  if (!var || !bindings || !out) return false;
  for (int i = 0; i < n_bindings; i++) {
    if (!schedule_same_var(bindings[i].var, var)) continue;
    if (poly_dtype_is_bool(var->dtype))
      *out = poly_arg_bool(bindings[i].value != 0);
    else if (poly_dtype_is_float(var->dtype))
      *out = poly_arg_float((double)bindings[i].value);
    else
      *out = poly_arg_int(bindings[i].value);
    return true;
  }
  return false;
}

static bool poly_eval_launch_expr(
    PolyUOp *u,
    const PolyVarBinding *bindings,
    int n_bindings,
    PolyArg *out
) {
  if (!u || !out) return false;
  /* tinygrad UOp._sym_fxn topologically renders each shared symbolic UOp once.
   * Evaluate the same unique-node order into a pass-local contiguous array. */
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_alloc(NULL, u, &n_topo);
  PolyArg *values = n_topo > 0 ? calloc((size_t)n_topo, sizeof(*values)) : NULL;
  PolyMap *memo = poly_map_new((size_t)(n_topo > 0 ? n_topo : 16));
  bool ok = topo && values && memo;

  for (int i = 0; i < n_topo && ok; i++) {
    PolyUOp *node = topo[i];
    PolyArg value = poly_arg_invalid();
    if (node->op == POLY_OP_CONST) {
      /* Launch dimensions and binding values are signed-64 API values.
       * Reject an exact Python-int CONST outside that domain rather than
       * reading its union storage as int64_t. */
      if (node->arg.kind == POLY_ARG_BIGINT)
        ok = false;
      else
        value = node->arg;
    } else if (node->op == POLY_OP_STACK && node->n_src == 0) {
      /* PARAM shape sources are rendered but never consumed by sym_infer;
       * current tinygrad's renderer emits the empty STACK independently. */
      value = poly_arg_int(0);
    } else if (schedule_is_alu_var(node)) {
      ok = poly_lookup_var_value(node, bindings, n_bindings, &value);
    } else if (node->op == POLY_OP_CAST) {
      PolyArg *operand =
          node->n_src >= 1
              ? poly_map_get(memo, poly_ptr_hash(node->src[0]), node->src[0], poly_ptr_eq)
              : NULL;
      if (!operand)
        ok = false;
      else
        /* Pinned tinygrad ops.py:1021-1035 evaluates symbolic CAST through
         * renderer_infer host conversion, without narrowing to dtype width. */
        value = poly_exec_alu(POLY_OP_CAST, node->dtype, operand, 1, false);
    } else if (poly_opset_has(POLY_GROUP_ALU, node->op) && node->n_src <= 3) {
      PolyArg operands[3];
      for (int j = 0; j < node->n_src; j++) {
        PolyArg *operand =
            poly_map_get(memo, poly_ptr_hash(node->src[j]), node->src[j], poly_ptr_eq);
        if (!operand) {
          ok = false;
          break;
        }
        operands[j] = *operand;
      }
      if (ok)
        /* sym_infer evaluates host arithmetic first; launch/estimate range
         * checks consume that value instead of a dtype-wrapped surrogate. */
        value = poly_exec_alu(node->op, node->dtype, operands, node->n_src, false);
    } else {
      ok = false;
    }
    if (!ok || value.kind == POLY_ARG_INVALID) {
      ok = false;
      break;
    }
    values[i] = value;
    poly_map_set(memo, poly_ptr_hash(node), node, &values[i], poly_ptr_eq);
  }

  PolyArg *result = ok ? poly_map_get(memo, poly_ptr_hash(u), u, poly_ptr_eq) : NULL;
  if (result)
    *out = *result;
  else
    ok = false;
  poly_map_destroy(memo);
  free(values);
  poly_toposort_free(topo);
  return ok;
}

static bool poly_launch_arg_to_i64(PolyArg arg, int64_t *out) {
  if (!out) return false;
  switch (arg.kind) {
  case POLY_ARG_BOOL:
    *out = arg.b ? 1 : 0;
    return true;
  case POLY_ARG_FLOAT:
    /* C launch/estimate consumers require an exact representable integer.
     * Checking before conversion avoids C float-to-integer undefined behavior. */
    if (!isfinite(arg.f) || arg.f < -0x1p63 || arg.f >= 0x1p63 || trunc(arg.f) != arg.f)
      return false;
    *out = (int64_t)arg.f;
    return true;
  case POLY_ARG_INT:
    *out = arg.i;
    return true;
  default:
    return false;
  }
}

static int poly_estimate_expr_infer(
    PolyUOp *expr,
    const PolyVarBinding *bindings,
    int n_bindings,
    uint64_t *out
) {
  if (!expr || !out) return -1;
  PolyArg value;
  if (!poly_eval_launch_expr(expr, bindings, n_bindings, &value)) return -1;
  int64_t signed_value = 0;
  if (!poly_launch_arg_to_i64(value, &signed_value)) return -1;
  if (signed_value < 0) return -1;
  *out = (uint64_t)signed_value;
  return 0;
}

int poly_estimates_infer(
    const PolyEstimates *estimates,
    const PolyVarBinding *bindings,
    int n_bindings,
    uint64_t *ops,
    uint64_t *lds,
    uint64_t *mem
) {
  if (!estimates || !ops || !lds || !mem || n_bindings < 0 || (n_bindings > 0 && !bindings))
    return -1;
  return poly_estimate_expr_infer(estimates->ops, bindings, n_bindings, ops) == 0 &&
                 poly_estimate_expr_infer(estimates->lds, bindings, n_bindings, lds) == 0 &&
                 poly_estimate_expr_infer(estimates->mem, bindings, n_bindings, mem) == 0
             ? 0
             : -1;
}

static int poly_launch_dim_upper_bound(PolyCtx *ctx, PolyUOp *expr) {
  if (!expr) return 1;
  int64_t lo = 0, hi = 1;
  poly_uop_minmax(ctx, expr, &lo, &hi);
  if (hi <= 0) return 1;
  if (hi > INT32_MAX) return INT32_MAX;
  return (int)hi;
}

static int poly_resolve_runner_launch_dims(
    PolyRunner *runner,
    const PolyVarBinding *bindings,
    int n_bindings
) {
  if (!runner) return -1;
  int grid[3], block[3];
  memcpy(grid, runner->grid, sizeof(grid));
  memcpy(block, runner->block, sizeof(block));
  for (int dim = 0; dim < 3; dim++) {
    if (runner->grid_exprs[dim]) {
      PolyArg value;
      if (!poly_eval_launch_expr(runner->grid_exprs[dim], bindings, n_bindings, &value)) return -1;
      int64_t resolved = 0;
      if (!poly_launch_arg_to_i64(value, &resolved) || resolved <= 0 || resolved > INT32_MAX)
        return -1;
      grid[dim] = (int)resolved;
    }
    if (runner->block_exprs[dim]) {
      PolyArg value;
      if (!poly_eval_launch_expr(runner->block_exprs[dim], bindings, n_bindings, &value)) return -1;
      int64_t resolved = 0;
      if (!poly_launch_arg_to_i64(value, &resolved) || resolved <= 0 || resolved > INT32_MAX)
        return -1;
      block[dim] = (int)resolved;
    }
    if (grid[dim] <= 0 || block[dim] <= 0) return -1;
  }

  /* Failed inference must not leave a cached runner with mixed old/new dims. */
  memcpy(runner->grid, grid, sizeof(grid));
  memcpy(runner->block, block, sizeof(block));
  return 0;
}

static void poly_runner_apply_program_launch_info(
    PolyCtx *ctx,
    PolyUOp *program,
    PolyRunner *runner
) {
  if (!ctx || !program || !runner || program->op != POLY_OP_PROGRAM) return;
  if (!runner->program) runner->program = program;
  program = runner->program;
  const PolyProgramInfo *info = poly_program_info(ctx, program);
  if (!info) return;

  /* Unit dimensions are metadata too: interpreter runners start zeroed. */
  for (int dim = 0; dim < 3; dim++) {
    int global = info->global_size[dim];
    runner->grid[dim] = global;
    runner->grid_exprs[dim] = info->global_exprs[dim];

    if (info->has_local_size && info->local_exprs[dim]) {
      int local = info->local_size[dim];
      runner->block[dim] = local;
      runner->block_exprs[dim] = info->local_exprs[dim];
    } else if (info->has_local_size) {
      int local = info->local_size[dim];
      runner->block[dim] = local;
      runner->block_exprs[dim] = NULL;
    } else if (!info->has_local_size) {
      runner->block[dim] = 1;
      runner->block_exprs[dim] = NULL;
    }
  }
}

static PolyUOp *linear_rebuild_preserving_metadata(PolyCtx *ctx, PolyUOp *original, PolyUOp **src) {
  if (!ctx || !original || (original->n_src > 0 && !src)) return NULL;
  if (original->tag || original->tag_arg.kind != POLY_ARG_NONE)
    return poly_uop_tagged_arg(
        ctx, original->op, original->dtype, src, original->n_src, original->arg, original->tag,
        original->tag_arg
    );
  return poly_uop(ctx, original->op, original->dtype, src, original->n_src, original->arg);
}

/* C-owned runner lifetime below Tinygrad's runtime_cache
 * (tinygrad/engine/realize.py:get_runtime, lines 125-130). */
static void poly_runner_cleanup(PolyRunner *runner, PolyDevice device) {
  if (!runner) return;
  const PolyBackendDesc *backend = poly_backend_get(device);
  if (runner->handle && !runner->borrowed_handle) {
    if (runner->free_handle)
      runner->free_handle(runner);
    else if (backend)
      backend->free_runner(runner);
  }
  free(runner->var_indices);
  memset(runner, 0, sizeof(*runner));
}

static void poly_runner_cleanup_local_mappings(PolyRunner *runner) {
  if (!runner) return;
  free(runner->var_indices);
  memset(runner, 0, sizeof(*runner));
}

static PolyRuntimeCacheEntry *poly_runtime_cache_entry_new(
    PolyCtx *ctx,
    PolyUOp *program,
    PolyUOp *device_uop,
    PolyDevice device,
    uint32_t env_stamp,
    PolyRunner *runner
) {
  if (!runner) return NULL;
  PolyRuntimeCacheEntry *entry = calloc(1, sizeof(*entry));
  if (!entry) return NULL;
  if (poly_uop_retain(ctx, program) != 0) {
    free(entry);
    return NULL;
  }
  if (poly_uop_retain(ctx, device_uop) != 0) {
    poly_uop_release(ctx, program);
    free(entry);
    return NULL;
  }
  entry->refcount = 1;
  entry->ctx = ctx;
  entry->accounted_bytes = sizeof(*entry);
  if (runner->handle_size > 0) entry->accounted_bytes += (size_t)runner->handle_size;
  entry->program = program;
  entry->device_uop = device_uop;
  entry->device = device;
  entry->env_stamp = env_stamp;
  entry->runner = *runner;
  if (!entry->runner.program) entry->runner.program = program;
  if (ctx) {
    ctx->runtime_artifact_entries++;
    ctx->runtime_artifact_live_bytes += entry->accounted_bytes;
  }
  return entry;
}

static PolyRuntimeCacheEntry *poly_runtime_cache_entry_retain(PolyRuntimeCacheEntry *entry) {
  if (entry) entry->refcount++;
  return entry;
}

static void poly_runtime_cache_entry_release(PolyRuntimeCacheEntry *entry) {
  if (!entry) return;
  entry->refcount--;
  if (entry->refcount > 0) return;
  if (entry->ctx) {
    if (entry->ctx->runtime_artifact_entries > 0) entry->ctx->runtime_artifact_entries--;
    if (entry->ctx->runtime_artifact_live_bytes >= entry->accounted_bytes)
      entry->ctx->runtime_artifact_live_bytes -= entry->accounted_bytes;
    else
      entry->ctx->runtime_artifact_live_bytes = 0;
  }
  poly_runner_cleanup(&entry->runner, entry->device);
  poly_uop_release(entry->ctx, entry->program);
  poly_uop_release(entry->ctx, entry->device_uop);
  free(entry);
}

static bool poly_engine_cache_enabled(void) {
  return poly_getenv_flag_default("POLY_PCACHE", true);
}

static uint32_t poly_program_device_cache_hash(
    PolyUOp *program,
    PolyUOp *device_uop,
    PolyDevice device,
    uint32_t env_stamp
) {
  uint32_t h = poly_ptr_hash(program);
  h ^= poly_ptr_hash(device_uop) + 0x7f4a7c15u + (h << 6) + (h >> 2);
  h ^= (uint32_t)device + 0x9e3779b9u + (h << 6) + (h >> 2);
  h ^= env_stamp + 0x85ebca6bu + (h << 6) + (h >> 2);
  return h;
}

static bool poly_runtime_cache_eq(const void *a, const void *b) {
  const PolyRuntimeCacheMapEntry *ka = a, *kb = b;
  return ka && kb && ka->device_uop == kb->device_uop && ka->device == kb->device &&
         ka->env_stamp == kb->env_stamp && ka->program == kb->program;
}

static bool poly_to_program_cache_eq(const void *a, const void *b) {
  const PolyToProgramCacheEntry *ka = a, *kb = b;
  return ka && kb && ka->device_uop == kb->device_uop && ka->device == kb->device &&
         ka->env_stamp == kb->env_stamp && ka->program == kb->program;
}

static bool to_program_cache_store(PolyCtx *ctx, uint32_t hash, PolyToProgramCacheEntry *entry) {
  if (!ctx || !entry || poly_uop_retain(ctx, entry->program) != 0) return false;
  if (poly_uop_retain(ctx, entry->device_uop) != 0) goto fail_device;
  if (poly_uop_retain(ctx, entry->prepared_program) != 0) goto fail_prepared;
  /* Tinygrad's to_program functools.cache owns the key and prepared PROGRAM.
   * These retains provide the same lifetime below the C map. */
  poly_map_set(ctx->to_program_cache, hash, entry, entry, poly_to_program_cache_eq);
  return true;

fail_prepared:
  poly_uop_release(ctx, entry->device_uop);
fail_device:
  poly_uop_release(ctx, entry->program);
  return false;
}

static void to_program_cache_release(const void *key, void *value, void *userdata) {
  (void)key;
  PolyCtx *ctx = userdata;
  PolyToProgramCacheEntry *entry = value;
  if (!ctx || !entry) return;
  poly_uop_release(ctx, entry->prepared_program);
  poly_uop_release(ctx, entry->device_uop);
  poly_uop_release(ctx, entry->program);
  free(entry);
}

static void poly_runtime_cache_entry_free(const void *key, void *value, void *userdata) {
  (void)key;
  (void)userdata;
  PolyRuntimeCacheMapEntry *entry = value;
  if (!entry) return;
  if (entry->runtime_program) entry->runtime_program->in_cache = false;
  poly_runtime_cache_entry_release(entry->runtime_program);
  free(entry);
}

#ifdef POLY_HAS_CUDA
static void poly_graph_cache_entry_free(const void *key, void *value, void *userdata) {
  (void)key;
  PolyCtx *ctx = userdata;
  PolyGraphCacheEntry *entry = value;
  if (!entry) return;
  poly_cuda_graph_destroy(entry->graph);
  for (int i = 0; i < entry->n_nodes; i++) {
    if (entry->runtime_entries[i]) {
      poly_runner_cleanup_local_mappings(&entry->runners[i]);
      poly_runtime_cache_entry_release(entry->runtime_entries[i]);
    } else {
      poly_runner_cleanup(&entry->runners[i], POLY_DEVICE_CUDA);
    }
  }
  free(entry->runners);
  free(entry->runtime_entries);
  poly_uop_release(ctx, entry->function);
  free(entry);
}

/* Current Tinygrad engine/realize.py:graph_cache lifecycle. */
static void poly_graph_cache_clear(PolyCtx *ctx) {
  if (!ctx || !ctx->graph_cache) return;
  poly_map_foreach(ctx->graph_cache, poly_graph_cache_entry_free, ctx);
  poly_map_clear(ctx->graph_cache);
}
#endif

void poly_runtime_cache_clear(PolyCtx *ctx) {
  if (!ctx || !ctx->runtime_cache) return;
  poly_map_foreach(ctx->runtime_cache, poly_runtime_cache_entry_free, NULL);
  poly_map_clear(ctx->runtime_cache);
}

size_t poly_runtime_cache_len(PolyCtx *ctx) {
  return ctx && ctx->runtime_cache ? poly_map_len(ctx->runtime_cache) : 0;
}

static void poly_runtime_cache_artifact_size_accum(const void *key, void *value, void *userdata) {
  (void)key;
  size_t *total = userdata;
  PolyRuntimeCacheMapEntry *entry = value;
  if (total && entry) *total += sizeof(*entry);
}

size_t poly_runtime_cache_artifact_bytes(PolyCtx *ctx) {
  if (!ctx) return 0;
  size_t total = ctx->runtime_artifact_live_bytes;
  if (ctx->runtime_cache)
    poly_map_foreach(ctx->runtime_cache, poly_runtime_cache_artifact_size_accum, &total);
  return total;
}

void poly_to_program_cache_clear(PolyCtx *ctx) {
  if (!ctx || !ctx->to_program_cache) return;
  poly_map_foreach(ctx->to_program_cache, to_program_cache_release, ctx);
  poly_map_clear(ctx->to_program_cache);
}

size_t poly_to_program_cache_len(PolyCtx *ctx) {
  return ctx && ctx->to_program_cache ? poly_map_len(ctx->to_program_cache) : 0;
}

static void poly_local_size_cache_release(const void *key, void *value, void *userdata) {
  PolyCtx *ctx = userdata;
  poly_uop_release(ctx, value);
  poly_uop_release(ctx, (PolyUOp *)key);
}

void poly_engine_ctx_cleanup(PolyCtx *ctx) {
  /* PolyCtx owns the C maps corresponding to Tinygrad's to_program_cache and
   * runtime_cache; retained LINEAR/PROGRAM UOps own their weak C records. */
#ifdef POLY_HAS_CUDA
  /* Graph nodes borrow runtime-cache CUDA program handles. */
  poly_graph_cache_clear(ctx);
#endif
  poly_runtime_cache_clear(ctx);
  poly_to_program_cache_clear(ctx);
  poly_map_foreach(ctx->local_size_cache, poly_local_size_cache_release, ctx);
  poly_map_clear(ctx->local_size_cache);
}

static void *cpu_alloc(size_t nbytes, void *dev_ctx) {
  (void)dev_ctx;
  return calloc(1, nbytes);
}

static void cpu_free_alloc(const PolyBuffer *buffer, void *dev_ctx) {
  (void)dev_ctx;
  free(buffer ? buffer->ptr : NULL);
}

static int cpu_copy_in(const PolyBuffer *dst, const PolyBuffer *src, size_t n, void *dev_ctx) {
  (void)dev_ctx;
  memcpy(dst->ptr, src->ptr, n);
  return 0;
}

static int cpu_copy_out(const PolyBuffer *dst, const PolyBuffer *src, size_t n, void *dev_ctx) {
  (void)dev_ctx;
  memcpy(dst->ptr, src->ptr, n);
  return 0;
}

static int cpu_copy_between(const PolyBuffer *dst, const PolyBuffer *src, size_t n, void *dev_ctx) {
  (void)dev_ctx;
  memcpy(dst->ptr, src->ptr, n);
  return 0;
}

const PolyAllocator POLY_CPU_ALLOCATOR = {
    .alloc = cpu_alloc,
    .free = cpu_free_alloc,
    .copy_in = cpu_copy_in,
    .copy_out = cpu_copy_out,
    .copy_between = cpu_copy_between,
    .host_addressable = true,
    .dev_ctx = NULL,
};

/* HOST/PYTHON allocator
 *
 * Imported buffers borrow frontend-owned bytes; memory-planned and empty
 * buffers own C allocations. `frontend_release` distinguishes those paths.
 *
 * Browser imports use a NULL pointer host key so JS can own HOST bytes.
 * Retiring that residency notifies the frontend to drop its strong owner.
 * Core-created HOST data may instead expose a Wasm pointer mapping.
 */

static void *host_alloc(size_t nbytes, void *dev_ctx) {
  (void)dev_ctx;
  /* Tinygrad@2026-08-22/a9069c177a9d schedule/memory.py:55-58 may create
   * PYTHON-device arenas for pre-copy dtype conversion. HOST is Polygrad's
   * storage-domain spelling for PYTHON, so internal arenas need owned bytes. */
  return calloc(1, nbytes);
}

static void host_free_alloc(const PolyBuffer *buffer, void *dev_ctx) {
  (void)dev_ctx;
  if (!buffer) return;
  if (buffer->frontend_release)
    buffer->frontend_release((uintptr_t)buffer);
  else if (buffer->memory_accounted)
    free(buffer->ptr);
  else
    poly_frontend_buffer_release_key((uintptr_t)buffer);
}

static int host_copy_in(const PolyBuffer *dst, const PolyBuffer *src, size_t n, void *dev_ctx) {
  (void)dev_ctx;
#ifdef __EMSCRIPTEN__
  if (dst && dst->device == POLY_DEVICE_HOST && src && src->device == POLY_DEVICE_WEBGPU) {
    return poly_webgpu_get_allocator()->copy_out(dst, src, n, NULL);
  }
  if (dst && dst->device == POLY_DEVICE_HOST && dst->ptr && src && src->ptr) {
    memcpy(dst->ptr, src->ptr, n);
    return 0;
  }
  if (dst && dst->device == POLY_DEVICE_HOST && src && src->ptr) {
    uintptr_t key = (uintptr_t)(dst->src ? dst->src : dst);
    return poly_browser_host_copy_in(key, src->ptr, n);
  }
#endif
  if (!dst || !dst->ptr || !src || !src->ptr) return -1;
  memcpy(dst->ptr, src->ptr, n);
  return 0;
}

static int host_copy_out(const PolyBuffer *dst, const PolyBuffer *src, size_t n, void *dev_ctx) {
  (void)dev_ctx;
#ifdef __EMSCRIPTEN__
  if (src && src->device == POLY_DEVICE_HOST && src->ptr && dst && dst->ptr) {
    memcpy(dst->ptr, src->ptr, n);
    return 0;
  }
  if (src && src->device == POLY_DEVICE_HOST && dst && dst->ptr) {
    uintptr_t key = (uintptr_t)(src->src ? src->src : src);
    return poly_browser_host_copy_out(key, dst->ptr, n);
  }
#endif
  if (!dst || !dst->ptr || !src || !src->ptr) return -1;
  memcpy(dst->ptr, src->ptr, n);
  return 0;
}

static int host_copy_between(
    const PolyBuffer *dst,
    const PolyBuffer *src,
    size_t n,
    void *dev_ctx
) {
  (void)dev_ctx;
#ifdef __EMSCRIPTEN__
  if (dst && src && dst->device == POLY_DEVICE_HOST && src->device == POLY_DEVICE_HOST) {
    if (dst->ptr && src->ptr) {
      memcpy(dst->ptr, src->ptr, n);
      return 0;
    }
    uint8_t *tmp = malloc(n);
    if (!tmp) return -1;
    uintptr_t src_key = (uintptr_t)(src->src ? src->src : src);
    uintptr_t dst_key = (uintptr_t)(dst->src ? dst->src : dst);
    int rc = poly_browser_host_copy_out(src_key, tmp, n);
    if (rc == 0) rc = poly_browser_host_copy_in(dst_key, tmp, n);
    free(tmp);
    return rc;
  }
#endif
  if (!dst || !dst->ptr || !src || !src->ptr) return -1;
  memcpy(dst->ptr, src->ptr, n);
  return 0;
}

static const PolyAllocator POLY_HOST_ALLOCATOR = {
    .alloc = host_alloc,
    .free = host_free_alloc,
    .copy_in = host_copy_in,
    .copy_out = host_copy_out,
    .copy_between = host_copy_between,
    .host_addressable =
#ifdef __EMSCRIPTEN__
        false,
#else
        true,
#endif
    .dev_ctx = NULL,
};

/* CUDA allocator */

#ifdef POLY_HAS_CUDA

static void *cuda_alloc_fn(size_t nbytes, void *dev_ctx) {
  (void)dev_ctx;
  unsigned long long dptr = poly_cuda_alloc(nbytes);
  return dptr ? (void *)(uintptr_t)dptr : NULL;
}

static void cuda_free_fn(const PolyBuffer *buffer, void *dev_ctx) {
  (void)dev_ctx;
  if (buffer && buffer->ptr) poly_cuda_free((unsigned long long)(uintptr_t)buffer->ptr);
}

static int cuda_copy_in_fn(const PolyBuffer *dst, const PolyBuffer *src, size_t n, void *dev_ctx) {
  (void)dev_ctx;
  return poly_cuda_copy_htod((unsigned long long)(uintptr_t)dst->ptr, src->ptr, n);
}

static int cuda_copy_out_fn(const PolyBuffer *dst, const PolyBuffer *src, size_t n, void *dev_ctx) {
  (void)dev_ctx;
  return poly_cuda_copy_dtoh(dst->ptr, (unsigned long long)(uintptr_t)src->ptr, n);
}

static int cuda_copy_between_fn(
    const PolyBuffer *dst,
    const PolyBuffer *src,
    size_t n,
    void *dev_ctx
) {
  (void)dev_ctx;
  return poly_cuda_copy_dtod(
      (unsigned long long)(uintptr_t)dst->ptr, (unsigned long long)(uintptr_t)src->ptr, n
  );
}

const PolyAllocator POLY_CUDA_ALLOCATOR = {
    .alloc = cuda_alloc_fn,
    .free = cuda_free_fn,
    .copy_in = cuda_copy_in_fn,
    .copy_out = cuda_copy_out_fn,
    .copy_between = cuda_copy_between_fn,
    .dev_ctx = NULL,
};

#endif /* POLY_HAS_CUDA */

/* HIP allocator */

#ifdef POLY_HAS_HIP

static void *hip_alloc_fn(size_t nbytes, void *dev_ctx) {
  (void)dev_ctx;
  return poly_hip_alloc(nbytes);
}

static void hip_free_fn(const PolyBuffer *buffer, void *dev_ctx) {
  (void)dev_ctx;
  if (buffer && buffer->ptr) poly_hip_free(buffer->ptr);
}

static int hip_copy_in_fn(const PolyBuffer *dst, const PolyBuffer *src, size_t n, void *dev_ctx) {
  (void)dev_ctx;
  return poly_hip_copy_htod(dst->ptr, src->ptr, n);
}

static int hip_copy_out_fn(const PolyBuffer *dst, const PolyBuffer *src, size_t n, void *dev_ctx) {
  (void)dev_ctx;
  return poly_hip_copy_dtoh(dst->ptr, src->ptr, n);
}

static int hip_copy_between_fn(
    const PolyBuffer *dst,
    const PolyBuffer *src,
    size_t n,
    void *dev_ctx
) {
  (void)dev_ctx;
  (void)dst;
  (void)src;
  (void)n;
  fprintf(stderr, "polygrad: hip_copy_between: not implemented\n");
  return -1;
}

const PolyAllocator POLY_HIP_ALLOCATOR = {
    .alloc = hip_alloc_fn,
    .free = hip_free_fn,
    .copy_in = hip_copy_in_fn,
    .copy_out = hip_copy_out_fn,
    .copy_between = hip_copy_between_fn,
    .dev_ctx = NULL,
};

#endif /* POLY_HAS_HIP */

/* ══════════════════════════════════════════════════════════════════════ */
/*  Backend-specific runner handle types                                 */
/* ══════════════════════════════════════════════════════════════════════ */

/* Interpreter: linearized UOp array */
typedef struct {
  PolyCtx *ctx;
  PolyUOp **lin;
  int n_lin;
} InterpHandle;

/* CUDA: compiled program handle */
#ifdef POLY_HAS_CUDA
typedef struct {
  PolyCudaProgram *prog;
} CudaRunnerHandle;
#endif

/* HIP: compiled program handle */
#ifdef POLY_HAS_HIP
typedef struct {
  PolyHipProgram *prog;
} HipRunnerHandle;
#endif

static bool poly_program_var_is_runtime(PolyUOp *var) {
  const char *name = poly_uop_expr(var);
  return name && strcmp(name, "core_id") == 0;
}

static int poly_bind_runner_vars(PolyCtx *ctx, PolyRunner *runner) {
  runner->n_vars = 0;
  runner->var_indices = NULL;
  if (!runner->program) return 0;
  const PolyProgramInfo *info = poly_program_info(ctx, runner->program);
  if (!info) return -1;
  for (int i = 0; i < info->n_vars; i++)
    if (!poly_program_var_is_runtime(info->vars[i])) runner->n_vars++;
  if (runner->n_vars == 0) return 0;
  runner->var_indices = malloc((size_t)runner->n_vars * sizeof(*runner->var_indices));
  if (!runner->var_indices) {
    runner->n_vars = 0;
    return -1;
  }
  int out = 0;
  for (int i = 0; i < info->n_vars; i++)
    if (!poly_program_var_is_runtime(info->vars[i])) runner->var_indices[out++] = i;
  return 0;
}

static const char *poly_program_arg_name(PolyUOp *program) {
  if (!program) return "test";
  if (program->arg.kind == POLY_ARG_PROGRAM_INFO && program->arg.program_info &&
      program->arg.program_info->name)
    return program->arg.program_info->name;
  if (program->arg.kind != POLY_ARG_STRING || !program->arg.str) return "test";
  return program->arg.str;
}

/* Pinned tinygrad compile_linear sends both SINK and PROGRAM call bodies to
 * to_program. do_to_program starts from an existing PROGRAM and pm_to_program
 * only fills missing LINEAR/SOURCE/BINARY stages, so a complete PROGRAM is a
 * fixed point (tinygrad/codegen/__init__.py:208-248 and
 * tinygrad/engine/realize.py:243-253). Imported executable artifacts cross
 * this same boundary: reconstruct runtime handles, but never run their kernel
 * body through backend rewrites a second time. */
/* do_to_program restores ProgramInfo without replacing supplied stages;
 * pm_to_program computes estimates only at the SINK+LINEAR boundary. */
static PolyUOp *poly_program_complete_metadata(
    PolyCtx *ctx,
    PolyUOp *call,
    PolyUOp *program,
    PolyDevice device
) {
  PolyUOp *sink = poly_program_kernel_body(program);
  if (!sink || sink->op != POLY_OP_SINK) return NULL;
  if (!poly_program_info(ctx, program)) {
    PolyProgramInfo *info =
        poly_program_info_build(ctx, call, sink, poly_kernel_name(sink, "test"), device);
    if (!info) return NULL;
    program = poly_uop_tagged_arg(
        ctx, program->op, program->dtype, program->src, program->n_src, poly_arg_program_info(info),
        program->tag, program->tag_arg
    );
    poly_program_info_destroy(info);
    if (!program) return NULL;
  }
  if (program->n_src == 2 && poly_program_linear(program) && !poly_program_estimates(program))
    return poly_program_with_linear(ctx, program, program->src[1]);
  return program;
}

static bool poly_program_is_complete_for_backend(
    PolyUOp *program,
    const PolyBackendDesc *backend,
    PolyDevice device
) {
  if (!program || program->op != POLY_OP_PROGRAM || !backend) return false;
#ifdef POLY_HAS_X86
  if (device == POLY_DEVICE_X86)
    return poly_program_linear(program) &&
           (poly_program_binary(program) || poly_program_source(program));
#else
  (void)device;
#endif
  if (!backend->rewrite_program) return true;
  if (!poly_program_linear(program)) return false;
  return !backend->render_source || poly_program_source(program);
}

#ifdef POLY_HAS_X86
static PolyUOp *poly_prepare_x86_program_for_backend(
    PolyCtx *ctx,
    PolyUOp *call,
    PolyUOp *base,
    PolyUOp *device_uop,
    PolyDevice device,
    uint32_t env_stamp,
    bool cache
);
#endif

static PolyUOp *poly_prepare_program_for_backend(
    PolyCtx *ctx,
    PolyUOp *call,
    PolyUOp *device_uop,
    PolyDevice device,
    uint32_t env_stamp,
    bool cache
) {
  if (!ctx || !call || call->op != POLY_OP_CALL) return NULL;
  PolyUOp *ast = poly_call_raw_body(call);
  if (!ast) return NULL;
  const PolyBackendDesc *backend = poly_backend_get(device);
  if (!backend) return NULL;
  PolyUOp *prepared_input =
      ast->op == POLY_OP_PROGRAM ? poly_program_complete_metadata(ctx, call, ast, device) : ast;
  if (!prepared_input) return NULL;
  if (poly_program_is_complete_for_backend(prepared_input, backend, device)) return prepared_input;
#ifdef POLY_HAS_X86
  if (device == POLY_DEVICE_X86)
    return poly_prepare_x86_program_for_backend(
        ctx, call, prepared_input, device_uop, device, env_stamp, cache
    );
#endif

  /* Pinned to_program caches the raw SINK ast.key before do_to_program builds
   * ProgramInfo (codegen/__init__.py:244-250). A PROGRAM input remains a
   * supported explicit precompiled boundary, but raw scheduled SINKs use the
   * same cache key here instead of being wrapped during schedule creation. */
  PolyToProgramCacheEntry key = {
      .program = ast,
      .device_uop = device_uop,
      .device = device,
      .env_stamp = env_stamp,
      .prepared_program = NULL,
  };
  uint32_t hash = poly_program_device_cache_hash(ast, device_uop, device, env_stamp);
  PolyToProgramCacheEntry *entry =
      (cache && poly_engine_cache_enabled() && ctx->to_program_cache)
          ? poly_map_get(ctx->to_program_cache, hash, &key, poly_to_program_cache_eq)
          : NULL;
  if (entry) return entry->prepared_program;

  PolyUOp *body = poly_program_body(ast);
  if (!body) return NULL;

  /* do_to_program resumes PROGRAM stages; only a raw SINK needs full lowering. */
  PolyUOp *prepared = prepared_input;
  if (ast->op != POLY_OP_PROGRAM) {
    PolyUOp *rewritten = backend->rewrite_program ? backend->rewrite_program(ctx, body) : body;
    if (!rewritten) return NULL;
    prepared =
        poly_program_from_call_body(ctx, call, rewritten, poly_program_arg_name(ast), device);
  }
  if (!prepared) return NULL;
  if (backend->rewrite_program) {
    prepared = poly_program_attach_linear(ctx, prepared);
    if (!prepared) return NULL;
    prepared = poly_program_ensure_source(ctx, backend, prepared, device, cache ? NULL : "test");
    if (!prepared) return NULL;
  }

  if (cache && poly_engine_cache_enabled() && ctx->to_program_cache) {
    entry = malloc(sizeof(*entry));
    if (entry) {
      entry->program = ast;
      entry->device_uop = device_uop;
      entry->device = device;
      entry->env_stamp = env_stamp;
      entry->prepared_program = prepared;
      if (!to_program_cache_store(ctx, hash, entry)) free(entry);
    }
  }

  return prepared;
}

static PolyUOp *poly_program_ensure_source(
    PolyCtx *ctx,
    const PolyBackendDesc *backend,
    PolyUOp *program,
    PolyDevice device,
    const char *name_override
) {
  if (!ctx || !backend || !program || program->op != POLY_OP_PROGRAM) return NULL;
  if (!backend->render_source || poly_program_source(program)) return program;

  char fn_name[64];
  stable_kernel_fn_name(ctx, fn_name, sizeof(fn_name), device, program);
  char *source = backend->render_source(ctx, program, name_override ? name_override : fn_name);
  if (!source) return NULL;
  g_program_source_render_count++;
  PolyUOp *with_source = poly_program_attach_source(ctx, program, source);
  free(source);
  return with_source;
}

static int poly_lower_compute_call_cached(
    PolyCtx *ctx,
    PolyUOp *call,
    PolyUOp *device_uop,
    PolyDevice device,
    uint32_t env_stamp,
    PolyRunner *out,
    bool cache,
    PolyRuntimeCacheEntry **runtime_entry_out
) {
  if (runtime_entry_out) *runtime_entry_out = NULL;
  const PolyBackendDesc *backend = poly_backend_get(device);
  if (!backend || !backend->lower_item) return -1;
  if (poly_backend_ensure_open(device) != 0) return -1;

  PolyUOp *program =
      poly_prepare_program_for_backend(ctx, call, device_uop, device, env_stamp, true);
  PolyUOp *body = poly_program_kernel_body(program);
  /* Tinygrad 2026-08-22/a9069c177a9d codegen/__init__.py:289-396 owns
   * representation validation; engine/realize.py:263-319 executes PROGRAMs
   * without a second spec_program pass. */
  if (!body) return -2;

  PolyRuntimeCacheMapEntry key = {
      .program = program,
      .device_uop = device_uop,
      .device = device,
      .env_stamp = env_stamp,
  };
  uint32_t hash = poly_program_device_cache_hash(program, device_uop, device, env_stamp);
  PolyRuntimeCacheMapEntry *entry =
      (poly_engine_cache_enabled() && ctx && ctx->runtime_cache)
          ? poly_map_get(ctx->runtime_cache, hash, &key, poly_runtime_cache_eq)
          : NULL;

  if (!entry) {
    if (poly_engine_cache_enabled() && ctx && ctx->runtime_cache) ctx->runtime_cache_misses++;
    char fn_name[64];
    stable_kernel_fn_name(ctx, fn_name, sizeof(fn_name), device, program);

    PolyRunner lowered = {0};
    if (backend->lower_item(ctx, program, fn_name, &lowered) != 0) return -1;

    /* get_runtime(cache=False) reuses hits above, but never publishes a miss. */
    if (cache && poly_engine_cache_enabled() && ctx && ctx->runtime_cache) {
      entry = calloc(1, sizeof(*entry));
      PolyRuntimeCacheEntry *runtime_entry =
          entry
              ? poly_runtime_cache_entry_new(ctx, program, device_uop, device, env_stamp, &lowered)
              : NULL;
      if (entry && runtime_entry) {
        entry->program = program;
        entry->device_uop = device_uop;
        entry->device = device;
        entry->env_stamp = env_stamp;
        entry->runtime_program = poly_runtime_cache_entry_retain(runtime_entry);
        entry->runtime_program->in_cache = true;
        /* runtime_cache owns only the compiled handle; call-specific argument
         * and variable mappings are rebuilt by run_linear. */
        runtime_entry->runner.n_params = 0;
        runtime_entry->runner.var_indices = NULL;
        runtime_entry->runner.n_vars = 0;
        poly_map_set(ctx->runtime_cache, hash, entry, entry, poly_runtime_cache_eq);
        if (runtime_entry_out) *runtime_entry_out = runtime_entry;
        *out = runtime_entry->runner;
        out->n_params = 0;
        out->var_indices = NULL;
        out->n_vars = 0;
        out->borrowed_handle = true;
        poly_runner_apply_program_launch_info(ctx, program, out);
        return 0;
      } else {
        free(entry);
        poly_runtime_cache_entry_release(runtime_entry);
        *out = lowered;
        poly_runner_apply_program_launch_info(ctx, program, out);
        return 0;
      }
    } else {
      *out = lowered;
      poly_runner_apply_program_launch_info(ctx, program, out);
      return 0;
    }
  }

  if (!entry || !entry->runtime_program) return -1;
  if (poly_engine_cache_enabled() && ctx && ctx->runtime_cache) ctx->runtime_cache_hits++;
  if (runtime_entry_out)
    *runtime_entry_out = poly_runtime_cache_entry_retain(entry->runtime_program);
  *out = entry->runtime_program->runner;
  out->n_params = 0;
  out->var_indices = NULL;
  out->n_vars = 0;
  out->borrowed_handle = true;
  poly_runner_apply_program_launch_info(ctx, program, out);
  return 0;
}

/* ══════════════════════════════════════════════════════════════════════ */
/*  Backend implementations (lower_item / execute / free_runner)         */
/* ══════════════════════════════════════════════════════════════════════ */

static void timing_free_args(void **bufs, int n_params) {
  if (bufs)
    for (int i = 0; i < n_params; i++)
      free(bufs[i]);
  free(bufs);
}

void **poly_args_from_ast(PolyCtx *ctx, PolyUOp *sink, int *n_args) {
  *n_args = 0;
  /* opt/postrange.py:args_from_ast, adapted to the existing C wrapper's
   * compact buffer-then-scalar arguments and implicit core_id. */
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, sink, &n_topo);
  int n_params = 0;
  void **bufs = NULL;
  if (!topo) goto cleanup;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (u->op != POLY_OP_PARAM) continue;
    if (u->arg.kind != POLY_ARG_PARAM || !u->arg.param) goto cleanup;
    const char *name = poly_uop_expr(u);
    if (poly_uop_is_alu_param(u) && name && strcmp(name, "core_id") == 0) continue;
    topo[n_params++] = u;
  }
  for (int i = 1; i < n_params; i++) {
    PolyUOp *param = topo[i];
    bool scalar = poly_uop_is_alu_param(param);
    int j = i;
    while (j > 0) {
      bool prior_scalar = poly_uop_is_alu_param(topo[j - 1]);
      if (prior_scalar < scalar || (prior_scalar == scalar && poly_program_buffer_slot(topo[j - 1]
                                                              ) <= poly_program_buffer_slot(param)))
        break;
      topo[j] = topo[j - 1];
      j--;
    }
    topo[j] = param;
  }
  /* search._time_program passes the complete rawbufs list. The rendered
   * wrapper reads every argument; silently truncating it corrupts memory. */
  bufs = calloc((size_t)(n_params ? n_params : 1), sizeof(*bufs));
  if (!bufs) goto cleanup;
  for (int i = 0; i < n_params; i++) {
    PolyUOp *param = topo[i];
    bool scalar = poly_uop_is_alu_param(param);
    /* The existing native wrapper takes all ALU values through int*, even
     * when its typed function subsequently converts to another dtype. */
    int itemsize = scalar ? (int)sizeof(int) : poly_dtype_itemsize(param->dtype);
    int64_t sz = scalar ? 1 : poly_uop_max_numel(ctx, param);
    if (sz < 0 || itemsize <= 0 || (uint64_t)sz > SIZE_MAX / (size_t)itemsize) goto cleanup;
    bufs[i] = calloc((size_t)(sz ? sz : 1), (size_t)itemsize);
    if (!bufs[i]) goto cleanup;
    if (scalar) {
      const PolyParamArg *arg = param->arg.param;
      int64_t lo = 0, hi = 0;
      if (!arg->has_minmax || !poly_arg_integer_to_i64(arg->min_val, &lo) ||
          !poly_arg_integer_to_i64(arg->max_val, &hi) || lo > hi ||
          (!poly_dtype_is_int(param->dtype) && !poly_dtype_eq(param->dtype, POLY_BOOL)))
        goto cleanup;
      /* Python's (lo+hi)//2, without overflowing the C sum. */
      int64_t value = lo + (int64_t)(((uint64_t)hi - (uint64_t)lo) / 2);
      if (value < INT_MIN || value > INT_MAX) goto cleanup;
      *(int *)bufs[i] = (int)value;
    } else if (poly_dtype_eq(param->dtype, POLY_FLOAT32)) {
      /* Preserve existing finite f32 timing inputs; other storage is zeroed. */
      float *fp = bufs[i];
      for (int64_t j = 0; j < sz; j++)
        fp[j] = 0.1f + (float)(j % 100) * 0.01f;
    }
  }
  poly_toposort_free(topo);

  *n_args = n_params;
  return bufs;
cleanup:
  poly_toposort_free(topo);
  timing_free_args(bufs, n_params);
  return NULL;
}

void poly_timing_buffers_free(PolyTimingBuffers *raw) {
  if (raw->buffers && raw->allocator && !raw->allocator->host_addressable)
    for (int i = 0; i < raw->count; i++)
      if (raw->buffers[i].ptr) raw->allocator->free(&raw->buffers[i], raw->allocator->dev_ctx);
  free(raw->buffers);
  free(raw->slots);
  timing_free_args(raw->host, raw->n_host);
  *raw = (PolyTimingBuffers){0};
}

bool poly_timing_buffers_init(
    PolyCtx *ctx,
    PolyUOp *sink,
    PolyDevice device,
    PolyTimingBuffers *raw
) {
  *raw = (PolyTimingBuffers){0};
  const PolyBackendDesc *backend = poly_backend_get(device);
  if (!backend || poly_backend_ensure_open(device) != 0 || !backend->get_allocator) return false;
  raw->allocator = backend->get_allocator();
  raw->host = poly_args_from_ast(ctx, sink, &raw->n_host);
  int n = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, sink, &n);
  if (!raw->host || !raw->allocator || !topo) goto failed;
  for (int i = 0; i < n; i++)
    if (topo[i]->op == POLY_OP_PARAM && !poly_uop_is_alu_param(topo[i]))
      topo[raw->count++] = topo[i];
  for (int i = 1; i < raw->count; i++) {
    PolyUOp *param = topo[i];
    int j = i;
    while (j && poly_program_buffer_slot(topo[j - 1]) > poly_program_buffer_slot(param)) {
      topo[j] = topo[j - 1];
      j--;
    }
    topo[j] = param;
  }
  if (raw->count > raw->n_host) goto failed;
  raw->buffers = calloc((size_t)(raw->count ? raw->count : 1), sizeof(*raw->buffers));
  raw->slots = calloc((size_t)(raw->count ? raw->count : 1), sizeof(*raw->slots));
  if (!raw->buffers || !raw->slots) goto failed;
  for (int i = 0; i < raw->count; i++) {
    int itemsize = poly_dtype_itemsize(topo[i]->dtype);
    int64_t count = poly_uop_max_numel(ctx, topo[i]);
    raw->slots[i] = poly_program_buffer_slot(topo[i]);
    if (raw->slots[i] < 0 || (i && raw->slots[i] == raw->slots[i - 1]) || count < 0 ||
        itemsize <= 0 || (uint64_t)(count ? count : 1) > SIZE_MAX / (size_t)itemsize)
      goto failed;
    size_t nbytes = (size_t)(count ? count : 1) * (size_t)itemsize;
    raw->buffers[i] = (PolyBuffer
    ){.ptr = raw->host[i],
      .nbytes = nbytes,
      .device = device,
      .allocator = raw->allocator,
      .valid = true};
    if (raw->allocator->host_addressable) continue;
    raw->buffers[i].ptr = raw->allocator->alloc(nbytes, raw->allocator->dev_ctx);
    if (!raw->buffers[i].ptr) goto failed;
    raw->buffers[i].owned = true;
    PolyBuffer host = {
        .ptr = raw->host[i], .nbytes = nbytes, .device = POLY_DEVICE_CPU, .valid = true};
    if (!raw->allocator->copy_in ||
        raw->allocator->copy_in(&raw->buffers[i], &host, nbytes, raw->allocator->dev_ctx) != 0)
      goto failed;
  }
  poly_toposort_free(topo);
  return true;
failed:
  poly_toposort_free(topo);
  poly_timing_buffers_free(raw);
  return false;
}

double poly_time_program(
    PolyCtx *ctx,
    PolyRunner *runner,
    PolyDevice device,
    const PolyTimingBuffers *raw,
    int reps,
    double early_stop_us,
    int max_global_size
) {
  int n_args = runner->n_params + runner->n_vars;
  void **args = calloc((size_t)(n_args ? n_args : 1), sizeof(*args));
  PolyVarBinding *bindings =
      calloc((size_t)(runner->n_vars ? runner->n_vars : 1), sizeof(*bindings));
  int *values = calloc((size_t)(runner->n_vars ? runner->n_vars : 1), sizeof(*values));
  const PolyProgramInfo *info = poly_program_info(ctx, runner->program);
  double elapsed = INFINITY;
  if (!args || !bindings || !values || !info) goto cleanup;
  for (int i = 0; i < runner->n_params; i++) {
    for (int j = 0; j < raw->count; j++)
      if (raw->slots[j] == info->globals[i]) {
        args[i] = raw->buffers[j].ptr;
        break;
      }
    if (!args[i]) goto cleanup;
  }
  for (int i = 0; i < runner->n_vars; i++) {
    PolyUOp *var = info->vars[runner->var_indices[i]];
    int64_t lo, hi;
    poly_uop_minmax(ctx, var, &lo, &hi);
    if (lo > hi) goto cleanup;
    int64_t value = lo + (int64_t)(((uint64_t)hi - (uint64_t)lo) / 2);
    if (value < INT_MIN || value > INT_MAX) goto cleanup;
    values[i] = (int)value;
    bindings[i] = (PolyVarBinding){.var = var, .value = value};
    args[runner->n_params + i] = &values[i];
  }
  elapsed = poly_time_call(
      runner, device, args, n_args, bindings, runner->n_vars, reps, early_stop_us, max_global_size
  );
cleanup:
  free(bindings);
  free(values);
  free(args);
  return elapsed;
}

int poly_time_call_prepare(PolyCtx *ctx, PolyUOp *call, PolyDevice device, PolyRunner *out) {
  if (!ctx || !call || !out) return -1;
  *out = (PolyRunner){0};
  const PolyBackendDesc *backend = poly_backend_get(device);
  if (!backend || !backend->lower_item || poly_backend_ensure_open(device) != 0) return -1;
  PolyUOp *program = poly_prepare_program_for_backend(
      ctx, call, poly_device_uop(ctx, device), device, poly_runtime_cache_env_stamp(), false
  );
  PolyUOp *linear = poly_program_linear(program);
  int limit = poly_getenv_int("BEAM_UOPS_MAX", 3000);
  if (!program || !linear) return -1;
  out->capture_binary = true;
  int status = backend->lower_item(ctx, program, "test", out);
  if (status != 0) {
    poly_runner_cleanup(out, device);
    return status;
  }
  /* search._try_compile tests the UOp budget after to_program compiles.
   * Rejecting earlier would hide unexpected compiler failures in strict mode. */
  if (limit > 0 && linear->n_src >= limit) {
    poly_runner_cleanup(out, device);
    return -1;
  }
  poly_runner_apply_program_launch_info(ctx, program, out);
  const PolyProgramInfo *info = poly_program_info(ctx, program);
  out->n_params = info->n_globals;
  if (poly_bind_runner_vars(ctx, out) != 0) {
    poly_runner_cleanup(out, device);
    return -1;
  }
  poly_uop_retain(ctx, out->program);
  if (out->compiled_binary) poly_uop_retain(ctx, out->compiled_binary);
  return 0;
}

double poly_time_call(
    PolyRunner *runner,
    PolyDevice device,
    void **args,
    int n_args,
    const PolyVarBinding *bindings,
    int n_bindings,
    int count,
    double early_stop_us,
    int max_global_size
) {
  const PolyBackendDesc *backend = poly_backend_get(device);
  if (!runner || !backend || count <= 0 || n_args != runner->n_params + runner->n_vars ||
      poly_resolve_runner_launch_dims(runner, bindings, n_bindings) != 0)
    return INFINITY;
  int grid[3];
  double original = 1, size = 1;
  for (int i = 0; i < 3; i++) {
    grid[i] = runner->grid[i] > 0 ? runner->grid[i] : 1;
    original *= grid[i];
  }
  size = original;
  while (max_global_size > 0 && size > max_global_size) {
    bool changed = false;
    for (int i = 2; i >= 0; i--) {
      if (grid[i] <= 16) continue;
      grid[i] /= 2;
      size = (double)grid[0] * grid[1] * grid[2];
      changed = true;
      break;
    }
    if (!changed) return INFINITY; /* Invalid test-grid budget must not hang the caller. */
  }
  int saved_grid[3];
  memcpy(saved_grid, runner->grid, sizeof(saved_grid));
  memcpy(runner->grid, grid, sizeof(grid));
  double best = INFINITY;
  bool saved_wait = runner->wait;
  runner->wait = true;
  for (int i = 0; i < count; i++) {
    runner->elapsed_us = NAN;
    double start = poly_now_ms();
    int rc = runner->execute    ? runner->execute(runner, args, n_args)
             : backend->execute ? backend->execute(runner, args, n_args)
                                : -1;
#ifdef POLY_HAS_HIP
    if (rc == 0 && device == POLY_DEVICE_HIP) rc = poly_hip_sync();
#endif
    double measured =
        isfinite(runner->elapsed_us) ? runner->elapsed_us : (poly_now_ms() - start) * 1000.0;
    double elapsed = measured * original / size;
    if (rc != 0) {
      best = INFINITY;
      break;
    }
    if (elapsed < best) best = elapsed;
    if (best > early_stop_us) break;
  }
  memcpy(runner->grid, saved_grid, sizeof(saved_grid));
  runner->wait = saved_wait;
  return best;
}

void poly_time_call_finish(PolyCtx *ctx, PolyRunner *runner, PolyDevice device) {
  if (!runner) return;
  PolyUOp *program = runner->program;
  PolyUOp *binary = runner->compiled_binary;
  poly_runner_cleanup(runner, device);
  if (program) poly_uop_release(ctx, program);
  if (binary) poly_uop_release(ctx, binary);
}

/* CPU backend */

#ifndef __EMSCRIPTEN__

static int cpu_execute_fn(void *self, void **args, int n_args);
static void cpu_free_fn(void *self);

static PolyRewriteOpts cpu_schedule_rewrite_opts(void) {
  return (PolyRewriteOpts){
      .optimize = true,
      .caps = poly_c_renderer_caps(),
      .device = POLY_DEVICE_CPU,
      .opt_policy = POLY_OPT_HEURISTIC,
      .extra_matcher = poly_clang_renderer_extra_matcher(),
  };
}

static PolyUOp *cpu_rewrite_program(PolyCtx *ctx, PolyUOp *sink) {
  PolyRewriteOpts opts = cpu_schedule_rewrite_opts();
  opts.optimize = poly_kernel_optimize_enabled(sink);
  opts.beam_width = poly_kernel_beam(sink);
  return poly_full_rewrite_to_sink_ex(ctx, sink, opts);
}

static char *cpu_render_source_impl(PolyCtx *ctx, PolyUOp *program, const char *fn_name) {
  PolyUOp *scheduled_root = poly_program_kernel_body(program);
  if (!scheduled_root) return NULL;
  int n_lin;
  bool lin_owned = false;
  PolyUOp **lin = poly_program_linear_uops(program, &n_lin);
  if (!lin) {
    lin = poly_do_linearize(ctx, scheduled_root, &n_lin);
    lin_owned = true;
  }
  if (!lin) return NULL;
  if (poly_debug_at_least(3)) {
    int n_weak = 0;
    for (int i = 0; i < n_lin; i++)
      if (lin[i] && poly_dtype_is_index(lin[i]->dtype)) n_weak++;
    fprintf(stderr, "[polygrad:cpu_lower] fn=%s linear=%d weakint=%d\n", fn_name, n_lin, n_weak);
    fflush(stderr);
  }

  char *src = poly_render_c(ctx, lin, n_lin, fn_name);
  if (lin_owned) free(lin);
  return src;
}

static char *cpu_render_source(PolyCtx *ctx, PolyUOp *program, const char *fn_name) {
  return cpu_render_source_impl(ctx, program, fn_name);
}

static int cpu_lower_item_impl(
    PolyCtx *ctx,
    PolyUOp *program,
    const char *fn_name,
    PolyRunner *out
) {
  PolyUOp *scheduled_root = poly_program_kernel_body(program);
  if (!scheduled_root) return -1;
  int n_lin;
  bool lin_owned = false;
  PolyUOp **lin = poly_program_linear_uops(program, &n_lin);
  if (!lin) {
    lin = poly_do_linearize(ctx, scheduled_root, &n_lin);
    lin_owned = true;
  }
  if (!lin) return -1;
  const PolyProgramInfo *program_info = poly_program_info(ctx, program);
  int cpu_threads =
      program_info && program_info->global_size[0] > 1 ? program_info->global_size[0] : 1;

  const char *src = poly_program_source_text(program);
  char *src_owned = NULL;
  if (!src) {
    src_owned = cpu_render_source_impl(ctx, program, fn_name);
    src = src_owned;
  }
  if (!src) {
    if (lin_owned) free(lin);
    return -1;
  }

  if (poly_dump_kernels_enabled())
    fprintf(stderr, "=== LOWER KERNEL %s ===\n%s\n=== END ===\n", fn_name, src);

  PolyProgram *prog = poly_compile_c(src, fn_name);
  if (!prog) {
    fprintf(stderr, "=== FAILED LOWER KERNEL %s ===\n%s\n=== END ===\n", fn_name, src);
    free(src_owned);
    if (lin_owned) free(lin);
    /* compiler_cpu.ClangCompiler.compile raises CalledProcessError/OSError;
     * strict BEAM propagates these rather than treating them as RuntimeError. */
    return -2;
  }
  free(src_owned);
  if (lin_owned) free(lin);

  if (out->capture_binary) {
    int size = 0;
    uint8_t *bytes = poly_program_read_binary(prog, &size);
    if (!bytes) {
      poly_program_destroy(prog);
      return -1;
    }
    out->compiled_binary = poly_uop0(ctx, POLY_OP_BINARY, POLY_UINT8, poly_arg_bytes(bytes, size));
    free(bytes);
    if (!out->compiled_binary) {
      poly_program_destroy(prog);
      return -1;
    }
  }
  out->kind = POLY_RUNNER_COMPILED;
  out->handle = prog;
  size_t prog_size = poly_program_estimated_size(prog);
  out->handle_size = prog_size > (size_t)INT_MAX ? INT_MAX : (int)prog_size;
  out->grid[0] = cpu_threads;
  out->grid[1] = 1;
  out->grid[2] = 1;
  out->block[0] = 1;
  out->block[1] = 1;
  out->block[2] = 1;
  out->execute = cpu_execute_fn;
  out->free_handle = cpu_free_fn;
  return 0;
}

static int cpu_lower_item(PolyCtx *ctx, PolyUOp *program, const char *fn_name, PolyRunner *out) {
  return cpu_lower_item_impl(ctx, program, fn_name, out);
}

static int cpu_execute_fn(void *self, void **args, int n_args) {
  PolyRunner *runner = (PolyRunner *)self;
  int threads = runner->grid[0] > 1 ? runner->grid[0] : 1;
  if (threads > 1)
    poly_program_call_threaded((PolyProgram *)runner->handle, args, n_args, threads);
  else
    poly_program_call((PolyProgram *)runner->handle, args, n_args);
  return 0;
}
static int cpu_execute(PolyRunner *runner, void **args, int n_args) {
  return cpu_execute_fn(runner, args, n_args);
}

static void cpu_free_fn(void *self) {
  PolyRunner *runner = (PolyRunner *)self;
  if (runner->handle) poly_program_destroy((PolyProgram *)runner->handle);
}
static void cpu_free_runner(PolyRunner *runner) {
  cpu_free_fn(runner);
}

static const PolyAllocator *cpu_get_allocator(void) {
  return &POLY_CPU_ALLOCATOR;
}

#endif /* !__EMSCRIPTEN__ */

static const PolyAllocator *host_get_allocator(void) {
  return &POLY_HOST_ALLOCATOR;
}

/* Interpreter backend */

static PolyRewriteOpts interp_rewrite_opts(PolyUOp *sink) {
  /* tinygrad@2026-08-22/a9069c177a9d runtime/ops_python.py:215-244 and
   * codegen/__init__.py:292-409: PythonRenderer capabilities drive the shared
   * full_rewrite_to_sink pipeline before PROGRAM construction. */
  return (PolyRewriteOpts){
      .optimize = poly_kernel_optimize_enabled(sink),
      .beam_width = poly_kernel_beam(sink),
      .caps =
          {
              .device = "PYTHON",
              .arch = poly_getenv_flag("IMAGE") ? "IMAGE_PITCH_ALIGNMENT=1" : "",
              .has_mulacc = true,
              .has_max = true,
              .has_threefry = false,
              .has_exp2 = true,
              .has_log2 = true,
              .has_sin = true,
              .has_fdiv = false,
              .supports_float16 = false,
              .supports_bfloat16 = true,
              .supports_fp8e4m3 = true,
              .supports_fp8e5m2 = true,
              .supports_fp8e4m3fnuz = true,
              .supports_fp8e5m2fnuz = true,
              .has_int64 = true,
              .has_local = false,
              .max_vec_width = 4,
          },
      .device = POLY_DEVICE_INTERP,
      .opt_policy = POLY_OPT_HEURISTIC,
  };
}

static PolyUOp *interp_rewrite_program(PolyCtx *ctx, PolyUOp *sink) {
  return poly_full_rewrite_to_sink_ex(ctx, sink, interp_rewrite_opts(sink));
}

static int interp_lower_item(PolyCtx *ctx, PolyUOp *program, const char *fn_name, PolyRunner *out) {
  (void)fn_name;
  int n_lin;
  PolyUOp **lin = poly_program_linear_uops(program, &n_lin);
  if (!lin) return -1;
  PolyUOp **copy = n_lin > 0 ? malloc((size_t)n_lin * sizeof(*copy)) : NULL;
  if (n_lin > 0 && !copy) return -1;
  if (n_lin > 0) memcpy(copy, lin, (size_t)n_lin * sizeof(*copy));
  lin = copy;

  InterpHandle *ih = malloc(sizeof(InterpHandle));
  if (!ih) {
    free(lin);
    return -1;
  }
  ih->ctx = ctx;
  ih->lin = lin;
  ih->n_lin = n_lin;

  out->kind = POLY_RUNNER_INTERP;
  /* INTERP executes LINEAR directly; its instruction graph is its library.
   * Do not introduce a bytecode dialect merely to obtain a search key. */
  if (out->capture_binary) out->compiled_binary = poly_program_linear(program);
  out->handle = ih;
  out->handle_size = (int)(sizeof(*ih) + (size_t)n_lin * sizeof(PolyUOp *));
  out->program = program;
  return 0;
}

static int interp_execute(PolyRunner *runner, void **args, int n_args) {
  InterpHandle *ih = (InterpHandle *)runner->handle;
  return poly_interp_eval(ih->ctx, ih->lin, ih->n_lin, args, n_args);
}

static void interp_free_runner(PolyRunner *runner) {
  if (runner->handle) {
    InterpHandle *ih = (InterpHandle *)runner->handle;
    free(ih->lin);
    free(ih);
  }
}

static const PolyAllocator *interp_get_allocator(void) {
  return &POLY_CPU_ALLOCATOR;
}

/* CUDA backend */

#ifdef POLY_HAS_CUDA

static void cuda_extract_dims(
    PolyCtx *ctx,
    PolyUOp **lin,
    int n_lin,
    int grid[3],
    int local[3],
    PolyUOp *grid_exprs[3],
    PolyUOp *block_exprs[3],
    int *launch_bounds
) {
  grid[0] = 1;
  grid[1] = 1;
  grid[2] = 1;
  local[0] = 1;
  local[1] = 1;
  local[2] = 1;
  grid_exprs[0] = grid_exprs[1] = grid_exprs[2] = NULL;
  block_exprs[0] = block_exprs[1] = block_exprs[2] = NULL;

  for (int j = 0; j < n_lin; j++) {
    if (lin[j]->op != POLY_OP_SPECIAL || lin[j]->n_src <= 0) continue;
    const char *sn = lin[j]->arg.str;
    if (!sn || !sn[0]) continue;
    int slen = (int)strlen(sn);
    int dim_idx = (slen > 0) ? sn[slen - 1] - '0' : 0;
    if (dim_idx < 0 || dim_idx > 2) dim_idx = 0;
    PolyUOp *bound_expr = lin[j]->src[0];
    int bound = poly_launch_dim_upper_bound(ctx, bound_expr);
    if (sn[0] == 'l') {
      local[dim_idx] = bound;
      block_exprs[dim_idx] = bound_expr;
    } else {
      grid[dim_idx] = bound;
      grid_exprs[dim_idx] = bound_expr;
    }
  }

  *launch_bounds = local[0] * local[1] * local[2];
  if (*launch_bounds <= 0) *launch_bounds = 1;
}

static char *cuda_render_source(PolyCtx *ctx, PolyUOp *program, const char *fn_name) {
  PolyUOp *scheduled_root = poly_program_kernel_body(program);
  if (!scheduled_root) return NULL;
  int n_lin;
  bool lin_owned = false;
  PolyUOp **lin = poly_program_linear_uops(program, &n_lin);
  if (!lin) {
    lin = poly_do_linearize(ctx, scheduled_root, &n_lin);
    lin_owned = true;
  }
  if (!lin) return NULL;

  int grid[3], local[3], launch_bounds = 1;
  PolyUOp *grid_exprs[3], *block_exprs[3];
  cuda_extract_dims(ctx, lin, n_lin, grid, local, grid_exprs, block_exprs, &launch_bounds);

  char *src = poly_render_cuda(ctx, lin, n_lin, fn_name, launch_bounds);
  if (lin_owned) free(lin);
  return src;
}

static int cuda_lower_item(PolyCtx *ctx, PolyUOp *program, const char *fn_name, PolyRunner *out) {
  PolyUOp *scheduled_root = poly_program_kernel_body(program);
  if (!scheduled_root) return -1;
  int n_lin;
  bool lin_owned = false;
  PolyUOp **lin = poly_program_linear_uops(program, &n_lin);
  if (!lin) {
    lin = poly_do_linearize(ctx, scheduled_root, &n_lin);
    lin_owned = true;
  }
  if (!lin) return -1;

  int grid[3], local[3], launch_bounds = 1;
  PolyUOp *grid_exprs[3], *block_exprs[3];
  cuda_extract_dims(ctx, lin, n_lin, grid, local, grid_exprs, block_exprs, &launch_bounds);

  const char *src = poly_program_source_text(program);
  char *src_owned = NULL;
  if (!src) {
    src_owned = poly_render_cuda(ctx, lin, n_lin, fn_name, launch_bounds);
    src = src_owned;
  }
  if (lin_owned) free(lin);
  if (!src) return -1;

  if (poly_dump_kernels_enabled())
    fprintf(stderr, "=== CUDA KERNEL %s ===\n%s\n=== END ===\n", fn_name, src);

  uint8_t *binary = NULL;
  int binary_size = 0;
  int status = -1;
  PolyCudaProgram *prog = poly_compile_cuda_with_binary(
      src, fn_name, out->capture_binary ? &binary : NULL, out->capture_binary ? &binary_size : NULL,
      &status
  );
  if (!prog) {
    fprintf(stderr, "=== FAILED CUDA KERNEL %s ===\n%s\n=== END ===\n", fn_name, src);
    free(src_owned);
    return status;
  }
  free(src_owned);

  if (out->capture_binary) {
    out->compiled_binary =
        poly_uop0(ctx, POLY_OP_BINARY, POLY_UINT8, poly_arg_bytes(binary, binary_size));
    free(binary);
    if (!out->compiled_binary) {
      poly_cuda_program_destroy(prog);
      return -1;
    }
  }
  CudaRunnerHandle *ch = malloc(sizeof(CudaRunnerHandle));
  if (!ch) {
    poly_cuda_program_destroy(prog);
    return -1;
  }
  ch->prog = prog;

  out->kind = POLY_RUNNER_COMPILED;
  out->handle = ch;
  out->handle_size = (int)sizeof(*ch);
  out->grid[0] = grid[0];
  out->grid[1] = grid[1];
  out->grid[2] = grid[2];
  out->block[0] = local[0];
  out->block[1] = local[1];
  out->block[2] = local[2];
  for (int i = 0; i < 3; i++) {
    out->grid_exprs[i] = grid_exprs[i];
    out->block_exprs[i] = block_exprs[i];
  }
  return 0;
}

static int cuda_execute(PolyRunner *runner, void **args, int n_args) {
  CudaRunnerHandle *ch = (CudaRunnerHandle *)runner->handle;

  /* cuLaunchKernel kernelParams: each element points TO the arg value.
   * Buffer params (0..n_params-1): args[i] is a device ptr; cast to CUdeviceptr.
   * Scalars arrive as int32 bindings. Like ops_cuda.encode_args, extend the
   * value into storage wide enough for the kernel's declared integer dtype;
   * cuLaunchKernel copies that dtype's width, not the host binding's width. */
  unsigned long long *dptrs = malloc((size_t)n_args * sizeof(unsigned long long));
  void **cuda_args = malloc((size_t)n_args * sizeof(void *));
  if (!dptrs || !cuda_args) {
    free(dptrs);
    free(cuda_args);
    return -1;
  }
  for (int i = 0; i < runner->n_params; i++) {
    dptrs[i] = (unsigned long long)(uintptr_t)args[i];
    cuda_args[i] = &dptrs[i];
  }
  for (int i = runner->n_params; i < n_args; i++) {
    dptrs[i] = (unsigned long long)(int64_t) * (int32_t *)args[i];
    cuda_args[i] = &dptrs[i];
  }

  int ret = runner->wait ? poly_cuda_launch_timed(
                               ch->prog, cuda_args, n_args, runner->grid[0], runner->grid[1],
                               runner->grid[2], runner->block[0], runner->block[1],
                               runner->block[2], &runner->elapsed_us
                           )
                         : poly_cuda_launch(
                               ch->prog, cuda_args, n_args, runner->grid[0], runner->grid[1],
                               runner->grid[2], runner->block[0], runner->block[1], runner->block[2]
                           );
  /* Waited launches already synchronize their end event. Ordinary launches
   * only enqueue; host readback remains a separate completion boundary. */

  free(dptrs);
  free(cuda_args);
  return ret;
}

static void cuda_free_runner(PolyRunner *runner) {
  if (runner->handle) {
    CudaRunnerHandle *ch = (CudaRunnerHandle *)runner->handle;
    poly_cuda_program_destroy(ch->prog);
    free(ch);
  }
}

static const PolyAllocator *cuda_get_allocator(void) {
  return &POLY_CUDA_ALLOCATOR;
}

#endif /* POLY_HAS_CUDA */

/* HIP backend */

#ifdef POLY_HAS_HIP

static char *hip_render_source(PolyCtx *ctx, PolyUOp *program, const char *fn_name) {
  PolyUOp *scheduled_root = poly_program_kernel_body(program);
  if (!scheduled_root) return NULL;
  int n_lin;
  bool lin_owned = false;
  PolyUOp **lin = poly_program_linear_uops(program, &n_lin);
  if (!lin) {
    lin = poly_do_linearize(ctx, scheduled_root, &n_lin);
    lin_owned = true;
  }
  if (!lin) return NULL;

  int grid_size = 0, local_size = 0;
  for (int j = 0; j < n_lin; j++) {
    if (lin[j]->op == POLY_OP_SPECIAL && lin[j]->n_src > 0 && lin[j]->src[0]->op == POLY_OP_CONST) {
      const char *sn = lin[j]->arg.str;
      if (sn && sn[0] == 'l')
        local_size = (int)lin[j]->src[0]->arg.i;
      else
        grid_size = (int)lin[j]->src[0]->arg.i;
    }
  }
  (void)grid_size;

  int block_size = local_size > 0 ? local_size : 256;
  char *src = poly_render_hip(ctx, lin, n_lin, fn_name, block_size, poly_hip_arch());
  if (lin_owned) free(lin);
  return src;
}

static int hip_lower_item(PolyCtx *ctx, PolyUOp *program, const char *fn_name, PolyRunner *out) {
  PolyUOp *scheduled_root = poly_program_kernel_body(program);
  if (!scheduled_root) return -1;
  int n_lin;
  bool lin_owned = false;
  PolyUOp **lin = poly_program_linear_uops(program, &n_lin);
  if (!lin) {
    lin = poly_do_linearize(ctx, scheduled_root, &n_lin);
    lin_owned = true;
  }
  if (!lin) return -1;

  /* Extract grid/block from SPECIAL ops */
  int grid_size = 0, local_size = 0;
  for (int j = 0; j < n_lin; j++) {
    if (lin[j]->op == POLY_OP_SPECIAL && lin[j]->n_src > 0 && lin[j]->src[0]->op == POLY_OP_CONST) {
      const char *sn = lin[j]->arg.str;
      if (sn && sn[0] == 'l')
        local_size = (int)lin[j]->src[0]->arg.i;
      else
        grid_size = (int)lin[j]->src[0]->arg.i;
    }
  }

  int block_size = local_size > 0 ? local_size : 256;

  const char *src = poly_program_source_text(program);
  char *src_owned = NULL;
  if (!src) {
    src_owned = poly_render_hip(ctx, lin, n_lin, fn_name, block_size, poly_hip_arch());
    src = src_owned;
  }
  if (lin_owned) free(lin);
  if (!src) return -1;

  if (poly_dump_kernels_enabled())
    fprintf(stderr, "=== HIP KERNEL %s ===\n%s\n=== END ===\n", fn_name, src);

  PolyHipProgram *prog = poly_compile_hip(src, fn_name);
  if (!prog) {
    fprintf(stderr, "=== FAILED HIP KERNEL %s ===\n%s\n=== END ===\n", fn_name, src);
    free(src_owned);
    return -1;
  }
  free(src_owned);

  /* Compute grid dimensions */
  int gx;
  if (local_size > 0 && grid_size > 0)
    gx = grid_size;
  else if (local_size > 0)
    gx = 1;
  else if (grid_size > 0)
    gx = (grid_size + block_size - 1) / block_size;
  else
    gx = 1;

  HipRunnerHandle *hh = malloc(sizeof(HipRunnerHandle));
  if (!hh) {
    poly_hip_program_destroy(prog);
    return -1;
  }
  hh->prog = prog;

  out->kind = POLY_RUNNER_COMPILED;
  out->handle = hh;
  out->handle_size = (int)sizeof(*hh);
  out->grid[0] = gx;
  out->grid[1] = 1;
  out->grid[2] = 1;
  out->block[0] = block_size;
  out->block[1] = 1;
  out->block[2] = 1;
  return 0;
}

static int hip_execute(PolyRunner *runner, void **args, int n_args) {
  HipRunnerHandle *hh = (HipRunnerHandle *)runner->handle;

  /* hipModuleLaunchKernel kernelParams: each element points TO the arg value.
   * Buffer params (0..n_params-1): args[i] is the device ptr, so &args[i] works.
   * Scalar params (n_params..): args[i] is already &int_val, use it directly. */
  void **hip_args = malloc((size_t)n_args * sizeof(void *));
  if (!hip_args) return -1;
  for (int i = 0; i < runner->n_params; i++)
    hip_args[i] = &args[i];
  for (int i = runner->n_params; i < n_args; i++)
    hip_args[i] = args[i];

  int ret = poly_hip_launch(
      hh->prog, hip_args, n_args, runner->grid[0], runner->grid[1], runner->grid[2],
      runner->block[0], runner->block[1], runner->block[2]
  );
  if (ret == 0) ret = poly_hip_sync();

  free(hip_args);
  return ret;
}

static void hip_free_runner(PolyRunner *runner) {
  if (runner->handle) {
    HipRunnerHandle *hh = (HipRunnerHandle *)runner->handle;
    poly_hip_program_destroy(hh->prog);
    free(hh);
  }
}

static const PolyAllocator *hip_get_allocator(void) {
  return &POLY_HIP_ALLOCATOR;
}

#endif /* POLY_HAS_HIP */

/* tinygrad-style x86 ISA backend */

#ifdef POLY_HAS_X86

static int x86_execute_fn(void *self, void **args, int n_args) {
  PolyRunner *runner = (PolyRunner *)self;
  const PolyProgramInfo *info = poly_program_info(NULL, runner->program);
  if (!info || n_args != runner->n_params + runner->n_vars) return -1;

  int abi_n_args = runner->n_params;
  int core_id_slot = -1;
  for (int i = 0; i < info->n_vars; i++) {
    PolyUOp *var = info->vars[i];
    int slot =
        var && var->arg.kind == POLY_ARG_PARAM && var->arg.param ? (int)var->arg.param->slot : -1;
    if (slot < 0) return -1;
    if (slot + 1 > abi_n_args) abi_n_args = slot + 1;
    if (poly_program_var_is_runtime(var)) core_id_slot = slot;
  }

  void *stack_args[64] = {0};
  void **abi_args = abi_n_args <= (int)(sizeof(stack_args) / sizeof(stack_args[0]))
                        ? stack_args
                        : calloc((size_t)abi_n_args, sizeof(*abi_args));
  if (!abi_args) return -1;
  for (int i = 0; i < runner->n_params; i++)
    abi_args[i] = args[i];
  for (int v = 0; v < runner->n_vars; v++) {
    int info_idx = runner->var_indices ? runner->var_indices[v] : -1;
    if (info_idx < 0 || info_idx >= info->n_vars || !args[runner->n_params + v]) {
      if (abi_args != stack_args) free(abi_args);
      return -1;
    }
    PolyUOp *var = info->vars[info_idx];
    int slot = (int)var->arg.param->slot;
    abi_args[slot] = (void *)(intptr_t)(*(int *)args[runner->n_params + v]);
  }

  int threads = runner->grid[0] > 1 ? runner->grid[0] : 1;
  int ret = core_id_slot >= 0
                ? poly_x86_program_call_threaded(
                      (PolyX86Program *)runner->handle, abi_args, abi_n_args, core_id_slot, threads
                  )
                : poly_x86_program_call((PolyX86Program *)runner->handle, abi_args, abi_n_args);
  if (abi_args != stack_args) free(abi_args);
  return ret;
}

static void x86_free_fn(void *self) {
  PolyRunner *runner = (PolyRunner *)self;
  if (runner->handle) poly_x86_program_destroy((PolyX86Program *)runner->handle);
}

static char *poly_hex_from_bytes(const uint8_t *bytes, int n_bytes) {
  if (!bytes || n_bytes <= 0) return NULL;
  char *hex = malloc((size_t)n_bytes * 2 + 1);
  if (!hex) return NULL;
  static const char digits[] = "0123456789abcdef";
  for (int i = 0; i < n_bytes; i++) {
    hex[2 * i] = digits[bytes[i] >> 4];
    hex[2 * i + 1] = digits[bytes[i] & 0x0f];
  }
  hex[(size_t)n_bytes * 2] = '\0';
  return hex;
}

static PolyUOp *poly_prepare_x86_program_for_backend(
    PolyCtx *ctx,
    PolyUOp *call,
    PolyUOp *base,
    PolyUOp *device_uop,
    PolyDevice device,
    uint32_t env_stamp,
    bool cache
) {
  if (!ctx || !call || call->op != POLY_OP_CALL) return NULL;
  PolyUOp *raw = poly_call_raw_body(call);
  if (!raw) return NULL;

  PolyToProgramCacheEntry key = {
      .program = raw,
      .device_uop = device_uop,
      .device = device,
      .env_stamp = env_stamp,
      .prepared_program = NULL,
  };
  uint32_t hash = poly_program_device_cache_hash(raw, device_uop, device, env_stamp);
  PolyToProgramCacheEntry *entry =
      (cache && poly_engine_cache_enabled() && ctx->to_program_cache)
          ? poly_map_get(ctx->to_program_cache, hash, &key, poly_to_program_cache_eq)
          : NULL;
  if (entry) return entry->prepared_program;

  PolyUOp *body = poly_program_body(base);
  if (!body) return NULL;
  if (raw->op != POLY_OP_PROGRAM) {
    body = poly_rewrite_x86(ctx, body);
    if (!body) return NULL;
    base = poly_program_from_call_body(ctx, call, body, poly_program_arg_name(raw), device);
  }
  if (!base) return NULL;

  /* Existing LINEAR already owns instruction selection and register allocation. */
  PolyUOp *linear = poly_program_linear(base);
  if (!linear) {
    int n_lin = 0;
    PolyUOp **lin = poly_linearize_x86_rewritten(ctx, body, &n_lin);
    if (!lin) return NULL;
    linear = poly_uop(ctx, POLY_OP_LINEAR, POLY_VOID, lin, n_lin, poly_arg_none());
    free(lin);
    if (!linear) return NULL;
  }
  PolyUOp *prepared = poly_program_with_linear(ctx, base, linear);
  if (!prepared) return NULL;

  int n_code = 0;
  uint8_t *code = poly_render_x86(linear->src, linear->n_src, &n_code);
  if (!code || n_code <= 0) {
    free(code);
    return NULL;
  }

  char *source = poly_hex_from_bytes(code, n_code);
  if (!source) {
    free(code);
    return NULL;
  }
  g_program_source_render_count++;
  prepared = poly_program_attach_source(ctx, prepared, source);
  free(source);
  if (!prepared) {
    free(code);
    return NULL;
  }
  prepared = poly_program_attach_binary(ctx, prepared, code, n_code);
  free(code);
  if (!prepared) return NULL;

  if (cache && poly_engine_cache_enabled() && ctx->to_program_cache) {
    entry = malloc(sizeof(*entry));
    if (entry) {
      entry->program = raw;
      entry->device_uop = device_uop;
      entry->device = device;
      entry->env_stamp = env_stamp;
      entry->prepared_program = prepared;
      if (!to_program_cache_store(ctx, hash, entry)) free(entry);
    }
  }

  return prepared;
}

static int x86_lower_item(PolyCtx *ctx, PolyUOp *program, const char *fn_name, PolyRunner *out) {
  (void)ctx;
  (void)fn_name;
  PolyX86Program *prog = NULL;
  int code_size = 0;
  PolyUOp *binary = poly_program_binary(program);
  if (binary && binary->arg.kind == POLY_ARG_BYTES && binary->arg.bytes.data &&
      binary->arg.bytes.n > 0) {
    code_size = binary->arg.bytes.n;
    prog = poly_compile_x86(binary->arg.bytes.data, binary->arg.bytes.n);
  } else {
    const char *source = poly_program_source_text(program);
    if (source) {
      int status = -1;
      prog = poly_compile_x86_source(source, &code_size, &status);
      if (!prog) return status;
    }
  }
  if (!prog) return -1;

  out->kind = POLY_RUNNER_COMPILED;
  out->handle = prog;
  out->handle_size = code_size;
  const PolyProgramInfo *program_info = poly_program_info(ctx, program);
  int threads = program_info && program_info->global_size[0] > 1 ? program_info->global_size[0] : 1;
  out->grid[0] = threads;
  out->grid[1] = 1;
  out->grid[2] = 1;
  out->block[0] = 1;
  out->block[1] = 1;
  out->block[2] = 1;
  out->execute = x86_execute_fn;
  out->free_handle = x86_free_fn;
  return 0;
}

static int x86_execute(PolyRunner *runner, void **args, int n_args) {
  return runner->execute(runner, args, n_args);
}

static void x86_free_runner(PolyRunner *runner) {
  if (runner->free_handle) runner->free_handle(runner);
}

#endif /* POLY_HAS_X86 */

static int backend_noop_ensure_open(void) {
  return 0;
}

#ifdef POLY_HAS_CUDA
static int cuda_ensure_open(void) {
  return poly_cuda_init();
}
#endif

/* ══════════════════════════════════════════════════════════════════════ */
/*  Backend registry                                                     */
/* ══════════════════════════════════════════════════════════════════════ */

static const PolyBackendDesc BACKENDS[] = {
    [POLY_DEVICE_AUTO] = {NULL, POLY_DEVICE_AUTO, false, NULL, NULL, NULL, NULL, NULL, NULL, NULL},
    [POLY_DEVICE_HOST] =
        {"host", POLY_DEVICE_HOST, false, interp_rewrite_program, NULL, interp_lower_item,
         interp_execute, interp_free_runner, backend_noop_ensure_open, host_get_allocator},
#ifndef __EMSCRIPTEN__
    [POLY_DEVICE_CPU] =
        {"cpu", POLY_DEVICE_CPU, false, cpu_rewrite_program, cpu_render_source, cpu_lower_item,
         cpu_execute, cpu_free_runner, backend_noop_ensure_open, cpu_get_allocator},
#else
    [POLY_DEVICE_CPU] = {NULL, POLY_DEVICE_CPU, false, NULL, NULL, NULL, NULL, NULL, NULL, NULL},
#endif
    [POLY_DEVICE_INTERP] =
        {"interp", POLY_DEVICE_INTERP, false, interp_rewrite_program, NULL, interp_lower_item,
         interp_execute, interp_free_runner, backend_noop_ensure_open, interp_get_allocator},
#ifdef POLY_HAS_CUDA
    [POLY_DEVICE_CUDA] =
        {"cuda", POLY_DEVICE_CUDA, false, poly_rewrite_cuda, cuda_render_source, cuda_lower_item,
         cuda_execute, cuda_free_runner, cuda_ensure_open, cuda_get_allocator},
#else
    [POLY_DEVICE_CUDA] = {NULL, POLY_DEVICE_CUDA, false, NULL, NULL, NULL, NULL, NULL, NULL, NULL},
#endif
#ifdef __EMSCRIPTEN__
    [POLY_DEVICE_WASM] =
        {"wasm", POLY_DEVICE_WASM, true, poly_rewrite_wasm, NULL, poly_wasm_lower_item,
         poly_wasm_execute, poly_wasm_free_runner, backend_noop_ensure_open,
         poly_wasm_get_allocator},
#else
    [POLY_DEVICE_WASM] = {NULL, POLY_DEVICE_WASM, false, NULL, NULL, NULL, NULL, NULL, NULL, NULL},
#endif
#ifdef __EMSCRIPTEN__
    [POLY_DEVICE_WEBGPU] =
        {"webgpu", POLY_DEVICE_WEBGPU, true, poly_rewrite_webgpu, poly_webgpu_render_source,
         poly_webgpu_lower_item, poly_webgpu_execute, poly_webgpu_free_runner,
         backend_noop_ensure_open, poly_webgpu_get_allocator},
#else
    [POLY_DEVICE_WEBGPU] =
        {NULL, POLY_DEVICE_WEBGPU, false, NULL, NULL, NULL, NULL, NULL, NULL, NULL},
#endif
#ifdef POLY_HAS_X86
    [POLY_DEVICE_X86] =
        {"x86", POLY_DEVICE_X86, false, NULL, NULL, x86_lower_item, x86_execute, x86_free_runner,
         backend_noop_ensure_open, cpu_get_allocator},
#else
    [POLY_DEVICE_X86] = {NULL, POLY_DEVICE_X86, false, NULL, NULL, NULL, NULL, NULL, NULL, NULL},
#endif
#ifdef POLY_HAS_HIP
    [POLY_DEVICE_HIP] =
        {"hip", POLY_DEVICE_HIP, false, poly_rewrite_hip, hip_render_source, hip_lower_item,
         hip_execute, hip_free_runner, poly_hip_init, hip_get_allocator},
#else
    [POLY_DEVICE_HIP] = {NULL, POLY_DEVICE_HIP, false, NULL, NULL, NULL, NULL, NULL, NULL, NULL},
#endif
    [POLY_DEVICE_DISK] =
        {"disk", POLY_DEVICE_DISK, false, NULL, NULL, NULL, NULL, NULL, backend_noop_ensure_open,
         poly_disk_get_allocator},
};

#define N_BACKENDS (sizeof(BACKENDS) / sizeof(BACKENDS[0]))

const PolyBackendDesc *poly_backend_get(PolyDevice device) {
  if (device < 0 || (size_t)device >= N_BACKENDS) return NULL;
  if (!BACKENDS[device].name) return NULL;
  return &BACKENDS[device];
}

int poly_backend_ensure_open(PolyDevice device) {
  const PolyBackendDesc *be = poly_backend_get(device);
  if (!be) return -1;
  return be->ensure_open ? be->ensure_open() : 0;
}

bool poly_device_is_host_addressable(PolyDevice device) {
  const PolyBackendDesc *be = poly_backend_get(device);
  return be && be->get_allocator()->host_addressable;
}

static uint64_t poly_counter_add_sat(uint64_t a, uint64_t b) {
  return UINT64_MAX - a < b ? UINT64_MAX : a + b;
}
static PolyUOp *poly_call_device_num_var(PolyCtx *ctx, PolyUOp *body);

/* C runtime-cache key extension for switches that change rendered PROGRAMs. */
static uint32_t poly_runtime_cache_env_stamp(void) {
  uint32_t stamp = 2166136261u;
#ifdef __EMSCRIPTEN__
  const uint8_t wasm_features = 1;
#else
  const uint8_t wasm_features = 0;
#endif
#ifdef POLY_HAS_X86
  uint32_t x86_features = poly_x86_feature_stamp();
#else
  uint32_t x86_features = 0;
#endif
  /* to_program_config keys code-generating policy, not just the input AST.
   * Keep full integer values: TC_OPT=0 and256 must not alias by truncation.
   * THREADS/SSA and compiled target features are C renderer configuration. */
  uint32_t values[] = {
      (uint32_t)poly_get_noopt(),
      (uint32_t)poly_get_default_float(),
      (uint32_t)poly_get_default_int(),
      (uint32_t)poly_getenv_int("TC_OPT", 0),
      (uint32_t)poly_getenv_int("TC", 1),
      (uint32_t)poly_getenv_int("TC_SELECT", -1),
      (uint32_t)poly_getenv_flag("NOLOCALS"),
      (uint32_t)poly_getenv_flag("IMAGE"),
      (uint32_t)poly_getenv_flag("ALLOW_TF32"),
      (uint32_t)poly_getenv_int("NUM_CPU_THREADS", 0),
      (uint32_t)poly_getenv_flag_default("THREADS", true),
      (uint32_t)(poly_getenv_flag("EXPAND_SSA") || poly_getenv_flag("POLY_EXPAND_SSA")),
      wasm_features,
      x86_features,
  };
  for (size_t i = 0; i < sizeof(values) / sizeof(values[0]); i++)
    for (int shift = 0; shift < 32; shift += 8) {
      stamp ^= (values[i] >> shift) & 0xFF;
      stamp *= 16777619u;
    }
  return stamp;
}

/* Current Tinygrad resolves `_device_num` per MULTI execution lane. */
static PolyUOp *poly_call_device_num_var(PolyCtx *ctx, PolyUOp *body) {
  if (!ctx || !body) return NULL;
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, body, &n_topo);
  if (!topo) return NULL;
  PolyUOp *found = NULL;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (poly_uop_is_alu_param(u) && poly_uop_expr(u) &&
        strcmp(poly_uop_expr(u), "_device_num") == 0) {
      found = u;
      break;
    }
  }
  poly_toposort_free(topo);
  return found;
}

/* Current Tinygrad engine/realize.py:pm_beam. */
static PolyUOp *poly_apply_call_beam(PolyCtx *ctx, PolyUOp *call, int beam) {
  if (!ctx || !call || call->op != POLY_OP_CALL || call->n_src < 1 || beam < 1) return call;
  PolyUOp *sink = call->src[0];
  if (!sink || sink->op != POLY_OP_SINK || sink->arg.kind != POLY_ARG_KERNEL_INFO ||
      !sink->arg.kernel_info || sink->arg.kernel_info->beam != 0)
    return call;
  PolyKernelInfo info = *sink->arg.kernel_info;
  info.beam = beam;
  PolyUOp *new_sink =
      sink->tag || sink->tag_arg.kind != POLY_ARG_NONE
          ? poly_uop_tagged_arg(
                ctx, POLY_OP_SINK, sink->dtype, sink->src, sink->n_src, poly_arg_kernel_info(&info),
                sink->tag, sink->tag_arg
            )
          : poly_uop(
                ctx, POLY_OP_SINK, sink->dtype, sink->src, sink->n_src, poly_arg_kernel_info(&info)
            );
  if (!new_sink) return NULL;
  PolyUOp **src = malloc((size_t)call->n_src * sizeof(*src));
  if (!src) return NULL;
  memcpy(src, call->src, (size_t)call->n_src * sizeof(*src));
  src[0] = new_sink;
  PolyUOp *ret = linear_rebuild_preserving_metadata(ctx, call, src);
  free(src);
  return ret;
}

/* engine/realize.py:optimize_local_size. Candidate order is shuffled without
 * consuming the Tensor RNG. Only the timing minimum and lexicographic tie
 * break affect selection; each member of the candidate product is tried twice. */
static bool optimize_local_size_candidates(
    const int global[3],
    double (*time_candidate)(void *, const int *),
    void *opaque,
    int best[3]
) {
  int dims[3][11], counts[3] = {0};
  const int sizes[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 1024};
  for (int d = 0; d < 3; d++) {
    if (global[d] <= 0) return false;
    dims[d][counts[d]++] = global[d];
    for (size_t i = 0; i < sizeof(sizes) / sizeof(*sizes); i++)
      if (sizes[i] < global[d]) dims[d][counts[d]++] = sizes[i];
  }
  int(*candidates)[3] = malloc(2 * 11 * 11 * 11 * sizeof(*candidates));
  if (!candidates) return false;
  int n = 0;
  for (int x = 0; x < counts[0]; x++)
    for (int y = 0; y < counts[1]; y++)
      for (int z = 0; z < counts[2]; z++) {
        int a = dims[0][x], b = dims[1][y], c = dims[2][z];
        if (a > 1024 || b > 1024 / a || c > 1024 / a / b) continue;
        for (int twice = 0; twice < 2; twice++) {
          candidates[n][0] = a;
          candidates[n][1] = b;
          candidates[n++][2] = c;
        }
      }
  uint32_t random = poly_ptr_hash(opaque) ^ (uint32_t)fmod(poly_now_ms() * 1000, UINT32_MAX);
  if (!random) random = 1;
  for (int i = n - 1; i > 0; i--) {
    random ^= random << 13;
    random ^= random >> 17;
    random ^= random << 5;
    int j = (int)(random % (uint32_t)(i + 1)), tmp[3];
    memcpy(tmp, candidates[i], sizeof(tmp));
    memcpy(candidates[i], candidates[j], sizeof(tmp));
    memcpy(candidates[j], tmp, sizeof(tmp));
  }
  double best_time = INFINITY;
  for (int i = 0; i < n; i++) {
    int *local = candidates[i];
    /* Tinygrad lets runtimes reject fractional grid quotients. Our launch ABI
     * is integer-only: reject, never silently floor or round the grid. */
    if (global[0] % local[0] || global[1] % local[1] || global[2] % local[2]) continue;
    double elapsed = time_candidate(opaque, local);
    if (!isfinite(elapsed)) continue;
    bool earlier = false;
    if (elapsed == best_time)
      for (int d = 0; d < 3; d++) {
        if (local[d] == best[d]) continue;
        earlier = local[d] < best[d];
        break;
      }
    if (elapsed < best_time || earlier) {
      best_time = elapsed;
      memcpy(best, local, 3 * sizeof(*best));
    }
  }
  free(candidates);
  return isfinite(best_time);
}

typedef struct {
  PolyCtx *ctx;
  PolyRunner *runner;
  PolyDevice device;
  const PolyTimingBuffers *buffers;
  int global[3];
} LocalSizeTiming;

static double local_size_time_candidate(void *opaque, const int *local) {
  LocalSizeTiming *timing = opaque;
  for (int d = 0; d < 3; d++) {
    timing->runner->grid[d] = timing->global[d] / local[d];
    timing->runner->block[d] = local[d];
    timing->runner->grid_exprs[d] = timing->runner->block_exprs[d] = NULL;
  }
  return poly_time_program(
      timing->ctx, timing->runner, timing->device, timing->buffers, 1, INFINITY, 0
  );
}

static PolyUOp *poly_optimize_local_size(
    PolyCtx *ctx,
    PolyUOp *call,
    PolyUOp *program,
    PolyUOp *device_uop,
    PolyDevice device
) {
  const PolyProgramInfo *info = poly_program_info(ctx, program);
  /* Renderer.has_local: these are the three local-workgroup backends. WGSL's
   * valid fixed-workgroup PROGRAMs already carry local_size and skip this pass. */
  if (!info) return NULL;
  if (info->has_local_size ||
      (device != POLY_DEVICE_CUDA && device != POLY_DEVICE_HIP && device != POLY_DEVICE_WEBGPU))
    return program;
  int global[3];
  for (int d = 0; d < 3; d++) {
    int64_t size = info->global_size[d];
    if (info->global_exprs[d]) {
      PolyUOp *value = poly_graph_rewrite(ctx, info->global_exprs[d], poly_symbolic());
      if (!value) return NULL;
      if (value->op == POLY_OP_CAST && value->n_src == 1 && poly_dtype_is_int(value->dtype))
        value = value->src[0];
      /* all_int admission: a symbolic upper bound is not a fixed launch. */
      if (value->op != POLY_OP_CONST || !poly_arg_integer_to_i64(value->arg, &size)) return program;
    }
    if (size <= 0 || size > INT_MAX) return NULL;
    global[d] = (int)size;
  }
  uint32_t hash = poly_ptr_hash(program);
  PolyUOp *cached = poly_map_get(ctx->local_size_cache, hash, program, poly_ptr_eq);
  if (cached) return cached;
  PolyRunner runner = {0};
  PolyRuntimeCacheEntry *runtime = NULL;
  PolyTimingBuffers buffers = {0};
  PolyUOp *selected = NULL;
  if (poly_lower_compute_call_cached(
          ctx, call, device_uop, device, poly_runtime_cache_env_stamp(), &runner, false, &runtime
      ) != 0)
    goto cleanup;
  runner.n_params = info->n_globals;
  if (poly_bind_runner_vars(ctx, &runner) != 0 ||
      !poly_timing_buffers_init(ctx, program->src[0], device, &buffers))
    goto cleanup;
  LocalSizeTiming timing = {.ctx = ctx, .runner = &runner, .device = device, .buffers = &buffers};
  memcpy(timing.global, global, sizeof(global));
  int local[3];
  if (!optimize_local_size_candidates(global, local_size_time_candidate, &timing, local)) {
    fprintf(stderr, "polygrad: all optimize_local_size executions failed\n");
    goto cleanup;
  }
  PolyProgramInfo chosen = *info;
  chosen.has_local_size = true;
  for (int d = 0; d < 3; d++) {
    chosen.global_size[d] = global[d] / local[d];
    chosen.local_size[d] = local[d];
    chosen.global_exprs[d] = chosen.local_exprs[d] = NULL;
  }
  selected = poly_uop_tagged_arg(
      ctx, program->op, program->dtype, program->src, program->n_src,
      poly_arg_program_info(&chosen), program->tag, program->tag_arg
  );
  /* Own only PROGRAM identities, not CALL arguments or user storage. Publish
   * after selection succeeds; failed timing must not cache a fallback. */
  if (!selected || poly_uop_retain(ctx, program) != 0) {
    selected = NULL;
    goto cleanup;
  }
  if (poly_uop_retain(ctx, selected) != 0) {
    poly_uop_release(ctx, program);
    selected = NULL;
    goto cleanup;
  }
  poly_map_set(ctx->local_size_cache, hash, program, selected, poly_ptr_eq);
  if (poly_map_get(ctx->local_size_cache, hash, program, poly_ptr_eq) != selected) {
    poly_uop_release(ctx, selected);
    poly_uop_release(ctx, program);
    selected = NULL;
  }
cleanup:
  poly_timing_buffers_free(&buffers);
  poly_runner_cleanup(&runner, device);
  poly_runtime_cache_entry_release(runtime);
  return selected;
}

#ifdef POLY_TESTING
bool poly_test_optimize_local_size(
    const int global[3],
    double (*time)(void *, const int *),
    void *opaque,
    int best[3]
) {
  return optimize_local_size_candidates(global, time, opaque, best);
}
int poly_test_runtime_cache_policy(PolyCtx *ctx, PolyUOp *call, PolyDevice device, bool cache) {
  PolyRunner runner = {0};
  PolyRuntimeCacheEntry *entry = NULL;
  int rc = poly_lower_compute_call_cached(
      ctx, call, poly_device_uop(ctx, device), device, poly_runtime_cache_env_stamp(), &runner,
      cache, &entry
  );
  poly_runner_cleanup(&runner, device);
  poly_runtime_cache_entry_release(entry);
  return rc;
}
#endif

/* Current Tinygrad engine/realize.py:compile_linear. beam=-1 selects BEAM. */
PolyUOp *poly_compile_linear(PolyCtx *ctx, PolyUOp *linear, int beam) {
  if (!ctx || !linear || linear->op != POLY_OP_LINEAR) return NULL;
  int beam_value = beam < 0 ? poly_get_beam() : beam;
  PolyUOp **calls = linear->n_src > 0 ? calloc((size_t)linear->n_src, sizeof(*calls)) : NULL;
  if (linear->n_src > 0 && !calls) return NULL;
  uint32_t env_stamp = poly_runtime_cache_env_stamp();
  bool ok = true;
  for (int i = 0; i < linear->n_src; i++) {
    PolyUOp *call = poly_apply_call_beam(ctx, linear->src[i], beam_value);
    if (!call || call->op != POLY_OP_CALL || call->n_src < 1) {
      ok = false;
      break;
    }
    PolyUOp *raw = call->src[0];
    if (raw->op != POLY_OP_SINK && raw->op != POLY_OP_PROGRAM) {
      calls[i] = call;
      continue;
    }
    PolyUOp *device_uop =
        call->n_src > 1 ? poly_uop_device_uop_cached(ctx, call->src[1], NULL) : NULL;
    if (device_uop && device_uop->arg.kind == POLY_ARG_STRING_TUPLE &&
        device_uop->arg.string_tuple.n > 0)
      device_uop = poly_device_uop_from_name(ctx, device_uop->arg.string_tuple.vals[0]);
    PolyDevice item_device = poly_device_from_device_uop(device_uop);
    if (!device_uop || item_device == POLY_DEVICE_AUTO) {
      ok = false;
      break;
    }
    PolyUOp *program =
        poly_prepare_program_for_backend(ctx, call, device_uop, item_device, env_stamp, true);
    if (!program) {
      ok = false;
      break;
    }
    PolyUOp **src = malloc((size_t)call->n_src * sizeof(*src));
    if (!src) {
      ok = false;
      break;
    }
    memcpy(src, call->src, (size_t)call->n_src * sizeof(*src));
    src[0] = program;
    calls[i] = linear_rebuild_preserving_metadata(ctx, call, src);
    free(src);
    if (calls[i]) {
      PolyUOp *selected = poly_optimize_local_size(ctx, calls[i], program, device_uop, item_device);
      if (!selected)
        calls[i] = NULL;
      else if (selected != program) {
        src = malloc((size_t)call->n_src * sizeof(*src));
        if (!src)
          calls[i] = NULL;
        else {
          memcpy(src, calls[i]->src, (size_t)call->n_src * sizeof(*src));
          src[0] = selected;
          calls[i] = linear_rebuild_preserving_metadata(ctx, calls[i], src);
          free(src);
        }
      }
    }
    if (!calls[i]) {
      ok = false;
      break;
    }
  }
  PolyUOp *compiled = ok ? linear_rebuild_preserving_metadata(ctx, linear, calls) : NULL;
  free(calls);
  return compiled;
}

typedef struct {
  PolyUOp *uop;
  PolyBuffer *container;
  PolyBuffer **items;
  int n_items;
} PolyResolvedLinearArg;

/* Current Tinygrad engine/realize.py:_resolve. Only PARAM and the aggregate
 * forms that can directly wrap PARAM are invocation-dependent. */
static PolyUOp *poly_resolve_linear_param(
    PolyCtx *ctx,
    PolyUOp *uop,
    PolyUOp **inputs,
    int n_inputs
) {
  if (!ctx || !uop || n_inputs < 0 || (n_inputs > 0 && !inputs)) return NULL;
  if (uop->op == POLY_OP_PARAM) {
    if (uop->arg.kind != POLY_ARG_PARAM || !uop->arg.param || uop->arg.param->slot < 0 ||
        uop->arg.param->slot >= n_inputs)
      return NULL;
    return inputs[uop->arg.param->slot];
  }
  if (uop->op == POLY_OP_MSTACK) {
    PolyUOp **src = uop->n_src > 0 ? malloc((size_t)uop->n_src * sizeof(*src)) : NULL;
    if (uop->n_src > 0 && !src) return NULL;
    bool ok = true;
    for (int i = 0; i < uop->n_src; i++) {
      src[i] = poly_resolve_linear_param(ctx, uop->src[i], inputs, n_inputs);
      if (!src[i]) {
        ok = false;
        break;
      }
    }
    PolyUOp *resolved = ok ? linear_rebuild_preserving_metadata(ctx, uop, src) : NULL;
    free(src);
    return resolved;
  }
  if ((uop->op == POLY_OP_MSELECT || uop->op == POLY_OP_SHRINK) && uop->n_src > 0 && uop->src[0] &&
      uop->src[0]->op == POLY_OP_PARAM) {
    PolyUOp **src = malloc((size_t)uop->n_src * sizeof(*src));
    if (!src) return NULL;
    memcpy(src, uop->src, (size_t)uop->n_src * sizeof(*src));
    src[0] = poly_resolve_linear_param(ctx, uop->src[0], inputs, n_inputs);
    PolyUOp *resolved = src[0] ? linear_rebuild_preserving_metadata(ctx, uop, src) : NULL;
    free(src);
    return resolved;
  }
  return uop;
}

#ifdef POLY_TESTING
PolyUOp *poly_test_resolve_linear_param(
    PolyCtx *ctx,
    PolyUOp *uop,
    PolyUOp **inputs,
    int n_inputs
) {
  return poly_resolve_linear_param(ctx, uop, inputs, n_inputs);
}
#endif

static int poly_ensure_linear_arg_buffer(PolyCtx *ctx, PolyUOp *uop, bool read) {
  if (!ctx || !uop) return -1;
  if (uop->op == POLY_OP_MSTACK) {
    for (int i = 0; i < uop->n_src; i++)
      if (poly_ensure_linear_arg_buffer(ctx, uop->src[i], read) != 0) return -1;
    return poly_uop_buffer_handle(ctx, uop) ? 0 : -1;
  }
  if (uop->op == POLY_OP_MSELECT && uop->n_src == 1) {
    if (poly_ensure_linear_arg_buffer(ctx, uop->src[0], read) != 0) return -1;
    return poly_uop_buffer_handle(ctx, uop) ? 0 : -1;
  }

  PolyBuffer *buffer = poly_uop_buffer_handle(ctx, uop);
  /* Current Tinygrad UOp.buffer + exec_kernel: tuple BUFFERs resolve to one
   * MultiBuffer whose children are allocated and checked per launch lane. */
  if (buffer && poly_buffer_is_multi(buffer)) {
    if (buffer->n_bufs <= 0 || !buffer->bufs) return -1;
    bool valid = true;
    for (int i = 0; i < buffer->n_bufs; i++) {
      PolyBuffer *child = buffer->bufs[i];
      if (!child || poly_buffer_is_multi(child) ||
          poly_buffer_handle_ensure_allocated(ctx, child) != 0)
        return -1;
      valid = valid && child->valid;
    }
    buffer->valid = valid;
    /* Tinygrad Buffer.ensure_allocated permits reads from Tensor.empty; its
     * initialization state is allocation, not defined bytes (device.py:137-143). */
    return 0;
  }

  const PolyUOp *identity = poly_uop_get_buffer_identity(uop);
  PolyUOp *device_uop = poly_uop_device_uop_cached(ctx, uop, NULL);
  if (!device_uop && identity)
    device_uop = poly_uop_device_uop_cached(ctx, (PolyUOp *)identity, NULL);
  PolyDevice device = poly_device_from_device_uop(device_uop);
  /* Tinygrad 2026-08-22 a9069c17 engine/realize.py:180-187 resolves each
   * physical CALL argument to its device-specific Buffer, then calls
   * ensure_allocated. C stores residency separately, so restore that same
   * immutable-UOp device binding before exposing the runtime handle. */
  if (identity && device_uop && device_uop->arg.kind != POLY_ARG_STRING_TUPLE &&
      device != POLY_DEVICE_AUTO) {
    if (poly_buffer_ensure_allocated(ctx, (PolyUOp *)identity, device) != 0) return -1;
    buffer = poly_uop_buffer_handle(ctx, uop);
  }
  if (buffer && !poly_buffer_is_multi(buffer)) {
    int rc = poly_buffer_handle_ensure_allocated(ctx, buffer);
    if (poly_debug_at_least(7) && (rc != 0 || (read && !buffer->valid)))
      fprintf(
          stderr,
          "[polygrad:run_linear] buffer op=%s ensure=%d read=%d valid=%d base=%p ptr=%p "
          "bytes=%zu offset=%zu\n",
          poly_op_name(uop->op), rc, read, buffer->valid, (void *)buffer->base, buffer->ptr,
          buffer->nbytes, buffer->offset
      );
    if (poly_debug_at_least(7) && buffer->base && rc != 0)
      fprintf(
          stderr, "[polygrad:run_linear] base device=%d valid=%d ptr=%p bytes=%zu src=%p\n",
          buffer->base->device, buffer->base->valid, buffer->base->ptr, buffer->base->nbytes,
          (void *)buffer->base->src
      );
    if (poly_debug_at_least(7) && (rc != 0 || (read && !buffer->valid))) {
      char *graph = poly_graph_str(uop);
      fprintf(
          stderr, "[polygrad:run_linear] buffer graph=%s\n", graph ? graph : "<allocation failure>"
      );
      free(graph);
    }
    if (rc != 0) return -1;
    return 0;
  }
  if (poly_debug_at_least(7)) {
    char *graph = poly_graph_str(uop);
    fprintf(
        stderr, "[polygrad:run_linear] unresolved buffer op=%s graph=%s\n", poly_op_name(uop->op),
        graph ? graph : "<allocation failure>"
    );
    free(graph);
  }
  if (!identity || !device_uop) return -1;
  if (device == POLY_DEVICE_AUTO || device_uop->arg.kind == POLY_ARG_STRING_TUPLE) return -1;

  int rc = read ? poly_buffer_ensure_device_current(ctx, (PolyUOp *)identity, device)
                : poly_buffer_ensure_device_allocated(ctx, (PolyUOp *)identity, device);
  if (rc != 0) return -1;
  buffer = poly_uop_buffer_handle(ctx, uop);
  return buffer && (!read || buffer->valid) ? 0 : -1;
}

static int poly_resolve_linear_arg(
    PolyCtx *ctx,
    PolyUOp *uop,
    bool read,
    PolyResolvedLinearArg *out
) {
  if (!ctx || !uop || !out || poly_ensure_linear_arg_buffer(ctx, uop, read) != 0) return -1;
  memset(out, 0, sizeof(*out));
  out->uop = uop;
  out->container = poly_uop_buffer_handle(ctx, uop);
  if (!out->container) return -1;
  if (poly_buffer_is_multi(out->container)) {
    out->n_items = out->container->n_bufs;
    out->items = out->n_items > 0 ? malloc((size_t)out->n_items * sizeof(*out->items)) : NULL;
    if (out->n_items <= 0 || !out->items) return -1;
    memcpy(out->items, out->container->bufs, (size_t)out->n_items * sizeof(*out->items));
  } else {
    out->items = malloc(sizeof(*out->items));
    if (!out->items) return -1;
    out->items[0] = out->container;
    out->n_items = 1;
  }
  return 0;
}

static void poly_resolved_linear_args_free(PolyResolvedLinearArg *args, int n_args) {
  if (!args) return;
  for (int i = 0; i < n_args; i++)
    free(args[i].items);
  free(args);
}

static int poly_linear_lane_count(
    const PolyProgramInfo *info,
    PolyResolvedLinearArg *args,
    int n_args
) {
  if (!info || (n_args > 0 && !args)) return -1;
  int lanes = 1;
  for (int i = 0; i < info->n_globals; i++) {
    int arg = info->globals[i];
    if (arg < 0 || arg >= n_args || args[arg].n_items <= 0) return -1;
    if (args[arg].n_items > 1) {
      if (lanes != 1 && lanes != args[arg].n_items) return -1;
      lanes = args[arg].n_items;
    }
  }
  for (int i = 0; i < info->n_globals; i++)
    if (args[info->globals[i]].n_items != lanes) return -1;
  return lanes;
}

static int poly_linear_lane_vars(
    PolyVarBinding *bindings,
    int n_bindings,
    PolyUOp *device_num,
    int lane,
    PolyVarBinding **out,
    int *n_out
) {
  if (!out || !n_out || n_bindings < 0 || (n_bindings > 0 && !bindings)) return -1;
  /* unwrap_multi merges one device binding into the call's variable map.
   * Bound both the C result count and its allocation before reading inputs. */
  if (device_num && n_bindings == INT_MAX) return -1;
  size_t capacity = (size_t)n_bindings + (device_num ? 1 : 0);
  if (capacity > SIZE_MAX / sizeof(PolyVarBinding)) return -1;
  PolyVarBinding *vars = capacity > 0 ? malloc((size_t)capacity * sizeof(*vars)) : NULL;
  if (capacity > 0 && !vars) return -1;
  if (n_bindings > 0) memcpy(vars, bindings, (size_t)n_bindings * sizeof(*vars));
  int count = n_bindings;
  if (device_num) {
    bool found = false;
    for (int i = 0; i < count; i++) {
      if (!schedule_same_var(vars[i].var, device_num)) continue;
      vars[i].value = lane;
      found = true;
      break;
    }
    if (!found) vars[count++] = (PolyVarBinding){.var = device_num, .value = lane};
  }
  *out = vars;
  *n_out = count;
  return 0;
}

#ifdef POLY_TESTING
int poly_test_linear_lane_vars(
    PolyVarBinding *bindings,
    int n_bindings,
    PolyUOp *device_num,
    int lane,
    PolyVarBinding **out,
    int *n_out
) {
  return poly_linear_lane_vars(bindings, n_bindings, device_num, lane, out, n_out);
}
#endif

static int poly_linear_stats(
    PolyCtx *ctx,
    PolyUOp *body,
    PolyVarBinding *bindings,
    int n_bindings,
    uint64_t copy_bytes,
    double elapsed_ms,
    bool update_stats
) {
  if (!ctx || !update_stats || ctx->stats_suppression_depth > 0) return 0;
  uint64_t ops = 0, mem = copy_bytes, lds = 0;
  const PolyEstimates *estimates =
      body && body->op == POLY_OP_PROGRAM ? poly_program_estimates(body) : NULL;
  if (estimates && estimates->ops && estimates->lds && estimates->mem &&
      poly_estimates_infer(estimates, bindings, n_bindings, &ops, &lds, &mem) != 0)
    return -1;
  ctx->kernel_count = poly_counter_add_sat(ctx->kernel_count, 1);
  ctx->global_ops = poly_counter_add_sat(ctx->global_ops, ops);
  ctx->global_mem = poly_counter_add_sat(ctx->global_mem, mem);
  if (elapsed_ms >= 0.0) ctx->time_sum_s += elapsed_ms / 1000.0;
  return 0;
}

/* Current Tinygrad engine/realize.py:exec_copy. */
static int poly_exec_linear_copy(
    PolyCtx *ctx,
    PolyResolvedLinearArg *args,
    int n_args,
    int lanes,
    bool update_stats
) {
  if (!ctx || !args || n_args != 2 || lanes <= 0) return -1;
  for (int lane = 0; lane < lanes; lane++) {
    PolyBuffer *dst = args[0].items[lane];
    PolyBuffer *src = args[1].items[lane];
    if (!dst || !src || poly_buffer_is_multi(dst) || poly_buffer_is_multi(src) ||
        poly_buffer_handle_ensure_allocated(ctx, dst) != 0 ||
        poly_buffer_handle_ensure_allocated(ctx, src) != 0)
      return -1;
    /* exec_copy accepts Tensor.empty storage. Allocation restores any current
     * mirror; valid is coherence metadata, not a defined-byte requirement. */
    bool timing = update_stats && ctx->stats_suppression_depth == 0 && poly_debug_at_least(2);
    double start = timing ? poly_now_ms() : 0.0;
    if (poly_buffer_copy(dst, src) != 0) return -1;
    double elapsed = timing ? poly_now_ms() - start : -1.0;
    dst->valid = true;
    if (dst->base) dst->base->valid = true;
    if (dst->src) dst->src->valid = false;
    ctx->launch_count++;
    if (poly_linear_stats(ctx, NULL, NULL, 0, dst->nbytes, elapsed, update_stats) != 0) return -1;
  }
  return 0;
}

/* Current exec_kernel publishes every ProgramInfo.outs buffer after launch. */
static int poly_commit_resolved_write(PolyCtx *ctx, PolyResolvedLinearArg *arg, int lane) {
  if (!ctx || !arg || lane < 0 || lane >= arg->n_items || !arg->items) return -1;
  PolyBuffer *written = arg->items[lane];
  if (!written) return -1;
  written->valid = true;
  if (written->src) written->src->valid = false;
  if (written->base) {
    written->base->valid = true;
    if (written->base->src) written->base->src->valid = false;
  }
  const PolyUOp *identity = poly_uop_get_buffer_identity(arg->uop);
  PolyBuffer *base = identity ? poly_buffer_get(ctx, (PolyUOp *)identity) : NULL;
  if (base) {
    base->valid = true;
    if (base->src) base->src->valid = false;
  }
  return 0;
}

/* v0.14 engine/realize.py:exec_kernel selects the runtime from the CALL's
 * exact device identity, not the PROGRAM's renderer target. */
static int poly_exec_linear_program(
    PolyCtx *ctx,
    PolyUOp *call,
    PolyResolvedLinearArg *resolved,
    int n_resolved,
    PolyVarBinding *bindings,
    int n_bindings,
    bool update_stats,
    bool wait
) {
  PolyUOp *program = call && call->n_src > 0 ? call->src[0] : NULL;
  const PolyProgramInfo *info = poly_program_info(ctx, program);
  int lanes = poly_linear_lane_count(info, resolved, n_resolved);
  if (!program || program->op != POLY_OP_PROGRAM || !info || lanes <= 0) return -1;

  PolyUOp *devices = call->n_src > 1 ? poly_uop_device_uop_cached(ctx, call->src[1], NULL) : NULL;
  if (!devices) return -1;
  int device_lanes = devices->arg.kind == POLY_ARG_STRING_TUPLE ? devices->arg.string_tuple.n : 1;
  if (device_lanes <= 0) return -1;
  /* Match zip(call devices, resolved lanes), including globals-free kernels. */
  if (lanes > device_lanes) lanes = device_lanes;

  PolyUOp *device_num = poly_call_device_num_var(ctx, poly_program_kernel_body(program));
  for (int lane = 0; lane < lanes; lane++) {
    PolyUOp *device_uop = devices->arg.kind == POLY_ARG_STRING_TUPLE
                              ? poly_device_uop_from_name(ctx, devices->arg.string_tuple.vals[lane])
                              : devices;
    PolyDevice device = poly_device_from_device_uop(device_uop);
    if (!device_uop || device == POLY_DEVICE_AUTO || !poly_device_can_execute(device)) return -1;

    PolyRunner runner = {0};
    PolyRuntimeCacheEntry *runtime_entry = NULL;
    int lower_rc = poly_lower_compute_call_cached(
        ctx, call, device_uop, device, poly_runtime_cache_env_stamp(), &runner, true, &runtime_entry
    );
    if (lower_rc != 0) return -1;
    runner.n_params = info->n_globals;
    if (poly_bind_runner_vars(ctx, &runner) != 0) {
      if (runtime_entry)
        poly_runtime_cache_entry_release(runtime_entry);
      else
        poly_runner_cleanup(&runner, device);
      return -1;
    }

    PolyVarBinding *lane_bindings = NULL;
    int n_lane_bindings = 0;
    void **args = NULL;
    int *var_values = NULL;
    int rc = -1;
    if (poly_linear_lane_vars(
            bindings, n_bindings, device_num, lane, &lane_bindings, &n_lane_bindings
        ) != 0)
      goto lane_cleanup;
    int n_runtime_args = runner.n_params + runner.n_vars;
    args = calloc((size_t)(n_runtime_args > 0 ? n_runtime_args : 1), sizeof(*args));
    var_values = runner.n_vars > 0 ? calloc((size_t)runner.n_vars, sizeof(*var_values)) : NULL;
    if (!args || (runner.n_vars > 0 && !var_values)) goto lane_cleanup;

    for (int i = 0; i < runner.n_params; i++) {
      int arg = info->globals[i];
      PolyBuffer *buffer = resolved[arg].items[lane];
      if (!buffer || poly_buffer_handle_get_buf(ctx, buffer, device, &args[i]) != 0)
        goto lane_cleanup;
    }
    for (int i = 0; i < runner.n_vars; i++) {
      int info_index = runner.var_indices[i];
      if (info_index < 0 || info_index >= info->n_vars) goto lane_cleanup;
      bool found = false;
      for (int j = 0; j < n_lane_bindings; j++) {
        if (!schedule_same_var(lane_bindings[j].var, info->vars[info_index])) continue;
        var_values[i] = (int)lane_bindings[j].value;
        args[runner.n_params + i] = &var_values[i];
        found = true;
        break;
      }
      if (!found) goto lane_cleanup;
    }
    if (poly_resolve_runner_launch_dims(&runner, lane_bindings, n_lane_bindings) != 0)
      goto lane_cleanup;

    /* exec_kernel forwards wait independently of update_stats. CUDA returns
     * device-event time; synchronous C backends use the call's wall time. */
    runner.wait = wait;
    runner.elapsed_us = NAN;
    double start = wait ? poly_now_ms() : 0.0;
    const PolyBackendDesc *backend = poly_backend_get(device);
    int exec_rc = runner.execute                ? runner.execute(&runner, args, n_runtime_args)
                  : backend && backend->execute ? backend->execute(&runner, args, n_runtime_args)
                                                : -1;
#ifdef POLY_HAS_HIP
    if (exec_rc == 0 && wait && device == POLY_DEVICE_HIP) exec_rc = poly_hip_sync();
#endif
    double elapsed = !wait                         ? -1.0
                     : isfinite(runner.elapsed_us) ? runner.elapsed_us / 1000.0
                                                   : poly_now_ms() - start;
    if (exec_rc != 0) goto lane_cleanup;

    for (int i = 0; i < info->n_outs; i++) {
      int arg = info->outs[i];
      if (arg < 0 || arg >= n_resolved ||
          poly_commit_resolved_write(ctx, &resolved[arg], lane) != 0)
        goto lane_cleanup;
    }
    ctx->launch_count++;
    if (poly_linear_stats(ctx, program, lane_bindings, n_lane_bindings, 0, elapsed, update_stats) !=
        0)
      goto lane_cleanup;
    rc = 0;

  lane_cleanup:
    free(args);
    free(var_values);
    free(lane_bindings);
    if (runtime_entry) {
      poly_runner_cleanup_local_mappings(&runner);
      poly_runtime_cache_entry_release(runtime_entry);
    } else {
      poly_runner_cleanup(&runner, device);
    }
    if (rc != 0) return -1;
  }
  return 0;
}

#ifdef POLY_HAS_CUDA
typedef struct {
  PolyBuffer *base;
  size_t start;
  size_t end;
  int node;
} GraphResourceRange;

typedef struct {
  GraphResourceRange *items;
  int n;
  int capacity;
} GraphResourceMap;

typedef struct {
  PolyUOp *call;
  PolyResolvedLinearArg *resolved;
  bool *outs;
  bool *ins;
  int n_args;
  void **runtime_args;
  int *scalar_values;
} PreparedGraphCall;

static int graph_resource_append(GraphResourceMap *map, GraphResourceRange range) {
  if (!map || !range.base || range.start >= range.end) return -1;
  if (map->n >= map->capacity) {
    int next = map->capacity ? map->capacity * 2 : 16;
    GraphResourceRange *grown = realloc(map->items, (size_t)next * sizeof(*grown));
    if (!grown) return -1;
    map->items = grown;
    map->capacity = next;
  }
  map->items[map->n++] = range;
  return 0;
}

static bool graph_resource_overlaps(
    const GraphResourceRange *range,
    PolyBuffer *base,
    size_t start,
    size_t end
) {
  return range && range->base == base && range->start < end && start < range->end;
}

/* Current Tinygrad device.py:DepsTracker subtracts a new write from both
 * prior read and write ranges before publishing the writer. */
static int graph_resource_remove(
    GraphResourceMap *map,
    PolyBuffer *base,
    size_t start,
    size_t end
) {
  GraphResourceMap kept = {0};
  for (int i = 0; i < map->n; i++) {
    GraphResourceRange range = map->items[i];
    if (!graph_resource_overlaps(&range, base, start, end)) {
      if (graph_resource_append(&kept, range) != 0) goto fail;
      continue;
    }
    if (range.start < start) {
      GraphResourceRange left = range;
      left.end = start < range.end ? start : range.end;
      if (left.start < left.end && graph_resource_append(&kept, left) != 0) goto fail;
    }
    if (end < range.end) {
      GraphResourceRange right = range;
      right.start = end > range.start ? end : range.start;
      if (right.start < right.end && graph_resource_append(&kept, right) != 0) goto fail;
    }
  }
  free(map->items);
  *map = kept;
  return 0;

fail:
  free(kept.items);
  return -1;
}

static int graph_resource_bounds(
    PolyBuffer *buffer,
    PolyBuffer **base,
    size_t *start,
    size_t *end
) {
  if (!buffer || !base || !start || !end || buffer->nbytes == 0 ||
      buffer->offset > SIZE_MAX - buffer->nbytes)
    return -1;
  *base = buffer->base ? buffer->base : buffer;
  *start = buffer->base ? buffer->offset : 0;
  *end = *start + buffer->nbytes;
  return 0;
}

static int graph_append_dependency(int **items, int *n, int *capacity, int node) {
  for (int i = 0; i < *n; i++)
    if ((*items)[i] == node) return 0;
  if (*n >= *capacity) {
    int next = *capacity ? *capacity * 2 : 8;
    int *grown = realloc(*items, (size_t)next * sizeof(*grown));
    if (!grown) return -1;
    *items = grown;
    *capacity = next;
  }
  (*items)[(*n)++] = node;
  return 0;
}

static int graph_collect_dependencies(
    const GraphResourceMap *map,
    PolyBuffer *base,
    size_t start,
    size_t end,
    int **dependencies,
    int *n_dependencies,
    int *capacity
) {
  for (int i = 0; i < map->n; i++)
    if (graph_resource_overlaps(&map->items[i], base, start, end) &&
        graph_append_dependency(dependencies, n_dependencies, capacity, map->items[i].node) != 0)
      return -1;
  return 0;
}

static int graph_build_dependencies(
    PreparedGraphCall *prepared,
    int n_calls,
    PolyCudaGraphCallSpec *specs,
    int ***owned_out
) {
  GraphResourceMap reads = {0}, writes = {0};
  int **owned = calloc((size_t)n_calls, sizeof(*owned));
  if (!owned) return -1;
  for (int node = 0; node < n_calls; node++) {
    int n_dependencies = 0, capacity = 0;
    for (int arg = 0; arg < prepared[node].n_args; arg++) {
      if (!prepared[node].ins[arg] && !prepared[node].outs[arg]) continue;
      PolyBuffer *base = NULL;
      size_t start = 0, end = 0;
      if (prepared[node].resolved[arg].n_items != 1 ||
          graph_resource_bounds(prepared[node].resolved[arg].items[0], &base, &start, &end) != 0 ||
          graph_collect_dependencies(
              &writes, base, start, end, &owned[node], &n_dependencies, &capacity
          ) != 0 ||
          (prepared[node].outs[arg] &&
           graph_collect_dependencies(
               &reads, base, start, end, &owned[node], &n_dependencies, &capacity
           ) != 0))
        goto fail;
    }
    specs[node].dependencies = owned[node];
    specs[node].n_dependencies = n_dependencies;
    for (int arg = 0; arg < prepared[node].n_args; arg++) {
      if (!prepared[node].ins[arg] && !prepared[node].outs[arg]) continue;
      PolyBuffer *base = NULL;
      size_t start = 0, end = 0;
      if (graph_resource_bounds(prepared[node].resolved[arg].items[0], &base, &start, &end) != 0)
        goto fail;
      GraphResourceRange range = {base, start, end, node};
      if (prepared[node].outs[arg]) {
        if (graph_resource_remove(&writes, base, start, end) != 0 ||
            graph_resource_remove(&reads, base, start, end) != 0 ||
            graph_resource_append(&writes, range) != 0)
          goto fail;
      } else if (graph_resource_append(&reads, range) != 0) {
        goto fail;
      }
    }
  }
  free(reads.items);
  free(writes.items);
  *owned_out = owned;
  return 0;

fail:
  for (int i = 0; i < n_calls; i++)
    free(owned[i]);
  free(owned);
  free(reads.items);
  free(writes.items);
  return -1;
}

static void prepared_graph_calls_free(PreparedGraphCall *calls, int n_calls) {
  if (!calls) return;
  for (int i = 0; i < n_calls; i++) {
    poly_resolved_linear_args_free(calls[i].resolved, calls[i].n_args);
    free(calls[i].outs);
    free(calls[i].ins);
    free(calls[i].runtime_args);
    free(calls[i].scalar_values);
  }
  free(calls);
}

static PolyGraphCacheEntry *graph_cache_entry_new(PolyCtx *ctx, PolyUOp *function, int n_nodes) {
  PolyGraphCacheEntry *entry = calloc(1, sizeof(*entry));
  if (!entry) return NULL;
  if (poly_uop_retain(ctx, function) != 0) {
    free(entry);
    return NULL;
  }
  entry->function = function;
  entry->n_nodes = n_nodes;
  entry->runners = calloc((size_t)n_nodes, sizeof(*entry->runners));
  entry->runtime_entries = calloc((size_t)n_nodes, sizeof(*entry->runtime_entries));
  if (!entry->runners || !entry->runtime_entries) {
    free(entry->runners);
    free(entry->runtime_entries);
    poly_uop_release(ctx, function);
    free(entry);
    return NULL;
  }
  return entry;
}

static void graph_cache_entry_destroy(PolyCtx *ctx, PolyGraphCacheEntry *entry) {
  if (!entry) return;
  poly_graph_cache_entry_free(NULL, entry, ctx);
}

/* Current Tinygrad engine/realize.py:get_graph_runtime + exec_graph. */
static int poly_exec_linear_graph(
    PolyCtx *ctx,
    PolyUOp *call,
    PolyVarBinding *bindings,
    int n_bindings,
    PolyUOp **input_uops,
    int n_input_uops,
    bool update_stats,
    bool wait
) {
  PolyUOp *function = call && call->n_src > 0 ? call->src[0] : NULL;
  PolyUOp *linear = function && function->op == POLY_OP_CUSTOM_FUNCTION &&
                            function->arg.kind == POLY_ARG_STRING && function->arg.str &&
                            strcmp(function->arg.str, "graph") == 0 && function->n_src == 1
                        ? function->src[0]
                        : NULL;
  if (!linear || linear->op != POLY_OP_LINEAR || linear->n_src <= 0) return -1;

  PolyGraphCacheEntry *entry =
      poly_map_get(ctx->graph_cache, poly_ptr_hash(function), function, poly_ptr_eq);
  bool new_entry = entry == NULL;
  if (new_entry && !(entry = graph_cache_entry_new(ctx, function, linear->n_src))) return -1;
  if (entry->n_nodes != linear->n_src) goto fail;

  PreparedGraphCall *prepared = calloc((size_t)linear->n_src, sizeof(*prepared));
  PolyCudaGraphCallSpec *specs = calloc((size_t)linear->n_src, sizeof(*specs));
  int **owned_dependencies = NULL;
  if (!prepared || !specs) goto fail_prepared;

  uint64_t total_ops = 0, total_mem = 0;
  for (int node = 0; node < linear->n_src; node++) {
    PreparedGraphCall *item = &prepared[node];
    item->call = linear->src[node];
    if (!item->call || item->call->op != POLY_OP_CALL || item->call->n_src < 1) goto fail_prepared;
    item->n_args = poly_call_n_buffer_args(item->call);
    item->outs = calloc((size_t)(item->n_args > 0 ? item->n_args : 1), sizeof(*item->outs));
    item->ins = calloc((size_t)(item->n_args > 0 ? item->n_args : 1), sizeof(*item->ins));
    item->resolved = calloc((size_t)(item->n_args > 0 ? item->n_args : 1), sizeof(*item->resolved));
    if (!item->outs || !item->ins || !item->resolved ||
        poly_call_get_outs_ins(ctx, item->call, item->outs, item->ins, item->n_args) != 0)
      goto fail_prepared;
    for (int arg = 0; arg < item->n_args; arg++) {
      PolyUOp *resolved = poly_resolve_linear_param(
          ctx, poly_call_buffer_arg(item->call, arg), input_uops, n_input_uops
      );
      if (!resolved ||
          poly_resolve_linear_arg(ctx, resolved, item->ins[arg], &item->resolved[arg]) != 0 ||
          item->resolved[arg].n_items != 1 ||
          item->resolved[arg].items[0]->device != POLY_DEVICE_CUDA)
        goto fail_prepared;
    }

    PolyUOp *body = item->call->src[0];
    if (body->op == POLY_OP_COPY) {
      if (item->n_args != 2 || !item->outs[0] || !item->ins[1]) goto fail_prepared;
      PolyBuffer *dst = item->resolved[0].items[0];
      PolyBuffer *src = item->resolved[1].items[0];
      if (poly_buffer_handle_ensure_allocated(ctx, dst) != 0 ||
          poly_buffer_handle_ensure_allocated(ctx, src) != 0 || !dst->ptr || !src->ptr ||
          dst->nbytes != src->nbytes)
        goto fail_prepared;
      specs[node].kind = POLY_CUDA_GRAPH_COPY;
      specs[node].value.copy.dst = dst->ptr;
      specs[node].value.copy.src = src->ptr;
      specs[node].value.copy.nbytes = dst->nbytes;
      total_mem = poly_counter_add_sat(total_mem, dst->nbytes);
      continue;
    }
    if (body->op != POLY_OP_PROGRAM) goto fail_prepared;

    const PolyProgramInfo *info = poly_program_info(ctx, body);
    if (!info) goto fail_prepared;
    PolyRunner *runner = &entry->runners[node];
    if (new_entry) {
      PolyUOp *device_uop = poly_device_uop(ctx, POLY_DEVICE_CUDA);
      if (!device_uop ||
          poly_lower_compute_call_cached(
              ctx, item->call, device_uop, POLY_DEVICE_CUDA, poly_runtime_cache_env_stamp(), runner,
              true, &entry->runtime_entries[node]
          ) != 0)
        goto fail_prepared;
      runner->n_params = info->n_globals;
      if (poly_bind_runner_vars(ctx, runner) != 0) goto fail_prepared;
    }
    CudaRunnerHandle *handle = (CudaRunnerHandle *)runner->handle;
    if (!handle || !handle->prog || runner->n_params != info->n_globals) goto fail_prepared;

    int n_runtime_args = runner->n_params + runner->n_vars;
    item->runtime_args =
        calloc((size_t)(n_runtime_args > 0 ? n_runtime_args : 1), sizeof(*item->runtime_args));
    item->scalar_values =
        runner->n_vars > 0 ? calloc((size_t)runner->n_vars, sizeof(*item->scalar_values)) : NULL;
    if (!item->runtime_args || (runner->n_vars > 0 && !item->scalar_values)) goto fail_prepared;
    for (int i = 0; i < runner->n_params; i++) {
      int arg = info->globals[i];
      PolyBuffer *buffer = item->resolved[arg].items[0];
      if (!buffer || poly_buffer_handle_ensure_allocated(ctx, buffer) != 0 || !buffer->ptr)
        goto fail_prepared;
      item->runtime_args[i] = buffer->ptr;
    }
    for (int i = 0; i < runner->n_vars; i++) {
      int info_index = runner->var_indices[i];
      bool found = false;
      if (info_index < 0 || info_index >= info->n_vars) goto fail_prepared;
      for (int j = 0; j < n_bindings; j++) {
        if (!schedule_same_var(bindings[j].var, info->vars[info_index])) continue;
        item->scalar_values[i] = (int)bindings[j].value;
        item->runtime_args[runner->n_params + i] = &item->scalar_values[i];
        found = true;
        break;
      }
      if (!found) goto fail_prepared;
    }
    if (poly_resolve_runner_launch_dims(runner, bindings, n_bindings) != 0) goto fail_prepared;
    specs[node].kind = POLY_CUDA_GRAPH_PROGRAM;
    specs[node].value.program.program = handle->prog;
    specs[node].value.program.args = item->runtime_args;
    specs[node].value.program.n_buffer_args = runner->n_params;
    specs[node].value.program.n_args = n_runtime_args;
    for (int dim = 0; dim < 3; dim++) {
      specs[node].value.program.grid[dim] = runner->grid[dim];
      specs[node].value.program.block[dim] = runner->block[dim];
    }
    uint64_t ops = 0, lds = 0, mem = 0;
    const PolyEstimates *estimates = poly_program_estimates(body);
    if (!estimates || poly_estimates_infer(estimates, bindings, n_bindings, &ops, &lds, &mem) != 0)
      goto fail_prepared;
    total_ops = poly_counter_add_sat(total_ops, ops);
    total_mem = poly_counter_add_sat(total_mem, mem);
  }

  if (graph_build_dependencies(prepared, linear->n_src, specs, &owned_dependencies) != 0)
    goto fail_prepared;
  if (new_entry) {
    entry->graph = poly_cuda_graph_create(specs, linear->n_src);
    if (!entry->graph) goto fail_prepared;
    poly_map_set(ctx->graph_cache, poly_ptr_hash(function), function, entry, poly_ptr_eq);
  } else if (poly_cuda_graph_update(entry->graph, specs, linear->n_src) != 0) {
    goto fail_prepared;
  }

  double elapsed_us = 0;
  int launch_rc = wait ? poly_cuda_graph_launch_timed(entry->graph, &elapsed_us)
                       : poly_cuda_graph_launch(entry->graph);
  if (launch_rc != 0) goto fail_prepared;
  double elapsed = wait ? elapsed_us / 1000.0 : -1.0;
  for (int node = 0; node < linear->n_src; node++)
    for (int arg = 0; arg < prepared[node].n_args; arg++)
      if (prepared[node].outs[arg] &&
          poly_commit_resolved_write(ctx, &prepared[node].resolved[arg], 0) != 0)
        goto fail_prepared;
  ctx->launch_count++;
  if (update_stats && ctx->stats_suppression_depth == 0) {
    ctx->kernel_count = poly_counter_add_sat(ctx->kernel_count, 1);
    ctx->global_ops = poly_counter_add_sat(ctx->global_ops, total_ops);
    ctx->global_mem = poly_counter_add_sat(ctx->global_mem, total_mem);
    if (elapsed >= 0.0) ctx->time_sum_s += elapsed / 1000.0;
  }

  for (int i = 0; i < linear->n_src; i++)
    free(owned_dependencies[i]);
  free(owned_dependencies);
  free(specs);
  prepared_graph_calls_free(prepared, linear->n_src);
  return 0;

fail_prepared:
  if (owned_dependencies) {
    for (int i = 0; i < linear->n_src; i++)
      free(owned_dependencies[i]);
    free(owned_dependencies);
  }
  free(specs);
  prepared_graph_calls_free(prepared, linear->n_src);
fail:
  if (new_entry) graph_cache_entry_destroy(ctx, entry);
  return -1;
}
#endif

/* C control body for current Tinygrad run_linear; the public wrapper below
 * scopes deferred residency collection across every early return. */
static int run_linear_impl(
    PolyCtx *ctx,
    PolyUOp *linear,
    PolyVarBinding *var_bindings,
    int n_var_bindings,
    PolyUOp **input_uops,
    int n_input_uops,
    bool update_stats,
    bool jit,
    bool wait
) {
  /* Approved PG-PARITY-013 boundary: unsupported exact accelerator ordinals
   * fail before the backend-class enum can alias them to ordinal zero. */
  if (!poly_uop_explicit_devices_supported(ctx, linear)) return -1;
  PolyUOp *executable = jit ? linear : poly_compile_linear(ctx, linear, -1);
  if (!executable || executable->op != POLY_OP_LINEAR) return -1;
  bool debug = poly_debug_at_least(7);

  for (int call_index = 0; call_index < executable->n_src; call_index++) {
    PolyUOp *call = executable->src[call_index];
    if (!call || call->op != POLY_OP_CALL || call->n_src < 1) return -1;
    PolyUOp *body = call->src[0];
    if (debug)
      fprintf(
          stderr, "[polygrad:run_linear] call=%d/%d body=%s args=%d\n", call_index,
          executable->n_src, poly_op_name(body->op), poly_call_n_buffer_args(call)
      );
    bool graph = body->op == POLY_OP_CUSTOM_FUNCTION && body->arg.kind == POLY_ARG_STRING &&
                 body->arg.str && strcmp(body->arg.str, "graph") == 0;
    if (graph) {
#ifdef POLY_HAS_CUDA
      if (poly_exec_linear_graph(
              ctx, call, var_bindings, n_var_bindings, input_uops, n_input_uops, update_stats, wait
          ) != 0)
        return -1;
      continue;
#else
      return -1;
#endif
    }
    if (body->op != POLY_OP_COPY && body->op != POLY_OP_PROGRAM) return -1;

    int n_args = poly_call_n_buffer_args(call);
    bool *outs = n_args > 0 ? calloc((size_t)n_args, sizeof(*outs)) : NULL;
    bool *ins = n_args > 0 ? calloc((size_t)n_args, sizeof(*ins)) : NULL;
    PolyResolvedLinearArg *resolved = n_args > 0 ? calloc((size_t)n_args, sizeof(*resolved)) : NULL;
    if ((n_args > 0 && (!outs || !ins || !resolved)) ||
        poly_call_get_outs_ins(ctx, call, outs, ins, n_args) != 0) {
      free(outs);
      free(ins);
      poly_resolved_linear_args_free(resolved, n_args);
      return -1;
    }

    bool ok = true;
    /* exec_kernel resolves every PARAM slot, but only ProgramInfo.globals
     * materialize storage. Eliminated CALL arguments must not allocate. */
    for (int i = 0; i < n_args; i++) {
      resolved[i].uop =
          poly_resolve_linear_param(ctx, poly_call_buffer_arg(call, i), input_uops, n_input_uops);
      if (!resolved[i].uop) {
        ok = false;
        break;
      }
    }
    const PolyProgramInfo *info = body->op == POLY_OP_PROGRAM ? poly_program_info(ctx, body) : NULL;
    int n_buffers = info ? info->n_globals : n_args;
    for (int slot = 0; ok && slot < n_buffers; slot++) {
      int i = info ? info->globals[slot] : slot;
      if (resolved[i].items) continue;
      PolyUOp *arg = resolved[i].uop;
      if (poly_resolve_linear_arg(ctx, arg, ins[i], &resolved[i]) != 0) {
        if (debug)
          fprintf(
              stderr, "[polygrad:run_linear] call=%d arg=%d resolve failed op=%s read=%d\n",
              call_index, i, arg ? poly_op_name(arg->op) : "NULL", ins[i]
          );
        ok = false;
        break;
      }
    }
    int rc = -1;
    if (ok && body->op == POLY_OP_COPY) {
      int lanes = 1;
      for (int i = 0; i < n_args; i++) {
        if (resolved[i].n_items > 1) {
          if (lanes != 1 && lanes != resolved[i].n_items) {
            ok = false;
            break;
          }
          lanes = resolved[i].n_items;
        }
      }
      for (int i = 0; ok && i < n_args; i++)
        if (resolved[i].n_items != lanes) ok = false;
      if (ok) rc = poly_exec_linear_copy(ctx, resolved, n_args, lanes, update_stats);
    } else if (ok) {
      rc = poly_exec_linear_program(
          ctx, call, resolved, n_args, var_bindings, n_var_bindings, update_stats, wait
      );
    }
    free(outs);
    free(ins);
    poly_resolved_linear_args_free(resolved, n_args);
    if (rc != 0) {
      if (debug)
        fprintf(
            stderr, "[polygrad:run_linear] call=%d body=%s execution failed\n", call_index,
            poly_op_name(body->op)
        );
      return -1;
    }
  }
  return 0;
}

/* Current Tinygrad engine/realize.py:run_linear. LINEAR is consumed directly;
 * no secondary schedule/template/runtime graph is constructed. */
int poly_run_linear(
    PolyCtx *ctx,
    PolyUOp *linear,
    PolyVarBinding *var_bindings,
    int n_var_bindings,
    PolyUOp **input_uops,
    int n_input_uops,
    bool update_stats,
    bool jit,
    bool wait
) {
  if (!ctx || !linear || linear->op != POLY_OP_LINEAR || n_var_bindings < 0 || n_input_uops < 0 ||
      (n_var_bindings > 0 && !var_bindings) || (n_input_uops > 0 && !input_uops))
    return -1;
  /* Tinygrad 2026-08-22/a9069c177a9d engine/realize.py:315-323 keeps LINEAR
   * alive as a Python local. Retain the C root; an allocator triggers
   * residency collection only when execution actually needs memory. */
  if (poly_uop_retain(ctx, linear) != 0) return -1;
  ctx->execution_depth++;
  int rc = run_linear_impl(
      ctx, linear, var_bindings, n_var_bindings, input_uops, n_input_uops, update_stats, jit,
      wait || poly_debug_at_least(2)
  );
  ctx->execution_depth--;
  poly_uop_release(ctx, linear);
  return rc;
}

/* Structural hash/eq for graph caches.
 * Cached schedules/programs need to match computations that are structurally
 * identical but use different BUFFER UOp instances, such as fresh training
 * step buffers. We hash/compare the computation DAG structure: ops, dtypes,
 * args, and connectivity, treating BUFFER storage identities as positional
 * placeholders (first encountered = 0, etc.). Exact movement views remain
 * ordinary graph topology, matching current Tinygrad call arguments.
 */

typedef struct {
  PolyMap *visited; /* UOp* -> 1-based index into hashes. Dynamic for model-scale DAGs. */
  uint32_t *hashes;
  int n_hashes;
  int cap_hashes;
  PolyUOp **bufs;
  int n_bufs;
  int cap_bufs;
} StructHashCtx;

static uint32_t struct_hash_impl(PolyUOp *u, StructHashCtx *ctx) {
  /* Check if already visited */
  void *memo_val = poly_map_get(ctx->visited, poly_ptr_hash(u), u, poly_ptr_eq);
  if (memo_val) return ctx->hashes[(int)((intptr_t)memo_val - 1)];

  uint32_t h = 0x811c9dc5; /* FNV-1a offset basis */

  if (u->op == POLY_OP_BUFFER) {
    /* Storage identities use positional IDs instead of pointer identity. */
    int buf_id = -1;
    for (int i = 0; i < ctx->n_bufs; i++) {
      if (ctx->bufs[i] == u) {
        buf_id = i;
        break;
      }
    }
    if (buf_id < 0) {
      if (ctx->n_bufs >= ctx->cap_bufs) {
        int new_cap = ctx->cap_bufs ? ctx->cap_bufs * 2 : 64;
        PolyUOp **new_bufs = realloc(ctx->bufs, (size_t)new_cap * sizeof(PolyUOp *));
        if (!new_bufs) return h;
        ctx->bufs = new_bufs;
        ctx->cap_bufs = new_cap;
      }
      buf_id = ctx->n_bufs;
      ctx->bufs[ctx->n_bufs++] = u;
    }
    h ^= (uint32_t)u->op;
    h *= 0x01000193;
    h ^= (uint32_t)u->dtype.priority;
    h *= 0x01000193;
    h ^= (uint32_t)u->dtype.bitsize;
    h *= 0x01000193;
    h ^= (uint32_t)buf_id;
    h *= 0x01000193;
    h ^= poly_arg_hash(u->arg);
    h *= 0x01000193;
  } else {
    h ^= (uint32_t)u->op;
    h *= 0x01000193;
    h ^= (uint32_t)u->dtype.priority;
    h *= 0x01000193;
    h ^= (uint32_t)u->dtype.bitsize;
    h *= 0x01000193;
    for (int i = 0; i < u->n_src; i++) {
      h ^= struct_hash_impl(u->src[i], ctx);
      h *= 0x01000193;
    }
    h ^= poly_arg_hash(u->arg);
    h *= 0x01000193;
  }

  if (ctx->n_hashes >= ctx->cap_hashes) {
    int new_cap = ctx->cap_hashes ? ctx->cap_hashes * 2 : 1024;
    uint32_t *new_hashes = realloc(ctx->hashes, (size_t)new_cap * sizeof(uint32_t));
    if (!new_hashes) return h;
    ctx->hashes = new_hashes;
    ctx->cap_hashes = new_cap;
  }
  int idx = ctx->n_hashes++;
  ctx->hashes[idx] = h;
  poly_map_set(ctx->visited, poly_ptr_hash(u), u, (void *)(intptr_t)(idx + 1), poly_ptr_eq);
  return h;
}

uint32_t poly_structural_hash(PolyUOp *u) {
  if (!u) return 0;
  StructHashCtx ctx;
  memset(&ctx, 0, sizeof(ctx));
  ctx.visited = poly_map_new(1024);
  if (!ctx.visited) return 0;
  uint32_t h = struct_hash_impl(u, &ctx);
  poly_map_destroy(ctx.visited);
  free(ctx.hashes);
  free(ctx.bufs);
  return h;
}

/* Structural equality */

typedef struct {
  PolyMap *a_to_b;
  PolyMap *b_to_a;
} BufPairs;

typedef struct {
  PolyMap *a_to_b;
  PolyMap *b_to_a;
} EqVisited;

static bool struct_eq_impl(PolyUOp *a, PolyUOp *b, BufPairs *bp, EqVisited *ev) {
  if (a == b) return true;
  if (!a || !b) return false;

  /* Check if this pair already visited (DAG sharing) */
  void *seen_b = poly_map_get(ev->a_to_b, poly_ptr_hash(a), a, poly_ptr_eq);
  if (seen_b) return seen_b == b;
  if (poly_map_get(ev->b_to_a, poly_ptr_hash(b), b, poly_ptr_eq)) return false;

  poly_map_set(ev->a_to_b, poly_ptr_hash(a), a, b, poly_ptr_eq);
  poly_map_set(ev->b_to_a, poly_ptr_hash(b), b, a, poly_ptr_eq);

  /* Both executable storage identities? Track positional correspondence. */
  bool a_storage = a->op == POLY_OP_BUFFER;
  bool b_storage = b->op == POLY_OP_BUFFER;
  if (a_storage || b_storage) {
    if (!a_storage || !b_storage || a->op != b->op) return false;
    if (!poly_dtype_eq(a->dtype, b->dtype)) return false;
    if (!poly_arg_eq(a->arg, b->arg)) return false;
    /* Check existing mapping */
    void *mapped_b = poly_map_get(bp->a_to_b, poly_ptr_hash(a), a, poly_ptr_eq);
    if (mapped_b) return mapped_b == b;
    if (poly_map_get(bp->b_to_a, poly_ptr_hash(b), b, poly_ptr_eq)) return false;
    poly_map_set(bp->a_to_b, poly_ptr_hash(a), a, b, poly_ptr_eq);
    poly_map_set(bp->b_to_a, poly_ptr_hash(b), b, a, poly_ptr_eq);
    return true;
  }

  /* Same op, dtype, n_src, arg? */
  if (a->op != b->op) return false;
  if (!poly_dtype_eq(a->dtype, b->dtype)) return false;
  if (a->n_src != b->n_src) return false;
  if (!poly_arg_eq(a->arg, b->arg)) return false;

  /* Recursively compare sources */
  for (int i = 0; i < a->n_src; i++) {
    if (!struct_eq_impl(a->src[i], b->src[i], bp, ev)) return false;
  }
  return true;
}

bool poly_structural_eq(const void *a, const void *b) {
  BufPairs bp = {.a_to_b = poly_map_new(64), .b_to_a = poly_map_new(64)};
  EqVisited ev = {.a_to_b = poly_map_new(1024), .b_to_a = poly_map_new(1024)};
  if (!bp.a_to_b || !bp.b_to_a || !ev.a_to_b || !ev.b_to_a) {
    if (bp.a_to_b) poly_map_destroy(bp.a_to_b);
    if (bp.b_to_a) poly_map_destroy(bp.b_to_a);
    if (ev.a_to_b) poly_map_destroy(ev.a_to_b);
    if (ev.b_to_a) poly_map_destroy(ev.b_to_a);
    return false;
  }
  bool ok = struct_eq_impl((PolyUOp *)a, (PolyUOp *)b, &bp, &ev);
  poly_map_destroy(bp.a_to_b);
  poly_map_destroy(bp.b_to_a);
  poly_map_destroy(ev.a_to_b);
  poly_map_destroy(ev.b_to_a);
  return ok;
}
