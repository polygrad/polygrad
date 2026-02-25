/*
 * uop.c — UOp creation with CSE, toposort, pretty-print
 *
 * Mirrors tinygrad's UOp class and UOpMetaClass hash-consing cache.
 * All UOps are allocated from the context's arena.
 */

#include "polygrad.h"
#include "arena.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* ── PolyArg equality and hashing ──────────────────────────────────────── */

bool poly_arg_eq(PolyArg a, PolyArg b) {
  if (a.kind != b.kind) {
    /* Migration compatibility: allow INT-vs-RANGE equality for RANGE ids. */
    if (poly_arg_is_range(a) && poly_arg_is_range(b)) {
      if (poly_range_axis_id(a) != poly_range_axis_id(b)) return false;
      if (poly_range_axis_type(a) != poly_range_axis_type(b)) return false;
      if (poly_range_n_extra(a) != poly_range_n_extra(b)) return false;
      int n_extra = poly_range_n_extra(a);
      if (n_extra == 0) return true;
      return memcmp(poly_range_extra(a), poly_range_extra(b),
                    (size_t)n_extra * sizeof(int64_t)) == 0;
    }
    return false;
  }
  switch (a.kind) {
    case POLY_ARG_NONE:    return true;
    case POLY_ARG_INVALID: return true;
    case POLY_ARG_INT:     return a.i == b.i;
    case POLY_ARG_FLOAT:   /* bitwise compare to distinguish -0.0 from 0.0 */
      { uint64_t ba, bb;
        memcpy(&ba, &a.f, 8); memcpy(&bb, &b.f, 8);
        return ba == bb; }
    case POLY_ARG_BOOL:    return a.b == b.b;
    case POLY_ARG_STRING:  return a.str == b.str || (a.str && b.str && strcmp(a.str, b.str) == 0);
    case POLY_ARG_OPS:     return a.ops == b.ops;
    case POLY_ARG_INT_TUPLE:
      if (a.int_tuple.n != b.int_tuple.n) return false;
      if (a.int_tuple.n == 0) return true;
      return memcmp(a.int_tuple.vals, b.int_tuple.vals, a.int_tuple.n * sizeof(int64_t)) == 0;
    case POLY_ARG_PAIR_TUPLE:
      if (a.pair_tuple.n != b.pair_tuple.n) return false;
      if (a.pair_tuple.n == 0) return true;
      return memcmp(a.pair_tuple.pairs, b.pair_tuple.pairs, a.pair_tuple.n * 2 * sizeof(int64_t)) == 0;
    case POLY_ARG_REDUCE_AXIS:
      if (a.reduce_axis.op != b.reduce_axis.op) return false;
      if (a.reduce_axis.n != b.reduce_axis.n) return false;
      if (a.reduce_axis.n == 0) return true;
      return memcmp(a.reduce_axis.axes, b.reduce_axis.axes, a.reduce_axis.n * sizeof(int64_t)) == 0;
    case POLY_ARG_RANGE:
      if (a.range.axis_id != b.range.axis_id) return false;
      if (a.range.axis_type != b.range.axis_type) return false;
      if (a.range.n_extra != b.range.n_extra) return false;
      if (a.range.n_extra == 0) return true;
      return memcmp(a.range.extra, b.range.extra, (size_t)a.range.n_extra * sizeof(int64_t)) == 0;
    case POLY_ARG_DEFINE_VAR:
      if (a.define_var.min_val != b.define_var.min_val) return false;
      if (a.define_var.max_val != b.define_var.max_val) return false;
      if (a.define_var.name == b.define_var.name) return true;
      if (!a.define_var.name || !b.define_var.name) return false;
      return strcmp(a.define_var.name, b.define_var.name) == 0;
  }
  return false;
}

static uint32_t hash_mix(uint32_t h, uint32_t v) {
  h ^= v;
  h *= 0x9e3779b9;
  h ^= h >> 16;
  return h;
}

uint32_t poly_arg_hash(PolyArg a) {
  uint32_t h = (uint32_t)a.kind;
  switch (a.kind) {
    case POLY_ARG_NONE:
    case POLY_ARG_INVALID:
      break;
    case POLY_ARG_INT:
      h = hash_mix(h, (uint32_t)(a.i ^ (a.i >> 32)));
      break;
    case POLY_ARG_FLOAT:
      { uint64_t bits; memcpy(&bits, &a.f, 8);
        h = hash_mix(h, (uint32_t)(bits ^ (bits >> 32))); }
      break;
    case POLY_ARG_BOOL:
      h = hash_mix(h, a.b ? 1 : 0);
      break;
    case POLY_ARG_STRING:
      if (a.str) {
        for (const char *p = a.str; *p; p++)
          h = hash_mix(h, (uint32_t)*p);
      }
      break;
    case POLY_ARG_OPS:
      h = hash_mix(h, (uint32_t)a.ops);
      break;
    case POLY_ARG_INT_TUPLE:
      for (int i = 0; i < a.int_tuple.n; i++)
        h = hash_mix(h, (uint32_t)(a.int_tuple.vals[i] ^ (a.int_tuple.vals[i] >> 32)));
      break;
    case POLY_ARG_PAIR_TUPLE:
      for (int i = 0; i < a.pair_tuple.n; i++) {
        h = hash_mix(h, (uint32_t)(a.pair_tuple.pairs[i][0] ^ (a.pair_tuple.pairs[i][0] >> 32)));
        h = hash_mix(h, (uint32_t)(a.pair_tuple.pairs[i][1] ^ (a.pair_tuple.pairs[i][1] >> 32)));
      }
      break;
    case POLY_ARG_REDUCE_AXIS:
      h = hash_mix(h, (uint32_t)a.reduce_axis.op);
      for (int i = 0; i < a.reduce_axis.n; i++)
        h = hash_mix(h, (uint32_t)(a.reduce_axis.axes[i] ^ (a.reduce_axis.axes[i] >> 32)));
      break;
    case POLY_ARG_RANGE:
      h = hash_mix(h, (uint32_t)(a.range.axis_id ^ (a.range.axis_id >> 32)));
      h = hash_mix(h, (uint32_t)a.range.axis_type);
      for (int i = 0; i < a.range.n_extra; i++)
        h = hash_mix(h, (uint32_t)(a.range.extra[i] ^ (a.range.extra[i] >> 32)));
      break;
    case POLY_ARG_DEFINE_VAR:
      if (a.define_var.name) {
        for (const char *p = a.define_var.name; *p; p++)
          h = hash_mix(h, (uint32_t)*p);
      }
      h = hash_mix(h, (uint32_t)(a.define_var.min_val ^ (a.define_var.min_val >> 32)));
      h = hash_mix(h, (uint32_t)(a.define_var.max_val ^ (a.define_var.max_val >> 32)));
      break;
  }
  return h;
}

/* ── CSE key: (op, dtype, src[], arg, tag) ────────────────────────────── */

typedef struct {
  PolyOps op;
  PolyDType dtype;
  PolyUOp **src;
  uint16_t n_src;
  PolyArg arg;
  int32_t tag;
} CseKey;

static uint32_t cse_hash(const CseKey *k) {
  uint32_t h = (uint32_t)k->op;
  h = hash_mix(h, (uint32_t)k->dtype.priority);
  h = hash_mix(h, (uint32_t)k->dtype.bitsize);
  h = hash_mix(h, (uint32_t)k->dtype.count);
  for (int i = 0; i < k->n_src; i++) {
    /* hash pointer as integer — identity-based like tinygrad */
    uintptr_t p = (uintptr_t)k->src[i];
    h = hash_mix(h, (uint32_t)(p ^ (sizeof(p) > 4 ? (uint32_t)(p >> 32) : 0)));
  }
  h = hash_mix(h, poly_arg_hash(k->arg));
  h = hash_mix(h, (uint32_t)k->tag);
  return h;
}

static bool cse_eq(const void *a, const void *b) {
  const CseKey *ka = a, *kb = b;
  if (ka->op != kb->op) return false;
  if (!poly_dtype_eq(ka->dtype, kb->dtype)) return false;
  if (ka->n_src != kb->n_src) return false;
  for (int i = 0; i < ka->n_src; i++)
    if (ka->src[i] != kb->src[i]) return false;
  if (!poly_arg_eq(ka->arg, kb->arg)) return false;
  if (ka->tag != kb->tag) return false;
  return true;
}

/* ── Context ──────────────────────────────────────────────────────────── */

struct PolyCtx {
  PolyArena *arena;
  PolyMap *cse;
  PolyMap *kernel_cache;  /* computation UOp* → PolyCachedKernel* (rendered bytes) */
};

PolyCtx *poly_ctx_new(void) {
  poly_init_group_ops();  /* ensure opset globals are initialized (Emscripten safety) */
  PolyCtx *ctx = malloc(sizeof(PolyCtx));
  if (!ctx) return NULL;
  ctx->arena = poly_arena_new(0);
  ctx->cse = poly_map_new(256);
  ctx->kernel_cache = poly_map_new(16);
  if (!ctx->arena || !ctx->cse || !ctx->kernel_cache) {
    if (ctx->arena) poly_arena_destroy(ctx->arena);
    if (ctx->cse) poly_map_destroy(ctx->cse);
    if (ctx->kernel_cache) poly_map_destroy(ctx->kernel_cache);
    free(ctx);
    return NULL;
  }
  return ctx;
}

static void free_cached_kernel(const void *key, void *value, void *userdata) {
  (void)key; (void)userdata;
  PolyCachedKernel *ck = value;
  free(ck->bytes);
  free(ck);
}

void poly_ctx_destroy(PolyCtx *ctx) {
  poly_map_foreach(ctx->kernel_cache, free_cached_kernel, NULL);
  poly_map_destroy(ctx->kernel_cache);
  poly_map_destroy(ctx->cse);
  poly_arena_destroy(ctx->arena);
  free(ctx);
}

bool poly_ctx_owns_ptr(PolyCtx *ctx, const void *p) {
  if (!ctx || !p) return false;
  uintptr_t addr = (uintptr_t)p;
  for (PolyArenaBlock *b = ctx->arena->head; b; b = b->next) {
    uintptr_t start = (uintptr_t)b->data;
    uintptr_t end = start + b->used;
    if (addr >= start && addr < end) return true;
  }
  return false;
}

PolyMap *poly_ctx_kernel_cache(PolyCtx *ctx) { return ctx->kernel_cache; }

/* ── UOp creation with CSE ────────────────────────────────────────────── */

PolyUOp *poly_uop(PolyCtx *ctx, PolyOps op, PolyDType dtype,
                 PolyUOp **src, int n_src, PolyArg arg)
{
  /* Build a CSE key on the stack */
  CseKey key = { op, dtype, src, (uint16_t)n_src, arg, 0 };
  uint32_t h = cse_hash(&key);

  /* DEFINE_LOCAL represents mutable accumulators — each REDUCE needs its
   * own unique accumulator, so skip CSE dedup for this op.  Without this,
   * two reductions in a multi-store kernel that happen to share identity
   * value and outer-range deps would get merged into a single acc variable,
   * corrupting both computations. */
  if (op != POLY_OP_DEFINE_LOCAL) {
    PolyUOp *existing = poly_map_get(ctx->cse, h, &key, cse_eq);
    if (existing) return existing;
  }

  /* Allocate new UOp in arena */
  PolyUOp *u = poly_arena_alloc(ctx->arena, sizeof(PolyUOp), _Alignof(PolyUOp));
  u->op = op;
  u->dtype = dtype;
  u->n_src = (uint16_t)n_src;
  u->arg = arg;
  u->tag = 0;
  u->hash = h;

  /* Copy src pointers into arena */
  if (n_src > 0) {
    u->src = poly_arena_alloc(ctx->arena, n_src * sizeof(PolyUOp*), _Alignof(PolyUOp*));
    memcpy(u->src, src, n_src * sizeof(PolyUOp*));
  } else {
    u->src = NULL;
  }

  /* Copy arg data that needs arena allocation */
  if (arg.kind == POLY_ARG_INT_TUPLE && arg.int_tuple.n > 0) {
    int64_t *vals = poly_arena_alloc(ctx->arena, arg.int_tuple.n * sizeof(int64_t), _Alignof(int64_t));
    memcpy(vals, arg.int_tuple.vals, arg.int_tuple.n * sizeof(int64_t));
    u->arg.int_tuple.vals = vals;
  } else if (arg.kind == POLY_ARG_PAIR_TUPLE && arg.pair_tuple.n > 0) {
    int64_t (*pairs)[2] = poly_arena_alloc(ctx->arena, arg.pair_tuple.n * 2 * sizeof(int64_t), _Alignof(int64_t));
    memcpy(pairs, arg.pair_tuple.pairs, arg.pair_tuple.n * 2 * sizeof(int64_t));
    u->arg.pair_tuple.pairs = pairs;
  } else if (arg.kind == POLY_ARG_REDUCE_AXIS && arg.reduce_axis.n > 0) {
    int64_t *axes = poly_arena_alloc(ctx->arena, arg.reduce_axis.n * sizeof(int64_t), _Alignof(int64_t));
    memcpy(axes, arg.reduce_axis.axes, arg.reduce_axis.n * sizeof(int64_t));
    u->arg.reduce_axis.axes = axes;
  } else if (arg.kind == POLY_ARG_RANGE && arg.range.n_extra > 0) {
    int64_t *extra = poly_arena_alloc(ctx->arena, (size_t)arg.range.n_extra * sizeof(int64_t), _Alignof(int64_t));
    memcpy(extra, arg.range.extra, (size_t)arg.range.n_extra * sizeof(int64_t));
    u->arg.range.extra = extra;
  } else if (arg.kind == POLY_ARG_STRING && arg.str) {
    size_t len = strlen(arg.str);
    char *s = poly_arena_alloc(ctx->arena, len + 1, 1);
    memcpy(s, arg.str, len + 1);
    u->arg.str = s;
  } else if (arg.kind == POLY_ARG_DEFINE_VAR && arg.define_var.name) {
    size_t len = strlen(arg.define_var.name);
    char *s = poly_arena_alloc(ctx->arena, len + 1, 1);
    memcpy(s, arg.define_var.name, len + 1);
    u->arg.define_var.name = s;
  }

  /* Also store the CSE key in the arena so it persists for hash map lookups */
  CseKey *stored_key = poly_arena_alloc(ctx->arena, sizeof(CseKey), _Alignof(CseKey));
  *stored_key = (CseKey){ op, dtype, u->src, (uint16_t)n_src, u->arg, 0 };

  poly_map_set(ctx->cse, h, stored_key, u, cse_eq);
  return u;
}

PolyUOp *poly_uop0(PolyCtx *ctx, PolyOps op, PolyDType dtype, PolyArg arg) {
  return poly_uop(ctx, op, dtype, NULL, 0, arg);
}

PolyUOp *poly_uop1(PolyCtx *ctx, PolyOps op, PolyDType dtype, PolyUOp *s0, PolyArg arg) {
  PolyUOp *src[] = { s0 };
  return poly_uop(ctx, op, dtype, src, 1, arg);
}

PolyUOp *poly_uop2(PolyCtx *ctx, PolyOps op, PolyDType dtype, PolyUOp *s0, PolyUOp *s1, PolyArg arg) {
  PolyUOp *src[] = { s0, s1 };
  return poly_uop(ctx, op, dtype, src, 2, arg);
}

PolyUOp *poly_uop3(PolyCtx *ctx, PolyOps op, PolyDType dtype,
                   PolyUOp *s0, PolyUOp *s1, PolyUOp *s2, PolyArg arg) {
  PolyUOp *src[] = { s0, s1, s2 };
  return poly_uop(ctx, op, dtype, src, 3, arg);
}

/* ── Toposort (iterative DFS, mirrors tinygrad's toposort) ────────────── */

static bool ptr_eq(const void *a, const void *b) { return a == b; }

static uint32_t ptr_hash(const void *p) {
  uintptr_t v = (uintptr_t)p;
  return (uint32_t)(v ^ (v >> 16) ^ (sizeof(v) > 4 ? (uint32_t)(v >> 32) : 0));
}

PolyUOp **poly_toposort(PolyCtx *ctx, PolyUOp *root, int *n_out) {
  int cap = 256;
  int n = 0;
  PolyUOp **result = poly_arena_alloc(ctx->arena, cap * sizeof(PolyUOp*), _Alignof(PolyUOp*));

  /* Visited set — pointer identity map */
  PolyMap *visited = poly_map_new(256);

  /* DFS stack: pairs of (UOp*, state) where state 0=first visit, 1=children pushed */
  int stack_cap = 256;
  int stack_top = 0;
  PolyUOp **stack = malloc(stack_cap * sizeof(PolyUOp*));
  int *state = malloc(stack_cap * sizeof(int));

  stack[stack_top] = root;
  state[stack_top] = 0;
  stack_top++;

  while (stack_top > 0) {
    PolyUOp *u = stack[stack_top - 1];
    int s = state[stack_top - 1];

    uint32_t vh = ptr_hash(u);
    if (s == 0 && poly_map_get(visited, vh, u, ptr_eq) != NULL) {
      stack_top--;
      continue;
    }

    if (s == 0) {
      /* First visit: push children in reverse order */
      state[stack_top - 1] = 1;
      for (int i = u->n_src - 1; i >= 0; i--) {
        uint32_t ch = ptr_hash(u->src[i]);
        if (poly_map_get(visited, ch, u->src[i], ptr_eq) != NULL) continue;
        if (stack_top >= stack_cap) {
          stack_cap *= 2;
          stack = realloc(stack, stack_cap * sizeof(PolyUOp*));
          state = realloc(state, stack_cap * sizeof(int));
        }
        stack[stack_top] = u->src[i];
        state[stack_top] = 0;
        stack_top++;
      }
    } else {
      /* Post-order: all children done, emit this node */
      stack_top--;
      if (poly_map_get(visited, vh, u, ptr_eq) != NULL) continue;
      /* Use a non-NULL sentinel as value */
      poly_map_set(visited, vh, u, (void*)(uintptr_t)1, ptr_eq);

      if (n >= cap) {
        int new_cap = cap * 2;
        PolyUOp **new_result = poly_arena_alloc(ctx->arena, new_cap * sizeof(PolyUOp*), _Alignof(PolyUOp*));
        memcpy(new_result, result, n * sizeof(PolyUOp*));
        result = new_result;
        cap = new_cap;
      }
      result[n++] = u;
    }
  }

  free(stack);
  free(state);
  poly_map_destroy(visited);
  *n_out = n;
  return result;
}

/* ── Pretty-print ─────────────────────────────────────────────────────── */

static void uop_print_one(PolyUOp *u, char *buf, int *pos, int cap) {
  int written = snprintf(buf + *pos, cap - *pos, "UOp(%s", poly_op_name(u->op));
  if (written > 0) *pos += written;

  /* dtype */
  if (!poly_dtype_eq(u->dtype, POLY_VOID)) {
    written = snprintf(buf + *pos, cap - *pos, ", %s", poly_dtype_name(u->dtype));
    if (written > 0) *pos += written;
  }

  /* arg */
  switch (u->arg.kind) {
    case POLY_ARG_NONE: break;
    case POLY_ARG_INT:
      written = snprintf(buf + *pos, cap - *pos, ", %ld", (long)u->arg.i);
      if (written > 0) *pos += written;
      break;
    case POLY_ARG_FLOAT:
      written = snprintf(buf + *pos, cap - *pos, ", %g", u->arg.f);
      if (written > 0) *pos += written;
      break;
    case POLY_ARG_BOOL:
      written = snprintf(buf + *pos, cap - *pos, ", %s", u->arg.b ? "True" : "False");
      if (written > 0) *pos += written;
      break;
    case POLY_ARG_STRING:
      written = snprintf(buf + *pos, cap - *pos, ", \"%s\"", u->arg.str ? u->arg.str : "");
      if (written > 0) *pos += written;
      break;
    case POLY_ARG_OPS:
      written = snprintf(buf + *pos, cap - *pos, ", %s", poly_op_name(u->arg.ops));
      if (written > 0) *pos += written;
      break;
    case POLY_ARG_INVALID:
      written = snprintf(buf + *pos, cap - *pos, ", Invalid");
      if (written > 0) *pos += written;
      break;
    case POLY_ARG_INT_TUPLE:
      written = snprintf(buf + *pos, cap - *pos, ", (");
      if (written > 0) *pos += written;
      for (int i = 0; i < u->arg.int_tuple.n; i++) {
        written = snprintf(buf + *pos, cap - *pos, "%s%ld", i ? "," : "", (long)u->arg.int_tuple.vals[i]);
        if (written > 0) *pos += written;
      }
      written = snprintf(buf + *pos, cap - *pos, ")");
      if (written > 0) *pos += written;
      break;
    case POLY_ARG_RANGE:
      written = snprintf(buf + *pos, cap - *pos, ", (%ld,%d",
                         (long)u->arg.range.axis_id, (int)u->arg.range.axis_type);
      if (written > 0) *pos += written;
      for (int i = 0; i < u->arg.range.n_extra; i++) {
        written = snprintf(buf + *pos, cap - *pos, ",%ld", (long)u->arg.range.extra[i]);
        if (written > 0) *pos += written;
      }
      written = snprintf(buf + *pos, cap - *pos, ")");
      if (written > 0) *pos += written;
      break;
    case POLY_ARG_DEFINE_VAR:
      written = snprintf(buf + *pos, cap - *pos, ", (\"%s\",%ld,%ld)",
                         u->arg.define_var.name ? u->arg.define_var.name : "?",
                         (long)u->arg.define_var.min_val, (long)u->arg.define_var.max_val);
      if (written > 0) *pos += written;
      break;
    default: break;
  }

  /* n_src */
  if (u->n_src > 0) {
    written = snprintf(buf + *pos, cap - *pos, ", src=%d", u->n_src);
    if (written > 0) *pos += written;
  }

  written = snprintf(buf + *pos, cap - *pos, ")");
  if (written > 0) *pos += written;
}

char *poly_uop_str(PolyUOp *u) {
  int cap = 256;
  char *buf = malloc(cap);
  int pos = 0;
  uop_print_one(u, buf, &pos, cap);
  buf[pos] = '\0';
  return buf;
}

char *poly_graph_str(PolyUOp *root) {
  /* Simple: just print the root node for now */
  /* A full graph print would need toposort, but that needs a ctx */
  return poly_uop_str(root);
}
