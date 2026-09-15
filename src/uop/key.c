/* Tinygrad UOp.key: op/dtype/arg and ordered child identities, never tags.
 * C uses an exact canonical DAG encoding instead of Python repr + SHA256.
 * This is a cache key, not an interchange format; there is no decoder.
 * Hashes only index maps/cache files: full bytes decide equality. */
#include "ops.h"
#include "../ctx.h"
#include "../utils.h"
#include <limits.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  uint8_t *data;
  size_t size, capacity;
} KeyBytes;

static bool key_bytes(KeyBytes *b, const void *data, size_t size) {
  if ((size && !data) || size > SIZE_MAX - b->size) return false;
  size_t required = b->size + size;
  if (required > b->capacity) {
    size_t capacity = b->capacity ? b->capacity : 128;
    while (capacity < required) {
      if (capacity > SIZE_MAX / 2) {
        capacity = required;
        break;
      }
      capacity *= 2;
    }
    uint8_t *next = realloc(b->data, capacity);
    if (!next) return false;
    b->data = next;
    b->capacity = capacity;
  }
  if (size) memcpy(b->data + b->size, data, size);
  b->size = required;
  return true;
}

static bool key_int(KeyBytes *b, uint64_t value) {
  uint8_t bytes[8];
  for (int i = 0; i < 8; i++)
    bytes[i] = (uint8_t)(value >> (8 * i));
  return key_bytes(b, bytes, sizeof(bytes));
}

static bool key_string(KeyBytes *b, const char *value) {
  size_t size = value ? strlen(value) : 0;
  return key_int(b, value ? size : UINT64_MAX) && key_bytes(b, value, size);
}

static bool key_dtype(KeyBytes *b, PolyDType d) {
  return key_int(b, d.priority) && key_int(b, d.bitsize) && key_int(b, (unsigned char)d.fmt) &&
         key_string(b, d.name);
}

static bool key_strings(KeyBytes *b, const char *const *values, int n) {
  if (n < 0 || (n && !values) || !key_int(b, n)) return false;
  for (int i = 0; i < n; i++)
    if (!key_string(b, values[i])) return false;
  return true;
}

static bool key_ints(KeyBytes *b, const int *values, int n) {
  if (n < 0 || (n && !values) || !key_int(b, n)) return false;
  for (int i = 0; i < n; i++)
    if (!key_int(b, values[i])) return false;
  return true;
}

static bool key_int64s(KeyBytes *b, const int64_t *values, int n) {
  if (n < 0 || (n && !values) || !key_int(b, n)) return false;
  for (int i = 0; i < n; i++)
    if (!key_int(b, values[i])) return false;
  return true;
}

typedef struct KeyRow {
  KeyBytes bytes;
  uint64_t id;
  struct KeyRow *next;
} KeyRow;

static bool key_equal(const void *a, const void *b) {
  const KeyRow *x = a, *y = b;
  return x->bytes.size == y->bytes.size && !memcmp(x->bytes.data, y->bytes.data, x->bytes.size);
}

static uint32_t key_hash(const KeyBytes *b) {
  uint32_t h = UINT32_C(2166136261);
  for (size_t i = 0; i < b->size; i++)
    h = (h ^ b->data[i]) * UINT32_C(16777619);
  return h;
}

static bool key_ref(KeyBytes *b, PolyMap *nodes, const PolyUOp *u) {
  if (!u) return key_int(b, 0);
  KeyRow *row = poly_map_get(nodes, poly_ptr_hash(u), u, poly_ptr_eq);
  return row && row != (void *)(uintptr_t)1 && key_int(b, row->id);
}

static bool key_opts(KeyBytes *b, const PolyOpt *opts, int n) {
  if (n < 0 || (n && !opts) || !key_int(b, n)) return false;
  for (int i = 0; i < n; i++) {
    PolyOpt o = opts[i];
    if (!key_int(b, o.op) || !key_int(b, o.has_axis) || (o.has_axis && !key_int(b, o.axis)) ||
        !key_int(b, o.arg_kind))
      return false;
    if (o.arg_kind == POLY_OPT_ARG_INT) {
      if (!key_int(b, o.arg)) return false;
    } else if (o.arg_kind == POLY_OPT_ARG_INT_TUPLE) {
      if (!key_int64s(b, o.arg_tuple, o.n_arg_tuple)) return false;
    } else if (o.arg_kind != POLY_OPT_ARG_NONE)
      return false;
  }
  return true;
}

static bool key_arg(KeyBytes *b, PolyMap *nodes, PolyArg a) {
  if (!key_int(b, a.kind)) return false;
  switch (a.kind) {
  case POLY_ARG_NONE:
  case POLY_ARG_INVALID:
    return true;
  case POLY_ARG_INT:
    return key_int(b, a.i);
  case POLY_ARG_BOOL:
    return key_int(b, a.b);
  case POLY_ARG_FLOAT: {
    uint64_t bits;
    memcpy(&bits, &a.f, sizeof(bits));
    return key_int(b, bits);
  }
  case POLY_ARG_BIGINT:
    if ((a.bigint.n_limbs && !a.bigint.limbs) || !key_int(b, a.bigint.sign) ||
        !key_int(b, a.bigint.n_limbs))
      return false;
    for (uint32_t i = 0; i < a.bigint.n_limbs; i++)
      if (!key_int(b, a.bigint.limbs[i])) return false;
    return true;
  case POLY_ARG_STRING:
    return key_string(b, a.str);
  case POLY_ARG_STRING_TUPLE:
    return key_strings(b, a.string_tuple.vals, a.string_tuple.n);
  case POLY_ARG_INT_TUPLE:
    return key_int64s(b, a.int_tuple.vals, a.int_tuple.n);
  case POLY_ARG_OPS:
    return key_int(b, a.ops);
  case POLY_ARG_DTYPE:
    return key_dtype(b, a.dtype);
  case POLY_ARG_REDUCE:
    return key_int(b, a.reduce.op) && key_int(b, a.reduce.num_axes);
  case POLY_ARG_RANGE:
    return key_int(b, a.range.axis_id) && key_int(b, a.range.axis_type) &&
           key_int64s(b, a.range.extra, a.range.n_extra);
  case POLY_ARG_ALLREDUCE:
    return key_int(b, a.allreduce.op) && key_int(b, a.allreduce.device_is_tuple) &&
           (a.allreduce.device_is_tuple ? key_strings(b, a.allreduce.devices, a.allreduce.n_devices)
                                        : key_string(b, a.allreduce.device));
  case POLY_ARG_BUFFERIZE_OPTS:
    return key_int(b, a.bufferize_opts.addrspace) && key_int(b, a.bufferize_opts.removable) &&
           key_int(b, a.bufferize_opts.device_is_int) &&
           key_int(b, a.bufferize_opts.device_is_tuple) &&
           (a.bufferize_opts.device_is_int ? key_int(b, a.bufferize_opts.device_int)
            : a.bufferize_opts.device_is_tuple
                ? key_strings(b, a.bufferize_opts.devices, a.bufferize_opts.n_devices)
                : key_string(b, a.bufferize_opts.device));
  case POLY_ARG_TENSOR_CORE:
    if (!key_ints(b, a.tensor_core.dims, 3) || !key_dtype(b, a.tensor_core.dtype_in) ||
        !key_string(b, a.tensor_core.device) || !key_int(b, a.tensor_core.threads) ||
        !key_int(b, a.tensor_core.has_upcast_axes))
      return false;
    for (int d = 0; d < 3; d++) {
      int n = a.tensor_core.n_upcast_axes[d];
      if (n < 0 || (n && !a.tensor_core.upcast_axes[d]) || !key_int(b, n)) return false;
      for (int i = 0; i < n; i++)
        if (!key_int(b, a.tensor_core.upcast_axes[d][i][0]) ||
            !key_int(b, a.tensor_core.upcast_axes[d][i][1]))
          return false;
    }
    return true;
  case POLY_ARG_PARAM: {
    const PolyParamArg *p = a.param;
    if (p && p->has_minmax &&
        ((p->min_val.kind != POLY_ARG_INT && p->min_val.kind != POLY_ARG_FLOAT &&
          p->min_val.kind != POLY_ARG_BOOL && p->min_val.kind != POLY_ARG_BIGINT) ||
         (p->max_val.kind != POLY_ARG_INT && p->max_val.kind != POLY_ARG_FLOAT &&
          p->max_val.kind != POLY_ARG_BOOL && p->max_val.kind != POLY_ARG_BIGINT)))
      return false;
    return p && key_int(b, p->slot) && key_dtype(b, p->dtype) && key_string(b, p->name) &&
           key_int(b, p->addrspace) && key_int(b, p->has_axis) &&
           (!p->has_axis || key_int(b, p->axis)) && key_int(b, p->volatile_) &&
           key_int(b, p->device_is_tuple) &&
           (p->device_is_tuple ? key_strings(b, p->devices, p->n_devices) : key_string(b, p->device)
           ) &&
           key_int(b, p->has_minmax) &&
           (!p->has_minmax || (key_arg(b, nodes, p->min_val) && key_arg(b, nodes, p->max_val))) &&
           key_int(b, p->has_multiple_of) && (!p->has_multiple_of || key_int(b, p->multiple_of));
  }
  case POLY_ARG_CALL_INFO: {
    const PolyCallInfo *c = a.call_info;
    /* Registry keys/host payloads are process-local, not persistent content. */
    return c && !c->has_grad_fxn && !c->has_aux && key_string(b, c->name) &&
           key_int(b, c->precompile) && key_int(b, c->precompile_backward);
  }
  case POLY_ARG_BYTES:
    return a.bytes.n >= 0 && key_int(b, a.bytes.n) && key_bytes(b, a.bytes.data, a.bytes.n);
  case POLY_ARG_KERNEL_INFO: {
    const PolyKernelInfo *k = a.kernel_info;
    if (!k || k->n_axis_types < 0 || (k->n_axis_types && !k->axis_types) ||
        !key_string(b, k->name) || !key_int(b, k->n_axis_types))
      return false;
    for (int i = 0; i < k->n_axis_types; i++)
      if (!key_int(b, k->axis_types[i])) return false;
    if (!key_int(b, k->dont_use_locals) || !key_int(b, k->beam) ||
        !key_opts(b, k->applied_opts, k->n_applied_opts) || !key_int(b, k->has_opts_to_apply) ||
        (k->has_opts_to_apply && !key_opts(b, k->opts_to_apply, k->n_opts_to_apply)) ||
        !key_int(b, k->estimates != NULL))
      return false;
    return !k->estimates ||
           (key_ref(b, nodes, k->estimates->ops) && key_ref(b, nodes, k->estimates->lds) &&
            key_ref(b, nodes, k->estimates->mem));
  }
  case POLY_ARG_PROGRAM_INFO: {
    const PolyProgramInfo *p = a.program_info;
    if (!p || p->n_vars < 0 || (p->n_vars && !p->vars) || !key_string(b, p->name) ||
        !key_string(b, p->target) || !key_ints(b, p->global_size, 3) ||
        !key_ints(b, p->local_size, 3) || !key_int(b, p->has_local_size))
      return false;
    for (int i = 0; i < 3; i++)
      if (!key_ref(b, nodes, p->global_exprs[i]) || !key_ref(b, nodes, p->local_exprs[i]))
        return false;
    if (!key_int(b, p->n_vars)) return false;
    for (int i = 0; i < p->n_vars; i++)
      if (!key_ref(b, nodes, p->vars[i])) return false;
    return key_ints(b, p->globals, p->n_globals) && key_ints(b, p->outs, p->n_outs) &&
           key_ints(b, p->ins, p->n_ins);
  }
  }
  return false;
}

/* Metadata UOp references are value fields too, not addresses to hash. Walk
 * them after ordinary sources, keeping the traversal independent of allocation
 * order and avoiding recursion on deep graphs. */
static int key_n_edges(const PolyUOp *u) {
  if (u->arg.kind == POLY_ARG_PROGRAM_INFO && u->arg.program_info) {
    int n = u->arg.program_info->n_vars;
    return n < 0 || n > INT_MAX - u->n_src - 6 ? -1 : u->n_src + n + 6;
  }
  return u->n_src +
         (u->arg.kind == POLY_ARG_KERNEL_INFO && u->arg.kernel_info && u->arg.kernel_info->estimates
              ? 3
              : 0);
}

static PolyUOp *key_edge(const PolyUOp *u, int i) {
  if (i < u->n_src) return u->src[i];
  i -= u->n_src;
  if (u->arg.kind == POLY_ARG_PROGRAM_INFO) {
    const PolyProgramInfo *p = u->arg.program_info;
    if (i < 3) return p->global_exprs[i];
    if (i < 6) return p->local_exprs[i - 3];
    return p->vars ? p->vars[i - 6] : NULL;
  }
  const PolyEstimates *e = u->arg.kernel_info->estimates;
  return i == 0 ? e->ops : i == 1 ? e->lds : e->mem;
}

uint8_t *poly_uop_key(PolyCtx *ctx, PolyUOp *root, size_t *size) {
  if (!size) return NULL;
  *size = 0;
  if (!ctx || !root || !poly_ctx_owns_ptr(ctx, root)) return NULL;
  typedef struct {
    PolyUOp *u;
    int next;
  } Frame;
  size_t depth = 1, capacity = 64;
  Frame *stack = malloc(capacity * sizeof(*stack));
  PolyMap *nodes = poly_map_new(128), *contents = poly_map_new(128);
  KeyRow *rows = NULL;
  uint64_t next_id = 1;
  KeyBytes out = {0};
  bool ok = false;
  if (!stack || !nodes || !contents || !key_int(&out, 1)) goto done;
  stack[0] = (Frame){root, 0};
  poly_map_set(nodes, poly_ptr_hash(root), root, (void *)(uintptr_t)1, poly_ptr_eq);
  while (depth) {
    Frame *frame = &stack[depth - 1];
    PolyUOp *u = frame->u;
    int n_edges = key_n_edges(u);
    if (n_edges < 0) goto done;
    if (frame->next < n_edges) {
      PolyUOp *child = key_edge(u, frame->next++);
      if (!child) continue;
      if (!poly_ctx_owns_ptr(ctx, child)) goto done;
      void *found = poly_map_get(nodes, poly_ptr_hash(child), child, poly_ptr_eq);
      if (found == (void *)(uintptr_t)1) goto done; /* cyclic metadata */
      if (found) continue;
      if (depth == capacity) {
        if (capacity > SIZE_MAX / 2 / sizeof(*stack)) goto done;
        size_t grown = capacity * 2;
        Frame *next = realloc(stack, grown * sizeof(*stack));
        if (!next) goto done;
        stack = next;
        capacity = grown;
      }
      stack[depth++] = (Frame){child, 0};
      poly_map_set(nodes, poly_ptr_hash(child), child, (void *)(uintptr_t)1, poly_ptr_eq);
      continue;
    }
    KeyRow candidate = {0};
    bool valid = key_int(&candidate.bytes, u->op) && key_dtype(&candidate.bytes, u->dtype) &&
                 key_arg(&candidate.bytes, nodes, u->arg) && key_int(&candidate.bytes, u->n_src);
    for (int i = 0; valid && i < u->n_src; i++)
      valid = key_ref(&candidate.bytes, nodes, u->src[i]);
    if (!valid) {
      free(candidate.bytes.data);
      goto done;
    }
    uint32_t hash = key_hash(&candidate.bytes);
    KeyRow *row = poly_map_get(contents, hash, &candidate, key_equal);
    if (row)
      free(candidate.bytes.data);
    else {
      row = malloc(sizeof(*row));
      if (!row) {
        free(candidate.bytes.data);
        goto done;
      }
      *row = (KeyRow){candidate.bytes, next_id++, rows};
      rows = row;
      if (!key_int(&out, row->bytes.size) || !key_bytes(&out, row->bytes.data, row->bytes.size))
        goto done;
      poly_map_set(contents, hash, row, row, key_equal);
    }
    poly_map_set(nodes, poly_ptr_hash(u), u, row, poly_ptr_eq);
    depth--;
  }
  ok = key_ref(&out, nodes, root);
done:
  while (rows) {
    KeyRow *next = rows->next;
    free(rows->bytes.data);
    free(rows);
    rows = next;
  }
  if (nodes) poly_map_destroy(nodes);
  if (contents) poly_map_destroy(contents);
  free(stack);
  if (!ok) {
    free(out.data);
    return NULL;
  }
  *size = out.size;
  return out.data;
}
