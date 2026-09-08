/*
 * poly_ir.c -- Binary IR codec for tensor-level UOp graphs
 *
 * poly.ir.uops@16 format:
 *   Header (32 bytes)
 *   String table (variable)
 *   Node table (variable, strict toposort order; scalar dtype ID and UOp shape)
 *   Interface table (named logical nodes with roles)
 *   Entrypoint table (named SINKs plus ABI metadata)
 *   Module table (exact named logical placement boundaries)
 */

#define _POSIX_C_SOURCE 200809L
#include "ir.h"
#include "ctx.h"
#include "frontend.h"
#include "engine/schedule.h"
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Byte helpers */

#ifdef POLY_TESTING
static int ir_topo_fail_after = -1;
void poly_test_ir_topo_fail_after(int count) {
  ir_topo_fail_after = count;
}
#endif

static PolyUOp **ir_toposort(PolyCtx *ctx, PolyUOp *root, int *count, bool scratch) {
#ifdef POLY_TESTING
  if (ir_topo_fail_after == 0) {
    ir_topo_fail_after = -1;
    *count = 0;
    return NULL;
  }
  if (ir_topo_fail_after > 0) ir_topo_fail_after--;
#endif
  return scratch ? poly_toposort_scratch(ctx, root, count) : poly_toposort_alloc(ctx, root, count);
}

typedef struct {
  uint8_t *data;
  int len;
  int cap;
} ByteBuf;

static void bb_init(ByteBuf *b) {
  b->cap = 4096;
  b->data = malloc(b->cap);
  b->len = 0;
}

static void bb_ensure(ByteBuf *b, int need) {
  while (b->len + need > b->cap) {
    b->cap *= 2;
    b->data = realloc(b->data, b->cap);
  }
}

static void bb_u8(ByteBuf *b, uint8_t v) {
  bb_ensure(b, 1);
  b->data[b->len++] = v;
}

static void bb_u16(ByteBuf *b, uint16_t v) {
  bb_ensure(b, 2);
  b->data[b->len++] = v & 0xFF;
  b->data[b->len++] = (v >> 8) & 0xFF;
}

static void bb_u32(ByteBuf *b, uint32_t v) {
  bb_ensure(b, 4);
  for (int i = 0; i < 4; i++)
    b->data[b->len++] = (v >> (i * 8)) & 0xFF;
}

static void bb_i32(ByteBuf *b, int32_t v) {
  bb_u32(b, (uint32_t)v);
}

static void bb_i64(ByteBuf *b, int64_t v) {
  bb_ensure(b, 8);
  uint64_t u = (uint64_t)v;
  for (int i = 0; i < 8; i++)
    b->data[b->len++] = (u >> (i * 8)) & 0xFF;
}

static void bb_f64(ByteBuf *b, double v) {
  uint64_t u;
  memcpy(&u, &v, sizeof(u));
  bb_i64(b, (int64_t)u);
}

static void bb_bytes(ByteBuf *b, const uint8_t *src, int n) {
  bb_ensure(b, n);
  memcpy(b->data + b->len, src, n);
  b->len += n;
}

/* Read helpers */

typedef struct {
  const uint8_t *data;
  int len;
  int pos;
} ByteReader;

static int br_remaining(ByteReader *r) {
  return r->len - r->pos;
}

static uint8_t br_u8(ByteReader *r) {
  if (r->pos >= r->len) return 0;
  return r->data[r->pos++];
}

static uint16_t br_u16(ByteReader *r) {
  if (r->pos + 2 > r->len) return 0;
  uint16_t v = (uint16_t)r->data[r->pos] | ((uint16_t)r->data[r->pos + 1] << 8);
  r->pos += 2;
  return v;
}

static uint32_t br_u32(ByteReader *r) {
  if (r->pos + 4 > r->len) return 0;
  uint32_t v = 0;
  for (int i = 0; i < 4; i++)
    v |= (uint32_t)r->data[r->pos++] << (i * 8);
  return v;
}

static int32_t br_i32(ByteReader *r) {
  return (int32_t)br_u32(r);
}

static int64_t br_i64(ByteReader *r) {
  if (r->pos + 8 > r->len) return 0;
  uint64_t v = 0;
  for (int i = 0; i < 8; i++)
    v |= (uint64_t)r->data[r->pos++] << (i * 8);
  return (int64_t)v;
}

static double br_f64(ByteReader *r) {
  int64_t i = br_i64(r);
  double d;
  memcpy(&d, &i, sizeof(d));
  return d;
}

/* Current Tinygrad codegen/opt/__init__.py:Opt wire value. */
static void bb_opt(ByteBuf *buf, const PolyOpt *opt) {
  bb_u8(buf, (uint8_t)opt->op);
  bb_u8(buf, opt->has_axis ? 1 : 0);
  bb_i32(buf, opt->axis);
  bb_u8(buf, (uint8_t)opt->arg_kind);
  if (opt->arg_kind == POLY_OPT_ARG_INT) {
    bb_i64(buf, opt->arg);
  } else if (opt->arg_kind == POLY_OPT_ARG_INT_TUPLE) {
    bb_u16(buf, (uint16_t)opt->n_arg_tuple);
    for (int i = 0; i < opt->n_arg_tuple; i++)
      bb_i64(buf, opt->arg_tuple[i]);
  }
}

static bool br_opt(ByteReader *r, PolyOpt *opt) {
  if (!r || !opt || br_remaining(r) < 7) return false;
  memset(opt, 0, sizeof(*opt));
  opt->op = (PolyOptOps)br_u8(r);
  opt->has_axis = br_u8(r) != 0;
  opt->axis = br_i32(r);
  opt->arg_kind = (PolyOptArgKind)br_u8(r);
  if (opt->op < POLY_OPT_TC || opt->op > POLY_OPT_SWAP || opt->arg_kind < POLY_OPT_ARG_NONE ||
      opt->arg_kind > POLY_OPT_ARG_INT_TUPLE)
    return false;
  if (opt->arg_kind == POLY_OPT_ARG_INT) {
    if (br_remaining(r) < 8) return false;
    opt->arg = br_i64(r);
  } else if (opt->arg_kind == POLY_OPT_ARG_INT_TUPLE) {
    if (br_remaining(r) < 2) return false;
    uint16_t n = br_u16(r);
    if (br_remaining(r) < (int64_t)n * 8) return false;
    int64_t *tuple = n > 0 ? malloc((size_t)n * sizeof(*tuple)) : NULL;
    if (n > 0 && !tuple) return false;
    for (int i = 0; i < n; i++)
      tuple[i] = br_i64(r);
    opt->arg_tuple = tuple;
    opt->n_arg_tuple = n;
  }
  return true;
}

static void free_opts(PolyOpt *opts, int n) {
  if (!opts) return;
  for (int i = 0; i < n; i++)
    if (opts[i].arg_kind == POLY_OPT_ARG_INT_TUPLE) free((void *)opts[i].arg_tuple);
  free(opts);
}

static bool br_prior_node_ref(
    ByteReader *r,
    PolyUOp **nodes,
    uint32_t current,
    bool nullable,
    PolyUOp **out
) {
  if (!out || br_remaining(r) < 4) return false;
  uint32_t idx = br_u32(r);
  if (nullable && idx == UINT32_MAX) {
    *out = NULL;
    return true;
  }
  if (idx >= current || !nodes[idx]) return false;
  *out = nodes[idx];
  return true;
}

/* Magic */

#define IR_MAGIC 0x52494750 /* "PGIR" LE */
#define PROGRAM_MAGIC 0x4d504750 /* "PGPM" LE */

/* Dtype index table */

#define N_DTYPES 20

static const PolyDType *dtype_table[N_DTYPES] = {
    &POLY_VOID,      &POLY_BOOL,     &POLY_INT8,    &POLY_UINT8,       &POLY_INT16,
    &POLY_UINT16,    &POLY_INT32,    &POLY_UINT32,  &POLY_INT64,       &POLY_UINT64,
    &POLY_FLOAT16,   &POLY_BFLOAT16, &POLY_FLOAT32, &POLY_FLOAT64,     &POLY_WEAKINT,
    &POLY_WEAKFLOAT, &POLY_FP8E4M3,  &POLY_FP8E5M2, &POLY_FP8E4M3FNUZ, &POLY_FP8E5M2FNUZ,
};

static int dtype_to_index(PolyDType dt) {
  dt = dt;
  for (int i = 0; i < N_DTYPES; i++)
    if (poly_dtype_eq(dt, *dtype_table[i])) return i;
  return -1;
}

/* String table builder */

typedef struct {
  char **strs;
  int n;
  int cap;
} StringTable;

static void st_init(StringTable *st) {
  st->cap = 64;
  st->strs = calloc(st->cap, sizeof(char *));
  st->n = 0;
}

static uint32_t st_add(StringTable *st, const char *s) {
  /* Linear search for dedup (small N expected) */
  for (int i = 0; i < st->n; i++)
    if (strcmp(st->strs[i], s) == 0) return (uint32_t)i;
  if (st->n >= st->cap) {
    st->cap *= 2;
    st->strs = realloc(st->strs, st->cap * sizeof(char *));
  }
  st->strs[st->n] = strdup(s);
  return (uint32_t)st->n++;
}

static void st_free(StringTable *st) {
  for (int i = 0; i < st->n; i++)
    free(st->strs[i]);
  free(st->strs);
}

/* Final backend PROGRAMs use tag_arg only for immutable scalar/register/label
 * metadata.  Preserve that metadata exactly in PGPM; reject pass-private or
 * structural tag payloads rather than silently dropping them. */
static bool program_tag_arg_valid(PolyArg arg) {
  switch (arg.kind) {
  case POLY_ARG_NONE:
  case POLY_ARG_INT:
  case POLY_ARG_FLOAT:
  case POLY_ARG_BOOL:
    return true;
  case POLY_ARG_INT_TUPLE:
    return arg.int_tuple.n >= 0 && arg.int_tuple.n <= UINT16_MAX &&
           (arg.int_tuple.n == 0 || arg.int_tuple.vals);
  case POLY_ARG_STRING:
    return arg.str != NULL;
  default:
    return false;
  }
}

static void program_tag_arg_collect_strings(StringTable *strings, PolyArg arg) {
  if (arg.kind == POLY_ARG_STRING && arg.str) st_add(strings, arg.str);
}

static void bb_program_tag_arg(ByteBuf *buf, StringTable *strings, PolyArg arg) {
  switch (arg.kind) {
  case POLY_ARG_NONE:
    break;
  case POLY_ARG_INT:
    bb_i64(buf, arg.i);
    break;
  case POLY_ARG_FLOAT:
    bb_f64(buf, arg.f);
    break;
  case POLY_ARG_BOOL:
    bb_u8(buf, arg.b ? 1 : 0);
    break;
  case POLY_ARG_INT_TUPLE:
    bb_u16(buf, (uint16_t)arg.int_tuple.n);
    for (int i = 0; i < arg.int_tuple.n; i++)
      bb_i64(buf, arg.int_tuple.vals[i]);
    break;
  case POLY_ARG_STRING:
    bb_u32(buf, st_add(strings, arg.str));
    break;
  default:
    break; /* validated before the output buffer is created */
  }
}

static bool br_program_tag_arg(
    ByteReader *r,
    uint8_t kind,
    char **strings,
    uint32_t n_strings,
    PolyArg *out
) {
  if (!r || !out || kind > POLY_ARG_DTYPE) return false;
  *out = poly_arg_none();
  out->kind = (PolyArgKind)kind;
  switch (out->kind) {
  case POLY_ARG_NONE:
    return true;
  case POLY_ARG_INT:
    if (br_remaining(r) < 8) return false;
    out->i = br_i64(r);
    return true;
  case POLY_ARG_FLOAT:
    if (br_remaining(r) < 8) return false;
    out->f = br_f64(r);
    return true;
  case POLY_ARG_BOOL:
    if (br_remaining(r) < 1) return false;
    out->b = br_u8(r) != 0;
    return true;
  case POLY_ARG_INT_TUPLE: {
    if (br_remaining(r) < 2) return false;
    uint16_t n = br_u16(r);
    if (br_remaining(r) < (int64_t)n * 8) return false;
    out->int_tuple.n = n;
    out->int_tuple.vals = n > 0 ? malloc((size_t)n * sizeof(int64_t)) : NULL;
    if (n > 0 && !out->int_tuple.vals) return false;
    for (int i = 0; i < n; i++)
      out->int_tuple.vals[i] = br_i64(r);
    return true;
  }
  case POLY_ARG_STRING: {
    if (br_remaining(r) < 4) return false;
    uint32_t idx = br_u32(r);
    if (idx >= n_strings || !strings[idx]) return false;
    out->str = strings[idx];
    return true;
  }
  default:
    return false;
  }
}

static void st_add_entrypoint_strings(StringTable *strings, const PolyIrEntrypoint *ep) {
  if (!strings || !ep) return;
  if (ep->name) st_add(strings, ep->name);
  for (int i = 0; i < ep->n_inputs; i++)
    if (ep->inputs && ep->inputs[i]) st_add(strings, ep->inputs[i]);
  for (int i = 0; i < ep->n_outputs; i++)
    if (ep->outputs && ep->outputs[i]) st_add(strings, ep->outputs[i]);
  if (ep->objective) st_add(strings, ep->objective);
}

static void free_ir_entrypoint(PolyIrEntrypoint *ep) {
  if (!ep) return;
  free((char *)ep->name);
  for (int i = 0; i < ep->n_inputs; i++)
    free((char *)ep->inputs[i]);
  free((char **)ep->inputs);
  for (int i = 0; i < ep->n_outputs; i++)
    free((char *)ep->outputs[i]);
  free((char **)ep->outputs);
  free((char *)ep->objective);
}

static void free_ir_module(PolyIrModule *module) {
  if (!module) return;
  free((char *)module->name);
  free(module->inputs);
}

static bool ir_interface_shape_valid(const PolyIrBufEntry *entry) {
  if (!entry || entry->ndim < 0 || entry->ndim > POLY_IR_MAX_DIMS) return false;
  int64_t numel = 1;
  for (int d = 0; d < entry->ndim; d++) {
    int64_t dim = entry->shape[d];
    if (dim < 0 || (dim != 0 && numel > INT64_MAX / dim)) return false;
    numel *= dim;
  }
  return true;
}

static bool node_list_contains(PolyUOp **nodes, int n_nodes, PolyUOp *needle) {
  for (int i = 0; i < n_nodes; i++)
    if (nodes[i] == needle) return true;
  return false;
}

static bool node_list_append_unique(
    PolyUOp ***nodes,
    int *n_nodes,
    int *cap_nodes,
    PolyUOp *value
) {
  if (!value || node_list_contains(*nodes, *n_nodes, value)) return value != NULL;
  if (*n_nodes >= *cap_nodes) {
    if (*cap_nodes > INT_MAX / 2) return false;
    int next = *cap_nodes > 0 ? *cap_nodes * 2 : 64;
    if ((size_t)next > SIZE_MAX / sizeof(PolyUOp *)) return false;
    PolyUOp **grown = realloc(*nodes, (size_t)next * sizeof(*grown));
    if (!grown) return false;
    *nodes = grown;
    *cap_nodes = next;
  }
  (*nodes)[(*n_nodes)++] = value;
  return true;
}

/* ProgramInfo launch metadata and KernelInfo estimates reference UOps outside
 * ordinary src arrays. Make them serialization dependencies. */
static bool program_topo_with_metadata(PolyCtx *ctx, PolyUOp ***topo_io, int *n_topo_io) {
  PolyUOp **old = *topo_io;
  int n_old = *n_topo_io;
  PolyUOp **ordered = NULL;
  int n_ordered = 0, cap_ordered = 0;

  for (int i = 0; i < n_old; i++) {
    PolyUOp *program = old[i];
    if (!program || program->op != POLY_OP_PROGRAM || program->arg.kind != POLY_ARG_PROGRAM_INFO ||
        !program->arg.program_info)
      continue;
    const PolyProgramInfo *info = program->arg.program_info;
    int n_refs = 6 + info->n_vars;
    if (info->n_vars < 0 || n_refs < 0) goto fail;
    PolyUOp **refs = calloc((size_t)(n_refs > 0 ? n_refs : 1), sizeof(*refs));
    if (!refs) goto fail;
    int nr = 0;
    for (int d = 0; d < 3; d++)
      refs[nr++] = info->global_exprs[d];
    for (int d = 0; d < 3; d++)
      refs[nr++] = info->local_exprs[d];
    for (int v = 0; v < info->n_vars; v++)
      refs[nr++] = info->vars[v];
    for (int r = 0; r < nr; r++) {
      if (!refs[r]) continue;
      if (!poly_ctx_owns_ptr(ctx, refs[r])) {
        fprintf(stderr, "poly_program_export: PROGRAM metadata escapes context\n");
        free(refs);
        goto fail;
      }
      int n_dep = 0;
      PolyUOp **dep = ir_toposort(ctx, refs[r], &n_dep, false);
      if (!dep || n_dep <= 0) {
        poly_toposort_free(dep);
        free(refs);
        goto fail;
      }
      for (int j = 0; j < n_dep; j++) {
        if (!node_list_append_unique(&ordered, &n_ordered, &cap_ordered, dep[j])) {
          poly_toposort_free(dep);
          free(refs);
          goto fail;
        }
      }
      poly_toposort_free(dep);
    }
    free(refs);
  }
  for (int i = 0; i < n_old; i++) {
    PolyUOp *sink = old[i];
    if (!sink || sink->op != POLY_OP_SINK || sink->arg.kind != POLY_ARG_KERNEL_INFO ||
        !sink->arg.kernel_info || !sink->arg.kernel_info->estimates)
      continue;
    PolyUOp *refs[3] = {
        sink->arg.kernel_info->estimates->ops,
        sink->arg.kernel_info->estimates->lds,
        sink->arg.kernel_info->estimates->mem,
    };
    for (int r = 0; r < 3; r++) {
      if (!refs[r]) continue;
      if (!poly_ctx_owns_ptr(ctx, refs[r])) goto fail;
      int n_dep = 0;
      PolyUOp **dep = ir_toposort(ctx, refs[r], &n_dep, false);
      if (!dep || n_dep <= 0) {
        poly_toposort_free(dep);
        goto fail;
      }
      for (int j = 0; j < n_dep; j++) {
        if (!node_list_append_unique(&ordered, &n_ordered, &cap_ordered, dep[j])) {
          poly_toposort_free(dep);
          goto fail;
        }
      }
      poly_toposort_free(dep);
    }
  }
  for (int i = 0; i < n_old; i++)
    if (!node_list_append_unique(&ordered, &n_ordered, &cap_ordered, old[i])) goto fail;
  free(old);
  *topo_io = ordered;
  *n_topo_io = n_ordered;
  return true;

fail:
  free(ordered);
  return false;
}

/* Export */

static uint8_t *poly_graph_export(const PolyIrSpec *spec, int *out_len, bool executable) {
  if (!out_len) return NULL;
  *out_len = 0;
  if (!spec || !spec->ctx) return NULL;

  if (spec->n_entrypoints <= 0 || !spec->entrypoints) {
    fprintf(stderr, "%s: no entrypoints\n", executable ? "poly_program_export" : "poly_ir_export");
    return NULL;
  }
  for (int i = 0; i < spec->n_entrypoints; i++) {
    if (!spec->entrypoints[i].name || !spec->entrypoints[i].sink ||
        (executable && spec->entrypoints[i].sink->op != POLY_OP_LINEAR)) {
      fprintf(
          stderr, "%s: invalid entrypoint %d\n",
          executable ? "poly_program_export" : "poly_ir_export", i
      );
      return NULL;
    }
  }
  if (spec->n_bufs < 0 || (spec->n_bufs > 0 && !spec->bufs)) return NULL;
  for (int i = 0; i < spec->n_bufs; i++) {
    if (!spec->bufs[i].name || !spec->bufs[i].buffer || spec->bufs[i].role > POLY_IR_ROLE_AUX ||
        !ir_interface_shape_valid(&spec->bufs[i])) {
      fprintf(stderr, "poly_ir_export: invalid interface row %d\n", i);
      return NULL;
    }
  }
  if (spec->n_modules < 0 || (spec->n_modules > 0 && !spec->modules) ||
      (executable && spec->n_modules != 0))
    return NULL;
  for (int i = 0; i < spec->n_modules; i++) {
    const PolyIrModule *module = &spec->modules[i];
    if (!module->name || !module->output || module->n_inputs < 0 ||
        (module->n_inputs > 0 && !module->inputs) ||
        !poly_ctx_owns_ptr(spec->ctx, module->output)) {
      fprintf(stderr, "poly_ir_export: invalid module row %d\n", i);
      return NULL;
    }
    for (int j = 0; j < i; j++) {
      if (strcmp(spec->modules[j].name, module->name) == 0 ||
          spec->modules[j].output == module->output) {
        fprintf(stderr, "poly_ir_export: duplicate module row %d\n", i);
        return NULL;
      }
    }
    for (int j = 0; j < module->n_inputs; j++) {
      if (!module->inputs[j] || module->inputs[j] == module->output ||
          !poly_ctx_owns_ptr(spec->ctx, module->inputs[j]) ||
          !poly_uop_reachable(spec->ctx, module->output, module->inputs[j])) {
        fprintf(stderr, "poly_ir_export: invalid module input %d:%d\n", i, j);
        return NULL;
      }
      for (int k = 0; k < j; k++) {
        if (module->inputs[k] == module->inputs[j]) {
          fprintf(stderr, "poly_ir_export: duplicate module input %d:%d\n", i, j);
          return NULL;
        }
      }
    }
  }

  /* Collect all nodes via toposort. Entrypoint sinks define executable graphs;
   * interface BUFFERs are also roots because package/checkpoint state can be
   * named without being read by a normal inference/loss entrypoint. */
  int n_nodes = 0;
  PolyUOp **topo = NULL;
  int topo_is_heap = 1;

  int cap_nodes = 0;
  /* PGIR owns one unique node list, not the sum of overlapping traversals.
   * Root order remains entrypoints then named state. A non-NULL root always
   * has at least one node: NULL/zero means allocation failure, not an empty graph. */
  for (int group = 0; group < 2; group++) {
    int n_roots = group == 0 ? spec->n_entrypoints : spec->n_bufs;
    for (int i = 0; i < n_roots; i++) {
      PolyUOp *root = group == 0 ? spec->entrypoints[i].sink : spec->bufs[i].buffer;
      PolyScratchMark scratch = poly_ctx_scratch_mark(spec->ctx);
      int count = 0;
      PolyUOp **nodes = ir_toposort(spec->ctx, root, &count, true);
      bool ok = nodes && count > 0;
      for (int j = 0; ok && j < count; j++)
        ok = node_list_append_unique(&topo, &n_nodes, &cap_nodes, nodes[j]);
      poly_ctx_scratch_rewind(spec->ctx, scratch);
      if (!ok) {
        free(topo);
        return NULL;
      }
    }
  }

  if (!topo || n_nodes == 0) {
    fprintf(stderr, "poly_ir_export: toposort failed\n");
    if (topo_is_heap) free(topo);
    return NULL;
  }
  if (!program_topo_with_metadata(spec->ctx, &topo, &n_nodes)) {
    fprintf(stderr, "poly_ir_export: invalid metadata dependency\n");
    free(topo);
    return NULL;
  }

  /* Build node index map (UOp pointer -> index) */
  /* Use a simple linear scan (good enough for small-medium graphs) */
  typedef struct {
    PolyUOp *uop;
    uint32_t idx;
  } NodeMapEntry;
  NodeMapEntry *node_map = malloc(n_nodes * sizeof(NodeMapEntry));
  if (!node_map) {
    if (topo_is_heap) free(topo);
    return NULL;
  }
  for (int i = 0; i < n_nodes; i++) {
    node_map[i].uop = topo[i];
    node_map[i].idx = (uint32_t)i;
  }

/* Helper: find index of a UOp */
#define FIND_IDX(u)                                                                                \
  ({                                                                                               \
    uint32_t _idx = UINT32_MAX;                                                                    \
    for (int _i = 0; _i < n_nodes; _i++)                                                           \
      if (node_map[_i].uop == (u)) {                                                               \
        _idx = node_map[_i].idx;                                                                   \
        break;                                                                                     \
      }                                                                                            \
    _idx;                                                                                          \
  })

  /* Module boundaries are metadata over the exported program, not extra graph
   * roots. Reject rows outside the selected entrypoint/interface closure
   * instead of serializing UINT32_MAX node references. */
  for (int i = 0; i < spec->n_modules; i++) {
    if (FIND_IDX(spec->modules[i].output) == UINT32_MAX) {
      fprintf(stderr, "poly_ir_export: module row %d is outside the program\n", i);
      free(node_map);
      if (topo_is_heap) free(topo);
      return NULL;
    }
    for (int j = 0; j < spec->modules[i].n_inputs; j++) {
      if (FIND_IDX(spec->modules[i].inputs[j]) == UINT32_MAX) {
        fprintf(stderr, "poly_ir_export: module input %d:%d is outside the program\n", i, j);
        free(node_map);
        if (topo_is_heap) free(topo);
        return NULL;
      }
    }
  }

  /* Tinygrad 2026-08-22/a9069c177a9d DType is scalar; vector width lives in
   * UOp shape and is therefore already encoded by the graph. */
  for (int i = 0; i < n_nodes; i++) {
    PolyDType dt = topo[i]->dtype;
    if (dtype_to_index(dt) < 0) {
      fprintf(
          stderr, "poly_ir_export: node %d has unknown dtype '%s'\n", i, dt.name ? dt.name : "?"
      );
      free(node_map);
      if (topo_is_heap) free(topo);
      return NULL;
    }
  }

  /* Build string table */
  StringTable strings;
  st_init(&strings);

  /* Collect strings from args */
  for (int i = 0; i < n_nodes; i++) {
    PolyArg a = topo[i]->arg;
    if (executable && !program_tag_arg_valid(topo[i]->tag_arg)) {
      fprintf(
          stderr, "poly_program_export: node %d has unsupported tag metadata kind %d\n", i,
          (int)topo[i]->tag_arg.kind
      );
      free(node_map);
      if (topo_is_heap) free(topo);
      st_free(&strings);
      return NULL;
    }
    if (executable) program_tag_arg_collect_strings(&strings, topo[i]->tag_arg);
    if (a.kind == POLY_ARG_STRING && a.str) st_add(&strings, a.str);
    if (a.kind == POLY_ARG_STRING_TUPLE)
      for (int j = 0; j < a.string_tuple.n; j++)
        st_add(&strings, a.string_tuple.vals[j]);
    if (a.kind == POLY_ARG_ALLREDUCE && a.allreduce.device_is_tuple)
      for (int j = 0; j < a.allreduce.n_devices; j++)
        st_add(&strings, a.allreduce.devices[j]);
    else if (a.kind == POLY_ARG_ALLREDUCE && a.allreduce.device)
      st_add(&strings, a.allreduce.device);
    if (a.kind == POLY_ARG_BUFFERIZE_OPTS && a.bufferize_opts.device_is_tuple) {
      if (a.bufferize_opts.n_devices <= 0 || a.bufferize_opts.n_devices > UINT16_MAX ||
          !a.bufferize_opts.devices) {
        fprintf(stderr, "poly_ir_export: invalid tuple BufferizeOpts device\n");
        free(node_map);
        if (topo_is_heap) free(topo);
        st_free(&strings);
        return NULL;
      }
      for (int j = 0; j < a.bufferize_opts.n_devices; j++) {
        if (!a.bufferize_opts.devices[j]) {
          fprintf(stderr, "poly_ir_export: invalid tuple BufferizeOpts member\n");
          free(node_map);
          if (topo_is_heap) free(topo);
          st_free(&strings);
          return NULL;
        }
        st_add(&strings, a.bufferize_opts.devices[j]);
      }
    } else if (a.kind == POLY_ARG_BUFFERIZE_OPTS && a.bufferize_opts.device) {
      st_add(&strings, a.bufferize_opts.device);
    }
    if (a.kind == POLY_ARG_TENSOR_CORE && a.tensor_core.device)
      st_add(&strings, a.tensor_core.device);
    if (a.kind == POLY_ARG_PARAM && a.param && a.param->name) st_add(&strings, a.param->name);
    if (a.kind == POLY_ARG_PARAM && a.param && a.param->device) st_add(&strings, a.param->device);
    if (a.kind == POLY_ARG_PARAM && a.param && a.param->device_is_tuple)
      for (int j = 0; j < a.param->n_devices; j++)
        st_add(&strings, a.param->devices[j]);
    if (a.kind == POLY_ARG_CALL_INFO && a.call_info && a.call_info->name)
      st_add(&strings, a.call_info->name);
    if (a.kind == POLY_ARG_KERNEL_INFO) {
      const PolyKernelInfo *info = a.kernel_info;
      if (!info || !info->name || info->n_axis_types < 0 || info->n_axis_types > UINT16_MAX ||
          info->n_applied_opts < 0 || info->n_opts_to_apply < 0 ||
          (info->n_axis_types > 0 && !info->axis_types) ||
          (info->n_applied_opts > 0 && !info->applied_opts) ||
          (info->n_opts_to_apply > 0 && !info->opts_to_apply)) {
        fprintf(stderr, "poly_ir_export: invalid KernelInfo metadata\n");
        free(node_map);
        if (topo_is_heap) free(topo);
        st_free(&strings);
        return NULL;
      }
      st_add(&strings, info->name);
    }
    if (a.kind == POLY_ARG_PROGRAM_INFO) {
      const PolyProgramInfo *info = a.program_info;
      if (!executable || !info || !info->target || info->n_vars < 0 || info->n_globals < 0 ||
          info->n_outs < 0 || info->n_ins < 0 || (info->n_vars > 0 && !info->vars) ||
          (info->n_globals > 0 && !info->globals) || (info->n_outs > 0 && !info->outs) ||
          (info->n_ins > 0 && !info->ins)) {
        fprintf(stderr, "poly_program_export: invalid PROGRAM metadata\n");
        free(node_map);
        if (topo_is_heap) free(topo);
        st_free(&strings);
        return NULL;
      }
      if (info->name) st_add(&strings, info->name);
      st_add(&strings, info->target);
    }
    if (a.kind == POLY_ARG_BYTES &&
        (!executable || a.bytes.n < 0 || (a.bytes.n > 0 && !a.bytes.data))) {
      fprintf(stderr, "poly_program_export: invalid BINARY payload\n");
      free(node_map);
      if (topo_is_heap) free(topo);
      st_free(&strings);
      return NULL;
    }
  }
  /* Collect strings from interface + entrypoints */
  for (int i = 0; i < spec->n_bufs; i++)
    st_add(&strings, spec->bufs[i].name);
  for (int i = 0; i < spec->n_entrypoints; i++)
    st_add_entrypoint_strings(&strings, &spec->entrypoints[i]);
  for (int i = 0; i < spec->n_modules; i++)
    st_add(&strings, spec->modules[i].name);

  /* Compute flags */
  uint32_t flags = 0;
  for (int i = 0; i < spec->n_entrypoints; i++) {
    if (strcmp(spec->entrypoints[i].name, "loss") == 0) flags |= 1; /* has_loss */
    if (strcmp(spec->entrypoints[i].name, "train_step") == 0) flags |= 2; /* has_train_step */
  }

  /* Write binary */

  ByteBuf buf;
  bb_init(&buf);

  /* Header (32 bytes) */
  bb_u32(&buf, executable ? PROGRAM_MAGIC : IR_MAGIC);
  bb_u32(&buf, executable ? POLY_PROGRAM_VERSION : POLY_IR_VERSION);
  bb_u32(&buf, executable ? POLYGRAD_ABI_VERSION : flags);
  bb_u32(&buf, (uint32_t)n_nodes);
  bb_u32(&buf, (uint32_t)strings.n);
  bb_u32(&buf, (uint32_t)spec->n_bufs);
  bb_u32(&buf, (uint32_t)spec->n_entrypoints);
  bb_u32(&buf, (uint32_t)spec->n_modules);

  /* String table */
  for (int i = 0; i < strings.n; i++) {
    uint16_t slen = (uint16_t)strlen(strings.strs[i]);
    bb_u16(&buf, slen);
    bb_bytes(&buf, (const uint8_t *)strings.strs[i], slen);
  }

  /* Node table */
  for (int i = 0; i < n_nodes; i++) {
    PolyUOp *u = topo[i];
    bb_u16(&buf, (uint16_t)u->op);
    bb_u8(&buf, (uint8_t)dtype_to_index(u->dtype));
    bb_i32(&buf, u->tag);
    bb_u16(&buf, u->n_src);
    bb_u8(&buf, (uint8_t)u->arg.kind);
    bb_u8(&buf, executable ? (uint8_t)u->tag_arg.kind : 0);
    /* Sources */
    for (int s = 0; s < u->n_src; s++) {
      uint32_t idx = FIND_IDX(u->src[s]);
      if (idx == UINT32_MAX) {
        fprintf(stderr, "poly_ir_export: src not found in toposort\n");
        free(node_map);
        if (topo_is_heap) free(topo);
        st_free(&strings);
        free(buf.data);
        return NULL;
      }
      bb_u32(&buf, idx);
    }

    /* Arg data */
    switch (u->arg.kind) {
    case POLY_ARG_NONE:
      break;
    case POLY_ARG_INT:
      bb_i64(&buf, u->arg.i);
      break;
    case POLY_ARG_BIGINT:
      bb_u8(&buf, u->arg.bigint.sign < 0 ? 1 : 0);
      bb_u32(&buf, u->arg.bigint.n_limbs);
      for (uint32_t limb = 0; limb < u->arg.bigint.n_limbs; limb++)
        bb_u32(&buf, u->arg.bigint.limbs[limb]);
      break;
    case POLY_ARG_FLOAT:
      bb_f64(&buf, u->arg.f);
      break;
    case POLY_ARG_BOOL:
      bb_u8(&buf, u->arg.b ? 1 : 0);
      break;
    case POLY_ARG_INT_TUPLE:
      bb_u16(&buf, (uint16_t)u->arg.int_tuple.n);
      for (int t = 0; t < u->arg.int_tuple.n; t++)
        bb_i64(&buf, u->arg.int_tuple.vals[t]);
      break;
    case POLY_ARG_STRING:
      bb_u32(&buf, st_add(&strings, u->arg.str));
      break;
    case POLY_ARG_STRING_TUPLE:
      bb_u16(&buf, (uint16_t)u->arg.string_tuple.n);
      for (int t = 0; t < u->arg.string_tuple.n; t++)
        bb_u32(&buf, st_add(&strings, u->arg.string_tuple.vals[t]));
      break;
    case POLY_ARG_OPS:
      bb_u16(&buf, (uint16_t)u->arg.ops);
      break;
    case POLY_ARG_REDUCE:
      bb_u16(&buf, (uint16_t)u->arg.reduce.op);
      bb_u16(&buf, (uint16_t)u->arg.reduce.num_axes);
      break;
    case POLY_ARG_ALLREDUCE:
      bb_u16(&buf, (uint16_t)u->arg.allreduce.op);
      if (u->arg.allreduce.device_is_tuple) {
        bb_u8(&buf, 2);
        bb_u16(&buf, (uint16_t)u->arg.allreduce.n_devices);
        for (int d = 0; d < u->arg.allreduce.n_devices; d++)
          bb_u32(&buf, st_add(&strings, u->arg.allreduce.devices[d]));
      } else {
        bb_u8(&buf, 1);
        bb_u32(&buf, st_add(&strings, u->arg.allreduce.device));
      }
      break;
    case POLY_ARG_RANGE:
      bb_i64(&buf, u->arg.range.axis_id);
      bb_u8(&buf, (uint8_t)u->arg.range.axis_type);
      bb_u16(&buf, (uint16_t)u->arg.range.n_extra);
      for (int t = 0; t < u->arg.range.n_extra; t++)
        bb_i64(&buf, u->arg.range.extra[t]);
      break;
    case POLY_ARG_BUFFERIZE_OPTS:
      if (u->arg.bufferize_opts.device_is_tuple) {
        bb_u8(&buf, 2);
        bb_u16(&buf, (uint16_t)u->arg.bufferize_opts.n_devices);
        for (int d = 0; d < u->arg.bufferize_opts.n_devices; d++)
          bb_u32(&buf, st_add(&strings, u->arg.bufferize_opts.devices[d]));
      } else if (u->arg.bufferize_opts.device) {
        bb_u8(&buf, 1);
        bb_u32(&buf, st_add(&strings, u->arg.bufferize_opts.device));
      } else {
        bb_u8(&buf, 0);
      }
      bb_u8(&buf, (uint8_t)u->arg.bufferize_opts.addrspace);
      bb_u8(&buf, u->arg.bufferize_opts.removable ? 1 : 0);
      break;
    case POLY_ARG_TENSOR_CORE:
      for (int d = 0; d < 3; d++)
        bb_i64(&buf, u->arg.tensor_core.dims[d]);
      bb_u8(&buf, (uint8_t)dtype_to_index(u->arg.tensor_core.dtype_in));
      bb_u32(&buf, st_add(&strings, u->arg.tensor_core.device));
      bb_i64(&buf, u->arg.tensor_core.threads);
      bb_u8(&buf, u->arg.tensor_core.has_upcast_axes ? 1 : 0);
      for (int d = 0; d < 3; d++) {
        bb_u16(&buf, (uint16_t)u->arg.tensor_core.n_upcast_axes[d]);
        for (int i = 0; i < u->arg.tensor_core.n_upcast_axes[d]; i++) {
          bb_i64(&buf, u->arg.tensor_core.upcast_axes[d][i][0]);
          bb_i64(&buf, u->arg.tensor_core.upcast_axes[d][i][1]);
        }
      }
      break;
    case POLY_ARG_PROGRAM_INFO:
      if (!executable || !u->arg.program_info) {
        fprintf(stderr, "poly_ir_export: PROGRAM metadata is not exportable IR\n");
        free(node_map);
        if (topo_is_heap) free(topo);
        st_free(&strings);
        free(buf.data);
        return NULL;
      } else {
        const PolyProgramInfo *info = u->arg.program_info;
        bb_u32(&buf, info->name ? st_add(&strings, info->name) : UINT32_MAX);
        bb_u32(&buf, st_add(&strings, info->target));
        bb_u8(&buf, info->has_local_size ? 1 : 0);
        bb_u8(&buf, 0);
        bb_u16(&buf, 0);
        for (int d = 0; d < 3; d++)
          bb_i32(&buf, info->global_size[d]);
        for (int d = 0; d < 3; d++)
          bb_i32(&buf, info->local_size[d]);
        for (int d = 0; d < 3; d++)
          bb_u32(&buf, info->global_exprs[d] ? FIND_IDX(info->global_exprs[d]) : UINT32_MAX);
        for (int d = 0; d < 3; d++)
          bb_u32(&buf, info->local_exprs[d] ? FIND_IDX(info->local_exprs[d]) : UINT32_MAX);
        bb_u32(&buf, (uint32_t)info->n_vars);
        for (int v = 0; v < info->n_vars; v++)
          bb_u32(&buf, FIND_IDX(info->vars[v]));
        bb_u32(&buf, (uint32_t)info->n_globals);
        for (int g = 0; g < info->n_globals; g++)
          bb_i32(&buf, info->globals[g]);
        bb_u32(&buf, (uint32_t)info->n_outs);
        for (int o = 0; o < info->n_outs; o++)
          bb_i32(&buf, info->outs[o]);
        bb_u32(&buf, (uint32_t)info->n_ins);
        for (int in = 0; in < info->n_ins; in++)
          bb_i32(&buf, info->ins[in]);
      }
      break;
    case POLY_ARG_KERNEL_INFO: {
      const PolyKernelInfo *info = u->arg.kernel_info;
      bb_u32(&buf, st_add(&strings, info->name));
      bb_u16(&buf, (uint16_t)info->n_axis_types);
      for (int axis = 0; axis < info->n_axis_types; axis++)
        bb_u8(&buf, (uint8_t)info->axis_types[axis]);
      bb_u8(&buf, info->dont_use_locals ? 1 : 0);
      bb_u32(&buf, (uint32_t)info->n_applied_opts);
      for (int opt = 0; opt < info->n_applied_opts; opt++)
        bb_opt(&buf, &info->applied_opts[opt]);
      bb_u8(&buf, info->has_opts_to_apply ? 1 : 0);
      bb_u32(&buf, (uint32_t)info->n_opts_to_apply);
      for (int opt = 0; opt < info->n_opts_to_apply; opt++)
        bb_opt(&buf, &info->opts_to_apply[opt]);
      bb_u32(
          &buf,
          info->estimates && info->estimates->ops ? FIND_IDX(info->estimates->ops) : UINT32_MAX
      );
      bb_u32(
          &buf,
          info->estimates && info->estimates->lds ? FIND_IDX(info->estimates->lds) : UINT32_MAX
      );
      bb_u32(
          &buf,
          info->estimates && info->estimates->mem ? FIND_IDX(info->estimates->mem) : UINT32_MAX
      );
      bb_i32(&buf, info->beam);
      break;
    }
    case POLY_ARG_BYTES:
      if (!executable || u->arg.bytes.n < 0 || (u->arg.bytes.n > 0 && !u->arg.bytes.data)) {
        fprintf(stderr, "poly_ir_export: BINARY byte payloads are not exportable IR\n");
        free(node_map);
        if (topo_is_heap) free(topo);
        st_free(&strings);
        free(buf.data);
        return NULL;
      }
      bb_u32(&buf, (uint32_t)u->arg.bytes.n);
      bb_bytes(&buf, u->arg.bytes.data, u->arg.bytes.n);
      break;
    case POLY_ARG_INVALID:
      break;
    case POLY_ARG_PARAM:
      if (!u->arg.param) {
        fprintf(stderr, "poly_ir_export: PARAM metadata is NULL\n");
        free(node_map);
        if (topo_is_heap) free(topo);
        st_free(&strings);
        free(buf.data);
        return NULL;
      }
      bb_i64(&buf, u->arg.param->slot);
      if (u->arg.param->device_is_tuple) {
        bb_u8(&buf, 2);
        bb_u16(&buf, (uint16_t)u->arg.param->n_devices);
        for (int d = 0; d < u->arg.param->n_devices; d++)
          bb_u32(&buf, st_add(&strings, u->arg.param->devices[d]));
      } else if (u->arg.param->device) {
        bb_u8(&buf, 1);
        bb_u32(&buf, st_add(&strings, u->arg.param->device));
      } else {
        bb_u8(&buf, 0);
      }
      bb_u8(&buf, (uint8_t)u->arg.param->addrspace);
      bb_u8(&buf, u->arg.param->has_axis ? 1 : 0);
      bb_i32(&buf, u->arg.param->axis);
      bb_u8(&buf, u->arg.param->has_minmax ? 1 : 0);
      bb_u32(&buf, u->arg.param->name ? st_add(&strings, u->arg.param->name) : UINT32_MAX);
      bb_i64(&buf, u->arg.param->min_val);
      bb_i64(&buf, u->arg.param->max_val);
      break;
    case POLY_ARG_CALL_INFO:
      if (!u->arg.call_info || u->arg.call_info->has_grad_fxn || u->arg.call_info->has_aux) {
        fprintf(stderr, "poly_ir_export: unsupported CallInfo callback/aux\n");
        free(node_map);
        if (topo_is_heap) free(topo);
        st_free(&strings);
        free(buf.data);
        return NULL;
      }
      bb_u32(&buf, u->arg.call_info->name ? st_add(&strings, u->arg.call_info->name) : UINT32_MAX);
      bb_u8(&buf, u->arg.call_info->precompile ? 1 : 0);
      bb_u8(&buf, u->arg.call_info->precompile_backward ? 1 : 0);
      break;
    case POLY_ARG_DTYPE:
      /* CAST/BITCAST.arg is required to equal the node dtype. The kind is
       * the complete wire identity; the existing dtype header is its payload. */
      break;
    }
    if (executable) bb_program_tag_arg(&buf, &strings, u->tag_arg);
  }

  /* Interface table */
  for (int i = 0; i < spec->n_bufs; i++) {
    bb_u32(&buf, st_add(&strings, spec->bufs[i].name));
    bb_u8(&buf, spec->bufs[i].role);
    /* Interface flags: bit0 is trainable; bit1 marks explicit metadata. */
    bool trainable = spec->bufs[i].trainable_set ? spec->bufs[i].trainable
                                                 : (spec->bufs[i].role == POLY_IR_ROLE_PARAM);
    bb_u8(&buf, 2 | (trainable ? 1 : 0));
    bb_u8(&buf, 0);
    bb_u8(&buf, 0); /* padding */
    uint32_t nidx = FIND_IDX(spec->bufs[i].buffer);
    bb_u32(&buf, nidx);
    bb_u16(&buf, (uint16_t)spec->bufs[i].ndim);
    bb_u16(&buf, 0); /* padding */
    for (int d = 0; d < spec->bufs[i].ndim; d++)
      bb_i64(&buf, spec->bufs[i].shape[d]);
  }

  /* Entrypoint table */
  for (int i = 0; i < spec->n_entrypoints; i++) {
    const PolyIrEntrypoint *ep = &spec->entrypoints[i];
    bb_u32(&buf, st_add(&strings, ep->name));
    uint32_t nidx = FIND_IDX(ep->sink);
    bb_u32(&buf, nidx);
    bb_u32(&buf, ep->flags);
    bb_u16(&buf, (uint16_t)ep->n_inputs);
    bb_u16(&buf, (uint16_t)ep->n_outputs);
    bb_u32(&buf, ep->objective ? st_add(&strings, ep->objective) : UINT32_MAX);
    for (int j = 0; j < ep->n_inputs; j++)
      bb_u32(&buf, st_add(&strings, ep->inputs[j]));
    for (int j = 0; j < ep->n_outputs; j++)
      bb_u32(&buf, st_add(&strings, ep->outputs[j]));
  }

  /* Exact logical module boundaries. Devices are intentionally absent: the
   * same portable program can be placed under a different policy. */
  for (int i = 0; i < spec->n_modules; i++) {
    const PolyIrModule *module = &spec->modules[i];
    bb_u32(&buf, st_add(&strings, module->name));
    bb_u32(&buf, (uint32_t)module->n_inputs);
    for (int j = 0; j < module->n_inputs; j++)
      bb_u32(&buf, FIND_IDX(module->inputs[j]));
    bb_u32(&buf, FIND_IDX(module->output));
  }

#undef FIND_IDX

  free(node_map);
  if (topo_is_heap) free(topo);
  st_free(&strings);

  *out_len = buf.len;
  return buf.data;
}

uint8_t *poly_ir_export(const PolyIrSpec *spec, int *out_len) {
  return poly_graph_export(spec, out_len, false);
}

uint8_t *poly_program_graph_export(const PolyIrSpec *spec, int *out_len) {
  return poly_graph_export(spec, out_len, true);
}

/* Import */

static int poly_graph_import(const uint8_t *data, int len, PolyIrSpec *out, bool executable) {
  memset(out, 0, sizeof(PolyIrSpec));

  ByteReader r = {data, len, 0};

  /* Header */
  if (br_remaining(&r) < 32) {
    fprintf(
        stderr, "%s: data too short for header\n",
        executable ? "poly_program_import" : "poly_ir_import"
    );
    return -1;
  }

  uint32_t magic = br_u32(&r);
  uint32_t expected_magic = executable ? PROGRAM_MAGIC : IR_MAGIC;
  if (magic != expected_magic) {
    fprintf(
        stderr, "%s: bad magic 0x%08x\n", executable ? "poly_program_import" : "poly_ir_import",
        magic
    );
    return -1;
  }
  uint32_t version = br_u32(&r);
  if (version != (executable ? POLY_PROGRAM_VERSION : POLY_IR_VERSION)) {
    fprintf(
        stderr, "%s: unsupported version %u\n",
        executable ? "poly_program_import" : "poly_ir_import", version
    );
    return -1;
  }
  uint32_t flags = br_u32(&r);
  if (executable && flags != POLYGRAD_ABI_VERSION) {
    fprintf(
        stderr, "poly_program_import: ABI mismatch, expected %u, got %u\n",
        (unsigned)POLYGRAD_ABI_VERSION, (unsigned)flags
    );
    return -1;
  }
  uint32_t n_nodes = br_u32(&r);
  uint32_t n_strings = br_u32(&r);
  uint32_t n_entries = br_u32(&r);
  uint32_t n_entrypts = br_u32(&r);
  uint32_t n_modules = br_u32(&r);
  if (executable && n_modules != 0) {
    fprintf(stderr, "poly_program_import: module table is not executable metadata\n");
    return -1;
  }
  if (n_nodes > INT_MAX || n_strings > INT_MAX || n_entries > INT_MAX || n_entrypts > INT_MAX ||
      n_modules > INT_MAX) {
    fprintf(stderr, "poly_ir_import: header count exceeds supported range\n");
    return -1;
  }

  /* String table */
  char **strings = calloc(n_strings, sizeof(char *));
  if (n_strings > 0 && !strings) return -1;
  for (uint32_t i = 0; i < n_strings; i++) {
    if (br_remaining(&r) < 2) goto fail_strings;
    uint16_t slen = br_u16(&r);
    if (br_remaining(&r) < slen) goto fail_strings;
    strings[i] = malloc((size_t)slen + 1);
    if (!strings[i]) goto fail_strings;
    memcpy(strings[i], r.data + r.pos, slen);
    strings[i][slen] = '\0';
    r.pos += slen;
  }

  /* Create context */
  PolyCtx *ctx = poly_ctx_new();
  if (!ctx) goto fail_strings;
  PolyUOp **nodes = calloc(n_nodes, sizeof(PolyUOp *));
  if (n_nodes > 0 && !nodes) goto fail_nodes;

  /* Node table */
  for (uint32_t i = 0; i < n_nodes; i++) {
    int node_header_bytes = executable ? 23 : 11;
    if (br_remaining(&r) < node_header_bytes) goto fail_nodes;

    uint16_t op_val = br_u16(&r);
    uint8_t dtype_idx = br_u8(&r);
    int32_t tag = br_i32(&r);
    uint16_t n_src = br_u16(&r);
    uint8_t arg_kind = br_u8(&r);
    uint8_t tag_arg_kind = br_u8(&r);
    if (op_val == 0 || op_val >= POLY_OP_COUNT) {
      fprintf(stderr, "poly_ir_import: invalid op %u at node %u\n", op_val, i);
      goto fail_nodes;
    }
    if (dtype_idx >= N_DTYPES) {
      fprintf(stderr, "poly_ir_import: invalid dtype index %u at node %u\n", dtype_idx, i);
      goto fail_nodes;
    }
    /* Read sources */
    if (br_remaining(&r) < (int)(n_src * 4)) goto fail_nodes;
    PolyUOp **srcs = NULL;
    if (n_src > 0) {
      srcs = malloc(n_src * sizeof(PolyUOp *));
      for (int s = 0; s < n_src; s++) {
        uint32_t src_idx = br_u32(&r);
        if (src_idx >= i) {
          fprintf(stderr, "poly_ir_import: forward ref at node %u src %d\n", i, s);
          free(srcs);
          goto fail_nodes;
        }
        srcs[s] = nodes[src_idx];
      }
    }

    /* Read arg */
    PolyArg arg;
    PolyArg tag_arg = poly_arg_none();
    PolyUOp *u = NULL;
    memset(&arg, 0, sizeof(arg));
    arg.kind = (PolyArgKind)arg_kind;
    PolyParamArg param_arg_tmp;
    PolyCallInfo call_info_tmp;
    PolyKernelInfo kernel_info_tmp;
    PolyEstimates kernel_estimates_tmp;
    PolyAxisType *kernel_axis_types_tmp = NULL;
    PolyOpt *kernel_applied_opts_tmp = NULL;
    PolyOpt *kernel_opts_to_apply_tmp = NULL;
    const char **param_devices_tmp = NULL;
    const char **bufferize_devices_tmp = NULL;
    const char **allreduce_devices_tmp = NULL;
    int64_t(*wmma_upcast_axes_tmp[3])[2] = {NULL, NULL, NULL};
    memset(&param_arg_tmp, 0, sizeof(param_arg_tmp));
    memset(&call_info_tmp, 0, sizeof(call_info_tmp));
    memset(&kernel_info_tmp, 0, sizeof(kernel_info_tmp));
    memset(&kernel_estimates_tmp, 0, sizeof(kernel_estimates_tmp));

    switch (arg.kind) {
    case POLY_ARG_NONE:
      break;
    case POLY_ARG_INT:
      arg.i = br_i64(&r);
      break;
    case POLY_ARG_BIGINT: {
      if (br_remaining(&r) < 5) {
        if (srcs) free(srcs);
        goto fail_nodes;
      }
      bool negative = br_u8(&r) != 0;
      uint32_t n_limbs = br_u32(&r);
      if (n_limbs == 0 || n_limbs > (uint32_t)(INT_MAX / (int)sizeof(uint32_t)) ||
          br_remaining(&r) < (int)(n_limbs * sizeof(uint32_t))) {
        if (srcs) free(srcs);
        goto fail_nodes;
      }
      uint32_t *limbs = malloc((size_t)n_limbs * sizeof(uint32_t));
      if (!limbs) {
        if (srcs) free(srcs);
        goto fail_nodes;
      }
      for (uint32_t limb = 0; limb < n_limbs; limb++)
        limbs[limb] = br_u32(&r);
      arg.bigint.sign = negative ? -1 : 1;
      arg.bigint.n_limbs = n_limbs;
      arg.bigint.limbs = limbs;
      break;
    }
    case POLY_ARG_FLOAT:
      arg.f = br_f64(&r);
      break;
    case POLY_ARG_BOOL:
      arg.b = br_u8(&r) != 0;
      break;
    case POLY_ARG_INT_TUPLE: {
      uint16_t count = br_u16(&r);
      int64_t *vals = malloc(count * sizeof(int64_t));
      for (int t = 0; t < count; t++)
        vals[t] = br_i64(&r);
      arg.int_tuple.vals = vals;
      arg.int_tuple.n = count;
      break;
    }
    case POLY_ARG_STRING: {
      uint32_t str_idx = br_u32(&r);
      if (str_idx < n_strings)
        arg.str = strings[str_idx];
      else
        arg.str = "";
      break;
    }
    case POLY_ARG_STRING_TUPLE: {
      if (br_remaining(&r) < 2) {
        if (srcs) free(srcs);
        goto fail_nodes;
      }
      uint16_t count = br_u16(&r);
      if (br_remaining(&r) < (int)((uint32_t)count * sizeof(uint32_t))) {
        if (srcs) free(srcs);
        goto fail_nodes;
      }
      const char **vals = count > 0 ? malloc((size_t)count * sizeof(*vals)) : NULL;
      if (count > 0 && !vals) {
        if (srcs) free(srcs);
        goto fail_nodes;
      }
      for (int t = 0; t < count; t++) {
        uint32_t str_idx = br_u32(&r);
        if (str_idx >= n_strings) {
          free(vals);
          if (srcs) free(srcs);
          goto fail_nodes;
        }
        vals[t] = strings[str_idx];
      }
      arg.string_tuple.vals = vals;
      arg.string_tuple.n = count;
      break;
    }
    case POLY_ARG_OPS:
      arg.ops = (PolyOps)br_u16(&r);
      break;
    case POLY_ARG_REDUCE:
      arg.reduce.op = (PolyOps)br_u16(&r);
      arg.reduce.num_axes = (int)br_u16(&r);
      break;
    case POLY_ARG_ALLREDUCE: {
      arg.allreduce.op = (PolyOps)br_u16(&r);
      uint8_t device_kind = br_u8(&r);
      if (device_kind == 1) {
        uint32_t device_idx = br_u32(&r);
        if (device_idx >= n_strings) {
          if (srcs) free(srcs);
          goto fail_nodes;
        }
        arg.allreduce.device = strings[device_idx];
      } else if (device_kind == 2) {
        uint16_t count = br_u16(&r);
        if (count == 0) {
          if (srcs) free(srcs);
          goto fail_nodes;
        }
        allreduce_devices_tmp = malloc((size_t)count * sizeof(*allreduce_devices_tmp));
        if (!allreduce_devices_tmp) {
          if (srcs) free(srcs);
          goto fail_nodes;
        }
        for (int d = 0; d < count; d++) {
          uint32_t device_idx = br_u32(&r);
          if (device_idx >= n_strings) {
            free(allreduce_devices_tmp);
            if (srcs) free(srcs);
            goto fail_nodes;
          }
          allreduce_devices_tmp[d] = strings[device_idx];
        }
        arg.allreduce.devices = allreduce_devices_tmp;
        arg.allreduce.n_devices = count;
        arg.allreduce.device_is_tuple = true;
      } else {
        if (srcs) free(srcs);
        goto fail_nodes;
      }
      break;
    }
    case POLY_ARG_RANGE: {
      arg.range.axis_id = br_i64(&r);
      arg.range.axis_type = (PolyAxisType)br_u8(&r);
      uint16_t n_extra = br_u16(&r);
      if (n_extra > 0) {
        int64_t *extra = malloc(n_extra * sizeof(int64_t));
        for (int t = 0; t < n_extra; t++)
          extra[t] = br_i64(&r);
        arg.range.extra = extra;
      } else {
        arg.range.extra = NULL;
      }
      arg.range.n_extra = n_extra;
      break;
    }
    case POLY_ARG_BUFFERIZE_OPTS: {
      uint8_t device_kind = br_u8(&r);
      if (device_kind == 1) {
        uint32_t device_idx = br_u32(&r);
        arg.bufferize_opts.device = device_idx < n_strings ? strings[device_idx] : NULL;
        if (!arg.bufferize_opts.device) {
          if (srcs) free(srcs);
          goto fail_nodes;
        }
      } else if (device_kind == 2) {
        uint16_t count = br_u16(&r);
        if (count == 0 || br_remaining(&r) < (int)((uint32_t)count * sizeof(uint32_t))) {
          if (srcs) free(srcs);
          goto fail_nodes;
        }
        bufferize_devices_tmp = malloc((size_t)count * sizeof(*bufferize_devices_tmp));
        if (!bufferize_devices_tmp) {
          if (srcs) free(srcs);
          goto fail_nodes;
        }
        for (int d = 0; d < count; d++) {
          uint32_t device_idx = br_u32(&r);
          if (device_idx >= n_strings) {
            free(bufferize_devices_tmp);
            if (srcs) free(srcs);
            goto fail_nodes;
          }
          bufferize_devices_tmp[d] = strings[device_idx];
        }
        arg.bufferize_opts.devices = bufferize_devices_tmp;
        arg.bufferize_opts.n_devices = count;
        arg.bufferize_opts.device_is_tuple = true;
      } else if (device_kind != 0) {
        if (srcs) free(srcs);
        goto fail_nodes;
      }
      arg.bufferize_opts.addrspace = (PolyAddrSpace)br_u8(&r);
      arg.bufferize_opts.removable = br_u8(&r) != 0;
      break;
    }
    case POLY_ARG_TENSOR_CORE: {
      for (int d = 0; d < 3; d++)
        arg.tensor_core.dims[d] = (int)br_i64(&r);
      uint8_t input_dtype_idx = br_u8(&r);
      uint32_t device_idx = br_u32(&r);
      if (input_dtype_idx >= N_DTYPES || device_idx >= n_strings) {
        if (srcs) free(srcs);
        goto fail_nodes;
      }
      arg.tensor_core.dtype_in = *dtype_table[input_dtype_idx];
      arg.tensor_core.device = strings[device_idx];
      arg.tensor_core.threads = (int)br_i64(&r);
      arg.tensor_core.has_upcast_axes = br_u8(&r) != 0;
      for (int d = 0; d < 3; d++) {
        if (br_remaining(&r) < 2) goto cleanup_node_arg;
        uint16_t count = br_u16(&r);
        if (br_remaining(&r) < (int64_t)count * 16) {
          goto cleanup_node_arg;
        }
        if (count > 0) {
          wmma_upcast_axes_tmp[d] = malloc((size_t)count * sizeof(*wmma_upcast_axes_tmp[d]));
          if (!wmma_upcast_axes_tmp[d]) {
            goto cleanup_node_arg;
          }
          for (int i = 0; i < count; i++) {
            wmma_upcast_axes_tmp[d][i][0] = br_i64(&r);
            wmma_upcast_axes_tmp[d][i][1] = br_i64(&r);
          }
        }
        arg.tensor_core.upcast_axes[d] = wmma_upcast_axes_tmp[d];
        arg.tensor_core.n_upcast_axes[d] = count;
      }
      break;
    }
    case POLY_ARG_PROGRAM_INFO: {
      if (!executable || br_remaining(&r) < 76) {
        if (!executable) fprintf(stderr, "poly_ir_import: PROGRAM metadata is not package IR\n");
        if (srcs) free(srcs);
        goto fail_nodes;
      }
      PolyProgramInfo *info =
          poly_arena_alloc(ctx->arena, sizeof(*info), _Alignof(PolyProgramInfo));
      if (!info) {
        if (srcs) free(srcs);
        goto fail_nodes;
      }
      memset(info, 0, sizeof(*info));
      uint32_t name_idx = br_u32(&r);
      if (name_idx != UINT32_MAX) {
        if (name_idx >= n_strings) {
          if (srcs) free(srcs);
          goto fail_nodes;
        }
        size_t name_len = strlen(strings[name_idx]);
        char *name = poly_arena_alloc(ctx->arena, name_len + 1, 1);
        if (!name) {
          if (srcs) free(srcs);
          goto fail_nodes;
        }
        memcpy(name, strings[name_idx], name_len + 1);
        info->name = name;
      }
      uint32_t target_idx = br_u32(&r);
      if (target_idx >= n_strings) {
        if (srcs) free(srcs);
        goto fail_nodes;
      }
      size_t target_len = strlen(strings[target_idx]);
      char *target = poly_arena_alloc(ctx->arena, target_len + 1, 1);
      if (!target) {
        if (srcs) free(srcs);
        goto fail_nodes;
      }
      memcpy(target, strings[target_idx], target_len + 1);
      info->target = target;
      info->has_local_size = br_u8(&r) != 0;
      (void)br_u8(&r);
      (void)br_u16(&r);
      for (int d = 0; d < 3; d++)
        info->global_size[d] = br_i32(&r);
      for (int d = 0; d < 3; d++)
        info->local_size[d] = br_i32(&r);
      for (int d = 0; d < 3; d++)
        if (!br_prior_node_ref(&r, nodes, i, true, &info->global_exprs[d])) {
          if (srcs) free(srcs);
          goto fail_nodes;
        }
      for (int d = 0; d < 3; d++)
        if (!br_prior_node_ref(&r, nodes, i, true, &info->local_exprs[d])) {
          if (srcs) free(srcs);
          goto fail_nodes;
        }
      if (br_remaining(&r) < 4) {
        if (srcs) free(srcs);
        goto fail_nodes;
      }
      uint32_t n_vars = br_u32(&r);
      if (n_vars > INT_MAX || br_remaining(&r) < (int64_t)n_vars * 4) {
        if (srcs) free(srcs);
        goto fail_nodes;
      }
      info->n_vars = (int)n_vars;
      if (n_vars > 0) {
        info->vars =
            poly_arena_alloc(ctx->arena, (size_t)n_vars * sizeof(*info->vars), _Alignof(PolyUOp *));
        if (!info->vars) {
          if (srcs) free(srcs);
          goto fail_nodes;
        }
        for (uint32_t v = 0; v < n_vars; v++)
          if (!br_prior_node_ref(&r, nodes, i, false, &info->vars[v])) {
            if (srcs) free(srcs);
            goto fail_nodes;
          }
      }
#define READ_PROGRAM_INT_ARRAY(field, count_field)                                                 \
  do {                                                                                             \
    if (br_remaining(&r) < 4) {                                                                    \
      if (srcs) free(srcs);                                                                        \
      goto fail_nodes;                                                                             \
    }                                                                                              \
    uint32_t _count = br_u32(&r);                                                                  \
    if (_count > INT_MAX || br_remaining(&r) < (int64_t)_count * 4) {                              \
      if (srcs) free(srcs);                                                                        \
      goto fail_nodes;                                                                             \
    }                                                                                              \
    info->count_field = (int)_count;                                                               \
    if (_count > 0) {                                                                              \
      info->field =                                                                                \
          poly_arena_alloc(ctx->arena, (size_t)_count * sizeof(*info->field), _Alignof(int));      \
      if (!info->field) {                                                                          \
        if (srcs) free(srcs);                                                                      \
        goto fail_nodes;                                                                           \
      }                                                                                            \
      for (uint32_t _j = 0; _j < _count; _j++)                                                     \
        info->field[_j] = br_i32(&r);                                                              \
    }                                                                                              \
  } while (0)
      READ_PROGRAM_INT_ARRAY(globals, n_globals);
      READ_PROGRAM_INT_ARRAY(outs, n_outs);
      READ_PROGRAM_INT_ARRAY(ins, n_ins);
#undef READ_PROGRAM_INT_ARRAY
      arg.program_info = info;
      break;
    }
    case POLY_ARG_KERNEL_INFO: {
      if (br_remaining(&r) < 11) {
        if (srcs) free(srcs);
        goto fail_nodes;
      }
      uint32_t name_idx = br_u32(&r);
      if (name_idx >= n_strings) {
        if (srcs) free(srcs);
        goto fail_nodes;
      }
      kernel_info_tmp.name = strings[name_idx];
      uint16_t n_axis_types = br_u16(&r);
      if (br_remaining(&r) < n_axis_types) {
        if (srcs) free(srcs);
        goto fail_nodes;
      }
      if (n_axis_types > 0) {
        kernel_axis_types_tmp = malloc((size_t)n_axis_types * sizeof(*kernel_axis_types_tmp));
        if (!kernel_axis_types_tmp) {
          if (srcs) free(srcs);
          goto fail_nodes;
        }
        for (int axis = 0; axis < n_axis_types; axis++)
          kernel_axis_types_tmp[axis] = (PolyAxisType)br_u8(&r);
      }
      kernel_info_tmp.axis_types = kernel_axis_types_tmp;
      kernel_info_tmp.n_axis_types = n_axis_types;
      if (br_remaining(&r) < 5) {
        free(kernel_axis_types_tmp);
        if (srcs) free(srcs);
        goto fail_nodes;
      }
      kernel_info_tmp.dont_use_locals = br_u8(&r) != 0;
      uint32_t n_applied = br_u32(&r);
      if (n_applied > INT_MAX) {
        free(kernel_axis_types_tmp);
        if (srcs) free(srcs);
        goto fail_nodes;
      }
      kernel_applied_opts_tmp =
          n_applied > 0 ? calloc((size_t)n_applied, sizeof(*kernel_applied_opts_tmp)) : NULL;
      if (n_applied > 0 && !kernel_applied_opts_tmp) {
        free(kernel_axis_types_tmp);
        if (srcs) free(srcs);
        goto fail_nodes;
      }
      for (uint32_t opt = 0; opt < n_applied; opt++) {
        if (br_opt(&r, &kernel_applied_opts_tmp[opt])) continue;
        free_opts(kernel_applied_opts_tmp, (int)n_applied);
        free(kernel_axis_types_tmp);
        if (srcs) free(srcs);
        goto fail_nodes;
      }
      kernel_info_tmp.applied_opts = kernel_applied_opts_tmp;
      kernel_info_tmp.n_applied_opts = (int)n_applied;
      if (br_remaining(&r) < 5) {
        free_opts(kernel_applied_opts_tmp, (int)n_applied);
        free(kernel_axis_types_tmp);
        if (srcs) free(srcs);
        goto fail_nodes;
      }
      kernel_info_tmp.has_opts_to_apply = br_u8(&r) != 0;
      uint32_t n_to_apply = br_u32(&r);
      if (n_to_apply > INT_MAX || (!kernel_info_tmp.has_opts_to_apply && n_to_apply != 0)) {
        free_opts(kernel_applied_opts_tmp, (int)n_applied);
        free(kernel_axis_types_tmp);
        if (srcs) free(srcs);
        goto fail_nodes;
      }
      kernel_opts_to_apply_tmp =
          n_to_apply > 0 ? calloc((size_t)n_to_apply, sizeof(*kernel_opts_to_apply_tmp)) : NULL;
      if (n_to_apply > 0 && !kernel_opts_to_apply_tmp) {
        free_opts(kernel_applied_opts_tmp, (int)n_applied);
        free(kernel_axis_types_tmp);
        if (srcs) free(srcs);
        goto fail_nodes;
      }
      for (uint32_t opt = 0; opt < n_to_apply; opt++) {
        if (br_opt(&r, &kernel_opts_to_apply_tmp[opt])) continue;
        free_opts(kernel_opts_to_apply_tmp, (int)n_to_apply);
        free_opts(kernel_applied_opts_tmp, (int)n_applied);
        free(kernel_axis_types_tmp);
        if (srcs) free(srcs);
        goto fail_nodes;
      }
      kernel_info_tmp.opts_to_apply = kernel_opts_to_apply_tmp;
      kernel_info_tmp.n_opts_to_apply = (int)n_to_apply;
      if (!br_prior_node_ref(&r, nodes, i, true, &kernel_estimates_tmp.ops) ||
          !br_prior_node_ref(&r, nodes, i, true, &kernel_estimates_tmp.lds) ||
          !br_prior_node_ref(&r, nodes, i, true, &kernel_estimates_tmp.mem) ||
          br_remaining(&r) < 4) {
        free_opts(kernel_opts_to_apply_tmp, (int)n_to_apply);
        free_opts(kernel_applied_opts_tmp, (int)n_applied);
        free(kernel_axis_types_tmp);
        if (srcs) free(srcs);
        goto fail_nodes;
      }
      if (kernel_estimates_tmp.ops || kernel_estimates_tmp.lds || kernel_estimates_tmp.mem)
        kernel_info_tmp.estimates = &kernel_estimates_tmp;
      kernel_info_tmp.beam = br_i32(&r);
      arg.kernel_info = &kernel_info_tmp;
      break;
    }
    case POLY_ARG_BYTES: {
      if (!executable || br_remaining(&r) < 4) {
        if (!executable)
          fprintf(stderr, "poly_ir_import: BINARY byte payloads are not package IR\n");
        if (srcs) free(srcs);
        goto fail_nodes;
      }
      uint32_t n_bytes = br_u32(&r);
      if (n_bytes > INT_MAX || br_remaining(&r) < (int64_t)n_bytes) {
        if (srcs) free(srcs);
        goto fail_nodes;
      }
      arg.bytes.data = r.data + r.pos;
      arg.bytes.n = (int)n_bytes;
      r.pos += (int)n_bytes;
      break;
    }
    case POLY_ARG_INVALID:
      break;
    case POLY_ARG_PARAM: {
      param_arg_tmp.slot = br_i64(&r);
      uint8_t device_kind = br_u8(&r);
      if (device_kind == 1) {
        uint32_t device_idx = br_u32(&r);
        param_arg_tmp.device = device_idx < n_strings ? strings[device_idx] : NULL;
        if (!param_arg_tmp.device) {
          if (srcs) free(srcs);
          goto fail_nodes;
        }
      } else if (device_kind == 2) {
        uint16_t count = br_u16(&r);
        param_devices_tmp = count > 0 ? malloc((size_t)count * sizeof(*param_devices_tmp)) : NULL;
        if (count > 0 && !param_devices_tmp) {
          if (srcs) free(srcs);
          goto fail_nodes;
        }
        for (int d = 0; d < count; d++) {
          uint32_t device_idx = br_u32(&r);
          if (device_idx >= n_strings) {
            free(param_devices_tmp);
            if (srcs) free(srcs);
            goto fail_nodes;
          }
          param_devices_tmp[d] = strings[device_idx];
        }
        param_arg_tmp.devices = param_devices_tmp;
        param_arg_tmp.n_devices = count;
        param_arg_tmp.device_is_tuple = true;
      } else if (device_kind != 0) {
        if (srcs) free(srcs);
        goto fail_nodes;
      }
      param_arg_tmp.addrspace = (PolyAddrSpace)br_u8(&r);
      param_arg_tmp.has_axis = br_u8(&r) != 0;
      param_arg_tmp.axis = br_i32(&r);
      param_arg_tmp.has_minmax = br_u8(&r) != 0;
      uint32_t name_idx = br_u32(&r);
      param_arg_tmp.name = name_idx < n_strings ? strings[name_idx] : NULL;
      param_arg_tmp.min_val = br_i64(&r);
      param_arg_tmp.max_val = br_i64(&r);
      arg.param = &param_arg_tmp;
      break;
    }
    case POLY_ARG_CALL_INFO: {
      if (br_remaining(&r) < 6) {
        if (srcs) free(srcs);
        goto fail_nodes;
      }
      uint32_t name_idx = br_u32(&r);
      call_info_tmp.name = name_idx < n_strings ? strings[name_idx] : NULL;
      if (name_idx != UINT32_MAX && !call_info_tmp.name) {
        if (srcs) free(srcs);
        goto fail_nodes;
      }
      call_info_tmp.precompile = br_u8(&r) != 0;
      call_info_tmp.precompile_backward = br_u8(&r) != 0;
      arg.call_info = &call_info_tmp;
      break;
    }
    case POLY_ARG_DTYPE:
      break;
    default:
      fprintf(stderr, "poly_ir_import: unknown arg kind %u at node %u\n", arg_kind, i);
      if (srcs) free(srcs);
      goto fail_nodes;
    }

    if (executable && !br_program_tag_arg(&r, tag_arg_kind, strings, n_strings, &tag_arg)) {
      fprintf(stderr, "poly_program_import: invalid tag metadata at node %u\n", i);
      goto cleanup_node_arg;
    }

    /* Create UOp -- restore tag to preserve BUFFER CSE-distinctness */
    PolyDType dtype = *dtype_table[dtype_idx];
    if (arg.kind == POLY_ARG_DTYPE) arg.dtype = dtype;
    u = (tag != 0 || tag_arg.kind != POLY_ARG_NONE)
            ? poly_uop_tagged_arg(ctx, (PolyOps)op_val, dtype, srcs, n_src, arg, tag, tag_arg)
            : poly_uop(ctx, (PolyOps)op_val, dtype, srcs, n_src, arg);

  cleanup_node_arg:
    /* UOp construction copies metadata; failed partial reads have no arena
     * owner. Both paths must release every temporary axis/table allocation. */
    if (arg.kind == POLY_ARG_INT_TUPLE && arg.int_tuple.vals)
      free(arg.int_tuple.vals);
    else if (arg.kind == POLY_ARG_BIGINT && arg.bigint.limbs)
      free((void *)arg.bigint.limbs);
    else if (arg.kind == POLY_ARG_RANGE && arg.range.extra)
      free(arg.range.extra);
    else if (arg.kind == POLY_ARG_STRING_TUPLE && arg.string_tuple.vals)
      free((void *)arg.string_tuple.vals);
    else if (arg.kind == POLY_ARG_TENSOR_CORE)
      for (int d = 0; d < 3; d++)
        free(wmma_upcast_axes_tmp[d]);
    else if (arg.kind == POLY_ARG_KERNEL_INFO) {
      free_opts(kernel_opts_to_apply_tmp, kernel_info_tmp.n_opts_to_apply);
      free_opts(kernel_applied_opts_tmp, kernel_info_tmp.n_applied_opts);
      free(kernel_axis_types_tmp);
    }
    if (tag_arg.kind == POLY_ARG_INT_TUPLE && tag_arg.int_tuple.vals) free(tag_arg.int_tuple.vals);
    if (bufferize_devices_tmp) free(bufferize_devices_tmp);
    if (allreduce_devices_tmp) free(allreduce_devices_tmp);
    if (param_devices_tmp) free(param_devices_tmp);

    if (!u) {
      free(srcs);
      goto fail_nodes;
    }

    if (tag > 0) poly_ctx_reserve_buf_tag(ctx, tag);
    if (op_val == POLY_OP_UNIQUE && arg.kind == POLY_ARG_INT)
      poly_ctx_reserve_unique_id(ctx, arg.i);

    nodes[i] = (PolyUOp *)u;
    if (srcs) free(srcs);
  }

  /* Interface table */
  out->n_bufs = (int)n_entries;
  out->bufs = calloc(n_entries, sizeof(PolyIrBufEntry));
  for (uint32_t i = 0; i < n_entries; i++) {
    if (br_remaining(&r) < 12) goto fail_bufs;
    uint32_t name_idx = br_u32(&r);
    uint8_t role = br_u8(&r);
    uint8_t iface_flags = br_u8(&r);
    br_u8(&r);
    br_u8(&r); /* padding */
    uint32_t node_idx = br_u32(&r);
    uint16_t ndim = br_u16(&r);
    br_u16(&r); /* padding */

    if ((executable && name_idx >= n_strings) || role > POLY_IR_ROLE_AUX || node_idx >= n_nodes ||
        ndim > POLY_IR_MAX_DIMS || br_remaining(&r) < (int)ndim * 8)
      goto fail_bufs;

    out->bufs[i].name = (name_idx < n_strings) ? strdup(strings[name_idx]) : strdup("");
    out->bufs[i].role = role;
    out->bufs[i].trainable_set = (iface_flags & 2) != 0;
    out->bufs[i].trainable =
        out->bufs[i].trainable_set ? ((iface_flags & 1) != 0) : (role == POLY_IR_ROLE_PARAM);
    out->bufs[i].buffer = nodes[node_idx];
    out->bufs[i].ndim = ndim;
    for (int d = 0; d < ndim; d++)
      out->bufs[i].shape[d] = br_i64(&r);
    if (!ir_interface_shape_valid(&out->bufs[i])) goto fail_bufs;
  }

  /* Entrypoint table */
  out->n_entrypoints = (int)n_entrypts;
  out->entrypoints = calloc(n_entrypts, sizeof(PolyIrEntrypoint));
  for (uint32_t i = 0; i < n_entrypts; i++) {
    if (br_remaining(&r) < 8) goto fail_ep;
    uint32_t name_idx = br_u32(&r);
    uint32_t node_idx = br_u32(&r);
    if (name_idx >= n_strings || node_idx >= n_nodes ||
        (executable && nodes[node_idx]->op != POLY_OP_LINEAR))
      goto fail_ep;
    out->entrypoints[i].name = (name_idx < n_strings) ? strdup(strings[name_idx]) : strdup("");
    out->entrypoints[i].sink = (node_idx < n_nodes) ? nodes[node_idx] : NULL;

    if (br_remaining(&r) < 12) goto fail_ep;
    out->entrypoints[i].flags = br_u32(&r);
    uint16_t n_inputs = br_u16(&r);
    uint16_t n_outputs = br_u16(&r);
    uint32_t objective_idx = br_u32(&r);
    out->entrypoints[i].n_inputs = n_inputs;
    out->entrypoints[i].n_outputs = n_outputs;
    if (objective_idx != UINT32_MAX) {
      if (executable && objective_idx >= n_strings) goto fail_ep;
      out->entrypoints[i].objective =
          (objective_idx < n_strings) ? strdup(strings[objective_idx]) : strdup("");
    }
    if (n_inputs > 0) {
      char **inputs = calloc(n_inputs, sizeof(char *));
      if (!inputs) goto fail_ep;
      out->entrypoints[i].inputs = (const char **)inputs;
      for (uint16_t j = 0; j < n_inputs; j++) {
        if (br_remaining(&r) < 4) goto fail_ep;
        uint32_t idx = br_u32(&r);
        if (executable && idx >= n_strings) goto fail_ep;
        inputs[j] = (idx < n_strings) ? strdup(strings[idx]) : strdup("");
      }
    }
    if (n_outputs > 0) {
      char **outputs = calloc(n_outputs, sizeof(char *));
      if (!outputs) goto fail_ep;
      out->entrypoints[i].outputs = (const char **)outputs;
      for (uint16_t j = 0; j < n_outputs; j++) {
        if (br_remaining(&r) < 4) goto fail_ep;
        uint32_t idx = br_u32(&r);
        if (executable && idx >= n_strings) goto fail_ep;
        outputs[j] = (idx < n_strings) ? strdup(strings[idx]) : strdup("");
      }
    }
  }

  /* Exact logical placement modules. */
  out->n_modules = (int)n_modules;
  out->modules = calloc(n_modules, sizeof(PolyIrModule));
  if (n_modules > 0 && !out->modules) goto fail_modules;
  for (uint32_t i = 0; i < n_modules; i++) {
    if (br_remaining(&r) < 12) goto fail_modules;
    uint32_t name_idx = br_u32(&r);
    uint32_t n_inputs = br_u32(&r);
    if (name_idx >= n_strings || n_inputs > INT32_MAX ||
        br_remaining(&r) < (int64_t)n_inputs * 4 + 4)
      goto fail_modules;
    out->modules[i].name = strdup(strings[name_idx]);
    out->modules[i].n_inputs = (int)n_inputs;
    out->modules[i].inputs = n_inputs > 0 ? calloc(n_inputs, sizeof(PolyUOp *)) : NULL;
    if (!out->modules[i].name || (n_inputs > 0 && !out->modules[i].inputs)) goto fail_modules;
    for (uint32_t j = 0; j < n_inputs; j++) {
      uint32_t node_idx = br_u32(&r);
      if (node_idx >= n_nodes) goto fail_modules;
      out->modules[i].inputs[j] = nodes[node_idx];
    }
    uint32_t output_idx = br_u32(&r);
    if (output_idx >= n_nodes) goto fail_modules;
    out->modules[i].output = nodes[output_idx];
  }

  if (executable && r.pos != r.len) {
    fprintf(stderr, "poly_program_import: trailing bytes\n");
    goto fail_modules;
  }

  out->ctx = ctx;

  /* Cleanup temp arrays */
  for (uint32_t i = 0; i < n_strings; i++)
    free(strings[i]);
  free(strings);
  free(nodes);
  return 0;

fail_modules:
  for (int i = 0; i < out->n_modules; i++)
    free_ir_module(&out->modules[i]);
  free(out->modules);
fail_ep:
  for (int i = 0; i < out->n_entrypoints; i++)
    free_ir_entrypoint(&out->entrypoints[i]);
  free(out->entrypoints);
fail_bufs:
  for (int i = 0; i < out->n_bufs; i++)
    free((char *)out->bufs[i].name);
  free(out->bufs);
fail_nodes:
  poly_ctx_destroy(ctx);
  free(nodes);
fail_strings:
  for (uint32_t i = 0; i < n_strings; i++)
    free(strings[i]);
  free(strings);
  return -1;
}

int poly_ir_import(const uint8_t *data, int len, PolyIrSpec *out) {
  return poly_graph_import(data, len, out, false);
}

int poly_program_graph_import(const uint8_t *data, int len, PolyIrSpec *out) {
  return poly_graph_import(data, len, out, true);
}

void poly_ir_spec_free(PolyIrSpec *spec) {
  if (!spec) return;
  for (int i = 0; i < spec->n_bufs; i++)
    free((char *)spec->bufs[i].name);
  free(spec->bufs);
  for (int i = 0; i < spec->n_entrypoints; i++)
    free_ir_entrypoint(&spec->entrypoints[i]);
  free(spec->entrypoints);
  for (int i = 0; i < spec->n_modules; i++)
    free_ir_module(&spec->modules[i]);
  free(spec->modules);
  spec->bufs = NULL;
  spec->entrypoints = NULL;
  spec->modules = NULL;
  spec->n_bufs = 0;
  spec->n_entrypoints = 0;
  spec->n_modules = 0;
}
