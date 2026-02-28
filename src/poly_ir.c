/*
 * poly_ir.c -- Binary IR codec for tensor-level UOp graphs
 *
 * poly.ir.uops@1 format:
 *   Header (32 bytes)
 *   String table (variable)
 *   Node table (variable, strict toposort order)
 *   Interface table (named buffers with roles)
 *   Entrypoint table (named SINKs)
 */

#define _POSIX_C_SOURCE 200809L
#include "poly_ir.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ── Byte helpers ───────────────────────────────────────────────────── */

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
  for (int i = 0; i < 4; i++) b->data[b->len++] = (v >> (i * 8)) & 0xFF;
}

static void bb_i32(ByteBuf *b, int32_t v) { bb_u32(b, (uint32_t)v); }

static void bb_i64(ByteBuf *b, int64_t v) {
  bb_ensure(b, 8);
  uint64_t u = (uint64_t)v;
  for (int i = 0; i < 8; i++) b->data[b->len++] = (u >> (i * 8)) & 0xFF;
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

/* ── Read helpers ───────────────────────────────────────────────────── */

typedef struct {
  const uint8_t *data;
  int len;
  int pos;
} ByteReader;

static int br_remaining(ByteReader *r) { return r->len - r->pos; }

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
  for (int i = 0; i < 4; i++) v |= (uint32_t)r->data[r->pos++] << (i * 8);
  return v;
}

static int32_t br_i32(ByteReader *r) { return (int32_t)br_u32(r); }

static int64_t br_i64(ByteReader *r) {
  if (r->pos + 8 > r->len) return 0;
  uint64_t v = 0;
  for (int i = 0; i < 8; i++) v |= (uint64_t)r->data[r->pos++] << (i * 8);
  return (int64_t)v;
}

static double br_f64(ByteReader *r) {
  int64_t i = br_i64(r);
  double d;
  memcpy(&d, &i, sizeof(d));
  return d;
}

/* ── Magic ──────────────────────────────────────────────────────────── */

#define IR_MAGIC 0x52494750  /* "PGIR" LE */
#define IR_VERSION 1

/* ── Dtype index table ──────────────────────────────────────────────── */

#define N_DTYPES 15

static const PolyDType *dtype_table[N_DTYPES] = {
  &POLY_VOID, &POLY_BOOL, &POLY_INT8, &POLY_UINT8,
  &POLY_INT16, &POLY_UINT16, &POLY_INT32, &POLY_UINT32,
  &POLY_INT64, &POLY_UINT64, &POLY_FLOAT16, &POLY_BFLOAT16,
  &POLY_FLOAT32, &POLY_FLOAT64, &POLY_INDEX,
};

static int dtype_to_index(PolyDType dt) {
  for (int i = 0; i < N_DTYPES; i++)
    if (poly_dtype_eq(dt, *dtype_table[i])) return i;
  return -1;
}

/* ── String table builder ───────────────────────────────────────────── */

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
  for (int i = 0; i < st->n; i++) free(st->strs[i]);
  free(st->strs);
}

/* ── Export ──────────────────────────────────────────────────────────── */

uint8_t *poly_ir_export(const PolyIrSpec *spec, int *out_len) {
  *out_len = 0;
  if (!spec || !spec->ctx) return NULL;

  /* Collect all SINKs from entrypoints */
  int n_sinks = spec->n_entrypoints;
  if (n_sinks == 0) {
    fprintf(stderr, "poly_ir_export: no entrypoints\n");
    return NULL;
  }

  /* Collect all nodes via toposort. For multiple entrypoints,
   * toposort each sink and merge (dedup by pointer). */
  int n_nodes = 0;
  PolyUOp **topo = NULL;
  int topo_is_heap = 0;  /* whether topo is malloc'd (needs free) */

  if (n_sinks == 1) {
    topo = poly_toposort(spec->ctx, spec->entrypoints[0].sink, &n_nodes);
  } else {
    /* Merge toposorts from all sinks */
    int total_cap = 0;
    int *counts = malloc(n_sinks * sizeof(int));
    PolyUOp ***per_sink = malloc(n_sinks * sizeof(PolyUOp **));
    for (int i = 0; i < n_sinks; i++) {
      per_sink[i] = poly_toposort(spec->ctx, spec->entrypoints[i].sink, &counts[i]);
      total_cap += counts[i];
    }

    /* Merge and dedup */
    PolyUOp **merged = malloc(total_cap * sizeof(PolyUOp *));
    int merged_n = 0;
    for (int i = 0; i < n_sinks; i++) {
      for (int j = 0; j < counts[i]; j++) {
        PolyUOp *u = per_sink[i][j];
        int dup = 0;
        for (int k = 0; k < merged_n; k++)
          if (merged[k] == u) { dup = 1; break; }
        if (!dup) merged[merged_n++] = u;
      }
    }

    topo = merged;
    n_nodes = merged_n;
    topo_is_heap = 1;
    free(counts);
    free(per_sink);
  }

  if (!topo || n_nodes == 0) {
    fprintf(stderr, "poly_ir_export: toposort failed\n");
    if (topo_is_heap) free(topo);
    return NULL;
  }

  /* Build node index map (UOp pointer -> index) */
  /* Use a simple linear scan (good enough for small-medium graphs) */
  typedef struct { PolyUOp *uop; uint32_t idx; } NodeMapEntry;
  NodeMapEntry *node_map = malloc(n_nodes * sizeof(NodeMapEntry));
  for (int i = 0; i < n_nodes; i++) {
    node_map[i].uop = topo[i];
    node_map[i].idx = (uint32_t)i;
  }

  /* Helper: find index of a UOp */
  #define FIND_IDX(u) ({ \
    uint32_t _idx = UINT32_MAX; \
    for (int _i = 0; _i < n_nodes; _i++) \
      if (node_map[_i].uop == (u)) { _idx = node_map[_i].idx; break; } \
    _idx; \
  })

  /* Validate: no pointer/vector dtypes */
  for (int i = 0; i < n_nodes; i++) {
    PolyDType dt = topo[i]->dtype;
    if (dt.is_ptr || dt.count > 1) {
      fprintf(stderr, "poly_ir_export: node %d has pointer/vector dtype (IR v1)\n", i);
      free(node_map);
      if (topo_is_heap) free(topo);
      return NULL;
    }
    if (dtype_to_index(dt) < 0) {
      fprintf(stderr, "poly_ir_export: node %d has unknown dtype '%s'\n",
              i, dt.name ? dt.name : "?");
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
    if (a.kind == POLY_ARG_STRING && a.str)
      st_add(&strings, a.str);
    if (a.kind == POLY_ARG_DEFINE_VAR && a.define_var.name)
      st_add(&strings, a.define_var.name);
  }
  /* Collect strings from interface + entrypoints */
  for (int i = 0; i < spec->n_bufs; i++)
    st_add(&strings, spec->bufs[i].name);
  for (int i = 0; i < spec->n_entrypoints; i++)
    st_add(&strings, spec->entrypoints[i].name);

  /* Compute flags */
  uint32_t flags = 0;
  for (int i = 0; i < spec->n_entrypoints; i++) {
    if (strcmp(spec->entrypoints[i].name, "loss") == 0)
      flags |= 1;  /* has_loss */
    if (strcmp(spec->entrypoints[i].name, "train_step") == 0)
      flags |= 2;  /* has_train_step */
  }

  /* ── Write binary ────────────────────────────────────────────── */

  ByteBuf buf;
  bb_init(&buf);

  /* Header (32 bytes) */
  bb_u32(&buf, IR_MAGIC);
  bb_u32(&buf, IR_VERSION);
  bb_u32(&buf, flags);
  bb_u32(&buf, (uint32_t)n_nodes);
  bb_u32(&buf, (uint32_t)strings.n);
  bb_u32(&buf, (uint32_t)spec->n_bufs);
  bb_u32(&buf, (uint32_t)spec->n_entrypoints);
  bb_u32(&buf, 0);  /* reserved */

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
    bb_u8(&buf, 0);  /* padding */

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
    case POLY_ARG_NONE: break;
    case POLY_ARG_INT: bb_i64(&buf, u->arg.i); break;
    case POLY_ARG_FLOAT: bb_f64(&buf, u->arg.f); break;
    case POLY_ARG_BOOL: bb_u8(&buf, u->arg.b ? 1 : 0); break;
    case POLY_ARG_INT_TUPLE:
      bb_u16(&buf, (uint16_t)u->arg.int_tuple.n);
      for (int t = 0; t < u->arg.int_tuple.n; t++)
        bb_i64(&buf, u->arg.int_tuple.vals[t]);
      break;
    case POLY_ARG_PAIR_TUPLE:
      bb_u16(&buf, (uint16_t)u->arg.pair_tuple.n);
      for (int t = 0; t < u->arg.pair_tuple.n; t++) {
        bb_i64(&buf, u->arg.pair_tuple.pairs[t][0]);
        bb_i64(&buf, u->arg.pair_tuple.pairs[t][1]);
      }
      break;
    case POLY_ARG_STRING:
      bb_u32(&buf, st_add(&strings, u->arg.str));
      break;
    case POLY_ARG_OPS:
      bb_u16(&buf, (uint16_t)u->arg.ops);
      break;
    case POLY_ARG_REDUCE_AXIS:
      bb_u16(&buf, (uint16_t)u->arg.reduce_axis.op);
      bb_u16(&buf, (uint16_t)u->arg.reduce_axis.n);
      for (int t = 0; t < u->arg.reduce_axis.n; t++)
        bb_i64(&buf, u->arg.reduce_axis.axes[t]);
      break;
    case POLY_ARG_RANGE:
      bb_i64(&buf, u->arg.range.axis_id);
      bb_u8(&buf, (uint8_t)u->arg.range.axis_type);
      bb_u16(&buf, (uint16_t)u->arg.range.n_extra);
      for (int t = 0; t < u->arg.range.n_extra; t++)
        bb_i64(&buf, u->arg.range.extra[t]);
      break;
    case POLY_ARG_DEFINE_VAR:
      bb_u32(&buf, st_add(&strings, u->arg.define_var.name));
      bb_i64(&buf, u->arg.define_var.min_val);
      bb_i64(&buf, u->arg.define_var.max_val);
      break;
    case POLY_ARG_INVALID: break;
    }
  }

  /* Interface table */
  for (int i = 0; i < spec->n_bufs; i++) {
    bb_u32(&buf, st_add(&strings, spec->bufs[i].name));
    bb_u8(&buf, spec->bufs[i].role);
    bb_u8(&buf, 0); bb_u8(&buf, 0); bb_u8(&buf, 0);  /* padding */
    uint32_t nidx = FIND_IDX(spec->bufs[i].buffer);
    bb_u32(&buf, nidx);
    bb_u16(&buf, (uint16_t)spec->bufs[i].ndim);
    bb_u16(&buf, 0);  /* padding */
    for (int d = 0; d < spec->bufs[i].ndim; d++)
      bb_i64(&buf, spec->bufs[i].shape[d]);
  }

  /* Entrypoint table */
  for (int i = 0; i < spec->n_entrypoints; i++) {
    bb_u32(&buf, st_add(&strings, spec->entrypoints[i].name));
    uint32_t nidx = FIND_IDX(spec->entrypoints[i].sink);
    bb_u32(&buf, nidx);
  }

  #undef FIND_IDX

  free(node_map);
  if (topo_is_heap) free(topo);
  st_free(&strings);

  *out_len = buf.len;
  return buf.data;
}

/* ── Import ──────────────────────────────────────────────────────────── */

int poly_ir_import(const uint8_t *data, int len, PolyIrSpec *out) {
  memset(out, 0, sizeof(PolyIrSpec));

  ByteReader r = { data, len, 0 };

  /* Header */
  if (br_remaining(&r) < 32) {
    fprintf(stderr, "poly_ir_import: data too short for header\n");
    return -1;
  }

  uint32_t magic = br_u32(&r);
  if (magic != IR_MAGIC) {
    fprintf(stderr, "poly_ir_import: bad magic 0x%08x\n", magic);
    return -1;
  }
  uint32_t version = br_u32(&r);
  if (version != IR_VERSION) {
    fprintf(stderr, "poly_ir_import: unsupported version %u\n", version);
    return -1;
  }
  /*uint32_t flags =*/ br_u32(&r);  /* flags (informational) */
  uint32_t n_nodes = br_u32(&r);
  uint32_t n_strings = br_u32(&r);
  uint32_t n_entries = br_u32(&r);
  uint32_t n_entrypts = br_u32(&r);
  /*uint32_t reserved =*/ br_u32(&r);

  /* String table */
  char **strings = calloc(n_strings, sizeof(char *));
  for (uint32_t i = 0; i < n_strings; i++) {
    if (br_remaining(&r) < 2) goto fail_strings;
    uint16_t slen = br_u16(&r);
    if (br_remaining(&r) < slen) goto fail_strings;
    strings[i] = malloc(slen + 1);
    memcpy(strings[i], r.data + r.pos, slen);
    strings[i][slen] = '\0';
    r.pos += slen;
  }

  /* Create context */
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp **nodes = calloc(n_nodes, sizeof(PolyUOp *));

  /* Node table */
  for (uint32_t i = 0; i < n_nodes; i++) {
    if (br_remaining(&r) < 10) goto fail_nodes;

    uint16_t op_val = br_u16(&r);
    uint8_t dtype_idx = br_u8(&r);
    int32_t tag = br_i32(&r);
    uint16_t n_src = br_u16(&r);
    uint8_t arg_kind = br_u8(&r);
    /*uint8_t pad =*/ br_u8(&r);

    if (op_val >= POLY_OP_COUNT) {
      fprintf(stderr, "poly_ir_import: invalid op %u at node %u\n", op_val, i);
      goto fail_nodes;
    }
    if (dtype_idx >= N_DTYPES) {
      fprintf(stderr, "poly_ir_import: invalid dtype index %u at node %u\n",
              dtype_idx, i);
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
    memset(&arg, 0, sizeof(arg));
    arg.kind = (PolyArgKind)arg_kind;

    switch (arg.kind) {
    case POLY_ARG_NONE: break;
    case POLY_ARG_INT:
      arg.i = br_i64(&r);
      break;
    case POLY_ARG_FLOAT:
      arg.f = br_f64(&r);
      break;
    case POLY_ARG_BOOL:
      arg.b = br_u8(&r) != 0;
      break;
    case POLY_ARG_INT_TUPLE: {
      uint16_t count = br_u16(&r);
      int64_t *vals = malloc(count * sizeof(int64_t));
      for (int t = 0; t < count; t++) vals[t] = br_i64(&r);
      arg.int_tuple.vals = vals;
      arg.int_tuple.n = count;
      break;
    }
    case POLY_ARG_PAIR_TUPLE: {
      uint16_t count = br_u16(&r);
      int64_t (*pairs)[2] = malloc(count * sizeof(int64_t[2]));
      for (int t = 0; t < count; t++) {
        pairs[t][0] = br_i64(&r);
        pairs[t][1] = br_i64(&r);
      }
      arg.pair_tuple.pairs = pairs;
      arg.pair_tuple.n = count;
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
    case POLY_ARG_OPS:
      arg.ops = (PolyOps)br_u16(&r);
      break;
    case POLY_ARG_REDUCE_AXIS: {
      arg.reduce_axis.op = (PolyOps)br_u16(&r);
      uint16_t n_axes = br_u16(&r);
      int64_t *axes = malloc(n_axes * sizeof(int64_t));
      for (int t = 0; t < n_axes; t++) axes[t] = br_i64(&r);
      arg.reduce_axis.axes = axes;
      arg.reduce_axis.n = n_axes;
      break;
    }
    case POLY_ARG_RANGE: {
      arg.range.axis_id = br_i64(&r);
      arg.range.axis_type = (PolyAxisType)br_u8(&r);
      uint16_t n_extra = br_u16(&r);
      if (n_extra > 0) {
        int64_t *extra = malloc(n_extra * sizeof(int64_t));
        for (int t = 0; t < n_extra; t++) extra[t] = br_i64(&r);
        arg.range.extra = extra;
      } else {
        arg.range.extra = NULL;
      }
      arg.range.n_extra = n_extra;
      break;
    }
    case POLY_ARG_DEFINE_VAR: {
      uint32_t name_idx = br_u32(&r);
      arg.define_var.name = (name_idx < n_strings) ? strings[name_idx] : "";
      arg.define_var.min_val = br_i64(&r);
      arg.define_var.max_val = br_i64(&r);
      break;
    }
    case POLY_ARG_INVALID: break;
    default:
      fprintf(stderr, "poly_ir_import: unknown arg kind %u at node %u\n",
              arg_kind, i);
      if (srcs) free(srcs);
      goto fail_nodes;
    }

    /* Create UOp (with tag) -- poly_uop copies arg data into arena */
    PolyUOp *u = poly_uop(ctx, (PolyOps)op_val, *dtype_table[dtype_idx],
                           srcs, n_src, arg);

    /* Free temporary malloc'd arg buffers (arena has its own copy now) */
    if (arg.kind == POLY_ARG_INT_TUPLE && arg.int_tuple.vals)
      free(arg.int_tuple.vals);
    else if (arg.kind == POLY_ARG_PAIR_TUPLE && arg.pair_tuple.pairs)
      free(arg.pair_tuple.pairs);
    else if (arg.kind == POLY_ARG_REDUCE_AXIS && arg.reduce_axis.axes)
      free(arg.reduce_axis.axes);
    else if (arg.kind == POLY_ARG_RANGE && arg.range.extra)
      free(arg.range.extra);

    if (tag != 0)
      ((PolyUOp *)u)->tag = tag;

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
    br_u8(&r); br_u8(&r); br_u8(&r);  /* padding */
    uint32_t node_idx = br_u32(&r);
    uint16_t ndim = br_u16(&r);
    br_u16(&r);  /* padding */

    out->bufs[i].name = (name_idx < n_strings) ? strdup(strings[name_idx]) : strdup("");
    out->bufs[i].role = role;
    out->bufs[i].buffer = (node_idx < n_nodes) ? nodes[node_idx] : NULL;
    out->bufs[i].ndim = ndim;
    for (int d = 0; d < ndim && d < 8; d++)
      out->bufs[i].shape[d] = br_i64(&r);
  }

  /* Entrypoint table */
  out->n_entrypoints = (int)n_entrypts;
  out->entrypoints = calloc(n_entrypts, sizeof(PolyIrEntrypoint));
  for (uint32_t i = 0; i < n_entrypts; i++) {
    if (br_remaining(&r) < 8) goto fail_ep;
    uint32_t name_idx = br_u32(&r);
    uint32_t node_idx = br_u32(&r);
    out->entrypoints[i].name = (name_idx < n_strings) ? strdup(strings[name_idx]) : strdup("");
    out->entrypoints[i].sink = (node_idx < n_nodes) ? nodes[node_idx] : NULL;
  }

  out->ctx = ctx;

  /* Cleanup temp arrays */
  for (uint32_t i = 0; i < n_strings; i++) free(strings[i]);
  free(strings);
  free(nodes);
  return 0;

fail_ep:
  for (int i = 0; i < out->n_entrypoints; i++)
    free((char *)out->entrypoints[i].name);
  free(out->entrypoints);
fail_bufs:
  for (int i = 0; i < out->n_bufs; i++)
    free((char *)out->bufs[i].name);
  free(out->bufs);
fail_nodes:
  poly_ctx_destroy(ctx);
  free(nodes);
fail_strings:
  for (uint32_t i = 0; i < n_strings; i++) free(strings[i]);
  free(strings);
  return -1;
}

void poly_ir_spec_free(PolyIrSpec *spec) {
  if (!spec) return;
  for (int i = 0; i < spec->n_bufs; i++)
    free((char *)spec->bufs[i].name);
  free(spec->bufs);
  for (int i = 0; i < spec->n_entrypoints; i++)
    free((char *)spec->entrypoints[i].name);
  free(spec->entrypoints);
  spec->bufs = NULL;
  spec->entrypoints = NULL;
  spec->n_bufs = 0;
  spec->n_entrypoints = 0;
}
