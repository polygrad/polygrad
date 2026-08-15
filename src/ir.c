/*
 * poly_ir.c -- Binary IR codec for tensor-level UOp graphs
 *
 * poly.ir.uops@8 format:
 *   Header (32 bytes)
 *   String table (variable)
 *   Node table (variable, strict toposort order; scalar dtype ID + vector count)
 *   Interface table (named buffers with roles)
 *   Entrypoint table (named SINKs plus v2+ ABI metadata)
 *
 * Import remains backward-compatible with v1 payloads (entrypoint name + SINK)
 * and v2-v7 payloads (including integer and scalar-string device metadata).
 */

#define _POSIX_C_SOURCE 200809L
#include "ir.h"
#include "ctx.h"
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Byte helpers */

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

/* Magic */

#define IR_MAGIC 0x52494750 /* "PGIR" LE */
#define IR_VERSION 9
#define IR_MIN_VERSION 1

/* Dtype index table */

#define N_DTYPES 15

static const PolyDType *dtype_table[N_DTYPES] = {
    &POLY_VOID,    &POLY_BOOL,     &POLY_INT8,    &POLY_UINT8,   &POLY_INT16,
    &POLY_UINT16,  &POLY_INT32,    &POLY_UINT32,  &POLY_INT64,   &POLY_UINT64,
    &POLY_FLOAT16, &POLY_BFLOAT16, &POLY_FLOAT32, &POLY_FLOAT64, &POLY_INDEX,
};

static int dtype_to_index(PolyDType dt) {
  dt = poly_dtype_scalar(dt);
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

/* Export */

uint8_t *poly_ir_export(const PolyIrSpec *spec, int *out_len) {
  *out_len = 0;
  if (!spec || !spec->ctx) return NULL;

  if (spec->n_entrypoints == 0) {
    fprintf(stderr, "poly_ir_export: no entrypoints\n");
    return NULL;
  }

  /* Collect all nodes via toposort. Entrypoint sinks define executable graphs;
   * interface BUFFERs are also roots because package/checkpoint state can be
   * named without being read by a normal inference/loss entrypoint. */
  int n_nodes = 0;
  PolyUOp **topo = NULL;
  int topo_is_heap = 1;

  int n_roots = spec->n_entrypoints + spec->n_bufs;
  int total_cap = 0;
  int *counts = calloc((size_t)n_roots, sizeof(int));
  PolyUOp ***per_root = calloc((size_t)n_roots, sizeof(PolyUOp **));
  if (!counts || !per_root) {
    free(counts);
    free(per_root);
    return NULL;
  }

  PolyScratchMark scratch = poly_ctx_scratch_mark(spec->ctx);
  int ri = 0;
  for (int i = 0; i < spec->n_entrypoints; i++, ri++) {
    per_root[ri] = poly_toposort_scratch(spec->ctx, spec->entrypoints[i].sink, &counts[ri]);
    if (!per_root[ri] && counts[ri] != 0) {
      poly_ctx_scratch_rewind(spec->ctx, scratch);
      free(counts);
      free(per_root);
      return NULL;
    }
    total_cap += counts[ri];
  }
  for (int i = 0; i < spec->n_bufs; i++, ri++) {
    per_root[ri] = poly_toposort_scratch(spec->ctx, spec->bufs[i].buffer, &counts[ri]);
    if (!per_root[ri] && counts[ri] != 0) {
      poly_ctx_scratch_rewind(spec->ctx, scratch);
      free(counts);
      free(per_root);
      return NULL;
    }
    total_cap += counts[ri];
  }

  PolyUOp **merged = total_cap > 0 ? malloc((size_t)total_cap * sizeof(PolyUOp *)) : NULL;
  if (total_cap > 0 && !merged) {
    poly_ctx_scratch_rewind(spec->ctx, scratch);
    free(counts);
    free(per_root);
    return NULL;
  }
  int merged_n = 0;
  for (int i = 0; i < n_roots; i++) {
    for (int j = 0; j < counts[i]; j++) {
      PolyUOp *u = per_root[i][j];
      int dup = 0;
      for (int k = 0; k < merged_n; k++)
        if (merged[k] == u) {
          dup = 1;
          break;
        }
      if (!dup) merged[merged_n++] = u;
    }
  }
  poly_ctx_scratch_rewind(spec->ctx, scratch);
  topo = merged;
  n_nodes = merged_n;
  free(counts);
  free(per_root);

  if (!topo || n_nodes == 0) {
    fprintf(stderr, "poly_ir_export: toposort failed\n");
    if (topo_is_heap) free(topo);
    return NULL;
  }

  /* Build node index map (UOp pointer -> index) */
  /* Use a simple linear scan (good enough for small-medium graphs) */
  typedef struct {
    PolyUOp *uop;
    uint32_t idx;
  } NodeMapEntry;
  NodeMapEntry *node_map = malloc(n_nodes * sizeof(NodeMapEntry));
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

  /* Tensor/package IR keeps vector dtypes (notably shape STACK[weakintN])
   * as scalar dtype ID + existing PolyDType.count. Pointer dtypes remain
   * lowered/execution state and are not portable tensor IR. */
  for (int i = 0; i < n_nodes; i++) {
    PolyDType dt = topo[i]->dtype;
    if (dt.is_ptr || dt.count == 0) {
      fprintf(stderr, "poly_ir_export: node %d has unsupported pointer/empty dtype\n", i);
      free(node_map);
      if (topo_is_heap) free(topo);
      return NULL;
    }
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
    if (a.kind == POLY_ARG_STRING && a.str) st_add(&strings, a.str);
    if (a.kind == POLY_ARG_STRING_TUPLE)
      for (int j = 0; j < a.string_tuple.n; j++)
        st_add(&strings, a.string_tuple.vals[j]);
    if (a.kind == POLY_ARG_DEFINE_VAR && a.define_var.name) st_add(&strings, a.define_var.name);
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
    if (a.kind == POLY_ARG_TENSOR_CORE && a.tensor_core.name)
      st_add(&strings, a.tensor_core.name);
    if (a.kind == POLY_ARG_PARAM && a.param && a.param->name)
      st_add(&strings, a.param->name);
    if (a.kind == POLY_ARG_PARAM && a.param && a.param->device) st_add(&strings, a.param->device);
    if (a.kind == POLY_ARG_PARAM && a.param && a.param->device_is_tuple)
      for (int j = 0; j < a.param->n_devices; j++)
        st_add(&strings, a.param->devices[j]);
    if (a.kind == POLY_ARG_CALL_INFO && a.call_info && a.call_info->name)
      st_add(&strings, a.call_info->name);
  }
  /* Collect strings from interface + entrypoints */
  for (int i = 0; i < spec->n_bufs; i++)
    st_add(&strings, spec->bufs[i].name);
  for (int i = 0; i < spec->n_entrypoints; i++)
    st_add_entrypoint_strings(&strings, &spec->entrypoints[i]);

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
  bb_u32(&buf, IR_MAGIC);
  bb_u32(&buf, IR_VERSION);
  bb_u32(&buf, flags);
  bb_u32(&buf, (uint32_t)n_nodes);
  bb_u32(&buf, (uint32_t)strings.n);
  bb_u32(&buf, (uint32_t)spec->n_bufs);
  bb_u32(&buf, (uint32_t)spec->n_entrypoints);
  bb_u32(&buf, 0); /* reserved */

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
    bb_u16(&buf, u->dtype.count);
    bb_i32(&buf, u->tag);
    bb_u16(&buf, u->n_src);
    bb_u8(&buf, (uint8_t)u->arg.kind);
    bb_u8(&buf, 0); /* padding */

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
    case POLY_ARG_STRING_TUPLE:
      bb_u16(&buf, (uint16_t)u->arg.string_tuple.n);
      for (int t = 0; t < u->arg.string_tuple.n; t++)
        bb_u32(&buf, st_add(&strings, u->arg.string_tuple.vals[t]));
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
      bb_u32(&buf, st_add(&strings, u->arg.tensor_core.name));
      for (int d = 0; d < 3; d++) bb_i64(&buf, u->arg.tensor_core.dims[d]);
      bb_i64(&buf, u->arg.tensor_core.threads);
      break;
    case POLY_ARG_PROGRAM_INFO:
      fprintf(stderr, "poly_ir_export: PROGRAM metadata is not exportable IR\n");
      free(node_map);
      if (topo_is_heap) free(topo);
      st_free(&strings);
      free(buf.data);
      return NULL;
    case POLY_ARG_BYTES:
      fprintf(stderr, "poly_ir_export: BINARY byte payloads are not exportable IR\n");
      free(node_map);
      if (topo_is_heap) free(topo);
      st_free(&strings);
      free(buf.data);
      return NULL;
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
      bb_u32(
          &buf, u->arg.param->name ? st_add(&strings, u->arg.param->name) : UINT32_MAX
      );
      bb_i64(&buf, u->arg.param->min_val);
      bb_i64(&buf, u->arg.param->max_val);
      break;
    case POLY_ARG_CALL_INFO:
      if (!u->arg.call_info || u->arg.call_info->has_grad_fxn ||
          u->arg.call_info->has_metadata || u->arg.call_info->has_aux) {
        fprintf(stderr, "poly_ir_export: unsupported CallInfo callback/metadata/aux\n");
        free(node_map);
        if (topo_is_heap) free(topo);
        st_free(&strings);
        free(buf.data);
        return NULL;
      }
      bb_u32(
          &buf, u->arg.call_info->name ? st_add(&strings, u->arg.call_info->name)
                                       : UINT32_MAX);
      bb_u8(&buf, u->arg.call_info->precompile ? 1 : 0);
      bb_u8(&buf, u->arg.call_info->precompile_backward ? 1 : 0);
      break;
    }
  }

  /* Interface table */
  for (int i = 0; i < spec->n_bufs; i++) {
    bb_u32(&buf, st_add(&strings, spec->bufs[i].name));
    bb_u8(&buf, spec->bufs[i].role);
    /* Interface flags live in the first historical padding byte so old
     * poly.ir.uops@1 payloads still import.
     *   bit0 = trainable value
     *   bit1 = trainability metadata present */
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

#undef FIND_IDX

  free(node_map);
  if (topo_is_heap) free(topo);
  st_free(&strings);

  *out_len = buf.len;
  return buf.data;
}

/* Import */

int poly_ir_import(const uint8_t *data, int len, PolyIrSpec *out) {
  memset(out, 0, sizeof(PolyIrSpec));

  ByteReader r = {data, len, 0};

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
  if (version < IR_MIN_VERSION || version > IR_VERSION) {
    fprintf(stderr, "poly_ir_import: unsupported version %u\n", version);
    return -1;
  }
  /*uint32_t flags =*/br_u32(&r); /* flags (informational) */
  uint32_t n_nodes = br_u32(&r);
  uint32_t n_strings = br_u32(&r);
  uint32_t n_entries = br_u32(&r);
  uint32_t n_entrypts = br_u32(&r);
  /*uint32_t reserved =*/br_u32(&r);

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
    int node_header_bytes = version >= 3 ? 13 : 11;
    if (br_remaining(&r) < node_header_bytes) goto fail_nodes;

    uint16_t op_val = br_u16(&r);
    uint8_t dtype_idx = br_u8(&r);
    uint16_t dtype_count = version >= 3 ? br_u16(&r) : 1;
    int32_t tag = br_i32(&r);
    uint16_t n_src = br_u16(&r);
    uint8_t arg_kind = br_u8(&r);
    /*uint8_t pad =*/br_u8(&r);

    if (op_val >= POLY_OP_COUNT) {
      fprintf(stderr, "poly_ir_import: invalid op %u at node %u\n", op_val, i);
      goto fail_nodes;
    }
    if (dtype_idx >= N_DTYPES) {
      fprintf(stderr, "poly_ir_import: invalid dtype index %u at node %u\n", dtype_idx, i);
      goto fail_nodes;
    }
    if (dtype_count == 0 || (dtype_idx == 0 && dtype_count != 1)) {
      fprintf(
          stderr, "poly_ir_import: invalid dtype count %u for dtype index %u at node %u\n",
          dtype_count, dtype_idx, i
      );
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
    PolyParamArg param_arg_tmp;
    PolyCallInfo call_info_tmp;
    const char **param_devices_tmp = NULL;
    const char **bufferize_devices_tmp = NULL;
    memset(&param_arg_tmp, 0, sizeof(param_arg_tmp));
    memset(&call_info_tmp, 0, sizeof(call_info_tmp));

    switch (arg.kind) {
    case POLY_ARG_NONE:
      break;
    case POLY_ARG_INT:
      arg.i = br_i64(&r);
      break;
    case POLY_ARG_BIGINT: {
      if (version < 4 || br_remaining(&r) < 5) {
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
      for (uint32_t limb = 0; limb < n_limbs; limb++) limbs[limb] = br_u32(&r);
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
    case POLY_ARG_PAIR_TUPLE: {
      uint16_t count = br_u16(&r);
      int64_t(*pairs)[2] = malloc(count * sizeof(int64_t[2]));
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
    case POLY_ARG_STRING_TUPLE: {
      if (version < 6 || br_remaining(&r) < 2) {
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
    case POLY_ARG_REDUCE_AXIS: {
      arg.reduce_axis.op = (PolyOps)br_u16(&r);
      uint16_t n_axes = br_u16(&r);
      int64_t *axes = malloc(n_axes * sizeof(int64_t));
      for (int t = 0; t < n_axes; t++)
        axes[t] = br_i64(&r);
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
        for (int t = 0; t < n_extra; t++)
          extra[t] = br_i64(&r);
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
    case POLY_ARG_BUFFERIZE_OPTS: {
      if (version >= 8) {
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
      } else if (version >= 5) {
        uint32_t device_idx = br_u32(&r);
        arg.bufferize_opts.device = device_idx < n_strings ? strings[device_idx] : NULL;
      } else {
        int64_t legacy_device = br_i64(&r);
        arg.bufferize_opts.device =
            legacy_device > POLY_DEVICE_AUTO && legacy_device <= POLY_DEVICE_DISK
                ? poly_device_name((PolyDevice)legacy_device)
                : NULL;
      }
      arg.bufferize_opts.addrspace = (PolyAddrSpace)br_u8(&r);
      arg.bufferize_opts.removable = br_u8(&r) != 0;
      break;
    }
    case POLY_ARG_TENSOR_CORE: {
      uint32_t name_idx = br_u32(&r);
      arg.tensor_core.name = (name_idx < n_strings) ? strings[name_idx] : "";
      for (int d = 0; d < 3; d++) arg.tensor_core.dims[d] = (int)br_i64(&r);
      arg.tensor_core.threads = (int)br_i64(&r);
      break;
    }
    case POLY_ARG_BYTES:
      fprintf(stderr, "poly_ir_import: BINARY byte payloads are not package IR\n");
      if (srcs) free(srcs);
      goto fail_nodes;
    case POLY_ARG_INVALID:
      break;
    case POLY_ARG_PARAM: {
      param_arg_tmp.slot = br_i64(&r);
      if (version >= 7) {
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
      } else if (version >= 5) {
        uint32_t device_idx = br_u32(&r);
        param_arg_tmp.device = device_idx < n_strings ? strings[device_idx] : NULL;
      } else {
        int32_t legacy_device = br_i32(&r);
        param_arg_tmp.device = legacy_device > POLY_DEVICE_AUTO && legacy_device <= POLY_DEVICE_DISK
                                   ? poly_device_name((PolyDevice)legacy_device)
                                   : NULL;
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
      if (version < 9 || br_remaining(&r) < 6) {
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
    default:
      fprintf(stderr, "poly_ir_import: unknown arg kind %u at node %u\n", arg_kind, i);
      if (srcs) free(srcs);
      goto fail_nodes;
    }

    /* Create UOp -- restore tag to preserve BUFFER CSE-distinctness */
    PolyDType dtype = *dtype_table[dtype_idx];
    if (dtype_count > 1) dtype = poly_dtype_vec(dtype, dtype_count);
    PolyUOp *u =
        (tag != 0)
            ? poly_uop_tagged(ctx, (PolyOps)op_val, dtype, srcs, n_src, arg, tag)
            : poly_uop(ctx, (PolyOps)op_val, dtype, srcs, n_src, arg);

    /* Free temporary malloc'd arg buffers (arena has its own copy now) */
    if (arg.kind == POLY_ARG_INT_TUPLE && arg.int_tuple.vals)
      free(arg.int_tuple.vals);
    else if (arg.kind == POLY_ARG_BIGINT && arg.bigint.limbs)
      free((void *)arg.bigint.limbs);
    else if (arg.kind == POLY_ARG_PAIR_TUPLE && arg.pair_tuple.pairs)
      free(arg.pair_tuple.pairs);
    else if (arg.kind == POLY_ARG_REDUCE_AXIS && arg.reduce_axis.axes)
      free(arg.reduce_axis.axes);
    else if (arg.kind == POLY_ARG_RANGE && arg.range.extra)
      free(arg.range.extra);
    else if (arg.kind == POLY_ARG_STRING_TUPLE && arg.string_tuple.vals)
      free((void *)arg.string_tuple.vals);
    if (bufferize_devices_tmp) free(bufferize_devices_tmp);
    if (param_devices_tmp) free(param_devices_tmp);

    if (tag != 0) ((PolyUOp *)u)->tag = tag;
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

    out->bufs[i].name = (name_idx < n_strings) ? strdup(strings[name_idx]) : strdup("");
    out->bufs[i].role = role;
    out->bufs[i].trainable_set = (iface_flags & 2) != 0;
    out->bufs[i].trainable =
        out->bufs[i].trainable_set ? ((iface_flags & 1) != 0) : (role == POLY_IR_ROLE_PARAM);
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

    if (version >= 2) {
      if (br_remaining(&r) < 12) goto fail_ep;
      out->entrypoints[i].flags = br_u32(&r);
      uint16_t n_inputs = br_u16(&r);
      uint16_t n_outputs = br_u16(&r);
      uint32_t objective_idx = br_u32(&r);
      out->entrypoints[i].n_inputs = n_inputs;
      out->entrypoints[i].n_outputs = n_outputs;
      if (objective_idx != UINT32_MAX)
        out->entrypoints[i].objective =
            (objective_idx < n_strings) ? strdup(strings[objective_idx]) : strdup("");
      if (n_inputs > 0) {
        char **inputs = calloc(n_inputs, sizeof(char *));
        if (!inputs) goto fail_ep;
        out->entrypoints[i].inputs = (const char **)inputs;
        for (uint16_t j = 0; j < n_inputs; j++) {
          if (br_remaining(&r) < 4) goto fail_ep;
          uint32_t idx = br_u32(&r);
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
          outputs[j] = (idx < n_strings) ? strdup(strings[idx]) : strdup("");
        }
      }
    }
  }

  out->ctx = ctx;

  /* Cleanup temp arrays */
  for (uint32_t i = 0; i < n_strings; i++)
    free(strings[i]);
  free(strings);
  free(nodes);
  return 0;

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

void poly_ir_spec_free(PolyIrSpec *spec) {
  if (!spec) return;
  for (int i = 0; i < spec->n_bufs; i++)
    free((char *)spec->bufs[i].name);
  free(spec->bufs);
  for (int i = 0; i < spec->n_entrypoints; i++)
    free_ir_entrypoint(&spec->entrypoints[i]);
  free(spec->entrypoints);
  spec->bufs = NULL;
  spec->entrypoints = NULL;
  spec->n_bufs = 0;
  spec->n_entrypoints = 0;
}
