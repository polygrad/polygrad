/* ONNX protobuf -> existing Tensor operations -> one sealed Model.
 * Reference: tinygrad/nn/onnx.py (OnnxPBParser and onnx_ops). The parser is
 * bounded independently of protobuf lengths; no file paths or code are executed.
 * Host-valued shape operands must be closed graphs, never runtime Tensor values. */
#include "onnx_loader.h"
#include "import_error.h"
#include "../models/factory.h"
#include "../models/layers.h"
#include "../nn/nn.h"
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ONNX_VALUES 4096
#define ONNX_IO 128
#define ONNX_RANK 16
#define ONNX_NAME 256
#define ONNX_BYTES ((size_t)1 << 30)

typedef struct {
  const uint8_t *data;
  size_t size;
} Bytes;
typedef struct {
  Bytes bytes;
  size_t pos;
  bool bad;
} Reader;
typedef struct {
  unsigned tag, wire;
  uint64_t integer;
  Bytes bytes;
} Field;
typedef struct {
  char name[ONNX_NAME];
  PolyTensor *tensor;
  uint8_t *host; /* owned initializer, retained until post-build coherent write */
  size_t nbytes;
  PolyDType dtype;
  int rank;
  int64_t shape[ONNX_RANK];
  bool initializer;
  bool constant; /* closed graph: safe to evaluate import-time shape operands */
  int scope;
  char binding[ONNX_NAME]; /* Model state names are global; ONNX names are lexical. */
} Value;
typedef struct {
  PolyCtx *ctx;
  PolyModel *model;
  const PolyOnnxOptions *options;
  cJSON *dimensions;
  Value values[ONNX_VALUES];
  int nvalues, opset, microsoft_opset, ml_opset, node_index;
  int scopes[33], depth, next_scope;
  size_t initializer_bytes;
  char op[ONNX_NAME];
} Import;

static bool fail(Import *d, PolyImportError code, const char *fmt, ...) {
  if (poly_import_last_error_code() == POLY_IMPORT_OK) {
    char msg[384];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);
    poly_import_error_set(
        code, "ONNX node %d (%s, opset %d): %s", d->node_index, d->op, d->opset, msg
    );
  }
  return false;
}

static uint64_t varint(Reader *r) {
  uint64_t v = 0;
  for (int i = 0; i < 10; i++) {
    if (r->pos == r->bytes.size) break;
    unsigned b = r->bytes.data[r->pos++];
    if (i == 9 && b > 1) break;
    v |= (uint64_t)(b & 127) << (7 * i);
    if (!(b & 128)) return v;
  }
  r->bad = true;
  return 0;
}

static bool next(Reader *r, Field *f) {
  if (r->bad || r->pos == r->bytes.size) return false;
  memset(f, 0, sizeof(*f));
  uint64_t key = varint(r);
  if (r->bad || !(key >> 3) || key >> 3 > 0x1fffffff) {
    r->bad = true;
    return false;
  }
  f->tag = key >> 3;
  f->wire = key & 7;
  if (f->wire == 0)
    f->integer = varint(r);
  else {
    uint64_t n = f->wire == 1 ? 8 : f->wire == 5 ? 4 : f->wire == 2 ? varint(r) : UINT64_MAX;
    if (r->bad || n > r->bytes.size - r->pos) {
      r->bad = true;
      return false;
    }
    f->bytes = (Bytes){r->bytes.data + r->pos, (size_t)n};
    r->pos += (size_t)n;
  }
  return !r->bad;
}

/* Singular protobuf fields are rejected when repeated rather than silently
 * choosing different last/first values across importers. */
static bool field(Import *d, Bytes b, unsigned tag, unsigned wire, Field *out, bool required) {
  Reader r = {.bytes = b};
  Field f;
  bool found = false;
  memset(out, 0, sizeof(*out));
  while (next(&r, &f))
    if (f.tag == tag) {
      if (found || f.wire != wire)
        return fail(d, POLY_IMPORT_ERR_PARSE, "invalid/duplicate field %u", tag);
      *out = f;
      found = true;
    }
  if (r.bad || (required && !found))
    return fail(d, POLY_IMPORT_ERR_PARSE, "malformed/missing field %u", tag);
  return true;
}

static bool text(Import *d, Bytes b, char out[ONNX_NAME], bool empty) {
  if ((!b.size && !empty) || b.size >= ONNX_NAME || (b.size && memchr(b.data, 0, b.size)))
    return fail(d, POLY_IMPORT_ERR_PARSE, "invalid name/string length");
  if (b.size) memcpy(out, b.data, b.size);
  out[b.size] = 0;
  return true;
}

static bool name_field(Import *d, Bytes b, unsigned tag, char out[ONNX_NAME], bool required) {
  Field f;
  return field(d, b, tag, 2, &f, required) && text(d, f.bytes, out, !required);
}

static bool integers(Import *d, Bytes b, unsigned tag, int64_t *out, int cap, int *count) {
  Reader r = {.bytes = b};
  Field f;
  *count = 0;
  while (next(&r, &f))
    if (f.tag == tag) {
      Reader packed = {.bytes = f.bytes};
      if (f.wire != 0 && f.wire != 2) return fail(d, POLY_IMPORT_ERR_PARSE, "invalid integer list");
      do {
        if (f.wire == 2 && packed.pos == packed.bytes.size) break;
        uint64_t v = f.wire == 0 ? f.integer : varint(&packed);
        if (packed.bad || *count == cap)
          return fail(d, POLY_IMPORT_ERR_PARSE, "invalid/oversized integer list");
        memcpy(&out[(*count)++], &v, sizeof(v));
        if (f.wire == 0) break;
      } while (true);
    }
  return !r.bad || fail(d, POLY_IMPORT_ERR_PARSE, "truncated integer list");
}

static bool dtype(Import *d, uint64_t id, PolyDType *dt) {
  switch (id) {
  case 1:
    *dt = POLY_FLOAT32;
    return true;
  case 2:
    *dt = POLY_UINT8;
    return true;
  case 3:
    *dt = POLY_INT8;
    return true;
  case 4:
    *dt = POLY_UINT16;
    return true;
  case 5:
    *dt = POLY_INT16;
    return true;
  case 12:
    *dt = POLY_UINT32;
    return true;
  case 13:
    *dt = POLY_UINT64;
    return true;
  case 16:
    *dt = POLY_BFLOAT16;
    return true;
  case 6:
    *dt = POLY_INT32;
    return true;
  case 7:
    *dt = POLY_INT64;
    return true;
  case 9:
    *dt = POLY_BOOL;
    return true;
  case 10:
    *dt = POLY_FLOAT16;
    return true;
  case 11:
    *dt = POLY_FLOAT64;
    return true;
  default:
    return fail(
        d, POLY_IMPORT_ERR_DTYPE_UNSUPPORTED, "unsupported tensor dtype %llu",
        (unsigned long long)id
    );
  }
}

static Value *lookup(Import *d, const char *name) {
  for (int depth = d->depth; depth >= 0; depth--)
    for (int i = 0; i < d->nvalues; i++)
      if (d->values[i].scope == d->scopes[depth] && !strcmp(d->values[i].name, name))
        return &d->values[i];
  return NULL;
}

static Value *add_value(Import *d, const char *name) {
  Value *existing = lookup(d, name);
  if (!name[0] || (existing && existing->scope == d->scopes[d->depth]) ||
      d->nvalues == ONNX_VALUES) {
    fail(d, POLY_IMPORT_ERR_PARSE, "duplicate/empty value '%s' or value limit exceeded", name);
    return NULL;
  }
  Value *v = &d->values[d->nvalues++];
  strcpy(v->name, name);
  v->scope = d->scopes[d->depth];
  if (v->scope)
    snprintf(v->binding, sizeof(v->binding), "__onnx_subgraph_%d", d->nvalues);
  else
    strcpy(v->binding, name);
  return v;
}

static bool shape_bytes(Import *d, Value *v) {
  size_t n = poly_dtype_itemsize(v->dtype);
  for (int i = 0; i < v->rank; i++) {
    if (v->shape[i] <= 0 || (uint64_t)v->shape[i] > ONNX_BYTES / n)
      return fail(d, POLY_IMPORT_ERR_SHAPE_MISMATCH, "'%s': nonpositive/oversized shape", v->name);
    n *= (size_t)v->shape[i];
  }
  v->nbytes = n;
  return true;
}

static bool decimal(Import *d, const char *s, size_t *out) {
  if (!s[0]) return fail(d, POLY_IMPORT_ERR_PARSE, "empty external offset/length");
  size_t v = 0;
  for (; *s; s++) {
    if (*s < '0' || *s > '9' || v > (SIZE_MAX - (unsigned)(*s - '0')) / 10)
      return fail(d, POLY_IMPORT_ERR_PARSE, "invalid external offset/length");
    v = v * 10 + *s - '0';
  }
  *out = v;
  return true;
}

static bool external_bytes(Import *d, Bytes proto, Bytes *raw) {
  Reader r = {.bytes = proto};
  Field f;
  char location[ONNX_NAME] = {0};
  size_t offset = 0, length = SIZE_MAX;
  unsigned seen = 0;
  while (next(&r, &f))
    if (f.tag == 13) {
      char key[ONNX_NAME], val[ONNX_NAME];
      if (f.wire != 2 || !name_field(d, f.bytes, 1, key, true) ||
          !name_field(d, f.bytes, 2, val, true))
        return false;
      unsigned bit = !strcmp(key, "location") ? 1
                     : !strcmp(key, "offset") ? 2
                     : !strcmp(key, "length") ? 4
                                              : 0;
      if (!bit || (seen & bit))
        return fail(d, POLY_IMPORT_ERR_PARSE, "unsupported/duplicate external key '%s'", key);
      seen |= bit;
      if (bit == 1)
        strcpy(location, val);
      else if (!decimal(d, val, bit == 2 ? &offset : &length))
        return false;
    }
  if (r.bad || !(seen & 1)) return fail(d, POLY_IMPORT_ERR_PARSE, "missing external location");
  const PolyOnnxOptions *o = d->options;
  for (int i = 0; o && i < o->n_external; i++)
    if (!strcmp(location, o->external_names[i])) {
      size_t size = (size_t)o->external_lengths[i];
      if (offset > size || (length != SIZE_MAX && length > size - offset))
        return fail(d, POLY_IMPORT_ERR_PARSE, "external byte range exceeds '%s'", location);
      *raw = (Bytes){o->external_data[i] + offset, length == SIZE_MAX ? size - offset : length};
      return true;
    }
  return fail(d, POLY_IMPORT_ERR_PARSE, "external data '%s' was not supplied", location);
}

static bool onnx_initializer(Import *d, Bytes proto, const char *output_name) {
  char name[ONNX_NAME];
  Field typ, raw, location;
  if (output_name)
    strcpy(name, output_name);
  else if (!name_field(d, proto, 8, name, true))
    return false;
  Value *v = add_value(d, name);
  if (!v || !field(d, proto, 2, 0, &typ, true) || !dtype(d, typ.integer, &v->dtype) ||
      !integers(d, proto, 1, v->shape, ONNX_RANK, &v->rank) || !shape_bytes(d, v) ||
      !field(d, proto, 9, 2, &raw, false) || !field(d, proto, 14, 0, &location, false))
    return false;
  if (location.integer > 1) return fail(d, POLY_IMPORT_ERR_PARSE, "invalid data_location");
  if (location.integer == 1) {
    if (raw.tag || !external_bytes(d, proto, &raw.bytes))
      return fail(d, POLY_IMPORT_ERR_PARSE, "ambiguous external storage");
    raw.tag = 9;
  }
  Reader r = {.bytes = proto};
  Field f;
  int storage_fields = raw.tag ? 1 : 0;
  unsigned typed_tag = typ.integer == 1                           ? 4
                       : typ.integer == 11                        ? 10
                       : typ.integer == 7                         ? 7
                       : (typ.integer == 12 || typ.integer == 13) ? 11
                                                                  : 5;
  size_t used = 0, item = poly_dtype_itemsize(v->dtype);
  if (v->nbytes > ONNX_BYTES - d->initializer_bytes)
    return fail(d, POLY_IMPORT_ERR_PARSE, "initializer byte budget exceeded");
  d->initializer_bytes += v->nbytes;
  v->host = malloc(v->nbytes);
  if (!v->host) return fail(d, POLY_IMPORT_ERR_INTERNAL, "initializer allocation failed");
  /* ONNX numeric payloads are little-endian, including packed fixed32/64. */
  const uint16_t endian = 1;
  if (*(const uint8_t *)&endian != 1)
    return fail(d, POLY_IMPORT_ERR_INTERNAL, "ONNX import requires a little-endian host");
  if (raw.tag) {
    if (raw.bytes.size != v->nbytes)
      return fail(
          d, POLY_IMPORT_ERR_WEIGHT_MISMATCH, "'%s': initializer byte count mismatch", name
      );
    memcpy(v->host, raw.bytes.data, v->nbytes);
    used = v->nbytes;
  }
  bool typed_seen = false;
  while (next(&r, &f))
    if (f.tag == 4 || f.tag == 5 || f.tag == 6 || f.tag == 7 || f.tag == 10 || f.tag == 11) {
      if (f.tag != typed_tag || raw.tag)
        return fail(d, POLY_IMPORT_ERR_PARSE, "mixed/unsupported initializer payload");
      if (!typed_seen) {
        storage_fields++;
        typed_seen = true;
      }
      bool floating = typ.integer == 1 || typ.integer == 11;
      if (floating) {
        if (f.wire != 2 && f.wire != (item == 4 ? 5u : 1u))
          return fail(d, POLY_IMPORT_ERR_PARSE, "invalid floating payload");
        if (f.bytes.size % item || f.bytes.size > v->nbytes - used)
          return fail(d, POLY_IMPORT_ERR_PARSE, "oversized floating payload");
        memcpy(v->host + used, f.bytes.data, f.bytes.size);
        used += f.bytes.size;
      } else {
        Reader p = {.bytes = f.bytes};
        if (f.wire != 0 && f.wire != 2)
          return fail(d, POLY_IMPORT_ERR_PARSE, "invalid integer payload");
        do {
          if (f.wire == 2 && p.pos == p.bytes.size) break;
          uint64_t n = f.wire == 0 ? f.integer : varint(&p);
          if (p.bad || item > v->nbytes - used || (typ.integer == 9 && n > 1))
            return fail(d, POLY_IMPORT_ERR_PARSE, "invalid integer payload size/value");
          memcpy(v->host + used, &n, item);
          used += item;
          if (f.wire == 0) break;
        } while (true);
      }
    }
  if (r.bad || storage_fields != 1 || used != v->nbytes)
    return fail(d, POLY_IMPORT_ERR_PARSE, "incomplete initializer '%s'", name);
  v->initializer = true;
  v->constant = true;
  if (poly_dtype_is_float(v->dtype))
    v->tensor = poly_model_param(d->model, v->binding, v->dtype, v->shape, v->rank);
  else {
    v->tensor = poly_model_aux_from_host(
        d->model, v->binding, v->dtype, v->shape, v->rank, v->host, v->nbytes
    );
  }
  if (v->tensor && poly_dtype_is_float(v->dtype)) {
    /* A downstream shape calculation may consume a closed initializer before
     * Model sealing. Initialize the declared owner now, not a borrowed COPY. */
    PolyUOp *buffer = (PolyUOp *)poly_uop_get_buffer_identity(v->tensor->uop_physical);
    PolyDevice device = poly_ctx_get_preferred_device(d->ctx);
    if (!buffer || poly_buffer_ensure_device_allocated(d->ctx, buffer, device) ||
        poly_buffer_write(d->ctx, buffer, v->host, v->nbytes))
      return false;
  }
  return v->tensor != NULL || fail(d, POLY_IMPORT_ERR_INTERNAL, "initializer declaration failed");
}

static bool value_info(Import *d, Bytes b, Value *v) {
  Field type, tensor, dt, shape;
  if (!name_field(d, b, 1, v->name, true) || !field(d, b, 2, 2, &type, true) ||
      !field(d, type.bytes, 1, 2, &tensor, true) || !field(d, tensor.bytes, 1, 0, &dt, true) ||
      !dtype(d, dt.integer, &v->dtype) || !field(d, tensor.bytes, 2, 2, &shape, true))
    return false;
  Reader r = {.bytes = shape.bytes};
  Field f;
  while (next(&r, &f))
    if (f.tag == 1) {
      if (f.wire != 2 || v->rank == ONNX_RANK)
        return fail(d, POLY_IMPORT_ERR_SHAPE_MISMATCH, "invalid tensor rank");
      Field fixed, symbolic;
      if (!field(d, f.bytes, 1, 0, &fixed, false) || !field(d, f.bytes, 2, 2, &symbolic, false))
        return false;
      if (!!fixed.tag == !!symbolic.tag)
        return fail(d, POLY_IMPORT_ERR_SHAPE_MISMATCH, "dimension must be fixed or named");
      int64_t n;
      if (fixed.tag) {
        if (fixed.integer > INT64_MAX)
          return fail(d, POLY_IMPORT_ERR_SHAPE_MISMATCH, "dimension overflow");
        n = (int64_t)fixed.integer;
      } else {
        char key[ONNX_NAME];
        if (!text(d, symbolic.bytes, key, false)) return false;
        cJSON *x = cJSON_GetObjectItemCaseSensitive(d->dimensions, key);
        if (!cJSON_IsNumber(x))
          return fail(d, POLY_IMPORT_ERR_SHAPE_MISMATCH, "supply dimension '%s' explicitly", key);
        n = (int64_t)x->valuedouble;
      }
      v->shape[v->rank++] = n;
    }
  return !r.bad && shape_bytes(d, v);
}

static int rank(Import *d, PolyTensor *t) {
  return poly_uop_ndim(d->ctx, t->uop_physical);
}
static const int64_t *shape(Import *d, PolyTensor *t) {
  return poly_uop_max_shape_dims(d->ctx, t->uop_physical);
}
static bool same_dtype(PolyTensor *a, PolyTensor *b) {
  return poly_dtype_eq(a->uop_physical->dtype, b->uop_physical->dtype);
}

/* OnnxRunner._get_python_const caches closed shape operands. Never read a
 * graph input at import: that would freeze one caller's data into the bundle. */
static bool onnx_const_data(Import *d, Value *v) {
  if (!v || !v->constant)
    return fail(d, POLY_IMPORT_ERR_UNSUPPORTED_OP, "shape operand depends on runtime data");
  if (v->host) return true;
  if (v->nbytes > 65536)
    return fail(d, POLY_IMPORT_ERR_UNSUPPORTED_OP, "host shape operand exceeds 64KiB");
  PolyTensor *t = poly_tensor_contiguous(d->ctx, v->tensor), *out = NULL;
  if (!t || poly_realize_tensors(d->ctx, &t, 1, &out)) return false;
  PolyUOp *buffer = out ? (PolyUOp *)poly_uop_get_buffer_identity(out->uop_physical) : NULL;
  v->host = malloc(v->nbytes ? v->nbytes : 1);
  return buffer && v->host && !poly_buffer_read(d->ctx, buffer, v->host, v->nbytes);
}

static bool onnx_ints(Import *d, Value *v, int64_t *values, int cap, int *count) {
  if (!v || v->rank > 1 ||
      (!poly_dtype_eq(v->dtype, POLY_INT64) && !poly_dtype_eq(v->dtype, POLY_INT32)) ||
      v->nbytes / poly_dtype_itemsize(v->dtype) > (size_t)cap || !onnx_const_data(d, v))
    return fail(d, POLY_IMPORT_ERR_SHAPE_MISMATCH, "expected constant INT32/INT64 scalar/vector");
  size_t item = poly_dtype_itemsize(v->dtype);
  *count = (int)(v->nbytes / item);
  for (int i = 0; i < *count; i++) {
    if (item == 8)
      memcpy(values + i, v->host + 8 * i, 8);
    else {
      int32_t value;
      memcpy(&value, v->host + 4 * i, 4);
      values[i] = value;
    }
  }
  return true;
}

static bool onnx_floats(Import *d, Value *v, double *values, int cap, int *count) {
  if (!v || v->rank > 1 ||
      (!poly_dtype_eq(v->dtype, POLY_FLOAT32) && !poly_dtype_eq(v->dtype, POLY_FLOAT64)) ||
      v->nbytes / poly_dtype_itemsize(v->dtype) > (size_t)cap || !onnx_const_data(d, v))
    return fail(d, POLY_IMPORT_ERR_SHAPE_MISMATCH, "expected constant FLOAT/DOUBLE scalar/vector");
  size_t item = poly_dtype_itemsize(v->dtype);
  *count = (int)(v->nbytes / item);
  for (int i = 0; i < *count; i++) {
    if (item == 8)
      memcpy(values + i, v->host + 8 * i, 8);
    else {
      float value;
      memcpy(&value, v->host + 4 * i, 4);
      values[i] = value;
    }
    if (!isfinite(values[i]))
      return fail(d, POLY_IMPORT_ERR_SHAPE_MISMATCH, "nonfinite shape operand");
  }
  return true;
}

static bool onnx_axis(Import *d, int64_t *axis, int ndim) {
  if (*axis < 0) *axis += ndim;
  return (*axis >= 0 && *axis < ndim) ||
         fail(d, POLY_IMPORT_ERR_SHAPE_MISMATCH, "axis outside input rank");
}

/* Python literals in the pinned operator programs are weak rank-zero scalars,
 * not UOp.const_like: the latter copies the reference's dtype AND shape. */
static PolyTensor *onnx_int(Import *d, int64_t value) {
  PolyUOp *u = poly_uop_const(d->ctx, poly_arg_int(value), POLY_WEAKINT);
  return u ? poly_tensor_create_with_roots(
                 d->ctx, u, u, POLY_TENSOR_VALUE, poly_ctx_get_preferred_device(d->ctx)
             )
           : NULL;
}
static PolyTensor *onnx_float(Import *d, double value) {
  PolyUOp *u = poly_uop_const(d->ctx, poly_arg_float(value), POLY_WEAKFLOAT);
  return u ? poly_tensor_create_with_roots(
                 d->ctx, u, u, POLY_TENSOR_VALUE, poly_ctx_get_preferred_device(d->ctx)
             )
           : NULL;
}

static PolyTensor *onnx_sub(Import *d, PolyTensor *x, PolyTensor *y) {
  /* Tensor.sub promotes both operands before negation (notably uint8 zero points). */
  return poly_tensor_alu2(d->ctx, POLY_OP_SUB, x, y);
}

static PolyTensor *onnx_not(Import *d, PolyTensor *x) {
  return poly_tensor_alu2(d->ctx, POLY_OP_CMPEQ, x, onnx_int(d, 0));
}

typedef struct {
  char name[ONNX_NAME];
  Bytes proto;
  int type;
} Attr;
typedef struct {
  Value *inputs[ONNX_IO];
  int nin;
  char outputs[ONNX_IO][ONNX_NAME];
  int nout;
  Attr attrs[32];
  int nattrs;
  bool captures_runtime;
  PolyTensor *extra[ONNX_IO];
} Node;

static Attr *attr(Node *n, const char *key) {
  for (int i = 0; i < n->nattrs; i++)
    if (!strcmp(n->attrs[i].name, key)) return &n->attrs[i];
  return NULL;
}
static bool attrs(Import *d, Node *n, const char *allowed) {
  for (int i = 0; i < n->nattrs; i++) {
    char key[ONNX_NAME + 2];
    snprintf(key, sizeof(key), "|%s|", n->attrs[i].name);
    if (!strstr(allowed, key))
      return fail(
          d, POLY_IMPORT_ERR_UNSUPPORTED_OP, "unsupported attribute '%s'", n->attrs[i].name
      );
  }
  return true;
}
static int64_t attr_int(Import *d, Node *n, const char *key, int64_t fallback) {
  Attr *a = attr(n, key);
  Field f;
  int64_t v;
  if (!a) return fallback;
  if (a->type != 2 || !field(d, a->proto, 3, 0, &f, true)) {
    fail(d, POLY_IMPORT_ERR_PARSE, "'%s' must be INT", key);
    return fallback;
  }
  memcpy(&v, &f.integer, 8);
  return v;
}
static double attr_float(Import *d, Node *n, const char *key, double fallback) {
  Attr *a = attr(n, key);
  Field f;
  float v;
  if (!a) return fallback;
  if (a->type != 1 || !field(d, a->proto, 2, 5, &f, true)) {
    fail(d, POLY_IMPORT_ERR_PARSE, "'%s' must be FLOAT", key);
    return fallback;
  }
  memcpy(&v, f.bytes.data, 4);
  if (!isfinite(v)) fail(d, POLY_IMPORT_ERR_PARSE, "nonfinite attribute '%s'", key);
  return v;
}
static bool attr_ints(Import *d, Node *n, const char *key, int64_t *out, int cap, int *count) {
  Attr *a = attr(n, key);
  *count = 0;
  if (!a) return true;
  return (a->type == 7 && integers(d, a->proto, 8, out, cap, count)) ||
         fail(d, POLY_IMPORT_ERR_PARSE, "'%s' must be INTS", key);
}
static bool attr_text(
    Import *d,
    Node *n,
    const char *key,
    const char *fallback,
    char out[ONNX_NAME]
) {
  Attr *a = attr(n, key);
  Field f;
  if (!a) {
    strcpy(out, fallback);
    return true;
  }
  return a->type == 3 && field(d, a->proto, 4, 2, &f, true) && text(d, f.bytes, out, false);
}
static bool arity_outputs(Import *d, Node *n, int min, int max, int maxout) {
  if (n->nin < min || n->nin > max || n->nout < 1 || n->nout > maxout || !n->outputs[0][0])
    return fail(d, POLY_IMPORT_ERR_UNSUPPORTED_OP, "unsupported input/output count");
  for (int i = 0; i < min; i++)
    if (!n->inputs[i]) return fail(d, POLY_IMPORT_ERR_PARSE, "missing required input %d", i);
  return true;
}
static bool arity(Import *d, Node *n, int min, int max) {
  return arity_outputs(d, n, min, max, 1);
}
static bool broadcast(Import *d, PolyTensor *a, PolyTensor *b) {
  int na = rank(d, a), nb = rank(d, b);
  const int64_t *sa = shape(d, a), *sb = shape(d, b);
  for (int i = 1; i <= na && i <= nb; i++)
    if (sa[na - i] != sb[nb - i] && sa[na - i] != 1 && sb[nb - i] != 1)
      return fail(d, POLY_IMPORT_ERR_SHAPE_MISMATCH, "incompatible broadcast dimensions");
  return same_dtype(a, b) || fail(d, POLY_IMPORT_ERR_DTYPE_UNSUPPORTED, "input dtypes must match");
}
static bool broadcast_into(Import *d, PolyTensor *target, PolyTensor *value) {
  if (!broadcast(d, target, value)) return false;
  int nt = rank(d, target), nv = rank(d, value);
  if (nv > nt) return false;
  const int64_t *t = shape(d, target), *v = shape(d, value);
  for (int i = 1; i <= nv; i++)
    if (v[nv - i] != 1 && v[nv - i] != t[nt - i]) return false;
  return true;
}
static PolyTensor *scale(Import *d, PolyTensor *x, double value) {
  return poly_tensor_alu2(d->ctx, POLY_OP_MUL, x, onnx_float(d, value));
}
static PolyTensor *transpose(Import *d, PolyTensor *x) {
  int n = rank(d, x);
  int64_t p[ONNX_RANK];
  if (n < 2 || n > ONNX_RANK) return NULL;
  for (int i = 0; i < n; i++)
    p[i] = i;
  p[n - 1] = n - 2;
  p[n - 2] = n - 1;
  return poly_tensor_permute(d->ctx, x, p, n);
}
static PolyTensor *matmul(Import *d, PolyTensor *a, PolyTensor *b) {
  int na = rank(d, a), nb = rank(d, b);
  const int64_t *sa = shape(d, a), *sb = shape(d, b);
  if (!na || !nb || sa[na - 1] != sb[nb > 1 ? nb - 2 : 0] || !same_dtype(a, b)) {
    fail(d, POLY_IMPORT_ERR_SHAPE_MISMATCH, "incompatible MatMul shape/dtype");
    return NULL;
  }
  for (int i = 3; i <= na && i <= nb; i++)
    if (sa[na - i] != sb[nb - i] && sa[na - i] != 1 && sb[nb - i] != 1) {
      fail(d, POLY_IMPORT_ERR_SHAPE_MISMATCH, "incompatible MatMul batches");
      return NULL;
    }
  return poly_tensor_dot(d->ctx, a, b);
}

static PolyTensor *onnx_clamp(Import *d, PolyTensor *x, double lo, double hi) {
  bool integer = poly_dtype_is_int(x->uop_physical->dtype);
  PolyTensor *a = integer ? onnx_int(d, (int64_t)lo) : onnx_float(d, lo);
  PolyTensor *b = integer ? onnx_int(d, (int64_t)hi) : onnx_float(d, hi);
  x = poly_tensor_alu3(d->ctx, POLY_OP_WHERE, poly_tensor_alu2(d->ctx, POLY_OP_CMPLT, x, a), a, x);
  return poly_tensor_alu3(
      d->ctx, POLY_OP_WHERE, poly_tensor_alu2(d->ctx, POLY_OP_CMPLT, b, x), b, x
  );
}

static PolyTensor *onnx_quant_cast(Import *d, PolyTensor *x, PolyDType dtype) {
  double lo, hi;
  if (poly_dtype_eq(dtype, POLY_UINT8)) {
    lo = 0;
    hi = 255;
  } else if (poly_dtype_eq(dtype, POLY_INT8)) {
    lo = -128;
    hi = 127;
  } else {
    fail(d, POLY_IMPORT_ERR_DTYPE_UNSUPPORTED, "quantized output must be INT8 or UINT8");
    return NULL;
  }
  return poly_tensor_cast(d->ctx, onnx_clamp(d, x, lo, hi), dtype);
}

/* onnx.py:_prepare_quantize. Scalar and per-axis parameters retain their dtype. */
static PolyTensor *onnx_quant_parameter(Import *d, PolyTensor *x, PolyTensor *v, int64_t axis) {
  int nv = rank(d, v), nx = rank(d, x);
  int64_t count = 1, dims[ONNX_RANK];
  for (int i = 0; i < nv; i++)
    count *= shape(d, v)[i];
  if (count == 1) return poly_tensor_reshape(d->ctx, v, dims, 0);
  if (!onnx_axis(d, &axis, nx) || nv != 1 || count != shape(d, x)[axis]) return NULL;
  for (int i = 0; i < nx; i++)
    dims[i] = i == axis ? count : 1;
  return poly_tensor_reshape(d->ctx, v, dims, nx);
}

static PolyTensor *onnx_index_axes(
    Import *d,
    PolyTensor *x,
    PolyTensor **indices,
    int offset,
    int count
) {
  int nr = rank(d, x), kinds[ONNX_RANK];
  int64_t steps[ONNX_RANK];
  PolyUOp *starts[ONNX_RANK], *sizes[ONNX_RANK];
  PolyTensor *rows[ONNX_RANK] = {0};
  if (offset < 0 || offset + count > nr) return NULL;
  for (int i = 0; i < nr; i++) {
    kinds[i] = i >= offset && i < offset + count ? POLY_INDEX_TENSOR : POLY_INDEX_SLICE;
    if (kinds[i] == POLY_INDEX_TENSOR) rows[i] = indices[i - offset];
    starts[i] = poly_uop_const_int(d->ctx, 0);
    sizes[i] = poly_uop_const_int(d->ctx, shape(d, x)[i]);
    steps[i] = 1;
  }
  return poly_tensor_getitem(d->ctx, x, kinds, starts, sizes, steps, rows, nr);
}

/* onnx.py:_resolve_pool_pads. Padding is reversed only at the Tensor boundary. */
static bool onnx_spatial(
    Import *d,
    Node *n,
    PolyTensor *x,
    const int64_t *weight_kernel,
    int64_t *kernel,
    int64_t *stride,
    int64_t *dilation,
    int64_t *padding,
    bool transposed
) {
  int dims = rank(d, x) - 2, count;
  if (dims < 1 || dims > ONNX_RANK - 2) return false;
  for (int i = 0; i < dims; i++) {
    kernel[i] = weight_kernel ? weight_kernel[i] : 0;
    stride[i] = dilation[i] = 1;
  }
  int64_t pads[2 * ONNX_RANK] = {0};
  if (!attr_ints(d, n, "kernel_shape", kernel, dims, &count) ||
      (attr(n, "kernel_shape") && count != dims) ||
      !attr_ints(d, n, "strides", stride, dims, &count) || (attr(n, "strides") && count != dims) ||
      !attr_ints(d, n, "dilations", dilation, dims, &count) ||
      (attr(n, "dilations") && count != dims) || !attr_ints(d, n, "pads", pads, 2 * dims, &count) ||
      (attr(n, "pads") && count != 2 * dims))
    return false;
  char mode[ONNX_NAME];
  if (!attr_text(d, n, "auto_pad", "NOTSET", mode)) return false;
  bool explicit_pad = !strcmp(mode, "NOTSET"), valid = !strcmp(mode, "VALID"),
       upper = !strcmp(mode, "SAME_UPPER");
  if (!explicit_pad && !valid && !upper && strcmp(mode, "SAME_LOWER")) return false;
  if (transposed) {
    explicit_pad = true;
    valid = false;
  }
  for (int i = 0; i < dims; i++) {
    if (kernel[i] < 1 || kernel[i] > INT_MAX || stride[i] < 1 || stride[i] > INT_MAX ||
        dilation[i] < 1 || dilation[i] > INT_MAX ||
        (weight_kernel && kernel[i] != weight_kernel[i]))
      return false;
    if (valid)
      pads[i] = pads[i + dims] = 0;
    else if (!explicit_pad) {
      /* The pin's SAME formula omits dilation. Do not silently substitute the ONNX formula. */
      int64_t extent = shape(d, x)[i + 2];
      int64_t total = ((extent - 1) / stride[i]) * stride[i] + kernel[i] - extent;
      if (dilation[i] != 1 || total < 0) return false;
      pads[i] = upper ? total / 2 : total - total / 2;
      pads[i + dims] = total - pads[i];
    }
    if (pads[i] < 0 || pads[i + dims] < 0 || pads[i] > INT_MAX || pads[i + dims] > INT_MAX)
      return false;
    padding[2 * (dims - i - 1)] = pads[i];
    padding[2 * (dims - i - 1) + 1] = pads[i + dims];
  }
  return true;
}

/* onnx.py:Resize uses Tensor arithmetic for coordinates, preserving float32
 * rounding before nearest-neighbor decisions and interpolation weights. */
static PolyTensor *onnx_resize(Import *d, Node *n, bool legacy) {
  PolyCtx *ctx = d->ctx;
  if (!arity(d, n, legacy ? 2 : 1, legacy ? 2 : 4) ||
      !attrs(
          d, n,
          legacy ? "|mode|"
                 : "|antialias||axes||coordinate_transformation_mode||cubic_coeff_a||exclude_"
                   "outside||extrapolation_value||keep_aspect_ratio_policy||mode||nearest_mode|"
      ) ||
      attr_int(d, n, "antialias", 0))
    return NULL;
  PolyTensor *x = n->inputs[0]->tensor;
  int nr = rank(d, x), na = 0, ns = 0, nz = 0;
  if (nr < 3) return NULL;
  int64_t axes[ONNX_RANK], perm[ONNX_RANK], inverse[ONNX_RANK], sizes[ONNX_RANK];
  double scales[ONNX_RANK];
  char mode[ONNX_NAME], transform[ONNX_NAME], nearest[ONNX_NAME], aspect[ONNX_NAME];
  if (!attr_text(d, n, "mode", "nearest", mode) ||
      !attr_text(d, n, "coordinate_transformation_mode", "half_pixel", transform) ||
      !attr_text(d, n, "nearest_mode", "round_prefer_floor", nearest) ||
      !attr_text(d, n, "keep_aspect_ratio_policy", "stretch", aspect) ||
      !attr_ints(d, n, "axes", axes, ONNX_RANK, &na))
    return NULL;
  if (strcmp(mode, "nearest") && strcmp(mode, "linear") && strcmp(mode, "cubic")) return NULL;
  if (!na)
    for (int i = 0; i < nr; i++)
      axes[na++] = i;
  unsigned used = 0;
  for (int i = 0; i < na; i++) {
    if (!onnx_axis(d, &axes[i], nr) || (used & (1u << axes[i]))) return NULL;
    used |= 1u << axes[i];
  }
  int k = 0;
  for (int i = 0; i < nr; i++)
    if (!(used & (1u << i))) perm[k++] = i;
  for (int i = 0; i < na; i++)
    perm[k++] = axes[i];
  for (int i = 0; i < nr; i++)
    inverse[perm[i]] = i;
  x = poly_tensor_permute(ctx, x, perm, nr);
  if (!x) return NULL;
  Value *scale_value = legacy ? n->inputs[1] : n->nin > 2 ? n->inputs[2] : NULL;
  Value *size_value = !legacy && n->nin > 3 ? n->inputs[3] : NULL;
  if (scale_value && !onnx_floats(d, scale_value, scales, ONNX_RANK, &ns)) return NULL;
  if (size_value && !onnx_ints(d, size_value, sizes, ONNX_RANK, &nz)) return NULL;
  int spatial = nr - 2;
  if ((!ns && !nz) || (ns && nz) || (ns && (ns < spatial || ns > nr)) ||
      (nz && (nz < spatial || nz > nr)))
    return NULL;
  const int64_t *input = shape(d, x);
  if (nz) {
    for (int i = 0; i < nz - spatial; i++)
      if (sizes[i] != input[nr - nz + i]) return NULL;
    for (int i = 0; i < spatial; i++) {
      sizes[i] = sizes[nz - spatial + i];
      if (sizes[i] <= 0) return NULL;
      scales[i] = (double)sizes[i] / input[i + 2];
    }
    if (strcmp(aspect, "stretch")) {
      bool smaller = !strcmp(aspect, "not_larger");
      if (!smaller && strcmp(aspect, "not_smaller")) return NULL;
      double ratio = scales[0];
      for (int i = 1; i < spatial; i++)
        if (smaller ? scales[i] < ratio : scales[i] > ratio) ratio = scales[i];
      for (int i = 0; i < spatial; i++) {
        scales[i] = ratio;
        sizes[i] = (int64_t)(ratio * input[i + 2] + .5);
      }
    }
  } else {
    for (int i = 0; i < ns - spatial; i++)
      if (scales[i] != 1) return NULL;
    for (int i = 0; i < spatial; i++) {
      scales[i] = scales[ns - spatial + i];
      double extent = scales[i] * input[i + 2];
      if (!(extent >= 1 && extent <= INT_MAX)) return NULL;
      sizes[i] = (int64_t)extent;
    }
  }
  PolyTensor *indexes[ONNX_RANK];
  for (int i = 0; i < spatial; i++) {
    if (sizes[i] <= 0 || sizes[i] > INT_MAX || !isfinite(scales[i]) || scales[i] <= 0) return NULL;
    PolyTensor *index = poly_tensor_arange_int(ctx, 0, sizes[i], 1, POLY_INT32, x->device);
    if (!strcmp(transform, "align_corners"))
      index =
          sizes[i] == 1
              ? onnx_int(d, 0)
              : poly_tensor_div(
                    ctx, poly_tensor_alu2(ctx, POLY_OP_MUL, index, onnx_int(d, input[i + 2] - 1)),
                    onnx_int(d, sizes[i] - 1), 0
                );
    else if (!strcmp(transform, "asymmetric"))
      index = poly_tensor_div(ctx, index, onnx_float(d, scales[i]), 0);
    else if (!strcmp(transform, "half_pixel") || !strcmp(transform, "pytorch_half_pixel") || !strcmp(transform, "half_pixel_symmetric")) {
      index = poly_tensor_div(
          ctx, poly_tensor_alu2(ctx, POLY_OP_ADD, index, onnx_float(d, .5)),
          onnx_float(d, scales[i]), 0
      );
      if (!strcmp(transform, "half_pixel_symmetric"))
        index = poly_tensor_alu2(
            ctx, POLY_OP_ADD,
            onnx_float(d, (input[i + 2] / 2.0) * (1 - sizes[i] / (input[i + 2] * scales[i]))), index
        );
      index = onnx_sub(d, index, onnx_float(d, .5));
      if (!strcmp(transform, "pytorch_half_pixel") && sizes[i] == 1) index = onnx_int(d, 0);
    } else
      return NULL;
    if (strcmp(mode, "cubic")) index = onnx_clamp(d, index, 0, input[i + 2] - 1);
    indexes[i] = index;
  }
  if (!strcmp(mode, "nearest")) {
    for (int i = 0; i < spatial; i++) {
      PolyTensor *index = indexes[i];
      if (!strcmp(nearest, "round_prefer_floor"))
        index = poly_tensor_ceil(ctx, onnx_sub(d, index, onnx_float(d, .5)));
      else if (!strcmp(nearest, "round_prefer_ceil"))
        index =
            poly_tensor_floor(ctx, poly_tensor_alu2(ctx, POLY_OP_ADD, index, onnx_float(d, .5)));
      else if (!strcmp(nearest, "floor"))
        index = poly_tensor_floor(ctx, index);
      else if (!strcmp(nearest, "ceil"))
        index = poly_tensor_ceil(ctx, index);
      else
        return NULL;
      int64_t rs[ONNX_RANK];
      for (int j = 0; j < spatial; j++)
        rs[j] = j == i ? sizes[j] : 1;
      indexes[i] = poly_tensor_expand(
          ctx, poly_tensor_reshape(ctx, poly_tensor_cast(ctx, index, POLY_INT32), rs, spatial),
          sizes, spatial
      );
    }
    x = onnx_index_axes(d, x, indexes, 2, spatial);
  } else {
    for (int i = 0; i < spatial; i++) {
      PolyTensor *index = indexes[i],
                 *low = poly_tensor_cast(ctx, poly_tensor_floor(ctx, index), POLY_INT32);
      int64_t rs[ONNX_RANK], big[ONNX_RANK];
      for (int j = 0; j < nr; j++) {
        rs[j] = j == i + 2 ? sizes[i] : 1;
        big[j] = j == i + 2 ? sizes[i] : shape(d, x)[j];
      }
      if (!strcmp(mode, "linear")) {
        PolyTensor *high = poly_tensor_cast(ctx, poly_tensor_ceil(ctx, index), POLY_INT32);
        PolyTensor *fraction = onnx_sub(d, index, poly_tensor_floor(ctx, index));
        low = poly_tensor_expand(ctx, poly_tensor_reshape(ctx, low, rs, nr), big, nr);
        high = poly_tensor_expand(ctx, poly_tensor_reshape(ctx, high, rs, nr), big, nr);
        fraction = poly_tensor_expand(ctx, poly_tensor_reshape(ctx, fraction, rs, nr), big, nr);
        PolyTensor *a = poly_tensor_gather_dim(ctx, x, i + 2, low),
                   *b = poly_tensor_gather_dim(ctx, x, i + 2, high);
        x = poly_tensor_alu2(
            ctx, POLY_OP_ADD, a, poly_tensor_alu2(ctx, POLY_OP_MUL, onnx_sub(d, b, a), fraction)
        );
      } else {
        double A = attr_float(d, n, "cubic_coeff_a", -.75);
        PolyTensor *ratio = onnx_sub(d, index, low), *weights[4], *neighbors[4];
        for (int j = 0; j < 4; j++) {
          neighbors[j] = poly_tensor_alu2(ctx, POLY_OP_ADD, low, onnx_int(d, j - 1));
          PolyTensor *distance = j == 0 ? poly_tensor_alu2(ctx, POLY_OP_ADD, ratio, onnx_int(d, 1))
                                 : j == 1 ? ratio
                                          : onnx_sub(d, onnx_int(d, j == 2 ? 1 : 2), ratio);
          /* polyN is Horner evaluation, not a reassociated cubic. */
          double coefficients[4] = {A, -5 * A, 8 * A, -4 * A};
          if (j == 1 || j == 2) {
            coefficients[0] = A + 2;
            coefficients[1] = -(A + 3);
            coefficients[2] = 0;
            coefficients[3] = 1;
          }
          PolyTensor *w = onnx_float(d, coefficients[0]);
          for (int c = 1; c < 4; c++)
            w = poly_tensor_alu2(
                ctx, POLY_OP_ADD, poly_tensor_alu2(ctx, POLY_OP_MUL, w, distance),
                onnx_float(d, coefficients[c])
            );
          if (attr_int(d, n, "exclude_outside", 0)) {
            PolyTensor *valid = poly_tensor_alu2(
                ctx, POLY_OP_AND,
                onnx_not(d, poly_tensor_alu2(ctx, POLY_OP_CMPLT, neighbors[j], onnx_int(d, 0))),
                poly_tensor_alu2(ctx, POLY_OP_CMPLT, neighbors[j], onnx_int(d, input[i + 2]))
            );
            w = poly_tensor_alu3(ctx, POLY_OP_WHERE, valid, w, onnx_int(d, 0));
          }
          weights[j] = w;
        }
        if (attr_int(d, n, "exclude_outside", 0)) {
          PolyTensor *total = weights[0];
          for (int j = 1; j < 4; j++)
            total = poly_tensor_alu2(ctx, POLY_OP_ADD, total, weights[j]);
          total = poly_tensor_alu2(ctx, POLY_OP_ADD, total, onnx_float(d, 1e-9));
          for (int j = 0; j < 4; j++)
            weights[j] = poly_tensor_div(ctx, weights[j], total, 0);
        }
        PolyTensor *sum = NULL;
        for (int j = 0; j < 4; j++) {
          PolyTensor *idx = poly_tensor_expand(
              ctx,
              poly_tensor_reshape(ctx, onnx_clamp(d, neighbors[j], 0, input[i + 2] - 1), rs, nr),
              big, nr
          );
          PolyTensor *w =
              poly_tensor_expand(ctx, poly_tensor_reshape(ctx, weights[j], rs, nr), big, nr);
          PolyTensor *v =
              poly_tensor_alu2(ctx, POLY_OP_MUL, poly_tensor_gather_dim(ctx, x, i + 2, idx), w);
          sum = sum ? poly_tensor_alu2(ctx, POLY_OP_ADD, sum, v) : v;
        }
        x = sum;
      }
      if (!x) return NULL;
    }
  }
  return x ? poly_tensor_permute(ctx, x, inverse, nr) : NULL;
}

static bool onnx_subgraph(Import *d, Bytes proto, Value **outputs, int *nout);

static PolyTensor *onnx_operation(Import *d, Node *n) {
  PolyCtx *ctx = d->ctx;
  PolyTensor *x = n->nin && n->inputs[0] ? n->inputs[0]->tensor : NULL;
  PolyTensor *y = n->nin > 1 && n->inputs[1] ? n->inputs[1]->tensor : NULL;
  const char *op = d->op;
  if (!strcmp(op, "If")) {
    if (!arity_outputs(d, n, 1, 1, ONNX_IO) || !attrs(d, n, "|then_branch||else_branch|") ||
        !poly_dtype_eq(x->uop_physical->dtype, POLY_BOOL) || n->inputs[0]->nbytes != 1)
      return NULL;
    Value *branches[2][ONNX_IO];
    for (int b = 0; b < 2; b++) {
      Attr *a = attr(n, b ? "then_branch" : "else_branch");
      Field graph;
      int count = 0;
      if (!a || a->type != 5 || !field(d, a->proto, 6, 2, &graph, true) ||
          !onnx_subgraph(d, graph.bytes, branches[b], &count))
        return NULL;
      if (count != n->nout) {
        fail(d, POLY_IMPORT_ERR_SHAPE_MISMATCH, "If branch output count mismatch");
        return NULL;
      }
    }
    /* Pinned If lowers equal-shaped branches with WHERE. Do not evaluate a
     * runtime condition on the host to freeze a data-dependent output shape. */
    for (int i = 0; i < n->nout; i++) {
      Value *a = branches[0][i], *b = branches[1][i];
      if (!poly_dtype_eq(a->dtype, b->dtype) || a->rank != b->rank ||
          (a->rank && memcmp(a->shape, b->shape, a->rank * sizeof(int64_t)))) {
        fail(d, POLY_IMPORT_ERR_SHAPE_MISMATCH, "If requires equal branch output shapes/dtypes");
        return NULL;
      }
      n->captures_runtime |= !a->constant || !b->constant;
      n->extra[i] = poly_tensor_alu3(ctx, POLY_OP_WHERE, x, b->tensor, a->tensor);
      if (!n->extra[i]) return NULL;
    }
    return n->extra[0];
  }
  if ((!strcmp(op, "Mish") || !strncmp(op, "Bitwise", 7)) && d->opset < 18) return NULL;
  if (!strcmp(op, "HardSwish") && d->opset < 14) return NULL;
  if (!strcmp(op, "CastLike") && d->opset < 15) return NULL;
  if ((!strcmp(op, "Gelu") && d->opset < 20) ||
      (!strcmp(op, "GroupNormalization") && d->opset < 21))
    return NULL;
  if (!strcmp(op, "Resize") || !strcmp(op, "Upsample"))
    return onnx_resize(d, n, !strcmp(op, "Upsample"));
  if (!strcmp(op, "CenterCropPad")) {
    if (d->opset < 18 || !arity(d, n, 2, 2) || !attrs(d, n, "|axes|")) return NULL;
    int nr = rank(d, x), ns, na;
    int64_t sizes[ONNX_RANK], axes[ONNX_RANK], crop[ONNX_RANK][2], pad[ONNX_RANK][2] = {{0}};
    if (!onnx_ints(d, n->inputs[1], sizes, ONNX_RANK, &ns) ||
        !attr_ints(d, n, "axes", axes, ONNX_RANK, &na))
      return NULL;
    if (!na)
      for (int i = 0; i < nr; i++)
        axes[na++] = i;
    if (ns != na) return NULL;
    for (int i = 0; i < nr; i++) {
      crop[i][0] = 0;
      crop[i][1] = shape(d, x)[i];
    }
    unsigned used = 0;
    for (int i = 0; i < na; i++) {
      if (!onnx_axis(d, &axes[i], nr) || sizes[i] <= 0 || (uint64_t)sizes[i] > ONNX_BYTES ||
          (used & (1u << axes[i])))
        return NULL;
      int a = (int)axes[i];
      used |= 1u << a;
      int64_t old = crop[a][1], size = sizes[i];
      if (size < old) {
        crop[a][0] = old / 2 - (size + 1) / 2;
        crop[a][1] = old / 2 + size / 2;
      } else {
        pad[a][0] = (size - old) / 2;
        pad[a][1] = (size - old + 1) / 2;
      }
    }
    return poly_tensor_pad_value_int(ctx, poly_tensor_shrink(ctx, x, crop, nr), pad, nr, 0);
  }
  if (!strcmp(op, "LRN")) {
    if (!arity(d, n, 1, 1) || !attrs(d, n, "|size||alpha||beta||bias|") || rank(d, x) != 4)
      return NULL;
    int64_t size = attr_int(d, n, "size", 0);
    if (size <= 0 || (uint64_t)size > ONNX_BYTES) return NULL;
    int64_t s[4];
    memcpy(s, shape(d, x), sizeof(s));
    int64_t dims[4] = {s[0], 1, s[1], s[2] * s[3]};
    int64_t pad[4][2] = {{0}, {0}, {(size - 1) / 2, size / 2}, {0}};
    int64_t kernel[2] = {size, 1}, stride[2] = {1, 1}, dilation[2] = {1, 1}, padding[4] = {0};
    PolyTensor *squared = poly_tensor_alu2(ctx, POLY_OP_MUL, x, x);
    PolyTensor *pooled = poly_tensor_avg_pool2d(
        ctx, poly_tensor_pad_value_int(ctx, poly_tensor_reshape(ctx, squared, dims, 4), pad, 4, 0),
        kernel, 2, stride, dilation, padding, 4, false, true
    );
    if (!pooled) return NULL;
    PolyTensor *den = poly_tensor_alu2(
        ctx, POLY_OP_ADD,
        scale(d, poly_tensor_reshape(ctx, pooled, s, 4), attr_float(d, n, "alpha", 1e-4)),
        onnx_float(d, attr_float(d, n, "bias", 1))
    );
    return poly_tensor_div(
        ctx, x,
        poly_tensor_alu2(ctx, POLY_OP_POW, den, onnx_float(d, attr_float(d, n, "beta", .75))), 0
    );
  }
  if (!strcmp(op, "ArrayFeatureExtractor")) {
    if (!arity(d, n, 2, 2) || !attrs(d, n, "") || rank(d, x) < 1 ||
        !poly_dtype_is_int(y->uop_physical->dtype))
      return NULL;
    return onnx_index_axes(d, x, &y, rank(d, x) - 1, 1);
  }
  if (!strcmp(op, "Binarizer")) {
    if (!arity(d, n, 1, 1) || !attrs(d, n, "|threshold|")) return NULL;
    return poly_tensor_cast(
        ctx,
        poly_tensor_alu2(ctx, POLY_OP_CMPLT, onnx_float(d, attr_float(d, n, "threshold", 0)), x),
        POLY_FLOAT32
    );
  }
  if (!strcmp(op, "BiasGelu") || !strcmp(op, "FastGelu")) {
    bool fast = !strcmp(op, "FastGelu");
    if (!arity(d, n, fast ? 1 : 2, 2) || !attrs(d, n, "") || (y && !broadcast_into(d, x, y)))
      return NULL;
    if (y) x = poly_tensor_alu2(ctx, POLY_OP_ADD, x, y);
    return fast ? poly_tensor_gelu(ctx, x) : poly_tensor_gelu_exact(ctx, x);
  }
  if (!strcmp(op, "SkipLayerNormalization")) {
    if (!arity_outputs(d, n, 3, 5, 4) || !attrs(d, n, "|epsilon|") || !broadcast(d, x, y))
      return NULL;
    x = poly_tensor_alu2(ctx, POLY_OP_ADD, x, y);
    if (n->nin > 4 && n->inputs[4]) x = poly_tensor_alu2(ctx, POLY_OP_ADD, x, n->inputs[4]->tensor);
    n->extra[3] = x;
    int64_t axis = rank(d, x) - 1;
    PolyTensor *center = onnx_sub(d, x, poly_tensor_mean(ctx, x, &axis, 1, true));
    PolyTensor *variance =
        poly_tensor_mean(ctx, poly_tensor_alu2(ctx, POLY_OP_MUL, center, center), &axis, 1, true);
    PolyTensor *inv = poly_tensor_reciprocal(
        ctx, poly_tensor_alu1(
                 ctx, POLY_OP_SQRT,
                 poly_tensor_alu2(
                     ctx, POLY_OP_ADD, variance, onnx_float(d, attr_float(d, n, "epsilon", 1e-12))
                 )
             )
    );
    PolyTensor *out = poly_tensor_alu2(
        ctx, POLY_OP_MUL, poly_tensor_alu2(ctx, POLY_OP_MUL, center, inv), n->inputs[2]->tensor
    );
    return n->nin > 3 && n->inputs[3]
               ? poly_tensor_alu2(ctx, POLY_OP_ADD, out, n->inputs[3]->tensor)
               : out;
  }
  if (!strcmp(op, "QLinearAdd") || !strcmp(op, "QLinearMul") ||
      !strcmp(op, "QLinearGlobalAveragePool")) {
    bool pool = !strcmp(op, "QLinearGlobalAveragePool"), add = !strcmp(op, "QLinearAdd");
    if (!arity(d, n, pool ? 5 : 8, pool ? 5 : 8) || !attrs(d, n, pool ? "|channels_last|" : ""))
      return NULL;
    PolyTensor *a = onnx_sub(d, poly_tensor_cast(ctx, x, POLY_INT32), n->inputs[2]->tensor), *out;
    if (pool || add) a = poly_tensor_alu2(ctx, POLY_OP_MUL, a, n->inputs[1]->tensor);
    int64_t order[ONNX_RANK], inverse[ONNX_RANK];
    int nr = rank(d, a);
    bool channels_last = pool && attr_int(d, n, "channels_last", 0);
    if (pool) {
      if (nr < 3) return NULL;
      for (int i = 0; i < nr; i++) {
        order[i] = i == 0 ? 0 : i == 1 ? nr - 1 : i - 1;
        inverse[order[i]] = i;
      }
      if (channels_last) a = poly_tensor_permute(ctx, a, order, nr);
      int64_t axes[ONNX_RANK];
      for (int i = 2; i < nr; i++)
        axes[i - 2] = i;
      out = poly_tensor_mean(ctx, a, axes, nr - 2, true);
    } else {
      PolyTensor *b = onnx_sub(
          d, poly_tensor_cast(ctx, n->inputs[3]->tensor, POLY_INT32), n->inputs[5]->tensor
      );
      if (add) b = poly_tensor_alu2(ctx, POLY_OP_MUL, b, n->inputs[4]->tensor);
      out = poly_tensor_alu2(ctx, add ? POLY_OP_ADD : POLY_OP_MUL, a, b);
      if (!add)
        out = poly_tensor_alu2(
            ctx, POLY_OP_MUL, out,
            poly_tensor_alu2(ctx, POLY_OP_MUL, n->inputs[1]->tensor, n->inputs[4]->tensor)
        );
    }
    int scale_index = pool ? 3 : 6;
    PolyTensor *zero = n->inputs[scale_index + 1]->tensor;
    out = onnx_quant_cast(
        d,
        poly_tensor_alu2(
            ctx, POLY_OP_ADD,
            poly_tensor_round(ctx, poly_tensor_div(ctx, out, n->inputs[scale_index]->tensor, 0)),
            zero
        ),
        zero->uop_physical->dtype
    );
    return channels_last ? poly_tensor_permute(ctx, out, inverse, nr) : out;
  }
  if (!strcmp(op, "ConstantOfShape")) {
    if (!arity(d, n, 1, 1) || !attrs(d, n, "|value|")) return NULL;
    int64_t dims[ONNX_RANK];
    int nr;
    if (!onnx_ints(d, n->inputs[0], dims, ONNX_RANK, &nr)) return NULL;
    for (int i = 0; i < nr; i++)
      if (dims[i] <= 0) return NULL;
    PolyTensor *value;
    Attr *a = attr(n, "value");
    if (a) {
      Field tensor;
      char name[ONNX_NAME];
      int serial = d->nvalues;
      do {
        snprintf(name, sizeof(name), "__onnx_constant_%d", serial++);
      } while (lookup(d, name));
      if (a->type != 4 || !field(d, a->proto, 5, 2, &tensor, true) ||
          !onnx_initializer(d, tensor.bytes, name))
        return NULL;
      Value *v = lookup(d, name);
      if (v->nbytes != (size_t)poly_dtype_itemsize(v->dtype)) return NULL;
      value = poly_tensor_reshape(ctx, v->tensor, dims, 0);
    } else
      value = poly_tensor_cast(ctx, onnx_int(d, 0), POLY_FLOAT32);
    return poly_tensor_expand(ctx, value, dims, nr);
  }
  if (!strcmp(op, "OneHot")) {
    if (!arity(d, n, 3, 3) || !attrs(d, n, "|axis|")) return NULL;
    int64_t depth, axis = attr_int(d, n, "axis", -1);
    int count, nr = rank(d, x);
    if (!onnx_ints(d, n->inputs[1], &depth, 1, &count) || count != 1 || depth <= 0 ||
        depth > INT_MAX || nr == ONNX_RANK || !onnx_axis(d, &axis, nr + 1))
      return NULL;
    PolyTensor *values = n->inputs[2]->tensor;
    if (rank(d, values) != 1 || shape(d, values)[0] != 2) return NULL;
    x = poly_tensor_cast(ctx, x, POLY_INT32);
    x = poly_tensor_alu3(
        ctx, POLY_OP_WHERE, poly_tensor_alu2(ctx, POLY_OP_CMPLT, x, onnx_int(d, 0)),
        poly_tensor_alu2(ctx, POLY_OP_ADD, x, onnx_int(d, depth)), x
    );
    PolyTensor *hot = poly_tensor_one_hot(ctx, x, depth);
    int64_t perm[ONNX_RANK];
    for (int i = 0; i <= nr; i++)
      perm[i] = i < axis ? i : i == axis ? nr : i - 1;
    hot = poly_tensor_permute(ctx, hot, perm, nr + 1);
    int64_t pair[1][2] = {{0, 1}}, scalar[1];
    PolyTensor *off = poly_tensor_reshape(ctx, poly_tensor_shrink(ctx, values, pair, 1), scalar, 0);
    pair[0][0] = 1;
    pair[0][1] = 2;
    PolyTensor *on = poly_tensor_reshape(ctx, poly_tensor_shrink(ctx, values, pair, 1), scalar, 0);
    return poly_tensor_alu3(ctx, POLY_OP_WHERE, poly_tensor_cast(ctx, hot, POLY_BOOL), on, off);
  }
  if (!strcmp(op, "HannWindow") || !strcmp(op, "HammingWindow") || !strcmp(op, "BlackmanWindow")) {
    if (!arity(d, n, 1, 1) || !attrs(d, n, "|output_datatype||periodic|")) return NULL;
    int64_t size;
    int count;
    PolyDType dt;
    if (!onnx_ints(d, n->inputs[0], &size, 1, &count) || count != 1 || size < 1 || size > INT_MAX ||
        !dtype(d, (uint64_t)attr_int(d, n, "output_datatype", 1), &dt))
      return NULL;
    int64_t period = attr_int(d, n, "periodic", 1) ? size : size - 1;
    if (!period) return NULL;
    double pi = 3.14159265358979323846, a = 0.5, b = 0.5, c = 0;
    if (!strcmp(op, "HammingWindow")) {
      a = 25.0 / 46;
      b = 21.0 / 46;
    }
    if (!strcmp(op, "BlackmanWindow")) {
      a = .42;
      b = .5;
      c = .08;
    }
    PolyTensor *index = poly_tensor_arange_int(ctx, 0, size, 1, POLY_INT32, x->device);
    PolyTensor *first = scale(d, poly_tensor_cos(ctx, scale(d, index, 2 * pi / period)), b);
    PolyTensor *second = scale(d, poly_tensor_cos(ctx, scale(d, index, 4 * pi / period)), c);
    return poly_tensor_cast(
        ctx, poly_tensor_alu2(ctx, POLY_OP_ADD, onnx_sub(d, onnx_float(d, a), first), second), dt
    );
  }
  if (!strcmp(op, "QuantizeLinear") || !strcmp(op, "DequantizeLinear")) {
    bool quantize = !strcmp(op, "QuantizeLinear");
    if (!arity(d, n, 2, 3) ||
        !attrs(
            d, n, quantize ? "|axis||saturate||block_size||output_dtype|" : "|axis||block_size|"
        ) ||
        attr_int(d, n, "block_size", 0) != 0 || !poly_dtype_is_float(y->uop_physical->dtype))
      return NULL;
    PolyTensor *zero = n->nin > 2 && n->inputs[2] ? n->inputs[2]->tensor : NULL;
    PolyDType output = POLY_UINT8;
    if (quantize && attr(n, "output_dtype") && attr_int(d, n, "output_dtype", 0) != 0 &&
        !dtype(d, (uint64_t)attr_int(d, n, "output_dtype", 0), &output))
      return NULL;
    if (zero) output = zero->uop_physical->dtype;
    if (!zero)
      zero = poly_tensor_cast(ctx, onnx_int(d, 0), quantize ? output : x->uop_physical->dtype);
    int64_t axis = attr_int(d, n, "axis", 1);
    y = onnx_quant_parameter(d, x, y, axis);
    zero = onnx_quant_parameter(d, x, zero, axis);
    if (!y || !zero) return NULL;
    if (!quantize)
      return poly_tensor_cast(
          ctx,
          poly_tensor_alu2(
              ctx, POLY_OP_MUL, onnx_sub(d, poly_tensor_cast(ctx, x, POLY_INT32), zero), y
          ),
          y->uop_physical->dtype
      );
    PolyTensor *v = poly_tensor_div(ctx, x, y, 0);
    if (poly_dtype_eq(output, POLY_UINT8))
      v = poly_tensor_cast(
          ctx,
          poly_tensor_alu2(
              ctx, POLY_OP_ADD, poly_tensor_alu2(ctx, POLY_OP_ADD, v, onnx_float(d, .4999999)), zero
          ),
          POLY_INT32
      );
    else
      v = poly_tensor_alu2(ctx, POLY_OP_ADD, poly_tensor_round(ctx, v), zero);
    return poly_tensor_contiguous(ctx, onnx_quant_cast(d, v, output));
  }
  if (!strcmp(op, "DynamicQuantizeLinear")) {
    if (!arity_outputs(d, n, 1, 1, 3) || !attrs(d, n, "") ||
        !poly_dtype_eq(x->uop_physical->dtype, POLY_FLOAT32))
      return NULL;
    int nr = rank(d, x);
    int64_t axes[ONNX_RANK];
    for (int i = 0; i < nr; i++)
      axes[i] = i;
    PolyTensor *maximum = poly_tensor_max(ctx, x, axes, nr, false),
               *minimum = poly_tensor_min(ctx, x, axes, nr, false);
    PolyTensor *scale = poly_tensor_div(
        ctx,
        poly_tensor_alu2(
            ctx, POLY_OP_ADD, poly_tensor_alu2(ctx, POLY_OP_MAX, maximum, onnx_int(d, 0)),
            poly_tensor_alu2(
                ctx, POLY_OP_MAX, poly_tensor_alu1(ctx, POLY_OP_NEG, minimum), onnx_int(d, 0)
            )
        ),
        onnx_int(d, 255), 0
    );
    PolyTensor *zero = onnx_quant_cast(
        d,
        poly_tensor_round(
            ctx, poly_tensor_alu1(ctx, POLY_OP_NEG, poly_tensor_div(ctx, minimum, scale, 0))
        ),
        POLY_UINT8
    );
    n->extra[1] = scale;
    n->extra[2] = zero;
    return onnx_quant_cast(
        d,
        poly_tensor_alu2(
            ctx, POLY_OP_ADD, poly_tensor_round(ctx, poly_tensor_div(ctx, x, scale, 0)), zero
        ),
        POLY_UINT8
    );
  }
  if (!strcmp(op, "MatMulInteger") || !strcmp(op, "QLinearMatMul")) {
    bool quantized = !strcmp(op, "QLinearMatMul");
    if (!arity(d, n, quantized ? 8 : 2, quantized ? 8 : 4) || !attrs(d, n, "")) return NULL;
    PolyTensor *a = x, *b = quantized ? n->inputs[3]->tensor : y;
    if (!poly_dtype_is_int(a->uop_physical->dtype) || !poly_dtype_is_int(b->uop_physical->dtype))
      return NULL;
    PolyTensor *az = quantized                    ? n->inputs[2]->tensor
                     : n->nin > 2 && n->inputs[2] ? n->inputs[2]->tensor
                                                  : onnx_int(d, 0);
    PolyTensor *bz = quantized                    ? n->inputs[5]->tensor
                     : n->nin > 3 && n->inputs[3] ? n->inputs[3]->tensor
                                                  : onnx_int(d, 0);
    a = onnx_sub(d, poly_tensor_cast(ctx, a, POLY_INT32), az);
    b = onnx_sub(d, poly_tensor_cast(ctx, b, POLY_INT32), bz);
    PolyTensor *out = matmul(d, a, b);
    if (!out || !quantized) return out;
    PolyTensor *scales =
        poly_tensor_alu2(ctx, POLY_OP_MUL, n->inputs[1]->tensor, n->inputs[4]->tensor);
    out = poly_tensor_round(
        ctx, poly_tensor_div(
                 ctx, poly_tensor_alu2(ctx, POLY_OP_MUL, out, scales), n->inputs[6]->tensor, 0
             )
    );
    return onnx_quant_cast(
        d, poly_tensor_alu2(ctx, POLY_OP_ADD, out, n->inputs[7]->tensor), n->inputs[7]->dtype
    );
  }
  if (!strcmp(op, "ScatterElements") || !strcmp(op, "Scatter")) {
    if (!arity(d, n, 3, 3) || !attrs(d, n, "|axis||reduction|") ||
        !poly_dtype_is_int(y->uop_physical->dtype))
      return NULL;
    int64_t axis = attr_int(d, n, "axis", 0);
    char reduction[ONNX_NAME];
    if (!onnx_axis(d, &axis, rank(d, x)) || !attr_text(d, n, "reduction", "none", reduction))
      return NULL;
    y = poly_tensor_alu2(
        ctx, POLY_OP_ADD, y,
        poly_tensor_alu3(
            ctx, POLY_OP_WHERE, poly_tensor_alu2(ctx, POLY_OP_CMPLT, y, onnx_int(d, 0)),
            onnx_int(d, shape(d, x)[axis]), onnx_int(d, 0)
        )
    );
    PolyTensor *updates = n->inputs[2]->tensor;
    if (!strcmp(reduction, "none")) return poly_tensor_scatter(ctx, x, (int)axis, y, updates, NULL);
    const char *kind = !strcmp(reduction, "add")   ? "sum"
                       : !strcmp(reduction, "mul") ? "prod"
                       : !strcmp(reduction, "min") ? "amin"
                       : !strcmp(reduction, "max") ? "amax"
                                                   : NULL;
    return kind ? poly_tensor_scatter_reduce(ctx, x, (int)axis, y, updates, kind, true) : NULL;
  }
  if (!strcmp(op, "GatherND")) {
    if (!arity(d, n, 2, 2) || !attrs(d, n, "|batch_dims|") || rank(d, y) < 1 ||
        !poly_dtype_is_int(y->uop_physical->dtype))
      return NULL;
    int nx = rank(d, x), ni = rank(d, y);
    int64_t batch = attr_int(d, n, "batch_dims", 0), index_count = shape(d, y)[ni - 1];
    if (batch < 0 || batch >= ni || batch >= nx || index_count < 1 || index_count > nx - batch)
      return NULL;
    int64_t xs[ONNX_RANK], is[ONNX_RANK], batch_shape[ONNX_RANK], total = 1;
    memcpy(xs, shape(d, x), nx * sizeof(int64_t));
    memcpy(is, shape(d, y), ni * sizeof(int64_t));
    for (int i = 0; i < batch; i++) {
      if (xs[i] != is[i]) return NULL;
      batch_shape[i] = xs[i];
      total *= xs[i];
    }
    PolyTensor *indices[ONNX_RANK];
    int nt = 0;
    if (batch) {
      int64_t xr[ONNX_RANK] = {total}, ir[ONNX_RANK] = {total};
      memcpy(xr + 1, xs + batch, (nx - batch) * sizeof(int64_t));
      memcpy(ir + 1, is + batch, (ni - batch) * sizeof(int64_t));
      x = poly_tensor_reshape(ctx, x, xr, nx - (int)batch + 1);
      y = poly_tensor_reshape(ctx, y, ir, ni - (int)batch + 1);
      ni = rank(d, y);
      int64_t bs[ONNX_RANK], big[ONNX_RANK];
      for (int i = 0; i < ni - 1; i++) {
        bs[i] = i == 0 ? total : 1;
        big[i] = shape(d, y)[i];
      }
      indices[nt++] = poly_tensor_expand(
          ctx,
          poly_tensor_reshape(
              ctx, poly_tensor_arange_int(ctx, 0, total, 1, POLY_INT32, x->device), bs, ni - 1
          ),
          big, ni - 1
      );
    }
    for (int j = 0; j < index_count; j++) {
      int64_t pairs[ONNX_RANK][2], rs[ONNX_RANK];
      for (int i = 0; i < ni; i++) {
        pairs[i][0] = 0;
        pairs[i][1] = shape(d, y)[i];
        rs[i] = shape(d, y)[i];
      }
      pairs[ni - 1][0] = j;
      pairs[ni - 1][1] = j + 1;
      indices[nt++] = poly_tensor_reshape(ctx, poly_tensor_shrink(ctx, y, pairs, ni), rs, ni - 1);
    }
    PolyTensor *out = onnx_index_axes(d, x, indices, 0, nt);
    if (!out || !batch) return out;
    int64_t result[ONNX_RANK];
    int nr = 0;
    for (int i = 0; i < batch; i++)
      result[nr++] = batch_shape[i];
    for (int i = 1; i < rank(d, out); i++)
      result[nr++] = shape(d, out)[i];
    return poly_tensor_reshape(ctx, out, result, nr);
  }
  if (!strcmp(op, "Trilu")) {
    if (!arity(d, n, 1, 2) || !attrs(d, n, "|upper|") || rank(d, x) < 2) return NULL;
    int64_t k = 0;
    int count;
    if (n->nin > 1 && n->inputs[1] && (!onnx_ints(d, n->inputs[1], &k, 1, &count) || count != 1))
      return NULL;
    if (k < INT_MIN || k > INT_MAX) return NULL;
    return attr_int(d, n, "upper", 1) ? poly_tensor_triu(ctx, x, (int)k)
                                      : poly_tensor_tril(ctx, x, (int)k);
  }
  if (!strcmp(op, "Slice")) {
    if (!arity(d, n, 3, 5) || !attrs(d, n, "")) return NULL;
    int64_t begin[ONNX_RANK], end[ONNX_RANK], axes[ONNX_RANK], strides[ONNX_RANK];
    int count, ne, na, ns, nr = rank(d, x);
    if (!onnx_ints(d, n->inputs[1], begin, ONNX_RANK, &count) ||
        !onnx_ints(d, n->inputs[2], end, ONNX_RANK, &ne) || count != ne)
      return NULL;
    for (int i = 0; i < count; i++) {
      axes[i] = i;
      strides[i] = 1;
    }
    if (n->nin > 3 && n->inputs[3] &&
        (!onnx_ints(d, n->inputs[3], axes, ONNX_RANK, &na) || na != count))
      return NULL;
    if (n->nin > 4 && n->inputs[4] &&
        (!onnx_ints(d, n->inputs[4], strides, ONNX_RANK, &ns) || ns != count))
      return NULL;
    int kinds[ONNX_RANK];
    PolyUOp *starts[ONNX_RANK], *sizes[ONNX_RANK];
    int64_t steps[ONNX_RANK];
    for (int i = 0; i < nr; i++) {
      kinds[i] = POLY_INDEX_SLICE;
      steps[i] = 1;
      starts[i] = poly_uop_const_int(ctx, 0);
      sizes[i] = poly_uop_const_int(ctx, shape(d, x)[i]);
    }
    unsigned mask = 0;
    for (int i = 0; i < count; i++) {
      if (!onnx_axis(d, &axes[i], nr) || (mask & (1u << axes[i])) || !strides[i] ||
          strides[i] == INT64_MIN)
        return NULL;
      mask |= 1u << axes[i];
      int a = (int)axes[i];
      int64_t dim = shape(d, x)[a];
      int64_t lo = begin[i], hi = end[i], lower = strides[i] > 0 ? 0 : -1,
              upper = strides[i] > 0 ? dim : dim - 1;
      if (lo < 0) lo += dim;
      if (hi < 0) hi += dim;
      lo = lo < lower ? lower : lo > upper ? upper : lo;
      hi = hi < lower ? lower : hi > upper ? upper : hi;
      int64_t start = strides[i] > 0 ? lo : hi + 1, span = strides[i] > 0 ? hi - lo : lo - hi;
      if (span <= 0) return NULL;
      starts[a] = poly_uop_const_int(ctx, start);
      sizes[a] = poly_uop_const_int(ctx, span);
      steps[a] = strides[i];
    }
    PolyTensor *indices[ONNX_RANK] = {0};
    return poly_tensor_getitem(ctx, x, kinds, starts, sizes, steps, indices, nr);
  }
  if (!strcmp(op, "Pad")) {
    if (!arity(d, n, 2, d->opset >= 18 ? 4 : 3) || !attrs(d, n, "|mode|")) return NULL;
    int64_t pads[ONNX_RANK * 2], axes[ONNX_RANK], pairs[ONNX_RANK][2] = {{0}};
    int count, na = rank(d, x), nr = na;
    if (!onnx_ints(d, n->inputs[1], pads, ONNX_RANK * 2, &count)) return NULL;
    for (int i = 0; i < nr; i++)
      axes[i] = i;
    if (n->nin > 3 && n->inputs[3] && !onnx_ints(d, n->inputs[3], axes, ONNX_RANK, &na))
      return NULL;
    if (count != 2 * na) return NULL;
    unsigned mask = 0;
    for (int i = 0; i < na; i++) {
      if (!onnx_axis(d, &axes[i], nr) || (mask & (1u << axes[i]))) return NULL;
      mask |= 1u << axes[i];
      pairs[axes[i]][0] = pads[i];
      pairs[axes[i]][1] = pads[i + na];
      if (pads[i] > INT_MAX || pads[i] < -INT_MAX || pads[i + na] > INT_MAX ||
          pads[i + na] < -INT_MAX)
        return NULL;
    }
    char mode[ONNX_NAME];
    if (!attr_text(d, n, "mode", "constant", mode)) return NULL;
    if (strcmp(mode, "constant")) {
      int id = !strcmp(mode, "wrap")      ? 1
               : !strcmp(mode, "reflect") ? 2
               : !strcmp(mode, "edge")    ? 3
                                          : 0;
      return id ? poly_tensor_pad_mode(ctx, x, &pairs[0][0], nr, id) : NULL;
    }
    Value *v = n->nin > 2 ? n->inputs[2] : NULL;
    if (v) {
      if (v->rank != 0 || !same_dtype(x, v->tensor) || !onnx_const_data(d, v)) return NULL;
      if (poly_dtype_eq(v->dtype, POLY_FLOAT32)) {
        float f;
        memcpy(&f, v->host, 4);
        return poly_tensor_pad_value_float(ctx, x, pairs, nr, f);
      }
      if (poly_dtype_eq(v->dtype, POLY_FLOAT64)) {
        double f;
        memcpy(&f, v->host, 8);
        return poly_tensor_pad_value_float(ctx, x, pairs, nr, f);
      }
      if (poly_dtype_eq(v->dtype, POLY_INT64)) {
        int64_t k;
        memcpy(&k, v->host, 8);
        return poly_tensor_pad_value_int(ctx, x, pairs, nr, k);
      }
      if (poly_dtype_eq(v->dtype, POLY_INT32)) {
        int32_t k;
        memcpy(&k, v->host, 4);
        return poly_tensor_pad_value_int(ctx, x, pairs, nr, k);
      }
      return NULL;
    }
    return poly_tensor_pad_value_int(ctx, x, pairs, nr, 0);
  }
  if (!strcmp(op, "Cast") || !strcmp(op, "CastLike")) {
    bool like = !strcmp(op, "CastLike");
    PolyDType dt;
    if (!arity(d, n, like ? 2 : 1, like ? 2 : 1) ||
        !attrs(d, n, like ? "|saturate|" : "|to||saturate|"))
      return NULL;
    if (like)
      dt = y->uop_physical->dtype;
    else if (!attr(n, "to") || !dtype(d, (uint64_t)attr_int(d, n, "to", 0), &dt))
      return NULL;
    return poly_tensor_cast(ctx, x, dt);
  }
  if (!strcmp(op, "Where")) {
    if (!arity(d, n, 3, 3) || !attrs(d, n, "") ||
        !poly_dtype_eq(x->uop_physical->dtype, POLY_BOOL) || !broadcast(d, y, n->inputs[2]->tensor))
      return NULL;
    return poly_tensor_alu3(ctx, POLY_OP_WHERE, x, y, n->inputs[2]->tensor);
  }
  if (!strcmp(op, "Less") || !strcmp(op, "Greater") || !strcmp(op, "Equal") ||
      !strcmp(op, "LessOrEqual") || !strcmp(op, "GreaterOrEqual")) {
    if (!arity(d, n, 2, 2) || !attrs(d, n, "") || !broadcast(d, x, y)) return NULL;
    if (!strcmp(op, "Equal")) return poly_tensor_alu2(ctx, POLY_OP_CMPEQ, x, y);
    bool reverse = !strcmp(op, "Greater") || !strcmp(op, "LessOrEqual");
    PolyTensor *cmp = poly_tensor_alu2(ctx, POLY_OP_CMPLT, reverse ? y : x, reverse ? x : y);
    return strstr(op, "OrEqual") ? onnx_not(d, cmp) : cmp;
  }
  if (!strcmp(op, "And") || !strcmp(op, "Or") || !strcmp(op, "Xor") || !strcmp(op, "BitwiseAnd") ||
      !strcmp(op, "BitwiseOr") || !strcmp(op, "BitwiseXor")) {
    if (!arity(d, n, 2, 2) || !attrs(d, n, "") || !broadcast(d, x, y)) return NULL;
    bool bitwise = !strncmp(op, "Bitwise", 7);
    if (bitwise ? !poly_dtype_is_int(x->uop_physical->dtype)
                : !poly_dtype_eq(x->uop_physical->dtype, POLY_BOOL))
      return NULL;
    return poly_tensor_alu2(
        ctx,
        strstr(op, "And")   ? POLY_OP_AND
        : strstr(op, "Xor") ? POLY_OP_XOR
                            : POLY_OP_OR,
        x, y
    );
  }
  if (!strcmp(op, "Not") || !strcmp(op, "BitwiseNot")) {
    if (!arity(d, n, 1, 1) || !attrs(d, n, "")) return NULL;
    if (!strcmp(op, "Not"))
      return poly_dtype_eq(x->uop_physical->dtype, POLY_BOOL) ? onnx_not(d, x) : NULL;
    return poly_dtype_is_int(x->uop_physical->dtype) ? poly_tensor_bitwise_not(ctx, x) : NULL;
  }
  if (!strcmp(op, "Squeeze") || !strcmp(op, "Unsqueeze")) {
    bool unsqueeze = !strcmp(op, "Unsqueeze");
    if (!arity(d, n, unsqueeze ? 2 : 1, 2) || !attrs(d, n, "")) return NULL;
    int64_t axes[ONNX_RANK], dims[ONNX_RANK];
    int count = 0, nr = rank(d, x), out = 0;
    const int64_t *sx = shape(d, x);
    bool has_axes = n->nin > 1 && n->inputs[1];
    if (has_axes && !onnx_ints(d, n->inputs[1], axes, ONNX_RANK, &count)) return NULL;
    int total = nr + (unsqueeze ? count : 0);
    if (total > ONNX_RANK) return NULL;
    unsigned mask = 0;
    for (int i = 0; i < count; i++) {
      if (!onnx_axis(d, &axes[i], total) || (mask & (1u << axes[i]))) return NULL;
      mask |= 1u << axes[i];
    }
    for (int i = 0, j = 0; i < total; i++) {
      if (unsqueeze)
        dims[out++] = mask & (1u << i) ? 1 : sx[j++];
      else if ((has_axes && (mask & (1u << i))) || (!has_axes && sx[i] == 1)) {
        if (sx[i] != 1) return NULL;
      } else
        dims[out++] = sx[i];
    }
    return poly_tensor_reshape(ctx, x, dims, out);
  }
  if (!strcmp(op, "Concat")) {
    if (!arity(d, n, 1, ONNX_IO) || !attrs(d, n, "|axis|") || !attr(n, "axis")) return NULL;
    int nr = rank(d, x);
    int64_t axis = attr_int(d, n, "axis", 0);
    if (!onnx_axis(d, &axis, nr)) return NULL;
    PolyTensor *xs[ONNX_IO];
    for (int i = 0; i < n->nin; i++) {
      if (!n->inputs[i] || rank(d, n->inputs[i]->tensor) != nr ||
          !same_dtype(x, n->inputs[i]->tensor))
        return NULL;
      xs[i] = n->inputs[i]->tensor;
      for (int j = 0; j < nr; j++)
        if (j != axis && shape(d, xs[i])[j] != shape(d, x)[j]) return NULL;
    }
    return poly_tensor_cat(ctx, xs, n->nin, (int)axis);
  }
  if (!strcmp(op, "Expand") || !strcmp(op, "Tile")) {
    if (!arity(d, n, 2, 2) || !attrs(d, n, "")) return NULL;
    int64_t target[ONNX_RANK], reshaped[ONNX_RANK * 2], expanded[ONNX_RANK * 2];
    int count, nr = rank(d, x);
    if (!onnx_ints(d, n->inputs[1], target, ONNX_RANK, &count)) return NULL;
    const int64_t *sx = shape(d, x);
    if (!strcmp(op, "Expand")) {
      int size = nr > count ? nr : count;
      for (int i = 0; i < size; i++) {
        int64_t a = i < nr ? sx[nr - 1 - i] : 1, b = i < count ? target[count - 1 - i] : 1;
        if (b < 1 || (a != b && a != 1 && b != 1)) return NULL;
        expanded[size - 1 - i] = a > b ? a : b;
        reshaped[size - 1 - i] = a;
      }
      return poly_tensor_expand(ctx, poly_tensor_reshape(ctx, x, reshaped, size), expanded, size);
    }
    if (count != nr || nr * 2 > ONNX_RANK) return NULL;
    for (int i = 0; i < nr; i++) {
      if (target[i] < 1 || (uint64_t)target[i] > ONNX_BYTES / (uint64_t)sx[i]) return NULL;
      reshaped[2 * i] = 1;
      reshaped[2 * i + 1] = sx[i];
      expanded[2 * i] = target[i];
      expanded[2 * i + 1] = sx[i];
      target[i] *= sx[i];
    }
    return poly_tensor_reshape(
        ctx,
        poly_tensor_expand(ctx, poly_tensor_reshape(ctx, x, reshaped, nr * 2), expanded, nr * 2),
        target, nr
    );
  }
  if (!strcmp(op, "Gather") || !strcmp(op, "GatherElements")) {
    if (!arity(d, n, 2, 2) || !attrs(d, n, "|axis|") ||
        !(poly_dtype_eq(y->uop_physical->dtype, POLY_INT32) ||
          poly_dtype_eq(y->uop_physical->dtype, POLY_INT64)))
      return NULL;
    int nr = rank(d, x);
    int64_t axis = attr_int(d, n, "axis", 0);
    if (!onnx_axis(d, &axis, nr)) return NULL;
    if (!strcmp(op, "GatherElements")) {
      if (rank(d, y) != nr) return NULL;
      for (int i = 0; i < nr; i++)
        if (i != axis && shape(d, y)[i] > shape(d, x)[i]) return NULL;
      y = poly_tensor_alu2(
          ctx, POLY_OP_ADD, y,
          poly_tensor_alu3(
              ctx, POLY_OP_WHERE, poly_tensor_alu2(ctx, POLY_OP_CMPLT, y, onnx_int(d, 0)),
              onnx_int(d, shape(d, x)[axis]), onnx_int(d, 0)
          )
      );
      return poly_tensor_gather_dim(ctx, x, (int)axis, y);
    }
    int kinds[ONNX_RANK];
    PolyTensor *indices[ONNX_RANK] = {0};
    PolyUOp *starts[ONNX_RANK] = {0}, *sizes[ONNX_RANK] = {0};
    int64_t steps[ONNX_RANK];
    for (int i = 0; i < nr; i++) {
      kinds[i] = POLY_INDEX_SLICE;
      steps[i] = 1;
      starts[i] = poly_uop_const_int(ctx, 0);
      sizes[i] = poly_uop_const_int(ctx, shape(d, x)[i]);
    }
    kinds[axis] = POLY_INDEX_TENSOR;
    indices[axis] = y;
    return poly_tensor_getitem(ctx, x, kinds, starts, sizes, steps, indices, nr);
  }
  /* Shape/movement operations preserve any supported dtype. Numerical operators
   * reject invalid ONNX types before Tensor promotion can reinterpret the graph. */
  bool movement = !strcmp(op, "Identity") || !strcmp(op, "Transpose") || !strcmp(op, "Flatten") ||
                  !strcmp(op, "Reshape");
  bool integer_ok =
      !strcmp(op, "Neg") || !strcmp(op, "Add") || !strcmp(op, "Sub") || !strcmp(op, "Mul") ||
      !strcmp(op, "Div") || !strcmp(op, "MatMul") || !strcmp(op, "Abs") || !strcmp(op, "Sign") ||
      !strcmp(op, "Clip") || !strcmp(op, "Max") || !strcmp(op, "Min") || !strcmp(op, "Sum") ||
      !strcmp(op, "Pow") || !strcmp(op, "Mod") || !strncmp(op, "Reduce", 6) ||
      !strcmp(op, "ArgMax") || !strcmp(op, "ArgMin") || !strcmp(op, "Split") ||
      !strcmp(op, "CumSum") || !strcmp(op, "TopK") || (!strcmp(op, "Relu") && d->opset >= 14);
  if (x && !movement && !poly_dtype_is_float(x->uop_physical->dtype) &&
      !(integer_ok && poly_dtype_is_int(x->uop_physical->dtype) &&
        !poly_dtype_eq(x->uop_physical->dtype, POLY_BOOL))) {
    fail(d, POLY_IMPORT_ERR_DTYPE_UNSUPPORTED, "unsupported operator input dtype");
    return NULL;
  }
#define UNARY(name, fn)                                                                            \
  if (!strcmp(op, name)) {                                                                         \
    if (!arity(d, n, 1, 1) || !attrs(d, n, "")) return NULL;                                       \
    return fn(ctx, x);                                                                             \
  }
  if (!strcmp(op, "Identity"))
    return arity(d, n, 1, 1) && attrs(d, n, "") ? poly_tensor_retain(x) : NULL;
  UNARY("Abs", poly_tensor_abs)
  UNARY("Sign", poly_tensor_sign)
  UNARY("Floor", poly_tensor_floor)
  UNARY("Ceil", poly_tensor_ceil)
  UNARY("Round", poly_tensor_round)
  UNARY("Reciprocal", poly_tensor_reciprocal)
  UNARY("IsNaN", poly_tensor_isnan)
  UNARY("Cos", poly_tensor_cos)
  UNARY("Tan", poly_tensor_tan)
  UNARY("Asin", poly_tensor_asin)
  UNARY("Acos", poly_tensor_acos)
  UNARY("Atan", poly_tensor_atan)
  UNARY("Asinh", poly_tensor_asinh)
  UNARY("Acosh", poly_tensor_acosh)
  UNARY("Atanh", poly_tensor_atanh)
  UNARY("Sinh", poly_tensor_sinh)
  UNARY("Cosh", poly_tensor_cosh)
  UNARY("Erf", poly_tensor_erf)
  if (!strcmp(op, "Gelu")) {
    char approximate[ONNX_NAME];
    if (!arity(d, n, 1, 1) || !attrs(d, n, "|approximate|") ||
        !attr_text(d, n, "approximate", "none", approximate))
      return NULL;
    return !strcmp(approximate, "none")   ? poly_tensor_gelu_exact(ctx, x)
           : !strcmp(approximate, "tanh") ? poly_tensor_gelu(ctx, x)
                                          : NULL;
  }
  UNARY("Softsign", poly_tensor_softsign)
  UNARY("HardSwish", poly_tensor_hardswish)
  UNARY("Mish", poly_tensor_mish)
  UNARY("Relu", poly_tensor_relu)
  UNARY("Sigmoid", poly_tensor_sigmoid)
  UNARY("Tanh", poly_tensor_tanh)
  UNARY("Exp", poly_tensor_exp)
  UNARY("Log", poly_tensor_log)
#undef UNARY
  if (!strcmp(op, "Sqrt") || !strcmp(op, "Neg") || !strcmp(op, "Sin")) {
    if (!arity(d, n, 1, 1) || !attrs(d, n, "")) return NULL;
    return poly_tensor_alu1(
        ctx,
        !strcmp(op, "Sqrt")  ? POLY_OP_SQRT
        : !strcmp(op, "Sin") ? POLY_OP_SIN
                             : POLY_OP_NEG,
        x
    );
  }
  if (!strcmp(op, "IsInf")) {
    if (!arity(d, n, 1, 1) || !attrs(d, n, "|detect_negative||detect_positive|")) return NULL;
    bool pos = attr_int(d, n, "detect_positive", 1) != 0,
         neg = attr_int(d, n, "detect_negative", 1) != 0;
    PolyTensor *mask = poly_tensor_isinf(ctx, x);
    if (pos && neg) return mask;
    PolyTensor *zero = onnx_float(d, 0);
    PolyTensor *sign = poly_tensor_alu2(ctx, POLY_OP_CMPLT, pos ? zero : x, pos ? x : zero);
    if (!pos && !neg) sign = onnx_int(d, 0);
    return poly_tensor_alu2(ctx, POLY_OP_AND, mask, sign);
  }
  if (!strcmp(op, "TopK")) {
    if (!arity_outputs(d, n, 2, 2, 2) || !attrs(d, n, "|axis||largest||sorted|")) return NULL;
    int64_t k, axis = attr_int(d, n, "axis", -1);
    int count;
    if (!onnx_ints(d, n->inputs[1], &k, 1, &count) || count != 1 ||
        !onnx_axis(d, &axis, rank(d, x)) || k <= 0 || k > shape(d, x)[axis] || k > INT_MAX)
      return NULL;
    PolyTensor *values = NULL, *indices = NULL;
    if (poly_tensor_topk(
            ctx, x, (int)k, (int)axis, attr_int(d, n, "largest", 1) != 0,
            attr_int(d, n, "sorted", 1) != 0, &values, &indices
        ))
      return NULL;
    n->extra[1] = poly_tensor_cast(ctx, indices, POLY_INT64);
    return values;
  }
  if (!strcmp(op, "Einsum")) {
    char equation[ONNX_NAME];
    PolyTensor *inputs[ONNX_IO];
    if (!arity(d, n, 1, ONNX_IO) || !attrs(d, n, "|equation|") || !attr(n, "equation") ||
        !attr_text(d, n, "equation", "", equation))
      return NULL;
    for (int i = 0; i < n->nin; i++) {
      if (!n->inputs[i]) return NULL;
      inputs[i] = n->inputs[i]->tensor;
    }
    return poly_tensor_einsum(ctx, equation, inputs, n->nin);
  }
  if (!strcmp(op, "BatchNormalization")) {
    if (!arity_outputs(d, n, 5, 5, 3) || !attrs(d, n, "|epsilon||momentum||training_mode|") ||
        rank(d, x) < 2)
      return NULL;
    PolyTensor *b = n->inputs[2]->tensor, *mean = n->inputs[3]->tensor, *var = n->inputs[4]->tensor;
    for (int i = 1; i < 5; i++) {
      PolyTensor *t = n->inputs[i]->tensor;
      if (rank(d, t) != 1 || shape(d, t)[0] != shape(d, x)[1] || !same_dtype(x, t)) return NULL;
    }
    double eps = attr_float(d, n, "epsilon", 1e-5);
    if (eps < 0) return NULL;
    if (attr_int(d, n, "training_mode", 0)) {
      /* Pinned onnx.py uses NCHW axes for this training-mode operator. */
      if (rank(d, x) != 4 || n->nout != 3) return NULL;
      int64_t axes[3] = {0, 2, 3}, dims[4] = {1, shape(d, x)[1], 1, 1};
      PolyTensor *f = poly_tensor_cast(
          ctx, poly_tensor_detach(ctx, x),
          poly_dtype_eq(x->uop_physical->dtype, POLY_FLOAT64) ? POLY_FLOAT64 : POLY_FLOAT32
      );
      PolyTensor *m = poly_tensor_mean(ctx, f, axes, 3, false);
      PolyTensor *centered = onnx_sub(d, f, poly_tensor_reshape(ctx, m, dims, 4));
      PolyTensor *v = poly_tensor_mean(
          ctx, poly_tensor_alu2(ctx, POLY_OP_MUL, centered, centered), axes, 3, false
      );
      double momentum = attr_float(d, n, "momentum", .9);
      n->extra[1] = poly_tensor_cast(
          ctx,
          poly_tensor_alu2(ctx, POLY_OP_ADD, scale(d, mean, momentum), scale(d, m, 1 - momentum)),
          mean->uop_physical->dtype
      );
      n->extra[2] = poly_tensor_cast(
          ctx,
          poly_tensor_alu2(ctx, POLY_OP_ADD, scale(d, var, momentum), scale(d, v, 1 - momentum)),
          var->uop_physical->dtype
      );
      mean = m;
      var = v;
    } else if (n->nout != 1)
      return NULL;
    PolyTensor *inv = poly_tensor_reciprocal(
        ctx, poly_tensor_alu1(
                 ctx, POLY_OP_SQRT, poly_tensor_alu2(ctx, POLY_OP_ADD, var, onnx_float(d, eps))
             )
    );
    int64_t axis = 1;
    return poly_tensor_cast(
        ctx, poly_tensor_batchnorm(ctx, x, y, b, mean, inv, &axis, 1), x->uop_physical->dtype
    );
  }
  if (!strcmp(op, "LeakyRelu") || !strcmp(op, "ThresholdedRelu") || !strcmp(op, "PRelu")) {
    bool prelu = !strcmp(op, "PRelu"), threshold = !strcmp(op, "ThresholdedRelu");
    if (!arity(d, n, prelu ? 2 : 1, prelu ? 2 : 1) || !attrs(d, n, prelu ? "" : "|alpha|"))
      return NULL;
    if (prelu && !broadcast_into(d, x, y)) return NULL;
    double alpha = attr_float(d, n, "alpha", threshold ? 1.0 : .01);
    if (!prelu && !threshold) return poly_tensor_leaky_relu(ctx, x, alpha);
    PolyTensor *limit = onnx_float(d, threshold ? alpha : 0);
    PolyTensor *other = threshold ? onnx_float(d, 0)
                        : prelu   ? poly_tensor_alu2(ctx, POLY_OP_MUL, x, y)
                                  : scale(d, x, alpha);
    return poly_tensor_alu3(
        ctx, POLY_OP_WHERE, poly_tensor_alu2(ctx, POLY_OP_CMPLT, limit, x), x, other
    );
  }
  if (!strcmp(op, "Clip")) {
    if (!arity(d, n, 1, 3) || !attrs(d, n, "")) return NULL;
    PolyTensor *out = x;
    for (int i = 1; i < n->nin; i++)
      if (n->inputs[i]) {
        PolyTensor *bound = n->inputs[i]->tensor;
        if (rank(d, bound) != 0 || !same_dtype(x, bound)) return NULL;
        PolyTensor *condition = i == 1 ? poly_tensor_alu2(ctx, POLY_OP_CMPLT, out, bound)
                                       : poly_tensor_alu2(ctx, POLY_OP_CMPLT, bound, out);
        out = poly_tensor_alu3(ctx, POLY_OP_WHERE, condition, bound, out);
      }
    return poly_tensor_retain(out);
  }
  if (!strcmp(op, "Pow") || !strcmp(op, "Mod")) {
    bool pow = !strcmp(op, "Pow");
    if (!arity(d, n, 2, 2) || !attrs(d, n, pow ? "" : "|fmod|")) return NULL;
    if (!pow && !broadcast(d, x, y)) return NULL;
    if (pow) {
      PolyTensor *out = poly_tensor_alu2(ctx, POLY_OP_POW, x, y);
      return poly_dtype_is_int(x->uop_physical->dtype)
                 ? poly_tensor_cast(ctx, poly_tensor_round(ctx, out), x->uop_physical->dtype)
                 : out;
    }
    bool fmod = attr_int(d, n, "fmod", 0) != 0;
    return poly_dtype_is_int(x->uop_physical->dtype)
               ? poly_tensor_alu2(ctx, fmod ? POLY_OP_CMOD : POLY_OP_FLOORMOD, x, y)
               : onnx_sub(
                     d, x,
                     poly_tensor_alu2(ctx, POLY_OP_MUL, poly_tensor_div(ctx, x, y, fmod ? 1 : 2), y)
                 );
  }
  if (!strcmp(op, "Shrink")) {
    if (!arity(d, n, 1, 1) || !attrs(d, n, "|bias||lambd|")) return NULL;
    double bias = attr_float(d, n, "bias", 0), threshold = attr_float(d, n, "lambd", .5);
    PolyTensor *left = poly_tensor_alu2(ctx, POLY_OP_CMPLT, x, onnx_float(d, -threshold));
    PolyTensor *right = poly_tensor_alu2(ctx, POLY_OP_CMPLT, onnx_float(d, threshold), x);
    return poly_tensor_alu2(
        ctx, POLY_OP_ADD,
        poly_tensor_alu2(
            ctx, POLY_OP_MUL, left, poly_tensor_alu2(ctx, POLY_OP_ADD, x, onnx_float(d, bias))
        ),
        poly_tensor_alu2(ctx, POLY_OP_MUL, right, onnx_sub(d, x, onnx_float(d, bias)))
    );
  }
  if (!strcmp(op, "Hardmax")) {
    if (!arity(d, n, 1, 1) || !attrs(d, n, "|axis|")) return NULL;
    int nr = rank(d, x);
    int64_t axis = attr_int(d, n, "axis", -1), permutation[ONNX_RANK];
    if (!onnx_axis(d, &axis, nr)) return NULL;
    PolyTensor *index = poly_tensor_argmax(ctx, x, (int)axis, false);
    PolyTensor *hot = poly_tensor_one_hot(ctx, index, shape(d, x)[axis]);
    for (int i = 0; i < nr; i++)
      permutation[i] = i < axis ? i : i == axis ? nr - 1 : i - 1;
    return poly_tensor_cast(
        ctx, poly_tensor_permute(ctx, hot, permutation, nr), x->uop_physical->dtype
    );
  }
  if (!strcmp(op, "LpNormalization") || !strcmp(op, "MeanVarianceNormalization")) {
    bool lp = !strcmp(op, "LpNormalization");
    if (!arity(d, n, 1, 1) || !attrs(d, n, lp ? "|axis||p|" : "|axes|")) return NULL;
    int64_t axes[ONNX_RANK] = {0, 2, 3};
    int na = 3, nr = rank(d, x);
    if (lp) {
      axes[0] = attr_int(d, n, "axis", -1);
      na = 1;
    } else if (attr(n, "axes") && !attr_ints(d, n, "axes", axes, ONNX_RANK, &na))
      return NULL;
    unsigned mask = 0;
    for (int i = 0; i < na; i++) {
      if (!onnx_axis(d, &axes[i], nr) || (mask & (1u << axes[i]))) return NULL;
      mask |= 1u << axes[i];
    }
    if (!lp) x = onnx_sub(d, x, poly_tensor_mean(ctx, x, axes, na, true));
    int64_t p = attr_int(d, n, "p", 2);
    if (p != 1 && p != 2) return NULL;
    PolyTensor *v = p == 1 ? poly_tensor_abs(ctx, x) : poly_tensor_alu2(ctx, POLY_OP_MUL, x, x);
    PolyTensor *den =
        lp ? poly_tensor_sum(ctx, v, axes, na, true) : poly_tensor_mean(ctx, v, axes, na, true);
    if (p == 2) den = poly_tensor_alu1(ctx, POLY_OP_SQRT, den);
    if (!lp) den = poly_tensor_alu2(ctx, POLY_OP_ADD, den, onnx_float(d, 1e-9));
    return poly_tensor_div(ctx, x, den, 0);
  }
  if (!strcmp(op, "Dropout")) {
    if (!arity_outputs(d, n, 1, 3, 2) || !attrs(d, n, "|seed|")) return NULL;
    if (n->nin > 2 && n->inputs[2]) {
      Value *training = n->inputs[2];
      if (!poly_dtype_eq(training->dtype, POLY_BOOL) || training->nbytes != 1 ||
          !onnx_const_data(d, training))
        return NULL;
      if (training->host[0]) {
        fail(d, POLY_IMPORT_ERR_UNSUPPORTED_OP, "training Dropout requires the pinned legacy RNG");
        return NULL;
      }
    }
    int64_t dims[ONNX_RANK];
    int nr = rank(d, x);
    memcpy(dims, shape(d, x), nr * sizeof(int64_t));
    n->extra[1] =
        poly_tensor_expand(ctx, poly_tensor_cast(ctx, onnx_int(d, 1), POLY_BOOL), dims, nr);
    return poly_tensor_retain(x);
  }
  if (!strcmp(op, "NegativeLogLikelihoodLoss") || !strcmp(op, "SoftmaxCrossEntropyLoss")) {
    bool softmax = !strcmp(op, "SoftmaxCrossEntropyLoss");
    if (!arity_outputs(d, n, 2, 3, softmax ? 2 : 1) || !attrs(d, n, "|ignore_index||reduction|"))
      return NULL;
    char reduction[ONNX_NAME];
    if (!attr_text(d, n, "reduction", "mean", reduction)) return NULL;
    int r = !strcmp(reduction, "none")   ? 0
            : !strcmp(reduction, "sum")  ? 1
            : !strcmp(reduction, "mean") ? 2
                                         : -1;
    if (r < 0 || rank(d, x) < 2 || !poly_dtype_is_int(y->uop_physical->dtype)) return NULL;
    PolyTensor *weight = n->nin > 2 && n->inputs[2] ? n->inputs[2]->tensor : NULL;
    PolyTensor *ignore =
        attr(n, "ignore_index") ? onnx_int(d, attr_int(d, n, "ignore_index", 0)) : NULL;
    if (softmax) x = poly_tensor_log_softmax(ctx, x, 1);
    n->extra[1] = x;
    return poly_tensor_nll_loss(ctx, x, y, weight, ignore, r);
  }
  if (!strcmp(op, "HardSigmoid")) {
    if (!arity(d, n, 1, 1) || !attrs(d, n, "|alpha||beta|")) return NULL;
    PolyTensor *out = poly_tensor_alu2(
        ctx, POLY_OP_ADD, scale(d, x, attr_float(d, n, "alpha", .2)),
        onnx_float(d, attr_float(d, n, "beta", .5))
    );
    PolyTensor *zero = onnx_float(d, 0);
    PolyTensor *one = onnx_float(d, 1);
    out = poly_tensor_alu3(
        ctx, POLY_OP_WHERE, poly_tensor_alu2(ctx, POLY_OP_CMPLT, out, zero), zero, out
    );
    return poly_tensor_alu3(
        ctx, POLY_OP_WHERE, poly_tensor_alu2(ctx, POLY_OP_CMPLT, one, out), one, out
    );
  }
  if (!strcmp(op, "Celu") || !strcmp(op, "Selu") || !strcmp(op, "Elu")) {
    bool selu = !strcmp(op, "Selu");
    if (!arity(d, n, 1, 1) || !attrs(d, n, selu ? "|alpha||gamma|" : "|alpha|")) return NULL;
    PolyTensor *alpha = onnx_float(d, attr_float(d, n, "alpha", selu ? 1.6732632423543772 : 1));
    if (!strcmp(op, "Celu")) return poly_tensor_celu(ctx, x, alpha);
    if (selu)
      return poly_tensor_selu(
          ctx, x, alpha, onnx_float(d, attr_float(d, n, "gamma", 1.0507009873554805))
      );
    return poly_tensor_elu(ctx, x, attr_float(d, n, "alpha", 1));
  }
  if (!strcmp(op, "Softplus")) {
    if (!arity(d, n, 1, 1) || !attrs(d, n, "")) return NULL;
    return poly_tensor_softplus(ctx, x, 1);
  }
  if (!strcmp(op, "Max") || !strcmp(op, "Min") || !strcmp(op, "Sum") || !strcmp(op, "Mean")) {
    if (!arity(d, n, 1, ONNX_IO) || !attrs(d, n, "")) return NULL;
    PolyTensor *out = x;
    for (int i = 1; i < n->nin; i++) {
      if (!n->inputs[i] || !broadcast(d, out, n->inputs[i]->tensor)) return NULL;
      PolyTensor *v = n->inputs[i]->tensor;
      out = !strcmp(op, "Min")
                ? poly_tensor_minimum(ctx, out, v)
                : poly_tensor_alu2(ctx, !strcmp(op, "Max") ? POLY_OP_MAX : POLY_OP_ADD, out, v);
    }
    return !strcmp(op, "Mean") ? poly_tensor_div(ctx, out, onnx_int(d, n->nin), 0)
                               : poly_tensor_retain(out);
  }
  if (!strcmp(op, "Add") || !strcmp(op, "Sub") || !strcmp(op, "Mul") || !strcmp(op, "Div")) {
    if (!arity(d, n, 2, 2) || !attrs(d, n, "") || !broadcast(d, x, y)) return NULL;
    if (!strcmp(op, "Div"))
      return poly_tensor_div(ctx, x, y, poly_dtype_is_int(x->uop_physical->dtype) ? 1 : 0);
    if (!strcmp(op, "Sub")) y = poly_tensor_alu1(ctx, POLY_OP_NEG, y);
    return poly_tensor_alu2(ctx, !strcmp(op, "Mul") ? POLY_OP_MUL : POLY_OP_ADD, x, y);
  }
  if (!strcmp(op, "MatMul")) return arity(d, n, 2, 2) && attrs(d, n, "") ? matmul(d, x, y) : NULL;
  if (!strcmp(op, "Gemm")) {
    if (!arity(d, n, 2, 3) || !attrs(d, n, "|alpha||beta||transA||transB|") || rank(d, x) != 2 ||
        rank(d, y) != 2)
      return NULL;
    int64_t ta = attr_int(d, n, "transA", 0), tb = attr_int(d, n, "transB", 0);
    if ((ta != 0 && ta != 1) || (tb != 0 && tb != 1)) return NULL;
    if (ta) x = transpose(d, x);
    if (tb) y = transpose(d, y);
    PolyTensor *v = matmul(d, x, y);
    if (!v) return NULL;
    v = scale(d, v, attr_float(d, n, "alpha", 1));
    if (n->nin == 3 && n->inputs[2]) {
      y = n->inputs[2]->tensor;
      if (!broadcast_into(d, v, y)) return NULL;
      v = poly_tensor_alu2(ctx, POLY_OP_ADD, v, scale(d, y, attr_float(d, n, "beta", 1)));
    }
    return v;
  }
  if (!strcmp(op, "Transpose")) {
    if (!arity(d, n, 1, 1) || !attrs(d, n, "|perm|")) return NULL;
    int nr = rank(d, x), count;
    int64_t p[ONNX_RANK];
    if (!attr_ints(d, n, "perm", p, ONNX_RANK, &count)) return NULL;
    if (!attr(n, "perm")) {
      count = nr;
      for (int i = 0; i < nr; i++)
        p[i] = nr - 1 - i;
    }
    if (count != nr) return NULL;
    unsigned seen = 0;
    for (int i = 0; i < nr; i++) {
      if (p[i] < 0 || p[i] >= nr || (seen & (1u << p[i]))) return NULL;
      seen |= 1u << p[i];
    }
    return poly_tensor_permute(ctx, x, p, nr);
  }
  if (!strcmp(op, "Flatten")) {
    if (!arity(d, n, 1, 1) || !attrs(d, n, "|axis|")) return NULL;
    int nr = rank(d, x);
    int64_t axis = attr_int(d, n, "axis", 1), s[2] = {1, 1};
    const int64_t *sx = shape(d, x);
    if (axis < 0) axis += nr;
    if (axis < 0 || axis > nr) return NULL;
    for (int i = 0; i < nr; i++)
      s[i >= axis] *= sx[i];
    return poly_tensor_reshape(ctx, x, s, 2);
  }
  if (!strcmp(op, "Reshape")) {
    if (!arity(d, n, 2, 2) || !attrs(d, n, d->opset >= 14 ? "|allowzero|" : "")) return NULL;
    Value *sv = n->inputs[1];
    int64_t dims[ONNX_RANK], total = 1, old = 1;
    int missing = -1;
    if (sv->rank != 1 || sv->shape[0] > ONNX_RANK || !poly_dtype_eq(sv->dtype, POLY_INT64) ||
        !onnx_const_data(d, sv))
      return NULL;
    int nr = (int)sv->shape[0], nx = rank(d, x);
    const int64_t *sx = shape(d, x);
    int64_t allow = attr_int(d, n, "allowzero", 0);
    if (allow != 0 && allow != 1) return NULL;
    for (int i = 0; i < nx; i++)
      old *= sx[i];
    for (int i = 0; i < nr; i++) {
      memcpy(&dims[i], sv->host + i * 8, 8);
      if (dims[i] == 0 && !allow) {
        if (i >= nx) return NULL;
        dims[i] = sx[i];
      }
      if (dims[i] == -1) {
        if (missing >= 0) return NULL;
        missing = i;
      } else {
        if (dims[i] <= 0 || dims[i] > INT64_MAX / total) return NULL;
        total *= dims[i];
      }
    }
    if (missing >= 0) {
      if (old % total) return NULL;
      dims[missing] = old / total;
    } else if (total != old)
      return NULL;
    return poly_tensor_reshape(ctx, x, dims, nr);
  }
  if (!strncmp(op, "Reduce", 6)) {
    const char *kind = op + 6;
    bool input_axes = d->opset >= 18 || !strcmp(kind, "Sum");
    if (!arity(d, n, 1, input_axes ? 2 : 1) ||
        !attrs(d, n, input_axes ? "|keepdims||noop_with_empty_axes|" : "|axes||keepdims|"))
      return NULL;
    int64_t axes[ONNX_RANK];
    int count = 0, nr = rank(d, x);
    if (input_axes && n->nin > 1 && n->inputs[1]) {
      if (!onnx_ints(d, n->inputs[1], axes, ONNX_RANK, &count)) return NULL;
    } else if (!input_axes && !attr_ints(d, n, "axes", axes, ONNX_RANK, &count))
      return NULL;
    if (!count && attr_int(d, n, "noop_with_empty_axes", 0)) return poly_tensor_retain(x);
    if (!count)
      for (int i = 0; i < nr; i++)
        axes[count++] = i;
    unsigned mask = 0;
    for (int i = 0; i < count; i++) {
      if (!onnx_axis(d, &axes[i], nr) || (mask & (1u << axes[i]))) return NULL;
      mask |= 1u << axes[i];
    }
    bool keep = attr_int(d, n, "keepdims", 1) != 0;
    if (!strcmp(kind, "Max")) return poly_tensor_max(ctx, x, axes, count, keep);
    if (!strcmp(kind, "Min")) return poly_tensor_min(ctx, x, axes, count, keep);
    if (!strcmp(kind, "Mean")) return poly_tensor_mean(ctx, x, axes, count, keep);
    if (!strcmp(kind, "Prod")) return poly_tensor_prod(ctx, x, axes, count, keep);
    PolyDType original = x->uop_physical->dtype;
    if (!strcmp(kind, "L2") &&
        (poly_dtype_eq(original, POLY_FLOAT16) || poly_dtype_eq(original, POLY_BFLOAT16)))
      x = poly_tensor_cast(ctx, x, POLY_FLOAT32);
    if (!strcmp(kind, "SumSquare") || !strcmp(kind, "L2"))
      x = poly_tensor_alu2(ctx, POLY_OP_MUL, x, x);
    else if (!strcmp(kind, "L1"))
      x = poly_tensor_abs(ctx, x);
    else if (!strcmp(kind, "LogSumExp"))
      x = poly_tensor_exp(ctx, x);
    else if (strcmp(kind, "Sum") && strcmp(kind, "LogSum"))
      return NULL;
    PolyTensor *out = poly_tensor_sum(ctx, x, axes, count, keep);
    if (!strcmp(kind, "L2"))
      return poly_tensor_cast(ctx, poly_tensor_alu1(ctx, POLY_OP_SQRT, out), original);
    if (!strcmp(kind, "LogSum") || !strcmp(kind, "LogSumExp")) return poly_tensor_log(ctx, out);
    return out;
  }
  if (!strcmp(op, "ArgMax") || !strcmp(op, "ArgMin")) {
    if (!arity(d, n, 1, 1) || !attrs(d, n, "|axis||keepdims||select_last_index|")) return NULL;
    int64_t axis = attr_int(d, n, "axis", 0);
    if (!onnx_axis(d, &axis, rank(d, x))) return NULL;
    int64_t size = shape(d, x)[axis];
    bool last = attr_int(d, n, "select_last_index", 0) != 0;
    if (!strcmp(op, "ArgMin")) x = poly_tensor_alu1(ctx, POLY_OP_NEG, x);
    if (last) x = poly_tensor_flip(ctx, x, &axis, 1);
    PolyTensor *out = poly_tensor_argmax(ctx, x, (int)axis, attr_int(d, n, "keepdims", 1) != 0);
    if (last) out = onnx_sub(d, onnx_int(d, size - 1), out);
    return poly_tensor_cast(ctx, out, POLY_INT64);
  }
  if (!strcmp(op, "Split")) {
    if (!arity_outputs(d, n, 1, 2, ONNX_IO) ||
        !attrs(d, n, d->opset >= 18 ? "|axis||num_outputs|" : "|axis|"))
      return NULL;
    int64_t axis = attr_int(d, n, "axis", 0), lengths[ONNX_IO], pairs[ONNX_RANK][2];
    int nr = rank(d, x), count = 0;
    if (!onnx_axis(d, &axis, nr)) return NULL;
    int64_t size = shape(d, x)[axis], offset = 0;
    if (n->nin > 1 && n->inputs[1]) {
      if (!onnx_ints(d, n->inputs[1], lengths, ONNX_IO, &count) || count != n->nout ||
          attr(n, "num_outputs"))
        return NULL;
    } else {
      if (attr_int(d, n, "num_outputs", n->nout) != n->nout || size % n->nout) return NULL;
      for (count = 0; count < n->nout; count++)
        lengths[count] = size / n->nout;
    }
    for (int j = 0; j < nr; j++) {
      pairs[j][0] = 0;
      pairs[j][1] = shape(d, x)[j];
    }
    for (int i = 0; i < count; i++) {
      if (lengths[i] <= 0 || lengths[i] > size - offset) return NULL;
      pairs[axis][0] = offset;
      pairs[axis][1] = offset += lengths[i];
      n->extra[i] = poly_tensor_shrink(ctx, x, pairs, nr);
      if (!n->extra[i]) return NULL;
    }
    return offset == size ? n->extra[0] : NULL;
  }
  if (!strcmp(op, "CumSum")) {
    if (!arity(d, n, 2, 2) || !attrs(d, n, "|exclusive||reverse|")) return NULL;
    int64_t axis;
    int count, nr = rank(d, x);
    if (!onnx_ints(d, n->inputs[1], &axis, 1, &count) || count != 1 || !onnx_axis(d, &axis, nr))
      return NULL;
    bool reverse = attr_int(d, n, "reverse", 0) != 0;
    if (reverse) x = poly_tensor_flip(ctx, x, &axis, 1);
    if (attr_int(d, n, "exclusive", 0)) {
      int64_t pads[ONNX_RANK][2] = {{0}}, pairs[ONNX_RANK][2];
      for (int i = 0; i < nr; i++) {
        pairs[i][0] = 0;
        pairs[i][1] = shape(d, x)[i];
      }
      pads[axis][0] = 1;
      x = poly_tensor_shrink(ctx, poly_tensor_pad_value_int(ctx, x, pads, nr, 0), pairs, nr);
    }
    x = poly_tensor_cumsum(ctx, x, (int)axis);
    return reverse ? poly_tensor_flip(ctx, x, &axis, 1) : x;
  }
  if (!strcmp(op, "Softmax") || !strcmp(op, "LogSoftmax")) {
    if (!arity(d, n, 1, 1) || !attrs(d, n, "|axis|")) return NULL;
    int nr = rank(d, x);
    int64_t axis = attr_int(d, n, "axis", -1);
    if (axis < 0) axis += nr;
    return axis >= 0 && axis < nr && poly_dtype_is_float(x->uop_physical->dtype)
               ? (!strcmp(op, "LogSoftmax") ? poly_tensor_log_softmax(ctx, x, (int)axis)
                                            : poly_tensor_softmax(ctx, x, (int)axis))
               : NULL;
  }
  if (!strcmp(op, "GlobalAveragePool") || !strcmp(op, "GlobalMaxPool")) {
    if (!arity(d, n, 1, 1) || !attrs(d, n, "") || rank(d, x) < 3) return NULL;
    int64_t axes[ONNX_RANK];
    int nr = rank(d, x);
    for (int i = 2; i < nr; i++)
      axes[i - 2] = i;
    return !strcmp(op, "GlobalMaxPool") ? poly_tensor_max(ctx, x, axes, nr - 2, true)
                                        : poly_tensor_mean(ctx, x, axes, nr - 2, true);
  }
  if (!strcmp(op, "Conv")) {
    if (!arity(d, n, 2, 3) ||
        !attrs(d, n, "|auto_pad||dilations||group||kernel_shape||pads||strides|") ||
        rank(d, x) < 3 || rank(d, x) != rank(d, y))
      return NULL;
    int dims = rank(d, x) - 2;
    int64_t stride[ONNX_RANK], dilation[ONNX_RANK], padding[2 * ONNX_RANK], kernel[ONNX_RANK];
    const int64_t *sx = shape(d, x), *sw = shape(d, y);
    if (!onnx_spatial(d, n, x, sw + 2, kernel, stride, dilation, padding, false)) return NULL;
    int64_t groups = attr_int(d, n, "group", 1);
    if (groups < 1 || groups > INT_MAX || sx[1] % groups || sw[0] % groups ||
        sw[1] != sx[1] / groups || !same_dtype(x, y))
      return NULL;
    for (int i = 0; i < dims; i++)
      if (sx[i + 2] + padding[2 * (dims - i - 1)] + padding[2 * (dims - i - 1) + 1] <
          dilation[i] * (sw[i + 2] - 1) + 1)
        return NULL;
    PolyTensor *bias = n->nin == 3 && n->inputs[2] ? n->inputs[2]->tensor : NULL;
    if (bias && (rank(d, bias) != 1 || shape(d, bias)[0] != sw[0] || !same_dtype(x, bias)))
      return NULL;
    return poly_tensor_conv2d(ctx, x, y, bias, (int)groups, stride, dilation, padding, 2 * dims);
  }
  if (!strcmp(op, "MaxPool") || !strcmp(op, "AveragePool")) {
    bool maximum = !strcmp(op, "MaxPool");
    if (!arity_outputs(d, n, 1, 1, maximum ? 2 : 1) ||
        !attrs(
            d, n,
            maximum
                ? "|auto_pad||ceil_mode||dilations||kernel_shape||pads||storage_order||strides|"
                : "|auto_pad||ceil_mode||count_include_pad||dilations||kernel_shape||pads||strides|"
        ))
      return NULL;
    int64_t kernel[ONNX_RANK], stride[ONNX_RANK], dilation[ONNX_RANK], padding[2 * ONNX_RANK];
    if (!onnx_spatial(d, n, x, NULL, kernel, stride, dilation, padding, false)) return NULL;
    int dims = rank(d, x) - 2;
    /* The pin returns channel-local indices, ONNX specifies flattened N/C offsets.
     * Reject that mismatch until the upstream correction has a reviewed port. */
    if (maximum && n->nout > 1 && shape(d, x)[0] * shape(d, x)[1] != 1) {
      fail(
          d, POLY_IMPORT_ERR_UNSUPPORTED_OP,
          "MaxPool indices require one batch/channel (pinned index convention)"
      );
      return NULL;
    }
    bool ceil_mode = attr_int(d, n, "ceil_mode", 0) != 0;
    if (!maximum)
      return poly_tensor_avg_pool2d(
          ctx, x, kernel, dims, stride, dilation, padding, 2 * dims, ceil_mode,
          attr_int(d, n, "count_include_pad", 0) != 0
      );
    PolyTensor *indices = NULL;
    PolyTensor *out = poly_tensor_max_pool2d(
        ctx, x, kernel, dims, stride, dilation, padding, 2 * dims, ceil_mode,
        n->nout > 1 ? &indices : NULL
    );
    if (indices && attr_int(d, n, "storage_order", 0)) indices = transpose(d, indices);
    n->extra[1] = indices ? poly_tensor_cast(ctx, indices, POLY_INT64) : NULL;
    return out;
  }
  if (!strcmp(op, "ConvTranspose")) {
    if (!arity(d, n, 2, 3) ||
        !attrs(
            d, n,
            "|auto_pad||dilations||group||kernel_shape||pads||strides||output_padding||output_"
            "shape|"
        ) ||
        rank(d, x) < 3 || rank(d, x) != rank(d, y) || !same_dtype(x, y))
      return NULL;
    int dims = rank(d, x) - 2, count;
    int64_t kernel[ONNX_RANK], stride[ONNX_RANK], dilation[ONNX_RANK], padding[2 * ONNX_RANK];
    int64_t output_pad[ONNX_RANK] = {0}, output_shape[ONNX_RANK];
    /* Resolve transposed-convolution padding separately from forward SAME padding. */
    char mode[ONNX_NAME];
    if (!attr_text(d, n, "auto_pad", "NOTSET", mode) ||
        (strcmp(mode, "NOTSET") && strcmp(mode, "VALID") && strcmp(mode, "SAME_UPPER") &&
         strcmp(mode, "SAME_LOWER")))
      return NULL;
    bool ok = onnx_spatial(d, n, x, shape(d, y) + 2, kernel, stride, dilation, padding, true);
    if (!ok || !attr_ints(d, n, "output_padding", output_pad, dims, &count) ||
        (attr(n, "output_padding") && count != dims) ||
        !attr_ints(d, n, "output_shape", output_shape, dims, &count) ||
        (attr(n, "output_shape") && count != dims))
      return NULL;
    int64_t groups = attr_int(d, n, "group", 1);
    if (groups < 1 || groups > INT_MAX || shape(d, x)[1] != shape(d, y)[0] ||
        shape(d, x)[1] % groups)
      return NULL;
    for (int i = 0; i < dims; i++) {
      if (output_pad[i] < 0 || (output_pad[i] >= stride[i] && output_pad[i] >= dilation[i]))
        return NULL;
      if (attr(n, "output_shape") || (!attr(n, "pads") && strcmp(mode, "NOTSET"))) {
        int64_t target = attr(n, "output_shape") ? output_shape[i] : shape(d, x)[i + 2] * stride[i];
        int64_t total = stride[i] * (shape(d, x)[i + 2] - 1) + output_pad[i] +
                        (kernel[i] - 1) * dilation[i] + 1 - target;
        if (target < 1 || total < 0) return NULL;
        int64_t first = !strcmp(mode, "SAME_UPPER") ? total / 2 : total - total / 2;
        padding[2 * (dims - i - 1)] = first;
        padding[2 * (dims - i - 1) + 1] = total - first;
      }
    }
    PolyTensor *bias = n->nin > 2 && n->inputs[2] ? n->inputs[2]->tensor : NULL;
    if (bias && (rank(d, bias) != 1 || shape(d, bias)[0] != shape(d, y)[1] * groups ||
                 !same_dtype(x, bias)))
      return NULL;
    return poly_tensor_conv_transpose2d(
        ctx, x, y, bias, (int)groups, stride, dilation, padding, 2 * dims, output_pad, dims
    );
  }
  if (!strcmp(op, "InstanceNormalization") || !strcmp(op, "GroupNormalization")) {
    if (!arity(d, n, 3, 3) ||
        !attrs(
            d, n,
            !strcmp(op, "InstanceNormalization") ? "|epsilon|" : "|epsilon||num_groups||stash_type|"
        ) ||
        rank(d, x) < 2 || attr_int(d, n, "stash_type", 1) != 1)
      return NULL;
    int nr = rank(d, x);
    const int64_t *sx = shape(d, x);
    int64_t groups = !strcmp(op, "InstanceNormalization") ? sx[1] : attr_int(d, n, "num_groups", 0);
    PolyTensor *bias = n->inputs[2]->tensor;
    if (groups < 1 || sx[1] % groups || rank(d, y) != 1 || shape(d, y)[0] != sx[1] ||
        rank(d, bias) != 1 || shape(d, bias)[0] != sx[1])
      return NULL;
    int64_t count = 1, rs[3] = {sx[0], groups, 0}, affine[ONNX_RANK], original[ONNX_RANK], axis = 2;
    memcpy(original, sx, nr * sizeof(int64_t));
    for (int i = 1; i < nr; i++)
      count *= sx[i];
    rs[2] = count / groups;
    for (int i = 0; i < nr; i++)
      affine[i] = i == 1 ? sx[1] : 1;
    PolyTensor *v = poly_tensor_cast(ctx, poly_tensor_reshape(ctx, x, rs, 3), POLY_FLOAT32);
    PolyTensor *centered = onnx_sub(d, v, poly_tensor_mean(ctx, v, &axis, 1, true));
    PolyTensor *variance = poly_tensor_mean(
        ctx, poly_tensor_alu2(ctx, POLY_OP_MUL, centered, centered), &axis, 1, true
    );
    double eps = attr_float(d, n, "epsilon", 1e-5);
    if (eps < 0 || !isfinite(eps)) return NULL;
    PolyTensor *inv = poly_tensor_reciprocal(
        ctx, poly_tensor_alu1(
                 ctx, POLY_OP_SQRT, poly_tensor_alu2(ctx, POLY_OP_ADD, variance, onnx_float(d, eps))
             )
    );
    v = poly_tensor_reshape(
        ctx,
        poly_tensor_cast(
            ctx, poly_tensor_alu2(ctx, POLY_OP_MUL, centered, inv), x->uop_physical->dtype
        ),
        original, nr
    );
    return poly_tensor_alu2(
        ctx, POLY_OP_ADD,
        poly_tensor_alu2(ctx, POLY_OP_MUL, v, poly_tensor_reshape(ctx, y, affine, nr)),
        poly_tensor_reshape(ctx, bias, affine, nr)
    );
  }
  if (!strcmp(op, "DepthToSpace") || !strcmp(op, "SpaceToDepth")) {
    if (!arity(d, n, 1, 1) ||
        !attrs(d, n, !strcmp(op, "DepthToSpace") ? "|blocksize||mode|" : "|blocksize|") ||
        rank(d, x) != 4)
      return NULL;
    int64_t block = attr_int(d, n, "blocksize", 0), sizes[2] = {block, block};
    char mode[ONNX_NAME];
    if (block <= 0 || block > INT_MAX || !attr_text(d, n, "mode", "DCR", mode) ||
        (strcmp(mode, "DCR") && strcmp(mode, "CRD")))
      return NULL;
    const char *formula = !strcmp(op, "SpaceToDepth") ? "b c (h h1) (w w1) -> b (h1 w1 c) h w"
                          : !strcmp(mode, "CRD")      ? "b (c h1 w1) h w -> b c (h h1) (w w1)"
                                                      : "b (h1 w1 c) h w -> b c (h h1) (w w1)";
    return poly_tensor_rearrange(ctx, formula, x, "h1 w1", sizes, 2);
  }
  if (!strcmp(op, "LayerNormalization")) {
    if (d->opset < 17 || !arity_outputs(d, n, 2, 3, 3) ||
        !attrs(d, n, "|axis||epsilon||stash_type|") || attr_int(d, n, "stash_type", 1) != 1)
      return NULL;
    int nr = rank(d, x);
    int64_t axis = attr_int(d, n, "axis", -1), axes[ONNX_RANK];
    if (axis < 0) axis += nr;
    double eps = attr_float(d, n, "epsilon", 1e-5);
    if (axis < 0 || axis >= nr || eps < 0 || !isfinite(eps) || !broadcast_into(d, x, y))
      return NULL;
    PolyTensor *bias = n->nin == 3 && n->inputs[2] ? n->inputs[2]->tensor : NULL;
    if (bias && !broadcast_into(d, x, bias)) return NULL;
    for (int i = (int)axis; i < nr; i++)
      axes[i - axis] = i;
    /* onnx.py:LayerNormalization keeps stash computation in float32 and casts
     * the normalized value back before scale/bias, not after the affine step. */
    PolyDType original = x->uop_physical->dtype;
    PolyTensor *x32 = poly_tensor_cast(ctx, x, POLY_FLOAT32);
    PolyTensor *mean = poly_tensor_mean(ctx, x32, axes, nr - (int)axis, true);
    PolyTensor *centered = onnx_sub(d, x32, mean);
    PolyTensor *variance = poly_tensor_mean(
        ctx, poly_tensor_alu2(ctx, POLY_OP_MUL, centered, centered), axes, nr - (int)axis, true
    );
    PolyTensor *inv = poly_tensor_reciprocal(
        ctx, poly_tensor_alu1(
                 ctx, POLY_OP_SQRT, poly_tensor_alu2(ctx, POLY_OP_ADD, variance, onnx_float(d, eps))
             )
    );
    n->extra[1] = mean;
    n->extra[2] = inv;
    PolyTensor *out = poly_tensor_alu2(
        ctx, POLY_OP_MUL,
        poly_tensor_cast(ctx, poly_tensor_alu2(ctx, POLY_OP_MUL, centered, inv), original), y
    );
    return bias ? poly_tensor_alu2(ctx, POLY_OP_ADD, out, bias) : out;
  }
  fail(d, POLY_IMPORT_ERR_UNSUPPORTED_OP, "operator is not supported");
  return NULL;
}

static bool onnx_literal(
    Import *d,
    const char *name,
    PolyDType dtype,
    const int64_t *dims,
    int ndim,
    const void *data,
    size_t len
) {
  Value *v = add_value(d, name);
  if (!v) return false;
  v->dtype = dtype;
  v->rank = ndim;
  if (ndim) memcpy(v->shape, dims, ndim * sizeof(int64_t));
  if (!shape_bytes(d, v) || len != v->nbytes || len > ONNX_BYTES - d->initializer_bytes)
    return false;
  d->initializer_bytes += len;
  v->host = malloc(len ? len : 1);
  if (!v->host) return false;
  memcpy(v->host, data, len);
  v->constant = v->initializer = true;
  v->tensor = poly_model_aux_from_host(d->model, v->binding, dtype, v->shape, ndim, v->host, len);
  return v->tensor != NULL;
}

static bool onnx_node(Import *d, Bytes proto) {
  Node n = {0};
  char domain[ONNX_NAME];
  if (!name_field(d, proto, 4, d->op, true) || !name_field(d, proto, 7, domain, false))
    return false;
  const char *microsoft = "|BiasGelu||FastGelu||SkipLayerNormalization||QLinearAdd||QLinearMul||"
                          "QLinearGlobalAveragePool|";
  const char *ml = "|Binarizer||ArrayFeatureExtractor|";
  char key[ONNX_NAME + 2];
  snprintf(key, sizeof(key), "|%s|", d->op);
  bool standard = !domain[0] || !strcmp(domain, "ai.onnx");
  if ((standard && (strstr(microsoft, key) || strstr(ml, key))) ||
      (!standard &&
       !((!strcmp(domain, "com.microsoft") && d->microsoft_opset && strstr(microsoft, key)) ||
         (!strcmp(domain, "ai.onnx.ml") && d->ml_opset && strstr(ml, key)))))
    return fail(
        d, POLY_IMPORT_ERR_UNSUPPORTED_OP, "unsupported operator/domain pair '%s:%s'", domain, d->op
    );
  Reader r = {.bytes = proto};
  Field f;
  while (next(&r, &f)) {
    if (f.tag == 1 || f.tag == 2) {
      char name[ONNX_NAME];
      if (f.wire != 2 || !text(d, f.bytes, name, true)) return false;
      if (f.tag == 1) {
        if (n.nin == ONNX_IO) return fail(d, POLY_IMPORT_ERR_PARSE, "too many node inputs");
        Value *v = name[0] ? lookup(d, name) : NULL;
        if (name[0] && !v)
          return fail(d, POLY_IMPORT_ERR_PARSE, "unknown/not-topological input '%s'", name);
        n.inputs[n.nin++] = v;
      } else {
        if (n.nout == ONNX_IO) return false;
        strcpy(n.outputs[n.nout++], name);
      }
    } else if (f.tag == 5) {
      if (f.wire != 2 || n.nattrs == 32)
        return fail(d, POLY_IMPORT_ERR_PARSE, "too many/invalid attributes");
      Attr *a = &n.attrs[n.nattrs];
      Field type;
      if (!name_field(d, f.bytes, 1, a->name, true) || attr(&n, a->name) ||
          !field(d, f.bytes, 20, 0, &type, true))
        return fail(d, POLY_IMPORT_ERR_PARSE, "invalid/duplicate attribute");
      a->proto = f.bytes;
      a->type = (int)type.integer;
      n.nattrs++;
    }
  }
  if (r.bad) return fail(d, POLY_IMPORT_ERR_PARSE, "truncated node");
  if (!strcmp(d->op, "Shape") || !strcmp(d->op, "Size")) {
    bool size = !strcmp(d->op, "Size");
    if (!arity(d, &n, 1, 1) || !attrs(d, &n, size || d->opset < 15 ? "" : "|start||end|"))
      return false;
    PolyTensor *x = n.inputs[0]->tensor;
    int nr = rank(d, x);
    int64_t first = attr_int(d, &n, "start", 0), last = attr_int(d, &n, "end", nr);
    if (first < 0) first += nr;
    if (last < 0) last += nr;
    first = first < 0 ? 0 : first > nr ? nr : first;
    last = last < 0 ? 0 : last > nr ? nr : last;
    int64_t count = last - first, product = 1;
    for (int i = 0; i < nr; i++)
      product *= shape(d, x)[i];
    if (!size && count <= 0)
      return fail(d, POLY_IMPORT_ERR_UNSUPPORTED_OP, "empty Shape output is unsupported");
    return onnx_literal(
        d, n.outputs[0], POLY_INT64, &count, size ? 0 : 1, size ? &product : shape(d, x) + first,
        size ? 8 : (size_t)count * 8
    );
  }
  if (!strcmp(d->op, "Constant")) {
    Field t;
    Attr *a = attr(&n, "value");
    if (!arity(d, &n, 0, 0) || n.nattrs != 1) return false;
    if (a)
      return a->type == 4 && field(d, a->proto, 5, 2, &t, true) &&
             onnx_initializer(d, t.bytes, n.outputs[0]);
    if ((a = attr(&n, "value_int"))) {
      int64_t v = attr_int(d, &n, "value_int", 0);
      return poly_import_last_error_code() == POLY_IMPORT_OK &&
             onnx_literal(d, n.outputs[0], POLY_INT64, NULL, 0, &v, 8);
    }
    if ((a = attr(&n, "value_float"))) {
      float v = (float)attr_float(d, &n, "value_float", 0);
      return poly_import_last_error_code() == POLY_IMPORT_OK &&
             onnx_literal(d, n.outputs[0], POLY_FLOAT32, NULL, 0, &v, 4);
    }
    if ((a = attr(&n, "value_ints"))) {
      int64_t values[512];
      int count;
      if (!attr_ints(d, &n, "value_ints", values, 512, &count)) return false;
      int64_t dim = count;
      return onnx_literal(d, n.outputs[0], POLY_INT64, &dim, 1, values, count * 8);
    }
    if ((a = attr(&n, "value_floats"))) {
      float values[512];
      int count = 0;
      Reader reader = {.bytes = a->proto};
      Field value;
      if (a->type != 6) return false;
      while (next(&reader, &value))
        if (value.tag == 7) {
          if ((value.wire != 2 && value.wire != 5) || value.bytes.size % 4 ||
              value.bytes.size / 4 > (size_t)(512 - count))
            return false;
          memcpy(values + count, value.bytes.data, value.bytes.size);
          count += (int)(value.bytes.size / 4);
        }
      int64_t dim = count;
      return !reader.bad && onnx_literal(d, n.outputs[0], POLY_FLOAT32, &dim, 1, values, count * 4);
    }
    return fail(d, POLY_IMPORT_ERR_UNSUPPORTED_OP, "unsupported Constant attribute");
  }
  PolyTensor *out = onnx_operation(d, &n);
  if (!out || poly_import_last_error_code() != POLY_IMPORT_OK)
    return fail(
        d, POLY_IMPORT_ERR_UNSUPPORTED_OP, "unsupported or invalid shape/attribute combination"
    );
  n.extra[0] = out;
  bool constant = !n.captures_runtime;
  for (int i = 0; i < n.nin; i++)
    if (n.inputs[i] && !n.inputs[i]->constant) constant = false;
  for (int i = 0; i < n.nout; i++) {
    if (!n.outputs[i][0]) continue;
    out = n.extra[i];
    Value *v = out ? add_value(d, n.outputs[i]) : NULL;
    if (!v) return false;
    v->tensor = out;
    v->constant = constant;
    v->dtype = out->uop_physical->dtype;
    v->rank = rank(d, out);
    if (v->rank < 0 || v->rank > ONNX_RANK) return false;
    if (v->rank) memcpy(v->shape, shape(d, out), v->rank * sizeof(int64_t));
    if (!shape_bytes(d, v)) return false;
  }
  return true;
}

static bool onnx_subgraph(Import *d, Bytes proto, Value **outputs, int *nout) {
  if (d->depth == 32) return fail(d, POLY_IMPORT_ERR_PARSE, "subgraph nesting exceeds 32");
  char parent_op[ONNX_NAME];
  strcpy(parent_op, d->op);
  d->scopes[++d->depth] = ++d->next_scope;
  bool ok = false;
  Reader r = {.bytes = proto};
  Field f;
  while (next(&r, &f)) {
    if (f.tag == 5 && (f.wire != 2 || !onnx_initializer(d, f.bytes, NULL))) goto done;
    if (f.tag == 11 || f.tag == 15) {
      fail(
          d, POLY_IMPORT_ERR_UNSUPPORTED_OP,
          "If branches require lexical captures and dense initializers"
      );
      goto done;
    }
  }
  if (r.bad) goto done;
  r = (Reader){.bytes = proto};
  while (next(&r, &f))
    if (f.tag == 1 && (f.wire != 2 || ++d->node_index > 2048 || !onnx_node(d, f.bytes))) goto done;
  if (r.bad) goto done;
  r = (Reader){.bytes = proto};
  while (next(&r, &f))
    if (f.tag == 12) {
      Value info = {0};
      if (f.wire != 2 || !value_info(d, f.bytes, &info) || *nout == ONNX_IO) goto done;
      Value *v = lookup(d, info.name);
      if (!v || !poly_dtype_eq(v->dtype, info.dtype) || v->rank != info.rank ||
          (info.rank && memcmp(v->shape, info.shape, info.rank * sizeof(int64_t)))) {
        fail(
            d, POLY_IMPORT_ERR_SHAPE_MISMATCH, "branch output '%s' disagrees with declaration",
            info.name
        );
        goto done;
      }
      outputs[(*nout)++] = v;
    }
  ok = !r.bad && *nout > 0;
done:
  d->depth--;
  strcpy(d->op, parent_op);
  return ok;
}

static bool onnx_graph(Import *d, Bytes proto) {
  Reader r = {.bytes = proto};
  Field f;
  const char *inputs[ONNX_IO], *outputs[ONNX_IO];
  int nin = 0, nout = 0;
  /* Initializers precede graph construction regardless of protobuf field order. */
  while (next(&r, &f))
    if (f.tag == 5) {
      if (f.wire != 2 || !onnx_initializer(d, f.bytes, NULL)) return false;
    } else if (f.tag == 15) {
      return fail(d, POLY_IMPORT_ERR_UNSUPPORTED_OP, "sparse initializers are unsupported");
    }
  if (r.bad) return false;
  r = (Reader){.bytes = proto};
  while (next(&r, &f))
    if (f.tag == 11) {
      Value info = {0};
      if (f.wire != 2 || !value_info(d, f.bytes, &info)) return false;
      Value *v = lookup(d, info.name);
      if (v) {
        if (!v->initializer || !poly_dtype_eq(v->dtype, info.dtype) || v->rank != info.rank ||
            memcmp(v->shape, info.shape, info.rank * sizeof(int64_t)))
          return fail(d, POLY_IMPORT_ERR_SHAPE_MISMATCH, "initializer/input declaration mismatch");
        continue;
      }
      if (nin == ONNX_IO) return false;
      v = add_value(d, info.name);
      if (!v) return false;
      *v = info;
      v->tensor = poly_model_input(d->model, v->name, v->dtype, v->shape, v->rank);
      if (!v->tensor) return false;
      inputs[nin++] = v->name;
    }
  if (r.bad) return false;
  r = (Reader){.bytes = proto};
  while (next(&r, &f))
    if (f.tag == 1) {
      if (f.wire != 2 || ++d->node_index > 2048 || !onnx_node(d, f.bytes)) return false;
    }
  if (r.bad) return false;
  r = (Reader){.bytes = proto};
  while (next(&r, &f))
    if (f.tag == 12) {
      Value info = {0};
      if (f.wire != 2 || !value_info(d, f.bytes, &info)) return false;
      Value *v = lookup(d, info.name);
      if (!v || nout == ONNX_IO)
        return fail(d, POLY_IMPORT_ERR_PARSE, "unknown output '%s'", info.name);
      if (!poly_dtype_eq(v->tensor->uop_physical->dtype, info.dtype) ||
          rank(d, v->tensor) != info.rank ||
          (info.rank && memcmp(shape(d, v->tensor), info.shape, info.rank * sizeof(int64_t))))
        return fail(
            d, POLY_IMPORT_ERR_SHAPE_MISMATCH, "output '%s' disagrees with declared shape/dtype",
            info.name
        );
      for (int i = 0; i < nout; i++)
        if (!strcmp(outputs[i], v->name)) return fail(d, POLY_IMPORT_ERR_PARSE, "duplicate output");
      if (poly_model_output(d->model, v->name, v->tensor) != POLY_STATUS_OK) return false;
      outputs[nout++] = v->name;
    }
  if (r.bad || !nout) return false;
  PolyModelError err = {0};
  if (poly_model_entrypoint(d->model, "forward", inputs, nin, outputs, nout, NULL) ||
      poly_model_build(d->model, &err))
    return fail(d, POLY_IMPORT_ERR_INTERNAL, "Model build: %s", err.message);
  for (int i = 0; i < d->nvalues; i++)
    if (d->values[i].initializer) {
      Value *v = &d->values[i];
      if (poly_model_write_buf_named(d->model, v->binding, v->host, v->nbytes))
        return fail(d, POLY_IMPORT_ERR_INTERNAL, "initializer upload failed");
      /* Inference imports do not guess which float initializers should train. */
      int index = -1;
      for (int j = 0; j < poly_model_buf_count(d->model); j++)
        if (!strcmp(poly_model_buf_name(d->model, j), v->binding)) {
          index = j;
          break;
        }
      if (index < 0 || poly_model_set_buf_trainable(d->model, index, false)) return false;
    }
  return true;
}

PolyModel *poly_onnx_load_into(
    PolyCtx *ctx,
    const uint8_t *data,
    int64_t len,
    const PolyOnnxOptions *options,
    PolyDevice device
) {
  poly_import_error_clear();
  Import *d = calloc(1, sizeof(*d));
  if (!d) {
    poly_import_error_set(POLY_IMPORT_ERR_INTERNAL, "ONNX allocation failed");
    return NULL;
  }
  d->options = options;
  bool ok = false, scoped = false;
  PolyModelFactoryScope scope = {0};
  if (!data || len <= 0 || (uint64_t)len > ONNX_BYTES) {
    fail(d, POLY_IMPORT_ERR_PARSE, "expected 1..1GiB model bytes");
    goto done;
  }
  if (options) {
    if (options->n_external < 0 || options->n_external > ONNX_VALUES ||
        (options->n_external &&
         (!options->external_names || !options->external_data || !options->external_lengths))) {
      fail(d, POLY_IMPORT_ERR_PARSE, "invalid external arrays");
      goto done;
    }
    for (int i = 0; i < options->n_external; i++) {
      if (!options->external_names[i] || !options->external_names[i][0] ||
          !options->external_data[i] || options->external_lengths[i] < 0 ||
          (uint64_t)options->external_lengths[i] > ONNX_BYTES) {
        fail(d, POLY_IMPORT_ERR_PARSE, "invalid external entry");
        goto done;
      }
      for (int j = 0; j < i; j++)
        if (!strcmp(options->external_names[i], options->external_names[j])) {
          fail(d, POLY_IMPORT_ERR_PARSE, "duplicate external name");
          goto done;
        }
    }
    if (options->dimensions_json) {
      PolyModelError err = {0};
      size_t n = strlen(options->dimensions_json);
      if (n > INT_MAX ||
          !(d->dimensions = model_factory_parse(options->dimensions_json, (int)n, &err))) {
        fail(d, POLY_IMPORT_ERR_PARSE, "invalid dimensions JSON: %s", err.message);
        goto done;
      }
      for (cJSON *v = d->dimensions->child; v; v = v->next)
        if (!cJSON_IsNumber(v) || !isfinite(v->valuedouble) || v->valuedouble < 1 ||
            v->valuedouble > ONNX_BYTES || trunc(v->valuedouble) != v->valuedouble) {
          fail(
              d, POLY_IMPORT_ERR_SHAPE_MISMATCH, "dimension '%s' must be a positive integer",
              v->string
          );
          goto done;
        }
    }
  }
  Bytes model = {data, (size_t)len};
  Field g, version;
  if (!field(d, model, 1, 0, &version, true) || version.integer < 3 || version.integer > 10 ||
      !field(d, model, 7, 2, &g, true)) {
    fail(d, POLY_IMPORT_ERR_PARSE, "supported ONNX IR versions are 3..10");
    goto done;
  }
  Reader r = {.bytes = model};
  Field f;
  while (next(&r, &f)) {
    if (f.tag == 20 || f.tag == 25) {
      fail(
          d, POLY_IMPORT_ERR_UNSUPPORTED_OP, "training graphs and local functions are unsupported"
      );
      goto done;
    }
    if (f.tag == 8) {
      char domain[ONNX_NAME];
      Field opset;
      if (f.wire != 2 || !name_field(d, f.bytes, 1, domain, false) ||
          !field(d, f.bytes, 2, 0, &opset, true))
        goto done;
      int *extension = !strcmp(domain, "com.microsoft") ? &d->microsoft_opset
                       : !strcmp(domain, "ai.onnx.ml")  ? &d->ml_opset
                                                        : NULL;
      if (extension) {
        if (*extension || opset.integer < 1 ||
            opset.integer > (extension == &d->ml_opset ? 3u : 1u)) {
          fail(d, POLY_IMPORT_ERR_UNSUPPORTED_OP, "unsupported/duplicate opset for '%s'", domain);
          goto done;
        }
        *extension = (int)opset.integer;
        continue;
      }
      if (domain[0] && strcmp(domain, "ai.onnx")) {
        fail(d, POLY_IMPORT_ERR_UNSUPPORTED_OP, "unsupported domain '%s'", domain);
        goto done;
      }
      if (d->opset || opset.integer < 13 || opset.integer > 21) {
        fail(d, POLY_IMPORT_ERR_UNSUPPORTED_OP, "one standard opset in 13..21 is required");
        goto done;
      }
      d->opset = (int)opset.integer;
    }
  }
  if (r.bad || !d->opset) {
    fail(d, POLY_IMPORT_ERR_PARSE, "missing/malformed opset");
    goto done;
  }
  if (!model_factory_begin(&scope, ctx, device)) {
    fail(d, POLY_IMPORT_ERR_INTERNAL, "requires idle context and executable device");
    goto done;
  }
  scoped = true;
  d->ctx = scope.ctx;
  d->model = poly_model_new(d->ctx, NULL);
  ok = d->model && onnx_graph(d, g.bytes);
done:
  if (!ok) {
    fail(d, POLY_IMPORT_ERR_PARSE, "invalid/unsupported graph");
    poly_model_free(d->model);
    d->model = NULL;
  }
  PolyModel *result = d->model;
  if (scoped) result = model_factory_end(&scope, result);
  for (int i = 0; i < d->nvalues; i++)
    free(d->values[i].host);
  cJSON_Delete(d->dimensions);
  free(d);
  return result;
}
PolyModel *poly_onnx_load(
    const uint8_t *data,
    int64_t len,
    const PolyOnnxOptions *options,
    PolyDevice device
) {
  return poly_onnx_load_into(NULL, data, len, options, device);
}
