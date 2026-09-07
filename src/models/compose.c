/* Approved C-only construction boundary: expand a bounded JSON definition into
 * existing Tensor operations and seal one Model. No definition survives sealing.
 * Math is owned by Tensor/nn (not this parser); sharing means reusing the same
 * parameter Tensor, as multiple calls to one tinygrad.nn.Linear do. */
#include "compose.h"
#include "../nn.h"
#include "../tensor.h"
#include "mlp.h"
#include "../../vendor/cjson/cJSON.h"
#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEF_NODES 1024
#define DEF_HANDLES (DEF_NODES * 8)
#define DEF_NAME 192
#define DEF_IO 64
#define DEF_ELEMENTS (16 * 1024 * 1024)

typedef struct {
  const char *name; /* borrowed from JSON */
  PolyTensor *tensor; /* borrowed from handles */
} DefValue;

typedef struct {
  char name[DEF_NAME];
  PolyTensor *weight, *bias;
  int64_t in_features, out_features;
} DefLinear;

typedef struct {
  PolyCtx *ctx;
  PolyModel *model;
  PolyModelError *err;
  const char *family;
  cJSON *modules;
  DefValue values[DEF_NODES + DEF_IO];
  int n_values, expanded, n_handles, n_linear;
  int64_t storage_elements;
  uint64_t seed;
  PolyTensor *handles[DEF_HANDLES];
  DefLinear linear[DEF_NODES];
  const cJSON *used_modules[DEF_NODES];
  int n_used_modules;
} Definition;

static bool def_error(Definition *d, const char *path, const char *fmt, ...) {
  if (!d->err->code) {
    d->err->code = POLY_STATUS_INVALID;
    d->err->func =
        !strcmp(d->family, "sequential") ? "poly_sequential_from_json" : "poly_graph_from_json";
    int n = snprintf(d->err->message, sizeof(d->err->message), "%s: ", path);
    if (n < 0 || n >= (int)sizeof(d->err->message)) return false;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(d->err->message + n, sizeof(d->err->message) - (size_t)n, fmt, ap);
    va_end(ap);
  }
  return false;
}

static cJSON *field(const cJSON *obj, const char *key) {
  return cJSON_GetObjectItemCaseSensitive(obj, key);
}

static bool fields(Definition *d, const cJSON *obj, const char *allowed, const char *path) {
  if (!cJSON_IsObject(obj)) return def_error(d, path, "expected object");
  for (const cJSON *v = obj->child; v; v = v->next) {
    char key[128];
    int n = snprintf(key, sizeof(key), "|%s|", v->string);
    if (!v->string[0] || strchr(v->string, '|') || n < 0 || n >= (int)sizeof(key) ||
        !strstr(allowed, key))
      return def_error(d, path, "unknown field '%s'", v->string);
  }
  return true;
}

static bool name_ok(Definition *d, const char *name, const char *path) {
  if (!name || !name[0] || strlen(name) > 63 ||
      !(isalpha((unsigned char)name[0]) || name[0] == '_'))
    return def_error(d, path, "expected identifier (1..63 ASCII letters/digits/underscores)");
  for (const unsigned char *p = (const unsigned char *)name; *p; p++)
    if (*p >= 128 || !(isalnum(*p) || *p == '_'))
      return def_error(d, path, "invalid identifier '%s'", name);
  return true;
}

static bool path_join(Definition *d, char *out, const char *prefix, const char *name) {
  int n = snprintf(out, DEF_NAME, "%s.%s", prefix, name);
  return n >= 0 && n < DEF_NAME ? true : def_error(d, prefix, "expanded name exceeds 191 bytes");
}

static bool integer(
    Definition *d,
    const cJSON *v,
    int64_t lo,
    int64_t hi,
    int64_t *out,
    const char *path
) {
  if (!cJSON_IsNumber(v) || !isfinite(v->valuedouble) || v->valuedouble < (double)lo ||
      v->valuedouble > (double)hi || floor(v->valuedouble) != v->valuedouble)
    return def_error(d, path, "expected integer in [%lld,%lld]", (long long)lo, (long long)hi);
  *out = (int64_t)v->valuedouble;
  return true;
}

/* Reject ambiguous duplicate keys at every level, including unused declarations.
 * The lexical preflight bounds recursion before cJSON itself parses the input. */
static bool unique_keys(Definition *d, const cJSON *obj, int *budget) {
  if (--*budget < 0) return def_error(d, "$", "JSON node budget exceeded");
  for (const cJSON *a = obj->child; a; a = a->next) {
    if (cJSON_IsObject(obj))
      for (const cJSON *b = obj->child; b != a; b = b->next)
        if (!strcmp(a->string, b->string))
          return def_error(d, "$", "duplicate key '%s'", a->string);
    if (!unique_keys(d, a, budget)) return false;
  }
  return true;
}

static bool json_preflight(Definition *d, const char *json, int len) {
  if (!json || len <= 0 || len > 1024 * 1024)
    return def_error(d, "$", "expected 1..1048576 JSON bytes");
  int depth = 0;
  bool quoted = false;
  for (int i = 0; i < len; i++) {
    unsigned char c = (unsigned char)json[i];
    if (!c) return def_error(d, "$", "embedded NUL");
    if (quoted && c == '\\') {
      if (i + 5 < len && !memcmp(json + i, "\\u0000", 6)) return def_error(d, "$", "escaped NUL");
      i++;
    } else if (c == '"')
      quoted = !quoted;
    else if (quoted && c < 32)
      return def_error(d, "$", "control character in string");
    else if (!quoted && (c == '-' || (c >= '0' && c <= '9'))) {
      /* cJSON's strtod parser accepts 01 and 1.; require JSON number grammar
       * before schema integer/range validation, without another numeric evaluator. */
      int p = i;
      if (json[p] == '-') p++;
      if (p == len || json[p] < '0' || json[p] > '9')
        return def_error(d, "$", "invalid JSON number");
      if (json[p] == '0')
        p++;
      else
        while (p < len && json[p] >= '0' && json[p] <= '9')
          p++;
      if (p < len && json[p] >= '0' && json[p] <= '9')
        return def_error(d, "$", "invalid JSON number");
      if (p < len && json[p] == '.') {
        int start = ++p;
        while (p < len && json[p] >= '0' && json[p] <= '9')
          p++;
        if (p == start) return def_error(d, "$", "invalid JSON number");
      }
      if (p < len && (json[p] == 'e' || json[p] == 'E')) {
        p++;
        if (p < len && (json[p] == '+' || json[p] == '-')) p++;
        int start = p;
        while (p < len && json[p] >= '0' && json[p] <= '9')
          p++;
        if (p == start) return def_error(d, "$", "invalid JSON number");
      }
      i = p - 1;
    } else if (!quoted && (c == '{' || c == '[')) {
      if (++depth > 32) return def_error(d, "$", "JSON nesting exceeds 32");
    } else if (!quoted && (c == '}' || c == ']'))
      depth--;
  }
  return true;
}

static PolyTensor *own(Definition *d, PolyTensor *t, const char *path) {
  if (!t) {
    def_error(d, path, "Tensor construction failed");
    return NULL;
  }
  if (d->n_handles == DEF_HANDLES) {
    poly_tensor_release(t);
    def_error(d, path, "Tensor handle budget exceeded");
    return NULL;
  }
  d->handles[d->n_handles++] = t;
  return t;
}

static bool shape(Definition *d, const cJSON *spec, int64_t *dims, int *ndim, const char *path) {
  if (!cJSON_IsArray(spec) || (*ndim = cJSON_GetArraySize(spec)) > 8)
    return def_error(d, path, "expected shape array of rank 0..8");
  int64_t elements = 1;
  for (int i = 0; i < *ndim; i++) {
    if (!integer(d, cJSON_GetArrayItem(spec, i), 1, DEF_ELEMENTS, &dims[i], path)) return false;
    if (elements > DEF_ELEMENTS / dims[i])
      return def_error(d, path, "shape exceeds element budget");
    elements *= dims[i];
  }
  return true;
}

static bool tensor_shape(Definition *d, PolyTensor *t, int64_t *dims, int *ndim, const char *path) {
  PolyUOp *u = poly_tensor_uop_physical(t);
  *ndim = poly_uop_ndim(d->ctx, u);
  const int64_t *s = poly_uop_max_shape_dims(d->ctx, u);
  if (*ndim < 0 || *ndim > 8 || (*ndim && !s))
    return def_error(d, path, "expected concrete rank <= 8");
  int64_t elements = 1;
  for (int i = 0; i < *ndim; i++) {
    if (s[i] < 1 || elements > DEF_ELEMENTS / s[i])
      return def_error(d, path, "shape exceeds element budget");
    dims[i] = s[i];
    elements *= s[i];
  }
  return true;
}

static bool reserve_storage(Definition *d, int64_t elements, const char *path) {
  if (elements > DEF_ELEMENTS - d->storage_elements)
    return def_error(d, path, "named storage exceeds element budget");
  d->storage_elements += elements;
  return true;
}

static bool publish_output(Definition *d, const char *name, PolyTensor *t) {
  int64_t dims[8], elements = 1;
  int ndim;
  if (!tensor_shape(d, t, dims, &ndim, name)) return false;
  for (int i = 0; i < ndim; i++)
    elements *= dims[i];
  return reserve_storage(d, elements, name) &&
         poly_model_output(d->model, name, t) == POLY_STATUS_OK;
}

static bool add_value(Definition *d, const char *name, PolyTensor *t, const char *path) {
  if (!name_ok(d, name, path)) return false;
  for (int i = 0; i < d->n_values; i++)
    if (!strcmp(d->values[i].name, name)) return def_error(d, path, "duplicate value '%s'", name);
  if (d->n_values == DEF_NODES + DEF_IO) return def_error(d, path, "value budget exceeded");
  d->values[d->n_values++] = (DefValue){name, t};
  return true;
}

static PolyTensor *lookup(Definition *d, const cJSON *ref, const char *path) {
  if (cJSON_IsString(ref))
    for (int i = 0; i < d->n_values; i++)
      if (!strcmp(d->values[i].name, ref->valuestring)) return d->values[i].tensor;
  def_error(
      d, path, "unknown or forward value reference '%s'",
      cJSON_IsString(ref) ? ref->valuestring : "<not a string>"
  );
  return NULL;
}

static PolyTensor *activation(Definition *d, PolyTensor *x, const char *kind, const char *path) {
  if (!strcmp(kind, "none")) return x;
  PolyTensor *out = NULL;
  if (!strcmp(kind, "relu"))
    out = poly_tensor_relu(d->ctx, x);
  else if (!strcmp(kind, "sigmoid"))
    out = poly_tensor_sigmoid(d->ctx, x);
  else if (!strcmp(kind, "tanh"))
    out = poly_tensor_tanh(d->ctx, x);
  else if (!strcmp(kind, "silu"))
    out = poly_tensor_silu(d->ctx, x);
  else if (!strcmp(kind, "gelu"))
    out = poly_tensor_gelu(d->ctx, x);
  else {
    def_error(d, path, "unknown activation '%s'", kind);
    return NULL;
  }
  return own(d, out, path);
}

static PolyTensor *apply(Definition *, const cJSON *, PolyTensor **, int, const char *, int);

static PolyTensor *sequence(
    Definition *d,
    const cJSON *layers,
    PolyTensor *x,
    const char *path,
    int depth
) {
  if (!cJSON_IsArray(layers) || !layers->child) {
    def_error(d, path, "expected nonempty layer array");
    return NULL;
  }
  for (const cJSON *layer = layers->child; layer; layer = layer->next) {
    const cJSON *name = field(layer, "name");
    if (!cJSON_IsString(name)) {
      def_error(d, path, "layer needs a name");
      return NULL;
    }
    if (!name_ok(d, name->valuestring, path)) return NULL;
    if (field(layer, "inputs")) {
      def_error(d, path, "Sequential layers cannot specify inputs");
      return NULL;
    }
    for (const cJSON *prior = layers->child; prior != layer; prior = prior->next) {
      const cJSON *pn = field(prior, "name");
      if (!strcmp(pn->valuestring, name->valuestring)) {
        def_error(d, path, "duplicate layer '%s'", name->valuestring);
        return NULL;
      }
    }
    char child[DEF_NAME];
    if (!path_join(d, child, path, name->valuestring)) return NULL;
    x = apply(d, layer, &x, 1, child, depth + 1);
    if (!x) return NULL;
  }
  return x;
}

static PolyTensor *apply(
    Definition *d,
    const cJSON *spec,
    PolyTensor **inputs,
    int n,
    const char *path,
    int depth
) {
  if (depth > 16 || ++d->expanded > DEF_NODES) {
    def_error(d, path, "expansion budget exceeded (depth 16, calls 1024)");
    return NULL;
  }
  const cJSON *call = field(spec, "call"), *type = field(spec, "type");
  if (call) {
    if (!fields(d, spec, "|name||inputs||call|", path)) return NULL;
    const cJSON *module = cJSON_IsString(call) ? field(d->modules, call->valuestring) : NULL;
    if (!module) {
      def_error(d, path, "unknown shared component");
      return NULL;
    }
    /* Shared declarations are deliberately leaf components, not recursive modules. */
    const cJSON *mt = field(module, "type");
    if (field(module, "call") || field(module, "name") || field(module, "inputs") ||
        !cJSON_IsString(mt) || !strcmp(mt->valuestring, "repeat")) {
      def_error(d, path, "shared declarations must be leaf components");
      return NULL;
    }
    bool seen = false;
    for (int i = 0; i < d->n_used_modules; i++)
      seen |= d->used_modules[i] == module;
    if (!seen) d->used_modules[d->n_used_modules++] = module;
    char scope[DEF_NAME];
    if (!path_join(d, scope, "modules", call->valuestring)) return NULL;
    return apply(d, module, inputs, n, scope, depth + 1);
  }
  if (!cJSON_IsString(type)) {
    def_error(d, path, "missing component type");
    return NULL;
  }
  const char *kind = type->valuestring;
  const char *allowed = "|name||inputs||type|";
  if (!strcmp(kind, "linear"))
    allowed = "|name||inputs||type||out_features||bias||activation|";
  else if (!strcmp(kind, "repeat"))
    allowed = "|name||inputs||type||count||body|";
  else if (!strcmp(kind, "reshape"))
    allowed = "|name||inputs||type||shape|";
  if (!fields(d, spec, allowed, path)) return NULL;
  bool binary =
      !strcmp(kind, "add") || !strcmp(kind, "sub") || !strcmp(kind, "mul") || !strcmp(kind, "div");
  if (n != (binary ? 2 : 1)) {
    def_error(d, path, "expected %d inputs, received %d", binary ? 2 : 1, n);
    return NULL;
  }
  PolyTensor *x = inputs[0], *out = NULL;
  int64_t dims[8];
  int rank;
  if (!tensor_shape(d, x, dims, &rank, path)) return NULL;
  if (!strcmp(kind, "repeat")) {
    int64_t count;
    const cJSON *body = field(spec, "body");
    if (!integer(d, field(spec, "count"), 1, DEF_NODES, &count, path)) return NULL;
    for (int64_t i = 0; i < count; i++) {
      char index[24], scope[DEF_NAME];
      snprintf(index, sizeof(index), "%lld", (long long)i);
      if (!path_join(d, scope, path, index)) return NULL;
      if (cJSON_IsArray(body))
        x = sequence(d, body, x, scope, depth + 1);
      else {
        if (!cJSON_IsObject(body) || field(body, "name") || field(body, "inputs")) {
          def_error(
              d, path, "body must be a component without name/inputs, or a named layer array"
          );
          return NULL;
        }
        x = apply(d, body, &x, 1, scope, depth + 1);
      }
      if (!x) return NULL;
    }
    return x;
  }
  if (!strcmp(kind, "linear")) {
    int64_t width;
    const cJSON *bias = field(spec, "bias"), *act = field(spec, "activation");
    if (!integer(d, field(spec, "out_features"), 1, DEF_ELEMENTS, &width, path)) return NULL;
    if (rank < 1 || (bias && !cJSON_IsBool(bias)) || (act && !cJSON_IsString(act))) {
      def_error(d, path, "linear needs rank >= 1, boolean bias and string activation");
      return NULL;
    }
    int64_t fan_in = dims[rank - 1], work = width;
    for (int i = 0; i < rank; i++) {
      if (work > DEF_ELEMENTS / dims[i]) {
        def_error(d, path, "linear exceeds element budget");
        return NULL;
      }
      work *= dims[i];
    }
    DefLinear *layer = NULL;
    for (int i = 0; i < d->n_linear; i++)
      if (!strcmp(d->linear[i].name, path)) layer = &d->linear[i];
    if (layer && layer->in_features != fan_in) {
      def_error(
          d, path, "shared linear expected %lld input features, received %lld",
          (long long)layer->in_features, (long long)fan_in
      );
      return NULL;
    }
    if (!layer) {
      bool use_bias = !bias || cJSON_IsTrue(bias);
      int64_t elements = width * fan_in + (use_bias ? width : 0);
      if (d->n_linear == DEF_NODES || !reserve_storage(d, elements, path)) {
        def_error(d, path, "parameter budget exceeded");
        return NULL;
      }
      layer = &d->linear[d->n_linear++];
      snprintf(layer->name, sizeof(layer->name), "%s", path);
      layer->in_features = fan_in;
      layer->out_features = width;
      char name[DEF_NAME];
      int64_t ws[] = {width, fan_in};
      if (!path_join(d, name, path, "weight")) return NULL;
      layer->weight = own(d, poly_model_param(d->model, name, POLY_FLOAT32, ws, 2), path);
      if (!layer->weight) return NULL;
      /* Wrapper-local retention: never change the caller's context policy. */
      if (poly_tensor_set_logical_policy(d->ctx, layer->weight, POLY_LOGICAL_ALWAYS)) return NULL;
      if (use_bias) {
        if (!path_join(d, name, path, "bias")) return NULL;
        layer->bias = own(d, poly_model_param(d->model, name, POLY_FLOAT32, &width, 1), path);
        if (!layer->bias) return NULL;
        if (poly_tensor_set_logical_policy(d->ctx, layer->bias, POLY_LOGICAL_ALWAYS)) return NULL;
      }
    }
    out = own(d, poly_tensor_linear_apply(d->ctx, x, layer->weight, layer->bias), path);
    return out ? activation(d, out, act ? act->valuestring : "none", path) : NULL;
  }
  if (binary) {
    int64_t rhs[8];
    int rr;
    if (!tensor_shape(d, inputs[1], rhs, &rr, path)) return NULL;
    int br = rank > rr ? rank : rr;
    int64_t elements = 1;
    for (int i = 1; i <= br; i++) {
      int64_t a = i <= rank ? dims[rank - i] : 1, b = i <= rr ? rhs[rr - i] : 1;
      if (a != b && a != 1 && b != 1) {
        def_error(
            d, path, "incompatible broadcast dimensions %lld and %lld", (long long)a, (long long)b
        );
        return NULL;
      }
      int64_t size = a > b ? a : b;
      if (elements > DEF_ELEMENTS / size) {
        def_error(d, path, "broadcast exceeds element budget");
        return NULL;
      }
      elements *= size;
    }
    if (!strcmp(kind, "div"))
      out = poly_tensor_div(d->ctx, x, inputs[1], 0);
    else if (!strcmp(kind, "sub")) {
      PolyTensor *neg = own(d, poly_tensor_alu1(d->ctx, POLY_OP_NEG, inputs[1]), path);
      out = neg ? poly_tensor_alu2(d->ctx, POLY_OP_ADD, x, neg) : NULL;
    } else
      out =
          poly_tensor_alu2(d->ctx, !strcmp(kind, "add") ? POLY_OP_ADD : POLY_OP_MUL, x, inputs[1]);
  } else if (!strcmp(kind, "sum") || !strcmp(kind, "mean")) {
    int64_t axes[8], elements = 1;
    for (int i = 0; i < rank; i++) {
      axes[i] = i;
      elements *= dims[i];
    }
    out = own(d, poly_tensor_sum(d->ctx, x, axes, rank, false), path);
    if (!out || !strcmp(kind, "sum")) return out;
    PolyTensor *scale =
        own(d, poly_tensor_const_like_float(d->ctx, out, 1.0 / (double)elements), path);
    return scale ? own(d, poly_tensor_alu2(d->ctx, POLY_OP_MUL, out, scale), path) : NULL;
  } else if (!strcmp(kind, "reshape")) {
    int64_t dest[8], before = 1, after = 1;
    int dr;
    if (!shape(d, field(spec, "shape"), dest, &dr, path)) return NULL;
    for (int i = 0; i < rank; i++)
      before *= dims[i];
    for (int i = 0; i < dr; i++)
      after *= dest[i];
    if (before != after) {
      def_error(d, path, "reshape changes element count");
      return NULL;
    }
    out = poly_tensor_reshape(d->ctx, x, dest, dr);
  } else if (!strcmp(kind, "identity"))
    return x;
  else if (!strcmp(kind, "none")) {
    def_error(d, path, "use identity for a no-op component");
    return NULL;
  } else if (!strcmp(kind, "square"))
    out = poly_tensor_alu2(d->ctx, POLY_OP_MUL, x, x);
  else if (!strcmp(kind, "exp"))
    out = poly_tensor_exp(d->ctx, x);
  else if (!strcmp(kind, "log"))
    out = poly_tensor_log(d->ctx, x);
  else
    return activation(d, x, kind, path);
  return own(d, out, path);
}

static bool input(Definition *d, const char *name, const cJSON *spec, bool sequential) {
  char path[DEF_NAME];
  if (!name_ok(d, name, "inputs") || !path_join(d, path, "inputs", name)) return false;
  if (!fields(d, spec, sequential ? "|name||shape||dtype|" : "|shape||dtype||role|", path))
    return false;
  const cJSON *dtype = field(spec, "dtype"), *role = field(spec, "role");
  if (!cJSON_IsString(dtype) || strcmp(dtype->valuestring, "float32"))
    return def_error(d, path, "only explicit float32 is supported in modeldef@1");
  bool target = role && cJSON_IsString(role) && !strcmp(role->valuestring, "target");
  if (role && !target && (!cJSON_IsString(role) || strcmp(role->valuestring, "input")))
    return def_error(d, path, "role must be input or target");
  int64_t dims[8];
  int ndim;
  if (!shape(d, field(spec, "shape"), dims, &ndim, path)) return false;
  int64_t elements = 1;
  for (int i = 0; i < ndim; i++)
    elements *= dims[i];
  if (!reserve_storage(d, elements, path)) return false;
  PolyTensor *t =
      own(d,
          target ? poly_model_target(d->model, name, POLY_FLOAT32, dims, ndim)
                 : poly_model_input(d->model, name, POLY_FLOAT32, dims, ndim),
          path);
  if (!t) return false;
  if (poly_tensor_set_logical_policy(d->ctx, t, POLY_LOGICAL_ALWAYS))
    return def_error(d, path, "cannot preserve logical roots");
  return add_value(d, name, t, path);
}

static bool names(Definition *d, const cJSON *array, const char **out, int *n, const char *path) {
  if (!cJSON_IsArray(array) || (*n = cJSON_GetArraySize(array)) > DEF_IO || !*n)
    return def_error(d, path, "expected 1..64 names");
  int i = 0;
  for (const cJSON *v = array->child; v; v = v->next) {
    if (!cJSON_IsString(v) || !name_ok(d, v->valuestring, path)) return false;
    for (int j = 0; j < i; j++)
      if (!strcmp(out[j], v->valuestring))
        return def_error(d, path, "duplicate name '%s'", v->valuestring);
    out[i++] = v->valuestring;
  }
  return true;
}

static bool construct(Definition *d, const cJSON *root) {
  const cJSON *type = field(root, "type"), *format = field(root, "format");
  if (format && (!cJSON_IsString(format) || strcmp(format->valuestring, "poly.modeldef@1")))
    return def_error(d, "$", "expected format poly.modeldef@1");
  bool seq = !strcmp(d->family, "sequential");
  if (type && (!cJSON_IsString(type) || strcmp(type->valuestring, d->family)))
    return def_error(d, "$", "type must match %s factory", d->family);
  if (!fields(
          d, root,
          seq ? "|format||type||seed||input||layers||output||modules|"
              : "|format||type||seed||inputs||nodes||outputs||entrypoints||modules|",
          "$"
      ))
    return false;
  int64_t seed = 42;
  if (field(root, "seed") && !integer(d, field(root, "seed"), 0, 9007199254740991LL, &seed, "seed"))
    return false;
  d->seed = (uint64_t)seed;
  d->modules = field(root, "modules");
  if (d->modules) {
    if (!cJSON_IsObject(d->modules) || cJSON_GetArraySize(d->modules) > DEF_NODES)
      return def_error(d, "modules", "expected at most 1024 named leaf components");
    for (const cJSON *m = d->modules->child; m; m = m->next)
      if (!name_ok(d, m->string, "modules")) return false;
  }
  const char *ins[DEF_IO], *outs[DEF_IO];
  int ni = 0, no = 0;
  if (seq) {
    const cJSON *in = field(root, "input"), *name = field(in, "name"),
                *output = field(root, "output");
    if (!cJSON_IsString(name) || !cJSON_IsString(output))
      return def_error(d, "$", "input.name and output must be strings");
    if (!input(d, name->valuestring, in, true) || !name_ok(d, output->valuestring, "output"))
      return false;
    PolyTensor *y = sequence(d, field(root, "layers"), d->values[0].tensor, "layers", 0);
    if (!y || !publish_output(d, output->valuestring, y)) return false;
    ins[ni++] = name->valuestring;
    outs[no++] = output->valuestring;
  } else {
    const cJSON *inputs = field(root, "inputs"), *nodes = field(root, "nodes"),
                *outputs = field(root, "outputs");
    if (!cJSON_IsObject(inputs) || !inputs->child || cJSON_GetArraySize(inputs) > DEF_IO)
      return def_error(d, "inputs", "expected 1..64 named input schemas");
    for (const cJSON *v = inputs->child; v; v = v->next) {
      if (!input(d, v->string, v, false)) return false;
      ins[ni++] = v->string;
    }
    if (!cJSON_IsArray(nodes) || cJSON_GetArraySize(nodes) > DEF_NODES)
      return def_error(d, "nodes", "expected at most 1024 ordered nodes");
    for (const cJSON *node = nodes->child; node; node = node->next) {
      const cJSON *name = field(node, "name"), *refs = field(node, "inputs");
      char path[DEF_NAME];
      if (!cJSON_IsString(name) || !name_ok(d, name->valuestring, "nodes") ||
          !path_join(d, path, "nodes", name->valuestring))
        return false;
      for (int i = 0; i < d->n_values; i++)
        if (!strcmp(d->values[i].name, name->valuestring))
          return def_error(d, path, "duplicate value '%s'", name->valuestring);
      if (!cJSON_IsArray(refs) || cJSON_GetArraySize(refs) < 1 || cJSON_GetArraySize(refs) > 2)
        return def_error(d, path, "expected 1 or 2 input references");
      PolyTensor *args[2];
      int n = 0;
      for (const cJSON *ref = refs->child; ref; ref = ref->next) {
        args[n] = lookup(d, ref, path);
        if (!args[n++]) return false;
      }
      PolyTensor *y = apply(d, node, args, n, path, 0);
      if (!y || !add_value(d, name->valuestring, y, path)) return false;
    }
    if (!cJSON_IsObject(outputs) || !outputs->child || cJSON_GetArraySize(outputs) > DEF_IO)
      return def_error(d, "outputs", "expected 1..64 named outputs");
    for (const cJSON *v = outputs->child; v; v = v->next) {
      if (!name_ok(d, v->string, "outputs")) return false;
      PolyTensor *y = lookup(d, v, "outputs");
      if (!y || !publish_output(d, v->string, y)) return false;
      outs[no++] = v->string;
    }
  }
  for (const cJSON *m = d->modules ? d->modules->child : NULL; m; m = m->next) {
    bool used = false;
    for (int i = 0; i < d->n_used_modules; i++)
      used |= d->used_modules[i] == m;
    if (!used) return def_error(d, "modules", "unused shared component '%s'", m->string);
  }
  const cJSON *entries = field(root, "entrypoints");
  if (!entries)
    return poly_model_entrypoint(d->model, "forward", ins, ni, outs, no, NULL) == POLY_STATUS_OK;
  if (!cJSON_IsArray(entries) || !entries->child || cJSON_GetArraySize(entries) > DEF_IO)
    return def_error(d, "entrypoints", "expected 1..64 entrypoints");
  for (const cJSON *ep = entries->child; ep; ep = ep->next) {
    if (!fields(d, ep, "|name||inputs||outputs||objective|", "entrypoints")) return false;
    const cJSON *name = field(ep, "name"), *objective = field(ep, "objective");
    if (!cJSON_IsString(name) || !name_ok(d, name->valuestring, "entrypoints")) return false;
    if (!names(d, field(ep, "inputs"), ins, &ni, name->valuestring) ||
        !names(d, field(ep, "outputs"), outs, &no, name->valuestring))
      return false;
    if (objective && !cJSON_IsString(objective))
      return def_error(d, name->valuestring, "objective must name an output");
    PolyEntrypointOptions opts = {.objective = objective ? objective->valuestring : NULL};
    if (poly_model_entrypoint(d->model, name->valuestring, ins, ni, outs, no, &opts)) return false;
  }
  return true;
}

static PolyModel *compose_from_json(
    PolyCtx *ctx,
    const char *json,
    int len,
    PolyModelError *err,
    const char *family
) {
  PolyModelError local = {0};
  if (!err) err = &local;
  memset(err, 0, sizeof(*err));
  Definition *d = calloc(1, sizeof(*d));
  if (!d) {
    err->code = POLY_STATUS_NOMEM;
    err->func = __func__;
    snprintf(err->message, sizeof(err->message), "definition allocation failed");
    return NULL;
  }
  d->ctx = ctx;
  d->err = err;
  d->family = family;
  cJSON *root = NULL;
  bool ok = false;
  if (!ctx) {
    def_error(d, "$", "a live borrowed context is required");
    goto done;
  }
  if (poly_ctx_get_logical_policy(ctx) == POLY_LOGICAL_NEVER) {
    def_error(d, "$", "portable Model requires logical construction");
    goto done;
  }
  if (!json_preflight(d, json, len)) goto done;
  const char *end = NULL;
  root = cJSON_ParseWithLengthOpts(json, (size_t)len, &end, false);
  if (!root) {
    def_error(d, "$", "invalid JSON near byte %ld", end ? (long)(end - json) : 0L);
    goto done;
  }
  while (end < json + len && isspace((unsigned char)*end))
    end++;
  if (end != json + len) {
    def_error(d, "$", "trailing JSON data");
    goto done;
  }
  int budget = 16384;
  if (!unique_keys(d, root, &budget)) goto done;
  d->model = poly_model_new(ctx, NULL);
  if (!d->model) {
    def_error(d, "$", "Model allocation failed");
    goto done;
  }
  if (!construct(d, root)) goto done;
  if (poly_model_build(d->model, err)) goto done;
  /* Initialize through the coherent buffer API after sealing. These bytes are
   * model state, never host-pointer aliases or a second residency table. */
  for (int i = 0; i < d->n_linear; i++) {
    DefLinear *layer = &d->linear[i];
    int64_t count = layer->in_features * layer->out_features;
    float *values = malloc((size_t)count * sizeof(float));
    char name[DEF_NAME];
    if (!values) {
      def_error(d, layer->name, "initializer allocation failed");
      goto done;
    }
    if (!path_join(d, name, layer->name, "weight")) {
      free(values);
      goto done;
    }
    poly_init_param_kaiming(d->seed, name, values, count, layer->in_features);
    int rc = poly_model_write_buf_named(d->model, name, values, (size_t)count * sizeof(float));
    if (!rc && layer->bias) {
      memset(values, 0, (size_t)layer->out_features * sizeof(float));
      if (!path_join(d, name, layer->name, "bias")) {
        free(values);
        goto done;
      }
      rc = poly_model_write_buf_named(
          d->model, name, values, (size_t)layer->out_features * sizeof(float)
      );
    }
    free(values);
    if (rc) {
      def_error(d, layer->name, "initializer write failed");
      goto done;
    }
  }
  ok = true;
done:
  if (!ok && !err->code) {
    const PolyModelError *cause = poly_model_last_error(d->model);
    if (cause && cause->code)
      *err = *cause;
    else
      def_error(d, "$", "construction failed");
  }
  /* Model owns retained graph roots; this frontend owns every temporary Tensor
   * handle, including identity returns. The borrowed context always stays live. */
  for (int i = d->n_handles - 1; i >= 0; i--)
    poly_tensor_release(d->handles[i]);
  PolyModel *result = ok ? d->model : NULL;
  if (!ok) poly_model_free(d->model);
  cJSON_Delete(root);
  free(d);
  return result;
}

PolyModel *poly_sequential_from_json(PolyCtx *ctx, const char *json, int len, PolyModelError *err) {
  return compose_from_json(ctx, json, len, err, "sequential");
}

PolyModel *poly_graph_from_json(PolyCtx *ctx, const char *json, int len, PolyModelError *err) {
  return compose_from_json(ctx, json, len, err, "graph");
}
