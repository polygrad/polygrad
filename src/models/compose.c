/* Approved C-only construction boundary: expand a bounded JSON definition into
 * existing Tensor operations and seal one Model. No definition survives sealing.
 * Math is owned by Tensor/nn (not this parser); sharing means reusing the same
 * parameter Tensor, as multiple calls to one tinygrad.nn.Linear do. */
#include "models.h"
#include "factory.h"
#include "layers.h"
#include "../nn/nn.h"
#include "../tensor.h"
#include "../mixin/gradient.h"
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
  const char *kind;
  PolyTensor *weight, *bias;
  int64_t in_features, out_features;
} DefLayer;

typedef struct {
  PolyCtx *ctx;
  PolyModel *model;
  PolyModelError *err;
  const char *family;
  cJSON *modules;
  DefValue values[DEF_NODES + DEF_IO];
  int n_values, expanded, n_handles, n_layers;
  int64_t storage_elements;
  uint64_t seed;
  PolyTensor *handles[DEF_HANDLES];
  DefLayer layers[DEF_NODES];
  PolyUOp *batch;
  const char *batch_name;
  int64_t batch_min, batch_max;
  const cJSON *used_modules[DEF_NODES];
  int n_used_modules;
  PolyTensor *states[DEF_IO];
  const char *state_names[DEF_IO];
  int n_states;
  PolyUOp *controls[DEF_IO];
  const char *control_names[DEF_IO];
  int n_controls;
} Definition;

static bool def_error(Definition *d, const char *path, const char *fmt, ...) {
  if (!d->err->code) {
    d->err->code = POLY_STATUS_INVALID;
    d->err->func = "poly_model_from_config";
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

static bool shape(
    Definition *d,
    const cJSON *spec,
    PolyUOp **dims,
    int64_t *max_dims,
    int *ndim,
    const char *path
) {
  if (!cJSON_IsArray(spec) || (*ndim = cJSON_GetArraySize(spec)) > 8)
    return def_error(d, path, "expected shape array of rank 0..8");
  int64_t elements = 1;
  for (int i = 0; i < *ndim; i++) {
    const cJSON *dim = cJSON_GetArrayItem(spec, i);
    if (cJSON_IsObject(dim)) {
      int64_t lo, hi;
      const cJSON *name = field(dim, "name");
      if (i != 0) return def_error(d, path, "only the leading dimension may be bounded");
      if (!fields(d, dim, "|name||min||max|", path)) return false;
      if (!cJSON_IsString(name)) return def_error(d, path, "bounded dimension requires a name");
      if (!name_ok(d, name->valuestring, path) ||
          !integer(d, field(dim, "min"), 1, DEF_ELEMENTS, &lo, path) ||
          !integer(d, field(dim, "max"), lo, DEF_ELEMENTS, &hi, path))
        return false;
      if (d->batch &&
          (strcmp(d->batch_name, name->valuestring) || d->batch_min != lo || d->batch_max != hi))
        return def_error(d, path, "bounded dimensions must use the same batch name and bounds");
      if (!d->batch) {
        d->batch_name = name->valuestring;
        d->batch_min = lo;
        d->batch_max = hi;
        PolyUOp *var = poly_uop_variable(
            d->ctx, d->batch_name, poly_arg_int(lo), poly_arg_int(hi), POLY_WEAKINT, 1, false
        );
        d->batch = var ? poly_uop_bind(d->ctx, var, hi) : NULL;
        if (!d->batch) return def_error(d, path, "batch variable construction failed");
      }
      dims[i] = d->batch;
      max_dims[i] = hi;
    } else {
      if (!integer(d, dim, 1, DEF_ELEMENTS, &max_dims[i], path)) return false;
      dims[i] = poly_uop0(d->ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(max_dims[i]));
    }
    if (elements > DEF_ELEMENTS / max_dims[i])
      return def_error(d, path, "shape exceeds element budget");
    elements *= max_dims[i];
  }
  return true;
}

static bool tensor_shape(Definition *d, PolyTensor *t, int64_t *dims, int *ndim, const char *path) {
  PolyUOp *u = poly_tensor_uop_physical(t);
  *ndim = poly_uop_ndim(d->ctx, u);
  const int64_t *s = poly_uop_max_shape_dims(d->ctx, u);
  if (*ndim < 0 || *ndim > 8 || (*ndim && !s))
    return def_error(d, path, "expected bounded shape of rank <= 8");
  int64_t elements = 1;
  for (int i = 0; i < *ndim; i++) {
    if (s[i] < 1 || elements > DEF_ELEMENTS / s[i])
      return def_error(d, path, "shape exceeds element budget");
    dims[i] = s[i];
    elements *= s[i];
  }
  return true;
}

static bool dtype(Definition *d, const cJSON *v, PolyDType *dt, const char *path) {
  if (!cJSON_IsString(v) || !poly_dtype_by_id(poly_dtype_id_by_name(v->valuestring), dt) ||
      poly_dtype_is_weak(*dt) || poly_dtype_itemsize(*dt) == 0)
    return def_error(d, path, "expected a concrete scalar dtype");
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
  PolyTensor *out = model_activation(d->ctx, x, kind);
  if (!out) {
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
  if (!strcmp(kind, "arange")) {
    int64_t start, stop, step;
    PolyDType dt;
    if (n || !fields(d, spec, "|name||inputs||type||start||stop||step||dtype|", path) ||
        !integer(d, field(spec, "start"), -DEF_ELEMENTS, DEF_ELEMENTS, &start, path) ||
        !integer(d, field(spec, "stop"), -DEF_ELEMENTS, DEF_ELEMENTS, &stop, path) ||
        !integer(d, field(spec, "step"), 1, DEF_ELEMENTS, &step, path) ||
        !dtype(d, field(spec, "dtype"), &dt, path) || stop <= start ||
        (stop - start + step - 1) / step > DEF_ELEMENTS) {
      def_error(d, path, "arange requires no inputs and a bounded positive-step range");
      return NULL;
    }
    return own(
        d,
        poly_tensor_arange_int(
            d->ctx, start, stop, step, dt, poly_ctx_get_preferred_device(d->ctx)
        ),
        path
    );
  }
  const char *allowed = "|name||inputs||type|";
  if (!strcmp(kind, "linear"))
    allowed = "|name||inputs||type||out_features||bias||activation|";
  else if (!strcmp(kind, "repeat"))
    allowed = "|name||inputs||type||count||body|";
  else if (!strcmp(kind, "reshape"))
    allowed = "|name||inputs||type||shape|";
  else if (!strcmp(kind, "permute"))
    allowed = "|name||inputs||type||axes|";
  else if (!strcmp(kind, "cast"))
    allowed = "|name||inputs||type||dtype|";
  else if (!strcmp(kind, "embedding"))
    allowed = "|name||inputs||type||vocab_size||embed_dim|";
  else if (!strcmp(kind, "layernorm") || !strcmp(kind, "rmsnorm"))
    allowed = "|name||inputs||type||eps||affine|";
  else if (!strcmp(kind, "rope"))
    allowed = "|name||inputs||type||theta|";
  else if (!strcmp(kind, "attention"))
    allowed = "|name||inputs||type||is_causal||enable_gqa|";
  else if (!strcmp(kind, "const_like"))
    allowed = "|name||inputs||type||value|";
  else if (!strcmp(kind, "sum") || !strcmp(kind, "mean") || !strcmp(kind, "max"))
    allowed = "|name||inputs||type||axes||keepdim|";
  if (!fields(d, spec, allowed, path)) return NULL;
  bool binary = !strcmp(kind, "add") || !strcmp(kind, "sub") || !strcmp(kind, "mul") ||
                !strcmp(kind, "div") || !strcmp(kind, "maximum") || !strcmp(kind, "lt") ||
                !strcmp(kind, "eq");
  bool attention = !strcmp(kind, "attention");
  bool gradient = !strcmp(kind, "gradient"), matmul = !strcmp(kind, "matmul");
  int arity = !strcmp(kind, "where") ? 3 : binary || gradient || matmul ? 2 : 1;
  if (attention ? (n != 3 && n != 4) : n != arity) {
    if (attention) {
      def_error(d, path, "expected query, key, value and optional mask inputs");
      return NULL;
    }
    def_error(d, path, "expected %d inputs, received %d", arity, n);
    return NULL;
  }
  PolyTensor *x = inputs[0], *out = NULL;
  int64_t dims[8];
  int rank;
  if (!tensor_shape(d, x, dims, &rank, path)) return NULL;
  PolyUOp *xu = poly_tensor_uop_physical(x);
  bool linear = !strcmp(kind, "linear"), embedding = !strcmp(kind, "embedding");
  bool norm = !strcmp(kind, "layernorm") || !strcmp(kind, "rmsnorm");
  bool rope = !strcmp(kind, "rope");
  if ((linear || norm) &&
      (rank < 1 || poly_uop_shape_dim(d->ctx, xu, rank - 1)->op != POLY_OP_CONST)) {
    def_error(d, path, "feature dimension must be fixed");
    return NULL;
  }
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
  if (linear || embedding || norm || rope) {
    DefLayer *layer = NULL;
    for (int i = 0; i < d->n_layers; i++)
      if (!strcmp(d->layers[i].name, path)) layer = &d->layers[i];
    if (!linear) {
      int64_t in = rank ? dims[rank - 1] : 0, width = in;
      double eps = !strcmp(kind, "rmsnorm") ? 1e-6 : 1e-5, theta = 10000;
      bool use_bias = !strcmp(kind, "layernorm"), affine = true;
      if (embedding) {
        if (!poly_dtype_is_int(xu->dtype)) {
          def_error(d, path, "embedding requires integer indices");
          return NULL;
        }
        if (!integer(d, field(spec, "vocab_size"), 1, DEF_ELEMENTS, &in, path) ||
            !integer(d, field(spec, "embed_dim"), 1, DEF_ELEMENTS, &width, path))
          return NULL;
        /* Embedding's one-hot WHERE has the vocabulary axis before reduction. */
        int64_t work = in * width;
        for (int i = 0; i < rank; i++) {
          if (work > DEF_ELEMENTS / dims[i]) {
            def_error(d, path, "embedding exceeds element budget");
            return NULL;
          }
          work *= dims[i];
        }
      } else if (norm) {
        const cJSON *e = field(spec, "eps"), *a = field(spec, "affine");
        if ((e && (!cJSON_IsNumber(e) || !isfinite(e->valuedouble) || e->valuedouble <= 0)) ||
            (a && !cJSON_IsBool(a))) {
          def_error(d, path, "expected positive finite eps and boolean affine");
          return NULL;
        }
        if (e) eps = e->valuedouble;
        affine = !a || cJSON_IsTrue(a);
      } else {
        const cJSON *t = field(spec, "theta");
        if (rank != 4 || dims[3] % 2 || poly_uop_shape_dim(d->ctx, xu, 2)->op != POLY_OP_CONST ||
            poly_uop_shape_dim(d->ctx, xu, 3)->op != POLY_OP_CONST ||
            (t && (!cJSON_IsNumber(t) || !isfinite(t->valuedouble) || t->valuedouble < 1))) {
          def_error(
              d, path, "rope needs [batch,heads,fixed_sequence,even_head_dim] and theta >= 1"
          );
          return NULL;
        }
        in = dims[2];
        width = dims[3];
        if (t) theta = t->valuedouble;
      }
      if (layer && (layer->in_features != in || layer->out_features != width)) {
        def_error(d, path, "shared %s dimensions do not match the first call", kind);
        return NULL;
      }
      if (!layer && affine) {
        int64_t elements = norm ? width * (use_bias ? 2 : 1) : in * width;
        if (d->n_layers == DEF_NODES || !reserve_storage(d, elements, path)) return NULL;
        layer = &d->layers[d->n_layers++];
        snprintf(layer->name, sizeof(layer->name), "%s", path);
        layer->kind = kind;
        layer->in_features = in;
        layer->out_features = width;
        PolyTensor *w = NULL, *b = NULL;
        if (embedding)
          w = poly_model_embedding_parameters(d->model, path, (int)in, (int)width);
        else if (norm) {
          if (poly_model_norm_parameters(d->model, path, (int)width, use_bias, &w, &b)) return NULL;
        } else {
          char name[DEF_NAME];
          PolyModelRoPEConfig config = {
              .length = (int)in, .dim = (int)width, .theta = theta, .factor = 1};
          if (!path_join(d, name, path, "freqs_cos")) return NULL;
          w = poly_model_rope_frequencies(d->model, name, &config, false);
          if (!path_join(d, name, path, "freqs_sin")) {
            poly_tensor_release(w);
            return NULL;
          }
          b = poly_model_rope_frequencies(d->model, name, &config, true);
        }
        layer->weight = own(d, w, path);
        if (b) layer->bias = own(d, b, path);
        if (!layer->weight || ((use_bias || rope) && !layer->bias)) return NULL;
      }
      PolyTensor *w = layer ? layer->weight : NULL, *b = layer ? layer->bias : NULL;
      out = embedding                  ? poly_tensor_embedding_apply(d->ctx, x, w)
            : rope                     ? poly_tensor_rope(d->ctx, x, w, b)
            : !strcmp(kind, "rmsnorm") ? poly_tensor_rmsnorm_apply(d->ctx, x, w, eps)
                                       : poly_tensor_layernorm_apply(d->ctx, x, w, b, -1, eps);
      out = own(d, out, path);
      int64_t result[8];
      int result_rank;
      return out && tensor_shape(d, out, result, &result_rank, path) ? out : NULL;
    }
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
      if (d->n_layers == DEF_NODES || !reserve_storage(d, elements, path)) {
        def_error(d, path, "parameter budget exceeded");
        return NULL;
      }
      layer = &d->layers[d->n_layers++];
      layer->kind = kind;
      snprintf(layer->name, sizeof(layer->name), "%s", path);
      layer->in_features = fan_in;
      layer->out_features = width;
      PolyTensor *weight = NULL, *bias_tensor = NULL;
      if (poly_model_linear_parameters(
              d->model, path, (int)fan_in, (int)width, use_bias, &weight, &bias_tensor
          )) {
        def_error(d, path, "Linear parameter construction failed");
        return NULL;
      }
      layer->weight = own(d, weight, path);
      if (bias_tensor) layer->bias = own(d, bias_tensor, path);
      if (!layer->weight || (use_bias && !layer->bias)) return NULL;
    }
    out = own(d, poly_tensor_linear_apply(d->ctx, x, layer->weight, layer->bias), path);
    return out ? activation(d, out, act ? act->valuestring : "none", path) : NULL;
  }
  if (gradient) {
    /* Tensor.gradient's implicit seed is valid only for a floating scalar.
     * Differentiate both existing roots; never reconstruct physical execution
     * from the portable graph. This is construction, not a gradient executor. */
    PolyTensor *target = inputs[1];
    if (rank != 0 || !poly_dtype_is_float(xu->dtype) ||
        !poly_dtype_is_float(poly_tensor_uop_physical(target)->dtype)) {
      def_error(d, path, "gradient requires a floating scalar and floating target");
      return NULL;
    }
    out = poly_tensor_gradient(d->ctx, x, target);
  } else if (matmul) {
    out = poly_tensor_dot(d->ctx, x, inputs[1]);
  } else if (!strcmp(kind, "const_like")) {
    const cJSON *value = field(spec, "value");
    if (!cJSON_IsNumber(value) || !isfinite(value->valuedouble) ||
        !poly_dtype_is_float(xu->dtype)) {
      def_error(d, path, "const_like requires a finite number and floating input");
      return NULL;
    }
    out = poly_tensor_const_like_float(d->ctx, x, value->valuedouble);
  } else if (!strcmp(kind, "where")) {
    out = poly_tensor_alu3(d->ctx, POLY_OP_WHERE, x, inputs[1], inputs[2]);
  } else if (binary) {
    PolyUOp *sources[] = {xu, poly_tensor_uop_physical(inputs[1])}, *broadcast[8];
    if (poly_uop_broadcast_shape(d->ctx, sources, 2, broadcast, 8) < 0) {
      def_error(d, path, "incompatible broadcast dimensions");
      return NULL;
    }
    if (!strcmp(kind, "div"))
      out = poly_tensor_div(d->ctx, x, inputs[1], 0);
    else if (!strcmp(kind, "sub")) {
      PolyTensor *neg = own(d, poly_tensor_alu1(d->ctx, POLY_OP_NEG, inputs[1]), path);
      out = neg ? poly_tensor_alu2(d->ctx, POLY_OP_ADD, x, neg) : NULL;
    } else
      out = poly_tensor_alu2(
          d->ctx,
          !strcmp(kind, "add")       ? POLY_OP_ADD
          : !strcmp(kind, "maximum") ? POLY_OP_MAX
          : !strcmp(kind, "lt")      ? POLY_OP_CMPLT
          : !strcmp(kind, "eq")      ? POLY_OP_CMPEQ
                                     : POLY_OP_MUL,
          x, inputs[1]
      );
  } else if (!strcmp(kind, "sum") || !strcmp(kind, "mean") || !strcmp(kind, "max")) {
    int64_t axes[8];
    const cJSON *specified = field(spec, "axes"), *keep = field(spec, "keepdim");
    int na = specified ? cJSON_GetArraySize(specified) : rank;
    if ((specified && (!cJSON_IsArray(specified) || na < 1 || na > rank)) ||
        (keep && !cJSON_IsBool(keep))) {
      def_error(d, path, "expected nonempty axes and boolean keepdim");
      return NULL;
    }
    bool seen[8] = {0};
    for (int i = 0; i < na; i++) {
      axes[i] = i;
      if (specified &&
          !integer(d, cJSON_GetArrayItem(specified, i), -rank, rank - 1, &axes[i], path))
        return NULL;
      if (axes[i] < 0) axes[i] += rank;
      if (seen[axes[i]]) {
        def_error(d, path, "duplicate reduction axis");
        return NULL;
      }
      seen[axes[i]] = true;
    }
    out = !strcmp(kind, "sum")   ? poly_tensor_sum(d->ctx, x, axes, na, cJSON_IsTrue(keep))
          : !strcmp(kind, "max") ? poly_tensor_max(d->ctx, x, axes, na, cJSON_IsTrue(keep))
                                 : poly_tensor_mean(d->ctx, x, axes, na, cJSON_IsTrue(keep));
  } else if (!strcmp(kind, "reshape")) {
    int64_t maximum[8];
    PolyUOp *dest[8];
    int dr;
    if (!shape(d, field(spec, "shape"), dest, maximum, &dr, path)) return NULL;
    out = poly_tensor_reshape_uop(d->ctx, x, dest, dr);
    if (!out) def_error(d, path, "reshape changes symbolic element count");
  } else if (!strcmp(kind, "permute")) {
    const cJSON *axes = field(spec, "axes");
    int64_t perm[8];
    bool seen[8] = {0};
    if (!cJSON_IsArray(axes) || cJSON_GetArraySize(axes) != rank) {
      def_error(d, path, "permute needs one axis per dimension");
      return NULL;
    }
    for (int i = 0; i < rank; i++) {
      if (!integer(d, cJSON_GetArrayItem(axes, i), 0, rank - 1, &perm[i], path)) return NULL;
      if (seen[perm[i]]) {
        def_error(d, path, "duplicate permutation axis");
        return NULL;
      }
      seen[perm[i]] = true;
    }
    out = poly_tensor_permute(d->ctx, x, perm, rank);
  } else if (!strcmp(kind, "cast")) {
    PolyDType dt;
    if (!dtype(d, field(spec, "dtype"), &dt, path)) return NULL;
    out = poly_tensor_cast(d->ctx, x, dt);
  } else if (attention) {
    const cJSON *causal = field(spec, "is_causal"), *gqa = field(spec, "enable_gqa");
    if ((causal && !cJSON_IsBool(causal)) || (gqa && !cJSON_IsBool(gqa))) {
      def_error(d, path, "attention flags must be boolean");
      return NULL;
    }
    int64_t kdims[8];
    int krank;
    if (!tensor_shape(d, inputs[1], kdims, &krank, path)) return NULL;
    if (rank < 2 || krank < 2) {
      def_error(d, path, "attention requires rank >= 2");
      return NULL;
    }
    /* Budget the attention scores, not just the smaller projected output. */
    int64_t work = dims[rank - 2] * kdims[krank - 2];
    int prefix = rank > krank ? rank - 2 : krank - 2;
    for (int i = 0; i < prefix; i++) {
      int qi = rank - 3 - i, ki = krank - 3 - i;
      int64_t qdim = qi >= 0 ? dims[qi] : 1, kdim = ki >= 0 ? kdims[ki] : 1;
      int64_t dim = qdim > kdim ? qdim : kdim;
      if (work > DEF_ELEMENTS / dim) {
        def_error(d, path, "attention exceeds element budget");
        return NULL;
      }
      work *= dim;
    }
    if (work > DEF_ELEMENTS) {
      def_error(d, path, "attention exceeds element budget");
      return NULL;
    }
    out = poly_tensor_sdpa(
        d->ctx, x, inputs[1], inputs[2], n == 4 ? inputs[3] : NULL, 0, cJSON_IsTrue(causal),
        cJSON_IsTrue(gqa), 0
    );
  } else if (!strcmp(kind, "identity"))
    return x;
  else if (!strcmp(kind, "none")) {
    def_error(d, path, "use identity for a no-op component");
    return NULL;
  } else if (!strcmp(kind, "square"))
    out = poly_tensor_alu2(d->ctx, POLY_OP_MUL, x, x);
  else if (!strcmp(kind, "neg"))
    out = poly_tensor_alu1(d->ctx, POLY_OP_NEG, x);
  else if (!strcmp(kind, "softplus"))
    out = poly_tensor_softplus(d->ctx, x, 1.0);
  else if (!strcmp(kind, "exp"))
    out = poly_tensor_exp(d->ctx, x);
  else if (!strcmp(kind, "log"))
    out = poly_tensor_log(d->ctx, x);
  else
    return activation(d, x, kind, path);
  out = own(d, out, path);
  int64_t result[8];
  int result_rank;
  return out && tensor_shape(d, out, result, &result_rank, path) ? out : NULL;
}

static bool input(Definition *d, const char *name, const cJSON *spec, bool sequential) {
  char path[DEF_NAME];
  if (!name_ok(d, name, "inputs") || !path_join(d, path, "inputs", name)) return false;
  if (!fields(d, spec, sequential ? "|name||shape||dtype|" : "|shape||dtype||role|", path))
    return false;
  const cJSON *role = field(spec, "role");
  PolyDType dt;
  if (!dtype(d, field(spec, "dtype"), &dt, path)) return false;
  bool target = role && cJSON_IsString(role) && !strcmp(role->valuestring, "target");
  if (role && !target && (!cJSON_IsString(role) || strcmp(role->valuestring, "input")))
    return def_error(d, path, "role must be input or target");
  int64_t dims[8];
  PolyUOp *symbolic[8];
  int ndim;
  if (!shape(d, field(spec, "shape"), symbolic, dims, &ndim, path)) return false;
  int64_t elements = 1;
  for (int i = 0; i < ndim; i++)
    elements *= dims[i];
  if (!reserve_storage(d, elements, path)) return false;
  PolyTensor *t =
      own(d,
          target ? poly_model_target_uop(d->model, name, dt, symbolic, ndim)
                 : poly_model_input_uop(d->model, name, dt, symbolic, ndim),
          path);
  if (!t) return false;
  if (poly_tensor_set_logical_policy(d->ctx, t, POLY_LOGICAL_ALWAYS))
    return def_error(d, path, "cannot preserve logical roots");
  return add_value(d, name, t, path);
}

static bool names(Definition *d, const cJSON *array, const char **out, int *n, const char *path) {
  if (!cJSON_IsArray(array) || (*n = cJSON_GetArraySize(array)) > DEF_IO)
    return def_error(d, path, "expected at most 64 names");
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

static bool declare_state_controls(Definition *d, const cJSON *root) {
  const cJSON *state = field(root, "state"), *controls = field(root, "controls");
  if (state && (!cJSON_IsObject(state) || cJSON_GetArraySize(state) > DEF_IO))
    return def_error(d, "state", "expected at most 64 named states");
  for (const cJSON *s = state ? state->child : NULL; s; s = s->next) {
    PolyDType dt;
    int rank;
    int64_t dims[8], size = 1;
    PolyUOp *sdims[8];
    const cJSON *sh = field(s, "shape");
    if (!fields(d, s, "|shape||dtype|", "state") || !dtype(d, field(s, "dtype"), &dt, "state"))
      return false;
    for (const cJSON *dim = sh ? sh->child : NULL; dim; dim = dim->next)
      if (!cJSON_IsNumber(dim)) return def_error(d, "state", "state shapes must be fixed");
    if (!shape(d, sh, sdims, dims, &rank, "state")) return false;
    for (int i = 0; i < rank; i++)
      size *= dims[i];
    if (!reserve_storage(d, size, "state")) return false;
    PolyTensor *t =
        own(d, poly_tensor_empty(d->ctx, dt, dims, rank, poly_ctx_get_preferred_device(d->ctx)),
            "state");
    if (!t || !add_value(d, s->string, t, "state") || poly_model_aux(d->model, s->string, t, 0))
      return false;
    d->state_names[d->n_states] = s->string;
    d->states[d->n_states++] = t;
  }
  if (controls && (!cJSON_IsObject(controls) || cJSON_GetArraySize(controls) > DEF_IO))
    return def_error(d, "controls", "expected at most 64 integer controls");
  for (const cJSON *c = controls ? controls->child : NULL; c; c = c->next) {
    int64_t lo, hi;
    if (!fields(d, c, "|min||max|", "controls") ||
        !integer(d, field(c, "min"), INT32_MIN, INT32_MAX, &lo, "controls") ||
        !integer(d, field(c, "max"), lo, INT32_MAX, &hi, "controls"))
      return false;
    PolyUOp *u = poly_uop_variable(
        d->ctx, c->string, poly_arg_int(lo), poly_arg_int(hi), POLY_WEAKINT, 1, false
    );
    PolyTensor *t =
        u ? own(d,
                poly_tensor_create_with_roots(
                    d->ctx, u, u, POLY_TENSOR_VALUE, poly_ctx_get_preferred_device(d->ctx)
                ),
                "controls")
          : NULL;
    if (!t || !add_value(d, c->string, t, "controls")) return false;
    d->control_names[d->n_controls] = c->string;
    d->controls[d->n_controls++] = u;
  }
  return true;
}

/* Each writing entrypoint owns its output bindings. Capture restores authoring
 * roots afterward, so effects cannot leak into another entrypoint's graph. */
static bool entry_outputs(
    Definition *d,
    const cJSON *definitions,
    const cJSON *writes,
    const char **outs,
    int no,
    int *published
) {
  if (!no) return def_error(d, "entrypoints", "at least one output is required");
  PolyTensor *values[DEF_IO], *wrapped[DEF_IO] = {0};
  int indices[DEF_IO];
  bool effects = writes && writes->child;
  if (writes && (!cJSON_IsObject(writes) || cJSON_GetArraySize(writes) > DEF_IO))
    return def_error(d, "writes", "expected at most 64 state assignments");
  for (int i = 0; i < no; i++) {
    const cJSON *ref = field(definitions, outs[i]);
    int index = 0;
    for (const cJSON *v = definitions->child; v && v != ref; v = v->next)
      index++;
    if (!ref || !(values[i] = lookup(d, ref, "outputs"))) return false;
    indices[i] = index;
    if (published[index] && (effects || published[index] == 2))
      return def_error(d, "outputs", "writing entrypoints require distinct output names");
  }
  PolyTensorCapture *capture = effects ? poly_tensor_capture_begin(d->ctx) : NULL;
  if (effects && !capture) return false;
  bool ok = false;
  int mutable[DEF_IO] = {0};
  /* Resolve all RHS handles before assigning. Tensor capture supplies the same
   * dependency/anti-dependency semantics as an ordinary captured Tensor program. */
  PolyTensor *rhs[DEF_IO] = {0};
  for (const cJSON *w = writes ? writes->child : NULL; w; w = w->next) {
    int j = 0;
    while (j < d->n_states && strcmp(d->state_names[j], w->string))
      j++;
    if (j == d->n_states) {
      def_error(d, "writes", "unknown state '%s'", w->string);
      goto done;
    }
    rhs[j] = lookup(d, w, "writes");
    int ar, br;
    int64_t a[8], b[8];
    if (!rhs[j] || !tensor_shape(d, d->states[j], a, &ar, "writes") ||
        !tensor_shape(d, rhs[j], b, &br, "writes"))
      goto done;
    if (ar != br || memcmp(a, b, (size_t)ar * sizeof(int64_t)) ||
        !poly_dtype_eq(d->states[j]->uop_physical->dtype, rhs[j]->uop_physical->dtype)) {
      def_error(d, "writes", "state assignment requires identical shape and dtype");
      goto done;
    }
    /* Writes are simultaneous. Force a scheduler-owned materialization even
     * for direct state views: Tensor.contiguous() can be an identity there,
     * while Tensor.clone() introduces an explicit unbound storage allocation.
     * CONTIGUOUS leaves scratch allocation/lifetime to the ordinary scheduler. */
    PolyUOp *l = poly_uop1(
        d->ctx, POLY_OP_CONTIGUOUS, rhs[j]->uop_logical->dtype, rhs[j]->uop_logical, poly_arg_none()
    );
    PolyUOp *p = poly_uop1(
        d->ctx, POLY_OP_CONTIGUOUS, rhs[j]->uop_physical->dtype, rhs[j]->uop_physical,
        poly_arg_none()
    );
    rhs[j] =
        own(d,
            poly_tensor_create_result(
                d->ctx, &rhs[j], 1, l, p, POLY_TENSOR_VALUE, poly_tensor_device(rhs[j])
            ),
            "writes");
    if (!rhs[j]) goto done;
    mutable[j] = 1;
  }
  for (int j = 0; j < d->n_states; j++)
    if (mutable[j]) {
      /* assign returns the already-owned target, not a new Tensor handle. */
      if (!poly_tensor_assign(d->ctx, d->states[j], rhs[j])) {
        def_error(d, "writes", "assignment construction failed");
        goto done;
      }
    }
  if (effects &&
      poly_tensor_capture_wrap(capture, d->states, mutable, d->n_states, values, no, wrapped))
    goto done;
  for (int i = 0; i < no; i++) {
    PolyTensor *out = effects ? own(d, wrapped[i], "outputs") : values[i];
    wrapped[i] = NULL;
    if (!out || (!published[indices[i]] && !publish_output(d, outs[i], out))) goto done;
    published[indices[i]] = effects ? 2 : 1;
  }
  ok = true;
done:
  for (int i = 0; i < no; i++)
    if (wrapped[i]) poly_tensor_release(wrapped[i]);
  if (capture) poly_tensor_capture_end(capture);
  return ok;
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
              : "|format||type||seed||inputs||nodes||outputs||entrypoints||modules||state||"
                "controls|",
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
    if (!cJSON_IsObject(inputs) || cJSON_GetArraySize(inputs) > DEF_IO)
      return def_error(d, "inputs", "expected at most 64 named input schemas");
    for (const cJSON *v = inputs->child; v; v = v->next) {
      if (!input(d, v->string, v, false)) return false;
      ins[ni++] = v->string;
    }
    if (!declare_state_controls(d, root)) return false;
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
      if (!cJSON_IsArray(refs) || cJSON_GetArraySize(refs) > 4)
        return def_error(d, path, "expected at most 4 input references");
      PolyTensor *args[4];
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
      if (!y) return false;
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
  const cJSON *outputs = field(root, "outputs");
  int published[DEF_IO] = {0};
  if (!entries) {
    if (!seq && !entry_outputs(d, outputs, NULL, outs, no, published)) return false;
    return poly_model_entrypoint(d->model, "forward", ins, ni, outs, no, NULL) == POLY_STATUS_OK;
  }
  if (!cJSON_IsArray(entries) || !entries->child || cJSON_GetArraySize(entries) > DEF_IO)
    return def_error(d, "entrypoints", "expected 1..64 entrypoints");
  for (const cJSON *ep = entries->child; ep; ep = ep->next) {
    if (!fields(d, ep, "|name||inputs||outputs||objective||writes||controls|", "entrypoints"))
      return false;
    const cJSON *name = field(ep, "name"), *objective = field(ep, "objective");
    if (!cJSON_IsString(name) || !name_ok(d, name->valuestring, "entrypoints")) return false;
    if (!names(d, field(ep, "inputs"), ins, &ni, name->valuestring) ||
        !names(d, field(ep, "outputs"), outs, &no, name->valuestring))
      return false;
    if (objective && !cJSON_IsString(objective))
      return def_error(d, name->valuestring, "objective must name an output");
    if (!entry_outputs(d, outputs, field(ep, "writes"), outs, no, published)) return false;
    PolyEntrypointOptions opts = {.objective = objective ? objective->valuestring : NULL};
    if (poly_model_entrypoint(d->model, name->valuestring, ins, ni, outs, no, &opts)) return false;
    const cJSON *controls = field(ep, "controls");
    if (controls) {
      const char *cn[DEF_IO];
      int nc;
      if (!names(d, controls, cn, &nc, "controls")) return false;
      for (int i = 0; i < nc; i++) {
        int j = 0;
        while (j < d->n_controls && strcmp(cn[i], d->control_names[j]))
          j++;
        if (j == d->n_controls) return def_error(d, "controls", "unknown control '%s'", cn[i]);
        if (poly_model_control(d->model, name->valuestring, cn[i], d->controls[j])) return false;
      }
    }
  }
  int i = 0;
  for (const cJSON *v = outputs->child; v; v = v->next, i++)
    if (!published[i] && !publish_output(d, v->string, lookup(d, v, "outputs"))) return false;
  return true;
}

static PolyModel *compose_build(
    PolyCtx *ctx,
    const cJSON *root,
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
  bool ok = false;
  if (!ctx) {
    def_error(d, "$", "a live borrowed context is required");
    goto done;
  }
  if (poly_ctx_get_logical_policy(ctx) == POLY_LOGICAL_NEVER) {
    def_error(d, "$", "portable Model requires logical construction");
    goto done;
  }
  d->model = poly_model_new(ctx, NULL);
  if (!d->model) {
    def_error(d, "$", "Model allocation failed");
    goto done;
  }
  if (!construct(d, root)) goto done;
  if (poly_model_build(d->model, err)) goto done;
  /* Initialize through the coherent buffer API after sealing. These bytes are
   * model state, never host-pointer aliases or a second residency table. */
  for (int i = 0; i < d->n_states; i++) {
    int index = poly_model_find_buf(d->model, d->state_names[i]);
    size_t bytes = poly_model_buf_nbytes(d->model, index);
    void *zero = calloc(1, bytes);
    int rc = zero ? poly_model_write_buf_named(d->model, d->state_names[i], zero, bytes) : -1;
    free(zero);
    if (rc) {
      def_error(d, "state", "zero initialization failed");
      goto done;
    }
  }
  for (int i = 0; i < d->n_layers; i++) {
    DefLayer *layer = &d->layers[i];
    if (!strcmp(layer->kind, "rope")) continue; /* AUX tables were captured before sealing. */
    bool norm = !strcmp(layer->kind, "layernorm") || !strcmp(layer->kind, "rmsnorm");
    int64_t fan_in = norm                                ? 0
                     : !strcmp(layer->kind, "embedding") ? layer->out_features
                                                         : layer->in_features;
    char name[DEF_NAME];
    if (!path_join(d, name, layer->name, "weight") ||
        model_init_param(d->model, name, d->seed, fan_in, norm ? 1 : 0)) {
      def_error(d, layer->name, "initializer write failed");
      goto done;
    }
    if (layer->bias &&
        (!path_join(d, name, layer->name, "bias") || model_init_param(d->model, name, 0, 0, 0))) {
      def_error(d, layer->name, "bias initializer write failed");
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
  free(d);
  return result;
}

PolyModel *model_sequential_from_config(PolyCtx *ctx, const cJSON *root, PolyModelError *err) {
  return compose_build(ctx, root, err, "sequential");
}

PolyModel *model_graph_from_config(PolyCtx *ctx, const cJSON *root, PolyModelError *err) {
  return compose_build(ctx, root, err, "graph");
}
