#define _POSIX_C_SOURCE 200809L
#include "hf_loader.h"
#include "models.h"
#include "factory.h"
#include "registry.h"
#include "../../vendor/cjson/cJSON.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <math.h>
#include <limits.h>
#include <stdarg.h>

bool model_factory_error(PolyModelError *err, const char *path, const char *fmt, ...) {
  if (err && !err->code) {
    err->code = POLY_STATUS_INVALID;
    err->func = "poly_model_from_config";
    int n = snprintf(err->message, sizeof(err->message), "%s: ", path);
    if (n < 0 || n >= (int)sizeof(err->message)) return false;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err->message + n, sizeof(err->message) - (size_t)n, fmt, ap);
    va_end(ap);
  }
  return false;
}

/* Reject ambiguous duplicate keys at every level, including unused declarations.
 * The lexical preflight bounds recursion before cJSON itself parses the input. */
static bool unique_keys(PolyModelError *err, const cJSON *obj, int *budget) {
  if (--*budget < 0) return model_factory_error(err, "$", "JSON node budget exceeded");
  for (const cJSON *a = obj->child; a; a = a->next) {
    if (cJSON_IsObject(obj))
      for (const cJSON *b = obj->child; b != a; b = b->next)
        if (!strcmp(a->string, b->string))
          return model_factory_error(err, "$", "duplicate key '%s'", a->string);
    if (!unique_keys(err, a, budget)) return false;
  }
  return true;
}

static bool json_preflight(PolyModelError *err, const char *json, int len) {
  if (!json || len <= 0 || len > 1024 * 1024)
    return model_factory_error(err, "$", "expected 1..1048576 JSON bytes");
  int depth = 0;
  bool quoted = false;
  for (int i = 0; i < len; i++) {
    unsigned char c = (unsigned char)json[i];
    if (!c) return model_factory_error(err, "$", "embedded NUL");
    if (quoted && c == '\\') {
      if (i + 5 < len && !memcmp(json + i, "\\u0000", 6))
        return model_factory_error(err, "$", "escaped NUL");
      i++;
    } else if (c == '"')
      quoted = !quoted;
    else if (quoted && c < 32)
      return model_factory_error(err, "$", "control character in string");
    else if (!quoted && (c == '-' || (c >= '0' && c <= '9'))) {
      /* cJSON's strtod parser accepts 01 and 1.; require JSON number grammar
       * before schema integer/range validation, without another numeric evaluator. */
      int p = i;
      if (json[p] == '-') p++;
      if (p == len || json[p] < '0' || json[p] > '9')
        return model_factory_error(err, "$", "invalid JSON number");
      if (json[p] == '0')
        p++;
      else
        while (p < len && json[p] >= '0' && json[p] <= '9')
          p++;
      if (p < len && json[p] >= '0' && json[p] <= '9')
        return model_factory_error(err, "$", "invalid JSON number");
      if (p < len && json[p] == '.') {
        int start = ++p;
        while (p < len && json[p] >= '0' && json[p] <= '9')
          p++;
        if (p == start) return model_factory_error(err, "$", "invalid JSON number");
      }
      if (p < len && (json[p] == 'e' || json[p] == 'E')) {
        p++;
        if (p < len && (json[p] == '+' || json[p] == '-')) p++;
        int start = p;
        while (p < len && json[p] >= '0' && json[p] <= '9')
          p++;
        if (p == start) return model_factory_error(err, "$", "invalid JSON number");
      }
      i = p - 1;
    } else if (!quoted && (c == '{' || c == '[')) {
      if (++depth > 32) return model_factory_error(err, "$", "JSON nesting exceeds 32");
    } else if (!quoted && (c == '}' || c == ']'))
      depth--;
  }
  return true;
}

cJSON *model_factory_parse(const char *json, int len, PolyModelError *err) {
  if (!json_preflight(err, json, len)) return NULL;
  const char *end = NULL;
  cJSON *root = cJSON_ParseWithLengthOpts(json, (size_t)len, &end, false);
  if (!root) {
    model_factory_error(err, "$", "invalid JSON near byte %ld", end ? (long)(end - json) : 0L);
    return NULL;
  }
  while (end < json + len && isspace((unsigned char)*end))
    end++;
  int budget = 16384;
  if (end != json + len)
    model_factory_error(err, "$", "trailing JSON data");
  else if (!cJSON_IsObject(root))
    model_factory_error(err, "$", "expected object");
  else if (unique_keys(err, root, &budget))
    return root;
  cJSON_Delete(root);
  return NULL;
}

bool model_config_integer(
    const cJSON *root,
    const char *key,
    int64_t lo,
    int64_t hi,
    bool required,
    PolyModelError *err
) {
  const cJSON *v = cJSON_GetObjectItemCaseSensitive(root, key);
  if (!v && !required) return true;
  if (!cJSON_IsNumber(v) || !isfinite(v->valuedouble) || v->valuedouble < (double)lo ||
      v->valuedouble > (double)hi || trunc(v->valuedouble) != v->valuedouble)
    return model_factory_error(
        err, key, "expected integer in [%lld,%lld]", (long long)lo, (long long)hi
    );
  return true;
}

bool model_config_sizes(
    const cJSON *root,
    const char *key,
    int min,
    int max,
    bool required,
    PolyModelError *err
) {
  const cJSON *v = cJSON_GetObjectItemCaseSensitive(root, key);
  if (!v && !required) return true;
  if (!cJSON_IsArray(v) || cJSON_GetArraySize(v) < min || cJSON_GetArraySize(v) > max)
    return model_factory_error(err, key, "expected %d..%d positive dimensions", min, max);
  for (const cJSON *d = v->child; d; d = d->next)
    if (!cJSON_IsNumber(d) || !isfinite(d->valuedouble) || d->valuedouble < 1 ||
        d->valuedouble > INT_MAX || trunc(d->valuedouble) != d->valuedouble)
      return model_factory_error(err, key, "expected positive integer dimensions");
  return true;
}

bool model_config_choice(
    const cJSON *root,
    const char *key,
    const char *choices,
    PolyModelError *err
) {
  const cJSON *v = cJSON_GetObjectItemCaseSensitive(root, key);
  if (!v) return true;
  char needle[64];
  if (!cJSON_IsString(v) || !v->valuestring[0] || strchr(v->valuestring, '|') ||
      snprintf(needle, sizeof(needle), "|%s|", v->valuestring) >= (int)sizeof(needle) ||
      !strstr(choices, needle))
    return model_factory_error(err, key, "expected one of %s", choices);
  return true;
}

bool model_config_training(const cJSON *root, PolyModelError *err) {
  return model_config_integer(root, "batch_size", 1, INT_MAX, false, err) &&
         model_config_integer(root, "seed", 0, INT64_C(9007199254740991), false, err) &&
         model_config_choice(root, "loss", "|none|mse|cross_entropy|", err);
}

static const PolyModelType model_types[] = {
    {"MLP", "mlp", model_mlp_build, NULL, NULL},
    {"TabM", "tabm", model_tabm_build, NULL, NULL},
    {"NAM", "nam", model_nam_build, NULL, NULL},
    {"GPT2", "gpt2", model_gpt2_build, poly_gpt2_from_hf_decoded_generic,
     poly_gpt2_from_gguf_decoded_generic},
    {"Sequential", "sequential", model_sequential_build, NULL, NULL},
    {"Graph", "graph", model_graph_build, NULL, NULL},
    {"Llama", "llama", model_llama_build, poly_llama_from_hf_decoded_generic, NULL},
    {"Qwen3", "qwen3", NULL, NULL, poly_qwen3_from_gguf_decoded_generic},
};

const char *poly_model_type_name(int index) {
  return index >= 0 && (size_t)index < sizeof(model_types) / sizeof(*model_types)
             ? model_types[index].name
             : NULL;
}

int poly_model_type_capabilities(int index) {
  if (!poly_model_type_name(index)) return 0;
  const PolyModelType *type = &model_types[index];
  return (type->build ? POLY_MODEL_CONSTRUCTIBLE : 0) |
         (type->from_hf_decoded ? POLY_MODEL_HF : 0) |
         (type->from_gguf_decoded ? POLY_MODEL_GGUF : 0);
}

const PolyModelType *model_type_find(const char *name) {
  if (!name) return NULL;
  for (size_t i = 0; i < sizeof(model_types) / sizeof(*model_types); i++)
    if (!strcmp(name, model_types[i].name) || !strcmp(name, model_types[i].tag))
      return &model_types[i];
  return NULL;
}

PolyModel *poly_model_from_config(
    PolyCtx *ctx,
    const char *family,
    const char *json,
    int len,
    PolyDevice device,
    PolyModelError *err
) {
  PolyModelError local = {0};
  if (!err) err = &local;
  memset(err, 0, sizeof(*err));
  cJSON *root = model_factory_parse(json, len, err);
  if (!root) return NULL;
  const cJSON *type = cJSON_GetObjectItemCaseSensitive(root, "type");
  const cJSON *format = cJSON_GetObjectItemCaseSensitive(root, "format");
  PolyModel *model = NULL;
  if ((!family && (!cJSON_IsString(type) || !cJSON_IsString(format))) ||
      (type && !cJSON_IsString(type)) ||
      (format && (!cJSON_IsString(format) || strcmp(format->valuestring, "poly.modeldef@1")))) {
    model_factory_error(err, "$", "expected format poly.modeldef@1 and a registered type");
    goto done;
  }
  const char *selected = family ? family : type->valuestring;
  const PolyModelType *desc = model_type_find(selected);
  if (!desc) {
    model_factory_error(err, "type", "unknown model type '%s'", selected);
    goto done;
  }
  if (type && strcmp(type->valuestring, desc->tag)) {
    model_factory_error(err, "type", "expected '%s'", desc->tag);
    goto done;
  }
  if (!desc->build) {
    model_factory_error(
        err, "type",
        "%s supports checkpoint import only; configuration construction is unavailable", desc->name
    );
    goto done;
  }
  PolyModelFactoryScope scope;
  if (!model_factory_begin(&scope, ctx, device)) {
    model_factory_error(err, "$", "construction requires an idle runtime and executable device");
    goto done;
  }
  model = desc->build(scope.ctx, root, err);
  if (!model && !err->code) model_factory_error(err, desc->name, "construction failed");
  model = model_factory_end(&scope, model);
done:
  cJSON_Delete(root);
  return model;
}

/* PolyModelConfig (cJSON wrapper) */

struct PolyModelConfig {
  cJSON *root; /* owned */
};

PolyModelConfig *poly_model_config_new(void) {
  PolyModelConfig *cfg = calloc(1, sizeof(PolyModelConfig));
  cfg->root = cJSON_CreateObject();
  return cfg;
}

PolyModelConfig *poly_model_config_from_json(const char *json, int len) {
  if (!json || len <= 0) return NULL;
  cJSON *root = cJSON_ParseWithLength(json, (size_t)len);
  if (!root) {
    fprintf(stderr, "poly_model_config_from_json: JSON parse error\n");
    return NULL;
  }
  PolyModelConfig *cfg = calloc(1, sizeof(PolyModelConfig));
  cfg->root = root;
  return cfg;
}

int poly_model_config_get_int(const PolyModelConfig *cfg, const char *key, int default_val) {
  if (!cfg || !cfg->root) return default_val;
  cJSON *item = cJSON_GetObjectItemCaseSensitive(cfg->root, key);
  if (!item || !cJSON_IsNumber(item)) return default_val;
  return item->valueint;
}

float poly_model_config_get_float(const PolyModelConfig *cfg, const char *key, float default_val) {
  if (!cfg || !cfg->root) return default_val;
  cJSON *item = cJSON_GetObjectItemCaseSensitive(cfg->root, key);
  if (!item || !cJSON_IsNumber(item)) return default_val;
  return (float)item->valuedouble;
}

const char *poly_model_config_get_string(
    const PolyModelConfig *cfg,
    const char *key,
    const char *default_val
) {
  if (!cfg || !cfg->root) return default_val;
  cJSON *item = cJSON_GetObjectItemCaseSensitive(cfg->root, key);
  if (!item || !cJSON_IsString(item)) return default_val;
  return item->valuestring;
}

void poly_model_config_set_int(PolyModelConfig *config, const char *key, int value) {
  if (!config || !config->root) return;
  cJSON_DeleteItemFromObject(config->root, key);
  cJSON_AddNumberToObject(config->root, key, (double)value);
}

void poly_model_config_set_float(PolyModelConfig *config, const char *key, float value) {
  if (!config || !config->root) return;
  cJSON_DeleteItemFromObject(config->root, key);
  cJSON_AddNumberToObject(config->root, key, (double)value);
}

void poly_model_config_free(PolyModelConfig *config) {
  if (!config) return;
  if (config->root) cJSON_Delete(config->root);
  free(config);
}
