#define _POSIX_C_SOURCE 200809L
#include "hf_loader.h"
#include "../../vendor/cjson/cJSON.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

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
