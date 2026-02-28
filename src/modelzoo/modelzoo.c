#define _POSIX_C_SOURCE 200809L
#include "modelzoo.h"
#include "../../vendor/cjson/cJSON.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* ── PolyModelConfig (cJSON wrapper) ──────────────────────────────── */

struct PolyModelConfig {
  cJSON *root;  /* owned */
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

const char *poly_model_config_get_string(const PolyModelConfig *cfg, const char *key, const char *default_val) {
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

/* ── Registry ─────────────────────────────────────────────────────── */

// Simple linked list registry for compiling C-level model builders
typedef struct RegistryEntry {
    char kind[64];
    PolyModelCreateFn create_fn;
    struct RegistryEntry* next;
} RegistryEntry;

static RegistryEntry* g_registry = NULL;

void poly_model_register(const char* kind, PolyModelCreateFn create_fn) {
    RegistryEntry* entry = (RegistryEntry*)malloc(sizeof(RegistryEntry));
    strncpy(entry->kind, kind, 63);
    entry->kind[63] = '\0';
    entry->create_fn = create_fn;
    entry->next = g_registry;
    g_registry = entry;
}

PolyModel* poly_model_create(PolyCtx* ctx, const char* kind, PolyModelConfig* config) {
    RegistryEntry* curr = g_registry;
    while (curr) {
        if (strcmp(curr->kind, kind) == 0) {
            PolyModel* model = curr->create_fn(ctx, config);
            if (model) {
                model->ctx = ctx;
                model->kind = kind;
            }
            return model;
        }
        curr = curr->next;
    }
    return NULL; // Model kind not found in registry
}

PolyUOp* poly_model_parameter(PolyModel* model, const char* name) {
    if (model && model->get_parameter) {
        return model->get_parameter(model, name);
    }
    return NULL;
}

PolyUOp** poly_model_forward(PolyModel* model, PolyUOp** inputs, int num_inputs, int* out_num_outputs) {
    if (model && model->forward) {
        return model->forward(model, inputs, num_inputs, out_num_outputs);
    }
    *out_num_outputs = 0;
    return NULL;
}

void poly_model_free(PolyModel* model) {
    if (model) {
        if (model->free) {
            model->free(model);
        }
        free(model);
    }
}

