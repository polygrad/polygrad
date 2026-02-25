#include "modelzoo.h"
#include <string.h>
#include <stdlib.h>

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

// Stubs for config management (would likely parse JSON manifest configs)
PolyModelConfig* poly_model_config_new() { return NULL; }
void poly_model_config_set_int(PolyModelConfig* config, const char* key, int value) {}
void poly_model_config_set_float(PolyModelConfig* config, const char* key, float value) {}
void poly_model_config_free(PolyModelConfig* config) {}
