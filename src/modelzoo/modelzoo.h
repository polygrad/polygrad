#ifndef POLYGRAD_MODELZOO_H
#define POLYGRAD_MODELZOO_H

#include "../polygrad.h"

// Opaque struct representing a model configuration.
typedef struct PolyModelConfig PolyModelConfig;

// Create a configuration object from a JSON string or typical param list.
PolyModelConfig* poly_model_config_new();
void poly_model_config_set_int(PolyModelConfig* config, const char* key, int value);
void poly_model_config_set_float(PolyModelConfig* config, const char* key, float value);
void poly_model_config_free(PolyModelConfig* config);


// Instantiated model graph with virtual dispatch
typedef struct PolyModel PolyModel;

// Model operations vtable
struct PolyModel {
    PolyCtx* ctx;
    const char* kind;
    // Forward pass: returns an array of output UOps
    PolyUOp** (*forward)(PolyModel* model, PolyUOp** inputs, int num_inputs, int* out_num_outputs);
    // Fetch parameter UOp by string name format (e.g. "blocks.0.attn.c_attn.weight")
    PolyUOp* (*get_parameter)(PolyModel* model, const char* name);
    // Cleanup internal memory tracking 
    void (*free)(PolyModel* model);
};

// Orchestration
PolyModel* poly_model_create(PolyCtx* ctx, const char* kind, PolyModelConfig* config);
PolyUOp* poly_model_parameter(PolyModel* model, const char* name);
PolyUOp** poly_model_forward(PolyModel* model, PolyUOp** inputs, int num_inputs, int* out_num_outputs);
void poly_model_free(PolyModel* model);

// Registry mechanism for models
typedef PolyModel* (*PolyModelCreateFn)(PolyCtx* ctx, PolyModelConfig* config);
void poly_model_register(const char* kind, PolyModelCreateFn create_fn);

#endif // POLYGRAD_MODELZOO_H
