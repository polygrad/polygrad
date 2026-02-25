#include "../modelzoo.h"
#include "../../nn.h"
#include <stdlib.h>
#include <string.h>

// ----------------------------------------------------------------------------
// Vision Transformer (ViT) Configuration & Architecture
// ----------------------------------------------------------------------------

typedef struct {
    int image_size;
    int patch_size;
    int num_classes;
    int dim;
    int depth;
    int heads;
    int mlp_dim;
} ViTConfig;

typedef struct {
    PolyModel base;
    ViTConfig cfg;
    
    // Patch Embedding
    PolyUOp* proj_weight;
    PolyUOp* proj_bias;
    PolyUOp* cls_token;
    PolyUOp* pos_embed;
    
    // Transformer Blocks...
    PolyUOp* norm_weight;
    PolyUOp* norm_bias;
    
    // classification head
    PolyUOp* head_weight;
    PolyUOp* head_bias;
} PolyViT;


PolyUOp** vit_forward(PolyModel* model, PolyUOp** inputs, int num_inputs, int* out_num_outputs) {
    PolyViT* vit = (PolyViT*)model;
    
    // inputs[0] should be pixel_values (B, C, H, W)
    PolyUOp* pixel_values = inputs[0];

    // Compute graph: Patch extraction (Conv2D), flattening, cls_token prepending,
    // positional embedding addition, transformer blocks (MHA + MLP), layer norm, MLP head.
    
    PolyUOp** outputs = (PolyUOp**)malloc(sizeof(PolyUOp*) * 1);
    outputs[0] = pixel_values; // STUB
    *out_num_outputs = 1;

    return outputs;
}

PolyUOp* vit_get_parameter(PolyModel* model, const char* name) {
    PolyViT* vit = (PolyViT*)model;
    if (strcmp(name, "embeddings.patch_embeddings.projection.weight") == 0) return vit->proj_weight;
    if (strcmp(name, "embeddings.patch_embeddings.projection.bias") == 0) return vit->proj_bias;
    if (strcmp(name, "embeddings.cls_token") == 0) return vit->cls_token;
    if (strcmp(name, "embeddings.position_embeddings") == 0) return vit->pos_embed;
    if (strcmp(name, "layernorm.weight") == 0) return vit->norm_weight;
    if (strcmp(name, "layernorm.bias") == 0) return vit->norm_bias;
    if (strcmp(name, "classifier.weight") == 0) return vit->head_weight;
    if (strcmp(name, "classifier.bias") == 0) return vit->head_bias;
    return NULL;
}

void vit_free(PolyModel* model) {}

PolyModel* create_vit_model(PolyCtx* ctx, PolyModelConfig* config) {
    PolyViT* model = (PolyViT*)malloc(sizeof(PolyViT));
    memset(model, 0, sizeof(PolyViT));
    
    model->cfg.image_size = 224;
    model->cfg.patch_size = 16;
    model->cfg.num_classes = 1000;
    model->cfg.dim = 768;
    model->cfg.depth = 12;
    model->cfg.heads = 12;
    model->cfg.mlp_dim = 3072;
    
    model->base.forward = vit_forward;
    model->base.get_parameter = vit_get_parameter;
    model->base.free = vit_free;
    
    return (PolyModel*)model;
}

void __attribute__((constructor)) register_vit() {
    poly_model_register("vit", create_vit_model);
}
