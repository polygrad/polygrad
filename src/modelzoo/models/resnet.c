#include "../modelzoo.h"
#include "../../nn.h"
#include <stdlib.h>
#include <string.h>

// ----------------------------------------------------------------------------
// ResNet Configuration & Architecture
// ----------------------------------------------------------------------------

typedef struct {
    int num_classes;
    int block_depths[4]; // e.g. {3, 4, 6, 3} for ResNet-50
} ResNetConfig;

typedef struct {
    PolyModel base;
    ResNetConfig cfg;
    
    // Conv1
    PolyUOp* conv1_weight;
    PolyUOp* bn1_weight;
    PolyUOp* bn1_bias;
    
    // Layers 1-4
    // (In reality this requires dynamic array tracking based on depth configuration)
    
    // FC Head
    PolyUOp* fc_weight;
    PolyUOp* fc_bias;
} PolyResNet;


PolyUOp** resnet_forward(PolyModel* model, PolyUOp** inputs, int num_inputs, int* out_num_outputs) {
    PolyResNet* resnet = (PolyResNet*)model;
    
    // inputs[0] should be pixel_values (B, C, H, W)
    PolyUOp* pixel_values = inputs[0];

    // Compute graph: Initial Conv2D/BatchNorm/ReLU/MaxPool, followed by
    // 4 bottleneck/basic block layers, global average pooling, and fully connected classifying head.
    
    PolyUOp** outputs = (PolyUOp**)malloc(sizeof(PolyUOp*) * 1);
    outputs[0] = pixel_values; // STUB
    *out_num_outputs = 1;

    return outputs;
}

PolyUOp* resnet_get_parameter(PolyModel* model, const char* name) {
    PolyResNet* resnet = (PolyResNet*)model;
    if (strcmp(name, "conv1.weight") == 0) return resnet->conv1_weight;
    if (strcmp(name, "bn1.weight") == 0) return resnet->bn1_weight;
    if (strcmp(name, "bn1.bias") == 0) return resnet->bn1_bias;
    if (strcmp(name, "fc.weight") == 0) return resnet->fc_weight;
    if (strcmp(name, "fc.bias") == 0) return resnet->fc_bias;
    return NULL;
}

void resnet_free(PolyModel* model) {}

PolyModel* create_resnet_model(PolyCtx* ctx, PolyModelConfig* config) {
    PolyResNet* model = (PolyResNet*)malloc(sizeof(PolyResNet));
    memset(model, 0, sizeof(PolyResNet));
    
    model->cfg.num_classes = 1000;
    model->cfg.block_depths[0] = 3;
    model->cfg.block_depths[1] = 4;
    model->cfg.block_depths[2] = 6;
    model->cfg.block_depths[3] = 3;
    
    model->base.forward = resnet_forward;
    model->base.get_parameter = resnet_get_parameter;
    model->base.free = resnet_free;
    
    return (PolyModel*)model;
}

void __attribute__((constructor)) register_resnet() {
    poly_model_register("resnet", create_resnet_model);
}
