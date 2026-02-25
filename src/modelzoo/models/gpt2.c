#include "../modelzoo.h"
#include "../../nn.h"
#include <stdlib.h>
#include <string.h>

// Specific C-level instantiation logic for GPT-2 UOp graphs
typedef struct {
    PolyModel base;
    
    // UOps tracking the model topology / params
    PolyUOp* wte_weight;
    PolyUOp* wpe_weight;
    
    // ...
} PolyGPT2;


// Forward pass: builds the lazy graph for logits dynamically based on inputs
PolyUOp** gpt2_forward(PolyModel* model, PolyUOp** inputs, int num_inputs, int* out_num_outputs) {
    PolyGPT2* gpt2 = (PolyGPT2*)model;
    
    // Construct the LLM forward pass using poly_alu, poly_dot, etc.
    // e.g., token embeddings + positional embeddings
    // This outputs UOps, not realized buffers.
    
    PolyUOp** outputs = (PolyUOp**)malloc(sizeof(PolyUOp*) * 1);
    // outputs[0] = final_logits_uop;
    *out_num_outputs = 1;

    return outputs;
}

PolyUOp* gpt2_get_parameter(PolyModel* model, const char* name) {
    PolyGPT2* gpt2 = (PolyGPT2*)model;
    // Simple string matching for now (in reality, a hash map or Trie)
    if (strcmp(name, "wte.weight") == 0) return gpt2->wte_weight;
    if (strcmp(name, "wpe.weight") == 0) return gpt2->wpe_weight;
    return NULL;
}

void gpt2_free(PolyModel* model) {
    // UOps themselves are owned by PolyCtx, so we just clean up the host struct
    // (If we were tracking extra host allocations here, we'd free them)
}

// Registry binder for C-level
PolyModel* create_gpt2_model(PolyCtx* ctx, PolyModelConfig* config) {
    PolyGPT2* model = (PolyGPT2*)malloc(sizeof(PolyGPT2));
    
    // Setup model parameters as UOps
    // model->wte_weight = poly_buffer_f32(...)
    
    // Wire up vtable
    model->base.forward = gpt2_forward;
    model->base.get_parameter = gpt2_get_parameter;
    model->base.free = gpt2_free;
    
    return (PolyModel*)model;
}

// Self-register function
void __attribute__((constructor)) register_gpt2() {
    poly_model_register("gpt2", create_gpt2_model);
}
