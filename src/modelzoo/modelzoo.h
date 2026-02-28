#ifndef POLYGRAD_MODELZOO_H
#define POLYGRAD_MODELZOO_H

#include "../polygrad.h"
#include "../instance.h"

/* ── Model configuration (cJSON wrapper) ──────────────────────────── */

typedef struct PolyModelConfig PolyModelConfig;

PolyModelConfig *poly_model_config_new(void);
PolyModelConfig *poly_model_config_from_json(const char *json, int len);
int poly_model_config_get_int(const PolyModelConfig *cfg, const char *key, int default_val);
float poly_model_config_get_float(const PolyModelConfig *cfg, const char *key, float default_val);
const char *poly_model_config_get_string(const PolyModelConfig *cfg, const char *key, const char *default_val);
void poly_model_config_set_int(PolyModelConfig *config, const char *key, int value);
void poly_model_config_set_float(PolyModelConfig *config, const char *key, float value);
void poly_model_config_free(PolyModelConfig *config);

/* ── Instantiated model graph with virtual dispatch ───────────────── */

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

/* ── Registry ─────────────────────────────────────────────────────── */

typedef PolyModel* (*PolyModelCreateFn)(PolyCtx* ctx, PolyModelConfig* config);
void poly_model_register(const char* kind, PolyModelCreateFn create_fn);

/* ── HuggingFace loader ──────────────────────────────────────────── */

/* Load a HF model from config.json + safetensors weight files.
 * No file I/O: caller reads files and passes byte buffers.
 * Returns ready-to-use PolyInstance or NULL on error. */
PolyInstance *poly_hf_load(
    const char *config_json, int config_len,
    const uint8_t **weight_files, const int64_t *weight_lens,
    int n_weight_files,
    int max_batch, int max_seq_len);

/* ── GPT-2 builder (exposed for testing) ─────────────────────────── */

typedef struct {
  int vocab_size;
  int n_embd;
  int n_head;
  int n_layer;
  int max_seq_len;
  float norm_eps;
} GPT2Config;

/* Build a GPT-2 PolyInstance from config. No weights loaded.
 * Caller loads weights via poly_instance_import_weights or poly_hf_load. */
PolyInstance *poly_gpt2_build(const GPT2Config *cfg, int max_batch);

#endif /* POLYGRAD_MODELZOO_H */
