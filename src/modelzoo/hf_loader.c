/*
 * hf_loader.c -- HuggingFace model loader
 *
 * Loads models from config.json + safetensors files into PolyInstance.
 * No file I/O: caller reads files and passes byte buffers.
 *
 * Flow:
 *   1. Parse config.json -> PolyModelConfig
 *   2. Read model_type -> dispatch to builder (gpt2)
 *   3. Builder creates PolyInstance with named param buffers
 *   4. For each safetensors file: decode, convert to F32, match by name
 */

#define _POSIX_C_SOURCE 200809L
#include "modelzoo.h"
#include "../poly_safetensors.h"
#include "../poly_instance.h"
#include "../../vendor/cjson/cJSON.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Strip "transformer." prefix from HF GPT-2 weight keys */
static const char *strip_prefix(const char *name, const char *prefix) {
  size_t plen = strlen(prefix);
  if (strncmp(name, prefix, plen) == 0) return name + plen;
  return name;
}

/* Check if a weight name should be ignored (non-parameter buffers) */
static int should_ignore_weight(const char *name) {
  /* HF GPT-2 stores constant causal masks as attn.bias and attn.masked_bias */
  if (strstr(name, "attn.bias") != NULL && strstr(name, "c_attn") == NULL &&
      strstr(name, "c_proj") == NULL)
    return 1;
  if (strstr(name, "attn.masked_bias") != NULL)
    return 1;
  return 0;
}

PolyInstance *poly_hf_load(
    const char *config_json, int config_len,
    const uint8_t **weight_files, const int64_t *weight_lens,
    int n_weight_files,
    int max_batch, int max_seq_len)
{
  if (!config_json || config_len <= 0) return NULL;

  /* Parse config.json */
  PolyModelConfig *cfg = poly_model_config_from_json(config_json, config_len);
  if (!cfg) return NULL;

  /* Determine model type */
  const char *model_type = poly_model_config_get_string(cfg, "model_type", "");

  PolyInstance *inst = NULL;

  if (strcmp(model_type, "gpt2") == 0) {
    /* Build GPT-2 config */
    GPT2Config gpt2_cfg;
    gpt2_cfg.vocab_size = poly_model_config_get_int(cfg, "vocab_size", 50257);
    gpt2_cfg.n_embd = poly_model_config_get_int(cfg, "n_embd", 768);
    gpt2_cfg.n_head = poly_model_config_get_int(cfg, "n_head", 12);
    gpt2_cfg.n_layer = poly_model_config_get_int(cfg, "n_layer", 12);
    gpt2_cfg.max_seq_len = max_seq_len > 0 ? max_seq_len :
        poly_model_config_get_int(cfg, "n_positions", 1024);
    gpt2_cfg.norm_eps = poly_model_config_get_float(cfg, "layer_norm_epsilon", 1e-5f);

    inst = poly_gpt2_build(&gpt2_cfg, max_batch > 0 ? max_batch : 1);
  } else {
    fprintf(stderr, "poly_hf_load: unsupported model_type '%s'\n", model_type);
    poly_model_config_free(cfg);
    return NULL;
  }

  poly_model_config_free(cfg);
  if (!inst) return NULL;

  /* Load weights from safetensors files */
  int warned_ignored = 0;
  int loaded_count = 0;
  int skipped_count = 0;

  for (int f = 0; f < n_weight_files; f++) {
    if (!weight_files[f] || weight_lens[f] <= 0) continue;

    int n_views = 0;
    char *metadata = NULL;
    PolySafetensorViewEx *views = poly_safetensors_decode_ex(
        weight_files[f], weight_lens[f], &n_views, &metadata);
    free(metadata);

    if (!views) {
      fprintf(stderr, "poly_hf_load: failed to decode weight file %d\n", f);
      continue;
    }

    for (int i = 0; i < n_views; i++) {
      /* Strip "transformer." prefix (HF GPT-2 convention) */
      const char *name = strip_prefix(views[i].name, "transformer.");

      /* Also strip "model." prefix (some HF models use it) */
      name = strip_prefix(name, "model.");

      /* Check if this is a non-parameter buffer to ignore */
      if (should_ignore_weight(name)) {
        if (!warned_ignored) {
          fprintf(stderr, "poly_hf_load: ignoring non-parameter buffer '%s' "
                  "(and similar)\n", views[i].name);
          warned_ignored = 1;
        }
        skipped_count++;
        free(views[i].name);
        continue;
      }

      /* Handle lm_head.weight -> aliased to wte.weight for GPT-2 */
      if (strcmp(name, "lm_head.weight") == 0) {
        /* GPT-2 uses weight tying: lm_head = wte. Check if wte exists */
        /* For now, just skip it -- wte.weight already serves as lm_head */
        free(views[i].name);
        continue;
      }

      /* Convert to F32 */
      float *f32_data = poly_safetensors_to_f32(&views[i]);
      if (!f32_data) {
        fprintf(stderr, "poly_hf_load: failed to convert '%s' to F32\n",
                views[i].name);
        free(views[i].name);
        continue;
      }

      /* Find matching buffer in instance */
      int n_bufs = poly_instance_buf_count(inst);
      int found = 0;
      for (int b = 0; b < n_bufs; b++) {
        const char *buf_name = poly_instance_buf_name(inst, b);
        if (buf_name && strcmp(buf_name, name) == 0) {
          int64_t numel;
          float *buf_data = poly_instance_buf_data(inst, b, &numel);
          if (buf_data && numel == views[i].numel) {
            memcpy(buf_data, f32_data, numel * sizeof(float));
            loaded_count++;
            found = 1;
          } else if (buf_data) {
            fprintf(stderr, "poly_hf_load: shape mismatch for '%s' "
                    "(instance %lld vs file %lld)\n", name,
                    (long long)numel, (long long)views[i].numel);
          }
          break;
        }
      }

      if (!found) {
        fprintf(stderr, "poly_hf_load: no matching buffer for '%s'\n", name);
      }

      free(f32_data);
      free(views[i].name);
    }

    free(views);
  }

  fprintf(stderr, "poly_hf_load: loaded %d parameters, skipped %d non-parameter buffers\n",
          loaded_count, skipped_count);

  return inst;
}
