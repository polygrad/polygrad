/* EmbeddingGemma2 text/image/audio encoder, expressed through shared Tensor/NN ops.
 * Reference: Transformers 6d3802a45f50ce6f2262029926265629010b2217.
 * No corresponding model exists in pinned Tinygrad. This is not a causal LM:
 * local attention is symmetric, Q/K/V are normalized, and score scaling is 1.
 */
#include "factory.h"
#include "layers.h"
#include "gemma.h"
#include "registry.h"
#include "../loaders/bind.h"
#include "../loaders/import_error.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  int dim, hidden, ple, output, vocab, layers, heads, batch, length, window;
  int hd[256], kv[256];
  bool local[256], composite, hidden_states, vision, audio;
  int patches, patch_size, image_token;
  int audio_frames, audio_features, audio_token;
  double theta[256], eps;
} EmbeddingConfig;

static bool modalities(const cJSON *root, bool *vision, bool *audio, PolyModelError *err) {
  const cJSON *v = cJSON_GetObjectItemCaseSensitive(root, "vision_config");
  const cJSON *a = cJSON_GetObjectItemCaseSensitive(root, "audio_config");
  *vision = v && !cJSON_IsNull(v);
  *audio = a && !cJSON_IsNull(a);
  const cJSON *selection = cJSON_GetObjectItemCaseSensitive(root, "modalities");
  if (!selection) return true;
  if (!cJSON_IsArray(selection))
    return model_factory_error(err, "modalities", "expected array including text");
  unsigned selected = 0;
  for (const cJSON *item = selection->child; item; item = item->next) {
    unsigned bit = !cJSON_IsString(item)                 ? 0
                   : !strcmp(item->valuestring, "text")  ? 1
                   : !strcmp(item->valuestring, "image") ? 2
                   : !strcmp(item->valuestring, "audio") ? 4
                                                         : 0;
    if (!bit || (selected & bit))
      return model_factory_error(err, "modalities", "expected distinct text, image or audio names");
    selected |= bit;
  }
  if (!(selected & 1) || ((selected & 2) && !*vision) || ((selected & 4) && !*audio))
    return model_factory_error(err, "modalities", "text is required; selected towers need configs");
  *vision = (selected & 2) != 0;
  *audio = (selected & 4) != 0;
  return true;
}

static bool config(const cJSON *root, EmbeddingConfig *c, PolyModelError *err) {
  memset(c, 0, sizeof(*c));
  if (!modalities(root, &c->vision, &c->audio, err)) return false;
  const cJSON *j = cJSON_GetObjectItemCaseSensitive(root, "text_config");
  c->composite = j != NULL;
  if (!j) j = root;
  if (!cJSON_IsObject(j)) return model_factory_error(err, "text_config", "expected object");
  const cJSON *hidden_states = cJSON_GetObjectItemCaseSensitive(root, "output_hidden_states");
  if (hidden_states && !cJSON_IsBool(hidden_states))
    return model_factory_error(err, "output_hidden_states", "expected boolean");
  c->hidden_states = cJSON_IsTrue(hidden_states);
  const cJSON *audio = cJSON_GetObjectItemCaseSensitive(root, "audio_config");
  if (c->audio) {
    if (!c->composite || !cJSON_IsObject(audio))
      return model_factory_error(err, "audio_config", "expected object alongside text_config");
    /* Time capacity must be supplied explicitly; it comes from the processor,
     * not the text token limit. Feature width follows HF's projection layout. */
    if (!cJSON_GetObjectItemCaseSensitive(root, "audio_seq_len"))
      return model_factory_error(err, "audio_seq_len", "required for audio inputs");
    if (!model_gemma_integer(root, "audio_seq_len", 0, 1, 65536, &c->audio_frames, err) ||
        !model_gemma_integer(root, "audio_token_id", 258881, 1, 2147483647, &c->audio_token, err))
      return false;
    const cJSON *channels = cJSON_GetObjectItemCaseSensitive(audio, "subsampling_conv_channels");
    const cJSON *first = cJSON_IsArray(channels) ? cJSON_GetArrayItem(channels, 0) : NULL;
    if (!cJSON_IsNumber(first) || first->valuedouble != first->valueint || first->valueint < 4 ||
        first->valueint > 4096)
      return model_factory_error(err, "subsampling_conv_channels", "invalid audio feature width");
    c->audio_features = first->valueint;
  }
  const cJSON *vision = cJSON_GetObjectItemCaseSensitive(root, "vision_config");
  if (c->vision) {
    if (!c->composite || !cJSON_IsObject(vision))
      return model_factory_error(err, "vision_config", "expected object alongside text_config");
    if (!model_config_choice(vision, "model_type", "|gemma4_vision|", err) ||
        !model_gemma_integer(vision, "patch_size", 16, 1, 256, &c->patch_size, err) ||
        !model_gemma_integer(root, "image_num_patches", 2520, 1, 16384, &c->patches, err) ||
        !model_gemma_integer(root, "image_token_id", 258880, 1, 2147483647, &c->image_token, err))
      return false;
  }
  const cJSON *bias = cJSON_GetObjectItemCaseSensitive(j, "attention_bias");
  if (bias && !cJSON_IsFalse(bias))
    return model_factory_error(err, "attention_bias", "must be false");
  if (!model_config_choice(j, "hidden_activation", "|gelu_pytorch_tanh|", err) ||
      !model_gemma_integer(j, "hidden_size", 512, 1, 65536, &c->dim, err) ||
      !model_gemma_integer(j, "intermediate_size", 2048, 1, 65536, &c->hidden, err) ||
      !model_gemma_integer(j, "hidden_size_per_layer_input", 512, 1, 65536, &c->ple, err) ||
      !model_gemma_integer(j, "embedding_dim", 768, 1, 65536, &c->output, err) ||
      !model_gemma_integer(j, "vocab_size", 262144, 1, 1048576, &c->vocab, err) ||
      !model_gemma_integer(j, "num_hidden_layers", 24, 1, 256, &c->layers, err) ||
      !model_gemma_integer(j, "num_attention_heads", 4, 1, 256, &c->heads, err) ||
      !model_gemma_integer(j, "sliding_window", 512, 1, 8192, &c->window, err) ||
      !model_gemma_integer(root, "batch_size", 1, 1, 1024, &c->batch, err) ||
      !model_gemma_integer(root, "max_seq_len", 32, 1, 8192, &c->length, err))
    return false;
  const cJSON *eps = cJSON_GetObjectItemCaseSensitive(j, "rms_norm_eps");
  c->eps = eps ? eps->valuedouble : 1e-6;
  if ((eps && !cJSON_IsNumber(eps)) || !isfinite(c->eps) || c->eps <= 0 || c->eps > 1)
    return model_factory_error(err, "rms_norm_eps", "expected finite value in (0,1]");
  const cJSON *types = cJSON_GetObjectItemCaseSensitive(j, "layer_types");
  const cJSON *overrides = cJSON_GetObjectItemCaseSensitive(j, "per_layer_config");
  const cJSON *ropes = cJSON_GetObjectItemCaseSensitive(j, "rope_parameters");
  if (!cJSON_IsArray(types) || cJSON_GetArraySize(types) != c->layers ||
      (overrides && !cJSON_IsObject(overrides)) || (ropes && !cJSON_IsObject(ropes)))
    return model_factory_error(
        err, "layer_types",
        "declare every layer and optional per_layer_config/rope_parameters objects"
    );
  for (int i = 0; i < c->layers; i++) {
    const cJSON *type = cJSON_GetArrayItem(types, i);
    if (!cJSON_IsString(type) || (strcmp(type->valuestring, "sliding_attention") &&
                                  strcmp(type->valuestring, "full_attention")))
      return model_factory_error(
          err, "layer_types", "expected sliding_attention or full_attention"
      );
    c->local[i] = !strcmp(type->valuestring, "sliding_attention");
    if (!model_gemma_integer(j, "head_dim", 256, 1, 65536, &c->hd[i], err) ||
        !model_gemma_integer(j, "num_key_value_heads", 2, 1, 256, &c->kv[i], err))
      return false;
    /* HF serializes layer numbers with or without leading zeros. Reject
     * aliases and unused keys instead of silently selecting a different shape. */
    bool seen = false;
    for (const cJSON *v = overrides ? overrides->child : NULL; v; v = v->next) {
      char *end = NULL;
      long index = strtol(v->string, &end, 10);
      if (!v->string[0] || *end || index < 0 || index >= c->layers || !cJSON_IsObject(v))
        return model_factory_error(err, "per_layer_config", "invalid layer override");
      if (index != i) continue;
      if (seen) return model_factory_error(err, "per_layer_config", "duplicate layer number");
      seen = true;
      for (const cJSON *field = v->child; field; field = field->next)
        if (strcmp(field->string, "head_dim") && strcmp(field->string, "num_key_value_heads"))
          return model_factory_error(
              err, "per_layer_config", "unsupported field %s", field->string
          );
      if (!model_gemma_integer(v, "head_dim", c->hd[i], 1, 65536, &c->hd[i], err) ||
          !model_gemma_integer(v, "num_key_value_heads", c->kv[i], 1, 256, &c->kv[i], err))
        return false;
    }
    if (c->hd[i] % 2 || c->heads % c->kv[i])
      return model_factory_error(
          err, "attention", "head_dim must be even and KV heads must divide heads"
      );
    c->theta[i] = c->local[i] ? 10000 : 1000000;
    const cJSON *r = ropes ? cJSON_GetObjectItemCaseSensitive(ropes, type->valuestring) : NULL;
    if (ropes && !cJSON_IsObject(r))
      return model_factory_error(err, "rope_parameters", "missing layer type");
    if (r) {
      if (!model_config_choice(r, "rope_type", "|default|", err)) return false;
      const cJSON *theta = cJSON_GetObjectItemCaseSensitive(r, "rope_theta");
      if (theta) {
        if (!cJSON_IsNumber(theta) || !isfinite(theta->valuedouble) || theta->valuedouble < 1)
          return model_factory_error(err, "rope_theta", "expected finite value >=1");
        c->theta[i] = theta->valuedouble;
      }
    }
  }
  return true;
}

static PolyTensor *attention(
    PolyModel *m,
    const EmbeddingConfig *c,
    int layer,
    const char *base,
    PolyTensor *x,
    PolyTensor *mask
) {
  int n = c->length, hd = c->hd[layer];
  PolyModelRoPEConfig rope = {.length = n, .dim = hd, .theta = c->theta[layer], .factor = 1};
  char name[192];
  snprintf(name, sizeof(name), "rope.%d.cos", layer);
  PolyTensor *cos = poly_model_rope_frequencies(m, name, &rope, false);
  snprintf(name, sizeof(name), "rope.%d.sin", layer);
  PolyTensor *sin = poly_model_rope_frequencies(m, name, &rope, true);
  ModelGemmaAttention a = {c->batch, n, c->dim, c->heads, c->kv[layer], hd, 1, c->eps, false};
  return model_gemma_attention(m, base, &a, x, mask, cos, sin);
}

/* Stable compaction map, not host-side boolean indexing: each valid soft token
 * writes its 1-based storage index at its prefix rank. Padding contributes zero
 * to scatter-sum. Placeholder prefix ranks then gather in processor order. */
static PolyTensor *insert_modality_tokens(
    PolyCtx *ctx,
    PolyTensor *x,
    PolyTensor *slots,
    PolyTensor *features,
    PolyTensor *valid,
    int batch,
    int length,
    int soft,
    int dim
) {
  int count = batch * soft;
  PolyTensor *vf =
      poly_tensor_cast(ctx, poly_tensor_reshape(ctx, valid, (int64_t[]){count}, 1), POLY_INT32);
  PolyTensor *rank = poly_tensor_cumsum(ctx, vf, 0);
  rank = poly_tensor_alu2(ctx, POLY_OP_SUB, rank, poly_tensor_const_like_int(ctx, rank, 1));
  rank = poly_tensor_alu2(ctx, POLY_OP_MAX, rank, poly_tensor_const_like_int(ctx, rank, 0));
  PolyTensor *index =
      poly_tensor_arange_int(ctx, 1, count + 1, 1, POLY_INT32, poly_ctx_get_preferred_device(ctx));
  PolyTensor *map = poly_tensor_scatter_reduce(
      ctx, poly_tensor_const_like_int(ctx, index, 0), 0, rank,
      poly_tensor_alu2(ctx, POLY_OP_MUL, index, vf), "sum", true
  );
  PolyTensor *flat = poly_tensor_cast(
      ctx, poly_tensor_reshape(ctx, slots, (int64_t[]){batch * length}, 1), POLY_INT32
  );
  rank = poly_tensor_cumsum(ctx, flat, 0);
  rank = poly_tensor_alu2(ctx, POLY_OP_SUB, rank, poly_tensor_const_like_int(ctx, rank, 1));
  rank = poly_tensor_alu2(ctx, POLY_OP_MAX, rank, poly_tensor_const_like_int(ctx, rank, 0));
  index = poly_tensor_index_select(ctx, map, 0, rank);
  index = poly_tensor_alu2(ctx, POLY_OP_SUB, index, poly_tensor_const_like_int(ctx, index, 1));
  features = poly_tensor_reshape(ctx, features, (int64_t[]){count, dim}, 2);
  features = poly_tensor_index_select(ctx, features, 0, index);
  features = poly_tensor_reshape(ctx, features, (int64_t[]){batch, length, dim}, 3);
  slots = poly_tensor_reshape(ctx, slots, (int64_t[]){batch, length, 1}, 3);
  return poly_tensor_alu3(ctx, POLY_OP_WHERE, slots, features, x);
}

static PolyModel *slot_check(
    PolyCtx *ctx,
    PolyTensor *ids,
    const char *metadata_name,
    PolyTensor *metadata,
    PolyTensor *slots,
    PolyTensor *valid,
    PolyModelError *err
) {
  /* Count the same pooled/subsampled mask used for insertion, globally across
   * the batch like HF masked_scatter. No duplicate pooling rules on the host. */
  PolyTensor *a =
      poly_tensor_sum(ctx, poly_tensor_cast(ctx, slots, POLY_INT32), (int64_t[]){0, 1}, 2, false);
  PolyTensor *b =
      poly_tensor_sum(ctx, poly_tensor_cast(ctx, valid, POLY_INT32), (int64_t[]){0, 1}, 2, false);
  PolyTensor *ok = poly_tensor_alu2(ctx, POLY_OP_CMPEQ, a, b);
  PolyBindingSpec bindings[] = {
      {"input_ids", POLY_ROLE_INPUT, ids, 0},
      {metadata_name, POLY_ROLE_INPUT, metadata, 0},
      {"valid", POLY_ROLE_OUTPUT, ok, 0},
  };
  const char *inputs[] = {"input_ids", metadata_name}, *outputs[] = {"valid"};
  PolyEntrypointSpec entry = {
      .name = "check", .inputs = inputs, .n_inputs = 2, .outputs = outputs, .n_outputs = 1};
  return poly_model_from_bindings(ctx, bindings, 3, &entry, 1, NULL, err);
}

PolyModel *model_embeddinggemma2_from_config(PolyCtx *ctx, const cJSON *root, PolyModelError *err) {
  EmbeddingConfig c;
  if (!config(root, &c, err)) return NULL;
  PolyModel *m = poly_model_new(ctx, NULL);
  if (!m) return NULL;
  PolyModel *image_check = NULL, *audio_check = NULL;
  const char *p = c.composite ? "language_model." : "";
  char name[192];
  char hidden_names[257][32];
  char vision_names[257][32];
  char audio_names[257][32];
  const char *outputs[777] = {"last_hidden_state", "sentence_embedding"};
  int n_outputs = 2;
  const char *inputs[6] = {"input_ids", "attention_mask"};
  int n_inputs = 2;
  PolyTensor *ids = poly_model_input(m, "input_ids", POLY_INT32, (int64_t[]){c.batch, c.length}, 2);
  PolyTensor *valid =
      poly_model_input(m, "attention_mask", POLY_INT32, (int64_t[]){c.batch, c.length}, 2);
  snprintf(name, sizeof(name), "%sembed_tokens", p);
  PolyTensor *slots = NULL, *tokens = ids;
  if (c.vision) {
    slots = poly_tensor_alu2(
        ctx, POLY_OP_CMPEQ, ids, poly_tensor_const_like_int(ctx, ids, c.image_token)
    );
    tokens =
        poly_tensor_alu3(ctx, POLY_OP_WHERE, slots, poly_tensor_const_like_int(ctx, ids, 0), ids);
  }
  PolyTensor *audio_slots = NULL;
  if (c.audio) {
    audio_slots = poly_tensor_alu2(
        ctx, POLY_OP_CMPEQ, ids, poly_tensor_const_like_int(ctx, ids, c.audio_token)
    );
    tokens = poly_tensor_alu3(
        ctx, POLY_OP_WHERE, audio_slots, poly_tensor_const_like_int(ctx, ids, 0), tokens
    );
  }
  PolyTensor *x =
      model_gemma_scale(ctx, poly_model_embedding(m, name, tokens, c.vocab, c.dim), sqrt(c.dim));
  if (c.vision) {
    inputs[n_inputs++] = "pixel_values";
    inputs[n_inputs++] = "image_position_ids";
    const cJSON *vision = cJSON_GetObjectItemCaseSensitive(root, "vision_config");
    PolyTensor *pixels = poly_model_input(
        m, "pixel_values", POLY_FLOAT32,
        (int64_t[]){c.batch, c.patches, 3 * c.patch_size * c.patch_size}, 3
    );
    PolyTensor *positions = poly_model_input(
        m, "image_position_ids", POLY_INT32, (int64_t[]){c.batch, c.patches, 2}, 3
    );
    PolyTensor *image_valid = NULL;
    PolyTensor *vision_states[257] = {0};
    PolyTensor *features = model_gemma4_vision(
        m, vision, c.batch, c.patches, pixels, positions, &image_valid,
        c.hidden_states ? vision_states : NULL, err
    );
    if (!features) goto fail;
    image_check = slot_check(ctx, ids, "image_position_ids", positions, slots, image_valid, err);
    if (!image_check) goto fail;
    for (int i = 0; i < 257 && vision_states[i]; i++) {
      snprintf(vision_names[i], sizeof(vision_names[i]), "vision_hidden_states.%d", i);
      if (poly_model_output(m, vision_names[i], vision_states[i]) != POLY_STATUS_OK) goto fail;
      outputs[n_outputs++] = vision_names[i];
    }
    const int64_t *dims = poly_uop_max_shape_dims(ctx, poly_tensor_uop_physical(features));
    if (!dims) goto fail;
    int soft = (int)dims[1], width = (int)dims[2];
    const cJSON *epsilon = cJSON_GetObjectItemCaseSensitive(vision, "rms_norm_eps");
    features =
        poly_tensor_rmsnorm_apply(ctx, features, NULL, epsilon ? epsilon->valuedouble : 1e-6);
    features =
        model_gemma_linear(m, "embed_vision.", "embedding_projection", features, width, c.dim);
    if (!features || poly_model_output(m, "image_hidden_states", features) != POLY_STATUS_OK ||
        poly_model_output(m, "image_attention_mask", image_valid) != POLY_STATUS_OK)
      goto fail;
    outputs[n_outputs++] = "image_hidden_states";
    outputs[n_outputs++] = "image_attention_mask";
    x = insert_modality_tokens(
        ctx, x, slots, features, image_valid, c.batch, c.length, soft, c.dim
    );
  }
  if (c.audio) {
    inputs[n_inputs++] = "input_features";
    inputs[n_inputs++] = "input_features_mask";
    const cJSON *audio = cJSON_GetObjectItemCaseSensitive(root, "audio_config");
    PolyTensor *features = poly_model_input(
        m, "input_features", POLY_FLOAT32, (int64_t[]){c.batch, c.audio_frames, c.audio_features}, 3
    );
    PolyTensor *audio_valid = poly_model_input(
        m, "input_features_mask", POLY_INT32, (int64_t[]){c.batch, c.audio_frames}, 2
    );
    PolyTensor *audio_metadata = audio_valid;
    PolyTensor *states[257] = {0};
    features = model_gemma4_audio(
        m, audio, c.batch, c.audio_frames, features, audio_valid, &audio_valid,
        c.hidden_states ? states : NULL, err
    );
    if (!features) goto fail;
    audio_check =
        slot_check(ctx, ids, "input_features_mask", audio_metadata, audio_slots, audio_valid, err);
    if (!audio_check) goto fail;
    for (int i = 0; i < 257 && states[i]; i++) {
      snprintf(audio_names[i], sizeof(audio_names[i]), "audio_hidden_states.%d", i);
      if (poly_model_output(m, audio_names[i], states[i]) != POLY_STATUS_OK) goto fail;
      outputs[n_outputs++] = audio_names[i];
    }
    const int64_t *dims = poly_uop_max_shape_dims(ctx, poly_tensor_uop_physical(features));
    if (!dims) goto fail;
    int soft = (int)dims[1], width = (int)dims[2];
    const cJSON *eps = cJSON_GetObjectItemCaseSensitive(audio, "rms_norm_eps");
    features = poly_tensor_rmsnorm_apply(ctx, features, NULL, eps ? eps->valuedouble : 1e-6);
    features =
        model_gemma_linear(m, "embed_audio.", "embedding_projection", features, width, c.dim);
    if (!features || poly_model_output(m, "audio_hidden_states", features) != POLY_STATUS_OK ||
        poly_model_output(m, "audio_attention_mask", audio_valid) != POLY_STATUS_OK)
      goto fail;
    outputs[n_outputs++] = "audio_hidden_states";
    outputs[n_outputs++] = "audio_attention_mask";
    x = insert_modality_tokens(
        ctx, x, audio_slots, features, audio_valid, c.batch, c.length, soft, c.dim
    );
  }
  x = poly_tensor_contiguous(ctx, x);
  if (c.hidden_states) {
    snprintf(hidden_names[0], sizeof(hidden_names[0]), "hidden_states.0");
    outputs[n_outputs++] = hidden_names[0];
    if (poly_model_output(m, hidden_names[0], x) != POLY_STATUS_OK) goto fail;
  }
  PolyTensor *ple = model_gemma_scale(
      ctx, model_gemma_linear(m, p, "ple.per_layer_model_projection", x, c.dim, c.layers * c.ple),
      1 / sqrt(c.dim)
  );
  ple = poly_tensor_reshape(ctx, ple, (int64_t[]){c.batch, c.length, c.layers, c.ple}, 4);
  ple = model_gemma_norm(m, p, "ple.per_layer_projection_norm", ple, c.ple, c.eps);
  PolyTensor *keymask =
      poly_tensor_alu2(ctx, POLY_OP_CMPNE, valid, poly_tensor_const_like_int(ctx, valid, 0));
  keymask = poly_tensor_reshape(ctx, keymask, (int64_t[]){c.batch, 1, 1, 1, c.length}, 5);
  PolyTensor *mask_float = poly_tensor_cast(ctx, keymask, POLY_FLOAT32);
  PolyTensor *zero = poly_tensor_const_like_float(ctx, mask_float, 0);
  PolyTensor *negative = poly_tensor_const_like_float(ctx, mask_float, -3.4028234663852886e38);
  PolyTensor *full_mask = poly_tensor_alu3(ctx, POLY_OP_WHERE, keymask, zero, negative);
  PolyTensor *pos =
      poly_tensor_arange_int(ctx, 0, c.length, 1, POLY_INT32, poly_ctx_get_preferred_device(ctx));
  PolyTensor *delta = poly_tensor_abs(
      ctx, poly_tensor_alu2(
               ctx, POLY_OP_SUB, poly_tensor_reshape(ctx, pos, (int64_t[]){c.length, 1}, 2), pos
           )
  );
  PolyTensor *local = poly_tensor_alu2(
      ctx, POLY_OP_CMPLT, delta, poly_tensor_const_like_int(ctx, delta, c.window + 1)
  );
  local = poly_tensor_alu2(ctx, POLY_OP_AND, keymask, local);
  PolyTensor *local_mask = poly_tensor_alu3(ctx, POLY_OP_WHERE, local, zero, negative);
  for (int i = 0; i < c.layers; i++) {
    char base[128];
    snprintf(base, sizeof(base), "%slayers.%d.", p, i);
    PolyTensor *h = model_gemma_norm(m, base, "input_layernorm", x, c.dim, c.eps);
    h = attention(m, &c, i, base, h, c.local[i] ? local_mask : full_mask);
    h = model_gemma_norm(m, base, "post_attention_layernorm", h, c.dim, c.eps);
    x = poly_tensor_alu2(ctx, POLY_OP_ADD, x, h);
    h = model_gemma_norm(m, base, "pre_feedforward_layernorm", x, c.dim, c.eps);
    PolyTensor *gate =
        poly_tensor_gelu(ctx, model_gemma_linear(m, base, "mlp.gate_proj", h, c.dim, c.hidden));
    h = poly_tensor_alu2(
        ctx, POLY_OP_MUL, gate, model_gemma_linear(m, base, "mlp.up_proj", h, c.dim, c.hidden)
    );
    h = model_gemma_linear(m, base, "mlp.down_proj", h, c.hidden, c.dim);
    x = poly_tensor_alu2(
        ctx, POLY_OP_ADD, x,
        model_gemma_norm(m, base, "post_feedforward_layernorm", h, c.dim, c.eps)
    );
    h = poly_tensor_gelu(
        ctx, model_gemma_linear(m, base, "ple_block.per_layer_input_gate", x, c.dim, c.ple)
    );
    PolyTensor *slice = poly_tensor_shrink(
        ctx, ple, (int64_t[][2]){{0, c.batch}, {0, c.length}, {i, i + 1}, {0, c.ple}}, 4
    );
    slice = poly_tensor_reshape(ctx, slice, (int64_t[]){c.batch, c.length, c.ple}, 3);
    h = poly_tensor_alu2(ctx, POLY_OP_MUL, h, slice);
    h = model_gemma_linear(m, base, "ple_block.per_layer_projection", h, c.ple, c.dim);
    h = model_gemma_norm(m, base, "ple_block.post_per_layer_input_norm", h, c.dim, c.eps);
    x = poly_tensor_alu2(ctx, POLY_OP_ADD, x, h);
    snprintf(name, sizeof(name), "%slayer_scalar", base);
    PolyTensor *scalar = poly_model_param(m, name, POLY_FLOAT32, (int64_t[]){1}, 1);
    x = poly_tensor_contiguous(ctx, poly_tensor_alu2(ctx, POLY_OP_MUL, x, scalar));
    if (!x) goto fail;
    if (c.hidden_states) {
      snprintf(hidden_names[i + 1], sizeof(hidden_names[i + 1]), "hidden_states.%d", i + 1);
      outputs[n_outputs++] = hidden_names[i + 1];
      if (poly_model_output(m, hidden_names[i + 1], x) != POLY_STATUS_OK) goto fail;
    }
  }
  x = model_gemma_norm(m, p, "norm", x, c.dim, c.eps);
  x = model_gemma_linear(m, p, "embedding_projection", x, c.dim, c.output);
  PolyTensor *weights = poly_tensor_cast(ctx, valid, POLY_FLOAT32);
  weights = poly_tensor_reshape(ctx, weights, (int64_t[]){c.batch, c.length, 1}, 3);
  PolyTensor *pool = poly_tensor_sum(
      ctx, poly_tensor_alu2(ctx, POLY_OP_MUL, x, weights), (int64_t[]){1}, 1, false
  );
  PolyTensor *count = poly_tensor_sum(ctx, weights, (int64_t[]){1}, 1, false);
  count = poly_tensor_alu2(ctx, POLY_OP_MAX, count, poly_tensor_const_like_float(ctx, count, 1e-9));
  pool = poly_tensor_div(ctx, pool, count, 0);
  pool = poly_tensor_normalize(ctx, pool, 2, -1, 1e-12);
  if (!pool || poly_model_output(m, "last_hidden_state", x) != POLY_STATUS_OK ||
      poly_model_output(m, "sentence_embedding", pool) != POLY_STATUS_OK ||
      poly_model_entrypoint(m, "forward", inputs, n_inputs, outputs, n_outputs, NULL) !=
          POLY_STATUS_OK ||
      poly_model_build(m, err) != POLY_STATUS_OK || poly_model_require_weights(m) != 0)
    goto fail;
  if ((image_check && poly_model_require(
                          m, "forward", "image features and token slots do not match", image_check
                      ) != POLY_STATUS_OK) ||
      (audio_check && poly_model_require(
                          m, "forward", "audio features and token slots do not match", audio_check
                      ) != POLY_STATUS_OK))
    goto fail;
  poly_model_free(image_check);
  poly_model_free(audio_check);
  return m;
fail:
  poly_model_free(image_check);
  poly_model_free(audio_check);
  if (err && !err->code && poly_model_last_error(m)->code) *err = *poly_model_last_error(m);
  model_factory_error(err, "EmbeddingGemma2", "graph construction failed");
  poly_model_free(m);
  return NULL;
}

PolyModel *model_embeddinggemma2_from_hf_decoded(
    const PolyHfDecoded *hf,
    const PolyGenericImportOpts *opts
) {
  if (!hf || !opts) return NULL;
  PolyModelError err = {0};
  cJSON *j = cJSON_Duplicate(hf->config, true);
  if (!j) return NULL;
  if (opts->max_batch > 0) {
    cJSON_DeleteItemFromObject(j, "batch_size");
    cJSON_AddNumberToObject(j, "batch_size", opts->max_batch);
  }
  if (opts->max_seq_len > 0) {
    cJSON_DeleteItemFromObject(j, "max_seq_len");
    cJSON_AddNumberToObject(j, "max_seq_len", opts->max_seq_len);
  }
  bool composite = cJSON_GetObjectItemCaseSensitive(j, "text_config") != NULL;
  bool has_vision, has_audio;
  if (!modalities(j, &has_vision, &has_audio, &err)) {
    cJSON_Delete(j);
    poly_import_error_set(POLY_IMPORT_ERR_UNSUPPORTED_MODEL, "%s", err.message);
    return NULL;
  }
  PolyModelFactoryScope scope;
  PolyModel *m = NULL;
  if (model_factory_begin(&scope, opts->ctx, opts->device))
    m = model_factory_end(&scope, model_embeddinggemma2_from_config(scope.ctx, j, &err));
  cJSON_Delete(j);
  if (!m) {
    poly_import_error_set(
        POLY_IMPORT_ERR_UNSUPPORTED_MODEL, "%s",
        err.message[0] ? err.message : "EmbeddingGemma2 construction failed"
    );
    return NULL;
  }
  int n = poly_model_buf_count(m);
  bool *seen = calloc((size_t)n, sizeof(bool));
  PolyBindIndex *index = poly_bind_index_create(m);
  if (!seen || !index) goto fail;
  for (int i = 0; i < hf->n_tensors; i++) {
    const PolyDecodedTensor *t = &hf->tensors[i];
    /* Explicitly disabled towers may remain in the full checkpoint. Never
     * ignore an unknown text weight or missing text parameter. */
    if (composite && ((!has_vision && (!strncmp(t->name, "vision_tower.", 13) ||
                                       !strncmp(t->name, "embed_vision.", 13))) ||
                      (!has_audio && (!strncmp(t->name, "audio_tower.", 12) ||
                                      !strncmp(t->name, "embed_audio.", 12)))))
      continue;
    int b = poly_bind_index_find(index, t->name);
    if (b < 0 || poly_model_buf_role(m, b) != POLY_ROLE_PARAM || seen[b] ||
        poly_import_bind_tensor(index, t->name, t, 0, -1) != 1) {
      poly_import_error_set(
          POLY_IMPORT_ERR_WEIGHT_MISMATCH, "EmbeddingGemma2: invalid or unexpected weight '%s'",
          t->name
      );
      goto fail;
    }
    seen[b] = true;
  }
  for (int b = 0; b < n; b++)
    if (poly_model_buf_role(m, b) == POLY_ROLE_PARAM && !seen[b]) {
      poly_import_error_set(
          POLY_IMPORT_ERR_WEIGHT_MISMATCH, "EmbeddingGemma2: missing weight '%s'",
          poly_model_buf_name(m, b)
      );
      goto fail;
    }
  free(seen);
  poly_bind_index_destroy(index);
  return m;
fail:
  free(seen);
  poly_bind_index_destroy(index);
  poly_model_free(m);
  if (poly_import_last_error_code() == POLY_IMPORT_OK)
    poly_import_error_set(POLY_IMPORT_ERR_INTERNAL, "EmbeddingGemma2 import allocation failed");
  return NULL;
}
