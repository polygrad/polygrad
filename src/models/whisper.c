#include "factory.h"
#include "layers.h"
#include "registry.h"
#include "../loaders/bind.h"
#include "../loaders/import_error.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* OpenAI Whisper / pinned examples/whisper.py. HF parameter names are retained.
 * Encoder projections populate cross-attention state once; decoding appends
 * self-attention state through the same cached SDPA used by dense decoders. */
typedef struct {
  PolyTensor *qkv[3][2], *out[2], *norm[2];
} WhisperAttention;
typedef struct {
  WhisperAttention self, cross;
  PolyTensor *fc1[2], *fc2[2], *norm[2], *self_cache, *cross_cache;
} WhisperBlock;
typedef struct {
  int dim, mels, audio, capacity, length, enc_layers, dec_layers, enc_heads, dec_heads;
  int enc_hidden, dec_hidden, vocab;
} WhisperConfig;

static bool whisper_int(
    const cJSON *j,
    const char *key,
    int fallback,
    int *out,
    PolyModelError *err
) {
  if (!model_config_integer(j, key, 1, 65536, false, err)) return false;
  const cJSON *value = cJSON_GetObjectItemCaseSensitive(j, key);
  *out = value ? value->valueint : fallback;
  return *out > 0 || model_factory_error(err, key, "missing positive dimension");
}

static bool whisper_config(const cJSON *j, WhisperConfig *c, PolyModelError *err) {
  int batch;
  if (!whisper_int(j, "d_model", 384, &c->dim, err) ||
      !whisper_int(j, "num_mel_bins", 80, &c->mels, err) ||
      !whisper_int(j, "max_source_positions", 1500, &c->audio, err) ||
      !whisper_int(j, "max_target_positions", 448, &c->capacity, err) ||
      !whisper_int(j, "max_seq_len", 1, &c->length, err) ||
      !whisper_int(j, "encoder_layers", 4, &c->enc_layers, err) ||
      !whisper_int(j, "decoder_layers", 4, &c->dec_layers, err) ||
      !whisper_int(j, "encoder_attention_heads", 6, &c->enc_heads, err) ||
      !whisper_int(j, "decoder_attention_heads", 6, &c->dec_heads, err) ||
      !whisper_int(j, "encoder_ffn_dim", 4 * c->dim, &c->enc_hidden, err) ||
      !whisper_int(j, "decoder_ffn_dim", 4 * c->dim, &c->dec_hidden, err) ||
      !whisper_int(j, "vocab_size", 51865, &c->vocab, err) ||
      !whisper_int(j, "batch_size", 1, &batch, err))
    return false;
  if (batch != 1 || c->length > c->capacity || c->dim % c->enc_heads || c->dim % c->dec_heads)
    return model_factory_error(
        err, "Whisper",
        "expected batch_size=1, divisible head width and max_seq_len <= max_target_positions"
    );
  const cJSON *act = cJSON_GetObjectItemCaseSensitive(j, "activation_function");
  const cJSON *scaled = cJSON_GetObjectItemCaseSensitive(j, "scale_embedding");
  const cJSON *tied = cJSON_GetObjectItemCaseSensitive(j, "tie_word_embeddings");
  if ((act && (!cJSON_IsString(act) || strcmp(act->valuestring, "gelu"))) ||
      (scaled && !cJSON_IsFalse(scaled)) || (tied && !cJSON_IsTrue(tied)))
    return model_factory_error(
        err, "Whisper", "requires gelu, scale_embedding=false and tied embeddings"
    );
  return true;
}

static bool whisper_attention_parameters(
    PolyModel *m,
    WhisperAttention *a,
    const char *prefix,
    int dim
) {
  char name[192];
  const char *parts[] = {"q_proj", "k_proj", "v_proj"};
  for (int i = 0; i < 3; i++) {
    snprintf(name, sizeof(name), "%s.%s", prefix, parts[i]);
    if (poly_model_linear_parameters(m, name, dim, dim, i != 1, &a->qkv[i][0], &a->qkv[i][1]))
      return false;
  }
  snprintf(name, sizeof(name), "%s.out_proj", prefix);
  if (poly_model_linear_parameters(m, name, dim, dim, true, &a->out[0], &a->out[1])) return false;
  snprintf(name, sizeof(name), "%s_layer_norm", prefix);
  return !poly_model_norm_parameters(m, name, dim, true, &a->norm[0], &a->norm[1]);
}

static bool whisper_block_parameters(
    PolyModel *m,
    WhisperBlock *b,
    const WhisperConfig *c,
    int i,
    bool decoder
) {
  char prefix[128], name[192];
  snprintf(prefix, sizeof(prefix), "model.%s.layers.%d", decoder ? "decoder" : "encoder", i);
  snprintf(name, sizeof(name), "%s.self_attn", prefix);
  if (!whisper_attention_parameters(m, &b->self, name, c->dim)) return false;
  int hidden = decoder ? c->dec_hidden : c->enc_hidden;
  snprintf(name, sizeof(name), "%s.fc1", prefix);
  if (poly_model_linear_parameters(m, name, c->dim, hidden, true, &b->fc1[0], &b->fc1[1]))
    return false;
  snprintf(name, sizeof(name), "%s.fc2", prefix);
  if (poly_model_linear_parameters(m, name, hidden, c->dim, true, &b->fc2[0], &b->fc2[1]))
    return false;
  snprintf(name, sizeof(name), "%s.final_layer_norm", prefix);
  if (poly_model_norm_parameters(m, name, c->dim, true, &b->norm[0], &b->norm[1])) return false;
  if (!decoder) return true;
  snprintf(name, sizeof(name), "%s.encoder_attn", prefix);
  if (!whisper_attention_parameters(m, &b->cross, name, c->dim)) return false;
  PolyCtx *ctx = poly_model_ctx(m);
  PolyTensor **caches[] = {&b->self_cache, &b->cross_cache};
  for (int kind = 0; kind < 2; kind++) {
    *caches[kind] = poly_tensor_empty(
        ctx, POLY_FLOAT32,
        (int64_t[]){2, 1, c->dec_heads, kind ? c->audio : c->capacity, c->dim / c->dec_heads}, 5,
        poly_ctx_get_preferred_device(ctx)
    );
    snprintf(name, sizeof(name), "%s.%s_cache", prefix, kind ? "cross" : "self");
    if (!*caches[kind] ||
        poly_model_aux(m, name, *caches[kind], POLY_BIND_F_TRANSIENT_ZERO) != POLY_STATUS_OK)
      return false;
  }
  return true;
}

static PolyTensor *whisper_heads(PolyCtx *ctx, PolyTensor *x, int heads, int dim) {
  if (!x) return NULL;
  PolyUOp *n = poly_uop_shape_dim(ctx, poly_tensor_uop(x), 1);
  x = poly_tensor_reshape_uop(
      ctx, x,
      (PolyUOp *[]
      ){poly_uop_const_int(ctx, 1), n, poly_uop_const_int(ctx, heads),
        poly_uop_const_int(ctx, dim / heads)},
      4
  );
  return poly_tensor_permute(ctx, x, (int64_t[]){0, 2, 1, 3}, 4);
}

static PolyTensor *whisper_cache_plane(
    PolyCtx *ctx,
    PolyTensor *cache,
    int plane,
    const WhisperConfig *c
) {
  PolyTensor *x = poly_tensor_shrink(
      ctx, cache,
      (int64_t[][2]
      ){{plane, plane + 1}, {0, 1}, {0, c->dec_heads}, {0, c->audio}, {0, c->dim / c->dec_heads}},
      5
  );
  return poly_tensor_reshape(
      ctx, x, (int64_t[]){1, c->dec_heads, c->audio, c->dim / c->dec_heads}, 4
  );
}

static PolyTensor *whisper_attention(
    PolyCtx *ctx,
    const WhisperConfig *c,
    const WhisperAttention *a,
    PolyTensor *x,
    PolyTensor *audio,
    PolyTensor *cache,
    PolyUOp *position,
    int heads,
    bool cross,
    bool causal
) {
  if (!x) return NULL;
  PolyTensor *q = whisper_heads(
      ctx, poly_tensor_linear_apply(ctx, x, a->qkv[0][0], a->qkv[0][1]), heads, c->dim
  );
  PolyTensor *kv[2];
  for (int i = 0; i < 2; i++)
    kv[i] = cross && !audio ? whisper_cache_plane(ctx, cache, i, c)
                            : whisper_heads(
                                  ctx,
                                  poly_tensor_linear_apply(
                                      ctx, cross ? audio : x, a->qkv[i + 1][0], a->qkv[i + 1][1]
                                  ),
                                  heads, c->dim
                              );
  PolyTensor *out = position && !cross
                        ? poly_tensor_cached_sdpa(ctx, q, kv[0], kv[1], cache, position)
                        : poly_tensor_sdpa(ctx, q, kv[0], kv[1], NULL, 0, causal, 0, 0);
  out = poly_tensor_permute(ctx, out, (int64_t[]){0, 2, 1, 3}, 4);
  PolyUOp *n = poly_uop_shape_dim(ctx, poly_tensor_uop(x), 1);
  out = poly_tensor_reshape_uop(
      ctx, out, (PolyUOp *[]){poly_uop_const_int(ctx, 1), n, poly_uop_const_int(ctx, c->dim)}, 3
  );
  return poly_tensor_linear_apply(ctx, out, a->out[0], a->out[1]);
}

static PolyTensor *whisper_blocks(
    PolyCtx *ctx,
    const WhisperConfig *c,
    WhisperBlock *blocks,
    PolyTensor *x,
    PolyTensor *audio,
    PolyUOp *position,
    bool decoder
) {
  int count = decoder ? c->dec_layers : c->enc_layers;
  for (int i = 0; i < count; i++) {
    WhisperBlock *b = &blocks[i];
    PolyTensor *h = poly_tensor_layernorm_apply(ctx, x, b->self.norm[0], b->self.norm[1], -1, 1e-5);
    h = whisper_attention(
        ctx, c, &b->self, h, NULL, b->self_cache, position, decoder ? c->dec_heads : c->enc_heads,
        false, decoder && !position
    );
    x = poly_tensor_alu2(ctx, POLY_OP_ADD, x, h);
    if (decoder) {
      h = poly_tensor_layernorm_apply(ctx, x, b->cross.norm[0], b->cross.norm[1], -1, 1e-5);
      h = whisper_attention(
          ctx, c, &b->cross, h, audio, b->cross_cache, NULL, c->dec_heads, true, false
      );
      x = poly_tensor_alu2(ctx, POLY_OP_ADD, x, h);
    }
    h = poly_tensor_layernorm_apply(ctx, x, b->norm[0], b->norm[1], -1, 1e-5);
    h = poly_tensor_gelu_exact(ctx, poly_tensor_linear_apply(ctx, h, b->fc1[0], b->fc1[1]));
    h = poly_tensor_linear_apply(ctx, h, b->fc2[0], b->fc2[1]);
    x = poly_tensor_contiguous(ctx, poly_tensor_alu2(ctx, POLY_OP_ADD, x, h));
    if (!x) return NULL;
  }
  return x;
}

static PolyTensor *whisper_decode(
    PolyCtx *ctx,
    const WhisperConfig *c,
    WhisperBlock *blocks,
    PolyTensor *tokens,
    PolyTensor *embedding,
    PolyTensor *positions,
    PolyTensor **norm,
    PolyTensor *audio,
    PolyUOp *position
) {
  PolyUOp *zero = poly_uop_const_int(ctx, 0),
          *n = poly_uop_shape_dim(ctx, poly_tensor_uop(tokens), 1);
  PolyTensor *pos = poly_tensor_shrink_uop(
      ctx, positions, (PolyUOp *[]){position ? position : zero, zero},
      (PolyUOp *[]){n, poly_uop_const_int(ctx, c->dim)}, 2
  );
  PolyTensor *x =
      poly_tensor_alu2(ctx, POLY_OP_ADD, poly_tensor_embedding_apply(ctx, tokens, embedding), pos);
  x = whisper_blocks(ctx, c, blocks, x, audio, position, true);
  x = poly_tensor_layernorm_apply(ctx, x, norm[0], norm[1], -1, 1e-5);
  return poly_tensor_linear_apply(ctx, x, embedding, NULL);
}

PolyModel *model_whisper_from_config(PolyCtx *ctx, const cJSON *j, PolyModelError *err) {
  WhisperConfig c;
  if (!whisper_config(j, &c, err)) return NULL;
  WhisperBlock *enc = calloc((size_t)c.enc_layers, sizeof(*enc));
  WhisperBlock *dec = calloc((size_t)c.dec_layers, sizeof(*dec));
  PolyTensor **states = calloc((size_t)c.dec_layers, sizeof(*states));
  int *mutable = calloc((size_t)c.dec_layers, sizeof(*mutable));
  PolyModel *m = poly_model_new(ctx, NULL);
  if (!m || !enc || !dec || !states || !mutable) goto fail;
  for (int i = 0; i < c.enc_layers; i++)
    if (!whisper_block_parameters(m, &enc[i], &c, i, false)) goto fail;
  for (int i = 0; i < c.dec_layers; i++)
    if (!whisper_block_parameters(m, &dec[i], &c, i, true)) goto fail;
  PolyTensor *features =
      poly_model_input(m, "input_features", POLY_FLOAT32, (int64_t[]){1, c.mels, 2 * c.audio}, 3);
  PolyTensor *x = features;
  for (int i = 0; i < 2; i++) {
    char name[96];
    snprintf(name, sizeof(name), "model.encoder.conv%d.weight", i + 1);
    PolyTensor *w =
        poly_model_param(m, name, POLY_FLOAT32, (int64_t[]){c.dim, i ? c.dim : c.mels, 3}, 3);
    snprintf(name, sizeof(name), "model.encoder.conv%d.bias", i + 1);
    PolyTensor *b = poly_model_param(m, name, POLY_FLOAT32, (int64_t[]){c.dim}, 1);
    x = poly_tensor_conv2d(
        ctx, x, w, b, 1, (int64_t[]){i + 1}, (int64_t[]){1}, (int64_t[]){1, 1}, 2
    );
    x = poly_tensor_gelu_exact(ctx, x);
  }
  x = poly_tensor_permute(ctx, x, (int64_t[]){0, 2, 1}, 3);
  PolyTensor *ap = poly_model_param(
      m, "model.encoder.embed_positions.weight", POLY_FLOAT32, (int64_t[]){c.audio, c.dim}, 2
  );
  x = whisper_blocks(ctx, &c, enc, poly_tensor_alu2(ctx, POLY_OP_ADD, x, ap), NULL, NULL, false);
  PolyTensor *audio = poly_model_layernorm(m, "model.encoder.layer_norm", x, c.dim, 1e-5);
  PolyTensor *embedding =
      poly_model_embedding_parameters(m, "model.decoder.embed_tokens", c.vocab, c.dim);
  PolyTensor *positions = poly_model_param(
      m, "model.decoder.embed_positions.weight", POLY_FLOAT32, (int64_t[]){c.capacity, c.dim}, 2
  );
  PolyTensor *norm[2];
  if (!audio || !embedding || !positions ||
      poly_model_norm_parameters(m, "model.decoder.layer_norm", c.dim, true, &norm[0], &norm[1]))
    goto fail;
  PolyTensor *tokens =
      poly_model_input(m, "decoder_input_ids", POLY_INT32, (int64_t[]){1, c.length}, 2);
  PolyTensor *full = whisper_decode(ctx, &c, dec, tokens, embedding, positions, norm, audio, NULL);
  if (!full || poly_model_output(m, "logits", full) != POLY_STATUS_OK ||
      poly_model_entrypoint(
          m, "forward", (const char *[]){"input_features", "decoder_input_ids"}, 2,
          (const char *[]){"logits"}, 1, NULL
      ) != POLY_STATUS_OK)
    goto fail;

  /* Wrap all cross-cache writes into encode's output, then restore construction
   * roots. Other entrypoints must read the declared state, not replay encode. */
  PolyTensorCapture *capture = poly_tensor_capture_begin(ctx);
  if (!capture) goto fail;
  bool ok = true;
  for (int i = 0; i < c.dec_layers && ok; i++) {
    PolyTensor *kv[2];
    for (int k = 0; k < 2; k++)
      kv[k] = whisper_heads(
          ctx,
          poly_tensor_linear_apply(
              ctx, audio, dec[i].cross.qkv[k + 1][0], dec[i].cross.qkv[k + 1][1]
          ),
          c.dec_heads, c.dim
      );
    PolyTensor *packed = poly_tensor_stack(ctx, kv, 2, 0);
    states[i] = dec[i].cross_cache;
    mutable[i] = 1;
    ok = packed && poly_tensor_assign(ctx, states[i], packed);
  }
  PolyTensor *encoded = NULL;
  if (ok)
    ok = poly_tensor_capture_wrap(capture, states, mutable, c.dec_layers, &audio, 1, &encoded) == 0;
  poly_tensor_capture_end(capture);
  if (!ok || !encoded || poly_model_output(m, "audio_features", encoded) != POLY_STATUS_OK ||
      poly_model_entrypoint(
          m, "encode", (const char *[]){"input_features"}, 1, (const char *[]){"audio_features"}, 1,
          NULL
      ) != POLY_STATUS_OK)
    goto fail;
  PolyTensor *decode = NULL;
  PolyUOp *decode_pos = NULL;
  for (int step = 0; step < 2; step++) {
    const char *entry = step ? "decode" : "prefill";
    const char *input = step ? "tokens_decode" : "tokens_prefill";
    const char *output = step ? "logits_decode" : "logits_prefill";
    char variable[64];
    snprintf(variable, sizeof(variable), "%s.start_pos", entry);
    PolyUOp *pos = poly_uop_variable(
        ctx, variable, poly_arg_int(0), poly_arg_int(c.capacity - 1), POLY_WEAKINT, 1, false
    );
    PolyUOp *n = step ? poly_uop_const_int(ctx, 1)
                      : poly_uop_variable(
                            ctx, "prefill.toks", poly_arg_int(1), poly_arg_int(c.length),
                            POLY_WEAKINT, 1, false
                        );
    PolyTensor *ids =
        poly_model_input_uop(m, input, POLY_INT32, (PolyUOp *[]){poly_uop_const_int(ctx, 1), n}, 2);
    PolyTensor *out = whisper_decode(ctx, &c, dec, ids, embedding, positions, norm, NULL, pos);
    if (step) out = poly_tensor_reshape(ctx, out, (int64_t[]){1, c.vocab}, 2);
    if (!out || poly_model_output(m, output, out) != POLY_STATUS_OK ||
        poly_model_entrypoint(m, entry, &input, 1, &output, 1, NULL) != POLY_STATUS_OK ||
        poly_model_control(m, entry, "start_pos", pos) != POLY_STATUS_OK)
      goto fail;
    if (step) {
      decode = out;
      decode_pos = pos;
    }
  }
  if (model_decoder_sampler(m, c.vocab, decode, decode_pos) ||
      poly_model_build(m, err) != POLY_STATUS_OK || poly_model_require_weights(m))
    goto fail;
  free(enc);
  free(dec);
  free(states);
  free(mutable);
  return m;
fail:
  model_factory_error(err, "Whisper", "graph construction failed");
  free(enc);
  free(dec);
  free(states);
  free(mutable);
  poly_model_free(m);
  return NULL;
}

PolyModel *model_whisper_from_hf_decoded(
    const PolyHfDecoded *hf,
    const PolyGenericImportOpts *opts
) {
  if (!hf || !opts) return NULL;
  PolyModelError err = {0};
  cJSON *j = cJSON_Duplicate(hf->config, true);
  if (!j) return NULL;
  if (opts->max_seq_len > 0) {
    cJSON_DeleteItemFromObject(j, "max_seq_len");
    cJSON_AddNumberToObject(j, "max_seq_len", opts->max_seq_len);
  }
  if (opts->max_batch > 0) {
    cJSON_DeleteItemFromObject(j, "batch_size");
    cJSON_AddNumberToObject(j, "batch_size", opts->max_batch);
  }
  PolyModelFactoryScope scope;
  PolyModel *m = NULL;
  if (model_factory_begin(&scope, opts->ctx, opts->device))
    m = model_factory_end(&scope, model_whisper_from_config(scope.ctx, j, &err));
  cJSON_Delete(j);
  if (!m) {
    poly_import_error_set(POLY_IMPORT_ERR_UNSUPPORTED_MODEL, "%s", err.message);
    return NULL;
  }
  int count = poly_model_buf_count(m);
  bool *seen = calloc((size_t)count, sizeof(bool));
  PolyBindIndex *index = poly_bind_index_create_planned(m, opts->plan);
  if (!seen || !index) goto fail;
  const PolyDecodedTensor *embedding = NULL;
  for (int i = 0; i < hf->n_tensors; i++)
    if (!strcmp(hf->tensors[i].name, "model.decoder.embed_tokens.weight"))
      embedding = &hf->tensors[i];
  for (int i = 0; i < hf->n_tensors; i++) {
    const PolyDecodedTensor *t = &hf->tensors[i];
    /* Safetensors omits this alias; PyTorch state dictionaries may include it.
     * Do not silently discard a genuinely different output projection. */
    if (!strcmp(t->name, "proj_out.weight")) {
      size_t width = t->dtype == POLY_DECODED_F32                                    ? 4
                     : t->dtype == POLY_DECODED_F64                                  ? 8
                     : t->dtype == POLY_DECODED_F16 || t->dtype == POLY_DECODED_BF16 ? 2
                                                                                     : 0;
      if (!embedding || !width || t->ndim != 2 || embedding->ndim != 2 ||
          t->dtype != embedding->dtype || t->numel != embedding->numel ||
          t->shape[0] != embedding->shape[0] || t->shape[1] != embedding->shape[1] ||
          (uint64_t)t->numel > SIZE_MAX / width ||
          !poly_import_tensors_equal(index, t, embedding, false)) {
        poly_import_error_set(
            POLY_IMPORT_ERR_WEIGHT_MISMATCH,
            "Whisper proj_out.weight must match tied decoder embeddings"
        );
        goto fail;
      }
      continue;
    }
    int b = poly_bind_index_find(index, t->name);
    if (b < 0 || poly_model_buf_role(m, b) != POLY_ROLE_PARAM || seen[b] ||
        poly_import_bind_tensor(index, t->name, t, 0, -1) != 1) {
      poly_import_error_set(
          POLY_IMPORT_ERR_WEIGHT_MISMATCH, "invalid, unexpected or duplicate Whisper weight '%s'",
          t->name
      );
      goto fail;
    }
    seen[b] = true;
  }
  for (int b = 0; b < count; b++)
    if (poly_model_buf_role(m, b) == POLY_ROLE_PARAM && !seen[b]) {
      poly_import_error_set(
          POLY_IMPORT_ERR_WEIGHT_MISMATCH, "missing Whisper weight '%s'", poly_model_buf_name(m, b)
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
    poly_import_error_set(POLY_IMPORT_ERR_INTERNAL, "Whisper checkpoint allocation failed");
  return NULL;
}
