#include "transformer.h"
#include "registry.h"
#include "../../vendor/cjson/cJSON.h"
#include <string.h>
#include "layers.h"
#include "../tensor.h"
#include "../nn/nn.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static PolyTensor *intermediate(PolyCtx *ctx, const ModelTransformerConfig *c, PolyTensor *x) {
  return c->materialize_intermediates ? poly_tensor_contiguous(ctx, x) : x;
}

typedef struct {
  PolyTensor *qkv[3], *qk_norm[2], *out, *attn_norm, *ffn_norm, *gate, *up, *down, *cache;
} TransformerBlock;

/* tinygrad llm/model.py: Transformer.forward sampling. RNG capture uses the
 * same Model AUX/effect contract as Tensor-authored models, not a host sampler. */
static int transformer_sampler(PolyModel *m, int vocab, PolyTensor *decode, PolyUOp *position) {
  PolyCtx *ctx = poly_model_ctx(m);
  PolyTensor *logits =
      poly_model_input(m, "sampling.logits", POLY_FLOAT32, (int64_t[]){1, vocab}, 2);
  PolyTensor *temperature =
      poly_model_input(m, "sampling.temperature", POLY_FLOAT32, (int64_t[]){1}, 1);
  if (!logits || !temperature) return -1;
  PolyTensorCapture *capture = poly_tensor_capture_begin(ctx);
  if (!capture) return -1;
  int rc = -1;
  PolyTensor *seed = NULL, *counter = NULL, *wrapped = NULL;
  PolyTensor *epsilon = poly_tensor_const_like_float(ctx, logits, 1e-12);
  PolyTensor *uniform = poly_tensor_rand(
      ctx, (int64_t[]){1, vocab}, 2, POLY_FLOAT32, poly_ctx_get_preferred_device(ctx), 1
  );
  PolyTensor *temp = epsilon ? poly_tensor_alu2(ctx, POLY_OP_MAX, temperature, epsilon) : NULL;
  PolyTensor *u = uniform && epsilon ? poly_tensor_alu2(ctx, POLY_OP_MAX, uniform, epsilon) : NULL;
  PolyTensor *log_u = u ? poly_tensor_log(ctx, u) : NULL;
  PolyTensor *minus_one = log_u ? poly_tensor_const_like_int(ctx, log_u, -1) : NULL;
  PolyTensor *negative = minus_one ? poly_tensor_alu2(ctx, POLY_OP_MUL, log_u, minus_one) : NULL;
  PolyTensor *noise = negative ? poly_tensor_log(ctx, negative) : NULL;
  PolyTensor *scaled = temp ? poly_tensor_div(ctx, logits, temp, 0) : NULL;
  PolyTensor *scores = scaled && noise ? poly_tensor_alu2(ctx, POLY_OP_SUB, scaled, noise) : NULL;
  PolyTensor *token = scores ? poly_tensor_argmax(ctx, scores, -1, true) : NULL;
  token = token ? poly_tensor_cast(ctx, token, POLY_INT32) : NULL;
  if (!token || poly_tensor_capture_rng(capture, 0, &seed, &counter) < 0) goto done;
  if (poly_tensor_capture_wrap(
          capture, (PolyTensor *[]){seed, counter}, (int[]){0, 1}, 2, &token, 1, &wrapped
      ) != 0)
    goto done;
  if (poly_model_aux(m, "sampling.seed", seed, 0) != POLY_STATUS_OK ||
      /* Conversation resets clear KV, not the RNG stream (Tinygrad.generate). */
      poly_model_aux(m, "sampling.counter", counter, 0) != POLY_STATUS_OK ||
      poly_model_output(m, "sampling.token", wrapped) != POLY_STATUS_OK ||
      poly_model_entrypoint(
          m, "sample", (const char *[]){"sampling.logits", "sampling.temperature"}, 2,
          (const char *[]){"sampling.token"}, 1, NULL
      ) != POLY_STATUS_OK)
    goto done;
  /* Reuse the captured sampler, including its exact seed/counter identities.
   * Substituting logits fuses rollout like Tinygrad.forward without introducing
   * a second RNG stream or an owned output snapshot between Model calls. */
  PolyUOp *logical =
      poly_uop_substitute(ctx, wrapped->uop_logical, &logits->uop_logical, &decode->uop_logical, 1);
  PolyUOp *physical = poly_uop_substitute(
      ctx, wrapped->uop_physical, &logits->uop_physical, &decode->uop_physical, 1
  );
  PolyTensor *fused =
      logical && physical
          ? poly_tensor_create_with_roots(ctx, logical, physical, POLY_TENSOR_VALUE, decode->device)
          : NULL;
  if (!fused) goto done;
  bool ok = poly_model_output(m, "sampling.decode_token", fused) == POLY_STATUS_OK &&
            poly_model_entrypoint(
                m, "decode_sample", (const char *[]){"tokens_decode", "sampling.temperature"}, 2,
                (const char *[]){"sampling.decode_token"}, 1, NULL
            ) == POLY_STATUS_OK;
  poly_tensor_release(fused);
  if (!ok) goto done;
  rc = 0;
done:
  poly_tensor_capture_end(capture);
  if (seed) poly_tensor_release(seed);
  if (counter) poly_tensor_release(counter);
  if (wrapped) poly_tensor_release(wrapped);
  if (!rc && poly_model_control(m, "decode_sample", "start_pos", position) != POLY_STATUS_OK)
    rc = -1;
  return rc;
}

/* tinygrad llm/model.py: TransformerBlock._attention. Cache STORE/AFTER and
 * active-prefix attention are shared Tensor operations, not Model execution. */
static PolyTensor *attention(
    PolyModel *model,
    const ModelTransformerConfig *c,
    PolyTensor *x,
    PolyTensor *cos,
    PolyTensor *sin,
    const TransformerBlock *block,
    PolyUOp *position
) {
  PolyCtx *ctx = poly_model_ctx(model);
  int hd = c->head_dim;
  PolyUOp *n = poly_uop_shape_dim(ctx, poly_tensor_uop(x), 1);
  PolyTensor *qkv[3];
  for (int i = 0; i < 3; i++) {
    int heads = i == 0 ? c->heads : c->kv_heads;
    PolyTensor *v = intermediate(ctx, c, poly_tensor_linear_apply(ctx, x, block->qkv[i], NULL));
    v = poly_tensor_reshape_uop(
        ctx, v,
        (PolyUOp *[]
        ){poly_uop_const_int(ctx, c->batch), n, poly_uop_const_int(ctx, heads),
          poly_uop_const_int(ctx, hd)},
        4
    );
    v = poly_tensor_permute(ctx, v, (int64_t[]){0, 2, 1, 3}, 4);
    if (i < 2 && c->qk_norm)
      v = intermediate(ctx, c, poly_tensor_rmsnorm_apply(ctx, v, block->qk_norm[i], c->eps));
    qkv[i] = i < 2 ? poly_tensor_rope(ctx, v, cos, sin) : v;
    if (!qkv[i]) return NULL;
  }
  PolyTensor *mask = NULL;
  if (!position && c->materialize_intermediates) {
    mask = poly_tensor_causal_mask(ctx, c->length);
    mask = poly_tensor_reshape(ctx, mask, (int64_t[]){1, 1, c->length, c->length}, 4);
    mask = poly_tensor_contiguous(ctx, mask);
    if (!mask) return NULL;
  }
  PolyTensor *out =
      position ? poly_tensor_cached_sdpa(ctx, qkv[0], qkv[1], qkv[2], block->cache, position)
               : poly_tensor_sdpa(ctx, qkv[0], qkv[1], qkv[2], mask, 0, !mask, 1, 0);
  out = intermediate(ctx, c, out);
  out = poly_tensor_permute(ctx, out, (int64_t[]){0, 2, 1, 3}, 4);
  out = poly_tensor_reshape_uop(
      ctx, out,
      (PolyUOp *[]){poly_uop_const_int(ctx, c->batch), n, poly_uop_const_int(ctx, c->heads * hd)}, 3
  );
  return intermediate(ctx, c, poly_tensor_linear_apply(ctx, out, block->out, NULL));
}

/* Each entrypoint starts at the same declared cache identities. Building one
 * graph must not retarget those handles to its AFTER nodes.
 * tinygrad llm/model.py: FFNBlock.__call__, _feed_forward, Transformer.forward. */
static PolyTensor *transformer_forward(
    PolyModel *m,
    const ModelTransformerConfig *c,
    const TransformerBlock *blocks,
    PolyTensor *tokens,
    PolyTensor *embedding,
    PolyTensor *norm,
    PolyTensor *head,
    PolyTensor *cos,
    PolyTensor *sin,
    PolyUOp *position
) {
  PolyCtx *ctx = poly_model_ctx(m);
  PolyUOp *zero = poly_uop_const_int(ctx, 0), *one = poly_uop_const_int(ctx, 1);
  PolyUOp *n = poly_uop_shape_dim(ctx, poly_tensor_uop(tokens), 1);
  PolyUOp *starts[] = {zero, zero, position ? position : zero, zero};
  PolyUOp *sizes[] = {one, one, n, poly_uop_const_int(ctx, c->head_dim / 2)};
  if (!c->materialize_intermediates || position || c->cache_capacity > c->length) {
    cos = poly_tensor_shrink_uop(ctx, cos, starts, sizes, 4);
    sin = poly_tensor_shrink_uop(ctx, sin, starts, sizes, 4);
  }
  PolyTensor *h = poly_tensor_contiguous(ctx, poly_tensor_embedding_apply(ctx, tokens, embedding));
  if (!h || !cos || !sin) return NULL;
  for (int i = 0; i < c->layers; i++) {
    const TransformerBlock *b = &blocks[i];
    PolyTensor *x = intermediate(ctx, c, poly_tensor_rmsnorm_apply(ctx, h, b->attn_norm, c->eps));
    h = poly_tensor_alu2(ctx, POLY_OP_ADD, h, attention(m, c, x, cos, sin, b, position));
    h = intermediate(ctx, c, h);
    x = intermediate(ctx, c, poly_tensor_rmsnorm_apply(ctx, h, b->ffn_norm, c->eps));
    PolyTensor *gate = intermediate(
        ctx, c, poly_tensor_silu(ctx, poly_tensor_linear_apply(ctx, x, b->gate, NULL))
    );
    PolyTensor *up = intermediate(ctx, c, poly_tensor_linear_apply(ctx, x, b->up, NULL));
    PolyTensor *ff = poly_tensor_linear_apply(
        ctx, intermediate(ctx, c, poly_tensor_alu2(ctx, POLY_OP_MUL, gate, up)), b->down, NULL
    );
    ff = intermediate(ctx, c, ff);
    h = poly_tensor_contiguous(ctx, poly_tensor_alu2(ctx, POLY_OP_ADD, h, ff));
    if (!h) return NULL;
  }
  h = intermediate(ctx, c, poly_tensor_rmsnorm_apply(ctx, h, norm, c->eps));
  if (position) {
    h = poly_tensor_shrink_uop(
        ctx, h, (PolyUOp *[]){zero, poly_uop_sub(ctx, n, one), zero},
        (PolyUOp *[]){poly_uop_const_int(ctx, c->batch), one, poly_uop_const_int(ctx, c->dim)}, 3
    );
    h = poly_tensor_reshape(ctx, h, (int64_t[]){c->batch, c->dim}, 2);
  }
  return poly_tensor_linear_apply(ctx, h, head, NULL);
}

/* Keep this builder out of the dynamic ABI, like the registry's import adapters. */
#if defined(__GNUC__)
__attribute__((visibility("hidden")))
#endif
PolyModel *
model_transformer_build(
    PolyCtx *ctx,
    const ModelTransformerConfig *c,
    const ModelTransformerNames *names,
    PolyModelError *err
) {
  if (!c || !names || c->dim < 1 || c->hidden_dim < 1 || c->heads < 1 || c->kv_heads < 1 ||
      c->heads % c->kv_heads || c->head_dim < 1 || c->head_dim % 2 ||
      c->heads > INT_MAX / c->head_dim || c->layers < 1 || c->vocab < 1 || c->batch < 1 ||
      c->length < 1 || (c->qk_norm && c->qk_norm != c->head_dim) || !isfinite(c->eps) ||
      c->eps <= 0 || !isfinite(c->theta) || c->theta < 1 || c->cache_capacity < 0 ||
      (c->cache_capacity &&
       (c->batch != 1 || c->prefill_chunk < 1 || c->prefill_chunk > c->cache_capacity))) {
    if (err) {
      err->code = POLY_STATUS_INVALID;
      snprintf(err->message, sizeof(err->message), "invalid dense Transformer configuration");
    }
    return NULL;
  }
  PolyModel *m = poly_model_new(ctx, NULL);
  if (!m) return NULL;
  TransformerBlock *blocks = calloc((size_t)c->layers, sizeof(*blocks));
  PolyTensor *bias = NULL;
  const char *stage = "embedding and rotary state";
  if (!blocks) goto fail;
  PolyTensor *tokens =
      poly_model_input(m, names->input, POLY_INT32, (int64_t[]){c->batch, c->length}, 2);
  PolyModelRoPEConfig rope = {
      c->cache_capacity > c->length ? c->cache_capacity : c->length,
      c->head_dim,
      c->theta,
      c->factor,
      c->low_freq,
      c->high_freq,
      c->original_context};
  PolyTensor *cos = poly_model_rope_frequencies(m, names->cos, &rope, false);
  PolyTensor *sin = poly_model_rope_frequencies(m, names->sin, &rope, true);
  PolyTensor *embedding = poly_model_embedding_parameters(m, names->embedding, c->vocab, c->dim);
  if (c->tied && names->head) {
    char alias[128];
    snprintf(alias, sizeof(alias), "%s.weight", names->head);
    if (!embedding || poly_model_state(m, alias, embedding, 0) != POLY_STATUS_OK) goto fail;
  }
  if (!tokens || !embedding || !cos || !sin) goto fail;
  for (int i = 0; i < c->layers; i++) {
    stage = "attention and feed-forward block";
    char prefix[64], name[128];
    snprintf(prefix, sizeof(prefix), names->block, i);
    snprintf(name, sizeof(name), "%s.%s", prefix, names->attn_norm);
    TransformerBlock *b = &blocks[i];
    if (poly_model_norm_parameters(m, name, c->dim, false, &b->attn_norm, &bias)) goto fail;
    for (int j = 0; j < 3; j++) {
      snprintf(name, sizeof(name), "%s.%s", prefix, names->qkv[j]);
      if (poly_model_linear_parameters(
              m, name, c->dim, (j ? c->kv_heads : c->heads) * (c->head_dim), false, &b->qkv[j],
              &bias
          ))
        goto fail;
    }
    if (c->qk_norm)
      for (int j = 0; j < 2; j++) {
        snprintf(name, sizeof(name), "%s.%s", prefix, names->qk_norm[j]);
        if (poly_model_norm_parameters(m, name, c->qk_norm, false, &b->qk_norm[j], &bias))
          goto fail;
      }
    snprintf(name, sizeof(name), "%s.%s", prefix, names->out);
    if (poly_model_linear_parameters(
            m, name, c->heads * c->head_dim, c->dim, false, &b->out, &bias
        ))
      goto fail;
    snprintf(name, sizeof(name), "%s.%s", prefix, names->ffn_norm);
    if (poly_model_norm_parameters(m, name, c->dim, false, &b->ffn_norm, &bias)) goto fail;
    snprintf(name, sizeof(name), "%s.%s", prefix, names->gate);
    if (poly_model_linear_parameters(m, name, c->dim, c->hidden_dim, false, &b->gate, &bias))
      goto fail;
    snprintf(name, sizeof(name), "%s.%s", prefix, names->up);
    if (poly_model_linear_parameters(m, name, c->dim, c->hidden_dim, false, &b->up, &bias))
      goto fail;
    snprintf(name, sizeof(name), "%s.%s", prefix, names->down);
    if (poly_model_linear_parameters(m, name, c->hidden_dim, c->dim, false, &b->down, &bias))
      goto fail;
    if (c->cache_capacity) {
      snprintf(name, sizeof(name), "%s.cache_kv", prefix);
      b->cache = poly_tensor_empty(
          ctx, POLY_FLOAT32, (int64_t[]){2, c->batch, c->kv_heads, c->cache_capacity, c->head_dim},
          5, poly_ctx_get_preferred_device(ctx)
      );
      if (!b->cache ||
          poly_model_aux(m, name, b->cache, POLY_BIND_F_TRANSIENT_ZERO) != POLY_STATUS_OK)
        goto fail;
    }
  }
  PolyTensor *norm = NULL, *head = embedding;
  if (poly_model_norm_parameters(m, names->norm, c->dim, false, &norm, &bias)) goto fail;
  stage = "output projection";
  if (!c->tied &&
      poly_model_linear_parameters(m, names->head, c->dim, c->vocab, false, &head, &bias))
    goto fail;
  PolyTensor *logits =
      transformer_forward(m, c, blocks, tokens, embedding, norm, head, cos, sin, NULL);
  if (!logits || poly_model_output(m, names->output, logits) != POLY_STATUS_OK ||
      poly_model_entrypoint(
          m, "forward", (const char *[]){names->input}, 1, (const char *[]){names->output}, 1, NULL
      ) != POLY_STATUS_OK)
    goto fail;
  PolyTensor *decode_logits = NULL;
  PolyUOp *decode_position = NULL;
  if (c->cache_capacity) {
    const char *entries[] = {"prefill", "decode", "prefill_full"};
    for (int i = 0; i < 3; i++) {
      stage = entries[i];
      ModelTransformerConfig step = *c;
      step.length = i == 1 ? 1 : c->prefill_chunk;
      char input[32], output[32], variable[32];
      snprintf(input, sizeof(input), "tokens_%s", entries[i]);
      snprintf(output, sizeof(output), "logits_%s", entries[i]);
      snprintf(variable, sizeof(variable), "%s.start_pos", entries[i]);
      PolyUOp *p = poly_uop_variable(
          ctx, variable, poly_arg_int(0), poly_arg_int(c->cache_capacity - 1), POLY_WEAKINT, 1,
          false
      );
      PolyUOp *n = step.length == 1 ? poly_uop_const_int(ctx, step.length)
                                    : poly_uop_variable(
                                          ctx, i == 2 ? "prefill_full.toks" : "prefill.toks",
                                          poly_arg_int(i == 2 ? step.length : 1),
                                          poly_arg_int(step.length), POLY_WEAKINT, 1, false
                                      );
      PolyTensor *t = poly_model_input_uop(
          m, input, POLY_INT32, (PolyUOp *[]){poly_uop_const_int(ctx, c->batch), n}, 2
      );
      PolyTensor *out =
          transformer_forward(m, &step, blocks, t, embedding, norm, head, cos, sin, p);
      if (!out || poly_model_output(m, output, out) != POLY_STATUS_OK ||
          poly_model_entrypoint(
              m, entries[i], (const char *[]){input}, 1, (const char *[]){output}, 1, NULL
          ) != POLY_STATUS_OK ||
          poly_model_control(m, entries[i], "start_pos", p) != POLY_STATUS_OK)
        goto fail;
      if (i == 1) {
        decode_logits = out;
        decode_position = p;
      }
    }
  }
  stage = "sampler";
  if (c->cache_capacity && transformer_sampler(m, c->vocab, decode_logits, decode_position) != 0)
    goto fail;
  if (poly_model_build(m, err) != POLY_STATUS_OK) goto fail;
  if (poly_model_require_weights(m) != 0) goto fail;
  if (c->cache_capacity) {
    char metadata[256];
    snprintf(
        metadata, sizeof(metadata),
        "{\"transformer\":{\"version\":1,\"architecture\":\"dense_causal\","
        "\"capacity\":%d,\"chunk\":%d,\"prefix_reuse\":true}}",
        c->cache_capacity, c->prefill_chunk
    );
    if (poly_model_set_metadata(m, metadata) != 0) goto fail;
  }
  free(blocks);
  return m;
fail:
  if (err && !err->message[0]) {
    const PolyModelError *detail = poly_model_last_error(m);
    if (detail && detail->message[0])
      *err = *detail;
    else {
      err->code = POLY_STATUS_ERROR;
      snprintf(err->message, sizeof(err->message), "%s: %s", names->label, stage);
    }
  }
  free(blocks);
  poly_model_free(m);
  return NULL;
}
/* Runtime half of the causal Transformer; Tinygrad keeps this on Transformer too. */
struct PolyTransformer {
  PolyModel *model;
  struct {
    const char *name, *input, *output;
  } entries[3];
  PolyModelError error;
  PolyTensor *logits; /* Owned device snapshot for the sampler; never a host mirror. */
  int32_t token;
  bool pending_token;
  const char *controls[3];
  bool fused_decode;
  PolyModelStateVersion observed;
  int capacity, chunk, vocab, position;
  bool valid, busy;
  int32_t *tokens;
#ifdef POLY_TESTING
  int fail_after;
#endif
};

static int binding_index(const PolyModel *model, const char *name) {
  for (int i = 0; i < poly_model_buf_count(model); i++)
    if (!strcmp(poly_model_buf_name(model, i), name)) return i;
  return -1;
}

static void transformer_set_error(PolyTransformer *t, const char *func, const char *message) {
  t->error = (PolyModelError){.code = message ? POLY_STATUS_INVALID : POLY_STATUS_OK, .func = func};
  if (message) snprintf(t->error.message, sizeof(t->error.message), "%s", message);
}
const PolyModelError *poly_transformer_last_error(const PolyTransformer *t) {
  return t ? &t->error : NULL;
}
static cJSON *transformer_config(const PolyModel *model, cJSON **document) {
  const char *json = poly_model_metadata(model);
  *document = json ? cJSON_Parse(json) : NULL;
  cJSON *config = cJSON_GetObjectItemCaseSensitive(*document, "transformer");
  cJSON *v = cJSON_GetObjectItemCaseSensitive(config, "version");
  cJSON *kind = cJSON_GetObjectItemCaseSensitive(config, "architecture");
  return cJSON_IsNumber(v) && v->valuedouble == 1 && cJSON_IsString(kind) &&
                 !strcmp(kind->valuestring, "dense_causal")
             ? config
             : NULL;
}
bool poly_transformer_available(const PolyModel *model) {
  cJSON *doc = NULL;
  bool ok = transformer_config(model, &doc) != NULL;
  cJSON_Delete(doc);
  return ok;
}
PolyModel *poly_transformer_model(const PolyTransformer *t) {
  return t ? t->model : NULL;
}

PolyTransformer *poly_transformer_from_model(PolyModel *model, PolyModelError *error) {
  if (error) *error = (PolyModelError){0};
  if (poly_model_is_busy(model)) {
    if (error) {
      error->code = POLY_STATUS_BAD_STAGE;
      snprintf(error->message, sizeof(error->message), "Transformer requires an idle built Model");
    }
    return NULL;
  }
  PolyTransformer *g = calloc(1, sizeof(*g));
  cJSON *doc = NULL;
  cJSON *config = transformer_config(model, &doc);
  if (!g || !config || !poly_model_has_transient(model)) goto invalid;
  cJSON *cap = cJSON_GetObjectItemCaseSensitive(config, "capacity");
  cJSON *chunk = cJSON_GetObjectItemCaseSensitive(config, "chunk");
  cJSON *reuse = cJSON_GetObjectItemCaseSensitive(config, "prefix_reuse");
  if (!cJSON_IsNumber(cap) || cap->valuedouble < 1 || cap->valuedouble > INT_MAX ||
      cap->valuedouble != cap->valueint || !cJSON_IsNumber(chunk) || chunk->valuedouble < 1 ||
      chunk->valuedouble > cap->valuedouble || chunk->valuedouble != chunk->valueint ||
      !cJSON_IsBool(reuse))
    goto invalid;
  g->capacity = cap->valueint;
  g->chunk = chunk->valueint;
  const char *entries[] = {"prefill", "decode", "prefill_full"};
  for (int i = 0; i < 3; i++) {
    const char *e = entries[i];
    /* These are optional optimizations; portable producers may expose only
     * the original dynamic prefill/decode contract. */
    if (i == 2 && poly_model_entrypoint_input_count(model, e) < 0) continue;
    if (poly_model_entrypoint_input_count(model, e) != 1 ||
        poly_model_entrypoint_output_count(model, e) != 1 ||
        poly_model_entrypoint_objective(model, e) || poly_model_control_count(model, e) != 1)
      goto invalid;
    const char *input = poly_model_entrypoint_input_name(model, e, 0);
    const char *output = poly_model_entrypoint_output_name(model, e, 0);
    int x = binding_index(model, input), y = binding_index(model, output);
    int64_t xmin[2], xmax[2], ymin[2], ymax[2], lo, hi;
    if (x < 0 || y < 0 || poly_model_buf_shape_bounds(model, x, xmin, xmax, 2) != 2 ||
        poly_model_buf_shape_bounds(model, y, ymin, ymax, 2) != 2 || xmin[0] != 1 || xmax[0] != 1 ||
        ymin[0] != 1 || ymax[0] != 1 || xmin[1] != (i == 2 ? g->chunk : 1) ||
        xmax[1] != (i == 1 ? 1 : g->chunk) || ymin[1] < 1 || ymax[1] > INT_MAX ||
        ymin[1] != ymax[1] || (i && ymax[1] != g->vocab))
      goto invalid;
    PolyUOp *xb = poly_model_get_buffer(model, input), *yb = poly_model_get_buffer(model, output);
    if (!xb || !yb || !poly_dtype_eq(xb->dtype, POLY_INT32) ||
        !poly_dtype_eq(yb->dtype, POLY_FLOAT32) ||
        poly_model_control_bounds(model, e, 0, &lo, &hi) != 0 || lo != 0 || hi != g->capacity - 1)
      goto invalid;
    g->entries[i].name = e;
    g->entries[i].input = input;
    g->entries[i].output = output;
    g->controls[i] = poly_model_control_name(model, e, 0);
    g->vocab = (int)ymax[1];
  }
  if (poly_model_entrypoint_input_count(model, "decode_sample") >= 0) {
    const char *a = poly_model_entrypoint_input_name(model, "decode_sample", 0);
    const char *b = poly_model_entrypoint_input_name(model, "decode_sample", 1);
    const char *y = poly_model_entrypoint_output_name(model, "decode_sample", 0);
    int64_t lo, hi, shape_lo[2], shape_hi[2];
    int bi = y ? binding_index(model, y) : -1;
    PolyUOp *buffer = y ? poly_model_get_buffer(model, y) : NULL;
    if (poly_model_entrypoint_input_count(model, "decode_sample") != 2 ||
        poly_model_entrypoint_output_count(model, "decode_sample") != 1 ||
        poly_model_entrypoint_objective(model, "decode_sample") || !a ||
        strcmp(a, g->entries[1].input) || !b || strcmp(b, "sampling.temperature") || !y ||
        strcmp(y, "sampling.decode_token") || !buffer ||
        !poly_dtype_eq(buffer->dtype, POLY_INT32) ||
        poly_model_buf_shape_bounds(model, bi, shape_lo, shape_hi, 2) != 2 || shape_lo[0] != 1 ||
        shape_hi[0] != 1 || shape_lo[1] != 1 || shape_hi[1] != 1 ||
        poly_model_control_count(model, "decode_sample") != 1 ||
        strcmp(poly_model_control_name(model, "decode_sample", 0), g->controls[1]) ||
        poly_model_control_bounds(model, "decode_sample", 0, &lo, &hi) || lo != 0 ||
        hi != g->capacity - 1)
      goto invalid;
    g->fused_decode = true;
  }
  if (poly_model_entrypoint_input_count(model, "sample") != 2 ||
      poly_model_entrypoint_output_count(model, "sample") != 1 ||
      poly_model_control_count(model, "sample") != 0 ||
      poly_model_entrypoint_objective(model, "sample"))
    goto invalid;
  const char *sample_names[] = {"sampling.logits", "sampling.temperature", "sampling.token"};
  for (int i = 0; i < 3; i++) {
    const char *name = i < 2 ? poly_model_entrypoint_input_name(model, "sample", i)
                             : poly_model_entrypoint_output_name(model, "sample", 0);
    if (!name || strcmp(name, sample_names[i])) goto invalid;
    int bi = binding_index(model, name);
    int64_t lo[2], hi[2];
    int rank = i == 1 ? 1 : 2;
    if (bi < 0 || poly_model_buf_shape_bounds(model, bi, lo, hi, 2) != rank || lo[0] != 1 ||
        hi[0] != 1 || (rank == 2 && (lo[1] != (i == 0 ? g->vocab : 1) || hi[1] != lo[1])))
      goto invalid;
    PolyUOp *buffer = poly_model_get_buffer(model, name);
    if (!buffer || !poly_dtype_eq(buffer->dtype, i == 2 ? POLY_INT32 : POLY_FLOAT32)) goto invalid;
  }
  if (cJSON_IsTrue(reuse)) {
    if ((size_t)g->capacity > SIZE_MAX / sizeof(*g->tokens)) goto invalid;
    g->tokens = malloc((size_t)g->capacity * sizeof(*g->tokens));
    if (!g->tokens) goto invalid;
  }
  g->observed = poly_model_state_version(model);
  g->valid = g->observed.transient == g->observed.reset;
  g->model = model; /* Ownership transfers only after all validation succeeds. */
  cJSON_Delete(doc);
  return g;
invalid:
  cJSON_Delete(doc);
  if (g) {
    free(g->tokens);
    free(g);
  }
  if (error) {
    error->code = POLY_STATUS_INVALID;
    error->func = __func__;
    snprintf(
        error->message, sizeof(error->message),
        "Model lacks a compatible dense causal Transformer contract"
    );
  }
  return NULL;
}

void poly_transformer_free(PolyTransformer *g) {
  if (!g) return;
  free(g->tokens);
  if (g->logits) poly_tensor_release(g->logits);
  poly_model_free(g->model);
  free(g);
}
int poly_transformer_vocab(const PolyTransformer *g) {
  return g ? g->vocab : 0;
}

static void transformer_observe(PolyTransformer *g) {
  PolyModelStateVersion v = poly_model_state_version(g->model);
  if (v.reset != g->observed.reset) {
    g->position = 0;
    g->valid = v.transient == v.reset;
  } else if (v.transient != g->observed.transient || (g->position > 0 && v.mutation != g->observed.mutation)) {
    g->valid = false;
  }
  if (!g->valid || v.reset != g->observed.reset) {
    if (g->logits) poly_tensor_release(g->logits);
    g->logits = NULL;
    g->pending_token = false;
  }
  g->observed = v;
}

int poly_transformer_position(PolyTransformer *g) {
  if (!g || g->busy) return -1;
  transformer_observe(g);
  return g->valid ? g->position : -1;
}

static const char *transformer_error(PolyTransformer *g) {
  if (g->busy || poly_model_is_busy(g->model)) return "decoder requires an idle runtime";
  transformer_observe(g);
  return g->valid ? NULL : "decoder state is invalid; reset the Transformer before continuing";
}

int poly_transformer_rewind(PolyTransformer *g, int position) {
  if (!g) return -1;
  const char *error = transformer_error(g);
  if (!error && !g->tokens) error = "decoder does not support prefix reuse";
  if (!error && (position < 0 || position > g->position))
    error = "rewind position outside committed prefix";
  transformer_set_error(g, __func__, error);
  if (error) return -1;
  /* Dropped suffix bytes remain but cannot be read before replacement writes. */
  g->position = position;
  if (g->logits) poly_tensor_release(g->logits);
  g->logits = NULL;
  g->pending_token = false;
  return 0;
}

static int transformer_tokens(
    PolyTransformer *g,
    const int32_t *tokens,
    int count,
    float *logits,
    int n_logits,
    bool reuse,
    bool device_output
) {
  if (!g) return -1;
  const char *error = transformer_error(g);
  if (!error && reuse && !g->tokens) error = "decoder does not support prefix reuse";
  if (!error && (!tokens || count <= 0 || (!device_output && (!logits || n_logits != g->vocab))))
    error = "invalid token or logits buffer";
  if (!error && count > g->capacity - (reuse ? 0 : g->position))
    error = "decoder capacity exceeded";
  for (int i = 0; !error && i < count; i++)
    if (tokens[i] < 0 || tokens[i] >= g->vocab) error = "token ID outside vocabulary";
  transformer_set_error(g, __func__, error);
  if (error) return -1;
  if (poly_model_check_ready(g->model) != 0) {
    g->error = *poly_model_last_error(g->model);
    return -1;
  }
  int start = g->position;
  if (reuse) {
    /* tinygrad llm/model.py Transformer.get_start_pos: compare tokens[:-1],
     * always execute the final token even for identical/shorter prompts. */
    start = 0;
    while (start < count - 1 && start < g->position && tokens[start] == g->tokens[start])
      start++;
    tokens += start;
    count -= start;
  }
  /* Admission precedes mutation. After a partial execution failure only a
   * successful Model reset permits reuse; a rewind cannot repair state. */
  g->busy = true;
  g->valid = false;
  if (g->logits) poly_tensor_release(g->logits);
  g->logits = NULL;
  g->pending_token = false;
  int used = 0, last = 1, rc = 0;
  while (used < count) {
    int n = count - used >= g->chunk ? g->chunk : count - used;
    last = n > 1 || n == g->chunk ? 0 : 1;
    if (n == g->chunk && g->entries[2].name) last = 2;
    const char *entry = g->entries[last].name;
    PolyIOBinding io = POLY_IO_BINDING_BYTES(
        g->entries[last].input, (void *)(tokens + used), (size_t)n * sizeof(*tokens), POLY_INT32
    );
    PolyControlBinding control = {g->controls[last], start + used};
    rc = device_output && used + n == count
             ? poly_model_call_tensors_with_controls(
                   g->model, entry, &io, 1, &control, 1, &g->logits, 1
               )
             : poly_model_call_with_controls(g->model, entry, &io, 1, &control, 1);
    if (rc) {
      g->error = *poly_model_last_error(g->model);
      break;
    }
    used += n;
#ifdef POLY_TESTING
    if (g->fail_after > 0 && --g->fail_after == 0) {
      transformer_set_error(g, __func__, "injected decoder execution failure");
      rc = -1;
      break;
    }
#endif
  }
  if (!rc && !device_output)
    rc = poly_model_read_buf_named(
        g->model, g->entries[last].output, logits, (size_t)n_logits * sizeof(*logits)
    );
  g->observed = poly_model_state_version(g->model);
  if (!rc) {
    if (g->tokens) memcpy(g->tokens + start, tokens, (size_t)count * sizeof(*tokens));
    g->position = start + count;
    g->valid = true;
  }
  g->busy = false;
  return rc;
}

int poly_transformer_append(
    PolyTransformer *g,
    const int32_t *tokens,
    int count,
    float *logits,
    int n_logits
) {
  return transformer_tokens(g, tokens, count, logits, n_logits, false, false);
}
int poly_transformer_prefill(
    PolyTransformer *g,
    const int32_t *tokens,
    int count,
    float *logits,
    int n_logits
) {
  return transformer_tokens(g, tokens, count, logits, n_logits, true, false);
}
#ifdef POLY_TESTING
void poly_transformer_test_fail_after(PolyTransformer *g, int calls) {
  if (g) g->fail_after = calls;
}
#endif

int poly_transformer_reset(PolyTransformer *t) {
  if (!t || t->busy) return -1;
  int rc = poly_model_reset_transient(t->model);
  transformer_observe(t);
  if (rc)
    transformer_set_error(t, __func__, "transient state reset failed");
  else
    transformer_set_error(t, __func__, NULL);
  return rc;
}
int poly_transformer_start(PolyTransformer *t, const int32_t *tokens, int count) {
  return transformer_tokens(t, tokens, count, NULL, 0, true, true);
}

int poly_transformer_next(PolyTransformer *t, float temperature, int32_t *token) {
  if (!t) return -1;
  const char *error = transformer_error(t);
  if (!error && (!token || !isfinite(temperature) || temperature < 0))
    error = "sampling requires a finite nonnegative temperature and token output";
  if (!error && !t->logits && !t->pending_token) error = "start a prompt before requesting a token";
  transformer_set_error(t, __func__, error);
  if (error) return -1;
  /* Like Tinygrad.generate, the last emitted token is not in the cache yet. */
  if (t->position >= t->capacity || (t->pending_token && t->position == t->capacity - 1)) return 1;
  if (t->pending_token && t->fused_decode) {
    int32_t previous = t->token;
    PolyIOBinding io[] = {
        POLY_IO_BINDING_BYTES(t->entries[1].input, &previous, sizeof(previous), POLY_INT32),
        POLY_IO_BINDING_BYTES(
            "sampling.temperature", &temperature, sizeof(temperature), POLY_FLOAT32
        )};
    PolyControlBinding control = {t->controls[1], t->position};
    t->busy = true;
    int rc = poly_model_call_with_controls(t->model, "decode_sample", io, 2, &control, 1);
    if (!rc)
      rc =
          poly_model_read_buf_named(t->model, "sampling.decode_token", &t->token, sizeof(t->token));
    t->observed = poly_model_state_version(t->model);
    if (rc) t->error = *poly_model_last_error(t->model);
#ifdef POLY_TESTING
    if (!rc && t->fail_after > 0 && --t->fail_after == 0) {
      transformer_set_error(t, __func__, "injected decoder execution failure");
      rc = -1;
    }
#endif
    t->valid = rc == 0;
    if (!rc) {
      if (t->tokens) t->tokens[t->position] = previous;
      t->position++;
      *token = t->token;
    }
    t->busy = false;
    return rc;
  }
  if (t->pending_token && transformer_tokens(t, &t->token, 1, NULL, 0, false, true) != 0) return -1;
  PolyIOBinding io[] = {
      {.name = "sampling.logits", .tensor = t->logits},
      POLY_IO_BINDING_BYTES(
          "sampling.temperature", &temperature, sizeof(temperature), POLY_FLOAT32
      )};
  t->busy = true;
  int rc = poly_model_call_with_controls(t->model, "sample", io, 2, NULL, 0);
  if (!rc) rc = poly_model_read_buf_named(t->model, "sampling.token", &t->token, sizeof(t->token));
  t->observed = poly_model_state_version(t->model);
  t->valid = rc == 0;
  if (rc)
    t->error = *poly_model_last_error(t->model);
  else {
    *token = t->token;
    t->pending_token = true;
  }
  if (t->logits) poly_tensor_release(t->logits);
  t->logits = NULL;
  t->busy = false;
  return rc;
}
