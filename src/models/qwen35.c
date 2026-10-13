/* Qwen3.5 text prefill. Shared NN programs own attention/convolution math;
 * this adapter owns checkpoint names, hybrid layout and Qwen's norm/gate order.
 * Reference: Transformers Qwen3_5TextModel; pinned tinygrad/llm/model.py GDN.
 * Full attention may be bidirectional. Delta attention is always causal. */
#include "factory.h"
#include "layers.h"
#include "registry.h"
#include "../loaders/bind.h"
#include "../loaders/import_error.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  int dim, hidden, vocab, layers, heads, kv, hd, kh, vh, kd, vd, conv, batch, length, rotary;
  int capacity;
  bool causal, linear[256], composite, lm, tied;
  const char *prefix, *vision_prefix;
  double eps, theta;
} Qwen35Config;

typedef struct {
  char name[192];
  PolyTensor *tensor;
} Qwen35Parameter;

typedef struct {
  PolyCtx *ctx;
  PolyModel *model;
  const PolyHfDecoded *weights;
  const char *prefix;
  Qwen35Parameter *parameters;
  int n_parameters, max_parameters;
  PolyTensor *freq, *conv[256], *delta[256], *kv[256];
} Qwen35Build;

static bool config(const cJSON *root, Qwen35Config *c, PolyModelError *err) {
  memset(c, 0, sizeof(*c));
  const cJSON *j = cJSON_GetObjectItemCaseSensitive(root, "text_config");
  c->composite = j != NULL;
  if (!j) j = root;
  if (!cJSON_IsObject(j)) return model_factory_error(err, "text_config", "expected object");
  const cJSON *architectures = cJSON_GetObjectItemCaseSensitive(root, "architectures");
  if (architectures && !cJSON_IsNull(architectures)) {
    const cJSON *arch = cJSON_GetArrayItem(architectures, 0);
    if (!cJSON_IsArray(architectures) || cJSON_GetArraySize(architectures) != 1 ||
        !cJSON_IsString(arch))
      return model_factory_error(err, "architectures", "expected one Qwen3.5 architecture");
    c->lm = !strcmp(arch->valuestring, "Qwen3_5ForConditionalGeneration");
    if (!c->lm && strcmp(arch->valuestring, c->composite ? "Qwen3_5Model" : "Qwen3_5TextModel"))
      return model_factory_error(err, "architectures", "unsupported Qwen3.5 architecture");
    if (c->lm && !c->composite)
      return model_factory_error(err, "text_config", "conditional generation needs text_config");
  }
  const cJSON *selection = cJSON_GetObjectItemCaseSensitive(root, "modalities");
  const cJSON *vision = cJSON_GetObjectItemCaseSensitive(root, "vision_config");
  if (c->composite || (vision && !cJSON_IsNull(vision)) || selection) {
    const cJSON *text = cJSON_GetArrayItem(selection, 0);
    if (!c->composite || !cJSON_IsArray(selection) || cJSON_GetArraySize(selection) != 1 ||
        !cJSON_IsString(text) || strcmp(text->valuestring, "text"))
      return model_factory_error(
          err, "modalities", "Qwen3.5 currently requires explicit [\"text\"] for composite models"
      );
  }
  c->prefix = c->composite ? (c->lm ? "model.language_model." : "language_model.") : "";
  c->vision_prefix = c->lm ? "model.visual." : "visual.";
  const cJSON *tied = cJSON_GetObjectItemCaseSensitive(root, "tie_word_embeddings");
  if (!tied) tied = cJSON_GetObjectItemCaseSensitive(j, "tie_word_embeddings");
  if (tied && !cJSON_IsBool(tied))
    return model_factory_error(err, "tie_word_embeddings", "expected boolean");
  c->tied = tied && cJSON_IsTrue(tied);
#define DIM(field, key, max)                                                                       \
  if (!model_config_integer(j, key, 1, max, true, err)) return false;                              \
  c->field = cJSON_GetObjectItemCaseSensitive(j, key)->valueint
  DIM(dim, "hidden_size", 65536);
  DIM(hidden, "intermediate_size", 262144);
  DIM(vocab, "vocab_size", 1048576);
  DIM(layers, "num_hidden_layers", 256);
  DIM(heads, "num_attention_heads", 256);
  DIM(kv, "num_key_value_heads", 256);
  DIM(hd, "head_dim", 1024);
  DIM(kh, "linear_num_key_heads", 256);
  DIM(vh, "linear_num_value_heads", 256);
  DIM(kd, "linear_key_head_dim", 1024);
  DIM(vd, "linear_value_head_dim", 1024);
  DIM(conv, "linear_conv_kernel_dim", 64);
#undef DIM
  if (!model_config_integer(root, "batch_size", 1, 1024, false, err) ||
      !model_config_integer(root, "max_seq_len", 1, 32768, false, err) ||
      !model_config_choice(j, "hidden_act", "|silu|", err))
    return false;
  const cJSON *v = cJSON_GetObjectItemCaseSensitive(root, "batch_size");
  c->batch = v ? v->valueint : 1;
  v = cJSON_GetObjectItemCaseSensitive(root, "max_seq_len");
  c->length = v ? v->valueint : 32;
  v = cJSON_GetObjectItemCaseSensitive(j, "attention_bias");
  if (v && !cJSON_IsFalse(v))
    return model_factory_error(err, "attention_bias", "only false is supported");
  v = cJSON_GetObjectItemCaseSensitive(j, "is_causal");
  if (v && !cJSON_IsBool(v)) return model_factory_error(err, "is_causal", "expected boolean");
  c->causal = !v || cJSON_IsTrue(v);
  if (!model_config_integer(root, "cache_capacity", 0, 65536, false, err)) return false;
  v = cJSON_GetObjectItemCaseSensitive(root, "cache_capacity");
  c->capacity = v ? v->valueint : 0;
  if (c->capacity && (!c->causal || c->capacity < c->length))
    return model_factory_error(
        err, "cache_capacity", "requires causal attention and capacity >= max_seq_len"
    );
  v = cJSON_GetObjectItemCaseSensitive(j, "rms_norm_eps");
  c->eps = v ? v->valuedouble : 1e-6;
  if ((v && !cJSON_IsNumber(v)) || !isfinite(c->eps) || c->eps <= 0)
    return model_factory_error(err, "rms_norm_eps", "expected finite positive number");
  const cJSON *rope = cJSON_GetObjectItemCaseSensitive(j, "rope_parameters");
  if (!cJSON_IsObject(rope) || !model_config_choice(rope, "rope_type", "|default|", err))
    return model_factory_error(err, "rope_parameters", "expected default rotary configuration");
  v = cJSON_GetObjectItemCaseSensitive(rope, "rope_theta");
  c->theta = v ? v->valuedouble : 10000000.;
  if ((v && !cJSON_IsNumber(v)) || !isfinite(c->theta) || c->theta <= 0)
    return model_factory_error(err, "rope_theta", "expected finite positive number");
  v = cJSON_GetObjectItemCaseSensitive(rope, "partial_rotary_factor");
  double factor = v ? v->valuedouble : .25;
  if ((v && !cJSON_IsNumber(v)) || !isfinite(factor) || factor <= 0 || factor > 1 ||
      c->hd * factor != floor(c->hd * factor))
    return model_factory_error(
        err, "partial_rotary_factor", "must give an integral even head width"
    );
  c->rotary = (int)(c->hd * factor);
  if (!c->rotary || c->rotary % 2 || c->heads % c->kv || c->vh % c->kh)
    return model_factory_error(err, "heads", "invalid rotary width or grouped head counts");
  const cJSON *types = cJSON_GetObjectItemCaseSensitive(j, "layer_types");
  if (!cJSON_IsArray(types) || cJSON_GetArraySize(types) != c->layers)
    return model_factory_error(err, "layer_types", "expected one attention type per layer");
  bool full_attention = false;
  for (int i = 0; i < c->layers; i++) {
    v = cJSON_GetArrayItem(types, i);
    if (!cJSON_IsString(v) ||
        (strcmp(v->valuestring, "linear_attention") && strcmp(v->valuestring, "full_attention")))
      return model_factory_error(err, "layer_types", "expected linear_attention or full_attention");
    c->linear[i] = !strcmp(v->valuestring, "linear_attention");
    full_attention |= !c->linear[i];
  }
  if (c->capacity && !full_attention)
    return model_factory_error(
        err, "cache_capacity", "cached hybrid entrypoints require a full-attention layer"
    );
  return true;
}

/* Unlike the legacy F32 factories, every declaration consults source metadata.
 * No checkpoint data is retained by these Tensor handles or converted here. */
static PolyTensor *parameter(
    Qwen35Build *b,
    const char *base,
    const char *suffix,
    const int64_t *shape,
    int ndim
) {
  char name[192];
  const char *prefix = !strcmp(base, "lm_head") ? "" : b->prefix;
  if (snprintf(name, sizeof(name), "%s%s%s", prefix, base, suffix) >= (int)sizeof(name))
    return NULL;
  /* Authoring-only handles: entrypoints share each declared weight, not copies.
   * The factory scope owns their lifetime; this table is freed before returning. */
  for (int i = 0; i < b->n_parameters; i++)
    if (!strcmp(b->parameters[i].name, name)) return b->parameters[i].tensor;
  if (b->n_parameters == b->max_parameters) return NULL;
  PolyTensor *t = poly_import_parameter(
      b->model, b->weights ? b->weights->tensors : NULL, b->weights ? b->weights->n_tensors : 0,
      name, shape, ndim
  );
  if (t) {
    Qwen35Parameter *p = &b->parameters[b->n_parameters++];
    memcpy(p->name, name, strlen(name) + 1);
    p->tensor = t;
  }
  return t;
}

static PolyTensor *linear(
    Qwen35Build *b,
    const char *base,
    const char *suffix,
    PolyTensor *x,
    int in,
    int out
) {
  char name[128];
  if (snprintf(name, sizeof(name), "%s.weight", suffix) >= (int)sizeof(name)) return NULL;
  PolyTensor *w = parameter(b, base, name, (int64_t[]){out, in}, 2);
  return w && x ? poly_tensor_contiguous(b->ctx, poly_tensor_linear_apply(b->ctx, x, w, NULL))
                : NULL;
}

static PolyTensor *norm(
    Qwen35Build *b,
    const char *base,
    const char *suffix,
    PolyTensor *x,
    int width,
    double eps,
    bool zero_centered
) {
  if (!x) return NULL;
  char name[128];
  if (snprintf(name, sizeof(name), "%s.weight", suffix) >= (int)sizeof(name)) return NULL;
  PolyTensor *w = parameter(b, base, name, (int64_t[]){width}, 1);
  if (!w) return NULL;
  PolyDType dtype = x->uop_physical->dtype;
  if (zero_centered) {
    w = poly_tensor_cast(b->ctx, w, POLY_FLOAT32);
    w = poly_tensor_alu2(b->ctx, POLY_OP_ADD, w, poly_tensor_const_like_float(b->ctx, w, 1));
    x = poly_tensor_cast(b->ctx, x, POLY_FLOAT32);
  }
  return poly_tensor_cast(b->ctx, poly_tensor_rmsnorm_apply(b->ctx, x, w, eps), dtype);
}

static PolyTensor *slice(PolyCtx *ctx, PolyTensor *x, int axis, int64_t lo, int64_t hi) {
  if (!x) return NULL;
  int rank = poly_uop_ndim(ctx, x->uop_physical);
  const int64_t *shape = poly_uop_max_shape_dims(ctx, x->uop_physical);
  int64_t pairs[POLY_MAX_DIMS][2];
  if (!shape || axis < 0 || axis >= rank) return NULL;
  for (int i = 0; i < rank; i++) {
    pairs[i][0] = 0;
    pairs[i][1] = shape[i];
  }
  pairs[axis][0] = lo;
  pairs[axis][1] = hi;
  return poly_tensor_shrink(ctx, x, pairs, rank);
}

static PolyTensor *delta(
    Qwen35Build *b,
    const Qwen35Config *c,
    const char *base,
    PolyTensor *x,
    PolyTensor *valid,
    int layer,
    bool cached
) {
  PolyCtx *ctx = b->ctx;
  int key = c->kh * c->kd, value = c->vh * c->vd, channels = 2 * key + value;
  PolyTensor *mask = poly_tensor_reshape(ctx, valid, (int64_t[]){c->batch, c->length, 1}, 3);
  x = poly_tensor_alu2(ctx, POLY_OP_MUL, x, poly_tensor_cast(ctx, mask, x->uop_physical->dtype));
  PolyTensor *qkv = linear(b, base, "linear_attn.in_proj_qkv", x, c->dim, channels);
  PolyTensor *z = linear(b, base, "linear_attn.in_proj_z", x, c->dim, value);
  PolyTensor *beta =
      poly_tensor_sigmoid(ctx, linear(b, base, "linear_attn.in_proj_b", x, c->dim, c->vh));
  PolyTensor *a = linear(b, base, "linear_attn.in_proj_a", x, c->dim, c->vh);
  PolyTensor *dt = parameter(b, base, "linear_attn.dt_bias", (int64_t[]){c->vh}, 1);
  PolyTensor *al = parameter(b, base, "linear_attn.A_log", (int64_t[]){c->vh}, 1);
  PolyTensor *w =
      parameter(b, base, "linear_attn.conv1d.weight", (int64_t[]){channels, 1, c->conv}, 3);
  if (!qkv || !z || !beta || !a || !dt || !al || !w) return NULL;
  PolyTensor *g = poly_tensor_softplus(
      ctx, poly_tensor_alu2(ctx, POLY_OP_ADD, poly_tensor_cast(ctx, a, POLY_FLOAT32), dt), 1
  );
  g = poly_tensor_alu2(
      ctx, POLY_OP_MUL, g, poly_tensor_exp(ctx, poly_tensor_cast(ctx, al, POLY_FLOAT32))
  );
  g = poly_tensor_alu2(ctx, POLY_OP_MUL, g, poly_tensor_const_like_float(ctx, g, -1));
  qkv = poly_tensor_permute(ctx, qkv, (int64_t[]){0, 2, 1}, 3);
  if (cached) {
    PolyTensor *history = c->conv > 1 ? slice(ctx, b->conv[layer], 2, 1, c->conv) : NULL;
    PolyTensor *joined = history ? poly_tensor_cat(ctx, (PolyTensor *[]){history, qkv}, 2, 2) : qkv;
    PolyTensor *next = slice(ctx, joined, 2, c->length - 1, c->length + c->conv - 1);
    if (!next || !poly_tensor_assign(ctx, b->conv[layer], next)) return NULL;
    qkv = joined;
  }
  qkv = poly_tensor_conv2d(
      ctx, qkv, w, NULL, channels, (int64_t[]){1}, (int64_t[]){1},
      (int64_t[]){cached ? 0 : c->conv - 1, cached ? 0 : c->conv - 1}, 2
  );
  qkv = poly_tensor_silu(ctx, slice(ctx, qkv, 2, 0, c->length));
  qkv = poly_tensor_permute(ctx, qkv, (int64_t[]){0, 2, 1}, 3);
  PolyTensor *q = poly_tensor_reshape(
      ctx, slice(ctx, qkv, 2, 0, key), (int64_t[]){c->batch, c->length, c->kh, 1, c->kd}, 5
  );
  PolyTensor *k = poly_tensor_reshape(
      ctx, slice(ctx, qkv, 2, key, 2 * key), (int64_t[]){c->batch, c->length, c->kh, 1, c->kd}, 5
  );
  PolyTensor *v = poly_tensor_reshape(
      ctx, slice(ctx, qkv, 2, 2 * key, channels), (int64_t[]){c->batch, c->length, c->vh, c->vd}, 4
  );
  q = poly_tensor_expand(ctx, q, (int64_t[]){c->batch, c->length, c->kh, c->vh / c->kh, c->kd}, 5);
  k = poly_tensor_expand(ctx, k, (int64_t[]){c->batch, c->length, c->kh, c->vh / c->kh, c->kd}, 5);
  q = poly_tensor_reshape(ctx, q, (int64_t[]){c->batch, c->length, c->vh, c->kd}, 4);
  k = poly_tensor_reshape(ctx, k, (int64_t[]){c->batch, c->length, c->vh, c->kd}, 4);
  PolyTensor *out = NULL, *state = NULL;
  int chunk = 1;
  while (chunk < 64 && chunk < c->length)
    chunk *= 2;
  if (poly_tensor_gated_delta_rule(
          ctx, q, k, v, g, beta, cached ? b->delta[layer] : NULL, chunk, true, &out, &state
      ))
    return NULL;
  if (cached && !poly_tensor_assign(ctx, b->delta[layer], state)) return NULL;
  out = norm(b, base, "linear_attn.norm", out, c->vd, c->eps, false);
  z = poly_tensor_reshape(ctx, z, (int64_t[]){c->batch, c->length, c->vh, c->vd}, 4);
  PolyDType dtype = v->uop_physical->dtype;
  out = poly_tensor_cast(
      ctx,
      poly_tensor_alu2(
          ctx, POLY_OP_MUL, out, poly_tensor_silu(ctx, poly_tensor_cast(ctx, z, POLY_FLOAT32))
      ),
      dtype
  );
  return linear(
      b, base, "linear_attn.out_proj",
      poly_tensor_reshape(ctx, out, (int64_t[]){c->batch, c->length, value}, 3), value, c->dim
  );
}

static PolyTensor *attention(
    Qwen35Build *b,
    const Qwen35Config *c,
    const char *base,
    PolyTensor *x,
    PolyTensor *mask,
    PolyTensor *cos,
    PolyTensor *sin,
    int layer,
    PolyUOp *position
) {
  PolyCtx *ctx = b->ctx;
  PolyTensor *qkv[3], *gate = NULL;
  const char *names[] = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"};
  for (int i = 0; i < 3; i++) {
    int heads = i ? c->kv : c->heads, width = i ? c->hd : 2 * c->hd;
    PolyTensor *v = linear(b, base, names[i], x, c->dim, heads * width);
    v = poly_tensor_reshape(ctx, v, (int64_t[]){c->batch, c->length, heads, width}, 4);
    if (!i) {
      gate = slice(ctx, v, 3, c->hd, 2 * c->hd);
      v = slice(ctx, v, 3, 0, c->hd);
    }
    if (i < 2)
      v = norm(b, base, i ? "self_attn.k_norm" : "self_attn.q_norm", v, c->hd, c->eps, true);
    v = poly_tensor_permute(ctx, v, (int64_t[]){0, 2, 1, 3}, 4);
    if (i < 2) {
      PolyTensor *rot = poly_tensor_rope(ctx, slice(ctx, v, 3, 0, c->rotary), cos, sin);
      if (c->rotary < c->hd)
        rot = poly_tensor_cat(ctx, (PolyTensor *[]){rot, slice(ctx, v, 3, c->rotary, c->hd)}, 2, 3);
      v = rot;
    }
    qkv[i] = v;
  }
  PolyTensor *out =
      position ? poly_tensor_cached_sdpa(ctx, qkv[0], qkv[1], qkv[2], b->kv[layer], position)
               : poly_tensor_sdpa(ctx, qkv[0], qkv[1], qkv[2], mask, 0, 0, 1, 0);
  out = poly_tensor_permute(ctx, out, (int64_t[]){0, 2, 1, 3}, 4);
  out = poly_tensor_alu2(ctx, POLY_OP_MUL, out, poly_tensor_sigmoid(ctx, gate));
  out = poly_tensor_reshape(ctx, out, (int64_t[]){c->batch, c->length, c->heads * c->hd}, 3);
  return linear(b, base, "self_attn.o_proj", out, c->heads * c->hd, c->dim);
}

static PolyTensor *qwen35_forward(
    Qwen35Build *b,
    const Qwen35Config *config,
    PolyTensor *ids,
    PolyTensor *valid,
    PolyTensor *positions,
    PolyUOp *position
) {
  Qwen35Config c = *config;
  PolyCtx *ctx = b->ctx;
  PolyTensor *table = parameter(b, "", "embed_tokens.weight", (int64_t[]){c.vocab, c.dim}, 2);
  if (!ids || !valid || !positions || !table) return NULL;
  PolyTensor *x = poly_tensor_embedding_apply(ctx, ids, table);
  PolyDType dtype = table->uop_physical->dtype;
  PolyTensor *pos = poly_tensor_reshape(
      ctx, poly_tensor_cast(ctx, positions, POLY_FLOAT32), (int64_t[]){c.batch, 1, c.length, 1}, 4
  );
  PolyTensor *angles = poly_tensor_alu2(ctx, POLY_OP_MUL, pos, b->freq);
  PolyTensor *cos = poly_tensor_cast(ctx, poly_tensor_cos(ctx, angles), dtype);
  PolyTensor *sin = poly_tensor_cast(ctx, poly_tensor_alu1(ctx, POLY_OP_SIN, angles), dtype);
  PolyTensor *allowed = poly_tensor_reshape(
      ctx, poly_tensor_cast(ctx, valid, POLY_BOOL), (int64_t[]){c.batch, 1, 1, c.length}, 4
  );
  if (c.causal) {
    PolyTensor *cm = poly_tensor_causal_mask(ctx, c.length);
    cm = poly_tensor_alu2(ctx, POLY_OP_CMPEQ, cm, poly_tensor_const_like_float(ctx, cm, 0));
    allowed = poly_tensor_alu2(ctx, POLY_OP_AND, allowed, cm);
  }
  PolyTensor *mask_values = poly_tensor_cast(ctx, allowed, dtype);
  /* A fully masked row must stay finite, including with F16 checkpoint weights. */
  double mask_min = poly_dtype_eq(dtype, POLY_FLOAT16) ? -65504.0 : -3e38;
  PolyTensor *mask = poly_tensor_alu3(
      ctx, POLY_OP_WHERE, allowed, poly_tensor_const_like_float(ctx, mask_values, 0),
      poly_tensor_const_like_float(ctx, mask_values, mask_min)
  );
  if (!x || !mask || !cos || !sin) return NULL;
  for (int i = 0; i < c.layers; i++) {
    char base[32];
    snprintf(base, sizeof(base), "layers.%d.", i);
    PolyTensor *h = norm(b, base, "input_layernorm", x, c.dim, c.eps, true);
    if (!h) return NULL;
    h = c.linear[i] ? delta(b, &c, base, h, valid, i, position != NULL)
                    : attention(b, &c, base, h, mask, cos, sin, i, position);
    if (!h) return NULL;
    x = poly_tensor_alu2(ctx, POLY_OP_ADD, x, h);
    h = norm(b, base, "post_attention_layernorm", x, c.dim, c.eps, true);
    PolyTensor *gate = poly_tensor_silu(ctx, linear(b, base, "mlp.gate_proj", h, c.dim, c.hidden));
    h = poly_tensor_alu2(
        ctx, POLY_OP_MUL, gate, linear(b, base, "mlp.up_proj", h, c.dim, c.hidden)
    );
    x = poly_tensor_contiguous(
        ctx,
        poly_tensor_alu2(ctx, POLY_OP_ADD, x, linear(b, base, "mlp.down_proj", h, c.hidden, c.dim))
    );
    if (!x) return NULL;
  }
  x = norm(b, "", "norm", x, c.dim, c.eps, true);
  if (c.lm) {
    PolyTensor *head =
        c.tied ? table : parameter(b, "lm_head", ".weight", (int64_t[]){c.vocab, c.dim}, 2);
    x = poly_tensor_linear_apply(ctx, x, head, NULL);
  }
  /* Host readback is F32; source weights and intermediate rounding stay typed. */
  return poly_tensor_cast(ctx, x, POLY_FLOAT32);
}

static PolyModel *qwen35_build(
    PolyCtx *ctx,
    const cJSON *j,
    const PolyHfDecoded *hf,
    PolyModelError *err
) {
  Qwen35Config c;
  if (!config(j, &c, err)) return NULL;
  Qwen35Build b = {
      .ctx = ctx,
      .model = poly_model_new(ctx, NULL),
      .weights = hf,
      .prefix = c.prefix,
      .max_parameters = c.layers * 24 + 3};
  b.parameters = calloc((size_t)b.max_parameters, sizeof(*b.parameters));
  PolyModel *m = b.model;
  if (!m || !b.parameters) goto fail;
  float frequencies[512];
  for (int i = 0; i < c.rotary / 2; i++)
    frequencies[i] = (float)pow(c.theta, -2.0 * i / c.rotary);
  b.freq = poly_model_aux_from_host(
      m, "rope.inv_freq", POLY_FLOAT32, (int64_t[]){1, 1, 1, c.rotary / 2}, 4, frequencies,
      c.rotary / 2 * sizeof(float)
  );
  int64_t shape[] = {c.batch, c.length};
  PolyTensor *ids = poly_model_input(m, "input_ids", POLY_INT32, shape, 2);
  PolyTensor *valid = poly_model_input(m, "attention_mask", POLY_INT32, shape, 2);
  PolyTensor *positions = poly_model_input(m, "position_ids", POLY_INT32, shape, 2);
  PolyTensor *x = qwen35_forward(&b, &c, ids, valid, positions, NULL);
  const char *forward_output = c.lm ? "logits" : "last_hidden_state";
  if (!x || poly_model_output(m, forward_output, x) != POLY_STATUS_OK ||
      poly_model_entrypoint(
          m, "forward", (const char *[]){"input_ids", "attention_mask", "position_ids"}, 3,
          &forward_output, 1, NULL
      ) != POLY_STATUS_OK)
    goto fail;
  if (c.capacity) {
    PolyDType dtype = b.parameters[0].tensor->uop_physical->dtype;
    PolyTensor *states[768];
    int mutable[768], count = 0;
    for (int i = 0; i < c.layers; i++) {
      const char *names[] = {"conv", "delta", "kv"};
      PolyTensor **slots[] = {&b.conv[i], &b.delta[i], &b.kv[i]};
      for (int kind = 0; kind < 3; kind++) {
        if (c.linear[i] ? kind == 2 : kind != 2) continue;
        int64_t dims[3][5] = {
            {c.batch, 2 * c.kh * c.kd + c.vh * c.vd, c.conv},
            {c.batch, c.vh, c.kd, c.vd},
            {2, c.batch, c.kv, c.capacity, c.hd}};
        PolyTensor *state = poly_tensor_empty(
            ctx, kind == 1 ? POLY_FLOAT32 : dtype, dims[kind], kind + 3,
            poly_ctx_get_preferred_device(ctx)
        );
        char name[96];
        snprintf(name, sizeof(name), "layers.%d.cache.%s", i, names[kind]);
        if (!state || poly_model_aux(m, name, state, POLY_BIND_F_TRANSIENT_ZERO) != POLY_STATUS_OK)
          goto fail;
        *slots[kind] = state;
        states[count] = state;
        mutable[count++] = 1;
      }
    }
    for (int step = 0; step < 2; step++) {
      const char *entry = step ? "decode" : "prefill";
      const char *input = step ? "tokens_decode" : "tokens_prefill";
      const char *output = c.lm ? (step ? "logits_decode" : "logits_prefill")
                                : (step ? "hidden_decode" : "hidden_prefill");
      char variable[64];
      snprintf(variable, sizeof(variable), "%s.start_pos", entry);
      Qwen35Config current = c;
      current.length = step ? 1 : c.length;
      PolyUOp *position = poly_uop_variable(
          ctx, variable, poly_arg_int(0), poly_arg_int(c.capacity - current.length), POLY_WEAKINT,
          1, false
      );
      ids = poly_model_input(m, input, POLY_INT32, (int64_t[]){c.batch, current.length}, 2);
      PolyTensor *range = poly_tensor_arange_int(
          ctx, 0, c.capacity, 1, POLY_INT32, poly_ctx_get_preferred_device(ctx)
      );
      positions = poly_tensor_shrink_uop(
          ctx, range, (PolyUOp *[]){position},
          (PolyUOp *[]){poly_uop_const_int(ctx, current.length)}, 1
      );
      positions = poly_tensor_reshape(ctx, positions, (int64_t[]){1, current.length}, 2);
      positions = poly_tensor_expand(ctx, positions, (int64_t[]){c.batch, current.length}, 2);
      valid = poly_tensor_const_like_int(ctx, ids, 1);
      PolyTensorCapture *capture = poly_tensor_capture_begin(ctx);
      if (!capture) goto fail;
      x = qwen35_forward(&b, &current, ids, valid, positions, position);
      if (c.lm && step) x = poly_tensor_reshape(ctx, x, (int64_t[]){c.batch, c.vocab}, 2);
      PolyTensor *wrapped = NULL;
      bool ok = x && !poly_tensor_capture_wrap(capture, states, mutable, count, &x, 1, &wrapped);
      poly_tensor_capture_end(capture);
      if (!ok || !wrapped || poly_model_output(m, output, wrapped) != POLY_STATUS_OK ||
          poly_model_entrypoint(m, entry, &input, 1, &output, 1, NULL) != POLY_STATUS_OK ||
          poly_model_control(m, entry, "start_pos", position) != POLY_STATUS_OK)
        goto fail;
      if (c.lm && step && c.batch == 1 && model_decoder_sampler(m, c.vocab, wrapped, position))
        goto fail;
    }
  }
  if (poly_model_build(m, err) != POLY_STATUS_OK || poly_model_require_weights(m)) goto fail;
  free(b.parameters);
  return m;
fail:
  if (m && err && !err->code && poly_model_last_error(m)->code) *err = *poly_model_last_error(m);
  model_factory_error(err, "Qwen3.5", "text graph construction failed");
  free(b.parameters);
  poly_model_free(m);
  return NULL;
}

PolyModel *model_qwen35_from_config(PolyCtx *ctx, const cJSON *j, PolyModelError *err) {
  return qwen35_build(ctx, j, NULL, err);
}

PolyModel *model_qwen35_from_hf_decoded(
    const PolyHfDecoded *hf,
    const PolyGenericImportOpts *opts
) {
  if (!hf || !opts) return NULL;
  if (!hf->tensors || hf->n_tensors <= 0) {
    poly_import_error_set(POLY_IMPORT_ERR_WEIGHT_MISMATCH, "Qwen3.5 requires text weights");
    return NULL;
  }
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
  PolyModelError err = {0};
  Qwen35Config c;
  if (!config(j, &c, &err)) {
    poly_import_error_set(POLY_IMPORT_ERR_UNSUPPORTED_MODEL, "%s", err.message);
    cJSON_Delete(j);
    return NULL;
  }
  PolyModelFactoryScope scope;
  PolyModel *m = NULL;
  if (model_factory_begin(&scope, opts->ctx, opts->device))
    m = model_factory_end(&scope, qwen35_build(scope.ctx, j, hf, &err));
  cJSON_Delete(j);
  if (!m) {
    if (poly_import_last_error_code() == POLY_IMPORT_OK)
      poly_import_error_set(
          POLY_IMPORT_ERR_UNSUPPORTED_MODEL, "%s",
          err.message[0] ? err.message : "Qwen3.5 construction failed"
      );
    return NULL;
  }
  PolyBindIndex *index = poly_bind_index_create(m);
  if (!index) goto fail;
  const PolyDecodedTensor *embedding = NULL;
  char embedding_name[192];
  snprintf(embedding_name, sizeof(embedding_name), "%sembed_tokens.weight", c.prefix);
  for (int i = 0; i < hf->n_tensors; i++)
    if (!strcmp(hf->tensors[i].name, embedding_name)) embedding = &hf->tensors[i];
  for (int i = 0; i < hf->n_tensors; i++) {
    const PolyDecodedTensor *t = &hf->tensors[i];
    /* Only the explicitly disabled tower may be omitted. Text and head names
     * stay strict; a tied alias must contain the same checkpoint bytes. */
    if (c.composite && !strncmp(t->name, c.vision_prefix, strlen(c.vision_prefix))) continue;
    if (c.lm && c.tied && !strcmp(t->name, "lm_head.weight")) {
      size_t width = t->dtype == POLY_DECODED_F32                                    ? 4
                     : t->dtype == POLY_DECODED_F16 || t->dtype == POLY_DECODED_BF16 ? 2
                                                                                     : 0;
      if (!embedding || !width || t->dtype != embedding->dtype || t->ndim != 2 ||
          embedding->ndim != 2 || t->numel != embedding->numel ||
          t->shape[0] != embedding->shape[0] || t->shape[1] != embedding->shape[1] ||
          (uint64_t)t->numel > SIZE_MAX / width ||
          memcmp(t->data, embedding->data, (size_t)t->numel * width)) {
        poly_import_error_set(
            POLY_IMPORT_ERR_WEIGHT_MISMATCH, "Qwen3.5 lm_head.weight must match tied embeddings"
        );
        poly_bind_index_destroy(index);
        goto fail;
      }
      continue;
    }
    int bi = poly_bind_index_find(index, t->name);
    if (bi < 0 || poly_model_buf_role(m, bi) != POLY_ROLE_PARAM ||
        poly_import_bind_tensor(index, t->name, t, 0, -1) != 1) {
      if (poly_import_last_error_code() == POLY_IMPORT_OK)
        poly_import_error_set(POLY_IMPORT_ERR_WEIGHT_MISMATCH, "unexpected weight '%s'", t->name);
      poly_bind_index_destroy(index);
      goto fail;
    }
  }
  poly_bind_index_destroy(index);
  return m;
fail:
  poly_model_free(m);
  return NULL;
}
