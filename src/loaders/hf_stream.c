#include "../models/hf_loader.h"
#include "../models/registry.h"
#include "../safetensors.h"
#include "bind.h"
#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  int file;
  uint64_t offset, length;
} Source;
typedef struct {
  const PolyDecodedTensor *a, *b;
  char *name;
  int transpose, crop;
  bool as_f32;
} Delivery;

struct PolyHfImport {
  PolyHfDecoded *hf;
  PolyGenericImportOpts opts;
  PolyImportPlan plan;
  PolyModel *model;
  PolyBindIndex *bindings;
  Source *sources;
  int files, count, current;
  Delivery *deliveries;
  void *alias;
  bool built, failed;
  char request[160];
};

static int fail(PolyHfImport *s, const char *message) {
  if (s) s->failed = true;
  if (poly_import_last_error_code() == POLY_IMPORT_OK)
    poly_import_error_set(POLY_IMPORT_ERR_PARSE, "%s", message);
  return -1;
}

static Delivery *append(PolyHfImport *s) {
  if (s->failed || s->count == INT_MAX || (size_t)(s->count + 1) > SIZE_MAX / sizeof(Delivery))
    return NULL;
  Delivery *next = realloc(s->deliveries, (size_t)(s->count + 1) * sizeof(*next));
  if (!next) {
    fail(s, "checkpoint delivery allocation failed");
    return NULL;
  }
  s->deliveries = next;
  Delivery *d = &next[s->count++];
  memset(d, 0, sizeof(*d));
  return d;
}

static int record_bind(
    void *user,
    const char *name,
    const PolyDecodedTensor *t,
    int transpose,
    int crop
) {
  PolyHfImport *s = user;
  Delivery *d = append(s);
  if (!d) return -1;
  d->a = t;
  d->transpose = transpose;
  d->crop = crop;
  d->name = malloc(strlen(name) + 1);
  if (!d->name) return fail(s, "checkpoint binding name allocation failed");
  strcpy(d->name, name);
  return 1;
}

static int record_equal(
    void *user,
    const PolyDecodedTensor *a,
    const PolyDecodedTensor *b,
    bool as_f32
) {
  Delivery *d = append(user);
  if (!d) return 0;
  d->a = a;
  d->b = b;
  d->as_f32 = as_f32;
  return 1;
}

PolyHfImport *poly_hf_import_new(
    PolyCtx *ctx,
    const char *config_json,
    int batch,
    int length,
    PolyDevice device
) {
  poly_import_error_clear();
  if (!ctx || !config_json || strlen(config_json) > INT_MAX) return NULL;
  PolyHfImport *s = calloc(1, sizeof(*s));
  if (!s) return NULL;
  if (poly_hf_decode(config_json, (int)strlen(config_json), NULL, NULL, 0, &s->hf)) {
    free(s);
    return NULL;
  }
  s->plan = (PolyImportPlan){s, record_bind, record_equal};
  s->opts = (PolyGenericImportOpts
  ){.ctx = ctx, .max_batch = batch, .max_seq_len = length, .device = device, .plan = &s->plan};
  return s;
}

int poly_hf_import_add_file(
    PolyHfImport *s,
    const uint8_t *header,
    int header_len,
    int64_t file_len
) {
  if (!s || s->built || s->failed || s->files == INT_MAX || header_len > 100000008)
    return fail(s, "invalid header import state or size");
  int n = 0;
  uint64_t *offsets = NULL;
  PolySafetensorViewEx *views = poly_safetensors_index(header, header_len, file_len, &n, &offsets);
  if (!views) return fail(s, "invalid safetensors header or file length");
  int old = s->hf->n_tensors;
  bool ok = n <= INT_MAX - old && (size_t)(old + n) <= SIZE_MAX / sizeof(PolyDecodedTensor);
  if (ok) {
    PolyDecodedTensor *t = realloc(s->hf->tensors, (size_t)(old + n ? old + n : 1) * sizeof(*t));
    if (!t)
      ok = false;
    else
      s->hf->tensors = t;
    Source *p = ok ? realloc(s->sources, (size_t)(old + n ? old + n : 1) * sizeof(*p)) : NULL;
    if (!p)
      ok = false;
    else
      s->sources = p;
  }
  if (ok)
    for (int i = 0; i < n; i++) {
      PolyDecodedTensor *t = &s->hf->tensors[old + i];
      *t = (PolyDecodedTensor
      ){.name = views[i].name,
        .ndim = views[i].ndim,
        .numel = views[i].numel,
        .dtype = (int)views[i].dtype};
      memcpy(t->shape, views[i].shape, sizeof(t->shape));
      views[i].name = NULL;
      s->sources[old + i] = (Source
      ){s->files, offsets[i], (uint64_t)t->numel * poly_safetensor_dtype_size(views[i].dtype)};
    }
  for (int i = 0; i < n; i++)
    free(views[i].name);
  free(views);
  free(offsets);
  if (!ok) return fail(s, "checkpoint index allocation failed");
  s->hf->n_tensors += n;
  s->files++;
  if (decoded_names_unique(s->hf->tensors, s->hf->n_tensors))
    return fail(s, "duplicate checkpoint name");
  return 0;
}

int poly_hf_import_build(PolyHfImport *s) {
  if (!s || s->built || s->failed || !s->files) return fail(s, "invalid checkpoint build state");
  s->built = true;
  /* No more headers may be added: adapter callbacks retain pointers into this
   * immutable index until delivery completes or the session is destroyed. */
  const PolyModelType *type = model_type_find(s->hf->model_type);
  if (!type || !type->from_hf_decoded) return fail(s, "model does not support HF import");
  s->model = type->from_hf_decoded(s->hf, &s->opts);
  if (!s->model || s->failed || poly_model_require_weights(s->model))
    return fail(s, "checkpoint graph construction failed");
  s->bindings = poly_bind_index_create(s->model);
  return s->bindings ? 0 : fail(s, "checkpoint binding allocation failed");
}

const char *poly_hf_import_request(PolyHfImport *s) {
  if (!s || !s->built || s->failed || s->current == s->count) return NULL;
  Delivery *d = &s->deliveries[s->current];
  const PolyDecodedTensor *t = s->alias ? d->b : d->a;
  Source *src = &s->sources[t - s->hf->tensors];
  snprintf(
      s->request, sizeof(s->request), "{\"file\":%d,\"offset\":%" PRIu64 ",\"length\":%" PRIu64 "}",
      src->file, src->offset, src->length
  );
  return s->request;
}

int poly_hf_import_feed(PolyHfImport *s, const uint8_t *data, size_t length) {
  if (!s || !s->built || s->failed || s->current >= s->count || (!data && length))
    return fail(s, "invalid checkpoint feed state");
  Delivery *d = &s->deliveries[s->current];
  const PolyDecodedTensor *t = s->alias ? d->b : d->a;
  if (s->sources[t - s->hf->tensors].length != length)
    return fail(s, "short or oversized checkpoint range");
  if (d->b && !s->alias) {
    s->alias = malloc(length ? length : 1);
    if (!s->alias) return fail(s, "tied checkpoint allocation failed");
    if (length) memcpy(s->alias, data, length);
    return 0;
  }
  PolyDecodedTensor value = *t;
  value.data = data;
  if (d->b) {
    PolyDecodedTensor other = *d->a;
    other.data = s->alias;
    bool same = poly_import_tensors_equal(NULL, &other, &value, d->as_f32);
    free(s->alias);
    s->alias = NULL;
    if (!same) return fail(s, "tied checkpoint weights differ");
  } else if (poly_import_bind_tensor(s->bindings, d->name, &value, d->transpose, d->crop) != 1)
    return fail(s, "checkpoint weight binding failed");
  s->current++;
  return 0;
}

PolyModel *poly_hf_import_finish(PolyHfImport *s) {
  if (!s || !s->built || s->failed || s->current != s->count || !s->model) {
    fail(s, "checkpoint delivery incomplete");
    return NULL;
  }
  if (!model_weights_ready(s->model)) {
    poly_import_error_set(
        POLY_IMPORT_ERR_WEIGHT_MISMATCH, "%s", poly_model_last_error(s->model)->message
    );
    s->failed = true;
    return NULL;
  }
  PolyModel *model = s->model;
  s->model = NULL;
  return model;
}

void poly_hf_import_free(PolyHfImport *s) {
  if (!s) return;
  poly_bind_index_destroy(s->bindings);
  poly_model_free(s->model);
  for (int i = 0; i < s->count; i++)
    free(s->deliveries[i].name);
  free(s->deliveries);
  free(s->sources);
  free(s->alias);
  poly_hf_decoded_free(s->hf);
  free(s);
}
