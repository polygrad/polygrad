/*
 * bind.c -- Bind index and tensor copy helper
 */

#define _POSIX_C_SOURCE 200809L
#include "bind.h"
#include "import_error.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>

#ifdef POLY_TESTING
static int bind_alloc_fail_after = -1;
void poly_test_bind_alloc_fail_after(int count) {
  bind_alloc_fail_after = count;
}
#endif

static void *bind_calloc(size_t count, size_t size) {
#ifdef POLY_TESTING
  if (bind_alloc_fail_after == 0) {
    bind_alloc_fail_after = -1;
    return NULL;
  }
  if (bind_alloc_fail_after > 0) bind_alloc_fail_after--;
#endif
  return calloc(count, size);
}

typedef struct {
  const char *name;
  int buf_idx;
} BindEntry;

struct PolyBindIndex {
  PolyModel *inst;
  BindEntry *entries;
  int capacity;
};

static unsigned int hash_name(const char *s, int cap) {
  unsigned int h = 2166136261u;
  for (; *s; s++)
    h = (h ^ (unsigned char)*s) * 16777619u;
  return h % (unsigned int)cap;
}

PolyBindIndex *poly_bind_index_create(PolyModel *inst) {
  if (!inst) return NULL;
  int n = poly_model_buf_count(inst);
  if (n < 0 || n > INT_MAX / 2) return NULL;
  int cap = n < 8 ? 16 : n * 2;
  if ((size_t)cap > SIZE_MAX / sizeof(BindEntry)) return NULL;

  PolyBindIndex *idx = bind_calloc(1, sizeof(PolyBindIndex));
  if (!idx) {
    poly_import_error_set(POLY_IMPORT_ERR_INTERNAL, "binding index allocation failed");
    return NULL;
  }
  idx->inst = inst;
  idx->capacity = cap;
  idx->entries = bind_calloc((size_t)cap, sizeof(BindEntry));
  if (!idx->entries) {
    poly_bind_index_destroy(idx);
    poly_import_error_set(POLY_IMPORT_ERR_INTERNAL, "binding table allocation failed");
    return NULL;
  }

  for (int i = 0; i < n; i++) {
    const char *name = poly_model_buf_name(inst, i);
    if (!name) continue;
    unsigned int h = hash_name(name, cap);
    while (idx->entries[h].name != NULL)
      h = (h + 1) % (unsigned int)cap;
    idx->entries[h].name = name;
    idx->entries[h].buf_idx = i;
  }

  return idx;
}

void poly_bind_index_destroy(PolyBindIndex *idx) {
  if (!idx) return;
  free(idx->entries);
  free(idx);
}

int poly_bind_index_find(const PolyBindIndex *idx, const char *name) {
  if (!idx || !name) return -1;
  unsigned int h = hash_name(name, idx->capacity);
  for (int probe = 0; probe < idx->capacity; probe++) {
    const BindEntry *e = &idx->entries[h];
    if (e->name == NULL) return -1;
    if (strcmp(e->name, name) == 0) return e->buf_idx;
    h = (h + 1) % (unsigned int)idx->capacity;
  }
  return -1;
}

int poly_bind_index_dst_shape(
    const PolyBindIndex *idx,
    const char *name,
    int64_t *shape_out,
    int max_dims
) {
  int bi = poly_bind_index_find(idx, name);
  if (bi < 0) return 0;
  return poly_model_buf_shape(idx->inst, bi, shape_out, max_dims);
}

int poly_import_bind_tensor(
    PolyBindIndex *idx,
    const char *name,
    const PolyDecodedTensor *tensor,
    int transpose_2d,
    int crop_axis
) {
  float *data = poly_decoded_tensor_to_f32(tensor);
  if (!data) {
    poly_import_error_set(
        POLY_IMPORT_ERR_WEIGHT_MISMATCH, "failed to convert weight '%s' (dtype=%d)", tensor->name,
        tensor->dtype
    );
    return -1;
  }
  int rc = poly_import_copy_named_tensor(
      idx, name, data, tensor->shape, tensor->ndim, transpose_2d, crop_axis
  );
  free(data);
  return rc;
}

int poly_import_copy_named_tensor(
    PolyBindIndex *idx,
    const char *dst_name,
    const float *src_data,
    const int64_t *src_shape,
    int src_ndim,
    int transpose_2d,
    int crop_axis
) {
  int bi = poly_bind_index_find(idx, dst_name);
  if (bi < 0) return 0; /* not found */

  size_t dst_bytes = poly_model_buf_nbytes(idx->inst, bi);
  if (poly_model_buf_dtype_id(idx->inst, bi) != poly_dtype_id_by_name("float32") || !src_data ||
      !src_shape || src_ndim < 0 || src_ndim > POLY_MAX_DIMS) {
    poly_import_error_set(POLY_IMPORT_ERR_INTERNAL, "invalid F32 binding data for '%s'", dst_name);
    return -1;
  }
  int64_t dst_numel = (int64_t)(dst_bytes / sizeof(float));

  int64_t src_numel = 1;
  for (int d = 0; d < src_ndim; d++) {
    if (src_shape[d] < 0 || (src_shape[d] && src_numel > INT64_MAX / src_shape[d])) {
      poly_import_error_set(POLY_IMPORT_ERR_SHAPE_MISMATCH, "invalid shape for '%s'", dst_name);
      return -1;
    }
    src_numel *= src_shape[d];
  }

  int64_t dst_shape[POLY_MAX_DIMS];
  int dst_ndim = poly_model_buf_shape(idx->inst, bi, dst_shape, POLY_MAX_DIMS);
  /* nn.state.load_state_dict checks shapes, not only storage size. Cropping
   * is a model-adapter decision: never infer it from an oversized source. */
  bool valid = src_ndim == dst_ndim && crop_axis >= -1 && crop_axis <= 0 &&
               !(transpose_2d && (src_ndim != 2 || crop_axis != -1));
  for (int d = 0; valid && d < dst_ndim; d++) {
    int64_t extent = src_shape[transpose_2d ? 1 - d : d];
    valid = d == crop_axis ? extent >= dst_shape[d] : extent == dst_shape[d];
  }
  if (!valid) {
    poly_import_error_set(
        POLY_IMPORT_ERR_SHAPE_MISMATCH, "shape mismatch for '%s' (transpose=%d, crop_axis=%d)",
        dst_name, transpose_2d, crop_axis
    );
    return -1;
  }

  if (transpose_2d) {
    if (src_ndim != 2 || dst_ndim != 2) {
      poly_import_error_set(
          POLY_IMPORT_ERR_SHAPE_MISMATCH,
          "transpose requires 2D tensors, got src=%dD dst=%dD for '%s'", src_ndim, dst_ndim,
          dst_name
      );
      return -1;
    }
    if (src_numel != dst_numel) {
      poly_import_error_set(
          POLY_IMPORT_ERR_SHAPE_MISMATCH, "numel mismatch for '%s': src=%lld dst=%lld", dst_name,
          (long long)src_numel, (long long)dst_numel
      );
      return -1;
    }
    float *dst_data = malloc(dst_bytes ? dst_bytes : 1);
    if (!dst_data) {
      poly_import_error_set(
          POLY_IMPORT_ERR_INTERNAL, "transpose allocation failed for '%s'", dst_name
      );
      return -1;
    }
    int64_t R = src_shape[0], C = src_shape[1];
    for (int64_t r = 0; r < R; r++)
      for (int64_t c = 0; c < C; c++)
        dst_data[c * R + r] = src_data[r * C + c];
    int rc = dst_bytes ? poly_model_write_buf(idx->inst, bi, dst_data, dst_bytes) : 0;
    free(dst_data);
    if (rc == 0) return 1;
    poly_import_error_set(POLY_IMPORT_ERR_INTERNAL, "buffer write failed for '%s'", dst_name);
    return -1;
  }

  if (dst_numel <= src_numel) {
    /* Only a validated leading-axis crop is contiguous in source storage. */
    if (!dst_bytes || poly_model_write_buf(idx->inst, bi, src_data, dst_bytes) == 0) return 1;
    poly_import_error_set(POLY_IMPORT_ERR_INTERNAL, "buffer write failed for '%s'", dst_name);
    return -1;
  }

  poly_import_error_set(
      POLY_IMPORT_ERR_SHAPE_MISMATCH, "numel mismatch for '%s': src=%lld < dst=%lld", dst_name,
      (long long)src_numel, (long long)dst_numel
  );
  return -1;
}
