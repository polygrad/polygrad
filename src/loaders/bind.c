/*
 * bind.c -- Bind index and tensor copy helper
 */

#define _POSIX_C_SOURCE 200809L
#include "bind.h"
#include "import_error.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef struct {
    const char *name;
    int buf_idx;
} BindEntry;

struct PolyBindIndex {
    PolyInstance *inst;
    BindEntry *entries;
    int capacity;
};

static unsigned int hash_name(const char *s, int cap) {
    unsigned int h = 2166136261u;
    for (; *s; s++) h = (h ^ (unsigned char)*s) * 16777619u;
    return h % (unsigned int)cap;
}

PolyBindIndex *poly_bind_index_create(PolyInstance *inst) {
    int n = poly_instance_buf_count(inst);
    int cap = n < 8 ? 16 : n * 2;

    PolyBindIndex *idx = calloc(1, sizeof(PolyBindIndex));
    idx->inst = inst;
    idx->capacity = cap;
    idx->entries = calloc((size_t)cap, sizeof(BindEntry));

    for (int i = 0; i < n; i++) {
        const char *name = poly_instance_buf_name(inst, i);
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

static int bind_find(const PolyBindIndex *idx, const char *name) {
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
    int max_dims)
{
    int bi = bind_find(idx, name);
    if (bi < 0) return 0;
    return poly_instance_buf_shape(idx->inst, bi, shape_out, max_dims);
}

int poly_import_copy_named_tensor(
    PolyBindIndex *idx,
    const char *dst_name,
    const float *src_data,
    const int64_t *src_shape,
    int src_ndim,
    int transpose_2d)
{
    int bi = bind_find(idx, dst_name);
    if (bi < 0) return 0;  /* not found */

    int64_t dst_numel;
    float *dst_data = poly_instance_buf_data(idx->inst, bi, &dst_numel);
    if (!dst_data) {
        /* Check raw data pointer to distinguish alloc failure from sync failure */
        poly_import_error_set(POLY_IMPORT_ERR_INTERNAL,
            "buffer '%s' data is NULL (bi=%d, numel=%lld)", dst_name, bi,
            (long long)dst_numel);
        return -1;
    }

    int64_t src_numel = 1;
    for (int d = 0; d < src_ndim; d++) src_numel *= src_shape[d];

    int64_t dst_shape[8];
    int dst_ndim = poly_instance_buf_shape(idx->inst, bi, dst_shape, 8);

    if (transpose_2d) {
        if (src_ndim != 2 || dst_ndim != 2) {
            poly_import_error_set(POLY_IMPORT_ERR_SHAPE_MISMATCH,
                "transpose requires 2D tensors, got src=%dD dst=%dD for '%s'",
                src_ndim, dst_ndim, dst_name);
            return -1;
        }
        if (src_numel != dst_numel) {
            poly_import_error_set(POLY_IMPORT_ERR_SHAPE_MISMATCH,
                "numel mismatch for '%s': src=%lld dst=%lld",
                dst_name, (long long)src_numel, (long long)dst_numel);
            return -1;
        }
        int64_t R = src_shape[0], C = src_shape[1];
        for (int64_t r = 0; r < R; r++)
            for (int64_t c = 0; c < C; c++)
                dst_data[c * R + r] = src_data[r * C + c];
        return 1;
    }

    if (dst_numel <= src_numel) {
        memcpy(dst_data, src_data, (size_t)dst_numel * sizeof(float));
        return 1;
    }

    poly_import_error_set(POLY_IMPORT_ERR_SHAPE_MISMATCH,
        "numel mismatch for '%s': src=%lld < dst=%lld",
        dst_name, (long long)src_numel, (long long)dst_numel);
    return -1;
}
