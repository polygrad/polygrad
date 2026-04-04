/*
 * hf_decode.c -- Generic HuggingFace config.json + safetensors decoder
 *
 * Parses HF files into PolyHfDecoded. Zero model-specific logic.
 */

#define _POSIX_C_SOURCE 200809L
#include "hf_decode.h"
#include "import_error.h"
#include "../safetensors.h"
#include <stdlib.h>
#include <string.h>

int poly_hf_decode(
    const char *config_json, int config_len,
    const uint8_t **weight_files, const int64_t *weight_lens,
    int n_weight_files,
    PolyHfDecoded **out)
{
    *out = NULL;

    if (!config_json || config_len <= 0) {
        poly_import_error_set(POLY_IMPORT_ERR_PARSE, "NULL or empty config.json");
        return -1;
    }

    cJSON *config = cJSON_ParseWithLength(config_json, (size_t)config_len);
    if (!config) {
        poly_import_error_set(POLY_IMPORT_ERR_PARSE, "failed to parse config.json");
        return -1;
    }

    cJSON *mt = cJSON_GetObjectItemCaseSensitive(config, "model_type");
    const char *model_type = (mt && cJSON_IsString(mt)) ? mt->valuestring : "";

    /* Count total tensors across all weight files */
    int total_tensors = 0;
    for (int f = 0; f < n_weight_files; f++) {
        if (!weight_files || !weight_files[f] || weight_lens[f] <= 0) continue;
        int n = 0;
        char *meta = NULL;
        PolySafetensorViewEx *views = poly_safetensors_decode_ex(
            weight_files[f], weight_lens[f], &n, &meta);
        free(meta);
        if (views) {
            for (int i = 0; i < n; i++) free(views[i].name);
            free(views);
        }
        total_tensors += n;
    }

    PolyHfDecoded *hf = calloc(1, sizeof(PolyHfDecoded));
    hf->config = config;
    hf->model_type = model_type;
    hf->tensors = total_tensors > 0
        ? calloc((size_t)total_tensors, sizeof(PolyDecodedTensor))
        : NULL;
    hf->n_tensors = 0;

    /* Decode all weight files */
    for (int f = 0; f < n_weight_files; f++) {
        if (!weight_files || !weight_files[f] || weight_lens[f] <= 0) continue;

        int n = 0;
        char *meta = NULL;
        PolySafetensorViewEx *views = poly_safetensors_decode_ex(
            weight_files[f], weight_lens[f], &n, &meta);
        free(meta);

        if (!views) {
            poly_import_error_set(POLY_IMPORT_ERR_PARSE,
                "failed to decode weight file %d", f);
            continue;
        }

        for (int i = 0; i < n; i++) {
            PolyDecodedTensor *dt = &hf->tensors[hf->n_tensors];
            dt->name  = views[i].name;
            dt->data  = views[i].raw_data;
            dt->ndim  = views[i].ndim;
            dt->numel = views[i].numel;
            dt->dtype = (int)views[i].dtype;
            for (int d = 0; d < views[i].ndim && d < 8; d++)
                dt->shape[d] = views[i].shape[d];
            hf->n_tensors++;
        }

        free(views);
    }

    *out = hf;
    return 0;
}

void poly_hf_decoded_free(PolyHfDecoded *hf) {
    if (!hf) return;
    for (int i = 0; i < hf->n_tensors; i++)
        free(hf->tensors[i].name);
    free(hf->tensors);
    if (hf->config) cJSON_Delete(hf->config);
    free(hf);
}
