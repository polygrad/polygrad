/*
 * gguf_decode.h -- GGUF binary format decoder
 *
 * Parses GGUF files into PolyGgufDecoded. Contains zero model-specific logic.
 * Implementation deferred to Phase 3.
 */

#ifndef POLY_GGUF_DECODE_H
#define POLY_GGUF_DECODE_H

#include "decoded.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char *key;
    int type;
    union {
        uint64_t u64;
        int64_t  i64;
        double   f64;
        struct { char *str; int len; } s;
    } val;
} PolyGgufKV;

typedef struct {
    PolyGgufKV         *kv;
    int                 n_kv;
    const char         *arch;       /* borrowed from kv */
    PolyDecodedTensor  *tensors;
    int                 n_tensors;
} PolyGgufDecoded;

int  poly_gguf_decode(const uint8_t *data, int64_t len, PolyGgufDecoded **out);
void poly_gguf_decoded_free(PolyGgufDecoded *gguf);

int         poly_gguf_kv_int(const PolyGgufDecoded *g, const char *key, int def);
double      poly_gguf_kv_float(const PolyGgufDecoded *g, const char *key, double def);
const char *poly_gguf_kv_string(const PolyGgufDecoded *g, const char *key, const char *def);

#ifdef __cplusplus
}
#endif

#endif /* POLY_GGUF_DECODE_H */
