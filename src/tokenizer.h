/*
 * tokenizer.h -- BPE tokenizer for LLM inference
 *
 * Vocabulary-ranked byte-level BPE following tinygrad's SimpleTokenizer.
 * Vocabulary and special-token IDs come from GGUF metadata.
 * JSON supports validated byte-level BPE pipelines with explicit merge ranks.
 */

#ifndef POLY_TOKENIZER_H
#define POLY_TOKENIZER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PolyTokenizer PolyTokenizer;

/* Creation */

/*
 * Create tokenizer from GGUF KV metadata.
 * Reads: tokenizer.ggml.tokens, tokenizer.ggml.token_type,
 *        tokenizer.ggml.bos_token_id, tokenizer.ggml.eos_token_id.
 * Returns NULL on error.
 */
#include "loaders/gguf_decode.h"
PolyTokenizer *poly_tokenizer_from_gguf(const PolyGgufDecoded *gguf);

/*
 * Create tokenizer from explicit vocab arrays.
 * tokens[i] is a UTF-8 string (using GPT-2 byte encoding).
 * types[i]: 1 = normal, all other values = literal special token.
 */
PolyTokenizer *poly_tokenizer_create(const char **tokens, const int *types, int n_tokens);

/*
 * Strict JSON byte-level BPE import. Returns NULL with a stderr diagnostic for
 * unsupported pipelines. No normalization, padding or special-token insertion.
 */
PolyTokenizer *poly_tokenizer_from_json(const char *json_data, int json_len);

/*
 * strict=0 permits skipping NFC normalization, and no other approximation.
 * diagnostic (optional, caller-owned) receives a NUL-terminated error on NULL,
 * warning on approximated success, or an empty string on exact success.
 * No diagnostic is printed; the caller must surface nonempty warnings.
 */
PolyTokenizer *poly_tokenizer_from_json_ex(
    const char *json_data,
    int json_len,
    int strict,
    char *diagnostic,
    int diagnostic_size
);

void poly_tokenizer_free(PolyTokenizer *tok);

/* Encode / Decode */

/*
 * Encode UTF-8 text to token IDs. Returns number of tokens written, or -1
 * for invalid UTF-8, missing byte tokens or allocation failure.
 * If ids_out is NULL, returns the count without writing.
 */
int poly_tokenize(const PolyTokenizer *tok, const char *text, int *ids_out, int max_ids);

/*
 * Decode token IDs to text. Returns number of bytes written
 * (excluding null terminator). If text_out is NULL, returns count.
 */
int poly_detokenize(
    const PolyTokenizer *tok,
    const int *ids,
    int n_ids,
    char *text_out,
    int max_len
);

/* Accessors */

int poly_tokenizer_vocab_size(const PolyTokenizer *tok);
int poly_tokenizer_bos_id(const PolyTokenizer *tok);
int poly_tokenizer_eos_id(const PolyTokenizer *tok);

#ifdef __cplusplus
}
#endif

#endif /* POLY_TOKENIZER_H */
