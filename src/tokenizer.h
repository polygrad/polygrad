/*
 * tokenizer.h -- BPE tokenizer for LLM inference
 *
 * Implements the BPE algorithm used by GPT-2, LLaMA, Qwen, and most
 * modern LLMs. Tokenizer is parameterized by vocabulary (token strings
 * + IDs) extracted from GGUF metadata or loaded from files.
 *
 * Follows tinygrad's SimpleTokenizer approach: uses vocab ordering
 * as merge priority (lower token ID = higher priority merge).
 */

#ifndef POLY_TOKENIZER_H
#define POLY_TOKENIZER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PolyTokenizer PolyTokenizer;

/* ── Creation ───────────────────────────────────────────────────── */

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
 * types[i]: 1 = normal, 3 = control/special, others = normal.
 */
PolyTokenizer *poly_tokenizer_create(
    const char **tokens, const int *types, int n_tokens);

/*
 * Create tokenizer from HF tokenizer.json content.
 * Parses model.vocab and added_tokens from the JSON.
 * json_data must be null-terminated.
 */
PolyTokenizer *poly_tokenizer_from_json(const char *json_data, int json_len);

void poly_tokenizer_free(PolyTokenizer *tok);

/* ── Encode / Decode ────────────────────────────────────────────── */

/*
 * Encode text to token IDs. Returns number of tokens written.
 * If ids_out is NULL, returns the count without writing.
 */
int poly_tokenize(const PolyTokenizer *tok, const char *text,
                  int *ids_out, int max_ids);

/*
 * Decode token IDs to text. Returns number of bytes written
 * (excluding null terminator). If text_out is NULL, returns count.
 */
int poly_detokenize(const PolyTokenizer *tok, const int *ids, int n_ids,
                    char *text_out, int max_len);

/* ── Accessors ──────────────────────────────────────────────────── */

int poly_tokenizer_vocab_size(const PolyTokenizer *tok);
int poly_tokenizer_bos_id(const PolyTokenizer *tok);
int poly_tokenizer_eos_id(const PolyTokenizer *tok);

#ifdef __cplusplus
}
#endif

#endif /* POLY_TOKENIZER_H */
