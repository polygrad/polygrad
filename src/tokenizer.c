/*
 * tokenizer.c -- BPE tokenizer
 *
 * Port of tinygrad's SimpleTokenizer. Uses vocab ordering as merge
 * priority (lower token ID = higher priority merge). Works for GPT-2,
 * LLaMA 3, Qwen3, and all models using the GGUF tokenizer format.
 *
 * Algorithm:
 *   1. Pre-tokenize: split text into words (UTF-8 aware)
 *   2. For each word: convert to bytes via GPT-2 byte encoding
 *   3. BPE: greedily merge byte pair whose merged token has lowest ID
 *   4. Decode: token IDs -> byte sequences -> UTF-8 text
 *
 * Reference: tinygrad/apps/llm.py SimpleTokenizer
 * Reference: llama.cpp/src/llama-vocab.cpp
 */

#define _POSIX_C_SOURCE 200809L
#include "tokenizer.h"
#include "loaders/gguf_decode.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>

/* ── GPT-2 byte encoder/decoder ─────────────────────────────────── */

/*
 * GPT-2 maps each byte value to a Unicode character for display.
 * Printable ASCII (33-126) and Latin-1 supplement (161-172, 174-255)
 * map to themselves. The remaining 68 bytes (0-32, 127-160, 173)
 * map to codepoints 256-323.
 *
 * This allows the vocabulary to use printable UTF-8 strings even for
 * tokens containing control characters or raw bytes.
 */

static int g_byte_to_char[256];   /* byte -> Unicode codepoint */
static int g_char_to_byte[512];   /* codepoint -> byte (0-323 range) */
static int g_byte_table_init = 0;

static void init_byte_table(void) {
    if (g_byte_table_init) return;
    memset(g_char_to_byte, -1, sizeof(g_char_to_byte));

    int n = 0;
    for (int b = 0; b < 256; b++) {
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
            g_byte_to_char[b] = b;
        } else {
            g_byte_to_char[b] = 256 + n;
            n++;
        }
        g_char_to_byte[g_byte_to_char[b]] = b;
    }
    g_byte_table_init = 1;
}

/*
 * Decode a UTF-8 token string (using GPT-2 byte encoding) into raw bytes.
 * Returns the number of bytes written. out must be at least as large as
 * the token string length.
 */
static int token_str_to_bytes(const char *s, uint8_t *out, int max_out) {
    int n = 0;
    const uint8_t *p = (const uint8_t *)s;
    while (*p && n < max_out) {
        int cp;
        /* Decode one UTF-8 codepoint */
        if (*p < 0x80) {
            cp = *p++;
        } else if ((*p & 0xE0) == 0xC0) {
            cp = (*p & 0x1F) << 6;
            p++;
            if ((*p & 0xC0) == 0x80) cp |= (*p++ & 0x3F);
        } else if ((*p & 0xF0) == 0xE0) {
            cp = (*p & 0x0F) << 12;
            p++;
            if ((*p & 0xC0) == 0x80) { cp |= (*p & 0x3F) << 6; p++; }
            if ((*p & 0xC0) == 0x80) { cp |= (*p++ & 0x3F); }
        } else {
            /* 4-byte UTF-8 or invalid: skip */
            p++;
            continue;
        }
        /* Map codepoint to byte via GPT-2 table */
        if (cp >= 0 && cp < 512 && g_char_to_byte[cp] >= 0)
            out[n++] = (uint8_t)g_char_to_byte[cp];
        else if (cp < 256)
            out[n++] = (uint8_t)cp;
    }
    return n;
}

/* ── Hash map for vocab lookup ──────────────────────────────────── */

typedef struct {
    uint8_t *key;       /* byte sequence (owned) */
    int key_len;
    int token_id;
} VocabEntry;

typedef struct {
    VocabEntry *entries;
    int capacity;
    int count;
} VocabMap;

static uint32_t hash_bytes(const uint8_t *data, int len) {
    uint32_t h = 2166136261u;
    for (int i = 0; i < len; i++)
        h = (h ^ data[i]) * 16777619u;
    return h;
}

static void vocab_map_init(VocabMap *m, int capacity) {
    m->capacity = capacity;
    m->count = 0;
    m->entries = calloc((size_t)capacity, sizeof(VocabEntry));
}

static void vocab_map_free(VocabMap *m) {
    for (int i = 0; i < m->capacity; i++)
        free(m->entries[i].key);
    free(m->entries);
}

static void vocab_map_insert(VocabMap *m, const uint8_t *key, int key_len, int token_id) {
    uint32_t h = hash_bytes(key, key_len) % (uint32_t)m->capacity;
    while (m->entries[h].key != NULL) {
        if (m->entries[h].key_len == key_len &&
            memcmp(m->entries[h].key, key, (size_t)key_len) == 0) {
            m->entries[h].token_id = token_id;
            return;
        }
        h = (h + 1) % (uint32_t)m->capacity;
    }
    m->entries[h].key = malloc((size_t)key_len);
    memcpy(m->entries[h].key, key, (size_t)key_len);
    m->entries[h].key_len = key_len;
    m->entries[h].token_id = token_id;
    m->count++;
}

/* Returns token_id or -1 if not found */
static int vocab_map_find(const VocabMap *m, const uint8_t *key, int key_len) {
    uint32_t h = hash_bytes(key, key_len) % (uint32_t)m->capacity;
    for (int probe = 0; probe < m->capacity; probe++) {
        if (m->entries[h].key == NULL) return -1;
        if (m->entries[h].key_len == key_len &&
            memcmp(m->entries[h].key, key, (size_t)key_len) == 0)
            return m->entries[h].token_id;
        h = (h + 1) % (uint32_t)m->capacity;
    }
    return -1;
}

/* ── Tokenizer struct ───────────────────────────────────────────── */

struct PolyTokenizer {
    VocabMap normal;        /* byte_seq -> token_id for normal tokens */
    /* Reverse map: token_id -> byte sequence */
    uint8_t **id_to_bytes;
    int *id_to_len;
    int vocab_size;
    int bos_id;
    int eos_id;
};

/* ── BPE core ───────────────────────────────────────────────────── */

/*
 * BPE encode a single word (as raw bytes).
 * Uses greedy pair merging: find the pair whose merged token has the
 * lowest vocab ID, merge it, repeat until no more merges.
 *
 * Matches tinygrad SimpleTokenizer._encode_word().
 */
static int bpe_encode_word(const PolyTokenizer *tok,
                           const uint8_t *word, int word_len,
                           int *ids_out, int max_ids)
{
    if (word_len <= 0) return 0;

    /* Check if the whole word is a single token */
    int whole = vocab_map_find(&tok->normal, word, word_len);
    if (whole >= 0) {
        if (ids_out && max_ids > 0) ids_out[0] = whole;
        return 1;
    }

    /* Start with individual bytes as parts */
    typedef struct { uint8_t *data; int len; } Part;
    int n_parts = word_len;
    Part *parts = malloc((size_t)word_len * sizeof(Part));
    for (int i = 0; i < word_len; i++) {
        parts[i].data = malloc(1);
        parts[i].data[0] = word[i];
        parts[i].len = 1;
    }

    /* Greedy merge loop */
    while (n_parts > 1) {
        int best_id = INT32_MAX;
        int best_j = -1;

        /* Find the pair whose merged token has the lowest ID */
        for (int j = 0; j < n_parts - 1; j++) {
            int merged_len = parts[j].len + parts[j + 1].len;
            uint8_t *merged = malloc((size_t)merged_len);
            memcpy(merged, parts[j].data, (size_t)parts[j].len);
            memcpy(merged + parts[j].len, parts[j + 1].data, (size_t)parts[j + 1].len);
            int tid = vocab_map_find(&tok->normal, merged, merged_len);
            free(merged);
            if (tid >= 0 && tid < best_id) {
                best_id = tid;
                best_j = j;
            }
        }

        if (best_j < 0) break;  /* no more merges possible */

        /* Merge parts[best_j] and parts[best_j + 1] */
        int new_len = parts[best_j].len + parts[best_j + 1].len;
        uint8_t *new_data = malloc((size_t)new_len);
        memcpy(new_data, parts[best_j].data, (size_t)parts[best_j].len);
        memcpy(new_data + parts[best_j].len, parts[best_j + 1].data,
               (size_t)parts[best_j + 1].len);
        free(parts[best_j].data);
        free(parts[best_j + 1].data);
        parts[best_j].data = new_data;
        parts[best_j].len = new_len;

        /* Remove parts[best_j + 1] */
        for (int k = best_j + 1; k < n_parts - 1; k++)
            parts[k] = parts[k + 1];
        n_parts--;
    }

    /* Convert parts to token IDs */
    int n_ids = 0;
    for (int i = 0; i < n_parts && n_ids < max_ids; i++) {
        int tid = vocab_map_find(&tok->normal, parts[i].data, parts[i].len);
        if (tid >= 0) {
            if (ids_out) ids_out[n_ids] = tid;
            n_ids++;
        }
    }

    for (int i = 0; i < n_parts; i++) free(parts[i].data);
    free(parts);
    return n_ids;
}

/* ── Pre-tokenization (word splitting) ──────────────────────────── */

/*
 * Simplified pre-tokenizer matching GPT-2/LLaMA pattern behavior.
 * Splits text into words at whitespace and punctuation boundaries.
 * Leading spaces are attached to the following word.
 *
 * This is simpler than the full Unicode-category regex in tinygrad
 * but handles the common cases correctly.
 */

static int is_letter(uint8_t c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
           c >= 0xC0;  /* rough: Latin extended and beyond */
}

static int is_digit(uint8_t c) {
    return c >= '0' && c <= '9';
}

typedef void (*word_callback)(const uint8_t *word, int len, void *ctx);

static void split_to_words(const uint8_t *text, int text_len,
                           word_callback cb, void *ctx)
{
    int i = 0;
    while (i < text_len) {
        int start = i;

        /* Consume optional leading space */
        if (i < text_len && text[i] == ' ') i++;

        if (i < text_len && is_letter(text[i])) {
            /* Letter word (with optional leading space) */
            while (i < text_len && is_letter(text[i])) i++;
        } else if (i < text_len && is_digit(text[i])) {
            /* Digit sequence (1-3 digits like tinygrad) */
            int d = 0;
            while (i < text_len && is_digit(text[i]) && d < 3) { i++; d++; }
        } else if (i < text_len && (text[i] == '\n' || text[i] == '\r')) {
            /* Newline sequence */
            while (i < text_len && (text[i] == '\n' || text[i] == '\r')) i++;
        } else if (i < text_len) {
            /* Single other character (punctuation, etc.) */
            i++;
        }

        if (i > start)
            cb(text + start, i - start, ctx);
    }
}

/* ── Public API ──────────────────────────────────────────────────── */

PolyTokenizer *poly_tokenizer_create(
    const char **tokens, const int *types, int n_tokens)
{
    init_byte_table();

    PolyTokenizer *tok = calloc(1, sizeof(PolyTokenizer));
    tok->vocab_size = n_tokens;
    tok->bos_id = -1;
    tok->eos_id = -1;

    /* Allocate reverse map */
    tok->id_to_bytes = calloc((size_t)n_tokens, sizeof(uint8_t *));
    tok->id_to_len = calloc((size_t)n_tokens, sizeof(int));

    /* Build normal token map (2x capacity for low collision rate) */
    vocab_map_init(&tok->normal, n_tokens * 2 + 1);

    uint8_t buf[1024];
    for (int i = 0; i < n_tokens; i++) {
        if (!tokens[i]) continue;
        int is_special = types && types[i] == 3;  /* control/special token */

        /* Convert token string to raw bytes via GPT-2 byte encoding */
        int blen;
        if (is_special) {
            /* Special tokens are literal UTF-8, no byte encoding */
            blen = (int)strlen(tokens[i]);
            if (blen > (int)sizeof(buf)) blen = (int)sizeof(buf);
            memcpy(buf, tokens[i], (size_t)blen);
        } else {
            blen = token_str_to_bytes(tokens[i], buf, (int)sizeof(buf));
        }

        /* Store in reverse map */
        tok->id_to_bytes[i] = malloc((size_t)blen);
        memcpy(tok->id_to_bytes[i], buf, (size_t)blen);
        tok->id_to_len[i] = blen;

        /* Normal tokens go in the BPE lookup map */
        if (!is_special)
            vocab_map_insert(&tok->normal, buf, blen, i);
    }

    return tok;
}

PolyTokenizer *poly_tokenizer_from_gguf(const PolyGgufDecoded *gguf) {
    if (!gguf) return NULL;

    int n_tokens = 0;
    const char **tokens = poly_gguf_kv_string_array(gguf, "tokenizer.ggml.tokens", &n_tokens);
    if (!tokens || n_tokens <= 0) {
        fprintf(stderr, "poly_tokenizer_from_gguf: no tokenizer.ggml.tokens\n");
        return NULL;
    }

    int n_types = 0;
    const int32_t *types_raw = poly_gguf_kv_int_array(gguf, "tokenizer.ggml.token_type", &n_types);
    /* Convert int32_t* to int* (same on most platforms) */
    int *types = NULL;
    if (types_raw && n_types == n_tokens)
        types = (int *)types_raw;

    PolyTokenizer *tok = poly_tokenizer_create(tokens, types, n_tokens);
    if (!tok) return NULL;

    tok->bos_id = poly_gguf_kv_int(gguf, "tokenizer.ggml.bos_token_id", -1);
    tok->eos_id = poly_gguf_kv_int(gguf, "tokenizer.ggml.eos_token_id", -1);

    return tok;
}

void poly_tokenizer_free(PolyTokenizer *tok) {
    if (!tok) return;
    vocab_map_free(&tok->normal);
    for (int i = 0; i < tok->vocab_size; i++)
        free(tok->id_to_bytes[i]);
    free(tok->id_to_bytes);
    free(tok->id_to_len);
    free(tok);
}

/* Callback context for tokenization */
typedef struct {
    const PolyTokenizer *tok;
    int *ids;
    int max_ids;
    int count;
} EncodeCtx;

static void encode_word_cb(const uint8_t *word, int len, void *ctx_) {
    EncodeCtx *ctx = (EncodeCtx *)ctx_;
    int remaining = ctx->max_ids - ctx->count;
    if (remaining <= 0) return;
    int n = bpe_encode_word(ctx->tok, word, len,
                            ctx->ids ? ctx->ids + ctx->count : NULL,
                            remaining);
    ctx->count += n;
}

int poly_tokenize(const PolyTokenizer *tok, const char *text,
                  int *ids_out, int max_ids)
{
    if (!tok || !text) return 0;
    EncodeCtx ctx = { .tok = tok, .ids = ids_out, .max_ids = max_ids, .count = 0 };
    split_to_words((const uint8_t *)text, (int)strlen(text), encode_word_cb, &ctx);
    return ctx.count;
}

int poly_detokenize(const PolyTokenizer *tok, const int *ids, int n_ids,
                    char *text_out, int max_len)
{
    if (!tok || !ids) return 0;
    int pos = 0;
    for (int i = 0; i < n_ids; i++) {
        int id = ids[i];
        if (id < 0 || id >= tok->vocab_size) continue;
        int blen = tok->id_to_len[id];
        if (text_out && pos + blen < max_len)
            memcpy(text_out + pos, tok->id_to_bytes[id], (size_t)blen);
        pos += blen;
    }
    if (text_out && max_len > 0)
        text_out[pos < max_len ? pos : max_len - 1] = '\0';
    return pos;
}

int poly_tokenizer_vocab_size(const PolyTokenizer *tok) {
    return tok ? tok->vocab_size : 0;
}
int poly_tokenizer_bos_id(const PolyTokenizer *tok) {
    return tok ? tok->bos_id : -1;
}
int poly_tokenizer_eos_id(const PolyTokenizer *tok) {
    return tok ? tok->eos_id : -1;
}
