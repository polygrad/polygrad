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

/* Special token entry for sentence-level splitting */
typedef struct {
    char *text;     /* literal UTF-8 string (e.g. "<think>") */
    int text_len;
    int token_id;
} SpecialToken;

struct PolyTokenizer {
    VocabMap normal;        /* byte_seq -> token_id for normal tokens */
    /* Reverse map: token_id -> byte sequence */
    uint8_t **id_to_bytes;
    int *id_to_len;
    int vocab_size;
    int bos_id;
    int eos_id;
    /* Special tokens for sentence-level splitting */
    SpecialToken *specials;
    int n_specials;
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
 * Port of the GPT-2/LLaMA pre-tokenization regex from tinygrad.
 * Splits text into words before BPE is applied independently per word.
 *
 * The regex alternations (in priority order):
 *   1. (?i:'s|'t|'re|'ve|'m|'ll|'d)  -- contractions
 *   2. [^LN]?[L]+                      -- optional non-letter/digit + letters
 *   3. [N]{1,3}                         -- 1-3 digits
 *   4.  ?[^ws,L,N]+[\r\n]*             -- opt space + punct seq + opt newlines
 *   5. [ws]*[\r\n]+                     -- whitespace before newlines
 *   6. [ws]+(?![^ws])                   -- trailing whitespace (end of string)
 *   7. [ws]+                            -- whitespace
 *
 * ASCII approximation: L = [A-Za-z\x80-\xFF], N = [0-9],
 *                      ws = [ \t\n\r\v\f]
 */

static int is_letter(uint8_t c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c >= 0x80;
}

static int is_digit(uint8_t c) {
    return c >= '0' && c <= '9';
}

static int is_ws(uint8_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
           c == '\v' || c == '\f';
}

static int is_newline(uint8_t c) {
    return c == '\n' || c == '\r';
}

/* Case-insensitive contraction check. Returns length matched or 0. */
static int match_contraction(const uint8_t *p, int remaining) {
    if (remaining < 2 || p[0] != '\'') return 0;
    uint8_t c = p[1] | 0x20;  /* lowercase */
    if (c == 's' || c == 't' || c == 'm' || c == 'd') return 2;
    if (remaining >= 3) {
        uint8_t c2 = p[2] | 0x20;
        if (c == 'r' && c2 == 'e') return 3;
        if (c == 'v' && c2 == 'e') return 3;
        if (c == 'l' && c2 == 'l') return 3;
    }
    return 0;
}

typedef void (*word_callback)(const uint8_t *word, int len, void *ctx);

/*
 * Implements the GPT-2/LLaMA pre-tokenization regex.
 * The regex tries alternations in order, first match wins.
 * Each alternation is tried at the current position.
 */
static void split_to_words(const uint8_t *text, int text_len,
                           word_callback cb, void *ctx)
{
    int i = 0;
    while (i < text_len) {
        /* Alt 1: contraction ('s, 't, 're, 've, 'm, 'll, 'd) */
        int clen = match_contraction(text + i, text_len - i);
        if (clen > 0) { cb(text + i, clen, ctx); i += clen; continue; }

        /* Alt 2: [^LN]?[L]+ -- optional non-letter/non-digit then letters.
         * The leading char must NOT be \r or \n. */
        {
            int j = i;
            if (j < text_len && !is_letter(text[j]) && !is_digit(text[j]) &&
                !is_newline(text[j]))
                j++;
            if (j < text_len && is_letter(text[j])) {
                while (j < text_len && is_letter(text[j])) j++;
                cb(text + i, j - i, ctx); i = j; continue;
            }
        }

        /* Alt 3: [N]{1,3} -- 1-3 digits */
        if (is_digit(text[i])) {
            int j = i, d = 0;
            while (j < text_len && is_digit(text[j]) && d < 3) { j++; d++; }
            cb(text + i, j - i, ctx); i = j; continue;
        }

        /* Alt 4: ' ?[^ws,L,N]+[\r\n]*' -- optional space, then 1+ punct/symbols,
         * then optional newlines. */
        {
            int j = i;
            if (j < text_len && text[j] == ' ') j++;
            int k = j;
            while (k < text_len && !is_ws(text[k]) &&
                   !is_letter(text[k]) && !is_digit(text[k]))
                k++;
            if (k > j) {
                while (k < text_len && is_newline(text[k])) k++;
                cb(text + i, k - i, ctx); i = k; continue;
            }
        }

        /* Alt 5 / 6 / 7: whitespace handling.
         *
         * Scan maximal whitespace run, then apply regex semantics:
         *   Alt 5: [ws]*[\r\n]+ -- match up to last newline in run
         *   Alt 6: [ws]+(?![^ws]) -- trailing ws or ws before more ws
         *          (backtrack by 1 if followed by non-ws)
         *   Alt 7: [ws]+ -- single ws byte fallback
         */
        if (is_ws(text[i])) {
            int j = i;
            int last_nl_end = -1;
            while (j < text_len && is_ws(text[j])) {
                if (is_newline(text[j])) last_nl_end = j + 1;
                j++;
            }

            /* Alt 5: consume up to last newline */
            if (last_nl_end >= 0) {
                cb(text + i, last_nl_end - i, ctx);
                i = last_nl_end; continue;
            }

            /* Alt 6: trailing whitespace at end of input */
            if (j == text_len) {
                cb(text + i, j - i, ctx); i = j; continue;
            }

            /* Alt 6: whitespace before non-ws -- leave 1 byte for next alt */
            if (j - i > 1) {
                cb(text + i, j - i - 1, ctx);
                i = j - 1; continue;
            }

            /* Alt 7: single whitespace byte */
            cb(text + i, 1, ctx); i++; continue;
        }

        /* Fallback: single byte */
        cb(text + i, 1, ctx);
        i++;
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
        /* tinygrad: type==1 is normal, everything else is special
         * (type 3=control, 4=user-defined, 6=unused, etc.) */
        int is_special = types && types[i] != 1;

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

    /* Collect special tokens for sentence-level splitting */
    int n_special = 0;
    for (int i = 0; i < n_tokens; i++)
        if (tokens[i] && types && types[i] != 1) n_special++;

    if (n_special > 0) {
        tok->specials = calloc((size_t)n_special, sizeof(SpecialToken));
        tok->n_specials = 0;
        for (int i = 0; i < n_tokens; i++) {
            if (!tokens[i] || !types || types[i] == 1) continue;
            int slen = (int)strlen(tokens[i]);
            tok->specials[tok->n_specials].text = strdup(tokens[i]);
            tok->specials[tok->n_specials].text_len = slen;
            tok->specials[tok->n_specials].token_id = i;
            tok->n_specials++;
        }
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
    for (int i = 0; i < tok->n_specials; i++)
        free(tok->specials[i].text);
    free(tok->specials);
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

/* Encode a chunk of normal text (no special tokens) via word split + BPE */
static int encode_chunk(const PolyTokenizer *tok, const uint8_t *text, int len,
                        int *ids_out, int max_ids) {
    EncodeCtx ctx = { .tok = tok, .ids = ids_out, .max_ids = max_ids, .count = 0 };
    split_to_words(text, len, encode_word_cb, &ctx);
    return ctx.count;
}

/* Find the earliest special token match in text[pos..end) */
static int find_special(const PolyTokenizer *tok, const char *text, int pos, int end,
                        int *match_len, int *token_id) {
    int best_pos = end;
    int best_len = 0;
    int best_id = -1;
    for (int s = 0; s < tok->n_specials; s++) {
        const char *needle = tok->specials[s].text;
        int nlen = tok->specials[s].text_len;
        /* Search for needle starting at pos */
        for (int j = pos; j + nlen <= end; j++) {
            if (memcmp(text + j, needle, (size_t)nlen) == 0) {
                if (j < best_pos || (j == best_pos && nlen > best_len)) {
                    best_pos = j;
                    best_len = nlen;
                    best_id = tok->specials[s].token_id;
                }
                break;  /* first occurrence of this special token */
            }
        }
    }
    if (best_id >= 0) {
        *match_len = best_len;
        *token_id = best_id;
    }
    return best_pos;
}

int poly_tokenize(const PolyTokenizer *tok, const char *text,
                  int *ids_out, int max_ids)
{
    if (!tok || !text) return 0;
    int text_len = (int)strlen(text);
    int count = 0;
    int pos = 0;

    /*
     * Two-level split matching tinygrad:
     * 1. Split on special tokens (sentence level)
     * 2. For each non-special chunk, split into words + BPE
     */
    while (pos < text_len && count < max_ids) {
        int match_len = 0, token_id = -1;
        int sp = (tok->n_specials > 0)
            ? find_special(tok, text, pos, text_len, &match_len, &token_id)
            : text_len;

        /* Encode text before the special token */
        if (sp > pos) {
            int remaining = max_ids - count;
            int n = encode_chunk(tok, (const uint8_t *)text + pos, sp - pos,
                                 ids_out ? ids_out + count : NULL, remaining);
            count += n;
        }

        /* Emit the special token */
        if (token_id >= 0 && count < max_ids) {
            if (ids_out) ids_out[count] = token_id;
            count++;
            pos = sp + match_len;
        } else {
            pos = sp;
        }
    }

    return count;
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
