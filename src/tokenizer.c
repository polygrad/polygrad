/*
 * tokenizer.c -- BPE tokenizer
 *
 * GGUF: pinned tinygrad/llm/cli.py SimpleTokenizer, vocabulary-ranked BPE.
 * HF tokenizer.json pipelines are intentionally unsupported.
 */

#define _POSIX_C_SOURCE 200809L
#include "tokenizer.h"
#include "loaders/gguf_decode.h"
#include "tokenizer_unicode.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#ifndef __EMSCRIPTEN__
#include <pthread.h>
#endif

/* GPT-2 byte encoder/decoder */

/*
 * GPT-2 maps each byte value to a Unicode character for display.
 * Printable ASCII (33-126) and Latin-1 supplement (161-172, 174-255)
 * map to themselves. The remaining 68 bytes (0-32, 127-160, 173)
 * map to codepoints 256-323.
 *
 * This allows the vocabulary to use printable UTF-8 strings even for
 * tokens containing control characters or raw bytes.
 */

static int g_byte_to_char[256]; /* byte -> Unicode codepoint */
static int g_char_to_byte[512]; /* codepoint -> byte (0-323 range) */
static void build_byte_table(void) {
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
}

static void init_byte_table(void) {
  /* Immutable after publication, shared by independent tokenizer owners. */
#ifndef __EMSCRIPTEN__
  static pthread_once_t once = PTHREAD_ONCE_INIT;
  pthread_once(&once, build_byte_table);
#else
  /* The packaged Wasm module has no shared-memory threads. */
  static bool initialized = false;
  if (!initialized) {
    build_byte_table();
    initialized = true;
  }
#endif
}

/*
 * Decode a UTF-8 token string (using GPT-2 byte encoding) into raw bytes.
 * Returns the number of bytes written. out must be at least as large as
 * the token string length.
 */
/* Strict UTF-8 decoding, without normalization or locale-dependent categories. */
static int decode_utf8(const uint8_t *s, int len, int32_t *out) {
  if (len <= 0) return -1;
  int n = s[0] < 0x80                    ? 1
          : s[0] >= 0xc2 && s[0] <= 0xdf ? 2
          : s[0] >= 0xe0 && s[0] <= 0xef ? 3
          : s[0] >= 0xf0 && s[0] <= 0xf4 ? 4
                                         : 0;
  if (!n || n > len) return -1;
  int32_t cp = s[0] & (n == 1 ? 0x7f : (1 << (7 - n)) - 1);
  for (int i = 1; i < n; i++) {
    if ((s[i] & 0xc0) != 0x80) return -1;
    cp = (cp << 6) | (s[i] & 0x3f);
  }
  if ((n == 2 && cp < 0x80) || (n == 3 && cp < 0x800) || (n == 4 && cp < 0x10000) ||
      (cp >= 0xd800 && cp <= 0xdfff) || cp > 0x10ffff)
    return -1;
  *out = cp;
  return n;
}

static int token_str_to_bytes(const char *s, uint8_t *out, int max_out) {
  int pos = 0, n = 0, len = (int)strlen(s);
  while (pos < len) {
    int32_t cp;
    int width = (int)decode_utf8((const uint8_t *)s + pos, len - pos, &cp);
    if (width <= 0 || cp >= 512 || g_char_to_byte[cp] < 0 || n == max_out) return -1;
    out[n++] = (uint8_t)g_char_to_byte[cp];
    pos += width;
  }
  return n;
}

/* Hash map for vocab lookup */

typedef struct {
  uint8_t *key; /* byte sequence (owned) */
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
  for (int i = 0; m->entries && i < m->capacity; i++)
    free(m->entries[i].key);
  free(m->entries);
}

static bool vocab_map_insert(VocabMap *m, const uint8_t *key, int key_len, int token_id) {
  uint32_t h = hash_bytes(key, key_len) % (uint32_t)m->capacity;
  while (m->entries[h].key != NULL) {
    if (m->entries[h].key_len == key_len && memcmp(m->entries[h].key, key, (size_t)key_len) == 0) {
      m->entries[h].token_id = token_id;
      return true;
    }
    h = (h + 1) % (uint32_t)m->capacity;
  }
  m->entries[h].key = malloc((size_t)key_len);
  if (!m->entries[h].key) return false;
  memcpy(m->entries[h].key, key, (size_t)key_len);
  m->entries[h].key_len = key_len;
  m->entries[h].token_id = token_id;
  m->count++;
  return true;
}

/* Returns token_id or -1 if not found */
static int vocab_map_find(const VocabMap *m, const uint8_t *key, int key_len) {
  uint32_t h = hash_bytes(key, key_len) % (uint32_t)m->capacity;
  for (int probe = 0; probe < m->capacity; probe++) {
    if (m->entries[h].key == NULL) return -1;
    if (m->entries[h].key_len == key_len && memcmp(m->entries[h].key, key, (size_t)key_len) == 0)
      return m->entries[h].token_id;
    h = (h + 1) % (uint32_t)m->capacity;
  }
  return -1;
}

/* Tokenizer struct */

/* Special token entry for sentence-level splitting */
typedef struct {
  char *text; /* literal UTF-8 string (e.g. "<think>") */
  int text_len;
  int token_id;
} SpecialToken;

/* Pre-tokenizer regex variant */
#define POLY_TOK_PRESET_LLAMA 0 /* LLaMA/Qwen: [N]{1,3} */
#define POLY_TOK_PRESET_GPT2 1 /* GPT-2: ' ?[N]+' (space+digits) */

struct PolyTokenizer {
  VocabMap normal; /* byte_seq -> token_id for normal tokens */
  /* Reverse map: token_id -> byte sequence */
  uint8_t **id_to_bytes;
  int *id_to_len;
  int vocab_size;
  int bos_id;
  int eos_id;
  int preset; /* POLY_TOK_PRESET_* */
  /* Special tokens for sentence-level splitting */
  SpecialToken *specials;
  int n_specials;
};

/* Pinned SimpleTokenizer._encode_word: whole words, then vocabulary ranks. */
static int bpe_encode_word(
    const PolyTokenizer *tok,
    const uint8_t *word,
    int word_len,
    int *ids_out,
    int max_ids
) {
  if (word_len <= 0) return 0;
  int whole = vocab_map_find(&tok->normal, word, word_len);
  if (whole >= 0) {
    if (ids_out && max_ids > 0) ids_out[0] = whole;
    return max_ids > 0 ? 1 : 0;
  }
  int *parts = malloc((size_t)word_len * sizeof(*parts));
  uint8_t *merged = malloc((size_t)word_len);
  if (!parts || !merged) {
    free(parts);
    free(merged);
    return -1;
  }
  int n = word_len;
  for (int i = 0; i < n; i++) {
    parts[i] = vocab_map_find(&tok->normal, word + i, 1);
    if (parts[i] < 0) {
      free(parts);
      free(merged);
      return -1;
    }
  }
  while (n > 1) {
    int best_rank = INT32_MAX, best_j = -1, best_id = -1;
    for (int j = 0; j < n - 1; j++) {
      int rank, id;
      int left = parts[j], right = parts[j + 1];
      int len = tok->id_to_len[left];
      memcpy(merged, tok->id_to_bytes[left], (size_t)len);
      memcpy(merged + len, tok->id_to_bytes[right], (size_t)tok->id_to_len[right]);
      rank = id = vocab_map_find(&tok->normal, merged, len + tok->id_to_len[right]);
      if (rank >= 0 && rank < best_rank) {
        best_rank = rank;
        best_j = j;
        best_id = id;
      }
    }
    if (best_j < 0) break;
    parts[best_j] = best_id;
    memmove(parts + best_j + 1, parts + best_j + 2, (size_t)(n - best_j - 2) * sizeof(*parts));
    n--;
  }
  if (n > max_ids) n = max_ids;
  if (ids_out) memcpy(ids_out, parts, (size_t)n * sizeof(*parts));
  free(parts);
  free(merged);
  return n;
}

/* Pinned SimpleTokenizer's Unicode L/N/Z classes, not UTF-8 byte classes. */
static bool in_ranges(int32_t c, const uint32_t ranges[][2], size_t n) {
  size_t lo = 0, hi = n;
  while (lo < hi) {
    size_t mid = lo + (hi - lo) / 2;
    if ((uint32_t)c > ranges[mid][1])
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo < n && (uint32_t)c >= ranges[lo][0];
}
static bool is_letter(int32_t c) {
  return in_ranges(c, tok_letters, sizeof(tok_letters) / sizeof(*tok_letters));
}
static bool is_digit(int32_t c) {
  return in_ranges(c, tok_numbers, sizeof(tok_numbers) / sizeof(*tok_numbers));
}
static bool is_ws(int32_t c) {
  return (c >= 9 && c <= 13) || c == 0x85 ||
         in_ranges(c, tok_spaces, sizeof(tok_spaces) / sizeof(*tok_spaces));
}
static bool is_newline(int32_t c) {
  return c == '\n' || c == '\r';
}

typedef void (*word_callback)(const uint8_t *word, int len, void *ctx);

static int split_to_words(
    const uint8_t *text,
    int text_len,
    int preset,
    word_callback cb,
    void *ctx
) {
  /* Offsets retain the exact original byte slices used by byte-level BPE. */
  int32_t *cp = malloc(((size_t)text_len + 1) * sizeof(*cp));
  int *off = malloc(((size_t)text_len + 1) * sizeof(*off));
  if (!cp || !off) {
    free(cp);
    free(off);
    return -1;
  }
  int n = 0, pos = 0;
  while (pos < text_len) {
    off[n] = pos;
    int len = (int)decode_utf8(text + pos, text_len - pos, &cp[n]);
    if (len <= 0) {
      free(cp);
      free(off);
      return -1;
    }
    pos += len;
    n++;
  }
  off[n] = text_len;
  for (int i = 0; i < n;) {
    int end = i;
    if (cp[i] == '\'') {
      const char *contractions[] = {"s", "t", "re", "ve", "m", "ll", "d"};
      for (int c = 0; c < 7; c++) {
        int len = (int)strlen(contractions[c]), j = 0;
        for (; j < len && i + 1 + j < n; j++) {
          int32_t ch = cp[i + 1 + j];
          /* Simple caseless matching includes long s (U+017F), which lower()
           * alone leaves unchanged. Contractions only contain ASCII letters. */
          if (preset != POLY_TOK_PRESET_GPT2) {
            if (ch >= 'A' && ch <= 'Z') ch += 'a' - 'A';
            if (ch == 0x17f) ch = 's';
          }
          if (ch != contractions[c][j]) break;
        }
        if (j == len) {
          end = i + 1 + len;
          break;
        }
      }
    }
    if (end == i) {
      int j = i;
      if (preset == POLY_TOK_PRESET_GPT2
              ? cp[j] == ' '
              : (!is_letter(cp[j]) && !is_digit(cp[j]) && !is_newline(cp[j])))
        j++;
      int k = j;
      while (k < n && is_letter(cp[k]))
        k++;
      if (k > j) end = k;
    }
    if (end == i) {
      int j = i;
      if (preset == POLY_TOK_PRESET_GPT2 && cp[j] == ' ') j++;
      int limit = preset == POLY_TOK_PRESET_GPT2 ? n : j + 3;
      int k = j;
      while (k < n && k < limit && is_digit(cp[k]))
        k++;
      if (k > j) end = k;
    }
    if (end == i) {
      int j = i + (cp[i] == ' '), k = j;
      while (k < n && !is_ws(cp[k]) && !is_letter(cp[k]) && !is_digit(cp[k]))
        k++;
      if (k > j) {
        if (preset != POLY_TOK_PRESET_GPT2)
          while (k < n && is_newline(cp[k]))
            k++;
        end = k;
      }
    }
    if (end == i && is_ws(cp[i])) {
      int j = i, last_nl = -1;
      while (j < n && is_ws(cp[j])) {
        if (is_newline(cp[j])) last_nl = j + 1;
        j++;
      }
      if (preset != POLY_TOK_PRESET_GPT2 && last_nl >= 0)
        end = last_nl;
      else
        end = j < n && j - i > 1 ? j - 1 : j;
    }
    if (end == i) end++;
    cb(text + off[i], off[end] - off[i], ctx);
    i = end;
  }
  free(cp);
  free(off);
  return 0;
}

/* Public API */

PolyTokenizer *poly_tokenizer_create(const char **tokens, const int *types, int n_tokens) {
  if (!tokens || n_tokens <= 0 || n_tokens > INT32_MAX / 2 - 1) return NULL;
  init_byte_table();

  PolyTokenizer *tok = calloc(1, sizeof(PolyTokenizer));
  if (!tok) return NULL;
  tok->vocab_size = n_tokens;
  tok->bos_id = -1;
  tok->eos_id = -1;

  /* Allocate reverse map */
  tok->id_to_bytes = calloc((size_t)n_tokens, sizeof(uint8_t *));
  tok->id_to_len = calloc((size_t)n_tokens, sizeof(int));

  /* Build normal token map (2x capacity for low collision rate) */
  vocab_map_init(&tok->normal, n_tokens * 2 + 1);
  if (!tok->id_to_bytes || !tok->id_to_len || !tok->normal.entries) goto fail;

  for (int i = 0; i < n_tokens; i++) {
    if (!tokens[i]) continue;
    /* tinygrad: type==1 is normal, everything else is special
     * (type 3=control, 4=user-defined, 6=unused, etc.) */
    int is_special = types && types[i] != 1;
    size_t slen = strlen(tokens[i]);
    if (!slen || slen >= INT32_MAX) goto fail;
    uint8_t *buf = malloc(slen);
    if (!buf) goto fail;
    tok->id_to_bytes[i] = buf;

    /* Convert token string to raw bytes via GPT-2 byte encoding */
    int blen;
    if (is_special) {
      /* Special tokens are literal UTF-8, no byte encoding */
      blen = (int)strlen(tokens[i]);
      memcpy(buf, tokens[i], (size_t)blen);
    } else {
      blen = token_str_to_bytes(tokens[i], buf, (int)slen);
      if (blen <= 0) goto fail;
    }

    /* Store in reverse map */
    tok->id_to_len[i] = blen;

    /* Normal tokens go in the BPE lookup map */
    if (!is_special && !vocab_map_insert(&tok->normal, buf, blen, i)) goto fail;
  }

  /* Collect special tokens for sentence-level splitting */
  int n_special = 0;
  for (int i = 0; i < n_tokens; i++)
    if (tokens[i] && types && types[i] != 1) n_special++;

  if (n_special > 0) {
    tok->specials = calloc((size_t)n_special, sizeof(SpecialToken));
    if (!tok->specials) goto fail;
    tok->n_specials = 0;
    for (int i = 0; i < n_tokens; i++) {
      if (!tokens[i] || !types || types[i] == 1) continue;
      int slen = (int)strlen(tokens[i]);
      tok->specials[tok->n_specials].text = strdup(tokens[i]);
      if (!tok->specials[tok->n_specials].text) goto fail;
      tok->specials[tok->n_specials].text_len = slen;
      tok->specials[tok->n_specials].token_id = i;
      tok->n_specials++;
    }
  }

  return tok;
fail:
  poly_tokenizer_free(tok);
  return NULL;
}

/* Retain the ABI entry point to fail explicitly instead of mis-tokenizing. */
PolyTokenizer *poly_tokenizer_from_json(const char *json_data, int json_len) {
  (void)json_data;
  (void)json_len;
  fprintf(
      stderr, "polygrad: tokenizer.json is unsupported; use Hugging Face tokenizers "
              "for JSON pipelines, or poly_tokenizer_from_gguf for GGUF BPE\n"
  );
  return NULL;
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
  if (types_raw && n_types == n_tokens) types = (int *)types_raw;

  PolyTokenizer *tok = poly_tokenizer_create(tokens, types, n_tokens);
  if (!tok) return NULL;

  tok->bos_id = poly_gguf_kv_int(gguf, "tokenizer.ggml.bos_token_id", -1);
  tok->eos_id = poly_gguf_kv_int(gguf, "tokenizer.ggml.eos_token_id", -1);

  /* Detect preset from tokenizer.ggml.pre */
  const char *pre = poly_gguf_kv_string(gguf, "tokenizer.ggml.pre", "");
  if (strcmp(pre, "gpt-2") == 0) tok->preset = POLY_TOK_PRESET_GPT2;
  /* "llama-bpe", "qwen2", "llama3" etc. all use LLAMA preset (default) */

  return tok;
}

void poly_tokenizer_free(PolyTokenizer *tok) {
  if (!tok) return;
  vocab_map_free(&tok->normal);
  for (int i = 0; tok->id_to_bytes && i < tok->vocab_size; i++)
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
  if (ctx->count < 0) return;
  int remaining = ctx->max_ids - ctx->count;
  if (remaining <= 0) return;
  int n = bpe_encode_word(ctx->tok, word, len, ctx->ids ? ctx->ids + ctx->count : NULL, remaining);
  ctx->count = n < 0 ? -1 : ctx->count + n;
}

/* Encode a chunk of normal text (no special tokens) via word split + BPE */
static int encode_chunk(
    const PolyTokenizer *tok,
    const uint8_t *text,
    int len,
    int *ids_out,
    int max_ids
) {
  EncodeCtx ctx = {.tok = tok, .ids = ids_out, .max_ids = max_ids, .count = 0};
  if (split_to_words(text, len, tok->preset, encode_word_cb, &ctx) < 0) ctx.count = -1;
  return ctx.count;
}

/* Find the earliest special token match in text[pos..end) */
static int find_special(
    const PolyTokenizer *tok,
    const char *text,
    int pos,
    int end,
    int *match_len,
    int *token_id
) {
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
        break; /* first occurrence of this special token */
      }
    }
  }
  if (best_id >= 0) {
    *match_len = best_len;
    *token_id = best_id;
  }
  return best_pos;
}

int poly_tokenize(const PolyTokenizer *tok, const char *text, int *ids_out, int max_ids) {
  if (!tok || !text || max_ids < 0 || strlen(text) >= INT32_MAX) return -1;
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
    int sp = (tok->n_specials > 0) ? find_special(tok, text, pos, text_len, &match_len, &token_id)
                                   : text_len;

    /* Encode text before the special token */
    if (sp > pos) {
      int remaining = max_ids - count;
      int n = encode_chunk(
          tok, (const uint8_t *)text + pos, sp - pos, ids_out ? ids_out + count : NULL, remaining
      );
      if (n < 0) return -1;
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

int poly_detokenize(
    const PolyTokenizer *tok,
    const int *ids,
    int n_ids,
    char *text_out,
    int max_len
) {
  if (!tok || n_ids < 0 || (!ids && n_ids)) return -1;
  int pos = 0;
  for (int i = 0; i < n_ids; i++) {
    int id = ids[i];
    if (id < 0 || id >= tok->vocab_size) continue;
    int blen = tok->id_to_len[id];
    if (blen > INT32_MAX - pos) return -1;
    if (text_out && pos + blen < max_len)
      memcpy(text_out + pos, tok->id_to_bytes[id], (size_t)blen);
    pos += blen;
  }
  if (text_out && max_len > 0) text_out[pos < max_len ? pos : max_len - 1] = '\0';
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
