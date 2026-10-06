/*
 * tokenizer.c -- BPE tokenizer
 *
 * GGUF: pinned tinygrad/llm/cli.py SimpleTokenizer, vocabulary-ranked BPE.
 * HF JSON supports validated byte-level BPE pipelines with explicit merge ranks.
 */

#define _POSIX_C_SOURCE 200809L
#include "tokenizer.h"
#include "loaders/gguf_decode.h"
#include "tokenizer_unicode.h"
#include "../vendor/cjson/cJSON.h"
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
#define POLY_TOK_PRESET_QWEN 2 /* JSON Qwen split: one digit */
#define POLY_TOK_PRESET_NONE 3 /* ByteLevel without regex */

struct PolyTokenizer {
  VocabMap normal; /* byte_seq -> token_id for normal tokens */
  VocabMap merges; /* JSON only: pair of IDs -> merge rank */
  int *merge_ids; /* merge rank -> output ID */
  bool explicit_merges, add_prefix_space;
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

/* GGUF follows pinned SimpleTokenizer's vocabulary ranks. HF JSON uses
 * pair-specific merge ranks: existence of the concatenated token is not enough. */
static int bpe_encode_word(
    const PolyTokenizer *tok,
    const uint8_t *word,
    int word_len,
    int *ids_out,
    int max_ids
) {
  if (word_len <= 0) return 0;
  int whole = vocab_map_find(&tok->normal, word, word_len);
  if (!tok->explicit_merges && whole >= 0) {
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
      if (tok->explicit_merges) {
        int pair[] = {parts[j], parts[j + 1]};
        rank = vocab_map_find(&tok->merges, (uint8_t *)pair, sizeof(pair));
        id = rank >= 0 ? tok->merge_ids[rank] : -1;
      } else {
        int left = parts[j], right = parts[j + 1], len = tok->id_to_len[left];
        memcpy(merged, tok->id_to_bytes[left], (size_t)len);
        memcpy(merged + len, tok->id_to_bytes[right], (size_t)tok->id_to_len[right]);
        rank = id = vocab_map_find(&tok->normal, merged, len + tok->id_to_len[right]);
      }
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
  if (preset == POLY_TOK_PRESET_NONE) {
    cb(text, text_len, ctx);
    free(cp);
    free(off);
    return 0;
  }
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
      int limit = preset == POLY_TOK_PRESET_GPT2 ? n : j + (preset == POLY_TOK_PRESET_QWEN ? 1 : 3);
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

static bool json_type(const cJSON *obj, const char *name) {
  const cJSON *t = cJSON_GetObjectItemCaseSensitive(obj, "type");
  return cJSON_IsString(t) && !strcmp(t->valuestring, name);
}
static bool json_off(const cJSON *obj, const char *name) {
  const cJSON *v = cJSON_GetObjectItemCaseSensitive(obj, name);
  return !v || cJSON_IsNull(v) || cJSON_IsFalse(v);
}
static bool json_empty(const cJSON *obj, const char *name) {
  const cJSON *v = cJSON_GetObjectItemCaseSensitive(obj, name);
  return json_off(obj, name) || (cJSON_IsString(v) && !v->valuestring[0]);
}
static bool json_bool_default(const cJSON *obj, const char *name, bool fallback, bool *out) {
  const cJSON *v = cJSON_GetObjectItemCaseSensitive(obj, name);
  if (!v) {
    *out = fallback;
    return true;
  }
  if (!cJSON_IsBool(v)) return false;
  *out = cJSON_IsTrue(v);
  return true;
}
static bool json_id(const cJSON *v, int limit) {
  return cJSON_IsNumber(v) && v->valuedouble == v->valueint && v->valueint >= 0 &&
         v->valueint < limit;
}
static const char *qwen_pattern =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| "
    "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
static const char *llama_pattern =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| "
    "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

static bool json_pipeline(
    const cJSON *root,
    int *preset,
    bool *prefix,
    bool *nfc,
    const char **error
) {
  const cJSON *normal = cJSON_GetObjectItemCaseSensitive(root, "normalizer");
  *nfc = json_type(normal, "NFC");
  *error = "unsupported normalizer (only NFC can be skipped with strict=false)";
  if (normal && !cJSON_IsNull(normal) && !*nfc) return false;
  *error = "unsupported post-processor or decoder (expected ByteLevel)";
  const char *parts[] = {"post_processor", "decoder"};
  for (int i = 0; i < 2; i++) {
    const cJSON *v = cJSON_GetObjectItemCaseSensitive(root, parts[i]);
    if (v && !cJSON_IsNull(v) && !json_type(v, "ByteLevel")) return false;
  }
  *error = "padding and truncation are unsupported";
  if (!json_off(root, "padding") || !json_off(root, "truncation")) return false;
  *error = "unsupported pre-tokenizer or split pattern";
  const cJSON *pre = cJSON_GetObjectItemCaseSensitive(root, "pre_tokenizer");
  *preset = POLY_TOK_PRESET_LLAMA;
  *prefix = false;
  if (!pre || cJSON_IsNull(pre)) return false;
  bool sequence = json_type(pre, "Sequence");
  if (sequence) {
    const cJSON *seq = cJSON_GetObjectItemCaseSensitive(pre, "pretokenizers");
    if (!cJSON_IsArray(seq) || cJSON_GetArraySize(seq) != 2) return false;
    const cJSON *split = cJSON_GetArrayItem(seq, 0);
    const cJSON *pattern = cJSON_GetObjectItemCaseSensitive(
        cJSON_GetObjectItemCaseSensitive(split, "pattern"), "Regex"
    );
    const cJSON *behavior = cJSON_GetObjectItemCaseSensitive(split, "behavior");
    if (!json_type(split, "Split") || !cJSON_IsString(pattern) || !cJSON_IsString(behavior) ||
        strcmp(behavior->valuestring, "Isolated") || !json_off(split, "invert"))
      return false;
    if (!strcmp(pattern->valuestring, qwen_pattern))
      *preset = POLY_TOK_PRESET_QWEN;
    else if (!strcmp(pattern->valuestring, llama_pattern))
      *preset = POLY_TOK_PRESET_LLAMA;
    else
      return false;
    pre = cJSON_GetArrayItem(seq, 1);
  }
  bool regex;
  if (!json_type(pre, "ByteLevel") || !json_bool_default(pre, "use_regex", true, &regex) ||
      !json_bool_default(pre, "add_prefix_space", true, prefix))
    return false;
  /* A second regex/prefix pass after Split would change token boundaries. */
  if (sequence) return !regex && !*prefix;
  *preset = regex ? POLY_TOK_PRESET_GPT2 : POLY_TOK_PRESET_NONE;
  return true;
}

PolyTokenizer *poly_tokenizer_from_json_ex(
    const char *json_data,
    int json_len,
    int strict,
    char *diagnostic,
    int diagnostic_size
) {
  const char *error = "invalid tokenizer JSON";
  if (diagnostic && diagnostic_size > 0) snprintf(diagnostic, (size_t)diagnostic_size, "%s", error);
  if (!json_data || json_len <= 0) return NULL;
  const char *end = NULL;
  cJSON *root = cJSON_ParseWithLengthOpts(json_data, (size_t)json_len, &end, 0);
  if (!root) return NULL;
  while (end < json_data + json_len && (*end == ' ' || *end == '\n' || *end == '\r' || *end == '\t')
  )
    end++;
  if (end != json_data + json_len) {
    cJSON_Delete(root);
    return NULL;
  }
  cJSON *model = cJSON_GetObjectItemCaseSensitive(root, "model");
  cJSON *kind = cJSON_GetObjectItemCaseSensitive(model, "type");
  cJSON *vocab = cJSON_GetObjectItemCaseSensitive(model, "vocab");
  cJSON *added = cJSON_GetObjectItemCaseSensitive(root, "added_tokens");
  cJSON *merges = cJSON_GetObjectItemCaseSensitive(model, "merges");
  PolyTokenizer *tok = NULL;
  const char **tokens = NULL;
  int *types = NULL;
  VocabMap names = {0};
  int preset = 0;
  bool prefix = false, nfc = false;
  error = "expected a byte-level BPE model with vocab and explicit merges";
  /* Older HF GPT-2 files omit type; an explicit merges array identifies BPE. */
  if (!cJSON_IsObject(root) || (kind && !json_type(model, "BPE")) || !cJSON_IsObject(vocab) ||
      !cJSON_IsArray(merges) || (added && !cJSON_IsArray(added)))
    goto fail;
  if (!json_pipeline(root, &preset, &prefix, &nfc, &error)) goto fail;
  error =
      "NFC normalization is unsupported; strict=false skips normalization and may change token IDs";
  if (nfc && strict) goto fail;
  error = "unsupported BPE options (dropout, unknown tokens, suffixes or ignore_merges)";
  const char *features[] = {"dropout",
                            "unk_token",
                            "continuing_subword_prefix",
                            "end_of_word_suffix",
                            "fuse_unk",
                            "byte_fallback",
                            "ignore_merges"};
  for (size_t i = 0; i < sizeof(features) / sizeof(*features); i++)
    if (!json_empty(model, features[i])) goto fail;

  error = "unsupported decoder (expected ByteLevel)";
  if (!json_type(cJSON_GetObjectItemCaseSensitive(root, "decoder"), "ByteLevel")) goto fail;
  error = "invalid vocabulary or added-token configuration";
  int limit = cJSON_GetArraySize(vocab) + cJSON_GetArraySize(added);
  if (limit <= 0 || limit > INT32_MAX / 2 - 1) goto fail;
  tokens = calloc((size_t)limit, sizeof(*tokens));
  types = malloc((size_t)limit * sizeof(*types));
  if (!tokens || !types) goto fail;
  for (int i = 0; i < limit; i++)
    types[i] = 1;
  int n_tokens = 0;
  cJSON *item;
  cJSON_ArrayForEach(item, vocab) {
    if (!json_id(item, limit) || tokens[item->valueint] || !item->string[0]) goto fail;
    tokens[item->valueint] = item->string;
    if (item->valueint >= n_tokens) n_tokens = item->valueint + 1;
  }
  int added_phase = -1;
  cJSON_ArrayForEach(item, added) {
    cJSON *id = cJSON_GetObjectItemCaseSensitive(item, "id");
    cJSON *content = cJSON_GetObjectItemCaseSensitive(item, "content");
    if (!json_id(id, limit) || !cJSON_IsString(content) || !content->valuestring[0] ||
        !json_off(item, "single_word") || !json_off(item, "lstrip") || !json_off(item, "rstrip") ||
        !cJSON_IsBool(cJSON_GetObjectItemCaseSensitive(item, "normalized")))
      goto fail;
    /* HF extracts unnormalized tokens before normalized ones. A single
     * longest-match pass is equivalent only when all tokens use one phase. */
    int phase = cJSON_IsTrue(cJSON_GetObjectItemCaseSensitive(item, "normalized"));
    if (added_phase >= 0 && added_phase != phase) {
      error = "mixed added-token matching phases are unsupported";
      goto fail;
    }
    added_phase = phase;
    if (tokens[id->valueint] && strcmp(tokens[id->valueint], content->valuestring)) goto fail;
    bool in_vocab = tokens[id->valueint] != NULL;
    tokens[id->valueint] = content->valuestring;
    /* Added non-special tokens also bypass BPE. This API decodes every ID,
     * so HF's special flag (skip-on-decode) does not change our behavior. */
    types[id->valueint] = in_vocab ? 1 : 3;
    if (id->valueint >= n_tokens) n_tokens = id->valueint + 1;
  }
  for (int i = 0; i < n_tokens; i++)
    if (!tokens[i]) goto fail;
  tok = poly_tokenizer_create(tokens, types, n_tokens);
  if (!tok) goto fail;
  /* Added tokens may shadow existing BPE tokens. Keep those tokens available
   * to the merge table while matching their literal spelling before BPE. */
  cJSON_ArrayForEach(item, added) {
    int id = cJSON_GetObjectItemCaseSensitive(item, "id")->valueint;
    if (types[id] == 1) {
      SpecialToken *sp = realloc(tok->specials, (size_t)(tok->n_specials + 1) * sizeof(*sp));
      if (!sp) goto fail;
      tok->specials = sp;
      char *literal = strdup(tokens[id]);
      if (!literal) goto fail;
      tok->specials[tok->n_specials++] = (SpecialToken){literal, (int)strlen(literal), id};
    }
    int len = (int)strlen(tokens[id]);
    uint8_t *decoded = malloc((size_t)len);
    if (!decoded) goto fail;
    /* HF ByteLevel::decode_chain converts an entire token or, if any
     * character is outside the byte alphabet, preserves the entire UTF-8
     * spelling. This also applies to literal added tokens. */
    int n = token_str_to_bytes(tokens[id], decoded, len);
    if (n < 0) {
      memcpy(decoded, tokens[id], (size_t)len);
      n = len;
    }
    free(tok->id_to_bytes[id]);
    tok->id_to_bytes[id] = decoded;
    tok->id_to_len[id] = n;
  }
  tok->preset = preset;
  tok->add_prefix_space = prefix;
  /* JSON BPE must cover every byte: without unk_token, HF would otherwise
   * silently discard bytes that this API promises to encode. */
  for (int b = 0; b < 256; b++) {
    uint8_t byte = (uint8_t)b;
    if (vocab_map_find(&tok->normal, &byte, 1) < 0) {
      error = "byte-level BPE vocabulary must contain all 256 bytes";
      goto fail;
    }
  }
  error = "invalid BPE merge table";
  vocab_map_init(&names, n_tokens * 2 + 1);
  if (!names.entries) goto fail;
  for (int i = 0; i < n_tokens; i++)
    if (types[i] == 1 &&
        !vocab_map_insert(&names, (const uint8_t *)tokens[i], (int)strlen(tokens[i]), i))
      goto fail;
  int count = cJSON_GetArraySize(merges);
  if (count > INT32_MAX / 2 - 1) goto fail;
  vocab_map_init(&tok->merges, count * 2 + 1);
  tok->merge_ids = calloc((size_t)count + 1, sizeof(*tok->merge_ids));
  if (!tok->merges.entries || !tok->merge_ids) goto fail;
  tok->explicit_merges = true;
  int rank = 0;
  cJSON_ArrayForEach(item, merges) {
    const char *left, *right;
    int llen, rlen;
    if (cJSON_IsString(item)) {
      left = item->valuestring;
      right = strchr(left, ' ');
      if (!right) goto fail;
      llen = (int)(right++ - left);
      rlen = (int)strlen(right);
    } else if (cJSON_IsArray(item) && cJSON_GetArraySize(item) == 2 && cJSON_IsString(cJSON_GetArrayItem(item, 0)) && cJSON_IsString(cJSON_GetArrayItem(item, 1))) {
      left = cJSON_GetArrayItem(item, 0)->valuestring;
      right = cJSON_GetArrayItem(item, 1)->valuestring;
      llen = (int)strlen(left);
      rlen = (int)strlen(right);
    } else
      goto fail;
    int pair[] = {
        vocab_map_find(&names, (const uint8_t *)left, llen),
        vocab_map_find(&names, (const uint8_t *)right, rlen)};
    if (pair[0] < 0 || pair[1] < 0 ||
        vocab_map_find(&tok->merges, (uint8_t *)pair, sizeof(pair)) >= 0)
      goto fail;
    char *joined = malloc((size_t)llen + rlen);
    if (!joined) goto fail;
    memcpy(joined, left, (size_t)llen);
    memcpy(joined + llen, right, (size_t)rlen);
    int id = vocab_map_find(&names, (const uint8_t *)joined, llen + rlen);
    free(joined);
    if (id < 0) goto fail;
    tok->merge_ids[rank] = id;
    if (!vocab_map_insert(&tok->merges, (uint8_t *)pair, sizeof(pair), rank++)) goto fail;
  }
  /* added_tokens does not define BOS/EOS roles. The diagnostic is a warning
   * only on successful best-effort import, never permission to ignore more. */
  if (diagnostic && diagnostic_size > 0)
    snprintf(
        diagnostic, (size_t)diagnostic_size, "%s",
        nfc ? "NFC normalization skipped; token IDs may differ from Hugging Face" : ""
    );
  vocab_map_free(&names);
  free(tokens);
  free(types);
  cJSON_Delete(root);
  return tok;
fail:
  if (diagnostic && diagnostic_size > 0) snprintf(diagnostic, (size_t)diagnostic_size, "%s", error);
  poly_tokenizer_free(tok);
  vocab_map_free(&names);
  free(tokens);
  free(types);
  cJSON_Delete(root);
  return NULL;
}

/* Existing callers retain strict behavior and receive the same diagnostic. */
PolyTokenizer *poly_tokenizer_from_json(const char *json_data, int json_len) {
  char diagnostic[256];
  PolyTokenizer *tok =
      poly_tokenizer_from_json_ex(json_data, json_len, 1, diagnostic, sizeof(diagnostic));
  if (!tok)
    fprintf(
        stderr, "polygrad: %s; use Hugging Face tokenizers for unsupported pipelines\n", diagnostic
    );
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
  vocab_map_free(&tok->merges);
  free(tok->merge_ids);
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
  uint8_t *prefixed = NULL;
  if (tok->add_prefix_space && len && text[0] != ' ') {
    prefixed = malloc((size_t)len + 1);
    if (!prefixed) return -1;
    prefixed[0] = ' ';
    memcpy(prefixed + 1, text, (size_t)len);
    text = prefixed;
    len++;
  }
  if (split_to_words(text, len, tok->preset, encode_word_cb, &ctx) < 0) ctx.count = -1;
  free(prefixed);
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
