/* PyTorch ZIP state-dict reader. Pickle is interpreted as a bounded data grammar:
 * GLOBAL/REDUCE identify known records, never resolve symbols or call functions.
 * No Python, filesystem extraction, dynamic imports or device allocation. */
#include "torch_decode.h"
#include "import_error.h"
#include <limits.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#ifdef POLY_TESTING
static int alloc_fail_after = -1;
void poly_test_torch_alloc_fail_after(int n) {
  alloc_fail_after = n;
}
static bool allocation_fails(void) {
  if (alloc_fail_after < 0) return false;
  if (!alloc_fail_after) return true;
  alloc_fail_after--;
  return false;
}
static void *checked_malloc(size_t n) {
  return allocation_fails() ? NULL : malloc(n);
}
static void *checked_calloc(size_t n, size_t s) {
  return allocation_fails() ? NULL : calloc(n, s);
}
static void *checked_realloc(void *p, size_t n) {
  return allocation_fails() ? NULL : realloc(p, n);
}
#define malloc checked_malloc
#define calloc checked_calloc
#define realloc checked_realloc
#endif

#define METADATA_LIMIT (64u * 1024u * 1024u)
#define PACKED_LIMIT (256u * 1024u * 1024u)
#define ITEM_LIMIT 1000000u
#define ZIP_LIMIT 100000u

typedef struct {
  const uint8_t *s;
  size_t n;
} Span;
typedef struct {
  Span name, bytes;
  uint32_t crc;
  bool checked;
  int dtype;
  uint64_t start, end;
} Entry;
typedef enum { NIL, INT, STR, TUPLE, DICT, LIST, GLOBAL, STORAGE, TENSOR } Kind;
typedef struct Value Value;
struct Value {
  Kind kind;
  int64_t number;
  Span str;
  Value **items;
  size_t n, cap;
};
typedef struct {
  const uint8_t *data;
  size_t len, pos, frame_end, budget;
  Value **values, **stack, **memo;
  size_t nv, ns, nm, capacity;
  Entry *entries;
  size_t ne;
  Span prefix;
  bool swap;
  uint32_t crc_table[256];
  const char *error;
  char error_detail[192];
} Parser;

static uint16_t u16(const uint8_t *p) {
  return (uint16_t)p[0] | (uint16_t)p[1] << 8;
}
static uint32_t u32(const uint8_t *p) {
  return (uint32_t)u16(p) | (uint32_t)u16(p + 2) << 16;
}
static uint64_t u64(const uint8_t *p) {
  return (uint64_t)u32(p) | (uint64_t)u32(p + 4) << 32;
}
static bool equal(Span s, const char *text) {
  return s.n == strlen(text) && !memcmp(s.s, text, s.n);
}
static int span_cmp(Span a, Span b) {
  int c = memcmp(a.s, b.s, a.n < b.n ? a.n : b.n);
  return c ? c : (a.n > b.n) - (a.n < b.n);
}
static int entry_cmp(const void *a, const void *b) {
  return span_cmp(((const Entry *)a)->name, ((const Entry *)b)->name);
}
static int entry_offset_cmp(const void *a, const void *b) {
  uint64_t x = ((const Entry *)a)->start, y = ((const Entry *)b)->start;
  return (x > y) - (x < y);
}
static bool fail(Parser *p, const char *error) {
  if (!p->error) p->error = error;
  return false;
}
static bool range(size_t len, uint64_t off, uint64_t n) {
  return off <= len && n <= len - off;
}
static const uint8_t *take(Parser *p, size_t n) {
  size_t end = p->frame_end ? p->frame_end : p->len;
  if (!range(end, p->pos, n)) {
    fail(p, "truncated pickle or frame");
    return NULL;
  }
  const uint8_t *v = p->data + p->pos;
  p->pos += n;
  return v;
}
static bool reserve(Parser *p, void **ptr, size_t old, size_t count, size_t size) {
  if (count > ITEM_LIMIT || count < old || size > METADATA_LIMIT / (count ? count : 1) ||
      (count - old) * size > METADATA_LIMIT - p->budget)
    return fail(p, "metadata resource limit");
  void *next = realloc(*ptr, count * size);
  if (!next) return fail(p, "allocation failed");
  *ptr = next;
  p->budget += (count - old) * size;
  return true;
}
static Value *value(Parser *p, Kind kind) {
  if (p->nv >= p->capacity || sizeof(Value) > METADATA_LIMIT - p->budget) {
    fail(p, "pickle object limit");
    return NULL;
  }
  Value *v = calloc(1, sizeof(*v));
  if (!v) {
    fail(p, "allocation failed");
    return NULL;
  }
  p->budget += sizeof(*v);
  v->kind = kind;
  p->values[p->nv++] = v;
  return v;
}
static bool append(Parser *p, Value *v, Value *item) {
  if (!item) return fail(p, "unexpected pickle mark");
  if (v->n == v->cap) {
    size_t cap = v->cap ? v->cap * 2 : 4;
    if (!reserve(p, (void **)&v->items, v->cap, cap, sizeof(*v->items))) return false;
    v->cap = cap;
  }
  v->items[v->n++] = item;
  return true;
}
static bool push(Parser *p, Value *v) {
  if (p->ns == p->capacity) return fail(p, "pickle stack limit");
  p->stack[p->ns++] = v;
  return true;
}
static Value *pop(Parser *p) {
  if (!p->ns) {
    fail(p, "pickle stack underflow");
    return NULL;
  }
  Value *v = p->stack[--p->ns];
  if (!v) fail(p, "unexpected pickle mark");
  return v;
}
static bool integer(Value *v, int64_t *n) {
  if (!v || v->kind != INT) return false;
  *n = v->number;
  return true;
}
static bool text(Value *v, const char *s) {
  return v && v->kind == STR && equal(v->str, s);
}
static Entry *lookup(Parser *p, Span suffix) {
  /* Avoid path interpolation and a temporary allocation on every storage lookup. */
  for (size_t i = 0; i < suffix.n; i++)
    if (!suffix.s[i] || suffix.s[i] == '\\') return NULL;
  size_t lo = 0, hi = p->ne;
  while (lo < hi) {
    size_t m = lo + (hi - lo) / 2;
    Span name = p->entries[m].name;
    int c = memcmp(name.s, p->prefix.s, name.n < p->prefix.n ? name.n : p->prefix.n);
    if (!c)
      c = name.n < p->prefix.n
              ? -1
              : span_cmp((Span){name.s + p->prefix.n, name.n - p->prefix.n}, suffix);
    if (c < 0)
      lo = m + 1;
    else if (c > 0)
      hi = m;
    else
      return &p->entries[m];
  }
  return NULL;
}
static Entry *named(Parser *p, const char *s) {
  return lookup(p, (Span){(const uint8_t *)s, strlen(s)});
}
static bool checksum(Parser *p, Entry *entry) {
  if (entry->checked) return true;
  uint32_t crc = UINT32_MAX;
  for (size_t i = 0; i < entry->bytes.n; i++)
    crc = p->crc_table[(crc ^ entry->bytes.s[i]) & 255] ^ (crc >> 8);
  if ((crc ^ UINT32_MAX) != entry->crc) return fail(p, "ZIP checksum mismatch");
  entry->checked = true;
  return true;
}
static bool zip_extra(Span extra, uint64_t *size, uint64_t *compressed, uint64_t *offset) {
  bool found = false;
  for (size_t at = 0; at < extra.n;) {
    if (!range(extra.n, at, 4)) return false;
    unsigned tag = u16(extra.s + at), n = u16(extra.s + at + 2);
    at += 4;
    if (!range(extra.n, at, n)) return false;
    if (tag == 1) {
      if (found) return false;
      found = true;
      size_t used = 0;
      uint64_t *fields[] = {size, compressed, offset};
      for (int i = 0; i < 3; i++)
        if (*fields[i] == UINT32_MAX) {
          if (!range(n, used, 8)) return false;
          *fields[i] = u64(extra.s + at + used);
          used += 8;
        }
    }
    at += n;
  }
  return *size != UINT32_MAX && *compressed != UINT32_MAX && *offset != UINT32_MAX;
}
static bool zip(Parser *p, const uint8_t *data, size_t len) {
  for (unsigned i = 0; i < 256; i++) {
    uint32_t c = i;
    for (int j = 0; j < 8; j++)
      c = (c >> 1) ^ ((c & 1) ? 0xedb88320u : 0);
    p->crc_table[i] = c;
  }
  if (len < 22) return fail(p, "not a ZIP checkpoint (legacy serialization is unsupported)");
  size_t end = len - 22, floor = len > 65557 ? len - 65557 : 0;
  while (u32(data + end) != 0x06054b50 || end + 22 + u16(data + end + 20) != len) {
    if (end == floor) return fail(p, "missing ZIP end record");
    end--;
  }
  const uint8_t *e = data + end;
  if (u16(e + 4) || u16(e + 6) || u16(e + 8) != u16(e + 10))
    return fail(p, "multi-disk ZIP unsupported");
  uint64_t count = u16(e + 10), bytes = u32(e + 12), offset = u32(e + 16);
  size_t directory_end = end;
  if (end >= 20 && u32(data + end - 20) == 0x07064b50) {
    const uint8_t *locator = data + end - 20;
    uint64_t zoff = u64(locator + 8);
    if (u32(locator + 4) || u32(locator + 16) != 1 || !range(end - 20, zoff, 56))
      return fail(p, "invalid ZIP64 locator");
    const uint8_t *z = data + (size_t)zoff;
    if (u32(z) != 0x06064b50 || u64(z + 4) < 44 || !range(end - 20, zoff + 12, u64(z + 4)) ||
        u32(z + 16) || u32(z + 20) || u64(z + 24) != u64(z + 32))
      return fail(p, "invalid ZIP64 directory");
    count = u64(z + 32);
    bytes = u64(z + 40);
    offset = u64(z + 48);
    directory_end = (size_t)zoff;
  }
  if (!count || count > ZIP_LIMIT || !range(directory_end, offset, bytes) ||
      offset + bytes != directory_end)
    return fail(p, "invalid ZIP directory bounds");
  p->entries = calloc((size_t)count, sizeof(*p->entries));
  if (!p->entries) return fail(p, "allocation failed");
  p->ne = (size_t)count;
  size_t at = (size_t)offset;
  for (size_t i = 0; i < p->ne; i++) {
    if (!range(directory_end, at, 46) || u32(data + at) != 0x02014b50)
      return fail(p, "invalid ZIP entry");
    const uint8_t *c = data + at;
    unsigned flags = u16(c + 8), method = u16(c + 10);
    size_t nn = u16(c + 28), en = u16(c + 30), cn = u16(c + 32);
    if ((flags & ~0x808u) || method || u16(c + 34) || !nn ||
        !range(directory_end, at + 46, nn + en + cn))
      return fail(p, "only unencrypted, uncompressed ZIP entries are supported");
    Span name = {c + 46, nn};
    if (memchr(name.s, 0, nn) || name.s[0] == '/' || memchr(name.s, '\\', nn))
      return fail(p, "invalid ZIP name");
    for (size_t k = 0, start = 0; k <= nn; k++)
      if (k == nn || name.s[k] == '/') {
        Span part = {name.s + start, k - start};
        if (!part.n || equal(part, ".") || equal(part, "..")) return fail(p, "invalid ZIP path");
        start = k + 1;
      }
    uint64_t sz = u32(c + 24), compressed = u32(c + 20), local = u32(c + 42);
    if (!zip_extra((Span){c + 46 + nn, en}, &sz, &compressed, &local) || sz != compressed ||
        !range((size_t)offset, local, 30))
      return fail(p, "invalid ZIP sizes");
    const uint8_t *h = data + (size_t)local;
    size_t hn = u16(h + 26), he = u16(h + 28);
    uint64_t begin = local + 30 + hn + he;
    if (u32(h) != 0x04034b50 || u16(h + 6) != flags || u16(h + 8) != method || hn != nn ||
        !range((size_t)offset, local + 30, hn + he) || memcmp(h + 30, name.s, nn) ||
        !range((size_t)offset, begin, sz))
      return fail(p, "inconsistent ZIP local header");
    if (!(flags & 8)) {
      uint64_t ls = u32(h + 22), lc = u32(h + 18), unused = 0;
      if (!zip_extra((Span){h + 30 + hn, he}, &ls, &lc, &unused) || ls != sz || lc != sz ||
          u32(h + 14) != u32(c + 16))
        return fail(p, "inconsistent ZIP local sizes");
    }
    p->entries[i] = (Entry
    ){name, {data + (size_t)begin, (size_t)sz}, u32(c + 16), false, -1, local, begin + sz};
    at += 46 + nn + en + cn;
  }
  if (at != directory_end) return fail(p, "invalid ZIP directory length");
  /* Apart from ambiguity, overlapping entries could amplify CRC scanning to
   * entries times file size. Real Torch archives use disjoint local records. */
  qsort(p->entries, p->ne, sizeof(*p->entries), entry_offset_cmp);
  for (size_t i = 1; i < p->ne; i++)
    if (p->entries[i].start < p->entries[i - 1].end) return fail(p, "overlapping ZIP entries");
  qsort(p->entries, p->ne, sizeof(*p->entries), entry_cmp);
  Entry *index = NULL;
  for (size_t i = 0; i < p->ne; i++) {
    Span n = p->entries[i].name;
    if (i && !span_cmp(n, p->entries[i - 1].name)) return fail(p, "duplicate ZIP entry");
    if (n.n >= 9 && !memcmp(n.s + n.n - 9, "/data.pkl", 9)) {
      if (index) return fail(p, "multiple pickle indexes");
      index = &p->entries[i];
      p->prefix = (Span){n.s, n.n - 8};
    }
  }
  if (!index || index->bytes.n > METADATA_LIMIT) return fail(p, "missing or oversized data.pkl");
  Entry *version = named(p, "version"), *order = named(p, "byteorder");
  if (!version || !(equal(version->bytes, "3\n") || equal(version->bytes, "2\n") ||
                    equal(version->bytes, "1\n")))
    return fail(p, "unsupported ZIP checkpoint version");
  if (!checksum(p, index) || !checksum(p, version) || (order && !checksum(p, order))) return false;
  bool little = true;
  if (order) {
    if (equal(order->bytes, "big"))
      little = false;
    else if (!equal(order->bytes, "little"))
      return fail(p, "invalid byte order");
  }
  const uint16_t one = 1;
  p->swap = little != (*(const uint8_t *)&one == 1);
  p->data = index->bytes.s;
  p->len = index->bytes.n;
  return true;
}

/* GLOBAL identifiers are exact module/name pairs. Even allowlisted symbols are
 * never called; REDUCE below validates their complete argument schema. */
enum { ORDERED = 100, REBUILD, REBUILD_V2 };
static Value *global(Parser *p, Span module, Span name) {
  int id = -1;
  if (equal(module, "collections") && equal(name, "OrderedDict")) id = ORDERED;
  if (equal(module, "torch._utils")) {
    if (equal(name, "_rebuild_tensor")) id = REBUILD;
    if (equal(name, "_rebuild_tensor_v2")) id = REBUILD_V2;
  }
  if (equal(module, "torch")) {
    const char *types[] = {"FloatStorage", "HalfStorage", "BFloat16Storage", "DoubleStorage",
                           "LongStorage",  "IntStorage",  "ShortStorage",    "CharStorage",
                           "ByteStorage",  "BoolStorage"};
    for (int i = 0; i < 10; i++)
      if (equal(name, types[i])) id = i;
  }
  if (id < 0) {
    /* File-controlled identifiers must not inject terminal controls or flood logs. */
    char *dst = p->error_detail;
    const char *prefix = "unsupported pickle global: ";
    memcpy(dst, prefix, strlen(prefix));
    dst += strlen(prefix);
    Span parts[] = {module, name};
    for (int j = 0; j < 2; j++) {
      if (j) *dst++ = '.';
      size_t n = parts[j].n < 64 ? parts[j].n : 64;
      for (size_t k = 0; k < n; k++) {
        uint8_t c = parts[j].s[k];
        *dst++ = c >= 33 && c <= 126 ? (char)c : '?';
      }
      if (parts[j].n > n) {
        memcpy(dst, "...", 3);
        dst += 3;
      }
    }
    *dst = 0;
    fail(p, p->error_detail);
    return NULL;
  }
  Value *v = value(p, GLOBAL);
  if (v) v->number = id;
  return v;
}
static bool line(Parser *p, Span *s) {
  size_t start = p->pos;
  const uint8_t *b;
  do {
    b = take(p, 1);
    if (!b) return false;
  } while (*b != '\n');
  *s = (Span){p->data + start, p->pos - start - 1};
  return true;
}
static bool empty(Value *v) {
  return v && (v->kind == NIL || (v->kind == DICT && !v->n));
}
static Value *reduce(Parser *p, Value *fn, Value *args) {
  if (!fn || fn->kind != GLOBAL || !args || args->kind != TUPLE) goto bad;
  if (fn->number == ORDERED) {
    if (args->n) goto bad;
    return value(p, DICT);
  }
  if (fn->number != REBUILD && fn->number != REBUILD_V2) goto bad;
  size_t n = args->n;
  if ((fn->number == REBUILD && n != 4) || (fn->number == REBUILD_V2 && n != 6 && n != 7)) goto bad;
  if (args->items[0]->kind != STORAGE || args->items[1]->kind != INT ||
      args->items[2]->kind != TUPLE || args->items[3]->kind != TUPLE)
    goto bad;
  if (n >= 6 &&
      (args->items[4]->kind != INT ||
       (args->items[4]->number != 0 && args->items[4]->number != 1) || !empty(args->items[5])))
    goto bad;
  if (n == 7 && !empty(args->items[6])) goto bad;
  Value *v = value(p, TENSOR);
  if (!v) return NULL;
  for (size_t i = 0; i < 4; i++)
    if (!append(p, v, args->items[i])) return NULL;
  return v;
bad:
  fail(p, "unsupported tensor constructor arguments");
  return NULL;
}
static Value *persistent(Parser *p, Value *args) {
  if (!args || args->kind != TUPLE || args->n != 5 || !text(args->items[0], "storage") ||
      args->items[1]->kind != GLOBAL || args->items[1]->number >= 10 ||
      args->items[2]->kind != STR || args->items[3]->kind != STR || args->items[4]->kind != INT ||
      args->items[4]->number < 0) {
    fail(p, "unsupported persistent storage");
    return NULL;
  }
  Value *v = value(p, STORAGE);
  if (!v) return NULL;
  v->number = args->items[1]->number;
  for (size_t i = 2; i < 5; i++)
    if (!append(p, v, args->items[i])) return NULL;
  return v;
}
static Value *pickle(Parser *p) {
  p->capacity = p->len < ITEM_LIMIT ? p->len + 1 : ITEM_LIMIT;
  if (!reserve(p, (void **)&p->values, 0, p->capacity, sizeof(Value *)) ||
      !reserve(p, (void **)&p->stack, 0, p->capacity, sizeof(Value *)) ||
      !reserve(p, (void **)&p->memo, 0, p->capacity, sizeof(Value *)))
    return NULL;
  memset(p->memo, 0, p->capacity * sizeof(Value *));
  const uint8_t *b = take(p, 2);
  if (!b || b[0] != 0x80 || b[1] < 2 || b[1] > 5) {
    fail(p, "only pickle protocols 2 through 5 are supported");
    return NULL;
  }
  while (p->pos < p->len && !p->error) {
    if (p->frame_end == p->pos) p->frame_end = 0;
    b = take(p, 1);
    if (!b) break;
    unsigned op = *b;
    Value *v = NULL;
    if (op == '.') {
      if (p->ns != 1 || !p->stack[0] || p->stack[0]->kind != DICT || p->pos != p->len)
        fail(p, "expected a single flat state dictionary");
      else
        return p->stack[0];
    } else if (op == 0x95) {
      if (p->frame_end) {
        fail(p, "nested pickle frame");
        break;
      }
      b = take(p, 8);
      if (!b || !range(p->len, p->pos, u64(b))) {
        fail(p, "invalid pickle frame");
        break;
      }
      p->frame_end = p->pos + (size_t)u64(b);
    } else if (op == '(') {
      push(p, NULL);
    } else if (op == 'N' || op == 0x88 || op == 0x89 || op == 'K' || op == 'M' || op == 'J' || op == 0x8a) {
      v = value(p, op == 'N' ? NIL : INT);
      if (!v) break;
      if (op == 0x88) v->number = 1;
      if (op == 'K' || op == 'M' || op == 'J') {
        b = take(p, op == 'K' ? 1 : op == 'M' ? 2 : 4);
        if (!b) break;
        v->number = op == 'K' ? b[0] : op == 'M' ? u16(b) : (int64_t)(int32_t)u32(b);
      } else if (op == 0x8a) {
        b = take(p, 1);
        if (!b) break;
        size_t n = *b;
        if (n > 8 || !(b = take(p, n))) {
          fail(p, "pickle integer exceeds int64");
          break;
        }
        uint64_t bits = 0;
        for (size_t i = 0; i < n; i++)
          bits |= (uint64_t)b[i] << (i * 8);
        if (n && n < 8 && (b[n - 1] & 128)) bits |= UINT64_MAX << (n * 8);
        memcpy(&v->number, &bits, 8);
      }
      push(p, v);
    } else if (op == 'X' || op == 0x8c || op == 0x8d) {
      b = take(p, op == 'X' ? 4 : op == 0x8c ? 1 : 8);
      if (!b) break;
      uint64_t n = op == 'X' ? u32(b) : op == 0x8c ? b[0] : u64(b);
      if (n > METADATA_LIMIT || !(b = take(p, (size_t)n))) break;
      v = value(p, STR);
      if (!v) break;
      v->str = (Span){b, (size_t)n};
      push(p, v);
    } else if (op == '}' || op == ')' || op == ']') {
      v = value(p, op == '}' ? DICT : op == ')' ? TUPLE : LIST);
      if (!v) break;
      push(p, v);
    } else if (op == 't' || op == 'u' || op == 'e' || op == 0x85 || op == 0x86 || op == 0x87) {
      bool marked = op == 't' || op == 'u' || op == 'e';
      size_t start = p->ns;
      if (marked) {
        while (start && p->stack[start - 1])
          start--;
        if (!start) {
          fail(p, "missing pickle mark");
          break;
        }
      } else {
        size_t n = op - 0x84;
        if (start < n) {
          fail(p, "tuple stack underflow");
          break;
        }
        start -= n;
      }
      if (op == 'u' || op == 'e') {
        if (start < 2 || !(v = p->stack[start - 2]) || v->kind != (op == 'u' ? DICT : LIST) ||
            (op == 'u' && (p->ns - start) % 2)) {
          fail(p, "invalid pickle container update");
          break;
        }
      } else
        v = value(p, TUPLE);
      if (!v) break;
      for (size_t i = start; i < p->ns; i++)
        if (!append(p, v, p->stack[i])) break;
      p->ns = marked ? start - 1 : start;
      if (op != 'u' && op != 'e') push(p, v);
    } else if (op == 's' || op == 'a') {
      size_t n = op == 's' ? 2 : 1;
      if (p->ns < n + 1 || !(v = p->stack[p->ns - n - 1]) || v->kind != (op == 's' ? DICT : LIST)) {
        fail(p, "invalid pickle container update");
        break;
      }
      for (size_t i = p->ns - n; i < p->ns; i++)
        if (!append(p, v, p->stack[i])) break;
      p->ns -= n;
    } else if (op == 'q' || op == 'r' || op == 'h' || op == 'j' || op == 0x94) {
      size_t idx = p->nm;
      if (op != 0x94) {
        b = take(p, op == 'q' || op == 'h' ? 1 : 4);
        if (!b) break;
        idx = op == 'q' || op == 'h' ? *b : u32(b);
      }
      if (idx >= p->capacity) {
        fail(p, "pickle memo limit");
        break;
      }
      if (op == 'h' || op == 'j') {
        if (!p->memo[idx]) {
          fail(p, "missing pickle memo");
          break;
        }
        push(p, p->memo[idx]);
      } else {
        if (!p->ns || !p->stack[p->ns - 1] || p->memo[idx]) {
          fail(p, "invalid pickle memo write");
          break;
        }
        p->memo[idx] = p->stack[p->ns - 1];
        if (idx >= p->nm) p->nm = idx + 1;
      }
    } else if (op == 'c' || op == 0x93) {
      Span module, name;
      if (op == 'c') {
        if (!line(p, &module) || !line(p, &name)) break;
      } else {
        Value *nv = pop(p), *mv = pop(p);
        if (!nv || !mv || nv->kind != STR || mv->kind != STR) {
          fail(p, "invalid STACK_GLOBAL");
          break;
        }
        module = mv->str;
        name = nv->str;
      }
      v = global(p, module, name);
      if (!v) break;
      push(p, v);
    } else if (op == 'Q' || op == 'R') {
      Value *args = pop(p);
      v = op == 'Q' ? persistent(p, args) : reduce(p, pop(p), args);
      if (!v) break;
      push(p, v);
    } else if (op == 'b') {
      /* nn.Module.state_dict attaches harmless version metadata to OrderedDict.
       * It never changes tensor interpretation; arbitrary object BUILD is forbidden. */
      Value *attrs = pop(p);
      if (!p->ns || !p->stack[p->ns - 1] || p->stack[p->ns - 1]->kind != DICT || !attrs ||
          attrs->kind != DICT || attrs->n != 2 || !text(attrs->items[0], "_metadata") ||
          attrs->items[1]->kind != DICT)
        fail(p, "unsupported pickle BUILD");
    } else
      fail(p, "unsupported pickle opcode");
  }
  fail(p, "missing pickle STOP");
  return NULL;
}

static bool tensor(Parser *p, Value *name, Value *v, PolyDecodedTensor *out, size_t *packed) {
  if (!name || name->kind != STR || !name->str.n || memchr(name->str.s, 0, name->str.n) || !v ||
      v->kind != TENSOR)
    return fail(p, "state dictionary must map names to tensors");
  Value *storage = v->items[0], *shape = v->items[2], *strides = v->items[3];
  int64_t offset = v->items[1]->number, size = storage->items[2]->number;
  if (offset < 0 || shape->n > 8 || shape->n != strides->n) return fail(p, "invalid tensor layout");
  Span key = storage->items[0]->str;
  if (!key.n || key.n > 128) return fail(p, "invalid storage key");
  uint8_t path[133];
  memcpy(path, "data/", 5);
  for (size_t i = 0; i < key.n; i++) {
    if (key.s[i] < '0' || key.s[i] > '9') return fail(p, "unsupported storage key");
    path[5 + i] = key.s[i];
  }
  Entry *entry = lookup(p, (Span){path, key.n + 5});
  const size_t widths[] = {4, 2, 2, 8, 8, 4, 2, 1, 1, 1};
  size_t width = widths[storage->number];
  if (!entry || (uint64_t)size > SIZE_MAX / width || entry->bytes.n != (size_t)size * width)
    return fail(p, "storage size does not match ZIP payload");
  if (entry->dtype != -1 && entry->dtype != storage->number)
    return fail(p, "conflicting shared storage dtype");
  entry->dtype = (int)storage->number;
  if (!checksum(p, entry)) return false;
  uint64_t numel = 1, last = (uint64_t)offset;
  int64_t stride[8];
  bool empty_tensor = false;
  for (size_t i = 0; i < shape->n; i++) {
    int64_t d, st;
    if (!integer(shape->items[i], &d) || !integer(strides->items[i], &st) || d < 0 || st < 0)
      return fail(p, "negative or noninteger shape/stride");
    out->shape[i] = d;
    stride[i] = st;
    if (!d) empty_tensor = true;
  }
  for (size_t i = 0; i < shape->n; i++) {
    uint64_t d = (uint64_t)out->shape[i], st = (uint64_t)stride[i];
    if (!empty_tensor) {
      if (numel > INT64_MAX / d || (d - 1 && st > (UINT64_MAX - last) / (d - 1)))
        return fail(p, "tensor layout overflow");
      numel *= d;
      last += (d - 1) * st;
    }
  }
  if (empty_tensor) numel = 0;
  if ((uint64_t)offset > (uint64_t)size || (numel && last >= (uint64_t)size) ||
      numel > SIZE_MAX / width)
    return fail(p, "tensor view exceeds storage");
  bool contiguous = true;
  uint64_t expected = 1;
  for (size_t i = shape->n; i-- > 0;) {
    if (out->shape[i] > 1 && (uint64_t)stride[i] != expected) contiguous = false;
    if (out->shape[i] && expected > UINT64_MAX / (uint64_t)out->shape[i])
      contiguous = false;
    else
      expected *= (uint64_t)out->shape[i];
  }
  bool copy = numel && (!contiguous || (p->swap && width > 1));
  size_t nbytes = (size_t)numel * width, header = (name->str.n + 1 + 7) & ~(size_t)7;
  if (copy && (nbytes > PACKED_LIMIT - *packed || header > SIZE_MAX - nbytes))
    return fail(p, "packed tensor resource limit");
  if (header > METADATA_LIMIT - p->budget) return fail(p, "tensor name resource limit");
  p->budget += header;
  out->name = malloc(header + (copy ? nbytes : 0));
  if (!out->name) return fail(p, "allocation failed");
  memcpy(out->name, name->str.s, name->str.n);
  out->name[name->str.n] = 0;
  out->ndim = (int)shape->n;
  out->numel = (int64_t)numel;
  out->dtype = (int)storage->number;
  out->data = entry->bytes.s + (size_t)offset * width;
  if (copy) {
    uint8_t *dst = (uint8_t *)out->name + header;
    out->data = dst;
    *packed += nbytes;
    for (uint64_t i = 0; i < numel; i++) {
      uint64_t rem = i, at = (uint64_t)offset;
      for (size_t j = shape->n; j-- > 0;) {
        at += rem % (uint64_t)out->shape[j] * (uint64_t)stride[j];
        rem /= (uint64_t)out->shape[j];
      }
      for (size_t b = 0; b < width; b++)
        dst[i * width + b] = entry->bytes.s[at * width + (p->swap ? width - 1 - b : b)];
    }
  }
  return true;
}

int poly_torch_decode(const uint8_t *data, size_t len, PolyDecodedTensor **out, int *count) {
  *out = NULL;
  *count = 0;
  Parser p = {0};
  PolyDecodedTensor *tensors = NULL;
  size_t n = 0, packed = 0;
  bool ok = false;
  if (!data || !zip(&p, data, len)) goto done;
  Value *root = pickle(&p);
  if (!root || p.error || root->n % 2 || root->n / 2 > ZIP_LIMIT) goto done;
  n = root->n / 2;
  tensors = calloc(n ? n : 1, sizeof(*tensors));
  if (!tensors) {
    fail(&p, "allocation failed");
    goto done;
  }
  for (size_t i = 0; i < n; i++)
    if (!tensor(&p, root->items[2 * i], root->items[2 * i + 1], &tensors[i], &packed)) goto done;
  if (decoded_names_unique(tensors, (int)n)) {
    fail(&p, "duplicate tensor name");
    goto done;
  }
  *out = tensors;
  *count = (int)n;
  ok = true;
done:
  for (size_t i = 0; i < p.nv; i++) {
    free(p.values[i]->items);
    free(p.values[i]);
  }
  free(p.values);
  free(p.stack);
  free(p.memo);
  free(p.entries);
  if (!ok) {
    if (tensors)
      for (size_t i = 0; i < n; i++)
        free(tensors[i].name);
    free(tensors);
    poly_import_error_set(
        POLY_IMPORT_ERR_PARSE, "PyTorch checkpoint: %s (pickle byte %zu)",
        p.error ? p.error : "invalid state dictionary", p.pos
    );
  }
  return ok ? 0 : -1;
}
