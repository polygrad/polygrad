/* utils.c -- generic C utilities, no polygrad dependencies */

#include "utils.h"

#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#ifndef __EMSCRIPTEN__
#include <time.h>
#endif

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
// clang-format off
EM_JS(int, poly_browser_debug_level, (), {
  if (typeof globalThis === 'undefined') return 0;
  const v = globalThis.__polygradDebugLevel;
  if (v === undefined || v === null) return 0;
  const n = Number(v);
  return Number.isFinite(n) ? (n | 0) : 0;
});
// clang-format on
#endif

void poly_free(void *ptr) {
  free(ptr);
}

bool poly_ptr_eq(const void *a, const void *b) { return a == b; }

uint32_t poly_ptr_hash(const void *p) {
  /* UOp/map caches are keyed by pointer identity. Keep the mixer width
   * explicitly 64-bit: on wasm32 `uintptr_t` is only 32-bit, and shifting it
   * by 33 is undefined C. That broke the pointer-hash distribution exactly in
   * the rewrite passes that rely on pointer-keyed PolyMaps. */
  uint64_t x = (uint64_t)(uintptr_t)p;
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  return (uint32_t)x;
}

int poly_getenv_int(const char *key, int default_value) {
  if (!key || !key[0]) return default_value;
  const char *v = getenv(key);
  if (!v || !v[0]) return default_value;

  char *end = NULL;
  long parsed = strtol(v, &end, 10);
  if (end == v) return default_value;
  while (*end && isspace((unsigned char)*end)) end++;
  if (*end != '\0') return default_value;
  return (int)parsed;
}

bool poly_getenv_flag(const char *key) {
  if (!key || !key[0]) return false;
  const char *v = getenv(key);
  if (!v || !v[0]) return false;
  return strcmp(v, "0") != 0 && strcmp(v, "false") != 0 && strcmp(v, "False") != 0 &&
         strcmp(v, "no") != 0 && strcmp(v, "NO") != 0;
}

bool poly_getenv_flag_default(const char *key, bool default_value) {
  if (!key || !key[0]) return default_value;
  const char *v = getenv(key);
  if (!v || !v[0]) return default_value;
  return poly_getenv_flag(key);
}

int poly_debug_level(void) {
  const char *poly_debug = getenv("POLY_DEBUG");
  if (poly_debug && poly_debug[0]) return poly_getenv_int("POLY_DEBUG", 0);
  const char *debug = getenv("DEBUG");
  if (debug && debug[0]) return poly_getenv_int("DEBUG", 0);
#ifdef __EMSCRIPTEN__
  return poly_browser_debug_level();
#else
  return 0;
#endif
}

bool poly_debug_at_least(int level) { return poly_debug_level() >= level; }

double poly_now_ms(void) {
#ifdef __EMSCRIPTEN__
  return emscripten_get_now();
#else
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
#endif
}

bool poly_dump_kernels_enabled(void) {
  return poly_getenv_flag("POLY_DUMP_KERNELS") || poly_debug_at_least(4);
}

bool poly_dump_graph_enabled(void) {
  return poly_getenv_flag("POLY_DUMP_KERNELS") || poly_debug_at_least(5);
}

bool poly_dump_linear_enabled(void) {
  return poly_getenv_flag("POLY_DUMP_KERNELS") || poly_debug_at_least(6);
}
