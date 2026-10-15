/* utils.c -- generic C utilities, no polygrad dependencies */

#define _GNU_SOURCE
#include "utils.h"

_Thread_local double poly_compile_deadline_ms;

#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include <limits.h>
#include <errno.h>

static uint32_t sha_rotr(uint32_t x, unsigned n) {
  return (x >> n) | (x << (32 - n));
}

void poly_sha256(const void *data, size_t size, uint8_t out[32]) {
  static const uint32_t k[64] = {
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
      0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
      0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
      0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
      0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
      0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
      0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
      0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
      0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
      0xc67178f2};
  uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                   0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  const uint8_t *input = data;
  size_t blocks = size / 64, tail = size % 64;
  /* At most two padding blocks, without adding padding to size (which could overflow). */
  for (size_t block = 0; block < blocks + (tail < 56 ? 1 : 2); block++) {
    uint8_t pad[64] = {0};
    const uint8_t *p;
    if (block < blocks)
      p = input + block * 64;
    else {
      if (block == blocks) {
        if (tail) memcpy(pad, input + blocks * 64, tail);
        pad[tail] = 0x80;
      }
      if (block == blocks + (tail < 56 ? 0 : 1))
        for (int i = 0; i < 8; i++)
          pad[63 - i] = (uint8_t)(((uint64_t)size * 8) >> (i * 8));
      p = pad;
    }
    uint32_t w[64];
    for (int i = 0; i < 16; i++)
      w[i] = (uint32_t)p[i * 4] << 24 | (uint32_t)p[i * 4 + 1] << 16 | (uint32_t)p[i * 4 + 2] << 8 |
             p[i * 4 + 3];
    for (int i = 16; i < 64; i++) {
      uint32_t x = w[i - 15], y = w[i - 2];
      w[i] = w[i - 16] + (sha_rotr(x, 7) ^ sha_rotr(x, 18) ^ (x >> 3)) + w[i - 7] +
             (sha_rotr(y, 17) ^ sha_rotr(y, 19) ^ (y >> 10));
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], z = h[7];
    for (int i = 0; i < 64; i++) {
      uint32_t t1 = z + (sha_rotr(e, 6) ^ sha_rotr(e, 11) ^ sha_rotr(e, 25)) +
                    ((e & f) ^ (~e & g)) + k[i] + w[i];
      uint32_t t2 =
          (sha_rotr(a, 2) ^ sha_rotr(a, 13) ^ sha_rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
      z = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }
    h[0] += a;
    h[1] += b;
    h[2] += c;
    h[3] += d;
    h[4] += e;
    h[5] += f;
    h[6] += g;
    h[7] += z;
  }
  for (int i = 0; i < 32; i++)
    out[i] = (uint8_t)(h[i / 4] >> (24 - 8 * (i % 4)));
}
#ifndef _WIN32
#include <unistd.h>
#endif
#if defined(__linux__) && !defined(__EMSCRIPTEN__)
#include <sched.h>
#endif

int poly_cpu_thread_count(void) {
  int override = poly_getenv_int("NUM_CPU_THREADS", 0);
  if (override > 0) return override;
#if defined(__linux__) && !defined(__EMSCRIPTEN__)
  /* tinygrad v0.14 helpers._get_cpu_count, Python <=3.12 path: a finite
   * cgroup-v2 quota takes precedence over affinity. Do not count host CPUs
   * for a process restricted with taskset/cpuset. */
  FILE *fp = fopen("/sys/fs/cgroup/cpu.max", "r");
  if (fp) {
    long long quota, period;
    int fields = fscanf(fp, "%lld %lld", &quota, &period);
    fclose(fp);
    if (fields == 2 && quota > 0 && period > 0) {
      long long count = quota / period;
      return count < 1 ? 1 : count > INT_MAX ? INT_MAX : (int)count;
    }
  }
  /* Kernel CPU masks can exceed CPU_SETSIZE even for a small process cpuset. */
  for (int bits = CPU_SETSIZE; bits <= INT_MAX / 2; bits *= 2) {
    cpu_set_t *set = CPU_ALLOC(bits);
    if (!set) break;
    size_t size = CPU_ALLOC_SIZE(bits);
    int status = sched_getaffinity(0, size, set);
    int error = errno;
    int count = status == 0 ? CPU_COUNT_S(size, set) : 0;
    CPU_FREE(set);
    if (status == 0) return count > 0 ? count : 1;
    if (error != EINVAL) break;
  }
#endif
#ifdef _SC_NPROCESSORS_ONLN
  long count = sysconf(_SC_NPROCESSORS_ONLN);
  if (count > 0 && count < INT_MAX) return (int)count;
#endif
  return 1;
}

char *poly_strfmt(const char *format, ...) {
  va_list args, copy;
  va_start(args, format);
  va_copy(copy, args);
  int size = vsnprintf(NULL, 0, format, args);
  va_end(args);
  char *out = size < 0 ? NULL : malloc((size_t)size + 1);
  if (out && vsnprintf(out, (size_t)size + 1, format, copy) != size) {
    free(out);
    out = NULL;
  }
  va_end(copy);
  return out;
}
#ifndef __EMSCRIPTEN__
#include <time.h>
#endif

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
// clang-format off
EM_JS(int, poly_browser_debug_level, (), {
  if (typeof globalThis === 'undefined') return 0;
  const env = typeof process !== 'undefined' && process.versions && process.versions.node ? process.env : {};
  const v = globalThis.__polygradDebugLevel ?? (env.POLY_DEBUG || env.DEBUG);
  if (v === undefined || v === null) return 0;
  const n = Number(v);
  return Number.isFinite(n) ? (n | 0) : 0;
});
// clang-format on
#endif

void poly_free(void *ptr) {
  free(ptr);
}

bool poly_ptr_eq(const void *a, const void *b) {
  return a == b;
}

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
  if (!v) return default_value;

  char *end = NULL;
  errno = 0;
  long parsed = strtol(v, &end, 10);
  bool valid = end != v && errno != ERANGE && parsed >= INT_MIN && parsed <= INT_MAX;
  while (*end && isspace((unsigned char)*end))
    end++;
  if (!valid || *end != '\0') {
    /* This value-only utility cannot propagate an execution error. Keep the
     * existing fallback, but never silently ignore malformed configuration or
     * narrow an overflowing long. Do not print environment values. */
    fprintf(
        stderr, "polygrad: %s requires a decimal integer in [%d, %d]; using default %d\n", key,
        INT_MIN, INT_MAX, default_value
    );
    return default_value;
  }
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

bool poly_debug_at_least(int level) {
  return poly_debug_level() >= level;
}

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
