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
#ifndef _WIN32
#include <unistd.h>
#endif
#if defined(__linux__) && !defined(__EMSCRIPTEN__)
#include <errno.h>
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
  if (!v || !v[0]) return default_value;

  char *end = NULL;
  long parsed = strtol(v, &end, 10);
  if (end == v) return default_value;
  while (*end && isspace((unsigned char)*end))
    end++;
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
