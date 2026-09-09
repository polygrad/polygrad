/* utils.h -- generic C utilities, no polygrad dependencies.
 *
 * Tinygrad centralizes verbosity behind a single numeric DEBUG level and
 * gates renderer/runtime dumps off DEBUG >= N. Polygrad historically used
 * scattered POLY_DEBUG_* booleans; these helpers provide the same shared
 * numeric mechanism for the core C paths.
 */

#ifndef POLY_UTILS_H
#define POLY_UTILS_H

#include <stdint.h>
#include <stdbool.h>

/* Pointer hashing (splitmix64 finalizer) */
bool poly_ptr_eq(const void *a, const void *b);
uint32_t poly_ptr_hash(const void *p);

/* Tinygrad-style env helpers */
int poly_getenv_int(const char *key, int default_value);
bool poly_getenv_flag(const char *key);
bool poly_getenv_flag_default(const char *key, bool default_value);

/* Shared debug level. Fallback order:
 *   1. POLY_DEBUG
 *   2. DEBUG
 *   3. default 0
 */
int poly_debug_level(void);
bool poly_debug_at_least(int level);
double poly_now_ms(void);

/* C equivalent of search._try_compile's scoped alarm. Zero disables it;
 * native driver calls are checked on return, never signal-longjmp'd. */
extern _Thread_local double poly_compile_deadline_ms;
static inline bool poly_compile_timed_out(void) {
  return poly_compile_deadline_ms > 0 && poly_now_ms() >= poly_compile_deadline_ms;
}

/* Tinygrad-like debug thresholds:
 *   DEBUG >= 4 : rendered kernels
 *   DEBUG >= 5 : rewritten sink / graph
 *   DEBUG >= 6 : linear UOps
 *
 * POLY_DUMP_KERNELS remains a compatibility override and enables all three.
 */
bool poly_dump_kernels_enabled(void); /* POLY_DUMP_KERNELS or DEBUG >= 4 */
bool poly_dump_graph_enabled(void); /* POLY_DUMP_KERNELS or DEBUG >= 5 */
bool poly_dump_linear_enabled(void); /* POLY_DUMP_KERNELS or DEBUG >= 6 */

#endif /* POLY_UTILS_H */
