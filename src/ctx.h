/* ctx.h -- PolyCtx struct definition (internal)
 *
 * Only files that need to access ctx fields directly include this.
 * Public API (poly_ctx_new, poly_ctx_destroy) is declared in polygrad.h.
 */

#ifndef POLY_CTX_H
#define POLY_CTX_H

#include "polygrad.h"
#include "arena.h"

struct PolyCtx {
  PolyArena *arena;
  PolyMap *cse;
  PolyMap *kernel_cache;
  PolyMap *shape_cache;
  PolyMap *buffers;
  /* Named buffer registry */
  PolyRegEntry **entries;
  int n_entries;
  int entries_cap;
  PolyMap *name_map;
  /* Named entrypoints */
  struct {
    const char *name;
    PolyUOp *sink;
  } *ep;
  int n_ep;
  int ep_cap;
  int32_t next_buf_tag;
  PolyDevice preferred_device;
};

#endif /* POLY_CTX_H */
