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
  PolyMap *schedule_cache;
  PolyMap *program_cache;
  PolyMap *shape_cache;
  PolyMap *buffers;
  PolyMap *tensors_by_uop;
  PolyTensor **tensors;
  int n_tensors;
  int tensors_cap;
  uint64_t next_tensor_order;
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
  int64_t next_unique_id;
  PolyDevice preferred_device;
};

int64_t poly_ctx_next_unique_id(PolyCtx *ctx);

#endif /* POLY_CTX_H */
