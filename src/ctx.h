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
  PolyArena *scratch;
  PolyMap *cse;
  PolyMap *schedule_cache;
  PolyMap *to_program_cache;
  PolyMap *runtime_cache;
  size_t runtime_artifact_entries;
  size_t runtime_artifact_live_bytes;
  size_t launch_count;
  size_t schedule_cache_hits;
  size_t schedule_cache_misses;
  size_t runtime_cache_hits;
  size_t runtime_cache_misses;
  size_t buffer_read_count;
  size_t buffer_read_bytes;
  size_t buffer_write_count;
  size_t buffer_write_bytes;
  size_t buffer_copy_count;
  size_t buffer_copy_bytes;
  uint64_t global_ops;
  uint64_t global_mem;
  double time_sum_s;
  uint64_t kernel_count;
  uint64_t mem_used;
  uint64_t mem_used_per_device[POLY_DEVICE_DISK + 1];
  PolyMap *mem_used_by_device;
  int stats_suppression_depth;
  PolyMap *shape_cache;
  PolyMap *buffers;
  PolyTensor **tensors;
  int n_tensors;
  int tensors_cap;
  uint64_t next_tensor_order;
  PolyJit *active_jit_capture;
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
  PolyFrontendBufferReleaseFn frontend_buffer_release;
};

typedef PolyArenaMark PolyScratchMark;

PolyScratchMark poly_ctx_scratch_mark(PolyCtx *ctx);
void poly_ctx_scratch_rewind(PolyCtx *ctx, PolyScratchMark mark);
void *poly_ctx_scratch_alloc(PolyCtx *ctx, size_t size, size_t align);

PolyUOp **poly_toposort_scratch(PolyCtx *ctx, PolyUOp *root, int *n_out);
PolyUOp **poly_toposort_ex_user_scratch(
    PolyCtx *ctx,
    PolyUOp *root,
    int *n_out,
    bool (*gate)(PolyUOp *, void *),
    void *user_data,
    bool enter_calls
);

int64_t poly_ctx_next_unique_id(PolyCtx *ctx);
void poly_ctx_record_memory_alloc(PolyCtx *ctx, PolyDevice device, size_t nbytes);
void poly_ctx_record_memory_free(PolyCtx *ctx, PolyDevice device, size_t nbytes);
void poly_ctx_record_memory_alloc_exact(
    PolyCtx *ctx,
    PolyUOp *device_uop,
    PolyDevice backend,
    size_t nbytes
);
void poly_ctx_record_memory_free_exact(
    PolyCtx *ctx,
    PolyUOp *device_uop,
    PolyDevice backend,
    size_t nbytes
);
uint64_t poly_ctx_mem_used_for_device_uop(PolyCtx *ctx, PolyUOp *device_uop);
void poly_ctx_reserve_unique_id(PolyCtx *ctx, int64_t id);
void poly_ctx_reserve_buf_tag(PolyCtx *ctx, int32_t tag);

#endif /* POLY_CTX_H */
