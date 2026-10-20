/* ctx.c -- PolyCtx lifecycle: create, destroy, accessors */

#include "ctx.h"
#include "device.h"
#include "engine/schedule.h"
#include "uop/ops.h"
#include "utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>

/* Tinygrad UOps disappear through Python refcounts, so stable JIT replay has
 * no global tracing pass. C batches its nonmoving mark/sweep until unreclaimed
 * UOp storage has grown by this bounded budget. */
#include <time.h>

/* Defined in ops.c */
void poly_init_group_ops(void);
static void free_buffer_record(const void *key, void *value, void *userdata);

/* Optional cleanup hooks (defined in other files, linked weakly). */
void poly_tensor_ctx_cleanup(PolyCtx *ctx) __attribute__((weak));
void poly_engine_ctx_cleanup(PolyCtx *ctx) __attribute__((weak));
void poly_schedule_ctx_cleanup(PolyCtx *ctx) __attribute__((weak));

int poly_shape_cache_evict_unmarked(PolyCtx *ctx, PolyMap *live);
void poly_shape_cache_destroy_all(PolyCtx *ctx);
int poly_shape_cache_live_value_roots(
    PolyCtx *ctx,
    PolyMap *live,
    PolyUOp ***roots_out,
    int *count_out
);

static void free_buffer_entry(const void *key, void *value, void *userdata) {
  (void)key;
  PolyCtx *ctx = (PolyCtx *)userdata;
  /* Free current + src chain and its C handle metadata. */
  poly_buffer_free_chain(ctx, (PolyBuffer *)value);
}

static void free_memory_entry(const void *key, void *value, void *userdata) {
  (void)key;
  (void)userdata;
  free(value);
}

static bool logical_policy_from_env(const char *value, PolyLogicalPolicy *policy) {
  if (!value || !policy) return false;
  if (strcmp(value, "0") == 0)
    *policy = POLY_LOGICAL_NEVER;
  else if (strcmp(value, "1") == 0)
    *policy = POLY_LOGICAL_ALWAYS;
  else if (strcmp(value, "2") == 0)
    *policy = POLY_LOGICAL_UNTIL_REALIZE;
  else
    return false;
  return true;
}

PolyCtx *poly_ctx_new(void) {
  poly_init_group_ops();
  if (poly_get_default_float() < 0 || poly_get_default_int() < 0) return NULL;
  const char *logical_env = getenv("POLY_LOGICAL");
  /* Polygrad logical-lifetime divergence: retain portable producers until
   * successful materialization, then keep only the exact resource spine. */
  PolyLogicalPolicy logical_policy = POLY_LOGICAL_UNTIL_REALIZE;
  if (logical_env && !logical_policy_from_env(logical_env, &logical_policy)) return NULL;
  PolyCtx *ctx = malloc(sizeof(PolyCtx));
  if (!ctx) return NULL;
  ctx->arena = poly_arena_new(0);
  ctx->kernel_policy = poly_getenv_int("POLY_KERNELS", 0) == 1;
  ctx->kernel_policy_locked = false;
  ctx->scratch = poly_arena_new(0);
  ctx->cse = poly_map_new(256);
  ctx->uop_storage = poly_map_new(256);
  ctx->uop_storage_bytes = 0;
  ctx->uop_storage_high_water = 0;
  ctx->schedule_cache = poly_map_new(16);
  ctx->to_program_cache = poly_map_new(16);
  ctx->runtime_cache = poly_map_new(16);
  ctx->local_size_cache = poly_map_new(16);
  ctx->graph_cache = poly_map_new(8);
  ctx->runtime_artifact_entries = 0;
  ctx->runtime_artifact_live_bytes = 0;
  ctx->launch_count = 0;
  ctx->runtime_cache_hits = 0;
  ctx->runtime_cache_misses = 0;
  ctx->buffer_read_count = 0;
  ctx->buffer_read_bytes = 0;
  ctx->buffer_write_count = 0;
  ctx->buffer_write_bytes = 0;
  ctx->buffer_copy_count = 0;
  ctx->buffer_copy_bytes = 0;
  ctx->global_ops = 0;
  ctx->global_mem = 0;
  ctx->time_sum_s = 0.0;
  ctx->kernel_count = 0;
  ctx->mem_used = 0;
  memset(ctx->mem_used_per_device, 0, sizeof(ctx->mem_used_per_device));
  ctx->mem_used_by_device = poly_map_new(8);
  ctx->stats_suppression_depth = 0;
  ctx->shape_cache = poly_map_new(64);
  ctx->buffers = poly_map_new(64);
  ctx->retained_uops = poly_map_new(16);
  ctx->residency_refs = poly_map_new(16);
  ctx->residency_candidates = poly_map_new(16);
  ctx->buffer_records = poly_map_new(64);
  ctx->buffer_owners = poly_map_new(64);
  ctx->residency_refs_valid = true;
  ctx->check_residency = poly_getenv_flag("POLY_CHECK_RESIDENCY");
#ifdef POLY_TESTING
  ctx->residency_walk_count = 0;
  ctx->residency_sweep_count = 0;
  ctx->residency_fail_stack_growth = false;
#endif
  ctx->collection_requested = false;
  ctx->collection_dirty = false;
  ctx->ir_collection_dirty = false;
  ctx->ir_collection_baseline_bytes = 0;
  ctx->collecting = false;
  ctx->execution_depth = 0;
  ctx->execution_write_epoch = 0;
  ctx->rng_states = poly_map_new(8);
  ctx->rng_seed = (uint64_t)time(NULL);
  ctx->rng_device_count = 0;
  ctx->tensors = NULL;
  ctx->n_tensors = 0;
  ctx->tensors_cap = 0;
  ctx->next_tensor_order = 1;
  ctx->active_jit_capture = NULL;
  ctx->tensor_capture = NULL;
  if (!ctx->arena || !ctx->scratch || !ctx->cse || !ctx->uop_storage || !ctx->schedule_cache ||
      !ctx->to_program_cache || !ctx->runtime_cache || !ctx->local_size_cache ||
      !ctx->graph_cache || !ctx->mem_used_by_device || !ctx->shape_cache || !ctx->buffers ||
      !ctx->retained_uops || !ctx->residency_refs || !ctx->rng_states) {
    if (ctx->arena) poly_arena_destroy(ctx->arena);
    if (ctx->scratch) poly_arena_destroy(ctx->scratch);
    if (ctx->cse) poly_map_destroy(ctx->cse);
    if (ctx->uop_storage) poly_map_destroy(ctx->uop_storage);
    if (ctx->schedule_cache) poly_map_destroy(ctx->schedule_cache);
    if (ctx->to_program_cache) poly_map_destroy(ctx->to_program_cache);
    if (ctx->runtime_cache) poly_map_destroy(ctx->runtime_cache);
    if (ctx->local_size_cache) poly_map_destroy(ctx->local_size_cache);
    if (ctx->graph_cache) poly_map_destroy(ctx->graph_cache);
    if (ctx->mem_used_by_device) poly_map_destroy(ctx->mem_used_by_device);
    if (ctx->shape_cache) poly_map_destroy(ctx->shape_cache);
    if (ctx->buffers) poly_map_destroy(ctx->buffers);
    if (ctx->retained_uops) poly_map_destroy(ctx->retained_uops);
    if (ctx->residency_refs) poly_map_destroy(ctx->residency_refs);
    poly_map_destroy(ctx->residency_candidates);
    poly_map_destroy(ctx->buffer_records);
    poly_map_destroy(ctx->buffer_owners);
    if (ctx->rng_states) poly_map_destroy(ctx->rng_states);
    free(ctx);
    return NULL;
  }
  ctx->next_buf_tag = 1;
  ctx->next_unique_id = 0;
  /* Snapshot a supported default without rejecting context creation: an
   * explicit device can override an unsupported host environment afterward. */
  ctx->preferred_device = poly_device_default();
  /* Polygrad logical-lifetime divergence: environment initializes one
   * context; later scoped changes are explicit and affect future Tensors. */
  ctx->logical_policy = logical_policy;
  ctx->frontend_buffer_release = NULL;
  return ctx;
}

void poly_ctx_destroy(PolyCtx *ctx) {
  if (!ctx) return;
  if (poly_tensor_ctx_cleanup) poly_tensor_ctx_cleanup(ctx);
  if (poly_engine_ctx_cleanup) poly_engine_ctx_cleanup(ctx);
  if (poly_schedule_ctx_cleanup) poly_schedule_ctx_cleanup(ctx);
  poly_map_destroy(ctx->schedule_cache);
  poly_map_destroy(ctx->to_program_cache);
  poly_map_destroy(ctx->runtime_cache);
  poly_map_destroy(ctx->local_size_cache);
  poly_map_destroy(ctx->graph_cache);
  poly_shape_cache_destroy_all(ctx);
  poly_map_destroy(ctx->shape_cache);
  /* Free owned buffer ptrs before destroying the map. */
  poly_map_foreach(ctx->buffers, free_buffer_entry, ctx);
  poly_map_destroy(ctx->buffers);
  poly_map_destroy(ctx->retained_uops);
  poly_map_destroy(ctx->residency_refs);
  poly_map_destroy(ctx->residency_candidates);
  poly_map_foreach(ctx->buffer_records, free_buffer_record, NULL);
  poly_map_destroy(ctx->buffer_records);
  poly_map_destroy(ctx->buffer_owners);
  poly_map_destroy(ctx->rng_states);
  poly_map_foreach(ctx->mem_used_by_device, free_memory_entry, NULL);
  poly_map_destroy(ctx->mem_used_by_device);
  free(ctx->tensors);
  poly_map_destroy(ctx->cse);
  poly_uop_storage_destroy_all(ctx);
  poly_map_destroy(ctx->uop_storage);
  poly_arena_destroy(ctx->scratch);
  poly_arena_destroy(ctx->arena);
  free(ctx);
}

static bool residency_ref_update(PolyCtx *ctx, PolyUOp *root, bool acquire);

void poly_ctx_root_acquired(PolyCtx *ctx, PolyUOp *root) {
  if (!ctx || !root) return;
  if (!ctx->residency_refs_valid || !residency_ref_update(ctx, root, true)) {
    ctx->residency_refs_valid = false;
    ctx->collection_requested = true;
  }
}

void poly_ctx_root_released(PolyCtx *ctx, PolyUOp *root) {
  if (!ctx || !root) return;
  if (!ctx->residency_refs_valid || !residency_ref_update(ctx, root, false)) {
    ctx->residency_refs_valid = false;
    ctx->collection_requested = true;
  }
}

int poly_uop_retain(PolyCtx *ctx, PolyUOp *uop) {
  if (!ctx || !uop || !poly_ctx_owns_ptr(ctx, uop)) return -1;
  uintptr_t count =
      (uintptr_t)poly_map_get(ctx->retained_uops, poly_ptr_hash(uop), uop, poly_ptr_eq);
  if (count == UINTPTR_MAX) return -1;
  poly_map_set(ctx->retained_uops, poly_ptr_hash(uop), uop, (void *)(count + 1), poly_ptr_eq);
  if (count == 0) poly_ctx_root_acquired(ctx, uop);
  return 0;
}

void poly_uop_release(PolyCtx *ctx, PolyUOp *uop) {
  if (!ctx || !uop) return;
  uintptr_t count =
      (uintptr_t)poly_map_get(ctx->retained_uops, poly_ptr_hash(uop), uop, poly_ptr_eq);
  if (count == 0) return;
  if (count == 1) {
    poly_ctx_root_released(ctx, uop);
    poly_map_remove(ctx->retained_uops, poly_ptr_hash(uop), uop, poly_ptr_eq);
    /* Tinygrad 2026-08-22/a9069c177a9d UOp.__del__ removes the weak CSE row
     * only after the final strong reference dies (uop/ops.py:241-246). */
    ctx->collection_dirty = true;
    ctx->ir_collection_dirty = true;
  } else {
    poly_map_set(ctx->retained_uops, poly_ptr_hash(uop), uop, (void *)(count - 1), poly_ptr_eq);
  }
}

typedef struct BufferRecord {
  PolyUOp *key;
  PolyBuffer **owned;
  struct BufferRecord **dependencies;
  size_t n_owned, n_dependencies, borrowers;
} BufferRecord;

static void residency_candidate(PolyCtx *ctx, PolyUOp *key) {
  poly_map_set(ctx->residency_candidates, poly_ptr_hash(key), key, key, poly_ptr_eq);
  ctx->collection_requested = true;
}

static void free_buffer_record(const void *key, void *value, void *userdata) {
  (void)key;
  (void)userdata;
  BufferRecord *record = value;
  free(record->owned);
  free(record->dependencies);
  free(record);
}

/* This index shares PolyMap's fatal-OOM policy: publishing incomplete storage
 * ownership would allow a later collection to free an aliased allocation. */
static void *residency_resize(void *ptr, size_t size) {
  void *out = realloc(ptr, size);
  if (!out) {
    fprintf(stderr, "polygrad: storage ownership allocation failed\n");
    abort();
  }
  return out;
}

static void buffer_record_clear(PolyCtx *ctx, BufferRecord *record) {
  for (size_t i = 0; i < record->n_owned; i++)
    poly_map_remove(
        ctx->buffer_owners, poly_ptr_hash(record->owned[i]), record->owned[i], poly_ptr_eq
    );
  for (size_t i = 0; i < record->n_dependencies; i++) {
    BufferRecord *owner = record->dependencies[i];
    if (!--owner->borrowers) residency_candidate(ctx, owner->key);
  }
  record->n_owned = record->n_dependencies = 0;
}

static void buffer_record_own(PolyCtx *ctx, BufferRecord *record, PolyBuffer *buffer) {
  if (!buffer) return;
  record->owned = residency_resize(record->owned, (record->n_owned + 1) * sizeof(*record->owned));
  record->owned[record->n_owned++] = buffer;
  poly_map_set(ctx->buffer_owners, poly_ptr_hash(buffer), buffer, record, poly_ptr_eq);
  buffer_record_own(ctx, record, buffer->src);
  if (buffer->owns_bufs)
    for (int i = 0; i < buffer->n_bufs; i++)
      buffer_record_own(ctx, record, buffer->bufs[i]);
}

static void buffer_record_borrow(PolyCtx *ctx, BufferRecord *record, PolyBuffer *buffer) {
  if (!buffer) return;
  BufferRecord *owner =
      poly_map_get(ctx->buffer_owners, poly_ptr_hash(buffer), buffer, poly_ptr_eq);
  if (!owner || owner == record) return;
  for (size_t i = 0; i < record->n_dependencies; i++)
    if (record->dependencies[i] == owner) return;
  record->dependencies = residency_resize(
      record->dependencies, (record->n_dependencies + 1) * sizeof(*record->dependencies)
  );
  record->dependencies[record->n_dependencies++] = owner;
  owner->borrowers++;
}

void poly_ctx_buffer_changed(PolyCtx *ctx, PolyUOp *key) {
  BufferRecord *record = poly_map_get(ctx->buffer_records, poly_ptr_hash(key), key, poly_ptr_eq);
  if (!record) {
    record = residency_resize(NULL, sizeof(*record));
    *record = (BufferRecord){.key = key};
    poly_map_set(ctx->buffer_records, poly_ptr_hash(key), key, record, poly_ptr_eq);
  } else
    buffer_record_clear(ctx, record);
  buffer_record_own(ctx, record, poly_buffer_get(ctx, key));
  for (size_t i = 0; i < record->n_owned; i++) {
    PolyBuffer *buffer = record->owned[i];
    buffer_record_borrow(ctx, record, buffer->base);
    if (!buffer->owns_bufs)
      for (int j = 0; j < buffer->n_bufs; j++)
        buffer_record_borrow(ctx, record, buffer->bufs[j]);
  }
  residency_candidate(ctx, key);
}

void poly_ctx_buffer_removed(PolyCtx *ctx, PolyUOp *key) {
  BufferRecord *record = poly_map_get(ctx->buffer_records, poly_ptr_hash(key), key, poly_ptr_eq);
  if (!record) return;
  buffer_record_clear(ctx, record);
  poly_map_remove(ctx->buffer_records, poly_ptr_hash(key), key, poly_ptr_eq);
  free_buffer_record(NULL, record, NULL);
}

void poly_ctx_buffer_handle_changed(PolyCtx *ctx, PolyBuffer *buffer) {
  BufferRecord *owner =
      poly_map_get(ctx->buffer_owners, poly_ptr_hash(buffer), buffer, poly_ptr_eq);
  if (owner) poly_ctx_buffer_changed(ctx, owner->key);
}

static void rebuild_buffer_record(const void *key, void *value, void *userdata) {
  (void)value;
  poly_ctx_buffer_changed(userdata, (PolyUOp *)key);
}

typedef struct {
  PolyCtx *expected;
  bool equal;
} BufferRecordCheck;

static void compare_buffer_record(const void *key, void *value, void *userdata) {
  BufferRecordCheck *check = userdata;
  BufferRecord *record = value;
  BufferRecord *expected =
      poly_map_get(check->expected->buffer_records, poly_ptr_hash(key), key, poly_ptr_eq);
  if (!expected || record->borrowers != expected->borrowers ||
      record->n_owned != expected->n_owned || record->n_dependencies != expected->n_dependencies) {
    check->equal = false;
    return;
  }
  for (size_t i = 0; i < record->n_owned; i++)
    if (record->owned[i] != expected->owned[i]) check->equal = false;
  for (size_t i = 0; i < record->n_dependencies; i++)
    if (record->dependencies[i]->key != expected->dependencies[i]->key) check->equal = false;
}

static bool buffer_records_check(PolyCtx *ctx) {
  /* Independent raw-metadata reconstruction; do not repair incremental drift.
   * Two passes establish every metadata owner before resolving alias edges. */
  PolyCtx expected = *ctx;
  expected.buffer_records = poly_map_new(64);
  expected.buffer_owners = poly_map_new(64);
  expected.residency_candidates = poly_map_new(16);
  poly_map_foreach(ctx->buffers, rebuild_buffer_record, &expected);
  poly_map_foreach(ctx->buffers, rebuild_buffer_record, &expected);
  BufferRecordCheck check = {.expected = &expected, .equal = true};
  poly_map_foreach(ctx->buffer_records, compare_buffer_record, &check);
  check.equal = check.equal &&
                poly_map_len(ctx->buffer_records) == poly_map_len(expected.buffer_records) &&
                poly_map_len(ctx->buffer_owners) == poly_map_len(expected.buffer_owners);
  poly_map_foreach(expected.buffer_records, free_buffer_record, NULL);
  poly_map_destroy(expected.buffer_records);
  poly_map_destroy(expected.buffer_owners);
  poly_map_destroy(expected.residency_candidates);
  if (!check.equal) fprintf(stderr, "polygrad: storage ownership drift\n");
  return check.equal;
}

static bool residency_ref_update(PolyCtx *ctx, PolyUOp *root, bool acquire) {
  /* Immutable source edges contribute once per live parent, not once per path
   * or Tensor wrapper. Follow edges only across zero, as Python UOp ownership
   * does. Storage destruction stays deferred to the existing safe points. */
  PolyUOp *local[64], **stack = local;
  size_t count = 1, capacity = 64;
  local[0] = root;
  bool ok = true;
  while (count && ok) {
    PolyUOp *uop = stack[--count];
    if (!uop) continue;
    uintptr_t refs =
        (uintptr_t)poly_map_get(ctx->residency_refs, poly_ptr_hash(uop), uop, poly_ptr_eq);
    if ((acquire && refs == UINTPTR_MAX) || (!acquire && !refs)) {
      ok = false;
      break;
    }
    uintptr_t next = acquire ? refs + 1 : refs - 1;
    if (next)
      poly_map_set(ctx->residency_refs, poly_ptr_hash(uop), uop, (void *)next, poly_ptr_eq);
    else
      poly_map_remove(ctx->residency_refs, poly_ptr_hash(uop), uop, poly_ptr_eq);
    if (acquire ? refs != 0 : next != 0) continue;
#ifdef POLY_TESTING
    ctx->residency_walk_count++;
#endif
    if (!acquire && (poly_map_get(ctx->buffers, poly_ptr_hash(uop), uop, poly_ptr_eq) ||
                     poly_map_get(ctx->graph_cache, poly_ptr_hash(uop), uop, poly_ptr_eq))) {
      residency_candidate(ctx, uop);
      ctx->collection_dirty = true;
    }
    size_t needed = count + (size_t)uop->n_src;
    if (needed > capacity) {
#ifdef POLY_TESTING
      if (ctx->residency_fail_stack_growth) {
        ctx->residency_fail_stack_growth = false;
        ok = false;
        break;
      }
#endif
      size_t size = needed > capacity * 2 ? needed : capacity * 2;
      PolyUOp **grown = size <= SIZE_MAX / sizeof(*stack) ? malloc(size * sizeof(*stack)) : NULL;
      if (!grown) {
        ok = false;
        break;
      }
      memcpy(grown, stack, count * sizeof(*stack));
      if (stack != local) free(stack);
      stack = grown;
      capacity = size;
    }
    /* PROGRAM bodies are not residency owners. CUSTOM_FUNCTION bodies contain
     * concrete graph-private scratch buffers and must remain reachable. */
    int first =
        uop->op == POLY_OP_CALL && uop->n_src && uop->src[0]->op != POLY_OP_CUSTOM_FUNCTION ? 1 : 0;
    for (int i = first; i < uop->n_src; i++)
      stack[count++] = uop->src[i];
  }
  if (stack != local) free(stack);
  return ok;
}

static void residency_retain_root(const void *key, void *value, void *userdata) {
  (void)value;
  PolyCtx *ctx = userdata;
  if (ctx->residency_refs_valid)
    ctx->residency_refs_valid = residency_ref_update(ctx, (PolyUOp *)key, true);
}

static bool residency_rebuild(PolyCtx *ctx) {
  /* A stack-allocation failure cannot make partially updated counts evidence
   * for freeing storage. Reconstruct from declared owners before any sweep. */
  poly_map_clear(ctx->residency_refs);
  ctx->residency_refs_valid = true;
  poly_map_foreach(ctx->retained_uops, residency_retain_root, ctx);
  for (int i = 0; ctx->residency_refs_valid && i < ctx->n_tensors; i++) {
    PolyTensor *tensor = ctx->tensors[i];
    if (tensor && tensor->owner_refs && tensor->uop_physical)
      ctx->residency_refs_valid = residency_ref_update(ctx, tensor->uop_physical, true);
  }
  return ctx->residency_refs_valid;
}

typedef struct {
  PolyMap *expected;
  bool equal;
} ResidencyCheck;

static void residency_compare(const void *key, void *value, void *userdata) {
  ResidencyCheck *check = userdata;
  void *expected = poly_map_get(check->expected, poly_ptr_hash(key), key, poly_ptr_eq);
  if (value != expected && check->equal) {
    fprintf(
        stderr, "polygrad: residency count drift at %p: incremental=%zu rebuilt=%zu\n", key,
        (size_t)(uintptr_t)value, (size_t)(uintptr_t)expected
    );
    check->equal = false;
  }
}

static bool residency_check(PolyCtx *ctx) {
  /* Rebuild from the owner registry, not the incremental map. Do not repair
   * a mismatch: that would hide a missing acquire/release on a root change. */
  PolyMap *incremental = ctx->residency_refs;
  ctx->residency_refs = poly_map_new(16);
#ifdef POLY_TESTING
  size_t walked = ctx->residency_walk_count;
#endif
  ResidencyCheck check = {.expected = ctx->residency_refs, .equal = residency_rebuild(ctx)};
  if (check.equal) {
    poly_map_foreach(incremental, residency_compare, &check);
    check.equal = check.equal && poly_map_len(incremental) == poly_map_len(check.expected);
  }
  poly_map_destroy(ctx->residency_refs);
  ctx->residency_refs = incremental;
  ctx->residency_refs_valid = true;
#ifdef POLY_TESTING
  ctx->residency_walk_count = walked;
#endif
  return check.equal;
}

typedef struct {
  PolyMap *live;
  PolyUOp **stack;
  int n_stack;
  int cap_stack;
  bool failed;
} IrMarker;

static bool ir_mark_root(IrMarker *marker, PolyUOp *root) {
  /* Python owns the complete source closure of each live Tinygrad UOp. This
   * non-moving C trace reproduces that relation without making CSE/cache rows
   * owners. CALL source zero is included because IR, unlike residency, owns
   * the executable graph too. */
  if (!marker || marker->failed) return false;
  if (!root) return true;
  if (marker->n_stack >= marker->cap_stack) {
    int capacity = marker->cap_stack ? marker->cap_stack * 2 : 256;
    PolyUOp **stack = realloc(marker->stack, (size_t)capacity * sizeof(*stack));
    if (!stack) return false;
    marker->stack = stack;
    marker->cap_stack = capacity;
  }
  marker->stack[marker->n_stack++] = root;
  while (marker->n_stack > 0) {
    PolyUOp *uop = marker->stack[--marker->n_stack];
    if (!uop || poly_map_get(marker->live, poly_ptr_hash(uop), uop, poly_ptr_eq)) continue;
    poly_map_set(marker->live, poly_ptr_hash(uop), uop, uop, poly_ptr_eq);
    int n_edges = uop->n_src + (uop->device_cache != NULL);
    if (n_edges > marker->cap_stack - marker->n_stack) {
      int capacity = marker->cap_stack;
      while (capacity - marker->n_stack < n_edges)
        capacity *= 2;
      PolyUOp **stack = realloc(marker->stack, (size_t)capacity * sizeof(*stack));
      if (!stack) return false;
      marker->stack = stack;
      marker->cap_stack = capacity;
    }
    for (int i = 0; i < uop->n_src; i++)
      marker->stack[marker->n_stack++] = uop->src[i];
    if (uop->device_cache) marker->stack[marker->n_stack++] = uop->device_cache;
    /* Tinygrad UOp args are Python owners. C copies ProgramInfo/KernelInfo
     * beside the UOp, so trace their symbolic UOp fields explicitly. */
    if (uop->arg.kind == POLY_ARG_PROGRAM_INFO && uop->arg.program_info) {
      const PolyProgramInfo *info = uop->arg.program_info;
      int n_meta = info->n_vars + 6;
      if (n_meta > marker->cap_stack - marker->n_stack) {
        int capacity = marker->cap_stack;
        while (capacity - marker->n_stack < n_meta)
          capacity *= 2;
        PolyUOp **stack = realloc(marker->stack, (size_t)capacity * sizeof(*stack));
        if (!stack) return false;
        marker->stack = stack;
        marker->cap_stack = capacity;
      }
      for (int i = 0; i < 3; i++) {
        marker->stack[marker->n_stack++] = info->global_exprs[i];
        marker->stack[marker->n_stack++] = info->local_exprs[i];
      }
      for (int i = 0; i < info->n_vars; i++)
        marker->stack[marker->n_stack++] = info->vars[i];
    } else if (uop->arg.kind == POLY_ARG_KERNEL_INFO && uop->arg.kernel_info &&
               uop->arg.kernel_info->estimates) {
      if (3 > marker->cap_stack - marker->n_stack) {
        int capacity = marker->cap_stack;
        while (capacity - marker->n_stack < 3)
          capacity *= 2;
        PolyUOp **stack = realloc(marker->stack, (size_t)capacity * sizeof(*stack));
        if (!stack) return false;
        marker->stack = stack;
        marker->cap_stack = capacity;
      }
      marker->stack[marker->n_stack++] = uop->arg.kernel_info->estimates->ops;
      marker->stack[marker->n_stack++] = uop->arg.kernel_info->estimates->lds;
      marker->stack[marker->n_stack++] = uop->arg.kernel_info->estimates->mem;
    }
  }
  return true;
}

static void mark_retained_ir(const void *key, void *value, void *userdata) {
  (void)value;
  IrMarker *marker = userdata;
  if (!marker || marker->failed) return;
  marker->failed = !ir_mark_root(marker, (PolyUOp *)key);
}

typedef struct {
  IrMarker *ir;
  PolyMap *visited;
  PolyBuffer **stack;
  int n_stack;
  int cap_stack;
  bool failed;
} BufferIrMarker;

static bool mark_buffer_ir_root(BufferIrMarker *marker, PolyBuffer *root) {
  if (!marker || marker->failed) return false;
  if (!root) return true;
  if (marker->n_stack >= marker->cap_stack) {
    int capacity = marker->cap_stack ? marker->cap_stack * 2 : 64;
    PolyBuffer **stack = realloc(marker->stack, (size_t)capacity * sizeof(*stack));
    if (!stack) return false;
    marker->stack = stack;
    marker->cap_stack = capacity;
  }
  marker->stack[marker->n_stack++] = root;
  while (marker->n_stack > 0) {
    PolyBuffer *buffer = marker->stack[--marker->n_stack];
    if (!buffer || poly_map_get(marker->visited, poly_ptr_hash(buffer), buffer, poly_ptr_eq))
      continue;
    poly_map_set(marker->visited, poly_ptr_hash(buffer), buffer, buffer, poly_ptr_eq);
    if (!ir_mark_root(marker->ir, buffer->device_uop) ||
        !ir_mark_root(marker->ir, buffer->memory_device_uop))
      return false;
    int n_owned = buffer->n_bufs + 2;
    if (n_owned > marker->cap_stack - marker->n_stack) {
      int capacity = marker->cap_stack;
      while (capacity - marker->n_stack < n_owned)
        capacity *= 2;
      PolyBuffer **stack = realloc(marker->stack, (size_t)capacity * sizeof(*stack));
      if (!stack) return false;
      marker->stack = stack;
      marker->cap_stack = capacity;
    }
    marker->stack[marker->n_stack++] = buffer->base;
    marker->stack[marker->n_stack++] = buffer->src;
    for (int i = 0; i < buffer->n_bufs; i++)
      marker->stack[marker->n_stack++] = buffer->bufs[i];
  }
  return true;
}

static void mark_buffer_row_ir(const void *key, void *value, void *userdata) {
  BufferIrMarker *marker = userdata;
  if (!marker || marker->failed) return;
  /* Tinygrad Buffer objects own their exact DEVICE identity. Polygrad keeps
   * equivalent runtime metadata outside BUFFER.src, so collection must close
   * that C-only ownership edge before evicting weak CSE rows. */
  marker->failed = !ir_mark_root(marker->ir, (PolyUOp *)key) ||
                   !mark_buffer_ir_root(marker, (PolyBuffer *)value);
}

static int mark_live_ir(PolyCtx *ctx, PolyUOp *transient_root, PolyMap *live) {
  IrMarker marker = {.live = live};
  for (int i = 0; !marker.failed && i < ctx->n_tensors; i++) {
    PolyTensor *tensor = ctx->tensors[i];
    if (!tensor || tensor->owner_refs == 0) continue;
    marker.failed =
        !ir_mark_root(&marker, tensor->uop_logical) || !ir_mark_root(&marker, tensor->uop_physical);
  }
  if (!marker.failed && transient_root) marker.failed = !ir_mark_root(&marker, transient_root);
  if (!marker.failed) poly_map_foreach(ctx->retained_uops, mark_retained_ir, &marker);
  BufferIrMarker buffers = {.ir = &marker, .visited = poly_map_new(64)};
  if (!marker.failed && !buffers.visited)
    marker.failed = true;
  else if (!marker.failed)
    poly_map_foreach(ctx->buffers, mark_buffer_row_ir, &buffers);
  marker.failed = marker.failed || buffers.failed;
  free(buffers.stack);
  poly_map_destroy(buffers.visited);
  free(marker.stack);
  return marker.failed ? -1 : 0;
}

static void queue_residency_row(const void *key, void *value, void *userdata) {
  (void)value;
  residency_candidate(userdata, (PolyUOp *)key);
}

static void evict_candidate_graph(const void *key, void *value, void *userdata) {
  (void)value;
  PolyCtx *ctx = userdata;
  if (!poly_map_get(ctx->residency_refs, poly_ptr_hash(key), key, poly_ptr_eq))
    poly_graph_cache_evict(ctx, (PolyUOp *)key);
}

static void sweep_candidate_buffer(const void *key, void *value, void *userdata) {
  (void)value;
  PolyCtx *ctx = userdata;
#ifdef POLY_TESTING
  ctx->residency_sweep_count++;
#endif
  BufferRecord *record = poly_map_get(ctx->buffer_records, poly_ptr_hash(key), key, poly_ptr_eq);
  /* All published aliases count, even dead ones. Removing a dead alias queues
   * its owner again, so storage outlives views regardless of hash-table order. */
  if (record && !record->borrowers &&
      !poly_map_get(ctx->residency_refs, poly_ptr_hash(key), key, poly_ptr_eq))
    poly_buffer_remove(ctx, (PolyUOp *)key);
}

static int poly_ctx_collect_with_root(PolyCtx *ctx, PolyUOp *transient_root, bool collect_ir) {
  if (!ctx || ctx->collecting) return -1;
  ctx->collecting = true;
  bool rebuild = !ctx->residency_refs_valid;
  bool failed = rebuild && !residency_rebuild(ctx);
  if (!failed && rebuild) {
    poly_map_foreach(ctx->buffers, queue_residency_row, ctx);
    poly_map_foreach(ctx->graph_cache, queue_residency_row, ctx);
  }
  if (!failed && ctx->check_residency && (!residency_check(ctx) || !buffer_records_check(ctx))) {
    fprintf(stderr, "polygrad: residency verification failed before collection\n");
    abort();
  }
  bool transient_retained = false;
  if (!failed && transient_root) {
    transient_retained = residency_ref_update(ctx, transient_root, true);
    if (!transient_retained) ctx->residency_refs_valid = false;
    failed = !transient_retained;
  }

  while (!failed && poly_map_len(ctx->residency_candidates)) {
    PolyMap *candidates = ctx->residency_candidates;
    ctx->residency_candidates = poly_map_new(16);
    /* Native graph objects must die before their referenced storage. */
    poly_map_foreach(candidates, evict_candidate_graph, ctx);
    poly_map_foreach(candidates, sweep_candidate_buffer, ctx);
    poly_map_destroy(candidates);
  }
  if (!failed) {
    ctx->collection_dirty = false;
    ctx->collection_requested = false;
  }
  PolyMap *live_ir = NULL;
  if (!failed && collect_ir) {
    live_ir = poly_map_new(256);
    PolyUOp **shape_roots = NULL;
    int n_shape_roots = 0;
    if (!live_ir || mark_live_ir(ctx, transient_root, live_ir) != 0 ||
        poly_shape_cache_live_value_roots(ctx, live_ir, &shape_roots, &n_shape_roots) != 0)
      failed = true;
    IrMarker shape_marker = {.live = live_ir};
    for (int i = 0; !failed && i < n_shape_roots; i++)
      failed = !ir_mark_root(&shape_marker, shape_roots[i]);
    free(shape_marker.stack);
    free(shape_roots);
    if (!failed && (poly_shape_cache_evict_unmarked(ctx, live_ir) != 0 ||
                    poly_uop_cse_evict_unmarked(ctx, live_ir) != 0))
      failed = true;
    if (!failed) {
      ctx->ir_collection_dirty = false;
      ctx->ir_collection_baseline_bytes = ctx->uop_storage_bytes;
    }
  }
  poly_map_destroy(live_ir);
  if (transient_retained) poly_ctx_root_released(ctx, transient_root);
  ctx->collecting = false;
  return failed ? -1 : 0;
}

int poly_ctx_collect(PolyCtx *ctx) {
  return poly_ctx_collect_with_root(ctx, NULL, true);
}

bool poly_ctx_ir_collection_due(const PolyCtx *ctx) {
  if (!ctx || !ctx->ir_collection_dirty) return false;
  if (ctx->uop_storage_bytes < ctx->ir_collection_baseline_bytes) return true;
  /* Amortize tracing against the retained graph, not a fixed allocation
   * interval that makes small calls repeatedly walk unrelated large Models. */
  size_t growth = ctx->ir_collection_baseline_bytes / 2;
  if (growth < POLY_IR_COLLECTION_MIN_GROWTH) growth = POLY_IR_COLLECTION_MIN_GROWTH;
  return ctx->uop_storage_bytes - ctx->ir_collection_baseline_bytes >= growth;
}

int poly_ctx_collect_at_safe_point(PolyCtx *ctx) {
  if (!ctx) return -1;
  bool collect_ir = poly_ctx_ir_collection_due(ctx);
  if (!ctx->collection_dirty && !ctx->collection_requested && !collect_ir) return 0;
  return poly_ctx_collect_with_root(ctx, NULL, collect_ir);
}

int poly_ctx_collect_before_allocation(PolyCtx *ctx, PolyUOp *transient_root) {
  if (!ctx || !ctx->collection_dirty) return ctx ? 0 : -1;
  /* poly_run_linear retains its complete LINEAR, so allocation-time
   * collection preserves every current CALL argument without a pre-run scan. */
  return poly_ctx_collect_with_root(ctx, transient_root, false);
}

bool poly_ctx_owns_ptr(PolyCtx *ctx, const void *p) {
  if (!ctx || !p) return false;
  if (poly_uop_storage_contains(ctx, p)) return true;
  uintptr_t addr = (uintptr_t)p;
  for (PolyArenaBlock *b = ctx->arena->head; b; b = b->next) {
    uintptr_t start = (uintptr_t)b->data;
    uintptr_t end = start + b->used;
    if (addr >= start && addr < end) return true;
  }
  return false;
}

void poly_ctx_set_preferred_device(PolyCtx *ctx, PolyDevice device) {
  if (!ctx) return;
  ctx->preferred_device = device;
}

PolyDevice poly_ctx_get_preferred_device(PolyCtx *ctx) {
  return ctx ? ctx->preferred_device : POLY_DEVICE_AUTO;
}

int poly_ctx_set_kernel_policy(PolyCtx *ctx, int policy) {
  if (!ctx || policy < 0 || policy > 1) return -1;
  /* Retained Model/JIT schedules must not silently outlive their selection
   * policy. Each context has its own caches; choose policy before lowering. */
  if (ctx->kernel_policy_locked && ctx->kernel_policy != policy) return -1;
  ctx->kernel_policy = policy;
  return 0;
}

int poly_ctx_set_logical_policy(PolyCtx *ctx, PolyLogicalPolicy policy) {
  /* Polygrad logical/placement boundary: this default is copied only by
   * subsequently constructed Tensor handles. */
  if (!ctx || policy < POLY_LOGICAL_NEVER || policy > POLY_LOGICAL_UNTIL_REALIZE) return -1;
  ctx->logical_policy = policy;
  return 0;
}

PolyLogicalPolicy poly_ctx_get_logical_policy(const PolyCtx *ctx) {
  return ctx ? ctx->logical_policy : POLY_LOGICAL_UNTIL_REALIZE;
}

void poly_ctx_set_frontend_buffer_release(PolyCtx *ctx, PolyFrontendBufferReleaseFn fn) {
  if (!ctx) return;
  ctx->frontend_buffer_release = fn;
}

PolyArena *poly_ctx_arena(PolyCtx *ctx) {
  return ctx->arena;
}
PolyMap *poly_ctx_shape_cache(PolyCtx *ctx) {
  return ctx->shape_cache;
}

typedef struct {
  size_t current;
  size_t source;
} BufferByteStats;

static void accum_buffer_bytes(const void *key, void *value, void *userdata) {
  (void)key;
  BufferByteStats *stats = (BufferByteStats *)userdata;
  PolyBuffer *b = (PolyBuffer *)value;
  if (!stats || !b) return;
  if (b->owned) stats->current += b->nbytes;
  if (b->src && b->src->owned) stats->source += b->src->nbytes;
}

int poly_ctx_stats(PolyCtx *ctx, PolyCtxStats *out) {
  if (!ctx || !out) return -1;
  /* Tinygrad 2026-08-22/a9069c177a9d helpers.py:298-306 exposes plain
   * GlobalCounters fields. A counter read must not change graph lifetime. */
  memset(out, 0, sizeof(*out));
  out->arena_bytes = poly_arena_used(ctx->arena) + ctx->uop_storage_bytes;
  out->arena_high_water = poly_arena_high_water(ctx->arena) + ctx->uop_storage_high_water;
  out->scratch_bytes = poly_arena_used(ctx->scratch);
  out->scratch_high_water = poly_arena_high_water(ctx->scratch);
  out->cse_entries = poly_map_len(ctx->cse);
  out->to_program_cache_entries = poly_map_len(ctx->to_program_cache);
  out->runtime_cache_entries = poly_map_len(ctx->runtime_cache);
  out->runtime_artifact_entries = ctx->runtime_artifact_entries;
  out->shape_cache_entries = poly_map_len(ctx->shape_cache);
  out->buffer_entries = poly_map_len(ctx->buffers);
  BufferByteStats buf_stats = {0};
  poly_map_foreach(ctx->buffers, accum_buffer_bytes, &buf_stats);
  out->buffer_owned_current_bytes = buf_stats.current;
  out->buffer_owned_source_bytes = buf_stats.source;
  out->buffer_owned_bytes = buf_stats.current + buf_stats.source;
  out->tensor_records = (size_t)ctx->n_tensors;
  out->compiled_artifact_bytes = poly_runtime_cache_artifact_bytes(ctx);
  out->launch_count = ctx->launch_count;
  out->runtime_cache_hits = ctx->runtime_cache_hits;
  out->runtime_cache_misses = ctx->runtime_cache_misses;
  out->buffer_read_count = ctx->buffer_read_count;
  out->buffer_read_bytes = ctx->buffer_read_bytes;
  out->buffer_write_count = ctx->buffer_write_count;
  out->buffer_write_bytes = ctx->buffer_write_bytes;
  out->buffer_copy_count = ctx->buffer_copy_count;
  out->buffer_copy_bytes = ctx->buffer_copy_bytes;
  out->global_ops = ctx->global_ops;
  out->global_mem = ctx->global_mem;
  out->time_sum_s = ctx->time_sum_s;
  out->kernel_count = ctx->kernel_count;
  out->mem_used = ctx->mem_used;
  return 0;
}

void poly_ctx_reset_counters(PolyCtx *ctx) {
  if (!ctx) return;
  ctx->global_ops = 0;
  ctx->global_mem = 0;
  ctx->time_sum_s = 0.0;
  ctx->kernel_count = 0;
}

uint64_t poly_ctx_mem_used_for_device(PolyCtx *ctx, PolyDevice device) {
  if (!ctx || device < POLY_DEVICE_AUTO || device > POLY_DEVICE_DISK) return 0;
  return ctx->mem_used_per_device[device];
}

typedef struct {
  PolyUOp *device_uop;
  uint64_t bytes;
} PolyDeviceMemoryEntry;

static PolyDeviceMemoryEntry *poly_ctx_memory_entry(
    PolyCtx *ctx,
    PolyUOp *device_uop,
    bool create
) {
  if (!ctx || !ctx->mem_used_by_device || !device_uop) return NULL;
  PolyDeviceMemoryEntry *entry =
      poly_map_get(ctx->mem_used_by_device, poly_ptr_hash(device_uop), device_uop, poly_ptr_eq);
  if (entry || !create) return entry;
  entry = malloc(sizeof(*entry));
  if (!entry) return NULL;
  *entry = (PolyDeviceMemoryEntry){.device_uop = device_uop, .bytes = 0};
  poly_map_set(ctx->mem_used_by_device, poly_ptr_hash(device_uop), device_uop, entry, poly_ptr_eq);
  return entry;
}

uint64_t poly_ctx_mem_used_for_device_uop(PolyCtx *ctx, PolyUOp *device_uop) {
  PolyDeviceMemoryEntry *entry = poly_ctx_memory_entry(ctx, device_uop, false);
  return entry ? entry->bytes : 0;
}

void poly_ctx_record_memory_alloc_exact(
    PolyCtx *ctx,
    PolyUOp *device_uop,
    PolyDevice backend,
    size_t nbytes
) {
  if (!ctx || nbytes == 0) return;
  /* New residency needs a coverage snapshot even if no owner was released. */
  ctx->collection_requested = true;
  uint64_t bytes = (uint64_t)nbytes;
  ctx->mem_used = UINT64_MAX - ctx->mem_used < bytes ? UINT64_MAX : ctx->mem_used + bytes;
  if (backend >= POLY_DEVICE_AUTO && backend <= POLY_DEVICE_DISK) {
    uint64_t *per_device = &ctx->mem_used_per_device[backend];
    *per_device = UINT64_MAX - *per_device < bytes ? UINT64_MAX : *per_device + bytes;
  }
  PolyDeviceMemoryEntry *entry = poly_ctx_memory_entry(ctx, device_uop, true);
  if (entry) entry->bytes = UINT64_MAX - entry->bytes < bytes ? UINT64_MAX : entry->bytes + bytes;
}

void poly_ctx_record_memory_free_exact(
    PolyCtx *ctx,
    PolyUOp *device_uop,
    PolyDevice backend,
    size_t nbytes
) {
  if (!ctx || nbytes == 0) return;
  uint64_t bytes = (uint64_t)nbytes;
  ctx->mem_used = ctx->mem_used >= bytes ? ctx->mem_used - bytes : 0;
  if (backend >= POLY_DEVICE_AUTO && backend <= POLY_DEVICE_DISK) {
    uint64_t *per_device = &ctx->mem_used_per_device[backend];
    *per_device = *per_device >= bytes ? *per_device - bytes : 0;
  }
  PolyDeviceMemoryEntry *entry = poly_ctx_memory_entry(ctx, device_uop, false);
  if (entry) {
    entry->bytes = entry->bytes >= bytes ? entry->bytes - bytes : 0;
    /* No residency remains to keep this DEVICE alive. Drop the weak key and
     * its record together rather than accumulating entries in the arena. */
    if (entry->bytes == 0) {
      poly_map_remove(ctx->mem_used_by_device, poly_ptr_hash(device_uop), device_uop, poly_ptr_eq);
      free(entry);
    }
  }
}

void poly_ctx_record_memory_alloc(PolyCtx *ctx, PolyDevice device, size_t nbytes) {
  poly_ctx_record_memory_alloc_exact(ctx, poly_device_uop(ctx, device), device, nbytes);
}

void poly_ctx_record_memory_free(PolyCtx *ctx, PolyDevice device, size_t nbytes) {
  poly_ctx_record_memory_free_exact(ctx, poly_device_uop(ctx, device), device, nbytes);
}

#ifdef __EMSCRIPTEN__
/* wasm_common.js reads this public struct manually. Keep the ABI facts
 * compile-checked instead of relying on an unverified offset table. */
_Static_assert(offsetof(PolyCtxStats, global_ops) == 96, "wasm PolyCtxStats.global_ops offset");
_Static_assert(offsetof(PolyCtxStats, global_mem) == 104, "wasm PolyCtxStats.global_mem offset");
_Static_assert(offsetof(PolyCtxStats, time_sum_s) == 112, "wasm PolyCtxStats.time_sum_s offset");
_Static_assert(
    offsetof(PolyCtxStats, kernel_count) == 120,
    "wasm PolyCtxStats.kernel_count offset"
);
_Static_assert(offsetof(PolyCtxStats, mem_used) == 128, "wasm PolyCtxStats.mem_used offset");
_Static_assert(sizeof(PolyCtxStats) == 136, "wasm PolyCtxStats size");
#endif

PolyScratchMark poly_ctx_scratch_mark(PolyCtx *ctx) {
  return poly_arena_mark(ctx ? ctx->scratch : NULL);
}

void poly_ctx_scratch_rewind(PolyCtx *ctx, PolyScratchMark mark) {
  if (!ctx) return;
  poly_arena_rewind(ctx->scratch, mark);
}

void *poly_ctx_scratch_alloc(PolyCtx *ctx, size_t size, size_t align) {
  if (!ctx || !ctx->scratch) return NULL;
  return poly_arena_alloc(ctx->scratch, size, align);
}

int64_t poly_ctx_next_unique_id(PolyCtx *ctx) {
  return ctx ? ctx->next_unique_id++ : 0;
}

void poly_ctx_reserve_unique_id(PolyCtx *ctx, int64_t id) {
  if (!ctx || id < 0) return;
  if (ctx->next_unique_id <= id) ctx->next_unique_id = id + 1;
}

void poly_ctx_reserve_buf_tag(PolyCtx *ctx, int32_t tag) {
  if (!ctx || tag <= 0) return;
  if (ctx->next_buf_tag <= tag) ctx->next_buf_tag = tag + 1;
}

int poly_abi_version(void) {
  return POLYGRAD_ABI_VERSION;
}
