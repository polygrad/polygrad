/* Current Tinygrad schedule/memory.py memory-plan rewrite. */

#include "schedule/memory.h"

#include "ctx.h"
#include "device.h"
#include "frontend_internal.h"
#include "runtime/support/memory.h"
#include "utils.h"

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  PolyUOp *buffer;
  PolyUOp *device;
  int first;
  int last;
  bool copy;
  size_t nbytes;
  size_t allocation_size;
  size_t offset;
  int lane;
  PolyUOp *replacement;
} MemoryBuffer;

typedef struct {
  PolyUOp *device;
  bool copy;
  size_t peak;
  PolyTLSFAllocator *allocator;
  PolyUOp *arena;
} MemoryLane;

typedef struct {
  int position;
  bool open;
  int buffer;
} MemoryEvent;

static size_t round_up(size_t value, size_t block) {
  size_t remainder = block ? value % block : 0;
  return remainder ? value + block - remainder : value;
}

/* Current Tinygrad schedule/memory.py:_collect_bufs. */
bool poly_collect_bufs(PolyUOp *uop, PolyUOp ***buffers, int *count, int *capacity) {
  if (!uop) return true;
  if (uop->op == POLY_OP_BUFFER) {
    if (*count >= *capacity) {
      int next = *capacity ? *capacity * 2 : 4;
      PolyUOp **grown = realloc(*buffers, (size_t)next * sizeof(*grown));
      if (!grown) return false;
      *buffers = grown;
      *capacity = next;
    }
    (*buffers)[(*count)++] = uop;
    return true;
  }
  if (uop->op != POLY_OP_MSELECT && uop->op != POLY_OP_MSTACK) return true;
  for (int i = 0; i < uop->n_src; i++)
    if (!poly_collect_bufs(uop->src[i], buffers, count, capacity)) return false;
  return true;
}

static bool held_buffer(PolyUOp *buffer, PolyUOp **held, int held_count) {
  for (int i = 0; i < held_count; i++)
    if (held[i] == buffer) return true;
  return false;
}

/* Current Tinygrad memory.py:_can_plan checks every scalar device in the
 * BUFFER's string-or-tuple device value. */
static bool can_plan_device(const char *name) {
  return name && strncmp(name, "DISK", 4) != 0 && strncmp(name, "TINYFS", 6) != 0 &&
         strncmp(name, "CL", 2) != 0 && strncmp(name, "WEBGPU", 6) != 0;
}

/* Current Tinygrad schedule/memory.py:_can_plan. */
static bool can_plan(PolyCtx *ctx, PolyUOp *buffer, PolyUOp **held, int held_count) {
  if (held_buffer(buffer, held, held_count)) return false;
  PolyUOp *device = poly_uop_device_uop_cached(ctx, buffer, NULL);
  if (!device || device->op != POLY_OP_DEVICE) return false;
  if (device->arg.kind == POLY_ARG_STRING) return can_plan_device(device->arg.str);
  if (device->arg.kind == POLY_ARG_STRING_TUPLE) {
    if (device->arg.string_tuple.n <= 0 || !device->arg.string_tuple.vals) return false;
    for (int i = 0; i < device->arg.string_tuple.n; i++)
      if (!can_plan_device(device->arg.string_tuple.vals[i])) return false;
    return true;
  }
  return false;
}

static int memory_buffer_index(MemoryBuffer *buffers, int count, PolyUOp *buffer) {
  for (int i = 0; i < count; i++)
    if (buffers[i].buffer == buffer) return i;
  return -1;
}

static int memory_lane_index(
    MemoryLane **lanes,
    int *count,
    int *capacity,
    PolyUOp *device,
    bool copy
) {
  for (int i = 0; i < *count; i++)
    if ((*lanes)[i].device == device && (*lanes)[i].copy == copy) return i;
  if (*count >= *capacity) {
    int next = *capacity ? *capacity * 2 : 4;
    MemoryLane *grown = realloc(*lanes, (size_t)next * sizeof(*grown));
    if (!grown) return -1;
    *lanes = grown;
    *capacity = next;
  }
  int index = (*count)++;
  (*lanes)[index] = (MemoryLane){.device = device, .copy = copy};
  return index;
}

static int compare_events(const void *left, const void *right) {
  const MemoryEvent *a = left, *b = right;
  if (a->position != b->position) return a->position < b->position ? -1 : 1;
  if (a->open != b->open) return a->open ? 1 : -1;
  return a->buffer - b->buffer;
}

static PolyUOp *new_arena(PolyCtx *ctx, PolyUOp *device, size_t size) {
  if (!ctx || !device || device->op != POLY_OP_DEVICE ||
      (device->arg.kind != POLY_ARG_STRING && device->arg.kind != POLY_ARG_STRING_TUPLE) ||
      size > INT64_MAX)
    return NULL;
  /* Current Tinygrad memory_plan_rewrite creates the arena with LaneKey's
   * exact device string; a backend enum would collapse CPU:1 into CPU. */
  return poly_uop_new_buffer(
      ctx, device, (int64_t)size, POLY_INT8, poly_ctx_next_unique_id(ctx)
  );
}

static PolyUOp *arena_view(
    PolyCtx *ctx,
    PolyUOp *arena,
    size_t offset,
    size_t nbytes,
    PolyDType dtype
) {
  if (offset > INT64_MAX || nbytes > INT64_MAX) return NULL;
  PolyUOp *start = poly_const_int(ctx, (int64_t)offset);
  PolyUOp *size = poly_const_int(ctx, (int64_t)nbytes);
  PolyUOp *slice = start && size ? poly_shrink_uop(ctx, arena, &start, &size, 1) : NULL;
  return slice ? poly_uop1(ctx, POLY_OP_BITCAST, dtype, slice, poly_arg_none())
               : NULL;
}

PolyUOp *poly_memory_plan_rewrite(
    PolyCtx *ctx,
    PolyUOp *linear,
    PolyUOp **held_buffers,
    int held_count
) {
  if (!ctx || !linear || linear->op != POLY_OP_LINEAR || held_count < 0 ||
      (held_count > 0 && !held_buffers))
    return NULL;
  if (poly_getenv_flag("POLY_NO_MEMORY_PLANNER")) return linear;

  MemoryBuffer *buffers = NULL;
  int buffer_count = 0, buffer_capacity = 0;
  for (int call_index = 0; call_index < linear->n_src; call_index++) {
    PolyUOp *call = linear->src[call_index];
    if (!call || call->op != POLY_OP_CALL || call->n_src < 1) goto fail;
    bool copy = call->src[0]->op == POLY_OP_COPY;
    for (int source = 1; source < call->n_src; source++) {
      PolyUOp **found = NULL;
      int found_count = 0, found_capacity = 0;
      if (!poly_collect_bufs(call->src[source], &found, &found_count, &found_capacity)) {
        free(found);
        goto fail;
      }
      for (int i = 0; i < found_count; i++) {
        PolyUOp *buffer = found[i];
        if (!can_plan(ctx, buffer, held_buffers, held_count)) continue;
        int index = memory_buffer_index(buffers, buffer_count, buffer);
        if (index < 0) {
          if (buffer_count >= buffer_capacity) {
            int next = buffer_capacity ? buffer_capacity * 2 : 8;
            MemoryBuffer *grown = realloc(buffers, (size_t)next * sizeof(*grown));
            if (!grown) {
              free(found);
              goto fail;
            }
            buffers = grown;
            buffer_capacity = next;
          }
          int64_t numel = poly_uop_max_numel(ctx, buffer);
          size_t itemsize = poly_dtype_itemsize(buffer->dtype);
          if (numel < 0 || itemsize == 0 || (uint64_t)numel > SIZE_MAX / itemsize) {
            free(found);
            goto fail;
          }
          index = buffer_count++;
          buffers[index] = (MemoryBuffer){
              .buffer = buffer,
              .device = poly_uop_device_uop_cached(ctx, buffer, NULL),
              .first = call_index,
              .last = call_index,
              .copy = copy,
              .nbytes = (size_t)numel * itemsize,
          };
          buffers[index].allocation_size = round_up(buffers[index].nbytes, 256);
        } else {
          buffers[index].last = call_index;
          buffers[index].copy = buffers[index].copy || copy;
        }
      }
      free(found);
    }
  }
  if (buffer_count == 0) {
    free(buffers);
    return linear;
  }

  size_t total_memory = 0;
  MemoryLane *lanes = NULL;
  int lane_count = 0, lane_capacity = 0;
  for (int i = 0; i < buffer_count; i++) {
    if (buffers[i].allocation_size > SIZE_MAX - total_memory) goto fail_lanes;
    total_memory += buffers[i].allocation_size;
    buffers[i].lane = memory_lane_index(
        &lanes, &lane_count, &lane_capacity, buffers[i].device, buffers[i].copy
    );
    if (buffers[i].lane < 0) goto fail_lanes;
  }
  if (total_memory > SIZE_MAX / 2) goto fail_lanes;
  total_memory *= 2;
  for (int i = 0; i < lane_count; i++) {
    lanes[i].allocator = poly_tlsf_allocator_new(total_memory, 256, 32);
    if (!lanes[i].allocator) goto fail_lanes;
  }

  MemoryEvent *events = malloc((size_t)buffer_count * 2 * sizeof(*events));
  if (!events) goto fail_lanes;
  for (int i = 0; i < buffer_count; i++) {
    int hold = buffers[i].copy ? buffers[i].last - buffers[i].first + 1 : 0;
    events[2 * i] = (MemoryEvent){buffers[i].first, true, i};
    events[2 * i + 1] = (MemoryEvent){buffers[i].last + 1 + hold, false, i};
  }
  qsort(events, (size_t)buffer_count * 2, sizeof(*events), compare_events);
  for (int i = 0; i < buffer_count * 2; i++) {
    MemoryBuffer *buffer = &buffers[events[i].buffer];
    MemoryLane *lane = &lanes[buffer->lane];
    if (events[i].open) {
      buffer->offset = poly_tlsf_allocator_alloc(lane->allocator, buffer->allocation_size);
      if (buffer->offset == SIZE_MAX) {
        free(events);
        goto fail_lanes;
      }
      size_t end = buffer->offset + buffer->nbytes;
      if (end > lane->peak) lane->peak = end;
    } else if (poly_tlsf_allocator_free(lane->allocator, buffer->offset) != 0) {
      free(events);
      goto fail_lanes;
    }
  }
  free(events);

  for (int i = 0; i < lane_count; i++) {
    size_t size = round_up(lanes[i].peak, 256);
    lanes[i].arena = new_arena(ctx, lanes[i].device, size);
    if (!lanes[i].arena) goto fail_lanes;
  }
  PolyUOp **from = malloc((size_t)buffer_count * sizeof(*from));
  PolyUOp **to = malloc((size_t)buffer_count * sizeof(*to));
  if (!from || !to) {
    free(from);
    free(to);
    goto fail_lanes;
  }
  for (int i = 0; i < buffer_count; i++) {
    buffers[i].replacement = arena_view(
        ctx, lanes[buffers[i].lane].arena, buffers[i].offset, buffers[i].nbytes,
        buffers[i].buffer->dtype
    );
    if (!buffers[i].replacement) {
      free(from);
      free(to);
      goto fail_lanes;
    }
    from[i] = buffers[i].buffer;
    to[i] = buffers[i].replacement;
  }
  PolyUOp *planned = poly_uop_substitute(ctx, linear, from, to, buffer_count);
  free(from);
  free(to);
  for (int i = 0; i < lane_count; i++) poly_tlsf_allocator_destroy(lanes[i].allocator);
  free(lanes);
  free(buffers);
  return planned;

fail_lanes:
  if (lanes)
    for (int i = 0; i < lane_count; i++) poly_tlsf_allocator_destroy(lanes[i].allocator);
  free(lanes);
fail:
  free(buffers);
  return NULL;
}
