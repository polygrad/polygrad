/* placer.c -- physical graph lowering.
 *
 * Logical UOps remain device-free. This pass projects a placed PolyTensor's
 * logical graph into a tinygrad-like physical graph using DEVICE and COPY UOps.
 */

#include "ctx.h"
#include "device.h"
#include "frontend_internal.h"
#include "tensor.h"
#include "utils.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  PolyCtx *ctx;
  PolyMap *memo[POLY_DEVICE_DISK + 1];
  PolyTensor *active_place;
  int suppress_place_facts;
} PolyPhysicalizer;

static PolyDevice device_from_string_arg(const char *s) {
  if (!s) return POLY_DEVICE_AUTO;
  if (strcmp(s, "CPU") == 0) return POLY_DEVICE_CPU;
  if (strcmp(s, "CUDA") == 0) return POLY_DEVICE_CUDA;
  if (strcmp(s, "HIP") == 0) return POLY_DEVICE_HIP;
  if (strcmp(s, "WEBGPU") == 0) return POLY_DEVICE_WEBGPU;
  if (strcmp(s, "INTERP") == 0) return POLY_DEVICE_INTERP;
  if (strcmp(s, "X86") == 0) return POLY_DEVICE_X86;
  return poly_device_by_name(s);
}

PolyDevice poly_device_from_device_uop(PolyUOp *device) {
  if (!device || device->op != POLY_OP_DEVICE) return POLY_DEVICE_AUTO;
  if (device->arg.kind == POLY_ARG_INT) return (PolyDevice)device->arg.i;
  if (device->arg.kind == POLY_ARG_STRING) return device_from_string_arg(device->arg.str);
  return POLY_DEVICE_AUTO;
}

/* Derived physical-device property for a UOp graph.
 *
 * Tinygrad exposes this as UOp.device backed by cached UOp._device:
 * DEVICE returns itself, COPY/BUFFER take their second source's DEVICE, AFTER
 * follows its value source, and generic ops inherit the first concrete device
 * found in their sources. Polygrad uses the same structural rule after
 * placement because physical graphs carry device as DEVICE/COPY/BUFFER UOps,
 * while logical/pre-placement graphs may still return AUTO.
 *
 * This public wrapper is for one-off queries. Hot passes should use the
 * cached backing helper below with a pass-local PolyMap.
 */
PolyDevice poly_uop_device(PolyUOp *u) {
  if (!u) return POLY_DEVICE_AUTO;
  PolyMap *cache = poly_map_new(64);
  if (!cache) return POLY_DEVICE_AUTO;
  PolyDevice device = poly_uop_device_cached(u, cache);
  poly_map_destroy(cache);
  return device;
}

/* Cached backing helper for poly_uop_device().
 *
 * Rangeify and schedule construction can ask for the device of many related
 * subgraphs. Without a shared cache, each query recursively walks the same
 * large physical graph again, which caused large training graphs to become
 * quadratic. This mirrors tinygrad's recursive_property cache without adding
 * mutable cache fields to every arena-allocated PolyUOp.
 */
PolyDevice poly_uop_device_cached(PolyUOp *u, PolyMap *cache) {
  if (!u) return POLY_DEVICE_AUTO;
  if (cache) {
    void *cached = poly_map_get(cache, poly_ptr_hash(u), u, poly_ptr_eq);
    if (cached) return (PolyDevice)((intptr_t)cached - 1);
  }

  PolyDevice result = POLY_DEVICE_AUTO;
  if (u->op == POLY_OP_DEVICE) {
    result = poly_device_from_device_uop(u);
    if (cache) poly_map_set(cache, poly_ptr_hash(u), u, (void *)(intptr_t)(result + 1), poly_ptr_eq);
    return result;
  }
  if (u->op == POLY_OP_COPY && u->n_src >= 2) {
    result = poly_device_from_device_uop(u->src[1]);
    if (cache) poly_map_set(cache, poly_ptr_hash(u), u, (void *)(intptr_t)(result + 1), poly_ptr_eq);
    return result;
  }
  if (u->op == POLY_OP_BUFFER && u->n_src >= 2) {
    result = poly_device_from_device_uop(u->src[1]);
    if (cache) poly_map_set(cache, poly_ptr_hash(u), u, (void *)(intptr_t)(result + 1), poly_ptr_eq);
    return result;
  }
  if (u->op == POLY_OP_PARAM && u->arg.kind == POLY_ARG_PARAM && u->arg.param &&
      u->arg.param->device != POLY_DEVICE_AUTO) {
    result = (PolyDevice)u->arg.param->device;
    if (cache)
      poly_map_set(cache, poly_ptr_hash(u), u, (void *)(intptr_t)(result + 1), poly_ptr_eq);
    return result;
  }
  if (u->op == POLY_OP_AFTER && u->n_src >= 1) {
    result = poly_uop_device_cached(u->src[0], cache);
    if (cache) poly_map_set(cache, poly_ptr_hash(u), u, (void *)(intptr_t)(result + 1), poly_ptr_eq);
    return result;
  }

  if (u->op != POLY_OP_CONST && u->op != POLY_OP_VCONST) {
    for (int i = 0; i < u->n_src; i++) {
      PolyDevice child = poly_uop_device_cached(u->src[i], cache);
      if (child == POLY_DEVICE_AUTO) continue;
      if (result == POLY_DEVICE_AUTO) {
        result = child;
      } else if (!poly_devices_share_storage(result, child)) {
        result = POLY_DEVICE_AUTO;
        break;
      }
    }
  }

  if (cache) poly_map_set(cache, poly_ptr_hash(u), u, (void *)(intptr_t)(result + 1), poly_ptr_eq);
  return result;
}

static PolyUOp *make_device_uop(PolyCtx *ctx, PolyDevice device) {
  return poly_uop0(ctx, POLY_OP_DEVICE, POLY_VOID, poly_arg_int((int64_t)device));
}

static PolyUOp *copy_to_device(PolyCtx *ctx, PolyUOp *value, PolyDevice device) {
  if (!ctx || !value || device == POLY_DEVICE_AUTO) return value;
  if (value->op == POLY_OP_COPY && value->n_src >= 2 &&
      poly_device_from_device_uop(value->src[1]) == device)
    return value;
  PolyUOp *src[2] = {value, make_device_uop(ctx, device)};
  return poly_uop(ctx, POLY_OP_COPY, value->dtype, src, 2, poly_arg_none());
}

static bool placement_devices_share_storage(PolyDevice a, PolyDevice b) {
  if (a == POLY_DEVICE_AUTO || b == POLY_DEVICE_AUTO) return false;
  if (poly_devices_share_storage(a, b)) return true;
  if (a == POLY_DEVICE_HOST || b == POLY_DEVICE_HOST ||
      a == POLY_DEVICE_DISK || b == POLY_DEVICE_DISK)
    return false;
  return poly_device_is_host_addressable(a) && poly_device_is_host_addressable(b);
}

static PolyUOp *ensure_on_device(PolyCtx *ctx, PolyUOp *value, PolyDevice device) {
  if (!ctx || !value || device == POLY_DEVICE_AUTO) return value;
  PolyDevice current = poly_uop_device(value);
  if (current != POLY_DEVICE_AUTO && placement_devices_share_storage(current, device)) return value;
  const PolyUOp *identity = poly_uop_get_buffer_identity(value);
  PolyBuffer *storage = identity ? poly_buffer_get(ctx, (PolyUOp *)identity) : NULL;
  if (storage && storage->ptr && storage->device != POLY_DEVICE_AUTO &&
      placement_devices_share_storage(storage->device, device))
    return value;
  if (current == POLY_DEVICE_AUTO && placement_devices_share_storage(poly_device_default(), device))
    return value;
  return copy_to_device(ctx, value, device);
}

static PolyUOp *memo_get(PolyPhysicalizer *p, PolyUOp *logical, PolyDevice device) {
  if (!p || !logical || device <= POLY_DEVICE_AUTO || device > POLY_DEVICE_DISK ||
      !p->memo[device])
    return NULL;
  return poly_map_get(
      p->memo[device], poly_ptr_hash(logical), logical, poly_ptr_eq
  );
}

static bool memo_put(PolyPhysicalizer *p, PolyUOp *logical, PolyDevice device, PolyUOp *physical) {
  if (!p || !logical || !physical || device <= POLY_DEVICE_AUTO ||
      device > POLY_DEVICE_DISK)
    return false;
  if (!p->memo[device]) p->memo[device] = poly_map_new(128);
  poly_map_set(
      p->memo[device], poly_ptr_hash(logical), logical, physical, poly_ptr_eq
  );
  return true;
}

static void physicalizer_destroy(PolyPhysicalizer *p) {
  if (!p) return;
  for (int device = POLY_DEVICE_AUTO + 1; device <= POLY_DEVICE_DISK; device++)
    poly_map_destroy(p->memo[device]);
}

static bool placement_rebuilds_sources(PolyUOp *u) {
  if (!u) return false;
  if (u->op == POLY_OP_CONST || u->op == POLY_OP_VCONST || u->op == POLY_OP_DEFINE_VAR ||
      u->op == POLY_OP_BIND || u->op == POLY_OP_DEVICE || u->op == POLY_OP_UNIQUE ||
      u->op == POLY_OP_LUNIQUE)
    return false;
  return true;
}

static bool placement_opaque_body_op(PolyOps op) {
  return op == POLY_OP_CALL || op == POLY_OP_FUNCTION;
}

static PolyUOp *lower_value(PolyPhysicalizer *p, PolyUOp *u, PolyDevice device);
static PolyUOp *lower_effect(PolyPhysicalizer *p, PolyUOp *u, PolyDevice device);

/* Placement changes sources at the logical->physical boundary, but the
 * rebuilt UOp keeps tinygrad's original op/dtype/arg/tag identity. */
static PolyUOp *placement_rebuild_with_sources(
    PolyPhysicalizer *p,
    PolyUOp *u,
    PolyUOp **src
) {
  if (!p || !p->ctx || !u || (u->n_src > 0 && !src)) return NULL;
  return (u->tag != 0 || u->tag_arg.kind != POLY_ARG_NONE)
             ? poly_uop_tagged_arg(
                   p->ctx, u->op, u->dtype, src, u->n_src, u->arg, u->tag,
                   u->tag_arg
               )
             : poly_uop(p->ctx, u->op, u->dtype, src, u->n_src, u->arg);
}

/* An assignment destination is an existing storage identity, not an ordinary
 * value input that placement may replace with COPY(storage, device).  A PLACE
 * assignment seeds the exact destination mapping before reaching this helper;
 * otherwise retain a buffer-identity destination and lower only the value
 * being written.  Nested AFTER versions recurse through lower_value so their
 * dependency chain remains intact. */
static PolyUOp *lower_assign_target(
    PolyPhysicalizer *p,
    PolyUOp *target,
    PolyDevice device
) {
  if (!p || !target) return NULL;
  PolyUOp *placed = memo_get(p, target, device);
  if (placed) return placed;
  if (poly_uop_has_buffer_identity(target)) return target;
  return lower_value(p, target, device);
}

/* At a tensor materialization boundary, a contiguous SHRINK of an already
 * realized buffer is a Buffer view in tinygrad: no kernel is scheduled, and a
 * later .to(device) copies only the selected byte range. Reuse Polygrad's
 * existing BUFFER_VIEW identity for that physical fact while retaining the
 * logical SHRINK/RESHAPE graph. This must not run during recursive value
 * lowering: tinygrad keeps nested movement logical and parameterizes an
 * already-realized slice only when it becomes a CALL argument. */
static PolyUOp *lower_contiguous_realized_view(
    PolyPhysicalizer *p,
    PolyUOp *u,
    PolyDevice device
) {
  if (!p || !u) return NULL;
  PolyUOp *identity = NULL;
  PolyShape shape = {.ndim = -1};
  int64_t numel = -1;
  size_t byte_offset = 0;
  if (!poly_uop_contiguous_view_info(
          p->ctx, u, &identity, &shape, &numel, &byte_offset
      ))
    return NULL;
  PolyBuffer *storage = poly_buffer_get(p->ctx, identity);
  if (!storage) return NULL;
  if (!placement_devices_share_storage(storage->device, device)) return NULL;

  PolyUOp *view = poly_buffer_view(p->ctx, identity, numel, byte_offset);
  if (!view) return NULL;
  if (shape.ndim == 1 && shape.dims[0] == numel) return view;
  return poly_reshape(p->ctx, view, shape.dims, shape.ndim);
}

static PolyUOp *find_assign_store_for_after(PolyUOp *u) {
  if (!u || u->op != POLY_OP_AFTER || u->n_src < 2 || !u->src[0]) return NULL;
  PolyUOp *target = u->src[0];
  for (int i = 1; i < u->n_src; i++) {
    PolyUOp *store = u->src[i];
    if (store && store->op == POLY_OP_STORE && store->n_src >= 2 && store->src[0] == target)
      return store;
  }
  return NULL;
}

static bool assign_chain_contains_version(PolyUOp *root, PolyUOp *version) {
  for (PolyUOp *u = root; find_assign_store_for_after(u); u = u->src[0])
    if (u == version) return true;
  return false;
}

/* tinygrad's .to() puts COPY in the Tensor UOp before assign creates any
 * AFTER versions.  Polygrad retains the portable versions in uop_logical and
 * projects that COPY at the PLACE boundary, so an earlier version can be
 * reached before the exact current PLACE row.  Resolve only exact members of
 * a live PLACE's assignment chain; sharing a terminal BUFFER is insufficient
 * because the source VALUE intentionally shares that provenance. */
static PolyTensor *find_place_owner_for_assign_version(
    PolyCtx *ctx,
    PolyUOp *version,
    PolyDevice device
) {
  if (!ctx || !version || version->op != POLY_OP_AFTER) return NULL;
  PolyTensor *best = NULL;
  for (int i = 0; i < ctx->n_tensors; i++) {
    PolyTensor *tensor = ctx->tensors[i];
    if (!tensor || tensor->role != POLY_TENSOR_PLACE) continue;
    /* Assignment-chain ownership is placement identity, not allocator
     * compatibility: CPU, INTERP, and x86 can share storage while retaining
     * distinct explicit COPY destinations. */
    if (device != POLY_DEVICE_AUTO && tensor->device != device) continue;
    if (!assign_chain_contains_version(tensor->uop_logical, version) &&
        !assign_chain_contains_version(tensor->uop_physical, version))
      continue;
    if (!best || tensor->order > best->order) best = tensor;
  }
  return best;
}

PolyDevice poly_tensor_resolved_device(PolyCtx *ctx, PolyTensor *tensor) {
  PolyDevice device = tensor ? tensor->device : POLY_DEVICE_AUTO;
  if (device == POLY_DEVICE_AUTO) device = poly_ctx_get_preferred_device(ctx);
  if (device == POLY_DEVICE_AUTO) device = poly_device_default();
  return device;
}

static PolyUOp *lower_tensor(PolyPhysicalizer *p, PolyTensor *tensor);

static PolyUOp *lower_place_source(PolyPhysicalizer *p, PolyTensor *place) {
  if (!p || !place) return NULL;
  if (place->source && place->source != place) {
    p->suppress_place_facts++;
    PolyUOp *ret = lower_tensor(p, place->source);
    p->suppress_place_facts--;
    return ret;
  }

  PolyDevice source_device = poly_ctx_get_preferred_device(p->ctx);
  if (source_device == POLY_DEVICE_AUTO) source_device = poly_device_default();
  return lower_value(p, place->uop_logical, source_device);
}

static PolyUOp *lower_place(PolyPhysicalizer *p, PolyTensor *place) {
  if (!p || !place) return NULL;
  PolyDevice target_device = poly_tensor_resolved_device(p->ctx, place);

  if (!place->source || place->source == place) {
    PolyTensor *prev_active = p->active_place;
    p->active_place = place;
    PolyUOp *root = poly_tensor_uop(place);
    PolyUOp *base = lower_value(p, root, target_device);
    p->active_place = prev_active;
    if (!base) return NULL;
    PolyUOp *target = ensure_on_device(p->ctx, base, target_device);
    if (!target || !memo_put(p, place->uop_logical, target_device, target)) return NULL;
    return target;
  }

  PolyTensor *prev_active = p->active_place;
  p->active_place = place;

  PolyUOp *target = lower_place_source(p, place);
  if (!target) {
    p->active_place = prev_active;
    return NULL;
  }
  target = copy_to_device(p->ctx, target, target_device);
  if (!target) {
    p->active_place = prev_active;
    return NULL;
  }

  /* A live realize-map substitution may have installed a current physical
   * alias for dependencies inside this pending PLACE assignment. Project the
   * complete assignment-version chain onto the placement-owned target COPY,
   * then lower that chain as one graph. Rebuilding only the outer STORE leaves
   * nested STOREs writing the portable source BUFFER instead of the placed
   * value, which can mutate imported host data and replay an old version. */
  PolyUOp *effect_root = poly_tensor_uop(place);
  PolyUOp *store = find_assign_store_for_after(effect_root);
  if (!store && effect_root != place->uop_logical) {
    effect_root = place->uop_logical;
    store = find_assign_store_for_after(place->uop_logical);
  }
  if (store) {
    PolyUOp *effect_base = effect_root;
    while (find_assign_store_for_after(effect_base)) effect_base = effect_base->src[0];
    /* Seed the existing physicalizer memo instead of graph-substituting the
     * base with COPY(base, device). A recursive substitution would enter that
     * self-containing COPY and reject the cycle; the memo is the placement
     * pass's canonical logical->physical relation. */
    if (!memo_put(p, effect_base, target_device, target)) {
      p->active_place = prev_active;
      return NULL;
    }
    target = lower_value(p, effect_root, target_device);
    if (!target) {
      p->active_place = prev_active;
      return NULL;
    }
  }

  p->active_place = prev_active;
  if (!memo_put(p, place->uop_logical, target_device, target)) return NULL;
  return target;
}

static PolyUOp *lower_tensor(PolyPhysicalizer *p, PolyTensor *tensor) {
  if (!p || !tensor) return NULL;
  PolyDevice device = poly_tensor_resolved_device(p->ctx, tensor);
  if (tensor->role == POLY_TENSOR_PLACE) return lower_place(p, tensor);
  PolyUOp *root = poly_tensor_uop(tensor);
  PolyUOp *base = lower_contiguous_realized_view(p, root, device);
  if (!base) base = lower_value(p, root, device);
  if (!base) return NULL;
  return ensure_on_device(p->ctx, base, device);
}

static PolyUOp *lower_value(PolyPhysicalizer *p, PolyUOp *u, PolyDevice device) {
  if (!p || !u) return NULL;
  if (device == POLY_DEVICE_AUTO) return u;

  PolyUOp *found = memo_get(p, u, device);
  if (found) return found;

  PolyUOp *result = NULL;
  PolyTensor *place_fact = p->suppress_place_facts
                                ? NULL
                                : poly_tensor_find_current(p->ctx, u, device, POLY_TENSOR_PLACE);
  if (place_fact && place_fact != p->active_place) {
    /* A newly-created PLACE can inherit its source's current physical root;
     * that root still needs the requested placement. A different current
     * physical root installed by callify is already tinygrad's mapped live
     * Tensor value and must not replay the retained logical assignment graph. */
    PolyUOp *source_current = place_fact->source ? poly_tensor_uop(place_fact->source) : NULL;
    if (place_fact->uop_physical == u && u != source_current) {
      result = u;
    } else {
      result = lower_place(p, place_fact);
      if (!result) return NULL;
    }
  }

  PolyTensor *fact = NULL;
  if (!result) fact = poly_tensor_find_current(p->ctx, u, device, POLY_TENSOR_VALUE);
  if (fact && fact->uop_physical && fact->uop_physical != u) {
    result = lower_value(p, fact->uop_physical, device);
    if (!result) return NULL;
  }

  if (!result && (u->op == POLY_OP_BUFFER || u->op == POLY_OP_PARAM)) {
    PolyDevice declared = poly_uop_device(u);
    PolyBuffer *buf = poly_buffer_get(p->ctx, u);
    if ((declared != POLY_DEVICE_AUTO && placement_devices_share_storage(declared, device)) ||
        (buf && buf->ptr && buf->device != POLY_DEVICE_AUTO &&
         placement_devices_share_storage(buf->device, device)))
      result = u;
    else
      result = copy_to_device(p->ctx, u, device);
  } else if (!result && (!placement_rebuilds_sources(u) || u->n_src == 0)) {
    result = u;
  } else if (!result && (u->op == POLY_OP_COPY || u->op == POLY_OP_DEVICE)) {
    result = u;
  } else if (!result && u->op == POLY_OP_AFTER) {
    PolyUOp *assign_store = find_assign_store_for_after(u);
    if (assign_store) {
      /* A saved inner version is no longer an exact tensors_by_uop key after
       * its PLACE advances to a later assignment.  Before preserving an
       * ordinary BUFFER destination, establish the owning PLACE's existing
       * base->COPY memo.  During that recursive lowering active_place prevents
       * re-entry, and every version retains the same two-source recurrence. */
      if (!find_assign_store_for_after(u->src[0]) && !p->suppress_place_facts) {
        PolyTensor *owner = find_place_owner_for_assign_version(p->ctx, u, device);
        if (owner && owner != p->active_place) {
          if (!lower_place(p, owner)) return NULL;
          result = memo_get(p, u, device);
          if (!result) return NULL;
        }
      }
    }
    if (assign_store && !result) {
      PolyUOp *new_target = lower_assign_target(p, u->src[0], device);
      PolyUOp *value = lower_value(p, assign_store->src[1], device);
      PolyUOp *store_stack[16];
      PolyUOp **store_src = assign_store->n_src > 16
                                ? malloc((size_t)assign_store->n_src * sizeof(*store_src))
                                : store_stack;
      if (!new_target || !value || !store_src) {
        if (store_src && store_src != store_stack) free(store_src);
        return NULL;
      }
      memcpy(store_src, assign_store->src, (size_t)assign_store->n_src * sizeof(*store_src));
      store_src[0] = new_target;
      store_src[1] = value;
      PolyUOp *new_store = placement_rebuild_with_sources(p, assign_store, store_src);
      if (store_src != store_stack) free(store_src);
      if (!new_store) return NULL;
      if (new_target != u->src[0] || new_store != assign_store) {
        PolyUOp *after_stack[16];
        PolyUOp **after_src = u->n_src > 16
                                  ? malloc((size_t)u->n_src * sizeof(*after_src))
                                  : after_stack;
        if (!after_src) return NULL;
        memcpy(after_src, u->src, (size_t)u->n_src * sizeof(*after_src));
        after_src[0] = new_target;
        for (int i = 1; i < u->n_src; i++)
          if (after_src[i] == assign_store) after_src[i] = new_store;
        result = placement_rebuild_with_sources(p, u, after_src);
        if (after_src != after_stack) free(after_src);
      } else {
        result = u;
      }
    }
  }

  if (!result && u->op == POLY_OP_AFTER) {
    PolyUOp *stack_src[16];
    PolyUOp **src = (u->n_src > 16) ? malloc((size_t)u->n_src * sizeof(PolyUOp *)) : stack_src;
    if (!src) return NULL;
    bool changed = false;
    if (poly_uop_has_buffer_identity(u->src[0])) {
      /* AFTER's value source is the effect target. Preserve buffer identity so
       * call/store scheduling can see the destination, and let realization
       * allocate or copy dependencies explicitly. */
      src[0] = u->src[0];
    } else {
      src[0] = lower_value(p, u->src[0], device);
      if (!src[0]) {
        if (src != stack_src) free(src);
        return NULL;
      }
      if (src[0] != u->src[0]) changed = true;
    }
    for (int i = 1; i < u->n_src; i++) {
      src[i] = lower_effect(p, u->src[i], device);
      if (!src[i]) {
        if (src != stack_src) free(src);
        return NULL;
      }
      if (src[i] != u->src[i]) changed = true;
    }
    result = changed ? placement_rebuild_with_sources(p, u, src) : u;
    if (src != stack_src) free(src);
  }

  if (!result && placement_opaque_body_op(u->op)) {
    if (u->n_src == 0) {
      result = u;
    } else {
      PolyUOp *stack_src[16];
      PolyUOp **src =
          (u->n_src > 16) ? malloc((size_t)u->n_src * sizeof(PolyUOp *)) : stack_src;
      if (!src) return NULL;
      /* Pinned RewriteContext pins CALL/FUNCTION.src[0] by exact identity and
       * visits src[1:] normally. Placement is Polygrad's boundary adaptation
       * of that deviceful graph, so it must preserve the same split. */
      src[0] = u->src[0];
      bool changed = false;
      for (int i = 1; i < u->n_src; i++) {
        src[i] = lower_value(p, u->src[i], device);
        if (!src[i]) {
          if (src != stack_src) free(src);
          return NULL;
        }
        if (src[i] != u->src[i]) changed = true;
      }
      result = changed ? placement_rebuild_with_sources(p, u, src) : u;
      if (src != stack_src) free(src);
    }
  }

  if (!result) {
    PolyUOp *stack_src[16];
    PolyUOp **src = (u->n_src > 16) ? malloc((size_t)u->n_src * sizeof(PolyUOp *)) : stack_src;
    if (!src) return NULL;
    bool changed = false;
    for (int i = 0; i < u->n_src; i++) {
      src[i] = lower_value(p, u->src[i], device);
      if (!src[i]) {
        if (src != stack_src) free(src);
        return NULL;
      }
      if (src[i] != u->src[i]) changed = true;
    }
    result = changed ? placement_rebuild_with_sources(p, u, src) : u;
    if (src != stack_src) free(src);
  }

  if (!memo_put(p, u, device, result)) return NULL;
  return result;
}

static PolyUOp *lower_effect(PolyPhysicalizer *p, PolyUOp *u, PolyDevice device) {
  if (!p || !u) return NULL;
  if (device == POLY_DEVICE_AUTO) return u;

  if (u->op == POLY_OP_STORE && u->n_src >= 2) {
    PolyUOp *value = lower_value(p, u->src[1], device);
    if (!value) return NULL;
    if (value == u->src[1]) return u;
    PolyUOp *stack_src[16];
    PolyUOp **src = u->n_src > 16
                        ? malloc((size_t)u->n_src * sizeof(*src))
                        : stack_src;
    if (!src) return NULL;
    memcpy(src, u->src, (size_t)u->n_src * sizeof(*src));
    src[1] = value;
    PolyUOp *result = placement_rebuild_with_sources(p, u, src);
    if (src != stack_src) free(src);
    return result;
  }

  if (u->op == POLY_OP_ASSIGN && u->n_src >= 2) {
    PolyUOp *value = lower_value(p, u->src[1], device);
    if (!value) return NULL;
    if (value == u->src[1]) return u;
    PolyUOp *stack_src[16];
    PolyUOp **src = u->n_src > 16
                        ? malloc((size_t)u->n_src * sizeof(*src))
                        : stack_src;
    if (!src) return NULL;
    memcpy(src, u->src, (size_t)u->n_src * sizeof(*src));
    src[1] = value;
    PolyUOp *result = placement_rebuild_with_sources(p, u, src);
    if (src != stack_src) free(src);
    return result;
  }

  return lower_value(p, u, device);
}

PolyUOp *poly_tensor_physicalize(PolyCtx *ctx, PolyTensor *tensor) {
  if (!ctx || !tensor || !tensor->uop_logical) return NULL;

  PolyPhysicalizer p = {.ctx = ctx};
  PolyUOp *physical = lower_tensor(&p, tensor);
  physicalizer_destroy(&p);
  return physical;
}

int poly_tensor_physicalize_many(
    PolyCtx *ctx,
    PolyTensor **tensors,
    int n,
    PolyUOp **out,
    PolyMap **placement_memo
) {
  if (!ctx || n < 0 || (n > 0 && (!tensors || !out))) return -1;
  PolyPhysicalizer p = {.ctx = ctx};
  if (placement_memo) {
    for (int device = POLY_DEVICE_AUTO + 1; device <= POLY_DEVICE_DISK; device++)
      p.memo[device] = placement_memo[device];
  }
  int rc = 0;
  for (int i = 0; i < n; i++) {
    out[i] = (tensors[i] && tensors[i]->uop_logical)
                 ? lower_tensor(&p, tensors[i])
                 : NULL;
    if (!out[i]) {
      rc = -1;
      break;
    }
  }
  if (placement_memo) {
    for (int device = POLY_DEVICE_AUTO + 1; device <= POLY_DEVICE_DISK; device++)
      placement_memo[device] = p.memo[device];
  } else {
    physicalizer_destroy(&p);
  }
  return rc;
}

void poly_tensor_physicalize_memo_destroy(PolyMap **placement_memo) {
  if (!placement_memo) return;
  for (int device = POLY_DEVICE_AUTO + 1; device <= POLY_DEVICE_DISK; device++) {
    poly_map_destroy(placement_memo[device]);
    placement_memo[device] = NULL;
  }
}

int poly_tensor_placement_audit(
    PolyCtx *ctx,
    PolyTensor *selected,
    PolyUOp *query_current,
    PolyDevice query_device,
    PolyPlacementAudit *out
) {
  if (!ctx || !selected || !out) return -1;
  memset(out, 0, sizeof(*out));

  out->selected = selected;
  out->selected_current = poly_tensor_uop(selected);
  out->selected_logical = poly_tensor_uop_logical(selected);
  out->selected_physical = poly_tensor_uop_physical(selected);
  out->selected_role = selected->role;
  out->selected_device = poly_tensor_resolved_device(ctx, selected);
  out->selected_source = selected->source;

  out->query_current = query_current ? query_current : out->selected_current;
  out->query_device = (query_device == POLY_DEVICE_AUTO) ? out->selected_device : query_device;
  if (!out->query_current || out->query_device == POLY_DEVICE_AUTO) return -1;

  out->place_fact =
      poly_tensor_find_current(ctx, out->query_current, out->query_device, POLY_TENSOR_PLACE);
  out->value_fact =
      poly_tensor_find_current(ctx, out->query_current, out->query_device, POLY_TENSOR_VALUE);

  /* Match lower_value's fact priority: an active PLACE fact is preferred over
   * a VALUE fact, and VALUE facts only affect lowering when they carry a
   * physical/current root distinct from the queried logical root. */
  if (out->place_fact && out->place_fact != selected) {
    out->matched_fact = out->place_fact;
    out->matched_role = POLY_TENSOR_PLACE;
  } else if (out->value_fact && out->value_fact->uop_physical &&
             out->value_fact->uop_physical != out->query_current) {
    out->matched_fact = out->value_fact;
    out->matched_role = POLY_TENSOR_VALUE;
  } else {
    out->matched_role = (PolyTensorRole)-1;
  }

  out->physical_root = poly_tensor_physicalize(ctx, selected);
  return out->physical_root ? 0 : -1;
}
