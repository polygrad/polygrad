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
    if (cache)
      poly_map_set(cache, poly_ptr_hash(u), u, (void *)(intptr_t)(result + 1), poly_ptr_eq);
    return result;
  }
  if (u->op == POLY_OP_COPY && u->n_src >= 2) {
    result = poly_device_from_device_uop(u->src[1]);
    if (cache)
      poly_map_set(cache, poly_ptr_hash(u), u, (void *)(intptr_t)(result + 1), poly_ptr_eq);
    return result;
  }
  if (u->op == POLY_OP_BUFFER && u->n_src >= 2) {
    result = poly_device_from_device_uop(u->src[1]);
    if (cache)
      poly_map_set(cache, poly_ptr_hash(u), u, (void *)(intptr_t)(result + 1), poly_ptr_eq);
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
    if (cache)
      poly_map_set(cache, poly_ptr_hash(u), u, (void *)(intptr_t)(result + 1), poly_ptr_eq);
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
  return poly_device_uop(ctx, device);
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
  if (a == POLY_DEVICE_HOST || b == POLY_DEVICE_HOST || a == POLY_DEVICE_DISK ||
      b == POLY_DEVICE_DISK)
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
  if (!p || !logical || device <= POLY_DEVICE_AUTO || device > POLY_DEVICE_DISK || !p->memo[device])
    return NULL;
  return poly_map_get(p->memo[device], poly_ptr_hash(logical), logical, poly_ptr_eq);
}

static bool memo_put(PolyPhysicalizer *p, PolyUOp *logical, PolyDevice device, PolyUOp *physical) {
  if (!p || !logical || !physical || device <= POLY_DEVICE_AUTO || device > POLY_DEVICE_DISK)
    return false;
  if (!p->memo[device]) p->memo[device] = poly_map_new(128);
  poly_map_set(p->memo[device], poly_ptr_hash(logical), logical, physical, poly_ptr_eq);
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
static PolyUOp *placement_rebuild_with_sources(PolyPhysicalizer *p, PolyUOp *u, PolyUOp **src) {
  if (!p || !p->ctx || !u || (u->n_src > 0 && !src)) return NULL;
  return (u->tag != 0 || u->tag_arg.kind != POLY_ARG_NONE)
             ? poly_uop_tagged_arg(
                   p->ctx, u->op, u->dtype, src, u->n_src, u->arg, u->tag, u->tag_arg
               )
             : poly_uop(p->ctx, u->op, u->dtype, src, u->n_src, u->arg);
}

/* An assignment destination is an existing storage identity, not an ordinary
 * value input that placement may replace with COPY(storage, device).  A PLACE
 * assignment seeds the exact destination mapping before reaching this helper;
 * otherwise retain a buffer-identity destination and lower only the value
 * being written.  Nested AFTER versions recurse through lower_value so their
 * dependency chain remains intact. */
static PolyUOp *lower_assign_target(PolyPhysicalizer *p, PolyUOp *target, PolyDevice device) {
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
static PolyUOp *lower_contiguous_realized_view(PolyPhysicalizer *p, PolyUOp *u, PolyDevice device) {
  if (!p || !u) return NULL;
  PolyUOp *identity = NULL;
  PolyShape shape = {.ndim = -1};
  int64_t numel = -1;
  size_t byte_offset = 0;
  if (!poly_uop_contiguous_view_info(p->ctx, u, &identity, &shape, &numel, &byte_offset))
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

PolyDevice poly_tensor_resolved_device(PolyCtx *ctx, PolyTensor *tensor) {
  PolyDevice device = tensor ? tensor->device : POLY_DEVICE_AUTO;
  if (device == POLY_DEVICE_AUTO) device = poly_ctx_get_preferred_device(ctx);
  if (device == POLY_DEVICE_AUTO) device = poly_device_default();
  return device;
}

static PolyUOp *lower_tensor(PolyPhysicalizer *p, PolyTensor *tensor);

static PolyUOp *lower_place_source(PolyPhysicalizer *p, PolyTensor *place) {
  if (!p || !place) return NULL;
  if (place->source && place->source != place) return lower_tensor(p, place->source);

  PolyDevice source_device = poly_ctx_get_preferred_device(p->ctx);
  if (source_device == POLY_DEVICE_AUTO) source_device = poly_device_default();
  return lower_value(p, place->uop_logical, source_device);
}

static PolyUOp *lower_place(PolyPhysicalizer *p, PolyTensor *place) {
  if (!p || !place) return NULL;
  PolyDevice target_device = poly_tensor_resolved_device(p->ctx, place);

  if (!place->source || place->source == place) {
    /* Explicit re-placement starts from the retained portable graph. Default
     * execution never calls this path. */
    PolyUOp *root = place->uop_logical;
    PolyUOp *base = lower_value(p, root, target_device);
    if (!base) return NULL;
    PolyUOp *target = ensure_on_device(p->ctx, base, target_device);
    if (!target || !memo_put(p, place->uop_logical, target_device, target)) return NULL;
    return target;
  }

  PolyUOp *target = lower_place_source(p, place);
  if (!target) return NULL;
  target = copy_to_device(p->ctx, target, target_device);
  if (!target) return NULL;

  /* Project the retained logical assignment-version chain onto the explicit
   * placement-owned target COPY. Realization never feeds a physical alias
   * back into this compiler. */
  PolyUOp *effect_root = place->uop_logical;
  PolyUOp *store = find_assign_store_for_after(effect_root);
  if (store) {
    PolyUOp *effect_base = effect_root;
    while (find_assign_store_for_after(effect_base))
      effect_base = effect_base->src[0];
    /* Seed the existing physicalizer memo instead of graph-substituting the
     * base with COPY(base, device). A recursive substitution would enter that
     * self-containing COPY and reject the cycle; the memo is the placement
     * pass's canonical logical->physical relation. */
    if (!memo_put(p, effect_base, target_device, target)) {
      return NULL;
    }
    target = lower_value(p, effect_root, target_device);
    if (!target) {
      return NULL;
    }
  }

  if (!memo_put(p, place->uop_logical, target_device, target)) return NULL;
  return target;
}

static PolyUOp *lower_tensor(PolyPhysicalizer *p, PolyTensor *tensor) {
  if (!p || !tensor) return NULL;
  PolyDevice device = poly_tensor_resolved_device(p->ctx, tensor);
  if (tensor->role == POLY_TENSOR_PLACE) return lower_place(p, tensor);
  /* This function is the explicit logical -> physical compiler. Tensor ops
   * and default realization consume uop_physical directly instead. */
  PolyUOp *root = tensor->uop_logical;
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

  if (!result && (u->op == POLY_OP_BUFFER || u->op == POLY_OP_PARAM)) {
    PolyDevice declared = poly_uop_device(u);
    if (declared != POLY_DEVICE_AUTO && placement_devices_share_storage(declared, device))
      result = u;
    else
      /* Runtime residency does not complete a portable BUFFER's physical IR.
       * Pinned _frompy keeps an explicit source BUFFER and target DEVICE COPY
       * (uop/ops.py:747-765); explicit re-placement must likewise publish a
       * device-bearing occurrence. */
      result = copy_to_device(p->ctx, u, device);
  } else if (!result && (!placement_rebuilds_sources(u) || u->n_src == 0)) {
    result = u;
  } else if (!result && (u->op == POLY_OP_COPY || u->op == POLY_OP_DEVICE)) {
    result = u;
  } else if (!result && u->op == POLY_OP_AFTER) {
    PolyUOp *assign_store = find_assign_store_for_after(u);
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
        PolyUOp **after_src =
            u->n_src > 16 ? malloc((size_t)u->n_src * sizeof(*after_src)) : after_stack;
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
      PolyUOp **src = (u->n_src > 16) ? malloc((size_t)u->n_src * sizeof(PolyUOp *)) : stack_src;
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
    PolyUOp **src = u->n_src > 16 ? malloc((size_t)u->n_src * sizeof(*src)) : stack_src;
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
    PolyUOp **src = u->n_src > 16 ? malloc((size_t)u->n_src * sizeof(*src)) : stack_src;
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

int poly_tensor_physicalize_many(PolyCtx *ctx, PolyTensor **tensors, int n, PolyUOp **out) {
  if (!ctx || n < 0 || (n > 0 && (!tensors || !out))) return -1;
  PolyPhysicalizer p = {.ctx = ctx};
  int rc = 0;
  for (int i = 0; i < n; i++) {
    out[i] = (tensors[i] && tensors[i]->uop_logical) ? lower_tensor(&p, tensors[i]) : NULL;
    if (!out[i]) {
      rc = -1;
      break;
    }
  }
  physicalizer_destroy(&p);
  return rc;
}

static bool place_exact_buffer_identity(PolyUOp *u) {
  return u && poly_uop_get_buffer_identity(u) == u;
}

static bool place_direct_buffer_binding(PolyUOp *u) {
  return place_exact_buffer_identity(u) && u->op == POLY_OP_BUFFER;
}

static int place_binding_index(PolyUOp **bindings, int n, PolyUOp *u) {
  if (!bindings || !u) return -1;
  for (int i = 0; i < n; i++)
    if (bindings[i] == u) return i;
  return -1;
}

static bool place_binding_shape_eq(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  if (!ctx || !a || !b || !poly_dtype_eq(a->dtype, b->dtype)) return false;
  PolyShape as = poly_uop_max_shape_cached(ctx, a);
  PolyShape bs = poly_uop_max_shape_cached(ctx, b);
  return as.ndim >= 0 && bs.ndim >= 0 && poly_shape_eq(as, bs);
}

static bool place_lowered_op(PolyOps op) {
  return op == POLY_OP_LINEAR || op == POLY_OP_PROGRAM || op == POLY_OP_SOURCE ||
         op == POLY_OP_BINARY;
}

static bool place_logical_forbidden_op(PolyOps op) {
  return op == POLY_OP_SINK || op == POLY_OP_STORE || op == POLY_OP_AFTER || op == POLY_OP_ASSIGN ||
         op == POLY_OP_DEVICE || op == POLY_OP_COPY || op == POLY_OP_CALL ||
         op == POLY_OP_FUNCTION || op == POLY_OP_CUSTOM_FUNCTION || op == POLY_OP_MULTI ||
         op == POLY_OP_MSELECT || op == POLY_OP_MSTACK || op == POLY_OP_ALLREDUCE ||
         place_lowered_op(op);
}

static bool place_validate_pure_logical(
    PolyCtx *ctx,
    PolyUOp *root,
    PolyUOp **logical_bindings,
    int n_bindings
) {
  if (!ctx || !root) return false;
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_ex_alloc(ctx, root, &n_topo, NULL, false);
  if (!topo) return false;
  bool valid = true;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (!u || place_logical_forbidden_op(u->op)) {
      valid = false;
      break;
    }
    if (place_exact_buffer_identity(u) &&
        place_binding_index(logical_bindings, n_bindings, u) < 0) {
      valid = false;
      break;
    }
  }
  free(topo);
  return valid;
}

static bool place_validate_logical_root(
    PolyCtx *ctx,
    PolyUOp *root,
    PolyUOp **logical_bindings,
    int n_bindings
) {
  if (!ctx || !root) return false;
  if (root->op != POLY_OP_SINK)
    return place_validate_pure_logical(ctx, root, logical_bindings, n_bindings);
  if (root->n_src == 0) return false;
  for (int i = 0; i < root->n_src; i++) {
    PolyUOp *store = root->src[i];
    if (!store || store->op != POLY_OP_STORE || store->n_src != 2 ||
        !place_exact_buffer_identity(store->src[0]) ||
        place_binding_index(logical_bindings, n_bindings, store->src[0]) < 0 ||
        !place_validate_pure_logical(ctx, store->src[1], logical_bindings, n_bindings))
      return false;
  }
  return true;
}

static bool place_validate_physical_root(PolyCtx *ctx, PolyUOp *root) {
  if (!ctx || !root) return false;
  int n_topo = 0;
  /* Pinned RewriteContext keeps CALL/FUNCTION bodies opaque by default.  Their
   * placeholders are not caller-visible storage bindings. */
  PolyUOp **topo = poly_toposort_ex_alloc(ctx, root, &n_topo, NULL, false);
  if (!topo) return false;
  bool valid = true;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (!u || place_lowered_op(u->op) ||
        (place_exact_buffer_identity(u) && poly_uop_device(u) == POLY_DEVICE_AUTO)) {
      valid = false;
      break;
    }
  }
  free(topo);
  return valid;
}

int poly_place_roots(
    PolyCtx *ctx,
    PolyUOp **logical_roots,
    PolyUOp **physical_templates,
    int n_roots,
    PolyUOp **logical_bindings,
    PolyUOp **template_bindings,
    PolyUOp **target_bindings,
    int n_bindings,
    PolyUOp **out_roots
) {
  if (!ctx || n_roots < 0 || n_bindings < 0 || (n_roots > 0 && (!logical_roots || !out_roots)) ||
      (n_bindings > 0 && (!logical_bindings || !target_bindings)) ||
      (physical_templates && n_bindings > 0 && !template_bindings))
    return -1;
  if (n_roots == 0) return 0;

  PolyUOp **source_roots = physical_templates ? physical_templates : logical_roots;
  PolyUOp **source_bindings = physical_templates ? template_bindings : logical_bindings;
  for (int i = 0; i < n_roots; i++) {
    if (!logical_roots[i] || !source_roots[i] || !poly_ctx_owns_ptr(ctx, logical_roots[i]) ||
        !poly_ctx_owns_ptr(ctx, source_roots[i]))
      return -1;
    if (!physical_templates &&
        !place_validate_logical_root(ctx, logical_roots[i], logical_bindings, n_bindings))
      return -1;
  }

  for (int i = 0; i < n_bindings; i++) {
    PolyUOp *logical = logical_bindings[i];
    PolyUOp *source = source_bindings[i];
    PolyUOp *target = target_bindings[i];
    /* Pinned `.to()` is an explicit COPY occurrence (tensor.py:327-335).
     * COPY/BUFFER_VIEW/PARAM may remain nested in a physical template, but
     * accepting one as the replaceable binding would erase that occurrence. */
    if (!place_direct_buffer_binding(logical) || !place_direct_buffer_binding(source) ||
        !place_direct_buffer_binding(target) || !poly_ctx_owns_ptr(ctx, logical) ||
        !poly_ctx_owns_ptr(ctx, source) || !poly_ctx_owns_ptr(ctx, target) ||
        poly_uop_device(target) == POLY_DEVICE_AUTO ||
        !place_binding_shape_eq(ctx, logical, target) ||
        !place_binding_shape_eq(ctx, source, target))
      return -1;
    for (int j = 0; j < i; j++) {
      if ((logical_bindings[j] == logical && target_bindings[j] != target) ||
          (source_bindings[j] == source && target_bindings[j] != target) ||
          ((logical_bindings[j] != logical || source_bindings[j] != source) &&
           target_bindings[j] == target))
        return -1;
    }
  }

  PolyUOp **candidates = calloc((size_t)n_roots, sizeof(*candidates));
  if (!candidates) return -1;
  int rc = poly_uop_substitute_many(
      ctx, source_roots, n_roots, source_bindings, target_bindings, n_bindings, candidates
  );
  for (int i = 0; rc == 0 && i < n_roots; i++)
    if (!place_validate_physical_root(ctx, candidates[i])) rc = -1;
  if (rc == 0) memcpy(out_roots, candidates, (size_t)n_roots * sizeof(*out_roots));
  free(candidates);
  return rc;
}
