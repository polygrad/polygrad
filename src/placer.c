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
  PolyUOp *logical;
  PolyDevice device;
  PolyUOp *physical;
} PolyLowerMemoEntry;

typedef struct {
  PolyCtx *ctx;
  PolyLowerMemoEntry *memo;
  int n_memo;
  int memo_cap;
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

  if (u->op != POLY_OP_CONST && u->op != POLY_OP_VCONST && u->op != POLY_OP_DEVICE) {
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
  if (a == POLY_DEVICE_HOST || b == POLY_DEVICE_HOST) return false;
  return poly_device_is_host_addressable(a) && poly_device_is_host_addressable(b);
}

static PolyUOp *ensure_on_device(PolyCtx *ctx, PolyUOp *value, PolyDevice device) {
  if (!ctx || !value || device == POLY_DEVICE_AUTO) return value;
  PolyDevice current = poly_uop_device(value);
  if (current != POLY_DEVICE_AUTO && placement_devices_share_storage(current, device)) return value;
  if (current == POLY_DEVICE_AUTO && placement_devices_share_storage(poly_device_default(), device))
    return value;
  return copy_to_device(ctx, value, device);
}

static PolyUOp *memo_get(PolyPhysicalizer *p, PolyUOp *logical, PolyDevice device) {
  for (int i = 0; i < p->n_memo; i++)
    if (p->memo[i].logical == logical && p->memo[i].device == device) return p->memo[i].physical;
  return NULL;
}

static bool memo_put(PolyPhysicalizer *p, PolyUOp *logical, PolyDevice device, PolyUOp *physical) {
  if (p->n_memo >= p->memo_cap) {
    int new_cap = p->memo_cap ? p->memo_cap * 2 : 128;
    PolyLowerMemoEntry *new_memo = realloc(p->memo, (size_t)new_cap * sizeof(PolyLowerMemoEntry));
    if (!new_memo) return false;
    p->memo = new_memo;
    p->memo_cap = new_cap;
  }
  p->memo[p->n_memo++] = (PolyLowerMemoEntry){logical, device, physical};
  return true;
}

static bool placement_rebuilds_sources(PolyUOp *u) {
  if (!u) return false;
  if (u->op == POLY_OP_CONST || u->op == POLY_OP_VCONST || u->op == POLY_OP_DEFINE_VAR ||
      u->op == POLY_OP_BIND || u->op == POLY_OP_DEVICE || u->op == POLY_OP_UNIQUE ||
      u->op == POLY_OP_LUNIQUE)
    return false;
  return true;
}

static PolyUOp *lower_value(PolyPhysicalizer *p, PolyUOp *u, PolyDevice device);
static PolyUOp *lower_effect(PolyPhysicalizer *p, PolyUOp *u, PolyDevice device);

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

static PolyDevice resolve_tensor_device(PolyCtx *ctx, PolyTensor *tensor) {
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
  PolyDevice target_device = resolve_tensor_device(p->ctx, place);

  if (!place->source || place->source == place) {
    PolyTensor *prev_active = p->active_place;
    p->active_place = place;
    PolyUOp *root = poly_tensor_uop(place);
    PolyUOp *base = lower_value(p, root, target_device);
    p->active_place = prev_active;
    if (!base) return NULL;
    return ensure_on_device(p->ctx, base, target_device);
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

  PolyUOp *store = find_assign_store_for_after(place->uop_logical);
  if (store) {
    PolyUOp *value = lower_value(p, store->src[1], target_device);
    if (!value) {
      p->active_place = prev_active;
      return NULL;
    }
    PolyUOp *new_store = poly_store_val(p->ctx, target, value);
    if (!new_store) {
      p->active_place = prev_active;
      return NULL;
    }
    PolyUOp *src[2] = {target, new_store};
    target = poly_uop(p->ctx, POLY_OP_AFTER, target->dtype, src, 2, poly_arg_none());
  }

  p->active_place = prev_active;
  return target;
}

static PolyUOp *lower_tensor(PolyPhysicalizer *p, PolyTensor *tensor) {
  if (!p || !tensor) return NULL;
  PolyDevice device = resolve_tensor_device(p->ctx, tensor);
  if (tensor->role == POLY_TENSOR_PLACE) return lower_place(p, tensor);
  PolyUOp *root = poly_tensor_uop(tensor);
  PolyUOp *base = lower_value(p, root, device);
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
    result = lower_place(p, place_fact);
    if (!result) return NULL;
  }

  PolyTensor *fact = NULL;
  if (!result) fact = poly_tensor_find_current(p->ctx, u, device, POLY_TENSOR_VALUE);
  if (fact && fact->uop_physical && fact->uop_physical != u) {
    result = lower_value(p, fact->uop_physical, device);
    if (!result) return NULL;
  }

  if (!result && (u->op == POLY_OP_BUFFER || u->op == POLY_OP_PARAM)) {
    PolyBuffer *buf = poly_buffer_get(p->ctx, u);
    if (buf && buf->ptr && buf->device != POLY_DEVICE_AUTO &&
        placement_devices_share_storage(buf->device, device))
      result = u;
    else
      result = copy_to_device(p->ctx, u, device);
  } else if (!placement_rebuilds_sources(u) || u->n_src == 0) {
    result = u;
  } else if (u->op == POLY_OP_COPY || u->op == POLY_OP_DEVICE) {
    result = u;
  } else if (u->op == POLY_OP_AFTER) {
    PolyUOp *assign_store = find_assign_store_for_after(u);
    if (assign_store) {
      PolyUOp *new_store = lower_effect(p, assign_store, device);
      if (!new_store) return NULL;
      if (new_store != assign_store) {
        PolyUOp *src[2] = {u->src[0], new_store};
        result = poly_uop(p->ctx, POLY_OP_AFTER, u->dtype, src, 2, u->arg);
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
    src[0] = lower_value(p, u->src[0], device);
    if (!src[0]) {
      if (src != stack_src) free(src);
      return NULL;
    }
    if (src[0] != u->src[0]) changed = true;
    for (int i = 1; i < u->n_src; i++) {
      src[i] = lower_effect(p, u->src[i], device);
      if (!src[i]) {
        if (src != stack_src) free(src);
        return NULL;
      }
      if (src[i] != u->src[i]) changed = true;
    }
    result = changed ? poly_uop(p->ctx, u->op, u->dtype, src, u->n_src, u->arg) : u;
    if (src != stack_src) free(src);
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
    result = changed ? poly_uop(p->ctx, u->op, u->dtype, src, u->n_src, u->arg) : u;
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
    return poly_store_val(p->ctx, u->src[0], value);
  }

  if (u->op == POLY_OP_ASSIGN && u->n_src >= 2) {
    PolyUOp *value = lower_value(p, u->src[1], device);
    if (!value) return NULL;
    if (value == u->src[1]) return u;
    return poly_uop2(p->ctx, POLY_OP_ASSIGN, u->dtype, u->src[0], value, u->arg);
  }

  return lower_value(p, u, device);
}

PolyUOp *poly_tensor_physicalize(PolyCtx *ctx, PolyTensor *tensor) {
  if (!ctx || !tensor || !tensor->uop_logical) return NULL;

  PolyPhysicalizer p = {.ctx = ctx};
  PolyUOp *physical = lower_tensor(&p, tensor);
  free(p.memo);
  return physical;
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
  out->selected_device = resolve_tensor_device(ctx, selected);
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
