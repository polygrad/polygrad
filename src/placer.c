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

#include <ctype.h>
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
  PolyDevice direct = poly_device_by_name(s);
  if (direct != POLY_DEVICE_AUTO) return direct;

  /* Exact identity and backend implementation are separate in pinned
   * tinygrad: Device["CPU:1"] is distinct, while its implementation class is
   * still CPUDevice (device.py:15-35; runtime/ops_cpu.py:122-142).  Derive the
   * backend from a numeric ordinal suffix without canonicalizing the identity
   * itself.  Admission remains a separate check below. */
  const char *sep = strchr(s, ':');
  if (!sep || sep == s || !sep[1]) return POLY_DEVICE_AUTO;
  const char *ordinal = sep + 1;
  if (*ordinal == '+' || *ordinal == '-') ordinal++;
  if (!*ordinal) return POLY_DEVICE_AUTO;
  for (const char *p = ordinal; *p; p++)
    if (!isdigit((unsigned char)*p)) return POLY_DEVICE_AUTO;

  size_t prefix_n = (size_t)(sep - s);
#define DEVICE_PREFIX(name, kind)                                                                  \
  if (prefix_n == sizeof(name) - 1 && strncmp(s, name, prefix_n) == 0) return kind
  DEVICE_PREFIX("CPU", POLY_DEVICE_CPU);
  DEVICE_PREFIX("CUDA", POLY_DEVICE_CUDA);
  DEVICE_PREFIX("HIP", POLY_DEVICE_HIP);
  DEVICE_PREFIX("WEBGPU", POLY_DEVICE_WEBGPU);
  DEVICE_PREFIX("INTERP", POLY_DEVICE_INTERP);
  DEVICE_PREFIX("WASM", POLY_DEVICE_WASM);
  DEVICE_PREFIX("X86", POLY_DEVICE_X86);
#undef DEVICE_PREFIX
  return POLY_DEVICE_AUTO;
}

static bool device_identity_name_supported(const char *name) {
  if (!name) return true;
  PolyDevice backend = device_from_string_arg(name);
  if (backend == POLY_DEVICE_AUTO) return false;

  /* Existing ordinal-zero and named DISK/CPU:X86 identities remain supported.
   * CPU numeric identities are safe to admit because the CPU implementation
   * is stateless per execution and all identity-bearing state is keyed by the
   * exact DEVICE UOp.  Accelerator ordinals stay fail-closed until their
   * contexts/allocators are per identity. */
  if (poly_device_by_name(name) != POLY_DEVICE_AUTO) return true;
  return backend == POLY_DEVICE_CPU;
}

static bool device_uop_identity_supported(PolyUOp *device) {
  if (!device || device->op != POLY_OP_DEVICE) return false;
  if (device->arg.kind == POLY_ARG_NONE) return true;
  /* Pinned tuple DEVICE identities are ordered exact graph values which
   * schedule/multi.py lowers to scalar occurrences (uop/ops.py:770-807;
   * schedule/multi.py:8-31). Admit only tuples whose members already satisfy
   * Polygrad's scalar backend-identity policy. */
  if (device->arg.kind == POLY_ARG_STRING_TUPLE) {
    if (device->arg.string_tuple.n <= 0 || !device->arg.string_tuple.vals) return false;
    for (int i = 0; i < device->arg.string_tuple.n; i++)
      if (!device_identity_name_supported(device->arg.string_tuple.vals[i])) return false;
    return true;
  }
  if (device->arg.kind != POLY_ARG_STRING) return false;
  return device_identity_name_supported(device->arg.str);
}

static char no_device_uop_cache_value;

/* Exact-device counterpart of tinygrad@2026-08-22/a9069c177a9d
 * uop/ops.py:847-861. DEVICE transports explicit placement policy in C;
 * PARAM/STAGE/BUFFER use ParamArg, COPY uses arg, AFTER follows its value,
 * and generic UOps take the first concrete source. */
PolyUOp *poly_uop_device_uop_cached(PolyCtx *ctx, PolyUOp *u, PolyMap *cache) {
  if (!ctx || !u) return NULL;
  if (cache) {
    void *cached = poly_map_get(cache, poly_ptr_hash(u), u, poly_ptr_eq);
    if (cached) return cached == &no_device_uop_cache_value ? NULL : cached;
  }

  PolyUOp *result = NULL;
  if (u->op == POLY_OP_DEVICE) {
    if (u->arg.kind == POLY_ARG_STRING || u->arg.kind == POLY_ARG_STRING_TUPLE) result = u;
  } else if ((u->op == POLY_OP_PARAM || u->op == POLY_OP_BUFFER) &&
             u->arg.kind == POLY_ARG_PARAM && u->arg.param) {
    if (u->arg.param->device_is_tuple)
      result = poly_device_uop_from_names(
          ctx, u->arg.param->devices, u->arg.param->n_devices
      );
    else if (u->arg.param->device)
      result = poly_device_uop_from_name(ctx, u->arg.param->device);
  } else if (u->op == POLY_OP_STAGE && u->arg.kind == POLY_ARG_BUFFERIZE_OPTS) {
    if (u->arg.bufferize_opts.device_is_tuple)
      result = poly_device_uop_from_names(
          ctx, u->arg.bufferize_opts.devices, u->arg.bufferize_opts.n_devices
      );
    else if (u->arg.bufferize_opts.device)
      result = poly_device_uop_from_name(ctx, u->arg.bufferize_opts.device);
  } else if (u->op == POLY_OP_COPY && u->arg.kind == POLY_ARG_STRING) {
    result = poly_device_uop_from_name(ctx, u->arg.str);
  } else if (u->op == POLY_OP_COPY && u->arg.kind == POLY_ARG_STRING_TUPLE) {
    result = poly_device_uop_from_names(ctx, u->arg.string_tuple.vals, u->arg.string_tuple.n);
  } else if (u->op == POLY_OP_ALLREDUCE && u->arg.kind == POLY_ARG_ALLREDUCE) {
    result = u->arg.allreduce.device_is_tuple
                 ? poly_device_uop_from_names(
                       ctx, u->arg.allreduce.devices, u->arg.allreduce.n_devices)
                 : poly_device_uop_from_name(ctx, u->arg.allreduce.device);
  } else if (u->op == POLY_OP_AFTER && u->n_src >= 1) {
    result = poly_uop_device_uop_cached(ctx, u->src[0], cache);
  } else if (u->op == POLY_OP_MSELECT && u->n_src >= 1 && u->arg.kind == POLY_ARG_INT) {
    PolyUOp *source_device = poly_uop_device_uop_cached(ctx, u->src[0], cache);
    if (source_device && source_device->arg.kind == POLY_ARG_STRING_TUPLE && u->arg.i >= 0 &&
        u->arg.i < source_device->arg.string_tuple.n)
      result = poly_device_uop_from_name(ctx, source_device->arg.string_tuple.vals[(int)u->arg.i]);
  } else if (u->op == POLY_OP_MSTACK) {
    const char **names = u->n_src ? malloc((size_t)u->n_src * sizeof(*names)) : NULL;
    bool complete = u->n_src == 0 || names != NULL;
    for (int i = 0; i < u->n_src && complete; i++) {
      PolyUOp *source_device = poly_uop_device_uop_cached(ctx, u->src[i], cache);
      if (!source_device || source_device->arg.kind != POLY_ARG_STRING) {
        complete = false;
        break;
      }
      names[i] = source_device->arg.str;
    }
    if (complete)
      result = poly_uop0(ctx, POLY_OP_DEVICE, POLY_VOID, poly_arg_string_tuple(names, u->n_src));
    free(names);
  } else if (u->op != POLY_OP_CONST) {
    for (int i = 0; i < u->n_src && !result; i++)
      result = poly_uop_device_uop_cached(ctx, u->src[i], cache);
  }

  if (cache)
    poly_map_set(
        cache, poly_ptr_hash(u), u, result ? (void *)result : (void *)&no_device_uop_cache_value,
        poly_ptr_eq
    );
  return result;
}

const char *poly_uop_device_name(PolyCtx *ctx, PolyUOp *u) {
  if (!ctx || !u) return NULL;
  PolyMap *cache = poly_map_new(64);
  if (!cache) return NULL;
  PolyUOp *device = poly_uop_device_uop_cached(ctx, u, cache);
  const char *name = device && device->arg.kind == POLY_ARG_STRING ? device->arg.str : NULL;
  poly_map_destroy(cache);
  return name;
}

PolyDevice poly_device_from_device_uop(PolyUOp *device) {
  if (!device || device->op != POLY_OP_DEVICE) return POLY_DEVICE_AUTO;
  if (device->arg.kind == POLY_ARG_STRING) return device_from_string_arg(device->arg.str);
  return POLY_DEVICE_AUTO;
}

/* Polygrad's approved fail-closed accelerator-identity boundary for Tinygrad's
 * exact ParamArg.device semantics (device.py:19-35; uop/ops.py:847-860). */
bool poly_uop_explicit_devices_supported(PolyCtx *ctx, PolyUOp *root) {
  if (!ctx || !root) return false;
  PolyScratchMark scratch = poly_ctx_scratch_mark(ctx);
  int n = 0;
  PolyUOp **topo =
      poly_toposort_ex_user_scratch(ctx, root, &n, NULL, NULL, false);
  if (!topo) {
    poly_ctx_scratch_rewind(ctx, scratch);
    return false;
  }
  bool supported = true;
  for (int i = 0; i < n; i++) {
    PolyUOp *u = topo[i];
    if (u && u->op == POLY_OP_DEVICE && !device_uop_identity_supported(u)) {
      supported = false;
      break;
    }
    if (u && (u->op == POLY_OP_PARAM || u->op == POLY_OP_BUFFER) &&
        u->arg.kind == POLY_ARG_PARAM && u->arg.param &&
        u->arg.param->device_is_tuple) {
      if (u->arg.param->n_devices <= 0 || !u->arg.param->devices) {
        supported = false;
        break;
      }
      for (int j = 0; j < u->arg.param->n_devices; j++) {
        if (!device_identity_name_supported(u->arg.param->devices[j])) {
          supported = false;
          break;
        }
      }
      if (!supported) break;
    } else if (u && (u->op == POLY_OP_PARAM || u->op == POLY_OP_BUFFER) &&
               u->arg.kind == POLY_ARG_PARAM && u->arg.param &&
               !device_identity_name_supported(u->arg.param->device)) {
      supported = false;
      break;
    }
    if (u && u->op == POLY_OP_STAGE && u->arg.kind == POLY_ARG_BUFFERIZE_OPTS) {
      if (u->arg.bufferize_opts.device_is_tuple) {
        if (u->arg.bufferize_opts.n_devices <= 0 || !u->arg.bufferize_opts.devices) {
          supported = false;
          break;
        }
        for (int j = 0; j < u->arg.bufferize_opts.n_devices; j++) {
          if (!device_identity_name_supported(u->arg.bufferize_opts.devices[j])) {
            supported = false;
            break;
          }
        }
        if (!supported) break;
      } else if (!device_identity_name_supported(u->arg.bufferize_opts.device)) {
        supported = false;
        break;
      }
    }
  }
  poly_ctx_scratch_rewind(ctx, scratch);
  return supported;
}

/* Derived physical-device property for a UOp graph.
 *
 * tinygrad@2026-08-22/a9069c177a9d uop/ops.py:847-861 reads COPY.arg and
 * BUFFER/ParamArg.device. Polygrad DEVICE remains C placement-policy metadata;
 * portable logical BUFFERs have no device and return AUTO.
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
  if (u->op == POLY_OP_COPY && u->arg.kind == POLY_ARG_STRING) {
    result = device_from_string_arg(u->arg.str);
    if (cache)
      poly_map_set(cache, poly_ptr_hash(u), u, (void *)(intptr_t)(result + 1), poly_ptr_eq);
    return result;
  }
  if (u->op == POLY_OP_BUFFER && u->arg.kind == POLY_ARG_PARAM && u->arg.param &&
      u->arg.param->device) {
    result = device_from_string_arg(u->arg.param->device);
    if (cache)
      poly_map_set(cache, poly_ptr_hash(u), u, (void *)(intptr_t)(result + 1), poly_ptr_eq);
    return result;
  }
  if (u->op == POLY_OP_PARAM && u->arg.kind == POLY_ARG_PARAM && u->arg.param &&
      (u->arg.param->device || u->arg.param->device_is_tuple)) {
    result = u->arg.param->device ? device_from_string_arg(u->arg.param->device)
                                  : POLY_DEVICE_AUTO;
    if (cache)
      poly_map_set(cache, poly_ptr_hash(u), u, (void *)(intptr_t)(result + 1), poly_ptr_eq);
    return result;
  }
  if (u->op == POLY_OP_STAGE && u->arg.kind == POLY_ARG_BUFFERIZE_OPTS &&
      !u->arg.bufferize_opts.device_is_tuple && u->arg.bufferize_opts.device) {
    result = device_from_string_arg(u->arg.bufferize_opts.device);
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

  if (u->op != POLY_OP_CONST) {
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

PolyUOp *poly_copy_to_device_uop(PolyCtx *ctx, PolyUOp *value, PolyUOp *device) {
  if (!ctx || !value || !device || device->op != POLY_OP_DEVICE) return NULL;
  PolyArg arg = poly_arg_none();
  if (device->arg.kind == POLY_ARG_STRING)
    arg = poly_arg_str(device->arg.str);
  else if (device->arg.kind == POLY_ARG_STRING_TUPLE)
    arg = poly_arg_string_tuple(device->arg.string_tuple.vals, device->arg.string_tuple.n);
  else
    return NULL;
  return poly_uop1(ctx, POLY_OP_COPY, value->dtype, value, arg);
}

static PolyUOp *copy_to_device(PolyCtx *ctx, PolyUOp *value, PolyDevice device) {
  if (!ctx || !value || device == POLY_DEVICE_AUTO) return value;
  if (value->op == POLY_OP_COPY && poly_uop_device(value) == device)
    return value;
  return poly_copy_to_device_uop(ctx, value, make_device_uop(ctx, device));
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
  if (u->op == POLY_OP_CONST || poly_uop_is_variable(u) ||
      poly_uop_is_bound_var(u) || u->op == POLY_OP_DEVICE || u->op == POLY_OP_UNIQUE)
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

  PolyUOp *view = poly_buffer_view(p->ctx, identity, u->dtype, numel, byte_offset);
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
  return op == POLY_OP_SINK || op == POLY_OP_STORE || op == POLY_OP_AFTER ||
         op == POLY_OP_DEVICE || op == POLY_OP_COPY || op == POLY_OP_CALL ||
         op == POLY_OP_FUNCTION || op == POLY_OP_CUSTOM_FUNCTION || op == POLY_OP_UNSHARD ||
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
    int n_roots,
    PolyUOp **logical_bindings,
    PolyUOp **target_bindings,
    int n_bindings,
    PolyUOp **out_roots
) {
  if (!ctx || n_roots < 0 || n_bindings < 0 || (n_roots > 0 && (!logical_roots || !out_roots)) ||
      (n_bindings > 0 && (!logical_bindings || !target_bindings)))
    return -1;
  if (n_roots == 0) return 0;

  for (int i = 0; i < n_roots; i++) {
    if (!logical_roots[i] || !poly_ctx_owns_ptr(ctx, logical_roots[i]))
      return -1;
    if (!place_validate_logical_root(ctx, logical_roots[i], logical_bindings, n_bindings))
      return -1;
  }

  for (int i = 0; i < n_bindings; i++) {
    PolyUOp *logical = logical_bindings[i];
    PolyUOp *target = target_bindings[i];
    if (!place_direct_buffer_binding(logical) || !place_direct_buffer_binding(target) ||
        !poly_ctx_owns_ptr(ctx, logical) || !poly_ctx_owns_ptr(ctx, target) ||
        poly_uop_device(target) == POLY_DEVICE_AUTO ||
        !place_binding_shape_eq(ctx, logical, target))
      return -1;
    for (int j = 0; j < i; j++) {
      if ((logical_bindings[j] == logical && target_bindings[j] != target) ||
          (logical_bindings[j] != logical && target_bindings[j] == target))
        return -1;
    }
  }

  PolyUOp **candidates = calloc((size_t)n_roots, sizeof(*candidates));
  if (!candidates) return -1;
  int rc = poly_uop_substitute_many(
      ctx, logical_roots, n_roots, logical_bindings, target_bindings, n_bindings, candidates
  );
  for (int i = 0; rc == 0 && i < n_roots; i++)
    if (!place_validate_physical_root(ctx, candidates[i])) rc = -1;
  if (rc == 0) memcpy(out_roots, candidates, (size_t)n_roots * sizeof(*out_roots));
  free(candidates);
  return rc;
}

typedef struct {
  PolyUOp **stops;
  int n_stops;
} PlaceModuleGate;

static bool place_module_region_gate(PolyUOp *u, void *user_data) {
  PlaceModuleGate *gate = user_data;
  if (!u || !gate) return false;
  for (int i = 0; i < gate->n_stops; i++)
    if (gate->stops[i] == u) return false;
  return true;
}

static bool place_same_device_uop(PolyUOp *a, PolyUOp *b) {
  return a && b && a == b && a->op == POLY_OP_DEVICE && b->op == POLY_OP_DEVICE;
}

static PolyUOp *place_exact_copy_to_device(PolyCtx *ctx, PolyUOp *value, PolyUOp *device) {
  if (!ctx || !value || !device || device->op != POLY_OP_DEVICE) return NULL;
  PolyMap *cache = poly_map_new(32);
  if (!cache) return NULL;
  PolyUOp *current = poly_uop_device_uop_cached(ctx, value, cache);
  poly_map_destroy(cache);
  if (place_same_device_uop(current, device)) return value;
  return poly_copy_to_device_uop(ctx, value, device);
}

static PolyUOp *place_binding_on_device_uop(PolyCtx *ctx, PolyUOp *logical, PolyUOp *device) {
  if (!ctx || !place_direct_buffer_binding(logical) || logical->n_src != 1 ||
      !logical->src[0] || logical->src[0]->op != POLY_OP_UNIQUE || !device ||
      device->op != POLY_OP_DEVICE || device->arg.kind != POLY_ARG_STRING ||
      !device_uop_identity_supported(device))
    return NULL;
  int64_t size = logical->arg.kind == POLY_ARG_INT ? logical->arg.i : -1;
  int64_t slot = logical->src[0]->arg.kind == POLY_ARG_INT ? logical->src[0]->arg.i : -1;
  if (size < 0 || slot < 0) return NULL;
  PolyUOp *physical = poly_uop_new_buffer(
      ctx, device, size, logical->dtype, slot
  );
  if (!physical) return NULL;
  return (logical->tag != 0 || logical->tag_arg.kind != POLY_ARG_NONE)
             ? poly_uop_tagged_arg(
                   ctx, POLY_OP_BUFFER, physical->dtype, physical->src,
                   physical->n_src, physical->arg,
                   logical->tag, logical->tag_arg
               )
             : physical;
}

static int place_module_output_index(
    const PolyPlaceModule *modules,
    int n_modules,
    PolyUOp *u
) {
  if (!modules || !u) return -1;
  for (int i = 0; i < n_modules; i++)
    if (modules[i].output == u) return i;
  return -1;
}

static bool place_module_has_input(const PolyPlaceModule *module, PolyUOp *u) {
  if (!module || !u) return false;
  for (int i = 0; i < module->n_inputs; i++)
    if (module->inputs[i] == u) return true;
  return false;
}

static bool place_scalar_devices_valid(PolyCtx *ctx, PolyUOp *root) {
  if (!ctx || !root) return false;
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_ex_alloc(ctx, root, &n_topo, NULL, false);
  PolyMap *device_cache = poly_map_new((size_t)(n_topo > 0 ? n_topo : 1) * 2 + 16);
  if (!topo || !device_cache) {
    free(topo);
    poly_map_destroy(device_cache);
    return false;
  }

  bool valid = true;
  for (int i = 0; i < n_topo && valid; i++) {
    PolyUOp *u = topo[i];
    if (!u || place_lowered_op(u->op) || u->op == POLY_OP_UNSHARD ||
        u->op == POLY_OP_MSELECT || u->op == POLY_OP_MSTACK ||
        u->op == POLY_OP_ALLREDUCE) {
      valid = false;
      break;
    }
    if (u->op == POLY_OP_DEVICE) {
      valid = u->arg.kind == POLY_ARG_STRING && device_uop_identity_supported(u);
      continue;
    }
    if (u->op == POLY_OP_BUFFER) {
      valid = u->n_src == 1 && u->arg.kind == POLY_ARG_PARAM && u->arg.param &&
              device_identity_name_supported(u->arg.param->device);
      continue;
    }
    if (u->op == POLY_OP_COPY) {
      valid = u->n_src == 1 && u->arg.kind == POLY_ARG_STRING &&
              device_identity_name_supported(u->arg.str);
      continue;
    }
    if (u->op == POLY_OP_SINK) continue;

    PolyUOp *expected = NULL;
    for (int j = 0; j < u->n_src; j++) {
      PolyUOp *source_device = poly_uop_device_uop_cached(ctx, u->src[j], device_cache);
      if (!source_device) continue;
      if (!expected)
        expected = source_device;
      else if (!place_same_device_uop(expected, source_device)) {
        valid = false;
        break;
      }
    }
  }

  poly_map_destroy(device_cache);
  free(topo);
  return valid;
}

int poly_place_module_map(
    PolyCtx *ctx,
    PolyUOp **logical_roots,
    int n_roots,
    PolyUOp **logical_bindings,
    int n_bindings,
    const PolyPlaceModule *modules,
    int n_modules,
    PolyUOp **out_bindings,
    PolyUOp **out_roots
) {
  if (!ctx || n_roots <= 0 || !logical_roots || !out_roots || n_bindings < 0 ||
      (n_bindings > 0 && (!logical_bindings || !out_bindings)) || n_modules <= 0 || !modules)
    return -1;

  for (int i = 0; i < n_roots; i++)
    if (!logical_roots[i] || !poly_ctx_owns_ptr(ctx, logical_roots[i]) ||
        !place_validate_logical_root(ctx, logical_roots[i], logical_bindings, n_bindings))
      return -1;

  for (int i = 0; i < n_bindings; i++) {
    if (!place_direct_buffer_binding(logical_bindings[i]) ||
        !poly_ctx_owns_ptr(ctx, logical_bindings[i]))
      return -1;
  }

  PolyUOp **placed_modules = calloc((size_t)n_modules, sizeof(*placed_modules));
  PolyUOp **target_bindings = calloc((size_t)n_bindings, sizeof(*target_bindings));
  PolyUOp **binding_devices = calloc((size_t)n_bindings, sizeof(*binding_devices));
  uint8_t *output_bindings = calloc((size_t)n_bindings, sizeof(*output_bindings));
  PolyUOp **candidates = calloc((size_t)n_roots, sizeof(*candidates));
  int max_inputs = 0;
  for (int i = 0; i < n_modules; i++)
    if (modules[i].n_inputs > max_inputs) max_inputs = modules[i].n_inputs;
  size_t max_subs = (size_t)n_bindings +
                    (size_t)(max_inputs > n_modules ? max_inputs : n_modules) + 16;
  PolyUOp **from = calloc(max_subs, sizeof(*from));
  PolyUOp **to = calloc(max_subs, sizeof(*to));
  int rc = -1;
  if (!placed_modules || (n_bindings > 0 && (!target_bindings || !binding_devices ||
                                             !output_bindings)) ||
      !candidates || !from || !to)
    goto cleanup;

  /* A logical SINK names output storage explicitly.  Its value is placed
   * first; the output BUFFER is then homed on that exact resulting device. */
  for (int i = 0; i < n_roots; i++) {
    PolyUOp *root = logical_roots[i];
    if (root->op != POLY_OP_SINK) continue;
    for (int j = 0; j < root->n_src; j++) {
      PolyUOp *store = root->src[j];
      int binding = place_binding_index(logical_bindings, n_bindings, store->src[0]);
      if (binding < 0) goto cleanup;
      output_bindings[binding] = 1;
    }
  }

  for (int i = 0; i < n_modules; i++) {
    const PolyPlaceModule *module = &modules[i];
    if (!module->name || !module->name[0] || !module->output ||
        !poly_ctx_owns_ptr(ctx, module->output) || module->n_inputs < 0 ||
        (module->n_inputs > 0 && !module->inputs) || !module->device ||
        !poly_ctx_owns_ptr(ctx, module->device) || module->device->op != POLY_OP_DEVICE ||
        module->device->arg.kind != POLY_ARG_STRING ||
        !device_uop_identity_supported(module->device) ||
        place_exact_buffer_identity(module->output))
      goto cleanup;
    for (int j = 0; j < i; j++)
      if (modules[j].output == module->output || strcmp(modules[j].name, module->name) == 0)
        goto cleanup;
    for (int j = 0; j < module->n_inputs; j++) {
      PolyUOp *input = module->inputs[j];
      if (!input || !poly_ctx_owns_ptr(ctx, input) || input == module->output ||
          !poly_uop_reachable(ctx, module->output, input))
        goto cleanup;
      for (int k = 0; k < j; k++)
        if (module->inputs[k] == input) goto cleanup;
      int producer = place_module_output_index(modules, i, input);
      int binding = place_binding_index(logical_bindings, n_bindings, input);
      if (producer < 0 && binding < 0) goto cleanup;
      if (binding >= 0) {
        if (output_bindings[binding]) goto cleanup;
        /* A named graph input is stored once, on its first consumer.  Later
         * consumers on another device receive an exact COPY at their cut. */
        if (!binding_devices[binding]) binding_devices[binding] = module->device;
      }
    }
    for (int j = 0; j < n_modules; j++) {
      if (j == i || !poly_uop_reachable(ctx, module->output, modules[j].output)) continue;
      if (j >= i || !place_module_has_input(module, modules[j].output)) goto cleanup;
    }

    PlaceModuleGate gate = {module->inputs, module->n_inputs};
    int n_region = 0;
    PolyUOp **region = poly_toposort_ex_user_alloc(
        ctx, module->output, &n_region, place_module_region_gate, &gate, false
    );
    if (!region) goto cleanup;
    bool region_valid = true;
    for (int j = 0; j < n_region && region_valid; j++) {
      PolyUOp *u = region[j];
      if (!u || place_logical_forbidden_op(u->op)) {
        region_valid = false;
        break;
      }
      if (!place_exact_buffer_identity(u)) continue;
      int binding = place_binding_index(logical_bindings, n_bindings, u);
      if (binding < 0) {
        region_valid = false;
        break;
      }
      if (output_bindings[binding]) {
        region_valid = false;
        break;
      }
      if (binding_devices[binding] &&
          !place_same_device_uop(binding_devices[binding], module->device)) {
        region_valid = false;
        break;
      }
      binding_devices[binding] = module->device;
    }
    free(region);
    if (!region_valid) goto cleanup;
    for (int j = 0; j < n_bindings; j++) {
      if (output_bindings[j] || target_bindings[j] || !binding_devices[j]) continue;
      target_bindings[j] =
          place_binding_on_device_uop(ctx, logical_bindings[j], binding_devices[j]);
      if (!target_bindings[j] ||
          !place_binding_shape_eq(ctx, logical_bindings[j], target_bindings[j]))
        goto cleanup;
    }

    int n_subs = 0;
    for (int j = 0; j < n_bindings; j++) {
      if (!target_bindings[j] || place_module_has_input(module, logical_bindings[j])) continue;
      from[n_subs] = logical_bindings[j];
      to[n_subs++] = target_bindings[j];
    }
    for (int j = 0; j < module->n_inputs; j++) {
      PolyUOp *input = module->inputs[j];
      int producer = place_module_output_index(modules, i, input);
      int binding = place_binding_index(logical_bindings, n_bindings, input);
      PolyUOp *placed_input = producer >= 0 ? placed_modules[producer]
                                           : (binding >= 0 ? target_bindings[binding] : NULL);
      PolyUOp *on_device = place_exact_copy_to_device(ctx, placed_input, module->device);
      if (!on_device) goto cleanup;
      from[n_subs] = input;
      to[n_subs++] = on_device;
    }
    PolyUOp *module_root = module->output;
    if (poly_uop_substitute_many(
            ctx, &module_root, 1, from, to, n_subs, &placed_modules[i]
        ) != 0 || !placed_modules[i] || !place_scalar_devices_valid(ctx, placed_modules[i]))
      goto cleanup;
    PolyMap *cache = poly_map_new(32);
    PolyUOp *output_device =
        cache ? poly_uop_device_uop_cached(ctx, placed_modules[i], cache) : NULL;
    poly_map_destroy(cache);
    if (!place_same_device_uop(output_device, module->device)) goto cleanup;
  }

  for (int i = 0; i < n_modules; i++) {
    bool reachable = false;
    for (int j = 0; j < n_roots && !reachable; j++)
      reachable = poly_uop_reachable(ctx, logical_roots[j], modules[i].output);
    if (!reachable) goto cleanup;
  }

  for (int i = 0; i < n_bindings; i++)
    if (!output_bindings[i] && !target_bindings[i]) goto cleanup;

  /* Derive every output storage device from its placed STORE value.  This is
   * the only device choice that does not insert an artificial final COPY. */
  int value_subs = 0;
  for (int i = 0; i < n_bindings; i++) {
    if (!target_bindings[i]) continue;
    from[value_subs] = logical_bindings[i];
    to[value_subs++] = target_bindings[i];
  }
  for (int i = 0; i < n_modules; i++) {
    from[value_subs] = modules[i].output;
    to[value_subs++] = placed_modules[i];
  }
  for (int i = 0; i < n_roots; i++) {
    PolyUOp *root = logical_roots[i];
    if (root->op != POLY_OP_SINK) continue;
    for (int j = 0; j < root->n_src; j++) {
      PolyUOp *store = root->src[j];
      int binding = place_binding_index(logical_bindings, n_bindings, store->src[0]);
      PolyUOp *placed_value = NULL;
      PolyUOp *value_root = store->src[1];
      if (binding < 0 || poly_uop_substitute_many(
                             ctx, &value_root, 1, from, to, value_subs, &placed_value
                         ) != 0 ||
          !placed_value)
        goto cleanup;
      PolyMap *cache = poly_map_new(32);
      PolyUOp *device = cache ? poly_uop_device_uop_cached(ctx, placed_value, cache) : NULL;
      poly_map_destroy(cache);
      if (!device || device->op != POLY_OP_DEVICE ||
          (binding_devices[binding] &&
           !place_same_device_uop(binding_devices[binding], device)))
        goto cleanup;
      binding_devices[binding] = device;
    }
  }

  for (int i = 0; i < n_bindings; i++) {
    if (!binding_devices[i]) goto cleanup;
    if (!target_bindings[i]) {
      target_bindings[i] =
          place_binding_on_device_uop(ctx, logical_bindings[i], binding_devices[i]);
      if (!target_bindings[i] ||
          !place_binding_shape_eq(ctx, logical_bindings[i], target_bindings[i]))
        goto cleanup;
    }
    for (int j = 0; j < i; j++) {
      if ((logical_bindings[j] == logical_bindings[i] &&
           target_bindings[j] != target_bindings[i]) ||
          (logical_bindings[j] != logical_bindings[i] &&
           target_bindings[j] == target_bindings[i]))
        goto cleanup;
    }
  }

  int n_subs = 0;
  for (int i = 0; i < n_bindings; i++) {
    from[n_subs] = logical_bindings[i];
    to[n_subs++] = target_bindings[i];
  }
  for (int i = 0; i < n_modules; i++) {
    from[n_subs] = modules[i].output;
    to[n_subs++] = placed_modules[i];
  }
  if (poly_uop_substitute_many(
          ctx, logical_roots, n_roots, from, to, n_subs, candidates
      ) != 0)
    goto cleanup;
  for (int i = 0; i < n_roots; i++)
    if (!place_validate_physical_root(ctx, candidates[i]) ||
        !place_scalar_devices_valid(ctx, candidates[i]))
      goto cleanup;

  memcpy(out_bindings, target_bindings, (size_t)n_bindings * sizeof(*out_bindings));
  memcpy(out_roots, candidates, (size_t)n_roots * sizeof(*out_roots));
  rc = 0;

cleanup:
  free(output_bindings);
  free(binding_devices);
  free(target_bindings);
  free(to);
  free(from);
  free(candidates);
  free(placed_modules);
  return rc;
}
