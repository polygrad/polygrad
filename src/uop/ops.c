/*
 * uop/ops.c — UOp creation with CSE, toposort, pretty-print
 *
 * Mirrors tinygrad's UOp class and UOpMetaClass hash-consing cache.
 * C owns weak UOp records and their immutable argument payloads.
 */

#include "polygrad.h"
#include "bigint.h"
#include "utils.h"
#include "ctx.h"
#include "device.h"
#include "uop/upat.h"
#include "schedule/rangeify.h"
#include "uop/ops.h"
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static bool uop_invalid_gate(PolyUOp *u) {
  return u && u->op == POLY_OP_WHERE && u->n_src == 3 && u->src[2] &&
         u->src[2]->op == POLY_OP_CONST && u->src[2]->arg.kind == POLY_ARG_INVALID;
}

bool poly_uop_addrspace(const PolyUOp *u, PolyAddrSpace *out) {
  if (!u || !out || !u->addrspace_cached) return false;
  *out = u->addrspace_cache;
  return true;
}

/* tinygrad@2026-08-22/a9069c177a9d uop/ops.py:866-878 uses
 * @recursive_property. UOps are immutable, so construction-time evaluation
 * is the equivalent C cache and avoids repeated walks of shared DAGs. */
static void cache_uop_addrspace(PolyUOp *u) {
  if (!u) return;
  u->addrspace_cached = false;
  if (u->op == POLY_OP_PARAM || u->op == POLY_OP_BUFFER) {
    if (u->arg.kind != POLY_ARG_PARAM || !u->arg.param) return;
    u->addrspace_cache = u->arg.param->addrspace;
    u->addrspace_cached = true;
    return;
  }
  if (u->op == POLY_OP_SPECIAL || u->op == POLY_OP_RANGE || u->op == POLY_OP_LOAD) {
    u->addrspace_cache = POLY_ADDR_ALU;
    u->addrspace_cached = true;
    return;
  }
  if (u->op == POLY_OP_INDEX || u->op == POLY_OP_CAST || u->op == POLY_OP_AFTER ||
      u->op == POLY_OP_REDUCE || u->op == POLY_OP_STORE || u->op == POLY_OP_MSTACK ||
      u->op == POLY_OP_MSELECT || u->op == POLY_OP_END || u->op == POLY_OP_UNSHARD ||
      poly_opset_has(POLY_GROUP_MOVEMENT, u->op)) {
    if (u->n_src > 0 && u->src[0]->addrspace_cached) {
      u->addrspace_cache = u->src[0]->addrspace_cache;
      u->addrspace_cached = true;
    }
    return;
  }
  if (u->op == POLY_OP_STACK || u->op == POLY_OP_WMMA || u->op == POLY_OP_GROUP ||
      poly_opset_has(POLY_GROUP_ELEMENTWISE, u->op)) {
    bool found = false;
    PolyAddrSpace common = POLY_ADDR_GLOBAL;
    for (int i = 0; i < u->n_src; i++) {
      if (!u->src[i]->addrspace_cached) continue;
      PolyAddrSpace candidate = u->src[i]->addrspace_cache;
      if (found && candidate != common) return;
      common = candidate;
      found = true;
    }
    if (!found) return;
    u->addrspace_cache = common;
    u->addrspace_cached = true;
  }
}

/* Tinygrad 2026-08-22/a9069c177a9d uop/ops.py:UOp.get_idx. */
PolyUOp *poly_uop_get_idx(PolyCtx *ctx, PolyUOp *u) {
  if (!ctx || !u) return NULL;
  if (u->op == POLY_OP_STACK) {
    PolyUOp *stack_src[16];
    PolyUOp **src = u->n_src <= (int)(sizeof(stack_src) / sizeof(stack_src[0]))
                        ? stack_src
                        : malloc((size_t)u->n_src * sizeof(*src));
    if (!src) return NULL;
    for (int i = 0; i < u->n_src; i++) {
      src[i] = poly_uop_get_idx(ctx, u->src[i]);
      if (!src[i]) {
        if (src != stack_src) free(src);
        return NULL;
      }
    }
    PolyUOp *ret = poly_uop(ctx, POLY_OP_STACK, u->dtype, src, u->n_src, poly_arg_none());
    if (src != stack_src) free(src);
    return ret;
  }
  return uop_invalid_gate(u) ? u->src[1] : u;
}

/* Tinygrad 2026-08-22/a9069c177a9d uop/ops.py:UOp.get_valid. */
PolyUOp *poly_uop_get_valid(PolyCtx *ctx, PolyUOp *u) {
  if (!ctx || !u) return NULL;
  if (u->op == POLY_OP_STACK) {
    PolyUOp *stack_src[16];
    PolyUOp **src = u->n_src <= (int)(sizeof(stack_src) / sizeof(stack_src[0]))
                        ? stack_src
                        : malloc((size_t)u->n_src * sizeof(*src));
    if (!src) return NULL;
    for (int i = 0; i < u->n_src; i++) {
      src[i] = poly_uop_get_valid(ctx, u->src[i]);
      if (!src[i]) {
        if (src != stack_src) free(src);
        return NULL;
      }
    }
    PolyUOp *ret = poly_uop_stack(ctx, src, u->n_src);
    if (src != stack_src) free(src);
    return ret;
  }
  if (uop_invalid_gate(u)) return u->src[0];
  return poly_uop0(
      ctx, POLY_OP_CONST, POLY_BOOL,
      poly_arg_bool(!(u->op == POLY_OP_CONST && u->arg.kind == POLY_ARG_INVALID))
  );
}

char poly_axis_letter(PolyArg arg) {
  static const char letters[] = {
      [POLY_AXIS_DEVICE] = 'd', [POLY_AXIS_GLOBAL] = 'g',       [POLY_AXIS_WARP] = 'w',
      [POLY_AXIS_LOCAL] = 'l',  [POLY_AXIS_WEAK] = 'L',         [POLY_AXIS_GROUP_REDUCE] = 'G',
      [POLY_AXIS_REDUCE] = 'R', [POLY_AXIS_UPCAST] = 'u',       [POLY_AXIS_UNROLL] = 'r',
      [POLY_AXIS_THREAD] = 't', [POLY_AXIS_PLACEHOLDER] = '\0', [POLY_AXIS_LOOP] = 'L',
  };
  PolyAxisType axis_type = poly_range_axis_type(arg);
  if (axis_type < 0 || axis_type >= (int)(sizeof(letters) / sizeof(letters[0])) ||
      !letters[axis_type])
    return '\0';
  return letters[axis_type];
}

char *poly_range_str(PolyArg arg) {
  int n_extra = poly_range_n_extra(arg);
  size_t cap = 24 + (size_t)n_extra * 24;
  char *name = malloc(cap);
  if (!name) return NULL;
  int used = 0;
  for (int i = -1; i < n_extra; i++) {
    int64_t value = i < 0 ? poly_range_axis_id(arg) : poly_range_extra(arg)[i];
    uint64_t magnitude = value < 0 ? (uint64_t)(-(value + 1)) + 1 : (uint64_t)value;
    used += snprintf(
        name + used, cap - (size_t)used, "%s%s%llu", i < 0 ? "" : "_", value < 0 ? "m" : "",
        (unsigned long long)magnitude
    );
  }
  return name;
}

typedef struct {
  PolyUOp **items;
  int count;
  int capacity;
} SplitUOps;

static bool split_uops_append(SplitUOps *out, PolyUOp *u) {
  if (out->count == out->capacity) {
    if (out->capacity > INT_MAX / 2) return false;
    int capacity = out->capacity ? out->capacity * 2 : 16;
    PolyUOp **items = realloc(out->items, (size_t)capacity * sizeof(*items));
    if (!items) return false;
    out->items = items;
    out->capacity = capacity;
  }
  out->items[out->count++] = u;
  return true;
}

static bool split_uops_collect(PolyUOp *u, PolyOps sep, SplitUOps *out) {
  if (u->op != sep) return split_uops_append(out, u);
  for (int i = 0; i < u->n_src; i++)
    if (!split_uops_collect(u->src[i], sep, out)) return false;
  return true;
}

PolyUOp **poly_uop_split(PolyUOp *u, PolyOps sep, int *n_out) {
  if (n_out) *n_out = 0;
  if (!u || !n_out) return NULL;
  SplitUOps out = {0};
  if (!split_uops_collect(u, sep, &out)) {
    free(out.items);
    return NULL;
  }
  *n_out = out.count;
  return out.items;
}

int poly_uop_resolve(PolyCtx *ctx, PolyUOp *u, int default_value) {
  if (!ctx || !u || !poly_dtype_eq(u->dtype, POLY_BOOL)) return -1;
  PolyUOp *simplified = poly_graph_rewrite(ctx, u, poly_symbolic());
  if (!simplified) return -1;
  int64_t vmin = 0, vmax = 0;
  poly_uop_minmax(ctx, simplified, &vmin, &vmax);
  return vmin == vmax ? (vmin != 0) : (default_value != 0);
}

/* Current tinygrad UOp.contiguous_view_offset rewrites
 * flatten().index(RANGE) through pm_mops+symbolic, then accepts only a unit
 * range plus a constant offset. */
int poly_uop_contiguous_view_offset(PolyCtx *ctx, PolyUOp *u, int64_t *out) {
  if (!ctx || !u || !out) return -1;
  const char *device = poly_uop_device_name(ctx, u);
  if (device && (!strncmp(device, "WEBGPU", 6) || !strncmp(device, "CL", 2))) return -1;

  int ndim = poly_uop_ndim(ctx, u);
  if (ndim < 0 || ndim > POLY_MAX_DIMS) return -1;
  PolyUOp *numel = poly_const_int(ctx, 1);
  for (int i = 0; i < ndim; i++) {
    PolyUOp *dim = poly_uop_shape_dim(ctx, u, i);
    if (!dim || !(numel = poly_uop2(ctx, POLY_OP_MUL, POLY_WEAKINT, numel, dim, poly_arg_none())))
      return -1;
  }
  numel = poly_graph_rewrite(ctx, numel, poly_symbolic());
  if (!numel) return -1;

  PolyUOp *flat = poly_reshape_uop(ctx, u, &numel, 1);
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, numel, poly_arg_range(0, POLY_AXIS_WEAK));
  PolyUOp *indexed = range ? poly_uop_index(ctx, flat, &range, 1) : NULL;
  PolyPatternMatcher *pm = poly_pm_concat(poly_pm_mops(), poly_symbolic());
  PolyUOp *rewritten = indexed && pm ? poly_graph_rewrite(ctx, indexed, pm) : NULL;
  poly_pm_destroy(pm);
  if (!rewritten || rewritten->op != POLY_OP_INDEX || rewritten->n_src != 2 ||
      poly_opset_has(POLY_GROUP_MOVEMENT, rewritten->src[0]->op))
    return -1;

  PolyUOp *index = rewritten->src[1];
  if (index->op == POLY_OP_RANGE) {
    *out = 0;
    return 0;
  }
  if (index->op == POLY_OP_ADD && index->n_src == 2) {
    PolyUOp *constant = NULL;
    if (index->src[0]->op == POLY_OP_RANGE) constant = index->src[1];
    if (index->src[1]->op == POLY_OP_RANGE) constant = index->src[0];
    if (constant && poly_uop_bind_value(constant, out) == 0) return 0;
  }

  int64_t concrete_numel = 0;
  return poly_uop_bind_value(numel, &concrete_numel) == 0 && concrete_numel == 1
             ? poly_uop_bind_value(index, out)
             : -1;
}

/* PolyArg equality and hashing */

static uint32_t hash_mix(uint32_t h, uint32_t v);

static bool poly_opt_eq(const PolyOpt *a, const PolyOpt *b) {
  if (a->op != b->op || a->has_axis != b->has_axis || (a->has_axis && a->axis != b->axis) ||
      a->arg_kind != b->arg_kind)
    return false;
  if (a->arg_kind == POLY_OPT_ARG_INT) return a->arg == b->arg;
  if (a->arg_kind != POLY_OPT_ARG_INT_TUPLE) return true;
  return a->n_arg_tuple == b->n_arg_tuple &&
         (a->n_arg_tuple == 0 ||
          (a->arg_tuple && b->arg_tuple &&
           memcmp(a->arg_tuple, b->arg_tuple, (size_t)a->n_arg_tuple * sizeof(*a->arg_tuple)) == 0)
         );
}

static bool poly_opts_eq(const PolyOpt *a, int na, const PolyOpt *b, int nb) {
  if (na != nb || (na > 0 && (!a || !b))) return false;
  for (int i = 0; i < na; i++)
    if (!poly_opt_eq(&a[i], &b[i])) return false;
  return true;
}

bool poly_kernel_info_eq(const PolyKernelInfo *a, const PolyKernelInfo *b) {
  if (a == b) return true;
  if (!a || !b || a->dont_use_locals != b->dont_use_locals || a->n_axis_types != b->n_axis_types ||
      a->has_opts_to_apply != b->has_opts_to_apply || a->beam != b->beam ||
      (a->name != b->name && (!a->name || !b->name || strcmp(a->name, b->name) != 0)))
    return false;
  if (a->n_axis_types > 0 &&
      (!a->axis_types || !b->axis_types ||
       memcmp(a->axis_types, b->axis_types, (size_t)a->n_axis_types * sizeof(*a->axis_types)) != 0))
    return false;
  if (!poly_opts_eq(a->applied_opts, a->n_applied_opts, b->applied_opts, b->n_applied_opts) ||
      !poly_opts_eq(a->opts_to_apply, a->n_opts_to_apply, b->opts_to_apply, b->n_opts_to_apply))
    return false;
  if (a->estimates == b->estimates) return true;
  return a->estimates && b->estimates && a->estimates->ops == b->estimates->ops &&
         a->estimates->lds == b->estimates->lds && a->estimates->mem == b->estimates->mem;
}

static uint32_t poly_opt_hash(uint32_t h, const PolyOpt *opt) {
  h = hash_mix(h, (uint32_t)opt->op);
  h = hash_mix(h, opt->has_axis ? (uint32_t)opt->axis + 1u : 0u);
  h = hash_mix(h, (uint32_t)opt->arg_kind);
  if (opt->arg_kind == POLY_OPT_ARG_INT)
    h = hash_mix(h, (uint32_t)(opt->arg ^ (opt->arg >> 32)));
  else if (opt->arg_kind == POLY_OPT_ARG_INT_TUPLE) {
    h = hash_mix(h, (uint32_t)opt->n_arg_tuple);
    for (int i = 0; i < opt->n_arg_tuple; i++)
      h = hash_mix(h, (uint32_t)(opt->arg_tuple[i] ^ (opt->arg_tuple[i] >> 32)));
  }
  return h;
}

uint32_t poly_kernel_info_hash(const PolyKernelInfo *info) {
  if (!info) return 0;
  uint32_t h = UINT32_C(2166136261);
  if (info->name)
    for (const char *p = info->name; *p; p++)
      h = hash_mix(h, (uint32_t)*p);
  h = hash_mix(h, info->dont_use_locals ? 1u : 0u);
  h = hash_mix(h, (uint32_t)info->n_axis_types);
  for (int i = 0; i < info->n_axis_types; i++)
    h = hash_mix(h, (uint32_t)info->axis_types[i]);
  h = hash_mix(h, (uint32_t)info->n_applied_opts);
  for (int i = 0; i < info->n_applied_opts; i++)
    h = poly_opt_hash(h, &info->applied_opts[i]);
  h = hash_mix(h, info->has_opts_to_apply ? 1u : 0u);
  h = hash_mix(h, (uint32_t)info->n_opts_to_apply);
  for (int i = 0; i < info->n_opts_to_apply; i++)
    h = poly_opt_hash(h, &info->opts_to_apply[i]);
  if (info->estimates) {
    h = hash_mix(h, (uint32_t)(uintptr_t)info->estimates->ops);
    h = hash_mix(h, (uint32_t)(uintptr_t)info->estimates->lds);
    h = hash_mix(h, (uint32_t)(uintptr_t)info->estimates->mem);
  }
  return hash_mix(h, (uint32_t)info->beam);
}

bool poly_arg_eq(PolyArg a, PolyArg b) {
  if (a.kind != b.kind) return false;
  switch (a.kind) {
  case POLY_ARG_NONE:
    return true;
  case POLY_ARG_INVALID:
    return true;
  case POLY_ARG_INT:
    return a.i == b.i;
  case POLY_ARG_BIGINT:
    return a.bigint.sign == b.bigint.sign && a.bigint.n_limbs == b.bigint.n_limbs &&
           (a.bigint.n_limbs == 0 ||
            (a.bigint.limbs && b.bigint.limbs &&
             memcmp(a.bigint.limbs, b.bigint.limbs, (size_t)a.bigint.n_limbs * sizeof(uint32_t)) ==
                 0));
  case POLY_ARG_FLOAT: /* bitwise compare to distinguish -0.0 from 0.0 */
  {
    uint64_t ba, bb;
    memcpy(&ba, &a.f, 8);
    memcpy(&bb, &b.f, 8);
    return ba == bb;
  }
  case POLY_ARG_BOOL:
    return a.b == b.b;
  case POLY_ARG_STRING:
    return a.str == b.str || (a.str && b.str && strcmp(a.str, b.str) == 0);
  case POLY_ARG_STRING_TUPLE:
    if (a.string_tuple.n != b.string_tuple.n) return false;
    for (int i = 0; i < a.string_tuple.n; i++) {
      const char *av = a.string_tuple.vals ? a.string_tuple.vals[i] : NULL;
      const char *bv = b.string_tuple.vals ? b.string_tuple.vals[i] : NULL;
      if (av != bv && (!av || !bv || strcmp(av, bv) != 0)) return false;
    }
    return true;
  case POLY_ARG_OPS:
    return a.ops == b.ops;
  case POLY_ARG_REDUCE:
    return a.reduce.op == b.reduce.op && a.reduce.num_axes == b.reduce.num_axes;
  case POLY_ARG_ALLREDUCE:
    if (a.allreduce.op != b.allreduce.op ||
        a.allreduce.device_is_tuple != b.allreduce.device_is_tuple)
      return false;
    if (a.allreduce.device_is_tuple) {
      if (a.allreduce.n_devices != b.allreduce.n_devices) return false;
      for (int i = 0; i < a.allreduce.n_devices; i++) {
        const char *av = a.allreduce.devices ? a.allreduce.devices[i] : NULL;
        const char *bv = b.allreduce.devices ? b.allreduce.devices[i] : NULL;
        if (av != bv && (!av || !bv || strcmp(av, bv) != 0)) return false;
      }
      return true;
    }
    return a.allreduce.device == b.allreduce.device ||
           (a.allreduce.device && b.allreduce.device &&
            strcmp(a.allreduce.device, b.allreduce.device) == 0);
  case POLY_ARG_INT_TUPLE:
    if (a.int_tuple.n != b.int_tuple.n) return false;
    if (a.int_tuple.n == 0) return true;
    return memcmp(a.int_tuple.vals, b.int_tuple.vals, a.int_tuple.n * sizeof(int64_t)) == 0;
  case POLY_ARG_RANGE:
    if (a.range.axis_id != b.range.axis_id) return false;
    if (a.range.axis_type != b.range.axis_type) return false;
    if (a.range.n_extra != b.range.n_extra) return false;
    if (a.range.n_extra == 0) return true;
    return memcmp(a.range.extra, b.range.extra, (size_t)a.range.n_extra * sizeof(int64_t)) == 0;
  case POLY_ARG_BUFFERIZE_OPTS:
    if (a.bufferize_opts.device_is_tuple != b.bufferize_opts.device_is_tuple ||
        a.bufferize_opts.device_is_int != b.bufferize_opts.device_is_int ||
        a.bufferize_opts.addrspace != b.bufferize_opts.addrspace ||
        a.bufferize_opts.removable != b.bufferize_opts.removable)
      return false;
    if (a.bufferize_opts.device_is_int)
      return a.bufferize_opts.device_int == b.bufferize_opts.device_int;
    if (a.bufferize_opts.device_is_tuple) {
      if (a.bufferize_opts.n_devices != b.bufferize_opts.n_devices) return false;
      for (int i = 0; i < a.bufferize_opts.n_devices; i++) {
        const char *av = a.bufferize_opts.devices ? a.bufferize_opts.devices[i] : NULL;
        const char *bv = b.bufferize_opts.devices ? b.bufferize_opts.devices[i] : NULL;
        if (av != bv && (!av || !bv || strcmp(av, bv) != 0)) return false;
      }
      return true;
    }
    return a.bufferize_opts.device == b.bufferize_opts.device ||
           (a.bufferize_opts.device && b.bufferize_opts.device &&
            strcmp(a.bufferize_opts.device, b.bufferize_opts.device) == 0);
  case POLY_ARG_TENSOR_CORE:
    if (a.tensor_core.threads != b.tensor_core.threads ||
        a.tensor_core.has_upcast_axes != b.tensor_core.has_upcast_axes ||
        !poly_dtype_eq(a.tensor_core.dtype_in, b.tensor_core.dtype_in) ||
        memcmp(a.tensor_core.dims, b.tensor_core.dims, sizeof(a.tensor_core.dims)) != 0)
      return false;
    if (a.tensor_core.device != b.tensor_core.device &&
        (!a.tensor_core.device || !b.tensor_core.device ||
         strcmp(a.tensor_core.device, b.tensor_core.device) != 0))
      return false;
    for (int d = 0; d < 3; d++) {
      if (a.tensor_core.n_upcast_axes[d] != b.tensor_core.n_upcast_axes[d]) return false;
      if (a.tensor_core.n_upcast_axes[d] > 0 &&
          (!a.tensor_core.upcast_axes[d] || !b.tensor_core.upcast_axes[d] ||
           memcmp(
               a.tensor_core.upcast_axes[d], b.tensor_core.upcast_axes[d],
               (size_t)a.tensor_core.n_upcast_axes[d] * sizeof(*a.tensor_core.upcast_axes[d])
           ) != 0))
        return false;
    }
    return true;
  case POLY_ARG_PROGRAM_INFO:
    return poly_program_info_eq(a.program_info, b.program_info);
  case POLY_ARG_KERNEL_INFO:
    return poly_kernel_info_eq(a.kernel_info, b.kernel_info);
  case POLY_ARG_BYTES:
    if (a.bytes.n != b.bytes.n) return false;
    if (a.bytes.n == 0) return true;
    if (!a.bytes.data || !b.bytes.data) return false;
    return memcmp(a.bytes.data, b.bytes.data, (size_t)a.bytes.n) == 0;
  case POLY_ARG_PARAM:
    if (a.param == b.param) return true;
    if (!a.param || !b.param) return false;
    if (a.param->slot != b.param->slot || !poly_dtype_eq(a.param->dtype, b.param->dtype) ||
        a.param->min_val != b.param->min_val || a.param->max_val != b.param->max_val ||
        a.param->has_minmax != b.param->has_minmax ||
        a.param->multiple_of != b.param->multiple_of ||
        a.param->has_multiple_of != b.param->has_multiple_of ||
        a.param->addrspace != b.param->addrspace || a.param->axis != b.param->axis ||
        a.param->has_axis != b.param->has_axis ||
        a.param->device_is_tuple != b.param->device_is_tuple ||
        a.param->n_devices != b.param->n_devices || a.param->volatile_ != b.param->volatile_)
      return false;
    if (a.param->device != b.param->device &&
        (!a.param->device || !b.param->device || strcmp(a.param->device, b.param->device) != 0))
      return false;
    for (int i = 0; i < a.param->n_devices; i++) {
      const char *av = a.param->devices ? a.param->devices[i] : NULL;
      const char *bv = b.param->devices ? b.param->devices[i] : NULL;
      if (av != bv && (!av || !bv || strcmp(av, bv) != 0)) return false;
    }
    if (a.param->name == b.param->name) return true;
    return a.param->name && b.param->name && strcmp(a.param->name, b.param->name) == 0;
  case POLY_ARG_CALL_INFO:
    if (a.call_info == b.call_info) return true;
    if (!a.call_info || !b.call_info) return false;
    if (a.call_info->precompile != b.call_info->precompile ||
        a.call_info->precompile_backward != b.call_info->precompile_backward ||
        a.call_info->has_grad_fxn != b.call_info->has_grad_fxn ||
        a.call_info->grad_fxn_key != b.call_info->grad_fxn_key ||
        a.call_info->has_aux != b.call_info->has_aux)
      return false;
    return a.call_info->name == b.call_info->name ||
           (a.call_info->name && b.call_info->name &&
            strcmp(a.call_info->name, b.call_info->name) == 0);
  case POLY_ARG_DTYPE:
    return poly_dtype_eq(a.dtype, b.dtype);
  }
  return false;
}

static uint32_t hash_mix(uint32_t h, uint32_t v) {
  h ^= v;
  h *= 0x9e3779b9;
  h ^= h >> 16;
  return h;
}

uint32_t poly_arg_hash(PolyArg a) {
  uint32_t h = (uint32_t)a.kind;
  switch (a.kind) {
  case POLY_ARG_NONE:
  case POLY_ARG_INVALID:
    break;
  case POLY_ARG_INT:
    h = hash_mix(h, (uint32_t)(a.i ^ (a.i >> 32)));
    break;
  case POLY_ARG_BIGINT:
    h = hash_mix(h, (uint32_t)(int32_t)a.bigint.sign);
    h = hash_mix(h, a.bigint.n_limbs);
    for (uint32_t i = 0; i < a.bigint.n_limbs; i++)
      h = hash_mix(h, a.bigint.limbs ? a.bigint.limbs[i] : 0);
    break;
  case POLY_ARG_FLOAT: {
    uint64_t bits;
    memcpy(&bits, &a.f, 8);
    h = hash_mix(h, (uint32_t)(bits ^ (bits >> 32)));
  } break;
  case POLY_ARG_BOOL:
    h = hash_mix(h, a.b ? 1 : 0);
    break;
  case POLY_ARG_STRING:
    if (a.str) {
      for (const char *p = a.str; *p; p++)
        h = hash_mix(h, (uint32_t)*p);
    }
    break;
  case POLY_ARG_STRING_TUPLE:
    h = hash_mix(h, (uint32_t)a.string_tuple.n);
    for (int i = 0; i < a.string_tuple.n; i++) {
      const char *value = a.string_tuple.vals ? a.string_tuple.vals[i] : NULL;
      if (value)
        for (const char *p = value; *p; p++)
          h = hash_mix(h, (uint32_t)*p);
      h = hash_mix(h, UINT32_C(0xff));
    }
    break;
  case POLY_ARG_OPS:
    h = hash_mix(h, (uint32_t)a.ops);
    break;
  case POLY_ARG_REDUCE:
    h = hash_mix(h, (uint32_t)a.reduce.op);
    h = hash_mix(h, (uint32_t)a.reduce.num_axes);
    break;
  case POLY_ARG_ALLREDUCE:
    h = hash_mix(h, (uint32_t)a.allreduce.op);
    h = hash_mix(h, a.allreduce.device_is_tuple ? 2u : 1u);
    if (a.allreduce.device_is_tuple) {
      h = hash_mix(h, (uint32_t)a.allreduce.n_devices);
      for (int i = 0; i < a.allreduce.n_devices; i++) {
        const char *value = a.allreduce.devices ? a.allreduce.devices[i] : NULL;
        if (value)
          for (const char *p = value; *p; p++)
            h = hash_mix(h, (uint32_t)*p);
        h = hash_mix(h, UINT32_C(0xff));
      }
    } else if (a.allreduce.device) {
      for (const char *p = a.allreduce.device; *p; p++)
        h = hash_mix(h, (uint32_t)*p);
    }
    break;
  case POLY_ARG_INT_TUPLE:
    for (int i = 0; i < a.int_tuple.n; i++)
      h = hash_mix(h, (uint32_t)(a.int_tuple.vals[i] ^ (a.int_tuple.vals[i] >> 32)));
    break;
  case POLY_ARG_RANGE:
    h = hash_mix(h, (uint32_t)(a.range.axis_id ^ (a.range.axis_id >> 32)));
    h = hash_mix(h, (uint32_t)a.range.axis_type);
    for (int i = 0; i < a.range.n_extra; i++)
      h = hash_mix(h, (uint32_t)(a.range.extra[i] ^ (a.range.extra[i] >> 32)));
    break;
  case POLY_ARG_BUFFERIZE_OPTS:
    h = hash_mix(
        h, a.bufferize_opts.device_is_int     ? 3u
           : a.bufferize_opts.device_is_tuple ? 2u
                                              : 1u
    );
    if (a.bufferize_opts.device_is_int) {
      uint64_t id = (uint64_t)a.bufferize_opts.device_int;
      h = hash_mix(h, (uint32_t)id);
      h = hash_mix(h, (uint32_t)(id >> 32));
    } else if (a.bufferize_opts.device_is_tuple) {
      h = hash_mix(h, (uint32_t)a.bufferize_opts.n_devices);
      for (int i = 0; i < a.bufferize_opts.n_devices; i++) {
        const char *value = a.bufferize_opts.devices ? a.bufferize_opts.devices[i] : NULL;
        if (value)
          for (const char *p = value; *p; p++)
            h = hash_mix(h, (uint32_t)*p);
        h = hash_mix(h, UINT32_C(0xff));
      }
    } else if (a.bufferize_opts.device) {
      for (const char *p = a.bufferize_opts.device; *p; p++)
        h = hash_mix(h, (uint32_t)*p);
    }
    h = hash_mix(h, (uint32_t)a.bufferize_opts.addrspace);
    h = hash_mix(h, a.bufferize_opts.removable ? 1u : 0u);
    break;
  case POLY_ARG_TENSOR_CORE:
    if (a.tensor_core.device) {
      for (const char *p = a.tensor_core.device; *p; p++)
        h = hash_mix(h, (uint32_t)*p);
    }
    for (int i = 0; i < 3; i++)
      h = hash_mix(h, (uint32_t)a.tensor_core.dims[i]);
    h = hash_mix(h, (uint32_t)a.tensor_core.dtype_in.priority);
    h = hash_mix(h, (uint32_t)a.tensor_core.dtype_in.bitsize);
    h = hash_mix(h, (uint32_t)a.tensor_core.threads);
    h = hash_mix(h, a.tensor_core.has_upcast_axes ? 1u : 0u);
    for (int d = 0; d < 3; d++) {
      h = hash_mix(h, (uint32_t)a.tensor_core.n_upcast_axes[d]);
      for (int i = 0; i < a.tensor_core.n_upcast_axes[d]; i++) {
        h = hash_mix(h, (uint32_t)a.tensor_core.upcast_axes[d][i][0]);
        h = hash_mix(h, (uint32_t)a.tensor_core.upcast_axes[d][i][1]);
      }
    }
    break;
  case POLY_ARG_PROGRAM_INFO:
    h = hash_mix(h, poly_program_info_hash(a.program_info));
    break;
  case POLY_ARG_KERNEL_INFO:
    h = hash_mix(h, poly_kernel_info_hash(a.kernel_info));
    break;
  case POLY_ARG_BYTES:
    for (int i = 0; i < a.bytes.n; i++)
      h = hash_mix(h, a.bytes.data ? a.bytes.data[i] : 0);
    break;
  case POLY_ARG_PARAM:
    if (a.param) {
      h = hash_mix(h, (uint32_t)(a.param->slot ^ (a.param->slot >> 32)));
      h = hash_mix(h, (uint32_t)a.param->dtype.priority);
      h = hash_mix(h, (uint32_t)a.param->dtype.bitsize);
      if (a.param->device)
        for (const char *p = a.param->device; *p; p++)
          h = hash_mix(h, (uint32_t)*p);
      h = hash_mix(h, a.param->device_is_tuple ? 1u : 0u);
      h = hash_mix(h, (uint32_t)a.param->n_devices);
      for (int i = 0; i < a.param->n_devices; i++) {
        const char *device = a.param->devices ? a.param->devices[i] : NULL;
        if (device)
          for (const char *p = device; *p; p++)
            h = hash_mix(h, (uint32_t)*p);
        h = hash_mix(h, UINT32_C(0xff));
      }
      h = hash_mix(h, (uint32_t)a.param->addrspace);
      h = hash_mix(h, (uint32_t)a.param->axis);
      h = hash_mix(h, a.param->has_axis ? 1u : 0u);
      h = hash_mix(h, a.param->has_minmax ? 1u : 0u);
      h = hash_mix(h, (uint32_t)(a.param->min_val ^ (a.param->min_val >> 32)));
      h = hash_mix(h, (uint32_t)(a.param->max_val ^ (a.param->max_val >> 32)));
      h = hash_mix(h, a.param->has_multiple_of ? 1u : 0u);
      h = hash_mix(h, (uint32_t)(a.param->multiple_of ^ (a.param->multiple_of >> 32)));
      h = hash_mix(h, a.param->volatile_ ? 1u : 0u);
      if (a.param->name) {
        for (const char *p = a.param->name; *p; p++)
          h = hash_mix(h, (uint32_t)*p);
      }
    }
    break;
  case POLY_ARG_CALL_INFO:
    if (a.call_info) {
      if (a.call_info->name)
        for (const char *p = a.call_info->name; *p; p++)
          h = hash_mix(h, (uint32_t)*p);
      h = hash_mix(h, a.call_info->precompile ? 1u : 0u);
      h = hash_mix(h, a.call_info->precompile_backward ? 1u : 0u);
      h = hash_mix(h, a.call_info->has_grad_fxn ? 1u : 0u);
      h = hash_mix(h, a.call_info->grad_fxn_key);
      h = hash_mix(h, a.call_info->has_aux ? 1u : 0u);
    }
    break;
  case POLY_ARG_DTYPE:
    h = hash_mix(h, (uint32_t)a.dtype.priority);
    h = hash_mix(h, (uint32_t)a.dtype.bitsize);
    if (a.dtype.name)
      for (const char *p = a.dtype.name; *p; p++)
        h = hash_mix(h, (uint32_t)*p);
    break;
  }
  return h;
}

/* CSE key: (op, dtype, src[], arg, tag) */

typedef struct {
  PolyOps op;
  PolyDType dtype;
  PolyUOp **src;
  uint16_t n_src;
  PolyArg arg;
  int32_t tag;
  PolyArg tag_arg;
} CseKey;

typedef struct PolyUOpOwnedAllocation {
  struct PolyUOpOwnedAllocation *next;
  size_t bytes;
  max_align_t alignment;
  unsigned char data[];
} PolyUOpOwnedAllocation;

typedef struct {
  PolyUOp uop;
  CseKey key;
  PolyUOpOwnedAllocation *allocations;
  size_t owned_bytes;
  PolyUOp *src[];
} PolyUOpStorage;

static PolyUOpStorage *uop_storage_get(PolyCtx *ctx, const PolyUOp *uop) {
  return ctx && uop ? poly_map_get(ctx->uop_storage, poly_ptr_hash(uop), uop, poly_ptr_eq) : NULL;
}

bool poly_uop_storage_contains(PolyCtx *ctx, const void *ptr) {
  return uop_storage_get(ctx, ptr) != NULL;
}

static PolyUOpStorage *uop_storage_new(PolyCtx *ctx, int n_src) {
  if (!ctx || n_src < 0 || (size_t)n_src > (SIZE_MAX - sizeof(PolyUOpStorage)) / sizeof(PolyUOp *))
    return NULL;
  size_t size = sizeof(PolyUOpStorage) + (size_t)n_src * sizeof(PolyUOp *);
  PolyUOpStorage *storage = calloc(1, size);
  return storage;
}

static size_t uop_storage_size(const PolyUOpStorage *storage) {
  return storage ? sizeof(*storage) + (size_t)storage->uop.n_src * sizeof(PolyUOp *) +
                       storage->owned_bytes
                 : 0;
}

static void *uop_storage_alloc(PolyUOpStorage *storage, size_t size, size_t align) {
  if (!storage || align > _Alignof(max_align_t) || size > SIZE_MAX - sizeof(PolyUOpOwnedAllocation))
    return NULL;
  size_t bytes = sizeof(PolyUOpOwnedAllocation) + size;
  PolyUOpOwnedAllocation *allocation = malloc(bytes);
  if (!allocation) return NULL;
  allocation->next = storage->allocations;
  allocation->bytes = bytes;
  storage->allocations = allocation;
  storage->owned_bytes += bytes;
  return allocation->data;
}

#ifdef POLY_TESTING
static _Thread_local int range_alloc_fail_after = -1;
void poly_test_range_alloc_fail_after(int count) {
  range_alloc_fail_after = count;
}
#endif

static void *uop_storage_alloc_live(
    PolyCtx *ctx,
    PolyUOpStorage *storage,
    size_t size,
    size_t align
) {
#ifdef POLY_TESTING
  if (range_alloc_fail_after == 0) return NULL;
  if (range_alloc_fail_after > 0) range_alloc_fail_after--;
#endif
  if (!storage) return NULL;
  size_t before = storage->owned_bytes;
  void *data = uop_storage_alloc(storage, size, align);
  if (!data || !ctx || uop_storage_get(ctx, &storage->uop) != storage) return data;
  ctx->uop_storage_bytes += storage->owned_bytes - before;
  if (ctx->uop_storage_bytes > ctx->uop_storage_high_water)
    ctx->uop_storage_high_water = ctx->uop_storage_bytes;
  return data;
}

static void uop_storage_dispose(PolyUOpStorage *storage) {
  if (!storage) return;
  PolyUOpOwnedAllocation *allocation = storage->allocations;
  while (allocation) {
    PolyUOpOwnedAllocation *next = allocation->next;
    free(allocation);
    allocation = next;
  }
  free(storage);
}

static void uop_storage_register(PolyCtx *ctx, PolyUOpStorage *storage) {
  PolyUOp *uop = &storage->uop;
  poly_map_set(ctx->uop_storage, poly_ptr_hash(uop), uop, storage, poly_ptr_eq);
  ctx->uop_storage_bytes += uop_storage_size(storage);
  if (ctx->uop_storage_bytes > ctx->uop_storage_high_water)
    ctx->uop_storage_high_water = ctx->uop_storage_bytes;
}

static void uop_storage_free(PolyCtx *ctx, PolyUOp *uop) {
  PolyUOpStorage *storage = uop_storage_get(ctx, uop);
  if (!storage) return;
  size_t size = uop_storage_size(storage);
  poly_map_remove(ctx->uop_storage, poly_ptr_hash(uop), uop, poly_ptr_eq);
  ctx->uop_storage_bytes -= size;
  uop_storage_dispose(storage);
}

typedef struct {
  PolyUOpStorage **items;
  size_t count;
} UOpStorageRows;

static void collect_uop_storage(const void *key, void *value, void *userdata) {
  (void)key;
  UOpStorageRows *rows = userdata;
  rows->items[rows->count++] = value;
}

void poly_uop_storage_destroy_all(PolyCtx *ctx) {
  if (!ctx || !ctx->uop_storage) return;
  size_t count = poly_map_len(ctx->uop_storage);
  PolyUOpStorage **items = count ? malloc(count * sizeof(*items)) : NULL;
  if (count && !items) abort();
  UOpStorageRows rows = {.items = items};
  poly_map_foreach(ctx->uop_storage, collect_uop_storage, &rows);
  for (size_t i = 0; i < rows.count; i++)
    uop_storage_dispose(rows.items[i]);
  free(items);
  poly_map_clear(ctx->uop_storage);
  ctx->uop_storage_bytes = 0;
}

static uint32_t cse_hash(const CseKey *k) {
  uint32_t h = (uint32_t)k->op;
  h = hash_mix(h, (uint32_t)k->dtype.priority);
  h = hash_mix(h, (uint32_t)k->dtype.bitsize);
  if (k->dtype.name) {
    const unsigned char *p = (const unsigned char *)k->dtype.name;
    while (*p)
      h = hash_mix(h, (uint32_t)*p++);
  }
  for (int i = 0; i < k->n_src; i++) {
    /* hash pointer as integer — identity-based like tinygrad */
    uintptr_t p = (uintptr_t)k->src[i];
    h = hash_mix(h, (uint32_t)(p ^ (sizeof(p) > 4 ? (uint32_t)(p >> 32) : 0)));
  }
  h = hash_mix(h, poly_arg_hash(k->arg));
  h = hash_mix(h, (uint32_t)k->tag);
  h = hash_mix(h, poly_arg_hash(k->tag_arg));
  return h;
}

static bool cse_eq(const void *a, const void *b) {
  const CseKey *ka = a, *kb = b;
  if (ka->op != kb->op) return false;
  if (!poly_dtype_eq(ka->dtype, kb->dtype)) return false;
  if (ka->n_src != kb->n_src) return false;
  for (int i = 0; i < ka->n_src; i++)
    if (ka->src[i] != kb->src[i]) return false;
  if (!poly_arg_eq(ka->arg, kb->arg)) return false;
  if (ka->tag != kb->tag) return false;
  if (!poly_arg_eq(ka->tag_arg, kb->tag_arg)) return false;
  return true;
}

typedef struct {
  PolyMap *live;
  CseKey **keys;
  PolyUOp **uops;
  uint32_t *hashes;
  size_t count;
} CseEvictionRows;

static void collect_dead_cse_row(const void *key, void *value, void *userdata) {
  CseEvictionRows *rows = userdata;
  if (!rows || !key || !value || poly_map_get(rows->live, poly_ptr_hash(value), value, poly_ptr_eq))
    return;
  rows->keys[rows->count] = (CseKey *)key;
  rows->uops[rows->count] = value;
  rows->hashes[rows->count++] = cse_hash(key);
}

int poly_uop_cse_evict_unmarked(PolyCtx *ctx, PolyMap *live) {
  /* Tinygrad 2026-08-22/a9069c177a9d UOpMetaClass.ucache weakly interns
   * UOps (uop/ops.py:186-202,238-245). Remove each dead weak row and its
   * non-moving C record at the same outer safe point. */
  if (!ctx || !ctx->cse || !live) return -1;
  size_t capacity = poly_map_len(ctx->cse);
  CseKey **keys = capacity ? malloc(capacity * sizeof(*keys)) : NULL;
  PolyUOp **uops = capacity ? malloc(capacity * sizeof(*uops)) : NULL;
  uint32_t *hashes = capacity ? malloc(capacity * sizeof(*hashes)) : NULL;
  if (capacity && (!keys || !uops || !hashes)) {
    free(keys);
    free(uops);
    free(hashes);
    return -1;
  }
  CseEvictionRows rows = {.live = live, .keys = keys, .uops = uops, .hashes = hashes};
  poly_map_foreach(ctx->cse, collect_dead_cse_row, &rows);
  for (size_t i = 0; i < rows.count; i++) {
    poly_map_remove(ctx->cse, rows.hashes[i], rows.keys[i], cse_eq);
    uop_storage_free(ctx, rows.uops[i]);
  }
  free(keys);
  free(uops);
  free(hashes);
  return 0;
}

/* struct PolyCtx and lifecycle are in ctx.h / ctx.c */

/* UOp creation with CSE */

static bool rank_tuple_valid(const void *data, int n) {
  return n >= 0 && n <= POLY_MAX_DIMS && (n == 0 || data != NULL);
}

static bool shape_value_rank_valid(PolyUOp *shape) {
  if (!shape) return false;
  int rank = shape->op == POLY_OP_STACK ? shape->n_src : 1;
  return rank >= 0 && rank <= POLY_MAX_DIMS;
}

static bool string_tuple_valid(const char **vals, int n) {
  if (n < 0 || n > UINT16_MAX || (n > 0 && !vals)) return false;
  for (int i = 0; i < n; i++)
    if (!vals[i] || !vals[i][0]) return false;
  return true;
}

static bool opt_valid(const PolyOpt *opt) {
  if (!opt || opt->op < POLY_OPT_TC || opt->op > POLY_OPT_SWAP ||
      opt->arg_kind < POLY_OPT_ARG_NONE || opt->arg_kind > POLY_OPT_ARG_INT_TUPLE)
    return false;
  return opt->arg_kind != POLY_OPT_ARG_INT_TUPLE ||
         (opt->n_arg_tuple >= 0 && (opt->n_arg_tuple == 0 || opt->arg_tuple));
}

static bool kernel_info_valid(const PolyKernelInfo *info) {
  if (!info || !info->name || !info->name[0] || info->beam < 0 || info->n_axis_types < 0 ||
      info->n_applied_opts < 0 || info->n_opts_to_apply < 0 ||
      (info->n_axis_types > 0 && !info->axis_types) ||
      (info->n_applied_opts > 0 && !info->applied_opts) ||
      (info->n_opts_to_apply > 0 && !info->opts_to_apply) ||
      (!info->has_opts_to_apply && info->n_opts_to_apply != 0))
    return false;
  for (int i = 0; i < info->n_axis_types; i++)
    if (info->axis_types[i] < POLY_AXIS_DEVICE || info->axis_types[i] > POLY_AXIS_LOOP)
      return false;
  for (int i = 0; i < info->n_applied_opts; i++)
    if (!opt_valid(&info->applied_opts[i])) return false;
  for (int i = 0; i < info->n_opts_to_apply; i++)
    if (!opt_valid(&info->opts_to_apply[i])) return false;
  return true;
}

static bool uop_rank_arg_valid(PolyOps op, PolyUOp **src, int n_src, PolyArg arg) {
  switch (op) {
  case POLY_OP_SINK:
    return arg.kind == POLY_ARG_NONE ||
           (arg.kind == POLY_ARG_KERNEL_INFO && kernel_info_valid(arg.kernel_info));
  case POLY_OP_DEVICE:
    return arg.kind == POLY_ARG_NONE || (arg.kind == POLY_ARG_STRING && arg.str && arg.str[0]) ||
           (arg.kind == POLY_ARG_STRING_TUPLE &&
            string_tuple_valid(arg.string_tuple.vals, arg.string_tuple.n));
  case POLY_OP_BUFFER:
    /* tinygrad@2026-08-22/a9069c177a9d UOp.new_buffer stores shape as the
     * sole source and device in ParamArg. Polygrad's portable BUFFER keeps
     * one UNIQUE source; the retired BUFFER(UNIQUE, DEVICE) form is invalid. */
    if (arg.kind != POLY_ARG_PARAM) return n_src == 1 && src;
    if (!arg.param || n_src != 1 || !src || arg.param->n_devices < 0) return false;
    if (arg.param->device_is_tuple)
      return !arg.param->device && string_tuple_valid(arg.param->devices, arg.param->n_devices);
    return arg.param->n_devices == 0 && !arg.param->devices;
  case POLY_OP_COPY:
    return n_src == 1 && src &&
           (arg.kind == POLY_ARG_STRING ||
            (arg.kind == POLY_ARG_STRING_TUPLE &&
             string_tuple_valid(arg.string_tuple.vals, arg.string_tuple.n)));
  case POLY_OP_RESHAPE:
    /* Pinned spec.py accepts any shape-value UOp in src[1]. UOp.as_shape
     * decodes CONST, STACK, and scalar symbolic expressions (ops.py:697-700). */
    return arg.kind == POLY_ARG_NONE && n_src == 2 && src && shape_value_rank_valid(src[1]);
  case POLY_OP_PERMUTE:
    return arg.kind == POLY_ARG_INT_TUPLE && rank_tuple_valid(arg.int_tuple.vals, arg.int_tuple.n);
  case POLY_OP_FLIP:
    if (arg.kind != POLY_ARG_INT_TUPLE || !rank_tuple_valid(arg.int_tuple.vals, arg.int_tuple.n))
      return false;
    for (int i = 0; i < arg.int_tuple.n; i++)
      if (arg.int_tuple.vals[i] != 0 && arg.int_tuple.vals[i] != 1) return false;
    return true;
  case POLY_OP_EXPAND:
    return arg.kind == POLY_ARG_NONE && n_src == 2 && src && shape_value_rank_valid(src[1]);
  case POLY_OP_SHRINK:
    return arg.kind == POLY_ARG_NONE && n_src == 3 && src && shape_value_rank_valid(src[1]) &&
           shape_value_rank_valid(src[2]);
  case POLY_OP_PAD:
    return arg.kind == POLY_ARG_NONE && n_src == 3 && src && shape_value_rank_valid(src[1]) &&
           shape_value_rank_valid(src[2]);
  case POLY_OP_UNSHARD:
    /* tinygrad@2026-08-22/a9069c177a9d uop/spec.py:180-182 requires integer
     * axis labels, but does not require nonnegative or sorted labels. */
    if (arg.kind != POLY_ARG_INT_TUPLE || arg.int_tuple.n <= 0 ||
        !rank_tuple_valid(arg.int_tuple.vals, arg.int_tuple.n) || n_src != arg.int_tuple.n + 1 ||
        !src)
      return false;
    for (int i = 0; i < arg.int_tuple.n; i++) {
      if (!src[i + 1] || !poly_dtype_is_int(src[i + 1]->dtype)) return false;
    }
    return true;
  case POLY_OP_REDUCE:
    return arg.kind == POLY_ARG_REDUCE && arg.reduce.num_axes >= 0 && n_src >= 1;
  case POLY_OP_ALLREDUCE:
    if (arg.kind != POLY_ARG_ALLREDUCE || n_src != 1 || !src) return false;
    if (arg.allreduce.device_is_tuple)
      return !arg.allreduce.device &&
             string_tuple_valid(arg.allreduce.devices, arg.allreduce.n_devices);
    return arg.allreduce.device && arg.allreduce.device[0] && !arg.allreduce.devices &&
           arg.allreduce.n_devices == 0;
  case POLY_OP_PARAM:
    if (arg.kind != POLY_ARG_PARAM) return true;
    if (!arg.param || arg.param->n_devices < 0) return false;
    if (arg.param->device_is_tuple)
      return !arg.param->device && string_tuple_valid(arg.param->devices, arg.param->n_devices);
    return arg.param->n_devices == 0 && !arg.param->devices;
  default:
    return true;
  }
}

static bool poly_arg_canonicalize_bigint(PolyArg *arg) {
  if (!arg || arg->kind != POLY_ARG_BIGINT) return true;
  if (arg->bigint.n_limbs > 0 && !arg->bigint.limbs) return false;
  while (arg->bigint.n_limbs > 0 && arg->bigint.limbs[arg->bigint.n_limbs - 1] == 0)
    arg->bigint.n_limbs--;
  if (arg->bigint.n_limbs == 0) {
    *arg = poly_arg_int(0);
    return true;
  }
  arg->bigint.sign = arg->bigint.sign < 0 ? -1 : 1;
  if (arg->bigint.n_limbs > 2) return true;
  uint64_t magnitude = arg->bigint.limbs[0];
  if (arg->bigint.n_limbs == 2) magnitude |= (uint64_t)arg->bigint.limbs[1] << 32;
  if (arg->bigint.sign > 0 && magnitude <= INT64_MAX) {
    *arg = poly_arg_int((int64_t)magnitude);
  } else if (arg->bigint.sign < 0 && magnitude <= (UINT64_C(1) << 63)) {
    *arg = poly_arg_int(magnitude == (UINT64_C(1) << 63) ? INT64_MIN : -(int64_t)magnitude);
  }
  return true;
}

static bool poly_arg_copy_to_storage(PolyUOpStorage *storage, PolyArg *dst) {
  if (!storage || !dst) return false;
  if (dst->kind == POLY_ARG_INT_TUPLE && dst->int_tuple.n > 0) {
    int64_t *vals =
        uop_storage_alloc(storage, dst->int_tuple.n * sizeof(int64_t), _Alignof(int64_t));
    if (!vals) return false;
    memcpy(vals, dst->int_tuple.vals, dst->int_tuple.n * sizeof(int64_t));
    dst->int_tuple.vals = vals;
  } else if (dst->kind == POLY_ARG_BIGINT && dst->bigint.n_limbs > 0 && dst->bigint.limbs) {
    uint32_t *limbs = uop_storage_alloc(
        storage, (size_t)dst->bigint.n_limbs * sizeof(uint32_t), _Alignof(uint32_t)
    );
    if (!limbs) return false;
    memcpy(limbs, dst->bigint.limbs, (size_t)dst->bigint.n_limbs * sizeof(uint32_t));
    dst->bigint.limbs = limbs;
  } else if (dst->kind == POLY_ARG_RANGE && dst->range.n_extra > 0) {
    int64_t *extra =
        uop_storage_alloc(storage, (size_t)dst->range.n_extra * sizeof(int64_t), _Alignof(int64_t));
    if (!extra) return false;
    memcpy(extra, dst->range.extra, (size_t)dst->range.n_extra * sizeof(int64_t));
    dst->range.extra = extra;
  } else if (dst->kind == POLY_ARG_STRING && dst->str) {
    size_t len = strlen(dst->str);
    char *s = uop_storage_alloc(storage, len + 1, 1);
    if (!s) return false;
    memcpy(s, dst->str, len + 1);
    dst->str = s;
  } else if (dst->kind == POLY_ARG_STRING_TUPLE && dst->string_tuple.n > 0) {
    const char **vals = uop_storage_alloc(
        storage, (size_t)dst->string_tuple.n * sizeof(*vals), _Alignof(const char *)
    );
    if (!vals) return false;
    for (int i = 0; i < dst->string_tuple.n; i++) {
      size_t len = strlen(dst->string_tuple.vals[i]);
      char *value = uop_storage_alloc(storage, len + 1, 1);
      if (!value) return false;
      memcpy(value, dst->string_tuple.vals[i], len + 1);
      vals[i] = value;
    }
    dst->string_tuple.vals = vals;
  } else if (dst->kind == POLY_ARG_ALLREDUCE && dst->allreduce.device_is_tuple && dst->allreduce.n_devices > 0) {
    const char **vals = uop_storage_alloc(
        storage, (size_t)dst->allreduce.n_devices * sizeof(*vals), _Alignof(const char *)
    );
    if (!vals) return false;
    for (int i = 0; i < dst->allreduce.n_devices; i++) {
      size_t len = strlen(dst->allreduce.devices[i]);
      char *value = uop_storage_alloc(storage, len + 1, 1);
      if (!value) return false;
      memcpy(value, dst->allreduce.devices[i], len + 1);
      vals[i] = value;
    }
    dst->allreduce.devices = vals;
  } else if (dst->kind == POLY_ARG_ALLREDUCE && dst->allreduce.device) {
    size_t len = strlen(dst->allreduce.device);
    char *value = uop_storage_alloc(storage, len + 1, 1);
    if (!value) return false;
    memcpy(value, dst->allreduce.device, len + 1);
    dst->allreduce.device = value;
  } else if (dst->kind == POLY_ARG_BUFFERIZE_OPTS &&
             dst->bufferize_opts.device_is_tuple && dst->bufferize_opts.n_devices > 0) {
    const char **vals = uop_storage_alloc(
        storage, (size_t)dst->bufferize_opts.n_devices * sizeof(*vals), _Alignof(const char *)
    );
    if (!vals) return false;
    for (int i = 0; i < dst->bufferize_opts.n_devices; i++) {
      size_t len = strlen(dst->bufferize_opts.devices[i]);
      char *value = uop_storage_alloc(storage, len + 1, 1);
      if (!value) return false;
      memcpy(value, dst->bufferize_opts.devices[i], len + 1);
      vals[i] = value;
    }
    dst->bufferize_opts.devices = vals;
  } else if (dst->kind == POLY_ARG_BUFFERIZE_OPTS && dst->bufferize_opts.device) {
    size_t len = strlen(dst->bufferize_opts.device);
    char *s = uop_storage_alloc(storage, len + 1, 1);
    if (!s) return false;
    memcpy(s, dst->bufferize_opts.device, len + 1);
    dst->bufferize_opts.device = s;
  } else if (dst->kind == POLY_ARG_TENSOR_CORE) {
    if (dst->tensor_core.device) {
      size_t len = strlen(dst->tensor_core.device);
      char *s = uop_storage_alloc(storage, len + 1, 1);
      if (!s) return false;
      memcpy(s, dst->tensor_core.device, len + 1);
      dst->tensor_core.device = s;
    }
    for (int d = 0; d < 3; d++) {
      int n = dst->tensor_core.n_upcast_axes[d];
      if (n <= 0) continue;
      int64_t(*pairs)[2] =
          uop_storage_alloc(storage, (size_t)n * sizeof(*pairs), _Alignof(int64_t));
      if (!pairs) return false;
      memcpy(pairs, dst->tensor_core.upcast_axes[d], (size_t)n * sizeof(*pairs));
      dst->tensor_core.upcast_axes[d] = pairs;
    }
  } else if (dst->kind == POLY_ARG_BYTES && dst->bytes.n > 0 && dst->bytes.data) {
    uint8_t *data = uop_storage_alloc(storage, (size_t)dst->bytes.n, 1);
    if (!data) return false;
    memcpy(data, dst->bytes.data, (size_t)dst->bytes.n);
    dst->bytes.data = data;
  } else if (dst->kind == POLY_ARG_PARAM && dst->param) {
    PolyParamArg *param = uop_storage_alloc(storage, sizeof(*param), _Alignof(PolyParamArg));
    if (!param) return false;
    *param = *dst->param;
    if (param->name) {
      size_t len = strlen(param->name);
      char *name = uop_storage_alloc(storage, len + 1, 1);
      if (!name) return false;
      memcpy(name, param->name, len + 1);
      param->name = name;
    }
    if (param->device) {
      size_t len = strlen(param->device);
      char *device = uop_storage_alloc(storage, len + 1, 1);
      if (!device) return false;
      memcpy(device, param->device, len + 1);
      param->device = device;
    }
    if (param->n_devices > 0) {
      const char **devices = uop_storage_alloc(
          storage, (size_t)param->n_devices * sizeof(*devices), _Alignof(const char *)
      );
      if (!devices) return false;
      for (int i = 0; i < param->n_devices; i++) {
        size_t len = strlen(param->devices[i]);
        char *device = uop_storage_alloc(storage, len + 1, 1);
        if (!device) return false;
        memcpy(device, param->devices[i], len + 1);
        devices[i] = device;
      }
      param->devices = devices;
    }
    dst->param = param;
  } else if (dst->kind == POLY_ARG_CALL_INFO && dst->call_info) {
    PolyCallInfo *info = uop_storage_alloc(storage, sizeof(*info), _Alignof(PolyCallInfo));
    if (!info) return false;
    *info = *dst->call_info;
    if (info->name) {
      size_t len = strlen(info->name);
      char *name = uop_storage_alloc(storage, len + 1, 1);
      if (!name) return false;
      memcpy(name, info->name, len + 1);
      info->name = name;
    }
    dst->call_info = info;
  } else if (dst->kind == POLY_ARG_PROGRAM_INFO && dst->program_info) {
    const PolyProgramInfo *src = dst->program_info;
    if (src->n_vars < 0 || src->n_globals < 0 || src->n_outs < 0 || src->n_ins < 0 ||
        (src->n_vars > 0 && !src->vars) || (src->n_globals > 0 && !src->globals) ||
        (src->n_outs > 0 && !src->outs) || (src->n_ins > 0 && !src->ins))
      return false;
    PolyProgramInfo *info = uop_storage_alloc(storage, sizeof(*info), _Alignof(PolyProgramInfo));
    if (!info) return false;
    *info = *src;
    if (src->name) {
      size_t len = strlen(src->name);
      char *name = uop_storage_alloc(storage, len + 1, 1);
      if (!name) return false;
      memcpy(name, src->name, len + 1);
      info->name = name;
    }
    if (src->target) {
      size_t len = strlen(src->target);
      char *target = uop_storage_alloc(storage, len + 1, 1);
      if (!target) return false;
      memcpy(target, src->target, len + 1);
      info->target = target;
    }
    if (src->n_vars > 0) {
      info->vars = uop_storage_alloc(
          storage, (size_t)src->n_vars * sizeof(*info->vars), _Alignof(PolyUOp *)
      );
      if (!info->vars) return false;
      memcpy(info->vars, src->vars, (size_t)src->n_vars * sizeof(*info->vars));
    }
    const int *source_arrays[3] = {src->globals, src->outs, src->ins};
    int counts[3] = {src->n_globals, src->n_outs, src->n_ins};
    int **target_arrays[3] = {&info->globals, &info->outs, &info->ins};
    for (int field = 0; field < 3; field++) {
      if (counts[field] <= 0) continue;
      *target_arrays[field] =
          uop_storage_alloc(storage, (size_t)counts[field] * sizeof(int), _Alignof(int));
      if (!*target_arrays[field]) return false;
      memcpy(*target_arrays[field], source_arrays[field], (size_t)counts[field] * sizeof(int));
    }
    dst->program_info = info;
  } else if (dst->kind == POLY_ARG_KERNEL_INFO && dst->kernel_info) {
    const PolyKernelInfo *src = dst->kernel_info;
    PolyKernelInfo *info = uop_storage_alloc(storage, sizeof(*info), _Alignof(PolyKernelInfo));
    if (!info) return false;
    *info = *src;
    if (src->name) {
      size_t len = strlen(src->name);
      char *name = uop_storage_alloc(storage, len + 1, 1);
      if (!name) return false;
      memcpy(name, src->name, len + 1);
      info->name = name;
    }
    if (src->n_axis_types > 0) {
      PolyAxisType *axis_types = uop_storage_alloc(
          storage, (size_t)src->n_axis_types * sizeof(*axis_types), _Alignof(PolyAxisType)
      );
      if (!axis_types) return false;
      memcpy(axis_types, src->axis_types, (size_t)src->n_axis_types * sizeof(*axis_types));
      info->axis_types = axis_types;
    }
    const PolyOpt *option_sets[2] = {src->applied_opts, src->opts_to_apply};
    int option_counts[2] = {src->n_applied_opts, src->n_opts_to_apply};
    const PolyOpt **destinations[2] = {&info->applied_opts, &info->opts_to_apply};
    for (int set = 0; set < 2; set++) {
      if (option_counts[set] <= 0) continue;
      PolyOpt *opts =
          uop_storage_alloc(storage, (size_t)option_counts[set] * sizeof(*opts), _Alignof(PolyOpt));
      if (!opts) return false;
      memcpy(opts, option_sets[set], (size_t)option_counts[set] * sizeof(*opts));
      for (int i = 0; i < option_counts[set]; i++) {
        if (opts[i].arg_kind != POLY_OPT_ARG_INT_TUPLE || opts[i].n_arg_tuple <= 0) continue;
        int64_t *tuple = uop_storage_alloc(
            storage, (size_t)opts[i].n_arg_tuple * sizeof(*tuple), _Alignof(int64_t)
        );
        if (!tuple) return false;
        memcpy(tuple, opts[i].arg_tuple, (size_t)opts[i].n_arg_tuple * sizeof(*tuple));
        opts[i].arg_tuple = tuple;
      }
      *destinations[set] = opts;
    }
    if (src->estimates) {
      PolyEstimates *estimates =
          uop_storage_alloc(storage, sizeof(*estimates), _Alignof(PolyEstimates));
      if (!estimates) return false;
      *estimates = *src->estimates;
      info->estimates = estimates;
    }
    dst->kernel_info = info;
  }
  return true;
}

static PolyUOp *poly_uop_internal(
    PolyCtx *ctx,
    PolyOps op,
    PolyDType dtype,
    PolyUOp **src,
    int n_src,
    PolyArg arg,
    int32_t tag,
    PolyArg tag_arg
) {
  if (!ctx || n_src < 0 || n_src > UINT16_MAX || (n_src > 0 && !src)) return NULL;
  for (int i = 0; i < n_src; i++)
    if (!src[i]) return NULL;
  /* Python ints have one value identity regardless of construction history.
   * Canonicalize the C carrier before hashing so signed-64 values cannot
   * acquire a second BIGINT CSE identity. */
  if (!poly_arg_canonicalize_bigint(&arg) || !poly_arg_canonicalize_bigint(&tag_arg)) return NULL;
  /* The integer arm owns no string/tuple storage. Reject ambiguous C carriers
   * before hashing or copying; Python's union value cannot represent them. */
  if (arg.kind == POLY_ARG_BUFFERIZE_OPTS && arg.bufferize_opts.device_is_int &&
      (arg.bufferize_opts.device_is_tuple || arg.bufferize_opts.device ||
       arg.bufferize_opts.devices || arg.bufferize_opts.n_devices))
    return NULL;
  /* Current tinygrad stores the target DType as CAST/BITCAST.arg and derives
   * the node dtype from it (uop/ops.py:169-172, mixin/dtype.py:16-50).
   * Keep accepting the old internal NONE spelling only at this constructor
   * boundary so every newly interned UOp has the current exact identity. */
  if (op == POLY_OP_CAST || op == POLY_OP_BITCAST) {
    if (arg.kind == POLY_ARG_NONE) arg = poly_arg_dtype(dtype);
    if (n_src != 1 || arg.kind != POLY_ARG_DTYPE || !poly_dtype_eq(arg.dtype, dtype)) return NULL;
  }

  PolyParamArg normalized_param;
  if (arg.kind == POLY_ARG_PARAM && arg.param) {
    normalized_param = *arg.param;
    if (normalized_param.dtype.bitsize == 0) normalized_param.dtype = dtype;
    arg.param = &normalized_param;
  }

  if (!uop_rank_arg_valid(op, src, n_src, arg)) return NULL;

  /* Build a CSE key on the stack */
  CseKey key = {op, dtype, src, (uint16_t)n_src, arg, tag, tag_arg};
  uint32_t h = cse_hash(&key);

  /* tinygrad@2026-08-22/a9069c177a9d uop/ops.py:186-202 interns every
   * (op,dtype,src,arg,tag), including INS. Tags distinguish occurrences. */
  PolyUOp *existing = poly_map_get(ctx->cse, h, &key, cse_eq);
  if (existing) return existing;

  /* C ownership mechanics for Tinygrad's weak UOp objects: one non-moving
   * record owns the node, ordered sources, and CSE key. */
  PolyUOpStorage *storage = uop_storage_new(ctx, n_src);
  if (!storage) return NULL;
  PolyUOp *u = &storage->uop;
  u->op = op;
  u->dtype = dtype;
  u->n_src = (uint16_t)n_src;
  u->arg = arg;
  u->tag = tag;
  u->tag_arg = tag_arg;
  u->hash = h;
  u->addrspace_cached = false;
  u->addrspace_cache = POLY_ADDR_GLOBAL;
  u->minmax_cached = false;
  u->minmax_vmin = 0;
  u->minmax_vmax = 0;
  u->ranges_cache = NULL;
  u->ended_ranges_cache = NULL;

  /* Copy src pointers into arena */
  if (n_src > 0) {
    u->src = storage->src;
    memcpy(u->src, src, n_src * sizeof(PolyUOp *));
  } else {
    u->src = NULL;
  }

  /* Tinygrad UOps own immutable arg/tag objects. Keep their C payloads in the
   * same weak record so the mark/sweep lifetime is identical. */
  if (!poly_arg_copy_to_storage(storage, &u->arg) ||
      !poly_arg_copy_to_storage(storage, &u->tag_arg)) {
    uop_storage_dispose(storage);
    return NULL;
  }
  cache_uop_addrspace(u);

  /* The key has exactly the UOp's weak lifetime. */
  CseKey *stored_key = &storage->key;
  *stored_key = (CseKey){op, dtype, u->src, (uint16_t)n_src, u->arg, tag, u->tag_arg};

  uop_storage_register(ctx, storage);
  poly_map_set(ctx->cse, h, stored_key, u, cse_eq);
  return u;
}

PolyUOp *poly_uop(
    PolyCtx *ctx,
    PolyOps op,
    PolyDType dtype,
    PolyUOp **src,
    int n_src,
    PolyArg arg
) {
  return poly_uop_internal(ctx, op, dtype, src, n_src, arg, 0, poly_arg_none());
}

/* Current tinygrad UOp.unshard (`tinygrad/uop/ops.py:667-681`). C callers
 * pass sorted axes explicitly; each axis keeps its sharding expression. */
PolyUOp *poly_unshard(
    PolyCtx *ctx,
    PolyUOp *value,
    const int64_t *axes,
    PolyUOp **ranges,
    int n_axes
) {
  if (!ctx || !value || !axes || !ranges || n_axes <= 0 || n_axes > POLY_MAX_DIMS) return NULL;
  PolyUOp *src[POLY_MAX_DIMS + 1];
  int64_t sorted_axes[POLY_MAX_DIMS];
  PolyUOp *sorted_ranges[POLY_MAX_DIMS];
  for (int i = 0; i < n_axes; i++) {
    if (axes[i] < 0 || axes[i] >= POLY_MAX_DIMS || !ranges[i] ||
        !poly_dtype_is_int(ranges[i]->dtype))
      return NULL;
    int pos = i;
    while (pos > 0 && sorted_axes[pos - 1] > axes[i]) {
      sorted_axes[pos] = sorted_axes[pos - 1];
      sorted_ranges[pos] = sorted_ranges[pos - 1];
      pos--;
    }
    if (pos > 0 && sorted_axes[pos - 1] == axes[i]) return NULL;
    sorted_axes[pos] = axes[i];
    sorted_ranges[pos] = ranges[i];
  }
  src[0] = value;
  for (int i = 0; i < n_axes; i++)
    src[i + 1] = sorted_ranges[i];
  return poly_uop(
      ctx, POLY_OP_UNSHARD, value->dtype, src, n_axes + 1, poly_arg_int_tuple(sorted_axes, n_axes)
  );
}

/* Current tinygrad UOp.allreduce (`tinygrad/uop/ops.py:654-656`). */
PolyUOp *poly_allreduce(PolyCtx *ctx, PolyUOp *value, PolyOps op, PolyUOp *device) {
  if (!ctx || !value || !device || device->op != POLY_OP_DEVICE) return NULL;
  PolyArg arg;
  if (device->arg.kind == POLY_ARG_STRING_TUPLE)
    arg = poly_arg_allreduce(op, NULL, device->arg.string_tuple.vals, device->arg.string_tuple.n);
  else if (device->arg.kind == POLY_ARG_STRING)
    arg = poly_arg_allreduce(op, device->arg.str, NULL, 0);
  else
    return NULL;
  return poly_uop1(ctx, POLY_OP_ALLREDUCE, value->dtype, value, arg);
}

/* Current tinygrad UOp.range (`tinygrad/uop/ops.py:563-565`). */
PolyUOp *poly_range(PolyCtx *ctx, int64_t bound, int64_t axis_id, PolyAxisType axis_type) {
  if (!ctx || bound < 0 || axis_type < POLY_AXIS_DEVICE || axis_type > POLY_AXIS_LOOP) return NULL;
  PolyUOp *bound_uop = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(bound));
  return bound_uop
             ? poly_uop1(
                   ctx, POLY_OP_RANGE, POLY_WEAKINT, bound_uop, poly_arg_range(axis_id, axis_type)
               )
             : NULL;
}

PolyUOp *poly_uop_tagged(
    PolyCtx *ctx,
    PolyOps op,
    PolyDType dtype,
    PolyUOp **src,
    int n_src,
    PolyArg arg,
    int32_t tag
) {
  return poly_uop_internal(ctx, op, dtype, src, n_src, arg, tag, poly_arg_none());
}

PolyUOp *poly_uop_tagged_arg(
    PolyCtx *ctx,
    PolyOps op,
    PolyDType dtype,
    PolyUOp **src,
    int n_src,
    PolyArg arg,
    int32_t tag,
    PolyArg tag_arg
) {
  return poly_uop_internal(ctx, op, dtype, src, n_src, arg, tag, tag_arg);
}

/* Current Tinygrad uop/ops.py:UOp.replace(src=...). */
PolyUOp *poly_uop_replace_src(PolyCtx *ctx, PolyUOp *u, PolyUOp **src) {
  if (!ctx || !u || (u->n_src > 0 && !src)) return NULL;
  return (u->tag != 0 || u->tag_arg.kind != POLY_ARG_NONE)
             ? poly_uop_tagged_arg(ctx, u->op, u->dtype, src, u->n_src, u->arg, u->tag, u->tag_arg)
             : poly_uop(ctx, u->op, u->dtype, src, u->n_src, u->arg);
}

PolyUOp *poly_uop0(PolyCtx *ctx, PolyOps op, PolyDType dtype, PolyArg arg) {
  return poly_uop(ctx, op, dtype, NULL, 0, arg);
}

PolyUOp *poly_uop1(PolyCtx *ctx, PolyOps op, PolyDType dtype, PolyUOp *s0, PolyArg arg) {
  PolyUOp *src[] = {s0};
  return poly_uop(ctx, op, dtype, src, 1, arg);
}

PolyUOp *poly_shape_to_shape_arg(PolyCtx *ctx, PolyUOp **items, int n_items) {
  /* Exact C port of current tinygrad uop/ops.py:shape_to_shape_arg: a sole
   * dimension is the shape argument itself; all other ranks use STACK. */
  if (!ctx || !rank_tuple_valid(items, n_items)) return NULL;
  for (int i = 0; i < n_items; i++)
    if (!items[i] || !poly_dtype_is_int(items[i]->dtype)) return NULL;
  if (n_items == 1) return items[0];
  return poly_uop(
      ctx, POLY_OP_STACK, n_items == 0 ? POLY_VOID : POLY_WEAKINT, items, n_items, poly_arg_none()
  );
}

int poly_broadcast_shape(PolyCtx *ctx, PolyUOp **src, int n_src, PolyUOp **out_dims, int max_dims) {
  if (!ctx || !src || n_src <= 0 || !out_dims || max_dims < 0) return -1;
  int ndim = 0;
  for (int i = 0; i < n_src; i++) {
    if (!src[i]) return -1;
    int source_ndim = poly_uop_ndim(ctx, src[i]);
    if (source_ndim < 0 || source_ndim > max_dims) return -1;
    if (source_ndim > ndim) ndim = source_ndim;
  }

  for (int axis = 0; axis < ndim; axis++) {
    PolyUOp *selected = NULL;
    int64_t selected_value = 0;
    bool selected_is_const = false;
    for (int i = 0; i < n_src; i++) {
      int source_ndim = poly_uop_ndim(ctx, src[i]);
      int source_axis = axis - (ndim - source_ndim);
      PolyUOp *dim = source_axis >= 0
                         ? poly_uop_shape_dim(ctx, src[i], source_axis)
                         : poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
      int64_t value = 0;
      bool is_const = dim && poly_uop_const_i64(dim, &value) == 0;
      if (!dim) return -1;
      if (is_const && value == 1) continue;
      if (!selected) {
        selected = dim;
        selected_value = value;
        selected_is_const = is_const;
      } else if (selected != dim && !(selected_is_const && is_const && selected_value == value)) {
        return -1;
      }
    }
    out_dims[axis] =
        selected ? selected : poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
  }
  return ndim;
}

int poly_uop_as_shape(PolyCtx *ctx, PolyUOp *shape_arg, PolyUOp **items, int max_items) {
  if (!ctx || !shape_arg || max_items < 0) return -1;
  int n = shape_arg->op == POLY_OP_STACK ? shape_arg->n_src : 1;
  if (n < 0 || n > max_items || (n > 0 && !items)) return -1;
  for (int i = 0; i < n; i++) {
    PolyUOp *item = shape_arg->op == POLY_OP_STACK ? shape_arg->src[i] : shape_arg;
    if (!item || !poly_dtype_is_int(item->dtype)) return -1;
    /* UOp.as_shape returns CONST.val directly; only symbolic lanes need
     * ssimplify. Avoid a rewriter traversal for every static shape bound. */
    items[i] = item->op == POLY_OP_CONST ? item : poly_graph_rewrite(ctx, item, poly_symbolic());
    if (!items[i]) return -1;
  }
  return n;
}

PolyUOp *poly_uop2(
    PolyCtx *ctx,
    PolyOps op,
    PolyDType dtype,
    PolyUOp *s0,
    PolyUOp *s1,
    PolyArg arg
) {
  PolyUOp *src[] = {s0, s1};
  return poly_uop(ctx, op, dtype, src, 2, arg);
}

PolyUOp *poly_uop3(
    PolyCtx *ctx,
    PolyOps op,
    PolyDType dtype,
    PolyUOp *s0,
    PolyUOp *s1,
    PolyUOp *s2,
    PolyArg arg
) {
  PolyUOp *src[] = {s0, s1, s2};
  return poly_uop(ctx, op, dtype, src, 3, arg);
}

/* Exact C port of current UOp._rop (tinygrad/uop/ops.py:630-638).
 * Public reduction axes are sorted, proved-singleton axes are skipped, actual
 * reduction axes are moved to the prefix, and REDUCE stores only their count. */
PolyUOp *poly_reduce_axis(
    PolyCtx *ctx,
    PolyOps reduce_op,
    PolyUOp *src,
    int64_t *axes,
    int n_axes
) {
  if (!ctx || !src || !rank_tuple_valid(axes, n_axes)) return NULL;
  if (n_axes == 0) return src;

  int ndim = poly_uop_ndim(ctx, src);
  if (ndim < 0 || ndim > POLY_MAX_DIMS) return NULL;
  int64_t requested[POLY_MAX_DIMS];
  for (int i = 0; i < n_axes; i++) {
    int64_t axis = axes[i] < 0 ? axes[i] + ndim : axes[i];
    if (axis < 0 || axis >= ndim) return NULL;
    requested[i] = axis;
  }
  for (int i = 1; i < n_axes; i++) {
    int64_t axis = requested[i];
    int j = i - 1;
    while (j >= 0 && requested[j] > axis) {
      requested[j + 1] = requested[j];
      j--;
    }
    requested[j + 1] = axis;
  }
  for (int i = 1; i < n_axes; i++)
    if (requested[i] == requested[i - 1]) return NULL;

  int64_t reduced[POLY_MAX_DIMS];
  int n_reduced = 0;
  for (int i = 0; i < n_axes; i++) {
    PolyUOp *dim = poly_uop_shape_dim(ctx, src, (int)requested[i]);
    int64_t vmin = 0, vmax = 0;
    if (!dim) return NULL;
    poly_uop_minmax(ctx, dim, &vmin, &vmax);
    if (vmin == 1 && vmax == 1) continue;
    reduced[n_reduced++] = requested[i];
  }

  PolyUOp *ret = src;
  if (n_reduced > 0) {
    int64_t perm[POLY_MAX_DIMS];
    int n_perm = 0;
    for (int i = 0; i < n_reduced; i++)
      perm[n_perm++] = reduced[i];
    for (int axis = 0; axis < ndim; axis++) {
      bool is_reduced = false;
      for (int i = 0; i < n_reduced; i++)
        if (reduced[i] == axis) {
          is_reduced = true;
          break;
        }
      if (!is_reduced) perm[n_perm++] = axis;
    }
    bool identity = n_perm == ndim;
    for (int i = 0; identity && i < ndim; i++)
      identity = perm[i] == i;
    PolyUOp *ordered = identity ? src : poly_permute(ctx, src, perm, ndim);
    ret = ordered
              ? poly_uop1(
                    ctx, POLY_OP_REDUCE, src->dtype, ordered, poly_arg_reduce(reduce_op, n_reduced)
                )
              : NULL;
    if (!ret) return NULL;
  }

  if (n_reduced != n_axes) {
    PolyUOp *shape[POLY_MAX_DIMS];
    int n_shape = 0;
    for (int axis = 0; axis < ndim; axis++) {
      bool requested_axis = false;
      for (int i = 0; i < n_axes; i++)
        if (requested[i] == axis) {
          requested_axis = true;
          break;
        }
      if (!requested_axis) {
        shape[n_shape] = poly_uop_shape_dim(ctx, src, axis);
        if (!shape[n_shape++]) return NULL;
      }
    }
    ret = poly_reshape_uop(ctx, ret, shape, n_shape);
  }
  return ret;
}

/* Toposort (iterative DFS, mirrors tinygrad's toposort) */

typedef enum {
  POLY_TOPO_RESULT_ARENA,
  POLY_TOPO_RESULT_OWNED,
  POLY_TOPO_RESULT_SCRATCH,
} PolyTopoResultStorage;

static PolyUOp **toposort_result_alloc(PolyCtx *ctx, int cap, PolyTopoResultStorage storage) {
  size_t nbytes = (size_t)cap * sizeof(PolyUOp *);
  switch (storage) {
  case POLY_TOPO_RESULT_OWNED:
    return malloc(nbytes);
  case POLY_TOPO_RESULT_SCRATCH:
    return ctx ? poly_ctx_scratch_alloc(ctx, nbytes, _Alignof(PolyUOp *)) : NULL;
  case POLY_TOPO_RESULT_ARENA:
  default:
    return ctx ? poly_arena_alloc(ctx->arena, nbytes, _Alignof(PolyUOp *)) : NULL;
  }
}

/* Shared iterative DFS worker. Either `gate_simple` or `gate_user` may be
 * non-NULL (never both). The gate signature difference is bridged here so
 * callers can use closure-style gating without another whole copy. */
static PolyUOp **toposort_worker(
    PolyCtx *ctx,
    PolyUOp *root,
    int *n_out,
    bool (*gate_simple)(PolyUOp *),
    bool (*gate_user)(PolyUOp *, void *),
    void *user_data,
    bool enter_calls,
    PolyTopoResultStorage storage
) {
  if (n_out) *n_out = 0;
  /* Owned temporary toposorts only traverse immutable UOps and need no arena.
   * This permits tinygrad-style topological symbolic evaluation in helpers
   * whose complete input is already the root UOp. */
  if (!root || !n_out || (!ctx && storage != POLY_TOPO_RESULT_OWNED)) return NULL;
  int cap = 256;
  int n = 0;
  PolyUOp **result = toposort_result_alloc(ctx, cap, storage);
  if (!result) return NULL;

  /* Visited set — pointer identity map */
  PolyMap *visited = poly_map_new(256);
  if (!visited) {
    if (storage == POLY_TOPO_RESULT_OWNED) free(result);
    return NULL;
  }

  /* DFS stack: pairs of (UOp*, state) where state 0=first visit, 1=children pushed */
  int stack_cap = 256;
  int stack_top = 0;
  PolyUOp **stack = malloc(stack_cap * sizeof(PolyUOp *));
  int *state = malloc(stack_cap * sizeof(int));
  if (!stack || !state) {
    free(stack);
    free(state);
    poly_map_destroy(visited);
    if (storage == POLY_TOPO_RESULT_OWNED) free(result);
    return NULL;
  }

  stack[stack_top] = root;
  state[stack_top] = 0;
  stack_top++;

  while (stack_top > 0) {
    PolyUOp *u = stack[stack_top - 1];
    int s = state[stack_top - 1];

    uint32_t vh = poly_ptr_hash(u);
    if (s == 0 && poly_map_get(visited, vh, u, poly_ptr_eq) != NULL) {
      stack_top--;
      continue;
    }

    if (s == 0) {
      /* Gate check: if gate returns false, skip this subtree entirely */
      bool pass = true;
      if (gate_simple && !gate_simple(u)) pass = false;
      if (gate_user && !gate_user(u, user_data)) pass = false;
      if (!pass) {
        stack_top--;
        continue;
      }
      /* First visit: push children in reverse order */
      state[stack_top - 1] = 1;
      /* CALL/FUNCTION src[0] is the opaque callee body. External arguments
       * remain ordinary graph inputs when enter_calls=false. */
      bool opaque_body = u->op == POLY_OP_CALL || u->op == POLY_OP_FUNCTION;
      int start = (!enter_calls && opaque_body && u->n_src > 0) ? 1 : 0;
      for (int i = u->n_src - 1; i >= start; i--) {
        uint32_t ch = poly_ptr_hash(u->src[i]);
        if (poly_map_get(visited, ch, u->src[i], poly_ptr_eq) != NULL) continue;
        if (stack_top >= stack_cap) {
          stack_cap *= 2;
          PolyUOp **new_stack = realloc(stack, (size_t)stack_cap * sizeof(PolyUOp *));
          int *new_state = realloc(state, (size_t)stack_cap * sizeof(int));
          if (!new_stack || !new_state) {
            free(new_stack ? new_stack : stack);
            free(new_state ? new_state : state);
            poly_map_destroy(visited);
            if (storage == POLY_TOPO_RESULT_OWNED) free(result);
            *n_out = 0;
            return NULL;
          }
          stack = new_stack;
          state = new_state;
        }
        stack[stack_top] = u->src[i];
        state[stack_top] = 0;
        stack_top++;
      }
    } else {
      /* Post-order: all children done, emit this node */
      stack_top--;
      if (poly_map_get(visited, vh, u, poly_ptr_eq) != NULL) continue;
      /* Use a non-NULL sentinel as value */
      poly_map_set(visited, vh, u, (void *)(uintptr_t)1, poly_ptr_eq);

      if (n >= cap) {
        int new_cap = cap * 2;
        PolyUOp **new_result = (storage == POLY_TOPO_RESULT_OWNED)
                                   ? realloc(result, (size_t)new_cap * sizeof(PolyUOp *))
                                   : toposort_result_alloc(ctx, new_cap, storage);
        if (!new_result) {
          free(stack);
          free(state);
          poly_map_destroy(visited);
          if (storage == POLY_TOPO_RESULT_OWNED) free(result);
          *n_out = 0;
          return NULL;
        }
        if (storage != POLY_TOPO_RESULT_OWNED)
          memcpy(new_result, result, (size_t)n * sizeof(PolyUOp *));
        result = new_result;
        cap = new_cap;
      }
      result[n++] = u;
    }
  }

  free(stack);
  free(state);
  poly_map_destroy(visited);
  *n_out = n;
  return result;
}

PolyUOp **poly_toposort_ex(
    PolyCtx *ctx,
    PolyUOp *root,
    int *n_out,
    bool (*gate)(PolyUOp *),
    bool enter_calls
) {
  return toposort_worker(ctx, root, n_out, gate, NULL, NULL, enter_calls, POLY_TOPO_RESULT_ARENA);
}

PolyUOp **poly_toposort_ex_user(
    PolyCtx *ctx,
    PolyUOp *root,
    int *n_out,
    bool (*gate)(PolyUOp *, void *),
    void *user_data,
    bool enter_calls
) {
  return toposort_worker(
      ctx, root, n_out, NULL, gate, user_data, enter_calls, POLY_TOPO_RESULT_ARENA
  );
}

PolyUOp **poly_toposort(PolyCtx *ctx, PolyUOp *root, int *n_out) {
  return toposort_worker(ctx, root, n_out, NULL, NULL, NULL, true, POLY_TOPO_RESULT_ARENA);
}

PolyUOp **poly_toposort_ex_alloc(
    PolyCtx *ctx,
    PolyUOp *root,
    int *n_out,
    bool (*gate)(PolyUOp *),
    bool enter_calls
) {
  return toposort_worker(ctx, root, n_out, gate, NULL, NULL, enter_calls, POLY_TOPO_RESULT_OWNED);
}

PolyUOp **poly_toposort_ex_user_alloc(
    PolyCtx *ctx,
    PolyUOp *root,
    int *n_out,
    bool (*gate)(PolyUOp *, void *),
    void *user_data,
    bool enter_calls
) {
  return toposort_worker(
      ctx, root, n_out, NULL, gate, user_data, enter_calls, POLY_TOPO_RESULT_OWNED
  );
}

PolyUOp **poly_toposort_alloc(PolyCtx *ctx, PolyUOp *root, int *n_out) {
  return toposort_worker(ctx, root, n_out, NULL, NULL, NULL, true, POLY_TOPO_RESULT_OWNED);
}

PolyUOp **poly_toposort_scratch(PolyCtx *ctx, PolyUOp *root, int *n_out) {
  return toposort_worker(ctx, root, n_out, NULL, NULL, NULL, true, POLY_TOPO_RESULT_SCRATCH);
}

PolyUOp **poly_toposort_ex_user_scratch(
    PolyCtx *ctx,
    PolyUOp *root,
    int *n_out,
    bool (*gate)(PolyUOp *, void *),
    void *user_data,
    bool enter_calls
) {
  return toposort_worker(
      ctx, root, n_out, NULL, gate, user_data, enter_calls, POLY_TOPO_RESULT_SCRATCH
  );
}

void poly_toposort_free(PolyUOp **topo) {
  free(topo);
}

/* Current tinygrad uop/ops.py:range_start. */
int poly_range_start(PolyOps op) {
  switch (op) {
  case POLY_OP_STAGE:
    return 1;
  case POLY_OP_REDUCE:
    return 1;
  case POLY_OP_WMMA:
    return 3;
  case POLY_OP_END:
    return 1;
  case POLY_OP_CALL:
    return 1;
  case POLY_OP_FUNCTION:
    return 1;
  case POLY_OP_LINEAR:
    return 0;
  default:
    return -1;
  }
}

/* Range helpers *
 * Ports tinygrad's `no_range` (codegen/simplify.py:75) and the
 * `_ranges`/`ranges`/`ended_ranges` triple (uop/ops.py:351-378).
 *
 * `poly_no_range(u)` = not any RANGE in u's backward slice (simple, exact).
 *
 * `poly_uop_ranges(u)` = the set of *active* RANGE UOps at u's position:
 *   ranges(u) = union(ranges(src) for src in u.src)
 *               minus ended_ranges(u)
 *               plus ({u} if u.op == RANGE else {})
 *
 * `ended_ranges(u)` follows tinygrad's semantics:
 *   - Ops in current tinygrad `range_start`: trailing sources past the
 *     recorded position are ended ranges.
 *   - AFTER: flatten(ended_ranges(src) for src in u.src[1:]) — recursive.
 *   - otherwise: empty.
 *
 * When an ended entry is not itself a RANGE (e.g. a bound expression like
 * ALU BUFFER variable, a symbolic bound, an AFTER value), tinygrad's `_ranges`
 * removes every range in that entry's `ranges` set from the result — not
 * just the entry itself. We mirror that exactly.
 *
 * Sets are represented as UOp-owned PolyUOp* arrays with linear-time
 * dedup. Typical range set sizes in realistic kernels are 0-8 elements, so
 * linear ops beat a hashmap. Computation is memoized per-UOp in a PolyMap
 * cache so a single pass-wide walk is O(N * avg_set_size). */

typedef struct PolyRangeSet {
  PolyUOpStorage *storage;
  PolyUOp **items;
  int n;
  int cap;
} PolyRangeSet;

/* These count/bool queries have no recoverable error channel. Match their
 * PolyMap's fatal allocation policy instead of returning a false empty set
 * that can silently remove reduction loops from the compiled graph. */
static _Noreturn void range_cache_failed(void) {
  fprintf(stderr, "polygrad: range cache computation failed\n");
  abort();
}

static PolyRangeSet *range_set_new(PolyCtx *ctx, PolyUOp *owner, int cap) {
  PolyUOpStorage *storage = uop_storage_get(ctx, owner);
  if (!storage) return NULL;
  PolyRangeSet *s =
      uop_storage_alloc_live(ctx, storage, sizeof(PolyRangeSet), _Alignof(PolyRangeSet));
  if (!s) return NULL;
  if (cap < 4) cap = 4;
  s->items =
      uop_storage_alloc_live(ctx, storage, (size_t)cap * sizeof(PolyUOp *), _Alignof(PolyUOp *));
  if (!s->items) return NULL;
  s->storage = storage;
  s->n = 0;
  s->cap = cap;
  return s;
}

static void range_set_grow(PolyCtx *ctx, PolyRangeSet *s, int need) {
  if (need <= s->cap) return;
  int new_cap = s->cap;
  while (new_cap < need)
    new_cap = new_cap > INT_MAX / 2 ? INT_MAX : new_cap * 2;
  if ((size_t)new_cap > SIZE_MAX / sizeof(PolyUOp *)) range_cache_failed();
  PolyUOp **new_items = uop_storage_alloc_live(
      ctx, s->storage, (size_t)new_cap * sizeof(PolyUOp *), _Alignof(PolyUOp *)
  );
  if (!new_items) range_cache_failed();
  memcpy(new_items, s->items, (size_t)s->n * sizeof(PolyUOp *));
  s->items = new_items;
  s->cap = new_cap;
}

static bool range_set_contains(const PolyRangeSet *s, PolyUOp *r) {
  for (int i = 0; i < s->n; i++)
    if (s->items[i] == r) return true;
  return false;
}

static void range_set_add(PolyCtx *ctx, PolyRangeSet *s, PolyUOp *r) {
  if (range_set_contains(s, r)) return;
  if (s->n == INT_MAX) range_cache_failed();
  range_set_grow(ctx, s, s->n + 1);
  s->items[s->n++] = r;
}

static void range_set_remove(PolyRangeSet *s, PolyUOp *r) {
  for (int i = 0; i < s->n; i++) {
    if (s->items[i] == r) {
      s->items[i] = s->items[s->n - 1];
      s->n--;
      return;
    }
  }
}

static void range_set_union(PolyCtx *ctx, PolyRangeSet *dst, const PolyRangeSet *src) {
  for (int i = 0; i < src->n; i++)
    range_set_add(ctx, dst, src->items[i]);
}

static void range_set_subtract(PolyRangeSet *dst, const PolyRangeSet *src) {
  for (int i = 0; i < src->n; i++)
    range_set_remove(dst, src->items[i]);
}

static const PolyRangeSet *ranges_memo_get(PolyMap *memo, PolyUOp *u) {
  if (!u) return NULL;
  if (u->ranges_cache) return (const PolyRangeSet *)u->ranges_cache;
  return memo ? (const PolyRangeSet *)poly_map_get(memo, poly_ptr_hash(u), u, poly_ptr_eq) : NULL;
}

static const PolyRangeSet *ended_memo_get(PolyMap *memo, PolyUOp *u) {
  if (!u) return NULL;
  if (u->ended_ranges_cache) return (const PolyRangeSet *)u->ended_ranges_cache;
  return memo ? (const PolyRangeSet *)poly_map_get(memo, poly_ptr_hash(u), u, poly_ptr_eq) : NULL;
}

static void ranges_memo_set(PolyMap *memo, PolyUOp *u, PolyRangeSet *ranges) {
  if (!u || !ranges) return;
  u->ranges_cache = ranges;
  if (memo) poly_map_set(memo, poly_ptr_hash(u), u, ranges, poly_ptr_eq);
}

static void ended_memo_set(PolyMap *memo, PolyUOp *u, PolyRangeSet *ranges) {
  if (!u || !ranges) return;
  u->ended_ranges_cache = ranges;
  if (memo) poly_map_set(memo, poly_ptr_hash(u), u, ranges, poly_ptr_eq);
}

/* ended_ranges per tinygrad ops.py:351-358. Writes into `out` (caller-owned).
 * For entries that are themselves RANGE nodes we just add them; for non-RANGE
 * entries we union in their full ranges set so the caller's subtract does the
 * right thing regardless of entry shape.
 *
 * This helper assumes `ranges_memo` and `ended_memo` already contain every
 * source node, which is true because compute_ranges() processes topological
 * order. */
static bool compute_ended_ranges_node(
    PolyCtx *ctx,
    PolyUOp *u,
    PolyRangeSet *out,
    PolyMap *ranges_memo,
    PolyMap *ended_memo
) {
  int rs = poly_range_start(u->op);
  if (rs >= 0) {
    if (rs > u->n_src) rs = u->n_src;
    for (int i = rs; i < u->n_src; i++) {
      PolyUOp *er = u->src[i];
      if (er->op == POLY_OP_RANGE) {
        range_set_add(ctx, out, er);
      } else {
        const PolyRangeSet *er_r = ranges_memo_get(ranges_memo, er);
        if (!er_r) return false;
        range_set_union(ctx, out, er_r);
      }
    }
    return true;
  }
  if (u->op == POLY_OP_AFTER) {
    /* flatten(ended_ranges(x) for x in src[1:]) */
    for (int i = 1; i < u->n_src; i++) {
      const PolyRangeSet *src_ended = ended_memo_get(ended_memo, u->src[i]);
      if (!src_ended) return false;
      range_set_union(ctx, out, src_ended);
    }
    return true;
  }
  /* No ended ranges for other ops. */
  return true;
}

typedef struct {
  PolyMap *ranges;
  PolyMap *ended;
} RangeComputeGateCtx;

static bool range_compute_gate(PolyUOp *u, void *user_data) {
  RangeComputeGateCtx *g = (RangeComputeGateCtx *)user_data;
  if (!g || !u) return false;
  return !ranges_memo_get(g->ranges, u) || !ended_memo_get(g->ended, u);
}

static PolyRangeSet *compute_ranges_with_ended(
    PolyCtx *ctx,
    PolyUOp *u,
    PolyMap *ranges_memo,
    PolyMap *ended_memo
) {
  PolyRangeSet *cached = (PolyRangeSet *)ranges_memo_get(ranges_memo, u);
  if (cached) return cached;

  bool own_ended = false;
  if (!ended_memo) {
    ended_memo = poly_map_new(64);
    if (!ended_memo) return NULL;
    own_ended = true;
  }

  int n_topo = 0;
  RangeComputeGateCtx gate = {ranges_memo, ended_memo};
  PolyScratchMark scratch = poly_ctx_scratch_mark(ctx);
  PolyUOp **topo = poly_toposort_ex_user_scratch(ctx, u, &n_topo, range_compute_gate, &gate, true);
  if (!topo) {
    poly_ctx_scratch_rewind(ctx, scratch);
    if (own_ended) poly_map_destroy(ended_memo);
    return NULL;
  }

  bool ok = true;
  for (int ti = 0; ok && ti < n_topo; ti++) {
    PolyUOp *cur = topo[ti];

    PolyRangeSet *ended = (PolyRangeSet *)ended_memo_get(ended_memo, cur);
    if (!ended) {
      ended = range_set_new(ctx, cur, 4);
      if (!ended) {
        ok = false;
        break;
      }
      ok = compute_ended_ranges_node(ctx, cur, ended, ranges_memo, ended_memo);
      if (!ok) break;
      ended_memo_set(ended_memo, cur, ended);
    }

    if (ranges_memo_get(ranges_memo, cur)) continue;

    PolyRangeSet *ret = range_set_new(ctx, cur, 4);
    if (!ret) {
      ok = false;
      break;
    }
    for (int i = 0; i < cur->n_src; i++) {
      const PolyRangeSet *src_ranges = ranges_memo_get(ranges_memo, cur->src[i]);
      if (!src_ranges) {
        ok = false;
        break;
      }
      range_set_union(ctx, ret, src_ranges);
    }
    if (!ok) break;

    range_set_subtract(ret, ended);
    if (cur->op == POLY_OP_RANGE) range_set_add(ctx, ret, cur);
    ranges_memo_set(ranges_memo, cur, ret);
  }

  cached = ok ? (PolyRangeSet *)ranges_memo_get(ranges_memo, u) : NULL;
  poly_ctx_scratch_rewind(ctx, scratch);
  if (own_ended) poly_map_destroy(ended_memo);
  return cached;
}

/* PolyUOpCache: per-pass query maps for minmax + ranges *
 * Owns PolyMaps keyed by PolyUOp*. The struct is opaque in the public
 * header; callers get it via poly_uop_cache_new and pass it to any `_ex`
 * query. Range-set values share the weak UOp record lifetime. Minmax
 * values live inline on the UOp. Destroying the cache only tears down the map
 * wrappers. */

struct PolyUOpCache {
  PolyMap *ranges; /* PolyUOp* -> PolyRangeSet* */
  PolyMap *ended; /* PolyUOp* -> PolyRangeSet* for ended_ranges */
};

PolyUOpCache *poly_uop_cache_new(void) {
  PolyUOpCache *c = malloc(sizeof(PolyUOpCache));
  if (!c) return NULL;
  c->ranges = poly_map_new(64);
  c->ended = poly_map_new(64);
  if (!c->ranges || !c->ended) {
    if (c->ranges) poly_map_destroy(c->ranges);
    if (c->ended) poly_map_destroy(c->ended);
    free(c);
    return NULL;
  }
  return c;
}

void poly_uop_cache_destroy(PolyUOpCache *c) {
  if (!c) return;
  if (c->ranges) poly_map_destroy(c->ranges);
  if (c->ended) poly_map_destroy(c->ended);
  free(c);
}

/* C-only extraction of a single BUFFER/PARAM identity for residency keys.
 * Unlike has_buffer_identity, this must not erase an MSELECT lane. */
const PolyUOp *poly_uop_get_buffer_identity(const PolyUOp *u) {
  while (u) {
    if (u->op == POLY_OP_RESHAPE || u->op == POLY_OP_UNSHARD) {
      if (u->n_src < 1) return NULL;
      u = u->src[0];
      continue;
    }
    if (u->op == POLY_OP_BUFFER || u->op == POLY_OP_PARAM) {
      return u;
    }
    return NULL;
  }
  return NULL;
}

/* Current Tinygrad uop/ops.py:UOp.base. */
PolyUOp *poly_uop_base(PolyUOp *u) {
  while (u && u->n_src > 0 &&
         (poly_opset_has(POLY_GROUP_MOVEMENT, u->op) || u->op == POLY_OP_DETACH))
    u = u->src[0];
  return u;
}

/* Current Tinygrad uop/ops.py:UOp.unsharded_base. */
PolyUOp *poly_uop_unsharded_base(PolyUOp *u) {
  if (u && u->n_src > 0 &&
      (poly_opset_has(POLY_GROUP_MOVEMENT, u->op) || u->op == POLY_OP_DETACH ||
       u->op == POLY_OP_UNSHARD))
    return poly_uop_base(u->src[0]);
  return u;
}

/* Current Tinygrad uop/ops.py:UOp.op_in_backward_slice_with_self. */
bool poly_uop_op_in_backward_slice_with_self(PolyCtx *ctx, PolyUOp *u, PolyOps op) {
  if (!ctx || !u) return false;
  int n = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, u, &n);
  bool found = false;
  for (int i = 0; i < n && !found; i++)
    found = topo[i]->op == op;
  poly_toposort_free(topo);
  return found;
}

/* Current Tinygrad uop/ops.py:UOp.buf_uop. */
PolyUOp *poly_uop_buf_uop(PolyCtx *ctx, PolyUOp *u) {
  if (!ctx || !u) return NULL;
  if (u->op == POLY_OP_BUFFER || u->op == POLY_OP_PARAM) return u;
  if (u->op == POLY_OP_MSELECT) {
    PolyUOp *src = u->n_src == 1 ? poly_uop_buf_uop(ctx, u->src[0]) : NULL;
    return src ? poly_uop1(ctx, POLY_OP_MSELECT, u->dtype, src, u->arg) : NULL;
  }
  if (u->op == POLY_OP_MSTACK) {
    PolyUOp **src = u->n_src > 0 ? malloc((size_t)u->n_src * sizeof(*src)) : NULL;
    if (u->n_src > 0 && !src) return NULL;
    bool ok = true;
    for (int i = 0; i < u->n_src; i++) {
      src[i] = poly_uop_buf_uop(ctx, u->src[i]);
      if (!src[i]) ok = false;
    }
    PolyUOp *ret =
        ok ? poly_uop(ctx, POLY_OP_MSTACK, u->dtype, src, u->n_src, poly_arg_none()) : NULL;
    free(src);
    return ret;
  }

  PolyUOp *base = poly_uop_base(u);
  if (base && base->op == POLY_OP_AFTER && base->n_src > 0) {
    PolyUOp *target = poly_uop_buf_uop(ctx, base->src[0]);
    return poly_uop_base(target);
  }
  PolyUOp *s = u;
  while (s && s->n_src > 0 && s->op != POLY_OP_BUFFER && s->op != POLY_OP_PARAM &&
         s->op != POLY_OP_STAGE && s->op != POLY_OP_MSTACK)
    s = s->src[0];
  return s;
}

/* Current Tinygrad UOp.has_buffer_identity; aggregate selection uses the
 * existing MSELECT/UNSHARD wrappers rather than a synthetic storage-view op. */
bool poly_uop_has_buffer_identity(const PolyUOp *u) {
  while (u && u->n_src > 0 &&
         (u->op == POLY_OP_RESHAPE || u->op == POLY_OP_UNSHARD || u->op == POLY_OP_MSELECT))
    u = u->src[0];
  return u && (u->op == POLY_OP_BUFFER || u->op == POLY_OP_PARAM);
}

bool poly_uop_reachable(PolyCtx *ctx, PolyUOp *root, PolyUOp *target) {
  if (!ctx || !root || !target) return false;
  if (root == target) return true;
  int n = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, root, &n);
  if (!topo) return false;
  /* Mirrors tinygrad's `t.uop in loss.uop.toposort()` frontend test. */
  bool found = false;
  for (int i = 0; i < n; i++) {
    if (topo[i] == target) {
      found = true;
      break;
    }
  }
  poly_toposort_free(topo);
  return found;
}

bool poly_no_range(PolyCtx *ctx, PolyUOp *u) {
  return poly_no_range_ex(ctx, u, NULL);
}

bool poly_no_range_ex(PolyCtx *ctx, PolyUOp *u, PolyUOpCache *cache) {
  (void)cache; /* no_range is a single backward-slice walk; caching wouldn't
                * help because we short-circuit on the first RANGE hit. */
  if (!u) return true;
  if (u->op == POLY_OP_RANGE) return false;
  int n = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, u, &n);
  if (!topo) return false;
  bool no_range = true;
  for (int i = 0; i < n; i++) {
    if (topo[i]->op == POLY_OP_RANGE) {
      no_range = false;
      break;
    }
  }
  poly_toposort_free(topo);
  return no_range;
}

bool poly_uop_in_ranges(PolyCtx *ctx, PolyUOp *u, PolyUOp *r) {
  return poly_uop_in_ranges_ex(ctx, u, r, NULL);
}

bool poly_uop_in_ranges_ex(PolyCtx *ctx, PolyUOp *u, PolyUOp *r, PolyUOpCache *cache) {
  if (!ctx || !u || !r || r->op != POLY_OP_RANGE) return false;
  if (!cache && u->ranges_cache)
    return range_set_contains((const PolyRangeSet *)u->ranges_cache, r);
  PolyMap *memo = cache ? cache->ranges : poly_map_new(64);
  PolyMap *ended = cache ? cache->ended : poly_map_new(64);
  if (!memo) return false;
  if (!ended) {
    if (!cache) poly_map_destroy(memo);
    return false;
  }
  const PolyRangeSet *s = compute_ranges_with_ended(ctx, u, memo, ended);
  if (!s) range_cache_failed();
  bool found = range_set_contains(s, r);
  if (!cache) {
    poly_map_destroy(memo);
    poly_map_destroy(ended);
  }
  return found;
}

int poly_uop_ranges(PolyCtx *ctx, PolyUOp *u, PolyUOp **out, int max_out) {
  return poly_uop_ranges_ex(ctx, u, out, max_out, NULL);
}

int poly_uop_ranges_ex(PolyCtx *ctx, PolyUOp *u, PolyUOp **out, int max_out, PolyUOpCache *cache) {
  if (!ctx || !u || !out || max_out <= 0) return 0;
  if (!cache && u->ranges_cache) {
    const PolyRangeSet *s = (const PolyRangeSet *)u->ranges_cache;
    int n_out = s->n < max_out ? s->n : max_out;
    memcpy(out, s->items, (size_t)n_out * sizeof(PolyUOp *));
    return n_out;
  }
  PolyMap *memo = cache ? cache->ranges : poly_map_new(64);
  PolyMap *ended = cache ? cache->ended : poly_map_new(64);
  if (!memo) return 0;
  if (!ended) {
    if (!cache) poly_map_destroy(memo);
    return 0;
  }
  const PolyRangeSet *s = compute_ranges_with_ended(ctx, u, memo, ended);
  if (!s) range_cache_failed();
  int n_out = s->n < max_out ? s->n : max_out;
  memcpy(out, s->items, (size_t)n_out * sizeof(PolyUOp *));
  if (!cache) {
    poly_map_destroy(memo);
    poly_map_destroy(ended);
  }
  return n_out;
}

/* Pretty-print */

/* DType.__repr__ uses the public dtypes attribute name, which intentionally
 * differs from several renderer/C names (tinygrad/dtype.py:44-67). */
static const char *dtype_arg_repr_name(PolyDType dtype) {
  const char *name = poly_dtype_name(dtype);
  if (!name) return "void";
  if (strcmp(name, "signed char") == 0) return "char";
  if (strcmp(name, "unsigned char") == 0) return "uchar";
  if (strcmp(name, "short") == 0) return "short";
  if (strcmp(name, "unsigned short") == 0) return "ushort";
  if (strcmp(name, "int") == 0) return "int";
  if (strcmp(name, "unsigned int") == 0) return "uint";
  if (strcmp(name, "long") == 0) return "long";
  if (strcmp(name, "unsigned long") == 0) return "ulong";
  if (strcmp(name, "__fp16") == 0) return "half";
  if (strcmp(name, "__bf16") == 0) return "bfloat16";
  if (strcmp(name, "float8_e4m3") == 0) return "fp8e4m3";
  if (strcmp(name, "float8_e5m2") == 0) return "fp8e5m2";
  if (strcmp(name, "float8_e4m3fnuz") == 0) return "fp8e4m3fnuz";
  if (strcmp(name, "float8_e5m2fnuz") == 0) return "fp8e5m2fnuz";
  if (strcmp(name, "float") == 0) return "float";
  if (strcmp(name, "double") == 0) return "double";
  return name;
}

static void uop_print_one(PolyUOp *u, char *buf, int *pos, int cap) {
  int written = snprintf(buf + *pos, cap - *pos, "UOp(%s", poly_op_name(u->op));
  if (written > 0) *pos += written;

  /* dtype */
  if (!poly_dtype_eq(u->dtype, POLY_VOID)) {
    written = snprintf(buf + *pos, cap - *pos, ", %s", poly_dtype_name(u->dtype));
    if (written > 0) *pos += written;
  }

  /* arg */
  switch (u->arg.kind) {
  case POLY_ARG_NONE:
    break;
  case POLY_ARG_INT:
    written = snprintf(buf + *pos, cap - *pos, ", %ld", (long)u->arg.i);
    if (written > 0) *pos += written;
    break;
  case POLY_ARG_ALLREDUCE:
    written = snprintf(buf + *pos, cap - *pos, ", (%s,", poly_op_name(u->arg.allreduce.op));
    if (written > 0) *pos += written;
    if (u->arg.allreduce.device_is_tuple) {
      written = snprintf(buf + *pos, cap - *pos, "(");
      if (written > 0) *pos += written;
      for (int i = 0; i < u->arg.allreduce.n_devices; i++) {
        written =
            snprintf(buf + *pos, cap - *pos, "%s'%s'", i ? "," : "", u->arg.allreduce.devices[i]);
        if (written > 0) *pos += written;
      }
      written = snprintf(buf + *pos, cap - *pos, "))");
    } else {
      written = snprintf(buf + *pos, cap - *pos, "'%s')", u->arg.allreduce.device);
    }
    if (written > 0) *pos += written;
    break;
  case POLY_ARG_BIGINT: {
    char *decimal = poly_arg_integer_to_decimal(u->arg);
    written = snprintf(buf + *pos, cap - *pos, ", %s", decimal ? decimal : "0");
    free(decimal);
    if (written > 0) *pos += written;
  } break;
  case POLY_ARG_REDUCE:
    written = snprintf(
        buf + *pos, cap - *pos, ", (%s,%d)", poly_op_name(u->arg.reduce.op), u->arg.reduce.num_axes
    );
    if (written > 0) *pos += written;
    break;
  case POLY_ARG_FLOAT:
    /* Pinned UOp.argstr uses Python's round-trippable float repr
     * (uop/ops.py:166-171). Seventeen significant decimal digits preserve a
     * C double exactly for graph diagnostics and parity tooling. */
    written = snprintf(buf + *pos, cap - *pos, ", %.17g", u->arg.f);
    if (written > 0) *pos += written;
    break;
  case POLY_ARG_BOOL:
    written = snprintf(buf + *pos, cap - *pos, ", %s", u->arg.b ? "True" : "False");
    if (written > 0) *pos += written;
    break;
  case POLY_ARG_STRING:
    written = snprintf(buf + *pos, cap - *pos, ", \"%s\"", u->arg.str ? u->arg.str : "");
    if (written > 0) *pos += written;
    break;
  case POLY_ARG_STRING_TUPLE:
    written = snprintf(buf + *pos, cap - *pos, ", (");
    if (written > 0) *pos += written;
    for (int i = 0; i < u->arg.string_tuple.n; i++) {
      written =
          snprintf(buf + *pos, cap - *pos, "%s\"%s\"", i ? "," : "", u->arg.string_tuple.vals[i]);
      if (written > 0) *pos += written;
    }
    written = snprintf(buf + *pos, cap - *pos, ")");
    if (written > 0) *pos += written;
    break;
  case POLY_ARG_OPS:
    written = snprintf(buf + *pos, cap - *pos, ", %s", poly_op_name(u->arg.ops));
    if (written > 0) *pos += written;
    break;
  case POLY_ARG_INVALID:
    written = snprintf(buf + *pos, cap - *pos, ", Invalid");
    if (written > 0) *pos += written;
    break;
  case POLY_ARG_INT_TUPLE:
    written = snprintf(buf + *pos, cap - *pos, ", (");
    if (written > 0) *pos += written;
    for (int i = 0; i < u->arg.int_tuple.n; i++) {
      if (u->op == POLY_OP_FLIP)
        written = snprintf(
            buf + *pos, cap - *pos, "%s%s", i ? "," : "",
            u->arg.int_tuple.vals[i] ? "True" : "False"
        );
      else
        written =
            snprintf(buf + *pos, cap - *pos, "%s%ld", i ? "," : "", (long)u->arg.int_tuple.vals[i]);
      if (written > 0) *pos += written;
    }
    written = snprintf(buf + *pos, cap - *pos, ")");
    if (written > 0) *pos += written;
    break;
  case POLY_ARG_RANGE:
    written = snprintf(
        buf + *pos, cap - *pos, ", (%ld,%d", (long)u->arg.range.axis_id, (int)u->arg.range.axis_type
    );
    if (written > 0) *pos += written;
    for (int i = 0; i < u->arg.range.n_extra; i++) {
      written = snprintf(buf + *pos, cap - *pos, ",%ld", (long)u->arg.range.extra[i]);
      if (written > 0) *pos += written;
    }
    written = snprintf(buf + *pos, cap - *pos, ")");
    if (written > 0) *pos += written;
    break;
  case POLY_ARG_BUFFERIZE_OPTS:
    written = snprintf(buf + *pos, cap - *pos, ", BufferizeOpts(device=");
    if (written > 0) *pos += written;
    if (u->arg.bufferize_opts.device_is_int) {
      written =
          snprintf(buf + *pos, cap - *pos, "%lld", (long long)u->arg.bufferize_opts.device_int);
    } else if (u->arg.bufferize_opts.device_is_tuple) {
      written = snprintf(buf + *pos, cap - *pos, "(");
      if (written > 0) *pos += written;
      for (int i = 0; i < u->arg.bufferize_opts.n_devices; i++) {
        written = snprintf(
            buf + *pos, cap - *pos, "%s\"%s\"", i ? "," : "", u->arg.bufferize_opts.devices[i]
        );
        if (written > 0) *pos += written;
      }
      written = snprintf(buf + *pos, cap - *pos, ")");
    } else {
      written = snprintf(
          buf + *pos, cap - *pos, "%s",
          u->arg.bufferize_opts.device ? u->arg.bufferize_opts.device : "None"
      );
    }
    if (written > 0) *pos += written;
    written = snprintf(
        buf + *pos, cap - *pos, ",addrspace=%d,removable=%d)", (int)u->arg.bufferize_opts.addrspace,
        (int)u->arg.bufferize_opts.removable
    );
    if (written > 0) *pos += written;
    break;
  case POLY_ARG_TENSOR_CORE:
    written = snprintf(
        buf + *pos, cap - *pos, ", WMMA((%d,%d,%d),%s,%s,threads=%d,axes=%s)",
        u->arg.tensor_core.dims[0], u->arg.tensor_core.dims[1], u->arg.tensor_core.dims[2],
        poly_dtype_name(u->arg.tensor_core.dtype_in),
        u->arg.tensor_core.device ? u->arg.tensor_core.device : "?", u->arg.tensor_core.threads,
        u->arg.tensor_core.has_upcast_axes ? "set" : "None"
    );
    if (written > 0) *pos += written;
    break;
  case POLY_ARG_PARAM:
    if (!u->arg.param) {
      written = snprintf(buf + *pos, cap - *pos, ", ParamArg(-1)");
      if (written > 0) *pos += written;
      break;
    }
    written = snprintf(
        buf + *pos, cap - *pos, ", ParamArg(%ld, dtypes.%s", (long)u->arg.param->slot,
        dtype_arg_repr_name(u->arg.param->dtype)
    );
    if (written > 0) *pos += written;
    if (u->arg.param->has_minmax) {
      written = snprintf(
          buf + *pos, cap - *pos, ", vmin_vmax=(%ld, %ld)", (long)u->arg.param->min_val,
          (long)u->arg.param->max_val
      );
      if (written > 0) *pos += written;
    }
    if (u->arg.param->has_multiple_of) {
      written =
          snprintf(buf + *pos, cap - *pos, ", multiple_of=%ld", (long)u->arg.param->multiple_of);
      if (written > 0) *pos += written;
    }
    if (u->arg.param->name) {
      written = snprintf(buf + *pos, cap - *pos, ", name='%s'", u->arg.param->name);
      if (written > 0) *pos += written;
    }
    if (u->arg.param->addrspace != POLY_ADDR_GLOBAL) {
      const char *addrspace = u->arg.param->addrspace == POLY_ADDR_LOCAL ? "AddrSpace.LOCAL"
                              : u->arg.param->addrspace == POLY_ADDR_REG ? "AddrSpace.REG"
                              : u->arg.param->addrspace == POLY_ADDR_ALU ? "AddrSpace.ALU"
                                                                         : "AddrSpace.UNKNOWN";
      written = snprintf(buf + *pos, cap - *pos, ", addrspace=%s", addrspace);
      if (written > 0) *pos += written;
    }
    if (u->arg.param->has_axis) {
      written = snprintf(buf + *pos, cap - *pos, ", axis=%d", u->arg.param->axis);
      if (written > 0) *pos += written;
    }
    if (u->arg.param->device_is_tuple) {
      written = snprintf(buf + *pos, cap - *pos, ", device=(");
      if (written > 0) *pos += written;
      for (int i = 0; i < u->arg.param->n_devices; i++) {
        written = snprintf(
            buf + *pos, cap - *pos, "%s'%s'%s", i ? ", " : "", u->arg.param->devices[i],
            u->arg.param->n_devices == 1 ? "," : ""
        );
        if (written > 0) *pos += written;
      }
      written = snprintf(buf + *pos, cap - *pos, ")");
      if (written > 0) *pos += written;
    } else if (u->arg.param->device) {
      written = snprintf(buf + *pos, cap - *pos, ", device='%s'", u->arg.param->device);
      if (written > 0) *pos += written;
    }
    if (u->arg.param->volatile_) {
      written = snprintf(buf + *pos, cap - *pos, ", volatile=True");
      if (written > 0) *pos += written;
    }
    written = snprintf(buf + *pos, cap - *pos, ")");
    if (written > 0) *pos += written;
    break;
  case POLY_ARG_CALL_INFO:
    if (!u->arg.call_info) {
      written = snprintf(buf + *pos, cap - *pos, ", CallInfo(None,None,False,False)");
    } else {
      const PolyCallInfo *info = u->arg.call_info;
      written = snprintf(
          buf + *pos, cap - *pos, ", CallInfo(%s,%s%s%s,%s,%s)",
          info->has_grad_fxn ? "<callback>" : "None", info->name ? "'" : "",
          info->name ? info->name : "None", info->name ? "'" : "",
          info->precompile ? "True" : "False", info->precompile_backward ? "True" : "False"
      );
    }
    if (written > 0) *pos += written;
    break;
  case POLY_ARG_KERNEL_INFO:
    written = snprintf(
        buf + *pos, cap - *pos, ", KernelInfo(name='%s', beam=%d)",
        u->arg.kernel_info && u->arg.kernel_info->name ? u->arg.kernel_info->name : "test",
        u->arg.kernel_info ? u->arg.kernel_info->beam : 0
    );
    if (written > 0) *pos += written;
    break;
  case POLY_ARG_DTYPE:
    written = snprintf(buf + *pos, cap - *pos, ", dtypes.%s", dtype_arg_repr_name(u->arg.dtype));
    if (written > 0) *pos += written;
    break;
  default:
    break;
  }

  /* n_src */
  if (u->n_src > 0) {
    written = snprintf(buf + *pos, cap - *pos, ", src=%d", u->n_src);
    if (written > 0) *pos += written;
  }

  written = snprintf(buf + *pos, cap - *pos, ")");
  if (written > 0) *pos += written;
}

char *poly_uop_str(PolyUOp *u) {
  int cap = 256;
  char *buf = malloc(cap);
  int pos = 0;
  uop_print_one(u, buf, &pos, cap);
  buf[pos] = '\0';
  return buf;
}

char *poly_graph_str(PolyUOp *root) {
  /* Simple: just print the root node for now */
  /* A full graph print would need toposort, but that needs a ctx */
  return poly_uop_str(root);
}

/* Recursive indented tree dump to a FILE*. Used by debug paths in
 * reduce_simplify.c, codegen.c, and any future pass that needs to inspect
 * IR shape. Caps recursion at max_depth to keep output bounded on cyclic
 * or very deep graphs (though hash-consed UOps cannot actually cycle). */
void poly_uop_dump_tree(FILE *fp, PolyUOp *u, int depth, int max_depth) {
  if (!u || !fp) return;
  for (int i = 0; i < depth; i++)
    fputc(' ', fp);
  fprintf(
      fp, "%s dt=%s n_src=%d", poly_op_name(u->op), u->dtype.name ? u->dtype.name : "?",
      (int)u->n_src
  );
  switch (u->arg.kind) {
  case POLY_ARG_INT:
    fprintf(fp, " i=%lld", (long long)u->arg.i);
    break;
  case POLY_ARG_BIGINT: {
    char *decimal = poly_arg_integer_to_decimal(u->arg);
    fprintf(fp, " i=%s", decimal ? decimal : "0");
    free(decimal);
  } break;
  case POLY_ARG_FLOAT:
    fprintf(fp, " f=%g", u->arg.f);
    break;
  case POLY_ARG_BOOL:
    fprintf(fp, " b=%d", (int)u->arg.b);
    break;
  case POLY_ARG_OPS:
    fprintf(fp, " op=%s", poly_op_name(u->arg.ops));
    break;
  case POLY_ARG_REDUCE:
    fprintf(fp, " reduce=(%s,%d)", poly_op_name(u->arg.reduce.op), u->arg.reduce.num_axes);
    break;
  case POLY_ARG_ALLREDUCE:
    fprintf(fp, " allreduce=(%s,", poly_op_name(u->arg.allreduce.op));
    if (u->arg.allreduce.device_is_tuple) {
      fprintf(fp, "(");
      for (int i = 0; i < u->arg.allreduce.n_devices; i++)
        fprintf(fp, "%s\"%s\"", i ? "," : "", u->arg.allreduce.devices[i]);
      fprintf(fp, "))");
    } else {
      fprintf(fp, "%s)", u->arg.allreduce.device);
    }
    break;
  case POLY_ARG_STRING_TUPLE:
    fprintf(fp, " strings=(");
    for (int i = 0; i < u->arg.string_tuple.n; i++)
      fprintf(fp, "%s\"%s\"", i ? "," : "", u->arg.string_tuple.vals[i]);
    fprintf(fp, ")");
    break;
  case POLY_ARG_RANGE:
    fprintf(fp, " axis=%lld", (long long)u->arg.range.axis_id);
    break;
  case POLY_ARG_BUFFERIZE_OPTS:
    fprintf(fp, " bufferize_opts=(device=");
    if (u->arg.bufferize_opts.device_is_int) {
      fprintf(fp, "%lld", (long long)u->arg.bufferize_opts.device_int);
    } else if (u->arg.bufferize_opts.device_is_tuple) {
      fprintf(fp, "(");
      for (int i = 0; i < u->arg.bufferize_opts.n_devices; i++)
        fprintf(fp, "%s\"%s\"", i ? "," : "", u->arg.bufferize_opts.devices[i]);
      fprintf(fp, ")");
    } else {
      fprintf(fp, "%s", u->arg.bufferize_opts.device ? u->arg.bufferize_opts.device : "None");
    }
    fprintf(
        fp, ",addrspace=%d,removable=%d)", (int)u->arg.bufferize_opts.addrspace,
        (int)u->arg.bufferize_opts.removable
    );
    break;
  case POLY_ARG_TENSOR_CORE:
    fprintf(
        fp, " wmma=((%d,%d,%d),%s,%s,threads=%d,axes=%s)", u->arg.tensor_core.dims[0],
        u->arg.tensor_core.dims[1], u->arg.tensor_core.dims[2],
        poly_dtype_name(u->arg.tensor_core.dtype_in),
        u->arg.tensor_core.device ? u->arg.tensor_core.device : "?", u->arg.tensor_core.threads,
        u->arg.tensor_core.has_upcast_axes ? "set" : "None"
    );
    break;
  case POLY_ARG_PARAM:
    fprintf(
        fp, " param=(slot=%lld,device=%s,addrspace=%d%s%s)",
        u->arg.param ? (long long)u->arg.param->slot : -1LL,
        u->arg.param && u->arg.param->device ? u->arg.param->device : "None",
        u->arg.param ? (int)u->arg.param->addrspace : 0,
        u->arg.param && u->arg.param->has_axis ? ",axis" : "",
        u->arg.param && u->arg.param->name ? ",name" : ""
    );
    break;
  case POLY_ARG_CALL_INFO:
    fprintf(
        fp, " call_info=(name=%s,precompile=%d,precompile_backward=%d%s%s)",
        u->arg.call_info && u->arg.call_info->name ? u->arg.call_info->name : "None",
        u->arg.call_info ? (int)u->arg.call_info->precompile : 0,
        u->arg.call_info ? (int)u->arg.call_info->precompile_backward : 0,
        u->arg.call_info && u->arg.call_info->has_grad_fxn ? ",grad_fxn" : "",
        u->arg.call_info && u->arg.call_info->has_aux ? ",aux" : ""
    );
    break;
  case POLY_ARG_KERNEL_INFO:
    fprintf(
        fp, " kernel_info=(name=%s,beam=%d)",
        u->arg.kernel_info && u->arg.kernel_info->name ? u->arg.kernel_info->name : "test",
        u->arg.kernel_info ? u->arg.kernel_info->beam : 0
    );
    break;
  case POLY_ARG_DTYPE:
    fprintf(fp, " dtype=dtypes.%s", dtype_arg_repr_name(u->arg.dtype));
    break;
  default:
    break;
  }
  fputc('\n', fp);
  if (depth >= max_depth) return;
  for (uint16_t i = 0; i < u->n_src; i++)
    poly_uop_dump_tree(fp, u->src[i], depth + 2, max_depth);
}

/* Named buffer registry */

#include <stdarg.h>

static uint32_t reg_str_hash(const char *s) {
  uint32_t h = 2166136261u;
  for (; *s; s++)
    h = (h ^ (uint8_t)*s) * 16777619u;
  return h;
}

static bool reg_str_eq(const void *a, const void *b) {
  return strcmp((const char *)a, (const char *)b) == 0;
}

static char *arena_strdup(PolyArena *a, const char *s) {
  size_t len = strlen(s) + 1;
  char *p = poly_arena_alloc(a, len, 1);
  if (p) memcpy(p, s, len);
  return p;
}

static char *arena_vsprintf(PolyArena *a, const char *fmt, va_list ap) {
  va_list ap2;
  va_copy(ap2, ap);
  int len = vsnprintf(NULL, 0, fmt, ap2);
  va_end(ap2);
  if (len < 0) return NULL;
  char *p = poly_arena_alloc(a, (size_t)len + 1, 1);
  if (p) vsnprintf(p, (size_t)len + 1, fmt, ap);
  return p;
}

/* Internal: grow the entries array */
static int reg_grow_entries(PolyCtx *ctx) {
  int new_cap = ctx->entries_cap ? ctx->entries_cap * 2 : 16;
  PolyRegEntry **p = realloc(ctx->entries, (size_t)new_cap * sizeof(PolyRegEntry *));
  if (!p) return -1;
  ctx->entries = p;
  ctx->entries_cap = new_cap;
  return 0;
}

/* Internal: register a named buffer with given role */
static PolyUOp *register_named(
    PolyCtx *ctx,
    PolyBufRole role,
    PolyDType dt,
    const int64_t *shape,
    int ndim,
    const char *name
) {
  if (!ctx || !name || ndim < 0 || ndim > 8) return NULL;

  /* Check for re-registration */
  uint32_t h = reg_str_hash(name);
  PolyRegEntry *existing = poly_map_get(ctx->name_map, h, name, reg_str_eq);
  if (existing) {
    /* Validate dtype: compare by priority+bitsize (works for both scalar and ptr dtypes) */
    if (existing->buffer->dtype.priority != dt.priority ||
        existing->buffer->dtype.bitsize != dt.bitsize) {
      fprintf(stderr, "poly_register: '%s' already registered with different dtype\n", name);
      return NULL;
    }
    /* Validate shape */
    if (existing->ndim != ndim) {
      fprintf(
          stderr, "poly_register: '%s' already registered with different ndim (%d vs %d)\n", name,
          existing->ndim, ndim
      );
      return NULL;
    }
    for (int i = 0; i < ndim; i++) {
      if (existing->shape[i] != shape[i]) {
        fprintf(stderr, "poly_register: '%s' already registered with different shape\n", name);
        return NULL;
      }
    }
    return existing->buffer;
  }

  /* Compute numel */
  int64_t numel = 1;
  for (int i = 0; i < ndim; i++)
    numel *= shape[i];

  /* PolyModel names an approved deviceless logical resource. Explicit
   * placement later creates Tinygrad-shaped physical BUFFER metadata. */
  PolyUOp *buf = poly_uop_new_logical_buffer(ctx, dt, numel);
  if (!buf) return NULL;

  /* Arena-alloc entry */
  PolyRegEntry *entry = poly_arena_alloc(ctx->arena, sizeof(PolyRegEntry), _Alignof(PolyRegEntry));
  if (!entry) return NULL;
  entry->name = arena_strdup(ctx->arena, name);
  entry->role = role;
  entry->buffer = buf;
  entry->ndim = ndim;
  entry->is_alias = false;
  entry->trainable = (role == POLY_ROLE_PARAM);
  for (int i = 0; i < ndim; i++)
    entry->shape[i] = shape[i];

  /* Append to entries array */
  if (ctx->n_entries >= ctx->entries_cap && reg_grow_entries(ctx) < 0) return NULL;
  ctx->entries[ctx->n_entries++] = entry;

  /* Insert into name map */
  poly_map_set(ctx->name_map, h, entry->name, entry, reg_str_eq);

  return buf;
}

static PolyUOp *register_existing_named(
    PolyCtx *ctx,
    PolyBufRole role,
    PolyUOp *buffer,
    const int64_t *shape,
    int ndim,
    const char *name,
    bool trainable
) {
  if (!ctx || !name || !buffer || ndim < 0 || ndim > 8) return NULL;
  const PolyUOp *identity = poly_uop_get_buffer_identity(buffer);
  if (!identity || identity != buffer) {
    fprintf(stderr, "poly_register_existing_buffer: '%s' is not a BUFFER identity\n", name);
    return NULL;
  }

  uint32_t h = reg_str_hash(name);
  PolyRegEntry *existing = poly_map_get(ctx->name_map, h, name, reg_str_eq);
  if (existing) {
    if (existing->buffer != buffer) {
      fprintf(stderr, "poly_register_existing_buffer: '%s' already names another buffer\n", name);
      return NULL;
    }
    if (existing->ndim != ndim) {
      fprintf(stderr, "poly_register_existing_buffer: '%s' registered with different ndim\n", name);
      return NULL;
    }
    for (int i = 0; i < ndim; i++) {
      if (existing->shape[i] != shape[i]) {
        fprintf(
            stderr, "poly_register_existing_buffer: '%s' registered with different shape\n", name
        );
        return NULL;
      }
    }
    existing->role = role;
    existing->trainable = trainable;
    return existing->buffer;
  }

  PolyRegEntry *entry = poly_arena_alloc(ctx->arena, sizeof(PolyRegEntry), _Alignof(PolyRegEntry));
  if (!entry) return NULL;
  entry->name = arena_strdup(ctx->arena, name);
  entry->role = role;
  entry->buffer = buffer;
  entry->ndim = ndim;
  entry->is_alias = false;
  entry->trainable = trainable;
  for (int i = 0; i < ndim; i++)
    entry->shape[i] = shape[i];

  if (ctx->n_entries >= ctx->entries_cap && reg_grow_entries(ctx) < 0) return NULL;
  ctx->entries[ctx->n_entries++] = entry;
  poly_map_set(ctx->name_map, h, entry->name, entry, reg_str_eq);
  return buffer;
}

/* Public registration wrappers */

PolyUOp *poly_param(
    PolyCtx *ctx,
    PolyDType dt,
    const int64_t *shape,
    int ndim,
    const char *fmt,
    ...
) {
  va_list ap;
  va_start(ap, fmt);
  char *name = arena_vsprintf(ctx->arena, fmt, ap);
  va_end(ap);
  if (!name) return NULL;
  return register_named(ctx, POLY_ROLE_PARAM, dt, shape, ndim, name);
}

PolyUOp *poly_input(
    PolyCtx *ctx,
    PolyDType dt,
    const int64_t *shape,
    int ndim,
    const char *fmt,
    ...
) {
  va_list ap;
  va_start(ap, fmt);
  char *name = arena_vsprintf(ctx->arena, fmt, ap);
  va_end(ap);
  if (!name) return NULL;
  return register_named(ctx, POLY_ROLE_INPUT, dt, shape, ndim, name);
}

PolyUOp *poly_output(
    PolyCtx *ctx,
    PolyDType dt,
    const int64_t *shape,
    int ndim,
    const char *fmt,
    ...
) {
  va_list ap;
  va_start(ap, fmt);
  char *name = arena_vsprintf(ctx->arena, fmt, ap);
  va_end(ap);
  if (!name) return NULL;
  return register_named(ctx, POLY_ROLE_OUTPUT, dt, shape, ndim, name);
}

PolyUOp *poly_target(
    PolyCtx *ctx,
    PolyDType dt,
    const int64_t *shape,
    int ndim,
    const char *fmt,
    ...
) {
  va_list ap;
  va_start(ap, fmt);
  char *name = arena_vsprintf(ctx->arena, fmt, ap);
  va_end(ap);
  if (!name) return NULL;
  return register_named(ctx, POLY_ROLE_TARGET, dt, shape, ndim, name);
}

PolyUOp *poly_aux(
    PolyCtx *ctx,
    PolyDType dt,
    const int64_t *shape,
    int ndim,
    const char *fmt,
    ...
) {
  va_list ap;
  va_start(ap, fmt);
  char *name = arena_vsprintf(ctx->arena, fmt, ap);
  va_end(ap);
  if (!name) return NULL;
  return register_named(ctx, POLY_ROLE_AUX, dt, shape, ndim, name);
}

PolyUOp *poly_register_buffer_by_id(
    PolyCtx *ctx,
    int role,
    int dtype_id,
    const int64_t *shape,
    int ndim,
    const char *name
) {
  PolyDType dt;
  if (!poly_dtype_by_id(dtype_id, &dt)) return NULL;
  if (role < POLY_ROLE_PARAM || role > POLY_ROLE_AUX) return NULL;
  return register_named(ctx, (PolyBufRole)role, dt, shape, ndim, name);
}

PolyUOp *poly_register_existing_buffer(
    PolyCtx *ctx,
    int role,
    PolyUOp *buffer,
    const int64_t *shape,
    int ndim,
    const char *name,
    bool trainable
) {
  if (role < POLY_ROLE_PARAM || role > POLY_ROLE_AUX) return NULL;
  return register_existing_named(ctx, (PolyBufRole)role, buffer, shape, ndim, name, trainable);
}

/* Alias */

int poly_alias(PolyCtx *ctx, const char *alias_name, const char *existing_name) {
  if (!ctx || !alias_name || !existing_name) return -1;

  uint32_t eh = reg_str_hash(existing_name);
  PolyRegEntry *existing = poly_map_get(ctx->name_map, eh, existing_name, reg_str_eq);
  if (!existing) return -1;

  uint32_t ah = reg_str_hash(alias_name);
  PolyRegEntry *check = poly_map_get(ctx->name_map, ah, alias_name, reg_str_eq);
  if (check) {
    return (check->buffer == existing->buffer) ? 0 : -1;
  }

  PolyRegEntry *entry = poly_arena_alloc(ctx->arena, sizeof(PolyRegEntry), _Alignof(PolyRegEntry));
  if (!entry) return -1;
  entry->name = arena_strdup(ctx->arena, alias_name);
  entry->role = existing->role;
  entry->buffer = existing->buffer;
  entry->ndim = existing->ndim;
  entry->is_alias = true;
  entry->trainable = existing->trainable;
  for (int i = 0; i < existing->ndim; i++)
    entry->shape[i] = existing->shape[i];

  if (ctx->n_entries >= ctx->entries_cap && reg_grow_entries(ctx) < 0) return -1;
  ctx->entries[ctx->n_entries++] = entry;
  poly_map_set(ctx->name_map, ah, entry->name, entry, reg_str_eq);
  return 0;
}

/* Lookup */

PolyUOp *poly_ctx_get(PolyCtx *ctx, const char *fmt, ...) {
  if (!ctx || !fmt) return NULL;
  char buf[256];
  va_list ap;
  va_start(ap, fmt);
  int n = vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  if (n < 0 || n >= (int)sizeof(buf)) return NULL;
  PolyRegEntry *entry = poly_map_get(ctx->name_map, reg_str_hash(buf), buf, reg_str_eq);
  return entry ? entry->buffer : NULL;
}

const PolyRegEntry *poly_ctx_get_entry(PolyCtx *ctx, const char *name) {
  if (!ctx || !name) return NULL;
  return poly_map_get(ctx->name_map, reg_str_hash(name), name, reg_str_eq);
}

int poly_ctx_set_trainable(PolyCtx *ctx, const char *name, bool trainable) {
  if (!ctx || !name) return -1;
  PolyRegEntry *entry = poly_map_get(ctx->name_map, reg_str_hash(name), name, reg_str_eq);
  if (!entry) return -1;
  entry->trainable = trainable;
  for (int i = 0; i < ctx->n_entries; i++) {
    PolyRegEntry *other = ctx->entries[i];
    if (other && other->buffer == entry->buffer) other->trainable = trainable;
  }
  return 0;
}

bool poly_ctx_is_trainable(PolyCtx *ctx, const char *name) {
  if (!ctx || !name) return false;
  PolyRegEntry *entry = poly_map_get(ctx->name_map, reg_str_hash(name), name, reg_str_eq);
  return entry ? entry->trainable : false;
}

/* Enumeration */

int poly_ctx_named_count(PolyCtx *ctx) {
  return ctx ? ctx->n_entries : 0;
}

const PolyRegEntry *poly_ctx_named_entry(PolyCtx *ctx, int i) {
  if (!ctx || i < 0 || i >= ctx->n_entries) return NULL;
  return ctx->entries[i];
}

/* Entrypoints */

int poly_register_entrypoint(PolyCtx *ctx, const char *name, PolyUOp *sink) {
  if (!ctx || !name || !sink) return -1;
  if (ctx->n_ep >= ctx->ep_cap) {
    int new_cap = ctx->ep_cap ? ctx->ep_cap * 2 : 4;
    void *p = realloc(ctx->ep, (size_t)new_cap * sizeof(ctx->ep[0]));
    if (!p) return -1;
    ctx->ep = p;
    ctx->ep_cap = new_cap;
  }
  ctx->ep[ctx->n_ep].name = arena_strdup(ctx->arena, name);
  ctx->ep[ctx->n_ep].sink = sink;
  ctx->n_ep++;
  return 0;
}

int poly_ctx_entrypoint_count(PolyCtx *ctx) {
  return ctx ? ctx->n_ep : 0;
}

const char *poly_ctx_entrypoint_name(PolyCtx *ctx, int i) {
  if (!ctx || i < 0 || i >= ctx->n_ep) return NULL;
  return ctx->ep[i].name;
}

PolyUOp *poly_ctx_entrypoint_sink(PolyCtx *ctx, int i) {
  if (!ctx || i < 0 || i >= ctx->n_ep) return NULL;
  return ctx->ep[i].sink;
}

/* UOp construction helpers (moved from frontend.c) */

int poly_op_count(void) {
  return (int)POLY_OP_COUNT;
}

PolyUOp *poly_const_float(PolyCtx *ctx, double value) {
  return poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKFLOAT, poly_arg_float(value));
}

PolyUOp *poly_const_double(PolyCtx *ctx, double value) {
  return poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT64, poly_arg_float(value));
}

PolyUOp *poly_const_int(PolyCtx *ctx, int64_t value) {
  return poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(value));
}

static bool uop_base_is_invalid(PolyUOp *u) {
  while (u && u->n_src > 0 &&
         (poly_opset_has(POLY_GROUP_MOVEMENT, u->op) || u->op == POLY_OP_DETACH))
    u = u->src[0];
  return u && u->op == POLY_OP_CONST && u->arg.kind == POLY_ARG_INVALID;
}

static bool promo_dtype(PolyUOp **src, int n_src, PolyDType *out) {
  if (!src || n_src <= 0 || !src[0] || !out) return false;
  PolyDType dtype = src[0]->dtype;
  for (int i = 1; i < n_src; i++)
    if (!src[i] || !poly_dtype_least_upper(dtype, src[i]->dtype, &dtype)) return false;
  *out = dtype;
  return true;
}

static bool param_is_image_shape(const PolyUOp *u) {
  if (!u || u->op != POLY_OP_PARAM || u->n_src != 1 || !u->src[0] ||
      u->src[0]->op != POLY_OP_STACK || u->src[0]->n_src != 3)
    return false;
  int64_t channels = 0;
  return poly_uop_const_i64(u->src[0]->src[2], &channels) == 0 && channels == 4;
}

bool poly_uop_is_image_shape(PolyCtx *ctx, const PolyUOp *u) {
  if (!ctx || !u) return false;
  PolyShape shape = poly_uop_max_shape_cached(ctx, u);
  return shape.ndim == 3 && shape.dims && shape.dims[2] == 4;
}

bool poly_dtype_from_uop(
    PolyOps op,
    PolyUOp **src,
    int n_src,
    PolyArg arg,
    PolyDType current_dtype,
    PolyDType *out
) {
  /* Direct C port of current tinygrad uop/ops.py:dtype_from_uop.  Polygrad's
   * retained compatibility ops have no upstream production rule and return
   * false, which makes _rebuild_dtype preserve their explicit dtype. */
  if (!out || n_src < 0 || (n_src > 0 && !src)) return false;
  switch (op) {
  case POLY_OP_STORE:
  case POLY_OP_LINEAR:
  case POLY_OP_SINK:
  case POLY_OP_PROGRAM:
  case POLY_OP_SOURCE:
  case POLY_OP_END:
  case POLY_OP_BARRIER:
  case POLY_OP_GROUP:
  case POLY_OP_IF:
  case POLY_OP_ENDIF:
  case POLY_OP_TUPLE:
  case POLY_OP_FUNCTION:
  case POLY_OP_CUSTOM_FUNCTION:
  case POLY_OP_REWRITE_ERROR:
    *out = POLY_VOID;
    return true;
  case POLY_OP_CALL:
    if (n_src > 0 && poly_dtype_eq(src[0]->dtype, POLY_VOID)) {
      *out = POLY_VOID;
      return true;
    }
    return false;
  case POLY_OP_CUSTOM:
  case POLY_OP_CUSTOMI:
  case POLY_OP_INS:
  case POLY_OP_NOOP:
    return false;
  case POLY_OP_INDEX:
    if (n_src <= 0) return false;
    *out = param_is_image_shape(src[0]) ? POLY_FLOAT32 : src[0]->dtype;
    return true;
  case POLY_OP_LOAD:
  case POLY_OP_AFTER:
  case POLY_OP_RANGE:
  case POLY_OP_CONTIGUOUS:
  case POLY_OP_CONTIGUOUS_BACKWARD:
  case POLY_OP_COPY:
  case POLY_OP_STAGE:
  case POLY_OP_DETACH:
  case POLY_OP_MSTACK:
  case POLY_OP_MSELECT:
  case POLY_OP_ALLREDUCE:
  case POLY_OP_SPECIAL:
  case POLY_OP_REDUCE:
    if (n_src <= 0) return false;
    *out = src[0]->dtype;
    return true;
  case POLY_OP_CMPLT:
  case POLY_OP_CMPNE:
  case POLY_OP_CMPEQ:
    *out = POLY_BOOL;
    return true;
  case POLY_OP_SIN:
  case POLY_OP_LOG2:
  case POLY_OP_EXP2:
  case POLY_OP_SQRT:
  case POLY_OP_RECIPROCAL:
    if (n_src <= 0) return false;
    if (uop_base_is_invalid(src[0])) {
      *out = POLY_BOOL;
      return true;
    }
    return poly_dtype_least_upper_float(src[0]->dtype, out);
  case POLY_OP_WHERE:
    if (n_src != 3 || !poly_dtype_is_bool(src[0]->dtype)) return false;
    return promo_dtype(src + 1, 2, out);
  case POLY_OP_STACK:
    if (n_src == 0) {
      *out = POLY_VOID;
      return true;
    }
    return promo_dtype(src, n_src, out);
  case POLY_OP_WMMA:
    if (n_src < 3) return false;
    *out = src[2]->dtype;
    return true;
  case POLY_OP_GETTUPLE: {
    if (n_src != 1 || arg.kind != POLY_ARG_INT || arg.i < 0) return false;
    PolyUOp *tuple = src[0];
    if (tuple->op == POLY_OP_FUNCTION && tuple->n_src > 0) tuple = tuple->src[0];
    if (tuple->op != POLY_OP_TUPLE || arg.i >= tuple->n_src) return false;
    *out = tuple->src[arg.i]->dtype;
    return true;
  }
  case POLY_OP_SHL:
  case POLY_OP_SHR:
    if (n_src <= 0) return false;
    for (int i = 0; i < n_src; i++)
      if (!src[i] || (!poly_dtype_is_int(src[i]->dtype) && !uop_base_is_invalid(src[i])))
        return false;
    *out = src[0]->dtype;
    return true;
  case POLY_OP_BINARY:
    *out = POLY_UINT8;
    return true;
  case POLY_OP_CAST:
  case POLY_OP_BITCAST:
    if (arg.kind != POLY_ARG_DTYPE) return false;
    *out = arg.dtype;
    return true;
  default:
    break;
  }

  if (poly_opset_has(POLY_GROUP_UNARY, op)) {
    if (n_src <= 0) return false;
    *out = src[0]->dtype;
    return true;
  }
  if (poly_opset_has(POLY_GROUP_BROADCASTABLE, op)) return promo_dtype(src, n_src, out);
  if (poly_opset_has(POLY_GROUP_MOVEMENT, op)) {
    if (n_src <= 0) return false;
    *out = src[0]->dtype;
    return true;
  }
  return false;
}

PolyDType poly_rebuild_dtype(PolyUOp *u, PolyUOp **new_src) {
  if (!u || (u->n_src > 0 && !new_src)) return u ? u->dtype : POLY_VOID;
  bool same = true;
  for (int i = 0; i < u->n_src; i++)
    if (!poly_dtype_eq(u->src[i]->dtype, new_src[i]->dtype)) {
      same = false;
      break;
    }
  if (same) return u->dtype;
  PolyDType dtype = u->dtype;
  return poly_dtype_from_uop(u->op, new_src, u->n_src, u->arg, u->dtype, &dtype) ? dtype : u->dtype;
}

/* Current UOp.const(b, dtype) routes the Python value through DType.const
 * before hash-consing the CONST (uop/ops.py:609-615, dtype.py:77-82).
 * Keep this scalar-only C carrier shared by const_like's optional dtype and
 * Tensor weak-CONST promotion so changing a weak dtype cannot retain the old
 * integer/float argument representation. */
PolyUOp *poly_uop_const(PolyCtx *ctx, PolyArg val, PolyDType dtype) {
  if (!ctx) return NULL;
  /* Current UOp.const ignores an explicit dtype for Invalid because Invalid
   * is the promotion-lattice bottom represented by a bool CONST. */
  if (val.kind == POLY_ARG_INVALID) dtype = POLY_BOOL;
  if (val.kind == POLY_ARG_INT || val.kind == POLY_ARG_BIGINT || val.kind == POLY_ARG_FLOAT ||
      val.kind == POLY_ARG_BOOL) {
    if (poly_dtype_is_float(dtype)) {
      double dval = (val.kind == POLY_ARG_BIGINT) ? poly_arg_integer_to_double(val)
                    : (val.kind == POLY_ARG_INT)  ? (double)val.i
                    : (val.kind == POLY_ARG_BOOL) ? (val.b ? 1.0 : 0.0)
                                                  : val.f;
      /* Current DType.const rounds concrete floats before UOp CSE
       * (tinygrad/dtype.py:77-82). Weakfloat remains mathematical. */
      PolyArg typed = poly_arg_float(dval);
      if (!poly_dtype_is_weak(dtype)) typed = poly_exec_alu(POLY_OP_CAST, dtype, &typed, 1, true);
      return poly_uop0(ctx, POLY_OP_CONST, dtype, typed);
    } else if (poly_dtype_is_bool(dtype)) {
      bool bval = (val.kind == POLY_ARG_BIGINT)
                      ? poly_arg_integer_to_u64_mod(val) != 0 || val.bigint.n_limbs > 2
                  : (val.kind == POLY_ARG_INT)  ? val.i != 0
                  : (val.kind == POLY_ARG_BOOL) ? val.b
                                                : val.f != 0.0;
      return poly_uop0(ctx, POLY_OP_CONST, dtype, poly_arg_bool(bval));
    } else {
      /* DType.const raises for NaN/Inf-to-int. C has no exception carrier, so
       * retain the original ALU by declining that rewrite. */
      if (val.kind == POLY_ARG_FLOAT && !isfinite(val.f)) return NULL;
      PolyArg ival = val.kind == POLY_ARG_BIGINT ? val
                                                 : poly_arg_int(
                                                       val.kind == POLY_ARG_INT    ? val.i
                                                       : val.kind == POLY_ARG_BOOL ? (val.b ? 1 : 0)
                                                                                   : (int64_t)val.f
                                                   );
      return poly_uop0(ctx, POLY_OP_CONST, dtype, ival);
    }
  }
  return poly_uop0(ctx, POLY_OP_CONST, dtype, val);
}

PolyUOp *poly_const_typed(PolyCtx *ctx, PolyDType dt, double value) {
  return poly_uop_const(ctx, poly_arg_float(value), dt);
}

PolyUOp *poly_const_like_dtype(PolyCtx *ctx, PolyUOp *ref, PolyArg val, PolyDType dtype) {
  if (!ctx || !ref) return NULL;
  PolyUOp *scalar = poly_uop_const(ctx, val, dtype);
  if (!scalar) return NULL;

  /* Current UOp.const_like(b, dtype) creates one scalar CONST and broadcasts
   * it to the reference shape (uop/ops.py:581-583). */
  int ref_ndim = poly_uop_ndim(ctx, ref);
  if (ref_ndim <= 0) return scalar;
  if (ref_ndim > POLY_MAX_DIMS) return NULL;
  PolyUOp *dims[POLY_MAX_DIMS];
  for (int i = 0; i < ref_ndim; i++) {
    dims[i] = poly_uop_shape_dim(ctx, ref, i);
    if (!dims[i]) return NULL;
  }
  return poly_expand_uop(ctx, scalar, dims, ref_ndim);
}

PolyUOp *poly_const_like(PolyCtx *ctx, PolyUOp *ref, PolyArg val) {
  if (!ctx || !ref) return NULL;

  return poly_const_like_dtype(ctx, ref, val, ref->dtype);
}

PolyUOp *poly_const_like_int(PolyCtx *ctx, PolyUOp *ref, int64_t val) {
  return poly_const_like(ctx, ref, poly_arg_int(val));
}

PolyUOp *poly_const_like_float(PolyCtx *ctx, PolyUOp *ref, double val) {
  return poly_const_like(ctx, ref, poly_arg_float(val));
}

PolyUOp *poly_const_like_bool(PolyCtx *ctx, PolyUOp *ref, bool val) {
  return poly_const_like(ctx, ref, poly_arg_bool(val));
}

PolyUOp *poly_identity_element(PolyCtx *ctx, PolyOps op, PolyDType dtype) {
  /* Pinned identity_element returns dtype.const({ADD:0, MUL:1,
   * MAX:dtype.min}) (uop/ops.py:47, dtype.py:82-92). Keep the identity typed;
   * routing floating -infinity through an integer cast is not equivalent. */
  if (!ctx || (op != POLY_OP_ADD && op != POLY_OP_MUL && op != POLY_OP_MAX)) return NULL;
  PolyDType scalar = dtype;
  if (poly_dtype_is_float(scalar)) {
    double value = op == POLY_OP_ADD ? 0.0 : op == POLY_OP_MUL ? 1.0 : -INFINITY;
    return poly_uop0(ctx, POLY_OP_CONST, dtype, poly_arg_float(value));
  }
  if (poly_dtype_is_bool(scalar)) {
    bool value = op == POLY_OP_MUL;
    return poly_uop0(ctx, POLY_OP_CONST, dtype, poly_arg_bool(value));
  }
  if (!poly_dtype_is_int(scalar)) return NULL;
  int64_t value = op == POLY_OP_MUL ? 1 : 0;
  if (op == POLY_OP_MAX && !poly_dtype_is_unsigned(scalar)) {
    int bits = (int)scalar.bitsize;
    value = bits >= 64 ? INT64_MIN : -(INT64_C(1) << (bits - 1));
  }
  return poly_uop0(ctx, POLY_OP_CONST, dtype, poly_arg_int(value));
}

PolyUOp *poly_uop_stack(PolyCtx *ctx, PolyUOp **src, int n_src) {
  if (!ctx || n_src < 0 || n_src > UINT16_MAX || (n_src > 0 && !src)) return NULL;
  if (n_src == 0) return poly_uop(ctx, POLY_OP_STACK, POLY_VOID, NULL, 0, poly_arg_none());

  /* Current UOp._mop(Ops.STACK) derives promo_dtype(src), converts scalar
   * CONST arguments directly to that dtype, and casts other non-invalid
   * values before constructing STACK (uop/ops.py:793-800). */
  PolyDType dtype = src[0]->dtype;
  for (int i = 1; i < n_src; i++)
    if (!src[i] || !poly_dtype_least_upper(dtype, src[i]->dtype, &dtype)) return NULL;

  PolyUOp *inline_src[16];
  PolyUOp **promoted = n_src > (int)(sizeof(inline_src) / sizeof(inline_src[0]))
                           ? malloc((size_t)n_src * sizeof(*promoted))
                           : inline_src;
  if (!promoted) return NULL;
  for (int i = 0; i < n_src; i++) {
    PolyUOp *u = src[i];
    if (uop_base_is_invalid(u) || poly_dtype_eq(u->dtype, dtype)) {
      promoted[i] = u;
    } else if (u->op == POLY_OP_CONST) {
      promoted[i] = poly_uop_const(ctx, u->arg, dtype);
    } else {
      promoted[i] = poly_uop1(ctx, POLY_OP_CAST, dtype, u, poly_arg_none());
    }
    if (!promoted[i]) {
      if (promoted != inline_src) free(promoted);
      return NULL;
    }
  }
  PolyUOp *ret = poly_uop(ctx, POLY_OP_STACK, dtype, promoted, n_src, poly_arg_none());
  if (promoted != inline_src) free(promoted);
  return ret;
}

PolyUOp *poly_uop_index(PolyCtx *ctx, PolyUOp *base, PolyUOp **indices, int n_indices) {
  if (!ctx || !base || n_indices < 0 || n_indices > POLY_MAX_DIMS || (n_indices > 0 && !indices))
    return NULL;

  /* Current UOp.index returns a constant lane of STACK directly.  This is
   * core UOp composition used by codegen devectorization as well as the FFI;
   * frontends must not reimplement it (uop/ops.py:559-562). */
  if (n_indices == 1 && base->op == POLY_OP_STACK && indices[0] &&
      indices[0]->op == POLY_OP_CONST) {
    int64_t lane = 0;
    if (poly_uop_const_i64(indices[0], &lane) != 0) return NULL;
    if (lane < 0) lane += base->n_src;
    return lane >= 0 && lane < base->n_src ? base->src[lane] : NULL;
  }

  PolyUOp *src[POLY_MAX_DIMS + 1];
  src[0] = base;
  for (int i = 0; i < n_indices; i++) {
    if (!indices[i]) return NULL;
    src[1 + i] = indices[i];
  }
  PolyDType out_dt = base->dtype;
  if (!poly_dtype_from_uop(POLY_OP_INDEX, src, n_indices + 1, poly_arg_none(), out_dt, &out_dt))
    return NULL;
  return poly_uop(ctx, POLY_OP_INDEX, out_dt, src, n_indices + 1, poly_arg_none());
}

PolyUOp *poly_uop_placeholder(
    PolyCtx *ctx,
    const int64_t *shape,
    int ndim,
    PolyDType dtype,
    int64_t slot,
    PolyAddrSpace addrspace,
    const char *device,
    bool volatile_
) {
  if (!ctx || ndim < 0 || ndim > POLY_MAX_DIMS || (ndim > 0 && !shape) ||
      (addrspace != POLY_ADDR_GLOBAL && addrspace != POLY_ADDR_LOCAL && addrspace != POLY_ADDR_REG
      ) ||
      (addrspace != POLY_ADDR_GLOBAL && device))
    return NULL;

  int64_t numel = 1;
  for (int i = 0; i < ndim; i++) {
    if (shape[i] < 0 || __builtin_mul_overflow(numel, shape[i], &numel)) return NULL;
  }

  /* Current tinygrad uop/ops.py:UOp.placeholder commits a strong value dtype,
   * stores prod(shape), then restores rank above the storage UOp. */
  dtype = poly_dtype_strong(dtype);
  PolyUOp *extent = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(numel));
  PolyParamArg param = {
      .slot = slot,
      .dtype = dtype,
      .addrspace = addrspace,
      .device = device,
      .volatile_ = volatile_,
  };
  PolyOps op = addrspace == POLY_ADDR_GLOBAL ? POLY_OP_PARAM : POLY_OP_BUFFER;
  PolyUOp *storage = extent ? poly_uop1(ctx, op, dtype, extent, poly_arg_param(&param)) : NULL;
  if (!storage || ndim <= 1) return storage;
  return poly_reshape(ctx, storage, (int64_t *)shape, ndim);
}

PolyUOp *poly_alu1(PolyCtx *ctx, PolyOps op, PolyUOp *src) {
  if (!ctx || !src) return NULL;
  PolyDType dtype = src->dtype;
  if (op == POLY_OP_SIN || op == POLY_OP_LOG2 || op == POLY_OP_EXP2 || op == POLY_OP_SQRT ||
      op == POLY_OP_RECIPROCAL) {
    /* Current UOp dtype inference keeps the operand graph intact and derives
     * the result type here (uop/ops.py:144-145,758-761; dtype.py:187-188). */
    if (uop_base_is_invalid(src)) {
      dtype = POLY_BOOL;
    } else if (!poly_dtype_least_upper_float(src->dtype, &dtype)) {
      return NULL;
    }
  }
  return poly_uop1(ctx, op, dtype, src, poly_arg_none());
}

PolyUOp *poly_alu2(PolyCtx *ctx, PolyOps op, PolyUOp *a, PolyUOp *b) {
  if (!ctx || !a || !b) return NULL;
  PolyDType dt;
  if (op == POLY_OP_SHL || op == POLY_OP_SHR) {
    PolyUOp *src[] = {a, b};
    if (!poly_dtype_from_uop(op, src, 2, poly_arg_none(), POLY_VOID, &dt)) return NULL;
  } else if (op == POLY_OP_CMPLT || op == POLY_OP_CMPNE || op == POLY_OP_CMPEQ) {
    dt = POLY_BOOL;
  } else {
    dt = a->dtype;
    if (!poly_dtype_eq(a->dtype, b->dtype) && !poly_dtype_least_upper(a->dtype, b->dtype, &dt))
      return NULL;
  }
  return poly_uop2(ctx, op, dt, a, b, poly_arg_none());
}

PolyUOp *poly_alu3(PolyCtx *ctx, PolyOps op, PolyUOp *a, PolyUOp *b, PolyUOp *c) {
  PolyDType dt = a->dtype;
  if (op == POLY_OP_WHERE) {
    if (!poly_dtype_is_bool(a->dtype)) return NULL;
    dt = b->dtype;
    if (!poly_dtype_eq(b->dtype, c->dtype) && !poly_dtype_least_upper(b->dtype, c->dtype, &dt))
      return NULL;
  }
  return poly_uop3(ctx, op, dt, a, b, c, poly_arg_none());
}

/* Tinygrad 2026-08-22/a9069c177a9d UOp.cast keeps lanes in UOp shape, not
 * DType, and returns self on exact scalar dtype identity (uop/ops.py:787). */
PolyUOp *poly_cast(PolyCtx *ctx, PolyUOp *x, PolyDType target) {
  if (!ctx || !x) return NULL;
  if (poly_dtype_eq(x->dtype, target)) return x;
  return poly_uop1(ctx, POLY_OP_CAST, target, x, poly_arg_none());
}

/* Current Tinygrad UOp.param/param_like (tinygrad/uop/ops.py:1158-1162). */
PolyUOp *poly_uop_param(PolyCtx *ctx, int slot, PolyUOp *like) {
  PolyUOp *variable = poly_uop_is_bound_var(like) ? like->src[0] : like;
  if (ctx && slot >= 0 && poly_uop_is_variable(variable)) {
    PolyParamArg param_arg = *variable->arg.param;
    char name[32];
    snprintf(name, sizeof(name), "p%d", slot);
    param_arg.slot = slot;
    param_arg.name = name;
    return poly_uop(
        ctx, POLY_OP_PARAM, variable->dtype, variable->src, variable->n_src,
        poly_arg_param(&param_arg)
    );
  }
  if (!ctx || slot < 0 || !like || poly_dtype_is_weak(like->dtype)) return NULL;

  int ndim = poly_uop_ndim(ctx, like);
  const int64_t *max_shape = poly_uop_max_shape_dims(ctx, like);
  if (ndim < 0 || ndim > POLY_MAX_DIMS || (ndim > 0 && !max_shape)) return NULL;
  PolyUOp *dim_src[POLY_MAX_DIMS];
  for (int i = 0; i < ndim; i++) {
    dim_src[i] = poly_uop_shape_dim(ctx, like, i);
    if (!dim_src[i]) dim_src[i] = poly_const_int(ctx, max_shape[i]);
    if (!dim_src[i]) return NULL;
  }
  PolyUOp *shape = poly_shape_to_shape_arg(ctx, dim_src, ndim);
  if (!shape) return NULL;

  PolyUOp *device = poly_uop_device_uop_cached(ctx, like, NULL);
  PolyParamArg param_arg = {
      .slot = slot,
      .name = NULL,
      .min_val = 0,
      .max_val = 0,
      .has_minmax = false,
      .addrspace = POLY_ADDR_GLOBAL,
      .axis = 0,
      .has_axis = false,
      .device = device && device->arg.kind == POLY_ARG_STRING ? device->arg.str : NULL,
      .devices = device && device->arg.kind == POLY_ARG_STRING_TUPLE ? device->arg.string_tuple.vals
                                                                     : NULL,
      .n_devices =
          device && device->arg.kind == POLY_ARG_STRING_TUPLE ? device->arg.string_tuple.n : 0,
      .device_is_tuple = device && device->arg.kind == POLY_ARG_STRING_TUPLE,
  };
  PolyUOp *src[1] = {shape};
  return poly_uop(ctx, POLY_OP_PARAM, like->dtype, src, 1, poly_arg_param(&param_arg));
}

PolyUOp *poly_uop_variable(
    PolyCtx *ctx,
    const char *name,
    int64_t min_val,
    int64_t max_val,
    PolyDType dtype,
    int64_t multiple_of,
    bool param
) {
  if (!ctx || !name || min_val > max_val || multiple_of <= 0) return NULL;
  PolyUOp *shape = poly_uop0(ctx, POLY_OP_STACK, POLY_VOID, poly_arg_none());
  PolyParamArg arg = {
      .slot = -1,
      .dtype = dtype,
      .name = name,
      .min_val = min_val,
      .max_val = max_val,
      .has_minmax = true,
      .multiple_of = multiple_of,
      .has_multiple_of = true,
      .addrspace = POLY_ADDR_ALU,
  };
  return shape ? poly_uop1(
                     ctx, param ? POLY_OP_PARAM : POLY_OP_BUFFER, dtype, shape, poly_arg_param(&arg)
                 )
               : NULL;
}

bool poly_uop_is_variable(const PolyUOp *u) {
  return u && u->op == POLY_OP_BUFFER && u->n_src == 1 && u->src[0] &&
         u->src[0]->op == POLY_OP_STACK && u->src[0]->n_src == 0 && u->arg.kind == POLY_ARG_PARAM &&
         u->arg.param && u->arg.param->has_minmax && u->arg.param->addrspace == POLY_ADDR_ALU;
}

bool poly_uop_is_bound_var(const PolyUOp *u) {
  return u && u->op == POLY_OP_AFTER && u->n_src == 2 && poly_uop_is_variable(u->src[0]) &&
         u->src[1] && u->src[1]->op == POLY_OP_STORE && u->src[1]->n_src == 2 &&
         u->src[1]->src[0] == u->src[0] && u->src[1]->src[1] &&
         u->src[1]->src[1]->op == POLY_OP_CONST;
}

bool poly_uop_is_alu_param(const PolyUOp *u) {
  return u && u->op == POLY_OP_PARAM && u->arg.kind == POLY_ARG_PARAM && u->arg.param &&
         u->arg.param->addrspace == POLY_ADDR_ALU;
}

/* Approved Polygrad logical/placement boundary. Physical execution must use
 * current Tinygrad UOp.new_buffer below, never this deviceless representation. */
PolyUOp *poly_uop_new_logical_buffer_with_slot(
    PolyCtx *ctx,
    PolyDType dtype,
    int64_t size,
    int64_t slot
) {
  if (!ctx || size < 0 || poly_dtype_is_weak(dtype)) return NULL;
  PolyUOp *unique = poly_uop0(ctx, POLY_OP_UNIQUE, POLY_VOID, poly_arg_int(slot));
  return unique ? poly_uop1(ctx, POLY_OP_BUFFER, dtype, unique, poly_arg_int(size)) : NULL;
}

PolyUOp *poly_uop_new_logical_buffer(PolyCtx *ctx, PolyDType dtype, int64_t size) {
  return ctx ? poly_uop_new_logical_buffer_with_slot(ctx, dtype, size, poly_ctx_next_unique_id(ctx))
             : NULL;
}

/* Current tinygrad uop/ops.py:UOp.new_buffer.  The DEVICE UOp is C-only
 * transport for the same str/tuple ParamArg.device value. */
PolyUOp *poly_uop_new_buffer(
    PolyCtx *ctx,
    PolyUOp *device,
    int64_t size,
    PolyDType dtype,
    int64_t slot
) {
  if (!ctx || !device || device->op != POLY_OP_DEVICE || size < 0 || poly_dtype_is_weak(dtype) ||
      (device->arg.kind != POLY_ARG_STRING && device->arg.kind != POLY_ARG_STRING_TUPLE) ||
      (device->arg.kind == POLY_ARG_STRING_TUPLE && device->arg.string_tuple.n <= 0))
    return NULL;
  PolyUOp *shape = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(size));
  PolyParamArg param = {
      .slot = slot,
      .dtype = dtype,
      .addrspace = POLY_ADDR_GLOBAL,
      .device = device->arg.kind == POLY_ARG_STRING ? device->arg.str : NULL,
      .devices = device->arg.kind == POLY_ARG_STRING_TUPLE ? device->arg.string_tuple.vals : NULL,
      .n_devices = device->arg.kind == POLY_ARG_STRING_TUPLE ? device->arg.string_tuple.n : 0,
      .device_is_tuple = device->arg.kind == POLY_ARG_STRING_TUPLE,
  };
  return shape ? poly_uop1(ctx, POLY_OP_BUFFER, dtype, shape, poly_arg_param(&param)) : NULL;
}

const char *poly_uop_expr(const PolyUOp *u) {
  if (poly_uop_is_bound_var(u)) u = u->src[0];
  if (!u || (u->op != POLY_OP_PARAM && u->op != POLY_OP_BUFFER) || u->arg.kind != POLY_ARG_PARAM ||
      !u->arg.param)
    return NULL;
  return u->arg.param->name;
}

PolyUOp *poly_uop_bind(PolyCtx *ctx, PolyUOp *var, int64_t value) {
  if (!ctx || !poly_uop_is_variable(var)) return NULL;
  const PolyParamArg *arg = var->arg.param;
  if (value < arg->min_val || value > arg->max_val ||
      (arg->has_multiple_of && value % arg->multiple_of != 0))
    return NULL;
  PolyUOp *val = poly_uop0(ctx, POLY_OP_CONST, var->dtype, poly_arg_int(value));
  PolyUOp *store = val ? poly_store_val(ctx, var, val) : NULL;
  return store ? poly_uop2(ctx, POLY_OP_AFTER, var->dtype, var, store, poly_arg_none()) : NULL;
}

PolyUOp *poly_store_val(PolyCtx *ctx, PolyUOp *buf, PolyUOp *value) {
  return poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, buf, value, poly_arg_none());
}

PolyUOp *poly_sink1(PolyCtx *ctx, PolyUOp *store) {
  return poly_uop(ctx, POLY_OP_SINK, POLY_VOID, &store, 1, poly_arg_none());
}

PolyUOp *poly_sink_n(PolyCtx *ctx, PolyUOp **stores, int n) {
  return poly_uop(ctx, POLY_OP_SINK, POLY_VOID, stores, n, poly_arg_none());
}
