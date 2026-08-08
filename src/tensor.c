/*
 * tensor.c -- Composed tensor ops (elementwise, reduction, creation, etc.)
 *
 * These are higher-level ops built from the core UOp primitives.
 */

#define _GNU_SOURCE
#include "tensor.h"
#include "bigint.h"
#include "ctx.h"
#include "device.h"
#include "engine/schedule.h"
#include "frontend_internal.h"
#include "utils.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_LN2
#define M_LN2 0.693147180559945309417
#endif

/* Dtype table for FFI (used by poly_cast_by_id) */

static const PolyDType *_dtype_table_ffi[] = {
    &POLY_VOID,    &POLY_BOOL,     &POLY_INT8,    &POLY_UINT8,   &POLY_INT16,
    &POLY_UINT16,  &POLY_INT32,    &POLY_UINT32,  &POLY_INT64,   &POLY_UINT64,
    &POLY_FLOAT16, &POLY_BFLOAT16, &POLY_FLOAT32, &POLY_FLOAT64,
};
#define N_DTYPE_FFI ((int)(sizeof(_dtype_table_ffi) / sizeof(_dtype_table_ffi[0])))

PolyUOp *poly_buffer_var(
    PolyCtx *ctx,
    PolyDType dt,
    PolyUOp *batch_var,
    const int64_t *inner_dims,
    int n_inner
) {
  if (!ctx || !batch_var || n_inner < 0 || n_inner >= POLY_MAX_DIMS || (n_inner > 0 && !inner_dims))
    return NULL;
  if (batch_var->op != POLY_OP_DEFINE_VAR && batch_var->op != POLY_OP_BIND) return NULL;
  PolyUOp *bound =
      batch_var->op == POLY_OP_BIND && batch_var->n_src >= 1 ? batch_var->src[0] : batch_var;
  if (!bound || bound->op != POLY_OP_DEFINE_VAR) return NULL;
  int64_t alloc = bound->arg.define_var.max_val;
  /* src[0] = UNIQUE (prevent CSE), src[1] = dynamic bound,
   * src[2..] = fixed inner dimension CONSTs. */
  int n_src = 2 + n_inner;
  PolyUOp *src[POLY_MAX_DIMS + 2];
  src[0] = poly_uop0(ctx, POLY_OP_UNIQUE, POLY_VOID, poly_arg_int(poly_ctx_next_unique_id(ctx)));
  src[1] = batch_var;
  for (int i = 0; i < n_inner; i++) {
    if (inner_dims[i] < 0) return NULL;
    if (inner_dims[i] != 0 && alloc > INT64_MAX / inner_dims[i]) return NULL;
    alloc *= inner_dims[i];
    src[2 + i] = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(inner_dims[i]));
  }
  return poly_uop(ctx, POLY_OP_BUFFER, dt, src, n_src, poly_arg_int(alloc));
}

PolyUOp *poly_store_buffer_update(PolyCtx *ctx, PolyUOp *target, PolyUOp *value) {
  /* Full-buffer assignment only. Movement views are normalized to their base
   * BUFFER for optimizer/direct core paths that update whole storage objects. */
  PolyUOp *base = target;
  while (poly_opset_has(POLY_GROUP_MOVEMENT, base->op) && base->n_src > 0)
    base = base->src[0];

  if (base != target && base->op == POLY_OP_BUFFER) {
    int64_t numel = (base->arg.kind == POLY_ARG_INT) ? base->arg.i : 0;
    if (numel > 0) {
      int64_t flat_shape[1] = {numel};
      value = poly_reshape(ctx, value, flat_shape, 1);
    }
    target = base;
  }

  return poly_store_val(ctx, target, value);
}

/* Core PolyTensor handles */

typedef struct {
  PolyTensor **items;
  int n;
  int cap;
} PolyTensorList;

static PolyUOp *tensor_current_uop(PolyTensor *tensor) {
  return tensor ? (tensor->uop_physical ? tensor->uop_physical : tensor->uop_logical) : NULL;
}

static bool tensor_roots_owned_by_ctx(PolyCtx *ctx, const PolyTensor *tensor) {
  /* Polygrad-specific arena boundary: never intern a destination DAG whose
   * sources point into another context. Dual-root construction requires both exact roots. */
  return ctx && tensor && tensor->uop_logical && tensor->uop_physical &&
         poly_ctx_owns_ptr(ctx, tensor->uop_logical) &&
         poly_ctx_owns_ptr(ctx, tensor->uop_physical);
}

static PolyTensorList *tensor_list_for_uop(PolyCtx *ctx, PolyUOp *uop) {
  if (!ctx || !uop) return NULL;
  return poly_map_get(ctx->tensors_by_uop, poly_ptr_hash(uop), uop, poly_ptr_eq);
}

static bool tensor_list_append(PolyTensorList *list, PolyTensor *tensor) {
  if (!list || !tensor) return false;
  for (int i = 0; i < list->n; i++)
    if (list->items[i] == tensor) return true;
  if (list->n >= list->cap) {
    int new_cap = list->cap ? list->cap * 2 : 4;
    PolyTensor **new_items = realloc(list->items, (size_t)new_cap * sizeof(PolyTensor *));
    if (!new_items) return false;
    list->items = new_items;
    list->cap = new_cap;
  }
  list->items[list->n++] = tensor;
  return true;
}

static bool tensor_index_add(PolyCtx *ctx, PolyUOp *current, PolyTensor *tensor) {
  if (!ctx || !current || !tensor) return false;
  PolyTensorList *list = tensor_list_for_uop(ctx, current);
  if (list) return tensor_list_append(list, tensor);

  /* Allocate and populate before publishing the list. A failed first append
   * must not leave an empty tensors_by_uop entry. */
  list = calloc(1, sizeof(PolyTensorList));
  if (!list) return false;
  if (!tensor_list_append(list, tensor)) {
    free(list);
    return false;
  }
  poly_map_set(ctx->tensors_by_uop, poly_ptr_hash(current), current, list, poly_ptr_eq);
  return true;
}

static bool tensor_index_add_current(PolyCtx *ctx, PolyTensor *tensor) {
  return tensor_index_add(ctx, tensor_current_uop(tensor), tensor);
}

static void tensor_index_remove(PolyCtx *ctx, PolyUOp *current, PolyTensor *tensor) {
  if (!ctx || !current || !tensor) return;
  PolyTensorList *list = tensor_list_for_uop(ctx, current);
  if (!list) return;
  for (int i = 0; i < list->n; i++) {
    if (list->items[i] != tensor) continue;
    list->items[i] = list->items[list->n - 1];
    list->n--;
    if (list->n == 0) {
      poly_map_remove(ctx->tensors_by_uop, poly_ptr_hash(current), current, poly_ptr_eq);
      free(list->items);
      free(list);
    }
    return;
  }
}

/* Commit fields only after the caller has registered tensor at its future
 * current root. This phase allocates nothing and cannot leave a half-indexed
 * Tensor on recoverable tensor-list allocation failure. */
static void tensor_replace_roots_commit_reserved(
    PolyCtx *ctx,
    PolyTensor *tensor,
    PolyUOp *uop_logical,
    PolyUOp *uop_physical,
    PolyTensorRole role,
    PolyDevice device
) {
  PolyUOp *old_current = tensor_current_uop(tensor);
  tensor->uop_logical = uop_logical;
  tensor->uop_physical = uop_physical;
  tensor->role = role;
  if (device != POLY_DEVICE_AUTO) tensor->device = device;
  if (role != POLY_TENSOR_PLACE) tensor->source = NULL;

  PolyUOp *new_current = tensor_current_uop(tensor);
  if (old_current != new_current) tensor_index_remove(ctx, old_current, tensor);
}

static bool tensor_assign_anchor_op(PolyOps op) {
  return poly_opset_has(POLY_GROUP_MOVEMENT, op) || op == POLY_OP_BITCAST;
}

static PolyUOp *tensor_assign_view_anchor(PolyUOp *u) {
  if (!u || poly_uop_has_buffer_identity(u)) return NULL;

  /* tinygrad retargets view assigns at the nearest buffer-identity level:
   * SHRINK(BUFFER) maps BUFFER -> AFTER(BUFFER, assign), while
   * PERMUTE(RESHAPE(BUFFER)) maps RESHAPE(BUFFER) -> AFTER(...).  Stop at
   * AFTER too, because a second pending view write chains onto that boundary. */
  PolyUOp *cur = u;
  while (cur && !poly_uop_has_buffer_identity(cur)) {
    if (cur->op == POLY_OP_AFTER) return cur != u ? cur : NULL;
    if (!tensor_assign_anchor_op(cur->op) || cur->n_src < 1) return NULL;
    cur = cur->src[0];
  }
  return (cur && cur != u) ? cur : NULL;
}

static PolyUOp *tensor_substitute_once(
    PolyCtx *ctx,
    PolyUOp *u,
    PolyUOp *from,
    PolyUOp *to,
    PolyMap *memo
) {
  if (!ctx || !u) return NULL;
  if (u == from) return to;

  PolyUOp *cached = poly_map_get(memo, poly_ptr_hash(u), u, poly_ptr_eq);
  if (cached) return cached;

  PolyUOp *stack_src[16];
  PolyUOp **new_src = (u->n_src > (int)(sizeof(stack_src) / sizeof(stack_src[0])))
                          ? malloc((size_t)u->n_src * sizeof(PolyUOp *))
                          : stack_src;
  if (!new_src) return NULL;

  bool changed = false;
  for (int i = 0; i < u->n_src; i++) {
    new_src[i] = tensor_substitute_once(ctx, u->src[i], from, to, memo);
    if (!new_src[i]) {
      if (new_src != stack_src) free(new_src);
      return NULL;
    }
    if (new_src[i] != u->src[i]) changed = true;
  }

  PolyUOp *result = changed ? poly_uop(ctx, u->op, u->dtype, new_src, u->n_src, u->arg) : u;
  poly_map_set(memo, poly_ptr_hash(u), u, result, poly_ptr_eq);
  if (new_src != stack_src) free(new_src);
  return result;
}

static int tensor_retarget_logical_uops(
    PolyCtx *ctx,
    PolyUOp *from,
    PolyUOp *to,
    PolyDevice device
) {
  if (!ctx || !from || !to || from == to) return 0;

  for (int i = 0; i < ctx->n_tensors; i++) {
    PolyTensor *t = ctx->tensors[i];
    if (!t) continue;
    if (device != POLY_DEVICE_AUTO && t->device != POLY_DEVICE_AUTO &&
        !poly_devices_share_storage(t->device, device))
      continue;

    PolyUOp *current = tensor_current_uop(t);
    if (!current) continue;
    PolyMap *memo = poly_map_new(64);
    if (!memo) return -1;
    /* View-assign replacements are intentionally self-containing
     * (BUFFER -> AFTER(BUFFER, STORE(...))). Do not recurse into the
     * replacement itself, matching tinygrad's substitute map behavior. */
    PolyUOp *new_current = tensor_substitute_once(ctx, current, from, to, memo);
    poly_map_destroy(memo);
    if (!new_current || new_current == current) continue;

    /* View assign is a semantic effect rewrite, not a runtime materialization
     * retarget. Keep it in logical IR as AFTER/STORE, matching tinygrad's
     * Tensor.assign graph shape. */
    if (poly_tensor_replace_roots(ctx, t, new_current, NULL, t->role, t->device) != 0) return -1;
  }
  return 0;
}

enum {
  TENSOR_MAP_SCOPE_EXACT = 1,
  TENSOR_MAP_SCOPE_PLACED = 2,
};

static int tensor_map_scope_node(PolyUOp *logical, PolyMap *exact, PolyMap *placement_memo) {
  int scope = 0;
  if (!logical || !exact) return scope;
  if (poly_map_get(exact, poly_ptr_hash(logical), logical, poly_ptr_eq))
    scope |= TENSOR_MAP_SCOPE_EXACT;
  if (placement_memo) {
    PolyUOp *placed = poly_map_get(placement_memo, poly_ptr_hash(logical), logical, poly_ptr_eq);
    if (placed && poly_map_get(exact, poly_ptr_hash(placed), placed, poly_ptr_eq))
      scope |= TENSOR_MAP_SCOPE_PLACED;
  }
  return scope;
}

typedef struct {
  PolyMap *scope;
} TensorMapScopeGate;

static bool tensor_map_scope_unseen(PolyUOp *u, void *user_data) {
  TensorMapScopeGate *gate = user_data;
  return u && gate && gate->scope && !poly_map_get(gate->scope, poly_ptr_hash(u), u, poly_ptr_eq);
}

static bool tensor_map_scope_visit(
    PolyCtx *ctx,
    PolyUOp *root,
    PolyMap *exact,
    PolyMap *placement_memo,
    PolyMap *scope_map
) {
  if (!ctx || !root || !exact || !scope_map) return false;
  if (poly_map_get(scope_map, poly_ptr_hash(root), root, poly_ptr_eq)) return true;

  int n_topo = 0;
  PolyScratchMark scratch = poly_ctx_scratch_mark(ctx);
  TensorMapScopeGate gate = {.scope = scope_map};
  PolyUOp **topo =
      poly_toposort_ex_user_scratch(ctx, root, &n_topo, tensor_map_scope_unseen, &gate, true);
  if (!topo || n_topo <= 0) {
    poly_ctx_scratch_rewind(ctx, scratch);
    return false;
  }
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    int scope = tensor_map_scope_node(u, exact, placement_memo);
    for (int j = 0; j < u->n_src; j++) {
      void *child = poly_map_get(scope_map, poly_ptr_hash(u->src[j]), u->src[j], poly_ptr_eq);
      if (child) {
        int child_scope = (int)((uintptr_t)child - 1);
        /* Scope discovery enters opaque bodies like tinygrad topovisit, so an
         * exact map key still reaches its owning CALL/FUNCTION. A placed-only
         * key inside src[0] must not nominate projection of that owner: the
         * body is already physical and is identity-pinned during substitution. */
        if ((u->op == POLY_OP_CALL || u->op == POLY_OP_FUNCTION) && j == 0)
          child_scope &= TENSOR_MAP_SCOPE_EXACT;
        scope |= child_scope;
      }
    }
    poly_map_set(scope_map, poly_ptr_hash(u), u, (void *)(uintptr_t)(scope + 1), poly_ptr_eq);
  }
  poly_ctx_scratch_rewind(ctx, scratch);
  return poly_map_get(scope_map, poly_ptr_hash(root), root, poly_ptr_eq) != NULL;
}

static int tensor_map_scope_get(PolyMap *scope_map, PolyUOp *root) {
  if (!scope_map || !root) return 0;
  void *scope = poly_map_get(scope_map, poly_ptr_hash(root), root, poly_ptr_eq);
  return scope ? (int)((uintptr_t)scope - 1) : 0;
}

int poly_tensor_apply_realize_map(
    PolyCtx *ctx,
    PolyUOp **from,
    PolyUOp **to,
    int n,
    PolyDevice device,
    PolyMap **placement_memo
) {
  if (!ctx || n < 0 || (n > 0 && (!from || !to))) return -1;
  if (n == 0) return 0;

  /* Pinned _apply_map_to_tensors filters live roots through one shared scope
   * cache, substitutes one temporary aggregate with one rewrite memo, and only
   * then mutates the Tensor wrappers. Polygrad's ctx-lifetime UOp arena cannot
   * intern that temporary SINK without retaining all of its roots, so use one
   * private multi-root substitution memo instead. */
  int n_tensors = ctx->n_tensors;
  if (n_tensors <= 0) return 0;
  size_t n_slots = (size_t)n_tensors;
  if (n_slots > SIZE_MAX / 2) return -1;
  size_t n_rewrite_slots = n_slots * 2;

  PolyUOp **updates = calloc((size_t)n_tensors, sizeof(*updates));
  PolyTensor **snapshot_tensors = calloc(n_slots, sizeof(*snapshot_tensors));
  PolyUOp **snapshot_roots = calloc(n_slots, sizeof(*snapshot_roots));
  PolyDevice *snapshot_devices = calloc(n_slots, sizeof(*snapshot_devices));
  int *snapshot_indices = calloc(n_slots, sizeof(*snapshot_indices));
  PolyUOp **ordinary_results = calloc(n_slots, sizeof(*ordinary_results));
  PolyTensor **project_tensors = calloc((size_t)n_tensors, sizeof(*project_tensors));
  int *project_slots = calloc((size_t)n_tensors, sizeof(*project_slots));
  PolyUOp **project_roots = calloc(n_slots, sizeof(*project_roots));
  PolyUOp **project_results = calloc(n_slots, sizeof(*project_results));
  PolyUOp **rewrite_roots = calloc(n_rewrite_slots, sizeof(*rewrite_roots));
  PolyUOp **rewrite_results = calloc(n_rewrite_slots, sizeof(*rewrite_results));
  int *rewrite_slots = calloc(n_rewrite_slots, sizeof(*rewrite_slots));
  int *rewrite_aux = calloc(n_rewrite_slots, sizeof(*rewrite_aux));
  uint8_t *rewrite_kind = calloc(n_rewrite_slots, sizeof(*rewrite_kind));

  int rc = -1;
  PolyMap *exact = NULL;
  PolyMap *scope_maps[POLY_DEVICE_DISK + 1] = {0};
  if (!updates || !snapshot_tensors || !snapshot_roots || !snapshot_devices || !snapshot_indices ||
      !ordinary_results || !project_tensors || !project_slots || !project_roots ||
      !project_results || !rewrite_roots || !rewrite_results || !rewrite_slots || !rewrite_aux ||
      !rewrite_kind)
    goto cleanup;

  int n_snapshot = 0;
  for (int i = 0; i < n_tensors; i++) {
    PolyTensor *tensor = ctx->tensors[i];
    if (!tensor) continue;
    PolyDevice resolved_device = poly_tensor_resolved_device(ctx, tensor);
    if (resolved_device <= POLY_DEVICE_AUTO || resolved_device > POLY_DEVICE_DISK) goto cleanup;
    if (device != POLY_DEVICE_AUTO && resolved_device != device) continue;
    PolyUOp *current = tensor_current_uop(tensor);
    if (!current) continue;
    snapshot_tensors[n_snapshot] = tensor;
    snapshot_roots[n_snapshot] = current;
    snapshot_devices[n_snapshot] = resolved_device;
    snapshot_indices[n_snapshot++] = i;
  }

  if (n_snapshot == 0) {
    rc = 0;
    goto cleanup;
  }
  exact = poly_map_new((size_t)n * 2 + 16);
  if (!exact) goto cleanup;
  for (int i = 0; i < n; i++) {
    if (!from[i] || !to[i] || from[i] == to[i]) continue;
    poly_map_set(exact, poly_ptr_hash(from[i]), from[i], (void *)(uintptr_t)1, poly_ptr_eq);
  }
  if (poly_map_len(exact) == 0) {
    rc = 0;
    goto cleanup;
  }

  int n_rewrite = 0;
  int n_project = 0;
  for (int slot = 0; slot < n_snapshot; slot++) {
    PolyTensor *tensor = snapshot_tensors[slot];
    PolyUOp *current = snapshot_roots[slot];
    PolyDevice resolved_device = snapshot_devices[slot];
    PolyMap *scope_map = scope_maps[resolved_device];
    if (!scope_map) {
      scope_map = scope_maps[resolved_device] = poly_map_new((size_t)n_snapshot * 2 + 16);
      if (!scope_map) goto cleanup;
    }
    PolyMap *device_memo = placement_memo ? placement_memo[resolved_device] : NULL;
    if (!tensor_map_scope_visit(ctx, current, exact, device_memo, scope_map)) goto cleanup;
    int scope = tensor_map_scope_get(scope_map, current);
    int root_scope = tensor_map_scope_node(current, exact, device_memo);
    /* tinygrad applies the map to its one current, already-placed Tensor.uop
     * (tensor.py:202-206). A Polygrad PLACE can instead retain the source's
     * physical root while its exact target projection is keyed by the
     * preserved logical root. Admit only that existing per-device exact
     * logical->physical proof; ordinary substitution scope remains current. */
    int logical_placement_scope =
        tensor_map_scope_node(tensor->uop_logical, exact, device_memo) & TENSOR_MAP_SCOPE_PLACED;
    root_scope |= logical_placement_scope;
    scope |= logical_placement_scope;
    ordinary_results[slot] = current;
    if (scope & TENSOR_MAP_SCOPE_EXACT) {
      rewrite_roots[n_rewrite] = current;
      rewrite_slots[n_rewrite] = slot;
      rewrite_kind[n_rewrite++] = 0;
    }

    /* Pinned tinygrad has one physical Tensor root, so an exact root map hit
     * cannot be displaced by a different exact hit in one of its descendants.
     * At Polygrad's boundary, prefer the existing direct logical->physical
     * memo correspondence when only that physical root is a callify map key. */
    bool exact_root_placement =
        (root_scope & TENSOR_MAP_SCOPE_PLACED) && !(root_scope & TENSOR_MAP_SCOPE_EXACT);
    bool needs_placed_map = exact_root_placement || !(scope & TENSOR_MAP_SCOPE_EXACT) ||
                            (current->op == POLY_OP_AFTER && current->n_src >= 1 &&
                             !poly_uop_has_buffer_identity(current->src[0]));
    /* tinygrad stores placement COPYs directly in Tensor.uop. Polygrad keeps
     * portable logical roots, so retry only roots with map-derived placement
     * evidence. PLACE role by itself is intentionally insufficient: unrelated
     * pending placements are outside tinygrad's live-map scope. */
    if (needs_placed_map && (scope & TENSOR_MAP_SCOPE_PLACED)) {
      project_tensors[n_project] = tensor;
      project_slots[n_project++] = slot;
    }
  }

  if (n_project > 0 && poly_tensor_physicalize_many(
                           ctx, project_tensors, n_project, project_roots, placement_memo
                       ) != 0)
    goto cleanup;

  for (int projection = 0; projection < n_project; projection++) {
    PolyUOp *placed = project_roots[projection];
    project_results[projection] = placed;
    int slot = project_slots[projection];
    PolyDevice resolved_device = snapshot_devices[slot];
    PolyMap *scope_map = scope_maps[resolved_device];
    PolyMap *device_memo = placement_memo ? placement_memo[resolved_device] : NULL;
    if (!tensor_map_scope_visit(ctx, placed, exact, device_memo, scope_map)) goto cleanup;
    if (!(tensor_map_scope_get(scope_map, placed) & TENSOR_MAP_SCOPE_EXACT)) continue;
    rewrite_roots[n_rewrite] = placed;
    rewrite_slots[n_rewrite] = slot;
    rewrite_aux[n_rewrite] = projection;
    rewrite_kind[n_rewrite++] = 1;
  }

  if (n_rewrite == 0) {
    rc = 0;
    goto cleanup;
  }
  if (poly_uop_substitute_many(ctx, rewrite_roots, n_rewrite, from, to, n, rewrite_results) != 0)
    goto cleanup;

  for (int rewrite = 0; rewrite < n_rewrite; rewrite++) {
    if (rewrite_kind[rewrite])
      project_results[rewrite_aux[rewrite]] = rewrite_results[rewrite];
    else
      ordinary_results[rewrite_slots[rewrite]] = rewrite_results[rewrite];
  }

  for (int slot = 0; slot < n_snapshot; slot++) {
    PolyUOp *current = snapshot_roots[slot];
    PolyUOp *realized = ordinary_results[slot];
    if (realized != current) updates[snapshot_indices[slot]] = realized;
  }
  for (int projection = 0; projection < n_project; projection++) {
    int slot = project_slots[projection];
    PolyUOp *current = snapshot_roots[slot];
    PolyUOp *placed = project_roots[projection];
    PolyUOp *placed_realized = project_results[projection];
    PolyDevice resolved_device = snapshot_devices[slot];
    PolyMap *device_memo = placement_memo ? placement_memo[resolved_device] : NULL;
    int root_scope = tensor_map_scope_node(current, exact, device_memo);
    root_scope |= tensor_map_scope_node(snapshot_tensors[slot]->uop_logical, exact, device_memo) &
                  TENSOR_MAP_SCOPE_PLACED;
    bool exact_root_placement =
        (root_scope & TENSOR_MAP_SCOPE_PLACED) && !(root_scope & TENSOR_MAP_SCOPE_EXACT);
    bool needs_placed_map = exact_root_placement || ordinary_results[slot] == current ||
                            (current->op == POLY_OP_AFTER && current->n_src >= 1 &&
                             !poly_uop_has_buffer_identity(current->src[0]));
    if (needs_placed_map && placed_realized != placed && placed_realized != current)
      updates[snapshot_indices[slot]] = placed_realized;
  }

  /* Reserve every future index row before mutating any live Tensor. The row is
   * temporarily ignored by lookup because tensor_current_uop still names its
   * old root. Rollback is therefore allocation-free and restores the exact
   * pre-commit registry if a later reservation fails. */
  for (int i = 0; i < n_tensors; i++) {
    PolyTensor *tensor = ctx->tensors[i];
    PolyUOp *realized = updates[i];
    if (!tensor || !realized) continue;
    if (!tensor_index_add(ctx, realized, tensor)) {
      for (int j = 0; j < i; j++) {
        if (ctx->tensors[j] && updates[j]) tensor_index_remove(ctx, updates[j], ctx->tensors[j]);
      }
      goto cleanup;
    }
  }

  for (int i = 0; i < n_tensors; i++) {
    PolyTensor *tensor = ctx->tensors[i];
    PolyUOp *realized = updates[i];
    if (!tensor || !realized) continue;
    /* tinygrad applies transform_to_call's becomes_map to every live Tensor.
     * Preserve Polygrad's export/provenance graph as uop_logical and install
     * that rewritten, executable counterpart only as uop_physical. tinygrad
     * publishes a mapped BUFFER before schedule creation and before JIT
     * capture executes it. At Polygrad's placement boundary, an exact
     * target-device buffer identity therefore completes the existing PLACE
     * graph lifecycle without claiming runtime residency. */
    PolyTensorRole role = tensor->role;
    PolyDevice tensor_device = tensor->device;
    if (role == POLY_TENSOR_PLACE && poly_uop_has_buffer_identity(realized)) {
      PolyDevice requested = poly_tensor_resolved_device(ctx, tensor);
      if (poly_uop_device(realized) == requested) {
        role = POLY_TENSOR_VALUE;
        tensor_device = requested;
      }
    }
    tensor_replace_roots_commit_reserved(
        ctx, tensor, tensor->uop_logical, realized, role, tensor_device
    );
  }
  rc = 0;

cleanup:
  for (int resolved_device = POLY_DEVICE_AUTO + 1; resolved_device <= POLY_DEVICE_DISK;
       resolved_device++)
    poly_map_destroy(scope_maps[resolved_device]);
  poly_map_destroy(exact);
  free(rewrite_kind);
  free(rewrite_aux);
  free(rewrite_slots);
  free(rewrite_results);
  free(rewrite_roots);
  free(project_results);
  free(project_roots);
  free(project_slots);
  free(project_tensors);
  free(ordinary_results);
  free(snapshot_indices);
  free(snapshot_devices);
  free(snapshot_roots);
  free(snapshot_tensors);
  free(updates);
  return rc;
}

PolyTensor *poly_tensor_find_current(
    PolyCtx *ctx,
    PolyUOp *current,
    PolyDevice device,
    PolyTensorRole role
) {
  PolyTensorList *list = tensor_list_for_uop(ctx, current);
  if (!list) return NULL;
  PolyTensor *best = NULL;
  for (int i = 0; i < list->n; i++) {
    PolyTensor *t = list->items[i];
    if (!t || tensor_current_uop(t) != current) continue;
    if (role != (PolyTensorRole)-1 && t->role != role) continue;
    if (device != POLY_DEVICE_AUTO) {
      /* PLACE is an exact requested execution device, even when two backends
       * share an allocator. VALUE lookup may reuse compatible residency, but
       * selecting another device's PLACE would change the COPY/DEVICE graph. */
      if (t->role == POLY_TENSOR_PLACE ? t->device != device
                                       : !poly_devices_share_storage(t->device, device))
        continue;
    }
    if (!best || t->order > best->order) best = t;
  }
  return best;
}

PolyTensor *poly_tensor_find_storage_identity(PolyCtx *ctx, const PolyUOp *storage) {
  if (!ctx || !storage) return NULL;
  PolyTensor *best = NULL;
  int best_score = -1;
  for (int i = 0; i < ctx->n_tensors; i++) {
    PolyTensor *t = ctx->tensors[i];
    if (!t) continue;
    const PolyUOp *identity = poly_uop_get_buffer_identity(tensor_current_uop(t));
    if (identity != storage) continue;
    int score = 0;
    if (t->provenance != POLY_TENSOR_PROVENANCE_UNKNOWN &&
        t->provenance != POLY_TENSOR_PROVENANCE_CONST_INIT)
      score = 1;
    if (t->requires_grad) score = 2;
    if (score > best_score || (score == best_score && (!best || t->order > best->order))) {
      best = t;
      best_score = score;
    }
  }
  return best;
}

static void free_tensor_list_entry(const void *key, void *value, void *userdata) {
  (void)key;
  (void)userdata;
  PolyTensorList *list = value;
  if (list) {
    free(list->items);
    free(list);
  }
}

void poly_tensor_ctx_cleanup(PolyCtx *ctx) {
  if (!ctx) return;
  if (ctx->tensors_by_uop) poly_map_foreach(ctx->tensors_by_uop, free_tensor_list_entry, NULL);
  if (ctx->tensors) {
    for (int i = 0; i < ctx->n_tensors; i++)
      free(ctx->tensors[i]);
  }
}

PolyTensor *poly_tensor_create_with_roots(
    PolyCtx *ctx,
    PolyUOp *uop_logical,
    PolyUOp *uop_physical,
    PolyTensorRole role,
    PolyDevice device
) {
  if (!ctx || !uop_logical) return NULL;

  if (ctx->n_tensors >= ctx->tensors_cap) {
    int new_cap = ctx->tensors_cap ? ctx->tensors_cap * 2 : 64;
    PolyTensor **new_tensors = realloc(ctx->tensors, (size_t)new_cap * sizeof(PolyTensor *));
    if (!new_tensors) return NULL;
    ctx->tensors = new_tensors;
    ctx->tensors_cap = new_cap;
  }

  PolyTensor *tensor = calloc(1, sizeof(PolyTensor));
  if (!tensor) return NULL;
  tensor->uop_logical = uop_logical;
  tensor->uop_physical = uop_physical;
  tensor->role = role;
  tensor->device = device;
  tensor->order = ctx->next_tensor_order++;
  tensor->provenance = POLY_TENSOR_PROVENANCE_UNKNOWN;

  if (!tensor_index_add_current(ctx, tensor)) {
    free(tensor);
    return NULL;
  }
  ctx->tensors[ctx->n_tensors++] = tensor;
  return tensor;
}

PolyTensor *poly_tensor_create(PolyCtx *ctx, PolyUOp *uop, PolyTensorRole role, PolyDevice device) {
  return poly_tensor_create_with_roots(ctx, uop, NULL, role, device);
}

int poly_tensor_custom_kernel(
    PolyCtx *ctx,
    PolyUOp *body,
    PolyTensor **inputs,
    int n_inputs,
    PolyTensor **outputs
) {
  if (!ctx || !body || !poly_ctx_owns_ptr(ctx, body) || n_inputs <= 0 || !inputs || !outputs)
    return -1;

  PolyUOp **logical_src = malloc((size_t)(n_inputs + 1) * sizeof(*logical_src));
  PolyUOp **physical_src = malloc((size_t)(n_inputs + 1) * sizeof(*physical_src));
  if (!logical_src || !physical_src) {
    free(logical_src);
    free(physical_src);
    return -1;
  }
  logical_src[0] = body;
  physical_src[0] = body;
  for (int i = 0; i < n_inputs; i++) {
    outputs[i] = NULL;
    if (!tensor_roots_owned_by_ctx(ctx, inputs[i])) {
      free(logical_src);
      free(physical_src);
      return -1;
    }
    logical_src[i + 1] = inputs[i]->uop_logical;
    physical_src[i + 1] = inputs[i]->uop_physical;
  }

  /* Pinned UOp.custom_kernel creates one opaque CALL over the exact ordered
   * contiguous Tensor.uop sources, then returns AFTER(source, same_call) for
   * every source (uop/ops.py:1093-1097). The Tensor boundary performs that construction
   * independently for retained logical and mandatory physical occurrences. */
  PolyUOp *logical_call = poly_uop(
      ctx, POLY_OP_CALL, POLY_VOID, logical_src, n_inputs + 1, poly_arg_none()
  );
  PolyUOp *physical_call = poly_uop(
      ctx, POLY_OP_CALL, POLY_VOID, physical_src, n_inputs + 1, poly_arg_none()
  );
  if (!logical_call || !physical_call) {
    free(logical_src);
    free(physical_src);
    return -1;
  }

  int rc = 0;
  for (int i = 0; i < n_inputs; i++) {
    PolyUOp *logical_after_src[2] = {logical_src[i + 1], logical_call};
    PolyUOp *physical_after_src[2] = {physical_src[i + 1], physical_call};
    PolyUOp *logical_after = poly_uop(
        ctx, POLY_OP_AFTER, logical_src[i + 1]->dtype, logical_after_src, 2,
        poly_arg_none()
    );
    PolyUOp *physical_after = poly_uop(
        ctx, POLY_OP_AFTER, physical_src[i + 1]->dtype, physical_after_src, 2,
        poly_arg_none()
    );
    if (!logical_after || !physical_after ||
        !(outputs[i] = poly_tensor_create_with_roots(
              ctx, logical_after, physical_after, POLY_TENSOR_VALUE, inputs[i]->device
          ))) {
      rc = -1;
      break;
    }
    outputs[i]->provenance = POLY_TENSOR_PROVENANCE_COMPUTED;
  }

  free(logical_src);
  free(physical_src);
  return rc;
}

PolyTensor *poly_tensor_empty(
    PolyCtx *ctx,
    PolyDType scalar_dtype,
    const int64_t *dims,
    int ndim,
    PolyDevice device
) {
  /* Pinned tinygrad UOp.empty/new_buffer (uop/ops.py:733-746) allocates one
   * UNIQUE and builds BUFFER(UNIQUE, DEVICE(device)). Polygrad retains a
   * device-free logical BUFFER with that same storage token; it must not
   * consume a second UNIQUE or influence the physical graph. */
  if (!ctx || ndim < 0 || ndim > POLY_MAX_DIMS || (ndim > 0 && !dims) ||
      !poly_device_can_execute(device))
    return NULL;

  int64_t numel = 1;
  for (int i = 0; i < ndim; i++) {
    if (dims[i] < 0 || (dims[i] != 0 && numel > INT64_MAX / dims[i])) return NULL;
    numel *= dims[i];
  }

  PolyUOp *unique =
      poly_uop0(ctx, POLY_OP_UNIQUE, POLY_VOID, poly_arg_int(poly_ctx_next_unique_id(ctx)));
  PolyUOp *device_uop = poly_uop0(ctx, POLY_OP_DEVICE, POLY_VOID, poly_arg_int((int64_t)device));
  if (!unique || !device_uop) return NULL;

  PolyUOp *logical = poly_uop1(ctx, POLY_OP_BUFFER, scalar_dtype, unique, poly_arg_int(numel));
  PolyUOp *physical_src[2] = {unique, device_uop};
  PolyUOp *physical =
      poly_uop(ctx, POLY_OP_BUFFER, scalar_dtype, physical_src, 2, poly_arg_int(numel));
  if (!logical || !physical) return NULL;

  if (ndim != 1 || dims[0] != numel) {
    logical = poly_reshape(ctx, logical, (int64_t *)dims, ndim);
    physical = poly_reshape(ctx, physical, (int64_t *)dims, ndim);
    if (!logical || !physical) return NULL;
  }
  return poly_tensor_create_with_roots(ctx, logical, physical, POLY_TENSOR_VALUE, device);
}

PolyTensor *poly_tensor_from_host(
    PolyCtx *ctx,
    void *ptr,
    size_t nbytes,
    PolyDType scalar_dtype,
    const int64_t *dims,
    int ndim
) {
  /* Pinned tinygrad UOp._frompy (uop/ops.py:747-765) creates
   * BUFFER(UNIQUE, DEVICE(PYTHON)), applies the input shape, then lets the
   * Tensor constructor COPY that exact source occurrence to its target.
   * Polygrad keeps a device-free logical twin but attaches bytes only to the
   * deviceful physical source. */
  if (!ctx || ndim < 1 || ndim > POLY_MAX_DIMS || !dims) return NULL;

  int64_t numel = 1;
  for (int i = 0; i < ndim; i++) {
    if (dims[i] < 0 || (dims[i] != 0 && numel > INT64_MAX / dims[i])) return NULL;
    numel *= dims[i];
  }
  int itemsize = poly_dtype_itemsize(scalar_dtype);
  if (itemsize <= 0 || (uint64_t)numel > SIZE_MAX / (size_t)itemsize ||
      nbytes != (size_t)numel * (size_t)itemsize)
    return NULL;

  PolyUOp *unique =
      poly_uop0(ctx, POLY_OP_UNIQUE, POLY_VOID, poly_arg_int(poly_ctx_next_unique_id(ctx)));
  PolyUOp *logical =
      unique ? poly_uop1(ctx, POLY_OP_BUFFER, scalar_dtype, unique, poly_arg_int(numel)) : NULL;
  PolyDevice source_device = POLY_DEVICE_AUTO;
  PolyUOp *physical = unique ? poly_buffer_from_host_unique(
                                   ctx, unique, scalar_dtype, numel, ptr, nbytes, &source_device
                               )
                             : NULL;
  if (!logical || !physical) return NULL;

  if (ndim != 1 || dims[0] != numel) {
    logical = poly_reshape(ctx, logical, (int64_t *)dims, ndim);
    physical = poly_reshape(ctx, physical, (int64_t *)dims, ndim);
    if (!logical || !physical) return NULL;
  }
  PolyTensor *tensor =
      poly_tensor_create_with_roots(ctx, logical, physical, POLY_TENSOR_VALUE, source_device);
  if (tensor) tensor->provenance = POLY_TENSOR_PROVENANCE_CONST_INIT;
  return tensor;
}

int poly_tensor_replace_roots(
    PolyCtx *ctx,
    PolyTensor *tensor,
    PolyUOp *uop_logical,
    PolyUOp *uop_physical,
    PolyTensorRole role,
    PolyDevice device
) {
  if (!ctx || !tensor || !uop_logical) return -1;
  PolyUOp *old_current = tensor_current_uop(tensor);
  PolyUOp *new_current = uop_physical ? uop_physical : uop_logical;

  bool reindex = old_current != new_current;
  if (reindex && !tensor_index_add(ctx, new_current, tensor)) return -1;
  tensor_replace_roots_commit_reserved(ctx, tensor, uop_logical, uop_physical, role, device);
  return 0;
}

int poly_tensor_set_physical(
    PolyCtx *ctx,
    PolyTensor *tensor,
    PolyUOp *uop_physical,
    PolyTensorRole role,
    PolyDevice device
) {
  if (!ctx || !tensor || !tensor->uop_logical || !uop_physical) return -1;
  PolyUOp *old_current = tensor_current_uop(tensor);
  if (old_current != uop_physical && !tensor_index_add(ctx, uop_physical, tensor)) return -1;
  tensor_replace_roots_commit_reserved(
      ctx, tensor, tensor->uop_logical, uop_physical, role, device
  );
  return 0;
}

/* Pinned tinygrad Tensor._apply_uop/Tensor.alu builds an operation directly
 * from the ordered current Tensor.uop operands (tensor.py:128-140). Keep the
 * active retained logical twin independent, but never derive the executable
 * result by substituting logical pointers: two ordered occurrences may share
 * one logical UOp and still have different current physical roots. */
static PolyTensor *tensor_alu(PolyCtx *ctx, PolyOps op, PolyTensor **inputs, int n) {
  if (!ctx || !inputs || n < 1 || n > 3) return NULL;
  PolyOpSet expected_group = n == 1   ? POLY_GROUP_UNARY
                             : n == 2 ? POLY_GROUP_BINARY
                                      : POLY_GROUP_TERNARY;
  if (!poly_opset_has(expected_group, op)) return NULL;
  PolyUOp *logical_src[3] = {0};
  PolyUOp *physical_src[3] = {0};
  PolyDevice device = POLY_DEVICE_AUTO;
  bool requires_grad = false;
  bool requires_grad_set = false;
  for (int i = 0; i < n; i++) {
    PolyTensor *input = inputs[i];
    if (!input || !input->uop_logical) return NULL;
    logical_src[i] = input->uop_logical;
    physical_src[i] = tensor_current_uop(input);
    if (!physical_src[i]) return NULL;
    if (input->device != POLY_DEVICE_AUTO) {
      if (device != POLY_DEVICE_AUTO && input->device != device) return NULL;
      device = input->device;
    }
    requires_grad |= input->requires_grad;
    requires_grad_set |= input->requires_grad_set;
  }

  /*
   * Pinned tinygrad mixin/__init__.py:439-449 promotes broadcast operands
   * before ALU construction. Keep that once in the C Tensor bridge so Python,
   * JS, WASM, and direct C callers construct identical logical and physical
   * CAST topology.
   */
  int promote_from = 0;
  if (op == POLY_OP_WHERE) {
    if (!poly_dtype_is_bool(logical_src[0]->dtype)) {
      logical_src[0] = poly_cast(ctx, logical_src[0], POLY_BOOL);
      physical_src[0] = poly_cast(ctx, physical_src[0], POLY_BOOL);
      if (!logical_src[0] || !physical_src[0]) return NULL;
    }
    promote_from = 1;
  }
  if (n - promote_from >= 2) {
    PolyDType common = logical_src[promote_from]->dtype;
    for (int i = promote_from + 1; i < n; i++)
      if (!poly_dtype_least_upper(common, logical_src[i]->dtype, &common)) return NULL;
    for (int i = promote_from; i < n; i++) {
      if (poly_dtype_eq(poly_dtype_scalar(logical_src[i]->dtype), common)) continue;
      logical_src[i] = poly_cast(ctx, logical_src[i], common);
      physical_src[i] = poly_cast(ctx, physical_src[i], common);
      if (!logical_src[i] || !physical_src[i]) return NULL;
    }
  }

  PolyUOp *logical = NULL;
  PolyUOp *physical = NULL;
  switch (n) {
  case 1:
    logical = poly_alu1(ctx, op, logical_src[0]);
    physical = poly_alu1(ctx, op, physical_src[0]);
    break;
  case 2:
    /* Pinned Tensor.sub is composed ADD/MUL, not a raw SUB UOp
     * (mixin/elementwise.py:90-109). Keep raw SUB available for imported IR,
     * but make the Tensor constructor use the high-level spelling. */
    logical = op == POLY_OP_SUB ? poly_sub(ctx, logical_src[0], logical_src[1])
                                : poly_alu2(ctx, op, logical_src[0], logical_src[1]);
    physical = op == POLY_OP_SUB ? poly_sub(ctx, physical_src[0], physical_src[1])
                                 : poly_alu2(ctx, op, physical_src[0], physical_src[1]);
    break;
  case 3:
    logical = poly_alu3(ctx, op, logical_src[0], logical_src[1], logical_src[2]);
    physical = poly_alu3(ctx, op, physical_src[0], physical_src[1], physical_src[2]);
    break;
  }
  if (!logical || !physical) return NULL;

  PolyTensor *out =
      poly_tensor_create_with_roots(ctx, logical, physical, POLY_TENSOR_VALUE, device);
  if (!out) return NULL;
  out->requires_grad = requires_grad;
  out->requires_grad_set = requires_grad_set;
  out->provenance = POLY_TENSOR_PROVENANCE_COMPUTED;
  return out;
}

PolyTensor *poly_tensor_alu1(PolyCtx *ctx, PolyOps op, PolyTensor *src) {
  PolyTensor *inputs[1] = {src};
  return tensor_alu(ctx, op, inputs, 1);
}

PolyTensor *poly_tensor_alu2(PolyCtx *ctx, PolyOps op, PolyTensor *a, PolyTensor *b) {
  PolyTensor *inputs[2] = {a, b};
  return tensor_alu(ctx, op, inputs, 2);
}

PolyTensor *poly_tensor_alu3(
    PolyCtx *ctx,
    PolyOps op,
    PolyTensor *a,
    PolyTensor *b,
    PolyTensor *c
) {
  PolyTensor *inputs[3] = {a, b, c};
  return tensor_alu(ctx, op, inputs, 3);
}

/* Pinned tinygrad Tensor.cast/bitcast applies UOp.cast/bitcast directly to the
 * one current Tensor.uop (tensor.py:862-904, uop/ops.py:513-521). Polygrad keeps
 * its retained logical twin, but the executable operation is built directly
 * from the exact current physical occurrence and stored even when CSE makes
 * both results pointer-identical. */
static PolyTensor *tensor_dtype_result(PolyCtx *ctx, PolyTensor *src, int dtype_id, bool bitcast) {
  PolyUOp *current = tensor_current_uop(src);
  if (!ctx || !src || !src->uop_logical || !current) return NULL;
  PolyUOp *logical = bitcast ? poly_bitcast_by_id(ctx, src->uop_logical, dtype_id)
                             : poly_cast_by_id(ctx, src->uop_logical, dtype_id);
  PolyUOp *physical = bitcast ? poly_bitcast_by_id(ctx, current, dtype_id)
                              : poly_cast_by_id(ctx, current, dtype_id);
  if (!logical || !physical) return NULL;
  PolyTensor *out =
      poly_tensor_create_with_roots(ctx, logical, physical, POLY_TENSOR_VALUE, src->device);
  if (!out) return NULL;
  out->requires_grad = src->requires_grad;
  out->requires_grad_set = src->requires_grad_set;
  out->provenance = POLY_TENSOR_PROVENANCE_COMPUTED;
  return out;
}

PolyTensor *poly_tensor_cast_by_id(PolyCtx *ctx, PolyTensor *src, int dtype_id) {
  return tensor_dtype_result(ctx, src, dtype_id, false);
}

PolyTensor *poly_tensor_bitcast_by_id(PolyCtx *ctx, PolyTensor *src, int dtype_id) {
  return tensor_dtype_result(ctx, src, dtype_id, true);
}

/* Pinned tinygrad Tensor._apply_uop (tensor.py:128-140) applies movement
 * directly to the current Tensor.uop. The Tensor boundary applies the same raw movement
 * independently to the retained logical root and exact physical occurrence;
 * neither root is derived by substituting logical identities. */
static PolyTensor *tensor_unary_result(
    PolyCtx *ctx,
    PolyTensor *src,
    PolyUOp *logical,
    PolyUOp *physical
) {
  if (!ctx || !src || !logical || !physical) return NULL;
  PolyTensor *out =
      poly_tensor_create_with_roots(ctx, logical, physical, POLY_TENSOR_VALUE, src->device);
  if (!out) return NULL;
  out->requires_grad = src->requires_grad;
  out->requires_grad_set = src->requires_grad_set;
  out->provenance = POLY_TENSOR_PROVENANCE_COMPUTED;
  return out;
}

PolyTensor *poly_tensor_contiguous(PolyCtx *ctx, PolyTensor *src) {
  PolyUOp *current = tensor_current_uop(src);
  if (!ctx || !src || !src->uop_logical || !current) return NULL;
  /* Pinned Tensor.contiguous applies UOp.contiguous to its current UOp
   * (tensor.py:742-746, uop/ops.py:587-591). The Tensor boundary applies that exact fold
   * independently to the retained logical root and physical occurrence. */
  /* Retained logical provenance records the explicit materialization request.
   * It is a Polygrad export/re-placement boundary, not the tinygrad execution
   * surface, and remains device-free. */
  PolyUOp *logical = poly_contiguous(ctx, src->uop_logical);
  PolyUOp *physical = current;
  if (poly_uop_device(physical) != POLY_DEVICE_AUTO) physical = poly_contiguous(ctx, physical);
  if (!logical || !physical) return NULL;
  return tensor_unary_result(ctx, src, logical, physical);
}

PolyTensor *poly_tensor_reshape(PolyCtx *ctx, PolyTensor *src, int64_t *dims, int ndim) {
  PolyUOp *current = tensor_current_uop(src);
  if (!ctx || !src || !src->uop_logical || !current) return NULL;
  return tensor_unary_result(
      ctx, src, poly_reshape(ctx, src->uop_logical, dims, ndim),
      poly_reshape(ctx, current, dims, ndim)
  );
}

PolyTensor *poly_tensor_expand(PolyCtx *ctx, PolyTensor *src, int64_t *dims, int ndim) {
  PolyUOp *current = tensor_current_uop(src);
  if (!ctx || !src || !src->uop_logical || !current) return NULL;
  return tensor_unary_result(
      ctx, src, poly_expand(ctx, src->uop_logical, dims, ndim),
      poly_expand(ctx, current, dims, ndim)
  );
}

PolyTensor *poly_tensor_expand_uop(
    PolyCtx *ctx,
    PolyTensor *src,
    PolyUOp **dims,
    int ndim
) {
  PolyUOp *current = tensor_current_uop(src);
  if (!ctx || !src || !src->uop_logical || !current) return NULL;
  /* Pinned tinygrad _broadcast_to/_mop constructs
   * EXPAND(value, shape_to_shape_arg(new_shape)) directly from Tensor.uop
   * (mixin/movement.py:116-143, uop/ops.py:710-722). */
  return tensor_unary_result(
      ctx, src,
      poly_expand_uop(ctx, src->uop_logical, dims, ndim),
      poly_expand_uop(ctx, current, dims, ndim)
  );
}

PolyTensor *poly_tensor_permute(PolyCtx *ctx, PolyTensor *src, int64_t *perm, int ndim) {
  PolyUOp *current = tensor_current_uop(src);
  if (!ctx || !src || !src->uop_logical || !current) return NULL;
  return tensor_unary_result(
      ctx, src, poly_permute(ctx, src->uop_logical, perm, ndim),
      poly_permute(ctx, current, perm, ndim)
  );
}

PolyTensor *poly_tensor_shrink(PolyCtx *ctx, PolyTensor *src, int64_t (*pairs)[2], int ndim) {
  PolyUOp *current = tensor_current_uop(src);
  if (!ctx || !src || !src->uop_logical || !current) return NULL;
  return tensor_unary_result(
      ctx, src, poly_shrink(ctx, src->uop_logical, pairs, ndim),
      poly_shrink(ctx, current, pairs, ndim)
  );
}

PolyTensor *poly_tensor_shrink_uop(
    PolyCtx *ctx,
    PolyTensor *src,
    PolyUOp **starts,
    PolyUOp **sizes,
    int ndim
) {
  PolyUOp *current = tensor_current_uop(src);
  if (!ctx || !src || !src->uop_logical || !current) return NULL;
  /* Pinned tinygrad movement._mop builds
   * SHRINK(value, shape_to_shape_arg(starts), shape_to_shape_arg(sizes))
   * directly from Tensor.uop (mixin/movement.py:173-193,
   * uop/ops.py:710-722). The Tensor boundary applies that same raw constructor to the
   * exact ordered logical and physical occurrences. */
  return tensor_unary_result(
      ctx, src,
      poly_shrink_uop(ctx, src->uop_logical, starts, sizes, ndim),
      poly_shrink_uop(ctx, current, starts, sizes, ndim)
  );
}

PolyTensor *poly_tensor_flip(PolyCtx *ctx, PolyTensor *src, int64_t *axes, int n_axes) {
  PolyUOp *current = tensor_current_uop(src);
  if (!ctx || !src || !src->uop_logical || !current) return NULL;
  return tensor_unary_result(
      ctx, src, poly_flip(ctx, src->uop_logical, axes, n_axes),
      poly_flip(ctx, current, axes, n_axes)
  );
}

PolyTensor *poly_tensor_pad_value(
    PolyCtx *ctx,
    PolyTensor *src,
    int64_t (*pairs)[2],
    int ndim,
    double value
) {
  PolyUOp *current = tensor_current_uop(src);
  if (!ctx || !src || !src->uop_logical || !current) return NULL;
  return tensor_unary_result(
      ctx, src, poly_pad_value(ctx, src->uop_logical, pairs, ndim, value),
      poly_pad_value(ctx, current, pairs, ndim, value)
  );
}

PolyTensor *poly_tensor_to_device(PolyCtx *ctx, PolyTensor *tensor, PolyDevice device) {
  if (!ctx || !tensor || !tensor->uop_logical) return NULL;
  if (tensor->device == device) return tensor;
  /* Pinned Tensor.to returns self when Tensor.uop has no device
   * (tensor.py:327-335). Pure CONST/arange graphs are device-free values, not
   * storage that requires a COPY or late placement. */
  if (tensor->uop_physical && poly_uop_device(tensor->uop_physical) == POLY_DEVICE_AUTO)
    return tensor;
  /* Pinned tinygrad Tensor.to (tensor.py:327-335) stores
   * self.uop.copy_to_device(device) immediately. This boundary therefore consumes
   * the exact stored physical occurrence directly. Keep physicalization only
   * as migration fallback for raw/unmigrated Tensors without that root. */
  PolyUOp *source_physical =
      tensor->uop_physical ? tensor->uop_physical : poly_tensor_physicalize(ctx, tensor);
  PolyUOp *target_device = poly_uop0(ctx, POLY_OP_DEVICE, POLY_VOID, poly_arg_int((int64_t)device));
  PolyUOp *copy_src[2] = {source_physical, target_device};
  PolyUOp *physical =
      (source_physical && target_device)
          ? poly_uop(ctx, POLY_OP_COPY, source_physical->dtype, copy_src, 2, poly_arg_none())
          : NULL;
  if (!physical) return NULL;
  PolyTensor *placed =
      poly_tensor_create_with_roots(ctx, tensor->uop_logical, physical, POLY_TENSOR_PLACE, device);
  if (placed) {
    placed->source = tensor;
    placed->requires_grad = tensor->requires_grad;
    placed->requires_grad_set = tensor->requires_grad_set;
    placed->provenance = tensor->provenance;
  }
  return placed;
}

PolyTensor *poly_tensor_assign(PolyCtx *ctx, PolyTensor *target, PolyTensor *value) {
  if (!ctx || !target || !value) return NULL;
  PolyUOp *target_logical = target->uop_logical;
  PolyUOp *value_logical = value->uop_logical;
  PolyUOp *target_current = poly_tensor_uop(target);
  PolyUOp *value_current = poly_tensor_uop(value);
  if (!target_logical || !value_logical || !target_current || !value_current) return NULL;

  /* Match tinygrad Tensor.assign: non-DISK assigns require same device and
   * dtype before constructing the AFTER/STORE effect graph. Without this,
   * a CPU target could silently physicalize a CUDA value back to CPU, which
   * changes user-visible placement semantics. */
  if (target->device != POLY_DEVICE_DISK && target->device != POLY_DEVICE_AUTO &&
      value->device != POLY_DEVICE_AUTO &&
      !poly_devices_share_storage(target->device, value->device))
    return NULL;
  if (!poly_dtype_eq(
          poly_dtype_scalar(target_logical->dtype), poly_dtype_scalar(value_logical->dtype)
      ) ||
      !poly_dtype_eq(
          poly_dtype_scalar(target_current->dtype), poly_dtype_scalar(value_current->dtype)
      ))
    return NULL;

  PolyUOp *current_store = poly_store_val(ctx, target_current, value_current);
  PolyUOp *current_src[2] = {target_current, current_store};
  PolyUOp *current_after =
      current_store
          ? poly_uop(ctx, POLY_OP_AFTER, target_current->dtype, current_src, 2, poly_arg_none())
          : NULL;
  if (!current_after) return NULL;

  PolyUOp *view_anchor = tensor_assign_view_anchor(target_current);
  if (view_anchor) {
    PolyUOp *anchor_src[2] = {view_anchor, current_after};
    PolyUOp *assigned_anchor =
        poly_uop(ctx, POLY_OP_AFTER, view_anchor->dtype, anchor_src, 2, poly_arg_none());
    if (!assigned_anchor) return NULL;
    if (tensor_retarget_logical_uops(ctx, view_anchor, assigned_anchor, target->device) != 0)
      return NULL;
    return target;
  }

  PolyUOp *logical_store = poly_store_val(ctx, target_logical, value_logical);
  PolyUOp *logical_src[2] = {target_logical, logical_store};
  PolyUOp *logical_after =
      logical_store
          ? poly_uop(ctx, POLY_OP_AFTER, target_logical->dtype, logical_src, 2, poly_arg_none())
          : NULL;
  if (!logical_after) return NULL;

  /* Polygrad keeps portable provenance and placed execution as separate roots.
   * tinygrad has one UOp, so its second assign naturally chains from the first
   * AFTER. Preserve that same value-version chain independently in both roots
   * instead of leaking target_current into uop_logical. */
  if (poly_tensor_replace_roots(
          ctx, target, logical_after, current_after, target->role, target->device
      ) != 0)
    return NULL;
  return target;
}

PolyTensor *poly_tensor_clone_into(PolyCtx *ctx, PolyTensor *target, PolyTensor *source) {
  if (!ctx || !target || !source || target == source) return NULL;
  if (target->device == POLY_DEVICE_AUTO) return NULL;

  PolyUOp *target_logical = target->uop_logical;
  PolyUOp *target_physical =
      target->uop_physical ? target->uop_physical : poly_tensor_physicalize(ctx, target);
  PolyUOp *source_logical = source->uop_logical;
  if (source_logical && source_logical->op == POLY_OP_AFTER && source->uop_physical)
    source_logical = source->uop_physical;
  /* Pinned UOp.clone (uop/ops.py:765-769) decides from self.device and
   * stores a device-free source directly. The stored current root is
   * that self UOp; wrapper preferred-device metadata must not manufacture a
   * COPY. Preserve the legacy fallback only for a missing physical root. */
  PolyUOp *source_physical =
      source->uop_physical ? source->uop_physical : poly_tensor_physicalize(ctx, source);
  if (!target_logical || !target_physical || !source_logical || !source_physical) return NULL;
  if (!poly_dtype_eq(
          poly_dtype_scalar(target_logical->dtype), poly_dtype_scalar(source_logical->dtype)
      ))
    return NULL;

  PolyUOp *logical_store = poly_store_val(ctx, target_logical, source_logical);
  PolyUOp *logical_src[2] = {target_logical, logical_store};
  PolyUOp *logical_after =
      logical_store
          ? poly_uop(ctx, POLY_OP_AFTER, target_logical->dtype, logical_src, 2, poly_arg_none())
          : NULL;

  PolyUOp *placed_source = source_physical;
  PolyDevice source_device = poly_uop_device(source_physical);
  if (source_device != POLY_DEVICE_AUTO && source_device != target->device) {
    PolyUOp *device =
        poly_uop0(ctx, POLY_OP_DEVICE, POLY_VOID, poly_arg_int((int64_t)target->device));
    PolyUOp *copy_src[2] = {source_physical, device};
    placed_source =
        device ? poly_uop(ctx, POLY_OP_COPY, source_physical->dtype, copy_src, 2, poly_arg_none())
               : NULL;
  }

  PolyUOp *physical_store =
      placed_source ? poly_store_val(ctx, target_physical, placed_source) : NULL;
  PolyUOp *physical_src[2] = {target_physical, physical_store};
  PolyUOp *physical_after =
      physical_store
          ? poly_uop(ctx, POLY_OP_AFTER, target_physical->dtype, physical_src, 2, poly_arg_none())
          : NULL;
  if (!logical_after || !physical_after) return NULL;

  if (poly_tensor_replace_roots(
          ctx, target, logical_after, physical_after, POLY_TENSOR_VALUE, target->device
      ) != 0)
    return NULL;
  return target;
}

PolyUOp *poly_tensor_uop(PolyTensor *tensor) {
  return tensor_current_uop(tensor);
}

PolyUOp *poly_tensor_uop_logical(PolyTensor *tensor) {
  return tensor ? tensor->uop_logical : NULL;
}

PolyUOp *poly_tensor_uop_physical(PolyTensor *tensor) {
  return tensor ? tensor->uop_physical : NULL;
}

PolyDevice poly_tensor_device(PolyTensor *tensor) {
  return tensor ? tensor->device : POLY_DEVICE_AUTO;
}

bool poly_tensor_requires_grad(PolyTensor *tensor) {
  return tensor ? tensor->requires_grad : false;
}

bool poly_tensor_requires_grad_is_set(PolyTensor *tensor) {
  return tensor ? tensor->requires_grad_set : false;
}

void poly_tensor_set_requires_grad(PolyTensor *tensor, bool requires_grad) {
  if (tensor) {
    tensor->requires_grad = requires_grad;
    tensor->requires_grad_set = true;
  }
}

PolyTensorProvenance poly_tensor_provenance(PolyTensor *tensor) {
  return tensor ? tensor->provenance : POLY_TENSOR_PROVENANCE_UNKNOWN;
}

void poly_tensor_set_provenance(PolyTensor *tensor, PolyTensorProvenance provenance) {
  if (!tensor) return;
  if (provenance < POLY_TENSOR_PROVENANCE_UNKNOWN || provenance > POLY_TENSOR_PROVENANCE_COMPUTED)
    provenance = POLY_TENSOR_PROVENANCE_UNKNOWN;
  tensor->provenance = provenance;
}

/* Internal helpers */

/* Helper: float constant matching the dtype of a given UOp.
 * For float inputs: creates a constant with the same float dtype.
 * For non-float inputs (comparisons producing bool): defaults to float32. */
static inline PolyUOp *cf(PolyCtx *ctx, PolyUOp *ref, double v) {
  PolyDType dt = poly_dtype_scalar(ref->dtype);
  if (poly_dtype_is_float(dt)) return poly_const_typed(ctx, dt, v);
  return poly_const_float(ctx, v);
}

/* Helper: const with explicit dtype -- use in special-math ops for dtype correctness */
static inline PolyUOp *cdt(PolyCtx *ctx, PolyDType dt, double v) {
  return poly_const_typed(ctx, dt, v);
}

static const PolyDType *ffi_dtype_from_id(int dtype_id) {
  if (dtype_id < 0 || dtype_id >= N_DTYPE_FFI) return NULL;
  return _dtype_table_ffi[dtype_id];
}

static bool ffi_dtype_is_integer_like(int dtype_id, PolyDType *out_dt) {
  const PolyDType *dt = ffi_dtype_from_id(dtype_id);
  if (!dt) return false;
  PolyDType sdt = poly_dtype_scalar(*dt);
  if (!poly_dtype_is_int(sdt) && !poly_dtype_is_bool(sdt)) return false;
  if (out_dt) *out_dt = sdt;
  return true;
}

static bool ffi_dtype_is_float_like(int dtype_id, PolyDType *out_dt) {
  const PolyDType *dt = ffi_dtype_from_id(dtype_id);
  if (!dt) return false;
  PolyDType sdt = poly_dtype_scalar(*dt);
  if (!poly_dtype_is_float(sdt)) return false;
  if (out_dt) *out_dt = sdt;
  return true;
}

static PolyUOp *poly_const_exact_int(PolyCtx *ctx, PolyDType dt, int64_t value) {
  dt = poly_dtype_scalar(dt);
  if (poly_dtype_is_bool(dt)) return poly_uop0(ctx, POLY_OP_CONST, dt, poly_arg_bool(value != 0));
  if (!poly_dtype_is_int(dt)) return NULL;
  return poly_uop0(ctx, POLY_OP_CONST, dt, poly_arg_int(value));
}

static PolyUOp *poly_const_exact_float(PolyCtx *ctx, PolyDType dt, double value) {
  dt = poly_dtype_scalar(dt);
  if (!poly_dtype_is_float(dt)) return NULL;
  return poly_uop0(ctx, POLY_OP_CONST, dt, poly_arg_float(value));
}

static bool poly_dtype_bound_const(PolyCtx *ctx, PolyDType dt, bool use_min, PolyUOp **out) {
  if (!ctx || !out) return false;
  dt = poly_dtype_scalar(dt);
  if (poly_dtype_is_float(dt)) {
    *out = poly_const_exact_float(ctx, dt, use_min ? -INFINITY : INFINITY);
    return *out != NULL;
  }
  if (poly_dtype_is_bool(dt)) {
    *out = poly_const_exact_int(ctx, dt, use_min ? 0 : 1);
    return *out != NULL;
  }
  if (!poly_dtype_is_int(dt)) return false;
  if (poly_dtype_is_unsigned(dt)) {
    if (use_min) {
      *out = poly_const_exact_int(ctx, dt, 0);
      return *out != NULL;
    }
    /* Pinned DType.max is the positive Python integer 2**bits-1
     * (dtype.py:84-100). Preserve that exact UOp arg; fixed-width renderers
     * apply the dtype bits only at their backend boundary. */
    PolyInt neg_one = {0}, maximum = {0};
    bool ok = poly_int_from_i64(&neg_one, -1) &&
              poly_int_truncate(&maximum, &neg_one, dt.bitsize, true);
    if (ok)
      *out = poly_uop0(ctx, POLY_OP_CONST, dt, poly_int_as_arg(&maximum));
    poly_int_free(&maximum);
    poly_int_free(&neg_one);
    return ok && *out != NULL;
  }

  int64_t v = 0;
  if (dt.bitsize >= 64)
    v = use_min ? INT64_MIN : INT64_MAX;
  else {
    int bits = (int)dt.bitsize;
    v = use_min ? -(1LL << (bits - 1)) : ((1LL << (bits - 1)) - 1);
  }
  *out = poly_const_exact_int(ctx, dt, v);
  return *out != NULL;
}

static PolyUOp *poly_empty_shaped(PolyCtx *ctx, PolyDType dt, const int64_t *shape, int ndim) {
  PolyUOp *buf = poly_buffer(ctx, dt, 0);
  if (!buf || ndim <= 1) return buf;
  return poly_reshape(ctx, buf, (int64_t *)shape, ndim);
}

static PolyUOp *poly_full_from_scalar(
    PolyCtx *ctx,
    const int64_t *shape,
    int ndim,
    PolyUOp *scalar
) {
  if (!scalar || ndim < 0 || ndim > POLY_MAX_DIMS) return NULL;
  if (ndim == 0) return scalar;
  if (!shape) return NULL;

  bool has_zero = false;
  for (int i = 0; i < ndim; i++) {
    if (shape[i] < 0) return NULL;
    if (shape[i] == 0) has_zero = true;
  }
  /* Pinned full(buffer=False) always builds CONST -> RESHAPE -> EXPAND,
   * including zero-sized output shapes (mixin/__init__.py:55-77). A zero
   * extent is valid here and must not introduce anonymous BUFFER identity. */
  if (!has_zero && poly_shape_numel_checked(shape, ndim) < 0) return NULL;

  int64_t ones[POLY_MAX_DIMS];
  for (int i = 0; i < ndim; i++)
    ones[i] = 1;
  PolyUOp *r = poly_reshape(ctx, scalar, ones, ndim);
  bool already_expanded = true;
  for (int i = 0; i < ndim; i++)
    if (shape[i] != 1) {
      already_expanded = false;
      break;
    }
  if (already_expanded) return r;
  return poly_expand(ctx, r, (int64_t *)shape, ndim);
}

static int64_t poly_arange_len(long double start, long double stop, long double step) {
  if (step == 0.0L) return -1;
  if ((step > 0.0L && start < stop) || (step < 0.0L && start > stop)) {
    long double span = (stop - start) / step;
    long double n = ceill(span - 1e-12L);
    if (n < 0.0L) return 0;
    if (n > (long double)INT64_MAX) return -1;
    return (int64_t)n;
  }
  return 0;
}

int64_t poly_shape_numel_checked(const int64_t *shape, int ndim) {
  if (ndim < 0 || ndim > POLY_MAX_DIMS) return -1;
  if (ndim == 0) return 1;
  if (!shape) return -1;
  int64_t n = 1;
  for (int i = 0; i < ndim; i++) {
    if (shape[i] <= 0) return -1;
    if (n > INT64_MAX / shape[i]) return -1;
    n *= shape[i];
  }
  return n;
}

bool poly_shape_equal(const int64_t *a, int a_ndim, const int64_t *b, int b_ndim) {
  if (!a || !b || a_ndim != b_ndim) return false;
  for (int i = 0; i < a_ndim; i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

static bool shape_equal_except_axis(
    const int64_t *full,
    int full_ndim,
    const int64_t *reduced,
    int reduced_ndim,
    int axis
) {
  if (!full || !reduced || full_ndim != reduced_ndim + 1) return false;
  if (axis < 0) axis += full_ndim;
  if (axis < 0 || axis >= full_ndim) return false;
  for (int i = 0, j = 0; i < full_ndim; i++) {
    if (i == axis) continue;
    if (full[i] != reduced[j++]) return false;
  }
  return true;
}

/* Internal: compute output shape for a single-axis reduction */

static void reduce_output_shape(
    const int64_t *shape,
    int ndim,
    int axis,
    int keepdim,
    int64_t *out_shape,
    int *out_ndim
) {
  if (axis < 0) axis += ndim;
  int on = 0;
  for (int i = 0; i < ndim; i++) {
    if (i == axis) {
      if (keepdim) out_shape[on++] = 1;
    } else {
      out_shape[on++] = shape[i];
    }
  }
  if (on == 0) {
    out_shape[0] = 1;
    on = 1;
  }
  *out_ndim = on;
}

/* Internal: do a single-axis reduce and optionally reshape away the axis */
static PolyUOp *do_reduce(
    PolyCtx *ctx,
    PolyOps reduce_op,
    PolyUOp *x,
    const int64_t *shape,
    int ndim,
    int axis,
    int keepdim,
    int64_t *out_shape,
    int *out_ndim
) {
  if (axis < 0) axis += ndim;
  int64_t axes[] = {axis};
  PolyUOp *r = poly_reduce_axis(ctx, reduce_op, x, axes, 1);
  reduce_output_shape(shape, ndim, axis, keepdim, out_shape, out_ndim);
  if (!keepdim) {
    r = poly_reshape(ctx, r, out_shape, *out_ndim);
  }
  return r;
}

static PolyUOp *reshape_logical_input(PolyCtx *ctx, PolyUOp *x, const int64_t *shape, int ndim) {
  if (!ctx || !x || (ndim > 0 && !shape)) return NULL;
  if (ndim == 0) return poly_reshape(ctx, x, NULL, 0);
  return poly_reshape(ctx, x, (int64_t *)shape, ndim);
}

/* Internal: read shape from UOp into local arrays */
static int uop_shape(PolyCtx *ctx, PolyUOp *u, int64_t *out_shape) {
  int ndim = poly_uop_ndim(ctx, u);
  if (ndim > 0) {
    const int64_t *dims = poly_uop_max_shape_dims(ctx, u);
    if (dims) memcpy(out_shape, dims, ndim * sizeof(int64_t));
  }
  return ndim;
}

/* erf tau helper */

/* A&S 7.1.26: tau(|x|) = t * P(t) * exp(-x^2) where t = 1/(1+p*|x|).
 * erf(x) = sign(x) * (1 - tau(|x|)).
 * erfc(x) = tau(x) for x >= 0, 2 - tau(|x|) for x < 0.
 * Computing tau directly avoids the 1-erf(x) cancellation in erfc. */
static PolyUOp *erf_tau(PolyCtx *ctx, PolyUOp *ax, PolyDType dt) {
  PolyUOp *t = poly_alu1(
      ctx, POLY_OP_RECIPROCAL,
      poly_alu2(
          ctx, POLY_OP_ADD, cdt(ctx, dt, 1.0),
          poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, 0.3275911), ax)
      )
  );
  PolyUOp *p = cdt(ctx, dt, 1.061405429);
  p = poly_alu2(ctx, POLY_OP_ADD, cdt(ctx, dt, -1.453152027), poly_alu2(ctx, POLY_OP_MUL, t, p));
  p = poly_alu2(ctx, POLY_OP_ADD, cdt(ctx, dt, 1.421413741), poly_alu2(ctx, POLY_OP_MUL, t, p));
  p = poly_alu2(ctx, POLY_OP_ADD, cdt(ctx, dt, -0.284496736), poly_alu2(ctx, POLY_OP_MUL, t, p));
  p = poly_alu2(ctx, POLY_OP_ADD, cdt(ctx, dt, 0.254829592), poly_alu2(ctx, POLY_OP_MUL, t, p));
  PolyUOp *x2 = poly_alu2(ctx, POLY_OP_MUL, ax, ax);
  PolyUOp *e = poly_exp(ctx, poly_alu1(ctx, POLY_OP_NEG, x2));
  return poly_alu2(ctx, POLY_OP_MUL, t, poly_alu2(ctx, POLY_OP_MUL, p, e));
}

/* lgamma Lanczos helper */

static PolyUOp *poly_lgamma_forward_lanczos(PolyCtx *ctx, PolyUOp *x, PolyDType dt) {
  /* Lanczos approximation with reflection. */
  const double g = 7.0;
  const double c0 = 0.99999999999980993;
  const double c[8] = {676.5203681218851,     -1259.1392167224028,  771.32342877765313,
                       -176.61502916214059,   12.507343278686905,   -0.13857109526572012,
                       9.9843695780195716e-6, 1.5056327351493116e-7};

  PolyUOp *xm1 = poly_alu2(ctx, POLY_OP_SUB, x, cdt(ctx, dt, 1.0));
  PolyUOp *a = cdt(ctx, dt, c0);
  for (int i = 0; i < 8; i++) {
    PolyUOp *den = poly_alu2(ctx, POLY_OP_ADD, xm1, cdt(ctx, dt, (double)(i + 1)));
    a = poly_alu2(ctx, POLY_OP_ADD, a, poly_alu2(ctx, POLY_OP_FDIV, cdt(ctx, dt, c[i]), den));
  }
  PolyUOp *t = poly_alu2(ctx, POLY_OP_ADD, xm1, cdt(ctx, dt, g + 0.5));
  PolyUOp *lg_pos = poly_alu2(
      ctx, POLY_OP_ADD, cdt(ctx, dt, 0.91893853320467274178),
      poly_alu2(
          ctx, POLY_OP_ADD,
          poly_alu2(
              ctx, POLY_OP_MUL, poly_alu2(ctx, POLY_OP_ADD, xm1, cdt(ctx, dt, 0.5)),
              poly_log(ctx, t)
          ),
          poly_alu2(ctx, POLY_OP_SUB, poly_log(ctx, a), t)
      )
  );

  PolyUOp *one_minus_x = poly_alu2(ctx, POLY_OP_SUB, cdt(ctx, dt, 1.0), x);
  PolyUOp *xm1r = poly_alu2(ctx, POLY_OP_SUB, one_minus_x, cdt(ctx, dt, 1.0));
  PolyUOp *ar = cdt(ctx, dt, c0);
  for (int i = 0; i < 8; i++) {
    PolyUOp *den = poly_alu2(ctx, POLY_OP_ADD, xm1r, cdt(ctx, dt, (double)(i + 1)));
    ar = poly_alu2(ctx, POLY_OP_ADD, ar, poly_alu2(ctx, POLY_OP_FDIV, cdt(ctx, dt, c[i]), den));
  }
  PolyUOp *tr = poly_alu2(ctx, POLY_OP_ADD, xm1r, cdt(ctx, dt, g + 0.5));
  PolyUOp *lg_ref_base = poly_alu2(
      ctx, POLY_OP_ADD, cdt(ctx, dt, 0.91893853320467274178),
      poly_alu2(
          ctx, POLY_OP_ADD,
          poly_alu2(
              ctx, POLY_OP_MUL, poly_alu2(ctx, POLY_OP_ADD, xm1r, cdt(ctx, dt, 0.5)),
              poly_log(ctx, tr)
          ),
          poly_alu2(ctx, POLY_OP_SUB, poly_log(ctx, ar), tr)
      )
  );
  PolyUOp *sinpix = poly_sin(ctx, poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, M_PI), x));
  PolyUOp *lg_ref = poly_alu2(
      ctx, POLY_OP_SUB,
      poly_alu2(ctx, POLY_OP_SUB, cdt(ctx, dt, log(M_PI)), poly_log(ctx, poly_abs(ctx, sinpix))),
      lg_ref_base
  );
  PolyUOp *cond = poly_alu2(ctx, POLY_OP_CMPLT, x, cdt(ctx, dt, 0.5));
  return poly_alu3(ctx, POLY_OP_WHERE, cond, lg_ref, lg_pos);
}

/* ══════════════════════════════════════════════════════════════════════ */
/*  Public functions                                                     */
/* ══════════════════════════════════════════════════════════════════════ */

/* Broadcasting (tinygrad _broadcasted / _broadcast_to) */

PolyUOp *poly_broadcast_to(PolyCtx *ctx, PolyUOp *x, const int64_t *shape, int ndim) {
  if (!ctx || !x || !shape || ndim < 0 || ndim > POLY_MAX_DIMS) return NULL;
  int64_t x_shape[POLY_MAX_DIMS];
  int x_ndim = uop_shape(ctx, x, x_shape);
  /* Pinned _broadcast_to treats scalar values as rank zero, then reshapes and
   * expands them to the common shape (mixin/movement.py:116-128). */
  if (x_ndim < 0) {
    if (poly_dtype_eq(poly_dtype_scalar(x->dtype), POLY_VOID)) return NULL;
    x_ndim = 0;
  }
  if (ndim < x_ndim) return NULL; /* can't broadcast to fewer dims */

  /* Already matching */
  if (x_ndim == ndim) {
    bool same = true;
    for (int i = 0; i < ndim; i++)
      if (x_shape[i] != shape[i]) {
        same = false;
        break;
      }
    if (same) return x;
  }

  /* Left-pad with 1s to match ndim (tinygrad _align_left) */
  int64_t aligned[POLY_MAX_DIMS];
  int pad = ndim - x_ndim;
  for (int i = 0; i < pad; i++)
    aligned[i] = 1;
  for (int i = 0; i < x_ndim; i++)
    aligned[pad + i] = x_shape[i];

  /* Validate: each aligned dim must be 1 or equal to target */
  for (int i = 0; i < ndim; i++) {
    if (aligned[i] != shape[i] && aligned[i] != 1) {
      fprintf(
          stderr, "poly_broadcast_to: incompatible dim %d: %lld vs %lld\n", i,
          (long long)aligned[i], (long long)shape[i]
      );
      return NULL;
    }
  }

  bool need_reshape = x_ndim != ndim;
  for (int i = 0; !need_reshape && i < x_ndim; i++)
    if (x_shape[i] != aligned[i]) need_reshape = true;
  PolyUOp *r = need_reshape ? poly_reshape(ctx, x, aligned, ndim) : x;

  /* Expand where aligned[i]==1 and shape[i]>1 */
  bool need_expand = false;
  for (int i = 0; i < ndim; i++)
    if (aligned[i] != shape[i]) {
      need_expand = true;
      break;
    }
  if (need_expand) r = poly_expand(ctx, r, (int64_t *)shape, ndim);
  return r;
}

bool poly_broadcast_pair(
    PolyCtx *ctx,
    PolyUOp **a,
    PolyUOp **b,
    int64_t *out_shape,
    int *out_ndim
) {
  if (!ctx || !a || !b || !*a || !*b || !out_shape || !out_ndim) return false;
  int64_t sa[POLY_MAX_DIMS], sb[POLY_MAX_DIMS];
  int na = uop_shape(ctx, *a, sa);
  int nb = uop_shape(ctx, *b, sb);

  /* Scalars or shapeless -- no broadcast needed */
  if (na <= 0 && nb <= 0) {
    *out_ndim = 0;
    return true;
  }
  if (na <= 0) {
    na = 0;
  }
  if (nb <= 0) {
    nb = 0;
  }

  /* Compute broadcast shape (tinygrad _broadcast_shape) */
  int nd = na > nb ? na : nb;
  if (nd > POLY_MAX_DIMS) {
    *out_ndim = 0;
    return false;
  }
  int pa = nd - na, pb = nd - nb;
  for (int i = 0; i < nd; i++) {
    int64_t da = (i >= pa) ? sa[i - pa] : 1;
    int64_t db = (i >= pb) ? sb[i - pb] : 1;
    if (da != db && da != 1 && db != 1) {
      fprintf(
          stderr, "poly_broadcast_pair: incompatible shapes at dim %d: %lld vs %lld\n", i,
          (long long)da, (long long)db
      );
      *out_ndim = 0;
      return false;
    }
    /* Pinned _broadcast_shape returns zero when any aligned dimension is zero
     * (uop/ops.py:63-71). max(0, 1) is not the broadcast result. */
    out_shape[i] = (da == 0 || db == 0) ? 0 : (da > db ? da : db);
  }
  *out_ndim = nd;

  *a = poly_broadcast_to(ctx, *a, out_shape, nd);
  *b = poly_broadcast_to(ctx, *b, out_shape, nd);
  return *a != NULL && *b != NULL;
}

/* Pinned true division broadcasts/promotes the pair, applies reciprocal to
 * the divisor, then multiplies (mixin/elementwise.py:219-234, :440-448).
 * Build that exact graph independently from ordered retained/current roots;
 * tensor-stage FDIV remains available only as raw/imported IR vocabulary. */
PolyTensor *poly_tensor_div(PolyCtx *ctx, PolyTensor *dividend, PolyTensor *divisor) {
  if (!ctx || !dividend || !divisor || !dividend->uop_logical || !divisor->uop_logical) return NULL;
  PolyUOp *logical_a = dividend->uop_logical;
  PolyUOp *logical_b = divisor->uop_logical;
  PolyUOp *physical_a = tensor_current_uop(dividend);
  PolyUOp *physical_b = tensor_current_uop(divisor);
  if (!physical_a || !physical_b) return NULL;

  int64_t logical_shape[POLY_MAX_DIMS], physical_shape[POLY_MAX_DIMS];
  int logical_ndim = 0, physical_ndim = 0;
  if (!poly_broadcast_pair(ctx, &logical_a, &logical_b, logical_shape, &logical_ndim) ||
      !poly_broadcast_pair(ctx, &physical_a, &physical_b, physical_shape, &physical_ndim))
    return NULL;

  PolyDType common;
  if (!poly_dtype_least_upper(logical_a->dtype, logical_b->dtype, &common)) return NULL;
  if (!poly_dtype_eq(poly_dtype_scalar(logical_a->dtype), common)) {
    logical_a = poly_cast(ctx, logical_a, common);
    physical_a = poly_cast(ctx, physical_a, common);
  }
  if (!poly_dtype_eq(poly_dtype_scalar(logical_b->dtype), common)) {
    logical_b = poly_cast(ctx, logical_b, common);
    physical_b = poly_cast(ctx, physical_b, common);
  }
  if (!logical_a || !logical_b || !physical_a || !physical_b) return NULL;

  if (!poly_dtype_is_float(common)) {
    logical_a = poly_cast(ctx, logical_a, POLY_FLOAT32);
    logical_b = poly_cast(ctx, logical_b, POLY_FLOAT32);
    physical_a = poly_cast(ctx, physical_a, POLY_FLOAT32);
    physical_b = poly_cast(ctx, physical_b, POLY_FLOAT32);
  }
  if (!logical_a || !logical_b || !physical_a || !physical_b) return NULL;

  PolyUOp *logical =
      poly_alu2(ctx, POLY_OP_MUL, logical_a, poly_alu1(ctx, POLY_OP_RECIPROCAL, logical_b));
  PolyUOp *physical =
      poly_alu2(ctx, POLY_OP_MUL, physical_a, poly_alu1(ctx, POLY_OP_RECIPROCAL, physical_b));
  if (!logical || !physical) return NULL;

  PolyDevice device = dividend->device;
  if (device == POLY_DEVICE_AUTO) device = divisor->device;
  if (divisor->device != POLY_DEVICE_AUTO && device != divisor->device) return NULL;
  PolyTensor *out =
      poly_tensor_create_with_roots(ctx, logical, physical, POLY_TENSOR_VALUE, device);
  if (!out) return NULL;
  out->requires_grad = dividend->requires_grad || divisor->requires_grad;
  out->requires_grad_set = dividend->requires_grad_set || divisor->requires_grad_set;
  out->provenance = POLY_TENSOR_PROVENANCE_COMPUTED;
  return out;
}

/* Pinned tinygrad _broadcast_to (mixin/movement.py:116-128) reshapes a scalar
 * to rank-many ones, then expands it using the exact destination shape UOps.
 * Keep symbolic dimensions as UOps instead of freezing their max-shape values. */
static PolyUOp *broadcast_scalar_like(PolyCtx *ctx, PolyUOp *scalar, PolyUOp *like) {
  if (!ctx || !scalar || !like) return NULL;
  int ndim = poly_uop_ndim(ctx, like);
  if (ndim < 0 || ndim > POLY_MAX_DIMS) return NULL;
  if (ndim == 0) return scalar;

  int64_t ones[POLY_MAX_DIMS];
  for (int i = 0; i < ndim; i++)
    ones[i] = 1;
  PolyUOp *shaped = poly_reshape(ctx, scalar, ones, ndim);
  if (!shaped) return NULL;

  PolyUOp *dims[POLY_MAX_DIMS];
  bool all_one = true;
  for (int i = 0; i < ndim; i++) {
    int64_t value = 0;
    PolyUOp *dim = poly_uop_shape_dim(ctx, like, i);
    if (!dim) return NULL;
    if (poly_uop_const_i64(dim, &value) == 0) {
      /* Pinned shape_to_shape_arg turns Python integer dimensions back into
       * weakint CONST lanes (uop/ops.py:85-88). This also lets CSE reuse an
       * existing movement shape STACK instead of introducing long lanes. */
      dims[i] = poly_uop0(ctx, POLY_OP_CONST, POLY_INDEX, poly_arg_int(value));
      if (value != 1) all_one = false;
    } else {
      dims[i] = dim;
      all_one = false;
    }
  }
  return all_one ? shaped : poly_expand_uop(ctx, shaped, dims, ndim);
}

/* Broadcasting binary ops */

PolyUOp *poly_add(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  int64_t shape[POLY_MAX_DIMS];
  int ndim;
  if (!poly_broadcast_pair(ctx, &a, &b, shape, &ndim)) return NULL;
  return poly_alu2(ctx, POLY_OP_ADD, a, b);
}

PolyUOp *poly_sub(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  int64_t shape[POLY_MAX_DIMS];
  int ndim;
  if (!poly_broadcast_pair(ctx, &a, &b, shape, &ndim)) return NULL;

  /* Pinned ElementwiseMixin.sub is `a + (-b)`, and negation is
   * multiplication by a broadcast `-1` except for bool logical-not
   * (mixin/elementwise.py:57-66,90-109). */
  PolyDType common;
  if (!poly_dtype_least_upper(a->dtype, b->dtype, &common)) return NULL;
  if (!poly_dtype_eq(poly_dtype_scalar(a->dtype), common)) a = poly_cast(ctx, a, common);
  if (!poly_dtype_eq(poly_dtype_scalar(b->dtype), common)) b = poly_cast(ctx, b, common);
  if (!a || !b) return NULL;

  PolyUOp *neg_b = NULL;
  if (poly_dtype_is_bool(common)) {
    neg_b = poly_logical_not(ctx, b);
  } else {
    PolyUOp *minus_one = broadcast_scalar_like(ctx, poly_const_typed(ctx, common, -1.0), b);
    neg_b = minus_one ? poly_alu2(ctx, POLY_OP_MUL, b, minus_one) : NULL;
  }
  return neg_b ? poly_alu2(ctx, POLY_OP_ADD, a, neg_b) : NULL;
}

PolyUOp *poly_mul(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  int64_t shape[POLY_MAX_DIMS];
  int ndim;
  if (!poly_broadcast_pair(ctx, &a, &b, shape, &ndim)) return NULL;
  return poly_alu2(ctx, POLY_OP_MUL, a, b);
}

PolyUOp *poly_div(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  int64_t shape[POLY_MAX_DIMS];
  int ndim;
  if (!poly_broadcast_pair(ctx, &a, &b, shape, &ndim)) return NULL;
  return poly_alu2(ctx, POLY_OP_FDIV, a, b);
}

/* Pinned Tensor._broadcasted broadcasts first, then applies
 * least_upper_dtype unless either operand is a pointer
 * (mixin/__init__.py:439-449). This is Tensor-expression construction, not a
 * rewrite of imported raw ALU graphs. */
static bool poly_broadcasted_pair(PolyCtx *ctx, PolyUOp **a, PolyUOp **b) {
  int64_t shape[POLY_MAX_DIMS];
  int ndim = 0;
  if (!poly_broadcast_pair(ctx, a, b, shape, &ndim)) return false;
  if (poly_dtype_eq((*a)->dtype, (*b)->dtype) || (*a)->dtype.is_ptr || (*b)->dtype.is_ptr)
    return true;

  PolyDType common;
  if (!poly_dtype_least_upper((*a)->dtype, (*b)->dtype, &common)) return false;
  if (!poly_dtype_eq(poly_dtype_scalar((*a)->dtype), common)) *a = poly_cast(ctx, *a, common);
  if (!poly_dtype_eq(poly_dtype_scalar((*b)->dtype), common)) *b = poly_cast(ctx, *b, common);
  return *a != NULL && *b != NULL;
}

/* Pinned UOp.ufix keeps every Python scalar in a floating receiver's dtype,
 * keeps integer literals in an integer receiver's dtype, and otherwise uses
 * dtypes.from_py. const_like supplies the receiver shape
 * (uop/ops.py:496-508). */
static PolyUOp *poly_ufix_const(
    PolyCtx *ctx,
    PolyUOp *self,
    PolyDType from_py_dtype,
    double value
) {
  if (!ctx || !self) return NULL;
  PolyDType self_dtype = poly_dtype_scalar(self->dtype);
  PolyDType scalar_dtype = poly_dtype_scalar(from_py_dtype);
  PolyDType dtype =
      (poly_dtype_is_float(self_dtype) ||
       (poly_dtype_is_int(self_dtype) && poly_dtype_is_int(scalar_dtype)))
          ? self_dtype
          : scalar_dtype;
  PolyUOp *scalar = poly_const_typed(ctx, dtype, value);
  return scalar ? broadcast_scalar_like(ctx, scalar, self) : NULL;
}

static PolyUOp *poly_broadcasted_alu2(PolyCtx *ctx, PolyOps op, PolyUOp *a, PolyUOp *b) {
  /* Pinned subtraction is ADD(a, MUL(b, -1)), which poly_sub owns. */
  if (op == POLY_OP_SUB) return poly_sub(ctx, a, b);
  if (!poly_broadcasted_pair(ctx, &a, &b)) return NULL;
  return poly_alu2(ctx, op, a, b);
}

static PolyUOp *poly_scalar_binop(
    PolyCtx *ctx,
    PolyOps op,
    PolyUOp *self,
    PolyDType from_py_dtype,
    double value,
    bool reverse
) {
  PolyUOp *scalar = poly_ufix_const(ctx, self, from_py_dtype, value);
  if (!scalar) return NULL;
  return reverse ? poly_broadcasted_alu2(ctx, op, scalar, self)
                 : poly_broadcasted_alu2(ctx, op, self, scalar);
}

/* Contiguous (realize barrier) */

PolyUOp *poly_contiguous(PolyCtx *ctx, PolyUOp *x) {
  if (!ctx || !x) return NULL;
  if (x->op == POLY_OP_CONTIGUOUS) return x;
  /* The device-free fold from pinned UOp.contiguous currently lives at the
   * Tensor boundary above. This raw helper still serves preserved legacy
   * logical builders; removing their explicit barrier is PG-PARITY-022. */
  if (poly_uop_has_buffer_identity(x)) return x;
  return poly_uop1(ctx, POLY_OP_CONTIGUOUS, x->dtype, x, poly_arg_none());
}

/* Math */

PolyUOp *poly_exp(PolyCtx *ctx, PolyUOp *x) {
  if (!ctx || !x) return NULL;
  /* Pinned exp promotes floating inputs to at least float32 and casts the
   * result back; non-floats stay at least default-float
   * (mixin/elementwise.py:491-503). */
  PolyDType input_dt = poly_dtype_scalar(x->dtype);
  PolyDType compute_dt;
  if (!poly_dtype_least_upper(input_dt, POLY_FLOAT32, &compute_dt)) return NULL;
  PolyUOp *compute_x = poly_dtype_eq(input_dt, compute_dt) ? x : poly_cast(ctx, x, compute_dt);
  PolyUOp *scale =
      compute_x ? broadcast_scalar_like(ctx, cdt(ctx, compute_dt, 1.0 / M_LN2), compute_x) : NULL;
  PolyUOp *out =
      scale ? poly_alu1(ctx, POLY_OP_EXP2, poly_alu2(ctx, POLY_OP_MUL, compute_x, scale)) : NULL;
  if (out && poly_dtype_is_float(input_dt) && !poly_dtype_eq(input_dt, compute_dt))
    out = poly_cast(ctx, out, input_dt);
  return out;
}

PolyUOp *poly_log(PolyCtx *ctx, PolyUOp *x) {
  if (!ctx || !x) return NULL;
  /* Pinned log2 promotes only non-floats; log then multiplies by a
   * broadcast same-dtype ln(2) (mixin/elementwise.py:505-527,812-823). */
  PolyDType input_dt = poly_dtype_scalar(x->dtype);
  PolyDType compute_dt = input_dt;
  if (!poly_dtype_is_float(input_dt) &&
      !poly_dtype_least_upper(input_dt, POLY_FLOAT32, &compute_dt))
    return NULL;
  PolyUOp *compute_x = poly_dtype_eq(input_dt, compute_dt) ? x : poly_cast(ctx, x, compute_dt);
  PolyUOp *log2_x = compute_x ? poly_alu1(ctx, POLY_OP_LOG2, compute_x) : NULL;
  PolyUOp *scale = log2_x ? broadcast_scalar_like(ctx, cdt(ctx, compute_dt, M_LN2), log2_x) : NULL;
  return scale ? poly_alu2(ctx, POLY_OP_MUL, log2_x, scale) : NULL;
}

PolyUOp *poly_log1p(PolyCtx *ctx, PolyUOp *x) {
  /* Stable near zero: log(1+x) ~ x - x^2/2 + x^3/3 */
  PolyDType dt = poly_dtype_scalar(x->dtype);
  PolyUOp *ax = poly_abs(ctx, x);
  PolyUOp *small = poly_alu2(ctx, POLY_OP_CMPLT, ax, cdt(ctx, dt, 1e-4));
  PolyUOp *x2 = poly_alu2(ctx, POLY_OP_MUL, x, x);
  PolyUOp *x3 = poly_alu2(ctx, POLY_OP_MUL, x2, x);
  PolyUOp *poly = poly_alu2(
      ctx, POLY_OP_ADD, x,
      poly_alu2(
          ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, -0.5), x2),
          poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, 1.0 / 3.0), x3)
      )
  );
  PolyUOp *direct = poly_log(ctx, poly_alu2(ctx, POLY_OP_ADD, cdt(ctx, dt, 1.0), x));
  return poly_alu3(ctx, POLY_OP_WHERE, small, poly, direct);
}

PolyUOp *poly_expm1(PolyCtx *ctx, PolyUOp *x) {
  /* Stable near zero: expm1(x) ~ x + x^2/2 + x^3/6 */
  PolyDType dt = poly_dtype_scalar(x->dtype);
  PolyUOp *ax = poly_abs(ctx, x);
  PolyUOp *small = poly_alu2(ctx, POLY_OP_CMPLT, ax, cdt(ctx, dt, 1e-4));
  PolyUOp *x2 = poly_alu2(ctx, POLY_OP_MUL, x, x);
  PolyUOp *x3 = poly_alu2(ctx, POLY_OP_MUL, x2, x);
  PolyUOp *poly = poly_alu2(
      ctx, POLY_OP_ADD, x,
      poly_alu2(
          ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, 0.5), x2),
          poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, 1.0 / 6.0), x3)
      )
  );
  PolyUOp *direct = poly_alu2(ctx, POLY_OP_SUB, poly_exp(ctx, x), cdt(ctx, dt, 1.0));
  return poly_alu3(ctx, POLY_OP_WHERE, small, poly, direct);
}

PolyUOp *poly_sin(PolyCtx *ctx, PolyUOp *x) {
  return poly_alu1(ctx, POLY_OP_SIN, x);
}

PolyUOp *poly_cos(PolyCtx *ctx, PolyUOp *x) {
  /* cos(x) = sin(pi/2 - x) */
  return poly_alu1(ctx, POLY_OP_SIN, poly_alu2(ctx, POLY_OP_SUB, cf(ctx, x, M_PI / 2.0), x));
}

PolyUOp *poly_tan(PolyCtx *ctx, PolyUOp *x) {
  /* tan(x) = sin(x) / cos(x) */
  return poly_alu2(ctx, POLY_OP_FDIV, poly_sin(ctx, x), poly_cos(ctx, x));
}

PolyUOp *poly_erf(PolyCtx *ctx, PolyUOp *x) {
  PolyDType dt = poly_dtype_scalar(x->dtype);
  PolyUOp *sign = poly_sign(ctx, x);
  PolyUOp *ax = poly_abs(ctx, x);
  PolyUOp *tau = erf_tau(ctx, ax, dt);
  return poly_alu2(ctx, POLY_OP_MUL, sign, poly_alu2(ctx, POLY_OP_SUB, cdt(ctx, dt, 1.0), tau));
}

PolyUOp *poly_erfc(PolyCtx *ctx, PolyUOp *x) {
  /* erfc(x) = tau(|x|) for x >= 0, 2 - tau(|x|) for x < 0.
   * No 1-erf(x) cancellation -- tau is computed directly. */
  PolyDType dt = poly_dtype_scalar(x->dtype);
  PolyUOp *ax = poly_abs(ctx, x);
  PolyUOp *tau = erf_tau(ctx, ax, dt);
  PolyUOp *neg = poly_alu2(ctx, POLY_OP_CMPLT, x, cdt(ctx, dt, 0.0));
  PolyUOp *erfc_neg = poly_alu2(ctx, POLY_OP_SUB, cdt(ctx, dt, 2.0), tau);
  return poly_alu3(ctx, POLY_OP_WHERE, neg, erfc_neg, tau);
}

PolyUOp *poly_erfinv(PolyCtx *ctx, PolyUOp *x) {
  /* Winitzki approximation (a=0.147). */
  PolyDType dt = poly_dtype_scalar(x->dtype);
  PolyUOp *a = cdt(ctx, dt, 0.147);
  PolyUOp *one = cdt(ctx, dt, 1.0);
  PolyUOp *x2 = poly_alu2(ctx, POLY_OP_MUL, x, x);
  PolyUOp *ln = poly_log(ctx, poly_alu2(ctx, POLY_OP_SUB, one, x2));
  PolyUOp *term1 = poly_alu2(
      ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_FDIV, cdt(ctx, dt, 2.0 / (M_PI * 0.147)), one),
      poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, 0.5), ln)
  );
  PolyUOp *term2 = poly_alu2(ctx, POLY_OP_FDIV, ln, a);
  PolyUOp *inside = poly_alu2(ctx, POLY_OP_SUB, poly_alu2(ctx, POLY_OP_MUL, term1, term1), term2);
  PolyUOp *root = poly_alu1(
      ctx, POLY_OP_SQRT, poly_alu2(ctx, POLY_OP_SUB, poly_alu1(ctx, POLY_OP_SQRT, inside), term1)
  );
  return poly_alu2(ctx, POLY_OP_MUL, poly_sign(ctx, x), root);
}

PolyUOp *poly_ndtri(PolyCtx *ctx, PolyUOp *x) {
  /* ndtri(p) = sqrt(2) * erfinv(2p-1) */
  PolyDType dt = poly_dtype_scalar(x->dtype);
  PolyUOp *arg = poly_alu2(
      ctx, POLY_OP_SUB, poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, 2.0), x), cdt(ctx, dt, 1.0)
  );
  return poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, sqrt(2.0)), poly_erfinv(ctx, arg));
}

PolyUOp *poly_digamma(PolyCtx *ctx, PolyUOp *x) {
  /* First-order asymptotic with recurrence to x>=6. */
  PolyDType dt = poly_dtype_scalar(x->dtype);
  PolyUOp *acc = cdt(ctx, dt, 0.0);
  PolyUOp *xx = x;
  for (int i = 0; i < 6; i++) {
    PolyUOp *cond = poly_alu2(ctx, POLY_OP_CMPLT, xx, cdt(ctx, dt, 6.0));
    PolyUOp *inv = poly_alu1(ctx, POLY_OP_RECIPROCAL, xx);
    acc = poly_alu3(ctx, POLY_OP_WHERE, cond, poly_alu2(ctx, POLY_OP_SUB, acc, inv), acc);
    xx =
        poly_alu3(ctx, POLY_OP_WHERE, cond, poly_alu2(ctx, POLY_OP_ADD, xx, cdt(ctx, dt, 1.0)), xx);
  }
  PolyUOp *inv = poly_alu1(ctx, POLY_OP_RECIPROCAL, xx);
  PolyUOp *inv2 = poly_alu2(ctx, POLY_OP_MUL, inv, inv);
  PolyUOp *inv4 = poly_alu2(ctx, POLY_OP_MUL, inv2, inv2);
  PolyUOp *inv6 = poly_alu2(ctx, POLY_OP_MUL, inv4, inv2);
  PolyUOp *asym = poly_alu2(
      ctx, POLY_OP_ADD, poly_log(ctx, xx),
      poly_alu2(
          ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, -0.5), inv),
          poly_alu2(
              ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, -1.0 / 12.0), inv2),
              poly_alu2(
                  ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, 1.0 / 120.0), inv4),
                  poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, dt, -1.0 / 252.0), inv6)
              )
          )
      )
  );
  return poly_alu2(ctx, POLY_OP_ADD, acc, asym);
}

PolyUOp *poly_lgamma(PolyCtx *ctx, PolyUOp *x) {
  /* Explicit VJP override:
   * y = detach(f(x)) + (x - detach(x))*digamma(x) */
  PolyDType dt = poly_dtype_scalar(x->dtype);
  PolyUOp *fwd = poly_lgamma_forward_lanczos(ctx, x, dt);
  PolyUOp *dx = poly_uop1(ctx, POLY_OP_DETACH, x->dtype, x, poly_arg_none());
  PolyUOp *df = poly_uop1(ctx, POLY_OP_DETACH, fwd->dtype, fwd, poly_arg_none());
  PolyUOp *delta = poly_alu2(ctx, POLY_OP_SUB, x, dx);
  PolyUOp *forced = poly_alu2(ctx, POLY_OP_MUL, delta, poly_digamma(ctx, x));
  return poly_alu2(ctx, POLY_OP_ADD, df, forced);
}

PolyUOp *poly_sigmoid(PolyCtx *ctx, PolyUOp *x) {
  /* Pinned sigmoid:
   * (1 + (x * (-1/log(2))).exp2()).reciprocal()
   * (mixin/elementwise.py:667-679). UOp.ufix keeps float16/float64 receivers
   * in their own dtype; integer/bool receivers promote at the multiplication. */
  if (!ctx || !x) return NULL;
  PolyUOp *scaled =
      poly_scalar_binop(ctx, POLY_OP_MUL, x, POLY_FLOAT32, -1.0 / M_LN2, false);
  PolyUOp *e = scaled ? poly_alu1(ctx, POLY_OP_EXP2, scaled) : NULL;
  PolyUOp *denom = e ? poly_scalar_binop(ctx, POLY_OP_ADD, e, POLY_INT32, 1.0, true) : NULL;
  return denom ? poly_alu1(ctx, POLY_OP_RECIPROCAL, denom) : NULL;
}

PolyUOp *poly_tanh_act(PolyCtx *ctx, PolyUOp *x) {
  /* Pinned tanh(x) = 2.0 * sigmoid(2.0 * x) - 1.0
   * (mixin/elementwise.py:739-749). */
  if (!ctx || !x) return NULL;
  PolyUOp *two_x = poly_scalar_binop(ctx, POLY_OP_MUL, x, POLY_FLOAT32, 2.0, true);
  PolyUOp *sigmoid = two_x ? poly_sigmoid(ctx, two_x) : NULL;
  PolyUOp *twice =
      sigmoid ? poly_scalar_binop(ctx, POLY_OP_MUL, sigmoid, POLY_FLOAT32, 2.0, true)
              : NULL;
  return twice ? poly_scalar_binop(ctx, POLY_OP_SUB, twice, POLY_FLOAT32, 1.0, false)
               : NULL;
}

PolyUOp *poly_abs(PolyCtx *ctx, PolyUOp *x) {
  /* abs(x) = x * sign(x) */
  return poly_alu2(ctx, POLY_OP_MUL, x, poly_sign(ctx, x));
}

PolyUOp *poly_sign(PolyCtx *ctx, PolyUOp *x) {
  /* sign(x) = ne(x,0).where(lt(x,0).where(-1, 1), 0) + x*0
   * The +x*0 preserves NaN (NaN*0=NaN, NaN+0=NaN) */
  PolyUOp *zero = cf(ctx, x, 0.0);
  PolyUOp *is_nonzero = poly_alu2(ctx, POLY_OP_CMPNE, x, zero);
  PolyUOp *is_neg = poly_alu2(ctx, POLY_OP_CMPLT, x, zero);
  PolyUOp *neg_or_pos = poly_alu3(ctx, POLY_OP_WHERE, is_neg, cf(ctx, x, -1.0), cf(ctx, x, 1.0));
  PolyUOp *result = poly_alu3(ctx, POLY_OP_WHERE, is_nonzero, neg_or_pos, zero);
  /* +x*0 to propagate NaN */
  return poly_alu2(ctx, POLY_OP_ADD, result, poly_alu2(ctx, POLY_OP_MUL, x, zero));
}

PolyUOp *poly_square(PolyCtx *ctx, PolyUOp *x) {
  return poly_alu2(ctx, POLY_OP_MUL, x, x);
}

PolyUOp *poly_rsqrt(PolyCtx *ctx, PolyUOp *x) {
  return poly_alu1(ctx, POLY_OP_RECIPROCAL, poly_alu1(ctx, POLY_OP_SQRT, x));
}

PolyUOp *poly_ceil(PolyCtx *ctx, PolyUOp *x) {
  /* ceil(x) = (x > (b=trunc(x))).where(b+1, b) */
  PolyUOp *b = poly_alu1(ctx, POLY_OP_TRUNC, x);
  PolyUOp *cond = poly_alu2(ctx, POLY_OP_CMPLT, b, x); /* b < x = x > b */
  return poly_alu3(ctx, POLY_OP_WHERE, cond, poly_alu2(ctx, POLY_OP_ADD, b, cf(ctx, x, 1.0)), b);
}

PolyUOp *poly_floor(PolyCtx *ctx, PolyUOp *x) {
  /* floor(x) = (x < (b=trunc(x))).where(b-1, b) */
  PolyUOp *b = poly_alu1(ctx, POLY_OP_TRUNC, x);
  PolyUOp *cond = poly_alu2(ctx, POLY_OP_CMPLT, x, b); /* x < b */
  return poly_alu3(ctx, POLY_OP_WHERE, cond, poly_alu2(ctx, POLY_OP_SUB, b, cf(ctx, x, 1.0)), b);
}

PolyUOp *poly_round_f(PolyCtx *ctx, PolyUOp *x) {
  /* round(x) with banker's rounding (round half to even):
   * (x > 0) == (trunc(x/2) == trunc(trunc(x)/2)) ? ceil(x-0.5) : floor(x+0.5) */
  PolyUOp *half = cf(ctx, x, 0.5);
  PolyUOp *two = cf(ctx, x, 2.0);
  PolyUOp *b = poly_alu1(ctx, POLY_OP_TRUNC, x);
  PolyUOp *x_gt_0 = poly_alu2(ctx, POLY_OP_CMPLT, cf(ctx, x, 0.0), x);
  PolyUOp *b_half = poly_alu2(ctx, POLY_OP_FDIV, b, two);
  PolyUOp *x_half = poly_alu2(ctx, POLY_OP_FDIV, x, two);
  PolyUOp *trunc_b_half = poly_alu1(ctx, POLY_OP_TRUNC, b_half);
  PolyUOp *trunc_x_half = poly_alu1(ctx, POLY_OP_TRUNC, x_half);
  PolyUOp *halves_eq = poly_eq(ctx, trunc_b_half, trunc_x_half);
  PolyUOp *cond = poly_eq(ctx, x_gt_0, halves_eq);
  return poly_alu3(
      ctx, POLY_OP_WHERE, cond, poly_ceil(ctx, poly_alu2(ctx, POLY_OP_SUB, x, half)),
      poly_floor(ctx, poly_alu2(ctx, POLY_OP_ADD, x, half))
  );
}

PolyUOp *poly_isinf(PolyCtx *ctx, PolyUOp *x) {
  PolyUOp *zero = cf(ctx, x, 0.0);
  PolyUOp *x_minus_x = poly_alu2(ctx, POLY_OP_SUB, x, x);
  PolyUOp *not_nan = poly_eq(ctx, x, x); /* true if not NaN */
  PolyUOp *sub_nan = poly_ne(ctx, x_minus_x, x_minus_x); /* true if x-x is NaN (inf case) */
  PolyUOp *not_zero = poly_ne(ctx, x, zero);
  /* all three must be true: use AND via MUL on bool-like values */
  PolyUOp *t1 = poly_alu2(ctx, POLY_OP_MUL, not_nan, sub_nan);
  return poly_alu2(ctx, POLY_OP_MUL, t1, not_zero);
}

PolyUOp *poly_isnan(PolyCtx *ctx, PolyUOp *x) {
  /* isnan(x) = (x != x) -- IEEE 754 */
  return poly_alu2(ctx, POLY_OP_CMPNE, x, x);
}

/* Activations */

PolyUOp *poly_relu(PolyCtx *ctx, PolyUOp *x) {
  /* relu(x) = where(0 < x, x, 0) */
  PolyUOp *zero = cf(ctx, x, 0.0);
  PolyUOp *cond = poly_alu2(ctx, POLY_OP_CMPLT, zero, x);
  return poly_alu3(ctx, POLY_OP_WHERE, cond, x, zero);
}

PolyUOp *poly_relu6(PolyCtx *ctx, PolyUOp *x) {
  /* relu6(x) = relu(x) - relu(x - 6) */
  return poly_alu2(
      ctx, POLY_OP_SUB, poly_relu(ctx, x),
      poly_relu(ctx, poly_alu2(ctx, POLY_OP_SUB, x, cf(ctx, x, 6.0)))
  );
}

PolyUOp *poly_leaky_relu(PolyCtx *ctx, PolyUOp *x, double neg_slope) {
  PolyUOp *cond = poly_alu2(ctx, POLY_OP_CMPLT, x, cf(ctx, x, 0.0));
  return poly_alu3(
      ctx, POLY_OP_WHERE, cond, poly_alu2(ctx, POLY_OP_MUL, cf(ctx, x, neg_slope), x), x
  );
}

PolyUOp *poly_gelu(PolyCtx *ctx, PolyUOp *x) {
  /* Pinned tanh GELU:
   * 0.5*x*(1 + (sqrt(2/pi)*(x + 0.044715*x**3)).tanh())
   * (mixin/elementwise.py:761-776). Keep every source operation's own ufix
   * and promotion boundary: integer x**3 remains integer until multiplied by
   * the floating coefficient. */
  if (!ctx || !x) return NULL;
  PolyUOp *half_x = poly_scalar_binop(ctx, POLY_OP_MUL, x, POLY_FLOAT32, 0.5, true);
  PolyUOp *x3 = poly_scalar_binop(ctx, POLY_OP_POW, x, POLY_INT32, 3.0, false);
  PolyUOp *cubic =
      x3 ? poly_scalar_binop(ctx, POLY_OP_MUL, x3, POLY_FLOAT32, 0.044715, true) : NULL;
  PolyUOp *inner = cubic ? poly_broadcasted_alu2(ctx, POLY_OP_ADD, x, cubic) : NULL;
  PolyUOp *scaled =
      inner ? poly_scalar_binop(
                  ctx, POLY_OP_MUL, inner, POLY_FLOAT32, sqrt(2.0 / M_PI), true
              )
            : NULL;
  PolyUOp *tanh = scaled ? poly_tanh_act(ctx, scaled) : NULL;
  PolyUOp *one_plus =
      tanh ? poly_scalar_binop(ctx, POLY_OP_ADD, tanh, POLY_INT32, 1.0, true) : NULL;
  return half_x && one_plus ? poly_broadcasted_alu2(ctx, POLY_OP_MUL, half_x, one_plus)
                           : NULL;
}

PolyUOp *poly_quick_gelu(PolyCtx *ctx, PolyUOp *x) {
  /* Pinned quick_gelu is self * (self * 1.702).sigmoid()
   * (mixin/elementwise.py:751-759). */
  if (!ctx || !x) return NULL;
  PolyUOp *inner = poly_scalar_binop(ctx, POLY_OP_MUL, x, POLY_FLOAT32, 1.702, false);
  PolyUOp *sigmoid = inner ? poly_sigmoid(ctx, inner) : NULL;
  return sigmoid ? poly_broadcasted_alu2(ctx, POLY_OP_MUL, x, sigmoid) : NULL;
}

PolyUOp *poly_silu(PolyCtx *ctx, PolyUOp *x) {
  PolyUOp *sigmoid = poly_sigmoid(ctx, x);
  return sigmoid ? poly_broadcasted_alu2(ctx, POLY_OP_MUL, x, sigmoid) : NULL;
}

PolyUOp *poly_elu(PolyCtx *ctx, PolyUOp *x, double alpha) {
  return poly_alu2(
      ctx, POLY_OP_SUB, poly_relu(ctx, x),
      poly_alu2(
          ctx, POLY_OP_MUL, cf(ctx, x, alpha),
          poly_relu(ctx, poly_alu2(ctx, POLY_OP_SUB, cf(ctx, x, 1.0), poly_exp(ctx, x)))
      )
  );
}

PolyUOp *poly_softplus(PolyCtx *ctx, PolyUOp *x, double beta) {
  PolyUOp *bx = poly_alu2(ctx, POLY_OP_MUL, cf(ctx, x, beta), x);
  PolyUOp *zero = cf(ctx, x, 0.0);
  PolyUOp *m = poly_alu2(ctx, POLY_OP_MAX, bx, zero);
  PolyUOp *ea = poly_exp(ctx, poly_alu2(ctx, POLY_OP_SUB, bx, m));
  PolyUOp *eb = poly_exp(ctx, poly_alu2(ctx, POLY_OP_SUB, zero, m));
  PolyUOp *lae = poly_alu2(ctx, POLY_OP_ADD, m, poly_log(ctx, poly_alu2(ctx, POLY_OP_ADD, ea, eb)));
  return poly_alu2(ctx, POLY_OP_MUL, cf(ctx, x, 1.0 / beta), lae);
}

PolyUOp *poly_mish(PolyCtx *ctx, PolyUOp *x) {
  return poly_alu2(ctx, POLY_OP_MUL, x, poly_tanh_act(ctx, poly_softplus(ctx, x, 1.0)));
}

PolyUOp *poly_hardtanh(PolyCtx *ctx, PolyUOp *x, double min_val, double max_val) {
  return poly_clamp(ctx, x, min_val, max_val);
}

PolyUOp *poly_hardswish(PolyCtx *ctx, PolyUOp *x) {
  return poly_alu2(
      ctx, POLY_OP_MUL,
      poly_alu2(
          ctx, POLY_OP_MUL, x, poly_relu6(ctx, poly_alu2(ctx, POLY_OP_ADD, x, cf(ctx, x, 3.0)))
      ),
      cf(ctx, x, 1.0 / 6.0)
  );
}

PolyUOp *poly_hardsigmoid(PolyCtx *ctx, PolyUOp *x) {
  PolyUOp *t = poly_alu2(
      ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_MUL, cf(ctx, x, 1.0 / 6.0), x), cf(ctx, x, 0.5)
  );
  return poly_alu2(
      ctx, POLY_OP_SUB, poly_relu(ctx, t),
      poly_relu(ctx, poly_alu2(ctx, POLY_OP_SUB, t, cf(ctx, x, 1.0)))
  );
}

/* Comparisons (broadcasting) */

/* Logical NOT — `CMPNE(x, CONST(true))` for
 * bool inputs, matching tinygrad's `logical_not()` after CAST elision
 * (mixin/elementwise.py:39-47 + symbolic.py:93-131). Raw `NEG(bool)` retains
 * arithmetic NEG semantics and is not a second logical-NOT spelling. */
PolyUOp *poly_logical_not(PolyCtx *ctx, PolyUOp *x) {
  PolyUOp *t = poly_const_typed(ctx, POLY_BOOL, 1);
  int64_t shape[POLY_MAX_DIMS];
  int ndim;
  if (!poly_broadcast_pair(ctx, &x, &t, shape, &ndim)) return NULL;
  return poly_alu2(ctx, POLY_OP_CMPNE, x, t);
}

/* All comparison helpers return BOOL, mirroring tinygrad's
 * mixin/elementwise.py:218-247:
 *   eq(a,b) = (a != b).logical_not()
 *   ne(a,b) = CMPNE(a,b)
 *   gt(a,b) = CMPLT(b,a)         (operand swap)
 *   lt(a,b) = CMPLT(a,b)
 *   ge(a,b) = (a < b).logical_not()
 *   le(a,b) = (a > b).logical_not() = (b < a).logical_not()
 *
 * Polygrad previously had ge/le returning float WHERE(0,1); fixed in P5
 * for tinygrad parity and to let Phase D's reduce_collapse Rule 4 match
 * polygrad's tril/triu masks. */
PolyUOp *poly_eq(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  int64_t s[POLY_MAX_DIMS];
  int nd;
  if (!poly_broadcast_pair(ctx, &a, &b, s, &nd)) return NULL;
  PolyUOp *ne = poly_alu2(ctx, POLY_OP_CMPNE, a, b);
  return poly_logical_not(ctx, ne);
}

PolyUOp *poly_ne(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  int64_t s[POLY_MAX_DIMS];
  int nd;
  if (!poly_broadcast_pair(ctx, &a, &b, s, &nd)) return NULL;
  return poly_alu2(ctx, POLY_OP_CMPNE, a, b);
}

PolyUOp *poly_gt(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  int64_t s[POLY_MAX_DIMS];
  int nd;
  if (!poly_broadcast_pair(ctx, &a, &b, s, &nd)) return NULL;
  return poly_alu2(ctx, POLY_OP_CMPLT, b, a);
}

PolyUOp *poly_ge(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  int64_t s[POLY_MAX_DIMS];
  int nd;
  if (!poly_broadcast_pair(ctx, &a, &b, s, &nd)) return NULL;
  PolyUOp *lt = poly_alu2(ctx, POLY_OP_CMPLT, a, b);
  return poly_logical_not(ctx, lt);
}

PolyUOp *poly_le(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  int64_t s[POLY_MAX_DIMS];
  int nd;
  if (!poly_broadcast_pair(ctx, &a, &b, s, &nd)) return NULL;
  PolyUOp *gt = poly_alu2(ctx, POLY_OP_CMPLT, b, a);
  return poly_logical_not(ctx, gt);
}

PolyUOp *poly_cast(PolyCtx *ctx, PolyUOp *x, PolyDType target) {
  if (!ctx || !x) return NULL;
  /* Pinned UOp.cast propagates the source vector count when given a scalar
   * target, then returns self for an exact dtype match (uop/ops.py:513-518). */
  if (target.count == 1 && x->dtype.count != 1) target = poly_dtype_vec(target, x->dtype.count);
  if (poly_dtype_eq(x->dtype, target)) return x;
  return poly_uop1(ctx, POLY_OP_CAST, target, x, poly_arg_none());
}

PolyUOp *poly_cast_by_id(PolyCtx *ctx, PolyUOp *x, int dtype_id) {
  if (dtype_id < 0 || dtype_id >= N_DTYPE_FFI) return NULL;
  return poly_cast(ctx, x, *_dtype_table_ffi[dtype_id]);
}

PolyUOp *poly_bitcast_by_id(PolyCtx *ctx, PolyUOp *x, int dtype_id) {
  const PolyDType *target = ffi_dtype_from_id(dtype_id);
  if (!ctx || !x || !target) return NULL;
  PolyDType src_scalar = poly_dtype_scalar(x->dtype);
  PolyDType target_scalar = poly_dtype_scalar(*target);
  if (poly_dtype_itemsize(src_scalar) != poly_dtype_itemsize(target_scalar)) return NULL;
  return poly_uop1(ctx, POLY_OP_BITCAST, target_scalar, x, poly_arg_none());
}

PolyUOp *poly_where_op(PolyCtx *ctx, PolyUOp *cond, PolyUOp *x, PolyUOp *y) {
  int64_t s[POLY_MAX_DIMS];
  int nd;
  if (!cond || !poly_broadcast_pair(ctx, &x, &y, s, &nd)) return NULL;
  /* tinygrad Tensor.where casts non-bool conditions to bool before building
   * Ops.WHERE. Keeping that in the core constructor preserves the expected
   * CMPNE(cond, 0) node in helper graphs such as nonzero-value padding. */
  if (!poly_dtype_is_bool(poly_dtype_scalar(cond->dtype))) cond = poly_cast(ctx, cond, POLY_BOOL);
  if (!poly_broadcast_pair(ctx, &cond, &x, s, &nd)) return NULL;
  /* UOp.alu broadcasts every shaped source to the final common shape
   * (uop/ops.py:543-551). The condition may enlarge x after the first x/y
   * pair, so complete y against that final shape too. */
  if (!poly_broadcast_pair(ctx, &x, &y, s, &nd)) return NULL;
  return poly_alu3(ctx, POLY_OP_WHERE, cond, x, y);
}

PolyUOp *poly_maximum(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  int64_t s[POLY_MAX_DIMS];
  int nd;
  if (!poly_broadcast_pair(ctx, &a, &b, s, &nd)) return NULL;
  return poly_alu2(ctx, POLY_OP_MAX, a, b);
}

/* Pinned ElementwiseMixin._inverse is ordinary broadcasted negation for
 * floating values and bitwise-not for integer/bool values
 * (mixin/elementwise.py:57-69,131-143,379-393). */
static PolyUOp *minimum_inverse(PolyCtx *ctx, PolyUOp *x) {
  if (!ctx || !x) return NULL;
  PolyDType dt = poly_dtype_scalar(x->dtype);
  if (poly_dtype_is_float(dt)) {
    PolyUOp *minus_one = broadcast_scalar_like(ctx, poly_const_exact_float(ctx, dt, -1.0), x);
    return minus_one ? poly_alu2(ctx, POLY_OP_MUL, x, minus_one) : NULL;
  }
  if (poly_dtype_is_bool(dt)) return poly_logical_not(ctx, x);
  if (!poly_dtype_is_int(dt)) return NULL;
  PolyUOp *mask = NULL;
  if (poly_dtype_is_unsigned(dt)) {
    if (!poly_dtype_bound_const(ctx, dt, false, &mask)) return NULL;
  } else {
    mask = poly_const_exact_int(ctx, dt, -1);
  }
  mask = mask ? broadcast_scalar_like(ctx, mask, x) : NULL;
  return mask ? poly_alu2(ctx, POLY_OP_XOR, x, mask) : NULL;
}

PolyUOp *poly_minimum(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  int64_t s[POLY_MAX_DIMS];
  int nd;
  if (!poly_broadcast_pair(ctx, &a, &b, s, &nd)) return NULL;
  PolyDType common;
  if (!poly_dtype_least_upper(a->dtype, b->dtype, &common)) return NULL;
  if (!poly_dtype_eq(poly_dtype_scalar(a->dtype), common)) a = poly_cast(ctx, a, common);
  if (!poly_dtype_eq(poly_dtype_scalar(b->dtype), common)) b = poly_cast(ctx, b, common);
  PolyUOp *inverse_a = minimum_inverse(ctx, a);
  PolyUOp *inverse_b = minimum_inverse(ctx, b);
  PolyUOp *maximum =
      (inverse_a && inverse_b) ? poly_alu2(ctx, POLY_OP_MAX, inverse_a, inverse_b) : NULL;
  return maximum ? minimum_inverse(ctx, maximum) : NULL;
}

PolyUOp *poly_clamp(PolyCtx *ctx, PolyUOp *x, double lo, double hi) {
  PolyUOp *lo_c = cf(ctx, x, lo);
  PolyUOp *hi_c = cf(ctx, x, hi);
  PolyUOp *lt_lo = poly_alu2(ctx, POLY_OP_CMPLT, x, lo_c);
  PolyUOp *clamped_lo = poly_alu3(ctx, POLY_OP_WHERE, lt_lo, lo_c, x);
  PolyUOp *gt_hi = poly_alu2(ctx, POLY_OP_CMPLT, hi_c, clamped_lo);
  return poly_alu3(ctx, POLY_OP_WHERE, gt_hi, hi_c, clamped_lo);
}

PolyUOp *poly_detach(PolyCtx *ctx, PolyUOp *x) {
  return poly_uop1(ctx, POLY_OP_DETACH, x->dtype, x, poly_arg_none());
}

/* ══════════════════════════════════════════════════════════════════════ */
/*  Movement-op helpers (port of tinygrad mixin/movement.py)             */
/*                                                                       */
/*  Pure tensor-level helpers built from existing primitives. Static     */
/*  int64_t shapes only -- polygrad does not yet match tinygrad's        */
/*  symbolic-shape movement args.                                        */
/* ══════════════════════════════════════════════════════════════════════ */

/* Tensor.repeat -- movement.py:465 */
PolyUOp *poly_repeat(PolyCtx *ctx, PolyUOp *x, const int64_t *repeats, int n_repeats) {
  int64_t in_shape[POLY_MAX_DIMS];
  int in_ndim = uop_shape(ctx, x, in_shape);
  if (in_ndim < 0 || in_ndim > n_repeats || n_repeats > POLY_MAX_DIMS) return NULL;

  /* _align_left: pad input shape with leading 1s to match n_repeats */
  int64_t base[POLY_MAX_DIMS];
  int pad = n_repeats - in_ndim;
  for (int i = 0; i < pad; i++)
    base[i] = 1;
  for (int i = 0; i < in_ndim; i++)
    base[pad + i] = in_shape[i];

  /* unsqueezed = flatten([[s] if r==1 else [1,s] for r,s in zip(repeats, base)])
   * expanded   = flatten([[s] if r==1 else [r,s] for r,s in zip(repeats, base)])
   * final      = [r*s for r,s in zip(repeats, base)] */
  int64_t unsq[POLY_MAX_DIMS * 2], exp[POLY_MAX_DIMS * 2], final_sh[POLY_MAX_DIMS];
  int n = 0;
  for (int i = 0; i < n_repeats; i++) {
    int64_t r = repeats[i], s = base[i];
    if (r == 1) {
      unsq[n] = s;
      exp[n] = s;
      n++;
    } else {
      unsq[n] = 1;
      exp[n] = r;
      n++;
      unsq[n] = s;
      exp[n] = s;
      n++;
    }
    final_sh[i] = r * s;
  }

  return poly_reshape(
      ctx, poly_expand(ctx, poly_reshape(ctx, x, unsq, n), exp, n), final_sh, n_repeats
  );
}

/* Tensor.shrink_to -- movement.py:168. ends[i] == -1 means no-op (keep dim). */
PolyUOp *poly_shrink_to(PolyCtx *ctx, PolyUOp *x, const int64_t *ends, int n_ends) {
  int64_t in_shape[POLY_MAX_DIMS];
  int in_ndim = uop_shape(ctx, x, in_shape);
  if (in_ndim != n_ends) return NULL;

  int64_t pairs[POLY_MAX_DIMS][2];
  bool any = false;
  for (int i = 0; i < n_ends; i++) {
    int64_t e = (ends[i] == -1) ? in_shape[i] : ends[i];
    pairs[i][0] = 0;
    pairs[i][1] = e;
    if (e != in_shape[i]) any = true;
  }
  return any ? poly_shrink(ctx, x, pairs, n_ends) : x;
}

/* Tensor._pool -- movement.py:487. General N-d pool via repeat/shrink/reshape/permute. */
PolyUOp *poly_pool(
    PolyCtx *ctx,
    PolyUOp *x,
    const int64_t *k_,
    int nk,
    const int64_t *stride_,
    const int64_t *dilation_
) {
  int64_t sh[POLY_MAX_DIMS];
  int ndim = uop_shape(ctx, x, sh);
  if (ndim < nk) return NULL;
  int noop = ndim - nk;

  int64_t s_[POLY_MAX_DIMS], d_[POLY_MAX_DIMS], i_[POLY_MAX_DIMS];
  int64_t o_[POLY_MAX_DIMS], f_[POLY_MAX_DIMS];
  for (int j = 0; j < nk; j++) {
    s_[j] = stride_ ? stride_[j] : 1;
    d_[j] = dilation_ ? dilation_[j] : 1;
    i_[j] = sh[noop + j];
    if (d_[j] * (k_[j] - 1) + 1 > i_[j]) return NULL;
    o_[j] = (i_[j] - d_[j] * (k_[j] - 1) + s_[j] - 1) / s_[j]; /* ceildiv */
    int64_t fn = o_[j] * s_[j] - d_[j];
    int64_t fv = fn <= 0 ? 1 : (fn + i_[j] - 1) / i_[j];
    f_[j] = fv < 1 ? 1 : fv;
  }
  if (noop + 3 * nk > POLY_MAX_DIMS) return NULL;

  /* x = repeat([1]*noop + [ceildiv(k*(i*f+d), i) for ...]) */
  int64_t rep[POLY_MAX_DIMS];
  for (int j = 0; j < noop; j++)
    rep[j] = 1;
  for (int j = 0; j < nk; j++) {
    int64_t num = k_[j] * (i_[j] * f_[j] + d_[j]);
    rep[noop + j] = (num + i_[j] - 1) / i_[j];
  }
  PolyUOp *r = poly_repeat(ctx, x, rep, ndim);

  /* shrink_to(noop + [k*(i*f+d) for ...]) */
  int64_t e1[POLY_MAX_DIMS];
  for (int j = 0; j < noop; j++)
    e1[j] = -1;
  for (int j = 0; j < nk; j++)
    e1[noop + j] = k_[j] * (i_[j] * f_[j] + d_[j]);
  r = poly_shrink_to(ctx, r, e1, ndim);

  /* reshape(noop + flatten((k, i*f+d) for ...)) */
  int64_t s1[POLY_MAX_DIMS];
  for (int j = 0; j < noop; j++)
    s1[j] = sh[j];
  for (int j = 0; j < nk; j++) {
    s1[noop + 2 * j] = k_[j];
    s1[noop + 2 * j + 1] = i_[j] * f_[j] + d_[j];
  }
  r = poly_reshape(ctx, r, s1, noop + 2 * nk);

  /* shrink_to(noop + flatten((k, o*s) for ...)).reshape(noop + flatten((k, o, s) for ...)) */
  int64_t e2[POLY_MAX_DIMS], s2[POLY_MAX_DIMS];
  for (int j = 0; j < noop; j++) {
    e2[j] = -1;
    s2[j] = sh[j];
  }
  for (int j = 0; j < nk; j++) {
    e2[noop + 2 * j] = k_[j];
    e2[noop + 2 * j + 1] = o_[j] * s_[j];
    s2[noop + 3 * j] = k_[j];
    s2[noop + 3 * j + 1] = o_[j];
    s2[noop + 3 * j + 2] = s_[j];
  }
  r = poly_reshape(ctx, poly_shrink_to(ctx, r, e2, noop + 2 * nk), s2, noop + 3 * nk);

  /* shrink_to(noop + flatten((k, o, 1) for ...)).reshape(noop + flatten((k, o) for ...)) */
  int64_t e3[POLY_MAX_DIMS], s3[POLY_MAX_DIMS];
  for (int j = 0; j < noop; j++) {
    e3[j] = -1;
    s3[j] = sh[j];
  }
  for (int j = 0; j < nk; j++) {
    e3[noop + 3 * j] = k_[j];
    e3[noop + 3 * j + 1] = o_[j];
    e3[noop + 3 * j + 2] = 1;
    s3[noop + 2 * j] = k_[j];
    s3[noop + 2 * j + 1] = o_[j];
  }
  r = poly_reshape(ctx, poly_shrink_to(ctx, r, e3, noop + 3 * nk), s3, noop + 2 * nk);

  /* permute(*range(noop), *[noop + i*2 + 1 for i in range(nk)],
   *[noop + i*2     for i in range(nk)]) */
  int64_t perm[POLY_MAX_DIMS];
  for (int j = 0; j < noop; j++)
    perm[j] = j;
  for (int j = 0; j < nk; j++)
    perm[noop + j] = noop + 2 * j + 1;
  for (int j = 0; j < nk; j++)
    perm[noop + nk + j] = noop + 2 * j;
  return poly_permute(ctx, r, perm, noop + 2 * nk);
}

static bool resolve_pool_padding(const int64_t *padding, int n_padding, int nk, int64_t *out) {
  if (nk < 0 || nk > POLY_MAX_DIMS || !out) return false;
  if (!padding || n_padding == 0) {
    for (int i = 0; i < 2 * nk; i++)
      out[i] = 0;
    return true;
  }
  if (n_padding == 1) {
    for (int i = 0; i < 2 * nk; i++)
      out[i] = padding[0];
    return true;
  }
  if (n_padding == nk) {
    for (int i = 0; i < nk; i++) {
      out[2 * (nk - 1 - i)] = padding[i];
      out[2 * (nk - 1 - i) + 1] = padding[i];
    }
    return true;
  }
  if (n_padding == 2 * nk) {
    for (int i = 0; i < 2 * nk; i++)
      out[i] = padding[i];
    return true;
  }
  return false;
}

static void flat_padding_to_pairs(const int64_t *padding, int nk, int ndim, int64_t (*pairs)[2]) {
  int noop = ndim - nk;
  for (int i = 0; i < noop; i++) {
    pairs[i][0] = 0;
    pairs[i][1] = 0;
  }
  for (int j = 0; j < nk; j++) {
    int pi = 2 * (nk - 1 - j);
    pairs[noop + j][0] = padding[pi];
    pairs[noop + j][1] = padding[pi + 1];
  }
}

static double dtype_min_identity(PolyDType dt) {
  PolyDType s = poly_dtype_scalar(dt);
  if (poly_dtype_is_float(s)) return -INFINITY;
  if (poly_dtype_is_bool(s)) return 0.0;
  if (poly_dtype_is_unsigned(s)) return 0.0;
  if (poly_dtype_eq(s, POLY_INT8)) return (double)INT8_MIN;
  if (poly_dtype_eq(s, POLY_INT16)) return (double)INT16_MIN;
  if (poly_dtype_eq(s, POLY_INT32)) return (double)INT32_MIN;
  if (poly_dtype_eq(s, POLY_INT64)) return (double)INT64_MIN;
  return (double)INT64_MIN;
}

PolyUOp *poly_max_pool2d(
    PolyCtx *ctx,
    PolyUOp *x,
    const int64_t *kernel,
    int n_kernel,
    const int64_t *stride,
    const int64_t *dilation,
    const int64_t *padding,
    int n_padding
) {
  if (!ctx || !x) return NULL;
  int nk = n_kernel == 0 ? 2 : n_kernel;
  if (nk <= 0 || nk > POLY_MAX_DIMS) return NULL;
  int64_t k[POLY_MAX_DIMS], s[POLY_MAX_DIMS], d[POLY_MAX_DIMS], pads[2 * POLY_MAX_DIMS];
  for (int i = 0; i < nk; i++) {
    k[i] = kernel ? kernel[i] : 2;
    s[i] = stride ? stride[i] : k[i];
    d[i] = dilation ? dilation[i] : 1;
    if (k[i] <= 0 || s[i] <= 0 || d[i] <= 0) return NULL;
  }
  if (!resolve_pool_padding(padding, n_padding, nk, pads)) return NULL;

  int64_t shape[POLY_MAX_DIMS];
  int ndim = uop_shape(ctx, x, shape);
  if (ndim < nk) return NULL;
  int64_t pad_pairs[POLY_MAX_DIMS][2];
  flat_padding_to_pairs(pads, nk, ndim, pad_pairs);
  PolyUOp *padded = poly_pad_value(ctx, x, pad_pairs, ndim, dtype_min_identity(x->dtype));
  PolyUOp *pooled = poly_pool(ctx, padded, k, nk, s, d);
  if (!pooled) return NULL;
  int64_t axes[POLY_MAX_DIMS];
  for (int i = 0; i < nk; i++)
    axes[i] = ndim + i;
  PolyUOp *r = poly_reduce_axis(ctx, POLY_OP_MAX, pooled, axes, nk);
  int64_t pshape[POLY_MAX_DIMS];
  int pndim = uop_shape(ctx, pooled, pshape);
  if (pndim < nk) return NULL;
  int out_ndim = pndim - nk;
  int64_t out_shape[POLY_MAX_DIMS];
  for (int i = 0; i < out_ndim; i++)
    out_shape[i] = pshape[i];
  return poly_reshape(ctx, r, out_shape, out_ndim);
}

PolyUOp *poly_conv2d(
    PolyCtx *ctx,
    PolyUOp *x,
    PolyUOp *weight,
    PolyUOp *bias,
    int groups,
    const int64_t *stride,
    const int64_t *dilation,
    const int64_t *padding,
    int n_padding
) {
  if (!ctx || !x || !weight || groups <= 0) return NULL;
  int64_t x_shape[POLY_MAX_DIMS], w_shape[POLY_MAX_DIMS];
  int x_ndim = uop_shape(ctx, x, x_shape);
  int w_ndim = uop_shape(ctx, weight, w_shape);
  if (x_ndim < 3 || w_ndim != x_ndim) return NULL;
  int hw_ndim = x_ndim - 2;
  if (hw_ndim <= 0 || hw_ndim > POLY_MAX_DIMS) return NULL;
  int64_t bs = x_shape[0], cin_total = x_shape[1];
  int64_t cout = w_shape[0], cin = w_shape[1];
  if (groups * cin != cin_total || cout % groups != 0) return NULL;

  int64_t s[POLY_MAX_DIMS], d[POLY_MAX_DIMS], pads[2 * POLY_MAX_DIMS];
  for (int i = 0; i < hw_ndim; i++) {
    s[i] = stride ? stride[i] : 1;
    d[i] = dilation ? dilation[i] : 1;
    if (s[i] <= 0 || d[i] <= 0) return NULL;
  }
  if (!resolve_pool_padding(padding, n_padding, hw_ndim, pads)) return NULL;
  int64_t pad_pairs[POLY_MAX_DIMS][2];
  flat_padding_to_pairs(pads, hw_ndim, x_ndim, pad_pairs);
  PolyUOp *xp = poly_pad_value(ctx, x, pad_pairs, x_ndim, 0.0);
  PolyUOp *pooled = poly_pool(ctx, xp, &w_shape[2], hw_ndim, s, d);
  if (!pooled) return NULL;

  int64_t pshape[POLY_MAX_DIMS];
  int pndim = uop_shape(ctx, pooled, pshape);
  if (pndim != x_ndim + hw_ndim) return NULL;
  int64_t rcout = cout / groups;
  int oyx_ndim = hw_ndim;

  int xrw_ndim = 4 + 2 * hw_ndim;
  if (xrw_ndim > POLY_MAX_DIMS) return NULL;
  int64_t xr_shape[POLY_MAX_DIMS], xe_shape[POLY_MAX_DIMS], perm[POLY_MAX_DIMS];
  int pos = 0;
  xr_shape[pos++] = bs;
  xr_shape[pos++] = groups;
  xr_shape[pos++] = cin;
  xr_shape[pos++] = 1;
  for (int i = 0; i < oyx_ndim; i++)
    xr_shape[pos++] = pshape[2 + i];
  for (int i = 0; i < hw_ndim; i++)
    xr_shape[pos++] = w_shape[2 + i];
  PolyUOp *xr = poly_reshape(ctx, pooled, xr_shape, xrw_ndim);
  memcpy(xe_shape, xr_shape, sizeof(int64_t) * xrw_ndim);
  xe_shape[3] = rcout;
  /* Pinned _broadcast_to returns self when the requested shape is unchanged
   * (mixin/movement.py:116-128). This occurs for one output channel and must
   * not leave a high-level no-op EXPAND in the Tensor graph. */
  PolyUOp *xe = poly_shape_equal(xr_shape, xrw_ndim, xe_shape, xrw_ndim)
                    ? xr
                    : poly_expand(ctx, xr, xe_shape, xrw_ndim);

  pos = 0;
  perm[pos++] = 0;
  perm[pos++] = 1;
  perm[pos++] = 3;
  for (int i = 0; i < oyx_ndim; i++)
    perm[pos++] = 4 + i;
  perm[pos++] = 2;
  for (int i = 0; i < hw_ndim; i++)
    perm[pos++] = 4 + oyx_ndim + i;
  xe = poly_permute(ctx, xe, perm, xrw_ndim);

  int64_t wr_shape[POLY_MAX_DIMS];
  pos = 0;
  wr_shape[pos++] = 1;
  wr_shape[pos++] = groups;
  wr_shape[pos++] = rcout;
  for (int i = 0; i < oyx_ndim; i++)
    wr_shape[pos++] = 1;
  wr_shape[pos++] = cin;
  for (int i = 0; i < hw_ndim; i++)
    wr_shape[pos++] = w_shape[2 + i];
  PolyUOp *wr = poly_reshape(ctx, weight, wr_shape, xrw_ndim);

  PolyUOp *mul = poly_mul(ctx, xe, wr);
  int64_t sum_axes[POLY_MAX_DIMS];
  int n_sum = 1 + oyx_ndim;
  for (int i = 0; i < n_sum; i++)
    sum_axes[i] = xrw_ndim - 1 - i;
  PolyUOp *reduced = poly_reduce_axis(ctx, POLY_OP_ADD, mul, sum_axes, n_sum);
  int64_t out_shape[POLY_MAX_DIMS];
  int out_ndim = 2 + oyx_ndim;
  out_shape[0] = bs;
  out_shape[1] = cout;
  for (int i = 0; i < oyx_ndim; i++)
    out_shape[2 + i] = pshape[2 + i];
  PolyUOp *ret = poly_reshape(ctx, reduced, out_shape, out_ndim);
  if (bias) {
    int64_t bshape[POLY_MAX_DIMS];
    bshape[0] = 1;
    bshape[1] = cout;
    for (int i = 0; i < hw_ndim; i++)
      bshape[2 + i] = 1;
    PolyUOp *br = poly_reshape(ctx, bias, bshape, out_ndim);
    ret = poly_add(ctx, ret, br);
  }
  return ret;
}

PolyUOp *poly_batchnorm(
    PolyCtx *ctx,
    PolyUOp *x,
    PolyUOp *weight,
    PolyUOp *bias,
    PolyUOp *mean,
    PolyUOp *invstd,
    const int64_t *axes,
    int n_axes
) {
  if (!ctx || !x || !mean || !invstd) return NULL;
  int64_t shape[POLY_MAX_DIMS];
  int ndim = uop_shape(ctx, x, shape);
  if (ndim < 0 || n_axes <= 0 || n_axes > ndim) return NULL;
  bool keep[POLY_MAX_DIMS] = {false};
  for (int i = 0; i < n_axes; i++) {
    int ax = (int)axes[i];
    if (ax < 0) ax += ndim;
    if (ax < 0 || ax >= ndim) return NULL;
    keep[ax] = true;
  }
  int64_t rshape[POLY_MAX_DIMS];
  for (int i = 0; i < ndim; i++)
    rshape[i] = keep[i] ? shape[i] : 1;
  PolyUOp *m = poly_reshape(ctx, mean, rshape, ndim);
  PolyUOp *centered = poly_sub(ctx, x, m);
  if (weight) centered = poly_mul(ctx, centered, poly_reshape(ctx, weight, rshape, ndim));
  PolyUOp *inv =
      (poly_uop_ndim(ctx, invstd) == n_axes) ? poly_reshape(ctx, invstd, rshape, ndim) : invstd;
  PolyUOp *ret = poly_mul(ctx, centered, inv);
  if (bias) ret = poly_add(ctx, ret, poly_reshape(ctx, bias, rshape, ndim));
  return ret;
}

/* The Tensor boundary applies pinned composite programs independently to the
 * retained logical roots and exact ordered current occurrences. The physical
 * result is never reconstructed by substituting logical identities. */
static PolyTensor *tensor_composite_result(
    PolyCtx *ctx,
    PolyUOp *logical,
    PolyUOp *physical,
    PolyTensor **inputs,
    int n_inputs
) {
  if (!ctx || !logical || !physical || !inputs || n_inputs <= 0) return NULL;
  PolyDevice device = POLY_DEVICE_AUTO;
  bool requires_grad = false;
  bool requires_grad_set = false;
  for (int i = 0; i < n_inputs; i++) {
    PolyTensor *input = inputs[i];
    if (!input) continue;
    if (input->device != POLY_DEVICE_AUTO) {
      if (device != POLY_DEVICE_AUTO && input->device != device) return NULL;
      device = input->device;
    }
    requires_grad |= input->requires_grad;
    requires_grad_set |= input->requires_grad_set;
  }
  PolyTensor *out =
      poly_tensor_create_with_roots(ctx, logical, physical, POLY_TENSOR_VALUE, device);
  if (!out) return NULL;
  out->requires_grad = requires_grad;
  out->requires_grad_set = requires_grad_set;
  out->provenance = POLY_TENSOR_PROVENANCE_COMPUTED;
  return out;
}

PolyTensor *poly_tensor_pool(
    PolyCtx *ctx,
    PolyTensor *src,
    const int64_t *kernel,
    int n_kernel,
    const int64_t *stride,
    const int64_t *dilation
) {
  PolyUOp *current = tensor_current_uop(src);
  if (!ctx || !src || !src->uop_logical || !current) return NULL;
  return tensor_unary_result(
      ctx, src, poly_pool(ctx, src->uop_logical, kernel, n_kernel, stride, dilation),
      poly_pool(ctx, current, kernel, n_kernel, stride, dilation)
  );
}

PolyTensor *poly_tensor_max_pool2d(
    PolyCtx *ctx,
    PolyTensor *src,
    const int64_t *kernel,
    int n_kernel,
    const int64_t *stride,
    const int64_t *dilation,
    const int64_t *padding,
    int n_padding
) {
  PolyUOp *current = tensor_current_uop(src);
  if (!ctx || !src || !src->uop_logical || !current) return NULL;
  return tensor_unary_result(
      ctx, src,
      poly_max_pool2d(
          ctx, src->uop_logical, kernel, n_kernel, stride, dilation, padding, n_padding
      ),
      poly_max_pool2d(ctx, current, kernel, n_kernel, stride, dilation, padding, n_padding)
  );
}

PolyTensor *poly_tensor_conv2d(
    PolyCtx *ctx,
    PolyTensor *src,
    PolyTensor *weight,
    PolyTensor *bias,
    int groups,
    const int64_t *stride,
    const int64_t *dilation,
    const int64_t *padding,
    int n_padding
) {
  PolyUOp *src_current = tensor_current_uop(src);
  PolyUOp *weight_current = tensor_current_uop(weight);
  PolyUOp *bias_current = tensor_current_uop(bias);
  if (!ctx || !src || !weight || !src->uop_logical || !weight->uop_logical || !src_current ||
      !weight_current || (bias && (!bias->uop_logical || !bias_current)))
    return NULL;
  PolyUOp *logical = poly_conv2d(
      ctx, src->uop_logical, weight->uop_logical, bias ? bias->uop_logical : NULL, groups, stride,
      dilation, padding, n_padding
  );
  PolyUOp *physical = poly_conv2d(
      ctx, src_current, weight_current, bias_current, groups, stride, dilation, padding, n_padding
  );
  PolyTensor *inputs[3] = {src, weight, bias};
  return tensor_composite_result(ctx, logical, physical, inputs, bias ? 3 : 2);
}

PolyTensor *poly_tensor_batchnorm(
    PolyCtx *ctx,
    PolyTensor *src,
    PolyTensor *weight,
    PolyTensor *bias,
    PolyTensor *mean,
    PolyTensor *invstd,
    const int64_t *axes,
    int n_axes
) {
  PolyUOp *src_current = tensor_current_uop(src);
  PolyUOp *weight_current = tensor_current_uop(weight);
  PolyUOp *bias_current = tensor_current_uop(bias);
  PolyUOp *mean_current = tensor_current_uop(mean);
  PolyUOp *invstd_current = tensor_current_uop(invstd);
  if (!ctx || !src || !mean || !invstd || !src->uop_logical || !mean->uop_logical ||
      !invstd->uop_logical || !src_current || !mean_current || !invstd_current ||
      (weight && (!weight->uop_logical || !weight_current)) ||
      (bias && (!bias->uop_logical || !bias_current)))
    return NULL;
  PolyUOp *logical = poly_batchnorm(
      ctx, src->uop_logical, weight ? weight->uop_logical : NULL, bias ? bias->uop_logical : NULL,
      mean->uop_logical, invstd->uop_logical, axes, n_axes
  );
  PolyUOp *physical = poly_batchnorm(
      ctx, src_current, weight_current, bias_current, mean_current, invstd_current, axes, n_axes
  );
  PolyTensor *inputs[5] = {src, mean, invstd, weight, bias};
  return tensor_composite_result(ctx, logical, physical, inputs, 5);
}

static PolyUOp *poly_unsqueeze_axis(PolyCtx *ctx, PolyUOp *x, int axis);

/* Pinned `_one_hot_along_dim` compares the integer index directly with a
 * right-aligned arange (mixin/__init__.py:1086-1091). Keep this helper as the
 * shared graph spelling for one_hot, gather, and single-Tensor indexing. */
static PolyUOp *one_hot_along_dim(PolyCtx *ctx, PolyUOp *index, int64_t num_classes, int dim) {
  if (!ctx || !index || num_classes < 0 || !poly_dtype_is_int(poly_dtype_scalar(index->dtype)))
    return NULL;
  int64_t shape[POLY_MAX_DIMS];
  int ndim = uop_shape(ctx, index, shape);
  if (ndim <= 0) return NULL;
  if (dim < 0) dim += ndim;
  if (dim < 0 || dim >= ndim) return NULL;
  int offset = ndim - dim - 1;
  int dtype_id = num_classes > (int64_t)INT32_MAX ? poly_dtype_id_by_name("int64")
                                                  : poly_dtype_id_by_name("int32");
  PolyUOp *classes = poly_arange_int_by_id(ctx, 0, num_classes, 1, dtype_id);
  if (!classes) return NULL;
  int64_t class_shape[POLY_MAX_DIMS];
  class_shape[0] = num_classes;
  for (int i = 0; i < offset; i++)
    class_shape[i + 1] = 1;
  /* `_broadcast_to` returns self for the unchanged one-dimensional arange.
   * Avoid an extra no-op RESHAPE when offset is zero. */
  if (offset > 0) classes = poly_reshape(ctx, classes, class_shape, offset + 1);
  return classes ? poly_eq(ctx, index, classes) : NULL;
}

/* Pinned `.sum(axis, dtype=x.dtype)` does not apply default accumulator
 * promotion (mixin/__init__.py:180 and reduce.py:_reduce). */
static PolyUOp *sum_axis_keep_dtype(PolyCtx *ctx, PolyUOp *x, int axis) {
  if (!ctx || !x) return NULL;
  int64_t shape[POLY_MAX_DIMS], out_shape[POLY_MAX_DIMS];
  int ndim = uop_shape(ctx, x, shape);
  if (ndim <= 0) return NULL;
  if (axis < 0) axis += ndim;
  if (axis < 0 || axis >= ndim) return NULL;
  int64_t reduce_axis[1] = {axis};
  PolyUOp *reduced = poly_reduce_axis(ctx, POLY_OP_ADD, x, reduce_axis, 1);
  if (!reduced) return NULL;
  int out_ndim = 0;
  for (int i = 0; i < ndim; i++)
    if (i != axis) out_shape[out_ndim++] = shape[i];
  return poly_reshape(ctx, reduced, out_shape, out_ndim);
}

PolyUOp *poly_one_hot(PolyCtx *ctx, PolyUOp *x, int64_t num_classes) {
  if (!ctx || !x || num_classes < 0 || !poly_dtype_is_int(poly_dtype_scalar(x->dtype))) return NULL;
  PolyUOp *index = poly_unsqueeze_axis(ctx, x, -1);
  PolyUOp *mask = index ? one_hot_along_dim(ctx, index, num_classes, -1) : NULL;
  if (!mask) return NULL;
  /* Pinned one_hot is comparison.where(1, 0), not CAST(bool, int32)
   * (mixin/__init__.py:1093-1104). */
  return poly_where_op(
      ctx, mask, poly_const_exact_int(ctx, POLY_INT32, 1), poly_const_exact_int(ctx, POLY_INT32, 0)
  );
}

PolyTensor *poly_tensor_one_hot(PolyCtx *ctx, PolyTensor *x, int64_t num_classes) {
  PolyUOp *current = tensor_current_uop(x);
  if (!ctx || !x || !x->uop_logical || !current) return NULL;
  PolyUOp *logical = poly_one_hot(ctx, x->uop_logical, num_classes);
  PolyUOp *physical = poly_one_hot(ctx, current, num_classes);
  PolyTensor *inputs[1] = {x};
  return tensor_composite_result(ctx, logical, physical, inputs, 1);
}

/* Port the one-Tensor-index branch of pinned `_getitem` literally
 * (mixin/__init__.py:134-180). It retains index insertion order and reduces
 * the inserted source axis instead of expressing indexing through gather's
 * transpose/sum-last graph. */
PolyUOp *poly_index_select(PolyCtx *ctx, PolyUOp *x, int dim, PolyUOp *index) {
  if (!ctx || !x || !index || !poly_dtype_is_int(poly_dtype_scalar(index->dtype))) return NULL;
  int64_t shape[POLY_MAX_DIMS], index_shape[POLY_MAX_DIMS];
  int ndim = uop_shape(ctx, x, shape);
  int index_ndim = uop_shape(ctx, index, index_shape);
  if (ndim <= 0 || index_ndim <= 0 || ndim + index_ndim > POLY_MAX_DIMS) return NULL;
  if (dim < 0) dim += ndim;
  if (dim < 0 || dim >= ndim) return NULL;

  PolyDType index_dtype = poly_dtype_scalar(index->dtype);
  PolyUOp *zero = broadcast_scalar_like(ctx, poly_const_typed(ctx, index_dtype, 0.0), index);
  PolyUOp *size =
      broadcast_scalar_like(ctx, poly_const_typed(ctx, index_dtype, (double)shape[dim]), index);
  PolyUOp *negative = (zero && size) ? poly_alu2(ctx, POLY_OP_CMPLT, index, zero) : NULL;
  PolyUOp *plus_size = (zero && size) ? poly_alu2(ctx, POLY_OP_ADD, index, size) : NULL;
  PolyUOp *normalized =
      (negative && plus_size) ? poly_where_op(ctx, negative, plus_size, index) : NULL;
  if (!normalized) return NULL;

  int64_t pre_reduce_shape[POLY_MAX_DIMS];
  int pre_ndim = 0;
  for (int i = 0; i < dim; i++)
    pre_reduce_shape[pre_ndim++] = shape[i];
  for (int i = 0; i < index_ndim; i++)
    pre_reduce_shape[pre_ndim++] = index_shape[i];
  for (int i = dim; i < ndim; i++)
    pre_reduce_shape[pre_ndim++] = shape[i];

  int64_t reshaped_index_shape[POLY_MAX_DIMS];
  int reshaped_index_ndim = 0;
  for (int i = 0; i < index_ndim; i++)
    reshaped_index_shape[reshaped_index_ndim++] = index_shape[i];
  for (int i = 0; i < ndim - dim; i++)
    reshaped_index_shape[reshaped_index_ndim++] = 1;
  PolyUOp *expanded_index =
      poly_reshape(ctx, normalized, reshaped_index_shape, reshaped_index_ndim);
  expanded_index =
      expanded_index ? poly_expand(ctx, expanded_index, pre_reduce_shape, pre_ndim) : NULL;
  PolyUOp *mask =
      expanded_index ? one_hot_along_dim(ctx, expanded_index, shape[dim], dim - ndim) : NULL;
  if (!mask) return NULL;

  int64_t reshaped_x_shape[POLY_MAX_DIMS];
  int reshaped_x_ndim = 0;
  for (int i = 0; i < dim; i++)
    reshaped_x_shape[reshaped_x_ndim++] = shape[i];
  for (int i = 0; i < index_ndim; i++)
    reshaped_x_shape[reshaped_x_ndim++] = 1;
  for (int i = dim; i < ndim; i++)
    reshaped_x_shape[reshaped_x_ndim++] = shape[i];
  PolyUOp *reshaped_x = poly_reshape(ctx, x, reshaped_x_shape, reshaped_x_ndim);
  PolyUOp *selected = reshaped_x ? poly_where_op(
                                       ctx, mask, reshaped_x,
                                       poly_const_typed(ctx, poly_dtype_scalar(x->dtype), 0.0)
                                   )
                                 : NULL;
  return selected ? sum_axis_keep_dtype(ctx, selected, dim + index_ndim) : NULL;
}

PolyTensor *poly_tensor_index_select(PolyCtx *ctx, PolyTensor *x, int dim, PolyTensor *index) {
  PolyUOp *x_current = tensor_current_uop(x);
  PolyUOp *index_current = tensor_current_uop(index);
  if (!ctx || !x || !index || !x->uop_logical || !index->uop_logical || !x_current ||
      !index_current)
    return NULL;
  PolyUOp *logical = poly_index_select(ctx, x->uop_logical, dim, index->uop_logical);
  PolyUOp *physical = poly_index_select(ctx, x_current, dim, index_current);
  PolyTensor *inputs[2] = {x, index};
  return tensor_composite_result(ctx, logical, physical, inputs, 2);
}

/* Pad with arbitrary value / circular / reflect / replicate */

/* Tensor.cat -- tensor.py:1364
 *   dim_cumsum = accumulate([t.shape[dim] for t in tensors], initial=0)
 *   for i, t in enumerate(tensors):
 *     tensors[i] = t.pad([(dim_cumsum[i], dim_cumsum[-1]-dim_cumsum[i+1])
 *                          if j==dim else None for j in range(t.ndim)])
 *   return reduce(add, tensors) */
PolyUOp *poly_cat(PolyCtx *ctx, PolyUOp **tensors, int n_tensors, int dim) {
  if (!ctx || !tensors || n_tensors <= 0) return NULL;
  if (n_tensors == 1) return tensors[0];

  int64_t sh0[POLY_MAX_DIMS];
  int ndim = uop_shape(ctx, tensors[0], sh0);
  if (ndim < 0) return NULL;
  if (dim < 0) dim += ndim;
  if (dim < 0 || dim >= ndim) return NULL;

  /* All tensors must match shape except along dim */
  int64_t shi[POLY_MAX_DIMS];
  int64_t *cum = malloc((size_t)(n_tensors + 1) * sizeof(*cum));
  if (!cum) return NULL;
  cum[0] = 0;
  for (int i = 0; i < n_tensors; i++) {
    int ni = uop_shape(ctx, tensors[i], shi);
    if (ni != ndim) {
      fprintf(stderr, "poly_cat: ndim mismatch at tensor %d: %d vs %d\n", i, ni, ndim);
      free(cum);
      return NULL;
    }
    for (int j = 0; j < ndim; j++) {
      if (j == dim) continue;
      if (shi[j] != sh0[j]) {
        fprintf(
            stderr, "poly_cat: shape mismatch at tensor %d dim %d: %lld vs %lld\n", i, j,
            (long long)shi[j], (long long)sh0[j]
        );
        free(cum);
        return NULL;
      }
    }
    cum[i + 1] = cum[i] + shi[dim];
  }
  int64_t total = cum[n_tensors];

  PolyUOp *acc = NULL;
  for (int i = 0; i < n_tensors; i++) {
    int64_t pads[POLY_MAX_DIMS][2];
    for (int j = 0; j < ndim; j++) {
      pads[j][0] = 0;
      pads[j][1] = 0;
    }
    pads[dim][0] = cum[i];
    pads[dim][1] = total - cum[i + 1];
    PolyUOp *padded = poly_pad(ctx, tensors[i], pads, ndim);
    if (!padded) {
      free(cum);
      return NULL;
    }
    acc = (i == 0) ? padded : poly_alu2(ctx, POLY_OP_ADD, acc, padded);
    if (!acc) {
      free(cum);
      return NULL;
    }
  }
  free(cum);
  return acc;
}

/* Tensor._pad_constant -- tensor.py:1067 */
PolyUOp *poly_pad_value(PolyCtx *ctx, PolyUOp *x, int64_t (*pads)[2], int ndim, double value) {
  if (!ctx || !x || ndim < 0 || (ndim > 0 && !pads)) return NULL;
  /* MovementMixin.pad returns a scalar unchanged for an empty movement
   * argument (uop/ops.py:712-713). _pad_constant returns that base directly
   * only for zero fill (mixin/__init__.py:359-368). */
  if (ndim == 0 && value == 0.0) return x;

  int64_t sh[POLY_MAX_DIMS];
  int xnd = uop_shape(ctx, x, sh);
  if (xnd != ndim) {
    fprintf(stderr, "poly_pad_value: ndim mismatch %d vs %d\n", xnd, ndim);
    return NULL;
  }

  /* has_neg = not all(p >= 0) */
  bool has_neg = false;
  for (int i = 0; i < ndim; i++)
    if (pads[i][0] < 0 || pads[i][1] < 0) {
      has_neg = true;
      break;
    }

  PolyUOp *X = x;
  int64_t nn_pads[POLY_MAX_DIMS][2];
  if (has_neg) {
    /* Shrink first for the negative parts:
     * shrink_pair = (-min(pB,0), min(pA + s, s)) */
    int64_t shr[POLY_MAX_DIMS][2];
    for (int i = 0; i < ndim; i++) {
      int64_t pB = pads[i][0], pA = pads[i][1], s = sh[i];
      shr[i][0] = -(pB < 0 ? pB : 0); /* -min(pB, 0) */
      int64_t end = pA + s;
      shr[i][1] = end < s ? end : s; /*  min(pA + s, s) */
    }
    X = poly_shrink(ctx, X, shr, ndim);
    /* Then pad with only the non-negative parts */
    for (int i = 0; i < ndim; i++) {
      nn_pads[i][0] = pads[i][0] > 0 ? pads[i][0] : 0;
      nn_pads[i][1] = pads[i][1] > 0 ? pads[i][1] : 0;
    }
  } else {
    for (int i = 0; i < ndim; i++) {
      nn_pads[i][0] = pads[i][0];
      nn_pads[i][1] = pads[i][1];
    }
  }

  /* Fast path: zero pad */
  PolyUOp *padded_X = poly_pad(ctx, X, nn_pads, ndim);
  if (value == 0.0) return padded_X;

  /* Tinygrad _pad_constant:
   *   MovementMixin.pad(X.const_like(1).cast(bool), pads).where(base, value)
   * Keep the same graph shape; do not synthesize base + fill. */
  int64_t mask_shape[POLY_MAX_DIMS];
  int mask_ndim = uop_shape(ctx, X, mask_shape);
  if (mask_ndim < 0) return NULL;
  PolyDType dt = poly_dtype_scalar(x->dtype);
  PolyUOp *ones = poly_const_typed(ctx, dt, 1.0);
  if (mask_ndim > 0) {
    int64_t mask_ones[POLY_MAX_DIMS];
    for (int i = 0; i < mask_ndim; i++)
      mask_ones[i] = 1;
    ones = poly_expand(ctx, poly_reshape(ctx, ones, mask_ones, mask_ndim), mask_shape, mask_ndim);
  }
  ones = poly_uop1(ctx, POLY_OP_CAST, POLY_BOOL, ones, poly_arg_none());
  PolyUOp *padded_ones = poly_pad(ctx, ones, nn_pads, ndim);
  PolyUOp *value_c = poly_const_typed(ctx, dt, value);
  return poly_where_op(ctx, padded_ones, padded_X, value_c);
}

/* Tensor._pad_circular -- tensor.py:1075
 *   X = self.repeat(tuple(1 + bool(pB) + bool(pA) for pB,pA in pX))
 *   return X.shrink(tuple((0 if pB == 0 else osh-pB,
 *                          xsh if pA == 0 else xsh-osh+pA)
 *                          for (pB,pA),osh,xsh in zip(pX, orig_shape, X.shape))) */
PolyUOp *poly_pad_circular(PolyCtx *ctx, PolyUOp *x, int64_t (*pads)[2], int ndim) {
  if (!ctx || !x || !pads || ndim <= 0) return NULL;

  int64_t sh[POLY_MAX_DIMS];
  int xnd = uop_shape(ctx, x, sh);
  if (xnd != ndim) {
    fprintf(stderr, "poly_pad_circular: ndim mismatch %d vs %d\n", xnd, ndim);
    return NULL;
  }

  for (int i = 0; i < ndim; i++) {
    if (pads[i][0] < 0 || pads[i][1] < 0) {
      fprintf(stderr, "poly_pad_circular: negative pads not supported\n");
      return NULL;
    }
    if (pads[i][0] > sh[i] || pads[i][1] > sh[i]) {
      fprintf(
          stderr, "poly_pad_circular: pad %lld/%lld exceeds dim %lld (would wrap >1x)\n",
          (long long)pads[i][0], (long long)pads[i][1], (long long)sh[i]
      );
      return NULL;
    }
  }

  /* repeats = [1 + (pB!=0) + (pA!=0) for ...] */
  int64_t reps[POLY_MAX_DIMS];
  for (int i = 0; i < ndim; i++)
    reps[i] = 1 + (pads[i][0] != 0 ? 1 : 0) + (pads[i][1] != 0 ? 1 : 0);
  PolyUOp *X = poly_repeat(ctx, x, reps, ndim);
  if (!X) return NULL;

  /* Compute the new (post-repeat) shape so we can shrink correctly */
  int64_t xsh[POLY_MAX_DIMS];
  if (uop_shape(ctx, X, xsh) < 0) return NULL;

  int64_t shr[POLY_MAX_DIMS][2];
  for (int i = 0; i < ndim; i++) {
    int64_t pB = pads[i][0], pA = pads[i][1];
    int64_t osh = sh[i], xs = xsh[i];
    shr[i][0] = (pB == 0) ? 0 : (osh - pB);
    shr[i][1] = (pA == 0) ? xs : (xs - osh + pA);
  }
  return poly_shrink(ctx, X, shr, ndim);
}

/* Common impl for reflect/replicate. mode_reflect=true means "reflect"
 * (skip the boundary element), false means "replicate" (repeat the boundary).
 * tensor.py:1081. */
static PolyUOp *poly_pad_reflect_replicate(
    PolyCtx *ctx,
    PolyUOp *x,
    int64_t (*pads)[2],
    int ndim,
    bool mode_reflect
) {
  if (!ctx || !x || !pads || ndim <= 0) return NULL;
  int64_t sh[POLY_MAX_DIMS];
  int xnd = uop_shape(ctx, x, sh);
  if (xnd != ndim) {
    fprintf(
        stderr, "poly_pad_%s: ndim mismatch %d vs %d\n", mode_reflect ? "reflect" : "replicate",
        xnd, ndim
    );
    return NULL;
  }

  /* tinygrad: pads = ((max(pB,0), max(pA,0)) for (pB,pA) in pX) -- positive only first */
  int64_t pos_pads[POLY_MAX_DIMS][2];
  for (int i = 0; i < ndim; i++) {
    pos_pads[i][0] = pads[i][0] > 0 ? pads[i][0] : 0;
    pos_pads[i][1] = pads[i][1] > 0 ? pads[i][1] : 0;
  }

  PolyUOp *X = x;
  for (int d = 0; d < ndim; d++) {
    int64_t pB = pos_pads[d][0], pA = pos_pads[d][1];
    if (pB == 0 && pA == 0) continue;

    int64_t cur_sh[POLY_MAX_DIMS];
    int cur_nd = uop_shape(ctx, X, cur_sh);
    if (cur_nd < 0) return NULL;
    int64_t s = cur_sh[d];

    if (mode_reflect && (pB >= s || pA >= s)) {
      fprintf(
          stderr, "poly_pad_reflect: pad (%lld,%lld) >= dim size %lld at dim %d\n", (long long)pB,
          (long long)pA, (long long)s, d
      );
      return NULL;
    }

    PolyUOp *xB = NULL, *xA = NULL;
    if (mode_reflect) {
      /* slcB = slice(pB, 0, -1) -> indices [pB, pB-1, ..., 1]
       * That's elements at index 1..pB+1, then flipped. */
      if (pB > 0) {
        int64_t shr[POLY_MAX_DIMS][2];
        for (int j = 0; j < ndim; j++) {
          shr[j][0] = 0;
          shr[j][1] = cur_sh[j];
        }
        shr[d][0] = 1;
        shr[d][1] = pB + 1;
        PolyUOp *sl = poly_shrink(ctx, X, shr, ndim);
        int64_t flip_axes[1] = {d};
        xB = poly_flip(ctx, sl, flip_axes, 1);
      }
      /* slcA = slice(s-2, s-2-pA, -1) -> indices [s-2, s-3, ..., s-1-pA]
       * That's elements at index s-1-pA..s-1 (exclusive s-1), then flipped. */
      if (pA > 0) {
        int64_t shr[POLY_MAX_DIMS][2];
        for (int j = 0; j < ndim; j++) {
          shr[j][0] = 0;
          shr[j][1] = cur_sh[j];
        }
        shr[d][0] = s - 1 - pA;
        shr[d][1] = s - 1;
        PolyUOp *sl = poly_shrink(ctx, X, shr, ndim);
        int64_t flip_axes[1] = {d};
        xA = poly_flip(ctx, sl, flip_axes, 1);
      }
    } else {
      /* replicate: shrink to (0,1) and expand to (pB,) on dim d */
      if (pB > 0) {
        int64_t shr[POLY_MAX_DIMS][2];
        for (int j = 0; j < ndim; j++) {
          shr[j][0] = 0;
          shr[j][1] = cur_sh[j];
        }
        shr[d][0] = 0;
        shr[d][1] = 1;
        int64_t exp_sh[POLY_MAX_DIMS];
        for (int j = 0; j < ndim; j++)
          exp_sh[j] = cur_sh[j];
        exp_sh[d] = pB;
        xB = poly_expand(ctx, poly_shrink(ctx, X, shr, ndim), exp_sh, ndim);
      }
      if (pA > 0) {
        int64_t shr[POLY_MAX_DIMS][2];
        for (int j = 0; j < ndim; j++) {
          shr[j][0] = 0;
          shr[j][1] = cur_sh[j];
        }
        shr[d][0] = s - 1;
        shr[d][1] = s;
        int64_t exp_sh[POLY_MAX_DIMS];
        for (int j = 0; j < ndim; j++)
          exp_sh[j] = cur_sh[j];
        exp_sh[d] = pA;
        xA = poly_expand(ctx, poly_shrink(ctx, X, shr, ndim), exp_sh, ndim);
      }
    }

    /* cat([xB, X, xA] for those that exist) */
    PolyUOp *parts[3];
    int n_parts = 0;
    if (xB) parts[n_parts++] = xB;
    parts[n_parts++] = X;
    if (xA) parts[n_parts++] = xA;
    X = poly_cat(ctx, parts, n_parts, d);
    if (!X) return NULL;
  }

  /* shrink after for negative pads (reflect/replicate must see full data first):
   * shrink = ((-min(pB,0), min(pA+s, s)) for ((pB,pA), s) in zip(pX, X.shape)) */
  bool has_neg = false;
  for (int i = 0; i < ndim; i++)
    if (pads[i][0] < 0 || pads[i][1] < 0) {
      has_neg = true;
      break;
    }
  if (has_neg) {
    int64_t cur_sh[POLY_MAX_DIMS];
    if (uop_shape(ctx, X, cur_sh) < 0) return NULL;
    int64_t shr[POLY_MAX_DIMS][2];
    for (int i = 0; i < ndim; i++) {
      int64_t pB = pads[i][0], pA = pads[i][1], s = cur_sh[i];
      shr[i][0] = -(pB < 0 ? pB : 0);
      int64_t end = pA + s;
      shr[i][1] = end < s ? end : s;
    }
    X = poly_shrink(ctx, X, shr, ndim);
  }
  return X;
}

PolyUOp *poly_pad_reflect(PolyCtx *ctx, PolyUOp *x, int64_t (*pads)[2], int ndim) {
  return poly_pad_reflect_replicate(ctx, x, pads, ndim, true);
}

PolyUOp *poly_pad_replicate(PolyCtx *ctx, PolyUOp *x, int64_t (*pads)[2], int ndim) {
  return poly_pad_reflect_replicate(ctx, x, pads, ndim, false);
}

/* Tensor._cumalu -- tensor.py:2048
 *   pl_sz = shape[axis] - int(not _include_initial)
 *   pooled = transpose(axis,-1).pad((pl_sz, -int(_include_initial)),
 *                                    value=identity_element(op,dtype))._pool((shape[axis],))
 *   return pooled.sum(-1).transpose(axis,-1)
 *
 * Supports ADD/MAX/MUL via poly_pad_value with the operator's identity element. */
PolyUOp *poly_cumalu(PolyCtx *ctx, PolyUOp *x, int axis, PolyOps op, bool include_initial) {
  if (op != POLY_OP_ADD && op != POLY_OP_MAX && op != POLY_OP_MUL) {
    fprintf(stderr, "poly_cumalu: op must be ADD, MAX, or MUL\n");
    return NULL;
  }

  int64_t sh[POLY_MAX_DIMS];
  int ndim = uop_shape(ctx, x, sh);
  if (ndim < 0) return NULL;
  if (axis < 0) axis += ndim;
  if (axis < 0 || axis >= ndim || sh[axis] == 0) return NULL;

  int64_t len = sh[axis];
  int64_t pl_sz = len - (include_initial ? 0 : 1);

  /* identity element per op */
  PolyDType dt = poly_dtype_scalar(x->dtype);
  double identity;
  if (op == POLY_OP_ADD)
    identity = 0.0;
  else if (op == POLY_OP_MUL)
    identity = 1.0;
  else /* MAX */ {
    if (poly_dtype_is_float(dt))
      identity = -INFINITY;
    else
      identity = (double)INT64_MIN;
  }

  /* transpose(axis, -1): swap axis with last */
  int64_t perm[POLY_MAX_DIMS];
  for (int i = 0; i < ndim; i++)
    perm[i] = i;
  perm[axis] = ndim - 1;
  perm[ndim - 1] = axis;
  bool need_transpose = (axis != ndim - 1);
  PolyUOp *r = need_transpose ? poly_permute(ctx, x, perm, ndim) : x;

  /* pad((pl_sz, -int(include_initial))) on the LAST axis only, value=identity. */
  int64_t pads[POLY_MAX_DIMS][2];
  for (int i = 0; i < ndim; i++) {
    pads[i][0] = 0;
    pads[i][1] = 0;
  }
  pads[ndim - 1][0] = pl_sz;
  pads[ndim - 1][1] = include_initial ? -1 : 0;
  r = poly_pad_value(ctx, r, pads, ndim, identity);

  /* _pool((len,)) on last axis. After _pool, last two dims are (windows, kernel). */
  int64_t k_arr[1] = {len};
  r = poly_pool(ctx, r, k_arr, 1, NULL, NULL);

  /* Reduce on last (kernel) axis with the requested op */
  int64_t axes[1] = {-1};
  /* poly_reduce_axis takes the actual op; output keeps dims, we reshape away the last */
  int64_t out_shape[POLY_MAX_DIMS];
  int out_ndim = 0;
  int64_t r_sh[POLY_MAX_DIMS];
  int r_nd = uop_shape(ctx, r, r_sh);
  if (r_nd < 0) return NULL;
  axes[0] = r_nd - 1; /* concrete axis */
  r = poly_reduce_axis(ctx, op, r, axes, 1);
  /* Drop the reduced axis */
  for (int i = 0; i < r_nd - 1; i++)
    out_shape[out_ndim++] = r_sh[i];
  if (out_ndim == 0) {
    out_shape[0] = 1;
    out_ndim = 1;
  }
  r = poly_reshape(ctx, r, out_shape, out_ndim);

  /* transpose(axis, -1): swap back */
  if (need_transpose) r = poly_permute(ctx, r, perm, ndim);
  return r;
}

/* Creation */

/* Pinned tinygrad full(buffer=False) -- mixin/__init__.py:55-77:
 *   const(fill_value).reshape((1,)*ndim).expand(shape)
 *
 * Pure UOp graph -- no const-registry, host malloc, or artificial identity.
 * Storage uniqueness belongs to the later clone when buffer=True. */
PolyUOp *poly_const_int_by_id(PolyCtx *ctx, int64_t value, int dtype_id) {
  PolyDType dt;
  if (!ffi_dtype_is_integer_like(dtype_id, &dt)) return NULL;
  return poly_const_exact_int(ctx, dt, value);
}

PolyUOp *poly_const_float_by_id(PolyCtx *ctx, double value, int dtype_id) {
  PolyDType dt;
  if (!ffi_dtype_is_float_like(dtype_id, &dt)) return NULL;
  return poly_const_exact_float(ctx, dt, value);
}

PolyUOp *poly_full_int_by_id(
    PolyCtx *ctx,
    const int64_t *shape,
    int ndim,
    int64_t fill_value,
    int dtype_id
) {
  PolyDType dt;
  if (!ffi_dtype_is_integer_like(dtype_id, &dt)) return NULL;
  PolyUOp *scalar = poly_const_exact_int(ctx, dt, fill_value);
  return poly_full_from_scalar(ctx, shape, ndim, scalar);
}

PolyUOp *poly_full_float_by_id(
    PolyCtx *ctx,
    const int64_t *shape,
    int ndim,
    double fill_value,
    int dtype_id
) {
  PolyDType dt;
  if (!ffi_dtype_is_float_like(dtype_id, &dt)) return NULL;
  PolyUOp *scalar = poly_const_exact_float(ctx, dt, fill_value);
  return poly_full_from_scalar(ctx, shape, ndim, scalar);
}

PolyUOp *poly_full(PolyCtx *ctx, const int64_t *shape, int ndim, double fill_value) {
  return poly_full_float_by_id(ctx, shape, ndim, fill_value, 12);
}

/* Pinned tinygrad arange -- mixin/__init__.py:245-273:
 *   Tensor.full((output_len,), step)._cumalu(0, Ops.ADD) + (start - step)
 *
 * Pure UOp graph. The cumulative sum is currently O(N^2) until the
 * range-collapse simplify pass lands in Phase D. */
PolyUOp *poly_arange_int_by_id(
    PolyCtx *ctx,
    int64_t start,
    int64_t stop,
    int64_t step,
    int dtype_id
) {
  PolyDType dt;
  if (!ffi_dtype_is_integer_like(dtype_id, &dt) || poly_dtype_is_bool(dt)) return NULL;
  if (step == 0) {
    fprintf(stderr, "polygrad: arange: step must be non-zero\n");
    return NULL;
  }

  int64_t n = poly_arange_len((long double)start, (long double)stop, (long double)step);
  if (n < 0) return NULL;
  if (n == 0) {
    int64_t empty_shape[1] = {0};
    return poly_full_int_by_id(ctx, empty_shape, 1, 0, dtype_id);
  }

  int64_t shape[1] = {n};
  PolyUOp *base = poly_full_int_by_id(ctx, shape, 1, step, dtype_id);
  PolyUOp *cumsum = poly_cumalu(ctx, base, 0, POLY_OP_ADD, false);
  PolyUOp *bias = poly_const_exact_int(ctx, dt, start - step);
  return poly_add(ctx, cumsum, bias);
}

PolyUOp *poly_arange_float_by_id(
    PolyCtx *ctx,
    double start,
    double stop,
    double step,
    int dtype_id
) {
  PolyDType dt;
  if (!ffi_dtype_is_float_like(dtype_id, &dt)) return NULL;
  if (step == 0.0) {
    fprintf(stderr, "polygrad: arange: step must be non-zero\n");
    return NULL;
  }

  int64_t n = poly_arange_len((long double)start, (long double)stop, (long double)step);
  if (n < 0) return NULL;
  if (n == 0) {
    int64_t empty_shape[1] = {0};
    return poly_full_float_by_id(ctx, empty_shape, 1, 0.0, dtype_id);
  }

  int64_t shape[1] = {n};
  PolyUOp *base = poly_full_float_by_id(ctx, shape, 1, step, dtype_id);
  PolyUOp *cumsum = poly_cumalu(ctx, base, 0, POLY_OP_ADD, false);
  PolyUOp *bias = poly_const_exact_float(ctx, dt, start - step);
  return poly_add(ctx, cumsum, bias);
}

PolyUOp *poly_arange(PolyCtx *ctx, double start, double stop, double step) {
  return poly_arange_float_by_id(ctx, start, stop, step, 12);
}

/* Pinned tinygrad linspace -- mixin/__init__.py:275-289
 *   (start + Tensor.arange(steps) * ((stop - start) / (steps - 1))).cast(dtype) */
PolyUOp *poly_linspace_by_id(PolyCtx *ctx, double start, double stop, int64_t steps, int dtype_id) {
  const PolyDType *out_ptr = ffi_dtype_from_id(dtype_id);
  if (!out_ptr) return NULL;
  PolyDType out_dt = poly_dtype_scalar(*out_ptr);
  if (poly_dtype_is_bool(out_dt) || poly_dtype_eq(out_dt, POLY_VOID)) {
    fprintf(stderr, "polygrad: linspace: bool/void dtypes are not supported\n");
    return NULL;
  }
  if (steps < 0) {
    fprintf(stderr, "polygrad: linspace: number of steps must be non-negative\n");
    return NULL;
  }
  if (steps == 0) return poly_buffer(ctx, out_dt, 0);

  PolyDType compute_dt = poly_dtype_eq(out_dt, POLY_FLOAT64) ? POLY_FLOAT64 : POLY_FLOAT32;
  int compute_id = poly_dtype_eq(compute_dt, POLY_FLOAT64) ? 13 : 12;

  if (steps == 1) {
    int64_t shape[1] = {1};
    PolyUOp *one = poly_full_float_by_id(ctx, shape, 1, start, compute_id);
    return poly_dtype_eq(out_dt, compute_dt) ? one : poly_cast(ctx, one, out_dt);
  }

  double scale = (stop - start) / (double)(steps - 1);
  PolyUOp *ar = poly_arange_float_by_id(ctx, 0.0, (double)steps, 1.0, compute_id);
  PolyUOp *s_c = poly_const_exact_float(ctx, compute_dt, scale);
  PolyUOp *start_c = poly_const_exact_float(ctx, compute_dt, start);
  PolyUOp *result = poly_add(ctx, start_c, poly_mul(ctx, ar, s_c));
  return poly_dtype_eq(out_dt, compute_dt) ? result : poly_cast(ctx, result, out_dt);
}

PolyUOp *poly_linspace(PolyCtx *ctx, double start, double stop, int64_t steps) {
  return poly_linspace_by_id(ctx, start, stop, steps, 12);
}

/* Pinned tinygrad eye -- mixin/__init__.py:292-309
 *   (arange(n).unsqueeze(-1) == arange(m)).cast(dtype) */
PolyUOp *poly_eye_by_id(PolyCtx *ctx, int64_t n, int64_t m, int dtype_id) {
  const PolyDType *out_ptr = ffi_dtype_from_id(dtype_id);
  if (!out_ptr) return NULL;
  PolyDType out_dt = poly_dtype_scalar(*out_ptr);
  if (poly_dtype_eq(out_dt, POLY_VOID) || n < 0 || m < 0) return NULL;
  if (n == 0 || m == 0) {
    int64_t shape[2] = {n, m};
    return poly_empty_shaped(ctx, out_dt, shape, 2);
  }

  PolyUOp *rows = poly_reshape(ctx, poly_arange_int_by_id(ctx, 0, n, 1, 6), (int64_t[]){n, 1}, 2);
  PolyUOp *cols = poly_reshape(ctx, poly_arange_int_by_id(ctx, 0, m, 1, 6), (int64_t[]){1, m}, 2);
  PolyUOp *eq_bool = poly_eq(ctx, rows, cols);
  return poly_dtype_is_bool(out_dt) ? eq_bool : poly_cast(ctx, eq_bool, out_dt);
}

PolyUOp *poly_eye(PolyCtx *ctx, int64_t n) {
  return poly_eye_by_id(ctx, n, n, 12);
}

/* tinygrad Tensor._tri -- mixin/__init__.py:310-311
 *   arange(r).unsqueeze(-1) + diagonal <= arange(c)
 *   Returns a bool mask of shape (r, c). */
static PolyUOp *poly_tri_mask(PolyCtx *ctx, int64_t r, int64_t c, int diagonal) {
  PolyUOp *rows = poly_reshape(ctx, poly_arange_int_by_id(ctx, 0, r, 1, 6), (int64_t[]){r, 1}, 2);
  PolyUOp *cols = poly_reshape(ctx, poly_arange_int_by_id(ctx, 0, c, 1, 6), (int64_t[]){1, c}, 2);
  PolyUOp *rows_shifted =
      (diagonal == 0) ? rows : poly_add(ctx, rows, poly_const_exact_int(ctx, POLY_INT32, diagonal));
  return poly_le(ctx, rows_shifted, cols);
}

/* tinygrad Tensor.tril -- tensor.py:2154
 *   _tri(rows, cols, diagonal+1).where(zeros_like(self), self) */
PolyUOp *poly_tril(PolyCtx *ctx, PolyUOp *x, int diagonal) {
  if (!x) return NULL;
  int64_t shape[POLY_MAX_DIMS];
  int ndim = uop_shape(ctx, x, shape);
  if (ndim < 2) return NULL;
  if (shape[ndim - 2] < 0 || shape[ndim - 1] < 0) return NULL;
  if (shape[ndim - 2] == 0 || shape[ndim - 1] == 0)
    return poly_empty_shaped(ctx, poly_dtype_scalar(x->dtype), shape, ndim);
  PolyUOp *mask = poly_tri_mask(ctx, shape[ndim - 2], shape[ndim - 1], diagonal + 1);
  PolyUOp *zero = poly_const_typed(ctx, poly_dtype_scalar(x->dtype), 0.0);
  return poly_where_op(ctx, mask, zero, x);
}

/* tinygrad Tensor.triu -- tensor.py:2131
 *   _tri(rows, cols, diagonal).where(self, zeros_like(self)) */
PolyUOp *poly_triu(PolyCtx *ctx, PolyUOp *x, int diagonal) {
  if (!x) return NULL;
  int64_t shape[POLY_MAX_DIMS];
  int ndim = uop_shape(ctx, x, shape);
  if (ndim < 2) return NULL;
  if (shape[ndim - 2] < 0 || shape[ndim - 1] < 0) return NULL;
  if (shape[ndim - 2] == 0 || shape[ndim - 1] == 0)
    return poly_empty_shaped(ctx, poly_dtype_scalar(x->dtype), shape, ndim);
  PolyUOp *mask = poly_tri_mask(ctx, shape[ndim - 2], shape[ndim - 1], diagonal);
  PolyUOp *zero = poly_const_typed(ctx, poly_dtype_scalar(x->dtype), 0.0);
  return poly_where_op(ctx, mask, x, zero);
}

static PolyUOp *poly_rand_compute(
    PolyCtx *ctx,
    const int64_t *shape,
    int ndim,
    uint64_t seed,
    PolyDType out_dt
) {
  if (ndim < 0 || ndim > POLY_MAX_DIMS) return NULL;
  if (ndim > 0 && !shape) return NULL;
  for (int i = 0; i < ndim; i++) {
    if (shape[i] < 0) return NULL;
    if (shape[i] == 0) return poly_empty_shaped(ctx, out_dt, shape, ndim);
  }

  int64_t numel = poly_shape_numel_checked(shape, ndim);
  if (numel < 0) return NULL;
  uint32_t key_lo = (uint32_t)(seed & 0xffffffffu);
  uint32_t key_hi = (uint32_t)(seed >> 32);
  uint32_t mixed_key = key_lo ^ ((key_hi << 16) | (key_hi >> 16));

  PolyUOp *counter_flat = poly_arange_int_by_id(ctx, 0, numel, 1, 7);
  if (!counter_flat) return NULL;
  PolyUOp *counter_t = reshape_logical_input(ctx, counter_flat, shape, ndim);
  PolyUOp *key_t = poly_const_exact_int(ctx, POLY_UINT32, (int64_t)mixed_key);

  /* Pinned tinygrad/mixin/rand.py:12-15 constructs THREEFRY on packed uint64
   * values, then extracts uint32 lanes. This C helper uses the low lane for
   * its existing stateless counter/key convention. */
  PolyUOp *counter_u64 =
      poly_uop1(ctx, POLY_OP_CAST, POLY_UINT64, counter_t, poly_arg_none());
  PolyUOp *key_u64 =
      poly_uop1(ctx, POLY_OP_CAST, POLY_UINT64, key_t, poly_arg_none());
  PolyUOp *bits_u64 =
      poly_uop2(ctx, POLY_OP_THREEFRY, POLY_UINT64, counter_u64, key_u64, poly_arg_none());
  PolyUOp *mask_u64 = poly_const_exact_int(ctx, POLY_UINT64, INT64_C(0xffffffff));
  PolyUOp *bits = poly_uop1(
      ctx, POLY_OP_CAST, POLY_UINT32,
      poly_uop2(ctx, POLY_OP_AND, POLY_UINT64, bits_u64, mask_u64, poly_arg_none()),
      poly_arg_none()
  );

  if (poly_dtype_eq(out_dt, POLY_FLOAT64)) {
    PolyUOp *key2_t = poly_const_exact_int(ctx, POLY_UINT32, (int64_t)(mixed_key ^ 0x9E3779B9u));
    PolyUOp *key2_u64 =
        poly_uop1(ctx, POLY_OP_CAST, POLY_UINT64, key2_t, poly_arg_none());
    PolyUOp *bits2_u64 = poly_uop2(
        ctx, POLY_OP_THREEFRY, POLY_UINT64, counter_u64, key2_u64, poly_arg_none()
    );
    PolyUOp *bits_lo = poly_uop1(
        ctx, POLY_OP_CAST, POLY_UINT32,
        poly_uop2(ctx, POLY_OP_AND, POLY_UINT64, bits2_u64, mask_u64, poly_arg_none()),
        poly_arg_none()
    );
    PolyUOp *hi27 =
        poly_alu2(ctx, POLY_OP_SHR, bits, poly_const_exact_int(ctx, POLY_UINT32, 5));
    PolyUOp *lo26 = poly_alu2(ctx, POLY_OP_SHR, bits_lo, poly_const_exact_int(ctx, POLY_UINT32, 6));
    PolyUOp *hi_f = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT64, hi27, poly_arg_none());
    PolyUOp *lo_f = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT64, lo26, poly_arg_none());
    PolyUOp *mant = poly_alu2(
        ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_MUL, hi_f, cdt(ctx, POLY_FLOAT64, 67108864.0)),
        lo_f
    );
    return poly_alu2(ctx, POLY_OP_MUL, mant, cdt(ctx, POLY_FLOAT64, 1.0 / 9007199254740992.0));
  }

  PolyUOp *hi24 = poly_alu2(ctx, POLY_OP_SHR, bits, poly_const_exact_int(ctx, POLY_UINT32, 8));
  PolyUOp *as_f = poly_uop1(ctx, POLY_OP_CAST, out_dt, hi24, poly_arg_none());
  return poly_alu2(ctx, POLY_OP_MUL, as_f, cdt(ctx, out_dt, 1.0 / 16777216.0));
}

PolyUOp *poly_rand_by_id(
    PolyCtx *ctx,
    const int64_t *shape,
    int ndim,
    uint64_t seed,
    int dtype_id
) {
  PolyDType out_dt;
  if (!ffi_dtype_is_float_like(dtype_id, &out_dt)) return NULL;
  PolyDType compute_dt = poly_dtype_eq(out_dt, POLY_FLOAT64) ? POLY_FLOAT64 : POLY_FLOAT32;
  PolyUOp *base = poly_rand_compute(ctx, shape, ndim, seed, compute_dt);
  if (!base) return NULL;
  return poly_dtype_eq(compute_dt, out_dt) ? base : poly_cast(ctx, base, out_dt);
}

PolyUOp *poly_rand(PolyCtx *ctx, const int64_t *shape, int ndim, uint64_t seed) {
  return poly_rand_by_id(ctx, shape, ndim, seed, 12);
}

PolyUOp *poly_randn_by_id(
    PolyCtx *ctx,
    const int64_t *shape,
    int ndim,
    uint64_t seed,
    int dtype_id
) {
  PolyDType out_dt;
  if (!ffi_dtype_is_float_like(dtype_id, &out_dt)) return NULL;
  PolyDType compute_dt = poly_dtype_eq(out_dt, POLY_FLOAT64) ? POLY_FLOAT64 : POLY_FLOAT32;

  PolyUOp *u1 = poly_rand_compute(ctx, shape, ndim, seed, compute_dt);
  PolyUOp *u2 = poly_rand_compute(ctx, shape, ndim, seed ^ 0x9E3779B97F4A7C15ull, compute_dt);
  if (!u1 || !u2) return NULL;
  PolyUOp *u1_safe = poly_maximum(ctx, u1, cdt(ctx, compute_dt, 1e-7));
  PolyUOp *r = poly_alu1(
      ctx, POLY_OP_SQRT,
      poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, compute_dt, -2.0), poly_log(ctx, u1_safe))
  );
  PolyUOp *theta = poly_alu2(ctx, POLY_OP_MUL, cdt(ctx, compute_dt, 2.0 * M_PI), u2);
  PolyUOp *result = poly_alu2(ctx, POLY_OP_MUL, r, poly_cos(ctx, theta));
  return poly_dtype_eq(compute_dt, out_dt) ? result : poly_cast(ctx, result, out_dt);
}

PolyUOp *poly_randn(PolyCtx *ctx, const int64_t *shape, int ndim, uint64_t seed) {
  return poly_randn_by_id(ctx, shape, ndim, seed, 12);
}

static PolyUOp *poly_transpose_last2(PolyCtx *ctx, PolyUOp *x);

static PolyDType poly_linalg_compute_dtype(PolyUOp *a, PolyUOp *b) {
  PolyDType adt = a ? poly_dtype_scalar(a->dtype) : POLY_FLOAT32;
  PolyDType bdt = b ? poly_dtype_scalar(b->dtype) : POLY_FLOAT32;
  return (poly_dtype_eq(adt, POLY_FLOAT64) || poly_dtype_eq(bdt, POLY_FLOAT64)) ? POLY_FLOAT64
                                                                                : POLY_FLOAT32;
}

static PolyUOp *poly_linalg_cast_compute(PolyCtx *ctx, PolyUOp *x, PolyDType compute_dt) {
  if (!x) return NULL;
  PolyDType dt = poly_dtype_scalar(x->dtype);
  if (poly_dtype_eq(dt, compute_dt)) return x;
  return poly_cast(ctx, x, compute_dt);
}

static PolyUOp *poly_linalg_full(
    PolyCtx *ctx,
    const int64_t *shape,
    int ndim,
    PolyDType dt,
    double value
) {
  return poly_full_from_scalar(ctx, shape, ndim, poly_const_typed(ctx, dt, value));
}

static PolyUOp *poly_linalg_eye_like(PolyCtx *ctx, const int64_t *shape, int ndim, PolyDType dt) {
  if (!ctx || !shape || ndim < 2 || shape[ndim - 2] != shape[ndim - 1]) return NULL;
  int64_t n = shape[ndim - 1];
  PolyUOp *rows = poly_reshape(ctx, poly_arange_int_by_id(ctx, 0, n, 1, 7), (int64_t[]){n, 1}, 2);
  PolyUOp *cols = poly_reshape(ctx, poly_arange_int_by_id(ctx, 0, n, 1, 7), (int64_t[]){1, n}, 2);
  PolyUOp *eye = poly_eq(ctx, rows, cols);
  if (!poly_dtype_eq(dt, POLY_BOOL)) eye = poly_cast(ctx, eye, dt);
  if (ndim == 2) return eye;

  int64_t view_shape[POLY_MAX_DIMS];
  for (int i = 0; i < ndim - 2; i++)
    view_shape[i] = 1;
  view_shape[ndim - 2] = n;
  view_shape[ndim - 1] = n;
  return poly_expand(ctx, poly_reshape(ctx, eye, view_shape, ndim), (int64_t *)shape, ndim);
}

static PolyUOp *poly_linalg_row_mask(PolyCtx *ctx, int ndim, int64_t n, int row_axis, int64_t row) {
  if (!ctx || ndim < 1 || ndim > POLY_MAX_DIMS || row_axis < 0 || row_axis >= ndim) return NULL;
  int64_t mask_shape[POLY_MAX_DIMS];
  for (int i = 0; i < ndim; i++)
    mask_shape[i] = 1;
  mask_shape[row_axis] = n;
  PolyUOp *idx = poly_arange_int_by_id(ctx, 0, n, 1, 6);
  if (!idx) return NULL;
  idx = poly_reshape(ctx, idx, mask_shape, ndim);
  return poly_eq(ctx, idx, poly_const_exact_int(ctx, POLY_INT32, row));
}

static PolyUOp *poly_linalg_col_mask(PolyCtx *ctx, int ndim, int64_t n, int col_axis, int64_t col) {
  return poly_linalg_row_mask(ctx, ndim, n, col_axis, col);
}

static PolyUOp *poly_linalg_slice_last2(
    PolyCtx *ctx,
    PolyUOp *x,
    int64_t row0,
    int64_t row1,
    int64_t col0,
    int64_t col1
);

static PolyUOp *poly_linalg_position_mask(
    PolyCtx *ctx,
    int ndim,
    int64_t n,
    int64_t row,
    int64_t col
) {
  PolyUOp *rm = poly_linalg_row_mask(ctx, ndim, n, ndim - 2, row);
  PolyUOp *cm = poly_linalg_col_mask(ctx, ndim, n, ndim - 1, col);
  if (!rm || !cm) return NULL;
  PolyUOp *rm_bc = rm, *cm_bc = cm;
  int64_t out_shape[POLY_MAX_DIMS];
  int out_ndim = 0;
  if (!poly_broadcast_pair(ctx, &rm_bc, &cm_bc, out_shape, &out_ndim)) return NULL;
  return poly_alu2(ctx, POLY_OP_AND, rm_bc, cm_bc);
}

static PolyUOp *poly_linalg_onehot_pivot_mask(PolyCtx *ctx, PolyUOp *u, int64_t k) {
  int64_t shape[POLY_MAX_DIMS];
  int ndim = uop_shape(ctx, u, shape);
  if (ndim < 2 || shape[ndim - 2] != shape[ndim - 1]) return NULL;
  int64_t n = shape[ndim - 1];
  if (k < 0 || k >= n) return NULL;

  PolyUOp *best_abs = poly_abs(ctx, poly_linalg_slice_last2(ctx, u, k, k + 1, k, k + 1));
  PolyUOp *pivot_mask = poly_linalg_row_mask(ctx, ndim, n, ndim - 2, k);
  if (!best_abs || !pivot_mask) return NULL;

  for (int64_t i = k + 1; i < n; i++) {
    PolyUOp *val_i = poly_abs(ctx, poly_linalg_slice_last2(ctx, u, i, i + 1, k, k + 1));
    PolyUOp *take_i = val_i ? poly_gt(ctx, val_i, best_abs) : NULL;
    PolyUOp *row_i = poly_linalg_row_mask(ctx, ndim, n, ndim - 2, i);
    pivot_mask = (take_i && row_i) ? poly_where_op(ctx, take_i, row_i, pivot_mask) : NULL;
    best_abs = take_i ? poly_where_op(ctx, take_i, val_i, best_abs) : NULL;
    if (!pivot_mask || !best_abs) return NULL;
  }
  return pivot_mask;
}

static PolyUOp *poly_linalg_slice_last2(
    PolyCtx *ctx,
    PolyUOp *x,
    int64_t row0,
    int64_t row1,
    int64_t col0,
    int64_t col1
) {
  int64_t shape[POLY_MAX_DIMS];
  int ndim = uop_shape(ctx, x, shape);
  if (ndim < 2) return NULL;
  if (row0 < 0 || row1 < row0 || row1 > shape[ndim - 2]) return NULL;
  if (col0 < 0 || col1 < col0 || col1 > shape[ndim - 1]) return NULL;
  int64_t pairs[POLY_MAX_DIMS][2];
  for (int i = 0; i < ndim; i++) {
    pairs[i][0] = 0;
    pairs[i][1] = shape[i];
  }
  pairs[ndim - 2][0] = row0;
  pairs[ndim - 2][1] = row1;
  pairs[ndim - 1][0] = col0;
  pairs[ndim - 1][1] = col1;
  return poly_shrink(ctx, x, pairs, ndim);
}

static bool poly_linalg_broadcast_batch_shape(
    const int64_t *a_shape,
    int a_ndim,
    const int64_t *b_shape,
    int b_ndim,
    int64_t *out_shape,
    int *out_ndim
) {
  if (!a_shape || !b_shape || !out_shape || !out_ndim) return false;
  if (a_ndim < 0 || b_ndim < 0 || a_ndim > POLY_MAX_DIMS || b_ndim > POLY_MAX_DIMS) return false;
  int nd = a_ndim > b_ndim ? a_ndim : b_ndim;
  if (nd > POLY_MAX_DIMS) return false;
  for (int i = 0; i < nd; i++) {
    int ai = i - (nd - a_ndim);
    int bi = i - (nd - b_ndim);
    int64_t ad = ai >= 0 ? a_shape[ai] : 1;
    int64_t bd = bi >= 0 ? b_shape[bi] : 1;
    if (ad != bd && ad != 1 && bd != 1) return false;
    out_shape[i] = ad > bd ? ad : bd;
  }
  *out_ndim = nd;
  return true;
}

static PolyUOp *poly_linalg_broadcast_last(
    PolyCtx *ctx,
    PolyUOp *x,
    const int64_t *target_batch_shape,
    int target_batch_ndim,
    const int64_t *tail_shape,
    int tail_ndim
) {
  if (!ctx || !x || !target_batch_shape || !tail_shape) return NULL;
  if (target_batch_ndim < 0 || tail_ndim < 0 || target_batch_ndim + tail_ndim > POLY_MAX_DIMS)
    return NULL;

  int64_t shape[POLY_MAX_DIMS];
  int ndim = uop_shape(ctx, x, shape);
  if (ndim < tail_ndim) return NULL;
  int batch_ndim = ndim - tail_ndim;
  if (batch_ndim > target_batch_ndim) return NULL;
  for (int i = 0; i < tail_ndim; i++)
    if (shape[batch_ndim + i] != tail_shape[i]) return NULL;

  int out_ndim = target_batch_ndim + tail_ndim;
  int64_t view_shape[POLY_MAX_DIMS], out_shape[POLY_MAX_DIMS];
  int pad = target_batch_ndim - batch_ndim;
  for (int i = 0; i < pad; i++)
    view_shape[i] = 1;
  for (int i = 0; i < batch_ndim; i++)
    view_shape[pad + i] = shape[i];
  for (int i = 0; i < tail_ndim; i++)
    view_shape[target_batch_ndim + i] = tail_shape[i];
  for (int i = 0; i < target_batch_ndim; i++)
    out_shape[i] = target_batch_shape[i];
  for (int i = 0; i < tail_ndim; i++)
    out_shape[target_batch_ndim + i] = tail_shape[i];

  PolyUOp *r = (pad > 0) ? poly_reshape(ctx, x, view_shape, out_ndim) : x;
  if (!r) return NULL;
  return poly_expand(ctx, r, out_shape, out_ndim);
}

static bool poly_linalg_prepare_system_inputs(
    PolyCtx *ctx,
    PolyUOp **a_io,
    PolyUOp **b_io,
    int64_t rhs_rows,
    int64_t solution_rows,
    bool *vector_rhs,
    int64_t *vector_out_shape,
    int *vector_out_ndim
) {
  if (!ctx || !a_io || !b_io || !*a_io || !*b_io || !vector_rhs || !vector_out_shape ||
      !vector_out_ndim)
    return false;

  int64_t a_shape[POLY_MAX_DIMS], b_shape[POLY_MAX_DIMS];
  int a_ndim = uop_shape(ctx, *a_io, a_shape);
  int b_ndim = uop_shape(ctx, *b_io, b_shape);
  if (a_ndim < 2 || b_ndim < 1 || a_ndim > POLY_MAX_DIMS || b_ndim > POLY_MAX_DIMS) return false;
  if (a_shape[a_ndim - 2] != rhs_rows || a_shape[a_ndim - 1] != solution_rows) return false;

  int a_batch_ndim = a_ndim - 2;
  bool can_vector = b_shape[b_ndim - 1] == rhs_rows;
  bool can_matrix = b_ndim >= 2 && b_shape[b_ndim - 2] == rhs_rows && b_shape[b_ndim - 1] > 0;
  bool as_matrix = can_matrix && b_ndim >= a_ndim;
  bool as_vector = can_vector && !as_matrix;
  if (!as_vector && !as_matrix) {
    if (can_matrix)
      as_matrix = true;
    else
      return false;
  }

  int b_batch_ndim = as_vector ? b_ndim - 1 : b_ndim - 2;
  int64_t batch_shape[POLY_MAX_DIMS];
  int batch_ndim = 0;
  if (!poly_linalg_broadcast_batch_shape(
          a_shape, a_batch_ndim, b_shape, b_batch_ndim, batch_shape, &batch_ndim
      ))
    return false;
  if (batch_ndim + 2 > POLY_MAX_DIMS) return false;

  int64_t a_tail[2] = {rhs_rows, solution_rows};
  PolyUOp *a_bc = poly_linalg_broadcast_last(ctx, *a_io, batch_shape, batch_ndim, a_tail, 2);
  if (!a_bc) return false;

  int64_t nrhs = as_vector ? 1 : b_shape[b_ndim - 1];
  int64_t b_tail_matrix[2] = {rhs_rows, nrhs};
  PolyUOp *b_bc = NULL;
  if (as_vector) {
    int64_t b_tail_vector[1] = {rhs_rows};
    b_bc = poly_linalg_broadcast_last(ctx, *b_io, batch_shape, batch_ndim, b_tail_vector, 1);
    if (!b_bc) return false;
    int64_t b_matrix_shape[POLY_MAX_DIMS];
    for (int i = 0; i < batch_ndim; i++)
      b_matrix_shape[i] = batch_shape[i];
    b_matrix_shape[batch_ndim] = rhs_rows;
    b_matrix_shape[batch_ndim + 1] = 1;
    b_bc = poly_reshape(ctx, b_bc, b_matrix_shape, batch_ndim + 2);
  } else {
    b_bc = poly_linalg_broadcast_last(ctx, *b_io, batch_shape, batch_ndim, b_tail_matrix, 2);
  }
  if (!b_bc) return false;

  *vector_rhs = as_vector;
  *vector_out_ndim = 0;
  if (as_vector) {
    for (int i = 0; i < batch_ndim; i++)
      vector_out_shape[i] = batch_shape[i];
    vector_out_shape[batch_ndim] = solution_rows;
    *vector_out_ndim = batch_ndim + 1;
  }
  *a_io = a_bc;
  *b_io = b_bc;
  return true;
}

static PolyUOp *poly_linalg_symmetrize(PolyCtx *ctx, PolyUOp *x, PolyDType dt) {
  PolyUOp *xt = poly_transpose_last2(ctx, x);
  if (!xt) return NULL;
  return poly_mul(ctx, poly_add(ctx, x, xt), poly_const_typed(ctx, dt, 0.5));
}

static PolyUOp *poly_linalg_jacobi_rotation(
    PolyCtx *ctx,
    PolyUOp *d,
    int64_t p,
    int64_t q,
    PolyDType dt
) {
  int64_t shape[POLY_MAX_DIMS];
  int ndim = uop_shape(ctx, d, shape);
  if (ndim < 2 || shape[ndim - 2] != shape[ndim - 1]) return NULL;
  int64_t n = shape[ndim - 1];

  PolyUOp *app = poly_linalg_slice_last2(ctx, d, p, p + 1, p, p + 1);
  PolyUOp *aqq = poly_linalg_slice_last2(ctx, d, q, q + 1, q, q + 1);
  PolyUOp *apq = poly_linalg_slice_last2(ctx, d, p, p + 1, q, q + 1);
  if (!app || !aqq || !apq) return NULL;

  PolyUOp *zero = poly_const_typed(ctx, dt, 0.0);
  PolyUOp *one = poly_const_typed(ctx, dt, 1.0);
  PolyUOp *neg_one = poly_const_typed(ctx, dt, -1.0);
  PolyUOp *two = poly_const_typed(ctx, dt, 2.0);
  PolyUOp *eps = poly_const_typed(ctx, dt, poly_dtype_eq(dt, POLY_FLOAT64) ? 1e-14 : 1e-7);
  PolyUOp *active = poly_gt(ctx, poly_abs(ctx, apq), eps);

  PolyUOp *theta = poly_div(ctx, poly_sub(ctx, aqq, app), poly_mul(ctx, two, apq));
  PolyUOp *sign_theta = poly_where_op(ctx, poly_ge(ctx, theta, zero), one, neg_one);
  PolyUOp *t = poly_div(
      ctx, sign_theta,
      poly_add(
          ctx, poly_abs(ctx, theta),
          poly_alu1(ctx, POLY_OP_SQRT, poly_add(ctx, poly_mul(ctx, theta, theta), one))
      )
  );
  t = poly_where_op(ctx, active, t, zero);
  PolyUOp *c =
      poly_div(ctx, one, poly_alu1(ctx, POLY_OP_SQRT, poly_add(ctx, one, poly_mul(ctx, t, t))));
  PolyUOp *s = poly_mul(ctx, t, c);
  c = poly_where_op(ctx, active, c, one);
  s = poly_where_op(ctx, active, s, zero);

  PolyUOp *g = poly_linalg_eye_like(ctx, shape, ndim, dt);
  PolyUOp *mask_pp = poly_linalg_position_mask(ctx, ndim, n, p, p);
  PolyUOp *mask_qq = poly_linalg_position_mask(ctx, ndim, n, q, q);
  PolyUOp *mask_pq = poly_linalg_position_mask(ctx, ndim, n, p, q);
  PolyUOp *mask_qp = poly_linalg_position_mask(ctx, ndim, n, q, p);
  if (!g || !mask_pp || !mask_qq || !mask_pq || !mask_qp) return NULL;
  g = poly_where_op(ctx, mask_pp, c, g);
  g = poly_where_op(ctx, mask_qq, c, g);
  g = poly_where_op(ctx, mask_pq, s, g);
  g = poly_where_op(ctx, mask_qp, poly_mul(ctx, neg_one, s), g);
  return g;
}

static int poly_linalg_eigh_jacobi(
    PolyCtx *ctx,
    PolyUOp *sym,
    PolyDType dt,
    PolyUOp **out_vecs,
    PolyUOp **out_diag_inv
) {
  if (!ctx || !sym || !out_vecs || !out_diag_inv) return -1;
  *out_vecs = NULL;
  *out_diag_inv = NULL;

  int64_t shape[POLY_MAX_DIMS];
  int ndim = uop_shape(ctx, sym, shape);
  if (ndim < 2 || ndim > POLY_MAX_DIMS || shape[ndim - 2] != shape[ndim - 1]) return -1;
  int64_t n = shape[ndim - 1];
  if (n <= 0) return -1;

  PolyUOp *d = poly_linalg_symmetrize(ctx, sym, dt);
  PolyUOp *v = poly_linalg_eye_like(ctx, shape, ndim, dt);
  if (!d || !v) return -1;

  int sweeps = n <= 2 ? 8 : 10;
  for (int sweep = 0; sweep < sweeps; sweep++) {
    for (int64_t p = 0; p < n; p++) {
      for (int64_t q = p + 1; q < n; q++) {
        PolyUOp *g = poly_linalg_jacobi_rotation(ctx, d, p, q, dt);
        if (!g) return -1;
        PolyUOp *gt = poly_transpose_last2(ctx, g);
        if (!gt) return -1;
        d = poly_dot(ctx, poly_dot(ctx, gt, d), g);
        v = poly_dot(ctx, v, g);
        if (!d || !v) return -1;
      }
    }
  }

  int64_t scalar_shape[POLY_MAX_DIMS];
  for (int i = 0; i < ndim - 2; i++)
    scalar_shape[i] = shape[i];
  scalar_shape[ndim - 2] = 1;
  scalar_shape[ndim - 1] = 1;
  PolyUOp *zero_scalar = poly_linalg_full(ctx, scalar_shape, ndim, dt, 0.0);
  if (!zero_scalar) return -1;
  PolyUOp *max_abs = zero_scalar;
  for (int64_t i = 0; i < n; i++) {
    PolyUOp *eig = poly_linalg_slice_last2(ctx, d, i, i + 1, i, i + 1);
    if (!eig) return -1;
    max_abs = poly_maximum(ctx, max_abs, poly_abs(ctx, eig));
    if (!max_abs) return -1;
  }

  PolyUOp *rcond = poly_const_typed(ctx, dt, poly_dtype_eq(dt, POLY_FLOAT64) ? 1e-12 : 1e-5);
  PolyUOp *tol = poly_mul(ctx, max_abs, rcond);
  PolyUOp *diag_inv = poly_linalg_full(ctx, shape, ndim, dt, 0.0);
  PolyUOp *zero = poly_const_typed(ctx, dt, 0.0);
  PolyUOp *one = poly_const_typed(ctx, dt, 1.0);
  if (!tol || !diag_inv || !zero || !one) return -1;

  for (int64_t i = 0; i < n; i++) {
    PolyUOp *eig = poly_linalg_slice_last2(ctx, d, i, i + 1, i, i + 1);
    PolyUOp *keep = eig ? poly_gt(ctx, poly_abs(ctx, eig), tol) : NULL;
    PolyUOp *inv = keep ? poly_where_op(ctx, keep, poly_div(ctx, one, eig), zero) : NULL;
    PolyUOp *mask = poly_linalg_position_mask(ctx, ndim, n, i, i);
    PolyUOp *term = (mask && inv) ? poly_where_op(ctx, mask, inv, zero) : NULL;
    diag_inv = term ? poly_add(ctx, diag_inv, term) : NULL;
    if (!diag_inv) return -1;
  }

  *out_vecs = v;
  *out_diag_inv = diag_inv;
  return 0;
}

PolyUOp *poly_triangular_solve(
    PolyCtx *ctx,
    PolyUOp *a,
    PolyUOp *b,
    int upper,
    int transpose_a,
    int unit_diagonal
) {
  if (!ctx || !a || !b) return NULL;

  int64_t a_shape[POLY_MAX_DIMS], b_shape[POLY_MAX_DIMS], solve_shape[POLY_MAX_DIMS];
  int a_ndim = uop_shape(ctx, a, a_shape);
  int b_ndim = uop_shape(ctx, b, b_shape);
  if (a_ndim < 2 || b_ndim < 1 || a_ndim > POLY_MAX_DIMS || b_ndim > POLY_MAX_DIMS) return NULL;
  int64_t n = a_shape[a_ndim - 1];
  if (n <= 0 || a_shape[a_ndim - 2] != n) return NULL;

  bool vector_rhs = false;
  int64_t vector_out_shape[POLY_MAX_DIMS];
  int vector_out_ndim = 0;
  if (!poly_linalg_prepare_system_inputs(
          ctx, &a, &b, n, n, &vector_rhs, vector_out_shape, &vector_out_ndim
      ))
    return NULL;
  int solve_ndim = uop_shape(ctx, b, solve_shape);
  if (solve_ndim < 2) return NULL;

  PolyDType compute_dt = poly_linalg_compute_dtype(a, b);
  a = poly_linalg_cast_compute(ctx, a, compute_dt);
  b = poly_linalg_cast_compute(ctx, b, compute_dt);
  if (!a || !b) return NULL;

  if (transpose_a) {
    a = poly_transpose_last2(ctx, a);
    upper = !upper;
    if (!a) return NULL;
  }

  PolyUOp *x = poly_linalg_full(ctx, solve_shape, solve_ndim, compute_dt, 0.0);
  PolyUOp *zero = poly_const_typed(ctx, compute_dt, 0.0);
  if (!x || !zero) return NULL;

  int row_axis = solve_ndim - 2;
  for (int64_t step = 0; step < n; step++) {
    int64_t i = upper ? (n - 1 - step) : step;
    PolyUOp *row_mask = poly_linalg_row_mask(ctx, solve_ndim, n, row_axis, i);
    PolyUOp *b_row_full = poly_where_op(ctx, row_mask, b, zero);
    PolyUOp *b_i = poly_sum_reduce(ctx, b_row_full, row_axis, 1);
    PolyUOp *a_row = poly_linalg_slice_last2(ctx, a, i, i + 1, 0, n);
    PolyUOp *a_row_t = poly_transpose_last2(ctx, a_row);
    PolyUOp *known = poly_sum_reduce(ctx, poly_mul(ctx, a_row_t, x), row_axis, 1);
    PolyUOp *xi = poly_sub(ctx, b_i, known);
    if (!unit_diagonal) {
      PolyUOp *diag = poly_linalg_slice_last2(ctx, a, i, i + 1, i, i + 1);
      xi = poly_div(ctx, xi, diag);
    }
    x = poly_where_op(ctx, row_mask, xi, x);
    if (!x) return NULL;
  }

  if (vector_rhs) return poly_reshape(ctx, x, vector_out_shape, vector_out_ndim);
  return x;
}

PolyUOp *poly_cholesky(PolyCtx *ctx, PolyUOp *x, int upper) {
  if (!ctx || !x) return NULL;
  int64_t shape[POLY_MAX_DIMS];
  int ndim = uop_shape(ctx, x, shape);
  if (ndim < 2 || ndim > POLY_MAX_DIMS) return NULL;
  int64_t n = shape[ndim - 1];
  if (n <= 0 || shape[ndim - 2] != n) return NULL;

  PolyDType compute_dt = poly_linalg_compute_dtype(x, NULL);
  x = poly_linalg_cast_compute(ctx, x, compute_dt);
  if (!x) return NULL;

  int64_t scalar_shape[POLY_MAX_DIMS];
  for (int i = 0; i < ndim - 2; i++)
    scalar_shape[i] = shape[i];
  scalar_shape[ndim - 2] = 1;
  scalar_shape[ndim - 1] = 1;

  size_t n_entries = (size_t)n * (size_t)n;
  if (n > 0 && n_entries / (size_t)n != (size_t)n) return NULL;
  PolyUOp **entries = calloc(n_entries, sizeof(*entries));
  if (!entries) return NULL;

  PolyUOp *zero_scalar = poly_linalg_full(ctx, scalar_shape, ndim, compute_dt, 0.0);
  if (!zero_scalar) {
    free(entries);
    return NULL;
  }

  for (int64_t j = 0; j < n; j++) {
    for (int64_t i = j; i < n; i++) {
      PolyUOp *a_ij = poly_linalg_slice_last2(ctx, x, i, i + 1, j, j + 1);
      if (!a_ij) {
        free(entries);
        return NULL;
      }
      PolyUOp *sum = zero_scalar;
      for (int64_t k = 0; k < j; k++) {
        PolyUOp *lik = entries[(size_t)i * (size_t)n + (size_t)k];
        PolyUOp *ljk = entries[(size_t)j * (size_t)n + (size_t)k];
        if (!lik || !ljk) {
          free(entries);
          return NULL;
        }
        PolyUOp *prod = poly_mul(ctx, lik, ljk);
        sum = poly_add(ctx, sum, prod);
        if (!sum) {
          free(entries);
          return NULL;
        }
      }

      PolyUOp *value = poly_sub(ctx, a_ij, sum);
      if (i == j) {
        value = poly_alu1(ctx, POLY_OP_SQRT, value);
      } else {
        PolyUOp *diag = entries[(size_t)j * (size_t)n + (size_t)j];
        if (!diag) {
          free(entries);
          return NULL;
        }
        value = poly_div(ctx, value, diag);
      }
      if (!value) {
        free(entries);
        return NULL;
      }
      entries[(size_t)i * (size_t)n + (size_t)j] = value;
    }
  }

  PolyUOp *l = poly_linalg_full(ctx, shape, ndim, compute_dt, 0.0);
  if (!l) {
    free(entries);
    return NULL;
  }
  PolyUOp *zero = poly_const_typed(ctx, compute_dt, 0.0);
  if (!zero) {
    free(entries);
    return NULL;
  }
  for (int64_t i = 0; i < n; i++) {
    for (int64_t j = 0; j <= i; j++) {
      PolyUOp *value = entries[(size_t)i * (size_t)n + (size_t)j];
      if (!value) {
        free(entries);
        return NULL;
      }
      PolyUOp *mask = poly_linalg_position_mask(ctx, ndim, n, i, j);
      PolyUOp *term = poly_where_op(ctx, mask, value, zero);
      l = poly_add(ctx, l, term);
      if (!l) {
        free(entries);
        return NULL;
      }
    }
  }

  PolyUOp *out = upper ? poly_transpose_last2(ctx, l) : l;
  free(entries);
  return out;
}

PolyUOp *poly_cholesky_solve(PolyCtx *ctx, PolyUOp *chol, PolyUOp *b, int upper) {
  if (!ctx || !chol || !b) return NULL;
  if (upper) {
    PolyUOp *y = poly_triangular_solve(ctx, chol, b, 1, 1, 0);
    if (!y) return NULL;
    y = poly_contiguous(ctx, y);
    if (!y) return NULL;
    return poly_triangular_solve(ctx, chol, y, 1, 0, 0);
  }
  PolyUOp *y = poly_triangular_solve(ctx, chol, b, 0, 0, 0);
  if (!y) return NULL;
  y = poly_contiguous(ctx, y);
  if (!y) return NULL;
  return poly_triangular_solve(ctx, chol, y, 0, 1, 0);
}

/* Reductions */

/* Pinned ReduceMixin.sum casts to sum_acc_dtype, emits one REDUCE over the
 * normalized axis tuple, reshapes away reduced axes when keepdim is false,
 * then casts half/bfloat results back (mixin/reduce.py:13-23,
 * dtype.py:274-278). Keep this root-level spelling reusable by the raw
 * one-axis API and the Tensor-handle boundary. */
static PolyUOp *sum_axes_root(PolyCtx *ctx, PolyUOp *x, int64_t *axes, int n_axes, bool keepdim) {
  if (!ctx || !x || n_axes < 0 || n_axes > POLY_MAX_DIMS || (n_axes > 0 && !axes)) return NULL;
  int64_t shape[POLY_MAX_DIMS], normalized[POLY_MAX_DIMS], out_shape[POLY_MAX_DIMS];
  int ndim = uop_shape(ctx, x, shape);
  if (ndim < 0) return NULL;
  for (int i = 0; i < n_axes; i++) {
    int64_t axis = axes[i] < 0 ? axes[i] + ndim : axes[i];
    if (axis < 0 || axis >= ndim) return NULL;
    normalized[i] = axis;
  }

  PolyDType input_dt = poly_dtype_scalar(x->dtype);
  PolyDType acc_dt;
  PolyDType floor_dt = poly_dtype_is_unsigned(input_dt) ? POLY_UINT32
                       : (poly_dtype_is_int(input_dt) || poly_dtype_is_bool(input_dt))
                           ? POLY_INT32
                           : POLY_FLOAT32;
  if (!poly_dtype_least_upper(input_dt, floor_dt, &acc_dt)) return NULL;
  PolyUOp *acc_x = poly_dtype_eq(input_dt, acc_dt) ? x : poly_cast(ctx, x, acc_dt);
  if (!acc_x) return NULL;

  PolyUOp *reduced = poly_reduce_axis(ctx, POLY_OP_ADD, acc_x, normalized, n_axes);
  if (!reduced) return NULL;
  if (!keepdim && n_axes > 0) {
    int out_ndim = 0;
    for (int i = 0; i < ndim; i++) {
      bool reduced_axis = false;
      for (int j = 0; j < n_axes; j++)
        if (normalized[j] == i) {
          reduced_axis = true;
          break;
        }
      if (!reduced_axis) out_shape[out_ndim++] = shape[i];
    }
    reduced = poly_reshape(ctx, reduced, out_shape, out_ndim);
  }
  if (reduced && poly_dtype_is_float(input_dt) && !poly_dtype_eq(input_dt, acc_dt))
    reduced = poly_cast(ctx, reduced, input_dt);
  return reduced;
}

/* Pinned ReduceMixin._reduce normalizes the full axis tuple once, emits one
 * UOp._rop REDUCE, and only then reshapes away reduced dimensions
 * (mixin/reduce.py:12-17, tensor.py:552, uop/ops.py:567-569). */
static PolyUOp *max_axes_root(PolyCtx *ctx, PolyUOp *x, int64_t *axes, int n_axes, bool keepdim) {
  if (!ctx || !x || n_axes < 0 || n_axes > POLY_MAX_DIMS || (n_axes > 0 && !axes)) return NULL;
  int64_t shape[POLY_MAX_DIMS], normalized[POLY_MAX_DIMS], out_shape[POLY_MAX_DIMS];
  int ndim = uop_shape(ctx, x, shape);
  if (ndim < 0) return NULL;
  for (int i = 0; i < n_axes; i++) {
    int64_t axis = axes[i] < 0 ? axes[i] + ndim : axes[i];
    if (axis < 0 || axis >= ndim) return NULL;
    normalized[i] = axis;
  }

  PolyUOp *reduced = poly_reduce_axis(ctx, POLY_OP_MAX, x, normalized, n_axes);
  if (!reduced) return NULL;
  if (!keepdim && n_axes > 0) {
    int out_ndim = 0;
    for (int i = 0; i < ndim; i++) {
      bool reduced_axis = false;
      for (int j = 0; j < n_axes; j++)
        if (normalized[j] == i) {
          reduced_axis = true;
          break;
        }
      if (!reduced_axis) out_shape[out_ndim++] = shape[i];
    }
    reduced = poly_reshape(ctx, reduced, out_shape, out_ndim);
  }
  return reduced;
}

PolyUOp *poly_sum_reduce(PolyCtx *ctx, PolyUOp *x, int axis, int keepdim) {
  int64_t axis64 = axis;
  return sum_axes_root(ctx, x, &axis64, 1, keepdim != 0);
}

PolyUOp *poly_max_reduce(PolyCtx *ctx, PolyUOp *x, int axis, int keepdim) {
  int64_t axis64 = axis;
  return max_axes_root(ctx, x, &axis64, 1, keepdim != 0);
}

PolyUOp *poly_mean_reduce(PolyCtx *ctx, PolyUOp *x, int axis, int keepdim) {
  int64_t shape[POLY_MAX_DIMS], out_shape[POLY_MAX_DIMS];
  int ndim, out_ndim;
  ndim = uop_shape(ctx, x, shape);
  if (ndim < 0) return NULL;
  if (axis < 0) axis += ndim;
  int64_t count = shape[axis];
  PolyUOp *s = do_reduce(ctx, POLY_OP_ADD, x, shape, ndim, axis, keepdim, out_shape, &out_ndim);
  return poly_alu2(ctx, POLY_OP_FDIV, s, cf(ctx, x, (double)count));
}

PolyUOp *poly_var_reduce(PolyCtx *ctx, PolyUOp *x, int axis, int keepdim, int correction) {
  int64_t shape[POLY_MAX_DIMS];
  int ndim;
  ndim = uop_shape(ctx, x, shape);
  if (ndim < 0) return NULL;
  if (axis < 0) axis += ndim;
  int64_t count = shape[axis];
  PolyUOp *x_view = reshape_logical_input(ctx, x, shape, ndim);
  if (!x_view) return NULL;
  /* var(x) = mean((x - mean(x))^2) * count / (count - correction) */
  /* First get mean with keepdim=1 for broadcast */
  PolyUOp *m = poly_mean_reduce(ctx, x_view, axis, 1);
  /* Expand mean back to full shape for subtraction */
  PolyUOp *m_expanded = poly_expand(ctx, m, (int64_t *)shape, ndim);
  /* (x - mean)^2 */
  PolyUOp *diff = poly_alu2(ctx, POLY_OP_SUB, x_view, m_expanded);
  PolyUOp *sq = poly_alu2(ctx, POLY_OP_MUL, diff, diff);
  /* sum of squares / (count - correction) */
  int64_t out_shape[POLY_MAX_DIMS];
  int out_ndim;
  PolyUOp *s = do_reduce(ctx, POLY_OP_ADD, sq, shape, ndim, axis, keepdim, out_shape, &out_ndim);
  double divisor = (double)(count - correction);
  if (divisor <= 0.0) divisor = 1.0;
  return poly_alu2(ctx, POLY_OP_FDIV, s, cf(ctx, x, divisor));
}

PolyUOp *poly_logsumexp(PolyCtx *ctx, PolyUOp *x, int axis, int keepdim) {
  int64_t shape[POLY_MAX_DIMS];
  int ndim;
  ndim = uop_shape(ctx, x, shape);
  if (ndim < 0) return NULL;
  if (axis < 0) axis += ndim;
  PolyUOp *x_view = reshape_logical_input(ctx, x, shape, ndim);
  if (!x_view) return NULL;
  int64_t keep_shape[POLY_MAX_DIMS];
  int keep_ndim = 0;
  PolyUOp *m = do_reduce(ctx, POLY_OP_MAX, x_view, shape, ndim, axis, 1, keep_shape, &keep_ndim);
  PolyUOp *shifted = poly_sub(ctx, x_view, m);
  PolyUOp *e = poly_exp(ctx, shifted);
  PolyUOp *s = do_reduce(ctx, POLY_OP_ADD, e, shape, ndim, axis, 1, keep_shape, &keep_ndim);
  PolyUOp *lse_keep = poly_add(ctx, poly_log(ctx, s), m);
  if (keepdim) return lse_keep;

  int64_t final_shape[POLY_MAX_DIMS];
  int fn = 0;
  for (int i = 0; i < ndim; i++) {
    if (i == axis) continue;
    final_shape[fn++] = shape[i];
  }
  if (fn == 0) return poly_reshape(ctx, lse_keep, NULL, 0);
  return poly_reshape(ctx, lse_keep, final_shape, fn);
}

/* Matmul */

PolyUOp *poly_dot(PolyCtx *ctx, PolyUOp *x, PolyUOp *w) {
  if (!ctx || !x || !w) return NULL;
  int64_t x_shape[POLY_MAX_DIMS], w_shape[POLY_MAX_DIMS];
  int x_ndim = uop_shape(ctx, x, x_shape);
  int w_ndim = uop_shape(ctx, w, w_shape);
  if (x_ndim < 0 || w_ndim < 0) return NULL;
  int64_t out_shape[POLY_MAX_DIMS];
  int out_ndim = 0;
  if (x_ndim < 1 || w_ndim < 1 || x_ndim > POLY_MAX_DIMS || w_ndim > POLY_MAX_DIMS) return NULL;

  int64_t K = x_shape[x_ndim - 1];
  int axis_w = w_ndim - (w_ndim >= 2 ? 2 : 1);
  if (K != w_shape[axis_w]) return NULL;

  int64_t xs[POLY_MAX_DIMS];
  int xn = 0;
  for (int i = 0; i < x_ndim - 1; i++)
    xs[xn++] = x_shape[i];
  int n_ones_x;
  {
    int a = x_ndim - 1, b = w_ndim - 1;
    n_ones_x = a < b ? a : b;
    if (n_ones_x > 1) n_ones_x = 1;
  }
  for (int i = 0; i < n_ones_x; i++)
    xs[xn++] = 1;
  xs[xn++] = K;
  PolyUOp *xr = poly_reshape(ctx, x, xs, xn);

  int64_t ws[POLY_MAX_DIMS];
  int wn = 0;
  for (int i = 0; i < w_ndim - 2; i++)
    ws[wn++] = w_shape[i];
  for (int i = 0; i < n_ones_x; i++)
    ws[wn++] = 1;
  for (int i = axis_w; i < w_ndim; i++)
    ws[wn++] = w_shape[i];
  PolyUOp *wr = poly_reshape(ctx, w, ws, wn);

  int new_axis_w = wn - 2;
  if (new_axis_w < 0) new_axis_w = 0;
  int64_t perm[POLY_MAX_DIMS];
  for (int i = 0; i < wn; i++)
    perm[i] = i;
  perm[wn - 1] = new_axis_w;
  perm[new_axis_w] = wn - 1;
  PolyUOp *wt = poly_permute(ctx, wr, perm, wn);
  int64_t wt_shape[POLY_MAX_DIMS];
  for (int i = 0; i < wn; i++)
    wt_shape[i] = ws[perm[i]];

  int max_ndim = xn > wn ? xn : wn;
  int64_t bc_shape[POLY_MAX_DIMS];
  for (int i = 0; i < max_ndim; i++) {
    int xi = i - (max_ndim - xn);
    int wi = i - (max_ndim - wn);
    int64_t xd = (xi >= 0) ? xs[xi] : 1;
    int64_t wd = (wi >= 0) ? wt_shape[wi] : 1;
    if (xd != wd && xd != 1 && wd != 1) return NULL;
    bc_shape[i] = xd > wd ? xd : wd;
  }

  PolyUOp *x_exp, *w_exp;
  if (xn < max_ndim) {
    int64_t padded[POLY_MAX_DIMS];
    int pad = max_ndim - xn;
    for (int i = 0; i < pad; i++)
      padded[i] = 1;
    for (int i = 0; i < xn; i++)
      padded[pad + i] = xs[i];
    x_exp = poly_reshape(ctx, xr, padded, max_ndim);
  } else {
    x_exp = xr;
  }
  if (wn < max_ndim) {
    int64_t padded[POLY_MAX_DIMS];
    int pad = max_ndim - wn;
    for (int i = 0; i < pad; i++)
      padded[i] = 1;
    for (int i = 0; i < wn; i++)
      padded[pad + i] = wt_shape[i];
    w_exp = poly_reshape(ctx, wt, padded, max_ndim);
  } else {
    w_exp = wt;
  }
  x_exp = poly_expand(ctx, x_exp, bc_shape, max_ndim);
  w_exp = poly_expand(ctx, w_exp, bc_shape, max_ndim);

  PolyUOp *mul = poly_alu2(ctx, POLY_OP_MUL, x_exp, w_exp);
  int64_t sum_axis[] = {max_ndim - 1};
  PolyUOp *summed = poly_reduce_axis(ctx, POLY_OP_ADD, mul, sum_axis, 1);

  int on = 0;
  for (int i = 0; i < max_ndim - 1; i++) {
    out_shape[on++] = bc_shape[i];
  }
  if (on == 0) {
    out_shape[0] = 1;
    on = 1;
  }
  out_ndim = on;

  return poly_reshape(ctx, summed, out_shape, on);
}

static PolyUOp *poly_transpose_last2(PolyCtx *ctx, PolyUOp *x) {
  int64_t shape[POLY_MAX_DIMS];
  int ndim = uop_shape(ctx, x, shape);
  if (ndim < 2) return NULL;
  int64_t perm[POLY_MAX_DIMS];
  for (int i = 0; i < ndim; i++)
    perm[i] = i;
  perm[ndim - 2] = ndim - 1;
  perm[ndim - 1] = ndim - 2;
  return poly_permute(ctx, x, perm, ndim);
}

static PolyUOp *poly_qr_column(PolyCtx *ctx, PolyUOp *r, int64_t col) {
  int64_t shape[POLY_MAX_DIMS];
  int ndim = uop_shape(ctx, r, shape);
  if (ndim < 2 || col < 0 || col >= shape[ndim - 1]) return NULL;
  int64_t pairs[POLY_MAX_DIMS][2];
  for (int i = 0; i < ndim; i++) {
    pairs[i][0] = 0;
    pairs[i][1] = shape[i];
  }
  pairs[ndim - 1][0] = col;
  pairs[ndim - 1][1] = col + 1;
  PolyUOp *s = poly_shrink(ctx, r, pairs, ndim);
  int64_t out_shape[POLY_MAX_DIMS];
  for (int i = 0; i < ndim - 1; i++)
    out_shape[i] = shape[i];
  return poly_reshape(ctx, s, out_shape, ndim - 1);
}

/* tinygrad Tensor.qr -- mixin/__init__.py:1703
 * Householder QR in tensor composition form.  For integer input tinygrad's
 * sqrt/div path promotes to float; Polygrad mirrors that by casting to f32. */
static int poly_qr_complete(PolyCtx *ctx, PolyUOp *x, PolyUOp **out_q, PolyUOp **out_r) {
  if (!ctx || !x || !out_q || !out_r) return -1;
  *out_q = NULL;
  *out_r = NULL;

  int64_t shape[POLY_MAX_DIMS];
  int ndim = uop_shape(ctx, x, shape);
  if (ndim < 2) return -1;
  int64_t m = shape[ndim - 2], n = shape[ndim - 1];
  if (m < 0 || n < 0) return -1;

  PolyDType dt = poly_dtype_scalar(x->dtype);
  if (!poly_dtype_is_float(dt)) {
    dt = POLY_FLOAT32;
    x = poly_cast(ctx, x, dt);
  }

  int64_t q_shape[POLY_MAX_DIMS];
  for (int i = 0; i < ndim - 2; i++)
    q_shape[i] = shape[i];
  q_shape[ndim - 2] = m;
  q_shape[ndim - 1] = m;

  PolyUOp *eye_bool = poly_eq(
      ctx, poly_reshape(ctx, poly_arange_int_by_id(ctx, 0, m, 1, 7), (int64_t[]){m, 1}, 2),
      poly_reshape(ctx, poly_arange_int_by_id(ctx, 0, m, 1, 7), (int64_t[]){1, m}, 2)
  );
  PolyUOp *q = poly_dtype_is_bool(dt) ? eye_bool : poly_cast(ctx, eye_bool, dt);
  if (ndim > 2) {
    int64_t q_view[POLY_MAX_DIMS];
    for (int i = 0; i < ndim - 2; i++)
      q_view[i] = 1;
    q_view[ndim - 2] = m;
    q_view[ndim - 1] = m;
    q = poly_expand(ctx, poly_reshape(ctx, q, q_view, ndim), q_shape, ndim);
  }

  PolyUOp *r = x;
  PolyUOp *idx = poly_arange_int_by_id(ctx, 0, m, 1, 7);
  int64_t steps = m < n ? m : n;
  PolyUOp *zero = poly_const_typed(ctx, dt, 0.0);
  PolyUOp *one = poly_const_typed(ctx, dt, 1.0);

  for (int64_t i = 0; i < steps; i++) {
    PolyUOp *i_c = poly_const_exact_int(ctx, POLY_INT32, i);
    PolyUOp *at_i = poly_eq(ctx, idx, i_c);
    PolyUOp *active_rows = poly_ge(ctx, idx, i_c);
    PolyUOp *col_i = poly_qr_column(ctx, r, i);
    if (!col_i) return -1;

    PolyUOp *x_vec = poly_where_op(ctx, active_rows, col_i, zero);
    PolyUOp *norm =
        poly_alu1(ctx, POLY_OP_SQRT, poly_sum_reduce(ctx, poly_square(ctx, x_vec), ndim - 2, 1));
    PolyUOp *x0 = poly_sum_reduce(ctx, poly_where_op(ctx, at_i, x_vec, zero), ndim - 2, 1);
    PolyUOp *active = poly_ne(ctx, norm, zero);
    PolyUOp *sgn = poly_where_op(ctx, poly_ne(ctx, x0, zero), poly_sign(ctx, x0), one);
    PolyUOp *u0 = poly_add(ctx, x0, poly_mul(ctx, sgn, norm));

    PolyUOp *safe_u0 = poly_where_op(ctx, active, u0, one);
    PolyUOp *v_num = poly_where_op(ctx, at_i, u0, x_vec);
    v_num = poly_contiguous(ctx, v_num);
    safe_u0 = poly_contiguous(ctx, safe_u0);
    if (!v_num || !safe_u0) return -1;
    PolyUOp *v_vec = poly_div(ctx, v_num, safe_u0);
    PolyUOp *v = poly_unsqueeze_axis(ctx, v_vec, -1);

    PolyUOp *safe_norm = poly_where_op(ctx, active, norm, one);
    PolyUOp *w_scale = poly_div(ctx, poly_mul(ctx, sgn, u0), safe_norm);
    PolyUOp *w =
        poly_mul(ctx, poly_unsqueeze_axis(ctx, poly_where_op(ctx, active, w_scale, zero), -1), v);

    PolyUOp *v_t = poly_transpose_last2(ctx, v);
    PolyUOp *w_t = poly_transpose_last2(ctx, w);
    if (!v_t || !w_t) return -1;
    r = poly_sub(ctx, r, poly_dot(ctx, w, poly_dot(ctx, v_t, r)));
    q = poly_sub(ctx, q, poly_dot(ctx, poly_dot(ctx, q, v), w_t));
    if (!r || !q) return -1;
  }

  *out_q = q;
  *out_r = r;
  return 0;
}

int poly_qr_ex(PolyCtx *ctx, PolyUOp *x, int mode, PolyUOp **out_q, PolyUOp **out_r) {
  if (!ctx || !x || !out_r) return -1;
  if (mode != POLY_QR_COMPLETE && mode != POLY_QR_REDUCED && mode != POLY_QR_R_ONLY) return -1;
  if (mode != POLY_QR_R_ONLY && !out_q) return -1;
  if (out_q) *out_q = NULL;
  *out_r = NULL;

  PolyUOp *q = NULL, *r = NULL;
  if (poly_qr_complete(ctx, x, &q, &r) != 0 || !q || !r) return -1;
  if (mode == POLY_QR_COMPLETE) {
    if (out_q) *out_q = q;
    *out_r = r;
    return 0;
  }

  int64_t shape[POLY_MAX_DIMS];
  int ndim = uop_shape(ctx, x, shape);
  if (ndim < 2) return -1;
  int64_t m = shape[ndim - 2], n = shape[ndim - 1];
  int64_t k = m < n ? m : n;
  PolyUOp *r_reduced = poly_linalg_slice_last2(ctx, r, 0, k, 0, n);
  if (!r_reduced) return -1;
  if (mode == POLY_QR_R_ONLY) {
    *out_r = r_reduced;
    return 0;
  }
  PolyUOp *q_reduced = poly_linalg_slice_last2(ctx, q, 0, m, 0, k);
  if (!q_reduced) return -1;
  *out_q = q_reduced;
  *out_r = r_reduced;
  return 0;
}

int poly_qr(PolyCtx *ctx, PolyUOp *x, PolyUOp **out_q, PolyUOp **out_r) {
  return poly_qr_ex(ctx, x, POLY_QR_COMPLETE, out_q, out_r);
}

static PolyUOp *poly_lu_solve_prepared(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  if (!ctx || !a || !b) return NULL;

  int64_t a_shape[POLY_MAX_DIMS], b_shape[POLY_MAX_DIMS];
  int a_ndim = uop_shape(ctx, a, a_shape);
  int b_ndim = uop_shape(ctx, b, b_shape);
  if (a_ndim < 2 || b_ndim != a_ndim || a_ndim > POLY_MAX_DIMS) return NULL;
  int64_t n = a_shape[a_ndim - 1];
  if (n <= 0 || a_shape[a_ndim - 2] != n) return NULL;
  if (b_shape[b_ndim - 2] != n || b_shape[b_ndim - 1] <= 0) return NULL;
  for (int i = 0; i < a_ndim - 2; i++)
    if (a_shape[i] != b_shape[i]) return NULL;

  PolyDType compute_dt = poly_linalg_compute_dtype(a, b);
  PolyUOp *u = poly_linalg_cast_compute(ctx, a, compute_dt);
  PolyUOp *rhs = poly_linalg_cast_compute(ctx, b, compute_dt);
  PolyUOp *zero = poly_const_typed(ctx, compute_dt, 0.0);
  if (!u || !rhs || !zero) return NULL;

  int row_axis = a_ndim - 2;
  int64_t nrhs = b_shape[b_ndim - 1];
  for (int64_t k = 0; k < n; k++) {
    PolyUOp *pivot_mask = poly_linalg_onehot_pivot_mask(ctx, u, k);
    PolyUOp *k_mask = poly_linalg_row_mask(ctx, a_ndim, n, row_axis, k);
    PolyUOp *u_k = poly_linalg_slice_last2(ctx, u, k, k + 1, 0, n);
    PolyUOp *rhs_k = poly_linalg_slice_last2(ctx, rhs, k, k + 1, 0, nrhs);
    PolyUOp *pivot_u =
        pivot_mask ? poly_sum_reduce(ctx, poly_where_op(ctx, pivot_mask, u, zero), row_axis, 1)
                   : NULL;
    PolyUOp *pivot_rhs =
        pivot_mask ? poly_sum_reduce(ctx, poly_where_op(ctx, pivot_mask, rhs, zero), row_axis, 1)
                   : NULL;
    if (!pivot_mask || !k_mask || !u_k || !rhs_k || !pivot_u || !pivot_rhs) return NULL;

    u = poly_where_op(ctx, k_mask, pivot_u, poly_where_op(ctx, pivot_mask, u_k, u));
    rhs = poly_where_op(ctx, k_mask, pivot_rhs, poly_where_op(ctx, pivot_mask, rhs_k, rhs));
    if (!u || !rhs) return NULL;

    u_k = poly_linalg_slice_last2(ctx, u, k, k + 1, 0, n);
    rhs_k = poly_linalg_slice_last2(ctx, rhs, k, k + 1, 0, nrhs);
    PolyUOp *pivot = poly_linalg_slice_last2(ctx, u, k, k + 1, k, k + 1);
    if (!u_k || !rhs_k || !pivot) return NULL;

    for (int64_t i = k + 1; i < n; i++) {
      PolyUOp *i_mask = poly_linalg_row_mask(ctx, a_ndim, n, row_axis, i);
      PolyUOp *u_i = poly_linalg_slice_last2(ctx, u, i, i + 1, 0, n);
      PolyUOp *rhs_i = poly_linalg_slice_last2(ctx, rhs, i, i + 1, 0, nrhs);
      PolyUOp *u_ik = poly_linalg_slice_last2(ctx, u, i, i + 1, k, k + 1);
      PolyUOp *factor = u_ik ? poly_div(ctx, u_ik, pivot) : NULL;
      PolyUOp *new_u_i = factor ? poly_sub(ctx, u_i, poly_mul(ctx, factor, u_k)) : NULL;
      PolyUOp *new_rhs_i = factor ? poly_sub(ctx, rhs_i, poly_mul(ctx, factor, rhs_k)) : NULL;
      if (!i_mask || !u_i || !rhs_i || !factor || !new_u_i || !new_rhs_i) return NULL;
      u = poly_where_op(ctx, i_mask, new_u_i, u);
      rhs = poly_where_op(ctx, i_mask, new_rhs_i, rhs);
      if (!u || !rhs) return NULL;
    }
  }

  u = poly_contiguous(ctx, u);
  rhs = poly_contiguous(ctx, rhs);
  if (!u || !rhs) return NULL;
  return poly_triangular_solve(ctx, u, rhs, 1, 0, 0);
}

PolyUOp *poly_solve(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  if (!ctx || !a || !b) return NULL;

  int64_t a_shape[POLY_MAX_DIMS];
  int a_ndim = uop_shape(ctx, a, a_shape);
  if (a_ndim < 2 || a_ndim > POLY_MAX_DIMS) return NULL;
  int64_t n = a_shape[a_ndim - 1];
  if (n <= 0 || a_shape[a_ndim - 2] != n) return NULL;

  bool vector_rhs = false;
  int64_t vector_out_shape[POLY_MAX_DIMS];
  int vector_out_ndim = 0;
  if (!poly_linalg_prepare_system_inputs(
          ctx, &a, &b, n, n, &vector_rhs, vector_out_shape, &vector_out_ndim
      ))
    return NULL;

  PolyUOp *x = poly_lu_solve_prepared(ctx, a, b);
  if (!x) return NULL;
  return vector_rhs ? poly_reshape(ctx, x, vector_out_shape, vector_out_ndim) : x;
}

PolyUOp *poly_lstsq(PolyCtx *ctx, PolyUOp *a, PolyUOp *b) {
  if (!ctx || !a || !b) return NULL;

  int64_t a_shape[POLY_MAX_DIMS];
  int a_ndim = uop_shape(ctx, a, a_shape);
  if (a_ndim < 2 || a_ndim > POLY_MAX_DIMS) return NULL;
  int64_t m = a_shape[a_ndim - 2];
  int64_t n = a_shape[a_ndim - 1];
  if (m <= 0 || n <= 0) return NULL;

  bool vector_rhs = false;
  int64_t vector_out_shape[POLY_MAX_DIMS];
  int vector_out_ndim = 0;
  if (!poly_linalg_prepare_system_inputs(
          ctx, &a, &b, m, n, &vector_rhs, vector_out_shape, &vector_out_ndim
      ))
    return NULL;

  PolyDType compute_dt = poly_linalg_compute_dtype(a, b);
  a = poly_linalg_cast_compute(ctx, a, compute_dt);
  b = poly_linalg_cast_compute(ctx, b, compute_dt);
  if (!a || !b) return NULL;

  PolyUOp *at = poly_transpose_last2(ctx, a);
  if (!at) return NULL;

  PolyUOp *x = NULL;
  if (m <= n) {
    PolyUOp *gram = poly_dot(ctx, a, at);
    PolyUOp *u = NULL, *diag_inv = NULL;
    if (!gram || poly_linalg_eigh_jacobi(ctx, gram, compute_dt, &u, &diag_inv) != 0) return NULL;
    PolyUOp *ut = poly_transpose_last2(ctx, u);
    PolyUOp *y = ut ? poly_dot(ctx, ut, b) : NULL;
    PolyUOp *z = y ? poly_dot(ctx, diag_inv, y) : NULL;
    PolyUOp *uz = z ? poly_dot(ctx, u, z) : NULL;
    x = uz ? poly_dot(ctx, at, uz) : NULL;
  } else {
    PolyUOp *gram = poly_dot(ctx, at, a);
    PolyUOp *v = NULL, *diag_inv = NULL;
    if (!gram || poly_linalg_eigh_jacobi(ctx, gram, compute_dt, &v, &diag_inv) != 0) return NULL;
    PolyUOp *vt = poly_transpose_last2(ctx, v);
    PolyUOp *c = poly_dot(ctx, at, b);
    PolyUOp *y = (vt && c) ? poly_dot(ctx, vt, c) : NULL;
    PolyUOp *z = y ? poly_dot(ctx, diag_inv, y) : NULL;
    x = z ? poly_dot(ctx, v, z) : NULL;
  }
  if (!x) return NULL;
  return vector_rhs ? poly_reshape(ctx, x, vector_out_shape, vector_out_ndim) : x;
}

int poly_tensor_qr_ex(
    PolyCtx *ctx,
    PolyTensor *src,
    int mode,
    PolyTensor **out_q,
    PolyTensor **out_r
) {
  if (!out_r || (mode != POLY_QR_R_ONLY && !out_q)) return -1;
  if (out_q) *out_q = NULL;
  *out_r = NULL;
  if (!tensor_roots_owned_by_ctx(ctx, src)) return -1;

  /* Pinned Tensor.qr delegates to the exact current Tensor.uop program
   * (mixin/__init__.py:1703-1719; test/null/test_tensor_uop_mixin.py:414-424).
   * The Tensor boundary applies the unchanged raw program independently to the retained
   * logical root and mandatory physical occurrence. */
  PolyUOp *logical_q = NULL, *logical_r = NULL;
  PolyUOp *physical_q = NULL, *physical_r = NULL;
  if (poly_qr_ex(ctx, src->uop_logical, mode, &logical_q, &logical_r) != 0 ||
      poly_qr_ex(ctx, src->uop_physical, mode, &physical_q, &physical_r) != 0)
    return -1;

  PolyTensor *r = tensor_unary_result(ctx, src, logical_r, physical_r);
  if (!r) return -1;
  if (mode == POLY_QR_R_ONLY) {
    *out_r = r;
    return 0;
  }
  PolyTensor *q = tensor_unary_result(ctx, src, logical_q, physical_q);
  if (!q) return -1;
  *out_q = q;
  *out_r = r;
  return 0;
}

PolyTensor *poly_tensor_triangular_solve(
    PolyCtx *ctx,
    PolyTensor *a,
    PolyTensor *b,
    int upper,
    int transpose_a,
    int unit_diagonal
) {
  if (!tensor_roots_owned_by_ctx(ctx, a) || !tensor_roots_owned_by_ctx(ctx, b)) return NULL;
  PolyUOp *logical = poly_triangular_solve(
      ctx, a->uop_logical, b->uop_logical, upper, transpose_a, unit_diagonal
  );
  PolyUOp *physical = poly_triangular_solve(
      ctx, a->uop_physical, b->uop_physical, upper, transpose_a, unit_diagonal
  );
  PolyTensor *inputs[2] = {a, b};
  return tensor_composite_result(ctx, logical, physical, inputs, 2);
}

PolyTensor *poly_tensor_cholesky(PolyCtx *ctx, PolyTensor *src, int upper) {
  if (!tensor_roots_owned_by_ctx(ctx, src)) return NULL;
  return tensor_unary_result(
      ctx, src, poly_cholesky(ctx, src->uop_logical, upper),
      poly_cholesky(ctx, src->uop_physical, upper)
  );
}

PolyTensor *poly_tensor_cholesky_solve(
    PolyCtx *ctx,
    PolyTensor *chol,
    PolyTensor *b,
    int upper
) {
  if (!tensor_roots_owned_by_ctx(ctx, chol) || !tensor_roots_owned_by_ctx(ctx, b)) return NULL;
  PolyUOp *logical =
      poly_cholesky_solve(ctx, chol->uop_logical, b->uop_logical, upper);
  PolyUOp *physical =
      poly_cholesky_solve(ctx, chol->uop_physical, b->uop_physical, upper);
  PolyTensor *inputs[2] = {chol, b};
  return tensor_composite_result(ctx, logical, physical, inputs, 2);
}

PolyTensor *poly_tensor_solve(PolyCtx *ctx, PolyTensor *a, PolyTensor *b) {
  if (!tensor_roots_owned_by_ctx(ctx, a) || !tensor_roots_owned_by_ctx(ctx, b)) return NULL;
  PolyTensor *inputs[2] = {a, b};
  return tensor_composite_result(
      ctx, poly_solve(ctx, a->uop_logical, b->uop_logical),
      poly_solve(ctx, a->uop_physical, b->uop_physical), inputs, 2
  );
}

PolyTensor *poly_tensor_lstsq(PolyCtx *ctx, PolyTensor *a, PolyTensor *b) {
  if (!tensor_roots_owned_by_ctx(ctx, a) || !tensor_roots_owned_by_ctx(ctx, b)) return NULL;
  PolyTensor *inputs[2] = {a, b};
  return tensor_composite_result(
      ctx, poly_lstsq(ctx, a->uop_logical, b->uop_logical),
      poly_lstsq(ctx, a->uop_physical, b->uop_physical), inputs, 2
  );
}

/* Softmax */

PolyUOp *poly_softmax(PolyCtx *ctx, PolyUOp *x, int axis) {
  if (!ctx || !x) return NULL;
  int64_t shape[POLY_MAX_DIMS];
  int ndim;
  ndim = uop_shape(ctx, x, shape);
  if (ndim < 0) return NULL;
  if (axis < 0) axis += ndim;
  if (axis < 0 || axis >= ndim) return NULL;

  int64_t max_shape[8];
  int max_ndim;
  /* Pinned _softmax detaches the keepdim MAX before subtraction
   * (mixin/__init__.py:743-747). */
  PolyUOp *m = do_reduce(ctx, POLY_OP_MAX, x, shape, ndim, axis, 1, max_shape, &max_ndim);
  m = m ? poly_detach(ctx, m) : NULL;
  PolyUOp *m_exp = m ? poly_expand(ctx, m, (int64_t *)shape, ndim) : NULL;
  PolyUOp *shifted = m_exp ? poly_sub(ctx, x, m_exp) : NULL;
  PolyUOp *e = poly_exp(ctx, shifted);
  if (!e) return NULL;

  PolyUOp *s = poly_sum_reduce(ctx, e, axis, 1);
  /* Pinned softmax is e * ss.reciprocal(), with ordinary broadcast
   * (mixin/__init__.py:749-770). */
  PolyUOp *reciprocal = s ? poly_alu1(ctx, POLY_OP_RECIPROCAL, s) : NULL;
  return reciprocal ? poly_mul(ctx, e, reciprocal) : NULL;
}

PolyUOp *poly_log_softmax(PolyCtx *ctx, PolyUOp *x, int axis) {
  if (!ctx || !x) return NULL;
  int64_t shape[POLY_MAX_DIMS];
  int ndim;
  ndim = uop_shape(ctx, x, shape);
  if (ndim < 0) return NULL;
  if (axis < 0) axis += ndim;
  if (axis < 0 || axis >= ndim) return NULL;

  int64_t max_shape[8];
  int max_ndim;
  PolyUOp *m = do_reduce(ctx, POLY_OP_MAX, x, shape, ndim, axis, 1, max_shape, &max_ndim);
  m = m ? poly_detach(ctx, m) : NULL;
  PolyUOp *m_exp = m ? poly_expand(ctx, m, (int64_t *)shape, ndim) : NULL;
  PolyUOp *shifted = m_exp ? poly_sub(ctx, x, m_exp) : NULL;
  PolyUOp *e = poly_exp(ctx, shifted);
  if (!e) return NULL;

  PolyUOp *s = poly_sum_reduce(ctx, e, axis, 1);
  PolyUOp *log_s = poly_log(ctx, s);
  /* Pinned log_softmax is m - ss.log(), with subtraction and broadcasting
   * supplied by the same high-level helpers (mixin/__init__.py:772-793). */
  return log_s ? poly_sub(ctx, shifted, log_s) : NULL;
}

/* The Tensor boundary applies composed Tensor methods independently to the retained logical
 * root and exact current physical occurrence. This is the same ordered
 * Tensor.uop construction as tinygrad's _apply_uop (tensor.py:128-140);
 * frontends receive a complete PolyTensor and perform no identity
 * substitution. */
PolyTensor *poly_tensor_exp(PolyCtx *ctx, PolyTensor *src) {
  PolyUOp *current = tensor_current_uop(src);
  if (!ctx || !src || !src->uop_logical || !current) return NULL;
  PolyUOp *logical = poly_exp(ctx, src->uop_logical);
  PolyUOp *physical = poly_exp(ctx, current);
  return tensor_unary_result(ctx, src, logical, physical);
}

PolyTensor *poly_tensor_log(PolyCtx *ctx, PolyTensor *src) {
  PolyUOp *current = tensor_current_uop(src);
  if (!ctx || !src || !src->uop_logical || !current) return NULL;
  PolyUOp *logical = poly_log(ctx, src->uop_logical);
  PolyUOp *physical = poly_log(ctx, current);
  return tensor_unary_result(ctx, src, logical, physical);
}

PolyTensor *poly_tensor_log1p(PolyCtx *ctx, PolyTensor *src) {
  if (!tensor_roots_owned_by_ctx(ctx, src)) return NULL;
  return tensor_unary_result(
      ctx, src, poly_log1p(ctx, src->uop_logical), poly_log1p(ctx, src->uop_physical)
  );
}

PolyTensor *poly_tensor_expm1(PolyCtx *ctx, PolyTensor *src) {
  if (!tensor_roots_owned_by_ctx(ctx, src)) return NULL;
  return tensor_unary_result(
      ctx, src, poly_expm1(ctx, src->uop_logical), poly_expm1(ctx, src->uop_physical)
  );
}

PolyTensor *poly_tensor_gelu(PolyCtx *ctx, PolyTensor *src) {
  if (!tensor_roots_owned_by_ctx(ctx, src)) return NULL;
  return tensor_unary_result(
      ctx, src, poly_gelu(ctx, src->uop_logical), poly_gelu(ctx, src->uop_physical)
  );
}

PolyTensor *poly_tensor_quick_gelu(PolyCtx *ctx, PolyTensor *src) {
  if (!tensor_roots_owned_by_ctx(ctx, src)) return NULL;
  return tensor_unary_result(
      ctx, src, poly_quick_gelu(ctx, src->uop_logical), poly_quick_gelu(ctx, src->uop_physical)
  );
}

PolyTensor *poly_tensor_detach(PolyCtx *ctx, PolyTensor *src) {
  PolyUOp *current = tensor_current_uop(src);
  if (!ctx || !src || !src->uop_logical || !current) return NULL;
  PolyTensor *out =
      tensor_unary_result(ctx, src, poly_detach(ctx, src->uop_logical), poly_detach(ctx, current));
  if (!out) return NULL;
  out->requires_grad = false;
  out->requires_grad_set = true;
  return out;
}

PolyTensor *poly_tensor_sum(
    PolyCtx *ctx,
    PolyTensor *src,
    int64_t *axes,
    int n_axes,
    bool keepdim
) {
  PolyUOp *current = tensor_current_uop(src);
  if (!ctx || !src || !src->uop_logical || !current) return NULL;
  PolyUOp *logical = sum_axes_root(ctx, src->uop_logical, axes, n_axes, keepdim);
  PolyUOp *physical = sum_axes_root(ctx, current, axes, n_axes, keepdim);
  return tensor_unary_result(ctx, src, logical, physical);
}

PolyTensor *poly_tensor_max(
    PolyCtx *ctx,
    PolyTensor *src,
    int64_t *axes,
    int n_axes,
    bool keepdim
) {
  PolyUOp *current = tensor_current_uop(src);
  if (!ctx || !src || !src->uop_logical || !current) return NULL;
  PolyUOp *logical = max_axes_root(ctx, src->uop_logical, axes, n_axes, keepdim);
  PolyUOp *physical = max_axes_root(ctx, current, axes, n_axes, keepdim);
  return tensor_unary_result(ctx, src, logical, physical);
}

PolyTensor *poly_tensor_minimum(PolyCtx *ctx, PolyTensor *a, PolyTensor *b) {
  PolyUOp *a_current = tensor_current_uop(a);
  PolyUOp *b_current = tensor_current_uop(b);
  if (!ctx || !a || !b || !a->uop_logical || !b->uop_logical || !a_current || !b_current)
    return NULL;
  /* Pinned Tensor._apply_uop consumes ordered current Tensor.uop operands and
   * minimum is inverse -> maximum -> inverse
   * (tensor.py:128-140; mixin/elementwise.py:366-393). The Tensor boundary builds that
   * program independently for each retained/executable root. */
  PolyUOp *logical = poly_minimum(ctx, a->uop_logical, b->uop_logical);
  PolyUOp *physical = poly_minimum(ctx, a_current, b_current);
  if (!logical || !physical) return NULL;
  PolyTensor *inputs[2] = {a, b};
  return tensor_composite_result(ctx, logical, physical, inputs, 2);
}

PolyTensor *poly_tensor_dot(PolyCtx *ctx, PolyTensor *src, PolyTensor *weight) {
  PolyUOp *current = tensor_current_uop(src);
  PolyUOp *weight_current = tensor_current_uop(weight);
  if (!ctx || !src || !weight || !src->uop_logical || !weight->uop_logical || !current ||
      !weight_current)
    return NULL;
  PolyUOp *logical = poly_dot(ctx, src->uop_logical, weight->uop_logical);
  PolyUOp *physical = poly_dot(ctx, current, weight_current);
  PolyTensor *inputs[2] = {src, weight};
  return tensor_composite_result(ctx, logical, physical, inputs, 2);
}

PolyTensor *poly_tensor_softmax(PolyCtx *ctx, PolyTensor *src, int axis) {
  PolyUOp *current = tensor_current_uop(src);
  if (!ctx || !src || !src->uop_logical || !current) return NULL;
  PolyUOp *logical = poly_softmax(ctx, src->uop_logical, axis);
  PolyUOp *physical = poly_softmax(ctx, current, axis);
  return tensor_unary_result(ctx, src, logical, physical);
}

PolyTensor *poly_tensor_log_softmax(PolyCtx *ctx, PolyTensor *src, int axis) {
  PolyUOp *current = tensor_current_uop(src);
  if (!ctx || !src || !src->uop_logical || !current) return NULL;
  PolyUOp *logical = poly_log_softmax(ctx, src->uop_logical, axis);
  PolyUOp *physical = poly_log_softmax(ctx, current, axis);
  return tensor_unary_result(ctx, src, logical, physical);
}

PolyUOp *poly_cross_entropy(PolyCtx *ctx, PolyUOp *logits, PolyUOp *target, int axis) {
  if (!ctx || !logits || !target) return NULL;
  int64_t logits_shape[POLY_MAX_DIMS], target_shape[POLY_MAX_DIMS];
  int logits_ndim = uop_shape(ctx, logits, logits_shape);
  int target_ndim = uop_shape(ctx, target, target_shape);
  if (logits_ndim < 1 || logits_ndim > POLY_MAX_DIMS || target_ndim < 0 ||
      target_ndim > POLY_MAX_DIMS)
    return NULL;

  if (axis < 0) axis += logits_ndim;
  if (axis < 0 || axis >= logits_ndim) return NULL;

  const bool dense_targets = poly_shape_equal(logits_shape, logits_ndim, target_shape, target_ndim);
  const bool sparse_targets =
      shape_equal_except_axis(logits_shape, logits_ndim, target_shape, target_ndim, axis);
  if (!dense_targets && !sparse_targets) return NULL;

  PolyUOp *weights =
      dense_targets ? poly_reshape(ctx, target, (int64_t *)target_shape, target_ndim) : target;
  if (sparse_targets) {
    const int64_t classes = logits_shape[axis];
    int arange_dtype_id = (classes > (int64_t)INT32_MAX) ? poly_dtype_id_by_name("int64")
                                                         : poly_dtype_id_by_name("int32");
    const PolyDType *class_dt_ptr = ffi_dtype_from_id(arange_dtype_id);
    if (!class_dt_ptr) return NULL;
    PolyDType class_dt = poly_dtype_scalar(*class_dt_ptr);
    PolyUOp *target_idx = poly_dtype_eq(poly_dtype_scalar(target->dtype), class_dt)
                              ? target
                              : poly_cast(ctx, target, class_dt);

    int64_t target_us_shape[POLY_MAX_DIMS];
    for (int i = 0; i < axis; i++)
      target_us_shape[i] = target_shape[i];
    target_us_shape[axis] = 1;
    for (int i = axis; i < target_ndim; i++)
      target_us_shape[i + 1] = target_shape[i];

    PolyUOp *target_us = poly_reshape(ctx, target_idx, target_us_shape, target_ndim + 1);
    PolyUOp *target_exp = poly_expand(ctx, target_us, (int64_t *)logits_shape, logits_ndim);

    PolyUOp *classes_uop = poly_arange_int_by_id(ctx, 0, classes, 1, arange_dtype_id);
    int64_t classes_shape[POLY_MAX_DIMS];
    for (int i = 0; i < logits_ndim; i++)
      classes_shape[i] = 1;
    classes_shape[axis] = classes;
    PolyUOp *classes_r = poly_reshape(ctx, classes_uop, classes_shape, logits_ndim);
    PolyUOp *classes_exp = poly_expand(ctx, classes_r, (int64_t *)logits_shape, logits_ndim);

    weights = poly_eq(ctx, target_exp, classes_exp);
  }

  PolyUOp *log_probs = poly_log_softmax(ctx, logits, axis);
  /* Cast bool weights to f32 before multiply (tinygrad does this via
   * Tensor._broadcasted dtype promotion; polygrad's C-level ALU doesn't
   * auto-promote, so explicit CAST is needed). */
  if (poly_dtype_is_bool(weights->dtype))
    weights = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, weights, poly_arg_none());
  PolyUOp *weighted = poly_alu2(ctx, POLY_OP_MUL, log_probs, weights);

  int64_t per_sample_shape[POLY_MAX_DIMS];
  int per_sample_ndim = 0;
  PolyUOp *per_sample = do_reduce(
      ctx, POLY_OP_ADD, weighted, logits_shape, logits_ndim, axis, 0, per_sample_shape,
      &per_sample_ndim
  );

  PolyUOp *total = per_sample;
  if (per_sample_ndim > 0) {
    int64_t axes[POLY_MAX_DIMS];
    for (int i = 0; i < per_sample_ndim; i++)
      axes[i] = i;
    total = poly_reduce_axis(ctx, POLY_OP_ADD, per_sample, axes, per_sample_ndim);
  }

  int64_t denom = poly_shape_numel_checked(per_sample_shape, per_sample_ndim);
  if (denom <= 0) return NULL;

  /* tinygrad Tensor.div is multiply by reciprocal, and cross_entropy applies
   * the leading negation outside the mean reduction. For a static denominator
   * this reaches linear IR as MUL(total, -1/denom), not FDIV followed by NEG. */
  PolyUOp *scale = cf(ctx, log_probs, -1.0 / (double)denom);
  PolyUOp *loss = poly_alu2(ctx, POLY_OP_MUL, total, scale);
  return poly_reshape(ctx, loss, NULL, 0);
}

static PolyUOp *poly_unsqueeze_axis(PolyCtx *ctx, PolyUOp *x, int axis) {
  int64_t shape[POLY_MAX_DIMS];
  int ndim = uop_shape(ctx, x, shape);
  if (ndim < 0 || ndim >= POLY_MAX_DIMS) return NULL;
  if (axis < 0) axis += ndim + 1;
  if (axis < 0 || axis > ndim) return NULL;
  int64_t out[POLY_MAX_DIMS];
  for (int i = 0, j = 0; i < ndim + 1; i++) {
    out[i] = (i == axis) ? 1 : shape[j++];
  }
  return poly_reshape(ctx, x, out, ndim + 1);
}

static PolyUOp *poly_flatten_axes(PolyCtx *ctx, PolyUOp *x, int start_dim, int end_dim) {
  int64_t shape[POLY_MAX_DIMS];
  int ndim = uop_shape(ctx, x, shape);
  if (ndim < 0) return NULL;
  if (start_dim < 0) start_dim += ndim;
  if (end_dim < 0) end_dim += ndim;
  if (start_dim < 0 || end_dim < start_dim || end_dim >= ndim) return NULL;
  int64_t out[POLY_MAX_DIMS];
  int on = 0;
  for (int i = 0; i < start_dim; i++)
    out[on++] = shape[i];
  int64_t prod = 1;
  for (int i = start_dim; i <= end_dim; i++) {
    if (shape[i] < 0 || prod > INT64_MAX / shape[i]) return NULL;
    prod *= shape[i];
  }
  out[on++] = prod;
  for (int i = end_dim + 1; i < ndim; i++)
    out[on++] = shape[i];
  return poly_reshape(ctx, x, out, on);
}

static bool poly_split_two_ones(PolyCtx *ctx, PolyUOp *x, int axis, PolyUOp **a, PolyUOp **b) {
  int64_t shape[POLY_MAX_DIMS];
  int ndim = uop_shape(ctx, x, shape);
  if (ndim < 0) return false;
  if (axis < 0) axis += ndim;
  if (axis < 0 || axis >= ndim || shape[axis] != 2) return false;
  int64_t pairs[POLY_MAX_DIMS][2];
  for (int i = 0; i < ndim; i++) {
    pairs[i][0] = 0;
    pairs[i][1] = shape[i];
  }
  pairs[axis][0] = 0;
  pairs[axis][1] = 1;
  *a = poly_shrink(ctx, x, pairs, ndim);
  pairs[axis][0] = 1;
  pairs[axis][1] = 2;
  *b = poly_shrink(ctx, x, pairs, ndim);
  return *a && *b;
}

static int poly_resolve_sort_flip_axes(int64_t *axes, int n_axes, int ndim) {
  if (!axes || n_axes < 0 || ndim < 0) return -1;
  for (int i = 0; i < n_axes; i++) {
    if (axes[i] < 0) axes[i] += ndim;
    if (axes[i] < 0 || axes[i] >= ndim) return -1;
    for (int j = 0; j < i; j++)
      if (axes[j] == axes[i]) return -1;
  }
  return 0;
}

static bool poly_sort_bound_const(PolyCtx *ctx, PolyDType dt, bool use_min, PolyUOp **out) {
  dt = poly_dtype_scalar(dt);
  if (poly_dtype_is_float(dt)) {
    *out = poly_const_exact_float(ctx, dt, use_min ? -INFINITY : INFINITY);
    return *out != NULL;
  }
  if (poly_dtype_is_bool(dt)) {
    *out = poly_const_exact_int(ctx, dt, use_min ? 0 : 1);
    return *out != NULL;
  }
  if (!poly_dtype_is_int(dt)) return false;

  int bits = poly_dtype_itemsize(dt) * 8;
  int64_t val;
  if (poly_dtype_is_unsigned(dt)) {
    if (use_min) {
      val = 0;
    } else {
      if (bits >= 63) return false;
      val = ((int64_t)1 << bits) - 1;
    }
  } else if (use_min) {
    val = (bits >= 64) ? INT64_MIN : -((int64_t)1 << (bits - 1));
  } else {
    val = (bits >= 64) ? INT64_MAX : (((int64_t)1 << (bits - 1)) - 1);
  }
  *out = poly_const_exact_int(ctx, dt, val);
  return *out != NULL;
}

static PolyUOp *poly_pad_with_scalar_nonnegative(
    PolyCtx *ctx,
    PolyUOp *x,
    int64_t (*pads)[2],
    int ndim,
    PolyUOp *value
) {
  if (!ctx || !x || !pads || !value) return NULL;
  int64_t shape[POLY_MAX_DIMS];
  int xndim = uop_shape(ctx, x, shape);
  if (xndim != ndim) return NULL;
  for (int i = 0; i < ndim; i++) {
    if (pads[i][0] < 0 || pads[i][1] < 0) return NULL;
  }

  PolyUOp *padded_x = poly_pad(ctx, x, pads, ndim);
  if (!padded_x) return NULL;

  int64_t ones_shape[POLY_MAX_DIMS];
  for (int i = 0; i < ndim; i++)
    ones_shape[i] = 1;
  PolyUOp *one = poly_const_typed(ctx, poly_dtype_scalar(x->dtype), 1.0);
  PolyUOp *mask = poly_expand(ctx, poly_reshape(ctx, one, ones_shape, ndim), shape, ndim);
  if (mask && !poly_dtype_is_bool(poly_dtype_scalar(mask->dtype)))
    mask = poly_cast(ctx, mask, POLY_BOOL);
  if (!mask) return NULL;
  PolyUOp *padded_mask = poly_pad(ctx, mask, pads, ndim);
  return poly_where_op(ctx, padded_mask, padded_x, value);
}

static PolyUOp *poly_sort_count_equal_before(PolyCtx *ctx, PolyUOp *mask, PolyUOp *t, int dim) {
  PolyUOp *lhs = poly_unsqueeze_axis(ctx, t, dim);
  PolyUOp *rhs = poly_unsqueeze_axis(ctx, t, dim + 1);
  if (!lhs || !rhs) return NULL;
  PolyUOp *eq = poly_eq(ctx, lhs, rhs);
  if (!eq) return NULL;
  int64_t out_shape[POLY_MAX_DIMS];
  int out_ndim = 0;
  PolyUOp *mask_bc = mask;
  PolyUOp *eq_bc = eq;
  if (!poly_broadcast_pair(ctx, &mask_bc, &eq_bc, out_shape, &out_ndim)) return NULL;
  PolyUOp *m = poly_alu2(ctx, POLY_OP_AND, mask_bc, eq_bc);
  PolyUOp *mi = poly_cast(ctx, m, POLY_INT32);
  return poly_sum_reduce(ctx, mi, dim + 1, 0);
}

int poly_sort(
    PolyCtx *ctx,
    PolyUOp *x,
    int dim,
    int descending,
    PolyUOp **out_values,
    PolyUOp **out_indices
) {
  if (!ctx || !x || !out_values || !out_indices) return -1;
  *out_values = NULL;
  *out_indices = NULL;

  int64_t orig_shape[POLY_MAX_DIMS];
  int orig_ndim = uop_shape(ctx, x, orig_shape);
  if (orig_ndim < 1) return -1;
  if (dim < 0) dim += orig_ndim;
  if (dim < 0 || dim >= orig_ndim) return -1;
  int64_t orig_len = orig_shape[dim];
  int int32_id = poly_dtype_id_by_name("int32");
  if (int32_id < 0) return -1;

  if (orig_len <= 1) {
    *out_values = x;
    *out_indices = poly_full_int_by_id(ctx, orig_shape, orig_ndim, 0, int32_id);
    return *out_indices ? 0 : -1;
  }

  int n_stages = 0;
  int64_t padded_len = 1;
  while (padded_len < orig_len) {
    if (padded_len > INT64_MAX / 2) return -1;
    padded_len *= 2;
    n_stages++;
  }
  if (orig_ndim + n_stages - 1 > POLY_MAX_DIMS) return -1;

  PolyUOp *pad_value = NULL;
  if (!poly_sort_bound_const(ctx, x->dtype, descending != 0, &pad_value)) return -1;

  int64_t pads[POLY_MAX_DIMS][2];
  for (int i = 0; i < orig_ndim; i++) {
    pads[i][0] = 0;
    pads[i][1] = (i == dim) ? (padded_len - orig_len) : 0;
  }
  PolyUOp *cur = poly_pad_with_scalar_nonnegative(ctx, x, pads, orig_ndim, pad_value);
  if (!cur) return -1;

  int64_t unflat_shape[POLY_MAX_DIMS];
  int un = 0;
  for (int i = 0; i < dim; i++)
    unflat_shape[un++] = orig_shape[i];
  for (int i = 0; i < n_stages; i++)
    unflat_shape[un++] = 2;
  for (int i = dim + 1; i < orig_ndim; i++)
    unflat_shape[un++] = orig_shape[i];
  cur = poly_reshape(ctx, cur, unflat_shape, un);
  if (!cur) return -1;

  for (int stage = 1; stage <= n_stages; stage++) {
    int crossover_dim = dim + n_stages - stage - 1;
    int64_t flip_axes[POLY_MAX_DIMS];
    int n_flip_axes = 0;
    if (stage != n_stages) {
      PolyUOp *blue = NULL, *green = NULL;
      if (!poly_split_two_ones(ctx, cur, crossover_dim, &blue, &green)) return -1;
      for (int i = 1; i < stage + 1 + (orig_ndim - dim); i++)
        flip_axes[n_flip_axes++] = -i;
      if (poly_resolve_sort_flip_axes(flip_axes, n_flip_axes, un) != 0) return -1;
      PolyUOp *flipped = poly_flip(ctx, green, flip_axes, n_flip_axes);
      PolyUOp *parts[2] = {blue, flipped};
      cur = poly_contiguous(ctx, poly_cat(ctx, parts, 2, crossover_dim));
      if (!cur) return -1;
    }

    for (int substage = stage - 1; substage >= 0; substage--) {
      int partner_dim = dim + n_stages - substage - 1;
      PolyUOp *top = NULL, *bottom = NULL;
      if (!poly_split_two_ones(ctx, cur, partner_dim, &top, &bottom)) return -1;
      PolyUOp *larger = poly_maximum(ctx, top, bottom);
      PolyUOp *smaller = poly_minimum(ctx, top, bottom);
      PolyUOp *parts[2] = {descending ? larger : smaller, descending ? smaller : larger};
      cur = poly_contiguous(ctx, poly_cat(ctx, parts, 2, partner_dim));
      if (!cur) return -1;
    }

    if (stage != n_stages) {
      PolyUOp *blue = NULL, *flipped_green = NULL;
      if (!poly_split_two_ones(ctx, cur, crossover_dim, &blue, &flipped_green)) return -1;
      PolyUOp *green = poly_flip(ctx, flipped_green, flip_axes, n_flip_axes);
      PolyUOp *parts[2] = {blue, green};
      cur = poly_cat(ctx, parts, 2, crossover_dim);
      if (!cur) return -1;
    }
  }

  cur = poly_flatten_axes(ctx, cur, dim, dim + n_stages - 1);
  if (!cur) return -1;
  cur = poly_shrink_to(ctx, cur, orig_shape, orig_ndim);
  if (!cur) return -1;

  int64_t mask_shape[POLY_MAX_DIMS];
  int mask_ndim = 0;
  mask_shape[mask_ndim++] = orig_len;
  mask_shape[mask_ndim++] = orig_len;
  for (int i = 0; i < orig_ndim - dim - 1; i++)
    mask_shape[mask_ndim++] = 1;
  PolyUOp *mask = poly_full_int_by_id(ctx, mask_shape, mask_ndim, 1, poly_dtype_id_by_name("bool"));
  if (!mask) return -1;
  mask = poly_tril(ctx, mask, 0);
  if (!mask) return -1;

  PolyUOp *count_orig = poly_sort_count_equal_before(ctx, mask, x, dim);
  PolyUOp *count_sorted = poly_sort_count_equal_before(ctx, mask, cur, dim);
  if (!count_orig || !count_sorted) return -1;

  PolyUOp *orig_us = poly_unsqueeze_axis(ctx, x, dim + 1);
  PolyUOp *sorted_us = poly_unsqueeze_axis(ctx, cur, dim);
  if (!orig_us || !sorted_us) return -1;
  PolyUOp *value_eq = poly_eq(ctx, orig_us, sorted_us);
  PolyUOp *count_eq = poly_eq(
      ctx, poly_unsqueeze_axis(ctx, count_orig, dim + 1),
      poly_unsqueeze_axis(ctx, count_sorted, dim)
  );
  if (!value_eq || !count_eq) return -1;
  int64_t cond_shape[POLY_MAX_DIMS];
  int cond_ndim = 0;
  PolyUOp *value_eq_bc = value_eq;
  PolyUOp *count_eq_bc = count_eq;
  if (!poly_broadcast_pair(ctx, &value_eq_bc, &count_eq_bc, cond_shape, &cond_ndim)) return -1;
  PolyUOp *cond = poly_alu2(ctx, POLY_OP_AND, value_eq_bc, count_eq_bc);
  PolyUOp *cond_i = poly_cast(ctx, cond, POLY_INT32);

  PolyUOp *idx = poly_arange_int_by_id(ctx, 0, orig_len, 1, int32_id);
  if (!idx) return -1;
  int64_t idx_shape[POLY_MAX_DIMS];
  for (int i = 0; i < orig_ndim; i++)
    idx_shape[i] = (i == dim) ? orig_len : 1;
  /* tinygrad/mixin/movement.py:145-165: Tensor.reshape returns the input when
   * the requested shape is unchanged. */
  int64_t current_idx_shape[POLY_MAX_DIMS];
  int current_idx_ndim = uop_shape(ctx, idx, current_idx_shape);
  bool idx_shape_matches = current_idx_ndim == orig_ndim;
  for (int i = 0; idx_shape_matches && i < orig_ndim; i++)
    idx_shape_matches = current_idx_shape[i] == idx_shape[i];
  if (!idx_shape_matches) idx = poly_reshape(ctx, idx, idx_shape, orig_ndim);
  idx = poly_unsqueeze_axis(ctx, idx, dim + 1);
  if (!idx) return -1;

  PolyUOp *cond_bc = cond_i;
  PolyUOp *idx_bc = idx;
  int64_t mul_shape[POLY_MAX_DIMS];
  int mul_ndim = 0;
  if (!poly_broadcast_pair(ctx, &cond_bc, &idx_bc, mul_shape, &mul_ndim)) return -1;
  PolyUOp *idx_masked = poly_alu2(ctx, POLY_OP_MUL, cond_bc, idx_bc);
  PolyUOp *idx_sum = poly_sum_reduce(ctx, idx_masked, dim, 0);
  if (!idx_sum) return -1;

  *out_values = cur;
  *out_indices = idx_sum;
  return 0;
}

int poly_tensor_sort(
    PolyCtx *ctx,
    PolyTensor *src,
    int dim,
    int descending,
    PolyTensor **out_values,
    PolyTensor **out_indices
) {
  if (!out_values || !out_indices) return -1;
  *out_values = NULL;
  *out_indices = NULL;
  if (!tensor_roots_owned_by_ctx(ctx, src)) return -1;

  /* tinygrad/mixin/__init__.py:994-1044 constructs both sort results from
   * the exact input Tensor.uop. The Tensor boundary applies that program independently to
   * the retained logical root and mandatory physical occurrence. */
  PolyUOp *logical_values = NULL, *logical_indices = NULL;
  PolyUOp *physical_values = NULL, *physical_indices = NULL;
  if (poly_sort(ctx, src->uop_logical, dim, descending, &logical_values, &logical_indices) != 0 ||
      poly_sort(ctx, src->uop_physical, dim, descending, &physical_values, &physical_indices) != 0)
    return -1;

  PolyTensor *values = tensor_unary_result(ctx, src, logical_values, physical_values);
  PolyTensor *indices = poly_tensor_create_with_roots(
      ctx, logical_indices, physical_indices, POLY_TENSOR_VALUE, src->device
  );
  if (!values || !indices) return -1;
  indices->requires_grad = false;
  indices->requires_grad_set = true;
  indices->provenance = POLY_TENSOR_PROVENANCE_COMPUTED;
  *out_values = values;
  *out_indices = indices;
  return 0;
}

PolyUOp *poly_argsort(PolyCtx *ctx, PolyUOp *x, int dim, int descending) {
  PolyUOp *values = NULL, *indices = NULL;
  if (poly_sort(ctx, x, dim, descending, &values, &indices) != 0) return NULL;
  (void)values;
  return indices;
}

int poly_topk(
    PolyCtx *ctx,
    PolyUOp *x,
    int64_t k,
    int dim,
    int largest,
    int sorted,
    PolyUOp **out_values,
    PolyUOp **out_indices
) {
  if (!ctx || !x || !out_values || !out_indices) return -1;
  if (!sorted) return -1;
  int64_t shape[POLY_MAX_DIMS];
  int ndim = uop_shape(ctx, x, shape);
  if (ndim < 1) return -1;
  if (dim < 0) dim += ndim;
  if (dim < 0 || dim >= ndim) return -1;
  if (k > shape[dim]) return -1;

  PolyUOp *values = NULL, *indices = NULL;
  if (poly_sort(ctx, x, dim, largest, &values, &indices) != 0) return -1;
  int64_t ends[POLY_MAX_DIMS];
  for (int i = 0; i < ndim; i++)
    ends[i] = (i == dim) ? k : -1;
  *out_values = poly_shrink_to(ctx, values, ends, ndim);
  *out_indices = poly_shrink_to(ctx, indices, ends, ndim);
  return (*out_values && *out_indices) ? 0 : -1;
}

int poly_tensor_topk(
    PolyCtx *ctx,
    PolyTensor *src,
    int64_t k,
    int dim,
    int largest,
    int sorted,
    PolyTensor **out_values,
    PolyTensor **out_indices
) {
  if (!out_values || !out_indices) return -1;
  *out_values = NULL;
  *out_indices = NULL;
  if (!tensor_roots_owned_by_ctx(ctx, src)) return -1;

  /* Pinned tinygrad/mixin/__init__.py:1057-1077 applies sort+shrink_to to the
   * exact current Tensor.uop. The Tensor boundary applies that unchanged program
   * independently to the retained logical root and mandatory physical
   * occurrence, exactly like poly_tensor_sort above. */
  PolyUOp *logical_values = NULL, *logical_indices = NULL;
  PolyUOp *physical_values = NULL, *physical_indices = NULL;
  if (poly_topk(
          ctx, src->uop_logical, k, dim, largest, sorted, &logical_values, &logical_indices
      ) != 0 ||
      poly_topk(
          ctx, src->uop_physical, k, dim, largest, sorted, &physical_values,
          &physical_indices
      ) != 0)
    return -1;

  PolyTensor *values = tensor_unary_result(ctx, src, logical_values, physical_values);
  PolyTensor *indices = poly_tensor_create_with_roots(
      ctx, logical_indices, physical_indices, POLY_TENSOR_VALUE, src->device
  );
  if (!values || !indices) return -1;
  indices->requires_grad = false;
  indices->requires_grad_set = true;
  indices->provenance = POLY_TENSOR_PROVENANCE_COMPUTED;
  *out_values = values;
  *out_indices = indices;
  return 0;
}

/* Einsum */

#define MAX_EINSUM_TENSORS 8

PolyUOp *poly_einsum(PolyCtx *ctx, const char *formula, PolyUOp **tensors, int n_tensors) {
  if (!ctx || !formula || !tensors || n_tensors <= 0 || n_tensors > MAX_EINSUM_TENSORS) return NULL;

  /* Read shapes from UOps */
  int64_t shape_store[MAX_EINSUM_TENSORS][POLY_MAX_DIMS];
  const int64_t *shapes[MAX_EINSUM_TENSORS];
  int ndims[MAX_EINSUM_TENSORS];
  for (int t = 0; t < n_tensors; t++) {
    if (!tensors[t] || !poly_ctx_owns_ptr(ctx, tensors[t])) return NULL;
    ndims[t] = uop_shape(ctx, tensors[t], shape_store[t]);
    shapes[t] = shape_store[t];
    if (ndims[t] < 0) return NULL;
  }

  char clean[256];
  int ci = 0;
  for (const char *p = formula; *p; p++) {
    if (*p == ' ') continue;
    if (ci >= (int)sizeof(clean) - 1) return NULL;
    clean[ci++] = *p;
  }
  clean[ci] = '\0';

  char lhs_buf[256], rhs_buf[64];
  char *arrow = strstr(clean, "->");
  if (arrow) {
    int lhs_len = (int)(arrow - clean);
    size_t rhs_len = strlen(arrow + 2);
    if (lhs_len <= 0 || lhs_len >= (int)sizeof(lhs_buf) || rhs_len >= sizeof(rhs_buf)) return NULL;
    memcpy(lhs_buf, clean, lhs_len);
    lhs_buf[lhs_len] = '\0';
    memcpy(rhs_buf, arrow + 2, rhs_len + 1);
  } else {
    if (strlen(clean) >= sizeof(lhs_buf)) return NULL;
    memcpy(lhs_buf, clean, strlen(clean) + 1);
    int count[26] = {0};
    for (char *p2 = lhs_buf; *p2; p2++)
      if (*p2 >= 'a' && *p2 <= 'z') count[*p2 - 'a']++;
    int ri = 0;
    for (int i = 0; i < 26; i++)
      if (count[i] == 1) rhs_buf[ri++] = (char)('a' + i);
    rhs_buf[ri] = '\0';
  }

  char *input_specs[MAX_EINSUM_TENSORS];
  int n_inputs = 0;
  char *pp = lhs_buf;
  while (*pp && n_inputs < MAX_EINSUM_TENSORS) {
    input_specs[n_inputs++] = pp;
    while (*pp && *pp != ',')
      pp++;
    if (*pp == ',') *pp++ = '\0';
  }
  if (n_inputs != n_tensors) return NULL;

  int64_t sz[26];
  bool has_letter[26];
  memset(has_letter, 0, sizeof(has_letter));
  for (int t = 0; t < n_tensors; t++) {
    const char *spec = input_specs[t];
    int spec_len = (int)strlen(spec);
    if (spec_len != ndims[t]) return NULL;
    for (int d = 0; d < spec_len; d++) {
      int li = spec[d] - 'a';
      if (li < 0 || li >= 26) return NULL;
      if (has_letter[li]) {
        if (sz[li] != shapes[t][d]) return NULL;
      } else {
        sz[li] = shapes[t][d];
        has_letter[li] = true;
      }
    }
  }

  /* Trace: extract diagonal when a letter repeats in a single input. */
  char trace_specs[MAX_EINSUM_TENSORS][64];
  PolyUOp *trace_tensors[MAX_EINSUM_TENSORS];
  int trace_ndims[MAX_EINSUM_TENSORS];
  int64_t trace_shapes[MAX_EINSUM_TENSORS][POLY_MAX_DIMS];

  for (int t = 0; t < n_tensors; t++) {
    strcpy(trace_specs[t], input_specs[t]);
    trace_tensors[t] = tensors[t];
    trace_ndims[t] = ndims[t];
    for (int d = 0; d < ndims[t]; d++)
      trace_shapes[t][d] = shapes[t][d];
  }

  for (int t = 0; t < n_tensors; t++) {
    char *s = trace_specs[t];
    int slen = (int)strlen(s);
    PolyUOp *x = trace_tensors[t];
    int x_ndim = trace_ndims[t];
    int64_t *x_shape = trace_shapes[t];

    for (int ci2 = 0; ci2 < slen; ci2++) {
      char c = s[ci2];
      int ki = -1;
      for (int k = ci2 + 1; k < slen; k++)
        if (s[k] == c) {
          ki = k;
          break;
        }
      if (ki < 0) continue;

      int64_t n = x_shape[ci2];

      int64_t perm[POLY_MAX_DIMS];
      int pi = 0;
      for (int d = 0; d < x_ndim; d++)
        if (d != ci2 && d != ki) perm[pi++] = d;
      perm[pi++] = ci2;
      perm[pi++] = ki;
      x = poly_permute(ctx, x, perm, x_ndim);

      int64_t pshape[POLY_MAX_DIMS];
      for (int d = 0; d < x_ndim; d++)
        pshape[d] = x_shape[perm[d]];
      memcpy(x_shape, pshape, x_ndim * sizeof(int64_t));

      int64_t flat_shape[POLY_MAX_DIMS];
      int flat_ndim = x_ndim - 1;
      for (int d = 0; d < flat_ndim - 1; d++)
        flat_shape[d] = x_shape[d];
      flat_shape[flat_ndim - 1] = n * n;
      x = poly_reshape(ctx, x, flat_shape, flat_ndim);

      int64_t pad_pairs[POLY_MAX_DIMS][2];
      for (int d = 0; d < flat_ndim; d++) {
        pad_pairs[d][0] = 0;
        pad_pairs[d][1] = 0;
      }
      pad_pairs[flat_ndim - 1][1] = n;
      x = poly_pad(ctx, x, pad_pairs, flat_ndim);

      int64_t uf_shape[POLY_MAX_DIMS];
      int uf_ndim = flat_ndim + 1;
      for (int d = 0; d < flat_ndim - 1; d++)
        uf_shape[d] = flat_shape[d];
      uf_shape[flat_ndim - 1] = n;
      uf_shape[flat_ndim] = n + 1;
      x = poly_reshape(ctx, x, uf_shape, uf_ndim);

      int64_t shrink_pairs[POLY_MAX_DIMS][2];
      for (int d = 0; d < uf_ndim; d++) {
        shrink_pairs[d][0] = 0;
        shrink_pairs[d][1] = uf_shape[d];
      }
      shrink_pairs[uf_ndim - 1][0] = 0;
      shrink_pairs[uf_ndim - 1][1] = 1;
      x = poly_shrink(ctx, x, shrink_pairs, uf_ndim);

      int64_t final_shape[POLY_MAX_DIMS];
      int final_ndim = uf_ndim - 1;
      for (int d = 0; d < final_ndim; d++)
        final_shape[d] = uf_shape[d];
      x = poly_reshape(ctx, x, final_shape, final_ndim);

      for (int k = ki; k < slen - 1; k++)
        s[k] = s[k + 1];
      s[slen - 1] = '\0';
      slen--;

      x_ndim = final_ndim;
      memcpy(x_shape, final_shape, final_ndim * sizeof(int64_t));

      ci2--;
    }

    trace_tensors[t] = x;
    trace_ndims[t] = x_ndim;
    input_specs[t] = trace_specs[t];
  }

  /* Rebuild size dict after trace reduction */
  memset(has_letter, 0, sizeof(has_letter));
  for (int t = 0; t < n_tensors; t++) {
    const char *spec = input_specs[t];
    int spec_len = (int)strlen(spec);
    for (int d = 0; d < spec_len; d++) {
      int li = spec[d] - 'a';
      if (!has_letter[li]) {
        sz[li] = trace_shapes[t][d];
        has_letter[li] = true;
      }
    }
  }

  char alpha[26];
  int n_alpha = 0;
  for (int i = 0; i < 26; i++)
    if (has_letter[i]) alpha[n_alpha++] = (char)('a' + i);

  PolyUOp *aligned[MAX_EINSUM_TENSORS];
  for (int t = 0; t < n_tensors; t++) {
    const char *spec = input_specs[t];
    int spec_len = (int)strlen(spec);
    PolyUOp *x = trace_tensors[t];
    if (spec_len == 0) {
      aligned[t] = x;
      continue;
    }

    char sorted_spec[27];
    memcpy(sorted_spec, spec, spec_len);
    sorted_spec[spec_len] = '\0';
    for (int i = 0; i < spec_len - 1; i++)
      for (int j = i + 1; j < spec_len; j++)
        if (sorted_spec[i] > sorted_spec[j]) {
          char tmp = sorted_spec[i];
          sorted_spec[i] = sorted_spec[j];
          sorted_spec[j] = tmp;
        }

    int64_t perm[POLY_MAX_DIMS];
    bool needs_perm = false;
    for (int i = 0; i < spec_len; i++) {
      for (int j = 0; j < spec_len; j++)
        if (spec[j] == sorted_spec[i]) {
          perm[i] = j;
          break;
        }
      if (perm[i] != i) needs_perm = true;
    }
    if (needs_perm) x = poly_permute(ctx, x, perm, spec_len);

    int64_t rshape[POLY_MAX_DIMS];
    for (int i = 0; i < n_alpha; i++) {
      bool found = false;
      for (int j = 0; j < spec_len; j++)
        if (sorted_spec[j] == alpha[i]) {
          found = true;
          break;
        }
      rshape[i] = found ? sz[(int)(alpha[i] - 'a')] : 1;
    }
    x = poly_reshape(ctx, x, rshape, n_alpha);

    int64_t full[POLY_MAX_DIMS];
    for (int i = 0; i < n_alpha; i++)
      full[i] = sz[(int)(alpha[i] - 'a')];
    x = poly_expand(ctx, x, full, n_alpha);

    aligned[t] = x;
  }

  PolyUOp *result = aligned[0];
  for (int t = 1; t < n_tensors; t++)
    result = poly_alu2(ctx, POLY_OP_MUL, result, aligned[t]);

  int64_t sum_axes[POLY_MAX_DIMS];
  int n_sum = 0;
  for (int i = 0; i < n_alpha; i++) {
    bool in_rhs = false;
    for (const char *r = rhs_buf; *r; r++)
      if (*r == alpha[i]) {
        in_rhs = true;
        break;
      }
    if (!in_rhs) sum_axes[n_sum++] = i;
  }
  if (n_sum > 0) result = poly_reduce_axis(ctx, POLY_OP_ADD, result, sum_axes, n_sum);

  char remaining[26];
  int n_remaining = 0;
  for (int i = 0; i < n_alpha; i++) {
    bool summed = false;
    for (int j = 0; j < n_sum; j++)
      if (sum_axes[j] == i) {
        summed = true;
        break;
      }
    if (!summed) remaining[n_remaining++] = alpha[i];
  }

  int64_t remaining_shape[POLY_MAX_DIMS];
  for (int i = 0; i < n_remaining; i++)
    remaining_shape[i] = sz[(int)(remaining[i] - 'a')];
  result = poly_reshape(ctx, result, n_remaining > 0 ? remaining_shape : NULL, n_remaining);
  if (!result) return NULL;

  int rhs_len = (int)strlen(rhs_buf);
  if (rhs_len != n_remaining) return NULL;

  int64_t out_perm[POLY_MAX_DIMS];
  bool needs_final_perm = false;
  for (int i = 0; i < rhs_len; i++) {
    for (int j = 0; j < n_remaining; j++)
      if (remaining[j] == rhs_buf[i]) {
        out_perm[i] = j;
        if (j != i) needs_final_perm = true;
        break;
      }
  }
  if (needs_final_perm) result = poly_permute(ctx, result, out_perm, rhs_len);

  return result;
}

PolyTensor *poly_tensor_einsum(
    PolyCtx *ctx,
    const char *formula,
    PolyTensor **tensors,
    int n_tensors
) {
  if (!ctx || !formula || !tensors || n_tensors <= 0 ||
      n_tensors > MAX_EINSUM_TENSORS)
    return NULL;

  PolyUOp *logical[MAX_EINSUM_TENSORS];
  PolyUOp *physical[MAX_EINSUM_TENSORS];
  for (int i = 0; i < n_tensors; i++) {
    if (!tensor_roots_owned_by_ctx(ctx, tensors[i])) return NULL;
    logical[i] = tensors[i]->uop_logical;
    physical[i] = tensors[i]->uop_physical;
  }

  /* Pinned Tensor.einsum applies one formula to its ordered Tensor.uop
   * operands (mixin/__init__.py:496-535). The Tensor boundary applies the unchanged raw
   * program independently to retained logical roots and mandatory physical
   * occurrences; it never recovers physical output by logical substitution. */
  PolyUOp *logical_result = poly_einsum(ctx, formula, logical, n_tensors);
  PolyUOp *physical_result = poly_einsum(ctx, formula, physical, n_tensors);
  return tensor_composite_result(
      ctx, logical_result, physical_result, tensors, n_tensors
  );
}

/* Rearrange (einops) */

#define MAX_REARRANGE_TOKENS POLY_MAX_DIMS
#define MAX_REARRANGE_TOKEN_BYTES 256

static int parse_rearrange_side(
    const char *s,
    char tokens[][MAX_REARRANGE_TOKEN_BYTES],
    int *n_tokens,
    int groups[][2],
    int *n_groups
) {
  if (!s || !tokens || !n_tokens || !groups || !n_groups) return -1;
  *n_tokens = 0;
  *n_groups = 0;
  int paren_start = -1;

  const char *p = s;
  while (*p) {
    while (*p == ' ' || *p == '\t')
      p++;
    if (!*p) break;

    if (*p == '(') {
      if (paren_start >= 0) return -1;
      paren_start = *n_tokens;
      p++;
      continue;
    }
    if (*p == ')') {
      if (paren_start < 0 || *n_groups >= MAX_REARRANGE_TOKENS) return -1;
      groups[*n_groups][0] = paren_start;
      groups[*n_groups][1] = *n_tokens;
      (*n_groups)++;
      paren_start = -1;
      p++;
      continue;
    }

    if (*n_tokens >= MAX_REARRANGE_TOKENS) return -1;
    size_t ti = 0;
    while (*p && *p != ' ' && *p != '\t' && *p != '(' && *p != ')') {
      if (ti + 1 >= MAX_REARRANGE_TOKEN_BYTES) return -1;
      tokens[*n_tokens][ti++] = *p++;
    }
    tokens[*n_tokens][ti] = '\0';
    (*n_tokens)++;
  }
  if (paren_start >= 0) return -1;
  return *n_tokens;
}

static int64_t find_axis_size(
    const char *name,
    const char *axis_names,
    const int64_t *axis_values,
    int n_axis_sizes
) {
  if (!axis_names || n_axis_sizes <= 0) return -1;
  const char *p = axis_names;
  int idx = 0;
  while (*p && idx < n_axis_sizes) {
    while (*p == ' ')
      p++;
    if (!*p) break;
    const char *start = p;
    while (*p && *p != ' ')
      p++;
    int len = (int)(p - start);
    if ((int)strlen(name) == len && memcmp(start, name, len) == 0) return axis_values[idx];
    idx++;
  }
  return -1;
}

PolyUOp *poly_rearrange(
    PolyCtx *ctx,
    const char *formula,
    PolyUOp *x,
    const char *axis_names,
    const int64_t *axis_values,
    int n_axis_sizes
) {
  if (!ctx || !formula || !x || !poly_ctx_owns_ptr(ctx, x) || n_axis_sizes < 0 ||
      n_axis_sizes > POLY_MAX_DIMS || (n_axis_sizes > 0 && (!axis_names || !axis_values)))
    return NULL;
  for (int i = 0; i < n_axis_sizes; i++)
    if (axis_values[i] <= 0) return NULL;
  int64_t shape[POLY_MAX_DIMS];
  int ndim;
  ndim = uop_shape(ctx, x, shape);
  if (ndim < 0) return NULL;

  const char *arrow_pos = strstr(formula, "->");
  if (!arrow_pos || strstr(arrow_pos + 2, "->")) return NULL;

  char lhs_str[256], rhs_str[256];
  size_t lhs_len = (size_t)(arrow_pos - formula);
  size_t rhs_len = strlen(arrow_pos + 2);
  if (lhs_len >= sizeof(lhs_str) || rhs_len >= sizeof(rhs_str)) return NULL;
  memcpy(lhs_str, formula, lhs_len);
  lhs_str[lhs_len] = '\0';
  memcpy(rhs_str, arrow_pos + 2, rhs_len + 1);

  char lhs_tok[MAX_REARRANGE_TOKENS][MAX_REARRANGE_TOKEN_BYTES];
  char rhs_tok[MAX_REARRANGE_TOKENS][MAX_REARRANGE_TOKEN_BYTES];
  int lhs_grp[MAX_REARRANGE_TOKENS][2], rhs_grp[MAX_REARRANGE_TOKENS][2];
  int n_lt = 0, n_rt = 0, n_lg = 0, n_rg = 0;

  if (parse_rearrange_side(lhs_str, lhs_tok, &n_lt, lhs_grp, &n_lg) < 0 ||
      parse_rearrange_side(rhs_str, rhs_tok, &n_rt, rhs_grp, &n_rg) < 0)
    return NULL;
  if (n_lt != n_rt) return NULL;
  bool used_lhs[MAX_REARRANGE_TOKENS] = {0};
  for (int i = 0; i < n_rt; i++) {
    int match = -1;
    for (int j = 0; j < n_lt; j++) {
      if (strcmp(rhs_tok[i], lhs_tok[j]) != 0) continue;
      if (match >= 0 || used_lhs[j]) return NULL;
      match = j;
    }
    if (match < 0) return NULL;
    used_lhs[match] = true;
  }

  PolyUOp *result = x;
  int64_t cur_shape[POLY_MAX_DIMS];
  int cur_ndim = ndim;
  memcpy(cur_shape, shape, ndim * sizeof(int64_t));

  /* Phase 1: Unflatten (lhs groups) */
  if (n_lg > 0) {
    bool in_group[MAX_REARRANGE_TOKENS];
    int g_id[MAX_REARRANGE_TOKENS];
    memset(in_group, 0, sizeof(in_group));
    for (int i = 0; i < MAX_REARRANGE_TOKENS; i++)
      g_id[i] = -1;
    for (int g = 0; g < n_lg; g++)
      for (int i = lhs_grp[g][0]; i < lhs_grp[g][1]; i++) {
        in_group[i] = true;
        g_id[i] = g;
      }

    int64_t new_shape[POLY_MAX_DIMS];
    int new_ndim = 0, input_dim = 0, ti = 0;

    while (ti < n_lt) {
      if (in_group[ti]) {
        int g = g_id[ti];
        int gs = lhs_grp[g][0], ge = lhs_grp[g][1];
        int gc = ge - gs;
        int64_t sub[POLY_MAX_DIMS];
        int64_t known = 1;
        int unk = -1;
        for (int i = 0; i < gc; i++) {
          const char *nm = lhs_tok[gs + i];
          if (strcmp(nm, "1") == 0) {
            sub[i] = 1;
          } else {
            int64_t v = find_axis_size(nm, axis_names, axis_values, n_axis_sizes);
            if (v > 0)
              sub[i] = v;
            else {
              if (unk >= 0) return NULL;
              unk = i;
              sub[i] = -1;
            }
          }
          if (sub[i] > 0 && __builtin_mul_overflow(known, sub[i], &known)) return NULL;
        }
        if (unk >= 0) {
          if (input_dim >= cur_ndim || known <= 0 || cur_shape[input_dim] < 0 ||
              cur_shape[input_dim] % known != 0)
            return NULL;
          sub[unk] = cur_shape[input_dim] / known;
        }
        if (new_ndim > POLY_MAX_DIMS - gc) return NULL;
        for (int i = 0; i < gc; i++)
          new_shape[new_ndim++] = sub[i];
        input_dim++;
        ti = ge;
      } else {
        if (strcmp(lhs_tok[ti], "1") == 0)
          new_shape[new_ndim++] = 1;
        else {
          if (input_dim >= cur_ndim) return NULL;
          new_shape[new_ndim++] = cur_shape[input_dim];
        }
        input_dim++;
        ti++;
      }
    }
    if (input_dim != cur_ndim || new_ndim != n_lt) return NULL;

    if (new_ndim != cur_ndim || memcmp(new_shape, cur_shape, cur_ndim * sizeof(int64_t)) != 0) {
      result = poly_reshape(ctx, result, new_shape, new_ndim);
      if (!result) return NULL;
      memcpy(cur_shape, new_shape, new_ndim * sizeof(int64_t));
      cur_ndim = new_ndim;
    }
  }

  /* Phase 2: Permute (lhs order -> rhs order) */
  if (cur_ndim != n_lt) return NULL;
  int64_t perm[POLY_MAX_DIMS];
  bool need_perm = false;
  for (int i = 0; i < n_rt; i++) {
    perm[i] = -1;
    for (int j = 0; j < n_lt; j++)
      if (strcmp(rhs_tok[i], lhs_tok[j]) == 0) {
        perm[i] = j;
        break;
      }
    if (perm[i] < 0) return NULL;
    if (perm[i] != i) need_perm = true;
  }
  if (need_perm) {
    result = poly_permute(ctx, result, perm, n_rt);
    if (!result) return NULL;
    int64_t ps[POLY_MAX_DIMS];
    for (int i = 0; i < n_rt; i++)
      ps[i] = cur_shape[perm[i]];
    memcpy(cur_shape, ps, n_rt * sizeof(int64_t));
    cur_ndim = n_rt;
  }

  /* Phase 3: Flatten (rhs groups, process right to left) */
  for (int g = n_rg - 1; g >= 0; g--) {
    int gs = rhs_grp[g][0], ge = rhs_grp[g][1];
    if (ge - gs <= 1) continue;
    int64_t flat = 1;
    if (gs < 0 || ge > cur_ndim) return NULL;
    for (int i = gs; i < ge; i++)
      if (cur_shape[i] < 0 || __builtin_mul_overflow(flat, cur_shape[i], &flat)) return NULL;
    int64_t ns[POLY_MAX_DIMS];
    int nn = 0;
    for (int i = 0; i < gs; i++)
      ns[nn++] = cur_shape[i];
    ns[nn++] = flat;
    for (int i = ge; i < cur_ndim; i++)
      ns[nn++] = cur_shape[i];
    result = poly_reshape(ctx, result, ns, nn);
    if (!result) return NULL;
    memcpy(cur_shape, ns, nn * sizeof(int64_t));
    cur_ndim = nn;
  }

  return result;
}

PolyTensor *poly_tensor_rearrange(
    PolyCtx *ctx,
    const char *formula,
    PolyTensor *tensor,
    const char *axis_names,
    const int64_t *axis_values,
    int n_axis_sizes
) {
  if (!tensor_roots_owned_by_ctx(ctx, tensor)) return NULL;
  /* Pinned rearrange is unflatten -> permute -> flatten on Tensor.uop
   * (mixin/movement.py:340-383). The Tensor boundary runs that unchanged raw program on
   * each exact root; frontends never reconstruct the physical occurrence. */
  PolyUOp *logical = poly_rearrange(
      ctx, formula, tensor->uop_logical, axis_names, axis_values, n_axis_sizes
  );
  PolyUOp *physical = poly_rearrange(
      ctx, formula, tensor->uop_physical, axis_names, axis_values, n_axis_sizes
  );
  return tensor_unary_result(ctx, tensor, logical, physical);
}

/* Gather (embedding lookup) */

PolyUOp *poly_gather_dim(PolyCtx *ctx, PolyUOp *x, int dim, PolyUOp *index) {
  if (!ctx || !x || !index) return NULL;
  int64_t shape[POLY_MAX_DIMS], index_shape[POLY_MAX_DIMS];
  int ndim = uop_shape(ctx, x, shape);
  int index_ndim = uop_shape(ctx, index, index_shape);
  if (ndim < 0 || index_ndim < 0 || ndim != index_ndim) return NULL;
  if (dim < 0) dim += ndim;
  if (dim < 0 || dim >= ndim) return NULL;
  for (int d = 0; d < ndim; d++)
    if (d != dim && shape[d] < index_shape[d]) return NULL;

  int64_t ends[POLY_MAX_DIMS];
  for (int d = 0; d < ndim; d++)
    ends[d] = (d == dim) ? -1 : index_shape[d];
  PolyUOp *xs = poly_shrink_to(ctx, x, ends, ndim);
  if (!xs) return NULL;

  PolyUOp *xu = poly_unsqueeze_axis(ctx, xs, -1);
  if (!xu) return NULL;
  int xu_ndim = ndim + 1;
  int64_t perm[POLY_MAX_DIMS];
  for (int i = 0; i < xu_ndim; i++)
    perm[i] = i;
  perm[dim] = xu_ndim - 1;
  perm[xu_ndim - 1] = dim;
  PolyUOp *xg = poly_permute(ctx, xu, perm, xu_ndim);
  if (!xg) return NULL;

  PolyUOp *index_u = poly_unsqueeze_axis(ctx, index, -1);
  if (!index_u) return NULL;
  PolyUOp *mask = one_hot_along_dim(ctx, index_u, shape[dim], -1);
  if (!mask) return NULL;
  /* Pinned gather spells this as `mask.where(x, 0)` and explicitly retains
   * x.dtype through the reduction (mixin/__init__.py:1123-1124). */
  PolyUOp *selected =
      poly_where_op(ctx, mask, xg, poly_const_typed(ctx, poly_dtype_scalar(xg->dtype), 0.0));
  if (!selected) return NULL;
  return sum_axis_keep_dtype(ctx, selected, -1);
}

PolyTensor *poly_tensor_gather_dim(PolyCtx *ctx, PolyTensor *x, int dim, PolyTensor *index) {
  PolyUOp *x_current = tensor_current_uop(x);
  PolyUOp *index_current = tensor_current_uop(index);
  if (!ctx || !x || !index || !x->uop_logical || !index->uop_logical || !x_current ||
      !index_current)
    return NULL;
  PolyUOp *logical = poly_gather_dim(ctx, x->uop_logical, dim, index->uop_logical);
  PolyUOp *physical = poly_gather_dim(ctx, x_current, dim, index_current);
  PolyTensor *inputs[2] = {x, index};
  return tensor_composite_result(ctx, logical, physical, inputs, 2);
}

typedef struct {
  PolyUOp *src;
  PolyUOp *mask;
  int64_t self_shape[POLY_MAX_DIMS];
  int ndim;
} PolyScatterPrepared;

static PolyUOp *poly_pad_to_scatter_self(
    PolyCtx *ctx,
    PolyUOp *x,
    const int64_t *self_shape,
    int ndim
) {
  if (!ctx || !x || !self_shape || ndim < 0 || ndim + 1 > POLY_MAX_DIMS) return NULL;
  int64_t cur_shape[POLY_MAX_DIMS];
  int cur_ndim = uop_shape(ctx, x, cur_shape);
  if (cur_ndim != ndim + 1) return NULL;
  int64_t pads[POLY_MAX_DIMS][2];
  for (int i = 0; i < ndim; i++) {
    if (cur_shape[i] > self_shape[i]) return NULL;
    pads[i][0] = 0;
    pads[i][1] = self_shape[i] - cur_shape[i];
  }
  pads[ndim][0] = 0;
  pads[ndim][1] = 0;
  return poly_pad(ctx, x, pads, ndim + 1);
}

static PolyUOp *poly_scatter_one_hot(
    PolyCtx *ctx,
    PolyUOp *index,
    const int64_t *self_shape,
    const int64_t *index_shape,
    int ndim,
    int dim
) {
  if (!ctx || !index || !self_shape || !index_shape || ndim < 0 || ndim + 1 > POLY_MAX_DIMS)
    return NULL;

  int arange_dtype_id = (self_shape[dim] > (int64_t)INT32_MAX) ? poly_dtype_id_by_name("int64")
                                                               : poly_dtype_id_by_name("int32");
  PolyUOp *ar = poly_arange_int_by_id(ctx, 0, self_shape[dim], 1, arange_dtype_id);
  if (!ar) return NULL;

  PolyUOp *index_u = poly_unsqueeze_axis(ctx, index, -1);
  if (!index_u) return NULL;

  int64_t ar_shape[POLY_MAX_DIMS];
  for (int i = 0; i < ndim; i++)
    ar_shape[i] = 1;
  ar_shape[ndim] = self_shape[dim];
  PolyUOp *ar_r = poly_reshape(ctx, ar, ar_shape, ndim + 1);
  if (!ar_r) return NULL;

  PolyUOp *mask = poly_eq(ctx, index_u, ar_r);
  if (!mask) return NULL;

  int64_t perm[POLY_MAX_DIMS];
  for (int i = 0; i < ndim + 1; i++)
    perm[i] = i;
  perm[dim] = ndim;
  perm[ndim] = dim;
  (void)index_shape;
  return poly_permute(ctx, mask, perm, ndim + 1);
}

static bool poly_prepare_scatter(
    PolyCtx *ctx,
    PolyUOp *self,
    int dim,
    PolyUOp *index,
    PolyUOp *src,
    PolyScatterPrepared *out
) {
  if (!ctx || !self || !index || !src || !out) return false;
  int64_t self_shape[POLY_MAX_DIMS], index_shape[POLY_MAX_DIMS], src_shape[POLY_MAX_DIMS];
  int ndim = uop_shape(ctx, self, self_shape);
  int index_ndim = uop_shape(ctx, index, index_shape);
  int src_ndim = uop_shape(ctx, src, src_shape);
  if (ndim < 0 || index_ndim != ndim || src_ndim != ndim || ndim + 1 > POLY_MAX_DIMS) return false;
  if (dim < 0) dim += ndim;
  if (dim < 0 || dim >= ndim) return false;
  if (!poly_dtype_eq(poly_dtype_scalar(self->dtype), poly_dtype_scalar(src->dtype))) return false;

  for (int d = 0; d < ndim; d++) {
    if (d != dim && self_shape[d] < index_shape[d]) return false;
    if (src_shape[d] < index_shape[d]) return false;
  }

  int64_t ends[POLY_MAX_DIMS];
  for (int d = 0; d < ndim; d++)
    ends[d] = index_shape[d];
  PolyUOp *src_s = poly_shrink_to(ctx, src, ends, ndim);
  if (!src_s) return false;

  PolyUOp *src_u = poly_unsqueeze_axis(ctx, src_s, -1);
  if (!src_u) return false;
  int64_t src_exp_shape[POLY_MAX_DIMS];
  for (int i = 0; i < ndim; i++)
    src_exp_shape[i] = index_shape[i];
  src_exp_shape[ndim] = self_shape[dim];
  src_u = poly_expand(ctx, src_u, src_exp_shape, ndim + 1);
  if (!src_u) return false;

  int64_t perm[POLY_MAX_DIMS];
  for (int i = 0; i < ndim + 1; i++)
    perm[i] = i;
  perm[dim] = ndim;
  perm[ndim] = dim;
  PolyUOp *src_t = poly_permute(ctx, src_u, perm, ndim + 1);
  if (!src_t) return false;
  src_t = poly_pad_to_scatter_self(ctx, src_t, self_shape, ndim);
  if (!src_t) return false;

  PolyUOp *mask = poly_scatter_one_hot(ctx, index, self_shape, index_shape, ndim, dim);
  if (!mask) return false;
  mask = poly_pad_to_scatter_self(ctx, mask, self_shape, ndim);
  if (!mask) return false;

  memcpy(out->self_shape, self_shape, sizeof(int64_t) * (size_t)ndim);
  out->ndim = ndim;
  out->src = src_t;
  out->mask = mask;
  return true;
}

static PolyUOp *poly_scatter_masked_merge(
    PolyCtx *ctx,
    PolyUOp *self,
    PolyUOp *values,
    PolyUOp *mask,
    const int64_t *self_shape,
    int ndim
) {
  if (!ctx || !self || !values || !mask || !self_shape || ndim < 0) return NULL;
  int64_t mask_shape[POLY_MAX_DIMS];
  int mask_ndim = uop_shape(ctx, mask, mask_shape);
  if (mask_ndim != ndim + 1) return NULL;
  int64_t dup = mask_shape[ndim];
  if (dup <= 0) return NULL;

  PolyUOp *acc_val = NULL;
  PolyUOp *acc_mask = NULL;
  for (int64_t k = 0; k < dup; k++) {
    int64_t pairs[POLY_MAX_DIMS][2];
    for (int i = 0; i < ndim; i++) {
      pairs[i][0] = 0;
      pairs[i][1] = self_shape[i];
    }
    pairs[ndim][0] = k;
    pairs[ndim][1] = k + 1;

    PolyUOp *mk = poly_shrink(ctx, mask, pairs, ndim + 1);
    PolyUOp *vk = poly_shrink(ctx, values, pairs, ndim + 1);
    if (!mk || !vk) return NULL;
    mk = poly_reshape(ctx, mk, (int64_t *)self_shape, ndim);
    vk = poly_reshape(ctx, vk, (int64_t *)self_shape, ndim);
    if (!mk || !vk) return NULL;

    if (!acc_val) {
      acc_val = vk;
      acc_mask = mk;
    } else {
      acc_val = poly_where_op(ctx, mk, vk, acc_val);
      acc_mask = poly_alu2(ctx, POLY_OP_OR, acc_mask, mk);
    }
    if (!acc_val || !acc_mask) return NULL;
  }
  return poly_where_op(ctx, acc_mask, acc_val, self);
}

static PolyUOp *poly_reduce_last_drop(
    PolyCtx *ctx,
    PolyOps op,
    PolyUOp *x,
    const int64_t *full_shape,
    int ndim
) {
  if (!ctx || !x || !full_shape || ndim < 0) return NULL;
  int64_t axes[1] = {ndim};
  PolyUOp *r = poly_reduce_axis(ctx, op, x, axes, 1);
  if (!r) return NULL;
  return poly_reshape(ctx, r, (int64_t *)full_shape, ndim);
}

PolyUOp *poly_scatter_reduce(
    PolyCtx *ctx,
    PolyUOp *self,
    int dim,
    PolyUOp *index,
    PolyUOp *src,
    const char *reduce,
    int include_self
) {
  if (!reduce) return NULL;
  PolyScatterPrepared p = {0};
  if (!poly_prepare_scatter(ctx, self, dim, index, src, &p)) return NULL;

  PolyDType dt = poly_dtype_scalar(src->dtype);
  PolyUOp *zero = poly_const_typed(ctx, dt, 0.0);
  PolyUOp *one = poly_const_typed(ctx, dt, 1.0);
  if (!zero || !one) return NULL;

  PolyUOp *mask_i = poly_where_op(
      ctx, p.mask, poly_const_exact_int(ctx, POLY_INT32, 1),
      poly_const_exact_int(ctx, POLY_INT32, 0)
  );
  PolyUOp *count = poly_reduce_last_drop(ctx, POLY_OP_ADD, mask_i, p.self_shape, p.ndim);
  if (!count) return NULL;
  PolyUOp *no_hit = poly_eq(ctx, count, poly_const_exact_int(ctx, POLY_INT32, 0));
  if (!no_hit) return NULL;

  if (strcmp(reduce, "sum") == 0 || strcmp(reduce, "mean") == 0) {
    PolyUOp *selected = poly_where_op(ctx, p.mask, p.src, zero);
    PolyUOp *sum = poly_reduce_last_drop(ctx, POLY_OP_ADD, selected, p.self_shape, p.ndim);
    if (!sum) return NULL;
    PolyUOp *base = include_self ? self : poly_where_op(ctx, no_hit, self, zero);
    PolyUOp *total = poly_add(ctx, sum, base);
    if (strcmp(reduce, "sum") == 0) return total;

    PolyUOp *inc = include_self ? poly_const_exact_int(ctx, POLY_INT32, 1)
                                : poly_where_op(
                                      ctx, no_hit, poly_const_exact_int(ctx, POLY_INT32, 1),
                                      poly_const_exact_int(ctx, POLY_INT32, 0)
                                  );
    PolyUOp *den = poly_add(ctx, count, inc);
    den = poly_cast(ctx, den, dt);
    return poly_div(ctx, total, den);
  }

  if (strcmp(reduce, "prod") == 0) {
    PolyUOp *selected = poly_where_op(ctx, p.mask, p.src, one);
    PolyUOp *prod = poly_reduce_last_drop(ctx, POLY_OP_MUL, selected, p.self_shape, p.ndim);
    if (!prod) return NULL;
    PolyUOp *base = include_self ? self : poly_where_op(ctx, no_hit, self, one);
    return poly_mul(ctx, prod, base);
  }

  if (strcmp(reduce, "amax") == 0 || strcmp(reduce, "amin") == 0) {
    bool is_min = strcmp(reduce, "amin") == 0;
    PolyUOp *fill = NULL;
    if (!poly_dtype_bound_const(ctx, dt, !is_min, &fill)) return NULL;
    PolyUOp *selected = poly_where_op(ctx, p.mask, p.src, fill);
    PolyUOp *reduced = NULL;
    if (is_min) {
      /* Pinned scatter amin calls Tensor.min, whose integer path is
       * inverse/MAX/inverse (mixin/__init__.py:1206-1211,
       * mixin/elementwise.py:379-393). */
      PolyUOp *inverse = minimum_inverse(ctx, selected);
      PolyUOp *max_inverse =
          inverse ? poly_reduce_last_drop(
                        ctx, POLY_OP_MAX, inverse, p.self_shape, p.ndim
                    )
                  : NULL;
      reduced = max_inverse ? minimum_inverse(ctx, max_inverse) : NULL;
    } else
      reduced = poly_reduce_last_drop(ctx, POLY_OP_MAX, selected, p.self_shape, p.ndim);
    if (!reduced) return NULL;

    PolyUOp *base = include_self ? self : poly_where_op(ctx, no_hit, self, fill);
    return is_min ? poly_minimum(ctx, reduced, base) : poly_maximum(ctx, reduced, base);
  }

  return NULL;
}

PolyUOp *poly_scatter(
    PolyCtx *ctx,
    PolyUOp *self,
    int dim,
    PolyUOp *index,
    PolyUOp *src,
    const char *reduce
) {
  if (reduce && strcmp(reduce, "add") == 0)
    return poly_scatter_reduce(ctx, self, dim, index, src, "sum", 1);
  if (reduce && strcmp(reduce, "multiply") == 0)
    return poly_scatter_reduce(ctx, self, dim, index, src, "prod", 1);
  if (reduce && reduce[0]) return NULL;

  PolyScatterPrepared p = {0};
  if (!poly_prepare_scatter(ctx, self, dim, index, src, &p)) return NULL;
  return poly_scatter_masked_merge(ctx, self, p.src, p.mask, p.self_shape, p.ndim);
}

PolyTensor *poly_tensor_scatter(
    PolyCtx *ctx,
    PolyTensor *self,
    int dim,
    PolyTensor *index,
    PolyTensor *src,
    const char *reduce
) {
  if (!tensor_roots_owned_by_ctx(ctx, self) || !tensor_roots_owned_by_ctx(ctx, index) ||
      !tensor_roots_owned_by_ctx(ctx, src))
    return NULL;
  /* Pinned scatter composes _pre_scatter + _masked_merge directly from the
   * ordered Tensor.uop inputs (mixin/__init__.py:1158-1174,1217-1258).
   * The Tensor boundary applies the unchanged raw program independently to both roots. */
  PolyUOp *logical =
      poly_scatter(ctx, self->uop_logical, dim, index->uop_logical, src->uop_logical, reduce);
  PolyUOp *physical = poly_scatter(
      ctx, self->uop_physical, dim, index->uop_physical, src->uop_physical, reduce
  );
  PolyTensor *inputs[3] = {self, index, src};
  return tensor_composite_result(ctx, logical, physical, inputs, 3);
}

PolyTensor *poly_tensor_scatter_reduce(
    PolyCtx *ctx,
    PolyTensor *self,
    int dim,
    PolyTensor *index,
    PolyTensor *src,
    const char *reduce,
    int include_self
) {
  if (!tensor_roots_owned_by_ctx(ctx, self) || !tensor_roots_owned_by_ctx(ctx, index) ||
      !tensor_roots_owned_by_ctx(ctx, src))
    return NULL;
  /* Pinned scatter_reduce runs the same reducer over exact ordered Tensor.uop
   * inputs (mixin/__init__.py:1176-1215). Keep the raw program identical. */
  PolyUOp *logical = poly_scatter_reduce(
      ctx, self->uop_logical, dim, index->uop_logical, src->uop_logical, reduce, include_self
  );
  PolyUOp *physical = poly_scatter_reduce(
      ctx, self->uop_physical, dim, index->uop_physical, src->uop_physical, reduce, include_self
  );
  PolyTensor *inputs[3] = {self, index, src};
  return tensor_composite_result(ctx, logical, physical, inputs, 3);
}

PolyUOp *poly_gather(PolyCtx *ctx, PolyUOp *table, PolyUOp *indices) {
  if (!ctx || !table || !indices) return NULL;
  int64_t table_shape[POLY_MAX_DIMS], idx_shape[POLY_MAX_DIMS];
  int table_ndim = uop_shape(ctx, table, table_shape);
  int idx_ndim = uop_shape(ctx, indices, idx_shape);
  if (table_ndim != 2 || idx_ndim < 1) return NULL;

  int64_t V = table_shape[0];
  int64_t D = table_shape[1];

  /* tinygrad tensor.py:_one_hot_along_dim chooses int32 unless num_classes
   * overflows int32, then int64. Keep gather on the same integer class range
   * so later collapse stages can see the same one-hot compare shape. */
  int arange_dtype_id =
      (V > (int64_t)INT32_MAX) ? poly_dtype_id_by_name("int64") : poly_dtype_id_by_name("int32");
  PolyUOp *arange_buf = poly_arange_int_by_id(ctx, 0, V, 1, arange_dtype_id);
  if (!arange_buf) return NULL;

  int64_t idx_us_shape[POLY_MAX_DIMS];
  int idx_us_ndim = idx_ndim + 1;
  for (int i = 0; i < idx_ndim; i++)
    idx_us_shape[i] = idx_shape[i];
  idx_us_shape[idx_ndim] = 1;
  PolyUOp *idx_us = poly_reshape(ctx, indices, idx_us_shape, idx_us_ndim);

  int64_t idx_bcast[POLY_MAX_DIMS];
  for (int i = 0; i < idx_ndim; i++)
    idx_bcast[i] = idx_shape[i];
  idx_bcast[idx_ndim] = V;
  PolyUOp *idx_exp = poly_expand(ctx, idx_us, idx_bcast, idx_us_ndim);

  int64_t arange_shape[POLY_MAX_DIMS];
  int arange_ndim = idx_us_ndim;
  for (int i = 0; i < idx_ndim; i++)
    arange_shape[i] = 1;
  arange_shape[idx_ndim] = V;
  PolyUOp *arange_r = poly_reshape(ctx, arange_buf, arange_shape, arange_ndim);
  PolyUOp *arange_exp = poly_expand(ctx, arange_r, idx_bcast, arange_ndim);

  PolyUOp *mask = poly_eq(ctx, idx_exp, arange_exp);

  int64_t mask_us_shape[POLY_MAX_DIMS];
  int mask_us_ndim = idx_us_ndim + 1;
  for (int i = 0; i < idx_us_ndim; i++)
    mask_us_shape[i] = idx_bcast[i];
  mask_us_shape[idx_us_ndim] = 1;
  PolyUOp *mask_us = poly_reshape(ctx, mask, mask_us_shape, mask_us_ndim);

  int64_t mask_bcast[POLY_MAX_DIMS];
  for (int i = 0; i < idx_us_ndim; i++)
    mask_bcast[i] = idx_bcast[i];
  mask_bcast[idx_us_ndim] = D;
  PolyUOp *mask_exp = poly_expand(ctx, mask_us, mask_bcast, mask_us_ndim);

  int64_t tbl_shape[POLY_MAX_DIMS];
  int tbl_ndim = mask_us_ndim;
  for (int i = 0; i < idx_ndim; i++)
    tbl_shape[i] = 1;
  tbl_shape[idx_ndim] = V;
  tbl_shape[idx_ndim + 1] = D;
  PolyUOp *tbl_r = poly_reshape(ctx, table, tbl_shape, tbl_ndim);
  PolyUOp *tbl_exp = poly_expand(ctx, tbl_r, mask_bcast, tbl_ndim);

  PolyUOp *zero = cf(ctx, tbl_exp, 0.0);
  PolyUOp *selected = poly_where_op(ctx, mask_exp, tbl_exp, zero);

  int64_t reduce_axes[] = {idx_ndim};
  PolyUOp *gathered = poly_reduce_axis(ctx, POLY_OP_ADD, selected, reduce_axes, 1);

  int64_t out_shape[POLY_MAX_DIMS];
  int out_ndim = idx_ndim + 1;
  for (int i = 0; i < idx_ndim; i++)
    out_shape[i] = idx_shape[i];
  out_shape[idx_ndim] = D;

  return poly_reshape(ctx, gathered, out_shape, out_ndim);
}

/* Additional composed ops */

PolyUOp *poly_rope(PolyCtx *ctx, PolyUOp *x, PolyUOp *freqs_cos, PolyUOp *freqs_sin) {
  int64_t shape[POLY_MAX_DIMS];
  int ndim;
  ndim = uop_shape(ctx, x, shape);
  if (ndim < 1) return NULL;
  int64_t half_dim = shape[ndim - 1] / 2;
  if (half_dim <= 0) return NULL;

  int64_t pairs1[POLY_MAX_DIMS][2], pairs2[POLY_MAX_DIMS][2];
  for (int i = 0; i < ndim - 1; i++) {
    pairs1[i][0] = 0;
    pairs1[i][1] = shape[i];
    pairs2[i][0] = 0;
    pairs2[i][1] = shape[i];
  }
  pairs1[ndim - 1][0] = 0;
  pairs1[ndim - 1][1] = half_dim;
  pairs2[ndim - 1][0] = half_dim;
  pairs2[ndim - 1][1] = shape[ndim - 1];

  PolyUOp *x1 = poly_shrink(ctx, x, pairs1, ndim);
  PolyUOp *x2 = poly_shrink(ctx, x, pairs2, ndim);

  PolyUOp *r1 = poly_alu2(
      ctx, POLY_OP_SUB, poly_alu2(ctx, POLY_OP_MUL, x1, freqs_cos),
      poly_alu2(ctx, POLY_OP_MUL, x2, freqs_sin)
  );
  PolyUOp *r2 = poly_alu2(
      ctx, POLY_OP_ADD, poly_alu2(ctx, POLY_OP_MUL, x2, freqs_cos),
      poly_alu2(ctx, POLY_OP_MUL, x1, freqs_sin)
  );

  int64_t pad1[POLY_MAX_DIMS][2], pad2[POLY_MAX_DIMS][2];
  for (int i = 0; i < ndim; i++) {
    pad1[i][0] = 0;
    pad1[i][1] = 0;
    pad2[i][0] = 0;
    pad2[i][1] = 0;
  }
  pad1[ndim - 1][1] = half_dim;
  pad2[ndim - 1][0] = half_dim;

  return poly_alu2(ctx, POLY_OP_ADD, poly_pad(ctx, r1, pad1, ndim), poly_pad(ctx, r2, pad2, ndim));
}

PolyUOp *poly_repeat_interleave(PolyCtx *ctx, PolyUOp *x, int repeats, int dim) {
  int64_t shape[POLY_MAX_DIMS];
  int ndim;
  ndim = uop_shape(ctx, x, shape);
  if (ndim < 1 || repeats <= 0) return NULL;
  if (dim < 0) dim += ndim;
  if (dim < 0 || dim >= ndim) return NULL;

  int64_t ins[POLY_MAX_DIMS];
  int ins_ndim = ndim + 1;
  for (int i = 0; i <= dim; i++)
    ins[i] = shape[i];
  ins[dim + 1] = 1;
  for (int i = dim + 1; i < ndim; i++)
    ins[i + 1] = shape[i];
  PolyUOp *r = poly_reshape(ctx, x, ins, ins_ndim);

  int64_t exp[POLY_MAX_DIMS];
  memcpy(exp, ins, ins_ndim * sizeof(int64_t));
  exp[dim + 1] = repeats;
  r = poly_expand(ctx, r, exp, ins_ndim);

  int64_t flat[POLY_MAX_DIMS];
  for (int i = 0; i < dim; i++)
    flat[i] = shape[i];
  flat[dim] = shape[dim] * repeats;
  for (int i = dim + 1; i < ndim; i++)
    flat[i] = shape[i];
  return poly_reshape(ctx, r, flat, ndim);
}

PolyUOp *poly_argmax(PolyCtx *ctx, PolyUOp *x, int axis, int keepdim) {
  int64_t shape[POLY_MAX_DIMS];
  int ndim;
  ndim = uop_shape(ctx, x, shape);
  if (ndim < 1) return NULL;
  if (axis < 0) axis += ndim;
  if (axis < 0 || axis >= ndim) return NULL;

  int64_t N = shape[axis];

  int64_t max_shape[POLY_MAX_DIMS];
  int max_ndim;
  PolyUOp *x_max = do_reduce(ctx, POLY_OP_MAX, x, shape, ndim, axis, 1, max_shape, &max_ndim);
  /* Pinned `self.eq(self.max(..., keepdim=True))` delegates broadcast to
   * `_broadcast_to`, which returns the reduced root unchanged when a
   * singleton axis already gives the input shape (mixin/__init__.py:959,
   * mixin/movement.py:116-128). */
  PolyUOp *m = poly_eq(ctx, x, x_max);

  /* Pinned tinygrad/mixin/__init__.py:944-963:
   *   m = self.eq(self.max(axis=axis, keepdim=True))
   *   idx = m * arange(N, 0, -1).reshape(N, *[1] * trailing)
   *   (N - idx.max(axis=axis, keepdim=keepdim)).cast(int32)
   * `arange` is default_int and Tensor promotion casts the bool mask to it. */
  PolyUOp *m_i = poly_cast(ctx, m, POLY_INT32);
  PolyUOp *desc = poly_arange_int_by_id(ctx, N, 0, -1, 6);
  int desc_ndim = ndim - axis;
  int64_t desc_shape[POLY_MAX_DIMS];
  desc_shape[0] = N;
  for (int i = 1; i < desc_ndim; i++)
    desc_shape[i] = 1;
  /* Tensor.reshape returns self when the requested shape is unchanged
   * (mixin/movement.py:145-161). arange already has shape (N,). */
  if (desc_ndim != 1) desc = poly_reshape(ctx, desc, desc_shape, desc_ndim);
  PolyUOp *idx = poly_mul(ctx, m_i, desc);
  if (!idx) return NULL;

  int64_t idx_max_shape[POLY_MAX_DIMS];
  int idx_max_ndim;
  PolyUOp *idx_max = do_reduce(
      ctx, POLY_OP_MAX, idx, shape, ndim, axis, keepdim != 0, idx_max_shape, &idx_max_ndim
  );
  PolyUOp *result = poly_sub(ctx, poly_const_int(ctx, N), idx_max);
  return poly_cast(ctx, result, POLY_INT32);
}

PolyTensor *poly_tensor_argmax(PolyCtx *ctx, PolyTensor *src, int axis, bool keepdim) {
  if (!tensor_roots_owned_by_ctx(ctx, src)) return NULL;
  PolyUOp *logical = poly_argmax(ctx, src->uop_logical, axis, keepdim);
  PolyUOp *physical = poly_argmax(ctx, src->uop_physical, axis, keepdim);
  if (!logical || !physical) return NULL;
  PolyTensor *out = tensor_unary_result(ctx, src, logical, physical);
  if (!out) return NULL;
  out->requires_grad = false;
  out->requires_grad_set = true;
  return out;
}

PolyUOp *poly_mse_loss(PolyCtx *ctx, PolyUOp *pred, PolyUOp *target) {
  int64_t shape[POLY_MAX_DIMS];
  int ndim;
  ndim = uop_shape(ctx, pred, shape);
  if (ndim < 0) return NULL;
  PolyUOp *diff = poly_alu2(ctx, POLY_OP_SUB, pred, target);
  PolyUOp *sq = poly_alu2(ctx, POLY_OP_MUL, diff, diff);
  int64_t out_shape[POLY_MAX_DIMS];
  int out_ndim;
  PolyUOp *r = sq;
  for (int i = ndim - 1; i >= 0; i--) {
    int64_t s[POLY_MAX_DIMS];
    int sn;
    sn = uop_shape(ctx, r, s);
    r = do_reduce(ctx, POLY_OP_ADD, r, s, sn, i, 0, out_shape, &out_ndim);
  }
  int64_t numel = 1;
  for (int i = 0; i < ndim; i++)
    numel *= shape[i];
  return poly_alu2(ctx, POLY_OP_FDIV, r, poly_const_float(ctx, (double)numel));
}

PolyUOp *poly_mae_loss(PolyCtx *ctx, PolyUOp *pred, PolyUOp *target) {
  int64_t shape[POLY_MAX_DIMS];
  int ndim;
  ndim = uop_shape(ctx, pred, shape);
  if (ndim < 0) return NULL;
  PolyUOp *diff = poly_alu2(ctx, POLY_OP_SUB, pred, target);
  PolyUOp *absdiff = poly_abs(ctx, diff);
  int64_t out_shape[POLY_MAX_DIMS];
  int out_ndim;
  PolyUOp *r = absdiff;
  for (int i = ndim - 1; i >= 0; i--) {
    int64_t s[POLY_MAX_DIMS];
    int sn;
    sn = uop_shape(ctx, r, s);
    r = do_reduce(ctx, POLY_OP_ADD, r, s, sn, i, 0, out_shape, &out_ndim);
  }
  int64_t numel = 1;
  for (int i = 0; i < ndim; i++)
    numel *= shape[i];
  return poly_alu2(ctx, POLY_OP_FDIV, r, poly_const_float(ctx, (double)numel));
}
