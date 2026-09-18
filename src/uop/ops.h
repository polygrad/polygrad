/* uop/ops.h -- UOp representation, construction and graph queries. */
#ifndef POLY_UOP_OPS_H
#define POLY_UOP_OPS_H

#include "../core.h"
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* UOp */

struct PolyUOp {
  PolyOps op;
  PolyDType dtype;
  PolyUOp **src;
  uint16_t n_src;
  PolyArg arg;
  int32_t tag;
  PolyArg tag_arg;
  uint32_t hash;
  bool addrspace_cached;
  PolyAddrSpace addrspace_cache;
  bool minmax_cached;
  int64_t minmax_vmin;
  int64_t minmax_vmax;
  void *ranges_cache;
  void *ended_ranges_cache;
};

/* Derive the backend implementation from an exact DEVICE identity.  This does
 * not imply that every identity of that backend is executable; schedule
 * ingress separately rejects unsupported runtime instances. */
PolyDevice poly_uop_device_from_device_uop(PolyUOp *device);
PolyDevice poly_uop_device(PolyUOp *u);
/* Exact concrete scalar device identity carried by the physical UOp graph.
 * Returns the canonical DEVICE string (for example "CPU:1") or NULL when the
 * graph has no concrete DEVICE identity.  Backend dispatch remains the
 * separate PolyDevice-valued poly_uop_device() compatibility query. */
const char *poly_uop_device_name(PolyCtx *ctx, PolyUOp *u);
/* Complete Tinygrad UOp.device metadata, not backend dispatch. Returns the
 * number of names, or -1 on invalid arguments/query failure. No device is
 * (0,false); an empty tuple is (0,true); a scalar is (1,false). Names and their
 * array are borrowed until context collection/destruction. Copy before a
 * collection safe point; this query does not retain a graph or storage. */
int poly_uop_device_names(PolyCtx *ctx, PolyUOp *u, const char ***names, bool *is_tuple);

/* Upgrade/downgrade a borrowed UOp to an explicit residency and IR owner. */
int poly_uop_retain(PolyCtx *ctx, PolyUOp *uop);
void poly_uop_release(PolyCtx *ctx, PolyUOp *uop);

/* Create a UOp (with CSE deduplication) */
PolyUOp *poly_uop(PolyCtx *ctx, PolyOps op, PolyDType dtype, PolyUOp **src, int n_src, PolyArg arg);

/* Create a UOp with a non-zero tag. Tag is part of the CSE key,
 * so a tagged node is distinct from an untagged node with the same
 * (op, dtype, src, arg). Matches tinygrad's UOp.replace(tag=...). */
PolyUOp *poly_uop_tagged(
    PolyCtx *ctx,
    PolyOps op,
    PolyDType dtype,
    PolyUOp **src,
    int n_src,
    PolyArg arg,
    int32_t tag
);
PolyUOp *poly_uop_tagged_arg(
    PolyCtx *ctx,
    PolyOps op,
    PolyDType dtype,
    PolyUOp **src,
    int n_src,
    PolyArg arg,
    int32_t tag,
    PolyArg tag_arg
);
/* Current Tinygrad UOp.replace(src=...): rebuild one immutable node while
 * preserving its op, dtype, arg, and tag metadata. */
PolyUOp *poly_uop_replace_src(PolyCtx *ctx, PolyUOp *u, PolyUOp **src);

/* Convenience: create a UOp with 0, 1, 2, or 3 sources */
PolyUOp *poly_uop0(PolyCtx *ctx, PolyOps op, PolyDType dtype, PolyArg arg);
PolyUOp *poly_uop1(PolyCtx *ctx, PolyOps op, PolyDType dtype, PolyUOp *s0, PolyArg arg);
PolyUOp *poly_uop2(
    PolyCtx *ctx,
    PolyOps op,
    PolyDType dtype,
    PolyUOp *s0,
    PolyUOp *s1,
    PolyArg arg
);
PolyUOp *poly_uop3(
    PolyCtx *ctx,
    PolyOps op,
    PolyDType dtype,
    PolyUOp *s0,
    PolyUOp *s1,
    PolyUOp *s2,
    PolyArg arg
);

/* Toposort: returns arena-allocated array of UOp pointers, sets *n_out.
 * _ex variant: gate callback (NULL=visit all, return false to skip subtree),
 * enter_calls (false = skip CALL/FUNCTION src[0], process src[1:] only).
 * _ex_user variant: gate carries user_data (closure-style, mirrors tinygrad's
 * `u.toposort(gate=lambda x: r in x.ranges)` where r is captured). */
PolyUOp **poly_uop_toposort(PolyCtx *ctx, PolyUOp *root, int *n_out);
PolyUOp **poly_uop_toposort_ex(
    PolyCtx *ctx,
    PolyUOp *root,
    int *n_out,
    bool (*gate)(PolyUOp *),
    bool enter_calls
);
PolyUOp **poly_uop_toposort_ex_user(
    PolyCtx *ctx,
    PolyUOp *root,
    int *n_out,
    bool (*gate)(PolyUOp *, void *),
    void *user_data,
    bool enter_calls
);

/* Owned toposort variants for local scans. These mirror tinygrad's temporary
 * `u.toposort()` result lifetime: the returned array is heap-owned and must be
 * released with poly_uop_toposort_free(). UOp nodes themselves remain ctx-owned.
 * `ctx` may be NULL because owned traversal allocates no arena/scratch data. */
PolyUOp **poly_uop_toposort_alloc(PolyCtx *ctx, PolyUOp *root, int *n_out);
PolyUOp **poly_uop_toposort_ex_alloc(
    PolyCtx *ctx,
    PolyUOp *root,
    int *n_out,
    bool (*gate)(PolyUOp *),
    bool enter_calls
);
PolyUOp **poly_uop_toposort_ex_user_alloc(
    PolyCtx *ctx,
    PolyUOp *root,
    int *n_out,
    bool (*gate)(PolyUOp *, void *),
    void *user_data,
    bool enter_calls
);
void poly_uop_toposort_free(PolyUOp **topo);

/* Per-pass cache for UOp queries (ranges, vmin/vmax) *
 * Tinygrad caches every queryable UOp property as @functools.cached_property
 * on the immutable UOp instance, which gives per-UOp-lifetime memoization
 * for free. Polygrad's UOps are also immutable and hash-consed,
 * and the default min/max query caches directly on the UOp. Callers may still
 * pass their own PolyUOpCache to scope batch/rewrite-local range queries and
 * throw those maps away cleanly.
 *
 * PolyUOpCache unifies the query maps that Phase D's reduce_collapse
 * driver uses together:
 *   - minmax: UOp inline cached (int64_t vmin, int64_t vmax), matching
 *     tinygrad's cached UOp._min_max property
 *   - ranges: UOp -> set of active RANGE ancestors per tinygrad u.ranges
 *   - ended_ranges: UOp -> set of ended RANGE ancestors per tinygrad u.ended_ranges
 *
 * Without caching, the minmax computation is exponential on diamond DAGs
 * (MUL alone is a 4-corner recurrence; hash-consed graphs like
 * `arange(n).reshape(n,1) - arange(n).reshape(1,n)` compound that with
 * shared subexpressions). The same holds for the ranges-set walk on any
 * graph where many ancestors query the same subtree.
 *
 * Usage:
 *   PolyUOpCache *c = poly_uop_cache_new();
 *   poly_uop_minmax_ex(ctx, u, c, &lo, &hi);
 *   bool ok = poly_uop_no_range_ex(ctx, u2, c);
 *   ... more queries reusing c ...
 *   poly_uop_cache_destroy(c);
 *
 * Lifetime: minmax values live inline on immutable UOps. Range-set values
 * are allocated from the PolyCtx arena and live until the ctx is destroyed,
 * matching tinygrad's UOp-lifetime cached properties. poly_uop_cache_destroy
 * frees the range-query PolyMap wrappers only. The minmax `_ex` entry accepts
 * a cache argument for API symmetry but does not allocate cache-owned values.
 * Cache invalidation is NOT automatic — if the graph is mutated via
 * poly_uop_substitute between queries, destroy and recreate the cache. */

typedef struct PolyUOpCache PolyUOpCache;

PolyUOpCache *poly_uop_cache_new(void);
void poly_uop_cache_destroy(PolyUOpCache *c);

/* Range helpers. `poly_uop_no_range` matches tinygrad codegen/simplify.py:75:
 *   def no_range(u): return not any(x.op is Ops.RANGE for x in u.backward_slice_with_self)
 *
 * `poly_uop_ranges` / `poly_uop_in_ranges` mirror tinygrad uop/ops.py:362-378:
 *   ranges(u) = union(ranges(s) for s in u.src) - ended_ranges(u) + ({u} if RANGE)
 *
 * where ended_ranges() matches ops.py:351-358 (trailing srcs past range_start,
 * AFTER recursively flattens effect dependencies. See src/uop/ops.c.
 * Range-set allocation failure follows PolyMap's fatal-OOM policy: these
 * count/bool APIs must not report a successful empty set after failure.
 *
 * Every helper has a public entry point and an `_ex` variant that takes a
 * caller-owned PolyUOpCache for batch queries. Use the `_ex` form in hot loops
 * that need a pass-local cache distinct from UOp-local cached properties. */
/* Returns the terminal buffer-identity UOp (BUFFER / PARAM)
 * after unwrapping RESHAPE/UNSHARD, or NULL for other forms. This does not
 * resolve MSELECT lanes; use the buffer APIs for their runtime storage. */
const PolyUOp *poly_uop_get_buffer_identity(const PolyUOp *u);

PolyUOp *poly_uop_base(PolyUOp *u);
PolyUOp *poly_uop_unsharded_base(PolyUOp *u);
bool poly_uop_op_in_backward_slice_with_self(PolyCtx *ctx, PolyUOp *u, PolyOps op);

/* Current Tinygrad UOp.buf_uop: return the storage-state UOp used by
 * scheduling and access analysis, preserving MSELECT/MSTACK structure. */
PolyUOp *poly_uop_buf_uop(PolyCtx *ctx, PolyUOp *u);

/* Current Tinygrad UOp.has_buffer_identity. */
bool poly_uop_has_buffer_identity(const PolyUOp *u);

/* tinygrad UOp.buffer analogue. Returns a direct buffer identity, or the exact
 * movement UOp with an attached runtime view when a contiguous movement over
 * realized storage is provable. It does not create/rewrite graph topology or
 * change a Tensor root. */
PolyUOp *poly_uop_buffer(PolyCtx *ctx, PolyUOp *u);

/* True when target appears in root's source graph. Frontends use this to match
 * tinygrad's backward discovery rule: live tensors with t.uop in loss.toposort. */
bool poly_uop_reachable(PolyCtx *ctx, PolyUOp *root, PolyUOp *target);

bool poly_uop_no_range(PolyCtx *ctx, PolyUOp *u);
bool poly_uop_no_range_ex(PolyCtx *ctx, PolyUOp *u, PolyUOpCache *cache);
bool poly_uop_in_ranges(PolyCtx *ctx, PolyUOp *u, PolyUOp *r);
bool poly_uop_in_ranges_ex(PolyCtx *ctx, PolyUOp *u, PolyUOp *r, PolyUOpCache *cache);
/* Copy the complete active-range set; -1 on invalid arguments or insufficient
 * capacity, without writing any output. Range-cache OOM remains fatal. */
int poly_uop_ranges(PolyCtx *ctx, PolyUOp *u, PolyUOp **out, int max_out);
int poly_uop_ranges_ex(PolyCtx *ctx, PolyUOp *u, PolyUOp **out, int max_out, PolyUOpCache *cache);

/* Compact integer projection of tinygrad UOp._min_max, cached on the UOp.
 *
 * The int64 result cannot represent fractional/infinite or full uint64 bounds.
 * Unknown endpoints fall back to dtype-derived limits; those are not a proof
 * that an arbitrary uint64 value fits int64. Internal semantic predicates use
 * typed/exact interval queries where this distinction matters. Derived integer
 * bounds from wide uint64 operands are computed exactly before projection.
 *
 * Overflow: corner multiplications and shifts use __builtin_*_overflow
 * detection and fall through to dtype bounds on overflow. This can produce
 * loose (but conservative and correct) intervals for pathological inputs;
 * practical Phase D workloads stay far below int64 saturation.
 *
 * Parity: verified against test/parity_scripts/tg_minmax_gt.py. */
void poly_uop_minmax(PolyCtx *ctx, PolyUOp *u, int64_t *vmin, int64_t *vmax);
void poly_uop_minmax_ex(
    PolyCtx *ctx,
    PolyUOp *u,
    PolyUOpCache *cache,
    int64_t *vmin,
    int64_t *vmax
);
/* Pinned tinygrad uop/ops.py:resolve: simplify a boolean UOp, return its
 * proven value when constant, otherwise the caller-provided default. */
int poly_uop_resolve(PolyCtx *ctx, PolyUOp *u, int default_value);

/* Pretty-print a UOp graph to a buffer (returns malloc'd string, caller frees) */
char *poly_uop_str(PolyUOp *u);
char *poly_uop_graph_str(PolyUOp *root);

/* Recursive indented IR tree dump to a FILE*. Used by passes (rangeify,
 * codegen, reduce_simplify) to print pre/post-rewrite IR for diagnostics.
 * Caps recursion at max_depth (suggested: 12-16 for full graphs). */
void poly_uop_dump_tree(FILE *fp, PolyUOp *u, int depth, int max_depth);

/* Shape */
/* ndim == -1 means "no tensor shape" (kernel-level ops like RANGE, LOAD) */

typedef struct {
  int64_t *dims; /* ownership is defined by the producing shape API */
  int ndim; /* -1 = no shape, 0 = scalar, >0 = tensor */
} PolyShape;

#define POLY_SHAPE_NONE ((PolyShape){NULL, -1})
#define POLY_MAX_DIMS 16

PolyShape poly_uop_max_shape(PolyCtx *ctx, PolyUOp *u);
int64_t poly_shape_numel(PolyShape s);
/* Current tinygrad UOp.max_numel(): product of the concrete maximum shape. */
int64_t poly_uop_max_numel(PolyCtx *ctx, const PolyUOp *u);
bool poly_shape_eq(PolyShape a, PolyShape b);

/* Lazy cached shape accessors -- computes on first access, O(1) thereafter.
 * Returned dims are borrowed until the next collection safe point. */
int poly_uop_ndim(PolyCtx *ctx, const PolyUOp *u);
const int64_t *poly_uop_max_shape_dims(PolyCtx *ctx, const PolyUOp *u);
PolyUOp *poly_uop_shape_dim(PolyCtx *ctx, const PolyUOp *u, int dim);
/* Current tinygrad uop/ops.py:shape_to_shape_arg. */
PolyUOp *poly_shape_to_shape_arg(PolyCtx *ctx, PolyUOp **items, int n_items);
/* Current tinygrad uop/ops.py:_broadcast_shape over UOp source shapes. */
int poly_uop_broadcast_shape(
    PolyCtx *ctx,
    PolyUOp **src,
    int n_src,
    PolyUOp **out_dims,
    int max_dims
);
/* Current tinygrad UOp.as_shape: one non-STACK UOp is one dimension and STACK
 * exposes its ordered sources. Items are symbolically simplified UOps so C
 * consumers retain exact symbolic shape expressions. Returns item count or -1. */
int poly_uop_as_shape(PolyCtx *ctx, PolyUOp *shape_arg, PolyUOp **items, int max_items);
/* Exact C port of tinygrad/uop/ops.py:broadcast_axes. Returns the number of
 * output axes that are added/expanded, or -1 for incompatible ranks. */
int poly_uop_broadcast_axes(
    PolyCtx *ctx,
    const PolyUOp *src,
    const PolyUOp *out,
    int *axes,
    int max_axes
);
/* Pinned UOp.axis query for multi-device shard propagation
 * (tinygrad/uop/ops.py:623-651). Returns false when the value is unsharded. */
bool poly_uop_axis(PolyCtx *ctx, const PolyUOp *u, int *out_axis);
/* Current tinygrad UOp.unshard: preserve ordered shard axes and their RANGE
 * sources in the tensor graph (`tinygrad/uop/ops.py:667-681`). */
PolyUOp *poly_uop_unshard(
    PolyCtx *ctx,
    PolyUOp *value,
    const int64_t *axes,
    PolyUOp **ranges,
    int n_axes
);
/* Current tinygrad UOp.allreduce(op, device). */
PolyUOp *poly_uop_allreduce(PolyCtx *ctx, PolyUOp *value, PolyOps op, PolyUOp *device);
/* Current tinygrad UOp.range (`tinygrad/uop/ops.py:563-565`). */
PolyUOp *poly_uop_range(PolyCtx *ctx, int64_t bound, int64_t axis_id, PolyAxisType axis_type);
int poly_uop_const_i64(const PolyUOp *u, int64_t *out);
PolyUOp *poly_uop_unbind_var(PolyUOp *u);
int poly_uop_bind_value(PolyUOp *u, int64_t *out);
/* Current tinygrad UOp.contiguous_view_offset. */
int poly_uop_contiguous_view_offset(PolyCtx *ctx, PolyUOp *u, int64_t *out);
PolyShape poly_uop_max_shape_cached(PolyCtx *ctx, const PolyUOp *u);

/* Current tinygrad/uop/ops.py:axis_letters and range_str. */
char poly_uop_axis_letter(PolyArg arg);
char *poly_uop_range_str(PolyArg arg);
/* Compare the complete (axis id, split path, AxisType) range key. */
int poly_uop_range_arg_cmp(PolyArg a, PolyArg b);

/* UOp.vconst_like: a scalar or flat STACK after movement lowering. */
PolyUOp *poly_uop_vconst_like(PolyCtx *ctx, PolyUOp *ref, PolyArg val);

/* Heap-backed result of current Tinygrad UOp.split_uop. Caller frees it. */
PolyUOp **poly_uop_split(PolyUOp *u, PolyOps sep, int *n_out);

/* Complete UOp.ranges in insertion order. Borrowed immutable cache storage:
 * keep u alive, do not free/sort the array, and do not cross a GC safe point
 * without retaining u. Empty sets still return a non-NULL array. */
PolyUOp *const *poly_uop_ranges_view(PolyCtx *ctx, PolyUOp *u, int *n_out);

/* UOp.divides: prove exact divisibility structurally, not by sampled bounds. */
PolyUOp *poly_uop_divides(PolyCtx *ctx, PolyUOp *u, int64_t factor);

/* Tinygrad helpers.py:is_image_shape on the UOp's encoded shape. */
bool poly_uop_is_image_shape(PolyCtx *ctx, const PolyUOp *u);

/* Tinygrad 2026-08-22/a9069c177a9d uop/ops.py:UOp.get_idx/get_valid. */
PolyUOp *poly_uop_get_idx(PolyCtx *ctx, PolyUOp *u);
PolyUOp *poly_uop_get_valid(PolyCtx *ctx, PolyUOp *u);

/* Tinygrad 2026-08-22/a9069c177a9d uop/ops.py:UOp.addrspace. */
bool poly_uop_addrspace(const PolyUOp *u, PolyAddrSpace *out);

/* UOp.key content identity, excluding tags. Caller frees the returned bytes;
 * NULL disables persistent caching for unsupported process-local payloads. */
uint8_t *poly_uop_key(PolyCtx *ctx, PolyUOp *root, size_t *size);

/* C mechanics for Tinygrad's weak UOpMetaClass.ucache and Python-owned UOps. */
int poly_uop_cse_evict_unmarked(PolyCtx *ctx, PolyMap *live);
bool poly_uop_storage_contains(PolyCtx *ctx, const void *ptr);
void poly_uop_storage_destroy_all(PolyCtx *ctx);

/* tinygrad's callified function uses value PARAMs as external storage
 * identities until rangeify lowers them to kernel pointer PARAMs. */
static inline bool poly_uop_is_shaped_value_param(const PolyUOp *u) {
  return u && u->op == POLY_OP_PARAM && u->arg.kind == POLY_ARG_PARAM && u->arg.param &&
         !u->arg.param->name && u->n_src == 1 && u->src[0] &&
         (u->src[0]->op == POLY_OP_STACK || poly_dtype_is_int(u->src[0]->dtype));
}

/* Pass-local backing cache for pinned UOp.axis. NULL/false is represented
 * explicitly in the map so shared unsharded subgraphs are not revisited. */
bool poly_uop_axis_cached(PolyCtx *ctx, const PolyUOp *u, PolyMap *cache, int *out_axis);

/* Apply one substitution map to several roots with one pass-local rewrite
 * memo. This is the allocation-free-root equivalent of tinygrad substituting
 * one temporary SINK: shared UOps are rewritten once, but no aggregate UOp is
 * interned in Polygrad's ctx-lifetime arena/CSE. */
int poly_uop_substitute_many(
    PolyCtx *ctx,
    PolyUOp **roots,
    int n_roots,
    PolyUOp **from,
    PolyUOp **to,
    int n,
    PolyUOp **out
);

#ifdef POLY_TESTING
/* Fail one nonempty substitution after count successful calls; -1 disables. */
void poly_uop_test_substitute_fail_after(int count);
#endif

/* Substitute UOps in a graph: replace from[i] with to[i] for i in [0,n).
 * Returns a new root UOp with substitutions applied. Used to reconnect
 * realized intermediate buffers back to their original computation graphs
 * before calling poly_uop_grad. */
PolyUOp *poly_uop_substitute(PolyCtx *ctx, PolyUOp *root, PolyUOp **from, PolyUOp **to, int n);

/* UOp construction helpers */

int poly_op_count(void);

PolyUOp *poly_uop_const_float(PolyCtx *ctx, double value);

PolyUOp *poly_uop_const_double(PolyCtx *ctx, double value);

PolyUOp *poly_uop_const_int(PolyCtx *ctx, int64_t value);

PolyUOp *poly_uop_const(PolyCtx *ctx, PolyArg value, PolyDType dtype);

PolyUOp *poly_uop_const_typed(PolyCtx *ctx, PolyDType dt, double value);

PolyUOp *poly_uop_const_like_dtype(PolyCtx *ctx, PolyUOp *ref, PolyArg val, PolyDType dtype);

PolyUOp *poly_uop_const_like(PolyCtx *ctx, PolyUOp *ref, PolyArg val);

PolyUOp *poly_uop_const_like_int(PolyCtx *ctx, PolyUOp *ref, int64_t val);

PolyUOp *poly_uop_const_like_float(PolyCtx *ctx, PolyUOp *ref, double val);

PolyUOp *poly_uop_const_like_bool(PolyCtx *ctx, PolyUOp *ref, bool val);

PolyUOp *poly_uop_identity_element(PolyCtx *ctx, PolyOps op, PolyDType dtype);

PolyUOp *poly_uop_cast(PolyCtx *ctx, PolyUOp *x, PolyDType target);

/* UOp.bitcast keeps identity for an equal dtype; shape validation belongs
 * to the consuming Tensor boundary, not raw graph construction. */
PolyUOp *poly_uop_bitcast(PolyCtx *ctx, PolyUOp *x, PolyDType target);

/* Endpoints are copied; NaN, reversed and nonnumeric bounds are rejected.
 * This metadata domain does not widen the runtime's integer binding ABI. */
PolyUOp *poly_uop_variable(
    PolyCtx *ctx,
    const char *name,
    PolyArg min_val,
    PolyArg max_val,
    PolyDType dtype,
    int64_t multiple_of,
    bool param
);

PolyUOp *poly_uop_param(PolyCtx *ctx, int slot, PolyUOp *like);

bool poly_uop_is_variable(const PolyUOp *u);

bool poly_uop_is_bound_var(const PolyUOp *u);

bool poly_uop_is_alu_param(const PolyUOp *u);

const char *poly_uop_expr(const PolyUOp *u);

PolyUOp *poly_uop_bind(PolyCtx *ctx, PolyUOp *var, int64_t value);

/* Current tinygrad uop/ops.py:dtype_from_uop and _rebuild_dtype.  The first
 * returns false for operations whose dtype remains explicitly owned by the
 * node; the second preserves that stored dtype in exactly those cases. */
bool poly_dtype_from_uop(
    PolyOps op,
    PolyUOp **src,
    int n_src,
    PolyArg arg,
    PolyDType current_dtype,
    PolyDType *out
);

PolyDType poly_uop_rebuild_dtype(PolyUOp *u, PolyUOp **new_src);

PolyUOp *poly_uop_alu1(PolyCtx *ctx, PolyOps op, PolyUOp *src);

PolyUOp *poly_uop_alu2(PolyCtx *ctx, PolyOps op, PolyUOp *a, PolyUOp *b);

PolyUOp *poly_uop_alu3(PolyCtx *ctx, PolyOps op, PolyUOp *a, PolyUOp *b, PolyUOp *c);

PolyUOp *poly_uop_store_val(PolyCtx *ctx, PolyUOp *buf, PolyUOp *value);

PolyUOp *poly_uop_sink1(PolyCtx *ctx, PolyUOp *store);

PolyUOp *poly_uop_sink_n(PolyCtx *ctx, PolyUOp **stores, int n);

/* Approved Polygrad portable logical-storage identity. Never executable. */
PolyUOp *poly_uop_new_logical_buffer(PolyCtx *ctx, PolyDType dtype, int64_t size);

PolyUOp *poly_uop_new_logical_buffer_with_slot(
    PolyCtx *ctx,
    PolyDType dtype,
    int64_t size,
    int64_t slot
);

/* Current tinygrad UOp.new_buffer(device, size, dtype, num=slot). */
PolyUOp *poly_uop_new_buffer(
    PolyCtx *ctx,
    PolyUOp *device,
    int64_t size,
    PolyDType dtype,
    int64_t slot
);

/* Current tinygrad UOp.copy_to_device: COPY has one value source and carries
 * the exact scalar/tuple target device in arg. */
PolyUOp *poly_uop_copy_to_device(PolyCtx *ctx, PolyUOp *value, PolyUOp *device);

PolyUOp *poly_uop_stack(PolyCtx *ctx, PolyUOp **src, int n_src);

/* Current UOp.index composition. */
PolyUOp *poly_uop_index(PolyCtx *ctx, PolyUOp *base, PolyUOp **indices, int n_indices);

/* Current tinygrad UOp.placeholder: storage is flat prod(shape), with rank
 * restored by RESHAPE. */
PolyUOp *poly_uop_placeholder(
    PolyCtx *ctx,
    const int64_t *shape,
    int ndim,
    PolyDType dtype,
    int64_t slot,
    PolyAddrSpace addrspace,
    const char *device,
    bool volatile_
);

PolyUOp *poly_uop_reduce_axis(
    PolyCtx *ctx,
    PolyOps reduce_op,
    PolyUOp *src,
    int64_t *axes,
    int n_axes
);

/* Tensor._pad_constant: negative padding shrinks; nonnegative zero padding
 * is equivalent to poly_uop_pad. */
PolyUOp *poly_uop_pad_value(PolyCtx *ctx, PolyUOp *x, int64_t (*pads)[2], int ndim, double value);

/* Shared host-shape checks implemented by shape.c. */
int64_t poly_shape_numel_checked(const int64_t *shape, int ndim);

bool poly_shape_equal(const int64_t *a, int a_ndim, const int64_t *b, int b_ndim);

/* UOp construction helpers used by custom kernels.
 * These only build UOps; execution still flows through normal CALL scheduling. */
PolyUOp *poly_uop_placeholder_like(PolyCtx *ctx, PolyUOp *like, int slot);

PolyUOp *poly_uop_load(PolyCtx *ctx, PolyUOp *addr);

PolyUOp *poly_uop_store(PolyCtx *ctx, PolyUOp *addr, PolyUOp *value);

PolyUOp *poly_uop_set(PolyCtx *ctx, PolyUOp *addr, PolyUOp *value, PolyUOp **ranges, int n_ranges);

PolyUOp *poly_uop_group(PolyCtx *ctx, PolyUOp **srcs, int n_src);

PolyUOp *poly_uop_end(PolyCtx *ctx, PolyUOp *body, PolyUOp **ranges, int n_ranges);

PolyUOp *poly_uop_sink(PolyCtx *ctx, PolyUOp **srcs, int n_src);

PolyUOp *poly_uop_sink_ex(PolyCtx *ctx, PolyUOp **srcs, int n_src, const char *name, int optimize);

PolyUOp *poly_uop_call(PolyCtx *ctx, PolyUOp *body, PolyUOp **args, int n_args);

PolyUOp *poly_uop_after(PolyCtx *ctx, PolyUOp *target, PolyUOp *effect);

PolyUOp *poly_uop_reduce(
    PolyCtx *ctx,
    PolyOps reduce_op,
    PolyUOp *expr,
    PolyUOp **ranges,
    int n_ranges
);

PolyUOp *poly_uop_flatten(PolyCtx *ctx, PolyUOp *u);

int64_t poly_uop_numel(PolyCtx *ctx, PolyUOp *u);

/* Full-buffer STORE effect for optimizer/direct core SINKs.
 * Tensor.assign itself still uses tinygrad's current-value shape:
 * AFTER(target, STORE(target, value)). Direct effect SINKs already sequence
 * stores explicitly, so they should contain STORE(target, value) entries.
 * Movement views are normalized to their base buffer for whole-storage updates. */
PolyUOp *poly_uop_store_buffer_update(PolyCtx *ctx, PolyUOp *target, PolyUOp *value);

#ifdef __cplusplus
}
#endif

#endif
