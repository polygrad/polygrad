/* C implementation of tinygrad schedule/allreduce.py. */

#include "schedule/allreduce.h"

#include "ctx.h"
#include "device.h"
#include "tensor.h"
#include "utils.h"

#include <limits.h>

static bool allreduce_static_shape(
    PolyCtx *ctx,
    PolyUOp *u,
    int64_t dims[POLY_MAX_DIMS],
    int *ndim_out
) {
  int ndim = poly_uop_ndim(ctx, u);
  if (ndim < 0 || ndim > POLY_MAX_DIMS) return false;
  for (int i = 0; i < ndim; i++) {
    PolyUOp *dim = poly_uop_shape_dim(ctx, u, i);
    if (poly_uop_const_i64(dim, &dims[i]) != 0 || dims[i] < 0) return false;
  }
  *ndim_out = ndim;
  return true;
}

static bool allreduce_shape_product(const int64_t *dims, int ndim, int64_t *out) {
  int64_t product = 1;
  for (int i = 0; i < ndim; i++) {
    if (__builtin_mul_overflow(product, dims[i], &product)) return false;
  }
  *out = product;
  return true;
}
static bool allreduce_uses_naive(int n_devices, int64_t numel) {
  int ring = poly_getenv_int("RING", 1);
  int all2all = poly_getenv_int("ALL2ALL", 0);
  int threshold = poly_getenv_int("RING_ALLREDUCE_THRESHOLD", 256000);
  bool use_all2all = all2all >= 2 || (n_devices > 2 && numel > threshold && all2all >= 1);
  bool use_ring = !use_all2all && (ring >= 2 || (n_devices > 2 && numel > threshold && ring >= 1));
  return !use_ring && !use_all2all;
}

/* Exact naive handle_allreduce branch from schedule/allreduce.py:6-18:
 * materialize the tuple input, copy every selected occurrence to the target
 * device identity, and reduce the copies with the requested operator. */
static PolyUOp *handle_allreduce_naive(PolyCtx *ctx, PolyUOp *buf, PolyUOp *red) {
  if (!ctx || !buf || !red || red->op != POLY_OP_ALLREDUCE || red->n_src != 1 ||
      red->arg.kind != POLY_ARG_ALLREDUCE)
    return NULL;
  PolyUOp *source_device = poly_uop_device_uop_cached(ctx, buf, NULL);
  if (!source_device || source_device->arg.kind != POLY_ARG_STRING_TUPLE ||
      source_device->arg.string_tuple.n <= 0)
    return NULL;

  /* UOp.contiguous returns PARAM/BUFFER identities unchanged
   * (uop/ops.py:587-591); materializing this shaped PARAM would add three
   * spurious kernels to the recursive collective body. */
  PolyUOp *contiguous = poly_contiguous(ctx, buf);
  PolyUOp *target_device = poly_uop_device_uop_cached(ctx, red, NULL);
  PolyUOp *reduced = NULL;
  for (int i = 0; contiguous && target_device && i < source_device->arg.string_tuple.n; i++) {
    PolyUOp *selected = poly_uop1(ctx, POLY_OP_MSELECT, buf->dtype, contiguous, poly_arg_int(i));
    PolyUOp *copy = selected ? poly_copy_to_device_uop(ctx, selected, target_device) : NULL;
    reduced =
        !reduced ? copy : (copy ? poly_alu2(ctx, red->arg.allreduce.op, reduced, copy) : NULL);
  }
  return reduced;
}

/* Exact create_allreduce_function topology from schedule/allreduce.py:57-62.
 * The output clone and shaped PARAMs remain ordinary UOps; the existing CALL
 * scheduler owns lowering and execution. */
PolyUOp *poly_create_allreduce_function(PolyCtx *ctx, PolyUOp *red) {
  if (!ctx || !red || red->op != POLY_OP_ALLREDUCE || red->n_src != 1 ||
      red->arg.kind != POLY_ARG_ALLREDUCE || !red->src[0])
    return NULL;

  int64_t shape[POLY_MAX_DIMS];
  int ndim = 0;
  if (!allreduce_static_shape(ctx, red, shape, &ndim)) return NULL;
  int64_t numel = 0;
  if (!allreduce_shape_product(shape, ndim, &numel)) return NULL;
  PolyUOp *source_device = poly_uop_device_uop_cached(ctx, red->src[0], NULL);
  if (!source_device || source_device->arg.kind != POLY_ARG_STRING_TUPLE ||
      source_device->arg.string_tuple.n <= 0 ||
      !allreduce_uses_naive(source_device->arg.string_tuple.n, numel))
    return NULL;

  PolyUOp *target_device = poly_uop_device_uop_cached(ctx, red, NULL);
  PolyUOp *output =
      target_device
          ? poly_uop_new_buffer(ctx, target_device, numel, red->dtype, poly_ctx_next_unique_id(ctx))
          : NULL;
  if (output && (ndim != 1 || shape[0] != numel)) output = poly_reshape(ctx, output, shape, ndim);

  PolyUOp *invalid = poly_uop_const(ctx, poly_arg_invalid(), red->dtype);
  PolyUOp *invalid_shaped = invalid;
  if (invalid_shaped && ndim > 0) {
    int64_t ones[POLY_MAX_DIMS];
    for (int i = 0; i < ndim; i++)
      ones[i] = 1;
    invalid_shaped = poly_reshape(ctx, invalid_shaped, ones, ndim);
    if (invalid_shaped) invalid_shaped = poly_expand(ctx, invalid_shaped, shape, ndim);
  }
  PolyUOp *output_store =
      output && invalid_shaped
          ? poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, output, invalid_shaped, poly_arg_none())
          : NULL;
  PolyUOp *output_after =
      output_store
          ? poly_uop2(ctx, POLY_OP_AFTER, red->dtype, output, output_store, poly_arg_none())
          : NULL;

  PolyUOp *to = poly_uop_param(ctx, 0, red);
  PolyUOp *src = poly_uop_param(ctx, 1, red->src[0]);
  PolyUOp *param_red = src ? poly_uop1(ctx, POLY_OP_ALLREDUCE, red->dtype, src, red->arg) : NULL;
  PolyUOp *reduced = param_red ? handle_allreduce_naive(ctx, src, param_red) : NULL;
  PolyUOp *store =
      to && reduced ? poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, to, reduced, poly_arg_none()) : NULL;
  PolyUOp *after =
      store ? poly_uop2(ctx, POLY_OP_AFTER, red->dtype, to, store, poly_arg_none()) : NULL;
  PolyUOp *body = after ? poly_sink1(ctx, after) : NULL;
  PolyUOp *contiguous = poly_contiguous(ctx, red->src[0]);
  PolyUOp *call_src[3] = {body, output_after, contiguous};
  PolyUOp *call =
      body && output_after && contiguous
          ? poly_uop(ctx, POLY_OP_CALL, POLY_VOID, call_src, 3, poly_arg_str("allreduce"))
          : NULL;
  return call ? poly_uop2(ctx, POLY_OP_AFTER, red->dtype, output_after, call, poly_arg_none())
              : NULL;
}

/* Pinned schedule/rangeify.py:138-149 resolve_function. FUNCTION bodies are
 * opaque to the surrounding rewrite, then substituted explicitly with a
 * single-pass walk: replacement arguments are not recursively rewritten, and
 * nested CALL/FUNCTION bodies remain opaque. */
