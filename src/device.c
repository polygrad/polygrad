/* device.c -- Buffer operations */

#include "device.h"
#include "ctx.h"
#include "utils.h"
#include "engine/schedule.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static PolyFrontendBufferReleaseFn g_frontend_buffer_release = NULL;

void poly_set_frontend_buffer_release(PolyFrontendBufferReleaseFn fn) {
  g_frontend_buffer_release = fn;
}

void poly_frontend_buffer_release_key(uintptr_t buffer_key) {
  if (g_frontend_buffer_release) g_frontend_buffer_release(buffer_key);
}

uint64_t poly_buffer_get_key(PolyCtx *ctx, PolyUOp *buf) {
  return (uint64_t)(uintptr_t)poly_buffer_get(ctx, buf);
}

PolyBuffer poly_buffer_make_host_view(void *ptr, size_t nbytes) {
  PolyDevice dev = poly_device_default();
  const PolyBackendDesc *be = poly_backend_get(dev);
  return (PolyBuffer){
      .ptr = ptr,
      .nbytes = nbytes,
      .device = dev,
      .owned = false,
      .allocator = be ? be->get_allocator() : NULL,
      .src = NULL,
      .valid = true,
  };
}

/* Device helpers */

PolyDevice poly_device_default(void) {
#ifdef __EMSCRIPTEN__
  return POLY_DEVICE_WASM;
#else
  return POLY_DEVICE_CPU;
#endif
}

bool poly_device_can_execute(PolyDevice dev) {
  return dev != POLY_DEVICE_AUTO && dev != POLY_DEVICE_HOST;
}

bool poly_devices_share_storage(PolyDevice a, PolyDevice b) {
  if (a == b) return true;
  const PolyBackendDesc *ba = poly_backend_get(a);
  const PolyBackendDesc *bb = poly_backend_get(b);
  if (!ba || !bb) return false;
  const PolyAllocator *aa = ba->get_allocator();
  const PolyAllocator *ab = bb->get_allocator();
  return aa && ab && aa == ab;
}

PolyDevice poly_device_by_name(const char *name) {
  if (!name || !name[0]) return POLY_DEVICE_AUTO;
  if (strcmp(name, "cpu") == 0) return POLY_DEVICE_CPU;
  if (strcmp(name, "interp") == 0) return POLY_DEVICE_INTERP;
  if (strcmp(name, "wasm") == 0) return POLY_DEVICE_WASM;
  if (strcmp(name, "webgpu") == 0) return POLY_DEVICE_WEBGPU;
  if (strcmp(name, "cuda") == 0) return POLY_DEVICE_CUDA;
  if (strcmp(name, "hip") == 0) return POLY_DEVICE_HIP;
  if (strcmp(name, "x64") == 0 || strcmp(name, "x64_jit") == 0) return POLY_DEVICE_X64_JIT;
  if (strcmp(name, "host") == 0) return POLY_DEVICE_HOST;
  return POLY_DEVICE_AUTO;
}

/* Create a BUFFER UOp and attach frontend-owned host bytes in ctx->buffers.
 * Wraps in RESHAPE(BUFFER) when ndim > 1. Frontends must keep ptr alive
 * for as long as the buffer is reachable. */
PolyUOp *poly_buffer_from_host(
    PolyCtx *ctx,
    void *ptr,
    size_t nbytes,
    int dtype_id,
    int64_t *dims,
    int ndim
) {
  if (!ctx) return NULL;
  PolyDType scalar;
  if (!poly_dtype_by_id(dtype_id, &scalar)) return NULL;
  /* Infer numel from shape when provided; fall back to nbytes / itemsize. */
  int64_t numel = 1;
  if (dims && ndim > 0) {
    for (int i = 0; i < ndim; i++) numel *= dims[i];
  } else {
    int isize = poly_dtype_itemsize(scalar);
    numel = (isize > 0) ? (int64_t)(nbytes / (size_t)isize) : 0;
    if (numel < 1) numel = 1;
  }
  PolyUOp *buf = poly_buffer(ctx, scalar, numel);
  if (!buf) return NULL;
  /* Register imported source data in ctx->buffers so graph-driven realize
   * can materialize or reuse it later without external binding arrays. */
  poly_buffer_set(ctx, buf, ptr, nbytes, (int)POLY_DEVICE_HOST);
  /* BUFFER is 1D; a multi-dim tensor needs RESHAPE on top so the scheduler
   * sees the intended shape. */
  if (ndim > 1) return poly_reshape(ctx, buf, dims, ndim);
  return buf;
}

void poly_buffer_free(PolyBuffer *b) {
  if (!b) return;
  if (b->owned && b->allocator && b->allocator->free)
    b->allocator->free(b, b->allocator->dev_ctx);
  b->ptr = NULL;
  b->owned = false;
  b->valid = false;
}

/* Free current residency + retained src. Assumes the no-chains invariant
 * (b->src is a root, never another current). Breaks the link before freeing
 * to avoid recursion surprises, and clears ownership to guard against double-
 * free if `src` is somehow shared (which the design forbids, but defense). */
void poly_buffer_free_chain(PolyBuffer *b) {
  if (!b) return;
  PolyBuffer *src = b->src;
  b->src = NULL;
  poly_buffer_free(b);
  if (src) poly_buffer_free(src);
}

void poly_buffer_set(PolyCtx *ctx, PolyUOp *buf, void *ptr, size_t nbytes, int device) {
  if (!ctx || !buf) return;
  /* Replacing the whole binding for this UOp: free the entire old chain. */
  PolyBuffer *old = poly_buffer_get(ctx, buf);
  if (old) poly_buffer_free_chain(old);

  const PolyBackendDesc *be = poly_backend_get((PolyDevice)device);
  PolyBuffer *h = poly_arena_alloc(ctx->arena, sizeof(PolyBuffer), _Alignof(PolyBuffer));
  *h = (PolyBuffer){
      .ptr = ptr,
      .nbytes = nbytes,
      .device = (PolyDevice)device,
      .owned = ((PolyDevice)device == POLY_DEVICE_HOST),
      .allocator = be ? be->get_allocator() : NULL,
      .src = NULL,
      .valid = true, /* frontend just gave us valid data */
  };
  poly_map_set(ctx->buffers, poly_ptr_hash(buf), buf, h, poly_ptr_eq);
}

void *poly_buffer_get_ptr(PolyCtx *ctx, PolyUOp *buf) {
  PolyBuffer *b = poly_buffer_get(ctx, buf);
  return b ? b->ptr : NULL;
}

PolyBuffer *poly_buffer_get(PolyCtx *ctx, PolyUOp *buf) {
  if (!ctx || !buf) return NULL;
  return poly_map_get(ctx->buffers, poly_ptr_hash(buf), buf, poly_ptr_eq);
}

void poly_buffer_remove(PolyCtx *ctx, PolyUOp *buf) {
  if (!ctx || !buf) return;
  /* Fully discarding this logical buffer: free current + src chain. */
  PolyBuffer *b = poly_buffer_get(ctx, buf);
  if (b) poly_buffer_free_chain(b);
  poly_map_remove(ctx->buffers, poly_ptr_hash(buf), buf, poly_ptr_eq);
}

bool poly_buffer_is_allocated(PolyCtx *ctx, PolyUOp *buf) {
  PolyBuffer *b = poly_buffer_get(ctx, buf);
  return b != NULL && (b->ptr != NULL || b->device == POLY_DEVICE_HOST);
}

static const PolyAllocator *buffer_allocator(const PolyBuffer *b) {
  if (!b) return NULL;
  if (b->allocator) return b->allocator;
  const PolyBackendDesc *be = poly_backend_get(b->device);
  return be ? be->get_allocator() : NULL;
}

int poly_buffer_copy(PolyBuffer *dst, const PolyBuffer *src) {
  if (!dst || !src) return -1;

  const PolyAllocator *dst_alloc = buffer_allocator(dst);
  const PolyAllocator *src_alloc = buffer_allocator(src);
  if (!dst_alloc || !src_alloc) return -1;

  size_t nbytes = dst->nbytes;
  if (src->nbytes > 0 && (nbytes == 0 || src->nbytes < nbytes)) nbytes = src->nbytes;
  if (nbytes == 0) return 0;

  int rc = -1;
  if (dst_alloc == src_alloc && dst_alloc->copy_between) {
    rc = dst_alloc->copy_between(dst, src, nbytes, dst_alloc->dev_ctx);
  } else if (dst_alloc->copy_in) {
    rc = dst_alloc->copy_in(dst, src, nbytes, dst_alloc->dev_ctx);
  } else if (src_alloc->copy_out) {
    rc = src_alloc->copy_out(dst, src, nbytes, src_alloc->dev_ctx);
  } else {
    void *tmp = malloc(nbytes);
    if (!tmp) return -1;
    PolyBuffer tmp_view = poly_buffer_make_host_view(tmp, nbytes);
    rc = src_alloc->copy_out(&tmp_view, src, nbytes, src_alloc->dev_ctx);
    if (rc == 0) rc = dst_alloc->copy_in(dst, &tmp_view, nbytes, dst_alloc->dev_ctx);
    free(tmp);
  }

  if (rc == 0) dst->valid = true;
  return rc;
}

int poly_buffer_allocate(PolyCtx *ctx, PolyUOp *buf, PolyDevice device) {
  if (!ctx || !buf) return -1;
  if (device == POLY_DEVICE_HOST) {
    fprintf(stderr, "poly_buffer_allocate: HOST is source-only and cannot be allocated\n");
    return -1;
  }

  PolyBuffer *existing = poly_buffer_get(ctx, buf);

  /* Already on target device with valid data: no-op */
  if (existing && existing->device == device && existing->valid) return 0;

  const PolyBackendDesc *be = poly_backend_get(device);
  if (!be) {
    fprintf(stderr, "poly_buffer_allocate: no backend for device %d\n", device);
    return -1;
  }
  const PolyAllocator *alloc = be->get_allocator();
  if (!alloc) return -1;

  int64_t numel = buf->arg.i;
  size_t itemsize = poly_dtype_itemsize(poly_dtype_scalar(buf->dtype));
  size_t nbytes = (size_t)numel * itemsize;
  if (nbytes == 0) nbytes = sizeof(float);

  void *ptr = alloc->alloc(nbytes, alloc->dev_ctx);
  if (!ptr) {
    fprintf(stderr, "poly_buffer_allocate: alloc(%zu) failed\n", nbytes);
    return -1;
  }

  /* Determine src: preserve root, no chains. If existing has src, inherit it.
   * Otherwise existing IS the root and becomes the new src. */
  PolyBuffer *new_src = NULL;
  if (existing) {
    new_src = existing->src ? existing->src : existing;
  }

  /* If existing is not becoming the new src, free its owned ptr. */
  if (existing && existing != new_src) {
    poly_buffer_free(existing);
  }

  PolyBuffer *h = poly_arena_alloc(ctx->arena, sizeof(PolyBuffer), _Alignof(PolyBuffer));
  *h = (PolyBuffer){
      .ptr = ptr,
      .nbytes = nbytes,
      .device = device,
      .owned = true,
      .allocator = alloc,
      .src = new_src,
      .valid = false, /* freshly allocated, not yet populated */
  };
  poly_map_set(ctx->buffers, poly_ptr_hash(buf), buf, h, poly_ptr_eq);
  return 0;
}

int poly_buffer_ensure_allocated(PolyCtx *ctx, PolyUOp *buf, PolyDevice device) {
  PolyBuffer *b = poly_buffer_get(ctx, buf);
  if (b && b->device == device && b->valid) return 0;
  return poly_buffer_allocate(ctx, buf, device);
}

int poly_buffer_copyin(PolyCtx *ctx, PolyUOp *buf, const void *src, size_t nbytes) {
  if (!ctx || !buf || !src) return -1;
  PolyBuffer *b = poly_buffer_get(ctx, buf);
  if (!b || !b->ptr || !b->allocator) return -1;
  PolyBuffer src_view = poly_buffer_make_host_view((void *)src, nbytes);
  int rc = b->allocator->copy_in(b, &src_view, nbytes, b->allocator->dev_ctx);
  if (rc == 0) b->valid = true;
  return rc;
}

int poly_buffer_copyout(PolyCtx *ctx, PolyUOp *buf, void *dst, size_t nbytes) {
  if (!ctx || !buf || !dst) return -1;
  PolyBuffer *b = poly_buffer_get(ctx, buf);
  if (!b || !b->ptr || !b->allocator) return -1;
  PolyBuffer dst_view = poly_buffer_make_host_view(dst, nbytes);
  int rc = b->allocator->copy_out(&dst_view, b, nbytes, b->allocator->dev_ctx);
  if (rc == 0 && b->src && dst == b->src->ptr) b->src->valid = true;
  return rc;
}

int poly_buffer_read(PolyCtx *ctx, PolyUOp *buf, void *dst, size_t nbytes) {
  return poly_buffer_copyout(ctx, buf, dst, nbytes);
}
