/* device.c -- Buffer operations */

#include "device.h"
#include "ctx.h"
#include "utils.h"
#include "engine/schedule.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static _Thread_local PolyFrontendBufferReleaseFn g_frontend_buffer_release = NULL;

void poly_set_frontend_buffer_release(PolyFrontendBufferReleaseFn fn) {
  g_frontend_buffer_release = fn;
}

void poly_frontend_buffer_release_key(uintptr_t buffer_key) {
  if (g_frontend_buffer_release) g_frontend_buffer_release(buffer_key);
}

static PolyFrontendBufferReleaseFn frontend_buffer_release_for_ctx(PolyCtx *ctx) {
  if (ctx && ctx->frontend_buffer_release) return ctx->frontend_buffer_release;
  return g_frontend_buffer_release;
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
      .frontend_release = NULL,
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

typedef struct {
  const char *name;
  PolyDevice device;
} PolyDeviceNameEntry;

static const PolyDeviceNameEntry POLY_DEVICE_NAMES[] = {
    {"auto", POLY_DEVICE_AUTO},   {"host", POLY_DEVICE_HOST},
    {"cpu", POLY_DEVICE_CPU},     {"interp", POLY_DEVICE_INTERP},
    {"wasm", POLY_DEVICE_WASM},   {"webgpu", POLY_DEVICE_WEBGPU},
    {"cuda", POLY_DEVICE_CUDA},   {"hip", POLY_DEVICE_HIP},
    {"x64", POLY_DEVICE_X64_JIT}, {"x64_jit", POLY_DEVICE_X64_JIT},
};

PolyDevice poly_device_by_name(const char *name) {
  if (!name || !name[0]) return POLY_DEVICE_AUTO;
  for (size_t i = 0; i < sizeof(POLY_DEVICE_NAMES) / sizeof(POLY_DEVICE_NAMES[0]); i++)
    if (strcmp(name, POLY_DEVICE_NAMES[i].name) == 0) return POLY_DEVICE_NAMES[i].device;
  return POLY_DEVICE_AUTO;
}

const char *poly_device_name(PolyDevice device) {
  for (size_t i = 0; i < sizeof(POLY_DEVICE_NAMES) / sizeof(POLY_DEVICE_NAMES[0]); i++)
    if (POLY_DEVICE_NAMES[i].device == device) return POLY_DEVICE_NAMES[i].name;
  return "auto";
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
  if (ndim < 0 || ndim > POLY_MAX_DIMS || (ndim > 0 && !dims)) return NULL;
  PolyDType scalar;
  if (!poly_dtype_by_id(dtype_id, &scalar)) return NULL;
  /* Infer numel from shape when provided; fall back to nbytes / itemsize. */
  int64_t numel = 1;
  if (dims && ndim > 0) {
    for (int i = 0; i < ndim; i++) {
      if (dims[i] < 0) return NULL;
      if (dims[i] != 0 && numel > INT64_MAX / dims[i]) return NULL;
      numel *= dims[i];
    }
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
      .frontend_release =
          ((PolyDevice)device == POLY_DEVICE_HOST) ? frontend_buffer_release_for_ctx(ctx) : NULL,
  };
  poly_map_set(ctx->buffers, poly_ptr_hash(buf), buf, h, poly_ptr_eq);
}

void poly_buffer_attach(PolyCtx *ctx, PolyUOp *buf, const PolyBuffer *handle) {
  if (!ctx || !buf || !handle) return;
  PolyBuffer *old = poly_buffer_get(ctx, buf);
  if (old) poly_buffer_free_chain(old);

  PolyBuffer *h = poly_arena_alloc(ctx->arena, sizeof(PolyBuffer), _Alignof(PolyBuffer));
  *h = *handle;
  h->owned = false;
  h->src = NULL;
  h->valid = true;
  h->frontend_release = NULL;
  if (!h->allocator) {
    const PolyBackendDesc *be = poly_backend_get(h->device);
    h->allocator = be ? be->get_allocator() : NULL;
  }
  poly_map_set(ctx->buffers, poly_ptr_hash(buf), buf, h, poly_ptr_eq);
}

void poly_buffer_adopt(PolyCtx *ctx, PolyUOp *buf, const PolyBuffer *handle) {
  if (!ctx || !buf || !handle) return;
  PolyBuffer *old = poly_buffer_get(ctx, buf);
  if (old) poly_buffer_free_chain(old);

  PolyBuffer *h = poly_arena_alloc(ctx->arena, sizeof(PolyBuffer), _Alignof(PolyBuffer));
  *h = *handle;
  h->frontend_release = NULL;
  if (!h->allocator) {
    const PolyBackendDesc *be = poly_backend_get(h->device);
    h->allocator = be ? be->get_allocator() : NULL;
  }
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
  bool dst_host_addressable = dst_alloc->host_addressable;
  bool src_host_addressable = src_alloc->host_addressable;
  if (dst_alloc == src_alloc && dst_alloc->copy_between) {
    rc = dst_alloc->copy_between(dst, src, nbytes, dst_alloc->dev_ctx);
  } else if (dst_host_addressable && src_alloc->copy_out) {
    rc = src_alloc->copy_out(dst, src, nbytes, src_alloc->dev_ctx);
  } else if (src_host_addressable && dst_alloc->copy_in) {
    rc = dst_alloc->copy_in(dst, src, nbytes, dst_alloc->dev_ctx);
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

static size_t poly_buffer_nbytes_for_uop(PolyUOp *buf) {
  if (!buf) return 0;
  int64_t numel = (buf->arg.kind == POLY_ARG_INT) ? buf->arg.i : 0;
  size_t itemsize = poly_dtype_itemsize(poly_dtype_scalar(buf->dtype));
  size_t nbytes = (numel > 0) ? (size_t)numel * itemsize : 0;
  return nbytes ? nbytes : sizeof(float);
}

static int poly_buffer_alloc_residency(
    PolyCtx *ctx,
    PolyUOp *buf,
    PolyDevice device,
    size_t nbytes,
    bool valid,
    PolyBuffer *src,
    PolyBuffer **out
) {
  if (!ctx || !buf || !out || device == POLY_DEVICE_AUTO || device == POLY_DEVICE_HOST) return -1;
  const PolyBackendDesc *be = poly_backend_get(device);
  if (!be) {
    fprintf(stderr, "poly_buffer_allocate: no backend for device %d\n", device);
    return -1;
  }
  if (poly_backend_ensure_open(device) != 0) {
    fprintf(stderr, "poly_buffer_allocate: backend '%s' failed to open\n", be->name);
    return -1;
  }
  const PolyAllocator *alloc = be->get_allocator();
  if (!alloc) return -1;
  void *ptr = alloc->alloc(nbytes, alloc->dev_ctx);
  if (!ptr) {
    fprintf(stderr, "poly_buffer_allocate: alloc(%zu) failed\n", nbytes);
    return -1;
  }

  PolyBuffer *h = poly_arena_alloc(ctx->arena, sizeof(PolyBuffer), _Alignof(PolyBuffer));
  *h = (PolyBuffer){
      .ptr = ptr,
      .nbytes = nbytes,
      .device = device,
      .owned = true,
      .allocator = alloc,
      .src = src,
      .valid = valid,
      .frontend_release = NULL,
  };
  *out = h;
  return 0;
}

static int poly_buffer_alloc_host_root(PolyCtx *ctx, size_t nbytes, bool valid, PolyBuffer **out) {
  if (!ctx || !out) return -1;
  PolyDevice device = poly_device_default();
  const PolyBackendDesc *be = poly_backend_get(device);
  const PolyAllocator *alloc = be ? be->get_allocator() : NULL;
  if (!alloc || !alloc->host_addressable || !alloc->alloc) return -1;
  void *ptr = alloc->alloc(nbytes, alloc->dev_ctx);
  if (!ptr) return -1;
  PolyBuffer *h = poly_arena_alloc(ctx->arena, sizeof(PolyBuffer), _Alignof(PolyBuffer));
  *h = (PolyBuffer){
      .ptr = ptr,
      .nbytes = nbytes,
      .device = device,
      .owned = true,
      .allocator = alloc,
      .src = NULL,
      .valid = valid,
      .frontend_release = NULL,
  };
  *out = h;
  return 0;
}

static bool poly_buffer_is_host_root(const PolyBuffer *b) {
  return b && !b->src && poly_device_is_host_addressable(b->device);
}

static PolyBuffer *poly_buffer_retained_root(PolyBuffer *b) {
  if (!b) return NULL;
  PolyBuffer *root = b;
  while (root->src)
    root = root->src;
  return poly_device_is_host_addressable(root->device) ? root : NULL;
}

static void poly_buffer_free_except_root(PolyBuffer *b, PolyBuffer *root) {
  while (b && b != root) {
    PolyBuffer *next = b->src;
    b->src = NULL;
    poly_buffer_free(b);
    b = next;
  }
}

int poly_buffer_ensure_host_current(PolyCtx *ctx, PolyUOp *buf, PolyBuffer **host_out) {
  if (host_out) *host_out = NULL;
  if (!ctx || !buf) return -1;
  PolyBuffer *cur = poly_buffer_get(ctx, buf);
  if (!cur) return -1;

  if (poly_buffer_is_host_root(cur)) {
    if (host_out) *host_out = cur;
    return cur->valid ? 0 : -1;
  }

  PolyBuffer *root = cur->src;
  if (root && poly_device_is_host_addressable(root->device)) {
    if (!root->valid) {
      if (!cur->valid || poly_buffer_copy(root, cur) != 0) return -1;
    }
    if (host_out) *host_out = root;
    return 0;
  }

  size_t nbytes = cur->nbytes ? cur->nbytes : poly_buffer_nbytes_for_uop(buf);
  if (poly_buffer_alloc_host_root(ctx, nbytes, false, &root) != 0) return -1;
  if (cur->valid) {
    if (poly_buffer_copy(root, cur) != 0) {
      poly_buffer_free(root);
      return -1;
    }
  }
  cur->src = root;
  if (host_out) *host_out = root;
  return root->valid ? 0 : -1;
}

int poly_buffer_alloc_owned_host(
    PolyCtx *ctx,
    PolyUOp *buf,
    size_t nbytes,
    bool zero,
    PolyBuffer **host_out
) {
  if (host_out) *host_out = NULL;
  if (!ctx || !buf || nbytes == 0) return -1;

  PolyBuffer *existing = poly_buffer_get(ctx, buf);
  if (existing) {
    PolyBuffer *host = NULL;
    if (poly_buffer_ensure_host_current(ctx, buf, &host) != 0 || !host || host->nbytes < nbytes)
      return -1;
    if (host_out) *host_out = host;
    return 0;
  }

  PolyBuffer *host = NULL;
  if (poly_buffer_alloc_host_root(ctx, nbytes, true, &host) != 0) return -1;
  if (zero && host->ptr) memset(host->ptr, 0, nbytes);
  poly_map_set(ctx->buffers, poly_ptr_hash(buf), buf, host, poly_ptr_eq);
  if (host_out) *host_out = host;
  return 0;
}

static int poly_buffer_ensure_host_root_for_write(
    PolyCtx *ctx,
    PolyUOp *buf,
    PolyBuffer **host_out
) {
  if (host_out) *host_out = NULL;
  if (!ctx || !buf) return -1;
  PolyBuffer *cur = poly_buffer_get(ctx, buf);
  if (!cur) {
    size_t nbytes = poly_buffer_nbytes_for_uop(buf);
    PolyBuffer *root = NULL;
    if (poly_buffer_alloc_host_root(ctx, nbytes, false, &root) != 0) return -1;
    poly_map_set(ctx->buffers, poly_ptr_hash(buf), buf, root, poly_ptr_eq);
    if (host_out) *host_out = root;
    return 0;
  }
  if (poly_buffer_is_host_root(cur)) {
    if (host_out) *host_out = cur;
    return 0;
  }
  if (cur->src && poly_device_is_host_addressable(cur->src->device)) {
    if (host_out) *host_out = cur->src;
    return 0;
  }
  PolyBuffer *root = NULL;
  size_t nbytes = cur->nbytes ? cur->nbytes : poly_buffer_nbytes_for_uop(buf);
  if (poly_buffer_alloc_host_root(ctx, nbytes, false, &root) != 0) return -1;
  cur->src = root;
  if (host_out) *host_out = root;
  return 0;
}

int poly_buffer_mark_host_written(PolyCtx *ctx, PolyUOp *buf) {
  PolyBuffer *root = NULL;
  if (poly_buffer_ensure_host_root_for_write(ctx, buf, &root) != 0 || !root) return -1;
  root->valid = true;
  PolyBuffer *cur = poly_buffer_get(ctx, buf);
  if (cur && cur != root && cur->src == root) cur->valid = false;
  return 0;
}

int poly_buffer_write(PolyCtx *ctx, PolyUOp *buf, const void *src, size_t nbytes) {
  if (!ctx || !buf || !src) return -1;
  PolyBuffer *root = NULL;
  if (poly_buffer_ensure_host_root_for_write(ctx, buf, &root) != 0 || !root || !root->ptr)
    return -1;
  if (nbytes == 0 || root->nbytes < nbytes) return -1;
  memcpy(root->ptr, src, nbytes);
  root->valid = true;
  PolyBuffer *cur = poly_buffer_get(ctx, buf);
  if (cur && cur != root && cur->src == root) cur->valid = false;
  return 0;
}

int poly_buffer_ensure_device_allocated(PolyCtx *ctx, PolyUOp *buf, PolyDevice device) {
  if (!ctx || !buf) return -1;
  if (device == POLY_DEVICE_AUTO) device = poly_device_default();
  if (device == POLY_DEVICE_HOST) device = poly_device_default();
  PolyBuffer *existing = poly_buffer_get(ctx, buf);
  if (existing && poly_devices_share_storage(existing->device, device) && existing->ptr) {
    existing->device = device;
    return 0;
  }

  size_t nbytes = existing && existing->nbytes ? existing->nbytes : poly_buffer_nbytes_for_uop(buf);
  PolyBuffer *src = poly_buffer_retained_root(existing);
  PolyBuffer *dst = NULL;
  if (poly_buffer_alloc_residency(ctx, buf, device, nbytes, false, src, &dst) != 0) return -1;
  poly_buffer_free_except_root(existing, src);
  poly_map_set(ctx->buffers, poly_ptr_hash(buf), buf, dst, poly_ptr_eq);
  return 0;
}

static const PolyBuffer *poly_buffer_valid_source(PolyBuffer *cur) {
  if (!cur) return NULL;
  if (cur->valid) return cur;
  if (cur->src && cur->src->valid) return cur->src;
  return NULL;
}

int poly_buffer_ensure_device_current(PolyCtx *ctx, PolyUOp *buf, PolyDevice device) {
  if (!ctx || !buf) return -1;
  if (device == POLY_DEVICE_AUTO) device = poly_device_default();
  if (device == POLY_DEVICE_HOST) device = poly_device_default();

  PolyBuffer *cur = poly_buffer_get(ctx, buf);
  if (cur && poly_devices_share_storage(cur->device, device)) {
    cur->device = device;
    if (cur->valid) return 0;
    if (!cur->src || !cur->src->valid) return -1;
    return poly_buffer_copy(cur, cur->src);
  }

  const PolyBuffer *source = poly_buffer_valid_source(cur);
  if (!source) return -1;

  size_t nbytes = cur && cur->nbytes ? cur->nbytes : poly_buffer_nbytes_for_uop(buf);
  PolyBuffer *src_root = poly_buffer_retained_root(cur);
  PolyBuffer *dst = NULL;
  if (poly_buffer_alloc_residency(ctx, buf, device, nbytes, false, src_root, &dst) != 0) return -1;
  if (poly_buffer_copy(dst, source) != 0) {
    poly_buffer_free(dst);
    return -1;
  }
  poly_buffer_free_except_root(cur, src_root);
  poly_map_set(ctx->buffers, poly_ptr_hash(buf), buf, dst, poly_ptr_eq);
  return 0;
}

int poly_buffer_mark_residency_written(PolyCtx *ctx, PolyUOp *buf, PolyDevice device) {
  if (!ctx || !buf) return -1;
  PolyBuffer *cur = poly_buffer_get(ctx, buf);
  if (!cur) return -1;
  if (!poly_devices_share_storage(cur->device, device) &&
      !(poly_device_is_host_addressable(cur->device) && poly_device_is_host_addressable(device)))
    return -1;
  cur->valid = true;
  if (cur->src) cur->src->valid = false;
  return 0;
}

int poly_buffer_allocate(PolyCtx *ctx, PolyUOp *buf, PolyDevice device) {
  PolyBuffer *cur = poly_buffer_get(ctx, buf);
  if (cur && (cur->valid || (cur->src && cur->src->valid)))
    return poly_buffer_ensure_device_current(ctx, buf, device);
  return poly_buffer_ensure_device_allocated(ctx, buf, device);
}

int poly_buffer_ensure_allocated(PolyCtx *ctx, PolyUOp *buf, PolyDevice device) {
  PolyBuffer *b = poly_buffer_get(ctx, buf);
  if (b && poly_devices_share_storage(b->device, device) && b->ptr) return 0;
  if (b && (b->valid || (b->src && b->src->valid)))
    return poly_buffer_ensure_device_current(ctx, buf, device);
  return poly_buffer_ensure_device_allocated(ctx, buf, device);
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
  if (!ctx || !buf || !dst) return -1;
  PolyBuffer *host = NULL;
  if (poly_buffer_ensure_host_current(ctx, buf, &host) != 0 || !host || !host->ptr) return -1;
  if (nbytes == 0 || host->nbytes < nbytes) return -1;
  memcpy(dst, host->ptr, nbytes);
  return 0;
}
