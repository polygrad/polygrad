/* device.c -- Buffer operations */

#include "device.h"
#include "ctx.h"
#include "utils.h"
#include "engine/schedule.h"
#include "runtime_webgpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#ifndef __EMSCRIPTEN__
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

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
      .memory_accounted = false,
      .memory_device = POLY_DEVICE_AUTO,
  };
}

static void poly_ctx_record_buffer_copy(PolyCtx *ctx, size_t nbytes) {
  if (!ctx || nbytes == 0) return;
  ctx->buffer_copy_count++;
  ctx->buffer_copy_bytes += nbytes;
}

static size_t poly_buffer_copy_nbytes(const PolyBuffer *dst, const PolyBuffer *src) {
  if (!dst || !src) return 0;
  size_t nbytes = dst->nbytes;
  if (src->nbytes > 0 && (nbytes == 0 || src->nbytes < nbytes)) nbytes = src->nbytes;
  return nbytes;
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
  return dev != POLY_DEVICE_AUTO && dev != POLY_DEVICE_HOST && dev != POLY_DEVICE_DISK;
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
    {"x86", POLY_DEVICE_X86},
    {"disk", POLY_DEVICE_DISK},
};

PolyDevice poly_device_by_name(const char *name) {
  if (!name || !name[0]) return POLY_DEVICE_AUTO;
  /* DISK device strings carry an arbitrary filesystem path. Recognize the
   * device prefix before copying short execution-device names into the fixed
   * normalization buffer. */
  if (tolower((unsigned char)name[0]) == 'd' &&
      tolower((unsigned char)name[1]) == 'i' &&
      tolower((unsigned char)name[2]) == 's' &&
      tolower((unsigned char)name[3]) == 'k' && name[4] == ':')
    return POLY_DEVICE_DISK;
  char buf[64];
  size_t n = strlen(name);
  if (n >= sizeof(buf)) return POLY_DEVICE_AUTO;
  for (size_t i = 0; i <= n; i++) buf[i] = (char)tolower((unsigned char)name[i]);
  name = buf;
  if (strncmp(name, "cpu:", 4) == 0) name += 4;
  if (strncmp(name, "disk:", 5) == 0) name = "disk";
  for (size_t i = 0; i < sizeof(POLY_DEVICE_NAMES) / sizeof(POLY_DEVICE_NAMES[0]); i++)
    if (strcmp(name, POLY_DEVICE_NAMES[i].name) == 0) return POLY_DEVICE_NAMES[i].device;
  return POLY_DEVICE_AUTO;
}

const char *poly_device_name(PolyDevice device) {
  for (size_t i = 0; i < sizeof(POLY_DEVICE_NAMES) / sizeof(POLY_DEVICE_NAMES[0]); i++)
    if (POLY_DEVICE_NAMES[i].device == device) return POLY_DEVICE_NAMES[i].name;
  return "auto";
}

static void *disk_alloc(size_t nbytes, void *dev_ctx) {
  (void)nbytes;
  (void)dev_ctx;
  return NULL;
}

static void disk_free_alloc(const PolyBuffer *buffer, void *dev_ctx) {
  (void)dev_ctx;
#ifndef __EMSCRIPTEN__
  if (buffer && buffer->ptr && buffer->nbytes) munmap(buffer->ptr, buffer->nbytes);
#else
  (void)buffer;
#endif
}

static int disk_copy_in(
    const PolyBuffer *dst, const PolyBuffer *src, size_t n, void *dev_ctx
) {
  (void)dev_ctx;
  if (!dst || !dst->ptr || !src || !src->ptr) return -1;
  memcpy(dst->ptr, src->ptr, n);
  return 0;
}

static int disk_copy_out(
    const PolyBuffer *dst, const PolyBuffer *src, size_t n, void *dev_ctx
) {
  (void)dev_ctx;
  if (!dst || !dst->ptr || !src || !src->ptr) return -1;
  memcpy(dst->ptr, src->ptr, n);
  return 0;
}

static int disk_copy_between(
    const PolyBuffer *dst, const PolyBuffer *src, size_t n, void *dev_ctx
) {
  return disk_copy_in(dst, src, n, dev_ctx);
}

static const PolyAllocator POLY_DISK_ALLOCATOR = {
    .alloc = disk_alloc,
    .free = disk_free_alloc,
    .copy_in = disk_copy_in,
    .copy_out = disk_copy_out,
    .copy_between = disk_copy_between,
    .host_addressable = true,
    .dev_ctx = NULL,
};

const PolyAllocator *poly_disk_get_allocator(void) { return &POLY_DISK_ALLOCATOR; }

PolyUOp *poly_buffer_from_file(PolyCtx *ctx, const char *path, int dtype_id) {
  if (!ctx || !path || !path[0]) return NULL;
#ifdef __EMSCRIPTEN__
  (void)dtype_id;
  return NULL;
#else
  PolyDType scalar;
  if (!poly_dtype_by_id(dtype_id, &scalar)) return NULL;
  int itemsize = poly_dtype_itemsize(scalar);
  if (itemsize <= 0) return NULL;

  bool shared = true;
  int fd = open(path, O_RDWR);
  if (fd < 0) {
    shared = false;
    fd = open(path, O_RDONLY);
  }
  if (fd < 0) return NULL;

  struct stat st;
  if (fstat(fd, &st) != 0 || st.st_size < 0) {
    close(fd);
    return NULL;
  }
  uint64_t file_bytes = (uint64_t)st.st_size;
  uint64_t mapped_bytes = file_bytes - (file_bytes % (uint64_t)itemsize);
  if (mapped_bytes > SIZE_MAX || mapped_bytes / (uint64_t)itemsize > INT64_MAX) {
    close(fd);
    return NULL;
  }

  void *ptr = NULL;
  if (mapped_bytes > 0) {
    int flags = shared ? MAP_SHARED : MAP_PRIVATE;
    ptr = mmap(NULL, (size_t)mapped_bytes, PROT_READ | PROT_WRITE, flags, fd, 0);
    if (ptr == MAP_FAILED) {
      close(fd);
      return NULL;
    }
  }
  close(fd);

  int64_t numel = (int64_t)(mapped_bytes / (uint64_t)itemsize);
  PolyUOp *buf = poly_buffer_on_device(ctx, scalar, numel, POLY_DEVICE_DISK);
  if (!buf) {
    if (ptr) munmap(ptr, (size_t)mapped_bytes);
    return NULL;
  }
  PolyBuffer mapped = {
      .ptr = ptr,
      .nbytes = (size_t)mapped_bytes,
      .device = POLY_DEVICE_DISK,
      .owned = mapped_bytes > 0,
      .allocator = &POLY_DISK_ALLOCATOR,
      .src = NULL,
      .valid = true,
      .frontend_release = NULL,
      .memory_accounted = false,
      .memory_device = POLY_DEVICE_DISK,
  };
  poly_buffer_adopt(ctx, buf, &mapped);
  return buf;
#endif
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
  /* Register imported source data in ctx->buffers so graph-driven realize can
   * materialize or reuse it later without external binding arrays. In
   * Emscripten, a nonzero pointer passed by the JS binding already points into
   * the WebAssembly linear memory used by WASM kernels, so adopting it as WASM
   * residency avoids an immediate HOST->WASM copy on every replay. A null
   * pointer is the browser/WebGPU host-key path and must remain HOST. */
#ifdef __EMSCRIPTEN__
  if (ptr) {
    PolyBuffer h = poly_buffer_make_host_view(ptr, nbytes);
    h.owned = true;
    poly_buffer_adopt(ctx, buf, &h);
  } else {
    poly_buffer_set(ctx, buf, ptr, nbytes, (int)POLY_DEVICE_HOST);
  }
#else
  poly_buffer_set(ctx, buf, ptr, nbytes, (int)POLY_DEVICE_HOST);
#endif
  /* BUFFER is 1D; a multi-dim tensor needs RESHAPE on top so the scheduler
   * sees the intended shape. */
  if (ndim > 1) return poly_reshape(ctx, buf, dims, ndim);
  return buf;
}

PolyUOp *poly_buffer_from_host_unique(
    PolyCtx *ctx,
    PolyUOp *unique,
    PolyDType scalar_dtype,
    int64_t numel,
    void *ptr,
    size_t nbytes,
    PolyDevice *out_source_device
) {
  if (!ctx || !unique || unique->op != POLY_OP_UNIQUE || numel < 0) return NULL;
#ifdef __EMSCRIPTEN__
  PolyDevice source_device = ptr ? POLY_DEVICE_WASM : POLY_DEVICE_HOST;
#else
  PolyDevice source_device = POLY_DEVICE_HOST;
#endif
  PolyUOp *device =
      poly_uop0(ctx, POLY_OP_DEVICE, POLY_VOID, poly_arg_int((int64_t)source_device));
  PolyUOp *src[2] = {unique, device};
  PolyUOp *buffer =
      device ? poly_uop(ctx, POLY_OP_BUFFER, scalar_dtype, src, 2, poly_arg_int(numel)) : NULL;
  if (!buffer) return NULL;

#ifdef __EMSCRIPTEN__
  if (ptr) {
    PolyBuffer imported = poly_buffer_make_host_view(ptr, nbytes);
    imported.owned = true;
    poly_buffer_adopt(ctx, buffer, &imported);
  } else {
    poly_buffer_set(ctx, buffer, ptr, nbytes, (int)POLY_DEVICE_HOST);
  }
#else
  poly_buffer_set(ctx, buffer, ptr, nbytes, (int)POLY_DEVICE_HOST);
#endif
  if (!poly_buffer_get(ctx, buffer)) return NULL;
  if (out_source_device) *out_source_device = source_device;
  return buffer;
}

void poly_buffer_free(PolyCtx *ctx, PolyBuffer *b) {
  if (!b) return;
  if (b->owned && b->allocator && b->allocator->free)
    b->allocator->free(b, b->allocator->dev_ctx);
  if (b->memory_accounted)
    poly_ctx_record_memory_free(ctx, b->memory_device, b->nbytes);
  b->ptr = NULL;
  b->owned = false;
  b->valid = false;
  b->memory_accounted = false;
  b->memory_device = POLY_DEVICE_AUTO;
}

/* Free current residency + retained src. Assumes the no-chains invariant
 * (b->src is a root, never another current). Breaks the link before freeing
 * to avoid recursion surprises, and clears ownership to guard against double-
 * free if `src` is somehow shared (which the design forbids, but defense). */
void poly_buffer_free_chain(PolyCtx *ctx, PolyBuffer *b) {
  if (!b) return;
  PolyBuffer *src = b->src;
  b->src = NULL;
  poly_buffer_free(ctx, b);
  if (src) poly_buffer_free(ctx, src);
}

void poly_buffer_set(PolyCtx *ctx, PolyUOp *buf, void *ptr, size_t nbytes, int device) {
  if (!ctx || !buf) return;
  /* Replacing the whole binding for this UOp: free the entire old chain. */
  PolyBuffer *old = poly_buffer_get(ctx, buf);
  if (old) poly_buffer_free_chain(ctx, old);

  const PolyBackendDesc *be = poly_backend_get((PolyDevice)device);
  PolyBuffer *h = poly_arena_alloc(ctx->arena, sizeof(PolyBuffer), _Alignof(PolyBuffer));
  if (!h) return;
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
      .memory_accounted = false,
      .memory_device = POLY_DEVICE_AUTO,
  };
  poly_map_set(ctx->buffers, poly_ptr_hash(buf), buf, h, poly_ptr_eq);
}

void poly_buffer_attach(PolyCtx *ctx, PolyUOp *buf, const PolyBuffer *handle) {
  if (!ctx || !buf || !handle) return;
  PolyBuffer *old = poly_buffer_get(ctx, buf);
  if (old) poly_buffer_free_chain(ctx, old);

  PolyBuffer *h = poly_arena_alloc(ctx->arena, sizeof(PolyBuffer), _Alignof(PolyBuffer));
  *h = *handle;
  h->owned = false;
  h->src = NULL;
  h->valid = true;
  h->frontend_release = NULL;
  h->memory_accounted = false;
  h->memory_device = POLY_DEVICE_AUTO;
  if (!h->allocator) {
    const PolyBackendDesc *be = poly_backend_get(h->device);
    h->allocator = be ? be->get_allocator() : NULL;
  }
  poly_map_set(ctx->buffers, poly_ptr_hash(buf), buf, h, poly_ptr_eq);
}

void poly_buffer_adopt(PolyCtx *ctx, PolyUOp *buf, const PolyBuffer *handle) {
  if (!ctx || !buf || !handle) return;
  PolyBuffer *old = poly_buffer_get(ctx, buf);
  if (old) poly_buffer_free_chain(ctx, old);

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
  if (b) poly_buffer_free_chain(ctx, b);
  poly_map_remove(ctx->buffers, poly_ptr_hash(buf), buf, poly_ptr_eq);
}

bool poly_buffer_is_allocated(PolyCtx *ctx, PolyUOp *buf) {
  PolyBuffer *b = poly_buffer_get(ctx, buf);
  return b != NULL && (b->ptr != NULL || b->nbytes == 0 || b->device == POLY_DEVICE_HOST);
}

static bool contiguous_view_shape_numel(PolyShape shape, uint64_t *out) {
  if (!out || shape.ndim < 0) return false;
  uint64_t numel = 1;
  for (int d = 0; d < shape.ndim; d++) {
    if (shape.dims[d] < 0 ||
        (shape.dims[d] != 0 && numel > UINT64_MAX / (uint64_t)shape.dims[d]))
      return false;
    numel *= (uint64_t)shape.dims[d];
  }
  *out = numel;
  return true;
}

bool poly_uop_contiguous_view_info(
    PolyCtx *ctx,
    PolyUOp *u,
    PolyUOp **out_identity,
    PolyShape *out_shape,
    int64_t *out_numel,
    size_t *out_byte_offset
) {
  if (!ctx || !u || !out_identity || !out_shape || !out_numel || !out_byte_offset)
    return false;
  *out_identity = NULL;
  *out_shape = (PolyShape){.ndim = -1};
  *out_numel = -1;
  *out_byte_offset = 0;

  int n_steps = 0;
  bool has_shrink = false;
  PolyUOp *base = u;
  while (base && base->n_src >= 1 &&
         (base->op == POLY_OP_RESHAPE || base->op == POLY_OP_SHRINK)) {
    if (base->op == POLY_OP_SHRINK) {
      bool canonical =
          base->arg.kind == POLY_ARG_NONE && base->n_src >= 3 &&
          base->src[1]->op == POLY_OP_STACK && base->src[2]->op == POLY_OP_STACK &&
          base->src[1]->n_src == base->src[2]->n_src;
      if (!canonical && base->arg.kind != POLY_ARG_PAIR_TUPLE) return false;
      has_shrink = true;
    }
    n_steps++;
    base = base->src[0];
  }
  if (!has_shrink || !base || !poly_uop_has_buffer_identity(base)) return false;
  PolyUOp *identity = (PolyUOp *)poly_uop_get_buffer_identity(base);
  PolyBuffer *storage = poly_buffer_get(ctx, identity);
  if (!storage || (!storage->ptr && storage->nbytes != 0) || !storage->valid)
    return false;

  PolyUOp **steps = malloc((size_t)n_steps * sizeof(*steps));
  if (!steps) return false;
  PolyUOp *cur = u;
  for (int i = 0; i < n_steps; i++, cur = cur->src[0]) steps[i] = cur;

  PolyShape shape = poly_uop_max_shape_cached(ctx, base);
  uint64_t base_numel = 0;
  if (!contiguous_view_shape_numel(shape, &base_numel)) {
    free(steps);
    return false;
  }
  uint64_t element_offset = 0;
  for (int i = n_steps - 1; i >= 0; i--) {
    PolyUOp *step = steps[i];
    PolyShape next = poly_uop_max_shape_cached(ctx, step);
    uint64_t current_numel = 0, next_numel = 0;
    if (!contiguous_view_shape_numel(shape, &current_numel) ||
        !contiguous_view_shape_numel(next, &next_numel)) {
      free(steps);
      return false;
    }
    if (step->op == POLY_OP_RESHAPE) {
      if (current_numel != next_numel) {
        free(steps);
        return false;
      }
      shape = next;
      continue;
    }

    bool canonical =
        step->arg.kind == POLY_ARG_NONE && step->n_src >= 3 &&
        step->src[1]->op == POLY_OP_STACK && step->src[2]->op == POLY_OP_STACK &&
        step->src[1]->n_src == shape.ndim && step->src[2]->n_src == shape.ndim;
    if ((!canonical &&
         (step->arg.kind != POLY_ARG_PAIR_TUPLE ||
          step->arg.pair_tuple.n != shape.ndim)) ||
        next.ndim != shape.ndim) {
      free(steps);
      return false;
    }
    uint64_t stride = 1, start = 0, last = 0, selected = 1;
    bool empty = false, ok = true;
    for (int d = shape.ndim - 1; d >= 0; d--) {
      int64_t begin = 0, end = 0;
      if (canonical) {
        int64_t length = 0;
        if (poly_uop_bind_value(step->src[1]->src[d], &begin) != 0 ||
            poly_uop_bind_value(step->src[2]->src[d], &length) != 0 ||
            __builtin_add_overflow(begin, length, &end)) {
          ok = false;
          break;
        }
      } else {
        begin = step->arg.pair_tuple.pairs[d][0];
        end = step->arg.pair_tuple.pairs[d][1];
      }
      int64_t dim = shape.dims[d];
      if (dim < 0 || begin < 0 || end < begin || end > dim) {
        ok = false;
        break;
      }
      uint64_t length = (uint64_t)(end - begin);
      if ((uint64_t)begin > UINT64_MAX / stride ||
          start > UINT64_MAX - (uint64_t)begin * stride) {
        ok = false;
        break;
      }
      start += (uint64_t)begin * stride;
      if (length == 0) {
        empty = true;
      } else {
        uint64_t tail = (uint64_t)(end - 1);
        if (tail > UINT64_MAX / stride || last > UINT64_MAX - tail * stride ||
            selected > UINT64_MAX / length) {
          ok = false;
          break;
        }
        last += tail * stride;
        selected *= length;
      }
      if ((uint64_t)dim != 0 && stride > UINT64_MAX / (uint64_t)dim) {
        ok = false;
        break;
      }
      stride *= (uint64_t)dim;
    }
    if (!ok || (!empty && (last < start || last - start + 1 != selected)) ||
        element_offset > UINT64_MAX - start) {
      free(steps);
      return false;
    }
    element_offset += start;
    shape = next;
  }
  free(steps);

  uint64_t numel = 0;
  size_t itemsize = poly_dtype_itemsize(poly_dtype_scalar(identity->dtype));
  if (!contiguous_view_shape_numel(shape, &numel) || numel > INT64_MAX ||
      itemsize == 0 || element_offset > SIZE_MAX / itemsize)
    return false;
  size_t byte_offset = (size_t)element_offset * itemsize;
  if (byte_offset > storage->nbytes ||
      numel > SIZE_MAX / itemsize ||
      (size_t)numel * itemsize > storage->nbytes - byte_offset)
    return false;

  *out_identity = identity;
  *out_shape = shape;
  *out_numel = (int64_t)numel;
  *out_byte_offset = byte_offset;
  return true;
}

PolyUOp *poly_buffer_view(
    PolyCtx *ctx,
    PolyUOp *base,
    int64_t numel,
    size_t byte_offset
) {
  if (!ctx || !base || numel < 0) return NULL;
  const PolyUOp *identity = poly_uop_get_buffer_identity(base);
  if (!identity) return NULL;
  PolyBuffer *parent = poly_buffer_get(ctx, (PolyUOp *)identity);
  if (!parent) return NULL;
  size_t itemsize = poly_dtype_itemsize(poly_dtype_scalar(identity->dtype));
  if (itemsize == 0 || (uint64_t)numel > SIZE_MAX / itemsize) return NULL;
  size_t nbytes = (size_t)numel * itemsize;
  if (byte_offset > parent->nbytes || nbytes > parent->nbytes - byte_offset) return NULL;

  PolyUOp *unique = poly_uop0(
      ctx, POLY_OP_UNIQUE, POLY_VOID, poly_arg_int(poly_ctx_next_unique_id(ctx))
  );
  int64_t view_arg_values[2] = {numel, (int64_t)byte_offset};
  PolyArg view_arg = {
      .kind = POLY_ARG_INT_TUPLE,
      .int_tuple = {view_arg_values, 2},
  };
  PolyUOp *view_src[2] = {(PolyUOp *)identity, unique};
  PolyUOp *view = poly_uop(
      ctx, POLY_OP_BUFFER_VIEW, poly_dtype_scalar(identity->dtype), view_src, 2, view_arg
  );
  if (!view) return NULL;

  PolyBuffer alias = *parent;
  alias.ptr = parent->ptr ? (void *)((char *)parent->ptr + byte_offset) : NULL;
  alias.nbytes = nbytes;
  alias.owned = false;
  alias.src = NULL;
  alias.frontend_release = NULL;
  alias.memory_accounted = false;
  alias.memory_device = POLY_DEVICE_AUTO;
  poly_buffer_attach(ctx, view, &alias);
  return view;
}

PolyUOp *poly_uop_buffer(PolyCtx *ctx, PolyUOp *u) {
  if (!ctx || !u) return NULL;
  if ((u->op == POLY_OP_CONTIGUOUS || u->op == POLY_OP_RESHAPE ||
       u->op == POLY_OP_DETACH || u->op == POLY_OP_AFTER) &&
      u->n_src >= 1)
    return poly_uop_buffer(ctx, u->src[0]);
  const PolyUOp *identity = poly_uop_get_buffer_identity(u);
  if (identity) return (PolyUOp *)identity;

  PolyUOp *view_base = NULL;
  PolyShape view_shape = {.ndim = -1};
  int64_t view_numel = -1;
  size_t byte_offset = 0;
  if (!poly_uop_contiguous_view_info(
          ctx, u, &view_base, &view_shape, &view_numel, &byte_offset
      ))
    return NULL;
  PolyBuffer *parent = poly_buffer_get(ctx, view_base);
  size_t itemsize = poly_dtype_itemsize(poly_dtype_scalar(view_base->dtype));
  if (!parent || itemsize == 0 || view_numel < 0 ||
      (uint64_t)view_numel > SIZE_MAX / itemsize)
    return NULL;
  size_t nbytes = (size_t)view_numel * itemsize;

  /* Pinned UOp.buffer returns Buffer.view for a contiguous movement
   * (uop/ops.py:838-852); it does not create a UOp. Attach that runtime view
   * to the exact immutable movement node instead of manufacturing a
   * BUFFER_VIEW/UNIQUE. Repeated access refreshes one ctx->buffers row. */
  PolyBuffer alias = *parent;
  alias.nbytes = nbytes;
  alias.owned = false;
  alias.src = NULL;
  alias.frontend_release = NULL;
  alias.memory_accounted = false;
  alias.memory_device = POLY_DEVICE_AUTO;
  if (parent->device == POLY_DEVICE_WEBGPU && byte_offset != 0) {
#ifdef __EMSCRIPTEN__
    uintptr_t view = poly_webgpu_create_buffer_view(
        (uintptr_t)parent->ptr, byte_offset, nbytes
    );
    if (!view) return NULL;
    alias.ptr = (void *)view;
    alias.owned = true;
#else
    return NULL;
#endif
  } else {
    alias.ptr = parent->ptr ? (void *)((char *)parent->ptr + byte_offset) : NULL;
  }

  PolyBuffer *cached = poly_buffer_get(ctx, u);
  if (cached) {
    poly_buffer_free_chain(ctx, cached);
    *cached = alias;
  } else if (alias.owned) {
    poly_buffer_adopt(ctx, u, &alias);
  } else {
    poly_buffer_attach(ctx, u, &alias);
  }
  return poly_buffer_get(ctx, u) ? u : NULL;
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

  size_t nbytes = poly_buffer_copy_nbytes(dst, src);
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
  if (!h) {
    PolyBuffer tmp = {.ptr = ptr, .nbytes = nbytes, .device = device, .owned = true, .allocator = alloc};
    alloc->free(&tmp, alloc->dev_ctx);
    return -1;
  }
  *h = (PolyBuffer){
      .ptr = ptr,
      .nbytes = nbytes,
      .device = device,
      .owned = true,
      .allocator = alloc,
      .src = src,
      .valid = valid,
      .frontend_release = NULL,
      .memory_accounted = true,
      .memory_device = device,
  };
  poly_ctx_record_memory_alloc(ctx, device, nbytes);
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
  if (!h) {
    PolyBuffer tmp = {.ptr = ptr, .nbytes = nbytes, .device = device, .owned = true, .allocator = alloc};
    alloc->free(&tmp, alloc->dev_ctx);
    return -1;
  }
  *h = (PolyBuffer){
      .ptr = ptr,
      .nbytes = nbytes,
      .device = device,
      .owned = true,
      .allocator = alloc,
      .src = NULL,
      .valid = valid,
      .frontend_release = NULL,
      .memory_accounted = true,
      .memory_device = device,
  };
  poly_ctx_record_memory_alloc(ctx, device, nbytes);
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

static void poly_buffer_free_except_root(PolyCtx *ctx, PolyBuffer *b, PolyBuffer *root) {
  while (b && b != root) {
    PolyBuffer *next = b->src;
    b->src = NULL;
    poly_buffer_free(ctx, b);
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
      size_t copied = poly_buffer_copy_nbytes(root, cur);
      if (!cur->valid || poly_buffer_copy(root, cur) != 0) return -1;
      poly_ctx_record_buffer_copy(ctx, copied);
    }
    if (host_out) *host_out = root;
    return 0;
  }

  size_t nbytes = cur->nbytes ? cur->nbytes : poly_buffer_nbytes_for_uop(buf);
  if (poly_buffer_alloc_host_root(ctx, nbytes, false, &root) != 0) return -1;
  if (cur->valid) {
    size_t copied = poly_buffer_copy_nbytes(root, cur);
    if (poly_buffer_copy(root, cur) != 0) {
      poly_buffer_free(ctx, root);
      return -1;
    }
    poly_ctx_record_buffer_copy(ctx, copied);
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
  PolyBuffer *cur = poly_buffer_get(ctx, buf);
  if (cur && cur->ptr && cur->allocator && !poly_device_is_host_addressable(cur->device)) {
    if (poly_buffer_copyin(ctx, buf, src, nbytes) != 0) return -1;
    ctx->buffer_write_count++;
    ctx->buffer_write_bytes += nbytes;
    return 0;
  }
  PolyBuffer *root = NULL;
  if (poly_buffer_ensure_host_root_for_write(ctx, buf, &root) != 0 || !root || !root->ptr)
    return -1;
  if (nbytes == 0 || root->nbytes < nbytes) return -1;
  memcpy(root->ptr, src, nbytes);
  root->valid = true;
  ctx->buffer_write_count++;
  ctx->buffer_write_bytes += nbytes;
  cur = poly_buffer_get(ctx, buf);
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
  poly_buffer_free_except_root(ctx, existing, src);
  poly_map_set(ctx->buffers, poly_ptr_hash(buf), buf, dst, poly_ptr_eq);
  return 0;
}

static const PolyBuffer *poly_buffer_valid_source(PolyBuffer *cur) {
  if (!cur) return NULL;
  /* Browser HOST residencies can be frontend keys with ptr=NULL.  If a concrete
   * valid mirror is attached, generic migration must copy from that mirror. */
  if (cur->device == POLY_DEVICE_HOST && !cur->ptr && cur->src && cur->src->valid)
    return cur->src;
  if (cur->valid) return cur;
  if (cur->src && cur->src->valid) return cur->src;
  return NULL;
}

int poly_buffer_ensure_device_current(PolyCtx *ctx, PolyUOp *buf, PolyDevice device) {
  if (!ctx || !buf) return -1;
  if (device == POLY_DEVICE_AUTO) device = poly_device_default();
  if (device == POLY_DEVICE_HOST) device = poly_device_default();

  PolyBuffer *cur = poly_buffer_get(ctx, buf);
  if (!cur) {
    size_t nbytes = poly_buffer_nbytes_for_uop(buf);
    PolyBuffer *dst = NULL;
    if (poly_buffer_alloc_residency(ctx, buf, device, nbytes, true, NULL, &dst) != 0) return -1;
    poly_map_set(ctx->buffers, poly_ptr_hash(buf), buf, dst, poly_ptr_eq);
    return 0;
  }
  if (cur && poly_devices_share_storage(cur->device, device)) {
    cur->device = device;
    if (cur->valid) return 0;
    if (!cur->src || !cur->src->valid) return -1;
    size_t copied = poly_buffer_copy_nbytes(cur, cur->src);
    int rc = poly_buffer_copy(cur, cur->src);
    if (rc == 0) poly_ctx_record_buffer_copy(ctx, copied);
    return rc;
  }

  const PolyBuffer *source = poly_buffer_valid_source(cur);
  if (!source) return -1;

  size_t nbytes = cur && cur->nbytes ? cur->nbytes : poly_buffer_nbytes_for_uop(buf);
  PolyBuffer *src_root = poly_buffer_retained_root(cur);
  PolyBuffer *dst = NULL;
  if (poly_buffer_alloc_residency(ctx, buf, device, nbytes, false, src_root, &dst) != 0) return -1;
  size_t copied = poly_buffer_copy_nbytes(dst, source);
  if (poly_buffer_copy(dst, source) != 0) {
    poly_buffer_free(ctx, dst);
    return -1;
  }
  poly_ctx_record_buffer_copy(ctx, copied);
  poly_buffer_free_except_root(ctx, cur, src_root);
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
  if (!b || !b->ptr || !b->allocator || nbytes == 0 || b->nbytes < nbytes) return -1;
  PolyBuffer src_view = poly_buffer_make_host_view((void *)src, nbytes);
  int rc = b->allocator->copy_in(b, &src_view, nbytes, b->allocator->dev_ctx);
  if (rc == 0) {
    b->valid = true;
    if (b->src) b->src->valid = false;
  }
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
  ctx->buffer_read_count++;
  ctx->buffer_read_bytes += nbytes;
  return 0;
}
