/* device.h -- Buffer type, allocator interface, buffer operations
 *
 * All buffer operations take PolyCtx* as first argument.
 */

#ifndef POLY_DEVICE_H
#define POLY_DEVICE_H

#include "polygrad.h"

/* Allocator interface: per-device memory operations */

typedef struct PolyBuffer PolyBuffer;

typedef struct PolyAllocator {
  void *(*alloc)(size_t nbytes, void *dev_ctx);
  void (*free)(const PolyBuffer *buffer, void *dev_ctx);
  int (*copy_in)(const PolyBuffer *dst, const PolyBuffer *src, size_t nbytes, void *dev_ctx);
  int (*copy_out)(const PolyBuffer *dst, const PolyBuffer *src, size_t nbytes, void *dev_ctx);
  int (*copy_between)(const PolyBuffer *dst, const PolyBuffer *src, size_t nbytes, void *dev_ctx);
  void *dev_ctx;
  bool host_addressable;
} PolyAllocator;

/* Buffer: device-specific memory reference.
 *   CPU/INTERP: host malloc'd pointer
 *   CUDA:       CUdeviceptr (cast to void*)
 *   HIP:        hipDeviceptr_t (void*)
 *   WASM:       offset into Emscripten heap
 *   WEBGPU:     GPUBuffer (host-managed, wrapped)
 *
 * Each buffer stores its allocator (resolved at allocation time).
 * `src` points to the root source buffer (usually host) for migration tracking.
 * `valid` indicates whether `ptr` contains the current logical contents.
 *
 * Lifecycle:
 *   1. Frontend attaches host data: {ptr=host, valid=true, src=NULL}
 *   2. Realize allocates device residency: {ptr=cuda, valid=false, src=&host}
 *   3. After copy_in: {ptr=cuda, valid=true, src->valid=true (unchanged)}
 *   4. After kernel writes: {ptr=cuda, valid=true, src->valid=false}
 */

struct PolyBuffer {
  void *ptr;
  size_t nbytes;
  PolyDevice device;
  bool owned; /* should allocator->free be called when this residency is retired? */
  const PolyAllocator *allocator; /* allocator for this buffer (set at allocation time) */
  PolyBuffer *src;  /* root source buffer (usually host), or NULL */
  bool valid; /* does ptr contain current logical contents? */
  PolyFrontendBufferReleaseFn frontend_release; /* imported HOST owner release hook */
  bool memory_accounted; /* contributes to ctx GlobalCounters.mem_used */
  PolyDevice memory_device; /* allocation device used for per-device accounting */
};

/* Attach buffer data to a BUFFER UOp.
 * FFI-friendly: takes flat scalars, constructs PolyBuffer internally. */
void poly_buffer_set(PolyCtx *ctx, PolyUOp *buf, void *ptr, size_t nbytes, int domain);

/* Attach an existing runtime buffer view to a BUFFER UOp without taking
 * ownership of the underlying allocation. This is used when instance/call
 * storage should be visible through ctx->buffers, which keeps execution on
 * the same schedule/realize path as normal tensors. */
void poly_buffer_attach(PolyCtx *ctx, PolyUOp *buf, const PolyBuffer *handle);

/* Adopt an existing runtime buffer as the authoritative ctx residency.
 * Unlike poly_buffer_attach(), this preserves handle->owned and is intended
 * for runtime-owned storage that ctx must release when the binding is retired. */
void poly_buffer_adopt(PolyCtx *ctx, PolyUOp *buf, const PolyBuffer *handle);

PolyUOp *poly_buffer_from_host(
    PolyCtx *ctx, void *ptr, size_t nbytes, int dtype_id,
    int64_t *dims, int ndim
);

/* Build a device-annotated frontend-host BUFFER from an existing UNIQUE and
 * attach the imported bytes only to that physical BUFFER. Returns the actual
 * source residency domain (HOST, or WASM for staged Emscripten bytes). */
PolyUOp *poly_buffer_from_host_unique(
    PolyCtx *ctx,
    PolyUOp *unique,
    PolyDType scalar_dtype,
    int64_t numel,
    void *ptr,
    size_t nbytes,
    PolyDevice *out_source_device
);

/* Create a read/write mmap-backed one-dimensional DISK buffer. The mapped
 * file owns its lifetime through ctx->buffers and is excluded from mem_used. */
PolyUOp *poly_buffer_from_file(PolyCtx *ctx, const char *path, int dtype_id);

/* Create an attached BUFFER_VIEW alias into an existing realized buffer. */
PolyUOp *poly_buffer_view(
    PolyCtx *ctx, PolyUOp *base, int64_t numel, size_t byte_offset
);

const PolyAllocator *poly_disk_get_allocator(void);

/* Notify the frontend that the given PolyBuffer* key is no longer needed. */
void poly_frontend_buffer_release_key(uintptr_t buffer_key);

/* Wrap a raw host-addressable pointer as an ephemeral runtime buffer view.
 * On native builds this is a CPU view; on Emscripten it is a WASM view. */
PolyBuffer poly_buffer_make_host_view(void *ptr, size_t nbytes);

/* Look up data pointer for a BUFFER UOp. Returns NULL if not attached. */
void *poly_buffer_get_ptr(PolyCtx *ctx, PolyUOp *buf);

/* Look up full PolyBuffer for a BUFFER UOp. Returns NULL if not attached. */
PolyBuffer *poly_buffer_get(PolyCtx *ctx, PolyUOp *buf);

/* Free this residency's ptr if owned. Resets ptr=NULL, owned=false, valid=false.
 * Does NOT touch b->src. Does not remove from the side table.
 * Use this when you want to free only the current residency (e.g. during
 * migration where the old current is discarded but src is retained). */
void poly_buffer_free(PolyCtx *ctx, PolyBuffer *b);

/* Free this residency AND its src chain. Use when fully discarding a logical
 * buffer (poly_buffer_remove, ctx destroy, full rebinding via poly_buffer_set). */
void poly_buffer_free_chain(PolyCtx *ctx, PolyBuffer *b);

/* Remove a buffer from the side table (frees current + src chain). */
void poly_buffer_remove(PolyCtx *ctx, PolyUOp *buf);

/* Copy logical contents from src residency into dst residency. */
int poly_buffer_copy(PolyBuffer *dst, const PolyBuffer *src);

/* Ensure device memory exists for a BUFFER UOp. If the buffer already has valid
 * logical contents, preserve them on the requested device. */
int poly_buffer_allocate(PolyCtx *ctx, PolyUOp *buf, PolyDevice device);

/* Ensure a buffer is allocated. Preserves valid contents when present. */
int poly_buffer_ensure_allocated(PolyCtx *ctx, PolyUOp *buf, PolyDevice device);

/* Ensure target-device residency exists for an output-only write. Does not copy
 * logical contents and may discard stale current residency. */
int poly_buffer_ensure_device_allocated(PolyCtx *ctx, PolyUOp *buf, PolyDevice device);

/* Allocate or reuse ctx-owned host storage for a BUFFER UOp. */
int poly_buffer_alloc_owned_host(
    PolyCtx *ctx,
    PolyUOp *buf,
    size_t nbytes,
    bool zero,
    PolyBuffer **host_out
);

/* Ensure target-device residency exists and contains current logical contents. */
int poly_buffer_ensure_device_current(PolyCtx *ctx, PolyUOp *buf, PolyDevice device);

/* Ensure a host-addressable root exists and contains current logical contents. */
int poly_buffer_ensure_host_current(PolyCtx *ctx, PolyUOp *buf, PolyBuffer **host_out);

/* Mark host root as the newest logical contents after caller mutation. */
int poly_buffer_mark_host_written(PolyCtx *ctx, PolyUOp *buf);

/* Mark the current residency as written by a successful backend execution. */
int poly_buffer_mark_residency_written(PolyCtx *ctx, PolyUOp *buf, PolyDevice device);

/* Host -> device data transfer. Buffer must be allocated. */
int poly_buffer_copyin(PolyCtx *ctx, PolyUOp *buf, const void *src, size_t nbytes);

/* Device -> host data transfer. Buffer must be allocated. */
int poly_buffer_copyout(PolyCtx *ctx, PolyUOp *buf, void *dst, size_t nbytes);

/* Read bytes from a realized buffer into dst. Alias for copyout, but used by
 * frontends that need a backend-aware readback path rather than raw ptr access. */
int poly_buffer_read(PolyCtx *ctx, PolyUOp *buf, void *dst, size_t nbytes);

/* Replace logical contents from host bytes, preserving cached device residency. */
int poly_buffer_write(PolyCtx *ctx, PolyUOp *buf, const void *src, size_t nbytes);

/* Check if a buffer has data in the side table. */
bool poly_buffer_is_allocated(PolyCtx *ctx, PolyUOp *buf);

#endif /* POLY_DEVICE_H */
