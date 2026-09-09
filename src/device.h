/* device.h -- Buffer type, allocator interface, buffer operations
 *
 * All buffer operations take PolyCtx* as first argument.
 */

#ifndef POLY_DEVICE_H
#define POLY_DEVICE_H

#include "polygrad.h"

/* Allocator interface: per-device memory operations */

typedef struct PolyBuffer PolyBuffer;

/* Pinned tinygrad stores canonical device strings directly in DEVICE.arg
 * (device.py:15-24, uop/ops.py:733-746).  These are the only constructors
 * production graph builders should use; the enum overload is the canonical
 * ordinal-zero wrapper for the current backend-only public API. AUTO remains
 * Polygrad's existing unresolved internal DEVICE(None) placeholder. */
PolyUOp *poly_device_uop_from_name(PolyCtx *ctx, const char *name);
PolyUOp *poly_device_uop_from_names(PolyCtx *ctx, const char **names, int n);
PolyUOp *poly_device_uop(PolyCtx *ctx, PolyDevice device);

/* Current Tinygrad UOp.device query with optional pass-local memoization. */
PolyUOp *poly_uop_device_uop_cached(PolyCtx *ctx, PolyUOp *u, PolyMap *cache);

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
  /* Current tinygrad Buffer._base/offset runtime metadata (device.py:102-155).
   * A view exists before allocation; ensure_allocated materializes its base
   * first, then derives this handle without changing UOp topology. */
  PolyBuffer *base;
  size_t offset;
  PolyDevice device;
  bool owned; /* should allocator->free be called when this residency is retired? */
  const PolyAllocator *allocator; /* allocator for this buffer (set at allocation time) */
  PolyBuffer *src; /* root source buffer (usually host), or NULL */
  bool valid; /* does ptr contain current logical contents? */
  PolyFrontendBufferReleaseFn frontend_release; /* imported HOST owner release hook */
  bool memory_accounted; /* contributes to ctx GlobalCounters.mem_used */
  PolyDevice memory_device; /* allocation device used for per-device accounting */
  PolyUOp *device_uop; /* exact canonical runtime residency identity */
  PolyUOp *memory_device_uop; /* exact identity used for allocation accounting */

  /* Pinned tinygrad represents a tuple-device BUFFER as MultiBuffer.bufs and
   * constructs MSTACK.buffer as a borrowing MultiBuffer over its ordered
   * source buffers (device.py:86-97, uop/ops.py:853-879).  This is runtime
   * storage only: it never uses the scalar migration `src` chain and never
   * changes UOp topology. */
  PolyBuffer **bufs;
  int n_bufs;
  bool owns_bufs;
};

/* Attach buffer data to a BUFFER UOp.
 * FFI-friendly: takes flat scalars, constructs PolyBuffer internally.
 * set/attach/adopt return 0 on publication, -1 on invalid arguments, metadata
 * allocation failure or a replacement that would invalidate existing aliases.
 * Failure preserves the previous binding and caller ownership. Raw metadata
 * pointers are invalid after successful replacement. Map OOM remains fatal. */
int poly_buffer_set(PolyCtx *ctx, PolyUOp *buf, void *ptr, size_t nbytes, int domain);

/* Attach an existing runtime buffer view to a BUFFER UOp without taking
 * ownership of the underlying allocation. Runtime call storage is visible
 * through ctx->buffers on the ordinary LINEAR/realize path. */
int poly_buffer_attach(PolyCtx *ctx, PolyUOp *buf, const PolyBuffer *handle);

/* Adopt an existing runtime buffer as the authoritative ctx residency.
 * Unlike poly_buffer_attach(), this preserves handle->owned and is intended
 * for runtime-owned storage that ctx must release when the binding is retired. */
int poly_buffer_adopt(PolyCtx *ctx, PolyUOp *buf, const PolyBuffer *handle);

PolyUOp *poly_buffer_from_host(
    PolyCtx *ctx,
    void *ptr,
    size_t nbytes,
    int dtype_id,
    int64_t *dims,
    int ndim
);

/* Build a frontend-host BUFFER from an existing UNIQUE and attach imported
 * bytes only to that physical BUFFER. The source storage domain is HOST. */
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

/* Prove the currently supported static contiguous movement-view subset
 * (RESHAPE/SHRINK over valid realized storage) without creating a UOp.
 * This is shared by Tinygrad-style UOp.buffer access and explicit placement. */
bool poly_uop_contiguous_view_info(
    PolyCtx *ctx,
    PolyUOp *u,
    PolyUOp **out_identity,
    PolyShape *out_shape,
    int64_t *out_numel,
    size_t *out_byte_offset
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

/* C analogue of tinygrad UOp.buffer. It resolves tuple BUFFER/MSTACK/MSELECT
 * runtime values as well as the existing scalar identities/views. */
PolyBuffer *poly_uop_buffer_handle(PolyCtx *ctx, PolyUOp *u);

bool poly_buffer_is_multi(const PolyBuffer *buffer);
PolyBuffer *poly_buffer_multi_child(PolyBuffer *buffer, int index);

/* Exact Buffer.ensure_allocated analogue for an already-resolved scalar
 * runtime handle. This is used after MultiBuffer lane resolution, where a
 * child has no independent BUFFER UOp key. */
int poly_buffer_handle_ensure_allocated(PolyCtx *ctx, PolyBuffer *buffer);

/* Current Tinygrad Buffer.get_buf(device) analogue for a resolved scalar
 * runtime handle. Browser HOST imports are JS-owned keys; HOST execution
 * materializes such a key into addressable Wasm memory on first use. */
int poly_buffer_handle_get_buf(PolyCtx *ctx, PolyBuffer *buffer, PolyDevice device, void **out);

/* Free this residency's ptr if owned. Resets ptr=NULL, owned=false, valid=false.
 * Does NOT touch b->src. Does not remove from the side table.
 * Use this when you want to free only the current residency (e.g. during
 * migration where the old current is discarded but src is retained). */
void poly_buffer_free(PolyCtx *ctx, PolyBuffer *b);

/* Free this residency, its src chain, and their C handle metadata. Use when
 * fully discarding a buffer (remove, ctx destroy, or full rebinding). */
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

/* Copy current realized bytes into caller-owned dst without retaining a host
 * mirror. Resolves views and authoritative storage through the backend. */
int poly_buffer_read(PolyCtx *ctx, PolyUOp *buf, void *dst, size_t nbytes);

/* Replace logical contents from host bytes, preserving cached device residency. */
int poly_buffer_write(PolyCtx *ctx, PolyUOp *buf, const void *src, size_t nbytes);

/* Check if a buffer has data in the side table. */
bool poly_buffer_is_allocated(PolyCtx *ctx, PolyUOp *buf);

#endif /* POLY_DEVICE_H */
