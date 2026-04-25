#ifndef POLY_RUNTIME_WEBGPU_H
#define POLY_RUNTIME_WEBGPU_H

#include "polygrad.h"
#include "device.h"
#include "engine/schedule.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __EMSCRIPTEN__
int poly_browser_host_copy_out(uintptr_t src_buffer_key, void *dst_ptr, size_t nbytes);
int poly_browser_host_copy_in(uintptr_t dst_buffer_key, const void *src_ptr, size_t nbytes);

int poly_webgpu_memset_zero(uintptr_t handle, size_t nbytes);

int poly_webgpu_lower_item(PolyCtx *ctx, PolyUOp *scheduled_root, const char *fn_name, PolyRunner *out);
int poly_webgpu_execute(PolyRunner *runner, void **args, int n_args);
void poly_webgpu_free_runner(PolyRunner *runner);
const PolyAllocator *poly_webgpu_get_allocator(void);
#endif

#ifdef __cplusplus
}
#endif

#endif
