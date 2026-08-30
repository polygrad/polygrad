#ifndef POLY_RUNTIME_WEBGPU_H
#define POLY_RUNTIME_WEBGPU_H

#include "polygrad.h"
#include "device.h"
#include "engine/schedule.h"

#ifdef __cplusplus
extern "C" {
#endif

/* C equivalent of Tinygrad Target.arch membership for shader-f16. */
bool poly_webgpu_supports_float16(void);

#ifdef __EMSCRIPTEN__
int poly_webgpu_memset_zero(uintptr_t handle, size_t nbytes);
uintptr_t poly_webgpu_create_buffer_view(uintptr_t base_handle, size_t byte_offset, size_t nbytes);

int poly_webgpu_lower_item(PolyCtx *ctx, PolyUOp *scheduled_root, const char *fn_name, PolyRunner *out);
char *poly_webgpu_render_source(PolyCtx *ctx, PolyUOp *program, const char *fn_name);
int poly_webgpu_execute(PolyRunner *runner, void **args, int n_args);
void poly_webgpu_free_runner(PolyRunner *runner);
const PolyAllocator *poly_webgpu_get_allocator(void);
#endif

#ifdef __cplusplus
}
#endif

#endif
