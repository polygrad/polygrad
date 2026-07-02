/*
 * schedule.c -- Tinygrad-aligned engine scheduling and compiled execution.
 *
 * This file plays the role of tinygrad/engine/schedule.py in C:
 *   - schedule construction from a tensor sink
 *   - backend lowering of schedule items
 *   - schedule execution with reusable workspace
 */

#define _POSIX_C_SOURCE 200809L
#include "engine/schedule.h"
#include "ctx.h"
#include "utils.h"
#include "frontend_internal.h"
#include "codegen.h"
#include "interp.h"
#include "schedule/rangeify.h"
#include "runtime_wasm.h"
#include "runtime_webgpu.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>

static PolyUOp *poly_program_source_identity(PolyUOp *program) {
  if (!program || program->op != POLY_OP_PROGRAM) return program;
  if (program->n_src >= 3 && program->src[2] && program->src[2]->op == POLY_OP_LINEAR)
    return program->src[2];
  if (program->n_src >= 1) return program->src[0];
  return program;
}

static void stable_kernel_fn_name(
    PolyCtx *ctx,
    char *out,
    size_t cap,
    PolyDevice device,
    PolyUOp *root
) {
  (void)ctx;
  uint32_t h = poly_structural_hash(poly_program_source_identity(root));
  /* The native CPU compiler hashes the full rendered source for its disk cache.
   * Counter-based function names make identical kernels render different source,
   * defeating tinygrad-style compile caching across eager realize calls. */
  snprintf(out, cap, "poly_k_%d_%08x", (int)device, h);
}

struct PolyRuntimeCacheEntry {
  int refcount;
  bool in_cache;
  PolyCtx *ctx;
  size_t accounted_bytes;
  PolyUOp *program;
  PolyDevice device;
  uint32_t env_stamp;
  PolyRunner runner;
};

typedef struct {
  PolyUOp *program;
  PolyDevice device;
  uint32_t env_stamp;
  PolyRuntimeCacheEntry *runtime_program;
} PolyRuntimeCacheMapEntry;

typedef struct {
  PolyUOp *program;
  PolyDevice device;
  uint32_t env_stamp;
  PolyUOp *prepared_program;
} PolyToProgramCacheEntry;

typedef struct {
  PolyUOp *template_buf;
  bool needs_zero;
} PolyScheduleIntermediateDesc;

struct PolyScheduleCacheEntry {
  int refcount;
  bool in_cache;
  PolyUOp *linear;
  PolyCallAccess *call_access;
  int n_calls;
  PolyScheduleIntermediateDesc *intermediates;
  int n_intermediates;
};

static PolyScheduleCacheEntry *poly_schedule_cache_entry_retain(PolyScheduleCacheEntry *entry);
static void poly_schedule_cache_entry_release(PolyScheduleCacheEntry *entry);

static uint32_t poly_schedule_lower_env_stamp(void);
static int poly_launch_dim_upper_bound(PolyCtx *ctx, PolyUOp *expr);
static PolyUOp *poly_program_ensure_source(
    PolyCtx *ctx,
    const PolyBackendDesc *backend,
    PolyUOp *program,
    PolyDevice device
);
static int g_program_source_render_count = 0;

int poly_program_source_render_count(void) {
  return g_program_source_render_count;
}

void poly_program_source_render_count_reset(void) {
  g_program_source_render_count = 0;
}

static char *poly_schedule_arena_strdup(PolyCtx *ctx, const char *s) {
  if (!ctx) return NULL;
  if (!s) s = "";
  size_t len = strlen(s);
  char *out = poly_arena_alloc(ctx->arena, len + 1, 1);
  if (!out) return NULL;
  memcpy(out, s, len + 1);
  return out;
}

PolyUOp *poly_schedule_call(const PolySchedule *schedule, int call_index) {
  if (!schedule || !schedule->template || !schedule->template->linear ||
      schedule->template->linear->op != POLY_OP_LINEAR || call_index < 0 ||
      call_index >= schedule->template->n_calls)
    return NULL;
  return schedule->template->linear->src[call_index];
}

static PolyUOp *poly_call_raw_body(PolyUOp *call) {
  if (!call || call->op != POLY_OP_CALL || call->n_src < 1) return NULL;
  return call->src[0];
}

static PolyUOp *poly_program_body(PolyUOp *body) {
  if (body && body->op == POLY_OP_PROGRAM && body->n_src >= 1) return body->src[0];
  return body;
}

static PolyUOp *poly_program_kernel_body(PolyUOp *program) {
  if (!program || program->op != POLY_OP_PROGRAM || program->n_src < 1) return NULL;
  return program->src[0];
}

PolyUOp *poly_program_linear(PolyUOp *program) {
  if (!program || program->op != POLY_OP_PROGRAM || program->n_src < 3) return NULL;
  PolyUOp *linear = program->src[2];
  return (linear && linear->op == POLY_OP_LINEAR) ? linear : NULL;
}

static PolyUOp *poly_program_source(PolyUOp *program) {
  if (!program || program->op != POLY_OP_PROGRAM || program->n_src < 4) return NULL;
  PolyUOp *source = program->src[3];
  return (source && source->op == POLY_OP_SOURCE) ? source : NULL;
}

static PolyUOp *poly_program_binary(PolyUOp *program) {
  if (!program || program->op != POLY_OP_PROGRAM || program->n_src < 5) return NULL;
  PolyUOp *binary = program->src[4];
  return (binary && binary->op == POLY_OP_BINARY) ? binary : NULL;
}

#ifndef __EMSCRIPTEN__
static const char *poly_program_source_text(PolyUOp *program) {
  PolyUOp *source = poly_program_source(program);
  if (!source || source->arg.kind != POLY_ARG_STRING) return NULL;
  return source->arg.str;
}
#endif

static PolyUOp **poly_program_linear_uops(PolyUOp *program, int *n_out) {
  PolyUOp *linear = poly_program_linear(program);
  if (!linear) return NULL;
  if (n_out) *n_out = linear->n_src;
  return linear->src;
}

PolyUOp *poly_schedule_call_body(const PolySchedule *schedule, int call_index) {
  return poly_program_body(poly_call_raw_body(poly_schedule_call(schedule, call_index)));
}

static bool poly_call_arg_is_var(PolyUOp *u) {
  return u && u->op == POLY_OP_DEFINE_VAR;
}

static bool poly_call_is_copy(PolyUOp *call) {
  return call && call->op == POLY_OP_CALL && call->n_src >= 1 && call->src[0] &&
         call->src[0]->op == POLY_OP_COPY;
}

static bool poly_call_is_view(PolyUOp *call) {
  return call && call->op == POLY_OP_CALL && call->n_src >= 1 && call->src[0] &&
         call->src[0]->op == POLY_OP_BUFFER_VIEW;
}

bool poly_schedule_call_is_copy(const PolySchedule *schedule, int call_index) {
  return poly_call_is_copy(poly_schedule_call(schedule, call_index));
}

static const char *poly_call_kind_name(PolyUOp *call) {
  if (poly_call_is_copy(call)) return "copy";
  if (poly_call_is_view(call)) return "view";
  PolyUOp *body = (call && call->n_src > 0) ? call->src[0] : NULL;
  if (body && body->op == POLY_OP_PROGRAM) return "program";
  if (body && body->op == POLY_OP_BUFFER_VIEW) return "view";
  if (body && body->op == POLY_OP_ENCDEC) return "encdec";
  return "compute";
}

static int poly_call_n_buffer_args(PolyUOp *call) {
  if (!call || call->op != POLY_OP_CALL || call->n_src < 1) return 0;
  int n = 0;
  for (int i = 1; i < call->n_src; i++)
    if (!poly_call_arg_is_var(call->src[i])) n++;
  return n;
}

int poly_schedule_call_n_buffer_args(const PolySchedule *schedule, int call_index) {
  return poly_call_n_buffer_args(poly_schedule_call(schedule, call_index));
}

static PolyUOp *poly_call_buffer_arg(PolyUOp *call, int param_idx) {
  if (!call || call->op != POLY_OP_CALL || param_idx < 0) return NULL;
  int seen = 0;
  for (int i = 1; i < call->n_src; i++) {
    if (poly_call_arg_is_var(call->src[i])) continue;
    if (seen++ == param_idx) return call->src[i];
  }
  return NULL;
}

static int poly_call_n_var_args(PolyUOp *call) {
  if (!call || call->op != POLY_OP_CALL || call->n_src < 1) return 0;
  int n = 0;
  for (int i = 1; i < call->n_src; i++)
    if (poly_call_arg_is_var(call->src[i])) n++;
  return n;
}

static PolyUOp *poly_call_var_arg(PolyUOp *call, int var_idx) {
  if (!call || call->op != POLY_OP_CALL || var_idx < 0) return NULL;
  int seen = 0;
  for (int i = 1; i < call->n_src; i++) {
    if (!poly_call_arg_is_var(call->src[i])) continue;
    if (seen++ == var_idx) return call->src[i];
  }
  return NULL;
}

static int poly_call_param_index_for_identity(PolyUOp *call, const PolyUOp *identity) {
  if (!call || !identity) return -1;
  int n_args = poly_call_n_buffer_args(call);
  if (identity->op == POLY_OP_PARAM && identity->arg.kind == POLY_ARG_INT) {
    int idx = (int)identity->arg.i;
    return (idx >= 0 && idx < n_args) ? idx : -1;
  }
  for (int i = 0; i < n_args; i++) {
    PolyUOp *arg = poly_call_buffer_arg(call, i);
    if (poly_uop_get_buffer_identity(arg) == identity) return i;
  }
  return -1;
}

static void poly_call_mark_access_param(PolyUOp *call, PolyUOp *ptr, bool *mask, int n_args) {
  if (!ptr || !mask) return;
  if (ptr->op == POLY_OP_PARAM && ptr->arg.kind == POLY_ARG_INT) {
    int idx = (int)ptr->arg.i;
    if (idx >= 0 && idx < n_args) mask[idx] = true;
    return;
  }
  if (ptr->op == POLY_OP_INDEX || ptr->op == POLY_OP_GEP) {
    if (ptr->n_src > 0) poly_call_mark_access_param(call, ptr->src[0], mask, n_args);
    return;
  }
  const PolyUOp *identity = poly_uop_get_buffer_identity(ptr);
  int idx = poly_call_param_index_for_identity(call, identity);
  if (idx >= 0 && idx < n_args) {
    mask[idx] = true;
    return;
  }
  if (ptr->op == POLY_OP_STACK || ptr->op == POLY_OP_TUPLE || ptr->op == POLY_OP_GROUP) {
    for (int i = 0; i < ptr->n_src; i++)
      poly_call_mark_access_param(call, ptr->src[i], mask, n_args);
  }
}

static int poly_call_mask_to_indices(
    PolyCtx *ctx,
    const bool *mask,
    int n,
    int **out_items,
    int *out_n
) {
  if (!ctx || !mask || !out_items || !out_n || n < 0) return -1;
  int count = 0;
  for (int i = 0; i < n; i++)
    if (mask[i]) count++;

  int *items = NULL;
  if (count > 0) {
    items = poly_arena_alloc(ctx->arena, (size_t)count * sizeof(int), _Alignof(int));
    if (!items) return -1;
    int w = 0;
    for (int i = 0; i < n; i++)
      if (mask[i]) items[w++] = i;
  }
  *out_items = items;
  *out_n = count;
  return 0;
}

const PolyProgramInfo *poly_program_info(PolyCtx *ctx, PolyUOp *program) {
  (void)ctx;
  if (!program || program->op != POLY_OP_PROGRAM) return NULL;
  if (program->arg.kind == POLY_ARG_PROGRAM_INFO) return program->arg.program_info;
  return NULL;
}

static bool str_eq(const char *a, const char *b) {
  return a == b || (a && b && strcmp(a, b) == 0);
}

static bool int_arr_eq(const int *a, int na, const int *b, int nb) {
  if (na != nb) return false;
  if (na == 0) return true;
  return a && b && memcmp(a, b, (size_t)na * sizeof(int)) == 0;
}

static uint32_t program_info_hash_mix(uint32_t h, uint32_t v) {
  h ^= v;
  h *= 0x9e3779b9;
  h ^= h >> 16;
  return h;
}

static uint32_t program_info_hash_ptr(uint32_t h, const void *p) {
  uintptr_t v = (uintptr_t)p;
  uint32_t folded = (uint32_t)v;
  if (sizeof(v) > 4) folded ^= (uint32_t)(v >> 32);
  h = program_info_hash_mix(h, folded);
  return h;
}

bool poly_program_info_eq(const PolyProgramInfo *a, const PolyProgramInfo *b) {
  if (a == b) return true;
  if (!a || !b) return false;
  if (!str_eq(a->name, b->name)) return false;
  if (memcmp(a->global_size, b->global_size, sizeof(a->global_size)) != 0) return false;
  if (memcmp(a->local_size, b->local_size, sizeof(a->local_size)) != 0) return false;
  if (a->has_local_size != b->has_local_size) return false;
  if (a->n_vars != b->n_vars || a->n_globals != b->n_globals || a->n_outs != b->n_outs ||
      a->n_ins != b->n_ins)
    return false;
  for (int i = 0; i < 3; i++) {
    if (a->global_exprs[i] != b->global_exprs[i]) return false;
    if (a->local_exprs[i] != b->local_exprs[i]) return false;
  }
  for (int i = 0; i < a->n_vars; i++)
    if (a->vars[i] != b->vars[i]) return false;
  return int_arr_eq(a->globals, a->n_globals, b->globals, b->n_globals) &&
         int_arr_eq(a->outs, a->n_outs, b->outs, b->n_outs) &&
         int_arr_eq(a->ins, a->n_ins, b->ins, b->n_ins);
}

uint32_t poly_program_info_hash(const PolyProgramInfo *info) {
  uint32_t h = 0x811c9dc5u;
  if (!info) return h;
  if (info->name) {
    for (const unsigned char *p = (const unsigned char *)info->name; *p; p++)
      h = program_info_hash_mix(h, (uint32_t)*p);
  }
  for (int i = 0; i < 3; i++) {
    h = program_info_hash_mix(h, (uint32_t)info->global_size[i]);
    h = program_info_hash_mix(h, (uint32_t)info->local_size[i]);
    h = program_info_hash_ptr(h, info->global_exprs[i]);
    h = program_info_hash_ptr(h, info->local_exprs[i]);
  }
  h = program_info_hash_mix(h, info->has_local_size ? 1u : 0u);
  h = program_info_hash_mix(h, (uint32_t)info->n_vars);
  for (int i = 0; i < info->n_vars; i++)
    h = program_info_hash_ptr(h, info->vars[i]);
  h = program_info_hash_mix(h, (uint32_t)info->n_globals);
  for (int i = 0; i < info->n_globals; i++)
    h = program_info_hash_mix(h, (uint32_t)info->globals[i]);
  h = program_info_hash_mix(h, (uint32_t)info->n_outs);
  for (int i = 0; i < info->n_outs; i++)
    h = program_info_hash_mix(h, (uint32_t)info->outs[i]);
  h = program_info_hash_mix(h, (uint32_t)info->n_ins);
  for (int i = 0; i < info->n_ins; i++)
    h = program_info_hash_mix(h, (uint32_t)info->ins[i]);
  return h;
}

static int poly_call_get_outs_ins_from_body(
    PolyCtx *ctx,
    PolyUOp *call,
    PolyUOp *body,
    bool *globals,
    bool *outs,
    bool *ins,
    int n_args
) {
  if (!ctx || !call || !outs || !ins || n_args < 0) return -1;
  if (globals && n_args > 0) memset(globals, 0, (size_t)n_args * sizeof(bool));
  if (n_args > 0) {
    memset(outs, 0, (size_t)n_args * sizeof(bool));
    memset(ins, 0, (size_t)n_args * sizeof(bool));
  }

  if (!body) return -1;
  if (body->op == POLY_OP_COPY) {
    if (n_args < 2) return -1;
    if (globals) {
      globals[0] = true;
      globals[1] = true;
    }
    outs[0] = true;
    ins[1] = true;
    return 0;
  }
  if (body->op == POLY_OP_BUFFER_VIEW) {
    if (n_args < 2) return -1;
    if (globals) {
      globals[0] = true;
      globals[1] = true;
    }
    outs[0] = true;
    ins[1] = true;
    return 0;
  }

  int cap = 64, n_stack = 0;
  PolyUOp **stack = malloc((size_t)cap * sizeof(PolyUOp *));
  PolyMap *seen = poly_map_new(128);
  if (!stack || !seen) {
    free(stack);
    if (seen) poly_map_destroy(seen);
    return -1;
  }
  stack[n_stack++] = body;
  while (n_stack > 0) {
    PolyUOp *u = stack[--n_stack];
    if (!u) continue;
    if (poly_map_get(seen, poly_ptr_hash(u), u, poly_ptr_eq)) continue;
    poly_map_set(seen, poly_ptr_hash(u), u, u, poly_ptr_eq);
    if (globals) poly_call_mark_access_param(call, u, globals, n_args);

    if (u->op == POLY_OP_STORE && u->n_src >= 1) {
      if (globals) poly_call_mark_access_param(call, u->src[0], globals, n_args);
      poly_call_mark_access_param(call, u->src[0], outs, n_args);
      for (int s = 1; s < u->n_src; s++) {
        if (n_stack >= cap) {
          cap *= 2;
          PolyUOp **new_stack = realloc(stack, (size_t)cap * sizeof(PolyUOp *));
          if (!new_stack) {
            free(stack);
            poly_map_destroy(seen);
            return -1;
          }
          stack = new_stack;
        }
        stack[n_stack++] = u->src[s];
      }
      continue;
    }

    if (u->op == POLY_OP_LOAD && u->n_src >= 1) {
      if (globals) poly_call_mark_access_param(call, u->src[0], globals, n_args);
      poly_call_mark_access_param(call, u->src[0], ins, n_args);
    } else {
      const PolyUOp *identity = poly_uop_get_buffer_identity(u);
      int idx = poly_call_param_index_for_identity(call, identity);
      if (idx >= 0 && idx < n_args) ins[idx] = true;
    }

    for (int s = 0; s < u->n_src; s++) {
      if (n_stack >= cap) {
        cap *= 2;
        PolyUOp **new_stack = realloc(stack, (size_t)cap * sizeof(PolyUOp *));
        if (!new_stack) {
          free(stack);
          poly_map_destroy(seen);
          return -1;
        }
        stack = new_stack;
      }
      stack[n_stack++] = u->src[s];
    }
  }
  free(stack);
  poly_map_destroy(seen);
  return 0;
}

static int poly_call_apply_program_info(
    const PolyProgramInfo *info,
    bool *outs,
    bool *ins,
    int n_args
) {
  if (!info || !outs || !ins || n_args < 0) return -1;
  memset(outs, 0, (size_t)n_args * sizeof(bool));
  memset(ins, 0, (size_t)n_args * sizeof(bool));
  for (int i = 0; i < info->n_outs; i++) {
    int idx = info->outs[i];
    if (idx < 0 || idx >= n_args) return -1;
    outs[idx] = true;
  }
  for (int i = 0; i < info->n_ins; i++) {
    int idx = info->ins[i];
    if (idx < 0 || idx >= n_args) return -1;
    ins[idx] = true;
  }
  return 0;
}

static void poly_program_info_collect_launch(PolyCtx *ctx, PolyUOp *body, PolyProgramInfo *info) {
  if (!ctx || !body || !info) return;
  info->global_size[0] = 1;
  info->global_size[1] = 1;
  info->global_size[2] = 1;
  info->local_size[0] = 1;
  info->local_size[1] = 1;
  info->local_size[2] = 1;
  info->has_local_size = true;

  int n_topo = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, body, &n_topo);
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (u && u->op == POLY_OP_DEFINE_VAR && u->arg.kind == POLY_ARG_DEFINE_VAR &&
        u->arg.define_var.name && strcmp(u->arg.define_var.name, "core_id") == 0) {
      int64_t n = u->arg.define_var.max_val + 1;
      if (n > 0 && n <= INT32_MAX) info->global_size[0] = (int)n;
      continue;
    }
    if (!u || u->op != POLY_OP_SPECIAL || u->n_src <= 0 || u->arg.kind != POLY_ARG_STRING)
      continue;
    const char *name = u->arg.str;
    if (!name || !name[0]) continue;
    int len = (int)strlen(name);
    int dim = (len > 0) ? name[len - 1] - '0' : 0;
    if (dim < 0 || dim > 2) dim = 0;
    PolyUOp *bound = u->src[0];
    int size = poly_launch_dim_upper_bound(ctx, bound);
    if (name[0] == 'i') info->has_local_size = false;
    if (name[0] == 'l') {
      if (info->has_local_size) {
        info->local_size[dim] = size;
        info->local_exprs[dim] = bound;
      }
    } else {
      info->global_size[dim] = size;
      info->global_exprs[dim] = bound;
    }
  }
  poly_toposort_free(topo);
}

static PolyUOp *poly_program_core_id_var(PolyCtx *ctx, PolyUOp *body) {
  if (!ctx || !body) return NULL;
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, body, &n_topo);
  if (!topo) return NULL;
  PolyUOp *out = NULL;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (u && u->op == POLY_OP_DEFINE_VAR && u->arg.kind == POLY_ARG_DEFINE_VAR &&
        u->arg.define_var.name && strcmp(u->arg.define_var.name, "core_id") == 0) {
      out = u;
      break;
    }
  }
  poly_toposort_free(topo);
  return out;
}

static int poly_call_get_outs_ins(PolyCtx *ctx, PolyUOp *call, bool *outs, bool *ins, int n_args) {
  if (!ctx || !call || call->n_src < 1) return -1;
  PolyUOp *body = call->src[0];
  if (body && body->op == POLY_OP_PROGRAM) {
    const PolyProgramInfo *info = poly_program_info(ctx, body);
    if (info) return poly_call_apply_program_info(info, outs, ins, n_args);
    return -1;
  }
  return poly_call_get_outs_ins_from_body(ctx, call, body, NULL, outs, ins, n_args);
}

static PolyProgramInfo *poly_program_info_build(
    PolyCtx *ctx,
    PolyUOp *call,
    PolyUOp *body,
    const char *program_name
) {
  if (!ctx || !call || !body || !program_name) return NULL;
  int n_args = poly_call_n_buffer_args(call);
  bool *globals = n_args > 0 ? calloc((size_t)n_args, sizeof(bool)) : NULL;
  bool *outs = n_args > 0 ? calloc((size_t)n_args, sizeof(bool)) : NULL;
  bool *ins = n_args > 0 ? calloc((size_t)n_args, sizeof(bool)) : NULL;
  if (n_args > 0 && (!globals || !outs || !ins)) {
    free(globals);
    free(outs);
    free(ins);
    return NULL;
  }
  if (poly_call_get_outs_ins_from_body(ctx, call, body, globals, outs, ins, n_args) != 0) {
    free(globals);
    free(outs);
    free(ins);
    return NULL;
  }

  PolyProgramInfo *info =
      poly_arena_alloc(ctx->arena, sizeof(PolyProgramInfo), _Alignof(PolyProgramInfo));
  if (!info) {
    free(globals);
    free(outs);
    free(ins);
    return NULL;
  }
  memset(info, 0, sizeof(*info));
  info->name = poly_schedule_arena_strdup(ctx, program_name);
  poly_program_info_collect_launch(ctx, body, info);

  int n_call_vars = poly_call_n_var_args(call);
  PolyUOp *core_id = poly_program_core_id_var(ctx, body);
  bool core_id_in_call = false;
  for (int i = 0; i < n_call_vars; i++) {
    if (poly_call_var_arg(call, i) == core_id) {
      core_id_in_call = true;
      break;
    }
  }
  int n_vars = n_call_vars + (core_id && !core_id_in_call ? 1 : 0);
  if (n_vars > 0) {
    info->vars = poly_arena_alloc(
        ctx->arena, (size_t)n_vars * sizeof(PolyUOp *), _Alignof(PolyUOp *)
    );
    if (!info->vars) {
      free(globals);
      free(outs);
      free(ins);
      return NULL;
    }
    for (int i = 0; i < n_call_vars; i++)
      info->vars[i] = poly_call_var_arg(call, i);
    if (core_id && !core_id_in_call) info->vars[n_call_vars] = core_id;
    info->n_vars = n_vars;
  }

  if (n_args > 0) {
    if (poly_call_mask_to_indices(ctx, globals, n_args, &info->globals, &info->n_globals) != 0) {
      free(globals);
      free(outs);
      free(ins);
      return NULL;
    }
  }

  if (poly_call_mask_to_indices(ctx, outs, n_args, &info->outs, &info->n_outs) != 0 ||
      poly_call_mask_to_indices(ctx, ins, n_args, &info->ins, &info->n_ins) != 0) {
    free(globals);
    free(outs);
    free(ins);
    return NULL;
  }
  free(globals);
  free(outs);
  free(ins);
  return info;
}

static PolyUOp *poly_program_from_call_body(
    PolyCtx *ctx,
    PolyUOp *call,
    PolyUOp *body,
    const char *name,
    PolyDevice device
) {
  if (!ctx || !call || call->op != POLY_OP_CALL || !body) return NULL;
  if (body->op == POLY_OP_PROGRAM) return body;

  const char *program_name = name ? name : "test";
  if (device == POLY_DEVICE_AUTO) device = poly_uop_device(body);

  PolyProgramInfo *info = poly_program_info_build(ctx, call, body, program_name);
  if (!info) return NULL;

  PolyUOp *dev = poly_uop0(ctx, POLY_OP_DEVICE, POLY_VOID, poly_arg_int((int64_t)device));
  PolyUOp *program_src[2] = {body, dev};
  PolyUOp *program =
      poly_uop(ctx, POLY_OP_PROGRAM, POLY_VOID, program_src, 2, poly_arg_program_info(info));
  if (!program) return NULL;
  return program;
}

PolyUOp *poly_program_from_call(PolyCtx *ctx, PolyUOp *call, const char *name) {
  if (!ctx || !call || call->op != POLY_OP_CALL || call->n_src < 1) return NULL;
  PolyUOp *body = call->src[0];
  if (!body) return NULL;
  if (body->op == POLY_OP_PROGRAM) return body;
  return poly_program_from_call_body(ctx, call, body, name, poly_uop_device(body));
}

static PolyUOp *poly_program_attach_linear(PolyCtx *ctx, PolyUOp *program) {
  if (!ctx || !program || program->op != POLY_OP_PROGRAM) return NULL;
  if (poly_program_linear(program)) return program;
  if (program->n_src < 2) return NULL;

  PolyUOp *body = poly_program_kernel_body(program);
  if (!body) return NULL;

  int n_lin = 0;
  PolyUOp **lin = poly_linearize_rewritten(ctx, body, &n_lin);
  if (!lin) return NULL;

  PolyUOp *linear = poly_uop(ctx, POLY_OP_LINEAR, POLY_VOID, lin, n_lin, poly_arg_none());
  free(lin);
  if (!linear) return NULL;

  PolyUOp *src[3] = {program->src[0], program->src[1], linear};
  PolyUOp *with_linear = poly_uop(ctx, POLY_OP_PROGRAM, POLY_VOID, src, 3, program->arg);
  return with_linear;
}

static PolyUOp *poly_program_attach_source(PolyCtx *ctx, PolyUOp *program, const char *source_text) {
  if (!ctx || !program || program->op != POLY_OP_PROGRAM) return NULL;
  if (poly_program_source(program)) return program;
  if (program->n_src != 3 || !poly_program_linear(program)) return NULL;
  PolyUOp *source = poly_uop0(ctx, POLY_OP_SOURCE, POLY_VOID, poly_arg_str(source_text));
  if (!source) return NULL;
  PolyUOp *src[4] = {program->src[0], program->src[1], program->src[2], source};
  return poly_uop(ctx, POLY_OP_PROGRAM, POLY_VOID, src, 4, program->arg);
}

static PolyUOp *poly_program_attach_binary(
    PolyCtx *ctx,
    PolyUOp *program,
    const uint8_t *bytes,
    int n_bytes
) {
  if (!ctx || !program || program->op != POLY_OP_PROGRAM || !bytes || n_bytes <= 0) return NULL;
  if (poly_program_binary(program)) return program;
  if (program->n_src != 4 || !poly_program_linear(program) || !poly_program_source(program)) return NULL;
  PolyUOp *binary = poly_uop0(ctx, POLY_OP_BINARY, POLY_VOID, poly_arg_bytes(bytes, n_bytes));
  if (!binary) return NULL;
  PolyUOp *src[5] = {program->src[0], program->src[1], program->src[2], program->src[3], binary};
  return poly_uop(ctx, POLY_OP_PROGRAM, POLY_VOID, src, 5, program->arg);
}

static int webgpu_collect_param_order(PolyUOp **lin, int n_lin, int *order, int n_params) {
  int seen = 0;
  for (int i = 0; i < n_lin; i++) {
    if (lin[i]->op != POLY_OP_PARAM) continue;
    if (seen >= n_params) return -1;
    int idx = (int)lin[i]->arg.i;
    if (idx < 0 || idx >= n_params) return -1;
    order[seen++] = idx;
  }
  return (seen == n_params) ? 0 : -1;
}

static int poly_schedule_slot_for_call_arg(
    const PolySchedule *sched,
    PolyUOp *call,
    int param_idx
) {
  if (!sched || !call || param_idx < 0) return -1;
  PolyUOp *buf = poly_call_buffer_arg(call, param_idx);
  for (int i = 0; i < sched->template->n_buf_slots; i++)
    if (sched->template->buf_slots[i].buf_uop == buf) return i;
  return -1;
}

static const PolyCallIO *poly_schedule_call_io(const PolySchedule *sched, int call_index) {
  if (!sched || !sched->run || !sched->run->call_io || call_index < 0 ||
      call_index >= sched->template->n_calls)
    return NULL;
  return &sched->run->call_io[call_index];
}

static void poly_call_access_free(PolyCallAccess *access) {
  if (!access) return;
  free(access->outs);
  free(access->ins);
  free(access->read_args);
  free(access->write_args);
  free(access->active_args);
  memset(access, 0, sizeof(*access));
}

static int clone_bool_array(bool **dst, const bool *src, int n) {
  if (!dst || n < 0) return -1;
  *dst = NULL;
  if (n == 0) return 0;
  if (!src) return -1;
  *dst = malloc((size_t)n * sizeof(bool));
  if (!*dst) return -1;
  memcpy(*dst, src, (size_t)n * sizeof(bool));
  return 0;
}

static int clone_int_array(int **dst, const int *src, int n) {
  if (!dst || n < 0) return -1;
  *dst = NULL;
  if (n == 0) return 0;
  if (!src) return -1;
  *dst = malloc((size_t)n * sizeof(int));
  if (!*dst) return -1;
  memcpy(*dst, src, (size_t)n * sizeof(int));
  return 0;
}

static int poly_call_access_clone(PolyCallAccess *dst, const PolyCallAccess *src) {
  if (!dst || !src || src->n_args < 0 || src->n_read_args < 0 || src->n_write_args < 0 ||
      src->n_active_args < 0)
    return -1;
  memset(dst, 0, sizeof(*dst));
  dst->n_args = src->n_args;
  dst->n_read_args = src->n_read_args;
  dst->n_write_args = src->n_write_args;
  dst->n_active_args = src->n_active_args;
  if (clone_bool_array(&dst->outs, src->outs, src->n_args) != 0 ||
      clone_bool_array(&dst->ins, src->ins, src->n_args) != 0 ||
      clone_int_array(&dst->read_args, src->read_args, src->n_read_args) != 0 ||
      clone_int_array(&dst->write_args, src->write_args, src->n_write_args) != 0 ||
      clone_int_array(&dst->active_args, src->active_args, src->n_active_args) != 0) {
    poly_call_access_free(dst);
    return -1;
  }
  return 0;
}

static PolyCallAccess *poly_call_access_clone_array(const PolyCallAccess *src, int n) {
  if (n < 0) return NULL;
  if (n == 0) return calloc(1, sizeof(PolyCallAccess));
  if (!src) return NULL;
  PolyCallAccess *out = calloc((size_t)n, sizeof(PolyCallAccess));
  if (!out) return NULL;
  for (int i = 0; i < n; i++) {
    if (poly_call_access_clone(&out[i], &src[i]) != 0) {
      for (int j = 0; j < i; j++)
        poly_call_access_free(&out[j]);
      free(out);
      return NULL;
    }
  }
  return out;
}

static void poly_call_io_free(PolyCallIO *io) {
  if (!io) return;
  free(io->arg_to_slot);
  memset(io, 0, sizeof(*io));
}

static int poly_call_io_clone_resolved(PolyCallIO *dst, const PolyCallIO *src) {
  if (!dst || !src) return -1;
  dst->n_args = src->n_args;
  dst->access = src->access;
  if (src->n_args <= 0) return 0;
  if (!src->arg_to_slot || !src->access) return -1;
  dst->arg_to_slot = malloc((size_t)src->n_args * sizeof(int));
  if (!dst->arg_to_slot) return -1;
  memcpy(dst->arg_to_slot, src->arg_to_slot, (size_t)src->n_args * sizeof(int));
  return 0;
}

static int poly_call_access_build_arg_lists(PolyCallAccess *access) {
  if (!access || access->n_args < 0 || !access->outs || !access->ins) return -1;

  free(access->read_args);
  free(access->write_args);
  free(access->active_args);
  access->read_args = NULL;
  access->write_args = NULL;
  access->active_args = NULL;
  access->n_read_args = 0;
  access->n_write_args = 0;
  access->n_active_args = 0;

  for (int i = 0; i < access->n_args; i++) {
    if (access->ins[i]) access->n_read_args++;
    if (access->outs[i]) access->n_write_args++;
    if (access->outs[i] || access->ins[i]) access->n_active_args++;
  }

  if (access->n_read_args > 0) {
    access->read_args = malloc((size_t)access->n_read_args * sizeof(int));
    if (!access->read_args) return -1;
  }
  if (access->n_write_args > 0) {
    access->write_args = malloc((size_t)access->n_write_args * sizeof(int));
    if (!access->write_args) return -1;
  }
  if (access->n_active_args > 0) {
    access->active_args = malloc((size_t)access->n_active_args * sizeof(int));
    if (!access->active_args) return -1;
  }

  int r = 0, w = 0, a = 0;
  for (int i = 0; i < access->n_args; i++) {
    if (access->ins[i]) access->read_args[r++] = i;
    if (access->outs[i]) access->write_args[w++] = i;
    if (access->outs[i] || access->ins[i]) access->active_args[a++] = i;
  }
  return 0;
}

static int poly_verify_parameterized_compute_call(PolyCtx *ctx, PolyUOp *call, int call_index) {
  if (!ctx || !call || call->op != POLY_OP_CALL || call->n_src < 1) return -1;
  if (poly_call_is_copy(call) || poly_call_is_view(call)) return 0;

  PolyUOp *body = poly_program_body(call->src[0]);
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, body, &n_topo);
  if (!topo && n_topo > 0) return -1;

  int n_args = poly_call_n_buffer_args(call);
  int bad_arg = -1;
  PolyUOp *bad_raw_buffer = NULL;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (!u) continue;
    if (u->op == POLY_OP_BUFFER) {
      bad_raw_buffer = u;
      break;
    }
    for (int a = 0; a < n_args; a++) {
      if (u != poly_call_buffer_arg(call, a)) continue;
      bad_arg = a;
      break;
    }
    if (bad_arg >= 0) break;
  }
  poly_toposort_free(topo);

  if (bad_raw_buffer) {
    fprintf(
        stderr,
        "polygrad: schedule: compute CALL %d body contains raw BUFFER %p; "
        "scheduled bodies must use PARAM placeholders\n",
        call_index, (void *)bad_raw_buffer
    );
    return -1;
  }
  if (bad_arg >= 0) {
    fprintf(
        stderr,
        "polygrad: schedule: compute CALL %d body contains external arg %d directly; "
        "scheduled bodies must be parameterized\n",
        call_index, bad_arg
    );
    return -1;
  }
  return 0;
}

static bool poly_schedule_validation_enabled(void) {
  const char *v = getenv("POLY_VALIDATE_SCHEDULE");
  return v && v[0] && strcmp(v, "0") != 0;
}

static int poly_call_io_init(PolyCtx *ctx, PolySchedule *sched, int call_index) {
  if (!ctx || !sched || !sched->run || call_index < 0 || call_index >= sched->template->n_calls ||
      !sched->template->call_access || !sched->run->call_io)
    return -1;
  PolyUOp *call = poly_schedule_call(sched, call_index);
  if (poly_schedule_validation_enabled() &&
      poly_verify_parameterized_compute_call(ctx, call, call_index) != 0)
    return -1;
  PolyCallAccess *access = &sched->template->call_access[call_index];
  PolyCallIO *io = &sched->run->call_io[call_index];
  int n_args = poly_call_n_buffer_args(call);
  bool access_precomputed =
      access->n_args != 0 || access->outs || access->ins || access->read_args ||
      access->write_args || access->active_args;
  if (access_precomputed) {
    if (access->n_args != n_args || (n_args > 0 && (!access->outs || !access->ins)))
      return -1;
    bool lists_absent = !access->read_args && !access->write_args && !access->active_args;
    bool lists_incomplete =
        (access->n_read_args > 0 && !access->read_args) ||
        (access->n_write_args > 0 && !access->write_args) ||
        (access->n_active_args > 0 && !access->active_args);
    if (lists_incomplete) return -1;
    if (lists_absent && poly_call_access_build_arg_lists(access) != 0) return -1;
  } else {
    access->n_args = n_args;
  }
  io->n_args = access->n_args;
  io->access = access;
  if (access->n_args <= 0) return 0;

  io->arg_to_slot = malloc((size_t)access->n_args * sizeof(int));
  if (!io->arg_to_slot) return -1;
  if (!access_precomputed) {
    access->outs = calloc((size_t)access->n_args, sizeof(bool));
    access->ins = calloc((size_t)access->n_args, sizeof(bool));
    if (!access->outs || !access->ins) return -1;
  }

  for (int i = 0; i < access->n_args; i++) {
    io->arg_to_slot[i] = poly_schedule_slot_for_call_arg(sched, call, i);
    if (io->arg_to_slot[i] < 0 || io->arg_to_slot[i] >= sched->template->n_buf_slots) return -1;
  }
  if (!access_precomputed) {
    if (poly_call_get_outs_ins(ctx, call, access->outs, access->ins, access->n_args) != 0)
      return -1;
    return poly_call_access_build_arg_lists(access);
  }
  return 0;
}

static bool poly_memory_plan_enabled(void) {
#ifdef __EMSCRIPTEN__
  return false;
#else
  return !poly_getenv_flag("POLY_NO_MEMORY_PLANNER");
#endif
}

static size_t poly_round_up_size(size_t x, size_t block) {
  if (block == 0) return x;
  size_t r = x % block;
  return r ? x + (block - r) : x;
}

typedef struct {
  size_t *starts;
  int n;
  int cap;
} PolyTLSFBucket;

typedef struct {
  size_t start;
  size_t size;
  size_t next;
  size_t prev;
  bool has_prev;
  bool is_free;
  bool alive;
} PolyTLSFBlock;

typedef struct {
  size_t size;
  size_t base;
  size_t block_size;
  int l2_bits;
  int n_l2;
  int n_l1;
  PolyTLSFBucket *buckets;
  int *lv1_entries;
  PolyTLSFBlock *blocks;
  int n_blocks;
  int cap_blocks;
} PolyTLSF;

typedef struct {
  PolyDevice device;
  bool copy_lane;
  size_t peak;
  PolyTLSF *alloc;
  int arena_slot;
} PolyMemLane;

typedef struct {
  int slot;
  int first;
  int last;
  bool copy_lane;
  PolyDevice device;
  size_t nbytes;
  size_t alloc_size;
  size_t offset;
  int lane;
  PolyUOp *view;
} PolyMemItem;

typedef struct {
  int t;
  bool open;
  int item;
} PolyMemEvent;

static int poly_mem_event_cmp(const void *a, const void *b) {
  const PolyMemEvent *ea = a, *eb = b;
  if (ea->t != eb->t) return (ea->t < eb->t) ? -1 : 1;
  if (ea->open != eb->open) return ea->open ? 1 : -1; /* closes before opens */
  return ea->item - eb->item;
}

static int poly_size_bit_length(size_t x) {
  int n = 0;
  while (x) {
    n++;
    x >>= 1;
  }
  return n;
}

static int poly_int_bit_length(int x) {
  int n = 0;
  unsigned int u = (unsigned int)x;
  while (u) {
    n++;
    u >>= 1;
  }
  return n;
}

static size_t poly_pow2_size(int exp) {
  if (exp <= 0) return 1;
  int bits = (int)(sizeof(size_t) * CHAR_BIT);
  if (exp >= bits) return (size_t)1 << (bits - 1);
  return (size_t)1 << exp;
}

static int poly_tlsf_lv1(PolyTLSF *a, size_t size) {
  (void)a;
  return poly_size_bit_length(size);
}

static int poly_tlsf_lv2(PolyTLSF *a, size_t size) {
  int bl = poly_size_bit_length(size);
  if (bl <= 0) return 0;
  size_t base = poly_pow2_size(bl - 1);
  size_t denom = poly_pow2_size(bl - a->l2_bits);
  int v = (int)((size - base) / denom);
  if (v < 0) return 0;
  if (v >= a->n_l2) return a->n_l2 - 1;
  return v;
}

static PolyTLSFBucket *poly_tlsf_bucket(PolyTLSF *a, size_t size) {
  int l1 = poly_tlsf_lv1(a, size);
  int l2 = poly_tlsf_lv2(a, size);
  if (l1 < 0 || l1 >= a->n_l1 || l2 < 0 || l2 >= a->n_l2) return NULL;
  return &a->buckets[l1 * a->n_l2 + l2];
}

static int poly_tlsf_bucket_push(PolyTLSFBucket *b, size_t start) {
  if (b->n >= b->cap) {
    int new_cap = b->cap ? b->cap * 2 : 4;
    size_t *ns = realloc(b->starts, (size_t)new_cap * sizeof(size_t));
    if (!ns) return -1;
    b->starts = ns;
    b->cap = new_cap;
  }
  b->starts[b->n++] = start;
  return 0;
}

static int poly_tlsf_bucket_remove(PolyTLSFBucket *b, size_t start) {
  if (!b) return -1;
  for (int i = 0; i < b->n; i++) {
    if (b->starts[i] != start) continue;
    memmove(b->starts + i, b->starts + i + 1, (size_t)(b->n - i - 1) * sizeof(size_t));
    b->n--;
    return 0;
  }
  return -1;
}

static int poly_tlsf_find_block(PolyTLSF *a, size_t start) {
  if (!a) return -1;
  for (int i = 0; i < a->n_blocks; i++)
    if (a->blocks[i].alive && a->blocks[i].start == start) return i;
  return -1;
}

static int poly_tlsf_add_or_update_block(
    PolyTLSF *a,
    size_t start,
    size_t size,
    size_t prev,
    bool has_prev,
    bool is_free
) {
  int idx = poly_tlsf_find_block(a, start);
  if (idx < 0) {
    if (a->n_blocks >= a->cap_blocks) {
      int new_cap = a->cap_blocks ? a->cap_blocks * 2 : 16;
      PolyTLSFBlock *nb = realloc(a->blocks, (size_t)new_cap * sizeof(PolyTLSFBlock));
      if (!nb) return -1;
      a->blocks = nb;
      a->cap_blocks = new_cap;
    }
    idx = a->n_blocks++;
  }
  a->blocks[idx] = (PolyTLSFBlock){
      .start = start,
      .size = size,
      .next = start + size,
      .prev = prev,
      .has_prev = has_prev,
      .is_free = is_free,
      .alive = true,
  };
  return idx;
}

static int poly_tlsf_insert_block(PolyTLSF *a, size_t start, size_t size, size_t prev, bool has_prev) {
  int existing = poly_tlsf_find_block(a, start);
  if (!has_prev && existing >= 0) {
    has_prev = a->blocks[existing].has_prev;
    prev = a->blocks[existing].prev;
  }
  PolyTLSFBucket *bucket = poly_tlsf_bucket(a, size);
  if (!bucket) return -1;
  if (poly_tlsf_bucket_push(bucket, start) != 0) return -1;
  int l1 = poly_tlsf_lv1(a, size);
  if (l1 >= 0 && l1 < a->n_l1) a->lv1_entries[l1]++;
  return poly_tlsf_add_or_update_block(a, start, size, prev, has_prev, true) >= 0 ? 0 : -1;
}

static int poly_tlsf_remove_block(PolyTLSF *a, size_t start, size_t size, size_t prev, bool has_prev) {
  int existing = poly_tlsf_find_block(a, start);
  if (!has_prev && existing >= 0) {
    has_prev = a->blocks[existing].has_prev;
    prev = a->blocks[existing].prev;
  }
  PolyTLSFBucket *bucket = poly_tlsf_bucket(a, size);
  if (!bucket || poly_tlsf_bucket_remove(bucket, start) != 0) return -1;
  int l1 = poly_tlsf_lv1(a, size);
  if (l1 >= 0 && l1 < a->n_l1) a->lv1_entries[l1]--;
  return poly_tlsf_add_or_update_block(a, start, size, prev, has_prev, false) >= 0 ? 0 : -1;
}

static void poly_tlsf_delete_block(PolyTLSF *a, size_t start) {
  int idx = poly_tlsf_find_block(a, start);
  if (idx >= 0) a->blocks[idx].alive = false;
}

static int poly_tlsf_split_block(PolyTLSF *a, size_t start, size_t size, size_t new_size) {
  if (!a || new_size == 0 || new_size >= size) return -1;
  int idx = poly_tlsf_find_block(a, start);
  if (idx < 0 || !a->blocks[idx].is_free) return -1;
  size_t nxt = a->blocks[idx].next;
  size_t prev = a->blocks[idx].prev;
  bool has_prev = a->blocks[idx].has_prev;
  if (poly_tlsf_remove_block(a, start, size, prev, has_prev) != 0) return -1;
  if (poly_tlsf_insert_block(a, start, new_size, prev, has_prev) != 0) return -1;
  if (poly_tlsf_insert_block(a, start + new_size, size - new_size, start, true) != 0) return -1;
  int nidx = poly_tlsf_find_block(a, nxt);
  if (nidx >= 0) {
    a->blocks[nidx].prev = start + new_size;
    a->blocks[nidx].has_prev = true;
  }
  return 0;
}

static int poly_tlsf_merge_right(PolyTLSF *a, size_t start) {
  int idx = poly_tlsf_find_block(a, start);
  if (idx < 0 || !a->blocks[idx].is_free) return -1;
  size_t size = a->blocks[idx].size;
  size_t prev = a->blocks[idx].prev;
  bool has_prev = a->blocks[idx].has_prev;
  size_t nxt = a->blocks[idx].next;

  while (true) {
    int nidx = poly_tlsf_find_block(a, nxt);
    if (nidx < 0 || !a->blocks[nidx].is_free) break;
    size_t nsize = a->blocks[nidx].size;
    size_t nnxt = a->blocks[nidx].next;
    if (poly_tlsf_remove_block(a, start, size, prev, has_prev) != 0) return -1;
    if (poly_tlsf_remove_block(a, nxt, nsize, start, true) != 0) return -1;
    if (poly_tlsf_insert_block(a, start, size + nsize, prev, has_prev) != 0) return -1;
    poly_tlsf_delete_block(a, nxt);
    size += nsize;
    nxt = nnxt;
  }

  int next_idx = poly_tlsf_find_block(a, nxt);
  if (next_idx >= 0) {
    a->blocks[next_idx].prev = start;
    a->blocks[next_idx].has_prev = true;
  }
  return 0;
}

static int poly_tlsf_merge_block(PolyTLSF *a, size_t start) {
  int idx = poly_tlsf_find_block(a, start);
  if (idx < 0) return -1;
  while (a->blocks[idx].has_prev) {
    int pidx = poly_tlsf_find_block(a, a->blocks[idx].prev);
    if (pidx < 0 || !a->blocks[pidx].is_free) break;
    start = a->blocks[pidx].start;
    idx = pidx;
  }
  return poly_tlsf_merge_right(a, start);
}

static PolyTLSF *poly_tlsf_new(size_t size, size_t block_size, int lv2_cnt) {
  PolyTLSF *a = calloc(1, sizeof(PolyTLSF));
  if (!a) return NULL;
  a->size = size;
  a->block_size = block_size ? block_size : 16;
  a->l2_bits = poly_int_bit_length(lv2_cnt > 0 ? lv2_cnt : 16);
  a->n_l2 = 1 << a->l2_bits;
  a->n_l1 = poly_size_bit_length(size) + 1;
  if (a->n_l1 <= 0) a->n_l1 = 1;
  a->buckets = calloc((size_t)a->n_l1 * (size_t)a->n_l2, sizeof(PolyTLSFBucket));
  a->lv1_entries = calloc((size_t)a->n_l1, sizeof(int));
  if (!a->buckets || !a->lv1_entries) {
    free(a->buckets);
    free(a->lv1_entries);
    free(a);
    return NULL;
  }
  if (size > 0 && poly_tlsf_insert_block(a, 0, size, 0, false) != 0) {
    for (int i = 0; i < a->n_l1 * a->n_l2; i++) free(a->buckets[i].starts);
    free(a->buckets);
    free(a->lv1_entries);
    free(a->blocks);
    free(a);
    return NULL;
  }
  return a;
}

static void poly_tlsf_destroy(PolyTLSF *a) {
  if (!a) return;
  for (int i = 0; i < a->n_l1 * a->n_l2; i++) free(a->buckets[i].starts);
  free(a->buckets);
  free(a->lv1_entries);
  free(a->blocks);
  free(a);
}

static size_t poly_tlsf_alloc(PolyTLSF *a, size_t req_size, size_t align) {
  if (!a) return (size_t)-1;
  if (align == 0) align = 1;
  if (req_size < a->block_size) req_size = a->block_size;
  size_t size = req_size + align - 1;
  if (size < a->block_size) size = a->block_size;
  int bl = poly_size_bit_length(size);
  size_t bucket = poly_pow2_size(bl - a->l2_bits);
  size = poly_round_up_size(size, bucket);

  int start_l1 = poly_tlsf_lv1(a, size);
  int size_bl = poly_size_bit_length(size);
  for (int l1 = start_l1; l1 < a->n_l1; l1++) {
    if (a->lv1_entries[l1] == 0) continue;
    int l2_start = (l1 == size_bl) ? poly_tlsf_lv2(a, size) : 0;
    for (int l2 = l2_start; l2 < a->n_l2; l2++) {
      PolyTLSFBucket *bucket_list = &a->buckets[l1 * a->n_l2 + l2];
      if (bucket_list->n <= 0) continue;

      size_t start = bucket_list->starts[0];
      int idx = poly_tlsf_find_block(a, start);
      if (idx < 0) return (size_t)-1;
      size_t nsize = a->blocks[idx].size;
      if (nsize < size) continue;

      size_t new_start = poly_round_up_size(start, align);
      if (new_start != start) {
        if (poly_tlsf_split_block(a, start, nsize, new_start - start) != 0) return (size_t)-1;
        start = new_start;
        idx = poly_tlsf_find_block(a, start);
        if (idx < 0) return (size_t)-1;
        nsize = a->blocks[idx].size;
      }

      if (nsize > req_size && poly_tlsf_split_block(a, start, nsize, req_size) != 0)
        return (size_t)-1;
      if (poly_tlsf_remove_block(a, start, req_size, 0, false) != 0) return (size_t)-1;
      return start + a->base;
    }
  }
  return (size_t)-1;
}

static int poly_tlsf_free(PolyTLSF *a, size_t start) {
  if (!a || start < a->base) return -1;
  start -= a->base;
  int idx = poly_tlsf_find_block(a, start);
  if (idx < 0) return -1;
  size_t size = a->blocks[idx].size;
  size_t prev = a->blocks[idx].prev;
  bool has_prev = a->blocks[idx].has_prev;
  if (poly_tlsf_insert_block(a, start, size, prev, has_prev) != 0) return -1;
  return poly_tlsf_merge_block(a, start);
}

static int poly_mem_lane_for(PolyMemLane **lanes, int *n_lanes, int *cap_lanes, PolyDevice dev, bool copy_lane) {
  for (int i = 0; i < *n_lanes; i++)
    if ((*lanes)[i].device == dev && (*lanes)[i].copy_lane == copy_lane) return i;
  if (*n_lanes >= *cap_lanes) {
    int new_cap = *cap_lanes ? *cap_lanes * 2 : 4;
    PolyMemLane *nl = realloc(*lanes, (size_t)new_cap * sizeof(PolyMemLane));
    if (!nl) return -1;
    memset(nl + *cap_lanes, 0, (size_t)(new_cap - *cap_lanes) * sizeof(PolyMemLane));
    *lanes = nl;
    *cap_lanes = new_cap;
  }
  int idx = (*n_lanes)++;
  (*lanes)[idx] = (PolyMemLane){
      .device = dev,
      .copy_lane = copy_lane,
      .arena_slot = -1,
  };
  return idx;
}

static int poly_schedule_slot_index_for_uop(PolyScheduleTemplate *tpl, PolyUOp *u) {
  if (!tpl || !u) return -1;
  for (int i = 0; i < tpl->n_buf_slots; i++)
    if (tpl->buf_slots[i].buf_uop == u) return i;
  return -1;
}

static int poly_schedule_memory_plan(PolyCtx *ctx, PolySchedule *sched) {
  if (!poly_memory_plan_enabled() || !ctx || !sched || !sched->template ||
      !sched->template->linear || sched->template->n_calls <= 1)
    return 0;

  PolyScheduleTemplate *tpl = sched->template;
  int n_slots_orig = tpl->n_buf_slots;
  if (n_slots_orig <= 0) return 0;

  int *slot_to_item = malloc((size_t)n_slots_orig * sizeof(int));
  if (!slot_to_item) return -1;
  for (int i = 0; i < n_slots_orig; i++) slot_to_item[i] = -1;

  PolyMemItem *items = NULL;
  int n_items = 0, cap_items = 0;
  for (int s = 0; s < n_slots_orig; s++) {
    PolyScheduleBufSlot *slot = &tpl->buf_slots[s];
    if (!slot->is_intermediate || slot->is_memory_arena || slot->has_memory_parent ||
        !slot->buf_uop || slot->nbytes <= 0)
      continue;
    if (n_items >= cap_items) {
      int new_cap = cap_items ? cap_items * 2 : 8;
      PolyMemItem *ni = realloc(items, (size_t)new_cap * sizeof(PolyMemItem));
      if (!ni) {
        free(slot_to_item);
        free(items);
        return -1;
      }
      items = ni;
      cap_items = new_cap;
    }
    slot_to_item[s] = n_items;
    items[n_items++] = (PolyMemItem){
        .slot = s,
        .first = INT_MAX,
        .last = -1,
        .device = slot->device,
        .nbytes = (size_t)slot->nbytes,
        .alloc_size = poly_round_up_size((size_t)slot->nbytes, 256),
        .lane = -1,
    };
  }
  if (n_items == 0) {
    free(slot_to_item);
    free(items);
    return 0;
  }

  for (int k = 0; k < tpl->n_calls; k++) {
    PolyUOp *call = tpl->linear->src[k];
    bool is_copy = poly_call_is_copy(call);
    int n_args = poly_call_n_buffer_args(call);
    for (int a = 0; a < n_args; a++) {
      PolyUOp *arg = poly_call_buffer_arg(call, a);
      int slot = poly_schedule_slot_index_for_uop(tpl, arg);
      if (slot < 0 || slot >= n_slots_orig) continue;
      int item = slot_to_item[slot];
      if (item < 0) continue;
      if (items[item].first == INT_MAX) items[item].first = k;
      items[item].last = k;
      if (is_copy) items[item].copy_lane = true;
    }
  }

  int compact = 0;
  for (int i = 0; i < n_items; i++) {
    if (items[i].first == INT_MAX || items[i].last < items[i].first) {
      slot_to_item[items[i].slot] = -1;
      continue;
    }
    if (compact != i) {
      items[compact] = items[i];
      slot_to_item[items[compact].slot] = compact;
    }
    compact++;
  }
  n_items = compact;
  if (n_items <= 1) {
    free(slot_to_item);
    free(items);
    return 0;
  }

  PolyMemLane *lanes = NULL;
  int n_lanes = 0, cap_lanes = 0;
  size_t total_memory = 0;
  for (int i = 0; i < n_items; i++) {
    total_memory += items[i].alloc_size;
    items[i].lane =
        poly_mem_lane_for(&lanes, &n_lanes, &cap_lanes, items[i].device, items[i].copy_lane);
    if (items[i].lane < 0) goto fail;
  }
  total_memory *= 2;
  if (total_memory == 0) total_memory = 256;
  for (int l = 0; l < n_lanes; l++) {
    lanes[l].alloc = poly_tlsf_new(total_memory, 256, 32);
    if (!lanes[l].alloc) goto fail;
  }

  PolyMemEvent *events = calloc((size_t)n_items * 2, sizeof(PolyMemEvent));
  if (!events) goto fail;
  for (int i = 0; i < n_items; i++) {
    int hold = items[i].copy_lane ? (items[i].last - items[i].first + 1) : 0;
    events[2 * i] = (PolyMemEvent){items[i].first, true, i};
    events[2 * i + 1] = (PolyMemEvent){items[i].last + 1 + hold, false, i};
  }
  qsort(events, (size_t)n_items * 2, sizeof(PolyMemEvent), poly_mem_event_cmp);

  for (int e = 0; e < n_items * 2; e++) {
    PolyMemItem *item = &items[events[e].item];
    PolyMemLane *lane = &lanes[item->lane];
    if (events[e].open) {
      item->offset = poly_tlsf_alloc(lane->alloc, item->alloc_size, 1);
      if (item->offset == (size_t)-1) {
        free(events);
        goto fail;
      }
      size_t end = item->offset + item->nbytes;
      if (end > lane->peak) lane->peak = end;
    } else if (poly_tlsf_free(lane->alloc, item->offset) != 0) {
      free(events);
      goto fail;
    }
  }
  free(events);

  int n_new_slots = n_lanes;
  PolyScheduleBufSlot *new_slots =
      realloc(tpl->buf_slots, (size_t)(n_slots_orig + n_new_slots) * sizeof(PolyScheduleBufSlot));
  if (!new_slots) goto fail;
  tpl->buf_slots = new_slots;
  memset(tpl->buf_slots + n_slots_orig, 0, (size_t)n_new_slots * sizeof(PolyScheduleBufSlot));
  tpl->n_buf_slots = n_slots_orig + n_new_slots;

  for (int l = 0; l < n_lanes; l++) {
    size_t arena_size = poly_round_up_size(lanes[l].peak, 256);
    if (arena_size == 0) arena_size = 256;
    PolyUOp *arena = poly_buffer(ctx, POLY_UINT8, (int64_t)arena_size);
    if (!arena) goto fail_after_resize;
    int slot = n_slots_orig + l;
    lanes[l].arena_slot = slot;
    tpl->buf_slots[slot] = (PolyScheduleBufSlot){
        .dtype = POLY_UINT8,
        .numel = (int64_t)arena_size,
        .nbytes = (int64_t)arena_size,
        .buf_uop = arena,
        .device = lanes[l].device,
        .is_intermediate = true,
        .needs_zero = false,
        .external_buf_idx = -1,
        .is_memory_arena = true,
    };
  }

  PolyUOp **old_uops = calloc((size_t)n_items, sizeof(PolyUOp *));
  if (!old_uops) goto fail_after_resize;
  for (int i = 0; i < n_items; i++) {
    PolyScheduleBufSlot *slot = &tpl->buf_slots[items[i].slot];
    old_uops[i] = slot->buf_uop;
    int64_t view_arg_vals[2] = {slot->numel, (int64_t)items[i].offset};
    PolyArg view_arg = {.kind = POLY_ARG_INT_TUPLE, .int_tuple = {view_arg_vals, 2}};
    PolyUOp *view_src[2] = {tpl->buf_slots[lanes[items[i].lane].arena_slot].buf_uop, old_uops[i]};
    PolyUOp *view = poly_uop(ctx, POLY_OP_BUFFER_VIEW, slot->dtype, view_src, 2, view_arg);
    if (!view) {
      free(old_uops);
      goto fail_after_resize;
    }
    items[i].view = view;
    slot->buf_uop = view;
    slot->has_memory_parent = true;
    slot->memory_parent_slot = lanes[items[i].lane].arena_slot;
    slot->memory_offset = (int64_t)items[i].offset;
    if (slot->needs_zero) {
      slot->has_zero_before_call = true;
      slot->zero_before_call = items[i].first;
    }
  }

  PolyUOp **new_linear_src = calloc((size_t)tpl->linear->n_src, sizeof(PolyUOp *));
  if (!new_linear_src) {
    free(old_uops);
    goto fail_after_resize;
  }
  for (int k = 0; k < tpl->linear->n_src; k++) {
    PolyUOp *call = tpl->linear->src[k];
    bool changed = false;
    PolyUOp **call_src = calloc((size_t)call->n_src, sizeof(PolyUOp *));
    if (!call_src) {
      free(new_linear_src);
      free(old_uops);
      goto fail_after_resize;
    }
    for (int s = 0; s < call->n_src; s++) {
      PolyUOp *src = call->src[s];
      if (s > 0 && !poly_call_arg_is_var(src)) {
        for (int i = 0; i < n_items; i++) {
          if (src == old_uops[i]) {
            src = items[i].view;
            changed = true;
            break;
          }
        }
      }
      call_src[s] = src;
    }
    new_linear_src[k] = changed ? poly_uop(ctx, POLY_OP_CALL, POLY_VOID, call_src, call->n_src, call->arg) : call;
    free(call_src);
    if (!new_linear_src[k]) {
      free(new_linear_src);
      free(old_uops);
      goto fail_after_resize;
    }
  }
  tpl->linear = poly_uop(ctx, POLY_OP_LINEAR, POLY_VOID, new_linear_src, tpl->linear->n_src, tpl->linear->arg);
  free(new_linear_src);
  free(old_uops);
  if (!tpl->linear) goto fail_after_resize;

  if (poly_debug_at_least(2)) {
    size_t old_bytes = 0, arena_bytes = 0;
    for (int i = 0; i < n_items; i++) old_bytes += items[i].alloc_size;
    for (int l = 0; l < n_lanes; l++) arena_bytes += poly_round_up_size(lanes[l].peak, 256);
    fprintf(
        stderr,
        "[polygrad:memory-plan] planned %d buffers into %d arenas: %.3f MB -> %.3f MB\n",
        n_items, n_lanes, old_bytes / 1000000.0, arena_bytes / 1000000.0
    );
  }

  for (int l = 0; l < n_lanes; l++) poly_tlsf_destroy(lanes[l].alloc);
  free(lanes);
  free(slot_to_item);
  free(items);
  return 0;

fail_after_resize:
  /* The template is unusable if arena/view construction failed after resizing. */
fail:
  if (lanes) {
    for (int l = 0; l < n_lanes; l++) poly_tlsf_destroy(lanes[l].alloc);
  }
  free(lanes);
  free(slot_to_item);
  free(items);
  return -1;
}

int poly_schedule_call_buffer_slot(const PolySchedule *schedule, int call_index, int arg_index) {
  const PolyCallIO *io = poly_schedule_call_io(schedule, call_index);
  if (!io || arg_index < 0 || arg_index >= io->n_args || !io->arg_to_slot) return -1;
  return io->arg_to_slot[arg_index];
}

int poly_schedule_external_slot_count(const PolySchedule *schedule) {
  if (!schedule || !schedule->template) return 0;
  int n = 0;
  for (int i = 0; i < schedule->template->n_buf_slots; i++) {
    const PolyScheduleBufSlot *slot = &schedule->template->buf_slots[i];
    if (slot->external_buf_idx >= n) n = slot->external_buf_idx + 1;
  }
  return n;
}

PolyUOp *poly_schedule_external_slot_buffer(const PolySchedule *schedule, int external_index) {
  if (!schedule || !schedule->template || external_index < 0) return NULL;
  for (int i = 0; i < schedule->template->n_buf_slots; i++) {
    const PolyScheduleBufSlot *slot = &schedule->template->buf_slots[i];
    if (slot->external_buf_idx == external_index) return slot->buf_uop;
  }
  return NULL;
}

static int webgpu_fill_param_slots(
    PolyCtx *ctx,
    const PolySchedule *sched,
    PolyUOp *call,
    int call_index,
    PolyRunner *runner
) {
  int n_params = poly_call_n_buffer_args(call);
  runner->n_params = n_params;
  if (n_params <= 0) return 0;

  int n_lin = 0;
  PolyUOp *body = poly_schedule_call_body(sched, call_index);
  PolyUOp **lin = poly_linearize_webgpu(ctx, body, &n_lin);
  if (!lin) return -1;

  int *param_order = malloc((size_t)n_params * sizeof(int));
  int *param_to_slot = malloc((size_t)n_params * sizeof(int));
  if (!param_order || !param_to_slot ||
      webgpu_collect_param_order(lin, n_lin, param_order, n_params) != 0) {
    free(param_order);
    free(param_to_slot);
    free(lin);
    return -1;
  }

  for (int i = 0; i < n_params; i++) {
    int src_idx = param_order[i];
    param_to_slot[i] = poly_schedule_call_buffer_slot(sched, call_index, src_idx);
    if (param_to_slot[i] < 0) {
      free(param_order);
      free(param_to_slot);
      free(lin);
      return -1;
    }
  }

  free(param_order);
  free(lin);
  runner->param_to_slot = param_to_slot;
  return 0;
}

static bool poly_lookup_var_value(
    PolyUOp *var,
    const PolyVarBinding *bindings,
    int n_bindings,
    PolyArg *out
) {
  if (!var || !bindings || !out) return false;
  for (int i = 0; i < n_bindings; i++) {
    if (bindings[i].var != var) continue;
    if (poly_dtype_is_bool(var->dtype))
      *out = poly_arg_bool(bindings[i].value != 0);
    else if (poly_dtype_is_float(var->dtype))
      *out = poly_arg_float((double)bindings[i].value);
    else
      *out = poly_arg_int(bindings[i].value);
    return true;
  }
  return false;
}

static bool poly_eval_launch_expr(
    PolyUOp *u,
    const PolyVarBinding *bindings,
    int n_bindings,
    PolyArg *out
) {
  if (!u || !out) return false;

  if (u->op == POLY_OP_CONST) {
    *out = u->arg;
    return true;
  }
  if (u->op == POLY_OP_DEFINE_VAR) return poly_lookup_var_value(u, bindings, n_bindings, out);

  if (u->op == POLY_OP_CAST) {
    PolyArg operand;
    if (u->n_src < 1 || !poly_eval_launch_expr(u->src[0], bindings, n_bindings, &operand))
      return false;
    *out = poly_exec_alu(POLY_OP_CAST, u->dtype, &operand, 1);
    return out->kind != POLY_ARG_INVALID;
  }

  if (poly_opset_has(POLY_GROUP_ALU, u->op)) {
    if (u->n_src > 3) return false;
    PolyArg operands[3];
    for (int i = 0; i < u->n_src; i++) {
      if (!poly_eval_launch_expr(u->src[i], bindings, n_bindings, &operands[i])) return false;
    }
    *out = poly_exec_alu(u->op, u->dtype, operands, u->n_src);
    return out->kind != POLY_ARG_INVALID;
  }

  return false;
}

static int64_t poly_launch_arg_to_i64(PolyArg arg) {
  switch (arg.kind) {
  case POLY_ARG_BOOL:
    return arg.b ? 1 : 0;
  case POLY_ARG_FLOAT:
    return (int64_t)arg.f;
  case POLY_ARG_INT:
  default:
    return arg.i;
  }
}

static int poly_launch_dim_upper_bound(PolyCtx *ctx, PolyUOp *expr) {
  if (!expr) return 1;
  int64_t lo = 0, hi = 1;
  poly_uop_minmax(ctx, expr, &lo, &hi);
  if (hi <= 0) return 1;
  if (hi > INT32_MAX) return INT32_MAX;
  return (int)hi;
}

static int poly_resolve_runner_launch_dims(
    PolyRunner *runner,
    const PolyVarBinding *bindings,
    int n_bindings
) {
  if (!runner) return -1;

  for (int dim = 0; dim < 3; dim++) {
    if (runner->grid_exprs[dim]) {
      PolyArg value;
      if (!poly_eval_launch_expr(runner->grid_exprs[dim], bindings, n_bindings, &value)) return -1;
      int64_t resolved = poly_launch_arg_to_i64(value);
      runner->grid[dim] = (resolved > 0 && resolved <= INT32_MAX) ? (int)resolved : 1;
    }
    if (runner->block_exprs[dim]) {
      PolyArg value;
      if (!poly_eval_launch_expr(runner->block_exprs[dim], bindings, n_bindings, &value)) return -1;
      int64_t resolved = poly_launch_arg_to_i64(value);
      runner->block[dim] = (resolved > 0 && resolved <= INT32_MAX) ? (int)resolved : 1;
    }
  }

  return 0;
}

static void poly_runner_apply_program_launch_info(PolyCtx *ctx, PolyUOp *program, PolyRunner *runner) {
  if (!ctx || !program || !runner || program->op != POLY_OP_PROGRAM) return;
  const PolyProgramInfo *info = poly_program_info(ctx, program);
  if (!info) return;

  bool has_launch_expr = false;
  for (int dim = 0; dim < 3; dim++) {
    if (info->global_exprs[dim] || info->local_exprs[dim] ||
        info->global_size[dim] != 1 || info->local_size[dim] != 1 || !info->has_local_size) {
      has_launch_expr = true;
      break;
    }
  }
  if (!has_launch_expr) return;

  for (int dim = 0; dim < 3; dim++) {
    int global = info->global_size[dim];
    runner->grid[dim] = global > 0 ? global : 1;
    runner->grid_exprs[dim] = info->global_exprs[dim];

    if (info->has_local_size && info->local_exprs[dim]) {
      int local = info->local_size[dim];
      runner->block[dim] = local > 0 ? local : 1;
      runner->block_exprs[dim] = info->local_exprs[dim];
    } else if (info->has_local_size) {
      int local = info->local_size[dim];
      runner->block[dim] = local > 0 ? local : 1;
      runner->block_exprs[dim] = NULL;
    } else if (!info->has_local_size) {
      runner->block[dim] = 1;
      runner->block_exprs[dim] = NULL;
    }
  }
}

static void debug_dump_webgpu_runner_args(
    const PolyCompiledSchedule *plan,
    int exec_step,
    int kernel_idx,
    const PolyRunner *runner,
    void **args
) {
#ifdef __EMSCRIPTEN__
  if (!plan || !plan->template || !runner || !args) return;
  if (plan->device != POLY_DEVICE_WEBGPU || !poly_debug_at_least(7)) return;

  PolySchedule sched_view = {.template = plan->template, .run = plan->run};
  const PolySchedule *sched = &sched_view;
  fprintf(
      stderr, "[webgpu-run] step=%d kernel=%d n_params=%d n_vars=%d\n", exec_step, kernel_idx,
      runner->n_params, runner->n_vars
  );
  for (int i = 0; i < runner->n_params; i++) {
    int slot = runner->param_to_slot ? runner->param_to_slot[i] : -1;
    const PolyScheduleBufSlot *bs =
        (slot >= 0 && slot < sched->template->n_buf_slots) ? &sched->template->buf_slots[slot] : NULL;
    fprintf(
        stderr,
        "  [param %d] slot=%d handle=%p buf_uop=%p interm=%d ext=%d nbytes=%lld numel=%lld\n", i,
        slot, args[i], bs ? (void *)bs->buf_uop : NULL, bs ? (int)bs->is_intermediate : -1,
        bs ? bs->external_buf_idx : -1, bs ? (long long)bs->nbytes : -1LL,
        bs ? (long long)bs->numel : -1LL
    );
  }
  for (int i = 0; i < runner->n_vars; i++) {
    void *var_ptr = args[runner->n_params + i];
    fprintf(stderr, "  [var %d] ptr=%p val=%d\n", i, var_ptr, var_ptr ? *(int *)var_ptr : 0);
  }
#else
  (void)plan;
  (void)exec_step;
  (void)kernel_idx;
  (void)runner;
  (void)args;
#endif
}

static void debug_dump_schedule_args(
    const PolySchedule *sched,
    PolyDevice device,
    int exec_step,
    int kernel_idx,
    const PolyRunner *runner,
    void **args
) {
  if (!sched || !runner || !args) return;
  if (!poly_debug_at_least(7)) return;

  fprintf(
      stderr, "[polygrad:args] device=%s step=%d kernel=%d n_params=%d n_vars=%d\n",
      poly_device_name(device), exec_step, kernel_idx, runner->n_params, runner->n_vars
  );
  for (int i = 0; i < runner->n_params; i++) {
    int slot = runner->param_to_slot ? runner->param_to_slot[i] : -1;
    const PolyScheduleBufSlot *bs =
        (slot >= 0 && slot < sched->template->n_buf_slots) ? &sched->template->buf_slots[slot] : NULL;
    fprintf(
        stderr,
        "  [param %d] slot=%d handle=%p buf_uop=%p interm=%d ext=%d nbytes=%lld numel=%lld\n", i,
        slot, args[i], bs ? (void *)bs->buf_uop : NULL, bs ? (int)bs->is_intermediate : -1,
        bs ? bs->external_buf_idx : -1, bs ? (long long)bs->nbytes : -1LL,
        bs ? (long long)bs->numel : -1LL
    );
  }
}

static bool is_intermediate_buffer_uop(PolyUOp *buf) {
  return buf && buf->op == POLY_OP_BUFFER && buf->n_src >= 1 && buf->src[0] &&
         buf->src[0]->op == POLY_OP_LUNIQUE;
}

static PolyDevice poly_schedule_slot_current_device(
    PolyCtx *ctx,
    const PolySchedule *sched,
    int slot_idx
) {
  if (!sched || slot_idx < 0 || slot_idx >= sched->template->n_buf_slots) return POLY_DEVICE_AUTO;
  PolyDevice dev = sched->template->buf_slots[slot_idx].device;
  if (dev != POLY_DEVICE_AUTO) return dev;
  if (ctx && !sched->template->buf_slots[slot_idx].is_intermediate) {
    PolyBuffer *b = poly_buffer_get(ctx, sched->template->buf_slots[slot_idx].buf_uop);
    if (b && b->device != POLY_DEVICE_AUTO) return b->device;
  }
  return POLY_DEVICE_AUTO;
}

static bool poly_schedule_slot_used_as_copy_source(const PolySchedule *sched, int slot_idx) {
  if (!sched) return false;
  for (int k = 0; k < sched->template->n_calls; k++) {
    PolyUOp *call = poly_schedule_call(sched, k);
    const PolyCallIO *io = poly_schedule_call_io(sched, k);
    if (!poly_call_is_copy(call) || !io || io->n_args < 2) continue;
    if (io->arg_to_slot[1] == slot_idx) return true;
  }
  return false;
}

static bool poly_schedule_slot_used_by_compute(const PolySchedule *sched, int slot_idx) {
  if (!sched) return false;
  for (int k = 0; k < sched->template->n_calls; k++) {
    PolyUOp *call = poly_schedule_call(sched, k);
    const PolyCallIO *io = poly_schedule_call_io(sched, k);
    if (poly_call_is_copy(call) || poly_call_is_view(call)) continue;
    const PolyCallAccess *access = io ? io->access : NULL;
    if (!io || !access) continue;
    for (int ai = 0; ai < access->n_active_args; ai++) {
      int arg = access->active_args[ai];
      if (io->arg_to_slot[arg] == slot_idx) return true;
    }
  }
  return false;
}

static PolyDevice poly_schedule_slot_target_device(
    PolyCtx *ctx,
    const PolySchedule *sched,
    int slot_idx,
    PolyDevice fallback
) {
  PolyDevice dev = poly_schedule_slot_current_device(ctx, sched, slot_idx);
  bool copy_source = poly_schedule_slot_used_as_copy_source(sched, slot_idx);
  bool compute_use = poly_schedule_slot_used_by_compute(sched, slot_idx);

  if (dev == POLY_DEVICE_AUTO) dev = fallback;
  if (dev == POLY_DEVICE_AUTO) dev = poly_device_default();

  /* HOST is valid as an explicit COPY source. It is not a kernel execution
   * domain, so a compute use must be migrated to the fallback executable
   * device unless that backend shares host-addressable storage. */
  if (dev == POLY_DEVICE_HOST && (!copy_source || compute_use)) {
    PolyDevice run = (fallback == POLY_DEVICE_AUTO) ? poly_device_default() : fallback;
    if (poly_device_can_execute(run) && !poly_device_is_host_addressable(run)) return run;
  }
  return dev;
}

static PolyDevice poly_schedule_slot_declared_execution_device(
    const PolySchedule *sched,
    int slot_idx,
    PolyDevice fallback
) {
  if (!sched || slot_idx < 0 || slot_idx >= sched->template->n_buf_slots) return POLY_DEVICE_AUTO;
  PolyDevice dev = sched->template->buf_slots[slot_idx].device;
  if (dev != POLY_DEVICE_AUTO && dev != POLY_DEVICE_HOST &&
      fallback != POLY_DEVICE_AUTO && fallback != POLY_DEVICE_HOST &&
      dev != fallback && poly_device_can_execute(fallback) &&
      poly_device_is_host_addressable(dev) && poly_device_is_host_addressable(fallback))
    return POLY_DEVICE_AUTO;
  if (dev != POLY_DEVICE_AUTO && dev != POLY_DEVICE_HOST &&
      poly_device_is_host_addressable(dev) && fallback != POLY_DEVICE_AUTO &&
      fallback != POLY_DEVICE_HOST && poly_device_can_execute(fallback) &&
      !poly_device_is_host_addressable(fallback))
    return POLY_DEVICE_AUTO;
  if (dev != POLY_DEVICE_AUTO && dev != POLY_DEVICE_HOST && poly_device_can_execute(dev))
    return dev;
  return POLY_DEVICE_AUTO;
}

static PolyDevice poly_call_device(
    PolyCtx *ctx,
    const PolySchedule *sched,
    int call_index,
    PolyDevice fallback
) {
  PolyUOp *call = poly_schedule_call(sched, call_index);
  if (!sched || !call) return fallback == POLY_DEVICE_AUTO ? poly_device_default() : fallback;
  const PolyCallIO *io = poly_schedule_call_io(sched, call_index);
  int n_params = io ? io->n_args : 0;
  if (n_params <= 0) return fallback == POLY_DEVICE_AUTO ? poly_device_default() : fallback;

  if (poly_call_is_copy(call) || poly_call_is_view(call)) {
    const PolyCallAccess *access = io->access;
    for (int wi = 0; wi < access->n_write_args; wi++) {
      int arg = access->write_args[wi];
      int slot = io->arg_to_slot[arg];
      PolyDevice dev = poly_schedule_slot_target_device(ctx, sched, slot, fallback);
      if (dev != POLY_DEVICE_AUTO && dev != POLY_DEVICE_HOST && poly_device_can_execute(dev)) {
        return dev;
      }
    }
  }

  /* For compute calls, residency is not placement. A CPU host root retained in
   * ctx->buffers must not pull a CUDA-preferred instance train step back to CPU.
   * Explicit slot/device annotations still win; otherwise the selected schedule
   * device is the call device, matching tinygrad's CALL execution boundary. */
  if (!poly_call_is_copy(call) && !poly_call_is_view(call)) {
    const PolyCallAccess *access = io->access;
    for (int ai = 0; ai < access->n_active_args; ai++) {
      int arg = access->active_args[ai];
      int slot = io->arg_to_slot[arg];
      PolyDevice dev = poly_schedule_slot_declared_execution_device(sched, slot, fallback);
      if (dev != POLY_DEVICE_AUTO) {
        return dev;
      }
    }
    if (fallback != POLY_DEVICE_AUTO && fallback != POLY_DEVICE_HOST &&
        poly_device_can_execute(fallback)) {
      return fallback;
    }
  }

  const PolyCallAccess *access = io->access;
  for (int ri = 0; ri < access->n_read_args; ri++) {
    int arg = access->read_args[ri];
    int slot = io->arg_to_slot[arg];
    PolyDevice dev = poly_schedule_slot_target_device(ctx, sched, slot, fallback);
    if (dev != POLY_DEVICE_AUTO && dev != POLY_DEVICE_HOST && poly_device_can_execute(dev)) {
      return dev;
    }
  }
  return fallback == POLY_DEVICE_AUTO ? poly_device_default() : fallback;
}

static PolyDevice poly_intermediate_slot_runtime_device(
    PolyCtx *ctx,
    PolySchedule *sched,
    int slot,
    PolyDevice fallback
) {
  if (!ctx || !sched || slot < 0 || slot >= sched->template->n_buf_slots)
    return fallback == POLY_DEVICE_AUTO ? poly_device_default() : fallback;

  PolyDevice copy_device = POLY_DEVICE_AUTO;
  for (int k = 0; k < sched->template->n_calls; k++) {
    const PolyCallIO *io = poly_schedule_call_io(sched, k);
    const PolyCallAccess *access = io ? io->access : NULL;
    if (!io || !access || io->n_args <= 0) continue;

    bool touches = false;
    for (int ai = 0; ai < access->n_active_args; ai++) {
      int arg = access->active_args[ai];
      if (io->arg_to_slot[arg] == slot) {
        touches = true;
        break;
      }
    }
    if (!touches) continue;

    PolyDevice device = poly_call_device(ctx, sched, k, fallback);
    if (poly_call_is_copy(poly_schedule_call(sched, k)) ||
        poly_call_is_view(poly_schedule_call(sched, k))) {
      if (copy_device == POLY_DEVICE_AUTO && device != POLY_DEVICE_AUTO)
        copy_device = device;
      continue;
    }
    if (device != POLY_DEVICE_AUTO && device != POLY_DEVICE_HOST && poly_device_can_execute(device))
      return device;
  }
  if (copy_device != POLY_DEVICE_AUTO) return copy_device;

  PolyDevice device = poly_schedule_slot_target_device(ctx, sched, slot, fallback);
  if (device == POLY_DEVICE_AUTO) device = fallback;
  if (device == POLY_DEVICE_AUTO) device = poly_device_default();
  return device;
}

static PolyDevice poly_memory_arena_slot_runtime_device(
    PolyCtx *ctx,
    PolySchedule *sched,
    int arena_slot,
    PolyDevice fallback
) {
  if (!ctx || !sched || arena_slot < 0 || arena_slot >= sched->template->n_buf_slots)
    return fallback == POLY_DEVICE_AUTO ? poly_device_default() : fallback;
  for (int i = 0; i < sched->template->n_buf_slots; i++) {
    const PolyScheduleBufSlot *slot = &sched->template->buf_slots[i];
    if (!slot->is_intermediate || !slot->has_memory_parent ||
        slot->memory_parent_slot != arena_slot)
      continue;
    PolyDevice device = poly_intermediate_slot_runtime_device(ctx, sched, i, fallback);
    if (device != POLY_DEVICE_AUTO) return device;
  }
  return fallback == POLY_DEVICE_AUTO ? poly_device_default() : fallback;
}

static bool collect_external_buf_order_from_kernel_graph(
    PolyUOp *kernel_graph,
    PolyUOp ***out_buf_order,
    int *out_n_external,
    PolyUOp **stack_buf_order,
    bool *out_owned
) {
  if (!kernel_graph || !out_buf_order || !out_n_external || !out_owned) return false;
  *out_buf_order = stack_buf_order;
  *out_n_external = 0;
  *out_owned = false;

  PolyUOp *all_stack[POLY_MAX_REALIZE_BUFS];
  int n_all = 0, n_visited = 0;
  poly_collect_buf_order(kernel_graph, all_stack, &n_all, NULL, &n_visited);
  if (n_all <= POLY_MAX_REALIZE_BUFS) {
    int n_external = 0;
    for (int i = 0; i < n_all; i++) {
      if (!is_intermediate_buffer_uop(all_stack[i])) stack_buf_order[n_external++] = all_stack[i];
    }
    *out_n_external = n_external;
    return true;
  }

  PolyUOp **all_bufs = NULL;
  if (!poly_collect_buf_order_alloc(kernel_graph, &all_bufs, &n_all, &n_visited)) return false;
  int n_external = 0;
  for (int i = 0; i < n_all; i++) {
    if (!is_intermediate_buffer_uop(all_bufs[i])) all_bufs[n_external++] = all_bufs[i];
  }
  *out_buf_order = all_bufs;
  *out_n_external = n_external;
  *out_owned = true;
  return true;
}

static bool uop_ptr_in_list(PolyUOp *u, PolyUOp **list, int n) {
  for (int i = 0; i < n; i++)
    if (list[i] == u) return true;
  return false;
}

static int collect_define_vars(PolyCtx *ctx, PolyUOp *root, PolyUOp **out, int cap) {
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, root, &n_topo);
  if (!topo) return 0;

  int n = 0;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (!u || u->op != POLY_OP_DEFINE_VAR || uop_ptr_in_list(u, out, n)) continue;
    if (n >= cap) {
      fprintf(stderr, "polygrad: too many DEFINE_VARs in schedule (cap=%d)\n", cap);
      poly_toposort_free(topo);
      return -1;
    }
    out[n++] = u;
  }
  poly_toposort_free(topo);
  return n;
}

static int find_var_binding(const PolyVarBinding *bindings, int n, PolyUOp *var) {
  for (int i = 0; i < n; i++)
    if (bindings[i].var == var) return i;
  return -1;
}

static int collect_bind_defaults(
    PolyCtx *ctx,
    PolyUOp *root,
    PolyUOp **used_vars,
    int n_used_vars,
    PolyVarBinding *out,
    int cap
) {
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, root, &n_topo);
  if (!topo) return 0;

  int n = 0;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (!u || u->op != POLY_OP_BIND || u->n_src < 2) continue;
    PolyUOp *var = u->src[0];
    PolyUOp *val = u->src[1];
    if (!var || var->op != POLY_OP_DEFINE_VAR || !val || val->op != POLY_OP_CONST) continue;
    if (!uop_ptr_in_list(var, used_vars, n_used_vars)) continue;
    if (val->arg.kind != POLY_ARG_INT) {
      fprintf(stderr, "polygrad: BIND default for DEFINE_VAR must be integer CONST\n");
      poly_toposort_free(topo);
      return -1;
    }

    int32_t value = (int32_t)val->arg.i;
    int existing = find_var_binding(out, n, var);
    if (existing >= 0) {
      if (out[existing].value != value) {
        fprintf(
            stderr, "polygrad: BIND mismatch for DEFINE_VAR: %d != %d\n", out[existing].value, value
        );
        poly_toposort_free(topo);
        return -1;
      }
      continue;
    }
    if (n >= cap) {
      fprintf(stderr, "polygrad: too many BIND defaults (cap=%d)\n", cap);
      poly_toposort_free(topo);
      return -1;
    }
    out[n++] = (PolyVarBinding){.var = var, .value = value};
  }
  poly_toposort_free(topo);
  return n;
}

static int schedule_slot_for_buf(
    PolyUOp *buf,
    PolyUOp **buf_order_slots,
    int n_buf_order_slots,
    PolyMap *inter_set
) {
  if (!buf || buf->op != POLY_OP_BUFFER) return -1;
  int pos = poly_find_buf_position(buf, buf_order_slots, n_buf_order_slots);
  if (pos >= 0) return pos;
  if (inter_set) {
    PolyUOp *v = poly_map_get(inter_set, poly_ptr_hash(buf), buf, poly_ptr_eq);
    if (v) return (int)((intptr_t)v - 1);
  }
  return -1;
}

static int schedule_slot_for_kernel_param(
    const PolyKernelScheduleResult *sr,
    int kernel_idx,
    int param_idx,
    PolyUOp **buf_order_slots,
    int n_buf_order_slots,
    PolyMap *inter_set
) {
  if (!sr || kernel_idx < 0 || kernel_idx >= sr->n_kernels) return -1;
  if (!sr->param_to_buf || !sr->kernel_n_params || !sr->param_to_buf[kernel_idx]) return -1;
  if (param_idx < 0 || param_idx >= sr->kernel_n_params[kernel_idx]) return -1;
  return schedule_slot_for_buf(
      sr->param_to_buf[kernel_idx][param_idx], buf_order_slots, n_buf_order_slots, inter_set
  );
}

static PolyUOp *linear_call_param_for_buf(PolyCtx *ctx, PolyUOp *buf, int external_slot) {
  if (external_slot >= 0)
    return poly_uop0(ctx, POLY_OP_PARAM, POLY_VOID, poly_arg_int(external_slot));
  return buf;
}

static PolyUOp *make_call_copy_body(
    PolyCtx *ctx,
    const PolySchedule *sched,
    int dst_slot,
    int src_slot
) {
  if (!ctx || !sched || dst_slot < 0 || dst_slot >= sched->template->n_buf_slots || src_slot < 0 ||
      src_slot >= sched->template->n_buf_slots)
    return NULL;
  PolyDType dtype = sched->template->buf_slots[src_slot].dtype;
  PolyUOp *src_param = poly_uop0(ctx, POLY_OP_PARAM, dtype, poly_arg_int(1));
  PolyUOp *dev = poly_uop0(
      ctx, POLY_OP_DEVICE, POLY_VOID, poly_arg_int((int64_t)sched->template->buf_slots[dst_slot].device)
  );
  PolyUOp *copy_src[2] = {src_param, dev};
  return poly_uop(ctx, POLY_OP_COPY, dtype, copy_src, 2, poly_arg_none());
}

typedef struct {
  PolyUOp *old_buf;
  PolyUOp *new_buf;
  bool needs_zero;
} PolyLinearReplayIntermediate;

static bool linear_replay_arg_is_external_param(PolyUOp *arg, int *slot_out) {
  if (!arg || arg->op != POLY_OP_PARAM || arg->arg.kind != POLY_ARG_INT) return false;
  if (slot_out) *slot_out = (int)arg->arg.i;
  return true;
}

static PolyUOp *linear_replay_new_intermediate(PolyCtx *ctx, PolyUOp *old_buf) {
  if (!ctx || !old_buf || old_buf->op != POLY_OP_BUFFER) return NULL;
  int64_t id = poly_ctx_next_unique_id(ctx);
  PolyUOp *lunique = poly_uop0(ctx, POLY_OP_LUNIQUE, POLY_VOID, poly_arg_int(id));
  if (!lunique) return NULL;
  PolyDevice dev_id = poly_uop_device(old_buf);
  PolyArg dev_arg = dev_id == POLY_DEVICE_AUTO ? poly_arg_none() : poly_arg_int((int64_t)dev_id);
  PolyUOp *dev = poly_uop0(ctx, POLY_OP_DEVICE, POLY_VOID, dev_arg);
  if (!dev) return NULL;
  PolyUOp *src[2] = {lunique, dev};
  return poly_uop(ctx, POLY_OP_BUFFER, old_buf->dtype, src, 2, old_buf->arg);
}

static int linear_replay_intermediate_index(
    PolyLinearReplayIntermediate *items,
    int n_items,
    PolyUOp *old_buf
) {
  for (int i = 0; i < n_items; i++)
    if (items[i].old_buf == old_buf) return i;
  return -1;
}

static int linear_replay_add_intermediate(
    PolyCtx *ctx,
    const PolyScheduleCacheEntry *cache_entry,
    PolyLinearReplayIntermediate **items,
    int *n_items,
    int *cap_items,
    PolyUOp *old_buf
) {
  if (!ctx || !items || !n_items || !cap_items || !old_buf || old_buf->op != POLY_OP_BUFFER)
    return -1;
  int idx = linear_replay_intermediate_index(*items, *n_items, old_buf);
  if (idx >= 0) return idx;
  if (!cache_entry) return -1;
  int entry_idx = -1;
  for (int i = 0; i < cache_entry->n_intermediates; i++) {
    if (cache_entry->intermediates[i].template_buf == old_buf) {
      entry_idx = i;
      break;
    }
  }
  if (entry_idx < 0) return -1;
  if (*n_items >= *cap_items) {
    int new_cap = *cap_items ? *cap_items * 2 : 8;
    PolyLinearReplayIntermediate *tmp =
        realloc(*items, (size_t)new_cap * sizeof(PolyLinearReplayIntermediate));
    if (!tmp) return -1;
    *items = tmp;
    *cap_items = new_cap;
  }
  PolyUOp *fresh = linear_replay_new_intermediate(ctx, old_buf);
  if (!fresh) return -1;
  idx = (*n_items)++;
  (*items)[idx].old_buf = old_buf;
  (*items)[idx].new_buf = fresh;
  (*items)[idx].needs_zero = cache_entry->intermediates[entry_idx].needs_zero;
  return idx;
}

static int linear_replay_collect_intermediates(
    PolyCtx *ctx,
    PolyUOp *linear_template,
    const PolyScheduleCacheEntry *cache_entry,
    PolyLinearReplayIntermediate **items,
    int *n_items,
    int *cap_items,
    int n_external
) {
  if (!ctx || !linear_template || linear_template->op != POLY_OP_LINEAR || !items || !n_items ||
      !cap_items)
    return -1;
  for (int k = 0; k < linear_template->n_src; k++) {
    PolyUOp *call = linear_template->src[k];
    if (!call || call->op != POLY_OP_CALL || call->n_src < 1) return -1;
    if (poly_call_is_view(call)) {
      if (poly_debug_at_least(7))
        fprintf(stderr, "[polygrad:linear-replay] reject call=%d kind=%s\n", k, poly_call_kind_name(call));
      return -1;
    }
    for (int i = 1; i < call->n_src; i++) {
      PolyUOp *arg = call->src[i];
      int external_slot = -1;
      if (poly_call_arg_is_var(arg)) continue;
      if (linear_replay_arg_is_external_param(arg, &external_slot)) {
        if (external_slot < 0 || external_slot >= n_external) return -1;
        continue;
      }
      if (!arg || arg->op != POLY_OP_BUFFER) {
        if (poly_debug_at_least(7))
          fprintf(
              stderr, "[polygrad:linear-replay] reject call=%d arg=%d op=%s\n", k, i,
              arg ? poly_op_name(arg->op) : "NULL"
          );
        return -1;
      }
      if (linear_replay_add_intermediate(ctx, cache_entry, items, n_items, cap_items, arg) < 0)
        return -1;
    }
  }
  return 0;
}

static PolyUOp *linear_replay_resolve_arg(
    PolyLinearReplayIntermediate *items,
    int n_items,
    PolyUOp **external_bufs,
    int n_external,
    PolyUOp *arg
) {
  int external_slot = -1;
  if (poly_call_arg_is_var(arg)) return arg;
  if (linear_replay_arg_is_external_param(arg, &external_slot))
    return (external_slot >= 0 && external_slot < n_external) ? external_bufs[external_slot] : NULL;
  int idx = linear_replay_intermediate_index(items, n_items, arg);
  return (idx >= 0) ? items[idx].new_buf : NULL;
}

static int poly_schedule_init_call_io_and_runtime(PolyCtx *ctx, PolySchedule *ps) {
  if (!ctx || !ps || !ps->template || !ps->run || !ps->template->linear) return -1;
  ps->template->n_calls = ps->template->linear->n_src;
  if (!ps->template->call_access)
    ps->template->call_access = calloc((size_t)ps->template->n_calls, sizeof(PolyCallAccess));
  ps->run->call_io = calloc((size_t)ps->template->n_calls, sizeof(PolyCallIO));
  ps->run->calls = calloc((size_t)ps->template->n_calls, sizeof(PolyCallRuntime));
  if (ps->template->n_calls > 0 && (!ps->template->call_access || !ps->run->call_io)) return -1;
  if (ps->template->n_calls > 0 && !ps->run->calls) return -1;
  for (int k = 0; k < ps->template->n_calls; k++) {
    ps->run->calls[k].call = ps->template->linear->src[k];
    if (poly_call_io_init(ctx, ps, k) != 0) return -1;
  }
  return 0;
}

static PolySchedule *build_schedule_from_linear_template(
    PolyCtx *ctx,
    PolyScheduleCacheEntry *cache_entry,
    PolyCompileMode mode,
    uint32_t graph_hash,
    PolyUOp **buf_order_orig,
    int n_bufs_orig,
    const PolyVarBinding *default_vars,
    int n_default_vars,
    bool apply_memory_plan
) {
  if (!ctx || !cache_entry || !cache_entry->linear || cache_entry->linear->op != POLY_OP_LINEAR ||
      n_bufs_orig < 0)
    return NULL;
  PolyUOp *linear_template = cache_entry->linear;

  PolyLinearReplayIntermediate *intermediates = NULL;
  int n_intermediates = 0, cap_intermediates = 0;
  if (linear_replay_collect_intermediates(
          ctx, linear_template, cache_entry, &intermediates, &n_intermediates, &cap_intermediates,
          n_bufs_orig
      ) != 0) {
    free(intermediates);
    return NULL;
  }

  PolySchedule *ps = calloc(1, sizeof(PolySchedule));
  if (!ps) {
    free(intermediates);
    return NULL;
  }
  ps->template = calloc(1, sizeof(PolyScheduleTemplate));
  ps->run = calloc(1, sizeof(PolyScheduleRuntime));
  if (!ps->template || !ps->run) {
    free(intermediates);
    poly_schedule_free(ps);
    return NULL;
  }

  ps->template->refcount = 1;
  ps->template->cache_entry = poly_schedule_cache_entry_retain(cache_entry);
  ps->template->mode = mode;
  ps->template->graph_hash = graph_hash;
  ps->template->loss_buf_slot = -1;
  ps->template->n_default_vars = n_default_vars;
  if (n_default_vars > 0) {
    ps->template->default_vars = malloc((size_t)n_default_vars * sizeof(PolyVarBinding));
    if (!ps->template->default_vars) goto cleanup;
    memcpy(ps->template->default_vars, default_vars, (size_t)n_default_vars * sizeof(PolyVarBinding));
  }

  ps->template->n_buf_slots = n_bufs_orig + n_intermediates;
  if (ps->template->n_buf_slots > 0) {
    ps->template->buf_slots = calloc((size_t)ps->template->n_buf_slots, sizeof(PolyScheduleBufSlot));
    if (!ps->template->buf_slots) goto cleanup;
  }

  for (int i = 0; i < n_bufs_orig; i++) {
    PolyScheduleBufSlot *slot = &ps->template->buf_slots[i];
    PolyUOp *buf = buf_order_orig[i];
    slot->buf_uop = buf;
    slot->is_intermediate = false;
    slot->external_buf_idx = i;
    slot->dtype = buf ? poly_dtype_scalar(buf->dtype) : POLY_FLOAT32;
    slot->numel = (buf && buf->arg.kind == POLY_ARG_INT) ? buf->arg.i : 0;
    slot->device = poly_uop_device(buf);
    if (slot->numel > 0) slot->nbytes = slot->numel * poly_dtype_itemsize(slot->dtype);
  }

  for (int i = 0; i < n_intermediates; i++) {
    PolyScheduleBufSlot *slot = &ps->template->buf_slots[n_bufs_orig + i];
    PolyUOp *buf = intermediates[i].new_buf;
    slot->buf_uop = buf;
    slot->is_intermediate = true;
    slot->external_buf_idx = -1;
    slot->dtype = buf ? poly_dtype_scalar(buf->dtype) : POLY_FLOAT32;
    slot->numel = (buf && buf->arg.kind == POLY_ARG_INT) ? buf->arg.i : 0;
    slot->device = poly_uop_device(buf);
    slot->nbytes = slot->numel * poly_dtype_itemsize(slot->dtype);
    slot->needs_zero = intermediates[i].needs_zero;
  }

  PolyUOp **linear_src = calloc((size_t)linear_template->n_src, sizeof(PolyUOp *));
  if (linear_template->n_src > 0 && !linear_src) goto cleanup;
  for (int k = 0; k < linear_template->n_src; k++) {
    PolyUOp *call = linear_template->src[k];
    if (!call || call->op != POLY_OP_CALL || call->n_src < 1) {
      free(linear_src);
      goto cleanup;
    }
    PolyUOp **call_src = calloc((size_t)call->n_src, sizeof(PolyUOp *));
    if (!call_src) {
      free(linear_src);
      goto cleanup;
    }
    call_src[0] = call->src[0];
    bool ok = true;
    for (int i = 1; i < call->n_src; i++) {
      call_src[i] = linear_replay_resolve_arg(
          intermediates, n_intermediates, buf_order_orig, n_bufs_orig, call->src[i]
      );
      if (!call_src[i]) {
        ok = false;
        break;
      }
    }
    if (ok) linear_src[k] = poly_uop(ctx, POLY_OP_CALL, POLY_VOID, call_src, call->n_src, call->arg);
    free(call_src);
    if (!ok || !linear_src[k]) {
      free(linear_src);
      goto cleanup;
    }
  }
  ps->template->linear =
      poly_uop(ctx, POLY_OP_LINEAR, POLY_VOID, linear_src, linear_template->n_src, linear_template->arg);
  free(linear_src);
  if (!ps->template->linear) goto cleanup;
  ps->template->n_calls = ps->template->linear->n_src;
  if (cache_entry->call_access) {
    if (cache_entry->n_calls != ps->template->n_calls) goto cleanup;
    ps->template->call_access =
        poly_call_access_clone_array(cache_entry->call_access, cache_entry->n_calls);
    if (!ps->template->call_access) goto cleanup;
  }

  if (apply_memory_plan && poly_schedule_memory_plan(ctx, ps) != 0) goto cleanup;
  if (poly_schedule_init_call_io_and_runtime(ctx, ps) != 0) goto cleanup;

  free(intermediates);
  return ps;

cleanup:
  free(intermediates);
  poly_schedule_free(ps);
  return NULL;
}

static PolySchedule *build_schedule_from_kernel_graph(
    PolyCtx *ctx,
    PolyUOp *kernel_graph,
    PolyCompileMode mode,
    uint32_t graph_hash,
    PolyUOp **buf_order_orig,
    int n_bufs_orig,
    const PolyVarBinding *default_vars,
    int n_default_vars,
    PolyUOp *linear_template,
    bool apply_memory_plan
) {
  if (!kernel_graph || kernel_graph->op != POLY_OP_SINK) {
    fprintf(stderr, "polygrad: build_schedule_from_kernel_graph: expected SINK\n");
    return NULL;
  }

  PolyKernelScheduleResult sr = poly_build_kernel_schedule_from_kernel_graph(ctx, kernel_graph);
  if (sr.n_kernels < 1) {
    poly_kernel_schedule_result_free(&sr);
    return NULL;
  }

  PolySchedule *ps = calloc(1, sizeof(PolySchedule));
  if (!ps) {
    poly_kernel_schedule_result_free(&sr);
    return NULL;
  }
  ps->template = calloc(1, sizeof(PolyScheduleTemplate));
  ps->run = calloc(1, sizeof(PolyScheduleRuntime));
  if (!ps->template || !ps->run) {
    poly_schedule_free(ps);
    poly_kernel_schedule_result_free(&sr);
    return NULL;
  }
  ps->template->refcount = 1;
  ps->template->mode = mode;
  ps->template->graph_hash = graph_hash;
  ps->template->loss_buf_slot = -1;

  ps->template->n_default_vars = n_default_vars;
  if (n_default_vars > 0) {
    ps->template->default_vars = malloc((size_t)n_default_vars * sizeof(PolyVarBinding));
    memcpy(ps->template->default_vars, default_vars, (size_t)n_default_vars * sizeof(PolyVarBinding));
  }

  int n_external = n_bufs_orig;
  int n_intermediate = sr.n_intermediates;
  ps->template->n_buf_slots = n_external + n_intermediate;

  if (ps->template->n_buf_slots > 0) {
    ps->template->buf_slots = calloc((size_t)ps->template->n_buf_slots, sizeof(PolyScheduleBufSlot));

    for (int i = 0; i < n_external; i++) {
      PolyScheduleBufSlot *slot = &ps->template->buf_slots[i];
      PolyUOp *buf = buf_order_orig[i];
      slot->buf_uop = buf;
      slot->is_intermediate = false;
      slot->external_buf_idx = i;
      slot->dtype = buf ? poly_dtype_scalar(buf->dtype) : POLY_FLOAT32;
      slot->numel = (buf && buf->arg.kind == POLY_ARG_INT) ? buf->arg.i : 0;
      slot->device = poly_uop_device(buf);
      if (slot->numel > 0) slot->nbytes = slot->numel * poly_dtype_itemsize(slot->dtype);
    }

    for (int b = 0; b < n_intermediate; b++) {
      PolyScheduleBufSlot *slot = &ps->template->buf_slots[n_external + b];
      slot->is_intermediate = true;
      slot->external_buf_idx = -1;
      if (sr.intermediate_buf_uops && sr.intermediate_buf_uops[b]) {
        slot->buf_uop = sr.intermediate_buf_uops[b];
        slot->dtype = poly_dtype_scalar(sr.intermediate_buf_uops[b]->dtype);
        slot->device = poly_uop_device(sr.intermediate_buf_uops[b]);
      } else {
        slot->dtype = POLY_FLOAT32;
        slot->device = POLY_DEVICE_AUTO;
      }
      slot->numel = sr.intermediate_sizes ? sr.intermediate_sizes[b] : 0;
      int itemsize = (sr.intermediate_itemsizes && sr.intermediate_itemsizes[b] > 0)
                         ? sr.intermediate_itemsizes[b]
                         : (int)sizeof(float);
      slot->nbytes = slot->numel * itemsize;
      slot->needs_zero = sr.intermediate_needs_zero ? sr.intermediate_needs_zero[b] : true;
    }
  }

  PolyMap *inter_set = NULL;
  if (n_intermediate > 0 && sr.intermediate_buf_uops) {
    inter_set = poly_map_new((size_t)(n_intermediate < 4 ? 4 : n_intermediate));
    for (int b = 0; b < n_intermediate; b++) {
      PolyUOp *ib = sr.intermediate_buf_uops[b];
      poly_map_set(
          inter_set, poly_ptr_hash(ib), ib, (PolyUOp *)(intptr_t)(n_external + b + 1), poly_ptr_eq
      );
    }
  }

  PolyUOp **linear_src = calloc((size_t)sr.n_kernels, sizeof(PolyUOp *));
  if (sr.n_kernels > 0 && !linear_src) goto cleanup;

  for (int step = 0; step < sr.n_kernels; step++) {
    int k = sr.exec_order ? sr.exec_order[step] : step;
    if (k < 0 || k >= sr.n_kernels) goto cleanup_linear;

    PolyUOp *cached_root = NULL;
    if (linear_template && linear_template->op == POLY_OP_LINEAR &&
        linear_template->n_src == sr.n_kernels) {
      cached_root = linear_template->src[step];
      if (cached_root && cached_root->op == POLY_OP_CALL && cached_root->n_src >= 1)
        cached_root = cached_root->src[0];
    }

    PolyUOp *body = cached_root ? cached_root : sr.kernels[k];
    bool is_copy = sr.kernel_kinds && sr.kernel_kinds[k] == POLY_KERNEL_ITEM_COPY;

    int n_params = is_copy ? 2 : sr.kernel_n_params[k];
    int n_vars = (sr.kernel_n_vars ? sr.kernel_n_vars[k] : 0);
    int n_call_src = 1 + n_params + n_vars;
    PolyUOp **call_src = calloc((size_t)n_call_src, sizeof(PolyUOp *));
    if (!call_src) goto cleanup_linear;

    if (is_copy) {
      int dst_slot = schedule_slot_for_kernel_param(
          &sr, k, sr.copy_dst_params ? sr.copy_dst_params[k] : -1, buf_order_orig, n_bufs_orig,
          inter_set
      );
      int src_slot = schedule_slot_for_kernel_param(
          &sr, k, sr.copy_src_params ? sr.copy_src_params[k] : -1, buf_order_orig, n_bufs_orig,
          inter_set
      );

      if (dst_slot < 0 || src_slot < 0) {
        fprintf(
            stderr,
            "polygrad: build_schedule_from_kernel_graph: unresolved COPY params in kernel %d\n", k
        );
        free(call_src);
        goto cleanup_linear;
      }
      if (!body || body->op != POLY_OP_COPY) {
        body = make_call_copy_body(ctx, ps, dst_slot, src_slot);
        if (!body) {
          free(call_src);
          goto cleanup_linear;
        }
      }
      call_src[1] = ps->template->buf_slots[dst_slot].buf_uop;
      call_src[2] = ps->template->buf_slots[src_slot].buf_uop;
    } else {
      for (int i = 0; i < n_params; i++) {
        int slot =
            schedule_slot_for_kernel_param(&sr, k, i, buf_order_orig, n_bufs_orig, inter_set);

        if (slot < 0) {
          fprintf(
              stderr,
              "polygrad: build_schedule_from_kernel_graph: unresolved param %d in kernel %d\n", i, k
          );
          free(call_src);
          goto cleanup_linear;
        }
        call_src[1 + i] = ps->template->buf_slots[slot].buf_uop;
      }
    }

    call_src[0] = body;
    for (int i = 0; i < n_vars; i++)
      call_src[1 + n_params + i] = sr.var_to_buf && sr.var_to_buf[k] ? sr.var_to_buf[k][i] : NULL;
    PolyUOp *call = poly_uop(ctx, POLY_OP_CALL, POLY_VOID, call_src, n_call_src, poly_arg_none());
    if (!call) {
      free(call_src);
      goto cleanup_linear;
    }
    if (!is_copy) {
      PolyUOp *program = poly_program_from_call(ctx, call, "test");
      if (!program) {
        free(call_src);
        goto cleanup_linear;
      }
      call_src[0] = program;
      call = poly_uop(ctx, POLY_OP_CALL, POLY_VOID, call_src, n_call_src, poly_arg_none());
      if (!call) {
        free(call_src);
        goto cleanup_linear;
      }
    }
    linear_src[step] = call;
    free(call_src);
  }

  if (inter_set) poly_map_destroy(inter_set);
  inter_set = NULL;

  ps->template->linear = poly_uop(ctx, POLY_OP_LINEAR, POLY_VOID, linear_src, sr.n_kernels, poly_arg_none());
  free(linear_src);
  linear_src = NULL;
  if (!ps->template->linear) goto cleanup;
  ps->template->n_calls = ps->template->linear->n_src;

  if (apply_memory_plan && poly_schedule_memory_plan(ctx, ps) != 0) goto cleanup;
  ps->template->n_calls = ps->template->linear->n_src;

  if (poly_schedule_init_call_io_and_runtime(ctx, ps) != 0) goto cleanup;

  poly_kernel_schedule_result_free(&sr);
  return ps;

cleanup_linear:
  free(linear_src);
  if (inter_set) {
    poly_map_destroy(inter_set);
    inter_set = NULL;
  }
cleanup:
  poly_schedule_free(ps);
  poly_kernel_schedule_result_free(&sr);
  return NULL;
}

/* tinygrad/engine/schedule.py: complete_create_schedule_with_vars */

static PolyScheduleCacheEntry *poly_lower_kernel_graph_to_cache_entry(
    PolyCtx *ctx,
    PolyUOp *kernel_graph
);
static PolyScheduleCacheEntry *poly_lower_sink_to_cache_entry_with_kernel_graph(
    PolyCtx *ctx,
    PolyUOp *sink,
    PolyCompileMode mode,
    PolyUOp *kernel_graph
);

static bool poly_schedule_cache_enabled(void) {
  return poly_getenv_flag_default("POLY_SCACHE", true);
}

PolySchedule *poly_complete_create_schedule_with_vars(
    PolyCtx *ctx,
    PolyUOp *sink,
    PolyCompileMode mode
) {
  if (!sink || sink->op != POLY_OP_SINK) {
    fprintf(stderr, "polygrad: complete_create_schedule_with_vars: expected SINK\n");
    return NULL;
  }

  bool timing = poly_debug_at_least(2);
  double t0 = timing ? poly_now_ms() : 0.0;
  if (timing) {
    fprintf(stderr, "[polygrad:create_schedule] begin sink=%p mode=%d\n", (void *)sink, (int)mode);
    fflush(stderr);
  }

  /* --- Collect pre-strip buffer ordering -------------------------------- */
  PolyUOp *buf_order_orig_stack[POLY_MAX_REALIZE_BUFS];
  PolyUOp **buf_order_orig = buf_order_orig_stack;
  int n_bufs_orig = 0, n_dfs = 0;
  bool buf_order_orig_owned = false;
  poly_collect_buf_order(sink, buf_order_orig_stack, &n_bufs_orig, NULL, &n_dfs);
  if (n_bufs_orig > POLY_MAX_REALIZE_BUFS) {
    buf_order_orig = NULL;
    n_bufs_orig = 0;
    n_dfs = 0;
    if (!poly_collect_buf_order_alloc(sink, &buf_order_orig, &n_bufs_orig, &n_dfs)) return NULL;
    buf_order_orig_owned = true;
  }
  double t_bufs = timing ? poly_now_ms() : 0.0;

  PolyUOp *used_vars[64];
  int n_used_vars = collect_define_vars(ctx, sink, used_vars, 64);
  if (n_used_vars < 0) {
    if (buf_order_orig_owned) free(buf_order_orig);
    return NULL;
  }

  PolyVarBinding bind_vals[64];
  int n_bind_vals = collect_bind_defaults(ctx, sink, used_vars, n_used_vars, bind_vals, 64);
  if (n_bind_vals < 0) {
    if (buf_order_orig_owned) free(buf_order_orig);
    return NULL;
  }
  double t_vars = timing ? poly_now_ms() : 0.0;
  if (timing) {
    fprintf(
        stderr,
        "[polygrad:create_schedule] prepass bufs=%d dfs=%d vars=%d binds=%d collect=%.3fms "
        "vars=%.3fms\n",
        n_bufs_orig, n_dfs, n_used_vars, n_bind_vals, t_bufs - t0, t_vars - t_bufs
    );
    fflush(stderr);
  }

  if (poly_schedule_cache_enabled()) {
    /* Tinygrad caches LINEAR templates, but it does not execute a cached
     * LINEAR directly. It resolves the cached CALL roots against the current
     * call's buffers before running. In Polygrad that current-call resolution
     * is build_schedule_from_kernel_graph(..., linear_template): the current
     * kernel graph supplies slots/intermediates/order, while the cached LINEAR
     * can donate already-lowered kernel roots when the schedule shape matches.
     */
    double t_lower0 = timing ? poly_now_ms() : 0.0;
    PolyUOp *kernel_graph = poly_get_kernel_graph(ctx, sink);
    if (!kernel_graph) {
      if (buf_order_orig_owned) free(buf_order_orig);
      return NULL;
    }
    PolyScheduleCacheEntry *cache_entry =
        poly_lower_sink_to_cache_entry_with_kernel_graph(ctx, sink, mode, kernel_graph);
    double t_lower1 = timing ? poly_now_ms() : 0.0;
    if (!cache_entry) {
      if (buf_order_orig_owned) free(buf_order_orig);
      return NULL;
    }
    uint32_t ghash = poly_structural_hash(kernel_graph) ^ (POLY_SCHED_CACHE_VERSION * 2654435761u);
    double t_build0 = timing ? poly_now_ms() : 0.0;
    PolySchedule *ps = build_schedule_from_linear_template(
        ctx, cache_entry, mode, ghash, buf_order_orig, n_bufs_orig, bind_vals, n_bind_vals, true
    );
    if (!ps) {
      /* Keep the old resolver as a correctness fallback for cached LINEAR
       * shapes that still contain non-replayable call arguments. */
      ps = build_schedule_from_kernel_graph(
          ctx, kernel_graph, mode, ghash, buf_order_orig, n_bufs_orig, bind_vals, n_bind_vals,
          cache_entry->linear, true
      );
    }
    double t_build1 = timing ? poly_now_ms() : 0.0;
    if (timing) {
      fprintf(
          stderr,
          "[polygrad:create_schedule] done path=cached_linear_current_graph lower=%.3fms "
          "build=%.3fms total=%.3fms calls=%d slots=%d\n",
          t_lower1 - t_lower0, t_build1 - t_build0, t_build1 - t0, ps ? ps->template->n_calls : -1,
          ps ? ps->template->n_buf_slots : -1
      );
      fflush(stderr);
    }
    if (buf_order_orig_owned) free(buf_order_orig);
    return ps;
  }

  double t_kernel0 = timing ? poly_now_ms() : 0.0;
  PolyUOp *kernel_graph = poly_get_kernel_graph(ctx, sink);
  double t_kernel1 = timing ? poly_now_ms() : 0.0;
  if (!kernel_graph) {
    if (buf_order_orig_owned) free(buf_order_orig);
    return NULL;
  }
  uint32_t ghash = poly_structural_hash(kernel_graph) ^ (POLY_SCHED_CACHE_VERSION * 2654435761u);
  double t_build0 = timing ? poly_now_ms() : 0.0;
  PolySchedule *ps = build_schedule_from_kernel_graph(
      ctx, kernel_graph, mode, ghash, buf_order_orig, n_bufs_orig, bind_vals, n_bind_vals, NULL,
      true
  );
  double t_build1 = timing ? poly_now_ms() : 0.0;
  if (timing) {
    fprintf(
        stderr,
        "[polygrad:create_schedule] done path=kernel_graph kernel=%.3fms build=%.3fms total=%.3fms "
        "calls=%d slots=%d\n",
        t_kernel1 - t_kernel0, t_build1 - t_build0, t_build1 - t0, ps ? ps->template->n_calls : -1,
        ps ? ps->template->n_buf_slots : -1
    );
    fflush(stderr);
  }
  if (buf_order_orig_owned) free(buf_order_orig);
  return ps;
}

PolySchedule *poly_schedule_replay_with_buffers(
    PolyCtx *ctx,
    const PolySchedule *captured,
    PolyUOp **external_bufs,
    int n_external
) {
  if (!ctx || !captured || !captured->template || !captured->template->cache_entry ||
      n_external < 0 || (n_external > 0 && !external_bufs))
    return NULL;
  if (n_external != poly_schedule_external_slot_count(captured)) return NULL;

  /* This is Polygrad's C-side equivalent of tinygrad JIT replay resolving
   * PARAM-backed CALL arguments against the current input_uops. The captured
   * schedule owns the parameterized LINEAR cache entry; this function creates a
   * fresh concrete schedule with current external BUFFER identities and fresh
   * intermediates, while preserving the captured outputs and default vars. */
  return build_schedule_from_linear_template(
      ctx, captured->template->cache_entry, captured->template->mode, captured->template->graph_hash,
      external_bufs, n_external, captured->template->default_vars, captured->template->n_default_vars,
      true
  );
}

static int schedule_append_var_binding(
    PolyVarBinding **vars,
    int *n_vars,
    int *cap_vars,
    PolyVarBinding binding
) {
  if (!vars || !n_vars || !cap_vars || !binding.var) return -1;
  for (int i = 0; i < *n_vars; i++) {
    if ((*vars)[i].var != binding.var) continue;
    return ((*vars)[i].value == binding.value) ? 0 : -1;
  }
  if (*n_vars >= *cap_vars) {
    int new_cap = *cap_vars ? *cap_vars * 2 : 8;
    PolyVarBinding *tmp = realloc(*vars, (size_t)new_cap * sizeof(PolyVarBinding));
    if (!tmp) return -1;
    *vars = tmp;
    *cap_vars = new_cap;
  }
  (*vars)[(*n_vars)++] = binding;
  return 0;
}

static int schedule_combined_intermediate_index(
    const PolyScheduleCacheEntry *entry,
    PolyUOp *template_buf
) {
  if (!entry || !template_buf) return -1;
  for (int i = 0; i < entry->n_intermediates; i++)
    if (entry->intermediates[i].template_buf == template_buf) return i;
  return -1;
}

static int schedule_combined_append_intermediate(
    PolyScheduleCacheEntry *dst,
    const PolyScheduleCacheEntry *src,
    PolyUOp *template_buf
) {
  if (!dst || !src || !template_buf) return -1;
  if (schedule_combined_intermediate_index(dst, template_buf) >= 0) return 0;
  int src_idx = schedule_combined_intermediate_index(src, template_buf);
  if (src_idx < 0) return -1;
  PolyScheduleIntermediateDesc *tmp =
      realloc(dst->intermediates, (size_t)(dst->n_intermediates + 1) * sizeof(*dst->intermediates));
  if (!tmp) return -1;
  dst->intermediates = tmp;
  dst->intermediates[dst->n_intermediates++] = src->intermediates[src_idx];
  return 0;
}

static bool schedule_prune_internal_contains(PolyUOp **items, int n_items, PolyUOp *buf) {
  for (int i = 0; i < n_items; i++)
    if (items[i] == buf) return true;
  return false;
}

static int schedule_prune_mark_internal(
    PolyUOp ***items,
    int *n_items,
    int *cap_items,
    PolyUOp *buf
) {
  if (!items || !n_items || !cap_items || !buf) return -1;
  if (schedule_prune_internal_contains(*items, *n_items, buf)) return 0;
  if (*n_items >= *cap_items) {
    int new_cap = *cap_items ? *cap_items * 2 : 8;
    PolyUOp **tmp = realloc(*items, (size_t)new_cap * sizeof(*tmp));
    if (!tmp) return -1;
    *items = tmp;
    *cap_items = new_cap;
  }
  (*items)[(*n_items)++] = buf;
  return 0;
}

static int schedule_prune_call_intersects_needed(
    PolyUOp *call,
    bool *needed_external,
    int n_external,
    PolyUOp **needed_internal,
    int n_needed_internal,
    bool *intersects_out
) {
  if (!call || (n_external > 0 && !needed_external) || !intersects_out) return -1;
  *intersects_out = false;
  for (int i = 1; i < call->n_src; i++) {
    PolyUOp *arg = call->src[i];
    int slot = -1;
    if (poly_call_arg_is_var(arg)) continue;
    if (linear_replay_arg_is_external_param(arg, &slot)) {
      if (slot < 0 || slot >= n_external) return -1;
      if (needed_external[slot]) {
        *intersects_out = true;
        return 0;
      }
      continue;
    }
    if (!arg || arg->op != POLY_OP_BUFFER) return -1;
    if (schedule_prune_internal_contains(needed_internal, n_needed_internal, arg)) {
      *intersects_out = true;
      return 0;
    }
  }
  return 0;
}

static int schedule_prune_mark_call_buffers(
    PolyUOp *call,
    bool *needed_external,
    int n_external,
    PolyUOp ***needed_internal,
    int *n_needed_internal,
    int *cap_needed_internal
) {
  if (!call || (n_external > 0 && !needed_external) || !needed_internal ||
      !n_needed_internal || !cap_needed_internal)
    return -1;
  for (int i = 1; i < call->n_src; i++) {
    PolyUOp *arg = call->src[i];
    int slot = -1;
    if (poly_call_arg_is_var(arg)) continue;
    if (linear_replay_arg_is_external_param(arg, &slot)) {
      if (slot < 0 || slot >= n_external) return -1;
      needed_external[slot] = true;
      continue;
    }
    if (!arg || arg->op != POLY_OP_BUFFER) return -1;
    if (schedule_prune_mark_internal(
            needed_internal, n_needed_internal, cap_needed_internal, arg
        ) != 0)
      return -1;
  }
  return 0;
}

PolySchedule *poly_schedule_prune_for_buffers(
    PolyCtx *ctx,
    const PolySchedule *captured,
    PolyUOp **needed_bufs,
    int n_needed
) {
  if (!ctx || !captured || !captured->template || !captured->template->cache_entry ||
      !captured->template->cache_entry->linear || n_needed < 0 ||
      (n_needed > 0 && !needed_bufs))
    return NULL;

  int n_external = poly_schedule_external_slot_count(captured);
  if (n_external < 0) return NULL;

  bool *needed_external = NULL;
  bool *keep = NULL;
  PolyUOp **external = NULL;
  PolyUOp **linear_src = NULL;
  PolyUOp **needed_internal = NULL;
  int n_needed_internal = 0, cap_needed_internal = 0;
  PolyScheduleCacheEntry *entry = NULL;
  PolySchedule *out = NULL;

  if (n_external > 0) {
    needed_external = calloc((size_t)n_external, sizeof(*needed_external));
    external = calloc((size_t)n_external, sizeof(*external));
    if (!needed_external || !external) goto cleanup;
  }

  for (int i = 0; i < n_external; i++) {
    external[i] = poly_schedule_external_slot_buffer(captured, i);
    for (int j = 0; j < n_needed; j++) {
      if (external[i] == needed_bufs[j]) {
        needed_external[i] = true;
        break;
      }
    }
  }

  PolyScheduleCacheEntry *src_entry = captured->template->cache_entry;
  PolyUOp *src_linear = src_entry->linear;
  int n_calls = src_linear->n_src;
  keep = calloc((size_t)(n_calls > 0 ? n_calls : 1), sizeof(*keep));
  if (!keep) goto cleanup;

  int n_keep = 0;
  for (int k = 0; k < n_calls; k++) {
    PolyUOp *call = src_linear->src[k];
    bool intersects = false;
    if (schedule_prune_call_intersects_needed(
            call, needed_external, n_external, needed_internal, n_needed_internal, &intersects
        ) != 0)
      goto cleanup;
    if (!intersects) continue;
    keep[k] = true;
    n_keep++;
    if (schedule_prune_mark_call_buffers(
            call, needed_external, n_external, &needed_internal, &n_needed_internal,
            &cap_needed_internal
        ) != 0)
      goto cleanup;
  }

  entry = calloc(1, sizeof(*entry));
  if (!entry) goto cleanup;
  entry->refcount = 1;
  entry->n_calls = n_keep;
  entry->n_intermediates = src_entry->n_intermediates;
  if (entry->n_intermediates > 0) {
    entry->intermediates =
        malloc((size_t)entry->n_intermediates * sizeof(*entry->intermediates));
    if (!entry->intermediates) goto cleanup;
    memcpy(
        entry->intermediates, src_entry->intermediates,
        (size_t)entry->n_intermediates * sizeof(*entry->intermediates)
    );
  }
  if (n_keep > 0) {
    linear_src = calloc((size_t)n_keep, sizeof(*linear_src));
    entry->call_access = calloc((size_t)n_keep, sizeof(*entry->call_access));
    if (!linear_src || !entry->call_access) goto cleanup;
  }

  int out_call = 0;
  for (int k = 0; k < n_calls; k++) {
    if (!keep[k]) continue;
    linear_src[out_call] = src_linear->src[k];
    if (poly_call_access_clone(&entry->call_access[out_call], &src_entry->call_access[k]) != 0)
      goto cleanup;
    out_call++;
  }

  entry->linear = poly_uop(ctx, POLY_OP_LINEAR, POLY_VOID, linear_src, n_keep, poly_arg_none());
  if (!entry->linear) goto cleanup;
  out = build_schedule_from_linear_template(
      ctx, entry, captured->template->mode, captured->template->graph_hash ^ 0x9e3779b9u,
      external, n_external, captured->template->default_vars, captured->template->n_default_vars,
      true
  );

cleanup:
  free(needed_external);
  free(keep);
  free(external);
  free(linear_src);
  free(needed_internal);
  poly_schedule_cache_entry_release(entry);
  return out;
}

PolySchedule *poly_schedule_replay_many_with_buffers(
    PolyCtx *ctx,
    PolySchedule **captured,
    int n_captured,
    PolyUOp **captured_external_bufs,
    PolyUOp **replay_external_bufs,
    int n_external
) {
  if (!ctx || !captured || n_captured <= 0 || n_external < 0 ||
      (n_external > 0 && (!captured_external_bufs || !replay_external_bufs)))
    return NULL;

  int n_calls = 0;
  PolyCompileMode mode = POLY_MODE_CALL;
  uint32_t graph_hash = POLY_SCHED_CACHE_VERSION * 2654435761u;
  for (int s = 0; s < n_captured; s++) {
    PolySchedule *sched = captured[s];
    if (!sched || !sched->template || !sched->template->cache_entry ||
        !sched->template->cache_entry->linear ||
        sched->template->cache_entry->linear->op != POLY_OP_LINEAR)
      return NULL;
    if (s == 0) mode = sched->template->mode;
    else if (sched->template->mode != mode) return NULL;
    n_calls += sched->template->cache_entry->linear->n_src;
    graph_hash ^= sched->template->graph_hash + 0x9e3779b9u + (graph_hash << 6) + (graph_hash >> 2);
  }

  PolyScheduleCacheEntry *entry = calloc(1, sizeof(*entry));
  PolyUOp **linear_src = NULL;
  PolyVarBinding *vars = NULL;
  int n_vars = 0, cap_vars = 0;
  PolySchedule *out = NULL;
  if (!entry) goto cleanup;
  entry->refcount = 1;
  entry->n_calls = n_calls;
  if (n_calls > 0) {
    linear_src = calloc((size_t)n_calls, sizeof(*linear_src));
    entry->call_access = calloc((size_t)n_calls, sizeof(*entry->call_access));
    if (!linear_src || !entry->call_access) goto cleanup;
  }

  int out_call = 0;
  for (int s = 0; s < n_captured; s++) {
    PolySchedule *sched = captured[s];
    PolyScheduleCacheEntry *src_entry = sched->template->cache_entry;
    PolyUOp *src_linear = src_entry->linear;
    for (int v = 0; v < sched->template->n_default_vars; v++) {
      if (schedule_append_var_binding(
              &vars, &n_vars, &cap_vars, sched->template->default_vars[v]
          ) != 0)
        goto cleanup;
    }
    for (int k = 0; k < src_linear->n_src; k++, out_call++) {
      PolyUOp *call = src_linear->src[k];
      if (!call || call->op != POLY_OP_CALL || call->n_src < 1) goto cleanup;
      PolyUOp **call_src = calloc((size_t)call->n_src, sizeof(*call_src));
      if (!call_src) goto cleanup;
      call_src[0] = call->src[0];
      bool ok = true;
      for (int i = 1; i < call->n_src; i++) {
        PolyUOp *arg = call->src[i];
        int old_slot = -1;
        if (poly_call_arg_is_var(arg)) {
          call_src[i] = arg;
        } else if (linear_replay_arg_is_external_param(arg, &old_slot)) {
          PolyUOp *old_buf = poly_schedule_external_slot_buffer(sched, old_slot);
          int new_slot = poly_find_buf_position(old_buf, captured_external_bufs, n_external);
          if (new_slot < 0) {
            ok = false;
            break;
          }
          call_src[i] = poly_uop0(ctx, POLY_OP_PARAM, arg->dtype, poly_arg_int(new_slot));
        } else if (arg && arg->op == POLY_OP_BUFFER) {
          if (schedule_combined_append_intermediate(entry, src_entry, arg) != 0) {
            ok = false;
            break;
          }
          call_src[i] = arg;
        } else {
          ok = false;
          break;
        }
      }
      if (ok) linear_src[out_call] = poly_uop(ctx, POLY_OP_CALL, POLY_VOID, call_src, call->n_src, call->arg);
      free(call_src);
      if (!ok || !linear_src[out_call]) goto cleanup;
      if (poly_call_access_clone(&entry->call_access[out_call], &src_entry->call_access[k]) != 0)
        goto cleanup;
    }
  }

  entry->linear = poly_uop(ctx, POLY_OP_LINEAR, POLY_VOID, linear_src, n_calls, poly_arg_none());
  if (!entry->linear) goto cleanup;
  out = build_schedule_from_linear_template(
      ctx, entry, mode, graph_hash, replay_external_bufs, n_external, vars, n_vars, true
  );

cleanup:
  free(linear_src);
  free(vars);
  poly_schedule_cache_entry_release(entry);
  return out;
}

static PolySchedule *poly_create_schedule_uncached(PolyCtx *ctx, PolyUOp *kernel_graph) {
  if (!kernel_graph || kernel_graph->op != POLY_OP_SINK) {
    fprintf(stderr, "polygrad: create_schedule: expected SINK\n");
    return NULL;
  }

  PolyUOp *buf_order_stack[POLY_MAX_REALIZE_BUFS];
  PolyUOp **buf_order = buf_order_stack;
  int n_external = 0;
  bool buf_order_owned = false;
  if (!collect_external_buf_order_from_kernel_graph(
          kernel_graph, &buf_order, &n_external, buf_order_stack, &buf_order_owned
      ))
    return NULL;
  uint32_t ghash = poly_structural_hash(kernel_graph) ^ (POLY_SCHED_CACHE_VERSION * 2654435761u);
  PolySchedule *schedule = build_schedule_from_kernel_graph(
      ctx, kernel_graph, POLY_MODE_CALL, ghash, buf_order, n_external, NULL, 0, NULL, false
  );
  if (buf_order_owned) free(buf_order);
  return schedule;
}

/* tinygrad/engine/schedule.py: create_schedule
 * Build the backend-neutral schedule from the kernel-graph boundary, resolving
 * cached LINEAR kernel roots against the current graph's buffers/vars. */
PolySchedule *poly_create_schedule(PolyCtx *ctx, PolyUOp *kernel_graph) {
  if (!kernel_graph || kernel_graph->op != POLY_OP_SINK) {
    fprintf(stderr, "polygrad: create_schedule: expected SINK\n");
    return NULL;
  }

  PolyScheduleCacheEntry *cache_entry =
      poly_schedule_cache_enabled() ? poly_lower_kernel_graph_to_cache_entry(ctx, kernel_graph) : NULL;
  if (poly_schedule_cache_enabled() && !cache_entry) return NULL;

  PolyUOp *buf_order_stack[POLY_MAX_REALIZE_BUFS];
  PolyUOp **buf_order = buf_order_stack;
  int n_external = 0;
  bool buf_order_owned = false;
  if (!collect_external_buf_order_from_kernel_graph(
          kernel_graph, &buf_order, &n_external, buf_order_stack, &buf_order_owned
      ))
    return NULL;
  uint32_t ghash = poly_structural_hash(kernel_graph) ^ (POLY_SCHED_CACHE_VERSION * 2654435761u);
  PolySchedule *schedule = build_schedule_from_linear_template(
      ctx, cache_entry, POLY_MODE_CALL, ghash, buf_order, n_external, NULL, 0, true
  );
  if (!schedule) {
    schedule = build_schedule_from_kernel_graph(
        ctx, kernel_graph, POLY_MODE_CALL, ghash, buf_order, n_external, NULL, 0,
        cache_entry ? cache_entry->linear : NULL,
        true
    );
  }
  if (buf_order_owned) free(buf_order);
  return schedule;
}

static void poly_runner_cleanup(PolyRunner *runner, PolyDevice device) {
  if (!runner) return;
  const PolyBackendDesc *backend = poly_backend_get(device);
  if (runner->handle && !runner->borrowed_handle) {
    if (runner->free_handle)
      runner->free_handle(runner);
    else if (backend)
      backend->free_runner(runner);
  }
  free(runner->param_to_slot);
  free(runner->var_indices);
  memset(runner, 0, sizeof(*runner));
}

static void poly_runner_cleanup_local_mappings(PolyRunner *runner) {
  if (!runner) return;
  free(runner->param_to_slot);
  free(runner->var_indices);
  memset(runner, 0, sizeof(*runner));
}

static PolyRuntimeCacheEntry *poly_runtime_cache_entry_new(
    PolyCtx *ctx,
    PolyUOp *program,
    PolyDevice device,
    uint32_t env_stamp,
    PolyRunner *runner
) {
  if (!runner) return NULL;
  PolyRuntimeCacheEntry *entry = calloc(1, sizeof(*entry));
  if (!entry) return NULL;
  entry->refcount = 1;
  entry->ctx = ctx;
  entry->accounted_bytes = sizeof(*entry);
  if (runner->handle_size > 0) entry->accounted_bytes += (size_t)runner->handle_size;
  entry->program = program;
  entry->device = device;
  entry->env_stamp = env_stamp;
  entry->runner = *runner;
  if (ctx) {
    ctx->runtime_artifact_entries++;
    ctx->runtime_artifact_live_bytes += entry->accounted_bytes;
  }
  return entry;
}

static PolyRuntimeCacheEntry *poly_runtime_cache_entry_retain(PolyRuntimeCacheEntry *entry) {
  if (entry) entry->refcount++;
  return entry;
}

static void poly_runtime_cache_entry_release(PolyRuntimeCacheEntry *entry) {
  if (!entry) return;
  entry->refcount--;
  if (entry->refcount <= 0) {
    if (entry->ctx) {
      if (entry->ctx->runtime_artifact_entries > 0) entry->ctx->runtime_artifact_entries--;
      if (entry->ctx->runtime_artifact_live_bytes >= entry->accounted_bytes)
        entry->ctx->runtime_artifact_live_bytes -= entry->accounted_bytes;
      else
        entry->ctx->runtime_artifact_live_bytes = 0;
    }
    poly_runner_cleanup(&entry->runner, entry->device);
    free(entry);
  }
}

static bool poly_program_cache_enabled(void) {
  return poly_getenv_flag_default("POLY_PCACHE", true);
}

static uint32_t poly_program_cache_hash(PolyUOp *program, PolyDevice device, uint32_t env_stamp) {
  /* tinygrad caches by PROGRAM ast.key. Polygrad UOps are ctx-interned, so for
   * the ctx-local runtime caches PROGRAM identity is the cached key object:
   * structurally identical scheduled kernels share the same PROGRAM through the
   * LINEAR/schedule cache, while intentionally distinct PROGRAM wrappers remain
   * distinct. Avoid recomputing full structural hashes on every warm call. */
  uint32_t h = poly_ptr_hash(program);
  h ^= ((uint32_t)device + 0x9e3779b9u + (h << 6) + (h >> 2));
  h ^= (env_stamp + 0x85ebca6bu + (h << 6) + (h >> 2));
  return h;
}

static bool poly_runtime_cache_eq(const void *a, const void *b) {
  const PolyRuntimeCacheMapEntry *ka = (const PolyRuntimeCacheMapEntry *)a;
  const PolyRuntimeCacheMapEntry *kb = (const PolyRuntimeCacheMapEntry *)b;
  return ka && kb && ka->device == kb->device && ka->env_stamp == kb->env_stamp &&
         ka->program == kb->program;
}

static bool poly_to_program_cache_eq(const void *a, const void *b) {
  const PolyToProgramCacheEntry *ka = (const PolyToProgramCacheEntry *)a;
  const PolyToProgramCacheEntry *kb = (const PolyToProgramCacheEntry *)b;
  return ka && kb && ka->device == kb->device && ka->env_stamp == kb->env_stamp &&
         ka->program == kb->program;
}

static void poly_runtime_cache_entry_free(const void *key, void *value, void *userdata) {
  (void)key;
  (void)userdata;
  PolyRuntimeCacheMapEntry *entry = (PolyRuntimeCacheMapEntry *)value;
  if (!entry) return;
  if (entry->runtime_program) entry->runtime_program->in_cache = false;
  poly_runtime_cache_entry_release(entry->runtime_program);
  free(entry);
}

void poly_runtime_cache_clear(PolyCtx *ctx) {
  if (!ctx || !ctx->runtime_cache) return;
  poly_map_foreach(ctx->runtime_cache, poly_runtime_cache_entry_free, NULL);
  poly_map_clear(ctx->runtime_cache);
}

size_t poly_runtime_cache_len(PolyCtx *ctx) {
  return (ctx && ctx->runtime_cache) ? poly_map_len(ctx->runtime_cache) : 0;
}

static void poly_runtime_cache_artifact_size_accum(const void *key, void *value, void *userdata) {
  (void)key;
  size_t *total = (size_t *)userdata;
  PolyRuntimeCacheMapEntry *entry = (PolyRuntimeCacheMapEntry *)value;
  if (!total || !entry) return;
  *total += sizeof(*entry);
}

size_t poly_runtime_cache_artifact_bytes(PolyCtx *ctx) {
  if (!ctx) return 0;
  size_t total = ctx->runtime_artifact_live_bytes;
  if (ctx->runtime_cache) poly_map_foreach(ctx->runtime_cache, poly_runtime_cache_artifact_size_accum, &total);
  return total;
}

size_t poly_program_cache_len(PolyCtx *ctx) {
  return poly_runtime_cache_len(ctx);
}

void poly_program_cache_clear(PolyCtx *ctx) {
  poly_runtime_cache_clear(ctx);
}

size_t poly_program_cache_artifact_bytes(PolyCtx *ctx) {
  return poly_runtime_cache_artifact_bytes(ctx);
}

void poly_to_program_cache_clear(PolyCtx *ctx) {
  if (ctx && ctx->to_program_cache) poly_map_clear(ctx->to_program_cache);
}

size_t poly_to_program_cache_len(PolyCtx *ctx) {
  return (ctx && ctx->to_program_cache) ? poly_map_len(ctx->to_program_cache) : 0;
}

static void poly_call_runtime_cleanup(PolyCallRuntime *rt) {
  if (!rt) return;
  if (rt->runtime_program) {
    poly_runner_cleanup_local_mappings(&rt->prg);
    poly_runtime_cache_entry_release(rt->runtime_program);
  } else if (rt->prg_valid) {
    poly_runner_cleanup(&rt->prg, rt->lowered_device);
  } else {
    poly_runner_cleanup_local_mappings(&rt->prg);
  }
  rt->runtime_program = NULL;
  rt->call = NULL;
  rt->prg_valid = false;
  rt->lowered_device = POLY_DEVICE_AUTO;
  rt->lowered_env_stamp = 0;
}

void poly_schedule_ctx_cleanup(PolyCtx *ctx) {
  /* ctx owns cached backend programs, matching tinygrad's global
   * to_program/runtime caches. LINEAR schedule UOps are arena-owned; schedule
   * cache entries only own heap metadata parallel to those UOps. */
  poly_schedule_cache_clear(ctx);
  poly_runtime_cache_clear(ctx);
  poly_to_program_cache_clear(ctx);
}

static void poly_schedule_runtime_cleanup(
    const PolyScheduleTemplate *tpl,
    PolyScheduleRuntime *run
) {
  if (!run) return;
  int n_calls = tpl ? tpl->n_calls : 0;

  for (int i = 0; i < n_calls; i++) {
    PolyCallRuntime *rt = &run->calls[i];
    poly_call_runtime_cleanup(rt);
  }
  if (run->call_io) {
    for (int i = 0; i < n_calls; i++)
      poly_call_io_free(&run->call_io[i]);
    free(run->call_io);
  }

  if (run->intermediates) {
    for (int i = 0; i < run->n_intermediates; i++) {
      if (run->intermediates[i].ptr && run->intermediates[i].allocator)
        run->intermediates[i].allocator->free(
            &run->intermediates[i], run->intermediates[i].allocator->dev_ctx
        );
    }
    free(run->intermediates);
  }
  if (run->kernel_args) {
    for (int i = 0; i < n_calls; i++)
      free(run->kernel_args[i]);
    free(run->kernel_args);
  }
  if (run->slot_views) {
    for (int i = 0; i < run->n_slot_views; i++) {
      if (run->slot_views[i].ptr && run->slot_views[i].allocator && run->slot_views[i].owned)
        run->slot_views[i].allocator->free(&run->slot_views[i], run->slot_views[i].allocator->dev_ctx);
    }
    free(run->slot_views);
  }
  free(run->slot_to_data);
  free(run->ctx_slots);
  free(run->merged_vars);
  free(run->var_int_storage);

  run->device = POLY_DEVICE_AUTO;
  run->call_io = NULL;
  run->allocator = NULL;
  run->intermediates = NULL;
  run->n_intermediates = 0;
  run->kernel_args = NULL;
  run->slot_to_data = NULL;
  run->n_slot_to_data = 0;
  run->slot_views = NULL;
  run->n_slot_views = 0;
  run->ctx_slots = NULL;
  run->n_ctx_slots = 0;
  run->merged_vars = NULL;
  run->merged_vars_cap = 0;
  run->var_int_storage = NULL;
  run->var_int_cap = 0;
}

static void poly_schedule_runtime_destroy(PolySchedule *sched) {
  if (!sched) return;
  poly_schedule_runtime_cleanup(sched->template, sched->run);
}

static size_t poly_schedule_runtime_owned_bytes(const PolyScheduleRuntime *run) {
  if (!run || !run->intermediates) return 0;
  size_t total = 0;
  for (int i = 0; i < run->n_intermediates; i++)
    if (run->intermediates[i].owned) total += run->intermediates[i].nbytes;
  return total;
}

size_t poly_schedule_runtime_intermediate_bytes(const PolySchedule *schedule) {
  return schedule ? poly_schedule_runtime_owned_bytes(schedule->run) : 0;
}

static void poly_schedule_template_destroy(PolyScheduleTemplate *tpl) {
  if (!tpl) return;
  if (tpl->call_access) {
    for (int i = 0; i < tpl->n_calls; i++)
      poly_call_access_free(&tpl->call_access[i]);
    free(tpl->call_access);
  }
  free(tpl->buf_slots);
  free(tpl->default_vars);
  free(tpl->grad_buf_slots);
  poly_schedule_cache_entry_release(tpl->cache_entry);
  free(tpl);
}

static PolyScheduleTemplate *poly_schedule_template_retain(PolyScheduleTemplate *tpl) {
  if (tpl) tpl->refcount++;
  return tpl;
}

static void poly_schedule_template_release(PolyScheduleTemplate *tpl) {
  if (!tpl) return;
  tpl->refcount--;
  if (tpl->refcount <= 0) poly_schedule_template_destroy(tpl);
}

void poly_schedule_free(PolySchedule *step) {
  if (!step) return;
  poly_schedule_runtime_destroy(step);
  if (step->run) {
    free(step->run->calls);
    free(step->run);
  }
  poly_schedule_template_release(step->template);
  free(step);
}

/* tinygrad/engine/schedule.py: lower_sink_to_linear
 * Polygrad exposes a LINEAR UOp whose sources are the scheduled roots in
 * execution order. This keeps the stage boundary inspectable without
 * introducing a second schedule container type. */

static PolyArg schedule_cache_buffer_param_arg(PolyUOp *u, int pos, int64_t *vals, int cap);
static bool schedule_cache_collect_inputs_fast(
    PolyUOp *u,
    PolyUOp **input_order,
    int *n_inputs,
    PolyUOp **visited,
    int *n_visited,
    int depth
);
static bool schedule_cache_collect_inputs_alloc(
    PolyUOp *u,
    PolyUOp ***out_input_order,
    int *n_inputs,
    int *n_visited
);

typedef struct {
  PolyUOp *u;
  int state;
} ScheduleCacheVisit;

static int schedule_cache_rewrite_child_count(PolyUOp *u, bool normalize_buffers) {
  if (!u) return 0;
  if (normalize_buffers && (u->op == POLY_OP_BUFFER || u->op == POLY_OP_BUFFER_VIEW)) return 0;
  if (u->op == POLY_OP_BIND && u->n_src >= 1) return 1;
  return u->n_src;
}

static bool schedule_cache_visit_push(
    ScheduleCacheVisit **stack,
    int *sp,
    int *cap,
    PolyUOp *u,
    int state
) {
  if (*sp >= *cap) {
    int new_cap = *cap * 2;
    ScheduleCacheVisit *new_stack = realloc(*stack, (size_t)new_cap * sizeof(ScheduleCacheVisit));
    if (!new_stack) return false;
    *stack = new_stack;
    *cap = new_cap;
  }
  (*stack)[(*sp)++] = (ScheduleCacheVisit){u, state};
  return true;
}

static PolyUOp *schedule_cache_rewrite_iter(
    PolyCtx *ctx,
    PolyUOp *root,
    PolyUOp **input_order,
    int n_inputs,
    bool normalize_buffers
) {
  if (!ctx || !root) return NULL;

  PolyMap *memo = poly_map_new(64);
  if (!memo) return NULL;

  int stack_cap = 256;
  int sp = 0;
  ScheduleCacheVisit *stack = malloc((size_t)stack_cap * sizeof(ScheduleCacheVisit));
  if (!stack) {
    poly_map_destroy(memo);
    return NULL;
  }
  bool ok = schedule_cache_visit_push(&stack, &sp, &stack_cap, root, 0);

  while (ok && sp > 0) {
    ScheduleCacheVisit cur = stack[--sp];
    PolyUOp *u = cur.u;
    if (!u) {
      ok = false;
      break;
    }
    if (poly_map_get(memo, poly_ptr_hash(u), u, poly_ptr_eq)) continue;

    if (cur.state == 0) {
      ok = schedule_cache_visit_push(&stack, &sp, &stack_cap, u, 1);
      if (!ok) break;

      int n_child = schedule_cache_rewrite_child_count(u, normalize_buffers);
      for (int i = n_child - 1; i >= 0; i--) {
        PolyUOp *child = u->src[i];
        if (!child) {
          ok = false;
          break;
        }
        if (poly_map_get(memo, poly_ptr_hash(child), child, poly_ptr_eq)) continue;
        ok = schedule_cache_visit_push(&stack, &sp, &stack_cap, child, 0);
        if (!ok) break;
      }
      continue;
    }

    PolyUOp *result = u;
    if (normalize_buffers && (u->op == POLY_OP_BUFFER || u->op == POLY_OP_BUFFER_VIEW)) {
      int pos = poly_find_buf_position(u, input_order, n_inputs);
      if (pos >= 0) {
        /* This is Polygrad's C analogue of tinygrad callify.pm_replace_buf for
         * schedule-cache identity: replace concrete BUFFER/BUFFER_VIEW identity
         * with a stable parameter position, while retaining dtype, size/view
         * metadata, and concrete device. This must be a PARAM-like key node:
         * Polygrad structural equality intentionally ignores BUFFER sources. */
        int64_t key_vals[64];
        result = poly_uop0(
            ctx, POLY_OP_PARAM, u->dtype,
            schedule_cache_buffer_param_arg(
                u, pos, key_vals, (int)(sizeof(key_vals) / sizeof(key_vals[0]))
            )
        );
      }
    } else if (u->op == POLY_OP_BIND && u->n_src >= 1) {
      result = poly_map_get(memo, poly_ptr_hash(u->src[0]), u->src[0], poly_ptr_eq);
      if (!result) ok = false;
    } else {
      PolyUOp *stack_src[16];
      PolyUOp **new_src = (u->n_src > (int)(sizeof(stack_src) / sizeof(stack_src[0])))
                              ? malloc((size_t)u->n_src * sizeof(PolyUOp *))
                              : stack_src;
      if (!new_src) {
        ok = false;
      } else {
        bool changed = false;
        for (int i = 0; i < u->n_src; i++) {
          new_src[i] = poly_map_get(memo, poly_ptr_hash(u->src[i]), u->src[i], poly_ptr_eq);
          if (!new_src[i]) {
            ok = false;
            break;
          }
          if (new_src[i] != u->src[i]) changed = true;
        }
        if (ok && changed) result = poly_uop(ctx, u->op, u->dtype, new_src, u->n_src, u->arg);
        if (new_src != stack_src) free(new_src);
      }
    }
    if (ok && !result) ok = false;
    if (ok) poly_map_set(memo, poly_ptr_hash(u), u, result, poly_ptr_eq);
  }

  PolyUOp *ret = ok ? poly_map_get(memo, poly_ptr_hash(root), root, poly_ptr_eq) : NULL;
  free(stack);
  poly_map_destroy(memo);
  return ret;
}

static PolyUOp *schedule_cache_key_for_kernel_graph(PolyCtx *ctx, PolyUOp *kernel_graph) {
  PolyUOp *input_order_stack[POLY_MAX_REALIZE_BUFS];
  PolyUOp *input_visited_stack[POLY_MAX_STRUCT_NODES];
  PolyUOp **input_order = input_order_stack;
  bool input_order_owned = false;
  int n_inputs = 0, n_visited = 0;
  if (!schedule_cache_collect_inputs_fast(
          kernel_graph, input_order_stack, &n_inputs, input_visited_stack, &n_visited, 0
      )) {
    input_order = NULL;
    n_inputs = 0;
    n_visited = 0;
    if (!schedule_cache_collect_inputs_alloc(kernel_graph, &input_order, &n_inputs, &n_visited))
      return NULL;
    input_order_owned = true;
  }
  PolyUOp *ret = schedule_cache_rewrite_iter(ctx, kernel_graph, input_order, n_inputs, true);
  if (input_order_owned) free(input_order);
  return ret;
}

static void poly_schedule_cache_entry_destroy(PolyScheduleCacheEntry *entry) {
  if (!entry) return;
  if (entry->call_access) {
    for (int i = 0; i < entry->n_calls; i++)
      poly_call_access_free(&entry->call_access[i]);
    free(entry->call_access);
  }
  free(entry->intermediates);
  free(entry);
}

static PolyScheduleCacheEntry *poly_schedule_cache_entry_retain(PolyScheduleCacheEntry *entry) {
  if (entry && entry->refcount > 0) entry->refcount++;
  return (entry && entry->refcount > 0) ? entry : NULL;
}

static void poly_schedule_cache_entry_release(PolyScheduleCacheEntry *entry) {
  if (!entry || entry->refcount <= 0) return;
  entry->refcount--;
  if (entry->refcount <= 0) poly_schedule_cache_entry_destroy(entry);
}

static void poly_schedule_cache_entry_free_iter(const void *key, void *value, void *userdata) {
  (void)key;
  (void)userdata;
  PolyScheduleCacheEntry *entry = (PolyScheduleCacheEntry *)value;
  if (entry) entry->in_cache = false;
  poly_schedule_cache_entry_release(entry);
}

static PolyScheduleCacheEntry *poly_schedule_cache_entry_from_schedule(
    const PolySchedule *schedule,
    PolyUOp *linear
) {
  if (!schedule || !schedule->template || !linear) return NULL;
  PolyScheduleCacheEntry *entry = calloc(1, sizeof(PolyScheduleCacheEntry));
  if (!entry) return NULL;
  entry->refcount = 1;
  entry->linear = linear;
  entry->n_calls = schedule->template->n_calls;
  if (entry->n_calls > 0) {
    entry->call_access = poly_call_access_clone_array(schedule->template->call_access, entry->n_calls);
    if (!entry->call_access) {
      poly_schedule_cache_entry_destroy(entry);
      return NULL;
    }
  }

  int n_intermediates = 0;
  for (int i = 0; i < schedule->template->n_buf_slots; i++) {
    const PolyScheduleBufSlot *slot = &schedule->template->buf_slots[i];
    if (slot->is_intermediate && !slot->is_memory_arena && !slot->has_memory_parent &&
        slot->buf_uop)
      n_intermediates++;
  }
  entry->n_intermediates = n_intermediates;
  if (n_intermediates > 0) {
    entry->intermediates = calloc((size_t)n_intermediates, sizeof(PolyScheduleIntermediateDesc));
    if (!entry->intermediates) {
      poly_schedule_cache_entry_destroy(entry);
      return NULL;
    }
    int j = 0;
    for (int i = 0; i < schedule->template->n_buf_slots; i++) {
      const PolyScheduleBufSlot *slot = &schedule->template->buf_slots[i];
      if (!slot->is_intermediate || slot->is_memory_arena || slot->has_memory_parent ||
          !slot->buf_uop)
        continue;
      entry->intermediates[j].template_buf = slot->buf_uop;
      entry->intermediates[j].needs_zero = slot->needs_zero;
      j++;
    }
  }
  return entry;
}

static PolyUOp *poly_build_linear_from_kernel_graph_uncached(
    PolyCtx *ctx,
    PolyUOp *kernel_graph,
    PolyUOp **external_buf_order,
    int n_external_buf_order,
    PolyScheduleCacheEntry **entry_out
) {
  bool timing = poly_debug_at_least(2);
  double t0 = timing ? poly_now_ms() : 0.0;
  if (timing) {
    fprintf(
        stderr, "[polygrad:build_linear] begin kernel_graph=%p external_order=%d\n",
        (void *)kernel_graph, n_external_buf_order
    );
    fflush(stderr);
  }
  PolySchedule *schedule = poly_create_schedule_uncached(ctx, kernel_graph);
  if (!schedule) return NULL;
  double t_schedule = timing ? poly_now_ms() : 0.0;
  if (timing) {
    fprintf(
        stderr, "[polygrad:build_linear] schedule calls=%d slots=%d ms=%.3f\n", schedule->template->n_calls,
        schedule->template->n_buf_slots, t_schedule - t0
    );
    fflush(stderr);
  }

  PolyUOp **external_bufs = NULL;
  bool external_bufs_owned = false;
  PolyUOp *external_bufs_stack[POLY_MAX_REALIZE_BUFS];
  int n_external = 0;
  if (external_buf_order) {
    /* LINEAR CALL params are replayed later against the schedule caller's
     * external buffer slots. For sink-level caching that slot list comes from
     * the original SINK, not from the optimized kernel graph. Keeping the same
     * numbering here prevents dropped/rewritten inputs from shifting params. */
    n_external = n_external_buf_order;
    external_bufs = external_buf_order;
  } else {
    /* Kernel-graph callers do not have a pre-rangeify SINK, so their natural
     * external order is the one discovered from the kernel graph itself. */
    if (!collect_external_buf_order_from_kernel_graph(
            kernel_graph, &external_bufs, &n_external, external_bufs_stack, &external_bufs_owned
        )) {
      poly_schedule_free(schedule);
      return NULL;
    }
  }
  PolyUOp **linear_src = calloc((size_t)schedule->template->n_calls, sizeof(PolyUOp *));
  if (!linear_src) {
    if (external_bufs_owned) free(external_bufs);
    poly_schedule_free(schedule);
    return NULL;
  }
  for (int step = 0; step < schedule->template->n_calls; step++) {
    PolyUOp *call = poly_schedule_call(schedule, step);
    if (!call || call->op != POLY_OP_CALL || call->n_src < 1) {
      free(linear_src);
      if (external_bufs_owned) free(external_bufs);
      poly_schedule_free(schedule);
      return NULL;
    }
    int n_call_src = call->n_src;
    PolyUOp **call_src = calloc((size_t)n_call_src, sizeof(PolyUOp *));
    if (!call_src) {
      free(linear_src);
      if (external_bufs_owned) free(external_bufs);
      poly_schedule_free(schedule);
      return NULL;
    }
    call_src[0] = call->src[0];
    for (int i = 1; i < call->n_src; i++) {
      PolyUOp *arg = call->src[i];
      if (poly_call_arg_is_var(arg)) {
        call_src[i] = arg;
      } else {
        int external_slot = poly_find_buf_position(arg, external_bufs, n_external);
        call_src[i] = linear_call_param_for_buf(ctx, arg, external_slot);
      }
    }
    linear_src[step] =
        poly_uop(ctx, POLY_OP_CALL, POLY_VOID, call_src, n_call_src, poly_arg_none());
    free(call_src);
  }
  PolyUOp *linear =
      poly_uop(ctx, POLY_OP_LINEAR, POLY_VOID, linear_src, schedule->template->n_calls, poly_arg_none());
  free(linear_src);
  if (external_bufs_owned) free(external_bufs);
  PolyScheduleCacheEntry *entry = NULL;
  if (linear && entry_out) {
    entry = poly_schedule_cache_entry_from_schedule(schedule, linear);
    if (!entry) {
      poly_schedule_free(schedule);
      return NULL;
    }
  }
  poly_schedule_free(schedule);
  if (timing) {
    double t_done = poly_now_ms();
    fprintf(
        stderr, "[polygrad:build_linear] done calls=%d build_calls=%.3fms total=%.3fms\n",
        linear ? linear->n_src : -1, t_done - t_schedule, t_done - t0
    );
    fflush(stderr);
  }
  if (entry_out) *entry_out = entry;
  return linear;
}

static PolyScheduleCacheEntry *poly_lower_kernel_graph_to_cache_entry(PolyCtx *ctx, PolyUOp *kernel_graph) {
  PolyUOp *cache_key = schedule_cache_key_for_kernel_graph(ctx, kernel_graph);
  if (!cache_key) return NULL;
  uint32_t cache_hash = poly_ptr_hash(cache_key) ^ (POLY_SCHED_CACHE_VERSION * 2654435761u);
  PolyScheduleCacheEntry *cached = poly_map_get(ctx->schedule_cache, cache_hash, cache_key, poly_ptr_eq);
  if (cached) return cached;

  PolyScheduleCacheEntry *entry = NULL;
  PolyUOp *linear = poly_build_linear_from_kernel_graph_uncached(ctx, kernel_graph, NULL, 0, &entry);
  if (!linear || !entry) return NULL;
  entry->in_cache = true;
  poly_map_set(ctx->schedule_cache, cache_hash, cache_key, entry, poly_ptr_eq);
  return entry;
}

static bool schedule_cache_stack_push(PolyUOp ***stack, int *sp, int *cap, PolyUOp *u) {
  if (*sp >= *cap) {
    int new_cap = *cap * 2;
    PolyUOp **new_stack = realloc(*stack, (size_t)new_cap * sizeof(PolyUOp *));
    if (!new_stack) return false;
    *stack = new_stack;
    *cap = new_cap;
  }
  (*stack)[(*sp)++] = u;
  return true;
}

static bool schedule_cache_collect_inputs_fast(
    PolyUOp *u,
    PolyUOp **input_order,
    int *n_inputs,
    PolyUOp **visited,
    int *n_visited,
    int depth
) {
  if (!u) return true;
  if (!input_order || !n_inputs || !visited || !n_visited) return false;
  if (depth > 512) return false;

  for (int i = 0; i < *n_visited; i++)
    if (visited[i] == u) return true;
  if (*n_visited >= POLY_MAX_STRUCT_NODES) return false;
  visited[(*n_visited)++] = u;

  if (u->op == POLY_OP_BUFFER || u->op == POLY_OP_BUFFER_VIEW) {
    if (*n_inputs >= POLY_MAX_REALIZE_BUFS) return false;
    input_order[(*n_inputs)++] = u;
    return true;
  }

  for (int i = 0; i < u->n_src; i++) {
    if (!schedule_cache_collect_inputs_fast(
            u->src[i], input_order, n_inputs, visited, n_visited, depth + 1
        ))
      return false;
  }
  return true;
}

static bool schedule_cache_collect_inputs_alloc(
    PolyUOp *u,
    PolyUOp ***out_input_order,
    int *n_inputs,
    int *n_visited
) {
  if (!u || !out_input_order || !n_inputs || !n_visited) return false;
  *out_input_order = NULL;
  *n_inputs = 0;
  *n_visited = 0;

  PolyMap *seen = poly_map_new(1024);
  if (!seen) return false;

  PolyUOp **input_order = NULL;
  int input_cap = 0;
  int cap = 1024;
  int sp = 0;
  PolyUOp **stack = malloc((size_t)cap * sizeof(PolyUOp *));
  bool ok = stack && schedule_cache_stack_push(&stack, &sp, &cap, u);

  while (ok && sp > 0) {
    PolyUOp *cur = stack[--sp];
    if (!cur) {
      ok = false;
      break;
    }
    if (poly_map_get(seen, poly_ptr_hash(cur), cur, poly_ptr_eq)) continue;
    poly_map_set(seen, poly_ptr_hash(cur), cur, cur, poly_ptr_eq);
    (*n_visited)++;

    if (cur->op == POLY_OP_BUFFER || cur->op == POLY_OP_BUFFER_VIEW) {
      if (*n_inputs >= input_cap) {
        int new_cap = input_cap ? input_cap * 2 : 16;
        PolyUOp **tmp = realloc(input_order, (size_t)new_cap * sizeof(PolyUOp *));
        if (!tmp) {
          ok = false;
          break;
        }
        input_order = tmp;
        input_cap = new_cap;
      }
      input_order[(*n_inputs)++] = cur;
      continue;
    }

    for (int i = cur->n_src - 1; i >= 0; i--) {
      ok = schedule_cache_stack_push(&stack, &sp, &cap, cur->src[i]);
      if (!ok) break;
    }
  }

  free(stack);
  poly_map_destroy(seen);
  if (!ok) {
    free(input_order);
    return false;
  }
  *out_input_order = input_order;
  return ok;
}

static PolyArg schedule_cache_buffer_param_arg(PolyUOp *u, int pos, int64_t *vals, int cap) {
  int n = 0;
  vals[n++] = (int64_t)(u ? u->op : POLY_OP_NOOP);
  vals[n++] = (int64_t)pos;
  vals[n++] = (int64_t)poly_uop_device(u);
  vals[n++] = (int64_t)(u ? u->arg.kind : POLY_ARG_NONE);
  if (u && u->arg.kind == POLY_ARG_INT) {
    vals[n++] = u->arg.i;
  } else if (u && u->arg.kind == POLY_ARG_INT_TUPLE) {
    vals[n++] = u->arg.int_tuple.n;
    for (int i = 0; i < u->arg.int_tuple.n && n < cap; i++)
      vals[n++] = u->arg.int_tuple.vals[i];
  }

  PolyArg arg = poly_arg_none();
  arg.kind = POLY_ARG_INT_TUPLE;
  arg.int_tuple.vals = vals;
  arg.int_tuple.n = n;
  return arg;
}

static PolyUOp *schedule_cache_key_for_sink(
    PolyCtx *ctx,
    PolyUOp *sink,
    PolyUOp **input_order,
    int n_inputs
) {
  return schedule_cache_rewrite_iter(ctx, sink, input_order, n_inputs, true);
}

static PolyScheduleCacheEntry *poly_lower_sink_to_cache_entry_with_kernel_graph(
    PolyCtx *ctx,
    PolyUOp *sink,
    PolyCompileMode mode,
    PolyUOp *kernel_graph
) {
  (void)mode;
  if (!ctx || !sink || sink->op != POLY_OP_SINK) return NULL;

  bool timing = poly_debug_at_least(2);
  double t0 = timing ? poly_now_ms() : 0.0;
  if (timing) {
    fprintf(stderr, "[polygrad:lower_sink_to_linear] begin sink=%p\n", (void *)sink);
    fflush(stderr);
  }

  PolyUOp *input_order_stack[POLY_MAX_REALIZE_BUFS];
  PolyUOp *input_visited_stack[POLY_MAX_STRUCT_NODES];
  PolyUOp **input_order = input_order_stack;
  bool input_order_owned = false;
  int n_inputs = 0, n_visited = 0;
  if (!schedule_cache_collect_inputs_fast(
          sink, input_order_stack, &n_inputs, input_visited_stack, &n_visited, 0
      )) {
    input_order = NULL;
    n_inputs = 0;
    n_visited = 0;
    if (!schedule_cache_collect_inputs_alloc(sink, &input_order, &n_inputs, &n_visited))
      return NULL;
    input_order_owned = true;
  }
  double t_inputs = timing ? poly_now_ms() : 0.0;

  PolyUOp *cache_key = schedule_cache_key_for_sink(ctx, sink, input_order, n_inputs);
  if (!cache_key) {
    if (input_order_owned) free(input_order);
    return NULL;
  }
  uint32_t cache_hash = poly_ptr_hash(cache_key) ^ (POLY_SCHED_CACHE_VERSION * 2654435761u);
  PolyScheduleCacheEntry *cached =
      poly_schedule_cache_enabled()
          ? poly_map_get(ctx->schedule_cache, cache_hash, cache_key, poly_ptr_eq)
          : NULL;
  double t_key = timing ? poly_now_ms() : 0.0;
  if (cached) {
    if (timing) {
      fprintf(
          stderr,
          "[polygrad:lower_sink_to_linear] cache hit inputs=%d visited=%d inputs_ms=%.3f "
          "key_ms=%.3f total=%.3f\n",
          n_inputs, n_visited, t_inputs - t0, t_key - t_inputs, t_key - t0
      );
      fflush(stderr);
    }
    if (input_order_owned) free(input_order);
    return cached;
  }
  if (timing) {
    fprintf(
        stderr,
        "[polygrad:lower_sink_to_linear] cache miss inputs=%d visited=%d inputs_ms=%.3f "
        "key_ms=%.3f\n",
        n_inputs, n_visited, t_inputs - t0, t_key - t_inputs
    );
    fflush(stderr);
  }

  PolyUOp *raw_external_stack[POLY_MAX_REALIZE_BUFS];
  PolyUOp **raw_external_bufs = raw_external_stack;
  int n_raw_external = 0, n_raw_visited = 0;
  bool raw_external_owned = false;
  poly_collect_buf_order(sink, raw_external_stack, &n_raw_external, NULL, &n_raw_visited);
  if (n_raw_external > POLY_MAX_REALIZE_BUFS) {
    raw_external_bufs = NULL;
    n_raw_external = 0;
    n_raw_visited = 0;
    if (!poly_collect_buf_order_alloc(sink, &raw_external_bufs, &n_raw_external, &n_raw_visited)) {
      if (input_order_owned) free(input_order);
      return NULL;
    }
    raw_external_owned = true;
  }
  double t_raw = timing ? poly_now_ms() : 0.0;

  if (!kernel_graph) kernel_graph = poly_get_kernel_graph(ctx, sink);
  if (!kernel_graph) {
    if (input_order_owned) free(input_order);
    if (raw_external_owned) free(raw_external_bufs);
    return NULL;
  }
  double t_kernel = timing ? poly_now_ms() : 0.0;
  PolyScheduleCacheEntry *entry = NULL;
  PolyUOp *linear = poly_build_linear_from_kernel_graph_uncached(
      ctx, kernel_graph, raw_external_bufs, n_raw_external,
      poly_schedule_cache_enabled() ? &entry : NULL
  );
  if (input_order_owned) free(input_order);
  if (raw_external_owned) free(raw_external_bufs);
  if (!linear) return NULL;
  if (poly_schedule_cache_enabled() && !entry) return NULL;
  double t_linear = timing ? poly_now_ms() : 0.0;
  if (poly_schedule_cache_enabled()) {
    entry->in_cache = true;
    poly_map_set(ctx->schedule_cache, cache_hash, cache_key, entry, poly_ptr_eq);
  }
  if (timing) {
    fprintf(
        stderr,
        "[polygrad:lower_sink_to_linear] done raw_bufs=%d raw_visited=%d raw=%.3fms kernel=%.3fms "
        "linear=%.3fms total=%.3fms calls=%d\n",
        n_raw_external, n_raw_visited, t_raw - t_key, t_kernel - t_raw, t_linear - t_kernel,
        t_linear - t0, linear ? linear->n_src : -1
    );
    fflush(stderr);
  }
  if (poly_schedule_cache_enabled()) return entry;

  static PolyScheduleCacheEntry uncached_entry;
  memset(&uncached_entry, 0, sizeof(uncached_entry));
  uncached_entry.linear = linear;
  return &uncached_entry;
}

static PolyUOp *poly_lower_sink_to_linear_with_kernel_graph(
    PolyCtx *ctx,
    PolyUOp *sink,
    PolyCompileMode mode,
    PolyUOp *kernel_graph
) {
  PolyScheduleCacheEntry *entry =
      poly_lower_sink_to_cache_entry_with_kernel_graph(ctx, sink, mode, kernel_graph);
  return entry ? entry->linear : NULL;
}

PolyUOp *poly_lower_sink_to_linear(PolyCtx *ctx, PolyUOp *sink, PolyCompileMode mode) {
  return poly_lower_sink_to_linear_with_kernel_graph(ctx, sink, mode, NULL);
}

size_t poly_schedule_cache_len(PolyCtx *ctx) {
  return (ctx && ctx->schedule_cache) ? poly_map_len(ctx->schedule_cache) : 0;
}

void poly_schedule_cache_clear(PolyCtx *ctx) {
  if (!ctx || !ctx->schedule_cache) return;
  poly_map_foreach(ctx->schedule_cache, poly_schedule_cache_entry_free_iter, NULL);
  poly_map_clear(ctx->schedule_cache);
}

/* ══════════════════════════════════════════════════════════════════════ */
/*  Allocators                                                          */
/* ══════════════════════════════════════════════════════════════════════ */

/* CPU allocator */

static void *cpu_alloc(size_t nbytes, void *dev_ctx) {
  (void)dev_ctx;
  return calloc(1, nbytes);
}

static void cpu_free_alloc(const PolyBuffer *buffer, void *dev_ctx) {
  (void)dev_ctx;
  free(buffer ? buffer->ptr : NULL);
}

static int cpu_copy_in(const PolyBuffer *dst, const PolyBuffer *src, size_t n, void *dev_ctx) {
  (void)dev_ctx;
  memcpy(dst->ptr, src->ptr, n);
  return 0;
}

static int cpu_copy_out(const PolyBuffer *dst, const PolyBuffer *src, size_t n, void *dev_ctx) {
  (void)dev_ctx;
  memcpy(dst->ptr, src->ptr, n);
  return 0;
}

static int cpu_copy_between(const PolyBuffer *dst, const PolyBuffer *src, size_t n, void *dev_ctx) {
  (void)dev_ctx;
  memcpy(dst->ptr, src->ptr, n);
  return 0;
}

const PolyAllocator POLY_CPU_ALLOCATOR = {
    .alloc = cpu_alloc,
    .free = cpu_free_alloc,
    .copy_in = cpu_copy_in,
    .copy_out = cpu_copy_out,
    .copy_between = cpu_copy_between,
    .host_addressable = true,
    .dev_ctx = NULL,
};

/* HOST allocator
 *
 * Native builds borrow frontend-owned host memory directly. Retiring a HOST
 * residency should drop the frontend's strong owner entry keyed by this
 * PolyBuffer* address value; the actual host bytes remain frontend-managed.
 *
 * Emscripten WebGPU imports use a NULL pointer host key so browser JS can own
 * zero-copy HOST bytes. Retiring that HOST residency still notifies the
 * frontend so it can drop its strong owner entry. Non-WebGPU WASM imports pass
 * a wasm-heap pointer and are adopted as WASM residency in poly_buffer_from_host.
 */

static void *host_alloc(size_t nbytes, void *dev_ctx) {
  (void)nbytes;
  (void)dev_ctx;
  return NULL;
}

static void host_free_alloc(const PolyBuffer *buffer, void *dev_ctx) {
  (void)dev_ctx;
  if (!buffer) return;
  if (buffer->frontend_release)
    buffer->frontend_release((uintptr_t)buffer);
  else
    poly_frontend_buffer_release_key((uintptr_t)buffer);
#ifdef __EMSCRIPTEN__
  free(buffer->ptr);
#endif
}

static int host_copy_in(const PolyBuffer *dst, const PolyBuffer *src, size_t n, void *dev_ctx) {
  (void)dev_ctx;
#ifdef __EMSCRIPTEN__
  if (dst && dst->device == POLY_DEVICE_HOST && src && src->device == POLY_DEVICE_WEBGPU) {
    return poly_webgpu_get_allocator()->copy_out(dst, src, n, NULL);
  }
  if (dst && dst->device == POLY_DEVICE_HOST && dst->ptr && src && src->ptr) {
    memcpy(dst->ptr, src->ptr, n);
    return 0;
  }
  if (dst && dst->device == POLY_DEVICE_HOST && src && src->ptr) {
    uintptr_t key = (uintptr_t)(dst->src ? dst->src : dst);
    return poly_browser_host_copy_in(key, src->ptr, n);
  }
#endif
  if (!dst || !dst->ptr || !src || !src->ptr) return -1;
  memcpy(dst->ptr, src->ptr, n);
  return 0;
}

static int host_copy_out(const PolyBuffer *dst, const PolyBuffer *src, size_t n, void *dev_ctx) {
  (void)dev_ctx;
#ifdef __EMSCRIPTEN__
  if (src && src->device == POLY_DEVICE_HOST && src->ptr && dst && dst->ptr) {
    memcpy(dst->ptr, src->ptr, n);
    return 0;
  }
  if (src && src->device == POLY_DEVICE_HOST && dst && dst->ptr) {
    uintptr_t key = (uintptr_t)(src->src ? src->src : src);
    return poly_browser_host_copy_out(key, dst->ptr, n);
  }
#endif
  if (!dst || !dst->ptr || !src || !src->ptr) return -1;
  memcpy(dst->ptr, src->ptr, n);
  return 0;
}

static int host_copy_between(
    const PolyBuffer *dst,
    const PolyBuffer *src,
    size_t n,
    void *dev_ctx
) {
  (void)dev_ctx;
#ifdef __EMSCRIPTEN__
  if (dst && src && dst->device == POLY_DEVICE_HOST && src->device == POLY_DEVICE_HOST) {
    if (dst->ptr && src->ptr) {
      memcpy(dst->ptr, src->ptr, n);
      return 0;
    }
    uint8_t *tmp = malloc(n);
    if (!tmp) return -1;
    uintptr_t src_key = (uintptr_t)(src->src ? src->src : src);
    uintptr_t dst_key = (uintptr_t)(dst->src ? dst->src : dst);
    int rc = poly_browser_host_copy_out(src_key, tmp, n);
    if (rc == 0) rc = poly_browser_host_copy_in(dst_key, tmp, n);
    free(tmp);
    return rc;
  }
#endif
  if (!dst || !dst->ptr || !src || !src->ptr) return -1;
  memcpy(dst->ptr, src->ptr, n);
  return 0;
}

static const PolyAllocator POLY_HOST_ALLOCATOR = {
    .alloc = host_alloc,
    .free = host_free_alloc,
    .copy_in = host_copy_in,
    .copy_out = host_copy_out,
    .copy_between = host_copy_between,
    .host_addressable =
#ifdef __EMSCRIPTEN__
        false,
#else
        true,
#endif
    .dev_ctx = NULL,
};

/* CUDA allocator */

#ifdef POLY_HAS_CUDA

static void *cuda_alloc_fn(size_t nbytes, void *dev_ctx) {
  (void)dev_ctx;
  unsigned long long dptr = poly_cuda_alloc(nbytes);
  return dptr ? (void *)(uintptr_t)dptr : NULL;
}

static void cuda_free_fn(const PolyBuffer *buffer, void *dev_ctx) {
  (void)dev_ctx;
  if (buffer && buffer->ptr) poly_cuda_free((unsigned long long)(uintptr_t)buffer->ptr);
}

static int cuda_copy_in_fn(const PolyBuffer *dst, const PolyBuffer *src, size_t n, void *dev_ctx) {
  (void)dev_ctx;
  return poly_cuda_copy_htod((unsigned long long)(uintptr_t)dst->ptr, src->ptr, n);
}

static int cuda_copy_out_fn(const PolyBuffer *dst, const PolyBuffer *src, size_t n, void *dev_ctx) {
  (void)dev_ctx;
  return poly_cuda_copy_dtoh(dst->ptr, (unsigned long long)(uintptr_t)src->ptr, n);
}

static int cuda_copy_between_fn(
    const PolyBuffer *dst,
    const PolyBuffer *src,
    size_t n,
    void *dev_ctx
) {
  (void)dev_ctx;
  return poly_cuda_copy_dtod(
      (unsigned long long)(uintptr_t)dst->ptr, (unsigned long long)(uintptr_t)src->ptr, n
  );
}

const PolyAllocator POLY_CUDA_ALLOCATOR = {
    .alloc = cuda_alloc_fn,
    .free = cuda_free_fn,
    .copy_in = cuda_copy_in_fn,
    .copy_out = cuda_copy_out_fn,
    .copy_between = cuda_copy_between_fn,
    .dev_ctx = NULL,
};

#endif /* POLY_HAS_CUDA */

/* HIP allocator */

#ifdef POLY_HAS_HIP

static void *hip_alloc_fn(size_t nbytes, void *dev_ctx) {
  (void)dev_ctx;
  return poly_hip_alloc(nbytes);
}

static void hip_free_fn(const PolyBuffer *buffer, void *dev_ctx) {
  (void)dev_ctx;
  if (buffer && buffer->ptr) poly_hip_free(buffer->ptr);
}

static int hip_copy_in_fn(const PolyBuffer *dst, const PolyBuffer *src, size_t n, void *dev_ctx) {
  (void)dev_ctx;
  return poly_hip_copy_htod(dst->ptr, src->ptr, n);
}

static int hip_copy_out_fn(const PolyBuffer *dst, const PolyBuffer *src, size_t n, void *dev_ctx) {
  (void)dev_ctx;
  return poly_hip_copy_dtoh(dst->ptr, src->ptr, n);
}

static int hip_copy_between_fn(
    const PolyBuffer *dst,
    const PolyBuffer *src,
    size_t n,
    void *dev_ctx
) {
  (void)dev_ctx;
  (void)dst;
  (void)src;
  (void)n;
  fprintf(stderr, "polygrad: hip_copy_between: not implemented\n");
  return -1;
}

const PolyAllocator POLY_HIP_ALLOCATOR = {
    .alloc = hip_alloc_fn,
    .free = hip_free_fn,
    .copy_in = hip_copy_in_fn,
    .copy_out = hip_copy_out_fn,
    .copy_between = hip_copy_between_fn,
    .dev_ctx = NULL,
};

#endif /* POLY_HAS_HIP */

/* ══════════════════════════════════════════════════════════════════════ */
/*  Backend-specific runner handle types                                 */
/* ══════════════════════════════════════════════════════════════════════ */

/* Interpreter: linearized UOp array */
typedef struct {
  PolyUOp **lin;
  int n_lin;
} InterpHandle;

/* CUDA: compiled program handle */
#ifdef POLY_HAS_CUDA
typedef struct {
  PolyCudaProgram *prog;
} CudaRunnerHandle;
#endif

/* HIP: compiled program handle */
#ifdef POLY_HAS_HIP
typedef struct {
  PolyHipProgram *prog;
} HipRunnerHandle;
#endif

typedef struct {
  size_t nbytes;
  PolyDevice dst_device;
  PolyDevice src_device;
  PolyBuffer *dst_identity;
  PolyBuffer *src_identity;
} CopyRunnerHandle;

static int copy_execute_fn(void *self, void **args, int n_args) {
  PolyRunner *runner = (PolyRunner *)self;
  CopyRunnerHandle *ch = (CopyRunnerHandle *)runner->handle;
  if (!ch || !args || n_args < 2) return -1;

  bool dst_keyed_host = ch->dst_device == POLY_DEVICE_HOST && ch->dst_identity != NULL;
  bool src_keyed_host = ch->src_device == POLY_DEVICE_HOST && ch->src_identity != NULL;
  /* Browser WebGPU keeps imported HOST bytes in JS-owned TypedArrays. Those
   * schedule slots intentionally have ptr=NULL; the copy backend uses the
   * retained PolyBuffer identity as the JS host-buffer key instead. */
  if ((!args[0] && !dst_keyed_host) || (!args[1] && !src_keyed_host)) return -1;

  const PolyBackendDesc *dst_backend = poly_backend_get(ch->dst_device);
  const PolyBackendDesc *src_backend = poly_backend_get(ch->src_device);
  if ((dst_backend && poly_backend_ensure_open(ch->dst_device) != 0) ||
      (src_backend && poly_backend_ensure_open(ch->src_device) != 0))
    return -1;
  const PolyAllocator *dst_alloc = dst_backend ? dst_backend->get_allocator() : NULL;
  const PolyAllocator *src_alloc = src_backend ? src_backend->get_allocator() : NULL;
  if (!dst_alloc || !src_alloc) return -1;

  PolyBuffer dst = {
      .ptr = args[0],
      .nbytes = ch->nbytes,
      .device = ch->dst_device,
      .owned = false,
      .allocator = dst_alloc,
      .src = ch->dst_identity,
      .valid = true,
  };
  PolyBuffer src = {
      .ptr = args[1],
      .nbytes = ch->nbytes,
      .device = ch->src_device,
      .owned = false,
      .allocator = src_alloc,
      .src = ch->src_identity,
      .valid = true,
  };

  int rc = poly_buffer_copy(&dst, &src);
  if (poly_debug_at_least(7)) {
    fprintf(
        stderr,
        "[polygrad:copy_exec] dst=%s ptr=%p key=%p src=%s ptr=%p key=%p nbytes=%zu rc=%d\n",
        poly_device_name(ch->dst_device), args[0], (void *)ch->dst_identity,
        poly_device_name(ch->src_device), args[1], (void *)ch->src_identity, ch->nbytes, rc
    );
    fflush(stderr);
  }
  return rc;
}

static bool runner_copy_param_allows_null(const PolyRunner *runner, int param_index) {
  if (!runner || runner->kind != POLY_RUNNER_COPY || !runner->handle) return false;
  CopyRunnerHandle *ch = (CopyRunnerHandle *)runner->handle;
  /* NULL is only valid for browser HOST endpoints where the real data lives in
   * the frontend registry and the copy handle captured its PolyBuffer key. */
  return (param_index == 0 && ch->dst_device == POLY_DEVICE_HOST && ch->dst_identity) ||
         (param_index == 1 && ch->src_device == POLY_DEVICE_HOST && ch->src_identity);
}

static void copy_free_fn(void *self) {
  PolyRunner *runner = (PolyRunner *)self;
  free(runner ? runner->handle : NULL);
}

static int poly_lower_copy_call(
    PolyCtx *ctx,
    PolySchedule *schedule,
    PolyUOp *call,
    int call_index,
    PolyDevice device,
    PolyRunner *out
) {
  if (!schedule || !call || !out || !poly_call_is_copy(call) || poly_call_n_buffer_args(call) < 2)
    return -1;

  int dst_slot = poly_schedule_call_buffer_slot(schedule, call_index, 0);
  int src_slot = poly_schedule_call_buffer_slot(schedule, call_index, 1);
  if (dst_slot < 0 || dst_slot >= schedule->template->n_buf_slots || src_slot < 0 ||
      src_slot >= schedule->template->n_buf_slots)
    return -1;

  PolyDevice placement_fallback =
      (schedule->run && schedule->run->device != POLY_DEVICE_AUTO) ? schedule->run->device : device;
  PolyDevice dst_device = schedule->template->buf_slots[dst_slot].is_intermediate
                              ? poly_intermediate_slot_runtime_device(ctx, schedule, dst_slot, placement_fallback)
                              : poly_schedule_slot_target_device(ctx, schedule, dst_slot, device);
  PolyDevice src_device = POLY_DEVICE_AUTO;
  PolyBuffer *src_buf = poly_buffer_get(ctx, schedule->template->buf_slots[src_slot].buf_uop);
  if (schedule->template->buf_slots[src_slot].is_intermediate) {
    src_device = poly_intermediate_slot_runtime_device(ctx, schedule, src_slot, placement_fallback);
  } else if (src_buf && src_buf->device != POLY_DEVICE_AUTO) {
    /* COPY is directional. Arg 1 is the source, so lower the runner for the
     * source's actual current residency. Placement of the destination or a
     * later compute consumer must not turn a HOST/WASM source pointer into a
     * fake GPU handle. */
    src_device = src_buf->device;
  } else {
    src_device = poly_schedule_slot_target_device(ctx, schedule, src_slot, device);
  }
  if (dst_device == POLY_DEVICE_AUTO) dst_device = device;
  if (src_device == POLY_DEVICE_AUTO) src_device = device;

  size_t dst_nbytes = (size_t)schedule->template->buf_slots[dst_slot].nbytes;
  size_t src_nbytes = (size_t)schedule->template->buf_slots[src_slot].nbytes;
  if (dst_nbytes == 0 || src_nbytes == 0 || dst_nbytes != src_nbytes) return -1;

  CopyRunnerHandle *ch = malloc(sizeof(CopyRunnerHandle));
  if (!ch) return -1;
  ch->nbytes = dst_nbytes;
  ch->dst_device = dst_device;
  ch->src_device = src_device;
  PolyBuffer *dst_buf = poly_buffer_get(ctx, schedule->template->buf_slots[dst_slot].buf_uop);
  ch->dst_identity = (dst_buf && dst_buf->device == POLY_DEVICE_HOST) ? dst_buf : NULL;
  ch->src_identity = (src_buf && src_buf->device == POLY_DEVICE_HOST) ? src_buf : NULL;
  if (poly_debug_at_least(7)) {
    fprintf(
        stderr,
        "[polygrad:copy_lower] call=%d dst_slot=%d src_slot=%d dst_dev=%s src_dev=%s "
        "dst_uop=%p src_uop=%p dst_cur=%s/%p src_cur=%s/%p\n",
        call_index, dst_slot, src_slot, poly_device_name(dst_device), poly_device_name(src_device),
        (void *)schedule->template->buf_slots[dst_slot].buf_uop,
        (void *)schedule->template->buf_slots[src_slot].buf_uop,
        dst_buf ? poly_device_name(dst_buf->device) : "none", dst_buf ? dst_buf->ptr : NULL,
        src_buf ? poly_device_name(src_buf->device) : "none", src_buf ? src_buf->ptr : NULL
    );
    fflush(stderr);
  }

  out->kind = POLY_RUNNER_COPY;
  out->handle = ch;
  out->handle_size = (int)sizeof(*ch);
  out->execute = copy_execute_fn;
  out->free_handle = copy_free_fn;
  return 0;
}

static int poly_bind_runner_param_slots(
    PolyCtx *ctx,
    const PolySchedule *schedule,
    PolyUOp *call,
    int call_index,
    PolyRunner *runner,
    PolyDevice device
) {
  /* Backend lowering produces a reusable program/runtime object. The current
   * schedule still owns the PARAM -> buffer-slot mapping, so cached programs
   * can run against fresh buffers just like tinygrad resolves cached LINEAR
   * calls back to the current buffer inputs. */
  if (!poly_call_is_copy(call) && device == POLY_DEVICE_WEBGPU)
    return webgpu_fill_param_slots(ctx, schedule, call, call_index, runner);

  int n_params = poly_call_n_buffer_args(call);
  runner->n_params = n_params;
  if (n_params <= 0) return 0;
  runner->param_to_slot = malloc((size_t)n_params * sizeof(int));
  if (!runner->param_to_slot) return -1;
  for (int i = 0; i < n_params; i++) {
    runner->param_to_slot[i] = poly_schedule_call_buffer_slot(schedule, call_index, i);
    if (runner->param_to_slot[i] < 0) return -1;
  }
  return 0;
}

static const char *poly_program_arg_name(PolyUOp *program) {
  if (!program) return "test";
  if (program->arg.kind == POLY_ARG_PROGRAM_INFO && program->arg.program_info &&
      program->arg.program_info->name)
    return program->arg.program_info->name;
  if (program->arg.kind != POLY_ARG_STRING || !program->arg.str) return "test";
  return program->arg.str;
}

#ifdef POLY_HAS_X86
static PolyUOp *poly_prepare_x86_program_for_backend(
    PolyCtx *ctx,
    PolyUOp *call,
    PolyDevice device,
    uint32_t env_stamp
);
#endif

static PolyUOp *poly_prepare_program_for_backend(
    PolyCtx *ctx,
    PolyUOp *call,
    PolyDevice device,
    uint32_t env_stamp
) {
  if (!ctx || !call || call->op != POLY_OP_CALL) return NULL;
#ifdef POLY_HAS_X86
  if (device == POLY_DEVICE_X86)
    return poly_prepare_x86_program_for_backend(ctx, call, device, env_stamp);
#endif
  const PolyBackendDesc *backend = poly_backend_get(device);
  if (!backend || !backend->rewrite_program) return poly_call_raw_body(call);

  PolyUOp *program = poly_call_raw_body(call);
  PolyUOp *body = poly_program_kernel_body(program);
  if (!program || !body) return NULL;

  PolyToProgramCacheEntry key = {
      .program = program,
      .device = device,
      .env_stamp = env_stamp,
      .prepared_program = NULL,
  };
  uint32_t hash = poly_program_cache_hash(program, device, env_stamp);
  PolyToProgramCacheEntry *entry =
      (poly_program_cache_enabled() && ctx->to_program_cache)
          ? poly_map_get(ctx->to_program_cache, hash, &key, poly_to_program_cache_eq)
          : NULL;
  if (entry) return entry->prepared_program;

  PolyUOp *rewritten = backend->rewrite_program(ctx, body);
  if (!rewritten) return NULL;

  PolyUOp *prepared =
      poly_program_from_call_body(ctx, call, rewritten, poly_program_arg_name(program), device);
  if (!prepared) return NULL;
  prepared = poly_program_attach_linear(ctx, prepared);
  if (!prepared) return NULL;
  prepared = poly_program_ensure_source(ctx, backend, prepared, device);
  if (!prepared) return NULL;

  if (poly_program_cache_enabled() && ctx->to_program_cache) {
    entry = poly_arena_alloc(ctx->arena, sizeof(*entry), _Alignof(PolyToProgramCacheEntry));
    if (entry) {
      entry->program = program;
      entry->device = device;
      entry->env_stamp = env_stamp;
      entry->prepared_program = prepared;
      poly_map_set(ctx->to_program_cache, hash, entry, entry, poly_to_program_cache_eq);
    }
  }

  return prepared;
}

static PolyUOp *poly_program_ensure_source(
    PolyCtx *ctx,
    const PolyBackendDesc *backend,
    PolyUOp *program,
    PolyDevice device
) {
  if (!ctx || !backend || !program || program->op != POLY_OP_PROGRAM) return NULL;
  if (!backend->render_source || poly_program_source(program)) return program;

  char fn_name[64];
  stable_kernel_fn_name(ctx, fn_name, sizeof(fn_name), device, program);
  char *source = backend->render_source(ctx, program, fn_name);
  if (!source) return NULL;
  g_program_source_render_count++;
  PolyUOp *with_source = poly_program_attach_source(ctx, program, source);
  free(source);
  return with_source;
}

PolyUOp *poly_schedule_call_to_program(
    PolyCtx *ctx,
    PolySchedule *schedule,
    int call_index,
    PolyDevice device
) {
  if (!ctx || !schedule || call_index < 0 || call_index >= schedule->template->n_calls) return NULL;
  uint32_t env_stamp = poly_schedule_lower_env_stamp();
  return poly_prepare_program_for_backend(ctx, poly_schedule_call(schedule, call_index), device, env_stamp);
}

static int poly_lower_compute_call_cached(
    PolyCtx *ctx,
    PolyUOp *call,
    PolyDevice device,
    uint32_t env_stamp,
    PolyRunner *out,
    PolyRuntimeCacheEntry **runtime_entry_out
) {
  if (runtime_entry_out) *runtime_entry_out = NULL;
  const PolyBackendDesc *backend = poly_backend_get(device);
  if (!backend || !backend->lower_item) return -1;
  if (poly_backend_ensure_open(device) != 0) return -1;

  PolyUOp *program = poly_prepare_program_for_backend(ctx, call, device, env_stamp);
  PolyUOp *body = poly_program_kernel_body(program);
  if (!body || !poly_validate_kernel_graph(ctx, body)) return -2;

  PolyRuntimeCacheMapEntry key = {
      .program = program,
      .device = device,
      .env_stamp = env_stamp,
  };
  uint32_t hash = poly_program_cache_hash(program, device, env_stamp);
  PolyRuntimeCacheMapEntry *entry =
      (poly_program_cache_enabled() && ctx && ctx->runtime_cache)
          ? poly_map_get(ctx->runtime_cache, hash, &key, poly_runtime_cache_eq)
          : NULL;

  if (!entry) {
    char fn_name[64];
    stable_kernel_fn_name(ctx, fn_name, sizeof(fn_name), device, program);

    PolyRunner lowered = {0};
    if (backend->lower_item(ctx, program, fn_name, &lowered) != 0) return -1;

    if (poly_program_cache_enabled() && ctx && ctx->runtime_cache) {
      entry = calloc(1, sizeof(*entry));
      PolyRuntimeCacheEntry *runtime_entry =
          entry ? poly_runtime_cache_entry_new(ctx, program, device, env_stamp, &lowered) : NULL;
      if (entry && runtime_entry) {
        entry->program = program;
        entry->device = device;
        entry->env_stamp = env_stamp;
        entry->runtime_program = poly_runtime_cache_entry_retain(runtime_entry);
        entry->runtime_program->in_cache = true;
        /* Cached backend runners must not carry call-specific slot metadata.
         * Slot metadata is rebuilt below for each fresh PolySchedule. */
        runtime_entry->runner.param_to_slot = NULL;
        runtime_entry->runner.n_params = 0;
        runtime_entry->runner.var_indices = NULL;
        runtime_entry->runner.n_vars = 0;
        poly_map_set(ctx->runtime_cache, hash, entry, entry, poly_runtime_cache_eq);
        if (runtime_entry_out) *runtime_entry_out = runtime_entry;
        *out = runtime_entry->runner;
        out->param_to_slot = NULL;
        out->n_params = 0;
        out->var_indices = NULL;
        out->n_vars = 0;
        out->borrowed_handle = true;
        poly_runner_apply_program_launch_info(ctx, program, out);
        return 0;
      } else {
        free(entry);
        poly_runtime_cache_entry_release(runtime_entry);
        *out = lowered;
        poly_runner_apply_program_launch_info(ctx, program, out);
        return 0;
      }
    } else {
      *out = lowered;
      poly_runner_apply_program_launch_info(ctx, program, out);
      return 0;
    }
  }

  if (!entry || !entry->runtime_program) return -1;
  if (runtime_entry_out)
    *runtime_entry_out = poly_runtime_cache_entry_retain(entry->runtime_program);
  *out = entry->runtime_program->runner;
  out->param_to_slot = NULL;
  out->n_params = 0;
  out->var_indices = NULL;
  out->n_vars = 0;
  out->borrowed_handle = true;
  poly_runner_apply_program_launch_info(ctx, program, out);
  return 0;
}

/* ══════════════════════════════════════════════════════════════════════ */
/*  Backend implementations (lower_item / execute / free_runner)         */
/* ══════════════════════════════════════════════════════════════════════ */

/* CPU backend */

#ifndef __EMSCRIPTEN__

static int cpu_execute_fn(void *self, void **args, int n_args);
static void cpu_free_fn(void *self);

static PolyRewriteOpts cpu_schedule_rewrite_opts(void) {
  return (PolyRewriteOpts){
      .optimize = true,
      .devectorize = 1,
      .caps = poly_c_renderer_caps(),
      .device = POLY_DEVICE_CPU,
      .opt_policy = POLY_OPT_HEURISTIC,
      .extra_matcher = poly_pm_c_renderer_extra(),
  };
}

static PolyUOp *cpu_rewrite_program(PolyCtx *ctx, PolyUOp *sink) {
  return poly_full_rewrite_to_sink_ex(ctx, sink, cpu_schedule_rewrite_opts());
}

static char *cpu_render_source_impl(
    PolyCtx *ctx,
    PolyUOp *program,
    const char *fn_name
) {
  PolyUOp *scheduled_root = poly_program_kernel_body(program);
  if (!scheduled_root) return NULL;
  int n_lin;
  bool lin_owned = false;
  PolyUOp **lin = poly_program_linear_uops(program, &n_lin);
  if (!lin) {
    lin = poly_linearize_rewritten(ctx, scheduled_root, &n_lin);
    lin_owned = true;
  }
  if (!lin) return NULL;
  if (poly_debug_at_least(3)) {
    int n_weak = 0;
    for (int i = 0; i < n_lin; i++)
      if (lin[i] && poly_dtype_is_index(lin[i]->dtype)) n_weak++;
    fprintf(stderr, "[polygrad:cpu_lower] fn=%s linear=%d weakint=%d\n", fn_name, n_lin, n_weak);
    fflush(stderr);
  }

  char *src = poly_render_c(lin, n_lin, fn_name);
  if (lin_owned) free(lin);
  return src;
}

static char *cpu_render_source(PolyCtx *ctx, PolyUOp *program, const char *fn_name) {
  return cpu_render_source_impl(ctx, program, fn_name);
}

static int cpu_lower_item_impl(
    PolyCtx *ctx,
    PolyUOp *program,
    const char *fn_name,
    PolyRunner *out
) {
  PolyUOp *scheduled_root = poly_program_kernel_body(program);
  if (!scheduled_root) return -1;
  int n_lin;
  bool lin_owned = false;
  PolyUOp **lin = poly_program_linear_uops(program, &n_lin);
  if (!lin) {
    lin = poly_linearize_rewritten(ctx, scheduled_root, &n_lin);
    lin_owned = true;
  }
  if (!lin) return -1;
  int cpu_threads = 1;
  for (int j = 0; j < n_lin; j++) {
    PolyUOp *u = lin[j];
    if (!u || u->op != POLY_OP_DEFINE_VAR || u->arg.kind != POLY_ARG_DEFINE_VAR ||
        !u->arg.define_var.name || strcmp(u->arg.define_var.name, "core_id") != 0)
      continue;
    int64_t n = u->arg.define_var.max_val + 1;
    if (n > 1 && n <= INT32_MAX) cpu_threads = (int)n;
  }

  const char *src = poly_program_source_text(program);
  char *src_owned = NULL;
  if (!src) {
    src_owned = cpu_render_source_impl(ctx, program, fn_name);
    src = src_owned;
  }
  if (!src) {
    if (lin_owned) free(lin);
    return -1;
  }

  if (poly_dump_kernels_enabled())
    fprintf(stderr, "=== LOWER KERNEL %s ===\n%s\n=== END ===\n", fn_name, src);

  PolyProgram *prog = poly_compile_c(src, fn_name);
  if (!prog) {
    fprintf(stderr, "=== FAILED LOWER KERNEL %s ===\n%s\n=== END ===\n", fn_name, src);
    free(src_owned);
    if (lin_owned) free(lin);
    return -1;
  }
  free(src_owned);
  if (lin_owned) free(lin);

  out->kind = POLY_RUNNER_COMPILED;
  out->handle = prog;
  size_t prog_size = poly_program_estimated_size(prog);
  out->handle_size = prog_size > (size_t)INT_MAX ? INT_MAX : (int)prog_size;
  out->grid[0] = cpu_threads;
  out->grid[1] = 1;
  out->grid[2] = 1;
  out->block[0] = 1;
  out->block[1] = 1;
  out->block[2] = 1;
  out->execute = cpu_execute_fn;
  out->free_handle = cpu_free_fn;
  return 0;
}

static int cpu_lower_item(
    PolyCtx *ctx,
    PolyUOp *program,
    const char *fn_name,
    PolyRunner *out
) {
  return cpu_lower_item_impl(ctx, program, fn_name, out);
}

static int cpu_execute_fn(void *self, void **args, int n_args) {
  PolyRunner *runner = (PolyRunner *)self;
  int threads = runner->grid[0] > 1 ? runner->grid[0] : 1;
  if (threads > 1)
    poly_program_call_threaded((PolyProgram *)runner->handle, args, n_args, threads);
  else
    poly_program_call((PolyProgram *)runner->handle, args, n_args);
  return 0;
}
static int cpu_execute(PolyRunner *runner, void **args, int n_args) {
  return cpu_execute_fn(runner, args, n_args);
}

static void cpu_free_fn(void *self) {
  PolyRunner *runner = (PolyRunner *)self;
  if (runner->handle) poly_program_destroy((PolyProgram *)runner->handle);
}
static void cpu_free_runner(PolyRunner *runner) {
  cpu_free_fn(runner);
}

static const PolyAllocator *cpu_get_allocator(void) {
  return &POLY_CPU_ALLOCATOR;
}

#endif /* !__EMSCRIPTEN__ */

static const PolyAllocator *host_get_allocator(void) {
  return &POLY_HOST_ALLOCATOR;
}

/* Interpreter backend */

static int interp_lower_item(
    PolyCtx *ctx,
    PolyUOp *program,
    const char *fn_name,
    PolyRunner *out
) {
  (void)fn_name;
  PolyUOp *scheduled_root = poly_program_kernel_body(program);
  if (!scheduled_root) return -1;
  int n_lin;
  PolyUOp **lin = poly_program_linear_uops(program, &n_lin);
  if (lin) {
    PolyUOp **copy = n_lin > 0 ? malloc((size_t)n_lin * sizeof(PolyUOp *)) : NULL;
    if (n_lin > 0 && !copy) return -1;
    if (n_lin > 0) memcpy(copy, lin, (size_t)n_lin * sizeof(PolyUOp *));
    lin = copy;
  } else {
    lin = poly_linearize(ctx, scheduled_root, &n_lin);
  }
  if (!lin) return -1;

  InterpHandle *ih = malloc(sizeof(InterpHandle));
  if (!ih) {
    free(lin);
    return -1;
  }
  ih->lin = lin;
  ih->n_lin = n_lin;

  out->kind = POLY_RUNNER_INTERP;
  out->handle = ih;
  out->handle_size = (int)(sizeof(*ih) + (size_t)n_lin * sizeof(PolyUOp *));
  return 0;
}

static int interp_execute(PolyRunner *runner, void **args, int n_args) {
  InterpHandle *ih = (InterpHandle *)runner->handle;
  return poly_interp_eval(ih->lin, ih->n_lin, args, n_args);
}

static void interp_free_runner(PolyRunner *runner) {
  if (runner->handle) {
    InterpHandle *ih = (InterpHandle *)runner->handle;
    free(ih->lin);
    free(ih);
  }
}

static const PolyAllocator *interp_get_allocator(void) {
  return &POLY_CPU_ALLOCATOR;
}

/* CUDA backend */

#ifdef POLY_HAS_CUDA

static void cuda_extract_dims(
    PolyCtx *ctx,
    PolyUOp **lin,
    int n_lin,
    int grid[3],
    int local[3],
    PolyUOp *grid_exprs[3],
    PolyUOp *block_exprs[3],
    int *launch_bounds
) {
  grid[0] = 1;
  grid[1] = 1;
  grid[2] = 1;
  local[0] = 1;
  local[1] = 1;
  local[2] = 1;
  grid_exprs[0] = grid_exprs[1] = grid_exprs[2] = NULL;
  block_exprs[0] = block_exprs[1] = block_exprs[2] = NULL;

  for (int j = 0; j < n_lin; j++) {
    if (lin[j]->op != POLY_OP_SPECIAL || lin[j]->n_src <= 0) continue;
    const char *sn = lin[j]->arg.str;
    if (!sn || !sn[0]) continue;
    int slen = (int)strlen(sn);
    int dim_idx = (slen > 0) ? sn[slen - 1] - '0' : 0;
    if (dim_idx < 0 || dim_idx > 2) dim_idx = 0;
    PolyUOp *bound_expr = lin[j]->src[0];
    int bound = poly_launch_dim_upper_bound(ctx, bound_expr);
    if (sn[0] == 'l') {
      local[dim_idx] = bound;
      block_exprs[dim_idx] = bound_expr;
    } else {
      grid[dim_idx] = bound;
      grid_exprs[dim_idx] = bound_expr;
    }
  }

  *launch_bounds = local[0] * local[1] * local[2];
  if (*launch_bounds <= 0) *launch_bounds = 1;
}

static char *cuda_render_source(PolyCtx *ctx, PolyUOp *program, const char *fn_name) {
  PolyUOp *scheduled_root = poly_program_kernel_body(program);
  if (!scheduled_root) return NULL;
  int n_lin;
  bool lin_owned = false;
  PolyUOp **lin = poly_program_linear_uops(program, &n_lin);
  if (!lin) {
    lin = poly_linearize_rewritten(ctx, scheduled_root, &n_lin);
    lin_owned = true;
  }
  if (!lin) return NULL;

  int grid[3], local[3], launch_bounds = 1;
  PolyUOp *grid_exprs[3], *block_exprs[3];
  cuda_extract_dims(ctx, lin, n_lin, grid, local, grid_exprs, block_exprs, &launch_bounds);

  char *src = poly_render_cuda(lin, n_lin, fn_name, launch_bounds);
  if (lin_owned) free(lin);
  return src;
}

static int cuda_lower_item(
    PolyCtx *ctx,
    PolyUOp *program,
    const char *fn_name,
    PolyRunner *out
) {
  PolyUOp *scheduled_root = poly_program_kernel_body(program);
  if (!scheduled_root) return -1;
  int n_lin;
  bool lin_owned = false;
  PolyUOp **lin = poly_program_linear_uops(program, &n_lin);
  if (!lin) {
    lin = poly_linearize_rewritten(ctx, scheduled_root, &n_lin);
    lin_owned = true;
  }
  if (!lin) return -1;

  int grid[3], local[3], launch_bounds = 1;
  PolyUOp *grid_exprs[3], *block_exprs[3];
  cuda_extract_dims(ctx, lin, n_lin, grid, local, grid_exprs, block_exprs, &launch_bounds);

  const char *src = poly_program_source_text(program);
  char *src_owned = NULL;
  if (!src) {
    src_owned = poly_render_cuda(lin, n_lin, fn_name, launch_bounds);
    src = src_owned;
  }
  if (lin_owned) free(lin);
  if (!src) return -1;

  if (poly_dump_kernels_enabled())
    fprintf(stderr, "=== CUDA KERNEL %s ===\n%s\n=== END ===\n", fn_name, src);

  PolyCudaProgram *prog = poly_compile_cuda(src, fn_name);
  if (!prog) {
    fprintf(stderr, "=== FAILED CUDA KERNEL %s ===\n%s\n=== END ===\n", fn_name, src);
    free(src_owned);
    return -1;
  }
  free(src_owned);

  CudaRunnerHandle *ch = malloc(sizeof(CudaRunnerHandle));
  if (!ch) {
    poly_cuda_program_destroy(prog);
    return -1;
  }
  ch->prog = prog;

  out->kind = POLY_RUNNER_COMPILED;
  out->handle = ch;
  out->handle_size = (int)sizeof(*ch);
  out->grid[0] = grid[0];
  out->grid[1] = grid[1];
  out->grid[2] = grid[2];
  out->block[0] = local[0];
  out->block[1] = local[1];
  out->block[2] = local[2];
  for (int i = 0; i < 3; i++) {
    out->grid_exprs[i] = grid_exprs[i];
    out->block_exprs[i] = block_exprs[i];
  }
  return 0;
}

static int cuda_execute(PolyRunner *runner, void **args, int n_args) {
  CudaRunnerHandle *ch = (CudaRunnerHandle *)runner->handle;

  /* cuLaunchKernel kernelParams: each element points TO the arg value.
   * Buffer params (0..n_params-1): args[i] is a device ptr; cast to CUdeviceptr.
   * Scalar params (n_params..): args[i] is already &int_val; use directly. */
  unsigned long long *dptrs = malloc((size_t)n_args * sizeof(unsigned long long));
  void **cuda_args = malloc((size_t)n_args * sizeof(void *));
  if (!dptrs || !cuda_args) {
    free(dptrs);
    free(cuda_args);
    return -1;
  }
  for (int i = 0; i < runner->n_params; i++) {
    dptrs[i] = (unsigned long long)(uintptr_t)args[i];
    cuda_args[i] = &dptrs[i];
  }
  for (int i = runner->n_params; i < n_args; i++) {
    cuda_args[i] = args[i];
  }

  int ret = poly_cuda_launch(
      ch->prog, cuda_args, n_args, runner->grid[0], runner->grid[1], runner->grid[2],
      runner->block[0], runner->block[1], runner->block[2]
  );
  if (ret == 0) ret = poly_cuda_sync();

  free(dptrs);
  free(cuda_args);
  return ret;
}

static void cuda_free_runner(PolyRunner *runner) {
  if (runner->handle) {
    CudaRunnerHandle *ch = (CudaRunnerHandle *)runner->handle;
    poly_cuda_program_destroy(ch->prog);
    free(ch);
  }
}

static const PolyAllocator *cuda_get_allocator(void) {
  return &POLY_CUDA_ALLOCATOR;
}

#endif /* POLY_HAS_CUDA */

/* HIP backend */

#ifdef POLY_HAS_HIP

static char *hip_render_source(PolyCtx *ctx, PolyUOp *program, const char *fn_name) {
  PolyUOp *scheduled_root = poly_program_kernel_body(program);
  if (!scheduled_root) return NULL;
  int n_lin;
  bool lin_owned = false;
  PolyUOp **lin = poly_program_linear_uops(program, &n_lin);
  if (!lin) {
    lin = poly_linearize_rewritten(ctx, scheduled_root, &n_lin);
    lin_owned = true;
  }
  if (!lin) return NULL;

  int grid_size = 0, local_size = 0;
  for (int j = 0; j < n_lin; j++) {
    if (lin[j]->op == POLY_OP_SPECIAL && lin[j]->n_src > 0 && lin[j]->src[0]->op == POLY_OP_CONST) {
      const char *sn = lin[j]->arg.str;
      if (sn && sn[0] == 'l')
        local_size = (int)lin[j]->src[0]->arg.i;
      else
        grid_size = (int)lin[j]->src[0]->arg.i;
    }
  }
  (void)grid_size;

  int block_size = local_size > 0 ? local_size : 256;
  char *src = poly_render_hip(lin, n_lin, fn_name, block_size);
  if (lin_owned) free(lin);
  return src;
}

static int hip_lower_item(
    PolyCtx *ctx,
    PolyUOp *program,
    const char *fn_name,
    PolyRunner *out
) {
  PolyUOp *scheduled_root = poly_program_kernel_body(program);
  if (!scheduled_root) return -1;
  int n_lin;
  bool lin_owned = false;
  PolyUOp **lin = poly_program_linear_uops(program, &n_lin);
  if (!lin) {
    lin = poly_linearize_rewritten(ctx, scheduled_root, &n_lin);
    lin_owned = true;
  }
  if (!lin) return -1;

  /* Extract grid/block from SPECIAL ops */
  int grid_size = 0, local_size = 0;
  for (int j = 0; j < n_lin; j++) {
    if (lin[j]->op == POLY_OP_SPECIAL && lin[j]->n_src > 0 && lin[j]->src[0]->op == POLY_OP_CONST) {
      const char *sn = lin[j]->arg.str;
      if (sn && sn[0] == 'l')
        local_size = (int)lin[j]->src[0]->arg.i;
      else
        grid_size = (int)lin[j]->src[0]->arg.i;
    }
  }

  int block_size = local_size > 0 ? local_size : 256;

  const char *src = poly_program_source_text(program);
  char *src_owned = NULL;
  if (!src) {
    src_owned = poly_render_hip(lin, n_lin, fn_name, block_size);
    src = src_owned;
  }
  if (lin_owned) free(lin);
  if (!src) return -1;

  if (poly_dump_kernels_enabled())
    fprintf(stderr, "=== HIP KERNEL %s ===\n%s\n=== END ===\n", fn_name, src);

  PolyHipProgram *prog = poly_compile_hip(src, fn_name);
  if (!prog) {
    fprintf(stderr, "=== FAILED HIP KERNEL %s ===\n%s\n=== END ===\n", fn_name, src);
    free(src_owned);
    return -1;
  }
  free(src_owned);

  /* Compute grid dimensions */
  int gx;
  if (local_size > 0 && grid_size > 0)
    gx = grid_size;
  else if (local_size > 0)
    gx = 1;
  else if (grid_size > 0)
    gx = (grid_size + block_size - 1) / block_size;
  else
    gx = 1;

  HipRunnerHandle *hh = malloc(sizeof(HipRunnerHandle));
  if (!hh) {
    poly_hip_program_destroy(prog);
    return -1;
  }
  hh->prog = prog;

  out->kind = POLY_RUNNER_COMPILED;
  out->handle = hh;
  out->handle_size = (int)sizeof(*hh);
  out->grid[0] = gx;
  out->grid[1] = 1;
  out->grid[2] = 1;
  out->block[0] = block_size;
  out->block[1] = 1;
  out->block[2] = 1;
  return 0;
}

static int hip_execute(PolyRunner *runner, void **args, int n_args) {
  HipRunnerHandle *hh = (HipRunnerHandle *)runner->handle;

  /* hipModuleLaunchKernel kernelParams: each element points TO the arg value.
   * Buffer params (0..n_params-1): args[i] is the device ptr, so &args[i] works.
   * Scalar params (n_params..): args[i] is already &int_val, use it directly. */
  void **hip_args = malloc((size_t)n_args * sizeof(void *));
  if (!hip_args) return -1;
  for (int i = 0; i < runner->n_params; i++)
    hip_args[i] = &args[i];
  for (int i = runner->n_params; i < n_args; i++)
    hip_args[i] = args[i];

  int ret = poly_hip_launch(
      hh->prog, hip_args, n_args, runner->grid[0], runner->grid[1], runner->grid[2],
      runner->block[0], runner->block[1], runner->block[2]
  );
  if (ret == 0) ret = poly_hip_sync();

  free(hip_args);
  return ret;
}

static void hip_free_runner(PolyRunner *runner) {
  if (runner->handle) {
    HipRunnerHandle *hh = (HipRunnerHandle *)runner->handle;
    poly_hip_program_destroy(hh->prog);
    free(hh);
  }
}

static const PolyAllocator *hip_get_allocator(void) {
  return &POLY_HIP_ALLOCATOR;
}

#endif /* POLY_HAS_HIP */

/* tinygrad-style x86 ISA backend */

#ifdef POLY_HAS_X86

static int x86_execute_fn(void *self, void **args, int n_args) {
  PolyRunner *runner = (PolyRunner *)self;
  int threads = runner->grid[0] > 1 ? runner->grid[0] : 1;
  if (threads > 1)
    return poly_x86_program_call_threaded((PolyX86Program *)runner->handle, args, n_args, threads);
  return poly_x86_program_call((PolyX86Program *)runner->handle, args, n_args);
}

static void x86_free_fn(void *self) {
  PolyRunner *runner = (PolyRunner *)self;
  if (runner->handle) poly_x86_program_destroy((PolyX86Program *)runner->handle);
}

static bool x86_can_handle(PolyCtx *ctx, PolyUOp *root) {
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, root, &n_topo);
  if (!topo) return false;
  bool ok = true;
  for (int ti = 0; ti < n_topo; ti++) {
    PolyUOp *u = topo[ti];
    if (!u) continue;

    PolyDType dt = u->dtype;
    PolyDType scalar = poly_dtype_scalar(dt);
    if (!dt.is_ptr && !poly_dtype_eq(dt, POLY_VOID) && poly_dtype_is_float(dt) &&
        scalar.bitsize != 16 && scalar.bitsize != 32 && scalar.bitsize != 64) {
      if (poly_debug_at_least(4)) {
        char *s = poly_uop_str(u);
        fprintf(stderr, "x86 can_handle: unsupported float dtype bits=%d uop=%s\n",
                scalar.bitsize, s ? s : "<uop>");
        free(s);
      }
      ok = false;
      break;
    }
    if (!dt.is_ptr && !poly_dtype_eq(dt, POLY_VOID) && !poly_dtype_is_float(dt) &&
        !poly_dtype_is_bool(scalar) && !poly_dtype_is_index(scalar) &&
        (!poly_dtype_is_int(scalar) || scalar.bitsize > 64)) {
      if (poly_debug_at_least(4)) {
        char *s = poly_uop_str(u);
        fprintf(stderr, "x86 can_handle: unsupported int dtype bits=%d uop=%s\n",
                scalar.bitsize, s ? s : "<uop>");
        free(s);
      }
      ok = false;
      break;
    }
    if (u->op == POLY_OP_THREEFRY) {
      if (poly_debug_at_least(4)) fprintf(stderr, "x86 can_handle: unsupported THREEFRY\n");
      ok = false;
      break;
    }
  }
  poly_toposort_free(topo);
  return ok;
}

static char *poly_hex_from_bytes(const uint8_t *bytes, int n_bytes) {
  if (!bytes || n_bytes <= 0) return NULL;
  char *hex = malloc((size_t)n_bytes * 2 + 1);
  if (!hex) return NULL;
  static const char digits[] = "0123456789abcdef";
  for (int i = 0; i < n_bytes; i++) {
    hex[2 * i] = digits[bytes[i] >> 4];
    hex[2 * i + 1] = digits[bytes[i] & 0x0f];
  }
  hex[(size_t)n_bytes * 2] = '\0';
  return hex;
}

static PolyUOp *poly_prepare_x86_program_for_backend(
    PolyCtx *ctx,
    PolyUOp *call,
    PolyDevice device,
    uint32_t env_stamp
) {
  if (!ctx || !call || call->op != POLY_OP_CALL) return NULL;
  PolyUOp *raw = poly_call_raw_body(call);
  if (!raw) return NULL;

  PolyToProgramCacheEntry key = {
      .program = raw,
      .device = device,
      .env_stamp = env_stamp,
      .prepared_program = NULL,
  };
  uint32_t hash = poly_program_cache_hash(raw, device, env_stamp);
  PolyToProgramCacheEntry *entry =
      (poly_program_cache_enabled() && ctx->to_program_cache)
          ? poly_map_get(ctx->to_program_cache, hash, &key, poly_to_program_cache_eq)
          : NULL;
  if (entry) return entry->prepared_program;

  PolyUOp *body = poly_program_body(raw);
  if (!body) return NULL;
  PolyUOp *rewritten = poly_rewrite_x86(ctx, body);
  if (!rewritten) return NULL;
  if (!x86_can_handle(ctx, rewritten)) return NULL;

  PolyUOp *base =
      poly_program_from_call_body(ctx, call, rewritten, poly_program_arg_name(raw), device);
  if (!base) return NULL;

  int n_lin = 0;
  PolyUOp **lin = poly_linearize_x86_rewritten(ctx, rewritten, &n_lin);
  if (!lin) return NULL;

  PolyUOp *linear = poly_uop(ctx, POLY_OP_LINEAR, POLY_VOID, lin, n_lin, poly_arg_none());
  if (!linear) {
    free(lin);
    return NULL;
  }
  PolyUOp *src3[3] = {base->src[0], base->src[1], linear};
  PolyUOp *prepared = poly_uop(ctx, POLY_OP_PROGRAM, POLY_VOID, src3, 3, base->arg);
  if (!prepared) {
    free(lin);
    return NULL;
  }

  int n_code = 0;
  uint8_t *code = poly_render_x86(lin, n_lin, &n_code);
  free(lin);
  if (!code || n_code <= 0) {
    free(code);
    return NULL;
  }

  char *source = poly_hex_from_bytes(code, n_code);
  if (!source) {
    free(code);
    return NULL;
  }
  g_program_source_render_count++;
  prepared = poly_program_attach_source(ctx, prepared, source);
  free(source);
  if (!prepared) {
    free(code);
    return NULL;
  }
  prepared = poly_program_attach_binary(ctx, prepared, code, n_code);
  free(code);
  if (!prepared) return NULL;

  if (poly_program_cache_enabled() && ctx->to_program_cache) {
    entry = poly_arena_alloc(ctx->arena, sizeof(*entry), _Alignof(PolyToProgramCacheEntry));
    if (entry) {
      entry->program = raw;
      entry->device = device;
      entry->env_stamp = env_stamp;
      entry->prepared_program = prepared;
      poly_map_set(ctx->to_program_cache, hash, entry, entry, poly_to_program_cache_eq);
    }
  }

  return prepared;
}

static int x86_lower_item(
    PolyCtx *ctx,
    PolyUOp *program,
    const char *fn_name,
    PolyRunner *out
) {
  (void)ctx;
  (void)fn_name;
  PolyX86Program *prog = NULL;
  int code_size = 0;
  PolyUOp *binary = poly_program_binary(program);
  if (binary && binary->arg.kind == POLY_ARG_BYTES && binary->arg.bytes.data &&
      binary->arg.bytes.n > 0) {
    code_size = binary->arg.bytes.n;
    prog = poly_compile_x86(binary->arg.bytes.data, binary->arg.bytes.n);
  } else {
    const char *source = poly_program_source_text(program);
    if (source) {
      code_size = (int)(strlen(source) / 2);
      prog = poly_compile_x86_source(source);
    }
  }
  if (!prog) return -1;

  out->kind = POLY_RUNNER_COMPILED;
  out->handle = prog;
  out->handle_size = code_size;
  int n_lin = 0;
  PolyUOp **lin = poly_program_linear_uops(program, &n_lin);
  int threads = 1;
  for (int j = 0; j < n_lin; j++) {
    PolyUOp *u = lin[j];
    if (!u || u->op != POLY_OP_DEFINE_VAR || u->arg.kind != POLY_ARG_DEFINE_VAR ||
        !u->arg.define_var.name || strcmp(u->arg.define_var.name, "core_id") != 0)
      continue;
    int64_t n = u->arg.define_var.max_val + 1;
    if (n > 1 && n <= INT32_MAX) threads = (int)n;
  }
  out->grid[0] = threads;
  out->grid[1] = 1;
  out->grid[2] = 1;
  out->block[0] = 1;
  out->block[1] = 1;
  out->block[2] = 1;
  out->execute = x86_execute_fn;
  out->free_handle = x86_free_fn;
  return 0;
}

static int x86_execute(PolyRunner *runner, void **args, int n_args) {
  return runner->execute(runner, args, n_args);
}

static void x86_free_runner(PolyRunner *runner) {
  if (runner->free_handle) runner->free_handle(runner);
}

#endif /* POLY_HAS_X86 */

static int backend_noop_ensure_open(void) {
  return 0;
}

#ifdef POLY_HAS_CUDA
static int cuda_ensure_open(void) {
  return poly_cuda_init();
}
#endif

/* ══════════════════════════════════════════════════════════════════════ */
/*  Backend registry                                                     */
/* ══════════════════════════════════════════════════════════════════════ */

static const PolyBackendDesc BACKENDS[] = {
    [POLY_DEVICE_AUTO] = {NULL, POLY_DEVICE_AUTO, false, NULL, NULL, NULL, NULL, NULL, NULL, NULL},
    [POLY_DEVICE_HOST] =
        {"host", POLY_DEVICE_HOST, false, NULL, NULL, NULL, NULL, NULL, backend_noop_ensure_open,
         host_get_allocator},
#ifndef __EMSCRIPTEN__
    [POLY_DEVICE_CPU] =
        {"cpu", POLY_DEVICE_CPU, false, cpu_rewrite_program, cpu_render_source, cpu_lower_item,
         cpu_execute, cpu_free_runner, backend_noop_ensure_open, cpu_get_allocator},
#else
    [POLY_DEVICE_CPU] = {NULL, POLY_DEVICE_CPU, false, NULL, NULL, NULL, NULL, NULL, NULL, NULL},
#endif
    [POLY_DEVICE_INTERP] =
        {"interp", POLY_DEVICE_INTERP, false, NULL, NULL, interp_lower_item, interp_execute,
         interp_free_runner, backend_noop_ensure_open, interp_get_allocator},
#ifdef POLY_HAS_CUDA
    [POLY_DEVICE_CUDA] =
        {"cuda", POLY_DEVICE_CUDA, false, poly_rewrite_cuda, cuda_render_source, cuda_lower_item,
         cuda_execute, cuda_free_runner, cuda_ensure_open, cuda_get_allocator},
#else
    [POLY_DEVICE_CUDA] = {NULL, POLY_DEVICE_CUDA, false, NULL, NULL, NULL, NULL, NULL, NULL, NULL},
#endif
#ifdef __EMSCRIPTEN__
    [POLY_DEVICE_WASM] =
        {"wasm", POLY_DEVICE_WASM, true, poly_rewrite_wasm_env, NULL, poly_wasm_lower_item,
         poly_wasm_execute, poly_wasm_free_runner, backend_noop_ensure_open, poly_wasm_get_allocator},
#else
    [POLY_DEVICE_WASM] = {NULL, POLY_DEVICE_WASM, false, NULL, NULL, NULL, NULL, NULL, NULL, NULL},
#endif
#ifdef __EMSCRIPTEN__
    [POLY_DEVICE_WEBGPU] =
        {"webgpu", POLY_DEVICE_WEBGPU, true, poly_rewrite_webgpu, poly_webgpu_render_source,
         poly_webgpu_lower_item, poly_webgpu_execute, poly_webgpu_free_runner, backend_noop_ensure_open,
         poly_webgpu_get_allocator},
#else
    [POLY_DEVICE_WEBGPU] = {NULL, POLY_DEVICE_WEBGPU, false, NULL, NULL, NULL, NULL, NULL, NULL, NULL},
#endif
#ifdef POLY_HAS_X86
    [POLY_DEVICE_X86] =
        {"x86", POLY_DEVICE_X86, false, NULL, NULL, x86_lower_item, x86_execute,
         x86_free_runner, backend_noop_ensure_open, cpu_get_allocator},
#else
    [POLY_DEVICE_X86] = {NULL, POLY_DEVICE_X86, false, NULL, NULL, NULL, NULL, NULL, NULL, NULL},
#endif
#ifdef POLY_HAS_HIP
    [POLY_DEVICE_HIP] =
        {"hip", POLY_DEVICE_HIP, false, poly_rewrite_hip, hip_render_source, hip_lower_item,
         hip_execute, hip_free_runner, poly_hip_init, hip_get_allocator},
#else
    [POLY_DEVICE_HIP] = {NULL, POLY_DEVICE_HIP, false, NULL, NULL, NULL, NULL, NULL, NULL, NULL},
#endif
};

#define N_BACKENDS (sizeof(BACKENDS) / sizeof(BACKENDS[0]))

const PolyBackendDesc *poly_backend_get(PolyDevice device) {
  if (device < 0 || (size_t)device >= N_BACKENDS) return NULL;
  if (!BACKENDS[device].name) return NULL;
  return &BACKENDS[device];
}

int poly_backend_ensure_open(PolyDevice device) {
  const PolyBackendDesc *be = poly_backend_get(device);
  if (!be) return -1;
  return be->ensure_open ? be->ensure_open() : 0;
}

bool poly_device_is_host_addressable(PolyDevice device) {
  const PolyBackendDesc *be = poly_backend_get(device);
  return be && be->get_allocator()->host_addressable;
}

static PolyDevice poly_schedule_runtime_slot_allocated_device(
    const PolyScheduleTemplate *tpl,
    const PolyScheduleRuntime *run,
    int slot_idx,
    PolyDevice fallback_device
) {
  if (!tpl || !run || slot_idx < 0 || slot_idx >= tpl->n_buf_slots)
    return fallback_device;

  if (run->slot_views && slot_idx < run->n_slot_views && run->slot_views[slot_idx].ptr)
    return run->slot_views[slot_idx].device;

  const PolyScheduleBufSlot *slot = &tpl->buf_slots[slot_idx];
  if (slot->is_intermediate && slot->has_memory_parent) {
    int parent = slot->memory_parent_slot;
    if (parent >= 0 && parent < tpl->n_buf_slots)
      return poly_schedule_runtime_slot_allocated_device(tpl, run, parent, fallback_device);
  }

  if (run->slot_to_data && slot_idx < run->n_slot_to_data) {
    void *ptr = run->slot_to_data[slot_idx];
    for (int i = 0; ptr && i < run->n_intermediates; i++) {
      if (run->intermediates[i].ptr == ptr)
        return run->intermediates[i].device;
    }
  }

  PolyDevice dev = slot->device;
  if (dev == POLY_DEVICE_AUTO || dev == POLY_DEVICE_HOST) dev = fallback_device;
  return dev;
}

static int poly_schedule_runtime_fill_parent_views(
    const PolyScheduleTemplate *tpl,
    PolyScheduleRuntime *run,
    PolyDevice fallback_device
) {
  if (!tpl || !run || !run->slot_to_data) return -1;
  bool need_slot_views = false;
  for (int i = 0; i < run->n_slot_to_data && i < tpl->n_buf_slots; i++) {
    const PolyScheduleBufSlot *slot = &tpl->buf_slots[i];
    if (!slot->is_intermediate || !slot->has_memory_parent) continue;
    int parent = slot->memory_parent_slot;
    if (parent < 0 || parent >= run->n_slot_to_data || parent >= tpl->n_buf_slots ||
        !run->slot_to_data[parent])
      return -1;
    PolyDevice parent_device =
        poly_schedule_runtime_slot_allocated_device(tpl, run, parent, fallback_device);
    if (parent_device == POLY_DEVICE_WEBGPU) need_slot_views = true;
  }

  if (need_slot_views && !run->slot_views) {
    run->slot_views = calloc((size_t)run->n_slot_to_data, sizeof(PolyBuffer));
    if (!run->slot_views) return -1;
    run->n_slot_views = run->n_slot_to_data;
  }

  for (int i = 0; i < run->n_slot_to_data && i < tpl->n_buf_slots; i++) {
    const PolyScheduleBufSlot *slot = &tpl->buf_slots[i];
    if (!slot->is_intermediate || !slot->has_memory_parent) continue;
    int parent = slot->memory_parent_slot;
    PolyDevice parent_device =
        poly_schedule_runtime_slot_allocated_device(tpl, run, parent, fallback_device);

    if (parent_device == POLY_DEVICE_WEBGPU) {
#ifdef __EMSCRIPTEN__
      if (slot->memory_offset < 0 || slot->nbytes <= 0) return -1;
      uintptr_t view = poly_webgpu_create_buffer_view(
          (uintptr_t)run->slot_to_data[parent],
          (size_t)slot->memory_offset,
          (size_t)slot->nbytes
      );
      if (!view) return -1;
      run->slot_views[i] = (PolyBuffer){
          .ptr = (void *)view,
          .nbytes = (size_t)slot->nbytes,
          .device = POLY_DEVICE_WEBGPU,
          .owned = true,
          .allocator = poly_webgpu_get_allocator(),
          .src = NULL,
          .valid = true,
          .frontend_release = NULL,
      };
      run->slot_to_data[i] = (void *)view;
#else
      return -1;
#endif
    } else {
      run->slot_to_data[i] =
          (void *)((char *)run->slot_to_data[parent] + slot->memory_offset);
    }
  }
  return 0;
}

/* Cache flush (called from napi_api.c) */

void poly_sched_cache_flush(void) {
  /* Per-context schedule caches are flushed when context is destroyed.
   * This ABI cleanup hook remains as a no-op for existing frontend cleanup
   * paths that still call it. */
}

/* ══════════════════════════════════════════════════════════════════════ */
/*  Executable step: lower, run, free                                    */
/* ══════════════════════════════════════════════════════════════════════ */

PolyCompiledSchedule *poly_lower_schedule(PolyCtx *ctx, PolySchedule *schedule, PolyDevice device) {
  if (!ctx || !schedule) return NULL;

  const PolyBackendDesc *backend = poly_backend_get(device);
  if (!backend || !poly_device_can_execute(device) || !backend->lower_item) {
    fprintf(stderr, "polygrad: compile_schedule: unsupported device %d\n", device);
    return NULL;
  }
  if (poly_backend_ensure_open(device) != 0) {
    fprintf(stderr, "polygrad: compile_schedule: backend '%s' failed to open\n", backend->name);
    return NULL;
  }

  PolyCompiledSchedule *plan = calloc(1, sizeof(PolyCompiledSchedule));
  if (!plan) return NULL;
  plan->ctx = ctx;
  plan->template = poly_schedule_template_retain(schedule->template);
  plan->device = device;
  plan->allocator = backend->get_allocator();
  plan->run = calloc(1, sizeof(PolyScheduleRuntime));
  if (!plan->run) {
    free(plan);
    return NULL;
  }
  plan->run->device = device;
  plan->run->allocator = plan->allocator;
  plan->run->calls = calloc((size_t)schedule->template->n_calls, sizeof(PolyCallRuntime));
  plan->run->call_io = calloc((size_t)schedule->template->n_calls, sizeof(PolyCallIO));
  if (schedule->template->n_calls > 0 && !plan->run->calls) goto cleanup;
  if (schedule->template->n_calls > 0 && !plan->run->call_io) goto cleanup;
  for (int k = 0; k < schedule->template->n_calls; k++) {
    const PolyCallIO *src_io = poly_schedule_call_io(schedule, k);
    if (poly_call_io_clone_resolved(&plan->run->call_io[k], src_io) != 0) goto cleanup;
  }

  /* Lower each CALL via backend vtable. Function names are structural so
   * backend compiler caches see identical kernels as identical source. */
  uint32_t env_stamp = poly_schedule_lower_env_stamp();
  for (int k = 0; k < schedule->template->n_calls; k++) {
    PolyUOp *call = poly_schedule_call(schedule, k);
    PolyCallRuntime *rt = &plan->run->calls[k];
    rt->call = call;
    PolyRunner *runner = &rt->prg;
    bool lowered_as_copy = false;

    if (poly_call_is_view(call)) {
      memset(runner, 0, sizeof(*runner));
      runner->kind = POLY_RUNNER_VIEW;
      rt->lowered_device = device;
      rt->lowered_env_stamp = env_stamp;
      rt->prg_valid = true;
      continue;
    }

    if (poly_call_is_copy(call)) {
      if (poly_lower_copy_call(ctx, schedule, call, k, device, runner) == 0) lowered_as_copy = true;
    }
    if (!lowered_as_copy) {
      int lower_rc =
          poly_lower_compute_call_cached(ctx, call, device, env_stamp, runner, &rt->runtime_program);
      if (lower_rc == -2) {
        fprintf(stderr, "polygrad: compile_schedule: kernel %d validation failed\n", k);
        goto cleanup;
      }
      if (lower_rc != 0) {
        fprintf(
            stderr, "polygrad: compile_schedule: backend '%s' failed for kernel %d\n",
            backend->name, k
        );
        goto cleanup;
      }
    }
    rt->call = call;
    rt->lowered_device = device;
    rt->lowered_env_stamp = env_stamp;
    rt->prg_valid = true;

    if (poly_bind_runner_param_slots(ctx, schedule, call, k, runner, device) != 0) {
      fprintf(stderr, "polygrad: compile_schedule: param remap failed for kernel %d\n", k);
      goto cleanup;
    }

    runner->n_vars = 0;
    runner->var_indices = NULL;
  }

  /* Allocate persistent intermediates */
  plan->run->n_intermediates = 0;
  for (int i = 0; i < schedule->template->n_buf_slots; i++)
    if (schedule->template->buf_slots[i].is_intermediate &&
        !schedule->template->buf_slots[i].has_memory_parent)
      plan->run->n_intermediates++;

  if (plan->run->n_intermediates > 0) {
    plan->run->intermediates = calloc((size_t)plan->run->n_intermediates, sizeof(PolyBuffer));
    if (!plan->run->intermediates) goto cleanup;
    int idx = 0;
    for (int i = 0; i < schedule->template->n_buf_slots; i++) {
      if (!schedule->template->buf_slots[i].is_intermediate) continue;
      if (schedule->template->buf_slots[i].has_memory_parent) continue;
      size_t nbytes = (size_t)schedule->template->buf_slots[i].nbytes;
      if (nbytes == 0) nbytes = sizeof(float);
      PolyDevice slot_device = schedule->template->buf_slots[i].is_memory_arena
                                   ? poly_memory_arena_slot_runtime_device(ctx, schedule, i, device)
                                   : poly_intermediate_slot_runtime_device(ctx, schedule, i, device);
      if (slot_device == POLY_DEVICE_HOST || slot_device == POLY_DEVICE_AUTO) slot_device = device;
      const PolyBackendDesc *slot_backend = poly_backend_get(slot_device);
      const PolyAllocator *slot_alloc = slot_backend ? slot_backend->get_allocator() : NULL;
      if (!slot_alloc) goto cleanup;
      void *ptr = slot_alloc->alloc(nbytes, slot_alloc->dev_ctx);
      if (!ptr) goto cleanup;
      plan->run->intermediates[idx] = (PolyBuffer){
          .ptr = ptr,
          .nbytes = nbytes,
          .device = slot_device,
          .owned = true,
          .allocator = slot_alloc,
      };
      idx++;
    }
  }

  /* Allocate persistent per-kernel args arrays */
  plan->run->kernel_args = calloc((size_t)schedule->template->n_calls, sizeof(void **));
  if (schedule->template->n_calls > 0 && !plan->run->kernel_args) goto cleanup;
  for (int k = 0; k < schedule->template->n_calls; k++) {
    PolyRunner *runner = &plan->run->calls[k].prg;
    int n_args = runner->n_params + poly_call_n_var_args(poly_schedule_call(schedule, k));
    plan->run->kernel_args[k] = calloc((size_t)(n_args > 0 ? n_args : 1), sizeof(void *));
    if (!plan->run->kernel_args[k]) goto cleanup;
  }

  /* Allocate persistent slot_to_data */
  plan->run->n_slot_to_data = schedule->template->n_buf_slots;
  plan->run->slot_to_data =
      calloc((size_t)(plan->run->n_slot_to_data > 0 ? plan->run->n_slot_to_data : 1), sizeof(void *));
  if (!plan->run->slot_to_data) goto cleanup;

  /* Pre-fill intermediate slot pointers (these don't change between runs) */
  {
    int idx = 0;
    for (int i = 0; i < plan->run->n_slot_to_data; i++) {
      if (schedule->template->buf_slots[i].is_intermediate &&
          !schedule->template->buf_slots[i].has_memory_parent &&
          idx < plan->run->n_intermediates) {
        plan->run->slot_to_data[i] = plan->run->intermediates[idx].ptr;
        idx++;
      }
    }
    if (poly_schedule_runtime_fill_parent_views(schedule->template, plan->run, device) != 0)
      goto cleanup;
  }

  /* Allocate merged vars array */
  {
    int total_vars = schedule->template->n_default_vars + 16; /* room for runtime overrides */
    plan->run->merged_vars = calloc((size_t)total_vars, sizeof(PolyVarBinding));
    plan->run->merged_vars_cap = total_vars;
  }

  /* Allocate var int storage */
  {
    int total_var_ints = 0;
    for (int k = 0; k < schedule->template->n_calls; k++)
      total_var_ints += poly_call_n_var_args(poly_schedule_call(schedule, k));
    if (total_var_ints == 0) total_var_ints = 16;
    plan->run->var_int_storage = calloc((size_t)total_var_ints, sizeof(int));
    plan->run->var_int_cap = total_var_ints;
  }

  return plan;

cleanup:
  poly_compiled_schedule_free(plan);
  return NULL;
}

/* Infer the execution device from attached runtime buffers, matching the
 * graph-driven realize path. HOST buffers never force the executor. */
PolyDevice poly_schedule_infer_device(PolyCtx *ctx, const PolySchedule *sched) {
  PolyDevice preferred = poly_ctx_get_preferred_device(ctx);
  if (preferred != POLY_DEVICE_AUTO && preferred != POLY_DEVICE_HOST &&
      poly_device_can_execute(preferred))
    return preferred;

  PolyDevice device = POLY_DEVICE_AUTO;
  for (int s = 0; sched && s < sched->template->n_buf_slots; s++) {
    PolyDevice hinted = sched->template->buf_slots[s].device;
    if (hinted != POLY_DEVICE_AUTO && hinted != POLY_DEVICE_HOST &&
        poly_device_can_execute(hinted)) {
      device = hinted;
      break;
    }
  }
  for (int s = 0; sched && s < sched->template->n_buf_slots; s++) {
    if (device != POLY_DEVICE_AUTO) break;
    if (sched->template->buf_slots[s].is_intermediate) continue;
    PolyBuffer *b = poly_buffer_get(ctx, sched->template->buf_slots[s].buf_uop);
    if (b && b->device != POLY_DEVICE_HOST && b->device != POLY_DEVICE_AUTO) {
      device = b->device;
      break;
    }
  }
  if (device == POLY_DEVICE_AUTO) {
    device = poly_ctx_get_preferred_device(ctx);
    if (device == POLY_DEVICE_AUTO) device = poly_device_default();
  }
  return device;
}

static PolyDevice poly_schedule_slot_runtime_device(
    PolyCtx *ctx,
    PolySchedule *sched,
    int slot,
    PolyDevice fallback
) {
  if (sched && sched->template && slot >= 0 && slot < sched->template->n_buf_slots &&
      sched->template->buf_slots[slot].has_memory_parent) {
    int parent = sched->template->buf_slots[slot].memory_parent_slot;
    if (parent >= 0 && parent < sched->template->n_buf_slots)
      return poly_schedule_slot_runtime_device(ctx, sched, parent, fallback);
  }
  PolyDevice device = poly_schedule_slot_target_device(ctx, sched, slot, fallback);
  if (device == POLY_DEVICE_AUTO) device = fallback;
  if (device == POLY_DEVICE_AUTO) device = poly_device_default();
  return device;
}

static bool poly_call_device_can_use_host_residency(PolyDevice device) {
  return device == POLY_DEVICE_CPU || device == POLY_DEVICE_INTERP || device == POLY_DEVICE_X86;
}

static PolyDevice poly_call_arg_runtime_device(
    PolyCtx *ctx,
    PolySchedule *sched,
    PolyUOp *call,
    int slot,
    bool is_out,
    bool is_in,
    PolyDevice call_device
) {
  if (poly_call_is_copy(call)) {
    /* COPY is directional: argument 0 is destination and argument 1 is source.
     * For an external source, use the current residency's device instead of a
     * slot placement hint. This keeps browser WebGPU HOST-key imports as HOST
     * COPY sources instead of staging them through WASM first. */
    if (is_in && !is_out && sched && slot >= 0 && slot < sched->template->n_buf_slots &&
        !sched->template->buf_slots[slot].is_intermediate) {
      PolyBuffer *b = poly_buffer_get(ctx, sched->template->buf_slots[slot].buf_uop);
      if (b && b->device != POLY_DEVICE_AUTO) return b->device;
    }
    return poly_schedule_slot_runtime_device(ctx, sched, slot, call_device);
  }

  PolyDevice device = call_device;
  if (device == POLY_DEVICE_AUTO) device = poly_device_default();

  if ((is_out || is_in) && poly_call_device_can_use_host_residency(device)) {
    PolyUOp *buf_uop =
        (slot >= 0 && slot < sched->template->n_buf_slots) ? sched->template->buf_slots[slot].buf_uop : NULL;
    PolyBuffer *b = poly_buffer_get(ctx, buf_uop);
    if (b && b->ptr && b->device == POLY_DEVICE_HOST) return POLY_DEVICE_HOST;
  }

  /* For PROGRAM-like calls, outs/ins are arguments to the same kernel. A
   * stale host/CPU output buffer from an earlier readback must not be passed
   * to a CUDA/WebGPU kernel just because it is the slot's current residency.
   * This mirrors tinygrad's per-CALL outs/ins boundary. */
  if ((is_out || is_in) && device != POLY_DEVICE_HOST && poly_device_can_execute(device))
    return device;

  return poly_schedule_slot_runtime_device(ctx, sched, slot, device);
}

static PolyBuffer *poly_call_host_slot_residency(PolyCtx *ctx, PolyUOp *buf_uop, bool is_in) {
  if (!ctx || !buf_uop) return NULL;
  if (is_in) {
    PolyBuffer *b = NULL;
    return poly_buffer_ensure_host_current(ctx, buf_uop, &b) == 0 ? b : NULL;
  }

  PolyBuffer *cur = poly_buffer_get(ctx, buf_uop);
  if (!cur) return NULL;
  if (cur->device == POLY_DEVICE_HOST) return cur;
  return cur->src;
}

static int poly_mark_call_host_residency_written(
    PolyCtx *ctx,
    PolyUOp *buf_uop,
    PolyBuffer *written
) {
  if (!ctx || !buf_uop || !written) return -1;
  PolyBuffer *cur = poly_buffer_get(ctx, buf_uop);
  if (!cur) return -1;
  if (cur == written) {
    cur->valid = true;
    if (cur->src) cur->src->valid = false;
    return 0;
  }
  if (cur->src == written) {
    written->valid = true;
    cur->valid = false;
    return 0;
  }
  return -1;
}

static int poly_prepare_call_buffer_slots_common(
    PolyCtx *ctx,
    PolySchedule *sched,
    int call_index,
    PolyDevice device,
    void **slot_data,
    bool only_missing,
    bool *prepared_slots,
    bool *used_prepared_slots,
    const char *label
) {
  if (!ctx || !sched || !slot_data) return -1;
  PolyUOp *call = poly_schedule_call(sched, call_index);
  const PolyCallIO *io = poly_schedule_call_io(sched, call_index);
  if (!call || !io) return -1;
  const PolyCallAccess *access = io->access;
  if (!access) return -1;
  if (!label) label = "run_schedule";
  int rc = 0;

  for (int ai = 0; ai < access->n_active_args; ai++) {
    int i = access->active_args[ai];
    int slot = io->arg_to_slot[i];
    if (slot < 0 || slot >= sched->template->n_buf_slots) {
      rc = -1;
      break;
    }
    if (sched->template->buf_slots[slot].is_intermediate) continue;
    if (only_missing && slot_data[slot]) continue;

    PolyUOp *buf_uop = sched->template->buf_slots[slot].buf_uop;
    PolyDevice slot_device =
        poly_call_arg_runtime_device(ctx, sched, call, slot, access->outs[i], access->ins[i], device);
    PolyBuffer *b = NULL;
    if (slot_device == POLY_DEVICE_HOST) {
      b = poly_call_host_slot_residency(ctx, buf_uop, access->ins[i]);
      bool allow_keyed_host =
          poly_call_is_copy(call) && b && b->device == POLY_DEVICE_HOST && !b->ptr;
      if (!b || (!b->ptr && !allow_keyed_host)) {
        fprintf(stderr, "polygrad: %s: buffer slot %d has no data attached\n", label, slot);
        rc = -1;
        break;
      }
      slot_data[slot] = b->ptr;
      if (prepared_slots) prepared_slots[slot] = true;
      if (used_prepared_slots) *used_prepared_slots = true;
      goto prepared;
    }

    if (access->ins[i]) {
      if (poly_buffer_ensure_device_current(ctx, buf_uop, slot_device) != 0) {
        fprintf(stderr, "polygrad: %s: buffer slot %d migration copy failed\n", label, slot);
        rc = -1;
        break;
      }
    } else if (access->outs[i]) {
      if (poly_buffer_ensure_device_allocated(ctx, buf_uop, slot_device) != 0) {
        fprintf(stderr, "polygrad: %s: buffer slot %d output alloc failed\n", label, slot);
        rc = -1;
        break;
      }
    }

    b = poly_buffer_get(ctx, buf_uop);
    if (!b) {
      fprintf(stderr, "polygrad: %s: buffer slot %d has no data attached\n", label, slot);
      rc = -1;
      break;
    }
    if (!b->ptr && b->device != POLY_DEVICE_HOST) {
      fprintf(stderr, "polygrad: %s: buffer slot %d has no residency pointer\n", label, slot);
      rc = -1;
      break;
    }
    slot_data[slot] = b->ptr;
    if (prepared_slots) prepared_slots[slot] = true;
    if (used_prepared_slots) *used_prepared_slots = true;
prepared:
    if (poly_debug_at_least(7)) {
      fprintf(
          stderr,
          "[polygrad:slot] call=%p arg=%d slot=%d access=%s%s device=%s ptr=%p valid=%d src=%p "
          "src_valid=%d\n",
          (void *)call, i, slot, access->outs[i] ? "out" : "", access->ins[i] ? "in" : "",
          poly_device_name(b->device), b->ptr, (int)b->valid, b->src ? b->src->ptr : NULL,
          b->src ? (int)b->src->valid : -1
      );
      fflush(stderr);
    }
  }
  return rc;
}

static int poly_prepare_call_buffer_slots(
    PolyCtx *ctx,
    PolySchedule *sched,
    int call_index,
    PolyDevice device,
    void **slot_data
) {
  return poly_prepare_call_buffer_slots_common(
      ctx, sched, call_index, device, slot_data, false, NULL, NULL, "run_schedule"
  );
}

static int poly_prepare_missing_compiled_call_slots(
    PolyCtx *ctx,
    PolySchedule *sched,
    int call_index,
    PolyDevice device,
    void **slot_data,
    bool *ctx_slots,
    bool *used_ctx_slots
) {
  if (!ctx_slots || !used_ctx_slots) return -1;
  return poly_prepare_call_buffer_slots_common(
      ctx, sched, call_index, device, slot_data, true, ctx_slots, used_ctx_slots, "plan_run"
  );
}

static int poly_commit_call_buffer_writes(
    PolyCtx *ctx,
    PolySchedule *sched,
    int call_index,
    PolyDevice device,
    const bool *commit_slots
) {
  if (!ctx || !sched) return -1;
  PolyUOp *call = poly_schedule_call(sched, call_index);
  const PolyCallIO *io = poly_schedule_call_io(sched, call_index);
  if (!call || !io) return -1;
  const PolyCallAccess *access = io->access;
  if (!access) return -1;
  int rc = 0;

  for (int wi = 0; wi < access->n_write_args; wi++) {
    int i = access->write_args[wi];
    int slot = io->arg_to_slot[i];
    if (slot < 0 || slot >= sched->template->n_buf_slots || sched->template->buf_slots[slot].is_intermediate) continue;
    if (commit_slots && !commit_slots[slot]) continue;
    PolyDevice slot_device =
        poly_call_arg_runtime_device(ctx, sched, call, slot, access->outs[i], access->ins[i], device);
    if (slot_device == POLY_DEVICE_HOST) {
      PolyUOp *buf_uop = sched->template->buf_slots[slot].buf_uop;
      PolyBuffer *written = poly_call_host_slot_residency(ctx, buf_uop, access->ins[i]);
      if (poly_mark_call_host_residency_written(ctx, buf_uop, written) != 0) {
        rc = -1;
        break;
      }
      continue;
    }
    if (poly_buffer_mark_residency_written(ctx, sched->template->buf_slots[slot].buf_uop, slot_device) != 0) {
      rc = -1;
      break;
    }
  }
  return rc;
}

static int64_t poly_buffer_view_offset_bytes(PolyUOp *view) {
  if (!view || view->op != POLY_OP_BUFFER_VIEW) return 0;
  if (view->arg.kind != POLY_ARG_INT_TUPLE || view->arg.int_tuple.n < 2 ||
      !view->arg.int_tuple.vals)
    return 0;
  int64_t off = view->arg.int_tuple.vals[1];
  return off > 0 ? off : 0;
}

static int poly_schedule_execute_view_call(
    PolyCtx *ctx,
    PolySchedule *sched,
    int call_index,
    PolyDevice device,
    PolyScheduleRuntime *run
) {
  if (!ctx || !sched || !run || !run->slot_to_data) return -1;
  void **slot_data = run->slot_to_data;
  PolyUOp *call = poly_schedule_call(sched, call_index);
  const PolyCallIO *io = poly_schedule_call_io(sched, call_index);
  if (!poly_call_is_view(call) || !io || io->n_args < 2) return -1;

  int dst_slot = io->arg_to_slot[0];
  int src_slot = io->arg_to_slot[1];
  if (dst_slot < 0 || dst_slot >= sched->template->n_buf_slots || src_slot < 0 ||
      src_slot >= sched->template->n_buf_slots)
    return -1;

  PolyScheduleBufSlot *dst_meta = &sched->template->buf_slots[dst_slot];
  PolyScheduleBufSlot *src_meta = &sched->template->buf_slots[src_slot];
  PolyBuffer *src_buf = NULL;
  PolyBuffer src_view = {0};

  if (src_meta->is_intermediate) {
    if (!slot_data[src_slot]) return -1;
    PolyDevice src_device = poly_intermediate_slot_runtime_device(ctx, sched, src_slot, device);
    const PolyBackendDesc *be = poly_backend_get(src_device);
    src_view = (PolyBuffer){
        .ptr = slot_data[src_slot],
        .nbytes = (size_t)(src_meta->nbytes > 0 ? src_meta->nbytes : dst_meta->nbytes),
        .device = src_device,
        .owned = false,
        .allocator = be ? be->get_allocator() : NULL,
        .src = NULL,
        .valid = true,
        .frontend_release = NULL,
    };
    src_buf = &src_view;
  } else {
    PolyDevice src_device =
        poly_call_arg_runtime_device(ctx, sched, call, src_slot, false, true, device);
    if (src_device == POLY_DEVICE_HOST) {
      src_buf = poly_call_host_slot_residency(ctx, src_meta->buf_uop, true);
    } else if (poly_buffer_get(ctx, src_meta->buf_uop)) {
      if (poly_buffer_ensure_device_current(ctx, src_meta->buf_uop, src_device) != 0)
        return -1;
      src_buf = poly_buffer_get(ctx, src_meta->buf_uop);
    } else if (slot_data[src_slot]) {
      src_view = poly_buffer_make_host_view(
          slot_data[src_slot],
          (size_t)(src_meta->nbytes > 0 ? src_meta->nbytes : dst_meta->nbytes)
      );
      src_buf = &src_view;
    }
  }

  if (!src_buf || !src_buf->ptr) return -1;
  size_t nbytes = (size_t)(dst_meta->nbytes > 0 ? dst_meta->nbytes : src_meta->nbytes);
  if (nbytes == 0) nbytes = src_buf->nbytes;
  size_t offset = (size_t)poly_buffer_view_offset_bytes(call->src[0]);
  if (offset > src_buf->nbytes || nbytes > src_buf->nbytes - offset) return -1;

  void *alias_ptr = NULL;
  PolyBuffer webgpu_alias = {0};
  bool has_webgpu_alias = false;
  if (src_buf->device == POLY_DEVICE_WEBGPU && offset != 0) {
#ifdef __EMSCRIPTEN__
    uintptr_t view =
        poly_webgpu_create_buffer_view((uintptr_t)src_buf->ptr, offset, nbytes);
    if (!view) return -1;
    webgpu_alias = *src_buf;
    webgpu_alias.ptr = (void *)view;
    webgpu_alias.nbytes = nbytes;
    webgpu_alias.owned = true;
    webgpu_alias.src = NULL;
    webgpu_alias.frontend_release = NULL;
    webgpu_alias.valid = src_buf->valid;
    alias_ptr = webgpu_alias.ptr;
    has_webgpu_alias = true;
#else
    return -1;
#endif
  } else {
    alias_ptr = (void *)((char *)src_buf->ptr + offset);
  }

  if (dst_meta->is_intermediate) {
    if (has_webgpu_alias) {
      if (!run->slot_views) {
        run->slot_views = calloc((size_t)run->n_slot_to_data, sizeof(PolyBuffer));
        if (!run->slot_views) return -1;
        run->n_slot_views = run->n_slot_to_data;
      }
      run->slot_views[dst_slot] = webgpu_alias;
    }
    slot_data[dst_slot] = alias_ptr;
  } else {
    PolyBuffer alias = has_webgpu_alias ? webgpu_alias : *src_buf;
    alias.ptr = alias_ptr;
    alias.nbytes = nbytes;
    if (!has_webgpu_alias) alias.owned = false;
    alias.src = NULL;
    alias.frontend_release = NULL;
    alias.valid = src_buf->valid;
    if (has_webgpu_alias)
      poly_buffer_adopt(ctx, dst_meta->buf_uop, &alias);
    else
      poly_buffer_attach(ctx, dst_meta->buf_uop, &alias);
    slot_data[dst_slot] = alias.ptr;
  }

  if (poly_debug_at_least(7)) {
    fprintf(
        stderr,
        "[polygrad:view] call=%d dst_slot=%d src_slot=%d ptr=%p offset=%zu nbytes=%zu "
        "device=%s\n",
        call_index, dst_slot, src_slot, slot_data[dst_slot], offset, nbytes,
        poly_device_name(src_buf->device)
    );
    fflush(stderr);
  }
  return 0;
}

static uint32_t poly_schedule_lower_env_stamp(void) {
  uint32_t stamp = 2166136261u;
#ifdef __EMSCRIPTEN__
  /* WASM runtime lowering currently emits SIMD-capable modules. Keep that
   * renderer feature in the existing tinygrad-style program/runtime cache key
   * so future scalar/SIMD/relaxed/threaded variants cannot collide. */
  const uint8_t wasm_features = 1;
#else
  const uint8_t wasm_features = 0;
#endif
#ifdef POLY_HAS_X86
  uint32_t x86_features = poly_x86_feature_stamp();
#else
  uint32_t x86_features = 0;
#endif
  uint8_t bytes[] = {
      (uint8_t)poly_getenv_flag("POLY_OPTIMIZE"),
      (uint8_t)(poly_getenv_int("POLY_DEVECTORIZE", 0) & 0xFF),
      (uint8_t)(poly_getenv_int("POLY_TC_OPT", 0) & 0xFF),
      (uint8_t)(poly_getenv_int("POLY_USE_TC", 1) & 0xFF),
      wasm_features,
      (uint8_t)(x86_features & 0xFF),
      (uint8_t)((x86_features >> 8) & 0xFF),
      (uint8_t)((x86_features >> 16) & 0xFF),
      (uint8_t)((x86_features >> 24) & 0xFF),
  };
  for (size_t i = 0; i < sizeof(bytes) / sizeof(bytes[0]); i++) {
    stamp ^= bytes[i];
    stamp *= 16777619u;
  }
  return stamp;
}

static void poly_zero_buffer(PolyBuffer *h) {
  if (!h || !h->ptr) return;
  if (h->device == POLY_DEVICE_CPU || h->device == POLY_DEVICE_INTERP
#ifdef POLY_HAS_X86
      || h->device == POLY_DEVICE_X86
#endif
#ifdef __EMSCRIPTEN__
      || h->device == POLY_DEVICE_WASM
#endif
  ) {
    memset(h->ptr, 0, h->nbytes);
  }
#ifdef POLY_HAS_CUDA
  else if (h->device == POLY_DEVICE_CUDA) {
    poly_cuda_memset((unsigned long long)(uintptr_t)h->ptr, 0, h->nbytes);
  }
#endif
#ifdef POLY_HAS_HIP
  else if (h->device == POLY_DEVICE_HIP) {
    poly_hip_memset(h->ptr, 0, h->nbytes);
  }
#endif
#ifdef __EMSCRIPTEN__
  else if (h->device == POLY_DEVICE_WEBGPU) {
    poly_webgpu_memset_zero((uintptr_t)h->ptr, h->nbytes);
  }
#endif
}

static void poly_zero_schedule_slot(
    PolyCtx *ctx,
    PolySchedule *sched,
    PolyScheduleRuntime *run,
    int slot_idx
) {
  if (!ctx || !sched || !run || !run->slot_to_data || slot_idx < 0 ||
      slot_idx >= sched->template->n_buf_slots)
    return;
  PolyScheduleBufSlot *slot = &sched->template->buf_slots[slot_idx];
  if (!slot->is_intermediate || !slot->needs_zero || !run->slot_to_data[slot_idx]) return;
  size_t nbytes = (size_t)(slot->nbytes > 0 ? slot->nbytes : (int64_t)sizeof(float));
  PolyDevice dev =
      poly_schedule_runtime_slot_allocated_device(sched->template, run, slot_idx, run->device);
  const PolyBackendDesc *be = poly_backend_get(dev);
  PolyBuffer tmp = {
      .ptr = run->slot_to_data[slot_idx],
      .nbytes = nbytes,
      .device = dev,
      .owned = false,
      .allocator = be ? be->get_allocator() : NULL,
      .src = NULL,
      .valid = true,
  };
  poly_zero_buffer(&tmp);
}

static void poly_zero_schedule_initial_intermediates(
    PolyCtx *ctx,
    PolySchedule *sched,
    PolyScheduleRuntime *run
) {
  if (!sched || !run) return;
  for (int s = 0; s < sched->template->n_buf_slots; s++) {
    PolyScheduleBufSlot *slot = &sched->template->buf_slots[s];
    if (!slot->is_intermediate || !slot->needs_zero || slot->has_zero_before_call) continue;
    poly_zero_schedule_slot(ctx, sched, run, s);
  }
}

static void poly_zero_schedule_call_intermediates(
    PolyCtx *ctx,
    PolySchedule *sched,
    PolyScheduleRuntime *run,
    int call_index
) {
  if (!sched || !run) return;
  for (int s = 0; s < sched->template->n_buf_slots; s++) {
    PolyScheduleBufSlot *slot = &sched->template->buf_slots[s];
    if (!slot->is_intermediate || !slot->needs_zero || !slot->has_zero_before_call) continue;
    if (slot->zero_before_call == call_index) poly_zero_schedule_slot(ctx, sched, run, s);
  }
}

static int poly_schedule_runtime_prepare(PolyCtx *ctx, PolySchedule *sched, PolyDevice device) {
  if (!ctx || !sched) return -1;

  const PolyBackendDesc *backend = poly_backend_get(device);
  if (!backend || !poly_device_can_execute(device) || !backend->lower_item) {
    fprintf(stderr, "polygrad: run_schedule: unsupported device %d\n", device);
    return -1;
  }
  if (poly_backend_ensure_open(device) != 0) {
    fprintf(stderr, "polygrad: run_schedule: backend '%s' failed to open\n", backend->name);
    return -1;
  }

  if (sched->run->device != POLY_DEVICE_AUTO && sched->run->device != device)
    poly_schedule_runtime_destroy(sched);

  if (sched->run->device == device && sched->run->slot_to_data) return 0;

  sched->run->device = device;
  sched->run->allocator = backend->get_allocator();

  sched->run->n_intermediates = 0;
  for (int i = 0; i < sched->template->n_buf_slots; i++)
    if (sched->template->buf_slots[i].is_intermediate &&
        !sched->template->buf_slots[i].has_memory_parent)
      sched->run->n_intermediates++;

  if (sched->run->n_intermediates > 0) {
    sched->run->intermediates = calloc((size_t)sched->run->n_intermediates, sizeof(PolyBuffer));
    if (!sched->run->intermediates) goto fail;
    int idx = 0;
    for (int i = 0; i < sched->template->n_buf_slots; i++) {
      if (!sched->template->buf_slots[i].is_intermediate) continue;
      if (sched->template->buf_slots[i].has_memory_parent) continue;
      size_t nbytes = (size_t)sched->template->buf_slots[i].nbytes;
      if (nbytes == 0) nbytes = sizeof(float);
      PolyDevice slot_device = sched->template->buf_slots[i].is_memory_arena
                                   ? poly_memory_arena_slot_runtime_device(ctx, sched, i, device)
                                   : poly_intermediate_slot_runtime_device(ctx, sched, i, device);
      if (slot_device == POLY_DEVICE_HOST || slot_device == POLY_DEVICE_AUTO) slot_device = device;
      const PolyBackendDesc *slot_backend = poly_backend_get(slot_device);
      const PolyAllocator *slot_alloc = slot_backend ? slot_backend->get_allocator() : NULL;
      if (!slot_alloc) goto fail;
      void *ptr = slot_alloc->alloc(nbytes, slot_alloc->dev_ctx);
      if (!ptr) goto fail;
      sched->run->intermediates[idx] = (PolyBuffer){
          .ptr = ptr,
          .nbytes = nbytes,
          .device = slot_device,
          .owned = true,
          .allocator = slot_alloc,
      };
      idx++;
    }
  }

  sched->run->kernel_args = calloc((size_t)sched->template->n_calls, sizeof(void **));
  if (!sched->run->kernel_args) goto fail;

  sched->run->n_slot_to_data = sched->template->n_buf_slots;
  sched->run->slot_to_data = calloc(
      (size_t)(sched->run->n_slot_to_data > 0 ? sched->run->n_slot_to_data : 1), sizeof(void *)
  );
  if (!sched->run->slot_to_data) goto fail;

  {
    int idx = 0;
    for (int i = 0; i < sched->run->n_slot_to_data; i++) {
      if (sched->template->buf_slots[i].is_intermediate &&
          !sched->template->buf_slots[i].has_memory_parent &&
          idx < sched->run->n_intermediates)
        sched->run->slot_to_data[i] = sched->run->intermediates[idx++].ptr;
    }
    if (poly_schedule_runtime_fill_parent_views(sched->template, sched->run, device) != 0)
      goto fail;
  }

  {
    int total_vars = sched->template->n_default_vars + 16;
    sched->run->merged_vars = calloc((size_t)total_vars, sizeof(PolyVarBinding));
    if (!sched->run->merged_vars) goto fail;
    sched->run->merged_vars_cap = total_vars;
  }

  {
    int total_var_ints = 0;
    for (int k = 0; k < sched->template->n_calls; k++)
      total_var_ints += poly_call_n_var_args(poly_schedule_call(sched, k));
    if (total_var_ints == 0) total_var_ints = 16;
    sched->run->var_int_storage = calloc((size_t)total_var_ints, sizeof(int));
    if (!sched->run->var_int_storage) goto fail;
    sched->run->var_int_cap = total_var_ints;
  }

  return 0;

fail:
  poly_schedule_runtime_destroy(sched);
  return -1;
}

static int poly_schedule_merge_runtime_vars(
    PolySchedule *sched,
    PolyVarBinding *var_bindings,
    int n_var_bindings
) {
  int n_all = 0;
  int needed = sched->template->n_default_vars + n_var_bindings;
  if (needed > sched->run->merged_vars_cap) {
    free(sched->run->merged_vars);
    sched->run->merged_vars = calloc((size_t)needed, sizeof(PolyVarBinding));
    if (!sched->run->merged_vars) {
      sched->run->merged_vars_cap = 0;
      return -1;
    }
    sched->run->merged_vars_cap = needed;
  }

  for (int i = 0; i < sched->template->n_default_vars; i++)
    sched->run->merged_vars[n_all++] = sched->template->default_vars[i];
  for (int i = 0; i < n_var_bindings; i++) {
    bool found = false;
    for (int j = 0; j < n_all; j++) {
      if (sched->run->merged_vars[j].var == var_bindings[i].var) {
        sched->run->merged_vars[j].value = var_bindings[i].value;
        found = true;
        break;
      }
    }
    if (!found) sched->run->merged_vars[n_all++] = var_bindings[i];
  }
  return n_all;
}

int poly_schedule_call_lower(
    PolyCtx *ctx,
    PolySchedule *schedule,
    int call_index,
    PolyDevice device
) {
  if (!ctx || !schedule || call_index < 0 || call_index >= schedule->template->n_calls) return -1;
  if (!schedule->run->slot_to_data && poly_schedule_runtime_prepare(ctx, schedule, device) != 0)
    return -1;

  PolyUOp *call = poly_schedule_call(schedule, call_index);
  PolyCallRuntime *rt = &schedule->run->calls[call_index];
  uint32_t env_stamp = poly_schedule_lower_env_stamp();
  bool timing = poly_debug_at_least(7);
  double t0 = timing ? poly_now_ms() : 0.0;
  if (timing) {
    fprintf(
        stderr,
        "[polygrad:call_lower] begin call=%d/%d kind=%s device=%s body=%p params=%d vars=%d\n",
        call_index, schedule->template->n_calls, poly_call_kind_name(call), poly_device_name(device),
        (void *)poly_schedule_call_body(schedule, call_index), poly_call_n_buffer_args(call),
        poly_call_n_var_args(call)
    );
    fflush(stderr);
  }
  if (rt->prg_valid && rt->lowered_device == device && rt->lowered_env_stamp == env_stamp) return 0;
  if (rt->prg_valid || rt->runtime_program) poly_call_runtime_cleanup(rt);

  if (poly_call_is_view(call)) {
    memset(&rt->prg, 0, sizeof(rt->prg));
    rt->prg.kind = POLY_RUNNER_VIEW;
    rt->call = call;
    rt->lowered_device = device;
    rt->lowered_env_stamp = env_stamp;
    rt->prg_valid = true;
    return 0;
  }

  const PolyBackendDesc *backend = poly_backend_get(device);
  if (!backend || !backend->lower_item) return -1;
  if (poly_backend_ensure_open(device) != 0) return -1;
  bool lowered_as_copy = false;

  if (poly_call_is_copy(call)) {
    if (poly_lower_copy_call(ctx, schedule, call, call_index, device, &rt->prg) == 0)
      lowered_as_copy = true;
  }
  if (!lowered_as_copy) {
    int lower_rc =
        poly_lower_compute_call_cached(ctx, call, device, env_stamp, &rt->prg, &rt->runtime_program);
    if (lower_rc == -2) {
      fprintf(stderr, "polygrad: call_lower: kernel %d validation failed\n", call_index);
      return -1;
    }
    if (lower_rc != 0) {
      fprintf(
          stderr, "polygrad: call_lower: backend '%s' failed for kernel %d\n", backend->name,
          call_index
      );
      memset(&rt->prg, 0, sizeof(rt->prg));
      return -1;
    }
  }
  rt->call = call;
  rt->lowered_device = device;
  rt->lowered_env_stamp = env_stamp;
  rt->prg_valid = true;

  if (poly_bind_runner_param_slots(ctx, schedule, call, call_index, &rt->prg, device) != 0) {
    poly_call_runtime_cleanup(rt);
    fprintf(stderr, "polygrad: call_lower: param remap failed for kernel %d\n", call_index);
    return -1;
  }
  rt->prg.n_vars = 0;
  rt->prg.var_indices = NULL;

  free(schedule->run->kernel_args[call_index]);
  int n_args = rt->prg.n_params + poly_call_n_var_args(call);
  schedule->run->kernel_args[call_index] = calloc((size_t)(n_args > 0 ? n_args : 1), sizeof(void *));
  if (!schedule->run->kernel_args[call_index]) {
    poly_call_runtime_cleanup(rt);
    return -1;
  }

  if (timing) {
    double t1 = poly_now_ms();
    fprintf(
        stderr,
        "[polygrad:call_lower] done call=%d kind=%s runner=%d params=%d wgsl_or_size=%d ms=%.3f\n",
        call_index, poly_call_kind_name(call), (int)rt->prg.kind, rt->prg.n_params,
        rt->prg.handle_size, t1 - t0
    );
    fflush(stderr);
  }
  return 0;
}

static int poly_schedule_execute_runner_call(
    PolySchedule *sched,
    int exec_step,
    int call_index,
    PolyRunner *runner,
    PolyDevice device,
    void **slot_to_data,
    void **args,
    PolyVarBinding *merged_vars,
    int n_all,
    int *var_int_idx,
    int **var_int_storage,
    int *var_int_cap,
    const PolyCompiledSchedule *compiled_debug_plan,
    const char *label
);

static int poly_schedule_call_run_prepared(
    PolyCtx *ctx,
    PolySchedule *sched,
    int exec_step,
    int call_index,
    int n_all,
    int *var_int_idx
) {
  PolyCallRuntime *rt = &sched->run->calls[call_index];
  if (poly_call_is_view(poly_schedule_call(sched, call_index))) {
    return poly_schedule_execute_view_call(ctx, sched, call_index, rt->lowered_device, sched->run);
  }

  PolyRunner *runner = &rt->prg;
  if (poly_prepare_call_buffer_slots(
          ctx, sched, call_index, rt->lowered_device, sched->run->slot_to_data
      ) != 0)
    return -1;

  int ret = poly_schedule_execute_runner_call(
      sched, exec_step, call_index, runner, rt->lowered_device, sched->run->slot_to_data,
      sched->run->kernel_args[call_index], sched->run->merged_vars, n_all, var_int_idx,
      &sched->run->var_int_storage, &sched->run->var_int_cap, NULL, "run_schedule"
  );
  if (ret != 0) return ret;

  if (poly_commit_call_buffer_writes(ctx, sched, call_index, rt->lowered_device, NULL) != 0)
    return -1;
  return 0;
}

static int poly_schedule_execute_runner_call(
    PolySchedule *sched,
    int exec_step,
    int call_index,
    PolyRunner *runner,
    PolyDevice device,
    void **slot_to_data,
    void **args,
    PolyVarBinding *merged_vars,
    int n_all,
    int *var_int_idx,
    int **var_int_storage,
    int *var_int_cap,
    const PolyCompiledSchedule *compiled_debug_plan,
    const char *label
) {
  if (!sched || !runner || !slot_to_data || !args || !var_int_idx || !var_int_storage ||
      !var_int_cap)
    return -1;
  if (!label) label = "run_schedule";
  PolyUOp *call = poly_schedule_call(sched, call_index);
  const PolyBackendDesc *backend = poly_backend_get(device);
  if (backend && poly_backend_ensure_open(device) != 0) return -1;
  if (!runner->handle) {
    fprintf(stderr, "polygrad: %s: runner %d has no handle\n", label, call_index);
    return -1;
  }

  int n_vars = poly_call_n_var_args(call);
  int n_args = runner->n_params + n_vars;

  for (int i = 0; i < runner->n_params; i++) {
    int slot = runner->param_to_slot[i];
    if (slot >= 0 && slot < sched->template->n_buf_slots) args[i] = slot_to_data[slot];
    if (!args[i] && !runner_copy_param_allows_null(runner, i)) {
      fprintf(
          stderr, "polygrad: %s: missing data for param %d (slot %d) in kernel %d\n", label, i,
          slot, call_index
      );
      return -1;
    }
  }
  if (n_vars > 0) {
    for (int v = 0; v < n_vars; v++) {
      PolyUOp *var = poly_call_var_arg(call, v);
      bool found = false;
      for (int vb = 0; vb < n_all; vb++) {
        if (merged_vars[vb].var == var) {
          if (*var_int_idx >= *var_int_cap) {
            int new_cap = (*var_int_cap > 0) ? (*var_int_cap * 2) : 16;
            while (*var_int_idx >= new_cap) new_cap *= 2;
            int *new_storage = realloc(*var_int_storage, (size_t)new_cap * sizeof(int));
            if (!new_storage) return -1;
            *var_int_storage = new_storage;
            *var_int_cap = new_cap;
          }
          (*var_int_storage)[*var_int_idx] = (int)merged_vars[vb].value;
          args[runner->n_params + v] = &(*var_int_storage)[*var_int_idx];
          (*var_int_idx)++;
          found = true;
          break;
        }
      }
      if (!found) {
        fprintf(stderr, "polygrad: %s: no binding for DEFINE_VAR in kernel %d\n", label, call_index);
        return -1;
      }
    }
  }

  if (poly_resolve_runner_launch_dims(runner, merged_vars, n_all) != 0) {
    fprintf(
        stderr, "polygrad: %s: failed to resolve launch dims for kernel %d\n", label, call_index
    );
    return -1;
  }

  if (compiled_debug_plan && compiled_debug_plan->device == POLY_DEVICE_WEBGPU)
    debug_dump_webgpu_runner_args(compiled_debug_plan, exec_step, call_index, runner, args);
  else
    debug_dump_schedule_args(sched, device, exec_step, call_index, runner, args);
  bool call_timing = poly_debug_at_least(7);
  double t_exec0 = call_timing ? poly_now_ms() : 0.0;
  int ret;
  if (runner->execute) {
    ret = runner->execute(runner, args, n_args);
  } else if (backend && backend->execute) {
    ret = backend->execute(runner, args, n_args);
  } else {
    fprintf(
        stderr, "polygrad: %s: no execute hook for backend '%s' kernel %d\n", label,
        backend && backend->name ? backend->name : "<unknown>", call_index
    );
    return -1;
  }
  if (call_timing) {
    double t_exec1 = poly_now_ms();
    fprintf(
        stderr,
        "[polygrad:%s] call=%d step=%d device=%s kind=%d params=%d vars=%d grid=(%d,%d,%d) "
        "block=(%d,%d,%d) exec=%.3fms ret=%d\n",
        label, call_index, exec_step, poly_device_name(device), (int)runner->kind, runner->n_params,
        n_vars, runner->grid[0], runner->grid[1], runner->grid[2], runner->block[0],
        runner->block[1], runner->block[2], t_exec1 - t_exec0, ret
    );
    fflush(stderr);
  }
  if (ret != 0) {
    fprintf(
        stderr, "polygrad: kernel %d/%d failed (params=%d grid=%d block=%d)\n", exec_step,
        sched->template->n_calls, runner->n_params, runner->grid[0], runner->block[0]
    );
  }
  return ret;
}

int poly_schedule_call_run(
    PolyCtx *ctx,
    PolySchedule *schedule,
    int call_index,
    PolyVarBinding *var_bindings,
    int n_var_bindings
) {
  if (!ctx || !schedule || call_index < 0 || call_index >= schedule->template->n_calls) return -1;

  PolyDevice device = poly_schedule_infer_device(ctx, schedule);
  if (poly_schedule_runtime_prepare(ctx, schedule, device) != 0) return -1;
  PolyDevice item_device = poly_call_device(ctx, schedule, call_index, device);
  if (poly_schedule_call_lower(ctx, schedule, call_index, item_device) != 0) return -1;
  poly_zero_schedule_initial_intermediates(ctx, schedule, schedule->run);
  poly_zero_schedule_call_intermediates(ctx, schedule, schedule->run, call_index);

  int n_all = poly_schedule_merge_runtime_vars(schedule, var_bindings, n_var_bindings);
  int var_int_idx = 0;
  return (n_all < 0)
             ? -1
             : poly_schedule_call_run_prepared(ctx, schedule, 0, call_index, n_all, &var_int_idx);
}

int poly_run_schedule(
    PolyCtx *ctx,
    PolySchedule *schedule,
    PolyVarBinding *var_bindings,
    int n_var_bindings
) {
  if (!ctx || !schedule) return -1;

  bool timing = poly_debug_at_least(2);
  double t0 = timing ? poly_now_ms() : 0.0;
  PolyDevice device = poly_schedule_infer_device(ctx, schedule);
  if (timing) {
    fprintf(
        stderr,
        "[polygrad:run_schedule] begin device=%s calls=%d slots=%d intermediates=%d "
        "default_vars=%d runtime_vars=%d\n",
        poly_device_name(device), schedule->template->n_calls, schedule->template->n_buf_slots,
        schedule->run->n_intermediates, schedule->template->n_default_vars, n_var_bindings
    );
    fflush(stderr);
  }
  if (poly_schedule_runtime_prepare(ctx, schedule, device) != 0) {
    fprintf(stderr, "polygrad: run_schedule: compile failed\n");
    return -1;
  }
  double t_prepare = timing ? poly_now_ms() : 0.0;
  double t_slots = timing ? poly_now_ms() : 0.0;

  poly_zero_schedule_initial_intermediates(ctx, schedule, schedule->run);
  double t_zero = timing ? poly_now_ms() : 0.0;

  int n_all = poly_schedule_merge_runtime_vars(schedule, var_bindings, n_var_bindings);
  if (n_all < 0) return -1;
  double t_vars = timing ? poly_now_ms() : 0.0;

  int ret = 0;
  int var_int_idx = 0;
  for (int k = 0; k < schedule->template->n_calls && ret == 0; k++) {
    PolyDevice item_device = poly_call_device(ctx, schedule, k, device);
    if (poly_debug_at_least(7)) {
      fprintf(
          stderr, "[polygrad:run_schedule] step=%d/%d call=%d device=%s lower begin\n", k + 1,
          schedule->template->n_calls, k, poly_device_name(item_device)
      );
      fflush(stderr);
    }
    if (poly_schedule_call_lower(ctx, schedule, k, item_device) != 0) {
      ret = -1;
      break;
    }
    poly_zero_schedule_call_intermediates(ctx, schedule, schedule->run, k);
    if (poly_debug_at_least(7)) {
      fprintf(
          stderr, "[polygrad:run_schedule] step=%d/%d call=%d run begin\n", k + 1,
          schedule->template->n_calls, k
      );
      fflush(stderr);
    }
    ret = poly_schedule_call_run_prepared(ctx, schedule, k, k, n_all, &var_int_idx);
    if (poly_debug_at_least(7)) {
      fprintf(
          stderr, "[polygrad:run_schedule] step=%d/%d call=%d run end ret=%d\n", k + 1,
          schedule->template->n_calls, k, ret
      );
      fflush(stderr);
    }
  }
  if (timing) {
    double t_done = poly_now_ms();
    fprintf(
        stderr,
        "[polygrad:run_schedule] done prepare=%.3fms slots=%.3fms zero=%.3fms vars=%.3fms "
        "loop=%.3fms total=%.3fms ret=%d\n",
        t_prepare - t0, t_slots - t_prepare, t_zero - t_slots, t_vars - t_zero, t_done - t_vars,
        t_done - t0, ret
    );
    fflush(stderr);
  }
  return ret;
}

int poly_run_compiled_schedule(
    PolyCompiledSchedule *plan,
    void **slot_data,
    int n_slots,
    PolyVarBinding *var_bindings,
    int n_var_bindings
) {
  if (!plan || !plan->template) return -1;
  bool timing = poly_debug_at_least(7);
  double t0 = timing ? poly_now_ms() : 0.0;
  PolySchedule sched_view = {.template = plan->template, .run = plan->run};
  PolySchedule *sched = &sched_view;
  PolyScheduleRuntime *run = plan->run;
  if (!run) return -1;

  const PolyBackendDesc *backend = poly_backend_get(plan->device);
  if (!backend) return -1;
  if (poly_backend_ensure_open(plan->device) != 0) return -1;
  double t_backend = timing ? poly_now_ms() : 0.0;

  int ret = 0;

  /* Fill external slots in persistent slot_to_data */
  for (int i = 0; i < run->n_slot_to_data; i++) {
    if (!sched->template->buf_slots[i].is_intermediate) {
      run->slot_to_data[i] = (i < n_slots && slot_data[i]) ? slot_data[i] : NULL;
    }
    /* intermediate slots are pre-filled at compile time and don't change */
  }
  if (run->n_ctx_slots < run->n_slot_to_data) {
    bool *ctx_slots = realloc(
        run->ctx_slots, (size_t)(run->n_slot_to_data > 0 ? run->n_slot_to_data : 1) * sizeof(bool)
    );
    if (!ctx_slots) return -1;
    run->ctx_slots = ctx_slots;
    run->n_ctx_slots = run->n_slot_to_data;
  }
  memset(run->ctx_slots, 0, (size_t)run->n_ctx_slots * sizeof(bool));
  bool *ctx_slots = run->ctx_slots;
  bool used_ctx_slots = false;
  double t_slots = timing ? poly_now_ms() : 0.0;

  poly_zero_schedule_initial_intermediates(plan->ctx, sched, run);
  double t_zero = timing ? poly_now_ms() : 0.0;

  /* Merge default vars with runtime overrides */
  int n_all = 0;

  /* Grow merged_vars if needed */
  int needed = sched->template->n_default_vars + n_var_bindings;
  if (needed > run->merged_vars_cap) {
    free(run->merged_vars);
    run->merged_vars = calloc((size_t)needed, sizeof(PolyVarBinding));
    run->merged_vars_cap = needed;
  }

  for (int i = 0; i < sched->template->n_default_vars; i++)
    run->merged_vars[n_all++] = sched->template->default_vars[i];
  for (int i = 0; i < n_var_bindings; i++) {
    bool found = false;
    for (int j = 0; j < n_all; j++) {
      if (run->merged_vars[j].var == var_bindings[i].var) {
        run->merged_vars[j].value = var_bindings[i].value;
        found = true;
        break;
      }
    }
    if (!found) run->merged_vars[n_all++] = var_bindings[i];
  }
  double t_vars = timing ? poly_now_ms() : 0.0;

  /* Execute LINEAR calls in order via backend vtable. */
  int var_int_idx = 0;
  double t_loop_prepare = 0.0;
  double t_loop_execute = 0.0;
  double t_loop_commit = 0.0;
  for (int k = 0; k < sched->template->n_calls && ret == 0; k++) {
    PolyRunner *runner = &run->calls[k].prg;
    poly_zero_schedule_call_intermediates(plan->ctx, sched, run, k);

    if (poly_call_is_view(poly_schedule_call(sched, k))) {
      ret = poly_schedule_execute_view_call(plan->ctx, sched, k, plan->device, run);
      continue;
    }

    memset(ctx_slots, 0, (size_t)run->n_ctx_slots * sizeof(bool));
    used_ctx_slots = false;
    double t_call_prepare0 = timing ? poly_now_ms() : 0.0;
    if (poly_prepare_missing_compiled_call_slots(
            plan->ctx, sched, k, plan->device, run->slot_to_data, ctx_slots, &used_ctx_slots
        ) != 0) {
      ret = -1;
      break;
    }
    double t_call_execute0 = timing ? poly_now_ms() : 0.0;
    if (timing) t_loop_prepare += t_call_execute0 - t_call_prepare0;

    ret = poly_schedule_execute_runner_call(
        sched, k, k, runner, plan->device, run->slot_to_data, run->kernel_args[k],
        run->merged_vars, n_all, &var_int_idx, &run->var_int_storage, &run->var_int_cap, plan,
        "plan_run"
    );
    double t_call_commit0 = timing ? poly_now_ms() : 0.0;
    if (timing) t_loop_execute += t_call_commit0 - t_call_execute0;
    if (ret == 0 && used_ctx_slots) {
      ret = poly_commit_call_buffer_writes(plan->ctx, sched, k, plan->device, ctx_slots);
    }
    if (timing) t_loop_commit += poly_now_ms() - t_call_commit0;
  }

  if (timing) {
    double t_done = poly_now_ms();
    fprintf(
        stderr,
        "[polygrad:compiled_schedule] device=%s calls=%d slots=%d backend=%.3fms "
        "slots=%.3fms zero=%.3fms vars=%.3fms loop=%.3fms "
        "loop_prepare=%.3fms loop_execute=%.3fms loop_commit=%.3fms total=%.3fms ret=%d\n",
        poly_device_name(plan->device), sched->template->n_calls, sched->template->n_buf_slots,
        t_backend - t0, t_slots - t_backend, t_zero - t_slots, t_vars - t_zero,
        t_done - t_vars, t_loop_prepare, t_loop_execute, t_loop_commit, t_done - t0, ret
    );
    fflush(stderr);
  }
  return ret;
}

void poly_compiled_schedule_free(PolyCompiledSchedule *plan) {
  if (!plan) return;

  if (plan->run) {
    poly_schedule_runtime_cleanup(plan->template, plan->run);
    free(plan->run->calls);
    free(plan->run);
  }
  poly_schedule_template_release(plan->template);

  free(plan);
}

size_t poly_compiled_schedule_runtime_intermediate_bytes(const PolyCompiledSchedule *plan) {
  return plan ? poly_schedule_runtime_owned_bytes(plan->run) : 0;
}
