/* jit.c -- tinygrad-style JIT capture/replay for raw Tensor realizes. */

#include "engine/jit.h"
#include "ctx.h"
#include "device.h"
#include "tensor.h"
#include "utils.h"

#include <stdlib.h>

struct PolyJit {
  PolyCtx *ctx;
  PolyUOp **input_buffers;
  PolyDType *input_dtypes;
  PolyDevice *input_devices;
  int n_inputs;
  PolySchedule **schedules;
  int n_schedules;
  int schedules_cap;
  int n_recorded_schedules;
  PolySchedule *captured_linear;
  PolyCompiledSchedule *compiled_linear;
  PolyDevice compiled_device;
  bool prune;
  bool capturing;
  bool captured;
};

static void poly_jit_clear(PolyJit *jit) {
  if (!jit) return;
  if (jit->schedules) {
    for (int i = 0; i < jit->n_schedules; i++)
      poly_schedule_free(jit->schedules[i]);
  }
  free(jit->schedules);
  poly_compiled_schedule_free(jit->compiled_linear);
  poly_schedule_free(jit->captured_linear);
  free(jit->input_buffers);
  free(jit->input_dtypes);
  free(jit->input_devices);
  jit->schedules = NULL;
  jit->captured_linear = NULL;
  jit->compiled_linear = NULL;
  jit->compiled_device = POLY_DEVICE_AUTO;
  jit->input_buffers = NULL;
  jit->input_dtypes = NULL;
  jit->input_devices = NULL;
  jit->n_schedules = 0;
  jit->schedules_cap = 0;
  jit->n_recorded_schedules = 0;
  jit->n_inputs = 0;
  jit->captured = false;
}

static PolyUOp *poly_jit_tensor_buffer(PolyTensor *tensor) {
  PolyUOp *current = poly_tensor_uop(tensor);
  const PolyUOp *buf = poly_uop_get_buffer_identity(current);
  return (PolyUOp *)buf;
}

static int poly_jit_input_index(PolyJit *jit, PolyUOp *buf) {
  if (!jit || !buf) return -1;
  for (int i = 0; i < jit->n_inputs; i++)
    if (jit->input_buffers[i] == buf) return i;
  return -1;
}

static bool poly_jit_capture_input_spec(PolyJit *jit, int index, PolyTensor *tensor, PolyUOp *buf) {
  if (!jit || index < 0 || index >= jit->n_inputs || !tensor || !buf) return false;
  PolyUOp *root = poly_tensor_uop(tensor);
  if (!root) return false;
  jit->input_buffers[index] = buf;
  jit->input_dtypes[index] = poly_dtype_scalar(root->dtype);
  jit->input_devices[index] = poly_uop_device(buf);
  return true;
}

static bool poly_jit_input_matches_spec(PolyJit *jit, int index, PolyTensor *tensor, PolyUOp *buf) {
  if (!jit || index < 0 || index >= jit->n_inputs || !tensor || !buf) return false;
  PolyUOp *root = poly_tensor_uop(tensor);
  if (!root) return false;
  if (!poly_dtype_eq(poly_dtype_scalar(root->dtype), jit->input_dtypes[index])) return false;
  if (poly_uop_device(buf) != jit->input_devices[index]) return false;
  return true;
}

static int poly_jit_append_unique_buffer(PolyUOp ***items, int *n_items, int *cap_items, PolyUOp *buf) {
  if (!items || !n_items || !cap_items || !buf) return -1;
  for (int i = 0; i < *n_items; i++)
    if ((*items)[i] == buf) return i;
  if (*n_items >= *cap_items) {
    int new_cap = *cap_items ? *cap_items * 2 : 8;
    PolyUOp **tmp = realloc(*items, (size_t)new_cap * sizeof(*tmp));
    if (!tmp) return -1;
    *items = tmp;
    *cap_items = new_cap;
  }
  (*items)[*n_items] = buf;
  return (*n_items)++;
}

static int poly_jit_collect_external_buffers(
    PolySchedule **schedules,
    int n_schedules,
    PolyUOp ***external_out,
    int *n_external_out
) {
  if (!schedules || n_schedules <= 0 || !external_out || !n_external_out) return -1;
  PolyUOp **external = NULL;
  int n_external = 0, cap_external = 0;
  for (int s = 0; s < n_schedules; s++) {
    PolySchedule *captured = schedules[s];
    int n_sched_external = poly_schedule_external_slot_count(captured);
    if (n_sched_external < 0) goto fail;
    for (int i = 0; i < n_sched_external; i++) {
      PolyUOp *buf = poly_schedule_external_slot_buffer(captured, i);
      if (!buf || poly_jit_append_unique_buffer(&external, &n_external, &cap_external, buf) < 0)
        goto fail;
    }
  }
  *external_out = external;
  *n_external_out = n_external;
  return 0;

fail:
  free(external);
  return -1;
}

static int poly_jit_build_captured_linear(PolyJit *jit) {
  if (!jit || !jit->ctx || jit->n_schedules <= 0) return -1;
  PolyUOp **external = NULL;
  int n_external = 0;
  if (poly_jit_collect_external_buffers(jit->schedules, jit->n_schedules, &external, &n_external) != 0)
    return -1;
  PolySchedule *combined = poly_schedule_replay_many_with_buffers(
      jit->ctx, jit->schedules, jit->n_schedules, external, external, n_external
  );
  free(external);
  if (!combined) return -1;
  if (jit->prune) {
    PolySchedule *pruned =
        poly_schedule_prune_for_buffers(jit->ctx, combined, jit->input_buffers, jit->n_inputs);
    poly_schedule_free(combined);
    combined = pruned;
    if (!combined) return -1;
    /* Tinygrad runs pruned onetime calls during capture finalization because
     * capture itself defers execution. Polygrad capture has already executed
     * the captured schedules, so pruning only removes those calls from replay. */
  }
  jit->captured_linear = combined;
  jit->n_recorded_schedules = jit->n_schedules;
  for (int i = 0; i < jit->n_schedules; i++) poly_schedule_free(jit->schedules[i]);
  free(jit->schedules);
  jit->schedules = NULL;
  jit->n_schedules = 0;
  jit->schedules_cap = 0;
  return 0;
}

PolyJit *poly_jit_new(PolyCtx *ctx) {
  if (!ctx) return NULL;
  PolyJit *jit = calloc(1, sizeof(*jit));
  if (!jit) return NULL;
  jit->ctx = ctx;
  return jit;
}

void poly_jit_free(PolyJit *jit) {
  if (!jit) return;
  if (jit->capturing && jit->ctx && jit->ctx->active_jit_capture == jit)
    jit->ctx->active_jit_capture = NULL;
  poly_jit_clear(jit);
  free(jit);
}

int poly_jit_set_prune(PolyJit *jit, bool prune) {
  if (!jit || jit->capturing || jit->captured) return -1;
  jit->prune = prune;
  return 0;
}

int poly_jit_begin_capture(PolyJit *jit, PolyTensor **inputs, int n_inputs) {
  if (!jit || !jit->ctx || n_inputs < 0 || (n_inputs > 0 && !inputs)) return -1;
  if (jit->ctx->active_jit_capture && jit->ctx->active_jit_capture != jit) return -1;

  poly_jit_clear(jit);
  if (n_inputs > 0) {
    jit->input_buffers = calloc((size_t)n_inputs, sizeof(*jit->input_buffers));
    jit->input_dtypes = calloc((size_t)n_inputs, sizeof(*jit->input_dtypes));
    jit->input_devices = calloc((size_t)n_inputs, sizeof(*jit->input_devices));
    if (!jit->input_buffers || !jit->input_dtypes || !jit->input_devices) {
      poly_jit_clear(jit);
      return -1;
    }
  }
  jit->n_inputs = n_inputs;
  for (int i = 0; i < n_inputs; i++) {
    PolyUOp *buf = poly_jit_tensor_buffer(inputs[i]);
    if (!buf) {
      poly_jit_clear(jit);
      return -1;
    }
    if (poly_jit_input_index(jit, buf) >= 0) {
      poly_jit_clear(jit);
      return -1;
    }
    if (!poly_jit_capture_input_spec(jit, i, inputs[i], buf)) {
      poly_jit_clear(jit);
      return -1;
    }
  }
  jit->capturing = true;
  jit->ctx->active_jit_capture = jit;
  return 0;
}

int poly_jit_end_capture(PolyJit *jit) {
  if (!jit || !jit->ctx || !jit->capturing) return -1;
  if (jit->ctx->active_jit_capture == jit) jit->ctx->active_jit_capture = NULL;
  jit->capturing = false;
  if (jit->n_schedules > 0 && poly_jit_build_captured_linear(jit) == 0) {
    jit->captured = true;
    return 0;
  }
  poly_jit_clear(jit);
  return -1;
}

void poly_jit_cancel_capture(PolyJit *jit) {
  if (!jit || !jit->capturing) return;
  if (jit->ctx && jit->ctx->active_jit_capture == jit) jit->ctx->active_jit_capture = NULL;
  jit->capturing = false;
  poly_jit_clear(jit);
}

bool poly_jit_is_captured(PolyJit *jit) {
  return jit && jit->captured;
}

bool poly_jit_is_capturing(PolyJit *jit) {
  return jit && jit->capturing;
}

int poly_jit_schedule_count(PolyJit *jit) {
  if (!jit) return 0;
  return jit->captured ? jit->n_recorded_schedules : jit->n_schedules;
}

int poly_jit_record_schedule(PolyJit *jit, PolySchedule *sched) {
  if (!jit || !sched) return -1;
  if (jit->n_schedules >= jit->schedules_cap) {
    int new_cap = jit->schedules_cap ? jit->schedules_cap * 2 : 4;
    PolySchedule **new_schedules = realloc(jit->schedules, (size_t)new_cap * sizeof(*new_schedules));
    if (!new_schedules) return -1;
    jit->schedules = new_schedules;
    jit->schedules_cap = new_cap;
  }
  jit->schedules[jit->n_schedules++] = sched;
  return 0;
}

static PolySchedule *poly_jit_build_replay_schedule(PolyJit *jit, PolyUOp **current_inputs) {
  if (!jit || !current_inputs || !jit->captured_linear) return NULL;
  PolyUOp **captured_external = NULL;
  PolyUOp **replay_external = NULL;
  int n_external = 0;
  if (poly_jit_collect_external_buffers(&jit->captured_linear, 1, &captured_external, &n_external) != 0)
    return NULL;

  replay_external = calloc((size_t)n_external, sizeof(*replay_external));
  if (n_external > 0 && !replay_external) goto fail;
  for (int i = 0; i < n_external; i++) {
    PolyUOp *buf = captured_external[i];
    int input_idx = poly_jit_input_index(jit, buf);
    replay_external[i] = (input_idx >= 0) ? current_inputs[input_idx] : buf;
  }

  PolySchedule *replay =
      poly_schedule_replay_with_buffers(jit->ctx, jit->captured_linear, replay_external, n_external);
  free(captured_external);
  free(replay_external);
  return replay;

fail:
  free(captured_external);
  free(replay_external);
  return NULL;
}

static bool poly_jit_inputs_are_captured(PolyJit *jit, PolyUOp **current_inputs) {
  if (!jit || (!current_inputs && jit->n_inputs > 0)) return false;
  for (int i = 0; i < jit->n_inputs; i++) {
    if (current_inputs[i] != jit->input_buffers[i]) return false;
  }
  return true;
}

static int poly_jit_run_captured_linear(
    PolyJit *jit,
    PolyVarBinding *var_bindings,
    int n_var_bindings
) {
  if (!jit || !jit->captured_linear) return -1;
  PolyDevice device = poly_schedule_infer_device(jit->ctx, jit->captured_linear);
  if (!jit->compiled_linear || jit->compiled_device != device) {
    poly_compiled_schedule_free(jit->compiled_linear);
    jit->compiled_linear = poly_lower_schedule(jit->ctx, jit->captured_linear, device);
    jit->compiled_device = jit->compiled_linear ? device : POLY_DEVICE_AUTO;
    if (!jit->compiled_linear) return -1;
  }
  return poly_run_compiled_schedule(jit->compiled_linear, NULL, 0, var_bindings, n_var_bindings);
}

int poly_jit_run_with_vars(
    PolyJit *jit,
    PolyTensor **inputs,
    int n_inputs,
    PolyVarBinding *var_bindings,
    int n_var_bindings
) {
  if (!jit || !jit->captured || !jit->ctx || (n_inputs > 0 && !inputs) || n_var_bindings < 0 ||
      (n_var_bindings > 0 && !var_bindings))
    return -1;
  if (n_inputs != jit->n_inputs) return -1;

  bool timing = poly_debug_at_least(7);
  double t0 = timing ? poly_now_ms() : 0.0;
  int ret = -1;
  PolyUOp *current_inputs_stack[16];
  PolyUOp **current_inputs = current_inputs_stack;
  if (n_inputs > (int)(sizeof(current_inputs_stack) / sizeof(current_inputs_stack[0]))) {
    current_inputs = calloc((size_t)n_inputs, sizeof(*current_inputs));
    if (!current_inputs) return -1;
  }

  for (int i = 0; i < n_inputs; i++) {
    current_inputs[i] = poly_jit_tensor_buffer(inputs[i]);
    if (!current_inputs[i] || !poly_jit_input_matches_spec(jit, i, inputs[i], current_inputs[i]))
      goto cleanup;
  }
  double t_inputs = timing ? poly_now_ms() : 0.0;

  if (poly_jit_inputs_are_captured(jit, current_inputs)) {
    ret = poly_jit_run_captured_linear(jit, var_bindings, n_var_bindings);
    if (timing) {
      double t_done = poly_now_ms();
      fprintf(
          stderr,
          "[polygrad:jit_run] path=captured inputs=%d input_check=%.3fms run=%.3fms "
          "total=%.3fms ret=%d\n",
          n_inputs, t_inputs - t0, t_done - t_inputs, t_done - t0, ret
      );
      fflush(stderr);
    }
    goto cleanup;
  }

  double t_replay0 = timing ? poly_now_ms() : 0.0;
  PolySchedule *replay = poly_jit_build_replay_schedule(jit, current_inputs);
  if (!replay) goto cleanup;
  double t_replay = timing ? poly_now_ms() : 0.0;
  int run_ret = poly_run_schedule(jit->ctx, replay, var_bindings, n_var_bindings);
  double t_run = timing ? poly_now_ms() : 0.0;
  poly_schedule_free(replay);
  if (run_ret != 0) goto cleanup;
  ret = 0;
  if (timing) {
    fprintf(
        stderr,
        "[polygrad:jit_run] path=replay inputs=%d input_check=%.3fms build=%.3fms run=%.3fms "
        "total=%.3fms ret=%d\n",
        n_inputs, t_inputs - t0, t_replay - t_replay0, t_run - t_replay, t_run - t0, ret
    );
    fflush(stderr);
  }

cleanup:
  if (current_inputs != current_inputs_stack) free(current_inputs);
  return ret;
}

int poly_jit_run(PolyJit *jit, PolyTensor **inputs, int n_inputs) {
  return poly_jit_run_with_vars(jit, inputs, n_inputs, NULL, 0);
}
