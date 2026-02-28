/**
 * ffi.js — koffi bindings to libpolygrad.so
 * Uses only the frontend.h surface to avoid passing PolyArg/PolyDType structs.
 */

'use strict'

const koffi = require('koffi')
const path = require('path')

// Find libpolygrad.so
const defaultPath = path.resolve(__dirname, '..', '..', 'build', 'libpolygrad.so')
const libPath = process.env.POLYGRAD_LIB || defaultPath

const lib = koffi.load(libPath)

// --- Function declarations ---

const poly_ctx_new = lib.func('void *poly_ctx_new()')
const poly_ctx_destroy = lib.func('void poly_ctx_destroy(void *ctx)')

const poly_op_count = lib.func('int poly_op_count()')
const poly_op_name = lib.func('const char *poly_op_name(int op)')

const poly_const_float = lib.func('void *poly_const_float(void *ctx, double value)')
const poly_const_int = lib.func('void *poly_const_int(void *ctx, int64_t value)')

const poly_alu1 = lib.func('void *poly_alu1(void *ctx, int op, void *src)')
const poly_alu2 = lib.func('void *poly_alu2(void *ctx, int op, void *a, void *b)')
const poly_alu3 = lib.func('void *poly_alu3(void *ctx, int op, void *a, void *b, void *c)')

const poly_store_val = lib.func('void *poly_store_val(void *ctx, void *buf, void *value)')
const poly_sink1 = lib.func('void *poly_sink1(void *ctx, void *store)')
const poly_sink_n = lib.func('void *poly_sink_n(void *ctx, void **stores, int n)')

const poly_buffer_f32 = lib.func('void *poly_buffer_f32(void *ctx, int64_t size)')

// Movement/Scheduling
const poly_reshape = lib.func('void *poly_reshape(void *ctx, void *src, int64_t *dims, int ndim)')
const poly_expand = lib.func('void *poly_expand(void *ctx, void *src, int64_t *dims, int ndim)')
const poly_reduce_axis = lib.func('void *poly_reduce_axis(void *ctx, int op, void *src, int64_t *axes, int n)')
const poly_permute = lib.func('void *poly_permute(void *ctx, void *src, int64_t *perm, int ndim)')
const poly_shrink = lib.func('void *poly_shrink(void *ctx, void *src, void *pairs, int ndim)')
const poly_flip = lib.func('void *poly_flip(void *ctx, void *src, int64_t *axes, int n)')
const poly_pad = lib.func('void *poly_pad(void *ctx, void *src, void *pairs, int ndim)')

// Autograd
const poly_grad = lib.func('void *poly_grad(void *ctx, void *loss, void *wrt_buffer)')

// Realize — stateful builder pattern
const poly_realize_begin = lib.func('void poly_realize_begin(void *ctx)')
const poly_realize_bind = lib.func('void poly_realize_bind(void *ctx, void *buffer, void *data)')
const poly_realize_exec = lib.func('int poly_realize_exec(void *ctx, void *sink)')

// --- Composed elementwise ops ---
const poly_exp = lib.func('void *poly_exp(void *ctx, void *x)')
const poly_log = lib.func('void *poly_log(void *ctx, void *x)')
const poly_sin = lib.func('void *poly_sin(void *ctx, void *x)')
const poly_cos = lib.func('void *poly_cos(void *ctx, void *x)')
const poly_tan = lib.func('void *poly_tan(void *ctx, void *x)')
const poly_sigmoid = lib.func('void *poly_sigmoid(void *ctx, void *x)')
const poly_tanh_act = lib.func('void *poly_tanh_act(void *ctx, void *x)')
const poly_abs = lib.func('void *poly_abs(void *ctx, void *x)')
const poly_sign = lib.func('void *poly_sign(void *ctx, void *x)')
const poly_square = lib.func('void *poly_square(void *ctx, void *x)')
const poly_rsqrt = lib.func('void *poly_rsqrt(void *ctx, void *x)')
const poly_ceil = lib.func('void *poly_ceil(void *ctx, void *x)')
const poly_floor = lib.func('void *poly_floor(void *ctx, void *x)')
const poly_round_f = lib.func('void *poly_round_f(void *ctx, void *x)')
const poly_isinf = lib.func('void *poly_isinf(void *ctx, void *x)')
const poly_isnan = lib.func('void *poly_isnan(void *ctx, void *x)')

// Activations
const poly_relu = lib.func('void *poly_relu(void *ctx, void *x)')
const poly_relu6 = lib.func('void *poly_relu6(void *ctx, void *x)')
const poly_leaky_relu = lib.func('void *poly_leaky_relu(void *ctx, void *x, double neg_slope)')
const poly_gelu = lib.func('void *poly_gelu(void *ctx, void *x)')
const poly_quick_gelu = lib.func('void *poly_quick_gelu(void *ctx, void *x)')
const poly_silu = lib.func('void *poly_silu(void *ctx, void *x)')
const poly_elu = lib.func('void *poly_elu(void *ctx, void *x, double alpha)')
const poly_softplus = lib.func('void *poly_softplus(void *ctx, void *x, double beta)')
const poly_mish = lib.func('void *poly_mish(void *ctx, void *x)')
const poly_hardtanh = lib.func('void *poly_hardtanh(void *ctx, void *x, double min_val, double max_val)')
const poly_hardswish = lib.func('void *poly_hardswish(void *ctx, void *x)')
const poly_hardsigmoid = lib.func('void *poly_hardsigmoid(void *ctx, void *x)')

// Comparisons
const poly_eq = lib.func('void *poly_eq(void *ctx, void *a, void *b)')
const poly_ne = lib.func('void *poly_ne(void *ctx, void *a, void *b)')
const poly_gt = lib.func('void *poly_gt(void *ctx, void *a, void *b)')
const poly_ge = lib.func('void *poly_ge(void *ctx, void *a, void *b)')
const poly_le = lib.func('void *poly_le(void *ctx, void *a, void *b)')
const poly_where_op = lib.func('void *poly_where_op(void *ctx, void *cond, void *x, void *y)')
const poly_maximum = lib.func('void *poly_maximum(void *ctx, void *a, void *b)')
const poly_minimum = lib.func('void *poly_minimum(void *ctx, void *a, void *b)')
const poly_clamp = lib.func('void *poly_clamp(void *ctx, void *x, double lo, double hi)')

// Shape-aware ops
const poly_max_reduce = lib.func('void *poly_max_reduce(void *ctx, void *x, int64_t *shape, int ndim, int axis, int keepdim, int64_t *out_shape, int *out_ndim)')
const poly_mean_reduce = lib.func('void *poly_mean_reduce(void *ctx, void *x, int64_t *shape, int ndim, int axis, int keepdim, int64_t *out_shape, int *out_ndim)')
const poly_dot = lib.func('void *poly_dot(void *ctx, void *x, int64_t *x_shape, int x_ndim, void *w, int64_t *w_shape, int w_ndim, int64_t *out_shape, int *out_ndim)')

// Einsum + Rearrange
const poly_einsum = lib.func('void *poly_einsum(void *ctx, const char *formula, void **tensors, void **shapes, int *ndims, int n, int64_t *out_shape, int *out_ndim)')
const poly_rearrange = lib.func('void *poly_rearrange(void *ctx, const char *formula, void *x, int64_t *shape, int ndim, const char *axis_names, int64_t *axis_values, int n_axis_sizes, int64_t *out_shape, int *out_ndim)')

// Build op name -> int mapping at load time
const opCount = poly_op_count()
const OPS = {}
for (let i = 0; i < opCount; i++) {
  const name = poly_op_name(i)
  if (name) OPS[name] = i
}

module.exports = {
  poly_ctx_new, poly_ctx_destroy,
  poly_const_float, poly_const_int,
  poly_alu1, poly_alu2, poly_alu3,
  poly_store_val, poly_sink1, poly_sink_n,
  poly_buffer_f32,
  poly_reshape, poly_expand, poly_reduce_axis,
  poly_permute, poly_shrink, poly_flip, poly_pad,
  poly_grad, poly_realize_begin, poly_realize_bind, poly_realize_exec,
  // Composed ops
  poly_exp, poly_log, poly_sin, poly_cos, poly_tan,
  poly_sigmoid, poly_tanh_act, poly_abs, poly_sign, poly_square, poly_rsqrt,
  poly_ceil, poly_floor, poly_round_f, poly_isinf, poly_isnan,
  // Activations
  poly_relu, poly_relu6, poly_leaky_relu, poly_gelu, poly_quick_gelu, poly_silu,
  poly_elu, poly_softplus, poly_mish, poly_hardtanh, poly_hardswish, poly_hardsigmoid,
  // Comparisons
  poly_eq, poly_ne, poly_gt, poly_ge, poly_le,
  poly_where_op, poly_maximum, poly_minimum, poly_clamp,
  // Shape-aware
  poly_max_reduce, poly_mean_reduce, poly_dot,
  poly_einsum, poly_rearrange,
  OPS, koffi,

  // PolyInstance (instance.h)
  poly_instance_from_ir: lib.func('void *poly_instance_from_ir(const uint8_t *ir_data, int ir_len, const uint8_t *weights_data, int weights_len)'),
  poly_instance_free: lib.func('void poly_instance_free(void *inst)'),
  poly_instance_param_count: lib.func('int poly_instance_param_count(void *inst)'),
  poly_instance_param_name: lib.func('const char *poly_instance_param_name(void *inst, int i)'),
  poly_instance_param_shape: lib.func('int poly_instance_param_shape(void *inst, int i, int64_t *shape_out, int max_dims)'),
  poly_instance_param_data: lib.func('float *poly_instance_param_data(void *inst, int i, int64_t *numel_out)'),
  poly_instance_buf_count: lib.func('int poly_instance_buf_count(void *inst)'),
  poly_instance_buf_name: lib.func('const char *poly_instance_buf_name(void *inst, int i)'),
  poly_instance_buf_role: lib.func('int poly_instance_buf_role(void *inst, int i)'),
  poly_instance_buf_shape: lib.func('int poly_instance_buf_shape(void *inst, int i, int64_t *shape_out, int max_dims)'),
  poly_instance_buf_data: lib.func('float *poly_instance_buf_data(void *inst, int i, int64_t *numel_out)'),
  poly_instance_export_weights: lib.func('void *poly_instance_export_weights(void *inst, int *out_len)'),
  poly_instance_import_weights: lib.func('int poly_instance_import_weights(void *inst, const uint8_t *data, int len)'),
  poly_instance_export_ir: lib.func('void *poly_instance_export_ir(void *inst, int *out_len)'),
  poly_instance_forward: lib.func('int poly_instance_forward(void *inst, void *inputs, int n_inputs)'),
  poly_instance_train_step: lib.func('int poly_instance_train_step(void *inst, void *io, int n_io, float *loss_out)'),
  poly_instance_set_optimizer: lib.func('int poly_instance_set_optimizer(void *inst, int kind, float lr, float beta1, float beta2, float eps, float weight_decay)'),
  poly_mlp_instance: lib.func('void *poly_mlp_instance(const char *spec_json, int spec_len)'),

  // HF loader (modelzoo/hf_loader.c)
  poly_hf_load: lib.func('void *poly_hf_load(const char *config_json, int config_len, const uint8_t **weight_files, const int64_t *weight_lens, int n_weight_files, int max_batch, int max_seq_len)')
}
