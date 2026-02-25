/*
 * recipe.c — Pre-compiled training plans and model builders
 *
 * Implements llm.c-style compiled training recipes using polygrad's
 * UOp infrastructure.  A PolyTrainPlan pre-compiles all kernels
 * (forward + backward + SGD) so the training loop is just pointer
 * swaps and kernel dispatch.
 */

#include "recipe.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ── Training Plan ─────────────────────────────────────────────────── */

PolyTrainPlan *poly_train_plan_create(
    PolyCtx *ctx,
    PolyUOp *loss_uop,
    PolyUOp *x_buf, int x_size,
    PolyUOp *y_buf, int y_size,
    PolyUOp **param_bufs, float **param_datas, int *param_sizes,
    int n_params,
    float lr)
{
  if (n_params > POLY_PLAN_MAX_PARAMS) {
    fprintf(stderr, "poly_train_plan_create: too many params (%d > %d)\n",
            n_params, POLY_PLAN_MAX_PARAMS);
    return NULL;
  }

  PolyTrainPlan *plan = calloc(1, sizeof(PolyTrainPlan));
  plan->ctx = ctx;
  plan->n_params = n_params;
  plan->x_buf = x_buf;
  plan->y_buf = y_buf;
  plan->x_size = x_size;
  plan->y_size = y_size;
  plan->lr = lr;

  for (int i = 0; i < n_params; i++) {
    plan->param_bufs[i] = param_bufs[i];
    plan->param_datas[i] = param_datas[i];
    plan->param_sizes[i] = param_sizes[i];
  }

  /* 1. Forward + loss SINK */
  plan->loss_buf = poly_buffer_f32(ctx, 1);
  PolyUOp *loss_store = poly_store_val(ctx, plan->loss_buf, loss_uop);
  plan->fwd_sink = poly_sink1(ctx, loss_store);

  /* 2. Gradient SINKs (one per parameter) */
  for (int i = 0; i < n_params; i++) {
    PolyUOp *grad_uop = poly_grad(ctx, loss_uop, param_bufs[i]);
    if (!grad_uop) {
      fprintf(stderr, "poly_train_plan_create: poly_grad failed for param %d\n", i);
      free(plan);
      return NULL;
    }

    plan->grad_bufs[i] = poly_buffer_f32(ctx, param_sizes[i]);
    plan->grad_datas[i] = calloc(param_sizes[i], sizeof(float));
    PolyUOp *grad_store = poly_store_val(ctx, plan->grad_bufs[i], grad_uop);
    plan->grad_sinks[i] = poly_sink1(ctx, grad_store);
  }

  /* 3. SGD update SINKs: param = param - lr * grad */
  for (int i = 0; i < n_params; i++) {
    PolyUOp *lr_const = poly_const_float(ctx, (double)lr);
    PolyUOp *scaled_grad = poly_alu2(ctx, POLY_OP_MUL,
                                       plan->grad_bufs[i], lr_const);
    PolyUOp *updated = poly_alu2(ctx, POLY_OP_SUB,
                                   param_bufs[i], scaled_grad);

    plan->update_bufs[i] = poly_buffer_f32(ctx, param_sizes[i]);
    plan->update_datas[i] = calloc(param_sizes[i], sizeof(float));
    PolyUOp *update_store = poly_store_val(ctx, plan->update_bufs[i], updated);
    plan->update_sinks[i] = poly_sink1(ctx, update_store);
  }

  return plan;
}

int poly_train_plan_compile(PolyTrainPlan *plan) {
  if (!plan) return -1;

  /* Helper: collect all buffer bindings for a sink */
  #define MAX_PLAN_BINDINGS 128
  PolyBufferBinding bindings[MAX_PLAN_BINDINGS];
  int nb;

  /* Compile forward + loss */
  nb = 0;
  bindings[nb++] = (PolyBufferBinding){ plan->x_buf, NULL }; /* data set at step time */
  bindings[nb++] = (PolyBufferBinding){ plan->y_buf, NULL };
  for (int i = 0; i < plan->n_params; i++)
    bindings[nb++] = (PolyBufferBinding){ plan->param_bufs[i], plan->param_datas[i] };
  bindings[nb++] = (PolyBufferBinding){ plan->loss_buf, plan->loss_data };

  /* Need to provide valid data for compilation to succeed.
   * Allocate dummy input/output data. */
  float *dummy_x = calloc(plan->x_size, sizeof(float));
  float *dummy_y = calloc(plan->y_size, sizeof(float));
  bindings[0].data = dummy_x;
  bindings[1].data = dummy_y;

  int ret = poly_realize(plan->ctx, plan->fwd_sink, bindings, nb);
  if (ret != 0) {
    fprintf(stderr, "poly_train_plan_compile: forward realize failed\n");
    free(dummy_x); free(dummy_y);
    return -1;
  }

  /* Compile gradient realizes */
  for (int i = 0; i < plan->n_params; i++) {
    nb = 0;
    bindings[nb++] = (PolyBufferBinding){ plan->x_buf, dummy_x };
    bindings[nb++] = (PolyBufferBinding){ plan->y_buf, dummy_y };
    for (int j = 0; j < plan->n_params; j++)
      bindings[nb++] = (PolyBufferBinding){ plan->param_bufs[j], plan->param_datas[j] };
    bindings[nb++] = (PolyBufferBinding){ plan->grad_bufs[i], plan->grad_datas[i] };
    /* Add loss buf in case gradient references it */
    bindings[nb++] = (PolyBufferBinding){ plan->loss_buf, plan->loss_data };

    ret = poly_realize(plan->ctx, plan->grad_sinks[i], bindings, nb);
    if (ret != 0) {
      fprintf(stderr, "poly_train_plan_compile: gradient %d realize failed\n", i);
      free(dummy_x); free(dummy_y);
      return -1;
    }
  }

  /* Compile SGD update realizes */
  for (int i = 0; i < plan->n_params; i++) {
    nb = 0;
    bindings[nb++] = (PolyBufferBinding){ plan->param_bufs[i], plan->param_datas[i] };
    bindings[nb++] = (PolyBufferBinding){ plan->grad_bufs[i], plan->grad_datas[i] };
    bindings[nb++] = (PolyBufferBinding){ plan->update_bufs[i], plan->update_datas[i] };

    ret = poly_realize(plan->ctx, plan->update_sinks[i], bindings, nb);
    if (ret != 0) {
      fprintf(stderr, "poly_train_plan_compile: update %d realize failed\n", i);
      free(dummy_x); free(dummy_y);
      return -1;
    }
  }

  free(dummy_x);
  free(dummy_y);
  plan->compiled = 1;
  return 0;
}

float poly_train_step(PolyTrainPlan *plan, float *x_data, float *y_data) {
  if (!plan) return -1.0f;

  PolyBufferBinding bindings[MAX_PLAN_BINDINGS];
  int nb;

  /* 1. Forward + loss */
  nb = 0;
  bindings[nb++] = (PolyBufferBinding){ plan->x_buf, x_data };
  bindings[nb++] = (PolyBufferBinding){ plan->y_buf, y_data };
  for (int i = 0; i < plan->n_params; i++)
    bindings[nb++] = (PolyBufferBinding){ plan->param_bufs[i], plan->param_datas[i] };
  bindings[nb++] = (PolyBufferBinding){ plan->loss_buf, plan->loss_data };

  poly_realize(plan->ctx, plan->fwd_sink, bindings, nb);

  /* 2. Backward: compute gradients */
  for (int i = 0; i < plan->n_params; i++) {
    nb = 0;
    bindings[nb++] = (PolyBufferBinding){ plan->x_buf, x_data };
    bindings[nb++] = (PolyBufferBinding){ plan->y_buf, y_data };
    for (int j = 0; j < plan->n_params; j++)
      bindings[nb++] = (PolyBufferBinding){ plan->param_bufs[j], plan->param_datas[j] };
    bindings[nb++] = (PolyBufferBinding){ plan->grad_bufs[i], plan->grad_datas[i] };
    bindings[nb++] = (PolyBufferBinding){ plan->loss_buf, plan->loss_data };

    poly_realize(plan->ctx, plan->grad_sinks[i], bindings, nb);
  }

  /* 3. SGD update: param = param - lr * grad */
  for (int i = 0; i < plan->n_params; i++) {
    nb = 0;
    bindings[nb++] = (PolyBufferBinding){ plan->param_bufs[i], plan->param_datas[i] };
    bindings[nb++] = (PolyBufferBinding){ plan->grad_bufs[i], plan->grad_datas[i] };
    bindings[nb++] = (PolyBufferBinding){ plan->update_bufs[i], plan->update_datas[i] };

    poly_realize(plan->ctx, plan->update_sinks[i], bindings, nb);

    /* Copy updated params back */
    memcpy(plan->param_datas[i], plan->update_datas[i],
           plan->param_sizes[i] * sizeof(float));
  }

  return plan->loss_data[0];
}

void poly_train_plan_free(PolyTrainPlan *plan) {
  if (!plan) return;
  for (int i = 0; i < plan->n_params; i++) {
    free(plan->grad_datas[i]);
    free(plan->update_datas[i]);
  }
  free(plan);
}

/* ── Data Loader ───────────────────────────────────────────────────── */

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

PolyDataLoader *poly_dataloader_open(const char *path,
                                      int batch_size,
                                      int input_size,
                                      int label_size) {
  int fd = open(path, O_RDONLY);
  if (fd < 0) {
    fprintf(stderr, "poly_dataloader_open: cannot open %s\n", path);
    return NULL;
  }

  struct stat st;
  fstat(fd, &st);
  size_t file_size = st.st_size;

  int sample_size = input_size + label_size;
  int n_samples = (int)(file_size / (sample_size * sizeof(float)));

  float *data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (data == MAP_FAILED) {
    fprintf(stderr, "poly_dataloader_open: mmap failed\n");
    close(fd);
    return NULL;
  }

  PolyDataLoader *loader = calloc(1, sizeof(PolyDataLoader));
  loader->data = data;
  loader->n_samples = n_samples;
  loader->sample_size = sample_size;
  loader->input_size = input_size;
  loader->label_size = label_size;
  loader->batch_size = batch_size;
  loader->current = 0;
  loader->fd = fd;
  loader->file_size = file_size;

  /* Sequential index array */
  loader->indices = malloc(n_samples * sizeof(int));
  for (int i = 0; i < n_samples; i++) loader->indices[i] = i;

  /* Contiguous batch buffers for gather */
  loader->x_batch = malloc(batch_size * input_size * sizeof(float));
  loader->y_batch = malloc(batch_size * label_size * sizeof(float));

  return loader;
}

PolyDataLoader *poly_dataloader_from_memory(float *data,
                                             int n_samples,
                                             int batch_size,
                                             int input_size,
                                             int label_size) {
  PolyDataLoader *loader = calloc(1, sizeof(PolyDataLoader));
  loader->data = data;
  loader->n_samples = n_samples;
  loader->sample_size = input_size + label_size;
  loader->input_size = input_size;
  loader->label_size = label_size;
  loader->batch_size = batch_size;
  loader->current = 0;
  loader->fd = -1;
  loader->file_size = 0;

  loader->indices = malloc(n_samples * sizeof(int));
  for (int i = 0; i < n_samples; i++) loader->indices[i] = i;

  loader->x_batch = malloc(batch_size * input_size * sizeof(float));
  loader->y_batch = malloc(batch_size * label_size * sizeof(float));

  return loader;
}

PolyBatch poly_dataloader_next(PolyDataLoader *loader) {
  PolyBatch batch = { NULL, NULL, 0 };
  if (!loader || loader->current >= loader->n_samples) return batch;

  int remaining = loader->n_samples - loader->current;
  int bs = remaining < loader->batch_size ? remaining : loader->batch_size;

  /* Gather samples by index into contiguous batch buffers */
  for (int i = 0; i < bs; i++) {
    int idx = loader->indices[loader->current + i];
    float *sample = loader->data + idx * loader->sample_size;
    memcpy(loader->x_batch + i * loader->input_size,
           sample, loader->input_size * sizeof(float));
    memcpy(loader->y_batch + i * loader->label_size,
           sample + loader->input_size, loader->label_size * sizeof(float));
  }

  batch.x = loader->x_batch;
  batch.y = loader->y_batch;
  batch.batch_size = bs;

  loader->current += bs;
  return batch;
}

void poly_dataloader_reset(PolyDataLoader *loader, int shuffle) {
  if (!loader) return;
  loader->current = 0;
  if (shuffle) poly_dataloader_shuffle(loader);
}

void poly_dataloader_shuffle(PolyDataLoader *loader) {
  if (!loader) return;
  for (int i = loader->n_samples - 1; i > 0; i--) {
    int j = rand() % (i + 1);
    int tmp = loader->indices[i];
    loader->indices[i] = loader->indices[j];
    loader->indices[j] = tmp;
  }
}

void poly_dataloader_close(PolyDataLoader *loader) {
  if (!loader) return;
  if (loader->fd >= 0) {
    munmap(loader->data, loader->file_size);
    close(loader->fd);
  }
  free(loader->indices);
  free(loader->x_batch);
  free(loader->y_batch);
  free(loader);
}

/* ── MLP Model Builder ─────────────────────────────────────────────── */

PolyMLP *poly_mlp_create(PolyCtx *ctx, int *layer_sizes, int n_layers) {
  if (n_layers < 2) {
    fprintf(stderr, "poly_mlp_create: need at least 2 layers\n");
    return NULL;
  }

  int n_linear = n_layers - 1;
  int n_params = n_linear * 2; /* weight + bias per layer */

  PolyMLP *model = calloc(1, sizeof(PolyMLP));
  model->ctx = ctx;
  model->n_layers = n_layers;
  model->layer_sizes = malloc(n_layers * sizeof(int));
  memcpy(model->layer_sizes, layer_sizes, n_layers * sizeof(int));
  model->n_params = n_params;
  model->param_bufs = calloc(n_params, sizeof(PolyUOp *));
  model->param_datas = calloc(n_params, sizeof(float *));
  model->param_sizes = calloc(n_params, sizeof(int));

  srand(42);

  for (int l = 0; l < n_linear; l++) {
    int in_dim = layer_sizes[l];
    int out_dim = layer_sizes[l + 1];

    /* Xavier uniform initialization */
    float bound = sqrtf(6.0f / (float)(in_dim + out_dim));

    /* Weight: out_dim × in_dim */
    int w_size = out_dim * in_dim;
    float *w_data = malloc(w_size * sizeof(float));
    for (int i = 0; i < w_size; i++)
      w_data[i] = ((float)rand() / RAND_MAX) * 2.0f * bound - bound;

    PolyUOp *w_buf = poly_buffer_f32(ctx, w_size);

    /* Realize weight to make it a leaf tensor */
    PolyUOp *w_store = poly_store_val(ctx, w_buf, w_buf);
    PolyUOp *w_sink = poly_sink1(ctx, w_store);
    PolyBufferBinding wb = { w_buf, w_data };
    poly_realize(ctx, w_sink, &wb, 1);

    model->param_bufs[l * 2] = w_buf;
    model->param_datas[l * 2] = w_data;
    model->param_sizes[l * 2] = w_size;

    /* Bias: out_dim */
    float *b_data = calloc(out_dim, sizeof(float));
    for (int i = 0; i < out_dim; i++)
      b_data[i] = ((float)rand() / RAND_MAX) * 2.0f * bound - bound;

    PolyUOp *b_buf = poly_buffer_f32(ctx, out_dim);

    PolyUOp *b_store = poly_store_val(ctx, b_buf, b_buf);
    PolyUOp *b_sink = poly_sink1(ctx, b_store);
    PolyBufferBinding bb = { b_buf, b_data };
    poly_realize(ctx, b_sink, &bb, 1);

    model->param_bufs[l * 2 + 1] = b_buf;
    model->param_datas[l * 2 + 1] = b_data;
    model->param_sizes[l * 2 + 1] = out_dim;
  }

  return model;
}

PolyUOp *poly_mlp_forward(PolyMLP *model, PolyUOp *x, int batch_size) {
  PolyCtx *ctx = model->ctx;
  int n_linear = model->n_layers - 1;

  for (int l = 0; l < n_linear; l++) {
    int in_dim = model->layer_sizes[l];
    int out_dim = model->layer_sizes[l + 1];

    PolyUOp *w = model->param_bufs[l * 2];
    PolyUOp *b = model->param_bufs[l * 2 + 1];

    /* Reshape weight from flat (out_dim*in_dim,) to 2D (out_dim, in_dim) */
    int64_t w_2d_shape[] = { out_dim, in_dim };
    PolyUOp *w_2d = poly_reshape(ctx, w, w_2d_shape, 2);

    /* Transpose from (out_dim, in_dim) to (in_dim, out_dim) */
    int64_t w_perm[] = { 1, 0 };
    PolyUOp *wt = poly_permute(ctx, w_2d, w_perm, 2);

    /* x @ wt: (batch, in_dim) × (in_dim, out_dim) → (batch, out_dim) */
    int64_t x_shape[] = { batch_size, in_dim };
    int64_t wt_shape[] = { in_dim, out_dim };
    int64_t dot_out_shape[16];
    int dot_out_ndim;
    x = poly_dot(ctx, x, x_shape, 2, wt, wt_shape, 2,
                 dot_out_shape, &dot_out_ndim);

    /* Add bias: reshape (out_dim) → (1, out_dim), expand → (batch, out_dim) */
    int64_t b_reshape[] = { 1, out_dim };
    PolyUOp *br = poly_reshape(ctx, b, b_reshape, 2);
    int64_t b_expand[] = { batch_size, out_dim };
    PolyUOp *be = poly_expand(ctx, br, b_expand, 2);
    x = poly_alu2(ctx, POLY_OP_ADD, x, be);

    /* Apply relu (except last layer) */
    if (l < n_linear - 1) {
      x = poly_relu(ctx, x);
    }
  }

  return x;
}

void poly_mlp_free(PolyMLP *model) {
  if (!model) return;
  for (int i = 0; i < model->n_params; i++)
    free(model->param_datas[i]);
  free(model->param_bufs);
  free(model->param_datas);
  free(model->param_sizes);
  free(model->layer_sizes);
  free(model);
}
