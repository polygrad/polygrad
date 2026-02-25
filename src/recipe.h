/*
 * recipe.h — Pre-compiled training plans for llm.c-style training recipes
 *
 * A PolyTrainPlan captures the full training pipeline (forward + loss +
 * backward + optimizer) as pre-compiled kernels.  After the first compile,
 * subsequent training steps skip scheduling and compilation entirely —
 * just swap data pointers and dispatch.
 *
 * Usage:
 *   PolyTrainPlan *plan = poly_train_plan_create(ctx, loss_sink,
 *       param_bufs, param_datas, n_params, lr);
 *   for (int i = 0; i < n_iters; i++) {
 *       float loss = poly_train_step(plan, x_data, y_data);
 *   }
 *   poly_train_plan_free(plan);
 */

#ifndef POLY_RECIPE_H
#define POLY_RECIPE_H

#include "frontend.h"
#include "sched.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum buffers/kernels in a training plan */
#define POLY_PLAN_MAX_PARAMS  64
#define POLY_PLAN_MAX_KERNELS 128

/* ── Training Plan ─────────────────────────────────────────────────── */

typedef struct {
  PolyCtx *ctx;

  /* Model parameters */
  int n_params;
  PolyUOp *param_bufs[POLY_PLAN_MAX_PARAMS];    /* BUFFER UOps for parameters */
  float *param_datas[POLY_PLAN_MAX_PARAMS];      /* Host memory for parameters */
  int param_sizes[POLY_PLAN_MAX_PARAMS];         /* Sizes in floats */

  /* Input/output buffers */
  PolyUOp *x_buf;        /* Input data BUFFER UOp */
  PolyUOp *y_buf;        /* Target data BUFFER UOp */
  int x_size;            /* Input size in floats */
  int y_size;            /* Target size in floats */

  /* Loss output */
  PolyUOp *loss_buf;     /* Loss scalar BUFFER UOp */
  float loss_data[1];    /* Loss scalar value */

  /* Forward + loss realize */
  PolyUOp *fwd_sink;     /* SINK for forward + loss */

  /* Gradient realizes (one per parameter) */
  PolyUOp *grad_sinks[POLY_PLAN_MAX_PARAMS];
  PolyUOp *grad_bufs[POLY_PLAN_MAX_PARAMS];
  float *grad_datas[POLY_PLAN_MAX_PARAMS];

  /* SGD update realizes (one per parameter) */
  PolyUOp *update_sinks[POLY_PLAN_MAX_PARAMS];
  PolyUOp *update_bufs[POLY_PLAN_MAX_PARAMS];
  float *update_datas[POLY_PLAN_MAX_PARAMS];

  float lr;

  /* Pre-compilation state: set after poly_train_plan_compile() */
  int compiled;
} PolyTrainPlan;

/* Create a training plan from a loss computation graph.
 *
 * loss_uop:   The UOp computing the scalar loss (before STORE/SINK wrapping).
 * x_buf, y_buf: BUFFER UOps for input data and target data.
 * param_bufs: Array of BUFFER UOps for model parameters.
 * param_datas: Array of host float* for parameter data.
 * param_sizes: Array of sizes (in floats) for each parameter.
 * n_params:   Number of parameters.
 * lr:         Learning rate for SGD.
 *
 * Builds the full pipeline: forward+loss, gradients, SGD updates. */
PolyTrainPlan *poly_train_plan_create(
    PolyCtx *ctx,
    PolyUOp *loss_uop,
    PolyUOp *x_buf, int x_size,
    PolyUOp *y_buf, int y_size,
    PolyUOp **param_bufs, float **param_datas, int *param_sizes,
    int n_params,
    float lr);

/* Pre-compile all kernels in the training plan.
 * After this call, poly_train_step() skips scheduling and compilation. */
int poly_train_plan_compile(PolyTrainPlan *plan);

/* Execute one training step: forward + backward + SGD update.
 * x_data, y_data: pointers to input and target data for this batch.
 * Returns the loss value. */
float poly_train_step(PolyTrainPlan *plan, float *x_data, float *y_data);

/* Free a training plan and all associated resources. */
void poly_train_plan_free(PolyTrainPlan *plan);

/* ── Data Loader ───────────────────────────────────────────────────── */

typedef struct {
  float *data;           /* mmap'd or malloc'd data */
  int n_samples;         /* Total number of samples */
  int sample_size;       /* Elements per sample (input + label) */
  int input_size;        /* Elements per input */
  int label_size;        /* Elements per label */
  int batch_size;        /* Batch size */
  int current;           /* Current sample index */
  int *indices;          /* Shuffle index array */
  float *x_batch;        /* Contiguous batch buffer for inputs */
  float *y_batch;        /* Contiguous batch buffer for labels */
  int fd;               /* File descriptor for mmap (-1 if malloc'd) */
  size_t file_size;     /* File size for munmap */
} PolyDataLoader;

typedef struct {
  float *x;             /* Pointer to input batch data */
  float *y;             /* Pointer to label batch data */
  int batch_size;       /* Actual batch size (may be smaller at end) */
} PolyBatch;

/* Open a binary float32 data file.
 * Format: [sample0_x..., sample0_y..., sample1_x..., sample1_y..., ...]
 * Each sample has input_size + label_size floats. */
PolyDataLoader *poly_dataloader_open(const char *path,
                                      int batch_size,
                                      int input_size,
                                      int label_size);

/* Create a dataloader from in-memory data (no file). */
PolyDataLoader *poly_dataloader_from_memory(float *data,
                                             int n_samples,
                                             int batch_size,
                                             int input_size,
                                             int label_size);

/* Get the next batch. Returns {NULL, NULL, 0} at end of epoch. */
PolyBatch poly_dataloader_next(PolyDataLoader *loader);

/* Reset to beginning (optionally shuffle). */
void poly_dataloader_reset(PolyDataLoader *loader, int shuffle);

/* Shuffle using Fisher-Yates. */
void poly_dataloader_shuffle(PolyDataLoader *loader);

/* Close and free. */
void poly_dataloader_close(PolyDataLoader *loader);

/* ── Model Builders ────────────────────────────────────────────────── */

typedef struct {
  PolyCtx *ctx;
  int n_layers;
  int *layer_sizes;      /* [input, hidden1, ..., output] */
  int n_params;
  PolyUOp **param_bufs;  /* BUFFER UOps for all weights + biases */
  float **param_datas;    /* Host data for all parameters */
  int *param_sizes;       /* Size of each parameter */
} PolyMLP;

/* Create an MLP with the given layer sizes.
 * layer_sizes: array of n_layers sizes [input_dim, hidden1, ..., output_dim].
 * Initializes parameters with Xavier uniform. */
PolyMLP *poly_mlp_create(PolyCtx *ctx, int *layer_sizes, int n_layers);

/* Forward pass: returns the output UOp (before loss).
 * batch_size: number of samples in the batch.
 * x has shape (batch_size, layer_sizes[0]).
 * Returns UOp with shape (batch_size, layer_sizes[n_layers-1]). */
PolyUOp *poly_mlp_forward(PolyMLP *model, PolyUOp *x, int batch_size);

/* Free MLP and parameter data. */
void poly_mlp_free(PolyMLP *model);

#ifdef __cplusplus
}
#endif

#endif /* POLY_RECIPE_H */
