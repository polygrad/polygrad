/* Current Tinygrad runtime/graph/cuda.py CUDAGraph driver boundary. */

#ifndef POLY_RUNTIME_GRAPH_CUDA_H
#define POLY_RUNTIME_GRAPH_CUDA_H

#include <stdbool.h>
#include <stddef.h>

typedef struct PolyCudaProgram PolyCudaProgram;
typedef struct PolyCudaGraph PolyCudaGraph;

typedef enum {
  POLY_CUDA_GRAPH_PROGRAM,
  POLY_CUDA_GRAPH_COPY,
} PolyCudaGraphCallKind;

/* C ABI storage for one PROGRAM or device-to-device COPY graph node. */
typedef struct {
  PolyCudaGraphCallKind kind;
  const int *dependencies;
  int n_dependencies;
  union {
    struct {
      PolyCudaProgram *program;
      void **args;
      int n_buffer_args;
      int n_args;
      int grid[3];
      int block[3];
    } program;
    struct {
      void *dst;
      void *src;
      size_t nbytes;
    } copy;
  } value;
} PolyCudaGraphCallSpec;

bool poly_cuda_graph_available(void);
PolyCudaGraph *poly_cuda_graph_create(const PolyCudaGraphCallSpec *specs, int n_specs);
int poly_cuda_graph_update(PolyCudaGraph *graph, const PolyCudaGraphCallSpec *specs, int n_specs);
int poly_cuda_graph_launch(PolyCudaGraph *graph);
void poly_cuda_graph_destroy(PolyCudaGraph *graph);

#endif /* POLY_RUNTIME_GRAPH_CUDA_H */
