/* Stateful installed-SDK example: owns graph terms, never executes them. */
#include "polygrad.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
  PolyCtx *ctx;
  PolyTensor *term;
  PolyTensor *input;
  char *name;
  int id;
} ExampleBuilder;
static int live, serial, closed_order;

int example_build(PolyCtx *ctx, PolyTensor **inputs, PolyTensor **outputs) {
  outputs[0] = poly_tensor_retain(inputs[0]);
  return outputs[0] != NULL;
}

ExampleBuilder *example_new(PolyCtx *ctx, PolyTensor *term, const char *name) {
  ExampleBuilder *b = calloc(1, sizeof(*b));
  if (!b) return NULL;
  b->name = malloc(strlen(name) + 1);
  if (!b->name) {
    free(b);
    return NULL;
  }
  strcpy(b->name, name);
  b->ctx = ctx;
  b->term = poly_tensor_retain(term);
  b->input = poly_tensor_retain(term);
  b->id = ++serial;
  live++;
  return b;
}
void example_close(ExampleBuilder *b) {
  if (!b) return;
  closed_order = closed_order * 10 + b->id;
  live--;
  poly_tensor_release(b->term);
  poly_tensor_release(b->input);
  free(b->name);
  free(b);
}
int example_add(ExampleBuilder *b, PolyTensor *term, const char *name) {
  PolyTensor *sum = poly_tensor_alu2(b->ctx, POLY_OP_ADD, b->term, term);
  if (!sum) return -1;
  if (!strcmp(name, "fail")) {
    poly_tensor_release(sum);
    return -1;
  }
  poly_tensor_release(b->term);
  b->term = sum;
  return 0;
}
PolyTensor *example_finish(ExampleBuilder *b) {
  return poly_tensor_exp(b->ctx, b->term);
}
PolyTensor *example_input(ExampleBuilder *b) {
  return poly_tensor_retain(b->input);
}
const char *example_name(ExampleBuilder *b) {
  return b->name;
}
char *example_name_copy(ExampleBuilder *b) {
  char *s = malloc(strlen(b->name) + 1);
  if (s) strcpy(s, b->name);
  return s;
}
void example_free_string(char *s) {
  free(s);
}
int example_live(void) {
  return live;
}
int example_order(void) {
  return closed_order;
}
int example_reset(void) {
  if (live) return -1;
  serial = closed_order = 0;
  return 0;
}
