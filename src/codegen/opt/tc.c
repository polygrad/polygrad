/* codegen/opt/tc.c -- Current Tinygrad TensorCore tables and helpers. */

#include "codegen/opt/tc.h"

#include <stdbool.h>
#include <string.h>

typedef struct {
  int dims[3];
  int threads;
  int elements_per_thread[3];
  const char *opts;
  const char *swizzle[2][3][POLY_TC_MAX_SWIZZLE];
  int swizzle_len[2][3];
} PolyTensorCoreLayout;

/* Current Tinygrad 2026-08-22/a9069c177a9d codegen/opt/tc.py:75-135. */
static const PolyTensorCoreLayout CUDA_81616 = {
    .dims = {8, 16, 16},
    .threads = 32,
    .elements_per_thread = {8, 4, 4},
    .opts = "u0l0l0l1l1l1u1",
    .swizzle =
        {
            {{"r1", "r2", "l2", "l3", "l4"}, {"u1", "r3"}, {"l0", "l1", "u0", "r0"}},
            {{"r1", "r2", "u0", "l0", "l1"}, {"r0", "r3"}, {"l2", "l3", "l4", "u1"}},
        },
    .swizzle_len = {{5, 2, 4}, {5, 2, 4}},
};

static const PolyTensorCoreLayout CUDA_81632_F8 = {
    .dims = {8, 16, 32},
    .threads = 32,
    .elements_per_thread = {16, 8, 4},
    .opts = "u0l0l0l1l1l1u1",
    .swizzle =
        {
            {{"r2", "r3", "l2", "l3", "l4"}, {"u1", "r4"}, {"l0", "l1", "u0", "r0", "r1"}},
            {{"r2", "r3", "u0", "l0", "l1"}, {"r1", "r4"}, {"l2", "l3", "l4", "u1", "r0"}},
        },
    .swizzle_len = {{5, 2, 5}, {5, 2, 5}},
};

static const PolyTensorCoreLayout CUDA_8168_F16 = {
    .dims = {8, 16, 8},
    .threads = 32,
    .elements_per_thread = {4, 2, 4},
    .opts = "u0l0l0l1l1l1u1",
    .swizzle =
        {
            {{"r1", "r2", "l2", "l3", "l4"}, {"r0", "u1"}, {"l0", "l1", "u0"}},
            {{"r1", "r2", "u0", "l0", "l1"}, {"u1", "r0"}, {"l2", "l3", "l4"}},
        },
    .swizzle_len = {{5, 2, 3}, {5, 2, 3}},
};

static const PolyTensorCoreLayout CUDA_8168_TF32 = {
    .dims = {8, 16, 8},
    .threads = 32,
    .elements_per_thread = {4, 2, 4},
    .opts = "u0l0l0l1l1l1u1",
    .swizzle =
        {
            {{"r0", "r1", "l2", "l3", "l4"}, {"u1", "r2"}, {"l0", "l1", "u0"}},
            {{"r0", "r1", "u0", "l0", "l1"}, {"u1", "r2"}, {"l2", "l3", "l4"}},
        },
    .swizzle_len = {{5, 2, 3}, {5, 2, 3}},
};

static const PolyTensorCoreLayout AMD_RDNA3 = {
    .dims = {16, 16, 16},
    .threads = 32,
    .elements_per_thread = {16, 16, 8},
    .opts = "l0l0l0l0l1u1u1u1",
    .swizzle =
        {
            {{"l4", "u0", "u1", "u2", "l0"}, {"r1", "r2", "r3"}, {"l1", "l2", "l3", "r0"}},
            {{"l0", "l1", "l2", "l3", "l4"}, {"r1", "r2", "r3"}, {"u0", "u1", "u2", "r0"}},
        },
    .swizzle_len = {{5, 3, 4}, {5, 3, 4}},
};

static const PolyTensorCoreLayout AMD_RDNA4 = {
    .dims = {16, 16, 16},
    .threads = 32,
    .elements_per_thread = {8, 8, 8},
    .opts = "l0l0l0l0u1u1u1l1",
    .swizzle =
        {
            {{"u0", "u1", "u2", "l4", "r2"}, {"r0", "r1", "r3"}, {"l0", "l1", "l2", "l3"}},
            {{"l0", "l1", "l2", "l3", "r2"}, {"r0", "r1", "r3"}, {"l4", "u0", "u1", "u2"}},
        },
    .swizzle_len = {{5, 3, 4}, {5, 3, 4}},
};

static const PolyTensorCoreLayout AMD_CDNA_161616 = {
    .dims = {16, 16, 16},
    .threads = 64,
    .elements_per_thread = {4, 4, 4},
    .opts = "l0l0l0l0u1u1l1l1",
    .swizzle =
        {
            {{"u0", "u1", "l4", "l5", "r2", "r3"}, {"r0", "r1"}, {"l0", "l1", "l2", "l3"}},
            {{"l0", "l1", "l2", "l3", "r2", "r3"}, {"r0", "r1"}, {"l4", "l5", "u0", "u1"}},
        },
    .swizzle_len = {{6, 2, 4}, {6, 2, 4}},
};

static const PolyTensorCoreLayout AMD_CDNA_161632 = {
    .dims = {16, 16, 32},
    .threads = 64,
    .elements_per_thread = {8, 8, 4},
    .opts = "l0l0l0l0u1u1l1l1",
    .swizzle =
        {
            {{"u0", "u1", "l4", "l5", "r3", "r4"}, {"r0", "r1"}, {"l0", "l1", "l2", "l3", "r2"}},
            {{"l0", "l1", "l2", "l3", "r3", "r4"}, {"r0", "r1"}, {"l4", "l5", "u0", "u1", "r2"}},
        },
    .swizzle_len = {{6, 2, 5}, {6, 2, 5}},
};

static const PolyTensorCoreLayout AMD_CDNA_1616128 = {
    .dims = {16, 16, 128},
    .threads = 64,
    .elements_per_thread = {32, 32, 4},
    .opts = "l0l0l0l0u1u1l1l1",
    .swizzle =
        {
            {{"u0", "u1", "l4", "l5", "r5", "r6"},
             {"r0", "r1"},
             {"l0", "l1", "l2", "l3", "r2", "r3", "r4"}},
            {{"l0", "l1", "l2", "l3", "r5", "r6"},
             {"r0", "r1"},
             {"l4", "l5", "u0", "u1", "r2", "r3", "r4"}},
        },
    .swizzle_len = {{6, 2, 7}, {6, 2, 7}},
};

static void init_tensor_core(
    PolyTensorCore *tc,
    const PolyTensorCoreLayout *layout,
    PolyDType dtype_in,
    PolyDType dtype_out
) {
  memset(tc, 0, sizeof(*tc));
  memcpy(tc->dims, layout->dims, sizeof(tc->dims));
  tc->threads = layout->threads;
  memcpy(tc->elements_per_thread, layout->elements_per_thread, sizeof(tc->elements_per_thread));
  tc->dtype_in = dtype_in;
  tc->dtype_out = dtype_out;
  tc->n_opts = (int)strlen(layout->opts) / 2;
  for (int i = 0; i < tc->n_opts; i++) {
    tc->opts[i].type = layout->opts[i * 2];
    tc->opts[i].dim = layout->opts[i * 2 + 1] - '0';
  }
  memcpy(tc->swizzle, layout->swizzle, sizeof(tc->swizzle));
  memcpy(tc->swizzle_len, layout->swizzle_len, sizeof(tc->swizzle_len));
}

static _Thread_local bool cuda_initialized;
static _Thread_local PolyTensorCore cuda_sm75[2];
static _Thread_local PolyTensorCore cuda_sm80[6];
static _Thread_local PolyTensorCore cuda_sm89[8];

static void init_cuda(void) {
  if (cuda_initialized) return;
  cuda_initialized = true;
  init_tensor_core(&cuda_sm75[0], &CUDA_8168_F16, POLY_FLOAT16, POLY_FLOAT32);
  init_tensor_core(&cuda_sm75[1], &CUDA_8168_F16, POLY_FLOAT16, POLY_FLOAT16);
  init_tensor_core(&cuda_sm80[0], &CUDA_81616, POLY_FLOAT16, POLY_FLOAT32);
  init_tensor_core(&cuda_sm80[1], &CUDA_81616, POLY_BFLOAT16, POLY_FLOAT32);
  init_tensor_core(&cuda_sm80[2], &CUDA_81616, POLY_FLOAT16, POLY_FLOAT16);
  memcpy(&cuda_sm80[3], cuda_sm75, sizeof(cuda_sm75));
  init_tensor_core(&cuda_sm80[5], &CUDA_8168_TF32, POLY_FLOAT32, POLY_FLOAT32);
  memcpy(cuda_sm89, cuda_sm80, sizeof(cuda_sm80));
  init_tensor_core(&cuda_sm89[6], &CUDA_81632_F8, POLY_FP8E4M3, POLY_FLOAT32);
  init_tensor_core(&cuda_sm89[7], &CUDA_81632_F8, POLY_FP8E5M2, POLY_FLOAT32);
}

const PolyTensorCore *poly_tc_get_cuda(int arch, int *count) {
  if (count) *count = 0;
  init_cuda();
  if (arch >= 89) {
    if (count) *count = 8;
    return cuda_sm89;
  }
  if (arch >= 80) {
    if (count) *count = 6;
    return cuda_sm80;
  }
  if (arch >= 75) {
    if (count) *count = 2;
    return cuda_sm75;
  }
  return NULL;
}

static _Thread_local bool amd_initialized;
static _Thread_local PolyTensorCore amd_rdna3[4];
static _Thread_local PolyTensorCore amd_rdna4[4];
static _Thread_local PolyTensorCore amd_cdna3[4];
static _Thread_local PolyTensorCore amd_cdna4[8];

static void init_amd(void) {
  if (amd_initialized) return;
  amd_initialized = true;
  init_tensor_core(&amd_rdna3[0], &AMD_RDNA3, POLY_FLOAT16, POLY_FLOAT32);
  init_tensor_core(&amd_rdna3[1], &AMD_RDNA3, POLY_FLOAT16, POLY_FLOAT16);
  init_tensor_core(&amd_rdna3[2], &AMD_RDNA3, POLY_BFLOAT16, POLY_FLOAT32);
  init_tensor_core(&amd_rdna3[3], &AMD_RDNA3, POLY_INT8, POLY_INT32);
  init_tensor_core(&amd_rdna4[0], &AMD_RDNA4, POLY_FLOAT16, POLY_FLOAT32);
  init_tensor_core(&amd_rdna4[1], &AMD_RDNA4, POLY_FLOAT16, POLY_FLOAT16);
  init_tensor_core(&amd_rdna4[2], &AMD_RDNA4, POLY_BFLOAT16, POLY_FLOAT32);
  init_tensor_core(&amd_rdna4[3], &AMD_RDNA4, POLY_BFLOAT16, POLY_BFLOAT16);
  init_tensor_core(&amd_cdna3[0], &AMD_CDNA_161632, POLY_FP8E5M2, POLY_FLOAT32);
  init_tensor_core(&amd_cdna3[1], &AMD_CDNA_161632, POLY_FP8E4M3, POLY_FLOAT32);
  init_tensor_core(&amd_cdna3[2], &AMD_CDNA_161616, POLY_FLOAT16, POLY_FLOAT32);
  init_tensor_core(&amd_cdna3[3], &AMD_CDNA_161616, POLY_BFLOAT16, POLY_FLOAT32);
  init_tensor_core(&amd_cdna4[0], &AMD_CDNA_1616128, POLY_FP8E5M2, POLY_FLOAT32);
  init_tensor_core(&amd_cdna4[1], &AMD_CDNA_1616128, POLY_FP8E4M3, POLY_FLOAT32);
  init_tensor_core(&amd_cdna4[2], &AMD_CDNA_161632, POLY_FP8E5M2, POLY_FLOAT32);
  init_tensor_core(&amd_cdna4[3], &AMD_CDNA_161632, POLY_FP8E4M3, POLY_FLOAT32);
  init_tensor_core(&amd_cdna4[4], &AMD_CDNA_161632, POLY_FLOAT16, POLY_FLOAT32);
  init_tensor_core(&amd_cdna4[5], &AMD_CDNA_161632, POLY_BFLOAT16, POLY_FLOAT32);
  init_tensor_core(&amd_cdna4[6], &AMD_CDNA_161616, POLY_FLOAT16, POLY_FLOAT32);
  init_tensor_core(&amd_cdna4[7], &AMD_CDNA_161616, POLY_BFLOAT16, POLY_FLOAT32);
}

const PolyTensorCore *poly_tc_get_amd(const char *arch, int *count) {
  if (count) *count = 0;
  init_amd();
  if (arch && strcmp(arch, "gfx942") == 0) {
    if (count) *count = 4;
    return amd_cdna3;
  }
  if (arch && strcmp(arch, "gfx950") == 0) {
    if (count) *count = 8;
    return amd_cdna4;
  }
  if (arch && (strcmp(arch, "gfx1200") == 0 || strcmp(arch, "gfx1201") == 0)) {
    if (count) *count = 4;
    return amd_rdna4;
  }
  if (count) *count = 4;
  return amd_rdna3;
}

int poly_tc_get_reduce_axes(const PolyTensorCore *tc, int out[][2]) {
  int k = tc->dims[2], n = 0;
  while (k > 1) {
    out[n][0] = n;
    out[n][1] = 2;
    n++;
    k /= 2;
  }
  return n;
}

int poly_tc_count_local(const PolyTensorCore *tc) {
  int n = 0;
  for (int i = 0; i < tc->n_opts; i++)
    n += tc->opts[i].type == 'l';
  return n;
}

int poly_tc_count_upcast(const PolyTensorCore *tc) {
  int n = 0;
  for (int i = 0; i < tc->n_opts; i++)
    n += tc->opts[i].type == 'u';
  return n;
}

int poly_tc_base_shape_str(const PolyTensorCore *tc, const char *out[], int max_n) {
  static const char *l_names[] = {"l0", "l1", "l2", "l3", "l4", "l5", "l6", "l7"};
  static const char *u_names[] = {"u0", "u1", "u2", "u3", "u4", "u5", "u6", "u7"};
  static const char *r_names[] = {"r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7"};
  int n = 0, local = 0, upcast = 0;
  for (int i = 0; i < tc->n_opts && n < max_n; i++)
    out[n++] = tc->opts[i].type == 'l' ? l_names[local++] : u_names[upcast++];
  int reduce_axes[16][2];
  int n_reduce = poly_tc_get_reduce_axes(tc, reduce_axes);
  for (int i = 0; i < n_reduce && n < max_n; i++)
    out[n++] = r_names[i];
  return n;
}

int poly_tc_base_upcast_axes(const PolyTensorCore *tc, const char *out[], int max_n) {
  static const char *r_names[] = {"r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7"};
  static const char *u_names[] = {"u0", "u1", "u2", "u3", "u4", "u5", "u6", "u7"};
  const char *forward[32];
  int reduce_axes[16][2], n_forward = 0;
  int n_reduce = poly_tc_get_reduce_axes(tc, reduce_axes);
  for (int i = 0; i < n_reduce; i++)
    forward[n_forward++] = r_names[i];
  for (int i = 0; i < poly_tc_count_upcast(tc); i++)
    forward[n_forward++] = u_names[i];
  int n = 0;
  for (int i = n_forward - 1; i >= 0 && n < max_n; i--)
    out[n++] = forward[i];
  return n;
}

static int build_remap(
    const PolyTensorCore *tc,
    int swizzle,
    const char *from[],
    const char *to[],
    int max_n
) {
  const char *shape[32], *flat[32];
  int n_shape = poly_tc_base_shape_str(tc, shape, 32), n_flat = 0;
  for (int group = 0; group < 3; group++)
    for (int i = 0; i < tc->swizzle_len[swizzle][group]; i++)
      flat[n_flat++] = tc->swizzle[swizzle][group][i];
  int n = n_shape < n_flat ? n_shape : n_flat;
  if (n > max_n) n = max_n;
  for (int i = 0; i < n; i++) {
    from[i] = shape[i];
    to[i] = flat[i];
  }
  return n;
}

void poly_tc_permute_for_shape_str(
    const PolyTensorCore *tc,
    int swizzle,
    const char *shape_str[],
    int n_shape,
    int perm[],
    int max_n
) {
  const char *from[32], *to[32];
  int n_remap = build_remap(tc, swizzle, from, to, 32);
  for (int i = 0; i < n_shape && i < max_n; i++) {
    const char *mapped = NULL;
    for (int j = 0; j < n_remap; j++)
      if (strcmp(shape_str[i], from[j]) == 0) {
        mapped = to[j];
        break;
      }
    perm[i] = i;
    if (!mapped) continue;
    for (int j = 0; j < n_shape; j++)
      if (strcmp(shape_str[j], mapped) == 0) {
        perm[i] = j;
        break;
      }
  }
}
