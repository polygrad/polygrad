/*
 * renderer/wasm.c — Polygrad's approved Wasm renderer divergence
 *
 * Walks linearized UOps and emits a valid
 * WASM binary module. The generated module imports shared
 * WebAssembly.Memory and exports a single kernel function.
 *
 * Supports two modes:
 *   use_simd=false → scalar f32 ops (one float per iteration)
 *   use_simd=true  → f32x4 SIMD ops (4 floats per iteration) + scalar epilogue
 *
 * Reference: WebAssembly Binary Format Specification
 */

#include "codegen/codegen.h"
#include "bigint.h"
#include "uop/spec.h"
#include "utils.h"
#include "wasm_builder.h"
#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* C storage for rendered UOp indices. */

typedef struct {
  PolyUOp **keys;
  int *vals;
  int cap;
} LocalMap;

typedef enum {
  WASM_MATMUL_AB = 0,
  WASM_MATMUL_ABT = 1,
} WasmMatmulKind;

typedef struct {
  int m;
  int n;
  int k;
  int out_param;
  int a_param;
  int b_param;
  WasmMatmulKind kind;
} WasmMatmulSpec;

static bool dt_is_v128(PolyDType dt, int lanes);

static uint32_t lm_hash(const void *p) {
  uintptr_t v = (uintptr_t)p;
  return (uint32_t)(v ^ (v >> 16) ^ (sizeof(v) > 4 ? (uint32_t)(v >> 32) : 0));
}

static void lm_init(LocalMap *m, int cap) {
  m->cap = cap < 16 ? 16 : cap;
  m->keys = calloc(m->cap, sizeof(PolyUOp *));
  m->vals = calloc(m->cap, sizeof(int));
}

static void lm_set(LocalMap *m, PolyUOp *key, int val) {
  uint32_t idx = lm_hash(key) % (uint32_t)m->cap;
  for (int i = 0; i < m->cap; i++) {
    uint32_t slot = (idx + i) % (uint32_t)m->cap;
    if (m->keys[slot] == NULL || m->keys[slot] == key) {
      m->keys[slot] = key;
      m->vals[slot] = val;
      return;
    }
  }
}

static int lm_get(LocalMap *m, PolyUOp *key) {
  uint32_t idx = lm_hash(key) % (uint32_t)m->cap;
  for (int i = 0; i < m->cap; i++) {
    uint32_t slot = (idx + i) % (uint32_t)m->cap;
    if (m->keys[slot] == key) return m->vals[slot];
    if (m->keys[slot] == NULL) {
      if (poly_debug_at_least(4) && key) {
        fprintf(
            stderr, "[polygrad:wasm] missing local for %s(%p) dt=%s n_src=%d\n",
            poly_op_name(key->op), (void *)key, key->dtype.name ? key->dtype.name : "?", key->n_src
        );
      }
      return -1;
    }
  }
  if (poly_debug_at_least(4) && key) {
    fprintf(
        stderr, "[polygrad:wasm] missing local for %s(%p) dt=%s n_src=%d\n", poly_op_name(key->op),
        (void *)key, key->dtype.name ? key->dtype.name : "?", key->n_src
    );
  }
  return -1;
}

static void lm_destroy(LocalMap *m) {
  free(m->keys);
  free(m->vals);
}

typedef struct {
  PolyUOp **keys;
  int **locals;
  int *sizes;
  int len;
  int cap;
} RegLocalMap;

static void rlm_init(RegLocalMap *m, int cap) {
  m->cap = cap < 16 ? 16 : cap;
  m->len = 0;
  m->keys = calloc((size_t)m->cap, sizeof(PolyUOp *));
  m->locals = calloc((size_t)m->cap, sizeof(int *));
  m->sizes = calloc((size_t)m->cap, sizeof(int));
}

static void rlm_set(RegLocalMap *m, PolyUOp *key, int *locals, int size) {
  if (!m || !key || !locals || size <= 0) return;
  for (int i = 0; i < m->len; i++) {
    if (m->keys[i] == key) {
      free(m->locals[i]);
      m->locals[i] = locals;
      m->sizes[i] = size;
      return;
    }
  }
  if (m->len >= m->cap) {
    free(locals);
    return;
  }
  m->keys[m->len] = key;
  m->locals[m->len] = locals;
  m->sizes[m->len] = size;
  m->len++;
}

static int rlm_get(RegLocalMap *m, PolyUOp *key, int idx) {
  if (!m || !key) return -1;
  for (int i = 0; i < m->len; i++) {
    if (m->keys[i] != key) continue;
    if (idx < 0 || idx >= m->sizes[i]) {
      if (poly_debug_at_least(4)) {
        fprintf(
            stderr,
            "[polygrad:wasm] register lane %d out of range for %s(%p) size=%d; using lane 0\n", idx,
            poly_op_name(key->op), (void *)key, m->sizes[i]
        );
      }
      return m->locals[i][0];
    }
    return m->locals[i][idx];
  }
  if (poly_debug_at_least(4)) {
    fprintf(
        stderr, "[polygrad:wasm] missing register local for %s(%p) lane=%d\n",
        poly_op_name(key->op), (void *)key, idx
    );
  }
  return -1;
}

static void rlm_destroy(RegLocalMap *m) {
  if (!m) return;
  for (int i = 0; i < m->len; i++)
    free(m->locals[i]);
  free(m->keys);
  free(m->locals);
  free(m->sizes);
}

/* Match the accumulator-base walking used by native renderers. pm_reduce can
 * route register arrays through AFTER/CAST/INDEX nodes, but the storage object
 * is still the underlying register/local buffer. */
static PolyUOp *wasm_acc_base(PolyUOp *u) {
  if (!u) return NULL;
  if (u->op == POLY_OP_BUFFER &&
      (poly_program_memory_is(u, POLY_ADDR_REG) || poly_program_memory_is(u, POLY_ADDR_LOCAL)))
    return u;
  if ((u->op == POLY_OP_AFTER || u->op == POLY_OP_CAST || u->op == POLY_OP_BITCAST ||
       u->op == POLY_OP_INDEX) &&
      u->n_src > 0)
    return wasm_acc_base(u->src[0]);
  return NULL;
}

static bool wasm_is_i32_address_base(PolyUOp *u) {
  if (!u) return false;
  if (u->op == POLY_OP_PARAM) return !poly_uop_is_alu_param(u);
  if (u->op == POLY_OP_BUFFER) return poly_program_memory_is(u, POLY_ADDR_GLOBAL);
  if ((u->op == POLY_OP_AFTER || u->op == POLY_OP_CAST || u->op == POLY_OP_BITCAST) && u->n_src > 0)
    return wasm_is_i32_address_base(u->src[0]);
  return false;
}

static bool wasm_const_index_checked(PolyUOp *u, int *out) {
  /* Current Tinygrad codegen emits INDEX(vector, CAST(CONST lane)); Wasm
   * SIMD extraction needs that rendered integer expression as an immediate. */
  if (u && u->op == POLY_OP_CAST && u->n_src == 1) u = u->src[0];
  int64_t value = 0;
  if (poly_uop_const_i64(u, &value) != 0 || value < INT_MIN || value > INT_MAX) return false;
  if (out) *out = (int)value;
  return true;
}

static int wasm_const_index(PolyUOp *u) {
  int value = 0;
  return wasm_const_index_checked(u, &value) ? value : 0;
}

static int64_t wasm_const_integer_bits(PolyUOp *u) {
  if (!u) return 0;
  if (u->arg.kind == POLY_ARG_BOOL) return u->arg.b ? 1 : 0;
  return (int64_t)poly_arg_integer_to_u64_mod(u->arg);
}

static PolyDType wasm_reg_base_dtype(PolyDType dtype) {
  return dtype;
}

static int wasm_shrink_width(PolyUOp *u) {
  int width = 1;
  if (u && u->op == POLY_OP_SHRINK && u->n_src >= 3)
    (void)wasm_const_index_checked(u->src[2], &width);
  return width > 0 ? width : 1;
}

static int wasm_shrink_offset(PolyUOp *u) {
  int offset = 0;
  if (u && u->op == POLY_OP_SHRINK && u->n_src >= 2)
    (void)wasm_const_index_checked(u->src[1], &offset);
  return offset;
}

static PolyUOp *wasm_load_shrink(PolyUOp *u) {
  if (!u || u->op != POLY_OP_LOAD || u->n_src < 1) return NULL;
  PolyUOp *addr = u->src[0];
  return (addr && addr->op == POLY_OP_SHRINK) ? addr : NULL;
}

static bool wasm_load_shrink_native_vec(PolyUOp *u, PolyDType *dtype_out, int *lanes_out) {
  PolyUOp *shr = wasm_load_shrink(u);
  if (!shr) return false;
  int width = wasm_shrink_width(shr);
  PolyDType scalar = u->dtype;
  if (!dt_is_v128(scalar, width)) return false;
  if (scalar.bitsize == 32 && width != 2 && width != 4) return false;
  if (scalar.bitsize == 64 && width != 2) return false;
  if (dtype_out) *dtype_out = scalar;
  if (lanes_out) *lanes_out = width;
  return true;
}

static int wasm_reg_storage_size(PolyUOp **uops, int n, PolyUOp *reg) {
  int64_t declared = poly_program_buffer_size(reg);
  int size = declared > INT_MAX ? INT_MAX : (int)declared;
  for (int i = 0; i < n; i++) {
    PolyUOp *u = uops[i];
    if (!u) continue;
    if (u->op == POLY_OP_INDEX && u->n_src >= 2 && wasm_acc_base(u->src[0]) == reg) {
      int idx = 0;
      if (wasm_const_index_checked(u->src[1], &idx) && idx >= 0 && idx + 1 > size) size = idx + 1;
    } else if (u->op == POLY_OP_SHRINK && u->n_src >= 3 && wasm_acc_base(u->src[0]) == reg) {
      int end = wasm_shrink_offset(u) + wasm_shrink_width(u);
      if (end > size) size = end;
    }
  }
  return size < 1 ? 1 : size;
}

/* Track which transcendentals are needed */

typedef struct {
  bool need_exp2f;
  bool need_log2f;
  bool need_sinf;
  bool need_exp2;
  bool need_log2;
  bool need_sin;
  bool need_powf;
  bool need_pow;
  bool need_recip; /* 1/x — not a WASM op, but can use f32.div */
} MathImports;

/* Pre-scan: determine imports and count params */

/* Tinygrad 2026-08-22/a9069c177a9d keeps original CALL indices in PARAM
 * slots, then passes only ProgramInfo.globals to the rendered function
 * (uop/ops.py:1239-1259, engine/realize.py:180-199). cstyle.py:204-220
 * sorts the used PARAMs by slot, so their positional ABI is dense. */
static int wasm_param_abi_index(PolyUOp **uops, int n, PolyUOp *param) {
  if (!uops || n < 0 || !param || param->op != POLY_OP_PARAM) return -1;
  int64_t slot = poly_program_buffer_slot(param);
  int index = 0;
  for (int i = 0; i < n; i++) {
    if (!uops[i] || uops[i]->op != POLY_OP_PARAM) continue;
    int64_t candidate = poly_program_buffer_slot(uops[i]);
    if (candidate >= slot) continue;
    bool first = true;
    for (int j = 0; j < i; j++)
      if (uops[j] && uops[j]->op == POLY_OP_PARAM &&
          poly_program_buffer_slot(uops[j]) == candidate) {
        first = false;
        break;
      }
    if (first) index++;
  }
  return index;
}

static int wasm_param_abi_count(PolyUOp **uops, int n) {
  int count = 0;
  for (int i = 0; i < n; i++) {
    if (!uops[i] || uops[i]->op != POLY_OP_PARAM) continue;
    int64_t slot = poly_program_buffer_slot(uops[i]);
    bool first = true;
    for (int j = 0; j < i; j++)
      if (uops[j] && uops[j]->op == POLY_OP_PARAM && poly_program_buffer_slot(uops[j]) == slot) {
        first = false;
        break;
      }
    if (first) count++;
  }
  return count;
}

static void prescan(
    PolyUOp **uops,
    int n,
    MathImports *math,
    int *n_params_out,
    int *n_ranges_out
) {
  memset(math, 0, sizeof(*math));
  int nr = 0;
  for (int i = 0; i < n; i++) {
    PolyUOp *u = uops[i];
    if (u->op == POLY_OP_RANGE) nr++;
    PolyDType scalar = u->dtype;
    bool f64 = poly_dtype_is_float(scalar) && scalar.bitsize == 64;
    if (u->op == POLY_OP_EXP2) {
      if (f64)
        math->need_exp2 = true;
      else
        math->need_exp2f = true;
    }
    if (u->op == POLY_OP_LOG2) {
      if (f64)
        math->need_log2 = true;
      else
        math->need_log2f = true;
    }
    if (u->op == POLY_OP_SIN) {
      if (f64)
        math->need_sin = true;
      else
        math->need_sinf = true;
    }
    if (u->op == POLY_OP_POW) {
      if (f64)
        math->need_pow = true;
      else
        math->need_powf = true;
    }
  }
  *n_params_out = wasm_param_abi_count(uops, n);
  *n_ranges_out = nr;
}

/* Build type section */

static void build_type_section(WasmBuf *mod, int n_params, MathImports *math) {
  WasmBuf sec;
  wb_init(&sec);

  /* Count function types needed */
  int n_types = 1; /* kernel type */
  bool need_unary_f32 = math->need_exp2f || math->need_log2f || math->need_sinf;
  bool need_unary_f64 = math->need_exp2 || math->need_log2 || math->need_sin;
  if (need_unary_f32) n_types++;
  if (need_unary_f64) n_types++;
  if (math->need_powf) n_types++;
  if (math->need_pow) n_types++;

  wb_uleb128(&sec, n_types);

  /* Type 0: kernel function — (i32, i32, ...) → () */
  wb_byte(&sec, WASM_TYPE_FUNC);
  wb_uleb128(&sec, n_params); /* param count */
  for (int i = 0; i < n_params; i++)
    wb_byte(&sec, WASM_TYPE_I32); /* all params are i32 byte offsets */
  wb_uleb128(&sec, 0); /* no results */

  /* Unary math imports preserve the operation dtype. */
  if (need_unary_f32) {
    wb_byte(&sec, WASM_TYPE_FUNC);
    wb_uleb128(&sec, 1); /* 1 param */
    wb_byte(&sec, WASM_TYPE_F32);
    wb_uleb128(&sec, 1); /* 1 result */
    wb_byte(&sec, WASM_TYPE_F32);
  }
  if (need_unary_f64) {
    wb_byte(&sec, WASM_TYPE_FUNC);
    wb_uleb128(&sec, 1);
    wb_byte(&sec, WASM_TYPE_F64);
    wb_uleb128(&sec, 1);
    wb_byte(&sec, WASM_TYPE_F64);
  }

  /* Binary math imports preserve the operation dtype too. */
  if (math->need_powf) {
    wb_byte(&sec, WASM_TYPE_FUNC);
    wb_uleb128(&sec, 2); /* 2 params */
    wb_byte(&sec, WASM_TYPE_F32);
    wb_byte(&sec, WASM_TYPE_F32);
    wb_uleb128(&sec, 1); /* 1 result */
    wb_byte(&sec, WASM_TYPE_F32);
  }
  if (math->need_pow) {
    wb_byte(&sec, WASM_TYPE_FUNC);
    wb_uleb128(&sec, 2);
    wb_byte(&sec, WASM_TYPE_F64);
    wb_byte(&sec, WASM_TYPE_F64);
    wb_uleb128(&sec, 1);
    wb_byte(&sec, WASM_TYPE_F64);
  }

  wb_section(mod, WASM_SEC_TYPE, &sec);
  wb_free(&sec);
}

/* Build import section */

static int build_import_section(WasmBuf *mod, MathImports *math) {
  WasmBuf sec;
  wb_init(&sec);

  int n_imports = 1; /* memory */
  if (math->need_exp2f) n_imports++;
  if (math->need_log2f) n_imports++;
  if (math->need_sinf) n_imports++;
  if (math->need_exp2) n_imports++;
  if (math->need_log2) n_imports++;
  if (math->need_sin) n_imports++;
  if (math->need_powf) n_imports++;
  if (math->need_pow) n_imports++;

  wb_uleb128(&sec, n_imports);

  /* Import 0: memory from "env" */
  wb_name(&sec, "env");
  wb_name(&sec, "memory");
  wb_byte(&sec, 0x02); /* import kind: memory */
  wb_byte(&sec, 0x00); /* limits: no max */
  wb_uleb128(&sec, 0); /* initial: 0 pages */

  int next_type = 1;
  int unary_f32_type = (math->need_exp2f || math->need_log2f || math->need_sinf) ? next_type++ : -1;
  int unary_f64_type = (math->need_exp2 || math->need_log2 || math->need_sin) ? next_type++ : -1;
  int powf_type = math->need_powf ? next_type++ : -1;
  int pow_type = math->need_pow ? next_type++ : -1;

  /* f32 math imports */
  int func_idx = 0;
  if (math->need_exp2f) {
    wb_name(&sec, "math");
    wb_name(&sec, "exp2f");
    wb_byte(&sec, 0x00); /* import kind: function */
    wb_uleb128(&sec, unary_f32_type);
    func_idx++;
  }
  if (math->need_log2f) {
    wb_name(&sec, "math");
    wb_name(&sec, "log2f");
    wb_byte(&sec, 0x00);
    wb_uleb128(&sec, unary_f32_type);
    func_idx++;
  }
  if (math->need_sinf) {
    wb_name(&sec, "math");
    wb_name(&sec, "sinf");
    wb_byte(&sec, 0x00);
    wb_uleb128(&sec, unary_f32_type);
    func_idx++;
  }

  /* f64 math imports */
  if (math->need_exp2) {
    wb_name(&sec, "math");
    wb_name(&sec, "exp2");
    wb_byte(&sec, 0x00);
    wb_uleb128(&sec, unary_f64_type);
    func_idx++;
  }
  if (math->need_log2) {
    wb_name(&sec, "math");
    wb_name(&sec, "log2");
    wb_byte(&sec, 0x00);
    wb_uleb128(&sec, unary_f64_type);
    func_idx++;
  }
  if (math->need_sin) {
    wb_name(&sec, "math");
    wb_name(&sec, "sin");
    wb_byte(&sec, 0x00);
    wb_uleb128(&sec, unary_f64_type);
    func_idx++;
  }

  /* Binary math imports */
  if (math->need_powf) {
    wb_name(&sec, "math");
    wb_name(&sec, "powf");
    wb_byte(&sec, 0x00);
    wb_uleb128(&sec, powf_type);
    func_idx++;
  }
  if (math->need_pow) {
    wb_name(&sec, "math");
    wb_name(&sec, "pow");
    wb_byte(&sec, 0x00);
    wb_uleb128(&sec, pow_type);
    func_idx++;
  }

  wb_section(mod, WASM_SEC_IMPORT, &sec);
  wb_free(&sec);

  return func_idx; /* number of imported functions (kernel func idx starts after) */
}

/* Build function section */

static void build_function_section(WasmBuf *mod) {
  WasmBuf sec;
  wb_init(&sec);
  wb_uleb128(&sec, 1); /* 1 function */
  wb_uleb128(&sec, 0); /* type index 0 = kernel type */
  wb_section(mod, WASM_SEC_FUNCTION, &sec);
  wb_free(&sec);
}

/* Build export section */

static void build_export_section(WasmBuf *mod, int kernel_func_idx) {
  WasmBuf sec;
  wb_init(&sec);
  wb_uleb128(&sec, 1); /* 1 export */
  wb_name(&sec, "kernel"); /* export name */
  wb_byte(&sec, WASM_EXPORT_FUNC); /* export kind */
  wb_uleb128(&sec, kernel_func_idx); /* function index */
  wb_section(mod, WASM_SEC_EXPORT, &sec);
  wb_free(&sec);
}

/* Dtype helpers */

static bool dt_is_f64(PolyDType dt) {
  return poly_dtype_is_float(dt) && dt.bitsize == 64;
}
static bool dt_is_i64(PolyDType dt) {
  return !poly_dtype_is_float(dt) && !poly_dtype_is_bool(dt) && dt.bitsize == 64;
}
static bool dt_is_64(PolyDType dt) {
  return dt.bitsize == 64;
}

static PolyRendererCaps poly_wasm_renderer_caps(void) {
  return (PolyRendererCaps){
      .device = "WASM",
      .has_mulacc = true,
      .has_threefry = false,
      .has_exp2 = true,
      .has_log2 = true,
      .has_sin = true,
      .has_fdiv = true,
      .supports_float16 = false,
      .supports_bfloat16 = false,
      .has_int64 = true,
      .has_local = false,
      .has_simd_int = false,
      .has_simd_float = true,
      .max_vec_width = 4,
  };
}

static bool dt_is_v128(PolyDType dt, int lanes) {
  int elem_bits = dt.bitsize;
  if (lanes <= 1) return false;
  if (elem_bits == 32) return lanes <= 4;
  if (elem_bits == 64) return lanes == 2;
  return false;
}

static int dt_v128_elem_bits(PolyDType dt) {
  PolyDType scalar = dt;
  return scalar.bitsize;
}

static bool wasm_graph_uses_f64(PolyCtx *ctx, PolyUOp *sink) {
  int n = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, sink, &n);
  if (!topo) return false;
  bool uses_f64 = false;
  for (int i = 0; i < n; i++) {
    PolyDType scalar = topo[i]->dtype;
    if (poly_dtype_is_float(scalar) && scalar.bitsize == 64) {
      uses_f64 = true;
      break;
    }
  }
  poly_toposort_free(topo);
  return uses_f64;
}

static PolyRendererCaps poly_wasm_renderer_caps_for_sink(PolyCtx *ctx, PolyUOp *sink) {
  PolyRendererCaps caps = poly_wasm_renderer_caps();
  if (wasm_graph_uses_f64(ctx, sink)) caps.max_vec_width = 1;
  return caps;
}

/* Which local bucket: 0=i32, 1=i64, 2=f32, 3=f64, 4=v128 */
static int dt_bucket(PolyDType dt, bool v128) {
  if (v128) return 4;
  if (poly_dtype_is_float(dt)) return dt.bitsize == 64 ? 3 : 2;
  return dt.bitsize == 64 ? 1 : 0;
}

/* Element size in bytes for buffer data */
static int dt_elem_size(PolyDType dt) {
  PolyDType base = dt;
  int sz = poly_dtype_itemsize(base);
  return sz > 0 ? sz : 4;
}

/* Log2 alignment for WASM load/store */
static int dt_align_log2(PolyDType dt) {
  int sz = dt_elem_size(dt);
  return sz >= 8 ? 3 : sz >= 4 ? 2 : sz >= 2 ? 1 : 0;
}

static void emit_local_get_as_i32(WasmBuf *body, int local, PolyDType dt) {
  wb_byte(body, WASM_OP_LOCAL_GET);
  wb_uleb128(body, local);
  /* Polygrad follows tinygrad's late index lowering, where sparse gather
   * labels can remain int64 in address expressions. This renderer targets
   * wasm32 linear memory, so memory addresses must be i32 stack values. */
  if (!poly_dtype_is_float(dt) && dt.bitsize == 64) wb_byte(body, WASM_OP_I32_WRAP_I64);
}

static void emit_cast_stack_value(WasmBuf *body, PolyDType src_dt, PolyDType dst_dt) {
  bool src_float = poly_dtype_is_float(src_dt);
  bool dst_float = poly_dtype_is_float(dst_dt);
  bool src_64 = dt_is_64(src_dt);
  bool dst_64 = dt_is_64(dst_dt);

  if (poly_dtype_is_bool(dst_dt)) {
    if (src_float) {
      if (src_64) {
        wb_byte(body, WASM_OP_F64_CONST);
        wb_f64(body, 0.0);
        wb_byte(body, WASM_OP_F64_NE);
      } else {
        wb_byte(body, WASM_OP_F32_CONST);
        wb_f32(body, 0.0f);
        wb_byte(body, WASM_OP_F32_NE);
      }
    } else if (src_64) {
      wb_byte(body, WASM_OP_I64_CONST);
      wb_sleb128(body, 0);
      wb_byte(body, WASM_OP_I64_NE);
    } else {
      wb_byte(body, WASM_OP_I32_CONST);
      wb_sleb128(body, 0);
      wb_byte(body, WASM_OP_I32_NE);
    }
    return;
  }

  if (src_float == dst_float && src_64 == dst_64) return;

  if (!src_float && !dst_float) {
    if (!src_64 && dst_64) {
      wb_byte(
          body, poly_dtype_is_unsigned(src_dt) ? WASM_OP_I64_EXTEND_I32_U : WASM_OP_I64_EXTEND_I32_S
      );
    } else if (src_64 && !dst_64) {
      wb_byte(body, WASM_OP_I32_WRAP_I64);
    }
    return;
  }

  if (src_float && dst_float) {
    if (!src_64 && dst_64)
      wb_byte(body, WASM_OP_F64_PROMOTE_F32);
    else if (src_64 && !dst_64)
      wb_byte(body, WASM_OP_F32_DEMOTE_F64);
    return;
  }

  if (!src_float && dst_float) {
    bool src_u = poly_dtype_is_unsigned(src_dt);
    if (src_64 && dst_64)
      wb_byte(body, src_u ? WASM_OP_F64_CONVERT_I64_U : WASM_OP_F64_CONVERT_I64_S);
    else if (src_64 && !dst_64)
      wb_byte(body, src_u ? WASM_OP_F32_CONVERT_I64_U : WASM_OP_F32_CONVERT_I64_S);
    else if (!src_64 && dst_64)
      wb_byte(body, src_u ? WASM_OP_F64_CONVERT_I32_U : WASM_OP_F64_CONVERT_I32_S);
    else
      wb_byte(body, src_u ? WASM_OP_F32_CONVERT_I32_U : WASM_OP_F32_CONVERT_I32_S);
    return;
  }

  bool dst_u = poly_dtype_is_unsigned(dst_dt);
  if (src_64 && dst_64)
    wb_byte(body, dst_u ? WASM_OP_I64_TRUNC_F64_U : WASM_OP_I64_TRUNC_F64_S);
  else if (src_64 && !dst_64)
    wb_byte(body, dst_u ? WASM_OP_I32_TRUNC_F64_U : WASM_OP_I32_TRUNC_F64_S);
  else if (!src_64 && dst_64)
    wb_byte(body, dst_u ? WASM_OP_I64_TRUNC_F32_U : WASM_OP_I64_TRUNC_F32_S);
  else
    wb_byte(body, dst_u ? WASM_OP_I32_TRUNC_F32_U : WASM_OP_I32_TRUNC_F32_S);
}

static PolyDType wasm_compare_dtype(PolyDType a, PolyDType b) {
  /* tinygrad's UOp spec requires comparison operands to share a base dtype.
   * Late Polygrad index lowering can still leave the common practical case
   * CAST(long, LOAD(int)) < int_bound. WASM has no implicit numeric casts, so
   * pick the wider operand type at the renderer boundary instead of emitting an
   * invalid mixed-width compare. */
  bool a_float = poly_dtype_is_float(a);
  bool b_float = poly_dtype_is_float(b);
  if (a_float || b_float) {
    if (a_float && a.bitsize == 64) return a;
    if (b_float && b.bitsize == 64) return b;
    return a_float ? a : b;
  }
  if (a.bitsize == 64) return a;
  if (b.bitsize == 64) return b;
  return a;
}

static bool has_simd_op(PolyOps op);
static bool wasm_uop_value_is_v128(PolyCtx *ctx, PolyUOp *u);
static void emit_alu_scalar(
    WasmBuf *code,
    PolyOps op,
    PolyDType dtype,
    MathImports *math,
    int n_imported_funcs
);

static bool wasm_is_compare_op(PolyOps op) {
  return op == POLY_OP_CMPLT || op == POLY_OP_CMPEQ || op == POLY_OP_CMPNE;
}

static PolyDType wasm_simd_value_dtype(PolyUOp *u) {
  if (!u) return POLY_FLOAT32;
  if (wasm_is_compare_op(u->op) && u->n_src >= 2)
    return wasm_compare_dtype(u->src[0]->dtype, u->src[1]->dtype);
  return u->dtype;
}

static bool wasm_stack_native_vec(PolyUOp *u, PolyDType *dtype_out, int *lanes_out);

static PolyDType wasm_native_v128_dtype(PolyUOp *u) {
  PolyDType dtype;
  if (wasm_load_shrink_native_vec(u, &dtype, NULL)) return dtype;
  if (wasm_stack_native_vec(u, &dtype, NULL)) return dtype;
  if (u && wasm_is_compare_op(u->op) && u->n_src >= 2) {
    PolyDType cmp = wasm_compare_dtype(u->src[0]->dtype, u->src[1]->dtype);
    return cmp.bitsize == 64 ? POLY_INT64 : POLY_INT32;
  }
  return wasm_simd_value_dtype(u);
}

static bool wasm_simd_loop_can_vectorize_alu(PolyUOp *u) {
  if (!u || !poly_opset_has(POLY_GROUP_ALU, u->op) || !has_simd_op(u->op)) return false;
  if (wasm_is_compare_op(u->op) && u->n_src >= 2)
    return poly_dtype_is_float(wasm_compare_dtype(u->src[0]->dtype, u->src[1]->dtype));
  if (u->op == POLY_OP_WHERE)
    return u->n_src >= 3 && poly_dtype_is_float(u->dtype) && wasm_is_compare_op(u->src[0]->op);
  return poly_dtype_is_float(u->dtype);
}

static bool wasm_simd_loop_local_is_v128(PolyUOp *u) {
  if (!u) return false;
  if (u->op == POLY_OP_STACK && u->n_src > 1 && dt_is_v128(u->dtype, u->n_src)) return true;
  if (u->op == POLY_OP_LOAD) return true;
  return wasm_simd_loop_can_vectorize_alu(u);
}

static void emit_local_get_for_alu_src(
    WasmBuf *body,
    int local,
    PolyDType src_dt,
    PolyDType alu_dt
) {
  wb_byte(body, WASM_OP_LOCAL_GET);
  wb_uleb128(body, local);
  emit_cast_stack_value(body, src_dt, alu_dt);
}

static PolyDType wasm_alu_src_dtype(PolyUOp *u, int src_idx) {
  if (!u) return POLY_INT32;
  if ((u->op == POLY_OP_CMPLT || u->op == POLY_OP_CMPEQ || u->op == POLY_OP_CMPNE) && u->n_src >= 2)
    return wasm_compare_dtype(u->src[0]->dtype, u->src[1]->dtype);
  (void)src_idx;
  return u->dtype;
}

static void emit_alu_sources(WasmBuf *body, LocalMap *locals, PolyUOp *u) {
  if (!u) return;

  if (u->op == POLY_OP_WHERE && u->n_src >= 3) {
    int s1 = lm_get(locals, u->src[1]);
    int s2 = lm_get(locals, u->src[2]);
    int s0 = lm_get(locals, u->src[0]);
    emit_local_get_for_alu_src(body, s1, u->src[1]->dtype, wasm_alu_src_dtype(u, 1));
    emit_local_get_for_alu_src(body, s2, u->src[2]->dtype, wasm_alu_src_dtype(u, 2));
    wb_byte(body, WASM_OP_LOCAL_GET);
    wb_uleb128(body, s0);
    PolyDType cond_dt = u->src[0]->dtype;
    if (poly_dtype_is_float(cond_dt)) {
      if (cond_dt.bitsize == 64) {
        wb_byte(body, WASM_OP_F64_CONST);
        wb_f64(body, 0.0);
        wb_byte(body, WASM_OP_F64_NE);
      } else {
        wb_byte(body, WASM_OP_F32_CONST);
        wb_f32(body, 0.0f);
        wb_byte(body, WASM_OP_F32_NE);
      }
    } else if (cond_dt.bitsize == 64) {
      wb_byte(body, WASM_OP_I64_CONST);
      wb_sleb128(body, 0);
      wb_byte(body, WASM_OP_I64_NE);
    }
    return;
  }

  if (u->op == POLY_OP_MULACC && u->n_src >= 3) {
    /* tinygrad defines MULACC(x, y, z) as (x * y) + z. WASM binary ops
     * consume the top two stack values, so push the addend first, then the
     * multiply operands. The following MUL consumes x/y and ADD combines with
     * z, matching tinygrad's PTX fma/mad operand order. */
    int order[3] = {2, 0, 1};
    for (int k = 0; k < 3; k++) {
      int j = order[k];
      int src = lm_get(locals, u->src[j]);
      emit_local_get_for_alu_src(body, src, u->src[j]->dtype, wasm_alu_src_dtype(u, j));
    }
    return;
  }

  int n_operands = poly_opset_has(POLY_GROUP_TERNARY, u->op)  ? 3
                   : poly_opset_has(POLY_GROUP_BINARY, u->op) ? 2
                                                              : 1;
  if (n_operands > u->n_src) n_operands = u->n_src;
  for (int j = 0; j < n_operands; j++) {
    int src = lm_get(locals, u->src[j]);
    emit_local_get_for_alu_src(body, src, u->src[j]->dtype, wasm_alu_src_dtype(u, j));
  }
}

static void emit_scalar_load_opcode(WasmBuf *body, PolyDType dt) {
  if (dt_is_f64(dt)) {
    wb_byte(body, WASM_OP_F64_LOAD);
  } else if (poly_dtype_is_float(dt)) {
    wb_byte(body, WASM_OP_F32_LOAD);
  } else if (dt_is_i64(dt)) {
    wb_byte(body, WASM_OP_I64_LOAD);
  } else {
    /* WASM locals are i32 for <=32-bit integers, but memory width must match
     * the buffer dtype. This is required for uint8/int16 tensors and mirrors C
     * load semantics instead of widening every buffer element to four bytes. */
    int sz = dt_elem_size(dt);
    bool is_u = poly_dtype_is_unsigned(dt) || poly_dtype_is_bool(dt);
    if (sz <= 1)
      wb_byte(body, is_u ? WASM_OP_I32_LOAD8_U : WASM_OP_I32_LOAD8_S);
    else if (sz == 2)
      wb_byte(body, is_u ? WASM_OP_I32_LOAD16_U : WASM_OP_I32_LOAD16_S);
    else
      wb_byte(body, WASM_OP_I32_LOAD);
  }
  wb_uleb128(body, dt_align_log2(dt));
  wb_uleb128(body, 0);
}

static void emit_scalar_store_opcode(WasmBuf *body, PolyDType dt) {
  if (dt_is_f64(dt)) {
    wb_byte(body, WASM_OP_F64_STORE);
  } else if (poly_dtype_is_float(dt)) {
    wb_byte(body, WASM_OP_F32_STORE);
  } else if (dt_is_i64(dt)) {
    wb_byte(body, WASM_OP_I64_STORE);
  } else {
    int sz = dt_elem_size(dt);
    if (sz <= 1)
      wb_byte(body, WASM_OP_I32_STORE8);
    else if (sz == 2)
      wb_byte(body, WASM_OP_I32_STORE16);
    else
      wb_byte(body, WASM_OP_I32_STORE);
  }
  wb_uleb128(body, dt_align_log2(dt));
  wb_uleb128(body, 0);
}

static void emit_v128_zero(WasmBuf *body) {
  wb_byte(body, WASM_SIMD_PREFIX);
  wb_uleb128(body, WASM_SIMD_V128_CONST);
  for (int i = 0; i < 16; i++)
    wb_byte(body, 0);
}

static void emit_v128_shuffle(WasmBuf *body, const uint8_t lanes[16]) {
  wb_byte(body, WASM_SIMD_PREFIX);
  wb_uleb128(body, WASM_SIMD_I8X16_SHUFFLE);
  for (int i = 0; i < 16; i++)
    wb_byte(body, lanes[i]);
}

static void emit_v128_load_opcode(WasmBuf *body, PolyDType dt, int lanes) {
  wb_byte(body, WASM_SIMD_PREFIX);
  int bits = dt.bitsize * lanes;
  if (bits >= 128)
    wb_uleb128(body, WASM_SIMD_V128_LOAD);
  else if (bits <= 32)
    wb_uleb128(body, WASM_SIMD_V128_LOAD32_ZERO);
  else
    wb_uleb128(body, WASM_SIMD_V128_LOAD64_ZERO);
  wb_uleb128(body, 2);
  wb_uleb128(body, 0);
}

static void emit_v128_store_opcode(WasmBuf *body, PolyDType dt, int lanes) {
  wb_byte(body, WASM_SIMD_PREFIX);
  int bits = dt.bitsize * lanes;
  if (bits >= 128) {
    wb_uleb128(body, WASM_SIMD_V128_STORE);
    wb_uleb128(body, 2);
    wb_uleb128(body, 0);
    return;
  }
  wb_uleb128(body, bits <= 32 ? WASM_SIMD_V128_STORE32_LANE : WASM_SIMD_V128_STORE64_LANE);
  wb_uleb128(body, 2);
  wb_uleb128(body, 0);
  wb_uleb128(body, 0);
}

static void emit_v128_splat(WasmBuf *body, PolyDType dt) {
  PolyDType scalar = dt;
  int elem_bits = dt_v128_elem_bits(dt);
  wb_byte(body, WASM_SIMD_PREFIX);
  if (poly_dtype_is_float(scalar))
    wb_uleb128(body, elem_bits == 64 ? WASM_SIMD_F64X2_SPLAT : WASM_SIMD_F32X4_SPLAT);
  else
    wb_uleb128(body, elem_bits == 64 ? WASM_SIMD_I64X2_SPLAT : WASM_SIMD_I32X4_SPLAT);
}

static void emit_v128_extract_lane(WasmBuf *body, PolyDType vec_dt, int lane) {
  PolyDType scalar = vec_dt;
  int elem_bits = dt_v128_elem_bits(vec_dt);
  wb_byte(body, WASM_SIMD_PREFIX);
  if (poly_dtype_is_float(scalar))
    wb_uleb128(body, elem_bits == 64 ? WASM_SIMD_F64X2_EXTRACT : WASM_SIMD_F32X4_EXTRACT);
  else
    wb_uleb128(body, elem_bits == 64 ? WASM_SIMD_I64X2_EXTRACT : WASM_SIMD_I32X4_EXTRACT);
  wb_uleb128(body, lane < 0 ? 0 : lane);
}

static void emit_v128_replace_lane(WasmBuf *body, PolyDType vec_dt, int lane) {
  PolyDType scalar = vec_dt;
  int elem_bits = dt_v128_elem_bits(vec_dt);
  wb_byte(body, WASM_SIMD_PREFIX);
  if (poly_dtype_is_float(scalar))
    wb_uleb128(body, elem_bits == 64 ? WASM_SIMD_F64X2_REPLACE : WASM_SIMD_F32X4_REPLACE);
  else
    wb_uleb128(body, elem_bits == 64 ? WASM_SIMD_I64X2_REPLACE : WASM_SIMD_I32X4_REPLACE);
  wb_uleb128(body, lane < 0 ? 0 : lane);
}

static bool emit_v128_cast_opcode(WasmBuf *body, PolyDType src_dt, PolyDType dst_dt, bool bitcast) {
  if (bitcast) return true;

  PolyDType src = src_dt;
  PolyDType dst = dst_dt;
  bool src_float = poly_dtype_is_float(src);
  bool dst_float = poly_dtype_is_float(dst);

  if (src_float == dst_float && src.bitsize == dst.bitsize) return true;

  if (src_float && dst_float) {
    if (src.bitsize == 32 && dst.bitsize == 64) {
      wb_byte(body, WASM_SIMD_PREFIX);
      wb_uleb128(body, WASM_SIMD_F64X2_PROMOTE_LOW_F32X4);
      return true;
    }
    if (src.bitsize == 64 && dst.bitsize == 32) {
      wb_byte(body, WASM_SIMD_PREFIX);
      wb_uleb128(body, WASM_SIMD_F32X4_DEMOTE_F64X2_ZERO);
      return true;
    }
    return false;
  }

  if (src_float && !dst_float && dst.bitsize == 32) {
    wb_byte(body, WASM_SIMD_PREFIX);
    if (src.bitsize == 32)
      wb_uleb128(
          body, poly_dtype_is_unsigned(dst) ? WASM_SIMD_I32X4_TRUNC_SAT_F32X4_U
                                            : WASM_SIMD_I32X4_TRUNC_SAT_F32X4_S
      );
    else if (src.bitsize == 64)
      wb_uleb128(
          body, poly_dtype_is_unsigned(dst) ? WASM_SIMD_I32X4_TRUNC_SAT_F64X2_U_ZERO
                                            : WASM_SIMD_I32X4_TRUNC_SAT_F64X2_S_ZERO
      );
    else
      return false;
    return true;
  }

  if (!src_float && dst_float && src.bitsize == 32) {
    wb_byte(body, WASM_SIMD_PREFIX);
    if (dst.bitsize == 32)
      wb_uleb128(
          body, poly_dtype_is_unsigned(src) ? WASM_SIMD_F32X4_CONVERT_I32X4_U
                                            : WASM_SIMD_F32X4_CONVERT_I32X4_S
      );
    else if (dst.bitsize == 64)
      wb_uleb128(
          body, poly_dtype_is_unsigned(src) ? WASM_SIMD_F64X2_CONVERT_LOW_I32X4_U
                                            : WASM_SIMD_F64X2_CONVERT_LOW_I32X4_S
      );
    else
      return false;
    return true;
  }

  return false;
}

static void emit_v128_cast_lanes(
    WasmBuf *body,
    int src_local,
    PolyDType src_dt,
    int src_lanes,
    PolyDType dst_dt,
    int dst_lanes
) {
  PolyDType src_scalar = src_dt;
  PolyDType dst_scalar = dst_dt;
  int n_lanes = dst_lanes;
  if (src_lanes > 0 && src_lanes < n_lanes) n_lanes = src_lanes;
  if (n_lanes <= 0) n_lanes = 1;

  emit_v128_zero(body);
  for (int j = 0; j < n_lanes; j++) {
    wb_byte(body, WASM_OP_LOCAL_GET);
    wb_uleb128(body, src_local);
    emit_v128_extract_lane(body, src_dt, j);
    emit_cast_stack_value(body, src_scalar, dst_scalar);
    emit_v128_replace_lane(body, dst_dt, j);
  }
}

static void emit_local_get_for_vector_src(
    WasmBuf *body,
    int local,
    PolyCtx *ctx,
    PolyUOp *src_uop,
    PolyDType vec_dt
) {
  wb_byte(body, WASM_OP_LOCAL_GET);
  wb_uleb128(body, local);
  if (wasm_uop_value_is_v128(ctx, src_uop)) return;
  emit_cast_stack_value(body, src_uop->dtype, vec_dt);
  emit_v128_splat(body, vec_dt);
}

static void emit_vector_sources(PolyCtx *ctx, WasmBuf *body, LocalMap *locals, PolyUOp *u) {
  if (!u) return;
  PolyDType vec_dt = wasm_simd_value_dtype(u);

  if (u->op == POLY_OP_MULACC && u->n_src >= 3) {
    int order[3] = {2, 0, 1};
    for (int k = 0; k < 3; k++) {
      int j = order[k];
      int src = lm_get(locals, u->src[j]);
      emit_local_get_for_vector_src(body, src, ctx, u->src[j], vec_dt);
    }
    return;
  }

  if (u->op == POLY_OP_WHERE && u->n_src >= 3) {
    int order[3] = {1, 2, 0}; /* v128.bitselect: true, false, mask */
    for (int k = 0; k < 3; k++) {
      int j = order[k];
      int src = lm_get(locals, u->src[j]);
      emit_local_get_for_vector_src(body, src, ctx, u->src[j], vec_dt);
    }
    return;
  }

  int n_operands = poly_opset_has(POLY_GROUP_TERNARY, u->op)  ? 3
                   : poly_opset_has(POLY_GROUP_BINARY, u->op) ? 2
                                                              : 1;
  if (n_operands > u->n_src) n_operands = u->n_src;
  for (int j = 0; j < n_operands; j++) {
    int src = lm_get(locals, u->src[j]);
    emit_local_get_for_vector_src(body, src, ctx, u->src[j], vec_dt);
  }
}

static void emit_local_get_lane_for_alu_src(
    WasmBuf *body,
    int local,
    PolyCtx *ctx,
    PolyUOp *src,
    PolyDType alu_dt,
    int lane
) {
  wb_byte(body, WASM_OP_LOCAL_GET);
  wb_uleb128(body, local);
  if (wasm_uop_value_is_v128(ctx, src)) {
    emit_v128_extract_lane(body, src->dtype, lane);
    emit_cast_stack_value(body, src->dtype, alu_dt);
  } else {
    emit_cast_stack_value(body, src->dtype, alu_dt);
  }
}

static void emit_scalar_condition_from_dtype(WasmBuf *body, PolyDType cond_dt) {
  if (poly_dtype_is_float(cond_dt)) {
    if (cond_dt.bitsize == 64) {
      wb_byte(body, WASM_OP_F64_CONST);
      wb_f64(body, 0.0);
      wb_byte(body, WASM_OP_F64_NE);
    } else {
      wb_byte(body, WASM_OP_F32_CONST);
      wb_f32(body, 0.0f);
      wb_byte(body, WASM_OP_F32_NE);
    }
  } else if (cond_dt.bitsize == 64) {
    wb_byte(body, WASM_OP_I64_CONST);
    wb_sleb128(body, 0);
    wb_byte(body, WASM_OP_I64_NE);
  }
}

static void emit_vector_alu_lane_fallback(
    WasmBuf *body,
    LocalMap *locals,
    PolyCtx *ctx,
    PolyUOp *u,
    MathImports *math,
    int n_imported_funcs
) {
  PolyDType dst_scalar = u->dtype;
  PolyDType alu_dtype = dst_scalar;
  if (u->op == POLY_OP_CMPLT || u->op == POLY_OP_CMPEQ || u->op == POLY_OP_CMPNE)
    alu_dtype = wasm_compare_dtype(u->src[0]->dtype, u->src[1]->dtype);

  int64_t n_lanes_i64 = poly_uop_max_numel(ctx, u);
  int n_lanes = n_lanes_i64 > 0 && n_lanes_i64 <= INT_MAX ? (int)n_lanes_i64 : 1;
  if (n_lanes <= 0) n_lanes = dst_scalar.bitsize == 64 ? 2 : 4;

  emit_v128_zero(body);
  for (int lane = 0; lane < n_lanes; lane++) {
    if (u->op == POLY_OP_RECIPROCAL) {
      if (dst_scalar.bitsize == 64) {
        wb_byte(body, WASM_OP_F64_CONST);
        wb_f64(body, 1.0);
      } else {
        wb_byte(body, WASM_OP_F32_CONST);
        wb_f32(body, 1.0f);
      }
    }

    if (u->op == POLY_OP_NEG && !poly_dtype_is_float(dst_scalar) &&
        !poly_dtype_is_bool(dst_scalar)) {
      if (dst_scalar.bitsize == 64) {
        wb_byte(body, WASM_OP_I64_CONST);
        wb_sleb128(body, 0);
      } else {
        wb_byte(body, WASM_OP_I32_CONST);
        wb_sleb128(body, 0);
      }
    }

    if (u->op == POLY_OP_WHERE && u->n_src >= 3) {
      int s1 = lm_get(locals, u->src[1]);
      int s2 = lm_get(locals, u->src[2]);
      int s0 = lm_get(locals, u->src[0]);
      emit_local_get_lane_for_alu_src(body, s1, ctx, u->src[1], dst_scalar, lane);
      emit_local_get_lane_for_alu_src(body, s2, ctx, u->src[2], dst_scalar, lane);
      PolyDType cond_dt = u->src[0]->dtype;
      emit_local_get_lane_for_alu_src(body, s0, ctx, u->src[0], cond_dt, lane);
      emit_scalar_condition_from_dtype(body, cond_dt);
    } else if (u->op == POLY_OP_MULACC && u->n_src >= 3) {
      int order[3] = {2, 0, 1};
      for (int k = 0; k < 3; k++) {
        int j = order[k];
        int src = lm_get(locals, u->src[j]);
        emit_local_get_lane_for_alu_src(body, src, ctx, u->src[j], wasm_alu_src_dtype(u, j), lane);
      }
    } else {
      int n_operands = poly_opset_has(POLY_GROUP_TERNARY, u->op)  ? 3
                       : poly_opset_has(POLY_GROUP_BINARY, u->op) ? 2
                                                                  : 1;
      if (n_operands > u->n_src) n_operands = u->n_src;
      for (int j = 0; j < n_operands; j++) {
        int src = lm_get(locals, u->src[j]);
        emit_local_get_lane_for_alu_src(body, src, ctx, u->src[j], wasm_alu_src_dtype(u, j), lane);
      }
    }

    emit_alu_scalar(body, u->op, alu_dtype, math, n_imported_funcs);
    emit_v128_replace_lane(body, u->dtype, lane);
  }
}

static void emit_local_get_for_simd_loop_src(
    WasmBuf *body,
    LocalMap *locals,
    PolyUOp *src_uop,
    PolyDType vec_dt
) {
  int local = lm_get(locals, src_uop);
  wb_byte(body, WASM_OP_LOCAL_GET);
  wb_uleb128(body, local);
  if (wasm_simd_loop_local_is_v128(src_uop)) return;
  emit_cast_stack_value(body, src_uop->dtype, vec_dt);
  emit_v128_splat(body, vec_dt);
}

static void emit_simd_loop_sources(WasmBuf *body, LocalMap *locals, PolyUOp *u) {
  if (!u) return;
  PolyDType vec_dt = wasm_simd_value_dtype(u);

  if (u->op == POLY_OP_MULACC && u->n_src >= 3) {
    int order[3] = {2, 0, 1};
    for (int k = 0; k < 3; k++)
      emit_local_get_for_simd_loop_src(body, locals, u->src[order[k]], vec_dt);
    return;
  }

  if (u->op == POLY_OP_WHERE && u->n_src >= 3) {
    int order[3] = {1, 2, 0}; /* v128.bitselect: true, false, mask */
    for (int k = 0; k < 3; k++)
      emit_local_get_for_simd_loop_src(body, locals, u->src[order[k]], vec_dt);
    return;
  }

  int n_operands = poly_opset_has(POLY_GROUP_TERNARY, u->op)  ? 3
                   : poly_opset_has(POLY_GROUP_BINARY, u->op) ? 2
                                                              : 1;
  if (n_operands > u->n_src) n_operands = u->n_src;
  for (int j = 0; j < n_operands; j++)
    emit_local_get_for_simd_loop_src(body, locals, u->src[j], vec_dt);
}

static void emit_scalar_where_sources(WasmBuf *body, LocalMap *locals, PolyUOp *u) {
  int s1 = lm_get(locals, u->src[1]);
  int s2 = lm_get(locals, u->src[2]);
  int s0 = lm_get(locals, u->src[0]);
  wb_byte(body, WASM_OP_LOCAL_GET);
  wb_uleb128(body, s1);
  wb_byte(body, WASM_OP_LOCAL_GET);
  wb_uleb128(body, s2);
  wb_byte(body, WASM_OP_LOCAL_GET);
  wb_uleb128(body, s0);
  PolyDType cond_dt = u->src[0]->dtype;
  if (poly_dtype_is_float(cond_dt)) {
    if (cond_dt.bitsize == 64) {
      wb_byte(body, WASM_OP_F64_CONST);
      wb_f64(body, 0.0);
      wb_byte(body, WASM_OP_F64_NE);
    } else {
      wb_byte(body, WASM_OP_F32_CONST);
      wb_f32(body, 0.0f);
      wb_byte(body, WASM_OP_F32_NE);
    }
  } else if (cond_dt.bitsize == 64) {
    wb_byte(body, WASM_OP_I64_CONST);
    wb_sleb128(body, 0);
    wb_byte(body, WASM_OP_I64_NE);
  }
}

/* Emit scalar ALU opcode */

static int math_import_index(const MathImports *math, PolyOps op, bool f64) {
  int idx = 0;
  if (math->need_exp2f) {
    if (op == POLY_OP_EXP2 && !f64) return idx;
    idx++;
  }
  if (math->need_log2f) {
    if (op == POLY_OP_LOG2 && !f64) return idx;
    idx++;
  }
  if (math->need_sinf) {
    if (op == POLY_OP_SIN && !f64) return idx;
    idx++;
  }
  if (math->need_exp2) {
    if (op == POLY_OP_EXP2 && f64) return idx;
    idx++;
  }
  if (math->need_log2) {
    if (op == POLY_OP_LOG2 && f64) return idx;
    idx++;
  }
  if (math->need_sin) {
    if (op == POLY_OP_SIN && f64) return idx;
    idx++;
  }
  if (math->need_powf) {
    if (op == POLY_OP_POW && !f64) return idx;
    idx++;
  }
  if (math->need_pow) {
    if (op == POLY_OP_POW && f64) return idx;
    idx++;
  }
  return -1;
}

static void emit_alu_scalar(
    WasmBuf *code,
    PolyOps op,
    PolyDType dtype,
    MathImports *math,
    int n_imported_funcs
) {
  bool is_int = !poly_dtype_is_float(dtype);
  bool is_unsigned = poly_dtype_is_unsigned(dtype);
  bool b64 = dt_is_64(dtype);
  switch (op) {
  /* Unary */
  case POLY_OP_NEG:
    if (poly_dtype_is_bool(dtype)) {
      /* Arithmetic NEG followed by bool conversion is identity. Canonicalize
       * the already-bool lane to 0/1; logical NOT is CMPNE(x,true). */
      wb_byte(code, WASM_OP_I32_EQZ);
      wb_byte(code, WASM_OP_I32_EQZ);
    } else if (is_int) {
      wb_byte(code, b64 ? WASM_OP_I64_SUB : WASM_OP_I32_SUB);
    } else {
      wb_byte(code, b64 ? WASM_OP_F64_NEG : WASM_OP_F32_NEG);
    }
    break;
  case POLY_OP_SQRT:
    wb_byte(code, b64 ? WASM_OP_F64_SQRT : WASM_OP_F32_SQRT);
    break;
  case POLY_OP_TRUNC:
    wb_byte(code, b64 ? WASM_OP_F64_TRUNC : WASM_OP_F32_TRUNC);
    break;
  case POLY_OP_EXP2: {
    int idx = math_import_index(math, op, b64);
    assert(idx >= 0);
    wb_byte(code, WASM_OP_CALL);
    wb_uleb128(code, idx);
    break;
  }
  case POLY_OP_LOG2: {
    int idx = math_import_index(math, op, b64);
    assert(idx >= 0);
    wb_byte(code, WASM_OP_CALL);
    wb_uleb128(code, idx);
    break;
  }
  case POLY_OP_SIN: {
    int idx = math_import_index(math, op, b64);
    assert(idx >= 0);
    wb_byte(code, WASM_OP_CALL);
    wb_uleb128(code, idx);
    break;
  }
  case POLY_OP_RECIPROCAL:
    wb_byte(code, b64 ? WASM_OP_F64_DIV : WASM_OP_F32_DIV);
    break;

  /* Binary */
  case POLY_OP_ADD:
    wb_byte(
        code, is_int ? (b64 ? WASM_OP_I64_ADD : WASM_OP_I32_ADD)
                     : (b64 ? WASM_OP_F64_ADD : WASM_OP_F32_ADD)
    );
    break;
  case POLY_OP_SUB:
    wb_byte(
        code, is_int ? (b64 ? WASM_OP_I64_SUB : WASM_OP_I32_SUB)
                     : (b64 ? WASM_OP_F64_SUB : WASM_OP_F32_SUB)
    );
    break;
  case POLY_OP_MUL:
    wb_byte(
        code, is_int ? (b64 ? WASM_OP_I64_MUL : WASM_OP_I32_MUL)
                     : (b64 ? WASM_OP_F64_MUL : WASM_OP_F32_MUL)
    );
    break;
  case POLY_OP_FDIV:
    wb_byte(code, b64 ? WASM_OP_F64_DIV : WASM_OP_F32_DIV);
    break;
  case POLY_OP_MAX:
    wb_byte(code, b64 ? WASM_OP_F64_MAX : WASM_OP_F32_MAX);
    break;
  case POLY_OP_CMPLT:
    if (is_int)
      wb_byte(
          code, b64 ? (is_unsigned ? WASM_OP_I64_LT_U : WASM_OP_I64_LT_S)
                    : (is_unsigned ? WASM_OP_I32_LT_U : WASM_OP_I32_LT_S)
      );
    else
      wb_byte(code, b64 ? WASM_OP_F64_LT : WASM_OP_F32_LT);
    break;
  case POLY_OP_CMPEQ:
    wb_byte(
        code,
        is_int ? (b64 ? WASM_OP_I64_EQ : WASM_OP_I32_EQ) : (b64 ? WASM_OP_F64_EQ : WASM_OP_F32_EQ)
    );
    break;
  case POLY_OP_CMPNE:
    wb_byte(
        code,
        is_int ? (b64 ? WASM_OP_I64_NE : WASM_OP_I32_NE) : (b64 ? WASM_OP_F64_NE : WASM_OP_F32_NE)
    );
    break;

  /* Integer-only ops */
  case POLY_OP_IDIV:
    wb_byte(
        code, b64 ? (is_unsigned ? WASM_OP_I64_DIV_U : WASM_OP_I64_DIV_S)
                  : (is_unsigned ? WASM_OP_I32_DIV_U : WASM_OP_I32_DIV_S)
    );
    break;
  case POLY_OP_MOD:
    wb_byte(
        code, b64 ? (is_unsigned ? WASM_OP_I64_REM_U : WASM_OP_I64_REM_S)
                  : (is_unsigned ? WASM_OP_I32_REM_U : WASM_OP_I32_REM_S)
    );
    break;
  case POLY_OP_SHL:
    wb_byte(code, b64 ? WASM_OP_I64_SHL : WASM_OP_I32_SHL);
    break;
  case POLY_OP_SHR:
    if (b64)
      wb_byte(code, is_unsigned ? WASM_OP_I64_SHR_U : WASM_OP_I64_SHR_S);
    else
      wb_byte(code, is_unsigned ? WASM_OP_I32_SHR_U : WASM_OP_I32_SHR_S);
    break;
  case POLY_OP_AND:
    wb_byte(code, b64 ? WASM_OP_I64_AND : WASM_OP_I32_AND);
    break;
  case POLY_OP_OR:
    wb_byte(code, b64 ? WASM_OP_I64_OR : WASM_OP_I32_OR);
    break;
  case POLY_OP_XOR:
    wb_byte(code, b64 ? WASM_OP_I64_XOR : WASM_OP_I32_XOR);
    break;

  /* Ternary */
  case POLY_OP_WHERE:
    wb_byte(code, WASM_OP_SELECT);
    break;
  case POLY_OP_MULACC:
    wb_byte(
        code, is_int ? (b64 ? WASM_OP_I64_MUL : WASM_OP_I32_MUL)
                     : (b64 ? WASM_OP_F64_MUL : WASM_OP_F32_MUL)
    );
    wb_byte(
        code, is_int ? (b64 ? WASM_OP_I64_ADD : WASM_OP_I32_ADD)
                     : (b64 ? WASM_OP_F64_ADD : WASM_OP_F32_ADD)
    );
    break;

  case POLY_OP_POW: {
    int idx = math_import_index(math, op, b64);
    assert(idx >= 0);
    wb_byte(code, WASM_OP_CALL);
    wb_uleb128(code, idx);
    break;
  }

  default:
    wb_byte(code, WASM_OP_NOP);
    break;
  }
  (void)n_imported_funcs;
}

/* Emit SIMD ALU opcode (f32x4) */

static void emit_alu_simd_f32x4(WasmBuf *code, PolyOps op) {
  switch (op) {
  case POLY_OP_NEG:
    wb_byte(code, WASM_SIMD_PREFIX);
    wb_uleb128(code, WASM_SIMD_F32X4_NEG);
    break;
  case POLY_OP_SQRT:
    wb_byte(code, WASM_SIMD_PREFIX);
    wb_uleb128(code, WASM_SIMD_F32X4_SQRT);
    break;
  case POLY_OP_ADD:
    wb_byte(code, WASM_SIMD_PREFIX);
    wb_uleb128(code, WASM_SIMD_F32X4_ADD);
    break;
  case POLY_OP_SUB:
    wb_byte(code, WASM_SIMD_PREFIX);
    wb_uleb128(code, WASM_SIMD_F32X4_SUB);
    break;
  case POLY_OP_MUL:
    wb_byte(code, WASM_SIMD_PREFIX);
    wb_uleb128(code, WASM_SIMD_F32X4_MUL);
    break;
  case POLY_OP_FDIV:
    wb_byte(code, WASM_SIMD_PREFIX);
    wb_uleb128(code, WASM_SIMD_F32X4_DIV);
    break;
  case POLY_OP_MAX:
    wb_byte(code, WASM_SIMD_PREFIX);
    wb_uleb128(code, WASM_SIMD_F32X4_MAX);
    break;
  case POLY_OP_CMPLT:
    wb_byte(code, WASM_SIMD_PREFIX);
    wb_uleb128(code, WASM_SIMD_F32X4_LT);
    break;
  case POLY_OP_CMPEQ:
    wb_byte(code, WASM_SIMD_PREFIX);
    wb_uleb128(code, WASM_SIMD_F32X4_EQ);
    break;
  case POLY_OP_CMPNE:
    wb_byte(code, WASM_SIMD_PREFIX);
    wb_uleb128(code, WASM_SIMD_F32X4_NE);
    break;
  case POLY_OP_WHERE:
    wb_byte(code, WASM_SIMD_PREFIX);
    wb_uleb128(code, WASM_SIMD_V128_BITSELECT);
    break;
  case POLY_OP_MULACC:
    wb_byte(code, WASM_SIMD_PREFIX);
    wb_uleb128(code, WASM_SIMD_F32X4_MUL);
    wb_byte(code, WASM_SIMD_PREFIX);
    wb_uleb128(code, WASM_SIMD_F32X4_ADD);
    break;
  default:
    break;
  }
}

/* Emit SIMD ALU opcode (f64x2) */

static void emit_alu_simd_f64x2(WasmBuf *code, PolyOps op) {
  switch (op) {
  case POLY_OP_NEG:
    wb_byte(code, WASM_SIMD_PREFIX);
    wb_uleb128(code, WASM_SIMD_F64X2_NEG);
    break;
  case POLY_OP_SQRT:
    wb_byte(code, WASM_SIMD_PREFIX);
    wb_uleb128(code, WASM_SIMD_F64X2_SQRT);
    break;
  case POLY_OP_ADD:
    wb_byte(code, WASM_SIMD_PREFIX);
    wb_uleb128(code, WASM_SIMD_F64X2_ADD);
    break;
  case POLY_OP_SUB:
    wb_byte(code, WASM_SIMD_PREFIX);
    wb_uleb128(code, WASM_SIMD_F64X2_SUB);
    break;
  case POLY_OP_MUL:
    wb_byte(code, WASM_SIMD_PREFIX);
    wb_uleb128(code, WASM_SIMD_F64X2_MUL);
    break;
  case POLY_OP_FDIV:
    wb_byte(code, WASM_SIMD_PREFIX);
    wb_uleb128(code, WASM_SIMD_F64X2_DIV);
    break;
  case POLY_OP_MAX:
    wb_byte(code, WASM_SIMD_PREFIX);
    wb_uleb128(code, WASM_SIMD_F64X2_MAX);
    break;
  case POLY_OP_CMPLT:
    wb_byte(code, WASM_SIMD_PREFIX);
    wb_uleb128(code, WASM_SIMD_F64X2_LT);
    break;
  case POLY_OP_CMPEQ:
    wb_byte(code, WASM_SIMD_PREFIX);
    wb_uleb128(code, WASM_SIMD_F64X2_EQ);
    break;
  case POLY_OP_CMPNE:
    wb_byte(code, WASM_SIMD_PREFIX);
    wb_uleb128(code, WASM_SIMD_F64X2_NE);
    break;
  case POLY_OP_WHERE:
    wb_byte(code, WASM_SIMD_PREFIX);
    wb_uleb128(code, WASM_SIMD_V128_BITSELECT);
    break;
  case POLY_OP_MULACC:
    wb_byte(code, WASM_SIMD_PREFIX);
    wb_uleb128(code, WASM_SIMD_F64X2_MUL);
    wb_byte(code, WASM_SIMD_PREFIX);
    wb_uleb128(code, WASM_SIMD_F64X2_ADD);
    break;
  default:
    break;
  }
}

/* Check if an op has a SIMD equivalent */

static bool has_simd_op(PolyOps op) {
  switch (op) {
  case POLY_OP_NEG:
  case POLY_OP_SQRT:
  case POLY_OP_ADD:
  case POLY_OP_SUB:
  case POLY_OP_MUL:
  case POLY_OP_FDIV:
  case POLY_OP_MAX:
  case POLY_OP_CMPLT:
  case POLY_OP_CMPEQ:
  case POLY_OP_CMPNE:
  case POLY_OP_WHERE:
  case POLY_OP_MULACC:
    return true;
  default:
    return false;
  }
}

static bool wasm_vector_alu_has_direct_simd(PolyCtx *ctx, PolyUOp *u) {
  if (!u || !poly_opset_has(POLY_GROUP_ALU, u->op) || !has_simd_op(u->op)) return false;
  int64_t lanes = poly_uop_max_numel(ctx, u);
  if (lanes <= 1 || lanes > INT_MAX || !dt_is_v128(wasm_simd_value_dtype(u), (int)lanes))
    return false;
  return wasm_simd_loop_can_vectorize_alu(u);
}

/* Tinygrad 2026-08-22/a9069c177a9d CStyleLanguage renders STACK with
 * render_type(x), whose width comes from UOp.max_numel (cstyle.py:52).
 * August devectorization therefore represents a four-lane register as a
 * scalar-typed STACK with four sources. */
static bool wasm_stack_native_vec(PolyUOp *u, PolyDType *dtype_out, int *lanes_out) {
  if (!u || u->op != POLY_OP_STACK || u->n_src <= 1) return false;
  PolyDType scalar = u->dtype;
  int lanes = u->n_src;
  if (!((scalar.bitsize == 32 && (lanes == 2 || lanes == 4)) || (scalar.bitsize == 64 && lanes == 2)
      ))
    return false;
  if (!dt_is_v128(scalar, lanes)) return false;
  if (dtype_out) *dtype_out = scalar;
  if (lanes_out) *lanes_out = lanes;
  return true;
}

/* Tinygrad 2026-08-22/a9069c177a9d renderer/cstyle.py:52-54 emits a shaped
 * STACK of lane-isomorphic scalar expressions as one native vector value.
 * Wasm has no source compiler, so recognize that same structure here. */
static bool wasm_structural_vec_expr(PolyCtx *ctx, PolyUOp **lanes, int n_lanes) {
  if (!ctx || !lanes || n_lanes <= 1 || n_lanes > 4) return false;

  PolyUOp *first = lanes[0];
  if (!first) return false;
  bool same = true;
  for (int lane = 1; lane < n_lanes; lane++)
    if (lanes[lane] != first) same = false;
  if (same) return poly_uop_max_numel(ctx, first) == 1;

  if (first->op == POLY_OP_INDEX && first->n_src >= 2) {
    PolyUOp *base = first->src[0];
    if (!wasm_uop_value_is_v128(ctx, base)) return false;
    for (int lane = 0; lane < n_lanes; lane++) {
      int index = -1;
      if (!lanes[lane] || lanes[lane]->op != POLY_OP_INDEX || lanes[lane]->n_src < 2 ||
          lanes[lane]->src[0] != base || !wasm_const_index_checked(lanes[lane]->src[1], &index) ||
          index != lane)
        return false;
    }
    return true;
  }

  if (!poly_opset_has(POLY_GROUP_ALU, first->op) || !wasm_simd_loop_can_vectorize_alu(first))
    return false;
  for (int lane = 1; lane < n_lanes; lane++)
    if (!lanes[lane] || lanes[lane]->op != first->op || lanes[lane]->n_src != first->n_src)
      return false;

  for (int src = 0; src < first->n_src; src++) {
    PolyUOp *src_lanes[4];
    for (int lane = 0; lane < n_lanes; lane++)
      src_lanes[lane] = lanes[lane]->src[src];
    if (!wasm_structural_vec_expr(ctx, src_lanes, n_lanes)) return false;
  }
  return true;
}

static void wasm_mark_structural_vec_expr(
    PolyUOp **uops,
    int n,
    PolyUOp **lanes,
    int n_lanes,
    bool *covered
) {
  bool same = true;
  for (int lane = 1; lane < n_lanes; lane++)
    if (lanes[lane] != lanes[0]) same = false;
  if (same) return;

  for (int lane = 0; lane < n_lanes; lane++)
    for (int i = 0; i < n; i++)
      if (uops[i] == lanes[lane]) covered[i] = true;

  if (lanes[0]->op == POLY_OP_INDEX) return;
  for (int src = 0; src < lanes[0]->n_src; src++) {
    PolyUOp *src_lanes[4];
    for (int lane = 0; lane < n_lanes; lane++)
      src_lanes[lane] = lanes[lane]->src[src];
    wasm_mark_structural_vec_expr(uops, n, src_lanes, n_lanes, covered);
  }
}

static void wasm_plan_structural_vecs(
    PolyCtx *ctx,
    PolyUOp **uops,
    int n,
    bool *fused_stack,
    bool *skip
) {
  bool *covered = calloc((size_t)n, sizeof(*covered));
  if (!covered) return;

  for (int i = 0; i < n; i++) {
    PolyUOp *u = uops[i];
    if (!wasm_stack_native_vec(u, NULL, NULL) || !wasm_structural_vec_expr(ctx, u->src, u->n_src))
      continue;
    fused_stack[i] = true;
    covered[i] = true;
    wasm_mark_structural_vec_expr(uops, n, u->src, u->n_src, covered);
  }

  for (int i = 0; i < n; i++) {
    if (!covered[i] || fused_stack[i]) continue;
    bool external_consumer = false;
    for (int j = 0; j < n && !external_consumer; j++)
      for (int src = 0; src < uops[j]->n_src; src++)
        if (uops[j]->src[src] == uops[i] && !covered[j]) {
          external_consumer = true;
          break;
        }
    skip[i] = !external_consumer;
  }
  free(covered);
}

static bool wasm_emit_structural_vec_expr(
    PolyCtx *ctx,
    WasmBuf *body,
    LocalMap *locals,
    PolyUOp **lanes,
    int n_lanes
) {
  PolyUOp *first = lanes[0];
  bool same = true;
  for (int lane = 1; lane < n_lanes; lane++)
    if (lanes[lane] != first) same = false;
  if (same) {
    int local = lm_get(locals, first);
    if (local < 0) return false;
    wb_byte(body, WASM_OP_LOCAL_GET);
    wb_uleb128(body, local);
    emit_v128_splat(body, first->dtype);
    return true;
  }

  if (first->op == POLY_OP_INDEX) {
    int local = lm_get(locals, first->src[0]);
    if (local < 0) return false;
    wb_byte(body, WASM_OP_LOCAL_GET);
    wb_uleb128(body, local);
    return true;
  }

  int order[3] = {0, 1, 2};
  int n_src = first->n_src;
  if (first->op == POLY_OP_WHERE) {
    order[0] = 1;
    order[1] = 2;
    order[2] = 0;
  } else if (first->op == POLY_OP_MULACC) {
    order[0] = 2;
    order[1] = 0;
    order[2] = 1;
  }
  for (int pos = 0; pos < n_src; pos++) {
    int src = order[pos];
    PolyUOp *src_lanes[4];
    for (int lane = 0; lane < n_lanes; lane++)
      src_lanes[lane] = lanes[lane]->src[src];
    if (!wasm_emit_structural_vec_expr(ctx, body, locals, src_lanes, n_lanes)) return false;
  }

  PolyDType dtype = wasm_simd_value_dtype(first);
  if (dtype.bitsize == 64)
    emit_alu_simd_f64x2(body, first->op);
  else
    emit_alu_simd_f32x4(body, first->op);
  return true;
}

static bool wasm_vector_alu_needs_lane_fallback(PolyCtx *ctx, PolyUOp *u) {
  int64_t lanes = u ? poly_uop_max_numel(ctx, u) : -1;
  return u && lanes > 1 && lanes <= INT_MAX && poly_opset_has(POLY_GROUP_ALU, u->op) &&
         dt_is_v128(u->dtype, (int)lanes) && !wasm_vector_alu_has_direct_simd(ctx, u);
}

static bool wasm_uop_value_is_v128(PolyCtx *ctx, PolyUOp *u) {
  PolyDType dtype;
  int64_t lanes = u ? poly_uop_max_numel(ctx, u) : -1;
  return u && lanes > 1 && lanes <= INT_MAX &&
         (dt_is_v128(u->dtype, (int)lanes) || wasm_vector_alu_has_direct_simd(ctx, u) ||
          wasm_load_shrink_native_vec(u, &dtype, NULL) || wasm_stack_native_vec(u, &dtype, NULL));
}

static PolyDType wasm_local_value_dtype(PolyUOp *u) {
  if (!u) return POLY_INT32;
  return u->dtype;
}

/* Check if entire kernel is SIMD-able */

/* Detect whether the kernel operates on f64 data (check LOAD dtypes) */
static bool kernel_is_f64(PolyUOp **uops, int n) {
  for (int i = 0; i < n; i++) {
    if (uops[i]->op == POLY_OP_LOAD && dt_is_f64(uops[i]->dtype)) return true;
    if (uops[i]->op == POLY_OP_STORE && uops[i]->n_src > 1 && dt_is_f64(uops[i]->src[1]->dtype))
      return true;
  }
  return false;
}

static bool wasm_simd_float_bits(PolyDType dt, int *bits) {
  PolyDType scalar = dt;
  if (!poly_dtype_is_float(scalar)) return false;
  if (scalar.bitsize != 32 && scalar.bitsize != 64) return false;
  if (bits) {
    if (*bits == 0)
      *bits = scalar.bitsize;
    else if (*bits != scalar.bitsize)
      return false;
  }
  return true;
}

static PolyUOp *wasm_simd_single_const_range(PolyUOp **uops, int n) {
  PolyUOp *range = NULL;
  for (int i = 0; i < n; i++) {
    if (uops[i]->op != POLY_OP_RANGE) continue;
    if (range) return NULL;
    if (uops[i]->n_src < 1 || !uops[i]->src[0] || uops[i]->src[0]->op != POLY_OP_CONST) return NULL;
    range = uops[i];
  }
  return range;
}

static bool wasm_simd_unit_index(PolyUOp *u, PolyUOp *range, int *bits) {
  if (!u || u->op != POLY_OP_INDEX || u->n_src < 2) return false;
  if (!u->src[0] || u->src[0]->op != POLY_OP_PARAM) return false;
  if (u->src[1] != range) return false;
  return wasm_simd_float_bits(u->src[0]->dtype, bits);
}

static bool wasm_simd_store_value(PolyUOp *u, int *bits) {
  if (!u) return false;
  if (u->op == POLY_OP_LOAD) return wasm_simd_float_bits(u->dtype, bits);
  if (wasm_simd_loop_can_vectorize_alu(u)) return wasm_simd_float_bits(u->dtype, bits);
  return false;
}

/* Tinygrad 2026-08-22/a9069c177a9d codegen/__init__.py:381 and
 * uop/ops.py:603 spell literals as CAST(strong, CONST(weak)). Wasm must emit
 * the mathematical CONST directly at the CAST width; materializing weakfloat
 * as f32 first would truncate a double literal. */
static bool wasm_is_casted_const(PolyUOp *u) {
  if (!u || u->op != POLY_OP_CAST || u->n_src != 1 || !u->src[0] ||
      u->src[0]->op != POLY_OP_CONST ||
      !(poly_dtype_is_weak(u->src[0]->dtype) || poly_dtype_is_bool(u->src[0]->dtype)))
    return false;
  return true;
}

static bool wasm_casted_const_identity(PolyUOp *u) {
  if (!wasm_is_casted_const(u)) return false;
  PolyDType src = u->src[0]->dtype;
  PolyDType dst = u->dtype;
  bool src_float = poly_dtype_is_float(src), dst_float = poly_dtype_is_float(dst);
  if (src_float != dst_float) return false;
  return src_float ? dt_is_f64(src) == dt_is_f64(dst) : dt_is_64(src) == dt_is_64(dst);
}

static bool kernel_is_simdable(PolyUOp **uops, int n) {
  PolyUOp *range = wasm_simd_single_const_range(uops, n);
  if (!range) return false;
  int bits = 0;

  for (int i = 0; i < n; i++) {
    PolyUOp *u = uops[i];
    switch (u->op) {
    case POLY_OP_PARAM:
    case POLY_OP_CONST:
    case POLY_OP_RANGE:
    case POLY_OP_END:
    case POLY_OP_SINK:
      break;
    case POLY_OP_INDEX:
      if (!wasm_simd_unit_index(u, range, &bits)) return false;
      break;
    case POLY_OP_LOAD:
      if (u->n_src < 1 || !wasm_simd_unit_index(u->src[0], range, &bits)) return false;
      if (!wasm_simd_float_bits(u->dtype, &bits)) return false;
      break;
    case POLY_OP_STORE:
      if (u->n_src < 2 || !wasm_simd_unit_index(u->src[0], range, &bits)) return false;
      if (!wasm_simd_float_bits(u->src[0]->dtype, &bits)) return false;
      if (!wasm_simd_store_value(u->src[1], &bits)) return false;
      break;
    default:
      if (!poly_opset_has(POLY_GROUP_ALU, u->op)) return false;
      break;
    }

    if (poly_opset_has(POLY_GROUP_ALU, u->op)) {
      bool relevant = poly_dtype_is_float(u->dtype) || wasm_is_compare_op(u->op);
      if (relevant && !wasm_simd_loop_can_vectorize_alu(u)) return false;
      if (wasm_is_compare_op(u->op) && u->n_src >= 2) {
        PolyDType cmp_dt = wasm_compare_dtype(u->src[0]->dtype, u->src[1]->dtype);
        if (!wasm_simd_float_bits(cmp_dt, &bits)) return false;
      } else if (u->op != POLY_OP_WHERE) {
        if (!wasm_simd_float_bits(u->dtype, &bits)) return false;
      }
    }
    /* Transcendentals don't have SIMD versions */
    if (u->op == POLY_OP_EXP2 || u->op == POLY_OP_LOG2 || u->op == POLY_OP_SIN ||
        u->op == POLY_OP_POW)
      return false;
    if (u->op == POLY_OP_RECIPROCAL) return false;
    if ((u->op == POLY_OP_CAST && !wasm_casted_const_identity(u)) || u->op == POLY_OP_BITCAST)
      return false;
    /* Reduce kernels are not SIMD-able (V1) */
    if (u->op == POLY_OP_BUFFER &&
        (poly_program_memory_is(u, POLY_ADDR_REG) || poly_program_memory_is(u, POLY_ADDR_LOCAL)))
      return false;
  }
  return true;
}

static bool wasm_reg_addr_index(PolyUOp *addr, PolyUOp **reg_base_out, int *idx_out) {
  PolyUOp *idx = poly_find_index_through_cast(addr);
  if (!idx || idx->op != POLY_OP_INDEX || idx->n_src < 2) return false;
  PolyUOp *reg_base = wasm_acc_base(idx->src[0]);
  if (!reg_base || reg_base->op != POLY_OP_BUFFER ||
      !poly_program_memory_is(reg_base, POLY_ADDR_REG))
    return false;
  PolyDType scalar = wasm_reg_base_dtype(reg_base->dtype);
  if (!poly_dtype_is_float(scalar)) return false;
  int reg_idx = 0;
  if (!wasm_const_index_checked(idx->src[1], &reg_idx)) return false;
  if (reg_idx < 0) return false;
  if (reg_base_out) *reg_base_out = reg_base;
  if (idx_out) *idx_out = reg_idx;
  return true;
}

static bool wasm_reg_addr_lane(PolyUOp *addr, PolyUOp **reg_base_out, int *lane_out) {
  PolyUOp *reg_base = NULL;
  int lane = -1;
  if (!wasm_reg_addr_index(addr, &reg_base, &lane)) return false;
  PolyDType scalar = wasm_reg_base_dtype(reg_base->dtype);
  int n_lanes = scalar.bitsize == 64 ? 2 : 4;
  if (lane < 0 || lane >= n_lanes) return false;
  if (reg_base_out) *reg_base_out = reg_base;
  if (lane_out) *lane_out = lane;
  return true;
}

static int alloc_local(
    PolyDType dt,
    bool v128,
    int *next_i32,
    int *next_i64,
    int *next_f32,
    int *next_f64,
    int *next_v128
);

/* Build code section (scalar) */

/* Allocate a local in the right bucket for a dtype */
static int alloc_local(
    PolyDType dt,
    bool v128,
    int *next_i32,
    int *next_i64,
    int *next_f32,
    int *next_f64,
    int *next_v128
) {
  int b = dt_bucket(dt, v128);
  switch (b) {
  case 0:
    return (*next_i32)++;
  case 1:
    return (*next_i64)++;
  case 2:
    return (*next_f32)++;
  case 3:
    return (*next_f64)++;
  case 4:
    return (*next_v128)++;
  }
  return (*next_i32)++;
}

/* Increment the right local counter during the counting pass */
static void count_local(
    PolyDType dt,
    bool v128,
    int *ni32,
    int *ni64,
    int *nf32,
    int *nf64,
    int *nv128
) {
  int b = dt_bucket(dt, v128);
  switch (b) {
  case 0:
    (*ni32)++;
    break;
  case 1:
    (*ni64)++;
    break;
  case 2:
    (*nf32)++;
    break;
  case 3:
    (*nf64)++;
    break;
  case 4:
    (*nv128)++;
    break;
  }
}

static void build_code_scalar(
    PolyCtx *ctx,
    WasmBuf *mod,
    PolyUOp **uops,
    int n,
    int n_params,
    MathImports *math,
    int n_imported_funcs
) {
  bool *fused_stack = calloc((size_t)n, sizeof(*fused_stack));
  bool *skip = calloc((size_t)n, sizeof(*skip));
  if (fused_stack && skip) wasm_plan_structural_vecs(ctx, uops, n, fused_stack, skip);

  /* --- Count locals needed (beyond function params) --- */
  int n_locals_i32 = 0, n_locals_i64 = 0, n_locals_f32 = 0, n_locals_f64 = 0, n_locals_v128 = 0;

  /* First pass: count locals */
  for (int i = 0; i < n; i++) {
    if (skip && skip[i]) continue;
    PolyUOp *u = uops[i];
    if (u->op == POLY_OP_RANGE) n_locals_i32++; /* loop counter always i32 */
    if (u->op == POLY_OP_LOAD) {
      bool is_reg_load = u->src[0]->op == POLY_OP_INDEX &&
                         poly_program_memory_is(u->src[0]->src[0], POLY_ADDR_REG);
      PolyDType shrink_dtype;
      if (wasm_load_shrink_native_vec(u, &shrink_dtype, NULL)) {
        count_local(
            shrink_dtype, true, &n_locals_i32, &n_locals_i64, &n_locals_f32, &n_locals_f64,
            &n_locals_v128
        );
      } else if (!is_reg_load) {
        count_local(
            u->dtype, wasm_uop_value_is_v128(ctx, u), &n_locals_i32, &n_locals_i64, &n_locals_f32,
            &n_locals_f64, &n_locals_v128
        );
      }
    }
    if (poly_opset_has(POLY_GROUP_ALU, u->op)) {
      PolyDType local_dt = wasm_local_value_dtype(u);
      count_local(
          local_dt, wasm_uop_value_is_v128(ctx, u), &n_locals_i32, &n_locals_i64, &n_locals_f32,
          &n_locals_f64, &n_locals_v128
      );
    }
    if ((u->op == POLY_OP_CAST && !wasm_casted_const_identity(u)) || u->op == POLY_OP_BITCAST)
      count_local(
          u->dtype, wasm_uop_value_is_v128(ctx, u), &n_locals_i32, &n_locals_i64, &n_locals_f32,
          &n_locals_f64, &n_locals_v128
      );
    if (u->op == POLY_OP_BUFFER && poly_program_memory_is(u, POLY_ADDR_REG)) {
      PolyDType base = wasm_reg_base_dtype(poly_program_buffer_dtype(u));
      int reg_size = wasm_reg_storage_size(uops, n, u);
      for (int r = 0; r < reg_size; r++)
        count_local(
            base, false, &n_locals_i32, &n_locals_i64, &n_locals_f32, &n_locals_f64, &n_locals_v128
        );
    }
    if (u->op == POLY_OP_STACK)
      count_local(
          wasm_local_value_dtype(u), wasm_uop_value_is_v128(ctx, u), &n_locals_i32, &n_locals_i64,
          &n_locals_f32, &n_locals_f64, &n_locals_v128
      );
    if (u->op == POLY_OP_SHRINK) {
      if (wasm_is_i32_address_base(u->src[0])) n_locals_i32++;
    }
    if (u->op == POLY_OP_INDEX) {
      if (wasm_reg_addr_index(u, NULL, NULL)) {
        /* register lane aliases an existing local */
      } else if (wasm_is_i32_address_base(u->src[0])) {
        n_locals_i32++; /* byte offsets always i32 */
      } else {
        count_local(
            u->dtype, wasm_uop_value_is_v128(ctx, u), &n_locals_i32, &n_locals_i64, &n_locals_f32,
            &n_locals_f64, &n_locals_v128
        );
      }
    }
    if (u->op == POLY_OP_CONST)
      count_local(
          u->dtype, false, &n_locals_i32, &n_locals_i64, &n_locals_f32, &n_locals_f64,
          &n_locals_v128
      );
  }

  /* Locals declaration: count of (count, type) pairs */
  int n_local_types = 0;
  if (n_locals_i32 > 0) n_local_types++;
  if (n_locals_i64 > 0) n_local_types++;
  if (n_locals_f32 > 0) n_local_types++;
  if (n_locals_f64 > 0) n_local_types++;
  if (n_locals_v128 > 0) n_local_types++;

  /* Function body buffer */
  WasmBuf body;
  wb_init(&body);

  wb_uleb128(&body, n_local_types);
  if (n_locals_i32 > 0) {
    wb_uleb128(&body, n_locals_i32);
    wb_byte(&body, WASM_TYPE_I32);
  }
  if (n_locals_i64 > 0) {
    wb_uleb128(&body, n_locals_i64);
    wb_byte(&body, WASM_TYPE_I64);
  }
  if (n_locals_f32 > 0) {
    wb_uleb128(&body, n_locals_f32);
    wb_byte(&body, WASM_TYPE_F32);
  }
  if (n_locals_f64 > 0) {
    wb_uleb128(&body, n_locals_f64);
    wb_byte(&body, WASM_TYPE_F64);
  }
  if (n_locals_v128 > 0) {
    wb_uleb128(&body, n_locals_v128);
    wb_byte(&body, WASM_TYPE_V128);
  }

  /* --- Assign local indices --- */
  LocalMap locals;
  lm_init(&locals, n * 2);
  RegLocalMap reg_locals;
  rlm_init(&reg_locals, n);

  /* Local layout: [params(i32)] [i32] [i64] [f32] [f64] [v128] */
  int next_i32 = n_params;
  int next_i64 = n_params + n_locals_i32;
  int next_f32 = n_params + n_locals_i32 + n_locals_i64;
  int next_f64 = n_params + n_locals_i32 + n_locals_i64 + n_locals_f32;
  int next_v128 = n_params + n_locals_i32 + n_locals_i64 + n_locals_f32 + n_locals_f64;

  /* --- Second pass: emit instructions --- */
  for (int i = 0; i < n; i++) {
    if (skip && skip[i]) continue;
    PolyUOp *u = uops[i];

    if (u->op == POLY_OP_SINK || u->op == POLY_OP_NOOP) continue;

    if (u->op == POLY_OP_GROUP) {
      for (int j = 0; j < u->n_src; j++) {
        PolyUOp *store = u->src[j];
        if (!store || store->op != POLY_OP_STORE || store->n_src < 2) continue;
        PolyUOp *base = NULL;
        int lane = -1;
        if (!wasm_reg_addr_lane(store->src[0], &base, &lane)) continue;
        int acc_local = rlm_get(&reg_locals, base, lane);
        int val_local = lm_get(&locals, store->src[1]);
        if (acc_local < 0 || val_local < 0) continue;
        wb_byte(&body, WASM_OP_LOCAL_GET);
        wb_uleb128(&body, val_local);
        wb_byte(&body, WASM_OP_LOCAL_SET);
        wb_uleb128(&body, acc_local);
      }
      continue;
    }

    /* --- PARAM --- */
    if (u->op == POLY_OP_PARAM) {
      lm_set(&locals, u, wasm_param_abi_index(uops, n, u));
      continue;
    }

    /* --- CONST --- */
    if (u->op == POLY_OP_CONST) {
      int local_idx =
          alloc_local(u->dtype, false, &next_i32, &next_i64, &next_f32, &next_f64, &next_v128);
      if (dt_is_f64(u->dtype)) {
        wb_byte(&body, WASM_OP_F64_CONST);
        wb_f64(&body, u->arg.f);
      } else if (poly_dtype_is_float(u->dtype)) {
        wb_byte(&body, WASM_OP_F32_CONST);
        wb_f32(&body, (float)u->arg.f);
      } else if (dt_is_i64(u->dtype)) {
        wb_byte(&body, WASM_OP_I64_CONST);
        wb_sleb128(&body, wasm_const_integer_bits(u));
      } else {
        wb_byte(&body, WASM_OP_I32_CONST);
        wb_sleb128(&body, (int32_t)wasm_const_integer_bits(u));
      }
      wb_byte(&body, WASM_OP_LOCAL_SET);
      wb_uleb128(&body, local_idx);
      lm_set(&locals, u, local_idx);
      continue;
    }

    /* --- register buffer --- */
    if (u->op == POLY_OP_BUFFER && poly_program_memory_is(u, POLY_ADDR_REG)) {
      PolyDType base = wasm_reg_base_dtype(poly_program_buffer_dtype(u));
      int reg_size = wasm_reg_storage_size(uops, n, u);
      int *reg_slots = malloc((size_t)reg_size * sizeof(int));
      if (!reg_slots) reg_size = 0;
      for (int r = 0; r < reg_size; r++) {
        int local_idx =
            alloc_local(base, false, &next_i32, &next_i64, &next_f32, &next_f64, &next_v128);
        reg_slots[r] = local_idx;
        if (dt_is_f64(base)) {
          wb_byte(&body, WASM_OP_F64_CONST);
          wb_f64(&body, 0.0);
        } else if (poly_dtype_is_float(base)) {
          wb_byte(&body, WASM_OP_F32_CONST);
          wb_f32(&body, 0.0f);
        } else if (dt_is_i64(base)) {
          wb_byte(&body, WASM_OP_I64_CONST);
          wb_sleb128(&body, 0);
        } else {
          wb_byte(&body, WASM_OP_I32_CONST);
          wb_sleb128(&body, 0);
        }
        wb_byte(&body, WASM_OP_LOCAL_SET);
        wb_uleb128(&body, local_idx);
      }
      if (reg_size > 0) {
        rlm_set(&reg_locals, u, reg_slots, reg_size);
        lm_set(&locals, u, reg_slots[0]);
      }
      continue;
    }

    /* --- AFTER --- */
    if (u->op == POLY_OP_AFTER) {
      int src_local = lm_get(&locals, u->src[0]);
      if (src_local >= 0) lm_set(&locals, u, src_local);
      continue;
    }

    /* --- VECTORIZE: construct packed value from scalar lanes --- */
    if (u->op == POLY_OP_STACK) {
      if (u->n_src == 1) {
        int src_local = lm_get(&locals, u->src[0]);
        if (src_local >= 0) lm_set(&locals, u, src_local);
        continue;
      }

      PolyDType value_dtype = wasm_local_value_dtype(u);
      int local_idx = alloc_local(
          value_dtype, wasm_uop_value_is_v128(ctx, u), &next_i32, &next_i64, &next_f32, &next_f64,
          &next_v128
      );
      bool emitted_fused = false;
      if (fused_stack && fused_stack[i]) {
        int start = body.len;
        emitted_fused = wasm_emit_structural_vec_expr(ctx, &body, &locals, u->src, u->n_src);
        if (!emitted_fused) body.len = start;
      }
      if (!emitted_fused && wasm_uop_value_is_v128(ctx, u) && u->n_src > 0) {
        int lanes = u->n_src;
        int n_lanes = u->n_src < lanes ? u->n_src : lanes;
        emit_v128_zero(&body);
        for (int j = 0; j < n_lanes; j++) {
          int sj = lm_get(&locals, u->src[j]);
          wb_byte(&body, WASM_OP_LOCAL_GET);
          wb_uleb128(&body, sj);
          emit_cast_stack_value(&body, u->src[j]->dtype, value_dtype);
          emit_v128_replace_lane(&body, value_dtype, j);
        }
      } else if (!emitted_fused) {
        if (u->n_src > 0) {
          int src_local = lm_get(&locals, u->src[0]);
          wb_byte(&body, WASM_OP_LOCAL_GET);
          wb_uleb128(&body, src_local);
          emit_cast_stack_value(&body, u->src[0]->dtype, u->dtype);
        } else if (dt_is_f64(u->dtype)) {
          wb_byte(&body, WASM_OP_F64_CONST);
          wb_f64(&body, 0.0);
        } else if (poly_dtype_is_float(u->dtype)) {
          wb_byte(&body, WASM_OP_F32_CONST);
          wb_f32(&body, 0.0f);
        } else if (dt_is_i64(u->dtype)) {
          wb_byte(&body, WASM_OP_I64_CONST);
          wb_sleb128(&body, 0);
        } else {
          wb_byte(&body, WASM_OP_I32_CONST);
          wb_sleb128(&body, 0);
        }
      }
      wb_byte(&body, WASM_OP_LOCAL_SET);
      wb_uleb128(&body, local_idx);
      lm_set(&locals, u, local_idx);
      continue;
    }

    /* --- SHRINK: late codegen memory slice, base + idx * scalar elem_size --- */
    if (u->op == POLY_OP_SHRINK) {
      if (wasm_is_i32_address_base(u->src[0])) {
        int local_idx = next_i32++;
        int base = lm_get(&locals, u->src[0]);
        int idx = lm_get(&locals, u->src[1]);
        PolyDType scalar = u->dtype;
        int elem_size = poly_dtype_itemsize(scalar);
        if (elem_size < 1) elem_size = 1;

        wb_byte(&body, WASM_OP_LOCAL_GET);
        wb_uleb128(&body, base);
        emit_local_get_as_i32(&body, idx, u->src[1]->dtype);
        wb_byte(&body, WASM_OP_I32_CONST);
        wb_sleb128(&body, elem_size);
        wb_byte(&body, WASM_OP_I32_MUL);
        wb_byte(&body, WASM_OP_I32_ADD);
        wb_byte(&body, WASM_OP_LOCAL_SET);
        wb_uleb128(&body, local_idx);
        lm_set(&locals, u, local_idx);
      } else {
        int src_local = lm_get(&locals, u->src[0]);
        if (src_local >= 0) lm_set(&locals, u, src_local);
      }
      continue;
    }

    /* --- INDEX: base + idx * elem_size (byte offset) --- */
    if (u->op == POLY_OP_INDEX) {
      PolyUOp *acc_base = wasm_acc_base(u->src[0]);
      if (acc_base && acc_base->op == POLY_OP_BUFFER &&
          poly_program_memory_is(acc_base, POLY_ADDR_REG)) {
        /* Register arrays are real indexed storage in tinygrad/C/WGSL. WASM
         * has no local arrays, but Qwen/reduce lowering indexes them with
         * constants after devectorization, so model each element as its own
         * local and map INDEX(reg, const_i) to that local. */
        int acc_local =
            rlm_get(&reg_locals, acc_base, u->n_src >= 2 ? wasm_const_index(u->src[1]) : 0);
        if (acc_local < 0) acc_local = lm_get(&locals, acc_base);
        lm_set(&locals, u, acc_local);
      } else if (wasm_is_i32_address_base(u->src[0])) {
        int local_idx = next_i32++;
        int base = lm_get(&locals, u->src[0]);
        int idx = lm_get(&locals, u->src[1]);
        int elem_size = dt_elem_size(u->src[0]->dtype);

        wb_byte(&body, WASM_OP_LOCAL_GET);
        wb_uleb128(&body, base);
        emit_local_get_as_i32(&body, idx, u->src[1]->dtype);
        wb_byte(&body, WASM_OP_I32_CONST);
        wb_sleb128(&body, elem_size);
        wb_byte(&body, WASM_OP_I32_MUL);
        wb_byte(&body, WASM_OP_I32_ADD);
        wb_byte(&body, WASM_OP_LOCAL_SET);
        wb_uleb128(&body, local_idx);
        lm_set(&locals, u, local_idx);
      } else {
        int local_idx =
            alloc_local(u->dtype, false, &next_i32, &next_i64, &next_f32, &next_f64, &next_v128);
        int src = lm_get(&locals, u->src[0]);
        int lane = wasm_const_index(u->src[1]);
        wb_byte(&body, WASM_OP_LOCAL_GET);
        wb_uleb128(&body, src);
        bool src_v128 = wasm_uop_value_is_v128(ctx, u->src[0]);
        PolyDType src_dt = src_v128 ? wasm_native_v128_dtype(u->src[0]) : u->src[0]->dtype;
        if (src_v128) {
          emit_v128_extract_lane(&body, src_dt, lane);
          emit_cast_stack_value(&body, src_dt, u->dtype);
        }
        wb_byte(&body, WASM_OP_LOCAL_SET);
        wb_uleb128(&body, local_idx);
        lm_set(&locals, u, local_idx);
      }
      continue;
    }

    /* --- RANGE --- */
    if (u->op == POLY_OP_RANGE) {
      int counter = next_i32++;
      lm_set(&locals, u, counter);

      wb_byte(&body, WASM_OP_I32_CONST);
      wb_sleb128(&body, 0);
      wb_byte(&body, WASM_OP_LOCAL_SET);
      wb_uleb128(&body, counter);

      wb_byte(&body, WASM_OP_BLOCK);
      wb_byte(&body, WASM_BLOCKTYPE_VOID);
      wb_byte(&body, WASM_OP_LOOP);
      wb_byte(&body, WASM_BLOCKTYPE_VOID);

      int bound = lm_get(&locals, u->src[0]);
      wb_byte(&body, WASM_OP_LOCAL_GET);
      wb_uleb128(&body, counter);
      wb_byte(&body, WASM_OP_LOCAL_GET);
      wb_uleb128(&body, bound);
      wb_byte(&body, WASM_OP_I32_GE_U);
      wb_byte(&body, WASM_OP_BR_IF);
      wb_uleb128(&body, 1);
      continue;
    }

    /* --- END --- */
    if (u->op == POLY_OP_END) {
      PolyUOp *range = NULL;
      for (int j = 0; j < u->n_src; j++) {
        if (u->src[j]->op == POLY_OP_RANGE) {
          range = u->src[j];
          break;
        }
      }
      if (range) {
        int counter = lm_get(&locals, range);
        wb_byte(&body, WASM_OP_LOCAL_GET);
        wb_uleb128(&body, counter);
        wb_byte(&body, WASM_OP_I32_CONST);
        wb_sleb128(&body, 1);
        wb_byte(&body, WASM_OP_I32_ADD);
        wb_byte(&body, WASM_OP_LOCAL_SET);
        wb_uleb128(&body, counter);
        wb_byte(&body, WASM_OP_BR);
        wb_uleb128(&body, 0);
      }
      wb_byte(&body, WASM_OP_END);
      wb_byte(&body, WASM_OP_END);
      continue;
    }

    /* --- LOAD --- */
    if (u->op == POLY_OP_LOAD) {
      PolyUOp *ld_idx = poly_find_index_through_cast(u->src[0]);
      PolyUOp *reg_base = NULL;
      int reg_lane = -1;
      bool is_reg = wasm_reg_addr_index(u->src[0], &reg_base, &reg_lane);
      PolyDType shrink_dtype;
      int shrink_lanes = 1;
      if (wasm_load_shrink_native_vec(u, &shrink_dtype, &shrink_lanes)) {
        int local_idx =
            alloc_local(shrink_dtype, true, &next_i32, &next_i64, &next_f32, &next_f64, &next_v128);
        PolyUOp *shr = wasm_load_shrink(u);
        PolyUOp *reg_base = wasm_acc_base(shr->src[0]);
        if (reg_base && reg_base->op == POLY_OP_BUFFER &&
            poly_program_memory_is(reg_base, POLY_ADDR_REG)) {
          int offset = wasm_shrink_offset(shr);
          int lanes = shrink_lanes;
          emit_v128_zero(&body);
          for (int lane = 0; lane < lanes; lane++) {
            int src_local = rlm_get(&reg_locals, reg_base, offset + lane);
            wb_byte(&body, WASM_OP_LOCAL_GET);
            wb_uleb128(&body, src_local);
            emit_cast_stack_value(&body, wasm_reg_base_dtype(reg_base->dtype), shrink_dtype);
            emit_v128_replace_lane(&body, shrink_dtype, lane);
          }
        } else {
          int addr = lm_get(&locals, u->src[0]);
          wb_byte(&body, WASM_OP_LOCAL_GET);
          wb_uleb128(&body, addr);
          emit_v128_load_opcode(&body, shrink_dtype, shrink_lanes);
        }
        wb_byte(&body, WASM_OP_LOCAL_SET);
        wb_uleb128(&body, local_idx);
        lm_set(&locals, u, local_idx);
      } else if (is_reg) {
        int acc_local = rlm_get(&reg_locals, reg_base, reg_lane);
        if (acc_local < 0) {
          acc_local = lm_get(&locals, u->src[0]);
          if (acc_local < 0) acc_local = lm_get(&locals, ld_idx);
        }
        lm_set(&locals, u, acc_local);
      } else {
        int local_idx = alloc_local(
            u->dtype, wasm_uop_value_is_v128(ctx, u), &next_i32, &next_i64, &next_f32, &next_f64,
            &next_v128
        );
        int addr = lm_get(&locals, u->src[0]);

        /* Pinned tinygrad final IR: LOAD(INDEX(buf, idx), alt, gate). */
        PolyUOp *gate_uop =
            (u->n_src >= 3 && poly_dtype_is_bool(u->src[2]->dtype)) ? u->src[2] : NULL;
        bool gated = gate_uop != NULL;

        if (gated) {
          /* if (gate) { val = load } else { val = alt/0 } */
          int gate_local = lm_get(&locals, gate_uop);
          wb_byte(&body, WASM_OP_LOCAL_GET);
          wb_uleb128(&body, gate_local);
          /* blocktype: result type of if-else */
          bool load_v128 = wasm_uop_value_is_v128(ctx, u);
          uint8_t bt = load_v128                       ? WASM_TYPE_V128
                       : dt_is_f64(u->dtype)           ? WASM_TYPE_F64
                       : poly_dtype_is_float(u->dtype) ? WASM_TYPE_F32
                                                       : WASM_TYPE_I32;
          wb_byte(&body, WASM_OP_IF);
          wb_byte(&body, bt);
        }

        wb_byte(&body, WASM_OP_LOCAL_GET);
        wb_uleb128(&body, addr);

        if (wasm_uop_value_is_v128(ctx, u)) {
          int64_t lanes = poly_uop_max_numel(ctx, u);
          emit_v128_load_opcode(&body, u->dtype, (int)lanes);
        } else
          emit_scalar_load_opcode(&body, u->dtype);

        if (gated) {
          wb_byte(&body, WASM_OP_ELSE);
          /* Push alt value: use src[1] if 2-source LOAD, else const 0 */
          if (u->n_src >= 2) {
            int alt_local = lm_get(&locals, u->src[1]);
            wb_byte(&body, WASM_OP_LOCAL_GET);
            wb_uleb128(&body, alt_local);
          } else {
            if (wasm_uop_value_is_v128(ctx, u)) {
              emit_v128_zero(&body);
            } else if (dt_is_f64(u->dtype)) {
              wb_byte(&body, WASM_OP_F64_CONST);
              wb_f64(&body, 0.0);
            } else if (poly_dtype_is_float(u->dtype)) {
              wb_byte(&body, WASM_OP_F32_CONST);
              wb_f32(&body, 0.0f);
            } else {
              wb_byte(&body, WASM_OP_I32_CONST);
              wb_sleb128(&body, 0);
            }
          }
          wb_byte(&body, WASM_OP_END);
        }

        wb_byte(&body, WASM_OP_LOCAL_SET);
        wb_uleb128(&body, local_idx);
        lm_set(&locals, u, local_idx);
      }
      continue;
    }

    /* --- STORE --- */
    if (u->op == POLY_OP_STORE) {
      int val = lm_get(&locals, u->src[1]);
      bool is_reg = false;
      PolyUOp *reg_base = NULL;
      int reg_lane = -1;
      if (wasm_reg_addr_index(u->src[0], &reg_base, &reg_lane)) {
        is_reg = true;
      }
      if (is_reg) {
        int acc_local = (reg_base && reg_lane >= 0) ? rlm_get(&reg_locals, reg_base, reg_lane)
                                                    : lm_get(&locals, u->src[0]);
        wb_byte(&body, WASM_OP_LOCAL_GET);
        wb_uleb128(&body, val);
        wb_byte(&body, WASM_OP_LOCAL_SET);
        wb_uleb128(&body, acc_local);
      } else {
        int addr = lm_get(&locals, u->src[0]);
        PolyDType val_dt = wasm_local_value_dtype(u->src[1]);
        bool val_is_v128 = wasm_uop_value_is_v128(ctx, u->src[1]);
        bool val_is_float = poly_dtype_is_float(val_dt);
        bool buf_is_float = poly_dtype_is_float(u->src[0]->dtype);
        bool buf_is_f64 = dt_is_f64(u->src[0]->dtype);
        wb_byte(&body, WASM_OP_LOCAL_GET);
        wb_uleb128(&body, addr);
        wb_byte(&body, WASM_OP_LOCAL_GET);
        wb_uleb128(&body, val);

        /* Type conversions for mismatched value/buffer dtypes */
        if (!val_is_v128 && buf_is_f64 && !val_is_float) {
          /* i32/i64 → f64 */
          if (dt_is_i64(val_dt))
            wb_byte(&body, WASM_OP_F64_CONVERT_I64_S);
          else
            wb_byte(&body, WASM_OP_F64_CONVERT_I32_S);
        } else if (!val_is_v128 && buf_is_float && !buf_is_f64 && !val_is_float) {
          /* i32 → f32 */
          wb_byte(&body, WASM_OP_F32_CONVERT_I32_S);
        }

        PolyDType buf_dt = u->src[0]->dtype;
        if (val_is_v128) {
          int64_t lanes = poly_uop_max_numel(ctx, u->src[1]);
          emit_v128_store_opcode(&body, val_dt, (int)lanes);
        } else
          emit_scalar_store_opcode(&body, buf_dt);
      }
      continue;
    }

    /* --- CAST / BITCAST --- */
    if (u->op == POLY_OP_CAST || u->op == POLY_OP_BITCAST) {
      if (wasm_casted_const_identity(u)) {
        int src = lm_get(&locals, u->src[0]);
        if (src >= 0) lm_set(&locals, u, src);
        continue;
      }
      if (u->op == POLY_OP_CAST && wasm_is_casted_const(u)) {
        int local_idx =
            alloc_local(u->dtype, false, &next_i32, &next_i64, &next_f32, &next_f64, &next_v128);
        if (dt_is_f64(u->dtype)) {
          wb_byte(&body, WASM_OP_F64_CONST);
          wb_f64(&body, u->src[0]->arg.f);
        } else if (poly_dtype_is_float(u->dtype)) {
          wb_byte(&body, WASM_OP_F32_CONST);
          wb_f32(&body, (float)u->src[0]->arg.f);
        } else if (dt_is_i64(u->dtype)) {
          wb_byte(&body, WASM_OP_I64_CONST);
          wb_sleb128(&body, wasm_const_integer_bits(u->src[0]));
        } else {
          wb_byte(&body, WASM_OP_I32_CONST);
          wb_sleb128(&body, (int32_t)wasm_const_integer_bits(u->src[0]));
        }
        wb_byte(&body, WASM_OP_LOCAL_SET);
        wb_uleb128(&body, local_idx);
        lm_set(&locals, u, local_idx);
        continue;
      }
      int local_idx = alloc_local(
          u->dtype, wasm_uop_value_is_v128(ctx, u), &next_i32, &next_i64, &next_f32, &next_f64,
          &next_v128
      );
      int src = lm_get(&locals, u->src[0]);
      wb_byte(&body, WASM_OP_LOCAL_GET);
      wb_uleb128(&body, src);

      PolyDType src_dt = wasm_local_value_dtype(u->src[0]);
      PolyDType dst_dt = u->dtype;
      bool src_v128 = wasm_uop_value_is_v128(ctx, u->src[0]);
      bool dst_v128 = wasm_uop_value_is_v128(ctx, u);
      int src_lanes = src_v128 ? (int)poly_uop_max_numel(ctx, u->src[0]) : 1;
      int dst_lanes = dst_v128 ? (int)poly_uop_max_numel(ctx, u) : 1;

      if (src_v128 || dst_v128) {
        if (src_v128 && dst_v128) {
          if (emit_v128_cast_opcode(&body, src_dt, dst_dt, u->op == POLY_OP_BITCAST)) {
            /* emitted directly over the loaded v128 */
          } else {
            /* Drop the already-loaded value by routing through the source local
             * again. Unsupported vector casts are rare; preserve correctness by
             * rebuilding the destination vector from scalar lane conversions. */
            wb_byte(&body, WASM_OP_DROP);
            emit_v128_cast_lanes(&body, src, src_dt, src_lanes, dst_dt, dst_lanes);
          }
        } else if (src_v128) {
          emit_v128_extract_lane(&body, src_dt, 0);
          emit_cast_stack_value(&body, src_dt, dst_dt);
        } else {
          emit_cast_stack_value(&body, src_dt, dst_dt);
          emit_v128_splat(&body, dst_dt);
        }

        wb_byte(&body, WASM_OP_LOCAL_SET);
        wb_uleb128(&body, local_idx);
        lm_set(&locals, u, local_idx);
        continue;
      }

      bool src_float = poly_dtype_is_float(src_dt);
      bool dst_float = poly_dtype_is_float(dst_dt);
      bool src_64 = dt_is_64(src_dt);
      bool dst_64 = dt_is_64(dst_dt);

      if (u->op == POLY_OP_BITCAST) {
        /* Bit-level reinterpret */
        if (src_float && !dst_float) {
          if (src_64)
            wb_byte(&body, WASM_OP_I64_REINTERPRET_F64);
          else
            wb_byte(&body, WASM_OP_I32_REINTERPRET_F32);
        } else if (!src_float && dst_float) {
          if (dst_64)
            wb_byte(&body, WASM_OP_F64_REINTERPRET_I64);
          else
            wb_byte(&body, WASM_OP_F32_REINTERPRET_I32);
        }
        /* same category: no-op (i32→i32, f32→f32) */
      } else {
        /* Value-converting CAST */
        if (src_float && dst_float) {
          /* f32→f64 or f64→f32 */
          if (!src_64 && dst_64)
            wb_byte(&body, WASM_OP_F64_PROMOTE_F32);
          else if (src_64 && !dst_64)
            wb_byte(&body, WASM_OP_F32_DEMOTE_F64);
        } else if (src_float && !dst_float) {
          /* float→int */
          bool dst_u = poly_dtype_is_unsigned(dst_dt);
          if (src_64 && dst_64)
            wb_byte(&body, dst_u ? WASM_OP_I64_TRUNC_F64_U : WASM_OP_I64_TRUNC_F64_S);
          else if (src_64 && !dst_64)
            wb_byte(&body, dst_u ? WASM_OP_I32_TRUNC_F64_U : WASM_OP_I32_TRUNC_F64_S);
          else if (!src_64 && dst_64) {
            wb_byte(&body, dst_u ? WASM_OP_I64_TRUNC_F32_U : WASM_OP_I64_TRUNC_F32_S);
          } else {
            wb_byte(&body, dst_u ? WASM_OP_I32_TRUNC_F32_U : WASM_OP_I32_TRUNC_F32_S);
          }
        } else if (!src_float && dst_float) {
          /* int→float */
          bool src_u = poly_dtype_is_unsigned(src_dt);
          if (src_64 && dst_64)
            wb_byte(&body, src_u ? WASM_OP_F64_CONVERT_I64_U : WASM_OP_F64_CONVERT_I64_S);
          else if (src_64 && !dst_64) {
            wb_byte(&body, src_u ? WASM_OP_F32_CONVERT_I64_U : WASM_OP_F32_CONVERT_I64_S);
          } else if (!src_64 && dst_64) {
            wb_byte(&body, src_u ? WASM_OP_F64_CONVERT_I32_U : WASM_OP_F64_CONVERT_I32_S);
          } else {
            wb_byte(&body, src_u ? WASM_OP_F32_CONVERT_I32_U : WASM_OP_F32_CONVERT_I32_S);
          }
        } else {
          /* int→int */
          if (!src_64 && dst_64) {
            bool src_u = poly_dtype_is_unsigned(src_dt);
            wb_byte(&body, src_u ? WASM_OP_I64_EXTEND_I32_U : WASM_OP_I64_EXTEND_I32_S);
          } else if (src_64 && !dst_64)
            wb_byte(&body, WASM_OP_I32_WRAP_I64);
          /* same size: no-op */
        }
      }

      wb_byte(&body, WASM_OP_LOCAL_SET);
      wb_uleb128(&body, local_idx);
      lm_set(&locals, u, local_idx);
      continue;
    }

    /* --- ALU --- */
    if (poly_opset_has(POLY_GROUP_ALU, u->op)) {
      bool vector_alu = wasm_vector_alu_has_direct_simd(ctx, u);
      bool vector_lane_fallback = wasm_vector_alu_needs_lane_fallback(ctx, u);
      PolyDType local_dt = wasm_local_value_dtype(u);
      int local_idx = alloc_local(
          local_dt, vector_alu || vector_lane_fallback, &next_i32, &next_i64, &next_f32, &next_f64,
          &next_v128
      );

      /* RECIPROCAL: push 1.0 first, then src, then div */
      if (!vector_alu && !vector_lane_fallback && u->op == POLY_OP_RECIPROCAL) {
        if (dt_is_f64(u->dtype)) {
          wb_byte(&body, WASM_OP_F64_CONST);
          wb_f64(&body, 1.0);
        } else {
          wb_byte(&body, WASM_OP_F32_CONST);
          wb_f32(&body, 1.0f);
        }
      }

      /* Integer NEG: emit 0 before src. Bool NEG is typed identity and is
       * normalized by two eqz instructions in emit_alu_scalar. */
      if (!vector_alu && !vector_lane_fallback && u->op == POLY_OP_NEG &&
          !poly_dtype_is_float(u->dtype) && !poly_dtype_is_bool(u->dtype)) {
        if (dt_is_i64(u->dtype)) {
          wb_byte(&body, WASM_OP_I64_CONST);
          wb_sleb128(&body, 0);
        } else {
          wb_byte(&body, WASM_OP_I32_CONST);
          wb_sleb128(&body, 0);
        }
      }

      /* Push operands (WHERE needs special order for WASM select).
       *
       * WASM select: (val_true, val_false, i32_cond).
       * The condition may be f32 (e.g. from poly_eq which returns a
       * float mask via WHERE(cmplt, 0.0, 1.0)). Convert to i32 via
       * f32.ne 0.0 before select. Same for f64 and i64 conditions. */
      if (vector_alu) {
        emit_vector_sources(ctx, &body, &locals, u);
      } else if (vector_lane_fallback) {
        emit_vector_alu_lane_fallback(&body, &locals, ctx, u, math, n_imported_funcs);
      } else {
        emit_alu_sources(&body, &locals, u);
      }

      /* Use input dtype for comparison ops */
      PolyDType alu_dtype = u->dtype;
      if (u->op == POLY_OP_CMPLT || u->op == POLY_OP_CMPEQ || u->op == POLY_OP_CMPNE) {
        alu_dtype = wasm_compare_dtype(u->src[0]->dtype, u->src[1]->dtype);
      }
      if (vector_alu) {
        PolyDType scalar = wasm_simd_value_dtype(u);
        if (scalar.bitsize == 64)
          emit_alu_simd_f64x2(&body, u->op);
        else
          emit_alu_simd_f32x4(&body, u->op);
      } else if (vector_lane_fallback) {
        /* already emitted and packed lane-wise */
      } else {
        emit_alu_scalar(&body, u->op, alu_dtype, math, n_imported_funcs);
      }

      wb_byte(&body, WASM_OP_LOCAL_SET);
      wb_uleb128(&body, local_idx);
      lm_set(&locals, u, local_idx);
      continue;
    }

    /* --- IF --- */
    if (u->op == POLY_OP_IF) {
      int cond = lm_get(&locals, u->src[0]);
      wb_byte(&body, WASM_OP_LOCAL_GET);
      wb_uleb128(&body, cond);
      wb_byte(&body, WASM_OP_IF);
      wb_byte(&body, WASM_BLOCKTYPE_VOID);
      continue;
    }

    /* --- ENDIF --- */
    if (u->op == POLY_OP_ENDIF) {
      wb_byte(&body, WASM_OP_END);
      continue;
    }
  }

  /* Function end */
  wb_byte(&body, WASM_OP_END);

  /* Wrap in code section */
  WasmBuf sec;
  wb_init(&sec);
  wb_uleb128(&sec, 1);
  wb_uleb128(&sec, body.len);
  wb_append(&sec, &body);
  wb_section(mod, WASM_SEC_CODE, &sec);

  wb_free(&body);
  wb_free(&sec);
  rlm_destroy(&reg_locals);
  lm_destroy(&locals);
  free(skip);
  free(fused_stack);
}

/* Build code section (SIMD) */

static void build_code_simd(
    PolyCtx *ctx,
    WasmBuf *mod,
    PolyUOp **uops,
    int n,
    int n_params,
    MathImports *math,
    int n_imported_funcs
) {
  /* SIMD codegen: split the innermost loop into:
   *   main loop:  i += lanes, v128 ops (f32x4: 4 lanes, f64x2: 2 lanes)
   *   epilogue:   i += 1, scalar ops (for remainder)
   *
   * For now, find the single RANGE/END pair and generate both loops.
   * If the kernel is not SIMD-able, fall back to scalar. */

  if (!kernel_is_simdable(uops, n)) {
    build_code_scalar(ctx, mod, uops, n, n_params, math, n_imported_funcs);
    return;
  }

  /* Find the RANGE and its bound */
  PolyUOp *range_uop = NULL;
  int range_bound_local = -1;
  for (int i = 0; i < n; i++) {
    if (uops[i]->op == POLY_OP_RANGE) {
      range_uop = uops[i];
      break;
    }
  }
  if (!range_uop) {
    /* No loop — just emit scalar */
    build_code_scalar(ctx, mod, uops, n, n_params, math, n_imported_funcs);
    return;
  }

  /* --- Detect kernel dtype for SIMD lane width --- */
  bool is_f64_kernel = kernel_is_f64(uops, n);
  int lanes = is_f64_kernel ? 2 : 4; /* f64x2: 2 lanes, f32x4: 4 lanes */
  int lane_mask = ~(lanes - 1); /* ~1 for f64x2, ~3 for f32x4 */
  int scalar_elem_size = is_f64_kernel ? 8 : 4;

  /* --- Count locals --- */
  /* For SIMD we need: i32 counter, i32 bound, i32 simd_bound,
   * plus i32 for each INDEX (x2 for simd+scalar),
   * v128 for each LOAD and ALU (SIMD path),
   * f32/f64 for each LOAD and ALU (scalar path). */

  int n_locals_i32 = 0, n_locals_i64 = 0, n_locals_f32 = 0, n_locals_f64 = 0, n_locals_v128 = 0;

  /* Count ops that need locals */
  int n_indices = 0, n_loads = 0, n_simd_alus = 0;
  for (int i = 0; i < n; i++) {
    PolyUOp *u = uops[i];
    if (u->op == POLY_OP_INDEX) n_indices++;
    if (u->op == POLY_OP_LOAD) n_loads++;
    if (poly_opset_has(POLY_GROUP_ALU, u->op)) {
      if (wasm_simd_loop_can_vectorize_alu(u))
        n_simd_alus++;
      else
        count_local(
            u->dtype, false, &n_locals_i32, &n_locals_i64, &n_locals_f32, &n_locals_f64,
            &n_locals_v128
        );
    }
    if (u->op == POLY_OP_CONST && !poly_dtype_is_float(u->dtype)) n_locals_i32++;
    if (u->op == POLY_OP_CONST && poly_dtype_is_float(u->dtype)) {
      if (dt_is_f64(u->dtype))
        n_locals_f64++;
      else
        n_locals_f32++;
    }
  }

  /* SIMD loop needs: counter, bound const, simd_bound */
  n_locals_i32 += 3 + n_indices * 2; /* indices for both simd and scalar paths */
  if (is_f64_kernel)
    n_locals_f64 += n_loads + n_simd_alus; /* scalar epilogue */
  else
    n_locals_f32 += n_loads + n_simd_alus;
  n_locals_v128 += n_loads + n_simd_alus; /* SIMD main loop */

  /* Function body */
  WasmBuf body;
  wb_init(&body);

  /* Declare locals */
  int n_local_types = 0;
  if (n_locals_i32 > 0) n_local_types++;
  if (n_locals_i64 > 0) n_local_types++;
  if (n_locals_f32 > 0) n_local_types++;
  if (n_locals_f64 > 0) n_local_types++;
  if (n_locals_v128 > 0) n_local_types++;

  wb_uleb128(&body, n_local_types);
  if (n_locals_i32 > 0) {
    wb_uleb128(&body, n_locals_i32);
    wb_byte(&body, WASM_TYPE_I32);
  }
  if (n_locals_i64 > 0) {
    wb_uleb128(&body, n_locals_i64);
    wb_byte(&body, WASM_TYPE_I64);
  }
  if (n_locals_f32 > 0) {
    wb_uleb128(&body, n_locals_f32);
    wb_byte(&body, WASM_TYPE_F32);
  }
  if (n_locals_f64 > 0) {
    wb_uleb128(&body, n_locals_f64);
    wb_byte(&body, WASM_TYPE_F64);
  }
  if (n_locals_v128 > 0) {
    wb_uleb128(&body, n_locals_v128);
    wb_byte(&body, WASM_TYPE_V128);
  }

  /* Local index assignment */
  LocalMap locals;
  lm_init(&locals, n * 2);

  /* Params: 0..n_params-1 */
  int next_i32 = n_params;
  int next_i64 = n_params + n_locals_i32;
  int next_f32 = n_params + n_locals_i32 + n_locals_i64;
  int next_f64 = n_params + n_locals_i32 + n_locals_i64 + n_locals_f32;
  int next_v128 = n_params + n_locals_i32 + n_locals_i64 + n_locals_f32 + n_locals_f64;

  /* Assign PARAM locals */
  for (int i = 0; i < n; i++) {
    if (uops[i]->op == POLY_OP_PARAM) {
      lm_set(&locals, uops[i], wasm_param_abi_index(uops, n, uops[i]));
    }
  }

  /* Handle CONST first (outside any loop) */
  for (int i = 0; i < n; i++) {
    PolyUOp *u = uops[i];
    if (u->op == POLY_OP_CONST) {
      int local_idx;
      if (dt_is_f64(u->dtype)) {
        local_idx = next_f64++;
        wb_byte(&body, WASM_OP_F64_CONST);
        wb_f64(&body, u->arg.f);
      } else if (poly_dtype_is_float(u->dtype)) {
        local_idx = next_f32++;
        wb_byte(&body, WASM_OP_F32_CONST);
        wb_f32(&body, (float)u->arg.f);
      } else {
        local_idx = next_i32++;
        wb_byte(&body, WASM_OP_I32_CONST);
        wb_sleb128(&body, (int32_t)wasm_const_integer_bits(u));
      }
      wb_byte(&body, WASM_OP_LOCAL_SET);
      wb_uleb128(&body, local_idx);
      lm_set(&locals, u, local_idx);

      /* Track bound for the RANGE */
      if (u == range_uop->src[0]) range_bound_local = local_idx;
    }
  }

  /* Counter local and SIMD bound local */
  int counter_local = next_i32++;
  int simd_bound_local = next_i32++;

  /* Compute simd_bound = bound & lane_mask (f32x4: ~3, f64x2: ~1) */
  wb_byte(&body, WASM_OP_LOCAL_GET);
  wb_uleb128(&body, range_bound_local);
  wb_byte(&body, WASM_OP_I32_CONST);
  wb_sleb128(&body, lane_mask);
  wb_byte(&body, WASM_OP_I32_AND);
  wb_byte(&body, WASM_OP_LOCAL_SET);
  wb_uleb128(&body, simd_bound_local);

  /* ═══ SIMD main loop: for (i = 0; i < simd_bound; i += lanes) ═══ */
  wb_byte(&body, WASM_OP_I32_CONST);
  wb_sleb128(&body, 0);
  wb_byte(&body, WASM_OP_LOCAL_SET);
  wb_uleb128(&body, counter_local);

  /* Assign counter as the RANGE local for SIMD path */
  lm_set(&locals, range_uop, counter_local);

  wb_byte(&body, WASM_OP_BLOCK);
  wb_byte(&body, WASM_BLOCKTYPE_VOID);
  wb_byte(&body, WASM_OP_LOOP);
  wb_byte(&body, WASM_BLOCKTYPE_VOID);

  /* if (counter >= simd_bound) break */
  wb_byte(&body, WASM_OP_LOCAL_GET);
  wb_uleb128(&body, counter_local);
  wb_byte(&body, WASM_OP_LOCAL_GET);
  wb_uleb128(&body, simd_bound_local);
  wb_byte(&body, WASM_OP_I32_GE_U);
  wb_byte(&body, WASM_OP_BR_IF);
  wb_uleb128(&body, 1);

  /* SIMD loop body: process INDEX, LOAD, ALU, STORE with v128 */
  for (int i = 0; i < n; i++) {
    PolyUOp *u = uops[i];

    if (wasm_casted_const_identity(u)) {
      int src = lm_get(&locals, u->src[0]);
      if (src >= 0) lm_set(&locals, u, src);
      continue;
    }

    if (u->op == POLY_OP_INDEX) {
      int local_idx = next_i32++;
      int base = lm_get(&locals, u->src[0]);
      int idx = lm_get(&locals, u->src[1]);

      wb_byte(&body, WASM_OP_LOCAL_GET);
      wb_uleb128(&body, base);
      emit_local_get_as_i32(&body, idx, u->src[1]->dtype);
      wb_byte(&body, WASM_OP_I32_CONST);
      /* The SIMD loop counter is an element index and advances by lane count,
       * so address calculation stays base + element_index * scalar_size. */
      wb_sleb128(&body, scalar_elem_size);
      wb_byte(&body, WASM_OP_I32_MUL);
      wb_byte(&body, WASM_OP_I32_ADD);
      wb_byte(&body, WASM_OP_LOCAL_SET);
      wb_uleb128(&body, local_idx);
      lm_set(&locals, u, local_idx);
    }

    if (u->op == POLY_OP_LOAD) {
      int local_idx = next_v128++;
      int addr = lm_get(&locals, u->src[0]);

      wb_byte(&body, WASM_OP_LOCAL_GET);
      wb_uleb128(&body, addr);
      wb_byte(&body, WASM_SIMD_PREFIX);
      wb_uleb128(&body, WASM_SIMD_V128_LOAD);
      wb_uleb128(&body, 2); /* align: 4 bytes */
      wb_uleb128(&body, 0); /* offset */
      wb_byte(&body, WASM_OP_LOCAL_SET);
      wb_uleb128(&body, local_idx);
      lm_set(&locals, u, local_idx);
    }

    if (poly_opset_has(POLY_GROUP_ALU, u->op)) {
      bool simd_alu = wasm_simd_loop_can_vectorize_alu(u);
      int local_idx =
          simd_alu ? next_v128++
                   : alloc_local(
                         u->dtype, false, &next_i32, &next_i64, &next_f32, &next_f64, &next_v128
                     );

      /* Push sources in renderer stack order, preserving tinygrad MULACC
       * semantics for scalar fallbacks inside mixed SIMD kernels. */
      if (simd_alu)
        emit_simd_loop_sources(&body, &locals, u);
      else if (u->op == POLY_OP_WHERE && u->n_src >= 3)
        emit_scalar_where_sources(&body, &locals, u);
      else
        emit_alu_sources(&body, &locals, u);

      if (simd_alu) {
        PolyDType simd_dt = wasm_simd_value_dtype(u);
        if (simd_dt.bitsize == 64)
          emit_alu_simd_f64x2(&body, u->op);
        else
          emit_alu_simd_f32x4(&body, u->op);
      } else {
        PolyDType alu_dtype = u->dtype;
        if (u->op == POLY_OP_CMPLT || u->op == POLY_OP_CMPEQ || u->op == POLY_OP_CMPNE)
          alu_dtype = wasm_compare_dtype(u->src[0]->dtype, u->src[1]->dtype);
        emit_alu_scalar(&body, u->op, alu_dtype, math, n_imported_funcs);
      }

      wb_byte(&body, WASM_OP_LOCAL_SET);
      wb_uleb128(&body, local_idx);
      lm_set(&locals, u, local_idx);
    }

    if (u->op == POLY_OP_STORE) {
      int addr = lm_get(&locals, u->src[0]);
      int val = lm_get(&locals, u->src[1]);

      wb_byte(&body, WASM_OP_LOCAL_GET);
      wb_uleb128(&body, addr);
      wb_byte(&body, WASM_OP_LOCAL_GET);
      wb_uleb128(&body, val);
      wb_byte(&body, WASM_SIMD_PREFIX);
      wb_uleb128(&body, WASM_SIMD_V128_STORE);
      wb_uleb128(&body, 2); /* align */
      wb_uleb128(&body, 0); /* offset */
    }
  }

  /* counter += lanes */
  wb_byte(&body, WASM_OP_LOCAL_GET);
  wb_uleb128(&body, counter_local);
  wb_byte(&body, WASM_OP_I32_CONST);
  wb_sleb128(&body, lanes);
  wb_byte(&body, WASM_OP_I32_ADD);
  wb_byte(&body, WASM_OP_LOCAL_SET);
  wb_uleb128(&body, counter_local);

  wb_byte(&body, WASM_OP_BR);
  wb_uleb128(&body, 0);
  wb_byte(&body, WASM_OP_END); /* end loop */
  wb_byte(&body, WASM_OP_END); /* end block */

  /* ═══ Scalar epilogue: for (i = simd_bound; i < bound; i++) ═══ */

  /* counter is already at simd_bound from the SIMD loop exit */

  wb_byte(&body, WASM_OP_BLOCK);
  wb_byte(&body, WASM_BLOCKTYPE_VOID);
  wb_byte(&body, WASM_OP_LOOP);
  wb_byte(&body, WASM_BLOCKTYPE_VOID);

  /* if (counter >= bound) break */
  wb_byte(&body, WASM_OP_LOCAL_GET);
  wb_uleb128(&body, counter_local);
  wb_byte(&body, WASM_OP_LOCAL_GET);
  wb_uleb128(&body, range_bound_local);
  wb_byte(&body, WASM_OP_I32_GE_U);
  wb_byte(&body, WASM_OP_BR_IF);
  wb_uleb128(&body, 1);

  /* Scalar epilogue body */
  for (int i = 0; i < n; i++) {
    PolyUOp *u = uops[i];

    if (wasm_casted_const_identity(u)) {
      int src = lm_get(&locals, u->src[0]);
      if (src >= 0) lm_set(&locals, u, src);
      continue;
    }

    if (u->op == POLY_OP_INDEX) {
      int local_idx = next_i32++;
      int base = lm_get(&locals, u->src[0]);

      wb_byte(&body, WASM_OP_LOCAL_GET);
      wb_uleb128(&body, base);
      wb_byte(&body, WASM_OP_LOCAL_GET);
      wb_uleb128(&body, counter_local); /* use counter directly */
      wb_byte(&body, WASM_OP_I32_CONST);
      wb_sleb128(&body, scalar_elem_size); /* 4 for f32, 8 for f64 */
      wb_byte(&body, WASM_OP_I32_MUL);
      wb_byte(&body, WASM_OP_I32_ADD);
      wb_byte(&body, WASM_OP_LOCAL_SET);
      wb_uleb128(&body, local_idx);
      lm_set(&locals, u, local_idx);
    }

    if (u->op == POLY_OP_LOAD) {
      int local_idx = is_f64_kernel ? next_f64++ : next_f32++;
      int addr = lm_get(&locals, u->src[0]);

      wb_byte(&body, WASM_OP_LOCAL_GET);
      wb_uleb128(&body, addr);
      emit_scalar_load_opcode(&body, u->dtype);
      wb_byte(&body, WASM_OP_LOCAL_SET);
      wb_uleb128(&body, local_idx);
      lm_set(&locals, u, local_idx);
    }

    if (poly_opset_has(POLY_GROUP_ALU, u->op)) {
      int local_idx;
      if (poly_dtype_is_float(u->dtype))
        local_idx = is_f64_kernel ? next_f64++ : next_f32++;
      else
        local_idx =
            alloc_local(u->dtype, false, &next_i32, &next_i64, &next_f32, &next_f64, &next_v128);

      if (u->op == POLY_OP_WHERE && u->n_src >= 3)
        emit_scalar_where_sources(&body, &locals, u);
      else
        emit_alu_sources(&body, &locals, u);

      PolyDType alu_dtype = u->dtype;
      if (u->op == POLY_OP_CMPLT || u->op == POLY_OP_CMPEQ || u->op == POLY_OP_CMPNE) {
        alu_dtype = wasm_compare_dtype(u->src[0]->dtype, u->src[1]->dtype);
      }
      emit_alu_scalar(&body, u->op, alu_dtype, math, n_imported_funcs);

      wb_byte(&body, WASM_OP_LOCAL_SET);
      wb_uleb128(&body, local_idx);
      lm_set(&locals, u, local_idx);
    }

    if (u->op == POLY_OP_STORE) {
      int addr = lm_get(&locals, u->src[0]);
      int val = lm_get(&locals, u->src[1]);
      bool buf_f64 = dt_is_f64(u->src[0]->dtype);
      bool buf_float = poly_dtype_is_float(u->src[0]->dtype);

      wb_byte(&body, WASM_OP_LOCAL_GET);
      wb_uleb128(&body, addr);
      wb_byte(&body, WASM_OP_LOCAL_GET);
      wb_uleb128(&body, val);
      /* Convert i32 → float when storing bool/int result into float buffer */
      if (!poly_dtype_is_float(u->src[1]->dtype) && buf_float) {
        if (buf_f64)
          wb_byte(&body, WASM_OP_F64_CONVERT_I32_S);
        else
          wb_byte(&body, WASM_OP_F32_CONVERT_I32_S);
      }
      PolyDType buf_dt = u->src[0]->dtype;
      emit_scalar_store_opcode(&body, buf_dt);
    }
  }

  /* counter++ */
  wb_byte(&body, WASM_OP_LOCAL_GET);
  wb_uleb128(&body, counter_local);
  wb_byte(&body, WASM_OP_I32_CONST);
  wb_sleb128(&body, 1);
  wb_byte(&body, WASM_OP_I32_ADD);
  wb_byte(&body, WASM_OP_LOCAL_SET);
  wb_uleb128(&body, counter_local);

  wb_byte(&body, WASM_OP_BR);
  wb_uleb128(&body, 0);
  wb_byte(&body, WASM_OP_END); /* end loop */
  wb_byte(&body, WASM_OP_END); /* end block */

  /* Function end */
  wb_byte(&body, WASM_OP_END);

  /* Wrap in code section */
  WasmBuf sec;
  wb_init(&sec);
  wb_uleb128(&sec, 1);
  wb_uleb128(&sec, body.len);
  wb_append(&sec, &body);
  wb_section(mod, WASM_SEC_CODE, &sec);

  wb_free(&body);
  wb_free(&sec);
  lm_destroy(&locals);
}

static bool wasm_const_i64(PolyUOp *u, int64_t *out) {
  if (!u || u->op != POLY_OP_CONST) return false;
  if (u->arg.kind == POLY_ARG_INT) {
    if (out) *out = u->arg.i;
    return true;
  }
  if (u->arg.kind == POLY_ARG_BIGINT) return poly_arg_integer_to_i64(u->arg, out);
  return false;
}

static bool wasm_range_bound(PolyUOp *u, int64_t *out) {
  if (!u || u->op != POLY_OP_RANGE || u->n_src < 1) return false;
  return wasm_const_i64(u->src[0], out);
}

typedef struct {
  int64_t c[3];
  int64_t offset;
} WasmAffine;

typedef struct {
  int m;
  int k;
  int out_param;
  PolyUOp *out_range;
  PolyUOp *red_range;
  PolyUOp *expr;
} WasmReduceSpec;

static bool wasm_affine_expr(PolyUOp *u, PolyUOp *r0, PolyUOp *r1, PolyUOp *r2, WasmAffine *out) {
  if (!u || !out) return false;
  memset(out, 0, sizeof(*out));
  if (u == r0) {
    out->c[0] = 1;
    return true;
  }
  if (u == r1) {
    out->c[1] = 1;
    return true;
  }
  if (u == r2) {
    out->c[2] = 1;
    return true;
  }
  int64_t v = 0;
  if (wasm_const_i64(u, &v)) {
    out->offset = v;
    return true;
  }
  if ((u->op == POLY_OP_ADD || u->op == POLY_OP_SUB) && u->n_src >= 2) {
    WasmAffine a, b;
    if (!wasm_affine_expr(u->src[0], r0, r1, r2, &a) ||
        !wasm_affine_expr(u->src[1], r0, r1, r2, &b))
      return false;
    for (int i = 0; i < 3; i++)
      out->c[i] = a.c[i] + (u->op == POLY_OP_SUB ? -b.c[i] : b.c[i]);
    out->offset = a.offset + (u->op == POLY_OP_SUB ? -b.offset : b.offset);
    return true;
  }
  if (u->op == POLY_OP_MUL && u->n_src >= 2) {
    int64_t scale = 0;
    PolyUOp *expr = NULL;
    if (wasm_const_i64(u->src[0], &scale)) {
      expr = u->src[1];
    } else if (wasm_const_i64(u->src[1], &scale)) {
      expr = u->src[0];
    } else {
      return false;
    }
    WasmAffine a;
    if (!wasm_affine_expr(expr, r0, r1, r2, &a)) return false;
    for (int i = 0; i < 3; i++)
      out->c[i] = a.c[i] * scale;
    out->offset = a.offset * scale;
    return true;
  }
  if (u->op == POLY_OP_SHL && u->n_src >= 2) {
    int64_t shift = 0;
    if (!wasm_const_i64(u->src[1], &shift) || shift < 0 || shift >= 62) return false;
    WasmAffine a;
    if (!wasm_affine_expr(u->src[0], r0, r1, r2, &a)) return false;
    int64_t scale = 1LL << shift;
    for (int i = 0; i < 3; i++)
      out->c[i] = a.c[i] * scale;
    out->offset = a.offset * scale;
    return true;
  }
  if (u->op == POLY_OP_MULACC && u->n_src >= 3) {
    int64_t scale = 0;
    PolyUOp *expr = NULL;
    if (wasm_const_i64(u->src[0], &scale)) {
      expr = u->src[1];
    } else if (wasm_const_i64(u->src[1], &scale)) {
      expr = u->src[0];
    } else {
      return false;
    }
    WasmAffine mul, add;
    if (!wasm_affine_expr(expr, r0, r1, r2, &mul) || !wasm_affine_expr(u->src[2], r0, r1, r2, &add))
      return false;
    for (int i = 0; i < 3; i++)
      out->c[i] = mul.c[i] * scale + add.c[i];
    out->offset = mul.offset * scale + add.offset;
    return true;
  }
  return false;
}

static bool wasm_index_param_affine(
    PolyUOp *u,
    PolyUOp *r0,
    PolyUOp *r1,
    PolyUOp *r2,
    int *param_out,
    WasmAffine *aff_out
) {
  if (!u || u->op != POLY_OP_INDEX || u->n_src < 2) return false;
  PolyUOp *base = u->src[0];
  while (base && (base->op == POLY_OP_CAST || base->op == POLY_OP_BITCAST ||
                  base->op == POLY_OP_RESHAPE || base->op == POLY_OP_EXPAND))
    base = base->src[0];
  if (!base || base->op != POLY_OP_PARAM || poly_uop_is_alu_param(base)) return false;
  if (!wasm_affine_expr(u->src[1], r0, r1, r2, aff_out)) return false;
  if (param_out) *param_out = (int)poly_program_buffer_slot(base);
  return true;
}

static bool wasm_affine_eq(WasmAffine a, int64_t c0, int64_t c1, int64_t c2) {
  return a.offset == 0 && a.c[0] == c0 && a.c[1] == c1 && a.c[2] == c2;
}

static bool wasm_const_f32(PolyUOp *u, float *out) {
  if (!u || u->op != POLY_OP_CONST) return false;
  if (u->arg.kind == POLY_ARG_FLOAT) {
    if (out) *out = (float)u->arg.f;
    return true;
  }
  if (u->arg.kind == POLY_ARG_INT) {
    if (out) *out = (float)u->arg.i;
    return true;
  }
  if (u->arg.kind == POLY_ARG_BIGINT) {
    if (out) *out = (float)poly_arg_integer_to_double(u->arg);
    return true;
  }
  return false;
}

static bool wasm_reduce_expr_supported(PolyUOp *u, const WasmReduceSpec *s);

static bool wasm_reduce_index_supported(PolyUOp *u, const WasmReduceSpec *s) {
  int param = -1;
  WasmAffine aff;
  if (!wasm_index_param_affine(u, s->red_range, s->out_range, NULL, &param, &aff)) return false;
  if (param == s->out_param) return false;
  PolyDType scalar = u->dtype;
  if (!poly_dtype_is_float(scalar) || scalar.bitsize != 32) return false;
  return aff.c[0] == 0 || aff.c[0] == 1;
}

static bool wasm_reduce_expr_supported(PolyUOp *u, const WasmReduceSpec *s) {
  if (!u || !s) return false;
  if (u->op == POLY_OP_INDEX) return wasm_reduce_index_supported(u, s);
  if (u->op == POLY_OP_CONST) {
    float v = 0.0f;
    return wasm_const_f32(u, &v);
  }
  if (u->op == POLY_OP_WHERE) {
    if (u->n_src < 3 || !wasm_is_compare_op(u->src[0]->op)) return false;
    return wasm_reduce_expr_supported(u->src[0], s) && wasm_reduce_expr_supported(u->src[1], s) &&
           wasm_reduce_expr_supported(u->src[2], s);
  }
  if (wasm_is_compare_op(u->op)) {
    return u->n_src >= 2 && wasm_reduce_expr_supported(u->src[0], s) &&
           wasm_reduce_expr_supported(u->src[1], s);
  }
  switch (u->op) {
  case POLY_OP_ADD:
  case POLY_OP_SUB:
  case POLY_OP_MUL:
  case POLY_OP_FDIV:
  case POLY_OP_MAX:
    return u->n_src >= 2 && wasm_reduce_expr_supported(u->src[0], s) &&
           wasm_reduce_expr_supported(u->src[1], s);
  case POLY_OP_NEG:
  case POLY_OP_SQRT:
    return u->n_src >= 1 && wasm_reduce_expr_supported(u->src[0], s);
  default:
    return false;
  }
}

static int wasm_reduce_expr_max_param(PolyUOp *u) {
  if (!u) return -1;
  int maxp = -1;
  if (u->op == POLY_OP_PARAM) maxp = (int)poly_program_buffer_slot(u);
  if (u->op == POLY_OP_INDEX && u->n_src >= 1) {
    PolyUOp *base = u->src[0];
    while (base && (base->op == POLY_OP_CAST || base->op == POLY_OP_BITCAST ||
                    base->op == POLY_OP_RESHAPE || base->op == POLY_OP_EXPAND))
      base = base->src[0];
    if (base && base->op == POLY_OP_PARAM) maxp = (int)poly_program_buffer_slot(base);
  }
  for (int i = 0; i < u->n_src; i++) {
    int child = wasm_reduce_expr_max_param(u->src[i]);
    if (child > maxp) maxp = child;
  }
  return maxp;
}

static bool wasm_reduce_expr_has_param(PolyUOp *u, int slot) {
  if (!u) return false;
  if (u->op == POLY_OP_PARAM && poly_program_buffer_slot(u) == slot) return true;
  for (int i = 0; i < u->n_src; i++)
    if (wasm_reduce_expr_has_param(u->src[i], slot)) return true;
  return false;
}

static bool wasm_match_row_reduce_root(PolyUOp *sink, WasmReduceSpec *spec) {
  if (!sink || sink->op != POLY_OP_SINK || sink->n_src != 1 || !spec) return false;
  PolyUOp *end = sink->src[0];
  if (!end || end->op != POLY_OP_END || end->n_src != 2) return false;
  PolyUOp *store = end->src[0];
  if (!store || store->op != POLY_OP_STORE || store->n_src < 2) return false;
  PolyUOp *reduce = store->src[1];
  if (!reduce || reduce->op != POLY_OP_REDUCE || reduce->arg.kind != POLY_ARG_REDUCE ||
      reduce->arg.reduce.op != POLY_OP_ADD || reduce->n_src != 2)
    return false;
  /* The specialized row-reduction module emits f32 locals/loads/stores.
   * Other tinygrad dtypes stay on the general typed WASM renderer. */
  if (!poly_dtype_eq(reduce->dtype, POLY_FLOAT32) ||
      !poly_dtype_eq(reduce->src[0]->dtype, POLY_FLOAT32))
    return false;

  PolyUOp *out_range = end->src[1];
  PolyUOp *red_range = reduce->src[1];
  int64_t m = 0, k = 0;
  if (!wasm_range_bound(out_range, &m) || !wasm_range_bound(red_range, &k)) return false;
  if (m <= 0 || k <= 0) return false;

  int out_param = -1;
  WasmAffine out_aff;
  if (!wasm_index_param_affine(store->src[0], red_range, out_range, NULL, &out_param, &out_aff))
    return false;
  if (!wasm_affine_eq(out_aff, 0, 1, 0)) return false;
  if (out_param < 0) return false;

  WasmReduceSpec tmp = {
      .m = (int)m,
      .k = (int)k,
      .out_param = out_param,
      .out_range = out_range,
      .red_range = red_range,
      .expr = reduce->src[0],
  };
  if (!wasm_reduce_expr_supported(tmp.expr, &tmp)) return false;
  /* The direct row-reduction emitter addresses PARAM locals by slot. Sparse
   * globals use the general renderer, whose ABI ranking matches Tinygrad's
   * compact ProgramInfo.globals invocation. */
  int max_param = wasm_reduce_expr_max_param(tmp.expr);
  if (tmp.out_param > max_param) max_param = tmp.out_param;
  for (int slot = 0; slot <= max_param; slot++)
    if (slot != tmp.out_param && !wasm_reduce_expr_has_param(tmp.expr, slot)) return false;
  *spec = tmp;
  return true;
}

static bool wasm_match_matmul_root(PolyUOp *sink, WasmMatmulSpec *spec) {
  if (!sink || sink->op != POLY_OP_SINK || sink->n_src != 1 || !spec) return false;
  PolyUOp *end = sink->src[0];
  if (!end || end->op != POLY_OP_END || end->n_src < 2) return false;
  PolyUOp *store = end->src[0];
  if (!store || store->op != POLY_OP_STORE || store->n_src < 2) return false;
  PolyUOp *reduce = store->src[1];
  if (!reduce || reduce->op != POLY_OP_REDUCE || reduce->arg.kind != POLY_ARG_REDUCE ||
      reduce->arg.reduce.op != POLY_OP_ADD || reduce->n_src != 2)
    return false;
  PolyUOp *mul = reduce->src[0];
  if (!mul || mul->op != POLY_OP_MUL || mul->n_src != 2) return false;
  /* wasm_emit_matmul_* is deliberately f32 SIMD. Keep integer and other
   * tinygrad matmul graphs on the general dtype-aware renderer. */
  if (!poly_dtype_eq(reduce->dtype, POLY_FLOAT32) || !poly_dtype_eq(mul->dtype, POLY_FLOAT32) ||
      !poly_dtype_eq(mul->src[0]->dtype, POLY_FLOAT32) ||
      !poly_dtype_eq(mul->src[1]->dtype, POLY_FLOAT32))
    return false;

  PolyUOp *r_k = reduce->src[1];
  if (end->n_src == 2) {
    PolyUOp *r_col = end->src[1];
    int64_t n = 0, k = 0;
    if (!wasm_range_bound(r_col, &n) || !wasm_range_bound(r_k, &k)) return false;
    if (n <= 0 || k <= 0 || n % 4 != 0) return false;

    int out_param = -1, a_param = -1, b_param = -1;
    WasmAffine out_aff, a_aff, b_aff;
    if (!wasm_index_param_affine(store->src[0], r_k, r_col, NULL, &out_param, &out_aff))
      return false;
    if (!wasm_index_param_affine(mul->src[0], r_k, r_col, NULL, &a_param, &a_aff) ||
        !wasm_index_param_affine(mul->src[1], r_k, r_col, NULL, &b_param, &b_aff))
      return false;

    if (!wasm_affine_eq(out_aff, 0, 1, 0)) return false;
    if (!wasm_affine_eq(a_aff, 1, 0, 0)) return false;
    if (!wasm_affine_eq(b_aff, 1, k, 0)) return false;
    if (out_param < 0 || a_param < 0 || b_param < 0 || out_param == a_param ||
        out_param == b_param || a_param == b_param)
      return false;

    *spec = (WasmMatmulSpec){
        .m = 1,
        .n = (int)n,
        .k = (int)k,
        .out_param = out_param,
        .a_param = a_param,
        .b_param = b_param,
        .kind = WASM_MATMUL_ABT,
    };
    return true;
  }

  PolyUOp *r_row = end->src[1];
  PolyUOp *r_col = end->src[2];
  int64_t m = 0, n = 0, k = 0;
  if (!wasm_range_bound(r_row, &m) || !wasm_range_bound(r_col, &n) || !wasm_range_bound(r_k, &k))
    return false;
  if (m <= 0 || n <= 0 || k <= 0) return false;

  int out_param = -1, a_param = -1, b_param = -1;
  WasmAffine out_aff, a_aff, b_aff;
  if (!wasm_index_param_affine(store->src[0], r_k, r_row, r_col, &out_param, &out_aff))
    return false;
  if (!wasm_index_param_affine(mul->src[0], r_k, r_row, r_col, &a_param, &a_aff) ||
      !wasm_index_param_affine(mul->src[1], r_k, r_row, r_col, &b_param, &b_aff))
    return false;

  if (!wasm_affine_eq(out_aff, 0, n, 1)) return false;
  if (!wasm_affine_eq(a_aff, 1, k, 0)) return false;

  WasmMatmulKind kind;
  if (wasm_affine_eq(b_aff, n, 0, 1)) {
    kind = WASM_MATMUL_AB;
  } else if (wasm_affine_eq(b_aff, 1, 0, k)) {
    kind = WASM_MATMUL_ABT;
  } else {
    return false;
  }
  if (kind == WASM_MATMUL_AB) {
    /* A@B has scalar epilogues for row and column tails. */
  } else {
    if (!((m == 1 || m % 4 == 0) && n % 4 == 0)) return false;
  }

  if (out_param < 0 || a_param < 0 || b_param < 0 || out_param == a_param || out_param == b_param ||
      a_param == b_param)
    return false;

  *spec = (WasmMatmulSpec){
      .m = (int)m,
      .n = (int)n,
      .k = (int)k,
      .out_param = out_param,
      .a_param = a_param,
      .b_param = b_param,
      .kind = kind,
  };
  return true;
}

bool poly_wasm_can_render_matmul(PolyUOp *sink) {
  WasmMatmulSpec spec;
  return wasm_match_matmul_root(sink, &spec);
}

bool poly_wasm_can_render_reduce(PolyUOp *sink) {
  WasmReduceSpec spec;
  return wasm_match_row_reduce_root(sink, &spec);
}

static void wasm_emit_local_get(WasmBuf *body, int local) {
  if (local < 0 && poly_debug_at_least(4))
    fprintf(stderr, "[polygrad:wasm] matmul emitter local.get(%d)\n", local);
  wb_byte(body, WASM_OP_LOCAL_GET);
  wb_uleb128(body, local);
}

static void wasm_emit_local_set(WasmBuf *body, int local) {
  if (local < 0 && poly_debug_at_least(4))
    fprintf(stderr, "[polygrad:wasm] matmul emitter local.set(%d)\n", local);
  wb_byte(body, WASM_OP_LOCAL_SET);
  wb_uleb128(body, local);
}

static void wasm_emit_i32_const(WasmBuf *body, int32_t v) {
  wb_byte(body, WASM_OP_I32_CONST);
  wb_sleb128(body, v);
}

static void wasm_emit_i32_add(WasmBuf *body) {
  wb_byte(body, WASM_OP_I32_ADD);
}
static void wasm_emit_i32_mul(WasmBuf *body) {
  wb_byte(body, WASM_OP_I32_MUL);
}
static void wasm_emit_i32_ge_u(WasmBuf *body) {
  wb_byte(body, WASM_OP_I32_GE_U);
}

static void wasm_emit_loop_header(WasmBuf *body) {
  wb_byte(body, WASM_OP_BLOCK);
  wb_byte(body, WASM_BLOCKTYPE_VOID);
  wb_byte(body, WASM_OP_LOOP);
  wb_byte(body, WASM_BLOCKTYPE_VOID);
}

static void wasm_emit_loop_footer(WasmBuf *body) {
  wb_byte(body, WASM_OP_BR);
  wb_uleb128(body, 0);
  wb_byte(body, WASM_OP_END);
  wb_byte(body, WASM_OP_END);
}

static void wasm_emit_break_if_ge_const(WasmBuf *body, int local, int bound) {
  wasm_emit_local_get(body, local);
  wasm_emit_i32_const(body, bound);
  wasm_emit_i32_ge_u(body);
  wb_byte(body, WASM_OP_BR_IF);
  wb_uleb128(body, 1);
}

static void wasm_emit_break_if_ge_local_plus_const(
    WasmBuf *body,
    int local,
    int base_local,
    int bound_offset
) {
  wasm_emit_local_get(body, local);
  wasm_emit_local_get(body, base_local);
  wasm_emit_i32_const(body, bound_offset);
  wasm_emit_i32_add(body);
  wasm_emit_i32_ge_u(body);
  wb_byte(body, WASM_OP_BR_IF);
  wb_uleb128(body, 1);
}

static void wasm_emit_break_if_ge_local(WasmBuf *body, int local, int bound_local) {
  wasm_emit_local_get(body, local);
  wasm_emit_local_get(body, bound_local);
  wasm_emit_i32_ge_u(body);
  wb_byte(body, WASM_OP_BR_IF);
  wb_uleb128(body, 1);
}

static void wasm_emit_break_if_local_plus_ge_const(
    WasmBuf *body,
    int lhs_local,
    int rhs_local,
    int bound
) {
  wasm_emit_local_get(body, lhs_local);
  wasm_emit_local_get(body, rhs_local);
  wasm_emit_i32_add(body);
  wasm_emit_i32_const(body, bound);
  wasm_emit_i32_ge_u(body);
  wb_byte(body, WASM_OP_BR_IF);
  wb_uleb128(body, 1);
}

static void wasm_emit_inc_const(WasmBuf *body, int local, int inc) {
  wasm_emit_local_get(body, local);
  wasm_emit_i32_const(body, inc);
  wasm_emit_i32_add(body);
  wasm_emit_local_set(body, local);
}

static void wasm_emit_param_element_addr(
    WasmBuf *body,
    int param,
    int idx_a_local,
    int stride_a,
    int idx_b_local,
    int stride_b,
    int idx_c_local,
    int stride_c
) {
  wasm_emit_local_get(body, param);
  bool any = false;
  if (idx_a_local >= 0 && stride_a != 0) {
    wasm_emit_local_get(body, idx_a_local);
    if (stride_a != 1) {
      wasm_emit_i32_const(body, stride_a);
      wasm_emit_i32_mul(body);
    }
    any = true;
  }
  if (idx_b_local >= 0 && stride_b != 0) {
    wasm_emit_local_get(body, idx_b_local);
    if (stride_b != 1) {
      wasm_emit_i32_const(body, stride_b);
      wasm_emit_i32_mul(body);
    }
    if (any) wasm_emit_i32_add(body);
    any = true;
  }
  if (idx_c_local >= 0 && stride_c != 0) {
    wasm_emit_local_get(body, idx_c_local);
    if (stride_c != 1) {
      wasm_emit_i32_const(body, stride_c);
      wasm_emit_i32_mul(body);
    }
    if (any) wasm_emit_i32_add(body);
    any = true;
  }
  if (!any) wasm_emit_i32_const(body, 0);
  wasm_emit_i32_const(body, 4);
  wasm_emit_i32_mul(body);
  wasm_emit_i32_add(body);
}

static void wasm_emit_f32_store_addr(WasmBuf *body) {
  emit_scalar_store_opcode(body, POLY_FLOAT32);
}

static void wasm_emit_f32_load_addr(WasmBuf *body) {
  emit_scalar_load_opcode(body, POLY_FLOAT32);
}

static void wasm_emit_v128_load_addr_align(WasmBuf *body, int align_log2) {
  wb_byte(body, WASM_SIMD_PREFIX);
  wb_uleb128(body, WASM_SIMD_V128_LOAD);
  wb_uleb128(body, align_log2);
  wb_uleb128(body, 0);
}

static void wasm_emit_v128_load32_splat_addr(WasmBuf *body) {
  wb_byte(body, WASM_SIMD_PREFIX);
  wb_uleb128(body, WASM_SIMD_V128_LOAD32_SPLAT);
  wb_uleb128(body, 2);
  wb_uleb128(body, 0);
}

static void wasm_emit_v128_store_addr_align(WasmBuf *body, int align_log2) {
  wb_byte(body, WASM_SIMD_PREFIX);
  wb_uleb128(body, WASM_SIMD_V128_STORE);
  wb_uleb128(body, align_log2);
  wb_uleb128(body, 0);
}

static void wasm_emit_f32x4_accumulate(
    WasmBuf *body,
    int acc_local,
    int lhs_local,
    int rhs_local,
    bool use_relaxed_madd
) {
  if (use_relaxed_madd) {
    wasm_emit_local_get(body, lhs_local);
    wasm_emit_local_get(body, rhs_local);
    wasm_emit_local_get(body, acc_local);
    wb_byte(body, WASM_SIMD_PREFIX);
    wb_uleb128(body, WASM_SIMD_F32X4_RELAXED_MADD);
    wasm_emit_local_set(body, acc_local);
    return;
  }

  wasm_emit_local_get(body, acc_local);
  wasm_emit_local_get(body, lhs_local);
  wasm_emit_local_get(body, rhs_local);
  wb_byte(body, WASM_SIMD_PREFIX);
  wb_uleb128(body, WASM_SIMD_F32X4_MUL);
  wb_byte(body, WASM_SIMD_PREFIX);
  wb_uleb128(body, WASM_SIMD_F32X4_ADD);
  wasm_emit_local_set(body, acc_local);
}

static void wasm_emit_f32x4_horizontal_sum(WasmBuf *body, int vec_local, int tmp_local) {
  static const uint8_t swap_halves[16] = {
      8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7,
  };
  static const uint8_t swap_pairs[16] = {
      4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11,
  };

  wasm_emit_local_get(body, vec_local);
  wasm_emit_local_get(body, vec_local);
  wasm_emit_local_get(body, vec_local);
  emit_v128_shuffle(body, swap_halves);
  wb_byte(body, WASM_SIMD_PREFIX);
  wb_uleb128(body, WASM_SIMD_F32X4_ADD);
  wasm_emit_local_set(body, tmp_local);

  wasm_emit_local_get(body, tmp_local);
  wasm_emit_local_get(body, tmp_local);
  wasm_emit_local_get(body, tmp_local);
  emit_v128_shuffle(body, swap_pairs);
  wb_byte(body, WASM_SIMD_PREFIX);
  wb_uleb128(body, WASM_SIMD_F32X4_ADD);
  emit_v128_extract_lane(body, POLY_FLOAT32, 0);
}

static void wasm_emit_f32x4_splat_const(WasmBuf *body, float v) {
  wb_byte(body, WASM_OP_F32_CONST);
  wb_f32(body, v);
  emit_v128_splat(body, POLY_FLOAT32);
}

static void wasm_emit_reduce_index_v128(
    WasmBuf *body,
    PolyUOp *idx,
    const WasmReduceSpec *s,
    int row_local,
    int kk_local
) {
  int param = -1;
  WasmAffine aff;
  if (!wasm_index_param_affine(idx, s->red_range, s->out_range, NULL, &param, &aff)) {
    emit_v128_zero(body);
    return;
  }

  wasm_emit_param_element_addr(
      body, param, row_local, (int)aff.c[1], kk_local, (int)aff.c[0], -1, 0
  );
  if (aff.offset != 0) {
    wasm_emit_i32_const(body, (int32_t)(aff.offset * 4));
    wasm_emit_i32_add(body);
  }

  if (aff.c[0] == 0) {
    emit_scalar_load_opcode(body, POLY_FLOAT32);
    emit_v128_splat(body, POLY_FLOAT32);
  } else {
    wasm_emit_v128_load_addr_align(body, 2);
  }
}

static bool wasm_emit_reduce_index_f32(
    WasmBuf *body,
    PolyUOp *idx,
    const WasmReduceSpec *s,
    int row_local,
    int kk_local
) {
  int param = -1;
  WasmAffine aff;
  if (!wasm_index_param_affine(idx, s->red_range, s->out_range, NULL, &param, &aff)) return false;

  wasm_emit_param_element_addr(
      body, param, row_local, (int)aff.c[1], kk_local, (int)aff.c[0], -1, 0
  );
  if (aff.offset != 0) {
    wasm_emit_i32_const(body, (int32_t)(aff.offset * 4));
    wasm_emit_i32_add(body);
  }
  emit_scalar_load_opcode(body, POLY_FLOAT32);
  return true;
}

static bool wasm_emit_reduce_expr_v128(
    WasmBuf *body,
    PolyUOp *u,
    const WasmReduceSpec *s,
    int row_local,
    int kk_local
) {
  if (!u) return false;
  if (u->op == POLY_OP_INDEX) {
    if (!wasm_reduce_index_supported(u, s)) return false;
    wasm_emit_reduce_index_v128(body, u, s, row_local, kk_local);
    return true;
  }
  if (u->op == POLY_OP_CONST) {
    float v = 0.0f;
    if (!wasm_const_f32(u, &v)) return false;
    wasm_emit_f32x4_splat_const(body, v);
    return true;
  }

  if (u->op == POLY_OP_WHERE && u->n_src >= 3) {
    if (!wasm_is_compare_op(u->src[0]->op)) return false;
    if (!wasm_emit_reduce_expr_v128(body, u->src[1], s, row_local, kk_local)) return false;
    if (!wasm_emit_reduce_expr_v128(body, u->src[2], s, row_local, kk_local)) return false;
    if (!wasm_emit_reduce_expr_v128(body, u->src[0], s, row_local, kk_local)) return false;
    wb_byte(body, WASM_SIMD_PREFIX);
    wb_uleb128(body, WASM_SIMD_V128_BITSELECT);
    return true;
  }

  if (u->op == POLY_OP_NEG || u->op == POLY_OP_SQRT) {
    if (u->n_src < 1) return false;
    if (!wasm_emit_reduce_expr_v128(body, u->src[0], s, row_local, kk_local)) return false;
    emit_alu_simd_f32x4(body, u->op);
    return true;
  }

  if (u->n_src < 2) return false;
  switch (u->op) {
  case POLY_OP_ADD:
  case POLY_OP_SUB:
  case POLY_OP_MUL:
  case POLY_OP_FDIV:
  case POLY_OP_MAX:
  case POLY_OP_CMPLT:
  case POLY_OP_CMPEQ:
  case POLY_OP_CMPNE:
    if (!wasm_emit_reduce_expr_v128(body, u->src[0], s, row_local, kk_local)) return false;
    if (!wasm_emit_reduce_expr_v128(body, u->src[1], s, row_local, kk_local)) return false;
    emit_alu_simd_f32x4(body, u->op);
    return true;
  default:
    return false;
  }
}

static bool wasm_emit_reduce_expr_f32(
    WasmBuf *body,
    PolyUOp *u,
    const WasmReduceSpec *s,
    int row_local,
    int kk_local
) {
  if (!u) return false;
  MathImports math = {0};

  if (u->op == POLY_OP_INDEX) {
    if (!wasm_reduce_index_supported(u, s)) return false;
    return wasm_emit_reduce_index_f32(body, u, s, row_local, kk_local);
  }
  if (u->op == POLY_OP_CONST) {
    float v = 0.0f;
    if (!wasm_const_f32(u, &v)) return false;
    wb_byte(body, WASM_OP_F32_CONST);
    wb_f32(body, v);
    return true;
  }

  if (u->op == POLY_OP_WHERE && u->n_src >= 3) {
    if (!wasm_is_compare_op(u->src[0]->op)) return false;
    if (!wasm_emit_reduce_expr_f32(body, u->src[1], s, row_local, kk_local)) return false;
    if (!wasm_emit_reduce_expr_f32(body, u->src[2], s, row_local, kk_local)) return false;
    if (!wasm_emit_reduce_expr_f32(body, u->src[0], s, row_local, kk_local)) return false;
    emit_alu_scalar(body, POLY_OP_WHERE, POLY_FLOAT32, &math, 0);
    return true;
  }

  if (u->op == POLY_OP_NEG || u->op == POLY_OP_SQRT) {
    if (u->n_src < 1) return false;
    if (!wasm_emit_reduce_expr_f32(body, u->src[0], s, row_local, kk_local)) return false;
    emit_alu_scalar(body, u->op, POLY_FLOAT32, &math, 0);
    return true;
  }

  if (u->n_src < 2) return false;
  switch (u->op) {
  case POLY_OP_ADD:
  case POLY_OP_SUB:
  case POLY_OP_MUL:
  case POLY_OP_FDIV:
  case POLY_OP_MAX:
    if (!wasm_emit_reduce_expr_f32(body, u->src[0], s, row_local, kk_local)) return false;
    if (!wasm_emit_reduce_expr_f32(body, u->src[1], s, row_local, kk_local)) return false;
    emit_alu_scalar(body, u->op, POLY_FLOAT32, &math, 0);
    return true;
  case POLY_OP_CMPLT:
  case POLY_OP_CMPEQ:
  case POLY_OP_CMPNE:
    if (!wasm_emit_reduce_expr_f32(body, u->src[0], s, row_local, kk_local)) return false;
    if (!wasm_emit_reduce_expr_f32(body, u->src[1], s, row_local, kk_local)) return false;
    emit_alu_scalar(body, u->op, POLY_FLOAT32, &math, 0);
    return true;
  default:
    return false;
  }
}

static void wasm_emit_matmul_ab_body(
    WasmBuf *body,
    const WasmMatmulSpec *s,
    int row,
    int col,
    int kk,
    int row_tile,
    int col_tile,
    int row_off,
    int k_tile,
    int tile_end,
    int b_addr,
    int a_addr,
    int out_addr,
    int acc0,
    int bvec0,
    int avec,
    int tail_acc0,
    bool use_relaxed_madd
) {
  const int vector_rows = (s->m / 4) * 4;
  const int tile_rows = vector_rows < 128 ? vector_rows : 128;
  const int vector_cols = (s->n / 16) * 16;
  const int tile_cols = vector_cols < 128 ? vector_cols : 128;
  const int tile_k = (s->k >= 2048 && s->k % 64 == 0) ? 64 : s->k;

  wasm_emit_i32_const(body, 0);
  wasm_emit_local_set(body, row_tile);
  wasm_emit_loop_header(body);
  wasm_emit_break_if_ge_const(body, row_tile, vector_rows);

  wasm_emit_i32_const(body, 0);
  wasm_emit_local_set(body, col_tile);
  wasm_emit_loop_header(body);
  wasm_emit_break_if_ge_const(body, col_tile, vector_cols);

  wasm_emit_i32_const(body, 0);
  wasm_emit_local_set(body, k_tile);
  wasm_emit_loop_header(body);
  wasm_emit_break_if_ge_const(body, k_tile, s->k);

  wasm_emit_local_get(body, k_tile);
  wasm_emit_i32_const(body, tile_k);
  wasm_emit_i32_add(body);
  wasm_emit_local_set(body, tile_end);
  wasm_emit_local_get(body, tile_end);
  wasm_emit_i32_const(body, s->k);
  wasm_emit_i32_ge_u(body);
  wb_byte(body, WASM_OP_IF);
  wb_byte(body, WASM_BLOCKTYPE_VOID);
  wasm_emit_i32_const(body, s->k);
  wasm_emit_local_set(body, tile_end);
  wb_byte(body, WASM_OP_END);

  wasm_emit_i32_const(body, 0);
  wasm_emit_local_set(body, row_off);
  wasm_emit_loop_header(body);
  wasm_emit_break_if_ge_const(body, row_off, tile_rows);
  wasm_emit_break_if_local_plus_ge_const(body, row_tile, row_off, s->m);

  wasm_emit_local_get(body, row_tile);
  wasm_emit_local_get(body, row_off);
  wasm_emit_i32_add(body);
  wasm_emit_local_set(body, row);

  wasm_emit_local_get(body, col_tile);
  wasm_emit_local_set(body, col);
  wasm_emit_loop_header(body);
  wasm_emit_break_if_ge_local_plus_const(body, col, col_tile, tile_cols);
  wasm_emit_break_if_ge_const(body, col, vector_cols);

  wasm_emit_param_element_addr(body, s->out_param, row, s->n, col, 1, -1, 0);
  wasm_emit_local_set(body, out_addr);
  for (int r = 1; r < 4; r++) {
    wasm_emit_local_get(body, out_addr);
    wasm_emit_i32_const(body, r * s->n * 4);
    wasm_emit_i32_add(body);
    wasm_emit_local_set(body, out_addr + r);
  }

  wasm_emit_local_get(body, k_tile);
  wb_byte(body, WASM_OP_I32_EQZ);
  wb_byte(body, WASM_OP_IF);
  wb_byte(body, WASM_BLOCKTYPE_VOID);
  for (int r = 0; r < 4; r++) {
    for (int v = 0; v < 4; v++) {
      emit_v128_zero(body);
      wasm_emit_local_set(body, acc0 + r * 4 + v);
    }
  }
  wb_byte(body, WASM_OP_ELSE);
  for (int r = 0; r < 4; r++) {
    for (int v = 0; v < 4; v++) {
      wasm_emit_local_get(body, out_addr + r);
      if (v != 0) {
        wasm_emit_i32_const(body, v * 4 * 4);
        wasm_emit_i32_add(body);
      }
      wasm_emit_v128_load_addr_align(body, 4);
      wasm_emit_local_set(body, acc0 + r * 4 + v);
    }
  }
  wb_byte(body, WASM_OP_END);

  wasm_emit_local_get(body, k_tile);
  wasm_emit_local_set(body, kk);

  wasm_emit_param_element_addr(body, s->b_param, k_tile, s->n, col, 1, -1, 0);
  wasm_emit_local_set(body, b_addr);
  for (int v = 1; v < 4; v++) {
    wasm_emit_local_get(body, b_addr);
    wasm_emit_i32_const(body, v * 4 * 4);
    wasm_emit_i32_add(body);
    wasm_emit_local_set(body, b_addr + v);
  }

  wasm_emit_param_element_addr(body, s->a_param, row, s->k, k_tile, 1, -1, 0);
  wasm_emit_local_set(body, a_addr);
  for (int r = 1; r < 4; r++) {
    wasm_emit_local_get(body, a_addr);
    wasm_emit_i32_const(body, r * s->k * 4);
    wasm_emit_i32_add(body);
    wasm_emit_local_set(body, a_addr + r);
  }

  wasm_emit_loop_header(body);
  wasm_emit_break_if_ge_local(body, kk, tile_end);

  for (int v = 0; v < 4; v++) {
    wasm_emit_local_get(body, b_addr + v);
    wasm_emit_v128_load_addr_align(body, 4);
    wasm_emit_local_set(body, bvec0 + v);
    wasm_emit_inc_const(body, b_addr + v, s->n * 4);
  }

  for (int r = 0; r < 4; r++) {
    wasm_emit_local_get(body, a_addr + r);
    wasm_emit_v128_load32_splat_addr(body);
    wasm_emit_local_set(body, avec);
    wasm_emit_inc_const(body, a_addr + r, 4);
    for (int v = 0; v < 4; v++) {
      wasm_emit_f32x4_accumulate(body, acc0 + r * 4 + v, avec, bvec0 + v, use_relaxed_madd);
    }
  }

  wasm_emit_inc_const(body, kk, 1);
  wasm_emit_loop_footer(body);

  for (int r = 0; r < 4; r++) {
    for (int v = 0; v < 4; v++) {
      wasm_emit_local_get(body, out_addr + r);
      if (v != 0) {
        wasm_emit_i32_const(body, v * 4 * 4);
        wasm_emit_i32_add(body);
      }
      wasm_emit_local_get(body, acc0 + r * 4 + v);
      wasm_emit_v128_store_addr_align(body, 4);
    }
  }

  wasm_emit_inc_const(body, col, 16);
  wasm_emit_loop_footer(body);

  wasm_emit_inc_const(body, row_off, 4);
  wasm_emit_loop_footer(body);

  wasm_emit_inc_const(body, k_tile, tile_k);
  wasm_emit_loop_footer(body);

  wasm_emit_inc_const(body, col_tile, tile_cols);
  wasm_emit_loop_footer(body);

  wasm_emit_inc_const(body, row_tile, tile_rows);
  wasm_emit_loop_footer(body);

  if (vector_cols != s->n) {
    wasm_emit_i32_const(body, 0);
    wasm_emit_local_set(body, row_tile);
    wasm_emit_loop_header(body);
    wasm_emit_break_if_ge_const(body, row_tile, vector_rows);

    wasm_emit_i32_const(body, 0);
    wasm_emit_local_set(body, row_off);
    wasm_emit_loop_header(body);
    wasm_emit_break_if_ge_const(body, row_off, tile_rows);
    wasm_emit_break_if_local_plus_ge_const(body, row_tile, row_off, s->m);

    wasm_emit_local_get(body, row_tile);
    wasm_emit_local_get(body, row_off);
    wasm_emit_i32_add(body);
    wasm_emit_local_set(body, row);

    wasm_emit_i32_const(body, vector_cols);
    wasm_emit_local_set(body, col);
    wasm_emit_loop_header(body);
    wasm_emit_break_if_ge_const(body, col, s->n);

    for (int r = 0; r < 4; r++) {
      wb_byte(body, WASM_OP_F32_CONST);
      wb_f32(body, 0.0f);
      wasm_emit_local_set(body, tail_acc0 + r);
    }

    wasm_emit_i32_const(body, 0);
    wasm_emit_local_set(body, kk);
    wasm_emit_loop_header(body);
    wasm_emit_break_if_ge_const(body, kk, s->k);

    for (int r = 0; r < 4; r++) {
      wasm_emit_local_get(body, tail_acc0 + r);

      wasm_emit_param_element_addr(body, s->a_param, row, s->k, kk, 1, -1, 0);
      if (r != 0) {
        wasm_emit_i32_const(body, r * s->k * 4);
        wasm_emit_i32_add(body);
      }
      wasm_emit_f32_load_addr(body);

      wasm_emit_param_element_addr(body, s->b_param, kk, s->n, col, 1, -1, 0);
      wasm_emit_f32_load_addr(body);

      wb_byte(body, WASM_OP_F32_MUL);
      wb_byte(body, WASM_OP_F32_ADD);
      wasm_emit_local_set(body, tail_acc0 + r);
    }

    wasm_emit_inc_const(body, kk, 1);
    wasm_emit_loop_footer(body);

    for (int r = 0; r < 4; r++) {
      wasm_emit_param_element_addr(body, s->out_param, row, s->n, col, 1, -1, 0);
      if (r != 0) {
        wasm_emit_i32_const(body, r * s->n * 4);
        wasm_emit_i32_add(body);
      }
      wasm_emit_local_get(body, tail_acc0 + r);
      wasm_emit_f32_store_addr(body);
    }

    wasm_emit_inc_const(body, col, 1);
    wasm_emit_loop_footer(body);

    wasm_emit_inc_const(body, row_off, 4);
    wasm_emit_loop_footer(body);

    wasm_emit_inc_const(body, row_tile, tile_rows);
    wasm_emit_loop_footer(body);
  }

  if (vector_rows != s->m) {
    wasm_emit_i32_const(body, vector_rows);
    wasm_emit_local_set(body, row);
    wasm_emit_loop_header(body);
    wasm_emit_break_if_ge_const(body, row, s->m);

    wasm_emit_i32_const(body, 0);
    wasm_emit_local_set(body, col);
    wasm_emit_loop_header(body);
    wasm_emit_break_if_ge_const(body, col, s->n);

    wb_byte(body, WASM_OP_F32_CONST);
    wb_f32(body, 0.0f);
    wasm_emit_local_set(body, tail_acc0);

    wasm_emit_i32_const(body, 0);
    wasm_emit_local_set(body, kk);
    wasm_emit_loop_header(body);
    wasm_emit_break_if_ge_const(body, kk, s->k);

    wasm_emit_local_get(body, tail_acc0);
    wasm_emit_param_element_addr(body, s->a_param, row, s->k, kk, 1, -1, 0);
    wasm_emit_f32_load_addr(body);
    wasm_emit_param_element_addr(body, s->b_param, kk, s->n, col, 1, -1, 0);
    wasm_emit_f32_load_addr(body);
    wb_byte(body, WASM_OP_F32_MUL);
    wb_byte(body, WASM_OP_F32_ADD);
    wasm_emit_local_set(body, tail_acc0);

    wasm_emit_inc_const(body, kk, 1);
    wasm_emit_loop_footer(body);

    wasm_emit_param_element_addr(body, s->out_param, row, s->n, col, 1, -1, 0);
    wasm_emit_local_get(body, tail_acc0);
    wasm_emit_f32_store_addr(body);

    wasm_emit_inc_const(body, col, 1);
    wasm_emit_loop_footer(body);

    wasm_emit_inc_const(body, row, 1);
    wasm_emit_loop_footer(body);
  }
}

static void wasm_emit_matmul_abt_body(
    WasmBuf *body,
    const WasmMatmulSpec *s,
    int row,
    int col,
    int kk,
    int row_tile,
    int col_tile,
    int row_off,
    int acc0,
    int sum,
    int bvec0,
    int avec,
    int b_addr,
    int a_addr,
    bool use_relaxed_madd
) {
  const int tile_rows = s->m < 128 ? s->m : 128;
  const int tile_cols = s->n < 128 ? s->n : 128;
  const int k_unroll = (s->k % 16 == 0) ? 4 : 1;
  const int vector_k = (s->k / 4) * 4;

  wasm_emit_i32_const(body, 0);
  wasm_emit_local_set(body, row_tile);
  wasm_emit_loop_header(body);
  wasm_emit_break_if_ge_const(body, row_tile, s->m);

  wasm_emit_i32_const(body, 0);
  wasm_emit_local_set(body, col_tile);
  wasm_emit_loop_header(body);
  wasm_emit_break_if_ge_const(body, col_tile, s->n);

  wasm_emit_i32_const(body, 0);
  wasm_emit_local_set(body, row_off);
  wasm_emit_loop_header(body);
  wasm_emit_break_if_ge_const(body, row_off, tile_rows);
  wasm_emit_break_if_local_plus_ge_const(body, row_tile, row_off, s->m);

  wasm_emit_local_get(body, row_tile);
  wasm_emit_local_get(body, row_off);
  wasm_emit_i32_add(body);
  wasm_emit_local_set(body, row);

  wasm_emit_local_get(body, col_tile);
  wasm_emit_local_set(body, col);
  wasm_emit_loop_header(body);
  wasm_emit_break_if_ge_local_plus_const(body, col, col_tile, tile_cols);

  for (int r = 0; r < 4; r++) {
    for (int c = 0; c < 4; c++) {
      emit_v128_zero(body);
      wasm_emit_local_set(body, acc0 + r * 4 + c);
    }
  }
  wasm_emit_i32_const(body, 0);
  wasm_emit_local_set(body, kk);
  wasm_emit_param_element_addr(body, s->a_param, row, s->k, -1, 0, -1, 0);
  wasm_emit_local_set(body, a_addr);
  for (int r = 1; r < 4; r++) {
    wasm_emit_local_get(body, a_addr);
    wasm_emit_i32_const(body, r * s->k * 4);
    wasm_emit_i32_add(body);
    wasm_emit_local_set(body, a_addr + r);
  }
  wasm_emit_param_element_addr(body, s->b_param, col, s->k, -1, 0, -1, 0);
  wasm_emit_local_set(body, b_addr);
  for (int c = 1; c < 4; c++) {
    wasm_emit_local_get(body, b_addr);
    wasm_emit_i32_const(body, c * s->k * 4);
    wasm_emit_i32_add(body);
    wasm_emit_local_set(body, b_addr + c);
  }
  wasm_emit_loop_header(body);
  wasm_emit_break_if_ge_const(body, kk, vector_k);

  for (int u = 0; u < k_unroll; u++) {
    for (int c = 0; c < 4; c++) {
      wasm_emit_local_get(body, b_addr + c);
      wasm_emit_v128_load_addr_align(body, 4);
      wasm_emit_local_set(body, bvec0 + c);
      wasm_emit_inc_const(body, b_addr + c, 16);
    }

    for (int r = 0; r < 4; r++) {
      wasm_emit_local_get(body, a_addr + r);
      wasm_emit_v128_load_addr_align(body, 4);
      wasm_emit_local_set(body, avec);
      wasm_emit_inc_const(body, a_addr + r, 16);
      for (int c = 0; c < 4; c++) {
        wasm_emit_f32x4_accumulate(body, acc0 + r * 4 + c, avec, bvec0 + c, use_relaxed_madd);
      }
    }
  }

  wasm_emit_inc_const(body, kk, 4 * k_unroll);
  wasm_emit_loop_footer(body);

  for (int r = 0; r < 4; r++) {
    for (int c = 0; c < 4; c++) {
      int acc = acc0 + r * 4 + c;
      wasm_emit_f32x4_horizontal_sum(body, acc, avec);
      wasm_emit_local_set(body, sum);

      if (vector_k != s->k) {
        wasm_emit_i32_const(body, vector_k);
        wasm_emit_local_set(body, kk);
        wasm_emit_loop_header(body);
        wasm_emit_break_if_ge_const(body, kk, s->k);

        wasm_emit_local_get(body, sum);
        wasm_emit_param_element_addr(body, s->a_param, row, s->k, kk, 1, -1, 0);
        if (r != 0) {
          wasm_emit_i32_const(body, r * s->k * 4);
          wasm_emit_i32_add(body);
        }
        wasm_emit_f32_load_addr(body);

        wasm_emit_param_element_addr(body, s->b_param, col, s->k, kk, 1, -1, 0);
        if (c != 0) {
          wasm_emit_i32_const(body, c * s->k * 4);
          wasm_emit_i32_add(body);
        }
        wasm_emit_f32_load_addr(body);

        wb_byte(body, WASM_OP_F32_MUL);
        wb_byte(body, WASM_OP_F32_ADD);
        wasm_emit_local_set(body, sum);

        wasm_emit_inc_const(body, kk, 1);
        wasm_emit_loop_footer(body);
      }

      wasm_emit_param_element_addr(body, s->out_param, row, s->n, col, 1, -1, 0);
      if (r != 0 || c != 0) {
        wasm_emit_i32_const(body, (r * s->n + c) * 4);
        wasm_emit_i32_add(body);
      }
      wasm_emit_local_get(body, sum);
      wasm_emit_f32_store_addr(body);
    }
  }

  wasm_emit_inc_const(body, col, 4);
  wasm_emit_loop_footer(body);

  wasm_emit_inc_const(body, row_off, 4);
  wasm_emit_loop_footer(body);

  wasm_emit_inc_const(body, col_tile, tile_cols);
  wasm_emit_loop_footer(body);

  wasm_emit_inc_const(body, row_tile, tile_rows);
  wasm_emit_loop_footer(body);
}

static void wasm_emit_matmul_abt_row1_body(
    WasmBuf *body,
    const WasmMatmulSpec *s,
    int col,
    int kk,
    int col_tile,
    int acc0,
    int sum,
    int bvec0,
    int avec,
    int b_addr,
    int a_addr,
    bool use_relaxed_madd
) {
  const int tile_cols = s->n < 128 ? s->n : 128;
  const int k_unroll = (s->k % 16 == 0) ? 4 : 1;
  const int vector_k = (s->k / 4) * 4;

  wasm_emit_i32_const(body, 0);
  wasm_emit_local_set(body, col_tile);
  wasm_emit_loop_header(body);
  wasm_emit_break_if_ge_const(body, col_tile, s->n);

  wasm_emit_local_get(body, col_tile);
  wasm_emit_local_set(body, col);
  wasm_emit_loop_header(body);
  wasm_emit_break_if_ge_local_plus_const(body, col, col_tile, tile_cols);

  for (int c = 0; c < 4; c++) {
    emit_v128_zero(body);
    wasm_emit_local_set(body, acc0 + c);
  }
  wasm_emit_i32_const(body, 0);
  wasm_emit_local_set(body, kk);
  wasm_emit_param_element_addr(body, s->a_param, -1, 0, -1, 0, -1, 0);
  wasm_emit_local_set(body, a_addr);
  wasm_emit_param_element_addr(body, s->b_param, col, s->k, -1, 0, -1, 0);
  wasm_emit_local_set(body, b_addr);
  for (int c = 1; c < 4; c++) {
    wasm_emit_local_get(body, b_addr);
    wasm_emit_i32_const(body, c * s->k * 4);
    wasm_emit_i32_add(body);
    wasm_emit_local_set(body, b_addr + c);
  }

  wasm_emit_loop_header(body);
  wasm_emit_break_if_ge_const(body, kk, vector_k);

  for (int u = 0; u < k_unroll; u++) {
    wasm_emit_local_get(body, a_addr);
    wasm_emit_v128_load_addr_align(body, 4);
    wasm_emit_local_set(body, avec);
    wasm_emit_inc_const(body, a_addr, 16);

    for (int c = 0; c < 4; c++) {
      wasm_emit_local_get(body, b_addr + c);
      wasm_emit_v128_load_addr_align(body, 4);
      wasm_emit_local_set(body, bvec0 + c);
      wasm_emit_inc_const(body, b_addr + c, 16);
      wasm_emit_f32x4_accumulate(body, acc0 + c, avec, bvec0 + c, use_relaxed_madd);
    }
  }

  wasm_emit_inc_const(body, kk, 4 * k_unroll);
  wasm_emit_loop_footer(body);

  for (int c = 0; c < 4; c++) {
    wasm_emit_f32x4_horizontal_sum(body, acc0 + c, avec);
    wasm_emit_local_set(body, sum);

    if (vector_k != s->k) {
      wasm_emit_i32_const(body, vector_k);
      wasm_emit_local_set(body, kk);
      wasm_emit_loop_header(body);
      wasm_emit_break_if_ge_const(body, kk, s->k);

      wasm_emit_local_get(body, sum);
      wasm_emit_param_element_addr(body, s->a_param, kk, 1, -1, 0, -1, 0);
      wasm_emit_f32_load_addr(body);
      wasm_emit_param_element_addr(body, s->b_param, col, s->k, kk, 1, -1, 0);
      if (c != 0) {
        wasm_emit_i32_const(body, c * s->k * 4);
        wasm_emit_i32_add(body);
      }
      wasm_emit_f32_load_addr(body);
      wb_byte(body, WASM_OP_F32_MUL);
      wb_byte(body, WASM_OP_F32_ADD);
      wasm_emit_local_set(body, sum);

      wasm_emit_inc_const(body, kk, 1);
      wasm_emit_loop_footer(body);
    }

    wasm_emit_param_element_addr(body, s->out_param, col, 1, -1, 0, -1, 0);
    if (c != 0) {
      wasm_emit_i32_const(body, c * 4);
      wasm_emit_i32_add(body);
    }
    wasm_emit_local_get(body, sum);
    wasm_emit_f32_store_addr(body);
  }

  wasm_emit_inc_const(body, col, 4);
  wasm_emit_loop_footer(body);

  wasm_emit_inc_const(body, col_tile, tile_cols);
  wasm_emit_loop_footer(body);
}

uint8_t *poly_render_wasm_matmul(PolyUOp *sink, int *size_out, bool use_relaxed_madd) {
  WasmMatmulSpec s;
  if (!wasm_match_matmul_root(sink, &s)) return NULL;
  if (poly_debug_at_least(4)) {
    fprintf(
        stderr,
        "[polygrad:wasm] matmul kind=%s m=%d n=%d k=%d params out=%d a=%d b=%d relaxed=%d\n",
        s.kind == WASM_MATMUL_ABT ? "ABT" : "AB", s.m, s.n, s.k, s.out_param, s.a_param, s.b_param,
        use_relaxed_madd ? 1 : 0
    );
  }
  int slots[3] = {s.out_param, s.a_param, s.b_param};
  int *params[3] = {&s.out_param, &s.a_param, &s.b_param};
  for (int i = 0; i < 3; i++) {
    int rank = 0;
    for (int j = 0; j < 3; j++)
      rank += slots[j] < slots[i];
    *params[i] = rank;
  }
  int n_params = 3;

  MathImports math = {0};
  WasmBuf mod;
  wb_init(&mod);
  wb_module_header(&mod);
  build_type_section(&mod, n_params, &math);
  int n_imported_funcs = build_import_section(&mod, &math);
  build_function_section(&mod);
  build_export_section(&mod, n_imported_funcs);

  WasmBuf body;
  wb_init(&body);

  int n_i32 = s.kind == WASM_MATMUL_ABT ? 14 : 20;
  int n_f32 = s.kind == WASM_MATMUL_ABT ? 1 : (((s.n % 16) != 0 || (s.m % 4) != 0) ? 4 : 0);
  int n_v128 = s.kind == WASM_MATMUL_AB ? 21 : 21;
  int n_local_types = 2 + (n_f32 > 0 ? 1 : 0);
  wb_uleb128(&body, n_local_types);
  wb_uleb128(&body, n_i32);
  wb_byte(&body, WASM_TYPE_I32);
  if (n_f32 > 0) {
    wb_uleb128(&body, n_f32);
    wb_byte(&body, WASM_TYPE_F32);
  }
  wb_uleb128(&body, n_v128);
  wb_byte(&body, WASM_TYPE_V128);

  int next = n_params;
  int row = next++;
  int col = next++;
  int kk = next++;
  int row_tile = -1, col_tile = -1, row_off = -1, k_tile = -1, tile_end = -1;
  int b_addr = -1, a_addr = -1, out_addr = -1;
  if (s.kind == WASM_MATMUL_ABT) {
    row_tile = next++;
    col_tile = next++;
    row_off = next++;
    b_addr = next;
    next += 4;
    a_addr = next;
    next += 4;
  } else {
    row_tile = next++;
    col_tile = next++;
    row_off = next++;
    k_tile = next++;
    tile_end = next++;
    b_addr = next;
    next += 4;
    a_addr = next;
    next += 4;
    out_addr = next;
    next += 4;
  }
  int sum = n_f32 > 0 ? next : -1;
  next += n_f32;
  int acc = next;

  /* f32x4.relaxed_madd is (lhs * rhs) + acc. ABT benefits from the fused
   * horizontal-dot inner loop; AB stays on strict mul+add until it wins
   * consistently in local/browser benchmarks. */
  bool use_relaxed_for_kind = use_relaxed_madd && s.kind == WASM_MATMUL_ABT;

  if (s.kind == WASM_MATMUL_AB)
    wasm_emit_matmul_ab_body(
        &body, &s, row, col, kk, row_tile, col_tile, row_off, k_tile, tile_end, b_addr, a_addr,
        out_addr, acc, acc + 16, acc + 20, sum, use_relaxed_for_kind
    );
  else if (s.m == 1)
    wasm_emit_matmul_abt_row1_body(
        &body, &s, col, kk, col_tile, acc, sum, acc + 16, acc + 20, b_addr, a_addr,
        use_relaxed_for_kind
    );
  else
    wasm_emit_matmul_abt_body(
        &body, &s, row, col, kk, row_tile, col_tile, row_off, acc, sum, acc + 16, acc + 20, b_addr,
        a_addr, use_relaxed_for_kind
    );

  wb_byte(&body, WASM_OP_END);

  WasmBuf sec;
  wb_init(&sec);
  wb_uleb128(&sec, 1);
  wb_uleb128(&sec, body.len);
  wb_append(&sec, &body);
  wb_section(&mod, WASM_SEC_CODE, &sec);
  wb_free(&body);
  wb_free(&sec);

  if (size_out) *size_out = mod.len;
  return mod.data;
}

uint8_t *poly_render_wasm_reduce(PolyUOp *sink, int *size_out) {
  WasmReduceSpec s;
  if (!wasm_match_row_reduce_root(sink, &s)) return NULL;
  if (poly_debug_at_least(4)) {
    fprintf(
        stderr, "[polygrad:wasm] row-reduce m=%d k=%d out=%d max_param=%d\n", s.m, s.k, s.out_param,
        wasm_reduce_expr_max_param(s.expr)
    );
  }

  int n_params = s.out_param;
  int expr_max = wasm_reduce_expr_max_param(s.expr);
  if (expr_max > n_params) n_params = expr_max;
  n_params++;

  MathImports math = {0};
  WasmBuf mod;
  wb_init(&mod);
  wb_module_header(&mod);
  build_type_section(&mod, n_params, &math);
  int n_imported_funcs = build_import_section(&mod, &math);
  build_function_section(&mod);
  build_export_section(&mod, n_imported_funcs);

  WasmBuf body;
  wb_init(&body);

  const int n_i32 = 2; /* row, kk */
  const int n_f32 = 1; /* sum */
  const int n_v128 = 2; /* acc, tmp */
  wb_uleb128(&body, 3);
  wb_uleb128(&body, n_i32);
  wb_byte(&body, WASM_TYPE_I32);
  wb_uleb128(&body, n_f32);
  wb_byte(&body, WASM_TYPE_F32);
  wb_uleb128(&body, n_v128);
  wb_byte(&body, WASM_TYPE_V128);

  int row = n_params;
  int kk = n_params + 1;
  int sum = n_params + n_i32;
  int acc = n_params + n_i32 + n_f32;
  int tmp = acc + 1;
  int vec_end = s.k & ~3;

  wasm_emit_i32_const(&body, 0);
  wasm_emit_local_set(&body, row);
  wasm_emit_loop_header(&body);
  wasm_emit_break_if_ge_const(&body, row, s.m);

  emit_v128_zero(&body);
  wasm_emit_local_set(&body, acc);

  wasm_emit_i32_const(&body, 0);
  wasm_emit_local_set(&body, kk);
  wasm_emit_loop_header(&body);
  wasm_emit_break_if_ge_const(&body, kk, vec_end);

  wasm_emit_local_get(&body, acc);
  if (!wasm_emit_reduce_expr_v128(&body, s.expr, &s, row, kk)) {
    wb_free(&body);
    wb_free(&mod);
    return NULL;
  }
  wb_byte(&body, WASM_SIMD_PREFIX);
  wb_uleb128(&body, WASM_SIMD_F32X4_ADD);
  wasm_emit_local_set(&body, acc);

  wasm_emit_inc_const(&body, kk, 4);
  wasm_emit_loop_footer(&body);

  wasm_emit_f32x4_horizontal_sum(&body, acc, tmp);
  wasm_emit_local_set(&body, sum);

  wasm_emit_loop_header(&body);
  wasm_emit_break_if_ge_const(&body, kk, s.k);

  wasm_emit_local_get(&body, sum);
  if (!wasm_emit_reduce_expr_f32(&body, s.expr, &s, row, kk)) {
    wb_free(&body);
    wb_free(&mod);
    return NULL;
  }
  wb_byte(&body, WASM_OP_F32_ADD);
  wasm_emit_local_set(&body, sum);

  wasm_emit_inc_const(&body, kk, 1);
  wasm_emit_loop_footer(&body);

  wasm_emit_param_element_addr(&body, s.out_param, row, 1, -1, 0, -1, 0);
  wasm_emit_local_get(&body, sum);
  wasm_emit_f32_store_addr(&body);

  wasm_emit_inc_const(&body, row, 1);
  wasm_emit_loop_footer(&body);

  wb_byte(&body, WASM_OP_END);

  WasmBuf sec;
  wb_init(&sec);
  wb_uleb128(&sec, 1);
  wb_uleb128(&sec, body.len);
  wb_append(&sec, &body);
  wb_section(&mod, WASM_SEC_CODE, &sec);
  wb_free(&body);
  wb_free(&sec);

  if (size_out) *size_out = mod.len;
  return mod.data;
}

/* Public API */

uint8_t *poly_render_wasm(PolyCtx *ctx, PolyUOp **uops, int n, int *size_out, bool use_simd) {
  MathImports math;
  int n_params, n_ranges;
  prescan(uops, n, &math, &n_params, &n_ranges);

  WasmBuf mod;
  wb_init(&mod);

  /* Module header */
  wb_module_header(&mod);

  /* Type section */
  build_type_section(&mod, n_params, &math);

  /* Import section (memory + math functions) */
  int n_imported_funcs = build_import_section(&mod, &math);

  /* Function section */
  build_function_section(&mod);

  /* Export section */
  build_export_section(&mod, n_imported_funcs); /* kernel func idx = after imports */

  /* Code section */
  if (use_simd) {
    build_code_simd(ctx, &mod, uops, n, n_params, &math, n_imported_funcs);
  } else {
    build_code_scalar(ctx, &mod, uops, n, n_params, &math, n_imported_funcs);
  }

  *size_out = mod.len;
  return mod.data; /* caller must free() */
}

PolyUOp *poly_rewrite_wasm(PolyCtx *ctx, PolyUOp *sink) {
  /* Approved Wasm divergence: specialized binary emitters compile supported
   * scheduled SINKs directly; unmatched kernels follow Tinygrad 2026-08-22/
   * a9069c177a9d codegen/__init__.py:289-396. */
  if (poly_wasm_can_render_matmul(sink) || poly_wasm_can_render_reduce(sink))
    return poly_type_verify_tensor(ctx, sink) ? sink : NULL;
  PolyRewriteOpts opts = {
      .optimize = poly_kernel_optimize_enabled(sink),
      .beam_width = poly_kernel_beam(sink),
      .caps = poly_wasm_renderer_caps_for_sink(ctx, sink),
      .device = POLY_DEVICE_WASM,
      .opt_policy = POLY_OPT_HEURISTIC,
  };
  return poly_full_rewrite_to_sink_ex(ctx, sink, opts);
}

PolyUOp **poly_linearize_wasm(PolyCtx *ctx, PolyUOp *sink, int *n_out) {
  sink = poly_rewrite_wasm(ctx, sink);
  return poly_do_linearize(ctx, sink, n_out);
}
