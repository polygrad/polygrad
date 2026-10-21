/*
 * renderer/wasm.c — Polygrad's approved Wasm renderer divergence
 *
 * Walks linearized UOps and emits a valid
 * WASM binary module. The generated module imports shared
 * WebAssembly.Memory and exports a single kernel function.
 *
 * The generic emitter supports typed scalars and shaped v128 values.
 * use_simd additionally permits eligible f32/f64 loops to use vector bodies
 * with scalar tails. Kernel selection and loop optimization happen upstream.
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
      /* Like pinned ClangRenderer, lower transcendental ops to kernel arithmetic.
       * Wasm has no native instructions for them; host imports cross into JS
       * once per element. */
      .has_exp2 = false,
      .has_log2 = false,
      .has_sin = false,
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
  PolyUOp **topo = poly_uop_toposort_alloc(ctx, sink, &n);
  if (!topo) return false;
  bool uses_f64 = false;
  for (int i = 0; i < n; i++) {
    PolyDType scalar = topo[i]->dtype;
    if (poly_dtype_is_float(scalar) && scalar.bitsize == 64) {
      uses_f64 = true;
      break;
    }
  }
  poly_uop_toposort_free(topo);
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

static void emit_narrow_integer(WasmBuf *body, PolyDType dst) {
  if (!poly_dtype_is_int(dst) || (dst.bitsize != 8 && dst.bitsize != 16)) return;
  /* PythonProgram CAST truncates to dtype width; CStyle expresses this through
   * a typed cast. Wasm stores every narrow scalar in i32, so do it explicitly. */
  wb_byte(body, WASM_OP_I32_CONST);
  if (poly_dtype_is_unsigned(dst)) {
    wb_sleb128(body, (1 << dst.bitsize) - 1);
    wb_byte(body, WASM_OP_I32_AND);
  } else {
    wb_sleb128(body, 32 - dst.bitsize);
    wb_byte(body, WASM_OP_I32_SHL);
    wb_byte(body, WASM_OP_I32_CONST);
    wb_sleb128(body, 32 - dst.bitsize);
    wb_byte(body, WASM_OP_I32_SHR_S);
  }
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

  /* Saturation matches the SIMD emitter and keeps eagerly evaluated WHERE
   * alternatives from trapping (notably discarded sine range reductions).
   * Narrow unsigned results still use signed staging before their width mask. */
  bool dst_u = poly_dtype_is_unsigned(dst_dt) && dst_dt.bitsize >= 32;
  wb_byte(body, WASM_MISC_PREFIX);
  wb_uleb128(body, (dst_64 ? 4 : 0) + (src_64 ? 2 : 0) + (dst_u ? 1 : 0));
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

static void emit_scalar_load_offset(WasmBuf *body, PolyDType dt, uint32_t offset) {
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
  wb_uleb128(body, offset);
}

static void emit_scalar_load_opcode(WasmBuf *body, PolyDType dt) {
  emit_scalar_load_offset(body, dt, 0);
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

static void emit_v128_load_offset(WasmBuf *body, PolyDType dt, int lanes, uint32_t offset) {
  wb_byte(body, WASM_SIMD_PREFIX);
  int bits = dt.bitsize * lanes;
  if (bits >= 128)
    wb_uleb128(body, WASM_SIMD_V128_LOAD);
  else if (bits <= 32)
    wb_uleb128(body, WASM_SIMD_V128_LOAD32_ZERO);
  else
    wb_uleb128(body, WASM_SIMD_V128_LOAD64_ZERO);
  wb_uleb128(body, 2);
  wb_uleb128(body, offset);
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

static void emit_local_get_boolean_mask(WasmBuf *body, int local) {
  /* WHERE selects whole values (Tinygrad CStyleLanguage.code_for_op).
   * bitselect instead consumes bits: normalize scalar truth to 0/-1,
   * then splat all 128 bits, equally valid for f32 and f64 lanes. */
  wb_byte(body, WASM_OP_LOCAL_GET);
  wb_uleb128(body, local);
  wb_byte(body, WASM_OP_I32_EQZ);
  wb_byte(body, WASM_OP_I32_CONST);
  wb_sleb128(body, 1);
  wb_byte(body, WASM_OP_I32_SUB);
  emit_v128_splat(body, POLY_INT32);
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
    emit_narrow_integer(body, dst_scalar);
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
      if (j == 0 && !wasm_uop_value_is_v128(ctx, u->src[j]))
        emit_local_get_boolean_mask(body, src);
      else
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
  /* Match PythonProgram.exec_alu: a narrow result is truncated before any
   * later widening/conversion. Otherwise uint8 subtraction becomes uint32. */
  emit_narrow_integer(code, dtype);
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

static bool wasm_reg_addr_index(PolyUOp *addr, PolyUOp **reg_base_out, int *idx_out);
static bool wasm_identity_indices(PolyUOp **lanes, int n);

/* Render-local tuple keys borrow immutable SSA nodes. Component keys also
 * describe synthetic transpose interleaves without constructing INDEX/STACK
 * UOps. Only the exact INDEX form used by those interleaves is normalized. */
typedef struct {
  PolyUOp *nodes[4];
  unsigned char components[4]; /* 0: whole node; 1..4: float32 vector lane. */
  int n;
} WasmLaneKey;

typedef struct {
  WasmLaneKey key;
  int matches, gather, block;
} WasmLaneEntry;

typedef struct {
  LocalMap buffers, snapshots;
  int *snapshot_locals;
  PolyMap *lanes;
  PolyArena *lane_arena;
  int n_gathers, gather_base, block;
} WasmRegVectors;

static bool wasm_lane_key_eq(const void *a, const void *b) {
  const WasmLaneKey *x = a, *y = b;
  if (x->n != y->n) return false;
  for (int i = 0; i < x->n; i++)
    if (x->nodes[i] != y->nodes[i] || x->components[i] != y->components[i]) return false;
  return true;
}

static WasmLaneEntry *wasm_lane_entry(WasmRegVectors *v, const WasmLaneKey *key) {
  uint32_t hash = (uint32_t)key->n;
  for (int i = 0; i < key->n; i++) {
    hash = (hash ^ lm_hash(key->nodes[i])) * UINT32_C(16777619);
    hash = (hash ^ key->components[i]) * UINT32_C(16777619);
  }
  WasmLaneEntry *entry = poly_map_get(v->lanes, hash, key, wasm_lane_key_eq);
  if (entry) return entry;
  /* Stable addresses survive recursive matching and map growth. The arena
   * belongs to this render, not to the context or the source graph. */
  entry = poly_arena_alloc(v->lane_arena, sizeof(*entry), _Alignof(WasmLaneEntry));
  if (!entry) {
    fprintf(stderr, "polygrad: Wasm lane cache allocation failed\n");
    abort(); /* Same allocation-failure policy as PolyMap. */
  }
  *entry = (WasmLaneEntry){.key = *key, .matches = -1, .gather = -1, .block = -1};
  poly_map_set(v->lanes, hash, &entry->key, entry, wasm_lane_key_eq);
  return entry;
}

static WasmLaneEntry *wasm_lane_tuple(WasmRegVectors *v, PolyUOp **lanes, int n) {
  WasmLaneKey key = {.n = n};
  assert(n > 1 && n <= 4);
  for (int i = 0; i < n; i++) {
    PolyUOp *u = lanes[i];
    key.nodes[i] = u;
    if (!u || u->op != POLY_OP_INDEX || u->n_src != 2 || u->arg.kind != POLY_ARG_NONE ||
        u->tag != 0 || u->tag_arg.kind != POLY_ARG_NONE || !poly_dtype_eq(u->dtype, POLY_FLOAT32))
      continue;
    PolyUOp *index = u->src[1];
    if (index->op == POLY_OP_CONST && index->n_src == 0 && index->tag == 0 &&
        index->tag_arg.kind == POLY_ARG_NONE && poly_dtype_eq(index->dtype, POLY_WEAKINT) &&
        index->arg.kind == POLY_ARG_INT && index->arg.i >= 0 && index->arg.i < 4) {
      key.nodes[i] = u->src[0];
      key.components[i] = (unsigned char)(index->arg.i + 1);
    }
  }
  return wasm_lane_entry(v, &key);
}

static int wasm_reg_packed_lane(WasmRegVectors *v, PolyUOp *reg, int lane) {
  int width = lm_get(&v->buffers, reg);
  return width > 1 ? (lane % (width / 4)) * 4 + lane / (width / 4) : lane;
}

/* Estimate component-gather work for an isomorphic four-output expression.
 * Constants and accumulator loads do not distinguish register layouts. */
static int wasm_lane_gather_cost(PolyUOp **lanes) {
  bool same = true;
  for (int i = 1; i < 4; i++)
    same &= lanes[i] == lanes[0];
  if (same) return 0;
  PolyUOp *u = lanes[0];
  for (int i = 1; i < 4; i++)
    if (lanes[i]->op != u->op || lanes[i]->n_src != u->n_src) return 32;
  if (u->op == POLY_OP_LOAD || u->op == POLY_OP_CONST) return 0;
  if (u->op == POLY_OP_INDEX) {
    return wasm_identity_indices(lanes, 4) ? 0 : 3;
  }
  if (!poly_opset_has(POLY_GROUP_ALU, u->op)) return 32;
  int cost = 0;
  for (int s = 0; s < u->n_src; s++) {
    PolyUOp *src[4];
    for (int i = 0; i < 4; i++)
      src[i] = lanes[i]->src[s];
    cost += wasm_lane_gather_cost(src);
  }
  return cost;
}

/* Whole-array writes permit either adjacent or strided banks. Choose using
 * operand reuse, not matrix dimensions. Reordering these stores is safe only
 * when all RHS values precede the complete, distinct register write sequence. */
static int wasm_reg_bank_layout(PolyUOp **uops, int n, PolyUOp *reg, int width) {
  int64_t adjacent = 0, strided = 0;
  PolyUOp **values = malloc((size_t)width * sizeof(*values));
  if (!values) return 1;
  for (int i = 0; i < n; i++) {
    PolyUOp *base = NULL;
    if (uops[i]->op != POLY_OP_STORE || !wasm_reg_addr_index(uops[i]->src[0], &base, NULL) ||
        base != reg)
      continue;
    for (int j = 0; j < width; j++) {
      PolyUOp *s = i + j < n ? uops[i + j] : NULL;
      int lane;
      if (!s || s->op != POLY_OP_STORE || !wasm_reg_addr_index(s->src[0], &base, &lane) ||
          base != reg || lane != j) {
        free(values);
        return 1;
      }
      values[j] = s->src[1];
    }
    for (int bank = 0; bank < width / 4; bank++) {
      PolyUOp *lanes[4];
      adjacent += wasm_lane_gather_cost(values + bank * 4);
      for (int l = 0; l < 4; l++)
        lanes[l] = values[bank + l * (width / 4)];
      strided += wasm_lane_gather_cost(lanes);
    }
    i += width - 1;
  }
  free(values);
  return strided < adjacent ? width : 1;
}

/* Promote complete four-lane banks, including larger register arrays. Partial writes, gated loads
 * and reinterpreted accesses keep scalar storage. This is renderer-local SLP, not a change to the
 * scheduled graph or its reduction order. */
static PolyUOp **wasm_pack_reg_stores(PolyCtx *ctx, PolyUOp **uops, int *count, WasmRegVectors *v) {
  int n = *count;
  lm_init(&v->buffers, n * 2);
  lm_init(&v->snapshots, n * 2);
  PolyUOp **out = malloc((size_t)n * sizeof(*out));
  v->snapshot_locals = calloc((size_t)n, sizeof(int));
  if (!out || !v->snapshot_locals) {
    free(out);
    return NULL;
  }
  for (int i = 0; i < n; i++) {
    PolyUOp *r = uops[i];
    if (r->op != POLY_OP_BUFFER || !poly_program_memory_is(r, POLY_ADDR_REG) ||
        !poly_dtype_eq(poly_program_buffer_dtype(r), POLY_FLOAT32) ||
        wasm_reg_storage_size(uops, n, r) <= 0 || wasm_reg_storage_size(uops, n, r) % 4)
      continue;
    bool valid = true;
    for (int j = 0; j < n && valid; j++) {
      PolyUOp *u = uops[j], *base = NULL;
      int lane = -1;
      if ((u->op != POLY_OP_LOAD && u->op != POLY_OP_STORE) ||
          !wasm_reg_addr_index(u->src[0], &base, &lane) || base != r)
        continue;
      if (u->op == POLY_OP_LOAD) {
        valid = u->n_src == 1 && poly_dtype_eq(u->dtype, POLY_FLOAT32);
        /* The scalar emitter, like pinned CStyle, aliases a single-use REG
         * load. Do not promote it if its consumer observes a later write. */
        int uses = 0, last = j;
        for (int k = j + 1; k < n; k++)
          for (int s = 0; s < uops[k]->n_src; s++)
            if (uops[k]->src[s] == u) {
              uses++;
              last = k;
            }
        if (uses == 1)
          for (int k = j + 1; k < last; k++) {
            PolyUOp *written = NULL;
            if (uops[k]->op == POLY_OP_STORE &&
                wasm_reg_addr_index(uops[k]->src[0], &written, NULL) && written == r)
              valid = false;
          }
        continue;
      }
      unsigned mask = 0;
      int bank = lane / 4;
      for (int k = 0; k < 4 && valid; k++) {
        PolyUOp *s = j + k < n ? uops[j + k] : NULL;
        valid = s && s->op == POLY_OP_STORE && s->n_src == 2 &&
                wasm_reg_addr_index(s->src[0], &base, &lane) && base == r && lane >= 0 &&
                lane / 4 == bank && !(mask & (1u << (lane % 4))) &&
                poly_dtype_eq(s->src[1]->dtype, POLY_FLOAT32) &&
                poly_uop_max_numel(ctx, s->src[1]) == 1;
        if (valid) mask |= 1u << (lane % 4);
      }
      j += 3;
    }
    /* SHRINK and typed aliases have separate memory contracts. */
    for (int j = 0; j < n && valid; j++)
      if ((uops[j]->op == POLY_OP_SHRINK || uops[j]->op == POLY_OP_CAST ||
           uops[j]->op == POLY_OP_BITCAST) &&
          uops[j]->n_src && wasm_acc_base(uops[j]->src[0]) == r)
        valid = false;
    if (valid)
      lm_set(&v->buffers, r, wasm_reg_bank_layout(uops, n, r, wasm_reg_storage_size(uops, n, r)));
  }
  int len = 0;
  for (int i = 0; i < n; i++) {
    PolyUOp *u = uops[i], *base = NULL;
    int lane = -1;
    if (u->op == POLY_OP_STORE && wasm_reg_addr_index(u->src[0], &base, &lane) &&
        lm_get(&v->buffers, base) > 0) {
      int width = lm_get(&v->buffers, base);
      int stores = width > 1 ? width : 4;
      for (int bank = 0; bank < stores / 4; bank++) {
        PolyUOp *values[4], *address = NULL;
        for (int k = 0; k < 4; k++) {
          PolyUOp *s = uops[i + (width > 1 ? bank + k * (width / 4) : k)];
          wasm_reg_addr_index(s->src[0], NULL, &lane);
          lane = wasm_reg_packed_lane(v, base, lane);
          values[lane % 4] = s->src[1];
          if (lane % 4 == 0) address = s->src[0];
        }
        out[len++] = poly_uop_stack(ctx, values, 4);
        out[len] = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, address, out[len - 1], poly_arg_none());
        len++;
      }
      i += stores - 1;
    } else
      out[len++] = u;
  }
  *count = len;
  RegLocalMap latest;
  rlm_init(&latest, len);
  for (int i = 0; i < n; i++) {
    PolyUOp *r = uops[i];
    if (r->op != POLY_OP_BUFFER || lm_get(&v->buffers, r) < 1) continue;
    int banks = wasm_reg_storage_size(uops, n, r) / 4;
    int *owners = malloc((size_t)banks * sizeof(*owners));
    if (!owners) continue;
    for (int b = 0; b < banks; b++)
      owners[b] = -1;
    rlm_set(&latest, r, owners, banks);
  }
  int boundary = -1;
  for (int i = 0; i < len; i++) {
    PolyUOp *u = out[i], *base = NULL;
    if (u->op == POLY_OP_STORE || u->op == POLY_OP_RANGE || u->op == POLY_OP_END ||
        u->op == POLY_OP_IF || u->op == POLY_OP_ENDIF)
      boundary = i;
    int lane = -1;
    if (u->op != POLY_OP_LOAD || !wasm_reg_addr_index(u->src[0], &base, &lane) ||
        lm_get(&v->buffers, base) < 1)
      continue;
    lane = wasm_reg_packed_lane(v, base, lane);
    int owner = rlm_get(&latest, base, lane / 4);
    if (owner <= boundary) {
      owner = i;
      for (int b = 0; b < latest.len; b++)
        if (latest.keys[b] == base) latest.locals[b][lane / 4] = owner;
    }
    lm_set(&v->snapshots, u, owner);
  }
  rlm_destroy(&latest);
  return out;
}

static bool wasm_reg_vector_lanes(WasmRegVectors *v, PolyUOp **lanes, int n) {
  if (!v || n != 4 || lanes[0]->op != POLY_OP_LOAD) return false;
  int owner = lm_get(&v->snapshots, lanes[0]);
  if (owner < 0) return false;
  for (int l = 0; l < 4; l++) {
    int index;
    PolyUOp *base = NULL;
    if (lanes[l]->op != POLY_OP_LOAD || lm_get(&v->snapshots, lanes[l]) != owner ||
        !wasm_reg_addr_index(lanes[l]->src[0], &base, &index) ||
        wasm_reg_packed_lane(v, base, index) % 4 != l)
      return false;
  }
  return true;
}

static bool wasm_scalar_load_lanes(PolyCtx *ctx, PolyUOp **lanes, int n) {
  if (n != 4) return false;
  if (lanes[0] == lanes[1] && lanes[0] == lanes[2] && lanes[0] == lanes[3]) return false;
  bool gather = false;
  for (int l = 0; l < n; l++) {
    PolyUOp *u = lanes[l];
    if (!u || !poly_dtype_eq(u->dtype, POLY_FLOAT32) || poly_uop_max_numel(ctx, u) != 1)
      return false;
    if (u->op == POLY_OP_LOAD && u->n_src == 1 && !wasm_acc_base(u->src[0])) {
      gather = true;
    } else if (u->op == POLY_OP_INDEX && u->n_src == 2 && poly_dtype_eq(u->src[0]->dtype, POLY_FLOAT32) && wasm_uop_value_is_v128(ctx, u->src[0])) {
      int index;
      int64_t width = poly_uop_max_numel(ctx, u->src[0]);
      if (!wasm_const_index_checked(u->src[1], &index) || index < 0 || index >= width) return false;
      gather |= width != 4;
    } else
      return false;
  }
  /* Four-wide extracts keep the existing shuffle path. Mixed scalar/two-wide
   * loads retain their SSA locals rather than introducing wider memory reads. */
  return gather;
}

/* Mixed-type SLP keeps the same four independent scalar expressions. Casts
 * reuse the vector conversions above; no math approximation or reassociation.
 * Bool vectors are all-bit masks, so numeric bool casts stay on the fallback. */
static bool wasm_structural_cast32(PolyUOp *u) {
  if (!u || (u->op != POLY_OP_CAST && u->op != POLY_OP_BITCAST) || u->n_src != 1) return false;
  PolyDType src = u->src[0]->dtype, dst = u->dtype;
  return src.bitsize == 32 && dst.bitsize == 32 &&
         (poly_dtype_is_float(src) || poly_dtype_is_int(src)) &&
         (poly_dtype_is_float(dst) || poly_dtype_is_int(dst));
}

static int wasm_structural_i32_opcode(PolyUOp *u) {
  PolyDType dt = wasm_simd_value_dtype(u);
  if (u->op == POLY_OP_WHERE)
    return u->n_src == 3 && dt.bitsize == 32 && poly_dtype_is_bool(u->src[0]->dtype)
               ? WASM_SIMD_V128_BITSELECT
               : -1;
  bool mask = poly_dtype_is_bool(dt);
  if (!mask && (!poly_dtype_is_int(dt) || dt.bitsize != 32)) return -1;
  switch (u->op) {
  case POLY_OP_AND:
    return WASM_SIMD_V128_AND;
  case POLY_OP_OR:
    return WASM_SIMD_V128_OR;
  case POLY_OP_XOR:
    return WASM_SIMD_V128_XOR;
  case POLY_OP_CMPEQ:
    return WASM_SIMD_I32X4_EQ;
  case POLY_OP_CMPNE:
    return WASM_SIMD_I32X4_NE;
  case POLY_OP_CMPLT:
    /* Unsigned order preserves false < true for the 0/-1 mask encoding. */
    return mask || poly_dtype_is_unsigned(dt) ? WASM_SIMD_I32X4_LT_U : WASM_SIMD_I32X4_LT_S;
  default:
    break;
  }
  if (mask) return -1;
  switch (u->op) {
  case POLY_OP_ADD:
    return WASM_SIMD_I32X4_ADD;
  case POLY_OP_SUB:
    return WASM_SIMD_I32X4_SUB;
  case POLY_OP_MUL:
    return WASM_SIMD_I32X4_MUL;
  case POLY_OP_SHL:
    return WASM_SIMD_I32X4_SHL;
  case POLY_OP_SHR:
    return poly_dtype_is_unsigned(dt) ? WASM_SIMD_I32X4_SHR_U : WASM_SIMD_I32X4_SHR_S;
  default:
    return -1;
  }
}

/* Tinygrad 2026-08-22/a9069c177a9d renderer/cstyle.py:52-54 emits a shaped
 * STACK of lane-isomorphic scalar expressions as one native vector value.
 * Wasm has no source compiler, so recognize that same structure here. */
static bool wasm_structural_vec_expr(PolyCtx *ctx, PolyUOp **lanes, int n_lanes, WasmRegVectors *v);

static bool wasm_structural_vec_expr_uncached(
    PolyCtx *ctx,
    PolyUOp **lanes,
    int n_lanes,
    WasmRegVectors *v
) {
  if (!ctx || !lanes || n_lanes <= 1 || n_lanes > 4) return false;

  PolyUOp *first = lanes[0];
  if (!first) return false;
  bool same = true;
  for (int lane = 1; lane < n_lanes; lane++)
    if (lanes[lane] != first) same = false;
  if (same) return poly_uop_max_numel(ctx, first) == 1;
  if (wasm_reg_vector_lanes(v, lanes, n_lanes)) return true;
  if (wasm_scalar_load_lanes(ctx, lanes, n_lanes)) return true;

  if (first->op == POLY_OP_INDEX && first->n_src >= 2) {
    PolyUOp *base = first->src[0];
    if (!wasm_uop_value_is_v128(ctx, base)) return false;
    for (int lane = 0; lane < n_lanes; lane++) {
      int index = -1;
      if (!lanes[lane] || lanes[lane]->op != POLY_OP_INDEX || lanes[lane]->n_src < 2 ||
          !wasm_const_index_checked(lanes[lane]->src[1], &index))
        return false;
      if (n_lanes == 4 && poly_dtype_eq(base->dtype, POLY_FLOAT32)) {
        PolyUOp *v = lanes[lane]->src[0];
        if (!poly_dtype_eq(v->dtype, POLY_FLOAT32) || poly_uop_max_numel(ctx, v) != 4 ||
            !wasm_uop_value_is_v128(ctx, v) || index < 0 || index >= 4)
          return false;
      } else if (lanes[lane]->src[0] != base || index != lane)
        return false;
    }
    return true;
  }

  bool mixed =
      n_lanes == 4 && (wasm_structural_cast32(first) || wasm_structural_i32_opcode(first) >= 0);
  if (!mixed &&
      (!poly_opset_has(POLY_GROUP_ALU, first->op) || !wasm_simd_loop_can_vectorize_alu(first)))
    return false;
  for (int lane = 1; lane < n_lanes; lane++)
    if (!lanes[lane] || lanes[lane]->op != first->op || lanes[lane]->n_src != first->n_src ||
        !poly_dtype_eq(lanes[lane]->dtype, first->dtype))
      return false;

  if (first->op == POLY_OP_SHL || first->op == POLY_OP_SHR) {
    /* Wasm SIMD shifts take one scalar count, not four lane counts. */
    if (first->src[1]->dtype.bitsize != 32 || poly_uop_max_numel(ctx, first->src[1]) != 1)
      return false;
    for (int lane = 1; lane < n_lanes; lane++)
      if (lanes[lane]->src[1] != first->src[1]) return false;
  }
  for (int src = 0; src < first->n_src; src++) {
    PolyUOp *src_lanes[4];
    for (int lane = 0; lane < n_lanes; lane++)
      src_lanes[lane] = lanes[lane]->src[src];
    if (!wasm_structural_vec_expr(ctx, src_lanes, n_lanes, v)) return false;
  }
  return true;
}

static bool wasm_structural_vec_expr(
    PolyCtx *ctx,
    PolyUOp **lanes,
    int n_lanes,
    WasmRegVectors *v
) {
  if (!ctx || !lanes || n_lanes <= 1 || n_lanes > 4) return false;
  if (!v || !v->lanes) return wasm_structural_vec_expr_uncached(ctx, lanes, n_lanes, v);
  /* Feasibility is invariant during this rendering. Match each lane tuple
   * once, not once per path through a shared expression DAG. */
  WasmLaneEntry *entry = wasm_lane_tuple(v, lanes, n_lanes);
  int cached = entry->matches;
  if (cached >= 0) return cached != 0;
  bool matches = wasm_structural_vec_expr_uncached(ctx, lanes, n_lanes, v);
  entry->matches = matches;
  return matches;
}

static bool wasm_identity_indices(PolyUOp **lanes, int n) {
  for (int k = 0; k < n; k++) {
    int index;
    if (lanes[k]->src[0] != lanes[0]->src[0] ||
        !wasm_const_index_checked(lanes[k]->src[1], &index) || index != k)
      return false;
  }
  return true;
}

/* Adjacent transpose columns share pair interleaves: [a0,b0,a1,b1]
 * and [c0,d0,c1,d1] supply both columns 0 and 1. Keys describe immutable
 * SSA components, so the existing block-local gather cache owns their reuse. */
static WasmLaneEntry *wasm_transpose_pair(WasmRegVectors *v, PolyUOp **lanes, int n, int pair) {
  int column;
  if (n != 4 || !wasm_const_index_checked(lanes[0]->src[1], &column)) return NULL;
  for (int i = 1; i < n; i++) {
    int index;
    if (!wasm_const_index_checked(lanes[i]->src[1], &index) || index != column) return NULL;
  }
  WasmLaneKey key = {.n = 4};
  for (int i = 0; i < 4; i++) {
    key.nodes[i] = lanes[2 * pair + i % 2]->src[0];
    key.components[i] = (unsigned char)((column & ~1) + i / 2 + 1);
  }
  return wasm_lane_entry(v, &key);
}

static void wasm_mark_structural_vec_expr(
    PolyCtx *ctx,
    PolyUOp **uops,
    int n,
    PolyUOp **lanes,
    int n_lanes,
    bool *covered,
    bool *required,
    WasmRegVectors *v
) {
  bool same = true;
  for (int lane = 1; lane < n_lanes; lane++)
    if (lanes[lane] != lanes[0]) same = false;
  if (same) {
    /* A broadcast scalar may also be covered by another vector expression.
     * Its scalar local and dependencies must remain available to the emitter. */
    for (int i = 0; i < n; i++)
      if (uops[i] == lanes[0]) required[i] = true;
    return;
  }
  if (lanes[0]->op != POLY_OP_INDEX && lanes[0]->op != POLY_OP_LOAD) {
    WasmLaneEntry *entry = wasm_lane_tuple(v, lanes, n_lanes);
    int cached = entry->gather;
    if (cached != -1) {
      if (cached == -2) entry->gather = v->n_gathers++;
      return;
    }
    /* Only repeated expressions need a temporary. Visiting their children
     * once also prevents exponential planning of shared expression DAGs. */
    entry->gather = -2;
  }
  /* Keep memory reads at their scheduled positions. Odd strides can prevent
   * core load vectorization without preventing lane-wise SIMD arithmetic. */
  if (wasm_scalar_load_lanes(ctx, lanes, n_lanes)) {
    for (int lane = 0; lane < n_lanes; lane++)
      for (int i = 0; i < n; i++)
        if (uops[i] == lanes[lane]) required[i] = true;
    WasmLaneEntry *entry = wasm_lane_tuple(v, lanes, n_lanes);
    if (entry->gather < 0) entry->gather = v->n_gathers++;
    return;
  }

  for (int lane = 0; lane < n_lanes; lane++)
    for (int i = 0; i < n; i++)
      if (uops[i] == lanes[lane]) covered[i] = true;

  if (lanes[0]->op == POLY_OP_INDEX) {
    if (!wasm_identity_indices(lanes, n_lanes)) {
      WasmLaneEntry *entry = wasm_lane_tuple(v, lanes, n_lanes);
      if (entry->gather < 0) entry->gather = v->n_gathers++;
      for (int pair = 0; pair < 2; pair++) {
        WasmLaneEntry *interleave = wasm_transpose_pair(v, lanes, n_lanes, pair);
        if (interleave && interleave->gather < 0) interleave->gather = v->n_gathers++;
      }
    }
    return;
  }
  if (lanes[0]->op == POLY_OP_LOAD) return;
  for (int src = 0; src < lanes[0]->n_src; src++) {
    PolyUOp *src_lanes[4];
    for (int lane = 0; lane < n_lanes; lane++)
      src_lanes[lane] = lanes[lane]->src[src];
    wasm_mark_structural_vec_expr(ctx, uops, n, src_lanes, n_lanes, covered, required, v);
  }
}

static void wasm_plan_structural_vecs(
    PolyCtx *ctx,
    PolyUOp **uops,
    int n,
    bool *fused_stack,
    bool *skip,
    WasmRegVectors *v
) {
  bool *covered = calloc((size_t)n, sizeof(*covered));
  bool *required = calloc((size_t)n, sizeof(*required));
  if (!covered || !required) {
    free(covered);
    free(required);
    return;
  }

  for (int i = 0; i < n; i++) {
    PolyUOp *u = uops[i];
    if (!wasm_stack_native_vec(u, NULL, NULL) ||
        !wasm_structural_vec_expr(ctx, u->src, u->n_src, v))
      continue;
    fused_stack[i] = true;
    covered[i] = true;
    wasm_mark_structural_vec_expr(ctx, uops, n, u->src, u->n_src, covered, required, v);
  }

  for (int i = 0; i < n; i++)
    skip[i] = covered[i] && !fused_stack[i] && !required[i];
  /* A shared scalar consumer needs its whole dependency chain, even when
   * those nodes also belong to a fused STACK. LINEAR is definition-first. */
  for (int i = n - 1; i >= 0; i--) {
    if (skip[i] || fused_stack[i]) continue;
    for (int j = 0; j < i; j++)
      if (skip[j])
        for (int src = 0; src < uops[i]->n_src; src++)
          if (uops[i]->src[src] == uops[j]) skip[j] = false;
  }
  free(covered);
  free(required);
}

typedef struct {
  int owner, lane, local;
  int roots[4];
} WasmScalarPack;

static PolyOps wasm_scalar_pack_op(PolyUOp *u) {
  /* The Wasm MULACC emitter already uses separate multiply/add instructions,
   * not fused FMA. Pack its product; leave the accumulator addition in place. */
  return u->op == POLY_OP_MULACC ? POLY_OP_MUL : u->op;
}

static int wasm_scalar_pack_n_src(PolyUOp *u) {
  return u->op == POLY_OP_MULACC ? 2 : u->n_src;
}

/* Recover independent lane arithmetic even when its consumers are scalar
 * reductions, not STACKs. This is renderer-local SLP: the original scalar
 * consumer tree and its rounding order remain unchanged. */
static int wasm_scalar_pack_lane(PolyCtx *ctx, PolyUOp *u) {
  if (!u || !poly_dtype_eq(u->dtype, POLY_FLOAT32) || poly_uop_max_numel(ctx, u) != 1) return -1;
  PolyOps op = wasm_scalar_pack_op(u);
  if (op != POLY_OP_ADD && op != POLY_OP_SUB && op != POLY_OP_MUL && op != POLY_OP_FDIV &&
      op != POLY_OP_NEG && op != POLY_OP_SQRT)
    return -1;
  int lane = -1;
  for (int s = 0; s < wasm_scalar_pack_n_src(u); s++) {
    PolyUOp *v = u->src[s];
    if (v->op != POLY_OP_INDEX || v->n_src < 2 || !wasm_uop_value_is_v128(ctx, v->src[0])) continue;
    int index;
    if (!poly_dtype_eq(v->src[0]->dtype, POLY_FLOAT32) || poly_uop_max_numel(ctx, v->src[0]) != 4 ||
        !wasm_const_index_checked(v->src[1], &index) || index < 0 || index >= 4 ||
        (lane >= 0 && lane != index))
      return -1;
    lane = index;
  }
  return lane;
}

static WasmScalarPack *wasm_plan_scalar_packs(
    PolyCtx *ctx,
    PolyUOp **uops,
    int n,
    const bool *skip
) {
  WasmScalarPack *packs = calloc((size_t)n, sizeof(*packs));
  if (!packs) return NULL;
  LocalMap positions;
  lm_init(&positions, n * 2);
  for (int i = 0; i < n; i++)
    lm_set(&positions, uops[i], i);
  for (int i = 0; i < n; i++) {
    if (packs[i].owner || (skip && skip[i])) continue;
    PolyUOp *first = uops[i];
    int lane = wasm_scalar_pack_lane(ctx, first);
    if (lane < 0) continue;
    PolyUOp *lanes[4] = {0};
    int roots[4] = {0}, found = 1;
    lanes[lane] = first;
    roots[lane] = i;
    for (int j = i + 1; j < n && found < 4; j++) {
      PolyUOp *v = uops[j];
      /* Never move arithmetic across effects or control-flow boundaries.
       * In particular a REG LOAD may alias a mutable Wasm local. */
      if (v->op == POLY_OP_STORE || v->op == POLY_OP_RANGE || v->op == POLY_OP_END ||
          v->op == POLY_OP_IF || v->op == POLY_OP_ENDIF)
        break;
      if (packs[j].owner || (skip && skip[j]) ||
          wasm_scalar_pack_op(v) != wasm_scalar_pack_op(first) ||
          wasm_scalar_pack_n_src(v) != wasm_scalar_pack_n_src(first))
        continue;
      int other = wasm_scalar_pack_lane(ctx, v);
      if (other < 0 || lanes[other]) continue;
      bool matches = true;
      for (int s = 0; s < wasm_scalar_pack_n_src(first); s++) {
        PolyUOp *a = first->src[s], *b = v->src[s];
        if (a == b) continue;
        if (a->op != POLY_OP_INDEX || b->op != POLY_OP_INDEX || a->n_src < 2 || b->n_src < 2 ||
            a->src[0] != b->src[0] || !wasm_uop_value_is_v128(ctx, a->src[0]))
          matches = false;
      }
      if (matches) {
        lanes[other] = v;
        roots[other] = j;
        found++;
      }
    }
    if (found != 4) continue;
    bool ready = true;
    for (int s = 0; s < wasm_scalar_pack_n_src(first); s++) {
      PolyUOp *operands[4];
      for (int k = 0; k < 4; k++)
        operands[k] = lanes[k]->src[s];
      if (!wasm_structural_vec_expr(ctx, operands, 4, NULL)) ready = false;
      PolyUOp *value = first->src[s];
      if (value->op == POLY_OP_INDEX && value->n_src >= 2 &&
          wasm_uop_value_is_v128(ctx, value->src[0]))
        value = value->src[0];
      int pos = lm_get(&positions, value);
      if (pos < 0 || pos >= i || (skip && skip[pos])) ready = false;
    }
    if (!ready) continue;
    for (int k = 0; k < 4; k++) {
      packs[roots[k]].owner = i + 1;
      packs[roots[k]].lane = k;
      packs[i].roots[k] = roots[k];
    }
  }
  lm_destroy(&positions);
  return packs;
}

static bool wasm_emit_structural_vec_expr(
    PolyCtx *ctx,
    WasmBuf *body,
    LocalMap *locals,
    PolyUOp **lanes,
    int n_lanes,
    WasmRegVectors *v
);

static bool wasm_emit_structural_vec_expr_uncached(
    PolyCtx *ctx,
    WasmBuf *body,
    LocalMap *locals,
    PolyUOp **lanes,
    int n_lanes,
    WasmRegVectors *v
) {
  PolyUOp *first = lanes[0];
  if (wasm_reg_vector_lanes(v, lanes, n_lanes)) {
    wb_byte(body, WASM_OP_LOCAL_GET);
    wb_uleb128(body, v->snapshot_locals[lm_get(&v->snapshots, first)]);
    return true;
  }
  if (wasm_scalar_load_lanes(ctx, lanes, n_lanes)) {
    WasmLaneEntry *entry = wasm_lane_tuple(v, lanes, n_lanes);
    int gather = entry->gather;
    if (gather >= 0 && entry->block == v->block) {
      wb_byte(body, WASM_OP_LOCAL_GET);
      wb_uleb128(body, v->gather_base + gather);
      return true;
    }
    emit_v128_zero(body);
    for (int l = 0; l < n_lanes; l++) {
      int local = lm_get(locals, lanes[l]);
      if (local < 0) return false;
      wb_byte(body, WASM_OP_LOCAL_GET);
      wb_uleb128(body, local);
      emit_v128_replace_lane(body, POLY_FLOAT32, l);
    }
    if (gather >= 0) {
      wb_byte(body, WASM_OP_LOCAL_TEE);
      wb_uleb128(body, v->gather_base + gather);
      entry->block = v->block;
    }
    return true;
  }
  bool same = true;
  for (int lane = 1; lane < n_lanes; lane++)
    if (lanes[lane] != first) same = false;
  if (same) {
    int lane, source;
    if (n_lanes == 4 && first->op == POLY_OP_INDEX && first->n_src == 2 &&
        poly_dtype_eq(first->dtype, POLY_FLOAT32) &&
        poly_dtype_eq(first->src[0]->dtype, POLY_FLOAT32) &&
        poly_uop_max_numel(ctx, first->src[0]) == 4 && wasm_uop_value_is_v128(ctx, first->src[0]) &&
        wasm_const_index_checked(first->src[1], &lane) && lane >= 0 && lane < 4 &&
        (source = lm_get(locals, first->src[0])) >= 0) {
      /* Broadcast the existing SSA vector lane, without a scalar round trip
       * or a memory reload that could cross an intervening mutation. */
      for (int i = 0; i < 2; i++) {
        wb_byte(body, WASM_OP_LOCAL_GET);
        wb_uleb128(body, source);
      }
      uint8_t mask[16];
      for (int i = 0; i < 16; i++)
        mask[i] = lane * 4 + i % 4;
      emit_v128_shuffle(body, mask);
      return true;
    }
    int local = lm_get(locals, first);
    if (local < 0) return false;
    if (poly_dtype_is_bool(first->dtype)) {
      emit_local_get_boolean_mask(body, local);
    } else {
      wb_byte(body, WASM_OP_LOCAL_GET);
      wb_uleb128(body, local);
      emit_v128_splat(body, first->dtype);
    }
    return true;
  }

  if (first->op == POLY_OP_INDEX) {
    int indices[4];
    for (int k = 0; k < n_lanes; k++)
      wasm_const_index_checked(lanes[k]->src[1], &indices[k]);
    if (!wasm_identity_indices(lanes, n_lanes)) {
      WasmLaneEntry *entry = wasm_lane_tuple(v, lanes, n_lanes);
      int gather = entry->gather;
      if (gather >= 0 && entry->block == v->block) {
        wb_byte(body, WASM_OP_LOCAL_GET);
        wb_uleb128(body, v->gather_base + gather);
        return true;
      }
      /* Transpose/gather vector components, not reduction terms. Recursing
       * through the unchanged ALU tree preserves each output's add order. */
      WasmLaneEntry *pairs[2] = {
          wasm_transpose_pair(v, lanes, n_lanes, 0), wasm_transpose_pair(v, lanes, n_lanes, 1)};
      int pair_locals[2] = {pairs[0] ? pairs[0]->gather : -1, pairs[1] ? pairs[1]->gather : -1};
      bool interleaved = pair_locals[0] >= 0 && pair_locals[1] >= 0;
      for (int pair = 0; pair < 2; pair++) {
        WasmLaneEntry *interleave = pairs[pair];
        int cached = pair_locals[pair];
        if (interleaved && interleave->block == v->block) {
          wb_byte(body, WASM_OP_LOCAL_GET);
          wb_uleb128(body, v->gather_base + cached);
          continue;
        }
        for (int k = 0; k < 2; k++) {
          int local = lm_get(locals, lanes[2 * pair + k]->src[0]);
          if (local < 0) return false;
          wb_byte(body, WASM_OP_LOCAL_GET);
          wb_uleb128(body, local);
        }
        uint8_t mask[16];
        for (int k = 0; k < 4; k++)
          for (int b = 0; b < 4; b++) {
            int index = interleaved ? (indices[0] & ~1) + k / 2 : indices[2 * pair + k % 2];
            mask[4 * k + b] = 16 * (k % 2) + 4 * index + b;
          }
        emit_v128_shuffle(body, mask);
        if (interleaved) {
          wb_byte(body, WASM_OP_LOCAL_TEE);
          wb_uleb128(body, v->gather_base + cached);
          interleave->block = v->block;
        }
      }
      uint8_t join[16];
      int offset = interleaved ? 8 * (indices[0] % 2) : 0;
      for (int i = 0; i < 16; i++)
        join[i] = 16 * (i / 8) + offset + i % 8;
      emit_v128_shuffle(body, join);
      if (gather >= 0) {
        wb_byte(body, WASM_OP_LOCAL_TEE);
        wb_uleb128(body, v->gather_base + gather);
        entry->block = v->block;
      }
      return true;
    }
    int local = lm_get(locals, first->src[0]);
    if (local < 0) return false;
    wb_byte(body, WASM_OP_LOCAL_GET);
    wb_uleb128(body, local);
    return true;
  }

  int order[3] = {0, 1, 2};
  int n_src = first->n_src;
  bool shift = first->op == POLY_OP_SHL || first->op == POLY_OP_SHR;
  if (shift) n_src = 1;
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
    if (!wasm_emit_structural_vec_expr(ctx, body, locals, src_lanes, n_lanes, v)) return false;
  }

  if (shift) {
    int local = lm_get(locals, first->src[1]);
    if (local < 0) return false;
    wb_byte(body, WASM_OP_LOCAL_GET);
    wb_uleb128(body, local);
  }
  if (wasm_structural_cast32(first))
    return emit_v128_cast_opcode(
        body, first->src[0]->dtype, first->dtype, first->op == POLY_OP_BITCAST
    );
  int opcode = wasm_structural_i32_opcode(first);
  if (opcode >= 0) {
    wb_byte(body, WASM_SIMD_PREFIX);
    wb_uleb128(body, opcode);
    return true;
  }
  PolyDType dtype = wasm_simd_value_dtype(first);
  if (dtype.bitsize == 64)
    emit_alu_simd_f64x2(body, first->op);
  else
    emit_alu_simd_f32x4(body, first->op);
  return true;
}

static bool wasm_emit_structural_vec_expr(
    PolyCtx *ctx,
    WasmBuf *body,
    LocalMap *locals,
    PolyUOp **lanes,
    int n_lanes,
    WasmRegVectors *v
) {
  WasmLaneEntry *entry = NULL;
  int cached = -1;
  if (lanes[0]->op != POLY_OP_INDEX && lanes[0]->op != POLY_OP_LOAD) {
    entry = wasm_lane_tuple(v, lanes, n_lanes);
    cached = entry->gather;
  }
  /* Reuse only within the current control-flow block. The original LOAD
   * snapshots and ordered operations remain the sources of these values. */
  if (cached >= 0 && entry->block == v->block) {
    wb_byte(body, WASM_OP_LOCAL_GET);
    wb_uleb128(body, v->gather_base + cached);
    return true;
  }
  if (!wasm_emit_structural_vec_expr_uncached(ctx, body, locals, lanes, n_lanes, v)) return false;
  if (cached >= 0) {
    wb_byte(body, WASM_OP_LOCAL_TEE);
    wb_uleb128(body, v->gather_base + cached);
    entry->block = v->block;
  }
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
  if (poly_dtype_is_int(dst) && dst.bitsize < 32) return false;
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
      /* This loop emitter has no masked-memory implementation. The generic
       * emitter preserves gate/alternate without touching masked-off memory. */
      if (u->n_src != 1 || !wasm_simd_unit_index(u->src[0], range, &bits)) return false;
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
  PolyUOp *idx = poly_as_index(addr);
  if (!idx || idx->op != POLY_OP_INDEX || idx->n_src < 2) return false;
  PolyUOp *reg_base = wasm_acc_base(idx->src[0]);
  if (!reg_base || reg_base->op != POLY_OP_BUFFER ||
      !poly_program_memory_is(reg_base, POLY_ADDR_REG))
    return false;
  int reg_idx = 0;
  if (!wasm_const_index_checked(idx->src[1], &reg_idx)) return false;
  if (reg_idx < 0) return false;
  if (reg_base_out) *reg_base_out = reg_base;
  if (idx_out) *idx_out = reg_idx;
  return true;
}

static bool wasm_load_aliases_reg(PolyUOp *load, LocalMap *child_count) {
  /* Tinygrad CStyleLanguage._render materializes multi-use REG LOADs.
   * Gates need a computed value; only a plain single-use load is a local alias. */
  return load->n_src == 1 && wasm_reg_addr_index(load->src[0], NULL, NULL) &&
         lm_get(child_count, load) == 1;
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

/* Keep all-i32 address arguments and the buffer-only fast path. Both Wasm
 * emitters convert the same signed64 numeric vals as the native C wrapper. */
static void emit_scalar_param(WasmBuf *body, PolyUOp *u, int abi, int local) {
  wb_byte(body, WASM_OP_LOCAL_GET);
  wb_uleb128(body, abi);
  wb_byte(body, WASM_OP_I64_LOAD);
  wb_uleb128(body, 3);
  wb_uleb128(body, 0);
  emit_cast_stack_value(body, POLY_INT64, u->dtype);
  wb_byte(body, WASM_OP_LOCAL_SET);
  wb_uleb128(body, local);
}

typedef struct {
  PolyUOp *address;
  uint32_t offset;
} WasmLoadAddress;

static bool wasm_address_constant(PolyUOp *u, int64_t *value) {
  if (!u || !poly_dtype_is_int(u->dtype) || u->dtype.bitsize < 32) return false;
  if (u->op == POLY_OP_CAST && u->n_src == 1 &&
      (!poly_dtype_is_int(u->src[0]->dtype) || u->src[0]->dtype.bitsize < 32))
    return false;
  int c = 0;
  if (!wasm_const_index_checked(u, &c)) return false;
  *value = c;
  return true;
}

/* Factor a nonnegative constant out of an affine element index. Only immutable
 * loop/parameter leaves are admitted: re-reading a register LOAD here could
 * observe a STORE after the original address was computed. */
static PolyUOp *wasm_index_without_constant(PolyCtx *ctx, PolyUOp *u, int64_t *offset, int depth) {
  *offset = 0;
  if (!u || depth > 64 || !poly_dtype_is_int(u->dtype) || u->dtype.bitsize < 32) return NULL;
  int64_t c = 0;
  if (wasm_address_constant(u, &c)) {
    if (c < 0 || c > INT_MAX) return NULL;
    *offset = c;
    return poly_uop_const(ctx, poly_arg_int(0), u->dtype);
  }
  if (u->op == POLY_OP_RANGE || (u->op == POLY_OP_PARAM && poly_uop_is_alu_param(u))) return u;
  if (u->op == POLY_OP_CAST && u->n_src == 1) {
    PolyUOp *x = wasm_index_without_constant(ctx, u->src[0], offset, depth + 1);
    return x ? poly_uop1(ctx, POLY_OP_CAST, u->dtype, x, u->arg) : NULL;
  }
  if (u->op == POLY_OP_MULACC && u->n_src == 3) {
    /* Late integer MULACC is exact modular multiply/add, not a floating FMA. */
    PolyUOp *product = poly_uop2(ctx, POLY_OP_MUL, u->dtype, u->src[0], u->src[1], poly_arg_none());
    PolyUOp *sum = poly_uop2(ctx, POLY_OP_ADD, u->dtype, product, u->src[2], poly_arg_none());
    return wasm_index_without_constant(ctx, sum, offset, depth + 1);
  }
  if (u->n_src != 2) return NULL;
  if (u->op == POLY_OP_SHL) {
    if (!wasm_address_constant(u->src[1], &c) || c < 0 || c > 30) return NULL;
    PolyUOp *factor = poly_uop_const(ctx, poly_arg_int(INT64_C(1) << c), u->dtype);
    PolyUOp *product = poly_uop2(ctx, POLY_OP_MUL, u->dtype, u->src[0], factor, poly_arg_none());
    return wasm_index_without_constant(ctx, product, offset, depth + 1);
  }
  if (u->op == POLY_OP_ADD) {
    int64_t a = 0, b = 0;
    PolyUOp *x = wasm_index_without_constant(ctx, u->src[0], &a, depth + 1);
    PolyUOp *y = wasm_index_without_constant(ctx, u->src[1], &b, depth + 1);
    if (!x || !y || a > INT_MAX - b) return NULL;
    *offset = a + b;
    if (poly_uop_const_i64(x, &c) == 0 && c == 0) return y;
    if (poly_uop_const_i64(y, &c) == 0 && c == 0) return x;
    return poly_uop2(ctx, u->op, u->dtype, x, y, u->arg);
  }
  if (u->op == POLY_OP_MUL) {
    PolyUOp *factor = u->src[1], *value = u->src[0];
    if (!wasm_address_constant(factor, &c)) {
      factor = u->src[0];
      value = u->src[1];
      if (!wasm_address_constant(factor, &c)) return NULL;
    }
    if (c < 1 || c > INT_MAX) return NULL;
    int64_t a = 0;
    PolyUOp *x = wasm_index_without_constant(ctx, value, &a, depth + 1);
    if (!x || a > INT_MAX / c) return NULL;
    *offset = a * c;
    return poly_uop2(ctx, u->op, u->dtype, x, factor, u->arg);
  }
  return NULL;
}

static WasmLoadAddress wasm_plan_load_address(PolyCtx *ctx, PolyUOp *load) {
  WasmLoadAddress result = {0};
  /* A gated LOAD does not dominate later uses of a shared address local. */
  if (load->op != POLY_OP_LOAD || load->n_src != 1) return result;
  PolyUOp *addr = load->src[0];
  if ((addr->op != POLY_OP_INDEX && addr->op != POLY_OP_SHRINK) || addr->n_src < 2) return result;
  PolyUOp *buffer = addr->src[0];
  if (buffer->op != POLY_OP_PARAM || !poly_program_memory_is(buffer, POLY_ADDR_GLOBAL) ||
      !poly_dtype_eq(buffer->dtype, addr->dtype) || !poly_dtype_eq(buffer->dtype, load->dtype))
    return result;
  int64_t count = poly_uop_max_numel(ctx, buffer);
  int64_t width = poly_uop_max_numel(ctx, load), lo = 0, hi = 0;
  int bytes = dt_elem_size(buffer->dtype);
  /* Memarg addition does not wrap like i32.add. Only valid accesses within a
   * declared allocation may use it; unknown/negative indices retain the exact
   * original arithmetic, including its masking and wrap behavior. */
  if (count < 1 || width < 1 || width > count || bytes < 1 || count > INT_MAX / bytes)
    return result;
  int64_t offset = 0;
  PolyUOp *index = wasm_index_without_constant(ctx, addr->src[1], &offset, 0);
  if (!index || offset > INT_MAX / bytes) return result;
  /* The expanded affine expression also proves bounds for late integer
   * MULACC, for which the shared interval analysis conservatively says unknown. */
  poly_uop_minmax(ctx, index, &lo, &hi);
  if (lo < 0 || hi > count - width - offset) return result;
  result.address = poly_uop2(ctx, POLY_OP_INDEX, addr->dtype, buffer, index, poly_arg_none());
  result.offset = (uint32_t)(offset * bytes);
  return result;
}

static void wasm_emit_address_index(WasmBuf *body, LocalMap *locals, PolyUOp *index) {
  int local = lm_get(locals, index);
  if (local >= 0) {
    emit_local_get_as_i32(body, local, index->dtype);
    return;
  }
  int64_t c = 0;
  if (poly_uop_const_i64(index, &c) == 0) {
    wb_byte(body, WASM_OP_I32_CONST);
    wb_sleb128(body, c);
    return;
  }
  if (index->op == POLY_OP_CAST) {
    wasm_emit_address_index(body, locals, index->src[0]);
    return; /* All admitted casts retain the low 32 address bits. */
  }
  /* Only new ADD/MUL nodes from wasm_index_without_constant reach this path. */
  assert(index->op == POLY_OP_ADD || index->op == POLY_OP_MUL);
  wasm_emit_address_index(body, locals, index->src[0]);
  wasm_emit_address_index(body, locals, index->src[1]);
  wb_byte(body, index->op == POLY_OP_ADD ? WASM_OP_I32_ADD : WASM_OP_I32_MUL);
}

static uint32_t wasm_emit_load_address(
    WasmBuf *body,
    LocalMap *locals,
    LocalMap *addresses,
    LocalMap *blocks,
    WasmLoadAddress plan,
    PolyUOp *original,
    int block,
    int *next_i32
) {
  if (!plan.address) {
    wb_byte(body, WASM_OP_LOCAL_GET);
    wb_uleb128(body, lm_get(locals, original));
    return 0;
  }
  int local = lm_get(addresses, plan.address);
  if (local < 0 || lm_get(blocks, plan.address) != block) {
    local = (*next_i32)++;
    wb_byte(body, WASM_OP_LOCAL_GET);
    wb_uleb128(body, lm_get(locals, plan.address->src[0]));
    wasm_emit_address_index(body, locals, plan.address->src[1]);
    wb_byte(body, WASM_OP_I32_CONST);
    wb_sleb128(body, dt_elem_size(plan.address->src[0]->dtype));
    wb_byte(body, WASM_OP_I32_MUL);
    wb_byte(body, WASM_OP_I32_ADD);
    wb_byte(body, WASM_OP_LOCAL_SET);
    wb_uleb128(body, local);
    lm_set(addresses, plan.address, local);
    lm_set(blocks, plan.address, block);
  }
  wb_byte(body, WASM_OP_LOCAL_GET);
  wb_uleb128(body, local);
  return plan.offset;
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
  WasmRegVectors reg_vectors = {0};
  reg_vectors.lanes = poly_map_new((size_t)n);
  reg_vectors.lane_arena = poly_arena_new(4096);
  PolyUOp **packed_uops = wasm_pack_reg_stores(ctx, uops, &n, &reg_vectors);
  if (packed_uops) uops = packed_uops;
  bool *fused_stack = calloc((size_t)n, sizeof(*fused_stack));
  bool *skip = calloc((size_t)n, sizeof(*skip));
  if (fused_stack && skip) wasm_plan_structural_vecs(ctx, uops, n, fused_stack, skip, &reg_vectors);
  WasmScalarPack *packs = wasm_plan_scalar_packs(ctx, uops, n, skip);
  WasmLoadAddress *load_addresses = calloc((size_t)n, sizeof(*load_addresses));
  LocalMap addresses, address_blocks;
  lm_init(&addresses, n * 2);
  lm_init(&address_blocks, n * 2);
  if (load_addresses)
    for (int i = 0; i < n; i++)
      load_addresses[i] = wasm_plan_load_address(ctx, uops[i]);

  /* Same direct-edge count as pinned CStyleLanguage._render, including
   * repeated operands. Keep counting and emission on the same alias rule. */
  LocalMap child_count;
  lm_init(&child_count, n * 2);
  for (int i = 0; i < n; i++)
    if (uops[i]->op == POLY_OP_LOAD) lm_set(&child_count, uops[i], 0);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < uops[i]->n_src; j++) {
      PolyUOp *src = uops[i]->src[j];
      if (src->op == POLY_OP_LOAD) lm_set(&child_count, src, lm_get(&child_count, src) + 1);
    }

  /* --- Count locals needed (beyond function params) --- */
  int n_locals_i32 = 0, n_locals_i64 = 0, n_locals_f32 = 0, n_locals_f64 = 0;
  int n_locals_v128 = reg_vectors.n_gathers;

  /* First pass: count locals */
  for (int i = 0; i < n; i++) {
    if (uops[i]->op == POLY_OP_LOAD && lm_get(&reg_vectors.snapshots, uops[i]) == i)
      n_locals_v128++;
    if (skip && skip[i]) continue;
    if (load_addresses && load_addresses[i].address) n_locals_i32++;
    if (packs && packs[i].owner == i + 1) n_locals_v128++;
    PolyUOp *u = uops[i];
    if (u->op == POLY_OP_PARAM && poly_uop_is_alu_param(u))
      count_local(
          u->dtype, false, &n_locals_i32, &n_locals_i64, &n_locals_f32, &n_locals_f64,
          &n_locals_v128
      );
    if (u->op == POLY_OP_RANGE) n_locals_i32++; /* loop counter always i32 */
    if (u->op == POLY_OP_LOAD) {
      PolyDType shrink_dtype;
      if (wasm_load_shrink_native_vec(u, &shrink_dtype, NULL)) {
        count_local(
            shrink_dtype, true, &n_locals_i32, &n_locals_i64, &n_locals_f32, &n_locals_f64,
            &n_locals_v128
        );
      } else if (lm_get(&reg_vectors.snapshots, u) >= 0 || !wasm_load_aliases_reg(u, &child_count)) {
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
      PolyDType base = poly_program_buffer_dtype(u);
      int reg_size = wasm_reg_storage_size(uops, n, u);
      bool vector = lm_get(&reg_vectors.buffers, u) > 0;
      if (vector) reg_size /= 4;
      for (int r = 0; r < reg_size; r++)
        count_local(
            base, vector, &n_locals_i32, &n_locals_i64, &n_locals_f32, &n_locals_f64, &n_locals_v128
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
  reg_vectors.gather_base = next_v128;
  next_v128 += reg_vectors.n_gathers;

  /* --- Second pass: emit instructions --- */
  for (int i = 0; i < n; i++) {
    /* Gathered SSA vector values survive stores, but not control-flow joins:
     * their first evaluation must dominate every reuse, including zero-trip loops.
     * Explicit locals avoid relying on the host engine to eliminate shuffles. */
    PolyOps op = uops[i]->op;
    if (op == POLY_OP_RANGE || op == POLY_OP_END || op == POLY_OP_IF || op == POLY_OP_ENDIF)
      reg_vectors.block++;
    /* Snapshot at the original LOAD, never at its later vector consumer.
     * Otherwise an intervening register STORE would change SSA semantics. */
    if (uops[i]->op == POLY_OP_LOAD && lm_get(&reg_vectors.snapshots, uops[i]) == i) {
      PolyUOp *base = NULL;
      int lane = 0;
      wasm_reg_addr_index(uops[i]->src[0], &base, &lane);
      int local = reg_vectors.snapshot_locals[i] = next_v128++;
      wb_byte(&body, WASM_OP_LOCAL_GET);
      wb_uleb128(&body, rlm_get(&reg_locals, base, lane));
      wb_byte(&body, WASM_OP_LOCAL_SET);
      wb_uleb128(&body, local);
    }
    if (skip && skip[i]) continue;
    PolyUOp *u = uops[i];

    /* GROUP orders effects; its STORE definitions already ran in LINEAR. */
    if (u->op == POLY_OP_SINK || u->op == POLY_OP_NOOP || u->op == POLY_OP_GROUP) continue;

    /* --- PARAM --- */
    if (u->op == POLY_OP_PARAM) {
      int abi = wasm_param_abi_index(uops, n, u);
      if (poly_uop_is_alu_param(u)) {
        int local =
            alloc_local(u->dtype, false, &next_i32, &next_i64, &next_f32, &next_f64, &next_v128);
        emit_scalar_param(&body, u, abi, local);
        lm_set(&locals, u, local);
      } else {
        lm_set(&locals, u, abi);
      }
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
      PolyDType base = poly_program_buffer_dtype(u);
      int reg_size = wasm_reg_storage_size(uops, n, u);
      bool vector = lm_get(&reg_vectors.buffers, u) > 0;
      int *reg_slots = malloc((size_t)reg_size * sizeof(int));
      if (!reg_slots) reg_size = 0;
      for (int r = 0; r < reg_size; r++) {
        int packed = vector ? wasm_reg_packed_lane(&reg_vectors, u, r) : r;
        if (vector && packed % 4) {
          int width = lm_get(&reg_vectors.buffers, u);
          reg_slots[r] = reg_slots[width > 1 ? r % (width / 4) : r - r % 4];
          continue;
        }
        int local_idx =
            alloc_local(base, vector, &next_i32, &next_i64, &next_f32, &next_f64, &next_v128);
        reg_slots[r] = local_idx;
        if (vector) {
          emit_v128_zero(&body);
        } else if (dt_is_f64(base)) {
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

    /* --- STACK: construct packed value from scalar lanes --- */
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
        emitted_fused =
            wasm_emit_structural_vec_expr(ctx, &body, &locals, u->src, u->n_src, &reg_vectors);
        if (!emitted_fused) {
          body.len = start;
          reg_vectors.block++;
        }
      }
      if (!emitted_fused && wasm_uop_value_is_v128(ctx, u) && u->n_src > 0) {
        emit_v128_zero(&body);
        for (int j = 0; j < u->n_src; j++) {
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

    /* --- LOAD: one gate/result contract for register, scalar and vector storage. --- */
    if (u->op == POLY_OP_LOAD) {
      PolyUOp *reg_base = NULL;
      int reg_lane = -1;
      bool is_reg = wasm_reg_addr_index(u->src[0], &reg_base, &reg_lane);
      PolyDType load_dtype = u->dtype;
      int lanes = 1;
      bool shrink_vec = wasm_load_shrink_native_vec(u, &load_dtype, &lanes);
      bool v128 = shrink_vec || wasm_uop_value_is_v128(ctx, u);
      int snapshot = lm_get(&reg_vectors.snapshots, u);
      if (snapshot < 0 && wasm_load_aliases_reg(u, &child_count)) {
        lm_set(&locals, u, rlm_get(&reg_locals, reg_base, reg_lane));
        continue;
      }

      int local_idx =
          alloc_local(load_dtype, v128, &next_i32, &next_i64, &next_f32, &next_f64, &next_v128);
      /* Pinned LOAD(address, alternate, gate) must not evaluate a masked-off
       * memory access. A select after an unconditional load can still trap. */
      bool gated = u->n_src == 3;
      if (gated) {
        static const uint8_t types[] = {
            WASM_TYPE_I32, WASM_TYPE_I64, WASM_TYPE_F32, WASM_TYPE_F64, WASM_TYPE_V128};
        wb_byte(&body, WASM_OP_LOCAL_GET);
        wb_uleb128(&body, lm_get(&locals, u->src[2]));
        wb_byte(&body, WASM_OP_IF);
        wb_byte(&body, types[dt_bucket(load_dtype, v128)]);
      }

      if (shrink_vec) {
        PolyUOp *shr = wasm_load_shrink(u);
        PolyUOp *base = wasm_acc_base(shr->src[0]);
        if (base && poly_program_memory_is(base, POLY_ADDR_REG)) {
          int offset = wasm_shrink_offset(shr);
          emit_v128_zero(&body);
          for (int lane = 0; lane < lanes; lane++) {
            wb_byte(&body, WASM_OP_LOCAL_GET);
            wb_uleb128(&body, rlm_get(&reg_locals, base, offset + lane));
            emit_cast_stack_value(&body, base->dtype, load_dtype);
            emit_v128_replace_lane(&body, load_dtype, lane);
          }
        } else {
          uint32_t offset = wasm_emit_load_address(
              &body, &locals, &addresses, &address_blocks,
              load_addresses ? load_addresses[i] : (WasmLoadAddress){0}, u->src[0],
              reg_vectors.block, &next_i32
          );
          emit_v128_load_offset(&body, load_dtype, lanes, offset);
        }
      } else if (is_reg) {
        wb_byte(&body, WASM_OP_LOCAL_GET);
        wb_uleb128(
            &body, snapshot >= 0 ? reg_vectors.snapshot_locals[snapshot]
                                 : rlm_get(&reg_locals, reg_base, reg_lane)
        );
        if (snapshot >= 0)
          emit_v128_extract_lane(
              &body, POLY_FLOAT32, wasm_reg_packed_lane(&reg_vectors, reg_base, reg_lane) % 4
          );
      } else {
        uint32_t offset = wasm_emit_load_address(
            &body, &locals, &addresses, &address_blocks,
            load_addresses ? load_addresses[i] : (WasmLoadAddress){0}, u->src[0], reg_vectors.block,
            &next_i32
        );
        if (v128)
          emit_v128_load_offset(&body, load_dtype, (int)poly_uop_max_numel(ctx, u), offset);
        else
          emit_scalar_load_offset(&body, load_dtype, offset);
      }

      if (gated) {
        wb_byte(&body, WASM_OP_ELSE);
        wb_byte(&body, WASM_OP_LOCAL_GET);
        wb_uleb128(&body, lm_get(&locals, u->src[1]));
        wb_byte(&body, WASM_OP_END);
      }
      wb_byte(&body, WASM_OP_LOCAL_SET);
      wb_uleb128(&body, local_idx);
      lm_set(&locals, u, local_idx);
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
        wb_byte(&body, WASM_OP_LOCAL_GET);
        wb_uleb128(&body, addr);
        wb_byte(&body, WASM_OP_LOCAL_GET);
        wb_uleb128(&body, val);

        PolyDType buf_dt = u->src[0]->dtype;
        /* Scalar destination-typed assignment, like CStyle. Use the same
         * signedness/width conversion as CAST; vector mismatch is rejected
         * before emission rather than silently reinterpreting v128 bits. */
        if (!val_is_v128) emit_cast_stack_value(&body, val_dt, buf_dt);
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
        emit_narrow_integer(&body, u->dtype);
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
          if (u->op == POLY_OP_CAST) emit_narrow_integer(&body, dst_dt);
        } else {
          emit_cast_stack_value(&body, src_dt, dst_dt);
          if (u->op == POLY_OP_CAST) emit_narrow_integer(&body, dst_dt);
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
        emit_cast_stack_value(&body, src_dt, dst_dt);
        emit_narrow_integer(&body, dst_dt);
      }

      wb_byte(&body, WASM_OP_LOCAL_SET);
      wb_uleb128(&body, local_idx);
      lm_set(&locals, u, local_idx);
      continue;
    }

    /* --- ALU --- */
    if (poly_opset_has(POLY_GROUP_ALU, u->op)) {
      if (packs && packs[i].owner) {
        WasmScalarPack *pack = &packs[packs[i].owner - 1];
        if (packs[i].owner == i + 1) {
          pack->local = next_v128++;
          for (int s = 0; s < wasm_scalar_pack_n_src(u); s++) {
            PolyUOp *operands[4];
            for (int k = 0; k < 4; k++)
              operands[k] = uops[pack->roots[k]]->src[s];
            wasm_emit_structural_vec_expr(ctx, &body, &locals, operands, 4, &reg_vectors);
          }
          emit_alu_simd_f32x4(&body, wasm_scalar_pack_op(u));
          wb_byte(&body, WASM_OP_LOCAL_SET);
          wb_uleb128(&body, pack->local);
        }
        int local = next_f32++;
        if (u->op == POLY_OP_MULACC) {
          wb_byte(&body, WASM_OP_LOCAL_GET);
          wb_uleb128(&body, lm_get(&locals, u->src[2]));
        }
        wb_byte(&body, WASM_OP_LOCAL_GET);
        wb_uleb128(&body, pack->local);
        emit_v128_extract_lane(&body, POLY_FLOAT32, packs[i].lane);
        if (u->op == POLY_OP_MULACC) wb_byte(&body, WASM_OP_F32_ADD);
        wb_byte(&body, WASM_OP_LOCAL_SET);
        wb_uleb128(&body, local);
        lm_set(&locals, u, local);
        continue;
      }
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
       * The condition may be f32 (e.g. from poly_uop_eq which returns a
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
  lm_destroy(&child_count);
  lm_destroy(&addresses);
  lm_destroy(&address_blocks);
  free(load_addresses);
  free(skip);
  free(fused_stack);
  free(packs);
  free(packed_uops);
  free(reg_vectors.snapshot_locals);
  lm_destroy(&reg_vectors.buffers);
  lm_destroy(&reg_vectors.snapshots);
  poly_map_destroy(reg_vectors.lanes);
  if (reg_vectors.lane_arena) poly_arena_destroy(reg_vectors.lane_arena);
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
    if (u->op == POLY_OP_PARAM && poly_uop_is_alu_param(u))
      count_local(
          u->dtype, false, &n_locals_i32, &n_locals_i64, &n_locals_f32, &n_locals_f64,
          &n_locals_v128
      );
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
      PolyUOp *u = uops[i];
      int abi = wasm_param_abi_index(uops, n, u);
      if (poly_uop_is_alu_param(u)) {
        int local =
            alloc_local(u->dtype, false, &next_i32, &next_i64, &next_f32, &next_f64, &next_v128);
        emit_scalar_param(&body, u, abi, local);
        lm_set(&locals, u, local);
      } else {
        lm_set(&locals, u, abi);
      }
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

/* Public API */

uint8_t *poly_render_wasm(PolyCtx *ctx, PolyUOp **uops, int n, int *size_out, bool use_simd) {
  if (size_out) *size_out = 0;
  /* REG arrays are represented by constant-indexed Wasm locals. Reject an
   * unsupported index before emission; it is never a linear-memory address. */
  for (int i = 0; i < n; i++) {
    PolyUOp *u = uops[i];
    if (u->op == POLY_OP_STORE && u->n_src >= 2 && poly_uop_max_numel(ctx, u->src[1]) > 1 &&
        !poly_dtype_eq(u->src[0]->dtype, u->src[1]->dtype)) {
      fprintf(
          stderr, "polygrad: vector STORE dtype mismatch; cast the value to the destination dtype\n"
      );
      return NULL;
    }
    if (u->op != POLY_OP_INDEX || u->n_src < 2) continue;
    PolyUOp *base = wasm_acc_base(u->src[0]);
    if (base && poly_program_memory_is(base, POLY_ADDR_REG) &&
        !wasm_reg_addr_index(u, NULL, NULL)) {
      fprintf(stderr, "[polygrad:wasm] unsupported nonconstant or negative REG index\n");
      return NULL;
    }
  }
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

  if (size_out) *size_out = mod.len;
  return mod.data; /* caller must free() */
}

PolyUOp *poly_rewrite_wasm(PolyCtx *ctx, PolyUOp *sink) {
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
