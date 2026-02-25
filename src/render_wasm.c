/*
 * render_wasm.c — WASM binary renderer for polygrad kernels
 *
 * Walks linearized UOps (same input as render_c.c) and emits a valid
 * WASM binary module. The generated module imports shared
 * WebAssembly.Memory and exports a single kernel function.
 *
 * Supports two modes:
 *   use_simd=false → scalar f32 ops (one float per iteration)
 *   use_simd=true  → f32x4 SIMD ops (4 floats per iteration) + scalar epilogue
 *
 * Reference: WebAssembly Binary Format Specification
 */

#include "codegen.h"
#include "wasm_builder.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── UOp → local index map (mirrors IntMap from render_c.c) ──────────── */

typedef struct {
  PolyUOp **keys;
  int *vals;
  int cap;
} LocalMap;

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
    if (m->keys[slot] == NULL) return -1;
  }
  return -1;
}

static void lm_destroy(LocalMap *m) {
  free(m->keys);
  free(m->vals);
}

/* ── Track which transcendentals are needed ──────────────────────────── */

typedef struct {
  bool need_exp2f;
  bool need_log2f;
  bool need_sinf;
  bool need_powf;
  bool need_recip; /* 1/x — not a WASM op, but can use f32.div */
} MathImports;

/* ── Pre-scan: determine imports and count params ────────────────────── */

static void prescan(PolyUOp **uops, int n, MathImports *math,
                    int *n_params_out, int *n_ranges_out) {
  memset(math, 0, sizeof(*math));
  int np = 0, nr = 0;
  for (int i = 0; i < n; i++) {
    PolyUOp *u = uops[i];
    if (u->op == POLY_OP_PARAM || u->op == POLY_OP_DEFINE_VAR) np++;
    if (u->op == POLY_OP_RANGE) nr++;
    if (u->op == POLY_OP_EXP2)  math->need_exp2f = true;
    if (u->op == POLY_OP_LOG2)  math->need_log2f = true;
    if (u->op == POLY_OP_SIN)   math->need_sinf = true;
    if (u->op == POLY_OP_POW)   math->need_powf = true;
  }
  *n_params_out = np;
  *n_ranges_out = nr;
}

/* ── Build type section ──────────────────────────────────────────────── */

static void build_type_section(WasmBuf *mod, int n_params, MathImports *math) {
  WasmBuf sec;
  wb_init(&sec);

  /* Count function types needed */
  int n_types = 1; /* kernel type */
  bool need_unary_math = math->need_exp2f || math->need_log2f || math->need_sinf;
  if (need_unary_math) n_types++;
  if (math->need_powf) n_types++;

  wb_uleb128(&sec, n_types);

  /* Type 0: kernel function — (i32, i32, ...) → () */
  wb_byte(&sec, WASM_TYPE_FUNC);
  wb_uleb128(&sec, n_params);   /* param count */
  for (int i = 0; i < n_params; i++)
    wb_byte(&sec, WASM_TYPE_I32);  /* all params are i32 byte offsets */
  wb_uleb128(&sec, 0);          /* no results */

  /* Type 1: unary math — (f32) → f32 */
  int math_type_idx = 1;
  if (need_unary_math) {
    wb_byte(&sec, WASM_TYPE_FUNC);
    wb_uleb128(&sec, 1);       /* 1 param */
    wb_byte(&sec, WASM_TYPE_F32);
    wb_uleb128(&sec, 1);       /* 1 result */
    wb_byte(&sec, WASM_TYPE_F32);
    math_type_idx = 1;
    (void)math_type_idx;
  }

  /* Type 2 (or 1 if no unary): binary math — (f32, f32) → f32 */
  if (math->need_powf) {
    wb_byte(&sec, WASM_TYPE_FUNC);
    wb_uleb128(&sec, 2);       /* 2 params */
    wb_byte(&sec, WASM_TYPE_F32);
    wb_byte(&sec, WASM_TYPE_F32);
    wb_uleb128(&sec, 1);       /* 1 result */
    wb_byte(&sec, WASM_TYPE_F32);
  }

  wb_section(mod, WASM_SEC_TYPE, &sec);
  wb_free(&sec);
}

/* ── Build import section ────────────────────────────────────────────── */

static int build_import_section(WasmBuf *mod, MathImports *math) {
  WasmBuf sec;
  wb_init(&sec);

  int n_imports = 1; /* memory */
  if (math->need_exp2f) n_imports++;
  if (math->need_log2f) n_imports++;
  if (math->need_sinf) n_imports++;
  if (math->need_powf) n_imports++;

  wb_uleb128(&sec, n_imports);

  /* Import 0: memory from "env" */
  wb_name(&sec, "env");
  wb_name(&sec, "memory");
  wb_byte(&sec, 0x02);        /* import kind: memory */
  wb_byte(&sec, 0x00);        /* limits: no max */
  wb_uleb128(&sec, 0);        /* initial: 0 pages */

  /* Math function imports — unary: type index 1 (f32→f32) */
  int func_idx = 0;
  bool need_unary = math->need_exp2f || math->need_log2f || math->need_sinf;
  if (math->need_exp2f) {
    wb_name(&sec, "math");
    wb_name(&sec, "exp2f");
    wb_byte(&sec, 0x00);      /* import kind: function */
    wb_uleb128(&sec, 1);      /* type index 1 */
    func_idx++;
  }
  if (math->need_log2f) {
    wb_name(&sec, "math");
    wb_name(&sec, "log2f");
    wb_byte(&sec, 0x00);
    wb_uleb128(&sec, 1);
    func_idx++;
  }
  if (math->need_sinf) {
    wb_name(&sec, "math");
    wb_name(&sec, "sinf");
    wb_byte(&sec, 0x00);
    wb_uleb128(&sec, 1);
    func_idx++;
  }
  /* Math function import — binary: type index 2 (or 1 if no unary) */
  if (math->need_powf) {
    int powf_type = need_unary ? 2 : 1;
    wb_name(&sec, "math");
    wb_name(&sec, "powf");
    wb_byte(&sec, 0x00);
    wb_uleb128(&sec, powf_type);
    func_idx++;
  }

  wb_section(mod, WASM_SEC_IMPORT, &sec);
  wb_free(&sec);

  return func_idx; /* number of imported functions (kernel func idx starts after) */
}

/* ── Build function section ──────────────────────────────────────────── */

static void build_function_section(WasmBuf *mod) {
  WasmBuf sec;
  wb_init(&sec);
  wb_uleb128(&sec, 1);     /* 1 function */
  wb_uleb128(&sec, 0);     /* type index 0 = kernel type */
  wb_section(mod, WASM_SEC_FUNCTION, &sec);
  wb_free(&sec);
}

/* ── Build export section ────────────────────────────────────────────── */

static void build_export_section(WasmBuf *mod, int kernel_func_idx) {
  WasmBuf sec;
  wb_init(&sec);
  wb_uleb128(&sec, 1);                        /* 1 export */
  wb_name(&sec, "kernel");                     /* export name */
  wb_byte(&sec, WASM_EXPORT_FUNC);            /* export kind */
  wb_uleb128(&sec, kernel_func_idx);           /* function index */
  wb_section(mod, WASM_SEC_EXPORT, &sec);
  wb_free(&sec);
}

/* ── Emit scalar ALU opcode ──────────────────────────────────────────── */

static void emit_alu_scalar(WasmBuf *code, PolyOps op, PolyDType dtype,
                            MathImports *math, int n_imported_funcs) {
  bool is_int = !poly_dtype_is_float(dtype);
  switch (op) {
  /* Unary */
  case POLY_OP_NEG:
    if (poly_dtype_is_bool(dtype)) {
      /* Boolean NOT: i32.eqz (0→1, non-zero→0) */
      wb_byte(code, WASM_OP_I32_EQZ);
    } else if (is_int) {
      /* Integer NEG: i32.const 0; src; i32.sub (caller pushes 0 first) */
      wb_byte(code, WASM_OP_I32_SUB);
    } else {
      wb_byte(code, WASM_OP_F32_NEG);
    }
    break;
  case POLY_OP_SQRT:       wb_byte(code, WASM_OP_F32_SQRT); break;
  case POLY_OP_TRUNC:      wb_byte(code, WASM_OP_F32_TRUNC); break;
  case POLY_OP_EXP2: {
    int idx = 0;
    (void)math;
    wb_byte(code, WASM_OP_CALL);
    wb_uleb128(code, idx);
    break;
  }
  case POLY_OP_LOG2: {
    int idx = math->need_exp2f ? 1 : 0;
    wb_byte(code, WASM_OP_CALL);
    wb_uleb128(code, idx);
    break;
  }
  case POLY_OP_SIN: {
    int idx = (math->need_exp2f ? 1 : 0) + (math->need_log2f ? 1 : 0);
    wb_byte(code, WASM_OP_CALL);
    wb_uleb128(code, idx);
    break;
  }
  case POLY_OP_RECIPROCAL: {
    wb_byte(code, WASM_OP_F32_DIV);
    break;
  }

  /* Binary — dtype-aware for ADD/SUB/MUL */
  case POLY_OP_ADD:   wb_byte(code, is_int ? WASM_OP_I32_ADD : WASM_OP_F32_ADD); break;
  case POLY_OP_SUB:   wb_byte(code, is_int ? WASM_OP_I32_SUB : WASM_OP_F32_SUB); break;
  case POLY_OP_MUL:   wb_byte(code, is_int ? WASM_OP_I32_MUL : WASM_OP_F32_MUL); break;
  case POLY_OP_FDIV:  wb_byte(code, WASM_OP_F32_DIV); break;
  case POLY_OP_MAX:   wb_byte(code, WASM_OP_F32_MAX); break;
  case POLY_OP_CMPLT: wb_byte(code, is_int ? WASM_OP_I32_LT_S : WASM_OP_F32_LT); break;
  case POLY_OP_CMPEQ: wb_byte(code, is_int ? WASM_OP_I32_EQ : WASM_OP_F32_EQ); break;
  case POLY_OP_CMPNE: wb_byte(code, is_int ? WASM_OP_I32_NE : WASM_OP_F32_NE); break;

  /* Integer-only ops */
  case POLY_OP_IDIV:  wb_byte(code, WASM_OP_I32_DIV_S); break;
  case POLY_OP_MOD:   wb_byte(code, WASM_OP_I32_REM_S); break;
  case POLY_OP_SHL:   wb_byte(code, WASM_OP_I32_SHL); break;
  case POLY_OP_SHR:   wb_byte(code, WASM_OP_I32_SHR_S); break;
  case POLY_OP_AND:   wb_byte(code, WASM_OP_I32_AND); break;
  case POLY_OP_OR:    wb_byte(code, WASM_OP_I32_OR); break;
  case POLY_OP_XOR:   wb_byte(code, WASM_OP_I32_XOR); break;

  /* Ternary */
  case POLY_OP_WHERE: wb_byte(code, WASM_OP_SELECT); break;
  case POLY_OP_MULACC:
    wb_byte(code, is_int ? WASM_OP_I32_MUL : WASM_OP_F32_MUL);
    wb_byte(code, is_int ? WASM_OP_I32_ADD : WASM_OP_F32_ADD);
    break;

  case POLY_OP_POW: {
    int idx = (math->need_exp2f ? 1 : 0) + (math->need_log2f ? 1 : 0) +
              (math->need_sinf ? 1 : 0);
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

/* ── Emit SIMD ALU opcode ────────────────────────────────────────────── */

static void emit_alu_simd(WasmBuf *code, PolyOps op) {
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
  default:
    /* Fallback: not all ops have SIMD equivalents.
     * For unsupported ops, the caller should use scalar path. */
    break;
  }
}

/* ── Check if an op has a SIMD equivalent ────────────────────────────── */

static bool has_simd_op(PolyOps op) {
  switch (op) {
  case POLY_OP_NEG: case POLY_OP_SQRT:
  case POLY_OP_ADD: case POLY_OP_SUB: case POLY_OP_MUL:
  case POLY_OP_FDIV: case POLY_OP_MAX:
    return true;
  default:
    return false;
  }
}

/* ── Check if entire kernel is SIMD-able ─────────────────────────────── */

static bool kernel_is_simdable(PolyUOp **uops, int n) {
  for (int i = 0; i < n; i++) {
    PolyUOp *u = uops[i];
    if (poly_opset_has(POLY_GROUP_ALU, u->op) && !has_simd_op(u->op))
      return false;
    /* Transcendentals don't have SIMD versions */
    if (u->op == POLY_OP_EXP2 || u->op == POLY_OP_LOG2 || u->op == POLY_OP_SIN ||
        u->op == POLY_OP_POW)
      return false;
    if (u->op == POLY_OP_RECIPROCAL)
      return false;
    if (u->op == POLY_OP_CAST || u->op == POLY_OP_BITCAST)
      return false;
    /* Reduce kernels are not SIMD-able (V1) */
    if (u->op == POLY_OP_DEFINE_LOCAL || u->op == POLY_OP_DEFINE_REG)
      return false;
  }
  return true;
}

/* ── Build code section (scalar) ─────────────────────────────────────── */

static void build_code_scalar(WasmBuf *mod, PolyUOp **uops, int n,
                              int n_params, MathImports *math,
                              int n_imported_funcs) {
  /* --- Count locals needed (beyond function params) --- */
  int n_locals_i32 = 0, n_locals_f32 = 0;

  /* First pass: count locals */
  for (int i = 0; i < n; i++) {
    PolyUOp *u = uops[i];
    if (u->op == POLY_OP_RANGE) n_locals_i32++;  /* loop counter */
    if (u->op == POLY_OP_LOAD) {
      /* Register LOADs alias the accumulator local — no new local needed */
      bool is_reg_load = (u->src[0]->op == POLY_OP_INDEX &&
                          u->src[0]->src[0]->dtype.is_ptr &&
                          u->src[0]->src[0]->dtype.addrspace == POLY_ADDR_REG);
      if (!is_reg_load) n_locals_f32++;
    }
    if (poly_opset_has(POLY_GROUP_ALU, u->op)) {
      if (poly_dtype_is_float(u->dtype)) n_locals_f32++;
      else n_locals_i32++;
    }
    if (u->op == POLY_OP_CAST || u->op == POLY_OP_BITCAST) {
      if (poly_dtype_is_float(u->dtype)) n_locals_f32++;
      else n_locals_i32++;
    }
    if (u->op == POLY_OP_DEFINE_LOCAL) {
      if (poly_dtype_is_float(u->dtype)) n_locals_f32++;
      else n_locals_i32++;
    }
    if (u->op == POLY_OP_DEFINE_REG) {
      /* Register accumulator: allocate a scalar local for the value */
      PolyDType base = poly_dtype_scalar(u->dtype);
      if (poly_dtype_is_float(base)) n_locals_f32++;
      else n_locals_i32++;
    }
    /* AFTER: no local needed (pass-through to src[0]) */
    if (u->op == POLY_OP_INDEX) {
      /* Register-based INDEX doesn't need a local (reuses accumulator) */
      if (!(u->src[0]->dtype.is_ptr && u->src[0]->dtype.addrspace == POLY_ADDR_REG))
        n_locals_i32++;
    }
    if (u->op == POLY_OP_CONST) {
      if (poly_dtype_is_float(u->dtype)) n_locals_f32++;
      else n_locals_i32++;
    }
  }

  /* Locals declaration: count of (count, type) pairs */
  int n_local_types = 0;
  if (n_locals_i32 > 0) n_local_types++;
  if (n_locals_f32 > 0) n_local_types++;

  /* Function body buffer (locals + instructions) */
  WasmBuf body;
  wb_init(&body);

  wb_uleb128(&body, n_local_types);
  if (n_locals_i32 > 0) {
    wb_uleb128(&body, n_locals_i32);
    wb_byte(&body, WASM_TYPE_I32);
  }
  if (n_locals_f32 > 0) {
    wb_uleb128(&body, n_locals_f32);
    wb_byte(&body, WASM_TYPE_F32);
  }

  /* --- Assign local indices --- */
  LocalMap locals;
  lm_init(&locals, n * 2);

  /* Params occupy local indices 0..n_params-1 */
  int next_i32 = n_params;  /* i32 locals start after params */
  int next_f32 = n_params + n_locals_i32;  /* f32 locals start after all i32 */

  /* --- Second pass: emit instructions --- */
  for (int i = 0; i < n; i++) {
    PolyUOp *u = uops[i];

    if (u->op == POLY_OP_SINK || u->op == POLY_OP_NOOP || u->op == POLY_OP_GROUP)
      continue;

    /* --- PARAM: already a function parameter --- */
    if (u->op == POLY_OP_PARAM || u->op == POLY_OP_DEFINE_VAR) {
      lm_set(&locals, u, (int)u->arg.i);
      continue;
    }

    /* --- CONST: push literal onto stack, store to local --- */
    if (u->op == POLY_OP_CONST) {
      int local_idx;
      if (poly_dtype_is_float(u->dtype)) {
        local_idx = next_f32++;
        wb_byte(&body, WASM_OP_F32_CONST);
        wb_f32(&body, (float)u->arg.f);
      } else {
        local_idx = next_i32++;
        wb_byte(&body, WASM_OP_I32_CONST);
        wb_sleb128(&body, u->arg.i);
      }
      wb_byte(&body, WASM_OP_LOCAL_SET);
      wb_uleb128(&body, local_idx);
      lm_set(&locals, u, local_idx);
      continue;
    }

    /* --- DEFINE_LOCAL: accumulator initialization --- */
    if (u->op == POLY_OP_DEFINE_LOCAL) {
      int local_idx;
      if (poly_dtype_is_float(u->dtype)) {
        local_idx = next_f32++;
        wb_byte(&body, WASM_OP_F32_CONST);
        float init = (u->arg.kind == POLY_ARG_FLOAT) ? (float)u->arg.f : 0.0f;
        wb_f32(&body, init);
      } else {
        local_idx = next_i32++;
        wb_byte(&body, WASM_OP_I32_CONST);
        int64_t init = (u->arg.kind == POLY_ARG_INT) ? u->arg.i : 0;
        wb_sleb128(&body, init);
      }
      wb_byte(&body, WASM_OP_LOCAL_SET);
      wb_uleb128(&body, local_idx);
      lm_set(&locals, u, local_idx);
      continue;
    }

    /* --- DEFINE_REG: register accumulator (like DEFINE_LOCAL) --- */
    if (u->op == POLY_OP_DEFINE_REG) {
      PolyDType base = poly_dtype_scalar(u->dtype);
      int local_idx;
      if (poly_dtype_is_float(base)) {
        local_idx = next_f32++;
        wb_byte(&body, WASM_OP_F32_CONST);
        wb_f32(&body, 0.0f);  /* will be initialized by STORE */
      } else {
        local_idx = next_i32++;
        wb_byte(&body, WASM_OP_I32_CONST);
        wb_sleb128(&body, 0);
      }
      wb_byte(&body, WASM_OP_LOCAL_SET);
      wb_uleb128(&body, local_idx);
      lm_set(&locals, u, local_idx);
      continue;
    }

    /* --- AFTER: pass-through (maps to same local as src[0]) --- */
    if (u->op == POLY_OP_AFTER) {
      int src_local = lm_get(&locals, u->src[0]);
      if (src_local >= 0)
        lm_set(&locals, u, src_local);
      continue;
    }

    /* --- INDEX: base + idx * 4 (byte offset), or register pass-through --- */
    if (u->op == POLY_OP_INDEX) {
      /* Register-based INDEX: just map to the accumulator local */
      if (u->src[0]->dtype.is_ptr &&
          u->src[0]->dtype.addrspace == POLY_ADDR_REG) {
        int acc_local = lm_get(&locals, u->src[0]);
        lm_set(&locals, u, acc_local);
      } else {
        int local_idx = next_i32++;
        int base = lm_get(&locals, u->src[0]);
        int idx  = lm_get(&locals, u->src[1]);

        wb_byte(&body, WASM_OP_LOCAL_GET);
        wb_uleb128(&body, base);
        wb_byte(&body, WASM_OP_LOCAL_GET);
        wb_uleb128(&body, idx);
        wb_byte(&body, WASM_OP_I32_CONST);
        wb_sleb128(&body, 4);  /* sizeof(float) */
        wb_byte(&body, WASM_OP_I32_MUL);
        wb_byte(&body, WASM_OP_I32_ADD);
        wb_byte(&body, WASM_OP_LOCAL_SET);
        wb_uleb128(&body, local_idx);
        lm_set(&locals, u, local_idx);
      }
      continue;
    }

    /* --- RANGE: for loop → block + loop + br_if --- */
    if (u->op == POLY_OP_RANGE) {
      int counter = next_i32++;
      lm_set(&locals, u, counter);

      /* Initialize counter to 0 */
      wb_byte(&body, WASM_OP_I32_CONST);
      wb_sleb128(&body, 0);
      wb_byte(&body, WASM_OP_LOCAL_SET);
      wb_uleb128(&body, counter);

      /* block $break */
      wb_byte(&body, WASM_OP_BLOCK);
      wb_byte(&body, WASM_BLOCKTYPE_VOID);

      /* loop $continue */
      wb_byte(&body, WASM_OP_LOOP);
      wb_byte(&body, WASM_BLOCKTYPE_VOID);

      /* if (counter >= bound) break */
      int bound = lm_get(&locals, u->src[0]);
      wb_byte(&body, WASM_OP_LOCAL_GET);
      wb_uleb128(&body, counter);
      wb_byte(&body, WASM_OP_LOCAL_GET);
      wb_uleb128(&body, bound);
      wb_byte(&body, WASM_OP_I32_GE_U);
      wb_byte(&body, WASM_OP_BR_IF);
      wb_uleb128(&body, 1);  /* branch to block (1 level out) */
      continue;
    }

    /* --- END: increment counter, branch to loop start, close --- */
    if (u->op == POLY_OP_END) {
      /* Find the RANGE this END closes (src[1] is the RANGE) */
      PolyUOp *range = NULL;
      for (int j = 0; j < u->n_src; j++) {
        if (u->src[j]->op == POLY_OP_RANGE) { range = u->src[j]; break; }
      }
      if (range) {
        int counter = lm_get(&locals, range);
        /* counter++ */
        wb_byte(&body, WASM_OP_LOCAL_GET);
        wb_uleb128(&body, counter);
        wb_byte(&body, WASM_OP_I32_CONST);
        wb_sleb128(&body, 1);
        wb_byte(&body, WASM_OP_I32_ADD);
        wb_byte(&body, WASM_OP_LOCAL_SET);
        wb_uleb128(&body, counter);

        /* br $continue (0 = innermost loop) */
        wb_byte(&body, WASM_OP_BR);
        wb_uleb128(&body, 0);
      }

      /* end loop, end block */
      wb_byte(&body, WASM_OP_END);
      wb_byte(&body, WASM_OP_END);
      continue;
    }

    /* --- LOAD: f32.load from memory, or local.get for register acc --- */
    if (u->op == POLY_OP_LOAD) {
      /* Check if loading from a register accumulator (INDEX → AFTER → DEFINE_REG) */
      bool is_reg = (u->src[0]->op == POLY_OP_INDEX &&
                     u->src[0]->src[0]->dtype.is_ptr &&
                     u->src[0]->src[0]->dtype.addrspace == POLY_ADDR_REG);
      if (is_reg) {
        /* Register load: just alias to the accumulator local */
        int acc_local = lm_get(&locals, u->src[0]);
        lm_set(&locals, u, acc_local);
      } else {
        int local_idx = next_f32++;
        int addr = lm_get(&locals, u->src[0]);

        wb_byte(&body, WASM_OP_LOCAL_GET);
        wb_uleb128(&body, addr);
        wb_byte(&body, WASM_OP_F32_LOAD);
        wb_uleb128(&body, 2);  /* align: 2^2 = 4 bytes */
        wb_uleb128(&body, 0);  /* offset: 0 */
        wb_byte(&body, WASM_OP_LOCAL_SET);
        wb_uleb128(&body, local_idx);
        lm_set(&locals, u, local_idx);
      }
      continue;
    }

    /* --- STORE: f32.store to memory or local.set to accumulator --- */
    if (u->op == POLY_OP_STORE) {
      int val = lm_get(&locals, u->src[1]);
      /* Check for register accumulator store (target is INDEX into reg) */
      bool is_reg = false;
      if (u->src[0]->op == POLY_OP_DEFINE_LOCAL) {
        is_reg = true;
      } else if (u->src[0]->op == POLY_OP_INDEX &&
                 u->src[0]->src[0]->dtype.is_ptr &&
                 u->src[0]->src[0]->dtype.addrspace == POLY_ADDR_REG) {
        is_reg = true;
      }
      if (is_reg) {
        /* Accumulator store: local.set */
        int acc_local = lm_get(&locals, u->src[0]);
        wb_byte(&body, WASM_OP_LOCAL_GET);
        wb_uleb128(&body, val);
        wb_byte(&body, WASM_OP_LOCAL_SET);
        wb_uleb128(&body, acc_local);
      } else {
        /* Buffer store: f32.store */
        int addr = lm_get(&locals, u->src[0]);
        wb_byte(&body, WASM_OP_LOCAL_GET);
        wb_uleb128(&body, addr);
        wb_byte(&body, WASM_OP_LOCAL_GET);
        wb_uleb128(&body, val);
        wb_byte(&body, WASM_OP_F32_STORE);
        wb_uleb128(&body, 2);  /* align */
        wb_uleb128(&body, 0);  /* offset */
      }
      continue;
    }

    /* --- CAST / BITCAST --- */
    if (u->op == POLY_OP_CAST || u->op == POLY_OP_BITCAST) {
      int local_idx;
      if (poly_dtype_is_float(u->dtype)) local_idx = next_f32++;
      else local_idx = next_i32++;

      int src = lm_get(&locals, u->src[0]);
      wb_byte(&body, WASM_OP_LOCAL_GET);
      wb_uleb128(&body, src);

      /* Determine conversion opcode */
      bool src_float = poly_dtype_is_float(u->src[0]->dtype);
      bool dst_float = poly_dtype_is_float(u->dtype);

      if (u->op == POLY_OP_BITCAST) {
        if (src_float && !dst_float)
          wb_byte(&body, WASM_OP_I32_REINTERPRET_F32);
        else if (!src_float && dst_float)
          wb_byte(&body, WASM_OP_F32_REINTERPRET_I32);
        /* same type: no-op */
      } else { /* CAST */
        if (src_float && !dst_float)
          wb_byte(&body, WASM_OP_I32_TRUNC_F32_S);
        else if (!src_float && dst_float)
          wb_byte(&body, WASM_OP_F32_CONVERT_I32_S);
        /* same type: no-op */
      }

      wb_byte(&body, WASM_OP_LOCAL_SET);
      wb_uleb128(&body, local_idx);
      lm_set(&locals, u, local_idx);
      continue;
    }

    /* --- ALU: push sources, emit op, store result --- */
    if (poly_opset_has(POLY_GROUP_ALU, u->op)) {
      int local_idx;
      if (poly_dtype_is_float(u->dtype)) local_idx = next_f32++;
      else local_idx = next_i32++;

      /* RECIPROCAL: push 1.0f first, then src, then div */
      if (u->op == POLY_OP_RECIPROCAL) {
        wb_byte(&body, WASM_OP_F32_CONST);
        float one = 1.0f;
        wb_f32(&body, one);
      }

      /* Integer NEG (non-bool): emit i32.const 0 before src, then i32.sub.
       * Boolean NEG uses i32.eqz (unary, no prefix needed). */
      if (u->op == POLY_OP_NEG && !poly_dtype_is_float(u->dtype) && !poly_dtype_is_bool(u->dtype)) {
        wb_byte(&body, WASM_OP_I32_CONST);
        wb_sleb128(&body, 0);
      }

      /* Push source operands.
       * Only emit actual operands (not dependency-only extra srcs).
       * Unary: 1, Binary: 2, Ternary: 3.
       * WHERE needs special order: WASM select expects (true, false, cond)
       * but polygrad has src[0]=cond, src[1]=true, src[2]=false. */
      if (u->op == POLY_OP_WHERE && u->n_src >= 3) {
        int s1 = lm_get(&locals, u->src[1]); /* true */
        int s2 = lm_get(&locals, u->src[2]); /* false */
        int s0 = lm_get(&locals, u->src[0]); /* cond */
        wb_byte(&body, WASM_OP_LOCAL_GET);
        wb_uleb128(&body, s1);
        wb_byte(&body, WASM_OP_LOCAL_GET);
        wb_uleb128(&body, s2);
        wb_byte(&body, WASM_OP_LOCAL_GET);
        wb_uleb128(&body, s0);
      } else {
        int n_operands = poly_opset_has(POLY_GROUP_TERNARY, u->op) ? 3
                       : poly_opset_has(POLY_GROUP_BINARY, u->op) ? 2
                       : 1; /* unary */
        if (n_operands > u->n_src) n_operands = u->n_src;
        for (int j = 0; j < n_operands; j++) {
          int src = lm_get(&locals, u->src[j]);
          wb_byte(&body, WASM_OP_LOCAL_GET);
          wb_uleb128(&body, src);
        }
      }

      /* For comparison ops, use the INPUT dtype to pick the right
       * instruction (e.g. f32.lt not i32.lt_s for float comparisons),
       * since the result dtype is always bool (i32). */
      PolyDType alu_dtype = u->dtype;
      if (u->op == POLY_OP_CMPLT || u->op == POLY_OP_CMPEQ ||
          u->op == POLY_OP_CMPNE) {
        alu_dtype = u->src[0]->dtype;
      }
      emit_alu_scalar(&body, u->op, alu_dtype, math, n_imported_funcs);

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
  wb_uleb128(&sec, 1);            /* 1 function body */
  wb_uleb128(&sec, body.len);     /* body size in bytes */
  wb_append(&sec, &body);
  wb_section(mod, WASM_SEC_CODE, &sec);

  wb_free(&body);
  wb_free(&sec);
  lm_destroy(&locals);
}

/* ── Build code section (SIMD) ───────────────────────────────────────── */

static void build_code_simd(WasmBuf *mod, PolyUOp **uops, int n,
                            int n_params, MathImports *math,
                            int n_imported_funcs) {
  /* SIMD codegen: split the innermost loop into:
   *   main loop:  i += 4, v128 ops
   *   epilogue:   i += 1, f32 ops (for remainder)
   *
   * For now, find the single RANGE/END pair and generate both loops.
   * If the kernel is not SIMD-able, fall back to scalar. */

  if (!kernel_is_simdable(uops, n)) {
    build_code_scalar(mod, uops, n, n_params, math, n_imported_funcs);
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
    build_code_scalar(mod, uops, n, n_params, math, n_imported_funcs);
    return;
  }

  /* --- Count locals --- */
  /* For SIMD we need: i32 counter, i32 bound, i32 simd_bound,
   * plus i32 for each INDEX (×2 for simd+scalar),
   * v128 for each LOAD and ALU (SIMD path),
   * f32 for each LOAD and ALU (scalar path).
   * This gets complex. For the initial implementation, allocate generously. */

  int n_locals_i32 = 0, n_locals_f32 = 0, n_locals_v128 = 0;

  /* Count ops that need locals */
  int n_indices = 0, n_loads = 0, n_alus = 0;
  for (int i = 0; i < n; i++) {
    PolyUOp *u = uops[i];
    if (u->op == POLY_OP_INDEX) n_indices++;
    if (u->op == POLY_OP_LOAD) n_loads++;
    if (poly_opset_has(POLY_GROUP_ALU, u->op)) n_alus++;
    if (u->op == POLY_OP_CONST && !poly_dtype_is_float(u->dtype)) n_locals_i32++;
    if (u->op == POLY_OP_CONST && poly_dtype_is_float(u->dtype)) n_locals_f32++;
  }

  /* SIMD loop needs: counter, bound const, simd_bound */
  n_locals_i32 += 3 + n_indices * 2; /* indices for both simd and scalar paths */
  n_locals_f32 += n_loads + n_alus;  /* scalar epilogue */
  n_locals_v128 += n_loads + n_alus; /* SIMD main loop */

  /* Function body */
  WasmBuf body;
  wb_init(&body);

  /* Declare locals */
  int n_local_types = 0;
  if (n_locals_i32 > 0) n_local_types++;
  if (n_locals_f32 > 0) n_local_types++;
  if (n_locals_v128 > 0) n_local_types++;

  wb_uleb128(&body, n_local_types);
  if (n_locals_i32 > 0) {
    wb_uleb128(&body, n_locals_i32);
    wb_byte(&body, WASM_TYPE_I32);
  }
  if (n_locals_f32 > 0) {
    wb_uleb128(&body, n_locals_f32);
    wb_byte(&body, WASM_TYPE_F32);
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
  int next_f32 = n_params + n_locals_i32;
  int next_v128 = n_params + n_locals_i32 + n_locals_f32;

  /* Assign PARAM locals */
  for (int i = 0; i < n; i++) {
    if (uops[i]->op == POLY_OP_PARAM || uops[i]->op == POLY_OP_DEFINE_VAR)
      lm_set(&locals, uops[i], (int)uops[i]->arg.i);
  }

  /* Handle CONST first (outside any loop) */
  for (int i = 0; i < n; i++) {
    PolyUOp *u = uops[i];
    if (u->op == POLY_OP_CONST) {
      int local_idx;
      if (poly_dtype_is_float(u->dtype)) {
        local_idx = next_f32++;
        wb_byte(&body, WASM_OP_F32_CONST);
        wb_f32(&body, (float)u->arg.f);
      } else {
        local_idx = next_i32++;
        wb_byte(&body, WASM_OP_I32_CONST);
        wb_sleb128(&body, u->arg.i);
      }
      wb_byte(&body, WASM_OP_LOCAL_SET);
      wb_uleb128(&body, local_idx);
      lm_set(&locals, u, local_idx);

      /* Track bound for the RANGE */
      if (u == range_uop->src[0])
        range_bound_local = local_idx;
    }
  }

  /* Counter local and SIMD bound local */
  int counter_local = next_i32++;
  int simd_bound_local = next_i32++;

  /* Compute simd_bound = bound & ~3 */
  wb_byte(&body, WASM_OP_LOCAL_GET);
  wb_uleb128(&body, range_bound_local);
  wb_byte(&body, WASM_OP_I32_CONST);
  wb_sleb128(&body, ~3);
  wb_byte(&body, WASM_OP_I32_AND);
  wb_byte(&body, WASM_OP_LOCAL_SET);
  wb_uleb128(&body, simd_bound_local);

  /* ═══ SIMD main loop: for (i = 0; i < simd_bound; i += 4) ═══ */
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

    if (u->op == POLY_OP_INDEX) {
      int local_idx = next_i32++;
      int base = lm_get(&locals, u->src[0]);
      int idx  = lm_get(&locals, u->src[1]);

      wb_byte(&body, WASM_OP_LOCAL_GET);
      wb_uleb128(&body, base);
      wb_byte(&body, WASM_OP_LOCAL_GET);
      wb_uleb128(&body, idx);
      wb_byte(&body, WASM_OP_I32_CONST);
      wb_sleb128(&body, 16); /* 4 floats × 4 bytes */
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
      wb_uleb128(&body, 2);  /* align: 4 bytes */
      wb_uleb128(&body, 0);  /* offset */
      wb_byte(&body, WASM_OP_LOCAL_SET);
      wb_uleb128(&body, local_idx);
      lm_set(&locals, u, local_idx);
    }

    if (poly_opset_has(POLY_GROUP_ALU, u->op)) {
      int local_idx = next_v128++;

      /* Push sources */
      for (int j = 0; j < u->n_src; j++) {
        int src = lm_get(&locals, u->src[j]);
        wb_byte(&body, WASM_OP_LOCAL_GET);
        wb_uleb128(&body, src);
      }

      emit_alu_simd(&body, u->op);

      wb_byte(&body, WASM_OP_LOCAL_SET);
      wb_uleb128(&body, local_idx);
      lm_set(&locals, u, local_idx);
    }

    if (u->op == POLY_OP_STORE) {
      int addr = lm_get(&locals, u->src[0]);
      int val  = lm_get(&locals, u->src[1]);

      wb_byte(&body, WASM_OP_LOCAL_GET);
      wb_uleb128(&body, addr);
      wb_byte(&body, WASM_OP_LOCAL_GET);
      wb_uleb128(&body, val);
      wb_byte(&body, WASM_SIMD_PREFIX);
      wb_uleb128(&body, WASM_SIMD_V128_STORE);
      wb_uleb128(&body, 2);  /* align */
      wb_uleb128(&body, 0);  /* offset */
    }
  }

  /* counter += 4 */
  wb_byte(&body, WASM_OP_LOCAL_GET);
  wb_uleb128(&body, counter_local);
  wb_byte(&body, WASM_OP_I32_CONST);
  wb_sleb128(&body, 4);
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

  /* Scalar body */
  for (int i = 0; i < n; i++) {
    PolyUOp *u = uops[i];

    if (u->op == POLY_OP_INDEX) {
      int local_idx = next_i32++;
      int base = lm_get(&locals, u->src[0]);

      wb_byte(&body, WASM_OP_LOCAL_GET);
      wb_uleb128(&body, base);
      wb_byte(&body, WASM_OP_LOCAL_GET);
      wb_uleb128(&body, counter_local); /* use counter directly */
      wb_byte(&body, WASM_OP_I32_CONST);
      wb_sleb128(&body, 4);
      wb_byte(&body, WASM_OP_I32_MUL);
      wb_byte(&body, WASM_OP_I32_ADD);
      wb_byte(&body, WASM_OP_LOCAL_SET);
      wb_uleb128(&body, local_idx);
      lm_set(&locals, u, local_idx);
    }

    if (u->op == POLY_OP_LOAD) {
      int local_idx = next_f32++;
      int addr = lm_get(&locals, u->src[0]);

      wb_byte(&body, WASM_OP_LOCAL_GET);
      wb_uleb128(&body, addr);
      wb_byte(&body, WASM_OP_F32_LOAD);
      wb_uleb128(&body, 2);
      wb_uleb128(&body, 0);
      wb_byte(&body, WASM_OP_LOCAL_SET);
      wb_uleb128(&body, local_idx);
      lm_set(&locals, u, local_idx);
    }

    if (poly_opset_has(POLY_GROUP_ALU, u->op)) {
      int local_idx = next_f32++;

      for (int j = 0; j < u->n_src; j++) {
        int src = lm_get(&locals, u->src[j]);
        wb_byte(&body, WASM_OP_LOCAL_GET);
        wb_uleb128(&body, src);
      }

      PolyDType alu_dtype = u->dtype;
      if (u->op == POLY_OP_CMPLT || u->op == POLY_OP_CMPEQ ||
          u->op == POLY_OP_CMPNE) {
        alu_dtype = u->src[0]->dtype;
      }
      emit_alu_scalar(&body, u->op, alu_dtype, math, n_imported_funcs);

      wb_byte(&body, WASM_OP_LOCAL_SET);
      wb_uleb128(&body, local_idx);
      lm_set(&locals, u, local_idx);
    }

    if (u->op == POLY_OP_STORE) {
      int addr = lm_get(&locals, u->src[0]);
      int val  = lm_get(&locals, u->src[1]);

      wb_byte(&body, WASM_OP_LOCAL_GET);
      wb_uleb128(&body, addr);
      wb_byte(&body, WASM_OP_LOCAL_GET);
      wb_uleb128(&body, val);
      wb_byte(&body, WASM_OP_F32_STORE);
      wb_uleb128(&body, 2);
      wb_uleb128(&body, 0);
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

/* ── Public API ──────────────────────────────────────────────────────── */

uint8_t *poly_render_wasm(PolyUOp **uops, int n, int *size_out, bool use_simd) {
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
    build_code_simd(&mod, uops, n, n_params, &math, n_imported_funcs);
  } else {
    build_code_scalar(&mod, uops, n, n_params, &math, n_imported_funcs);
  }

  *size_out = mod.len;
  return mod.data; /* caller must free() */
}
