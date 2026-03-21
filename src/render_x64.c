/*
 * render_x64.c — x86-64 JIT renderer and runtime for polygrad kernels
 *
 * Walks linearized UOps (same input as render_c.c / render_wasm.c) and emits
 * x86-64 machine code directly. Uses mmap+mprotect for executable memory.
 * No external compiler dependency (no fork, no clang, no dlopen).
 *
 * Architecture: stack-slot based (like WASM locals). Each UOp value gets a
 * stack slot at [rbp - offset]. Operations load from slots into scratch
 * registers, compute, store results back.
 *
 * Calling convention: System V AMD64 ABI. Generated function signature is
 * void fn(void** args) — args pointer arrives in RDI.
 */

#ifdef POLY_HAS_X64

#include "codegen.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <math.h>
#include <cpuid.h>

/* ══════════════════════════════════════════════════════════════════════ */
/*  CPU feature detection (Phase 4)                                      */
/* ══════════════════════════════════════════════════════════════════════ */

typedef struct {
  bool has_sse2;   /* always true on x86-64 */
  bool has_avx2;   /* cpuid leaf 7, ebx bit 5 */
  bool has_fma;    /* cpuid leaf 1, ecx bit 12 */
  bool os_avx_ok;  /* OSXSAVE + XGETBV confirms YMM state is saved */
} X64CpuCaps;

static X64CpuCaps g_cpu_caps;
static bool g_cpu_caps_detected = false;

static X64CpuCaps detect_cpu_caps(void) {
  X64CpuCaps caps = { .has_sse2 = true };
  unsigned int eax, ebx, ecx, edx;

  /* Leaf 1: check FMA + OSXSAVE */
  if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
    caps.has_fma = (ecx >> 12) & 1;
    bool osxsave = (ecx >> 27) & 1;

    /* Check OS support for YMM state (mandatory before any AVX codegen) */
    if (osxsave) {
      unsigned int xcr0_lo, xcr0_hi;
      __asm__ volatile("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));
      caps.os_avx_ok = (xcr0_lo & 0x6) == 0x6; /* bits 1 (SSE) + 2 (AVX) */
    }
  }

  /* Leaf 7: check AVX2 */
  if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
    caps.has_avx2 = (ebx >> 5) & 1;
  }

  /* Gate AVX2/FMA on OS support */
  if (!caps.os_avx_ok) {
    caps.has_avx2 = false;
    caps.has_fma = false;
  }
  return caps;
}

static X64CpuCaps get_cpu_caps(void) {
  if (!g_cpu_caps_detected) {
    g_cpu_caps = detect_cpu_caps();
    g_cpu_caps_detected = true;
  }
  return g_cpu_caps;
}
#include <dlfcn.h>

/* ══════════════════════════════════════════════════════════════════════ */
/*  Growable byte buffer                                                 */
/* ══════════════════════════════════════════════════════════════════════ */

typedef struct {
  uint8_t *data;
  int len, cap;
} X64Buf;

static void xb_grow(X64Buf *b, int need) {
  if (b->len + need <= b->cap) return;
  int nc = b->cap < 256 ? 256 : b->cap;
  while (nc < b->len + need) nc *= 2;
  b->data = realloc(b->data, nc);
  b->cap = nc;
}

static void xb_byte(X64Buf *b, uint8_t v) {
  xb_grow(b, 1);
  b->data[b->len++] = v;
}

static void xb_i32(X64Buf *b, int32_t v) {
  xb_grow(b, 4);
  memcpy(b->data + b->len, &v, 4);
  b->len += 4;
}

static void xb_i64(X64Buf *b, int64_t v) {
  xb_grow(b, 8);
  memcpy(b->data + b->len, &v, 8);
  b->len += 8;
}

/* ══════════════════════════════════════════════════════════════════════ */
/*  UOp → stack slot map (same pattern as render_wasm.c LocalMap)        */
/* ══════════════════════════════════════════════════════════════════════ */

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

/* ══════════════════════════════════════════════════════════════════════ */
/*  x86-64 register encoding constants                                   */
/* ══════════════════════════════════════════════════════════════════════ */

enum {
  RAX = 0, RCX = 1, RDX = 2, RBX = 3,
  RSP = 4, RBP = 5, RSI = 6, RDI = 7,
  R8 = 8, R9 = 9, R10 = 10, R11 = 11,
  R12 = 12, R13 = 13, R14 = 14, R15 = 15,
};

enum {
  XMM0 = 0, XMM1 = 1, XMM2 = 2, XMM3 = 3,
};

/* Stack slot base offset from RBP.
 * After prologue: push rbp, r15, r14, r13, r12, rbx = 6 callee saves.
 * Slots start at [rbp - SLOT_BASE - 8*slot_index]. */
#define SLOT_BASE 48
/* Each slot is 32 bytes (AVX2-aligned). Scalar values use 4-8 bytes,
 * SSE packed uses 16 bytes, AVX2 packed uses all 32 bytes.
 * 32-byte slots ensure YMM spills never overlap adjacent values. */
#define SLOT_BYTES 32

static int slot_offset(int slot_idx) {
  return SLOT_BASE + SLOT_BYTES * slot_idx;
}

/* ── Vector configuration: parameterizes emission for SSE vs AVX2 ──── */
typedef struct {
  int width;       /* elements per vector: 4 (SSE) or 8 (AVX2) */
  int reg_bits;    /* 128 (XMM) or 256 (YMM) */
  bool use_vex;    /* true for VEX-encoded instructions (AVX2) */
} VecConfig;

/* SSE configuration (current default) */
static const VecConfig VCFG_SSE = { .width = 4, .reg_bits = 128, .use_vex = false };
/* AVX2 configuration */
static const VecConfig VCFG_AVX2 = { .width = 8, .reg_bits = 256, .use_vex = true };

/* ══════════════════════════════════════════════════════════════════════ */
/*  x86-64 instruction encoding helpers                                  */
/* ══════════════════════════════════════════════════════════════════════ */

/* ModR/M byte: [mod(2) | reg(3) | r/m(3)] */
static void emit_modrm(X64Buf *b, int mod, int reg, int rm) {
  xb_byte(b, (uint8_t)((mod << 6) | ((reg & 7) << 3) | (rm & 7)));
}

/* REX prefix: 0100WRXB */
static void emit_rex(X64Buf *b, int w, int r, int x, int bv) {
  uint8_t rex = 0x40 | (w << 3) | (r << 2) | (x << 1) | bv;
  if (rex != 0x40) xb_byte(b, rex);
}

/* Always emit REX (even when 0x40, needed for some instructions) */
static void emit_rex_always(X64Buf *b, int w, int r, int x, int bv) {
  xb_byte(b, (uint8_t)(0x40 | (w << 3) | (r << 2) | (x << 1) | bv));
}

/* Emit ModR/M for [rbp - disp32] addressing.
 * RBP as base requires mod!=00 (mod=00 rm=101 means RIP-relative).
 * Always use mod=10 (disp32) for simplicity. */
static void emit_modrm_rbp_disp32(X64Buf *b, int reg, int32_t disp) {
  emit_modrm(b, 2, reg, RBP); /* mod=10, rm=101(rbp) */
  xb_i32(b, disp);
}

/* mov r64, [rbp + disp32] — REX.W 8B /r */
static void emit_mov_r64_rbp(X64Buf *b, int reg, int disp) {
  emit_rex_always(b, 1, reg >> 3, 0, 0); /* REX.W, R if reg>=8 */
  xb_byte(b, 0x8B);
  emit_modrm_rbp_disp32(b, reg, disp);
}

/* mov [rbp + disp32], r64 — REX.W 89 /r */
static void emit_mov_rbp_r64(X64Buf *b, int reg, int disp) {
  emit_rex_always(b, 1, reg >> 3, 0, 0);
  xb_byte(b, 0x89);
  emit_modrm_rbp_disp32(b, reg, disp);
}

/* mov r32, [rbp + disp32] — 8B /r (no REX.W for 32-bit) */
static void emit_mov_r32_rbp(X64Buf *b, int reg, int disp) {
  if (reg >= 8) emit_rex(b, 0, reg >> 3, 0, 0);
  xb_byte(b, 0x8B);
  emit_modrm_rbp_disp32(b, reg, disp);
}

/* mov [rbp + disp32], r32 — 89 /r */
static void emit_mov_rbp_r32(X64Buf *b, int reg, int disp) {
  if (reg >= 8) emit_rex(b, 0, reg >> 3, 0, 0);
  xb_byte(b, 0x89);
  emit_modrm_rbp_disp32(b, reg, disp);
}

/* mov r32, imm32 — B8+rd id */
static void emit_mov_r32_imm32(X64Buf *b, int reg, int32_t imm) {
  if (reg >= 8) emit_rex(b, 0, 0, 0, 1);
  xb_byte(b, 0xB8 + (reg & 7));
  xb_i32(b, imm);
}

/* mov r64, imm64 — REX.W B8+rd io */
static void emit_mov_r64_imm64(X64Buf *b, int reg, int64_t imm) {
  emit_rex_always(b, 1, 0, 0, reg >> 3);
  xb_byte(b, 0xB8 + (reg & 7));
  xb_i64(b, imm);
}

/* mov dword [rbp + disp32], imm32 — C7 /0 */
static void emit_mov_rbp_imm32(X64Buf *b, int disp, int32_t imm) {
  xb_byte(b, 0xC7);
  emit_modrm_rbp_disp32(b, 0, disp);
  xb_i32(b, imm);
}

/* mov qword [rbp + disp32], imm32 (sign-extended) — REX.W C7 /0 */
static void emit_mov_rbp_imm64_sx(X64Buf *b, int disp, int32_t imm) {
  emit_rex_always(b, 1, 0, 0, 0);
  xb_byte(b, 0xC7);
  emit_modrm_rbp_disp32(b, 0, disp);
  xb_i32(b, imm);
}

/* mov r64, [r64 + disp32] — REX.W 8B /r with base register */
static void emit_mov_r64_mem(X64Buf *b, int dst, int base, int32_t disp) {
  emit_rex_always(b, 1, dst >> 3, 0, base >> 3);
  xb_byte(b, 0x8B);
  if (disp == 0 && (base & 7) != RBP) {
    emit_modrm(b, 0, dst, base);
    if ((base & 7) == RSP) xb_byte(b, 0x24); /* SIB for RSP */
  } else if (disp >= -128 && disp <= 127) {
    emit_modrm(b, 1, dst, base);
    if ((base & 7) == RSP) xb_byte(b, 0x24);
    xb_byte(b, (uint8_t)(int8_t)disp);
  } else {
    emit_modrm(b, 2, dst, base);
    if ((base & 7) == RSP) xb_byte(b, 0x24);
    xb_i32(b, disp);
  }
}

/* mov r32, [r64] — 8B /r with mod=00 */
static void emit_mov_r32_mem_base(X64Buf *b, int dst, int base) {
  if (dst >= 8 || base >= 8) emit_rex(b, 0, dst >> 3, 0, base >> 3);
  xb_byte(b, 0x8B);
  emit_modrm(b, 0, dst, base);
  if ((base & 7) == RSP) xb_byte(b, 0x24);
  if ((base & 7) == RBP) { /* mod=00 rm=101 is RIP-relative, need disp8=0 */
    /* re-encode with mod=01 disp8=0 */
    b->len -= 1; /* back up modrm */
    if ((base & 7) == RSP) b->len -= 1;
    emit_modrm(b, 1, dst, base);
    if ((base & 7) == RSP) xb_byte(b, 0x24);
    xb_byte(b, 0);
  }
}

/* movsxd r64, r32 — REX.W 63 /r (mod=11) */
static void emit_movsxd(X64Buf *b, int dst, int src) {
  emit_rex_always(b, 1, dst >> 3, 0, src >> 3);
  xb_byte(b, 0x63);
  emit_modrm(b, 3, dst, src);
}

/* lea r64, [base + index*scale] — REX.W 8D /r with SIB */
static void emit_lea_sib(X64Buf *b, int dst, int base, int index, int scale) {
  int ss;
  switch (scale) {
    case 1: ss = 0; break;
    case 2: ss = 1; break;
    case 4: ss = 2; break;
    case 8: ss = 3; break;
    default: ss = 0; break; /* shouldn't reach here */
  }
  emit_rex_always(b, 1, dst >> 3, index >> 3, base >> 3);
  xb_byte(b, 0x8D);
  /* mod=00 rm=100(SIB) — but if base is RBP, need mod=01 disp8=0 */
  if ((base & 7) == RBP) {
    emit_modrm(b, 1, dst, 4); /* rm=100 = SIB */
    xb_byte(b, (uint8_t)((ss << 6) | ((index & 7) << 3) | (base & 7)));
    xb_byte(b, 0); /* disp8 = 0 */
  } else {
    emit_modrm(b, 0, dst, 4); /* rm=100 = SIB */
    xb_byte(b, (uint8_t)((ss << 6) | ((index & 7) << 3) | (base & 7)));
  }
}

/* Generic integer ALU: op r64, [rbp+disp] or op r32, [rbp+disp]
 * opcode is the primary opcode byte. w=1 for 64-bit. */
static void emit_alu_r_rbp(X64Buf *b, int w, uint8_t opcode, int reg, int disp) {
  emit_rex_always(b, w, reg >> 3, 0, 0);
  xb_byte(b, opcode);
  emit_modrm_rbp_disp32(b, reg, disp);
}

/* Generic integer ALU: op r, r (mod=11) */
static void emit_alu_rr(X64Buf *b, int w, uint8_t opcode, int dst, int src) {
  emit_rex_always(b, w, dst >> 3, 0, src >> 3);
  xb_byte(b, opcode);
  emit_modrm(b, 3, dst, src);
}

/* push r64 */
static void emit_push(X64Buf *b, int reg) {
  if (reg >= 8) emit_rex(b, 0, 0, 0, 1);
  xb_byte(b, 0x50 + (reg & 7));
}

/* pop r64 */
static void emit_pop(X64Buf *b, int reg) {
  if (reg >= 8) emit_rex(b, 0, 0, 0, 1);
  xb_byte(b, 0x58 + (reg & 7));
}

/* sub rsp, imm32 — REX.W 81 /5 id */
static void emit_sub_rsp_imm32(X64Buf *b, int32_t imm) {
  emit_rex_always(b, 1, 0, 0, 0);
  xb_byte(b, 0x81);
  emit_modrm(b, 3, 5, RSP);
  xb_i32(b, imm);
}

/* add rsp, imm32 — REX.W 81 /0 id */
static void emit_add_rsp_imm32(X64Buf *b, int32_t imm) {
  emit_rex_always(b, 1, 0, 0, 0);
  xb_byte(b, 0x81);
  emit_modrm(b, 3, 0, RSP);
  xb_i32(b, imm);
}

/* add r64, imm32 — REX.W 81 /0 id */
static void emit_add_r64_imm32(X64Buf *b, int reg, int32_t imm) {
  emit_rex_always(b, 1, 0, 0, reg >> 3);
  xb_byte(b, 0x81);
  emit_modrm(b, 3, 0, reg);
  xb_i32(b, imm);
}

/* ret */
static void emit_ret(X64Buf *b) { xb_byte(b, 0xC3); }

/* ── SSE scalar instructions ───────────────────────────────────────── */

/* movss xmm, [rbp + disp32] — F3 0F 10 /r */
static void emit_movss_xmm_rbp(X64Buf *b, int xmm, int disp) {
  xb_byte(b, 0xF3);
  if (xmm >= 8) emit_rex(b, 0, xmm >> 3, 0, 0);
  xb_byte(b, 0x0F);
  xb_byte(b, 0x10);
  emit_modrm_rbp_disp32(b, xmm, disp);
}

/* movss [rbp + disp32], xmm — F3 0F 11 /r */
static void emit_movss_rbp_xmm(X64Buf *b, int xmm, int disp) {
  xb_byte(b, 0xF3);
  if (xmm >= 8) emit_rex(b, 0, xmm >> 3, 0, 0);
  xb_byte(b, 0x0F);
  xb_byte(b, 0x11);
  emit_modrm_rbp_disp32(b, xmm, disp);
}

/* movss xmm, [r64] — F3 0F 10 /r with base register */
static void emit_movss_xmm_mem(X64Buf *b, int xmm, int base) {
  xb_byte(b, 0xF3);
  if (xmm >= 8 || base >= 8) emit_rex(b, 0, xmm >> 3, 0, base >> 3);
  xb_byte(b, 0x0F);
  xb_byte(b, 0x10);
  if ((base & 7) == RBP) {
    emit_modrm(b, 1, xmm, base); /* disp8=0 for RBP */
    xb_byte(b, 0);
  } else {
    emit_modrm(b, 0, xmm, base);
    if ((base & 7) == RSP) xb_byte(b, 0x24);
  }
}

/* movups xmm, [r64] — 0F 10 /r with base register (packed 128-bit load) */
static void emit_movups_xmm_mem(X64Buf *b, int xmm, int base) {
  if (xmm >= 8 || base >= 8) emit_rex(b, 0, xmm >> 3, 0, base >> 3);
  xb_byte(b, 0x0F); xb_byte(b, 0x10);
  if ((base & 7) == RBP) { emit_modrm(b, 1, xmm, base); xb_byte(b, 0); }
  else { emit_modrm(b, 0, xmm, base); if ((base & 7) == RSP) xb_byte(b, 0x24); }
}

/* movups [r64], xmm — 0F 11 /r (packed 128-bit store) */
static void emit_movups_mem_xmm(X64Buf *b, int xmm, int base) {
  if (xmm >= 8 || base >= 8) emit_rex(b, 0, xmm >> 3, 0, base >> 3);
  xb_byte(b, 0x0F); xb_byte(b, 0x11);
  if ((base & 7) == RBP) { emit_modrm(b, 1, xmm, base); xb_byte(b, 0); }
  else { emit_modrm(b, 0, xmm, base); if ((base & 7) == RSP) xb_byte(b, 0x24); }
}

/* movss [r64], xmm — F3 0F 11 /r */
static void emit_movss_mem_xmm(X64Buf *b, int xmm, int base) {
  xb_byte(b, 0xF3);
  if (xmm >= 8 || base >= 8) emit_rex(b, 0, xmm >> 3, 0, base >> 3);
  xb_byte(b, 0x0F);
  xb_byte(b, 0x11);
  if ((base & 7) == RBP) {
    emit_modrm(b, 1, xmm, base);
    xb_byte(b, 0);
  } else {
    emit_modrm(b, 0, xmm, base);
    if ((base & 7) == RSP) xb_byte(b, 0x24);
  }
}

/* movss xmm, xmm — F3 0F 10 /r (mod=11) */
static void emit_movss_xmm_xmm(X64Buf *b, int dst, int src) {
  xb_byte(b, 0xF3);
  if (dst >= 8 || src >= 8) emit_rex(b, 0, dst >> 3, 0, src >> 3);
  xb_byte(b, 0x0F);
  xb_byte(b, 0x10);
  emit_modrm(b, 3, dst, src);
}

/* Generic SSE scalar op: F3 0F <opcode> xmm, [rbp+disp] */
static void emit_sse_scalar_rbp(X64Buf *b, uint8_t opcode, int xmm, int disp) {
  xb_byte(b, 0xF3);
  if (xmm >= 8) emit_rex(b, 0, xmm >> 3, 0, 0);
  xb_byte(b, 0x0F);
  xb_byte(b, opcode);
  emit_modrm_rbp_disp32(b, xmm, disp);
}

/* Generic SSE scalar op: F3 0F <opcode> xmm, xmm (mod=11) */
static void emit_sse_scalar_rr(X64Buf *b, uint8_t opcode, int dst, int src) {
  xb_byte(b, 0xF3);
  if (dst >= 8 || src >= 8) emit_rex(b, 0, dst >> 3, 0, src >> 3);
  xb_byte(b, 0x0F);
  xb_byte(b, opcode);
  emit_modrm(b, 3, dst, src);
}

/* movd xmm, r32 — 66 0F 6E /r */
static void emit_movd_xmm_r32(X64Buf *b, int xmm, int r32) {
  xb_byte(b, 0x66);
  if (xmm >= 8 || r32 >= 8) emit_rex(b, 0, xmm >> 3, 0, r32 >> 3);
  xb_byte(b, 0x0F);
  xb_byte(b, 0x6E);
  emit_modrm(b, 3, xmm, r32);
}

/* movd r32, xmm — 66 0F 7E /r */
static void emit_movd_r32_xmm(X64Buf *b, int r32, int xmm) {
  xb_byte(b, 0x66);
  if (xmm >= 8 || r32 >= 8) emit_rex(b, 0, xmm >> 3, 0, r32 >> 3);
  xb_byte(b, 0x0F);
  xb_byte(b, 0x7E);
  emit_modrm(b, 3, xmm, r32);
}

/* xorps xmm, xmm — 0F 57 /r */
static void emit_xorps(X64Buf *b, int dst, int src) {
  if (dst >= 8 || src >= 8) emit_rex(b, 0, dst >> 3, 0, src >> 3);
  xb_byte(b, 0x0F);
  xb_byte(b, 0x57);
  emit_modrm(b, 3, dst, src);
}

/* andps xmm, xmm — 0F 54 /r */
static void emit_andps(X64Buf *b, int dst, int src) {
  if (dst >= 8 || src >= 8) emit_rex(b, 0, dst >> 3, 0, src >> 3);
  xb_byte(b, 0x0F);
  xb_byte(b, 0x54);
  emit_modrm(b, 3, dst, src);
}

/* andnps xmm, xmm — 0F 55 /r */
static void emit_andnps(X64Buf *b, int dst, int src) {
  if (dst >= 8 || src >= 8) emit_rex(b, 0, dst >> 3, 0, src >> 3);
  xb_byte(b, 0x0F);
  xb_byte(b, 0x55);
  emit_modrm(b, 3, dst, src);
}

/* orps xmm, xmm — 0F 56 /r */
static void emit_orps(X64Buf *b, int dst, int src) {
  if (dst >= 8 || src >= 8) emit_rex(b, 0, dst >> 3, 0, src >> 3);
  xb_byte(b, 0x0F);
  xb_byte(b, 0x56);
  emit_modrm(b, 3, dst, src);
}

/* cmpss xmm, xmm, imm8 — F3 0F C2 /r ib */
static void emit_cmpss_rr(X64Buf *b, int dst, int src, uint8_t pred) {
  xb_byte(b, 0xF3);
  if (dst >= 8 || src >= 8) emit_rex(b, 0, dst >> 3, 0, src >> 3);
  xb_byte(b, 0x0F);
  xb_byte(b, 0xC2);
  emit_modrm(b, 3, dst, src);
  xb_byte(b, pred);
}

/* cvtsi2ss xmm, r32 — F3 0F 2A /r */
static void emit_cvtsi2ss(X64Buf *b, int xmm, int r32) {
  xb_byte(b, 0xF3);
  if (xmm >= 8 || r32 >= 8) emit_rex(b, 0, xmm >> 3, 0, r32 >> 3);
  xb_byte(b, 0x0F);
  xb_byte(b, 0x2A);
  emit_modrm(b, 3, xmm, r32);
}

/* cvttss2si r32, xmm — F3 0F 2C /r */
static void emit_cvttss2si(X64Buf *b, int r32, int xmm) {
  xb_byte(b, 0xF3);
  if (r32 >= 8 || xmm >= 8) emit_rex(b, 0, r32 >> 3, 0, xmm >> 3);
  xb_byte(b, 0x0F);
  xb_byte(b, 0x2C);
  emit_modrm(b, 3, r32, xmm);
}

/* ── Jump instructions ─────────────────────────────────────────────── */

/* jmp rel32 — E9 cd (returns offset of rel32 for patching) */
static int emit_jmp_rel32(X64Buf *b, int32_t rel) {
  xb_byte(b, 0xE9);
  int off = b->len;
  xb_i32(b, rel);
  return off;
}

/* jge rel32 — 0F 8D cd (returns offset of rel32 for patching) */
static int emit_jge_rel32(X64Buf *b, int32_t rel) {
  xb_byte(b, 0x0F);
  xb_byte(b, 0x8D);
  int off = b->len;
  xb_i32(b, rel);
  return off;
}

/* je rel32 — 0F 84 cd (returns offset of rel32 for patching) */
static int emit_je_rel32(X64Buf *b, int32_t rel) {
  xb_byte(b, 0x0F);
  xb_byte(b, 0x84);
  int off = b->len;
  xb_i32(b, rel);
  return off;
}

/* test r32, r32 — 85 /r */
static void emit_test_r32(X64Buf *b, int r1, int r2) {
  if (r1 >= 8 || r2 >= 8) emit_rex(b, 0, r1 >> 3, 0, r2 >> 3);
  xb_byte(b, 0x85);
  emit_modrm(b, 3, r1, r2);
}

/* cmp r64, [rbp+disp] — REX.W 3B /r */
static void emit_cmp_r64_rbp(X64Buf *b, int reg, int disp) {
  emit_alu_r_rbp(b, 1, 0x3B, reg, disp);
}

/* neg r32 — F7 /3 */
static void emit_neg_r32(X64Buf *b, int reg) {
  if (reg >= 8) emit_rex(b, 0, 0, 0, reg >> 3);
  xb_byte(b, 0xF7);
  emit_modrm(b, 3, 3, reg);
}

/* neg r64 — REX.W F7 /3 */
static void emit_neg_r64(X64Buf *b, int reg) {
  emit_rex_always(b, 1, 0, 0, reg >> 3);
  xb_byte(b, 0xF7);
  emit_modrm(b, 3, 3, reg);
}

/* cdq — sign-extend EAX into EDX:EAX */
static void emit_cdq(X64Buf *b) { xb_byte(b, 0x99); }

/* cqo — sign-extend RAX into RDX:RAX */
static void emit_cqo(X64Buf *b) { emit_rex_always(b, 1, 0, 0, 0); xb_byte(b, 0x99); }

/* idiv r/m32 — F7 /7 */
static void emit_idiv_r32(X64Buf *b, int reg) {
  if (reg >= 8) emit_rex(b, 0, 0, 0, reg >> 3);
  xb_byte(b, 0xF7);
  emit_modrm(b, 3, 7, reg);
}

/* idiv r/m64 — REX.W F7 /7 */
static void emit_idiv_r64(X64Buf *b, int reg) {
  emit_rex_always(b, 1, 0, 0, reg >> 3);
  xb_byte(b, 0xF7);
  emit_modrm(b, 3, 7, reg);
}

/* imul r64, r64 — REX.W 0F AF /r */
static void emit_imul_r64_r64(X64Buf *b, int dst, int src) {
  emit_rex_always(b, 1, dst >> 3, 0, src >> 3);
  xb_byte(b, 0x0F);
  xb_byte(b, 0xAF);
  emit_modrm(b, 3, dst, src);
}

/* imul r32, r32 — 0F AF /r */
static void emit_imul_r32_r32(X64Buf *b, int dst, int src) {
  if (dst >= 8 || src >= 8) emit_rex(b, 0, dst >> 3, 0, src >> 3);
  xb_byte(b, 0x0F);
  xb_byte(b, 0xAF);
  emit_modrm(b, 3, dst, src);
}

/* imul r64, imm32 — REX.W 69 /r id */
static void emit_imul_r64_imm32(X64Buf *b, int dst, int src, int32_t imm) {
  emit_rex_always(b, 1, dst >> 3, 0, src >> 3);
  xb_byte(b, 0x69);
  emit_modrm(b, 3, dst, src);
  xb_i32(b, imm);
}

/* setcc r8 — 0F 9x /0 (reg=0 means r/m field) */
static void emit_setcc(X64Buf *b, uint8_t cc, int reg) {
  if (reg >= 8) emit_rex(b, 0, 0, 0, reg >> 3);
  xb_byte(b, 0x0F);
  xb_byte(b, 0x90 + cc);
  emit_modrm(b, 3, 0, reg);
}

/* movzx r32, r8 — 0F B6 /r */
static void emit_movzx_r32_r8(X64Buf *b, int dst, int src) {
  if (dst >= 8 || src >= 8) emit_rex(b, 0, dst >> 3, 0, src >> 3);
  xb_byte(b, 0x0F);
  xb_byte(b, 0xB6);
  emit_modrm(b, 3, dst, src);
}

/* and r32, imm32 — 81 /4 id */
static void emit_and_r32_imm32(X64Buf *b, int reg, int32_t imm) {
  if (reg >= 8) emit_rex(b, 0, 0, 0, reg >> 3);
  xb_byte(b, 0x81);
  emit_modrm(b, 3, 4, reg);
  xb_i32(b, imm);
}

/* call r64 — FF /2 */
static void emit_call_r64(X64Buf *b, int reg) {
  if (reg >= 8) emit_rex(b, 0, 0, 0, reg >> 3);
  xb_byte(b, 0xFF);
  emit_modrm(b, 3, 2, reg);
}

/* inc qword [rbp+disp] — REX.W FF /0 */
static void emit_inc_rbp_q(X64Buf *b, int disp) {
  emit_rex_always(b, 1, 0, 0, 0);
  xb_byte(b, 0xFF);
  emit_modrm_rbp_disp32(b, 0, disp);
}

/* Patch a rel32 at offset `patch_off` to jump to `target_off` */
static void patch_rel32(X64Buf *b, int patch_off, int target_off) {
  int32_t rel = target_off - (patch_off + 4);
  memcpy(b->data + patch_off, &rel, 4);
}

/* ══════════════════════════════════════════════════════════════════════ */
/*  Loop tracking + register assignment for loop counters                */
/* ══════════════════════════════════════════════════════════════════════ */

/* GPRs dedicated to loop counters (callee-saved, survive iterations) */
static const int LOOP_GPRS[] = { R12, R13, R14, RBX };
#define N_LOOP_GPRS 4

/* Track which UOps live in dedicated GPRs */
typedef struct {
  PolyUOp *uop;
  int gpr;
} RegAssign;

#define MAX_REG_ASSIGNS 8

typedef struct {
  PolyUOp *range;       /* which RANGE UOp */
  int jge_disp_offset;  /* offset of rel32 in the jge instruction */
  int loop_body_start;  /* byte offset of loop condition check */
  int counter_slot;     /* stack slot index for loop counter */
  int gpr;              /* dedicated GPR for counter (-1 = memory fallback) */
} LoopPatch;

#define MAX_LOOP_DEPTH 32

/* Helper: find GPR assignment for a UOp */
static int find_reg(RegAssign *regs, int n_regs, PolyUOp *u) {
  for (int i = 0; i < n_regs; i++)
    if (regs[i].uop == u) return regs[i].gpr;
  return -1;
}

/* Emit inc r64 */
static void emit_inc_r64(X64Buf *b, int reg) {
  emit_rex_always(b, 1, 0, 0, reg >> 3);
  xb_byte(b, 0xFF);
  emit_modrm(b, 3, 0, reg);
}

/* ── SIB-addressed SSE loads/stores (for fused INDEX+LOAD/STORE) ───── */

/* movss xmm, [base + index*scale] — F3 (REX) 0F 10 ModRM SIB */
static void emit_movss_xmm_sib(X64Buf *b, int xmm, int base, int index, int scale) {
  int ss = 0;
  switch (scale) { case 1: ss=0; break; case 2: ss=1; break; case 4: ss=2; break; case 8: ss=3; break; }
  xb_byte(b, 0xF3);
  if (xmm >= 8 || index >= 8 || base >= 8)
    emit_rex(b, 0, xmm >> 3, index >> 3, base >> 3);
  xb_byte(b, 0x0F);
  xb_byte(b, 0x10);
  if ((base & 7) == RBP) {
    emit_modrm(b, 1, xmm, 4); /* mod=01, rm=SIB, disp8 */
    xb_byte(b, (uint8_t)((ss << 6) | ((index & 7) << 3) | (base & 7)));
    xb_byte(b, 0);
  } else {
    emit_modrm(b, 0, xmm, 4); /* mod=00, rm=SIB */
    xb_byte(b, (uint8_t)((ss << 6) | ((index & 7) << 3) | (base & 7)));
  }
}

/* movss [base + index*scale], xmm — F3 (REX) 0F 11 ModRM SIB */
static void emit_movss_sib_xmm(X64Buf *b, int xmm, int base, int index, int scale) {
  int ss = 0;
  switch (scale) { case 1: ss=0; break; case 2: ss=1; break; case 4: ss=2; break; case 8: ss=3; break; }
  xb_byte(b, 0xF3);
  if (xmm >= 8 || index >= 8 || base >= 8)
    emit_rex(b, 0, xmm >> 3, index >> 3, base >> 3);
  xb_byte(b, 0x0F);
  xb_byte(b, 0x11);
  if ((base & 7) == RBP) {
    emit_modrm(b, 1, xmm, 4);
    xb_byte(b, (uint8_t)((ss << 6) | ((index & 7) << 3) | (base & 7)));
    xb_byte(b, 0);
  } else {
    emit_modrm(b, 0, xmm, 4);
    xb_byte(b, (uint8_t)((ss << 6) | ((index & 7) << 3) | (base & 7)));
  }
}

/* SSE scalar op with SIB addressing: F3 (REX) 0F <opcode> xmm, [base+index*scale] */
static void emit_sse_scalar_sib(X64Buf *b, uint8_t opcode, int xmm,
                                 int base, int index, int scale) {
  int ss = 0;
  switch (scale) { case 1: ss=0; break; case 2: ss=1; break; case 4: ss=2; break; case 8: ss=3; break; }
  xb_byte(b, 0xF3);
  if (xmm >= 8 || index >= 8 || base >= 8)
    emit_rex(b, 0, xmm >> 3, index >> 3, base >> 3);
  xb_byte(b, 0x0F);
  xb_byte(b, opcode);
  if ((base & 7) == RBP) {
    emit_modrm(b, 1, xmm, 4);
    xb_byte(b, (uint8_t)((ss << 6) | ((index & 7) << 3) | (base & 7)));
    xb_byte(b, 0);
  } else {
    emit_modrm(b, 0, xmm, 4);
    xb_byte(b, (uint8_t)((ss << 6) | ((index & 7) << 3) | (base & 7)));
  }
}

static bool valid_sib_scale(int s) { return s == 1 || s == 2 || s == 4 || s == 8; }

/* Load an integer source value into a GPR.
 * Checks register assignment first (loop counters, PARAMs), falls back to stack. */
static void emit_load_int_src(X64Buf *buf, int dst_gpr, int w,
                               PolyUOp *src_uop, int src_slot,
                               RegAssign *regs, int n_regs) {
  int reg = find_reg(regs, n_regs, src_uop);
  if (reg >= 0) {
    emit_alu_rr(buf, w, 0x8B, dst_gpr, reg); /* mov dst, src_reg */
  } else if (src_slot >= 0) {
    emit_alu_r_rbp(buf, w, 0x8B, dst_gpr, -slot_offset(src_slot));
  }
}

/* Resolve the index operand of an INDEX UOp to a GPR.
 * Handles direct register assignments (RANGE counter) and
 * SHL(RANGE_in_reg, CONST) patterns from the vec UPCAST path.
 * Returns the GPR number, or -1 if the index must be loaded from stack.
 * When SHL is detected, emits inline computation into RCX. */
static int resolve_index_to_gpr(X64Buf *buf, PolyUOp *idx_src,
                                 RegAssign *regs, int n_regs,
                                 LocalMap *locals) {
  /* Direct: RANGE counter in a register */
  int r = find_reg(regs, n_regs, idx_src);
  if (r >= 0) return r;

  /* Fallback: load from stack slot into RCX */
  if (locals) {
    int slot = lm_get(locals, idx_src);
    if (slot >= 0) {
      emit_mov_r64_rbp(buf, RCX, -slot_offset(slot));
      return RCX;
    }
  }
  return -1;
}

/* Walk through transparent pointer casts to find the underlying INDEX.
 * The vec path produces LOAD→CAST→INDEX (pointer type cast for vec ptr). */
static PolyUOp *find_index_through_cast(PolyUOp *u) {
  while (u && (u->op == POLY_OP_CAST || u->op == POLY_OP_BITCAST)
         && u->n_src > 0 && u->dtype.is_ptr)
    u = u->src[0];
  return (u && u->op == POLY_OP_INDEX) ? u : NULL;
}

/* ── Packed SSE (no F3 prefix — operates on all 4 float lanes) ──────── */

/* movups xmm, [rbp+disp] — 0F 10 /r (no prefix) */
static void emit_movups_xmm_rbp(X64Buf *b, int xmm, int disp) {
  if (xmm >= 8) emit_rex(b, 0, xmm >> 3, 0, 0);
  xb_byte(b, 0x0F); xb_byte(b, 0x10);
  emit_modrm_rbp_disp32(b, xmm, disp);
}

/* movups [rbp+disp], xmm — 0F 11 /r */
static void emit_movups_rbp_xmm(X64Buf *b, int xmm, int disp) {
  if (xmm >= 8) emit_rex(b, 0, xmm >> 3, 0, 0);
  xb_byte(b, 0x0F); xb_byte(b, 0x11);
  emit_modrm_rbp_disp32(b, xmm, disp);
}

/* movups xmm, [base+index*scale] — (REX) 0F 10 ModRM SIB */
static void emit_movups_xmm_sib(X64Buf *b, int xmm, int base, int index, int scale) {
  int ss = 0;
  switch (scale) { case 1: ss=0; break; case 2: ss=1; break; case 4: ss=2; break; case 8: ss=3; break; }
  if (xmm >= 8 || index >= 8 || base >= 8)
    emit_rex(b, 0, xmm >> 3, index >> 3, base >> 3);
  xb_byte(b, 0x0F); xb_byte(b, 0x10);
  if ((base & 7) == RBP) {
    emit_modrm(b, 1, xmm, 4);
    xb_byte(b, (uint8_t)((ss << 6) | ((index & 7) << 3) | (base & 7)));
    xb_byte(b, 0);
  } else {
    emit_modrm(b, 0, xmm, 4);
    xb_byte(b, (uint8_t)((ss << 6) | ((index & 7) << 3) | (base & 7)));
  }
}

/* movups [base+index*scale], xmm — (REX) 0F 11 ModRM SIB */
static void emit_movups_sib_xmm(X64Buf *b, int xmm, int base, int index, int scale) {
  int ss = 0;
  switch (scale) { case 1: ss=0; break; case 2: ss=1; break; case 4: ss=2; break; case 8: ss=3; break; }
  if (xmm >= 8 || index >= 8 || base >= 8)
    emit_rex(b, 0, xmm >> 3, index >> 3, base >> 3);
  xb_byte(b, 0x0F); xb_byte(b, 0x11);
  if ((base & 7) == RBP) {
    emit_modrm(b, 1, xmm, 4);
    xb_byte(b, (uint8_t)((ss << 6) | ((index & 7) << 3) | (base & 7)));
    xb_byte(b, 0);
  } else {
    emit_modrm(b, 0, xmm, 4);
    xb_byte(b, (uint8_t)((ss << 6) | ((index & 7) << 3) | (base & 7)));
  }
}

/* Packed SSE op: (REX) 0F <opcode> xmm, xmm — no F3 prefix */
static void emit_sse_packed_rr(X64Buf *b, uint8_t opcode, int dst, int src) {
  if (dst >= 8 || src >= 8) emit_rex(b, 0, dst >> 3, 0, src >> 3);
  xb_byte(b, 0x0F); xb_byte(b, opcode);
  emit_modrm(b, 3, dst, src);
}

/* Packed SSE op with SIB: (REX) 0F <opcode> xmm, [base+index*scale] */
static void emit_sse_packed_sib(X64Buf *b, uint8_t opcode, int xmm,
                                 int base, int index, int scale) {
  int ss = 0;
  switch (scale) { case 1: ss=0; break; case 2: ss=1; break; case 4: ss=2; break; case 8: ss=3; break; }
  if (xmm >= 8 || index >= 8 || base >= 8)
    emit_rex(b, 0, xmm >> 3, index >> 3, base >> 3);
  xb_byte(b, 0x0F); xb_byte(b, opcode);
  if ((base & 7) == RBP) {
    emit_modrm(b, 1, xmm, 4);
    xb_byte(b, (uint8_t)((ss << 6) | ((index & 7) << 3) | (base & 7)));
    xb_byte(b, 0);
  } else {
    emit_modrm(b, 0, xmm, 4);
    xb_byte(b, (uint8_t)((ss << 6) | ((index & 7) << 3) | (base & 7)));
  }
}

/* movups xmm, xmm — 0F 10 /r (mod=11) */
static void emit_movups_xmm_xmm(X64Buf *b, int dst, int src) {
  if (dst >= 8 || src >= 8) emit_rex(b, 0, dst >> 3, 0, src >> 3);
  xb_byte(b, 0x0F); xb_byte(b, 0x10);
  emit_modrm(b, 3, dst, src);
}

/* cmp r64, imm32 (sign-extended) — REX.W 81 /7 id */
static void emit_cmp_r64_imm32(X64Buf *b, int reg, int32_t imm) {
  emit_rex_always(b, 1, 0, 0, reg >> 3);
  xb_byte(b, 0x81);
  emit_modrm(b, 3, 7, reg);
  xb_i32(b, imm);
}

/* ══════════════════════════════════════════════════════════════════════ */
/*  VEX-encoded AVX2 instruction helpers (Phase 6)                        */
/*  2-byte VEX: C5 [R vvvv L pp]                                         */
/*  3-byte VEX: C4 [RXB mmmmm] [W vvvv L pp]                             */
/*  R/X/B are inverted REX bits. vvvv is inverted src reg (1111=unused).  */
/*  L=0 for XMM (128-bit), L=1 for YMM (256-bit).                        */
/*  pp: 00=none, 01=66, 10=F3, 11=F2.  mmmmm: 01=0F, 02=0F38, 03=0F3A. */
/* ══════════════════════════════════════════════════════════════════════ */

/* 2-byte VEX: C5 [R vvvv L pp] — for 0F-map ops with no X/B extension */
static void emit_vex2(X64Buf *b, int R, int vvvv, int L, int pp) {
  xb_byte(b, 0xC5);
  xb_byte(b, (uint8_t)(((R & 1) << 7) | ((~vvvv & 0xF) << 3) | ((L & 1) << 2) | (pp & 3)));
}

/* 3-byte VEX: C4 [RXB mmmmm] [W vvvv L pp] — for 0F38/0F3A maps or R8+ regs */
static void emit_vex3(X64Buf *b, int R, int X, int B, int mmmmm,
                      int W, int vvvv, int L, int pp) {
  xb_byte(b, 0xC4);
  xb_byte(b, (uint8_t)(((R & 1) << 7) | ((X & 1) << 6) | ((B & 1) << 5) | (mmmmm & 0x1F)));
  xb_byte(b, (uint8_t)(((W & 1) << 7) | ((~vvvv & 0xF) << 3) | ((L & 1) << 2) | (pp & 3)));
}

/* Helper: emit VEX prefix choosing 2-byte or 3-byte form.
 * 2-byte form is only valid when: map=0F, W=0, X=1, B=1 (no ext needed). */
static void emit_vex_auto(X64Buf *b, int dst, int src1, int src2_or_rm,
                           int L, int pp, int mmmmm, int W) {
  int R = (dst < 8) ? 1 : 0;
  int X = 1; /* no index extension in reg-reg */
  int B = (src2_or_rm < 8) ? 1 : 0;
  int vvvv = src1; /* will be inverted in emit functions */
  if (mmmmm == 1 && W == 0 && X == 1 && B == 1) {
    emit_vex2(b, R, vvvv, L, pp);
  } else {
    emit_vex3(b, R, X, B, mmmmm, W, vvvv, L, pp);
  }
}

/* vmovups ymm, [rbp+disp32] — VEX.256 0F 10 /r (pp=00, L=1) */
static void emit_vmovups_ymm_rbp(X64Buf *b, int ymm, int disp) {
  int R = (ymm < 8) ? 1 : 0;
  /* RBP base: B=1 (rbp < 8) */
  emit_vex2(b, R, 0, 1, 0x00); /* vvvv=0 → inverted to 1111 (unused), L=1, pp=00 */
  xb_byte(b, 0x10);
  emit_modrm_rbp_disp32(b, ymm, disp);
}

/* vmovups [rbp+disp32], ymm — VEX.256 0F 11 /r */
static void emit_vmovups_rbp_ymm(X64Buf *b, int ymm, int disp) {
  int R = (ymm < 8) ? 1 : 0;
  emit_vex2(b, R, 0, 1, 0x00);
  xb_byte(b, 0x11);
  emit_modrm_rbp_disp32(b, ymm, disp);
}

/* vmovups ymm, [base+index*scale] — VEX.256 0F 10 /r with SIB */
static void emit_vmovups_ymm_sib(X64Buf *b, int ymm, int base, int index, int scale) {
  int ss = 0;
  switch (scale) { case 1: ss=0; break; case 2: ss=1; break; case 4: ss=2; break; case 8: ss=3; break; }
  int R = (ymm < 8) ? 1 : 0;
  int X = (index < 8) ? 1 : 0;
  int B = (base < 8) ? 1 : 0;
  emit_vex3(b, R, X, B, 1, 0, 0, 1, 0x00);
  xb_byte(b, 0x10);
  if ((base & 7) == RBP) {
    emit_modrm(b, 1, ymm, 4);
    xb_byte(b, (uint8_t)((ss << 6) | ((index & 7) << 3) | (base & 7)));
    xb_byte(b, 0);
  } else {
    emit_modrm(b, 0, ymm, 4);
    xb_byte(b, (uint8_t)((ss << 6) | ((index & 7) << 3) | (base & 7)));
  }
}

/* vmovups [base+index*scale], ymm — VEX.256 0F 11 /r with SIB */
static void emit_vmovups_sib_ymm(X64Buf *b, int ymm, int base, int index, int scale) {
  int ss = 0;
  switch (scale) { case 1: ss=0; break; case 2: ss=1; break; case 4: ss=2; break; case 8: ss=3; break; }
  int R = (ymm < 8) ? 1 : 0;
  int X = (index < 8) ? 1 : 0;
  int B = (base < 8) ? 1 : 0;
  emit_vex3(b, R, X, B, 1, 0, 0, 1, 0x00);
  xb_byte(b, 0x11);
  if ((base & 7) == RBP) {
    emit_modrm(b, 1, ymm, 4);
    xb_byte(b, (uint8_t)((ss << 6) | ((index & 7) << 3) | (base & 7)));
    xb_byte(b, 0);
  } else {
    emit_modrm(b, 0, ymm, 4);
    xb_byte(b, (uint8_t)((ss << 6) | ((index & 7) << 3) | (base & 7)));
  }
}

/* vmovups ymm, ymm — VEX.256 0F 10 /r (reg-reg) */
static void emit_vmovups_ymm_ymm(X64Buf *b, int dst, int src) {
  int R = (dst < 8) ? 1 : 0;
  int B = (src < 8) ? 1 : 0;
  if (B)
    emit_vex2(b, R, 0, 1, 0x00);
  else
    emit_vex3(b, R, 1, B, 1, 0, 0, 1, 0x00);
  xb_byte(b, 0x10);
  emit_modrm(b, 3, dst, src);
}

/* vmovups ymm, [rax+0] — VEX.256 0F 10 /r (base reg, no disp) */
static void emit_vmovups_ymm_mem(X64Buf *b, int ymm, int base) {
  int R = (ymm < 8) ? 1 : 0;
  int B = (base < 8) ? 1 : 0;
  if (B)
    emit_vex2(b, R, 0, 1, 0x00);
  else
    emit_vex3(b, R, 1, B, 1, 0, 0, 1, 0x00);
  xb_byte(b, 0x10);
  emit_modrm(b, 0, ymm, base);
}

/* vmovups [rax+0], ymm — VEX.256 0F 11 /r (base reg, no disp) */
static void emit_vmovups_mem_ymm(X64Buf *b, int ymm, int base) {
  int R = (ymm < 8) ? 1 : 0;
  int B = (base < 8) ? 1 : 0;
  if (B)
    emit_vex2(b, R, 0, 1, 0x00);
  else
    emit_vex3(b, R, 1, B, 1, 0, 0, 1, 0x00);
  xb_byte(b, 0x11);
  emit_modrm(b, 0, ymm, base);
}

/* VEX packed ALU reg-reg: vaddps/vmulps/vsubps/vdivps/vmaxps
 * 3-operand: dst = src1 op src2.  VEX.pp=00 (no prefix), map=0F. */
static void emit_vex_packed_rrr(X64Buf *b, uint8_t opc, int dst, int src1, int src2, int L) {
  emit_vex_auto(b, dst, src1, src2, L, 0x00, 1, 0);
  xb_byte(b, opc);
  emit_modrm(b, 3, dst, src2);
}

/* VEX packed ALU with SIB memory operand: dst = src1 op [base+idx*scale] */
static void emit_vex_packed_rr_sib(X64Buf *b, uint8_t opc, int dst, int src1,
                                    int base, int idx, int scale, int L) {
  int ss = 0;
  switch (scale) { case 1: ss=0; break; case 2: ss=1; break; case 4: ss=2; break; case 8: ss=3; break; }
  int R = (dst < 8) ? 1 : 0;
  int X = (idx < 8) ? 1 : 0;
  int B = (base < 8) ? 1 : 0;
  emit_vex3(b, R, X, B, 1, 0, src1, L, 0x00);
  xb_byte(b, opc);
  if ((base & 7) == RBP) {
    emit_modrm(b, 1, dst, 4);
    xb_byte(b, (uint8_t)((ss << 6) | ((idx & 7) << 3) | (base & 7)));
    xb_byte(b, 0);
  } else {
    emit_modrm(b, 0, dst, 4);
    xb_byte(b, (uint8_t)((ss << 6) | ((idx & 7) << 3) | (base & 7)));
  }
}

/* VEX packed unary (sqrt): vsqrtps dst, src — VEX 0F 51 */
static void emit_vex_packed_sqrt(X64Buf *b, int dst, int src, int L) {
  emit_vex_auto(b, dst, 0, src, L, 0x00, 1, 0); /* vvvv=0 → unused */
  xb_byte(b, 0x51);
  emit_modrm(b, 3, dst, src);
}

/* vxorps dst, src1, src2 — VEX 0F 57 */
static void emit_vxorps(X64Buf *b, int dst, int src1, int src2, int L) {
  emit_vex_auto(b, dst, src1, src2, L, 0x00, 1, 0);
  xb_byte(b, 0x57);
  emit_modrm(b, 3, dst, src2);
}

/* vandps dst, src1, src2 — VEX 0F 54 */
static void emit_vandps(X64Buf *b, int dst, int src1, int src2, int L) {
  emit_vex_auto(b, dst, src1, src2, L, 0x00, 1, 0);
  xb_byte(b, 0x54);
  emit_modrm(b, 3, dst, src2);
}

/* vandnps dst, src1, src2 — VEX 0F 55 (dst = ~src1 & src2) */
static void emit_vandnps(X64Buf *b, int dst, int src1, int src2, int L) {
  emit_vex_auto(b, dst, src1, src2, L, 0x00, 1, 0);
  xb_byte(b, 0x55);
  emit_modrm(b, 3, dst, src2);
}

/* vorps dst, src1, src2 — VEX 0F 56 */
static void emit_vorps(X64Buf *b, int dst, int src1, int src2, int L) {
  emit_vex_auto(b, dst, src1, src2, L, 0x00, 1, 0);
  xb_byte(b, 0x56);
  emit_modrm(b, 3, dst, src2);
}

/* vbroadcastss ymm, xmm — VEX.256 0F38 18 /r (AVX2) */
static void emit_vbroadcastss_ymm_xmm(X64Buf *b, int dst, int src) {
  int R = (dst < 8) ? 1 : 0;
  int B = (src < 8) ? 1 : 0;
  emit_vex3(b, R, 1, B, 2, 0, 0, 1, 0x01); /* map=0F38, pp=01(66), L=1 */
  xb_byte(b, 0x18);
  emit_modrm(b, 3, dst, src);
}

/* vfmadd231ps/ss dst, src1, src2 — VEX 0F38 B8/B9 /r (FMA3)
 * dst = src1 * src2 + dst
 * Packed (B8): vfmadd231ps, L selects 128/256-bit
 * Scalar (B9): vfmadd231ss, L ignored (LIG) */
static void emit_vfmadd231(X64Buf *b, int dst, int src1, int src2, int L, bool scalar) {
  int R = (dst < 8) ? 1 : 0;
  int B = (src2 < 8) ? 1 : 0;
  emit_vex3(b, R, 1, B, 2, 0, src1, scalar ? 0 : L, 0x01); /* map=0F38, pp=01(66) */
  xb_byte(b, scalar ? 0xB9 : 0xB8);
  emit_modrm(b, 3, dst, src2);
}

/* vextractf128 xmm, ymm, imm8 — VEX.256 0F3A 19 /r ib */
static void emit_vextractf128(X64Buf *b, int dst, int src, uint8_t imm) {
  int R = (src < 8) ? 1 : 0; /* note: src in reg field, dst in r/m */
  int B = (dst < 8) ? 1 : 0;
  emit_vex3(b, R, 1, B, 3, 0, 0, 1, 0x01); /* map=0F3A, pp=01(66), L=1 */
  xb_byte(b, 0x19);
  emit_modrm(b, 3, src, dst);
  xb_byte(b, imm);
}

/* vpshufd dst, src, imm8 — VEX 0F 70 /r ib (pp=01 for 66 prefix) */
static void emit_vpshufd(X64Buf *b, int dst, int src, uint8_t imm, int L) {
  emit_vex_auto(b, dst, 0, src, L, 0x01, 1, 0); /* vvvv=0 → unused, pp=01(66) */
  xb_byte(b, 0x70);
  emit_modrm(b, 3, dst, src);
  xb_byte(b, imm);
}

/* vzeroupper — C5 F8 77 (clears upper 128 bits of all YMM regs) */
static void emit_vzeroupper(X64Buf *b) {
  xb_byte(b, 0xC5);
  xb_byte(b, 0xF8);
  xb_byte(b, 0x77);
}

/* ══════════════════════════════════════════════════════════════════════ */
/*  Packed integer SIMD instructions (Phase 8: SIMD integer in XMM/YMM)  */
/*  All use VEX pp=01(66) map=01(0F). 3-operand NDS form for reg-reg,    */
/*  NDD form for immediate shifts. L=0 for 128-bit, L=1 for 256-bit.    */
/* ══════════════════════════════════════════════════════════════════════ */

/* vpaddd dst, src1, src2 — VEX.66.0F FE /r (packed int32 add) */
static void emit_vpaddd(X64Buf *b, int dst, int src1, int src2, int L) {
  emit_vex_auto(b, dst, src1, src2, L, 0x01, 1, 0);
  xb_byte(b, 0xFE);
  emit_modrm(b, 3, dst, src2);
}

/* vpsubd dst, src1, src2 — VEX.66.0F FA /r (packed int32 subtract) */
static void emit_vpsubd(X64Buf *b, int dst, int src1, int src2, int L) {
  emit_vex_auto(b, dst, src1, src2, L, 0x01, 1, 0);
  xb_byte(b, 0xFA);
  emit_modrm(b, 3, dst, src2);
}

/* vpand dst, src1, src2 — VEX.66.0F DB /r (packed bitwise AND) */
static void emit_vpand(X64Buf *b, int dst, int src1, int src2, int L) {
  emit_vex_auto(b, dst, src1, src2, L, 0x01, 1, 0);
  xb_byte(b, 0xDB);
  emit_modrm(b, 3, dst, src2);
}

/* vpor dst, src1, src2 — VEX.66.0F EB /r (packed bitwise OR) */
static void emit_vpor(X64Buf *b, int dst, int src1, int src2, int L) {
  emit_vex_auto(b, dst, src1, src2, L, 0x01, 1, 0);
  xb_byte(b, 0xEB);
  emit_modrm(b, 3, dst, src2);
}

/* vpslld dst, src, imm8 — VEX.66.0F 72 /6 ib (packed int32 shift left)
 * NDD form: vvvv=dst, r/m=src, reg=6 (opcode extension) */
static void emit_vpslld_imm(X64Buf *b, int dst, int src, uint8_t imm, int L) {
  emit_vex_auto(b, 6, dst, src, L, 0x01, 1, 0); /* reg=6 (/6), vvvv=dst */
  xb_byte(b, 0x72);
  emit_modrm(b, 3, 6, src);
  xb_byte(b, imm);
}

/* vpsrld dst, src, imm8 — VEX.66.0F 72 /2 ib (packed int32 logical shift right)
 * NDD form: vvvv=dst, r/m=src, reg=2 (opcode extension) */
static void emit_vpsrld_imm(X64Buf *b, int dst, int src, uint8_t imm, int L) {
  emit_vex_auto(b, 2, dst, src, L, 0x01, 1, 0); /* reg=2 (/2), vvvv=dst */
  xb_byte(b, 0x72);
  emit_modrm(b, 3, 2, src);
  xb_byte(b, imm);
}

/* ══════════════════════════════════════════════════════════════════════ */
/*  Width-parameterized vector load/store helpers (Phase 5b + Phase 6)   */
/*  Dispatch to SSE (movss/movups) or AVX2 (vmovups YMM) based on       */
/*  VecConfig.                                                           */
/* ══════════════════════════════════════════════════════════════════════ */

/* Width-aware vector load from stack slot.
 * vec_width determines instruction: >=8→vmovups YMM, 2-4→movups XMM, 0-1→movss */
static void emit_width_load_rbp(X64Buf *b, int reg, int disp, int vec_width) {
  if (vec_width >= 8)      emit_vmovups_ymm_rbp(b, reg, disp);
  else if (vec_width >= 2) emit_movups_xmm_rbp(b, reg, disp);
  else                     emit_movss_xmm_rbp(b, reg, disp);
}

/* Width-aware vector store to stack slot */
static void emit_width_store_rbp(X64Buf *b, int reg, int disp, int vec_width) {
  if (vec_width >= 8)      emit_vmovups_rbp_ymm(b, reg, disp);
  else if (vec_width >= 2) emit_movups_rbp_xmm(b, reg, disp);
  else                     emit_movss_rbp_xmm(b, reg, disp);
}

/* ══════════════════════════════════════════════════════════════════════ */
/*  XMM register file: keep float values in XMM1-XMM7 to avoid          */
/*  stack round-trips. XMM0 reserved as scratch for non-file ops.        */
/* ══════════════════════════════════════════════════════════════════════ */

#define XF_SIZE 15     /* XMM1 through XMM15 */
#define XF_BASE 1      /* first allocatable XMM register number */

typedef struct {
  int slot;      /* stack slot this register mirrors (-1 = free) */
  bool dirty;    /* value only in register, not yet in stack */
  int vec_width; /* 0 or 1 = scalar (movss, 4 bytes), 4 = SSE packed (movups, 16 bytes) */
  bool pinned;   /* true if pinned (accumulator); never evicted */
} XfEntry;

typedef struct {
  XfEntry e[XF_SIZE];
  int next_evict;  /* round-robin pointer */
} XmmFile;

static void xf_init(XmmFile *f) {
  for (int i = 0; i < XF_SIZE; i++) f->e[i] = (XfEntry){ .slot = -1 };
  f->next_evict = 0;
}

/* Find register holding slot (-1 if not cached) */
static int xf_find(XmmFile *f, int slot) {
  if (slot < 0) return -1;
  for (int i = 0; i < XF_SIZE; i++)
    if (f->e[i].slot == slot) return XF_BASE + i;
  return -1;
}

/* Allocate register for slot, evicting if needed.
 * avoid1/avoid2/avoid3 = regs not to evict (-1 = unused).
 * slot_last_use + cur_pos enable Belady's eviction (evict furthest next_use).
 * Pass slot_last_use=NULL for legacy round-robin fallback. */
static int xf_alloc_belady(XmmFile *f, X64Buf *buf, int slot,
                           int avoid1, int avoid2, int avoid3,
                           const int *slot_last_use, int cur_pos) {
  int r = xf_find(f, slot);
  if (r >= 0) return r;

  /* Find free */
  for (int i = 0; i < XF_SIZE; i++) {
    if (f->e[i].slot < 0) {
      f->e[i] = (XfEntry){ .slot = slot };
      return XF_BASE + i;
    }
  }

  /* Evict: Belady's (evict furthest next_use) or round-robin fallback */
  int best_ei = -1, best_dist = -1;
  for (int i = 0; i < XF_SIZE; i++) {
    int reg = XF_BASE + i;
    if (reg == avoid1 || reg == avoid2 || reg == avoid3) continue;
    if (f->e[i].pinned) continue; /* never evict pinned (accumulator) */
    if (slot_last_use) {
      int dist = (f->e[i].slot >= 0) ? slot_last_use[f->e[i].slot] - cur_pos : 0;
      if (dist > best_dist) { best_dist = dist; best_ei = i; }
    } else {
      /* Round-robin fallback */
      best_ei = f->next_evict;
      f->next_evict = (f->next_evict + 1) % XF_SIZE;
      if ((XF_BASE + best_ei) != avoid1 && (XF_BASE + best_ei) != avoid2 && !f->e[best_ei].pinned)
        break;
      best_ei = -1; /* try next */
    }
  }
  if (best_ei < 0) best_ei = 0; /* shouldn't happen */
  int reg = XF_BASE + best_ei;
  if (f->e[best_ei].dirty) {
    emit_width_store_rbp(buf, reg,
                         -slot_offset(f->e[best_ei].slot), f->e[best_ei].vec_width);
  }
  f->e[best_ei] = (XfEntry){ .slot = slot };
  return reg;
}

/* Legacy wrapper: round-robin eviction (no Belady's info) */
static int xf_alloc(XmmFile *f, X64Buf *buf, int slot, int avoid1, int avoid2) {
  return xf_alloc_belady(f, buf, slot, avoid1, avoid2, -1, NULL, 0);
}

/* Get slot's register, loading from stack if not cached.
 * avoid1 = register not to evict (-1 = no constraint). */
static int xf_get_avoid(XmmFile *f, X64Buf *buf, int slot, int avoid1) {
  int r = xf_find(f, slot);
  if (r >= 0) return r;
  r = xf_alloc(f, buf, slot, avoid1, -1);
  emit_movss_xmm_rbp(buf, r, -slot_offset(slot));
  return r;
}
static int xf_get(XmmFile *f, X64Buf *buf, int slot) {
  return xf_get_avoid(f, buf, slot, -1);
}

/* Get slot's register for packed value, loading with the given width */
static int xf_get_packed_w(XmmFile *f, X64Buf *buf, int slot, int width) {
  int r = xf_find(f, slot);
  if (r >= 0) return r;
  r = xf_alloc(f, buf, slot, -1, -1);
  emit_width_load_rbp(buf, r, -slot_offset(slot), width);
  f->e[r - XF_BASE].vec_width = width;
  return r;
}

/* Clear all entries without spilling (iteration-local values discarded) */
static void xf_clear(XmmFile *f) {
  for (int i = 0; i < XF_SIZE; i++) f->e[i] = (XfEntry){ .slot = -1 };
}

/* Flush: spill all dirty entries using per-entry vec_width, then clear */
static void xf_flush(XmmFile *f, X64Buf *buf) {
  for (int i = 0; i < XF_SIZE; i++) {
    if (f->e[i].slot >= 0 && f->e[i].dirty) {
      emit_width_store_rbp(buf, XF_BASE + i,
                           -slot_offset(f->e[i].slot), f->e[i].vec_width);
    }
  }
  xf_clear(f);
}

/* ══════════════════════════════════════════════════════════════════════ */
/*  Renderer: walk linearized UOps, emit x86-64 machine code             */
/* ══════════════════════════════════════════════════════════════════════ */

/* Determine if a dtype is float for choosing instruction family.
 * For vec types: bitsize = element_bits * count, so check per-element size. */
static bool dtype_is_float(PolyDType dt) {
  int elem_bits = (dt.count > 1) ? (dt.bitsize / dt.count) : dt.bitsize;
  return elem_bits == 32 && (dt.fmt == 'f' || (dt.name && strcmp(dt.name, "float") == 0));
}

/* Check if dtype is a packed vector float (f32 x N, N > 1) */
static bool dtype_is_vec_float(PolyDType dt) {
  return dtype_is_float(dt) && dt.count > 1;
}

static bool dtype_is_int(PolyDType dt) {
  return !dtype_is_float(dt) && dt.bitsize > 0;
}

/* Walk through AFTER/CAST/BITCAST/INDEX to find the underlying DEFINE_REG
 * or DEFINE_LOCAL.  The core's pm_reduce emits AFTER(DEFINE_REG, ...)
 * chains, so INDEX(AFTER(DEFINE_REG)) must resolve to the DEFINE_REG for
 * accumulator LOAD/STORE fast paths. */
static PolyUOp *resolve_acc_base_fn(PolyUOp *u) {
  if (!u) return NULL;
  if (u->op == POLY_OP_DEFINE_REG || u->op == POLY_OP_DEFINE_LOCAL) return u;
  if (u->op == POLY_OP_AFTER && u->n_src > 0)   return resolve_acc_base_fn(u->src[0]);
  if (u->op == POLY_OP_CAST && u->n_src > 0)    return resolve_acc_base_fn(u->src[0]);
  if (u->op == POLY_OP_BITCAST && u->n_src > 0)  return resolve_acc_base_fn(u->src[0]);
  if (u->op == POLY_OP_INDEX && u->n_src > 0)    return resolve_acc_base_fn(u->src[0]);
  return NULL;
}

uint8_t *poly_render_x64(PolyUOp **uops, int n, int *size_out) {
  X64Buf buf = {0};
  LocalMap locals;
  lm_init(&locals, n * 2);

  /* ── Pre-scan: count slots, RANGEs, PARAMs, detect max vec width ─── */
  int n_slots = 0;
  int n_params = 0;
  int n_ranges = 0;
  int max_vec_width = 0;
  for (int i = 0; i < n; i++) {
    PolyUOp *u = uops[i];
    if (u->op == POLY_OP_SINK || u->op == POLY_OP_NOOP ||
        u->op == POLY_OP_GROUP || u->op == POLY_OP_ENDIF)
      continue;
    if (u->op == POLY_OP_AFTER) continue;
    if (u->op == POLY_OP_STORE) continue;
    if (u->op == POLY_OP_END) continue;
    if (u->op == POLY_OP_PARAM) n_params++;
    if (u->op == POLY_OP_RANGE) n_ranges++;
    if (u->dtype.count > max_vec_width) max_vec_width = u->dtype.count;
    n_slots++;
  }

  /* Select VecConfig based on IR vector width and CPU caps */
  X64CpuCaps cpu = get_cpu_caps();
  bool use_avx2 = (max_vec_width >= 8) && cpu.has_avx2 && cpu.os_avx_ok;
  (void)VCFG_AVX2; (void)VCFG_SSE; /* used by helper functions */

  /* ── Phase 3a: precompute last consumer index per UOp ────────────── */
  /* last_consumer[i] = the latest UOp index j that references uops[i]
   * as a source. Used for dead-value freeing and Belady's eviction.
   * Build a ptr→index map for O(1) source lookups. */
  int *last_consumer = calloc((size_t)n, sizeof(int));
  for (int i = 0; i < n; i++) last_consumer[i] = i; /* default: self */
  /* Simple open-addressing hash map: UOp* → linearized index */
  typedef struct { PolyUOp *key; int idx; } UOpMapEntry;
  int uop_map_cap = n < 32 ? 64 : (n * 4);
  UOpMapEntry *uop_map = calloc((size_t)uop_map_cap, sizeof(UOpMapEntry));
  for (int i = 0; i < n; i++) {
    uint64_t h = ((uint64_t)(uintptr_t)uops[i] >> 3) * 0x9E3779B97F4A7C15ULL;
    int pos = (int)(h % (uint64_t)uop_map_cap);
    while (uop_map[pos].key) pos = (pos + 1) % uop_map_cap;
    uop_map[pos].key = uops[i];
    uop_map[pos].idx = i;
  }
  for (int i = 0; i < n; i++) {
    PolyUOp *u = uops[i];
    for (int k = 0; k < u->n_src; k++) {
      /* Find source's index via hash map */
      uint64_t sh = ((uint64_t)(uintptr_t)u->src[k] >> 3) * 0x9E3779B97F4A7C15ULL;
      int sp = (int)(sh % (uint64_t)uop_map_cap);
      int j = -1;
      while (uop_map[sp].key) {
        if (uop_map[sp].key == u->src[k]) { j = uop_map[sp].idx; break; }
        sp = (sp + 1) % uop_map_cap;
      }
      if (j >= 0 && i > last_consumer[j]) last_consumer[j] = i;
    }
  }
  /* Per-slot last-use tracking (filled during rendering) */
  int *slot_last_use = calloc((size_t)(n_slots + 16), sizeof(int));
  for (int i = 0; i < n_slots + 16; i++) slot_last_use[i] = INT32_MAX;

  /* GPR allocation plan: LOOP_GPRS = {R12, R13, R14, RBX}
   * Reserve first n_ranges GPRs for loop counters.
   * Remaining GPRs go to PARAMs (base pointers). */
  int n_range_gprs = n_ranges < N_LOOP_GPRS ? n_ranges : N_LOOP_GPRS;
  int n_param_gprs_avail = N_LOOP_GPRS - n_range_gprs;
  int n_param_gprs = n_params < n_param_gprs_avail ? n_params : n_param_gprs_avail;
  /* PARAM GPRs start after range GPRs in LOOP_GPRS array */
  int param_gpr_start = n_range_gprs;

  /* Frame size: must leave RSP 16-byte aligned.
   * We push 6 callee-saved regs (RBP, R15, R14, R13, R12, RBX).
   * After CALL: RSP = 8 mod 16. After 6 pushes: RSP = 8-48 = 8 mod 16.
   * So frame_size must be 8 mod 16 to get 0 mod 16. */
  /* Frame size: each slot is SLOT_BYTES (16). SSE values fit in 1 slot.
   * Align to 32 bytes (AVX2-ready) and ensure 16-byte RSP alignment. */
  int raw_frame = n_slots * SLOT_BYTES;
  int frame_size = ((raw_frame + 31) & ~31) + 8; /* 8 mod 16 for RSP alignment */
  if (frame_size < 8) frame_size = 8;

  /* ── Prologue ────────────────────────────────────────────────────── */
  emit_push(&buf, RBP);
  /* mov rbp, rsp */
  emit_rex_always(&buf, 1, 0, 0, 0);
  xb_byte(&buf, 0x89);
  emit_modrm(&buf, 3, RSP, RBP);

  emit_push(&buf, R15);
  emit_push(&buf, R14);
  emit_push(&buf, R13);
  emit_push(&buf, R12);
  emit_push(&buf, RBX);
  emit_sub_rsp_imm32(&buf, frame_size);

  /* mov r15, rdi (save args pointer) */
  emit_rex_always(&buf, 1, RDI >> 3, 0, R15 >> 3);
  xb_byte(&buf, 0x89);
  emit_modrm(&buf, 3, RDI, R15);

  /* Preload sign mask for NEG: broadcast 0x80000000 to all lanes */
  emit_mov_r32_imm32(&buf, RAX, (int32_t)0x80000000u);
  emit_movd_xmm_r32(&buf, XMM0, RAX);
  if (use_avx2) {
    /* vbroadcastss ymm0, xmm0 — broadcast to all 8 lanes */
    emit_vbroadcastss_ymm_xmm(&buf, XMM0, XMM0);
  } else {
    /* shufps xmm0, xmm0, 0 — broadcast to all 4 lanes */
    xb_byte(&buf, 0x0F); xb_byte(&buf, 0xC6);
    emit_modrm(&buf, 3, XMM0, XMM0); xb_byte(&buf, 0x00);
  }

  /* Macro to reload sign mask in XMM0 (AVX2-aware).
   * Used after XMM0 is clobbered by scratch operations. */
  #define RELOAD_SIGN_MASK() do { \
    emit_mov_r32_imm32(&buf, RAX, (int32_t)0x80000000u); \
    emit_movd_xmm_r32(&buf, XMM0, RAX); \
    if (use_avx2) { \
      emit_vbroadcastss_ymm_xmm(&buf, XMM0, XMM0); \
    } else { \
      xb_byte(&buf, 0x0F); xb_byte(&buf, 0xC6); \
      emit_modrm(&buf, 3, XMM0, XMM0); xb_byte(&buf, 0x00); \
    } \
    rax_slot = -1; \
  } while(0)

  /* ── Walk UOps ───────────────────────────────────────────────────── */
  int next_slot = 0;
  LoopPatch loop_stack[MAX_LOOP_DEPTH];
  int loop_depth = 0;
  int n_loop_regs = 0; /* how many LOOP_GPRS have been assigned */

  /* Register assignments for loop counters and PARAM base pointers */
  RegAssign reg_assigns[MAX_REG_ASSIGNS];
  int n_reg_assigns = 0;
  int n_param_gprs_assigned = 0;

  /* Value caches */
  int rax_slot = -1;
  int xmm0_slot = -1; /* legacy: for non-register-file paths */

  /* XMM register file for float values */
  XmmFile xf;
  xf_init(&xf);

  /* Deferred LOADs: single-use LOADs from fusable INDEX that get folded
   * into their consumer ALU op as SIB memory operands. */
  typedef struct { PolyUOp *uop; int base_reg, idx_reg, itemsize; } DeferredSIB;
  DeferredSIB deferred[32];
  int n_deferred = 0;

  /* Wrap lm_set to also track slot last-use for Belady's eviction.
   * IMPORTANT: `i` must be the current UOp index in scope. */
  #define LM_SET(u_ptr, slot_val) do { \
    lm_set(&locals, (u_ptr), (slot_val)); \
    if ((slot_val) >= 0 && (slot_val) < n_slots + 16) \
      slot_last_use[slot_val] = last_consumer[i]; \
  } while(0)

  /* IF forward-jump patch stack */
  int if_patch_stack[MAX_LOOP_DEPTH];
  int if_depth = 0;

  for (int i = 0; i < n; i++) {
    PolyUOp *u = uops[i];

    /* Skip non-code UOps */
    if (u->op == POLY_OP_SINK || u->op == POLY_OP_NOOP || u->op == POLY_OP_GROUP)
      continue;

    /* ── Phase 3a: free dead XMM entries ──────────────────────────── */
    for (int ri = 0; ri < XF_SIZE; ri++) {
      int s = xf.e[ri].slot;
      if (s >= 0 && !xf.e[ri].pinned && slot_last_use[s] < i) {
        /* Last consumer already processed; free without spill */
        xf.e[ri] = (XfEntry){ .slot = -1 };
      }
    }

    /* ── PARAM: load buffer pointer from args array ──────────────── */
    if (u->op == POLY_OP_PARAM) {
      int slot = next_slot++;
      int off = -slot_offset(slot);
      int param_idx = (int)u->arg.i;

      /* Try to assign a dedicated GPR for this base pointer */
      int gpr = -1;
      if (n_param_gprs_assigned < n_param_gprs && n_reg_assigns < MAX_REG_ASSIGNS) {
        gpr = LOOP_GPRS[param_gpr_start + n_param_gprs_assigned];
        n_param_gprs_assigned++;
        reg_assigns[n_reg_assigns++] = (RegAssign){ u, gpr };
        /* Load base pointer directly into dedicated GPR */
        emit_mov_r64_mem(&buf, gpr, R15, 8 * param_idx);
      } else {
        /* Fallback: load to RAX, store to slot */
        emit_mov_r64_mem(&buf, RAX, R15, 8 * param_idx);
        emit_mov_rbp_r64(&buf, RAX, off);
        rax_slot = slot;
      }
      /* Also store to slot for fallback consumers */
      if (gpr >= 0) {
        emit_mov_rbp_r64(&buf, gpr, off);
      }
      LM_SET(u, slot);
      continue;
    }

    /* ── DEFINE_VAR: load int variable from args array ───────────── */
    if (u->op == POLY_OP_DEFINE_VAR) {
      int slot = next_slot++;
      int off = -slot_offset(slot);
      int var_idx = (int)u->arg.i;
      /* mov rax, [r15 + 8*var_idx] */
      emit_mov_r64_mem(&buf, RAX, R15, 8 * var_idx);
      /* mov eax, [rax] — dereference to get int value */
      emit_mov_r32_mem_base(&buf, RAX, RAX);
      /* movsxd rax, eax — sign-extend to 64-bit */
      emit_movsxd(&buf, RAX, RAX);
      /* mov [rbp - off], rax */
      emit_mov_rbp_r64(&buf, RAX, off);
      LM_SET(u, slot);
      continue;
    }

    /* ── CONST: store immediate value to slot ────────────────────── */
    if (u->op == POLY_OP_CONST) {
      int slot = next_slot++;
      int off = -slot_offset(slot);

      if (dtype_is_float(u->dtype)) {
        float fv = (float)u->arg.f;
        int32_t bits;
        memcpy(&bits, &fv, 4);
        if (u->dtype.count > 1) {
          /* Vec CONST: broadcast to all lanes via XMM0 scratch */
          emit_mov_r32_imm32(&buf, RAX, bits);
          emit_movd_xmm_r32(&buf, XMM0, RAX);
          if (use_avx2 && u->dtype.count >= 8) {
            /* vbroadcastss ymm0, xmm0 — broadcast to all 8 lanes */
            emit_vbroadcastss_ymm_xmm(&buf, XMM0, XMM0);
            emit_vmovups_rbp_ymm(&buf, XMM0, off);
          } else {
            /* shufps xmm0, xmm0, 0 — broadcast lane 0 */
            xb_byte(&buf, 0x0F); xb_byte(&buf, 0xC6);
            emit_modrm(&buf, 3, XMM0, XMM0); xb_byte(&buf, 0x00);
            emit_movups_rbp_xmm(&buf, XMM0, off);
          }
          RELOAD_SIGN_MASK();
        } else {
          emit_mov_rbp_imm32(&buf, off, bits);
        }
      } else {
        /* Integer constant */
        int64_t iv = u->arg.i;
        if (iv >= INT32_MIN && iv <= INT32_MAX) {
          emit_mov_rbp_imm64_sx(&buf, off, (int32_t)iv);
        } else {
          /* Need full 64-bit immediate */
          emit_mov_r64_imm64(&buf, RAX, iv);
          emit_mov_rbp_r64(&buf, RAX, off);
        }
      }
      LM_SET(u, slot);
      continue;
    }

    /* ── DEFINE_LOCAL / DEFINE_REG: initialize accumulator ───────── */
    if (u->op == POLY_OP_DEFINE_LOCAL || u->op == POLY_OP_DEFINE_REG) {
      int slot = next_slot++;
      int off = -slot_offset(slot);
      /* Initialize to 0 (zero the full 8 bytes) */
      emit_mov_rbp_imm64_sx(&buf, off, 0);
      LM_SET(u, slot);
      continue;
    }

    /* ── AFTER: alias to src[0] ──────────────────────────────────── */
    if (u->op == POLY_OP_AFTER) {
      int s0 = lm_get(&locals, u->src[0]);
      if (s0 >= 0) LM_SET(u, s0);
      continue;
    }

    /* ── INDEX: pointer arithmetic (base + idx * itemsize) ───────── */
    if (u->op == POLY_OP_INDEX) {
      {
      }

      int slot = next_slot++;
      int off = -slot_offset(slot);

      int base_slot = lm_get(&locals, u->src[0]);
      int idx_slot = lm_get(&locals, u->src[1]);
      if (base_slot < 0 || idx_slot < 0) goto skip_slot;

      /* Get itemsize from the buffer's dtype */
      int itemsize = poly_dtype_itemsize(u->src[0]->dtype);
      if (u->src[0]->dtype.is_ptr && u->src[0]->dtype.bitsize > 0)
        itemsize = u->src[0]->dtype.bitsize / 8;

      /* Check if all consumers of this INDEX can fuse via SIB.
       * If so, skip emission — LOAD/STORE will use [base+idx*scale] directly. */
      int base_reg = find_reg(reg_assigns, n_reg_assigns, u->src[0]);
      int idx_reg_check = resolve_index_to_gpr(&buf, u->src[1],
                                                reg_assigns, n_reg_assigns, &locals);
      /* Undo any code emitted by resolve (it was just a check) — actually
       * resolve_index_to_gpr may emit a load from slot. But for find_reg hit
       * (RANGE or R8-assigned SHL), no code is emitted. Safe for the check. */
      if (base_reg >= 0 && idx_reg_check >= 0 && valid_sib_scale(itemsize)) {
        /* Check all consumers: every consumer must be LOAD/STORE/CAST that
         * chains to LOAD/STORE (which can fuse via find_index_through_cast). */
        bool all_fuse = true;
        for (int j = i + 1; j < n && all_fuse; j++) {
          for (int k = 0; k < uops[j]->n_src; k++) {
            if (uops[j]->src[k] == u) {
              PolyOps cop = uops[j]->op;
              if (cop != POLY_OP_LOAD && cop != POLY_OP_STORE && cop != POLY_OP_CAST)
                all_fuse = false;
            }
          }
        }
        if (all_fuse) {
          LM_SET(u, slot);
          continue; /* skip INDEX emission */
        }
      }
      if (base_reg >= 0) {
        emit_alu_rr(&buf, 1, 0x8B, RAX, base_reg);
      } else if (base_slot != rax_slot) {
        emit_mov_r64_rbp(&buf, RAX, -slot_offset(base_slot));
      }

      int idx_reg = resolve_index_to_gpr(&buf, u->src[1],
                                          reg_assigns, n_reg_assigns, &locals);
      if (idx_reg >= 0) {
        if (idx_reg != RCX) emit_alu_rr(&buf, 1, 0x8B, RCX, idx_reg);
      } else {
        emit_mov_r64_rbp(&buf, RCX, -slot_offset(idx_slot));
      }

      if (itemsize == 1 || itemsize == 2 || itemsize == 4 || itemsize == 8) {
        emit_lea_sib(&buf, RAX, RAX, RCX, itemsize);
      } else {
        emit_imul_r64_imm32(&buf, RCX, RCX, itemsize);
        emit_alu_rr(&buf, 1, 0x03, RAX, RCX);
      }

      emit_mov_rbp_r64(&buf, RAX, off);
      rax_slot = slot;
      LM_SET(u, slot);
      continue;
    }

    /* ── LOAD: dereference pointer ───────────────────────────────── */
    if (u->op == POLY_OP_LOAD) {
      int slot = next_slot++;
      int off = -slot_offset(slot);

      int ptr_slot = lm_get(&locals, u->src[0]);
      if (ptr_slot < 0) goto skip_slot;

      /* Check if this is a register load (DEFINE_REG accumulator).
       * Walk through AFTER/INDEX chains to find the underlying DEFINE_REG. */
      {
        PolyUOp *acc = resolve_acc_base_fn(u->src[0]);
        if (acc) {
          int reg_slot = lm_get(&locals, acc);
          if (reg_slot >= 0) {
            LM_SET(u, reg_slot);
            next_slot--;
            continue;
          }
        }
      }

      /* Float LOAD (scalar or packed): use register file */
      if (dtype_is_float(u->dtype)) {
        bool packed = dtype_is_vec_float(u->dtype);

        /* Check for fusable INDEX (walk through pointer CAST/BITCAST) */
        PolyUOp *idx_uop = find_index_through_cast(u->src[0]);
        if (idx_uop) {
          int base_reg = find_reg(reg_assigns, n_reg_assigns, idx_uop->src[0]);
          int idx_reg = resolve_index_to_gpr(&buf, idx_uop->src[1],
                                              reg_assigns, n_reg_assigns, &locals);
          int is = poly_dtype_itemsize(idx_uop->src[0]->dtype);
          if (idx_uop->src[0]->dtype.is_ptr && idx_uop->src[0]->dtype.bitsize > 0)
            is = idx_uop->src[0]->dtype.bitsize / 8;
          if (base_reg >= 0 && idx_reg >= 0 && valid_sib_scale(is)) {
            /* Check if this LOAD can be deferred into a consumer ALU op.
             * Only for scalar (deferred SIB assumes stable idx_reg; SHL-computed
             * RCX may be clobbered between deferred emit and ALU consumption). */
            int n_consumers = 0;
            bool can_defer = false;
            for (int j = i + 1; j < n && n_consumers <= 1; j++) {
              PolyUOp *fj = uops[j];
              for (int k = 0; k < fj->n_src; k++) {
                if (fj->src[k] == u) {
                  n_consumers++;
                  if (k == 1 && !packed && poly_opset_has(POLY_GROUP_ALU, fj->op) &&
                      dtype_is_float(fj->dtype) && fj->n_src >= 2)
                    can_defer = true;
                }
              }
            }
            if (can_defer && n_consumers == 1 && n_deferred < 32) {
              deferred[n_deferred++] = (DeferredSIB){ u, base_reg, idx_reg, is };
              LM_SET(u, slot);
              continue;
            }

            /* Not deferred: emit fused SIB LOAD */
            int dst = xf_alloc(&xf, &buf, slot, -1, -1);
            if (packed && u->dtype.count >= 8)
              emit_vmovups_ymm_sib(&buf, dst, base_reg, idx_reg, is);
            else if (packed)
              emit_movups_xmm_sib(&buf, dst, base_reg, idx_reg, is);
            else
              emit_movss_xmm_sib(&buf, dst, base_reg, idx_reg, is);
            xf.e[dst - XF_BASE].dirty = true;
            xf.e[dst - XF_BASE].vec_width = packed ? u->dtype.count : 0;
            LM_SET(u, slot);
            continue;
          }
        }

        /* Non-fused float LOAD via pointer */
        {
          int dst2 = xf_alloc(&xf, &buf, slot, -1, -1);
          if (ptr_slot != rax_slot)
            emit_mov_r64_rbp(&buf, RAX, -slot_offset(ptr_slot));
          if (packed && u->dtype.count >= 8)
            emit_vmovups_ymm_mem(&buf, dst2, RAX);
          else if (packed)
            emit_movups_xmm_mem(&buf, dst2, RAX);
          else
            emit_movss_xmm_mem(&buf, dst2, RAX);
          xf.e[dst2 - XF_BASE].dirty = true;
          xf.e[dst2 - XF_BASE].vec_width = packed ? u->dtype.count : 0;
          LM_SET(u, slot);
          continue;
        }
      }

      /* Integer LOAD: use stack slots (no register file).
       * Check for fused SIB addressing (INDEX was skipped). */
      {
        PolyUOp *idx_uop = find_index_through_cast(u->src[0]);
        if (idx_uop) {
          int base_reg = find_reg(reg_assigns, n_reg_assigns, idx_uop->src[0]);
          int idx_reg = resolve_index_to_gpr(&buf, idx_uop->src[1],
                                              reg_assigns, n_reg_assigns, &locals);
          int is = poly_dtype_itemsize(idx_uop->src[0]->dtype);
          if (idx_uop->src[0]->dtype.is_ptr && idx_uop->src[0]->dtype.bitsize > 0)
            is = idx_uop->src[0]->dtype.bitsize / 8;
          if (base_reg >= 0 && idx_reg >= 0 && valid_sib_scale(is)) {
            /* SIB integer LOAD: mov r32/r64, [base + idx * scale] */
            int ss = 0;
            switch (is) { case 1:ss=0;break; case 2:ss=1;break; case 4:ss=2;break; case 8:ss=3;break; }
            bool w64 = (u->dtype.bitsize > 32);
            if (w64 || base_reg >= 8 || idx_reg >= 8)
              emit_rex_always(&buf, w64 ? 1 : 0, RAX >> 3, idx_reg >> 3, base_reg >> 3);
            xb_byte(&buf, 0x8B); /* mov r, [base+idx*scale] */
            if ((base_reg & 7) == RBP) {
              emit_modrm(&buf, 1, RAX, 4);
              xb_byte(&buf, (uint8_t)((ss << 6) | ((idx_reg & 7) << 3) | (base_reg & 7)));
              xb_byte(&buf, 0);
            } else {
              emit_modrm(&buf, 0, RAX, 4);
              xb_byte(&buf, (uint8_t)((ss << 6) | ((idx_reg & 7) << 3) | (base_reg & 7)));
            }
            if (w64) emit_mov_rbp_r64(&buf, RAX, off);
            else emit_mov_rbp_r32(&buf, RAX, off);
            rax_slot = slot;
            LM_SET(u, slot);
            continue;
          }
        }
      }
      /* Non-fused integer LOAD: dereference pointer from slot */
      if (ptr_slot != rax_slot)
        emit_mov_r64_rbp(&buf, RAX, -slot_offset(ptr_slot));
      if (u->dtype.bitsize <= 32) {
        emit_mov_r32_mem_base(&buf, RAX, RAX);
        emit_mov_rbp_r32(&buf, RAX, off);
      } else {
        emit_mov_r64_mem(&buf, RAX, RAX, 0);
        emit_mov_rbp_r64(&buf, RAX, off);
      }
      rax_slot = slot;
      LM_SET(u, slot);
      continue;
    }

    /* ── STORE: write value to memory or accumulator ─────────────── */
    if (u->op == POLY_OP_STORE) {
      int ptr_slot = lm_get(&locals, u->src[0]);
      int val_slot = lm_get(&locals, u->src[1]);
      if (ptr_slot < 0 || val_slot < 0) continue;

      /* Check if storing to a DEFINE_LOCAL/DEFINE_REG (accumulator).
       * Walk through AFTER/INDEX chains to find the underlying DEFINE_REG. */
      {
        PolyUOp *acc = resolve_acc_base_fn(u->src[0]);
        if (acc) {
          int acc_slot = lm_get(&locals, acc);
          if (acc_slot >= 0) {
            if (dtype_is_float(u->src[1]->dtype)) {
              int vr = xf_find(&xf, val_slot);
              if (vr < 0) { vr = XMM0; emit_movss_xmm_rbp(&buf, XMM0, -slot_offset(val_slot)); }
              emit_movss_rbp_xmm(&buf, vr, -slot_offset(acc_slot));
              /* Update register file: acc_slot now has this value */
              int ar = xf_find(&xf, acc_slot);
              if (ar >= 0) emit_movss_xmm_xmm(&buf, ar, vr);
            } else {
              emit_mov_r64_rbp(&buf, RAX, -slot_offset(val_slot));
              emit_mov_rbp_r64(&buf, RAX, -slot_offset(acc_slot));
              rax_slot = acc_slot;
            }
          }
          continue;
        }
      }

      if (dtype_is_float(u->src[1]->dtype)) {
        bool packed = dtype_is_vec_float(u->src[1]->dtype);
        int sw = u->src[1]->dtype.count;
        /* Get value from register file (or load from stack) */
        int vr = xf_find(&xf, val_slot);
        if (vr < 0) {
          vr = XMM0;
          if (packed && sw >= 8) emit_vmovups_ymm_rbp(&buf, XMM0, -slot_offset(val_slot));
          else if (packed) emit_movups_xmm_rbp(&buf, XMM0, -slot_offset(val_slot));
          else emit_movss_xmm_rbp(&buf, XMM0, -slot_offset(val_slot));
        }

        /* Check for fused INDEX store (walk through pointer CAST/BITCAST) */
        PolyUOp *st_idx = find_index_through_cast(u->src[0]);
        if (st_idx) {
          PolyUOp *idx_uop = st_idx;
          int base_reg = find_reg(reg_assigns, n_reg_assigns, idx_uop->src[0]);
          int idx_reg = resolve_index_to_gpr(&buf, idx_uop->src[1],
                                              reg_assigns, n_reg_assigns, &locals);
          int is = poly_dtype_itemsize(idx_uop->src[0]->dtype);
          if (idx_uop->src[0]->dtype.is_ptr && idx_uop->src[0]->dtype.bitsize > 0)
            is = idx_uop->src[0]->dtype.bitsize / 8;
          if (base_reg >= 0 && idx_reg >= 0 && valid_sib_scale(is)) {
            if (packed && sw >= 8) emit_vmovups_sib_ymm(&buf, vr, base_reg, idx_reg, is);
            else if (packed) emit_movups_sib_xmm(&buf, vr, base_reg, idx_reg, is);
            else emit_movss_sib_xmm(&buf, vr, base_reg, idx_reg, is);
            continue;
          }
        }

        /* Non-fused: store via pointer in RAX */
        if (ptr_slot != rax_slot)
          emit_mov_r64_rbp(&buf, RAX, -slot_offset(ptr_slot));
        if (packed && sw >= 8) emit_vmovups_mem_ymm(&buf, vr, RAX);
        else if (packed) emit_movups_mem_xmm(&buf, vr, RAX);
        else emit_movss_mem_xmm(&buf, vr, RAX);
      } else {
        if (u->src[1]->dtype.bitsize <= 32) {
          emit_mov_r32_rbp(&buf, RCX, -slot_offset(val_slot));
          if (RCX >= 8 || RAX >= 8) emit_rex(&buf, 0, RCX >> 3, 0, RAX >> 3);
          xb_byte(&buf, 0x89);
          emit_modrm(&buf, 0, RCX, RAX);
        } else {
          emit_mov_r64_rbp(&buf, RCX, -slot_offset(val_slot));
          emit_rex_always(&buf, 1, RCX >> 3, 0, RAX >> 3);
          xb_byte(&buf, 0x89);
          emit_modrm(&buf, 0, RCX, RAX);
        }
      }
      rax_slot = -1;
      continue;
    }

    /* ── RANGE: loop start ───────────────────────────────────────── */
    if (u->op == POLY_OP_RANGE) {
      int slot = next_slot++;
      LM_SET(u, slot);

      int bound_slot = lm_get(&locals, u->src[0]);
      if (bound_slot < 0) goto skip_slot;

      /* Check if bound is a compile-time constant */
      PolyUOp *bound_uop = u->src[0];
      bool bound_is_const = (bound_uop->op == POLY_OP_CONST &&
                             bound_uop->arg.kind == POLY_ARG_INT);
      int64_t bound_val = bound_is_const ? bound_uop->arg.i : 0;

      /* Try to assign a dedicated GPR for the loop counter */
      int gpr = -1;
      if (n_loop_regs < N_LOOP_GPRS) {
        gpr = LOOP_GPRS[n_loop_regs++];
        if (n_reg_assigns < MAX_REG_ASSIGNS)
          reg_assigns[n_reg_assigns++] = (RegAssign){ u, gpr };

        emit_alu_rr(&buf, 1, 0x33, gpr, gpr);
      } else {
        emit_mov_rbp_imm64_sx(&buf, -slot_offset(slot), 0);
      }

      /* Invalidate caches at loop entry */
      rax_slot = -1;
      xmm0_slot = -1;
      xf_clear(&xf);

      /* Loop condition check (backward jump target) */
      int loop_start = buf.len;

      if (gpr >= 0) {
        if (bound_is_const && bound_val >= INT32_MIN && bound_val <= INT32_MAX) {
          emit_cmp_r64_imm32(&buf, gpr, (int32_t)bound_val);
        } else {
          emit_alu_r_rbp(&buf, 1, 0x3B, gpr, -slot_offset(bound_slot));
        }
      } else {
        emit_mov_r64_rbp(&buf, RAX, -slot_offset(slot));
        rax_slot = -1;
        emit_cmp_r64_rbp(&buf, RAX, -slot_offset(bound_slot));
      }

      int jge_off = emit_jge_rel32(&buf, 0);

      if (loop_depth < MAX_LOOP_DEPTH) {
        loop_stack[loop_depth++] = (LoopPatch){
          .range = u,
          .jge_disp_offset = jge_off,
          .loop_body_start = loop_start,
          .counter_slot = slot,
          .gpr = gpr,
        };
      }

      continue;
    }

    /* ── END: loop close ─────────────────────────────────────────── */
    if (u->op == POLY_OP_END) {
      /* Find matching RANGE — END.src[1] is the RANGE being closed */
      PolyUOp *range = (u->n_src >= 2) ? u->src[1] : NULL;
      int match = -1;
      for (int d = loop_depth - 1; d >= 0; d--) {
        if (loop_stack[d].range == range) { match = d; break; }
      }
      if (match < 0) continue;

      LoopPatch lp = loop_stack[match];
      /* Remove from stack (shift down) */
      for (int d = match; d < loop_depth - 1; d++)
        loop_stack[d] = loop_stack[d + 1];
      loop_depth--;

      /* Invalidate caches before backward jump */
      rax_slot = -1;
      xmm0_slot = -1;
      xf_clear(&xf); /* discard iteration-local values (no spill needed) */

      /* Increment counter */
      if (lp.gpr >= 0) {
        emit_inc_r64(&buf, lp.gpr);
      } else {
        emit_inc_rbp_q(&buf, -slot_offset(lp.counter_slot));
      }

      /* Jump back to loop start */
      int jmp_off = emit_jmp_rel32(&buf, 0);
      patch_rel32(&buf, jmp_off, lp.loop_body_start);

      /* Patch the forward jge to point here (after the jmp) */
      patch_rel32(&buf, lp.jge_disp_offset, buf.len);

      /* Release GPR when loop exits */
      if (lp.gpr >= 0 && n_loop_regs > 0) n_loop_regs--;
      continue;
    }

    /* ── IF: conditional forward jump ────────────────────────────── */
    if (u->op == POLY_OP_IF) {
      int cond_slot = lm_get(&locals, u->src[0]);
      if (cond_slot < 0) continue;

      /* Load condition, test, jump if zero */
      emit_mov_r32_rbp(&buf, RAX, -slot_offset(cond_slot));
      emit_test_r32(&buf, RAX, RAX);
      int je_off = emit_je_rel32(&buf, 0);

      if (if_depth < MAX_LOOP_DEPTH)
        if_patch_stack[if_depth++] = je_off;
      continue;
    }

    /* ── ENDIF: patch IF forward jump ────────────────────────────── */
    if (u->op == POLY_OP_ENDIF) {
      if (if_depth > 0) {
        int je_off = if_patch_stack[--if_depth];
        patch_rel32(&buf, je_off, buf.len);
      }
      continue;
    }

    /* ── CAST: type conversion ───────────────────────────────────── */
    if (u->op == POLY_OP_CAST) {
      int s0 = lm_get(&locals, u->src[0]);
      if (s0 < 0) { next_slot++; goto skip_slot; }

      bool src_float = dtype_is_float(u->src[0]->dtype);
      bool dst_float = dtype_is_float(u->dtype);

      /* Pointer reinterpretation (same bits, different ptr type): alias */
      if ((src_float == dst_float) && u->src[0]->dtype.is_ptr && u->dtype.is_ptr) {
        LM_SET(u, s0); /* alias to source slot */
        continue;
      }

      int slot = next_slot++;
      int off = -slot_offset(slot);

      if (src_float && !dst_float) {
        /* float → int: spill XMM if dirty, then cvttss2si */
        int xr = xf_find(&xf, s0);
        if (xr >= 0 && xf.e[xr - XF_BASE].dirty) {
          emit_width_store_rbp(&buf, xr, -slot_offset(s0), xf.e[xr - XF_BASE].vec_width);
          xf.e[xr - XF_BASE].dirty = false;
        }
        emit_movss_xmm_rbp(&buf, XMM0, -slot_offset(s0));
        if (u->dtype.bitsize > 32) {
          /* cvttss2si rax, xmm0 (64-bit): F3 REX.W 0F 2C /r */
          xb_byte(&buf, 0xF3);
          emit_rex_always(&buf, 1, 0, 0, 0);
          xb_byte(&buf, 0x0F); xb_byte(&buf, 0x2C);
          emit_modrm(&buf, 3, RAX, XMM0);
        } else {
          emit_cvttss2si(&buf, RAX, XMM0);
        }
        emit_mov_rbp_r64(&buf, RAX, off);
        RELOAD_SIGN_MASK();
      } else if (!src_float && dst_float) {
        /* int → float: load with correct width, cvtsi2ss */
        if (u->src[0]->dtype.bitsize > 32) {
          emit_mov_r64_rbp(&buf, RAX, -slot_offset(s0));
          /* cvtsi2ss xmm0, rax (64-bit source) */
          xb_byte(&buf, 0xF3);
          emit_rex_always(&buf, 1, 0, 0, 0);
          xb_byte(&buf, 0x0F); xb_byte(&buf, 0x2A);
          emit_modrm(&buf, 3, XMM0, RAX);
        } else {
          emit_mov_r32_rbp(&buf, RAX, -slot_offset(s0));
          emit_cvtsi2ss(&buf, XMM0, RAX);
        }
        emit_movss_rbp_xmm(&buf, XMM0, off);
        RELOAD_SIGN_MASK();
      } else {
        /* int → int: width-and-signedness-correct extension/truncation */
        int src_bits = u->src[0]->dtype.bitsize;
        int dst_bits = u->dtype.bitsize;
        bool src_unsigned = poly_dtype_is_unsigned(u->src[0]->dtype);
        if (dst_bits > src_bits) {
          /* Widening: zero-extend or sign-extend */
          if (src_bits <= 32) {
            if (src_unsigned) {
              /* Zero-extend: mov eax, [slot] (implicit zero-extension to 64-bit) */
              emit_mov_r32_rbp(&buf, RAX, -slot_offset(s0));
            } else {
              /* Sign-extend: movsxd rax, [slot] */
              emit_mov_r32_rbp(&buf, RAX, -slot_offset(s0));
              emit_movsxd(&buf, RAX, RAX);
            }
          } else {
            emit_mov_r64_rbp(&buf, RAX, -slot_offset(s0));
          }
        } else {
          /* Narrowing or same-width: just copy */
          if (dst_bits <= 32)
            emit_mov_r32_rbp(&buf, RAX, -slot_offset(s0));
          else
            emit_mov_r64_rbp(&buf, RAX, -slot_offset(s0));
        }
        if (dst_bits <= 32)
          emit_mov_rbp_r32(&buf, RAX, off);
        else
          emit_mov_rbp_r64(&buf, RAX, off);
        rax_slot = slot; xmm0_slot = -1;
      }
      LM_SET(u, slot);
      continue;
    }

    /* ── BITCAST: reinterpret bits across float/int domains ──────── */
    if (u->op == POLY_OP_BITCAST) {
      int slot = next_slot++;
      int off = -slot_offset(slot);
      int s0 = lm_get(&locals, u->src[0]);
      if (s0 < 0) goto skip_slot;

      bool src_float = dtype_is_float(u->src[0]->dtype);
      int src_bits = u->src[0]->dtype.count > 1
        ? (u->src[0]->dtype.bitsize / u->src[0]->dtype.count)
        : u->src[0]->dtype.bitsize;
      bool wide = (src_bits > 32);

      /* If source is in XMM register file (float), spill to stack first */
      if (src_float) {
        int xr = xf_find(&xf, s0);
        if (xr >= 0 && xf.e[xr - XF_BASE].dirty) {
          emit_width_store_rbp(&buf, xr, -slot_offset(s0), xf.e[xr - XF_BASE].vec_width);
          xf.e[xr - XF_BASE].dirty = false;
        }
      }

      /* Width-correct copy: use 32-bit moves for 32-bit types to avoid
       * reading/writing garbage in upper bits. Critical for pow2if where
       * ((q+127)<<23) must be an exact 32-bit pattern. */
      if (wide) {
        emit_mov_r64_rbp(&buf, RAX, -slot_offset(s0));
        emit_mov_rbp_r64(&buf, RAX, off);
      } else {
        emit_mov_r32_rbp(&buf, RAX, -slot_offset(s0));
        emit_mov_rbp_r32(&buf, RAX, off);
      }
      rax_slot = slot; xmm0_slot = -1;
      LM_SET(u, slot);
      continue;
    }

    /* ── GEP: extract scalar lane from vector register ──────────── */
    if (u->op == POLY_OP_GEP && u->n_src >= 1 && dtype_is_float(u->dtype) &&
        u->dtype.count == 1 && u->src[0]->dtype.count > 1) {
      int slot = next_slot++;
      int off = -slot_offset(slot);
      (void)off;
      int lane = 0;
      if (u->arg.kind == POLY_ARG_INT) lane = (int)u->arg.i;
      else if (u->arg.kind == POLY_ARG_INT_TUPLE && u->arg.int_tuple.n == 1)
        lane = (int)u->arg.int_tuple.vals[0];

      /* Get source vec from register file */
      int sr = xf_find(&xf, lm_get(&locals, u->src[0]));
      if (sr < 0) sr = xf_get_packed_w(&xf, &buf, lm_get(&locals, u->src[0]), u->src[0]->dtype.count);

      /* Allocate scalar destination (avoid evicting source) */
      int dr = xf_alloc(&xf, &buf, slot, sr, -1);

      if (use_avx2 && u->src[0]->dtype.count >= 8 && lane >= 4) {
        /* High 128 bits of YMM: vextractf128 to XMM0, then pshufd */
        emit_vextractf128(&buf, XMM0, sr, 1);
        int hilo_lane = lane - 4;
        if (hilo_lane == 0) {
          emit_movss_xmm_xmm(&buf, dr, XMM0);
        } else {
          uint8_t imm = (uint8_t)(hilo_lane | (hilo_lane << 2) | (hilo_lane << 4) | (hilo_lane << 6));
          /* pshufd: 66 0F 70 /r ib */
          xb_byte(&buf, 0x66);
          if (dr >= 8) emit_rex(&buf, 0, dr >> 3, 0, 0);
          xb_byte(&buf, 0x0F); xb_byte(&buf, 0x70);
          emit_modrm(&buf, 3, dr, XMM0);
          xb_byte(&buf, imm);
        }
      } else if (lane == 0) {
        /* Lane 0: just movss (low 32 bits) */
        emit_movss_xmm_xmm(&buf, dr, sr);
      } else {
        /* pshufd dst, src, imm8: move lane to position 0
         * imm8 selects source lane for each output lane. We only care about lane 0.
         * imm8 = lane | (lane<<2) | (lane<<4) | (lane<<6) broadcasts the lane. */
        uint8_t imm = (uint8_t)(lane | (lane << 2) | (lane << 4) | (lane << 6));
        if (use_avx2) {
          emit_vpshufd(&buf, dr, sr, imm, 0); /* VEX.128 pshufd */
        } else {
          /* pshufd: 66 0F 70 /r ib */
          xb_byte(&buf, 0x66);
          if (dr >= 8 || sr >= 8) emit_rex(&buf, 0, dr >> 3, 0, sr >> 3);
          xb_byte(&buf, 0x0F); xb_byte(&buf, 0x70);
          emit_modrm(&buf, 3, dr, sr);
          xb_byte(&buf, imm);
        }
      }
      xf.e[dr - XF_BASE].dirty = true;
      xf.e[dr - XF_BASE].vec_width = 0; /* result is scalar */
      LM_SET(u, slot);
      continue;
    }

    /* ── VECTORIZE: construct vector from scalar values ──────────── */
    if (u->op == POLY_OP_VECTORIZE && u->n_src >= 2 && dtype_is_float(u->dtype) &&
        u->dtype.count > 1) {
      int slot = next_slot++;
      int n_lanes = u->n_src;
      int voff = -slot_offset(slot);

      if (n_lanes > 4 && n_lanes <= 8) {
        /* 5-8 wide: store all scalars to stack, load as YMM vector.
         * Zero the full 32-byte slot first to avoid reading uninitialized
         * bytes when n_lanes < 8 (e.g. 5, 6, 7). */
        if (n_lanes < 8) {
          /* Zero the slot via two 64-bit zero stores */
          emit_mov_rbp_imm64_sx(&buf, voff, 0);
          emit_mov_rbp_imm64_sx(&buf, voff + 8, 0);
          emit_mov_rbp_imm64_sx(&buf, voff + 16, 0);
          emit_mov_rbp_imm64_sx(&buf, voff + 24, 0);
        }
        for (int j = 0; j < n_lanes; j++) {
          int sj = lm_get(&locals, u->src[j]);
          int srj = (sj >= 0) ? xf_find(&xf, sj) : -1;
          if (srj < 0 && sj >= 0) srj = xf_get(&xf, &buf, sj);
          if (srj >= 0) emit_movss_rbp_xmm(&buf, srj, voff + j * 4);
        }
        int dr = xf_alloc(&xf, &buf, slot, -1, -1);
        emit_vmovups_ymm_rbp(&buf, dr, voff);
        xf.e[dr - XF_BASE].dirty = true;
        xf.e[dr - XF_BASE].vec_width = n_lanes;
        RELOAD_SIGN_MASK();
        LM_SET(u, slot);
        continue;
      }

      /* Get all scalar sources from register file */
      int sr[4] = {-1, -1, -1, -1};
      for (int j = 0; j < n_lanes && j < 4; j++) {
        int sj = lm_get(&locals, u->src[j]);
        sr[j] = (sj >= 0) ? xf_find(&xf, sj) : -1;
        if (sr[j] < 0 && sj >= 0) sr[j] = xf_get(&xf, &buf, sj);
      }

      /* Allocate packed destination */
      int dr = xf_alloc(&xf, &buf, slot, sr[0], sr[1]);

      if (n_lanes == 4) {
        /* SSE2 construction: unpcklps + movlhps
         * Step 1: xmm_tmp0 = {s0, s1, ?, ?} via unpcklps
         * Step 2: xmm_tmp1 = {s2, s3, ?, ?} via unpcklps
         * Step 3: result = {s0, s1, s2, s3} via movlhps */

        /* Use XMM0 as scratch for the second pair */
        /* Copy s0 to dr, interleave with s1 */
        if (dr != sr[0]) emit_movss_xmm_xmm(&buf, dr, sr[0]);
        /* unpcklps dr, sr[1]: dr = {dr[0], sr1[0], dr[1], sr1[1]} = {s0, s1, ?, ?} */
        if (sr[1] >= 8 || dr >= 8) emit_rex(&buf, 0, dr >> 3, 0, sr[1] >> 3);
        xb_byte(&buf, 0x0F); xb_byte(&buf, 0x14);
        emit_modrm(&buf, 3, dr, sr[1]);

        /* Build second pair in XMM0 */
        emit_movss_xmm_xmm(&buf, XMM0, sr[2]);
        /* unpcklps xmm0, sr[3] */
        if (sr[3] >= 8) emit_rex(&buf, 0, 0, 0, sr[3] >> 3);
        xb_byte(&buf, 0x0F); xb_byte(&buf, 0x14);
        emit_modrm(&buf, 3, XMM0, sr[3]);

        /* movlhps dr, xmm0: dr = {dr[0], dr[1], xmm0[0], xmm0[1]} = {s0, s1, s2, s3} */
        if (dr >= 8) emit_rex(&buf, 0, dr >> 3, 0, 0);
        xb_byte(&buf, 0x0F); xb_byte(&buf, 0x16);
        emit_modrm(&buf, 3, dr, XMM0);
      } else {
        /* Fallback for non-4-wide: store to stack slots, load as packed */
        for (int j = 0; j < n_lanes; j++) {
          if (sr[j] >= 0) emit_movss_rbp_xmm(&buf, sr[j], voff + j * 4);
        }
        emit_movups_xmm_rbp(&buf, dr, voff);
      }

      xf.e[dr - XF_BASE].dirty = true;
      xf.e[dr - XF_BASE].vec_width = 4;
      RELOAD_SIGN_MASK();
      LM_SET(u, slot);
      continue;
    }

    /* ── ALU operations ──────────────────────────────────────────── */
    if (poly_opset_has(POLY_GROUP_ALU, u->op)) {
      int slot = next_slot++;
      int off = -slot_offset(slot);

      /* Determine if this is a float or int operation */
      bool is_float_op = dtype_is_float(u->dtype);
      /* For comparisons, use input dtype */
      if (u->op == POLY_OP_CMPLT || u->op == POLY_OP_CMPEQ || u->op == POLY_OP_CMPNE)
        is_float_op = u->n_src > 0 && dtype_is_float(u->src[0]->dtype);

      if (is_float_op) {
        /* ── Float ALU ──────────────────────────────────────────── */
        int s0 = (u->n_src > 0) ? lm_get(&locals, u->src[0]) : -1;
        int s1 = (u->n_src > 1) ? lm_get(&locals, u->src[1]) : -1;
        int s2 = (u->n_src > 2) ? lm_get(&locals, u->src[2]) : -1;

        /* ── Register-file float ops (scalar or packed SSE) ────── */
        bool pk = dtype_is_vec_float(u->dtype);
        /* For comparisons, check input dtype */
        if (u->op == POLY_OP_CMPLT || u->op == POLY_OP_CMPEQ || u->op == POLY_OP_CMPNE)
          pk = u->n_src > 0 && dtype_is_vec_float(u->src[0]->dtype);

        /* Get src0 from register file */
        int sr0 = (s0 >= 0) ? xf_find(&xf, s0) : -1;
        if (sr0 < 0 && s0 >= 0) {
          sr0 = pk ? xf_get_packed_w(&xf, &buf, s0, u->dtype.count) : xf_get(&xf, &buf, s0);
        }

        /* For binary ops: check if src1 was deferred (can fuse as SIB operand) */
        DeferredSIB *ds1 = NULL;
        if (u->n_src > 1) {
          for (int d = 0; d < n_deferred; d++)
            if (deferred[d].uop == u->src[1]) { ds1 = &deferred[d]; break; }
        }

        /* Get src1 from register file (only if not deferred) */
        int sr1 = -1;
        if (!ds1 && s1 >= 0 && u->n_src > 1 &&
            u->op != POLY_OP_NEG && u->op != POLY_OP_SQRT &&
            u->op != POLY_OP_RECIPROCAL && u->op != POLY_OP_TRUNC &&
            u->op != POLY_OP_EXP2 && u->op != POLY_OP_LOG2 &&
            u->op != POLY_OP_SIN) {
          sr1 = pk ? xf_get_packed_w(&xf, &buf, s1, u->dtype.count) : xf_get_avoid(&xf, &buf, s1, sr0);
        }

        /* Allocate destination register.
         * Optimization: reuse sr0 if it's a register-file entry whose slot
         * won't be needed again (the source UOp has no other consumers after this).
         * Simple heuristic: if sr0's slot is not in the locals map for any
         * future UOp, reuse it. Conservative fallback: always reuse sr0 for
         * binary ops (the register file will reload if needed later). */
        int dr;
        if (sr0 >= XF_BASE && sr0 < XF_BASE + XF_SIZE &&
            (u->op == POLY_OP_ADD || u->op == POLY_OP_SUB || u->op == POLY_OP_MUL ||
             u->op == POLY_OP_FDIV || u->op == POLY_OP_MAX || u->op == POLY_OP_MULACC ||
             u->op == POLY_OP_NEG || u->op == POLY_OP_SQRT) &&
            u->op != POLY_OP_RECIPROCAL) {
          /* Reuse sr0's register as destination — saves a movss copy.
           * Only spill if src0 is used again later (check remaining UOps). */
          int si = sr0 - XF_BASE;
          if (xf.e[si].dirty) {
            /* Check if src0's UOp is referenced by any later UOp */
            bool src0_used_later = false;
            PolyUOp *src0_uop = u->src[0];
            for (int j = i + 1; j < n && !src0_used_later; j++) {
              PolyUOp *fj = uops[j];
              for (int k = 0; k < fj->n_src; k++)
                if (fj->src[k] == src0_uop) { src0_used_later = true; break; }
            }
            if (src0_used_later) {
              emit_width_store_rbp(&buf, sr0,
                                   -slot_offset(xf.e[si].slot), xf.e[si].vec_width);
            }
            xf.e[si].dirty = false;
          }
          xf.e[si].slot = slot;
          dr = sr0;
        } else {
          dr = xf_alloc(&xf, &buf, slot, sr0, sr1 >= 0 ? sr1 : -1);
        }

        switch (u->op) {
          /* Binary: copy s0→dst, op with s1 */
          case POLY_OP_ADD:
          case POLY_OP_SUB:
          case POLY_OP_MUL:
          case POLY_OP_FDIV:
          case POLY_OP_MAX: {
            bool pk = dtype_is_vec_float(u->dtype);
            uint8_t opc = 0;
            switch (u->op) {
              case POLY_OP_ADD:  opc = 0x58; break;
              case POLY_OP_SUB:  opc = 0x5C; break;
              case POLY_OP_MUL:  opc = 0x59; break;
              case POLY_OP_FDIV: opc = 0x5E; break;
              case POLY_OP_MAX:  opc = 0x5F; break;
              default: break;
            }
            if (pk && use_avx2) {
              /* AVX 3-operand: dst = sr0 op sr1 (non-destructive) */
              int vL = (u->dtype.count >= 8) ? 1 : 0;
              if (ds1)
                emit_vex_packed_rr_sib(&buf, opc, dr, sr0, ds1->base_reg, ds1->idx_reg, ds1->itemsize, vL);
              else
                emit_vex_packed_rrr(&buf, opc, dr, sr0, sr1, vL);
            } else if (pk) {
              if (dr != sr0) emit_movups_xmm_xmm(&buf, dr, sr0);
              if (ds1)
                emit_sse_packed_sib(&buf, opc, dr, ds1->base_reg, ds1->idx_reg, ds1->itemsize);
              else
                emit_sse_packed_rr(&buf, opc, dr, sr1);
            } else {
              if (dr != sr0) emit_movss_xmm_xmm(&buf, dr, sr0);
              if (ds1)
                emit_sse_scalar_sib(&buf, opc, dr, ds1->base_reg, ds1->idx_reg, ds1->itemsize);
              else
                emit_sse_scalar_rr(&buf, opc, dr, sr1);
            }
            break;
          }

          /* Unary */
          case POLY_OP_NEG:
            /* XOR with sign mask (preloaded in XMM0 in prologue) */
            if (use_avx2 && dtype_is_vec_float(u->dtype)) {
              int vL = (u->dtype.count >= 8) ? 1 : 0;
              emit_vxorps(&buf, dr, sr0, XMM0, vL);
            } else {
              if (dr != sr0) {
                if (dtype_is_vec_float(u->dtype))
                  emit_movups_xmm_xmm(&buf, dr, sr0);
                else
                  emit_movss_xmm_xmm(&buf, dr, sr0);
              }
              emit_xorps(&buf, dr, XMM0);
            }
            break;
          case POLY_OP_SQRT:
            if (use_avx2 && dtype_is_vec_float(u->dtype))
              emit_vex_packed_sqrt(&buf, dr, sr0, (u->dtype.count >= 8) ? 1 : 0);
            else if (dtype_is_vec_float(u->dtype))
              emit_sse_packed_rr(&buf, 0x51, dr, sr0);
            else
              emit_sse_scalar_rr(&buf, 0x51, dr, sr0);
            break;
          case POLY_OP_RECIPROCAL: {
            /* 1.0 / x: load 1.0f into XMM0 (scratch), divide by sr0, store in dr */
            emit_mov_r32_imm32(&buf, RAX, 0x3F800000);
            emit_movd_xmm_r32(&buf, XMM0, RAX);
            if (use_avx2 && dtype_is_vec_float(u->dtype)) {
              int vL = (u->dtype.count >= 8) ? 1 : 0;
              if (vL) emit_vbroadcastss_ymm_xmm(&buf, XMM0, XMM0);
              else { xb_byte(&buf, 0x0F); xb_byte(&buf, 0xC6);
                     emit_modrm(&buf, 3, XMM0, XMM0); xb_byte(&buf, 0x00); }
              emit_vex_packed_rrr(&buf, 0x5E, dr, XMM0, sr0, vL);
            } else if (dtype_is_vec_float(u->dtype)) {
              /* Broadcast 1.0f to all lanes: shufps xmm0, xmm0, 0 */
              xb_byte(&buf, 0x0F); xb_byte(&buf, 0xC6);
              emit_modrm(&buf, 3, XMM0, XMM0); xb_byte(&buf, 0x00);
              emit_sse_packed_rr(&buf, 0x5E, XMM0, sr0); /* xmm0 = 1.0/sr0 */
              if (dr != XMM0) emit_movups_xmm_xmm(&buf, dr, XMM0);
            } else {
              emit_sse_scalar_rr(&buf, 0x5E, XMM0, sr0);
              if (dr != XMM0) emit_movss_xmm_xmm(&buf, dr, XMM0);
            }
            RELOAD_SIGN_MASK();
            break;
          }
          case POLY_OP_TRUNC:
            if (dr != sr0) emit_movss_xmm_xmm(&buf, dr, sr0);
            emit_cvttss2si(&buf, RAX, dr);
            emit_cvtsi2ss(&buf, dr, RAX);
            rax_slot = -1;
            break;

          /* Comparisons: use XMM0 as scratch, result is int → goes to stack */
          case POLY_OP_CMPLT:
          case POLY_OP_CMPEQ:
          case POLY_OP_CMPNE: {
            uint8_t pred = u->op == POLY_OP_CMPLT ? 1 : u->op == POLY_OP_CMPEQ ? 0 : 4;
            emit_movss_xmm_xmm(&buf, XMM0, sr0);
            emit_cmpss_rr(&buf, XMM0, sr1, pred);
            emit_movd_r32_xmm(&buf, RAX, XMM0);
            emit_and_r32_imm32(&buf, RAX, 1);
            emit_mov_rbp_r64(&buf, RAX, off);
            /* Free the destination XMM since result is int (in stack) */
            xf.e[dr - XF_BASE].slot = -1;
            xf.e[dr - XF_BASE].dirty = false;
            RELOAD_SIGN_MASK();
            break;
          }

          /* Ternary */
          case POLY_OP_WHERE: {
            /* cond is int (in stack), true/false are float (in reg file).
             * Load both sources first, then revalidate dr to prevent
             * eviction of the pre-allocated destination. */
            int sr_true = xf_find(&xf, s1);
            if (sr_true < 0) sr_true = xf_get(&xf, &buf, s1);
            int sr_false = xf_find(&xf, s2);
            if (sr_false < 0) sr_false = xf_get_avoid(&xf, &buf, s2, sr_true);
            /* Revalidate sr_true after sr_false load (may have evicted it) */
            int sr_true_check = xf_find(&xf, s1);
            if (sr_true_check < 0) sr_true = xf_get_avoid(&xf, &buf, s1, sr_false);
            else sr_true = sr_true_check;
            /* Revalidate dr: loading sr_true/sr_false may have evicted it */
            if (xf_find(&xf, slot) < 0)
              dr = xf_alloc_belady(&xf, &buf, slot, sr_true, sr_false, -1,
                                    slot_last_use, i);
            emit_mov_r32_rbp(&buf, RAX, -slot_offset(s0)); /* cond from stack */
            emit_neg_r32(&buf, RAX);
            emit_movd_xmm_r32(&buf, XMM0, RAX);
            if (dr != sr_true) emit_movss_xmm_xmm(&buf, dr, sr_true);
            emit_andps(&buf, dr, XMM0);
            emit_andnps(&buf, XMM0, sr_false);
            emit_orps(&buf, dr, XMM0);
            rax_slot = -1;
            /* WHERE clobbers XMM0 (used for sign mask); reload */
            RELOAD_SIGN_MASK();
            break;
          }

          case POLY_OP_MULACC: {
            /* MULACC: dst = src0 * src1 + src2
             * Load src2 first, then re-allocate dr protecting all three
             * source registers to prevent eviction conflicts. */
            int sr_s2 = xf_find(&xf, s2);
            if (sr_s2 < 0) {
              /* Load sr_s2 avoiding eviction of sr0 and sr1 */
              if (pk)
                sr_s2 = xf_get_packed_w(&xf, &buf, s2, u->dtype.count);
              else {
                sr_s2 = xf_alloc(&xf, &buf, s2, sr0, sr1 >= 0 ? sr1 : -1);
                emit_movss_xmm_rbp(&buf, sr_s2, -slot_offset(s2));
              }
            }
            /* Re-allocate dr protecting all 3 source registers.
             * Pre-switch dr may have been evicted by sr_s2 load;
             * sr1 must also be protected from eviction. */
            if (xf_find(&xf, slot) < 0) {
              dr = xf_alloc_belady(&xf, &buf, slot, sr0, sr1, sr_s2,
                                    slot_last_use, i);
            }
            if (use_avx2 && cpu.has_fma) {
              int vL = pk ? ((u->dtype.count >= 8) ? 1 : 0) : 0;
              /* vfmadd231ps/ss: dst += sr0 * sr1 (dst must hold src2) */
              if (dr != sr_s2) {
                if (pk && vL) emit_vmovups_ymm_ymm(&buf, dr, sr_s2);
                else if (pk) emit_movups_xmm_xmm(&buf, dr, sr_s2);
                else emit_movss_xmm_xmm(&buf, dr, sr_s2);
              }
              emit_vfmadd231(&buf, dr, sr0, sr1, vL, !pk);
            } else if (pk) {
              if (dr != sr0) emit_movups_xmm_xmm(&buf, dr, sr0);
              emit_sse_packed_rr(&buf, 0x59, dr, sr1); /* mulps */
              emit_sse_packed_rr(&buf, 0x58, dr, sr_s2); /* addps */
            } else {
              if (dr != sr0) emit_movss_xmm_xmm(&buf, dr, sr0);
              emit_sse_scalar_rr(&buf, 0x59, dr, sr1); /* mulss */
              emit_sse_scalar_rr(&buf, 0x58, dr, sr_s2); /* addss */
            }
            break;
          }

          /* Transcendentals via libm — flush register file before call */
          case POLY_OP_EXP2:
          case POLY_OP_LOG2:
          case POLY_OP_SIN: {
            void *fn_addr = NULL;
            if (u->op == POLY_OP_EXP2) fn_addr = (void*)(uintptr_t)exp2f;
            else if (u->op == POLY_OP_LOG2) fn_addr = (void*)(uintptr_t)log2f;
            else fn_addr = (void*)(uintptr_t)sinf;
            xf_flush(&xf, &buf); /* caller-saved XMMs clobbered by call */
            emit_movss_xmm_rbp(&buf, XMM0, -slot_offset(s0));
            emit_mov_r64_imm64(&buf, RAX, (int64_t)(uintptr_t)fn_addr);
            emit_call_r64(&buf, RAX);
            /* Result in XMM0, put in register file */
            dr = xf_alloc(&xf, &buf, slot, -1, -1);
            emit_movss_xmm_xmm(&buf, dr, XMM0);
            break;
          }
          case POLY_OP_POW: {
            xf_flush(&xf, &buf);
            emit_movss_xmm_rbp(&buf, XMM0, -slot_offset(s0));
            emit_movss_xmm_rbp(&buf, XMM1, -slot_offset(s1));
            emit_mov_r64_imm64(&buf, RAX, (int64_t)(uintptr_t)powf);
            emit_call_r64(&buf, RAX);
            dr = xf_alloc(&xf, &buf, slot, -1, -1);
            emit_movss_xmm_xmm(&buf, dr, XMM0);
            break;
          }

          default:
            fprintf(stderr, "x64 jit: unhandled float ALU op %s at index %d\n",
                    poly_op_name(u->op), i);
            goto x64_fail;
        }
        /* Mark destination register as dirty (value not in stack) */
        if (u->op != POLY_OP_CMPLT && u->op != POLY_OP_CMPEQ &&
            u->op != POLY_OP_CMPNE) {
          xf.e[dr - XF_BASE].dirty = true;
          xf.e[dr - XF_BASE].vec_width = pk ? u->dtype.count : 0;
        }
      } else {
        /* ── Integer ALU ────────────────────────────────────────── */
        int s0 = (u->n_src > 0) ? lm_get(&locals, u->src[0]) : -1;
        int s1 = (u->n_src > 1) ? lm_get(&locals, u->src[1]) : -1;
        int s2 = (u->n_src > 2) ? lm_get(&locals, u->src[2]) : -1;
        bool wide = u->dtype.bitsize > 32;

        /* For comparisons, use src dtype width */
        if (u->op == POLY_OP_CMPLT || u->op == POLY_OP_CMPEQ || u->op == POLY_OP_CMPNE)
          wide = u->n_src > 0 && u->src[0]->dtype.bitsize > 32;

        int w = wide ? 1 : 0;

        switch (u->op) {
          case POLY_OP_ADD:
            emit_alu_r_rbp(&buf, w, 0x8B, RAX, -slot_offset(s0)); /* mov */
            emit_alu_r_rbp(&buf, w, 0x03, RAX, -slot_offset(s1)); /* add */
            emit_mov_rbp_r64(&buf, RAX, off);
            break;
          case POLY_OP_SUB:
            emit_alu_r_rbp(&buf, w, 0x8B, RAX, -slot_offset(s0));
            emit_alu_r_rbp(&buf, w, 0x2B, RAX, -slot_offset(s1)); /* sub */
            emit_mov_rbp_r64(&buf, RAX, off);
            break;
          case POLY_OP_MUL:
            emit_alu_r_rbp(&buf, w, 0x8B, RAX, -slot_offset(s0));
            emit_alu_r_rbp(&buf, w, 0x8B, RCX, -slot_offset(s1));
            if (wide) emit_imul_r64_r64(&buf, RAX, RCX);
            else emit_imul_r32_r32(&buf, RAX, RCX);
            emit_mov_rbp_r64(&buf, RAX, off);
            break;
          case POLY_OP_IDIV:
            emit_alu_r_rbp(&buf, w, 0x8B, RAX, -slot_offset(s0));
            emit_alu_r_rbp(&buf, w, 0x8B, RCX, -slot_offset(s1));
            if (wide) { emit_cqo(&buf); emit_idiv_r64(&buf, RCX); }
            else { emit_cdq(&buf); emit_idiv_r32(&buf, RCX); }
            emit_mov_rbp_r64(&buf, RAX, off); /* quotient in RAX */
            break;
          case POLY_OP_MOD:
            emit_alu_r_rbp(&buf, w, 0x8B, RAX, -slot_offset(s0));
            emit_alu_r_rbp(&buf, w, 0x8B, RCX, -slot_offset(s1));
            if (wide) { emit_cqo(&buf); emit_idiv_r64(&buf, RCX); }
            else { emit_cdq(&buf); emit_idiv_r32(&buf, RCX); }
            emit_mov_rbp_r64(&buf, RDX, off); /* remainder in RDX */
            break;
          case POLY_OP_SHL: {
            /* Check if src[0] is RANGE_in_reg and src[1] is CONST:
             * compute into R8 (caller-saved, stable in loop body) for reuse. */
            int src_gpr = find_reg(reg_assigns, n_reg_assigns, u->src[0]);
            bool const_shift = (u->n_src > 1 && u->src[1]->op == POLY_OP_CONST &&
                                u->src[1]->arg.kind == POLY_ARG_INT);
            if (src_gpr >= 0 && const_shift && n_reg_assigns < MAX_REG_ASSIGNS) {
              int shift = (int)u->src[1]->arg.i;
              emit_alu_rr(&buf, 1, 0x8B, R8, src_gpr); /* mov r8, range_reg */
              emit_rex_always(&buf, 1, 0, 0, R8 >> 3);
              xb_byte(&buf, 0xC1);
              emit_modrm(&buf, 3, 4, R8); /* shl r8, imm8 */
              xb_byte(&buf, (uint8_t)shift);
              /* R8 is register-assigned; skip slot store (consumers use find_reg) */
              reg_assigns[n_reg_assigns++] = (RegAssign){ u, R8 };
            } else {
              emit_load_int_src(&buf, RAX, w, u->src[0], s0, reg_assigns, n_reg_assigns);
              emit_load_int_src(&buf, RCX, 0, u->src[1], s1, reg_assigns, n_reg_assigns);
              emit_rex_always(&buf, w, 0, 0, 0);
              xb_byte(&buf, 0xD3);
              emit_modrm(&buf, 3, 4, RAX);
              emit_mov_rbp_r64(&buf, RAX, off);
            }
            break;
          }
          case POLY_OP_SHR: {
            emit_alu_r_rbp(&buf, w, 0x8B, RAX, -slot_offset(s0));
            emit_alu_r_rbp(&buf, 0, 0x8B, RCX, -slot_offset(s1));
            /* Use logical shr for unsigned types, arithmetic sar for signed */
            bool is_unsigned = poly_dtype_is_unsigned(u->src[0]->dtype);
            emit_rex_always(&buf, w, 0, 0, 0);
            xb_byte(&buf, 0xD3);
            emit_modrm(&buf, 3, is_unsigned ? 5 : 7, RAX); /* /5=shr, /7=sar */
            emit_mov_rbp_r64(&buf, RAX, off);
            break;
          }
          case POLY_OP_AND:
            emit_alu_r_rbp(&buf, w, 0x8B, RAX, -slot_offset(s0));
            emit_alu_r_rbp(&buf, w, 0x23, RAX, -slot_offset(s1)); /* and */
            emit_mov_rbp_r64(&buf, RAX, off);
            break;
          case POLY_OP_OR:
            emit_alu_r_rbp(&buf, w, 0x8B, RAX, -slot_offset(s0));
            emit_alu_r_rbp(&buf, w, 0x0B, RAX, -slot_offset(s1)); /* or */
            emit_mov_rbp_r64(&buf, RAX, off);
            break;
          case POLY_OP_XOR:
            emit_alu_r_rbp(&buf, w, 0x8B, RAX, -slot_offset(s0));
            emit_alu_r_rbp(&buf, w, 0x33, RAX, -slot_offset(s1)); /* xor */
            emit_mov_rbp_r64(&buf, RAX, off);
            break;
          case POLY_OP_NEG:
            emit_alu_r_rbp(&buf, w, 0x8B, RAX, -slot_offset(s0));
            if (wide) emit_neg_r64(&buf, RAX);
            else emit_neg_r32(&buf, RAX);
            emit_mov_rbp_r64(&buf, RAX, off);
            break;
          case POLY_OP_MAX:
            emit_alu_r_rbp(&buf, w, 0x8B, RAX, -slot_offset(s0));
            emit_alu_r_rbp(&buf, w, 0x8B, RCX, -slot_offset(s1));
            /* cmp rax, rcx; cmovl rax, rcx */
            emit_alu_rr(&buf, w, 0x3B, RAX, RCX);
            /* cmovl: 0F 4C /r */
            emit_rex_always(&buf, w, RAX >> 3, 0, RCX >> 3);
            xb_byte(&buf, 0x0F);
            xb_byte(&buf, 0x4C);
            emit_modrm(&buf, 3, RAX, RCX);
            emit_mov_rbp_r64(&buf, RAX, off);
            break;

          /* Integer comparisons */
          case POLY_OP_CMPLT:
            emit_alu_r_rbp(&buf, w, 0x8B, RAX, -slot_offset(s0));
            emit_alu_r_rbp(&buf, w, 0x3B, RAX, -slot_offset(s1)); /* cmp */
            emit_setcc(&buf, 0x0C, RAX); /* setl al */
            emit_movzx_r32_r8(&buf, RAX, RAX);
            emit_mov_rbp_r64(&buf, RAX, off);
            break;
          case POLY_OP_CMPEQ:
            emit_alu_r_rbp(&buf, w, 0x8B, RAX, -slot_offset(s0));
            emit_alu_r_rbp(&buf, w, 0x3B, RAX, -slot_offset(s1));
            emit_setcc(&buf, 0x04, RAX); /* sete al */
            emit_movzx_r32_r8(&buf, RAX, RAX);
            emit_mov_rbp_r64(&buf, RAX, off);
            break;
          case POLY_OP_CMPNE:
            emit_alu_r_rbp(&buf, w, 0x8B, RAX, -slot_offset(s0));
            emit_alu_r_rbp(&buf, w, 0x3B, RAX, -slot_offset(s1));
            emit_setcc(&buf, 0x05, RAX); /* setne al */
            emit_movzx_r32_r8(&buf, RAX, RAX);
            emit_mov_rbp_r64(&buf, RAX, off);
            break;

          case POLY_OP_WHERE:
            /* Integer WHERE: test cond, cmov */
            emit_mov_r32_rbp(&buf, RAX, -slot_offset(s0)); /* cond */
            emit_alu_r_rbp(&buf, w, 0x8B, RCX, -slot_offset(s1)); /* true */
            emit_alu_r_rbp(&buf, w, 0x8B, RDX, -slot_offset(s2)); /* false */
            emit_test_r32(&buf, RAX, RAX);
            /* cmovz rcx, rdx (if cond==0, take false) */
            emit_rex_always(&buf, w, RCX >> 3, 0, RDX >> 3);
            xb_byte(&buf, 0x0F);
            xb_byte(&buf, 0x44); /* cmove */
            emit_modrm(&buf, 3, RCX, RDX);
            emit_mov_rbp_r64(&buf, RCX, off);
            break;

          case POLY_OP_MULACC:
            emit_alu_r_rbp(&buf, w, 0x8B, RAX, -slot_offset(s0));
            emit_alu_r_rbp(&buf, w, 0x8B, RCX, -slot_offset(s1));
            if (wide) emit_imul_r64_r64(&buf, RAX, RCX);
            else emit_imul_r32_r32(&buf, RAX, RCX);
            emit_alu_r_rbp(&buf, w, 0x03, RAX, -slot_offset(s2)); /* add */
            emit_mov_rbp_r64(&buf, RAX, off);
            break;

          case POLY_OP_THREEFRY:
            /* THREEFRY is decomposed by pm_decomp before reaching renderer.
             * If it survives, just store 0 as fallback. */
            emit_mov_rbp_imm64_sx(&buf, off, 0);
            break;

          default:
            fprintf(stderr, "x64 jit: unhandled int ALU op %s at index %d\n",
                    poly_op_name(u->op), i);
            goto x64_fail;
        }
        /* Integer ALU clobbers RAX */
        rax_slot = -1;
        xmm0_slot = -1; /* be conservative: int ops may interleave with float */
      }
      LM_SET(u, slot);
      continue;
    }

    /* ── Unhandled: hard-fail ────────────────────────────────────── */
    fprintf(stderr, "x64 jit: unhandled UOp %s (op=%d) at index %d\n",
            poly_op_name(u->op), u->op, i);
    goto x64_fail;

skip_slot:
    next_slot--; /* reclaim slot on error */
    continue;
  }

  /* ── Epilogue ────────────────────────────────────────────────────── */
  if (use_avx2) emit_vzeroupper(&buf);
  emit_add_rsp_imm32(&buf, frame_size);
  emit_pop(&buf, RBX);
  emit_pop(&buf, R12);
  emit_pop(&buf, R13);
  emit_pop(&buf, R14);
  emit_pop(&buf, R15);
  emit_pop(&buf, RBP);
  emit_ret(&buf);
  #undef RELOAD_SIGN_MASK

  lm_destroy(&locals);
  free(last_consumer);
  free(slot_last_use);
  free(uop_map);

  *size_out = buf.len;
  return buf.data;

x64_fail:
  lm_destroy(&locals);
  free(last_consumer);
  free(slot_last_use);
  free(uop_map);
  free(buf.data);
  return NULL;
}

/* ══════════════════════════════════════════════════════════════════════ */
/*  Linearize with x64-specific caps (Phase 4)                           */
/* ══════════════════════════════════════════════════════════════════════ */

PolyUOp **poly_linearize_x64(PolyCtx *ctx, PolyUOp *sink, int *n_out) {
  X64CpuCaps cpu = get_cpu_caps();
  PolyRewriteOpts opts = {0};

  /* Read env overrides (matching poly_linearize_env behavior exactly).
   * env_true: returns true if set and not "0"/"false".
   * Default: OPTIMIZE=1 implies DEVECTORIZE=1 (scalar ALU, vec load/store). */
  const char *ov = getenv("POLY_OPTIMIZE");
  bool opt = ov && ov[0] != '\0' && strcmp(ov, "0") != 0 && strcmp(ov, "false") != 0;
  const char *dv = getenv("POLY_DEVECTORIZE");
  int devec = (dv && dv[0] != '\0') ? atoi(dv) : (opt ? 1 : 0);
  int beam = 0;
  const char *bv = getenv("POLY_BEAM");
  if (bv && bv[0] != '\0') beam = atoi(bv);
  opts.optimize = opt;
  opts.devectorize = devec;
  opts.beam_width = beam;

  /* x64-specific caps from CPUID */
  opts.caps.has_mulacc = cpu.has_fma && cpu.has_avx2 && cpu.os_avx_ok;
  opts.caps.has_threefry = false;
  opts.caps.max_vec_width = (cpu.has_avx2 && cpu.os_avx_ok) ? 8 : 4;

  return poly_linearize_ex(ctx, sink, opts, n_out);
}

/* ══════════════════════════════════════════════════════════════════════ */
/*  Runtime: mmap + mprotect executable memory                           */
/* ══════════════════════════════════════════════════════════════════════ */

struct PolyX64Program {
  void *code;
  size_t code_size;
  void (*call_fn)(void **);
};

PolyX64Program *poly_compile_x64(uint8_t *code, int code_size) {
  if (!code || code_size <= 0) return NULL;

  long page_size = sysconf(_SC_PAGESIZE);
  size_t alloc_size = ((size_t)code_size + page_size - 1) & ~(page_size - 1);

  void *mem = mmap(NULL, alloc_size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (mem == MAP_FAILED) return NULL;

  memcpy(mem, code, code_size);

  if (mprotect(mem, alloc_size, PROT_READ | PROT_EXEC) != 0) {
    munmap(mem, alloc_size);
    return NULL;
  }

  PolyX64Program *prog = malloc(sizeof(PolyX64Program));
  if (!prog) { munmap(mem, alloc_size); return NULL; }
  prog->code = mem;
  prog->code_size = alloc_size;
  /* Safe cast via memcpy to avoid strict aliasing violation */
  memcpy(&prog->call_fn, &mem, sizeof(prog->call_fn));
  return prog;
}

void poly_x64_program_call(PolyX64Program *prog, void **args, int n_args) {
  (void)n_args;
  prog->call_fn(args);
}

void poly_x64_program_destroy(PolyX64Program *prog) {
  if (!prog) return;
  munmap(prog->code, prog->code_size);
  free(prog);
}

#endif /* POLY_HAS_X64 */
