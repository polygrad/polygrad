/*
 * render_x86.c -- tinygrad-style x86 ISA backend.
 *
 * This backend ports tinygrad's X86Renderer shape:
 *   rewritten SINK -> POLY_OP_INS -> regalloc/post-regalloc -> SOURCE/BINARY.
 *
 * CPU/C remains the Clang-backed backend; this file is the direct ISA path.
 */

#include "codegen.h"
#include "ctx.h"
#include "engine/schedule.h"
#include "utils.h"

#ifdef POLY_HAS_X86

#include <assert.h>
#include <errno.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#ifndef MAP_ANONYMOUS
#ifdef MAP_ANON
#define MAP_ANONYMOUS MAP_ANON
#elif defined(__linux__)
#define MAP_ANONYMOUS 0x20
#endif
#endif

static void x86_dump_binary_if_requested(const uint8_t *code, int code_size) {
  const char *dir = getenv("POLY_X86_DUMP_DIR");
  if (!dir || !dir[0] || !code || code_size <= 0) return;
  static unsigned counter = 0;
  char path[512];
  unsigned id = counter++;
  int n = snprintf(path, sizeof(path), "%s/x86_%04u_%d.bin", dir, id, code_size);
  if (n <= 0 || (size_t)n >= sizeof(path)) return;
  FILE *fp = fopen(path, "wb");
  if (!fp) return;
  (void)fwrite(code, 1, (size_t)code_size, fp);
  fclose(fp);
}

typedef enum {
  POLY_X86_FRAME_INDEX = 1,
  POLY_X86_LABEL,
  POLY_X86_DEFINE,
  POLY_X86_LEA,
  POLY_X86_MOV,
  POLY_X86_MOVm,
  POLY_X86_MOVi,
  POLY_X86_MOVABS,
  POLY_X86_VMOVSS,
  POLY_X86_VMOVSD,
  POLY_X86_VMOVUPS,
  POLY_X86_VMOVSSm,
  POLY_X86_VMOVSDm,
  POLY_X86_VMOVUPSm,
  POLY_X86_MOVZX,
  POLY_X86_MOVSX,
  POLY_X86_MOVSXD,
  POLY_X86_VPMOVZXBW,
  POLY_X86_VPMOVZXBD,
  POLY_X86_VPMOVZXBQ,
  POLY_X86_VPMOVZXWD,
  POLY_X86_VPMOVZXWQ,
  POLY_X86_VPMOVZXDQ,
  POLY_X86_VPMOVSXBW,
  POLY_X86_VPMOVSXBD,
  POLY_X86_VPMOVSXBQ,
  POLY_X86_VPMOVSXWD,
  POLY_X86_VPMOVSXWQ,
  POLY_X86_VPMOVSXDQ,
  POLY_X86_VCVTDQ2PS,
  POLY_X86_VCVTDQ2PD,
  POLY_X86_VCVTTPS2DQ,
  POLY_X86_VCVTTPD2DQ,
  POLY_X86_VCVTPH2PS,
  POLY_X86_VCVTPS2PH,
  POLY_X86_VCVTPS2PD,
  POLY_X86_VCVTPD2PS,
  POLY_X86_VCVTSS2SD,
  POLY_X86_VCVTSD2SS,
  POLY_X86_VCVTSI2SS,
  POLY_X86_VCVTSI2SD,
  POLY_X86_VCVTTSS2SI,
  POLY_X86_VCVTTSD2SI,
  POLY_X86_VMOVD,
  POLY_X86_VMOVQ,
  POLY_X86_VMOVDm,
  POLY_X86_VMOVQm,
  POLY_X86_VUCOMISS,
  POLY_X86_VUCOMISD,
  POLY_X86_VCMPSS,
  POLY_X86_VCMPSD,
  POLY_X86_VCMPPS,
  POLY_X86_VCMPPD,
  POLY_X86_VPCMPGTB,
  POLY_X86_VPCMPGTW,
  POLY_X86_VPCMPGTD,
  POLY_X86_VPCMPGTQ,
  POLY_X86_VPCMPEQB,
  POLY_X86_VPCMPEQW,
  POLY_X86_VPCMPEQD,
  POLY_X86_VPCMPEQQ,
  POLY_X86_SETNE,
  POLY_X86_SETE,
  POLY_X86_SETL,
  POLY_X86_SETB,
  POLY_X86_CMOVNE,
  POLY_X86_CMOVE,
  POLY_X86_CMOVL,
  POLY_X86_CMOVB,
  POLY_X86_VPBLENDVB,
  POLY_X86_VBLENDVPS,
  POLY_X86_VBLENDVPD,
  POLY_X86_JNE,
  POLY_X86_JE,
  POLY_X86_JL,
  POLY_X86_JB,
  POLY_X86_JGE,
  POLY_X86_JMP,
  POLY_X86_VSHUFPS,
  POLY_X86_VSHUFPD,
  POLY_X86_VINSERTPS,
  POLY_X86_VPSRLDQ,
  POLY_X86_VPEXTRB,
  POLY_X86_VPEXTRW,
  POLY_X86_VPEXTRD,
  POLY_X86_VPEXTRQ,
  POLY_X86_VPINSRB,
  POLY_X86_VPINSRW,
  POLY_X86_VPINSRD,
  POLY_X86_VPINSRQ,
  POLY_X86_VPBROADCASTB,
  POLY_X86_VPBROADCASTW,
  POLY_X86_VPBROADCASTD,
  POLY_X86_VPBROADCASTQ,
  POLY_X86_VBROADCASTSS,
  POLY_X86_IDIV,
  POLY_X86_DIV,
  POLY_X86_ADD,
  POLY_X86_ADDi,
  POLY_X86_SUB,
  POLY_X86_SUBi,
  POLY_X86_IMUL,
  POLY_X86_IMULi,
  POLY_X86_AND,
  POLY_X86_ANDi,
  POLY_X86_XOR,
  POLY_X86_XORi,
  POLY_X86_OR,
  POLY_X86_ORi,
  POLY_X86_SHL,
  POLY_X86_SHLi,
  POLY_X86_SHR,
  POLY_X86_SHRi,
  POLY_X86_SAR,
  POLY_X86_SARi,
  POLY_X86_CMP,
  POLY_X86_CMPi,
  POLY_X86_VROUNDSS,
  POLY_X86_VROUNDSD,
  POLY_X86_VROUNDPS,
  POLY_X86_VROUNDPD,
  POLY_X86_VSQRTSS,
  POLY_X86_VSQRTSD,
  POLY_X86_VSQRTPS,
  POLY_X86_VSQRTPD,
  POLY_X86_VADDSS,
  POLY_X86_VADDSD,
  POLY_X86_VADDPS,
  POLY_X86_VADDPD,
  POLY_X86_VSUBSS,
  POLY_X86_VSUBSD,
  POLY_X86_VSUBPS,
  POLY_X86_VSUBPD,
  POLY_X86_VMULSS,
  POLY_X86_VMULSD,
  POLY_X86_VMULPS,
  POLY_X86_VMULPD,
  POLY_X86_VDIVSS,
  POLY_X86_VDIVSD,
  POLY_X86_VDIVPS,
  POLY_X86_VDIVPD,
  POLY_X86_VMAXSS,
  POLY_X86_VMAXSD,
  POLY_X86_VMAXPS,
  POLY_X86_VMAXPD,
  POLY_X86_VMINSS,
  POLY_X86_VMINSD,
  POLY_X86_VMINPS,
  POLY_X86_VMINPD,
  POLY_X86_VPADDB,
  POLY_X86_VPADDW,
  POLY_X86_VPADDD,
  POLY_X86_VPADDQ,
  POLY_X86_VPSUBB,
  POLY_X86_VPSUBW,
  POLY_X86_VPSUBD,
  POLY_X86_VPSUBQ,
  POLY_X86_VPMULLW,
  POLY_X86_VPMULLD,
  POLY_X86_VPAND,
  POLY_X86_VPOR,
  POLY_X86_VPXOR,
  POLY_X86_VPSLLVD,
  POLY_X86_VPSLLVQ,
  POLY_X86_VPSRLVD,
  POLY_X86_VPSRLVQ,
  POLY_X86_VPSRAVD,
  POLY_X86_VFMADD213SS,
  POLY_X86_VFMADD213SD,
  POLY_X86_VFMADD213PS,
  POLY_X86_VFMADD213PD,
  POLY_X86_RET,
} PolyX86Op;

typedef enum {
  X86_REG_RAX = 0,
  X86_REG_RCX = 1,
  X86_REG_RDX = 2,
  X86_REG_RBX = 3,
  X86_REG_RSP = 4,
  X86_REG_RBP = 5,
  X86_REG_RSI = 6,
  X86_REG_RDI = 7,
  X86_REG_R8 = 8,
  X86_REG_R9 = 9,
  X86_REG_R10 = 10,
  X86_REG_R11 = 11,
  X86_REG_R12 = 12,
  X86_REG_R13 = 13,
  X86_REG_R14 = 14,
  X86_REG_R15 = 15,
} X86Gpr;

static const int x86_callee_saved_gprs[] = {
    X86_REG_RBX, X86_REG_RBP, X86_REG_R12, X86_REG_R13, X86_REG_R14, X86_REG_R15,
};

typedef enum {
  X86_REG_CLASS_NONE = 0,
  X86_REG_CLASS_GPR,
  X86_REG_CLASS_WGPR,
  X86_REG_CLASS_XMM,
  X86_REG_CLASS_FIXED_RAX,
  X86_REG_CLASS_FIXED_RDX,
} X86RegClass;

#define X86_TAG_REAL 0x40000000
#define X86_TAG_VIRT 0x20000000
#define X86_TAG_CLASS_SHIFT 24
#define X86_TAG_CLASS_MASK 0x0F
#define X86_TAG_ID_MASK 0xFFFFFF

static int32_t x86_tag_real(X86RegClass cls, int reg) {
  return X86_TAG_REAL | ((int32_t)cls << X86_TAG_CLASS_SHIFT) | (reg & X86_TAG_ID_MASK);
}

static int32_t x86_tag_virtual(X86RegClass cls, int id) {
  return X86_TAG_VIRT | ((int32_t)cls << X86_TAG_CLASS_SHIFT) | (id & X86_TAG_ID_MASK);
}

static bool x86_tag_is_real(int32_t tag) {
  return tag > 0 && (tag & X86_TAG_REAL) != 0;
}

static X86RegClass x86_tag_class(int32_t tag) {
  return (X86RegClass)((tag >> X86_TAG_CLASS_SHIFT) & X86_TAG_CLASS_MASK);
}

static int x86_tag_id(int32_t tag) {
  return tag & X86_TAG_ID_MASK;
}

static X86RegClass x86_real_class_for_constraint(X86RegClass cls) {
  switch (cls) {
  case X86_REG_CLASS_FIXED_RAX:
  case X86_REG_CLASS_FIXED_RDX:
    return X86_REG_CLASS_WGPR;
  default:
    return cls;
  }
}

static const char *x86_op_name(PolyX86Op op) {
  switch (op) {
  case POLY_X86_FRAME_INDEX: return "FRAME_INDEX";
  case POLY_X86_LABEL: return "LABEL";
  case POLY_X86_DEFINE: return "DEFINE";
  case POLY_X86_LEA: return "LEA";
  case POLY_X86_MOV: return "MOV";
  case POLY_X86_MOVm: return "MOVm";
  case POLY_X86_MOVi: return "MOVi";
  case POLY_X86_MOVABS: return "MOVABS";
  case POLY_X86_VMOVSS: return "VMOVSS";
  case POLY_X86_VMOVSD: return "VMOVSD";
  case POLY_X86_VMOVUPS: return "VMOVUPS";
  case POLY_X86_VMOVSSm: return "VMOVSSm";
  case POLY_X86_VMOVSDm: return "VMOVSDm";
  case POLY_X86_VMOVUPSm: return "VMOVUPSm";
  case POLY_X86_MOVZX: return "MOVZX";
  case POLY_X86_MOVSX: return "MOVSX";
  case POLY_X86_MOVSXD: return "MOVSXD";
  case POLY_X86_VPMOVZXBW: return "VPMOVZXBW";
  case POLY_X86_VPMOVZXBD: return "VPMOVZXBD";
  case POLY_X86_VPMOVZXBQ: return "VPMOVZXBQ";
  case POLY_X86_VPMOVZXWD: return "VPMOVZXWD";
  case POLY_X86_VPMOVZXWQ: return "VPMOVZXWQ";
  case POLY_X86_VPMOVZXDQ: return "VPMOVZXDQ";
  case POLY_X86_VPMOVSXBW: return "VPMOVSXBW";
  case POLY_X86_VPMOVSXBD: return "VPMOVSXBD";
  case POLY_X86_VPMOVSXBQ: return "VPMOVSXBQ";
  case POLY_X86_VPMOVSXWD: return "VPMOVSXWD";
  case POLY_X86_VPMOVSXWQ: return "VPMOVSXWQ";
  case POLY_X86_VPMOVSXDQ: return "VPMOVSXDQ";
  case POLY_X86_VCVTDQ2PS: return "VCVTDQ2PS";
  case POLY_X86_VCVTDQ2PD: return "VCVTDQ2PD";
  case POLY_X86_VCVTTPS2DQ: return "VCVTTPS2DQ";
  case POLY_X86_VCVTTPD2DQ: return "VCVTTPD2DQ";
  case POLY_X86_VCVTPH2PS: return "VCVTPH2PS";
  case POLY_X86_VCVTPS2PH: return "VCVTPS2PH";
  case POLY_X86_VCVTPS2PD: return "VCVTPS2PD";
  case POLY_X86_VCVTPD2PS: return "VCVTPD2PS";
  case POLY_X86_VCVTSS2SD: return "VCVTSS2SD";
  case POLY_X86_VCVTSD2SS: return "VCVTSD2SS";
  case POLY_X86_VCVTSI2SS: return "VCVTSI2SS";
  case POLY_X86_VCVTSI2SD: return "VCVTSI2SD";
  case POLY_X86_VCVTTSS2SI: return "VCVTTSS2SI";
  case POLY_X86_VCVTTSD2SI: return "VCVTTSD2SI";
  case POLY_X86_VMOVD: return "VMOVD";
  case POLY_X86_VMOVQ: return "VMOVQ";
  case POLY_X86_VMOVDm: return "VMOVDm";
  case POLY_X86_VMOVQm: return "VMOVQm";
  case POLY_X86_VPEXTRB: return "VPEXTRB";
  case POLY_X86_VPEXTRW: return "VPEXTRW";
  case POLY_X86_VPEXTRD: return "VPEXTRD";
  case POLY_X86_VPEXTRQ: return "VPEXTRQ";
  case POLY_X86_VPINSRB: return "VPINSRB";
  case POLY_X86_VPINSRW: return "VPINSRW";
  case POLY_X86_VPINSRD: return "VPINSRD";
  case POLY_X86_VPINSRQ: return "VPINSRQ";
  case POLY_X86_VPBROADCASTB: return "VPBROADCASTB";
  case POLY_X86_VPBROADCASTW: return "VPBROADCASTW";
  case POLY_X86_VPBROADCASTD: return "VPBROADCASTD";
  case POLY_X86_VPBROADCASTQ: return "VPBROADCASTQ";
  case POLY_X86_VBROADCASTSS: return "VBROADCASTSS";
  case POLY_X86_VUCOMISS: return "VUCOMISS";
  case POLY_X86_VUCOMISD: return "VUCOMISD";
  case POLY_X86_VCMPSS: return "VCMPSS";
  case POLY_X86_VCMPSD: return "VCMPSD";
  case POLY_X86_VCMPPS: return "VCMPPS";
  case POLY_X86_VCMPPD: return "VCMPPD";
  case POLY_X86_VPCMPGTB: return "VPCMPGTB";
  case POLY_X86_VPCMPGTW: return "VPCMPGTW";
  case POLY_X86_VPCMPGTD: return "VPCMPGTD";
  case POLY_X86_VPCMPGTQ: return "VPCMPGTQ";
  case POLY_X86_VPCMPEQB: return "VPCMPEQB";
  case POLY_X86_VPCMPEQW: return "VPCMPEQW";
  case POLY_X86_VPCMPEQD: return "VPCMPEQD";
  case POLY_X86_VPCMPEQQ: return "VPCMPEQQ";
  case POLY_X86_SETNE: return "SETNE";
  case POLY_X86_SETE: return "SETE";
  case POLY_X86_SETL: return "SETL";
  case POLY_X86_SETB: return "SETB";
  case POLY_X86_CMOVNE: return "CMOVNE";
  case POLY_X86_CMOVE: return "CMOVE";
  case POLY_X86_CMOVL: return "CMOVL";
  case POLY_X86_CMOVB: return "CMOVB";
  case POLY_X86_VPBLENDVB: return "VPBLENDVB";
  case POLY_X86_VBLENDVPS: return "VBLENDVPS";
  case POLY_X86_VBLENDVPD: return "VBLENDVPD";
  case POLY_X86_JNE: return "JNE";
  case POLY_X86_JE: return "JE";
  case POLY_X86_JL: return "JL";
  case POLY_X86_JB: return "JB";
  case POLY_X86_JGE: return "JGE";
  case POLY_X86_JMP: return "JMP";
  case POLY_X86_VSHUFPS: return "VSHUFPS";
  case POLY_X86_VSHUFPD: return "VSHUFPD";
  case POLY_X86_VINSERTPS: return "VINSERTPS";
  case POLY_X86_VPSRLDQ: return "VPSRLDQ";
  case POLY_X86_IDIV: return "IDIV";
  case POLY_X86_DIV: return "DIV";
  case POLY_X86_ADD: return "ADD";
  case POLY_X86_ADDi: return "ADDi";
  case POLY_X86_SUB: return "SUB";
  case POLY_X86_SUBi: return "SUBi";
  case POLY_X86_IMUL: return "IMUL";
  case POLY_X86_IMULi: return "IMULi";
  case POLY_X86_AND: return "AND";
  case POLY_X86_ANDi: return "ANDi";
  case POLY_X86_XOR: return "XOR";
  case POLY_X86_XORi: return "XORi";
  case POLY_X86_OR: return "OR";
  case POLY_X86_ORi: return "ORi";
  case POLY_X86_SHL: return "SHL";
  case POLY_X86_SHLi: return "SHLi";
  case POLY_X86_SHR: return "SHR";
  case POLY_X86_SHRi: return "SHRi";
  case POLY_X86_SAR: return "SAR";
  case POLY_X86_SARi: return "SARi";
  case POLY_X86_CMP: return "CMP";
  case POLY_X86_CMPi: return "CMPi";
  case POLY_X86_VROUNDSS: return "VROUNDSS";
  case POLY_X86_VROUNDSD: return "VROUNDSD";
  case POLY_X86_VROUNDPS: return "VROUNDPS";
  case POLY_X86_VROUNDPD: return "VROUNDPD";
  case POLY_X86_VSQRTSS: return "VSQRTSS";
  case POLY_X86_VSQRTSD: return "VSQRTSD";
  case POLY_X86_VSQRTPS: return "VSQRTPS";
  case POLY_X86_VSQRTPD: return "VSQRTPD";
  case POLY_X86_VADDSS: return "VADDSS";
  case POLY_X86_VADDSD: return "VADDSD";
  case POLY_X86_VADDPS: return "VADDPS";
  case POLY_X86_VADDPD: return "VADDPD";
  case POLY_X86_VSUBSS: return "VSUBSS";
  case POLY_X86_VSUBSD: return "VSUBSD";
  case POLY_X86_VSUBPS: return "VSUBPS";
  case POLY_X86_VSUBPD: return "VSUBPD";
  case POLY_X86_VMULSS: return "VMULSS";
  case POLY_X86_VMULSD: return "VMULSD";
  case POLY_X86_VMULPS: return "VMULPS";
  case POLY_X86_VMULPD: return "VMULPD";
  case POLY_X86_VDIVSS: return "VDIVSS";
  case POLY_X86_VDIVSD: return "VDIVSD";
  case POLY_X86_VDIVPS: return "VDIVPS";
  case POLY_X86_VDIVPD: return "VDIVPD";
  case POLY_X86_VMAXSS: return "VMAXSS";
  case POLY_X86_VMAXSD: return "VMAXSD";
  case POLY_X86_VMAXPS: return "VMAXPS";
  case POLY_X86_VMAXPD: return "VMAXPD";
  case POLY_X86_VMINSS: return "VMINSS";
  case POLY_X86_VMINSD: return "VMINSD";
  case POLY_X86_VMINPS: return "VMINPS";
  case POLY_X86_VMINPD: return "VMINPD";
  case POLY_X86_VPADDB: return "VPADDB";
  case POLY_X86_VPADDW: return "VPADDW";
  case POLY_X86_VPADDD: return "VPADDD";
  case POLY_X86_VPADDQ: return "VPADDQ";
  case POLY_X86_VPSUBB: return "VPSUBB";
  case POLY_X86_VPSUBW: return "VPSUBW";
  case POLY_X86_VPSUBD: return "VPSUBD";
  case POLY_X86_VPSUBQ: return "VPSUBQ";
  case POLY_X86_VPMULLW: return "VPMULLW";
  case POLY_X86_VPMULLD: return "VPMULLD";
  case POLY_X86_VPAND: return "VPAND";
  case POLY_X86_VPOR: return "VPOR";
  case POLY_X86_VPXOR: return "VPXOR";
  case POLY_X86_VPSLLVD: return "VPSLLVD";
  case POLY_X86_VPSLLVQ: return "VPSLLVQ";
  case POLY_X86_VPSRLVD: return "VPSRLVD";
  case POLY_X86_VPSRLVQ: return "VPSRLVQ";
  case POLY_X86_VPSRAVD: return "VPSRAVD";
  case POLY_X86_VFMADD213SS: return "VFMADD213SS";
  case POLY_X86_VFMADD213SD: return "VFMADD213SD";
  case POLY_X86_VFMADD213PS: return "VFMADD213PS";
  case POLY_X86_VFMADD213PD: return "VFMADD213PD";
  case POLY_X86_RET: return "RET";
  default: return "X86?";
  }
}

typedef struct {
  uint8_t *data;
  int len;
  int cap;
} X86Buf;

static void xb_grow(X86Buf *b, int need) {
  if (b->len + need <= b->cap) return;
  int nc = b->cap ? b->cap * 2 : 256;
  while (b->len + need > nc)
    nc *= 2;
  uint8_t *nd = realloc(b->data, (size_t)nc);
  if (!nd) abort();
  b->data = nd;
  b->cap = nc;
}

static void xb_byte(X86Buf *b, uint8_t v) {
  xb_grow(b, 1);
  b->data[b->len++] = v;
}

static void xb_i32(X86Buf *b, int32_t v) {
  xb_grow(b, 4);
  memcpy(b->data + b->len, &v, 4);
  b->len += 4;
}

static void xb_i64(X86Buf *b, int64_t v) {
  xb_grow(b, 8);
  memcpy(b->data + b->len, &v, 8);
  b->len += 8;
}

static void emit_modrm(X86Buf *b, int mod, int reg, int rm) {
  xb_byte(b, (uint8_t)(((mod & 3) << 6) | ((reg & 7) << 3) | (rm & 7)));
}

static void emit_rex(X86Buf *b, int w, int r, int x, int rm) {
  xb_byte(b, (uint8_t)(0x40 | ((w & 1) << 3) | ((r & 1) << 2) | ((x & 1) << 1) | (rm & 1)));
}

static int scale_to_ss(int scale) {
  switch (scale) {
  case 1: return 0;
  case 2: return 1;
  case 4: return 2;
  case 8: return 3;
  default: return 0;
  }
}

static void emit_vex2(X86Buf *b, int R, int vvvv, int L, int pp) {
  xb_byte(b, 0xC5);
  xb_byte(b, (uint8_t)(((R & 1) << 7) | ((~vvvv & 0xF) << 3) | ((L & 1) << 2) | (pp & 3)));
}

static void emit_vex3(X86Buf *b, int R, int X, int B, int mmmmm, int W, int vvvv, int L, int pp) {
  xb_byte(b, 0xC4);
  xb_byte(b, (uint8_t)(((R & 1) << 7) | ((X & 1) << 6) | ((B & 1) << 5) | (mmmmm & 0x1F)));
  xb_byte(b, (uint8_t)(((W & 1) << 7) | ((~vvvv & 0xF) << 3) | ((L & 1) << 2) | (pp & 3)));
}

static bool x86_ins_op(PolyUOp *u, PolyX86Op *out) {
  if (!u || u->op != POLY_OP_INS || u->arg.kind != POLY_ARG_INT) return false;
  if (out) *out = (PolyX86Op)u->arg.i;
  return true;
}

static const char *x86_uop_label(PolyUOp *u) {
  if (!u) return NULL;
  return u->tag_arg.kind == POLY_ARG_STRING ? u->tag_arg.str : NULL;
}

static bool x86_op_in(const PolyX86Op *ops, int n, PolyX86Op op) {
  for (int i = 0; i < n; i++)
    if (ops[i] == op) return true;
  return false;
}

static bool x86_readmem2nd(PolyX86Op op);
static bool x86_readmem3rd(PolyX86Op op);

static bool x86_is_two_address(PolyX86Op op) {
  static const PolyX86Op ops[] = {
      POLY_X86_ADD, POLY_X86_ADDi, POLY_X86_AND, POLY_X86_ANDi, POLY_X86_XOR,
      POLY_X86_XORi, POLY_X86_OR, POLY_X86_ORi, POLY_X86_IMUL, POLY_X86_SUB,
      POLY_X86_SUBi, POLY_X86_SHL, POLY_X86_SHLi, POLY_X86_SHR, POLY_X86_SHRi,
      POLY_X86_SAR, POLY_X86_SARi, POLY_X86_IDIV, POLY_X86_DIV, POLY_X86_VFMADD213SS,
      POLY_X86_VFMADD213SD, POLY_X86_VFMADD213PS, POLY_X86_VFMADD213PD, POLY_X86_CMOVNE,
      POLY_X86_CMOVE, POLY_X86_CMOVL, POLY_X86_CMOVB,
  };
  return x86_op_in(ops, (int)(sizeof(ops) / sizeof(ops[0])), op);
}

static bool x86_readmem1st(PolyX86Op op) {
  static const PolyX86Op ops[] = {
      POLY_X86_MOV, POLY_X86_VMOVSS, POLY_X86_VMOVSD, POLY_X86_VMOVUPS, POLY_X86_MOVZX,
      POLY_X86_MOVSX, POLY_X86_MOVSXD, POLY_X86_VMOVD, POLY_X86_VMOVQ,
      POLY_X86_VPMOVZXBW, POLY_X86_VPMOVZXBD, POLY_X86_VPMOVZXBQ,
      POLY_X86_VPMOVZXWD, POLY_X86_VPMOVZXWQ, POLY_X86_VPMOVZXDQ,
      POLY_X86_VPMOVSXBW, POLY_X86_VPMOVSXBD, POLY_X86_VPMOVSXBQ,
      POLY_X86_VPMOVSXWD, POLY_X86_VPMOVSXWQ, POLY_X86_VPMOVSXDQ,
      POLY_X86_VCVTDQ2PS,
      POLY_X86_VCVTDQ2PD, POLY_X86_VCVTTPS2DQ, POLY_X86_VCVTTPD2DQ, POLY_X86_VCVTTSS2SI,
      POLY_X86_VCVTTSD2SI, POLY_X86_VCVTPH2PS, POLY_X86_VCVTPS2PD, POLY_X86_VCVTPD2PS,
      POLY_X86_VROUNDPS, POLY_X86_VROUNDPD, POLY_X86_VSQRTPS, POLY_X86_VSQRTPD,
      POLY_X86_VPBROADCASTB, POLY_X86_VPBROADCASTW, POLY_X86_VPBROADCASTD,
      POLY_X86_VPBROADCASTQ, POLY_X86_VBROADCASTSS, POLY_X86_CMPi, POLY_X86_IMULi, POLY_X86_LEA,
  };
  return x86_op_in(ops, (int)(sizeof(ops) / sizeof(ops[0])), op);
}

static bool x86_readmem2nd(PolyX86Op op) {
  static const PolyX86Op ops[] = {
      POLY_X86_ADD, POLY_X86_SUB, POLY_X86_AND, POLY_X86_OR, POLY_X86_XOR, POLY_X86_SHL,
      POLY_X86_SHR, POLY_X86_SAR, POLY_X86_IMUL, POLY_X86_CMP, POLY_X86_VADDSS,
      POLY_X86_VADDSD, POLY_X86_VADDPS, POLY_X86_VADDPD, POLY_X86_VSUBSS, POLY_X86_VSUBSD,
      POLY_X86_VSUBPS, POLY_X86_VSUBPD, POLY_X86_VMULSS, POLY_X86_VMULSD, POLY_X86_VMULPS,
      POLY_X86_VMULPD, POLY_X86_VDIVSS, POLY_X86_VDIVSD, POLY_X86_VDIVPS, POLY_X86_VDIVPD,
      POLY_X86_VPADDB, POLY_X86_VPADDW, POLY_X86_VPADDD, POLY_X86_VPADDQ,
      POLY_X86_VPSUBB, POLY_X86_VPSUBW, POLY_X86_VPSUBD, POLY_X86_VPSUBQ,
      POLY_X86_VPCMPEQB, POLY_X86_VPCMPEQW, POLY_X86_VPCMPEQD, POLY_X86_VPCMPEQQ,
      POLY_X86_VPBLENDVB, POLY_X86_VBLENDVPS, POLY_X86_VBLENDVPD, POLY_X86_VPCMPGTB,
      POLY_X86_VPCMPGTW, POLY_X86_VPCMPGTD, POLY_X86_VPCMPGTQ, POLY_X86_VCMPSS,
      POLY_X86_VCMPSD, POLY_X86_VCMPPS, POLY_X86_VCMPPD, POLY_X86_VPMULLW,
      POLY_X86_VPMULLD, POLY_X86_VROUNDSS, POLY_X86_VROUNDSD, POLY_X86_VSQRTSS,
      POLY_X86_VSQRTSD, POLY_X86_VSHUFPS, POLY_X86_VSHUFPD, POLY_X86_VINSERTPS,
      POLY_X86_VPINSRB, POLY_X86_VPINSRW, POLY_X86_VPINSRD, POLY_X86_VPINSRQ,
      POLY_X86_VPAND, POLY_X86_VPOR, POLY_X86_VPXOR, POLY_X86_VPSLLVD, POLY_X86_VPSLLVQ,
      POLY_X86_VPSRLVD, POLY_X86_VPSRLVQ, POLY_X86_VPSRAVD, POLY_X86_CMOVNE,
      POLY_X86_CMOVE, POLY_X86_CMOVL, POLY_X86_CMOVB, POLY_X86_VMAXSS, POLY_X86_VMAXSD,
      POLY_X86_VMAXPS, POLY_X86_VMAXPD, POLY_X86_VMINSS, POLY_X86_VMINSD, POLY_X86_VMINPS,
      POLY_X86_VMINPD, POLY_X86_VCVTSI2SS, POLY_X86_VCVTSI2SD, POLY_X86_VCVTSS2SD,
      POLY_X86_VCVTSD2SS, POLY_X86_VUCOMISS, POLY_X86_VUCOMISD, POLY_X86_IDIV,
      POLY_X86_DIV,
  };
  return x86_op_in(ops, (int)(sizeof(ops) / sizeof(ops[0])), op);
}

static bool x86_readmem3rd(PolyX86Op op) {
  static const PolyX86Op ops[] = {
      POLY_X86_VFMADD213SS,
      POLY_X86_VFMADD213SD,
      POLY_X86_VFMADD213PS,
      POLY_X86_VFMADD213PD,
  };
  return x86_op_in(ops, (int)(sizeof(ops) / sizeof(ops[0])), op);
}

static bool x86_rm1st(PolyX86Op op) {
  return x86_readmem1st(op) || (x86_readmem2nd(op) && x86_is_two_address(op)) ||
         op == POLY_X86_VPSRLDQ;
}

static bool x86_writemem(PolyX86Op op) {
  static const PolyX86Op ops[] = {
      POLY_X86_MOVm, POLY_X86_MOVi, POLY_X86_VMOVSSm, POLY_X86_VMOVSDm, POLY_X86_VMOVUPSm,
      POLY_X86_VMOVDm, POLY_X86_VMOVQm, POLY_X86_ADDi, POLY_X86_SUBi, POLY_X86_ANDi,
      POLY_X86_ORi, POLY_X86_XORi, POLY_X86_SHLi, POLY_X86_SHRi, POLY_X86_SARi,
      POLY_X86_SETNE, POLY_X86_SETE, POLY_X86_SETL, POLY_X86_SETB, POLY_X86_VCVTPS2PH,
      POLY_X86_VPEXTRB, POLY_X86_VPEXTRW, POLY_X86_VPEXTRD, POLY_X86_VPEXTRQ,
  };
  return x86_op_in(ops, (int)(sizeof(ops) / sizeof(ops[0])), op);
}

static bool x86_readflags(PolyX86Op op) {
  static const PolyX86Op ops[] = {
      POLY_X86_CMOVNE, POLY_X86_CMOVE, POLY_X86_CMOVL, POLY_X86_CMOVB,
      POLY_X86_SETNE, POLY_X86_SETE, POLY_X86_SETL, POLY_X86_SETB,
      POLY_X86_JNE, POLY_X86_JE, POLY_X86_JL, POLY_X86_JB, POLY_X86_JGE,
  };
  return x86_op_in(ops, (int)(sizeof(ops) / sizeof(ops[0])), op);
}

static bool x86_writeflags(PolyX86Op op) {
  static const PolyX86Op ops[] = {
      POLY_X86_CMP, POLY_X86_CMPi, POLY_X86_ADD, POLY_X86_ADDi, POLY_X86_SUB,
      POLY_X86_SUBi, POLY_X86_IMUL, POLY_X86_IMULi, POLY_X86_IDIV, POLY_X86_DIV,
      POLY_X86_SHL, POLY_X86_SHLi, POLY_X86_SHR, POLY_X86_SHRi, POLY_X86_SAR,
      POLY_X86_SARi, POLY_X86_AND, POLY_X86_ANDi, POLY_X86_XOR, POLY_X86_XORi,
      POLY_X86_OR, POLY_X86_ORi, POLY_X86_VUCOMISS, POLY_X86_VUCOMISD,
  };
  return x86_op_in(ops, (int)(sizeof(ops) / sizeof(ops[0])), op);
}

static bool x86_is_jump(PolyX86Op op) {
  return op == POLY_X86_JE || op == POLY_X86_JNE || op == POLY_X86_JL ||
         op == POLY_X86_JB || op == POLY_X86_JGE || op == POLY_X86_JMP;
}

static int x86_uop_reg(PolyUOp *u) {
  if (!u) return -1;
  if (u->tag_arg.kind == POLY_ARG_INT_TUPLE && u->tag_arg.int_tuple.n > 0)
    return (int)u->tag_arg.int_tuple.vals[0];
  if (u->tag != 0) return u->tag;
  return -1;
}

static int x86_uop_def_regs(PolyUOp *u, int32_t *defs, int cap) {
  if (!u || !defs || cap <= 0) return 0;
  int n = 0;
  if (u->tag_arg.kind == POLY_ARG_INT_TUPLE) {
    for (int i = 0; i < u->tag_arg.int_tuple.n && n < cap; i++)
      defs[n++] = (int32_t)u->tag_arg.int_tuple.vals[i];
  } else if (u->tag != 0) {
    defs[n++] = u->tag;
  }
  return n;
}

static int x86_uop_reg_index(PolyUOp *u) {
  int r = x86_uop_reg(u);
  return r >= 0 ? x86_tag_id(r) : -1;
}

static bool x86_tag_is_virtual(int32_t tag);
static int x86_virtual_id(int32_t tag);
static PolyUOp *x86_disp_const(PolyCtx *ctx, int64_t disp);
static PolyX86Op x86_mov_op_for_dtype(PolyDType dt, bool store);
static bool x86_ins_is_load_addr(PolyUOp *u);
static PolyUOp *x86_graph_scalar_load_from_reg_vector_lane(
    PolyCtx *ctx,
    PolyUOp *load,
    PolyDType lane_dtype,
    int lane
);
static PolyX86Op x86_float_bin_op(PolyOps op, PolyDType dt);
static PolyX86Op x86_int_bin_op(PolyOps op, PolyDType dt, bool imm);
static PolyX86Op x86_vector_int_extend_op(PolyDType src_dt, PolyDType dst_dt);
static bool x86_ins_has_memory_tuple_at(PolyUOp *u, int src_idx);
static PolyUOp *x86_clone_with_regs(
    PolyCtx *ctx,
    PolyUOp *u,
    PolyUOp **srcs,
    int n_src,
    int32_t *defs,
    int n_defs
);

static bool x86_uop_const_i64(PolyUOp *u, int64_t *out) {
  if (!u || u->op != POLY_OP_CONST || u->arg.kind != POLY_ARG_INT) return false;
  if (out) *out = u->arg.i;
  return true;
}

static int x86_dtype_itemsize(PolyDType dt) {
  return poly_dtype_itemsize(poly_dtype_scalar(dt));
}

static PolyArg x86_arg_int_tuple(int64_t *vals, int n) {
  PolyArg a = {.kind = POLY_ARG_INT_TUPLE};
  a.int_tuple.vals = vals;
  a.int_tuple.n = n;
  return a;
}

static PolyArg x86_arg_string(const char *s) {
  return poly_arg_str(s);
}

static bool x86_const_is_imm(PolyUOp *u) {
  return u && u->op == POLY_OP_CONST && u->tag_arg.kind == POLY_ARG_BOOL && u->tag_arg.b;
}

static PolyUOp *x86_const_i(PolyCtx *ctx, PolyDType dt, int64_t v) {
  return poly_uop_tagged_arg(
      ctx, POLY_OP_CONST, dt, NULL, 0, poly_arg_int(v), 0, poly_arg_bool(true)
  );
}

static PolyUOp *x86_noop(PolyCtx *ctx) {
  return poly_uop0(ctx, POLY_OP_NOOP, POLY_VOID, poly_arg_none());
}

static PolyDType x86_u64(void) {
  return POLY_UINT64;
}

static X86RegClass x86_class_for_dtype(PolyDType dt, bool buffer_value) {
  if (buffer_value || dt.is_ptr)
    return X86_REG_CLASS_WGPR;
  if (dt.count > 1)
    return X86_REG_CLASS_XMM;
  if (poly_dtype_is_int(poly_dtype_scalar(dt)) || poly_dtype_is_bool(poly_dtype_scalar(dt)))
    return X86_REG_CLASS_WGPR;
  return X86_REG_CLASS_XMM;
}

static int x86_value_size(PolyDType dt, bool buffer_value) {
  if (buffer_value || dt.is_ptr) return 8;
  PolyDType s = poly_dtype_scalar(dt);
  int item = poly_dtype_itemsize(s);
  if (item <= 0) item = 8;
  int count = dt.count > 1 ? dt.count : 1;
  return item * count;
}

static PolyUOp *x86_ins_ex(
    PolyCtx *ctx,
    PolyX86Op op,
    PolyDType dtype,
    PolyUOp **srcs,
    int n_src,
    const int32_t *defs,
    int n_defs,
    PolyArg tag_arg
) {
  int64_t stack_defs[8];
  PolyArg reg_arg = poly_arg_none();
  if (n_defs > 0) {
    for (int i = 0; i < n_defs && i < 8; i++)
      stack_defs[i] = defs[i];
    reg_arg = x86_arg_int_tuple(stack_defs, n_defs < 8 ? n_defs : 8);
  }
  int32_t tag = n_defs > 0 ? defs[0] : 0;
  PolyUOp *u =
      poly_uop_tagged_arg(ctx, POLY_OP_INS, dtype, srcs, n_src, poly_arg_int((int64_t)op), tag, reg_arg);
  if (tag_arg.kind != POLY_ARG_NONE)
    u = poly_uop_tagged_arg(ctx, POLY_OP_INS, dtype, srcs, n_src, poly_arg_int((int64_t)op), tag, tag_arg);
  return u;
}

static PolyUOp *x86_ins(
    PolyCtx *ctx,
    PolyX86Op op,
    PolyDType dtype,
    PolyUOp **srcs,
    int n_src,
    int32_t def
) {
  int32_t defs[1] = {def};
  return x86_ins_ex(ctx, op, dtype, srcs, n_src, def ? defs : NULL, def ? 1 : 0, poly_arg_none());
}

static PolyUOp *x86_ins_nodef(
    PolyCtx *ctx,
    PolyX86Op op,
    PolyDType dtype,
    PolyUOp **srcs,
    int n_src
) {
  return x86_ins_ex(ctx, op, dtype, srcs, n_src, NULL, 0, poly_arg_none());
}

static PolyUOp *x86_ins_label(PolyCtx *ctx, const char *label) {
  return poly_uop_tagged_arg(
      ctx, POLY_OP_INS, POLY_VOID, NULL, 0, poly_arg_int(POLY_X86_LABEL), 0, x86_arg_string(label)
  );
}

static PolyUOp *x86_ins_jump(PolyCtx *ctx, PolyX86Op op, PolyUOp **srcs, int n_src, const char *label) {
  return poly_uop_tagged_arg(
      ctx, POLY_OP_INS, POLY_VOID, srcs, n_src, poly_arg_int((int64_t)op), 0, x86_arg_string(label)
  );
}

static PolyUOp *x86_define_reg(PolyCtx *ctx, PolyDType dtype, int32_t reg_tag) {
  if (reg_tag == 0)
    return x86_ins_ex(ctx, POLY_X86_DEFINE, dtype, NULL, 0, NULL, 0, poly_arg_none());
  int32_t defs[1] = {reg_tag};
  return x86_ins_ex(ctx, POLY_X86_DEFINE, dtype, NULL, 0, defs, 1, poly_arg_none());
}

static PolyUOp *x86_def_scratch(PolyCtx *ctx, PolyDType dtype) {
  return x86_define_reg(ctx, dtype, 0);
}

static PolyUOp *x86_stack_arg_slot(PolyCtx *ctx, int64_t caller_disp, int32_t reg_tag) {
  int32_t defs[1] = {reg_tag};
  return x86_ins_ex(
      ctx, POLY_X86_FRAME_INDEX, x86_u64(), NULL, 0, defs, 1, poly_arg_int(-caller_disp)
  );
}

static bool x86_const_as_i64(PolyUOp *u, int64_t *out) {
  if (!u || u->op != POLY_OP_CONST) return false;
  if (u->arg.kind == POLY_ARG_INT) {
    if (out) *out = u->arg.i;
    return true;
  }
  if (u->arg.kind == POLY_ARG_BOOL) {
    if (out) *out = u->arg.b ? 1 : 0;
    return true;
  }
  return false;
}

static PolyUOp *x86_imm_for_const(PolyCtx *ctx, PolyUOp *c) {
  int64_t v = 0;
  if (!x86_const_as_i64(c, &v)) return NULL;
  PolyDType dt = c->dtype;
  PolyDType scalar = poly_dtype_scalar(dt);
  if (poly_dtype_eq(scalar, POLY_INT64) || poly_dtype_eq(scalar, POLY_UINT64)) {
    int64_t sx = (int64_t)(int32_t)v;
    if ((uint64_t)sx != (uint64_t)v) return NULL;
    dt = POLY_INT32;
  }
  return x86_const_i(ctx, dt, v);
}

static int32_t x86_virtual_tag_for_dtype(PolyDType dt, bool buffer_value, int *next_vreg) {
  X86RegClass cls = x86_class_for_dtype(dt, buffer_value);
  return x86_tag_virtual(cls, (*next_vreg)++);
}

static int32_t x86_virtual_fixed_wgpr(int real_reg, int *next_vreg) {
  X86RegClass cls = X86_REG_CLASS_NONE;
  if (real_reg == X86_REG_RAX) cls = X86_REG_CLASS_FIXED_RAX;
  else if (real_reg == X86_REG_RDX) cls = X86_REG_CLASS_FIXED_RDX;
  else return 0;
  return x86_tag_virtual(cls, (*next_vreg)++);
}

static int x86_abi_gpr_for_arg(int arg_idx) {
  static const int sysv_args[6] = {
      X86_REG_RDI, X86_REG_RSI, X86_REG_RDX, X86_REG_RCX, X86_REG_R8, X86_REG_R9,
  };
  return (arg_idx >= 0 && arg_idx < 6) ? sysv_args[arg_idx] : -1;
}

typedef struct {
  PolyUOp **old_uops;
  PolyUOp **new_uops;
  int n;
  int cap;
} X86UOpMap;

static PolyUOp *x86_map_get(X86UOpMap *m, PolyUOp *old) {
  if (!old) return NULL;
  for (int i = m->n - 1; i >= 0; i--)
    if (m->old_uops[i] == old) return m->new_uops[i];
  return old;
}

static int x86_map_put(X86UOpMap *m, PolyUOp *old, PolyUOp *new_uop) {
  if (!old || !new_uop) return 0;
  for (int i = m->n - 1; i >= 0; i--) {
    if (m->old_uops[i] == old) {
      m->new_uops[i] = new_uop;
      return 0;
    }
  }
  if (m->n >= m->cap) {
    int nc = m->cap ? m->cap * 2 : 128;
    PolyUOp **no = realloc(m->old_uops, (size_t)nc * sizeof(*no));
    PolyUOp **nn = realloc(m->new_uops, (size_t)nc * sizeof(*nn));
    if (!no || !nn) {
      free(no);
      free(nn);
      return -1;
    }
    m->old_uops = no;
    m->new_uops = nn;
    m->cap = nc;
  }
  m->old_uops[m->n] = old;
  m->new_uops[m->n] = new_uop;
  m->n++;
  return 0;
}

static void x86_map_free(X86UOpMap *m) {
  if (!m) return;
  free(m->old_uops);
  free(m->new_uops);
  *m = (X86UOpMap){0};
}

typedef struct {
  PolyUOp **items;
  int n;
  int cap;
} X86UOpVec;

static int x86_vec_push(X86UOpVec *v, PolyUOp *u) {
  if (!u) return 0;
  if (v->n >= v->cap) {
    int nc = v->cap ? v->cap * 2 : 256;
    PolyUOp **ni = realloc(v->items, (size_t)nc * sizeof(*ni));
    if (!ni) return -1;
    v->items = ni;
    v->cap = nc;
  }
  v->items[v->n++] = u;
  return 0;
}

static int x86_replace_srcs(PolyUOp **dst, int max_dst, PolyUOp *u, X86UOpMap *map) {
  if (!u || u->n_src > max_dst) return -1;
  for (int i = 0; i < u->n_src; i++)
    dst[i] = x86_map_get(map, u->src[i]);
  return u->n_src;
}

static bool x86_dtype_is_float_bits(PolyDType dt, int bits) {
  PolyDType s = poly_dtype_scalar(dt);
  return poly_dtype_is_float(s) && s.bitsize == bits;
}

static bool x86_is_float_dtype(PolyDType dt) {
  return poly_dtype_is_float(poly_dtype_scalar(dt));
}

static bool x86_is_int_dtype(PolyDType dt) {
  PolyDType s = poly_dtype_scalar(dt);
  return poly_dtype_is_int(s) || poly_dtype_is_bool(s);
}

static PolyDType x86_int_for_float(PolyDType dt) {
  PolyDType s = poly_dtype_scalar(dt);
  if (s.bitsize == 16) return dt.count > 1 ? poly_dtype_vec(POLY_INT16, dt.count) : POLY_INT16;
  if (s.bitsize == 32) return dt.count > 1 ? poly_dtype_vec(POLY_INT32, dt.count) : POLY_INT32;
  return dt.count > 1 ? poly_dtype_vec(POLY_INT64, dt.count) : POLY_INT64;
}

static PolyDType x86_dtype_with_lanes(PolyDType scalar, int lanes) {
  return lanes > 1 ? poly_dtype_vec(scalar, lanes) : scalar;
}

static bool x86_dtype_is_float16(PolyDType dt) {
  PolyDType s = poly_dtype_scalar(dt);
  return poly_dtype_is_float(s) && s.bitsize == 16;
}

static bool x86_dtype_is_float32(PolyDType dt) {
  PolyDType s = poly_dtype_scalar(dt);
  return poly_dtype_is_float(s) && s.bitsize == 32;
}

static bool x86_dtype_is_float64(PolyDType dt) {
  PolyDType s = poly_dtype_scalar(dt);
  return poly_dtype_is_float(s) && s.bitsize == 64;
}

static bool x86_dtype_is_int_bits_any_sign(PolyDType dt, int bits) {
  PolyDType s = poly_dtype_scalar(dt);
  return poly_dtype_is_int(s) && s.bitsize == bits;
}

static bool x86_dtype_is_int8_or_bool(PolyDType dt) {
  PolyDType s = poly_dtype_scalar(dt);
  return poly_dtype_is_bool(s) || (poly_dtype_is_int(s) && s.bitsize == 8);
}

static bool x86_dtype_is_int16(PolyDType dt) {
  PolyDType s = poly_dtype_scalar(dt);
  return poly_dtype_is_int(s) && s.bitsize == 16;
}

static bool x86_op_is_comparison(PolyOps op) {
  return op == POLY_OP_CMPLT || op == POLY_OP_CMPEQ || op == POLY_OP_CMPNE;
}

static bool x86_op_is_alu_for_extra(PolyOps op) {
  switch (op) {
  case POLY_OP_CAST:
  case POLY_OP_BITCAST:
  case POLY_OP_EXP2:
  case POLY_OP_LOG2:
  case POLY_OP_SIN:
  case POLY_OP_SQRT:
  case POLY_OP_NEG:
  case POLY_OP_TRUNC:
  case POLY_OP_ADD:
  case POLY_OP_MUL:
  case POLY_OP_SHL:
  case POLY_OP_SHR:
  case POLY_OP_CDIV:
  case POLY_OP_MAX:
  case POLY_OP_CMOD:
  case POLY_OP_CMPLT:
  case POLY_OP_CMPNE:
  case POLY_OP_CMPEQ:
  case POLY_OP_XOR:
  case POLY_OP_OR:
  case POLY_OP_AND:
  case POLY_OP_SUB:
  case POLY_OP_FDIV:
  case POLY_OP_WHERE:
  case POLY_OP_MULACC:
    return true;
  default: return false;
  }
}

static PolyUOp *rule_x86_bool_cmp_legalize(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || !x86_op_is_comparison(u->op) || u->n_src != 2) return NULL;
  if (!poly_dtype_is_bool(poly_dtype_scalar(u->src[0]->dtype)) ||
      !poly_dtype_is_bool(poly_dtype_scalar(u->src[1]->dtype)))
    return NULL;
  PolyUOp *x = u->src[0];
  PolyUOp *y = u->src[1];
  if (u->op == POLY_OP_CMPNE)
    return poly_uop2(ctx, POLY_OP_XOR, u->dtype, x, y, poly_arg_none());
  if (u->op == POLY_OP_CMPEQ) {
    PolyUOp *xy = poly_uop2(ctx, POLY_OP_XOR, u->dtype, x, y, poly_arg_none());
    PolyUOp *t = poly_const_like_bool(ctx, xy, true);
    return poly_uop2(ctx, POLY_OP_XOR, u->dtype, xy, t, poly_arg_none());
  }
  PolyUOp *t = poly_const_like_bool(ctx, x, true);
  PolyUOp *not_x = poly_uop2(ctx, POLY_OP_XOR, x->dtype, x, t, poly_arg_none());
  return poly_uop2(ctx, POLY_OP_AND, u->dtype, not_x, y, poly_arg_none());
}

static PolyUOp *rule_x86_cast_legalize(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || u->op != POLY_OP_CAST || u->n_src != 1) return NULL;
  PolyUOp *y = u->src[0];
  PolyDType src = y->dtype;
  PolyDType dst = u->dtype;
  PolyDType dst_s = poly_dtype_scalar(dst);
  int lanes = dst.count > 1 ? dst.count : 1;

  if (x86_dtype_is_float16(src) &&
      (x86_dtype_is_float64(dst) || poly_dtype_is_int(dst_s) || poly_dtype_is_bool(dst_s))) {
    PolyUOp *f32 = poly_uop1(ctx, POLY_OP_CAST, x86_dtype_with_lanes(POLY_FLOAT32, lanes), y, poly_arg_none());
    return poly_uop1(ctx, POLY_OP_CAST, dst, f32, u->arg);
  }

  if ((x86_dtype_is_float64(src) || poly_dtype_is_int(poly_dtype_scalar(src)) ||
       poly_dtype_is_bool(poly_dtype_scalar(src))) &&
      x86_dtype_is_float16(dst)) {
    PolyUOp *f32 = poly_uop1(ctx, POLY_OP_CAST, x86_dtype_with_lanes(POLY_FLOAT32, lanes), y, poly_arg_none());
    return poly_uop1(ctx, POLY_OP_CAST, dst, f32, u->arg);
  }

  if (x86_is_float_dtype(src) && (x86_dtype_is_int8_or_bool(dst) || x86_dtype_is_int16(dst))) {
    PolyUOp *i32 = poly_uop1(ctx, POLY_OP_CAST, x86_dtype_with_lanes(POLY_INT32, lanes), y, poly_arg_none());
    return poly_uop1(ctx, POLY_OP_CAST, dst, i32, u->arg);
  }

  if ((x86_dtype_is_int8_or_bool(src) || x86_dtype_is_int16(src)) && x86_is_float_dtype(dst)) {
    PolyUOp *i32 = poly_uop1(ctx, POLY_OP_CAST, x86_dtype_with_lanes(POLY_INT32, src.count), y, poly_arg_none());
    return poly_uop1(ctx, POLY_OP_CAST, dst, i32, u->arg);
  }

  if (poly_dtype_eq(poly_dtype_scalar(src), POLY_UINT32) && x86_is_float_dtype(dst)) {
    PolyUOp *i64 = poly_uop1(ctx, POLY_OP_CAST, x86_dtype_with_lanes(POLY_INT64, src.count), y, poly_arg_none());
    return poly_uop1(ctx, POLY_OP_CAST, dst, i64, u->arg);
  }

  if (poly_dtype_eq(poly_dtype_scalar(src), POLY_UINT64) && x86_is_float_dtype(dst)) {
    PolyUOp *one = poly_const_like_int(ctx, y, 1);
    PolyUOp *two_f = poly_const_like_float(ctx, u, 2.0);
    PolyUOp *shr = poly_uop2(ctx, POLY_OP_SHR, src, y, one, poly_arg_none());
    PolyUOp *shr_i64 = poly_uop1(ctx, POLY_OP_CAST, x86_dtype_with_lanes(POLY_INT64, src.count), shr, poly_arg_none());
    PolyUOp *hi = poly_uop2(
        ctx, POLY_OP_MUL, dst,
        poly_uop1(ctx, POLY_OP_CAST, dst, shr_i64, poly_arg_none()), two_f, poly_arg_none()
    );
    PolyUOp *lo_bits = poly_uop2(ctx, POLY_OP_AND, src, y, one, poly_arg_none());
    PolyUOp *lo_i64 = poly_uop1(ctx, POLY_OP_CAST, x86_dtype_with_lanes(POLY_INT64, src.count), lo_bits, poly_arg_none());
    PolyUOp *lo = poly_uop1(ctx, POLY_OP_CAST, dst, lo_i64, poly_arg_none());
    return poly_uop2(ctx, POLY_OP_ADD, dst, hi, lo, poly_arg_none());
  }

  return NULL;
}

static PolyUOp *rule_x86_int8_mul_legalize(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || u->op != POLY_OP_MUL || u->n_src != 2 || !x86_dtype_is_int_bits_any_sign(u->dtype, 8))
    return NULL;
  PolyDType i16 = x86_dtype_with_lanes(POLY_INT16, u->dtype.count);
  PolyUOp *a = poly_uop1(ctx, POLY_OP_CAST, i16, u->src[0], poly_arg_none());
  PolyUOp *bb = poly_uop1(ctx, POLY_OP_CAST, i16, u->src[1], poly_arg_none());
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, i16, a, bb, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_CAST, u->dtype, mul, poly_arg_none());
}

static PolyUOp *rule_x86_int8_where_legalize(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || u->op != POLY_OP_WHERE || u->n_src != 3 || u->dtype.count != 1 ||
      !x86_dtype_is_int8_or_bool(u->dtype))
    return NULL;
  PolyUOp *a = poly_uop1(ctx, POLY_OP_CAST, POLY_INT16, u->src[1], poly_arg_none());
  PolyUOp *bb = poly_uop1(ctx, POLY_OP_CAST, POLY_INT16, u->src[2], poly_arg_none());
  PolyUOp *w = poly_uop3(ctx, POLY_OP_WHERE, POLY_INT16, u->src[0], a, bb, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_CAST, u->dtype, w, poly_arg_none());
}

static PolyUOp *rule_x86_float16_alu_legalize(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || !x86_op_is_alu_for_extra(u->op) || !x86_dtype_is_float16(u->dtype)) return NULL;
  if (u->op == POLY_OP_CAST || u->op == POLY_OP_BITCAST || x86_op_is_comparison(u->op)) return NULL;
  PolyDType f32 = x86_dtype_with_lanes(POLY_FLOAT32, u->dtype.count);
  PolyUOp *srcs[8];
  if (u->n_src > 8) return NULL;
  for (int i = 0; i < u->n_src; i++) {
    if (poly_dtype_is_bool(poly_dtype_scalar(u->src[i]->dtype)))
      srcs[i] = u->src[i];
    else
      srcs[i] = poly_uop1(ctx, POLY_OP_CAST, f32, u->src[i], poly_arg_none());
  }
  PolyUOp *inner = poly_uop(ctx, u->op, f32, srcs, u->n_src, u->arg);
  return poly_uop1(ctx, POLY_OP_CAST, u->dtype, inner, poly_arg_none());
}

static PolyUOp *rule_x86_float16_cmp_legalize(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || !x86_op_is_comparison(u->op) || u->n_src != 2 || !x86_dtype_is_float16(u->src[0]->dtype))
    return NULL;
  PolyDType f32 = x86_dtype_with_lanes(POLY_FLOAT32, u->src[0]->dtype.count);
  PolyUOp *a = poly_uop1(ctx, POLY_OP_CAST, f32, u->src[0], poly_arg_none());
  PolyUOp *bb = poly_uop1(ctx, POLY_OP_CAST, f32, u->src[1], poly_arg_none());
  return poly_uop2(ctx, u->op, u->dtype, a, bb, poly_arg_none());
}

static PolyUOp *rule_x86_mixed_float_int_cmp_legalize(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || !x86_op_is_comparison(u->op) || u->n_src != 2) return NULL;
  PolyDType a_dt = u->src[0]->dtype;
  PolyDType b_dt = u->src[1]->dtype;
  bool a_float = x86_is_float_dtype(a_dt);
  bool b_float = x86_is_float_dtype(b_dt);
  if (a_float == b_float) return NULL;

  PolyDType int_dt = a_float ? b_dt : a_dt;
  PolyDType int_s = poly_dtype_scalar(int_dt);
  if (!poly_dtype_is_int(int_s) || poly_dtype_is_bool(int_s)) return NULL;

  PolyDType float_dt = a_float ? a_dt : b_dt;
  int float_lanes = float_dt.count > 1 ? float_dt.count : 1;
  int int_lanes = int_dt.count > 1 ? int_dt.count : 1;
  if (float_lanes != int_lanes) return NULL;

  PolyDType cmp_dt = x86_dtype_is_float16(float_dt)
                         ? x86_dtype_with_lanes(POLY_FLOAT32, float_lanes)
                         : float_dt;
  PolyUOp *a = a_float ? u->src[0] : poly_uop1(ctx, POLY_OP_CAST, cmp_dt, u->src[0], poly_arg_none());
  PolyUOp *bb = b_float ? u->src[1] : poly_uop1(ctx, POLY_OP_CAST, cmp_dt, u->src[1], poly_arg_none());
  if (a_float && !poly_dtype_eq(a->dtype, cmp_dt))
    a = poly_uop1(ctx, POLY_OP_CAST, cmp_dt, a, poly_arg_none());
  if (b_float && !poly_dtype_eq(bb->dtype, cmp_dt))
    bb = poly_uop1(ctx, POLY_OP_CAST, cmp_dt, bb, poly_arg_none());
  return poly_uop2(ctx, u->op, u->dtype, a, bb, poly_arg_none());
}

static PolyUOp *rule_x86_packed_int_cmpne_legalize(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || u->op != POLY_OP_CMPNE || u->n_src != 2 || u->src[0]->dtype.count <= 1 ||
      !poly_dtype_is_int(poly_dtype_scalar(u->src[0]->dtype)))
    return NULL;
  PolyUOp *eq = poly_uop2(ctx, POLY_OP_CMPEQ, u->dtype, u->src[0], u->src[1], poly_arg_none());
  return poly_uop2(ctx, POLY_OP_XOR, u->dtype, eq, poly_const_like_bool(ctx, eq, true), poly_arg_none());
}

static PolyUOp *rule_x86_float_where_mask_legalize(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || u->op != POLY_OP_WHERE || u->n_src != 3 || !x86_is_float_dtype(u->dtype)) return NULL;
  PolyUOp *m = u->src[0];
  if (!m || !poly_dtype_is_bool(poly_dtype_scalar(m->dtype))) return NULL;
  if (x86_op_is_comparison(m->op) && m->n_src == 2 && x86_is_float_dtype(m->src[0]->dtype)) {
    PolyUOp *mask = poly_uop2(ctx, m->op, u->dtype, m->src[0], m->src[1], m->arg);
    return poly_uop3(ctx, POLY_OP_WHERE, u->dtype, mask, u->src[1], u->src[2], u->arg);
  }
  if (m->n_src > 0 && x86_is_float_dtype(m->src[0]->dtype)) return NULL;
  PolyUOp *cast = poly_uop1(ctx, POLY_OP_CAST, u->dtype, m, poly_arg_none());
  PolyUOp *zero = poly_const_like_float(ctx, cast, 0.0);
  PolyUOp *mask = poly_uop2(ctx, POLY_OP_CMPNE, u->dtype, cast, zero, poly_arg_none());
  return poly_uop3(ctx, POLY_OP_WHERE, u->dtype, mask, u->src[1], u->src[2], poly_arg_none());
}

static PolyUOp *rule_x86_neg_to_sub(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || u->op != POLY_OP_NEG || u->n_src != 1) return NULL;
  return poly_uop2(ctx, POLY_OP_SUB, u->dtype, poly_const_like_int(ctx, u, 0), u->src[0], poly_arg_none());
}

static PolyUOp *rule_x86_cmod_to_cdiv(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || u->op != POLY_OP_CMOD || u->n_src != 2) return NULL;
  PolyUOp *q = poly_uop2(ctx, POLY_OP_CDIV, u->dtype, u->src[0], u->src[1], poly_arg_none());
  PolyUOp *prod = poly_uop2(ctx, POLY_OP_MUL, u->dtype, u->src[1], q, poly_arg_none());
  return poly_uop2(ctx, POLY_OP_SUB, u->dtype, u->src[0], prod, poly_arg_none());
}

static _Thread_local PolyPatternMatcher *g_pm_x86_extra = NULL;
static PolyPatternMatcher *poly_pm_x86_extra(void) {
  if (g_pm_x86_extra) return g_pm_x86_extra;
  PolyRule rules[] = {
      {poly_pat_op(POLY_OP_CMPNE, NULL, 0, NULL), rule_x86_bool_cmp_legalize},
      {poly_pat_op(POLY_OP_CMPEQ, NULL, 0, NULL), rule_x86_bool_cmp_legalize},
      {poly_pat_op(POLY_OP_CMPLT, NULL, 0, NULL), rule_x86_bool_cmp_legalize},
      {poly_pat_op(POLY_OP_CAST, NULL, 0, NULL), rule_x86_cast_legalize},
      {poly_pat_op(POLY_OP_MUL, NULL, 0, NULL), rule_x86_int8_mul_legalize},
      {poly_pat_op(POLY_OP_WHERE, NULL, 0, NULL), rule_x86_int8_where_legalize},
      {poly_pat_ops(POLY_GROUP_ALU, NULL, 0, NULL), rule_x86_float16_alu_legalize},
      {poly_pat_op(POLY_OP_CMPLT, NULL, 0, NULL), rule_x86_float16_cmp_legalize},
      {poly_pat_op(POLY_OP_CMPEQ, NULL, 0, NULL), rule_x86_float16_cmp_legalize},
      {poly_pat_op(POLY_OP_CMPNE, NULL, 0, NULL), rule_x86_float16_cmp_legalize},
      {poly_pat_op(POLY_OP_CMPLT, NULL, 0, NULL), rule_x86_mixed_float_int_cmp_legalize},
      {poly_pat_op(POLY_OP_CMPEQ, NULL, 0, NULL), rule_x86_mixed_float_int_cmp_legalize},
      {poly_pat_op(POLY_OP_CMPNE, NULL, 0, NULL), rule_x86_mixed_float_int_cmp_legalize},
      {poly_pat_op(POLY_OP_CMPNE, NULL, 0, NULL), rule_x86_packed_int_cmpne_legalize},
      {poly_pat_op(POLY_OP_WHERE, NULL, 0, NULL), rule_x86_float_where_mask_legalize},
      {poly_pat_op(POLY_OP_NEG, NULL, 0, NULL), rule_x86_neg_to_sub},
      {poly_pat_op(POLY_OP_CMOD, NULL, 0, NULL), rule_x86_cmod_to_cdiv},
  };
  g_pm_x86_extra = poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0])));
  return g_pm_x86_extra;
}

static bool x86_pre_isel_op_has_structural_width(PolyOps op) {
  switch (op) {
  case POLY_OP_CAST:
  case POLY_OP_BITCAST:
  case POLY_OP_EXP2:
  case POLY_OP_LOG2:
  case POLY_OP_SIN:
  case POLY_OP_SQRT:
  case POLY_OP_NEG:
  case POLY_OP_TRUNC:
  case POLY_OP_ADD:
  case POLY_OP_MUL:
  case POLY_OP_SHL:
  case POLY_OP_SHR:
  case POLY_OP_CDIV:
  case POLY_OP_MAX:
  case POLY_OP_CMPLT:
  case POLY_OP_CMPNE:
  case POLY_OP_CMPEQ:
  case POLY_OP_XOR:
  case POLY_OP_OR:
  case POLY_OP_AND:
  case POLY_OP_SUB:
  case POLY_OP_FDIV:
  case POLY_OP_WHERE:
  case POLY_OP_MULACC:
    return true;
  default:
    return false;
  }
}

static PolyUOp *rule_x86_pre_isel_structural_width(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || u->dtype.is_ptr || poly_dtype_eq(u->dtype, POLY_VOID)) return NULL;
  int c = 1;
  if (u->op == POLY_OP_STACK) {
    c = u->n_src;
  } else if (x86_pre_isel_op_has_structural_width(u->op)) {
    for (int i = 0; i < u->n_src; i++) {
      if (!u->src[i] || u->src[i]->dtype.is_ptr) continue;
      if (u->src[i]->dtype.count > c) c = u->src[i]->dtype.count;
    }
  } else {
    return NULL;
  }
  if (c <= 1 || c == u->dtype.count) return NULL;
  PolyDType dt = poly_dtype_vec(poly_dtype_scalar(u->dtype), c);
  return poly_uop_tagged_arg(ctx, u->op, dt, u->src, u->n_src, u->arg, u->tag, u->tag_arg);
}

static PolyUOp *rule_x86_pre_isel_load_shrink_width(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || u->op != POLY_OP_LOAD || u->n_src < 1 || u->dtype.is_ptr ||
      poly_dtype_eq(u->dtype, POLY_VOID))
    return NULL;
  PolyUOp *addr = u->src[0];
  if (!addr || addr->op != POLY_OP_SHRINK || addr->n_src < 3) return NULL;
  int64_t c = 0;
  if (!x86_const_as_i64(addr->src[2], &c) || c <= u->dtype.count) return NULL;
  PolyDType dt = poly_dtype_vec(poly_dtype_scalar(u->dtype), (int)c);
  return poly_uop_tagged_arg(ctx, u->op, dt, u->src, u->n_src, u->arg, u->tag, u->tag_arg);
}

static PolyUOp *x86_pre_isel_addr_as_u64(PolyCtx *ctx, PolyUOp *addr) {
  if (!addr) return NULL;
  return poly_uop_tagged_arg(ctx, addr->op, POLY_UINT64, addr->src, addr->n_src,
                             addr->arg, addr->tag, addr->tag_arg);
}

static PolyUOp *x86_pre_isel_local_buffer(PolyCtx *ctx, PolyDType elem_dt, int count) {
  if (!ctx) return NULL;
  if (count <= 0) count = 1;
  PolyDType ptr = poly_dtype_ptr(poly_dtype_scalar(elem_dt), count, POLY_ADDR_LOCAL);
  int32_t tag = ctx->next_buf_tag++;
  return poly_uop_tagged(ctx, POLY_OP_BUFFER, ptr, NULL, 0, poly_arg_int(tag), tag);
}

static PolyUOp *rule_x86_pre_isel_gated_load(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || u->op != POLY_OP_LOAD || u->n_src != 3) return NULL;
  PolyUOp *addr = u->src[0];
  if (!addr || (addr->op != POLY_OP_INDEX && addr->op != POLY_OP_SHRINK) || addr->n_src < 2)
    return NULL;
  PolyUOp *alt = u->src[1];
  PolyUOp *gate = u->src[2];
  if (!alt || !gate || !poly_dtype_is_bool(poly_dtype_scalar(gate->dtype))) return NULL;

  int count = u->dtype.count > 1 ? u->dtype.count : 1;
  PolyUOp *local = x86_pre_isel_local_buffer(ctx, poly_dtype_scalar(addr->src[0]->dtype), count);
  if (!local) return NULL;
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *local_idx = poly_uop2(ctx, POLY_OP_INDEX, POLY_UINT64, local, zero, poly_arg_none());
  PolyUOp *addr_u64 = x86_pre_isel_addr_as_u64(ctx, addr);
  if (!addr_u64) return NULL;
  PolyUOp *sel_srcs[3] = {gate, addr_u64, local_idx};
  PolyUOp *sel = poly_uop(ctx, POLY_OP_WHERE, POLY_UINT64, sel_srcs, 3, poly_arg_none());

  PolyUOp *store_addr = count == 1 ? local_idx : local;
  PolyUOp *scratch_store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, store_addr, alt, poly_arg_none());
  PolyUOp *after_srcs[2] = {sel, scratch_store};
  PolyUOp *ptr = poly_uop(ctx, POLY_OP_AFTER, addr->dtype, after_srcs, 2, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_LOAD, u->dtype, ptr, u->arg);
}

static PolyUOp *rule_x86_pre_isel_gated_store(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || u->op != POLY_OP_STORE || u->n_src != 3) return NULL;
  PolyUOp *addr = u->src[0];
  if (!addr || (addr->op != POLY_OP_INDEX && addr->op != POLY_OP_SHRINK) || addr->n_src < 2)
    return NULL;
  PolyUOp *val = u->src[1];
  PolyUOp *gate = u->src[2];
  if (!val || !gate || !poly_dtype_is_bool(poly_dtype_scalar(gate->dtype))) return NULL;

  int count = val->dtype.count > 1 ? val->dtype.count : 1;
  PolyUOp *local = x86_pre_isel_local_buffer(ctx, poly_dtype_scalar(addr->src[0]->dtype), count);
  if (!local) return NULL;
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *local_idx = poly_uop2(ctx, POLY_OP_INDEX, POLY_UINT64, local, zero, poly_arg_none());
  PolyUOp *addr_u64 = x86_pre_isel_addr_as_u64(ctx, addr);
  if (!addr_u64) return NULL;
  PolyUOp *sel_srcs[3] = {gate, addr_u64, local_idx};
  PolyUOp *sel = poly_uop(ctx, POLY_OP_WHERE, POLY_UINT64, sel_srcs, 3, poly_arg_none());
  PolyUOp *ptr = poly_uop1(ctx, POLY_OP_AFTER, addr->dtype, sel, poly_arg_none());
  return poly_uop2(ctx, POLY_OP_STORE, u->dtype, ptr, val, u->arg);
}

static PolyUOp *x86_pre_isel_noop(PolyCtx *ctx, PolyDType dtype, PolyUOp *src) {
  return poly_uop1(ctx, POLY_OP_NOOP, dtype, src, poly_arg_none());
}

static PolyUOp *rule_x86_pre_isel_cast_noop(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || u->op != POLY_OP_CAST || u->n_src != 1) return NULL;
  PolyUOp *y = u->src[0];
  PolyDType ys = poly_dtype_scalar(y->dtype);
  PolyDType xs = poly_dtype_scalar(u->dtype);
  if (poly_dtype_eq(ys, POLY_UINT32) && poly_dtype_is_int(xs) &&
      poly_dtype_itemsize(xs) == 8 && y->dtype.count == 1)
    return x86_pre_isel_noop(ctx, u->dtype, y);
  if ((poly_dtype_is_int(ys) || poly_dtype_is_bool(ys)) && poly_dtype_is_int(xs) &&
      poly_dtype_itemsize(ys) == poly_dtype_itemsize(xs))
    return x86_pre_isel_noop(ctx, u->dtype, y);
  if (poly_dtype_is_int(ys) && poly_dtype_is_int(xs) &&
      poly_dtype_itemsize(xs) < poly_dtype_itemsize(ys) && y->dtype.count == 1)
    return x86_pre_isel_noop(ctx, u->dtype, y);
  return NULL;
}

static PolyUOp *rule_x86_pre_isel_bitcast_noop(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || u->op != POLY_OP_BITCAST || u->n_src != 1) return NULL;
  PolyDType ys = poly_dtype_scalar(u->src[0]->dtype);
  PolyDType xs = poly_dtype_scalar(u->dtype);
  bool real_scalar_float_int = u->src[0]->dtype.count == 1 && u->dtype.count == 1 &&
      ((poly_dtype_is_float(ys) && poly_dtype_is_int(xs)) ||
       (poly_dtype_is_int(ys) && poly_dtype_is_float(xs)));
  if (real_scalar_float_int) return NULL;
  return x86_pre_isel_noop(ctx, u->dtype, u->src[0]);
}

static PolyUOp *rule_x86_pre_isel_noop_of_noop(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || u->op != POLY_OP_NOOP || u->n_src != 1 || u->src[0]->op != POLY_OP_NOOP ||
      u->src[0]->n_src != 1)
    return NULL;
  return x86_pre_isel_noop(ctx, u->dtype, u->src[0]->src[0]);
}

static int x86_const_lane_from_uop(PolyUOp *u, bool *ok) {
  *ok = false;
  if (!u) return 0;
  if (u->op == POLY_OP_GEP) {
    if (u->arg.kind == POLY_ARG_INT) {
      *ok = true;
      return (int)u->arg.i;
    }
    if (u->arg.kind == POLY_ARG_INT_TUPLE && u->arg.int_tuple.n > 0) {
      *ok = true;
      return (int)u->arg.int_tuple.vals[0];
    }
    return 0;
  }
  if (u->op == POLY_OP_INDEX && u->n_src >= 2) {
    int64_t v = 0;
    if (x86_const_as_i64(u->src[1], &v)) {
      *ok = true;
      return (int)v;
    }
  }
  return 0;
}

static PolyUOp *x86_lane_base_uop(PolyUOp *u) {
  if (!u) return NULL;
  if ((u->op == POLY_OP_GEP || u->op == POLY_OP_INDEX) && u->n_src >= 1)
    return u->src[0];
  return NULL;
}

static PolyUOp *rule_x86_pre_isel_stack_sequential_noop(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || u->op != POLY_OP_STACK || u->n_src <= 1) return NULL;
  PolyUOp *base = x86_lane_base_uop(u->src[0]);
  if (!base) return NULL;
  for (int i = 0; i < u->n_src; i++) {
    if (x86_lane_base_uop(u->src[i]) != base) return NULL;
    bool ok = false;
    int lane = x86_const_lane_from_uop(u->src[i], &ok);
    if (!ok || lane != i) return NULL;
  }
  return x86_pre_isel_noop(ctx, u->dtype, base);
}

static PolyUOp *rule_x86_pre_isel_scalar_where_gate_compare(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || u->op != POLY_OP_WHERE || u->n_src != 3 || u->dtype.count != 1) return NULL;
  PolyUOp *m = u->src[0];
  if (!m || !poly_dtype_is_bool(poly_dtype_scalar(m->dtype)) || x86_op_is_comparison(m->op)) return NULL;
  PolyUOp *zero = poly_const_like_int(ctx, m, 0);
  PolyUOp *cmp = poly_uop2(ctx, POLY_OP_CMPNE, POLY_BOOL, m, zero, poly_arg_none());
  return poly_uop3(ctx, POLY_OP_WHERE, u->dtype, cmp, u->src[1], u->src[2], u->arg);
}

static _Thread_local PolyPatternMatcher *g_pm_x86_pre_isel = NULL;
static PolyPatternMatcher *poly_pm_x86_pre_isel(void) {
  if (g_pm_x86_pre_isel) return g_pm_x86_pre_isel;
  PolyOpSet pre_set = {{0, 0}};
  pre_set = poly_opset_add(pre_set, POLY_OP_STACK);
  pre_set = poly_opset_add(pre_set, POLY_OP_CAST);
  pre_set = poly_opset_add(pre_set, POLY_OP_BITCAST);
  pre_set = poly_opset_add(pre_set, POLY_OP_EXP2);
  pre_set = poly_opset_add(pre_set, POLY_OP_LOG2);
  pre_set = poly_opset_add(pre_set, POLY_OP_SIN);
  pre_set = poly_opset_add(pre_set, POLY_OP_SQRT);
  pre_set = poly_opset_add(pre_set, POLY_OP_NEG);
  pre_set = poly_opset_add(pre_set, POLY_OP_TRUNC);
  pre_set = poly_opset_add(pre_set, POLY_OP_ADD);
  pre_set = poly_opset_add(pre_set, POLY_OP_MUL);
  pre_set = poly_opset_add(pre_set, POLY_OP_SHL);
  pre_set = poly_opset_add(pre_set, POLY_OP_SHR);
  pre_set = poly_opset_add(pre_set, POLY_OP_CDIV);
  pre_set = poly_opset_add(pre_set, POLY_OP_MAX);
  pre_set = poly_opset_add(pre_set, POLY_OP_CMPLT);
  pre_set = poly_opset_add(pre_set, POLY_OP_CMPNE);
  pre_set = poly_opset_add(pre_set, POLY_OP_CMPEQ);
  pre_set = poly_opset_add(pre_set, POLY_OP_XOR);
  pre_set = poly_opset_add(pre_set, POLY_OP_OR);
  pre_set = poly_opset_add(pre_set, POLY_OP_AND);
  pre_set = poly_opset_add(pre_set, POLY_OP_SUB);
  pre_set = poly_opset_add(pre_set, POLY_OP_FDIV);
  pre_set = poly_opset_add(pre_set, POLY_OP_WHERE);
  pre_set = poly_opset_add(pre_set, POLY_OP_MULACC);
  PolyRule rules[] = {
      {poly_pat_op(POLY_OP_LOAD, NULL, 0, NULL), rule_x86_pre_isel_load_shrink_width},
      {poly_pat_op(POLY_OP_NEG, NULL, 0, NULL), rule_x86_neg_to_sub},
      {poly_pat_ops(pre_set, NULL, 0, NULL), rule_x86_pre_isel_structural_width},
      {poly_pat_op(POLY_OP_CAST, NULL, 0, NULL), rule_x86_pre_isel_cast_noop},
      {poly_pat_op(POLY_OP_BITCAST, NULL, 0, NULL), rule_x86_pre_isel_bitcast_noop},
      {poly_pat_op(POLY_OP_NOOP, NULL, 0, NULL), rule_x86_pre_isel_noop_of_noop},
      {poly_pat_op(POLY_OP_STACK, NULL, 0, NULL), rule_x86_pre_isel_stack_sequential_noop},
      {poly_pat_op(POLY_OP_LOAD, NULL, 0, NULL), rule_x86_pre_isel_gated_load},
      {poly_pat_op(POLY_OP_STORE, NULL, 0, NULL), rule_x86_pre_isel_gated_store},
      {poly_pat_op(POLY_OP_WHERE, NULL, 0, NULL), rule_x86_pre_isel_scalar_where_gate_compare},
  };
  g_pm_x86_pre_isel = poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0])));
  return g_pm_x86_pre_isel;
}

typedef struct {
  PolyMap *uses;
  PolyMap *single_consumer;
  PolyMap *structural_consts;
  PolyMap *address_memory_uses;
  PolyUOp **func_args;
  int n_func_args;
  int next_vreg;
} X86IselUseCtx;

static int x86_func_arg_key0(PolyUOp *u) {
  if (!u) return 3;
  if (u->op == POLY_OP_SPECIAL) return 2;
  if (u->op == POLY_OP_PARAM && u->arg.kind == POLY_ARG_INT) return 0;
  return 1;
}

static int64_t x86_func_arg_key1(PolyUOp *u) {
  if (!u) return 0;
  if ((u->op == POLY_OP_PARAM || u->op == POLY_OP_SPECIAL) && u->arg.kind == POLY_ARG_INT)
    return u->arg.i;
  return (int64_t)(uintptr_t)u;
}

static bool x86_func_arg_less(PolyUOp *a, PolyUOp *b) {
  int ak0 = x86_func_arg_key0(a), bk0 = x86_func_arg_key0(b);
  if (ak0 != bk0) return ak0 < bk0;
  int64_t ak1 = x86_func_arg_key1(a), bk1 = x86_func_arg_key1(b);
  if (ak1 != bk1) return ak1 < bk1;
  return (uintptr_t)a < (uintptr_t)b;
}

static int x86_func_arg_index(X86IselUseCtx *uctx, PolyUOp *u) {
  if (!uctx || !u) return -1;
  for (int i = 0; i < uctx->n_func_args; i++)
    if (uctx->func_args[i] == u) return i;
  return -1;
}

static uintptr_t x86_isel_use_count(X86IselUseCtx *uctx, PolyUOp *u) {
  if (!uctx || !uctx->uses || !u) return 0;
  return (uintptr_t)poly_map_get(uctx->uses, poly_ptr_hash(u), u, poly_ptr_eq);
}

static PolyUOp *x86_isel_single_consumer(X86IselUseCtx *uctx, PolyUOp *u) {
  if (!uctx || !uctx->single_consumer || !u || x86_isel_use_count(uctx, u) != 1) return NULL;
  return (PolyUOp *)poly_map_get(uctx->single_consumer, poly_ptr_hash(u), u, poly_ptr_eq);
}

static bool x86_isel_is_structural_const(X86IselUseCtx *uctx, PolyUOp *u) {
  return uctx && uctx->structural_consts && u &&
         poly_map_get(uctx->structural_consts, poly_ptr_hash(u), u, poly_ptr_eq) != NULL;
}

static bool x86_isel_is_memory_address_use(X86IselUseCtx *uctx, PolyUOp *u) {
  return uctx && uctx->address_memory_uses && u &&
         poly_map_get(uctx->address_memory_uses, poly_ptr_hash(u), u, poly_ptr_eq) != NULL;
}

static void x86_isel_inc_use(PolyMap *uses, PolyUOp *u) {
  if (!uses || !u) return;
  uintptr_t n = (uintptr_t)poly_map_get(uses, poly_ptr_hash(u), u, poly_ptr_eq);
  poly_map_set(uses, poly_ptr_hash(u), u, (void *)(n + 1), poly_ptr_eq);
}

static void x86_isel_note_consumer(PolyMap *map, PolyUOp *producer, PolyUOp *consumer) {
  if (!map || !producer || !consumer) return;
  if (!poly_map_get(map, poly_ptr_hash(producer), producer, poly_ptr_eq))
    poly_map_set(map, poly_ptr_hash(producer), producer, consumer, poly_ptr_eq);
}

static void x86_isel_mark_structural_const(PolyMap *map, PolyUOp *u) {
  if (!map || !u || u->op != POLY_OP_CONST) return;
  poly_map_set(map, poly_ptr_hash(u), u, u, poly_ptr_eq);
}

static void x86_isel_mark_memory_address_use(PolyMap *map, PolyUOp *u) {
  if (!map || !u || (u->op != POLY_OP_INDEX && u->op != POLY_OP_SHRINK)) return;
  poly_map_set(map, poly_ptr_hash(u), u, u, poly_ptr_eq);
}

static int x86_src_count(PolyUOp *consumer, PolyUOp *src) {
  if (!consumer || !src) return 0;
  int n = 0;
  for (int i = 0; i < consumer->n_src; i++)
    if (consumer->src[i] == src) n++;
  return n;
}

static bool x86_isel_foldable(X86IselUseCtx *uctx, PolyUOp *consumer, PolyUOp *producer) {
  return x86_isel_use_count(uctx, producer) == 1 && x86_src_count(consumer, producer) == 1;
}

static bool x86_dtype_can_fma(PolyDType dt) {
  if (!x86_is_float_dtype(dt)) return false;
  int bits = poly_dtype_scalar(dt).bitsize;
  return bits == 32 || bits == 64;
}

static int32_t x86_graph_vreg(PolyDType dt, bool buffer_value) {
  X86IselUseCtx *uctx = (X86IselUseCtx *)poly_graph_rewrite_userctx();
  int local_next = 0;
  return x86_virtual_tag_for_dtype(dt, buffer_value, uctx ? &uctx->next_vreg : &local_next);
}

static int32_t x86_graph_fixed_wgpr(int real_reg) {
  X86IselUseCtx *uctx = (X86IselUseCtx *)poly_graph_rewrite_userctx();
  int local_next = 0;
  return x86_virtual_fixed_wgpr(real_reg, uctx ? &uctx->next_vreg : &local_next);
}

static PolyDType x86_graph_abi_dtype(PolyUOp *u) {
  if (u && u->op == POLY_OP_PARAM) return x86_u64();
  return u ? u->dtype : POLY_VOID;
}

static PolyUOp *rule_x86_isel_abi_graph(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || (u->op != POLY_OP_PARAM && u->op != POLY_OP_SPECIAL)) return NULL;
  if (x86_uop_reg(u) >= 0) return NULL;
  X86IselUseCtx *uctx = (X86IselUseCtx *)poly_graph_rewrite_userctx();
  int arg_idx = (u->op == POLY_OP_PARAM && u->arg.kind == POLY_ARG_INT)
                    ? (int)u->arg.i
                    : x86_func_arg_index(uctx, u);
  int abi = x86_abi_gpr_for_arg(arg_idx);
  if (abi < 0) return NULL;
  PolyDType dt = x86_graph_abi_dtype(u);
  PolyUOp *arg = poly_uop_tagged_arg(
      ctx, u->op, dt, NULL, 0, u->arg,
      x86_tag_real(X86_REG_CLASS_WGPR, abi), poly_arg_none()
  );
  PolyUOp *srcs[1] = {arg};
  return x86_ins(ctx, POLY_X86_MOV, dt, srcs, 1, x86_graph_vreg(dt, u->op == POLY_OP_PARAM));
}

static int64_t x86_float_bits(PolyDType dt, double v);

static PolyUOp *rule_x86_isel_const_graph(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || u->op != POLY_OP_CONST || x86_const_is_imm(u) || u->dtype.count > 1)
    return NULL;
  X86IselUseCtx *uctx = (X86IselUseCtx *)poly_graph_rewrite_userctx();
  if (x86_isel_is_structural_const(uctx, u)) return NULL;
  if (u->arg.kind == POLY_ARG_INT || u->arg.kind == POLY_ARG_BOOL) {
    int64_t v = u->arg.kind == POLY_ARG_BOOL ? (u->arg.b ? 1 : 0) : u->arg.i;
    PolyUOp *imm = x86_const_i(ctx, u->dtype, v);
    PolyUOp *srcs[1] = {imm};
    PolyX86Op op = x86_value_size(u->dtype, false) == 8 ? POLY_X86_MOVABS : POLY_X86_MOVi;
    return x86_ins(ctx, op, u->dtype, srcs, 1, x86_graph_vreg(u->dtype, false));
  }
  if (u->arg.kind == POLY_ARG_FLOAT && x86_is_float_dtype(u->dtype)) {
    PolyDType idt = x86_int_for_float(u->dtype);
    PolyUOp *bits = poly_uop0(ctx, POLY_OP_CONST, idt, poly_arg_int(x86_float_bits(u->dtype, u->arg.f)));
    return poly_uop1(ctx, POLY_OP_BITCAST, u->dtype, bits, poly_arg_none());
  }
  return NULL;
}

static PolyUOp *rule_x86_isel_cast_void_graph(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)ctx;
  (void)b;
  if (!u || u->op != POLY_OP_CAST || u->n_src != 1 || !poly_dtype_eq(u->dtype, POLY_VOID))
    return NULL;
  return poly_dtype_eq(u->src[0]->dtype, POLY_VOID) ? u->src[0] : NULL;
}

static PolyUOp *rule_x86_isel_float_index_lane0_graph(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || u->op != POLY_OP_INDEX || u->n_src != 2 || !x86_is_float_dtype(u->dtype))
    return NULL;
  PolyUOp *base = u->src[0];
  if (!base || base->dtype.count <= 1) return NULL;
  int64_t lane = -1;
  if (!x86_const_as_i64(u->src[1], &lane) || lane != 0) return NULL;
  PolyUOp *scalar_load =
      x86_graph_scalar_load_from_reg_vector_lane(ctx, base, u->dtype, (int)lane);
  if (scalar_load) return scalar_load;
  return poly_uop1(ctx, POLY_OP_NOOP, u->dtype, base, poly_arg_none());
}

static PolyUOp *rule_x86_isel_wide_load_lane_graph(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || u->op != POLY_OP_INDEX || u->n_src != 2) return NULL;
  PolyUOp *load = u->src[0];
  if (!load || load->op != POLY_OP_LOAD || load->n_src != 1 || load->dtype.count <= 1)
    return NULL;
  if (x86_value_size(load->dtype, false) <= 16) return NULL;
  int64_t lane = -1;
  if (!x86_const_as_i64(u->src[1], &lane) || lane < 0 || lane >= load->dtype.count)
    return NULL;
  PolyUOp *shr = load->src[0];
  if (!shr || shr->op != POLY_OP_SHRINK || shr->n_src < 3) return NULL;
  PolyUOp *base = shr->src[0];
  PolyUOp *start = shr->src[1];
  if (!base || !start) return NULL;
  PolyUOp *idx = start;
  int64_t start_i = 0;
  if (x86_const_as_i64(start, &start_i)) {
    idx = poly_uop0(ctx, POLY_OP_CONST, start->dtype, poly_arg_int(start_i + lane));
  } else if (lane != 0) {
    PolyUOp *lane_uop = poly_uop0(ctx, POLY_OP_CONST, start->dtype, poly_arg_int(lane));
    idx = poly_uop2(ctx, POLY_OP_ADD, start->dtype, start, lane_uop, poly_arg_none());
  }
  PolyUOp *addr = poly_uop2(ctx, POLY_OP_INDEX, base->dtype, base, idx, poly_arg_none());
  return poly_uop1(ctx, POLY_OP_LOAD, u->dtype, addr, poly_arg_none());
}

static PolyUOp *rule_x86_isel_vector_index_graph(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || (u->op != POLY_OP_INDEX && u->op != POLY_OP_GEP)) return NULL;
  PolyUOp *base = u->n_src > 0 ? u->src[0] : NULL;
  if (!base || base->dtype.is_ptr || base->dtype.count <= 1) return NULL;
  PolyX86Op base_op = 0;
  if (!x86_ins_op(base, &base_op)) return NULL;
  int64_t lane = -1;
  if (u->op == POLY_OP_INDEX) {
    if (u->n_src < 2 || !x86_const_as_i64(u->src[1], &lane)) return NULL;
  } else if (u->arg.kind == POLY_ARG_INT) {
    lane = u->arg.i;
  } else if (u->arg.kind == POLY_ARG_INT_TUPLE && u->arg.int_tuple.n == 1) {
    lane = u->arg.int_tuple.vals[0];
  } else {
    return NULL;
  }
  if (lane < 0 || lane >= base->dtype.count) return NULL;
  PolyUOp *scalar_load =
      x86_graph_scalar_load_from_reg_vector_lane(ctx, base, u->dtype, (int)lane);
  if (scalar_load) return scalar_load;
  PolyX86Op op = x86_is_float_dtype(u->dtype) ? POLY_X86_VPSRLDQ
                : x86_dtype_itemsize(u->dtype) == 1 ? POLY_X86_VPEXTRB
                : x86_dtype_itemsize(u->dtype) == 2 ? POLY_X86_VPEXTRW
                : x86_dtype_itemsize(u->dtype) == 4 ? POLY_X86_VPEXTRD
                                                    : POLY_X86_VPEXTRQ;
  int64_t immv = x86_is_float_dtype(u->dtype) ? lane * x86_dtype_itemsize(u->dtype) : lane;
  PolyUOp *srcs[2] = {base, x86_const_i(ctx, POLY_UINT8, immv)};
  return x86_ins(ctx, op, u->dtype, srcs, 2, x86_graph_vreg(u->dtype, false));
}

static PolyUOp *rule_x86_isel_range_graph(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || u->op != POLY_OP_RANGE || u->n_src < 1 || u->n_src > 8) return NULL;
  PolyUOp *srcs[8];
  bool changed = false;
  for (int i = 0; i < u->n_src; i++) srcs[i] = u->src[i];
  int64_t bound = 0;
  if (u->src[0] && u->src[0]->op == POLY_OP_CONST && !x86_const_is_imm(u->src[0]) &&
      x86_const_as_i64(u->src[0], &bound)) {
    srcs[0] = x86_const_i(ctx, u->src[0]->dtype, bound);
    changed = true;
  }
  int32_t tag = u->tag;
  if (x86_uop_reg(u) < 0) {
    tag = x86_graph_vreg(u->dtype, false);
    changed = true;
  }
  if (!changed) return NULL;
  return poly_uop_tagged_arg(ctx, POLY_OP_RANGE, u->dtype, srcs, u->n_src, u->arg, tag, u->tag_arg);
}

static int x86_graph_fold_address(PolyCtx *ctx, PolyUOp *addr, PolyUOp *out[4]);

static bool x86_graph_stack_base_lane(PolyUOp *u, PolyUOp **base, int *lane) {
  if (!u || !base || !lane) return false;
  if (u->op == POLY_OP_NOOP && u->n_src == 1) u = u->src[0];
  *base = u;
  *lane = 0;
  PolyX86Op op = 0;
  if (x86_ins_op(u, &op) && u->n_src >= 2) {
    int64_t immv = 0;
    switch (op) {
    case POLY_X86_VPSRLDQ:
      if (!x86_const_as_i64(u->src[1], &immv)) return false;
      *base = u->src[0];
      *lane = x86_dtype_itemsize(u->dtype) > 0 ? (int)(immv / x86_dtype_itemsize(u->dtype)) : 0;
      return *base != NULL;
    case POLY_X86_VPEXTRB:
    case POLY_X86_VPEXTRW:
    case POLY_X86_VPEXTRD:
    case POLY_X86_VPEXTRQ:
      if (!x86_const_as_i64(u->src[1], &immv)) return false;
      *base = u->src[0];
      *lane = (int)immv;
      return *base != NULL;
    default:
      break;
    }
  }
  if (u->op == POLY_OP_INDEX && u->n_src >= 2) {
    int64_t li = 0;
    if (!x86_const_as_i64(u->src[1], &li)) return false;
    *base = u->src[0];
    *lane = (int)li;
    return *base != NULL;
  }
  if (u->op == POLY_OP_GEP && u->n_src >= 1) {
    if (u->arg.kind == POLY_ARG_INT)
      *lane = (int)u->arg.i;
    else if (u->arg.kind == POLY_ARG_INT_TUPLE && u->arg.int_tuple.n > 0)
      *lane = (int)u->arg.int_tuple.vals[0];
    else
      return false;
    *base = u->src[0];
    return *base != NULL;
  }
  return true;
}

static PolyUOp *rule_x86_isel_stack_graph(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || u->op != POLY_OP_STACK || u->n_src <= 1) return NULL;
  int ns = u->n_src;

  if (x86_is_float_dtype(u->dtype) && x86_dtype_is_float_bits(u->dtype, 64)) {
    if (!(ns == 2 || ns == 4)) return NULL;
    PolyUOp *base[4] = {0};
    int lane[4] = {0};
    for (int i = 0; i < ns; i++) {
      if (!x86_graph_stack_base_lane(u->src[i], &base[i], &lane[i])) return NULL;
    }
    if (lane[0] > 1 || lane[1] > 1) return NULL;
    if (ns == 4 && !(base[0] == base[2] && base[1] == base[3] && lane[2] > 1 && lane[3] > 1))
      return NULL;

    int64_t immv = 0;
    for (int i = 0; i < ns; i++)
      immv |= ((int64_t)lane[i] & 0x3) << i;
    PolyUOp *imm = x86_const_i(ctx, POLY_UINT8, immv);
    PolyUOp *srcs[3] = {base[0], base[1], imm};
    return x86_ins(ctx, POLY_X86_VSHUFPD, u->dtype, srcs, 3, x86_graph_vreg(u->dtype, false));
  }

  if (x86_is_float_dtype(u->dtype) && x86_dtype_is_float_bits(u->dtype, 32)) {
    bool same = true;
    for (int i = 1; i < ns; i++) {
      if (u->src[i] != u->src[0]) {
        same = false;
        break;
      }
    }
    if (same) {
      PolyUOp *srcs[1] = {u->src[0]};
      return x86_ins(ctx, POLY_X86_VBROADCASTSS, u->dtype, srcs, 1, x86_graph_vreg(u->dtype, false));
    }
    if (ns == 4) {
      PolyUOp *base[4] = {0};
      int lane[4] = {0};
      bool ok = true;
      for (int i = 0; i < 4; i++) {
        if (!x86_graph_stack_base_lane(u->src[i], &base[i], &lane[i]) || lane[i] < 0 || lane[i] > 3) {
          ok = false;
          break;
        }
      }
      if (ok && base[0] == base[1] && base[2] == base[3]) {
        int64_t immv = 0;
        for (int i = 0; i < 4; i++)
          immv |= ((int64_t)lane[i] & 0x3) << (2 * i);
        PolyUOp *srcs[3] = {base[0], base[2], x86_const_i(ctx, POLY_UINT8, immv)};
        return x86_ins(ctx, POLY_X86_VSHUFPS, u->dtype, srcs, 3, x86_graph_vreg(u->dtype, false));
      }
    }
    PolyUOp *cur = x86_def_scratch(ctx, u->dtype);
    bool cur_is_def = true;
    for (int i = 0; i < ns; i++) {
      PolyUOp *base = NULL;
      int lane = 0;
      if (!x86_graph_stack_base_lane(u->src[i], &base, &lane) || lane < 0 || lane > 3)
        return NULL;
      if (i == 0 && lane == 0) {
        cur = base;
        cur_is_def = false;
        continue;
      }
      PolyUOp *srcs[3] = {
          cur, base, x86_const_i(ctx, POLY_UINT8, ((int64_t)lane << 6) | ((int64_t)i << 4)),
      };
      cur = x86_ins(ctx, POLY_X86_VINSERTPS, u->dtype, srcs, 3, x86_graph_vreg(u->dtype, false));
      cur_is_def = false;
    }
    return cur_is_def ? NULL : cur;
  }

  if (x86_dtype_is_float16(u->dtype)) {
    PolyUOp *cur = x86_def_scratch(ctx, u->dtype);
    PolyUOp *cache_src[16] = {0};
    PolyUOp *cache_word[16] = {0};
    int n_cache = 0;
    for (int i = 0; i < ns; i++) {
      PolyUOp *src = u->src[i];
      if (src && src->op == POLY_OP_LOAD && src->n_src == 1) {
        PolyUOp *word = NULL;
        for (int j = 0; j < n_cache; j++) {
          if (cache_src[j] == src) {
            word = cache_word[j];
            break;
          }
        }
        if (word) {
          PolyUOp *srcs[3] = {cur, word, x86_const_i(ctx, POLY_UINT8, i)};
          cur = x86_ins(ctx, POLY_X86_VPINSRW, u->dtype, srcs, 3, x86_graph_vreg(u->dtype, false));
          continue;
        }
        PolyUOp *addr[4];
        if (x86_graph_fold_address(ctx, src->src[0], addr) != 0) return NULL;
        PolyUOp *srcs[6] = {
            cur, addr[0], addr[1], addr[2], addr[3], x86_const_i(ctx, POLY_UINT8, i),
        };
        cur = x86_ins(ctx, POLY_X86_VPINSRW, u->dtype, srcs, 6, x86_graph_vreg(u->dtype, false));
        if (n_cache < (int)(sizeof(cache_src) / sizeof(cache_src[0]))) {
          PolyUOp *ext_srcs[2] = {cur, x86_const_i(ctx, POLY_UINT8, i)};
          cache_word[n_cache] = x86_ins(
              ctx, POLY_X86_VPEXTRW, POLY_INT16, ext_srcs, 2, x86_graph_vreg(POLY_INT16, false)
          );
          cache_src[n_cache++] = src;
        }
      } else if (src && x86_ins_op(src, NULL)) {
        PolyUOp *ext_srcs[2] = {src, x86_const_i(ctx, POLY_UINT8, 0)};
        PolyUOp *word = x86_ins(
            ctx, POLY_X86_VPEXTRW, POLY_INT16, ext_srcs, 2, x86_graph_vreg(POLY_INT16, false)
        );
        PolyUOp *srcs[3] = {cur, word, x86_const_i(ctx, POLY_UINT8, i)};
        cur = x86_ins(ctx, POLY_X86_VPINSRW, u->dtype, srcs, 3, x86_graph_vreg(u->dtype, false));
      } else {
        return NULL;
      }
    }
    return cur;
  }

  if (x86_is_int_dtype(u->dtype)) {
    bool same = true;
    for (int i = 1; i < ns; i++) {
      if (u->src[i] != u->src[0]) {
        same = false;
        break;
      }
    }
    int item = x86_dtype_itemsize(poly_dtype_scalar(u->dtype));
    if (item <= 0) return NULL;
    if (same) {
      PolyX86Op bcast = item == 1 ? POLY_X86_VPBROADCASTB
                         : item == 2 ? POLY_X86_VPBROADCASTW
                         : item == 4 ? POLY_X86_VPBROADCASTD
                                     : POLY_X86_VPBROADCASTQ;
      PolyUOp *srcs[1] = {u->src[0]};
      return x86_ins(ctx, bcast, u->dtype, srcs, 1, x86_graph_vreg(u->dtype, false));
    }
    PolyX86Op op = item == 1 ? POLY_X86_VPINSRB
                    : item == 2 ? POLY_X86_VPINSRW
                    : item == 4 ? POLY_X86_VPINSRD
                                : POLY_X86_VPINSRQ;
    PolyUOp *cur = x86_def_scratch(ctx, u->dtype);
    for (int i = 0; i < ns; i++) {
      PolyUOp *srcs[3] = {cur, u->src[i], x86_const_i(ctx, POLY_UINT8, i)};
      cur = x86_ins(ctx, op, u->dtype, srcs, 3, x86_graph_vreg(u->dtype, false));
    }
    return cur;
  }

  return NULL;
}

static PolyUOp *x86_graph_scalar_low_lane(PolyUOp *u) {
  if (!u) return NULL;
  if (u->op == POLY_OP_NOOP && u->n_src == 1) return u;
  if (u->op == POLY_OP_STACK && u->n_src > 0) return u->src[0];
  return u;
}

static PolyUOp *x86_graph_address_index_src(PolyCtx *ctx, PolyUOp *idx) {
  if (!idx) return NULL;
  PolyDType s = poly_dtype_scalar(idx->dtype);
  if (!poly_dtype_is_int(s) || poly_dtype_is_unsigned(s) || s.bitsize >= 64)
    return idx;
  int64_t vmin = 0, vmax = 0;
  poly_uop_minmax(ctx, idx, &vmin, &vmax);
  return vmin < 0 ? poly_uop1(ctx, POLY_OP_CAST, POLY_INT64, idx, poly_arg_none()) : idx;
}

static int x86_graph_fold_address(PolyCtx *ctx, PolyUOp *addr, PolyUOp *out[4]) {
  if (!ctx || !addr || !out) return -1;
  out[0] = addr;
  out[1] = x86_noop(ctx);
  out[2] = x86_disp_const(ctx, 0);
  out[3] = x86_const_i(ctx, POLY_UINT8, x86_dtype_itemsize(addr->dtype));
  if ((addr->op != POLY_OP_INDEX && addr->op != POLY_OP_SHRINK) || addr->n_src < 2)
    return 0;

  PolyUOp *base = addr->src[0];
  PolyUOp *idx = addr->src[1];
  int scale = (base->op == POLY_OP_PARAM || base->op == POLY_OP_BUFFER ||
               base->op == POLY_OP_AFTER)
                  ? x86_dtype_itemsize(base->dtype)
                  : 1;
  int size = x86_dtype_itemsize(base->dtype);
  if (size <= 0) size = x86_dtype_itemsize(addr->dtype);
  if (size <= 0) size = 8;

  int64_t c = 0;
  out[0] = base;
  out[3] = x86_const_i(ctx, POLY_UINT8, size);
  if (x86_const_as_i64(idx, &c)) {
    out[1] = x86_noop(ctx);
    out[2] = x86_disp_const(ctx, c * scale);
    return 0;
  }
  if (idx->op == POLY_OP_ADD && idx->n_src == 2 && x86_const_as_i64(idx->src[1], &c)) {
    out[1] = x86_graph_address_index_src(ctx, idx->src[0]);
    out[2] = x86_disp_const(ctx, c * scale);
    return out[1] ? 0 : -1;
  }
  out[1] = x86_graph_address_index_src(ctx, idx);
  out[2] = x86_disp_const(ctx, 0);
  return out[1] ? 0 : -1;
}

static bool x86_graph_address_is_reg_backed(PolyUOp *addr[4]) {
  return addr && addr[0] && addr[0]->dtype.is_ptr &&
         addr[0]->dtype.addrspace == POLY_ADDR_REG;
}

static bool x86_graph_load_address_tuple(PolyCtx *ctx, PolyUOp *load, PolyUOp *addr[4]) {
  if (!ctx || !load || !addr) return false;
  if (load->op == POLY_OP_LOAD && load->n_src == 1)
    return x86_graph_fold_address(ctx, load->src[0], addr) == 0;
  if (x86_ins_is_load_addr(load)) {
    for (int i = 0; i < 4; i++) addr[i] = load->src[i];
    return true;
  }
  return false;
}

static PolyUOp *x86_graph_scalar_load_from_reg_vector_lane(
    PolyCtx *ctx,
    PolyUOp *load,
    PolyDType lane_dtype,
    int lane
) {
  if (!ctx || !load || lane < 0 || !x86_is_float_dtype(lane_dtype))
    return NULL;
  PolyUOp *addr[4];
  if (!x86_graph_load_address_tuple(ctx, load, addr) ||
      !x86_graph_address_is_reg_backed(addr))
    return NULL;
  int item = x86_dtype_itemsize(lane_dtype);
  if (item <= 0) return NULL;
  int64_t disp = 0;
  if (!x86_const_as_i64(addr[2], &disp)) return NULL;
  addr[2] = x86_disp_const(ctx, disp + (int64_t)lane * item);
  addr[3] = x86_const_i(ctx, POLY_UINT8, item);
  return x86_ins(
      ctx, x86_mov_op_for_dtype(lane_dtype, false), lane_dtype, addr, 4,
      x86_graph_vreg(lane_dtype, false)
  );
}

static PolyX86Op x86_vector_int_cmp_op(PolyOps op, PolyDType dt, bool *swap);
static PolyUOp *x86_alias_value_as_dtype(PolyCtx *ctx, PolyUOp *src, PolyDType dtype);
static PolyUOp *x86_graph_flag_compare(PolyCtx *ctx, PolyUOp *mask);

static PolyUOp *rule_x86_isel_fma_graph(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || u->op != POLY_OP_ADD || u->n_src != 2 || !x86_dtype_can_fma(u->dtype))
    return NULL;
  X86IselUseCtx *uctx = (X86IselUseCtx *)poly_graph_rewrite_userctx();
  for (int side = 0; side < 2; side++) {
    PolyUOp *mul = u->src[side];
    if (!mul || mul->op != POLY_OP_MUL || mul->n_src != 2) continue;
    if (!poly_dtype_eq(mul->dtype, u->dtype)) continue;
    if (!x86_isel_foldable(uctx, u, mul)) continue;
    PolyUOp *addend = u->src[side ^ 1];
    return poly_uop3(ctx, POLY_OP_MULACC, u->dtype, mul->src[0], mul->src[1], addend, poly_arg_none());
  }
  return NULL;
}

static bool x86_graph_load_dtype_supported(PolyDType dt) {
  if (x86_dtype_is_float16(dt)) {
    int bytes = x86_value_size(dt, false);
    return dt.count <= 1 || bytes == 4 || bytes == 8 || bytes == 16;
  }
  if (x86_dtype_is_float_bits(dt, 64) && dt.count > 1 && x86_value_size(dt, false) != 16)
    return false;
  return true;
}

static PolyUOp *rule_x86_isel_load_graph(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || u->op != POLY_OP_LOAD || u->n_src != 1 || !x86_graph_load_dtype_supported(u->dtype))
    return NULL;
  PolyUOp *addr[4];
  if (x86_graph_fold_address(ctx, u->src[0], addr) != 0) return NULL;
  int32_t def = x86_graph_vreg(u->dtype, false);
  if (x86_dtype_is_float16(u->dtype) && u->dtype.count <= 1) {
    PolyUOp *srcs[6] = {
        x86_define_reg(ctx, u->dtype, def), addr[0], addr[1], addr[2], addr[3],
        x86_const_i(ctx, POLY_UINT8, 0),
    };
    return x86_ins(ctx, POLY_X86_VPINSRW, u->dtype, srcs, 6, def);
  }
  return x86_ins(ctx, x86_mov_op_for_dtype(u->dtype, false), u->dtype, addr, 4, def);
}

static PolyUOp *rule_x86_isel_float_bin_graph(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || u->n_src != 2 || !x86_is_float_dtype(u->dtype) || x86_dtype_is_float16(u->dtype))
    return NULL;
  PolyX86Op op = x86_float_bin_op(u->op, u->dtype);
  if (!op) return NULL;
  PolyUOp *srcs[2] = {u->src[0], u->src[1]};
  if (u->dtype.count <= 1) {
    srcs[0] = x86_graph_scalar_low_lane(srcs[0]);
    srcs[1] = x86_graph_scalar_low_lane(srcs[1]);
    if (!srcs[0] || !srcs[1]) return NULL;
  }
  return x86_ins(ctx, op, u->dtype, srcs, 2, x86_graph_vreg(u->dtype, false));
}

static PolyUOp *rule_x86_isel_float_unary_graph(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || u->n_src != 1 || !x86_is_float_dtype(u->dtype) || x86_dtype_is_float16(u->dtype))
    return NULL;
  PolyX86Op op = 0;
  PolyUOp *srcs[3];
  int ns = 0;
  srcs[ns++] = u->src[0];
  if (u->op == POLY_OP_SQRT) {
    op = x86_dtype_is_float_bits(u->dtype, 64)
             ? (u->dtype.count > 1 ? POLY_X86_VSQRTPD : POLY_X86_VSQRTSD)
             : (u->dtype.count > 1 ? POLY_X86_VSQRTPS : POLY_X86_VSQRTSS);
    if (u->dtype.count == 1) srcs[ns++] = u->src[0];
  } else if (u->op == POLY_OP_TRUNC) {
    op = x86_dtype_is_float_bits(u->dtype, 64)
             ? (u->dtype.count > 1 ? POLY_X86_VROUNDPD : POLY_X86_VROUNDSD)
             : (u->dtype.count > 1 ? POLY_X86_VROUNDPS : POLY_X86_VROUNDSS);
    if (u->dtype.count == 1) srcs[ns++] = u->src[0];
    srcs[ns++] = x86_const_i(ctx, POLY_UINT8, 3);
  }
  if (!op) return NULL;
  return x86_ins(ctx, op, u->dtype, srcs, ns, x86_graph_vreg(u->dtype, false));
}

static PolyUOp *rule_x86_isel_compare_graph(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || !x86_op_is_comparison(u->op) || u->n_src != 2) return NULL;
  X86IselUseCtx *uctx = (X86IselUseCtx *)poly_graph_rewrite_userctx();
  PolyUOp *single_consumer = x86_isel_single_consumer(uctx, u);
  if (poly_dtype_eq(u->dtype, POLY_BOOL) && single_consumer &&
      ((single_consumer->op == POLY_OP_WHERE && single_consumer->n_src > 0 && single_consumer->src[0] == u) ||
       (single_consumer->op == POLY_OP_IF && single_consumer->n_src > 0 && single_consumer->src[0] == u)))
    return NULL;
  PolyUOp *lhs = u->src[0];
  PolyUOp *rhs = u->src[1];
  if (u->dtype.count <= 1) {
    lhs = x86_graph_scalar_low_lane(lhs);
    rhs = x86_graph_scalar_low_lane(rhs);
    if (!lhs || !rhs) return NULL;
  }
  PolyDType src_dt = lhs->dtype;
  if (poly_dtype_eq(u->dtype, POLY_BOOL) && x86_is_float_dtype(src_dt) &&
      !x86_dtype_is_float16(src_dt)) {
    PolyUOp *imm = x86_const_i(
        ctx, POLY_UINT8,
        u->op == POLY_OP_CMPLT ? 1 : u->op == POLY_OP_CMPNE ? 4 : 0
    );
    PolyUOp *cmp_srcs[3] = {lhs, rhs, imm};
    int32_t mask_def = x86_graph_vreg(src_dt, false);
    PolyX86Op cmp_op = x86_dtype_is_float_bits(src_dt, 64) ? POLY_X86_VCMPSD : POLY_X86_VCMPSS;
    PolyUOp *mask = x86_ins(ctx, cmp_op, src_dt, cmp_srcs, 3, mask_def);
    PolyDType idt = x86_int_for_float(src_dt);
    int32_t mov_def = x86_graph_vreg(idt, false);
    PolyUOp *mov_srcs[1] = {mask};
    PolyX86Op mov_op = x86_dtype_itemsize(idt) == 8 ? POLY_X86_VMOVQm : POLY_X86_VMOVDm;
    PolyUOp *bits = x86_ins(ctx, mov_op, idt, mov_srcs, 1, mov_def);
    PolyUOp *one = x86_const_i(ctx, idt, 1);
    PolyUOp *one_imm = x86_imm_for_const(ctx, one);
    if (!one_imm) return NULL;
    PolyUOp *and_srcs[2] = {bits, one_imm};
    PolyUOp *one_bit = x86_ins(ctx, POLY_X86_ANDi, idt, and_srcs, 2, x86_graph_vreg(idt, false));
    return poly_uop1(ctx, POLY_OP_NOOP, POLY_BOOL, one_bit, poly_arg_none());
  }
  if (poly_dtype_eq(u->dtype, POLY_BOOL) && x86_is_int_dtype(src_dt)) {
    PolyUOp *flag = x86_graph_flag_compare(ctx, u);
    if (!flag) return NULL;
    PolyX86Op setop = u->op == POLY_OP_CMPLT
                          ? (poly_dtype_is_unsigned(poly_dtype_scalar(src_dt)) ? POLY_X86_SETB
                                                                                : POLY_X86_SETL)
                      : u->op == POLY_OP_CMPEQ ? POLY_X86_SETE
                                               : POLY_X86_SETNE;
    PolyUOp *srcs[1] = {flag};
    return x86_ins(ctx, setop, u->dtype, srcs, 1, x86_graph_vreg(u->dtype, false));
  }
  if (x86_is_float_dtype(src_dt) && !x86_dtype_is_float16(src_dt) &&
      (!poly_dtype_eq(u->dtype, POLY_BOOL) || src_dt.count > 1)) {
    PolyUOp *imm = x86_const_i(
        ctx, POLY_UINT8,
        u->op == POLY_OP_CMPLT ? 1 : u->op == POLY_OP_CMPNE ? 4 : 0
    );
    PolyUOp *srcs[3] = {lhs, rhs, imm};
    PolyX86Op op = x86_dtype_is_float_bits(src_dt, 64)
                       ? (src_dt.count > 1 ? POLY_X86_VCMPPD : POLY_X86_VCMPSD)
                       : (src_dt.count > 1 ? POLY_X86_VCMPPS : POLY_X86_VCMPSS);
    return x86_ins(ctx, op, u->dtype, srcs, 3, x86_graph_vreg(u->dtype, false));
  }
  bool swap = false;
  PolyX86Op op = x86_vector_int_cmp_op(u->op, src_dt, &swap);
  if (!op) return NULL;
  PolyUOp *srcs[2] = {swap ? rhs : lhs, swap ? lhs : rhs};
  return x86_ins(ctx, op, u->dtype, srcs, 2, x86_graph_vreg(u->dtype, false));
}

static PolyUOp *x86_graph_cmp_mask_as_value(PolyCtx *ctx, PolyUOp *mask) {
  if (!ctx || !mask || !x86_op_is_comparison(mask->op) || mask->n_src != 2) return NULL;
  PolyUOp *srcs[2] = {mask->src[0], mask->src[1]};
  if (mask->dtype.count <= 1 || srcs[0]->dtype.count <= 1 || srcs[1]->dtype.count <= 1) {
    srcs[0] = x86_graph_scalar_low_lane(srcs[0]);
    srcs[1] = x86_graph_scalar_low_lane(srcs[1]);
    if (!srcs[0] || !srcs[1]) return NULL;
  }
  return poly_uop(ctx, mask->op, srcs[0]->dtype, srcs, 2, mask->arg);
}

static PolyUOp *x86_graph_materialize_scalar_int_const(PolyCtx *ctx, PolyUOp *u) {
  if (!u || u->op != POLY_OP_CONST || u->dtype.count > 1 || !x86_is_int_dtype(u->dtype))
    return u;
  if (u->arg.kind != POLY_ARG_INT && u->arg.kind != POLY_ARG_BOOL) return u;
  int64_t v = u->arg.kind == POLY_ARG_BOOL ? (u->arg.b ? 1 : 0) : u->arg.i;
  PolyUOp *imm = x86_const_i(ctx, u->dtype, v);
  PolyUOp *srcs[1] = {imm};
  PolyX86Op op = x86_value_size(u->dtype, false) == 8 ? POLY_X86_MOVABS : POLY_X86_MOVi;
  return x86_ins(ctx, op, u->dtype, srcs, 1, x86_graph_vreg(u->dtype, false));
}

static PolyUOp *x86_graph_flag_compare(PolyCtx *ctx, PolyUOp *mask) {
  if (!ctx || !mask || !x86_op_is_comparison(mask->op) || mask->n_src != 2) return NULL;
  PolyDType lhs_dt = mask->src[0]->dtype;
  PolyX86Op cmp_op = 0;
  PolyUOp *srcs[2] = {mask->src[0], mask->src[1]};
  if (x86_is_float_dtype(lhs_dt)) {
    cmp_op = x86_dtype_is_float_bits(lhs_dt, 64) ? POLY_X86_VUCOMISD : POLY_X86_VUCOMISS;
  } else {
    PolyUOp *imm = mask->src[1] && mask->src[1]->op == POLY_OP_CONST ? x86_imm_for_const(ctx, mask->src[1]) : NULL;
    if (imm) {
      srcs[1] = imm;
      cmp_op = POLY_X86_CMPi;
    } else {
      cmp_op = POLY_X86_CMP;
    }
  }
  return x86_ins_nodef(ctx, cmp_op, POLY_VOID, srcs, 2);
}

static PolyUOp *rule_x86_isel_where_graph(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || u->op != POLY_OP_WHERE || u->n_src != 3) return NULL;
  PolyUOp *mask = u->src[0];
  if (!mask || !x86_op_is_comparison(mask->op) || mask->n_src != 2) return NULL;

  if (x86_is_float_dtype(u->dtype) && mask->op == POLY_OP_CMPLT &&
      x86_is_float_dtype(mask->src[0]->dtype)) {
    PolyUOp *srcs[2] = {mask->src[0], mask->src[1]};
    if (u->dtype.count <= 1) {
      srcs[0] = x86_graph_scalar_low_lane(srcs[0]);
      srcs[1] = x86_graph_scalar_low_lane(srcs[1]);
      if (!srcs[0] || !srcs[1]) return NULL;
    }
    if (u->src[1] == mask->src[1] && u->src[2] == mask->src[0]) {
      PolyX86Op op = x86_dtype_is_float_bits(u->dtype, 64)
                         ? (u->dtype.count > 1 ? POLY_X86_VMAXPD : POLY_X86_VMAXSD)
                         : (u->dtype.count > 1 ? POLY_X86_VMAXPS : POLY_X86_VMAXSS);
      return x86_ins(ctx, op, u->dtype, srcs, 2, x86_graph_vreg(u->dtype, false));
    }
    if (u->src[1] == mask->src[0] && u->src[2] == mask->src[1]) {
      PolyX86Op op = x86_dtype_is_float_bits(u->dtype, 64)
                         ? (u->dtype.count > 1 ? POLY_X86_VMINPD : POLY_X86_VMINSD)
                         : (u->dtype.count > 1 ? POLY_X86_VMINPS : POLY_X86_VMINSS);
      return x86_ins(ctx, op, u->dtype, srcs, 2, x86_graph_vreg(u->dtype, false));
    }
  }

  if (x86_is_float_dtype(u->dtype) && x86_is_float_dtype(mask->src[0]->dtype)) {
    PolyUOp *mask_value = x86_graph_cmp_mask_as_value(ctx, mask);
    if (!mask_value) return NULL;
    PolyX86Op op = x86_dtype_is_float_bits(u->dtype, 64) ? POLY_X86_VBLENDVPD
                                                         : POLY_X86_VBLENDVPS;
    PolyUOp *falsev = u->src[2];
    PolyUOp *truev = u->src[1];
    if (u->dtype.count <= 1) {
      falsev = x86_graph_scalar_low_lane(falsev);
      truev = x86_graph_scalar_low_lane(truev);
      if (!falsev || !truev) return NULL;
    }
    PolyUOp *srcs[3] = {falsev, truev, mask_value};
    return x86_ins(ctx, op, u->dtype, srcs, 3, x86_graph_vreg(u->dtype, false));
  }

  if (u->dtype.count > 1 && x86_is_int_dtype(u->dtype) && x86_is_int_dtype(mask->src[0]->dtype)) {
    PolyUOp *mask_value = x86_graph_cmp_mask_as_value(ctx, mask);
    if (!mask_value) return NULL;
    PolyUOp *srcs[3] = {u->src[2], u->src[1], mask_value};
    return x86_ins(ctx, POLY_X86_VPBLENDVB, u->dtype, srcs, 3, x86_graph_vreg(u->dtype, false));
  }

  if (u->dtype.count <= 1) {
    PolyUOp *flag = x86_graph_flag_compare(ctx, mask);
    if (!flag) return NULL;
    PolyX86Op op = mask->op == POLY_OP_CMPLT
                       ? (poly_dtype_is_unsigned(poly_dtype_scalar(mask->src[0]->dtype)) ? POLY_X86_CMOVB
                                                                                         : POLY_X86_CMOVL)
                   : mask->op == POLY_OP_CMPEQ ? POLY_X86_CMOVE
                                                : POLY_X86_CMOVNE;
    PolyUOp *srcs[3] = {
        x86_graph_materialize_scalar_int_const(ctx, u->src[2]),
        x86_graph_materialize_scalar_int_const(ctx, u->src[1]),
        flag,
    };
    return x86_ins(ctx, op, u->dtype, srcs, 3, x86_graph_vreg(u->dtype, false));
  }
  return NULL;
}

static PolyUOp *rule_x86_isel_if_graph(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || u->op != POLY_OP_IF || u->n_src < 1) return NULL;
  PolyUOp *mask = u->src[0];
  if (!mask || !x86_op_is_comparison(mask->op) || mask->n_src != 2) return NULL;
  if (u->tag_arg.kind != POLY_ARG_STRING || !u->tag_arg.str) return NULL;
  PolyUOp *flag = x86_graph_flag_compare(ctx, mask);
  if (!flag) return NULL;
  PolyX86Op op = mask->op == POLY_OP_CMPLT
                     ? (poly_dtype_is_unsigned(poly_dtype_scalar(mask->src[0]->dtype)) ? POLY_X86_JB
                                                                                       : POLY_X86_JL)
                 : mask->op == POLY_OP_CMPEQ ? POLY_X86_JE
                                              : POLY_X86_JNE;
  PolyUOp *srcs[1] = {flag};
  return x86_ins_jump(ctx, op, srcs, 1, u->tag_arg.str);
}

static PolyUOp *rule_x86_isel_scalar_int_bin_graph(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || u->n_src != 2 || !x86_is_int_dtype(u->dtype))
    return NULL;
  if (u->dtype.count > 1) {
    PolyX86Op op = x86_int_bin_op(u->op, u->dtype, false);
    if (!op) return NULL;
    PolyUOp *srcs[2] = {u->src[0], u->src[1]};
    return x86_ins(ctx, op, u->dtype, srcs, 2, x86_graph_vreg(u->dtype, false));
  }
  bool imm = false;
  PolyUOp *rhs = u->src[1];
  if (rhs && rhs->op == POLY_OP_CONST) {
    rhs = x86_imm_for_const(ctx, rhs);
    if (!rhs) return NULL;
    if (u->op == POLY_OP_SHL || u->op == POLY_OP_SHR)
      rhs = x86_const_i(ctx, POLY_UINT8, rhs->arg.i);
    imm = true;
  }
  PolyX86Op op = x86_int_bin_op(u->op, u->dtype, imm);
  if (!op) return NULL;
  PolyUOp *srcs[2] = {u->src[0], rhs};
  return x86_ins(ctx, op, u->dtype, srcs, 2, x86_graph_vreg(u->dtype, false));
}

static PolyUOp *rule_x86_isel_cdiv_graph(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || u->op != POLY_OP_CDIV || u->n_src != 2 || u->dtype.count > 1 ||
      !x86_is_int_dtype(u->dtype))
    return NULL;

  PolyDType sdt = poly_dtype_scalar(u->dtype);
  int bytes = x86_dtype_itemsize(u->dtype);
  if (bytes <= 0) return NULL;
  bool is_unsigned = poly_dtype_is_unsigned(sdt);
  int32_t dividend_rax = x86_graph_fixed_wgpr(X86_REG_RAX);
  int32_t quotient_rax = x86_graph_fixed_wgpr(X86_REG_RAX);
  int32_t remainder_rdx = bytes == 1 ? 0 : x86_graph_fixed_wgpr(X86_REG_RDX);
  if (!dividend_rax || !quotient_rax || (bytes > 1 && !remainder_rdx)) return NULL;

  PolyUOp *lhs = x86_graph_materialize_scalar_int_const(ctx, u->src[0]);
  PolyUOp *rhs = x86_graph_materialize_scalar_int_const(ctx, u->src[1]);
  PolyUOp *divisor_srcs[1] = {rhs};
  PolyUOp *divisor = x86_ins(
      ctx, POLY_X86_MOV, u->src[1]->dtype, divisor_srcs, 1,
      x86_graph_vreg(u->src[1]->dtype, false)
  );

  PolyUOp *dividend = NULL;
  if (bytes == 1) {
    PolyDType ext_dt = is_unsigned ? POLY_UINT16 : POLY_INT16;
    PolyX86Op movx = is_unsigned ? POLY_X86_MOVZX : POLY_X86_MOVSX;
    PolyUOp *srcs[1] = {lhs};
    dividend = x86_ins(ctx, movx, ext_dt, srcs, 1, dividend_rax);
  } else {
    PolyUOp *srcs[1] = {lhs};
    dividend = x86_ins(ctx, POLY_X86_MOV, u->dtype, srcs, 1, dividend_rax);
  }

  PolyUOp *ext = NULL;
  if (bytes > 1) {
    int32_t ext_rdx = x86_graph_fixed_wgpr(X86_REG_RDX);
    if (!ext_rdx) return NULL;
    if (is_unsigned) {
      PolyUOp *zero = x86_const_i(ctx, POLY_UINT32, 0);
      PolyUOp *srcs[1] = {zero};
      ext = x86_ins(ctx, POLY_X86_MOVi, POLY_UINT32, srcs, 1, ext_rdx);
    } else {
      int32_t copy_rdx = x86_graph_fixed_wgpr(X86_REG_RDX);
      if (!copy_rdx) return NULL;
      PolyUOp *copy_srcs[1] = {lhs};
      PolyUOp *rdx_copy = x86_ins(ctx, POLY_X86_MOV, u->dtype, copy_srcs, 1, copy_rdx);
      PolyUOp *sh = x86_const_i(ctx, POLY_UINT8, bytes * 8 - 1);
      PolyUOp *srcs[2] = {rdx_copy, sh};
      ext = x86_ins(ctx, POLY_X86_SARi, u->dtype, srcs, 2, ext_rdx);
    }
  }

  PolyUOp *idiv_srcs[3];
  int ns = 0;
  idiv_srcs[ns++] = dividend;
  idiv_srcs[ns++] = divisor;
  if (ext) idiv_srcs[ns++] = ext;
  int32_t defs[2] = {quotient_rax, remainder_rdx};
  PolyX86Op op = is_unsigned ? POLY_X86_DIV : POLY_X86_IDIV;
  PolyUOp *idiv = x86_ins_ex(ctx, op, u->dtype, idiv_srcs, ns, defs, bytes == 1 ? 1 : 2, poly_arg_none());

  PolyUOp *mov_srcs[1] = {idiv};
  return x86_ins(ctx, POLY_X86_MOV, u->dtype, mov_srcs, 1, x86_graph_vreg(u->dtype, false));
}

static PolyUOp *rule_x86_isel_cast_graph(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || u->op != POLY_OP_CAST || u->n_src != 1) return NULL;
  PolyUOp *src = u->src[0];
  if (!src || poly_dtype_eq(u->dtype, src->dtype)) return NULL;
  PolyX86Op op = 0;
  PolyUOp *srcs[3];
  int ns = 1;
  srcs[0] = src;

  if (x86_is_int_dtype(src->dtype) && x86_is_float_dtype(u->dtype)) {
    if (src->dtype.count > 1) {
      if (poly_dtype_eq(poly_dtype_scalar(src->dtype), POLY_INT32) &&
          x86_dtype_is_float32(u->dtype))
        op = POLY_X86_VCVTDQ2PS;
      else if (poly_dtype_eq(poly_dtype_scalar(src->dtype), POLY_INT32) &&
               x86_dtype_is_float_bits(u->dtype, 64))
        op = POLY_X86_VCVTDQ2PD;
      else
        return NULL;
    } else {
      op = x86_dtype_is_float_bits(u->dtype, 64) ? POLY_X86_VCVTSI2SD : POLY_X86_VCVTSI2SS;
      srcs[0] = x86_def_scratch(ctx, u->dtype);
      srcs[1] = x86_graph_materialize_scalar_int_const(ctx, src);
      ns = 2;
    }
  } else if (x86_is_float_dtype(src->dtype) && x86_is_int_dtype(u->dtype)) {
    if (src->dtype.count > 1) {
      if (poly_dtype_eq(poly_dtype_scalar(u->dtype), POLY_INT32))
        op = x86_dtype_is_float_bits(src->dtype, 64) ? POLY_X86_VCVTTPD2DQ : POLY_X86_VCVTTPS2DQ;
      else
        return NULL;
    } else {
      op = x86_dtype_is_float_bits(src->dtype, 64) ? POLY_X86_VCVTTSD2SI : POLY_X86_VCVTTSS2SI;
    }
  } else if (x86_is_float_dtype(src->dtype) && x86_is_float_dtype(u->dtype)) {
    if (x86_dtype_is_float16(src->dtype) && x86_dtype_is_float32(u->dtype)) {
      op = POLY_X86_VCVTPH2PS;
    } else if (x86_dtype_is_float32(src->dtype) && x86_dtype_is_float16(u->dtype)) {
      op = POLY_X86_VCVTPS2PH;
      srcs[1] = x86_const_i(ctx, POLY_UINT8, 4);
      ns = 2;
    } else if (x86_dtype_is_float32(src->dtype) && x86_dtype_is_float64(u->dtype)) {
      if (src->dtype.count > 1) {
        op = POLY_X86_VCVTPS2PD;
      } else {
        op = POLY_X86_VCVTSS2SD;
        srcs[1] = src;
        ns = 2;
      }
    } else if (x86_dtype_is_float64(src->dtype) && x86_dtype_is_float32(u->dtype)) {
      if (src->dtype.count > 1) {
        op = POLY_X86_VCVTPD2PS;
      } else {
        op = POLY_X86_VCVTSD2SS;
        srcs[1] = src;
        ns = 2;
      }
    } else {
      return NULL;
    }
  } else if (x86_is_int_dtype(src->dtype) && x86_is_int_dtype(u->dtype)) {
    if ((op = x86_vector_int_extend_op(src->dtype, u->dtype)) != 0) {
      /* vector extend */
    } else if (src->dtype.count <= 1 &&
               poly_dtype_eq(poly_dtype_scalar(src->dtype), POLY_INT32) &&
               x86_dtype_itemsize(u->dtype) == 8) {
      op = POLY_X86_MOVSXD;
    } else if (src->dtype.count <= 1 &&
               x86_dtype_itemsize(u->dtype) > x86_dtype_itemsize(src->dtype)) {
      op = poly_dtype_is_unsigned(poly_dtype_scalar(src->dtype)) ||
           poly_dtype_is_bool(poly_dtype_scalar(src->dtype)) ? POLY_X86_MOVZX
                                                             : POLY_X86_MOVSX;
    } else {
      return NULL;
    }
  }
  if (!op) return NULL;
  return x86_ins(ctx, op, u->dtype, srcs, ns, x86_graph_vreg(u->dtype, false));
}

static PolyUOp *rule_x86_isel_bitcast_graph(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || u->op != POLY_OP_BITCAST || u->n_src != 1) return NULL;
  PolyUOp *src = u->src[0];
  if (!src || poly_dtype_eq(u->dtype, src->dtype)) return NULL;
  PolyX86Op op = 0;
  PolyUOp *srcs[2] = {src, NULL};
  int ns = 1;
  if (x86_dtype_is_float16(src->dtype) && x86_is_int_dtype(u->dtype)) {
    op = POLY_X86_VPEXTRW;
    srcs[1] = x86_const_i(ctx, POLY_UINT8, 0);
    ns = 2;
  } else if (x86_is_int_dtype(src->dtype) && x86_is_float_dtype(u->dtype)) {
    op = x86_dtype_itemsize(u->dtype) == 8 ? POLY_X86_VMOVQ : POLY_X86_VMOVD;
    srcs[0] = x86_graph_materialize_scalar_int_const(ctx, src);
  } else if (x86_is_float_dtype(src->dtype) && x86_is_int_dtype(u->dtype)) {
    op = x86_dtype_itemsize(src->dtype) == 8 ? POLY_X86_VMOVQm : POLY_X86_VMOVDm;
  } else {
    return NULL;
  }
  return x86_ins(ctx, op, u->dtype, srcs, ns, x86_graph_vreg(u->dtype, false));
}

static PolyUOp *rule_x86_isel_mulacc_ins_graph(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || u->op != POLY_OP_MULACC || u->n_src != 3 || !x86_dtype_can_fma(u->dtype))
    return NULL;
  PolyX86Op op = x86_dtype_is_float_bits(u->dtype, 64)
                     ? (u->dtype.count > 1 ? POLY_X86_VFMADD213PD : POLY_X86_VFMADD213SD)
                     : (u->dtype.count > 1 ? POLY_X86_VFMADD213PS : POLY_X86_VFMADD213SS);
  PolyUOp *srcs[3] = {u->src[0], u->src[1], u->src[2]};
  return x86_ins(ctx, op, u->dtype, srcs, 3, x86_graph_vreg(u->dtype, false));
}

static PolyUOp *rule_x86_isel_readmem_graph(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || u->op != POLY_OP_INS) return NULL;
  PolyX86Op op = 0;
  if (!x86_ins_op(u, &op)) return NULL;
  X86IselUseCtx *uctx = (X86IselUseCtx *)poly_graph_rewrite_userctx();

  int candidates[3];
  int n_candidates = 0;
  if (x86_readmem3rd(op) && u->n_src >= 3) candidates[n_candidates++] = 2;
  if (x86_readmem2nd(op) && u->n_src >= 2) candidates[n_candidates++] = 1;
  if (x86_readmem1st(op) && u->n_src >= 1) candidates[n_candidates++] = 0;
  for (int ci = 0; ci < n_candidates; ci++) {
    int cand = candidates[ci];
    if (x86_ins_has_memory_tuple_at(u, cand)) continue;
    if (op == POLY_X86_VINSERTPS && cand == 1 && u->n_src >= 3) {
      int64_t immv = 0;
      if (x86_const_as_i64(u->src[2], &immv) && ((immv >> 6) & 0x3) != 0)
        continue;
    }
    PolyUOp *load = u->src[cand];
    if (!load) continue;
    if (!x86_isel_foldable(uctx, u, load)) continue;
    PolyUOp *addr[4];
    if (!x86_graph_load_address_tuple(ctx, load, addr)) return NULL;

    int ns = u->n_src - 1 + 4;
    PolyUOp **srcs = malloc((size_t)ns * sizeof(*srcs));
    if (!srcs) return NULL;
    int pos = 0;
    for (int s = 0; s < u->n_src; s++) {
      if (s == cand) {
        for (int a = 0; a < 4; a++) srcs[pos++] = addr[a];
      } else {
        srcs[pos++] = u->src[s];
      }
    }
    int32_t defs[8];
    int n_defs = x86_uop_def_regs(u, defs, (int)(sizeof(defs) / sizeof(defs[0])));
    PolyUOp *out = x86_clone_with_regs(ctx, u, srcs, ns, defs, n_defs);
    free(srcs);
    return out;
  }
  return NULL;
}

static PolyUOp *rule_x86_isel_store_graph(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || u->op != POLY_OP_STORE || u->n_src < 2) return NULL;
  PolyUOp *value = u->src[1];
  if (!value) return NULL;
  if (value->op == POLY_OP_STACK && x86_dtype_is_float_bits(value->dtype, 64))
    return NULL;
  PolyUOp *srcs[6];
  if (x86_graph_fold_address(ctx, u->src[0], srcs) != 0) return NULL;
  int ns = 5;
  srcs[4] = value;
  PolyX86Op op = 0;
  if (value->op == POLY_OP_CONST && x86_is_int_dtype(value->dtype) && value->dtype.count <= 1) {
    op = POLY_X86_MOVi;
    srcs[4] = x86_imm_for_const(ctx, value);
    if (!srcs[4]) return NULL;
  } else if (x86_dtype_is_float16(value->dtype) && value->dtype.count == 1) {
    op = POLY_X86_VPEXTRW;
    srcs[5] = x86_const_i(ctx, POLY_UINT8, 0);
    ns = 6;
  } else {
    op = x86_mov_op_for_dtype(value->dtype, true);
  }
  return x86_ins_nodef(ctx, op, POLY_VOID, srcs, ns);
}

static PolyUOp *rule_x86_isel_wide_stack_store_graph(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || u->op != POLY_OP_STORE || u->n_src < 2) return NULL;
  PolyUOp *addr = u->src[0];
  PolyUOp *value = u->src[1];
  if (!addr || addr->op != POLY_OP_SHRINK || addr->n_src < 3 ||
      !value || value->op != POLY_OP_STACK || value->n_src <= 1)
    return NULL;
  if (!x86_dtype_is_float_bits(value->dtype, 64) || x86_value_size(value->dtype, false) <= 16)
    return NULL;
  PolyUOp *base = addr->src[0];
  PolyUOp *start = addr->src[1];
  if (!base || !start) return NULL;
  PolyUOp **stores = malloc(sizeof(PolyUOp *) * (size_t)value->n_src);
  if (!stores) return NULL;
  int64_t start_i = 0;
  bool start_const = x86_const_as_i64(start, &start_i);
  for (int lane = 0; lane < value->n_src; lane++) {
    PolyUOp *idx = start;
    if (start_const) {
      idx = poly_uop0(ctx, POLY_OP_CONST, start->dtype, poly_arg_int(start_i + lane));
    } else if (lane != 0) {
      PolyUOp *lane_uop = poly_uop0(ctx, POLY_OP_CONST, start->dtype, poly_arg_int(lane));
      idx = poly_uop2(ctx, POLY_OP_ADD, start->dtype, start, lane_uop, poly_arg_none());
    }
    PolyUOp *lane_addr = poly_uop2(ctx, POLY_OP_INDEX, base->dtype, base, idx, poly_arg_none());
    stores[lane] = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, lane_addr, value->src[lane], poly_arg_none());
  }
  PolyUOp *group = poly_uop(ctx, POLY_OP_GROUP, POLY_VOID, stores, value->n_src, poly_arg_none());
  free(stores);
  return group;
}

static PolyUOp *rule_x86_isel_address_lea_graph(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || (u->op != POLY_OP_INDEX && u->op != POLY_OP_SHRINK) || u->n_src < 1)
    return NULL;
  X86IselUseCtx *uctx = (X86IselUseCtx *)poly_graph_rewrite_userctx();
  if (x86_isel_is_memory_address_use(uctx, u) && !poly_dtype_eq(u->dtype, POLY_UINT64))
    return NULL;
  if (u->op == POLY_OP_SHRINK && u->n_src >= 3) {
    int64_t width = 0;
    if (x86_const_as_i64(u->src[2], &width) && width > 1) return NULL;
  }
  PolyUOp *base = u->src[0];
  if (!base || base->dtype.count != 1) return NULL;
  PolyUOp *addr[4];
  if (x86_graph_fold_address(ctx, u, addr) != 0) return NULL;
  return x86_ins(ctx, POLY_X86_LEA, x86_u64(), addr, 4, x86_graph_vreg(x86_u64(), false));
}

static PolyUOp *rule_x86_isel_sink_graph(PolyCtx *ctx, PolyUOp *u, const PolyBindings *b) {
  (void)b;
  if (!u || u->op != POLY_OP_SINK) return NULL;
  PolyX86Op first_op = 0;
  if (u->n_src > 0 && x86_ins_op(u->src[0], &first_op) && first_op == POLY_X86_RET)
    return NULL;
  PolyUOp *srcs[80];
  int ns = 0;
  for (int i = 0; i < u->n_src && ns < (int)(sizeof(srcs) / sizeof(srcs[0])); i++)
    srcs[ns++] = u->src[i];
  for (int i = 0; i < (int)(sizeof(x86_callee_saved_gprs) / sizeof(x86_callee_saved_gprs[0])); i++) {
    if (ns >= (int)(sizeof(srcs) / sizeof(srcs[0]))) return NULL;
    int32_t reg = x86_tag_real(X86_REG_CLASS_WGPR, x86_callee_saved_gprs[i]);
    srcs[ns++] = x86_define_reg(ctx, x86_u64(), reg);
  }
  PolyUOp *ret = x86_ins_nodef(ctx, POLY_X86_RET, POLY_VOID, srcs, ns);
  return poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, ret, u->arg);
}

static _Thread_local PolyPatternMatcher *g_pm_x86_graph_readmem = NULL;
static PolyPatternMatcher *poly_pm_x86_graph_readmem(void) {
  if (g_pm_x86_graph_readmem) return g_pm_x86_graph_readmem;
  PolyRule rules[] = {
      {poly_pat_op(POLY_OP_INS, NULL, 0, NULL), rule_x86_isel_readmem_graph},
  };
  g_pm_x86_graph_readmem = poly_pm_new(rules, (int)(sizeof(rules) / sizeof(rules[0])));
  return g_pm_x86_graph_readmem;
}

static _Thread_local PolyPatternMatcher *g_pm_x86_graph_isel_ordered = NULL;
static PolyPatternMatcher *poly_pm_x86_graph_isel_ordered(void) {
  if (g_pm_x86_graph_isel_ordered) return g_pm_x86_graph_isel_ordered;
  PolyNamedRule rules[] = {
      /* Op -> Op, matching tinygrad's first isel block. */
      {.pat = poly_pat_op(POLY_OP_CAST, NULL, 0, NULL), .fn = rule_x86_isel_cast_void_graph, .name = "x86.isel.cast_void"},
      {.pat = poly_pat_op(POLY_OP_INDEX, NULL, 0, NULL), .fn = rule_x86_isel_wide_load_lane_graph, .name = "x86.isel.wide_load_lane"},
      {.pat = poly_pat_op(POLY_OP_INDEX, NULL, 0, NULL), .fn = rule_x86_isel_float_index_lane0_graph, .name = "x86.isel.float_index_lane0"},
      {.pat = poly_pat_op(POLY_OP_RANGE, NULL, 0, NULL), .fn = rule_x86_isel_range_graph, .name = "x86.isel.range"},

      /* Op -> X86Op. Keep these in tinygrad x86.py isel_matcher order. */
      {.pat = poly_pat_op(POLY_OP_SINK, NULL, 0, NULL), .fn = rule_x86_isel_sink_graph, .name = "x86.isel.sink_ret"},
      {.pat = poly_pat_op(POLY_OP_PARAM, NULL, 0, NULL), .fn = rule_x86_isel_abi_graph, .name = "x86.isel.abi_param"},
      {.pat = poly_pat_op(POLY_OP_SPECIAL, NULL, 0, NULL), .fn = rule_x86_isel_abi_graph, .name = "x86.isel.abi_special"},
      {.pat = poly_pat_op(POLY_OP_CONST, NULL, 0, NULL), .fn = rule_x86_isel_const_graph, .name = "x86.isel.const"},

      {.pat = poly_pat_op(POLY_OP_WHERE, NULL, 0, NULL), .fn = rule_x86_isel_where_graph, .name = "x86.isel.where"},
      {.pat = poly_pat_op(POLY_OP_IF, NULL, 0, NULL), .fn = rule_x86_isel_if_graph, .name = "x86.isel.if"},
      {.pat = poly_pat_op(POLY_OP_CMPLT, NULL, 0, NULL), .fn = rule_x86_isel_compare_graph, .name = "x86.isel.compare_lt"},
      {.pat = poly_pat_op(POLY_OP_CMPEQ, NULL, 0, NULL), .fn = rule_x86_isel_compare_graph, .name = "x86.isel.compare_eq"},
      {.pat = poly_pat_op(POLY_OP_CMPNE, NULL, 0, NULL), .fn = rule_x86_isel_compare_graph, .name = "x86.isel.compare_ne"},

      {.pat = poly_pat_op(POLY_OP_SQRT, NULL, 0, NULL), .fn = rule_x86_isel_float_unary_graph, .name = "x86.isel.float_unary_sqrt"},
      {.pat = poly_pat_op(POLY_OP_TRUNC, NULL, 0, NULL), .fn = rule_x86_isel_float_unary_graph, .name = "x86.isel.float_unary_trunc"},
      {.pat = poly_pat_op(POLY_OP_STACK, NULL, 0, NULL), .fn = rule_x86_isel_stack_graph, .name = "x86.isel.stack_shuffle"},
      {.pat = poly_pat_op(POLY_OP_INDEX, NULL, 0, NULL), .fn = rule_x86_isel_vector_index_graph, .name = "x86.isel.vector_index"},
      {.pat = poly_pat_op(POLY_OP_GEP, NULL, 0, NULL), .fn = rule_x86_isel_vector_index_graph, .name = "x86.isel.vector_gep"},

      {.pat = poly_pat_op(POLY_OP_ADD, NULL, 0, NULL), .fn = rule_x86_isel_fma_graph, .name = "x86.isel.fma"},
      {.pat = poly_pat_op(POLY_OP_MULACC, NULL, 0, NULL), .fn = rule_x86_isel_mulacc_ins_graph, .name = "x86.isel.mulacc"},

      {.pat = poly_pat_op(POLY_OP_ADD, NULL, 0, NULL), .fn = rule_x86_isel_float_bin_graph, .name = "x86.isel.float_add"},
      {.pat = poly_pat_op(POLY_OP_SUB, NULL, 0, NULL), .fn = rule_x86_isel_float_bin_graph, .name = "x86.isel.float_sub"},
      {.pat = poly_pat_op(POLY_OP_MUL, NULL, 0, NULL), .fn = rule_x86_isel_float_bin_graph, .name = "x86.isel.float_mul"},
      {.pat = poly_pat_op(POLY_OP_FDIV, NULL, 0, NULL), .fn = rule_x86_isel_float_bin_graph, .name = "x86.isel.float_fdiv"},
      {.pat = poly_pat_op(POLY_OP_MAX, NULL, 0, NULL), .fn = rule_x86_isel_float_bin_graph, .name = "x86.isel.float_max"},

      {.pat = poly_pat_op(POLY_OP_CDIV, NULL, 0, NULL), .fn = rule_x86_isel_cdiv_graph, .name = "x86.isel.cdiv"},
      {.pat = poly_pat_op(POLY_OP_ADD, NULL, 0, NULL), .fn = rule_x86_isel_scalar_int_bin_graph, .name = "x86.isel.int_add"},
      {.pat = poly_pat_op(POLY_OP_SUB, NULL, 0, NULL), .fn = rule_x86_isel_scalar_int_bin_graph, .name = "x86.isel.int_sub"},
      {.pat = poly_pat_op(POLY_OP_MUL, NULL, 0, NULL), .fn = rule_x86_isel_scalar_int_bin_graph, .name = "x86.isel.int_mul"},
      {.pat = poly_pat_op(POLY_OP_AND, NULL, 0, NULL), .fn = rule_x86_isel_scalar_int_bin_graph, .name = "x86.isel.int_and"},
      {.pat = poly_pat_op(POLY_OP_OR, NULL, 0, NULL), .fn = rule_x86_isel_scalar_int_bin_graph, .name = "x86.isel.int_or"},
      {.pat = poly_pat_op(POLY_OP_XOR, NULL, 0, NULL), .fn = rule_x86_isel_scalar_int_bin_graph, .name = "x86.isel.int_xor"},
      {.pat = poly_pat_op(POLY_OP_SHL, NULL, 0, NULL), .fn = rule_x86_isel_scalar_int_bin_graph, .name = "x86.isel.int_shl"},
      {.pat = poly_pat_op(POLY_OP_SHR, NULL, 0, NULL), .fn = rule_x86_isel_scalar_int_bin_graph, .name = "x86.isel.int_shr"},

      {.pat = poly_pat_op(POLY_OP_CAST, NULL, 0, NULL), .fn = rule_x86_isel_cast_graph, .name = "x86.isel.cast"},
      {.pat = poly_pat_op(POLY_OP_BITCAST, NULL, 0, NULL), .fn = rule_x86_isel_bitcast_graph, .name = "x86.isel.bitcast"},

      {.pat = poly_pat_op(POLY_OP_INDEX, NULL, 0, NULL), .fn = rule_x86_isel_address_lea_graph, .name = "x86.isel.address_index"},
      {.pat = poly_pat_op(POLY_OP_SHRINK, NULL, 0, NULL), .fn = rule_x86_isel_address_lea_graph, .name = "x86.isel.address_shrink"},
      {.pat = poly_pat_op(POLY_OP_LOAD, NULL, 0, NULL), .fn = rule_x86_isel_load_graph, .name = "x86.isel.load"},
      {.pat = poly_pat_op(POLY_OP_STORE, NULL, 0, NULL), .fn = rule_x86_isel_wide_stack_store_graph, .name = "x86.isel.wide_stack_store"},
      {.pat = poly_pat_op(POLY_OP_STORE, NULL, 0, NULL), .fn = rule_x86_isel_store_graph, .name = "x86.isel.store"},

      /* X86Op -> X86Op and vreg allocation happen last in tinygrad. */
      {.pat = poly_pat_op(POLY_OP_INS, NULL, 0, NULL), .fn = rule_x86_isel_readmem_graph, .name = "x86.isel.readmem"},
  };
  g_pm_x86_graph_isel_ordered = poly_pm_new_named(rules, (int)(sizeof(rules) / sizeof(rules[0])));
  return g_pm_x86_graph_isel_ordered;
}

static PolyUOp *poly_graph_rewrite_x86_with_uses(
    PolyCtx *ctx,
    PolyUOp *sink,
    PolyPatternMatcher *pm,
    int *next_vreg
) {
  if (!ctx || !sink || !pm) return sink;
  int n = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, sink, &n);
  if (!topo) return NULL;
  X86IselUseCtx uctx = {
      .uses = poly_map_new((size_t)(n > 0 ? n * 2 : 16)),
      .single_consumer = poly_map_new((size_t)(n > 0 ? n : 16)),
      .structural_consts = poly_map_new((size_t)(n > 0 ? n : 16)),
      .address_memory_uses = poly_map_new((size_t)(n > 0 ? n : 16)),
      .next_vreg = next_vreg ? *next_vreg : 0,
  };
  if (!uctx.uses || !uctx.single_consumer || !uctx.structural_consts || !uctx.address_memory_uses) {
    poly_map_destroy(uctx.uses);
    poly_map_destroy(uctx.single_consumer);
    poly_map_destroy(uctx.structural_consts);
    poly_map_destroy(uctx.address_memory_uses);
    poly_toposort_free(topo);
    return NULL;
  }
  for (int i = 0; i < n; i++) {
    PolyUOp *u = topo[i];
    if (!u) continue;
    for (int s = 0; s < u->n_src; s++) {
      x86_isel_inc_use(uctx.uses, u->src[s]);
      x86_isel_note_consumer(uctx.single_consumer, u->src[s], u);
      if (u->op == POLY_OP_LOAD || u->op == POLY_OP_STORE || u->op == POLY_OP_INS)
        x86_isel_mark_memory_address_use(uctx.address_memory_uses, u->src[s]);
    }
    if (u->op == POLY_OP_RANGE && u->n_src > 0)
      x86_isel_mark_structural_const(uctx.structural_consts, u->src[0]);
    if ((u->op == POLY_OP_INDEX || u->op == POLY_OP_GEP) && u->n_src > 1)
      x86_isel_mark_structural_const(uctx.structural_consts, u->src[1]);
    if (u->op == POLY_OP_SHRINK) {
      if (u->n_src > 1) x86_isel_mark_structural_const(uctx.structural_consts, u->src[1]);
      if (u->n_src > 2) x86_isel_mark_structural_const(uctx.structural_consts, u->src[2]);
    }
    if (u->op == POLY_OP_PARAM || u->op == POLY_OP_SPECIAL) uctx.n_func_args++;
  }
  if (uctx.n_func_args > 0) {
    uctx.func_args = malloc((size_t)uctx.n_func_args * sizeof(*uctx.func_args));
    if (!uctx.func_args) {
      poly_map_destroy(uctx.uses);
      poly_map_destroy(uctx.single_consumer);
      poly_map_destroy(uctx.structural_consts);
      poly_map_destroy(uctx.address_memory_uses);
      poly_toposort_free(topo);
      return NULL;
    }
    int j = 0;
    for (int i = 0; i < n; i++) {
      PolyUOp *u = topo[i];
      if (u && (u->op == POLY_OP_PARAM || u->op == POLY_OP_SPECIAL))
        uctx.func_args[j++] = u;
    }
    for (int i = 1; i < uctx.n_func_args; i++) {
      PolyUOp *cur = uctx.func_args[i];
      int j2 = i - 1;
      while (j2 >= 0 && x86_func_arg_less(cur, uctx.func_args[j2])) {
        uctx.func_args[j2 + 1] = uctx.func_args[j2];
        j2--;
      }
      uctx.func_args[j2 + 1] = cur;
    }
  }
  poly_toposort_free(topo);
  PolyUOp *out = poly_graph_rewrite_ctx_ex(ctx, sink, pm, &uctx, true);
  if (next_vreg) *next_vreg = uctx.next_vreg;
  free(uctx.func_args);
  poly_map_destroy(uctx.uses);
  poly_map_destroy(uctx.single_consumer);
  poly_map_destroy(uctx.structural_consts);
  poly_map_destroy(uctx.address_memory_uses);
  return out;
}

static PolyUOp *poly_graph_rewrite_x86_isel(PolyCtx *ctx, PolyUOp *sink) {
  if (!ctx || !sink) return NULL;
  int next_vreg = 0;
  PolyUOp *out = poly_graph_rewrite_x86_with_uses(
      ctx, sink, poly_pm_x86_graph_isel_ordered(), &next_vreg
  );
  if (!out) return NULL;
  /* tinygrad's single ordered isel matcher includes X86Op->X86Op ReadMem
   * folding and graph_rewrite reaches a fixed point. Polygrad's C rewrite can
   * expose new ReadMem candidates after LOAD becomes INS(VMOV*) or after a
   * previous fold removes an intermediate load, so close the selected graph
   * under the ReadMem rule with fresh use counts before linearization. */
  for (int i = 0; i < 4; i++) {
    PolyUOp *next =
        poly_graph_rewrite_x86_with_uses(ctx, out, poly_pm_x86_graph_readmem(), &next_vreg);
    if (!next) return NULL;
    if (next == out) return out;
    out = next;
  }
  return out;
}

static bool x86_graph_isel_residual_allowed(PolyUOp *u) {
  if (!u) return true;
  switch (u->op) {
  case POLY_OP_NOOP:
  case POLY_OP_GROUP:
  case POLY_OP_BARRIER:
  case POLY_OP_AFTER:
  case POLY_OP_PARAM:
  case POLY_OP_SPECIAL:
  case POLY_OP_CONST:
  case POLY_OP_INS:
  case POLY_OP_RANGE:
  case POLY_OP_END:
  case POLY_OP_DEFINE_VAR:
  case POLY_OP_BUFFER:
  case POLY_OP_SINK:
    return true;
  default:
    return false;
  }
}

static bool x86_validate_graph_isel_residuals(PolyUOp **lin, int n) {
  if (!lin) return false;
  int n_bad = 0;
  for (int i = 0; i < n; i++) {
    PolyUOp *u = lin[i];
    if (x86_graph_isel_residual_allowed(u)) continue;
    if (poly_debug_at_least(4)) {
      if (n_bad == 0) fprintf(stderr, "x86 graph isel residuals:\n");
      char *s = poly_uop_str(u);
      fprintf(stderr, "  base %04d %s\n", i, s ? s : "<uop>");
      free(s);
    }
    n_bad++;
  }
  if (poly_debug_at_least(5) && !n_bad) fprintf(stderr, "x86 graph isel residuals: none\n");
  return n_bad == 0;
}

static int64_t x86_float_bits(PolyDType dt, double v) {
  PolyDType s = poly_dtype_scalar(dt);
  if (s.bitsize == 16) {
    float f = (float)v;
    uint32_t bits = 0;
    memcpy(&bits, &f, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000u;
    uint32_t exp = (bits >> 23) & 0xffu;
    uint32_t mant = bits & 0x7fffffu;
    if (exp == 0xffu) return (int64_t)(sign | 0x7c00u | (mant ? 0x0200u : 0u));
    int32_t half_exp = (int32_t)exp - 127 + 15;
    if (half_exp >= 31) return (int64_t)(sign | 0x7c00u);
    if (half_exp <= 0) {
      if (half_exp < -10) return (int64_t)sign;
      mant |= 0x800000u;
      int shift = 14 - half_exp;
      uint32_t half_mant = mant >> shift;
      uint32_t rem = mant & ((1u << shift) - 1u);
      uint32_t halfway = 1u << (shift - 1);
      if (rem > halfway || (rem == halfway && (half_mant & 1u))) half_mant++;
      return (int64_t)(sign | half_mant);
    }
    uint32_t half_mant = mant >> 13;
    uint32_t rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (half_mant & 1u))) {
      half_mant++;
      if (half_mant == 0x400u) {
        half_mant = 0;
        half_exp++;
        if (half_exp >= 31) return (int64_t)(sign | 0x7c00u);
      }
    }
    return (int64_t)(sign | ((uint32_t)half_exp << 10) | half_mant);
  }
  if (s.bitsize == 32) {
    float f = (float)v;
    uint32_t bits = 0;
    memcpy(&bits, &f, sizeof(bits));
    return (int64_t)bits;
  }
  double d = v;
  uint64_t bits = 0;
  memcpy(&bits, &d, sizeof(bits));
  return (int64_t)bits;
}

typedef struct {
  PolyCtx *ctx;
  X86UOpMap map;
  X86UOpVec out;
  int next_vreg;
  int next_label;
  int n_buffer_args;
  int next_var_arg;
  int core_id_arg;
} X86IselCtx;

static int x86_isel_one(X86IselCtx *isel, PolyUOp *u);
static int x86_emit_const(X86IselCtx *isel, PolyUOp *u);

static int x86_emit_and_map(X86IselCtx *isel, PolyUOp *old, PolyUOp *new_uop) {
  if (x86_vec_push(&isel->out, new_uop) != 0) return -1;
  return old ? x86_map_put(&isel->map, old, new_uop) : 0;
}

static PolyUOp *x86_mapped_src(X86IselCtx *isel, PolyUOp *u) {
  return x86_map_get(&isel->map, u);
}

static PolyUOp *x86_materialized_src(X86IselCtx *isel, PolyUOp *u) {
  PolyUOp *m = x86_mapped_src(isel, u);
  if (!m) return NULL;
  if (m->op == POLY_OP_CONST) {
    if (x86_emit_const(isel, m) != 0) return NULL;
    m = x86_mapped_src(isel, m);
  }
  return m;
}

static PolyUOp *x86_alias_value_as_dtype(PolyCtx *ctx, PolyUOp *src, PolyDType dtype) {
  if (!src) return NULL;
  int32_t reg = x86_uop_reg(src);
  return poly_uop_tagged_arg(ctx, src->op, dtype, src->src, src->n_src, src->arg, src->tag,
                             reg >= 0 ? x86_arg_int_tuple((int64_t[]){reg}, 1) : src->tag_arg);
}

static int64_t x86_local_buffer_nbytes(PolyUOp *u) {
  if (!u || !u->dtype.is_ptr) return 0;
  PolyDType s = poly_dtype_scalar(u->dtype);
  int item = poly_dtype_itemsize(s);
  if (item <= 0) item = 1;
  int count = u->dtype.count > 1 ? u->dtype.count : 1;
  int64_t n = u->dtype.ptr_size > 0 ? u->dtype.ptr_size : 1;
  int64_t bytes = n * (int64_t)item * (int64_t)count;
  return bytes > 0 ? bytes : item;
}

static int x86_emit_buffer(X86IselCtx *isel, PolyUOp *u) {
  if (!u || !u->dtype.is_ptr) return -1;
  if (u->dtype.addrspace != POLY_ADDR_REG && u->dtype.addrspace != POLY_ADDR_LOCAL) return -1;
  int32_t def = x86_virtual_tag_for_dtype(x86_u64(), true, &isel->next_vreg);
  PolyUOp *buf = poly_uop_tagged_arg(
      isel->ctx, POLY_OP_BUFFER, u->dtype, NULL, 0, u->arg, def,
      x86_arg_int_tuple((int64_t[]){def}, 1)
  );
  return x86_emit_and_map(isel, u, buf);
}

static PolyUOp *x86_disp_const(PolyCtx *ctx, int64_t disp) {
  PolyDType dt = (disp >= INT8_MIN && disp <= INT8_MAX) ? POLY_INT8 : POLY_INT32;
  return x86_const_i(ctx, dt, disp);
}

static PolyX86Op x86_mov_op_for_dtype(PolyDType dt, bool store) {
  int bytes = x86_value_size(dt, false);
  if (bytes >= 16) return store ? POLY_X86_VMOVUPSm : POLY_X86_VMOVUPS;
  if (dt.count <= 1 && x86_is_int_dtype(dt)) return store ? POLY_X86_MOVm : POLY_X86_MOV;
  if (bytes == 8) return store ? POLY_X86_VMOVSDm : POLY_X86_VMOVSD;
  return store ? POLY_X86_VMOVSSm : POLY_X86_VMOVSS;
}

static PolyX86Op x86_float_bin_op(PolyOps op, PolyDType dt) {
  bool f64 = x86_dtype_is_float_bits(dt, 64);
  bool vec = dt.count > 1;
  switch (op) {
  case POLY_OP_ADD: return f64 ? (vec ? POLY_X86_VADDPD : POLY_X86_VADDSD)
                               : (vec ? POLY_X86_VADDPS : POLY_X86_VADDSS);
  case POLY_OP_SUB: return f64 ? (vec ? POLY_X86_VSUBPD : POLY_X86_VSUBSD)
                               : (vec ? POLY_X86_VSUBPS : POLY_X86_VSUBSS);
  case POLY_OP_MUL: return f64 ? (vec ? POLY_X86_VMULPD : POLY_X86_VMULSD)
                               : (vec ? POLY_X86_VMULPS : POLY_X86_VMULSS);
  case POLY_OP_FDIV: return f64 ? (vec ? POLY_X86_VDIVPD : POLY_X86_VDIVSD)
                                : (vec ? POLY_X86_VDIVPS : POLY_X86_VDIVSS);
  case POLY_OP_MAX: return f64 ? (vec ? POLY_X86_VMAXPD : POLY_X86_VMAXSD)
                               : (vec ? POLY_X86_VMAXPS : POLY_X86_VMAXSS);
  default: return 0;
  }
}

static PolyX86Op x86_int_bin_op(PolyOps op, PolyDType dt, bool imm) {
  if (dt.count > 1) {
    int bits = poly_dtype_scalar(dt).bitsize;
    switch (op) {
    case POLY_OP_ADD:
      return bits <= 8 ? POLY_X86_VPADDB : bits <= 16 ? POLY_X86_VPADDW
                                                       : bits <= 32 ? POLY_X86_VPADDD : POLY_X86_VPADDQ;
    case POLY_OP_SUB:
      return bits <= 8 ? POLY_X86_VPSUBB : bits <= 16 ? POLY_X86_VPSUBW
                                                       : bits <= 32 ? POLY_X86_VPSUBD : POLY_X86_VPSUBQ;
    case POLY_OP_MUL: return bits == 16 ? POLY_X86_VPMULLW : bits == 32 ? POLY_X86_VPMULLD : 0;
    case POLY_OP_AND: return POLY_X86_VPAND;
    case POLY_OP_OR: return POLY_X86_VPOR;
    case POLY_OP_XOR: return POLY_X86_VPXOR;
    case POLY_OP_SHL: return bits == 32 ? POLY_X86_VPSLLVD : bits == 64 ? POLY_X86_VPSLLVQ : 0;
    case POLY_OP_SHR:
      if (poly_dtype_is_unsigned(poly_dtype_scalar(dt)))
        return bits == 32 ? POLY_X86_VPSRLVD : bits == 64 ? POLY_X86_VPSRLVQ : 0;
      return bits == 32 ? POLY_X86_VPSRAVD : 0;
    default: return 0;
    }
  }
  switch (op) {
  case POLY_OP_ADD: return imm ? POLY_X86_ADDi : POLY_X86_ADD;
  case POLY_OP_SUB: return imm ? POLY_X86_SUBi : POLY_X86_SUB;
  case POLY_OP_MUL: return imm ? POLY_X86_IMULi : POLY_X86_IMUL;
  case POLY_OP_AND: return imm ? POLY_X86_ANDi : POLY_X86_AND;
  case POLY_OP_OR: return imm ? POLY_X86_ORi : POLY_X86_OR;
  case POLY_OP_XOR: return imm ? POLY_X86_XORi : POLY_X86_XOR;
  case POLY_OP_SHL: return imm ? POLY_X86_SHLi : POLY_X86_SHL;
  case POLY_OP_SHR: return poly_dtype_is_unsigned(poly_dtype_scalar(dt)) ? (imm ? POLY_X86_SHRi : POLY_X86_SHR)
                                                                         : (imm ? POLY_X86_SARi : POLY_X86_SAR);
  default: return 0;
  }
}

static PolyX86Op x86_vector_int_cmp_op(PolyOps op, PolyDType dt, bool *swap) {
  if (swap) *swap = false;
  if (dt.count <= 1 || !x86_is_int_dtype(dt)) return 0;
  int bits = poly_dtype_scalar(dt).bitsize;
  if (op == POLY_OP_CMPEQ) {
    return bits <= 8 ? POLY_X86_VPCMPEQB : bits <= 16 ? POLY_X86_VPCMPEQW
                                                       : bits <= 32 ? POLY_X86_VPCMPEQD : POLY_X86_VPCMPEQQ;
  }
  if (op == POLY_OP_CMPLT && !poly_dtype_is_unsigned(poly_dtype_scalar(dt))) {
    if (swap) *swap = true;
    return bits <= 8 ? POLY_X86_VPCMPGTB : bits <= 16 ? POLY_X86_VPCMPGTW
                                                       : bits <= 32 ? POLY_X86_VPCMPGTD : POLY_X86_VPCMPGTQ;
  }
  return 0;
}

static PolyX86Op x86_vector_int_extend_op(PolyDType src_dt, PolyDType dst_dt) {
  if (src_dt.count <= 1 || dst_dt.count <= 1) return 0;
  if (!x86_is_int_dtype(src_dt) || !x86_is_int_dtype(dst_dt)) return 0;
  int sb = poly_dtype_scalar(src_dt).bitsize;
  int db = poly_dtype_scalar(dst_dt).bitsize;
  if (db <= sb) return 0;
  bool uns = poly_dtype_is_unsigned(poly_dtype_scalar(src_dt)) ||
             poly_dtype_is_bool(poly_dtype_scalar(src_dt));
  if (uns) {
    if (sb == 8 && db == 16) return POLY_X86_VPMOVZXBW;
    if (sb == 8 && db == 32) return POLY_X86_VPMOVZXBD;
    if (sb == 8 && db == 64) return POLY_X86_VPMOVZXBQ;
    if (sb == 16 && db == 32) return POLY_X86_VPMOVZXWD;
    if (sb == 16 && db == 64) return POLY_X86_VPMOVZXWQ;
    if (sb == 32 && db == 64) return POLY_X86_VPMOVZXDQ;
  } else {
    if (sb == 8 && db == 16) return POLY_X86_VPMOVSXBW;
    if (sb == 8 && db == 32) return POLY_X86_VPMOVSXBD;
    if (sb == 8 && db == 64) return POLY_X86_VPMOVSXBQ;
    if (sb == 16 && db == 32) return POLY_X86_VPMOVSXWD;
    if (sb == 16 && db == 64) return POLY_X86_VPMOVSXWQ;
    if (sb == 32 && db == 64) return POLY_X86_VPMOVSXDQ;
  }
  return 0;
}

static int x86_emit_param(X86IselCtx *isel, PolyUOp *u) {
  int arg_idx = (u->arg.kind == POLY_ARG_INT) ? (int)u->arg.i : 0;
  int abi = x86_abi_gpr_for_arg(arg_idx);
  int32_t v = x86_virtual_tag_for_dtype(x86_u64(), true, &isel->next_vreg);
  PolyUOp *mov = NULL;
  if (abi >= 0) {
    PolyUOp *def = x86_define_reg(isel->ctx, x86_u64(), x86_tag_real(X86_REG_CLASS_WGPR, abi));
    PolyUOp *srcs[1] = {def};
    mov = x86_ins(isel->ctx, POLY_X86_MOV, x86_u64(), srcs, 1, v);
    if (x86_vec_push(&isel->out, def) != 0) return -1;
  } else {
    if (arg_idx < 6) return -1;
    int64_t caller_disp = ((int64_t)arg_idx - 5) * 8;
    int32_t addr_v = x86_virtual_tag_for_dtype(x86_u64(), true, &isel->next_vreg);
    PolyUOp *addr = x86_stack_arg_slot(isel->ctx, caller_disp, addr_v);
    if (x86_vec_push(&isel->out, addr) != 0) return -1;
    PolyUOp *srcs[4] = {
        addr, x86_noop(isel->ctx), x86_disp_const(isel->ctx, 0), x86_const_i(isel->ctx, POLY_UINT8, 8),
    };
    mov = x86_ins(isel->ctx, POLY_X86_MOV, x86_u64(), srcs, 4, v);
  }
  if (x86_emit_and_map(isel, u, mov) != 0) return -1;
  return 0;
}

static bool x86_is_core_id_var(PolyUOp *u) {
  return u && u->op == POLY_OP_DEFINE_VAR && u->arg.kind == POLY_ARG_DEFINE_VAR &&
         u->arg.define_var.name && strcmp(u->arg.define_var.name, "core_id") == 0;
}

static int x86_emit_define_var(X86IselCtx *isel, PolyUOp *u) {
  bool is_core_id = x86_is_core_id_var(u);
  int arg_idx = is_core_id ? isel->core_id_arg : isel->next_var_arg++;
  if (arg_idx < 0) return -1;
  int abi = x86_abi_gpr_for_arg(arg_idx);

  if (is_core_id) {
    if (abi < 0) return -1;
    int32_t val_v = x86_virtual_tag_for_dtype(u->dtype, false, &isel->next_vreg);
    PolyUOp *def = x86_define_reg(isel->ctx, u->dtype, x86_tag_real(X86_REG_CLASS_WGPR, abi));
    PolyUOp *srcs[1] = {def};
    if (x86_vec_push(&isel->out, def) != 0) return -1;
    return x86_emit_and_map(isel, u, x86_ins(isel->ctx, POLY_X86_MOV, u->dtype, srcs, 1, val_v));
  }

  int32_t ptr_v = x86_virtual_tag_for_dtype(x86_u64(), true, &isel->next_vreg);
  PolyUOp *ptr = NULL;
  if (abi >= 0) {
    PolyUOp *def = x86_define_reg(isel->ctx, x86_u64(), x86_tag_real(X86_REG_CLASS_WGPR, abi));
    PolyUOp *srcs[1] = {def};
    ptr = x86_ins(isel->ctx, POLY_X86_MOV, x86_u64(), srcs, 1, ptr_v);
    if (x86_vec_push(&isel->out, def) != 0) return -1;
    if (x86_vec_push(&isel->out, ptr) != 0) return -1;
  } else {
    int64_t caller_disp = ((int64_t)arg_idx - 5) * 8;
    int32_t addr_v = x86_virtual_tag_for_dtype(x86_u64(), true, &isel->next_vreg);
    PolyUOp *addr = x86_stack_arg_slot(isel->ctx, caller_disp, addr_v);
    if (x86_vec_push(&isel->out, addr) != 0) return -1;
    PolyUOp *srcs[4] = {
        addr, x86_noop(isel->ctx), x86_disp_const(isel->ctx, 0), x86_const_i(isel->ctx, POLY_UINT8, 8),
    };
    ptr = x86_ins(isel->ctx, POLY_X86_MOV, x86_u64(), srcs, 4, ptr_v);
    if (x86_vec_push(&isel->out, ptr) != 0) return -1;
  }

  int32_t val_v = x86_virtual_tag_for_dtype(u->dtype, false, &isel->next_vreg);
  PolyUOp *load_srcs[4] = {
      ptr, x86_noop(isel->ctx), x86_disp_const(isel->ctx, 0),
      x86_const_i(isel->ctx, POLY_UINT8, x86_value_size(u->dtype, false)),
  };
  PolyUOp *load = x86_ins(isel->ctx, POLY_X86_MOV, u->dtype, load_srcs, 4, val_v);
  return x86_emit_and_map(isel, u, load);
}

static int x86_emit_const(X86IselCtx *isel, PolyUOp *u) {
  if (u->arg.kind == POLY_ARG_INT || u->arg.kind == POLY_ARG_BOOL) {
    int64_t v = u->arg.kind == POLY_ARG_BOOL ? (u->arg.b ? 1 : 0) : u->arg.i;
    PolyDType dt = u->dtype;
    int32_t def = x86_virtual_tag_for_dtype(dt, false, &isel->next_vreg);
    PolyUOp *imm = x86_const_i(isel->ctx, dt, v);
    PolyUOp *srcs[1] = {imm};
    PolyX86Op op = x86_value_size(dt, false) == 8 ? POLY_X86_MOVABS : POLY_X86_MOVi;
    return x86_emit_and_map(isel, u, x86_ins(isel->ctx, op, dt, srcs, 1, def));
  }
  if (u->arg.kind == POLY_ARG_FLOAT) {
    PolyDType idt = x86_int_for_float(u->dtype);
    int64_t bits = x86_float_bits(u->dtype, u->arg.f);
    PolyUOp *ic = x86_const_i(isel->ctx, idt, bits);
    if (x86_emit_const(isel, ic) != 0) return -1;
    PolyUOp *imov = x86_mapped_src(isel, ic);
    int32_t def = x86_virtual_tag_for_dtype(u->dtype, false, &isel->next_vreg);
    PolyUOp *srcs[1] = {imov};
    PolyX86Op op = x86_dtype_is_float_bits(u->dtype, 64) ? POLY_X86_VMOVQ : POLY_X86_VMOVD;
    return x86_emit_and_map(isel, u, x86_ins(isel->ctx, op, u->dtype, srcs, 1, def));
  }
  return -1;
}

static int x86_emit_prebuilt_ins(X86IselCtx *isel, PolyUOp *u) {
  if (!isel || !u || u->op != POLY_OP_INS) return -1;
  PolyUOp *src_stack[32];
  PolyUOp **srcs = u->n_src > (int)(sizeof(src_stack) / sizeof(src_stack[0]))
                       ? malloc((size_t)u->n_src * sizeof(*srcs))
                       : src_stack;
  if (u->n_src > 0 && !srcs) return -1;
  for (int i = 0; i < u->n_src; i++) {
    PolyUOp *s = u->src[i];
    if (!s || (s->op == POLY_OP_NOOP && s->n_src == 0)) {
      srcs[i] = s;
      continue;
    }
    if (s->op == POLY_OP_CONST && !x86_is_float_dtype(s->dtype)) {
      srcs[i] = s;
      continue;
    }
    srcs[i] = x86_materialized_src(isel, s);
    if (!srcs[i]) {
      if (srcs != src_stack) free(srcs);
      return -1;
    }
  }

  int32_t defs[8];
  int n_defs = x86_uop_def_regs(u, defs, (int)(sizeof(defs) / sizeof(defs[0])));
  if (n_defs == 0 && !poly_dtype_eq(u->dtype, POLY_VOID)) {
    defs[n_defs++] = x86_virtual_tag_for_dtype(u->dtype, false, &isel->next_vreg);
  }
  PolyUOp *nu = x86_clone_with_regs(isel->ctx, u, srcs, u->n_src, defs, n_defs);
  if (srcs != src_stack) free(srcs);
  return nu ? x86_emit_and_map(isel, u, nu) : -1;
}

static int x86_emit_range(X86IselCtx *isel, PolyUOp *u) {
  PolyUOp *srcs[8];
  int ns = x86_replace_srcs(srcs, 8, u, &isel->map);
  if (ns < 1) return -1;
  if (u->src[0]->op == POLY_OP_CONST) srcs[0] = x86_imm_for_const(isel->ctx, u->src[0]);
  int32_t def = x86_virtual_tag_for_dtype(u->dtype, false, &isel->next_vreg);
  PolyUOp *ru = poly_uop_tagged_arg(
      isel->ctx, POLY_OP_RANGE, u->dtype, srcs, ns, u->arg, def,
      x86_arg_int_tuple((int64_t[]){def}, 1)
  );
  return x86_emit_and_map(isel, u, ru);
}

static int x86_emit_end(X86IselCtx *isel, PolyUOp *u) {
  PolyUOp *srcs[8];
  int ns = x86_replace_srcs(srcs, 8, u, &isel->map);
  if (ns < 1) return -1;
  PolyUOp *eu = poly_uop(isel->ctx, POLY_OP_END, u->dtype, srcs, ns, u->arg);
  return x86_emit_and_map(isel, u, eu);
}

static int x86_isel_one(X86IselCtx *isel, PolyUOp *u) {
  if (!u) return -1;
  switch (u->op) {
  case POLY_OP_NOOP:
    if (u->n_src > 0) {
      PolyUOp *base = x86_mapped_src(isel, u->src[0]);
      if (!base) return -1;
      if (base->op == POLY_OP_STACK && base->n_src > 0 && u->dtype.count <= 1)
        return x86_map_put(&isel->map, u, base->src[0]);
      if (!poly_dtype_eq(base->dtype, u->dtype))
        base = x86_alias_value_as_dtype(isel->ctx, base, u->dtype);
      return x86_map_put(&isel->map, u, base);
    }
    return x86_map_put(&isel->map, u, u);
  case POLY_OP_GROUP:
  case POLY_OP_BARRIER:
    return x86_map_put(&isel->map, u, u);
  case POLY_OP_AFTER:
    if (u->n_src > 0) {
      PolyUOp *base = x86_mapped_src(isel, u->src[0]);
      return base ? x86_map_put(&isel->map, u, base) : -1;
    }
    return x86_map_put(&isel->map, u, u);
  case POLY_OP_PARAM:
  case POLY_OP_SPECIAL:
    if (x86_uop_reg(u) >= 0)
      return x86_map_put(&isel->map, u, u);
    return x86_emit_param(isel, u);
  case POLY_OP_DEFINE_VAR:
    return x86_emit_define_var(isel, u);
  case POLY_OP_BUFFER:
    return x86_emit_buffer(isel, u);
  case POLY_OP_CONST:
    return x86_map_put(&isel->map, u, u);
  case POLY_OP_INS:
    return x86_emit_prebuilt_ins(isel, u);
  case POLY_OP_RANGE:
    return x86_emit_range(isel, u);
  case POLY_OP_END:
    return x86_emit_end(isel, u);
  case POLY_OP_SINK: {
    if (u->n_src > 0) {
      PolyUOp *first = x86_mapped_src(isel, u->src[0]);
      PolyX86Op first_op = 0;
      if (first && x86_ins_op(first, &first_op) && first_op == POLY_X86_RET)
        return x86_map_put(&isel->map, u, first);
    }
    PolyUOp *srcs[80];
    int ns = x86_replace_srcs(srcs, 64, u, &isel->map);
    if (ns < 0) return -1;
    for (int i = 0; i < (int)(sizeof(x86_callee_saved_gprs) / sizeof(x86_callee_saved_gprs[0])); i++) {
      if (ns >= (int)(sizeof(srcs) / sizeof(srcs[0]))) return -1;
      int32_t reg = x86_tag_real(X86_REG_CLASS_WGPR, x86_callee_saved_gprs[i]);
      PolyUOp *def = x86_define_reg(isel->ctx, x86_u64(), reg);
      if (x86_vec_push(&isel->out, def) != 0) return -1;
      srcs[ns++] = def;
    }
    PolyUOp *ret = x86_ins_nodef(isel->ctx, POLY_X86_RET, POLY_VOID, srcs, ns);
    return x86_emit_and_map(isel, u, ret);
  }
  default:
    fprintf(stderr, "x86 isel: unsupported op %s\n", poly_op_name(u->op));
    return -1;
  }
}

static PolyUOp **x86_isel_linear(PolyCtx *ctx, PolyUOp **lin, int n, int *n_out) {
  if (n_out) *n_out = 0;
  X86IselCtx isel = {.ctx = ctx};
  int max_param = -1;
  int max_graph_vreg = -1;
  int n_define_vars = 0;
  for (int i = 0; i < n; i++) {
    PolyUOp *u = lin[i];
    if (!u) continue;
    int32_t defs[16];
    int n_defs = x86_uop_def_regs(u, defs, (int)(sizeof(defs) / sizeof(defs[0])));
    for (int d = 0; d < n_defs; d++) {
      if (x86_tag_is_virtual(defs[d]) && x86_virtual_id(defs[d]) > max_graph_vreg)
        max_graph_vreg = x86_virtual_id(defs[d]);
    }
    if ((u->op == POLY_OP_PARAM || u->op == POLY_OP_SPECIAL) && u->arg.kind == POLY_ARG_INT &&
        u->arg.i > max_param)
      max_param = (int)u->arg.i;
    if (u->op == POLY_OP_DEFINE_VAR && !x86_is_core_id_var(u)) n_define_vars++;
  }
  isel.n_buffer_args = max_param + 1;
  isel.next_var_arg = isel.n_buffer_args;
  isel.core_id_arg = isel.n_buffer_args + n_define_vars;
  isel.next_vreg = max_graph_vreg + 1;
  for (int i = 0; i < n; i++) {
    if (x86_isel_one(&isel, lin[i]) != 0) {
      if (poly_debug_at_least(4)) {
        char *s = poly_uop_str(lin[i]);
        fprintf(stderr, "x86 isel failed at %d: %s\n", i, s ? s : "<uop>");
        free(s);
      }
      free(isel.out.items);
      x86_map_free(&isel.map);
      return NULL;
    }
  }
  x86_map_free(&isel.map);
  if (n_out) *n_out = isel.out.n;
  return isel.out.items;
}

typedef struct {
  PolyUOp **items;
  int n;
  int cap;
} X86PtrSet;

static bool x86_ptrset_contains(X86PtrSet *s, PolyUOp *u) {
  if (!s || !u) return false;
  for (int i = 0; i < s->n; i++)
    if (s->items[i] == u) return true;
  return false;
}

static int x86_ptrset_add(X86PtrSet *s, PolyUOp *u) {
  if (!s || !u || x86_ptrset_contains(s, u)) return 0;
  if (s->n >= s->cap) {
    int nc = s->cap ? s->cap * 2 : 16;
    PolyUOp **ni = realloc(s->items, (size_t)nc * sizeof(*ni));
    if (!ni) return -1;
    s->items = ni;
    s->cap = nc;
  }
  s->items[s->n++] = u;
  return 0;
}

static void x86_ptrset_remove(X86PtrSet *s, PolyUOp *u) {
  if (!s || !u) return;
  for (int i = 0; i < s->n; i++) {
    if (s->items[i] != u) continue;
    memmove(&s->items[i], &s->items[i + 1], (size_t)(s->n - i - 1) * sizeof(*s->items));
    s->n--;
    return;
  }
}

static PolyUOp *x86_flag_def_for(PolyUOp *u) {
  if (!u) return NULL;
  PolyX86Op op = 0;
  if (x86_ins_op(u, &op)) {
    if (x86_writeflags(op)) return u;
    if (x86_readflags(op) && u->n_src > 0) return u->src[u->n_src - 1];
  }
  if (u->op == POLY_OP_RANGE || u->op == POLY_OP_END) return u;
  return NULL;
}

static PolyUOp **x86_pre_regalloc_linear(PolyUOp **lin, int n, int *n_out) {
  if (n_out) *n_out = 0;
  X86UOpVec out = {0};
  X86PtrSet clobbered = {0};
  PolyUOp *lock = NULL;

  for (int i = 0; i < n; i++) {
    PolyUOp *u = lin[i];
    PolyUOp *flag_def = x86_flag_def_for(u);
    if (flag_def) {
      if (lock && lock != flag_def && x86_ptrset_add(&clobbered, lock) != 0) goto fail;
      lock = flag_def;
      if (x86_ptrset_contains(&clobbered, flag_def)) {
        x86_ptrset_remove(&clobbered, flag_def);
        if (x86_vec_push(&out, flag_def) != 0) goto fail;
      }
    }
    if (x86_vec_push(&out, u) != 0) goto fail;
  }

  free(clobbered.items);
  if (n_out) *n_out = out.n;
  return out.items;

fail:
  free(clobbered.items);
  free(out.items);
  if (n_out) *n_out = 0;
  return NULL;
}

static PolyUOp **x86_sort_reg_defs_for_regalloc(PolyUOp **lin, int n, int *n_out) {
  if (n_out) *n_out = 0;
  if (!lin || n < 0) return NULL;
  PolyUOp **out = malloc((size_t)(n > 0 ? n : 1) * sizeof(*out));
  if (!out) return NULL;
  int pos = 0;
  for (int pass = 0; pass < 2; pass++) {
    for (int i = 0; i < n; i++) {
      PolyUOp *u = lin[i];
      PolyX86Op op = 0;
      bool def_ins = u && u->op == POLY_OP_INS && u->n_src == 0 &&
                     x86_ins_op(u, &op) && op == POLY_X86_DEFINE;
      if ((pass == 0 && def_ins) || (pass == 1 && !def_ins))
        out[pos++] = u;
    }
  }
  if (n_out) *n_out = pos;
  return out;
}

static bool x86_ins_is_load_addr(PolyUOp *u) {
  PolyX86Op op = 0;
  if (!u || !x86_ins_op(u, &op) || u->n_src != 4 || poly_dtype_eq(u->dtype, POLY_VOID))
    return false;
  switch (op) {
  case POLY_X86_MOV:
  case POLY_X86_VMOVSS:
  case POLY_X86_VMOVSD:
  case POLY_X86_VMOVUPS:
  case POLY_X86_VMOVD:
  case POLY_X86_VMOVQ:
    return true;
  default:
    return false;
  }
}

static bool x86_ins_has_memory_tuple_at(PolyUOp *u, int src_idx) {
  if (!u || src_idx < 0 || src_idx + 3 >= u->n_src) return false;
  PolyUOp *disp = u->src[src_idx + 2];
  PolyUOp *size = u->src[src_idx + 3];
  return disp && disp->op == POLY_OP_CONST && size && size->op == POLY_OP_CONST;
}

static bool x86_tag_is_virtual(int32_t tag);
static bool x86_tag_is_real(int32_t tag);
static X86RegClass x86_tag_class(int32_t tag);
static int x86_tag_id(int32_t tag);

static PolyUOp *x86_clone_with_regs(
    PolyCtx *ctx,
    PolyUOp *u,
    PolyUOp **srcs,
    int n_src,
    int32_t *defs,
    int n_defs
);

typedef struct {
  int32_t vreg;
  int32_t real;
  PolyUOp *uop;
} X86LiveReg;

typedef struct {
  int32_t vreg;
  int offset;
  int size;
  PolyDType dtype;
  bool buffer_value;
  bool preserve_real_source;
} X86SpillSlot;

typedef struct {
  int *pos;
  int n;
  int cap;
} X86LiveRange;

typedef struct {
  int32_t vreg;
  int32_t real;
} X86LoopLiveIn;

typedef struct {
  X86LoopLiveIn *items;
  int n;
  int cap;
} X86LoopLiveSet;

typedef struct {
  X86LoopLiveSet *items;
  int n;
  int cap;
} X86LoopLiveStack;

typedef struct {
  int32_t vreg;
  int first;
  int next;
} X86LoopLiveCandidate;

typedef struct {
  PolyCtx *ctx;
  PolyUOp **uops;
  int n;
  int max_vreg;
  int *first;
  int *last;
  X86LiveRange *ranges;
  X86LiveReg live[64];
  int n_live;
  X86LoopLiveSet *reals_by_idx;
  X86LoopLiveSet *insert_before_by_idx;
  X86LoopLiveSet *spill_before_by_idx;
  int *local_offsets;
  X86SpillSlot *spills;
  int n_spills;
  int cap_spills;
  int stack_size;
  X86LoopLiveStack loop_live;
} X86RegAllocCtx;

static bool x86_tag_is_virtual(int32_t tag) {
  return tag > 0 && (tag & X86_TAG_VIRT) != 0;
}

static bool x86_tag_is_reg_key(int32_t tag) {
  return x86_tag_is_virtual(tag) || x86_tag_is_real(tag);
}

static int x86_virtual_id(int32_t tag) {
  return x86_tag_id(tag);
}

static PolyUOp *x86_alias_with_reg(PolyCtx *ctx, PolyUOp *u, int32_t reg) {
  if (!u || reg <= 0) return u;
  int64_t vals[1] = {reg};
  return poly_uop_tagged_arg(ctx, u->op, u->dtype, u->src, u->n_src, u->arg, reg, x86_arg_int_tuple(vals, 1));
}

static int x86_live_find(X86RegAllocCtx *ra, int32_t vreg) {
  for (int i = 0; i < ra->n_live; i++)
    if (ra->live[i].vreg == vreg) return i;
  return -1;
}

static int x86_live_find_real(X86RegAllocCtx *ra, int32_t real) {
  for (int i = 0; i < ra->n_live; i++)
    if (ra->live[i].real == real) return i;
  return -1;
}

static void x86_live_remove_at(X86RegAllocCtx *ra, int idx) {
  if (idx < 0 || idx >= ra->n_live) return;
  memmove(&ra->live[idx], &ra->live[idx + 1], (size_t)(ra->n_live - idx - 1) * sizeof(ra->live[0]));
  ra->n_live--;
}

static void x86_live_set(X86RegAllocCtx *ra, int32_t vreg, int32_t real, PolyUOp *uop) {
  int idx = x86_live_find(ra, vreg);
  if (idx < 0) {
    if (ra->n_live >= (int)(sizeof(ra->live) / sizeof(ra->live[0]))) abort();
    idx = ra->n_live++;
  }
  ra->live[idx] = (X86LiveReg){vreg, real, uop};
}

static void x86_live_ranges_free(X86RegAllocCtx *ra) {
  if (!ra || !ra->ranges) return;
  for (int i = 0; i <= ra->max_vreg; i++)
    free(ra->ranges[i].pos);
  free(ra->ranges);
  ra->ranges = NULL;
}

static int x86_live_range_add(X86RegAllocCtx *ra, int32_t vreg, int pos) {
  if (!ra || !x86_tag_is_virtual(vreg)) return 0;
  int id = x86_virtual_id(vreg);
  if (id < 0 || id > ra->max_vreg) return 0;
  X86LiveRange *lr = &ra->ranges[id];
  if (lr->n > 0 && lr->pos[lr->n - 1] == pos) return 0;
  for (int i = 0; i < lr->n; i++)
    if (lr->pos[i] == pos) return 0;
  if (lr->n >= lr->cap) {
    int nc = lr->cap ? lr->cap * 2 : 4;
    int *np = realloc(lr->pos, (size_t)nc * sizeof(*np));
    if (!np) return -1;
    lr->pos = np;
    lr->cap = nc;
  }
  lr->pos[lr->n++] = pos;
  if (pos < ra->first[id]) ra->first[id] = pos;
  if (pos > ra->last[id]) ra->last[id] = pos;
  return 0;
}

static int x86_int_cmp(const void *pa, const void *pb) {
  int a = *(const int *)pa, b = *(const int *)pb;
  return (a > b) - (a < b);
}

static void x86_live_ranges_sort(X86RegAllocCtx *ra) {
  if (!ra || !ra->ranges) return;
  for (int id = 0; id <= ra->max_vreg; id++) {
    X86LiveRange *lr = &ra->ranges[id];
    if (lr->n <= 1) continue;
    qsort(lr->pos, (size_t)lr->n, sizeof(*lr->pos), x86_int_cmp);
    int w = 0;
    for (int r = 0; r < lr->n; r++) {
      if (w == 0 || lr->pos[r] != lr->pos[w - 1])
        lr->pos[w++] = lr->pos[r];
    }
    lr->n = w;
    ra->first[id] = lr->pos[0];
    ra->last[id] = lr->pos[lr->n - 1];
  }
}

static void x86_loop_live_set_free(X86LoopLiveSet *s) {
  if (!s) return;
  free(s->items);
  *s = (X86LoopLiveSet){0};
}

static void x86_loop_live_stack_free(X86LoopLiveStack *st) {
  if (!st) return;
  for (int i = 0; i < st->n; i++)
    x86_loop_live_set_free(&st->items[i]);
  free(st->items);
  *st = (X86LoopLiveStack){0};
}

static int x86_loop_live_set_push(X86LoopLiveSet *s, int32_t vreg, int32_t real) {
  for (int i = 0; i < s->n; i++) {
    if (s->items[i].vreg == vreg) {
      s->items[i].real = real;
      return 0;
    }
  }
  if (s->n >= s->cap) {
    int nc = s->cap ? s->cap * 2 : 16;
    X86LoopLiveIn *ni = realloc(s->items, (size_t)nc * sizeof(*ni));
    if (!ni) return -1;
    s->items = ni;
    s->cap = nc;
  }
  s->items[s->n++] = (X86LoopLiveIn){vreg, real};
  return 0;
}

static bool x86_loop_live_set_get(const X86LoopLiveSet *s, int32_t vreg, int32_t *real_out) {
  if (!s) return false;
  for (int i = 0; i < s->n; i++) {
    if (s->items[i].vreg != vreg) continue;
    if (real_out) *real_out = s->items[i].real;
    return true;
  }
  return false;
}

static void x86_loop_live_sets_free(X86LoopLiveSet *sets, int n) {
  if (!sets) return;
  for (int i = 0; i < n; i++)
    x86_loop_live_set_free(&sets[i]);
  free(sets);
}

static int x86_loop_live_stack_push(X86LoopLiveStack *st, X86LoopLiveSet set) {
  if (st->n >= st->cap) {
    int nc = st->cap ? st->cap * 2 : 8;
    X86LoopLiveSet *ni = realloc(st->items, (size_t)nc * sizeof(*ni));
    if (!ni) return -1;
    st->items = ni;
    st->cap = nc;
  }
  st->items[st->n++] = set;
  return 0;
}

static bool x86_loop_live_stack_pop(X86LoopLiveStack *st, X86LoopLiveSet *out) {
  if (!st || st->n <= 0) return false;
  *out = st->items[--st->n];
  st->items[st->n] = (X86LoopLiveSet){0};
  return true;
}

static int x86_spill_find(X86RegAllocCtx *ra, int32_t vreg) {
  for (int i = 0; i < ra->n_spills; i++)
    if (ra->spills[i].vreg == vreg) return i;
  return -1;
}

static bool x86_real_is_future_abi_source(X86RegAllocCtx *ra, int32_t real, int at);

static int x86_spill_slot(X86RegAllocCtx *ra, int32_t vreg, PolyDType dt, bool buffer_value) {
  int idx = x86_spill_find(ra, vreg);
  if (idx >= 0) return idx;
  if (ra->n_spills >= ra->cap_spills) {
    int nc = ra->cap_spills ? ra->cap_spills * 2 : 64;
    X86SpillSlot *ns = realloc(ra->spills, (size_t)nc * sizeof(*ns));
    if (!ns) return -1;
    ra->spills = ns;
    ra->cap_spills = nc;
  }
  int size = x86_value_size(dt, buffer_value);
  if (size <= 0) size = 8;
  int align = size > 16 ? 16 : size;
  int offset = ra->stack_size;
  int rem = offset % align;
  if (rem) offset += align - rem;
  ra->spills[ra->n_spills] =
      (X86SpillSlot){.vreg = vreg, .offset = offset, .size = size, .dtype = dt, .buffer_value = buffer_value};
  ra->stack_size = offset + size;
  return ra->n_spills++;
}

static int x86_preserve_future_abi_source_before(
    X86RegAllocCtx *ra,
    int idx,
    int32_t real
) {
  if (!ra || idx < 0 || idx >= ra->n || !x86_tag_is_real(real)) return 0;
  if (!x86_real_is_future_abi_source(ra, real, idx + 1)) return 0;
  int existing = x86_spill_find(ra, real);
  if (existing >= 0 && ra->spills[existing].preserve_real_source) return 0;
  int slot_idx = x86_spill_slot(ra, real, x86_u64(), false);
  if (slot_idx < 0) return -1;
  ra->spills[slot_idx].preserve_real_source = true;
  return x86_loop_live_set_push(&ra->spill_before_by_idx[idx], real, real);
}

static int x86_stack_alloc(X86RegAllocCtx *ra, int size, int align) {
  if (size <= 0) size = 1;
  if (align <= 0) align = 1;
  if (align > 16) align = 16;
  int offset = ra->stack_size;
  int rem = offset % align;
  if (rem) offset += align - rem;
  ra->stack_size = offset + size;
  return offset;
}

static PolyUOp *x86_rsp(PolyCtx *ctx) {
  return x86_define_reg(ctx, x86_u64(), x86_tag_real(X86_REG_CLASS_WGPR, X86_REG_RSP));
}

static void x86_stack_address(PolyCtx *ctx, int offset, int size, PolyUOp *out[4]) {
  out[0] = x86_rsp(ctx);
  out[1] = x86_noop(ctx);
  out[2] = x86_disp_const(ctx, offset);
  out[3] = x86_const_i(ctx, POLY_UINT8, size);
}

static PolyUOp *x86_fill_from_slot(PolyCtx *ctx, const X86SpillSlot *slot, int32_t real) {
  PolyUOp *srcs[4];
  x86_stack_address(ctx, slot->offset, slot->size, srcs);
  PolyX86Op op = x86_mov_op_for_dtype(slot->buffer_value ? x86_u64() : slot->dtype, false);
  return x86_ins(ctx, op, slot->buffer_value ? x86_u64() : slot->dtype, srcs, 4, real);
}

static PolyUOp *x86_spill_to_slot(PolyCtx *ctx, const X86SpillSlot *slot, PolyUOp *value) {
  PolyUOp *srcs[5];
  x86_stack_address(ctx, slot->offset, slot->size, srcs);
  srcs[4] = value;
  PolyX86Op op = x86_mov_op_for_dtype(slot->buffer_value ? x86_u64() : slot->dtype, true);
  return x86_ins_nodef(ctx, op, POLY_VOID, srcs, 5);
}

static PolyUOp *x86_copy_to_reg(PolyCtx *ctx, PolyUOp *src, int32_t real);

static const int32_t *x86_real_pool(X86RegClass cls, int *n_out) {
  static const int32_t wgpr[] = {
      /* tinygrad's WGPR pool is every GPR except RSP. Fixed-register DIV/IDIV
       * constraints are represented by real-register defs and handled by the
       * same def-conflict spill path as other constrained instructions. */
      X86_REG_RAX, X86_REG_RCX, X86_REG_RDX, X86_REG_RBX, X86_REG_RBP,
      X86_REG_RSI, X86_REG_RDI, X86_REG_R8,  X86_REG_R9,  X86_REG_R10,
      X86_REG_R11, X86_REG_R12, X86_REG_R13, X86_REG_R14, X86_REG_R15,
  };
  static const int32_t fixed_rax[] = {X86_REG_RAX};
  static const int32_t fixed_rdx[] = {X86_REG_RDX};
  static const int32_t xmm[] = {
      0, 1, 2, 3, 4, 5, 6, 7,
      8, 9, 10, 11, 12, 13, 14, 15,
  };
  if (cls == X86_REG_CLASS_FIXED_RAX) {
    *n_out = 1;
    return fixed_rax;
  }
  if (cls == X86_REG_CLASS_FIXED_RDX) {
    *n_out = 1;
    return fixed_rdx;
  }
  if (cls == X86_REG_CLASS_XMM) {
    *n_out = (int)(sizeof(xmm) / sizeof(xmm[0]));
    return xmm;
  }
  *n_out = (int)(sizeof(wgpr) / sizeof(wgpr[0]));
  return wgpr;
}

static bool x86_real_avoided(int32_t real, const int32_t *avoid, int n_avoid) {
  for (int i = 0; i < n_avoid; i++)
    if (avoid[i] == real) return true;
  return false;
}

static void x86_avoid_push(int32_t *avoid, int *n_avoid, int cap, int32_t real) {
  if (!avoid || !n_avoid || !x86_tag_is_real(real) || *n_avoid >= cap) return;
  if (x86_real_avoided(real, avoid, *n_avoid)) return;
  avoid[(*n_avoid)++] = real;
}

static void x86_fixed_source_avoids(PolyUOp *u, int src_idx, int32_t *avoid, int *n_avoid, int cap) {
  PolyX86Op op;
  if (!u || !x86_ins_op(u, &op)) return;
  if ((op == POLY_X86_DIV || op == POLY_X86_IDIV) && src_idx == 1) {
    /* tinygrad's idiv selector constrains the divisor to WGPR - {RAX,RDX}.
     * RAX/RDX are the implicit dividend/remainder registers for DIV/IDIV. */
    x86_avoid_push(avoid, n_avoid, cap, x86_tag_real(X86_REG_CLASS_WGPR, X86_REG_RAX));
    x86_avoid_push(avoid, n_avoid, cap, x86_tag_real(X86_REG_CLASS_WGPR, X86_REG_RDX));
  }
}

static bool x86_define_real_wgpr(PolyUOp *u, int32_t *real_out) {
  PolyX86Op op;
  if (!u || !x86_ins_op(u, &op) || op != POLY_X86_DEFINE) return false;
  int32_t real = x86_uop_reg(u);
  if (!x86_tag_is_real(real) || x86_tag_class(real) != X86_REG_CLASS_WGPR) return false;
  if (real_out) *real_out = real;
  return true;
}

static bool x86_real_wgpr_source(PolyUOp *u, int32_t *real_out) {
  if (!u) return false;
  int32_t real = x86_uop_reg(u);
  if (!x86_tag_is_real(real) || x86_tag_class(real) != X86_REG_CLASS_WGPR) return false;
  if (real_out) *real_out = real;
  return true;
}

static bool x86_abi_real_wgpr_source(PolyUOp *src, int32_t *real_out) {
  if (!src) return false;
  if (x86_define_real_wgpr(src, real_out)) return true;
  if ((src->op == POLY_OP_PARAM || src->op == POLY_OP_SPECIAL) &&
      x86_real_wgpr_source(src, real_out))
    return true;
  return false;
}

static bool x86_mov_reads_real_wgpr_source(PolyUOp *u, int32_t *real_out) {
  PolyX86Op op;
  if (!u || !x86_ins_op(u, &op) || op != POLY_X86_MOV || u->n_src != 1) return false;
  return x86_abi_real_wgpr_source(u->src[0], real_out);
}

static bool x86_real_is_future_abi_source(X86RegAllocCtx *ra, int32_t real, int at) {
  if (!ra || !x86_tag_is_real(real) || x86_tag_class(real) != X86_REG_CLASS_WGPR) return false;
  for (int i = at; i < ra->n; i++) {
    int32_t source_real = 0;
    if (x86_mov_reads_real_wgpr_source(ra->uops[i], &source_real) && source_real == real)
      return true;
  }
  return false;
}

static bool x86_regalloc_pseudo_op(PolyUOp *u);

static int x86_next_use(X86RegAllocCtx *ra, int32_t vreg, int at) {
  if (x86_tag_is_real(vreg)) {
    for (int i = at; i < ra->n; i++) {
      PolyUOp *u = ra->uops[i];
      if (!u || x86_regalloc_pseudo_op(u)) continue;
      int32_t defs[16];
      int n_defs = x86_uop_def_regs(u, defs, (int)(sizeof(defs) / sizeof(defs[0])));
      for (int d = 0; d < n_defs; d++)
        if (defs[d] == vreg) return i;
      for (int s = 0; s < u->n_src; s++)
        if (x86_uop_reg(u->src[s]) == vreg) return i;
    }
    return INT32_MAX;
  }
  int id = x86_virtual_id(vreg);
  if (id < 0 || id > ra->max_vreg) return INT32_MAX;
  X86LiveRange *lr = &ra->ranges[id];
  for (int i = 0; i < lr->n; i++)
    if (lr->pos[i] >= at) return lr->pos[i];
  if (ra->last && ra->last[id] >= at) return ra->last[id];
  return INT32_MAX;
}

static int x86_loop_live_next_pos(X86RegAllocCtx *ra, int32_t vreg, int start, int end) {
  int id = x86_virtual_id(vreg);
  if (!ra || id < 0 || id > ra->max_vreg || !ra->ranges) return INT32_MAX;
  X86LiveRange *lr = &ra->ranges[id];
  for (int i = 0; i < lr->n; i++)
    if (lr->pos[i] >= start && lr->pos[i] < end) return lr->pos[i];
  return INT32_MAX;
}

static bool x86_loop_live_candidate_has(X86LoopLiveCandidate *cands, int n, int32_t vreg) {
  for (int i = 0; i < n; i++)
    if (cands[i].vreg == vreg) return true;
  return false;
}

static bool x86_loop_live_class_saturated(X86LoopLiveSet *set, int32_t vreg) {
  X86RegClass cls = x86_tag_class(vreg);
  X86RegClass real_cls = x86_real_class_for_constraint(cls);
  int n_pool = 0;
  (void)x86_real_pool(cls, &n_pool);
  int used[32];
  int n_used = 0;
  for (int i = 0; i < set->n; i++) {
    if (x86_tag_class(set->items[i].real) != real_cls) continue;
    bool seen = false;
    for (int j = 0; j < n_used; j++) {
      if (used[j] == set->items[i].real) {
        seen = true;
        break;
      }
    }
    if (!seen && n_used < (int)(sizeof(used) / sizeof(used[0]))) used[n_used++] = set->items[i].real;
  }
  return n_used >= n_pool;
}

static bool x86_regalloc_pseudo_op(PolyUOp *u) {
  if (!u) return true;
  return u->op == POLY_OP_CONST || u->op == POLY_OP_NOOP || u->op == POLY_OP_AFTER ||
         u->op == POLY_OP_BARRIER || u->op == POLY_OP_GROUP || u->op == POLY_OP_STACK;
}

static bool x86_small_vreg_seen(int32_t *items, int n, int32_t v) {
  for (int i = 0; i < n; i++)
    if (items[i] == v) return true;
  return false;
}

static void x86_collect_ranges(X86RegAllocCtx *ra) {
  ra->max_vreg = 0;
  for (int i = 0; i < ra->n; i++) {
    PolyUOp *u = ra->uops[i];
    if (!u) continue;
    int32_t defs[16];
    int n_defs = x86_uop_def_regs(u, defs, (int)(sizeof(defs) / sizeof(defs[0])));
    for (int j = 0; j < n_defs; j++) {
      int32_t r = defs[j];
      if (x86_tag_is_virtual(r) && x86_virtual_id(r) > ra->max_vreg) ra->max_vreg = x86_virtual_id(r);
    }
    for (int j = 0; j < u->n_src; j++) {
      int32_t r = x86_uop_reg(u->src[j]);
      if (x86_tag_is_virtual(r) && x86_virtual_id(r) > ra->max_vreg) ra->max_vreg = x86_virtual_id(r);
    }
  }
  ra->first = malloc((size_t)(ra->max_vreg + 1) * sizeof(*ra->first));
  ra->last = malloc((size_t)(ra->max_vreg + 1) * sizeof(*ra->last));
  ra->ranges = calloc((size_t)(ra->max_vreg + 1), sizeof(*ra->ranges));
  if (!ra->first || !ra->last || !ra->ranges) return;
  for (int i = 0; i <= ra->max_vreg; i++) {
    ra->first[i] = INT32_MAX;
    ra->last[i] = -1;
  }
  int32_t active_ranges_stack[128];
  int32_t *active_ranges = active_ranges_stack;
  int n_active_ranges = 0;
  int cap_active_ranges = (int)(sizeof(active_ranges_stack) / sizeof(active_ranges_stack[0]));
  for (int i = ra->n - 1; i >= 0; i--) {
    PolyUOp *u = ra->uops[i];
    if (!u || x86_regalloc_pseudo_op(u)) continue;
    int32_t defs[16];
    int n_defs = x86_uop_def_regs(u, defs, (int)(sizeof(defs) / sizeof(defs[0])));
    for (int j = 0; j < n_defs; j++)
      if (x86_live_range_add(ra, defs[j], i) != 0) return;
    int32_t seen_srcs[64];
    int n_seen_srcs = 0;
    for (int j = 0; j < u->n_src; j++) {
      int32_t r = x86_uop_reg(u->src[j]);
      if (!x86_tag_is_virtual(r) || x86_small_vreg_seen(seen_srcs, n_seen_srcs, r)) continue;
      if (n_seen_srcs < (int)(sizeof(seen_srcs) / sizeof(seen_srcs[0]))) seen_srcs[n_seen_srcs++] = r;
      if (x86_live_range_add(ra, r, i) != 0) return;
    }

    for (int j = 0; j < n_defs; j++) {
      int32_t v = defs[j];
      if (!x86_tag_is_virtual(v)) continue;
      int id = x86_virtual_id(v);
      if (id < 0 || id > ra->max_vreg || ra->last[id] < 0) continue;
      int extend_to = -1;
      for (int ri = 0; ri < n_active_ranges; ri++) {
        int rid = x86_virtual_id(active_ranges[ri]);
        if (rid < 0 || rid > ra->max_vreg || ra->first[rid] == INT32_MAX || ra->last[rid] < 0)
          continue;
        if (ra->first[rid] <= ra->last[id] && ra->last[id] < ra->last[rid] &&
            ra->last[rid] > extend_to)
          extend_to = ra->last[rid];
      }
      if (extend_to >= 0 && x86_live_range_add(ra, v, extend_to) != 0) return;
    }

    if (u->op == POLY_OP_RANGE) {
      int32_t r = x86_uop_reg(u);
      if (x86_tag_is_virtual(r)) {
        if (n_active_ranges >= cap_active_ranges) {
          int nc = cap_active_ranges * 2;
          int32_t *nr = active_ranges == active_ranges_stack
                            ? malloc((size_t)nc * sizeof(*nr))
                            : realloc(active_ranges, (size_t)nc * sizeof(*nr));
          if (!nr) {
            if (active_ranges != active_ranges_stack) free(active_ranges);
            return;
          }
          if (active_ranges == active_ranges_stack)
            memcpy(nr, active_ranges_stack, (size_t)n_active_ranges * sizeof(*nr));
          active_ranges = nr;
          cap_active_ranges = nc;
        }
        active_ranges[n_active_ranges++] = r;
      }
    }
  }
  if (active_ranges != active_ranges_stack) free(active_ranges);
  x86_live_ranges_sort(ra);
}

static PolyUOp *x86_clone_with_regs(PolyCtx *ctx, PolyUOp *u, PolyUOp **srcs, int n_src, int32_t *defs, int n_defs) {
  int64_t vals[8];
  PolyArg tag_arg = u ? u->tag_arg : poly_arg_none();
  int32_t tag = u ? u->tag : 0;
  if (n_defs > 0) {
    for (int i = 0; i < n_defs && i < 8; i++)
      vals[i] = defs[i];
    tag_arg = x86_arg_int_tuple(vals, n_defs < 8 ? n_defs : 8);
    tag = defs[0];
  }
  return poly_uop_tagged_arg(ctx, u->op, u->dtype, srcs, n_src, u->arg, tag, tag_arg);
}

static PolyUOp *x86_vdef(X86RegAllocCtx *ra, int32_t vreg) {
  if (x86_tag_is_real(vreg)) {
    for (int i = 0; i < ra->n; i++) {
      PolyUOp *u = ra->uops[i];
      if (!u || x86_regalloc_pseudo_op(u)) continue;
      int32_t defs[16];
      int n_defs = x86_uop_def_regs(u, defs, (int)(sizeof(defs) / sizeof(defs[0])));
      for (int d = 0; d < n_defs; d++)
        if (defs[d] == vreg) return u;
    }
    return NULL;
  }
  int id = x86_virtual_id(vreg);
  if (!ra || id < 0 || id > ra->max_vreg || !ra->first || ra->first[id] < 0 ||
      ra->first[id] >= ra->n)
    return NULL;
  return ra->uops[ra->first[id]];
}

static int x86_spill_slot_for_vreg(X86RegAllocCtx *ra, int32_t vreg) {
  int idx = x86_spill_find(ra, vreg);
  if (idx >= 0) return idx;
  PolyUOp *def = x86_vdef(ra, vreg);
  if (!def) return -1;
  bool buffer_value = def->op == POLY_OP_BUFFER;
  return x86_spill_slot(ra, vreg, buffer_value ? x86_u64() : def->dtype, buffer_value);
}

static void x86_debug_regalloc_summary(X86RegAllocCtx *ra) {
  if (!poly_debug_at_least(7) || !ra) return;
  fprintf(
      stderr,
      "[polygrad:x86-regalloc] uops=%d stack=%d spills=%d\n",
      ra->n, ra->stack_size, ra->n_spills
  );
  int n = ra->n_spills < 96 ? ra->n_spills : 96;
  for (int i = 0; i < n; i++) {
    X86SpillSlot *slot = &ra->spills[i];
    PolyUOp *def = x86_vdef(ra, slot->vreg);
    PolyX86Op xop = 0;
    const char *op_name = (def && x86_ins_op(def, &xop)) ? x86_op_name(xop)
                                                          : poly_op_name(def ? def->op : POLY_OP_NOOP);
    fprintf(
        stderr,
        "  spill[%d] v=%d disp=%d size=%d def_op=%s dtype=%s%s\n",
        i, slot->vreg, slot->offset, slot->size, op_name,
        def && def->dtype.name ? def->dtype.name : "?",
        slot->preserve_real_source ? " preserve_real_source" : ""
    );
  }
}

static int x86_candidate_pool(X86RegClass cls, int32_t *out, int cap) {
  int n_pool = 0;
  const int32_t *pool = x86_real_pool(cls, &n_pool);
  X86RegClass real_cls = x86_real_class_for_constraint(cls);
  int n = 0;
  for (int i = 0; i < n_pool && n < cap; i++)
    out[n++] = x86_tag_real(real_cls, pool[i]);
  return n;
}

static int x86_candidate_pool_excluding(
    X86RegClass cls,
    const int32_t *avoid,
    int n_avoid,
    int32_t *out,
    int cap
) {
  int32_t pool[64];
  int n_pool = x86_candidate_pool(cls, pool, (int)(sizeof(pool) / sizeof(pool[0])));
  int n = 0;
  for (int i = 0; i < n_pool && n < cap; i++) {
    if (x86_real_avoided(pool[i], avoid, n_avoid)) continue;
    out[n++] = pool[i];
  }
  return n;
}

static int32_t x86_fixed_constraint_real(int32_t vreg) {
  if (!x86_tag_is_virtual(vreg)) return 0;
  switch (x86_tag_class(vreg)) {
  case X86_REG_CLASS_FIXED_RAX:
    return x86_tag_real(X86_REG_CLASS_WGPR, X86_REG_RAX);
  case X86_REG_CLASS_FIXED_RDX:
    return x86_tag_real(X86_REG_CLASS_WGPR, X86_REG_RDX);
  default:
    return 0;
  }
}

static void x86_debug_regalloc_uop(const char *msg, int idx, PolyUOp *u, int32_t vreg) {
  if (!poly_debug_at_least(4)) return;
  PolyX86Op op = 0;
  const char *xop = (u && x86_ins_op(u, &op)) ? x86_op_name(op) : poly_op_name(u ? u->op : POLY_OP_NOOP);
  fprintf(
      stderr,
      "x86 regalloc: %s at uop %d op=%s vreg=%d class=%d real_req=%d\n",
      msg, idx, xop, vreg, (int)x86_tag_class(vreg), x86_fixed_constraint_real(vreg)
  );
  if (u) {
    char *s = poly_uop_str(u);
    fprintf(stderr, "  uop: %s\n", s ? s : "<uop>");
    free(s);
  }
}

static int x86_real_next_distance(X86RegAllocCtx *ra, int32_t real, int at) {
  int owner = x86_live_find_real(ra, real);
  if (owner < 0) return ra ? ra->n : INT32_MAX;
  int nu = x86_next_use(ra, ra->live[owner].vreg, at);
  return nu == INT32_MAX ? (ra ? ra->n : INT32_MAX) : nu - at;
}

static int32_t x86_tg_alloc_real_from_candidates(
    X86RegAllocCtx *ra,
    int32_t owner_vreg,
    const int32_t *cands,
    int n_cands,
    int at
) {
  if (!ra || !cands || n_cands <= 0) return 0;
  int best = -1;
  int best_dist = INT32_MIN;
  int32_t required_real = x86_fixed_constraint_real(owner_vreg);
  for (int i = 0; i < n_cands; i++) {
    if (!x86_tag_is_real(cands[i])) continue;
    if (owner_vreg != cands[i] && cands[i] != required_real &&
        x86_real_is_future_abi_source(ra, cands[i], at))
      continue;
    int dist = x86_real_next_distance(ra, cands[i], at);
    if (dist > best_dist) {
      best = i;
      best_dist = dist;
    }
  }
  if (best < 0) return 0;
  int live_idx = x86_live_find_real(ra, cands[best]);
  if (live_idx >= 0) {
    int32_t victim = ra->live[live_idx].vreg;
    int32_t real = ra->live[live_idx].real;
    if (x86_tag_is_reg_key(victim) && x86_next_use(ra, victim, at) != INT32_MAX &&
        x86_spill_slot_for_vreg(ra, victim) < 0)
      return 0;
    x86_live_remove_at(ra, live_idx);
    return real;
  }
  return cands[best];
}

static int32_t x86_tg_fill(
    X86RegAllocCtx *ra,
    int32_t vreg,
    int at,
    const int32_t *cands,
    int n_cands
) {
  if (!ra || !x86_tag_is_reg_key(vreg) || at < 0 || at >= ra->n) return 0;
  int slot_idx = x86_spill_slot_for_vreg(ra, vreg);
  if (slot_idx < 0) {
    x86_debug_regalloc_uop("fill spill slot missing", at, ra->uops[at], vreg);
    return 0;
  }
  int32_t fixed_real[1];
  if ((!cands || n_cands <= 0) && x86_tag_is_real(vreg)) {
    fixed_real[0] = vreg;
    cands = fixed_real;
    n_cands = 1;
  }
  int32_t local_cands[64];
  if (!cands || n_cands <= 0) {
    n_cands = x86_candidate_pool(
        x86_tag_class(vreg), local_cands, (int)(sizeof(local_cands) / sizeof(local_cands[0]))
    );
    cands = local_cands;
  }
  int32_t real = x86_tg_alloc_real_from_candidates(ra, vreg, cands, n_cands, at);
  if (!real) {
    x86_debug_regalloc_uop("fill real allocation failed", at, ra->uops[at], vreg);
    if (poly_debug_at_least(4)) {
      fprintf(stderr, "  fill candidates:");
      for (int ci = 0; ci < n_cands; ci++) fprintf(stderr, " %d", cands[ci]);
      fprintf(stderr, "\n");
    }
    return 0;
  }
  int32_t required_real = x86_fixed_constraint_real(vreg);
  if (required_real && real == required_real &&
      x86_preserve_future_abi_source_before(ra, at, real) != 0) {
    x86_debug_regalloc_uop("fill preserve future ABI source failed", at, ra->uops[at], vreg);
    return 0;
  }
  if (x86_loop_live_set_push(&ra->insert_before_by_idx[at], vreg, real) != 0) {
    x86_debug_regalloc_uop("fill insert record failed", at, ra->uops[at], vreg);
    return 0;
  }
  return real;
}

static bool x86_lr_has_pos_between(X86RegAllocCtx *ra, int32_t vreg, int start, int end) {
  int id = x86_virtual_id(vreg);
  if (!ra || id < 0 || id > ra->max_vreg || !ra->ranges) return false;
  X86LiveRange *lr = &ra->ranges[id];
  for (int i = 0; i < lr->n; i++)
    if (lr->pos[i] >= start && lr->pos[i] < end) return true;
  return false;
}

static int x86_tg_loop_live_cmp(const void *pa, const void *pb) {
  const X86LoopLiveCandidate *a = (const X86LoopLiveCandidate *)pa;
  const X86LoopLiveCandidate *b = (const X86LoopLiveCandidate *)pb;
  if (a->next != b->next) return (a->next > b->next) - (a->next < b->next);
  if (a->first != b->first) return (a->first > b->first) - (a->first < b->first);
  return (a->vreg > b->vreg) - (a->vreg < b->vreg);
}

static int x86_tg_record_loop_live_ins(X86RegAllocCtx *ra, int range_idx) {
  if (!ra || range_idx < 0 || range_idx >= ra->n) return -1;
  PolyUOp *range = ra->uops[range_idx];
  int32_t range_reg = x86_uop_reg(range);
  int rid = x86_virtual_id(range_reg);
  int end_idx = (rid >= 0 && rid <= ra->max_vreg) ? ra->last[rid] : -1;
  if (end_idx <= range_idx) {
    X86LoopLiveSet empty = {0};
    return x86_loop_live_stack_push(&ra->loop_live, empty);
  }

  X86LoopLiveSet set = {0};
  X86LoopLiveCandidate stack_cands[128];
  X86LoopLiveCandidate *cands = stack_cands;
  int n_cands = 0;
  int cap_cands = (int)(sizeof(stack_cands) / sizeof(stack_cands[0]));

  #define X86_TG_ADD_LOOP_CAND(v_) do { \
    int32_t _v = (v_); \
    if (!x86_tag_is_virtual(_v) || x86_loop_live_candidate_has(cands, n_cands, _v)) break; \
    if (!x86_lr_has_pos_between(ra, _v, range_idx, end_idx)) break; \
    if (n_cands >= cap_cands) { \
      int nc = cap_cands * 2; \
      X86LoopLiveCandidate *ns = (cands == stack_cands) \
          ? malloc((size_t)nc * sizeof(*ns)) \
          : realloc(cands, (size_t)nc * sizeof(*ns)); \
      if (!ns) { x86_loop_live_set_free(&set); return -1; } \
      if (cands == stack_cands) memcpy(ns, stack_cands, (size_t)n_cands * sizeof(*ns)); \
      cands = ns; \
      cap_cands = nc; \
    } \
    int id = x86_virtual_id(_v); \
    cands[n_cands++] = (X86LoopLiveCandidate){ \
        .vreg = _v, \
        .first = (id >= 0 && id <= ra->max_vreg) ? ra->first[id] : INT32_MAX, \
        .next = x86_loop_live_next_pos(ra, _v, range_idx, end_idx), \
    }; \
  } while (0)

  for (int i = 0; i < ra->n_live; i++) X86_TG_ADD_LOOP_CAND(ra->live[i].vreg);
  for (int i = 0; i < ra->n_spills; i++) X86_TG_ADD_LOOP_CAND(ra->spills[i].vreg);

  #undef X86_TG_ADD_LOOP_CAND

  qsort(cands, (size_t)n_cands, sizeof(*cands), x86_tg_loop_live_cmp);
  for (int ci = 0; ci < n_cands; ci++) {
    int32_t v = cands[ci].vreg;
    if (x86_loop_live_class_saturated(&set, v)) continue;
    if (x86_live_find(ra, v) < 0) {
      int32_t real = x86_tg_fill(ra, v, range_idx, NULL, 0);
      if (!real) continue;
      x86_live_set(ra, v, real, NULL);
    }
    int live_idx = x86_live_find(ra, v);
    if (live_idx >= 0 && x86_loop_live_set_push(&set, v, ra->live[live_idx].real) != 0) {
      if (cands != stack_cands) free(cands);
      x86_loop_live_set_free(&set);
      return -1;
    }
  }
  if (cands != stack_cands) free(cands);
  if (x86_loop_live_stack_push(&ra->loop_live, set) != 0) {
    x86_loop_live_set_free(&set);
    return -1;
  }
  return 0;
}

static int x86_tg_restore_loop_live_ins(X86RegAllocCtx *ra, int end_idx) {
  X86LoopLiveSet set = {0};
  if (!x86_loop_live_stack_pop(&ra->loop_live, &set)) return 0;
  int rc = 0;
  for (int i = 0; i < set.n; i++) {
    int live_idx = x86_live_find(ra, set.items[i].vreg);
    if (live_idx >= 0 && ra->live[live_idx].real == set.items[i].real) continue;
    int32_t cand[1] = {set.items[i].real};
    int32_t real = x86_tg_fill(ra, set.items[i].vreg, end_idx, cand, 1);
    if (!real) {
      rc = -1;
      break;
    }
    x86_live_set(ra, set.items[i].vreg, real, NULL);
  }
  x86_loop_live_set_free(&set);
  return rc;
}

static int x86_tg_allocate_registers(X86RegAllocCtx *ra) {
  for (int i = 0; i < ra->n; i++) {
    PolyUOp *u = ra->uops[i];
    if (!u || x86_regalloc_pseudo_op(u)) continue;

    if (u->op != POLY_OP_END) {
      for (int j = 0; j < u->n_src; j++) {
        int32_t v = x86_uop_reg(u->src[j]);
        if (!x86_tag_is_reg_key(v)) continue;
        if (x86_tag_is_real(v)) {
          int32_t real = v;
          int slot_idx = x86_spill_find(ra, v);
          bool fill_preserved_abi = slot_idx >= 0 && ra->spills[slot_idx].preserve_real_source &&
                                    x86_abi_real_wgpr_source(u->src[j], NULL);
          if (slot_idx >= 0 && (!ra->spills[slot_idx].preserve_real_source || fill_preserved_abi)) {
            int32_t cand[1] = {v};
            real = x86_tg_fill(ra, v, i, cand, 1);
            if (!real) {
              x86_debug_regalloc_uop("source real fill failed", i, u, v);
              return -1;
            }
          } else {
            int conflict = x86_live_find_real(ra, v);
            if (conflict >= 0) x86_live_remove_at(ra, conflict);
          }
          x86_live_set(ra, v, real, NULL);
          if (x86_loop_live_set_push(&ra->reals_by_idx[i], v, v) != 0) {
            x86_debug_regalloc_uop("source real record failed", i, u, v);
            return -1;
          }
          continue;
        }
        int32_t cands[64];
        int n_cands = 0;
        int32_t avoid[8];
        int n_avoid = 0;
        x86_fixed_source_avoids(u, j, avoid, &n_avoid, (int)(sizeof(avoid) / sizeof(avoid[0])));
        if (n_avoid > 0)
          n_cands = x86_candidate_pool_excluding(
              x86_tag_class(v), avoid, n_avoid, cands, (int)(sizeof(cands) / sizeof(cands[0]))
          );
        int live_idx = x86_live_find(ra, v);
        if (live_idx >= 0 && n_avoid > 0 &&
            x86_real_avoided(ra->live[live_idx].real, avoid, n_avoid)) {
          x86_live_remove_at(ra, live_idx);
          live_idx = -1;
        }
        if (live_idx < 0) {
          int32_t real = x86_tg_fill(ra, v, i, n_cands ? cands : NULL, n_cands);
          if (!real) {
            x86_debug_regalloc_uop("source virtual fill failed", i, u, v);
            return -1;
          }
          x86_live_set(ra, v, real, NULL);
          live_idx = x86_live_find(ra, v);
        }
        if (live_idx < 0) {
          x86_debug_regalloc_uop("source live missing after fill", i, u, v);
          return -1;
        }
        if (x86_loop_live_set_push(&ra->reals_by_idx[i], v, ra->live[live_idx].real) != 0) {
          x86_debug_regalloc_uop("source virtual record failed", i, u, v);
          return -1;
        }
      }
    }

    int32_t src_reals[64];
    int n_src_reals = 0;
    for (int j = 0; j < u->n_src; j++) {
      int32_t v = x86_uop_reg(u->src[j]);
      int32_t real = 0;
      if (x86_tag_is_reg_key(v) && x86_loop_live_set_get(&ra->reals_by_idx[i], v, &real))
        x86_avoid_push(src_reals, &n_src_reals, (int)(sizeof(src_reals) / sizeof(src_reals[0])), real);
    }

	    int32_t defs[16];
	    int n_defs = x86_uop_def_regs(u, defs, (int)(sizeof(defs) / sizeof(defs[0])));
	    for (int j = 0; j < n_defs; j++) {
      int32_t v = defs[j];
      if (x86_tag_is_real(v)) {
        PolyX86Op op = 0;
        bool is_real_source_define = x86_ins_op(u, &op) && op == POLY_X86_DEFINE;
        if (!is_real_source_define && x86_preserve_future_abi_source_before(ra, i, v) != 0) return -1;
        int conflict = x86_live_find_real(ra, v);
        if (conflict >= 0) x86_live_remove_at(ra, conflict);
        x86_live_set(ra, v, v, NULL);
	        if (x86_loop_live_set_push(&ra->reals_by_idx[i], v, v) != 0) return -1;
        continue;
      }
      if (!x86_tag_is_virtual(v)) continue;
      int32_t cands[80];
      int n_cands = 0;
      if (u->op == POLY_OP_INS && u->arg.kind == POLY_ARG_INT &&
          x86_is_two_address((PolyX86Op)u->arg.i) && j == 0 && u->n_src > 0) {
        int32_t src0 = x86_uop_reg(u->src[0]);
        int32_t src0_real = 0;
        if (x86_tag_is_virtual(src0) &&
            x86_loop_live_set_get(&ra->reals_by_idx[i], src0, &src0_real) &&
            x86_tag_class(src0_real) == x86_real_class_for_constraint(x86_tag_class(v)))
          cands[n_cands++] = src0_real;
        int32_t pool[64];
        int n_pool = x86_candidate_pool(x86_tag_class(v), pool, (int)(sizeof(pool) / sizeof(pool[0])));
        for (int pi = 0; pi < n_pool && n_cands < (int)(sizeof(cands) / sizeof(cands[0])); pi++) {
          if (x86_real_avoided(pool[pi], src_reals, n_src_reals)) continue;
          if (x86_real_avoided(pool[pi], cands, n_cands)) continue;
          cands[n_cands++] = pool[pi];
        }
      } else {
        n_cands = x86_candidate_pool(x86_tag_class(v), cands, (int)(sizeof(cands) / sizeof(cands[0])));
      }
      int alloc_at = u->op == POLY_OP_RANGE ? i : i + 1;
      int32_t real = x86_tg_alloc_real_from_candidates(ra, v, cands, n_cands, alloc_at);
      if (!real) {
        x86_debug_regalloc_uop("def allocation failed", i, u, v);
        if (poly_debug_at_least(4)) {
          fprintf(stderr, "  candidates:");
          for (int ci = 0; ci < n_cands; ci++) fprintf(stderr, " %d", cands[ci]);
          fprintf(stderr, "\n");
        }
        return -1;
      }
      int32_t required_real = x86_fixed_constraint_real(v);
      if (required_real && real == required_real &&
          x86_preserve_future_abi_source_before(ra, i, real) != 0) {
        x86_debug_regalloc_uop("def preserve future ABI source failed", i, u, v);
        return -1;
      }
      x86_live_set(ra, v, real, NULL);
      if (x86_loop_live_set_push(&ra->reals_by_idx[i], v, real) != 0) {
        x86_debug_regalloc_uop("def record failed", i, u, v);
        return -1;
      }
    }

    if (u->op == POLY_OP_BUFFER) {
      int64_t nbytes = x86_local_buffer_nbytes(u);
      if (nbytes <= 0 || nbytes > INT32_MAX) return -1;
      ra->local_offsets[i] = ra->stack_size;
      ra->stack_size += (int)nbytes;
    }

    if (u->op == POLY_OP_RANGE && x86_tg_record_loop_live_ins(ra, i) != 0) {
      x86_debug_regalloc_uop("loop live record failed", i, u, x86_uop_reg(u));
      return -1;
    }
    if (u->op == POLY_OP_END && x86_tg_restore_loop_live_ins(ra, i) != 0) {
      x86_debug_regalloc_uop("loop live restore failed", i, u, x86_uop_reg(u));
      return -1;
    }
  }
  return 0;
}

static PolyUOp **x86_regalloc_linear(PolyCtx *ctx, PolyUOp **lin, int n, int *n_out) {
  if (n_out) *n_out = 0;
  X86RegAllocCtx ra = {.ctx = ctx, .uops = lin, .n = n};
  X86UOpVec out = {0};
  X86UOpMap rewrite_map = {0};
  x86_collect_ranges(&ra);
  if (!ra.first || !ra.last || !ra.ranges) goto fail;
	  ra.reals_by_idx = calloc((size_t)n, sizeof(*ra.reals_by_idx));
	  ra.insert_before_by_idx = calloc((size_t)n, sizeof(*ra.insert_before_by_idx));
	  ra.spill_before_by_idx = calloc((size_t)n, sizeof(*ra.spill_before_by_idx));
	  ra.local_offsets = malloc((size_t)n * sizeof(*ra.local_offsets));
	  if (!ra.reals_by_idx || !ra.insert_before_by_idx || !ra.spill_before_by_idx || !ra.local_offsets)
	    goto fail;
  for (int i = 0; i < n; i++) ra.local_offsets[i] = -1;
  if (x86_tg_allocate_registers(&ra) != 0) goto fail;
  x86_debug_regalloc_summary(&ra);

  for (int i = 0; i < n; i++) {
    PolyUOp *u = lin[i];
    if (!u) continue;
    if (x86_regalloc_pseudo_op(u)) {
      if (x86_vec_push(&out, u) != 0) goto fail;
      continue;
    }

    PolyUOp *new_srcs_stack[64];
    PolyUOp **new_srcs = new_srcs_stack;
    if (u->n_src > 64) {
      new_srcs = malloc((size_t)u->n_src * sizeof(*new_srcs));
      if (!new_srcs) goto fail;
    }
    for (int j = 0; j < u->n_src; j++) {
      PolyUOp *s = u->src[j];
      PolyUOp *mapped = x86_map_get(&rewrite_map, s);
      int32_t v = x86_uop_reg(s);
      int32_t real = 0;
      if (x86_tag_is_reg_key(v) && x86_loop_live_set_get(&ra.reals_by_idx[i], v, &real)) {
        int slot_idx = x86_spill_find(&ra, v);
        bool fill_preserved_abi = slot_idx >= 0 && ra.spills[slot_idx].preserve_real_source &&
                                  x86_abi_real_wgpr_source(s, NULL);
        new_srcs[j] = (slot_idx >= 0 && (!ra.spills[slot_idx].preserve_real_source || fill_preserved_abi))
                          ? x86_fill_from_slot(ctx, &ra.spills[slot_idx], real)
                          : x86_alias_with_reg(ctx, mapped, real);
      } else {
        new_srcs[j] = mapped;
      }
    }

    int32_t defs[16];
    int n_defs = x86_uop_def_regs(u, defs, (int)(sizeof(defs) / sizeof(defs[0])));
    int32_t new_defs[16];
    for (int j = 0; j < n_defs; j++) {
      int32_t v = defs[j];
      int32_t real = 0;
      if (x86_tag_is_virtual(v)) {
        if (!x86_loop_live_set_get(&ra.reals_by_idx[i], v, &real)) {
          if (new_srcs != new_srcs_stack) free(new_srcs);
          goto fail;
        }
        new_defs[j] = real;
      } else if (x86_tag_is_real(v) && x86_loop_live_set_get(&ra.reals_by_idx[i], v, &real)) {
        new_defs[j] = real;
      } else {
        new_defs[j] = v;
      }
    }

    PolyUOp *nu = NULL;
    PolyX86Op xop = 0;
    if (u->op == POLY_OP_BUFFER) {
      if (n_defs != 1 || ra.local_offsets[i] < 0) {
        if (new_srcs != new_srcs_stack) free(new_srcs);
        goto fail;
      }
      PolyUOp *addr[4];
      x86_stack_address(ctx, ra.local_offsets[i], 1, addr);
      nu = x86_ins(ctx, POLY_X86_LEA, x86_u64(), addr, 4, new_defs[0]);
    } else if (x86_ins_op(u, &xop) && xop == POLY_X86_FRAME_INDEX) {
      if (n_defs != 1 || u->tag_arg.kind != POLY_ARG_INT) {
        if (new_srcs != new_srcs_stack) free(new_srcs);
        goto fail;
      }
      if (u->tag_arg.i < 0) {
        nu = x86_ins_ex(ctx, POLY_X86_FRAME_INDEX, x86_u64(), NULL, 0, new_defs, n_defs, u->tag_arg);
      } else {
        int size = (u->tag_arg.i > 0 && u->tag_arg.i <= INT32_MAX) ? (int)u->tag_arg.i : 1;
        int offset = x86_stack_alloc(&ra, size, size >= 16 ? 16 : size);
        PolyUOp *addr[4];
        x86_stack_address(ctx, offset, 1, addr);
        nu = x86_ins(ctx, POLY_X86_LEA, x86_u64(), addr, 4, new_defs[0]);
      }
    } else {
      nu = x86_clone_with_regs(ctx, u, new_srcs, u->n_src, new_defs, n_defs);
    }
    if (new_srcs != new_srcs_stack) free(new_srcs);
    if (x86_map_put(&rewrite_map, u, nu) != 0) goto fail;

    X86LoopLiveSet *spill_before = &ra.spill_before_by_idx[i];
    for (int bi = 0; bi < spill_before->n; bi++) {
      int slot_idx = x86_spill_find(&ra, spill_before->items[bi].vreg);
      if (slot_idx < 0) goto fail;
      PolyUOp *val = x86_define_reg(ctx, ra.spills[slot_idx].dtype, spill_before->items[bi].real);
      if (x86_vec_push(&out, x86_spill_to_slot(ctx, &ra.spills[slot_idx], val)) != 0)
        goto fail;
    }

    X86LoopLiveSet *before = &ra.insert_before_by_idx[i];
    for (int bi = 0; bi < before->n; bi++) {
      int slot_idx = x86_spill_find(&ra, before->items[bi].vreg);
      if (slot_idx < 0) goto fail;
      if (x86_vec_push(&out, x86_fill_from_slot(ctx, &ra.spills[slot_idx], before->items[bi].real)) != 0)
        goto fail;
    }
    if (x86_vec_push(&out, nu) != 0) goto fail;
    for (int j = 0; j < n_defs; j++) {
      int32_t v = defs[j];
      if (!x86_tag_is_reg_key(v)) continue;
      int slot_idx = x86_spill_find(&ra, v);
      if (slot_idx < 0) continue;
      if (x86_tag_is_real(v) && ra.spills[slot_idx].preserve_real_source) continue;
      PolyUOp *val = x86_alias_with_reg(ctx, nu, new_defs[j]);
      if (x86_vec_push(&out, x86_spill_to_slot(ctx, &ra.spills[slot_idx], val)) != 0)
        goto fail;
    }
  }

  int stack_size = ra.stack_size;
  bool has_caller_stack_args = false;
  for (int i = 0; i < out.n; i++) {
    PolyX86Op op = 0;
    if (x86_ins_op(out.items[i], &op) && op == POLY_X86_FRAME_INDEX &&
        out.items[i]->tag_arg.kind == POLY_ARG_INT && out.items[i]->tag_arg.i < 0) {
      has_caller_stack_args = true;
      break;
    }
  }
  if (stack_size > 0 || has_caller_stack_args) {
    X86UOpVec framed = {0};
    PolyUOp *rsp = x86_rsp(ctx);
    PolyUOp *imm = x86_const_i(ctx, POLY_INT32, stack_size);
    int32_t rsp_def = x86_tag_real(X86_REG_CLASS_WGPR, X86_REG_RSP);
    PolyUOp *sub_srcs[2] = {rsp, imm};
    PolyUOp *add_srcs[2] = {rsp, imm};
    if (stack_size > 0)
      x86_vec_push(&framed, x86_ins(ctx, POLY_X86_SUBi, x86_u64(), sub_srcs, 2, rsp_def));
    for (int i = 0; i < out.n; i++) {
      if (i == out.n - 1 && stack_size > 0)
        x86_vec_push(&framed, x86_ins(ctx, POLY_X86_ADDi, x86_u64(), add_srcs, 2, rsp_def));
      PolyX86Op op = 0;
      if (x86_ins_op(out.items[i], &op) && op == POLY_X86_FRAME_INDEX &&
          out.items[i]->tag_arg.kind == POLY_ARG_INT && out.items[i]->tag_arg.i < 0) {
        int32_t def = x86_uop_reg(out.items[i]);
        int64_t caller_disp = -out.items[i]->tag_arg.i;
        int64_t off64 = (int64_t)stack_size + caller_disp;
        if (off64 < INT32_MIN || off64 > INT32_MAX) goto fail;
        PolyUOp *addr[4];
        x86_stack_address(ctx, (int)off64, 1, addr);
        x86_vec_push(&framed, x86_ins(ctx, POLY_X86_LEA, x86_u64(), addr, 4, def));
      } else {
        x86_vec_push(&framed, out.items[i]);
      }
    }
    free(out.items);
    out = framed;
  }

  free(ra.first);
  free(ra.last);
  x86_live_ranges_free(&ra);
  x86_loop_live_sets_free(ra.reals_by_idx, ra.n);
  x86_loop_live_sets_free(ra.insert_before_by_idx, ra.n);
  x86_loop_live_sets_free(ra.spill_before_by_idx, ra.n);
  free(ra.local_offsets);
  free(ra.spills);
  x86_loop_live_stack_free(&ra.loop_live);
  x86_map_free(&rewrite_map);
  if (n_out) *n_out = out.n;
  return out.items;

fail:
  free(out.items);
  free(ra.first);
  free(ra.last);
  x86_live_ranges_free(&ra);
  x86_loop_live_sets_free(ra.reals_by_idx, ra.n);
  x86_loop_live_sets_free(ra.insert_before_by_idx, ra.n);
  x86_loop_live_sets_free(ra.spill_before_by_idx, ra.n);
  free(ra.local_offsets);
  free(ra.spills);
  x86_loop_live_stack_free(&ra.loop_live);
  x86_map_free(&rewrite_map);
  if (n_out) *n_out = 0;
  return NULL;
}

typedef struct {
  PolyUOp *range;
  int32_t acc;
  char loop_label[64];
  char out_label[64];
} X86LoopLabel;

typedef struct {
  X86LoopLabel *items;
  int n;
  int cap;
} X86LoopLabels;

static bool x86_range_args_same_loop(PolyArg a, PolyArg b) {
  if (poly_arg_eq(a, b)) return true;
  if (!poly_arg_is_range(a) || !poly_arg_is_range(b)) return false;
  if (poly_range_axis_id(a) != poly_range_axis_id(b)) return false;
  if (poly_range_axis_type(a) != poly_range_axis_type(b)) return false;
  int na = poly_range_n_extra(a), nb = poly_range_n_extra(b);
  if (na != nb) return false;
  int64_t *ea = poly_range_extra(a), *eb = poly_range_extra(b);
  for (int i = 0; i < na; i++)
    if (ea[i] != eb[i]) return false;
  return true;
}

static X86LoopLabel *x86_loop_label_for(X86LoopLabels *ls, PolyUOp *range) {
  if (!ls || !range) return NULL;
  for (int i = 0; i < ls->n; i++) {
    X86LoopLabel *it = &ls->items[i];
    if (it->range == range) return it;
    if (it->range && range->op == POLY_OP_RANGE && it->range->op == POLY_OP_RANGE &&
        x86_range_args_same_loop(it->range->arg, range->arg))
      return it;
  }
  return NULL;
}

static X86LoopLabel *x86_loop_label_add(X86LoopLabels *ls, PolyUOp *range, int ordinal) {
  X86LoopLabel *old = x86_loop_label_for(ls, range);
  if (old) return old;
  if (ls->n >= ls->cap) {
    int nc = ls->cap ? ls->cap * 2 : 32;
    X86LoopLabel *ni = realloc(ls->items, (size_t)nc * sizeof(*ni));
    if (!ni) return NULL;
    ls->items = ni;
    ls->cap = nc;
  }
  X86LoopLabel *it = &ls->items[ls->n++];
  it->range = range;
  it->acc = x86_uop_reg(range);
  snprintf(it->loop_label, sizeof(it->loop_label), ".LOOP_%d", ordinal);
  snprintf(it->out_label, sizeof(it->out_label), ".LOOP_OUT_%d", ordinal);
  return it;
}

static PolyUOp *x86_copy_to_reg(PolyCtx *ctx, PolyUOp *src, int32_t real) {
  PolyDType dt = src->dtype;
  if (src->op == POLY_OP_CONST && (src->arg.kind == POLY_ARG_INT || src->arg.kind == POLY_ARG_BOOL) &&
      dt.count <= 1 && x86_is_int_dtype(dt)) {
    int64_t v = src->arg.kind == POLY_ARG_BOOL ? (src->arg.b ? 1 : 0) : src->arg.i;
    PolyUOp *imm = x86_const_i(ctx, dt, v);
    PolyUOp *srcs[1] = {imm};
    PolyX86Op op = x86_value_size(dt, false) == 8 ? POLY_X86_MOVABS : POLY_X86_MOVi;
    return x86_ins(ctx, op, dt, srcs, 1, real);
  }
  PolyX86Op op = x86_mov_op_for_dtype(dt, false);
  PolyUOp *srcs[1] = {src};
  return x86_ins(ctx, op, dt, srcs, 1, real);
}

static int x86_post_regalloc_one(
    PolyCtx *ctx,
    PolyUOp *u,
    int ordinal,
    X86LoopLabels *loops,
    X86UOpVec *out
) {
  if (!u) return -1;
  if (u->op == POLY_OP_RANGE) {
    if (u->n_src < 1) return -1;
    X86LoopLabel *lbl = x86_loop_label_add(loops, u, ordinal);
    if (!lbl) return -1;
    if (poly_debug_at_least(5))
      fprintf(stderr, "x86 post RANGE ordinal=%d acc=%d label=%s\n", ordinal, x86_uop_reg(u), lbl->loop_label);
    int32_t acc = x86_uop_reg(u);
    if (!acc) return -1;
    PolyUOp *zero = x86_const_i(ctx, u->dtype, 0);
    PolyUOp *one = x86_const_i(ctx, u->dtype, 1);
    PolyUOp *mov_srcs[1] = {zero};
    PolyUOp *cmp_srcs[2] = {x86_alias_with_reg(ctx, u, acc), u->src[0]};
    PolyUOp *cmp = x86_ins_nodef(
        ctx, u->src[0]->op == POLY_OP_CONST ? POLY_X86_CMPi : POLY_X86_CMP, POLY_VOID, cmp_srcs, 2
    );
    PolyUOp *jmp_srcs[1] = {cmp};
    (void)one;
    if (x86_vec_push(out, x86_ins(ctx, POLY_X86_MOVi, u->dtype, mov_srcs, 1, acc)) != 0) return -1;
    if (x86_vec_push(out, x86_ins_label(ctx, lbl->loop_label)) != 0) return -1;
    if (x86_vec_push(out, cmp) != 0) return -1;
    if (x86_vec_push(out, x86_ins_jump(ctx, POLY_X86_JGE, jmp_srcs, 1, lbl->out_label)) != 0) return -1;
    return 0;
  }
  if (u->op == POLY_OP_END) {
    PolyUOp *range = NULL;
    for (int i = 0; i < u->n_src; i++) {
      if (u->src[i] && u->src[i]->op == POLY_OP_RANGE) {
        range = u->src[i];
        break;
      }
    }
    if (!range) return 0;
    X86LoopLabel *lbl = x86_loop_label_for(loops, range);
    if (!lbl) {
      if (poly_debug_at_least(4)) {
        fprintf(stderr, "x86 post END missing loop label for range tag=%d\n", x86_uop_reg(range));
      }
      return -1;
    }
    int32_t acc = lbl->acc;
    if (!acc) return -1;
    if (poly_debug_at_least(5))
      fprintf(
          stderr,
          "x86 post END range_tag=%d selected_acc=%d label=%s out=%s\n",
          x86_uop_reg(range), acc, lbl->loop_label, lbl->out_label
      );
    PolyUOp *one = x86_const_i(ctx, range->dtype, 1);
    PolyUOp *add_srcs[1] = {one};
    if (x86_vec_push(out, x86_ins(ctx, POLY_X86_ADDi, range->dtype, add_srcs, 1, acc)) != 0) return -1;
    if (x86_vec_push(out, x86_ins_jump(ctx, POLY_X86_JMP, NULL, 0, lbl->loop_label)) != 0) return -1;
    if (x86_vec_push(out, x86_ins_label(ctx, lbl->out_label)) != 0) return -1;
    return 0;
  }
  if (u->op != POLY_OP_INS || u->arg.kind != POLY_ARG_INT) {
    return x86_vec_push(out, u);
  }
  PolyX86Op op = (PolyX86Op)u->arg.i;
  if (!x86_is_two_address(op)) return x86_vec_push(out, u);
  if (u->n_src < 1) return x86_vec_push(out, u);
  int32_t dst = x86_uop_reg(u);
  int32_t src0 = x86_uop_reg(u->src[0]);
  if (!dst || !src0) return -1;
  if (dst != src0) {
    PolyUOp *copy = x86_copy_to_reg(ctx, u->src[0], dst);
    if (x86_vec_push(out, copy) != 0) return -1;
  }
  PolyUOp *new_srcs[16];
  int ns = u->n_src - 1;
  if (ns > 16) return -1;
  for (int i = 0; i < ns; i++)
    new_srcs[i] = u->src[i + 1];
  PolyUOp *nu = x86_clone_with_regs(ctx, u, new_srcs, ns, (int32_t[]){dst}, 1);
  return x86_vec_push(out, nu);
}

static PolyUOp **x86_post_regalloc_linear(PolyCtx *ctx, PolyUOp **lin, int n, int *n_out) {
  if (n_out) *n_out = 0;
  X86UOpVec out = {0};
  X86LoopLabels loops = {0};
  for (int i = 0; i < n; i++) {
    if (x86_post_regalloc_one(ctx, lin[i], i, &loops, &out) != 0) {
      if (poly_debug_at_least(4)) {
        char *s = poly_uop_str(lin[i]);
        fprintf(stderr, "x86 post-regalloc failed at %d: %s\n", i, s ? s : "<uop>");
        free(s);
        for (int si = 0; si < lin[i]->n_src; si++) {
          char *ss = poly_uop_str(lin[i]->src[si]);
          fprintf(stderr, "  post-src[%d] tag=%d tag_arg=%d %s\n",
                  si, lin[i]->src[si] ? lin[i]->src[si]->tag : 0,
                  lin[i]->src[si] ? lin[i]->src[si]->tag_arg.kind : 0,
                  ss ? ss : "<uop>");
          free(ss);
        }
      }
      free(out.items);
      free(loops.items);
      return NULL;
    }
  }
  free(loops.items);
  if (n_out) *n_out = out.n;
  return out.items;
}

static int x86_emit_encode(
    X86Buf *b,
    PolyUOp *x,
    int opc,
    int fixed_reg,
    int pp,
    int sel,
    int we,
    PolyUOp *reg_uop,
    PolyUOp *rm_uop,
    PolyUOp *idx_uop,
    PolyUOp *disp_uop,
    PolyUOp *sz_uop,
    PolyUOp *vvvv_uop,
    PolyUOp *imm_uop
) {
  int reg = fixed_reg >= 0 ? fixed_reg : x86_uop_reg_index(reg_uop);
  int rm = x86_uop_reg_index(rm_uop);
  int idx = (idx_uop && idx_uop->op != POLY_OP_NOOP) ? x86_uop_reg_index(idx_uop) : 4;
  if (reg < 0 || rm < 0 || idx < 0) return -1;

  int rm_sz = (sz_uop && sz_uop->op == POLY_OP_CONST) ? (int)sz_uop->arg.i
                                                       : x86_value_size(rm_uop->dtype, false);
  int reg_sz = reg_uop ? x86_value_size(reg_uop->dtype, false) : 0;
  int sz = reg_sz ? reg_sz : rm_sz;

  if (sel) {
    int vvvv = vvvv_uop ? x86_uop_reg_index(vvvv_uop) : 0;
    if (vvvv < 0) return -1;
    int l = (reg_sz > 16 || rm_sz > 16) ? 1 : 0;
    int r = reg >> 3, ix = idx >> 3, br = rm >> 3;
    if (sel == 1 && ix == 0 && br == 0 && we == 0)
      emit_vex2(b, (~r) & 1, vvvv, l, pp);
    else
      emit_vex3(b, (~r) & 1, (~ix) & 1, (~br) & 1, sel, we, vvvv, l, pp);
  } else {
    if (sz == 2) xb_byte(b, 0x66);
    int r = reg >> 3, ix = idx >> 3, br = rm >> 3;
    bool w = (sz == 8);
    if (w || r || ix || br || (reg_sz == 1 && (reg >> 2)) || (rm_sz == 1 && (rm >> 2)))
      emit_rex(b, w ? 1 : 0, r, ix, br);
    if ((rm_sz == 1 || reg_sz == 1) && x && x->arg.kind == POLY_ARG_INT &&
        !x86_readflags((PolyX86Op)x->arg.i) && (PolyX86Op)x->arg.i != POLY_X86_LEA)
      opc -= 1;
  }

  if (opc > 0xFF) xb_byte(b, (uint8_t)(opc >> 8));
  xb_byte(b, (uint8_t)(opc & 0xFF));

  bool has_mem = disp_uop != NULL;
  int64_t disp = 0;
  if (has_mem && !x86_uop_const_i64(disp_uop, &disp)) return -1;
  int reg3 = reg & 7, rm3 = rm & 7, idx3 = idx & 7;
  int mod = 3;
  if (has_mem) {
    if (disp != 0 || rm3 == 5)
      mod = x86_dtype_itemsize(disp_uop->dtype) == 1 ? 1 : 2;
    else
      mod = 0;
  }
  bool no_index = idx3 == 4 && ((idx >> 3) == 0);
  /* RSP/R12 as a memory base must be encoded through a SIB byte even when
   * there is no index register. */
  int enc_rm = (has_mem && (!no_index || rm3 == 4)) ? 4 : rm3;
  emit_modrm(b, mod, reg3, enc_rm);
  if (enc_rm == 4 && mod != 3) {
    int ss = scale_to_ss(no_index ? 1 : rm_sz);
    xb_byte(b, (uint8_t)((ss << 6) | (idx3 << 3) | rm3));
  }
  if (mod == 1)
    xb_byte(b, (uint8_t)(int8_t)disp);
  else if (mod == 2)
    xb_i32(b, (int32_t)disp);

  if (imm_uop) {
    int64_t imm = 0;
    if (x86_uop_const_i64(imm_uop, &imm)) {
      int isz = x86_dtype_itemsize(imm_uop->dtype);
      if (isz == 1)
        xb_byte(b, (uint8_t)imm);
      else if (isz == 2) {
        xb_byte(b, (uint8_t)imm);
        xb_byte(b, (uint8_t)(imm >> 8));
      } else if (isz == 4)
        xb_i32(b, (int32_t)imm);
      else
        xb_i64(b, imm);
    } else {
      int rimm = x86_uop_reg_index(imm_uop);
      if (rimm < 0) return -1;
      xb_byte(b, (uint8_t)((rimm & 0xF) << 4));
    }
  }
  return 0;
}

static int x86_encode_op(
    X86Buf *b,
    PolyUOp *x,
    PolyX86Op op,
    int opc,
    int fixed_reg,
    int pp,
    int sel,
    int we
) {
  PolyUOp *address[4] = {NULL, NULL, NULL, NULL};
  PolyUOp *rest[8] = {NULL};
  int n_rest = 0;
  if (x86_writemem(op)) {
    if (x->n_src > 4) {
      for (int i = 0; i < 4; i++)
        address[i] = x->src[i];
      for (int i = 4; i < x->n_src && n_rest < 8; i++)
        rest[n_rest++] = x->src[i];
    } else {
      address[0] = x;
      for (int i = 0; i < x->n_src && n_rest < 8; i++)
        rest[n_rest++] = x->src[i];
    }
    if (n_rest < 1) return -1;
    if (fixed_reg < 0)
      return x86_emit_encode(
          b, x, opc, -1, pp, sel, we, rest[0], address[0], address[1], address[2],
          address[3], NULL, n_rest > 1 ? rest[1] : NULL
      );
    return x86_emit_encode(
        b, x, opc, fixed_reg, pp, sel, we, NULL, address[0], address[1], address[2],
        address[3], NULL, rest[0]
    );
  }

  if (x86_rm1st(op)) {
    if (x->n_src > 3) {
      for (int i = 0; i < 4; i++)
        address[i] = x->src[i];
      for (int i = 4; i < x->n_src && n_rest < 8; i++)
        rest[n_rest++] = x->src[i];
    } else {
      if (x->n_src < 1) return -1;
      address[0] = x->src[0];
      for (int i = 1; i < x->n_src && n_rest < 8; i++)
        rest[n_rest++] = x->src[i];
    }
    PolyUOp *imm = (n_rest > 0 && rest[0]->op == POLY_OP_CONST) ? rest[0] : NULL;
    if (fixed_reg < 0)
      return x86_emit_encode(
          b, x, opc, -1, pp, sel, we, x, address[0], address[1], address[2], address[3],
          NULL, imm
      );
    return x86_emit_encode(
        b, x, opc, fixed_reg, pp, sel, we, NULL, address[0], address[1], address[2],
        address[3], sel ? x : NULL, imm
    );
  }

  if (x86_readmem3rd(op) && x86_is_two_address(op)) {
    if (x->n_src > 4) {
      address[0] = x->src[1];
      address[1] = x->src[2];
      address[2] = x->src[3];
      address[3] = x->src[4];
      rest[0] = x->src[0];
      n_rest = 1;
      for (int i = 5; i < x->n_src && n_rest < 8; i++)
        rest[n_rest++] = x->src[i];
    } else {
      if (x->n_src < 2) return -1;
      address[0] = x->src[1];
      rest[0] = x->src[0];
      n_rest = 1;
      for (int i = 2; i < x->n_src && n_rest < 8; i++)
        rest[n_rest++] = x->src[i];
    }
    return x86_emit_encode(
        b, x, opc, fixed_reg, pp, sel, we, x, address[0], address[1], address[2],
        address[3], rest[0], n_rest > 1 ? rest[1] : NULL
    );
  }

  if (x86_readmem3rd(op) && x->n_src >= 3) {
    if (x->n_src > 5) {
      address[0] = x->src[2];
      address[1] = x->src[3];
      address[2] = x->src[4];
      address[3] = x->src[5];
      rest[0] = x->src[0];
      rest[1] = x->src[1];
      n_rest = 2;
      for (int i = 6; i < x->n_src && n_rest < 8; i++)
        rest[n_rest++] = x->src[i];
    } else {
      address[0] = x->src[2];
      rest[0] = x->src[0];
      rest[1] = x->src[1];
      n_rest = 2;
    }
    return x86_emit_encode(
        b, x, opc, fixed_reg, pp, sel, we, x, address[0], address[1], address[2],
        address[3], rest[0], rest[1]
    );
  }

  if (x86_readmem2nd(op)) {
    if (x->n_src > 4) {
      address[0] = x->src[1];
      address[1] = x->src[2];
      address[2] = x->src[3];
      address[3] = x->src[4];
      rest[0] = x->src[0];
      n_rest = 1;
      for (int i = 5; i < x->n_src && n_rest < 8; i++)
        rest[n_rest++] = x->src[i];
    } else {
      if (x->n_src < 2) return -1;
      address[0] = x->src[1];
      rest[0] = x->src[0];
      n_rest = 1;
      for (int i = 2; i < x->n_src && n_rest < 8; i++)
        rest[n_rest++] = x->src[i];
    }
    if (poly_dtype_eq(x->dtype, POLY_VOID))
      return x86_emit_encode(
          b, x, opc, fixed_reg, pp, sel, we, rest[0], address[0], address[1], address[2],
          address[3], NULL, NULL
      );
    return x86_emit_encode(
        b, x, opc, fixed_reg, pp, sel, we, x, address[0], address[1], address[2],
        address[3], rest[0], n_rest > 1 ? rest[1] : NULL
    );
  }
  return -1;
}

static int x86_encode_setcc(X86Buf *b, PolyUOp *u, int opc) {
  int rm = x86_uop_reg_index(u);
  if (rm < 0) return -1;
  int reg = 0;
  int rm_sz = 1;
  int reg_sz = 0;
  int r = 0, ix = 0, br = rm >> 3;
  if (br || ((rm_sz == 1 || reg_sz == 1) && (rm >> 2))) emit_rex(b, 0, r, ix, br);
  xb_byte(b, (uint8_t)(opc >> 8));
  xb_byte(b, (uint8_t)(opc & 0xFF));
  emit_modrm(b, 3, reg, rm);
  return 0;
}

static int x86_encode_instruction(X86Buf *b, PolyUOp *u, PolyX86Op op) {
  switch (op) {
  case POLY_X86_LABEL:
  case POLY_X86_DEFINE:
    return 0;
  case POLY_X86_RET:
    xb_byte(b, 0xC3);
    return 0;
  case POLY_X86_MOVABS: {
    int reg = x86_uop_reg_index(u);
    int64_t imm = 0;
    if (reg < 0 || u->n_src < 1 || !x86_uop_const_i64(u->src[0], &imm)) return -1;
    emit_rex(b, 1, 0, 0, reg >> 3);
    xb_byte(b, (uint8_t)(0xB8 + (reg & 7)));
    xb_i64(b, imm);
    return 0;
  }
  case POLY_X86_MOV: return x86_encode_op(b, u, op, 0x8B, -1, 0, 0, 0);
  case POLY_X86_MOVi: return x86_encode_op(b, u, op, 0xC7, 0, 0, 0, 0);
  case POLY_X86_MOVm: return x86_encode_op(b, u, op, 0x89, -1, 0, 0, 0);
  case POLY_X86_LEA: return x86_encode_op(b, u, op, 0x8D, -1, 0, 0, 0);
  case POLY_X86_VMOVSS: return x86_encode_op(b, u, op, 0x10, -1, 2, 1, 0);
  case POLY_X86_VMOVSSm: return x86_encode_op(b, u, op, 0x11, -1, 2, 1, 0);
  case POLY_X86_VMOVSD: return x86_encode_op(b, u, op, 0x10, -1, 3, 1, 0);
  case POLY_X86_VMOVSDm: return x86_encode_op(b, u, op, 0x11, -1, 3, 1, 0);
  case POLY_X86_VMOVUPS: return x86_encode_op(b, u, op, 0x10, -1, 0, 1, 0);
  case POLY_X86_VMOVUPSm: return x86_encode_op(b, u, op, 0x11, -1, 0, 1, 0);
  case POLY_X86_VMOVD: return x86_encode_op(b, u, op, 0x6E, -1, 1, 1, 0);
  case POLY_X86_VMOVQ: return x86_encode_op(b, u, op, 0x6E, -1, 1, 1, 1);
  case POLY_X86_VMOVDm: return x86_encode_op(b, u, op, 0x7E, -1, 1, 1, 0);
  case POLY_X86_VMOVQm: return x86_encode_op(b, u, op, 0x7E, -1, 1, 1, 1);

  case POLY_X86_MOVZX: return x86_encode_op(b, u, op, 0x0FB7, -1, 0, 0, 0);
  case POLY_X86_MOVSX: return x86_encode_op(b, u, op, 0x0FBF, -1, 0, 0, 0);
  case POLY_X86_MOVSXD: return x86_encode_op(b, u, op, 0x63, -1, 0, 0, 0);
  case POLY_X86_VPMOVZXBW: return x86_encode_op(b, u, op, 0x30, -1, 1, 2, 0);
  case POLY_X86_VPMOVZXBD: return x86_encode_op(b, u, op, 0x31, -1, 1, 2, 0);
  case POLY_X86_VPMOVZXBQ: return x86_encode_op(b, u, op, 0x32, -1, 1, 2, 0);
  case POLY_X86_VPMOVZXWD: return x86_encode_op(b, u, op, 0x33, -1, 1, 2, 0);
  case POLY_X86_VPMOVZXWQ: return x86_encode_op(b, u, op, 0x34, -1, 1, 2, 0);
  case POLY_X86_VPMOVZXDQ: return x86_encode_op(b, u, op, 0x35, -1, 1, 2, 0);
  case POLY_X86_VPMOVSXBW: return x86_encode_op(b, u, op, 0x20, -1, 1, 2, 0);
  case POLY_X86_VPMOVSXBD: return x86_encode_op(b, u, op, 0x21, -1, 1, 2, 0);
  case POLY_X86_VPMOVSXBQ: return x86_encode_op(b, u, op, 0x22, -1, 1, 2, 0);
  case POLY_X86_VPMOVSXWD: return x86_encode_op(b, u, op, 0x23, -1, 1, 2, 0);
  case POLY_X86_VPMOVSXWQ: return x86_encode_op(b, u, op, 0x24, -1, 1, 2, 0);
  case POLY_X86_VPMOVSXDQ: return x86_encode_op(b, u, op, 0x25, -1, 1, 2, 0);
  case POLY_X86_VCVTSS2SD: return x86_encode_op(b, u, op, 0x5A, -1, 2, 1, 0);
  case POLY_X86_VCVTSD2SS: return x86_encode_op(b, u, op, 0x5A, -1, 3, 1, 0);
  case POLY_X86_VCVTPH2PS: return x86_encode_op(b, u, op, 0x13, -1, 1, 2, 0);
  case POLY_X86_VCVTPS2PH: return x86_encode_op(b, u, op, 0x1D, -1, 1, 3, 0);
  case POLY_X86_VCVTDQ2PS: return x86_encode_op(b, u, op, 0x5B, -1, 0, 1, 0);
  case POLY_X86_VCVTDQ2PD: return x86_encode_op(b, u, op, 0xE6, -1, 2, 1, 0);
  case POLY_X86_VCVTPS2PD: return x86_encode_op(b, u, op, 0x5A, -1, 0, 1, 0);
  case POLY_X86_VCVTPD2PS: return x86_encode_op(b, u, op, 0x5A, -1, 1, 1, 0);
  case POLY_X86_VCVTTPS2DQ: return x86_encode_op(b, u, op, 0x5B, -1, 2, 1, 0);
  case POLY_X86_VCVTTPD2DQ: return x86_encode_op(b, u, op, 0xE6, -1, 1, 1, 0);
  case POLY_X86_VCVTSI2SS:
    return x86_encode_op(
        b, u, op, 0x2A, -1, 2, 1,
        (u->n_src > 4 && u->src[4]->op == POLY_OP_CONST) ? (u->src[4]->arg.i == 8)
                                                          : (u->n_src > 1 && x86_dtype_itemsize(u->src[1]->dtype) == 8)
    );
  case POLY_X86_VCVTSI2SD:
    return x86_encode_op(
        b, u, op, 0x2A, -1, 3, 1,
        (u->n_src > 4 && u->src[4]->op == POLY_OP_CONST) ? (u->src[4]->arg.i == 8)
                                                          : (u->n_src > 1 && x86_dtype_itemsize(u->src[1]->dtype) == 8)
    );
  case POLY_X86_VCVTTSS2SI:
    return x86_encode_op(b, u, op, 0x2C, -1, 2, 1, x86_dtype_itemsize(u->dtype) == 8);
  case POLY_X86_VCVTTSD2SI:
    return x86_encode_op(b, u, op, 0x2C, -1, 3, 1, x86_dtype_itemsize(u->dtype) == 8);

  case POLY_X86_IDIV: return x86_encode_op(b, u, op, 0xF7, 7, 0, 0, 0);
  case POLY_X86_DIV: return x86_encode_op(b, u, op, 0xF7, 6, 0, 0, 0);
  case POLY_X86_SHLi: return x86_encode_op(b, u, op, 0xC1, 4, 0, 0, 0);
  case POLY_X86_SHRi: return x86_encode_op(b, u, op, 0xC1, 5, 0, 0, 0);
  case POLY_X86_SARi: return x86_encode_op(b, u, op, 0xC1, 7, 0, 0, 0);
  case POLY_X86_ADD: return x86_encode_op(b, u, op, 0x03, -1, 0, 0, 0);
  case POLY_X86_ADDi: return x86_encode_op(b, u, op, 0x81, 0, 0, 0, 0);
  case POLY_X86_SUB: return x86_encode_op(b, u, op, 0x2B, -1, 0, 0, 0);
  case POLY_X86_SUBi: return x86_encode_op(b, u, op, 0x81, 5, 0, 0, 0);
  case POLY_X86_AND: return x86_encode_op(b, u, op, 0x23, -1, 0, 0, 0);
  case POLY_X86_ANDi: return x86_encode_op(b, u, op, 0x81, 4, 0, 0, 0);
  case POLY_X86_XOR: return x86_encode_op(b, u, op, 0x33, -1, 0, 0, 0);
  case POLY_X86_XORi: return x86_encode_op(b, u, op, 0x81, 6, 0, 0, 0);
  case POLY_X86_OR: return x86_encode_op(b, u, op, 0x0B, -1, 0, 0, 0);
  case POLY_X86_ORi: return x86_encode_op(b, u, op, 0x81, 1, 0, 0, 0);
  case POLY_X86_CMP: return x86_encode_op(b, u, op, 0x3B, -1, 0, 0, 0);
  case POLY_X86_CMPi: return x86_encode_op(b, u, op, 0x81, 7, 0, 0, 0);
  case POLY_X86_IMUL: return x86_encode_op(b, u, op, 0x0FAF, -1, 0, 0, 0);
  case POLY_X86_IMULi: return x86_encode_op(b, u, op, 0x69, -1, 0, 0, 0);
  case POLY_X86_SETB: return x86_encode_setcc(b, u, 0x0F92);
  case POLY_X86_SETL: return x86_encode_setcc(b, u, 0x0F9C);
  case POLY_X86_SETE: return x86_encode_setcc(b, u, 0x0F94);
  case POLY_X86_SETNE: return x86_encode_setcc(b, u, 0x0F95);

  case POLY_X86_VPAND: return x86_encode_op(b, u, op, 0xDB, -1, 1, 1, 0);
  case POLY_X86_VPXOR: return x86_encode_op(b, u, op, 0xEF, -1, 1, 1, 0);
  case POLY_X86_VPOR: return x86_encode_op(b, u, op, 0xEB, -1, 1, 1, 0);
  case POLY_X86_VSQRTSS: return x86_encode_op(b, u, op, 0x51, -1, 2, 1, 0);
  case POLY_X86_VSQRTPS: return x86_encode_op(b, u, op, 0x51, -1, 0, 1, 0);
  case POLY_X86_VSQRTSD: return x86_encode_op(b, u, op, 0x51, -1, 3, 1, 0);
  case POLY_X86_VSQRTPD: return x86_encode_op(b, u, op, 0x51, -1, 1, 1, 0);
  case POLY_X86_VROUNDSS: return x86_encode_op(b, u, op, 0x0A, -1, 1, 3, 0);
  case POLY_X86_VROUNDPS: return x86_encode_op(b, u, op, 0x08, -1, 1, 3, 0);
  case POLY_X86_VROUNDSD: return x86_encode_op(b, u, op, 0x0B, -1, 1, 3, 0);
  case POLY_X86_VROUNDPD: return x86_encode_op(b, u, op, 0x09, -1, 1, 3, 0);
  case POLY_X86_VPSLLVD: return x86_encode_op(b, u, op, 0x47, -1, 1, 2, 0);
  case POLY_X86_VPSLLVQ: return x86_encode_op(b, u, op, 0x47, -1, 1, 2, 1);
  case POLY_X86_VPSRLVD: return x86_encode_op(b, u, op, 0x45, -1, 1, 2, 0);
  case POLY_X86_VPSRLVQ: return x86_encode_op(b, u, op, 0x45, -1, 1, 2, 1);
  case POLY_X86_VPSRAVD: return x86_encode_op(b, u, op, 0x46, -1, 1, 2, 0);
  case POLY_X86_VPCMPGTB: return x86_encode_op(b, u, op, 0x64, -1, 1, 1, 0);
  case POLY_X86_VPCMPGTW: return x86_encode_op(b, u, op, 0x65, -1, 1, 1, 0);
  case POLY_X86_VPCMPGTD: return x86_encode_op(b, u, op, 0x66, -1, 1, 1, 0);
  case POLY_X86_VPCMPGTQ: return x86_encode_op(b, u, op, 0x37, -1, 1, 2, 0);
  case POLY_X86_VPCMPEQB: return x86_encode_op(b, u, op, 0x74, -1, 1, 1, 0);
  case POLY_X86_VPCMPEQW: return x86_encode_op(b, u, op, 0x75, -1, 1, 1, 0);
  case POLY_X86_VPCMPEQD: return x86_encode_op(b, u, op, 0x76, -1, 1, 1, 0);
  case POLY_X86_VPCMPEQQ: return x86_encode_op(b, u, op, 0x29, -1, 1, 2, 0);
  case POLY_X86_VPMULLW: return x86_encode_op(b, u, op, 0xD5, -1, 1, 1, 0);
  case POLY_X86_VPMULLD: return x86_encode_op(b, u, op, 0x40, -1, 1, 2, 0);
  case POLY_X86_VPADDB: return x86_encode_op(b, u, op, 0xFC, -1, 1, 1, 0);
  case POLY_X86_VPADDW: return x86_encode_op(b, u, op, 0xFD, -1, 1, 1, 0);
  case POLY_X86_VPADDD: return x86_encode_op(b, u, op, 0xFE, -1, 1, 1, 0);
  case POLY_X86_VPADDQ: return x86_encode_op(b, u, op, 0xD4, -1, 1, 1, 0);
  case POLY_X86_VPSUBB: return x86_encode_op(b, u, op, 0xF8, -1, 1, 1, 0);
  case POLY_X86_VPSUBW: return x86_encode_op(b, u, op, 0xF9, -1, 1, 1, 0);
  case POLY_X86_VPSUBD: return x86_encode_op(b, u, op, 0xFA, -1, 1, 1, 0);
  case POLY_X86_VPSUBQ: return x86_encode_op(b, u, op, 0xFB, -1, 1, 1, 0);

  case POLY_X86_VUCOMISS: return x86_encode_op(b, u, op, 0x2E, -1, 0, 1, 0);
  case POLY_X86_VUCOMISD: return x86_encode_op(b, u, op, 0x2E, -1, 1, 1, 0);
  case POLY_X86_VADDSS: return x86_encode_op(b, u, op, 0x58, -1, 2, 1, 0);
  case POLY_X86_VADDPS: return x86_encode_op(b, u, op, 0x58, -1, 0, 1, 0);
  case POLY_X86_VADDSD: return x86_encode_op(b, u, op, 0x58, -1, 3, 1, 0);
  case POLY_X86_VADDPD: return x86_encode_op(b, u, op, 0x58, -1, 1, 1, 0);
  case POLY_X86_VSUBSS: return x86_encode_op(b, u, op, 0x5C, -1, 2, 1, 0);
  case POLY_X86_VSUBPS: return x86_encode_op(b, u, op, 0x5C, -1, 0, 1, 0);
  case POLY_X86_VSUBSD: return x86_encode_op(b, u, op, 0x5C, -1, 3, 1, 0);
  case POLY_X86_VSUBPD: return x86_encode_op(b, u, op, 0x5C, -1, 1, 1, 0);
  case POLY_X86_VMULSS: return x86_encode_op(b, u, op, 0x59, -1, 2, 1, 0);
  case POLY_X86_VMULPS: return x86_encode_op(b, u, op, 0x59, -1, 0, 1, 0);
  case POLY_X86_VMULSD: return x86_encode_op(b, u, op, 0x59, -1, 3, 1, 0);
  case POLY_X86_VMULPD: return x86_encode_op(b, u, op, 0x59, -1, 1, 1, 0);
  case POLY_X86_VDIVSS: return x86_encode_op(b, u, op, 0x5E, -1, 2, 1, 0);
  case POLY_X86_VDIVPS: return x86_encode_op(b, u, op, 0x5E, -1, 0, 1, 0);
  case POLY_X86_VDIVSD: return x86_encode_op(b, u, op, 0x5E, -1, 3, 1, 0);
  case POLY_X86_VDIVPD: return x86_encode_op(b, u, op, 0x5E, -1, 1, 1, 0);
  case POLY_X86_VCMPSS: return x86_encode_op(b, u, op, 0xC2, -1, 2, 1, 0);
  case POLY_X86_VCMPPS: return x86_encode_op(b, u, op, 0xC2, -1, 0, 1, 0);
  case POLY_X86_VCMPSD: return x86_encode_op(b, u, op, 0xC2, -1, 3, 1, 0);
  case POLY_X86_VCMPPD: return x86_encode_op(b, u, op, 0xC2, -1, 1, 1, 0);
  case POLY_X86_VMAXSS: return x86_encode_op(b, u, op, 0x5F, -1, 2, 1, 0);
  case POLY_X86_VMAXPS: return x86_encode_op(b, u, op, 0x5F, -1, 0, 1, 0);
  case POLY_X86_VMAXSD: return x86_encode_op(b, u, op, 0x5F, -1, 3, 1, 0);
  case POLY_X86_VMAXPD: return x86_encode_op(b, u, op, 0x5F, -1, 1, 1, 0);
  case POLY_X86_VMINSS: return x86_encode_op(b, u, op, 0x5D, -1, 2, 1, 0);
  case POLY_X86_VMINPS: return x86_encode_op(b, u, op, 0x5D, -1, 0, 1, 0);
  case POLY_X86_VMINSD: return x86_encode_op(b, u, op, 0x5D, -1, 3, 1, 0);
  case POLY_X86_VMINPD: return x86_encode_op(b, u, op, 0x5D, -1, 1, 1, 0);

  case POLY_X86_CMOVB: return x86_encode_op(b, u, op, 0x0F42, -1, 0, 0, 0);
  case POLY_X86_CMOVL: return x86_encode_op(b, u, op, 0x0F4C, -1, 0, 0, 0);
  case POLY_X86_CMOVE: return x86_encode_op(b, u, op, 0x0F44, -1, 0, 0, 0);
  case POLY_X86_CMOVNE: return x86_encode_op(b, u, op, 0x0F45, -1, 0, 0, 0);
  case POLY_X86_VFMADD213SS: return x86_encode_op(b, u, op, 0xA9, -1, 1, 2, 0);
  case POLY_X86_VFMADD213SD: return x86_encode_op(b, u, op, 0xA9, -1, 1, 2, 1);
  case POLY_X86_VFMADD213PS: return x86_encode_op(b, u, op, 0xA8, -1, 1, 2, 0);
  case POLY_X86_VFMADD213PD: return x86_encode_op(b, u, op, 0xA8, -1, 1, 2, 1);
  case POLY_X86_VBLENDVPS: return x86_encode_op(b, u, op, 0x4A, -1, 1, 3, 0);
  case POLY_X86_VBLENDVPD: return x86_encode_op(b, u, op, 0x4B, -1, 1, 3, 0);
  case POLY_X86_VPBLENDVB: return x86_encode_op(b, u, op, 0x4C, -1, 1, 3, 0);
  case POLY_X86_VPBROADCASTB: return x86_encode_op(b, u, op, 0x78, -1, 1, 2, 0);
  case POLY_X86_VPBROADCASTW: return x86_encode_op(b, u, op, 0x79, -1, 1, 2, 0);
  case POLY_X86_VPBROADCASTD: return x86_encode_op(b, u, op, 0x58, -1, 1, 2, 0);
  case POLY_X86_VPBROADCASTQ: return x86_encode_op(b, u, op, 0x59, -1, 1, 2, 0);
  case POLY_X86_VBROADCASTSS: return x86_encode_op(b, u, op, 0x18, -1, 1, 2, 0);
  case POLY_X86_VPSRLDQ: return x86_encode_op(b, u, op, 0x73, 3, 1, 1, 0);
  case POLY_X86_VPINSRB: return x86_encode_op(b, u, op, 0x20, -1, 1, 3, 0);
  case POLY_X86_VPINSRW: return x86_encode_op(b, u, op, 0xC4, -1, 1, 1, 0);
  case POLY_X86_VPINSRD: return x86_encode_op(b, u, op, 0x22, -1, 1, 3, 0);
  case POLY_X86_VPINSRQ: return x86_encode_op(b, u, op, 0x22, -1, 1, 3, 1);
  case POLY_X86_VSHUFPS: return x86_encode_op(b, u, op, 0xC6, -1, 0, 1, 0);
  case POLY_X86_VSHUFPD: return x86_encode_op(b, u, op, 0xC6, -1, 1, 1, 0);
  case POLY_X86_VINSERTPS: return x86_encode_op(b, u, op, 0x21, -1, 1, 3, 0);
  case POLY_X86_VPEXTRB: return x86_encode_op(b, u, op, 0x14, -1, 1, 3, 0);
  case POLY_X86_VPEXTRW: return x86_encode_op(b, u, op, 0x15, -1, 1, 3, 0);
  case POLY_X86_VPEXTRD: return x86_encode_op(b, u, op, 0x16, -1, 1, 3, 0);
  case POLY_X86_VPEXTRQ: return x86_encode_op(b, u, op, 0x16, -1, 1, 3, 1);

  case POLY_X86_JE:
    xb_byte(b, 0x0F); xb_byte(b, 0x84); xb_i32(b, 0); return 0;
  case POLY_X86_JNE:
    xb_byte(b, 0x0F); xb_byte(b, 0x85); xb_i32(b, 0); return 0;
  case POLY_X86_JL:
    xb_byte(b, 0x0F); xb_byte(b, 0x8C); xb_i32(b, 0); return 0;
  case POLY_X86_JB:
    xb_byte(b, 0x0F); xb_byte(b, 0x82); xb_i32(b, 0); return 0;
  case POLY_X86_JGE:
    xb_byte(b, 0x0F); xb_byte(b, 0x8D); xb_i32(b, 0); return 0;
  case POLY_X86_JMP:
    xb_byte(b, 0xE9); xb_i32(b, 0); return 0;
  default:
    return -1;
  }
}

typedef struct {
  const char *name;
  int offset;
} X86LabelPos;

typedef struct {
  const char *name;
  int disp_offset;
} X86JumpFixup;

static int label_pos_find(X86LabelPos *labels, int n, const char *name) {
  if (!name) return -1;
  for (int i = 0; i < n; i++)
    if (labels[i].name && strcmp(labels[i].name, name) == 0) return i;
  return -1;
}

static int label_pos_add(X86LabelPos **labels, int *n, int *cap, const char *name, int offset) {
  if (!name) return -1;
  int idx = label_pos_find(*labels, *n, name);
  if (idx >= 0) {
    (*labels)[idx].offset = offset;
    return 0;
  }
  if (*n >= *cap) {
    int nc = *cap ? *cap * 2 : 16;
    X86LabelPos *nl = realloc(*labels, (size_t)nc * sizeof(*nl));
    if (!nl) return -1;
    *labels = nl;
    *cap = nc;
  }
  (*labels)[(*n)++] = (X86LabelPos){name, offset};
  return 0;
}

static int jump_fixup_add(X86JumpFixup **fixups, int *n, int *cap, const char *name, int disp_offset) {
  if (!name) return -1;
  if (*n >= *cap) {
    int nc = *cap ? *cap * 2 : 16;
    X86JumpFixup *nf = realloc(*fixups, (size_t)nc * sizeof(*nf));
    if (!nf) return -1;
    *fixups = nf;
    *cap = nc;
  }
  (*fixups)[(*n)++] = (X86JumpFixup){name, disp_offset};
  return 0;
}

uint8_t *poly_render_x86(PolyUOp **uops, int n, int *size_out) {
  if (size_out) *size_out = 0;
  if (!uops || n <= 0) return NULL;

  X86Buf b = {0};
  X86LabelPos *labels = NULL;
  X86JumpFixup *fixups = NULL;
  int n_labels = 0, cap_labels = 0, n_fixups = 0, cap_fixups = 0;

  for (int i = 0; i < n; i++) {
    PolyUOp *u = uops[i];
    if (u->op == POLY_OP_SINK || u->op == POLY_OP_GROUP || u->op == POLY_OP_NOOP) continue;
    PolyX86Op op;
    if (!x86_ins_op(u, &op)) {
      fprintf(stderr, "x86 renderer: expected POLY_OP_INS, got %s at %d\n", poly_op_name(u->op), i);
      free(b.data);
      free(labels);
      free(fixups);
      return NULL;
    }
    if (op == POLY_X86_LABEL) {
      if (label_pos_add(&labels, &n_labels, &cap_labels, x86_uop_label(u), b.len) != 0) {
        free(b.data);
        free(labels);
        free(fixups);
        return NULL;
      }
      continue;
    }
    int before = b.len;
    if (poly_debug_at_least(6)) {
      char *us = poly_uop_str(u);
      fprintf(
          stderr, "x86 emit %04d off=%04d %-12s %s\n",
          i, before, x86_op_name(op), us ? us : "<uop>"
      );
      free(us);
    }
    if (x86_encode_instruction(&b, u, op) != 0) {
      char *us = poly_uop_str(u);
      fprintf(
          stderr, "x86 renderer: failed to encode instruction %s at %d: %s\n",
          x86_op_name(op), i, us ? us : "<uop>"
      );
      free(us);
      if (poly_debug_at_least(6)) {
        for (int si = 0; si < u->n_src; si++) {
          char *ss = poly_uop_str(u->src[si]);
          fprintf(
              stderr, "  src[%d] tag=%d tag_arg=%d %s\n",
              si, u->src[si] ? u->src[si]->tag : 0,
              u->src[si] ? u->src[si]->tag_arg.kind : 0,
              ss ? ss : "<uop>"
          );
          free(ss);
        }
      }
      free(b.data);
      free(labels);
      free(fixups);
      return NULL;
    }
    if (x86_is_jump(op)) {
      int disp_offset = (op == POLY_X86_JMP) ? before + 1 : before + 2;
      if (jump_fixup_add(&fixups, &n_fixups, &cap_fixups, x86_uop_label(u), disp_offset) != 0) {
        free(b.data);
        free(labels);
        free(fixups);
        return NULL;
      }
    }
  }

  for (int i = 0; i < n_fixups; i++) {
    int li = label_pos_find(labels, n_labels, fixups[i].name);
    if (li < 0 || fixups[i].disp_offset < 0 || fixups[i].disp_offset + 4 > b.len) {
      fprintf(
          stderr, "x86 renderer: unresolved jump label %s\n",
          fixups[i].name ? fixups[i].name : "<null>"
      );
      free(b.data);
      free(labels);
      free(fixups);
      return NULL;
    }
    int32_t rel = (int32_t)(labels[li].offset - (fixups[i].disp_offset + 4));
    memcpy(b.data + fixups[i].disp_offset, &rel, 4);
  }

  free(labels);
  free(fixups);
  if (size_out) *size_out = b.len;
  x86_dump_binary_if_requested(b.data, b.len);
  if (poly_debug_at_least(6)) {
    fprintf(stderr, "x86 code hex");
    for (int i = 0; i < b.len; i++) fprintf(stderr, "%02x", b.data[i]);
    fprintf(stderr, "\n");
  }
  return b.data;
}

char *poly_render_x86_source(PolyUOp **uops, int n) {
  int code_size = 0;
  uint8_t *code = poly_render_x86(uops, n, &code_size);
  if (!code || code_size <= 0) {
    free(code);
    return NULL;
  }
  char *hex = malloc((size_t)code_size * 2 + 1);
  if (!hex) {
    free(code);
    return NULL;
  }
  static const char digits[] = "0123456789abcdef";
  for (int i = 0; i < code_size; i++) {
    hex[2 * i] = digits[code[i] >> 4];
    hex[2 * i + 1] = digits[code[i] & 0x0f];
  }
  hex[(size_t)code_size * 2] = '\0';
  free(code);
  return hex;
}

static int x86_thread_count(void) {
  int n_env = poly_getenv_int("CPU_COUNT", 0);
  if (n_env > 0) return n_env;
#ifdef _SC_NPROCESSORS_ONLN
  long n = sysconf(_SC_NPROCESSORS_ONLN);
  if (n > 0 && n < INT32_MAX) return (int)n;
#endif
  return 1;
}

static PolyRendererCaps poly_x86_caps(void) {
  bool has_threads = poly_getenv_flag_default("THREADS", true);
  return (PolyRendererCaps){
      /* tinygrad X86Renderer does not advertise Ops.MULACC to generic rewrite.
       * It folds MUL+ADD to VFMADD only inside x86 instruction selection when
       * the MUL is foldable. */
      .has_mulacc = false,
      .has_threefry = false,
      .has_local = false,
      .has_threads = has_threads,
      .has_simd_int = true,
      .has_simd_float = true,
      .max_vec_width = 4,
      .max_threads = has_threads ? x86_thread_count() : 0,
  };
}

uint32_t poly_x86_feature_stamp(void) {
  uint32_t stamp = 1u; /* x86 backend stamp version */
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
  __builtin_cpu_init();
  if (__builtin_cpu_supports("sse4.1")) stamp |= 1u << 1;
  if (__builtin_cpu_supports("sse4.2")) stamp |= 1u << 2;
  if (__builtin_cpu_supports("avx")) stamp |= 1u << 3;
  if (__builtin_cpu_supports("avx2")) stamp |= 1u << 4;
  if (__builtin_cpu_supports("fma")) stamp |= 1u << 5;
  if (__builtin_cpu_supports("f16c")) stamp |= 1u << 6;
  if (__builtin_cpu_supports("bmi2")) stamp |= 1u << 7;
#endif
  return stamp;
}

PolyUOp *poly_rewrite_x86(PolyCtx *ctx, PolyUOp *sink) {
  PolyRewriteOpts opts = {
      .optimize = true,
      .devectorize = 1,
      .caps = poly_x86_caps(),
      .device = POLY_DEVICE_CPU,
      .opt_policy = POLY_OPT_HEURISTIC,
      .extra_matcher = poly_pm_x86_extra(),
  };
  return poly_full_rewrite_to_sink_ex(ctx, sink, opts);
}

PolyUOp **poly_linearize_x86_rewritten(PolyCtx *ctx, PolyUOp *sink, int *n_out) {
  double t0 = poly_now_ms();
  if (poly_debug_at_least(4)) fprintf(stderr, "x86 linearize: pre-isel rewrite begin\n");
  sink = poly_graph_rewrite_ex(ctx, sink, poly_pm_x86_pre_isel(), true);
  if (poly_debug_at_least(4))
    fprintf(stderr, "x86 linearize: pre-isel rewrite done %.3fms\n", poly_now_ms() - t0);
  t0 = poly_now_ms();
  if (poly_debug_at_least(4)) fprintf(stderr, "x86 linearize: isel rewrite begin\n");
  sink = poly_graph_rewrite_x86_isel(ctx, sink);
  if (poly_debug_at_least(4))
    fprintf(stderr, "x86 linearize: isel rewrite done %.3fms\n", poly_now_ms() - t0);
  if (!sink) {
    if (n_out) *n_out = 0;
    return NULL;
  }
  int n_base = 0;
  PolyUOp **base = poly_linearize_rewritten(ctx, sink, &n_base);
  if (!base) {
    if (n_out) *n_out = 0;
    return NULL;
  }
  if (!x86_validate_graph_isel_residuals(base, n_base)) {
    free(base);
    if (n_out) *n_out = 0;
    return NULL;
  }
  int n_isel = 0;
  PolyUOp **isel = x86_isel_linear(ctx, base, n_base, &n_isel);
  free(base);
  if (!isel) {
    if (poly_debug_at_least(4)) fprintf(stderr, "x86 linearize: isel failed\n");
    if (n_out) *n_out = 0;
    return NULL;
  }
  if (poly_debug_at_least(4)) {
    fprintf(stderr, "x86 linearize: isel n=%d\n", n_isel);
    for (int i = 0; i < n_isel; i++) {
      char *s = poly_uop_str(isel[i]);
      fprintf(stderr, "  isel %02d %s tag=%d tag_arg=%d\n", i, s ? s : "<uop>", isel[i]->tag, isel[i]->tag_arg.kind);
      free(s);
    }
  }
  PolyUOp **fold = isel;
  int n_fold = n_isel;
  if (poly_debug_at_least(4)) fprintf(stderr, "x86 linearize: fold-loads n=%d (graph)\n", n_fold);
  int n_pre = 0;
  PolyUOp **pre = x86_pre_regalloc_linear(fold, n_fold, &n_pre);
  free(fold);
  if (!pre) {
    if (poly_debug_at_least(4)) fprintf(stderr, "x86 linearize: pre-regalloc failed\n");
    if (n_out) *n_out = 0;
    return NULL;
  }
  if (poly_debug_at_least(4)) fprintf(stderr, "x86 linearize: pre-regalloc n=%d\n", n_pre);
  int n_sorted = 0;
  PolyUOp **sorted = x86_sort_reg_defs_for_regalloc(pre, n_pre, &n_sorted);
  free(pre);
  if (!sorted) {
    if (poly_debug_at_least(4)) fprintf(stderr, "x86 linearize: reg-def sort failed\n");
    if (n_out) *n_out = 0;
    return NULL;
  }
  if (poly_debug_at_least(4)) fprintf(stderr, "x86 linearize: reg-def sort n=%d\n", n_sorted);
  int n_reg = 0;
  PolyUOp **reg = x86_regalloc_linear(ctx, sorted, n_sorted, &n_reg);
  free(sorted);
  if (!reg) {
    if (poly_debug_at_least(4)) fprintf(stderr, "x86 linearize: regalloc failed\n");
    if (n_out) *n_out = 0;
    return NULL;
  }
  if (poly_debug_at_least(4)) fprintf(stderr, "x86 linearize: regalloc n=%d\n", n_reg);
  PolyUOp **post = x86_post_regalloc_linear(ctx, reg, n_reg, n_out);
  free(reg);
  if (!post && poly_debug_at_least(4)) fprintf(stderr, "x86 linearize: post-regalloc failed\n");
  if (post && n_out && poly_debug_at_least(4)) fprintf(stderr, "x86 linearize: post n=%d\n", *n_out);
  return post;
}

PolyUOp **poly_linearize_x86(PolyCtx *ctx, PolyUOp *sink, int *n_out) {
  sink = poly_rewrite_x86(ctx, sink);
  if (!sink) {
    if (n_out) *n_out = 0;
    return NULL;
  }
  return poly_linearize_x86_rewritten(ctx, sink, n_out);
}

struct PolyX86Program {
  void *code;
  size_t code_size;
  void *entry;
};

PolyX86Program *poly_compile_x86(const uint8_t *code, int code_size) {
  if (!code || code_size <= 0) return NULL;
  long page_size = sysconf(_SC_PAGESIZE);
  size_t alloc_size = ((size_t)code_size + (size_t)page_size - 1) & ~((size_t)page_size - 1);
  void *mem = mmap(NULL, alloc_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (mem == MAP_FAILED) return NULL;
  memcpy(mem, code, (size_t)code_size);
  if (mprotect(mem, alloc_size, PROT_READ | PROT_EXEC) != 0) {
    munmap(mem, alloc_size);
    return NULL;
  }
  PolyX86Program *prog = calloc(1, sizeof(*prog));
  if (!prog) {
    munmap(mem, alloc_size);
    return NULL;
  }
  prog->code = mem;
  prog->code_size = alloc_size;
  prog->entry = mem;
  return prog;
}

static int hex_nibble(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

PolyX86Program *poly_compile_x86_source(const char *source) {
  if (!source) return NULL;
  size_t len = strlen(source);
  if (len == 0 || (len & 1)) return NULL;
  int code_size = (int)(len / 2);
  uint8_t *code = malloc((size_t)code_size);
  if (!code) return NULL;
  for (int i = 0; i < code_size; i++) {
    int hi = hex_nibble(source[2 * i]);
    int lo = hex_nibble(source[2 * i + 1]);
    if (hi < 0 || lo < 0) {
      free(code);
      return NULL;
    }
    code[i] = (uint8_t)((hi << 4) | lo);
  }
  PolyX86Program *prog = poly_compile_x86(code, code_size);
  free(code);
  return prog;
}

int poly_x86_program_call(PolyX86Program *prog, void **args, int n_args) {
  if (!prog || !args || n_args < 0) return -1;
  switch (n_args) {
  case 0: {
    typedef void (*Fn)(void);
    Fn fn;
    memcpy(&fn, &prog->entry, sizeof(fn));
    fn();
    return 0;
  }
  case 1: {
    typedef void (*Fn)(void *);
    Fn fn;
    memcpy(&fn, &prog->entry, sizeof(fn));
    fn(args[0]);
    return 0;
  }
  case 2: {
    typedef void (*Fn)(void *, void *);
    Fn fn;
    memcpy(&fn, &prog->entry, sizeof(fn));
    fn(args[0], args[1]);
    return 0;
  }
  case 3: {
    typedef void (*Fn)(void *, void *, void *);
    Fn fn;
    memcpy(&fn, &prog->entry, sizeof(fn));
    fn(args[0], args[1], args[2]);
    return 0;
  }
  case 4: {
    typedef void (*Fn)(void *, void *, void *, void *);
    Fn fn;
    memcpy(&fn, &prog->entry, sizeof(fn));
    fn(args[0], args[1], args[2], args[3]);
    return 0;
  }
  case 5: {
    typedef void (*Fn)(void *, void *, void *, void *, void *);
    Fn fn;
    memcpy(&fn, &prog->entry, sizeof(fn));
    fn(args[0], args[1], args[2], args[3], args[4]);
    return 0;
  }
  case 6: {
    typedef void (*Fn)(void *, void *, void *, void *, void *, void *);
    Fn fn;
    memcpy(&fn, &prog->entry, sizeof(fn));
    fn(args[0], args[1], args[2], args[3], args[4], args[5]);
    return 0;
  }
  case 7: {
    typedef void (*Fn)(void *, void *, void *, void *, void *, void *, void *);
    Fn fn;
    memcpy(&fn, &prog->entry, sizeof(fn));
    fn(args[0], args[1], args[2], args[3], args[4], args[5], args[6]);
    return 0;
  }
  case 8: {
    typedef void (*Fn)(void *, void *, void *, void *, void *, void *, void *, void *);
    Fn fn;
    memcpy(&fn, &prog->entry, sizeof(fn));
    fn(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]);
    return 0;
  }
  case 9: {
    typedef void (*Fn)(void *, void *, void *, void *, void *, void *, void *, void *, void *);
    Fn fn;
    memcpy(&fn, &prog->entry, sizeof(fn));
    fn(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8]);
    return 0;
  }
  case 10: {
    typedef void (*Fn)(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
    Fn fn;
    memcpy(&fn, &prog->entry, sizeof(fn));
    fn(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9]);
    return 0;
  }
  case 11: {
    typedef void (*Fn)(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
    Fn fn;
    memcpy(&fn, &prog->entry, sizeof(fn));
    fn(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9],
       args[10]);
    return 0;
  }
  case 12: {
    typedef void (*Fn)(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *,
                       void *);
    Fn fn;
    memcpy(&fn, &prog->entry, sizeof(fn));
    fn(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9],
       args[10], args[11]);
    return 0;
  }
  case 13: {
    typedef void (*Fn)(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *,
                       void *, void *);
    Fn fn;
    memcpy(&fn, &prog->entry, sizeof(fn));
    fn(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9],
       args[10], args[11], args[12]);
    return 0;
  }
  case 14: {
    typedef void (*Fn)(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *,
                       void *, void *, void *);
    Fn fn;
    memcpy(&fn, &prog->entry, sizeof(fn));
    fn(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9],
       args[10], args[11], args[12], args[13]);
    return 0;
  }
  case 15: {
    typedef void (*Fn)(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *,
                       void *, void *, void *, void *);
    Fn fn;
    memcpy(&fn, &prog->entry, sizeof(fn));
    fn(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9],
       args[10], args[11], args[12], args[13], args[14]);
    return 0;
  }
  case 16: {
    typedef void (*Fn)(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *,
                       void *, void *, void *, void *, void *);
    Fn fn;
    memcpy(&fn, &prog->entry, sizeof(fn));
    fn(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9],
       args[10], args[11], args[12], args[13], args[14], args[15]);
    return 0;
  }
  default: return -1;
  }
}

int poly_x86_program_call_core(PolyX86Program *prog, void **args, int n_args, int core_id) {
  if (!prog || !args || n_args < 0) return -1;
  enum { STACK_CAP = 32 };
  void *stack_args[STACK_CAP];
  void **call_args = stack_args;
  if (n_args + 1 > STACK_CAP) {
    call_args = malloc((size_t)(n_args + 1) * sizeof(void *));
    if (!call_args) return -1;
  }
  for (int i = 0; i < n_args; i++) call_args[i] = args[i];
  call_args[n_args] = (void *)(intptr_t)core_id;
  int rc = poly_x86_program_call(prog, call_args, n_args + 1);
  if (call_args != stack_args) free(call_args);
  return rc;
}

typedef struct {
  int core_id;
  bool live;
  pthread_t thread;
} PolyX86ThreadCall;

typedef struct {
  pthread_mutex_t mu;
  pthread_cond_t start_cv;
  pthread_cond_t done_cv;
  PolyX86ThreadCall workers[64];
  PolyX86Program *prog;
  void **args;
  int n_args;
  int requested_threads;
  int active_workers;
  uint64_t generation;
  bool running;
  bool stop;
} PolyX86ThreadPool;

static PolyX86ThreadPool g_x86_thread_pool = {
    .mu = PTHREAD_MUTEX_INITIALIZER,
    .start_cv = PTHREAD_COND_INITIALIZER,
    .done_cv = PTHREAD_COND_INITIALIZER,
};
static pthread_once_t g_x86_thread_pool_once = PTHREAD_ONCE_INIT;

static void poly_x86_thread_pool_shutdown(void);

static void poly_x86_thread_pool_once(void) {
  atexit(poly_x86_thread_pool_shutdown);
}

static void *poly_x86_thread_call_main(void *opaque) {
  PolyX86ThreadCall *tc = (PolyX86ThreadCall *)opaque;
  PolyX86ThreadPool *p = &g_x86_thread_pool;
  uint64_t seen_generation = 0;

  pthread_mutex_lock(&p->mu);
  for (;;) {
    while (!p->stop && (!p->running || p->generation == seen_generation ||
                        tc->core_id >= p->requested_threads)) {
      pthread_cond_wait(&p->start_cv, &p->mu);
    }
    if (p->stop) break;

    PolyX86Program *prog = p->prog;
    void **args = p->args;
    int n_args = p->n_args;
    int core_id = tc->core_id;
    seen_generation = p->generation;
    pthread_mutex_unlock(&p->mu);

    (void)poly_x86_program_call_core(prog, args, n_args, core_id);

    pthread_mutex_lock(&p->mu);
    p->active_workers--;
    if (p->active_workers == 0) pthread_cond_signal(&p->done_cv);
  }
  pthread_mutex_unlock(&p->mu);
  return NULL;
}

static int poly_x86_thread_cap(int threads) {
  int max_threads = (int)(sizeof(g_x86_thread_pool.workers) / sizeof(g_x86_thread_pool.workers[0]));
  if (threads < 1) return 1;
  if (threads > max_threads) return max_threads;
  return threads;
}

static void poly_x86_thread_pool_ensure_locked(PolyX86ThreadPool *p, int threads) {
  for (int t = 1; t < threads; t++) {
    PolyX86ThreadCall *tc = &p->workers[t];
    if (tc->live) continue;
    tc->core_id = t;
    if (pthread_create(&tc->thread, NULL, poly_x86_thread_call_main, tc) == 0) tc->live = true;
  }
}

static void poly_x86_thread_pool_shutdown(void) {
  PolyX86ThreadPool *p = &g_x86_thread_pool;
  pthread_mutex_lock(&p->mu);
  p->stop = true;
  pthread_cond_broadcast(&p->start_cv);
  pthread_mutex_unlock(&p->mu);

  int max_threads = (int)(sizeof(p->workers) / sizeof(p->workers[0]));
  for (int t = 1; t < max_threads; t++) {
    if (!p->workers[t].live) continue;
    pthread_join(p->workers[t].thread, NULL);
    p->workers[t].live = false;
  }
}

int poly_x86_program_call_threaded(PolyX86Program *prog, void **args, int n_args, int threads) {
  if (!prog || !args) return -1;
  if (threads <= 1) return poly_x86_program_call(prog, args, n_args);

  pthread_once(&g_x86_thread_pool_once, poly_x86_thread_pool_once);
  threads = poly_x86_thread_cap(threads);

  PolyX86ThreadPool *p = &g_x86_thread_pool;
  pthread_mutex_lock(&p->mu);
  while (p->running)
    pthread_cond_wait(&p->done_cv, &p->mu);

  poly_x86_thread_pool_ensure_locked(p, threads);

  p->prog = prog;
  p->args = args;
  p->n_args = n_args;
  p->requested_threads = threads;
  p->active_workers = 0;
  for (int t = 1; t < threads; t++)
    if (p->workers[t].live) p->active_workers++;
  p->running = true;
  p->generation++;
  pthread_cond_broadcast(&p->start_cv);
  pthread_mutex_unlock(&p->mu);

  int rc = poly_x86_program_call_core(prog, args, n_args, 0);

  pthread_mutex_lock(&p->mu);
  for (int t = 1; t < threads; t++) {
    if (p->workers[t].live) continue;
    pthread_mutex_unlock(&p->mu);
    int worker_rc = poly_x86_program_call_core(prog, args, n_args, t);
    if (worker_rc != 0) rc = worker_rc;
    pthread_mutex_lock(&p->mu);
  }
  while (p->active_workers > 0)
    pthread_cond_wait(&p->done_cv, &p->mu);
  p->running = false;
  p->prog = NULL;
  p->args = NULL;
  p->n_args = 0;
  pthread_cond_broadcast(&p->done_cv);
  pthread_mutex_unlock(&p->mu);
  return rc;
}

void poly_x86_program_destroy(PolyX86Program *prog) {
  if (!prog) return;
  if (prog->code) munmap(prog->code, prog->code_size);
  free(prog);
}

#endif /* POLY_HAS_X86 */
