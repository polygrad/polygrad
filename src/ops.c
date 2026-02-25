/*
 * ops.c — Ops enum helpers and GroupOp bitmask sets
 *
 * Mirrors tinygrad's Ops enum and GroupOp class from uop/__init__.py
 */

#include "polygrad.h"

/* ── Op names ─────────────────────────────────────────────────────────── */

static const char *op_names[] = {
  [0] = "INVALID",
  [POLY_OP_DEFINE_VAR] = "DEFINE_VAR",
  [POLY_OP_BIND] = "BIND",
  [POLY_OP_SPECIAL] = "SPECIAL",
  [POLY_OP_DEFINE_LOCAL] = "DEFINE_LOCAL",
  [POLY_OP_DEFINE_REG] = "DEFINE_REG",
  [POLY_OP_NOOP] = "NOOP",
  [POLY_OP_REWRITE_ERROR] = "REWRITE_ERROR",
  [POLY_OP_PARAM] = "PARAM",
  [POLY_OP_CALL] = "CALL",
  [POLY_OP_PROGRAM] = "PROGRAM",
  [POLY_OP_LINEAR] = "LINEAR",
  [POLY_OP_SOURCE] = "SOURCE",
  [POLY_OP_BINARY] = "BINARY",
  [POLY_OP_SINK] = "SINK",
  [POLY_OP_AFTER] = "AFTER",
  [POLY_OP_GROUP] = "GROUP",
  [POLY_OP_GEP] = "GEP",
  [POLY_OP_VECTORIZE] = "VECTORIZE",
  [POLY_OP_INDEX] = "INDEX",
  [POLY_OP_LOAD] = "LOAD",
  [POLY_OP_STORE] = "STORE",
  [POLY_OP_WMMA] = "WMMA",
  [POLY_OP_CAST] = "CAST",
  [POLY_OP_BITCAST] = "BITCAST",
  [POLY_OP_EXP2] = "EXP2",
  [POLY_OP_LOG2] = "LOG2",
  [POLY_OP_SIN] = "SIN",
  [POLY_OP_SQRT] = "SQRT",
  [POLY_OP_RECIPROCAL] = "RECIPROCAL",
  [POLY_OP_NEG] = "NEG",
  [POLY_OP_TRUNC] = "TRUNC",
  [POLY_OP_ADD] = "ADD",
  [POLY_OP_MUL] = "MUL",
  [POLY_OP_SHL] = "SHL",
  [POLY_OP_SHR] = "SHR",
  [POLY_OP_IDIV] = "IDIV",
  [POLY_OP_MAX] = "MAX",
  [POLY_OP_MOD] = "MOD",
  [POLY_OP_CMPLT] = "CMPLT",
  [POLY_OP_CMPNE] = "CMPNE",
  [POLY_OP_CMPEQ] = "CMPEQ",
  [POLY_OP_XOR] = "XOR",
  [POLY_OP_OR] = "OR",
  [POLY_OP_AND] = "AND",
  [POLY_OP_THREEFRY] = "THREEFRY",
  [POLY_OP_SUB] = "SUB",
  [POLY_OP_FDIV] = "FDIV",
  [POLY_OP_POW] = "POW",
  [POLY_OP_WHERE] = "WHERE",
  [POLY_OP_MULACC] = "MULACC",
  [POLY_OP_BARRIER] = "BARRIER",
  [POLY_OP_RANGE] = "RANGE",
  [POLY_OP_IF] = "IF",
  [POLY_OP_END] = "END",
  [POLY_OP_ENDIF] = "ENDIF",
  [POLY_OP_VCONST] = "VCONST",
  [POLY_OP_CONST] = "CONST",
  [POLY_OP_CUSTOM] = "CUSTOM",
  [POLY_OP_CUSTOMI] = "CUSTOMI",
  [POLY_OP_INS] = "INS",
  [POLY_OP_UNIQUE] = "UNIQUE",
  [POLY_OP_DEVICE] = "DEVICE",
  [POLY_OP_ASSIGN] = "ASSIGN",
  [POLY_OP_LUNIQUE] = "LUNIQUE",
  [POLY_OP_CONTIGUOUS] = "CONTIGUOUS",
  [POLY_OP_CONTIGUOUS_BACKWARD] = "CONTIGUOUS_BACKWARD",
  [POLY_OP_DETACH] = "DETACH",
  [POLY_OP_BUFFERIZE] = "BUFFERIZE",
  [POLY_OP_COPY] = "COPY",
  [POLY_OP_BUFFER] = "BUFFER",
  [POLY_OP_BUFFER_VIEW] = "BUFFER_VIEW",
  [POLY_OP_MSELECT] = "MSELECT",
  [POLY_OP_MSTACK] = "MSTACK",
  [POLY_OP_ENCDEC] = "ENCDEC",
  [POLY_OP_RESHAPE] = "RESHAPE",
  [POLY_OP_PERMUTE] = "PERMUTE",
  [POLY_OP_EXPAND] = "EXPAND",
  [POLY_OP_PAD] = "PAD",
  [POLY_OP_SHRINK] = "SHRINK",
  [POLY_OP_FLIP] = "FLIP",
  [POLY_OP_MULTI] = "MULTI",
  [POLY_OP_REDUCE_AXIS] = "REDUCE_AXIS",
  [POLY_OP_REDUCE] = "REDUCE",
  [POLY_OP_ALLREDUCE] = "ALLREDUCE",
  [POLY_OP_UNROLL] = "UNROLL",
  [POLY_OP_CONTRACT] = "CONTRACT",
  [POLY_OP_CAT] = "CAT",
  [POLY_OP_PTRCAT] = "PTRCAT",
};

const char *poly_op_name(PolyOps op) {
  if (op >= 0 && op < POLY_OP_COUNT) return op_names[op];
  return "UNKNOWN";
}

/* ── GroupOp bitmask construction helper ──────────────────────────────── */

#define OPSET(...) opset_build((PolyOps[]){__VA_ARGS__}, \
  sizeof((PolyOps[]){__VA_ARGS__}) / sizeof(PolyOps))

static PolyOpSet opset_build(const PolyOps *ops, int n) {
  PolyOpSet s = {{ 0, 0 }};
  for (int i = 0; i < n; i++) {
    s.bits[ops[i] / 64] |= (uint64_t)1 << (ops[i] % 64);
  }
  return s;
}

/* ── GroupOp sets (mirrors tinygrad GroupOp from uop/__init__.py) ──────── */

PolyOpSet POLY_GROUP_UNARY = {{ 0, 0 }};
PolyOpSet POLY_GROUP_BINARY = {{ 0, 0 }};
PolyOpSet POLY_GROUP_TERNARY = {{ 0, 0 }};
PolyOpSet POLY_GROUP_ALU = {{ 0, 0 }};
PolyOpSet POLY_GROUP_ELEMENTWISE = {{ 0, 0 }};
PolyOpSet POLY_GROUP_MOVEMENT = {{ 0, 0 }};
PolyOpSet POLY_GROUP_COMMUTATIVE = {{ 0, 0 }};
PolyOpSet POLY_GROUP_ASSOCIATIVE = {{ 0, 0 }};
PolyOpSet POLY_GROUP_IDEMPOTENT = {{ 0, 0 }};
PolyOpSet POLY_GROUP_COMPARISON = {{ 0, 0 }};
PolyOpSet POLY_GROUP_UNSAFEPAD = {{ 0, 0 }};
PolyOpSet POLY_GROUP_BUFFER = {{ 0, 0 }};
PolyOpSet POLY_GROUP_IRREDUCIBLE = {{ 0, 0 }};

static bool g_group_ops_initialized = false;

void poly_init_group_ops(void);

__attribute__((constructor))
void poly_init_group_ops(void) {
  if (g_group_ops_initialized) return;
  g_group_ops_initialized = true;
  #define SET(name, ...) name = OPSET(__VA_ARGS__)

  SET(POLY_GROUP_UNARY,
    POLY_OP_EXP2, POLY_OP_LOG2, POLY_OP_SIN, POLY_OP_SQRT,
    POLY_OP_RECIPROCAL, POLY_OP_NEG, POLY_OP_TRUNC);

  SET(POLY_GROUP_BINARY,
    POLY_OP_ADD, POLY_OP_MUL, POLY_OP_IDIV, POLY_OP_MAX, POLY_OP_MOD,
    POLY_OP_CMPLT, POLY_OP_CMPNE, POLY_OP_CMPEQ,
    POLY_OP_XOR, POLY_OP_SHL, POLY_OP_SHR, POLY_OP_OR, POLY_OP_AND,
    POLY_OP_THREEFRY, POLY_OP_SUB, POLY_OP_FDIV, POLY_OP_POW);

  SET(POLY_GROUP_TERNARY, POLY_OP_WHERE, POLY_OP_MULACC);

  POLY_GROUP_ALU = poly_opset_union(
    poly_opset_union(POLY_GROUP_UNARY, POLY_GROUP_BINARY), POLY_GROUP_TERNARY);

  POLY_GROUP_ELEMENTWISE = poly_opset_union(
    POLY_GROUP_ALU, OPSET(POLY_OP_CAST, POLY_OP_BITCAST));

  SET(POLY_GROUP_MOVEMENT,
    POLY_OP_RESHAPE, POLY_OP_EXPAND, POLY_OP_PERMUTE,
    POLY_OP_PAD, POLY_OP_SHRINK, POLY_OP_FLIP);

  SET(POLY_GROUP_COMMUTATIVE,
    POLY_OP_ADD, POLY_OP_MUL, POLY_OP_MAX,
    POLY_OP_CMPNE, POLY_OP_CMPEQ,
    POLY_OP_XOR, POLY_OP_AND, POLY_OP_OR);

  SET(POLY_GROUP_ASSOCIATIVE,
    POLY_OP_ADD, POLY_OP_MUL, POLY_OP_AND, POLY_OP_OR, POLY_OP_MAX);

  SET(POLY_GROUP_IDEMPOTENT,
    POLY_OP_OR, POLY_OP_AND, POLY_OP_MAX);

  SET(POLY_GROUP_COMPARISON,
    POLY_OP_CMPLT, POLY_OP_CMPNE, POLY_OP_CMPEQ);

  SET(POLY_GROUP_UNSAFEPAD,
    POLY_OP_RECIPROCAL, POLY_OP_LOG2, POLY_OP_EXP2, POLY_OP_IDIV, POLY_OP_POW);

  SET(POLY_GROUP_BUFFER,
    POLY_OP_LOAD, POLY_OP_STORE, POLY_OP_CONST, POLY_OP_DEFINE_VAR);

  SET(POLY_GROUP_IRREDUCIBLE,
    POLY_OP_CONST, POLY_OP_DEFINE_VAR, POLY_OP_SPECIAL, POLY_OP_RANGE);

  #undef SET
}
