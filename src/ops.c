/*
 * ops.c — Ops enum helpers and GroupOp bitmask sets
 *
 * Mirrors tinygrad's Ops enum and GroupOp class from uop/__init__.py
 */

#include "polygrad.h"

/* Op names */

static const char *op_names[] = {
    [POLY_OP_SPECIAL] = "SPECIAL",
    [POLY_OP_NOOP] = "NOOP",
    [POLY_OP_REWRITE_ERROR] = "REWRITE_ERROR",
    [POLY_OP_PARAM] = "PARAM",
    [POLY_OP_FUNCTION] = "FUNCTION",
    [POLY_OP_CALL] = "CALL",
    [POLY_OP_PROGRAM] = "PROGRAM",
    [POLY_OP_LINEAR] = "LINEAR",
    [POLY_OP_SOURCE] = "SOURCE",
    [POLY_OP_BINARY] = "BINARY",
    [POLY_OP_SINK] = "SINK",
    [POLY_OP_AFTER] = "AFTER",
    [POLY_OP_GROUP] = "GROUP",
    [POLY_OP_STACK] = "STACK",
    [POLY_OP_TUPLE] = "TUPLE",
    [POLY_OP_GETTUPLE] = "GETTUPLE",
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
    [POLY_OP_CDIV] = "CDIV",
    [POLY_OP_MAX] = "MAX",
    [POLY_OP_CMOD] = "CMOD",
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
    [POLY_OP_FLOORDIV] = "FLOORDIV",
    [POLY_OP_FLOORMOD] = "FLOORMOD",
    [POLY_OP_WHERE] = "WHERE",
    [POLY_OP_MULACC] = "MULACC",
    [POLY_OP_BARRIER] = "BARRIER",
    [POLY_OP_RANGE] = "RANGE",
    [POLY_OP_IF] = "IF",
    [POLY_OP_END] = "END",
    [POLY_OP_ENDIF] = "ENDIF",
    [POLY_OP_CONST] = "CONST",
    [POLY_OP_CUSTOM] = "CUSTOM",
    [POLY_OP_CUSTOMI] = "CUSTOMI",
    [POLY_OP_INS] = "INS",
    [POLY_OP_UNIQUE] = "UNIQUE",
    [POLY_OP_DEVICE] = "DEVICE",
    [POLY_OP_CONTIGUOUS] = "CONTIGUOUS",
    [POLY_OP_CONTIGUOUS_BACKWARD] = "CONTIGUOUS_BACKWARD",
    [POLY_OP_DETACH] = "DETACH",
    [POLY_OP_STAGE] = "STAGE",
    [POLY_OP_COPY] = "COPY",
    [POLY_OP_BUFFER] = "BUFFER",
    [POLY_OP_MSELECT] = "MSELECT",
    [POLY_OP_MSTACK] = "MSTACK",
    [POLY_OP_CUSTOM_FUNCTION] = "CUSTOM_FUNCTION",
    [POLY_OP_RESHAPE] = "RESHAPE",
    [POLY_OP_PERMUTE] = "PERMUTE",
    [POLY_OP_EXPAND] = "EXPAND",
    [POLY_OP_PAD] = "PAD",
    [POLY_OP_SHRINK] = "SHRINK",
    [POLY_OP_FLIP] = "FLIP",
    [POLY_OP_UNSHARD] = "UNSHARD",
    [POLY_OP_REDUCE] = "REDUCE",
    [POLY_OP_ALLREDUCE] = "ALLREDUCE",
};

const char *poly_op_name(PolyOps op) {
  return op > 0 && op < POLY_OP_COUNT ? op_names[op] : NULL;
}

static const int op_values[POLY_OP_COUNT] = {
    [POLY_OP_SPECIAL] = 1,
    [POLY_OP_BUFFER] = 2,
    [POLY_OP_NOOP] = 3,
    [POLY_OP_REWRITE_ERROR] = 4,
    [POLY_OP_PARAM] = 5,
    [POLY_OP_FUNCTION] = 6,
    [POLY_OP_CALL] = 7,
    [POLY_OP_PROGRAM] = 8,
    [POLY_OP_LINEAR] = 9,
    [POLY_OP_SOURCE] = 10,
    [POLY_OP_BINARY] = 11,
    [POLY_OP_SINK] = 12,
    [POLY_OP_AFTER] = 13,
    [POLY_OP_GROUP] = 14,
    [POLY_OP_STACK] = 15,
    [POLY_OP_TUPLE] = 16,
    [POLY_OP_GETTUPLE] = 17,
    [POLY_OP_INDEX] = 19,
    [POLY_OP_SHRINK] = 20,
    [POLY_OP_LOAD] = 21,
    [POLY_OP_STORE] = 22,
    [POLY_OP_WMMA] = 23,
    [POLY_OP_CAST] = 24,
    [POLY_OP_BITCAST] = 25,
    [POLY_OP_EXP2] = 26,
    [POLY_OP_LOG2] = 27,
    [POLY_OP_SIN] = 28,
    [POLY_OP_SQRT] = 29,
    [POLY_OP_RECIPROCAL] = 30,
    [POLY_OP_NEG] = 31,
    [POLY_OP_TRUNC] = 32,
    [POLY_OP_ADD] = 33,
    [POLY_OP_MUL] = 34,
    [POLY_OP_SHL] = 35,
    [POLY_OP_SHR] = 36,
    [POLY_OP_CDIV] = 37,
    [POLY_OP_MAX] = 38,
    [POLY_OP_CMOD] = 39,
    [POLY_OP_CMPLT] = 40,
    [POLY_OP_CMPNE] = 41,
    [POLY_OP_CMPEQ] = 42,
    [POLY_OP_XOR] = 43,
    [POLY_OP_OR] = 44,
    [POLY_OP_AND] = 45,
    [POLY_OP_THREEFRY] = 46,
    [POLY_OP_SUB] = 47,
    [POLY_OP_FDIV] = 48,
    [POLY_OP_POW] = 49,
    [POLY_OP_FLOORDIV] = 50,
    [POLY_OP_FLOORMOD] = 51,
    [POLY_OP_WHERE] = 52,
    [POLY_OP_MULACC] = 53,
    [POLY_OP_BARRIER] = 54,
    [POLY_OP_RANGE] = 55,
    [POLY_OP_IF] = 56,
    [POLY_OP_END] = 57,
    [POLY_OP_ENDIF] = 58,
    [POLY_OP_CONST] = 60,
    [POLY_OP_CUSTOM] = 61,
    [POLY_OP_CUSTOMI] = 62,
    [POLY_OP_INS] = 63,
    [POLY_OP_CONTIGUOUS] = 64,
    [POLY_OP_CONTIGUOUS_BACKWARD] = 65,
    [POLY_OP_DETACH] = 66,
    [POLY_OP_STAGE] = 67,
    [POLY_OP_COPY] = 68,
    [POLY_OP_MSELECT] = 69,
    [POLY_OP_MSTACK] = 70,
    [POLY_OP_CUSTOM_FUNCTION] = 71,
    [POLY_OP_RESHAPE] = 72,
    [POLY_OP_PERMUTE] = 73,
    [POLY_OP_EXPAND] = 74,
    [POLY_OP_PAD] = 75,
    [POLY_OP_FLIP] = 76,
    [POLY_OP_UNSHARD] = 77,
    [POLY_OP_REDUCE] = 78,
    [POLY_OP_ALLREDUCE] = 79,
    [POLY_OP_UNIQUE] = 81,
    [POLY_OP_DEVICE] = 82,
};

int poly_op_value(PolyOps op) {
  if (op >= 0 && op < POLY_OP_COUNT && op_values[op] != 0) return op_values[op];
  return (int)op;
}

/* GroupOp bitmask construction helper */

#define OPSET(...)                                                                                 \
  opset_build((PolyOps[]){__VA_ARGS__}, sizeof((PolyOps[]){__VA_ARGS__}) / sizeof(PolyOps))

static PolyOpSet opset_build(const PolyOps *ops, int n) {
  PolyOpSet s = {{0, 0}};
  for (int i = 0; i < n; i++) {
    s.bits[ops[i] / 64] |= (uint64_t)1 << (ops[i] % 64);
  }
  return s;
}

/* GroupOp sets (mirrors tinygrad GroupOp from uop/__init__.py) */

PolyOpSet POLY_GROUP_UNARY = {{0, 0}};
PolyOpSet POLY_GROUP_BINARY = {{0, 0}};
PolyOpSet POLY_GROUP_TERNARY = {{0, 0}};
PolyOpSet POLY_GROUP_BROADCASTABLE = {{0, 0}};
PolyOpSet POLY_GROUP_ALU = {{0, 0}};
PolyOpSet POLY_GROUP_ELEMENTWISE = {{0, 0}};
PolyOpSet POLY_GROUP_MOVEMENT = {{0, 0}};
PolyOpSet POLY_GROUP_COMMUTATIVE = {{0, 0}};
PolyOpSet POLY_GROUP_ASSOCIATIVE = {{0, 0}};
PolyOpSet POLY_GROUP_IDEMPOTENT = {{0, 0}};
PolyOpSet POLY_GROUP_COMPARISON = {{0, 0}};
PolyOpSet POLY_GROUP_UNSAFEPAD = {{0, 0}};
PolyOpSet POLY_GROUP_DEFINES = {{0, 0}};
PolyOpSet POLY_GROUP_IRREDUCIBLE = {{0, 0}};

static bool g_group_ops_initialized = false;

void poly_init_group_ops(void);

__attribute__((constructor)) void poly_init_group_ops(void) {
  if (g_group_ops_initialized) return;
  g_group_ops_initialized = true;
#define SET(name, ...) name = OPSET(__VA_ARGS__)

  SET(POLY_GROUP_UNARY, POLY_OP_EXP2, POLY_OP_LOG2, POLY_OP_SIN, POLY_OP_SQRT, POLY_OP_RECIPROCAL,
      POLY_OP_NEG, POLY_OP_TRUNC);

  SET(POLY_GROUP_BINARY, POLY_OP_ADD, POLY_OP_MUL, POLY_OP_IDIV, POLY_OP_MAX, POLY_OP_MOD,
      POLY_OP_CMPLT, POLY_OP_CMPNE, POLY_OP_CMPEQ, POLY_OP_XOR, POLY_OP_SHL, POLY_OP_SHR,
      POLY_OP_OR, POLY_OP_AND, POLY_OP_THREEFRY, POLY_OP_SUB, POLY_OP_FDIV, POLY_OP_POW,
      POLY_OP_FLOORDIV, POLY_OP_FLOORMOD);

  SET(POLY_GROUP_TERNARY, POLY_OP_WHERE, POLY_OP_MULACC);

  POLY_GROUP_BROADCASTABLE = poly_opset_union(POLY_GROUP_BINARY, POLY_GROUP_TERNARY);

  POLY_GROUP_ALU =
      poly_opset_union(poly_opset_union(POLY_GROUP_UNARY, POLY_GROUP_BINARY), POLY_GROUP_TERNARY);

  POLY_GROUP_ELEMENTWISE = poly_opset_union(POLY_GROUP_ALU, OPSET(POLY_OP_CAST, POLY_OP_BITCAST));

  SET(POLY_GROUP_MOVEMENT, POLY_OP_RESHAPE, POLY_OP_EXPAND, POLY_OP_PERMUTE, POLY_OP_PAD,
      POLY_OP_SHRINK, POLY_OP_FLIP);

  SET(POLY_GROUP_COMMUTATIVE, POLY_OP_ADD, POLY_OP_MUL, POLY_OP_MAX, POLY_OP_CMPNE, POLY_OP_CMPEQ,
      POLY_OP_XOR, POLY_OP_AND, POLY_OP_OR);

  SET(POLY_GROUP_ASSOCIATIVE, POLY_OP_ADD, POLY_OP_MUL, POLY_OP_AND, POLY_OP_OR, POLY_OP_MAX);

  SET(POLY_GROUP_IDEMPOTENT, POLY_OP_OR, POLY_OP_AND, POLY_OP_MAX);

  SET(POLY_GROUP_COMPARISON, POLY_OP_CMPLT, POLY_OP_CMPNE, POLY_OP_CMPEQ);

  SET(POLY_GROUP_UNSAFEPAD, POLY_OP_RECIPROCAL, POLY_OP_LOG2, POLY_OP_EXP2, POLY_OP_IDIV,
      POLY_OP_MOD, POLY_OP_FLOORDIV, POLY_OP_FLOORMOD, POLY_OP_POW);

  SET(POLY_GROUP_DEFINES, POLY_OP_PARAM, POLY_OP_BUFFER);

  SET(POLY_GROUP_IRREDUCIBLE, POLY_OP_CONST, POLY_OP_SPECIAL, POLY_OP_RANGE, POLY_OP_PARAM);

#undef SET
}
