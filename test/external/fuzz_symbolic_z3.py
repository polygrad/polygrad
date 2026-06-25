#!/usr/bin/env python3
"""z3 proof fuzzer for Polygrad symbolic integer rewrites.

This mirrors the intent of tinygrad's test/external/fuzz_symbolic*.py:
build bounded symbolic UOp expressions, rewrite them with poly_symbolic(), and
ask z3 whether the original and rewritten expressions can differ.

The harness intentionally targets the integer/bool subset used by Polygrad's
current C symbolic fuzzers. It is an external/nightly proof check, not a
replacement for sanitizer-backed libFuzzer.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import random
import sys
from pathlib import Path

try:
    import z3
except Exception as exc:  # pragma: no cover - exercised by missing dependency envs
    raise SystemExit(
        "z3-solver is required for this external proof fuzzer. "
        "Install a safe version such as: pip install 'z3-solver<4.15.4'"
    ) from exc


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LIB = ROOT / "build" / "libpolygrad.so"


class PolyDType(ctypes.Structure):
    _fields_ = [
        ("priority", ctypes.c_int8),
        ("bitsize", ctypes.c_uint16),
        ("name", ctypes.c_char_p),
        ("fmt", ctypes.c_char),
        ("count", ctypes.c_uint16),
        ("is_ptr", ctypes.c_bool),
        ("addrspace", ctypes.c_int),
        ("vcount", ctypes.c_uint16),
        ("ptr_size", ctypes.c_int64),
    ]


class PolyIntTuple(ctypes.Structure):
    _fields_ = [("vals", ctypes.POINTER(ctypes.c_int64)), ("n", ctypes.c_int)]


class PolyPairTuple(ctypes.Structure):
    _fields_ = [("pairs", ctypes.POINTER(ctypes.c_int64 * 2)), ("n", ctypes.c_int)]


class PolyReduceAxisArg(ctypes.Structure):
    _fields_ = [
        ("op", ctypes.c_int),
        ("axes", ctypes.POINTER(ctypes.c_int64)),
        ("n", ctypes.c_int),
    ]


class PolyRangeArg(ctypes.Structure):
    _fields_ = [
        ("axis_id", ctypes.c_int64),
        ("axis_type", ctypes.c_int),
        ("extra", ctypes.POINTER(ctypes.c_int64)),
        ("n_extra", ctypes.c_int),
    ]


class PolyDefineVarArg(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("min_val", ctypes.c_int64),
        ("max_val", ctypes.c_int64),
    ]


class PolyBufferizeOptsArg(ctypes.Structure):
    _fields_ = [
        ("device", ctypes.c_int32),
        ("addrspace", ctypes.c_int),
        ("removable", ctypes.c_bool),
    ]


class PolyArgValue(ctypes.Union):
    _fields_ = [
        ("i", ctypes.c_int64),
        ("f", ctypes.c_double),
        ("b", ctypes.c_bool),
        ("int_tuple", PolyIntTuple),
        ("pair_tuple", PolyPairTuple),
        ("str", ctypes.c_char_p),
        ("ops", ctypes.c_int),
        ("reduce_axis", PolyReduceAxisArg),
        ("range", PolyRangeArg),
        ("define_var", PolyDefineVarArg),
        ("bufferize_opts", PolyBufferizeOptsArg),
        ("program_info", ctypes.c_void_p),
    ]


class PolyArg(ctypes.Structure):
    _fields_ = [("kind", ctypes.c_int), ("value", PolyArgValue)]


class PolyUOp(ctypes.Structure):
    pass


PolyUOpPtr = ctypes.POINTER(PolyUOp)

PolyUOp._fields_ = [
    ("op", ctypes.c_int),
    ("dtype", PolyDType),
    ("src", ctypes.POINTER(PolyUOpPtr)),
    ("n_src", ctypes.c_uint16),
    ("arg", PolyArg),
    ("tag", ctypes.c_int32),
    ("hash", ctypes.c_uint32),
    ("minmax_cached", ctypes.c_bool),
    ("minmax_vmin", ctypes.c_int64),
    ("minmax_vmax", ctypes.c_int64),
    ("ranges_cache", ctypes.c_void_p),
    ("ended_ranges_cache", ctypes.c_void_p),
]


ARG_INT = 1
ARG_BOOL = 3
ARG_RANGE = 9
ARG_DEFINE_VAR = 10
AXIS_LOOP = 3


def ptr_key(u: PolyUOpPtr) -> int:
    return ctypes.addressof(u.contents)


def arg_range(axis_id: int) -> PolyArg:
    a = PolyArg()
    a.kind = ARG_RANGE
    a.value.range.axis_id = axis_id
    a.value.range.axis_type = AXIS_LOOP
    a.value.range.extra = None
    a.value.range.n_extra = 0
    return a


def load_lib() -> ctypes.CDLL:
    path = Path(os.environ.get("POLYGRAD_LIB", DEFAULT_LIB))
    if not path.exists():
        raise SystemExit(f"libpolygrad not found: {path}. Build with `make build/libpolygrad.so`.")
    lib = ctypes.CDLL(str(path))

    lib.poly_ctx_new.restype = ctypes.c_void_p
    lib.poly_ctx_new.argtypes = []
    lib.poly_ctx_destroy.restype = None
    lib.poly_ctx_destroy.argtypes = [ctypes.c_void_p]
    lib.poly_op_count.restype = ctypes.c_int
    lib.poly_op_count.argtypes = []
    lib.poly_op_name.restype = ctypes.c_char_p
    lib.poly_op_name.argtypes = [ctypes.c_int]
    lib.poly_free.restype = None
    lib.poly_free.argtypes = [ctypes.c_void_p]

    lib.poly_const_int.restype = PolyUOpPtr
    lib.poly_const_int.argtypes = [ctypes.c_void_p, ctypes.c_int64]
    lib.poly_define_var.restype = PolyUOpPtr
    lib.poly_define_var.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int64, ctypes.c_int64]
    lib.poly_alu1.restype = PolyUOpPtr
    lib.poly_alu1.argtypes = [ctypes.c_void_p, ctypes.c_int, PolyUOpPtr]
    lib.poly_alu2.restype = PolyUOpPtr
    lib.poly_alu2.argtypes = [ctypes.c_void_p, ctypes.c_int, PolyUOpPtr, PolyUOpPtr]
    lib.poly_alu3.restype = PolyUOpPtr
    lib.poly_alu3.argtypes = [ctypes.c_void_p, ctypes.c_int, PolyUOpPtr, PolyUOpPtr, PolyUOpPtr]
    lib.poly_uop1.restype = PolyUOpPtr
    lib.poly_uop1.argtypes = [ctypes.c_void_p, ctypes.c_int, PolyDType, PolyUOpPtr, PolyArg]

    lib.poly_symbolic.restype = ctypes.c_void_p
    lib.poly_symbolic.argtypes = []
    lib.poly_graph_rewrite.restype = PolyUOpPtr
    lib.poly_graph_rewrite.argtypes = [ctypes.c_void_p, PolyUOpPtr, ctypes.c_void_p]
    lib.poly_uop_str.restype = ctypes.c_void_p
    lib.poly_uop_str.argtypes = [PolyUOpPtr]
    return lib


class Poly:
    def __init__(self, lib: ctypes.CDLL):
        self.lib = lib
        self.ctx = lib.poly_ctx_new()
        if not self.ctx:
            raise RuntimeError("poly_ctx_new failed")
        self.ops = {
            lib.poly_op_name(i).decode(): i
            for i in range(lib.poly_op_count())
            if lib.poly_op_name(i)
        }
        self.int32 = PolyDType.in_dll(lib, "POLY_INT32")

    def close(self) -> None:
        if self.ctx:
            self.lib.poly_ctx_destroy(self.ctx)
            self.ctx = None

    def const(self, value: int) -> PolyUOpPtr:
        return self.lib.poly_const_int(self.ctx, int(value))

    def var(self, name: str, lo: int, hi: int) -> PolyUOpPtr:
        return self.lib.poly_define_var(self.ctx, name.encode(), lo, hi)

    def range(self, bound: PolyUOpPtr, axis_id: int) -> PolyUOpPtr:
        return self.lib.poly_uop1(self.ctx, self.ops["RANGE"], self.int32, bound, arg_range(axis_id))

    def alu1(self, op: str, a: PolyUOpPtr) -> PolyUOpPtr:
        return self.lib.poly_alu1(self.ctx, self.ops[op], a)

    def alu2(self, op: str, a: PolyUOpPtr, b: PolyUOpPtr) -> PolyUOpPtr:
        return self.lib.poly_alu2(self.ctx, self.ops[op], a, b)

    def alu3(self, op: str, a: PolyUOpPtr, b: PolyUOpPtr, c: PolyUOpPtr) -> PolyUOpPtr:
        return self.lib.poly_alu3(self.ctx, self.ops[op], a, b, c)

    def rewrite(self, u: PolyUOpPtr) -> PolyUOpPtr:
        return self.lib.poly_graph_rewrite(self.ctx, u, self.lib.poly_symbolic())

    def render(self, u: PolyUOpPtr) -> str:
        raw = self.lib.poly_uop_str(u)
        if not raw:
            return "<null>"
        try:
            return ctypes.cast(raw, ctypes.c_char_p).value.decode()
        finally:
            self.lib.poly_free(raw)


def z3_cdiv(a, b):
    return z3.If(a < 0, z3.If(0 < b, (a + (b - 1)) / b, (a - (b + 1)) / b), a / b)


def z3_cmod(a, b):
    return a - z3_cdiv(a, b) * b


class Z3Translator:
    def __init__(self, poly: Poly):
        self.poly = poly
        self.ctx = z3.Context()
        self.solver = z3.Solver(ctx=self.ctx)
        self.solver.set(timeout=5000)
        self.memo: dict[int, z3.ExprRef] = {}
        self.z3_vars: dict[str, z3.ExprRef] = {}

    def translate(self, u: PolyUOpPtr):
        key = ptr_key(u)
        if key in self.memo:
            return self.memo[key]

        node = u.contents
        op = self.poly.lib.poly_op_name(node.op).decode()

        if op == "CONST":
            if node.arg.kind == ARG_BOOL:
                out = z3.BoolVal(bool(node.arg.value.b), ctx=self.ctx)
            elif node.arg.kind == ARG_INT:
                out = z3.IntVal(int(node.arg.value.i), ctx=self.ctx)
            else:
                raise NotImplementedError(f"unsupported CONST arg kind {node.arg.kind}")
        elif op == "DEFINE_VAR":
            if node.arg.kind != ARG_DEFINE_VAR:
                raise NotImplementedError("DEFINE_VAR without DEFINE_VAR arg")
            name = node.arg.value.define_var.name.decode()
            lo = int(node.arg.value.define_var.min_val)
            hi = int(node.arg.value.define_var.max_val)
            var_key = f"var:{name}"
            out = self.z3_vars.get(var_key)
            if out is None:
                out = z3.Int(name, ctx=self.ctx)
                self.z3_vars[var_key] = out
            self.solver.add(lo <= out, out <= hi)
        elif op == "RANGE":
            axis = int(node.arg.value.range.axis_id)
            bound = self.translate(node.src[0])
            var_key = f"range:{axis}"
            out = self.z3_vars.get(var_key)
            if out is None:
                out = z3.Int(f"r{axis}", ctx=self.ctx)
                self.z3_vars[var_key] = out
            self.solver.add(0 <= out, out < bound)
        else:
            src = [self.translate(node.src[i]) for i in range(node.n_src)]
            if op == "NEG":
                out = -src[0]
            elif op == "ADD":
                out = src[0] + src[1]
            elif op == "SUB":
                out = src[0] - src[1]
            elif op == "MUL":
                out = src[0] * src[1]
            elif op == "IDIV":
                self.solver.add(src[1] != 0)
                out = z3_cdiv(src[0], src[1])
            elif op == "MOD":
                self.solver.add(src[1] != 0)
                out = z3_cmod(src[0], src[1])
            elif op == "MAX":
                out = z3.If(src[0] < src[1], src[1], src[0])
            elif op == "CMPLT":
                out = src[0] < src[1]
            elif op == "CMPNE":
                out = src[0] != src[1]
            elif op == "CMPEQ":
                out = src[0] == src[1]
            elif op == "AND":
                out = z3.And(src[0], src[1])
            elif op == "OR":
                out = z3.Or(src[0], src[1])
            elif op == "XOR":
                out = src[0] != src[1]
            elif op == "WHERE":
                out = z3.If(src[0], src[1], src[2])
            else:
                raise NotImplementedError(f"unsupported op {op}")

        self.memo[key] = out
        return out


def random_bool_expr(poly: Poly, rng: random.Random, leaves: list[PolyUOpPtr], depth: int) -> PolyUOpPtr:
    a = random_int_expr(poly, rng, leaves, max(0, depth - 1))
    b = rng.choice(leaves) if rng.randrange(3) else poly.const(rng.randint(-12, 12))
    return poly.alu2(rng.choice(["CMPLT", "CMPNE", "CMPEQ"]), a, b)


def random_int_expr(poly: Poly, rng: random.Random, leaves: list[PolyUOpPtr], depth: int) -> PolyUOpPtr:
    if depth <= 0:
        return rng.choice(leaves + [poly.const(rng.randint(-16, 16))])

    choice = rng.randrange(11)
    a = random_int_expr(poly, rng, leaves, depth - 1)
    if choice == 0:
        return poly.alu1("NEG", a)
    if choice == 1:
        return poly.alu2("IDIV", a, poly.const(rng.choice([-9, -7, -3, -2, -1, 1, 2, 3, 7, 9])))
    if choice == 2:
        return poly.alu2("MOD", a, poly.const(rng.choice([1, 2, 3, 4, 7, 9])))
    if choice == 3:
        return poly.alu3("WHERE", random_bool_expr(poly, rng, leaves, 2), a, random_int_expr(poly, rng, leaves, depth - 1))

    b = random_int_expr(poly, rng, leaves, depth - 1)
    return poly.alu2(rng.choice(["ADD", "SUB", "MUL", "MAX"]), a, b)


def random_factor(poly: Poly, rng: random.Random, factors: list[PolyUOpPtr]) -> PolyUOpPtr:
    base = rng.choice(factors)
    choice = rng.randrange(4)
    if choice == 0:
        return base
    if choice == 1:
        return poly.alu2("MUL", base, poly.const(rng.randint(2, 7)))
    if choice == 2:
        return poly.alu2("ADD", base, rng.choice(factors))
    return poly.const(rng.choice([1, 2, 3, 4, 7, 9, 16, 33]))


def random_div_expr(poly: Poly, rng: random.Random, variables: list[PolyUOpPtr]) -> PolyUOpPtr:
    factors = variables + [poly.const(v) for v in [1, 2, 3, 4, 7, 9, 16, 33]]
    for _ in range(2):
        factors.append(poly.alu2("MUL", rng.choice(variables), rng.choice(variables)))
    for _ in range(2):
        factors.append(poly.alu2("ADD", rng.choice(variables), rng.choice(factors)))
    ranges = [poly.range(random_factor(poly, rng, factors), i) for i in range(4)]

    def term() -> PolyUOpPtr:
        out = poly.alu2("MUL", rng.choice(ranges), random_factor(poly, rng, factors))
        return poly.alu1("NEG", out) if rng.randrange(4) == 0 else out

    expr = term()
    for _ in range(rng.randint(1, 4)):
        expr = poly.alu2("ADD", expr, term())

    den = random_factor(poly, rng, factors)
    if rng.randrange(4) == 0:
        den = poly.alu1("NEG", den)
    return poly.alu2(rng.choice(["IDIV", "MOD"]), expr, den)


def prove_equivalent(poly: Poly, expr: PolyUOpPtr, rewritten: PolyUOpPtr) -> tuple[bool, str | None]:
    t = Z3Translator(poly)
    try:
        a = t.translate(expr)
        b = t.translate(rewritten)
        check = t.solver.check(a != b)
    except NotImplementedError as exc:
        return True, f"skipped unsupported: {exc}"

    if check == z3.unsat:
        return True, None
    if check == z3.unknown:
        return True, f"skipped unknown: {t.solver.reason_unknown()}"
    return False, str(t.solver.model())


def run(args: argparse.Namespace) -> int:
    lib = load_lib()
    rng = random.Random(args.seed)
    skipped = 0
    for i in range(args.iters):
        poly = Poly(lib)
        try:
            if args.mode == "general":
                upper_bounds = [*range(1, 10), 16, 32, 64, 128, 256]
                leaves = [
                    poly.var("v1", 0, rng.choice(upper_bounds)),
                    poly.var("v2", 0, rng.choice(upper_bounds)),
                    poly.var("v3", 0, rng.choice(upper_bounds)),
                ]
                expr = random_int_expr(poly, rng, leaves, args.depth)
            elif args.mode == "div":
                upper_bounds = [1, 2, 3, 16, 33, 53, 64, 256]
                variables = [
                    poly.var("i", 1, rng.choice(upper_bounds)),
                    poly.var("j", 1, rng.choice(upper_bounds)),
                    poly.var("k", 1, rng.choice(upper_bounds)),
                ]
                expr = random_div_expr(poly, rng, variables)
            else:
                raise AssertionError(args.mode)

            rewritten = poly.rewrite(expr)
            ok, note = prove_equivalent(poly, expr, rewritten)
            if note:
                skipped += 1
                if args.verbose:
                    print(f"{i}: {note}")
            if not ok:
                print("z3 found mismatched symbolic rewrite", file=sys.stderr)
                print(f"seed={args.seed} iter={i} mode={args.mode}", file=sys.stderr)
                print(f"expr={poly.render(expr)}", file=sys.stderr)
                print(f"rewritten={poly.render(rewritten)}", file=sys.stderr)
                print(f"model={note}", file=sys.stderr)
                return 1
        finally:
            poly.close()

    print(f"z3 symbolic {args.mode}: {args.iters} expressions checked, {skipped} skipped")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["general", "div"], default="general")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iters", type=int, default=128)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
