#!/usr/bin/env python3
"""z3 proof fuzzer for Polygrad symbolic integer rewrites.

This mirrors the intent of tinygrad's test/external/fuzz_symbolic*.py:
build bounded symbolic UOp expressions, rewrite them with poly_symbolic(), and
ask z3 whether the original and rewritten expressions can differ.

The general mode models unbounded symbolic integers, the div mode models
tinygrad's weak-index FLOORDIV/FLOORMOD domain, and the fixed mode uses z3
bit-vectors for typed signed/unsigned wraparound. It is an external/nightly
proof check, not a replacement for sanitizer-backed libFuzzer or structural
tinygrad parity probes.

Fixed-width mode is deliberately stricter than the pinned symbolic interval
rules: its uint8_add_wrap_cmp case also disagrees with Tinygrad v0.14.0.
Keep that failure visible; it is not a Polygrad-only parity finding.
"""

from __future__ import annotations

import argparse
import ctypes
import itertools
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
    ]


class PolyIntTuple(ctypes.Structure):
    _fields_ = [("vals", ctypes.POINTER(ctypes.c_int64)), ("n", ctypes.c_int)]


class PolyReduceArg(ctypes.Structure):
    _fields_ = [
        ("op", ctypes.c_int),
        ("num_axes", ctypes.c_int),
    ]


class PolyRangeArg(ctypes.Structure):
    _fields_ = [
        ("axis_id", ctypes.c_int64),
        ("axis_type", ctypes.c_int),
        ("extra", ctypes.POINTER(ctypes.c_int64)),
        ("n_extra", ctypes.c_int),
    ]


class PolyParamArg(ctypes.Structure):
    pass

class PolyBufferizeOptsArg(ctypes.Structure):
    _fields_ = [
        ("device", ctypes.c_char_p),
        ("devices", ctypes.POINTER(ctypes.c_char_p)),
        ("n_devices", ctypes.c_int32),
        ("device_is_tuple", ctypes.c_bool),
        ("addrspace", ctypes.c_int),
        ("removable", ctypes.c_bool),
        ("device_is_int", ctypes.c_bool),
        ("device_int", ctypes.c_int64),
    ]


class PolyBigInt(ctypes.Structure):
    _fields_ = [("sign", ctypes.c_int8), ("n_limbs", ctypes.c_uint32),
                ("limbs", ctypes.POINTER(ctypes.c_uint32))]


class PolyTensorCoreArg(ctypes.Structure):
    # Largest union arm; it determines PolyArg's by-value ABI even though this
    # integer fuzzer never constructs WMMA. No opaque padding guesses.
    _fields_ = [("dims", ctypes.c_int * 3), ("dtype_in", PolyDType),
                ("device", ctypes.c_char_p), ("threads", ctypes.c_int),
                ("upcast_axes", ctypes.POINTER(ctypes.c_int64 * 2) * 3),
                ("n_upcast_axes", ctypes.c_int * 3), ("has_upcast_axes", ctypes.c_bool)]


class PolyArgValue(ctypes.Union):
    _fields_ = [
        ("i", ctypes.c_int64),
        ("f", ctypes.c_double),
        ("b", ctypes.c_bool),
        ("int_tuple", PolyIntTuple),
        ("str", ctypes.c_char_p),
        ("ops", ctypes.c_int),
        ("reduce", PolyReduceArg),
        ("range", PolyRangeArg),
        ("param", ctypes.POINTER(PolyParamArg)),
        ("bufferize_opts", PolyBufferizeOptsArg),
        ("program_info", ctypes.c_void_p),
        ("bigint", PolyBigInt),
        ("tensor_core", PolyTensorCoreArg),
    ]


class PolyArg(ctypes.Structure):
    _fields_ = [("kind", ctypes.c_int), ("value", PolyArgValue)]


PolyParamArg._fields_ = [
    ("slot", ctypes.c_int64),
    ("dtype", PolyDType),
    ("name", ctypes.c_char_p),
    ("min_val", PolyArg),
    ("max_val", PolyArg),
    ("has_minmax", ctypes.c_bool),
    ("multiple_of", ctypes.c_int64),
    ("has_multiple_of", ctypes.c_bool),
    ("addrspace", ctypes.c_int),
    ("axis", ctypes.c_int32),
    ("has_axis", ctypes.c_bool),
    ("device", ctypes.c_char_p),
    ("devices", ctypes.POINTER(ctypes.c_char_p)),
    ("n_devices", ctypes.c_int32),
    ("device_is_tuple", ctypes.c_bool),
    ("volatile_", ctypes.c_bool),
]


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
    ("tag_arg", PolyArg),
    ("hash", ctypes.c_uint32),
    ("addrspace_cached", ctypes.c_bool),
    ("addrspace_cache", ctypes.c_int),
    ("minmax_cached", ctypes.c_bool),
    ("minmax_vmin", ctypes.c_int64),
    ("minmax_vmax", ctypes.c_int64),
    ("ranges_cache", ctypes.c_void_p),
    ("ended_ranges_cache", ctypes.c_void_p),
]


ARG_INT = 1
ARG_BOOL = 3
ARG_RANGE = 9
ARG_PARAM = 15
ARG_BIGINT = 16
AXIS_LOOP = 11


def integer_arg(arg: PolyArg) -> int:
    if arg.kind == ARG_INT:
        return int(arg.value.i)
    if arg.kind == ARG_BOOL:
        return int(arg.value.b)
    if arg.kind == ARG_BIGINT:
        big = arg.value.bigint
        return int(big.sign) * sum(int(big.limbs[i]) << (32 * i) for i in range(big.n_limbs))
    raise NotImplementedError(f"non-integer bound/literal kind {arg.kind}")


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


def arg_none() -> PolyArg:
    a = PolyArg()
    a.kind = 0
    return a


def arg_int(value: int) -> PolyArg:
    a = PolyArg()
    value = int(value)
    if -(1 << 63) <= value < (1 << 63):
        a.kind = ARG_INT
        a.value.i = value
    else:
        magnitude = abs(value)
        count = (magnitude.bit_length() + 31) // 32
        limbs = (ctypes.c_uint32 * count)(*((magnitude >> (32*i)) & 0xffffffff for i in range(count)))
        a.kind = ARG_BIGINT
        a.value.bigint = PolyBigInt(-1 if value < 0 else 1, count, limbs)
        a._limbs = limbs  # Keep caller storage alive until the core copies it.
    return a


def load_lib() -> ctypes.CDLL:
    path = Path(os.environ.get("POLYGRAD_LIB", DEFAULT_LIB))
    if not path.exists():
        raise SystemExit(f"libpolygrad not found: {path}. Build with `make build/libpolygrad.so`.")
    lib = ctypes.CDLL(str(path))
    # These raw layouts and enum values are tied to this ABI. A new core must
    # stop here until the declarations are reviewed, not corrupt ctypes calls.
    lib.poly_abi_version.restype = ctypes.c_int
    lib.poly_abi_version.argtypes = []
    if (abi := lib.poly_abi_version()) != 79:
        raise SystemExit(f"Z3 harness requires reviewed ABI79 layouts; core has ABI{abi}")

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
    lib.poly_dtype_id_by_name.restype = ctypes.c_int
    lib.poly_dtype_id_by_name.argtypes = [ctypes.c_char_p]
    lib.poly_uop_variable_by_id.restype = PolyUOpPtr
    lib.poly_uop_variable_by_id.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, PolyUOpPtr, PolyUOpPtr,
        ctypes.c_int, ctypes.c_int64, ctypes.c_bool,
    ]
    lib.poly_alu1.restype = PolyUOpPtr
    lib.poly_alu1.argtypes = [ctypes.c_void_p, ctypes.c_int, PolyUOpPtr]
    lib.poly_alu2.restype = PolyUOpPtr
    lib.poly_alu2.argtypes = [ctypes.c_void_p, ctypes.c_int, PolyUOpPtr, PolyUOpPtr]
    lib.poly_alu3.restype = PolyUOpPtr
    lib.poly_alu3.argtypes = [ctypes.c_void_p, ctypes.c_int, PolyUOpPtr, PolyUOpPtr, PolyUOpPtr]
    lib.poly_uop1.restype = PolyUOpPtr
    lib.poly_uop1.argtypes = [ctypes.c_void_p, ctypes.c_int, PolyDType, PolyUOpPtr, PolyArg]
    lib.poly_uop0.restype = PolyUOpPtr
    lib.poly_uop0.argtypes = [ctypes.c_void_p, ctypes.c_int, PolyDType, PolyArg]
    lib.poly_uop2.restype = PolyUOpPtr
    lib.poly_uop2.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        PolyDType,
        PolyUOpPtr,
        PolyUOpPtr,
        PolyArg,
    ]

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
        self.weakint = PolyDType.in_dll(lib, "POLY_WEAKINT")
        self.fixed_dtypes = {
            name: PolyDType.in_dll(lib, f"POLY_{name.upper()}")
            for name in ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64")
        }

    def close(self) -> None:
        if self.ctx:
            self.lib.poly_ctx_destroy(self.ctx)
            self.ctx = None

    def const(self, value: int) -> PolyUOpPtr:
        return self.lib.poly_const_int(self.ctx, int(value))

    def const_typed(self, dtype: PolyDType, value: int) -> PolyUOpPtr:
        return self.lib.poly_uop0(self.ctx, self.ops["CONST"], dtype, arg_int(value))

    def var(self, name: str, lo: int, hi: int) -> PolyUOpPtr:
        return self.var_typed(self.weakint, name, lo, hi)

    def var_typed(self, dtype: PolyDType, name: str, lo: int, hi: int) -> PolyUOpPtr:
        dtype_id = self.lib.poly_dtype_id_by_name(dtype_name(dtype).encode())
        return self.lib.poly_uop_variable_by_id(
            self.ctx, name.encode(), self.const_typed(self.weakint, lo),
            self.const_typed(self.weakint, hi), dtype_id, 1, False,
        )

    def range(self, bound: PolyUOpPtr, axis_id: int, dtype: PolyDType | None = None) -> PolyUOpPtr:
        return self.lib.poly_uop1(
            self.ctx,
            self.ops["RANGE"],
            self.int32 if dtype is None else dtype,
            bound,
            arg_range(axis_id),
        )

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

    def render_tree(self, u: PolyUOpPtr, max_depth: int = 12) -> str:
        lines: list[str] = []

        def visit(node_ptr: PolyUOpPtr, depth: int) -> None:
            node = node_ptr.contents
            op = self.lib.poly_op_name(node.op).decode()
            arg = ""
            if node.arg.kind == ARG_INT:
                arg = f" arg={node.arg.value.i}"
            elif node.arg.kind == ARG_BOOL:
                arg = f" arg={bool(node.arg.value.b)}"
            elif node.arg.kind == ARG_PARAM and node.arg.value.param:
                var = node.arg.value.param.contents
                arg = f" arg=({var.name.decode()},{integer_arg(var.min_val)},{integer_arg(var.max_val)})"
            lines.append(f"{'  ' * depth}{op}:{dtype_name(node.dtype)}{arg}")
            if depth >= max_depth:
                if node.n_src:
                    lines.append(f"{'  ' * (depth + 1)}...")
                return
            for source_index in range(node.n_src):
                visit(node.src[source_index], depth + 1)

        visit(u, 0)
        return "\n".join(lines)


def z3_cdiv(a, b):
    return z3.If(a < 0, z3.If(0 < b, (a + (b - 1)) / b, (a - (b + 1)) / b), a / b)


def z3_cmod(a, b):
    return a - z3_cdiv(a, b) * b


def z3_floordiv(a, b):
    trunc = z3_cdiv(a, b)
    rem = a - trunc * b
    return trunc - z3.If(z3.And(rem != 0, (a < 0) != (b < 0)), 1, 0)


def z3_floormod(a, b):
    return a - z3_floordiv(a, b) * b


def dtype_name(dtype: PolyDType) -> str:
    return dtype.name.decode() if dtype.name else ""


def dtype_is_unsigned(dtype: PolyDType) -> bool:
    return dtype.priority in (2, 4, 6, 8)


def dtype_is_bool(dtype: PolyDType) -> bool:
    return dtype.priority == 0 and dtype.bitsize == 1


def bv_signed_value(value, dtype: PolyDType):
    return z3.BV2Int(value, is_signed=not dtype_is_unsigned(dtype))


def bv_cdiv(a, b, dtype: PolyDType):
    return z3.UDiv(a, b) if dtype_is_unsigned(dtype) else a / b


def bv_cmod(a, b, dtype: PolyDType):
    return z3.URem(a, b) if dtype_is_unsigned(dtype) else z3.SRem(a, b)


def bv_floordiv(a, b, dtype: PolyDType):
    if dtype_is_unsigned(dtype):
        return z3.UDiv(a, b)
    trunc = a / b
    rem = z3.SRem(a, b)
    zero = z3.BitVecVal(0, dtype.bitsize, ctx=a.ctx)
    one = z3.BitVecVal(1, dtype.bitsize, ctx=a.ctx)
    adjust = z3.And(rem != zero, (a < zero) != (b < zero))
    return trunc - z3.If(adjust, one, zero)


def bv_floormod(a, b, dtype: PolyDType):
    return a - bv_floordiv(a, b, dtype) * b


def bv_cast(value, source_dtype: PolyDType, target_dtype: PolyDType):
    """Model tinygrad/C integer CAST with fixed-width bit-vector semantics."""
    source_bool, target_bool = dtype_is_bool(source_dtype), dtype_is_bool(target_dtype)
    if source_bool:
        if target_bool:
            return value
        return z3.If(
            value,
            z3.BitVecVal(1, target_dtype.bitsize, ctx=value.ctx),
            z3.BitVecVal(0, target_dtype.bitsize, ctx=value.ctx),
        )
    if target_bool:
        return value != z3.BitVecVal(0, source_dtype.bitsize, ctx=value.ctx)

    source_bits, target_bits = source_dtype.bitsize, target_dtype.bitsize
    if target_bits == source_bits:
        return value
    if target_bits < source_bits:
        return z3.Extract(target_bits - 1, 0, value)
    extension = target_bits - source_bits
    return z3.ZeroExt(extension, value) if dtype_is_unsigned(source_dtype) else z3.SignExt(extension, value)


class Z3Translator:
    def __init__(self, poly: Poly, fixed_width: bool = False):
        self.poly = poly
        self.fixed_width = fixed_width
        self.ctx = z3.Context()
        self.solver = z3.Solver(ctx=self.ctx)
        self.solver.set(timeout=5000)
        self.memo: dict[int, z3.ExprRef] = {}
        self.z3_vars: dict[str, z3.ExprRef] = {}
        self.finite_domains: dict[str, tuple[z3.ExprRef, tuple[int, ...]]] = {}
        self.finite_domains_complete = True

    def translate(self, u: PolyUOpPtr):
        key = ptr_key(u)
        if key in self.memo:
            return self.memo[key]

        node = u.contents
        op = self.poly.lib.poly_op_name(node.op).decode()

        if op == "CONST":
            if node.arg.kind == ARG_BOOL:
                out = z3.BoolVal(bool(node.arg.value.b), ctx=self.ctx)
            elif node.arg.kind in (ARG_INT, ARG_BIGINT):
                out = (
                    z3.BitVecVal(integer_arg(node.arg), node.dtype.bitsize, ctx=self.ctx)
                    if self.fixed_width and not dtype_is_bool(node.dtype)
                    else z3.IntVal(integer_arg(node.arg), ctx=self.ctx)
                )
            else:
                raise NotImplementedError(f"unsupported CONST arg kind {node.arg.kind}")
        elif op in ("BUFFER", "PARAM") and node.arg.kind == ARG_PARAM and node.arg.value.param:
            param = node.arg.value.param.contents
            if not param.has_minmax or param.addrspace != 3:
                raise NotImplementedError(f"non-variable {op}")
            name = param.name.decode()
            lo = integer_arg(param.min_val)
            hi = integer_arg(param.max_val)
            var_key = f"var:{name}:{dtype_name(node.dtype)}"
            out = self.z3_vars.get(var_key)
            if out is None:
                out = (
                    z3.BitVec(name, node.dtype.bitsize, ctx=self.ctx)
                    if self.fixed_width
                    else z3.Int(name, ctx=self.ctx)
                )
                self.z3_vars[var_key] = out
            if self.fixed_width and hi - lo <= 64:
                # Fixed-mode boundary ranges are deliberately small. State the
                # exact finite bit-vector domain instead of routing through
                # BV2Int, which makes otherwise tiny division/modulo proofs
                # needlessly expensive while representing the same values.
                self.solver.add(
                    z3.Or(
                        *(
                            out == z3.BitVecVal(value, node.dtype.bitsize, ctx=self.ctx)
                            for value in range(lo, hi + 1)
                        )
                    )
                )
                values = tuple(range(lo, hi + 1))
                if var_key in self.finite_domains:
                    previous = self.finite_domains[var_key][1]
                    values = tuple(value for value in previous if lo <= value <= hi)
                self.finite_domains[var_key] = (out, values)
            else:
                if self.fixed_width:
                    self.finite_domains_complete = False
                bounded = bv_signed_value(out, node.dtype) if self.fixed_width else out
                self.solver.add(lo <= bounded, bounded <= hi)
        elif op == "RANGE":
            if self.fixed_width:
                raise NotImplementedError("fixed-width RANGE")
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
            elif op in ("CDIV", "IDIV"):
                self.solver.add(src[1] != 0)
                out = bv_cdiv(src[0], src[1], node.dtype) if self.fixed_width else z3_cdiv(src[0], src[1])
            elif op in ("CMOD", "MOD"):
                self.solver.add(src[1] != 0)
                out = bv_cmod(src[0], src[1], node.dtype) if self.fixed_width else z3_cmod(src[0], src[1])
            elif op == "FLOORDIV":
                self.solver.add(src[1] != 0)
                out = (
                    bv_floordiv(src[0], src[1], node.dtype)
                    if self.fixed_width
                    else z3_floordiv(src[0], src[1])
                )
            elif op == "FLOORMOD":
                self.solver.add(src[1] != 0)
                out = (
                    bv_floormod(src[0], src[1], node.dtype)
                    if self.fixed_width
                    else z3_floormod(src[0], src[1])
                )
            elif op == "SHL":
                out = src[0] << src[1]
            elif op == "SHR":
                out = z3.LShR(src[0], src[1]) if dtype_is_unsigned(node.dtype) else src[0] >> src[1]
            elif op == "MAX":
                less = (
                    z3.ULT(src[0], src[1])
                    if self.fixed_width and dtype_is_unsigned(node.dtype)
                    else src[0] < src[1]
                )
                out = z3.If(less, src[1], src[0])
            elif op == "CMPLT":
                src_dtype = node.src[0].contents.dtype
                out = (
                    z3.ULT(src[0], src[1])
                    if self.fixed_width and dtype_is_unsigned(src_dtype)
                    else src[0] < src[1]
                )
            elif op == "CMPNE":
                out = src[0] != src[1]
            elif op == "CMPEQ":
                out = src[0] == src[1]
            elif op == "AND":
                out = z3.And(src[0], src[1]) if z3.is_bool(src[0]) else src[0] & src[1]
            elif op == "OR":
                out = z3.Or(src[0], src[1]) if z3.is_bool(src[0]) else src[0] | src[1]
            elif op == "XOR":
                out = src[0] != src[1] if z3.is_bool(src[0]) else src[0] ^ src[1]
            elif op == "WHERE":
                out = z3.If(src[0], src[1], src[2])
            elif op == "CAST" and self.fixed_width:
                if node.n_src != 1:
                    raise NotImplementedError("malformed fixed-width CAST")
                out = bv_cast(src[0], node.src[0].contents.dtype, node.dtype)
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
        return poly.alu2("CDIV", a, poly.const(rng.choice([-9, -7, -3, -2, -1, 1, 2, 3, 7, 9])))
    if choice == 2:
        return poly.alu2("CMOD", a, poly.const(rng.choice([1, 2, 3, 4, 7, 9])))
    if choice == 3:
        return poly.alu3("WHERE", random_bool_expr(poly, rng, leaves, 2), a, random_int_expr(poly, rng, leaves, depth - 1))

    b = random_int_expr(poly, rng, leaves, depth - 1)
    return poly.alu2(rng.choice(["ADD", "SUB", "MUL", "MAX"]), a, b)


def random_factor(
    poly: Poly,
    rng: random.Random,
    factors: list[PolyUOpPtr],
    dtype: PolyDType | None = None,
) -> PolyUOpPtr:
    base = rng.choice(factors)
    choice = rng.randrange(4)
    if choice == 0:
        return base
    if choice == 1:
        const = poly.const(rng.randint(2, 7)) if dtype is None else poly.const_typed(dtype, rng.randint(2, 7))
        return poly.alu2("MUL", base, const)
    if choice == 2:
        return poly.alu2("ADD", base, rng.choice(factors))
    value = rng.choice([1, 2, 3, 4, 7, 9, 16, 33])
    return poly.const(value) if dtype is None else poly.const_typed(dtype, value)


def random_div_expr(poly: Poly, rng: random.Random, variables: list[PolyUOpPtr]) -> PolyUOpPtr:
    factors = variables + [poly.const_typed(poly.weakint, v) for v in [1, 2, 3, 4, 7, 9, 16, 33]]
    for _ in range(2):
        factors.append(poly.alu2("MUL", rng.choice(variables), rng.choice(variables)))
    for _ in range(2):
        factors.append(poly.alu2("ADD", rng.choice(variables), rng.choice(factors)))
    ranges = [
        poly.range(random_factor(poly, rng, factors, poly.weakint), i, poly.weakint)
        for i in range(4)
    ]

    def term() -> PolyUOpPtr:
        out = poly.alu2("MUL", rng.choice(ranges), random_factor(poly, rng, factors, poly.weakint))
        return poly.alu1("NEG", out) if rng.randrange(4) == 0 else out

    expr = term()
    for _ in range(rng.randint(1, 4)):
        expr = poly.alu2("ADD", expr, term())

    den = random_factor(poly, rng, factors, poly.weakint)
    if rng.randrange(4) == 0:
        den = poly.alu1("NEG", den)
    return poly.alu2(rng.choice(["FLOORDIV", "FLOORMOD"]), expr, den)


FIXED_RANGES = {
    "int8": [(-128, -121), (-5, 5), (120, 127)],
    "uint8": [(0, 7), (120, 127), (248, 255)],
    "int16": [(-(1 << 15), -(1 << 15) + 7), (-5, 5), ((1 << 15) - 8, (1 << 15) - 1)],
    "uint16": [(0, 7), ((1 << 15) - 8, (1 << 15) - 1), ((1 << 16) - 8, (1 << 16) - 1)],
    "int32": [(-(1 << 31), -(1 << 31) + 7), (-5, 5), ((1 << 31) - 8, (1 << 31) - 1)],
    "uint32": [(0, 7), ((1 << 31) - 8, (1 << 31) - 1), ((1 << 32) - 8, (1 << 32) - 1)],
    "int64": [(-(1 << 63), -(1 << 63) + 7), (-5, 5), ((1 << 63) - 8, (1 << 63) - 1)],
    "uint64": [(0, 7), ((1 << 31) - 8, (1 << 31) - 1), ((1 << 63) - 8, (1 << 63) - 1)],
}


def fixed_regressions(poly: Poly) -> list[tuple[str, PolyUOpPtr]]:
    u8, i8 = poly.fixed_dtypes["uint8"], poly.fixed_dtypes["int8"]
    cases: list[tuple[str, PolyUOpPtr]] = []

    u8_add = poly.alu2("ADD", poly.var_typed(u8, "u8_add", 250, 251), poly.const_typed(u8, 10))
    cases.append(("uint8_add_wrap_cmp", poly.alu2("CMPLT", u8_add, poly.const_typed(u8, 5))))

    i8_add = poly.alu2("ADD", poly.var_typed(i8, "i8_add", 120, 121), poly.const_typed(i8, 10))
    cases.append(("int8_add_wrap_cmp", poly.alu2("CMPLT", i8_add, poly.const_typed(i8, 0))))

    u8_sub = poly.alu2("SUB", poly.var_typed(u8, "u8_sub", 0, 1), poly.const_typed(u8, 2))
    cases.append(("uint8_sub_wrap_cmp", poly.alu2("CMPLT", u8_sub, poly.const_typed(u8, 5))))

    i8_mul = poly.alu2("MUL", poly.var_typed(i8, "i8_mul", 64, 65), poly.const_typed(i8, 2))
    cases.append(("int8_mul_wrap_cmp", poly.alu2("CMPLT", i8_mul, poly.const_typed(i8, 0))))

    u8_shl = poly.alu2("SHL", poly.var_typed(u8, "u8_shl", 128, 129), poly.const_typed(u8, 1))
    cases.append(("uint8_shl_wrap_cmp", poly.alu2("CMPLT", u8_shl, poly.const_typed(u8, 1))))

    for name, dtype in poly.fixed_dtypes.items():
        x = poly.var_typed(dtype, f"{name}_all_ones", 2, 3)
        numerator = poly.alu2("SHR", x, poly.const_typed(dtype, 1))
        all_ones = poly.const_typed(dtype, -1)
        cases.append((f"{name}_cdiv_all_ones", poly.alu2("CDIV", numerator, all_ones)))
        cases.append((f"{name}_cmod_all_ones", poly.alu2("CMOD", numerator, all_ones)))
    return cases


def random_fixed_expr(poly: Poly, rng: random.Random, iteration: int, depth: int) -> PolyUOpPtr:
    name = rng.choice(tuple(poly.fixed_dtypes))
    dtype = poly.fixed_dtypes[name]
    lo0, hi0 = rng.choice(FIXED_RANGES[name])
    lo1, hi1 = rng.choice(FIXED_RANGES[name])
    leaves = [
        poly.var_typed(dtype, f"x{iteration}", lo0, hi0),
        poly.var_typed(dtype, f"y{iteration}", lo1, hi1),
    ]
    const_values = [0, 1, 2, 3, 7, 9] if dtype_is_unsigned(dtype) else [-9, -3, -2, -1, 0, 1, 2, 3, 7, 9]
    denominator_values = [1, 2, 3, 7, 9] if dtype_is_unsigned(dtype) else [-9, -3, -2, -1, 1, 2, 3, 7, 9]

    def const() -> PolyUOpPtr:
        # Random expressions use canonical source constants. A negative raw
        # integer in an unsigned CONST is a useful all-ones shorthand, but the
        # pinned tinygrad constant folder does not normalize such operands even
        # though runtime kernels do. fixed_regressions() covers that deliberate
        # encoding separately at every width.
        return poly.const_typed(dtype, rng.choice(const_values))

    def integer(level: int) -> PolyUOpPtr:
        if level <= 0:
            return rng.choice(leaves + [const()])
        choice = rng.randrange(9)
        a = integer(level - 1)
        if choice == 0:
            return poly.alu1("NEG", a)
        if choice == 1:
            return poly.alu2("SHL", a, poly.const_typed(dtype, rng.choice([1, 2, 3])))
        if choice in (2, 3, 4, 5):
            den = poly.const_typed(dtype, rng.choice(denominator_values))
            return poly.alu2(("CDIV", "CMOD", "FLOORDIV", "FLOORMOD")[choice - 2], a, den)
        b = integer(level - 1)
        return poly.alu2(("ADD", "SUB", "MUL")[choice - 6], a, b)

    a = integer(depth)
    if rng.randrange(2):
        return a
    b = integer(max(0, depth - 1))
    return poly.alu2(rng.choice(["CMPLT", "CMPNE", "CMPEQ"]), a, b)


def prove_equivalent(
    poly: Poly,
    expr: PolyUOpPtr,
    rewritten: PolyUOpPtr,
    *,
    fixed_width: bool = False,
) -> tuple[bool, str | None]:
    t = Z3Translator(poly, fixed_width=fixed_width)
    try:
        a = t.translate(expr)
        b = t.translate(rewritten)
    except NotImplementedError as exc:
        return True, f"skipped unsupported: {exc}"

    if a.sort() != b.sort():
        return False, f"Z3 sort mismatch: original={a.sort()}, rewritten={b.sort()}"
    a = z3.simplify(a)
    b = z3.simplify(b)

    if fixed_width and t.finite_domains_complete:
        domains = list(t.finite_domains.values())
        for values in itertools.product(*(domain for _, domain in domains)):
            substitutions = tuple(
                (var, z3.BitVecVal(value, var.size(), ctx=t.ctx))
                for (var, _), value in zip(domains, values)
            )
            constraints = [
                z3.simplify(z3.substitute(assertion, *substitutions))
                for assertion in t.solver.assertions()
            ]
            if any(z3.is_false(constraint) for constraint in constraints):
                continue
            ground_a = z3.simplify(z3.substitute(a, *substitutions))
            ground_b = z3.simplify(z3.substitute(b, *substitutions))
            equal = z3.simplify(ground_a == ground_b)
            if z3.is_true(equal):
                continue

            ground_solver = z3.Solver(ctx=t.ctx)
            ground_solver.set(timeout=5000)
            ground_solver.add(*constraints, ground_a != ground_b)
            ground_check = ground_solver.check()
            assignment = ", ".join(
                f"{var}={value}" for (var, _), value in zip(domains, values)
            )
            if ground_check == z3.sat:
                return False, assignment
            if ground_check == z3.unknown:
                return True, f"skipped unknown for {assignment}: {ground_solver.reason_unknown()}"
        return True, None

    check = t.solver.check(a != b)

    if check == z3.unsat:
        return True, None
    if check == z3.unknown:
        primary_reason = t.solver.reason_unknown()
        goal = z3.Goal(ctx=t.ctx, models=True)
        goal.add(*t.solver.assertions(), a != b)
        tactic = z3.TryFor(
            z3.Then(
                z3.Tactic("simplify", ctx=t.ctx),
                z3.Tactic("solve-eqs", ctx=t.ctx),
                z3.Tactic("smt", ctx=t.ctx),
            ),
            5000,
            ctx=t.ctx,
        )
        try:
            subgoals = tactic(goal)
        except z3.Z3Exception as exc:
            return True, f"skipped unknown: {primary_reason}; tactic: {exc}"

        for subgoal in subgoals:
            if subgoal.inconsistent():
                continue
            subsolver = z3.Solver(ctx=t.ctx)
            subsolver.set(timeout=5000)
            subsolver.add(subgoal.as_expr())
            subcheck = subsolver.check()
            if subcheck == z3.sat:
                return False, str(subgoal.convert_model(subsolver.model()))
            if subcheck == z3.unknown:
                return True, (
                    f"skipped unknown: {primary_reason}; tactic subgoal: "
                    f"{subsolver.reason_unknown()}"
                )
        return True, None
    return False, str(t.solver.model())


def run(args: argparse.Namespace) -> int:
    lib = load_lib()
    poly = Poly(lib)
    try:
        for value in (2**64-1, 2**130, -2**130):
            variable = poly.var(f"abi_{value}", value, value)
            param = variable.contents.arg.value.param.contents
            assert integer_arg(param.min_val) == integer_arg(param.max_val) == value
        ranged = poly.range(poly.const(4), 17)
        assert ranged.contents.arg.kind == ARG_RANGE
        assert ranged.contents.arg.value.range.axis_id == 17
        assert ranged.contents.arg.value.range.axis_type == AXIS_LOOP
    finally:
        poly.close()
    print("Z3 harness ABI79: typed scalar and RANGE controls pass")
    rng = random.Random(args.seed)
    skipped = 0
    checked = 0

    if args.mode == "fixed":
        poly = Poly(lib)
        try:
            for label, expr in fixed_regressions(poly):
                rewritten = poly.rewrite(expr)
                ok, note = prove_equivalent(poly, expr, rewritten, fixed_width=True)
                checked += 1
                if ok and note:
                    skipped += 1
                    print(f"fixed-width proof did not complete: case={label}: {note}", file=sys.stderr)
                    print(f"expr:\n{poly.render_tree(expr)}", file=sys.stderr)
                    print(f"rewritten:\n{poly.render_tree(rewritten)}", file=sys.stderr)
                    return 2
                if not ok:
                    print("z3 found mismatched fixed-width symbolic rewrite", file=sys.stderr)
                    print(f"case={label}", file=sys.stderr)
                    print(f"expr:\n{poly.render_tree(expr)}", file=sys.stderr)
                    print(f"rewritten:\n{poly.render_tree(rewritten)}", file=sys.stderr)
                    print(f"model={note}", file=sys.stderr)
                    return 1
        finally:
            poly.close()

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
                    poly.var_typed(poly.weakint, "i", 1, rng.choice(upper_bounds)),
                    poly.var_typed(poly.weakint, "j", 1, rng.choice(upper_bounds)),
                    poly.var_typed(poly.weakint, "k", 1, rng.choice(upper_bounds)),
                ]
                expr = random_div_expr(poly, rng, variables)
            elif args.mode == "fixed":
                expr = random_fixed_expr(poly, rng, i, min(args.depth, 4))
            else:
                raise AssertionError(args.mode)

            rewritten = poly.rewrite(expr)
            ok, note = prove_equivalent(poly, expr, rewritten, fixed_width=args.mode == "fixed")
            checked += 1
            if ok and note:
                skipped += 1
                if args.mode == "fixed":
                    print(
                        f"fixed-width proof did not complete: seed={args.seed} iter={i}: {note}",
                        file=sys.stderr,
                    )
                    print(f"expr:\n{poly.render_tree(expr)}", file=sys.stderr)
                    print(f"rewritten:\n{poly.render_tree(rewritten)}", file=sys.stderr)
                    return 2
                if args.verbose:
                    print(f"{i}: {note}")
                    print(f"expr:\n{poly.render_tree(expr)}")
                    print(f"rewritten:\n{poly.render_tree(rewritten)}")
            if not ok:
                print("z3 found mismatched symbolic rewrite", file=sys.stderr)
                print(f"seed={args.seed} iter={i} mode={args.mode}", file=sys.stderr)
                print(f"expr:\n{poly.render_tree(expr)}", file=sys.stderr)
                print(f"rewritten:\n{poly.render_tree(rewritten)}", file=sys.stderr)
                print(f"model={note}", file=sys.stderr)
                return 1
        finally:
            poly.close()

    print(f"z3 symbolic {args.mode}: {checked} expressions checked, {skipped} skipped")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["general", "div", "fixed"], default="general")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iters", type=int, default=128)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
