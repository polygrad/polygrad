"""Ground-truth generator for poly_uop_minmax parity.

Mirrors reviewed cases in tinygrad/uop/ops.py::UOp._min_max.
Values captured here are asserted verbatim in test/test_sym.c.

Run from anywhere:
  references/.venv-tinygrad-py311/bin/python test/parity_scripts/tg_minmax_gt.py

Self-bootstraps PYTHONPATH from its own location so it always loads
references/tinygrad_latest regardless of cwd or any conda-installed tinygrad.
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_TG_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", "references", "tinygrad_latest"))
assert os.path.isdir(os.path.join(_TG_ROOT, "tinygrad")), \
    f"tinygrad_latest not found at {_TG_ROOT}"
sys.path.insert(0, _TG_ROOT)

import tinygrad
_loaded = os.path.dirname(os.path.abspath(tinygrad.__file__))
assert _loaded == os.path.join(_TG_ROOT, "tinygrad"), \
    f"loaded tinygrad from {_loaded}, expected {_TG_ROOT}/tinygrad"
print(f"# tinygrad loaded from: {_loaded}")

from tinygrad.uop.ops import UOp, Ops, ParamArg
from tinygrad.dtype import dtypes

def mm(name, u):
    print("[%s] vmin=%s vmax=%s" % (name, u.vmin, u.vmax))

# === CONST / VCONST / DEFINE_VAR / BIND ===
c0  = UOp.const(dtypes.int32, 0)
c1  = UOp.const(dtypes.int32, 1)
c3  = UOp.const(dtypes.int32, 3)
c5  = UOp.const(dtypes.int32, 5)
c10 = UOp.const(dtypes.int32, 10)
c20 = UOp.const(dtypes.int32, 20)
cn1 = UOp.const(dtypes.int32, -1)
cn2 = UOp.const(dtypes.int32, -2)
cn5 = UOp.const(dtypes.int32, -5)
dv  = UOp.variable("x", 2, 7, dtype=dtypes.int32)
dvn = UOp.variable("y", -3, 4, dtype=dtypes.int32)
bv  = UOp.variable("z", -1000, 1000, dtype=dtypes.int32)

print("=== CONST / DEFINE_VAR ===")
mm("CONST 5",           c5)
mm("CONST -2",          cn2)
mm("DEFINE_VAR[2..7]",  dv)
mm("DEFINE_VAR[-3..4]", dvn)

print("=== STACK constants ===")
vc_pos = UOp(Ops.STACK, dtypes.int32.vec(3), tuple(UOp.const(dtypes.int32, x) for x in (1, 2, 3)))
vc_mix = UOp(Ops.STACK, dtypes.int32.vec(4), tuple(UOp.const(dtypes.int32, x) for x in (-2, 0, 5, 10)))
mm("STACK(1,2,3)",       vc_pos)
mm("STACK(-2,0,5,10)",   vc_mix)

print("=== RANGE / SPECIAL ===")
r10 = UOp.range(10, 0)
r1  = UOp.range(1, 1)   # degenerate (expect vmin=0 vmax=0)
r7  = UOp.range(7, 2)
mm("RANGE(10)", r10)
mm("RANGE(1)",  r1)
mm("RANGE(7)",  r7)

print("=== ADD / SUB ===")
mm("r+3",    r10 + c3)
mm("r+dv",   r10 + dv)
mm("r-3",    r10 - c3)
mm("3-r",    c3 - r10)
mm("dv-dvn", dv - dvn)
mm("dvn-dv", dvn - dv)

print("=== MUL (4-corner) ===")
mm("r*3",     r10 * c3)
mm("r*-2",    r10 * cn2)
mm("dv*3",    dv * c3)
mm("dv*dv",   dv * dv)
mm("dv*dvn",  dv * dvn)
mm("dvn*dvn", dvn * dvn)

print("=== IDIV (sign-definite only) ===")
mm("r//3",  r10 // c3)
mm("r//5",  r10 // c5)
mm("dv//3", dv // c3)
min_i64 = UOp.variable("min_i64", -(1 << 63), -(1 << 63), dtype=dtypes.int64)
mm("i64_min cdiv -1", UOp(Ops.CDIV, dtypes.int64, (min_i64, UOp.const(dtypes.int64, -1))))
mm("i64_min floordiv -1", UOp(Ops.FLOORDIV, dtypes.int64, (min_i64, UOp.const(dtypes.int64, -1))))
empty = UOp.range(0, 0)
mm("empty cdiv 3", UOp(Ops.CDIV, dtypes.int, (empty, c3)))
mm("empty floordiv 3", UOp(Ops.FLOORDIV, dtypes.int, (empty, c3)))
mm("empty floormod 3", UOp(Ops.FLOORMOD, dtypes.int, (empty, c3)))

print("=== MOD ===")
mm("r%3",   r10 % c3)
mm("r%5",   r10 % c5)
mm("dv%3",  dv % c3)
mm("dvn%3", dvn % c3)    # dvn vmin=-3, vmax=4; mod by positive const
numerator_i64 = UOp.variable("numerator_i64", 0, 1, dtype=dtypes.int64)
negative_divisor_i64 = UOp.variable("negative_divisor_i64", -(1 << 63), -1, dtype=dtypes.int64)
mm("negative i64 cmod", UOp(Ops.CMOD, dtypes.int64, (numerator_i64, negative_divisor_i64)))
mm("negative i64 floormod", UOp(Ops.FLOORMOD, dtypes.int64, (numerator_i64, negative_divisor_i64)))

print("=== SHL / SHR (const rhs) ===")
mm("r<<2",  r10 << UOp.const(dtypes.int32, 2))
mm("r>>1",  r10 >> UOp.const(dtypes.int32, 1))
mm("dv<<1", dv << UOp.const(dtypes.int32, 1))

print("=== XOR with -1 (bitwise NOT) ===")
mm("r xor -1", r10 ^ cn1)
mm("dv xor -1", dv ^ cn1)

print("=== AND (int with non-negative const) ===")
mm("r & 5",  r10 & c5)
mm("dv & 5", dv & c5)

print("=== MAX ===")
mm("max(r,5)",    r10.maximum(c5))
mm("max(dv,5)",   dv.maximum(c5))
mm("max(r,dv)",   r10.maximum(dv))
mm("max(dvn,dv)", dvn.maximum(dv))

print("=== CMPLT (bool bounds) ===")
mm("r<5",   r10 < c5)    # partially True/False
mm("r<-1",  r10 < cn1)   # statically False
mm("r<20",  r10 < c20)   # statically True
mm("5<r",   c5 < r10)

print("=== CMPNE (bool bounds) ===")
mm("r!=5",  r10.ne(c5))
mm("r!=-1", r10.ne(cn1))  # statically True
mm("r!=r",  r10.ne(r10))  # NOT statically False (tinygrad only checks point-const equality)

print("=== AND/OR on bool ===")
cond1 = r10 < c5
cond2 = r10 < c10
mm("AND(r<5, r<10)", cond1 & cond2)
mm("OR(r<5, r<-1)",  cond1 | (r10 < cn1))

print("=== VECTOR BOOL / GEP / AND ===")
f32x4 = dtypes.float32.vec(4)
bool4 = dtypes.bool.vec(4)
base = UOp(Ops.PARAM, f32x4, arg=ParamArg(0))
exponent = UOp(Ops.PARAM, f32x4, arg=ParamArg(1))
zero_vec = UOp(Ops.CONST, f32x4, arg=0.0)
base_eq = UOp(Ops.CMPEQ, bool4, (base, zero_vec))
exp_eq = UOp(Ops.CMPEQ, bool4, (exponent, zero_vec))
base_lane = UOp(Ops.GEP, dtypes.bool, (base_eq,), (0,))
exp_lane = UOp(Ops.GEP, dtypes.bool, (exp_eq,), (0,))
both = UOp(Ops.AND, dtypes.bool, (base_lane, exp_lane))
gate = UOp(Ops.CMPNE, dtypes.bool, (
    UOp(Ops.CAST, dtypes.float32, (UOp(Ops.CAST, dtypes.int32, (both,)),)),
    UOp.const(dtypes.float32, 0.0),
))
mm("CMPEQ f32x4", base_eq)
mm("GEP(CMPEQ f32x4)", base_lane)
mm("AND(GEP comparisons)", both)
mm("casted bool gate", gate)

print("=== WHERE (int branches only) ===")
mm("WHERE(r<5, 3, 5)",  cond1.where(c3, c5))
mm("WHERE(r<5, -2, 5)", cond1.where(cn2, c5))
mm("WHERE(r<5, dv, dvn)", cond1.where(dv, dvn))

print("=== CAST (monotone: float/signed int/weakint only) ===")
mm("CAST dv i32->i16", dv.cast(dtypes.int16))    # (2, 7)
mm("CAST r i32->i8",   r10.cast(dtypes.int8))    # (0, 9)
mm("CAST dv i32->f32", dv.cast(dtypes.float32))  # float — int target: (2, 7)

print("=== GEP / UNROLL / VECTORIZE / BIND ===")
vec = UOp(Ops.STACK, dtypes.int32.vec(3), src=(c3, c5, dv))
mm("STACK(3,5,dv)", vec)   # union over srcs
mm("GEP[0] of vec",     vec.gep(0))
mm("GEP[2] of vec",     vec.gep(2))

print("=== Chained (diamond stress) ===")
# (r+3)*2
mm("(r+3)*2",        (r10 + c3) * UOp.const(dtypes.int32, 2))
# ((dv*2) - dvn) + 1
mm("(dv*2-dvn)+1",   ((dv * UOp.const(dtypes.int32, 2)) - dvn) + c1)

print("=== Float fallthrough (sentinel: dtype range) ===")
# polygrad uses (INT64_MIN, INT64_MAX) sentinel; tinygrad uses Python floats.
# Just document the tinygrad value for reference — polygrad skips float bounds.
mm("CONST 1.5f", UOp.const(dtypes.float32, 1.5))
mm("CONST 0.0f", UOp.const(dtypes.float32, 0.0))
