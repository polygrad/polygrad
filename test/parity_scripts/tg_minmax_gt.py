"""Ground-truth generator for poly_uop_minmax parity.

Mirrors every case in tinygrad/uop/ops.py::UOp._min_max (lines 856-897).
Values captured here are asserted verbatim in test/test_sym.c.

Run from anywhere:
  conda run -n tiny python /home/anton/projects/polygrad/polygrad/test/parity_scripts/tg_minmax_gt.py

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

from tinygrad.uop.ops import UOp, Ops
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

print("=== VCONST ===")
# VCONST: all positive lanes, mixed lanes
vc_pos = UOp(Ops.VCONST, dtypes.int32.vec(3), arg=(1, 2, 3))
vc_mix = UOp(Ops.VCONST, dtypes.int32.vec(4), arg=(-2, 0, 5, 10))
mm("VCONST(1,2,3)",       vc_pos)
mm("VCONST(-2,0,5,10)",   vc_mix)

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

print("=== MOD ===")
mm("r%3",   r10 % c3)
mm("r%5",   r10 % c5)
mm("dv%3",  dv % c3)
mm("dvn%3", dvn % c3)    # dvn vmin=-3, vmax=4; mod by positive const

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

print("=== WHERE (int branches only) ===")
mm("WHERE(r<5, 3, 5)",  cond1.where(c3, c5))
mm("WHERE(r<5, -2, 5)", cond1.where(cn2, c5))
mm("WHERE(r<5, dv, dvn)", cond1.where(dv, dvn))

print("=== CAST (monotone: float/signed int/weakint only) ===")
mm("CAST dv i32->i16", dv.cast(dtypes.int16))    # (2, 7)
mm("CAST r i32->i8",   r10.cast(dtypes.int8))    # (0, 9)
mm("CAST dv i32->f32", dv.cast(dtypes.float32))  # float — int target: (2, 7)

print("=== GEP / UNROLL / VECTORIZE / BIND ===")
vec = UOp(Ops.VECTORIZE, dtypes.int32.vec(3), src=(c3, c5, dv))
mm("VECTORIZE(3,5,dv)", vec)   # union over srcs
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
