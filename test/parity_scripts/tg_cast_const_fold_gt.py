"""Ground truth for tinygrad's CAST(CONST) constant fold.

Tinygrad source:
  - uop/symbolic.py:125: (UPat(Ops.CAST, name="root", src=(UPat.cvar("c"),)),
      lambda root, c: root.const_like(c.arg))
  - uop/ops.py:451: const_like calls UOp.const(self.dtype.base, b)
  - uop/ops.py:493-499: UOp.const calls dtype.const(b) — the normaliser
  - dtype.py:92-100: DType.const(val) returns
        ConstFloat(float(val)) if is_float else bool(val) if is_bool else int(val)

Key invariant captured here: after CAST(CONST_INT(5), float).simplify(),
the resulting CONST.arg is a *float* 5.0, NOT an int 5 wearing a float
dtype tag. Polygrad's poly_const_like initially skipped this normalisation
and produced CONST(dtype=float, arg.kind=INT, i=5) — a tagged-union
mismatch that the codegen mis-lowered to a denormal/zero. This script
documents the expected behaviour so any future regression in
poly_const_like / rule_cast_const fails the parity check immediately.

Run from anywhere:
  conda run -n tiny python /home/anton/projects/polygrad/polygrad/test/parity_scripts/tg_cast_const_fold_gt.py
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

from tinygrad.uop.ops import UOp, Ops, graph_rewrite
from tinygrad.dtype import dtypes
from tinygrad.uop.symbolic import symbolic_simple


def case(label, before):
    print(f"\n=== {label} ===")
    print("BEFORE:")
    print(f"  {before.op.name} dtype={before.dtype} arg={before.arg!r}")
    for i, c in enumerate(before.src):
        print(f"  src[{i}] {c.op.name} dtype={c.dtype} arg={c.arg!r} type={type(c.arg).__name__}")
    after = graph_rewrite(before, symbolic_simple, name=label)
    print("AFTER:")
    print(f"  {after.op.name} dtype={after.dtype} arg={after.arg!r} type={type(after.arg).__name__}")


# A: cast int CONST -> float
c5_int = UOp.const(dtypes.int32, 5)
case("A_cast_int_to_float", c5_int.cast(dtypes.float32))

# B: cast float CONST -> int (truncates)
c_pi = UOp.const(dtypes.float32, 3.7)
case("B_cast_float_to_int", c_pi.cast(dtypes.int32))

# C: cast bool CONST -> float
c_t = UOp.const(dtypes.bool, True)
case("C_cast_bool_to_float", c_t.cast(dtypes.float32))

# D: cast int CONST -> bool
c_zero_int = UOp.const(dtypes.int32, 0)
case("D_cast_zero_int_to_bool", c_zero_int.cast(dtypes.bool))

# E: the exact failing case from expand_reduce_e2e --
#    MUL(CONST_FLOAT(1.0), CAST(CONST_INT(5), float)) must fold to CONST_FLOAT(5.0)
mul = UOp.const(dtypes.float32, 1.0) * UOp.const(dtypes.int32, 5).cast(dtypes.float32)
case("E_mul_float_cast_int_to_float", mul)
