"""Ground truth for tinygrad's pm_reduce_collapse rules + reduce_collapse driver.

Tinygrad source: codegen/simplify.py:94-149

  pm_reduce_collapse = pm_reduce_unparented + PatternMatcher([
    # 1. lift_add_from_cmplt: ((x+y).or_casted() < c) -> x < (c.cast(y.dtype)-y)
    # 2. lift_mul_from_cmplt: ((x*y) < c) -> x < ((c+y-1)//y)
    # 3. fold_range_below:    ((r<cut).where(0,val)).reduce_add(r) -> ...
    # 4. fold_range_two_sided: ((r>=lower)&(r<upper)).where(val,0).reduce_add(r) -> ...
    # 5. fold_range_above:    ((r<cut).where(val,0)).reduce_add(r) -> ...
    # 6. reduce_add_distribute: (x+y).reduce_add(*r) -> x.reduce_add(*r) + y.reduce_add(*r)
    # 7. and_on_where:        ((DEFINE_VAR & y).where(c,0)).reduce_add(*r) -> ...
    # 8. mul_casted_bool:     x * gate.cast() -> gate.where(x, 0)
  ]) + symbolic

  reduce_collapse(red, u, pm) walks each reduce range, gates the value subtree
  on `r in node.ranges`, replaces external deps with fresh DEFINE_VARs (with
  their vmin/vmax), graph_rewrites the substituted form, checks no_range, and
  substitutes back. Lines 129-142.

  pm_reduce_simplify = pm_reduce_unparented + PatternMatcher([
    (UPat(Ops.REDUCE, src=(UPat.var("u"),), allow_any_len=True, arg=Ops.ADD,
        name="red"), reduce_collapse),
  ])

This script captures the IR shape after running reduce_collapse / pm_reduce_simplify
on the canonical arange-style and eye-style cases that the polygrad C tests assert.

Run from anywhere:
  conda run -n tiny python /home/anton/projects/polygrad/polygrad/test/parity_scripts/tg_reduce_collapse_gt.py
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
from tinygrad.codegen.simplify import (
    pm_reduce_collapse, pm_reduce_simplify, reduce_collapse,
)


def desc(u, depth=0, seen=None):
    if seen is None: seen = set()
    pad = '  ' * depth
    s = f"{pad}{u.op.name} dt={u.dtype}"
    if u.arg is not None: s += f" arg={u.arg!r}"
    print(s)
    if id(u) in seen: return
    seen.add(id(u))
    if depth < 8:
        for c in u.src: desc(c, depth + 1, seen)


def case(label, before):
    print(f"\n=== {label} ===")
    print("BEFORE:")
    desc(before)
    after = graph_rewrite(before, pm_reduce_simplify, name=label)
    print("AFTER:")
    desc(after)
    if after is before: print("(unchanged)")


# === Canonical arange case ===
# arange(5) = REDUCE_ADD(WHERE(r<r.src[0], 1, 0)... actually arange uses cumsum
# which is essentially: REDUCE_ADD over r0 of WHERE(r1<=outer_r, 1, 0)
# Tinygrad's Tensor.arange / _cumalu builds:
#   REDUCE_ADD(WHERE(r_inner < cut, 1, 0), r_inner)
# Rule 5 (fold_range_above) collapses this to: cut.max(0).min(r.src[0]) * 1
print("\n#### Rule 5: fold_range_above (the arange pattern) ####")
r_inner = UOp.range(5, 0)
cut = UOp.const(dtypes.int32, 3)
val = UOp.const(dtypes.int32, 1)
zero = UOp.const(dtypes.int32, 0)
where_val = (r_inner < cut).where(val, zero)
red = where_val.reduce(r_inner, arg=Ops.ADD)
case("rule5_fold_range_above", red)

# === Rule 3: fold_range_below ===
# ((r<cut).where(0, val)).reduce_add(r) -> (r.src[0]-cut).max(0).min(r.src[0]).cast * val
print("\n#### Rule 3: fold_range_below ####")
r0 = UOp.range(5, 0)
cut3 = UOp.const(dtypes.int32, 3)
val1 = UOp.const(dtypes.int32, 1)
zero0 = UOp.const(dtypes.int32, 0)
where_blo = (r0 < cut3).where(zero0, val1)
red_blo = where_blo.reduce(r0, arg=Ops.ADD)
case("rule3_fold_range_below", red_blo)

# === Rule 4: fold_range_two_sided ===
# ((r >= lower) & (r < upper)).where(val, 0).reduce_add(r)
print("\n#### Rule 4: fold_range_two_sided ####")
r0b = UOp.range(7, 0)
lower = UOp.const(dtypes.int32, 2)
upper = UOp.const(dtypes.int32, 5)
val1b = UOp.const(dtypes.int32, 1)
zero0b = UOp.const(dtypes.int32, 0)
ge_lower = (r0b < lower).logical_not()
lt_upper = (r0b < upper)
mask = ge_lower & lt_upper
where_two = mask.where(val1b, zero0b)
red_two = where_two.reduce(r0b, arg=Ops.ADD)
case("rule4_fold_range_two_sided", red_two)

# === Rule 6: reduce_add_distribute ===
# (x + y).reduce_add(r) -> x.reduce_add(r) + y.reduce_add(r)
print("\n#### Rule 6: reduce_add_distribute ####")
r0c = UOp.range(5, 0)
x6 = UOp.const(dtypes.int32, 2)
y6 = UOp.const(dtypes.int32, 3)
red_dist = (x6 + y6).reduce(r0c, arg=Ops.ADD)
case("rule6_reduce_add_distribute", red_dist)

# === Rule 1: lift_add_from_cmplt ===
print("\n#### Rule 1: lift_add_from_cmplt ####")
r0d = UOp.range(10, 0)
y1 = UOp.const(dtypes.int32, 4)
c1 = UOp.const(dtypes.int32, 7)
lt_form = (r0d + y1) < c1
print("BEFORE:")
desc(lt_form)
after = graph_rewrite(lt_form, pm_reduce_collapse, name="rule1")
print("AFTER:")
desc(after)

# === Rule 2: lift_mul_from_cmplt ===
print("\n#### Rule 2: lift_mul_from_cmplt ####")
r0e = UOp.range(10, 0)
y2 = UOp.const(dtypes.int32, 3)
c2 = UOp.const(dtypes.int32, 11)
lt_mul = (r0e * y2) < c2
print("BEFORE:")
desc(lt_mul)
after = graph_rewrite(lt_mul, pm_reduce_collapse, name="rule2")
print("AFTER:")
desc(after)

# === Rule 8: mul_casted_bool ===
print("\n#### Rule 8: mul_casted_bool ####")
gate = UOp.const(dtypes.bool, True)
x8 = UOp.const(dtypes.int32, 7)
mul_form = x8 * gate.cast(dtypes.int32)
print("BEFORE:")
desc(mul_form)
after = graph_rewrite(mul_form, pm_reduce_collapse, name="rule8")
print("AFTER:")
desc(after)

# === E2E arange via Tensor.arange (the real failing case) ===
print("\n#### E2E: tinygrad Tensor.arange(5) IR after pm_reduce_simplify ####")
from tinygrad import Tensor
t = Tensor.arange(5)
sched = t.schedule()
print(f"#kernels = {len(sched)}")
