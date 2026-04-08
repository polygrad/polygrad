"""Ground truth for tinygrad's reduce_unparented (codegen/simplify.py:77-92).

Run from anywhere:
  conda run -n tiny python /home/anton/projects/polygrad/polygrad/test/parity_scripts/tg_reduce_unparented_gt.py

The script self-bootstraps PYTHONPATH from its own location, so it always
loads references/tinygrad_latest regardless of cwd or any conda-installed
tinygrad. Asserts the loaded tinygrad path before importing anything else.
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
from tinygrad.codegen.simplify import pm_reduce_unparented


def desc(u, depth=0):
    s = "  " * depth + f"{u.op.name} dtype={u.dtype} arg={u.arg!r}"
    print(s)
    for c in u.src:
        desc(c, depth + 1)


def case(label, before, after_expected_str=None):
    print(f"\n=== {label} ===")
    print("BEFORE:")
    desc(before)
    after = graph_rewrite(before, pm_reduce_unparented, name=label)
    print("AFTER:")
    desc(after)
    if after is before:
        print("(unchanged)")
    if after_expected_str is not None:
        print(f"EXPECTED: {after_expected_str}")


# Build ranges
r0 = UOp.range(5, 0)  # used
r1 = UOp.range(7, 1)  # unparented
r2 = UOp.range(3, 2)  # unparented

# Case A: ADD reduce, value depends only on r0, trailing range r1 unused.
# Expected: REDUCE(value, r0) * 7
val_a = r0 + UOp.const(dtypes.int, 1)
red_a = val_a.reduce(r0, r1, arg=Ops.ADD)
case("A_add_one_unused", red_a, "REDUCE(value, r0) * 7")

# Case B: ADD reduce, value is just a constant, both ranges unparented.
# Expected: const * 5 * 7 (since red.dtype != src[0].dtype path doesn't fire,
# value has no parented ranges left so we drop trailing srcs entirely)
val_b = UOp.const(dtypes.int, 4)
red_b = val_b.reduce(r0, r1, arg=Ops.ADD)
case("B_add_const_all_unused", red_b, "4 * 5 * 7")

# Case C: MUL reduce with one unparented trailing range.
# Expected: REDUCE(value, r0) ** 7
val_c = r0 * UOp.const(dtypes.int, 2)
red_c = val_c.reduce(r0, r1, arg=Ops.MUL)
case("C_mul_one_unused", red_c, "REDUCE(value, r0) ** 7")

# Case D: MAX reduce, one unparented trailing range.
# Per simplify.py:82-87, MAX has no multiplier path; ret = REDUCE(value, parented_only).
val_d = r0 + UOp.const(dtypes.int, 0)
red_d = val_d.reduce(r0, r1, arg=Ops.MAX)
case("D_max_one_unused", red_d, "REDUCE(value, r0)")

# Case E: ADD reduce, all ranges parented (must be unchanged).
val_e = r0 + r1
red_e = val_e.reduce(r0, r1, arg=Ops.ADD)
case("E_all_parented_noop", red_e, "(unchanged)")

# Case F: ADD reduce, two unparented ranges.
val_f = r0 + UOp.const(dtypes.int, 0)
red_f = val_f.reduce(r0, r1, r2, arg=Ops.ADD)
case("F_add_two_unused", red_f, "REDUCE(value, r0) * 7 * 3")
