"""Ground truth for the logical-NOT representation in tinygrad's IR.

Tinygrad: `x.logical_not()` (mixin/elementwise.py:25-33) lowers to
  CMPNE(CAST(x, bool), CONST(true))
which symbolic_simple (uop/symbolic.py:126) collapses to
  CMPNE(x, CONST(true))           when x is already bool
i.e. `x != True`.

Polygrad uses a different canonical form: `NEG(bool_uop)` at the kernel
level (alu.c:169 maps NEG-on-bool to `!a`). Both forms are semantically
equivalent — they're the same operator with different IR encodings — but
the patterns Phase D's reduce_collapse uses must reference *polygrad's*
form, not tinygrad's UPat construction directly.

This script captures the tinygrad-side IR shape for reference. The
polygrad-side is asserted in test/test_sym.c::TEST(sym, logical_not_canon).

Run from anywhere:
  conda run -n tiny python /home/anton/projects/polygrad/polygrad/test/parity_scripts/tg_logical_not_gt.py

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

def show(name, u):
    print("[%s] op=%s dtype=%s n_src=%d arg=%s" %
          (name, u.op, u.dtype, len(u.src), u.arg))

# Build (x < 3).logical_not() at the UOp level (no Tensor sugar).
x = UOp.range(5, 0)
c3 = UOp.const(dtypes.int32, 3)
lt = x < c3                           # CMPLT
print("=== before logical_not ===")
show("CMPLT(x,3)", lt)

ln = lt.logical_not()
print("=== after logical_not (raw) ===")
show("logical_not", ln)
for i, s in enumerate(ln.src):
    show("  src[%d]" % i, s)

# After symbolic simplification — uses graph_rewrite + symbolic.
from tinygrad.uop.symbolic import symbolic
from tinygrad.uop.ops import graph_rewrite
ln_simp = graph_rewrite(ln, symbolic, name="simplify_logical_not")
print("=== after symbolic ===")
show("simplified", ln_simp)
for i, s in enumerate(ln_simp.src):
    show("  src[%d]" % i, s)
