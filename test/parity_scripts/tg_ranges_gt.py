from tinygrad.uop.ops import UOp, Ops
from tinygrad.dtype import dtypes
from tinygrad.codegen.simplify import no_range

r = UOp.range(5, 0)
r2 = UOp.range(7, 1)
c = UOp.const(dtypes.weakint, 3)

print("=== tinygrad ground truth ===")
print("[c1] RANGE(5,0)        no_range=%s  |ranges|=%d" % (no_range(r), len(r.ranges)))
print("[c2] r+3               no_range=%s  |ranges|=%d" % (no_range(r+c), len((r+c).ranges)))
red = (r+c).reduce(r, arg=Ops.ADD)
print("[c3] REDUCE(r+3,r)     no_range=%s  |ranges|=%d" % (no_range(red), len(red.ranges)))
print("[c4] r+r2              |ranges|=%d" % (len((r+r2).ranges)))
red2 = (r+r2).reduce(r, arg=Ops.ADD)
print("[c5] REDUCE(r+r2,r)    no_range=%s  |ranges|=%d  r?=%s  r2?=%s" % (
    no_range(red2), len(red2.ranges), r in red2.ranges, r2 in red2.ranges))
