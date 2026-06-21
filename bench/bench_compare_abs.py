#!/usr/bin/env python3
"""Compare absolute benchmark smoke JSON files.

The policy is deliberately local-machine oriented: a run fails only when it is
slower than both the relative and absolute thresholds recorded in the baseline.
That avoids noise failures on tiny microsecond-scale workloads while still
catching real scheduler/codegen regressions.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


DEFAULT_THRESHOLD_PCT = 0.10
DEFAULT_THRESHOLD_ABS_US = 5.0


def _load(path: Path) -> dict:
    if not path.exists():
        print(f"No benchmark baseline found: {path}", file=sys.stderr)
        print("Run: make bench-local-baseline", file=sys.stderr)
        raise SystemExit(2)
    return json.loads(path.read_text())


def _median(row: dict) -> float:
    val = float(row["median_us"])
    if not math.isfinite(val):
        raise ValueError(f"non-finite median_us: {val}")
    return val


def _threshold(row: dict, key: str, default: float) -> float:
    val = float(row.get(key, default))
    return val if math.isfinite(val) and val >= 0.0 else default


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", type=Path)
    ap.add_argument("current", type=Path)
    ap.add_argument(
        "--warn-only",
        action="store_true",
        help="print failures but return success; useful while collecting a new baseline",
    )
    args = ap.parse_args()

    baseline = _load(args.baseline)
    current = _load(args.current)
    b_bench = baseline.get("benchmarks", {})
    c_bench = current.get("benchmarks", {})

    failures: list[str] = []
    all_names = sorted(set(b_bench) | set(c_bench))

    print(f"{'status':10} {'workload':24} {'baseline':>12} {'current':>12} {'change':>10} {'limit':>12}")
    for name in all_names:
        if name not in b_bench:
            failures.append(f"{name}: missing from baseline")
            print(f"{'NEW':10} {name:24} {'-':>12} {_median(c_bench[name]):12.3f} {'-':>10} {'-':>12}")
            continue
        if name not in c_bench:
            failures.append(f"{name}: missing from current run")
            print(f"{'GONE':10} {name:24} {_median(b_bench[name]):12.3f} {'-':>12} {'-':>10} {'-':>12}")
            continue

        base = _median(b_bench[name])
        cur = _median(c_bench[name])
        pct = _threshold(b_bench[name], "threshold_pct", DEFAULT_THRESHOLD_PCT)
        abs_us = _threshold(b_bench[name], "threshold_abs_us", DEFAULT_THRESHOLD_ABS_US)
        limit = max(base * (1.0 + pct), base + abs_us)
        change = (cur / base - 1.0) if base > 0 else 0.0

        status = "PASS"
        if cur > limit:
            status = "FAIL"
            failures.append(
                f"{name}: {cur:.3f}us > {limit:.3f}us "
                f"(baseline {base:.3f}us, pct {pct:.1%}, abs {abs_us:.3f}us)"
            )
        elif cur < base * 0.90:
            status = "FASTER"

        print(f"{status:10} {name:24} {base:12.3f} {cur:12.3f} {change:10.1%} {limit:12.3f}")

    if failures:
        print("\nBenchmark regression check failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 0 if args.warn_only else 1

    print("\nBenchmark regression check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
