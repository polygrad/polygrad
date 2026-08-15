#!/usr/bin/env python3
"""Compare executable model compatibility artifacts against pinned Tinygrad."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "compare_tensor_graphs", ROOT / "test" / "compare_tensor_graphs.py"
)
graph_compare = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(graph_compare)


def compare_values(name, tinygrad, polygrad, findings):
    if isinstance(tinygrad, dict):
        if set(tinygrad) != set(polygrad):
            findings.append(f"{name}: keys differ")
            return
        for key in sorted(tinygrad):
            compare_values(f"{name}.{key}", tinygrad[key], polygrad[key], findings)
        return
    ta, pa = np.asarray(tinygrad), np.asarray(polygrad)
    if ta.shape != pa.shape:
        findings.append(f"{name}: shape {ta.shape} != {pa.shape}")
    elif not np.isfinite(ta).all() or not np.isfinite(pa).all():
        findings.append(f"{name}: non-finite value")
    elif not np.allclose(ta, pa, rtol=1e-5, atol=1e-6):
        findings.append(f"{name}: max_abs={float(np.max(np.abs(ta-pa)))}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tinygrad")
    parser.add_argument("polygrad")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    tg = json.loads(Path(args.tinygrad).read_text())
    pg = json.loads(Path(args.polygrad).read_text())
    commit = subprocess.run(
        ["git", "-C", ROOT / "references" / "tinygrad_latest", "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if tg["reference_commit"] != commit or pg["reference_commit"] != commit:
        raise RuntimeError("stale compatibility source lock")
    if tg["schema_version"] != 1 or pg["schema_version"] != 1:
        raise RuntimeError("unsupported compatibility artifact schema")
    if set(tg["cases"]) != set(pg["cases"]):
        raise RuntimeError("compatibility case sets differ")

    report = {"schema_version": 1, "reference_commit": commit, "cases": {},
              "summary": {"pass": 0, "fail": 0}}
    for name in sorted(tg["cases"]):
        tc, pc = tg["cases"][name], pg["cases"][name]
        findings = []
        for field in ("surface", "state_names", "state_shapes", "jit_count"):
            if tc[field] != pc[field]:
                findings.append(f"{field}: differs")
        graph_findings = graph_compare.compare_graph(
            name, tc["forward_graph"], pc["forward_graph"]
        )
        findings.extend(
            f"forward_graph:{row['kind']}:{row['path']}" for row in graph_findings
        )
        for field in ("forward", "loss", "grads", "updated", "jit"):
            compare_values(field, tc[field], pc[field], findings)
        passed = not findings
        report["cases"][name] = {"passed": passed, "findings": findings}
        report["summary"]["pass" if passed else "fail"] += 1

    Path(args.output).write_text(json.dumps(report, sort_keys=True) + "\n")
    print(f"tinygrad compatibility: {report['summary']['pass']} pass, "
          f"{report['summary']['fail']} fail")
    if report["summary"]["fail"]:
        name = next(name for name, row in report["cases"].items() if not row["passed"])
        print(f"first failure: {name}: {report['cases'][name]['findings'][0]}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
