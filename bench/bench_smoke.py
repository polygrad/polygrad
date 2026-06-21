#!/usr/bin/env python3
"""Run the fast absolute C benchmark smoke suite.

This wrapper keeps machine-specific metadata out of the C runner. The JSON it
writes is intended for local baselines and CI artifacts, not for committed
cross-machine performance claims.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _git(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_dirty() -> bool | None:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL)
        return bool(out.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def _default_output() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("bench/results") / f"smoke-{stamp}.json"


def _load_runner_json(cmd: list[str]) -> dict:
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        if proc.stdout:
            print(proc.stdout, file=sys.stderr, end="")
        if proc.stderr:
            print(proc.stderr, file=sys.stderr, end="")
        raise SystemExit(proc.returncode)
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        print("bench_smoke.py: runner did not emit valid JSON", file=sys.stderr)
        print(proc.stdout, file=sys.stderr)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr, end="")
        raise SystemExit(2) from exc


def _print_table(payload: dict, output: Path) -> None:
    print(f"Wrote {output}")
    print(f"{'workload':24} {'median_us':>12} {'mad_us':>10} {'min_us':>10} {'max_us':>10}")
    for name, row in payload.get("benchmarks", {}).items():
        print(
            f"{name:24} "
            f"{row['median_us']:12.3f} "
            f"{row['mad_us']:10.3f} "
            f"{row['min_us']:10.3f} "
            f"{row['max_us']:10.3f}"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", default="build/bench_smoke", help="compiled C benchmark runner")
    ap.add_argument("--output", type=Path, default=None, help="JSON output path")
    ap.add_argument("--samples", type=int, default=7)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30)
    args = ap.parse_args()

    output = args.output or _default_output()
    cmd = [
        args.runner,
        "--samples",
        str(args.samples),
        "--iters",
        str(args.iters),
        "--warmup",
        str(args.warmup),
    ]

    payload = _load_runner_json(cmd)
    payload["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    payload["command"] = cmd
    payload["git"] = {
        "commit": _git(["rev-parse", "HEAD"]),
        "dirty": _git_dirty(),
    }
    payload["machine"] = {
        "node": platform.node(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _print_table(payload, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
