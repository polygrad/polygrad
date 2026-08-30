#!/usr/bin/env python3
"""Record a filtered chronological Python call trace for a reference probe.

This is migration-routing evidence, not a semantic parity oracle.  Run the same
probe against two reference checkouts and pair the resulting route delta with
UOp, kernel, and value comparisons.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import runpy
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, required=True)
  parser.add_argument("--probe", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument(
    "--qualname", action="append", default=[],
    help="fnmatch pattern for recorded function qualified names; repeatable",
  )
  args = parser.parse_args()

  root = args.root.resolve()
  root_prefix = f"{root}{os.sep}"
  counts: Counter[tuple[str, int, str]] = Counter()
  transitions: list[dict[str, object]] = []
  n_calls = 0
  last_key: tuple[str, int, str] | None = None

  def profile(frame, event, _arg):
    nonlocal n_calls, last_key
    if event != "call": return
    filename = frame.f_code.co_filename
    if not filename.startswith(root_prefix): return
    key = (filename[len(root_prefix):], frame.f_code.co_firstlineno, frame.f_code.co_qualname)
    if args.qualname and not any(fnmatch.fnmatchcase(key[2], pattern) for pattern in args.qualname): return
    n_calls += 1
    counts[key] += 1
    if key != last_key:
      transitions.append({"seq": n_calls - 1, "path": key[0], "line": key[1], "qualname": key[2]})
      last_key = key

  sys.setprofile(profile)
  try:
    runpy.run_path(str(args.probe.resolve()), run_name="__main__")
  finally:
    sys.setprofile(None)

  payload = {
    "schema_version": 1,
    "root": str(root),
    "probe": str(args.probe.resolve()),
    "call_count": n_calls,
    "transitions": transitions,
    "counts": [
      {"path": path, "line": line, "qualname": qualname, "count": count}
      for (path, line, qualname), count in sorted(counts.items())
    ],
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
  print(f"wrote {args.output}: {n_calls} calls, {len(counts)} unique routes, {len(transitions)} transitions")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
