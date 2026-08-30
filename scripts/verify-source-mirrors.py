#!/usr/bin/env python3
"""Verify py/csrc and js/csrc mirror the active C sources."""

from __future__ import annotations

import filecmp
import os
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PY_SYNC = ROOT / "py" / "scripts" / "sync-csrc.py"
JS_EXCLUDE = {
    "wasm.c",
    "wasm_builder.c",
    "wgsl.c",
    "llama3.c",
    "resnet.c",
    "vit.c",
}


def py_manifest() -> list[Path]:
    manifest = runpy.run_path(str(PY_SYNC))
    rels = manifest["SOURCES"] + manifest["HEADERS"]
    return [Path(rel) for rel in rels]


def js_manifest() -> list[Path]:
    rels: list[Path] = []
    for base in (ROOT / "src", ROOT / "vendor" / "cjson"):
        if not base.exists():
            continue
        for dirpath, _, filenames in os.walk(base):
            for filename in filenames:
                if not (filename.endswith(".c") or filename.endswith(".h")):
                    continue
                if filename in JS_EXCLUDE:
                    continue
                rels.append(Path(dirpath, filename).relative_to(ROOT))
    return sorted(rels)


def check_tree(label: str, dst_root: Path, rels: list[Path]) -> list[str]:
    errors: list[str] = []
    for rel in rels:
        src = ROOT / rel
        dst = dst_root / rel
        if not src.is_file():
            errors.append(f"{label}: missing source {rel}")
            continue
        if not dst.is_file():
            errors.append(f"{label}: missing mirror {dst.relative_to(ROOT)}")
            continue
        if not filecmp.cmp(src, dst, shallow=False):
            errors.append(f"{label}: stale mirror {dst.relative_to(ROOT)}")
    return errors


def main() -> int:
    errors: list[str] = []
    errors.extend(check_tree("py", ROOT / "py" / "csrc", py_manifest()))
    errors.extend(check_tree("js", ROOT / "js" / "csrc", js_manifest()))

    if errors:
        print("verify-source-mirrors: source mirrors are stale", file=sys.stderr)
        for err in errors:
            print(f"  {err}", file=sys.stderr)
        print(
            "Run: python py/scripts/sync-csrc.py && node js/scripts/sync-csrc.js",
            file=sys.stderr,
        )
        return 1
    print("verify-source-mirrors: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
