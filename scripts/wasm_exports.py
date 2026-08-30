#!/usr/bin/env python3
"""Emit Emscripten exports referenced by the handwritten JS frontend."""

import pathlib
import re
import sys


roots = [pathlib.Path(arg) for arg in sys.argv[1:]] or [pathlib.Path("js/src")]
exports = {"malloc", "free"}
for root in roots:
  for path in root.rglob("*.js"):
    source = path.read_text(encoding="utf-8")
    exports.update(re.findall(r"\bModule\._(poly_[A-Za-z0-9_]+|malloc|free)\b", source))
    exports.update(re.findall(r"\b(?:cwrap|ccall)\(\s*['\"](poly_[A-Za-z0-9_]+)", source))

print(",".join(f"_{name}" for name in sorted(exports)))
