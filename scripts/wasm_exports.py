#!/usr/bin/env python3
"""Emit Emscripten exports referenced by the handwritten JS frontend."""

import pathlib
import re
import sys
import json


roots = [pathlib.Path(arg) for arg in sys.argv[1:]] or [pathlib.Path("js/src")]
exports = {"malloc", "free"}
# Keep the strict C entry point available alongside JS's diagnostic/options form.
exports.add("poly_tokenizer_from_json")
exports.update(json.loads((pathlib.Path(__file__).resolve().parents[1] / 'js/src/extension_api.json').read_text()))
for root in roots:
  for path in root.rglob("*.js"):
    source = path.read_text(encoding="utf-8")
    exports.update(re.findall(r"\bModule\._(poly_[A-Za-z0-9_]+|malloc|free)\b", source))
    exports.update(re.findall(r"\b(?:cwrap|ccall)\(\s*['\"](poly_[A-Za-z0-9_]+)", source))

print(",".join(f"_{name}" for name in sorted(exports)))
