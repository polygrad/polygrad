#!/usr/bin/env python3
"""Check C/C++ header usability and the core -> FFI dependency boundary."""
import os
from pathlib import Path
import re
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent.parent
PUBLIC = ("polygrad.h", "tensor.h", "frontend.h", "model.h", "nn.h", "optim.h")
OWNERS = ("mixin/elementwise.h", "uop/ops.h", "placer.h", "device.h",
          "engine/schedule.h", "engine/realize.h")


def main():
    sources = {p: p.read_text() for p in (ROOT / "src").rglob("*")
               if p.suffix in (".c", ".h")}
    for path, source in sources.items():
        if path.name not in ("frontend.c", "frontend.h"):
            assert not re.search(r'#\s*include\s*["<](?:[^">]*/)?frontend(?:_internal)?\.h[">]', source), path
    adapters = re.findall(r'^(?:PolyUOp \*|PolyTensor \*|int |void |uint32_t )(poly_\w+)\(',
                          sources[ROOT / "src/frontend.c"], re.M)
    for path, source in sources.items():
        if path.suffix == ".c" and path.name != "frontend.c":
            for name in adapters:
                assert not re.search(r'\b' + name + r'\(', source), f"{path} calls FFI adapter {name}"
    # These public entry headers must not duplicate declarations. Including an
    # owner is sufficient; repeating a signature risks C/C++ linkage drift.
    seen = {}
    for header in PUBLIC[:3]:
        for name in re.findall(r'^\w[^\n;{}]*\b(poly_\w+)\([^;{]*;',
                               (ROOT / "src" / header).read_text(), re.M):
            assert name not in seen, f"{name}: {seen.get(name)}, {header}"
            seen[name] = header
    count = 0
    for compiler, language, standard in ((os.getenv("CC", "cc"), "c", "c11"),
                                          (os.getenv("CXX", "c++"), "c++", "c++17")):
        for headers in [(h,) for h in PUBLIC + OWNERS] + [PUBLIC, tuple(reversed(PUBLIC))]:
            unit = "".join(f'#include "{h}"\n' for h in headers)
            subprocess.run(shlex.split(compiler) + [f"-std={standard}", "-Werror",
                           "-Isrc", "-x", language, "-fsyntax-only", "-"],
                           input=unit, text=True, cwd=ROOT, check=True)
            count += 1
    print(f"Headers: {count} C/C++ checks passed; unique public declarations; no core FFI includes")


if __name__ == "__main__":
    main()
