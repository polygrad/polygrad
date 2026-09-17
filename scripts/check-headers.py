#!/usr/bin/env python3
"""Check C/C++ header usability and the core -> FFI dependency boundary."""
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent.parent
PUBLIC = ("polygrad.h", "tensor.h", "frontend.h", "model.h", "nn/nn.h", "nn/optim.h", "models/layers.h", "models/models.h")
OWNERS = ("mixin/elementwise.h", "uop/ops.h", "placer.h", "device.h",
          "engine/schedule.h", "engine/realize.h", "schedule/schedule.h",
          "models/mlp.h", "models/tabm.h", "models/nam.h", "models/gpt2.h",
          "models/qwen3.h", "models/llama.h", "models/hf_loader.h", "loaders/gguf_loader.h")


def main():
    (ROOT / 'temp').mkdir(exist_ok=True)
    # A linkable core must also run independently of Model/codec objects. Two
    # fresh processes exercise ASLR-independent UOp content keys, not pointers.
    with tempfile.TemporaryDirectory(prefix='core-header-', dir=ROOT / 'temp') as directory:
        executable = Path(directory) / 'core'
        unit = '''#include "polygrad.h"
#include "uop/ops.h"
#include <stdio.h>
#include <stdlib.h>
int main(void) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *sum = poly_alu2(ctx, POLY_OP_ADD, poly_const_int(ctx, 2), poly_const_int(ctx, 3));
  size_t size = 0;
  unsigned char *key = poly_uop_key(ctx, sum, &size);
  if (!key || !size) return 1;
  for (size_t i = 0; i < size; i++) printf("%02x", key[i]);
  free(key);
  poly_ctx_destroy(ctx);
  return 0;
}
'''
        subprocess.run(shlex.split(os.getenv('CC', 'cc')) + ['-std=c11', '-Isrc', '-x', 'c', '-',
                       '-Lbuild', '-lpolygrad-core-check', f'-Wl,-rpath,{ROOT / "build"}', '-o', str(executable)],
                       input=unit, text=True, cwd=ROOT, check=True)
        first = subprocess.check_output([executable])
        assert first and first == subprocess.check_output([executable]), 'process-dependent UOp key'
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
    for header in (*PUBLIC[:3], "schedule/schedule.h"):
        for name in re.findall(r'^\w[^\n;{}]*\b(poly_\w+)\([^;{]*;',
                               (ROOT / "src" / header).read_text(), re.M):
            assert name not in seen, f"{name}: {seen.get(name)}, {header}"
            seen[name] = header
    count = 0
    for compiler, language, standard in ((os.getenv("CC", "cc"), "c", "c11"),
                                          (os.getenv("CXX", "c++"), "c++", "c++17")):
        for headers in [(h,) for h in PUBLIC + OWNERS] + [PUBLIC, tuple(reversed(PUBLIC))]:
            unit = "".join(f'#include "{h}"\n' for h in headers)
            if headers == ("polygrad.h",):
                # Context controls must be usable without compiler-owner headers.
                unit += "size_t (*cache_len)(PolyCtx *) = poly_schedule_cache_len;\n"
                unit += "int (*cache_clear)(PolyCtx *) = poly_schedule_cache_clear;\n"
                unit += "int (*abi_version)(void) = poly_abi_version;\n"
            subprocess.run(shlex.split(compiler) + [f"-std={standard}", "-Werror",
                           "-Isrc", "-x", language, "-fsyntax-only", "-"],
                           input=unit, text=True, cwd=ROOT, check=True)
            count += 1
    print(f"Headers: {count} C/C++ checks passed; core-only link and process-stable key; "
          "unique public declarations; no core FFI includes")


if __name__ == "__main__":
    main()
