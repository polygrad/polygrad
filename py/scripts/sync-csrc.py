#!/usr/bin/env python3
"""
sync-csrc.py -- Copy C sources from the polygrad repo into py/csrc/.

Manifest-driven: the SOURCES list below must match the Makefile's SRC +
CODEC_SRC. HEADERS lists all headers needed for compilation. This is the
single source of truth for which C files ship in the Python sdist.

Usage:
    python scripts/sync-csrc.py          # run from py/
    python py/scripts/sync-csrc.py       # run from repo root
"""

import os
import shutil
import sys

# Authoritative source list -- must match Makefile SRC + CODEC_SRC
SOURCES = [
    # SRC (core)
    'src/ops.c',
    'src/dtype.c',
    'src/arena.c',
    'src/hashmap.c',
    'src/utils.c',
    'src/bigint.c',
    'src/selftest.c',
    'src/ctx.c',
    'src/device.c',
    # Explicit logical placement and shared UOp device queries.
    'src/placer.c',
    'src/engine/jit.c',
    'src/engine/realize.c',
    'src/uop/ops.c',
    'src/uop/spec.c',
    'src/uop/movement.c',
    'src/uop/weak.c',
    'src/mixin/elementwise.c',
    'src/mixin/movement.c',
    'src/uop/upat.c',
    'src/alu.c',
    'src/uop/symbolic.c',
    'src/uop/symbolic.h',
    'src/shape.c',
    'src/autograd.c',
    'src/codegen/codegen.c',
    'src/codegen/opt/tc.c',
    'src/codegen/decomp/dtype.c',
    'src/codegen/gpudims.c',
    'src/codegen/late/coalesce.c',
    'src/codegen/late/gater.c',
    'src/codegen/late/linearizer.c',
    'src/renderer/cstyle.c',
    'src/renderer/cuda.c',
    'src/renderer/wgsl.c',
    'src/runtime_cpu.c',
    'src/runtime/support/memory.c',
    'src/runtime_wasm.c',
    'src/runtime_webgpu.c',
    'src/runtime_cuda.c',
    'src/wasm_builder.c',
    'src/renderer/wasm.c',
    'src/frontend.c',
    'src/tensor.c',
    'src/engine/schedule.c',
    'src/interp.c',
    'src/bundle.c',
    'src/schedule/rangeify.c',
    'src/schedule/multi.c',
    'src/schedule/allreduce.c',
    'src/schedule/schedule.c',
    'src/schedule/memory.c',
    'src/codegen/simplify.c',
    'src/schedule/indexing.c',
    'src/nn.c',
    'src/optim.c',
    'src/tokenizer.c',
    'src/renderer/hip.c',
    'src/runtime_hip.c',
    # CODEC_SRC
    'vendor/cjson/cJSON.c',
    'src/safetensors.c',
    'src/wlrn.c',
    'src/ir.c',
    'src/model.c',
    'src/models/compose.c',
    'src/models/mlp.c',
    'src/models/tabm.c',
    'src/models/nam.c',
    'src/models/registry.c',
    'src/models/gpt2.c',
    'src/models/qwen3.c',
    'src/models/hf_loader.c',
    # LOADER_SRC
    'src/loaders/decoded.c',
    'src/loaders/import_error.c',
    'src/loaders/bind.c',
    'src/loaders/hf_decode.c',
    'src/loaders/gguf_decode.c',
    'src/loaders/gguf_loader.c',
    'src/loaders/import_desc.c',
]

# Headers needed for compilation
HEADERS = [
    'src/polygrad.h',
    'src/arena.h',
    'src/utils.h',
    'src/bigint.h',
    'src/ctx.h',
    'src/device.h',
    'src/uop/spec.h',
    'src/uop/ops.h',
    'src/uop/movement.h',
    'src/uop/weak.h',
    'src/engine/jit.h',
    'src/engine/realize.h',
    'src/runtime_wasm.h',
    'src/runtime_webgpu.h',
    'src/codegen/codegen.h',
    'src/codegen/decomp/dtype.h',
    'src/codegen/late/coalesce.h',
    'src/codegen/late/gater.h',
    'src/codegen/late/linearizer.h',
    'src/codegen/opt/tc.h',
    'src/renderer/renderer.h',
    'src/renderer/cstyle.h',
    'src/renderer/isa/x86.h',
    'src/runtime/graph/cuda.h',
    'src/runtime/support/memory.h',
    'src/frontend.h',
    'src/tensor.h',
    'src/placer.h',
    'src/mixin/elementwise.h',
    'src/engine/schedule.h',
    'src/interp.h',
    'src/bundle.h',
    'src/schedule/indexing.h',
    'src/schedule/schedule.h',
    'src/schedule/memory.h',
    'src/model.h',
    'src/models/compose.h',
    'src/ir.h',
    'src/models/mlp.h',
    'src/models/nam.h',
    'src/models/tabm.h',
    'src/models/models.h',
    'src/nn.h',
    'src/optim.h',
    'src/tokenizer.h',
    'src/uop/upat.h',
    'src/codegen/simplify.h',
    'src/safetensors.h',
    'src/wasm_builder.h',
    'src/wlrn.h',
    'vendor/cjson/cJSON.h',
    # Loader headers
    'src/loaders/decoded.h',
    'src/loaders/import_error.h',
    'src/loaders/bind.h',
    'src/loaders/hf_decode.h',
    'src/loaders/gguf_decode.h',
    'src/loaders/import_desc.h',
    # Model headers referenced by gpt2.c
    'src/models/gpt2.h',
    'src/models/qwen3.h',
    'src/models/hf_loader.h',
    'src/schedule/rangeify.h',
    'src/schedule/multi.h',
    'src/schedule/allreduce.h',
]


def main():
    # Determine repo root: this script lives at py/scripts/sync-csrc.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    py_dir = os.path.dirname(script_dir)
    repo_root = os.path.dirname(py_dir)

    csrc_dir = os.path.join(py_dir, 'csrc')

    # Clean and recreate
    if os.path.exists(csrc_dir):
        shutil.rmtree(csrc_dir)

    all_files = SOURCES + HEADERS
    missing = []
    for f in all_files:
        src = os.path.join(repo_root, f)
        if not os.path.isfile(src):
            missing.append(f)

    if missing:
        print(f'ERROR: {len(missing)} manifest files not found in repo root ({repo_root}):',
              file=sys.stderr)
        for f in missing:
            print(f'  {f}', file=sys.stderr)
        sys.exit(1)

    copied = 0
    for f in all_files:
        src = os.path.join(repo_root, f)
        dst = os.path.join(csrc_dir, f)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        # Do not preserve mtimes here: editable builds rely on fresh csrc/
        # timestamps so setuptools recompiles changed native sources.
        shutil.copy(src, dst)
        copied += 1

    print(f'sync-csrc: copied {copied} files to {csrc_dir}')


if __name__ == '__main__':
    main()
