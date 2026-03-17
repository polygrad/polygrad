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
    'src/uop.c',
    'src/pat.c',
    'src/alu.c',
    'src/sym.c',
    'src/shape.c',
    'src/sched.c',
    'src/autograd.c',
    'src/codegen.c',
    'src/render_c.c',
    'src/render_cuda.c',
    'src/render_wgsl.c',
    'src/runtime_cpu.c',
    'src/runtime_cuda.c',
    'src/wasm_builder.c',
    'src/render_wasm.c',
    'src/frontend.c',
    'src/exec_plan.c',
    'src/interp.c',
    'src/bundle.c',
    'src/rangeify.c',
    'src/indexing.c',
    'src/nn.c',
    # CODEC_SRC
    'vendor/cjson/cJSON.c',
    'src/safetensors.c',
    'src/wlrn.c',
    'src/ir.c',
    'src/instance.c',
    'src/model_mlp.c',
    'src/model_tabm.c',
    'src/model_nam.c',
    'src/modelzoo/modelzoo.c',
    'src/modelzoo/models/gpt2.c',
    'src/modelzoo/hf_loader.c',
]

# Headers needed for compilation
HEADERS = [
    'src/polygrad.h',
    'src/arena.h',
    'src/codegen.h',
    'src/frontend.h',
    'src/frontend_internal.h',
    'src/exec_plan.h',
    'src/interp.h',
    'src/bundle.h',
    'src/indexing.h',
    'src/instance.h',
    'src/ir.h',
    'src/model_mlp.h',
    'src/model_nam.h',
    'src/model_tabm.h',
    'src/nn.h',
    'src/pat.h',
    'src/rangeify.h',
    'src/recipe.h',
    'src/safetensors.h',
    'src/scheduler.h',
    'src/wasm_builder.h',
    'src/wlrn.h',
    'src/modelzoo/modelzoo.h',
    'vendor/cjson/cJSON.h',
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
