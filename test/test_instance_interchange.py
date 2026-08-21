#!/usr/bin/env python3
"""Cross-frontend PGIR+safetensors product acceptance gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'py'))
os.environ.setdefault('POLYGRAD_LIB', str(ROOT / 'build' / 'libpolygrad.so'))

from polygrad.instance import Instance, OPTIM_ADAM  # noqa: E402
from polygrad.models import MLP  # noqa: E402


def safetensor_names(data: bytes) -> set[str]:
    header_len = int.from_bytes(data[:8], 'little')
    return set(json.loads(data[8:8 + header_len]).keys()) - {'__metadata__'}


def train_step(instance: Instance) -> float:
    return instance.train_step(
        x=np.array([1.0, 2.0], dtype=np.float32),
        y=np.array([3.0], dtype=np.float32),
    )


def compare_checkpoint(source: Instance, restored: Instance, names: set[str], atol: float) -> None:
    for name in sorted(names):
        source_i, restored_i = source.find_buf(name), restored.find_buf(name)
        if source_i < 0 or restored_i < 0:
            raise AssertionError(f'missing checkpoint row {name!r}')
        np.testing.assert_allclose(
            restored.buf_data(restored_i), source.buf_data(source_i), rtol=0.0, atol=atol,
            err_msg=f'checkpoint row {name}',
        )


def run_core(work: Path, core: str, source_after: Instance, expected_loss: float) -> None:
    subprocess.run(
        ['node', str(ROOT / 'js' / 'test' / 'instance_interchange.js'), str(work), core],
        cwd=ROOT, check=True,
    )
    result = json.loads((work / f'javascript-{core}-result.json').read_text())
    atol = 0.0 if core == 'native' else 1e-6
    np.testing.assert_allclose(result['trainLoss'], expected_loss, rtol=0.0, atol=atol)

    resumed_weights = (work / f'javascript-{core}-resumed.safetensors').read_bytes()
    resumed = Instance.from_ir((work / 'python-train.pgir').read_bytes(), resumed_weights)
    try:
        compare_checkpoint(source_after, resumed, safetensor_names(resumed_weights), atol)
    finally:
        resumed.free()

    inference = Instance.from_ir(
        (work / f'javascript-{core}-inference.pgir').read_bytes(),
        (work / f'javascript-{core}-inference.safetensors').read_bytes(),
    )
    try:
        output = inference.forward(x=np.array([1.25, -0.5], dtype=np.float32))['output']
        np.testing.assert_allclose(
            output, np.asarray(result['inferenceOutput'], dtype=np.float32),
            rtol=0.0, atol=atol,
        )
    finally:
        inference.free()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cores', default='native,wasm')
    args = parser.parse_args()
    cores = [value.strip() for value in args.cores.split(',') if value.strip()]
    if any(core not in {'native', 'wasm'} for core in cores):
        raise SystemExit('--cores accepts native,wasm')

    (ROOT / 'temp').mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='instance-interchange.', dir=ROOT / 'temp'))
    source = MLP({
        'layers': [2, 1], 'activation': 'none', 'bias': False,
        'loss': 'mse', 'batch_size': 1, 'seed': 7,
    })
    try:
        source.set_optimizer(OPTIM_ADAM, lr=0.05)
        for _ in range(3):
            train_step(source)
        (work / 'python-train.pgir').write_bytes(source.export_ir())
        checkpoint = source.export_weights()
        (work / 'python-train.safetensors').write_bytes(checkpoint)
        expected_loss = train_step(source)

        for core in cores:
            run_core(work, core, source, expected_loss)
            print(f'instance interchange {core}: pass')
    finally:
        source.free()
        shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
    main()
