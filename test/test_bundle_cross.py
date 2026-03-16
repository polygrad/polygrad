#!/usr/bin/env python3
"""
Cross-language bundle test: save in Python, verify in JS.

Generates test/fixtures/mlp_cross.polybndl + mlp_cross.json.
JS test loads the bundle and checks predictions match.
"""

import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'py'))
from polygrad.instance import Instance

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')

def generate_fixture():
    os.makedirs(FIXTURE_DIR, exist_ok=True)

    inst = Instance.mlp({
        'layers': [2, 4, 1],
        'activation': 'relu',
        'bias': True,
        'loss': 'mse',
        'batch_size': 1,
        'seed': 42
    })

    # Deterministic weights via param_data (mutable numpy view)
    for i in range(inst.param_count):
        data = inst.param_data(i)
        if data is not None:
            for j in range(len(data)):
                data[j] = (j % 7 - 3) * 0.1

    # Forward
    x = np.array([1.0, 2.0], dtype=np.float32)
    out = inst.forward(x=x)

    # Save bundle
    bundle_bytes = inst.save_bundle()
    bundle_path = os.path.join(FIXTURE_DIR, 'mlp_cross.polybndl')
    with open(bundle_path, 'wb') as f:
        f.write(bundle_bytes)

    # Sidecar
    sidecar = {
        'input': x.tolist(),
        'output': {k: v.tolist() for k, v in out.items()},
        'param_count': inst.param_count,
        'buf_count': inst.buf_count,
    }
    sidecar_path = os.path.join(FIXTURE_DIR, 'mlp_cross.json')
    with open(sidecar_path, 'w') as f:
        json.dump(sidecar, f, indent=2)

    print(f'bundle: {bundle_path} ({len(bundle_bytes)} bytes)')
    print(f'sidecar: {sidecar_path}')
    print(f'output: {sidecar["output"]}')

    # Self-test: reload in Python
    inst2 = Instance.from_bundle(bundle_bytes)
    out2 = inst2.forward(x=x)
    for k in out:
        assert np.allclose(out[k], out2[k], atol=1e-6), f'round-trip mismatch: {k}'
    print('python round-trip: OK')

if __name__ == '__main__':
    generate_fixture()
