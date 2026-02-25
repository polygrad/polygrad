#!/usr/bin/env python
"""Benchmark: Python MLP training loop with per-phase timing.

Measures scheduling + compile overhead vs kernel execution time per
training iteration. Shows the impact of schedule caching (iter 0 vs 1+).
"""

import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'py'))

from polygrad import Tensor
from polygrad.nn import Linear, SGD, get_parameters
import numpy as np

# ── Configuration ──────────────────────────────────────────────────────

CONFIGS = [
    {'name': 'tiny',   'dims': [4, 4, 2],     'batch': 4,  'iters': 100},
    {'name': 'small',  'dims': [16, 8, 4],     'batch': 8,  'iters': 50},
    {'name': 'medium', 'dims': [64, 32, 16],   'batch': 16, 'iters': 20},
    {'name': 'large',  'dims': [784, 128, 10], 'batch': 4,  'iters': 5},
]

LR = 0.001
N_WARMUP = 1


def run_config(cfg):
    dims = cfg['dims']
    batch = cfg['batch']
    n_iters = cfg['iters']

    Tensor.manual_seed(42)
    np.random.seed(42)

    # Build MLP
    layers = []
    for i in range(len(dims) - 1):
        layers.append(Linear(dims[i], dims[i + 1]))

    def forward(x):
        for i, fc in enumerate(layers):
            x = fc(x)
            if i < len(layers) - 1:
                x = x.relu()
        return x

    params = []
    for fc in layers:
        params.extend(get_parameters(fc))
    opt = SGD(params, lr=LR)

    x_data = np.random.randn(batch, dims[0]).astype(np.float32) * 0.01
    y_data = np.random.randn(batch, dims[-1]).astype(np.float32) * 0.01

    times = []
    for i in range(N_WARMUP + n_iters):
        t0 = time.perf_counter()

        opt.zero_grad()
        x = Tensor(x_data)
        y = Tensor(y_data)
        pred = forward(x)
        loss = (pred - y).square().mean()

        t1 = time.perf_counter()
        loss.backward()
        t2 = time.perf_counter()
        opt.step()
        t3 = time.perf_counter()
        lv = loss.item()
        t4 = time.perf_counter()

        rec = {
            'forward': (t1 - t0) * 1000,
            'backward': (t2 - t1) * 1000,
            'step': (t3 - t2) * 1000,
            'item': (t4 - t3) * 1000,
            'total': (t4 - t0) * 1000,
            'loss': lv,
        }
        times.append(rec)

    return times


print(f"{'='*70}")
print(f"Python MLP Training Benchmark (schedule caching)")
print(f"{'='*70}\n")

for cfg in CONFIGS:
    name = cfg['name']
    dims = cfg['dims']
    batch = cfg['batch']
    n_iters = cfg['iters']

    print(f"--- {name}: MLP({' -> '.join(str(d) for d in dims)}), batch={batch} ---")

    times = run_config(cfg)

    # Iter 0 = cold (compile + schedule), iter 1+ = warm (cache hit)
    cold = times[0]
    warm = times[N_WARMUP:]

    def avg(recs, key):
        return sum(r[key] for r in recs) / len(recs)

    print(f"  Cold (iter 0):  fwd={cold['forward']:8.1f}ms  bwd={cold['backward']:8.1f}ms  "
          f"step={cold['step']:8.1f}ms  total={cold['total']:8.1f}ms")
    print(f"  Warm (avg):     fwd={avg(warm,'forward'):8.1f}ms  bwd={avg(warm,'backward'):8.1f}ms  "
          f"step={avg(warm,'step'):8.1f}ms  total={avg(warm,'total'):8.1f}ms")
    speedup = cold['total'] / avg(warm, 'total') if avg(warm, 'total') > 0 else float('inf')
    print(f"  Speedup:        {speedup:.0f}x (cold/warm)")
    print(f"  Loss:           {cold['loss']:.6f} -> {warm[-1]['loss']:.6f}")
    print(f"  Throughput:     {1000/avg(warm,'total'):.1f} iter/sec (warm)")
    print()

print(f"{'='*70}")
print(f"Schedule caching eliminates poly_schedule_v2() on cache hit.")
print(f"Remaining warm time is pure kernel execution (scalar C loops).")
print(f"{'='*70}")
