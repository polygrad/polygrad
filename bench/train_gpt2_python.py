#!/usr/bin/env python
"""Benchmark: GPT-2 training loop with per-phase timing.

Measures forward, backward, and optimizer step time per iteration
for a tiny GPT-2 model. Demonstrates end-to-end training through
the segment-wise backward pipeline.
"""

import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'py'))

from polygrad import Tensor
from polygrad.nn import Adam, get_parameters
from polygrad.nn.gpt2 import GPT2
import numpy as np

# ── Configuration ──────────────────────────────────────────────────────

CONFIGS = [
    {'name': 'tiny-1L',  'n_layers': 1, 'n_heads': 1, 'dim': 32,
     'vocab_size': 50, 'seq_len': 4, 'iters': 5},
    {'name': 'tiny-2L',  'n_layers': 2, 'n_heads': 2, 'dim': 64,
     'vocab_size': 256, 'seq_len': 8, 'iters': 3},
]

LR = 0.001
N_WARMUP = 1


def run_config(cfg):
    Tensor.manual_seed(42)
    np.random.seed(42)

    model = GPT2(
        n_layers=cfg['n_layers'], n_heads=cfg['n_heads'],
        dim=cfg['dim'], vocab_size=cfg['vocab_size'],
        max_seq_len=cfg['seq_len'] * 2,
    )
    params = get_parameters(model)
    n_params = sum(int(np.prod(p.shape)) for p in params)
    opt = Adam(params, lr=LR)

    tokens = np.random.randint(0, cfg['vocab_size'],
                               (1, cfg['seq_len'])).astype(np.float32)

    times = []
    for i in range(N_WARMUP + cfg['iters']):
        t0 = time.perf_counter()

        opt.zero_grad()
        x = Tensor(tokens)
        logits = model(x)
        logits.realize()
        loss = (logits * logits).sum()
        loss.realize()

        t1 = time.perf_counter()
        loss.backward()
        t2 = time.perf_counter()
        lv = loss.item()
        opt.step()
        t3 = time.perf_counter()

        rec = {
            'forward': (t1 - t0) * 1000,
            'backward': (t2 - t1) * 1000,
            'step': (t3 - t2) * 1000,
            'total': (t3 - t0) * 1000,
            'loss': lv,
        }
        times.append(rec)

    return times, n_params


print(f"{'='*70}")
print(f"GPT-2 Training Benchmark")
print(f"{'='*70}\n")

for cfg in CONFIGS:
    name = cfg['name']
    print(f"--- {name}: GPT2(L={cfg['n_layers']}, H={cfg['n_heads']}, "
          f"D={cfg['dim']}, V={cfg['vocab_size']}, T={cfg['seq_len']}) ---")

    times, n_params = run_config(cfg)
    print(f"  Parameters: {len(times)} tensors, {n_params:,} weights")

    cold = times[0]
    warm = times[N_WARMUP:]

    def avg(recs, key):
        return sum(r[key] for r in recs) / len(recs)

    print(f"  Cold (iter 0):  fwd={cold['forward']:8.1f}ms  "
          f"bwd={cold['backward']:8.1f}ms  step={cold['step']:8.1f}ms  "
          f"total={cold['total']:8.1f}ms")
    print(f"  Warm (avg):     fwd={avg(warm,'forward'):8.1f}ms  "
          f"bwd={avg(warm,'backward'):8.1f}ms  step={avg(warm,'step'):8.1f}ms  "
          f"total={avg(warm,'total'):8.1f}ms")
    print(f"  Loss:           {cold['loss']:.4f} -> {warm[-1]['loss']:.4f}")
    print(f"  Throughput:     {1000/avg(warm,'total'):.2f} iter/sec (warm)")
    print()

print(f"{'='*70}")
print(f"Segment-wise backward through realize() boundaries.")
print(f"{'='*70}")
