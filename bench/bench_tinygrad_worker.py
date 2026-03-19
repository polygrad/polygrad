#!/usr/bin/env python
"""Tinygrad benchmark worker -- runs in `tiny` conda env, outputs JSON to stdout.

Usage: conda run -n tiny python bench/bench_tinygrad_worker.py
"""

import json
import os
import sys
import time

os.environ.update({'DEBUG': '0', 'CACHELEVEL': '0', 'VIZ': '0'})

import numpy as np
from tinygrad import Tensor

SIZES_ELEM = [1024, 100_000, 1_000_000]
SIZES_MATMUL = [64, 256]
ITERS_ELEM = {1024: 50, 100_000: 20, 1_000_000: 10}
ITERS_MATMUL = {64: 50, 256: 10}
WARMUP = 3


def _time_iters(fn, iters):
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e6)
    times.sort()
    return times[len(times) // 2]


def bench_vecadd(n, iters):
    np.random.seed(42)
    a = Tensor(np.random.randn(n).astype(np.float32))
    b = Tensor(np.random.randn(n).astype(np.float32))
    for _ in range(WARMUP):
        (a + b).numpy()
    return _time_iters(lambda: (a + b).numpy(), iters)


def bench_neg(n, iters):
    np.random.seed(42)
    a = Tensor(np.random.randn(n).astype(np.float32))
    for _ in range(WARMUP):
        (-a).numpy()
    return _time_iters(lambda: (-a).numpy(), iters)


def bench_exp(n, iters):
    np.random.seed(42)
    a = Tensor(np.random.randn(n).astype(np.float32))
    for _ in range(WARMUP):
        a.exp().numpy()
    return _time_iters(lambda: a.exp().numpy(), iters)


def bench_reduce_sum(n, iters):
    np.random.seed(42)
    a = Tensor(np.random.randn(n).astype(np.float32))
    for _ in range(WARMUP):
        a.sum().numpy()
    return _time_iters(lambda: a.sum().numpy(), iters)


def bench_matmul(n, iters):
    np.random.seed(42)
    a = Tensor(np.random.randn(n, n).astype(np.float32))
    b = Tensor(np.random.randn(n, n).astype(np.float32))
    for _ in range(WARMUP):
        a.matmul(b).numpy()
    return _time_iters(lambda: a.matmul(b).numpy(), iters)


def bench_fused_chain(n, iters):
    np.random.seed(42)
    a = Tensor(np.random.randn(n).astype(np.float32))
    b = Tensor(np.random.randn(n).astype(np.float32))
    for _ in range(WARMUP):
        ((a + b) * (a - b) + a * b).numpy()
    return _time_iters(lambda: ((a + b) * (a - b) + a * b).numpy(), iters)


def bench_backward(n, iters):
    np.random.seed(42)
    a_np = np.random.randn(n).astype(np.float32)

    def run():
        a = Tensor(a_np, requires_grad=True)
        loss = (a * a).sum()
        loss.backward()
        a.grad.numpy()

    for _ in range(WARMUP):
        run()
    return _time_iters(run, iters)


WORKLOADS = {
    'vecadd': (bench_vecadd, SIZES_ELEM, ITERS_ELEM),
    'neg': (bench_neg, SIZES_ELEM, ITERS_ELEM),
    'exp': (bench_exp, SIZES_ELEM, ITERS_ELEM),
    'reduce_sum': (bench_reduce_sum, SIZES_ELEM, ITERS_ELEM),
    'matmul': (bench_matmul, SIZES_MATMUL, ITERS_MATMUL),  # skip 1024 (too slow for tinygrad CPU)
    'fused_chain': (bench_fused_chain, SIZES_ELEM, ITERS_ELEM),
    'backward': (bench_backward, SIZES_ELEM, ITERS_ELEM),
}


def main():
    results = {}
    for name, (fn, sizes, iters_map) in WORKLOADS.items():
        for n in sizes:
            iters = iters_map.get(n, 50)
            try:
                median_us = fn(n, iters)
                results[f'{name}_{n}'] = {'median_us': median_us, 'iters': iters}
            except Exception as e:
                print(f'# tinygrad {name}_{n} failed: {e}', file=sys.stderr)
                results[f'{name}_{n}'] = None
    json.dump({'backend': 'tinygrad', 'results': results}, sys.stdout)


if __name__ == '__main__':
    main()
