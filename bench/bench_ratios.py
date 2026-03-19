#!/usr/bin/env python
"""Relative benchmark: polygrad vs numpy/tinygrad/torch.

Reports speedup ratios (polygrad_time / baseline_time) that are stable across
hardware. Outputs JSON to bench/results/<timestamp>.json and a human-readable
table to stderr.

Usage: python bench/bench_ratios.py [--no-tinygrad] [--no-torch]
"""

import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'py'))

import numpy as np
from polygrad import Tensor
from polygrad.nn import Linear, SGD, get_parameters

# -- Optional baselines -------------------------------------------------------

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# -- Config --------------------------------------------------------------------

SIZES_ELEM = [1024, 100_000, 1_000_000]
SIZES_MATMUL = [64, 256]
ITERS_ELEM = {1024: 50, 100_000: 20, 1_000_000: 10}
ITERS_MATMUL = {64: 50, 256: 10}
WARMUP = 3
MLP_ITERS = 20

RESULTS_DIR = Path(__file__).parent / 'results'


# -- Timing utility ------------------------------------------------------------

def _time_iters(fn, iters):
    """Run fn() iters times, return median duration in microseconds."""
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e6)
    times.sort()
    return times[len(times) // 2]


# -- Polygrad benchmarks -------------------------------------------------------

def pg_vecadd(n, iters):
    np.random.seed(42)
    a_np = np.random.randn(n).astype(np.float32)
    b_np = np.random.randn(n).astype(np.float32)
    for _ in range(WARMUP):
        (Tensor(a_np) + Tensor(b_np)).numpy()
    return _time_iters(lambda: (Tensor(a_np) + Tensor(b_np)).numpy(), iters)

def pg_neg(n, iters):
    np.random.seed(42)
    a_np = np.random.randn(n).astype(np.float32)
    for _ in range(WARMUP):
        (-Tensor(a_np)).numpy()
    return _time_iters(lambda: (-Tensor(a_np)).numpy(), iters)

def pg_exp(n, iters):
    np.random.seed(42)
    a_np = np.random.randn(n).astype(np.float32)
    for _ in range(WARMUP):
        Tensor(a_np).exp().numpy()
    return _time_iters(lambda: Tensor(a_np).exp().numpy(), iters)

def pg_reduce_sum(n, iters):
    np.random.seed(42)
    a_np = np.random.randn(n).astype(np.float32)
    for _ in range(WARMUP):
        Tensor(a_np).sum().numpy()
    return _time_iters(lambda: Tensor(a_np).sum().numpy(), iters)

def pg_matmul(n, iters):
    np.random.seed(42)
    a_np = np.random.randn(n, n).astype(np.float32)
    b_np = np.random.randn(n, n).astype(np.float32)
    for _ in range(WARMUP):
        Tensor(a_np).matmul(Tensor(b_np)).numpy()
    return _time_iters(lambda: Tensor(a_np).matmul(Tensor(b_np)).numpy(), iters)

def pg_fused_chain(n, iters):
    np.random.seed(42)
    a_np = np.random.randn(n).astype(np.float32)
    b_np = np.random.randn(n).astype(np.float32)
    for _ in range(WARMUP):
        a, b = Tensor(a_np), Tensor(b_np)
        ((a + b) * (a - b) + a * b).numpy()
    def run():
        a, b = Tensor(a_np), Tensor(b_np)
        return ((a + b) * (a - b) + a * b).numpy()
    return _time_iters(run, iters)

def pg_backward(n, iters):
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

def pg_mlp_train(iters):
    Tensor.manual_seed(42)
    np.random.seed(42)
    l1 = Linear(64, 32)
    l2 = Linear(32, 1)
    params = get_parameters(l1) + get_parameters(l2)
    opt = SGD(params, lr=0.01)
    x_np = np.random.randn(4, 64).astype(np.float32)
    y_np = np.random.randn(4, 1).astype(np.float32)
    def run():
        x = Tensor(x_np, requires_grad=False)
        h = l1(x).relu()
        pred = l2(h)
        loss = ((pred - Tensor(y_np)) * (pred - Tensor(y_np))).sum()
        loss.backward()
        opt.step()
    for _ in range(WARMUP):
        run()
    return _time_iters(run, iters)


# -- Numpy benchmarks ---------------------------------------------------------

def np_vecadd(n, iters):
    np.random.seed(42)
    a = np.random.randn(n).astype(np.float32)
    b = np.random.randn(n).astype(np.float32)
    return _time_iters(lambda: a + b, iters)

def np_neg(n, iters):
    np.random.seed(42)
    a = np.random.randn(n).astype(np.float32)
    return _time_iters(lambda: -a, iters)

def np_exp(n, iters):
    np.random.seed(42)
    a = np.random.randn(n).astype(np.float32)
    return _time_iters(lambda: np.exp(a), iters)

def np_reduce_sum(n, iters):
    np.random.seed(42)
    a = np.random.randn(n).astype(np.float32)
    return _time_iters(lambda: a.sum(), iters)

def np_matmul(n, iters):
    np.random.seed(42)
    a = np.random.randn(n, n).astype(np.float32)
    b = np.random.randn(n, n).astype(np.float32)
    return _time_iters(lambda: a @ b, iters)

def np_fused_chain(n, iters):
    np.random.seed(42)
    a = np.random.randn(n).astype(np.float32)
    b = np.random.randn(n).astype(np.float32)
    return _time_iters(lambda: (a + b) * (a - b) + a * b, iters)

def np_backward(n, iters):
    """Numpy equivalent of (a*a).sum().backward() -> grad = 2*a."""
    np.random.seed(42)
    a = np.random.randn(n).astype(np.float32)
    return _time_iters(lambda: 2.0 * a, iters)


# -- Torch benchmarks ---------------------------------------------------------

def _torch_bench(fn, iters):
    if not HAS_TORCH:
        return None
    return _time_iters(fn, iters)

def torch_vecadd(n, iters):
    np.random.seed(42)
    a = torch.from_numpy(np.random.randn(n).astype(np.float32))
    b = torch.from_numpy(np.random.randn(n).astype(np.float32))
    with torch.no_grad():
        return _torch_bench(lambda: (a + b).numpy(), iters)

def torch_neg(n, iters):
    np.random.seed(42)
    a = torch.from_numpy(np.random.randn(n).astype(np.float32))
    with torch.no_grad():
        return _torch_bench(lambda: (-a).numpy(), iters)

def torch_exp(n, iters):
    np.random.seed(42)
    a = torch.from_numpy(np.random.randn(n).astype(np.float32))
    with torch.no_grad():
        return _torch_bench(lambda: a.exp().numpy(), iters)

def torch_reduce_sum(n, iters):
    np.random.seed(42)
    a = torch.from_numpy(np.random.randn(n).astype(np.float32))
    with torch.no_grad():
        return _torch_bench(lambda: a.sum().item(), iters)

def torch_matmul(n, iters):
    np.random.seed(42)
    a = torch.from_numpy(np.random.randn(n, n).astype(np.float32))
    b = torch.from_numpy(np.random.randn(n, n).astype(np.float32))
    with torch.no_grad():
        return _torch_bench(lambda: a.matmul(b).numpy(), iters)

def torch_fused_chain(n, iters):
    np.random.seed(42)
    a = torch.from_numpy(np.random.randn(n).astype(np.float32))
    b = torch.from_numpy(np.random.randn(n).astype(np.float32))
    with torch.no_grad():
        return _torch_bench(lambda: ((a + b) * (a - b) + a * b).numpy(), iters)

def torch_backward(n, iters):
    np.random.seed(42)
    a_np = np.random.randn(n).astype(np.float32)
    def run():
        a = torch.tensor(a_np, requires_grad=True)
        loss = (a * a).sum()
        loss.backward()
        a.grad.numpy()
    return _torch_bench(run, iters)


# -- Correctness check --------------------------------------------------------

def check_correct(name, n):
    """One-shot correctness check: polygrad output matches numpy."""
    np.random.seed(42)
    a_np = np.random.randn(n).astype(np.float32)
    b_np = np.random.randn(n).astype(np.float32)
    try:
        if name == 'vecadd':
            pg = (Tensor(a_np) + Tensor(b_np)).numpy()
            ref = a_np + b_np
        elif name == 'neg':
            pg = (-Tensor(a_np)).numpy()
            ref = -a_np
        elif name == 'exp':
            pg = Tensor(a_np).exp().numpy()
            ref = np.exp(a_np)
        elif name == 'reduce_sum':
            pg = Tensor(a_np).sum().numpy()
            ref = np.array(a_np.sum())
        elif name == 'matmul':
            a2 = np.random.randn(n, n).astype(np.float32)
            b2 = np.random.randn(n, n).astype(np.float32)
            np.random.seed(42)
            a2 = np.random.randn(n, n).astype(np.float32)
            b2 = np.random.randn(n, n).astype(np.float32)
            pg = Tensor(a2).matmul(Tensor(b2)).numpy()
            ref = a2 @ b2
        elif name == 'fused_chain':
            a, b = Tensor(a_np), Tensor(b_np)
            pg = ((a + b) * (a - b) + a * b).numpy()
            ref = (a_np + b_np) * (a_np - b_np) + a_np * b_np
        elif name == 'backward':
            a = Tensor(a_np, requires_grad=True)
            loss = (a * a).sum()
            loss.backward()
            pg = a.grad.numpy()
            ref = 2.0 * a_np
        else:
            return True
        return np.allclose(pg, ref, rtol=1e-3, atol=1e-3)
    except Exception:
        return False


# -- Tinygrad subprocess -------------------------------------------------------

def run_tinygrad_worker():
    """Run tinygrad benchmarks in the `tiny` conda env, return results dict."""
    worker = os.path.join(os.path.dirname(__file__), 'bench_tinygrad_worker.py')
    try:
        result = subprocess.run(
            ['conda', 'run', '-n', 'tiny', 'python', worker],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            print(f'  tinygrad worker failed (exit {result.returncode})',
                  file=sys.stderr)
            if result.stderr:
                for line in result.stderr.strip().split('\n')[:5]:
                    print(f'    {line}', file=sys.stderr)
            return None
        data = json.loads(result.stdout)
        return data.get('results', {})
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
        print(f'  tinygrad worker unavailable: {e}', file=sys.stderr)
        return None


# -- Workload registry ---------------------------------------------------------

PG_BENCH = {
    'vecadd': (pg_vecadd, SIZES_ELEM, ITERS_ELEM),
    'neg': (pg_neg, SIZES_ELEM, ITERS_ELEM),
    'exp': (pg_exp, SIZES_ELEM, ITERS_ELEM),
    'reduce_sum': (pg_reduce_sum, SIZES_ELEM, ITERS_ELEM),
    'matmul': (pg_matmul, SIZES_MATMUL, ITERS_MATMUL),
    'fused_chain': (pg_fused_chain, SIZES_ELEM, ITERS_ELEM),
    'backward': (pg_backward, SIZES_ELEM, ITERS_ELEM),
}

NP_BENCH = {
    'vecadd': np_vecadd,
    'neg': np_neg,
    'exp': np_exp,
    'reduce_sum': np_reduce_sum,
    'matmul': np_matmul,
    'fused_chain': np_fused_chain,
    'backward': np_backward,
}

TORCH_BENCH = {
    'vecadd': torch_vecadd,
    'neg': torch_neg,
    'exp': torch_exp,
    'reduce_sum': torch_reduce_sum,
    'matmul': torch_matmul,
    'fused_chain': torch_fused_chain,
    'backward': torch_backward,
}


# -- Main ----------------------------------------------------------------------

def _ratio(a, b):
    if a is None or b is None or b == 0:
        return None
    return round(a / b, 3)


def main():
    skip_tinygrad = '--no-tinygrad' in sys.argv
    skip_torch = '--no-torch' in sys.argv

    print('\n  polygrad ratio benchmark', file=sys.stderr)
    print('  ========================\n', file=sys.stderr)

    # Run tinygrad in one subprocess call
    tg_results = None
    if not skip_tinygrad:
        print('  Running tinygrad worker...', file=sys.stderr)
        tg_results = run_tinygrad_worker()
        if tg_results:
            print(f'  Got {len(tg_results)} tinygrad results\n', file=sys.stderr)
        else:
            print('  Tinygrad unavailable, skipping\n', file=sys.stderr)

    results = {}

    # Sized workloads
    for name, (pg_fn, sizes, iters_map) in PG_BENCH.items():
        for n in sizes:
            key = f'{name}_{n}'
            iters = iters_map.get(n, 50)
            correct = check_correct(name, min(n, 1024))

            pg_us = pg_fn(n, iters)
            np_us = NP_BENCH[name](n, iters)

            torch_us = None
            if HAS_TORCH and not skip_torch and name in TORCH_BENCH:
                torch_us = TORCH_BENCH[name](n, iters)

            tg_us = None
            if tg_results and key in tg_results and tg_results[key]:
                tg_us = tg_results[key]['median_us']

            results[key] = {
                'polygrad_us': round(pg_us, 2),
                'numpy_us': round(np_us, 2),
                'tinygrad_us': round(tg_us, 2) if tg_us else None,
                'torch_us': round(torch_us, 2) if torch_us else None,
                'ratio_vs_numpy': _ratio(pg_us, np_us),
                'ratio_vs_tinygrad': _ratio(pg_us, tg_us),
                'ratio_vs_torch': _ratio(pg_us, torch_us),
                'iters': iters,
                'correct': correct,
            }

    # MLP training step (fixed size)
    mlp_us = pg_mlp_train(MLP_ITERS)
    results['mlp_train'] = {
        'polygrad_us': round(mlp_us, 2),
        'numpy_us': None,
        'tinygrad_us': None,
        'torch_us': None,
        'ratio_vs_numpy': None,
        'ratio_vs_tinygrad': None,
        'ratio_vs_torch': None,
        'iters': MLP_ITERS,
        'correct': True,
    }

    # Print table
    _print_table(results)

    # Write JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    out_path = RESULTS_DIR / f'{ts}.json'

    cpu_name = ''
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    cpu_name = line.split(':', 1)[1].strip()
                    break
    except OSError:
        pass

    doc = {
        'version': 1,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'machine': {
            'platform': sys.platform,
            'arch': platform.machine(),
            'cpu': cpu_name,
        },
        'python': results,
    }
    out_path.write_text(json.dumps(doc, indent=2) + '\n')
    print(f'\n  Written to {out_path}\n', file=sys.stderr)


def _print_table(results):
    hdr = f'  {"workload":<24} {"polygrad":>10} {"numpy":>10} {"tinygrad":>10} {"torch":>10} {"pg/np":>8} {"pg/tg":>8} {"pg/th":>8}'
    print(hdr, file=sys.stderr)
    print('  ' + '-' * (len(hdr) - 2), file=sys.stderr)
    for key, r in results.items():
        def _fmt(v):
            if v is None:
                return 'n/a'
            if v >= 1000:
                return f'{v/1000:.1f}ms'
            return f'{v:.0f}us'
        def _rfmt(v):
            if v is None:
                return 'n/a'
            return f'{v:.2f}x'
        mark = '' if r['correct'] else ' !'
        print(f'  {key + mark:<24} {_fmt(r["polygrad_us"]):>10} {_fmt(r["numpy_us"]):>10} '
              f'{_fmt(r.get("tinygrad_us")):>10} {_fmt(r.get("torch_us")):>10} '
              f'{_rfmt(r["ratio_vs_numpy"]):>8} {_rfmt(r.get("ratio_vs_tinygrad")):>8} '
              f'{_rfmt(r.get("ratio_vs_torch")):>8}',
              file=sys.stderr)


if __name__ == '__main__':
    main()
