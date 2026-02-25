"""
bench_tinygrad.py — Benchmark tinygrad CPU backend for comparison with polygrad.

Extended benchmark scope:
- Forward elementwise kernels (legacy section)
- Graph kernels (reduce + movement chains)
- Autograd kernels

Usage: python bench_tinygrad.py [N] [iters_elementwise] [iters_graph]
"""

import os
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'references' / 'tinygrad'))

os.environ['DEBUG'] = '0'
os.environ['VIZ'] = '0'
os.environ['PROFILE'] = '0'
os.environ['TRACEMETA'] = '0'
os.environ['CACHELEVEL'] = '0'

from tinygrad import Device, Tensor  # noqa: E402
from tinygrad.helpers import Context  # noqa: E402

Device.DEFAULT = 'CPU'
LN2_F = np.float32(np.log(2.0))


def flatten(x):
    return np.asarray(x, dtype=np.float32).reshape(-1)


def print_row(name, iters, compile_us, exec_us, ok):
    print(f'{name:<22} {iters:8d} {compile_us:12.0f} {exec_us:12.1f} {"PASS" if ok else "FAIL":>10}')


def run_case(name, iters, run_once, expected, atol=1e-3):
    t0 = time.monotonic()
    out = flatten(run_once())
    compile_us = (time.monotonic() - t0) * 1e6

    t1 = time.monotonic()
    for _ in range(iters):
        out = flatten(run_once())
    exec_us = (time.monotonic() - t1) / iters * 1e6

    ok = np.allclose(out, flatten(expected), atol=atol, rtol=1e-3)
    print_row(name, iters, compile_us, exec_us, ok)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    iters_elementwise = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    iters_graph = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    n = max(1, n)
    iters_elementwise = max(1, iters_elementwise)
    iters_graph = max(1, iters_graph)

    print(f'tinygrad benchmark  N={n}  iters_elementwise={iters_elementwise}  iters_graph={iters_graph}')
    print(f'{"case":<22} {"iters":>8} {"compile_us":>12} {"exec_us":>12} {"correct":>10}')
    print(f'{"----------------------":<22} {"--------":>8} {"------------":>12} {"------------":>12} {"----------":>10}')

    with Context(DEBUG=0, VIZ=0, PROFILE=0, TRACEMETA=0, CACHELEVEL=0):
        print('\n[forward_elementwise]')
        a = np.arange(1, n + 1, dtype=np.float32)
        b = a * np.float32(0.5)
        x_exp2 = (np.arange(n, dtype=np.float32) % np.float32(9.0) - np.float32(4.0)) * np.float32(0.5)

        run_case('add', iters_elementwise, lambda: (Tensor(a.copy()) + Tensor(b.copy())).numpy(), a + b)
        run_case('mul', iters_elementwise, lambda: (Tensor(a.copy()) * Tensor(b.copy())).numpy(), a * b)
        run_case('sub', iters_elementwise, lambda: (Tensor(a.copy()) - Tensor(b.copy())).numpy(), a - b)
        run_case('neg', iters_elementwise, lambda: (-Tensor(a.copy())).numpy(), -a)
        run_case('sqrt', iters_elementwise, lambda: Tensor(a.copy()).sqrt().numpy(), np.sqrt(a))
        run_case('exp2', iters_elementwise, lambda: Tensor(x_exp2.copy()).exp2().numpy(), np.exp2(x_exp2))

        print('\n[graph_and_autograd]')
        cols = 256
        rows = max(1, n // cols)
        total_reduce = rows * cols
        red_in = np.arange(1, total_reduce + 1, dtype=np.float32).reshape(rows, cols)
        run_case(
            'reduce_sum_axis1',
            iters_graph,
            lambda: Tensor(red_in.copy()).sum(axis=1).numpy(),
            red_in.sum(axis=1),
        )

        move_in = np.arange(1, n + 1, dtype=np.float32)
        run_case(
            'chain_pad_flip',
            iters_graph,
            lambda: Tensor(move_in.copy()).pad(((1, 1),))[::-1].numpy(),
            np.pad(move_in, (1, 1), mode='constant')[::-1],
        )

        x_mul = (np.arange(n, dtype=np.float32) % np.float32(101.0)) - np.float32(50.0)
        run_case(
            'grad_mul_sum',
            iters_graph,
            lambda: _grad_mul_sum(x_mul),
            2.0 * x_mul,
        )

        x_grad_exp2 = (np.arange(n, dtype=np.float32) % np.float32(9.0) - np.float32(4.0)) * np.float32(0.5)
        run_case(
            'grad_exp2_sum',
            iters_graph,
            lambda: _grad_exp2_sum(x_grad_exp2),
            np.exp2(x_grad_exp2) * LN2_F,
            atol=2e-3,
        )

        x_div = (np.arange(n, dtype=np.float32) % np.float32(17.0)) - np.float32(8.0)
        y_div = (np.arange(n, dtype=np.float32) % np.float32(11.0)) - np.float32(5.0)
        y_div = np.where(y_div == 0.0, 1.0, y_div) * np.float32(0.5)
        run_case(
            'grad_fdiv_sum_y',
            iters_graph,
            lambda: _grad_fdiv_sum_y(x_div, y_div),
            -x_div / (y_div * y_div),
            atol=2e-3,
        )

        cols_move = 3
        rows_move = max(1, n // cols_move)
        total_move = rows_move * cols_move
        x_move = np.arange(1, total_move + 1, dtype=np.float32)
        run_case(
            'grad_chain_movement',
            iters_graph,
            lambda: _grad_chain_movement(x_move, rows_move, cols_move),
            np.ones((total_move,), dtype=np.float32),
        )


def _grad_mul_sum(x):
    t = Tensor(x.copy(), requires_grad=True)
    (t * t).sum().backward()
    return t.grad.numpy()


def _grad_exp2_sum(x):
    t = Tensor(x.copy(), requires_grad=True)
    t.exp2().sum().backward()
    return t.grad.numpy()


def _grad_fdiv_sum_y(x, y):
    tx = Tensor(x.copy(), requires_grad=False)
    ty = Tensor(y.copy(), requires_grad=True)
    (tx / ty).sum().backward()
    return ty.grad.numpy()


def _grad_chain_movement(x, rows, cols):
    t = Tensor(x.copy(), requires_grad=True)
    t.reshape(rows, cols).permute(1, 0).sum().backward()
    return t.grad.numpy()


if __name__ == '__main__':
    main()
