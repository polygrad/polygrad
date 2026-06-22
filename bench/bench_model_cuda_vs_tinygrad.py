#!/usr/bin/env python3
"""Model-level CUDA benchmark: Polygrad Instance MLP vs tinygrad MLP.

This is intentionally not a scalar-op microbenchmark. Each timed Polygrad call
uses the public Instance API with host inputs, CUDA execution, and output/loss
readback. The tinygrad side mirrors that shape by constructing host inputs in
the timed loop, moving them to CUDA, realizing, synchronizing, and reading the
result.

Usage:
  PYTHONPATH=references/tinygrad_latest python bench/bench_model_cuda_vs_tinygrad.py
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import textwrap
import time
import typing
from pathlib import Path

import numpy as np

if not hasattr(typing, "Self"):
    try:
        from typing_extensions import Self as _Self

        typing.Self = _Self
    except Exception:
        pass

from tinygrad import Tensor, Device
from tinygrad.helpers import DEV
from tinygrad.nn.optim import SGD


REPO = Path(__file__).resolve().parents[1]
NODE = os.environ.get("NODE") or shutil.which("node") or "/home/anton/tools/node19/bin/node"
CASES = [
    {"name": "mlp_fwd_b32_256_512_128", "layers": [256, 512, 128], "batch": 32, "iters": 20},
    {"name": "mlp_fwd_b16_1024_1024_1024", "layers": [1024, 1024, 1024], "batch": 16, "iters": 10},
    {"name": "mlp_train_b32_32_64_16", "layers": [32, 64, 16], "batch": 32, "iters": 10, "train": True},
]
WARMUP = 3


def median_us(fn, iters: int) -> float:
    vals = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        vals.append((time.perf_counter() - t0) * 1e6)
    return float(statistics.median(vals))


def sync_cuda() -> None:
    Device["CUDA"].synchronize()


def tg_tensor(data: np.ndarray, *, requires_grad: bool = False) -> Tensor:
    if not requires_grad:
        return Tensor(data)
    try:
        return Tensor(data, requires_grad=True)
    except TypeError as exc:
        if "requires_grad" not in str(exc):
            raise
    t = Tensor(data)
    if hasattr(t, "is_param"):
        t.is_param = True
    else:
        t.requires_grad = True
    return t


def init_arrays(layers: list[int], batch: int) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    rng = np.random.default_rng(42)
    x = rng.normal(0.0, 1.0, size=(batch, layers[0])).astype(np.float32)
    y = rng.normal(0.0, 1.0, size=(batch, layers[-1])).astype(np.float32)
    params = []
    for i, (din, dout) in enumerate(zip(layers[:-1], layers[1:])):
        scale = np.float32(1.0 / np.sqrt(max(1, din)))
        w = rng.normal(0.0, float(scale), size=(din, dout)).astype(np.float32)
        b = np.zeros((dout,), dtype=np.float32)
        params.append((w, b))
    return x, y, params


def tinygrad_forward_case(case: dict) -> float:
    layers, batch, iters = case["layers"], case["batch"], case["iters"]
    x_np, _y_np, params_np = init_arrays(layers, batch)
    params = [(tg_tensor(w).realize(), tg_tensor(b).realize()) for w, b in params_np]
    sync_cuda()

    def fwd(x):
        for i, (w, b) in enumerate(params):
            x = x.matmul(w) + b
            if i != len(params) - 1:
                x = x.relu()
        return x

    def run():
        out = fwd(tg_tensor(x_np).realize()).realize()
        sync_cuda()
        _ = out.numpy()

    for _ in range(WARMUP):
        run()
    return median_us(run, iters)


def tinygrad_train_case(case: dict) -> float:
    layers, batch, iters = case["layers"], case["batch"], case["iters"]
    x_np, y_np, params_np = init_arrays(layers, batch)
    Tensor.training = True
    params = []
    for w_np, b_np in params_np:
        w = tg_tensor(w_np, requires_grad=True).realize()
        b = tg_tensor(b_np, requires_grad=True).realize()
        params.extend([w, b])
    opt = SGD(params, lr=0.01)
    sync_cuda()

    def fwd(x):
        for i in range(0, len(params), 2):
            x = x.matmul(params[i]) + params[i + 1]
            if i + 2 < len(params):
                x = x.relu()
        return x

    def run():
        opt.zero_grad()
        pred = fwd(tg_tensor(x_np).realize())
        target = tg_tensor(y_np).realize()
        loss = ((pred - target) * (pred - target)).mean()
        loss.backward()
        opt.step()
        sync_cuda()
        _ = float(loss.numpy())

    for _ in range(WARMUP):
        run()
    return median_us(run, iters)


def run_polygrad_node(cases: list[dict]) -> dict[str, float]:
    js = r"""
    'use strict'
    const polygrad = require(process.cwd() + '/js/src/index')
    const { performance } = require('perf_hooks')

    const cases = JSON.parse(process.argv[2])
    const WARMUP = Number(process.argv[3])

    function medianUs(fn, iters) {
      const vals = []
      for (let i = 0; i < iters; i++) {
        const t0 = performance.now()
        fn()
        vals.push((performance.now() - t0) * 1000)
      }
      vals.sort((a, b) => a - b)
      return vals[Math.floor(vals.length / 2)]
    }

    function fill(arr, seed) {
      let x = seed >>> 0
      for (let i = 0; i < arr.length; i++) {
        x = (1664525 * x + 1013904223) >>> 0
        arr[i] = ((x / 0xffffffff) * 2 - 1)
      }
    }

    async function main() {
      const pg = await polygrad.create({ core: 'native' })
      const out = {}
      for (const tc of cases) {
        const spec = {
          layers: tc.layers,
          activation: 'relu',
          bias: true,
          loss: 'mse',
          batch_size: tc.batch,
          seed: 42
        }
        const inst = pg.models.MLP(spec)
        const x = new Float32Array(tc.batch * tc.layers[0])
        const y = new Float32Array(tc.batch * tc.layers[tc.layers.length - 1])
        fill(x, 1)
        fill(y, 2)
        try {
          if (tc.train) {
            inst.setOptimizer(pg.OPTIM_SGD, 0.01)
            for (let i = 0; i < WARMUP; i++) inst.trainStep({ x, y })
            out[tc.name] = medianUs(() => inst.trainStep({ x, y }), tc.iters)
          } else {
            for (let i = 0; i < WARMUP; i++) inst.forward({ x })
            out[tc.name] = medianUs(() => inst.forward({ x }), tc.iters)
          }
        } finally {
          inst.dispose()
        }
      }
      process.stdout.write(JSON.stringify(out))
    }

    main().catch(err => {
      console.error(err && err.stack || err)
      process.exit(1)
    })
    """
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(textwrap.dedent(js))
        path = f.name
    try:
      env = os.environ.copy()
      env["POLY_DEVICE"] = "cuda"
      proc = subprocess.run(
          [NODE, path, json.dumps(cases), str(WARMUP)],
          cwd=REPO,
          env=env,
          text=True,
          capture_output=True,
          timeout=180,
      )
      if proc.returncode != 0:
          raise RuntimeError(
              "Polygrad Node benchmark failed\n"
              f"stdout:\n{proc.stdout}\n"
              f"stderr:\n{proc.stderr}\n"
          )
      return json.loads(proc.stdout)
    finally:
      try:
        os.unlink(path)
      except OSError:
        pass


def main() -> None:
    DEV.value = "CUDA"
    os.environ.setdefault("CUDA", "1")
    os.environ.setdefault("CACHELEVEL", "2")
    os.environ.setdefault("DEBUG", "0")

    polygrad = run_polygrad_node(CASES)
    rows = []
    for case in CASES:
        if case.get("train"):
            tg_us = tinygrad_train_case(case)
        else:
            tg_us = tinygrad_forward_case(case)
        pg_us = float(polygrad[case["name"]])
        rows.append((case["name"], pg_us, tg_us, tg_us / pg_us if pg_us > 0 else float("inf")))

    print("\n  model-level CUDA benchmark: Polygrad Instance vs tinygrad")
    print("  ========================================================")
    print("  Mode: host input -> CUDA execution -> host output/loss readback")
    print()
    print(f"  {'case':<32} {'polygrad':>12} {'tinygrad':>12} {'tg/polygrad':>12}")
    print("  " + "-" * 72)
    for name, pg_us, tg_us, ratio in rows:
        print(f"  {name:<32} {pg_us:10.0f}us {tg_us:10.0f}us {ratio:11.2f}x")


if __name__ == "__main__":
    main()
