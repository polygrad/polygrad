"""
bench_vs_tinygrad.py — Compare polygrad CUDA vs tinygrad CUDA

Usage:
  conda run -n tiny python bench/bench_vs_tinygrad.py

Measures: compilation time (first call) and execution time (cached, average of N iters).
Both frameworks use CUDA backend with kernel caching.
"""

import os
import time
import subprocess
import sys

os.environ["CUDA"] = "1"
os.environ["CACHELEVEL"] = "2"  # enable tinygrad disk cache
os.environ["DEBUG"] = "0"
os.environ["VERBOSE"] = "0"

from tinygrad import Tensor, Device
Device.DEFAULT = "CUDA"

def sync():
    Device["CUDA"].synchronize()

SIZES = [1024, 10_000, 100_000, 1_000_000]
ITERS_SMALL = 50
ITERS_LARGE = 20

def bench_tinygrad_vecadd(n, iters):
    a = Tensor.rand(n).realize()
    b = Tensor.rand(n).realize()
    sync()

    # Warmup / compile
    c = (a + b).realize()
    sync()

    # Benchmark
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        c = (a + b).realize()
        sync()
    us = (time.perf_counter() - t0) / iters * 1e6
    return us

def bench_tinygrad_mul(n, iters):
    a = Tensor.rand(n).realize()
    b = Tensor.rand(n).realize()
    sync()

    c = (a * b).realize()
    sync()

    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        c = (a * b).realize()
        sync()
    us = (time.perf_counter() - t0) / iters * 1e6
    return us

def bench_tinygrad_reduce_sum(n, iters):
    a = Tensor.rand(n).realize()
    sync()

    c = a.sum().realize()
    sync()

    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        c = a.sum().realize()
        sync()
    us = (time.perf_counter() - t0) / iters * 1e6
    return us

def bench_tinygrad_exp2(n, iters):
    a = Tensor.rand(n).realize()
    sync()

    c = a.exp2().realize()
    sync()

    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        c = a.exp2().realize()
        sync()
    us = (time.perf_counter() - t0) / iters * 1e6
    return us

def bench_tinygrad_chain(n, iters):
    """(a + b) * a - chain of 2 ops fused into one kernel"""
    a = Tensor.rand(n).realize()
    b = Tensor.rand(n).realize()
    sync()

    c = ((a + b) * a).realize()
    sync()

    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        c = ((a + b) * a).realize()
        sync()
    us = (time.perf_counter() - t0) / iters * 1e6
    return us


# ── Run polygrad benchmark ──────────────────────────────────────────────────

def run_polygrad_bench():
    """Run bench_cuda and parse output"""
    result = subprocess.run(
        ["./build/bench_cuda"],
        capture_output=True, text=True, timeout=120
    )
    lines = result.stdout.strip().split("\n")
    polygrad = {}
    for line in lines:
        line = line.strip()
        if not line or "===" in line or ":" == line[-1:]:
            continue
        # Parse: "vecadd  N=1024      CPU:        9 us  GPU:      182 us  speedup: 0.05x"
        parts = line.split()
        if len(parts) >= 8 and "GPU:" in line:
            op = parts[0]
            n_str = parts[1].replace("N=", "")
            gpu_idx = parts.index("GPU:") + 1
            gpu_us = float(parts[gpu_idx])
            cpu_idx = parts.index("CPU:") + 1
            cpu_us = float(parts[cpu_idx])
            polygrad[(op, int(n_str))] = {"cpu": cpu_us, "gpu": gpu_us}
    return polygrad


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  polygrad vs tinygrad CUDA benchmark")
    print("  ================================\n")

    # Build polygrad first
    print("  Building polygrad bench_cuda...")
    subprocess.run(["make", "build/bench_cuda"], capture_output=True, timeout=60)

    # Run polygrad
    print("  Running polygrad benchmark...")
    polygrad = run_polygrad_bench()

    print()

    # Header
    fmt = "  {:<8} N={:<10} {:>10} {:>10} {:>10} {:>10}"
    print(fmt.format("op", "", "tinygrad", "polygrad-gpu", "polygrad-cpu", "tg/polygrad"))
    print("  " + "-" * 72)

    for op_name, bench_fn in [
        ("vecadd", bench_tinygrad_vecadd),
        ("mul", bench_tinygrad_mul),
        ("chain", bench_tinygrad_chain),
        ("exp2", bench_tinygrad_exp2),
        ("reduce", bench_tinygrad_reduce_sum),
    ]:
        for n in SIZES:
            iters = ITERS_LARGE if n >= 100_000 else ITERS_SMALL
            tg_us = bench_fn(n, iters)

            # Lookup polygrad results
            # Map op names: "chain" doesn't exist in polygrad bench
            polygrad_op = op_name
            if polygrad_op == "chain" or polygrad_op == "exp2":
                polygrad_op = None  # no direct polygrad equivalent

            if polygrad_op and (polygrad_op, n) in polygrad:
                t = polygrad[(polygrad_op, n)]
                ratio = tg_us / t["gpu"] if t["gpu"] > 0 else 0
                print(fmt.format(
                    op_name, str(n),
                    f"{tg_us:.0f} us",
                    f"{t['gpu']:.0f} us",
                    f"{t['cpu']:.0f} us",
                    f"{ratio:.2f}x"
                ))
            else:
                print(fmt.format(
                    op_name, str(n),
                    f"{tg_us:.0f} us",
                    "n/a",
                    "n/a",
                    ""
                ))

    print()
