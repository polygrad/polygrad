#!/usr/bin/env python3
"""CPU parity benchmark for Polygrad vs references/tinygrad_latest.

This is a regression guard for eager realization and cache behavior. It runs
Polygrad and tinygrad in separate subprocesses, with matching workloads,
matching input-reuse policy, and clean per-run compiler cache directories.

Default mode measures op construction plus realize(), not readback. Correctness
is checked outside the timed region. Use --include-readback when readback cost is
the thing being investigated.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import typing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np


WORKLOADS = ("add_1024", "chain_1024", "sum_1024", "movement_1024", "matmul_16", "backward_1024")
FRESH_ONLY = {"backward_1024"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _modes(mode: str) -> list[str]:
    return ["reuse-inputs", "fresh-inputs"] if mode == "both" else [mode]


def _median_us(values: list[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def _percentile_us(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, round((len(xs) - 1) * q)))
    return float(xs[idx])


def _as_float_array(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _check(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    actual = _as_float_array(actual)
    expected = _as_float_array(expected)
    if actual.shape != expected.shape:
        return {
            "ok": False,
            "max_abs_err": None,
            "reason": f"shape mismatch {actual.shape} != {expected.shape}",
        }
    if actual.size == 0:
        return {"ok": True, "max_abs_err": 0.0, "reason": None}
    max_abs_err = float(np.max(np.abs(actual - expected)))
    ok = bool(np.allclose(actual, expected, rtol=1e-4, atol=1e-4))
    return {"ok": ok, "max_abs_err": max_abs_err, "reason": None if ok else "value mismatch"}


def _tensor_make(Tensor: Any, data: np.ndarray, *, requires_grad: bool = False):
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


def _build_workload(Tensor: Any, name: str, mode: str) -> tuple[Callable[[], Any], np.ndarray]:
    if name in FRESH_ONLY and mode != "fresh-inputs":
        raise ValueError(f"{name} requires --mode fresh-inputs or --mode both")

    def tensor(x: np.ndarray, *, requires_grad: bool = False):
        return _tensor_make(Tensor, x, requires_grad=requires_grad)

    if name == "add_1024":
        a_np = np.linspace(-1.0, 1.0, 1024, dtype=np.float32)
        b_np = np.linspace(0.25, 2.25, 1024, dtype=np.float32)
        expected = a_np + b_np
        if mode == "reuse-inputs":
            a, b = tensor(a_np), tensor(b_np)
            return lambda: a + b, expected
        return lambda: tensor(a_np) + tensor(b_np), expected

    if name == "chain_1024":
        a_np = np.linspace(-0.75, 1.25, 1024, dtype=np.float32)
        b_np = np.linspace(0.5, 1.5, 1024, dtype=np.float32)
        expected = (a_np + b_np) * np.float32(2.0) - b_np
        if mode == "reuse-inputs":
            a, b = tensor(a_np), tensor(b_np)
            return lambda: (a + b) * 2.0 - b, expected

        def run_chain():
            a, b = tensor(a_np), tensor(b_np)
            return (a + b) * 2.0 - b

        return run_chain, expected

    if name == "sum_1024":
        a_np = np.linspace(-1.0, 1.0, 1024, dtype=np.float32)
        expected = np.asarray(a_np.sum(), dtype=np.float32)
        if mode == "reuse-inputs":
            a = tensor(a_np)
            return lambda: a.sum(), expected
        return lambda: tensor(a_np).sum(), expected

    if name == "movement_1024":
        a_np = np.arange(1024, dtype=np.float32).reshape(32, 32)
        expected = a_np.reshape(16, 64).transpose(1, 0).copy()
        if mode == "reuse-inputs":
            a = tensor(a_np)
            return lambda: a.reshape(16, 64).permute(1, 0).contiguous(), expected
        return lambda: tensor(a_np).reshape(16, 64).permute(1, 0).contiguous(), expected

    if name == "matmul_16":
        a_np = (np.arange(256, dtype=np.float32).reshape(16, 16) - 50.0) / 64.0
        b_np = (np.arange(256, dtype=np.float32).reshape(16, 16).T + 3.0) / 32.0
        expected = a_np @ b_np
        if mode == "reuse-inputs":
            a, b = tensor(a_np), tensor(b_np)
            return lambda: a.matmul(b), expected
        return lambda: tensor(a_np).matmul(tensor(b_np)), expected

    if name == "backward_1024":
        a_np = np.linspace(-1.0, 1.0, 1024, dtype=np.float32)
        expected = np.float32(2.0) * a_np

        def run_backward():
            a = tensor(a_np, requires_grad=True)
            loss = (a * a).sum()
            loss.backward()
            return a.grad

        return run_backward, expected

    raise KeyError(name)


def _execute(op: Callable[[], Any], include_readback: bool) -> tuple[Any, np.ndarray | None]:
    out = op()
    if out is None:
        raise RuntimeError("workload returned None")
    if include_readback:
        return out, np.asarray(out.numpy())
    out.realize()
    return out, None


def _time_one(op: Callable[[], Any], include_readback: bool) -> tuple[float, Any, np.ndarray | None]:
    t0 = time.perf_counter()
    out, arr = _execute(op, include_readback)
    return (time.perf_counter() - t0) * 1e6, out, arr


def _run_workload(Tensor: Any, cache_stats: Callable[[], dict[str, int]], name: str, mode: str,
                  iters: int, warmup: int, include_readback: bool) -> dict[str, Any]:
    gc.collect()
    op, expected = _build_workload(Tensor, name, mode)

    cold_us, cold_out, cold_arr = _time_one(op, include_readback)
    if cold_arr is None:
        cold_arr = np.asarray(cold_out.numpy())
    correctness = _check(cold_arr, expected)
    cold_out = None
    cold_arr = None
    gc.collect()

    for _ in range(warmup):
        _, out, arr = _time_one(op, include_readback)
        out = None
        arr = None

    times: list[float] = []
    for _ in range(iters):
        # Drop the previous realized result before constructing the next graph.
        out = None
        arr = None
        dt_us, out, arr = _time_one(op, include_readback)
        times.append(dt_us)

    out = None
    arr = None
    gc.collect()
    return {
        "name": name,
        "mode": mode,
        "iters": iters,
        "warmup": warmup,
        "include_readback": include_readback,
        "cold_us": float(cold_us),
        "warm_median_us": _median_us(times),
        "warm_min_us": float(min(times)) if times else float("nan"),
        "warm_p90_us": _percentile_us(times, 0.90),
        "correct": correctness["ok"],
        "max_abs_err": correctness["max_abs_err"],
        "reason": correctness["reason"],
        "cache_stats": cache_stats(),
    }


def _import_polygrad(repo_root: Path, lib_path: Path) -> tuple[Any, Callable[[], dict[str, int]]]:
    os.environ["POLYGRAD_LIB"] = str(lib_path)
    sys.path.insert(0, str(repo_root / "py"))
    import ctypes
    from polygrad import Tensor, _default_ctx, _ffi

    loaded = Path(_ffi._lib._name).resolve()
    expected = lib_path.resolve()
    if loaded != expected:
        raise RuntimeError(f"Polygrad loaded {loaded}, expected {expected}")

    _ffi._lib.poly_schedule_cache_len.restype = ctypes.c_size_t
    _ffi._lib.poly_schedule_cache_len.argtypes = [_ffi._ptr]

    def cache_stats() -> dict[str, int]:
        return {"schedule_cache": int(_ffi._lib.poly_schedule_cache_len(_default_ctx))}

    return Tensor, cache_stats


def _import_tinygrad(tinygrad_path: Path) -> tuple[Any, Callable[[], dict[str, int]]]:
    sys.path.insert(0, str(tinygrad_path))
    if not hasattr(typing, "Self"):
        try:
            from typing_extensions import Self as _Self

            typing.Self = _Self
        except Exception:
            pass
    from tinygrad import Tensor

    def cache_stats() -> dict[str, int]:
        stats: dict[str, int] = {}
        try:
            from tinygrad.schedule import schedule_cache
            stats["schedule_cache"] = len(schedule_cache)
        except Exception:
            pass
        try:
            from tinygrad.codegen import to_program_cache
            stats["to_program_cache"] = len(to_program_cache)
        except Exception:
            pass
        try:
            from tinygrad.engine.realize import runtime_cache
            stats["runtime_cache"] = len(runtime_cache)
        except Exception:
            pass
        return stats

    return Tensor, cache_stats


def _run_worker(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    os.environ.setdefault("DEBUG", "0")
    os.environ.setdefault("VIZ", "0")
    os.environ.setdefault("PROFILE", "0")
    os.environ.setdefault("DEV", "CPU")
    os.environ.setdefault("POLY_DEVICE", "cpu")
    os.environ.setdefault("POLY_CACHE", "1")
    os.environ.setdefault("CACHELEVEL", "2")
    os.environ.setdefault("SCACHE", "1")

    if args.backend == "polygrad":
        Tensor, cache_stats = _import_polygrad(repo_root, Path(args.polygrad_lib).resolve())
    elif args.backend == "tinygrad":
        Tensor, cache_stats = _import_tinygrad(Path(args.tinygrad_path).resolve())
    else:
        raise ValueError(args.backend)

    results = []
    for mode in _modes(args.mode):
        for name in args.workload:
            try:
                results.append(_run_workload(
                    Tensor, cache_stats, name, mode, args.iters, args.warmup, args.include_readback
                ))
            except ValueError as exc:
                results.append({"name": name, "mode": mode, "skipped": True, "reason": str(exc)})
            except Exception as exc:
                results.append({"name": name, "mode": mode, "failed": True, "reason": repr(exc)})

    payload = {
        "backend": args.backend,
        "repo_root": str(repo_root),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


def _worker_env(base: dict[str, str], cache_dir: Path) -> dict[str, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    env = dict(base)
    env.update({
        "DEBUG": "0",
        "VIZ": "0",
        "PROFILE": "0",
        "DEV": "CPU",
        "POLY_DEVICE": "cpu",
        "POLY_CACHE": "1",
        "CACHELEVEL": "2",
        "SCACHE": "1",
        "XDG_CACHE_HOME": str(cache_dir),
    })
    return env


def _run_backend(args: argparse.Namespace, backend: str, cache_dir: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--backend", backend,
        "--repo-root", str(Path(args.repo_root).resolve()),
        "--polygrad-lib", str(Path(args.polygrad_lib).resolve()),
        "--tinygrad-path", str(Path(args.tinygrad_path).resolve()),
        "--mode", args.mode,
        "--iters", str(args.iters),
        "--warmup", str(args.warmup),
    ]
    for name in args.workload:
        cmd.extend(["--workload", name])
    if args.include_readback:
        cmd.append("--include-readback")

    proc = subprocess.run(
        cmd,
        cwd=str(Path(args.repo_root).resolve()),
        env=_worker_env(os.environ, cache_dir),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        if proc.stdout:
            print(proc.stdout, file=sys.stderr)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"{backend} worker exited with {proc.returncode}")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{backend} worker did not produce JSON: {proc.stdout!r}") from exc


def _run_backend_modes(args: argparse.Namespace, backend: str, cache_root: Path) -> dict[str, Any]:
    # Keep modes in separate workers/caches so "first" means first execution for
    # that mode, not "first after a different mode already warmed the caches".
    combined: dict[str, Any] | None = None
    results: list[dict[str, Any]] = []
    for mode in _modes(args.mode):
        mode_args = argparse.Namespace(**vars(args))
        mode_args.mode = mode
        payload = _run_backend(mode_args, backend, cache_root / backend / mode)
        if combined is None:
            combined = dict(payload)
        results.extend(payload["results"])
    assert combined is not None
    combined["mode"] = args.mode
    combined["results"] = results
    return combined


def _result_map(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(r.get("name"), r.get("mode")): r for r in payload["results"]}


def _format_us(x: float | None) -> str:
    if x is None:
        return "-"
    return f"{x:9.1f}"


def _print_table(pg_payload: dict[str, Any], tg_payload: dict[str, Any], threshold: float) -> list[dict[str, Any]]:
    pg = _result_map(pg_payload)
    tg = _result_map(tg_payload)
    rows: list[dict[str, Any]] = []
    keys = sorted(set(pg) | set(tg), key=lambda x: (x[1], x[0]))

    print()
    print("Polygrad vs tinygrad_latest CPU parity benchmark")
    print("Timing: op construction + realize()" + (" + readback" if any(
        r.get("include_readback") for r in pg.values() if isinstance(r, dict)
    ) else ""))
    print()
    print(f"{'workload':<18} {'mode':<12} {'pg first':>9} {'tg first':>9} "
          f"{'pg warm':>9} {'tg warm':>9} {'ratio':>7} {'status':>8}")
    print("-" * 88)

    for key in keys:
        p = pg.get(key)
        t = tg.get(key)
        name, mode = key
        status = "OK"
        ratio: float | None = None

        if p is None or t is None:
            status = "MISSING"
        elif p.get("skipped") or t.get("skipped"):
            status = "SKIP"
        elif p.get("failed") or t.get("failed"):
            status = "FAIL"
        elif not p.get("correct") or not t.get("correct"):
            status = "BADVAL"
        else:
            tg_warm = float(t["warm_median_us"])
            ratio = float(p["warm_median_us"]) / tg_warm if tg_warm > 0 else float("inf")
            status = "OK" if ratio <= threshold else "SLOW"

        rows.append({"name": name, "mode": mode, "polygrad": p, "tinygrad": t,
                     "ratio": ratio, "status": status})
        ratio_s = "-" if ratio is None else f"{ratio:7.3f}"
        print(f"{name:<18} {mode:<12} "
              f"{_format_us(None if not p or 'cold_us' not in p else p['cold_us'])} "
              f"{_format_us(None if not t or 'cold_us' not in t else t['cold_us'])} "
              f"{_format_us(None if not p or 'warm_median_us' not in p else p['warm_median_us'])} "
              f"{_format_us(None if not t or 'warm_median_us' not in t else t['warm_median_us'])} "
              f"{ratio_s} {status:>8}")

    print()
    print("Cache stats after worker run:")
    print(f"  polygrad: {pg_payload['results'][-1].get('cache_stats', {}) if pg_payload['results'] else {}}")
    print(f"  tinygrad: {tg_payload['results'][-1].get('cache_stats', {}) if tg_payload['results'] else {}}")
    return rows


def _default_workloads(mode: str) -> list[str]:
    names = ["add_1024", "chain_1024", "sum_1024", "movement_1024", "matmul_16"]
    if mode in {"fresh-inputs", "both"}:
        names.append("backward_1024")
    return names


def _run_driver(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    polygrad_lib = Path(args.polygrad_lib).resolve()
    tinygrad_path = Path(args.tinygrad_path).resolve()

    if not polygrad_lib.is_file():
        raise SystemExit(f"missing Polygrad library: {polygrad_lib}\nRun: make build/libpolygrad.so")
    if not tinygrad_path.is_dir():
        raise SystemExit(f"missing tinygrad reference tree: {tinygrad_path}")

    cache_root: Path | None = None
    if args.keep_cache:
        cache_root = Path(args.cache_root).resolve() if args.cache_root else repo_root / "bench" / ".cache" / "tinygrad_parity"
        cache_root.mkdir(parents=True, exist_ok=True)
    else:
        cache_root = Path(tempfile.mkdtemp(prefix="polygrad-bench-parity-"))

    try:
        pg_payload = _run_backend_modes(args, "polygrad", cache_root)
        tg_payload = _run_backend_modes(args, "tinygrad", cache_root)
        rows = _print_table(pg_payload, tg_payload, args.fail_threshold)

        output = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "polygrad": pg_payload,
            "tinygrad": tg_payload,
            "rows": rows,
            "fail_threshold": args.fail_threshold,
            "cache_root": str(cache_root),
        }
        if args.json:
            json_path = Path(args.json).resolve()
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
            print(f"Wrote {json_path}")

        bad = [r for r in rows if r["status"] not in {"OK", "SKIP"}]
        if bad and not args.no_fail:
            print("Benchmark failed:", ", ".join(f"{r['name']}:{r['mode']}={r['status']}" for r in bad),
                  file=sys.stderr)
            return 1
        return 0
    finally:
        if cache_root is not None and not args.keep_cache:
            shutil.rmtree(cache_root, ignore_errors=True)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--backend", choices=("polygrad", "tinygrad"), help=argparse.SUPPRESS)
    parser.add_argument("--repo-root", default=str(_repo_root()))
    parser.add_argument("--polygrad-lib", default=str(_repo_root() / "build" / "libpolygrad.so"))
    parser.add_argument("--tinygrad-path", default=str(_repo_root() / "references" / "tinygrad_latest"))
    parser.add_argument("--workload", action="append", choices=WORKLOADS,
                        help="Workload to run. Repeat to select multiple. Default: core forward workloads.")
    parser.add_argument("--mode", choices=("reuse-inputs", "fresh-inputs", "both"), default="reuse-inputs",
                        help="Whether inputs are reused across iterations or recreated inside the timed loop.")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--include-readback", action="store_true",
                        help="Time numpy()/readback too. Default times realize() only.")
    parser.add_argument("--fail-threshold", type=float, default=1.10,
                        help="Fail when Polygrad warm median / tinygrad warm median exceeds this value.")
    parser.add_argument("--no-fail", action="store_true", help="Always exit 0 after printing the table.")
    parser.add_argument("--json", help="Optional JSON output path.")
    parser.add_argument("--keep-cache", action="store_true",
                        help="Keep and reuse benchmark compiler caches under --cache-root.")
    parser.add_argument("--cache-root", help="Cache root used with --keep-cache.")
    args = parser.parse_args(argv)
    if args.worker and not args.backend:
        parser.error("--worker requires --backend")
    if args.workload is None:
        args.workload = _default_workloads(args.mode)
    if args.iters <= 0:
        parser.error("--iters must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.worker:
        return _run_worker(args)
    return _run_driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
