#!/usr/bin/env python3
import argparse
import contextlib
import json
import os
import pathlib
import platform
import subprocess
import sys
from typing import Callable
import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
if sys.version_info < (3, 11):
    raise RuntimeError("tinygrad_latest parity requires Python 3.11+; set PARITY_PY=references/.venv-tinygrad-py311/bin/python")

os.environ.setdefault("CACHELEVEL", "0")
os.environ.setdefault("DEVECTORIZE", "-1")
# Always use CPU for value evaluation; CUDA flag only affects IR extraction
os.environ["DEV"] = "CPU"
sys.path.insert(0, str(ROOT / "references" / "tinygrad_latest"))

from tinygrad import Context, Tensor, dtypes  # noqa: E402
from tinygrad.codegen import to_program, full_rewrite_to_sink  # noqa: E402
from tinygrad.codegen.late.linearizer import linearize  # noqa: E402
from tinygrad.helpers import ContextVar, Target  # noqa: E402
from tinygrad.renderer.cstyle import ClangRenderer, CUDARenderer, HIPRenderer  # noqa: E402
from tinygrad.uop.ops import Ops  # noqa: E402


def _tiny_context(**kwargs):
    supported = getattr(ContextVar, "_cache", {})
    filtered = {k: v for k, v in kwargs.items() if k in supported}
    if not filtered:
        return contextlib.nullcontext()
    return Context(**filtered)


def _cpu_target() -> Target:
    target = Target.parse("CPU")
    if getattr(target, "arch", ""):
        return target
    machine = platform.machine().lower()
    arch = {
        "amd64": "x86_64,native",
        "x86_64": "x86_64,native",
        "aarch64": "arm64,native",
        "arm64": "arm64,native",
        "riscv64": "riscv64,native",
    }.get(machine, f"{machine},native")
    return Target.parse(f"CPU:CLANG:{arch}")


def _tensor(data, *, requires_grad=None, **kwargs) -> Tensor:
    if requires_grad is None:
        return Tensor(data, **kwargs)
    try:
        return Tensor(data, requires_grad=requires_grad, **kwargs)
    except TypeError as exc:
        if "requires_grad" not in str(exc):
            raise
    t = Tensor(data, **kwargs)
    if hasattr(t, "is_param"):
        t.is_param = bool(requires_grad)
    else:
        t.requires_grad = bool(requires_grad)
    return t


def _flatten(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).reshape(-1)


def _to_output_tuple(outputs) -> tuple[Tensor, ...]:
    if isinstance(outputs, Tensor):
        return (outputs,)
    if isinstance(outputs, tuple):
        return outputs
    if isinstance(outputs, list):
        return tuple(outputs)
    raise TypeError(f"expected Tensor|tuple[Tensor,...]|list[Tensor], got {type(outputs)!r}")


def _concat_outputs(outputs: tuple[Tensor, ...]) -> np.ndarray:
    outputs[0].realize(*outputs[1:])
    parts = [_flatten(t.numpy()) for t in outputs]
    if len(parts) == 1:
        return parts[0]
    return np.concatenate(parts, axis=0)


def _program_linear_ops(program) -> list[str]:
    """tinygrad latest returns a PROGRAM UOp; its linearized kernel is carried
    as an Ops.LINEAR source rather than the older program.uops attribute."""
    for src in program.src:
        if src.op is Ops.LINEAR:
            return [u.op.name for u in src.src]
    raise RuntimeError("tinygrad PROGRAM has no LINEAR source")


# ---- case builders ----


def build_vecadd() -> tuple[Tensor, ...]:
    a = np.arange(1, 17, dtype=np.float32)
    b = a * np.float32(0.5)
    return (Tensor(a) + Tensor(b),)


def build_chain() -> tuple[Tensor, ...]:
    a = np.arange(1, 9, dtype=np.float32)
    b = np.full((8,), np.float32(2.0), dtype=np.float32)
    c = np.full((8,), np.float32(0.5), dtype=np.float32)
    return ((Tensor(a) + Tensor(b)) * Tensor(c),)


def build_broadcast_scalar() -> tuple[Tensor, ...]:
    a = np.arange(1, 9, dtype=np.float32)
    return (Tensor(a) + 2.0,)


def build_reduce_sum_axis1() -> tuple[Tensor, ...]:
    a = np.arange(1, 13, dtype=np.float32).reshape(4, 3)
    return (Tensor(a).sum(axis=1),)


def build_reduce_scalar_chain() -> tuple[Tensor, ...]:
    a = np.arange(1, 9, dtype=np.float32)
    b = np.arange(10, 18, dtype=np.float32)
    return (Tensor(a).sum() + Tensor(b),)


def build_reduce_vector_chain() -> tuple[Tensor, ...]:
    a = np.arange(1, 13, dtype=np.float32).reshape(4, 3)
    b = np.arange(100, 112, dtype=np.float32).reshape(4, 3)
    s = Tensor(a).sum(axis=1).reshape(4, 1).expand(4, 3)
    return (s + Tensor(b),)


def build_shared_scalar_reduce_branches() -> tuple[Tensor, ...]:
    a = np.arange(1, 9, dtype=np.float32)
    c0 = np.arange(10, 18, dtype=np.float32)
    e0 = np.arange(20, 28, dtype=np.float32)
    s = Tensor(a).sum()
    return (s + Tensor(c0), s * Tensor(e0))


def build_permute_2d() -> tuple[Tensor, ...]:
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    return (Tensor(a).permute(1, 0),)


def build_shrink_2d() -> tuple[Tensor, ...]:
    a = np.arange(12, dtype=np.float32).reshape(4, 3)
    return (Tensor(a)[1:3, 0:3],)


def build_pad_2d() -> tuple[Tensor, ...]:
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    return (Tensor(a).pad(((1, 1), (1, 1))),)


def build_chain_pad_flip() -> tuple[Tensor, ...]:
    a = np.array([1, 2, 3], dtype=np.float32)
    return (Tensor(a).pad(((1, 1),))[::-1],)


def build_grad_mul_sum() -> tuple[Tensor, ...]:
    x = _tensor(np.arange(-3, 5, dtype=np.float32), requires_grad=True)
    (x * x).sum().backward()
    return (x.grad,)


def build_grad_exp2_sum() -> tuple[Tensor, ...]:
    x = _tensor(np.array([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0], dtype=np.float32), requires_grad=True)
    x.exp2().sum().backward()
    return (x.grad,)


def build_grad_fdiv_sum_x() -> tuple[Tensor, ...]:
    x = _tensor(np.array([1.0, 2.0, 3.0, -1.0, -2.0, 4.0], dtype=np.float32), requires_grad=True)
    y = _tensor(np.array([2.0, 4.0, -2.0, 5.0, -3.0, 8.0], dtype=np.float32), requires_grad=False)
    (x / y).sum().backward()
    return (x.grad,)


def build_grad_fdiv_sum_y() -> tuple[Tensor, ...]:
    x = _tensor(np.array([1.0, 2.0, 3.0, -1.0, -2.0, 4.0], dtype=np.float32), requires_grad=False)
    y = _tensor(np.array([2.0, 4.0, -2.0, 5.0, -3.0, 8.0], dtype=np.float32), requires_grad=True)
    (x / y).sum().backward()
    return (y.grad,)


def build_grad_chain_movement() -> tuple[Tensor, ...]:
    x = _tensor(np.arange(1, 7, dtype=np.float32), requires_grad=True)
    x.reshape(2, 3).permute(1, 0).sum().backward()
    return (x.grad,)


def build_neg_1d() -> tuple[Tensor, ...]:
    x = np.arange(-3, 5, dtype=np.float32)
    return (-Tensor(x),)


def build_exp2_1d() -> tuple[Tensor, ...]:
    x = np.array([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0], dtype=np.float32)
    return (Tensor(x).exp2(),)


def build_sqrt_1d() -> tuple[Tensor, ...]:
    x = np.array([1.0, 4.0, 9.0, 16.0, 25.0, 0.25], dtype=np.float32)
    return (Tensor(x).sqrt(),)


def build_mul_1d() -> tuple[Tensor, ...]:
    x = np.arange(1, 9, dtype=np.float32)
    y = x * np.float32(0.5)
    return (Tensor(x) * Tensor(y),)


def build_where_1d() -> tuple[Tensor, ...]:
    x = Tensor(np.arange(-3, 5, dtype=np.float32))
    return ((x > 0).where(x, 0),)


def build_reduce_sum_all() -> tuple[Tensor, ...]:
    a = np.arange(1, 13, dtype=np.float32)
    return (Tensor(a).sum(),)


def build_reduce_max_1d() -> tuple[Tensor, ...]:
    a = np.array([-7, -5, -3, -1, 1, 3, 5, 7], dtype=np.float32)
    return (Tensor(a).max(),)


def build_reshape_reduce() -> tuple[Tensor, ...]:
    a = np.arange(1, 13, dtype=np.float32)
    return (Tensor(a).reshape(4, 3).sum(axis=1),)


def build_expand_alu_reduce() -> tuple[Tensor, ...]:
    a = np.array([1, 2, 3, 4], dtype=np.float32)
    b = (np.arange(1, 13, dtype=np.float32) * np.float32(0.1)).reshape(4, 3)
    ae = Tensor(a).reshape(4, 1).expand(4, 3)
    return ((ae + Tensor(b)).sum(axis=1),)


def build_multi_movement() -> tuple[Tensor, ...]:
    a = np.arange(1, 13, dtype=np.float32).reshape(3, 4)
    return (Tensor(a).permute(1, 0)[0:2, 0:3][::-1, :],)


def build_grad_log2_sum() -> tuple[Tensor, ...]:
    x = _tensor(np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0], dtype=np.float32), requires_grad=True)
    x.log2().sum().backward()
    return (x.grad,)


def build_grad_sqrt_sum() -> tuple[Tensor, ...]:
    x = _tensor(np.array([1.0, 4.0, 9.0, 16.0, 25.0, 0.25], dtype=np.float32), requires_grad=True)
    x.sqrt().sum().backward()
    return (x.grad,)


def build_grad_where_sum() -> tuple[Tensor, ...]:
    x = _tensor(np.arange(-3, 5, dtype=np.float32), requires_grad=True)
    (x > 0).where(x, 0).sum().backward()
    return (x.grad,)


def build_grad_multi_use() -> tuple[Tensor, ...]:
    x = _tensor(np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32), requires_grad=True)
    ((x * x) + (x * 2.0)).sum().backward()
    return (x.grad,)


def build_matmul_small() -> tuple[Tensor, ...]:
    a = Tensor(np.array([1, 2, 3, 4, 5, 6], dtype=np.float32))
    b = Tensor(np.array([1, 4, 2, 5, 3, 6], dtype=np.float32))

    ae = a.reshape(2, 1, 3).expand(2, 2, 3)
    be = b.reshape(3, 2).permute(1, 0).reshape(1, 2, 3).expand(2, 2, 3)
    return ((ae * be).sum(axis=2),)


def build_matmul_broadcast() -> tuple[Tensor, ...]:
    a = Tensor(np.array([
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
    ], dtype=np.float32))
    b = Tensor(np.array([
        [[1, 10], [100, 1000]],
    ], dtype=np.float32))
    return (a.matmul(b),)


def build_cross_entropy_nonlast_axis() -> tuple[Tensor, ...]:
    logits = Tensor(np.zeros((2, 3, 2), dtype=np.float32))
    target = Tensor(np.array([[0, 2], [1, 0]], dtype=np.int32))
    return (logits.cross_entropy(target),)


# New helpers (movement / pad / cumalu / full / arange)
def build_full_1d() -> tuple[Tensor, ...]:
    return (Tensor.full((5,), 7.5),)


def build_full_2d() -> tuple[Tensor, ...]:
    return (Tensor.full((3, 4), -2.0),)


def build_arange_simple() -> tuple[Tensor, ...]:
    return (Tensor.arange(0, 5, 1, dtype=dtypes.float32),)


def build_arange_start_step() -> tuple[Tensor, ...]:
    return (Tensor.arange(2, 8, 3, dtype=dtypes.float32),)


def build_linspace_5() -> tuple[Tensor, ...]:
    return (Tensor.linspace(0, 10, 5),)


def build_eye_3() -> tuple[Tensor, ...]:
    return (Tensor.eye(3),)


def build_tril_3x4_diag0() -> tuple[Tensor, ...]:
    m = Tensor(np.arange(1, 13, dtype=np.float32).reshape(3, 4).tolist())
    return (m.tril(0),)


def build_triu_3x4_diag0() -> tuple[Tensor, ...]:
    m = Tensor(np.arange(1, 13, dtype=np.float32).reshape(3, 4).tolist())
    return (m.triu(0),)


def build_repeat_1d() -> tuple[Tensor, ...]:
    return (Tensor([1.0, 2.0, 3.0]).repeat([4]),)


def build_pool_1d_k3() -> tuple[Tensor, ...]:
    return (Tensor([0.0, 1.0, 2.0, 3.0, 4.0])._pool((3,)),)


def build_cat_1d() -> tuple[Tensor, ...]:
    return (Tensor([1.0, 2.0]).cat(Tensor([3.0, 4.0, 5.0]), dim=0),)


def build_pad_value_1d() -> tuple[Tensor, ...]:
    return (Tensor([1.0, 2.0, 3.0]).pad(((2, 1),), value=9.0),)


def build_pad_circular_1d() -> tuple[Tensor, ...]:
    return (Tensor([1.0, 2.0, 3.0]).pad(((1, 2),), mode="circular"),)


def build_pad_reflect_1d() -> tuple[Tensor, ...]:
    return (Tensor([1.0, 2.0, 3.0, 4.0]).pad(((2, 1),), mode="reflect"),)


def build_pad_replicate_1d() -> tuple[Tensor, ...]:
    return (Tensor([1.0, 2.0, 3.0, 4.0]).pad(((2, 1),), mode="replicate"),)


def build_cumsum_1d() -> tuple[Tensor, ...]:
    return (Tensor([1.0, 2.0, 3.0, 4.0, 5.0]).cumsum(0),)


def build_cumprod_1d() -> tuple[Tensor, ...]:
    return (Tensor([1.0, 2.0, 3.0, 4.0]).cumprod(0),)


def build_cummax_1d() -> tuple[Tensor, ...]:
    # cummax returns (values, indices); take values only for parity
    return (Tensor([1.0, 3.0, 2.0, 5.0, 4.0]).cummax(0)[0],)


CaseBuilder = Callable[[], tuple[Tensor, ...]]
CASES: dict[str, CaseBuilder] = {
    "vecadd": build_vecadd,
    "chain": build_chain,
    "broadcast_scalar": build_broadcast_scalar,
    "reduce_sum_axis1": build_reduce_sum_axis1,
    "reduce_scalar_chain": build_reduce_scalar_chain,
    "reduce_vector_chain": build_reduce_vector_chain,
    "shared_scalar_reduce_branches": build_shared_scalar_reduce_branches,
    "permute_2d": build_permute_2d,
    "shrink_2d": build_shrink_2d,
    "pad_2d": build_pad_2d,
    "chain_pad_flip": build_chain_pad_flip,
    "grad_mul_sum": build_grad_mul_sum,
    "grad_exp2_sum": build_grad_exp2_sum,
    "grad_fdiv_sum_x": build_grad_fdiv_sum_x,
    "grad_fdiv_sum_y": build_grad_fdiv_sum_y,
    "grad_chain_movement": build_grad_chain_movement,
    "neg_1d": build_neg_1d,
    "exp2_1d": build_exp2_1d,
    "sqrt_1d": build_sqrt_1d,
    "mul_1d": build_mul_1d,
    "where_1d": build_where_1d,
    "reduce_sum_all": build_reduce_sum_all,
    "reduce_max_1d": build_reduce_max_1d,
    "reshape_reduce": build_reshape_reduce,
    "expand_alu_reduce": build_expand_alu_reduce,
    "multi_movement": build_multi_movement,
    "grad_log2_sum": build_grad_log2_sum,
    "grad_sqrt_sum": build_grad_sqrt_sum,
    "grad_where_sum": build_grad_where_sum,
    "grad_multi_use": build_grad_multi_use,
    "matmul_small": build_matmul_small,
    "matmul_broadcast": build_matmul_broadcast,
    "cross_entropy_nonlast_axis": build_cross_entropy_nonlast_axis,
    # New helpers (movement/pad/cumalu/full/arange)
    "full_1d":           build_full_1d,
    "full_2d":           build_full_2d,
    "arange_simple":     build_arange_simple,
    "arange_start_step": build_arange_start_step,
    "linspace_5":        build_linspace_5,
    "eye_3":             build_eye_3,
    "tril_3x4_diag0":    build_tril_3x4_diag0,
    "triu_3x4_diag0":    build_triu_3x4_diag0,
    "repeat_1d":         build_repeat_1d,
    "pool_1d_k3":        build_pool_1d_k3,
    "cat_1d":            build_cat_1d,
    "pad_value_1d":      build_pad_value_1d,
    "pad_circular_1d":   build_pad_circular_1d,
    "pad_reflect_1d":    build_pad_reflect_1d,
    "pad_replicate_1d":  build_pad_replicate_1d,
    "cumsum_1d":         build_cumsum_1d,
    "cumprod_1d":        build_cumprod_1d,
    "cummax_1d":         build_cummax_1d,
}

STRUCTURE_KEYS: tuple[str, ...] = (
    "RANGE",
    "END",
    "REDUCE",
    "STORE",
    "LOAD",
    "INDEX",
    "AFTER",
    "DEFINE_LOCAL",
    "DEFINE_REG",
)


def kernel_structure(ops: list[str]) -> dict[str, int]:
    counts = {k: 0 for k in STRUCTURE_KEYS}
    range_depth = 0
    max_range_depth = 0
    min_range_depth = 0
    for op in ops:
        if op in counts:
            counts[op] += 1
        if op == "RANGE":
            range_depth += 1
            max_range_depth = max(max_range_depth, range_depth)
        elif op == "END":
            range_depth -= 1
            min_range_depth = min(min_range_depth, range_depth)
    counts["MAX_RANGE_DEPTH"] = max_range_depth
    counts["FINAL_RANGE_DEPTH"] = range_depth
    counts["MIN_RANGE_DEPTH"] = min_range_depth
    return counts


def run_polygrad_case(runner: pathlib.Path, case: str, mode: str,
                      cuda: bool = False, hip: bool = False) -> dict:
    env = os.environ.copy()
    # ASan-instrumented parity_runner is spawned from Python. On this platform,
    # detect_leaks=0 alone can produce recursive DEADLYSIGNAL reports in the
    # child process; halt_on_error=0 preserves the intended non-leak behavior
    # and lets subprocess capture the JSON report.
    asan_opts = env.get("ASAN_OPTIONS", "")
    if "detect_leaks=" not in asan_opts:
        asan_opts = (asan_opts + ":detect_leaks=0") if asan_opts else "detect_leaks=0"
    if "halt_on_error=" not in asan_opts:
        asan_opts = asan_opts + ":halt_on_error=0"
    env["ASAN_OPTIONS"] = asan_opts
    if mode == "full":
        env["POLY_SPLIT_SINK_STORES"] = "1"
        env["POLY_PARITY_MOVEMENT_AS_COPY"] = "1"
    if cuda:
        env["POLY_PARITY_CUDA"] = "1"
    if hip:
        env["POLY_PARITY_HIP"] = "1"
    raw = subprocess.check_output([str(runner), case], text=True, env=env)
    payload = json.loads(raw)

    if "n_kernels" not in payload or "kernels" not in payload:
        raise RuntimeError(f"runner output missing kernel metadata for case {case}")
    if "data" not in payload:
        raise RuntimeError(f"runner output missing data for case {case}")

    kernels = payload["kernels"]
    if not isinstance(kernels, list):
        raise RuntimeError(f"runner kernels field is not a list for case {case}")

    normalized_kernels = []
    for idx, kernel in enumerate(kernels):
        ops = kernel.get("ops") if isinstance(kernel, dict) else None
        if not isinstance(ops, list):
            raise RuntimeError(f"runner kernel {idx} missing ops list for case {case}")
        op_names = [str(op) for op in ops]
        normalized_kernels.append({"ops": op_names, "structure": kernel_structure(op_names)})

    n_kernels = int(payload["n_kernels"])
    if n_kernels != len(normalized_kernels):
        raise RuntimeError(
            f"runner n_kernels mismatch for case {case}: n_kernels={n_kernels}, len(kernels)={len(normalized_kernels)}"
        )

    return {
        "n_kernels": n_kernels,
        "kernels": normalized_kernels,
        "data": _flatten(payload["data"]),
    }


def evaluate_tinygrad_case(case_builder: CaseBuilder) -> np.ndarray:
    # Older tinygrad CPU could render invalid C for vector bool masks under
    # DEVECTORIZE=0. Current tinygrad removed that context key; keep the
    # override only when the reference checkout still exposes it.
    with _tiny_context(DEVECTORIZE=1):
        outputs = _to_output_tuple(case_builder())
        return _concat_outputs(outputs)


def extract_tinygrad_kernels(case_builder: CaseBuilder, optimize: bool = True, renderer=None) -> dict:
    # Some older tinygrad backends produced non-runnable IR when
    # DEVECTORIZE=-1. Current tinygrad removed that context key.
    with _tiny_context(DEVECTORIZE=0):
        outputs = _to_output_tuple(case_builder())
        linear_schedule = outputs[0].schedule_linear(*outputs[1:])

        if renderer is None:
            renderer = ClangRenderer(_cpu_target())
        kernels = []
        for call in linear_schedule.src:
            if call.op is not Ops.CALL or len(call.src) == 0:
                continue
            ast = call.src[0]
            # Current tinygrad schedule_linear includes COPY calls for host
            # transfers. The historical parity report compares only lowered
            # compute SINK kernels, matching the Polygrad runner metadata.
            if ast.op is not Ops.SINK:
                continue
            if optimize:
                program = to_program(ast, renderer)
                ops = _program_linear_ops(program)
            else:
                rw = full_rewrite_to_sink(ast, renderer, optimize=False)
                ops = [u.op.name for u in linearize(rw)]
            kernels.append({"ops": ops, "structure": kernel_structure(ops)})

        return {"n_kernels": len(kernels), "kernels": kernels}


def first_mismatch(a: np.ndarray, b: np.ndarray, atol: float, rtol: float):
    for i, (x, y) in enumerate(zip(a, b)):
        if not np.isclose(x, y, atol=atol, rtol=rtol):
            return i
    return None


def _ops_window(ops: list[str], idx: int, radius: int = 3) -> str:
    lo = max(0, idx - radius)
    hi = min(len(ops), idx + radius + 1)
    return " ".join(ops[lo:hi])


def compare_kernel_reports(polygrad: dict, tiny: dict) -> tuple[bool, str]:
    if polygrad["n_kernels"] != tiny["n_kernels"]:
        return (
            False,
            f"kernel count mismatch: polygrad={polygrad['n_kernels']} tinygrad={tiny['n_kernels']}",
        )

    for k_idx, (tk, sk) in enumerate(zip(polygrad["kernels"], tiny["kernels"])):
        t_ops = tk["ops"]
        s_ops = sk["ops"]
        t_structure = tk.get("structure", kernel_structure(t_ops))
        s_structure = sk.get("structure", kernel_structure(s_ops))

        if t_structure != s_structure:
            return (
                False,
                f"kernel {k_idx} structure mismatch: polygrad={t_structure} tinygrad={s_structure}",
            )

        if len(t_ops) != len(s_ops):
            return (
                False,
                f"kernel {k_idx} op count mismatch: polygrad={len(t_ops)} tinygrad={len(s_ops)}",
            )

        for i, (to, so) in enumerate(zip(t_ops, s_ops)):
            if to != so:
                return (
                    False,
                    (
                        f"kernel {k_idx} first op mismatch at index {i}: polygrad={to} tinygrad={so}\n"
                        f"    polygrad window: {_ops_window(t_ops, i)}\n"
                        f"    tiny window: {_ops_window(s_ops, i)}"
                    ),
                )

    return True, ""


def kernel_summary(report: dict) -> str:
    counts = [len(k["ops"]) for k in report["kernels"]]
    depths = [k.get("structure", {}).get("MAX_RANGE_DEPTH", 0) for k in report["kernels"]]
    if not counts:
        return "0 kernels"
    if len(counts) == 1:
        return f"1 kernel, {counts[0]} ops, max_range_depth={depths[0]}"
    return (
        f"{len(counts)} kernels, ops={'+'.join(str(c) for c in counts)}, "
        f"max_range_depth={'+'.join(str(d) for d in depths)}"
    )


def dump_ops(case: str, polygrad: dict, tiny: dict) -> None:
    print(f"[DUMP] {case}")
    print("  polygrad:")
    for k_idx, kernel in enumerate(polygrad["kernels"]):
        print(f"    k{k_idx} ({len(kernel['ops'])} ops): {' '.join(kernel['ops'])}")
        print(f"      structure: {kernel.get('structure', kernel_structure(kernel['ops']))}")
    print("  tinygrad:")
    for k_idx, kernel in enumerate(tiny["kernels"]):
        print(f"    k{k_idx} ({len(kernel['ops'])} ops): {' '.join(kernel['ops'])}")
        print(f"      structure: {kernel.get('structure', kernel_structure(kernel['ops']))}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Differential parity tests: polygrad vs tinygrad")
    parser.add_argument("--runner", type=pathlib.Path, required=True, help="path to polygrad parity runner executable")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--mode", choices=["values", "full"], default="values")
    parser.add_argument("--dump", action="store_true", help="dump per-kernel op sequences")
    parser.add_argument("--no-opt", action="store_true", help="disable tinygrad optimization (no UPCAST/UNROLL)")
    parser.add_argument("--cuda", action="store_true", help="use CUDA backend (CUDARenderer) instead of CPU")
    parser.add_argument("--hip", action="store_true", help="use HIP backend (AMDHIPRenderer) instead of CPU")
    parser.add_argument("cases", nargs="*", help="subset of cases to run")
    args = parser.parse_args()

    if args.dump and args.mode != "full":
        print("--dump requires --mode full", file=sys.stderr)
        return 2

    if not args.runner.exists():
        print(f"runner not found: {args.runner}", file=sys.stderr)
        return 2

    selected = args.cases if args.cases else list(CASES.keys())
    unknown = [c for c in selected if c not in CASES]
    if unknown:
        print(f"unknown cases: {', '.join(unknown)}", file=sys.stderr)
        return 2

    if args.cuda:
        import subprocess as _sp
        arch = _sp.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True,
        ).strip().split("\n")[0]
        renderer = CUDARenderer(Target.parse("CUDA:NV:sm_" + arch.replace(".", "")))
    elif args.hip:
        hip_arch = os.environ.get("HIP_ARCH", "gfx90a")
        renderer = HIPRenderer(Target.parse("HIP:AMD:" + hip_arch))
    else:
        renderer = ClangRenderer(_cpu_target())

    failures: list[tuple[str, str]] = []
    ir_diverged: list[tuple[str, str]] = []
    passed = 0

    for case in selected:
        builder = CASES[case]

        try:
            polygrad_report = run_polygrad_case(args.runner, case, args.mode,
                                                  cuda=args.cuda, hip=args.hip)
        except Exception as exc:  # noqa: BLE001
            failures.append((case, f"runner error: {exc}"))
            print(f"[FAIL] {case} runner error")
            continue

        try:
            tiny_out = evaluate_tinygrad_case(builder)
        except Exception as exc:  # noqa: BLE001
            failures.append((case, f"tinygrad eval error: {exc}"))
            print(f"[FAIL] {case} tinygrad eval error")
            continue

        polygrad_out = polygrad_report["data"]

        if polygrad_out.shape != tiny_out.shape:
            failures.append((case, f"shape mismatch: polygrad={polygrad_out.shape}, tinygrad={tiny_out.shape}"))
            print(f"[FAIL] {case} shape mismatch")
            continue

        if not np.allclose(polygrad_out, tiny_out, atol=args.atol, rtol=args.rtol):
            idx = first_mismatch(polygrad_out, tiny_out, args.atol, args.rtol)
            if idx is None:
                failures.append((case, "values differ but could not localize mismatch"))
            else:
                failures.append(
                    (
                        case,
                        f"first mismatch at index {idx}: polygrad={polygrad_out[idx]:.8f} tinygrad={tiny_out[idx]:.8f}",
                    )
                )
            print(f"[FAIL] {case}")
            continue

        if args.mode == "values":
            passed += 1
            print(f"[PASS] {case}")
            continue

        try:
            tiny_report = extract_tinygrad_kernels(builder, optimize=not args.no_opt, renderer=renderer)
        except Exception as exc:  # noqa: BLE001
            failures.append((case, f"tinygrad kernel extraction error: {exc}"))
            print(f"[FAIL] {case} kernel extraction error")
            continue

        if args.dump:
            dump_ops(case, polygrad_report, tiny_report)

        ir_match, ir_msg = compare_kernel_reports(polygrad_report, tiny_report)
        if ir_match:
            passed += 1
            print(f"[PASS] {case:<22} ({kernel_summary(polygrad_report)})")
        else:
            ir_diverged.append((case, ir_msg))
            print(
                f"[IR_DIVERGE] {case:<16} "
                f"(polygrad: {kernel_summary(polygrad_report)}; tinygrad: {kernel_summary(tiny_report)})"
            )

    total = len(selected)
    if args.mode == "values":
        print(f"\nSummary: {passed} passed, {len(failures)} failed, {total} total")
    else:
        print(
            f"\nSummary: {passed} passed, {len(ir_diverged)} ir_diverge, "
            f"{len(failures)} failed, {total} total"
        )

    if ir_diverged:
        print("IR divergences:")
        for case, msg in ir_diverged:
            print(f"  - {case}: {msg}")

    if failures:
        print("Failures:")
        for case, msg in failures:
            print(f"  - {case}: {msg}")
        return 1

    if args.mode == "full" and ir_diverged:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
