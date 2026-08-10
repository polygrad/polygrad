#!/usr/bin/env python3
"""Matched bounded CUDA semantic and timing gates for pinned HLB CIFAR.

This driver launches ``hlb_cifar_worker.py`` in separate tinygrad and Polygrad
processes. It owns cache isolation, source/runtime provenance, shared model
bytes, semantic comparison, no-live-Tensor JIT lifecycle timing, and durable
result artifacts. It intentionally does not run the full 1,000-step workload.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "bench" / "hlb_cifar_worker.py"
TINYGRAD = ROOT / "references" / "tinygrad_latest"
DEFAULT_PYTHON = ROOT / "references" / ".venv-tinygrad-py311" / "bin" / "python"
DEFAULT_LIB = ROOT / "build" / "libpolygrad.so"
RESULT_PREFIX = "HLB_RESULT "
EXPECTED_TINYGRAD_COMMIT = "ba1d3baae81c96b3ed72900cde87bb307933c61f"
EXPECTED_HLB_SHA256 = "942edfefc75647c0d0f440b2114964a7c4c69bf4f88ab45803f387d86231baae"
MAX_RTOL = 1e-5
MAX_ATOL = 1e-5
EXPECTED_FORWARD_KEYS = {"whitening", "input", "logits"}
EXPECTED_DATASET_KEYS = {
    "train_images", "train_labels", "test_images", "test_labels",
}
EXPECTED_STATE_COUNT = 39
EXPECTED_SEMANTIC_JITS = [
    ("train_cifar.<locals>.augmentations", 1, False, 0),
    ("train_cifar.<locals>.modelEMA.update", 0, False, 0),
    ("train_cifar.<locals>.train_step", 1, False, 0),
    ("train_cifar.<locals>.eval_step", 0, False, 0),
    ("train_cifar.<locals>.eval_step", 0, False, 0),
]
COMMON_CONFIG = {
    "DEV": "CUDA",
    "POLY_DEVICE": "cuda",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "BS": "32",
    "EVAL_BS": "32",
    "GPUS": "1",
    "SEED": "201",
    "BEAM": "0",
    "WINO": "0",
    "LATEBEAM": "0",
    "LATEWINO": "0",
    "DISABLE_BACKWARD": "0",
    "RANDOM_CROP": "0",
    "RANDOM_FLIP": "0",
    "CUTMIX": "0",
    "EMA": "0",
    "SYNCBN": "0",
    "CACHELEVEL": "2",
    "SCACHE": "1",
    "POLY_SCACHE": "1",
    "POLY_CACHE": "1",
    "DEBUG": "0",
    "VIZ": "0",
    "PROFILE": "0",
    "SYNTHETIC_TRAIN_SAMPLES": "32",
    "SYNTHETIC_TEST_SAMPLES": "32",
    "HLB_TRACK_JIT": "1",
    "HLB_RESET_JITS_AFTER_STEP": "-1",
    "HLB_CAPTURE_EXACT_STEP_VALUES": "0",
}
SEMANTIC_CONFIG = {
    **COMMON_CONFIG,
    "STEPS": "1",
    "HLB_CAPTURE_FORWARD": "1",
    "HLB_CAPTURE_PERMUTATIONS": "1",
}
TIMING_CONFIG = {
    **COMMON_CONFIG,
    "STEPS": "8",
    "EVAL_STEPS": "9",
    "HLB_CAPTURE_FORWARD": "0",
    "HLB_CAPTURE_PERMUTATIONS": "0",
}
REPLAY_CONFIG = {
    **COMMON_CONFIG,
    "STEPS": "4",
    "EVAL_STEPS": "5",
    "HLB_CAPTURE_FORWARD": "0",
    "HLB_CAPTURE_PERMUTATIONS": "0",
    "HLB_CAPTURE_EXACT_STEP_VALUES": "1",
}
REPLAY_EAGER_CONFIG = {
    **REPLAY_CONFIG,
    "HLB_RESET_JITS_AFTER_STEP": "1",
}
TIMING_PHASES = ["eager", "capture", "first_replay"] + ["steady_replay"] * 5


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_text(command: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def run_bytes(command: list[str], cwd: Path) -> bytes:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"command failed: {command!r}") from exc
    return result.stdout


def git_record(root: Path) -> dict[str, Any]:
    commit = run_text(["git", "rev-parse", "HEAD"], root)
    status = run_text(["git", "status", "--porcelain=v1", "--untracked-files=all"], root)
    if commit is None or status is None:
        raise RuntimeError(f"failed to inspect git provenance at {root}")
    tracked_diff = run_bytes(["git", "diff", "--binary", "HEAD", "--"], root)
    untracked_text = run_text(
        ["git", "ls-files", "--others", "--exclude-standard"], root
    )
    if untracked_text is None:
        raise RuntimeError(f"failed to inspect untracked provenance at {root}")
    untracked = []
    for relative in filter(None, untracked_text.splitlines()):
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"untracked provenance path is not a file: {path}")
        untracked.append({
            "path": relative,
            "size": path.stat().st_size,
            "sha256": sha256(path),
        })
    content_record = {
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "untracked": untracked,
    }
    return {
        "root": str(root),
        "commit": commit,
        "dirty": bool(status),
        "status_sha256": hashlib.sha256((status or "").encode()).hexdigest(),
        "content_sha256": hashlib.sha256(
            json.dumps(content_record, sort_keys=True).encode()
        ).hexdigest(),
        **content_record,
    }


def selected_cuda_device(
    logical_device: str,
    cuda_visible_devices: str | None,
    cuda_device_order: str,
    inventory: str | None,
) -> dict[str, Any]:
    """Resolve one logical CUDA ordinal to the physical GPU recorded by nvidia-smi."""
    device = logical_device.upper()
    if not device.startswith("CUDA"):
        return {}
    logical_ordinal = int(device.split(":", 1)[1]) if ":" in device else 0
    if not inventory:
        raise RuntimeError("nvidia-smi did not report CUDA device identity")
    if cuda_device_order != "PCI_BUS_ID":
        raise RuntimeError(
            f"CUDA_DEVICE_ORDER must be PCI_BUS_ID, got {cuda_device_order!r}"
        )
    rows = []
    for line in inventory.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            raise RuntimeError(f"unexpected nvidia-smi device row: {line!r}")
        rows.append(dict(zip(
            ("index", "name", "uuid", "pci_bus_id", "driver_version", "compute_cap"),
            fields,
        )))
    ordered_rows = sorted(rows, key=lambda row: row["pci_bus_id"])
    visible = (
        [token.strip() for token in cuda_visible_devices.split(",")]
        if cuda_visible_devices not in (None, "") else
        [str(index) for index in range(len(ordered_rows))]
    )
    if logical_ordinal >= len(visible):
        raise RuntimeError(
            f"logical CUDA ordinal {logical_ordinal} is outside visible set {visible!r}"
        )
    selector = visible[logical_ordinal]
    if selector.isdigit():
        physical_ordinal = int(selector)
        matches = (
            [ordered_rows[physical_ordinal]]
            if physical_ordinal < len(ordered_rows) else []
        )
    else:
        matches = [
            row for row in rows
            if row["uuid"] == selector or row["uuid"].startswith(selector)
        ]
    if len(matches) != 1:
        raise RuntimeError(
            f"CUDA selector {selector!r} matched {len(matches)} physical GPUs"
        )
    return {
        "logical_device": device,
        "cuda_visible_devices": cuda_visible_devices,
        "cuda_device_order": cuda_device_order,
        **matches[0],
    }


def host_manifest(
    python: Path, library: Path, mode: str, config: dict[str, str]
) -> dict[str, Any]:
    ambient_device_order = os.environ.get("CUDA_DEVICE_ORDER")
    if ambient_device_order not in (None, config["CUDA_DEVICE_ORDER"]):
        raise RuntimeError(
            "ambient CUDA_DEVICE_ORDER conflicts with locked benchmark order: "
            f"{ambient_device_order!r} != {config['CUDA_DEVICE_ORDER']!r}"
        )
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    nvidia_smi = run_text(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,pci.bus_id,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        ROOT,
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "polygrad": git_record(ROOT),
        "tinygrad": git_record(TINYGRAD),
        "driver_sha256": sha256(Path(__file__)),
        "worker_sha256": sha256(WORKER),
        "makefile_sha256": sha256(ROOT / "Makefile"),
        "hlb_sha256": sha256(TINYGRAD / "examples" / "hlb_cifar10.py"),
        "polygrad_library": {"path": str(library), "sha256": sha256(library)},
        "python": {
            "path": str(python),
            "version": run_text([str(python), "--version"], ROOT),
        },
        "host": {
            "node": platform.node(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cuda_visible_devices": cuda_visible_devices,
            "cuda_device_order": config["CUDA_DEVICE_ORDER"],
            "nvidia_smi": nvidia_smi,
            "selected_cuda_device": selected_cuda_device(
                config["DEV"], cuda_visible_devices,
                config["CUDA_DEVICE_ORDER"], nvidia_smi
            ),
        },
        "mode": mode,
        "locked_config": dict(config),
    }
    if manifest["tinygrad"]["commit"] != EXPECTED_TINYGRAD_COMMIT:
        raise RuntimeError(
            "pinned tinygrad revision mismatch: "
            f"{manifest['tinygrad']['commit']} != {EXPECTED_TINYGRAD_COMMIT}"
        )
    if manifest["tinygrad"]["dirty"]:
        raise RuntimeError("pinned tinygrad checkout is dirty")
    if manifest["hlb_sha256"] != EXPECTED_HLB_SHA256:
        raise RuntimeError(
            "pinned HLB source digest mismatch: "
            f"{manifest['hlb_sha256']} != {EXPECTED_HLB_SHA256}"
        )
    return manifest


def base_environment(config: dict[str, str]) -> dict[str, str]:
    keep = (
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "CUDA_VISIBLE_DEVICES",
        "CUDA_PATH",
        "CUDA_HOME",
    )
    env = {name: os.environ[name] for name in keep if name in os.environ}
    env.update({
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    })
    env.update(config)
    return env


def engine_environment(
    engine: str,
    root: Path,
    library: Path,
    initial_state: Path | None,
    config: dict[str, str],
    capture_forward: bool,
    capture_final_state: bool,
) -> dict[str, str]:
    cache = root / "cache"
    tmp = root / "tmp"
    cache.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    env = base_environment(config)
    env.update({
        "TMPDIR": str(tmp),
        "POLY_TMPDIR": str(tmp),
        "XDG_CACHE_HOME": str(cache),
        "CUDA_CACHE_PATH": str(cache / "cuda"),
        "HLB_INITIAL_STATE_OUT": str(root / "initial_state.npz"),
    })
    if capture_forward:
        env["HLB_FORWARD_OUT"] = str(root / "forward.npz")
    if capture_final_state:
        env["HLB_FINAL_STATE_OUT"] = str(root / "final_state.npz")
    if engine == "tinygrad":
        env["PYTHONPATH"] = str(TINYGRAD)
    else:
        env.update({
            "PYTHONPATH": os.pathsep.join((str(ROOT / "py"), str(TINYGRAD))),
            "POLYGRAD_LIB": str(library),
        })
    if initial_state is not None:
        if not initial_state.is_file():
            raise RuntimeError(f"missing shared initial state: {initial_state}")
        env["HLB_MODEL_STATE"] = str(initial_state)
    return env


def prepare_initial_state(
    python: Path,
    library: Path,
    run_root: Path,
    initial_state: Path,
    timeout: int,
    config: dict[str, str],
) -> None:
    prepare_root = run_root / "prepare"
    prepare_root.mkdir(parents=True, exist_ok=True)
    env = engine_environment(
        "tinygrad", prepare_root, library, None, config,
        capture_forward=False, capture_final_state=False,
    )
    env.update({
        "HLB_INITIAL_STATE_OUT": str(initial_state),
        "HLB_CAPTURE_FORWARD": "0",
        "HLB_CAPTURE_PERMUTATIONS": "0",
        "HLB_TRACK_JIT": "0",
    })
    env.pop("HLB_FINAL_STATE_OUT", None)
    env.pop("HLB_FORWARD_OUT", None)
    command = [str(python), str(WORKER), "--engine", "tinygrad", "--prepare-state"]
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout,
    )
    (prepare_root / "stdout.log").write_text(result.stdout, encoding="utf-8")
    (prepare_root / "stderr.log").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0 or not initial_state.is_file():
        raise RuntimeError(
            f"initial-state preparation failed with {result.returncode}; "
            f"see {prepare_root / 'stderr.log'}"
        )


def run_worker(
    engine: str,
    python: Path,
    library: Path,
    run_root: Path,
    shared_initial_state: Path,
    timeout: int,
    config: dict[str, str],
    capture_forward: bool,
    capture_final_state: bool,
) -> tuple[dict[str, Any], dict[str, str]]:
    engine_root = run_root / engine
    engine_root.mkdir(parents=True, exist_ok=True)
    env = engine_environment(
        engine, engine_root, library, shared_initial_state, config,
        capture_forward=capture_forward,
        capture_final_state=capture_final_state,
    )
    command = [str(python), str(WORKER), "--engine", engine]
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout,
    )
    (engine_root / "stdout.log").write_text(result.stdout, encoding="utf-8")
    (engine_root / "stderr.log").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"{engine} HLB worker exited {result.returncode}; "
            f"see {engine_root / 'stderr.log'}"
        )
    records = [
        line[len(RESULT_PREFIX):]
        for line in result.stdout.splitlines()
        if line.startswith(RESULT_PREFIX)
    ]
    if len(records) != 1:
        raise RuntimeError(
            f"{engine} emitted {len(records)} HLB_RESULT records; "
            f"see {engine_root / 'stdout.log'}"
        )
    payload = json.loads(records[0])
    (engine_root / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    recorded_env = {name: env[name] for name in sorted(env)}
    return payload, recorded_env


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def provenance_findings(
    tinygrad: dict[str, Any],
    polygrad: dict[str, Any],
    manifest: dict[str, Any],
    config: dict[str, str],
) -> tuple[list[str], dict[str, Any]]:
    findings: list[str] = []
    hlb_source = (TINYGRAD / "examples" / "hlb_cifar10.py").read_text(
        encoding="utf-8"
    )
    if hlb_source.count("from tinygrad") != 4:
        findings.append("provenance: pinned HLB import count is not four")
    polygrad_source_sha = hashlib.sha256(
        hlb_source.replace("from tinygrad", "from polygrad").encode()
    ).hexdigest()
    expected = {
        "tinygrad": {
            "engine": "tinygrad",
            "device": config["DEV"].upper(),
            "cuda_visible_devices": manifest["host"]["cuda_visible_devices"],
            "selected_cuda_device": manifest["host"]["selected_cuda_device"],
            "hlb_sha256": manifest["hlb_sha256"],
            "executed_sha256": manifest["hlb_sha256"],
            "import_substitutions": 0,
            "worker_path": str(WORKER.resolve()),
            "worker_sha256": manifest["worker_sha256"],
            "engine_module_path": str((TINYGRAD / "tinygrad" / "__init__.py").resolve()),
            "engine_module_sha256": sha256(TINYGRAD / "tinygrad" / "__init__.py"),
            "extra_path": str((TINYGRAD / "extra" / "lr_scheduler.py").resolve()),
            "extra_sha256": sha256(TINYGRAD / "extra" / "lr_scheduler.py"),
            "loaded_library": None,
        },
        "polygrad": {
            "engine": "polygrad",
            "device": config["DEV"].upper(),
            "cuda_visible_devices": manifest["host"]["cuda_visible_devices"],
            "selected_cuda_device": manifest["host"]["selected_cuda_device"],
            "hlb_sha256": manifest["hlb_sha256"],
            "executed_sha256": polygrad_source_sha,
            "import_substitutions": 4,
            "worker_path": str(WORKER.resolve()),
            "worker_sha256": manifest["worker_sha256"],
            "engine_module_path": str((ROOT / "py" / "polygrad" / "__init__.py").resolve()),
            "engine_module_sha256": sha256(ROOT / "py" / "polygrad" / "__init__.py"),
            "extra_path": str((ROOT / "py" / "extra" / "lr_scheduler.py").resolve()),
            "extra_sha256": sha256(ROOT / "py" / "extra" / "lr_scheduler.py"),
            "loaded_library": manifest["polygrad_library"],
        },
    }
    observed: dict[str, Any] = {}
    for name, result in (("tinygrad", tinygrad), ("polygrad", polygrad)):
        source = result.get("source") or {}
        record = {
            "engine": result.get("engine"),
            "device": str(result.get("device", "")).upper(),
            "cuda_visible_devices": result.get("cuda_visible_devices"),
            "selected_cuda_device": result.get("selected_cuda_device"),
            "hlb_sha256": source.get("hlb_sha256"),
            "executed_sha256": source.get("executed_sha256"),
            "import_substitutions": source.get("import_substitutions"),
            "worker_path": (source.get("worker") or {}).get("path"),
            "worker_sha256": (source.get("worker") or {}).get("sha256"),
            "engine_module_path": (source.get("engine_module") or {}).get("path"),
            "engine_module_sha256": (source.get("engine_module") or {}).get("sha256"),
            "extra_path": (source.get("extra_lr_scheduler") or {}).get("path"),
            "extra_sha256": (source.get("extra_lr_scheduler") or {}).get("sha256"),
            "loaded_library": source.get("loaded_library"),
        }
        observed[name] = record
        for field, wanted in expected[name].items():
            if field == "loaded_library" and wanted is not None:
                wanted = {"path": str(Path(wanted["path"]).resolve()), "sha256": wanted["sha256"]}
            if record[field] != wanted:
                findings.append(
                    f"provenance.{name}.{field}: {record[field]!r} != {wanted!r}"
                )
        observed_config = result.get("config") or {}
        if observed_config != config or observed_config != manifest["locked_config"]:
            findings.append(
                f"provenance.{name}.config: {observed_config!r} != {config!r}"
            )
    return findings, {"expected": expected, "observed": observed}


def exact_npz(reference: Path, subject: Path, label: str) -> tuple[list[str], dict[str, Any]]:
    ref, sub = load_npz(reference), load_npz(subject)
    findings: list[str] = []
    detail: dict[str, Any] = {"arrays": {}}
    if ref.keys() != sub.keys():
        findings.append(f"{label}: archive keys differ")
        detail["reference_keys"], detail["subject_keys"] = list(ref), list(sub)
        return findings, detail
    for name in ref:
        a, b = ref[name], sub[name]
        a_numeric = np.issubdtype(a.dtype, np.number) or np.issubdtype(a.dtype, np.bool_)
        b_numeric = np.issubdtype(b.dtype, np.number) or np.issubdtype(b.dtype, np.bool_)
        a_finite = a_numeric and bool(np.isfinite(a).all())
        b_finite = b_numeric and bool(np.isfinite(b).all())
        same = a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
        detail["arrays"][name] = {
            "shape": list(a.shape),
            "dtype": str(a.dtype),
            "reference_sha256": hashlib.sha256(a.tobytes()).hexdigest(),
            "subject_sha256": hashlib.sha256(b.tobytes()).hexdigest(),
            "exact": same,
            "reference_finite": a_finite,
            "subject_finite": b_finite,
        }
        if not a_finite or not b_finite:
            findings.append(
                f"{label}.{name}: numeric/finite={a_numeric}/{a_finite} "
                f"vs {b_numeric}/{b_finite}"
            )
        if not same:
            findings.append(f"{label}.{name}: bytes differ")
    return findings, detail


def shared_initial_state_findings(
    manifest: dict[str, Any], run_root: Path
) -> tuple[list[str], dict[str, Any]]:
    record = manifest["shared_initial_state"]
    shared = Path(record["path"])
    findings: list[str] = []
    observed_digest = sha256(shared)
    if observed_digest != record["sha256"]:
        findings.append(
            f"shared_initial_state.digest: {observed_digest} != {record['sha256']}"
        )
    detail: dict[str, Any] = {
        "recorded_sha256": record["sha256"],
        "observed_sha256": observed_digest,
        "engines": {},
    }
    for engine in ("tinygrad", "polygrad"):
        engine_findings, engine_detail = exact_npz(
            shared, run_root / engine / "initial_state.npz",
            f"shared_initial_state.{engine}",
        )
        findings.extend(engine_findings)
        detail["engines"][engine] = engine_detail
    return findings, detail


def semantic_evidence_findings(engine: str, result: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    dataset = result.get("dataset")
    if not isinstance(dataset, dict) or set(dataset) != EXPECTED_DATASET_KEYS:
        findings.append(
            f"evidence.{engine}.dataset: expected {sorted(EXPECTED_DATASET_KEYS)!r}"
        )
    state_order = result.get("state_order")
    initial_state = result.get("initial_state")
    final_state = result.get("final_state")
    if not isinstance(state_order, list) or len(state_order) != EXPECTED_STATE_COUNT:
        findings.append(
            f"evidence.{engine}.state_order: expected {EXPECTED_STATE_COUNT} names"
        )
    for label, state in (("initial_state", initial_state), ("final_state", final_state)):
        if not isinstance(state, dict) or len(state) != EXPECTED_STATE_COUNT:
            findings.append(
                f"evidence.{engine}.{label}: expected {EXPECTED_STATE_COUNT} arrays"
            )
    forward = result.get("forward")
    if not isinstance(forward, dict) or set(forward) != EXPECTED_FORWARD_KEYS:
        findings.append(
            f"evidence.{engine}.forward: expected {sorted(EXPECTED_FORWARD_KEYS)!r}"
        )
    permutations = result.get("permutations")
    if not isinstance(permutations, list) or len(permutations) != 1:
        findings.append(f"evidence.{engine}.permutations: expected one capture")
    observed_jits = [
        (
            record.get("name"), record.get("cnt"), record.get("captured"),
            record.get("replay_count"),
        )
        for record in result.get("jit", [])
        if isinstance(record, dict)
    ]
    if observed_jits != EXPECTED_SEMANTIC_JITS:
        findings.append(
            f"evidence.{engine}.jit: {observed_jits!r} != "
            f"{EXPECTED_SEMANTIC_JITS!r}"
        )
    return findings


def state_schema_findings(
    engine: str,
    result: dict[str, Any],
    authoritative_path: Path,
    initial_path: Path,
    final_path: Path,
) -> list[str]:
    findings: list[str] = []
    authoritative = load_npz(authoritative_path)
    expected_order = list(authoritative)
    if len(expected_order) != EXPECTED_STATE_COUNT:
        findings.append(
            f"state_schema.authoritative: expected {EXPECTED_STATE_COUNT} names, "
            f"got {len(expected_order)}"
        )
        return findings
    state_order = result.get("state_order")
    if state_order != expected_order:
        findings.append(
            f"state_schema.{engine}.state_order: {state_order!r} != "
            f"authoritative order"
        )
    for label, summary in (
        ("initial_state", result.get("initial_state")),
        ("final_state", result.get("final_state")),
    ):
        if not isinstance(summary, dict) or set(summary) != set(expected_order):
            findings.append(
                f"state_schema.{engine}.{label}: keys do not match state_order"
            )
            continue
        for name, expected in authoritative.items():
            record = summary[name]
            if not isinstance(record, dict) or \
               record.get("shape") != list(expected.shape) or \
               record.get("dtype") != str(expected.dtype):
                findings.append(
                    f"state_schema.{engine}.{label}.{name}: "
                    "reported shape/dtype differ"
                )
    for label, path in (("initial_state", initial_path), ("final_state", final_path)):
        archive = load_npz(path)
        archive_order = list(archive)
        if archive_order != expected_order:
            findings.append(
                f"state_schema.{engine}.{label}_archive: keys/order differ"
            )
            continue
        for name, expected in authoritative.items():
            observed = archive[name]
            if observed.shape != expected.shape or observed.dtype != expected.dtype:
                findings.append(
                    f"state_schema.{engine}.{label}_archive.{name}: "
                    "shape/dtype differ"
                )
    return findings


def numeric_npz(
    reference: Path,
    subject: Path,
    label: str,
    rtol: float,
    atol: float,
) -> tuple[list[str], dict[str, Any]]:
    ref, sub = load_npz(reference), load_npz(subject)
    findings: list[str] = []
    detail: dict[str, Any] = {"rtol": rtol, "atol": atol, "arrays": {}}
    if ref.keys() != sub.keys():
        findings.append(f"{label}: archive keys differ")
        detail["reference_keys"], detail["subject_keys"] = list(ref), list(sub)
        return findings, detail
    for name in ref:
        a, b = ref[name], sub[name]
        if a.shape != b.shape or a.dtype != b.dtype:
            findings.append(f"{label}.{name}: shape/dtype differ")
            detail["arrays"][name] = {
                "reference_shape": list(a.shape),
                "subject_shape": list(b.shape),
                "reference_dtype": str(a.dtype),
                "subject_dtype": str(b.dtype),
            }
            continue
        if a.size:
            a64, b64 = a.astype(np.float64), b.astype(np.float64)
            finite = np.isfinite(a64) & np.isfinite(b64)
            if finite.any():
                delta = np.abs(a64[finite] - b64[finite])
                max_abs = float(delta.max())
                denom = np.maximum(np.abs(a64[finite]), atol)
                max_rel = float((delta / denom).max())
            else:
                max_abs = max_rel = None
            reference_nan = int(np.isnan(a64).sum())
            subject_nan = int(np.isnan(b64).sum())
            reference_inf = int(np.isinf(a64).sum())
            subject_inf = int(np.isinf(b64).sum())
        else:
            max_abs = max_rel = 0.0
            reference_nan = subject_nan = 0
            reference_inf = subject_inf = 0
        all_finite = not any(
            (reference_nan, subject_nan, reference_inf, subject_inf)
        )
        ok = all_finite and bool(np.allclose(a, b, rtol=rtol, atol=atol))
        detail["arrays"][name] = {
            "shape": list(a.shape),
            "dtype": str(a.dtype),
            "max_abs": max_abs,
            "max_rel": max_rel,
            "reference_nan": reference_nan,
            "subject_nan": subject_nan,
            "reference_inf": reference_inf,
            "subject_inf": subject_inf,
            "all_finite": all_finite,
            "within_tolerance": ok,
        }
        if not ok:
            findings.append(
                f"{label}.{name}: max_abs={max_abs} max_rel={max_rel} "
                f"nan={reference_nan}/{subject_nan} "
                f"inf={reference_inf}/{subject_inf}"
            )
    return findings, detail


def diagnostic_numeric_npz(
    reference: Path,
    subject: Path,
    label: str,
    rtol: float,
    atol: float,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Require comparable finite archives, but report numeric drift diagnostically."""
    _, detail = numeric_npz(reference, subject, label, rtol, atol)
    findings: list[str] = []
    diagnostics: list[str] = []
    if "reference_keys" in detail or "subject_keys" in detail:
        findings.append(f"{label}: archive keys differ")
        return findings, diagnostics, detail
    for name, record in detail["arrays"].items():
        if "reference_shape" in record:
            findings.append(f"{label}.{name}: shape/dtype differ")
        elif not record["all_finite"]:
            findings.append(
                f"{label}.{name}: nonfinite values "
                f"nan={record['reference_nan']}/{record['subject_nan']} "
                f"inf={record['reference_inf']}/{record['subject_inf']}"
            )
        elif not record["within_tolerance"]:
            diagnostics.append(
                f"{label}.{name}: accumulated numeric drift "
                f"max_abs={record['max_abs']} max_rel={record['max_rel']}"
            )
    return findings, diagnostics, detail


def manifest_binding_findings(
    semantic: dict[str, Any], timing: dict[str, Any]
) -> tuple[list[str], dict[str, bool]]:
    """Bind the semantic canary to the exact source/runtime used for timing."""
    fields = (
        "polygrad", "tinygrad", "driver_sha256", "worker_sha256",
        "makefile_sha256", "hlb_sha256", "polygrad_library", "python", "host",
        "shared_initial_state",
    )
    equal = {name: semantic.get(name) == timing.get(name) for name in fields}
    return (
        [f"semantic_binding.{name}: semantic/timing records differ"
         for name, same in equal.items() if not same],
        equal,
    )


def revalidate_manifest_binding(
    python: Path,
    library: Path,
    baseline: dict[str, Any],
    mode: str,
    config: dict[str, str],
) -> tuple[list[str], dict[str, Any]]:
    """Re-snapshot all source/runtime evidence around long-running workers."""
    current = host_manifest(python, library, mode, config)
    current["shared_initial_state"] = dict(baseline["shared_initial_state"])
    findings, binding = manifest_binding_findings(baseline, current)
    return findings, {"binding": binding, "manifest": current}


def step_record_findings(
    engine: str, records: list[dict[str, Any]], expected_steps: int
) -> list[str]:
    findings = []
    if len(records) != expected_steps:
        findings.append(
            f"steps.{engine}: parsed {len(records)} records, expected {expected_steps}"
        )
    indices = [record.get("step") for record in records]
    if indices != list(range(expected_steps)):
        findings.append(
            f"steps.{engine}: indices {indices!r} != {list(range(expected_steps))!r}"
        )
    return findings


def compare_results(
    tinygrad: dict[str, Any],
    polygrad: dict[str, Any],
    manifest: dict[str, Any],
    run_root: Path,
    rtol: float,
    atol: float,
    config: dict[str, str],
) -> dict[str, Any]:
    findings: list[str] = []
    findings.extend(semantic_evidence_findings("tinygrad", tinygrad))
    findings.extend(semantic_evidence_findings("polygrad", polygrad))
    provenance_errors, provenance = provenance_findings(
        tinygrad, polygrad, manifest, config
    )
    findings.extend(provenance_errors)
    exact_fields: dict[str, bool] = {}
    for name in (
        "config", "dataset", "initial_state", "state_order", "permutations", "jit"
    ):
        equal = tinygrad[name] == polygrad[name]
        exact_fields[name] = equal
        if not equal:
            findings.append(f"{name}: records differ")

    tg_root, pg_root = run_root / "tinygrad", run_root / "polygrad"
    initial_findings, initial = exact_npz(
        tg_root / "initial_state.npz", pg_root / "initial_state.npz", "initial_state"
    )
    shared_findings, shared = shared_initial_state_findings(manifest, run_root)
    forward_findings, forward = numeric_npz(
        tg_root / "forward.npz", pg_root / "forward.npz", "forward", rtol, atol
    )
    final_findings, final = numeric_npz(
        tg_root / "final_state.npz", pg_root / "final_state.npz", "final_state", rtol, atol
    )
    findings.extend(initial_findings)
    findings.extend(shared_findings)
    findings.extend(forward_findings)
    findings.extend(final_findings)
    authoritative_path = Path(manifest["shared_initial_state"]["path"])
    for engine, root, result in (
        ("tinygrad", tg_root, tinygrad), ("polygrad", pg_root, polygrad)
    ):
        findings.extend(state_schema_findings(
            engine, result, authoritative_path,
            root / "initial_state.npz", root / "final_state.npz",
        ))

    step_detail: list[dict[str, Any]] = []
    expected_steps = int(config["STEPS"])
    for engine, records in (("tinygrad", tinygrad["steps"]), ("polygrad", polygrad["steps"])):
        findings.extend(step_record_findings(engine, records, expected_steps))
    if len(tinygrad["steps"]) != len(polygrad["steps"]):
        findings.append(
            f"steps: count {len(tinygrad['steps'])} != {len(polygrad['steps'])}"
        )
    for index, (tg_step, pg_step) in enumerate(zip(tinygrad["steps"], polygrad["steps"])):
        record = {"step": index}
        for key in ("loss", "lr"):
            a, b = float(tg_step[key]), float(pg_step[key])
            finite = bool(np.isfinite(a) and np.isfinite(b))
            ok = finite and bool(np.isclose(a, b, rtol=rtol, atol=atol))
            record[key] = {"tinygrad": a, "polygrad": b, "abs": abs(a - b), "ok": ok}
            if not ok:
                findings.append(
                    f"steps[{index}].{key}: {a:.9g} != {b:.9g} finite={finite}"
                )
        step_detail.append(record)

    kernel_equal = tinygrad["counters"]["kernel_count"] == polygrad["counters"]["kernel_count"]
    if not kernel_equal:
        findings.append(
            "counters.kernel_count: "
            f"{tinygrad['counters']['kernel_count']} != {polygrad['counters']['kernel_count']}"
        )
    counter_detail = {
        name: {
            "tinygrad": tinygrad["counters"][name],
            "polygrad": polygrad["counters"][name],
            "equal": tinygrad["counters"][name] == polygrad["counters"][name],
        }
        for name in ("kernel_count", "global_ops", "global_mem", "mem_used")
    }

    return {
        "status": "passed" if not findings else "failed",
        "rtol": rtol,
        "atol": atol,
        "findings": findings,
        "provenance": provenance,
        "exact_fields": exact_fields,
        "initial_state": initial,
        "shared_initial_state": shared,
        "forward": forward,
        "final_state": final,
        "steps": step_detail,
        "counters": counter_detail,
    }


def exact_step_value_findings(
    engine: str, label: str, records: list[dict[str, Any]], expected_steps: int = 4
) -> list[str]:
    """Validate exact raw loss/LR records before they can prove equality."""
    errors: list[str] = []
    prefix = f"replay.{label}.{engine}.exact_step_values"
    if len(records) != expected_steps:
        errors.append(
            f"{prefix}: recorded {len(records)}, expected {expected_steps}"
        )
    array_keys = {"shape", "dtype", "sha256", "finite", "values_hex"}
    for index, record in enumerate(records):
        record_prefix = f"{prefix}[{index}]"
        if not isinstance(record, dict) or set(record) != {"step", "loss", "lr"}:
            errors.append(f"{record_prefix}: record schema differs")
            continue
        if type(record["step"]) is not int or record["step"] != index:
            errors.append(f"{record_prefix}: step {record['step']!r} != {index}")
        if not isinstance(record["lr"], list) or len(record["lr"]) != 2:
            errors.append(f"{record_prefix}: expected two LR records")
            continue
        for array_name, array in (
            ("loss", record["loss"]),
            ("lr[0]", record["lr"][0]),
            ("lr[1]", record["lr"][1]),
        ):
            array_prefix = f"{record_prefix}.{array_name}"
            if not isinstance(array, dict) or set(array) != array_keys:
                errors.append(f"{array_prefix}: schema differs")
                continue
            shape = array["shape"]
            valid_shape = (
                isinstance(shape, list) and
                all(
                    isinstance(dim, int) and not isinstance(dim, bool) and dim >= 0
                    for dim in shape
                )
            )
            if not valid_shape:
                errors.append(f"{array_prefix}: invalid shape {shape!r}")
            try:
                dtype = np.dtype(array["dtype"])
                valid_dtype = dtype.kind == "f"
            except (TypeError, ValueError):
                dtype, valid_dtype = None, False
            if not valid_dtype:
                errors.append(f"{array_prefix}: invalid float dtype {array['dtype']!r}")
            digest = array["sha256"]
            valid_digest = (
                isinstance(digest, str) and len(digest) == 64 and
                all(char in "0123456789abcdef" for char in digest)
            )
            if not valid_digest:
                errors.append(f"{array_prefix}: invalid sha256 {digest!r}")
            values_hex = array["values_hex"]
            count = 1
            if valid_shape:
                for dim in shape:
                    count *= dim
            valid_values = isinstance(values_hex, list)
            if not valid_values:
                errors.append(f"{array_prefix}: values_hex is not a list")
            if valid_values and valid_shape and len(values_hex) != count:
                errors.append(
                    f"{array_prefix}: value count {len(values_hex)} != {count}"
                )
                valid_values = False
            parsed_values = []
            if valid_values:
                for value in values_hex:
                    try:
                        parsed = float.fromhex(value) if isinstance(value, str) else None
                    except ValueError:
                        parsed = None
                    if parsed is None or not np.isfinite(parsed) or parsed.hex() != value:
                        errors.append(f"{array_prefix}: invalid exact value {value!r}")
                        valid_values = False
                        break
                    parsed_values.append(parsed)
            if array["finite"] is not True:
                errors.append(f"{array_prefix}: finite is not true")
            if valid_shape and valid_dtype and valid_values and valid_digest:
                try:
                    reconstructed = np.asarray(parsed_values, dtype=dtype).reshape(shape)
                except (OverflowError, TypeError, ValueError):
                    errors.append(f"{array_prefix}: values cannot reconstruct array")
                else:
                    actual = hashlib.sha256(
                        np.ascontiguousarray(reconstructed).tobytes(order="C")
                    ).hexdigest()
                    if actual != digest:
                        errors.append(
                            f"{array_prefix}: sha256 {digest} != reconstructed {actual}"
                        )
    return errors


def replay_canary_findings(
    retained: dict[str, dict[str, Any]],
    eager: dict[str, dict[str, Any]],
    retained_manifest: dict[str, Any],
    eager_manifest: dict[str, Any],
    timing_manifest: dict[str, Any],
    retained_root: Path,
    eager_root: Path,
) -> tuple[list[str], dict[str, Any]]:
    """Compare retained replay with a reset-after-capture eager control."""
    findings: list[str] = []
    detail: dict[str, Any] = {"engines": {}}
    for label, candidate in (
        ("retained", retained_manifest), ("eager", eager_manifest)
    ):
        binding_findings, binding = manifest_binding_findings(
            candidate, timing_manifest
        )
        findings.extend(f"replay.{label}.{item}" for item in binding_findings)
        detail[f"{label}_binding"] = binding

    for label, results, manifest, config, root in (
        ("retained", retained, retained_manifest, REPLAY_CONFIG, retained_root),
        ("eager", eager, eager_manifest, REPLAY_EAGER_CONFIG, eager_root),
    ):
        provenance_errors, provenance = provenance_findings(
            results["tinygrad"], results["polygrad"], manifest, config
        )
        findings.extend(f"replay.{label}.{item}" for item in provenance_errors)
        shared_errors, shared = shared_initial_state_findings(manifest, root)
        findings.extend(f"replay.{label}.{item}" for item in shared_errors)
        detail[label] = {"provenance": provenance, "shared_initial_state": shared}

    authoritative_path = Path(timing_manifest["shared_initial_state"]["path"])
    expected_jit = {
        "retained": {"cnt": 4, "captured": True, "replay_count": 2},
        "eager": {"cnt": 2, "captured": True, "replay_count": 0},
    }
    expected_phases = {
        "retained": ["eager", "capture", "first_replay", "steady_replay"],
        "eager": ["eager", "capture", "eager", "capture"],
    }

    for engine in ("tinygrad", "polygrad"):
        engine_detail: dict[str, Any] = {}
        for label, results, root in (
            ("retained", retained, retained_root),
            ("eager", eager, eager_root),
        ):
            result = results[engine]
            findings.extend(
                f"replay.{label}.{item}"
                for item in step_record_findings(engine, result.get("steps") or [], 4)
            )
            schema_errors = state_schema_findings(
                engine, result, authoritative_path,
                root / engine / "initial_state.npz",
                root / engine / "final_state.npz",
            )
            findings.extend(f"replay.{label}.{item}" for item in schema_errors)
            observed_jits = {}
            for jit_name in ("augmentations", "train_step"):
                matches = [
                    record for record in result.get("jit") or []
                    if jit_name in record.get("name", "")
                ]
                if len(matches) != 1:
                    findings.append(
                        f"replay.{label}.{engine}.jit.{jit_name}: expected one"
                    )
                    observed_jits[jit_name] = None
                    continue
                observed = {
                    name: matches[0].get(name)
                    for name in ("cnt", "captured", "replay_count")
                }
                observed_jits[jit_name] = observed
                if observed != expected_jit[label]:
                    findings.append(
                        f"replay.{label}.{engine}.jit.{jit_name}: "
                        f"{observed!r} != {expected_jit[label]!r}"
                    )
            phases = [
                window.get("phase") for window in result.get("timing_windows") or []
            ]
            if phases != expected_phases[label]:
                findings.append(
                    f"replay.{label}.{engine}.phases: {phases!r} != "
                    f"{expected_phases[label]!r}"
                )
            exact_values = result.get("exact_step_values") or []
            findings.extend(exact_step_value_findings(engine, label, exact_values))
            engine_detail[label] = {
                "jit": observed_jits,
                "phases": phases,
                "exact_step_values": exact_values,
            }

        for field in ("dataset", "initial_state", "state_order"):
            if retained[engine].get(field) != eager[engine].get(field):
                findings.append(f"replay.{engine}.{field}: records differ")
        retained_values = retained[engine].get("exact_step_values") or []
        eager_values = eager[engine].get("exact_step_values") or []
        if retained_values != eager_values:
            findings.append(f"replay.{engine}.exact_step_values: records differ")
        engine_detail["exact_step_values"] = {
            "retained": retained_values,
            "eager": eager_values,
            "exact": retained_values == eager_values,
        }
        state_errors, state = exact_npz(
            retained_root / engine / "final_state.npz",
            eager_root / engine / "final_state.npz",
            f"replay.{engine}.final_state",
        )
        findings.extend(state_errors)
        engine_detail["final_state"] = state
        detail["engines"][engine] = engine_detail
    return findings, detail


def benchmark_readiness(
    tinygrad: dict[str, Any], polygrad: dict[str, Any],
    config: dict[str, str] = SEMANTIC_CONFIG,
) -> dict[str, Any]:
    findings: list[str] = []
    if config["HLB_CAPTURE_FORWARD"] != "0" or \
       config["HLB_CAPTURE_PERMUTATIONS"] != "0":
        findings.append("timing run is instrumented with live Tensor capture")
    for engine, result in (("tinygrad", tinygrad), ("polygrad", polygrad)):
        train_jits = [
            record for record in result["jit"]
            if "train_step" in record["name"]
        ]
        if len(train_jits) != 1:
            findings.append(f"{engine}: expected one train-step JIT record")
            continue
        record = train_jits[0]
        if record["cnt"] < 3 or not record["captured"] or record["replay_count"] < 1:
            findings.append(
                f"{engine}: no proved eager/capture/replay lifecycle: {record!r}"
            )
    counters = {}
    for name in ("kernel_count", "global_ops", "global_mem", "mem_used"):
        a, b = tinygrad["counters"][name], polygrad["counters"][name]
        counters[name] = {"tinygrad": a, "polygrad": b, "equal": a == b}
        if a != b:
            findings.append(f"counter.{name}: {a} != {b}")
    findings.append(
        "semantic mode does not execute matched no-live-Tensor timing windows"
    )
    return {
        "status": "ready" if not findings else "not_ready",
        "findings": findings,
        "counters": counters,
    }


def timing_window_findings(
    engine: str, result: dict[str, Any], config: dict[str, str]
) -> tuple[list[str], list[dict[str, Any]]]:
    findings = step_record_findings(
        engine, result.get("steps") or [], int(config["STEPS"])
    )
    if result.get("exact_step_values") != []:
        findings.append(
            f"timing_windows.{engine}: exact step capture was not disabled"
        )
    windows = result.get("timing_windows") or []
    if len(windows) != len(TIMING_PHASES):
        findings.append(
            f"timing_windows.{engine}: parsed {len(windows)}, "
            f"expected {len(TIMING_PHASES)}"
        )
    observed_phases = [window.get("phase") for window in windows]
    if observed_phases != TIMING_PHASES:
        findings.append(
            f"timing_windows.{engine}: phases {observed_phases!r} != "
            f"{TIMING_PHASES!r}"
        )

    detail = []
    steps = result.get("steps") or []
    for index, (step, window) in enumerate(zip(steps, windows)):
        expected_jit = {
            "cnt": index + 1,
            "captured": index >= 1,
            "replay_count": max(0, index - 1),
        }
        if step.get("step") != index or window.get("step") != index:
            findings.append(
                f"timing_windows.{engine}[{index}]: step identity differs"
            )
        train_jits = window.get("train_jits") or []
        if len(train_jits) != 1:
            findings.append(
                f"timing_windows.{engine}[{index}]: expected one train JIT, "
                f"got {len(train_jits)}"
            )
            train_jit = None
        else:
            train_jit = train_jits[0]
            for key, expected in expected_jit.items():
                if train_jit.get(key) != expected:
                    findings.append(
                        f"timing_windows.{engine}[{index}].jit.{key}: "
                        f"{train_jit.get(key)!r} != {expected!r}"
                    )

        timings = {
            key: float(step.get(key, float("nan")))
            for key in ("run_ms", "enqueue_ms", "readback_ms")
        }
        valid_timings = (
            np.isfinite(timings["run_ms"]) and timings["run_ms"] > 0 and
            all(
                np.isfinite(timings[key]) and timings[key] >= 0
                for key in ("enqueue_ms", "readback_ms")
            )
        )
        if not valid_timings:
            findings.append(
                f"timing_windows.{engine}[{index}]: nonfinite/negative timing "
                f"{timings!r}"
            )
        residual_ms = abs(
            timings["run_ms"] - timings["enqueue_ms"] - timings["readback_ms"]
        )
        if not np.isfinite(residual_ms) or residual_ms > 0.03:
            findings.append(
                f"timing_windows.{engine}[{index}]: run != enqueue+readback, "
                f"residual_ms={residual_ms:.6g}"
            )
        counters = window.get("counters") or {}
        for name in ("global_ops", "global_mem", "kernel_count", "mem_used"):
            value = counters.get(name)
            if not isinstance(value, int) or value < 0:
                findings.append(
                    f"timing_windows.{engine}[{index}].counters.{name}: "
                    f"invalid {value!r}"
                )
        time_sum_s = counters.get("time_sum_s")
        if not isinstance(time_sum_s, (int, float)) or \
           not np.isfinite(time_sum_s) or time_sum_s < 0:
            findings.append(
                f"timing_windows.{engine}[{index}].counters.time_sum_s: "
                f"invalid {time_sum_s!r}"
            )
        detail.append({
            "step": index,
            "phase": window.get("phase"),
            "jit": train_jit,
            "timing": timings,
            "timing_sum_residual_ms": residual_ms,
            "counters": counters,
        })
    return findings, detail


def compare_timing_results(
    tinygrad: dict[str, Any],
    polygrad: dict[str, Any],
    manifest: dict[str, Any],
    run_root: Path,
    rtol: float,
    atol: float,
    config: dict[str, str],
) -> dict[str, Any]:
    findings: list[str] = []
    diagnostics: list[str] = []
    provenance_errors, provenance = provenance_findings(
        tinygrad, polygrad, manifest, config
    )
    findings.extend(provenance_errors)
    if config["HLB_CAPTURE_FORWARD"] != "0" or \
       config["HLB_CAPTURE_PERMUTATIONS"] != "0":
        findings.append("timing config retains live semantic-capture Tensors")
    if config["HLB_TRACK_JIT"] != "1":
        findings.append("timing config does not track JIT lifecycle")

    exact_fields: dict[str, bool] = {}
    for name in ("config", "dataset", "initial_state", "state_order", "jit"):
        equal = tinygrad.get(name) == polygrad.get(name)
        exact_fields[name] = equal
        if not equal:
            findings.append(f"{name}: records differ")
    for engine, result in (("tinygrad", tinygrad), ("polygrad", polygrad)):
        if result.get("forward") is not None:
            findings.append(f"{engine}: timing run retained forward capture")
        if result.get("permutations") != []:
            findings.append(f"{engine}: timing run retained permutation Tensors")
        if result.get("final_state") is None:
            findings.append(f"{engine}: timing run omitted post-window final state")
        eval_jits = [
            record for record in result.get("jit") or []
            if "eval_step" in record.get("name", "")
        ]
        if not eval_jits or any(record.get("cnt") != 0 for record in eval_jits):
            findings.append(f"{engine}: timing run executed evaluation JIT")

    shared_findings, shared = shared_initial_state_findings(manifest, run_root)
    findings.extend(shared_findings)
    authoritative_path = Path(manifest["shared_initial_state"]["path"])
    for engine, result in (("tinygrad", tinygrad), ("polygrad", polygrad)):
        findings.extend(state_schema_findings(
            engine, result, authoritative_path,
            run_root / engine / "initial_state.npz",
            run_root / engine / "final_state.npz",
        ))
    final_findings, final_diagnostics, final = diagnostic_numeric_npz(
        run_root / "tinygrad" / "final_state.npz",
        run_root / "polygrad" / "final_state.npz",
        "final_state", rtol, atol,
    )
    findings.extend(final_findings)
    diagnostics.extend(final_diagnostics)
    tg_findings, tg_windows = timing_window_findings("tinygrad", tinygrad, config)
    pg_findings, pg_windows = timing_window_findings("polygrad", polygrad, config)
    findings.extend(tg_findings)
    findings.extend(pg_findings)

    paired_steps = []
    for index, (tg_step, pg_step, tg_window, pg_window) in enumerate(zip(
        tinygrad.get("steps") or [], polygrad.get("steps") or [],
        tg_windows, pg_windows,
    )):
        record: dict[str, Any] = {
            "step": index,
            "phase": tg_window["phase"],
        }
        for key in ("loss", "lr"):
            a, b = float(tg_step[key]), float(pg_step[key])
            finite = bool(np.isfinite(a) and np.isfinite(b))
            ok = finite and bool(np.isclose(a, b, rtol=rtol, atol=atol))
            record[key] = {
                "tinygrad": a, "polygrad": b, "abs": abs(a - b), "ok": ok,
            }
            if not finite:
                findings.append(
                    f"steps[{index}].{key}: nonfinite {a:.9g}/{b:.9g}"
                )
            elif key == "lr" and not ok:
                findings.append(
                    f"steps[{index}].lr: {a:.9g} != {b:.9g}"
                )
            elif key == "loss" and not ok:
                diagnostics.append(
                    f"steps[{index}].loss: accumulated numeric drift "
                    f"{a:.9g} != {b:.9g}"
                )
        tg_run = tg_window["timing"]["run_ms"]
        pg_run = pg_window["timing"]["run_ms"]
        record["run_ms"] = {
            "tinygrad": tg_run,
            "polygrad": pg_run,
            "polygrad_over_tinygrad": pg_run / tg_run if tg_run > 0 else None,
        }
        for name in ("kernel_count", "global_mem"):
            a = tg_window["counters"].get(name)
            b = pg_window["counters"].get(name)
            equal = a == b
            record.setdefault("counters", {})[name] = {
                "tinygrad": a, "polygrad": b, "equal": equal,
            }
            if not equal:
                findings.append(f"steps[{index}].counter.{name}: {a} != {b}")
        for name in ("global_ops", "mem_used", "time_sum_s"):
            a = tg_window["counters"].get(name)
            b = pg_window["counters"].get(name)
            record.setdefault("counters", {})[name] = {
                "tinygrad": a, "polygrad": b, "equal": a == b,
            }
        paired_steps.append(record)

    for name in ("global_ops", "mem_used", "time_sum_s"):
        differing = [
            row["step"] for row in paired_steps
            if not row.get("counters", {}).get(name, {}).get("equal", False)
        ]
        if differing:
            diagnostics.append(
                f"counter.{name} differs at steps {differing}; reported but not "
                "used as a wall-time workload-equality proof"
            )

    phase_summary: dict[str, Any] = {}
    for phase in ("eager", "capture", "first_replay", "steady_replay"):
        rows = [row for row in paired_steps if row["phase"] == phase]
        tg_values = [row["run_ms"]["tinygrad"] for row in rows]
        pg_values = [row["run_ms"]["polygrad"] for row in rows]
        phase_summary[phase] = {
            "samples": len(rows),
            "tinygrad_run_ms": tg_values,
            "polygrad_run_ms": pg_values,
            "tinygrad_median_ms": statistics.median(tg_values) if tg_values else None,
            "polygrad_median_ms": statistics.median(pg_values) if pg_values else None,
            "polygrad_over_tinygrad_median": (
                statistics.median(pg_values) / statistics.median(tg_values)
                if tg_values and statistics.median(tg_values) > 0 else None
            ),
        }

    return {
        "status": "ready" if not findings else "not_ready",
        "findings": findings,
        "diagnostics": diagnostics,
        "rtol": rtol,
        "atol": atol,
        "provenance": provenance,
        "exact_fields": exact_fields,
        "shared_initial_state": shared,
        "final_state": final,
        "steps": paired_steps,
        "phase_summary": phase_summary,
    }


def timing_repeatability_findings(
    forward: dict[str, dict[str, Any]],
    reverse: dict[str, dict[str, Any]],
    forward_root: Path,
    reverse_root: Path,
) -> tuple[list[str], dict[str, Any]]:
    """Require each engine to reproduce its own trajectory in both run orders."""
    def step_signature(result: dict[str, Any]) -> list[tuple[Any, Any, Any]]:
        return [
            (record.get("step"), record.get("loss"), record.get("lr"))
            for record in result.get("steps") or []
        ]

    def window_signature(result: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "step": window.get("step"),
                "phase": window.get("phase"),
                "train_jits": window.get("train_jits"),
                "counters": {
                    name: (window.get("counters") or {}).get(name)
                    for name in (
                        "kernel_count", "global_ops", "global_mem", "mem_used"
                    )
                },
            }
            for window in result.get("timing_windows") or []
        ]

    findings: list[str] = []
    detail: dict[str, Any] = {}
    for engine in ("tinygrad", "polygrad"):
        first, second = forward[engine], reverse[engine]
        state_findings, state = exact_npz(
            forward_root / engine / "final_state.npz",
            reverse_root / engine / "final_state.npz",
            f"repeatability.{engine}.final_state",
        )
        findings.extend(state_findings)
        checks = {
            "steps": step_signature(first) == step_signature(second),
            "jit": first.get("jit") == second.get("jit"),
            "windows": window_signature(first) == window_signature(second),
            "final_state": not state_findings,
        }
        for name, same in checks.items():
            if not same and name != "final_state":
                findings.append(f"repeatability.{engine}.{name}: records differ")
        detail[engine] = {"checks": checks, "final_state": state}
    return findings, detail


def combine_timing_runs(runs: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    findings = [
        f"{label}: {finding}"
        for label, run in runs
        for finding in run["findings"]
    ]
    diagnostics = [
        f"{label}: {diagnostic}"
        for label, run in runs
        for diagnostic in run["diagnostics"]
    ]
    phase_summary: dict[str, Any] = {}
    for phase in ("eager", "capture", "first_replay", "steady_replay"):
        tg_values = [
            value for _, run in runs
            for value in run["phase_summary"][phase]["tinygrad_run_ms"]
        ]
        pg_values = [
            value for _, run in runs
            for value in run["phase_summary"][phase]["polygrad_run_ms"]
        ]
        phase_summary[phase] = {
            "samples": len(tg_values),
            "tinygrad_run_ms": tg_values,
            "polygrad_run_ms": pg_values,
            "tinygrad_median_ms": statistics.median(tg_values) if tg_values else None,
            "polygrad_median_ms": statistics.median(pg_values) if pg_values else None,
            "polygrad_over_tinygrad_median": (
                statistics.median(pg_values) / statistics.median(tg_values)
                if tg_values and statistics.median(tg_values) > 0 else None
            ),
        }
    return {
        "status": "ready" if not findings else "not_ready",
        "findings": findings,
        "diagnostics": diagnostics,
        "execution_orders": [label for label, _ in runs],
        "runs": {label: run for label, run in runs},
        "phase_summary": phase_summary,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("semantic", "timing"), default="semantic")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--polygrad-lib", type=Path, default=DEFAULT_LIB)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--manifest-only", action="store_true")
    for option in ("--mode", "--python", "--polygrad-lib"):
        occurrences = sum(
            token == option or token.startswith(f"{option}=") for token in argv
        )
        if occurrences > 1:
            parser.error(f"{option} may be specified only once")
    return parser.parse_args(argv)


def overall_status(semantic_status: str, readiness_status: str) -> str:
    if semantic_status != "passed":
        return "failed"
    return "ready" if readiness_status == "ready" else "not_ready"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if not np.isfinite(args.rtol) or not 0 <= args.rtol <= MAX_RTOL:
        raise SystemExit(f"--rtol must be within [0, {MAX_RTOL}], got {args.rtol!r}")
    if not np.isfinite(args.atol) or not 0 <= args.atol <= MAX_ATOL:
        raise SystemExit(f"--atol must be within [0, {MAX_ATOL}], got {args.atol!r}")
    # Preserve the virtual-environment launcher. Path.resolve() follows its
    # interpreter symlink and bypasses the venv's site-packages.
    python = Path(os.path.abspath(args.python))
    library = args.polygrad_lib.resolve()
    if not python.is_file():
        raise SystemExit(f"missing Python: {python}")
    if not library.is_file():
        raise SystemExit(f"missing Polygrad library: {library}; run make build/libpolygrad.so")
    if not WORKER.is_file() or not TINYGRAD.is_dir():
        raise SystemExit("missing tracked HLB worker or pinned tinygrad tree")
    config = SEMANTIC_CONFIG if args.mode == "semantic" else TIMING_CONFIG
    if args.manifest_only:
        manifest = host_manifest(python, library, args.mode, config)
        print("HLB_MANIFEST " + json.dumps({
            "status": "valid",
            "mode": args.mode,
            "tinygrad_commit": manifest["tinygrad"]["commit"],
            "hlb_sha256": manifest["hlb_sha256"],
            "polygrad_library": manifest["polygrad_library"],
        }, sort_keys=True))
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = (
        args.output_dir.resolve()
        if args.output_dir
        else ROOT / "temp" / "hlb_benchmark" / stamp
    )
    if run_root.exists():
        raise SystemExit(f"output directory already exists: {run_root}")
    run_root.mkdir(parents=True)
    manifest = host_manifest(python, library, args.mode, config)
    (run_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    shared_initial_state = run_root / "shared" / "initial_state.npz"
    shared_initial_state.parent.mkdir(parents=True, exist_ok=True)
    prepare_initial_state(
        python, library, run_root, shared_initial_state, args.timeout,
        SEMANTIC_CONFIG if args.mode == "timing" else config,
    )
    manifest["shared_initial_state"] = {
        "path": str(shared_initial_state),
        "sha256": sha256(shared_initial_state),
    }
    (run_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.mode == "semantic":
        tinygrad, tg_env = run_worker(
            "tinygrad", python, library, run_root, shared_initial_state,
            args.timeout, config, capture_forward=True, capture_final_state=True,
        )
        polygrad, pg_env = run_worker(
            "polygrad", python, library, run_root, shared_initial_state,
            args.timeout, config, capture_forward=True, capture_final_state=True,
        )
        semantic = compare_results(
            tinygrad, polygrad, manifest, run_root, args.rtol, args.atol, config
        )
        source_findings, source_revalidation = revalidate_manifest_binding(
            python, library, manifest, args.mode, config
        )
        semantic["findings"].extend(source_findings)
        semantic["status"] = "passed" if not semantic["findings"] else "failed"
        readiness = benchmark_readiness(tinygrad, polygrad, config)
        output = {
            "schema": 1,
            "mode": args.mode,
            "manifest": manifest,
            "environments": {"tinygrad": tg_env, "polygrad": pg_env},
            "tinygrad": tinygrad,
            "polygrad": polygrad,
            "execution_order": ["tinygrad", "polygrad"],
            "semantic": semantic,
            "source_revalidation": source_revalidation,
            "benchmark_readiness": readiness,
            "overall_status": overall_status(semantic["status"], readiness["status"]),
        }
    else:
        semantic_root = run_root / "semantic_canary"
        semantic_root.mkdir()
        semantic_manifest = host_manifest(
            python, library, "semantic", SEMANTIC_CONFIG
        )
        semantic_manifest["shared_initial_state"] = dict(
            manifest["shared_initial_state"]
        )
        (semantic_root / "manifest.json").write_text(
            json.dumps(semantic_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        semantic_tinygrad, semantic_tg_env = run_worker(
            "tinygrad", python, library, semantic_root, shared_initial_state,
            args.timeout, SEMANTIC_CONFIG, capture_forward=True,
            capture_final_state=True,
        )
        semantic_polygrad, semantic_pg_env = run_worker(
            "polygrad", python, library, semantic_root, shared_initial_state,
            args.timeout, SEMANTIC_CONFIG, capture_forward=True,
            capture_final_state=True,
        )
        semantic = compare_results(
            semantic_tinygrad, semantic_polygrad, semantic_manifest,
            semantic_root, args.rtol, args.atol, SEMANTIC_CONFIG,
        )
        binding_findings, binding = manifest_binding_findings(
            semantic_manifest, manifest
        )
        source_findings, post_semantic_revalidation = revalidate_manifest_binding(
            python, library, manifest, args.mode, config
        )
        binding_findings.extend(source_findings)
        semantic_output = {
            "schema": 1,
            "mode": "semantic",
            "manifest": semantic_manifest,
            "environments": {
                "tinygrad": semantic_tg_env,
                "polygrad": semantic_pg_env,
            },
            "tinygrad": semantic_tinygrad,
            "polygrad": semantic_polygrad,
            "execution_order": ["tinygrad", "polygrad"],
            "semantic": semantic,
            "source_revalidation": post_semantic_revalidation,
            "overall_status": semantic["status"],
        }
        (semantic_root / "comparison.json").write_text(
            json.dumps(semantic_output, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if semantic["status"] != "passed" or binding_findings:
            timing = {
                "status": "not_ready",
                "findings": binding_findings or [
                    "semantic canary failed; timing was not executed"
                ],
                "diagnostics": [],
                "phase_summary": {},
            }
            output = {
                "schema": 1,
                "mode": args.mode,
                "manifest": manifest,
                "semantic_canary": semantic_output,
                "semantic_binding": binding,
                "timing": timing,
                "overall_status": "failed",
            }
            output_path = run_root / "comparison.json"
            output_path.write_text(
                json.dumps(output, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"HLB semantic canary failed: {output_path}")
            return 1

        replay_root = run_root / "replay_canary"
        retained_root = replay_root / "retained"
        eager_root = replay_root / "eager"
        retained_root.mkdir(parents=True)
        eager_root.mkdir(parents=True)
        retained_manifest = host_manifest(
            python, library, "replay_retained", REPLAY_CONFIG
        )
        eager_manifest = host_manifest(
            python, library, "replay_eager", REPLAY_EAGER_CONFIG
        )
        for root, candidate in (
            (retained_root, retained_manifest), (eager_root, eager_manifest)
        ):
            candidate["shared_initial_state"] = dict(
                manifest["shared_initial_state"]
            )
            (root / "manifest.json").write_text(
                json.dumps(candidate, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        retained_tinygrad, retained_tg_env = run_worker(
            "tinygrad", python, library, retained_root, shared_initial_state,
            args.timeout, REPLAY_CONFIG, capture_forward=False,
            capture_final_state=True,
        )
        retained_polygrad, retained_pg_env = run_worker(
            "polygrad", python, library, retained_root, shared_initial_state,
            args.timeout, REPLAY_CONFIG, capture_forward=False,
            capture_final_state=True,
        )
        eager_tinygrad, eager_tg_env = run_worker(
            "tinygrad", python, library, eager_root, shared_initial_state,
            args.timeout, REPLAY_EAGER_CONFIG, capture_forward=False,
            capture_final_state=True,
        )
        eager_polygrad, eager_pg_env = run_worker(
            "polygrad", python, library, eager_root, shared_initial_state,
            args.timeout, REPLAY_EAGER_CONFIG, capture_forward=False,
            capture_final_state=True,
        )
        replay_findings, replay_detail = replay_canary_findings(
            {
                "tinygrad": retained_tinygrad,
                "polygrad": retained_polygrad,
            },
            {"tinygrad": eager_tinygrad, "polygrad": eager_polygrad},
            retained_manifest,
            eager_manifest,
            manifest,
            retained_root,
            eager_root,
        )
        pre_timing_findings, pre_timing_revalidation = revalidate_manifest_binding(
            python, library, manifest, args.mode, config
        )
        replay_findings.extend(pre_timing_findings)
        replay_output = {
            "status": "passed" if not replay_findings else "failed",
            "findings": replay_findings,
            "detail": replay_detail,
            "source_revalidation": pre_timing_revalidation,
            "retained": {
                "manifest": retained_manifest,
                "environments": {
                    "tinygrad": retained_tg_env,
                    "polygrad": retained_pg_env,
                },
                "tinygrad": retained_tinygrad,
                "polygrad": retained_polygrad,
            },
            "eager": {
                "manifest": eager_manifest,
                "environments": {
                    "tinygrad": eager_tg_env,
                    "polygrad": eager_pg_env,
                },
                "tinygrad": eager_tinygrad,
                "polygrad": eager_polygrad,
            },
        }
        (replay_root / "comparison.json").write_text(
            json.dumps(replay_output, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if replay_findings:
            output = {
                "schema": 1,
                "mode": args.mode,
                "manifest": manifest,
                "semantic_canary": semantic_output,
                "semantic_binding": binding,
                "replay_canary": replay_output,
                "timing": {
                    "status": "not_ready",
                    "findings": ["replay semantic canary failed"],
                    "diagnostics": [],
                    "phase_summary": {},
                },
                "overall_status": "failed",
            }
            output_path = run_root / "comparison.json"
            output_path.write_text(
                json.dumps(output, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"HLB replay canary failed: {output_path}")
            return 1

        tinygrad, tg_env = run_worker(
            "tinygrad", python, library, run_root, shared_initial_state,
            args.timeout, config, capture_forward=False, capture_final_state=True,
        )
        polygrad, pg_env = run_worker(
            "polygrad", python, library, run_root, shared_initial_state,
            args.timeout, config, capture_forward=False, capture_final_state=True,
        )
        forward_timing = compare_timing_results(
            tinygrad, polygrad, manifest, run_root, args.rtol, args.atol, config
        )
        reverse_root = run_root / "polygrad_then_tinygrad"
        reverse_polygrad, reverse_pg_env = run_worker(
            "polygrad", python, library, reverse_root, shared_initial_state,
            args.timeout, config, capture_forward=False, capture_final_state=True,
        )
        reverse_tinygrad, reverse_tg_env = run_worker(
            "tinygrad", python, library, reverse_root, shared_initial_state,
            args.timeout, config, capture_forward=False, capture_final_state=True,
        )
        reverse_timing = compare_timing_results(
            reverse_tinygrad, reverse_polygrad, manifest, reverse_root,
            args.rtol, args.atol, config,
        )
        timing = combine_timing_runs([
            ("tinygrad_then_polygrad", forward_timing),
            ("polygrad_then_tinygrad", reverse_timing),
        ])
        repeatability_findings, repeatability = timing_repeatability_findings(
            {"tinygrad": tinygrad, "polygrad": polygrad},
            {"tinygrad": reverse_tinygrad, "polygrad": reverse_polygrad},
            run_root,
            reverse_root,
        )
        post_timing_findings, post_timing_revalidation = revalidate_manifest_binding(
            python, library, manifest, args.mode, config
        )
        timing["findings"].extend(repeatability_findings)
        timing["findings"].extend(post_timing_findings)
        timing["status"] = "ready" if not timing["findings"] else "not_ready"
        timing["repeatability"] = repeatability
        timing["source_revalidation"] = {
            "before_timing": pre_timing_revalidation,
            "after_timing": post_timing_revalidation,
        }
        output = {
            "schema": 1,
            "mode": args.mode,
            "manifest": manifest,
            "environments": {"tinygrad": tg_env, "polygrad": pg_env},
            "tinygrad": tinygrad,
            "polygrad": polygrad,
            "execution_order": ["tinygrad", "polygrad"],
            "semantic_canary": semantic_output,
            "semantic_binding": binding,
            "replay_canary": replay_output,
            "timing": timing,
            "counterbalanced": {
                "execution_order": ["polygrad", "tinygrad"],
                "environments": {
                    "tinygrad": reverse_tg_env,
                    "polygrad": reverse_pg_env,
                },
                "tinygrad": reverse_tinygrad,
                "polygrad": reverse_polygrad,
            },
            "overall_status": overall_status(
                semantic["status"], timing["status"]
            ),
        }
    output_path = run_root / "comparison.json"
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.mode == "semantic":
        print(f"HLB semantic {semantic['status']}: {output_path}")
        if semantic["findings"]:
            print(f"first finding: {semantic['findings'][0]}")
        print(
            "counters: "
            f"tinygrad={tinygrad['counters']} polygrad={polygrad['counters']}"
        )
        print(f"benchmark readiness: {readiness['status']}")
        if readiness["findings"]:
            print(f"first readiness finding: {readiness['findings'][0]}")
        return 0 if semantic["status"] == "passed" else 1
    print(f"HLB timing {timing['status']}: {output_path}")
    if timing["findings"]:
        print(f"first finding: {timing['findings'][0]}")
    for phase, record in timing["phase_summary"].items():
        print(
            f"{phase}: samples={record['samples']} "
            f"tinygrad={record['tinygrad_median_ms']}ms "
            f"polygrad={record['polygrad_median_ms']}ms "
            f"ratio={record['polygrad_over_tinygrad_median']}"
        )
    return 0 if timing["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
