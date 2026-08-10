#!/usr/bin/env python3
"""Execute pinned HLB CIFAR under one engine for the matched benchmark driver.

The HLB source remains byte-for-byte pinned for tinygrad. Polygrad changes only
the four ``from tinygrad`` imports. Both engines receive the same deterministic
in-memory CIFAR provider so the benchmark is network-independent.
"""

from pathlib import Path
import argparse
import builtins
import faulthandler
import hashlib
import json
import os
import re
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HLB = ROOT / "references" / "tinygrad_latest" / "examples" / "hlb_cifar10.py"

parser = argparse.ArgumentParser()
parser.add_argument("--engine", choices=("tinygrad", "polygrad"), required=True)
parser.add_argument("--prepare-state", action="store_true")
args = parser.parse_args()
is_tinygrad = args.engine == "tinygrad"
if args.prepare_state and not is_tinygrad:
    parser.error("--prepare-state requires --engine tinygrad")

for name, value in {
    "BS": "1",
    "EVAL_BS": "1",
    "STEPS": "1",
    "GPUS": "1",
    "BEAM": "0",
    "WINO": "0",
    "LATEBEAM": "0",
    "LATEWINO": "0",
    "DISABLE_BACKWARD": "1",
    "RANDOM_CROP": "0",
    "RANDOM_FLIP": "0",
    "CUTMIX": "0",
    "EMA": "0",
}.items():
    os.environ.setdefault(name, value)

source = HLB.read_text(encoding="utf-8")
source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
import_substitutions = 0
if not is_tinygrad:
    import_substitutions = source.count("from tinygrad")
    if import_substitutions != 4:
        raise RuntimeError(
            f"expected four pinned tinygrad imports, found {import_substitutions}"
        )
    source = source.replace("from tinygrad", "from polygrad")
    if "from tinygrad" in source:
        raise RuntimeError("incomplete tinygrad import substitution")
executed_source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()

source_lines = []
timing_windows = []
exact_step_values = []
step_prefix_pattern = re.compile(r"^\s*(?P<step>\d+)\s+[0-9.]+ ms run,")
reset_jits_after_step = int(os.getenv("HLB_RESET_JITS_AFTER_STEP", "-1"))
capture_exact_step_values = bool(
    int(os.getenv("HLB_CAPTURE_EXACT_STEP_VALUES", "0"))
)


def jit_state(name, jit):
    captured = getattr(jit, "captured", None)
    cnt = int(getattr(jit, "cnt", 0))
    return {
        "name": name,
        "cnt": cnt,
        "captured": bool(captured),
        "replay_count": int(getattr(jit, "replay_count", max(0, cnt - 2))),
    }


def jit_phase(record):
    cnt = record["cnt"]
    if cnt == 1:
        return "eager"
    if cnt == 2:
        return "capture"
    if cnt == 3:
        return "first_replay"
    if cnt >= 4:
        return "steady_replay"
    return "not_called"


def source_print(*values, **kwargs):
    """Mirror pinned output while retaining its structured step records."""
    if kwargs.get("file") in (None, sys.stdout):
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        rendered = sep.join(str(value) for value in values) + end
        source_lines.append(rendered)
        if (match := step_prefix_pattern.match(rendered)) is not None:
            train_pairs = [
                (name, jit) for name, jit in tracked_jits if "train_step" in name
            ]
            train_jits = [jit_state(name, jit) for name, jit in train_pairs]
            counters = namespace["GlobalCounters"]
            timing_windows.append({
                "step": int(match.group("step")),
                "phase": jit_phase(train_jits[0]) if len(train_jits) == 1 else None,
                "train_jits": train_jits,
                "counters": {
                    "global_ops": int(counters.global_ops),
                    "global_mem": int(counters.global_mem),
                    "kernel_count": int(counters.kernel_count),
                    "time_sum_s": float(counters.time_sum_s),
                    "mem_used": int(counters.mem_used),
                },
            })
            if int(match.group("step")) == reset_jits_after_step:
                reset_pairs = [
                    (name, jit) for name, jit in tracked_jits
                    if "augmentations" in name or "train_step" in name
                ]
                if len(reset_pairs) != 2:
                    raise RuntimeError(
                        "replay reset requires augmentation and train_step JITs"
                    )
                for _, jit in reset_pairs:
                    jit.reset()
    builtins.print(*values, **kwargs)


namespace = {
    "__name__": "hlb_exact_small_probe",
    "__file__": str(HLB),
    "print": source_print,
}
exec(compile(source, str(HLB), "exec"), namespace, namespace)

Tensor = namespace["Tensor"]
nn = namespace["nn"]
tracked_jits = []
if bool(int(os.getenv("HLB_TRACK_JIT", "1"))):
    original_tinyjit = namespace["TinyJit"]

    class TrackedTinyJit:
        """Transparent JIT proxy with optional untimed raw-result capture."""

        def __init__(self, name, jit):
            self.name = name
            self.jit = jit

        def __call__(self, *call_args, **call_kwargs):
            result = self.jit(*call_args, **call_kwargs)
            if capture_exact_step_values and self.name.endswith(".train_step"):
                if len(call_args) < 3 or len(call_args[2]) != 2:
                    raise RuntimeError(
                        "exact step capture requires two positional LR schedulers"
                    )
                exact_step_values.append({
                    "step": len(exact_step_values),
                    "loss": exact_array_record(result.numpy()),
                    "lr": [
                        exact_array_record(scheduler.optimizer.lr.numpy())
                        for scheduler in call_args[2]
                    ],
                })
            return result

        def __getattr__(self, name):
            return getattr(self.jit, name)

    def tracked_tinyjit(*jit_args, **jit_kwargs):
        jit = original_tinyjit(*jit_args, **jit_kwargs)
        fxn = jit_args[0] if jit_args and callable(jit_args[0]) else getattr(jit, "fxn", None)
        name = getattr(fxn, "__qualname__", getattr(fxn, "__name__", repr(fxn)))
        tracked = TrackedTinyJit(name, jit) if capture_exact_step_values else jit
        tracked_jits.append((name, tracked))
        return tracked

    namespace["TinyJit"] = tracked_tinyjit

def array_summary(data, *, include_values=False):
    arr = np.ascontiguousarray(np.asarray(data))
    raw = arr.view(np.uint8)
    summary = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "sum": float(arr.astype(np.float64).sum()),
    }
    if include_values:
        summary["values"] = arr.reshape(-1).astype(np.float64).tolist()
    else:
        summary["first"] = arr.reshape(-1)[:12].astype(np.float64).tolist()
    return summary


def exact_array_record(data):
    """Record exact numeric bytes without display-format quantization."""
    arr = np.asarray(data)
    contiguous = np.ascontiguousarray(arr)
    values = contiguous.reshape(-1)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
        "finite": bool(np.isfinite(values).all()),
        "values_hex": [float(value).hex() for value in values],
    }


def state_arrays_and_summary(model):
    state = namespace["get_state_dict"](model)
    arrays = {
        name: np.asarray(value.numpy()) for name, value in state.items()
    }
    summary = {
        name: array_summary(data)
        for name, data in arrays.items()
    }
    return state, arrays, summary


def canonical_tensor_dtype(dtype):
    name = getattr(dtype, "name", str(dtype))
    return {
        "bool": "bool",
        "char": "int8",
        "uchar": "uint8",
        "short": "int16",
        "ushort": "uint16",
        "int": "int32",
        "uint": "uint32",
        "long": "int64",
        "ulong": "uint64",
        "half": "float16",
        "float": "float32",
        "double": "float64",
    }.get(name, name)


capture_forward = bool(int(os.getenv("HLB_CAPTURE_FORWARD", "1")))
immediate_input_out = os.getenv("HLB_IMMEDIATE_INPUT_OUT")
initial_state_out = os.getenv("HLB_INITIAL_STATE_OUT")
final_state_out = os.getenv("HLB_FINAL_STATE_OUT")
capture_final_state = bool(final_state_out)
captured = {}
captured_input_immediate = {}
captured_model = {}
model_state_path = os.getenv("HLB_MODEL_STATE")


class InitialStatePrepared(Exception):
    pass


if capture_forward or initial_state_out or capture_final_state or model_state_path:
    state_arrays = None
    if model_state_path:
        with np.load(model_state_path, allow_pickle=False) as model_state:
            state_arrays = {
                name: np.asarray(model_state[name]) for name in model_state.files
            }
    original_model_init = namespace["SpeedyResNet"].__init__

    def model_init_with_state(self, *model_args, **kwargs):
        original_model_init(self, *model_args, **kwargs)
        if "model" not in captured_model:
            captured_model["model"] = self
        if capture_forward:
            captured["whitening"] = model_args[0]
        if state_arrays is not None:
            state = namespace["get_state_dict"](self)
            state_names = list(state)
            if state_names != list(state_arrays):
                raise RuntimeError(
                    "state name/order mismatch: "
                    f"model={state_names!r} archive={list(state_arrays)!r}"
                )
            for name, value in state.items():
                data = state_arrays[name]
                if tuple(value.shape) != data.shape:
                    raise RuntimeError(
                        f"state shape mismatch for {name}: "
                        f"model={tuple(value.shape)} archive={data.shape}"
                    )
                model_dtype = canonical_tensor_dtype(value.dtype)
                if model_dtype != data.dtype.name:
                    raise RuntimeError(
                        f"state dtype mismatch for {name}: "
                        f"model={model_dtype} archive={data.dtype.name}"
                    )
                if name != "whitening":
                    value.assign(data).realize()
        if initial_state_out and "initial_state" not in captured_model:
            _, arrays, summary = state_arrays_and_summary(self)
            np.savez(initial_state_out, **arrays)
            captured_model["initial_state"] = summary
            captured_model["state_order"] = list(arrays)
            if args.prepare_state:
                raise InitialStatePrepared

    namespace["SpeedyResNet"].__init__ = model_init_with_state

if capture_forward:
    original_model_call = namespace["SpeedyResNet"].__call__

    def model_call_with_capture(self, x, training=True):
        if training and immediate_input_out and "before_model" not in captured_input_immediate:
            captured_input_immediate["before_model"] = np.asarray(x.numpy()).copy()
        out = original_model_call(self, x, training=training)
        if training:
            captured["input"] = x
            captured["logits"] = out
            if immediate_input_out and "after_model" not in captured_input_immediate:
                captured_input_immediate["after_model"] = np.asarray(x.numpy()).copy()
        return out

    namespace["SpeedyResNet"].__call__ = model_call_with_capture

captured_permutations = []
if bool(int(os.getenv("HLB_CAPTURE_PERMUTATIONS", "1"))):
    original_randperm = Tensor.randperm

    def tracked_randperm(*rand_args, **rand_kwargs):
        result = original_randperm(*rand_args, **rand_kwargs)
        captured_permutations.append(result)
        return result

    Tensor.randperm = staticmethod(tracked_randperm)

dataset_arrays = {}


def synthetic_cifar():
    train_count = int(os.getenv("SYNTHETIC_TRAIN_SAMPLES", "2"))
    test_count = int(os.getenv("SYNTHETIC_TEST_SAMPLES", "1"))
    if train_count < namespace["BS"] or test_count < namespace["EVAL_BS"]:
        raise ValueError(
            "synthetic sample counts must cover one complete train/eval batch: "
            f"train={train_count} BS={namespace['BS']} "
            f"test={test_count} EVAL_BS={namespace['EVAL_BS']}"
        )
    train = (np.arange(train_count * 3 * 32 * 32, dtype=np.uint32) % 251).astype(np.uint8)
    train_labels = (np.arange(train_count, dtype=np.uint32) % 10).astype(np.uint8)
    test = (np.arange(test_count * 3 * 32 * 32, dtype=np.uint32) % 241).astype(np.uint8)
    test_labels = (np.arange(test_count, dtype=np.uint32) % 10).astype(np.uint8)
    dataset_arrays.update({
        "train_images": train.reshape(train_count, 3, 32, 32),
        "train_labels": train_labels,
        "test_images": test.reshape(test_count, 3, 32, 32),
        "test_labels": test_labels,
    })
    return (
        Tensor(dataset_arrays["train_images"]),
        Tensor(train_labels),
        Tensor(dataset_arrays["test_images"]),
        Tensor(test_labels),
    )


original_cifar = nn.datasets.cifar
stack_after = int(os.getenv("HLB_STACK_AFTER", "0"))
if stack_after > 0:
    faulthandler.dump_traceback_later(stack_after, repeat=True)
started = time.monotonic()
nn.datasets.cifar = synthetic_cifar
prepared_state = False
try:
    builtins.print(
        "CONFIG",
        "ENGINE", args.engine,
        "DEV", namespace["Device"].DEFAULT,
        "BS", namespace["BS"],
        "STEPS", namespace["STEPS"],
        "DISABLE_BACKWARD", os.environ["DISABLE_BACKWARD"],
    )
    namespace["train_cifar"]()
except InitialStatePrepared:
    prepared_state = True
finally:
    nn.datasets.cifar = original_cifar
    if stack_after > 0:
        faulthandler.cancel_dump_traceback_later()

if prepared_state:
    builtins.print(f"HLB_STATE_PREPARED {initial_state_out}")
    raise SystemExit(0)

global_counters = namespace["GlobalCounters"]
counter_record = {
    name: getattr(global_counters, name)
    for name in (
        "global_ops",
        "global_mem",
        "kernel_count",
        "time_sum_s",
        "mem_used",
    )
}

if capture_forward:
    capture_arrays = {
        name: np.asarray(value.numpy()) for name, value in captured.items()
    }
    if capture_out := os.getenv("HLB_FORWARD_OUT"):
        np.savez(capture_out, **capture_arrays)
    builtins.print(
        "FORWARD_CAPTURE",
        json.dumps(
            {name: array_summary(data) for name, data in capture_arrays.items()},
            sort_keys=True,
        ),
    )

if immediate_input_out:
    np.savez(immediate_input_out, **captured_input_immediate)
    builtins.print(
        "IMMEDIATE_INPUT_CAPTURE",
        json.dumps(
            {
                name: array_summary(data)
                for name, data in captured_input_immediate.items()
            },
            sort_keys=True,
        ),
    )

if capture_final_state:
    final_state, final_arrays, final_state_summary = state_arrays_and_summary(
        captured_model["model"]
    )
    np.savez(final_state_out, **final_arrays)
    builtins.print(
        "FINAL_STATE_CAPTURE",
        json.dumps(final_state_summary, sort_keys=True),
    )
else:
    final_state_summary = None

builtins.print("HLB_COUNTERS", json.dumps(counter_record, sort_keys=True))


jit_records = [jit_state(name, jit) for name, jit in tracked_jits]
builtins.print("HLB_JITS", json.dumps(jit_records, sort_keys=True))

step_pattern = re.compile(
    r"^\s*(?P<step>\d+)\s+(?P<run_ms>[0-9.]+) ms run,\s+"
    r"(?P<enqueue_ms>[0-9.]+) ms python,\s+(?P<readback_ms>[0-9.]+) ms "
    r"(?P<device>.*?),\s+(?P<loss>\S+) loss,\s+"
    r"(?P<lr>\S+) LR,\s+(?P<mem_gb>[0-9.]+) GB used,\s+"
    r"(?P<gflops>\S+) GFLOPS,\s+(?P<gops>\S+) GOPS$"
)
steps = []
for line in "".join(source_lines).splitlines():
    match = step_pattern.match(line)
    if match is None:
        continue
    record = match.groupdict()
    record["step"] = int(record["step"])
    for key in ("run_ms", "enqueue_ms", "readback_ms", "loss", "lr", "mem_gb", "gflops", "gops"):
        record[key] = float(record[key])
    steps.append(record)

permutations = [
    array_summary(np.asarray(value.numpy()), include_values=True)
    for value in captured_permutations
]


def path_record(path):
    value = Path(path).resolve()
    return {
        "path": str(value),
        "sha256": hashlib.sha256(value.read_bytes()).hexdigest(),
    }


def selected_cuda_device(logical_device):
    device = str(logical_device).upper()
    visible_record = os.getenv("CUDA_VISIBLE_DEVICES")
    device_order = os.getenv("CUDA_DEVICE_ORDER")
    if not device.startswith("CUDA"):
        return {}
    if device_order != "PCI_BUS_ID":
        raise RuntimeError(
            f"CUDA_DEVICE_ORDER must be PCI_BUS_ID, got {device_order!r}"
        )
    logical_ordinal = int(device.split(":", 1)[1]) if ":" in device else 0
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,pci.bus_id,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        timeout=10,
    )
    rows = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            raise RuntimeError(f"unexpected nvidia-smi device row: {line!r}")
        rows.append(dict(zip(
            ("index", "name", "uuid", "pci_bus_id", "driver_version", "compute_cap"),
            fields,
        )))
    ordered_rows = sorted(rows, key=lambda row: row["pci_bus_id"])
    visible = (
        [token.strip() for token in visible_record.split(",")]
        if visible_record not in (None, "") else
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
        "cuda_visible_devices": visible_record,
        "cuda_device_order": device_order,
        **matches[0],
    }


module_name = "tinygrad" if is_tinygrad else "polygrad"
module = sys.modules[module_name]
extra_module = sys.modules["extra.lr_scheduler"]
loaded_library = None
if not is_tinygrad:
    from polygrad import _ffi

    loaded_library = path_record(_ffi._lib._name)

result = {
    "schema": 1,
    "engine": args.engine,
    "config": {
        name: os.environ[name]
        for name in (
            "DEV", "POLY_DEVICE", "CUDA_DEVICE_ORDER", "BS", "EVAL_BS",
            "STEPS", "EVAL_STEPS",
            "GPUS", "SEED",
            "BEAM", "WINO", "LATEBEAM", "LATEWINO", "DISABLE_BACKWARD",
            "RANDOM_CROP", "RANDOM_FLIP", "CUTMIX", "EMA", "SYNCBN",
            "CACHELEVEL", "SCACHE", "POLY_SCACHE", "POLY_CACHE", "DEBUG",
            "VIZ", "PROFILE", "HLB_CAPTURE_FORWARD",
            "HLB_CAPTURE_PERMUTATIONS", "HLB_TRACK_JIT",
            "HLB_RESET_JITS_AFTER_STEP", "HLB_CAPTURE_EXACT_STEP_VALUES",
            "SYNTHETIC_TRAIN_SAMPLES", "SYNTHETIC_TEST_SAMPLES",
        )
        if name in os.environ
    },
    "source": {
        "hlb_path": str(HLB),
        "hlb_sha256": source_sha256,
        "executed_sha256": executed_source_sha256,
        "import_substitutions": import_substitutions,
        "worker": path_record(__file__),
        "engine_module": path_record(module.__file__),
        "extra_lr_scheduler": path_record(extra_module.__file__),
        "loaded_library": loaded_library,
    },
    "device": str(namespace["Device"].DEFAULT),
    "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
    "selected_cuda_device": selected_cuda_device(namespace["Device"].DEFAULT),
    "dataset": {
        name: array_summary(data) for name, data in dataset_arrays.items()
    },
    "initial_state": captured_model.get("initial_state"),
    "state_order": captured_model.get("state_order"),
    "forward": {
        name: array_summary(data) for name, data in capture_arrays.items()
    } if capture_forward else None,
    "permutations": permutations,
    "final_state": final_state_summary,
    "steps": steps,
    "exact_step_values": exact_step_values,
    "timing_windows": timing_windows,
    "jit": jit_records,
    "counters": counter_record,
    "elapsed_s": time.monotonic() - started,
}
builtins.print("HLB_RESULT", json.dumps(result, sort_keys=True))
builtins.print(f"HLB_WORKER_COMPLETED elapsed={result['elapsed_s']:.3f}s")
