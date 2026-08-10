import json
import numpy as np
import pytest
import subprocess

from bench import bench_hlb_cifar as hlb


def test_numeric_npz_rejects_matching_nonfinite_values(tmp_path):
    reference = tmp_path / "reference.npz"
    subject = tmp_path / "subject.npz"
    data = np.asarray([1.0, np.nan, np.inf], dtype=np.float32)
    np.savez(reference, value=data)
    np.savez(subject, value=data)

    findings, detail = hlb.numeric_npz(
        reference, subject, "nonfinite", rtol=1e-5, atol=1e-5
    )

    assert findings
    assert not detail["arrays"]["value"]["all_finite"]
    assert not detail["arrays"]["value"]["within_tolerance"]


def test_timing_numeric_drift_is_diagnostic_but_nonfinite_is_blocking(tmp_path):
    reference = tmp_path / "reference.npz"
    subject = tmp_path / "subject.npz"
    np.savez(reference, value=np.asarray([1.0], dtype=np.float32))
    np.savez(subject, value=np.asarray([2.0], dtype=np.float32))

    findings, diagnostics, detail = hlb.diagnostic_numeric_npz(
        reference, subject, "trajectory", rtol=1e-5, atol=1e-5
    )

    assert findings == []
    assert diagnostics and "accumulated numeric drift" in diagnostics[0]
    assert not detail["arrays"]["value"]["within_tolerance"]

    np.savez(subject, value=np.asarray([np.nan], dtype=np.float32))
    findings, diagnostics, _ = hlb.diagnostic_numeric_npz(
        reference, subject, "trajectory", rtol=1e-5, atol=1e-5
    )
    assert findings and "nonfinite" in findings[0]
    assert diagnostics == []


def test_timing_manifest_is_bound_to_fresh_semantic_canary():
    fields = (
        "polygrad", "tinygrad", "driver_sha256", "worker_sha256",
        "makefile_sha256", "hlb_sha256", "polygrad_library", "python", "host",
        "shared_initial_state",
    )
    semantic = {name: {"value": name} for name in fields}
    timing = {name: dict(value) for name, value in semantic.items()}

    findings, equal = hlb.manifest_binding_findings(semantic, timing)
    assert findings == []
    assert all(equal.values())

    timing["polygrad_library"]["value"] = "stale"
    findings, equal = hlb.manifest_binding_findings(semantic, timing)
    assert findings == [
        "semantic_binding.polygrad_library: semantic/timing records differ"
    ]
    assert not equal["polygrad_library"]


def test_cuda_device_binding_preserves_visible_ordinal_and_physical_identity():
    inventory = (
        "7, GPU A, GPU-aaaa, 00000000:02:00.0, 555.1, 8.0\n"
        "3, GPU B, GPU-bbbb, 00000000:01:00.0, 555.1, 8.6"
    )

    default = hlb.selected_cuda_device("CUDA:0", None, "PCI_BUS_ID", inventory)
    first = hlb.selected_cuda_device("CUDA:0", "1,0", "PCI_BUS_ID", inventory)
    second = hlb.selected_cuda_device("CUDA:1", "1,0", "PCI_BUS_ID", inventory)

    assert default["index"] == "3"
    assert default["pci_bus_id"] == "00000000:01:00.0"
    assert first["logical_device"] == "CUDA:0"
    assert first["index"] == "7"
    assert first["uuid"] == "GPU-aaaa"
    assert first["pci_bus_id"] == "00000000:02:00.0"
    assert second["index"] == "3"


def test_source_revalidation_rejects_mutation_after_canary(monkeypatch):
    fields = (
        "polygrad", "tinygrad", "driver_sha256", "worker_sha256",
        "makefile_sha256", "hlb_sha256", "polygrad_library", "python", "host",
    )
    baseline = {name: {"value": name} for name in fields}
    baseline["shared_initial_state"] = {"path": "/state", "sha256": "state"}
    current = {name: dict(value) for name, value in baseline.items()}
    current["polygrad"]["value"] = "mutated"
    monkeypatch.setattr(hlb, "host_manifest", lambda *_: current)

    findings, detail = hlb.revalidate_manifest_binding(
        hlb.DEFAULT_PYTHON, hlb.DEFAULT_LIB, baseline, "timing", hlb.TIMING_CONFIG
    )

    assert findings == [
        "semantic_binding.polygrad: semantic/timing records differ"
    ]
    assert not detail["binding"]["polygrad"]


def test_failed_semantic_canary_prevents_timing_workers(tmp_path, monkeypatch):
    library = tmp_path / "libpolygrad.so"
    library.write_bytes(b"test")
    output = tmp_path / "run"
    fields = (
        "polygrad", "tinygrad", "driver_sha256", "worker_sha256",
        "makefile_sha256", "hlb_sha256", "polygrad_library", "python", "host",
    )
    baseline = {name: {"value": name} for name in fields}

    def manifest(*_):
        return {name: dict(value) for name, value in baseline.items()}

    def prepare(_, __, ___, path, ____, _____):
        np.savez(path, weight=np.asarray([1.0], dtype=np.float32))

    calls = []

    def worker(engine, *args, **kwargs):
        calls.append(engine)
        return {}, {}

    monkeypatch.setattr(hlb, "host_manifest", manifest)
    monkeypatch.setattr(hlb, "prepare_initial_state", prepare)
    monkeypatch.setattr(hlb, "run_worker", worker)
    monkeypatch.setattr(
        hlb, "compare_results", lambda *_: {"status": "failed", "findings": ["boom"]}
    )

    status = hlb.main([
        "--mode", "timing",
        "--python", "/usr/bin/python3",
        "--polygrad-lib", str(library),
        "--output-dir", str(output),
    ])

    assert status == 1
    assert calls == ["tinygrad", "polygrad"]
    comparison = json.loads((output / "comparison.json").read_text())
    assert comparison["overall_status"] == "failed"
    assert comparison["timing"]["status"] == "not_ready"


def test_exact_npz_requires_named_state_identity(tmp_path):
    reference = tmp_path / "reference.npz"
    subject = tmp_path / "subject.npz"
    data = np.asarray([1.0, 2.0], dtype=np.float32)
    np.savez(reference, weight=data)
    np.savez(subject, renamed_weight=data)

    findings, _ = hlb.exact_npz(reference, subject, "state")

    assert findings == ["state: archive keys differ"]


def test_exact_npz_rejects_identical_nonfinite_state(tmp_path):
    reference = tmp_path / "reference.npz"
    subject = tmp_path / "subject.npz"
    np.savez(reference, weight=np.asarray([np.inf], dtype=np.float32))
    np.savez(subject, weight=np.asarray([np.inf], dtype=np.float32))

    findings, detail = hlb.exact_npz(reference, subject, "state")

    assert findings == ["state.weight: numeric/finite=True/False vs True/False"]
    assert detail["arrays"]["weight"]["exact"]
    assert not detail["arrays"]["weight"]["reference_finite"]


def test_step_records_cannot_vacuously_parse_zero_steps():
    findings = hlb.step_record_findings("tinygrad", [], expected_steps=1)
    assert findings == [
        "steps.tinygrad: parsed 0 records, expected 1",
        "steps.tinygrad: indices [] != [0]",
    ]


def test_output_directory_must_be_new(tmp_path):
    with pytest.raises(SystemExit, match="output directory already exists"):
        hlb.main(["--output-dir", str(tmp_path)])


@pytest.mark.parametrize(
    ("flag", "value"),
    (("--rtol", "nan"), ("--atol", "-1"), ("--atol", "1e308")),
)
def test_tolerances_must_stay_within_reviewed_bound(flag, value):
    with pytest.raises(SystemExit, match=r"must be within \[0,"):
        hlb.main([flag, value])


def test_locked_runtime_arguments_cannot_be_duplicated(capsys):
    with pytest.raises(SystemExit) as exc:
        hlb.parse_args([
            "--polygrad-lib", "/first.so", "--polygrad-lib", "/stale.so"
        ])
    assert exc.value.code == 2
    assert "--polygrad-lib may be specified only once" in capsys.readouterr().err


def test_shared_archive_is_authoritative_for_both_engines(tmp_path):
    shared = tmp_path / "shared.npz"
    np.savez(shared, weight=np.asarray([1.0], dtype=np.float32))
    for engine in ("tinygrad", "polygrad"):
        root = tmp_path / engine
        root.mkdir()
        np.savez(root / "initial_state.npz", weight=np.asarray([2.0], dtype=np.float32))
    manifest = {
        "shared_initial_state": {"path": str(shared), "sha256": hlb.sha256(shared)}
    }

    findings, _ = hlb.shared_initial_state_findings(manifest, tmp_path)

    assert findings == [
        "shared_initial_state.tinygrad.weight: bytes differ",
        "shared_initial_state.polygrad.weight: bytes differ",
    ]


def test_missing_semantic_instrumentation_is_not_accepted():
    findings = hlb.semantic_evidence_findings("tinygrad", {})
    assert any("dataset" in finding for finding in findings)
    assert any("state_order" in finding for finding in findings)
    assert any("forward" in finding for finding in findings)
    assert any("permutations" in finding for finding in findings)
    assert any("jit" in finding for finding in findings)


def test_overall_status_distinguishes_failure_from_incomplete_timing():
    assert hlb.overall_status("failed", "not_ready") == "failed"
    assert hlb.overall_status("passed", "not_ready") == "not_ready"
    assert hlb.overall_status("passed", "ready") == "ready"


def test_instrumented_one_call_run_is_not_benchmark_ready():
    result = {
        "jit": [{
            "name": "train_cifar.<locals>.train_step",
            "cnt": 1,
            "captured": False,
            "replay_count": 0,
        }],
        "counters": {
            "kernel_count": 1,
            "global_ops": 2,
            "global_mem": 3,
            "mem_used": 4,
        },
    }

    readiness = hlb.benchmark_readiness(result, result)

    assert readiness["status"] == "not_ready"
    assert any("instrumented" in finding for finding in readiness["findings"])
    assert any("no proved eager/capture/replay" in finding for finding in readiness["findings"])


def timing_result():
    steps = []
    windows = []
    for index, phase in enumerate(hlb.TIMING_PHASES):
        steps.append({
            "step": index,
            "run_ms": 10.0 + index,
            "enqueue_ms": 4.0 + index,
            "readback_ms": 6.0,
            "loss": 1.0,
            "lr": 0.01,
        })
        windows.append({
            "step": index,
            "phase": phase,
            "train_jits": [{
                "name": "train_cifar.<locals>.train_step",
                "cnt": index + 1,
                "captured": index >= 1,
                "replay_count": max(0, index - 1),
            }],
            "counters": {
                "global_ops": 100,
                "global_mem": 200,
                "kernel_count": 3,
                "time_sum_s": 0.0,
                "mem_used": 400,
            },
        })
    return {
        "steps": steps,
        "exact_step_values": [],
        "timing_windows": windows,
        "jit": [{"name": "train", "cnt": 8, "captured": True, "replay_count": 6}],
    }


def test_counterbalanced_runs_require_exact_same_engine_repeatability(tmp_path):
    forward_root = tmp_path / "forward"
    reverse_root = tmp_path / "reverse"
    for root in (forward_root, reverse_root):
        for engine in ("tinygrad", "polygrad"):
            path = root / engine
            path.mkdir(parents=True)
            np.savez(path / "final_state.npz", weight=np.asarray([1.0], dtype=np.float32))
    forward = {engine: timing_result() for engine in ("tinygrad", "polygrad")}
    reverse = {engine: timing_result() for engine in ("tinygrad", "polygrad")}

    findings, detail = hlb.timing_repeatability_findings(
        forward, reverse, forward_root, reverse_root
    )
    assert findings == []
    assert all(all(record["checks"].values()) for record in detail.values())

    np.savez(
        reverse_root / "polygrad" / "final_state.npz",
        weight=np.asarray([2.0], dtype=np.float32),
    )
    findings, _ = hlb.timing_repeatability_findings(
        forward, reverse, forward_root, reverse_root
    )
    assert findings == ["repeatability.polygrad.final_state.weight: bytes differ"]


def test_timing_config_has_uninstrumented_repeated_steady_replay():
    assert hlb.TIMING_CONFIG["HLB_CAPTURE_FORWARD"] == "0"
    assert hlb.TIMING_CONFIG["HLB_CAPTURE_PERMUTATIONS"] == "0"
    assert int(hlb.TIMING_CONFIG["EVAL_STEPS"]) > int(hlb.TIMING_CONFIG["STEPS"])
    assert hlb.TIMING_PHASES[:3] == ["eager", "capture", "first_replay"]
    assert hlb.TIMING_PHASES[3:] == ["steady_replay"] * 5
    assert hlb.REPLAY_CONFIG["HLB_RESET_JITS_AFTER_STEP"] == "-1"
    assert hlb.REPLAY_EAGER_CONFIG["HLB_RESET_JITS_AFTER_STEP"] == "1"
    assert hlb.TIMING_CONFIG["HLB_CAPTURE_EXACT_STEP_VALUES"] == "0"
    assert hlb.REPLAY_CONFIG["HLB_CAPTURE_EXACT_STEP_VALUES"] == "1"


def test_replay_canary_requires_same_engine_eager_equivalence(tmp_path, monkeypatch):
    retained_root = tmp_path / "retained"
    eager_root = tmp_path / "eager"
    shared = tmp_path / "shared.npz"
    np.savez(shared, weight=np.asarray([1.0], dtype=np.float32))
    for root in (retained_root, eager_root):
        for engine in ("tinygrad", "polygrad"):
            path = root / engine
            path.mkdir(parents=True)
            np.savez(path / "initial_state.npz", weight=np.asarray([1.0], dtype=np.float32))
            np.savez(path / "final_state.npz", weight=np.asarray([2.0], dtype=np.float32))

    def result(*, reset):
        phases = (
            ["eager", "capture", "eager", "capture"]
            if reset else
            ["eager", "capture", "first_replay", "steady_replay"]
        )
        def exact(value):
            data = np.asarray(value, dtype=np.float32)
            return {
                "shape": list(data.shape),
                "dtype": str(data.dtype),
                "sha256": hlb.hashlib.sha256(data.tobytes()).hexdigest(),
                "finite": True,
                "values_hex": [float(item).hex() for item in data.reshape(-1)],
            }

        return {
            "dataset": {"same": True},
            "initial_state": {"weight": {}},
            "final_state": {"weight": {}},
            "state_order": ["weight"],
            "steps": [
                {"step": step, "loss": float(step), "lr": 0.01}
                for step in range(4)
            ],
            "exact_step_values": [
                {
                    "step": step,
                    "loss": exact(float(step)),
                    "lr": [exact([0.02]), exact([0.01])],
                }
                for step in range(4)
            ],
            "timing_windows": [
                {"step": step, "phase": phase} for step, phase in enumerate(phases)
            ],
            "jit": [
                {
                    "name": f"train_cifar.<locals>.{name}",
                    "cnt": 2 if reset else 4,
                    "captured": True,
                    "replay_count": 0 if reset else 2,
                }
                for name in ("augmentations", "train_step")
            ],
        }

    retained = {engine: result(reset=False) for engine in ("tinygrad", "polygrad")}
    eager = {engine: result(reset=True) for engine in ("tinygrad", "polygrad")}
    manifest = {"shared_initial_state": {"path": str(shared)}}
    monkeypatch.setattr(hlb, "manifest_binding_findings", lambda *_: ([], {}))
    monkeypatch.setattr(hlb, "provenance_findings", lambda *_: ([], {}))
    monkeypatch.setattr(hlb, "shared_initial_state_findings", lambda *_: ([], {}))
    monkeypatch.setattr(hlb, "state_schema_findings", lambda *_: [])

    findings, detail = hlb.replay_canary_findings(
        retained, eager, manifest, manifest, manifest, retained_root, eager_root
    )
    assert findings == []
    assert detail["engines"]["polygrad"]["final_state"]["arrays"]["weight"]["exact"]

    np.savez(
        eager_root / "polygrad" / "final_state.npz",
        weight=np.asarray([3.0], dtype=np.float32),
    )
    findings, _ = hlb.replay_canary_findings(
        retained, eager, manifest, manifest, manifest, retained_root, eager_root
    )
    assert findings == ["replay.polygrad.final_state.weight: bytes differ"]


def test_replay_canary_rejects_sub_display_precision_loss_mismatch(
    tmp_path, monkeypatch
):
    retained_root = tmp_path / "retained"
    eager_root = tmp_path / "eager"
    shared = tmp_path / "shared.npz"
    np.savez(shared, weight=np.asarray([1.0], dtype=np.float32))
    for root in (retained_root, eager_root):
        for engine in ("tinygrad", "polygrad"):
            path = root / engine
            path.mkdir(parents=True)
            np.savez(path / "initial_state.npz", weight=np.asarray([1.0], dtype=np.float32))
            np.savez(path / "final_state.npz", weight=np.asarray([2.0], dtype=np.float32))

    def exact(value):
        data = np.asarray(value, dtype=np.float32)
        return {
            "shape": list(data.shape),
            "dtype": str(data.dtype),
            "sha256": hlb.hashlib.sha256(data.tobytes()).hexdigest(),
            "finite": True,
            "values_hex": [float(item).hex() for item in data.reshape(-1)],
        }

    def result(*, reset):
        phases = (
            ["eager", "capture", "eager", "capture"]
            if reset else
            ["eager", "capture", "first_replay", "steady_replay"]
        )
        return {
            "dataset": {"same": True},
            "initial_state": {"weight": {}},
            "final_state": {"weight": {}},
            "state_order": ["weight"],
            "steps": [
                {"step": step, "loss": 1.23, "lr": 0.01} for step in range(4)
            ],
            "exact_step_values": [
                {
                    "step": step,
                    "loss": exact(1.231 if reset and step == 2 else 1.230),
                    "lr": [exact([0.02]), exact([0.01])],
                }
                for step in range(4)
            ],
            "timing_windows": [
                {"step": step, "phase": phase} for step, phase in enumerate(phases)
            ],
            "jit": [
                {
                    "name": f"train_cifar.<locals>.{name}",
                    "cnt": 2 if reset else 4,
                    "captured": True,
                    "replay_count": 0 if reset else 2,
                }
                for name in ("augmentations", "train_step")
            ],
        }

    retained = {engine: result(reset=False) for engine in ("tinygrad", "polygrad")}
    eager = {engine: result(reset=True) for engine in ("tinygrad", "polygrad")}
    manifest = {"shared_initial_state": {"path": str(shared)}}
    monkeypatch.setattr(hlb, "manifest_binding_findings", lambda *_: ([], {}))
    monkeypatch.setattr(hlb, "provenance_findings", lambda *_: ([], {}))
    monkeypatch.setattr(hlb, "shared_initial_state_findings", lambda *_: ([], {}))
    monkeypatch.setattr(hlb, "state_schema_findings", lambda *_: [])

    findings, _ = hlb.replay_canary_findings(
        retained, eager, manifest, manifest, manifest, retained_root, eager_root
    )

    assert findings == [
        "replay.tinygrad.exact_step_values: records differ",
        "replay.polygrad.exact_step_values: records differ",
    ]


def test_timing_environment_reads_final_state_only_after_windows(tmp_path):
    env = hlb.engine_environment(
        "polygrad", tmp_path / "engine", tmp_path / "libpolygrad.so", None,
        hlb.TIMING_CONFIG, capture_forward=False, capture_final_state=True,
    )

    assert "HLB_FORWARD_OUT" not in env
    assert env["HLB_CAPTURE_FORWARD"] == "0"
    assert env["HLB_CAPTURE_PERMUTATIONS"] == "0"
    assert env["HLB_FINAL_STATE_OUT"].endswith("/final_state.npz")


def test_timing_windows_require_exact_jit_lifecycle_and_timer_sum():
    findings, detail = hlb.timing_window_findings(
        "tinygrad", timing_result(), hlb.TIMING_CONFIG
    )

    assert findings == []
    assert len(detail) == 8
    assert detail[1]["jit"]["captured"]
    assert detail[2]["jit"]["replay_count"] == 1


def test_timing_windows_reject_lifecycle_and_unsynchronized_timer_drift():
    result = timing_result()
    result["timing_windows"][3]["phase"] = "first_replay"
    result["timing_windows"][3]["train_jits"][0]["replay_count"] = 0
    result["steps"][3]["run_ms"] = 99.0

    findings, _ = hlb.timing_window_findings(
        "polygrad", result, hlb.TIMING_CONFIG
    )

    assert any("phases" in finding for finding in findings)
    assert any("replay_count" in finding for finding in findings)
    assert any("run != enqueue+readback" in finding for finding in findings)


def test_timing_windows_reject_raw_replay_instrumentation():
    result = timing_result()
    result["exact_step_values"] = [{"unexpected": True}]

    findings, _ = hlb.timing_window_findings(
        "polygrad", result, hlb.TIMING_CONFIG
    )

    assert "timing_windows.polygrad: exact step capture was not disabled" in findings


def test_exact_step_value_validation_fails_closed():
    valid = {
        "shape": [],
        "dtype": "float32",
        "sha256": "e00e5eb9444182f352323374ef4e08ebcb784725fdd4fd612d7730540b3e0c8c",
        "finite": True,
        "values_hex": ["0x1.0000000000000p+0"],
    }

    records = [
        {"step": step, "loss": dict(valid), "lr": [dict(valid), dict(valid)]}
        for step in range(4)
    ]
    assert hlb.exact_step_value_findings("polygrad", "retained", records) == []

    cases = {
        "missing": records[:3],
        "float_step": [
            {**record, "step": float(record["step"])} for record in records
        ],
        "non_list_values": [
            {
                **record,
                "loss": {**record["loss"], "values_hex": "not-a-list"},
            }
            for record in records
        ],
        "nonfinite": [
            {
                **record,
                "loss": {
                    **record["loss"],
                    "finite": False,
                    "values_hex": ["nan"],
                },
            }
            for record in records
        ],
        "malformed": [
            {
                **record,
                "loss": {
                    "shape": "invalid",
                    "dtype": None,
                    "sha256": "x",
                    "finite": "false",
                    "values_hex": ["nan"],
                },
            }
            for record in records
        ],
        "digest": [
            {
                **record,
                "loss": {**record["loss"], "sha256": "0" * 64},
            }
            for record in records
        ],
    }
    for name, candidate in cases.items():
        assert hlb.exact_step_value_findings(
            "polygrad", name, candidate
        ), name


def test_benchmark_mode_cannot_be_duplicated(capsys):
    with pytest.raises(SystemExit) as exc:
        hlb.parse_args(["--mode", "semantic", "--mode", "timing"])
    assert exc.value.code == 2
    assert "--mode may be specified only once" in capsys.readouterr().err


def test_timing_runs_are_counterbalanced_before_ready():
    def run(offset):
        phase_summary = {}
        for phase, count in (
            ("eager", 1), ("capture", 1), ("first_replay", 1),
            ("steady_replay", 5),
        ):
            phase_summary[phase] = {
                "tinygrad_run_ms": [10.0 + offset] * count,
                "polygrad_run_ms": [20.0 + offset] * count,
            }
        return {
            "findings": [],
            "diagnostics": [],
            "phase_summary": phase_summary,
        }

    combined = hlb.combine_timing_runs([
        ("tinygrad_then_polygrad", run(0.0)),
        ("polygrad_then_tinygrad", run(2.0)),
    ])

    assert combined["status"] == "ready"
    assert combined["execution_orders"] == [
        "tinygrad_then_polygrad", "polygrad_then_tinygrad",
    ]
    assert combined["phase_summary"]["steady_replay"]["samples"] == 10
    assert combined["phase_summary"]["steady_replay"]["tinygrad_median_ms"] == 11.0
    assert combined["phase_summary"]["steady_replay"]["polygrad_median_ms"] == 21.0


def test_timing_state_schema_rejects_mutually_truncated_archives(tmp_path):
    names = [f"state.{index}" for index in range(38)]
    arrays = {name: np.asarray([index], dtype=np.float32) for index, name in enumerate(names)}
    initial = tmp_path / "initial.npz"
    final = tmp_path / "final.npz"
    np.savez(initial, **arrays)
    np.savez(final, **arrays)
    result = {
        "state_order": names,
        "initial_state": {name: {} for name in names},
        "final_state": {name: {} for name in names},
    }

    findings = hlb.state_schema_findings(
        "tinygrad", result, initial, initial, final
    )

    assert findings == ["state_schema.authoritative: expected 39 names, got 38"]


def test_timing_state_schema_binds_final_archive_to_state_order(tmp_path):
    names = [f"state.{index}" for index in range(39)]
    arrays = {name: np.asarray([index], dtype=np.float32) for index, name in enumerate(names)}
    initial = tmp_path / "initial.npz"
    final = tmp_path / "final.npz"
    authoritative = tmp_path / "authoritative.npz"
    np.savez(authoritative, **arrays)
    np.savez(initial, **arrays)
    np.savez(final, **{name: arrays[name] for name in names[:-1]})
    summary = {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in arrays.items()
    }
    result = {
        "state_order": names,
        "initial_state": summary,
        "final_state": summary,
    }

    findings = hlb.state_schema_findings(
        "polygrad", result, authoritative, initial, final
    )

    assert findings == [
        "state_schema.polygrad.final_state_archive: keys/order differ"
    ]


def test_state_schema_rejects_mutually_corrupted_shape_and_dtype(tmp_path):
    names = [f"state.{index}" for index in range(39)]
    authoritative_arrays = {
        name: np.asarray([index], dtype=np.float32) for index, name in enumerate(names)
    }
    corrupted_arrays = {
        name: value for name, value in authoritative_arrays.items()
    }
    corrupted_arrays[names[-1]] = np.asarray([[1]], dtype=np.int32)
    authoritative = tmp_path / "authoritative.npz"
    initial = tmp_path / "initial.npz"
    final = tmp_path / "final.npz"
    np.savez(authoritative, **authoritative_arrays)
    np.savez(initial, **corrupted_arrays)
    np.savez(final, **corrupted_arrays)
    summary = {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in corrupted_arrays.items()
    }
    result = {
        "state_order": names,
        "initial_state": summary,
        "final_state": summary,
    }

    findings = hlb.state_schema_findings(
        "polygrad", result, authoritative, initial, final
    )

    assert findings == [
        f"state_schema.polygrad.initial_state.{names[-1]}: reported shape/dtype differ",
        f"state_schema.polygrad.final_state.{names[-1]}: reported shape/dtype differ",
        f"state_schema.polygrad.initial_state_archive.{names[-1]}: shape/dtype differ",
        f"state_schema.polygrad.final_state_archive.{names[-1]}: shape/dtype differ",
    ]


@pytest.mark.parametrize(
    "target",
    ("bench-hlb-cuda-semantic", "bench-hlb-cuda-timing", "bench-hlb-cuda-manifest"),
)
def test_hlb_cuda_make_targets_fail_without_cuda(target):
    result = subprocess.run(
        ["make", "HAS_CUDA=0", target],
        cwd=hlb.ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode != 0
    assert "requires HAS_CUDA=1" in result.stderr
