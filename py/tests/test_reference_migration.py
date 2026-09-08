"""Audit tooling tests use synthetic evidence, never production closure claims."""

import copy
import hashlib
import json
import subprocess
import sys

import pytest

from scripts import reference_migration as migration


def put(root, name, text):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put_catalogue(root, declarations):
    return put(root, "test/tensor_graph_cases.py", declarations +
               "\nimport json\nprint(json.dumps({k: v[0] for k, v in CASES.items()}))\n")


def test_graph_corpus_includes_generated_registrations_without_running_cases(tmp_path):
    put_catalogue(tmp_path, """
def construct():
    raise AssertionError('catalogue must not construct a graph')
CASES = {'initial': ('tensor', construct)}
for name in ('first', 'second'):
    CASES[name] = ('callify', construct)
""")
    assert migration.graph_corpus(tmp_path) == {
        "initial": "tensor", "first": "callify", "second": "callify",
    }


@pytest.mark.parametrize("engine", ["tinygrad", "polygrad"])
def test_live_graph_catalogue_needs_no_frontend_or_site_packages(engine, monkeypatch):
    monkeypatch.setenv("ENGINE", engine)
    script = migration.ROOT / "test/tensor_graph_cases.py"
    result = subprocess.run([sys.executable, "-I", "-S", str(script), "--list"],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    catalogue = json.loads(result.stdout)
    assert catalogue["scan_owner_gradient_split"] == "tensor"
    assert catalogue["disk_copy_callify_output"] == "callify"
    assert catalogue == migration.graph_corpus(migration.ROOT)


@pytest.mark.parametrize("payload", ["[]", "{}", '{"case": 3}', '{"": "tensor"}'])
def test_graph_catalogue_rejects_invalid_metadata(tmp_path, payload):
    put(tmp_path, "test/tensor_graph_cases.py", f"print({payload!r})\n")
    with pytest.raises(ValueError, match="invalid graph case catalogue"):
        migration.graph_corpus(tmp_path)


def test_inventory_includes_unrouted_symbols_and_non_symbol_changes(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    put(old, "ops.py", "import old\ndef cast(): return 1\ndef other(): return 2\n")
    put(new, "ops.py", "import new\ndef cast(): return 3\ndef added(): return 4\n")
    rows = migration.symbol_changes(old, new, ["ops.py"], ["cast"])
    assert {r["symbol"] for r in rows} == {"cast", "other", "added"}
    assert {r["symbol"] for r in rows if r["routed"]} == {"cast"}
    inventory = migration.source_inventory(old, new, ["ops.py"])
    assert {r["id"] for r in inventory} == {
        "ops.py:@file", "ops.py:cast", "ops.py:other", "ops.py:added",
    }
    assert all(r["baseline_sha256"] or r["target_sha256"] for r in inventory)
    put(new, "ops.py", "import another\ndef cast(): return 3\ndef added(): return 4\n")
    assert inventory != migration.source_inventory(old, new, ["ops.py"])


def test_inventory_does_not_silently_drop_parse_errors(tmp_path):
    put(tmp_path, "bad.py", "def broken(\n")
    with pytest.raises(SyntaxError):
        migration.python_symbols(tmp_path / "bad.py")


@pytest.fixture
def gate(tmp_path):
    source_hash = put(tmp_path, "src/owner.c", "void owner(void) {}\n")
    test_hash = put(tmp_path, "test/test_owner.c", "void test_owner(void) {}\n")
    put_catalogue(tmp_path, "CASES = {'owner_case': ('tensor', None)}\n")
    log_hash = put(tmp_path, "temp/run.log", "owner_case PASS\n")
    artifact_hash = put(tmp_path, "build/runner", "synthetic test artifact\n")
    report = {
        "schema_version": 1, "reference_commit": "target",
        "summary": {"pass": 1, "fail": 0, "allowed": 0, "unregistered": 0},
        "cases": {"owner_case": {"stage": "tensor", "passed": True, "findings": []}},
    }
    report_hash = put(tmp_path, "temp/graph.json", json.dumps(report))
    inventory = [{"id": "ops.py:@file", "baseline_sha256": "old", "target_sha256": "new"}]
    row = {
        **inventory[0], "disposition": "equivalent", "reason": "Reviewed complete file diff.",
        "counterparts": [{"path": "src/owner.c", "sha256": source_hash, "symbol": "owner"}],
        "tests": [{"path": "test/test_owner.c", "sha256": test_hash,
                   "check": "native", "case": "owner_case"}],
    }
    audit = {"schema_version": 1, "baseline_commit": "baseline", "target_commit": "target", "rows": [row]}
    put(tmp_path, "audit.json", json.dumps(audit))
    ledger = {
        "baseline": {"commit": "baseline"}, "target": {"commit": "target"},
        "required_checks": ["physical_graph", "native"],
        "rule_groups": [{"id": "matcher", "status": "pass"}], "matcher_aggregates": [],
        "graph_report": {"path": "temp/graph.json", "sha256": report_hash},
        "waves": [{"id": "owner", "source_status": "closed", "implementation_status": "closed",
                   "source_inventory": inventory, "source_audit": "audit.json"}],
    }
    inputs = migration.source_manifest(tmp_path)
    check = {
        "command": "synthetic owner test", "exit_code": 0, "passed": 1, "failed": 0, "skipped": 0,
        "passed_cases": ["owner_case"], "source_sha256": migration.manifest_hash(inputs),
        "log": {"path": "temp/run.log", "sha256": log_hash},
        "artifacts": [{"path": "build/runner", "sha256": artifact_hash}],
    }
    evidence = {"schema_version": 1, "source_inputs": inputs,
                "checks": {name: copy.deepcopy(check) for name in ledger["required_checks"]}}
    evidence["checks"]["physical_graph"]["report_sha256"] = report_hash
    register = {"schema_version": 1, "reference": {"commit": "target"}, "entries": []}
    return tmp_path, ledger, report, evidence, register, audit


def errors(gate):
    root, ledger, report, evidence, register, _ = gate
    return migration.release_errors(ledger, report, evidence, register, root)


def update_audit(gate):
    put(gate[0], "audit.json", json.dumps(gate[5]))


def test_strict_accepts_complete_synthetic_evidence(gate):
    assert errors(gate) == []


@pytest.mark.parametrize("status", ["open", "unclassified", "broad_green", None])
def test_strict_rejects_open_source_audits(gate, status):
    gate[1]["waves"][0]["source_status"] = status
    assert any("source audit is not closed" in e for e in errors(gate))


@pytest.mark.parametrize("field", ["waves", "required_checks", "rule_groups"])
def test_strict_rejects_empty_scope(gate, field):
    gate[1][field] = []
    assert errors(gate)


@pytest.mark.parametrize("mutation", ["missing", "unclassified", "stale", "duplicate", "unregistered"])
def test_strict_rejects_incomplete_or_stale_dispositions(gate, mutation):
    rows = gate[5]["rows"]
    if mutation == "missing":
        rows.clear()
    elif mutation == "unclassified":
        rows[0]["disposition"] = "pending"
    elif mutation == "stale":
        rows[0]["target_sha256"] = "stale"
    elif mutation == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    else:
        rows[0].update(disposition="approved_divergence", divergence_id="not_registered", stage="tensor")
    update_audit(gate)
    assert errors(gate)


@pytest.mark.parametrize("name", ["audit.json", "src/owner.c", "test/test_owner.c", "temp/run.log", "build/runner"])
def test_strict_rejects_missing_or_modified_evidence(gate, name):
    (gate[0] / name).unlink()
    assert errors(gate)


def test_strict_rejects_new_source_files(gate):
    put(gate[0], "src/new.c", "void new_owner(void) {}\n")
    assert any("source inputs" in e for e in errors(gate))


@pytest.mark.parametrize("mutation", ["absent", "failed", "zero_tests", "skipped", "wrong_generation", "missing_case"])
def test_strict_rejects_unexecuted_required_tests(gate, mutation):
    checks = gate[3]["checks"]
    if mutation == "absent":
        del checks["native"]
    elif mutation == "failed":
        checks["native"]["exit_code"] = 1
    elif mutation == "zero_tests":
        checks["native"]["passed"] = 0
    elif mutation == "skipped":
        checks["native"].update(passed=0, skipped=1, passed_cases=[])
    elif mutation == "wrong_generation":
        checks["native"]["source_sha256"] = "stale"
    else:
        checks["native"]["passed_cases"] = ["another_case"]
    assert errors(gate)


@pytest.mark.parametrize("mutation", ["failure", "empty", "unregistered", "forged_allowance", "stale_graph"])
def test_strict_checks_graph_cases_not_just_summary(gate, mutation):
    report = gate[2]
    if mutation == "failure":
        report["cases"]["owner_case"]["passed"] = False
    elif mutation == "empty":
        report["cases"] = {}
    elif mutation in {"unregistered", "forged_allowance"}:
        report["cases"]["owner_case"]["findings"] = [
            {"id": "debt", "allowed": True, "status": "approved"}]
        if mutation == "forged_allowance":
            gate[4]["entries"] = [{"id": "debt", "status": "open_debt", "stages": ["tensor"]}]
    else:
        gate[3]["checks"]["physical_graph"]["report_sha256"] = "stale"
    assert errors(gate)


def test_approved_graph_allowance_requires_exact_stage(gate):
    gate[4]["entries"] = [{"id": "approved", "status": "approved", "stages": ["logical"]}]
    gate[2]["cases"]["owner_case"]["findings"] = [{"id": "approved", "allowed": True, "status": "approved"}]
    assert errors(gate)


def test_strict_accepts_only_registered_stage_scoped_allowance(gate):
    gate[4]["entries"] = [{"id": "approved", "status": "approved", "stages": ["tensor"]}]
    gate[2]["cases"]["owner_case"]["findings"] = [{"id": "approved", "allowed": True, "status": "approved"}]
    gate[2]["summary"]["allowed"] = 1
    assert errors(gate) == []


def test_strict_rejects_dirty_pinned_reference(gate):
    gate[1]["reference_errors"] = ["reference checkout is dirty"]
    assert errors(gate) == ["reference checkout is dirty"]


def test_synthetic_runtime_does_not_satisfy_hardware_requirement(gate):
    gate[1]["waves"][0].update(runtime_status="synthetic_green", required_checks=["multi_gpu"])
    assert "multi_gpu: missing required execution result" in errors(gate)


@pytest.mark.parametrize("mutation", ["duplicate_cases", "boolean_count", "missing_count", "malformed_cases", "stale_artifact"])
def test_strict_rejects_invalid_run_records(gate, mutation):
    check = gate[3]["checks"]["native"]
    if mutation == "duplicate_cases":
        check["passed_cases"] = ["owner_case", "owner_case"]
    elif mutation == "boolean_count":
        check["passed"] = True
    elif mutation == "missing_count":
        del check["skipped"]
    elif mutation == "malformed_cases":
        check["passed_cases"] = [{}]
    else:
        put(gate[0], "build/runner", "changed artifact")
    assert errors(gate)


def test_strict_rejects_missing_review_rationale(gate):
    gate[5]["rows"][0]["reason"] = "  "
    update_audit(gate)
    assert errors(gate)


def test_strict_rejects_stale_counterpart_after_fresh_runtime_run(gate):
    put(gate[0], "src/owner.c", "void changed_owner(void) {}\n")
    inputs = migration.source_manifest(gate[0])
    gate[3]["source_inputs"] = inputs
    for check in gate[3]["checks"].values():
        check["source_sha256"] = migration.manifest_hash(inputs)
    assert any("stale file src/owner.c" in e for e in errors(gate))


@pytest.mark.parametrize("generated", [False, True])
def test_filtered_graph_report_cannot_certify_full_corpus(gate, generated):
    declarations = ("CASES = {'owner_case': ('tensor', None)}\n"
                    "for name in ('omitted',): CASES[name] = ('tensor', None)\n") if generated else (
                    "CASES = {'owner_case': ('tensor', None), 'omitted': ('tensor', None)}\n")
    put_catalogue(gate[0], declarations)
    inputs = migration.source_manifest(gate[0])
    gate[3]["source_inputs"] = inputs
    for check in gate[3]["checks"].values():
        check["source_sha256"] = migration.manifest_hash(inputs)
    assert any("graph corpus" in e for e in errors(gate))


def test_graph_catalogue_process_failure_is_a_gate_error(gate):
    put(gate[0], "test/tensor_graph_cases.py", "raise RuntimeError('broken catalogue')\n")
    assert any("cannot read catalogue" in e for e in errors(gate))


def test_graph_report_cannot_change_stages(gate):
    gate[2]["cases"]["owner_case"]["stage"] = "logical"
    assert any("graph corpus" in e for e in errors(gate))
