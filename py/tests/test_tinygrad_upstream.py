"""Synthetic runner evidence must never be confused with upstream acceptance."""

import copy
import json
import subprocess
import sys

import pytest

from scripts import tinygrad_upstream as upstream


def test_aliases_preserve_module_identity_and_block_reference_fallback(tmp_path):
    provider = tmp_path / "fakeprovider"
    provider.mkdir()
    (provider / "__init__.py").write_text("class Tensor: pass\n")
    (provider / "ops.py").write_text("from . import Tensor\n")
    reference = tmp_path / "tinygrad"
    reference.mkdir()
    (reference / "__init__.py").write_text("raise AssertionError('reference loaded')\n")
    (reference / "missing.py").write_text("raise AssertionError('fallback loaded')\n")
    code = f"""
import importlib, sys
sys.path[:0] = [{str(upstream.ROOT)!r}, {str(tmp_path)!r}]
from scripts.tinygrad_upstream import ProviderAliases
sys.meta_path.insert(0, ProviderAliases('fakeprovider'))
import tinygrad, tinygrad.ops, fakeprovider, fakeprovider.ops
assert tinygrad is fakeprovider
assert tinygrad.ops is fakeprovider.ops
assert tinygrad.Tensor is tinygrad.ops.Tensor
assert tinygrad.ops.__spec__.name == 'fakeprovider.ops'
assert importlib.reload(fakeprovider.ops) is tinygrad.ops
try: import tinygrad.missing
except ModuleNotFoundError as exc: assert exc.name == 'fakeprovider.missing'
else: raise AssertionError('missing provider did not fail closed')
"""
    subprocess.run([sys.executable, "-I", "-c", code], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("engine", ["polygrad", "tinygrad"])
def test_provider_evidence_rejects_foreign_module(tmp_path, monkeypatch, engine):
    from types import ModuleType
    module = ModuleType("tinygrad")
    path = tmp_path / "foreign.py"
    path.write_text("")
    module.__file__ = str(path)
    monkeypatch.setitem(sys.modules, "tinygrad", module)
    _, errors = upstream.provider_evidence(engine, tmp_path / "reference", tmp_path)
    assert any("foreign provider" in err for err in errors)
    if engine == "polygrad":
        assert any("duplicate provider object" in err for err in errors)


@pytest.mark.parametrize("engine", ["polygrad", "tinygrad"])
def test_provider_root_does_not_shadow_upstream_namespace_helpers(tmp_path, engine):
    root, reference = tmp_path / "pg", tmp_path / "tg"
    for path, content in (
        (root / "py/polygrad/__init__.py", "class Tensor: pass\n"),
        (root / "py/extra/__init__.py", "raise AssertionError('foreign extra package loaded')\n"),
        (reference / "tinygrad/__init__.py", "class Tensor: pass\n"),
        (reference / "extra/gradcheck.py", "from tinygrad import Tensor\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    code = f"""
import sys
from pathlib import Path
sys.path.insert(0, {str(upstream.ROOT)!r})
from scripts.tinygrad_upstream import configure_imports
configure_imports({engine!r}, Path({str(reference)!r}), Path({str(root)!r}))
import tinygrad, extra.gradcheck
assert Path(extra.gradcheck.__file__).resolve() == Path({str(reference / 'extra/gradcheck.py')!r})
assert extra.gradcheck.Tensor is tinygrad.Tensor
assert tinygrad.Tensor.__module__ == {engine!r}
"""
    subprocess.run([sys.executable, "-I", "-c", code], check=True, capture_output=True, text=True)


def phases(status="passed", detail="AssertionError: expected 2"):
    return {
        "setup": {"outcome": "passed"},
        "call": {"outcome": status, **({"detail": detail} if status != "passed" else {})},
        "teardown": {"outcome": "passed"},
    }


@pytest.mark.parametrize("status", ["passed", "failed", "skipped"])
def test_phase_outcomes(status):
    assert upstream.outcome(phases(status)) == status


def test_setup_skip_error_teardown_error_and_xfail_are_not_passes():
    assert upstream.outcome({}) == "incomplete"
    assert upstream.outcome({"call": {"outcome": "passed"}}) == "incomplete"
    for phase in ("setup", "teardown"):
        rows = phases()
        rows[phase] = {"outcome": "failed"}
        assert upstream.outcome(rows) == "failed"
    assert upstream.outcome({"setup": {"outcome": "skipped"}, "teardown": {"outcome": "passed"}}) == "skipped"
    rows = phases("skipped")
    rows["call"]["xfail"] = "upstream expected failure"
    assert upstream.outcome(rows) == "xfailed"
    rows["call"]["outcome"] = "passed"
    assert upstream.outcome(rows) == "xpassed"


@pytest.fixture
def report():
    return {"contract": {"pin": "pinned", "tests": "hash", "environment": {"DEV": "CPU"}}, "errors": [],
            "tests": {"polygrad:test/a.py::test_ok": {"status": "passed", "phases": phases()},
                      "polygrad:test/a.py::test_bad": {"status": "failed", "phases": phases("failed")}}}


def test_candidate_requires_manual_nonpass_review(report):
    baseline = upstream.baseline_candidate(report)
    assert upstream.ratchet_errors(report, baseline) == ["unreviewed nonpass: polygrad:test/a.py::test_bad"]
    baseline["tests"]["polygrad:test/a.py::test_bad"]["reason"] = "Synthetic fixture only"
    assert upstream.ratchet_errors(report, baseline) == []


@pytest.mark.parametrize("change", ["pin", "missing", "new", "regression", "skip", "failure_change", "improvement", "collection"])
def test_ratchet_detects_contract_and_outcome_changes(report, change):
    baseline = upstream.baseline_candidate(report)
    baseline["tests"]["polygrad:test/a.py::test_bad"]["reason"] = "Synthetic fixture only"
    if change == "pin":
        report["contract"] = {"pin": "another"}
    elif change == "missing":
        del report["tests"]["polygrad:test/a.py::test_ok"]
    elif change == "new":
        report["tests"]["polygrad:test/a.py::test_new"] = {"status": "passed", "phases": phases()}
    elif change in ("regression", "skip"):
        status = "failed" if change == "regression" else "skipped"
        report["tests"]["polygrad:test/a.py::test_ok"] = {"status": status, "phases": phases(status)}
    elif change == "failure_change":
        report["tests"]["polygrad:test/a.py::test_bad"]["phases"] = phases("failed", "RuntimeError: unrelated defect")
    elif change == "improvement":
        report["tests"]["polygrad:test/a.py::test_bad"] = {"status": "passed", "phases": phases()}
    else:
        report["errors"] = ["collection failed"]
    assert upstream.ratchet_errors(report, baseline)


def test_candidate_cannot_bless_collection_or_crash(report):
    report["errors"] = ["process exit -11"]
    with pytest.raises(ValueError, match="cannot baseline"):
        upstream.baseline_candidate(report)


@pytest.mark.parametrize("change", ["exit", "collection", "lost_test", "empty", "duplicate", "incomplete", "provider"])
def test_execution_admission_rejects_incomplete_evidence(change):
    run = {"exit_code": 0, "collected": ["test_ok"], "tests": {"test_ok": {"status": "passed"}}}
    assert upstream.execution_errors(run) == []
    if change == "exit":
        run["exit_code"] = -11
    elif change == "collection":
        run["collection"] = [{"outcome": "skipped"}]
    elif change == "lost_test":
        run["tests"] = {}
    elif change == "empty":
        run["collected"], run["tests"] = [], {}
    elif change == "duplicate":
        run["collected"] *= 2
    elif change == "incomplete":
        run["tests"]["test_ok"]["status"] = "incomplete"
    else:
        run["errors"] = ["foreign provider"]
    assert upstream.execution_errors(run)


def test_events_preserve_crash_progress(tmp_path):
    from types import SimpleNamespace
    path = tmp_path / "events.jsonl"
    with path.open("w") as events:
        results = upstream.Results(events, tmp_path)
        results.pytest_collection_finish(SimpleNamespace(items=[SimpleNamespace(nodeid="test_ok")]))
        results.pytest_runtest_logstart("test_ok", None)
        results.pytest_runtest_logreport(SimpleNamespace(nodeid="test_ok", when="call", outcome="passed"))
        assert [json.loads(row)["event"] for row in path.read_text().splitlines()] == ["collected", "start", "phase"]
    assert upstream.outcome(results.tests["test_ok"]) == "incomplete"


def test_candidate_delta_does_not_require_same_pin_or_hide_missing_tests(report):
    report["contract"]["reference_commit"] = "old"
    after = copy.deepcopy(report)
    after["contract"]["reference_commit"] = "new"
    del after["tests"]["polygrad:test/a.py::test_ok"]
    after["tests"]["polygrad:test/a.py::test_bad"] = {"status": "passed", "phases": phases()}
    delta = upstream.compare_reports(report, after)
    assert delta["removed"] == ["polygrad:test/a.py::test_ok"]
    assert list(delta["changed"]) == ["polygrad:test/a.py::test_bad"]
    assert not delta["incomplete_evidence"]
    after["errors"] = ["collection failed"]
    assert upstream.compare_reports(report, after)["incomplete_evidence"]


def test_pytest_worker_records_real_setup_errors_skips_and_failures(tmp_path):
    reference = tmp_path / "reference"
    package = reference / "tinygrad"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("class Device: DEFAULT = 'CPU'\n")
    tests = reference / "test"
    tests.mkdir()
    (tests / "test_synthetic.py").write_text(
        "import pytest\n"
        "def test_pass(): assert 1 == 1\n"
        "def test_fail(): assert 1 == 2\n"
        "@pytest.mark.skip(reason='synthetic skip')\ndef test_skip(): pass\n"
        "@pytest.fixture\ndef bad_setup(): raise RuntimeError('synthetic setup failure')\n"
        "def test_error(bad_setup): pass\n"
        "@pytest.mark.xfail(reason='synthetic xfail')\ndef test_xfail(): assert False\n"
    )
    request = {"reference": str(reference), "engine": "tinygrad", "test": "test/test_synthetic.py",
               "result": str(tmp_path / "result.json"), "events": str(tmp_path / "events.jsonl")}
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    process = subprocess.run([sys.executable, "-I", str(upstream.ROOT / "scripts/tinygrad_upstream.py"), "--child", str(path)],
                             cwd=reference, capture_output=True, text=True,
                             env={**__import__('os').environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"})
    assert process.returncode == 1, process.stdout + process.stderr
    result = json.loads((tmp_path / "result.json").read_text())
    assert upstream.execution_errors(result) == []
    assert sorted(t["status"] for t in result["tests"].values()) == ["failed", "failed", "passed", "skipped", "xfailed"]
