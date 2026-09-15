#!/usr/bin/env python3
"""Run upstream tests with isolated, identity-preserving providers.

This is a compatibility diagnostic/ratchet, not a graph-parity allowance. Each
file runs in its own process; collection failures and crashes cannot be baselined.
No upstream implementation may fill a missing Polygrad module or API.
The default lane is unchanged; cpu-ops is an explicit hash-locked test adaptation.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import importlib
import importlib.abc
import importlib.metadata
import importlib.util
import json
import os
import platform
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import sysconfig
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "references/tinygrad_latest"
CPU_OPS_SHA256 = 'b0b8f94a538c555d3ecc094cc7d001de157805b340671b5efa845d54e0cefaf1'
CPU_NN_SHA256 = {
    'test/backend/test_nn.py': 'a64ce69e34511cc179a518c92ce2e91b51ff51fdf43a44fcc22c88c7bda8f23b',
    'test/backend/test_optim.py': '1b6a8537c1c826ee5500cca1134a6a4a61dc302b2a3440393d64fc163370faa6',
}
NN_HELPERS_SHA256 = 'cfe8184a8d5349030a74bfc3322823574ea060713f71685524d6f6b9fbf98366'
ENVIRONMENT = {
    "DEV": "CPU", "CACHELEVEL": "0", "DEBUG": "0",
    "FORWARD_ONLY": "0", "TINY_BACKEND": "0", "SKIP_SLOW_TEST": "0", "IMAGE": "0",
    "DERANDOMIZE_CI": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTHONHASHSEED": "0",
}


def worker_environment(adapter, *, engine=None, logical_policy=None, poly_device=None):
    env = dict(ENVIRONMENT, **({'RUN_SLOW': '1'} if adapter == 'cpu-nn' else {}))
    # The reference remains the pinned CPU oracle; these are explicit Polygrad
    # execution lanes, not ambient settings or edits to upstream test bodies.
    if engine == 'polygrad':
        if logical_policy is not None:
            env['POLY_LOGICAL'] = {'never': '0', 'always': '1', 'until_realize': '2'}[logical_policy]
        if poly_device is not None: env['POLY_DEV'] = poly_device
    return env


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def reference_lock(reference, tests, adapter, inputs):
    """Opt-in Linux CPU control reuse: lock code, dependencies and toolchain.

    Polygrad source/library changes deliberately do not invalidate the oracle.
    This does not cache Polygrad execution or replace a final fresh matrix.
    """
    if sys.platform != 'linux' or not shutil.which('ldd'):
        raise ValueError('reference reuse requires Linux and ldd dependency inspection')
    paths = {Path(__file__).resolve(), Path(sys.executable).resolve()}
    for key in ('stdlib', 'platstdlib', 'purelib', 'platlib'):
        directory = Path(sysconfig.get_path(key))
        paths.update(p.resolve() for p in directory.rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and '.git' not in p.parts)
    tools = {}
    for name in ('clang', 'clang++', 'cc', 'ld', 'ld.lld', 'ar'):
        executable = shutil.which(name)
        tools[name] = str(Path(executable).resolve()) if executable else None
        if executable: paths.add(Path(executable).resolve())
    # Native libraries outside the venv can change while wheel metadata stays
    # unchanged. Include the interpreter/compiler's resolved ELF dependencies.
    for executable in [sys.executable, *filter(None, tools.values())]:
        result = subprocess.run(['ldd', executable], capture_output=True, text=True)
        for match in re.findall(r'(/[^\s()]+)', result.stdout):
            if Path(match).is_file(): paths.add(Path(match).resolve())
    cpu = Path('/proc/cpuinfo').read_text().split('\n\n')[0]
    cpu = '\n'.join(line for line in cpu.splitlines() if line.split(':')[0].strip()
                    in {'vendor_id', 'cpu family', 'model', 'model name', 'stepping', 'microcode', 'flags'})
    return {'schema_version': 1, 'reference': str(reference), 'tests': tests, 'adapter': adapter,
            'upstream_inputs': {p: h for p, h in inputs.items() if Path(p).is_relative_to(reference)},
            'files': {str(p): digest(p) for p in sorted(paths)}, 'tools': tools,
            'environment': {k: os.environ.get(k) for k in ('PATH', 'HOME', 'LANG', 'LD_LIBRARY_PATH')},
            'worker_environment': worker_environment(adapter), 'python': sys.version,
            'platform': platform.platform(), 'cpu': cpu}


def reuse_reference_runs(path, sha256, lock, selections):
    """Accept only original complete reference runs and unchanged raw artifacts."""
    path = Path(path).resolve()
    if digest(path) != sha256: raise ValueError('reference report hash mismatch')
    report = json.loads(path.read_text())
    if report.get('reference_lock') != lock: raise ValueError('reference lock mismatch')
    if report.get('errors'): raise ValueError('reference report has execution errors')
    reused = {}
    for run in report.get('runs', []):
        if run['engine'] != 'tinygrad': continue
        if run.get('reused_from'): raise ValueError('reference reuse must name the original execution')
        if execution_errors(run): raise ValueError('reference execution is incomplete')
        if not run.get('artifacts'): raise ValueError('reference has no locked artifacts')
        for artifact, expected in run['artifacts'].items():
            if not Path(artifact).is_file() or digest(artifact) != expected:
                raise ValueError(f'reference artifact changed: {artifact}')
        if run['selection'] in reused: raise ValueError('duplicate reference selection')
        reused[run['selection']] = dict(run, reused_from={'report': str(path), 'sha256': sha256})
    if set(reused) != set(selections): raise ValueError('reference selection mismatch')
    return reused


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def adapt_cpu_ops(source):
    """Remove only NIR's import/skip guard, whose predicate is false in this lane."""
    if hashlib.sha256(source.encode()).hexdigest() != CPU_OPS_SHA256:
        raise ValueError('CPU ops adapter source lock mismatch; review the upstream change')
    for line in ('from tinygrad.renderer.nir import NIRRenderer',
                 '  @unittest.skipIf(isinstance(Device[Device.DEFAULT].renderer, NIRRenderer), "TODO: broken in LVP")'):
        if source.count(line + '\n') != 1:
            raise ValueError('CPU ops adapter expected one reviewed import/guard')
        source = source.replace(line + '\n', '\n')
    return source


def check_cpu_ops_mode(device, renderer, interface, image):
    if (device, renderer, interface, image) != ('CPU', '', '', 0):
        raise ValueError('CPU ops adapter requires DEV=CPU, no renderer/interface override, IMAGE=0')


def adapt_cpu_nn(source, path, helpers, reference=REFERENCE):
    """Keep test bodies; load private compiler helpers only when a test uses them.

    The two affected schedule tests still execute their original assertions:
    missing compiler APIs fail those tests, rather than preventing collection.
    Capability decorators are extracted verbatim from the locked test helper,
    never from an upstream runtime implementation.
    """
    if (hashlib.sha256(source.encode()).hexdigest() != CPU_NN_SHA256.get(path)
            or hashlib.sha256(helpers.encode()).hexdigest() != NN_HELPERS_SHA256):
        raise ValueError('CPU NN adapter source lock mismatch; review the upstream change')
    names = {'slow', 'not_support_multi_device', 'needs_second_gpu'}
    segments = []
    for node in ast.parse(helpers).body:
        if ((isinstance(node, ast.FunctionDef) and node.name in names)
                or (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'slow' for t in node.targets))):
            segments.append(ast.get_source_segment(helpers, node))
    prelude = 'import os, functools\nfrom tinygrad.helpers import DEV\n' + '\n\n'.join(segments) + '\n'
    for line in ('from test.helpers import check_schedule',
                 'from tinygrad.engine.realize import run_linear',
                 'from test.helpers import not_support_multi_device, needs_second_gpu, slow',
                 'from test.helpers import needs_second_gpu, slow'):
        source = source.replace(line + '\n', '\n')
    # Insert after the original imports, before @slow can be evaluated.
    pos = source.index('\n@slow') if path.endswith('test_nn.py') else source.index('\nnp.random.seed')
    lazy = ('\ndef check_schedule(*args, **kwargs):\n'
            '    import importlib.util\n'
            f'    spec = importlib.util.spec_from_file_location("_upstream_nn_helpers", {str(reference / "test/helpers.py")!r})\n'
            '    module = importlib.util.module_from_spec(spec)\n'
            '    spec.loader.exec_module(module)\n'
            '    return module.check_schedule(*args, **kwargs)\n'
            '\ndef run_linear(*args, **kwargs):\n'
            '    from tinygrad.engine.realize import run_linear as implementation\n'
            '    return implementation(*args, **kwargs)\n')
    return source[:pos] + '\n' + prelude + lazy + source[pos:]


class ProviderAliases(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Alias real modules, including lazy imports, without duplicate FFI owners."""

    def __init__(self, provider="polygrad"):
        self.provider, self.specs = provider, {}

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "tinygrad" and not fullname.startswith("tinygrad."):
            return None
        name = self.provider + fullname[len("tinygrad"):]
        # Raising here is intentional: returning None lets PathFinder execute
        # another copy of a provider file under tinygrad.*, or load upstream.
        module = importlib.import_module(name)
        self.specs[fullname] = module.__spec__
        return importlib.util.spec_from_loader(fullname, self, is_package=hasattr(module, "__path__"))

    def create_module(self, spec):
        return sys.modules[self.provider + spec.name[len("tinygrad"):]]

    def exec_module(self, module):
        # importlib installs the alias spec even when create_module returns an
        # existing module. Keep the provider's reload/relative-import identity.
        alias = module.__spec__.name
        module.__spec__ = self.specs[alias]


def provider_evidence(engine, reference, root=ROOT):
    modules, errors = {}, []
    prefix = root / "py/polygrad" if engine == "polygrad" else reference / "tinygrad"
    for name, module in sorted(sys.modules.items()):
        if name != "tinygrad" and not name.startswith("tinygrad."):
            continue
        origin = getattr(module, "__file__", None)
        actual = "polygrad" + name[len("tinygrad"):]
        if not origin or not Path(origin).resolve().is_relative_to(prefix.resolve()):
            errors.append(f"foreign provider for {name}: {origin}")
        if engine == "polygrad" and module is not sys.modules.get(actual):
            errors.append(f"duplicate provider object for {name}")
        modules[name] = {"path": origin, "sha256": digest(origin) if origin else None}
    if "tinygrad" not in modules:
        errors.append("provider was not imported")
    if engine == "tinygrad" and "polygrad" in sys.modules:
        errors.append("Polygrad imported in reference process")
    return modules, errors


def outcome(phases):
    if not phases or "teardown" not in phases:
        return "incomplete"
    if any(p["outcome"] == "failed" for p in phases.values()):
        return "failed"
    if any(p.get("xfail") for p in phases.values()):
        return "xpassed" if phases.get("call", {}).get("outcome") == "passed" else "xfailed"
    if any(p["outcome"] == "skipped" for p in phases.values()):
        return "skipped"
    return "passed" if phases.get("call", {}).get("outcome") == "passed" else "incomplete"


class Results:
    def __init__(self, events, reference, adapted=None):
        self.events, self.reference = events, reference
        self.adapted = adapted
        self.tests, self.collection, self.collected = {}, [], []

    def emit(self, event, **data):
        self.events.write(json.dumps({"event": event, **data}) + "\n")
        self.events.flush()

    def pytest_collectreport(self, report):
        if report.outcome != "passed":
            row = {"nodeid": report.nodeid, "outcome": report.outcome, "detail": str(report.longrepr)}
            self.collection.append(row)
            self.emit("collection", **row)

    def pytest_collection_finish(self, session):
        self.collected = [item.nodeid for item in session.items]
        self.emit("collected", nodeids=self.collected)

    def pytest_runtest_logstart(self, nodeid, location):
        self.emit("start", nodeid=nodeid)

    def pytest_runtest_logreport(self, report):
        row = {"outcome": report.outcome}
        if hasattr(report, "wasxfail"):
            row["xfail"] = str(report.wasxfail)
        if report.outcome != "passed":
            crash = getattr(report.longrepr, "reprcrash", None)
            text = f"{crash.path}:{crash.lineno}: {crash.message}" if crash else str(report.longrepr)
            if self.adapted:
                text = text.replace(str(self.adapted), '<adapted>')
            for path, label in ((self.reference, "<reference>"), (ROOT, "<polygrad>")):
                text = text.replace(str(path), label)
            row["detail"] = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", text)
        self.tests.setdefault(report.nodeid, {})[report.when] = row
        self.emit("phase", nodeid=report.nodeid, when=report.when, **row)


def configure_imports(engine, reference, root=ROOT):
    # -P/-s and the sanitized environment exclude cwd/PYTHONPATH/user site.
    sys.path.insert(0, str(reference))
    if engine == "polygrad":
        if any(n == "tinygrad" or n.startswith("tinygrad.") for n in sys.modules):
            raise RuntimeError("upstream provider already loaded before alias installation")
        sys.meta_path.insert(0, ProviderAliases())
        # Load only the provider package. Adding its parent to sys.path lets
        # py/extra shadow upstream's namespace-package test helpers, even in
        # the reference control. Submodules resolve through this package path.
        spec = importlib.util.spec_from_file_location("polygrad", root / "py/polygrad/__init__.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["polygrad"] = module
        spec.loader.exec_module(module)


def child(request):
    reference = Path(request["reference"]).resolve()
    result = {"collection": [], "tests": {}, "errors": [], "collected": []}
    try:
        import pytest
        import random
        import numpy as np
        random.seed(0)
        np.random.seed(0)
        configure_imports(request["engine"], reference)
        import tinygrad
        result["device"] = tinygrad.Device.DEFAULT
        result["versions"] = {name: importlib.metadata.version(name) for name in ("pytest", "numpy")}
        for name in ("torch", "hypothesis"):
            try:
                result["versions"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                result["versions"][name] = None
        if request["engine"] == "polygrad":
            from polygrad import _ffi
            loaded = Path(_ffi._lib._name).resolve()
            if loaded != Path(request["library"]).resolve():
                raise RuntimeError(f"wrong Polygrad library: {loaded}")
            result["library"] = {"path": str(loaded), "sha256": digest(loaded)}
        test, pytest_root, adapted = request['test'], reference, None
        if request.get('adapter') in ('cpu-ops', 'cpu-nn'):
            from tinygrad.helpers import DEV, IMAGE
            check_cpu_ops_mode(tinygrad.Device.DEFAULT, DEV.renderer, DEV.interface, IMAGE.value)
            if request['engine'] == 'tinygrad' and request['adapter'] == 'cpu-ops':
                from tinygrad.renderer.nir import NIRRenderer
                if isinstance(tinygrad.Device["CPU"].renderer, NIRRenderer):
                    raise ValueError('CPU ops adapter cannot remove an active NIR skip')
            original, *node = test.split('::')
            if request['adapter'] == 'cpu-ops' and original != 'test/backend/test_ops.py':
                raise ValueError('CPU ops adapter only supports test/backend/test_ops.py')
            adapted = Path(request['result']).parent / ('adapted-' + request['engine'])
            dest = adapted / original
            dest.parent.mkdir(parents=True, exist_ok=True)
            source = (reference / original).read_text(encoding='utf-8')
            modified = (adapt_cpu_ops(source) if request['adapter'] == 'cpu-ops' else
                        adapt_cpu_nn(source, original, (reference / 'test/helpers.py').read_text(encoding='utf-8'), reference))
            dest.write_text(modified, encoding='utf-8')
            result['adaptation'] = {'id':request['adapter'], 'original_sha256':digest(reference / original),
                                    'path':str(dest), 'sha256':digest(dest),
                                    'renderer_type':type(tinygrad.Device["CPU"].renderer).__name__}
            test, pytest_root = str(dest) + ''.join('::' + n for n in node), adapted
        with Path(request["events"]).open("w") as events:
            plugin = Results(events, reference, adapted)
            code = pytest.main([
                "-c", os.devnull, "--rootdir", str(pytest_root), "--noconftest",
                "--import-mode=importlib", "-q", "-ra", "--tb=short", "-p", "no:cacheprovider",
                test,
            ], plugins=[plugin])
        if adapted and digest(dest) != result['adaptation']['sha256']:
            result['errors'].append('adapted test source changed during execution')
        result.update(exit_code=int(code), collection=plugin.collection, collected=plugin.collected)
        result["tests"] = {node: {"status": outcome(phases), "phases": phases}
                           for node, phases in plugin.tests.items()}
    except Exception:
        result["errors"].append(traceback.format_exc())
        result["exit_code"] = 3
    result["modules"], errors = provider_evidence(request["engine"], reference)
    result["errors"].extend(errors)
    write_json(request["result"], result)
    return result["exit_code"]


def execution_errors(run):
    errors = list(run.get("errors", []))
    if run.get("collection"):
        errors.append("collection failed or skipped")
    if run.get("exit_code") not in (0, 1):
        errors.append(f"incomplete process: exit {run.get('exit_code')}")
    nodes = run.get("collected", [])
    if not nodes or len(nodes) != len(set(nodes)):
        errors.append("empty or duplicate collection")
    tests = run.get("tests", {})
    if set(nodes) != set(tests):
        errors.append("collected/executed tests differ")
    if any(t["status"] == "incomplete" for t in tests.values()):
        errors.append("incomplete test phases")
    if run.get("exit_code") == 1 and not any(t["status"] == "failed" for t in tests.values()):
        errors.append("pytest failure without failed tests")
    if run.get("exit_code") == 0 and any(t["status"] == "failed" for t in tests.values()):
        errors.append("pytest success with failed tests")
    return errors


def signature(test):
    return {"status": test["status"], "nonpassing_phases": {
        k: {**v, **({'detail':re.sub(r'(?m)^([^\n]+\.py):\d+:', r'\1:LINE:', v['detail'])}
                    if 'detail' in v else {})}
        for k, v in test["phases"].items() if v["outcome"] != "passed" or v.get("xfail")
    }}


def baseline_candidate(report):
    if report["errors"]:
        raise ValueError("cannot baseline collection, provider, freshness, or execution errors")
    return {"schema_version": 1, "contract": report["contract"], "tests": {
        key: {"expected": signature(test), "reason": "" if test["status"] == "passed" else "UNREVIEWED"}
        for key, test in report["tests"].items()
    }}


def ratchet_errors(report, baseline):
    errors = list(report["errors"])
    if baseline.get("schema_version") != 1 or baseline.get("contract") != report["contract"]:
        errors.append("baseline contract changed: review pin, suite, provider, environment and dependencies")
    old, new = baseline.get("tests", {}), report["tests"]
    for key in sorted(set(old) - set(new)):
        errors.append(f"missing test: {key}")
    for key, test in new.items():
        if key not in old:
            errors.append(f"unreviewed new test: {key}")
            continue
        expected = old[key]["expected"]
        if expected["status"] != "passed" and old[key].get("reason", "").strip() in ("", "UNREVIEWED"):
            errors.append(f"unreviewed nonpass: {key}")
        if signature(test) != expected:
            # Improvements require explicit promotion too: otherwise an old
            # expected failure could later return unnoticed.
            errors.append(f"changed outcome (review and promote improvements): {key}")
    return errors


def compare_reports(before, after):
    """Candidate-ref triage only; never change the accepted pin or baseline."""
    old, new = before["tests"], after["tests"]
    return {
        "before_commit": before["contract"]["reference_commit"],
        "after_commit": after["contract"]["reference_commit"],
        "incomplete_evidence": bool(before["errors"] or after["errors"]),
        "added": sorted(set(new) - set(old)), "removed": sorted(set(old) - set(new)),
        "changed": {k: {"before": signature(old[k]), "after": signature(new[k])}
                    for k in sorted(set(old) & set(new)) if signature(old[k]) != signature(new[k])},
    }


def unaccepted_outcomes(tests, *, allow_reference_skips=False):
    rejected = []
    for key, test in tests.items():
        if test['status'] == 'passed': continue
        engine, _, node = key.partition(':')
        other = tests.get(('tinygrad' if engine == 'polygrad' else 'polygrad') + ':' + node)
        # This permits an unchanged upstream skip, never two matching failures
        # or a provider-specific skip that hides a missing implementation.
        if (allow_reference_skips and engine in ('tinygrad', 'polygrad') and test['status'] == 'skipped'
                and other is not None and signature(test) == signature(other)):
            continue
        rejected.append(key)
    return rejected


def source_inputs(reference, tests, library):
    paths = {Path(__file__).resolve(), ROOT / "Makefile", library}
    for directory, suffixes in ((ROOT / "src", (".c", ".h")), (ROOT / "py/polygrad", (".py",)),
                                (reference / "tinygrad", (".py",)), (reference / "test", (".py",)),
                                (reference / "extra", (".py",))):
        paths.update(p for p in directory.rglob("*") if p.suffix in suffixes and p.is_file())
    paths.update(reference / test.split("::")[0] for test in tests)
    return {str(p.resolve()): digest(p) for p in sorted(paths)}


def run_one(output, engine, test, reference, library, timeout, adapter=None, logical_policy=None, poly_device=None):
    key = hashlib.sha256(test.encode()).hexdigest()[:12]
    stem = output / f"{engine}-{key}"
    request = {"engine": engine, "test": test, "reference": str(reference), "library": str(library),
               "result": str(stem.with_suffix(".json")), "events": str(stem.with_suffix(".jsonl")), 'adapter':adapter}
    write_json(stem.with_suffix(".request.json"), request)
    env = {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "LD_LIBRARY_PATH") if k in os.environ}
    env.update(worker_environment(adapter, engine=engine, logical_policy=logical_policy, poly_device=poly_device),
               POLY_LIB=str(library), POLY_TMPDIR=str(output / "cc_tmp"),
               TMPDIR=str(output / "cc_tmp"), XDG_CACHE_HOME=str(output / "cache"))
    command = [sys.executable, "-P", "-s", str(Path(__file__).resolve()), "--child", str(stem.with_suffix(".request.json"))]
    started = time.monotonic()
    with stem.with_suffix(".log").open("w") as log:
        process = subprocess.Popen(command, cwd=reference, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Include compiler children; do not leave a timed-out JIT competing
            # with the next file for temporary shared-library paths.
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            code = 124
    path = Path(request["result"])
    result = json.loads(path.read_text()) if path.exists() else {"errors": ["worker produced no final report"]}
    if result.get("exit_code", code) != code:
        result.setdefault("errors", []).append("worker and process exit codes differ")
    result.update(engine=engine, selection=test, exit_code=code, command=command,
                  duration_seconds=time.monotonic() - started, log=str(stem.with_suffix(".log")),
                  events=request["events"])
    artifacts = [path, Path(request['events']), stem.with_suffix('.log'), stem.with_suffix('.request.json')]
    if result.get('adaptation'): artifacts.append(Path(result['adaptation']['path']))
    result['artifacts'] = {str(p): digest(p) for p in artifacts if p.is_file()}
    print(f"{engine} {test}: {dict(Counter(t['status'] for t in result.get('tests', {}).values()))} "
          f"exit={code} collection={len(result.get('collection', []))}", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="append", help="reference-relative file or exact pytest nodeid; repeatable")
    parser.add_argument("--engine", choices=("polygrad", "tinygrad", "both"), default="both")
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--library", type=Path, default=ROOT / "build/libpolygrad.so")
    parser.add_argument("--output", type=Path, default=ROOT / "temp/tinygrad_upstream")
    parser.add_argument("--timeout", type=float, default=1800, help="per-file process-tree limit in seconds")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--compare-with", type=Path, help="attach outcome delta against a previous run; not acceptance")
    parser.add_argument("--write-baseline", type=Path, help="write new candidate only; nonpasses require reviewed reasons")
    parser.add_argument('--adapter', choices=['cpu-ops', 'cpu-nn'], help='explicit CPU-only test adaptation; default tests are unchanged')
    parser.add_argument('--logical-policy', choices=['always', 'until_realize', 'never'], help='explicit Polygrad logical-retention lane')
    parser.add_argument('--poly-device', choices=['cpu', 'interp'], help='Polygrad backend; Tinygrad control stays on CPU')
    parser.add_argument('--allow-reference-skips', action='store_true', help='accept only identical skips in both engines; failures remain errors')
    parser.add_argument('--record-reference-lock', action='store_true', help='content-lock this CPU reference execution for later reuse')
    parser.add_argument('--reuse-reference', type=Path, help='reuse only the original locked Tinygrad report; Polygrad always executes')
    parser.add_argument('--reference-sha256', help='required digest of --reuse-reference')
    parser.add_argument("--child", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child:
        return child(json.loads(args.child.read_text()))
    if bool(args.reuse_reference) != bool(args.reference_sha256):
        parser.error('--reuse-reference and --reference-sha256 must be provided together')
    if args.reuse_reference and args.engine != 'both':
        parser.error('reference reuse requires --engine both; Polygrad must execute')
    reference, library, output = args.reference.resolve(), args.library.resolve(), args.output.resolve()
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    tests = args.test or (list(baseline["contract"]["test_sha256"]) if baseline else ["test/backend/test_ops.py"])
    if len(tests) != len(set(tests)) or args.timeout <= 0:
        parser.error("duplicate selections or invalid timeout")
    for test in tests:
        path = (reference / test.split("::")[0]).resolve()
        if not path.is_relative_to(reference / "test") or not path.is_file():
            parser.error(f"not an upstream test: {test}")
        if args.adapter == 'cpu-ops' and test.split('::')[0] != 'test/backend/test_ops.py':
            parser.error('CPU ops adapter only supports test/backend/test_ops.py')
        if args.adapter == 'cpu-nn' and test.split('::')[0] not in CPU_NN_SHA256:
            parser.error('CPU NN adapter only supports test_nn.py and test_optim.py')
    if args.write_baseline and args.write_baseline.exists():
        parser.error("refusing to overwrite a baseline; write a separate review candidate")
    output.mkdir(parents=True, exist_ok=True)
    # A run directory is an evidence unit, never a blend of old/new workers.
    if any(output.iterdir()):
        parser.error("output directory must be empty; use a fresh evidence directory")
    (output / "cc_tmp").mkdir()
    (output / "cache").mkdir()
    inputs = source_inputs(reference, tests, library)
    commit = subprocess.check_output(["git", "-C", str(reference), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(reference), "status", "--porcelain", "--untracked-files=no"], text=True)
    if dirty:
        parser.error("reference has tracked edits; choose a clean pinned/candidate checkout")
    try:
        lock = reference_lock(reference, tests, args.adapter, inputs) if args.record_reference_lock or args.reuse_reference else None
        reused = reuse_reference_runs(args.reuse_reference, args.reference_sha256, lock, tests) if args.reuse_reference else {}
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    runs = []
    for test in tests:
        for engine in (("tinygrad", "polygrad") if args.engine == "both" else (args.engine,)):
            if engine == 'tinygrad' and test in reused:
                print(f'tinygrad {test}: reused original control {args.reuse_reference}', flush=True)
                runs.append(reused[test])
            else:
                runs.append(run_one(output, engine, test, reference, library, args.timeout, args.adapter,
                                    args.logical_policy, args.poly_device))
    errors = [f"{r['engine']} {r['selection']}: {err}" for r in runs for err in execution_errors(r)]
    if inputs != source_inputs(reference, tests, library):
        errors.append("source/artifact inputs changed during execution")
    if lock is not None and lock != reference_lock(reference, tests, args.adapter, inputs):
        errors.append('reference environment changed during execution')
    contract = {"reference_commit": commit, "test_sha256": {t: digest(reference / t.split('::')[0]) for t in tests},
                "upstream_sha256": hashlib.sha256(json.dumps({p.replace(str(reference), '<reference>'): h for p, h in inputs.items()
                    if Path(p).is_relative_to(reference)}, sort_keys=True).encode()).hexdigest(),
                "engines": args.engine, "environment": worker_environment(args.adapter), "python": sys.version,
                "runner_sha256": digest(__file__), "versions": [r.get("versions") for r in runs],
                "devices": [r.get("device") for r in runs], 'adapter':args.adapter,
                'failure_signature':'source-path-and-message-v2'}
    if args.logical_policy is not None or args.poly_device is not None:
        contract['polygrad_environment'] = worker_environment(args.adapter, engine='polygrad',
            logical_policy=args.logical_policy, poly_device=args.poly_device)
    if args.allow_reference_skips: contract['allow_reference_skips'] = True
    cases = {}
    for run in runs:
        for node, test in run.get("tests", {}).items():
            key = run["engine"] + ":" + node
            if key in cases:
                errors.append(f"overlapping selection: {key}")
            cases[key] = test
    if args.engine == "both":
        for tg, pg in zip(runs[::2], runs[1::2]):
            if tg.get("collected") != pg.get("collected"):
                errors.append(f"provider collection mismatch: {tg['selection']}")
    report = {"schema_version": 1, "contract": contract, "source_inputs": inputs, "runs": runs,
              "tests": cases, "errors": errors, "summary": dict(Counter(t['status'] for t in cases.values()))}
    if lock is not None: report['reference_lock'] = lock
    report["summary_by_engine"] = {engine: dict(Counter(t["status"] for key, t in cases.items() if key.startswith(engine + ":")))
                                   for engine in sorted({r["engine"] for r in runs})}
    if args.compare_with:
        report["delta"] = compare_reports(json.loads(args.compare_with.read_text()), report)
    if args.baseline:
        report["ratchet_errors"] = ratchet_errors(report, baseline)
    write_json(output / "report.json", report)
    if args.write_baseline:
        try:
            candidate = baseline_candidate(report)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        candidate["evidence"] = {"report": str(output / "report.json"), "sha256": digest(output / "report.json")}
        write_json(args.write_baseline, candidate)
    print(json.dumps({"summary": report["summary_by_engine"], "errors": errors, "ratchet_errors": report.get("ratchet_errors", []),
                      "report": str(output / "report.json")}, indent=2))
    if args.baseline:
        return int(bool(report["ratchet_errors"]))
    return int(bool(errors) or bool(unaccepted_outcomes(cases, allow_reference_skips=args.allow_reference_skips)))


if __name__ == "__main__":
    raise SystemExit(main())
