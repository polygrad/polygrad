#!/usr/bin/env python3
"""Run the CUDA/Wasm release matrix serially, retaining every gate's result."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.reference_migration import file_hash, manifest_hash, source_manifest

SCOPE = json.loads((ROOT / 'test/fixtures/release_050_scope.json').read_text(encoding='utf-8'))
EXCLUSIONS = SCOPE['excluded_capabilities']


def candidate_version(root):
    match = re.search(r'^version\s*=\s*"([^"]+)"', (root / 'py/pyproject.toml').read_text(), re.M)
    if not match:
        raise ValueError('missing Python package version')
    package = json.loads((root / 'js/package.json').read_text())
    lock = json.loads((root / 'js/package-lock.json').read_text())
    versions = [match.group(1), package['version'], lock['version'], lock['packages']['']['version']]
    if len(set(versions)) != 1:
        raise ValueError(f'package version mismatch: {versions}')
    return versions[0]


def release_gates():
    # Expand aggregates once. Do not nest test-all/verify or the alias
    # test-parity-opt: those would repeat whole suites. Analyze precedes HLB
    # because its temporary .plist files affect HLB's checkout source lock.
    targets = '''
        test-release-preflight format-check verify-source-mirrors test-headers test-release-py-performance test-analyze-reviewed
        test test-x86 test-interp test-cuda test-qwen3
        test-harness-skip-accounting test-release-gates
        test-py test-py-x86 test-hf-e2e
        test-js-native-cpu test-js-native-x86 test-js-native-interp
        test-js-native-cuda test-js-native-gc
        test-bigint-wasm test-runtime-wasm test-autograd-wasm test-nn-wasm
        test-indexing-wasm test-materialization-wasm
        test-js-wasm test-js-package test-browser test-browser-qwen3
        test-model-interchange test-py-sdist-install test-py-min-install test-js-package-install
        test-parity test-parity-ir test-parity-ir-opt test-parity-cuda
        test-parity-graph test-release-op-census
        test-compat-tinygrad-upstream-ratchet test-compat-tinygrad-ops test-compat-tinygrad-nn
        test-compat-tinygrad-policy
        test-compat-tinygrad-tier1 test-compat-tinygrad-convnext
        test-reference-parity fuzz-smoke test-symbolic-z3-supported
        test-release-c-performance bench-hlb-cuda-semantic bench-hlb-cuda-timing
    '''.split()
    gates = [dict(target=target, variables={}) for target in targets]
    for gate in gates:
        if gate['target'] == 'test-release-py-performance':
            gate['variables']['PY_PERF_OUTPUT'] = '{output}/python-performance/report.json'
        elif gate['target'] == 'test-release-c-performance':
            gate['variables']['C_PERF_OUTPUT'] = '{output}/c-performance/report.json'
        elif gate['target'] in ('bench-hlb-cuda-semantic', 'bench-hlb-cuda-timing'):
            gate['variables']['HLB_BENCH_DIR'] = '{output}/hlb'
        elif gate['target'] == 'test-compat-tinygrad-upstream-ratchet':
            gate['variables']['UPSTREAM_COMPAT_DIR'] = '{output}/upstream-nine'
        elif gate['target'] == 'test-compat-tinygrad-ops':
            # This 427-case file exceeded 1800s after 405 completed cases. Keep
            # assertions intact and allow a longer bounded process lifetime.
            gate['variables'].update(
                UPSTREAM_COMPAT_DIR='{output}/upstream-ops',
                UPSTREAM_COMPAT_ARGS='--timeout 3600 --baseline test/fixtures/tinygrad_upstream_ops_cpu_014_baseline.json')
        elif gate['target'] == 'test-compat-tinygrad-nn':
            gate['variables'].update(
                UPSTREAM_COMPAT_DIR='{output}/upstream-nn',
                UPSTREAM_COMPAT_ARGS='--baseline test/fixtures/tinygrad_upstream_nn_cpu_014_baseline.json')
        elif gate['target'] == 'test-compat-tinygrad-policy':
            gate['variables'].update(UPSTREAM_COMPAT_DIR='{output}/upstream-policy',
                                     UPSTREAM_POLICY_TESTS='test/backend/test_setitem.py test/backend/test_tensor.py test/null/test_indexing.py')
    return gates


def preflight(variables):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(errors='backslashreplace')
    try:
        print(f'Candidate package version: {candidate_version(ROOT)}')
    except (OSError, ValueError, KeyError) as exc:
        print(f'Candidate metadata: {exc}')
        return 1
    # The core may build with GCC while generated CPU kernels require __fp16.
    # Exercise that contract, not the compiler executable's spelling.
    checks = [('CC', [variables.get('CC', 'clang'), '-fsyntax-only', '-x', 'c', '-'],
               'void kernel(__fp16 *out, const __fp16 *in) { out[0] = in[0]; }\n')]
    version_check = (
        'import platform, sys; print(sys.executable, platform.python_implementation(), '
        'platform.python_version()); '
        'sys.exit(not (sys.implementation.name == "cpython" and sys.version_info[:2] == (3, 11)))'
    )
    for name in ('PYTHON', 'PARITY_PY'):
        checks.append((name, shlex.split(variables.get(name, sys.executable)) + ['-c', version_check], None))
    checks.append(('PYTHON_MIN', shlex.split(variables.get('PYTHON_MIN', 'python3.9')) + ['-c',
                   'import sys, venv; print(sys.executable, sys.version); '
                   'sys.exit(sys.version_info[:2] != (3, 9))'], None))
    # Fail before the matrix when isolated-package/proof tooling is missing.
    checks.append(('PYTHON build', shlex.split(variables.get('PYTHON', sys.executable)) +
                   ['-c', 'import build'], None))
    checks.append(('PARITY_PY z3', shlex.split(variables.get('PARITY_PY', sys.executable)) +
                   ['-c', 'import z3'], None))
    for name, default in (('CLANG_FORMAT', 'clang-format-14'), ('ANALYZER_CC', 'clang-14')):
        checks.append((name, shlex.split(variables.get(name, default)) + ['--version'], None))
    failed = False
    for name, command, source in checks:
        print(f'{name}: {shlex.join(command)}', flush=True)
        try:
            result = subprocess.run(command, input=source, encoding='utf-8', errors='replace', capture_output=True, timeout=30)
            print(result.stdout + result.stderr, end='')
            if result.returncode:
                failed = True
                print(f'{name}: preflight failed (exit {result.returncode})')
            if name == 'CLANG_FORMAT' and not re.search(r'\bversion 14\.', result.stdout):
                failed = True
                print('CLANG_FORMAT: release formatting requires version 14')
            if name == 'ANALYZER_CC':
                review = json.loads((ROOT / 'test/fixtures/analyzer_reviews.json').read_text(encoding='utf-8'))
                if result.stdout != review['context']['clang']:
                    failed = True
                    print('ANALYZER_CC: compiler differs from the reviewed analyzer context')
        except (OSError, subprocess.TimeoutExpired) as exc:
            failed = True
            print(f'{name}: {exc}')
    if failed:
        print('Use a compiler accepting CPU __fp16 kernels and CPython 3.11 for PYTHON/PARITY_PY. '
              'PYTHON_MIN must be Python 3.9; HF_PYTHON is independent.')
    return int(failed)


def release_environment(root):
    env = os.environ.copy()
    # A local test selector or make -i/-j must not turn the full gate into a
    # filtered, error-ignoring or concurrent run. Tool/fixture choices are
    # passed as explicit Make assignments, not inherited MAKEFLAGS.
    for key in ('MAKEFLAGS', 'MFLAGS', 'MAKEOVERRIDES', 'PYTHONHOME', 'PYTHONPATH',
                'POLY_TEST_FILTER', 'POLY_LIB', 'POLY_CORE', 'COMPAT_CASES',
                'POLY_REQUIRE_HF', 'NODE_OPTIONS', 'NODE_PATH'):
        env.pop(key, None)
    env.update(DEV='CPU', POLY_DEV='cpu', PYTEST_ADDOPTS='',
               PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTHONUTF8='1', POLY_BROWSER_DEVICES='auto,interp,webgpu',
               POLY_BROWSER_SKIP_UNAVAILABLE='0')
    for key, directory in (('POLY_TMPDIR', 'cc_tmp'), ('TMPDIR', 'cc_tmp'),
                           ('EM_CACHE', 'emscripten-cache'), ('XDG_CACHE_HOME', 'xdg_cache'),
                           ('XDG_RUNTIME_DIR', 'xdg_runtime')):
        path = root / 'temp' / directory
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        env.setdefault(key, str(path))
    return env


def save_summary(output, report):
    # A killed process leaves the last complete checkpoint, never half JSON.
    pending = output / 'summary.json.tmp'
    pending.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    pending.replace(output / 'summary.json')


def machine_conditions():
    # Load is context, not proof of timing validity: affinity and other users'
    # work can make a machine noisy even below its logical CPU count.
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else list(range(os.cpu_count() or 1))
    load = list(os.getloadavg()) if hasattr(os, 'getloadavg') else None
    return dict(load_average=load, affinity=cpus, cpu_count=os.cpu_count(),
                busy=bool(load and load[0] > len(cpus)))


def stop_process(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    # make may exit before a child that ignores TERM. Kill the surviving
    # group even when the immediate parent has already been reaped.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_release(root, output, make, gates, variables):
    output.mkdir(parents=True, exist_ok=False)
    env = release_environment(root)
    common = dict(variables, HAS_CUDA='1', HAS_HIP='0',
                  ASAN_OPTIONS='detect_leaks=1:protect_shadow_gap=0',
                  UBSAN_OPTIONS='print_stacktrace=1:halt_on_error=1',
                  FUZZ_ASAN_OPTIONS='symbolize=0:detect_leaks=1:protect_shadow_gap=0',
                  FUZZ_SMOKE_ARGS='-runs=256 -max_len=512 -timeout=5 -seed=20260913',
                  UPSTREAM_COMPAT_ARGS='', UPSTREAM_COMPAT_TESTS='',
                  UPSTREAM_COMPAT_BASELINE='test/fixtures/tinygrad_upstream_014_baseline.json',
                  GRAPH_PARITY_CASE_ARGS='',
                  GRAPH_PARITY_DIR=str(output / 'graphs'),
                  OP_PARITY_DIR=str(output / 'op-census'),
                  COMPAT_TIER1_DIR=str(output / 'tier1'),
                  COMPAT_CONVNEXT_DIR=str(output / 'convnext'),
                  ANALYZER_REVIEW_DIR=str(output / 'analyzer'),
                  REFERENCE_RELEASE_DIR=str(output / 'reference-parity'),
                  FUZZ_WORK_DIR=str(output / 'fuzz-corpus'),
                  BENCH_SMOKE_JSON=str(output / 'smoke.json'))
    inputs = source_manifest(root)
    version = candidate_version(ROOT)
    report = dict(schema_version=2, status='running', acceptance_scope=f'polygrad-{version}-supported',
                  candidate_version=version,
                  source_inputs=inputs, source_sha256=manifest_hash(inputs),
                  deferred_certification=SCOPE['deferred_certification'],
                  started_at=datetime.now(timezone.utc).isoformat(),
                  exclusions=EXCLUSIONS, gates=[])
    for index, gate in enumerate(gates, 1):
        assignments = dict(common, **{key: value.replace('{output}', str(output))
                                      for key, value in gate['variables'].items()})
        command = make + ['--no-print-directory', '-j1', gate['target']]
        command += [f'{key}={value}' for key, value in assignments.items()]
        report['gates'].append(dict(target=gate['target'], command=command, status='not_run',
                                    exit_code=None, duration_seconds=None,
                                    log=f'{index:02d}-{gate["target"]}.log'))
    save_summary(output, report)
    print(f'Release evidence: {output}', flush=True)
    interrupted = False
    for row in report['gates']:
        start = time.monotonic()
        process = None
        row['status'] = 'running'
        row['machine_conditions'] = dict(before=machine_conditions())
        save_summary(output, report)
        with (output / row['log']).open('w', encoding='utf-8') as log:
            log.write(shlex.join(row['command']) + '\n')
            log.flush()
            try:
                # Isolate the entire make/test tree so interruption cannot
                # leave GPU/browser/compiler children running in the background.
                process = subprocess.Popen(row['command'], cwd=root, env=env,
                                           stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                row['exit_code'] = process.wait()
            except OSError as exc:
                log.write(f'{exc}\n')
                row['exit_code'] = 127
            except KeyboardInterrupt:
                interrupted = True
                if process is not None:
                    stop_process(process)
                row['exit_code'] = 130
        row['duration_seconds'] = round(time.monotonic() - start, 3)
        row['machine_conditions']['after'] = machine_conditions()
        row['log_sha256'] = file_hash(output / row['log'])
        row['status'] = 'interrupted' if interrupted else ('passed' if row['exit_code'] == 0 else 'failed')
        save_summary(output, report)
        if interrupted or (row['target'] == 'test-release-preflight' and row['exit_code'] != 0):
            break
    report['source_unchanged'] = source_manifest(root) == inputs
    report['errors'] = [] if report['source_unchanged'] else ['source inputs changed during release execution']
    report['artifacts'] = {name: file_hash(root / name) for name in (
        'build/polygrad_test', 'build/libpolygrad.so', 'build/polygrad_parity_runner',
        'build/polygrad_parity_runner_cuda', 'build/core.sync.js', 'build/core.async.js',
        'js/build/Release/polygrad_napi.node', 'build/bench_smoke',
    ) if (root / name).is_file()}
    report['status'] = ('interrupted' if interrupted else
                        'passed' if report['source_unchanged'] and
                        all(row['status'] == 'passed' for row in report['gates']) else 'failed')
    report['finished_at'] = datetime.now(timezone.utc).isoformat()
    save_summary(output, report)
    print(f'Release checks: {report["status"]}', flush=True)
    for row in report['gates']:
        print(f'{row["status"]:11} {row["target"]}: exit={row["exit_code"]}, '
              f'seconds={row["duration_seconds"]}, log={row["log"]}')
    print('Excluded: ' + '; '.join(EXCLUSIONS))
    for error in report['errors']:
        print(error)
    print('Individual test counts/skips and detailed findings remain in each gate log; '
          'a passing ratchet is not an all-passing upstream suite.')
    return 130 if interrupted else int(report['status'] != 'passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--make', default='make')
    parser.add_argument('--output', default='')
    parser.add_argument('--make-var', action='append', default=[])
    parser.add_argument('--list', action='store_true', help='show gates without executing or creating output')
    parser.add_argument('--preflight', action='store_true', help='check candidate compiler and Python contracts only')
    args = parser.parse_args()
    gates = release_gates()
    if args.list:
        print(json.dumps(dict(exclusions=EXCLUSIONS, deferred_certification=SCOPE['deferred_certification'], gates=gates), indent=2))
        return 0
    variables = dict(value.split('=', 1) for value in args.make_var)
    if args.preflight:
        return preflight(variables)
    output = (Path(args.output) if args.output else ROOT / 'temp' /
              f'release-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}').resolve()
    (ROOT / 'temp').mkdir(exist_ok=True)
    # Serializing within a run is insufficient if two callers start releases.
    with (ROOT / 'temp/test-release.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('another test-release run owns the shared build outputs')
        def interrupt(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupt)
        return run_release(ROOT, output, shlex.split(args.make), gates, variables)


if __name__ == '__main__':
    raise SystemExit(main())
