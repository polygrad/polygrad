"""Release orchestration controls use tiny Make fixtures, not the real matrix."""

import json
from pathlib import Path
import runpy
import signal
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def runner():
    return runpy.run_path(str(ROOT / 'scripts/test_release.py'))


def test_release_manifest_covers_required_lanes_once(runner):
    targets = [gate['target'] for gate in runner['release_gates']()]
    assert len(targets) == len(set(targets))
    assert {'test', 'test-x86', 'test-interp', 'test-cuda', 'test-py',
            'test-py-x86', 'test-hf-e2e', 'test-qwen3', 'test-browser',
            'test-js-native-cpu', 'test-js-native-x86', 'test-js-native-interp',
            'test-js-native-cuda', 'test-js-native-gc', 'test-js-wasm',
            'test-py-sdist-install', 'test-js-package-install',
            'test-parity', 'test-parity-graph', 'test-parity-cuda',
            'test-compat-tinygrad-upstream-ratchet', 'test-compat-tinygrad-ops',
            'test-compat-tinygrad-nn', 'test-compat-tinygrad-policy', 'test-nn-wasm',
            'test-reference-parity', 'test-analyze-reviewed', 'test-release-op-census', 'format-check',
            'verify-source-mirrors', 'fuzz-smoke', 'bench-smoke-regression',
            'bench-hlb-cuda-semantic', 'bench-hlb-cuda-timing'} <= set(targets)
    assert not {'test-all', 'verify', 'test-parity-opt', 'test-release-packages',
                'reference-migration-check', 'test-parity-op-census', 'analyze',
                'test-hip', 'publish-py', 'publish-js'} & set(targets)
    assert targets.index('test-analyze-reviewed') < targets.index('bench-hlb-cuda-semantic')
    assert targets[-1] == 'bench-hlb-cuda-timing'
    ops = next(g for g in runner['release_gates']() if g['target'] == 'test-compat-tinygrad-ops')
    assert '--baseline test/fixtures/tinygrad_upstream_ops_cpu_014_baseline.json' in ops['variables']['UPSTREAM_COMPAT_ARGS']
    assert '--timeout 3600' in ops['variables']['UPSTREAM_COMPAT_ARGS']
    nn = next(g for g in runner['release_gates']() if g['target'] == 'test-compat-tinygrad-nn')
    assert '--baseline test/fixtures/tinygrad_upstream_nn_cpu_014_baseline.json' in nn['variables']['UPSTREAM_COMPAT_ARGS']
    policy = next(g for g in runner['release_gates']() if g['target'] == 'test-compat-tinygrad-policy')
    assert policy['variables']['UPSTREAM_POLICY_TESTS'].split() == [
        'test/backend/test_setitem.py', 'test/backend/test_tensor.py', 'test/null/test_indexing.py']
    assert policy['variables']['UPSTREAM_COMPAT_DIR'] == '{output}/upstream-policy'


def test_release_continues_after_failure_and_preserves_logs(runner, tmp_path):
    (tmp_path / 'Makefile').write_text(
        'first:\n\t@echo first\n\t@exit 7\n'
        'second:\n\t@test "$(TOKEN)" = "two words"\n\t@echo second\n')
    gates = [dict(target=t, variables={}) for t in ('first', 'second')]
    output = tmp_path / 'results'
    status = runner['run_release'](tmp_path, output, ['make'], gates, {'TOKEN': 'two words'})
    assert status == 1
    report = json.loads((output / 'summary.json').read_text())
    assert report['status'] == 'failed'
    assert [g['status'] for g in report['gates']] == ['failed', 'passed']
    assert report['gates'][0]['exit_code'] != 0
    assert all(g['duration_seconds'] >= 0 for g in report['gates'])
    assert 'first' in (output / report['gates'][0]['log']).read_text()
    assert 'second' in (output / report['gates'][1]['log']).read_text()
    with pytest.raises(FileExistsError):
        runner['run_release'](tmp_path, output, ['make'], gates, {})


def test_release_serializes_recursive_make_and_clears_filters(runner, tmp_path, monkeypatch):
    monkeypatch.setenv('MAKEFLAGS', 'ij8')
    monkeypatch.setenv('MAKEOVERRIDES', 'TOKEN=wrong')
    monkeypatch.setenv('POLY_TEST_FILTER', 'one-test')
    monkeypatch.setenv('DEV', 'X86')
    monkeypatch.setenv('POLY_BROWSER_SKIP_UNAVAILABLE', '1')
    (tmp_path / 'Makefile').write_text(
        'first: a b\n\t@test "$$DEV" = CPU\n'
        '\t@test -z "$$POLY_TEST_FILTER"\n'
        '\t@test "$$POLY_BROWSER_SKIP_UNAVAILABLE" = 0\n'
        '\t@test "$$POLY_BROWSER_DEVICES" = auto,interp,webgpu\n'
        '\t@test "$(HAS_CUDA)" = 1\n\t@test "$(HAS_HIP)" = 0\n'
        'a:\n\t@mkdir active\n\t@sleep 0.1\n\t@rmdir active\n'
        'b:\n\t@mkdir active\n\t@sleep 0.1\n\t@rmdir active\n')
    assert runner['run_release'](tmp_path, tmp_path / 'results', ['make'],
                                 [dict(target='first', variables={})], {}) == 0


def test_release_missing_executable_is_a_failure(runner, tmp_path):
    output = tmp_path / 'results'
    assert runner['run_release'](tmp_path, output, [str(tmp_path / 'missing')],
                                 [dict(target='first', variables={})], {}) == 1
    row = json.loads((output / 'summary.json').read_text())['gates'][0]
    assert row['status'] == 'failed'
    assert row['exit_code'] == 127


def test_release_rejects_source_changes_during_execution(runner, tmp_path):
    (tmp_path / 'Makefile').write_text('first:\n\t@mkdir -p src\n\t@echo changed > src/new.c\n')
    output = tmp_path / 'results'
    assert runner['run_release'](tmp_path, output, ['make'],
                                 [dict(target='first', variables={})], {}) == 1
    report = json.loads((output / 'summary.json').read_text())
    assert report['source_unchanged'] is False
    assert report['gates'][0]['status'] == 'passed'
    assert report['gates'][0]['log_sha256']


def test_generated_browser_bundle_does_not_invalidate_source_evidence(runner, tmp_path):
    (tmp_path / 'Makefile').write_text('first:\n\t@mkdir -p js/test/browser\n\t@echo generated > js/test/browser/tests.js\n')
    output = tmp_path / 'results'
    assert runner['run_release'](tmp_path, output, ['make'],
                                 [dict(target='first', variables={})], {}) == 0
    report = json.loads((output / 'summary.json').read_text())
    assert report['source_unchanged'] is True
    assert 'js/test/browser/tests.js' not in report['source_inputs']
    assert 'Makefile' in report['source_inputs']


def test_release_interrupt_stops_children_and_marks_remaining_not_run(runner, tmp_path, monkeypatch):
    class Process:
        pid = 12345
        waits = 0

        def wait(self, timeout=None):
            self.waits += 1
            if self.waits == 1:
                raise KeyboardInterrupt
            return 0

    signals = []
    monkeypatch.setattr(subprocess, 'Popen', lambda *args, **kwargs: Process())
    monkeypatch.setattr(runner['os'], 'killpg', lambda pid, sig: signals.append((pid, sig)))
    output = tmp_path / 'results'
    gates = [dict(target=t, variables={}) for t in ('first', 'second')]
    assert runner['run_release'](tmp_path, output, ['make'], gates, {}) == 130
    report = json.loads((output / 'summary.json').read_text())
    assert report['status'] == 'interrupted'
    assert [g['status'] for g in report['gates']] == ['interrupted', 'not_run']
    assert signals == [(12345, signal.SIGTERM), (12345, signal.SIGKILL)]


def test_release_lock_rejects_a_second_runner(runner, tmp_path, monkeypatch, capsys):
    # Use a private lock: this test itself also runs inside the real release.
    monkeypatch.setitem(runner['main'].__globals__, 'ROOT', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['test_release.py', '--output', str(tmp_path / 'results')])
    (tmp_path / 'temp').mkdir()
    with (tmp_path / 'temp/test-release.lock').open('a') as lock:
        runner['fcntl'].flock(lock, runner['fcntl'].LOCK_EX | runner['fcntl'].LOCK_NB)
        with pytest.raises(SystemExit) as rejected:
            runner['main']()
    assert rejected.value.code == 2
    assert 'another test-release run' in capsys.readouterr().err
    assert not (tmp_path / 'results').exists()


def test_release_make_dry_run_does_not_execute_matrix(tmp_path):
    output = tmp_path / 'never-created'
    marker = tmp_path / 'interpreter-started'
    interpreter = tmp_path / 'interpreter.py'
    interpreter.write_text(f'from pathlib import Path\nPath({str(marker)!r}).touch()\n')
    command = ['make', '-n', 'test-release', f'RELEASE_DIR={output}',
               f'PARITY_PY={sys.executable} {interpreter}']
    run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert 'scripts/test_release.py' in run.stdout
    assert not marker.exists()
    assert not output.exists()
