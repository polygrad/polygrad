"""Release orchestration controls use tiny Make fixtures, not the real matrix."""

import json
import os
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
    assert targets[0] == 'test-release-preflight'
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
            'verify-source-mirrors', 'fuzz-smoke', 'test-release-c-performance',
            'bench-hlb-cuda-semantic', 'bench-hlb-cuda-timing'} <= set(targets)
    assert not {'test-all', 'verify', 'test-parity-opt', 'test-release-packages',
                'reference-migration-check', 'test-parity-op-census', 'analyze',
                'test-hip', 'publish-py', 'publish-js'} & set(targets)
    assert targets.index('test-analyze-reviewed') < targets.index('bench-hlb-cuda-semantic')
    assert 'test-symbolic-z3-supported' in targets
    assert 'test-release-py-performance' in targets
    assert targets.index('test-release-py-performance') < targets.index('test')
    perf = next(g for g in runner['release_gates']() if g['target'] == 'test-release-py-performance')
    assert perf['variables']['PY_PERF_OUTPUT'] == '{output}/python-performance/report.json'
    c_perf = next(g for g in runner['release_gates']() if g['target'] == 'test-release-c-performance')
    assert c_perf['variables']['C_PERF_OUTPUT'] == '{output}/c-performance/report.json'
    assert 'bench-smoke-regression' not in targets
    assert 'test-symbolic-z3' not in targets and 'test-symbolic-z3-fixed' not in targets
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


@pytest.mark.parametrize('mismatch', [False, True])
def test_candidate_versions_must_agree(runner, tmp_path, mismatch):
    (tmp_path / 'py').mkdir()
    (tmp_path / 'js').mkdir()
    (tmp_path / 'py/pyproject.toml').write_text('[project]\nversion = "0.5.2"\n')
    (tmp_path / 'js/package.json').write_text(json.dumps(dict(version='0.5.2')))
    (tmp_path / 'js/package-lock.json').write_text(json.dumps(dict(
        version='0.5.2', packages={'': dict(version='0.5.1' if mismatch else '0.5.2')})))
    if mismatch:
        with pytest.raises(ValueError, match='version mismatch'):
            runner['candidate_version'](tmp_path)
    else:
        assert runner['candidate_version'](tmp_path) == '0.5.2'


def test_release_stops_before_matrix_when_preflight_fails(runner, tmp_path):
    (tmp_path / 'Makefile').write_text(
        'test-release-preflight:\n\t@exit 3\n'
        'next:\n\t@touch should-not-run\n')
    gates = [dict(target=t, variables={}) for t in ('test-release-preflight', 'next')]
    output = tmp_path / 'results'
    assert runner['run_release'](tmp_path, output, ['make'], gates, {}) == 1
    assert not (tmp_path / 'should-not-run').exists()
    report = json.loads((output / 'summary.json').read_text())
    assert [g['status'] for g in report['gates']] == ['failed', 'not_run']


def test_release_preflight_checks_actual_compiler_and_python(runner, tmp_path):
    variables = dict(CC='clang', PYTHON=sys.executable, PARITY_PY=sys.executable)
    assert runner['preflight'](variables) == 0
    # GCC accepts the C11 core, but not the CPU renderer's __fp16 storage type.
    assert runner['preflight'](dict(variables, CC='gcc')) == 1
    wrong_python = tmp_path / 'python'
    wrong_python.write_text('#!/bin/sh\necho "CPython 3.12"\nexit 1\n')
    wrong_python.chmod(0o700)
    assert runner['preflight'](dict(variables, PYTHON=str(wrong_python))) == 1


def test_release_preflight_decodes_diagnostics_under_ascii_locale(tmp_path):
    compiler = tmp_path / 'compiler'
    compiler.write_bytes(b'#!/bin/sh\nprintf "\\342\\200\\230bad type\\342\\200\\231\\n" >&2\nexit 1\n')
    compiler.chmod(0o700)
    code = (f"import runpy; r=runpy.run_path({str(ROOT / 'scripts/test_release.py')!r}); "
            f"assert r['preflight']({dict(CC=str(compiler), PYTHON=sys.executable, PARITY_PY=sys.executable)!r}) == 1")
    result = subprocess.run([sys.executable, '-c', code],
                            env=dict(os.environ, LC_ALL='C', PYTHONUTF8='0', PYTHONCOERCECLOCALE='0'),
                            capture_output=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('tool,version', [('CLANG_FORMAT', 'clang-format version 18.0.0'),
                                         ('ANALYZER_CC', 'unreviewed clang')])
def test_release_preflight_rejects_unreviewed_tools(runner, tmp_path, tool, version):
    executable = tmp_path / 'tool'
    executable.write_text(f'#!/bin/sh\necho "{version}"\n', encoding='utf-8')
    executable.chmod(0o700)
    assert runner['preflight'](dict(CC='clang', PYTHON=sys.executable, PARITY_PY=sys.executable,
                                   **{tool:str(executable)})) == 1


def test_release_subprocesses_use_utf8(runner, tmp_path):
    assert runner['release_environment'](tmp_path)['PYTHONUTF8'] == '1'


def test_release_make_defaults_do_not_export_builtin_cc_or_ambient_python():
    env = {key: value for key, value in os.environ.items()
           if key not in ('CC', 'PYTHON', 'MAKEFLAGS', 'MAKEOVERRIDES', 'MFLAGS')}
    command = ['make', '-n', 'test-release', 'PARITY_PY=reviewed-python']
    defaults = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, check=True)
    assert "--make-var 'CC=clang'" in defaults.stdout
    assert "--make-var 'PYTHON=reviewed-python'" in defaults.stdout
    explicit = subprocess.run(command + ['CC=gcc', 'PYTHON=other-python'],
                              cwd=ROOT, env=env, text=True, capture_output=True, check=True)
    assert "--make-var 'CC=gcc'" in explicit.stdout
    assert "--make-var 'PYTHON=other-python'" in explicit.stdout


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


@pytest.mark.parametrize('target,variable,directory', [
    ('test-release-py-performance', 'PY_PERF_OUTPUT', 'python-performance'),
    ('test-release-c-performance', 'C_PERF_OUTPUT', 'c-performance'),
])
def test_performance_failure_blocks_release_and_keeps_private_report_path(runner, tmp_path, target, variable, directory):
    (tmp_path / 'Makefile').write_text(
        f'{target}:\n\t@test "$({variable})" != override.json\n\t@exit 1\n'
        'next:\n\t@echo next\n')
    gate = next(g for g in runner['release_gates']() if g['target'] == target)
    output = tmp_path / 'results'
    assert runner['run_release'](tmp_path, output, ['make'], [gate, dict(target='next', variables={})],
                                 {variable: 'override.json'}) == 1
    report = json.loads((output / 'summary.json').read_text())
    assert report['status'] == 'failed'
    assert [g['status'] for g in report['gates']] == ['failed', 'passed']
    assert f'{variable}={output}/{directory}/report.json' in report['gates'][0]['command']
    assert 'before' in report['gates'][0]['machine_conditions']
    assert 'after' in report['gates'][0]['machine_conditions']


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
