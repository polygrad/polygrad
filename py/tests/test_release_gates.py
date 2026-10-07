"""Negative controls for fixture gates; no network or GPU work is required."""

import json
import os
from pathlib import Path
import runpy
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_graph_divergence_requires_both_exact_reviewed_graphs():
    import copy
    checker = runpy.run_path(str(ROOT / 'test/compare_tensor_graphs.py'))
    digest, reviewed = checker['graph_digest'], checker['reviewed_graph_pair']
    tg = {'root': 0, 'nodes': [{'op': 'ADD', 'dtype': 'float32', 'arg': None, 'src': []}]}
    pg = {'root': 0, 'nodes': [{'op': 'WHERE', 'dtype': 'float32', 'arg': None, 'src': []}]}
    entry = {'id': 'test-only', 'status': 'approved', 'stages': ['tensor'],
             'graph_pairs': {'case': {'tinygrad': digest(tg), 'polygrad': digest(pg)}}}
    entries = {entry['id']: entry}
    assert reviewed(entries, 'case', 'tensor', tg, pg) == 'test-only'
    assert reviewed(entries, 'other', 'tensor', tg, pg) is None
    assert reviewed(entries, 'case', 'runtime', tg, pg) is None
    for side in (0, 1):
        for field, value in [('op', 'SUB'), ('dtype', 'float64'), ('arg', 1), ('src', [0])]:
            graphs = copy.deepcopy([tg, pg])
            graphs[side]['nodes'][0][field] = value
            assert reviewed(entries, 'case', 'tensor', *graphs) is None
    entry['status'] = 'open_debt'
    assert reviewed(entries, 'case', 'tensor', tg, pg) is None


@pytest.mark.parametrize('failure', ['', 'missing-wheel', 'tampered'])
@pytest.mark.parametrize('relative', [False, True])
def test_publish_python_uses_verified_staged_archives(tmp_path, failure, relative):
    import hashlib
    import re

    version = re.search(r'^version = "([^"]+)"', (ROOT / 'py/pyproject.toml').read_text(), re.M)[1]
    release = tmp_path / 'release'
    wheels = release / 'wheels'
    wheels.mkdir(parents=True)
    sdist = release / f'polygrad-{version}.tar.gz'
    sdist.write_bytes(b'sdist')
    wheel = wheels / f'polygrad-{version}-cp39-cp39-manylinux_2_28_x86_64.whl'
    wheel.write_bytes(b'wheel')
    checksum_path = sdist.name if relative else sdist
    (release / 'SHA256SUMS').write_text(f'{hashlib.sha256(sdist.read_bytes()).hexdigest()}  {checksum_path}\n')
    (wheels / 'SHA256SUMS').write_text(f'{hashlib.sha256(wheel.read_bytes()).hexdigest()}  {wheel.name}\n')
    if failure == 'missing-wheel':
        wheel.unlink()
    elif failure == 'tampered':
        wheel.write_bytes(b'changed')
    calls = tmp_path / 'calls.jsonl'
    twine = tmp_path / 'twine'
    twine.write_text(f'#!{sys.executable}\nimport json, sys\n'
                     f'with open({str(calls)!r}, "a") as f: f.write(json.dumps(sys.argv[1:]) + "\\n")\n')
    twine.chmod(0o700)
    # Even the old target must not build or contact a registry in this regression.
    result = subprocess.run(['make', '--no-print-directory', '-o', 'build-py-sdist',
                             'publish-py', f'PUBLISH_DIR={release}', f'TWINE={twine}'],
                            cwd=ROOT, capture_output=True, text=True)
    if failure:
        assert result.returncode != 0, result.stdout + result.stderr
        assert not calls.exists()
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert [json.loads(line) for line in calls.read_text().splitlines()] == [
            ['check', str(sdist), str(wheel)], ['upload', str(sdist), str(wheel)]]


@pytest.mark.parametrize('failure', ['', 'build', 'missing-wheel'])
@pytest.mark.parametrize('relative', [False, True])
def test_manylinux_stages_only_complete_matrix(tmp_path, monkeypatch, failure, relative):
    import hashlib
    import re

    version = re.search(r'^version = "([^"]+)"', (ROOT / 'py/pyproject.toml').read_text(), re.M)[1]
    release = tmp_path / 'release'
    release.mkdir()
    sdist = release / f'polygrad-{version}.tar.gz'
    sdist.write_bytes(b'fixture archive; no real compilation in this harness test')
    checksum = hashlib.sha256(sdist.read_bytes()).hexdigest()
    (release / 'SHA256SUMS').write_text(f'{checksum}  {sdist.name if relative else sdist}\n')
    runner = tmp_path / 'apptainer'
    runner.write_text(f'#!{sys.executable}\n' + '''
import os, pathlib, sys
args = sys.argv[1:]
if args[0] == 'pull':
    pathlib.Path(args[1]).write_bytes(b'mock image')
else:
    # --containall must not put compiler/pip intermediates in the session tmpfs.
    assert '--workdir' in args
    scratch = pathlib.Path(args[args.index('--workdir') + 1])
    assert scratch.is_dir()
    if os.environ['MOCK_FAILURE'] == 'build':
        sys.exit(23)
    work = next(pathlib.Path(a.split(':')[0]) for a in args if a.endswith(':/work'))
    version = args[-1]
    wheels = work / 'wheels'
    wheels.mkdir()
    versions = (39, 310, 311, 312) if os.environ['MOCK_FAILURE'] else (39, 310, 311, 312, 313)
    for v in versions:
        (wheels / f'polygrad-{version}-cp{v}-cp{v}-manylinux_2_28_x86_64.whl').write_bytes(b'mock wheel')
''')
    runner.chmod(0o700)
    for name in ('APPTAINER_CONTAINER', 'SINGULARITY_CONTAINER'):
        monkeypatch.delenv(name, raising=False)
    env = dict(os.environ, APPTAINER=str(runner), MOCK_FAILURE=failure,
               MANYLINUX_WORK_DIR=str(tmp_path / 'work'))
    command = ['make', '--no-print-directory', 'build-py-manylinux',
               f'MANYLINUX_RELEASE_DIR={release}', 'MANYLINUX_IMAGE=docker://fixture']
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    if failure:
        assert result.returncode != 0, result.stdout + result.stderr
        assert not (release / 'wheels').exists()
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert len(list((release / 'wheels').glob('*.whl'))) == 5
        subprocess.run(['sha256sum', '-c', 'SHA256SUMS'], cwd=release / 'wheels', check=True)
        repeated = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        assert repeated.returncode != 0
        assert 'already exists' in repeated.stderr


def test_c_harness_probe_preserves_debug_link_flags(tmp_path, monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, '', '')

    monkeypatch.setenv('LDFLAGS_DEBUG', '-lm -ldl -fsanitize=address,undefined -no-pie')
    monkeypatch.setattr(subprocess, 'run', run)
    test_c_harness_registration_capacity(tmp_path, 1)
    assert calls[0][-4:] == ['-lm', '-ldl', '-fsanitize=address,undefined', '-no-pie']


@pytest.mark.parametrize('available', [0, 1])
def test_c_harness_registration_capacity(tmp_path, available):
    source = tmp_path / 'registry.c'
    source.write_text('''#include "test_harness.h"
TestEntry g_tests[MAX_TESTS];
int g_n_tests;
int g_current_test_skipped;
__attribute__((constructor(101))) static void fill_registry(void) {
  g_n_tests = MAX_TESTS - AVAILABLE;
}
TEST(harness, boundary) { PASS(); }
int main(void) { return g_n_tests == MAX_TESTS ? 0 : 1; }
''')
    binary = tmp_path / 'registry'
    # Use the same link configuration as the sanitized C suite, including any
    # toolchain-specific executable layout flags (e.g. -no-pie).
    link_flags = shlex.split(os.environ.get('LDFLAGS_DEBUG', '-lm -ldl -fsanitize=address,undefined'))
    subprocess.run([*shlex.split(os.environ.get('CC', 'cc')), '-std=gnu11',
                    '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                    '-Itest', '-Isrc', f'-DAVAILABLE={available}', str(source),
                    '-o', str(binary), *link_flags], cwd=ROOT, check=True, capture_output=True)
    run = subprocess.run([str(binary)], capture_output=True, text=True)
    assert run.returncode == (0 if available else 2), run.stderr
    if not available:
        assert 'test registry capacity exceeded' in run.stderr
    assert 'AddressSanitizer' not in run.stderr


def test_model_compatibility_artifact_uses_canonical_reference(tmp_path):
    env = dict(os.environ, ENGINE='polygrad', COMPAT_CASES='mlp_mnist',
               DEV='CPU', POLY_DEV='cpu', PYTHONPATH='test:py',
               POLY_LIB=str(ROOT / 'build/libpolygrad.so'))
    run = subprocess.run([sys.executable, 'test/tinygrad_compat_cases.py'],
                         cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    artifact = json.loads(run.stdout)
    register = json.loads((ROOT / 'test/fixtures/parity_divergences.json').read_text(encoding='utf-8'))
    assert artifact['reference_commit'] == register['reference']['commit']

    # A canonical label must not make a genuinely stale artifact acceptable.
    artifact['reference_commit'] = '0' * 40
    stale = tmp_path / 'stale.json'
    stale.write_text(json.dumps(artifact))
    rejected = subprocess.run(
        [sys.executable, 'test/compare_tinygrad_compat.py', str(stale), str(stale),
         '--output', str(tmp_path / 'report.json')],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    assert rejected.returncode != 0
    assert 'stale compatibility source lock' in rejected.stderr


def test_python_x86_target_selects_frontend_and_core_device():
    recipe = subprocess.run(['make', '-n', 'test-py-x86'], cwd=ROOT,
                            capture_output=True, text=True, check=True)
    command = next(line for line in recipe.stdout.splitlines()
                   if ' -m pytest ' in line and 'POLY_DEV=' in line)
    env = dict(os.environ)
    env.pop('DEV', None)
    for token in shlex.split(command):
        if '=' not in token:
            break
        key, value = token.split('=', 1)
        env[key] = value
    # The prefixed override must select the same Python and C lane even when
    # the caller supplied a different DEV default.
    probe = subprocess.run([sys.executable, '-c',
                            "import os; from polygrad import Tensor, Device; "
                            "assert os.environ['POLY_DEV'].lower() == 'x86'; "
                            "assert Device.DEFAULT == 'X86', Device.DEFAULT; "
                            "assert Tensor([1.0]).device == 'X86'"],
                           cwd=ROOT, env=env, capture_output=True, text=True)
    assert probe.returncode == 0, probe.stdout + probe.stderr


def test_package_gate_clears_checkout_and_backend_overrides(monkeypatch):
    helpers = runpy.run_path(str(ROOT / 'test/test_package_install.py'))
    keys = ('PYTHONPATH', 'PYTHONHOME', 'POLY_LIB', 'POLY_CORE',
            'POLYGRAD_SKIP_NATIVE', 'NODE_PATH', 'NODE_OPTIONS',
            'PIP_TARGET', 'PIP_PREFIX', 'PIP_USER',
            'npm_config_ignore_scripts', 'NPM_CONFIG_IGNORE_SCRIPTS')
    for key in keys:
        monkeypatch.setenv(key, 'contaminated')
    env = helpers['clean_environment']()
    assert all(key not in env for key in keys)
    assert all(os.environ[key] == 'contaminated' for key in keys)


def test_package_gate_rejects_missing_or_ambiguous_artifact(tmp_path):
    select = runpy.run_path(str(ROOT / 'test/test_package_install.py'))['only_artifact']
    with pytest.raises(RuntimeError, match='exactly one'):
        select(tmp_path, '*.tgz')
    (tmp_path / 'first.tgz').touch()
    assert select(tmp_path, '*.tgz') == tmp_path / 'first.tgz'
    (tmp_path / 'second.tgz').touch()
    with pytest.raises(RuntimeError, match='exactly one'):
        select(tmp_path, '*.tgz')


def test_package_gate_rejects_wrong_floor_before_install():
    result = subprocess.run([sys.executable, str(ROOT / 'test/test_package_install.py'),
                             'python', '--require-python', '0.0'], capture_output=True, text=True)
    assert result.returncode == 2
    assert 'requires 0.0; running' in result.stderr
    assert 'Evidence:' not in result.stdout


def test_qwen_fixture_gate_rejects_missing_file(tmp_path):
    run = subprocess.run(['make', '-s', 'require-qwen3-gguf', f'QWEN3_GGUF={tmp_path}/missing'],
                         cwd=ROOT, capture_output=True, text=True)
    assert run.returncode != 0
    assert 'GGUF fixture not found' in run.stdout + run.stderr


@pytest.mark.parametrize('invoke_compiler', [False, True])
def test_package_fallback_proves_compiler_execution_without_verbose_logs(tmp_path, monkeypatch, invoke_compiler):
    install = runpy.run_path(str(ROOT / 'test/test_package_install.py'))['node_install']

    def fake_run(command, cwd, env, log):
        if command[1] == 'pack':
            (tmp_path / 'polygrad-test.tgz').touch()
        elif command[1] == 'install':
            native = cwd.name == 'native'
            if native:
                addon = cwd / 'node_modules/polygrad/build/Release/polygrad_napi.node'
                addon.parent.mkdir(parents=True)
                addon.touch()
            elif invoke_compiler:
                assert subprocess.run([env['CC']], capture_output=True).returncode == 1
            log.write_text('native addon built successfully' if native else
                           'native addon build failed (will use WASM fallback)', encoding='utf-8')

    monkeypatch.setitem(install.__globals__, 'run', fake_run)
    if invoke_compiler:
        install(tmp_path, {}, 'npm', 'node')
    else:
        with pytest.raises(RuntimeError, match='forced compiler failure was not observed'):
            install(tmp_path, {}, 'npm', 'node')


def test_wino_environment_request_is_rejected():
    run = subprocess.run([sys.executable, '-c', 'import polygrad.helpers'], cwd=ROOT,
                         env=dict(os.environ, WINO='1'), capture_output=True, text=True)
    assert run.returncode != 0
    assert 'NotImplementedError: Winograd convolution is not implemented' in run.stderr


@pytest.mark.parametrize('dependency', ['huggingface_hub', 'transformers', 'torch'])
@pytest.mark.parametrize('strict', [False, True])
def test_hf_missing_dependency_accounts_for_every_test(tmp_path, dependency, strict):
    report = tmp_path / 'hf.xml'
    # Block imports even on fully provisioned release machines. Run the real
    # module, not a replacement test whose collection behavior could differ.
    code = f'''
import importlib.abc, sys, pytest
class MissingDependency(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == {dependency!r}:
            raise ModuleNotFoundError('blocked dependency: ' + fullname, name=fullname)
sys.meta_path.insert(0, MissingDependency())
raise SystemExit(pytest.main(['py/tests/test_hf_e2e.py', '-q', '--junitxml={report}']))
'''
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTEST_ADDOPTS='',
               POLY_REQUIRE_HF=str(int(strict)), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    run = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=env,
                         capture_output=True, text=True)
    cases = ET.parse(report).findall('.//testcase')
    assert len(cases) == 8, run.stdout + run.stderr
    assert run.returncode == (1 if strict else 0), run.stdout + run.stderr
    assert all(case.find('error' if strict else 'skipped') is not None for case in cases)


@pytest.mark.parametrize('strict', [False, True])
def test_qwen_without_cuda_is_not_a_pass(strict):
    binary = ROOT / 'build/polygrad_test'
    if not binary.exists():
        pytest.skip('build/polygrad_test not built')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='',
               POLY_QWEN3_GGUF=str(ROOT / 'temp/Qwen3-0.6B-Q8_0.gguf'),
               ASAN_OPTIONS='detect_leaks=1:protect_shadow_gap=0')
    args = [str(binary), 'qwen3.forward_cuda']
    if strict:
        args.append('--require-no-skips')
    run = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True)
    output = run.stdout + run.stderr
    # A CPU-only build must reject the absent test, never certify CUDA.
    if "no tests matched filter" in output:
        assert run.returncode != 0
        return
    assert '[SKIP] forward_cuda' in output, output
    assert '[PASS] forward_cuda' not in output, output
    assert run.returncode == (2 if strict else 0), output
