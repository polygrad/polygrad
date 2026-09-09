"""Negative controls for fixture gates; no network or GPU work is required."""

import os
from pathlib import Path
import runpy
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_package_gate_clears_checkout_and_backend_overrides(monkeypatch):
    helpers = runpy.run_path(str(ROOT / 'test/test_package_install.py'))
    keys = ('PYTHONPATH', 'PYTHONHOME', 'POLYGRAD_LIB', 'POLY_CORE',
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


def test_qwen_fixture_gate_rejects_missing_file(tmp_path):
    run = subprocess.run(['make', '-s', 'require-qwen3-gguf', f'QWEN3_GGUF={tmp_path}/missing'],
                         cwd=ROOT, capture_output=True, text=True)
    assert run.returncode != 0
    assert 'GGUF fixture not found' in run.stdout + run.stderr


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
