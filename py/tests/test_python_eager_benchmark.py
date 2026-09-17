"""Performance acceptance contracts, without timing-dependent unit assertions."""

import copy
from pathlib import Path
import runpy
import json
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def bench():
    return runpy.run_path(str(ROOT / 'bench/bench_python_eager.py'))


def rows(baseline, candidate):
    return [dict(round=i, label=label, median_us=value)
            for i, pair in enumerate(zip(baseline, candidate))
            for label, value in zip(('baseline', 'candidate'), pair)]


def test_budget_uses_paired_ratios_not_ratio_of_medians(bench):
    # Drift makes ratio-of-medians 1.01, hiding three 2x slower pairs.
    result = bench['summarize'](rows([1, 100, 100, 1, 100], [2, 101, 200, 2, 101]), 1.02)
    assert result['ratio'] == 2
    assert not result['passed']


def test_command_rejects_regression_hidden_by_drift(bench, tmp_path, monkeypatch):
    values = {'baseline': iter([1, 100, 100, 1, 100]), 'candidate': iter([2, 101, 200, 2, 101])}
    def fake_run(command, **kwargs):
        label = 'candidate' if command[0] == sys.executable else 'baseline'
        prefix = ROOT if label == 'candidate' else Path('/baseline')
        value = next(values[label])
        row = dict(median_us=value, samples_us=[value] * 9, python=sys.version, numpy_version='2.4.6',
                   prefix=str(prefix), version='0.5.1', package=str(prefix / 'py/polygrad/__init__.py'),
                   library=str(prefix / 'build/libpolygrad.so'))
        return SimpleNamespace(returncode=0, stdout=json.dumps(row), stderr='')
    monkeypatch.setattr(bench['subprocess'], 'run', fake_run)
    monkeypatch.setattr(sys, 'argv', ['bench', '--baseline-python', '/baseline/bin/python',
                                     '--workload', 'eager', '--rounds', '5',
                                     '--output', str(tmp_path / 'report.json')])
    assert bench['main']() == 1


def test_budget_keeps_every_pair_and_does_not_retry_for_a_pass(bench):
    data = rows([100] * 5, [90, 95, 96, 110, 100])
    before = copy.deepcopy(data)
    result = bench['summarize'](data, 1.02)
    assert result['ratio'] == .96
    assert result['pair_ratios'] == [.9, .95, .96, 1.1, 1.0]
    assert result['passed'] and data == before


def test_training_regression_cannot_hide_behind_fast_eager_calls(bench):
    data = [dict(row, workload=name)
            for name, times in [('eager', [70] * 5), ('training', [110] * 5)]
            for row in rows([100] * 5, times)]
    report = bench['summarize_workloads'](data, 1.02)
    assert report['workloads']['eager']['passed']
    assert not report['workloads']['training']['passed']
    assert not report['passed']


def test_default_gate_measures_training_eager_and_jit_readback(bench, tmp_path, monkeypatch):
    calls = []
    def fake_run(command, **kwargs):
        label = 'candidate' if command[0] == sys.executable else 'baseline'
        workload = 'jit_readback' if 'TinyJit' in command[-1] else 'training' if 'Adam' in command[-1] else 'eager'
        calls.append((label, workload))
        prefix = ROOT if label == 'candidate' else Path('/baseline')
        value = 110 if label == 'candidate' and workload == 'training' else 100
        row = dict(median_us=value, samples_us=[value] * 9, python=sys.version, numpy_version='2.4.6',
                   prefix=str(prefix), version='0.5.1', package=str(prefix / 'py/polygrad/__init__.py'),
                   library=str(prefix / 'build/libpolygrad.so'))
        return SimpleNamespace(returncode=0, stdout=json.dumps(row), stderr='')
    monkeypatch.setattr(bench['subprocess'], 'run', fake_run)
    output = tmp_path / 'report.json'
    monkeypatch.setattr(sys, 'argv', ['bench', '--baseline-python', '/baseline/bin/python',
                                     '--rounds', '5', '--output', str(output)])
    assert bench['main']() == 1
    assert len(calls) == 30
    assert sum(workload == 'jit_readback' for _, workload in calls) == 10
    assert not json.loads(output.read_text())['workloads']['training']['passed']


@pytest.mark.parametrize('bad', [0, -1, float('nan'), float('inf')])
def test_invalid_timings_fail_closed(bench, bad):
    data = rows([100] * 5, [100] * 5)
    data[0]['median_us'] = bad
    with pytest.raises(ValueError):
        bench['summarize'](data, 1.02)


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'few'])
def test_incomplete_pairs_fail_closed(bench, mutation):
    data = rows([100] * 5, [100] * 5)
    if mutation == 'missing': data.pop()
    elif mutation == 'duplicate': data.append(dict(data[0]))
    else: data = data[:6]
    with pytest.raises(ValueError):
        bench['summarize'](data, 1.02)


def pair_metadata():
    return [dict(label=label, package=str(prefix / 'py/polygrad/__init__.py'),
                 library=str(prefix / 'build/libpolygrad.so'), prefix=str(prefix),
                 python=sys.version, numpy_version='2.4.6')
            for label, prefix in [('baseline', Path('/baseline')), ('candidate', ROOT)]]


@pytest.mark.parametrize('mutation', ['package', 'library', 'python', 'numpy', 'outside', 'candidate'])
def test_contaminated_pair_fails_closed(bench, mutation):
    pair = pair_metadata()
    bench['validate_pair'](pair)
    if mutation in ('package', 'library'):
        pair[0][mutation] = pair[1][mutation]
    elif mutation == 'outside': pair[0]['library'] = '/another/lib.so'
    elif mutation == 'candidate': pair[1]['library'] = str(ROOT / 'stale/lib.so')
    else: pair[0]['python' if mutation == 'python' else 'numpy_version'] = 'different'
    with pytest.raises(ValueError):
        bench['validate_pair'](pair)


def test_tampered_archive_is_rejected_before_install(bench, tmp_path, monkeypatch):
    artifact = tmp_path / 'wrong.tar.gz'
    artifact.write_bytes(b'not the published package')
    def no_subprocess(*args, **kwargs):
        pytest.fail('unverified archive reached installation')
    monkeypatch.setattr(bench['subprocess'], 'run', no_subprocess)
    with pytest.raises(ValueError, match='pinned published artifact'):
        bench['prepare_baseline'](tmp_path / 'baseline', artifact)
    assert not (tmp_path / 'baseline/venv').exists()


def test_existing_baseline_directory_is_not_reused(bench, tmp_path):
    with pytest.raises(FileExistsError):
        bench['prepare_baseline'](tmp_path)


def test_preparation_uses_isolated_pinned_install(bench, tmp_path, monkeypatch):
    import hashlib
    import numpy
    from test import test_package_install

    artifact = tmp_path / 'package.tar.gz'
    artifact.write_bytes(b'fixture package')
    spec = dict(version='fixture', filename=artifact.name,
                sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
    read_text = Path.read_text
    def fixture_text(path, *args, **kwargs):
        return json.dumps(spec) if path.name == 'python_performance_baseline.json' else read_text(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', fixture_text)
    calls = []
    monkeypatch.setattr(test_package_install, 'run', lambda command, cwd, env, log: calls.append((command, env)))
    for name in ('PIP_TARGET', 'PIP_PREFIX', 'PIP_USER', 'PYTHONPATH', 'PYTHONHOME', 'POLY_LIB'):
        monkeypatch.setenv(name, '/unrelated')
    work = tmp_path / 'baseline'
    python, version = bench['prepare_baseline'](work, artifact)
    assert python == work / 'venv/bin/python' and version == 'fixture'
    install, env = calls[1]
    assert install[:7] == [str(python), '-I', '-m', 'pip', '--isolated', 'install', '--no-cache-dir']
    assert f'numpy=={numpy.__version__}' in install
    assert install[-1] == str(work / artifact.name)
    assert not set(('PIP_TARGET', 'PIP_PREFIX', 'PIP_USER', 'PYTHONPATH', 'PYTHONHOME', 'POLY_LIB')) & env.keys()
    assert json.loads((work / 'provenance.json').read_text())['sha256'] == spec['sha256']
