"""Paired C acceptance must retain failures and reject incomplete evidence."""

import copy
from pathlib import Path
import runpy

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def bench():
    return runpy.run_path(str(ROOT / 'bench/bench_c_paired.py'))


def rows(baseline, candidate):
    return [dict(round=i, label=label, benchmarks={
        name: dict(median_us=value, threshold_pct=pct, threshold_abs_us=absolute)
        for name, pct, absolute in [('sum_1024', .1, 5), ('movement_1024', .1, 5),
                                    ('chain_1024', .1, 5), ('matmul_16', .12, 8)]})
        for i, pair in enumerate(zip(baseline, candidate))
        for label, value in zip(('baseline', 'candidate'), pair)]


def test_paired_drift_cannot_hide_regression(bench):
    data = rows([10, 100, 100, 10, 100], [20, 101, 200, 20, 101])
    before = copy.deepcopy(data)
    report = bench['summarize'](data)
    assert not report['passed']
    assert report['workloads']['chain_1024']['ratio'] == 2
    assert data == before


def test_preserves_small_absolute_noise_budget(bench):
    report = bench['summarize'](rows([10] * 5, [14] * 5))
    assert report['passed']


def test_one_regressing_workload_fails_gate(bench):
    data = rows([100] * 5, [70] * 5)
    for row in data:
        if row['label'] == 'candidate':
            row['benchmarks']['chain_1024']['median_us'] = 125
    assert not bench['summarize'](data)['passed']


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'few', 'workload', 'nan', 'zero', 'budget'])
def test_invalid_evidence_fails_closed(bench, mutation):
    data = rows([100] * 5, [100] * 5)
    if mutation == 'missing': data.pop()
    elif mutation == 'duplicate': data.append(copy.deepcopy(data[0]))
    elif mutation == 'few': data = data[:6]
    elif mutation == 'workload': data[0]['benchmarks'].pop('matmul_16')
    elif mutation == 'nan': data[0]['benchmarks']['matmul_16']['median_us'] = float('nan')
    elif mutation == 'zero': data[0]['benchmarks']['matmul_16']['median_us'] = 0
    else: data[0]['benchmarks']['matmul_16']['threshold_pct'] = 999
    with pytest.raises(ValueError):
        bench['summarize'](data)
