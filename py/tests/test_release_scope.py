"""Release exclusions are exact product limits, never physical-parity allowances."""

import copy
import json
from pathlib import Path
import runpy

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def gate():
    module = runpy.run_path(str(ROOT / 'test/op_vocabulary_census.py'))
    scope = json.loads((ROOT / 'test/fixtures/release_050_scope.json').read_text())
    findings = [dict(side=r['side'], op=r['op'], id=r['id'], status='open_debt', allowed=False, present=True)
                for r in scope['excluded_vocabulary']]
    report = dict(reference_commit=scope['reference_commit'], findings=findings)
    return module, scope, report


def test_scoped_vocabulary_does_not_relabel_open_debts(gate):
    module, scope, report = gate
    before = copy.deepcopy(report)
    assert module['release_scope_errors'](report, scope, '0.5.0') == []
    assert report == before
    assert all(not row['allowed'] for row in report['findings'])


@pytest.mark.parametrize('mutation', ['new', 'stale', 'resolved', 'wrong_id', 'duplicate', 'missing', 'version', 'reference'])
def test_scoped_vocabulary_rejects_new_or_changed_gaps(gate, mutation):
    module, scope, report = gate
    version = '0.5.0'
    if mutation == 'new':
        report['findings'].append(dict(side='tinygrad_only', op='NEW', id=None, status='unregistered', allowed=False, present=True))
    elif mutation == 'stale':
        report['findings'][0]['present'] = False
    elif mutation == 'resolved':
        report['findings'][0]['status'] = 'resolved'
    elif mutation == 'wrong_id':
        report['findings'][0]['id'] = 'wrong'
    elif mutation == 'duplicate':
        scope['excluded_vocabulary'].append(scope['excluded_vocabulary'][0])
    elif mutation == 'missing':
        report['findings'].pop()
    elif mutation == 'version':
        version = '0.6.0'
    else:
        report['reference_commit'] = 'other'
    assert module['release_scope_errors'](report, scope, version)
