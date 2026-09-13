"""Reviewed diagnostics do not permit new warnings, stale proofs or incomplete runs."""

import copy
import runpy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def case():
    check = runpy.run_path(str(ROOT / 'scripts/check_analyzer.py'))
    diagnostic = dict(path='src/owner.c', line=4, column=2, message='possible NULL', checker='core.NullDereference')
    context = dict(sources=['src/owner.c'], flags=['-std=c11'], clang='clang test',
                   source_inputs={'src/owner.c': 'hash'})
    review = dict(schema_version=1, context=context,
                  diagnostics=[dict(diagnostic, disposition='false_positive', rationale='Checked caller admits nonnull.')])
    log = '==> src/owner.c\nsrc/owner.c:4:2: warning: possible NULL [core.NullDereference]\nanalyzer-exit: src/owner.c 0\n'
    return check, context, review, log


def test_reviewed_warning_keeps_raw_failure_visible(case):
    check, context, review, log = case
    report = check['evaluate'](log, 2, context, review)
    assert report['status'] == 'passed'
    assert report['raw_exit_code'] == 2
    assert report['reviewed_warnings'] == 1


@pytest.mark.parametrize('mutation', ['new', 'duplicate', 'error', 'tool_exit', 'incomplete', 'unparsed'])
def test_reviewed_gate_rejects_new_or_incomplete_diagnostics(case, mutation):
    check, context, review, log = case
    if mutation == 'new':
        log = log.replace('possible NULL', 'different bug')
    elif mutation == 'duplicate':
        log += 'src/owner.c:4:2: warning: possible NULL [core.NullDereference]\n'
    elif mutation == 'error':
        log += 'src/owner.c:9:2: error: failed to compile\n'
    elif mutation == 'tool_exit':
        log = log.replace('analyzer-exit: src/owner.c 0', 'analyzer-exit: src/owner.c 139')
    elif mutation == 'incomplete':
        log = log.replace('analyzer-exit: src/owner.c 0', '')
    else:
        log += 'clang: warning: unknown diagnostic format\n'
    assert check['evaluate'](log, 2, context, review)['status'] == 'failed'


@pytest.mark.parametrize('field', ['sources', 'flags', 'clang', 'source_inputs'])
def test_reviewed_gate_rejects_stale_context(case, field):
    check, context, review, log = case
    changed = copy.deepcopy(context)
    changed[field] = [] if isinstance(context[field], list) else 'changed'
    assert check['evaluate'](log, 2, changed, review)['status'] == 'failed'


def test_review_requires_reason_and_exact_disposition(case):
    check, context, review, log = case
    review['diagnostics'][0]['rationale'] = ''
    assert check['evaluate'](log, 2, context, review)['status'] == 'failed'


def test_clean_raw_run_needs_no_warning_allowances(case):
    check, context, review, _ = case
    report = check['evaluate']('==> src/owner.c\nanalyzer-exit: src/owner.c 0\n', 0, context, review)
    assert report['status'] == 'passed'
    assert report['reviewed_warnings'] == 0


def test_warning_without_nonzero_raw_gate_is_inconsistent(case):
    check, context, review, log = case
    assert check['evaluate'](log, 0, context, review)['status'] == 'failed'
