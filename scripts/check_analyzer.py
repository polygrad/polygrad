#!/usr/bin/env python3
"""Run raw Clang analysis; accept only exact, source-bound reviewed diagnostics."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess


ROOT = Path(__file__).resolve().parents[1]
WARNING = re.compile(r'^(.*?):(\d+):(\d+): warning: (.*?) \[([^\]]+)\]$')
FIELDS = ('path', 'line', 'column', 'message', 'checker')


def context(root, sources, flags, clang):
    paths = {root / path for path in sources}
    for directory in ('src', 'vendor'):
        paths.update((root / directory).rglob('*.h'))
    return dict(sources=sources, flags=flags, clang=clang,
                source_inputs={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sorted(paths)})


def evaluate(log, raw_exit_code, current, review):
    errors, diagnostics, exits = [], [], []
    for line in log.splitlines():
        match = WARNING.fullmatch(line)
        if match:
            path, lineno, column, message, checker = match.groups()
            diagnostics.append(dict(path=path, line=int(lineno), column=int(column), message=message, checker=checker))
        elif re.search(r'\b(warning|error|fatal error):', line):
            errors.append(f'unreviewable diagnostic: {line}')
        if line.startswith('analyzer-exit: '):
            try:
                path, code = line[len('analyzer-exit: '):].rsplit(' ', 1)
                exits.append(path)
                if int(code) != 0:
                    errors.append(f'{path}: analyzer process exited {code}')
            except ValueError:
                errors.append(f'invalid completion: {line}')
    if Counter(exits) != Counter(current['sources']) or not exits:
        errors.append('missing, duplicate or unexpected translation-unit completions')
    if raw_exit_code not in (0, 2) or bool(diagnostics) != (raw_exit_code != 0):
        errors.append('raw analyzer exit is inconsistent with its diagnostics')
    if review.get('schema_version') != 1 or not isinstance(review.get('diagnostics'), list):
        errors.append('invalid reviewed-diagnostic schema')
    if review.get('context') != current:
        errors.append('stale review: source inputs, selected units, flags or Clang version changed')
    reviewed = Counter()
    for row in review.get('diagnostics', []):
        if row.get('disposition') != 'false_positive' or not row.get('rationale', '').strip():
            errors.append('review requires an explicit false-positive disposition and rationale')
        reviewed[tuple(row.get(field) for field in FIELDS)] += 1
    actual = Counter(tuple(row[field] for field in FIELDS) for row in diagnostics)
    for key, count in (actual - reviewed).items():
        errors.append(f'unreviewed diagnostic ({count}): {key}')
    return dict(schema_version=1, status='failed' if errors else 'passed', raw_exit_code=raw_exit_code,
                reviewed_warnings=sum((actual & reviewed).values()), diagnostics=diagnostics,
                absent_reviewed_warnings=sum((reviewed - actual).values()), errors=errors, context=current)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', default='test/fixtures/analyzer_reviews.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--make', default='make')
    parser.add_argument('--sources', required=True)
    parser.add_argument('--flags', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources, flags = shlex.split(args.sources), shlex.split(args.flags)
    clang = subprocess.run(['clang', '--version'], check=True, text=True, capture_output=True).stdout
    before = context(ROOT, sources, flags, clang)
    review = json.loads((ROOT / args.review).read_text(encoding='utf-8'))
    command = shlex.split(args.make) + ['--no-print-directory', '-j1', 'analyze', 'HAS_CUDA=1', 'HAS_HIP=0',
                                       'ANALYZE_SRC=' + ' '.join(sources),
                                       'CFLAGS_COMMON=' + shlex.join(flags), 'ANALYZE_FLAGS=']
    env = {key: value for key, value in os.environ.items() if key not in ('MAKEFLAGS', 'MFLAGS', 'MAKEOVERRIDES')}
    log = output / 'raw.log'
    with log.open('w', encoding='utf-8') as stream:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    report = evaluate(log.read_text(encoding='utf-8'), result.returncode, before, review)
    if context(ROOT, sources, flags, clang) != before:
        report['errors'].append('analysis inputs changed during execution')
        report['status'] = 'failed'
    report.update(command=command, log=dict(path=str(log), sha256=hashlib.sha256(log.read_bytes()).hexdigest()))
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'Reviewed analyzer: {report["status"]}; raw exit {result.returncode}, '
          f'{len(report["diagnostics"])} warnings, {report["reviewed_warnings"]} reviewed. {output}')
    for error in report['errors']:
        print(error)
    return int(report['status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
