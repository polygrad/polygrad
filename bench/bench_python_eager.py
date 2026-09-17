"""Paired, isolated eager-loop guard against an installed Python package."""

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


PROBE = '''
import json, statistics, sys, time
import polygrad
from polygrad import Tensor, _ffi
a = Tensor([1., 2., 3., 4.]).realize()
def step():
    assert a.mul(2).add(1).relu().sum().item() == 24.
for _ in range(30): step()
samples = []
for _ in range(9):
    start = time.perf_counter()
    for _ in range(200): step()
    samples.append((time.perf_counter() - start) * 5000)
print(json.dumps(dict(median_us=statistics.median(samples), samples_us=samples,
                     python=sys.version, version=polygrad.__version__,
                     package=polygrad.__file__, library=_ffi._lib._name)))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-python', required=True, help='Python with the baseline package installed')
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--max-ratio', type=float, default=1.02)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.rounds < 3 or args.max_ratio <= 0:
        parser.error('use at least three rounds and a positive max-ratio')
    root = Path(__file__).resolve().parents[1]
    # Both children use the same execution settings. The baseline must load its
    # installed package, never the checkout's Python files or native library.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith('POLY') and k not in ('PYTHONPATH', 'DEV', 'DEBUG', 'BEAM', 'NOOPT')}
    env.update(POLY_DEV='CPU', DEBUG='0', BEAM='0', NOOPT='0')
    rows = []
    for round_id in range(args.rounds):
        for label in (('baseline', 'candidate') if round_id % 2 == 0 else ('candidate', 'baseline')):
            child_env = dict(env)
            python = str(Path(args.baseline_python).absolute())
            if label == 'candidate':
                python = sys.executable
                child_env.update(PYTHONPATH=str(root / 'py'), POLY_LIB=str(root / 'build/libpolygrad.so'))
            result = subprocess.run([python, '-c', PROBE], env=child_env, cwd=root,
                                    capture_output=True, text=True, timeout=120)
            if result.returncode:
                raise RuntimeError(f'{label} failed ({result.returncode}): {result.stdout}\n{result.stderr}')
            row = dict(json.loads(result.stdout), label=label, round=round_id)
            rows.append(row)
            print(f"{label} {round_id}: {row['median_us']:.1f} us", flush=True)
        first_pair = rows[:2]
        if any(Path(first_pair[0][key]).resolve() == Path(first_pair[1][key]).resolve()
               for key in ('package', 'library')):
            raise RuntimeError('Baseline and candidate must use independent package/library installations')
    baseline, candidate = (statistics.median(r['median_us'] for r in rows if r['label'] == label)
                           for label in ('baseline', 'candidate'))
    ratio = candidate / baseline
    report = dict(baseline_us=baseline, candidate_us=candidate, ratio=ratio,
                  max_ratio=args.max_ratio, passed=ratio <= args.max_ratio, rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'candidate/baseline: {ratio:.3f} (limit {args.max_ratio:.3f})')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
