"""Paired eager, training and JIT/readback guards against an installed package."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import urllib.request


ROOT = Path(__file__).resolve().parents[1]


PROBE = '''
import json, statistics, sys, time
import polygrad, numpy
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
                     prefix=sys.prefix, numpy_version=numpy.__version__,
                     package=polygrad.__file__, library=_ffi._lib._name)))
'''


TRAINING_PROBE = '''
import json, statistics, sys, time
import polygrad, numpy as np
from polygrad import Tensor, Context, _ffi
from polygrad.nn.optim import Adam
x = Tensor(np.linspace(-1, 1, 32, dtype=np.float32).reshape(8, 4)).is_param_(False).realize()
y = Tensor(np.linspace(-.5, .5, 16, dtype=np.float32).reshape(8, 2)).is_param_(False).realize()
params = [Tensor(np.linspace(-.2, .3, 32, dtype=np.float32).reshape(4, 8)).realize(),
          Tensor(np.full(8, .1, np.float32)).realize(),
          Tensor(np.linspace(-.3, .2, 16, dtype=np.float32).reshape(8, 2)).realize(),
          Tensor(np.full(2, .1, np.float32)).realize()]
opt = Adam(params, lr=.001)
def step():
    opt.zero_grad()
    loss = (((x @ params[0] + params[1]).relu() @ params[2] + params[3]) - y).square().mean()
    loss.backward()
    opt.step()
    return loss.item()
with Context(TRAINING=1):
    initial = step()
    for _ in range(14): step()
    samples = []
    for _ in range(7):
        start = time.perf_counter()
        for _ in range(20): final = step()
        samples.append((time.perf_counter() - start) * 50000)
# Pinned Tinygrad CPU and each Polygrad logical policy agree on this trajectory.
# Check outside timing; a skipped optimizer/backward must not be a fast pass.
np.testing.assert_allclose([initial, final], [0.07240158319473267, 0.0011362460209056735],
                          rtol=1e-4, atol=1e-7)
print(json.dumps(dict(median_us=statistics.median(samples), samples_us=samples,
                     initial_loss=initial, final_loss=final,
                     python=sys.version, version=polygrad.__version__,
                     prefix=sys.prefix, numpy_version=np.__version__,
                     package=polygrad.__file__, library=_ffi._lib._name)))
'''

JIT_READBACK_PROBE = '''
import json, statistics, sys, time
import polygrad, numpy as np
from polygrad import Tensor, TinyJit, Context, _ffi
from polygrad.nn import Linear
from polygrad.nn.optim import Adam
from polygrad.nn.state import get_parameters
Tensor.manual_seed(0)
l1, l2 = Linear(16, 32), Linear(32, 1)
opt = Adam(get_parameters([l1, l2]), lr=.001)
x, y = Tensor.randn(8, 16).realize(), Tensor.randn(8, 1).realize()
@TinyJit
def step(x, y):
    with Context(TRAINING=1):
        opt.zero_grad()
        loss = (l2(l1(x).relu()) - y).square().mean()
        loss.backward()
        opt.step()
        return loss.realize()
initial = step(x, y).item()
for _ in range(19): step(x, y).item()
samples = []
for _ in range(150):
    start = time.perf_counter()
    final = step(x, y).item()
    samples.append((time.perf_counter() - start) * 1e6)
# Include host readback: omitting it hid redundant residency collection.
# Pinned Tinygrad v0.14.0 agrees on this seeded Adam trajectory.
np.testing.assert_allclose([initial, final], [0.7308312058448792, 7.073858341755113e-06],
                          rtol=1e-4, atol=1e-7)
print(json.dumps(dict(median_us=statistics.median(samples), samples_us=samples,
                     initial_loss=initial, final_loss=final,
                     python=sys.version, version=polygrad.__version__,
                     prefix=sys.prefix, numpy_version=np.__version__,
                     package=polygrad.__file__, library=_ffi._lib._name)))
'''

WORKLOADS = dict(eager=PROBE, training=TRAINING_PROBE, jit_readback=JIT_READBACK_PROBE)


def summarize(rows, max_ratio):
    if not math.isfinite(max_ratio) or max_ratio <= 0:
        raise ValueError('budget must be finite and positive')
    pairs = {}
    for row in rows:
        label, value = row['label'], row['median_us']
        pair = pairs.setdefault(row['round'], {})
        if label not in ('baseline', 'candidate') or label in pair or not math.isfinite(value) or value <= 0:
            raise ValueError('duplicate label or invalid timing')
        pair[label] = value
    if len(pairs) < 5 or sorted(pairs) != list(range(len(pairs))) or any(len(pair) != 2 for pair in pairs.values()):
        raise ValueError('at least five complete sequential pairs required')
    ratios = [pairs[i]['candidate'] / pairs[i]['baseline'] for i in range(len(pairs))]
    # Pair adjacent processes before aggregating: ratio-of-medians can mask
    # regressions when machine speed changes between pairs. Never drop/retry a pair.
    ratio = statistics.median(ratios)
    return dict(baseline_us=statistics.median(pair['baseline'] for pair in pairs.values()),
                candidate_us=statistics.median(pair['candidate'] for pair in pairs.values()),
                ratio=ratio, pair_ratios=ratios, max_ratio=max_ratio, passed=ratio <= max_ratio, rows=rows)


def summarize_workloads(rows, max_ratio):
    names = sorted({row['workload'] for row in rows})
    if not names or any(name not in WORKLOADS for name in names):
        raise ValueError('unknown or missing workload')
    reports = {name: summarize([row for row in rows if row['workload'] == name], max_ratio) for name in names}
    # Never average workloads: fast getters cannot compensate for slow training.
    return dict(passed=all(report['passed'] for report in reports.values()), workloads=reports)


def validate_pair(pair):
    baseline, candidate = (next(row for row in pair if row['label'] == label) for label in ('baseline', 'candidate'))
    for key in ('package', 'library'):
        if Path(baseline[key]).resolve() == Path(candidate[key]).resolve():
            raise ValueError('Baseline and candidate must use independent package/library installations')
        if not Path(baseline[key]).resolve().is_relative_to(Path(baseline['prefix']).resolve()):
            raise ValueError('baseline must load from its isolated environment')
    if (Path(candidate['package']).resolve() != ROOT / 'py/polygrad/__init__.py' or
            Path(candidate['library']).resolve() != ROOT / 'build/libpolygrad.so'):
        raise ValueError('candidate loaded outside checkout')
    if any(baseline[key] != candidate[key] for key in ('python', 'numpy_version')):
        raise ValueError('baseline and candidate require the same Python and NumPy versions')


def verify_archive(artifact, spec):
    if hashlib.sha256(artifact.read_bytes()).hexdigest() != spec['sha256']:
        raise ValueError('baseline archive differs from pinned published artifact')


def prepare_baseline(work, archive=None):
    # Reuse the installed-package harness, never a checkout or editable baseline.
    sys.path.insert(0, str(ROOT))
    from test.test_package_install import clean_environment, run
    import numpy
    spec = json.loads((ROOT / 'test/fixtures/python_performance_baseline.json').read_text())
    work.mkdir(parents=True, exist_ok=False)
    artifact = work / spec['filename']
    if archive:
        verify_archive(archive, spec)
        shutil.copyfile(archive, artifact)
    else:
        with urllib.request.urlopen(spec['url'], timeout=60) as response, artifact.open('wb') as output:
            shutil.copyfileobj(response, output)
        verify_archive(artifact, spec)
    install_env = clean_environment()
    run([sys.executable, '-m', 'venv', str(work / 'venv')], work, install_env, work / 'venv.log')
    python = work / 'venv/bin/python'
    run([str(python), '-I', '-m', 'pip', '--isolated', 'install', '--no-cache-dir',
         '--index-url', 'https://pypi.org/simple', '--timeout', '30', '--retries', '2',
         f'numpy=={numpy.__version__}', str(artifact)], work, install_env, work / 'install.log')
    (work / 'provenance.json').write_text(json.dumps(dict(spec, numpy_version=numpy.__version__,
                                                       python=sys.version), indent=2) + '\n')
    return python, spec['version']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    baseline = parser.add_mutually_exclusive_group(required=True)
    baseline.add_argument('--baseline-python', help='Python with the baseline package installed')
    baseline.add_argument('--prepare-baseline', action='store_true', help='Install the pinned published sdist into a fresh venv')
    parser.add_argument('--baseline-sdist', type=Path, help='Use a local copy of the hash-pinned sdist')
    parser.add_argument('--rounds', type=int, default=9)
    parser.add_argument('--workload', choices=['all', *WORKLOADS], default='all')
    parser.add_argument('--max-ratio', type=float, default=1.02)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.rounds < 5 or not math.isfinite(args.max_ratio) or args.max_ratio <= 0:
        parser.error('use at least five rounds and a finite positive max-ratio')
    if args.baseline_sdist and not args.prepare_baseline:
        parser.error('--baseline-sdist requires --prepare-baseline')
    root = ROOT
    # Both children use the same execution settings. The baseline must load its
    # installed package, never the checkout's Python files or native library.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith('POLY') and k not in ('PYTHONPATH', 'PYTHONHOME', 'DEV', 'DEBUG', 'BEAM', 'NOOPT')}
    env.update(POLY_DEV='CPU', DEBUG='0', BEAM='0', NOOPT='0')
    expected_version = None
    if args.prepare_baseline:
        args.baseline_python, expected_version = prepare_baseline(args.output.resolve().parent / 'baseline', args.baseline_sdist)
    rows = []
    for workload in WORKLOADS if args.workload == 'all' else [args.workload]:
        for round_id in range(args.rounds):
            for label in (('baseline', 'candidate') if round_id % 2 == 0 else ('candidate', 'baseline')):
                child_env = dict(env)
                python = str(Path(args.baseline_python).absolute())
                if label == 'candidate':
                    python = sys.executable
                    child_env.update(PYTHONPATH=str(root / 'py'), POLY_LIB=str(root / 'build/libpolygrad.so'))
                result = subprocess.run([python, '-c', WORKLOADS[workload]], env=child_env, cwd=root,
                                        capture_output=True, text=True, timeout=120)
                if result.returncode:
                    raise RuntimeError(f'{label}/{workload} failed ({result.returncode}): {result.stdout}\n{result.stderr}')
                row = dict(json.loads(result.stdout), workload=workload, label=label, round=round_id)
                rows.append(row)
                print(f"{workload} {label} {round_id}: {row['median_us']:.1f} us", flush=True)
            validate_pair(rows[-2:])
            if expected_version and next(row['version'] for row in rows[-2:] if row['label'] == 'baseline') != expected_version:
                raise ValueError('installed baseline version differs from pinned version')
    report = summarize_workloads(rows, args.max_ratio)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    for name, result in report['workloads'].items():
        print(f"{name} median paired candidate/baseline: {result['ratio']:.3f} (limit {args.max_ratio:.3f})")
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
