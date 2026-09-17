"""Compare the same C smoke driver against the release checkpoint and current core."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.test_release import machine_conditions

# Match bench_smoke.c's existing budgets; change them only through review.
BUDGETS = dict(sum_1024=(.10, 5), movement_1024=(.10, 5), chain_1024=(.10, 5), matmul_16=(.12, 8))


def summarize(rows):
    pairs = {}
    for row in rows:
        pair = pairs.setdefault(row['round'], {})
        label = row['label']
        if label not in ('baseline', 'candidate') or label in pair or set(row['benchmarks']) != set(BUDGETS):
            raise ValueError('duplicate label or missing/unexpected workload')
        for name, (pct, absolute) in BUDGETS.items():
            value = row['benchmarks'][name]
            if not math.isfinite(value['median_us']) or value['median_us'] <= 0:
                raise ValueError('invalid timing')
            if (value['threshold_pct'], value['threshold_abs_us']) != (pct, absolute):
                raise ValueError('benchmark budget changed')
        pair[label] = row['benchmarks']
    if len(pairs) < 5 or sorted(pairs) != list(range(len(pairs))) or any(len(p) != 2 for p in pairs.values()):
        raise ValueError('at least five complete sequential pairs required')
    workloads = {}
    for name, (pct, absolute) in BUDGETS.items():
        baseline = [pairs[i]['baseline'][name]['median_us'] for i in range(len(pairs))]
        candidate = [pairs[i]['candidate'][name]['median_us'] for i in range(len(pairs))]
        ratios = [c / b for b, c in zip(baseline, candidate)]
        # Apply the existing relative/absolute allowance within each adjacent
        # pair. Keep every pair; never average away a slow workload.
        budget_ratios = [c / max(b * (1 + pct), b + absolute) for b, c in zip(baseline, candidate)]
        workloads[name] = dict(baseline_us=statistics.median(baseline), candidate_us=statistics.median(candidate),
                               ratio=statistics.median(ratios), pair_ratios=ratios, budget_ratios=budget_ratios,
                               passed=statistics.median(budget_ratios) <= 1)
    return dict(passed=all(w['passed'] for w in workloads.values()), workloads=workloads)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(work, spec, cc, env):
    # Isolated small source trees avoid changing the checkout or using a stale
    # executable. Both builds use the current benchmark and identical CPU flags.
    work.mkdir(parents=True, exist_ok=False)
    baseline, candidate = work / 'baseline', work / 'candidate'
    baseline.mkdir()
    archive = work / 'baseline.tar'
    subprocess.run(['git', 'archive', '--format=tar', f'--output={archive}', spec['commit'],
                    'Makefile', 'src', 'vendor', 'scripts/wasm_exports.py'], cwd=ROOT, check=True)
    subprocess.run(['tar', '-xf', str(archive), '-C', str(baseline)], check=True)
    candidate.mkdir()
    for directory in ('src', 'vendor'):
        shutil.copytree(ROOT / directory, candidate / directory)
    shutil.copyfile(ROOT / 'Makefile', candidate / 'Makefile')
    (candidate / 'scripts').mkdir()
    shutil.copyfile(ROOT / 'scripts/wasm_exports.py', candidate / 'scripts/wasm_exports.py')
    builds = {}
    for label, tree in [('baseline', baseline), ('candidate', candidate)]:
        (tree / 'bench').mkdir()
        (tree / 'test').mkdir()
        shutil.copyfile(ROOT / 'bench/bench_smoke.c', tree / 'bench/bench_smoke.c')
        command = ['make', '-j1', 'build/bench_smoke', f'CC={cc}', f'PYTHON={sys.executable}',
                   'HAS_CUDA=0', 'HAS_HIP=0', 'HAS_X86=0',
                   'CFLAGS_RELEASE=-std=c11 -D_POSIX_C_SOURCE=200809L -pipe -Isrc -O2']
        with (work / f'{label}-build.log').open('w') as log:
            log.write(shlex.join(command) + '\n')
            log.flush()
            subprocess.run(command, cwd=tree, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        builds[label] = dict(command=command, executable=str(tree / 'build/bench_smoke'),
                             sha256=file_hash(tree / 'build/bench_smoke'),
                             sources={str(p.relative_to(tree)): file_hash(p)
                                      for p in sorted(tree.rglob('*')) if p.suffix in ('.c', '.h')})
    return builds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cc', default='clang')
    parser.add_argument('--rounds', type=int, default=9)
    args = parser.parse_args()
    if args.rounds < 5:
        parser.error('at least five pairs required')
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    spec = json.loads((ROOT / 'test/fixtures/c_performance_baseline.json').read_text())
    env = {k: v for k, v in os.environ.items() if not k.startswith('POLY') and k not in
           ('DEV', 'DEBUG', 'BEAM', 'NOOPT', 'MAKEFLAGS', 'MFLAGS', 'MAKEOVERRIDES')}
    env.update(POLY_DEV='CPU', DEV='CPU', DEBUG='0', BEAM='0', NOOPT='0', CC=args.cc)
    # Preserve the caller's compiler scratch location, but not execution policy.
    if 'POLY_TMPDIR' in os.environ:
        env['POLY_TMPDIR'] = os.environ['POLY_TMPDIR']
    report = dict(status='running', baseline=spec, rows=[], driver_sha256=file_hash(ROOT / 'bench/bench_smoke.c'),
                  compiler=subprocess.check_output(shlex.split(args.cc) + ['--version'], text=True))
    def save():
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    save()
    try:
        builds = prepare(args.output.parent / 'builds', spec, args.cc, env)
        report['builds'] = builds
        for i in range(args.rounds):
            for label in (('baseline', 'candidate') if i % 2 == 0 else ('candidate', 'baseline')):
                command = [builds[label]['executable'], '--samples', '7', '--iters', '300', '--warmup', '30']
                before = machine_conditions()
                result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
                after = machine_conditions()
                (args.output.parent / f'{i:02d}-{label}.log').write_text(result.stdout + result.stderr)
                if result.returncode:
                    raise RuntimeError(f'{label} round {i}: exit {result.returncode}')
                row = dict(json.loads(result.stdout), round=i, label=label, command=command,
                           machine_conditions=dict(before=before, after=after))
                report['rows'].append(row)
                save()
        report.update(summarize(report['rows']))
        report['status'] = 'passed' if report['passed'] else 'failed'
    except Exception as exc:
        report.update(status='failed', error=str(exc))
        raise
    finally:
        save()
    for name, result in report['workloads'].items():
        print(f"{name}: paired candidate/baseline {result['ratio']:.3f}; passed={result['passed']}")
    return int(not report['passed'])


if __name__ == '__main__':
    raise SystemExit(main())
