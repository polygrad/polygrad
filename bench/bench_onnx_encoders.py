"""Paired CPU encoder measurements, including host input and full output readback.

Original FP32 checkpoints, pinned Tinygrad and ORT; no ONNX graph rewriting.
Compilation/search is reported separately from warm latency. No speed threshold.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    'minilm': ('sentence-transformers/all-MiniLM-L6-v2', '1110a243fdf4706b3f48f1d95db1a4f5529b4d41',
               '6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452'),
    'bert-tiny': ('sentence-transformers-testing/stsb-bert-tiny-onnx', 'da60a9ed87aeb7e57d9b15be5d021b28d153f584',
                  'c26b1b3e2f210b1e49e7a1a5067f6451c7c9039ac5a91848c5d455372fd5d618'),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inputs(length, case):
    ids = (np.arange(length, dtype=np.int64) * (17 + case) + 2000 + case)[None, :]
    mask = np.ones_like(ids)
    ids[0, 0] = 101
    end = length - max(1, length // (4 + case))
    ids[0, end-1] = 102
    ids[0, end:] = 0
    mask[0, end:] = 0
    return dict(input_ids=ids, attention_mask=mask, token_type_ids=np.zeros_like(ids))


def ort_session(path, threads=1):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])


def summarize(rows, lengths, arms):
    summary = []
    for length in lengths:
        for engine, beam, gemm in arms:
            group = [r for r in rows if r['sequence'] == length and r['engine'] == engine and r['beam'] == beam and r['cpu_gemm'] == gemm]
            if not group or any('status' in r for r in group):
                summary.append(dict(sequence=length, engine=engine, beam=beam, cpu_gemm=gemm, status='incomplete'))
                continue
            ratios = []
            for row in group:
                reference = next((r for r in rows if r['sequence'] == length and r['round'] == row['round'] and r['engine'] == 'ort' and 'median_ms' in r), None)
                if reference:
                    ratios.append(row['median_ms']/reference['median_ms'])
            summary.append(dict(sequence=length, engine=engine, beam=beam, cpu_gemm=gemm,
                                median_ms=statistics.median(r['median_ms'] for r in group),
                                paired_ort_ratio=statistics.median(ratios) if len(ratios) == len(group) else None))
    return summary


def worker(args, path):
    cases = [inputs(args.length, i) for i in range(2)]
    with np.load(args.output / f'oracle-{args.length}.npz') as data:
        expected = [data[f'output{i}'] for i in range(2)]
    start = time.perf_counter()
    if args.engine == 'ort':
        import onnxruntime  # Exclude module import, as for the other two engines.
        start = time.perf_counter()
        model = ort_session(path, args.threads)
        def run(values):
            return model.run(['last_hidden_state'], values)[0]
    elif args.engine == 'polygrad':
        sys.path.insert(0, str(ROOT / 'py'))
        import polygrad as pg
        from polygrad import _ffi
        if Path(_ffi.get_lib()._name).resolve() != Path(os.environ['POLY_LIB']).resolve():
            raise RuntimeError('Polygrad loaded a different library than the benchmark selected')
        runtime = pg.Runtime(device='CPU')
        start = time.perf_counter()
        model = runtime.Model.from_onnx(path.read_bytes(), dimensions={'batch_size': 1, 'sequence_length': args.length})
        def run(values):
            return model.call('forward', values)['last_hidden_state']
    else:
        sys.path.insert(0, str(ROOT / 'references/tinygrad_014'))
        from tinygrad import Tensor, TinyJit
        from tinygrad.nn.onnx import OnnxRunner
        start = time.perf_counter()
        model = OnnxRunner(path)
        names = tuple(cases[0])
        @TinyJit
        def call(*tensors):
            return model(dict(zip(names, tensors)))['last_hidden_state'].realize()
        def run(values):
            return call(*(Tensor(values[k]).realize() for k in names)).numpy()
    setup_ms = (time.perf_counter() - start) * 1000
    errors = []
    def checked(i):
        start = time.perf_counter()
        out = run(cases[i % 2])
        elapsed = (time.perf_counter() - start) * 1000
        ref = expected[i % 2]
        np.testing.assert_allclose(out, ref, atol=2e-5, rtol=2e-5)
        errors.append(float(np.max(np.abs(out-ref))))
        return elapsed
    load_before = os.getloadavg()
    first_ms = checked(0)
    warmup_start = time.perf_counter()
    for i in range(args.warmups):
        checked(i)
    warmup_ms = (time.perf_counter() - warmup_start) * 1000
    samples = [checked(i) for i in range(args.samples)]
    row = dict(engine=args.engine, beam=int(os.environ['BEAM']), sequence=args.length,
               setup_ms=setup_ms, first_ms=first_ms, warmup_ms=warmup_ms,
               median_ms=statistics.median(samples), samples_ms=samples, max_error=max(errors),
               peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               address_space_limit_bytes=resource.getrlimit(resource.RLIMIT_AS)[0],
               threads=args.threads, affinity=sorted(os.sched_getaffinity(0)), load_before=load_before, load_after=os.getloadavg())
    if args.engine == 'polygrad':
        model.dispose()
        runtime.dispose()
    Path(args.result).write_text(json.dumps(row, indent=2) + '\n')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, default='minilm')
    parser.add_argument('--directory', type=Path, default=ROOT / 'temp/onnx-encoders')
    parser.add_argument('--output', type=Path, default=ROOT / 'temp/onnx-encoder-matrix')
    parser.add_argument('--fetch', action='store_true')
    parser.add_argument('--lengths', type=int, nargs='+', default=[128, 512])
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--warmups', type=int, default=5)
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--beams', type=int, nargs='+', default=[0, 2])
    parser.add_argument('--cpu-gemm', type=int, choices=[0, 1], nargs='+', default=[0],
                        help='Polygrad physical AVX2 kernel selection; compare 0 1 in alternating pairs')
    parser.add_argument('--engines', nargs='+', choices=['polygrad', 'tinygrad', 'ort'], default=['polygrad', 'tinygrad', 'ort'])
    affinity = parser.add_mutually_exclusive_group()
    affinity.add_argument('--cpu', type=int, help='pin every worker to this allowed logical CPU')
    affinity.add_argument('--cpus', type=int, nargs='+', help='allowed CPU set for all engines; select physical cores explicitly')
    parser.add_argument('--threads', type=int, default=1, help='intra-op worker budget for all engines; inter-op remains one')
    parser.add_argument('--timeout', type=float, default=1800, help='per-arm timeout, including search')
    parser.add_argument('--memory-mib', type=int, default=4096, help='per-process address-space limit, inherited by compilers')
    parser.add_argument('--cache-level', type=int, choices=[0, 1, 2], default=2,
                        help='normal caching in fresh private directories; 0 is an uncached diagnostic')
    parser.add_argument('--engine', help=argparse.SUPPRESS)
    parser.add_argument('--length', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--result', help=argparse.SUPPRESS)
    args = parser.parse_args()
    cpus = {args.cpu} if args.cpu is not None else set(args.cpus) if args.cpus else os.sched_getaffinity(0)
    if not cpus <= os.sched_getaffinity(0) or not 1 <= args.threads <= len(cpus):
        parser.error('threads must fit the requested allowed CPU set')
    os.sched_setaffinity(0, cpus)
    if args.memory_mib < 1 or args.rounds < 1 or args.warmups < 3 or args.samples < 1 or any(n < 4 or n > 512 for n in args.lengths):
        parser.error('need positive rounds/samples/memory, warmups >=3, and sequence lengths in 4..512')
    limit = args.memory_mib * 1024 * 1024
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    repo, revision, sha = MODELS[args.model]
    directory = args.directory / args.model
    if args.fetch:
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo, 'onnx/model.onnx', revision=revision, local_dir=directory)
    path = directory / 'onnx/model.onnx'
    if digest(path) != sha:
        raise ValueError('checkpoint hash mismatch')
    if args.fetch:
        print(f'{repo}@{revision}: {path} SHA256={sha}')
        return 0
    if args.engine:
        return worker(args, path)
    import onnxruntime as ort
    import onnx
    args.output.mkdir(parents=True, exist_ok=True)
    # A partial rerun must not retain the previous run's successful summary.
    (args.output / 'summary.json').unlink(missing_ok=True)
    (args.output / 'results.json').write_text('[]\n')
    model = ort_session(path)
    for length in args.lengths:
        outputs = [model.run(['last_hidden_state'], inputs(length, i))[0] for i in range(2)]
        assert all(np.isfinite(v).all() for v in outputs)
        np.savez(args.output / f'oracle-{length}.npz', output0=outputs[0], output1=outputs[1])
    del model
    lib = Path(os.environ.get('POLY_LIB', ROOT / 'build/libpolygrad.so')).resolve()
    cache_root = Path(tempfile.mkdtemp(prefix='cache-', dir=args.output.resolve()))
    manifest = dict(model=args.model, repository=repo, revision=revision, sha256=sha,
                    python=sys.version, numpy=np.__version__, onnx=onnx.__version__, ort=ort.__version__,
                    tinygrad_commit=subprocess.check_output(['git', '-C', str(ROOT / 'references/tinygrad_014'), 'rev-parse', 'HEAD'], text=True).strip(),
                    library=str(lib), library_sha256=digest(lib), arguments=vars(args),
                    compiler=os.environ.get('CC', 'clang'), cpu_arch=os.environ.get('POLY_CPU_ARCH', 'native'),
                    contract=f'FP32, batch 1, host int64 inputs and full last_hidden_state readback; intra-op thread budget {args.threads} (not observed concurrency); two changing padded inputs; atol=rtol=2e-5',
                    cache_root=str(cache_root), cache_level=args.cache_level,
                    cache='private empty caches per engine; reused across shapes/rounds; first call and warmup include compilation/search/capture')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str) + '\n')
    arms = [(engine, beam, gemm) for engine in args.engines for beam in ([0] if engine == 'ort' else args.beams)
            for gemm in (args.cpu_gemm if engine == 'polygrad' else [0])]
    rows = []
    for length in args.lengths:
        for round_id in range(args.rounds):
            order = arms if round_id % 2 == 0 else list(reversed(arms))
            for engine, beam, gemm in order:
                stem = args.output / f'{length}-{round_id+1}-{engine}-beam{beam}-gemm{gemm}'
                result = stem.with_suffix('.json')
                if result.exists():
                    result.unlink()
                cache = cache_root / f'{engine}-gemm{gemm}'
                env = dict(os.environ, DEV='CPU', POLY_LIB=str(lib), BEAM=str(beam), CACHELEVEL=str(args.cache_level),
                           XDG_CACHE_HOME=str(cache), CACHEDB=str(cache / 'tinygrad/cache.db'),
                           NUM_CPU_THREADS=str(args.threads), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', POLY_CPU_GEMM=str(gemm))
                for key in ('POLY_DEVICE', 'POLY_DEV', 'IGNORE_BEAM_CACHE'):
                    env.pop(key, None)
                cmd = [sys.executable, str(Path(__file__).resolve()), '--model', args.model,
                       '--directory', str(args.directory.resolve()), '--output', str(args.output.resolve()),
                       '--engine', engine, '--length', str(length), '--result', str(result.resolve()),
                       '--warmups', str(args.warmups), '--samples', str(args.samples), '--memory-mib', str(args.memory_mib), '--threads', str(args.threads)]
                print(f'seq={length} pair={round_id+1} {engine} BEAM={beam} GEMM={gemm}: starting', flush=True)
                with stem.with_suffix('.log').open('w') as log:
                    try:
                        process = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=log, timeout=args.timeout)
                        row = json.loads(result.read_text()) if process.returncode == 0 and result.exists() else dict(status='failed', returncode=process.returncode)
                    except subprocess.TimeoutExpired:
                        row = dict(status='timeout')
                row.update(engine=engine, beam=beam, cpu_gemm=gemm, sequence=length, round=round_id+1)
                rows.append(row)
                (args.output / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
                print(f'  {row.get("median_ms", row.get("status"))}', flush=True)
    summary = summarize(rows, args.lengths, arms)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    return int(any('status' in r for r in rows))


if __name__ == '__main__':
    raise SystemExit(main())
