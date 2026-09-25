"""Experimental UOp AVX2 matmul+bias+GELU; no production kernel selection.

Compare identical 4x24 tiles with/without packed panels. Every output is checked
against float64 and ORT. Timings are isolated kernels, not full Model latency.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
import onnxruntime as ort
from scipy.special import erf

ROOT = Path(__file__).resolve().parents[1]


def load(path, name, nargs):
    library = ctypes.CDLL(str(path.resolve()))
    fn = getattr(library, name)
    fn.argtypes, fn.restype = [ctypes.c_void_p] * nargs, None
    return fn


def reference(a, b, bias):
    m, k = a.shape
    n = b.shape[1]
    nodes = [h.make_node('MatMul', ['x', 'w'], ['mm']), h.make_node('Add', ['mm', 'b'], ['z']),
             h.make_node('Mul', ['z', 'scale'], ['scaled']), h.make_node('Erf', ['scaled'], ['erf']),
             h.make_node('Add', ['erf', 'one'], ['e']), h.make_node('Mul', ['z', 'half'], ['hz']),
             h.make_node('Mul', ['hz', 'e'], ['out'])]
    initializers = [nh.from_array(v, name) for name, v in (
        ('w', b), ('b', bias), ('scale', np.array(2**-.5, 'f')), ('one', np.array(1, 'f')), ('half', np.array(.5, 'f')))]
    graph = h.make_graph(nodes, 'gemm', [h.make_tensor_value_info('x', 1, [m, k])],
                         [h.make_tensor_value_info('out', 1, [m, n])], initializers)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(h.make_model(graph, ir_version=8, opset_imports=[h.make_opsetid('', 14)]).SerializeToString(),
                                  options, providers=['CPUExecutionProvider'])
    return session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'temp/cpu-gemm')
    parser.add_argument('--build', type=Path, default=ROOT / 'build/bench-cpu-gemm')
    parser.add_argument('--cpu', type=int)
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--shapes', nargs='+', default=['4,7,24', '128,384,1536', '128,1536,384'])
    args = parser.parse_args()
    if args.samples < 1 or args.rounds < 1:
        parser.error('samples and rounds must be positive')
    flags = Path('/proc/cpuinfo').read_text()
    if ' avx2 ' not in flags or ' fma ' not in flags:
        raise RuntimeError('this experimental kernel requires AVX2 and FMA')
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'results.json').write_text('[]\n')
    adapter = ctypes.CDLL(str((args.build / 'adapter.so').resolve()))
    custom = adapter.bench_custom_run
    custom.argtypes, custom.restype = [ctypes.c_int]*4 + [ctypes.c_void_p]*4, ctypes.c_int
    # Unsupported tails must decline before reading any pointer.
    for shape in [(3, 7, 24), (4, 7, 25), (4, 0, 24)]:
        assert custom(*shape, 1, None, None, None, None) == -1
    manifest = dict(arguments=vars(args), affinity=sorted(os.sched_getaffinity(0)),
                    compiler=subprocess.check_output(['clang', '--version'], text=True),
                    head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    library_sha256=hashlib.sha256((ROOT/'build/libpolygrad.so').read_bytes()).hexdigest(),
                    source_sha256=hashlib.sha256((ROOT/'bench/kernels/cpu_gemm.c').read_bytes()).hexdigest(),
                    contract='FP32 independent column lanes; explicit FMA, sequential K; atol=rtol=2e-5; single core; packing timed separately')
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2, default=str)+'\n')
    results = []
    for shape in args.shapes:
        m, k, n = map(int, shape.split(','))
        fns = []
        for mode in range(6):
            stem = args.output / f'{m}-{k}-{n}-{mode}'
            subprocess.run([str(args.build/'generator'), str(m), str(k), str(n), str(mode), str(stem.with_suffix('.c'))], check=True)
            subprocess.run(['clang', '-O2', '-march=native', '-fPIC', '-shared', str(stem.with_suffix('.c')), '-lm',
                            '-o', str(stem.with_suffix('.so'))], check=True)
            asm = subprocess.check_output(['objdump', '-d', str(stem.with_suffix('.so'))], text=True)
            (stem.with_suffix('.asm')).write_text(asm)
            if mode in (0, 1, 3, 4):
                assert any('vfmadd' in line and 'ymm' in line for line in asm.splitlines())
            fns.append(load(stem.with_suffix('.so'), 'pack_weights' if mode == 2 else 'gelu' if mode == 5 else 'avx2_gemm',
                            2 if mode in (2, 5) else 4))
        rng = np.random.default_rng(42)
        a = rng.normal(0, .2, (m, k)).astype('f')
        b = rng.normal(0, .1, (k, n)).astype('f')
        bias = rng.normal(0, .1, n).astype('f')
        packed = np.empty((n//24, k, 24), 'f')
        out = np.empty((m, n), 'f')
        intermediate = np.empty_like(out)
        def pack():
            fns[2](packed.ctypes.data, b.ctypes.data)
        def run(mode):
            if mode == 2:
                pack()
            fns[mode != 0](out.ctypes.data, bias.ctypes.data, a.ctypes.data, (b if mode == 0 else packed).ctypes.data)
            return out
        def matmul(mode):
            fns[3+mode](intermediate.ctypes.data, bias.ctypes.data, a.ctypes.data,
                        (packed if mode else b).ctypes.data)
        def epilogue():
            fns[5](out.ctypes.data, intermediate.ctypes.data)
        def split(mode, prepare=False):
            if prepare: pack()
            matmul(mode)
            epilogue()
            return out
        errors = []
        # Changing weights catches stale preparation and repeated-call state.
        for case in range(2):
            if case:
                b *= -.75
                a += .03125
            pack()
            np.testing.assert_array_equal(packed, b.reshape(k, n//24, 24).transpose(1, 0, 2))
            session = reference(a, b, bias)
            expected = session.run(None, {'x': a})[0]
            z = a.astype('d') @ b.astype('d') + bias
            double = .5*z*(1+erf(z/np.sqrt(2)))
            for mode in range(3):
                actual = run(mode)
                np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
                np.testing.assert_allclose(actual, double, atol=2e-5, rtol=2e-5)
                errors.append(float(np.max(np.abs(actual-double))))
            for mode in range(2):
                np.testing.assert_allclose(split(mode), expected, atol=2e-5, rtol=2e-5)
                np.testing.assert_allclose(intermediate, z, atol=2e-5, rtol=2e-5)
                assert custom(m,k,n,mode,out.ctypes.data,bias.ctypes.data,a.ctypes.data,(b if mode == 0 else packed).ctypes.data) == 0
                np.testing.assert_allclose(out, expected, atol=2e-5, rtol=2e-5)
        arms = [('unpacked', lambda: run(0)), ('prepacked', lambda: run(1)),
                ('pack_and_run', lambda: run(2)), ('packing', pack),
                ('matmul_unpacked', lambda: matmul(0)), ('matmul_prepacked', lambda: matmul(1)),
                ('gelu', epilogue), ('split_unpacked', lambda: split(0)),
                ('split_prepacked', lambda: split(1)), ('split_pack_and_run', lambda: split(1, True)),
                ('ort', lambda: session.run(None, {'x': a}))]
        for _, fn in arms:
            for _ in range(5): fn()
        for round_id in range(args.rounds):
            for name, fn in (arms if round_id % 2 == 0 else list(reversed(arms))):
                samples = []
                load_before = os.getloadavg()
                for _ in range(args.samples):
                    start = time.perf_counter_ns()
                    fn()
                    samples.append((time.perf_counter_ns()-start)/1e6)
                row = dict(shape=[m,k,n], arm=name, round=round_id+1, median_ms=statistics.median(samples),
                           samples_ms=samples, load_before=load_before, load_after=os.getloadavg(), max_error=max(errors))
                results.append(row)
                (args.output/'results.json').write_text(json.dumps(results, indent=2)+'\n')
                print(json.dumps({k:v for k,v in row.items() if k != 'samples_ms'}), flush=True)


if __name__ == '__main__':
    main()
