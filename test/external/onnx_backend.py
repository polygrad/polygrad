"""Run the official ONNX node corpus unchanged against Polygrad and pinned Tinygrad.

Each case runs in a subprocess: an unsupported kernel must not lose the rest of
the report. Tinygrad exclusions are read from the pin, not copied or counted as
passes. This is an opt-in diagnostic lane, not an all-passing release gate.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
PIN = ROOT / 'references/tinygrad_014'
REFERENCE = PIN / 'test/external/external_test_onnx_backend.py'
sys.path[:0] = [str(ROOT / 'py'), str(PIN)]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def polygrad_backend():
    from onnx.backend.base import Backend, BackendRep
    import polygrad as pg

    class Rep(BackendRep):
        def __init__(self, model):
            self.proto = model
            initializers = {v.name for v in model.graph.initializer}
            self.inputs = [v for v in model.graph.input if v.name not in initializers]
            self.runtime = pg.Runtime(device=os.environ.get('DEV', 'CPU'))
            self.model = None
            self.dimensions = None

        def run(self, inputs, **kwargs):
            if len(inputs) != len(self.inputs):
                raise ValueError('ONNX input count mismatch')
            dimensions = {}
            for declaration, value in zip(self.inputs, inputs):
                if not declaration.type.HasField('tensor_type'):
                    raise NotImplementedError('Polygrad Model inputs must be tensors')
                for dim, size in zip(declaration.type.tensor_type.shape.dim, value.shape):
                    if dim.dim_param:
                        if dim.dim_param in dimensions and dimensions[dim.dim_param] != size:
                            raise ValueError('conflicting ONNX named dimensions')
                        dimensions[dim.dim_param] = size
            # Exercise the advertised specialization contract; never rewrite
            # graph versions, attributes, dtypes, or expected outputs.
            if self.model is None or self.dimensions != dimensions:
                if self.model is not None:
                    self.model.dispose()
                    self.model = None
                self.model = self.runtime.Model.from_onnx(self.proto.SerializeToString(), dimensions=dimensions)
                self.dimensions = dimensions
            result = self.model.call('forward', dict(zip((v.name for v in self.inputs), inputs)))
            return tuple(result[v.name] for v in self.proto.graph.output)

        def close(self):
            if self.model is not None:
                self.model.dispose()
            self.runtime.dispose()

    class PolygradBackend(Backend):
        instances = []

        @classmethod
        def prepare(cls, model, device='CPU', **kwargs):
            result = Rep(model)
            cls.instances.append(result)
            return result

        @classmethod
        def supports_device(cls, device):
            # ONNX's CPU/CUDA label is independent of DEV, as in the pin.
            return device == 'CPU'

    return PolygradBackend


def worker(args):
    from onnx.backend.test import BackendTest
    if args.engine == 'tinygrad':
        backend = runpy.run_path(str(REFERENCE))['TinygradBackend']
    else:
        backend = polygrad_backend()
    case_type = BackendTest(backend, __name__).test_cases['OnnxBackendNodeModelTest']
    result = unittest.TestResult()
    start = time.monotonic()
    try:
        case_type(args.case).run(result)
    finally:
        for instance in getattr(backend, 'instances', []):
            instance.close()
    status, detail = 'passed', ''
    if result.failures:
        status, detail = 'mismatch', result.failures[0][1]
    elif result.errors:
        status, detail = 'error', result.errors[0][1]
    elif result.skipped:
        status, detail = 'skipped', result.skipped[0][1]
    payload = dict(status=status, detail=detail, seconds=time.monotonic()-start)
    Path(args.result).write_text(json.dumps(payload))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', choices=['polygrad', 'tinygrad', 'both'], default='both')
    parser.add_argument('--filter', default='.*', help='regex against official CPU node case names')
    parser.add_argument('--exclusions', choices=['tinygrad', 'none'], default='tinygrad')
    parser.add_argument('--output', type=Path, default=ROOT / 'temp/onnx-backend')
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--timeout', type=float, default=60)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--case', help=argparse.SUPPRESS)
    parser.add_argument('--result', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.case:
        return worker(args)
    if args.jobs < 1 or args.timeout <= 0:
        parser.error('jobs and timeout must be positive')
    import onnx
    from onnx.backend.test.loader import load_model_tests
    pin = runpy.run_path(str(REFERENCE))['backend_test']
    patterns = sorted(p.pattern for p in pin._exclude_patterns)
    cases = []
    for case in load_model_tests(kind='node'):
        name = case.name + '_cpu'
        if not re.search(args.filter, name):
            continue
        path = Path(case.model_dir) / 'model.onnx'
        model = onnx.load(path, load_external_data=False)
        opsets = {v.domain: v.version for v in model.opset_import}
        cases.append(dict(name=name, ir=model.ir_version, opsets=opsets, sha256=digest(path),
                          tinygrad_exclusions=[p for p in patterns if re.search(p, name)]))
    if not cases:
        parser.error('filter selected no node cases')
    engines = ['polygrad', 'tinygrad'] if args.engine == 'both' else [args.engine]
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = dict(onnx=onnx.__version__, numpy=__import__('numpy').__version__,
                    python=sys.version, device=os.environ.get('DEV', 'CPU'), filter=args.filter,
                    reference_sha256=digest(REFERENCE), importer_sha256=digest(ROOT / 'src/loaders/onnx_loader.c'),
                    tinygrad_commit=subprocess.check_output(['git', '-C', str(PIN), 'rev-parse', 'HEAD'], text=True).strip(),
                    exclusions=args.exclusions, exclusion_patterns=patterns, cases=cases)
    lib = Path(os.environ.get('POLY_LIB', ROOT / 'build/libpolygrad.so')).resolve()
    manifest.update(library=str(lib), library_sha256=digest(lib))
    if args.list:
        (args.output / 'inventory.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print(f'{len(cases)} node cases; {sum(bool(c["tinygrad_exclusions"]) for c in cases)} match pinned exclusions')
        return 0
    # A partial rerun must not retain the previous run's successful summary.
    (args.output / 'summary.json').unlink(missing_ok=True)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    def execute(task):
        engine, case = task
        row = dict(engine=engine, **case)
        if args.exclusions == 'tinygrad' and case['tinygrad_exclusions']:
            return dict(row, status='excluded', seconds=0)
        stem = args.output / (engine + '-' + case['name'])
        result = stem.with_suffix('.json')
        if result.exists():
            result.unlink()  # A crashed rerun must not inherit an old success.
        cmd = [sys.executable, str(Path(__file__).resolve()), '--engine', engine, '--case', case['name'], '--result', str(result)]
        with stem.with_suffix('.log').open('w') as log:
            try:
                process = subprocess.run(cmd, stdout=log, stderr=log, timeout=args.timeout, cwd=ROOT)
                if process.returncode or not result.exists():
                    row.update(status='crash', returncode=process.returncode)
                else:
                    row.update(json.loads(result.read_text()))
                    standard = case['opsets'].get('', case['opsets'].get('ai.onnx', 0))
                    if engine == 'polygrad' and row['status'] == 'error' and (
                        not 3 <= case['ir'] <= 10 or not 13 <= standard <= 23
                    ) and ('supported ONNX IR versions' in row['detail'] or 'standard opset in' in row['detail']):
                        row['status'] = 'version_rejected'
            except subprocess.TimeoutExpired:
                row.update(status='timeout', seconds=args.timeout)
        return row

    rows = []
    with (args.output / 'results.jsonl').open('w') as stream, ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for row in pool.map(execute, ((engine, case) for case in cases for engine in engines)):
            rows.append(row)
            stream.write(json.dumps(row) + '\n')
            stream.flush()
            print(f'{len(rows)}/{len(cases)*len(engines)} {row["engine"]} {row["name"]}: {row["status"]}', flush=True)
    summary = {engine: dict(Counter(r['status'] for r in rows if r['engine'] == engine)) for engine in engines}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    return int(any(r['status'] not in ('passed', 'excluded') for r in rows))


if __name__ == '__main__':
    raise SystemExit(main())
