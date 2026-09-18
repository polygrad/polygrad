#!/usr/bin/env python3
"""Cross-frontend PGIR+safetensors product acceptance gate."""

from __future__ import annotations

import argparse
import base64
import ctypes
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'py'))
os.environ.setdefault('POLY_LIB', str(ROOT / 'build' / 'libpolygrad.so'))

from polygrad.model import Model, OPTIM_ADAM, ROLE_PARAM, ROLE_AUX  # noqa: E402
from polygrad.models import MLP, Graph  # noqa: E402
from polygrad import create, _ffi  # noqa: E402


def check_components(model):
    oracle = json.loads((ROOT / 'test/fixtures/model_components_expected.json').read_text())
    for case in reversed(oracle['cases']):
        outputs = model.forward(tokens=np.array(case['tokens'], np.int32))
        np.testing.assert_allclose(outputs['prediction'], case['prediction'], atol=2e-5)
        np.testing.assert_allclose(outputs['mean'], case['mean'], atol=2e-5)


def check_qwen(model):
    oracle = json.loads((ROOT / 'test/fixtures/qwen3.json').read_text())
    assert model.entrypoints()[0]['inputs'] == ['x']
    output = model.forward(x=np.array(oracle['tokens'], np.int32))['output']
    np.testing.assert_allclose(output, oracle['logits'], atol=3e-5, rtol=3e-5)


def vision_cases():
    cases = json.loads((ROOT / 'test/fixtures/vision.json').read_text())['cases']
    return [cases[i] for i in (0, 1, 2, 4)]


def check_vision(model, case):
    inputs = {k:np.array(v, np.int32 if k == 'input_ids' else np.float32) for k,v in case['inputs'].items()}
    outputs = model.forward(**inputs)
    for name, expected in case['outputs'].items():
        np.testing.assert_allclose(outputs[name], expected, atol=5e-5, rtol=5e-4)


def export_c_lstm(work: Path) -> None:
    """Exercise C model construction without a Python Tensor/model recipe."""
    lib = ctypes.CDLL(_ffi._lib._name)
    ptr, i64 = ctypes.c_void_p, ctypes.c_int64
    signatures = {
        'poly_model_new': (ptr, [ptr, ptr]),
        'poly_model_input': (ptr, [ptr, ctypes.c_char_p, _ffi.PolyDType, ctypes.POINTER(i64), ctypes.c_int]),
        'poly_model_lstm_cell': (ctypes.c_int, [ptr, ctypes.c_char_p, ptr, ptr, ptr, ctypes.c_int, ctypes.c_int, ctypes.c_bool,
                                              ctypes.POINTER(ptr), ctypes.POINTER(ptr)]),
        'poly_model_output': (ctypes.c_int, [ptr, ctypes.c_char_p, ptr]),
        'poly_model_entrypoint': (ctypes.c_int, [ptr, ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p), ctypes.c_int,
                                               ctypes.POINTER(ctypes.c_char_p), ctypes.c_int, ptr]),
        'poly_model_build': (ctypes.c_int, [ptr, ptr]),
    }
    for name, (result, arguments) in signatures.items():
        function = getattr(lib, name)
        function.restype, function.argtypes = result, arguments
    ctx = _ffi._lib.poly_ctx_new()
    handle = lib.poly_model_new(ctx, None)
    model = Model._from_handle(handle, ctx)
    try:
        shape = (i64 * 2)(1, 2)
        dtype = _ffi.PolyDType()
        assert _ffi._lib.poly_dtype_by_id(_ffi._lib.poly_dtype_id_by_name(b'float32'), ctypes.byref(dtype))
        x, h, c = [lib.poly_model_input(handle, name, dtype, shape, 2) for name in (b'x', b'h', b'c')]
        hidden, cell = ptr(), ptr()
        assert lib.poly_model_lstm_cell(handle, b'cell', x, h, c, 2, 2, False, ctypes.byref(hidden), ctypes.byref(cell)) == 0
        assert lib.poly_model_output(handle, b'hidden', hidden) == 0
        assert lib.poly_model_output(handle, b'cell_state', cell) == 0
        inputs = (ctypes.c_char_p * 3)(b'x', b'h', b'c')
        outputs = (ctypes.c_char_p * 2)(b'hidden', b'cell_state')
        assert lib.poly_model_entrypoint(handle, b'forward', inputs, 3, outputs, 2, None) == 0
        assert lib.poly_model_build(handle, None) == 0
        model.write_buffer('cell.weight_ih', np.full((8, 2), .25, np.float32))
        model.write_buffer('cell.weight_hh', np.full((8, 2), .125, np.float32))
        (work / 'c-lstm.bundle').write_bytes(model.save_bundle(include_optimizer=False))
        first = model.forward(x=np.array([[1, 2]], np.float32), h=np.zeros((1, 2), np.float32), c=np.zeros((1, 2), np.float32))
        second = model.forward(x=np.array([[1, 2]], np.float32), h=first['hidden'], c=first['cell_state'])
        (work / 'c-lstm-expected.json').write_text(json.dumps({k: v.reshape(-1).tolist() for k, v in second.items()}))
    finally:
        model.free()
        _ffi._lib.poly_ctx_destroy(ctx)


def safetensor_names(data: bytes) -> set[str]:
    header_len = int.from_bytes(data[:8], 'little')
    return set(json.loads(data[8:8 + header_len]).keys()) - {'__metadata__'}


def train_step(model: Model) -> float:
    return model.train_step(
        x=np.array([1.0, 2.0], dtype=np.float32),
        y=np.array([3.0], dtype=np.float32),
    )


def compare_checkpoint(source: Model, restored: Model, names: set[str], atol: float) -> None:
    for name in sorted(names):
        source_i, restored_i = source.find_buf(name), restored.find_buf(name)
        if source_i < 0 or restored_i < 0:
            raise AssertionError(f'missing checkpoint row {name!r}')
        np.testing.assert_allclose(
            restored.buf_data(restored_i), source.buf_data(source_i), rtol=0.0, atol=atol,
            err_msg=f'checkpoint row {name}',
        )


def check_artifact_imports(work: Path, device: str) -> None:
    """Every generated fixture crosses private, cold and populated contexts.

    Fresh import/export canonicalizes allocation IDs without dropping fields.
    This codec-based oracle complements, not replaces, the independent direct
    kernel/value tests and exact C identity/slot assertions.
    """
    def private(data, bundle, weights=None):
        buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
        weight_buf = (ctypes.c_uint8 * len(weights)).from_buffer_copy(weights) if weights is not None else None
        ptr = (_ffi._lib.poly_model_from_bundle(buf, len(data)) if bundle else
               _ffi._lib.poly_model_from_ir(buf, len(data), weight_buf, len(weights) if weights is not None else 0))
        return Model._from_handle(ptr)

    def canonical(ir):
        model = private(ir, False)
        try: return model.export_ir()
        finally: model.dispose()

    def execute(model):
        # Capture C stderr, not Python's sys.stderr wrapper. Compare exact
        # source/signatures only between cold contexts on the same backend.
        saved = os.dup(2)
        previous = os.environ.get('POLY_DUMP_KERNELS')
        outputs = {}
        try:
            os.environ['POLY_DUMP_KERNELS'] = '1'
            with tempfile.TemporaryFile(mode='w+') as trace:
                os.dup2(trace.fileno(), 2)
                try:
                    for ep in model.entrypoints():
                        inputs = {name: np.ones(model.buf_shape(model.find_buf(name)),
                                                dtype=model.buf_dtype(model.find_buf(name)))
                                  for name in ep['inputs']}
                        outputs[ep['name']] = model.call(ep['name'], **inputs)
                finally:
                    os.dup2(saved, 2)
                    trace.seek(0)
                    source = trace.read()
                    # Preserve diagnostics if compilation/execution raises.
                    print(source, end='', file=sys.stderr)
            kernels = set(re.findall(r'^=== KERNEL (\S+) ===\n(.*?)\n=== END ===$', source, re.M | re.S))
            return outputs, kernels
        finally:
            os.close(saved)
            if previous is None: os.environ.pop('POLY_DUMP_KERNELS', None)
            else: os.environ['POLY_DUMP_KERNELS'] = previous

    artifacts = sorted(p for p in work.iterdir() if p.suffix in {'.bundle', '.pgb', '.pgir'})
    for path in artifacts:
        data, bundle = path.read_bytes(), path.suffix != '.pgir'
        # Raw PGIR is executable evidence only together with its checkpoint.
        weights = None if bundle else path.with_suffix('.safetensors').read_bytes()
        cold, shared = create(device=device), create(device=device)
        models = []
        try:
            live = shared.Tensor([19.0])
            primer = Model(lambda x: x+37, inputs={'x': shared.Tensor.empty(7)})
            models.append(primer)
            def load(data, runtime):
                return Model.load(data, runtime=runtime) if bundle else Model.from_ir(data, weights, runtime=runtime)
            reference = private(data, bundle, weights)
            models.append(reference)
            expected = canonical(reference.export_ir())
            assert canonical(expected) == expected, f'{path.name}: canonical IR not stable'
            copies = []
            for runtime in (cold, shared, shared):
                copies.append(load(data, runtime=runtime))
                models.append(copies[-1])
            a, b, c = copies
            for model in (a, b, c):
                assert model.entrypoints() == reference.entrypoints(), path.name
                ir = model.export_ir()
                assert canonical(ir) == expected, f'{path.name}: import changed canonical IR'
                for i in range(reference.buf_count):
                    name = reference.buf_name(i)
                    j = model.find_buf(name)
                    assert j >= 0 and model.buf_role(j) == reference.buf_role(i), (path.name, name)
                    assert model.buf_shape_bounds(j) == reference.buf_shape_bounds(i), (path.name, name)
                    if reference.buf_role(i) in (ROLE_PARAM, ROLE_AUX):
                        np.testing.assert_array_equal(model.read_buffer(name), reference.read_buffer(name))
            if bundle or weights is not None:
                state = {c.buf_name(i): c.read_buffer(c.buf_name(i)) for i in range(c.buf_count)
                         if c.buf_role(i) in (ROLE_PARAM, ROLE_AUX)}
                for name, original in state.items():
                    if not original.size: continue
                    changed = original.copy()
                    changed.flat[0] = 0 if original.flat[0] != 0 else 1
                    b.write_buffer(name, changed)
                    # Mutate every state family, including RNG and optimizer
                    # buffers. Checking all siblings also catches cross-name aliasing.
                    for other, expected_state in state.items():
                        np.testing.assert_array_equal(c.read_buffer(other), expected_state)
                    np.testing.assert_array_equal(a.read_buffer(name), original)
                    np.testing.assert_array_equal(reference.read_buffer(name), original)
                b.dispose()
                shared.clear_schedule_cache()
                shared.collect()
                for name, expected_state in state.items():
                    np.testing.assert_array_equal(c.read_buffer(name), expected_state)
                reference.place(device)
                expected_outputs, expected_kernels = execute(reference)
                for model in (a, c):
                    actual_outputs, actual_kernels = execute(model)
                    assert actual_kernels == expected_kernels, f'{path.name}:{device}: kernel signature/hash/source changed'
                    assert actual_outputs.keys() == expected_outputs.keys()
                    for ep, expected_rows in expected_outputs.items():
                        assert actual_outputs[ep].keys() == expected_rows.keys()
                        for name, expected_output in expected_rows.items():
                            np.testing.assert_array_equal(actual_outputs[ep][name], expected_output,
                                                          err_msg=f'{path.name}:{device}:{ep}:{name}')
                    for name in state:
                        np.testing.assert_array_equal(model.read_buffer(name), reference.read_buffer(name))
                if device == 'cpu':
                    assert expected_kernels, f'{path.name}: no compiled CPU evidence'
            np.testing.assert_array_equal(live.numpy(), [19])
            np.testing.assert_array_equal(primer.forward(x=np.arange(7, dtype=np.float32))['output'],
                                          np.arange(7, dtype=np.float32)+37)
        finally:
            for model in reversed(models): model.dispose()
            shared.dispose()
            cold.dispose()
    print(f'model import matrix {device}: {len(artifacts)} artifacts pass (private/cold/shared, canonical IR, state isolation, execution/kernels)')


def run_core(work: Path, core: str, source_after: Model, expected_loss: float) -> None:
    subprocess.run(
        ['node', str(ROOT / 'js' / 'test' / 'model_interchange.js'), str(work), core],
        cwd=ROOT, check=True,
    )
    result = json.loads((work / f'javascript-{core}-result.json').read_text())
    stateful = Model.load((work / f'javascript-{core}-stateful.bundle').read_bytes())
    try:
        expected = json.loads((work / 'stateful-expected.json').read_text())
        np.testing.assert_allclose(result['statefulLoss'], expected['loss'], rtol=1e-5)
        for name, values in expected['state'].items():
            np.testing.assert_allclose(stateful.read_buffer(name), values, rtol=1e-5, atol=1e-6)
    finally:
        stateful.dispose()
    variable = Model.load((work / f'javascript-{core}-variable.bundle').read_bytes())
    try:
        assert variable.buf_shape_bounds(variable.find_buf('x')) == ((1,32),(2,2))
        for n in (17,3,11):
            x = np.arange(n*2,dtype=np.float32).reshape(n,2)
            np.testing.assert_array_equal(variable.forward(x=x)['prediction'], x*2)
    finally:
        variable.dispose()
    qwen = Model.load((work / f'javascript-{core}-qwen.bundle').read_bytes())
    try:
        check_qwen(qwen)
    finally:
        qwen.dispose()
    components = Model.load((work / f'javascript-{core}-components.bundle').read_bytes())
    try:
        check_components(components)
    finally:
        components.dispose()
    for case in vision_cases():
        model = Model.load((work / f"javascript-{core}-{case['name']}.bundle").read_bytes())
        try:
            check_vision(model, case)
        finally:
            model.dispose()
    composed = Model.from_bundle((work / f'javascript-{core}-graph.bundle').read_bytes())
    try:
        np.testing.assert_array_equal(composed.forward(x=np.array([[1, 2]], np.float32))['prediction'], [[28, 61]])
        assert composed.param_count == 1
    finally:
        composed.free()
    atol = 0.0 if core == 'native' else 1e-6
    np.testing.assert_allclose(result['trainLoss'], expected_loss, rtol=0.0, atol=atol)

    resumed_weights = (work / f'javascript-{core}-resumed.safetensors').read_bytes()
    resumed = Model.from_ir((work / 'python-train.pgir').read_bytes(), resumed_weights)
    try:
        compare_checkpoint(source_after, resumed, safetensor_names(resumed_weights), atol)
    finally:
        resumed.free()

    inference = Model.from_ir(
        (work / f'javascript-{core}-inference.pgir').read_bytes(),
        (work / f'javascript-{core}-inference.safetensors').read_bytes(),
    )
    try:
        output = inference.forward(x=np.array([1.25, -0.5], dtype=np.float32))['output']
        np.testing.assert_allclose(
            output.reshape(-1), np.asarray(result['inferenceOutput'], dtype=np.float32),
            rtol=0.0, atol=atol,
        )
    finally:
        inference.free()

    custom = Model.from_bundle((work / f'javascript-{core}-custom.bundle').read_bytes())
    try:
        np.testing.assert_array_equal(custom.forward(x=np.array([3], np.float32))['prediction'], [7])
        np.testing.assert_array_equal(custom.call('double', x=np.array([3], np.float32))['twice'], [14])
        custom.write_buffer('weight', np.array([4], np.float32))
        np.testing.assert_array_equal(custom.read_buffer('tied'), [4])
        np.testing.assert_array_equal(custom.forward(x=np.array([3], np.float32))['prediction'], [13])
    finally:
        custom.free()


def export_custom(work: Path) -> None:
    rt = create(device='interp', logical='always')
    w = rt.Tensor([2.0])
    offset = rt.Tensor([1.0]).is_param_(False)
    def net(x):
        pred = x*w + offset
        return {'prediction': pred, 'twice': pred*2}
    model = Model.from_callable(net, inputs={'x': rt.Tensor.empty(1)},
                        params={'weight': w, 'tied': w, 'offset': offset}, entrypoints=[
                            {'name': 'forward', 'inputs': ['x'], 'outputs': ['prediction']},
                            {'name': 'double', 'inputs': ['x'], 'outputs': ['twice']}])
    try:
        (work / 'python-custom.bundle').write_bytes(model.save_bundle(include_optimizer=False))
    finally:
        model.free()
        rt.dispose()


def export_variable(work: Path) -> None:
    from polygrad.tensor import Variable
    rt = create(device='interp',logical='always')
    n = Variable('interchange_batch',1,32,_ctx=rt._ctx)
    model = Model(lambda x:{'prediction':x*2}, inputs={'x':rt.Tensor.empty(n.bind(17),2)})
    try:
        (work / 'python-variable.bundle').write_bytes(model.save(include_optimizer=False))
    finally:
        model.dispose(); rt.dispose()


def export_stateful(work: Path) -> None:
    from polygrad import Tensor
    from polygrad.helpers import TRAINING
    Tensor.manual_seed(123)
    weight = Tensor([1.0])
    counter = Tensor([0.0]).is_param_(False)
    def author(x):
        if TRAINING.value: counter.assign(counter+1)
        return (x*weight).dropout(0.5)
    model = Model(author, inputs={'x':Tensor.empty(16)}, targets={'y':Tensor.empty(16)},
                  params={'weight':weight,'counter':counter},loss=lambda out,y:(out-y).square().mean())
    try:
        model.set_optimizer('adam',lr=0.01)
        io = {'x':np.ones(16,np.float32),'y':np.zeros(16,np.float32)}
        model.train_step(**io)
        (work / 'python-stateful.bundle').write_bytes(model.save())
        loss = model.train_step(**io)
        state = {model.buf_name(i):model.read_buffer(model.buf_name(i)).tolist()
                 for i in range(model.buf_count) if model.buf_role(i) in (ROLE_PARAM,ROLE_AUX)}
        (work / 'stateful-expected.json').write_text(json.dumps({'loss':loss,'state':state}))
    finally:
        model.dispose()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cores', default='native,wasm')
    parser.add_argument('--output', type=Path, help='keep generated artifacts in a new directory')
    args = parser.parse_args()
    cores = [value.strip() for value in args.cores.split(',') if value.strip()]
    if any(core not in {'native', 'wasm'} for core in cores):
        raise SystemExit('--cores accepts native,wasm')

    (ROOT / 'temp').mkdir(exist_ok=True)
    if args.output is not None:
        work = args.output.resolve()
        work.mkdir(parents=True, exist_ok=False)
    else:
        work = Path(tempfile.mkdtemp(prefix='model-interchange.', dir=ROOT / 'temp'))
    completed = False
    subprocess.run([sys.executable, str(ROOT / 'py/examples/linear_export.py'), str(work / 'linear.pgb')],
                   check=True, capture_output=True, text=True)
    for core in cores:
        result = subprocess.run(['node', str(ROOT / 'js/examples/linear_predict.js'), str(work / 'linear.pgb')],
                                env={**os.environ, 'POLY_CORE': core}, check=True, capture_output=True, text=True)
        np.testing.assert_allclose(json.loads(result.stdout), [11, 14, 17, 20, 23], atol=1e-4)
        print(f'linear example {core}: pass (named prediction, path save/load)')
    export_c_lstm(work)
    export_custom(work)
    export_variable(work)
    export_stateful(work)
    for case in vision_cases():
        model = Model.from_hf(config_json=json.dumps(case['config']),
                              weight_bytes_list=[base64.b64decode(case['weights'])], max_batch=2)
        try:
            check_vision(model, case)
            (work / f"python-{case['name']}.bundle").write_bytes(model.save())
        finally:
            model.dispose()
    oracle = json.loads((ROOT / 'test/fixtures/qwen3.json').read_text())
    qwen = Model.from_gguf(base64.b64decode(oracle['gguf']), max_seq_len=4)
    try:
        check_qwen(qwen)
        (work / 'python-qwen.bundle').write_bytes(qwen.save())
    finally:
        qwen.dispose()
    components = Graph((ROOT / 'test/fixtures/model_components.json').read_bytes())
    try:
        oracle = json.loads((ROOT / 'test/fixtures/model_components_expected.json').read_text())
        for name, values in [('nodes.embedding.weight', oracle['table']), ('nodes.head.weight', oracle['linear'])]:
            components.write_buffer(name, np.array(values, np.float32))
        check_components(components)
        (work / 'python-components.bundle').write_bytes(components.save(include_optimizer=False))
    finally:
        components.dispose()
    graph = Graph((ROOT / 'test/fixtures/model_definition.json').read_bytes())
    try:
        graph.write_buffer('modules.shared.weight', np.array([1, 2, 3, 4], np.float32))
        (work / 'python-graph.bundle').write_bytes(graph.save_bundle(include_optimizer=False))
    finally:
        graph.free()
    source = MLP({
        'layers': [2, 1], 'activation': 'none', 'bias': False,
        'loss': 'mse', 'batch_size': 1, 'seed': 7,
    })
    try:
        source.set_optimizer(OPTIM_ADAM, lr=0.05)
        for _ in range(3):
            train_step(source)
        (work / 'python-train.pgir').write_bytes(source.export_ir())
        checkpoint = source.export_weights()
        (work / 'python-train.safetensors').write_bytes(checkpoint)
        expected_loss = train_step(source)

        for core in cores:
            run_core(work, core, source, expected_loss)
            print(f'model interchange {core}: pass (Adam continuation and custom authoring both directions)')
        for device in ('interp', 'cpu'):
            check_artifact_imports(work, device)
        completed = True
    finally:
        source.free()
        if completed and args.output is None:
            shutil.rmtree(work)
        else:
            print(f'model interchange artifacts: {work}')


if __name__ == '__main__':
    main()
