#!/usr/bin/env python3
"""Cross-frontend PGIR+safetensors product acceptance gate."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'py'))
os.environ.setdefault('POLY_LIB', str(ROOT / 'build' / 'libpolygrad.so'))

from polygrad.model import Model, OPTIM_ADAM  # noqa: E402
from polygrad.models import MLP, Graph  # noqa: E402
from polygrad import create, _ffi  # noqa: E402


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
    model = Model(handle, _ctx=ctx)
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


def run_core(work: Path, core: str, source_after: Model, expected_loss: float) -> None:
    subprocess.run(
        ['node', str(ROOT / 'js' / 'test' / 'model_interchange.js'), str(work), core],
        cwd=ROOT, check=True,
    )
    result = json.loads((work / f'javascript-{core}-result.json').read_text())
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
    model = Model.trace(net, inputs={'x': rt.Tensor.empty(1)},
                        state={'weight': w, 'tied': w, 'offset': offset}, entrypoints=[
                            {'name': 'forward', 'inputs': ['x'], 'outputs': ['prediction']},
                            {'name': 'double', 'inputs': ['x'], 'outputs': ['twice']}])
    try:
        (work / 'python-custom.bundle').write_bytes(model.save_bundle(include_optimizer=False))
    finally:
        model.free()
        rt.dispose()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cores', default='native,wasm')
    args = parser.parse_args()
    cores = [value.strip() for value in args.cores.split(',') if value.strip()]
    if any(core not in {'native', 'wasm'} for core in cores):
        raise SystemExit('--cores accepts native,wasm')

    (ROOT / 'temp').mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='model-interchange.', dir=ROOT / 'temp'))
    export_c_lstm(work)
    export_custom(work)
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
    finally:
        source.free()
        shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
    main()
