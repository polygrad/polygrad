"""C ONNX import, ordinary Model execution and portable round trips."""
import base64
import json
from pathlib import Path

import numpy as np
import pytest
import polygrad as pg

CASES = json.loads((Path(__file__).resolve().parents[2] / 'test/fixtures/onnx.json').read_text())['cases']


@pytest.mark.parametrize('case', CASES, ids=lambda c: c['name'])
def test_onnx_model_and_bundle(case):
    with pg.Runtime(device='CPU') as rt:
        data = bytearray(base64.b64decode(case['onnx']))
        external = {k: bytearray(base64.b64decode(v)) for k, v in case['external'].items()}
        model = rt.Model.from_onnx(data, dimensions=case['dimensions'], external_data=external)
        assert type(model) is pg.Model
        data[:] = bytes(len(data))
        for v in external.values():
            v[:] = bytes(len(v))
        try:
            restored = rt.Model.load(model.save())
            try:
                inputs = {k: np.array(v['values'], np.float32).reshape(v['shape']) for k, v in case['inputs'].items()}
                for current in (model, restored):
                    actual = current.call('forward', inputs)
                    for name, ref in case['outputs'].items():
                        np.testing.assert_allclose(actual[name], np.array(ref['values']).reshape(ref['shape']), atol=2e-5, rtol=2e-5)
                    if case['name'].startswith('if_'):
                        opposite = -inputs['x']
                        expected = opposite + 2 if opposite.sum() > 0 else opposite * 3
                        np.testing.assert_array_equal(current.call('forward', {'x': opposite})['y'], expected)
            finally:
                restored.dispose()
        finally:
            model.dispose()


def _varint(n):
    out = bytearray()
    while n > 127:
        out.append((n & 127) | 128)
        n >>= 7
    return bytes(out) + bytes([n])


def _field(tag, value):
    return _varint(tag * 8 + 2) + _varint(len(value)) + value


def _fields(data):
    pos = 0
    def integer():
        nonlocal pos
        result, shift = 0, 0
        while True:
            b = data[pos]
            pos += 1
            result |= (b & 127) << shift
            if b < 128:
                return result
            shift += 7
    while pos < len(data):
        key = integer()
        if key & 7 == 0:
            yield key >> 3, integer()
        else:
            n = integer() if key & 7 == 2 else 4 if key & 7 == 5 else 8
            value = data[pos:pos+n]
            pos += n
            yield key >> 3, value


def _graph(data):
    return next(value for tag, value in _fields(data) if tag == 7)


def _replace_graph(data, graph):
    return b'\x08\x08' + _field(7, graph) + _field(8, b'\x0a\x00\x10\x11')


def test_onnx_import_rejections_leave_runtime_usable():
    raw = base64.b64decode(CASES[0]['onnx'])
    graph = _graph(raw)
    initializer = next(value for tag, value in _fields(graph) if tag == 5)
    node = next(value for tag, value in _fields(graph) if tag == 1)
    # Repeated fields, unsupported operators/attributes and invalid integer
    # encodings are tested without making ONNX a package/test dependency.
    invalid = [
        (raw.replace(b'Gemm', b'Nope'), 'Nope'),
        (raw + _field(7, graph), 'duplicate field 7'),
        (_replace_graph(raw, graph + _field(5, initializer)), 'duplicate'),
        (_replace_graph(raw, graph.replace(_field(1, node), _field(1, node + _field(5,
            _field(1, b'unknown') + b'\x18\x01\xa0\x01\x02')), 1)), 'unsupported attribute'),
        (raw[:-1], 'malformed'),
        (raw + b'\x80' * 10 + b'\x01', 'malformed'),
        (raw.replace(b'\x10\x11', b'\x10\x63'), 'opset'),
    ]
    with pg.Runtime(device='INTERP') as rt:
        live = rt.Tensor([2.0]).realize()
        for data, message in invalid:
            with pytest.raises(ValueError, match=message):
                rt.Model.from_onnx(data, dimensions={'batch': 2})
            assert (live + 1).item() == 3
        with pytest.raises(ValueError, match="supply dimension 'batch'"):
            rt.Model.from_onnx(raw)
        for value in (0, -1, 1.5, '2', True):
            with pytest.raises(ValueError, match='positive integer'):
                rt.Model.from_onnx(raw, dimensions={'batch': value})


def test_onnx_external_files_are_explicit_and_range_checked():
    case = next(c for c in CASES if c['name'] == 'external')
    raw = base64.b64decode(case['onnx'])
    payload = base64.b64decode(case['external']['weights.bin'])
    with pg.Runtime(device='INTERP') as rt:
        with pytest.raises(ValueError, match='was not supplied'):
            rt.Model.from_onnx(raw)
        with pytest.raises(ValueError, match='external byte range'):
            rt.Model.from_onnx(raw, external_data={'weights.bin': payload[:-1]})
        for name in ('unrelated.bin', '../weights.bin'):
            with pytest.raises(ValueError, match='was not supplied'):
                rt.Model.from_onnx(raw, external_data={name: payload})


def test_onnx_rejects_duplicate_raw_initializer_storage():
    raw = base64.b64decode(CASES[0]['onnx'])
    graph = _graph(raw)
    weight = next(value for tag, value in _fields(graph) if tag == 5)
    duplicated = weight + _field(9, b'bad')
    graph = graph.replace(_field(5, weight), _field(5, duplicated), 1)
    with pytest.raises(ValueError, match='duplicate field 9'):
        pg.Model.from_onnx(_replace_graph(raw, graph), dimensions={'batch': 2}, device='INTERP')


def test_onnx_imports_have_independent_mutable_weight_storage():
    case = CASES[0]
    raw = base64.b64decode(case['onnx'])
    with pg.Runtime(device='CPU') as rt:
        first = rt.Model.from_onnx(raw, dimensions=case['dimensions'])
        second = rt.Model.from_onnx(raw, dimensions=case['dimensions'])
        try:
            before = second.read_buffer('w').copy()
            assert not any(row['trainable'] for row in first.bindings())
            first.write_buffer('w', np.zeros_like(before))
            np.testing.assert_array_equal(second.read_buffer('w'), before)
            assert not hasattr(first, 'generate')
        finally:
            first.dispose()
            second.dispose()


def _int_field(tag, value):
    return _varint(tag * 8) + _varint(value)


def _info(name, dims, dtype=1):
    shape = b''.join(_field(1, _int_field(1, dim)) for dim in dims)
    return _field(1, name.encode()) + _field(2, _field(1, _int_field(1, dtype) + _field(2, shape)))


def _small_graph(op, inputs, outputs):
    node = b''.join(_field(1, name.encode()) for name, _, _ in inputs)
    node += _field(2, outputs[0][0].encode()) + _field(4, op.encode())
    graph = _field(1, node)
    graph += b''.join(_field(11, _info(*row)) for row in inputs)
    graph += b''.join(_field(12, _info(*row)) for row in outputs)
    return _int_field(1, 8) + _field(7, graph) + _field(8, _int_field(2, 17))


def test_onnx_rejects_implicit_dtype_promotion_and_reverse_norm_broadcast():
    with pg.Runtime(device='INTERP') as rt:
        # An integer Exp would silently become float under Tensor promotion.
        bad_exp = _small_graph('Exp', [('x', [2], 6)], [('y', [2], 1)])
        with pytest.raises(ValueError, match='dtype'):
            rt.Model.from_onnx(bad_exp)
        bad_norm = _small_graph('LayerNormalization', [('x', [1, 4], 1), ('scale', [2, 4], 1)], [('y', [2, 4], 1)])
        with pytest.raises(ValueError, match='shape/attribute'):
            rt.Model.from_onnx(bad_norm)
        bad_broadcast = _small_graph('Add', [('x', [2, 3], 1), ('b', [2], 1)], [('y', [2, 3], 1)])
        with pytest.raises(ValueError, match='broadcast'):
            rt.Model.from_onnx(bad_broadcast)


def test_onnx_integer_division_truncates_and_preserves_dtype():
    raw = _small_graph('Div', [('x', [4], 6), ('d', [4], 6)], [('y', [4], 6)])
    with pg.Runtime(device='INTERP') as rt:
        model = rt.Model.from_onnx(raw)
        try:
            out = model.call('forward', {'x': np.array([-7, 7, -7, 7], np.int32),
                                         'd': np.array([3, -3, -3, 3], np.int32)})['y']
            np.testing.assert_array_equal(out, [-2, -2, 2, 2])
            assert out.dtype == np.int32
        finally:
            model.dispose()


def test_onnx_path_loading(tmp_path):
    path = tmp_path / 'model.onnx'
    path.write_bytes(base64.b64decode(CASES[0]['onnx']))
    model = pg.Model.from_onnx(path, dimensions=CASES[0]['dimensions'], device='INTERP')
    model.dispose()
