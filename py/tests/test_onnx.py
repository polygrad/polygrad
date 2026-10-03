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
                    if case['name'] == 'scalar_constants':
                        assert {r['name'] for r in current.bindings() if r['role'] == 0} == {'w'}
                        current.write_buffer('w', np.array(3., np.float32))
                        np.testing.assert_array_equal(current.call('forward', inputs)['weighted'], inputs['x']**2 * 3)
            finally:
                restored.dispose()
        finally:
            model.dispose()


def test_onnx_scatter_preserves_caller_tensor():
    case = next(c for c in CASES if c['name'] == 'scatter_nd_add')
    original = np.array(case['inputs']['x']['values'], np.float32).reshape(3, 4)
    with pg.Runtime(device='CPU') as rt:
        model = rt.Model.from_onnx(base64.b64decode(case['onnx']))
        try:
            x = rt.Tensor(original).realize()
            for _ in range(2):
                out = model.call('forward', {'x': x})['y']
                np.testing.assert_array_equal(out.numpy(), np.array(case['outputs']['y']['values']).reshape(3, 4))
                np.testing.assert_array_equal(x.numpy(), original)
        finally:
            model.dispose()


def test_onnx_inferred_outputs_still_validate_explicit_dimensions():
    case = next(c for c in CASES if c['name'] == 'inferred_output_dimension')
    raw = base64.b64decode(case['onnx'])
    with pg.Runtime(device='CPU') as rt:
        model = rt.Model.from_onnx(raw, dimensions={'output_rows': 3, 'output_columns': 2})
        model.dispose()
        for dims in ({'output_rows': 2}, {'output_columns': 3}):
            with pytest.raises(ValueError, match='disagrees with declared shape'):
                rt.Model.from_onnx(raw, dimensions=dims)
        # A repeated symbol is one dimension, not a separate wildcard per axis.
        dim = _field(1, _field(2, b'same_extent'))
        output = _field(1, b'y') + _field(2, _field(1, b'\x08\x01' + _field(2, dim + dim)))
        graph = b''.join(_field(tag, output if tag == 12 else value) for tag, value in _fields(_graph(raw)))
        with pytest.raises(ValueError, match='disagrees with declared shape'):
            rt.Model.from_onnx(_replace_graph(raw, graph))


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


def test_onnx_large_graph_stays_bounded():
    raw = base64.b64decode(CASES[0]['onnx'])
    def graph(count):
        nodes = []
        for i in range(count):
            node = _field(1, b'x' if i == 0 else f'identity{i-1}'.encode())
            node += _field(2, f'identity{i}'.encode()) + _field(4, b'Identity')
            nodes.append(_field(1, node))
        return _replace_graph(raw, _graph(raw) + b''.join(nodes))
    with pg.Runtime(device='INTERP') as rt:
        model = rt.Model.from_onnx(graph(2050), dimensions={'batch': 2})
        model.dispose()
        with pytest.raises(ValueError, match='limit|unsupported graph'):
            rt.Model.from_onnx(graph(4100), dimensions={'batch': 2})


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


def _small_graph(op, inputs, outputs, opset=17, attributes=None):
    node = b''.join(_field(1, name.encode()) for name, _, _ in inputs)
    node += _field(2, outputs[0][0].encode()) + _field(4, op.encode())
    node += b''.join(_field(5, _field(1, key.encode()) + _int_field(3, value) + _int_field(20, 2))
                     for key, value in (attributes or {}).items())
    graph = _field(1, node)
    graph += b''.join(_field(11, _info(*row)) for row in inputs)
    graph += b''.join(_field(12, _info(*row)) for row in outputs)
    return _int_field(1, 8) + _field(7, graph) + _field(8, _int_field(2, opset))


def test_onnx_range_rejects_runtime_lengths_and_invalid_constants():
    with pg.Runtime(device='INTERP') as rt:
        runtime = _small_graph('Range', [(k, [], 1) for k in ('start', 'end', 'step')], [('y', [3], 1)])
        with pytest.raises(ValueError, match='runtime data'):
            rt.Model.from_onnx(runtime)
        for end, step in [(3., 0.), (3., np.nan), (np.inf, 1.), (1e30, 1.)]:
            graph = b''
            for name, value in [('start', 0.), ('end', end), ('step', step)]:
                tensor = _int_field(2, 1) + _field(9, np.array(value, np.float32).tobytes())
                attr = _field(1, b'value') + _field(5, tensor) + _int_field(20, 4)
                graph += _field(1, _field(2, name.encode()) + _field(4, b'Constant') + _field(5, attr))
            node = b''.join(_field(1, k.encode()) for k in ('start', 'end', 'step'))
            node += _field(2, b'y') + _field(4, b'Range')
            raw = _replace_graph(b'', graph + _field(1, node) + _field(12, _info('y', [3], 1)))
            with pytest.raises(ValueError, match='Range'):
                rt.Model.from_onnx(raw)


@pytest.mark.parametrize('dtype,code,value', [
    ('float32', 1, -0.), ('float32', 1, np.inf), ('float32', 1, np.nan),
    ('float64', 11, 1.0000000000000002), ('float16', 10, 2**-24),
    ('int8', 3, -128), ('uint8', 2, 255), ('int16', 5, -32768),
    ('uint16', 4, 65535), ('int32', 6, -2**31), ('uint32', 12, 2**32-1),
    ('int64', 7, -2**63+1), ('uint64', 13, 2**64-1), ('bool', 9, True),
    ('uint16', 16, 0x4010),  # bfloat16 2.25, read back through Cast(float32).
])
def test_onnx_scalar_literal_dtype_and_bundle(dtype, code, value):
    expected = np.array(value, dtype=dtype)
    tensor = _int_field(2, code) + _field(9, expected.tobytes())
    attr = _field(1, b'value') + _field(5, tensor) + _int_field(20, 4)
    node = _field(2, b'y') + _field(4, b'Constant') + _field(5, attr)
    graph = _field(1, node)
    output = 'y'
    if code == 16:
        cast = _field(1, b'y') + _field(2, b'z') + _field(4, b'Cast')
        cast += _field(5, _field(1, b'to') + _int_field(3, 1) + _int_field(20, 2))
        graph += _field(1, cast)
        output, code, expected = 'z', 1, np.array(2.25, np.float32)
    raw = _replace_graph(b'', graph + _field(12, _info(output, [], code)))
    with pg.Runtime(device='CPU') as rt:
        model = rt.Model.from_onnx(raw)
        restored = rt.Model.load(model.save())
        try:
            for current in (model, restored):
                actual = current.call('forward')[output]
                assert actual.shape == expected.shape and actual.dtype == expected.dtype
                if np.isnan(expected) and np.issubdtype(expected.dtype, np.floating):
                    assert np.isnan(actual)
                else:
                    assert actual.tobytes() == expected.tobytes()
        finally:
            restored.dispose()
            model.dispose()


def test_onnx_attention_rejects_reference_disagreement_and_invalid_signatures():
    with pg.Runtime(device='INTERP') as rt:
        for shapes, version, error in [
            ([[1,4,2,4],[1,2,3,4],[1,2,3,4]], 23, 'multi-KV-head GQA'),
            ([[1,2,2,4],[1,2,3,4],[1,2,3,4]], 21, 'shape/attribute'),
            ([[1,2,2,4],[],[1,2,3,4]], 23, 'shape/attribute'),
            ([[1,2,2,4],[1,2,3,5],[1,2,3,4]], 23, 'shape/attribute'),
        ]:
            raw = _small_graph('Attention', [(key, dims, 1) for key,dims in zip(['q','k','v'],shapes)],
                               [('y',[1,2,2,4],1)], opset=version)
            with pytest.raises(ValueError, match=error):
                rt.Model.from_onnx(raw)
        inputs = [(key, [1,2,2,4], 1) for key in ['q','k','v']]
        inputs += [('mask',[2,5],1),('pk',[1,2,3,4],1),('pv',[1,2,3,4],1)]
        raw = _small_graph('Attention', inputs, [('y',[1,2,2,4],1)], opset=23, attributes={'is_causal':1})
        with pytest.raises(ValueError, match='causal Attention with past'):
            rt.Model.from_onnx(raw)


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
