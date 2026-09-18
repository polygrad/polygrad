"""Shared C vision models: HF oracle, ownership and portable artifacts."""
import base64
import json
import sys
import struct
from pathlib import Path
import numpy as np
import pytest
import polygrad as pg

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'test'))
from vision_fixture import load_vision_cases

CASES = load_vision_cases()


def _replace_weight(blob, name, shape, values, *, remove=False):
    size = struct.unpack('<Q', blob[:8])[0]
    header = json.loads(blob[8:8 + size])
    data = blob[8 + size:]
    chunks = {key: data[entry['data_offsets'][0]:entry['data_offsets'][1]]
              for key, entry in header.items() if key != '__metadata__'}
    if remove:
        del header[name]
        del chunks[name]
    else:
        raw = np.asarray(values, dtype='<f4').tobytes()
        header[name] = dict(dtype='F32', shape=shape)
        chunks[name] = raw
    # Keep a valid, gap-free safetensors file so rejection tests reach the
    # Model importer, rather than merely failing the container decoder.
    data = bytearray()
    for key, chunk in chunks.items():
        header[key]['data_offsets'] = [len(data), len(data) + len(chunk)]
        data.extend(chunk)
    encoded = json.dumps(header, separators=(',', ':')).encode()
    encoded += b' ' * (-len(encoded) % 8)
    return struct.pack('<Q', len(encoded)) + encoded + data


@pytest.mark.parametrize('mutation', ['unexpected', 'registers', 'empty_register_shape', 'missing', 'duplicate'])
def test_vision_import_rejects_invalid_weights(mutation):
    case = CASES[-1]  # DINOv3 with zero registers; its empty HF parameter is allowed.
    weights = base64.b64decode(case['weights'])
    if mutation == 'unexpected':
        weights = _replace_weight(weights, 'unexpected.weight', [1], [1.])
    elif mutation == 'registers':
        weights = _replace_weight(weights, 'embeddings.register_tokens', [1, 1, 16], np.zeros(16))
    elif mutation == 'empty_register_shape':
        weights = _replace_weight(weights, 'embeddings.register_tokens', [0], [])
    elif mutation == 'missing':
        weights = _replace_weight(weights, 'norm.weight', [], [], remove=True)
    shards = [weights, weights] if mutation == 'duplicate' else [weights]
    with pg.create(device='INTERP') as rt:
        expected = {'unexpected': r'invalid.*unexpected.weight',
                    'registers': r'invalid.*embeddings.register_tokens',
                    'empty_register_shape': r'invalid.*embeddings.register_tokens',
                    'missing': r'missing vision weight.*norm.weight', 'duplicate': 'duplicate'}
        with pytest.raises(RuntimeError, match=expected[mutation]):
            pg.Model.from_hf(config_json=json.dumps(case['config']), weight_bytes_list=shards,
                             max_batch=2, runtime=rt)
        np.testing.assert_array_equal(rt.Tensor([1.]).numpy(), [1.])


def test_vision_fixture_rejects_payload_drift():
    from vision_fixture import expand_case
    packed = json.loads((Path(__file__).resolve().parents[2] / 'test/fixtures/vision.json').read_text())['cases'][0]
    packed['weights']['sha256'] = '0' * 64
    with pytest.raises(AssertionError, match='checkpoint formula drift'):
        expand_case(packed)


@pytest.mark.parametrize('case', [c for c in CASES if not c['config'].get('use_gated_mlp')])
def test_vision_default_config_matches_explicit(case):
    # Defaults are from Transformers 5.3.0 configuration classes, not C output.
    defaults = {'CLIP': (1e-5, 'quick_gelu'), 'ViT': (1e-12, 'gelu'),
                'DINOv2': (1e-6, 'gelu'), 'DINOv3': (1e-5, 'gelu')}
    config = json.loads(json.dumps(case['config']))
    encoders = [config['vision_config'], config['text_config']] if case['name'] == 'CLIP' else [config]
    eps, act = defaults[case['name']]
    for encoder in encoders:
        assert encoder.pop('layer_norm_eps') == eps
        assert encoder.pop('hidden_act') == act
    with pg.create(device='CPU') as rt:
        explicit = pg.Model.from_hf(config_json=json.dumps(case['config']),
            weight_bytes_list=[base64.b64decode(case['weights'])], max_batch=2, runtime=rt)
        implicit = None
        try:
            implicit = pg.Model.from_hf(config_json=json.dumps(config),
                weight_bytes_list=[base64.b64decode(case['weights'])], max_batch=2, runtime=rt)
            # Exact artifact identity checks constants even if the output tolerance
            # could hide a wrong epsilon; execution also exercises each fallback.
            assert implicit.save() == explicit.save()
            inputs = {k: np.array(v, dtype=np.int32 if k == 'input_ids' else np.float32)
                      for k, v in case['inputs'].items()}
            expected = explicit.forward(**inputs)
            for name, value in implicit.forward(**inputs).items():
                np.testing.assert_array_equal(value, expected[name])
                np.testing.assert_allclose(value, case['outputs'][name], rtol=5e-4, atol=5e-5)
        finally:
            if implicit is not None: implicit.dispose()
            explicit.dispose()

@pytest.mark.parametrize('case', CASES, ids=lambda c: c['name'] + ('-gated' if c['config'].get('use_gated_mlp') or c['config'].get('use_swiglu_ffn') else ''))
@pytest.mark.parametrize('device', ['CPU', 'INTERP', 'CUDA'])
def test_vision_models_reference(case, device):
    from polygrad.device import Device
    if device == 'CUDA' and not Device.cuda_available():
        pytest.skip('poly_cuda_available() is false')
    with pg.create(device=device) as rt:
        model = pg.Model.from_hf(config_json=json.dumps(case['config']),
                                weight_bytes_list=[base64.b64decode(case['weights'])], max_batch=2, runtime=rt)
        restored = None
        try:
            inputs = {k:np.array(v, dtype=np.int32 if k == 'input_ids' else np.float32) for k,v in case['inputs'].items()}
            actual = model.forward(**inputs)
            for k, expected in case['outputs'].items():
                np.testing.assert_allclose(actual[k], expected, rtol=5e-4, atol=5e-5)
            bundle = model.save()
            restored = pg.Model.load(bundle, runtime=rt)
            assert restored.save() == bundle
            for k, actual in restored.forward(**inputs).items():
                np.testing.assert_allclose(actual, case['outputs'][k], rtol=5e-4, atol=5e-5)
            with pytest.raises((ValueError, RuntimeError), match='dtype'):
                model.forward(**{**inputs, 'pixel_values': inputs['pixel_values'].astype(np.float64)})
        finally:
            if restored is not None: restored.dispose()
            model.dispose()

@pytest.mark.parametrize('case', CASES)
def test_vision_models_readiness_and_invalid_configs(case):
    config = {**case['config'], 'batch_size': 2}
    with pg.create(device='INTERP') as rt:
        model = getattr(rt.models, case['name'])(config)
        try:
            with pytest.raises((RuntimeError, ValueError), match='not initialized'):
                model.save()
        finally:
            model.dispose()
        invalid = json.loads(json.dumps(config))
        encoder = invalid['vision_config'] if case['name'] == 'CLIP' else invalid
        encoder['num_attention_heads'] = 3
        with pytest.raises((RuntimeError, ValueError), match='heads|width'):
            getattr(rt.models, case['name'])(invalid)
        with pytest.raises(RuntimeError, match='missing.*weight'):
            pg.Model.from_hf(config_json=json.dumps(config), weight_bytes_list=[], max_batch=2, runtime=rt)
        np.testing.assert_array_equal(rt.Tensor([1.0, 2.0]).numpy(), [1., 2.])
