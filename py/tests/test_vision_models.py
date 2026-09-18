"""Shared C vision models: HF oracle, ownership and portable artifacts."""
import base64
import json
from pathlib import Path
import numpy as np
import pytest
import polygrad as pg

CASES = json.loads((Path(__file__).resolve().parents[2] / 'test/fixtures/vision.json').read_text())['cases']

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
