"""Small pinned HF fixture; no Transformers installation or download required."""
import base64
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import polygrad as pg

FIXTURE = json.loads((Path(__file__).resolve().parents[2] / 'test/fixtures/embeddinggemma2.json').read_text(encoding='utf-8'))
IMAGE_FIXTURE = json.loads((Path(__file__).resolve().parents[2] / 'test/fixtures/embeddinggemma2_image.json').read_text(encoding='utf-8'))
AUDIO_FIXTURE = json.loads((Path(__file__).resolve().parents[2] / 'test/fixtures/embeddinggemma2_audio.json').read_text(encoding='utf-8'))


@pytest.mark.parametrize('fixture,token,label', [(IMAGE_FIXTURE,30,'image'), (AUDIO_FIXTURE,28,'audio')])
@pytest.mark.parametrize('delta', [-1,1])
def test_embeddinggemma2_rejects_wrong_slots_before_writes(fixture, token, label, delta, device='CPU'):
    with pg.create(device=device) as rt:
        model = pg.Model.from_hf(config_json=json.dumps(fixture['config']),
                                weight_bytes_list=[base64.b64decode(fixture['weights'])],
                                max_batch=2,max_seq_len=fixture['config']['max_seq_len'],runtime=rt)
        restored = rt.Model.load(model.save())
        from_ir = pg.Model.from_ir(model.export_ir(), model.export_weights(), runtime=rt)
        try:
            case=fixture['cases'][0]['inputs']
            good={k:np.array(v,np.float32 if k in ('pixel_values','input_features') else np.int32) for k,v in case.items()}
            bad={k:v.copy() for k,v in good.items()}
            ids=bad['input_ids']
            location=np.argwhere(ids==token if delta<0 else ids!=token)[0]
            ids[tuple(location)]=1 if delta<0 else token
            for m in (model,restored,from_ir):
                expected=m.forward(**good)['sentence_embedding'].copy()
                np.testing.assert_allclose(expected, fixture['cases'][0]['sentence_embedding'],
                                           atol=2e-5, rtol=2e-4)
                before=m.read_buffer('input_ids').copy()
                with pytest.raises(Exception,match=f'{label} features and token slots do not match'):
                    m.forward(**bad)
                np.testing.assert_array_equal(m.read_buffer('input_ids'),before)
                np.testing.assert_array_equal(m.forward(**good)['sentence_embedding'],expected)
            tensor_ids = rt.Tensor(good['input_ids'])
            try:
                with pytest.raises(Exception, match='requires host data'):
                    model.forward(**{**good, 'input_ids':tensor_ids})
            finally:
                tensor_ids.dispose()
        finally:
            from_ir.dispose()
            restored.dispose()
            model.dispose()


@pytest.mark.parametrize('label,token', [('image',30), ('audio',28)])
def test_embeddinggemma2_slots_without_compiler(label, token):
    # Fresh process: neither an in-memory runner nor a disk cache may hide a
    # dependency on the C compiler when running an interpreter-only model.
    script = f"""
import runpy
suite = runpy.run_path({str(Path(__file__).resolve())!r})
for delta in (-1, 1):
    suite['test_embeddinggemma2_rejects_wrong_slots_before_writes'](
        suite[{(label.upper() + '_FIXTURE')!r}], {token}, {label!r}, delta, device='INTERP')
"""
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True,
                            env={**os.environ, 'CC':'/nonexistent/cc', 'POLY_CACHE':'0'},
                            timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('fixture', [FIXTURE, IMAGE_FIXTURE, AUDIO_FIXTURE], ids=['text','image','audio'])
def test_embeddinggemma2_reference_and_portability(fixture):
    with pg.create(device='CPU') as rt:
        m = pg.Model.from_hf(config_json=json.dumps(fixture['config']),
                            weight_bytes_list=[base64.b64decode(fixture['weights'])],
                            max_batch=2, max_seq_len=fixture['config'].get('max_seq_len',6), runtime=rt)
        try:
            restored = rt.Model.load(m.save())
            try:
                for case in fixture['cases']:
                    raw = case.get('inputs') or {k:case[k] for k in ('input_ids','attention_mask')}
                    inputs = {k:np.array(v,np.float32 if k in ('pixel_values','input_features') else np.int32) for k,v in raw.items()}
                    for model in (m, restored):
                        result = model.forward(**inputs)
                        for name in ('last_hidden_state', 'sentence_embedding'):
                            np.testing.assert_allclose(result[name], case[name], atol=2e-5, rtol=2e-4)
                        if 'image_hidden_states' in case:
                            selected=result['image_hidden_states'][result['image_attention_mask'].astype(bool)]
                            np.testing.assert_allclose(selected,case['image_hidden_states'],atol=2e-5,rtol=2e-4)
                        if 'audio_hidden_states' in case:
                            selected=result['audio_hidden_states'][result['audio_attention_mask'].astype(bool)]
                            np.testing.assert_allclose(selected,case['audio_hidden_states'],atol=2e-5,rtol=2e-4)
            finally:
                restored.dispose()
        finally:
            m.dispose()


@pytest.mark.parametrize('change', [
    {'attention_bias': True}, {'hidden_activation':'silu'}, {'num_key_value_heads':3},
    {'layer_types':['causal_attention','full_attention']}, {'head_dim':3},
    {'rms_norm_eps':-1}, {'per_layer_config':{'2':{'head_dim':8}}},
    {'per_layer_config':{'1':{'head_dim':8}, '01':{'head_dim':8}}},
    {'rope_parameters':{'sliding_attention':{'rope_type':'linear','rope_theta':10000}}},
])
def test_embeddinggemma2_rejects_unsupported_config(change):
    with pg.create(device='CPU') as rt:
        with pytest.raises(ValueError):
            rt.models.EmbeddingGemma2Text({**FIXTURE['config'], **change})


def test_embeddinggemma2_requires_explicit_modality_selection():
    with pg.create(device='CPU') as rt:
        with pytest.raises(ValueError, match='audio_seq_len'):
            rt.models.EmbeddingGemma2({'text_config':FIXTURE['config'], 'audio_config':{}})


def test_embeddinggemma2_text_modalities_import():
    config = {**AUDIO_FIXTURE['config'], 'modalities': ['text']}
    config.pop('audio_seq_len', None)
    with pg.create(device='CPU') as rt:
        model = pg.Model.from_hf(config_json=json.dumps(config),
                                weight_bytes_list=[base64.b64decode(AUDIO_FIXTURE['weights'])],
                                max_batch=2, max_seq_len=config['max_seq_len'], runtime=rt)
        try:
            restored = rt.Model.load(model.save())
            try:
                inputs = {'input_ids': np.ones((2, config['max_seq_len']), np.int32),
                          'attention_mask': np.ones((2, config['max_seq_len']), np.int32)}
                expected = model.forward(**inputs)
                assert 'audio_hidden_states' not in expected
                np.testing.assert_array_equal(restored.forward(**inputs)['sentence_embedding'],
                                              expected['sentence_embedding'])
            finally:
                restored.dispose()
        finally:
            model.dispose()


@pytest.mark.parametrize('modalities', [[], 'text', ['audio'], ['text', 'video'],
                                       ['text', 'text'], ['text', 'image']])
def test_embeddinggemma2_invalid_modalities(modalities):
    with pg.create(device='CPU') as rt:
        with pytest.raises(ValueError, match='modalities'):
            rt.models.EmbeddingGemma2({**AUDIO_FIXTURE['config'], 'modalities': modalities})


@pytest.mark.parametrize('change', [
    {'hidden_act':'relu'}, {'hidden_size':7}, {'num_attention_heads':3},
    {'subsampling_conv_channels':[7,2]}, {'subsampling_conv_channels':[8]},
    {'attention_chunk_size':0}, {'attention_context_left':0},
    {'attention_context_right':-1}, {'rms_norm_eps':-1}, {'use_clipped_linears':3},
    {'attention_logit_cap':0}, {'gradient_clipping':-1},
])
def test_embeddinggemma2_rejects_invalid_audio_config(change):
    config=AUDIO_FIXTURE['config']
    with pg.create(device='CPU') as rt:
        with pytest.raises(ValueError):
            rt.models.EmbeddingGemma2({**config,'audio_config':{**config['audio_config'],**change}})


@pytest.mark.parametrize('change', [
    {'use_clipped_linears':True}, {'attention_bias':True}, {'head_dim':6},
    {'pooling_kernel_size':3}, {'rope_parameters':{'rope_type':'default'}},
])
def test_embeddinggemma2_rejects_unsupported_vision_config(change):
    config=IMAGE_FIXTURE['config']
    with pg.create(device='CPU') as rt:
        with pytest.raises(ValueError):
            rt.models.EmbeddingGemma2({**config,'vision_config':{**config['vision_config'],**change}})


def test_embeddinggemma2_rejects_missing_weights_and_cleans_up():
    with pg.create(device='CPU') as rt:
        # No weights must never produce a runnable random model.
        model = rt.models.EmbeddingGemma2Text({**FIXTURE['config'], 'max_seq_len':6})
        try:
            with pytest.raises(Exception, match='not initialized'):
                model.save()
        finally:
            model.dispose()
