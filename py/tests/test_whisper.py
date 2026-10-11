"""Whisper factory: internal cross/self caches and fused sampling."""
import json
from pathlib import Path
import numpy as np
import pytest
import polygrad as pg


@pytest.fixture
def whisper_reference():
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    from safetensors.torch import save
    torch.set_num_threads(1)
    config = transformers.WhisperConfig(
        vocab_size=16, num_mel_bins=4, d_model=8, encoder_layers=1, decoder_layers=1,
        encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=32,
        decoder_ffn_dim=32, max_source_positions=4, max_target_positions=8,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, decoder_start_token_id=1,
        suppress_tokens=[], begin_suppress_tokens=[], dropout=0., attention_dropout=0.,
        activation_dropout=0., use_cache=True)
    config._attn_implementation = 'eager'
    torch.manual_seed(17)
    reference = transformers.WhisperForConditionalGeneration(config).float().eval()
    weights = {k:v.contiguous() for k,v in reference.state_dict().items() if k != 'proj_out.weight'}
    features = np.linspace(-1, 1, 32, dtype=np.float32).reshape(1, 4, 8)
    tokens = np.array([[1, 5, 7]], np.int32)
    with torch.no_grad():
        expected = reference(input_features=torch.from_numpy(features),
                             decoder_input_ids=torch.from_numpy(tokens).long()).logits.numpy()
    return config, weights, features, tokens, expected


@pytest.mark.parametrize('device', ['CPU', 'INTERP'])
@pytest.mark.parametrize('tied_copy', [False, True])
def test_whisper_logits_caches_and_sampling(whisper_reference, device, tied_copy):
    from safetensors.torch import save
    config, weights, features, tokens, expected = whisper_reference
    if tied_copy:
        weights['proj_out.weight'] = weights['model.decoder.embed_tokens.weight'].clone()
    with pg.create(device=device) as rt:
        model = rt.Model.from_hf(config_json=json.dumps(config.to_dict()),
                                weight_bytes_list=[save(weights)], max_seq_len=3)
        try:
            got = model.forward(input_features=features, decoder_input_ids=tokens)['logits']
            np.testing.assert_allclose(got, expected, atol=3e-5, rtol=3e-4)
            model.call('encode', {'input_features':features})
            pieces = []
            for pos in range(3):
                out = model.call('decode', {'tokens_decode':tokens[:, pos:pos+1]},
                                 controls={'start_pos':pos})['logits_decode']
                pieces.append(out[:, None, :])
            np.testing.assert_allclose(np.concatenate(pieces, axis=1), expected, atol=3e-5, rtol=3e-4)
            before = model.read_buffer('model.decoder.layers.0.self_cache').copy()
            with pytest.raises(RuntimeError, match='view bounds'):
                model.call('prefill', {'tokens_prefill': tokens}, controls={'start_pos':7})
            np.testing.assert_array_equal(model.read_buffer('model.decoder.layers.0.self_cache'), before)
            restored = rt.Model.load(model.save())
            try:
                # KV is transient, like the existing transformer factories.
                np.testing.assert_array_equal(restored.read_buffer('model.decoder.layers.0.self_cache'), 0)
                restored.call('encode', {'input_features':features})
                replay = restored.call('prefill', {'tokens_prefill':tokens}, controls={'start_pos':0})['logits_prefill']
                np.testing.assert_allclose(replay, expected, atol=3e-5, rtol=3e-4)
            finally:
                restored.dispose()
            model.reset_transient()
            np.testing.assert_array_equal(model.read_buffer('model.decoder.layers.0.self_cache'), 0)
            np.testing.assert_array_equal(model.read_buffer('model.decoder.layers.0.cross_cache'), 0)
            model.call('encode', {'input_features':features})
            out = model.call('decode_sample', {'tokens_decode':tokens[:, :1],
                                              'sampling.temperature':np.array([0.],np.float32)},
                             controls={'start_pos':0})['sampling.decode_token']
            np.testing.assert_array_equal(out.reshape(-1), expected[:, 0].argmax(-1))
        finally:
            model.dispose()
        if tied_copy:
            weights['proj_out.weight'][0, 0] += 1
            with pytest.raises(RuntimeError, match='must match tied decoder embeddings'):
                rt.Model.from_hf(config_json=json.dumps(config.to_dict()), weight_bytes_list=[save(weights)])


def test_whisper_pinned_reference(whisper_reference, monkeypatch):
    """Exercise the pin's in-place self/cross caches, not a reimplementation."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / 'references/tinygrad_latest'))
    from examples.whisper import Whisper
    from tinygrad import Tensor, Context
    from tinygrad.nn.state import load_state_dict
    config, weights, features, tokens, expected = whisper_reference
    with Context(DEV='CPU'):
        pin = Whisper(dict(n_mels=4, n_audio_ctx=4, n_audio_state=8, n_audio_head=2,
                           n_audio_layer=1, n_vocab=16, n_text_ctx=8, n_text_state=8,
                           n_text_head=2, n_text_layer=1))
        converted = {}
        for key, value in weights.items():
            key = key.removeprefix('model.')
            key = key.replace('.embed_positions.weight', '.positional_embedding')
            key = key.replace('.embed_tokens.', '.token_embedding.')
            key = key.replace('encoder.layer_norm.', 'encoder.ln_post.')
            key = key.replace('decoder.layer_norm.', 'decoder.ln.')
            key = key.replace('.layers.', '.blocks.')
            for source, dest in [('self_attn_layer_norm', 'attn_ln'), ('encoder_attn_layer_norm', 'cross_attn_ln'),
                                 ('final_layer_norm', 'mlp_ln'), ('self_attn', 'attn'), ('encoder_attn', 'cross_attn'),
                                 ('q_proj', 'query'), ('k_proj', 'key'), ('v_proj', 'value'), ('out_proj', 'out'),
                                 ('fc1', 'mlp.0'), ('fc2', 'mlp.2')]:
                key = key.replace(source, dest)
            converted[key] = Tensor(value.numpy(), device='CPU')
        # The pin includes its constructed causal mask in state_dict, unlike HF.
        converted['decoder.mask'] = pin.decoder.mask
        load_state_dict(pin, converted, strict=True)
        audio = pin.encoder(Tensor(features, device='CPU'))
        outputs = []
        for pos in range(3):
            outputs.append(pin.decoder.forward(Tensor(tokens[:, pos:pos+1], device='CPU'), pos,
                                               audio if pos == 0 else None).numpy())
        # The pin uses tanh-approximate GELU; HF/OpenAI use exact GELU.
        np.testing.assert_allclose(np.concatenate(outputs, axis=1), expected, atol=3e-5, rtol=3e-4)
