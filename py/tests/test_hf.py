"""Tests for HuggingFace model loading."""

import ctypes
import base64
from contextlib import ExitStack
import json
import struct
from pathlib import Path
import numpy as np
import pytest
from polygrad.hf import generate, load_hf_bytes, _find_safetensors, _get_vocab_size
from polygrad.model import Model, ROLE_AUX


@pytest.mark.parametrize('device', ['CPU', 'INTERP', 'CUDA'])
def test_qwen3_rotary_state_and_roundtrip(device):
    import polygrad as pg
    from polygrad.device import Device

    if device == 'CUDA' and not Device.cuda_available():
        pytest.skip('poly_cuda_available() is false in the selected library')
    fixture = json.loads((Path(__file__).resolve().parents[2] / 'test/fixtures/qwen3.json').read_text())
    with pg.create(device=device) as rt:
        model = Model.from_gguf(base64.b64decode(fixture['gguf']), max_seq_len=4, runtime=rt)
        restored = None
        try:
            assert model.entrypoints()[0]['inputs'] == ['x']
            for binding in model.bindings():
                if binding['name'] in ('rope_cos', 'rope_sin'):
                    assert binding['role'] == ROLE_AUX
                    assert binding['shape'] == (1, 1, 4, 4)
                    assert not binding['trainable']
                    np.testing.assert_allclose(model.read_buffer(binding['name']).reshape(4, 4),
                                               fixture[binding['name']], atol=2e-7, rtol=2e-6)
            bundle = model.save()
            restored = Model.load(bundle, runtime=rt)
            assert restored.save() == bundle
            tokens = np.array(fixture['tokens'], dtype=np.int32)
            for current in (model, restored):
                assert current.entrypoints()[0]['inputs'] == ['x']
                np.testing.assert_allclose(current.forward(x=tokens)['output'], fixture['logits'],
                                           atol=3e-5, rtol=3e-5)
            # Imported AUX has independent storage, despite sharing the runtime.
            before = model.read_buffer('rope_cos').copy()
            restored.write_buffer('rope_cos', np.zeros(16, dtype=np.float32))
            np.testing.assert_array_equal(model.read_buffer('rope_cos'), before)
        finally:
            if restored is not None:
                restored.dispose()
            model.dispose()


def test_qwen3_cached_gguf_roundtrip():
    import polygrad as pg
    from polygrad.models import Transformer
    fixture = json.loads((Path(__file__).resolve().parents[2] / 'test/fixtures/qwen3.json').read_text())
    data = base64.b64decode(fixture['gguf'])
    tokens = np.array(fixture['tokens'], dtype=np.int32)
    expected = np.array(fixture['logits'], dtype=np.float32).reshape(1, 4, -1)
    with pg.create(device='CPU') as rt:
        model = Transformer.from_gguf(data, max_seq_len=4, cache_capacity=4,
                               prefill_chunk_size=2, runtime=rt)
        restored = None
        try:
            assert isinstance(model, Transformer)
            for i in range(4):
                np.testing.assert_allclose(model.append_tokens(tokens[:, i:i+1]), expected[:, i],
                                           atol=3e-5, rtol=3e-5)
            with pytest.raises(RuntimeError, match='capacity'):
                model.append_tokens(tokens[:, :1])
            restored = Transformer.load(model.save(), runtime=rt)
            assert restored.decode_position == 0
            np.testing.assert_allclose(restored.prefill_tokens(tokens), expected[:, -1],
                                       atol=3e-5, rtol=3e-5)
            restored.reset()
            assert list(restored.generate(tokens[:, :3], max_tokens=1)) == [int(expected[0, 2].argmax())]
        finally:
            if restored is not None: restored.dispose()
            model.dispose()
        for options in ({'cache_capacity': -1}, {'cache_capacity': 4, 'prefill_chunk_size': 5},
                        {'prefill_chunk_size': 2}, {'cache_capacity': 4, 'max_batch': 2}):
            with pytest.raises((ValueError, RuntimeError), match='cache|prefill'):
                Model.from_gguf(data, runtime=rt, **options)
        with pytest.raises(ValueError, match='prefill|decode|Transformer'):
            Transformer.from_gguf(data, runtime=rt)
        bound = rt.models.Transformer.from_gguf(data, cache_capacity=4)
        try:
            assert bound._ctx == rt._ctx
            assert isinstance(bound, Transformer)
        finally:
            bound.dispose()


@pytest.mark.parametrize('kind', ['gpt2', 'qwen3'])
def test_checkpoint_import_is_quiet_by_default(kind, capfd, monkeypatch):
    import polygrad as pg
    monkeypatch.setenv('POLY_DEBUG', '0')
    with pg.create(device='INTERP') as rt:
        if kind == 'gpt2':
            values = np.ones((32, 16), dtype=np.float32)
            weights = make_safetensors({'transformer.wte.weight': ('F32', values.shape, values)})
            model = load_hf_bytes(GPT2_TINY_CONFIG, [weights], runtime=rt)
        else:
            fixture = json.loads((Path(__file__).resolve().parents[2] / 'test/fixtures/qwen3.json').read_text())
            model = Model.from_gguf(base64.b64decode(fixture['gguf']), max_seq_len=4, runtime=rt)
        model.dispose()
    assert capfd.readouterr().err == ''


@pytest.mark.parametrize('alias', ['transformer.wte.weight', 'wte.weight'])
def test_gpt2_checkpoint_rejects_duplicate_binding(alias):
    import polygrad as pg
    values = np.ones((32, 16), dtype=np.float32)
    first = make_safetensors({'transformer.wte.weight': ('F32', values.shape, values)})
    second = make_safetensors({alias: ('F32', values.shape, values * 2)})
    with pg.create(device='INTERP') as rt:
        live = rt.Tensor([19.0]).realize()
        with pytest.raises(RuntimeError, match="duplicate.*wte.weight"):
            model = load_hf_bytes(GPT2_TINY_CONFIG, [first, second], runtime=rt)
            model.dispose()
        assert (live + 1).item() == 20


@pytest.mark.parametrize('name', ['unused.weight', 'transformer.h.0.attn.bias'])
def test_gpt2_checkpoint_skips_unknown_once_but_rejects_duplicate_names(name, capfd, monkeypatch):
    import polygrad as pg
    monkeypatch.setenv('POLY_DEBUG', '0')
    values = np.ones(1, dtype=np.float32)
    shard = make_safetensors({name: ('F32', values.shape, values)})
    with pg.create(device='INTERP') as rt:
        model = load_hf_bytes(GPT2_TINY_CONFIG, [shard], runtime=rt)
        model.dispose()
        assert capfd.readouterr().err == ''
        with pytest.raises(RuntimeError, match='duplicate weight'):
            model = load_hf_bytes(GPT2_TINY_CONFIG, [shard, shard], runtime=rt)
            model.dispose()


def test_gguf_rejects_duplicate_names_before_model_dispatch():
    import polygrad as pg
    data = b'GGUF' + struct.pack('<IQQ', 3, 2, 0)
    for offset in (0, 32):
        data += struct.pack('<Q', 1) + b'x' + struct.pack('<IQIQ', 1, 1, 0, offset)
    data += bytes((-len(data)) % 32) + bytes(36)
    with pg.create(device='INTERP') as rt:
        with pytest.raises(RuntimeError, match="duplicate weight 'x'"):
            Model.from_gguf(data, runtime=rt)


def test_checkpoint_abi_uses_generic_loaders_only():
    from polygrad import _ffi

    lib = _ffi.get_lib()
    for name in ('poly_linear', 'poly_layernorm', 'poly_rmsnorm', 'poly_embedding',
                 'poly_param', 'poly_input', 'poly_output', 'poly_target', 'poly_aux',
                 'poly_alias', 'poly_register_buffer', 'poly_register_buffer_by_id',
                 'poly_register_existing_buffer', 'poly_ctx_get', 'poly_ctx_get_entry',
                 'poly_ctx_named_count', 'poly_ctx_named_entry', 'poly_register_entrypoint',
                 'poly_ctx_entrypoint_count', 'poly_ctx_entrypoint_name', 'poly_ctx_entrypoint_sink',
                 'poly_ctx_set_trainable', 'poly_ctx_is_trainable',
                 'poly_model_from_ctx', 'poly_model_from_sinks'):
        assert not hasattr(lib, name), f'obsolete ctx-global constructor: {name}'
    for name in ('poly_hf_load', 'poly_hf_load_into', 'poly_gguf_load', 'poly_gguf_load_into'):
        assert getattr(lib, name)
    for model, formats in (('gpt2', ('hf', 'gguf')), ('llama', ('hf',)), ('qwen3', ('gguf',))):
        for fmt in formats:
            for name in (f'poly_{model}_from_{fmt}', f'poly_{model}_from_{fmt}_decoded',
                         f'poly_{model}_from_{fmt}_decoded_generic', f'model_{model}_from_{fmt}_decoded'):
                with pytest.raises(AttributeError, match='undefined symbol'):
                    getattr(lib, name)


def test_qwen3_gguf_zero_heads_preserves_runtime():
    import polygrad as pg

    def string(value):
        data = value.encode()
        return struct.pack('<Q', len(data)) + data

    data = b'GGUF' + struct.pack('<IQQ', 3, 0, 2)
    data += string('general.architecture') + struct.pack('<I', 8) + string('qwen3')
    data += string('qwen3.attention.head_count') + struct.pack('<II', 4, 0)
    data += bytes((-len(data)) % 32)
    with pg.create(device='INTERP') as rt:
        live = rt.Tensor([2.0]).realize()
        with pytest.raises(RuntimeError, match='attention.head_count.*positive'):
            Model.from_gguf(data, runtime=rt)
        assert (live + 1).item() == 3


@pytest.mark.parametrize('shape,transpose', [((4, 2), 0), ((2, 3), 0), ((3, 2), 1)])
def test_checkpoint_binding_rejects_wrong_shape_without_writing(shape, transpose):
    import polygrad as pg
    from polygrad import _ffi
    lib = _ffi.get_lib()
    lib.poly_bind_index_create.argtypes = [ctypes.c_void_p]
    lib.poly_bind_index_create.restype = ctypes.c_void_p
    lib.poly_bind_index_destroy.argtypes = [ctypes.c_void_p]
    lib.poly_bind_index_destroy.restype = None
    lib.poly_import_copy_named_tensor.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int64), ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.poly_import_copy_named_tensor.restype = ctypes.c_int
    with pg.create(device='INTERP') as rt:
        model = pg.models.MLP(layers=[2, 3], runtime=rt)
        index = lib.poly_bind_index_create(model._ptr)
        try:
            before = model.read_buffer('layers.0.weight')
            values = np.ones(shape, dtype=np.float32)
            dims = (ctypes.c_int64 * len(shape))(*shape)
            rc = lib.poly_import_copy_named_tensor(index, b'layers.0.weight',
                values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), dims, len(shape), transpose, -1)
            assert rc == -1
            np.testing.assert_array_equal(model.read_buffer('layers.0.weight'), before)
        finally:
            lib.poly_bind_index_destroy(index)
            model.dispose()


def make_safetensors(tensors):
    """Build a minimal safetensors file from {name: (dtype_str, shape, data_bytes)}.

    Args:
        tensors: dict mapping name to (dtype_str, shape, numpy_array).

    Returns:
        bytes: Complete safetensors file.
    """
    header = {}
    offset = 0
    data_parts = []

    for name, (dtype_str, shape, arr) in tensors.items():
        raw = arr.tobytes()
        header[name] = {
            'dtype': dtype_str,
            'shape': list(shape),
            'data_offsets': [offset, offset + len(raw)]
        }
        data_parts.append(raw)
        offset += len(raw)

    header_json = json.dumps(header).encode('utf-8')
    header_size = len(header_json)

    result = struct.pack('<Q', header_size) + header_json
    for part in data_parts:
        result += part
    return result


@pytest.mark.parametrize('storage', [bytes, bytearray, memoryview])
def test_hf_borrows_immutable_shards_only_during_import(monkeypatch, storage):
    import polygrad as pg
    import polygrad.hf as hf
    from types import SimpleNamespace

    case = LLAMA_CASES[-1]
    weights = llama_weights(case)
    raw = make_safetensors({k: ('F32', v.shape, v) for k, v in weights.items()})
    shard = storage(raw)
    expected_address = ctypes.cast(ctypes.c_char_p(raw), ctypes.c_void_p).value
    lib = hf._get_lib()
    addresses = []

    def load(*args):
        addresses.append(ctypes.cast(args[3][0], ctypes.c_void_p).value)
        assert ctypes.string_at(args[3][0], args[4][0]) == raw
        return lib.poly_hf_load_into(*args)

    monkeypatch.setattr(hf, '_get_lib', lambda: SimpleNamespace(
        poly_hf_load_into=load, poly_import_last_error_message=lib.poly_import_last_error_message))
    with pg.create(device='cpu') as rt:
        model = load_hf_bytes(json.dumps(case['config']), [shard], max_seq_len=3, runtime=rt)
        try:
            if storage is bytes:
                assert addresses == [expected_address], 'immutable shard was copied at the FFI boundary'
            if storage is bytearray:
                shard[:] = bytes(len(shard))
            del shard
            actual = model.forward(tokens=np.array(case['tokens'], np.int32))['logits']
            np.testing.assert_allclose(actual.reshape(-1), case['logits'], atol=2e-5, rtol=2e-5)
        finally:
            model.dispose()


GPT2_TINY_CONFIG = json.dumps({
    'model_type': 'gpt2',
    'vocab_size': 32,
    'n_embd': 16,
    'n_head': 2,
    'n_layer': 1,
    'n_positions': 8,
    'layer_norm_epsilon': 1e-5
})


@pytest.mark.parametrize('shape,accepted', [((8, 16), True), ((8, 17), False), ((1, 16), False)])
def test_gpt2_position_crop_is_explicit_and_axis_checked(shape, accepted):
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    checkpoint = make_safetensors({'transformer.wpe.weight': ('F32', shape, values)})
    if not accepted:
        with pytest.raises(RuntimeError, match='shape mismatch'):
            load_hf_bytes(GPT2_TINY_CONFIG, [checkpoint], max_seq_len=2)
    else:
        model = load_hf_bytes(GPT2_TINY_CONFIG, [checkpoint], max_seq_len=2)
        try:
            np.testing.assert_array_equal(model.read_buffer('wpe.weight'), values[:2].reshape(-1))
        finally:
            model.dispose()


def test_known_type_rejects_unsupported_import_format():
    config = json.dumps({'model_type': 'qwen3'})
    checkpoint = make_safetensors({'weight': ('F32', (1,), np.ones(1, dtype=np.float32))})
    with pytest.raises(RuntimeError, match='Qwen3 does not support HF import'):
        load_hf_bytes(config, [checkpoint])


LLAMA_CASES = json.loads((Path(__file__).resolve().parents[2] / 'test/fixtures/llama.json').read_text())['cases']


def llama_weights(case):
    return {name: np.array([(1.0 if len(shape) == 1 else 0.0) +
                            ((i * 7 + sum(name.encode())) % 23 - 11) * 0.017
                            for i in range(int(np.prod(shape)))], np.float32).reshape(shape)
            for name, shape in case['weights'].items()}


def test_transformer_specialization_owns_one_model():
    import polygrad as pg
    from polygrad.models import Transformer
    from polygrad.models.transformer import Transformer as Implementation
    assert Transformer is Implementation
    assert not hasattr(pg.Model, 'generate')
    case = LLAMA_CASES[0]
    with pg.create(device='interp') as rt, ExitStack() as cleanup:
        plain = rt.models.Llama(case['config'])
        cleanup.callback(plain.dispose)
        assert isinstance(plain, pg.Model) and not hasattr(plain, 'append_tokens')
        original = plain._ptr
        with pytest.raises(ValueError, match='Transformer contract'):
            Transformer.from_model(plain)
        assert plain._ptr == original
        cached = rt.models.Llama({**case['config'], 'cache_capacity':5, 'prefill_chunk_size':4})
        cleanup.callback(cached.dispose)
        assert isinstance(cached, Transformer) and isinstance(cached, pg.Model)
        for name, data in llama_weights(case).items(): cached.write_buffer(name, data)
        bundle = cached.save()
        generic = rt.Model.load(bundle)
        cleanup.callback(generic.dispose)
        assert not hasattr(generic, 'append_tokens')
        assert generic.save() == bundle  # Application metadata survives generic round trips.
        specialized = Transformer.from_model(generic)
        assert specialized is generic
        assert Transformer.from_model(specialized) is specialized
        assert specialized.decode_position == 0
        specialized.append_tokens(np.array([1, 4], np.int32))
        assert specialized.decode_position == 2
        specialized.dispose()
        specialized.dispose()
        with pytest.raises(RuntimeError, match='disposed'):
            specialized.append_tokens(np.array([1], np.int32))


@pytest.mark.parametrize('device', ['cpu', 'interp', 'cuda'])
def test_transformer_generate_stays_in_c(device):
    import polygrad as pg
    if device == 'cuda' and not pg.Device.cuda_available():
        pytest.fail('requested CUDA Transformer lane: poly_cuda_available() is false')
    case = LLAMA_CASES[0]
    with pg.create(device=device) as rt, ExitStack() as cleanup:
        model = rt.models.Llama({**case['config'], 'max_seq_len':5,
                                'cache_capacity':5, 'prefill_chunk_size':4})
        plain = rt.models.Llama({**case['config'], 'max_seq_len':5})
        for m in (model, plain):
            cleanup.callback(m.dispose)
            for name, data in llama_weights(case).items(): m.write_buffer(name, data)
        tokens = [1, 4]
        expected = []
        for _ in range(3):
            padded = np.array([tokens + [0] * (5-len(tokens))], np.int32)
            token = int(plain.forward(tokens=padded)['logits'][0, len(tokens)-1].argmax())
            expected.append(token)
            tokens.append(token)
        assert list(model.generate(np.array([1,4], np.int32), temperature=0, max_tokens=3)) == expected
        assert model.decode_position == 4  # Last yielded token is not consumed yet, as in Tinygrad.
        model.reset()
        assert list(model.generate(np.array([1,4], np.int32), temperature=0, max_tokens=99)) == expected


@pytest.mark.parametrize('device', ['cpu', 'interp', 'cuda'])
def test_transformer_specialized_entries_share_sampling_state(device):
    import polygrad as pg
    case = LLAMA_CASES[0]
    with pg.create(device=device) as rt, ExitStack() as cleanup:
        rt.Tensor.manual_seed(0)
        model = rt.models.Llama({**case['config'], 'cache_capacity':8, 'prefill_chunk_size':4})
        cleanup.callback(model.dispose)
        for name, data in llama_weights(case).items(): model.write_buffer(name, data)
        saved = model.save()
        other = rt.models.Transformer.load(saved)
        cleanup.callback(other.dispose)
        prompt = np.array([[1,4,7,2]], np.int32)
        expected = model.call('prefill', {'tokens_prefill':prompt}, controls={'start_pos':0})['logits_prefill']
        actual = other.call('prefill_full', {'tokens_prefill_full':prompt}, controls={'start_pos':0})['logits_prefill_full']
        np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-5)
        for pos, temperature in [(4,0.7),(5,0.0),(6,1.2)]:
            token = np.array([[3]], np.int32)
            logits = model.call('decode', {'tokens_decode':token}, controls={'start_pos':pos})['logits_decode']
            temp = np.array([temperature], np.float32)
            want = model.call('sample', {'sampling.logits':logits,'sampling.temperature':temp})['sampling.token']
            got = other.call('decode_sample', {'tokens_decode':token,'sampling.temperature':temp},
                             controls={'start_pos':pos})['sampling.decode_token']
            np.testing.assert_array_equal(got, want)
            np.testing.assert_array_equal(other.read_buffer('sampling.counter'), model.read_buffer('sampling.counter'))
            for layer in range(case['config']['num_hidden_layers']):
                name = f'model.layers.{layer}.cache_kv'
                np.testing.assert_allclose(other.read_buffer(name), model.read_buffer(name), atol=3e-5, rtol=3e-5)
        before = other.read_buffer('model.layers.0.cache_kv').copy()
        with pytest.raises(RuntimeError, match='view bounds'):
            other.call('prefill_full', {'tokens_prefill_full':prompt}, controls={'start_pos':7})
        np.testing.assert_array_equal(other.read_buffer('model.layers.0.cache_kv'), before)


@pytest.mark.parametrize('device', ['cpu', 'interp', 'cuda'])
def test_transformer_generation_reclaims_dropped_storage(device):
    import polygrad as pg
    case = LLAMA_CASES[0]
    with pg.create(device=device) as rt, ExitStack() as cleanup:
        model = rt.models.Llama({**case['config'], 'cache_capacity':8, 'prefill_chunk_size':4})
        cleanup.callback(model.dispose)
        for name, data in llama_weights(case).items(): model.write_buffer(name, data)
        tokens = model.generate(np.array([1,4], np.int32), temperature=0, max_tokens=6)
        for _ in range(3): next(tokens)
        baseline = rt.GlobalCounters.mem_used
        scratch = rt.Tensor.empty(1 << 20)
        scratch.numpy()
        assert rt.GlobalCounters.mem_used >= (4 << 20)
        scratch.dispose()
        next(tokens)
        assert rt.GlobalCounters.mem_used <= baseline


@pytest.mark.parametrize('device', ['cpu', 'interp', 'cuda'])
def test_transformer_sampler_matches_pinned_gumbel(device):
    import polygrad as pg
    if device == 'cuda' and not pg.Device.cuda_available():
        pytest.fail('requested CUDA Transformer lane: poly_cuda_available() is false')
    case = LLAMA_CASES[0]
    with pg.create(device=device) as rt:
        rt.Tensor.manual_seed(0)
        model = rt.models.Llama({**case['config'], 'vocab_size':11,
                                'cache_capacity':5, 'prefill_chunk_size':4})
        try:
            for binding in model.bindings():
                if binding['name'].endswith('.weight'):
                    model.write_buffer(binding['name'], np.zeros(binding['shape'], np.float32))
            inputs = {'sampling.logits':np.linspace(-1,1,11,dtype=np.float32).reshape(1,11),
                      'sampling.temperature':np.array([0.7], np.float32)}
            # Pinned llm/model.py:378 formula, seed=0, eight sequential draws.
            expected = [10,10,10,10,9,10,7,5,6,6,2,0,0,10,10,10]
            saved = model.save()
            for run in range(2):
                model.reset()
                actual = [int(model.call('sample', inputs)['sampling.token'].item()) for _ in range(8)]
                assert actual == expected[run*8:(run+1)*8]
            # Conversation reset preserves RNG progress; checkpoints carry it.
            assert model.save() != saved
            progressed = rt.models.Transformer.load(model.save())
            try:
                np.testing.assert_array_equal(progressed.read_buffer('sampling.counter'),
                                              model.read_buffer('sampling.counter'))
                assert progressed.call('sample', inputs)['sampling.token'].item() == model.call('sample', inputs)['sampling.token'].item()
            finally:
                progressed.dispose()
            restored = rt.models.Transformer.load(saved)
            try:
                assert restored.decode_position == 0
                assert int(restored.call('sample', inputs)['sampling.token'].item()) == expected[0]
            finally:
                restored.dispose()
        finally:
            model.dispose()


@pytest.mark.parametrize('device', ['cpu', 'interp', 'cuda'])
def test_llama_prefix_reuse_and_rewind(device):
    import polygrad as pg
    if device == 'cuda' and not pg.Device.cuda_available():
        pytest.fail('requested CUDA Transformer lane: poly_cuda_available() is false')
    case = LLAMA_CASES[0]
    with pg.create(device=device) as rt, ExitStack() as cleanup:
        config = {**case['config'], 'max_seq_len':5, 'cache_capacity':5, 'prefill_chunk_size':4}
        model, fresh = rt.models.Llama(config), rt.models.Llama(config)
        for m in (model, fresh):
            cleanup.callback(m.dispose)
            for name, data in llama_weights(case).items(): m.write_buffer(name, data)
        initial = model.save()
        for prompt in ([1,4,7,2], [1,4,7,2], [1,4,7,2,9], [1,4,3], [1,4], [9,2], [1]):
            tokens = np.array(prompt, np.int32)
            fresh.reset_transient()
            expected = fresh.append_tokens(tokens)
            np.testing.assert_allclose(model.prefill_tokens(tokens), expected, atol=3e-5, rtol=3e-5)
            assert model.decode_position == len(prompt)
        model.prefill_tokens(np.array([1,4,7,2], np.int32))
        before = model.read_buffer('model.layers.0.cache_kv').copy()
        model.rewind(2)
        assert model.decode_position == 2
        np.testing.assert_array_equal(model.read_buffer('model.layers.0.cache_kv'), before)
        for bad in (-1, 3, 2**40):
            with pytest.raises((ValueError, RuntimeError)): model.rewind(bad)
            assert model.decode_position == 2
        with pytest.raises(TypeError): model.rewind(1.5)
        for bad in ([1,99], [1]*6, []):
            with pytest.raises(RuntimeError): model.prefill_tokens(np.array(bad, np.int32))
            assert model.decode_position == 2
            np.testing.assert_array_equal(model.read_buffer('model.layers.0.cache_kv'), before)
        fresh.reset_transient()
        expected = fresh.append_tokens(np.array([1,4,3], np.int32))
        np.testing.assert_allclose(model.append_tokens(np.array([3], np.int32)), expected, atol=3e-5, rtol=3e-5)
        assert model.save() == initial
        restored = rt.models.Transformer.load(model.save())
        cleanup.callback(restored.dispose)
        assert restored.decode_position == 0
        with pytest.raises(RuntimeError): restored.rewind(1)
        np.testing.assert_allclose(restored.prefill_tokens(np.array([1,4,3], np.int32)), expected, atol=3e-5, rtol=3e-5)
        name, weights = next(iter(llama_weights(case).items()))
        model.write_buffer(name, weights)
        assert model.decode_position == -1
        with pytest.raises(RuntimeError): model.rewind(0)
        with pytest.raises(RuntimeError): model.prefill_tokens(np.array([1], np.int32))
        model.reset_transient()
        model.rewind(0)


@pytest.mark.parametrize('device', ['cpu', 'interp', 'cuda'])
def test_llama_variable_prefill_admission_and_import(device):
    import polygrad as pg
    if device == 'cuda' and not pg.Device.cuda_available():
        pytest.fail('requested CUDA Transformer lane: poly_cuda_available() is false')
    case = LLAMA_CASES[0]
    with pg.create(device=device) as rt, ExitStack() as cleanup:
        plain = rt.models.Llama({**case['config'], 'max_seq_len': 5})
        cached = rt.models.Llama({**case['config'], 'max_seq_len': 5,
                                  'cache_capacity': 5, 'prefill_chunk_size': 4})
        for model in (plain, cached):
            cleanup.callback(model.dispose)
            for name, data in llama_weights(case).items(): model.write_buffer(name, data)
        tokens = np.array([[1, 4, 7, 2, 9]], np.int32)
        expected = plain.forward(tokens=tokens)['logits']
        restored = rt.models.Transformer.load(cached.save())
        cleanup.callback(restored.dispose)
        for model in (cached, restored):
            compiled = None
            for n in (1, 3, 2, 4):
                model.reset_transient()
                got = model.call('prefill', {'tokens_prefill': tokens[:, :n]},
                                 controls={'start_pos': 0})['logits_prefill']
                np.testing.assert_allclose(got, expected[:, n-1], atol=3e-5, rtol=3e-5)
                current = {k:v for k,v in rt.stats().items() if k in (
                    'to_program_cache_entries', 'runtime_cache_entries', 'runtime_artifact_entries')}
                if compiled is not None:
                    assert current == compiled, (compiled, current)
                compiled = current
            model.reset_transient()
            model.append_tokens(tokens[:, :1])
            before = model.read_buffer('model.layers.0.cache_kv').copy()
            # Position and width individually fit their bounds; their sum does not.
            for tensor_io in (False, True):
                value = rt.Tensor(tokens[:, :3]) if tensor_io else tokens[:, :3]
                try:
                    with pytest.raises(RuntimeError, match=r'start_pos=3.*tokens_prefill\[1\]=3'):
                        model.call('prefill', {'tokens_prefill': value}, controls={'start_pos': 3})
                finally:
                    if tensor_io: value.dispose()
                assert model.decode_position == 1
                np.testing.assert_array_equal(model.read_buffer('model.layers.0.cache_kv'), before)
            got = model.append_tokens(tokens[:, 1:4])
            np.testing.assert_allclose(got, expected[:, 3], atol=3e-5, rtol=3e-5)
            got = model.append_tokens(tokens[:, 4:])
            np.testing.assert_allclose(got, expected[:, 4], atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize('device', ['cpu', 'interp', 'cuda'])
@pytest.mark.parametrize('case', LLAMA_CASES, ids=['llama2-gqa', 'llama2-mha', 'llama3', 'llama32-tied'])
def test_llama_cached_partitions_and_fresh_import(device, case):
    import polygrad as pg
    if device == 'cuda' and not pg.Device.cuda_available():
        pytest.fail('requested CUDA Transformer lane: poly_cuda_available() is false')
    weights = llama_weights(case)
    with pg.create(device=device) as rt, ExitStack() as cleanup:
        plain = rt.models.Llama({**case['config'], 'max_seq_len':5})
        cached = rt.models.Llama({**case['config'], 'max_seq_len':5,
                                  'cache_capacity':5, 'prefill_chunk_size':2})
        for model in (plain, cached):
            cleanup.callback(model.dispose)
            for name, data in weights.items(): model.write_buffer(name, data)
        assert plain.param_count == cached.param_count
        tokens = np.array([[1,4,7,2,9]], np.int32)
        expected = plain.forward(tokens=tokens)['logits']
        initial_bundle = cached.save()
        warm = None
        for chunks in ((5,), (2, 2, 1), (1, 1, 1, 1, 1)):
            cached.reset_transient()
            pos = 0
            for n in chunks:
                got = cached.append_tokens(tokens[:, pos:pos+n])
                pos += n
                assert cached.decode_position == pos
                np.testing.assert_allclose(got, expected[:, pos-1], atol=3e-5, rtol=3e-5)
            with pytest.raises(RuntimeError, match='capacity'):
                cached.append_tokens(tokens[:, :1])
            assert cached.decode_position == 5
            retained = {k:v for k,v in rt.stats().items() if k in (
                'to_program_cache_entries', 'runtime_cache_entries', 'runtime_artifact_entries',
                'buffer_owned_bytes')}
            if warm is not None:
                assert retained == warm, (warm, retained)
            warm = retained
        assert cached.save() == initial_bundle
        cached.reset_transient()
        for bad in (np.array([[1, -1]], np.int32), np.array([[1, case['config']['vocab_size']]], np.int32)):
            with pytest.raises(RuntimeError, match='vocabulary'):
                cached.append_tokens(bad)
            assert cached.decode_position == 0
        with pytest.raises(TypeError, match='int32'):
            cached.append_tokens(tokens.astype(np.int64))
        cached.append_tokens(tokens[:, :1])
        # Read-only forward and rejected raw calls preserve the checked prefix.
        np.testing.assert_allclose(cached.forward(tokens=tokens)['logits'], expected, atol=3e-5, rtol=3e-5)
        assert cached.decode_position == 1
        for tensor_io in (False, True):
            t = rt.Tensor(tokens[:, :1]) if tensor_io else tokens[:, :1]
            try:
                with pytest.raises(RuntimeError, match='control'):
                    cached.call('decode', {'tokens_decode': t}, controls={'start_pos': 5})
                assert cached.decode_position == 1
            finally:
                if tensor_io: t.dispose()
        t = rt.Tensor(tokens)
        result = cached.forward(tokens=t)['logits']
        try:
            np.testing.assert_allclose(result.numpy(), expected, atol=3e-5, rtol=3e-5)
            assert cached.decode_position == 1
        finally:
            result.dispose()
            t.dispose()
        np.testing.assert_allclose(cached.append_tokens(tokens[:,1:2]), expected[:,1], atol=3e-5, rtol=3e-5)
        name, weight = next(iter(weights.items()))
        cached.write_buffer(name, weight)
        assert cached.decode_position == -1
        with pytest.raises(RuntimeError, match='reset the Transformer'):
            cached.append_tokens(tokens[:, :1])
        for chunks in ((2,2,1), (1,1,1,1,1)):
            cached.reset_transient()
            pos = 0
            for n in chunks:
                ep = 'prefill' if n == 2 else 'decode'
                out = cached.call(ep, {f'tokens_{ep}':tokens[:,pos:pos+n]}, controls={'start_pos':pos})
                np.testing.assert_allclose(out[f'logits_{ep}'], expected[:,pos+n-1], atol=3e-5, rtol=3e-5)
                pos += n
        restored = rt.models.Transformer.load(cached.save())
        cleanup.callback(restored.dispose)
        assert restored.decode_position == 0
        np.testing.assert_allclose(restored.append_tokens(tokens), expected[:, -1], atol=3e-5, rtol=3e-5)
        restored.reset_transient()
        out = restored.call('decode', {'tokens_decode':tokens[:,:1]}, controls={'start_pos':0})
        np.testing.assert_allclose(out['logits_decode'], expected[:,0], atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize('device', ['cpu', 'interp'])
def test_llama_omitted_epsilon_matches_hf_default(device):
    import polygrad as pg
    case = LLAMA_CASES[0]
    config = dict(case['config'])
    config.pop('rms_norm_eps', None)
    weights = llama_weights(case)
    # Small embedding magnitudes make the epsilon contract observable.
    weights['model.embed_tokens.weight'] *= 0.01
    checkpoint = make_safetensors({k: ('F32', v.shape, v) for k, v in weights.items()})
    with pg.create(device=device) as rt, ExitStack() as cleanup:
        values = []
        for cfg in (config, {**config, 'rms_norm_eps': 1e-6}):
            model = load_hf_bytes(json.dumps(cfg), [checkpoint], max_seq_len=3, runtime=rt)
            cleanup.callback(model.dispose)
            values.append(model.forward(tokens=np.array(case['tokens'], np.int32))['logits'])
        np.testing.assert_array_equal(values[0], values[1])


def test_llama_cuda_checkpoint_io_does_not_retain_host_shadows():
    import polygrad as pg
    from polygrad import _ffi
    if not hasattr(_ffi._lib, 'poly_cuda_available') or not _ffi._lib.poly_cuda_available():
        pytest.skip('poly_cuda_available() is false in the selected library')
    case = LLAMA_CASES[-1]
    weights = llama_weights(case)
    checkpoint = make_safetensors({k: ('F32', v.shape, v) for k, v in weights.items()})
    with pg.create(device='cuda') as rt, ExitStack() as cleanup:
        model = load_hf_bytes(json.dumps(case['config']), [checkpoint], max_seq_len=3, runtime=rt)
        cleanup.callback(model.dispose)
        rt.collect()
        assert rt.GlobalCounters.mem_used_per_device.get('CPU', 0) == 0
        key = 'model.embed_tokens.weight'
        np.testing.assert_array_equal(model.read_buffer(key), weights[key].reshape(-1))
        restored = rt.Model.load(model.save(include_optimizer=False))
        cleanup.callback(restored.dispose)
        restored.write_buffer('lm_head.weight', np.zeros_like(weights[key]))
        np.testing.assert_array_equal(restored.read_buffer(key), 0)
        np.testing.assert_array_equal(model.read_buffer(key), weights[key].reshape(-1))
        rt.collect()
        assert rt.GlobalCounters.mem_used_per_device.get('CPU', 0) == 0
        actual = model.forward(tokens=np.array(case['tokens'], np.int32))['logits']
        np.testing.assert_allclose(actual.reshape(-1), case['logits'], atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize('device', ['cpu', 'interp'])
@pytest.mark.parametrize('case', LLAMA_CASES, ids=['llama2-gqa', 'llama2-mha', 'llama3', 'llama32-tied'])
def test_llama_requires_explicit_weights_before_execution_or_export(device, case):
    import polygrad as pg
    with pg.create(device=device) as rt:
        with ExitStack() as cleanup:
            model = rt.models.Llama(case['config'])
            cleanup.callback(model.dispose)
            tokens = np.array(case['tokens'], np.int32)
            weights = llama_weights(case)
            items = list(weights.items())
            for count in (0, 1):
                with pytest.raises(RuntimeError, match='weight.*not initialized'):
                    model.forward(tokens=tokens)
                for export in (model.export_ir, model.export_weights, model.save):
                    with pytest.raises(RuntimeError, match='weight.*not initialized'):
                        export()
                name, data = items[count]
                model.write_buffer(name, data)
            for name, data in items[2:]:
                model.write_buffer(name, data)
            np.testing.assert_allclose(model.forward(tokens=tokens)['logits'].reshape(-1),
                                       case['logits'], atol=2e-5, rtol=2e-5)
            assert model.save() == model.save()


@pytest.mark.parametrize('device', ['cpu', 'interp'])
@pytest.mark.parametrize('case', LLAMA_CASES, ids=['llama2-gqa', 'llama2-mha', 'llama3', 'llama32-tied'])
def test_llama_family_reference_and_shared_import(device, case):
    import polygrad as pg
    weights = llama_weights(case)
    checkpoint = make_safetensors({k: ('F32', v.shape, v) for k,v in weights.items()})
    with pg.create(device=device) as rt:
        live = rt.Tensor([37.])
        with ExitStack() as cleanup:
            direct = rt.models.Llama(case['config'])
            cleanup.callback(direct.dispose)
            imported = load_hf_bytes(json.dumps(case['config']), [checkpoint], max_seq_len=3, runtime=rt)
            cleanup.callback(imported.dispose)
            for name, data in weights.items(): direct.write_buffer(name, data)
            for model in (direct, imported):
                np.testing.assert_allclose(model.read_buffer('freqs_cos').reshape(-1), case['freqs_cos'], atol=2e-6)
                np.testing.assert_allclose(model.read_buffer('freqs_sin').reshape(-1), case['freqs_sin'], atol=2e-6)
                tokens = np.array(case['tokens'], np.int32)
                first = model.forward(tokens=tokens)['logits']
                np.testing.assert_allclose(first.reshape(-1), case['logits'], atol=2e-5, rtol=2e-5)
                changed = model.forward(tokens=np.array([[1,4,7]], np.int32))['logits']
                np.testing.assert_allclose(changed[:, :2], first[:, :2], atol=2e-6)
            restored = rt.Model.load(imported.save(include_optimizer=False))
            cleanup.callback(restored.dispose)
            np.testing.assert_allclose(restored.forward(tokens=tokens)['logits'].reshape(-1), case['logits'], atol=2e-5, rtol=2e-5)
            key = 'model.embed_tokens.weight'
            restored.write_buffer(key, np.zeros_like(weights[key]))
            np.testing.assert_array_equal(imported.read_buffer(key), weights[key].reshape(-1))
            if case['config']['tie_word_embeddings']:
                np.testing.assert_array_equal(restored.read_buffer('lm_head.weight'), np.zeros(weights[key].size, np.float32))
            np.testing.assert_array_equal(live.numpy(), [37.])


@pytest.mark.parametrize('change', [dict(hidden_size=7), dict(num_key_value_heads=3),
    dict(max_seq_len=17), dict(hidden_act='relu'), dict(attention_bias=True),
    dict(rope_scaling={'rope_type':'dynamic'}), dict(rope_theta=-1), dict(num_hidden_layers=1.5)])
def test_llama_rejects_unsupported_configuration_without_touching_runtime(change):
    import polygrad as pg
    with pg.create(device='interp') as rt:
        live = rt.Tensor([19.])
        with pytest.raises(ValueError, match='Llama'):
            rt.models.Llama({**LLAMA_CASES[0]['config'], **change})
        np.testing.assert_array_equal(live.numpy(), [19.])


@pytest.mark.parametrize('damage', ['missing', 'shape', 'duplicate', 'tied_conflict',
                                  'tied_missing_both', 'tied_head_shape', 'untied_head_only'])
def test_llama_checkpoint_rejects_incomplete_or_conflicting_state(damage):
    import polygrad as pg
    case = LLAMA_CASES[3 if damage.startswith('tied_') else 0]
    weights = llama_weights(case)
    if damage == 'missing': weights.pop('model.norm.weight')
    if damage == 'shape': weights['model.norm.weight'] = np.zeros(2, np.float32)
    if damage == 'tied_conflict': weights['lm_head.weight'] = np.zeros_like(weights['model.embed_tokens.weight'])
    if damage in ('tied_missing_both', 'tied_head_shape', 'untied_head_only'):
        weights.pop('model.embed_tokens.weight')
    if damage == 'tied_head_shape': weights['lm_head.weight'] = np.zeros((2, 2), np.float32)
    checkpoint = make_safetensors({k: ('F32', v.shape, v) for k,v in weights.items()})
    with pg.create(device='interp') as rt:
        live = rt.Tensor([31.])
        expected = {'missing':'missing weight', 'shape':'invalid weight',
                    'duplicate':'duplicate', 'tied_conflict':'tied lm_head',
                    'tied_missing_both':'missing weight', 'tied_head_shape':'invalid weight',
                    'untied_head_only':'missing weight'}[damage]
        with pytest.raises((RuntimeError, ValueError), match=expected):
            load_hf_bytes(json.dumps(case['config']), [checkpoint] * (2 if damage == 'duplicate' else 1), runtime=rt)
        np.testing.assert_array_equal(live.numpy(), [31.])


def test_llama_default_runtime_and_batched_rows():
    from polygrad import models
    case = LLAMA_CASES[0]
    model = models.Llama({**case['config'], 'batch_size': 2})
    try:
        for name, data in llama_weights(case).items(): model.write_buffer(name, data)
        out = model.forward(tokens=np.array(case['tokens'] * 2, np.int32))['logits']
        assert out.shape == (2, 3, 11)
        for row in out: np.testing.assert_allclose(row.reshape(-1), case['logits'], atol=2e-5, rtol=2e-5)
    finally:
        model.dispose()


@pytest.mark.parametrize('layout', ['both-embed-first', 'both-head-first', 'head-only'])
def test_llama_tied_checkpoint_accepts_either_alias(layout):
    case = LLAMA_CASES[-1]
    weights = llama_weights(case)
    head = {'lm_head.weight': weights['model.embed_tokens.weight']}
    weights = {**head, **weights} if layout == 'both-head-first' else {**weights, **head}
    if layout == 'head-only':
        del weights['model.embed_tokens.weight']
    checkpoint = make_safetensors({k: ('F32', v.shape, v) for k,v in weights.items()})
    model = load_hf_bytes(json.dumps(case['config']), [checkpoint], max_seq_len=3)
    try:
        out = model.forward(tokens=np.array(case['tokens'], np.int32))['logits']
        np.testing.assert_allclose(out.reshape(-1), case['logits'], atol=2e-5, rtol=2e-5)
    finally:
        model.dispose()


class TestHFLoadBasic:
    def test_runtime_bound_imports_are_independent(self):
        import polygrad as pg
        rt = pg.create(device='interp')
        models = []
        try:
            live = rt.Tensor([19.0])
            weights = make_safetensors({
                'transformer.wte.weight': ('F32', (32, 16), np.zeros((32, 16), dtype=np.float32))
            })
            a = load_hf_bytes(GPT2_TINY_CONFIG, [weights], runtime=rt)
            b = pg.Model.from_hf(config_json=GPT2_TINY_CONFIG, weight_bytes_list=[weights], runtime=rt)
            models.extend([a, b])
            assert a._ctx == b._ctx == rt._ctx
            a.write_buffer('wte.weight', np.ones((32, 16), dtype=np.float32))
            np.testing.assert_array_equal(b.read_buffer('wte.weight'), np.zeros(512))
            with pytest.raises(RuntimeError):
                load_hf_bytes(GPT2_TINY_CONFIG, [b'invalid'], runtime=rt)
            a.dispose()
            pg._ffi.get_lib().poly_ctx_collect(rt._ctx)
            np.testing.assert_array_equal(b.read_buffer('wte.weight'), np.zeros(512))
            np.testing.assert_array_equal(live.numpy(), [19])
        finally:
            for model in models: model.dispose()
            rt.dispose()

    def test_load_from_config_only(self):
        """Load with empty weight files should still create the instance."""
        # Create a dummy weight file (empty tensor list is invalid, so use a valid one)
        wte = np.zeros((32, 16), dtype=np.float32)
        st = make_safetensors({
            'transformer.wte.weight': ('F32', (32, 16), wte)
        })
        inst = load_hf_bytes(GPT2_TINY_CONFIG, [st])
        assert inst is not None
        assert inst.param_count == 16  # wte + wpe + 1 layer (12) + ln_f (2)
        inst.free()

    def test_param_names(self):
        """Verify all expected parameter names exist."""
        wte = np.zeros((32, 16), dtype=np.float32)
        st = make_safetensors({
            'transformer.wte.weight': ('F32', (32, 16), wte)
        })
        inst = load_hf_bytes(GPT2_TINY_CONFIG, [st])

        names = set()
        for i in range(inst.param_count):
            names.add(inst.param_name(i))

        assert 'wte.weight' in names
        assert 'wpe.weight' in names
        assert 'h.0.ln_1.weight' in names
        assert 'h.0.attn.c_attn.weight' in names
        assert 'h.0.mlp.c_fc.weight' in names
        assert 'ln_f.weight' in names
        inst.free()

    def test_weight_loading(self):
        """Verify weights are actually loaded into the instance."""
        wte = np.arange(32 * 16, dtype=np.float32).reshape(32, 16) * 0.001
        st = make_safetensors({
            'transformer.wte.weight': ('F32', (32, 16), wte)
        })
        inst = load_hf_bytes(GPT2_TINY_CONFIG, [st])

        # Find wte.weight and verify data
        for i in range(inst.param_count):
            if inst.param_name(i) == 'wte.weight':
                data = inst.param_data(i)
                np.testing.assert_allclose(data[:5], wte.ravel()[:5], atol=1e-6)
                break
        else:
            pytest.fail('wte.weight not found')
        inst.free()

    def test_model_prefix_stripping(self):
        """Both 'transformer.' and 'model.' prefixes should be stripped."""
        wte = np.ones((32, 16), dtype=np.float32)
        st = make_safetensors({
            'model.wte.weight': ('F32', (32, 16), wte)
        })
        inst = load_hf_bytes(GPT2_TINY_CONFIG, [st])
        # Should still load because 'model.' prefix is stripped
        assert inst is not None
        inst.free()


class TestHFLoadEdgeCases:
    def test_zero_attention_heads_reject_without_crashing_runtime(self):
        import subprocess
        import sys
        code = '''
from polygrad import create, Model
import json
rt = create(device='interp')
try:
    Model.from_hf(config_json=json.dumps(dict(model_type='gpt2', vocab_size=8,
        n_embd=4, n_head=0, n_layer=1, n_positions=2)), weight_bytes_list=[], runtime=rt)
except RuntimeError:
    assert rt.Tensor([19.0]).numpy().tolist() == [19.0]
else:
    raise AssertionError('zero attention heads accepted')
finally:
    rt.dispose()
'''
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        assert result.returncode == 0, (result.returncode, result.stderr)

    @pytest.mark.parametrize('dtype,shape,npdtype', [('F32', (1,), np.float32), ('F64', (32, 16), np.float64)])
    def test_weight_conversion_or_copy_failure_rejects_model(self, dtype, shape, npdtype):
        shard = make_safetensors({'transformer.wte.weight': (dtype, shape, np.ones(shape, dtype=npdtype))})
        with pytest.raises(RuntimeError, match='NULL'):
            load_hf_bytes(GPT2_TINY_CONFIG, [shard])

    def test_quantized_gguf_weights_match_pinned_bit_planes(self):
        fixture = json.loads((Path(__file__).resolve().parents[2] / 'test/fixtures/gguf_quantized_blocks.json').read_text())
        def string(s):
            b = s.encode()
            return struct.pack('<Q', len(b)) + b
        for case in fixture['cases']:
            data = b'GGUF' + struct.pack('<IQQ', 3, 1, 5)
            data += string('general.architecture') + struct.pack('<I', 8) + string('gpt2')
            for key, value in [('embedding_length', 32), ('attention.head_count', 2), ('block_count', 1), ('context_length', 2)]:
                data += string('gpt2.' + key) + struct.pack('<II', 4, value)
            data += string('token_embd.weight') + struct.pack('<IQQIQ', 2, 32, 8, case['type'], 0)
            data += bytes((-len(data)) % 32)
            repeats = 256 // len(case['values'])
            data += bytes(case['bytes']) * repeats
            model = Model.from_gguf(data, max_batch=1, max_seq_len=2)
            try:
                np.testing.assert_array_equal(model.read_buffer('wte.weight').reshape(-1), case['values'] * repeats)
            finally:
                model.free()

    @pytest.mark.parametrize('shards', [[struct.pack('<Q', 1) + b'{'], [b'']])
    def test_malformed_shard_fails_before_model_publication(self, shards):
        with pytest.raises(RuntimeError, match='NULL'):
            load_hf_bytes(GPT2_TINY_CONFIG, shards)

    def test_gguf_invalid_length_fails_before_model_publication(self):
        data = b'GGUF' + struct.pack('<IQQQ', 3, 0, 1, 2**64 - 1) + bytes(32)
        with pytest.raises(RuntimeError, match='NULL'):
            Model.from_gguf(data)

    def test_unsupported_model_type(self):
        """Unsupported model types should raise."""
        config = json.dumps({'model_type': 'llama', 'vocab_size': 100})
        with pytest.raises(RuntimeError, match='NULL'):
            load_hf_bytes(config, [])

    def test_attn_bias_ignored(self):
        """Non-parameter buffers (attn.bias) should be silently ignored."""
        bias = np.zeros((1, 1, 8, 8), dtype=np.float32)
        st = make_safetensors({
            'transformer.h.0.attn.bias': ('F32', (1, 1, 8, 8), bias)
        })
        inst = load_hf_bytes(GPT2_TINY_CONFIG, [st])
        assert inst is not None
        inst.free()

    def test_lm_head_weight_skipped(self):
        """lm_head.weight should be skipped (weight tying with wte)."""
        wte = np.ones((32, 16), dtype=np.float32) * 0.5
        lm_head = np.ones((32, 16), dtype=np.float32) * 0.9
        st = make_safetensors({
            'transformer.wte.weight': ('F32', (32, 16), wte),
            'lm_head.weight': ('F32', (32, 16), lm_head)
        })
        inst = load_hf_bytes(GPT2_TINY_CONFIG, [st])
        # wte should have been loaded with the wte data, not lm_head
        for i in range(inst.param_count):
            if inst.param_name(i) == 'wte.weight':
                data = inst.param_data(i)
                np.testing.assert_allclose(data[0], 0.5, atol=1e-6)
                break
        inst.free()

    def test_f16_weights(self):
        """F16 weights should be decoded and converted to F32."""
        wte = np.zeros((32, 16), dtype=np.float16)
        wte[0, 0] = 1.0
        wte[0, 1] = -0.5
        st = make_safetensors({
            'transformer.wte.weight': ('F16', (32, 16), wte)
        })
        inst = load_hf_bytes(GPT2_TINY_CONFIG, [st])
        for i in range(inst.param_count):
            if inst.param_name(i) == 'wte.weight':
                data = inst.param_data(i)
                np.testing.assert_allclose(data[0], 1.0, atol=1e-3)
                np.testing.assert_allclose(data[1], -0.5, atol=1e-3)
                break
        inst.free()

    def test_multiple_weight_files(self):
        """Multiple safetensors files (sharded) should all be loaded."""
        wte = np.ones((32, 16), dtype=np.float32) * 0.1
        wpe = np.ones((8, 16), dtype=np.float32) * 0.2
        st1 = make_safetensors({
            'transformer.wte.weight': ('F32', (32, 16), wte)
        })
        st2 = make_safetensors({
            'transformer.wpe.weight': ('F32', (8, 16), wpe)
        })
        inst = load_hf_bytes(GPT2_TINY_CONFIG, [st1, st2])

        found_wte = found_wpe = False
        for i in range(inst.param_count):
            name = inst.param_name(i)
            data = inst.param_data(i)
            if name == 'wte.weight':
                np.testing.assert_allclose(data[0], 0.1, atol=1e-6)
                found_wte = True
            elif name == 'wpe.weight':
                np.testing.assert_allclose(data[0], 0.2, atol=1e-6)
                found_wpe = True

        assert found_wte, 'wte.weight not loaded'
        assert found_wpe, 'wpe.weight not loaded'
        inst.free()


class TestHFMultiLayer:
    def test_3_layer_model(self):
        """3-layer GPT-2 should have 40 parameters."""
        config = json.dumps({
            'model_type': 'gpt2',
            'vocab_size': 64,
            'n_embd': 32,
            'n_head': 4,
            'n_layer': 3,
            'n_positions': 16,
            'layer_norm_epsilon': 1e-5
        })
        st = make_safetensors({
            'transformer.wte.weight': ('F32', (64, 32), np.zeros((64, 32), dtype=np.float32))
        })
        inst = load_hf_bytes(config, [st])
        # 2 (wte+wpe) + 3*12 (layers) + 2 (ln_f) = 40
        assert inst.param_count == 40
        inst.free()


class TestGetVocabSize:
    def test_from_instance(self):
        st = make_safetensors({
            'transformer.wte.weight': ('F32', (32, 16), np.zeros((32, 16), dtype=np.float32))
        })
        inst = load_hf_bytes(GPT2_TINY_CONFIG, [st])
        assert _get_vocab_size(inst) == 32
        inst.free()


class TestGenerateTokenDtype:
    class FakeModel:
        param_count = 1
        buf_count = 1

        @staticmethod
        def param_name(_index):
            return 'wte.weight'

        @staticmethod
        def param_shape(_index):
            return [4, 2]

        @staticmethod
        def buf_name(_index):
            return 'x'

        @staticmethod
        def buf_shape(_index):
            return [1, 4]

        @staticmethod
        def forward(**inputs):
            assert inputs['x'].dtype == np.int32
            assert inputs['positions'].dtype == np.int32
            logits = np.zeros((1, 4, 4), dtype=np.float32)
            logits[..., 2] = 1.0
            return {'output': logits}

    def test_integer_tokens_are_normalized_to_int32(self):
        result = generate(
            self.FakeModel(), np.array([[0, 1]], dtype=np.int64),
            max_new_tokens=1, top_k=1,
        )
        assert result.dtype == np.int32
        np.testing.assert_array_equal(result[:, :2], [[0, 1]])

    def test_float_tokens_are_rejected(self):
        with pytest.raises(TypeError, match='tokens must have an integer dtype'):
            generate(
                self.FakeModel(), np.array([[0.0, 1.0]], dtype=np.float32),
                max_new_tokens=1,
            )
