"""Tests for HuggingFace model loading."""

import ctypes
from contextlib import ExitStack
import json
import struct
from pathlib import Path
import numpy as np
import pytest
from polygrad.hf import generate, load_hf_bytes, _find_safetensors, _get_vocab_size
from polygrad.model import Model


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


LLAMA_CASES = json.loads((Path(__file__).resolve().parents[2] / 'test/fixtures/llama.json').read_text())['cases']


def llama_weights(case):
    return {name: np.array([(1.0 if len(shape) == 1 else 0.0) +
                            ((i * 7 + sum(name.encode())) % 23 - 11) * 0.017
                            for i in range(int(np.prod(shape)))], np.float32).reshape(shape)
            for name, shape in case['weights'].items()}


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
