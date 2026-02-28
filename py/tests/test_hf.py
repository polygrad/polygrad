"""Tests for HuggingFace model loading."""

import ctypes
import json
import struct
import numpy as np
import pytest
from polygrad.hf import load_hf_bytes, _find_safetensors, _get_vocab_size
from polygrad.instance import Instance


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


GPT2_TINY_CONFIG = json.dumps({
    'model_type': 'gpt2',
    'vocab_size': 32,
    'n_embd': 16,
    'n_head': 2,
    'n_layer': 1,
    'n_positions': 8,
    'layer_norm_epsilon': 1e-5
})


class TestHFLoadBasic:
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
