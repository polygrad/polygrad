"""
End-to-end HuggingFace GPT-2 loading and text generation test.

Downloads openai-community/gpt2 (124M), loads via poly_hf_load,
runs forward pass, validates logits against HF Transformers reference,
and generates text.

Requires: huggingface_hub, transformers, torch, numpy
"""

import numpy as np
import pytest
import os
import importlib

from polygrad.hf import download_hf, load_hf, generate, _get_vocab_size


MODEL_ID = 'openai-community/gpt2'
PROMPT = 'The meaning of life is'


@pytest.fixture(scope='module', autouse=True)
def required_dependencies():
    # Fixture-time admission preserves all eight test identities. Release
    # gates require the reference implementation, not a collection skip.
    for name in ('huggingface_hub', 'transformers', 'torch'):
        try:
            importlib.import_module(name)
        except ImportError as exc:
            message = f'HF E2E requires {name}: {exc}'
            if os.environ.get('POLY_REQUIRE_HF') == '1':
                pytest.fail(message, pytrace=False)
            pytest.skip(message)


@pytest.fixture(scope='module')
def model_path():
    """Download GPT-2 124M once for all tests in this module."""
    return download_hf(MODEL_ID)


@pytest.fixture(scope='module')
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_ID)


@pytest.fixture(scope='module')
def instance(model_path):
    """Load GPT-2 into polygrad with seq_len=128 for fast tests."""
    return load_hf(model_path, max_batch=1, max_seq_len=128)


class TestGPT2Load:
    def test_param_count(self, instance):
        """GPT-2 124M has 148 parameter tensors (wte + wpe + 12 layers * 12 + ln_f * 2)."""
        n = instance.param_count
        assert n == 148, f'expected 148 params, got {n}'

    def test_vocab_size(self, instance):
        assert _get_vocab_size(instance) == 50257

    def test_wte_shape(self, instance):
        for i in range(instance.param_count):
            if instance.param_name(i) == 'wte.weight':
                shape = instance.param_shape(i)
                assert shape == [50257, 768], f'wte shape: {shape}'
                return
        pytest.fail('wte.weight not found')

    def test_wte_not_zero(self, instance):
        """Verify weights were actually loaded (not all zeros)."""
        for i in range(instance.param_count):
            if instance.param_name(i) == 'wte.weight':
                data = instance.param_data(i)
                assert np.abs(data).max() > 0.01, 'wte.weight is all zeros'
                return
        pytest.fail('wte.weight not found')


class TestGPT2Forward:
    def test_forward_produces_logits(self, instance, tokenizer):
        tokens = tokenizer.encode(PROMPT)
        max_seq_len = 128

        x = np.zeros((1, max_seq_len), dtype=np.int32)
        x[0, :len(tokens)] = tokens

        positions = np.arange(max_seq_len, dtype=np.int32).reshape(1, -1)
        outputs = instance.forward(x=x, positions=positions)
        logits = outputs.get('output')
        assert logits is not None, 'no output buffer'

        logits = logits.reshape(1, max_seq_len, 50257)
        assert logits.shape == (1, 128, 50257)
        assert np.all(np.isfinite(logits)), 'logits contain NaN/Inf'

    def test_logits_match_transformers(self, instance, tokenizer, model_path):
        """Compare polygrad logits against HF Transformers reference."""
        import torch
        from transformers import GPT2LMHeadModel

        tokens = tokenizer.encode(PROMPT)
        max_seq_len = 128

        # Polygrad forward
        x = np.zeros((1, max_seq_len), dtype=np.int32)
        x[0, :len(tokens)] = tokens
        positions = np.arange(max_seq_len, dtype=np.int32).reshape(1, -1)
        outputs = instance.forward(x=x, positions=positions)
        poly_logits = outputs['output'].reshape(1, max_seq_len, 50257)

        # Get logits at last real token position
        last_pos = len(tokens) - 1
        poly_next = poly_logits[0, last_pos, :]

        # HF Transformers reference
        hf_model = GPT2LMHeadModel.from_pretrained(model_path)
        hf_model.eval()
        with torch.no_grad():
            input_ids = torch.tensor([tokens], dtype=torch.long)
            hf_out = hf_model(input_ids)
            hf_next = hf_out.logits[0, last_pos, :].numpy()

        # Compare top-5 predictions (should agree on ranking)
        poly_top5 = np.argsort(poly_next)[-5:][::-1]
        hf_top5 = np.argsort(hf_next)[-5:][::-1]

        print(f'Polygrad top-5: {poly_top5} = {[tokenizer.decode([t]) for t in poly_top5]}')
        print(f'HF top-5:       {hf_top5} = {[tokenizer.decode([t]) for t in hf_top5]}')

        # Top-1 must match
        assert poly_top5[0] == hf_top5[0], (
            f'top-1 mismatch: polygrad={poly_top5[0]} '
            f'({tokenizer.decode([poly_top5[0]])}) vs '
            f'hf={hf_top5[0]} ({tokenizer.decode([hf_top5[0]])})'
        )

        # At least 3 of top-5 should overlap
        overlap = len(set(poly_top5) & set(hf_top5))
        assert overlap >= 3, f'only {overlap}/5 top tokens match'

        # Logit values should be close (not exact due to float32 accumulation order)
        cos_sim = np.dot(poly_next, hf_next) / (
            np.linalg.norm(poly_next) * np.linalg.norm(hf_next) + 1e-8)
        print(f'Cosine similarity: {cos_sim:.6f}')
        assert cos_sim > 0.99, f'logit cosine similarity too low: {cos_sim}'


class TestGPT2Generate:
    def test_greedy_generation(self, instance, tokenizer):
        """Generate text with temperature=0.01 (near-greedy) and verify it's coherent."""
        tokens = tokenizer.encode(PROMPT)
        token_array = np.array(tokens, dtype=np.int32).reshape(1, -1)

        result = generate(instance, token_array, max_new_tokens=20,
                         temperature=0.01, top_k=1)
        generated_ids = result[0].astype(int).tolist()
        text = tokenizer.decode(generated_ids)

        print(f'Generated: {text}')

        # Basic sanity: text should be longer than prompt
        assert len(text) > len(PROMPT)
        # Should start with the prompt
        assert text.startswith(PROMPT) or PROMPT.rstrip() in text
        # Should contain mostly printable characters
        printable_ratio = sum(c.isprintable() or c.isspace() for c in text) / len(text)
        assert printable_ratio > 0.9, f'too many non-printable chars: {printable_ratio}'

    def test_sampling_diversity(self, instance, tokenizer):
        """Two runs with temperature=0.8 should produce different outputs."""
        tokens = tokenizer.encode(PROMPT)
        token_array = np.array(tokens, dtype=np.int32).reshape(1, -1)

        np.random.seed(42)
        r1 = generate(instance, token_array, max_new_tokens=10,
                      temperature=0.8, top_k=40)
        np.random.seed(123)
        r2 = generate(instance, token_array, max_new_tokens=10,
                      temperature=0.8, top_k=40)

        t1 = tokenizer.decode(r1[0].astype(int).tolist())
        t2 = tokenizer.decode(r2[0].astype(int).tolist())
        print(f'Sample 1: {t1}')
        print(f'Sample 2: {t2}')

        # With different seeds, outputs should differ
        assert t1 != t2, 'two samples with different seeds produced identical text'
