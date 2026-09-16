"""Opt-in real-checkpoint gate; downloading is a separate Make target.

Xenova/llama2.c-stories15M at 17c2f1eabe1e163acc15ad35e225794e7b907682
is a trained Llama 2-style model, not a Meta Llama 2/3 checkpoint. This gate
does not certify cached decoding: each invocation recomputes the fixed window.
"""
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

DEVICES = os.environ.get('LLAMA_TEST_DEVICES', 'cpu').split()
if not DEVICES:
    raise ValueError('LLAMA_TEST_DEVICES must name at least one backend')


@pytest.fixture(scope='module')
def checkpoint():
    path = Path(os.environ.get('POLY_LLAMA_CHECKPOINT', 'temp/llama-stories15m'))
    expected = {
        'config.json': '20e21467e9774b66d793f802cfc777e5c3b0eb3e2989183f6771c4a06fd10402',
        'model.safetensors': 'b05d853da20dfc009d6d2905f1fc6281c7b5f1b83baa21e41e9383c9ea7098df',
    }
    for name, digest in expected.items():
        if not (path / name).is_file():
            message = f'{path / name} missing; run make fetch-llama-pretrained'
            if os.environ.get('POLY_REQUIRE_LLAMA') == '1':
                pytest.fail(message, pytrace=False)
            pytest.skip(message)
        assert hashlib.sha256((path / name).read_bytes()).hexdigest() == digest, name
    return path


@pytest.fixture(scope='module')
def reference(checkpoint):
    import torch
    from transformers import LlamaForCausalLM

    torch.set_num_threads(1)
    net = LlamaForCausalLM.from_pretrained(
        checkpoint, local_files_only=True, attn_implementation='eager', dtype=torch.float32).eval()
    # Fixed IDs avoid making tokenizer implementations part of numerical parity.
    # More than ten positions exercises the HF half-split rotary weight layout.
    tokens = np.array([[1, 9038, 2501, 263, 931, 29892, 727, 471, 263, 2217, 1503, 13]], np.int32)
    steps = []
    with torch.no_grad():
        for _ in range(3):
            logits = net(torch.from_numpy(tokens.astype(np.int64)), use_cache=False).logits.numpy().copy()
            steps.append((tokens.copy(), logits))
            token = int(logits[0, -1].argmax())
            tokens = np.concatenate((tokens, np.array([[token]], np.int32)), axis=1)
    return steps


@pytest.mark.parametrize('device', DEVICES)
def test_pretrained_llama_logits_replay_and_bundle(checkpoint, reference, device):
    import polygrad as pg
    from polygrad.hf import load_hf

    window = reference[-1][0].shape[1]
    with pg.create(device=device) as rt:
        model = load_hf(checkpoint, max_seq_len=window, runtime=rt)
        try:
            for tokens, expected in reference:
                # Padding cannot affect the preceding logits in a causal model.
                padded = np.pad(tokens, ((0, 0), (0, window - tokens.shape[1])))
                actual = model.forward(tokens=padded)['logits'][:, :tokens.shape[1]]
                resident = rt.GlobalCounters.mem_used_per_device
                assert resident.get(device.upper(), 0) > 0, resident
                np.testing.assert_allclose(actual, expected, atol=3e-4, rtol=3e-4)
                assert int(actual[0, -1].argmax()) == int(expected[0, -1].argmax())
                print(f'{device}: length={tokens.shape[1]} max_abs={np.max(np.abs(actual-expected)):.8g}'
                      f' next_token={int(actual[0, -1].argmax())}')
            restored = rt.Model.load(model.save(include_optimizer=False))
            try:
                model.dispose()
                actual = restored.forward(tokens=reference[-1][0])['logits']
                np.testing.assert_allclose(actual, reference[-1][1], atol=3e-4, rtol=3e-4)
            finally:
                restored.dispose()
        finally:
            model.dispose()
