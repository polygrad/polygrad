"""Pinned Llama 3.2 1B forward oracle and Polygrad check in separate processes.

No download or code from the checkpoint is executed. The Make target runs the
reference to completion before loading Polygrad, including between backends.
This checks full-window recomputation, not KV caching or multi-GB bundle export.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource

import numpy as np


WEIGHTS_SHA = '68a2e4be76fa709455a60272fba8e512c02d81c46e6c671cc9449e374fd6809a'
CONFIG_SHA = '8f028e2cd88148fad38c7dece460947681adf6178e46e11ece9742aa49b378dd'
PROMPT = 'In a quiet village near the mountains, there lived a'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def reference(checkpoint, output):
    import torch
    import transformers
    from transformers import AutoTokenizer, LlamaForCausalLM

    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    tokens = np.asarray([tokenizer.encode(PROMPT)], dtype=np.int32)
    assert 10 <= tokens.shape[1] <= 24, tokens.shape
    model = LlamaForCausalLM.from_pretrained(
        checkpoint, local_files_only=True, dtype=torch.float32, attn_implementation='eager').eval()
    arrays = {}
    with torch.no_grad():
        for step in range(3):
            logits = model(torch.from_numpy(tokens.astype(np.int64)), use_cache=False).logits.numpy().copy()
            assert np.isfinite(logits).all(), 'non-finite reference logits'
            arrays[f'tokens{step}'], arrays[f'logits{step}'] = tokens, logits
            token = int(logits[0, -1].argmax())
            print(json.dumps({'reference_step': step, 'length': tokens.shape[1], 'next_token': token}), flush=True)
            tokens = np.concatenate((tokens, np.array([[token]], dtype=np.int32)), axis=1)
    metadata = dict(weights_sha=WEIGHTS_SHA, config_sha=CONFIG_SHA, prompt=PROMPT,
                    transformers=transformers.__version__, torch=torch.__version__, dtype='float32', steps=3)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, metadata=np.array(json.dumps(metadata)), **arrays)
    print('Reference:', output, 'completion:', tokenizer.decode(tokens[0]), flush=True)


def verify(checkpoint, oracle_path, device):
    import polygrad as pg
    from polygrad import _ffi
    from polygrad.hf import load_hf

    print('Library:', _ffi._lib._name, flush=True)
    with np.load(oracle_path, allow_pickle=False) as oracle:
        metadata = json.loads(str(oracle['metadata']))
        assert metadata['weights_sha'] == WEIGHTS_SHA and metadata['config_sha'] == CONFIG_SHA
        assert metadata['dtype'] == 'float32' and metadata['steps'] == 3
        window = oracle['tokens2'].shape[1]
        with pg.create(device=device) as rt:
            model = load_hf(checkpoint, max_seq_len=window, runtime=rt)
            try:
                for step in range(3):
                    tokens, expected = oracle[f'tokens{step}'], oracle[f'logits{step}']
                    padded = np.pad(tokens, ((0, 0), (0, window - tokens.shape[1])))
                    actual = model.forward(tokens=padded)['logits'][:, :tokens.shape[1]]
                    assert rt.GlobalCounters.mem_used_per_device.get(device.upper(), 0) > 0
                    np.testing.assert_allclose(actual, expected, atol=5e-4, rtol=3e-4, equal_nan=False)
                    assert int(actual[0, -1].argmax()) == int(expected[0, -1].argmax())
                    print(json.dumps({'device': device, 'step': step, 'length': tokens.shape[1],
                        'max_abs': float(np.abs(actual - expected).max()),
                        'next_token': int(actual[0, -1].argmax()),
                        'resident_bytes': rt.GlobalCounters.mem_used_per_device}), flush=True)
                # Revisit the first input after later calls: input staging/cache
                # reuse must not leave the last invocation's values behind.
                tokens = oracle['tokens0']
                padded = np.pad(tokens, ((0, 0), (0, window - tokens.shape[1])))
                actual = model.forward(tokens=padded)['logits'][:, :tokens.shape[1]]
                np.testing.assert_allclose(actual, oracle['logits0'], atol=5e-4, rtol=3e-4, equal_nan=False)
            finally:
                model.dispose()
    print(f'PASS: {device}: three full-logit/greedy checks and first-input replay', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('reference', 'verify'))
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--oracle', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    assert digest(args.checkpoint / 'config.json') == CONFIG_SHA, 'config identity mismatch'
    assert digest(args.checkpoint / 'model.safetensors') == WEIGHTS_SHA, 'weight identity mismatch'
    if args.phase == 'reference':
        reference(args.checkpoint, args.oracle)
    else:
        verify(args.checkpoint, args.oracle, args.device)
    print('peak_rss_kib:', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
