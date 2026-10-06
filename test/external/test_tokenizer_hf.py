"""Opt-in, pinned real tokenizer.json checks; no model weights are downloaded."""
import ctypes as C
import hashlib
import json
import random
import unicodedata

import pytest
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'py/tests'))
from test_tokenizer import tokenizer_lib, json_spec, load_json
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer


@pytest.mark.parametrize('repo,revision,digest', [
    ('openai-community/gpt2', '607a30d783dfa663caf39e06633721c8d4cfcd7e',
     '8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6'),
    ('Qwen/Qwen2.5-0.5B', '060db6499f32faf8b98477b0a26969ef7d8b9987',
     'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539'),
])
def test_hf_byte_bpe(repo, revision, digest):
    data = Path(hf_hub_download(repo, 'tokenizer.json', revision=revision, cache_dir=os.environ.get('POLY_TOKENIZER_CACHE'))).read_bytes()
    assert hashlib.sha256(data).hexdigest() == digest
    reference = original = Tokenizer.from_str(data.decode())
    spec = json.loads(data)
    lib = tokenizer_lib()
    handle, diagnostic = load_json(lib, spec)
    if spec.get('normalizer') == {'type': 'NFC'}:
        assert not handle and 'NFC' in diagnostic
        handle, diagnostic = load_json(lib, spec, strict=False)
        assert handle and 'NFC normalization skipped' in diagnostic
        # Compare best effort with exactly the declared approximation.
        raw_spec = dict(spec, normalizer=None)
        reference = Tokenizer.from_str(json.dumps(raw_spec))
    else:
        assert handle and not diagnostic, diagnostic
    try:
        texts = ['hello...\n\nworld', 'cafe\u0301', 'a\u0315\u0300', '각', 'WE\'RE HERE', ' 123456789', "a'ſfoo"]
        texts += [a + b + c for a in ['a', 'hello', '世界', ' ', '.', '123']
                  for b in ['🙂', '。', '—', '\u0301', '\u00a0', '\u2003', '１２', '١٢', '\r\n', "'RE"]
                  for c in ['b', 'world', '你好', ' ', '.', '456']]
        texts += ['ab<end>' * 5000, 'hello ' * 5000]
        texts += ['hello' + t['content'] + 'cafe\u0301' for t in spec['added_tokens']]
        rng = random.Random(0)
        alphabet = "abcXYZ'sſtrevlmd 123\t\n\r\u00a0\u2003\u0085\u2028éß世🙂—。\u0301\u0315\u0300１２١٢"
        texts += [''.join(rng.choices(alphabet, k=rng.randrange(1, 60))) for _ in range(1000)]
        normalization_differences = 0
        for text in texts:
            expected = reference.encode(text).ids
            original_ids = original.encode(text).ids
            if text == unicodedata.normalize('NFC', text):
                assert expected == original_ids
            normalization_differences += expected != original_ids
            out = (C.c_int * (len(text.encode()) * 3 + 1))()
            n = lib.poly_tokenize(handle, text.encode(), out, len(out))
            assert n >= 0 and list(out[:n]) == expected, (repo, repr(text), list(out[:max(n, 0)]), expected)
            size = lib.poly_detokenize(handle, out, n, None, 0)
            decoded = C.create_string_buffer(size + 1)
            assert lib.poly_detokenize(handle, out, n, decoded, size + 1) == size
            assert decoded.value.decode(errors='replace') == reference.decode(expected, skip_special_tokens=False)
        if spec.get('normalizer'):
            assert normalization_differences > 0, 'the best-effort warning must describe a real difference'
    finally:
        lib.poly_tokenizer_free(handle)

@pytest.mark.parametrize('regex,prefix', [(True, False), (True, True), (False, False), (False, True)])
def test_synthetic_bpe_against_hf(regex, prefix):
    spec = json_spec()
    spec['pre_tokenizer'].update(use_regex=regex, add_prefix_space=prefix)
    # Exercise both merge serialization forms and literal added-token decoding.
    spec['model']['merges'] = ['b c', 'a b']
    reference = Tokenizer.from_str(json.dumps(spec))
    lib = tokenizer_lib()
    handle, diagnostic = load_json(lib, spec)
    assert handle and not diagnostic
    try:
        for text in ['abc', 'ab bc', 'abc<end>ab', 'é e\u0301', '  a\n\nb', '', '<end>abc']:
            out = (C.c_int * 100)()
            n = lib.poly_tokenize(handle, text.encode(), out, len(out))
            assert list(out[:n]) == reference.encode(text).ids, repr(text)
    finally:
        lib.poly_tokenizer_free(handle)
