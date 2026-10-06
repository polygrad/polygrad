"""C tokenizer contracts shared by all frontends."""
import ctypes as C
import json
from pathlib import Path

import pytest

from polygrad._ffi import get_lib


def test_public_tokenizer_json_lifetime_and_warning():
    import polygrad as pg
    raw = (Path(__file__).resolve().parents[2] / 'test/fixtures/tokenizer_byte_bpe.json').read_bytes()
    with pg.Tokenizer.from_json(raw) as tok:
        text = 'hello world! ' * 5000
        assert tok.decode(tok.encode(text)) == text
        assert tok.encode('') == [] and tok.decode([]) == ''
        assert tok.vocab_size > 256
        assert tok.bos_id == -1
    tok.free()
    with pytest.raises(RuntimeError, match='freed'):
        tok.encode('hello')
    config = json.loads(raw)
    config['normalizer'] = {'type':'NFC'}
    with pytest.raises(ValueError, match='NFC'):
        pg.Tokenizer.from_json(json.dumps(config))
    with pg.Runtime(device='cpu') as rt, pytest.warns(UserWarning, match='NFC'):
        with rt.Tokenizer.from_json(json.dumps(config), strict=False) as tok:
            assert tok.decode(tok.encode('e\u0301')) == 'e\u0301'
    with pytest.raises(TypeError, match='bool'):
        pg.Tokenizer.from_json(raw, strict='false')


def test_public_tokenizer_gguf():
    import struct
    import polygrad as pg
    u32 = lambda n: struct.pack('<I', n)
    u64 = lambda n: struct.pack('<Q', n)
    string = lambda s: u64(len(s.encode())) + s.encode()
    raw = b'GGUF' + u32(3) + u64(0) + u64(4)
    raw += string('tokenizer.ggml.tokens') + u32(9) + u32(8) + u64(4)
    raw += b''.join(map(string, ['a','b','ab','<end>']))
    raw += string('tokenizer.ggml.token_type') + u32(9) + u32(5) + u64(4)
    raw += b''.join(map(u32, [1,1,1,3]))
    raw += string('tokenizer.ggml.pre') + u32(8) + string('llama3')
    raw += string('tokenizer.ggml.eos_token_id') + u32(4) + u32(3)
    raw += b'\0' * (-len(raw) % 32)
    with pg.Tokenizer.from_gguf(raw) as tok:
        assert (tok.vocab_size, tok.bos_id, tok.eos_id) == (4,-1,3)
        assert tok.encode('ab<end>') == [2,3]
        assert tok.decode([2,3]) == 'ab<end>'
    with pytest.raises(ValueError, match='GGUF'):
        pg.Tokenizer.from_gguf(b'bad')


def tokenizer_lib():
    lib = get_lib()
    lib.poly_tokenizer_from_json.argtypes = [C.c_char_p, C.c_int]
    lib.poly_tokenizer_from_json.restype = C.c_void_p
    lib.poly_tokenizer_from_json_ex.argtypes = [C.c_char_p, C.c_int, C.c_int, C.c_char_p, C.c_int]
    lib.poly_tokenizer_from_json_ex.restype = C.c_void_p
    lib.poly_tokenizer_create.argtypes = [C.POINTER(C.c_char_p), C.POINTER(C.c_int), C.c_int]
    lib.poly_tokenizer_create.restype = C.c_void_p
    lib.poly_tokenizer_free.argtypes = [C.c_void_p]
    lib.poly_tokenize.argtypes = [C.c_void_p, C.c_char_p, C.POINTER(C.c_int), C.c_int]
    lib.poly_detokenize.argtypes = [C.c_void_p, C.POINTER(C.c_int), C.c_int, C.c_char_p, C.c_int]
    return lib


@pytest.mark.parametrize('raw', [
    b'{"model":{"type":"BPE","vocab":{"a":0}}}',
    b'{"model":{"type":"WordPiece","vocab":{"a":0}}}',
    b'{"model":{"type":"Unigram","vocab":[["a",0]]}}',
    b'{}', b'invalid', b'',
])
def test_tokenizer_json_is_unsupported(raw, capfd):
    lib = tokenizer_lib()
    tok = lib.poly_tokenizer_from_json(raw, len(raw))
    if tok:
        lib.poly_tokenizer_free(tok)
    assert not tok
    assert "Hugging Face" in capfd.readouterr().err


def test_tokenizer_vocab_long_outputs():
    lib = tokenizer_lib()
    tokens = (C.c_char_p * 4)(b'a', b'b', b'ab', b'<end>')
    types = (C.c_int * 4)(1, 1, 1, 3)
    tok = lib.poly_tokenizer_create(tokens, types, 4)
    assert tok
    try:
        text = b'ab<end>' * 5000
        count = lib.poly_tokenize(tok, text, None, 2**31 - 1)
        assert count == 10000
        ids = (C.c_int * count)()
        assert lib.poly_tokenize(tok, text, ids, count) == count
        assert list(ids) == [2, 3] * 5000
        size = lib.poly_detokenize(tok, ids, count, None, 0)
        assert size == len(text)
        out = C.create_string_buffer(size + 1)
        assert lib.poly_detokenize(tok, ids, count, out, size + 1) == size
        assert out.value == text
    finally:
        lib.poly_tokenizer_free(tok)


def test_tokenizer_json_explicit_merges():
    lib = tokenizer_lib()
    raw = (Path(__file__).resolve().parents[2] / 'test/fixtures/tokenizer_byte_bpe.json').read_bytes()
    tok = lib.poly_tokenizer_from_json(raw, len(raw))
    assert tok, 'valid byte-level BPE must load'
    try:
        ids = (C.c_int * 16)()
        count = lib.poly_tokenize(tok, b'abc<end>ab', ids, len(ids))
        # abc exists in the vocabulary but no merge produces it; bc has priority.
        assert list(ids[:count]) == [97, 257, 259, 256]
    finally:
        lib.poly_tokenizer_free(tok)


def json_spec():
    return json.loads((Path(__file__).resolve().parents[2] /
        'test/fixtures/tokenizer_byte_bpe.json').read_text(encoding='utf-8'))


def load_json(lib, spec, strict=True):
    raw = json.dumps(spec, ensure_ascii=False).encode()
    diagnostic = C.create_string_buffer(256)
    tok = lib.poly_tokenizer_from_json_ex(raw, len(raw), strict, diagnostic, len(diagnostic))
    return tok, diagnostic.value.decode()


def test_tokenizer_json_nfc_is_explicit_best_effort():
    lib = tokenizer_lib()
    spec = json_spec()
    spec['normalizer'] = {'type': 'NFC'}
    tok, error = load_json(lib, spec)
    assert not tok and 'NFC' in error and 'strict=false' in error
    tok, warning = load_json(lib, spec, strict=False)
    assert tok and 'NFC normalization skipped' in warning
    try:
        # Skipping NFC is observable and must not be advertised as exact.
        text = 'e\u0301'.encode()
        ids = (C.c_int * 8)()
        n = lib.poly_tokenize(tok, text, ids, len(ids))
        out = C.create_string_buffer(32)
        lib.poly_detokenize(tok, ids, n, out, len(out))
        assert out.value == text
    finally:
        lib.poly_tokenizer_free(tok)


@pytest.mark.parametrize('field,value,diagnostic', [
    ('normalizer', {'type': 'Lowercase'}, 'normalizer'),
    ('pre_tokenizer', {'type': 'Whitespace'}, 'pre-tokenizer'),
    ('post_processor', {'type': 'TemplateProcessing'}, 'post-processor'),
    ('decoder', {'type': 'WordPiece'}, 'decoder'),
    ('padding', {'length': 32}, 'padding'),
])
def test_tokenizer_json_best_effort_does_not_ignore_other_features(field, value, diagnostic):
    lib = tokenizer_lib()
    spec = json_spec()
    spec[field] = value
    for strict in (True, False):
        tok, error = load_json(lib, spec, strict)
        if tok:
            lib.poly_tokenizer_free(tok)
        assert not tok and diagnostic in error


@pytest.mark.parametrize('mutation', ['ignore_merges', 'added_lstrip', 'bad_id', 'bad_merge',
                                    'missing_byte', 'duplicate_merge'])
def test_tokenizer_json_rejects_invalid_or_unsupported_bpe(mutation):
    lib = tokenizer_lib()
    spec = json_spec()
    if mutation == 'ignore_merges':
        spec['model']['ignore_merges'] = True
    elif mutation == 'added_lstrip':
        spec['added_tokens'][0]['lstrip'] = True
    elif mutation == 'bad_id':
        spec['model']['vocab']['a'] = -1
    elif mutation == 'bad_merge':
        spec['model']['merges'] = [['abc', 'abc']]
    elif mutation == 'missing_byte':
        del spec['model']['vocab']['a']
    else:
        spec['model']['merges'] *= 2
    tok, error = load_json(lib, spec, strict=False)
    if tok:
        lib.poly_tokenizer_free(tok)
    assert not tok and error


def test_tokenizer_json_rejects_mixed_added_token_matching_phases():
    spec = json_spec()
    spec['added_tokens'] = [
        dict(id=256, content='ab', single_word=False, lstrip=False, rstrip=False,
             normalized=False, special=True),
        dict(id=258, content='abc', single_word=False, lstrip=False, rstrip=False,
             normalized=True, special=True),
    ]
    lib = tokenizer_lib()
    tok, error = load_json(lib, spec)
    if tok:
        lib.poly_tokenizer_free(tok)
    # HF extracts unnormalized 'ab' first, whereas a single longest-match pass
    # would choose 'abc'. Neither strict nor best effort may silently do that.
    assert not tok and 'matching phases' in error
