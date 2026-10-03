"""C tokenizer contracts shared by all frontends."""
import ctypes as C

import pytest

from polygrad._ffi import get_lib


def tokenizer_lib():
    lib = get_lib()
    lib.poly_tokenizer_from_json.argtypes = [C.c_char_p, C.c_int]
    lib.poly_tokenizer_from_json.restype = C.c_void_p
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
