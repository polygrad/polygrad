"""GGUF vocabulary-ranked BPE versus the pinned Tinygrad implementation."""
import ctypes as C
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'references/tinygrad_014'))
sys.path.insert(0, str(ROOT / 'py/tests'))
from tinygrad.llm.cli import SimpleTokenizer
from test_tokenizer import tokenizer_lib


def test_gguf_vocabulary_matches_tinygrad():
    # Merge candidates crossing character-class boundaries make a wrong split
    # observable; byte-only vocabularies would hide splitter regressions.
    fragments = ["hello", " world", "123456", "...\n\n", "é", "e\u0301",
                 "你好", "١٢٣٤", "\u00a0x", "😀!", "'S", "'ſ", "\r\n", "  x", "\u2003x"]
    texts = ["", "hello world", "hello...\n\nworld", "<end>hello<end>"] + fragments
    rng = random.Random(42)
    for _ in range(1000):
        extra = chr(rng.choice([0x3b1, 0x323af, 0x11f02, 0x1e4d0, 0x85, 0x2028]))
        texts.append(''.join(rng.choices(fragments, k=8)) + extra)
    bs = [*range(33, 127), *range(161, 173), *range(174, 256)]
    encoder = {b: chr(b) for b in bs}
    encoder.update({b: chr(256+i) for i, b in enumerate(b for b in range(256) if b not in bs)})
    raw_vocab = {bytes([b]): b for b in range(256)}
    for text in texts[:19]:
        raw = text.encode()
        for size in range(2, 9):
            for i in range(len(raw) - size + 1):
                raw_vocab.setdefault(raw[i:i+size], len(raw_vocab))
    normal = {''.join(encoder[b] for b in raw): idx for raw, idx in raw_vocab.items()}
    special_id = len(normal)
    reference = SimpleTokenizer(normal, {'<end>': special_id})
    words = list(normal) + ['<end>']
    lib = tokenizer_lib()
    tokens = (C.c_char_p * len(words))(*(word.encode() for word in words))
    types = (C.c_int * len(words))(*([1] * len(normal) + [3]))
    tok = lib.poly_tokenizer_create(tokens, types, len(words))
    assert tok
    try:
        for text in texts:
            expected = reference.encode(text)
            count = lib.poly_tokenize(tok, text.encode(), None, 2**31 - 1)
            assert count == len(expected), repr(text)
            ids = (C.c_int * count)()
            assert lib.poly_tokenize(tok, text.encode(), ids, count) == count
            assert list(ids) == expected, repr(text)
            size = lib.poly_detokenize(tok, ids, count, None, 0)
            out = C.create_string_buffer(size + 1)
            lib.poly_detokenize(tok, ids, count, out, size + 1)
            assert out.value.decode() == reference.decode(expected)
    finally:
        lib.poly_tokenizer_free(tok)
