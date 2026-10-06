"""C-backed tokenization, independent of a Tensor runtime or device."""

import ctypes as C
import operator
import warnings
import weakref

from . import _ffi


class Tokenizer:
    def __init__(self, handle):
        if not handle:
            raise ValueError('null tokenizer handle')
        self._handle = handle
        self._lib = _ffi.get_lib()
        self._finalizer = weakref.finalize(self, self._lib.poly_tokenizer_free, handle)

    @classmethod
    def from_json(cls, data, *, strict=True):
        """Load supported byte-level BPE; strict=False may skip NFC with a warning."""
        if not isinstance(strict, bool):
            raise TypeError('strict must be a bool')
        if isinstance(data, str):
            data = data.encode('utf-8')
        raw = data if isinstance(data, bytes) else memoryview(data).tobytes()
        if len(raw) > 2**31 - 1:
            raise ValueError('tokenizer JSON exceeds int32 length')
        diagnostic = C.create_string_buffer(256)
        handle = _ffi.get_lib().poly_tokenizer_from_json_ex(raw, len(raw), strict, diagnostic, len(diagnostic))
        message = diagnostic.value.decode('utf-8', errors='replace')
        if not handle:
            raise ValueError(message or 'tokenizer JSON import failed')
        tok = cls(handle)
        try:
            if message:
                warnings.warn(message, UserWarning, stacklevel=2)
        except BaseException:
            tok.free()
            raise
        return tok

    @classmethod
    def from_gguf(cls, data):
        """Load tokenizer metadata from GGUF bytes, without loading model weights."""
        # GGUF can contain gigabytes of weights. Keep immutable bytes borrowed
        # for the decode call; only mutable buffers need a private snapshot.
        raw = data if isinstance(data, bytes) else memoryview(data).tobytes()
        lib, decoded = _ffi.get_lib(), _ffi._ptr()
        try:
            if lib.poly_gguf_decode(raw, len(raw), C.byref(decoded)) != 0 or not decoded:
                raise ValueError('invalid GGUF tokenizer data')
            handle = lib.poly_tokenizer_from_gguf(decoded)
            if not handle:
                raise ValueError('GGUF tokenizer metadata missing or unsupported')
            return cls(handle)
        finally:
            if decoded:
                lib.poly_gguf_decoded_free(decoded)

    def _live(self):
        if not self._finalizer.alive:
            raise RuntimeError('tokenizer has been freed')
        return self._handle

    def encode(self, text):
        if not isinstance(text, str):
            raise TypeError('text must be a string')
        if '\0' in text:
            raise ValueError('tokenizer text cannot contain NUL')
        handle, raw = self._live(), text.encode('utf-8')
        count = self._lib.poly_tokenize(handle, raw, None, 2**31 - 1)
        if count < 0:
            raise ValueError('tokenization failed')
        ids = (C.c_int * count)()
        if self._lib.poly_tokenize(handle, raw, ids, count) != count:
            raise RuntimeError('tokenization failed')
        return list(ids)

    def decode(self, ids):
        values = [operator.index(value) for value in ids]
        if len(values) > 2**31 - 1 or any(not -2**31 <= value < 2**31 for value in values):
            raise ValueError('token IDs and count must fit int32')
        handle = self._live()
        array = (C.c_int * len(values))(*values)
        size = self._lib.poly_detokenize(handle, array, len(values), None, 0)
        if size < 0 or size >= 2**31 - 1:
            raise ValueError('detokenization failed or output too large')
        output = C.create_string_buffer(size + 1)
        if self._lib.poly_detokenize(handle, array, len(values), output, len(output)) != size:
            raise RuntimeError('detokenization failed')
        return output.raw[:size].decode('utf-8', errors='replace')

    @property
    def vocab_size(self):
        return self._lib.poly_tokenizer_vocab_size(self._live())

    @property
    def bos_id(self):
        return self._lib.poly_tokenizer_bos_id(self._live())

    @property
    def eos_id(self):
        return self._lib.poly_tokenizer_eos_id(self._live())

    def free(self):
        self._finalizer()

    def __enter__(self):
        self._live()
        return self

    def __exit__(self, *_):
        self.free()
