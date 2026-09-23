"""C-backed causal Transformer; generic graph operations remain on Model."""

import ctypes
import operator
import numpy as np

from .. import _ffi
from ..model import Model


class Transformer(Model):
    """C-backed dense causal Transformer. Model methods operate on its owned graph.

    from_model specializes the supplied object in place: aliases refer to the
    same owner. It does not create a second owner of the underlying Model.
    """
    def __init__(self, spec, *, runtime=None, model_type='llama'):
        from . import _build
        model = _build(model_type, spec, runtime, specialize=False)
        try:
            self._adopt(model)
            self._attach()
        except BaseException:
            self.free()
            model.free()
            raise

    def _attach(self):
        error = _ffi.PolyModelError()
        handle = _ffi.get_lib().poly_transformer_from_model(self._ptr, ctypes.byref(error))
        if not handle:
            raise ValueError(bytes(error.message).decode('utf-8', 'replace'))
        self._transformer = handle

    @classmethod
    def from_model(cls, model):
        if not isinstance(model, Model) or not model._ptr:
            raise TypeError('Transformer.from_model requires a live Model')
        if isinstance(model, Transformer):
            return model
        # Successful C adoption is the only ownership transition. Class change
        # keeps the runtime's existing weak owner registration and all aliases.
        error = _ffi.PolyModelError()
        handle = _ffi.get_lib().poly_transformer_from_model(model._ptr, ctypes.byref(error))
        if not handle:
            raise ValueError(bytes(error.message).decode('utf-8', 'replace'))
        model._transformer = handle
        model.__class__ = cls
        return model

    @classmethod
    def load(cls, source, *, runtime=None):
        model = Model.load(source, runtime=runtime)
        try:
            return cls.from_model(model)
        except BaseException:
            model.dispose()
            raise

    @classmethod
    def from_bundle(cls, data, *, runtime=None):
        return cls.load(data, runtime=runtime)

    def free(self):
        if getattr(self, '_transformer', None):
            _ffi.get_lib().poly_transformer_free(self._transformer)
            self._transformer = self._ptr = self._ctx = None
        else:
            super().free()

    def _transformer_handle(self):
        if not self._ptr or not self._transformer:
            raise RuntimeError('Transformer is disposed')
        return self._transformer

    def _transformer_error(self, fallback):
        error = _ffi.get_lib().poly_transformer_last_error(self._transformer_handle())
        message = error.contents.message.decode('utf-8', 'replace').strip() if error else ''
        return RuntimeError(message or fallback)

    def reset(self):
        if _ffi.get_lib().poly_transformer_reset(self._transformer_handle()) != 0:
            raise self._transformer_error('Transformer reset failed')

    @property
    def decode_position(self):
        """Committed token count, or -1 if no valid checked decoder is available."""
        return _ffi.get_lib().poly_transformer_position(self._transformer_handle())

    def append_tokens(self, tokens):
        """Append int32 tokens; return last-token logits [1,vocabulary].

        C owns chunking, capacity checks and the cursor. After a failed execution
        or a direct state-changing Model operation, reset_transient is required.
        """
        return self._decode_tokens(tokens, _ffi.get_lib().poly_transformer_append)

    def prefill_tokens(self, tokens):
        """Process a complete int32 prompt, reusing its matching committed prefix.

        Requires a prefix-reusable decoder. Returns last-token logits [1,vocabulary].
        The final prompt token is always recomputed. Invalid state still requires reset.
        """
        return self._decode_tokens(tokens, _ffi.get_lib().poly_transformer_prefill)

    def rewind(self, position):
        """Discard tokens after position without clearing cache bytes; dense decoders only."""
        position = operator.index(position)
        if not 0 <= position <= 2**31 - 1:
            raise ValueError('rewind position must fit a nonnegative int32')
        if _ffi.get_lib().poly_transformer_rewind(self._transformer_handle(), position) != 0:
            raise self._transformer_error('decoder rewind failed')

    def generate(self, tokens, *, temperature=0.0, max_tokens=None):
        """Yield token IDs; C owns prompt reuse, sampling, capacity and KV state."""
        tokens = np.asarray(tokens)
        if tokens.dtype != np.int32 or not (tokens.ndim == 1 or tokens.ndim == 2 and tokens.shape[0] == 1):
            raise TypeError('generate expects int32 tokens with shape [N] or [1,N]')
        tokens = np.ascontiguousarray(tokens)
        if tokens.size > 2**31 - 1:
            raise ValueError('too many tokens')
        if max_tokens is not None:
            max_tokens = operator.index(max_tokens)
            if max_tokens < 0: raise ValueError('max_tokens must be nonnegative')
        temperature = float(temperature)
        if not np.isfinite(temperature) or temperature < 0:
            raise ValueError('temperature must be finite and nonnegative')
        if max_tokens == 0:
            return
        handle = self._transformer_handle()
        lib = _ffi.get_lib()
        if lib.poly_transformer_start(handle, tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), tokens.size) != 0:
            raise self._transformer_error('Transformer prefill failed')
        count, token = 0, ctypes.c_int32()
        while max_tokens is None or count < max_tokens:
            self._transformer_handle()  # Disposal between yields must not enter C.
            rc = lib.poly_transformer_next(handle, temperature, ctypes.byref(token))
            if rc == 1: return
            if rc != 0: raise self._transformer_error('Transformer sampling failed')
            count += 1
            yield token.value

    def _decode_tokens(self, tokens, fn):
        tokens = np.asarray(tokens)
        if tokens.dtype != np.int32 or not (tokens.ndim == 1 or tokens.ndim == 2 and tokens.shape[0] == 1):
            raise TypeError('decoder expects int32 tokens with shape [N] or [1,N]')
        tokens = np.ascontiguousarray(tokens)
        if tokens.size > 2**31 - 1:
            raise ValueError('too many tokens')
        transformer = self._transformer_handle()
        vocab = _ffi.get_lib().poly_transformer_vocab(transformer)
        if vocab <= 0:
            raise RuntimeError('Model has no checked decoder entrypoints')
        result = np.empty((1, vocab), dtype=np.float32)
        if fn(transformer,
                tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), tokens.size,
                result.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), vocab) != 0:
            raise self._transformer_error('decoder append failed; reset_transient before retrying')
        return result
