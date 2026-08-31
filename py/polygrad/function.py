"""Pinned tinygrad-style value-producing ``function`` capture."""

from __future__ import annotations

import functools

from . import _ffi
from .tensor import Tensor
from .uop.ops import UOp


def _state_values(obj):
    """Port of ``nn.state.get_state_dict(..., tensor_type=(Tensor, UOp))``.

    Ordering is observable because it assigns FUNCTION PARAM slots. Keep
    Python insertion order and recurse through ``__dict__`` exactly like the
    pinned helper (tinygrad/nn/state.py:87-107).
    """
    if isinstance(obj, (Tensor, UOp)):
        return [obj]
    if hasattr(obj, '_asdict'):
        return _state_values(obj._asdict())
    if hasattr(obj, '__dict__'):
        return _state_values(obj.__dict__)
    out = []
    if isinstance(obj, (list, tuple)):
        for value in obj:
            out.extend(_state_values(value))
    elif isinstance(obj, dict):
        for value in obj.values():
            out.extend(_state_values(value))
    return out


class _function:
    depth = 0

    def __init__(self, fxn, *, precompile, precompile_backward,
                 allow_implicit, grad_fxn):
        if grad_fxn is not None:
            raise NotImplementedError(
                'Polygrad function grad_fxn callbacks are not yet implemented')
        self.fxn = fxn
        self.precompile = bool(precompile)
        self.precompile_backward = bool(precompile_backward)
        self.allow_implicit = bool(allow_implicit)
        self.grad_fxn = None

    def __get__(self, obj, objtype=None):
        return functools.partial(self.__call__, obj) if obj is not None else self

    def __call__(self, *args, **kwargs):
        values = _state_values((args, kwargs))
        inputs = []
        for value in values:
            if isinstance(value, Tensor):
                logical = value._core_uop_logical_raw(value._tensor)
                physical = value._core_uop_physical_raw(value._tensor)
                ctx = value._ctx
            else:
                logical = physical = value.raw
                ctx = value.ctx
            if not logical or not physical:
                raise RuntimeError('function input lacks a logical or physical root')
            inputs.append((ctx, logical, physical))

        _function.depth += 1
        try:
            ret = self.fxn(*args, **kwargs)
        finally:
            _function.depth -= 1
        if isinstance(ret, Tensor):
            results = (ret,)
            tuple_result = False
        elif isinstance(ret, tuple) and all(isinstance(x, Tensor) for x in ret):
            results = ret
            tuple_result = True
        else:
            raise RuntimeError(f'function return type {type(ret)} not supported')
        if not results:
            raise RuntimeError('function cannot return an empty tuple')
        ctx = results[0]._ctx
        if any(t._ctx != ctx for t in results) or any(x[0] != ctx for x in inputs):
            raise ValueError('function tensors must share a context')

        result_arr = (_ffi._ptr * len(results))(*[t._tensor for t in results])
        logical_arr = ((_ffi._ptr * len(inputs))(*[x[1] for x in inputs])
                       if inputs else None)
        physical_arr = ((_ffi._ptr * len(inputs))(*[x[2] for x in inputs])
                        if inputs else None)
        output_arr = (_ffi._ptr * len(results))()
        name = getattr(self.fxn, '__qualname__', None) or type(self.fxn).__qualname__
        rc = _ffi._lib.poly_tensor_function(
            ctx, result_arr, len(results), logical_arr, physical_arr, len(inputs),
            name.encode('utf-8'), self.allow_implicit,
            self.precompile, self.precompile_backward, output_arr,
        )
        if rc == -2:
            raise RuntimeError(
                f'function {name} has implicit buffer(s), but allow_implicit=False')
        if rc != 0:
            raise RuntimeError(f'poly_tensor_function failed for {name}')
        outputs = tuple(Tensor(
            _ctx=ctx, _tensor=output_arr[i], _dtype=results[i]._dtype_str,
            _device=results[i]._device,
        ) for i in range(len(results)))
        return outputs if tuple_result else outputs[0]


def function(fxn=None, *, precompile=False, precompile_backward=False,
             allow_implicit=False, grad_fxn=None):
    if fxn is None:
        return lambda f: _function(
            f, precompile=precompile, precompile_backward=precompile_backward,
            allow_implicit=allow_implicit, grad_fxn=grad_fxn)
    return _function(
        fxn, precompile=precompile, precompile_backward=precompile_backward,
        allow_implicit=allow_implicit, grad_fxn=grad_fxn)


__all__ = ['function']
