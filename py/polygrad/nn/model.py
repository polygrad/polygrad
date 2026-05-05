"""nn.model -- lightweight export authoring over lazy Tensor graphs.

This mirrors Keras' functional boundary without making normal tinygrad-style
objects inherit from a base class: users build tensors lazily, then name the
roots that become portable Instance entrypoints.
"""

import ctypes

from .. import _ffi
from ..device import Device
from ..instance import Instance, ROLE_INPUT, ROLE_OUTPUT, ROLE_PARAM, ROLE_TARGET
from ..tensor import Tensor, _dtype_id, _int64_array, _ptr_value
from .state import get_state_dict


def _shape_tuple(shape):
    if shape is None:
        raise ValueError("shape is required for exported ABI tensors")
    if isinstance(shape, int):
        return (shape,)
    return tuple(int(x) for x in shape)


def _normalize_named_tensors(value, default_name):
    if value is None:
        return {}
    if isinstance(value, Tensor):
        return {default_name: value}
    return dict(value)


def _register_named_tensor(ctx, name, shape, dtype, role, device=None):
    shape = _shape_tuple(shape)
    dims, ndim = _int64_array(shape)
    raw = _ffi._lib.poly_register_buffer_by_id(
        ctx, int(role), _dtype_id(dtype), dims, ndim, name.encode("utf-8")
    )
    if not raw:
        raise RuntimeError(f"failed to register named buffer {name!r}")
    uop = raw
    if ndim > 1:
        uop = _ffi._lib.poly_reshape(ctx, raw, dims, ndim)
        if not uop:
            raise RuntimeError(f"failed to shape named buffer {name!r}")
    return Tensor(
        _ctx=ctx, _uop=uop, _dtype=dtype,
        _device=Device.canonicalize(device), requires_grad=False
    )


def Input(name, shape=None, dtype="float32", device=None, ctx=None):
    """Create a named input placeholder tensor for nn.Model export."""
    if shape is None and not isinstance(name, str):
        shape, name = name, "input"
    from .. import _default_ctx
    return _register_named_tensor(ctx or _default_ctx, name, shape, dtype, ROLE_INPUT, device)


def Target(name, shape=None, dtype="float32", device=None, ctx=None):
    """Create a named target placeholder tensor for train/loss entrypoints."""
    if shape is None and not isinstance(name, str):
        shape, name = name, "target"
    from .. import _default_ctx
    return _register_named_tensor(ctx or _default_ctx, name, shape, dtype, ROLE_TARGET, device)


class Model:
    """Named endpoint package for already-built lazy Tensor graphs.

    `Model` is an authoring object. `export()` returns the portable runtime
    `Instance`; keeping those separate lets one Python object produce multiple
    ABI packages without becoming the runtime artifact itself.
    """

    def __init__(
        self,
        inputs,
        outputs=None,
        *,
        targets=None,
        losses=None,
        params=None,
        name=None,
    ):
        self.name = name
        self.inputs = _normalize_named_tensors(inputs, "input")
        self.targets = _normalize_named_tensors(targets, "target")
        self.outputs = _normalize_named_tensors(outputs, "output")
        self.losses = _normalize_named_tensors(losses, "loss")
        self.params = params
        tensors = list(self.inputs.values()) + list(self.targets.values())
        tensors += list(self.outputs.values()) + list(self.losses.values())
        if not tensors:
            raise ValueError("Model requires at least one input/output/loss tensor")
        ctxs = {_ptr_value(t._ctx) for t in tensors if isinstance(t, Tensor)}
        if len(ctxs) != 1:
            raise ValueError("all Model tensors must belong to the same PolyCtx")
        self._ctx = tensors[0]._ctx

    def __call__(self, *args, **kwargs):
        if args or kwargs:
            raise TypeError("nn.Model is an endpoint package; call the original layer object or Instance")
        if len(self.outputs) == 1:
            return next(iter(self.outputs.values()))
        return dict(self.outputs)

    @classmethod
    def trace(
        cls,
        obj,
        *,
        inputs,
        targets=None,
        outputs=None,
        loss=None,
        name=None,
    ):
        """Trace a callable object once with named placeholders, then package it."""
        inps = _normalize_named_tensors(inputs, "input")
        positional = list(inps.values())
        traced = obj(*positional)
        outs = outputs if outputs is not None else {"output": traced}
        losses = None
        if loss is not None:
            tgts = _normalize_named_tensors(targets, "target")
            losses = {"loss": loss(traced, *tgts.values())}
        return cls(
            inps, outs, targets=targets, losses=losses,
            params=get_state_dict(obj), name=name
        )

    def _param_items(self):
        if self.params is None:
            return []
        if isinstance(self.params, dict):
            return list(self.params.items())
        return [(f"param_{i}", p) for i, p in enumerate(self.params)]

    def _register_params(self):
        for name, tensor in self._param_items():
            if not isinstance(tensor, Tensor):
                continue
            if not tensor.uop.has_buffer_identity() or not tensor.uop.is_realized:
                tensor.realize()
            buf = tensor.uop.buffer
            if buf is None:
                raise RuntimeError(f"parameter {name!r} has no buffer identity")
            dims, ndim = _int64_array(tensor.shape)
            raw = _ffi._lib.poly_register_existing_buffer(
                self._ctx, ROLE_PARAM, buf.raw, dims, ndim,
                str(name).encode("utf-8"), bool(tensor.requires_grad)
            )
            if not raw:
                raise RuntimeError(f"failed to register parameter {name!r}")

    def _output_buffer(self, name, tensor):
        dims, ndim = _int64_array(tensor.shape)
        raw = _ffi._lib.poly_register_buffer_by_id(
            self._ctx, ROLE_OUTPUT, _dtype_id(tensor.dtype), dims, ndim,
            name.encode("utf-8")
        )
        if not raw:
            raise RuntimeError(f"failed to register output buffer {name!r}")
        return raw

    def _entry_sink(self, tensors):
        stores = []
        for name, tensor in tensors.items():
            if not isinstance(tensor, Tensor):
                raise TypeError(f"entrypoint {name!r} value is not a Tensor")
            out_buf = self._output_buffer(name, tensor)
            store = _ffi._lib.poly_store_val(self._ctx, out_buf, tensor.uop)
            if not store:
                raise RuntimeError(f"failed to build STORE for {name!r}")
            stores.append(store)
        if len(stores) == 1:
            sink = _ffi._lib.poly_sink1(self._ctx, stores[0])
        else:
            arr = (_ffi._ptr * len(stores))(*stores)
            sink = _ffi._lib.poly_sink_n(self._ctx, arr, len(stores))
        if not sink:
            raise RuntimeError("failed to build entrypoint sink")
        return sink

    def export(self):
        self._register_params()
        names = []
        sinks = []
        if self.outputs:
            names.append(b"forward")
            sinks.append(self._entry_sink(self.outputs))
        if self.losses:
            names.append(b"loss")
            sinks.append(self._entry_sink(self.losses))
        if not sinks:
            raise ValueError("Model.export requires outputs or losses")
        name_arr = (ctypes.c_char_p * len(names))(*names)
        sink_arr = (_ffi._ptr * len(sinks))(*sinks)
        ptr = _ffi._lib.poly_instance_from_sinks(self._ctx, name_arr, sink_arr, len(sinks))
        if not ptr:
            raise RuntimeError("poly_instance_from_sinks failed")
        return Instance(ptr)

    def fit(self, data=None, *, epochs=1, optimizer="sgd", lr=0.01,
            beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0,
            on_step=None, **io):
        """Export this authoring graph and train it through Instance.fit()."""
        inst = self.export()
        self.instance = inst
        return inst.fit(
            data, epochs=epochs, optimizer=optimizer, lr=lr,
            beta1=beta1, beta2=beta2, eps=eps, weight_decay=weight_decay,
            on_step=on_step, **io,
        )

    def save_bundle(self):
        return self.export().save_bundle()


def trace(obj, **kwargs):
    return Model.trace(obj, **kwargs)


def export(obj, **kwargs):
    return trace(obj, **kwargs).export()
