"""Model-family constructors.

These constructors call C model-family builders and return generic
``Model`` runtime objects. Model families do not belong on ``Model``:
``Model`` is the runnable/exportable artifact, while this module owns named
architecture factories such as MLP, TabM, and NAM.
"""

import json
import ctypes

from . import _ffi
from .device import _device_id
from .model import Model


def _normalize_spec(spec):
    if isinstance(spec, dict):
        spec = json.dumps(spec)
    if isinstance(spec, str):
        return spec.encode("utf-8")
    if isinstance(spec, bytes):
        return spec
    raise TypeError("polygrad model spec must be a dict, JSON string, or bytes")


def MLP(spec=None, *, layers=None, activation="relu", bias=True, loss="none",
        batch_size=1, seed=42, device=None, **extra):
    """Build an MLP model family instance."""
    if spec is None:
        if layers is None:
            raise TypeError("MLP requires layers or a spec")
        spec = {
            "layers": layers,
            "activation": activation,
            "bias": bias,
            "loss": loss,
            "batch_size": batch_size,
            "seed": seed,
        }
        spec.update(extra)
    elif extra or layers is not None:
        if not isinstance(spec, dict):
            raise TypeError("MLP keyword overrides require a dict spec")
        spec = {**spec, **extra}
        if layers is not None:
            spec["layers"] = layers
    data = _normalize_spec(spec)
    ptr = _ffi.get_lib().poly_mlp_from_json(data, len(data), _device_id(device))
    return Model(ptr)


def TabM(spec=None, *, device=None, **kwargs):
    """Build a TabM model family instance."""
    if spec is None:
        spec = kwargs
    elif kwargs:
        if not isinstance(spec, dict):
            raise TypeError("TabM keyword overrides require a dict spec")
        spec = {**spec, **kwargs}
    data = _normalize_spec(spec)
    ptr = _ffi.get_lib().poly_tabm_from_json(data, len(data), _device_id(device))
    return Model(ptr)


def NAM(spec=None, *, device=None, **kwargs):
    """Build a NAM model family instance."""
    if spec is None:
        spec = kwargs
    elif kwargs:
        if not isinstance(spec, dict):
            raise TypeError("NAM keyword overrides require a dict spec")
        spec = {**spec, **kwargs}
    data = _normalize_spec(spec)
    ptr = _ffi.get_lib().poly_nam_from_json(data, len(data), _device_id(device))
    return Model(ptr)


def _compose(family, spec, runtime):
    from . import _default_ctx, Runtime

    if runtime is not None:
        if not isinstance(runtime, Runtime):
            raise TypeError('runtime must be a Polygrad Runtime')
        runtime._check_live()
    ctx = runtime._ctx if runtime is not None else _default_ctx
    if not ctx:
        raise RuntimeError('polygrad runtime has been disposed')
    if isinstance(spec, dict):
        spec = json.dumps(spec, allow_nan=False)
    data = _normalize_spec(spec)
    if len(data) > 1048576:
        raise ValueError('model configuration exceeds 1048576 JSON bytes')
    err = _ffi.PolyModelError()
    ptr = getattr(_ffi.get_lib(), f'poly_{family}_from_json')(ctx, data, len(data), ctypes.byref(err))
    if not ptr:
        raise ValueError(bytes(err.message).split(b'\0', 1)[0].decode('utf-8', 'replace'))
    return Model(ptr, _ctx=ctx)


def Sequential(spec, *, runtime=None):
    """Build an ordered component stack in C; return an ordinary Model.

    Accept a configuration dict, JSON string or bytes. Uses the default context
    unless runtime is supplied. That context must allow logical construction.
    Repeat creates fresh parameters; explicit shared-component calls reuse state.
    """
    return _compose('sequential', spec, runtime)


def Graph(spec, *, runtime=None):
    """Build ordered named connections in C; return an ordinary Model.

    Takes the same configuration forms and context ownership as Sequential.
    Inputs reference preceding nodes; explicit entrypoints can select objectives.
    """
    return _compose('graph', spec, runtime)


__all__ = ["MLP", "TabM", "NAM", "Sequential", "Graph"]
