"""Registered model types and their construction/import capabilities.

These constructors call C model-family builders and return generic
``Model`` runtime objects. This module owns named architecture factories; Model.from_hf/from_gguf are
format-loading conveniences delegating to the C loaders.
"""

import json
import ctypes

from . import _ffi
from .device import _device_id
from .model import Model
from .model import _import_context


def _normalize_spec(spec):
    if isinstance(spec, dict):
        spec = json.dumps(spec, allow_nan=False)
    if isinstance(spec, str):
        return spec.encode("utf-8")
    if isinstance(spec, bytes):
        return spec
    raise TypeError("polygrad model spec must be a dict, JSON string, or bytes")


def MLP(spec=None, *, layers=None, activation="relu", bias=True, loss="none",
        batch_size=1, seed=42, device=None, runtime=None, **extra):
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
    return _build('mlp', spec, runtime, device)



def _build(family, spec, runtime=None, device=None):
    ctx = _import_context(runtime)
    data = _normalize_spec(spec)
    err = _ffi.PolyModelError()
    ptr = _ffi.get_lib().poly_model_from_config(
        ctx, family.encode() if family else None, data, len(data),
        _device_id(device) if device is not None else 0, ctypes.byref(err))
    if not ptr:
        raise ValueError(bytes(err.message).decode('utf-8', 'replace'))
    return Model._from_handle(ptr, ctx)


def Sequential(spec, *, runtime=None):
    """Build an ordered component stack in C; return an ordinary Model.

    Accept a configuration dict, JSON string or bytes. Uses the default context
    unless runtime is supplied. Construction scopes logical retention and
    restores the runtime's policy without changing existing Tensors.
    Repeat creates fresh parameters; explicit shared-component calls reuse state.
    """
    return _build('sequential', spec, runtime)


def Graph(spec, *, runtime=None):
    """Build ordered named connections in C; return an ordinary Model.

    Takes the same configuration forms and context ownership as Sequential.
    Inputs reference preceding nodes; explicit entrypoints can select objectives.
    """
    return _build('graph', spec, runtime)


def Llama(spec, *, runtime=None):
    """Construct a dense Llama from HF-style config in the owning C runtime.

    Fixed int32 ``tokens[batch_size,max_seq_len]`` -> float32 ``logits``.
    Supports unscaled/Llama-3 RoPE and tied embeddings. No KV cache or sampler.
    Populate parameters explicitly, or use Model.from_hf for pretrained weights.
    """
    return _build('llama', spec, runtime)


def _type_factory(name):
    def factory(spec=None, *, runtime=None, device=None, **kwargs):
        if spec is None:
            spec = kwargs
        elif kwargs:
            if not isinstance(spec, dict):
                raise TypeError(f'{name} keyword overrides require a dict spec')
            spec = {**spec, **kwargs}
        return _build(name, spec, runtime, device)
    factory.__name__ = name
    factory.__doc__ = f"Construct {name} through the shared C model-type registry."
    return factory


def list(*, runtime=None):
    """Return supported model types and JSON/HF/GGUF capabilities from C."""
    lib = _ffi.get_lib()
    result, index = [], 0
    while (name := lib.poly_model_type_name(index)) is not None:
        flags = lib.poly_model_type_capabilities(index)
        result.append(dict(name=name.decode('ascii'), constructible=bool(flags & 1),
                           hf=bool(flags & 2), gguf=bool(flags & 4)))
        index += 1
    return result


__all__ = ['list']
for entry in list():
    if entry['constructible']:
        name = entry['name']
        __all__.append(name)
        if name not in globals():
            globals()[name] = _type_factory(name)
del name, entry
