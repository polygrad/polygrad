"""Model-family constructors.

These constructors call C model-family builders and return generic
``Instance`` runtime objects. Model families do not belong on ``Instance``:
``Instance`` is the runnable/exportable artifact, while this module owns named
architecture factories such as MLP, TabM, and NAM.
"""

import json

from . import _ffi
from .instance import Instance


def _normalize_spec(spec):
    if isinstance(spec, dict):
        spec = json.dumps(spec)
    if isinstance(spec, str):
        return spec.encode("utf-8")
    if isinstance(spec, bytes):
        return spec
    raise TypeError("polygrad model spec must be a dict, JSON string, or bytes")


def MLP(spec=None, *, layers=None, activation="relu", bias=True, loss="none",
        batch_size=1, seed=42, **extra):
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
    ptr = _ffi.get_lib().poly_mlp_from_json(data, len(data))
    return Instance(ptr)


def TabM(spec=None, **kwargs):
    """Build a TabM model family instance."""
    if spec is None:
        spec = kwargs
    elif kwargs:
        if not isinstance(spec, dict):
            raise TypeError("TabM keyword overrides require a dict spec")
        spec = {**spec, **kwargs}
    data = _normalize_spec(spec)
    ptr = _ffi.get_lib().poly_tabm_instance(data, len(data))
    return Instance(ptr)


def NAM(spec=None, **kwargs):
    """Build a NAM model family instance."""
    if spec is None:
        spec = kwargs
    elif kwargs:
        if not isinstance(spec, dict):
            raise TypeError("NAM keyword overrides require a dict spec")
        spec = {**spec, **kwargs}
    data = _normalize_spec(spec)
    ptr = _ffi.get_lib().poly_nam_instance(data, len(data))
    return Instance(ptr)


__all__ = ["MLP", "TabM", "NAM"]
