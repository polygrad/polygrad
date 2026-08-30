"""nn — Neural network modules for polygrad (tinygrad-compatible)."""

from . import datasets, optim
from .modules import (
    Linear,
    LayerNorm,
    LayerNorm2d,
    GroupNorm,
    RMSNorm,
    Embedding,
    Dropout,
    Conv2d,
    BatchNorm,
)
from .state import get_state_dict, load_state_dict, get_parameters, safe_load, safe_load_metadata

# Pinned tinygrad/nn/__init__.py:60: dimensional BatchNorm spellings are exact
# aliases because the implementation derives its reduction axes from x.ndim.
BatchNorm2d = BatchNorm3d = BatchNorm

__all__ = [
    "Linear",
    "LayerNorm",
    "LayerNorm2d",
    "GroupNorm",
    "RMSNorm",
    "Embedding",
    "Dropout",
    "Conv2d",
    "BatchNorm",
    "BatchNorm2d",
    "BatchNorm3d",
    "datasets",
    "optim",
    "get_state_dict",
    "load_state_dict",
    "get_parameters",
    "safe_load",
    "safe_load_metadata",
]
