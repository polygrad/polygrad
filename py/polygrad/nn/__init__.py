"""nn — Neural network modules for polygrad (tinygrad-compatible)."""

from . import datasets, optim
from .modules import (
    Linear, LayerNorm, GroupNorm, RMSNorm, Embedding, Dropout,
    Conv2d, BatchNorm,
)
from .optim import Optimizer, OptimizerGroup, SGD, Adam, AdamW
from .state import get_state_dict, load_state_dict, get_parameters

__all__ = [
    'Linear', 'LayerNorm', 'GroupNorm', 'RMSNorm', 'Embedding', 'Dropout',
    'Conv2d', 'BatchNorm',
    'datasets', 'optim',
    'Optimizer', 'OptimizerGroup', 'SGD', 'Adam', 'AdamW',
    'get_state_dict', 'load_state_dict', 'get_parameters',
]
