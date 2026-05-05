"""nn — Neural network modules for polygrad (tinygrad-compatible)."""

from . import optim
from .modules import (
    Linear, LayerNorm, GroupNorm, RMSNorm, Embedding, Dropout,
    Conv2d, BatchNorm,
)
from .optim import Optimizer, OptimizerGroup, SGD, Adam, AdamW
from .state import get_state_dict, load_state_dict, get_parameters
from .model import Input, Target, Model, trace, export
from .gpt2 import GPT2, Attention, FeedForward, TransformerBlock, GPT2_CONFIGS

__all__ = [
    'Linear', 'LayerNorm', 'GroupNorm', 'RMSNorm', 'Embedding', 'Dropout',
    'Conv2d', 'BatchNorm',
    'optim',
    'Optimizer', 'OptimizerGroup', 'SGD', 'Adam', 'AdamW',
    'get_state_dict', 'load_state_dict', 'get_parameters',
    'Input', 'Target', 'Model', 'trace', 'export',
    'GPT2', 'Attention', 'FeedForward', 'TransformerBlock', 'GPT2_CONFIGS',
]
