"""nn — Neural network modules for polygrad (tinygrad-compatible)."""

from .modules import (
    Linear, LayerNorm, GroupNorm, RMSNorm, Embedding, Dropout,
    Conv2d, BatchNorm,
)
from .optim import Optimizer, SGD, Adam, AdamW
from .state import get_state_dict, load_state_dict, get_parameters
from .gpt2 import GPT2, Attention, FeedForward, TransformerBlock, GPT2_CONFIGS

__all__ = [
    'Linear', 'LayerNorm', 'GroupNorm', 'RMSNorm', 'Embedding', 'Dropout',
    'Conv2d', 'BatchNorm',
    'Optimizer', 'SGD', 'Adam', 'AdamW',
    'get_state_dict', 'load_state_dict', 'get_parameters',
    'GPT2', 'Attention', 'FeedForward', 'TransformerBlock', 'GPT2_CONFIGS',
]
