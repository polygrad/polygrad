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
    Conv1d,
    ConvTranspose1d,
    ConvTranspose2d,
    InstanceNorm,
    LSTMCell,
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
    "Conv1d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "InstanceNorm",
    "LSTMCell",
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


def _bind_runtime(runtime):
    """Bind only ownership; package-level layers keep Tinygrad's default API."""
    import functools
    from types import SimpleNamespace
    from . import modules, state

    def bind_method(function, *, tensor_input=False):
        @functools.wraps(function)
        def call(self, *args, **kwargs):
            runtime._check_live()
            if tensor_input:
                x = args[0] if args else kwargs.get('x', kwargs.get('idx'))
                if x is not None and x._ctx != runtime._ctx:
                    raise ValueError('NN input belongs to another Runtime')
            return function(self, *args, **kwargs)
        return call

    entries = {}
    for name in __all__:
        cls = getattr(modules, name, None)
        if not isinstance(cls, type):
            continue
        # Subclasses replace the private constructor factory only. Existing
        # methods, signatures and state traversal are shared with default NN.
        attrs = {'_Tensor': runtime.Tensor, '__module__': cls.__module__,
                 '__init__': bind_method(cls.__init__),
                 '__call__': bind_method(cls.__call__, tensor_input=True)}
        if hasattr(cls, 'calc_stats'):
            attrs['calc_stats'] = bind_method(cls.calc_stats, tensor_input=True)
        entries[name] = type(name, (cls,), attrs)
    entries['Conv1d'] = functools.partial(modules.Conv1d, _factory=entries['Conv2d'])
    entries['ConvTranspose1d'] = functools.partial(modules.ConvTranspose1d, _factory=entries['ConvTranspose2d'])
    entries['BatchNorm2d'] = entries['BatchNorm3d'] = entries['BatchNorm']
    entries['state'] = state._bind_runtime(runtime)
    for name in ('get_state_dict', 'get_parameters', 'load_state_dict', 'safe_load', 'safe_load_metadata'):
        entries[name] = getattr(entries['state'], name)
    entries['optim'] = optim._bind_runtime(runtime)
    entries['datasets'] = datasets._bind_runtime(runtime)
    return SimpleNamespace(**entries)
