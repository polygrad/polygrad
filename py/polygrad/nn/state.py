"""nn.state — State dict utilities for polygrad (tinygrad-compatible)."""


def get_parameters(obj):
    """Recursively collect all Tensor parameters from an object."""
    from ..tensor import Tensor
    params = []
    seen = set()

    def _collect(o, prefix=''):
        oid = id(o)
        if oid in seen:
            return
        seen.add(oid)

        if isinstance(o, Tensor):
            if o.requires_grad:
                params.append(o)
        elif isinstance(o, (list, tuple)):
            for item in o:
                _collect(item)
        elif isinstance(o, dict):
            for v in o.values():
                _collect(v)
        elif hasattr(o, '__dict__'):
            for v in o.__dict__.values():
                _collect(v)

    _collect(obj)
    return params


def get_state_dict(obj, prefix=''):
    """Get a flat dict of name → Tensor for all parameters."""
    from ..tensor import Tensor
    state = {}
    seen = set()

    def _collect(o, pfx):
        oid = id(o)
        if oid in seen:
            return
        seen.add(oid)

        if isinstance(o, Tensor):
            state[pfx] = o
        elif isinstance(o, (list, tuple)):
            for i, item in enumerate(o):
                _collect(item, f'{pfx}.{i}' if pfx else str(i))
        elif isinstance(o, dict):
            for k, v in o.items():
                _collect(v, f'{pfx}.{k}' if pfx else k)
        elif hasattr(o, '__dict__'):
            for k, v in o.__dict__.items():
                if k.startswith('_'):
                    continue
                _collect(v, f'{pfx}.{k}' if pfx else k)

    _collect(obj, prefix)
    return state


def load_state_dict(obj, state_dict, strict=True):
    """Load parameters from a state dict into an object."""
    from ..tensor import Tensor
    current = get_state_dict(obj)
    for key, val in state_dict.items():
        if key in current:
            target = current[key]
            if isinstance(val, Tensor):
                data = val.numpy()
            else:
                import numpy as np
                data = np.asarray(val, dtype=np.float32)
            new_t = Tensor(data, requires_grad=target.requires_grad)
            target._uop = new_t._uop
            target._buffer = new_t._buffer
            target._data = new_t._data
            target._shape = new_t._shape
            target._inputs = []
        elif strict:
            raise KeyError(f'Unexpected key: {key}')
