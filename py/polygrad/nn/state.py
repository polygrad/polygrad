"""nn.state — State dict utilities for polygrad (tinygrad-compatible)."""

import functools
import io
import pathlib
import tarfile


class TensorIO(io.RawIOBase):
    def __init__(self, tensor):
        from ..dtype import dtypes, to_dtype

        if tensor.ndim != 1 or to_dtype(tensor.dtype) != dtypes.uint8:
            raise ValueError("Tensor must be 1d and of dtype uint8!")
        self._position, self._tensor = 0, tensor

    def readable(self):
        return True

    def read(self, size=-1):
        buf = super().read(size)
        if buf is None:
            raise ValueError("io.RawIOBase.read returned None")
        return buf

    def readinto(self, buffer):
        data = self._tensor[self._position:self._position + len(buffer)].data()
        buffer[:len(data)] = data
        self._position += len(data)
        return len(data)

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self._position = min(
            len(self._tensor),
            max(0, [offset, self._position + offset, len(self._tensor) + offset][whence]),
        )
        return self._position

    def __enter__(self):
        return self

    def write(self, value):
        raise io.UnsupportedOperation("TensorIO.write not supported")

    def writelines(self, lines):
        raise io.UnsupportedOperation("TensorIO.writelines not supported")


def accept_filename(func):
    @functools.wraps(func)
    def wrapper(filename):
        from ..tensor import Tensor

        return func(
            Tensor(pathlib.Path(filename)) if not isinstance(filename, Tensor) else filename
        )

    return wrapper


@accept_filename
def tar_extract(tensor):
    """Return regular tar members as lazy tensor views into the archive."""
    with tarfile.open(fileobj=TensorIO(tensor), mode="r") as tar:
        return {
            member.name: tensor[member.offset_data:member.offset_data + member.size]
            for member in tar
            if member.type == tarfile.REGTYPE
        }


def get_parameters(obj):
    """Return every Tensor in the state dict; optimizers partition parameters and buffers."""
    return list(get_state_dict(obj).values())


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
            target._tensor = new_t._tensor
            target._data = new_t._data
        elif strict:
            raise KeyError(f'Unexpected key: {key}')
