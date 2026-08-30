"""dtypes -- port of tinygrad's dtype.py for polygrad."""

from __future__ import annotations
from typing import Final, ClassVar, Callable, Literal, Union
import math
import struct
import ctypes
import functools
from dataclasses import dataclass, fields
from enum import Enum, auto


from .helpers import getenv


class ConstFloat(float):
    """Float subclass that distinguishes -0.0 from 0.0 and where nan == nan."""
    __slots__ = ('bits',)
    bits: int

    def __new__(cls, v: float):
        obj = super().__new__(cls, v)
        obj.bits = struct.unpack('<Q', struct.pack('<d', v))[0]
        return obj

    def __eq__(self, other):
        if self is other:
            return True
        if isinstance(other, float) and math.isnan(self) and math.isnan(other):
            return True
        return float.__eq__(self, other)

    def __hash__(self):
        return hash(self.bits)

    def __repr__(self):
        return f"ConstFloat({float.__repr__(self)})"

    def __str__(self):
        return float.__repr__(self)


class InvalidType:
    _instance: ClassVar['InvalidType | None'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cls._instance

    def __eq__(self, other):
        return self is other

    def __lt__(self, other):
        return self is not other

    def __gt__(self, other):
        return self is not other

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return "Invalid"

    def __reduce__(self):
        return (InvalidType, ())

    def __format__(self, spec):
        return "Invalid"


Invalid = InvalidType()

PyConst = Union[float, int, bool]
ConstType = Union[PyConst, InvalidType]

FmtStr = Literal['?', 'b', 'B', 'h', 'H', 'i', 'I', 'q', 'Q', 'e', 'f', 'd']


class DTypeMetaClass(type):
    dcache: dict = {}

    def __call__(cls, *args, **kwargs):
        ret = DTypeMetaClass.dcache.get(args, None)
        if ret is not None:
            return ret
        DTypeMetaClass.dcache[args] = ret = super().__call__(*args)
        return ret


class AddrSpace(Enum):
    def __repr__(self):
        return str(self)
    GLOBAL = auto()
    LOCAL = auto()
    REG = auto()


@dataclass(frozen=True, eq=False)
class DType(metaclass=DTypeMetaClass):
    priority: int
    bitsize: int
    name: str
    fmt: FmtStr | None

    @property
    def itemsize(self) -> int:
        return (self.bitsize + 7) // 8

    @staticmethod
    def new(priority: int, bitsize: int, name: str, fmt):
        return DType(priority, bitsize, name, fmt)

    def __reduce__(self):
        return type(self), tuple(getattr(self, f.name) for f in fields(self))

    def __repr__(self):
        return f"dtypes.{INVERSE_DTYPES_DICT.get(self.name, self.name)}"

    def __lt__(self, o):
        return (self.priority, self.bitsize, self.name, self.fmt) < \
               (o.priority, o.bitsize, o.name, o.fmt)

    @functools.cached_property
    def min(self):
        if dtypes.is_int(self):
            return 0 if dtypes.is_unsigned(self) else -2 ** (self.bitsize - 1)
        return -float("inf") if dtypes.is_float(self) else False

    @functools.cached_property
    def max(self):
        if dtypes.is_int(self):
            return 2 ** self.bitsize - 1 + self.min
        return float("inf") if dtypes.is_float(self) else True

    def const(self, val):
        if isinstance(val, tuple):
            return tuple(map(self.const, val))
        if isinstance(val, InvalidType):
            return val
        if isinstance(val, float) and math.isnan(val):
            val = math.nan
        return ConstFloat(float(val)) if dtypes.is_float(self) \
            else bool(val) if dtypes.is_bool(self) \
            else int(val)


class dtypes:
    @staticmethod
    @functools.cache
    def is_float(x) -> bool:
        if isinstance(x, str):
            return x in _FLOAT_NAMES
        return x in (dtypes.floats + (dtypes.weakfloat,))

    @staticmethod
    @functools.cache
    def is_int(x) -> bool:
        if isinstance(x, str):
            return x in _INT_NAMES
        return x in (dtypes.ints + (dtypes.weakint,))

    @staticmethod
    @functools.cache
    def is_unsigned(x) -> bool:
        if isinstance(x, str):
            return x in _UINT_NAMES
        return x in dtypes.uints

    @staticmethod
    def is_bool(x) -> bool:
        if isinstance(x, str):
            return x == 'bool'
        return x == dtypes.bool

    @staticmethod
    def from_py(x):
        if isinstance(x, (bool, InvalidType)):
            return dtypes.bool
        if isinstance(x, float):
            return dtypes.weakfloat
        if isinstance(x, int):
            return dtypes.weakint
        if isinstance(x, (list, tuple)):
            return strong_dtype(max(dtypes.from_py(xi) for xi in x)) if x else dtypes.default_float
        raise RuntimeError(f"Could not infer dtype of {x} with type {type(x)}")

    @staticmethod
    def finfo(dtype):
        if not dtypes.is_float(dtype):
            raise ValueError(f"{dtype} is not a floating point type")
        return {dtypes.float16: (5, 10), dtypes.bfloat16: (8, 7),
                dtypes.float32: (8, 23), dtypes.float64: (11, 52),
                dtypes.fp8e4m3: (4, 3), dtypes.fp8e5m2: (5, 2),
                dtypes.fp8e4m3fnuz: (4, 3), dtypes.fp8e5m2fnuz: (5, 2)}[dtype]

    void: Final[DType] = DType.new(-1, 0, "void", None)
    weakint: Final[DType] = DType.new(0, 800, "weakint", None)
    bool: Final[DType] = DType.new(0, 1, "bool", '?')
    int8: Final[DType] = DType.new(1, 8, "signed char", 'b')
    uint8: Final[DType] = DType.new(2, 8, "unsigned char", 'B')
    int16: Final[DType] = DType.new(3, 16, "short", 'h')
    uint16: Final[DType] = DType.new(4, 16, "unsigned short", 'H')
    int32: Final[DType] = DType.new(5, 32, "int", 'i')
    uint32: Final[DType] = DType.new(6, 32, "unsigned int", 'I')
    int64: Final[DType] = DType.new(7, 64, "long", 'q')
    uint64: Final[DType] = DType.new(8, 64, "unsigned long", 'Q')
    _uint128: Final[DType] = DType.new(8, 128, "uint128", None)
    _uint256: Final[DType] = DType.new(8, 256, "uint256", None)
    weakfloat: Final[DType] = DType.new(9, 800, "weakfloat", None)
    fp8e4m3: Final[DType] = DType.new(10, 8, "float8_e4m3", None)
    fp8e5m2: Final[DType] = DType.new(11, 8, "float8_e5m2", None)
    fp8e4m3fnuz: Final[DType] = DType.new(10, 8, "float8_e4m3fnuz", None)
    fp8e5m2fnuz: Final[DType] = DType.new(11, 8, "float8_e5m2fnuz", None)
    float16: Final[DType] = DType.new(12, 16, "half", 'e')
    bfloat16: Final[DType] = DType.new(13, 16, "__bf16", None)
    float32: Final[DType] = DType.new(14, 32, "float", 'f')
    float64: Final[DType] = DType.new(15, 64, "double", 'd')

    half = float16
    float = float32
    double = float64
    uchar = uint8
    ushort = uint16
    uint = uint32
    ulong = uint64
    char = int8
    short = int16
    int = int32
    long = int64

    default_float: ClassVar[DType] = float32
    default_int: ClassVar[DType] = int32

    fp8_ocp = (fp8e4m3, fp8e5m2)
    fp8_fnuz = (fp8e4m3fnuz, fp8e5m2fnuz)
    fp8s = fp8_ocp + fp8_fnuz
    floats = fp8s + (float16, bfloat16, float32, float64)
    int8s = (uint8, int8)
    int16s = (uint16, int16)
    int32s = (uint32, int32)
    int64s = (uint64, int64)
    uints = (uint8, uint16, uint32, uint64)
    sints = (int8, int16, int32, int64)
    ints = uints + sints
    weaks = (weakint, weakfloat)
    all = floats + ints + (bool,)


if (env_default_float := getenv("DEFAULT_FLOAT", "")):
    dtypes.default_float = getattr(dtypes, env_default_float.lower())
    assert dtypes.is_float(dtypes.default_float), \
        f"{env_default_float} is not a float dtype"

DTypeLike = Union[str, DType]


def to_dtype(dtype):
    return dtype if isinstance(dtype, DType) else getattr(dtypes, dtype.lower())


def strong_dtype(dtype):
    return {dtypes.weakint: dtypes.default_int,
            dtypes.weakfloat: dtypes.default_float}.get(dtype, dtype)


def weak_dtype(dtype):
    return dtypes.weakfloat if dtypes.is_float(dtype) else \
           dtypes.weakint if dtypes.is_int(dtype) else dtype


# Current tinygrad dtype.py:171-188 promotion lattice.
promo_lattice = {
    dtypes.bool: [dtypes.weakint],
    dtypes.weakint: [dtypes.int8, dtypes.uint8],
    dtypes.int8: [dtypes.int16],
    dtypes.int16: [dtypes.int32],
    dtypes.int32: [dtypes.int64],
    dtypes.int64: [dtypes.weakfloat],
    dtypes.uint8: [dtypes.int16, dtypes.uint16],
    dtypes.uint16: [dtypes.int32, dtypes.uint32],
    dtypes.uint32: [dtypes.int64, dtypes.uint64],
    dtypes.uint64: [dtypes.weakfloat],
    dtypes.weakfloat: [dtypes.fp8e4m3, dtypes.fp8e5m2,
                       dtypes.fp8e4m3fnuz, dtypes.fp8e5m2fnuz],
    dtypes.fp8e4m3: [dtypes.float16, dtypes.bfloat16],
    dtypes.fp8e5m2: [dtypes.float16, dtypes.bfloat16],
    dtypes.fp8e4m3fnuz: [dtypes.float16, dtypes.bfloat16],
    dtypes.fp8e5m2fnuz: [dtypes.float16, dtypes.bfloat16],
    dtypes.float16: [dtypes.float32],
    dtypes.bfloat16: [dtypes.float32],
    dtypes.float32: [dtypes.float64],
}


@functools.cache
def _get_recursive_parents(dtype):
    if dtype == dtypes.float64:
        return {dtypes.float64}
    return set.union(*[_get_recursive_parents(d) for d in promo_lattice.get(dtype, [])], {dtype})


@functools.cache
def least_upper_dtype(*ds):
    return min(set.intersection(*[_get_recursive_parents(d) for d in ds]))


def least_upper_float(dt):
    return dtypes.weakfloat if dt is dtypes.weakint else \
           dt if dtypes.is_float(dt) else least_upper_dtype(dt, dtypes.default_float)


DTYPES_DICT = {k: v for k, v in dtypes.__dict__.items()
               if isinstance(v, DType) and not k.startswith(("default", "void", "weak", "_"))}
INVERSE_DTYPES_DICT = {**{v.name: k for k, v in DTYPES_DICT.items()},
                       "void": "void", "weakint": "weakint", "weakfloat": "weakfloat"}

# String-name sets for backward-compat is_float/is_int on str inputs.
_FLOAT_NAMES = {d.name for d in dtypes.floats + (dtypes.weakfloat,)} | \
               {'float32', 'float64', 'float16', 'bfloat16', 'weakfloat'}
_INT_NAMES = {d.name for d in dtypes.ints} | {'int8', 'int16', 'int32', 'int64',
                                               'uint8', 'uint16', 'uint32', 'uint64'}
_UINT_NAMES = {d.name for d in dtypes.uints} | {'uint8', 'uint16', 'uint32', 'uint64'}


# --- numpy interop (used by Tensor.numpy) ---

def _to_np_dtype(dtype):
    """Map a DType (or polygrad legacy dtype string) to a NumPy dtype class.
    bfloat16 and fp8s widen to float32 for NumPy interop (tinygrad does the
    same -- Tensor.numpy calls .float().numpy() for those paths)."""
    import numpy as np
    if isinstance(dtype, str):
        return _STR_TO_NP.get(dtype, np.float32)
    if dtype in {dtypes.bfloat16, *dtypes.fp8s}:
        return np.float32
    return np.dtype(dtype.fmt).type if dtype.fmt is not None else None


def _from_np_dtype(npdtype):
    import numpy as np
    return DTYPES_DICT[np.dtype(npdtype).name]


# Literal pinned tinygrad/dtype.py:367-378 Torch dtype interop. Torch remains
# an optional import and is loaded only when this boundary is called.
@functools.cache
def _to_torch_dtype(dtype):
    import numpy as np
    import torch

    if dtype == dtypes.uint64:
        return torch.uint64
    if dtype == dtypes.bfloat16:
        return torch.bfloat16
    if dtype in dtypes.fp8s:
        return torch.uint8
    try:
        return torch.from_numpy(np.array([], dtype=_to_np_dtype(dtype))).dtype
    except TypeError:
        return None


@functools.cache
def _from_torch_dtype(torchdtype):
    return {
        torch_value: dtype
        for dtype in DTYPES_DICT.values()
        if (torch_value := _to_torch_dtype(dtype)) is not None
    }[torchdtype]


_STR_TO_NP = None

def _init_str_to_np():
    global _STR_TO_NP
    import numpy as np
    _STR_TO_NP = {
        'float32': np.float32, 'float64': np.float64, 'float16': np.float16,
        'bfloat16': np.float32,  # widen per tinygrad semantics
        'int8': np.int8, 'uint8': np.uint8,
        'int16': np.int16, 'uint16': np.uint16,
        'int32': np.int32, 'uint32': np.uint32,
        'int64': np.int64, 'uint64': np.uint64,
        'bool': np.bool_,
    }


_init_str_to_np()
