"""Small shared helpers matching tinygrad's public helper semantics."""

from __future__ import annotations

import contextlib
import ctypes
import decimal
import functools
import getpass
import gzip
import hashlib
import itertools
import operator
import os
import pathlib
import platform
import math
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, ClassVar, Generic, Iterable, Iterator, Optional, TypeVar


T = TypeVar("T")


# Pinned tinygrad/helpers.py:15-17 public platform facts.
OSX = platform.system() == "Darwin"
ARCH_X86 = any(value in platform.processor() for value in ("Intel", "i386", "x86_64"))


def prod(x: Iterable[T]):
    """Multiply an iterable, returning integer one for an empty input."""
    return functools.reduce(operator.mul, x, 1)


# Pinned tinygrad/helpers.py:66. Python integer division deliberately handles
# negative and arbitrary-precision inputs with the same alignment semantics.
def round_up(num: int, amt: int) -> int:
    return (num + amt - 1) // amt * amt


# Pinned tinygrad/helpers.py:49,88-91. These are literal public helper ports.
def flatten(values: Iterable[Iterable[T]]):
    return [item for sublist in values for item in sublist]


def partition(itr: Iterable[T], fxn: Callable[[T], bool]) -> tuple[list[T], list[T]]:
    ret: tuple[list[T], list[T]] = ([], [])
    for value in itr:
        (ret[0] if fxn(value) else ret[1]).append(value)
    return ret


# Literal pinned tinygrad/helpers.py:301-305 public timer.
class Timing(contextlib.ContextDecorator):
    def __init__(self, prefix="", on_exit=None, enabled=True):
        self.prefix, self.on_exit, self.enabled = prefix, on_exit, enabled

    def __enter__(self):
        self.st = time.perf_counter_ns()

    def __exit__(self, *exc):
        self.et = time.perf_counter_ns() - self.st
        if self.enabled:
            suffix = self.on_exit(self.et) if self.on_exit else ""
            print(f"{self.prefix}{self.et * 1e-6:6.2f} ms" + suffix)


# Literal pinned tinygrad/helpers.py:308-327 Python profiler.
def _format_fcn(fcn):
    return f"{fcn[0]}:{fcn[1]}:{fcn[2]}"


class Profiling(contextlib.ContextDecorator):
    def __init__(self, enabled=True, sort="cumtime", frac=0.2, fn=None, ts=1):
        self.enabled, self.sort, self.frac, self.fn, self.time_scale = (
            enabled,
            sort,
            frac,
            fn,
            1e3 / ts,
        )

    def __enter__(self):
        import cProfile

        self.pr = cProfile.Profile()
        if self.enabled:
            self.pr.enable()

    def __exit__(self, *exc):
        if self.enabled:
            self.pr.disable()
            if self.fn:
                self.pr.dump_stats(self.fn)
            import pstats

            stats = pstats.Stats(self.pr).strip_dirs().sort_stats(self.sort)
            for fcn in stats.fcn_list[0 : int(len(stats.fcn_list) * self.frac)]:
                (_primitive_calls, num_calls, tottime, cumtime, callers) = stats.stats[fcn]
                scallers = sorted(callers.items(), key=lambda value: -value[1][2])
                print(
                    f"n:{num_calls:8d}  tm:{tottime * self.time_scale:7.2f}ms  "
                    f"tot:{cumtime * self.time_scale:7.2f}ms",
                    colored(_format_fcn(fcn).ljust(50), "yellow"),
                    colored(
                        f"<- {(scallers[0][1][2] / tottime) * 100:3.0f}% "
                        f"{_format_fcn(scallers[0][0])}",
                        "BLACK",
                    )
                    if scallers
                    else "",
                )


# Literal pinned tinygrad/helpers.py:329,355-374 point-event contract.
def perf_counter_us() -> decimal.Decimal:
    return decimal.Decimal(time.perf_counter_ns()) / 1000


class ProfileEvent:
    pass


@dataclass(frozen=True)
class ProfilePointEvent(ProfileEvent):
    device: str
    name: str
    key: Any
    arg: Any = field(default_factory=dict)
    ts: decimal.Decimal = field(default_factory=perf_counter_us)


cpu_events: list[ProfileEvent] = []


def profile_marker(name: str, color="gray") -> None:
    cpu_events.append(
        ProfilePointEvent("TINY", "marker", None, {"name": name, "color": color})
    )


# Pinned tinygrad/helpers.py:30,98-103,116-123. These helpers are public API
# and are also used by the exact torch archive loader in nn.state.
def argsort(x):
    return type(x)(sorted(range(len(x)), key=x.__getitem__))


def get_child(obj, key):
    for part in key.split("."):
        if part.isnumeric():
            obj = obj[int(part)]
        elif isinstance(obj, dict):
            obj = obj[part]
        else:
            obj = getattr(obj, part)
    return obj


def canonicalize_strides(shape, strides):
    return tuple(0 if size == 1 else stride for size, stride in zip(shape, strides))


@functools.cache
def strides_for_shape(shape):
    if not shape:
        return ()
    strides = tuple(itertools.accumulate(reversed(shape[1:]), operator.mul, initial=1))[
        ::-1
    ]
    return canonicalize_strides(shape, strides)


@functools.cache
def getenv(key: str, default: Any = 0):
    return type(default)(os.getenv(key, default))


def temp(x: str, append_user: bool = False) -> str:
    return (pathlib.Path(tempfile.gettempdir()) /
            (f"{x}.{getpass.getuser()}" if append_user else x)).as_posix()


class Context(contextlib.ContextDecorator):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        self.old_context = {key: ContextVar._cache[key].value for key in self.kwargs}
        self.old_logical_policy = None
        if "LOGICAL" in self.kwargs:
            from . import _default_ctx, _ffi
            self.old_logical_policy = int(_ffi.get_lib().poly_ctx_get_logical_policy(_default_ctx))
            policy = _normalize_logical_policy(self.kwargs["LOGICAL"])
            if policy is not None and _ffi.get_lib().poly_ctx_set_logical_policy(_default_ctx, policy) != 0:
                raise ValueError(f"invalid logical policy {self.kwargs['LOGICAL']!r}")
        try:
            for key, value in self.kwargs.items():
                ContextVar._cache[key].value = value
        except Exception:
            # Validating setters can reject a mode before __enter__ returns;
            # contextlib will not call __exit__ for that failed entry.
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *args):
        if self.old_logical_policy is not None:
            from . import _default_ctx, _ffi
            if _ffi.get_lib().poly_ctx_set_logical_policy(_default_ctx, self.old_logical_policy) != 0:
                raise RuntimeError("failed to restore logical policy")
        for key, value in self.old_context.items():
            ContextVar._cache[key].value = value


class ContextVar(Generic[T]):
    _cache: ClassVar[dict[str, "ContextVar"]] = {}

    def __init__(self, key: str, default_value: T):
        if key in ContextVar._cache:
            raise RuntimeError(f"attempt to recreate ContextVar {key}")
        ContextVar._cache[key] = self
        self.value, self.key = getenv(key, default_value), key

    def __bool__(self):
        return bool(self.value)

    def __eq__(self, value):
        return self.value == value

    def __ge__(self, value):
        return self.value >= value

    def __gt__(self, value):
        return self.value > value

    def __lt__(self, value):
        return self.value < value

    def tolist(self, obj=None):
        assert isinstance(self.value, str)
        return [
            getattr(obj, value) if obj else value
            for value in self.value.split(",")
            if value
        ]


@dataclass(frozen=True)
class Target:
    """Pinned target syntax describes a request, not backend availability."""
    device: str = ''
    renderer: str = ''
    arch: str = ''
    interface: str = ''
    indices: str = ''

    @staticmethod
    def parse(s: str) -> Target:
        split = s.split('+')
        if len(split) == 2:
            interface = split[0].rsplit(':', 1)
            iface, indices = (interface[0], interface[1]) if len(interface) == 2 else (interface[0], '')
            s = split[1]
        elif len(split) > 2:
            raise RuntimeError(f'too many \'+\' in target string: {s!r}')
        else:
            iface, indices = '', ''
        parts = [value.upper() if i < 2 else value for i, value in enumerate(s.split(':'))]
        if len(parts) > 3:
            raise RuntimeError(f'too many \':\' in target string: {s!r}')
        return Target(*(parts + [''] * (3 - len(parts))), iface, indices)

    def __repr__(self):
        first = re.sub(':*$', '', ':'.join([self.interface, self.indices]))
        second = re.sub(':*$', '', ':'.join([self.device, self.renderer, self.arch]))
        return (first + '+' if first else '') + second

    def replacedefault(self, **kwargs):
        return replace(self, **{k:v for k,v in kwargs.items() if not getattr(self, k)})


class _DEV(ContextVar):
    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = (value if isinstance(value, list) else [value] if isinstance(value, Target)
                       else [Target.parse(t) for t in value.split(';')])

    def __repr__(self):
        return ';'.join(repr(t) for t in self._value)

    def __getattr__(self, key):
        return getattr(self._value[0], key)

    def target(self, dev, **kwargs):
        assert getenv(f'{dev}_CC', '') == '', f'{dev}_CC is deprecated, use DEV targets'
        target = next((t for t in self._value if not t.device or t.device == dev), Target(device=dev))
        return replace(target.replacedefault(**kwargs), device=dev)


class _IMAGE(ContextVar):
    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        # The public frontends have no image-capable execution target. C image
        # rewrite tests are separate; a Python context must not silently enable it.
        if value != 0:
            raise NotImplementedError('IMAGE execution is not supported by the Python frontend')
        self._value = value


class _CHECK_OOB(ContextVar):
    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        # PG-PARITY-026: uop/spec.py's optional bounds verifier has no C port.
        # Zero is truthful; a configuration flag must not pretend to verify.
        if value != 0:
            raise NotImplementedError('CHECK_OOB bounds verifier is not implemented')
        self._value = value


class _NOOPT(ContextVar):
    @property
    def value(self):
        from . import _ffi
        return _ffi.get_lib().poly_get_noopt()

    @value.setter
    def value(self, value):
        from . import _ffi
        if not isinstance(value, int) or not -(2**31) <= value <= 2**31 - 1:
            raise ValueError('NOOPT must be an int32')
        _ffi.get_lib().poly_set_noopt(value)


NOOPT = _NOOPT("NOOPT", 0)
DEV = _DEV("DEV", "")
CHECK_OOB = _CHECK_OOB("CHECK_OOB", 0)
IMAGE = _IMAGE("IMAGE", 0)
DEBUG = ContextVar("DEBUG", 0)
BEAM = ContextVar("BEAM", 0)
# Pinned tinygrad/helpers.py:241. Keep the public configuration object rather
# than exposing a frozen boolean so Context(JIT=...) and environment overrides work.
JIT = ContextVar("JIT", 2 if OSX and ARCH_X86 else 1)
WINO = ContextVar("WINO", 0)
NO_COLOR = ContextVar("NO_COLOR", 0)
# Current tinygrad/helpers.py:237. Dropout, BatchNorm, and optimizers share one
# scoped training-mode owner rather than storing mode on Tensor.
TRAINING = ContextVar("TRAINING", 0)
LOGICAL = ContextVar("LOGICAL", "")


def _normalize_logical_policy(value):
    if value is None:
        return None
    if value is False:
        return 0
    if value is True:
        return 1
    if isinstance(value, int) and not isinstance(value, bool) and value in (0, 1, 2):
        return value
    if isinstance(value, str):
        policies = {"never": 0, "always": 1, "until_realize": 2}
        if value in policies:
            return policies[value]
    raise ValueError(
        "logical policy must be never, always, until_realize, False/0, True/1, 2, or None"
    )


def _logical_policy_name(value):
    try:
        return ("never", "always", "until_realize")[int(value)]
    except (IndexError, TypeError, ValueError):
        raise RuntimeError(f"unknown core logical policy {value!r}") from None


def _logical_state_name(value):
    try:
        return ("available", "never_constructed", "retired", "unsupported_resource")[int(value)]
    except (IndexError, TypeError, ValueError):
        raise RuntimeError(f"unknown core logical state {value!r}") from None

cache_dir = os.path.join(
    getenv(
        "XDG_CACHE_HOME",
        os.path.expanduser(
            "~/Library/Caches" if os.sys.platform == "darwin" else "~/.cache"
        ),
    ),
    "polygrad",
)


def _ensure_downloads_dir():
    return pathlib.Path(cache_dir) / "downloads"


def fetch(
    url: str,
    name: pathlib.Path | str | None = None,
    subdir: str | None = None,
    gunzip: bool = False,
    allow_caching=not getenv("DISABLE_HTTP_CACHE"),
    headers: dict[str, str] = {},
    sha256: str | None = None,
):
    """Fetch a URL into Polygrad's cache and return its local path."""
    import urllib.request

    if url.startswith(("/", ".")):
        return pathlib.Path(url)
    if name is not None and (isinstance(name, pathlib.Path) or "/" in name):
        fp = pathlib.Path(name)
    else:
        header_hash = (
            "_"
            + hashlib.md5(
                "\n".join(
                    f"{key.strip()}:{value.strip()}"
                    for key, value in sorted(headers.items())
                ).encode("utf-8")
            ).hexdigest()
            if headers
            else ""
        )
        filename = (
            (name or hashlib.md5(url.encode("utf-8")).hexdigest())
            + header_hash
            + (".gunzip" if gunzip else "")
        )
        fp = _ensure_downloads_dir() / (subdir or "") / filename

    cached_hash_matches = not sha256 or (
        fp.is_file() and hashlib.sha256(fp.read_bytes()).hexdigest() == sha256
    )
    if not fp.is_file() or not allow_caching or not cached_hash_matches:
        fp.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(
            url, headers={"User-Agent": "polygrad", **headers}
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            assert response.status in {200, 206}, response.status
            expected_length = (
                int(response.headers.get("content-length", 0)) if not gunzip else None
            )
            readfile = gzip.GzipFile(fileobj=response) if gunzip else response
            digest = hashlib.sha256() if sha256 else None
            written = 0
            with tempfile.NamedTemporaryFile(dir=fp.parent, delete=False) as tmp:
                while chunk := readfile.read(16384):
                    if digest:
                        digest.update(chunk)
                    written += tmp.write(chunk)
                tmp_path = pathlib.Path(tmp.name)
            if digest and digest.hexdigest() != sha256:
                tmp_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"fetch sha mismatch, expected {sha256} but got {digest.hexdigest()}"
                )
            tmp_path.replace(fp)
            if expected_length and written < expected_length:
                raise RuntimeError(
                    f"fetch size incomplete, {written} < {expected_length}"
                )
    return fp


def colored(st, color: Optional[str], background=False):
    if NO_COLOR:
        return st
    colors = ["black", "red", "green", "yellow", "blue", "magenta", "cyan", "white"]
    if color is None:
        return st
    code = (
        10 * background
        + 60 * (color.upper() == color)
        + 30
        + colors.index(color.lower())
    )
    return f"\u001b[{code}m{st}\u001b[0m"


# Literal pinned tinygrad/helpers.py:527-530 ctypes memory-view boundary.
def to_mv(ptr: int, sz: int) -> memoryview:
    return memoryview((ctypes.c_uint8 * sz).from_address(ptr)).cast("B")


def mv_address(mv):
    return ctypes.addressof(ctypes.c_char.from_buffer(mv))


# Direct port of pinned tinygrad/helpers.py:538-575. The examples use the
# returned object's counters and set_description in addition to iteration.
class tqdm(Generic[T]):
    def __init__(
        self,
        iterable: Iterable[T] | None = None,
        desc: str = "",
        disable: bool | None = False,
        unit: str = "it",
        unit_scale=False,
        total: int | None = None,
        rate: int = 100,
    ):
        self.disable = not sys.stderr.isatty() if disable is None else disable
        self.iterable = iterable
        self.unit = unit
        self.unit_scale = unit_scale
        self.rate = rate
        self.st = time.perf_counter()
        self.i, self.n, self.skip = -1, 0, 1
        self.t = getattr(iterable, "__len__", lambda: 0)() if total is None else total
        self.set_description(desc)
        self.update(0)

    def __iter__(self) -> Iterator[T]:
        assert self.iterable is not None, "need an iterable to iterate"
        for item in self.iterable:
            yield item
            self.update(1)
        self.update(close=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.update(close=True)

    def set_description(self, desc: str):
        self.desc = f"{desc}: " if desc else ""

    def update(self, n: int = 0, close: bool = False):
        self.n, self.i = self.n + n, self.i + 1
        if self.disable or (not close and self.i % self.skip != 0):
            return
        prog = self.n / self.t if self.t else 0
        elapsed = time.perf_counter() - self.st
        ncols = shutil.get_terminal_size().columns
        if elapsed and self.i / elapsed > self.rate and self.i:
            self.skip = max(int(self.i / elapsed) // self.rate, 1)

        def hms(value):
            return ":".join(
                f"{part:02d}" if index else str(part)
                for index, part in enumerate(
                    [
                        int(value) // 3600,
                        int(value) % 3600 // 60,
                        int(value) % 60,
                    ]
                )
                if index or part
            )

        def si(value):
            if not value:
                return "0.00"
            exponent = round(math.log(value, 1000), 6)
            digits = int(3 - 3 * math.fmod(exponent, 1))
            rendered = f"{value / 1000 ** int(exponent):.{digits}f}"[:4].rstrip(".")
            if rendered == "1000":
                return (
                    f"{value / 1000 ** (int(exponent) + 1):.3f}"[:4].rstrip(".")
                    + " kMGTPEZY"[int(exponent) + 1]
                )
            return rendered + " kMGTPEZY"[int(exponent)].strip()

        progress = (
            f"{si(self.n)}{f'/{si(self.t)}' if self.t else self.unit}"
            if self.unit_scale
            else f"{self.n}{f'/{self.t}' if self.t else self.unit}"
        )
        estimate = (
            f"<{hms(elapsed / prog - elapsed) if self.n else '?'}" if self.t else ""
        )
        iterations = (
            (si(self.n / elapsed) if self.unit_scale else f"{self.n / elapsed:5.2f}")
            if self.n
            else "?"
        )
        suffix = f"{progress} [{hms(elapsed)}{estimate}, {iterations}{self.unit}/s]"
        size = max(ncols - len(self.desc) - 3 - 2 - 2 - len(suffix), 1)
        fraction = size * prog
        fill = ("█" * int(fraction) + " ▏▎▍▌▋▊▉"[int(8 * fraction) % 8].strip()).ljust(
            size, " "
        )
        bar = (
            "\r"
            + self.desc
            + (f"{100 * prog:3.0f}%|{fill}| " if self.t else "")
            + suffix
        )
        print(bar[: ncols + 1], flush=True, end="\n" * close, file=sys.stderr)

    @classmethod
    def write(cls, value: str):
        print(f"\r\033[K{value}", flush=True, file=sys.stderr)


def trange(n: int, **kwargs) -> tqdm[int]:
    return tqdm(range(n), total=n, **kwargs)


__all__ = [
    "OSX",
    "ARCH_X86",
    "DEV",
    "Target",
    "IMAGE",
    "DEBUG",
    "BEAM",
    "JIT",
    "WINO",
    "NO_COLOR",
    "LOGICAL",
    "Context",
    "ContextVar",
    "GlobalCounters",
    "Profiling",
    "ProfileEvent",
    "ProfilePointEvent",
    "cache_dir",
    "colored",
    "fetch",
    "getenv",
    "temp",
    "mv_address",
    "prod",
    "profile_marker",
    "cpu_events",
    "to_mv",
    "tqdm",
    "trange",
]
