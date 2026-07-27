"""Small shared helpers matching tinygrad's public helper semantics."""

from __future__ import annotations

import contextlib
import functools
import gzip
import hashlib
import operator
import os
import pathlib
import tempfile
from typing import Any, ClassVar, Generic, Iterable, Optional, TypeVar


T = TypeVar("T")


def prod(x: Iterable[T]):
    """Multiply an iterable, returning integer one for an empty input."""
    return functools.reduce(operator.mul, x, 1)


@functools.cache
def getenv(key: str, default: Any = 0):
    return type(default)(os.getenv(key, default))


class Context(contextlib.ContextDecorator):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        self.old_context = {
            key: ContextVar._cache[key].value for key in self.kwargs
        }
        for key, value in self.kwargs.items():
            ContextVar._cache[key].value = value

    def __exit__(self, *args):
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


DEV = ContextVar("DEV", "")
DEBUG = ContextVar("DEBUG", 0)
BEAM = ContextVar("BEAM", 0)
WINO = ContextVar("WINO", 0)
NO_COLOR = ContextVar("NO_COLOR", 0)

cache_dir = os.path.join(
    getenv(
        "XDG_CACHE_HOME",
        os.path.expanduser("~/Library/Caches" if os.sys.platform == "darwin" else "~/.cache"),
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
            "_" + hashlib.md5(
                "\n".join(
                    f"{key.strip()}:{value.strip()}"
                    for key, value in sorted(headers.items())
                ).encode("utf-8")
            ).hexdigest()
            if headers else ""
        )
        filename = (
            (name or hashlib.md5(url.encode("utf-8")).hexdigest())
            + header_hash
            + (".gunzip" if gunzip else "")
        )
        fp = _ensure_downloads_dir() / (subdir or "") / filename

    cached_hash_matches = (
        not sha256 or (
            fp.is_file() and hashlib.sha256(fp.read_bytes()).hexdigest() == sha256
        )
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
    colors = [
        "black", "red", "green", "yellow", "blue", "magenta", "cyan", "white"
    ]
    if color is None:
        return st
    code = (
        10 * background
        + 60 * (color.upper() == color)
        + 30
        + colors.index(color.lower())
    )
    return f"\u001b[{code}m{st}\u001b[0m"


__all__ = [
    "DEV", "DEBUG", "BEAM", "WINO", "NO_COLOR", "Context", "ContextVar",
    "GlobalCounters", "cache_dir", "colored", "fetch", "getenv", "prod"
]
