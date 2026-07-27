import gzip
import hashlib
import io

import pytest

import polygrad.helpers as helpers
from polygrad.helpers import (
    BEAM,
    NO_COLOR,
    WINO,
    Context,
    ContextVar,
    colored,
    fetch,
    getenv,
    prod,
)


def test_prod_matches_tinygrad_empty_generator_and_mixed_numeric_semantics():
    assert prod(()) == 1
    assert prod(value for value in (2, 3, 4)) == 24
    assert prod((2.5, 2)) == 5.0
    assert prod([True, False]) == 0


def test_getenv_is_cached_per_signature_and_propagates_conversion_errors(monkeypatch):
    getenv.cache_clear()
    monkeypatch.setenv("PG_HELPER_TEST_CACHE", "7")
    assert getenv("PG_HELPER_TEST_CACHE", 0) == 7
    assert getenv("PG_HELPER_TEST_CACHE", "") == "7"

    monkeypatch.setenv("PG_HELPER_TEST_CACHE", "9")
    assert getenv("PG_HELPER_TEST_CACHE", 0) == 7
    assert getenv("PG_HELPER_TEST_CACHE", "") == "7"

    monkeypatch.setenv("PG_HELPER_TEST_BAD", "bad")
    with pytest.raises(ValueError):
        getenv("PG_HELPER_TEST_BAD", 1)


def test_contextvar_comparison_duplicate_and_tolist_semantics(monkeypatch):
    monkeypatch.setenv("PG_HELPER_TEST_CONTEXTVAR", "2")
    value = ContextVar("PG_HELPER_TEST_CONTEXTVAR", 0)
    assert value.value == 2
    assert bool(value)
    assert value == 2
    assert value >= 2
    assert value > 1
    assert value < 3

    with pytest.raises(RuntimeError, match="attempt to recreate ContextVar"):
        ContextVar("PG_HELPER_TEST_CONTEXTVAR", 0)

    monkeypatch.setenv("PG_HELPER_TEST_LIST", "red,green,,blue")
    listed = ContextVar("PG_HELPER_TEST_LIST", "")

    class Colors:
        red = 1
        green = 2
        blue = 3

    assert listed.tolist() == ["red", "green", "blue"]
    assert listed.tolist(Colors) == [1, 2, 3]
    with pytest.raises(AssertionError):
        value.tolist()


def test_context_nests_restores_decorates_and_rejects_unknown_keys():
    old_beam, old_wino = BEAM.value, WINO.value
    with Context(BEAM=5, WINO=1) as entered:
        assert entered is None
        assert (BEAM.value, WINO.value) == (5, 1)
        with Context(BEAM=7):
            assert (BEAM.value, WINO.value) == (7, 1)
        assert (BEAM.value, WINO.value) == (5, 1)
    assert (BEAM.value, WINO.value) == (old_beam, old_wino)

    with pytest.raises(RuntimeError, match="probe"):
        with Context(BEAM=9):
            raise RuntimeError("probe")
    assert BEAM.value == old_beam

    @Context(BEAM=13)
    def decorated():
        return BEAM.value

    assert decorated() == 13
    assert BEAM.value == old_beam
    with pytest.raises(KeyError):
        with Context(PG_HELPER_TEST_UNKNOWN=1):
            pass


def test_colored_matches_tinygrad_bright_background_and_no_color_behavior():
    old_no_color = NO_COLOR.value
    with Context(NO_COLOR=0):
        assert colored("x", "RED") == "\x1b[91mx\x1b[0m"
        assert colored("x", "blue", background=True) == "\x1b[44mx\x1b[0m"
        assert colored("x", None) == "x"
        with pytest.raises(ValueError):
            colored("x", "not-a-color")

    with Context(NO_COLOR=1):
        assert colored("x", "not-a-color") == "x"
    assert NO_COLOR.value == old_no_color


def test_public_context_globals_use_tinygrad_keys():
    assert (BEAM.key, WINO.key, NO_COLOR.key) == ("BEAM", "WINO", "NO_COLOR")


class _FetchResponse(io.BytesIO):
    def __init__(self, payload, status=200, content_length=True):
        super().__init__(payload)
        self.status = status
        self.headers = {
            "content-length": str(len(payload)) if content_length else "0"
        }


def test_fetch_local_cache_refresh_and_gunzip(tmp_path, monkeypatch):
    local = tmp_path / "local.bin"
    local.write_bytes(b"local")
    assert fetch(str(local)) == local

    downloads = tmp_path / "downloads"
    monkeypatch.setattr(helpers, "_ensure_downloads_dir", lambda: downloads)
    payloads = [b"first", b"second", gzip.compress(b"expanded")]
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request.full_url, timeout))
        return _FetchResponse(payloads.pop(0))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    url = "https://example.invalid/data"
    path = fetch(url, name="asset")
    assert path == downloads / "asset"
    assert path.read_bytes() == b"first"
    assert fetch(url, name="asset") == path
    assert len(calls) == 1

    assert fetch(url, name="asset", allow_caching=False).read_bytes() == b"second"
    assert fetch(url + ".gz", name="asset.gz", gunzip=True).read_bytes() == b"expanded"
    assert calls == [(url, 10), (url, 10), (url + ".gz", 10)]


def test_fetch_sha256_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(helpers, "_ensure_downloads_dir", lambda: tmp_path)
    payloads = [b"verified", b"corrupt"]

    def fake_urlopen(request, timeout):
        return _FetchResponse(payloads.pop(0))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    expected = hashlib.sha256(b"verified").hexdigest()
    path = fetch("https://example.invalid/good", name="good", sha256=expected)
    assert path.read_bytes() == b"verified"

    with pytest.raises(RuntimeError, match="fetch sha mismatch"):
        fetch("https://example.invalid/bad", name="bad", sha256=expected)
    assert not (tmp_path / "bad").exists()
