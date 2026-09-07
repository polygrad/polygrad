import gzip
import hashlib
import io
import ctypes
import decimal
import os

import pytest

import polygrad.helpers as helpers
from polygrad.helpers import (
    BEAM,
    ARCH_X86,
    JIT,
    NO_COLOR,
    OSX,
    Profiling,
    WINO,
    Context,
    ContextVar,
    Timing,
    argsort,
    colored,
    cpu_events,
    fetch,
    flatten,
    get_child,
    getenv,
    partition,
    prod,
    profile_marker,
    round_up,
    strides_for_shape,
    to_mv,
    tqdm,
    trange,
)


def test_check_oob_zero_and_unsupported_enable_restore_context():
    with Context(CHECK_OOB=0):
        assert helpers.CHECK_OOB.value == 0
    old = helpers.DEBUG.value
    with pytest.raises(NotImplementedError, match='bounds verifier'):
        with Context(DEBUG=old + 1, CHECK_OOB=1):
            pytest.fail('unsupported verification must not be silently enabled')
    assert helpers.DEBUG.value == old
    assert helpers.CHECK_OOB.value == 0


def test_prod_matches_tinygrad_empty_generator_and_mixed_numeric_semantics():
    assert prod(()) == 1
    assert prod(value for value in (2, 3, 4)) == 24
    assert prod((2.5, 2)) == 5.0
    assert prod([True, False]) == 0


def test_round_up_matches_pinned_negative_aligned_and_large_cases():
    assert [round_up(n, amt) for n, amt in (
        (-3, 4), (-4, 4), (6, 4), (8, 4), (232, 24984), (24984, 232),
    )] == [0, -4, 8, 8, 24984, 25056]


def test_flatten_and_partition_match_pinned_order_and_single_pass_semantics():
    assert flatten(((1, 2), [], (3,), range(4, 6))) == [1, 2, 3, 4, 5]

    seen = []

    def source():
        for value in range(6):
            seen.append(value)
            yield value

    assert partition(source(), lambda value: value % 2 == 0) == (
        [0, 2, 4],
        [1, 3, 5],
    )
    assert seen == [0, 1, 2, 3, 4, 5]


def test_timing_matches_pinned_elapsed_callback_and_disabled_output(monkeypatch, capsys):
    values = iter((1_000_000_000, 1_012_345_678, 2_000_000_000, 2_000_000_111))
    monkeypatch.setattr(helpers.time, "perf_counter_ns", lambda: next(values))
    callback_values = []

    with Timing("elapsed ", lambda et: callback_values.append(et) or ", done") as entered:
        pass
    disabled = Timing("hidden ", enabled=False)
    with disabled:
        pass

    assert entered is None
    assert callback_values == [12_345_678]
    assert disabled.et == 111
    assert capsys.readouterr().out == "elapsed  12.35 ms, done\n"


def test_profiling_matches_pinned_disabled_and_stats_file_semantics(tmp_path, capsys):
    with Profiling(enabled=False) as entered:
        sum(range(4))
    assert entered is None
    assert capsys.readouterr().out == ""

    stats_path = tmp_path / "profile.stats"
    with Profiling(enabled=True, frac=0, fn=stats_path):
        sum(range(8))
    assert stats_path.is_file() and stats_path.stat().st_size > 0
    assert capsys.readouterr().out == ""


def test_profile_marker_appends_pinned_point_event():
    before = len(cpu_events)
    profile_marker("probe", "blue")
    try:
        event = cpu_events[-1]
        assert len(cpu_events) == before + 1
        assert (event.device, event.name, event.key, event.arg) == (
            "TINY",
            "marker",
            None,
            {"name": "probe", "color": "blue"},
        )
        assert isinstance(event.ts, decimal.Decimal)
    finally:
        cpu_events.pop()


def test_state_path_and_stride_helpers_match_pinned_tinygrad():
    class Box:
        pass

    box = Box()
    box.layers = [{"weight": 7}, {"weight": 11}]
    assert get_child(box, "layers.1.weight") == 11
    assert argsort([30, 10, 20]) == [1, 2, 0]
    assert argsort((30, 10, 20)) == (1, 2, 0)
    assert strides_for_shape(()) == ()
    assert strides_for_shape((2, 1, 3)) == (3, 0, 1)


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


@pytest.mark.parametrize('text,expected', [
    ('cpu', ('CPU', '', '', '', '')),
    ('cpu:x86', ('CPU', 'X86', '', '', '')),
    ('MOCK:2+NV:PTX:sm_80', ('NV', 'PTX', 'sm_80', 'MOCK', '2')),
])
def test_target_metadata_matches_pinned_parser(text, expected):
    from polygrad.helpers import Target
    target = Target.parse(text)
    assert tuple(vars(target).values()) == expected
    assert Target.parse(repr(target)) == target
    assert target.replacedefault(arch='fallback').arch == (expected[2] or 'fallback')


def test_dev_targets_restore_and_keep_device_selection_truthful():
    from polygrad import Device
    from polygrad.helpers import DEV, Target
    old = DEV.value
    with Context(DEV='CPU'):
        assert DEV.value == [Target(device='CPU')]
        assert (DEV.device, DEV.renderer, DEV.interface) == ('CPU', '', '')
        assert Device.DEFAULT == 'CPU'
        with Context(DEV='INTERP'):
            assert Device.DEFAULT == 'INTERP'
        assert Device.DEFAULT == 'CPU'
        with Context(DEV='CPU:LLVM'):
            assert DEV.renderer == 'LLVM'
            with pytest.raises(ValueError, match='Unsupported device'):
                Device.DEFAULT
    assert DEV.value == old


def test_image_mode_rejects_unsupported_requests_without_partial_context():
    from polygrad.helpers import IMAGE
    old = BEAM.value
    try:
        with Context(IMAGE=0):
            assert IMAGE.value == 0
        with pytest.raises(NotImplementedError, match='IMAGE'):
            with Context(BEAM=44, IMAGE=1):
                pytest.fail('unsupported image mode entered')
        assert IMAGE.value == 0 and BEAM.value == old
        with pytest.raises(NotImplementedError, match='IMAGE'):
            IMAGE.value = 2
        assert IMAGE.value == 0
    finally:
        BEAM.value = old


def test_failed_context_entry_restores_prior_settings():
    class Rejecting(ContextVar):
        @property
        def value(self):
            return 0

        @value.setter
        def value(self, v):
            if v != 0:
                raise ValueError('unsupported setting')

    Rejecting('PG_HELPER_REJECT', 0)
    old = BEAM.value
    try:
        with pytest.raises(ValueError, match='unsupported setting'):
            with Context(BEAM=44, PG_HELPER_REJECT=1):
                pass
        assert BEAM.value == old
    finally:
        BEAM.value = old


def test_failed_context_entry_restores_core_logical_policy():
    from polygrad import Tensor
    from polygrad.helpers import LOGICAL
    baseline, setting = Tensor([1.0]).logical_policy, LOGICAL.value
    replacement = 'always' if baseline != 'always' else 'never'
    with pytest.raises(NotImplementedError, match='IMAGE'):
        with Context(LOGICAL=replacement, IMAGE=1):
            pytest.fail('unsupported context entered')
    assert LOGICAL.value == setting
    assert Tensor([2.0]).logical_policy == baseline


def test_image_environment_rejects_before_execution():
    import os
    import subprocess
    import sys
    proc = subprocess.run([sys.executable, '-c', 'import polygrad'], capture_output=True,
                          text=True, env={**os.environ, 'IMAGE': '1'})
    assert proc.returncode != 0
    assert 'IMAGE' in proc.stderr and 'NotImplementedError' in proc.stderr


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
    assert (BEAM.key, JIT.key, WINO.key, NO_COLOR.key) == (
        "BEAM",
        "JIT",
        "WINO",
        "NO_COLOR",
    )
    default_jit = 2 if OSX and ARCH_X86 else 1
    assert JIT.value == int(os.getenv("JIT", default_jit))


def test_to_mv_matches_pinned_writable_zero_copy_ctypes_view():
    storage = (ctypes.c_uint8 * 4)(1, 2, 3, 4)
    view = to_mv(ctypes.addressof(storage), 4)
    assert (view.format, view.shape, view.readonly, list(view)) == (
        "B",
        (4,),
        False,
        [1, 2, 3, 4],
    )
    view[1] = 9
    assert list(storage) == [1, 9, 3, 4]


def test_from_torch_dtype_matches_pinned_inverse_mapping():
    torch = pytest.importorskip("torch")
    from polygrad.dtype import _from_torch_dtype, dtypes

    expected = {
        torch.bool: dtypes.bool,
        torch.uint8: dtypes.uint8,
        torch.int16: dtypes.int16,
        torch.int32: dtypes.int32,
        torch.int64: dtypes.int64,
        torch.float16: dtypes.float16,
        torch.bfloat16: dtypes.bfloat16,
        torch.float32: dtypes.float32,
        torch.float64: dtypes.float64,
    }
    assert {torch_dtype: _from_torch_dtype(torch_dtype) for torch_dtype in expected} == expected


def test_trange_matches_pinned_iteration_and_counter_semantics():
    bar = trange(3, desc="start", disable=True)
    assert list(bar) == [0, 1, 2]
    bar.set_description("done")
    assert (bar.n, bar.i, bar.t, bar.desc) == (3, 4, 3, "done: ")

    empty = trange(0, disable=True)
    assert list(empty) == []
    assert (empty.n, empty.i) == (0, 1)

    manual = tqdm(total=2, unit="row", disable=True)
    manual.update()
    manual.update()
    manual.update(close=True)
    assert (manual.n, manual.i, manual.t, manual.unit) == (0, 3, 2, "row")


class _FetchResponse(io.BytesIO):
    def __init__(self, payload, status=200, content_length=True):
        super().__init__(payload)
        self.status = status
        self.headers = {"content-length": str(len(payload)) if content_length else "0"}


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
