import pytest

import extra.bench_log as bench_log
import polygrad
from polygrad import GlobalCounters, Tensor
from polygrad.helpers import Context, DEBUG


def test_bench_log_surface_and_clear_replaces_event_rows():
    assert [(event.name, event.value) for event in bench_log.BenchEvent] == [
        ("LOAD_WEIGHTS", "load_weights"),
        ("STEP", "step"),
        ("FULL", "full"),
        ("MLPERF_INIT", "mlperf_init"),
        ("MLPERF_RUN", "mlperf_run"),
    ]
    assert [(event.name, event.value) for event in bench_log.InstantBenchEvent] == [
        ("GFLOPS", "gflops"),
    ]

    bench_log.log_event_instant(bench_log.InstantBenchEvent.GFLOPS, 123.5)
    old_step = bench_log._events[bench_log.BenchEvent.STEP]
    bench_log.clear_events()
    assert old_step is not bench_log._events[bench_log.BenchEvent.STEP]
    assert all(value == {"wall": [], "kernel": []}
               for event, value in bench_log._events.items()
               if isinstance(event, bench_log.BenchEvent))
    assert bench_log._events[bench_log.InstantBenchEvent.GFLOPS] == []


def test_wall_time_event_records_and_propagates_exceptions(monkeypatch):
    bench_log.clear_events()
    times = iter((100.0, 100.25, 200.0, 200.5))
    monkeypatch.setattr(bench_log.time, "monotonic", lambda: next(times))

    with bench_log.WallTimeEvent(bench_log.BenchEvent.STEP) as entered:
        assert entered.start == 100.0
    assert entered.time == 0.25
    assert bench_log._events[bench_log.BenchEvent.STEP]["wall"] == [0.25]

    with pytest.raises(RuntimeError, match="sentinel"):
        with bench_log.WallTimeEvent(bench_log.BenchEvent.FULL):
            raise RuntimeError("sentinel")
    assert bench_log._events[bench_log.BenchEvent.FULL]["wall"] == [0.5]


def test_kernel_time_event_default_guard_is_exact(monkeypatch):
    monkeypatch.setattr(bench_log, "DEBUG", 0)
    with pytest.raises(
        Exception,
        match="^KernelTimeEvent should only be used in DEBUG >= 2$",
    ):
        bench_log.KernelTimeEvent(bench_log.BenchEvent.STEP)


def test_kernel_time_event_tracks_real_global_counters(monkeypatch):
    bench_log.clear_events()
    monkeypatch.setenv("DEBUG", "2")
    x = Tensor.arange(8, dtype="float32").realize(do_update_stats=False)
    GlobalCounters.reset()

    with Context(DEBUG=2):
        with bench_log.KernelTimeEvent(bench_log.BenchEvent.STEP) as entered:
            (x + 1).realize()

    recorded = bench_log._events[bench_log.BenchEvent.STEP]["kernel"]
    assert entered.start == 0.0
    assert recorded == [GlobalCounters.time_sum_s]
    assert recorded[0] >= 0.0


def test_helpers_exports_the_canonical_global_counters_object():
    from polygrad import helpers

    assert DEBUG.key == "DEBUG"
    assert helpers.GlobalCounters is GlobalCounters
    assert helpers.GlobalCounters is polygrad.GlobalCounters
