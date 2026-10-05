"""Ownership on failed capture, independent of graph/buffer collection."""
import pytest
import numpy as np
from polygrad import Jit, Runtime, _ffi


@pytest.mark.parametrize('failure', ['exception', 'empty'])
def test_failed_captures_release_native_handle_and_retry(monkeypatch, failure):
    new, free = _ffi._lib.poly_jit_new, _ffi._lib.poly_jit_free
    active = set()

    def counted_new(ctx):
        handle = new(ctx)
        if handle:
            active.add(handle)
        return handle

    def counted_free(handle):
        active.remove(handle)
        return free(handle)

    monkeypatch.setattr(_ffi._lib, 'poly_jit_new', counted_new)
    monkeypatch.setattr(_ffi._lib, 'poly_jit_free', counted_free)
    with Runtime(device='cpu') as rt:
        x = rt.Tensor([1., 2.]).realize()
        failing = False

        def body(value):
            if failing:
                if failure == 'empty':
                    return value
                raise ValueError('capture probe')
            return value + 3

        f = Jit(body)
        try:
            np.testing.assert_equal(f(x).numpy(), [4., 5.])
            failing = True
            for _ in range(20):
                with pytest.raises(Exception, match='capture probe|didn.t JIT anything'):
                    f(x)
                assert f.cnt == 1
                assert not active, 'cancelled capture still owns a native JIT'
            failing = False
            np.testing.assert_equal(f(x).numpy(), [4., 5.])
            np.testing.assert_equal(f(x).numpy(), [4., 5.])
            assert f.captured and f.replay_count == 1
            assert len(active) == 1
        finally:
            f.reset()
            # Keep the failing regression run leak-free too.
            for handle in list(active):
                counted_free(handle)
        assert not active
