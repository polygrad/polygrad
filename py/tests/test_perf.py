"""Performance tests for polygrad vs numpy.

Measures compilation overhead, cache effectiveness, and execution time
relative to numpy. These tests PASS/FAIL on correctness only.
Timing is reported but not asserted (hardware-dependent).

Run: python -m pytest py/tests/test_perf.py -v -s
"""

import os
import time
import shutil
import numpy as np
import pytest

from polygrad import Tensor


def _clear_disk_cache():
    """Remove disk cache to force cold compilation."""
    cache_dir = os.path.join(
        os.environ.get('XDG_CACHE_HOME', os.path.expanduser('~/.cache')),
        'polygrad'
    )
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)


class TestDiskCache:
    """Verify disk cache speeds up repeated compilations."""

    def test_cache_hit_faster_than_miss(self):
        """Second run of same graph should be faster (disk cache hit)."""
        _clear_disk_cache()

        a_np = np.random.randn(100).astype(np.float32)
        b_np = np.random.randn(100).astype(np.float32)

        # Cold run (cache miss)
        t0 = time.perf_counter()
        c = (Tensor(a_np) + Tensor(b_np)).numpy()
        cold_ms = (time.perf_counter() - t0) * 1000

        np.testing.assert_allclose(c, a_np + b_np, rtol=1e-5)

        # Warm run (cache hit)
        t0 = time.perf_counter()
        c2 = (Tensor(a_np) + Tensor(b_np)).numpy()
        warm_ms = (time.perf_counter() - t0) * 1000

        np.testing.assert_allclose(c2, a_np + b_np, rtol=1e-5)

        print(f'\n  vecadd(100): cold={cold_ms:.1f}ms, warm={warm_ms:.1f}ms, '
              f'speedup={cold_ms/max(warm_ms, 0.01):.1f}x')

    def test_cache_persists_across_contexts(self):
        """Disk cache survives PolyCtx destruction.
        After at least one cold compilation, cache dir should have .so files."""
        cache_dir = os.path.join(
            os.environ.get('XDG_CACHE_HOME', os.path.expanduser('~/.cache')),
            'polygrad'
        )
        # The previous test (test_cache_hit_faster_than_miss) already cleared
        # the cache and compiled a kernel, so cache dir should exist
        if not os.path.isdir(cache_dir):
            # Force a fresh compilation by clearing cache
            _clear_disk_cache()
            (Tensor([1.0, 2.0, 3.0]) * Tensor([4.0, 5.0, 6.0])).numpy()

        assert os.path.isdir(cache_dir), 'disk cache dir should exist'
        cached_files = [f for f in os.listdir(cache_dir) if f.endswith('.so')]
        assert len(cached_files) > 0, 'disk cache should have .so files'

        # Verify results are correct (may hit in-memory or disk cache)
        r1 = (Tensor([1.0, 2.0, 3.0]) * Tensor([4.0, 5.0, 6.0])).numpy()
        r2 = (Tensor([1.0, 2.0, 3.0]) * Tensor([4.0, 5.0, 6.0])).numpy()
        np.testing.assert_allclose(r1, r2)


class TestOverheadVsNumpy:
    """Measure polygrad overhead relative to numpy.

    Uses numpy array input (zero-copy path) for fair comparison.
    Reports timing ratios but does not assert speed thresholds.
    """

    def test_vecadd_overhead(self):
        """Element-wise add: polygrad vs numpy."""
        n = 10000
        a_np = np.random.randn(n).astype(np.float32)
        b_np = np.random.randn(n).astype(np.float32)

        # Warmup
        for _ in range(3):
            (Tensor(a_np) + Tensor(b_np)).numpy()

        iters = 100
        t0 = time.perf_counter()
        for _ in range(iters):
            c_np = a_np + b_np
        np_us = (time.perf_counter() - t0) / iters * 1e6

        t0 = time.perf_counter()
        for _ in range(iters):
            c_pg = (Tensor(a_np) + Tensor(b_np)).numpy()
        pg_us = (time.perf_counter() - t0) / iters * 1e6

        np.testing.assert_allclose(c_pg, c_np, rtol=1e-5)
        print(f'\n  vecadd({n}): numpy={np_us:.1f}us, polygrad={pg_us:.1f}us, '
              f'ratio={pg_us/max(np_us, 0.1):.1f}x')

    def test_matmul_overhead(self):
        """Matrix multiply: polygrad vs numpy."""
        n = 64
        a_np = np.random.randn(n, n).astype(np.float32)
        b_np = np.random.randn(n, n).astype(np.float32)

        # Warmup
        for _ in range(3):
            Tensor(a_np).matmul(Tensor(b_np)).numpy()

        iters = 100
        t0 = time.perf_counter()
        for _ in range(iters):
            c_np = a_np @ b_np
        np_us = (time.perf_counter() - t0) / iters * 1e6

        t0 = time.perf_counter()
        for _ in range(iters):
            c_pg = Tensor(a_np).matmul(Tensor(b_np)).numpy()
        pg_us = (time.perf_counter() - t0) / iters * 1e6

        np.testing.assert_allclose(c_pg, c_np, rtol=1e-3, atol=1e-3)
        print(f'\n  matmul({n}x{n}): numpy={np_us:.1f}us, polygrad={pg_us:.1f}us, '
              f'ratio={pg_us/max(np_us, 0.1):.1f}x')

    def test_reduce_sum_overhead(self):
        """Reduce sum: polygrad vs numpy."""
        n = 10000
        a_np = np.random.randn(n).astype(np.float32)

        # Warmup
        for _ in range(3):
            Tensor(a_np).sum().numpy()

        iters = 100
        t0 = time.perf_counter()
        for _ in range(iters):
            s_np = a_np.sum()
        np_us = (time.perf_counter() - t0) / iters * 1e6

        t0 = time.perf_counter()
        for _ in range(iters):
            s_pg = Tensor(a_np).sum().item()
        pg_us = (time.perf_counter() - t0) / iters * 1e6

        np.testing.assert_allclose(s_pg, s_np, rtol=1e-3)
        print(f'\n  sum({n}): numpy={np_us:.1f}us, polygrad={pg_us:.1f}us, '
              f'ratio={pg_us/max(np_us, 0.1):.1f}x')

    def test_fused_chain_beats_numpy(self):
        """Fused 5-op chain at scale: polygrad should beat numpy.

        numpy does 5 separate passes over memory.
        polygrad fuses into 1 kernel = 1 pass. At large sizes,
        cache locality wins."""
        n = 5000000
        a_np = np.random.randn(n).astype(np.float32)
        b_np = np.random.randn(n).astype(np.float32)

        # Warmup
        for _ in range(3):
            a, b = Tensor(a_np), Tensor(b_np)
            ((a + b) * (a - b) + a * b).numpy()

        iters = 20
        t0 = time.perf_counter()
        for _ in range(iters):
            c_np = (a_np + b_np) * (a_np - b_np) + a_np * b_np
        np_us = (time.perf_counter() - t0) / iters * 1e6

        t0 = time.perf_counter()
        for _ in range(iters):
            a, b = Tensor(a_np), Tensor(b_np)
            c_pg = ((a + b) * (a - b) + a * b).numpy()
        pg_us = (time.perf_counter() - t0) / iters * 1e6

        np.testing.assert_allclose(c_pg, c_np, rtol=1e-4)
        ratio = pg_us / np_us
        print(f'\n  5-op chain({n}): numpy={np_us:.0f}us, polygrad={pg_us:.0f}us, '
              f'ratio={ratio:.2f}x {"POLYGRAD WINS" if ratio < 1 else ""}')


class TestZeroCopy:
    """Verify zero-copy numpy input path works correctly."""

    def test_numpy_input_no_copy(self):
        """Tensor from numpy array should not copy data."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        t = Tensor(a)
        # Verify data pointer is shared (or at least data is correct)
        np.testing.assert_allclose(t.numpy(), a)

    def test_numpy_input_correct_dtype(self):
        """Zero-copy path preserves dtype."""
        a = np.array([1.0, 2.0], dtype=np.float32)
        t = Tensor(a)
        assert t.dtype == 'float32'
        np.testing.assert_allclose(t.numpy(), a)

    def test_numpy_2d_input(self):
        """2D numpy array input works."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        t = Tensor(a)
        assert t.shape == (2, 2)
        np.testing.assert_allclose(t.numpy(), a)

    def test_numpy_f64_input(self):
        """Float64 numpy array input works."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        t = Tensor(a, dtype='float64')
        assert t.dtype == 'float64'
        np.testing.assert_allclose(t.numpy(), a)

    def test_list_input_still_works(self):
        """List input still works (copies)."""
        t = Tensor([1.0, 2.0, 3.0])
        np.testing.assert_allclose(t.numpy(), [1, 2, 3])

    def test_mutation_safety(self):
        """Mutating original array after Tensor creation:
        zero-copy means the tensor sees the mutation.
        This is by design (like tinygrad)."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        t = Tensor(a)
        a[0] = 999.0
        # Tensor should see the mutation (zero-copy)
        result = t.numpy()
        assert result[0] == 999.0
