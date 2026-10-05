"""Optional interval reduction lowering; authoring uses ordinary Tensor ops."""
import os
import numpy as np
import pytest
import polygrad as pg


def interval_sum(rt, values, lo, hi):
    n, width = values.shape
    index = rt.Tensor.arange(n, dtype='int32').reshape(1, n, 1)
    return ((index >= lo) & (index < hi)).where(values.reshape(1, n, width), 0).sum(1)


@pytest.mark.parametrize('kernels', [False, True])
@pytest.mark.parametrize('n,width', [(17, 3), (257, 1), (1000, 4)])
def test_interval_sum_bounds_gradients_and_portable_model(kernels, n, width, monkeypatch, capfd):
    monkeypatch.setenv('POLY_DEBUG_KERNELS', '1')
    values = np.random.default_rng(31).normal(size=(n, width)).astype(np.float32)
    # Empty, reversed, overlapping, clipped and extreme int32 bounds. The
    # contract is the mask's interval intersection, not unchecked CSR offsets.
    lo = np.array([0, 0, 3, 8, -5, n, 2**31-1, -2**31], np.int32).reshape(-1, 1, 1)
    hi = np.array([0, 3, 8, 3, n+9, n, -2**31, 2**31-1], np.int32).reshape(-1, 1, 1)
    mask = (np.arange(n)[None, :] >= lo.reshape(-1, 1)) & (np.arange(n)[None, :] < hi.reshape(-1, 1))
    expected = np.where(mask[..., None], values[None], 0).sum(1)
    grad = np.broadcast_to(mask.sum(0)[:, None], values.shape).astype(np.float32)
    with pg.Runtime(device=os.getenv('SEGMENT_DEVICE', 'CPU'), logical='always', kernels=kernels) as rt:
        x = rt.Tensor.empty(n, width)
        a, b = rt.Tensor.empty(lo.shape, dtype='int32'), rt.Tensor.empty(hi.shape, dtype='int32')
        y = interval_sum(rt, x, a, b)
        pending, seen = [y.uop_physical, y.uop_logical], set()
        while pending:
            u = pending.pop()
            if u.raw in seen:
                continue
            seen.add(u.raw)
            assert u.op_name not in ('CALL', 'CUSTOM', 'CUSTOMI')
            pending.extend(u.src)
        dx = y.sum().gradient(x)[0]
        model = rt.Model.from_tensors(inputs={'x': x, 'lo': a, 'hi': b}, outputs={'y': y, 'dx': dx})
        try:
            saved = model.save()
            for scale in (1, -2):
                got = model.call('forward', {'x': values*scale, 'lo': lo, 'hi': hi})
                np.testing.assert_allclose(got['y'], expected*scale, rtol=2e-6, atol=2e-6)
                np.testing.assert_array_equal(got['dx'], grad)
            trace = capfd.readouterr().err
            assert ('segment_sum: selected' in trace) == kernels
            assert model.save() == saved
            with pg.Runtime(device='CPU', kernels=not kernels) as other:
                restored = other.Model.load(saved)
                try:
                    got = restored.call('forward', {'x': values, 'lo': lo, 'hi': hi})
                    np.testing.assert_allclose(got['y'], expected, rtol=2e-6, atol=2e-6)
                    np.testing.assert_array_equal(got['dx'], grad)
                finally:
                    restored.dispose()
        finally:
            model.dispose()
            for t in (dx, y, b, a, x):
                t.dispose()


def test_interval_sum_excludes_nonfinite_values():
    with pg.Runtime(device='CPU', kernels=True) as rt:
        values = rt.Tensor(np.array([[np.nan], [2], [np.inf], [3]], np.float32))
        lo = rt.Tensor(np.array([1, 3, 2], np.int32).reshape(3, 1, 1))
        hi = rt.Tensor(np.array([2, 4, 2], np.int32).reshape(3, 1, 1))
        np.testing.assert_array_equal(interval_sum(rt, values, lo, hi).numpy().ravel(), [2, 3, 0])
