"""Opt-in physical GEMM: mutation, autograd and portable Model contracts."""
import numpy as np
import os
import pytest
import polygrad as pg


def test_runtime_kernel_policy(monkeypatch):
    monkeypatch.setenv('POLY_KERNELS', '1')
    from polygrad import _ffi
    with pg.Runtime(device='CPU', kernels=False) as rt:
        assert _ffi.get_lib().poly_ctx_set_kernel_policy(rt._ctx, 2) == -1
        np.testing.assert_array_equal(rt.Tensor([1., 2.]).realize().numpy(), [1., 2.])
        assert _ffi.get_lib().poly_ctx_set_kernel_policy(rt._ctx, 1) == -1
        assert _ffi.get_lib().poly_ctx_set_kernel_policy(rt._ctx, 0) == 0
    with pytest.raises(TypeError, match='kernels must be a bool'):
        pg.Runtime(kernels='auto')


@pytest.mark.parametrize('shape', [(4, 7, 24), (8, 32, 48), (3, 7, 24), (4, 7, 25)])
def test_cpu_gemm_tensor_and_gradients(shape, monkeypatch):
    monkeypatch.setenv('POLY_KERNELS', '1')
    m, k, n = shape
    a = (np.arange(m*k).reshape(m, k) % 17 - 8).astype(np.float32) / 31
    b = (np.arange(k*n).reshape(k, n) % 13 - 6).astype(np.float32) / 23
    with pg.Runtime(device='CPU', logical='always') as rt:
        x, w = rt.Tensor(a), rt.Tensor(b.T).transpose()
        y = x @ w
        # Backend selection must not put opaque CUSTOM/CALL operations in the
        # authoring graphs: differentiation and exports still see the dot.
        for root in (y.uop_logical, y.uop_physical):
            pending, seen = [root], set()
            while pending:
                u = pending.pop()
                if u.raw in seen:
                    continue
                seen.add(u.raw)
                assert u.op_name not in ('CALL', 'CUSTOM', 'CUSTOMI')
                pending.extend(u.src)
        y.sum().backward()
        np.testing.assert_allclose(y.numpy(), a @ b, atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(x.grad.numpy(), np.ones((m, n), np.float32) @ b.T, atol=2e-5, rtol=2e-5)


def test_cpu_gemm_model_repacking_and_portability(monkeypatch):
    monkeypatch.setenv('POLY_KERNELS', '1')
    a = np.arange(56, dtype=np.float32).reshape(2, 4, 7) / 71
    b = (np.arange(168, dtype=np.float32).reshape(7, 24) % 13 - 6) / 23
    with pg.Runtime(device='CPU', logical='always') as rt:
        w = rt.Tensor(b).realize()
        model = rt.Model(lambda x: (x @ w).relu(), inputs={'x': rt.Tensor.empty(2, 4, 7)}, params={'w': w})
        try:
            original = model.save()
            for scale in (1., -0.75, 2.):
                model.write_buffer('w', b * scale)
                for _ in range(2):
                    np.testing.assert_allclose(model.forward(x=a)['output'], np.maximum(a @ (b * scale), 0), atol=2e-5, rtol=2e-5)
            model.write_buffer('w', b)
            assert model.save() == original
            for device, enabled in [('INTERP', '1'), ('CPU', '0')]:
                monkeypatch.setenv('POLY_KERNELS', enabled)
                with pg.Runtime(device=device) as other:
                    restored = other.Model.load(original)
                    try:
                        restored.place(device)
                        np.testing.assert_allclose(restored.forward(x=a)['output'], np.maximum(a @ b, 0), atol=2e-5, rtol=2e-5)
                    finally:
                        restored.dispose()
        finally:
            model.dispose()


def test_cpu_gemm_backward_transposed_views(monkeypatch, capfd):
    monkeypatch.setenv('POLY_KERNELS', '1')
    monkeypatch.setenv('POLY_DEBUG_KERNELS', '1')
    rng = np.random.default_rng(8)
    # Explicit x@w, dy@w.T, and x.T@dy have tile-compatible shapes. Check the
    # actual autograd graph too: it need not have canonical dot topology.
    a, b, dy = [rng.normal(size=s).astype(np.float32) / 8
                for s in ((8, 24), (24, 48), (8, 48))]
    with pg.Runtime(device='CPU') as rt:
        x, w = rt.Tensor(a), rt.Tensor(b)
        np.testing.assert_allclose((rt.Tensor(dy) @ w.transpose()).numpy(), dy @ b.T, atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose((x.transpose() @ rt.Tensor(dy)).numpy(), a.T @ dy, atol=2e-5, rtol=2e-5)
        enabled = 'gemm_avx2_4x24: selected' in capfd.readouterr().err
        if os.environ.get('POLY_REQUIRE_KERNELS') == '1':
            assert enabled, 'AVX2/FMA provider was not selected on the required CPU target'
        ((x @ w) * rt.Tensor(dy)).sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), dy @ b.T, atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(w.grad.numpy(), a.T @ dy, atol=2e-5, rtol=2e-5)
        if enabled:
            trace = capfd.readouterr().err
            assert 'M=8 N=24 K=48 gemm_avx2_4x24: selected' in trace
            assert 'M=48 N=24 K=8 gemm_avx2_4x24: selected' in trace


@pytest.mark.parametrize('view', ['permute', 'shrink', 'expand'])
def test_cpu_gemm_computed_input_layout(view, monkeypatch):
    monkeypatch.setenv('POLY_KERNELS', '1')
    values = (np.arange(2*3*4*7).reshape(2, 3, 4, 7) % 19 - 9).astype(np.float32) / 37
    with pg.Runtime(device='CPU') as rt:
        x = rt.Tensor(values).sum(axis=1)
        ref = values.sum(axis=1)
        if view == 'shrink':
            x, ref = x[:, :2], ref[:, :2]
        elif view == 'expand':
            x, ref = x[:1].expand(2, 4, 7), np.broadcast_to(ref[:1], (2, 4, 7))
        x = x.permute(1, 0, 2).reshape(-1, 14)
        ref = ref.transpose(1, 0, 2).reshape(-1, 14)
        # Repeat rows so the shrink case also qualifies for the 4-row tile.
        if view == 'shrink':
            x, ref = x.repeat(2, 1), np.tile(ref, (2, 1))
        w = (np.arange(14*24).reshape(14, 24) % 13 - 6).astype(np.float32) / 23
        np.testing.assert_allclose((x @ rt.Tensor(w)).numpy(), ref @ w, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize('shape', [(1, 4, 7, 8), (2, 8, 16, 32)])
def test_cpu_attention_probabilities(shape, monkeypatch):
    monkeypatch.setenv('POLY_KERNELS', '1')
    heads, rows, keys, cols = shape
    rng = np.random.default_rng(7)
    scores = rng.normal(size=(heads, rows, keys)).astype(np.float32)
    scores[:, :, -2:] = -np.inf
    values = rng.normal(size=(heads, keys, cols)).astype(np.float32)
    p = np.exp(scores.astype(np.float64) - scores.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    with pg.Runtime(device='CPU', logical='always') as rt:
        x, v = rt.Tensor(scores), rt.Tensor(values)
        y = x.softmax(-1) @ v
        y.sum().backward()
        np.testing.assert_allclose(y.numpy(), p @ values, atol=2e-5, rtol=2e-5)
        dp = values.sum(-1)[:, None, :]
        np.testing.assert_allclose(x.grad.numpy(), p * (dp - (p * dp).sum(-1, keepdims=True)), atol=2e-5, rtol=2e-5)
        model = rt.Model(lambda x: x.softmax(-1) @ v, inputs={'x': rt.Tensor.empty(*scores.shape)}, params={'v': v})
        try:
            saved = model.save()
            for scale in (1., -.5):
                model.write_buffer('v', values * scale)
                for _ in range(2):
                    np.testing.assert_allclose(model.forward(x=scores)['output'], p @ (values * scale), atol=2e-5, rtol=2e-5)
            model.write_buffer('v', values)
            assert model.save() == saved
        finally:
            model.dispose()


@pytest.mark.parametrize('shape', [(12, 1024, 72), (4, 4096, 120), (16, 1024, 96), (4, 7, 24)])
def test_cpu_gemm_worker_partitions(shape, monkeypatch):
    monkeypatch.setenv('POLY_KERNELS', '1')
    monkeypatch.setenv('THREADS', '1')
    m, k, n = shape
    rng = np.random.default_rng(19)
    a, b = [rng.normal(size=s).astype(np.float32) / 16 for s in ((m, k), (k, n))]
    serial = None
    # Three row tiles, five column tiles and four row tiles exercise even
    # splits below the budget. Small work and indivisible axes stay serial.
    for threads in (1, 2, 3, 4, 5):
        monkeypatch.setenv('NUM_CPU_THREADS', str(threads))
        with pg.Runtime(device='CPU') as rt:
            w = rt.Tensor(b).realize()
            model = rt.Model(lambda x: x @ w, inputs={'x': rt.Tensor.empty(m, k)}, params={'w': w})
            try:
                result = model.forward(x=a)['output']
                np.testing.assert_allclose(result, a.astype(np.float64) @ b, atol=2e-5, rtol=2e-5)
                if serial is None:
                    serial = result.copy()
                else:
                    np.testing.assert_array_equal(result, serial)
                model.write_buffer('w', -b)
                np.testing.assert_array_equal(model.forward(x=a)['output'], -serial)
            finally:
                model.dispose()
