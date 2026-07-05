"""Tests for the polygrad Python Tensor class."""

import numpy as np
import pytest

from polygrad import Device, Jit, JitError, Tensor, Variable, can_run, compile as pg_compile, jit, stats as pg_stats
from polygrad.uop.ops import AxisType, KernelInfo, UOp


class TestCreation:
    def test_from_list(self):
        t = Tensor([1.0, 2.0, 3.0])
        assert t.shape == (3,)
        np.testing.assert_allclose(t.numpy(), [1, 2, 3])

    def test_from_scalar(self):
        t = Tensor(42.0)
        assert t.shape == (1,)
        np.testing.assert_allclose(t.numpy(), [42])

    def test_from_2d(self):
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        assert t.shape == (2, 3)
        np.testing.assert_allclose(t.numpy(), [[1, 2, 3], [4, 5, 6]])

    def test_triu_tril_batched_last_two_dims(self):
        arr = np.arange(1, 19, dtype=np.float32).reshape(2, 3, 3)
        np.testing.assert_allclose(
            Tensor(arr).triu().numpy(),
            np.triu(arr),
        )
        np.testing.assert_allclose(
            Tensor(arr).tril(1).numpy(),
            np.tril(arr, k=1),
        )

        z = Tensor.zeros(5, 0, 3)
        assert z.triu().shape == (5, 0, 3)
        assert z.tril().shape == (5, 0, 3)
        np.testing.assert_allclose(z.triu().numpy(), np.zeros((5, 0, 3), dtype=np.float32))

    def test_empty_creates_unrealized_buffer_placeholder(self):
        t = Tensor.empty((2, 3))
        assert t.shape == (2, 3)
        assert t.uop.has_buffer_identity()
        assert not t.uop.is_realized

    def test_default_context_stats_track_core_work(self):
        before = pg_stats()
        x = Tensor.empty((3,), dtype='float32')
        x.copy_from(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        y = (x + 1).realize()
        np.testing.assert_allclose(y.numpy(), [2.0, 3.0, 4.0])
        after = pg_stats()
        assert after['buffer_write_bytes'] >= before['buffer_write_bytes'] + 12
        assert after['buffer_read_bytes'] >= before['buffer_read_bytes'] + 12
        assert after['launch_count'] >= before['launch_count'] + 1
        assert after['runtime_cache_misses'] >= before['runtime_cache_misses']

    def test_can_run_probes_core_op_shape_support(self):
        assert can_run('add', dtype='float32', shape=(4,))
        assert can_run('matmul', dtype='float32', shapes=((2, 3), (3, 4)))
        assert can_run('gather', dtype='float32', shape=(2, 3))
        assert can_run('sort', dtype='float32', shape=(2, 3))
        assert can_run('argsort', dtype='float32', shape=(2, 3))
        assert can_run('topk', dtype='float32', shape=(2, 3))
        with pytest.raises(ValueError, match='shape is required'):
            can_run('add', dtype='float32')
        with pytest.raises(ValueError, match='shape queries require an op'):
            can_run(shape=(4,), dtype='float32')

    def test_custom_kernel_uop_call(self):
        def add_kernel(c, a, b):
            c, a, b = c.flatten(), a.flatten(), b.flatten()
            i = UOp.range(c.ctx, c.numel(), 0)
            return c[i].store(a[i] + b[i]).end(i).sink()

        a = Tensor([1.0, 2.0, 3.0, 4.0])
        b = Tensor([10.0, 20.0, 30.0, 40.0])
        c = Tensor.empty((4,), dtype='float32')
        out = c.custom_kernel(a, b, fxn=add_kernel)[0]
        np.testing.assert_allclose(out.numpy(), [11.0, 22.0, 33.0, 44.0])

    def test_custom_kernel_multi_output_backward_like_tinygrad(self):
        def addmul_kernel(c, d, a, b):
            c, d, a, b = c.flatten(), d.flatten(), a.flatten(), b.flatten()
            i = UOp.range(c.ctx, c.numel(), 0)
            store_c = c[i].store(a[i] + b[i])
            store_d = d[i].store(a[i] * b[i])
            return store_c.group(store_d).end(i).sink(arg=KernelInfo(name='addmul'))

        def backward_addmul(grad_c, grad_d, call):
            _c, _d, a, b = call.src[1:]
            grad_a = (Tensor(grad_c) + Tensor(grad_d) * Tensor(b)).uop
            grad_b = (Tensor(grad_c) + Tensor(grad_d) * Tensor(a)).uop
            return (None, None, grad_a, grad_b)

        rng = np.random.default_rng(7)
        a_np = rng.standard_normal((4, 4), dtype=np.float32)
        b_np = rng.standard_normal((4, 4), dtype=np.float32)

        a_ref = Tensor(a_np, requires_grad=True)
        b_ref = Tensor(b_np, requires_grad=True)
        ((a_ref + b_ref).sum() + (a_ref * b_ref).sum()).backward()

        a = Tensor(a_np, requires_grad=True)
        b = Tensor(b_np, requires_grad=True)
        a.realize(b)
        c, d, _, _ = Tensor.empty((4, 4)).custom_kernel(
            Tensor.empty((4, 4)), a, b, fxn=addmul_kernel, grad_fxn=backward_addmul
        )
        (c.sum() + d.sum()).backward()
        np.testing.assert_allclose(a.grad.numpy(), a_ref.grad.numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(b.grad.numpy(), b_ref.grad.numpy(), rtol=1e-5, atol=1e-6)

    def test_custom_kernel_reuses_buffers_after_input_update(self):
        def add_kernel(c, a, b):
            c, a, b = c.flatten(), a.flatten(), b.flatten()
            i = UOp.range(c.ctx, c.numel(), 0)
            return c[i].store(a[i] + b[i]).end(i).sink()

        a = Tensor.empty((4,), dtype='float32')
        b = Tensor([10.0, 20.0, 30.0, 40.0])
        c = Tensor.empty((4,), dtype='float32')
        for vals, expected in (
            ([1.0, 2.0, 3.0, 4.0], [11.0, 22.0, 33.0, 44.0]),
            ([5.0, 6.0, 7.0, 8.0], [15.0, 26.0, 37.0, 48.0]),
        ):
            a.copy_from(np.array(vals, dtype=np.float32))
            out = c.custom_kernel(a, b, fxn=add_kernel)[0]
            assert out.uop_logical.op_name == 'AFTER'
            out.realize()
            assert out.uop_physical.has_buffer_identity()
            np.testing.assert_allclose(out.numpy(), expected)

    def test_empty_rejects_name_like_tinygrad(self):
        with pytest.raises(TypeError, match='Tensor.empty does not accept name'):
            Tensor.empty((2, 3), name='z')

    def test_empty_symbolic_shape_survives_realize_and_add(self):
        n = Variable('N', 1, 8)
        x = Tensor.empty(n.bind(4))
        assert not isinstance(x.shape[0], int)

        x.realize()
        assert not isinstance(x.shape[0], int)

        y = x + 1
        assert not isinstance(y.shape[0], int)
        y.realize()
        assert not isinstance(y.shape[0], int)

        with pytest.raises(AssertionError, match='no data if shape is symbolic'):
            y.numpy()

    def test_zeros(self):
        t = Tensor.zeros(3, 4)
        assert t.shape == (3, 4)
        np.testing.assert_allclose(t.numpy(), np.zeros((3, 4)))

    def test_ones(self):
        t = Tensor.ones(2, 3)
        assert t.shape == (2, 3)
        np.testing.assert_allclose(t.numpy(), np.ones((2, 3)))

    def test_full(self):
        t = Tensor.full((2, 2), 7.0)
        np.testing.assert_allclose(t.numpy(), np.full((2, 2), 7.0))

    def test_bfloat16_numpy_widens_to_float32(self):
        t = Tensor([1.0, 2.0, 3.0, 4.0], dtype='bfloat16')
        y = (t + t).realize()
        out = y.numpy()
        assert out.dtype == np.float32
        np.testing.assert_allclose(out, [2.0, 4.0, 6.0, 8.0])

    def test_numpy_many_batches_realize_and_readback(self):
        t = Tensor([1.0, 2.0, 3.0])
        add, mul = Tensor.numpy_many(t + 1, t * 2)
        assert isinstance(add, np.ndarray)
        assert isinstance(mul, np.ndarray)
        np.testing.assert_allclose(add, [2.0, 3.0, 4.0])
        np.testing.assert_allclose(mul, [2.0, 4.0, 6.0])

        i32 = Tensor(np.array([1, 2, 3], dtype=np.int32)) + Tensor(np.array([10, 20, 30], dtype=np.int32))
        f64 = Tensor(np.array([1.0, 2.0], dtype=np.float64)) + 0.5
        empty = Tensor.zeros(0)
        out_i32, out_f64, out_empty = Tensor.numpy_many([i32, f64, empty])
        assert out_i32.dtype == np.int32
        assert out_f64.dtype == np.float64
        assert out_empty.shape == (0,)
        np.testing.assert_array_equal(out_i32, [11, 22, 33])
        np.testing.assert_allclose(out_f64, [1.5, 2.5])

    def test_arange(self):
        t = Tensor.arange(5)
        np.testing.assert_allclose(t.numpy(), np.arange(5, dtype=np.float32))

    def test_rand_manual_seed_deterministic(self):
        Tensor.manual_seed(42)
        a = Tensor.rand(5).numpy()
        b = Tensor.rand(5).numpy()
        Tensor.manual_seed(42)
        c = Tensor.rand(5).numpy()
        d = Tensor.rand(5).numpy()
        np.testing.assert_allclose(a, c)
        np.testing.assert_allclose(b, d)
        assert not np.allclose(a, b)

    def test_randn_manual_seed_deterministic(self):
        Tensor.manual_seed(42)
        a = Tensor.randn(5).numpy()
        b = Tensor.randn(5).numpy()
        Tensor.manual_seed(42)
        c = Tensor.randn(5).numpy()
        d = Tensor.randn(5).numpy()
        np.testing.assert_allclose(a, c)
        np.testing.assert_allclose(b, d)
        assert not np.allclose(a, b)

    def test_randint_range_and_manual_seed(self):
        Tensor.manual_seed(42)
        a = Tensor.randint(5, 10, shape=(64,), dtype='int32').numpy()
        Tensor.manual_seed(42)
        b = Tensor.randint(5, 10, shape=(64,), dtype='int32').numpy()
        np.testing.assert_array_equal(a, b)
        assert np.all(a >= 5)
        assert np.all(a < 10)

    def test_rand_rejects_int_dtype(self):
        with pytest.raises(ValueError, match='rand only supports float dtypes'):
            Tensor.rand(4, dtype='int32')

    def test_item(self):
        t = Tensor([42.0])
        assert t.item() == pytest.approx(42.0)

    def test_copy_from_preserves_buffer_identity_and_updates_jit_input(self):
        x = Tensor.empty((3,), dtype='float32')
        buf = x.uop.buffer
        x.copy_from(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        assert x.uop.buffer == buf
        np.testing.assert_allclose(x.numpy(), [1.0, 2.0, 3.0])

        @Jit
        def f(a):
            return (a + 1).realize()

        np.testing.assert_allclose(f(x).numpy(), [2.0, 3.0, 4.0])
        np.testing.assert_allclose(f(x).numpy(), [2.0, 3.0, 4.0])
        assert f.captured

        x.update_from(np.array([10.0, 20.0, 30.0], dtype=np.float32))
        np.testing.assert_allclose(f(x).numpy(), [11.0, 21.0, 31.0])
        assert x.uop.buffer == buf


class TestJit:
    def test_jit_replays_raw_tensor_realize(self):
        @Jit
        def f(x):
            return (x + 1).realize()

        x0 = Tensor([1.0, 2.0, 3.0]).realize()
        y0 = f(x0)
        np.testing.assert_allclose(y0.numpy(), [2.0, 3.0, 4.0])
        assert not f.captured

        x1 = Tensor([10.0, 20.0, 30.0]).realize()
        y1 = f(x1)
        np.testing.assert_allclose(y1.numpy(), [11.0, 21.0, 31.0])
        assert f.captured
        assert f.schedule_count == 1
        stats = f.stats()
        assert stats['captured']
        assert stats['call_count'] == 2
        assert stats['replay_count'] == 0
        assert stats['schedule_count'] == 1
        assert stats['last_call_ms'] >= 0

        x2 = Tensor([100.0, 200.0, 300.0]).realize()
        y2 = f(x2)
        assert y2 is y1
        np.testing.assert_allclose(y2.numpy(), [101.0, 201.0, 301.0])
        np.testing.assert_allclose(x1.numpy(), [10.0, 20.0, 30.0])
        np.testing.assert_allclose(x2.numpy(), [100.0, 200.0, 300.0])
        assert f.stats()['replay_count'] == 1

    def test_jit_replays_assign_with_current_input(self):
        @Jit
        def f(x):
            return x.assign(x + 1).realize()

        x0 = Tensor([1.0, 2.0, 3.0]).realize()
        y0 = f(x0)
        np.testing.assert_allclose(y0.numpy(), [2.0, 3.0, 4.0])
        np.testing.assert_allclose(x0.numpy(), [2.0, 3.0, 4.0])
        assert not f.captured

        x1 = Tensor([10.0, 20.0, 30.0]).realize()
        y1 = f(x1)
        np.testing.assert_allclose(y1.numpy(), [11.0, 21.0, 31.0])
        np.testing.assert_allclose(x1.numpy(), [11.0, 21.0, 31.0])
        assert f.captured

        x2 = Tensor([100.0, 200.0, 300.0]).realize()
        y2 = f(x2)
        assert y2 is y1
        np.testing.assert_allclose(y2.numpy(), [11.0, 21.0, 31.0])
        np.testing.assert_allclose(x1.numpy(), [11.0, 21.0, 31.0])
        np.testing.assert_allclose(x2.numpy(), [101.0, 201.0, 301.0])

    def test_jit_replays_write_only_assign_with_current_input(self):
        @Jit
        def f(x):
            return x.assign(Tensor([7.0, 8.0, 9.0]).realize()).realize()

        x0 = Tensor([1.0, 2.0, 3.0]).realize()
        y0 = f(x0)
        np.testing.assert_allclose(y0.numpy(), [7.0, 8.0, 9.0])
        np.testing.assert_allclose(x0.numpy(), [7.0, 8.0, 9.0])
        assert not f.captured

        x1 = Tensor([10.0, 20.0, 30.0]).realize()
        y1 = f(x1)
        np.testing.assert_allclose(y1.numpy(), [7.0, 8.0, 9.0])
        np.testing.assert_allclose(x1.numpy(), [7.0, 8.0, 9.0])
        assert f.captured

        x2 = Tensor([100.0, 200.0, 300.0]).realize()
        y2 = f(x2)
        assert y2 is y1
        np.testing.assert_allclose(y2.numpy(), [7.0, 8.0, 9.0])
        np.testing.assert_allclose(x1.numpy(), [7.0, 8.0, 9.0])
        np.testing.assert_allclose(x2.numpy(), [7.0, 8.0, 9.0])

    def test_jit_replays_multiple_realizes(self):
        @Jit
        def f(x):
            y = (x + 1).realize()
            z = (x * 2).realize()
            return y, z

        x0 = Tensor([1.0, 2.0, 3.0]).realize()
        y0, z0 = f(x0)
        np.testing.assert_allclose(y0.numpy(), [2.0, 3.0, 4.0])
        np.testing.assert_allclose(z0.numpy(), [2.0, 4.0, 6.0])
        assert not f.captured

        x1 = Tensor([10.0, 20.0, 30.0]).realize()
        ret1 = f(x1)
        y1, z1 = ret1
        np.testing.assert_allclose(y1.numpy(), [11.0, 21.0, 31.0])
        np.testing.assert_allclose(z1.numpy(), [20.0, 40.0, 60.0])
        assert f.captured
        assert f.schedule_count == 2

        x2 = Tensor([100.0, 200.0, 300.0]).realize()
        ret2 = f(x2)
        assert ret2 is ret1
        np.testing.assert_allclose(y1.numpy(), [101.0, 201.0, 301.0])
        np.testing.assert_allclose(z1.numpy(), [200.0, 400.0, 600.0])
        np.testing.assert_allclose(x2.numpy(), [100.0, 200.0, 300.0])

    def test_jit_preserves_list_and_dict_returns_like_tinygrad(self):
        @Jit
        def as_list(x):
            return [(x + 1).realize()]

        @Jit
        def as_dict(x):
            return {'out': (x * 2).realize()}

        np.testing.assert_allclose(as_list(Tensor([1.0, 2.0]).realize())[0].numpy(), [2.0, 3.0])
        list_ret = as_list(Tensor([10.0, 20.0]).realize())
        assert as_list.captured
        np.testing.assert_allclose(list_ret[0].numpy(), [11.0, 21.0])
        replayed_list = as_list(Tensor([100.0, 200.0]).realize())
        assert replayed_list is list_ret
        np.testing.assert_allclose(list_ret[0].numpy(), [101.0, 201.0])

        np.testing.assert_allclose(as_dict(Tensor([1.0, 2.0]).realize())['out'].numpy(), [2.0, 4.0])
        dict_ret = as_dict(Tensor([10.0, 20.0]).realize())
        assert as_dict.captured
        np.testing.assert_allclose(dict_ret['out'].numpy(), [20.0, 40.0])
        replayed_dict = as_dict(Tensor([100.0, 200.0]).realize())
        assert replayed_dict is dict_ret
        np.testing.assert_allclose(dict_ret['out'].numpy(), [200.0, 400.0])

    def test_jit_rejects_duplicate_input_buffers(self):
        @Jit
        def f(x, y):
            return (x + y).realize()

        x = Tensor([1.0, 2.0, 3.0]).realize()
        with pytest.raises(JitError, match='duplicate inputs'):
            f(x, x)

    def test_jit_rejects_shape_mismatch_after_capture(self):
        @Jit
        def f(x):
            return (x + 1).realize()

        f(Tensor([1.0, 2.0, 3.0]).realize())
        f(Tensor([10.0, 20.0, 30.0]).realize())
        assert f.captured

        with pytest.raises(JitError, match='args mismatch'):
            f(Tensor([100.0, 200.0, 300.0, 400.0]).realize())

    def test_jit_rejects_dtype_mismatch_after_capture(self):
        @Jit
        def f(x):
            return (x + 1).realize()

        f(Tensor([1.0, 2.0, 3.0], dtype='float32').realize())
        f(Tensor([10.0, 20.0, 30.0], dtype='float32').realize())
        assert f.captured

        with pytest.raises(JitError, match='args mismatch'):
            f(Tensor([100, 200, 300], dtype='int32').realize())

    def test_compile_realizes_lazy_return_like_tinygrad(self):
        compiled = pg_compile(lambda x: x * 3 - 1, Tensor([1.0, 2.0, 3.0]))
        out = compiled(Tensor([4.0, 5.0, 6.0]))
        np.testing.assert_allclose(out.numpy(), [11.0, 14.0, 17.0])
        assert compiled.schedule_count == 1

    def test_jit_prune_skips_onetime_side_realize_on_replay(self):
        side = Tensor([-1.0, -1.0, -1.0]).realize()
        seed = Tensor([7.0, 8.0, 9.0]).realize()

        def raw(x):
            side.assign(seed).realize()
            return (x + 1).realize()

        f = Jit(raw, prune=True)
        np.testing.assert_allclose(f(Tensor([1.0, 2.0, 3.0]).realize()).numpy(), [2.0, 3.0, 4.0])
        np.testing.assert_allclose(side.numpy(), [7.0, 8.0, 9.0])
        np.testing.assert_allclose(f(Tensor([10.0, 20.0, 30.0]).realize()).numpy(), [11.0, 21.0, 31.0])
        assert f.captured
        assert f.schedule_count == 2

        side.assign(Tensor([-9.0, -9.0, -9.0])).realize()
        y = f(Tensor([100.0, 200.0, 300.0]).realize())
        np.testing.assert_allclose(y.numpy(), [101.0, 201.0, 301.0])
        np.testing.assert_allclose(side.numpy(), [-9.0, -9.0, -9.0])

    def test_jit_prune_decorator_form(self):
        @jit(prune=True)
        def f(x):
            return (x * 2).realize()

        np.testing.assert_allclose(f(Tensor([1.0, 2.0]).realize()).numpy(), [2.0, 4.0])
        np.testing.assert_allclose(f(Tensor([3.0, 4.0]).realize()).numpy(), [6.0, 8.0])
        assert f.captured

    def test_jit_symbolic_empty_shape_replays_with_runtime_var(self):
        n = Variable('N', 1, 8)

        @Jit
        def f(x):
            return (x + 1).realize()

        y0 = f(Tensor.empty(n.bind(4)))
        assert not f.captured
        assert not isinstance(y0.shape[0], int)

        y1 = f(Tensor.empty(n.bind(4)))
        assert f.captured
        assert f.schedule_count == 1
        assert not isinstance(y1.shape[0], int)

        y2 = f(Tensor.empty(n.bind(6)))
        assert y2 is y1
        assert not isinstance(y2.shape[0], int)

    def test_compile_warms_capture_and_replays(self):
        def f(x):
            return (x + 1).realize()

        sample = Tensor([1.0, 2.0, 3.0]).realize()
        compiled = pg_compile(f, [sample])
        assert compiled.schedule_count == 1
        stats = compiled.stats()
        assert stats['capture_runs'] == 2
        assert stats['compile_ms'] >= 0
        assert stats['input_count'] == 1

        out = compiled.run([Tensor([10.0, 20.0, 30.0]).realize()])
        np.testing.assert_allclose(out.numpy(), [11.0, 21.0, 31.0])
        stats = compiled.stats()
        assert stats['run_count'] == 1
        assert stats['last_run_ms'] >= 0
        assert stats['schedule_count'] == 1

        with pytest.raises(JitError, match='args mismatch'):
            compiled.run([Tensor([1.0, 2.0, 3.0, 4.0]).realize()])

        compiled.dispose()
        with pytest.raises(JitError, match='disposed'):
            compiled.run([sample])

    def test_jit_captures_custom_kernel_and_replays_after_input_update(self):
        def add_kernel(c, a, b):
            c, a, b = c.flatten(), a.flatten(), b.flatten()
            i = UOp.range(c.ctx, c.numel(), 0)
            return c[i].store(a[i] + b[i]).end(i).sink()

        @Jit
        def f(a, b):
            c = Tensor.empty((4,), dtype='float32')
            return c.custom_kernel(a, b, fxn=add_kernel)[0]

        a = Tensor.empty((4,), dtype='float32')
        b = Tensor([10.0, 20.0, 30.0, 40.0])
        a.copy_from(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        np.testing.assert_allclose(f(a, b).numpy(), [11.0, 22.0, 33.0, 44.0])
        np.testing.assert_allclose(f(a, b).numpy(), [11.0, 22.0, 33.0, 44.0])
        assert f.captured
        assert f.schedule_count == 1

        a.copy_from(np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32))
        np.testing.assert_allclose(f(a, b).numpy(), [15.0, 26.0, 37.0, 48.0])
        assert f.replay_count == 1

    def test_compile_captures_custom_kernel_and_replays_after_input_update(self):
        def add_kernel(c, a, b):
            c, a, b = c.flatten(), a.flatten(), b.flatten()
            i = UOp.range(c.ctx, c.numel(), 0)
            return c[i].store(a[i] + b[i]).end(i).sink()

        def f(a, b):
            c = Tensor.empty((4,), dtype='float32')
            return c.custom_kernel(a, b, fxn=add_kernel)[0]

        a = Tensor.empty((4,), dtype='float32')
        b = Tensor([10.0, 20.0, 30.0, 40.0])
        a.copy_from(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        compiled = pg_compile(f, [a, b])
        assert compiled.schedule_count == 1

        a.copy_from(np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32))
        out = compiled.run([a, b])
        np.testing.assert_allclose(out.numpy(), [15.0, 26.0, 37.0, 48.0])
        assert compiled.stats()['run_count'] == 1

    def test_compile_captures_custom_kernel_fused_reduction_and_replays_after_input_update(self):
        def summary_kernel(out, a, b):
            out, a, b = out.flatten(), a.flatten(), b.flatten()
            c = UOp.range(out.ctx, 2, 0)
            r = UOp.range(out.ctx, 4, 1, AxisType.REDUCE)
            offset = c * 4 + r
            term = a[offset] * b[r]
            return out[c].store(term.sum(r)).end(c).sink()

        def f(a, b):
            out = Tensor.empty((2,), dtype='float32')
            return out.custom_kernel(a, b, fxn=summary_kernel)[0]

        a = Tensor.empty((8,), dtype='float32')
        b = Tensor([1.0, 2.0, 3.0, 4.0])
        a.copy_from(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32))
        compiled = pg_compile(f, [a, b])
        assert compiled.schedule_count == 1
        np.testing.assert_allclose(compiled.run([a, b]).numpy(), [30.0, 70.0])

        a.copy_from(np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float32))
        np.testing.assert_allclose(compiled.run([a, b]).numpy(), [40.0, 80.0])
        assert compiled.stats()['run_count'] == 2

    def test_compile_captures_custom_kernel_multi_output_fused_reductions(self):
        def summary_kernel(out0, out1, a, b):
            out0, out1, a, b = out0.flatten(), out1.flatten(), a.flatten(), b.flatten()
            c = UOp.range(out0.ctx, 2, 0)
            r = UOp.range(out0.ctx, 4, 1, AxisType.REDUCE)
            term = a[c * 4 + r]
            s0 = term.sum(r)
            s1 = (term * b[r]).sum(r)
            st0 = out0[c].store(s0)
            st1 = out1[c].store(s1)
            return st0.end(c).sink(st1.end(c))

        def f(a, b):
            out0 = Tensor.empty((2,), dtype='float32')
            out1 = Tensor.empty((2,), dtype='float32')
            outs = out0.custom_kernel(out1, a, b, fxn=summary_kernel)
            return [outs[0], outs[1]]

        a = Tensor.empty((8,), dtype='float32')
        b = Tensor([1.0, 2.0, 3.0, 4.0])
        a.copy_from(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32))
        compiled = pg_compile(f, [a, b])
        out0, out1 = compiled.run([a, b])
        np.testing.assert_allclose(out0.numpy(), [10.0, 26.0])
        np.testing.assert_allclose(out1.numpy(), [30.0, 70.0])

        a.copy_from(np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float32))
        out0, out1 = compiled.run([a, b])
        np.testing.assert_allclose(out0.numpy(), [14.0, 30.0])
        np.testing.assert_allclose(out1.numpy(), [40.0, 80.0])
        assert compiled.stats()['run_count'] == 2

    def test_compile_captures_grouped_custom_kernel_compact_summary_with_intercept_reductions(self):
        def summary_kernel(out, x, y):
            out, x, y = out.flatten(), x.flatten(), y.flatten()
            candidates = 2
            rows = 128
            c = UOp.range(out.ctx, candidates, 0)
            r = UOp.range(out.ctx, rows, 1, AxisType.REDUCE)
            one = UOp.const(out.ctx, 1.0)
            xv = x[c * rows + r]
            yv = y[r]
            stats = [
                one.sum(r),
                xv.sum(r),
                (xv * xv).sum(r),
                yv.sum(r),
                (xv * yv).sum(r),
            ]
            stores = [out[c + stat * candidates].store(s) for stat, s in enumerate(stats)]
            return stores[0].group(*stores[1:]).end(c).sink()

        def f(x, y):
            out = Tensor.empty((10,), dtype='float32')
            return out.custom_kernel(x, y, fxn=summary_kernel)[0]

        def expected(xv, yv):
            out = np.zeros((10,), dtype=np.float32)
            out[0:2] = 128.0
            for c in range(2):
                xs = xv[c * 128:(c + 1) * 128]
                out[c + 2] = xs.sum()
                out[c + 4] = (xs * xs).sum()
                out[c + 6] = yv.sum()
                out[c + 8] = (xs * yv).sum()
            return out

        x = Tensor.empty((256,), dtype='float32')
        y_data = np.arange(1, 129, dtype=np.float32)
        y = Tensor(y_data)
        x0 = np.arange(1, 257, dtype=np.float32)
        x.copy_from(x0)
        compiled = pg_compile(f, [x, y])
        np.testing.assert_allclose(compiled.run([x, y]).numpy(), expected(x0, y_data))

        x1 = np.arange(2, 258, dtype=np.float32)
        x.copy_from(x1)
        np.testing.assert_allclose(compiled.run([x, y]).numpy(), expected(x1, y_data))

    def test_compile_captures_sym_style_custom_kernel_fused_summary_reductions(self):
        candidates = 4
        rows = 32
        terms = 5
        stats_per_candidate = 2 + 2 * terms + (terms * (terms + 1)) // 2

        def summary_kernel(out, x, y):
            out, x, y = out.flatten(), x.flatten(), y.flatten()
            c = UOp.range(out.ctx, candidates, 0)
            r = UOp.range(out.ctx, rows, 1, AxisType.REDUCE)
            one = UOp.const(out.ctx, 1.0)
            yv = y[r]

            def term_at(t):
                return x[(c * terms + t) * rows + r]

            stats = [one.sum(r), yv.sum(r)]
            for t in range(terms):
                tv = term_at(t)
                stats.append(tv.sum(r))
                stats.append((tv * yv).sum(r))
            for i in range(terms):
                ti = term_at(i)
                for j in range(i, terms):
                    stats.append((ti * term_at(j)).sum(r))
            stores = [out[c + stat * candidates].store(s) for stat, s in enumerate(stats)]
            return stores[0].group(*stores[1:]).end(c).sink()

        def f(x, y):
            out = Tensor.empty((candidates * stats_per_candidate,), dtype='float32')
            return out.custom_kernel(x, y, fxn=summary_kernel)[0]

        def make_x(shift=0):
            i = np.arange(candidates * terms * rows, dtype=np.float32)
            return np.sin((i + shift) * 0.013).astype(np.float32) + np.cos((np.mod(i, 17)) * 0.07).astype(np.float32) + 0.001 * i

        y_data = np.array([np.cos(i * 0.05) - 0.25 for i in range(rows)], dtype=np.float32)

        def expected(xv):
            out = np.zeros((candidates * stats_per_candidate,), dtype=np.float32)
            for c in range(candidates):
                out[c] = rows
                out[c + candidates] = y_data.sum()
                stat = 2
                for t in range(terms):
                    tv = xv[(c * terms + t) * rows:(c * terms + t + 1) * rows]
                    out[c + stat * candidates] = tv.sum()
                    out[c + (stat + 1) * candidates] = (tv * y_data).sum()
                    stat += 2
                for i in range(terms):
                    ti = xv[(c * terms + i) * rows:(c * terms + i + 1) * rows]
                    for j in range(i, terms):
                        tj = xv[(c * terms + j) * rows:(c * terms + j + 1) * rows]
                        out[c + stat * candidates] = (ti * tj).sum()
                        stat += 1
            return out

        x = Tensor.empty((candidates * terms * rows,), dtype='float32')
        y = Tensor(y_data)
        x0 = make_x(0)
        x.copy_from(x0)
        compiled = pg_compile(f, [x, y])
        np.testing.assert_allclose(compiled.run([x, y]).numpy(), expected(x0), rtol=2e-5, atol=2e-3)

        x1 = make_x(3)
        x.copy_from(x1)
        np.testing.assert_allclose(compiled.run([x, y]).numpy(), expected(x1), rtol=2e-5, atol=2e-3)

    def test_compile_captures_custom_kernel_tinygrad_style_set_accumulator_reduction(self):
        def sum_kernel(out, x):
            out, x = out.flatten(), x.flatten()
            candidates = 4
            rows = 64
            c = UOp.range(out.ctx, candidates, 0)
            r = UOp.range(out.ctx, rows, 1, AxisType.REDUCE)
            acc = out[c].set(0.0)
            acc = acc[c].set(acc.after(r)[c] + x[c * rows + r], end=r)
            return acc.end(c).sink(arg=KernelInfo(name='custom_sum_4_64', opts_to_apply=()))

        def f(x):
            out = Tensor.empty((4,), dtype='float32')
            return out.custom_kernel(x, fxn=sum_kernel)[0]

        x_data = np.arange(1, 257, dtype=np.float32)
        x = Tensor(x_data)
        compiled = pg_compile(f, [x])
        expected = x_data.reshape(4, 64).sum(axis=1)
        np.testing.assert_allclose(compiled.run([x]).numpy(), expected, rtol=1e-5, atol=1e-5)

    def test_compile_captures_staged_custom_kernel_outputs_feeding_another_custom_kernel(self):
        candidates = 4
        rows = 64

        def term_kernel(t0, t1, x, y):
            t0, t1, x, y = t0.flatten(), t1.flatten(), x.flatten(), y.flatten()
            c = UOp.range(t0.ctx, candidates, 0)
            r = UOp.range(t0.ctx, rows, 1)
            idx = c * rows + r
            xv = x[idx]
            yv = y[r]
            st0 = t0[idx].store(xv + yv)
            st1 = t1[idx].store(xv * yv)
            return st0.group(st1).end(c, r).sink(
                arg=KernelInfo(name='custom_stage_terms_4_64', opts_to_apply=())
            )

        def summary_kernel(out, t0, t1):
            out, t0, t1 = out.flatten(), t0.flatten(), t1.flatten()
            c = UOp.range(out.ctx, candidates, 0)
            r = UOp.range(out.ctx, rows, 1, AxisType.REDUCE)
            idx = c * rows + r
            s0 = t0[idx].sum(r)
            s1 = t1[idx].sum(r)
            st0 = out[c].store(s0)
            st1 = out[c + candidates].store(s1)
            return st0.group(st1).end(c).sink(
                arg=KernelInfo(name='custom_stage_summary_4_64', opts_to_apply=())
            )

        def f(x, y):
            t0 = Tensor.empty((candidates * rows,), dtype='float32')
            t1 = Tensor.empty((candidates * rows,), dtype='float32')
            staged = t0.custom_kernel(t1, x, y, fxn=term_kernel)
            out = Tensor.empty((candidates * 2,), dtype='float32')
            return out.custom_kernel(staged[0], staged[1], fxn=summary_kernel)[0]

        x_data = np.array([np.sin(i * 0.01) + i * 0.001 for i in range(candidates * rows)], dtype=np.float32)
        y_data = np.array([np.cos(i * 0.02) - 0.25 for i in range(rows)], dtype=np.float32)
        expected = np.zeros((candidates * 2,), dtype=np.float32)
        for c in range(candidates):
            xs = x_data[c * rows:(c + 1) * rows]
            expected[c] = (xs + y_data).sum()
            expected[c + candidates] = (xs * y_data).sum()

        compiled = pg_compile(f, [Tensor(x_data), Tensor(y_data)])
        np.testing.assert_allclose(compiled.run([Tensor(x_data), Tensor(y_data)]).numpy(), expected, rtol=1e-5, atol=1e-5)

    def test_compiled_custom_kernel_consumer_reads_producer_output_after_readback(self):
        n = 1024

        def producer_kernel(out, x):
            out, x = out.flatten(), x.flatten()
            i = UOp.range(out.ctx, out.numel(), 0)
            return out[i].store(x[i] * 2 + 1).end(i).sink(
                arg=KernelInfo(name='custom_producer_readback_rebind', opts_to_apply=())
            )

        def consumer_kernel(out, y):
            out, y = out.flatten(), y.flatten()
            r = UOp.range(out.ctx, n, 0, AxisType.REDUCE)
            return out[0].store(y[r].sum(r)).sink(
                arg=KernelInfo(name='custom_consumer_readback_rebind', opts_to_apply=())
            )

        x0 = np.arange(n, dtype=np.float32) / 17
        x = Tensor(x0)
        producer = pg_compile(
            lambda tx: Tensor.empty((n,), dtype='float32').custom_kernel(tx, fxn=producer_kernel)[0],
            [x],
        )

        first_y = producer.run([x])
        first_y.realize()
        consumer = pg_compile(
            lambda ty: Tensor.empty((1,), dtype='float32').custom_kernel(ty, fxn=consumer_kernel)[0],
            [first_y],
        )

        first = consumer.run([first_y])
        np.testing.assert_allclose(first.numpy(), [(x0 * 2 + 1).sum()], rtol=1e-5, atol=1e-2)

        x1 = 10 + np.arange(n, dtype=np.float32) / 11
        x.copy_from(x1)
        second_y = producer.run([x])
        second = consumer.run([second_y])
        np.testing.assert_allclose(second.numpy(), [(x1 * 2 + 1).sum()], rtol=1e-5, atol=1e-2)

    def test_custom_kernel_exposes_uop_compare_where_and_unary_methods(self):
        def select_kernel(out, a, b):
            out, a, b = out.flatten(), a.flatten(), b.flatten()
            i = UOp.range(out.ctx, out.numel(), 0)
            av = a[i]
            bv = b[i]
            selected = av.lt(0).where(-av, av.max(bv))
            return out[i].store(selected).end(i).sink()

        out = Tensor.empty((4,), dtype='float32')
        a = Tensor([-3.0, 2.0, 5.0, -1.0])
        b = Tensor([1.0, 4.0, 3.0, 9.0])
        np.testing.assert_allclose(
            out.custom_kernel(a, b, fxn=select_kernel)[0].numpy(),
            [3.0, 4.0, 5.0, 1.0],
        )

    def test_custom_kernel_exposes_uop_floor_div_and_mod(self):
        from polygrad.dtype import dtypes

        def index_kernel(out):
            out = out.flatten()
            i = UOp.range(out.ctx, out.numel(), 0)
            row = i.floormod(4)
            col = i.floordiv(4)
            return out[i].store((col * 10 + row).cast(dtypes.float32)).end(i).sink()

        out = Tensor.empty((12,), dtype='float32')
        np.testing.assert_allclose(
            out.custom_kernel(fxn=index_kernel)[0].numpy(),
            [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23],
        )

        def signed_kernel(out, x, y):
            out, x, y = out.flatten(), x.flatten(), y.flatten()
            i = UOp.range(out.ctx, x.numel(), 0)
            q = x[i].floordiv(y[i])
            r = x[i].floormod(y[i])
            return out[i].store(q).end(i).sink(out[i + x.numel()].store(r).end(i))

        x = Tensor(np.array([-7, -7, 7, 7, -1, 1, 0], dtype=np.int32))
        y = Tensor(np.array([3, -3, -3, 3, 4, -4, 3], dtype=np.int32))
        out = Tensor.empty((14,), dtype='int32')
        np.testing.assert_array_equal(
            out.custom_kernel(x, y, fxn=signed_kernel)[0].numpy(),
            np.array([-3, 2, -3, 2, -1, -1, 0, 2, -1, -2, 1, 3, -3, 0], dtype=np.int32),
        )

    def test_custom_kernel_numeric_literals_follow_float_operand_dtype(self):
        def literal_kernel(out, x):
            out, x = out.flatten(), x.flatten()
            i = UOp.range(out.ctx, x.numel(), 0)
            same_add = (x[i] - 1) / (x[i] + 1)
            same_mul = (x[i] * 2) / (x[i] * 3)
            s0 = out[i].store(same_add)
            s1 = out[i + x.numel()].store(same_mul)
            return s0.end(i).sink(s1.end(i))

        x_np = (np.arange(16, dtype=np.float32) / 10.0) + 1.0
        out = Tensor.empty((32,), dtype='float32')
        got = out.custom_kernel(Tensor(x_np), fxn=literal_kernel)[0].numpy()
        expected = np.concatenate([
            (x_np - 1.0) / (x_np + 1.0),
            (x_np * 2.0) / (x_np * 3.0),
        ])
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


class TestElementwise:
    def test_add(self):
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])
        c = a + b
        np.testing.assert_allclose(c.numpy(), [5, 7, 9])

    def test_sub(self):
        a = Tensor([10, 20, 30])
        b = Tensor([1, 2, 3])
        np.testing.assert_allclose((a - b).numpy(), [9, 18, 27])

    def test_mul(self):
        a = Tensor([2, 3, 4])
        b = Tensor([5, 6, 7])
        np.testing.assert_allclose((a * b).numpy(), [10, 18, 28])

    def test_div(self):
        a = Tensor([10, 20, 30])
        b = Tensor([2, 4, 5])
        np.testing.assert_allclose((a / b).numpy(), [5, 5, 6])

    def test_neg(self):
        a = Tensor([1, -2, 3])
        np.testing.assert_allclose((-a).numpy(), [-1, 2, -3])

    def test_scalar_add(self):
        a = Tensor([1, 2, 3])
        c = a + 2.0
        np.testing.assert_allclose(c.numpy(), [3, 4, 5])

    def test_scalar_mul(self):
        a = Tensor([1, 2, 3])
        c = a * 3.0
        np.testing.assert_allclose(c.numpy(), [3, 6, 9])

    def test_exp2(self):
        a = Tensor([0, 1, 2, 3])
        np.testing.assert_allclose(a.exp2().numpy(), [1, 2, 4, 8])

    def test_sqrt(self):
        a = Tensor([1, 4, 9, 16])
        np.testing.assert_allclose(a.sqrt().numpy(), [1, 2, 3, 4])

    def test_chain(self):
        a = Tensor([1, 2, 3, 4])
        b = Tensor([0.5, 0.5, 0.5, 0.5])
        c = (a + Tensor([2, 2, 2, 2])) * b
        np.testing.assert_allclose(c.numpy(), [1.5, 2, 2.5, 3])


class TestMovement:
    def test_reshape(self):
        a = Tensor([1, 2, 3, 4, 5, 6])
        b = a.reshape(2, 3)
        assert b.shape == (2, 3)
        np.testing.assert_allclose(b.numpy(), [[1, 2, 3], [4, 5, 6]])

    def test_permute(self):
        a = Tensor(np.arange(12, dtype=np.float32).reshape(3, 4).tolist())
        b = a.permute(1, 0)
        assert b.shape == (4, 3)
        expected = np.arange(12, dtype=np.float32).reshape(3, 4).T
        np.testing.assert_allclose(b.numpy(), expected)

    def test_flip(self):
        a = Tensor([1, 2, 3, 4, 5])
        b = a.flip(0)
        np.testing.assert_allclose(b.numpy(), [5, 4, 3, 2, 1])

    def test_pad(self):
        a = Tensor([1, 2, 3])
        b = a.pad(((1, 1),))
        assert b.shape == (5,)
        np.testing.assert_allclose(b.numpy(), [0, 1, 2, 3, 0])

    def test_roll_1d(self):
        a = Tensor.arange(5)
        np.testing.assert_allclose(a.roll(2, 0).numpy(), np.roll(np.arange(5, dtype=np.float32), 2))
        np.testing.assert_allclose(a.roll(-1, 0).numpy(), np.roll(np.arange(5, dtype=np.float32), -1))
        np.testing.assert_allclose(a.roll(7, 0).numpy(), np.roll(np.arange(5, dtype=np.float32), 7))

    def test_roll_dims_none_flattens(self):
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        a = Tensor(arr.tolist())
        b = a.roll(2)
        assert b.shape == (3, 4)
        np.testing.assert_allclose(b.numpy(), np.roll(arr.reshape(-1), 2).reshape(3, 4))

    def test_roll_2d_single_and_tuple_dims(self):
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        a = Tensor(arr.tolist())
        np.testing.assert_allclose(a.roll(1, 0).numpy(), np.roll(arr, 1, axis=0))
        np.testing.assert_allclose(a.roll(-2, -1).numpy(), np.roll(arr, -2, axis=-1))
        np.testing.assert_allclose(a.roll((1, -2), (0, 1)).numpy(), np.roll(arr, (1, -2), axis=(0, 1)))

    def test_roll_rejects_shift_dim_length_mismatch(self):
        with pytest.raises(RuntimeError, match=r"len\(dims\)=2 != len\(shifts\)=1"):
            Tensor.arange(12).reshape(3, 4).roll(1, (0, 1))

    def test_gather_matches_tinygrad_probe(self):
        t = Tensor([[1.0, 2.0], [3.0, 4.0]])
        idx = Tensor(np.array([[0, 0], [1, 0]], dtype=np.int32), dtype='int32')
        out = t.gather(1, idx)
        assert out.shape == (2, 2)
        np.testing.assert_allclose(out.numpy(), [[1.0, 1.0], [4.0, 3.0]])
        x3 = Tensor.arange(24).reshape(2, 3, 4)
        idx3 = Tensor(np.array([[[0, 2], [1, 0]], [[2, 1], [0, 2]]], dtype=np.int32), dtype='int32')
        out3 = x3.gather(1, idx3)
        assert out3.shape == (2, 2, 2)
        np.testing.assert_allclose(out3.numpy(), [[[0, 9], [4, 1]], [[20, 17], [12, 21]]])

    def test_scatter_matches_tinygrad_probe(self):
        idx0 = Tensor(np.array([[0, 1, 2, 0]], dtype=np.int32), dtype='int32')
        src = Tensor.arange(1, 11).reshape(2, 5)
        base0 = Tensor.zeros(3, 5, dtype=src.dtype)
        np.testing.assert_allclose(base0.scatter(0, idx0, src).numpy(), [[1, 0, 0, 4, 0], [0, 2, 0, 0, 0], [0, 0, 3, 0, 0]])

        base = Tensor(np.zeros((3, 5), dtype=np.float32))
        idx1 = Tensor(np.array([[0, 1, 2], [0, 1, 4], [2, 3, 4]], dtype=np.int32), dtype='int32')
        src1 = Tensor(np.array([[1, 2, 3], [6, 7, 8], [9, 10, 11]], dtype=np.float32))
        np.testing.assert_allclose(base.scatter(1, idx1, src1).numpy(), [[1, 2, 3, 0, 0], [6, 7, 0, 0, 8], [0, 0, 9, 10, 11]])

        dup_idx = Tensor(np.array([[1, 1, 2]], dtype=np.int32), dtype='int32')
        dup_src = Tensor(np.array([[7, 9, 8]], dtype=np.float32))
        np.testing.assert_allclose(Tensor([[0.0, 0.0, 0.0, 0.0]]).scatter(1, dup_idx, dup_src).numpy(), [[0, 9, 8, 0]])

        scalar_idx = Tensor(np.array([[2], [3]], dtype=np.int32), dtype='int32')
        np.testing.assert_allclose(Tensor.full((2, 4), 2.0).scatter(1, scalar_idx, 1.23, reduce='add').numpy(), [[2, 2, 3.23, 2], [2, 2, 2, 3.23]], rtol=1e-6)
        np.testing.assert_allclose(Tensor.full((2, 4), 2.0).scatter(1, scalar_idx, 1.23, reduce='multiply').numpy(), [[2, 2, 2.46, 2], [2, 2, 2, 2.46]], rtol=1e-6)

        with pytest.raises(TypeError, match="must be one of"):
            base.scatter(1, idx1, src1, reduce='sum')
        with pytest.raises(TypeError, match="non-scalar src"):
            base.scatter(1, idx1, src1, reduce='add')

    def test_scatter_reduce_matches_tinygrad_probe(self):
        base = Tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        idx = Tensor(np.array([[0, 0, 1, 1, 2, 2, 3, 3, 4, 4]], dtype=np.int32), dtype='int32')
        src = Tensor([[1.0, 6.0, 2.0, 7.0, 3.0, 8.0, 4.0, 9.0, 5.0, 10.0]])
        np.testing.assert_allclose(base.scatter_reduce(1, idx, src, 'sum').numpy(), [[8, 11, 14, 17, 20]])
        np.testing.assert_allclose(base.scatter_reduce(1, idx, src, 'prod').numpy(), [[6, 28, 72, 144, 250]])
        np.testing.assert_allclose(base.scatter_reduce(1, idx, src, 'mean', include_self=False).numpy(), [[3.5, 4.5, 5.5, 6.5, 7.5]])
        extreme_base = Tensor([[-10.0, 20.0, 0.0, 5.0, 10.0]])
        np.testing.assert_allclose(extreme_base.scatter_reduce(1, idx, src, 'amax').numpy(), [[6, 20, 8, 9, 10]])
        np.testing.assert_allclose(extreme_base.scatter_reduce(1, idx, src, 'amin').numpy(), [[-10, 2, 0, 4, 5]])
        with pytest.raises(RuntimeError, match="must be one of"):
            base.scatter_reduce(1, idx, src, 'max')


class TestStepSlicing:
    def test_step2_1d(self):
        a = Tensor([1, 2, 3, 4, 5, 6, 7, 8])
        b = a[::2]
        assert b.shape == (4,)
        np.testing.assert_allclose(b.numpy(), [1, 3, 5, 7])

    def test_step3_1d(self):
        a = Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
        b = a[::3]
        assert b.shape == (3,)
        np.testing.assert_allclose(b.numpy(), [1, 4, 7])

    def test_step2_with_start_stop(self):
        a = Tensor([0, 1, 2, 3, 4, 5, 6, 7])
        b = a[1:7:2]
        assert b.shape == (3,)
        np.testing.assert_allclose(b.numpy(), [1, 3, 5])

    def test_step_non_divisible(self):
        # 7 elements, step 3 -> ceildiv(7,3)=3 elements
        a = Tensor([0, 1, 2, 3, 4, 5, 6])
        b = a[::3]
        assert b.shape == (3,)
        np.testing.assert_allclose(b.numpy(), [0, 3, 6])

    def test_negative_step(self):
        a = Tensor([1, 2, 3, 4, 5, 6])
        b = a[::-1]
        np.testing.assert_allclose(b.numpy(), [6, 5, 4, 3, 2, 1])

    def test_negative_step2(self):
        a = Tensor([1, 2, 3, 4, 5, 6])
        b = a[::-2]
        assert b.shape == (3,)
        np.testing.assert_allclose(b.numpy(), [6, 4, 2])

    def test_step_2d_axis0(self):
        a = Tensor(np.arange(12, dtype=np.float32).reshape(4, 3).tolist())
        b = a[::2]
        assert b.shape == (2, 3)
        np.testing.assert_allclose(b.numpy(), [[0, 1, 2], [6, 7, 8]])

    def test_step_2d_axis1(self):
        a = Tensor(np.arange(12, dtype=np.float32).reshape(3, 4).tolist())
        b = a[:, ::2]
        assert b.shape == (3, 2)
        np.testing.assert_allclose(b.numpy(), [[0, 2], [4, 6], [8, 10]])

    def test_step_2d_both(self):
        a = Tensor(np.arange(16, dtype=np.float32).reshape(4, 4).tolist())
        b = a[::2, ::2]
        assert b.shape == (2, 2)
        np.testing.assert_allclose(b.numpy(), [[0, 2], [8, 10]])

    def test_step_backward(self):
        a = Tensor([1, 2, 3, 4, 5, 6, 7, 8], requires_grad=True)
        b = a[::2]  # [1, 3, 5, 7]
        c = b.sum()
        c.backward()
        # grad should be [1,0,1,0,1,0,1,0]
        np.testing.assert_allclose(a.grad.numpy(), [1, 0, 1, 0, 1, 0, 1, 0])


class TestReduce:
    def test_sum_all(self):
        a = Tensor([1, 2, 3, 4])
        s = a.sum()
        assert s.item() == pytest.approx(10.0)

    def test_sum_axis(self):
        a = Tensor([[1, 2, 3], [4, 5, 6]])
        s = a.reshape(2, 3).sum(axis=1)
        np.testing.assert_allclose(s.numpy(), [6, 15])

    def test_argmax_matches_tinygrad_probe(self):
        a = Tensor([[1.2, 0.5, 1.2], [2.2, 1.9, 0.0]])
        np.testing.assert_allclose(a.argmax(axis=1).numpy(), [0, 0])
        np.testing.assert_allclose(a.argmax(axis=1, keepdim=True).numpy(), [[0], [0]])
        assert a.argmax().item() == 3

    def test_sort_argsort_topk_match_tinygrad_probe(self):
        x = Tensor([[0.1, 0.5, 1.2, 3.4, 2.1], [2.2, 1.9, 0.3, 4.5, 0.8]])
        vals, idx = x.sort(1, False)
        np.testing.assert_allclose(vals.numpy(), [[0.1, 0.5, 1.2, 2.1, 3.4],
                                                  [0.3, 0.8, 1.9, 2.2, 4.5]], rtol=1e-6)
        np.testing.assert_array_equal(idx.numpy(), [[0, 1, 2, 4, 3], [2, 4, 1, 0, 3]])

        vals, idx = x.topk(2, dim=1)
        np.testing.assert_allclose(vals.numpy(), [[3.4, 2.1], [4.5, 2.2]], rtol=1e-6)
        np.testing.assert_array_equal(idx.numpy(), [[3, 4], [3, 0]])

        vals, idx = x.topk(2, dim=1, largest=False)
        np.testing.assert_allclose(vals.numpy(), [[0.1, 0.5], [0.3, 0.8]], rtol=1e-6)
        np.testing.assert_array_equal(idx.numpy(), [[0, 1], [2, 4]])

        t = Tensor([[2, 3, 4, 1], [1, 4, 3, 2]])
        np.testing.assert_array_equal(t.argsort().numpy(), [[3, 0, 1, 2], [0, 3, 2, 1]])

    def test_topk_tie_order_and_errors_match_tinygrad_probe(self):
        tie = Tensor([[1.0, 1.0, 0.0, 1.0]])
        vals, idx = tie.topk(3, dim=1)
        np.testing.assert_allclose(vals.numpy(), [[1.0, 1.0, 1.0]])
        np.testing.assert_array_equal(idx.numpy(), [[0, 1, 3]])
        with pytest.raises(ValueError, match='selected index k=6 is out of range'):
            Tensor([[0.1, 0.2]]).topk(6, dim=1)
        with pytest.raises(NotImplementedError, match='sorted_=False'):
            Tensor([[0.1, 0.2]]).topk(1, dim=1, sorted_=False)


class TestMatmulAndLoss:
    def test_einsum_c_api_wrapper(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([[5.0, 6.0], [7.0, 8.0]])
        out = Tensor.einsum('ij,jk->ik', a, b)
        assert out.shape == (2, 2)
        np.testing.assert_allclose(out.numpy(), np.array([[19.0, 22.0], [43.0, 50.0]], dtype=np.float32))

        with pytest.raises(ValueError, match='poly_einsum failed'):
            Tensor.einsum('a->' + ('a' * 80), Tensor([1.0]))

    def test_matmul_shape_mismatch_raises(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([[1.0, 2.0, 3.0]])
        with pytest.raises(ValueError, match='cannot dot'):
            a @ b

    def test_matmul_broadcast_batch_values(self):
        a_np = np.array([
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ], dtype=np.float32)
        b_np = np.array([
            [[1.0, 10.0], [100.0, 1000.0]],
        ], dtype=np.float32)
        out = Tensor(a_np) @ Tensor(b_np)
        assert out.shape == (2, 2, 2)
        np.testing.assert_allclose(out.numpy(), np.matmul(a_np, b_np), rtol=1e-6)

    def test_matmul_broadcast_mismatch_raises(self):
        a = Tensor(np.zeros((2, 3, 4), dtype=np.float32))
        b = Tensor(np.zeros((5, 4, 6), dtype=np.float32))
        with pytest.raises(ValueError, match='cannot dot'):
            a @ b

    def test_qr_matches_tinygrad_probe(self):
        cases = [
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
            np.array([
                [[1.0, 2.0], [3.0, 4.0]],
                [[2.0, 0.0], [0.0, 2.0]],
            ], dtype=np.float32),
            np.array([
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[2.0, 1.0], [0.0, 3.0], [4.0, 5.0]],
            ], dtype=np.float32),
            np.array([
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[2.0, 1.0, 0.0], [0.0, 3.0, 4.0]],
            ], dtype=np.float32),
        ]
        for arr in cases:
            q, r = Tensor(arr).qr()
            assert q.shape == arr.shape[:-2] + (arr.shape[-2], arr.shape[-2])
            assert r.shape == arr.shape
            q_np, r_np = q.numpy(), r.numpy()
            assert not np.isnan(q_np).any()
            assert not np.isnan(r_np).any()
            np.testing.assert_allclose(np.matmul(q_np, r_np), arr, rtol=1e-4, atol=1e-4)

    def test_qr_zero_column_and_int_promote_match_tinygrad_probe(self):
        arr = np.array([[0.0, 1.0], [0.0, 2.0]], dtype=np.float32)
        q, r = Tensor(arr).qr()
        np.testing.assert_allclose(q.numpy() @ r.numpy(), arr, rtol=1e-4, atol=1e-4)
        assert not np.isnan(q.numpy()).any()
        assert not np.isnan(r.numpy()).any()

        qi, ri = Tensor(np.array([[1, 2], [3, 4]], dtype=np.int32)).qr()
        assert qi.dtype == 'float32'
        assert ri.dtype == 'float32'
        np.testing.assert_allclose(qi.numpy() @ ri.numpy(), [[1, 2], [3, 4]], rtol=1e-4, atol=1e-4)

    def test_qr_reduced_and_r_modes_match_numpy_shapes(self):
        cases = [
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
            np.array([
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[2.0, 1.0], [0.0, 3.0], [4.0, 5.0]],
            ], dtype=np.float32),
        ]
        for arr in cases:
            q, r = Tensor(arr).qr(mode='reduced')
            nq, nr = np.linalg.qr(arr, mode='reduced')
            assert q.shape == nq.shape
            assert r.shape == nr.shape
            np.testing.assert_allclose(np.matmul(q.numpy(), r.numpy()), arr, rtol=1e-4, atol=1e-4)

            r_only = Tensor(arr).qr(mode='r')
            assert r_only.shape == nr.shape

        with pytest.raises(ValueError, match='qr mode'):
            Tensor(cases[0]).qr(mode='raw')

    def test_triangular_solve_matches_numpy_torch_probe(self):
        torch = pytest.importorskip('torch')
        lower = np.array([[2.0, 0.0, 0.0], [1.0, 3.0, 0.0], [-2.0, 0.5, 4.0]], dtype=np.float32)
        upper = np.array([[2.0, -1.0, 0.5], [0.0, 3.0, 2.0], [0.0, 0.0, 4.0]], dtype=np.float32)
        b_vec = np.array([2.0, 7.0, 9.0], dtype=np.float32)
        b_mat = np.array([[2.0, 1.0], [7.0, 2.0], [9.0, 3.0]], dtype=np.float32)
        lower_batched = np.stack([lower, lower + np.eye(3, dtype=np.float32)], axis=0)
        b_batched = np.stack([b_mat, b_mat + 1.0], axis=0)
        b_broadcast_vec = np.array([2.0, 7.0, 9.0], dtype=np.float32)
        cases = [
            (lower, b_vec, False, False, False),
            (lower, b_mat, False, False, False),
            (upper, b_mat, True, False, False),
            (lower, b_mat, False, True, False),
            (upper, b_mat, True, True, False),
            (lower + np.diag([3.0, 4.0, 5.0]).astype(np.float32), b_mat, False, False, True),
            (lower_batched, b_batched, False, False, False),
            (lower_batched, b_broadcast_vec, False, False, False),
        ]
        for a, b, upper_flag, transpose_a, unit_diagonal in cases:
            eff_a = np.swapaxes(a, -1, -2) if transpose_a else a
            eff_upper = (not upper_flag) if transpose_a else upper_flag
            np_a = eff_a.copy()
            if unit_diagonal:
                diag = np.arange(np_a.shape[-1])
                np_a[..., diag, diag] = 1.0
            expected = np.linalg.solve(np_a, b)

            got = Tensor(a).triangular_solve(
                Tensor(b),
                upper=upper_flag,
                transpose_a=transpose_a,
                unit_diagonal=unit_diagonal,
            )
            assert got.shape == expected.shape
            np.testing.assert_allclose(got.numpy(), expected, rtol=1e-5, atol=1e-5)

            if b.ndim >= 2:
                torch_expected = torch.linalg.solve_triangular(
                    torch.tensor(eff_a),
                    torch.tensor(b),
                    upper=eff_upper,
                    left=True,
                    unitriangular=unit_diagonal,
                ).numpy()
                np.testing.assert_allclose(got.numpy(), torch_expected, rtol=1e-5, atol=1e-5)

    def test_triangular_solve_edge_cases(self):
        a1 = np.array([[4.0]], dtype=np.float32)
        b1 = np.array([8.0], dtype=np.float32)
        np.testing.assert_allclose(Tensor(a1).triangular_solve(Tensor(b1)).numpy(), [2.0], rtol=1e-6)

        a_int = np.array([[2, 0], [4, 2]], dtype=np.int32)
        b_int = np.array([2, 8], dtype=np.int32)
        x_int = Tensor(a_int).triangular_solve(Tensor(b_int))
        assert x_int.dtype == 'float32'
        np.testing.assert_allclose(x_int.numpy(), np.linalg.solve(a_int.astype(np.float32), b_int), rtol=1e-6)

        a64 = np.array([[2.0, 0.0], [1.0, 4.0]], dtype=np.float64)
        b64 = np.array([[2.0], [9.0]], dtype=np.float64)
        x64 = Tensor(a64, dtype='float64').triangular_solve(Tensor(b64, dtype='float64'))
        assert x64.dtype == 'float64'
        np.testing.assert_allclose(x64.numpy(), np.linalg.solve(a64, b64), rtol=1e-12, atol=1e-12)

        with pytest.raises(ValueError, match='cannot triangular_solve'):
            Tensor(np.eye(2, dtype=np.float32)).triangular_solve(Tensor(np.ones((3,), dtype=np.float32)))

        singular = Tensor(np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32))
        out = singular.triangular_solve(Tensor(np.array([1.0, 2.0], dtype=np.float32))).numpy()
        assert np.isinf(out).any() or np.isnan(out).any()

    def test_cholesky_matches_numpy_torch_probe(self):
        torch = pytest.importorskip('torch')
        cases = [
            np.array([[4.0]], dtype=np.float32),
            np.array([[4.0, 2.0], [2.0, 5.0]], dtype=np.float32),
            np.array([[6.0, 2.0, 1.0], [2.0, 5.0, 2.0], [1.0, 2.0, 4.0]], dtype=np.float32),
            np.eye(4, dtype=np.float32) * 4.0,
            np.array([
                [[4.0, 2.0], [2.0, 5.0]],
                [[9.0, 3.0], [3.0, 2.0]],
            ], dtype=np.float32),
        ]
        for a in cases:
            l = Tensor(a).cholesky()
            expected = np.linalg.cholesky(a)
            torch_expected = torch.linalg.cholesky(torch.tensor(a)).numpy()
            np.testing.assert_allclose(l.numpy(), expected, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(l.numpy(), torch_expected, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(l.numpy() @ np.swapaxes(l.numpy(), -1, -2), a, rtol=1e-5, atol=1e-5)

        a = np.array([[4.0, 2.0], [2.0, 5.0]], dtype=np.float32)
        u = Tensor(a).cholesky(upper=True)
        np.testing.assert_allclose(u.numpy(), np.swapaxes(np.linalg.cholesky(a), -1, -2), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(np.swapaxes(u.numpy(), -1, -2) @ u.numpy(), a, rtol=1e-5, atol=1e-5)

    def test_cholesky_edge_cases(self):
        a_int = np.array([[4, 2], [2, 5]], dtype=np.int32)
        l_int = Tensor(a_int).cholesky()
        assert l_int.dtype == 'float32'
        np.testing.assert_allclose(l_int.numpy(), np.linalg.cholesky(a_int.astype(np.float32)), rtol=1e-5)

        a64 = np.array([[4.0, 2.0], [2.0, 5.0]], dtype=np.float64)
        l64 = Tensor(a64, dtype='float64').cholesky()
        assert l64.dtype == 'float64'
        np.testing.assert_allclose(l64.numpy(), np.linalg.cholesky(a64), rtol=1e-12, atol=1e-12)

        with pytest.raises(ValueError, match='cannot cholesky'):
            Tensor(np.ones((2, 3), dtype=np.float32)).cholesky()

        bad = Tensor(np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float32)).cholesky().numpy()
        assert np.isnan(bad).any()

    def test_cholesky_solve_matches_torch_probe(self):
        torch = pytest.importorskip('torch')
        a = np.array([[4.0, 2.0], [2.0, 5.0]], dtype=np.float32)
        b = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b_vec = np.array([1.0, 4.0], dtype=np.float32)
        for upper in [False, True]:
            factor = Tensor(a).cholesky(upper=upper)
            got = factor.cholesky_solve(Tensor(b), upper=upper)
            torch_expected = torch.cholesky_solve(
                torch.tensor(b),
                torch.linalg.cholesky(torch.tensor(a), upper=upper),
                upper=upper,
            ).numpy()
            np.testing.assert_allclose(got.numpy(), torch_expected, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(a @ got.numpy(), b, rtol=1e-5, atol=1e-5)

        ab = np.stack([a, np.array([[9.0, 3.0], [3.0, 2.0]], dtype=np.float32)], axis=0)
        bb_vec = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
        for upper in [False, True]:
            factor = Tensor(ab).cholesky(upper=upper)
            got = factor.cholesky_solve(Tensor(bb_vec), upper=upper)
            torch_expected = torch.cholesky_solve(
                torch.tensor(bb_vec).unsqueeze(-1),
                torch.linalg.cholesky(torch.tensor(ab), upper=upper),
                upper=upper,
            ).squeeze(-1).numpy()
            np.testing.assert_allclose(got.numpy(), torch_expected, rtol=1e-5, atol=1e-5)

            got_broadcast = factor.cholesky_solve(Tensor(b_vec), upper=upper)
            torch_broadcast = torch.cholesky_solve(
                torch.tensor(b_vec).reshape(1, 2, 1).expand(2, 2, 1),
                torch.linalg.cholesky(torch.tensor(ab), upper=upper),
                upper=upper,
            ).squeeze(-1).numpy()
            np.testing.assert_allclose(got_broadcast.numpy(), torch_broadcast, rtol=1e-5, atol=1e-5)

    def test_solve_matches_numpy_torch_probe(self):
        torch = pytest.importorskip('torch')
        a = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float32)
        b_vec = np.array([1.0, 4.0], dtype=np.float32)
        b_mat = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        got_vec = Tensor(a).solve(Tensor(b_vec))
        np.testing.assert_allclose(got_vec.numpy(), np.linalg.solve(a, b_vec), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            got_vec.numpy(),
            torch.linalg.solve(torch.tensor(a), torch.tensor(b_vec)).numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

        got_mat = Tensor(a).solve(Tensor(b_mat))
        np.testing.assert_allclose(got_mat.numpy(), np.linalg.solve(a, b_mat), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            got_mat.numpy(),
            torch.linalg.solve(torch.tensor(a), torch.tensor(b_mat)).numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

        pivot_a = np.array([[0.0, 2.0], [1.0, 3.0]], dtype=np.float32)
        pivot_b_vec = np.array([4.0, 5.0], dtype=np.float32)
        pivot_b_mat = np.array([[4.0, 1.0], [5.0, 2.0]], dtype=np.float32)
        got_pivot_vec = Tensor(pivot_a).solve(Tensor(pivot_b_vec))
        np.testing.assert_allclose(got_pivot_vec.numpy(), np.linalg.solve(pivot_a, pivot_b_vec), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            got_pivot_vec.numpy(),
            torch.linalg.solve(torch.tensor(pivot_a), torch.tensor(pivot_b_vec)).numpy(),
            rtol=1e-5,
            atol=1e-5,
        )
        got_pivot_mat = Tensor(pivot_a).solve(Tensor(pivot_b_mat))
        np.testing.assert_allclose(got_pivot_mat.numpy(), np.linalg.solve(pivot_a, pivot_b_mat), rtol=1e-5, atol=1e-5)

        pivot3 = np.array([[0.0, 2.0, 1.0], [1.0, 0.0, 3.0], [4.0, 1.0, 8.0]], dtype=np.float32)
        pivot3_b = np.array([3.0, 4.0, 13.0], dtype=np.float32)
        got_pivot3 = Tensor(pivot3).solve(Tensor(pivot3_b))
        np.testing.assert_allclose(got_pivot3.numpy(), np.linalg.solve(pivot3, pivot3_b), rtol=1e-5, atol=1e-5)

        ab = np.stack([a, a + np.eye(2, dtype=np.float32)], axis=0)
        bb = np.stack([b_mat, b_mat + 1.0], axis=0)
        got_batch = Tensor(ab).solve(Tensor(bb))
        expected_batch = np.stack([np.linalg.solve(ab[i], bb[i]) for i in range(2)])
        np.testing.assert_allclose(got_batch.numpy(), expected_batch, rtol=1e-5, atol=1e-5)

        bb_vec = np.stack([b_vec, np.array([2.0, 5.0], dtype=np.float32)], axis=0)
        got_batch_vec = Tensor(ab).solve(Tensor(bb_vec))
        expected_batch_vec = np.stack([np.linalg.solve(ab[i], bb_vec[i]) for i in range(2)])
        np.testing.assert_allclose(got_batch_vec.numpy(), expected_batch_vec, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            got_batch_vec.numpy(),
            torch.linalg.solve(torch.tensor(ab), torch.tensor(bb_vec)).numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

        pivot_ab = np.stack([pivot_a, np.array([[3.0, 1.0], [0.0, 2.0]], dtype=np.float32)], axis=0)
        pivot_bb_vec = np.stack([pivot_b_vec, np.array([7.0, 4.0], dtype=np.float32)], axis=0)
        got_pivot_batch = Tensor(pivot_ab).solve(Tensor(pivot_bb_vec))
        expected_pivot_batch = np.stack([np.linalg.solve(pivot_ab[i], pivot_bb_vec[i]) for i in range(2)])
        np.testing.assert_allclose(got_pivot_batch.numpy(), expected_pivot_batch, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            got_pivot_batch.numpy(),
            torch.linalg.solve(torch.tensor(pivot_ab), torch.tensor(pivot_bb_vec)).numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

        got_broadcast_vec = Tensor(ab).solve(Tensor(b_vec))
        expected_broadcast_vec = np.stack([np.linalg.solve(ab[i], b_vec) for i in range(2)])
        np.testing.assert_allclose(got_broadcast_vec.numpy(), expected_broadcast_vec, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            got_broadcast_vec.numpy(),
            torch.linalg.solve(torch.tensor(ab), torch.tensor(b_vec)).numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

        got_singleton_matrix = Tensor(ab).solve(Tensor(b_mat.reshape(1, 2, 2)))
        expected_singleton_matrix = np.stack([np.linalg.solve(ab[i], b_mat) for i in range(2)])
        np.testing.assert_allclose(got_singleton_matrix.numpy(), expected_singleton_matrix, rtol=1e-5, atol=1e-5)

        a64 = np.array([[3.0, 1.0], [1.0, 2.0]], dtype=np.float64)
        b64 = np.array([[4.0], [5.0]], dtype=np.float64)
        got64 = Tensor(a64).solve(Tensor(b64))
        assert got64.dtype == 'float64'
        np.testing.assert_allclose(got64.numpy(), np.linalg.solve(a64, b64), rtol=1e-10, atol=1e-10)

        with pytest.raises(ValueError, match='cannot solve'):
            Tensor(np.ones((2, 3), dtype=np.float32)).solve(Tensor(np.ones((2,), dtype=np.float32)))

    def test_lstsq_matches_numpy_torch_probe(self):
        torch = pytest.importorskip('torch')
        def torch_lstsq(a, b):
            # Use the SVD-based driver so rank-deficient cases check the same
            # minimum-norm policy as numpy.linalg.lstsq instead of a QR-driver
            # default that can vary across LAPACK builds.
            return torch.linalg.lstsq(torch.tensor(a), torch.tensor(b), driver='gelsd').solution.numpy()

        a = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]], dtype=np.float32)
        b_vec = np.array([1.0, 2.0, 2.5], dtype=np.float32)
        b_mat = np.array([[1.0, 0.5], [2.0, 1.0], [2.5, 1.5]], dtype=np.float32)

        got_vec = Tensor(a).lstsq(Tensor(b_vec))
        np.testing.assert_allclose(got_vec.numpy(), np.linalg.lstsq(a, b_vec, rcond=None)[0], rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(
            got_vec.numpy(),
            torch_lstsq(a, b_vec),
            rtol=2e-5,
            atol=2e-5,
        )

        got_mat = Tensor(a).lstsq(Tensor(b_mat))
        np.testing.assert_allclose(got_mat.numpy(), np.linalg.lstsq(a, b_mat, rcond=None)[0], rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(
            got_mat.numpy(),
            torch_lstsq(a, b_mat),
            rtol=2e-5,
            atol=2e-5,
        )

        ab = np.stack([a, np.array([[1.0, 0.0], [1.0, 1.5], [1.0, 3.0]], dtype=np.float32)], axis=0)
        bb = np.stack([b_mat, b_mat + 0.25], axis=0)
        got_batch = Tensor(ab).lstsq(Tensor(bb))
        expected_batch = np.stack([np.linalg.lstsq(ab[i], bb[i], rcond=None)[0] for i in range(2)])
        np.testing.assert_allclose(got_batch.numpy(), expected_batch, rtol=2e-5, atol=2e-5)

        bb_vec = np.stack([b_vec, np.array([1.25, 2.25, 2.75], dtype=np.float32)], axis=0)
        got_batch_vec = Tensor(ab).lstsq(Tensor(bb_vec))
        expected_batch_vec = np.stack([np.linalg.lstsq(ab[i], bb_vec[i], rcond=None)[0] for i in range(2)])
        np.testing.assert_allclose(got_batch_vec.numpy(), expected_batch_vec, rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(
            got_batch_vec.numpy(),
            torch_lstsq(ab, bb_vec),
            rtol=2e-5,
            atol=2e-5,
        )

        got_broadcast_vec = Tensor(ab).lstsq(Tensor(b_vec))
        expected_broadcast_vec = np.stack([np.linalg.lstsq(ab[i], b_vec, rcond=None)[0] for i in range(2)])
        np.testing.assert_allclose(got_broadcast_vec.numpy(), expected_broadcast_vec, rtol=2e-5, atol=2e-5)

        a64 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]], dtype=np.float64)
        b64 = np.array([1.0, 2.0, 4.0], dtype=np.float64)
        got64 = Tensor(a64).lstsq(Tensor(b64))
        assert got64.dtype == 'float64'
        np.testing.assert_allclose(got64.numpy(), np.linalg.lstsq(a64, b64, rcond=None)[0], rtol=1e-10, atol=1e-10)

        wide = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 1.0]], dtype=np.float32)
        wide_b_vec = np.array([1.0, 2.0], dtype=np.float32)
        wide_b_mat = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
        got_wide_vec = Tensor(wide).lstsq(Tensor(wide_b_vec))
        np.testing.assert_allclose(
            got_wide_vec.numpy(),
            np.linalg.lstsq(wide, wide_b_vec, rcond=None)[0],
            rtol=3e-5,
            atol=3e-5,
        )
        np.testing.assert_allclose(
            got_wide_vec.numpy(),
            torch_lstsq(wide, wide_b_vec),
            rtol=3e-5,
            atol=3e-5,
        )
        got_wide_mat = Tensor(wide).lstsq(Tensor(wide_b_mat))
        np.testing.assert_allclose(
            got_wide_mat.numpy(),
            np.linalg.lstsq(wide, wide_b_mat, rcond=None)[0],
            rtol=4e-5,
            atol=4e-5,
        )
        np.testing.assert_allclose(
            got_wide_mat.numpy(),
            torch_lstsq(wide, wide_b_mat),
            rtol=4e-5,
            atol=4e-5,
        )

        rankdef_cases = [
            (
                np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32),
                np.array([3.0, 6.0], dtype=np.float32),
            ),
            (
                np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32),
                np.array([1.0, 2.0, 3.0], dtype=np.float32),
            ),
            (
                np.array([[1.0, 1.0, 0.0], [2.0, 2.0, 0.0]], dtype=np.float32),
                np.array([3.0, 6.0], dtype=np.float32),
            ),
            (
                np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32),
                np.array([[3.0, 1.0], [6.0, 2.0]], dtype=np.float32),
            ),
        ]
        for rd_a, rd_b in rankdef_cases:
            got = Tensor(rd_a).lstsq(Tensor(rd_b)).numpy()
            np.testing.assert_allclose(
                got,
                np.linalg.lstsq(rd_a, rd_b, rcond=None)[0],
                rtol=2e-4,
                atol=2e-4,
            )
            np.testing.assert_allclose(
                got,
                torch_lstsq(rd_a, rd_b),
                rtol=2e-4,
                atol=2e-4,
            )

        with pytest.raises(ValueError, match='cannot lstsq'):
            Tensor(np.ones((2, 3), dtype=np.float32)).lstsq(Tensor(np.ones((3,), dtype=np.float32)))

    def test_cross_entropy_sparse_targets(self):
        logits = Tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        target = Tensor([0.0, 2.0])
        loss = logits.cross_entropy(target)
        assert loss.shape == ()
        np.testing.assert_allclose(loss.numpy(), np.log(3.0), rtol=1e-6)

    def test_cross_entropy_dense_targets(self):
        logits = Tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        target = Tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        loss = logits.cross_entropy(target)
        assert loss.shape == ()
        np.testing.assert_allclose(loss.numpy(), np.log(3.0), rtol=1e-6)

    def test_cross_entropy_sparse_targets_non_last_axis(self):
        logits = Tensor(np.zeros((2, 3, 2), dtype=np.float32))
        target = Tensor(np.array([[0.0, 2.0], [1.0, 0.0]], dtype=np.float32))
        loss = logits.cross_entropy(target, axis=-2)
        assert loss.shape == ()
        np.testing.assert_allclose(loss.numpy(), np.log(3.0), rtol=1e-6)

    def test_cross_entropy_default_matches_tinygrad_class_axis(self):
        logits = Tensor(np.zeros((2, 3, 2), dtype=np.float32))
        target = Tensor(np.array([[0.0, 2.0], [1.0, 0.0]], dtype=np.float32))
        loss = logits.cross_entropy(target)
        assert loss.shape == ()
        np.testing.assert_allclose(loss.numpy(), np.log(3.0), rtol=1e-6)

    def test_cross_entropy_dense_targets_non_last_axis(self):
        logits = Tensor(np.zeros((2, 3, 2), dtype=np.float32))
        target = Tensor(np.array([
            [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
        ], dtype=np.float32))
        loss = logits.cross_entropy(target, axis=1)
        assert loss.shape == ()
        np.testing.assert_allclose(loss.numpy(), np.log(3.0), rtol=1e-6)

    def test_cross_entropy_shape_mismatch_raises(self):
        logits = Tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        target = Tensor([[1.0, 0.0], [0.0, 1.0]])
        with pytest.raises(ValueError, match='shape mismatch'):
            logits.cross_entropy(target)


class TestAutograd:
    def test_grad_mul_sum(self):
        x = Tensor([1, 2, 3, 4], requires_grad=True)
        loss = (x * x).sum()
        loss.backward()
        assert x.grad is not None
        np.testing.assert_allclose(x.grad.numpy(), [2, 4, 6, 8])

    def test_grad_neg_sum(self):
        x = Tensor([1, 2, 3], requires_grad=True)
        loss = (-x).sum()
        loss.backward()
        np.testing.assert_allclose(x.grad.numpy(), [-1, -1, -1])

    def test_grad_max_reduce(self):
        x = Tensor([[1.0, 3.0, 2.0], [4.0, 5.0, 6.0]], requires_grad=True)
        loss = x.max(axis=1).sum()
        loss.backward()
        assert x.grad is not None
        np.testing.assert_allclose(x.grad.numpy(), [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


class TestDevice:
    def test_device_lookup(self):
        assert Device['cpu'] == 'CPU'
        assert Device['CUDA'] == 'CUDA'

    def test_requires_grad_inplace(self):
        a = Tensor([1.0])
        assert a.requires_grad_(True) is a
        assert a.requires_grad is True

    def test_to_device_roundtrip(self):
        a = Tensor([1.0, 2.0, 3.0], dtype='float64')
        b = (a + 1).to('cuda')
        assert b.device == 'CUDA'

        c = b.to('cpu')
        assert c.device == 'CPU'
        if Device.cuda_available():
            np.testing.assert_allclose(c.numpy(), [2.0, 3.0, 4.0])
        else:
            with pytest.raises(RuntimeError, match='poly_realize_tensors'):
                c.numpy()

    def test_to_cuda_runtime_behavior(self):
        a = Tensor([1.0, 2.0, 3.0])
        b = (a * 2).to('cuda')
        if Device.cuda_available():
            np.testing.assert_allclose(b.numpy(), [2.0, 4.0, 6.0])
        else:
            with pytest.raises(RuntimeError, match='CUDA'):
                b.numpy()

    def test_assign_rejects_device_mismatch_like_tinygrad(self):
        a = Tensor([1.0], device='cpu')
        v = Tensor([5.0], device='cpu').to('cuda')
        with pytest.raises(RuntimeError, match='assign device mismatch CPU != CUDA'):
            a.assign(v)

    def test_assign_rejects_dtype_mismatch_like_tinygrad(self):
        a = Tensor([1.0], dtype='float32')
        v = Tensor([5.0], dtype='float64')
        with pytest.raises(RuntimeError, match='assign dtype mismatch float32 != float64'):
            a.assign(v)

    def test_assign_to_same_device_place_keeps_place_target(self):
        a = Tensor([1.0], device='cpu').to('cuda')
        v = Tensor([5.0], device='cpu').to('cuda')
        assert a.assign(v) is a
        assert a.device == 'CUDA'
        assert not a.uop.has_buffer_identity()


class TestRepr:
    def test_repr(self):
        t = Tensor([1, 2, 3])
        assert 'shape=(3,)' in repr(t)
        assert 'float32' in repr(t)

    def test_bool_raises_like_tinygrad(self):
        with pytest.raises(TypeError, match="__bool__ on Tensor is not defined"):
            bool(Tensor([1.0]))

    def test_repr_f64(self):
        t = Tensor([1, 2, 3], dtype='float64')
        assert 'float64' in repr(t)


class TestFloat64:
    """Tests for float64 dtype support."""

    def test_numpy_dtype_is_preserved_like_tinygrad(self):
        f64 = Tensor(np.array([1.0, 2.0], dtype=np.float64))
        assert f64.dtype == 'float64'
        assert f64.numpy().dtype == np.float64

        i64 = Tensor(np.array([1, 2], dtype=np.int64))
        assert i64.dtype == 'int64'
        assert i64.numpy().dtype == np.int64

        default_list = Tensor([1.0, 2.0])
        assert default_list.dtype == 'float32'

    def test_creation_from_list(self):
        t = Tensor([1.0, 2.0, 3.0], dtype='float64')
        assert t.dtype == 'float64'
        assert t.shape == (3,)
        assert t.numpy().dtype == np.float64
        np.testing.assert_allclose(t.numpy(), [1, 2, 3])

    def test_creation_2d(self):
        t = Tensor([[1, 2], [3, 4]], dtype='float64')
        assert t.dtype == 'float64'
        assert t.shape == (2, 2)
        assert t.numpy().dtype == np.float64
        np.testing.assert_allclose(t.numpy(), [[1, 2], [3, 4]])

    def test_zeros_f64(self):
        t = Tensor.zeros(4, dtype='float64')
        assert t.dtype == 'float64'
        assert t.numpy().dtype == np.float64
        np.testing.assert_allclose(t.numpy(), [0, 0, 0, 0])

    def test_ones_f64(self):
        t = Tensor.ones(3, dtype='float64')
        assert t.dtype == 'float64'
        np.testing.assert_allclose(t.numpy(), [1, 1, 1])

    def test_full_f64(self):
        t = Tensor.full((2, 3), 7.0, dtype='float64')
        assert t.dtype == 'float64'
        np.testing.assert_allclose(t.numpy(), np.full((2, 3), 7.0))

    def test_eye_f64(self):
        t = Tensor.eye(3, dtype='float64')
        assert t.dtype == 'float64'
        np.testing.assert_allclose(t.numpy(), np.eye(3))

    def test_arange_f64(self):
        t = Tensor.arange(5, dtype='float64')
        assert t.dtype == 'float64'
        np.testing.assert_allclose(t.numpy(), [0, 1, 2, 3, 4])

    def test_add_f64(self):
        a = Tensor([1.0, 2.0, 3.0], dtype='float64')
        b = Tensor([4.0, 5.0, 6.0], dtype='float64')
        c = (a + b).numpy()
        assert c.dtype == np.float64
        np.testing.assert_allclose(c, [5, 7, 9])

    def test_mul_f64(self):
        a = Tensor([1.0, 2.0, 3.0], dtype='float64')
        b = Tensor([4.0, 5.0, 6.0], dtype='float64')
        c = (a * b).numpy()
        assert c.dtype == np.float64
        np.testing.assert_allclose(c, [4, 10, 18])

    def test_neg_f64(self):
        a = Tensor([1.0, -2.0, 3.0], dtype='float64')
        c = (-a).numpy()
        assert c.dtype == np.float64
        np.testing.assert_allclose(c, [-1, 2, -3])

    def test_scalar_add_f64(self):
        a = Tensor([1.0, 2.0, 3.0], dtype='float64')
        c = (a + 10.0).numpy()
        assert c.dtype == np.float64
        np.testing.assert_allclose(c, [11, 12, 13])

    def test_sum_f64(self):
        a = Tensor([1.0, 2.0, 3.0, 4.0], dtype='float64')
        s = a.sum().item()
        np.testing.assert_allclose(s, 10.0)

    def test_exp_f64(self):
        a = Tensor([0.0, 1.0, 2.0], dtype='float64')
        c = a.exp().numpy()
        assert c.dtype == np.float64
        np.testing.assert_allclose(c, np.exp([0, 1, 2]), rtol=1e-10)

    def test_sqrt_f64(self):
        a = Tensor([1.0, 4.0, 9.0], dtype='float64')
        c = a.sqrt().numpy()
        assert c.dtype == np.float64
        np.testing.assert_allclose(c, [1, 2, 3], rtol=1e-14)

    def test_chain_f64(self):
        a = Tensor([1.0, 2.0, 3.0], dtype='float64')
        b = Tensor([4.0, 5.0, 6.0], dtype='float64')
        c = ((a + b) * a - b).numpy()
        assert c.dtype == np.float64
        np.testing.assert_allclose(c, [(1+4)*1-4, (2+5)*2-5, (3+6)*3-6])

    def test_reshape_f64(self):
        a = Tensor([1, 2, 3, 4, 5, 6], dtype='float64')
        b = a.reshape(2, 3).numpy()
        assert b.dtype == np.float64
        np.testing.assert_allclose(b, [[1, 2, 3], [4, 5, 6]])

    def test_backward_f64(self):
        a = Tensor([1.0, 2.0, 3.0], dtype='float64', requires_grad=True)
        b = Tensor([4.0, 5.0, 6.0], dtype='float64')
        loss = (a * b).sum()
        loss.backward()
        assert a.grad is not None
        np.testing.assert_allclose(a.grad.numpy(), [4, 5, 6], rtol=1e-14)

    def test_dtype_propagation(self):
        """Ensure dtype propagates through ops."""
        a = Tensor([1.0, 2.0], dtype='float64')
        b = a + 1.0
        assert b.dtype == 'float64'
        c = b * 2.0
        assert c.dtype == 'float64'
        d = c.exp()
        assert d.dtype == 'float64'

    def test_default_is_f32(self):
        """Ensure default dtype is still float32."""
        a = Tensor([1.0, 2.0])
        assert a.dtype == 'float32'
        assert a.numpy().dtype == np.float32


class TestMaterializationParity:
    def test_realize_retargets_live_tensors_sharing_lazy_uop(self):
        a = Tensor([1.0]).realize()
        x1 = a + 1
        x2 = a + 1
        assert x1.uop == x2.uop

        x1.realize()

        assert x1.uop == x2.uop
        assert x1.uop.buffer == x2.uop.buffer
        np.testing.assert_allclose(x2.numpy(), [2.0])

    def test_realize_rewrites_downstream_live_graph_to_snapshot(self):
        a = Tensor([1.0]).realize()
        x = a + 1
        y = x + 2

        x.realize()
        a.assign(Tensor([10.0])).realize()

        np.testing.assert_allclose(x.numpy(), [2.0])
        np.testing.assert_allclose(y.numpy(), [4.0])

    def test_separate_realizes_do_not_alias_assign(self):
        a = Tensor([1.0]).realize()
        y1 = (a + 1).realize()
        y2 = (a + 1).realize()

        assert y1.uop.buffer != y2.uop.buffer

        y2.assign(Tensor([5.0])).realize()
        np.testing.assert_allclose(y1.numpy(), [2.0])
        np.testing.assert_allclose(y2.numpy(), [5.0])

    def test_assign_realized_targets_reuse_current_buffer(self):
        a = Tensor([1.0]).realize()
        a_buf = a.uop.buffer

        # assign() follows tinygrad by creating an effect graph first; realizing
        # that effect writes the existing target storage and returns to the same
        # current buffer root.
        a.assign(Tensor([5.0]))
        assert not a.uop.has_buffer_identity()
        a.realize()
        assert a.uop.buffer == a_buf
        np.testing.assert_allclose(a.numpy(), [5.0])

        x = (Tensor([1.0]) + 1).realize()
        x_buf = x.uop.buffer
        x.assign(Tensor([9.0])).realize()
        assert x.uop.buffer == x_buf
        np.testing.assert_allclose(x.numpy(), [9.0])

    def test_to_keeps_separate_realized_same_logical_occurrences(self):
        a = Tensor([1.0]).realize()
        x1 = (a + 1).realize()
        x2 = (a + 1).realize()
        assert x1.uop.buffer != x2.uop.buffer

        # x1/x2 preserve the same exportable logical expression, but .to()
        # must carry the occurrence-specific realized source into placement.
        x1_cuda = x1.to('cuda')
        x2_cuda = x2.to('cuda')
        assert x1_cuda.uop.buffer == x1.uop.buffer
        assert x2_cuda.uop.buffer == x2.uop.buffer
        assert x1_cuda.uop.buffer != x2_cuda.uop.buffer

        y1 = x1_cuda + 1
        y2 = x2_cuda + 1
        assert y1.uop != y2.uop

    def test_nested_to_keeps_realized_current_and_export_logical_separate(self):
        x = (Tensor([1.0]) + 1).realize()
        x_cuda = x.to('cuda')
        x_cpu = x_cuda.to('cpu')

        # Polygrad keeps .to() out of the portable logical graph, but the
        # current/physical root must stay tied to the realized buffer selected
        # by this tensor occurrence so later ops can physicalize the copy chain.
        assert x_cpu.uop == x.uop
        assert x_cpu.uop_physical == x.uop
        assert x_cpu.uop_logical == x.uop_logical

        y = x_cpu + 1
        assert y.device == 'CPU'
        assert y.uop != x_cpu.uop
