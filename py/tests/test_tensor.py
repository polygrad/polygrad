"""Tests for the polygrad Python Tensor class."""

import numpy as np
import pytest

from polygrad import Device, Jit, JitError, Tensor, Variable, compile as pg_compile, jit


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

    def test_empty_creates_unrealized_buffer_placeholder(self):
        t = Tensor.empty((2, 3))
        assert t.shape == (2, 3)
        assert t.uop.has_buffer_identity()
        assert not t.uop.is_realized

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
