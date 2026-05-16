"""Tests for the polygrad Python Tensor class."""

import numpy as np
import pytest

from polygrad import Device, Tensor


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
