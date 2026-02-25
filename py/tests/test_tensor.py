"""Tests for the polygrad Python Tensor class."""

import numpy as np
import pytest

from polygrad import Tensor


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


class TestReduce:
    def test_sum_all(self):
        a = Tensor([1, 2, 3, 4])
        s = a.sum()
        assert s.item() == pytest.approx(10.0)

    def test_sum_axis(self):
        a = Tensor([[1, 2, 3], [4, 5, 6]])
        s = a.reshape(2, 3).sum(axis=1)
        np.testing.assert_allclose(s.numpy(), [6, 15])


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


class TestStubs:
    def test_to_raises(self):
        a = Tensor([1])
        with pytest.raises(AttributeError):
            a.to('cuda')


class TestRepr:
    def test_repr(self):
        t = Tensor([1, 2, 3])
        assert 'shape=(3,)' in repr(t)
        assert 'float32' in repr(t)
