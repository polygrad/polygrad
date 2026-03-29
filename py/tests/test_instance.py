"""Tests for the PolyInstance Python wrapper."""

import ctypes
import os
import numpy as np
import pytest
from polygrad.instance import Instance, OPTIM_SGD, OPTIM_ADAM, OPTIM_ADAMW


class TestMLPCreate:
    def test_create_simple(self):
        inst = Instance.mlp({
            'layers': [2, 4, 1], 'activation': 'relu',
            'bias': True, 'loss': 'mse', 'batch_size': 1, 'seed': 42
        })
        assert inst.param_count == 4
        assert inst.param_name(0) == 'layers.0.weight'
        assert inst.param_shape(0) == (4, 2)
        inst.free()

    def test_create_no_bias(self):
        inst = Instance.mlp({
            'layers': [3, 2], 'activation': 'none',
            'bias': False, 'loss': 'none', 'batch_size': 1, 'seed': 42
        })
        assert inst.param_count == 1
        assert inst.param_name(0) == 'layers.0.weight'
        assert inst.param_shape(0) == (2, 3)
        inst.free()

    def test_deterministic_init(self):
        spec = {
            'layers': [2, 4, 1], 'activation': 'relu',
            'bias': True, 'loss': 'mse', 'batch_size': 1, 'seed': 42
        }
        i1 = Instance.mlp(spec)
        i2 = Instance.mlp(spec)
        np.testing.assert_array_equal(i1.param_data(0), i2.param_data(0))
        i1.free()
        i2.free()

    def test_null_spec(self):
        with pytest.raises(RuntimeError):
            Instance.mlp('{}')


class TestForward:
    def test_forward_produces_output(self):
        inst = Instance.mlp({
            'layers': [2, 4, 1], 'activation': 'relu',
            'bias': True, 'loss': 'mse', 'batch_size': 1, 'seed': 42
        })
        outputs = inst.forward(x=np.array([1.0, 2.0], dtype=np.float32))
        assert 'output' in outputs
        assert np.isfinite(outputs['output'][0])
        inst.free()

    def test_forward_deterministic(self):
        inst = Instance.mlp({
            'layers': [2, 4, 1], 'activation': 'relu',
            'bias': True, 'loss': 'mse', 'batch_size': 1, 'seed': 42
        })
        out1 = inst.forward(x=np.array([1.0, 2.0], dtype=np.float32))
        out2 = inst.forward(x=np.array([1.0, 2.0], dtype=np.float32))
        np.testing.assert_array_equal(out1['output'], out2['output'])
        inst.free()


class TestTrain:
    def test_train_sgd(self):
        inst = Instance.mlp({
            'layers': [2, 1], 'activation': 'none',
            'bias': True, 'loss': 'mse', 'batch_size': 1, 'seed': 42
        })
        inst.set_optimizer(OPTIM_SGD, lr=0.05)
        x = np.array([1.0, 2.0], dtype=np.float32)
        y = np.array([5.0], dtype=np.float32)

        losses = []
        for _ in range(50):
            loss = inst.train_step(x=x, y=y)
            assert np.isfinite(loss)
            losses.append(loss)

        assert losses[-1] < losses[0]
        inst.free()

    def test_train_multi_layer(self):
        inst = Instance.mlp({
            'layers': [1, 4, 1], 'activation': 'relu',
            'bias': True, 'loss': 'mse', 'batch_size': 1, 'seed': 42
        })
        inst.set_optimizer(OPTIM_SGD, lr=0.01)
        x = np.array([1.0], dtype=np.float32)
        y = np.array([2.0], dtype=np.float32)

        losses = []
        for _ in range(100):
            loss = inst.train_step(x=x, y=y)
            assert np.isfinite(loss)
            losses.append(loss)

        assert losses[-1] < losses[0]
        inst.free()


class TestWeightIO:
    def test_export_import_round_trip(self):
        spec = {
            'layers': [2, 4, 1], 'activation': 'relu',
            'bias': True, 'loss': 'mse', 'batch_size': 1, 'seed': 42
        }
        inst = Instance.mlp(spec)
        original_w = inst.param_data(0).copy()

        # Export
        st_bytes = inst.export_weights()
        assert st_bytes is not None
        assert len(st_bytes) > 0

        # Create fresh instance with different seed
        spec2 = dict(spec, seed=99)
        inst2 = Instance.mlp(spec2)
        different_w = inst2.param_data(0).copy()
        assert not np.array_equal(original_w, different_w)

        # Import weights from first instance
        inst2.import_weights(st_bytes)
        np.testing.assert_array_equal(inst2.param_data(0), original_w)

        inst.free()
        inst2.free()


class TestBufferEnumeration:
    def test_buf_roles(self):
        inst = Instance.mlp({
            'layers': [2, 1], 'activation': 'none',
            'bias': True, 'loss': 'mse', 'batch_size': 1, 'seed': 42
        })
        roles = {}
        for i in range(inst.buf_count):
            name = inst.buf_name(i)
            role = inst.buf_role(i)
            roles[name] = role

        assert 'layers.0.weight' in roles
        assert 'x' in roles
        inst.free()

    def test_find_buf(self):
        inst = Instance.mlp({
            'layers': [2, 1], 'activation': 'none',
            'bias': True, 'loss': 'mse', 'batch_size': 1, 'seed': 42
        })
        idx = inst.find_buf('output')
        assert idx >= 0
        assert inst.buf_name(idx) == 'output'
        assert inst.find_buf('nonexistent') == -1
        inst.free()


class TestParams:
    def test_param_iteration(self):
        inst = Instance.mlp({
            'layers': [2, 4, 1], 'activation': 'relu',
            'bias': True, 'loss': 'mse', 'batch_size': 1, 'seed': 42
        })
        params = list(inst.params())
        assert len(params) == 4
        assert params[0][0] == 'layers.0.weight'
        assert params[0][1] == (4, 2)
        assert params[0][2] is not None
        assert len(params[0][2]) == 8  # 4*2
        inst.free()


class TestTrainBatch:
    """batch_size>1 training (P0 regression coverage)."""

    def test_train_batch2_mse(self):
        inst = Instance.mlp({
            'layers': [2, 3, 2], 'activation': 'relu',
            'bias': False, 'loss': 'mse', 'batch_size': 2, 'seed': 42
        })
        inst.set_optimizer(OPTIM_SGD, lr=0.01)
        x = np.ones((2, 2), dtype=np.float32) * 0.5
        y = np.ones((2, 2), dtype=np.float32) * 0.3
        first = inst.train_step(x=x, y=y)
        for _ in range(49):
            last = inst.train_step(x=x, y=y)
        assert np.isfinite(first) and np.isfinite(last)
        assert last < first
        inst.free()

    def test_train_batch32_cross_entropy(self):
        inst = Instance.mlp({
            'layers': [4, 8, 3], 'activation': 'relu',
            'bias': True, 'loss': 'cross_entropy', 'batch_size': 32, 'seed': 42
        })
        inst.set_optimizer(OPTIM_SGD, lr=0.01)
        x = np.array([[(i % 7) * 0.1] * 4 for i in range(32)], dtype=np.float32)
        y = np.zeros((32, 3), dtype=np.float32)
        for i in range(32):
            y[i, i % 3] = 1.0
        first = inst.train_step(x=x, y=y)
        for _ in range(29):
            last = inst.train_step(x=x, y=y)
        assert np.isfinite(first) and np.isfinite(last)
        assert last < first
        inst.free()


class TestTrainOptimizers:
    """Adam/AdamW optimizer coverage."""

    def test_train_adam(self):
        inst = Instance.mlp({
            'layers': [2, 4, 1], 'activation': 'relu',
            'bias': True, 'loss': 'mse', 'batch_size': 1, 'seed': 42
        })
        inst.set_optimizer(OPTIM_ADAM, lr=0.01)
        x = np.array([1.0, 2.0], dtype=np.float32)
        y = np.array([3.0], dtype=np.float32)
        first = inst.train_step(x=x, y=y)
        for _ in range(49):
            last = inst.train_step(x=x, y=y)
        assert np.isfinite(first) and np.isfinite(last)
        assert last < first
        inst.free()

    def test_train_adamw(self):
        inst = Instance.mlp({
            'layers': [2, 4, 1], 'activation': 'relu',
            'bias': True, 'loss': 'mse', 'batch_size': 1, 'seed': 42
        })
        inst.set_optimizer(OPTIM_ADAMW, lr=0.01, weight_decay=0.01)
        x = np.array([1.0, 2.0], dtype=np.float32)
        y = np.array([3.0], dtype=np.float32)
        first = inst.train_step(x=x, y=y)
        for _ in range(49):
            last = inst.train_step(x=x, y=y)
        assert np.isfinite(first) and np.isfinite(last)
        assert last < first
        inst.free()


class TestSetDevice:
    """set_device + training (CUDA train_step crash regression)."""

    def test_train_with_set_device(self):
        lib = ctypes.CDLL(os.environ.get('POLYGRAD_LIB', 'build/libpolygrad.so'))
        lib.poly_instance_set_device.restype = ctypes.c_int
        lib.poly_instance_set_device.argtypes = [ctypes.c_void_p, ctypes.c_int]

        inst = Instance.mlp({
            'layers': [2, 4, 1], 'activation': 'relu',
            'bias': True, 'loss': 'mse', 'batch_size': 1, 'seed': 42
        })
        rc = lib.poly_instance_set_device(inst._ptr, 0)  # AUTO
        assert rc == 0

        inst.set_optimizer(OPTIM_SGD, lr=0.05)
        x = np.array([1.0, 2.0], dtype=np.float32)
        y = np.array([5.0], dtype=np.float32)
        first = inst.train_step(x=x, y=y)
        for _ in range(9):
            last = inst.train_step(x=x, y=y)
        assert np.isfinite(first) and np.isfinite(last)
        assert last < first
        inst.free()


class TestInstanceTensorParity:
    """Verify Instance and Tensor APIs produce same results."""

    def test_linear_sgd_parity(self):
        from polygrad.tensor import Tensor

        # Instance path
        inst = Instance.mlp({
            'layers': [2, 1], 'activation': 'none',
            'bias': False, 'loss': 'mse', 'batch_size': 1, 'seed': 42
        })
        W_init = inst.param_data(0).copy()  # shape (1, 2)
        inst.set_optimizer(OPTIM_SGD, lr=0.01)
        x_data = np.array([1.0, 2.0], dtype=np.float32)
        y_data = np.array([3.0], dtype=np.float32)
        inst_losses = []
        for _ in range(5):
            loss = inst.train_step(x=x_data, y=y_data)
            inst_losses.append(loss)
        inst.free()

        # Tensor path (manual SGD)
        W = Tensor(W_init.reshape(1, 2), requires_grad=True)
        tensor_losses = []
        for _ in range(5):
            x = Tensor(x_data.reshape(1, 2))
            y = Tensor(y_data.reshape(1, 1))
            pred = x.matmul(W.transpose())
            diff = pred - y
            loss = (diff * diff).sum()
            tensor_losses.append(loss.item())
            loss.backward()
            # Manual SGD: W = W - lr * grad
            W = Tensor((W.numpy() - 0.01 * W.grad.numpy()), requires_grad=True)

        for i in range(5):
            np.testing.assert_allclose(inst_losses[i], tensor_losses[i], rtol=1e-4,
                                       err_msg=f'step {i} loss mismatch')
