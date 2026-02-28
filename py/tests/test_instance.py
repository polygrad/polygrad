"""Tests for the PolyInstance Python wrapper."""

import numpy as np
import pytest
from polygrad.instance import Instance, OPTIM_SGD, OPTIM_ADAM


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
