"""Tests for polygrad.nn module — layers, optimizers, state dict."""

import math
import numpy as np
import pytest
from polygrad import Instance, Tensor
from polygrad.nn import (
    Linear, LayerNorm, GroupNorm, RMSNorm, Embedding, Dropout, Conv2d, BatchNorm,
    SGD, Adam, AdamW, OptimizerGroup,
    get_parameters, get_state_dict, load_state_dict,
)


# ── Helpers ──

def approx(a, b, tol=1e-4):
    return np.allclose(a, b, atol=tol)


# ── Linear ──

class TestLinear:
    def test_forward_shape(self):
        m = Linear(3, 2)
        x = Tensor.rand(4, 3)
        y = m(x)
        assert y.shape == (4, 2)

    def test_forward_no_bias(self):
        m = Linear(3, 2, bias=False)
        assert m.bias is None
        x = Tensor.rand(1, 3)
        y = m(x)
        assert y.shape == (1, 2)

    def test_weight_initializer_stays_lazy_like_tinygrad(self):
        m = Linear(2, 3)
        assert not m.weight.uop.has_buffer_identity()
        assert m.weight.requires_grad
        assert not m.bias.uop.has_buffer_identity()
        assert m.bias.requires_grad

    def test_backward(self):
        m = Linear(2, 1)
        x = Tensor([[1.0, 2.0]])
        loss = (m(x) - Tensor([[1.0]])).square().sum()
        loss.backward()
        assert m.weight.grad is not None
        assert m.bias.grad is not None
        assert m.weight.grad.shape == (1, 2)
        assert m.bias.grad.shape == (1,)


# ── LayerNorm ──

class TestLayerNorm:
    def test_forward_shape(self):
        ln = LayerNorm(4)
        x = Tensor.rand(2, 4)
        y = ln(x)
        assert y.shape == (2, 4)

    def test_output_normalized(self):
        ln = LayerNorm(4)
        x = Tensor([[1.0, 2.0, 3.0, 4.0]])
        y = ln(x)
        data = y.numpy()
        # After layer norm, mean ≈ 0, std ≈ 1 (with affine weight=1, bias=0)
        assert abs(data.mean()) < 0.1
        assert abs(data.std() - 1.0) < 0.2

    def test_backward(self):
        ln = LayerNorm(4)
        x = Tensor([[1.0, 2.0, 3.0, 4.0]])
        loss = ln(x).sum()
        loss.backward()
        assert ln.weight.grad is not None
        assert ln.bias.grad is not None


# ── RMSNorm ──

class TestRMSNorm:
    def test_forward_shape(self):
        rn = RMSNorm(4)
        x = Tensor.rand(2, 4)
        y = rn(x)
        assert y.shape == (2, 4)

    def test_backward(self):
        rn = RMSNorm(4)
        x = Tensor([[1.0, 2.0, 3.0, 4.0]])
        loss = rn(x).sum()
        loss.backward()
        assert rn.weight.grad is not None
        assert rn.weight.grad.shape == (4,)


# ── GroupNorm ──

class TestGroupNorm:
    def test_forward_shape(self):
        gn = GroupNorm(2, 4)
        x = Tensor.rand(2, 4, 3, 3)
        y = gn(x)
        assert y.shape == (2, 4, 3, 3)

    def test_backward(self):
        gn = GroupNorm(2, 4)
        x = Tensor.rand(2, 4, 3, 3)
        loss = gn(x).sum()
        loss.backward()
        assert gn.weight.grad is not None
        assert gn.bias.grad is not None


# ── Conv2d ──

class TestConv2d:
    def test_forward_shape(self):
        conv = Conv2d(3, 4, kernel_size=3, stride=2, padding=1)
        x = Tensor.rand(2, 3, 8, 8)
        y = conv(x)
        assert y.shape == (2, 4, 4, 4)

    def test_backward(self):
        conv = Conv2d(3, 2, kernel_size=3, stride=1, padding=1)
        x = Tensor.rand(1, 3, 5, 5)
        loss = conv(x).sum()
        loss.backward()
        assert conv.weight.grad is not None
        assert conv.bias.grad is not None


# ── BatchNorm ──

class TestBatchNorm:
    def test_forward_shape(self):
        bn = BatchNorm(4)
        x = Tensor.rand(2, 4, 3, 3)
        Tensor.training = True
        y = bn(x)
        Tensor.training = False
        assert y.shape == (2, 4, 3, 3)

    def test_running_stats_update(self):
        bn = BatchNorm(4)
        x = Tensor.rand(2, 4, 3, 3)
        before_mean = bn.running_mean.numpy().copy()
        before_var = bn.running_var.numpy().copy()
        Tensor.training = True
        _ = bn(x).realize()
        Tensor.training = False
        after_mean = bn.running_mean.numpy()
        after_var = bn.running_var.numpy()
        assert not np.allclose(before_mean, after_mean)
        assert not np.allclose(before_var, after_var)

    def test_running_var_bias_correction(self):
        bn = BatchNorm(4, momentum=0.1)
        x_np = np.arange(2 * 4 * 3 * 3, dtype=np.float32).reshape(2, 4, 3, 3)
        x = Tensor(x_np)
        Tensor.training = True
        _ = bn(x).realize()
        Tensor.training = False

        # Expected update: rv = 0.9*1 + 0.1*(N/(N-C))*batch_var
        mean = x_np.mean(axis=(0, 2, 3))
        centered = x_np - mean.reshape(1, 4, 1, 1)
        var = (centered * centered).mean(axis=(0, 2, 3))
        n = x_np.size
        c = x_np.shape[1]
        corr = n / (n - c)
        expected = 0.9 * np.ones(4, dtype=np.float32) + 0.1 * corr * var
        assert np.allclose(bn.running_var.numpy(), expected, atol=1e-4)

    def test_backward(self):
        bn = BatchNorm(4)
        x = Tensor.rand(2, 4, 3, 3)
        Tensor.training = True
        loss = bn(x).sum()
        loss.backward()
        Tensor.training = False
        assert bn.weight.grad is not None
        assert bn.bias.grad is not None


# ── Embedding ──

class TestEmbedding:
    def test_forward_shape(self):
        emb = Embedding(10, 4)
        idx = Tensor([0, 3, 7])
        y = emb(idx)
        assert y.shape == (3, 4)

    def test_values(self):
        emb = Embedding(5, 3)
        weight = emb.weight.numpy()
        idx = Tensor([0, 2])
        y = emb(idx).numpy()
        assert approx(y[0], weight[0])
        assert approx(y[1], weight[2])

    def test_rejects_non_integer_indices(self):
        emb = Embedding(5, 3)
        with pytest.raises(TypeError, match='Expected integer dtype'):
            emb(Tensor([0.0, 2.0]))


# ── Dropout ──

class TestDropout:
    def test_eval_passthrough(self):
        Tensor.training = False
        d = Dropout(0.5)
        x = Tensor([1.0, 2.0, 3.0])
        y = d(x)
        assert approx(y.numpy(), x.numpy())

    def test_zero_p_passthrough(self):
        Tensor.training = True
        d = Dropout(0.0)
        x = Tensor([1.0, 2.0, 3.0])
        y = d(x)
        assert approx(y.numpy(), x.numpy())
        Tensor.training = False


# ── SGD ──

class TestSGD:
    def test_step_updates(self):
        m = Linear(2, 1)
        opt = SGD(get_parameters(m), lr=0.1)
        x = Tensor([[1.0, 2.0]])
        loss = (m(x) - Tensor([[1.0]])).square().sum()
        old_w = m.weight.numpy().copy()
        loss.backward()
        opt.step()
        new_w = m.weight.numpy()
        assert not np.allclose(old_w, new_w)

    def test_loss_decreases(self):
        Tensor.manual_seed(42)
        m = Linear(2, 1)
        opt = SGD(get_parameters(m), lr=0.01)
        losses = []
        for _ in range(5):
            opt.zero_grad()
            x = Tensor([[1.0, 2.0], [3.0, 4.0]])
            target = Tensor([[5.0], [11.0]])
            loss = (m(x) - target).square().mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        # Loss should decrease
        assert losses[-1] < losses[0]

    def test_zero_grad(self):
        m = Linear(2, 1)
        opt = SGD(get_parameters(m), lr=0.1)
        x = Tensor([[1.0, 2.0]])
        loss = m(x).sum()
        loss.backward()
        assert m.weight.grad is not None
        opt.zero_grad()
        assert m.weight.grad is None

    def test_optimizer_group(self):
        p1 = Tensor([1.0], requires_grad=True).realize()
        p2 = Tensor([2.0], requires_grad=True).realize()
        p1._grad = Tensor([1.0])
        p2._grad = Tensor([2.0])
        group = OptimizerGroup(SGD([p1], lr=0.1), SGD([p2], lr=0.2))
        group.step()
        assert approx(p1.numpy(), [0.9])
        assert approx(p2.numpy(), [1.6])


# ── Adam ──

class TestAdam:
    def test_loss_decreases(self):
        Tensor.manual_seed(42)
        m = Linear(2, 1)
        opt = Adam(get_parameters(m), lr=0.01)
        losses = []
        for _ in range(10):
            opt.zero_grad()
            x = Tensor([[1.0, 2.0]])
            target = Tensor([[3.0]])
            loss = (m(x) - target).square().sum()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0]

    def test_beta_power_state_updates_in_graph(self):
        p = Tensor([1.0], requires_grad=True).realize()
        opt = Adam([p], lr=0.1)
        p._grad = Tensor([1.0])
        scheduled = opt.schedule_step()
        assert len(scheduled) == 5
        assert scheduled[0] is opt.b1_t
        assert scheduled[1] is opt.b2_t
        assert scheduled[2] is opt.m[0]
        assert scheduled[3] is opt.v[0]
        assert scheduled[4] is p
        scheduled[0].realize(*scheduled[1:])
        assert approx(opt.b1_t.numpy(), [0.9], tol=1e-6)
        assert approx(opt.b2_t.numpy(), [0.999], tol=1e-6)
        assert approx(opt.m[0].numpy(), [0.1], tol=1e-6)
        assert approx(opt.v[0].numpy(), [0.001], tol=1e-6)
        assert approx(p.numpy(), [0.9], tol=1e-4)


# ── AdamW ──

class TestAdamW:
    def test_loss_decreases(self):
        Tensor.manual_seed(42)
        m = Linear(2, 1)
        opt = AdamW(get_parameters(m), lr=0.01, weight_decay=0.01)
        losses = []
        for _ in range(10):
            opt.zero_grad()
            x = Tensor([[1.0, 2.0]])
            target = Tensor([[3.0]])
            loss = (m(x) - target).square().sum()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0]

    def test_weight_decay_uses_core_update(self):
        p = Tensor([1.0], requires_grad=True).realize()
        opt = AdamW([p], lr=0.1, weight_decay=0.01)
        p._grad = Tensor([0.0])
        opt.step()
        assert approx(p.numpy(), [0.999], tol=1e-5)


# ── ASSIGN ──

class TestAssign:
    def test_assign_basic(self):
        a = Tensor([1.0, 2.0, 3.0])
        a.assign(a + 10).realize()
        assert approx(a.numpy(), [11.0, 12.0, 13.0])

    def test_assign_self_mul(self):
        a = Tensor([1.0, 2.0, 3.0, 4.0])
        a.assign(a * 2).realize()
        assert approx(a.numpy(), [2.0, 4.0, 6.0, 8.0])


# ── Instance export ──

class TestInstanceExport:
    def test_typed_integer_input_preserves_bytes_and_rejects_float_binding(self):
        x = Tensor.empty((3,), dtype='int32')
        out = x.cast('float32')
        inst = Instance.from_tensors(inputs={'typed_x': x}, outputs={'typed_out': out})

        result = inst.forward(typed_x=np.array([0, 1, 2], dtype=np.int32))
        np.testing.assert_array_equal(result['typed_out'], np.array([0, 1, 2], dtype=np.float32))
        with pytest.raises(RuntimeError, match='forward failed'):
            inst.forward(typed_x=np.array([0, 1, 2], dtype=np.float32))

    def test_functional_model_exports_selected_forward_entrypoint(self):
        from polygrad import _ffi

        w = Tensor([[2.0], [3.0]], requires_grad=True).realize()
        x = Tensor.empty((1, 2))
        y = x.dot(w)
        assert _ffi._lib.poly_uop_reachable(x._ctx, y.uop_logical.raw, w.uop_logical.raw)
        if w.uop_logical.raw != w.uop.raw:
            assert not _ffi._lib.poly_uop_reachable(x._ctx, y.uop_logical.raw, w.uop.raw)
            assert y.uop_physical is not None
            assert _ffi._lib.poly_uop_reachable(x._ctx, y.uop_physical.raw, w.uop.raw)

        inst = Instance.from_tensors(
            inputs={"py_export_x": x},
            outputs={"py_export_output": y},
            params={"py_export_w": w},
        )
        assert inst.param_count == 1
        assert inst.param_name(0) == "py_export_w"

        out = inst.forward(py_export_x=np.array([[10.0, 20.0]], dtype=np.float32))
        assert "py_export_output" in out
        assert np.allclose(out["py_export_output"], [80.0], atol=1e-5)

    def test_from_tensors_uses_instance_local_bindings(self):
        from polygrad import _ffi

        w = Tensor([[2.0]], requires_grad=True).realize()
        x = Tensor.empty((1, 1))
        y = x.dot(w)
        before = _ffi._lib.poly_ctx_named_count(x._ctx)

        inst = Instance.from_tensors(
            inputs={"local_x": x},
            outputs={"local_y": y},
            params={"local_w": w},
        )

        assert _ffi._lib.poly_ctx_named_count(x._ctx) == before
        assert inst.param_name(0) == "local_w"
        out = inst.forward(local_x=np.array([[3.0]], dtype=np.float32))
        assert np.allclose(out["local_y"], [6.0], atol=1e-5)

    def test_from_bindings_primitive_uses_instance_local_bindings(self):
        from polygrad import _ffi

        w = Tensor([[7.0]], requires_grad=True).realize()
        x = Tensor.empty((1, 1))
        y = x.dot(w)
        before = _ffi._lib.poly_ctx_named_count(x._ctx)

        inst = Instance.from_bindings(
            bindings=[
                {"name": "bind_x", "role": "input", "tensor": x},
                {"name": "bind_w", "role": "state", "tensor": w},
                {"name": "bind_y", "role": "output", "tensor": y},
            ],
            entrypoints=[
                {"name": "forward", "inputs": ["bind_x"], "outputs": ["bind_y"]},
            ],
        )

        assert _ffi._lib.poly_ctx_named_count(x._ctx) == before
        assert inst.param_name(0) == "bind_w"
        out = inst.forward(bind_x=np.array([[3.0]], dtype=np.float32))
        assert np.allclose(out["bind_y"], [21.0], atol=1e-5)

    def test_from_bindings_uses_tensor_requires_grad_for_trainability(self):
        w = Tensor([[7.0]], requires_grad=False).realize()
        x = Tensor.empty((1, 1))
        y = x.dot(w)

        inst = Instance.from_bindings(
            bindings=[
                {"name": "x", "role": "input", "tensor": x},
                {"name": "w", "role": "state", "tensor": w},
                {"name": "y", "role": "output", "tensor": y},
            ],
            entrypoints=[
                {"name": "forward", "inputs": ["x"], "outputs": ["y"]},
            ],
        )

        assert inst.param_trainable(0) is False

    def test_from_bindings_rejects_lazy_parameter_without_current_storage(self):
        w = Tensor([[7.0]], requires_grad=True)
        x = Tensor.empty((1, 1))
        y = x.dot(w)

        with pytest.raises(RuntimeError, match="w.*has no buffer identity"):
            Instance.from_bindings(
                bindings=[
                    {"name": "x", "role": "input", "tensor": x},
                    {"name": "w", "role": "state", "tensor": w},
                    {"name": "y", "role": "output", "tensor": y},
                ],
                entrypoints=[
                    {"name": "forward", "inputs": ["x"], "outputs": ["y"]},
                ],
            )

    def test_from_tensors_keeps_tinygrad_style_plain_object(self):
        class LinearNet:
            def __init__(self):
                self.weight = Tensor([[4.0], [5.0]], requires_grad=True).realize()

            def __call__(self, x):
                return x.dot(self.weight)

        net = LinearNet()
        x = Tensor.empty((1, 2))
        out_tensor = net(x)
        inst = Instance.from_tensors(
            inputs={"py_trace_x": x},
            outputs={"output": out_tensor},
            params={"weight": net.weight},
        )

        out = inst.forward(py_trace_x=np.array([[2.0, 3.0]], dtype=np.float32))
        assert np.allclose(out["output"], [23.0], atol=1e-5)

    def test_constructor_state_names_survive_ir_roundtrip(self):
        w = Tensor([[2.0], [3.0]], requires_grad=True).realize()
        x = Tensor.empty((1, 2))
        logits = x.dot(w)

        inst = Instance(
            inputs={"x": x},
            state={"layers.0.weight": w},
            outputs={"logits": logits},
            entrypoints=[
                {"name": "forward", "inputs": ["x"], "outputs": ["logits"]},
            ],
        )
        assert inst.param_count == 1
        assert inst.param_name(0) == "layers.0.weight"
        out = inst.forward(x=np.array([[10.0, 20.0]], dtype=np.float32))
        assert np.allclose(out["logits"], [80.0], atol=1e-5)

        inst2 = Instance.from_ir(inst.export_ir(), inst.export_weights())
        assert inst2.param_count == 1
        assert inst2.param_name(0) == "layers.0.weight"
        out2 = inst2.forward(x=np.array([[10.0, 20.0]], dtype=np.float32))
        assert np.allclose(out2["logits"], [80.0], atol=1e-5)

    def test_tinygrad_training_aliases_exist(self):
        t = Tensor.kaiming_uniform(2, 3)
        assert t.shape == (2, 3)
        before = Tensor.training
        with Tensor.train():
            assert Tensor.training
        assert Tensor.training == before

    def test_model_fit_uses_instance_training_path(self):
        w = Tensor([[1.0]], requires_grad=True).realize()
        x = Tensor.empty((1, 1))
        y = Tensor.empty((1, 1))
        pred = x.dot(w)
        loss = (pred - y).square().mean()
        inst = Instance.from_tensors(
            inputs={"fit_x": x},
            targets={"fit_y": y},
            outputs={"fit_out": pred},
            losses={"loss": loss},
            params={"fit_w": w},
        )
        losses = inst.fit(
            {
                "fit_x": np.array([[1.0]], dtype=np.float32),
                "fit_y": np.array([[3.0]], dtype=np.float32),
            },
            epochs=4,
            optimizer="sgd",
            lr=0.1,
        )
        assert len(losses) == 4
        assert losses[-1] < losses[0]

    def test_assign_with_other(self):
        a = Tensor([1.0, 2.0, 3.0, 4.0])
        b = Tensor([10.0, 20.0, 30.0, 40.0])
        a.assign(a + b).realize()
        assert approx(a.numpy(), [11.0, 22.0, 33.0, 44.0])

    def test_assign_preserves_shape(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        a.assign(a * 3).realize()
        assert a.shape == (2, 2)
        assert approx(a.numpy(), [[3.0, 6.0], [9.0, 12.0]])

    def test_assign_chained(self):
        a = Tensor([1.0, 2.0, 3.0])
        a.assign(a + 1).realize()
        a.assign(a * 2).realize()
        assert approx(a.numpy(), [4.0, 6.0, 8.0])


# ── State dict ──

class TestStateDict:
    def test_get_state_dict(self):
        m = Linear(3, 2)
        sd = get_state_dict(m)
        assert 'weight' in sd
        assert 'bias' in sd
        assert sd['weight'].shape == (2, 3)
        assert sd['bias'].shape == (2,)

    def test_get_parameters(self):
        m = Linear(3, 2)
        params = get_parameters(m)
        assert len(params) == 2

    def test_load_state_dict(self):
        m1 = Linear(2, 1)
        m2 = Linear(2, 1)
        sd = get_state_dict(m1)
        load_state_dict(m2, sd)
        assert approx(m1.weight.numpy(), m2.weight.numpy())
        assert approx(m1.bias.numpy(), m2.bias.numpy())

    def test_nested(self):
        class Model:
            def __init__(self):
                self.l1 = Linear(2, 3)
                self.l2 = Linear(3, 1)
        m = Model()
        sd = get_state_dict(m)
        assert 'l1.weight' in sd
        assert 'l1.bias' in sd
        assert 'l2.weight' in sd
        assert 'l2.bias' in sd

    def test_get_parameters_nested(self):
        class Model:
            def __init__(self):
                self.l1 = Linear(2, 3)
                self.l2 = Linear(3, 1)
        m = Model()
        params = get_parameters(m)
        assert len(params) == 4  # weight+bias for each layer

    def test_get_parameters_includes_buffers_for_optimizer_partition(self):
        bn = BatchNorm(4)
        state = get_state_dict(bn)
        params = get_parameters(bn)
        assert [id(x) for x in params] == [id(x) for x in state.values()]
        assert any(x is bn.running_mean for x in params)
        assert any(x is bn.running_var for x in params)
        opt = SGD(params, lr=0.1)
        assert [id(x) for x in opt.params] == [id(bn.weight), id(bn.bias)]
        assert [id(x) for x in opt.buffers] == [id(bn.running_mean), id(bn.running_var)]


class TestParameterMarkerParity:
    def test_default_and_result_markers_match_tinygrad(self):
        source = Tensor.zeros(1).is_param_(False)
        assert Tensor.zeros(1).is_param is True
        assert (source + 1).is_param is True
        assert source.clone().is_param is False

    def test_optimizer_uses_marker_and_enables_polygrad_autograd(self):
        param = Tensor.zeros(1)
        buffer = Tensor.ones(1).is_param_(False)
        assert param.requires_grad is False
        opt = SGD([param, buffer], lr=0.1)
        assert opt.params == [param]
        assert opt.buffers == [buffer]
        assert param.requires_grad is True
        assert buffer.requires_grad is False

    def test_hlb_style_custom_norm_split(self):
        class CustomNorm:
            def __init__(self):
                self.weight = Tensor.ones(4).is_param_(False)
                self.bias = Tensor.zeros(4)
                self.running_mean = Tensor.zeros(4).is_param_(False)

        state = get_state_dict(CustomNorm())
        bias = [value for name, value in state.items() if value.is_param and 'bias' in name]
        non_bias = [value for name, value in state.items() if value.is_param and 'bias' not in name]
        assert len(bias) == 1
        assert len(non_bias) == 0
        assert [id(x) for x in SGD(bias, lr=0.1).params] == [id(x) for x in bias]


# ── Segment-wise backward ──

class TestLiveTensorBackward:
    """Tests for tinygrad-style live-UOp gradient target discovery."""

    def test_two_segment_chain(self):
        """Gradient flows through a lazy matmul chain."""
        Tensor.manual_seed(42)
        x = Tensor.rand(1, 4).realize()
        w1 = (Tensor.rand(4, 4) * 0.1).realize()
        w1.requires_grad = True
        w1._requires_grad = True
        w2 = (Tensor.rand(2, 4) * 0.1).realize()
        w2.requires_grad = True
        w2._requires_grad = True

        h = x.matmul(w1.T)
        out = h.matmul(w2.T)
        loss = out.mean()
        loss.backward()

        assert w1.grad is not None
        assert w2.grad is not None
        assert np.all(np.isfinite(w1.grad.numpy()))
        assert np.all(np.isfinite(w2.grad.numpy()))
        assert np.linalg.norm(w1.grad.numpy()) > 0
        assert np.linalg.norm(w2.grad.numpy()) > 0

    def test_three_segment_chain(self):
        """Gradient flows through a deeper lazy matmul chain."""
        Tensor.manual_seed(42)
        x = Tensor.rand(1, 4).realize()
        w1 = (Tensor.rand(4, 4) * 0.1).realize()
        w1.requires_grad = True
        w1._requires_grad = True
        w2 = (Tensor.rand(4, 4) * 0.1).realize()
        w2.requires_grad = True
        w2._requires_grad = True
        w3 = (Tensor.rand(2, 4) * 0.1).realize()
        w3.requires_grad = True
        w3._requires_grad = True

        h1 = x.matmul(w1.T)
        h2 = h1.matmul(w2.T)
        out = h2.matmul(w3.T)
        loss = out.mean()
        loss.backward()

        for w in [w1, w2, w3]:
            assert w.grad is not None
            assert np.all(np.isfinite(w.grad.numpy()))
            assert np.linalg.norm(w.grad.numpy()) > 0

    def test_segment_matches_direct(self):
        """Live-tensor backward matches direct backward for a matmul chain."""
        # Direct (no realize boundaries)
        Tensor.manual_seed(42)
        rng = np.random.RandomState(42)
        w_np = (rng.randn(4, 4) * 0.1).astype(np.float32)
        x_np = rng.randn(1, 4).astype(np.float32)

        w1 = Tensor(w_np, requires_grad=True)
        x1 = Tensor(x_np)
        h1 = x1.matmul(w1.T)
        loss1 = h1.mean()
        loss1.backward()
        direct_grad = w1.grad.numpy().copy()

        # Same graph from realized source buffers, without materializing the
        # trainable path before backward.
        w2 = Tensor(w_np.copy()).realize()
        w2.requires_grad = True
        w2._requires_grad = True
        x2 = Tensor(x_np.copy()).realize()
        h2 = x2.matmul(w2.T)
        loss2 = h2.mean()
        loss2.backward()
        live_grad = w2.grad.numpy()

        np.testing.assert_allclose(live_grad, direct_grad, rtol=1e-4)

    def test_matmul_transpose_backward(self):
        """Backward through q @ k.T pattern (attention-style)."""
        Tensor.manual_seed(42)
        q = Tensor.rand(1, 2, 4, 8).realize()
        q.requires_grad = True
        q._requires_grad = True
        k = Tensor.rand(1, 2, 4, 8).realize()
        k.requires_grad = True
        k._requires_grad = True

        scores = q.matmul(k.transpose(-2, -1))
        loss = (scores * 0.25).sum()
        loss.backward()

        assert np.all(np.isfinite(q.grad.numpy()))
        assert np.all(np.isfinite(k.grad.numpy()))

    def test_softmax_backward_through_realize(self):
        """Backward through softmax produces usable gradients."""
        Tensor.manual_seed(42)
        x = Tensor.rand(2, 4).realize()
        x.requires_grad = True
        x._requires_grad = True

        # Use a non-trivial loss (weighted sum, not plain sum which has trivial zero grad)
        w = Tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
        y = x.softmax(axis=-1)
        loss = (y * w).sum()
        loss.backward()

        assert x.grad is not None
        assert np.all(np.isfinite(x.grad.numpy()))

    def test_layernorm_backward(self):
        """Backward through LayerNorm (mean→var→normalize)."""
        Tensor.manual_seed(42)
        ln = LayerNorm(4)
        x = Tensor.rand(2, 4).realize()
        x.requires_grad = True
        x._requires_grad = True

        y = ln(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert np.all(np.isfinite(x.grad.numpy()))
        assert ln.weight.grad is not None
        assert ln.bias.grad is not None

    def test_mlp_backward(self):
        """MLP with two Linear layers and ReLU."""
        Tensor.manual_seed(42)
        l1 = Linear(4, 8)
        l2 = Linear(8, 2)

        x = Tensor.rand(2, 4)
        h = l1(x).relu()
        out = l2(h)
        loss = out.sum()
        loss.backward()

        assert l1.weight.grad is not None
        assert l2.weight.grad is not None
        assert np.all(np.isfinite(l1.weight.grad.numpy()))
        assert np.all(np.isfinite(l2.weight.grad.numpy()))
        assert np.linalg.norm(l2.weight.grad.numpy()) > 0

    def test_grad_accumulation_multiple_paths(self):
        """A parameter used through multiple lazy paths gets accumulated gradient."""
        Tensor.manual_seed(42)
        w = (Tensor.rand(4, 4) * 0.1).realize()
        w.requires_grad = True
        w._requires_grad = True

        x = Tensor.rand(1, 4).realize()
        h1 = x.matmul(w.T)
        h2 = h1.matmul(w.T)  # w used again
        loss = h2.sum()
        loss.backward()

        assert w.grad is not None
        assert np.all(np.isfinite(w.grad.numpy()))
        # Gradient should be non-zero (accumulated from both segments)
        assert np.linalg.norm(w.grad.numpy()) > 0

    def test_mlp_training_with_sgd(self):
        """End-to-end: MLP training reduces loss."""
        Tensor.manual_seed(42)
        l1 = Linear(4, 8)
        l2 = Linear(8, 1)
        params = get_parameters(l1) + get_parameters(l2)
        opt = SGD(params, lr=0.01)
        x = Tensor.rand(2, 4)
        target = Tensor([[1.0], [0.0]])

        losses = []
        for _ in range(5):
            opt.zero_grad()
            h = l1(x).relu()
            out = l2(h)
            loss = (out - target).square().sum()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0]

    def test_qkv_diamond_shared_intermediate(self):
        """Verify gradients flow through q/k/v shared qkv intermediate."""
        Tensor.manual_seed(42)
        x = Tensor.rand(1, 4, 8)
        # w shape (8, 24) so x @ w gives (1, 4, 24) which splits into 3 × 8
        w = (Tensor.rand(8, 24) * 0.1).realize()
        w.requires_grad = True
        w._requires_grad = True

        # Attention-like pattern: shared projection split 3 ways
        qkv = x.dot(w)
        q = qkv.shrink(((0, 1), (0, 4), (0, 8)))
        k = qkv.shrink(((0, 1), (0, 4), (8, 16)))
        v = qkv.shrink(((0, 1), (0, 4), (16, 24)))

        loss = (q * q + k * k + v * v).sum()
        loss.backward()

        g = w.grad.numpy()
        third = g.shape[1] // 3
        # All three sections must have non-zero gradients
        assert np.abs(g[:, :third]).sum() > 0, "q section gradient is zero"
        assert np.abs(g[:, third:2*third]).sum() > 0, "k section gradient is zero"
        assert np.abs(g[:, 2*third:]).sum() > 0, "v section gradient is zero"

        # Compare magnitude across sections — they should be similar (same loss weight)
        norms = [np.linalg.norm(g[:, i*third:(i+1)*third]) for i in range(3)]
        ratio = max(norms) / min(norms)
        assert ratio < 10, f"Gradient section norms differ too much: {norms} (ratio={ratio:.1f})"

    def test_qkv_diamond_training(self):
        """QKV diamond pattern: lazy gradient produces loss decrease over steps."""
        Tensor.manual_seed(42)
        x = Tensor.rand(1, 4, 8)
        w = (Tensor.rand(8, 24) * 0.1).realize()
        w.requires_grad = True
        w._requires_grad = True

        losses = []
        for _ in range(5):
            # Match tinygrad: realization is a materialization boundary, so
            # trainable sources must remain in the current UOp graph until
            # backward has selected live gradient targets.
            qkv = x.dot(w)
            q = qkv.shrink(((0, 1), (0, 4), (0, 8)))
            k = qkv.shrink(((0, 1), (0, 4), (8, 16)))
            v = qkv.shrink(((0, 1), (0, 4), (16, 24)))
            loss = (q * q + k * k + v * v).sum()
            loss.backward()
            losses.append(loss.item())
            # Manual SGD step
            updated = (w - w.grad * 0.01).realize()
            w._tensor = updated._tensor
            w._data = updated._data
            w._grad = None

        assert all(np.isfinite(l) for l in losses), f"NaN/Inf in losses: {losses}"
        assert losses[-1] < losses[0], \
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


# ── Variable (dynamic shapes) ──

class TestVariable:
    def test_creation(self):
        from polygrad import Variable
        v = Variable("N", 1, 128)
        assert v.name == "N"
        assert v.min_val == 1
        assert v.max_val == 128

    def test_bind(self):
        from polygrad import Variable
        v = Variable("N", 1, 128)
        bound = v.bind(32)
        assert int(bound) == 32
        assert bound.variable is v

    def test_bind_out_of_range(self):
        from polygrad import Variable
        v = Variable("N", 1, 128)
        with pytest.raises(AssertionError):
            v.bind(0)
        with pytest.raises(AssertionError):
            v.bind(200)

    def test_repr(self):
        from polygrad import Variable
        v = Variable("N", 1, 128)
        assert "N" in repr(v)
        bound = v.bind(32)
        assert "32" in repr(bound)
