"""Tests for polygrad.nn module — layers, optimizers, state dict."""

import base64
import math
import zlib
import numpy as np
import pytest
from polygrad import GlobalCounters, Instance, Tensor, _ffi
from polygrad.nn import (
    Linear,
    LayerNorm,
    LayerNorm2d,
    GroupNorm,
    RMSNorm,
    Embedding,
    Dropout,
    Conv2d,
    BatchNorm,
    BatchNorm2d,
    BatchNorm3d,
    SGD,
    Adam,
    AdamW,
    OptimizerGroup,
    get_parameters,
    get_state_dict,
    load_state_dict,
)
from polygrad.nn.state import safe_load, safe_load_metadata, torch_load


# ── Helpers ──


def approx(a, b, tol=1e-4):
    return np.allclose(a, b, atol=tol)


def graph_op_counts(root):
    counts, seen, stack = {}, set(), [root]
    while stack:
        node = stack.pop()
        if not node or node in seen:
            continue
        seen.add(node)
        counts[node.op_name] = counts.get(node.op_name, 0) + 1
        stack.extend(node.src)
    return counts


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

    def test_layernorm2d_exact_pinned_composition(self):
        xv = (
            np.arange(2 * 3 * 2 * 2, dtype=np.float32).reshape(2, 3, 2, 2) - 7
        ) / 5
        ln = LayerNorm2d(3, eps=1e-5)
        ln.weight.assign(Tensor([1.5, -0.5, 2.0])).realize()
        ln.bias.assign(Tensor([0.25, -0.75, 0.5])).realize()
        out = ln(Tensor(xv))

        assert out.shape == (2, 3, 2, 2)
        assert out.uop.op_name == "PERMUTE"
        assert graph_op_counts(out.uop) == {
            "ADD": 3,
            "BUFFER": 3,
            "CONST": 6,
            "COPY": 1,
            "DEVICE": 2,
            "EXPAND": 7,
            "MUL": 6,
            "PERMUTE": 2,
            "RECIPROCAL": 2,
            "REDUCE": 2,
            "RESHAPE": 6,
            "SQRT": 1,
            "STACK": 5,
            "UNIQUE": 3,
        }
        nhwc = xv.transpose(0, 2, 3, 1)
        centered = nhwc - nhwc.mean(axis=-1, keepdims=True)
        expected = centered / np.sqrt((centered * centered).mean(axis=-1, keepdims=True) + 1e-5)
        expected = expected * np.asarray([1.5, -0.5, 2.0]) + np.asarray([0.25, -0.75, 0.5])
        np.testing.assert_allclose(out.numpy(), expected.transpose(0, 3, 1, 2), rtol=1e-6, atol=1e-6)


# ── RMSNorm ──


class TestRMSNorm:
    def test_exact_pinned_expression_and_optional_affine(self):
        x = (Tensor.arange(8).float().reshape(2, 4) - 3.0) / 5.0
        rn = RMSNorm(4, eps=1e-5)
        expected = (
            x.float() *
            (x.float().square().mean(axis=-1, keepdim=True) + 1e-5).rsqrt()
        ).cast(x.dtype) * rn.weight
        assert rn(x).uop.raw == expected.uop.raw

        no_affine = RMSNorm(4, elementwise_affine=False)
        assert no_affine.weight is None
        assert no_affine(x).uop.raw == no_affine._norm(x.float()).cast(x.dtype).uop.raw

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
    def test_dimensional_names_are_exact_aliases(self):
        assert BatchNorm2d is BatchNorm
        assert BatchNorm3d is BatchNorm

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
        before_count = bn.num_batches_tracked.numpy().copy()
        Tensor.training = True
        _ = bn(x).realize()
        Tensor.training = False
        after_mean = bn.running_mean.numpy()
        after_var = bn.running_var.numpy()
        after_count = bn.num_batches_tracked.numpy()
        assert not np.allclose(before_mean, after_mean)
        assert not np.allclose(before_var, after_var)
        assert np.array_equal(after_count, before_count + 1)

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
    def test_initializer_and_selector_match_pinned_expressions(self):
        Tensor.manual_seed(123)
        expected_weight = Tensor.glorot_uniform(5, 3).numpy()
        Tensor.manual_seed(123)
        emb = Embedding(5, 3)
        np.testing.assert_array_equal(emb.weight.numpy(), expected_weight)

        idx = Tensor.arange(3).cast('int32')
        expected = (
            Tensor.arange(5).eq(idx.unsqueeze(-1)).unsqueeze(-1)
            .where(emb.weight, 0).sum(axis=-2, dtype=emb.weight.dtype)
        )
        assert emb(idx).uop.raw == expected.uop.raw

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
        with pytest.raises(TypeError, match="Expected integer dtype"):
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
    def test_scalar_rank8_and_shared_multi_output_round_trip(self):
        scalar_x = Tensor.empty(())
        scalar_w = Tensor(3.0, requires_grad=True)
        scalar = Instance.from_tensors(
            inputs={"x": scalar_x},
            outputs={"output": scalar_x * scalar_w},
            params={"w": scalar_w},
        )
        scalar_restored = Instance.from_ir(scalar.export_ir(), scalar.export_weights())
        try:
            result = scalar_restored.forward(x=np.array(2.0, dtype=np.float32))["output"]
            assert result.shape == ()
            assert result.item() == 6.0
        finally:
            scalar_restored.free()
            scalar.free()

        rank8_shape = (1,) * 8
        rank8_x = Tensor.empty(rank8_shape)
        rank8_w = Tensor.ones(*rank8_shape, requires_grad=True)
        shared = rank8_x + rank8_w
        rank8 = Instance.from_tensors(
            inputs={"x": rank8_x},
            outputs={"plus": shared + 1.0, "minus": shared - 1.0},
            params={"w": rank8_w},
        )
        rank8_restored = Instance.from_ir(rank8.export_ir(), rank8.export_weights())
        try:
            result = rank8_restored.forward(
                x=np.full(rank8_shape, 2.0, dtype=np.float32)
            )
            assert result["plus"].shape == rank8_shape
            assert result["minus"].shape == rank8_shape
            np.testing.assert_array_equal(result["plus"], np.full(rank8_shape, 4.0))
            np.testing.assert_array_equal(result["minus"], np.full(rank8_shape, 2.0))
        finally:
            rank8_restored.free()
            rank8.free()

    def test_duplicate_abi_storage_alias_fails_closed_like_tinyjit(self):
        x = Tensor.empty(2)
        with pytest.raises(RuntimeError):
            Instance.from_tensors(
                inputs={"a": x, "b": x}, outputs={"output": x + x}
            )

    def test_dynamic_input_alias_with_persistent_state_fails_closed(self):
        x = Tensor.empty(2)
        with pytest.raises(RuntimeError):
            Instance.from_tensors(
                inputs={"x": x}, outputs={"output": x + x}, params={"w": x}
            )

    def test_output_alias_of_dynamic_input_round_trips(self):
        x = Tensor.empty(2)
        source = Instance.from_tensors(inputs={"x": x}, outputs={"output": x})
        restored = Instance.from_ir(source.export_ir())
        try:
            value = np.array([3.0, 4.0], dtype=np.float32)
            np.testing.assert_array_equal(source.forward(x=value)["output"], value)
            np.testing.assert_array_equal(restored.forward(x=value)["output"], value)
        finally:
            restored.free()
            source.free()

    def test_named_partial_view_state_fails_closed(self):
        base = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        view = base[1:3]
        x = Tensor.empty(2)
        with pytest.raises(RuntimeError, match="unsupported named view state"):
            Instance.from_tensors(
                inputs={"x": x}, outputs={"output": x + view},
                params={"base": base, "view": view},
            )

    def test_input_dependent_named_state_effect_fails_closed(self):
        x = Tensor.empty(1)
        w = Tensor([1.0], requires_grad=False)
        output = w.assign(w + x)
        with pytest.raises(RuntimeError, match="depends on an input or target"):
            Instance.from_tensors(
                inputs={"x": x}, outputs={"output": output}, params={"w": w}
            )

    def test_stochastic_output_requires_named_rng_state(self):
        Tensor.manual_seed(123)
        x = Tensor.empty(2)
        with pytest.raises(RuntimeError, match="unbound storage"):
            Instance.from_tensors(
                inputs={"x": x}, outputs={"output": x + Tensor.rand(2)}
            )

    def test_assigned_realized_input_history_fails_closed(self):
        x = Tensor.empty(2)
        x.assign(Tensor([4.0, 5.0])).realize()
        output = x * Tensor([2.0, 3.0]) + 1.0
        with pytest.raises(RuntimeError, match="no buffer identity"):
            Instance.from_tensors(inputs={"x": x}, outputs={"output": output})

    def test_float16_state_preserves_exact_dtype_and_storage_bits(self):
        w = Tensor([1.5, -2.0], dtype="float16", requires_grad=True)
        x = Tensor.empty((2,), dtype="float16")
        source = Instance.from_tensors(
            inputs={"x": x}, outputs={"output": x + w}, params={"w": w}
        )
        try:
            raw = source.param_data(0)
            assert raw.dtype == np.float16
            np.testing.assert_array_equal(raw.view(np.uint16), [0x3E00, 0xC000])
            assert source.param_dtype(0) == "float16"

            restored = Instance.from_ir(source.export_ir(), source.export_weights())
            try:
                restored_raw = restored.param_data(0)
                assert restored_raw.dtype == np.float16
                np.testing.assert_array_equal(
                    restored_raw.view(np.uint16), [0x3E00, 0xC000]
                )
                assert restored.param_dtype(0) == "float16"
            finally:
                restored.free()
        finally:
            source.free()

    def test_typed_integer_input_preserves_bytes_and_rejects_float_binding(self):
        x = Tensor.empty((3,), dtype="int32")
        out = x.cast("float32")
        inst = Instance.from_tensors(inputs={"typed_x": x}, outputs={"typed_out": out})

        result = inst.forward(typed_x=np.array([0, 1, 2], dtype=np.int32))
        np.testing.assert_array_equal(
            result["typed_out"], np.array([0, 1, 2], dtype=np.float32)
        )
        with pytest.raises(RuntimeError, match=r"call\('forward'\) failed"):
            inst.forward(typed_x=np.array([0, 1, 2], dtype=np.float32))

    def test_functional_model_exports_selected_forward_entrypoint(self):
        from polygrad import _ffi

        w = Tensor([[2.0], [3.0]], requires_grad=True).realize()
        x = Tensor.empty((1, 2))
        y = x.dot(w)
        assert _ffi._lib.poly_uop_reachable(
            x._ctx, y.uop_logical.raw, w.uop_logical.raw
        )
        if w.uop_logical.raw != w.uop.raw:
            assert not _ffi._lib.poly_uop_reachable(
                x._ctx, y.uop_logical.raw, w.uop.raw
            )
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

    def test_from_bindings_snapshots_named_lazy_parameter(self):
        w = Tensor([[7.0]], requires_grad=True)
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
        out = inst.forward(x=np.array([[3.0]], dtype=np.float32))
        assert np.allclose(out["y"], [21.0], atol=1e-5)

    def test_from_ir_freshly_initializes_closed_named_value(self):
        w = (
            Tensor.full((2,), 3.0, buffer=False)
            + Tensor.full((2,), 1.0, buffer=False)
        )
        w.requires_grad = True
        x = Tensor.empty((2,))
        y = x * w
        source = Instance.from_bindings(
            bindings=[
                {"name": "x", "role": "input", "tensor": x},
                {"name": "w", "role": "state", "tensor": w},
                {"name": "output", "role": "output", "tensor": y},
            ],
            entrypoints=[{"name": "forward", "inputs": ["x"], "outputs": ["output"]}],
        )
        try:
            fresh = Instance.from_ir(source.export_ir())
            try:
                np.testing.assert_array_equal(fresh.param_data(0), [4.0, 4.0])
                out = fresh.forward(x=np.array([2.0, 3.0], dtype=np.float32))
                np.testing.assert_array_equal(out["output"], [8.0, 12.0])
            finally:
                fresh.free()
        finally:
            source.free()

    def test_from_ir_rejects_stateful_rng_initializer_without_checkpoint(self):
        Tensor.manual_seed(7)
        w = Tensor.rand(2)
        w.requires_grad = True
        x = Tensor.empty((2,))
        source = Instance.from_bindings(
            bindings=[
                {"name": "x", "role": "input", "tensor": x},
                {"name": "w", "role": "state", "tensor": w},
                {"name": "output", "role": "output", "tensor": x * w},
            ],
            entrypoints=[{"name": "forward", "inputs": ["x"], "outputs": ["output"]}],
        )
        try:
            with pytest.raises(RuntimeError, match="NULL pointer"):
                Instance.from_ir(source.export_ir())
        finally:
            source.free()

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

    def test_explicit_module_device_map_uses_named_value_bindings(self):
        x = Tensor.empty((2,))
        w0 = Tensor([3.0, 4.0], requires_grad=True)
        w1 = Tensor([2.0, 3.0], requires_grad=True)
        hidden = x + w0
        output = hidden * w1

        inst = Instance.from_tensors(
            inputs={"x": x},
            outputs={"output": output},
            params={"layers.0.weight": w0, "layers.1.weight": w1},
            modules=[
                {"name": "layers.0", "inputs": [x], "output": hidden},
                {"name": "layers.1", "inputs": [hidden], "output": output},
            ],
        )
        ir_before = inst.export_ir()
        weights_before = inst.export_weights()
        inst.set_device_map({"layers.0": "CPU", "layers.1": "CPU:1"})
        result = inst.forward(x=np.array([1.0, 2.0], dtype=np.float32))
        np.testing.assert_array_equal(
            result["output"], np.array([8.0, 18.0], dtype=np.float32)
        )
        assert inst.export_ir() == ir_before
        assert inst.export_weights() == weights_before

        with pytest.raises(ValueError, match="incomplete"):
            inst.set_device_map({"layers.0": "CPU"})

        inst.set_device_map({"layers.0": "CPU:1", "layers.1": "CPU"})
        result = inst.forward(x=np.array([1.0, 2.0], dtype=np.float32))
        np.testing.assert_array_equal(
            result["output"], np.array([8.0, 18.0], dtype=np.float32)
        )
        assert inst.export_ir() == ir_before
        assert inst.export_weights() == weights_before
        restored = Instance.from_ir(inst.export_ir(), inst.export_weights())
        try:
            restored_result = restored.forward(
                x=np.array([1.0, 2.0], dtype=np.float32)
            )
            np.testing.assert_array_equal(
                restored_result["output"], np.array([8.0, 18.0], dtype=np.float32)
            )
        finally:
            restored.free()

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
    def test_safe_load_matches_pinned_lazy_disk_graph_and_values(self, tmp_path):
        fixture = tmp_path / "state.safetensors"
        fixture.write_bytes(base64.b64decode(
            "MAEAAAAAAAB7Il9fbWV0YWRhdGFfXyI6eyJmb3JtYXQiOiJwb2x5Z3JhZC1wYXJpdHkiLCJ2ZXJzaW9uIjoiMSJ9LCJmbG9hdCI6eyJkdHlwZSI6IkYzMiIsInNoYXBlIjpbMiwyXSwiZGF0YV9vZmZzZXRzIjpbMCwxNl19LCJsb25nIjp7ImR0eXBlIjoiSTY0Iiwic2hhcGUiOlsyXSwiZGF0YV9vZmZzZXRzIjpbMTYsMzJdfSwiYm9vbCI6eyJkdHlwZSI6IkJPT0wiLCJzaGFwZSI6WzIsMl0sImRhdGFfb2Zmc2V0cyI6WzMyLDM2XX0sInNjYWxhciI6eyJkdHlwZSI6IkYzMiIsInNoYXBlIjpbXSwiZGF0YV9vZmZzZXRzIjpbMzYsNDBdfX0gICAgICAgAACgPwAAIMAAAHBAAACQQAAAAAAA////AwAAAAABAAABAAABAAAwQA=="
        ))

        source, data_start, metadata = safe_load_metadata(fixture)
        loaded = safe_load(fixture)
        assert data_start == 312
        assert len(source) == 352
        assert metadata["__metadata__"] == {
            "format": "polygrad-parity",
            "version": "1",
        }
        assert list(loaded) == ["float", "long", "bool", "scalar"]
        assert loaded["float"].uop.op_name == "RESHAPE"
        assert graph_op_counts(loaded["float"].uop) == {
            "BITCAST": 1,
            "BUFFER": 1,
            "CONST": 5,
            "DEVICE": 1,
            "RESHAPE": 1,
            "SHRINK": 2,
            "STACK": 5,
            "UNIQUE": 1,
        }
        assert loaded["float"].uop.src[0].op_name == "BITCAST"
        assert loaded["float"].uop.src[0].src[0].op_name == "SHRINK"
        stack = [loaded["float"].uop]
        buffer_node = None
        while stack:
            node = stack.pop()
            if node.op_name == "BUFFER":
                buffer_node = node
                break
            stack.extend(node.src)
        assert buffer_node is not None
        graph_device = _ffi._lib.poly_uop_device_name(
            loaded["float"]._ctx, buffer_node.raw
        )
        assert graph_device.decode() == f"DISK:{fixture.resolve()}"
        physical_before = loaded["float"].uop.raw
        kernels_before = GlobalCounters.kernel_count
        loaded["float"].realize()
        assert loaded["float"].uop.raw == physical_before
        assert GlobalCounters.kernel_count == kernels_before
        np.testing.assert_array_equal(
            loaded["float"].numpy(),
            np.asarray([[1.25, -2.5], [3.75, 4.5]], dtype=np.float32),
        )
        assert GlobalCounters.kernel_count == kernels_before
        np.testing.assert_array_equal(
            loaded["long"].numpy(),
            np.asarray([-(1 << 40), (1 << 40) + 3], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            loaded["bool"].numpy(),
            np.asarray([[True, False], [False, True]], dtype=np.bool_),
        )
        assert loaded["scalar"].item() == 2.75
        with pytest.raises(RuntimeError, match="unsupported size in bitcast"):
            source[:3].bitcast("float32")

    def test_torch_load_modern_zip_values_dtypes_and_strides(self, tmp_path):
        # zlib(base64) of the exact 2,496-byte torch.save fixture used by the
        # paired pinned probe. Keeping bytes here avoids a runtime PyTorch test
        # dependency while exercising the real archive/pickle format.
        packed = "eNrFlklv00AUx8d2tpodSkkLlNI13W1naYrEogjlYiTSAFLEgch1TGNhxX32pCpcgEOrcuPECSE4Uc7cUb8CFHHkgoQ4wYEPEGbsEsUuSSyExEijSUZvfv//OO89pyBzIYRiMbRvjKJLCJuWWi0bplIp39XXcd3SypIgZYSsmJqvKFiZW71n5HMX0O2/Ho9YVTUNQ1OxbtZs/rpV0SytclVXMQ9osghMosQTM6pZw/pK3azbwKqOq7lyHeuGzZctbbmuG5Uy1mq2aZXXJB64RKIUJadsEqmsaBByj/B5chN8w93kIVxiSIwAkRJHFVbrEJUjGGJLMpJZmduEHpmTmU3gn1SpkwMYDhbhkGMHW0rNXjVtrQKHq0StGqq6NBGOVB3KUUrhZHYTjskMhR13KScw9BbhZClMgg2ztmJDnwvYs3iN7DUdnnKYEsQJk8PQ7zA3YEBmNuC0yzuD4WwRBh3esmkaNpzz8HJkr8kbcnhJOE94IQzD7k2JxxGykJuOuswxDONFmChF6CNUFUOxIOG5ZQomCYHBMEUIk5PuqWkMM0WYrc8V5Ghs6N2b3V0SSmehbYqNoYGOKbZ8H2smzYh8rq9Njhk6xoZGJTcufumnjulsLzmMprpm9byQz40HzGmEXl5GaGgHodUrCD0lc5t83rlCHT3/+ep1nGjGuziKd3ck5nO96D/qS+31yWg0Gr9ptJSQ86NHY2+3+dA/0k+20ydSDNX6sdX4TgRQqIvWTHetVD6XCNzRSANxnvXD7c9b3fVH0HRH/TXNskkfzOcmghpI8lT9/YuVWyzhsx3VJTTYUX3Oub6tWbpi6A8U2pHLeiWfG+hgRsgIophdzIrZhQVRSiYlSUgKQjqVzKYz5JsoSal0ZlFKU5fC42dLCeIj4bhkqNmmz9aGMbrPeZD3kJ/Y2g/GPDSVRcHajh/ZWlDDHuQnFgVoK8F5VS4AT/TzWgvOy/sWhCf5ea1F5fMXCsBL+nmtReLlfQzCS/l5rWk/4uHdDKMgheYHtmao5AF+6AL8c+0U5EhkZo8wyM02aT2+9SvrrusRdyXn6P8XdGcvgHH2whH3TI8T78b+ApnwOk8="
        archive = tmp_path / "state.pth"
        archive.write_bytes(zlib.decompress(base64.b64decode(packed)))
        state = torch_load(archive)

        assert list(state) == ["contiguous", "transposed", "longs", "bools", "scalar"]
        np.testing.assert_array_equal(
            state["contiguous"].numpy(),
            np.asarray([[1.25, -2.5, 3.75], [4.5, -5.25, 6.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            state["transposed"].numpy(),
            np.asarray([[1.25, 4.5], [-2.5, -5.25], [3.75, 6.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            state["longs"].numpy(),
            np.asarray([-(1 << 40), 0, (1 << 40) + 3], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            state["bools"].numpy(),
            np.asarray([[True, False], [False, True]], dtype=np.bool_),
        )
        assert state["scalar"].shape == ()
        assert state["scalar"].item() == 2.75

    def test_get_state_dict(self):
        m = Linear(3, 2)
        sd = get_state_dict(m)
        assert "weight" in sd
        assert "bias" in sd
        assert sd["weight"].shape == (2, 3)
        assert sd["bias"].shape == (2,)

    def test_get_state_dict_preserves_named_tensor_aliases(self):
        # Pinned nn/state.py:87-107 emits every named path; object identity is
        # not a reason to discard a model-state alias.
        shared = Tensor([1.0, 2.0])
        sd = get_state_dict({"left": shared, "right": shared})
        assert list(sd) == ["left", "right"]
        assert sd["left"] is sd["right"]

    def test_get_state_dict_stops_cycles_without_dropping_diamond_aliases(self):
        shared = Tensor([1.0, 2.0])
        root = {"left": {"weight": shared}, "right": {"weight": shared}}
        root["self"] = root
        sd = get_state_dict(root)
        assert list(sd) == ["left.weight", "right.weight"]
        assert sd["left.weight"] is sd["right.weight"]

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
        assert "l1.weight" in sd
        assert "l1.bias" in sd
        assert "l2.weight" in sd
        assert "l2.bias" in sd

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
        assert any(x is bn.num_batches_tracked for x in params)
        opt = SGD(params, lr=0.1)
        assert [id(x) for x in opt.params] == [id(bn.weight), id(bn.bias)]
        assert [id(x) for x in opt.buffers] == [
            id(bn.num_batches_tracked),
            id(bn.running_mean),
            id(bn.running_var),
        ]


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
        bias = [
            value for name, value in state.items() if value.is_param and "bias" in name
        ]
        non_bias = [
            value
            for name, value in state.items()
            if value.is_param and "bias" not in name
        ]
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
        assert np.abs(g[:, third : 2 * third]).sum() > 0, "k section gradient is zero"
        assert np.abs(g[:, 2 * third :]).sum() > 0, "v section gradient is zero"

        # Compare magnitude across sections — they should be similar (same loss weight)
        norms = [np.linalg.norm(g[:, i * third : (i + 1) * third]) for i in range(3)]
        ratio = max(norms) / min(norms)
        assert ratio < 10, (
            f"Gradient section norms differ too much: {norms} (ratio={ratio:.1f})"
        )

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
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )


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
