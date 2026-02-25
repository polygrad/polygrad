"""Tests for polygrad.nn module — layers, optimizers, state dict."""

import math
import numpy as np
import pytest
from polygrad import Tensor
from polygrad.nn import (
    Linear, LayerNorm, GroupNorm, RMSNorm, Embedding, Dropout, Conv2d, BatchNorm,
    SGD, Adam, AdamW,
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

    def test_weight_is_leaf(self):
        m = Linear(2, 3)
        assert m.weight._is_leaf()
        assert m.weight.requires_grad
        assert m.bias._is_leaf()
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


# ── Segment-wise backward ──

class TestSegmentBackward:
    """Tests for gradient computation through realize() boundaries.

    The segment-wise backward processes realize boundaries as segment
    boundaries, computing local VJPs and chaining upstream gradients.
    """

    def test_two_segment_chain(self):
        """Gradient flows through one realize boundary: x→w1→realize→w2→loss."""
        Tensor.manual_seed(42)
        x = Tensor.rand(1, 4).realize()
        w1 = (Tensor.rand(4, 4) * 0.1).realize()
        w1.requires_grad = True
        w1._requires_grad = True
        w2 = (Tensor.rand(2, 4) * 0.1).realize()
        w2.requires_grad = True
        w2._requires_grad = True

        h = x.matmul(w1.T).realize()
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
        """Gradient flows through two realize boundaries."""
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

        h1 = x.matmul(w1.T).realize()
        h2 = h1.matmul(w2.T).realize()
        out = h2.matmul(w3.T)
        loss = out.mean()
        loss.backward()

        for w in [w1, w2, w3]:
            assert w.grad is not None
            assert np.all(np.isfinite(w.grad.numpy()))
            assert np.linalg.norm(w.grad.numpy()) > 0

    def test_segment_matches_direct(self):
        """Segment-wise backward matches direct backward for matmul chain."""
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

        # Segment-wise (with realize)
        w2 = Tensor(w_np.copy()).realize()
        w2.requires_grad = True
        w2._requires_grad = True
        x2 = Tensor(x_np.copy()).realize()
        h2 = x2.matmul(w2.T).realize()
        loss2 = h2.mean()
        loss2.backward()
        segment_grad = w2.grad.numpy()

        np.testing.assert_allclose(segment_grad, direct_grad, rtol=1e-4)

    def test_matmul_transpose_backward(self):
        """Backward through q @ k.T pattern (attention-style)."""
        Tensor.manual_seed(42)
        q = Tensor.rand(1, 2, 4, 8).realize()
        q.requires_grad = True
        q._requires_grad = True
        k = Tensor.rand(1, 2, 4, 8).realize()
        k.requires_grad = True
        k._requires_grad = True

        scores = q.matmul(k.transpose(-2, -1)).realize()
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
        y = x.softmax(axis=-1).realize()
        loss = (y * w).sum()
        loss.backward()

        assert x.grad is not None
        assert np.all(np.isfinite(x.grad.numpy()))

    def test_layernorm_backward_through_realize(self):
        """Backward through LayerNorm (mean→var→normalize, with realizes)."""
        Tensor.manual_seed(42)
        ln = LayerNorm(4)
        x = Tensor.rand(2, 4).realize()
        x.requires_grad = True
        x._requires_grad = True

        y = ln(x).realize()
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert np.all(np.isfinite(x.grad.numpy()))
        assert ln.weight.grad is not None
        assert ln.bias.grad is not None

    def test_mlp_backward(self):
        """MLP with two Linear layers and ReLU, realize between layers."""
        Tensor.manual_seed(42)
        l1 = Linear(4, 8)
        l2 = Linear(8, 2)

        x = Tensor.rand(2, 4)
        h = l1(x).relu().realize()
        out = l2(h)
        loss = out.sum()
        loss.backward()

        assert l1.weight.grad is not None
        assert l2.weight.grad is not None
        assert np.all(np.isfinite(l1.weight.grad.numpy()))
        assert np.all(np.isfinite(l2.weight.grad.numpy()))
        assert np.linalg.norm(l2.weight.grad.numpy()) > 0

    def test_grad_accumulation_multiple_paths(self):
        """A parameter used in multiple segments gets accumulated gradient."""
        Tensor.manual_seed(42)
        w = (Tensor.rand(4, 4) * 0.1).realize()
        w.requires_grad = True
        w._requires_grad = True

        x = Tensor.rand(1, 4).realize()
        h1 = x.matmul(w.T).realize()
        h2 = h1.matmul(w.T).realize()  # w used again
        loss = h2.sum()
        loss.backward()

        assert w.grad is not None
        assert np.all(np.isfinite(w.grad.numpy()))
        # Gradient should be non-zero (accumulated from both segments)
        assert np.linalg.norm(w.grad.numpy()) > 0

    def test_mlp_training_with_sgd(self):
        """End-to-end: MLP training with realize boundaries reduces loss."""
        Tensor.manual_seed(42)
        l1 = Linear(4, 8)
        l2 = Linear(8, 1)
        params = get_parameters(l1) + get_parameters(l2)
        opt = SGD(params, lr=0.01)

        losses = []
        for _ in range(5):
            opt.zero_grad()
            x = Tensor.rand(2, 4)
            h = l1(x).relu().realize()
            out = l2(h)
            target = Tensor([[1.0], [0.0]])
            loss = (out - target).square().sum()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0]

    def test_qkv_diamond_shared_intermediate(self):
        """Verify gradients flow through q/k/v shared qkv intermediate.

        Regression test: segment-wise backward must process shared intermediates
        only after ALL contributing segments have accumulated upstream gradients
        (Kahn's algorithm). Without this, qkv only gets ~1/3 of its gradient.
        """
        Tensor.manual_seed(42)
        x = Tensor.rand(1, 4, 8)
        # w shape (8, 24) so x @ w gives (1, 4, 24) which splits into 3 × 8
        w = (Tensor.rand(8, 24) * 0.1).realize()
        w.requires_grad = True
        w._requires_grad = True

        # Attention-like pattern: shared projection split 3 ways
        qkv = x.dot(w).realize()
        q = qkv.shrink(((0, 1), (0, 4), (0, 8))).realize()
        k = qkv.shrink(((0, 1), (0, 4), (8, 16))).realize()
        v = qkv.shrink(((0, 1), (0, 4), (16, 24))).realize()

        loss = (q * q + k * k + v * v).sum()
        loss.realize()
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
        """QKV diamond pattern: gradient produces loss decrease over steps.

        Functional test: the segment-wise backward through the shared qkv
        intermediate produces usable gradients that reduce the loss.
        """
        Tensor.manual_seed(42)
        x = Tensor.rand(1, 4, 8)
        w = (Tensor.rand(8, 24) * 0.1).realize()
        w.requires_grad = True
        w._requires_grad = True

        losses = []
        for _ in range(5):
            qkv = x.dot(w).realize()
            q = qkv.shrink(((0, 1), (0, 4), (0, 8))).realize()
            k = qkv.shrink(((0, 1), (0, 4), (8, 16))).realize()
            v = qkv.shrink(((0, 1), (0, 4), (16, 24))).realize()
            loss = (q * q + k * k + v * v).sum()
            loss.realize()
            loss.backward()
            losses.append(loss.item())
            # Manual SGD step
            updated = (w - w.grad * 0.01).realize()
            w._uop = updated._uop
            w._buffer = updated._buffer
            w._data = updated._data
            w._inputs = []
            w._grad = None

        assert all(np.isfinite(l) for l in losses), f"NaN/Inf in losses: {losses}"
        assert losses[-1] < losses[0], \
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


class TestGPT2:
    """Smoke tests for GPT-2 model."""

    def test_forward(self):
        """GPT-2 tiny forward produces correct output shape."""
        from polygrad.nn.gpt2 import GPT2
        model = GPT2(n_layers=1, n_heads=2, dim=32, vocab_size=64, max_seq_len=16)
        Tensor.manual_seed(42)
        tokens = Tensor(np.random.randint(0, 64, (1, 4)).astype(np.float32))
        logits = model(tokens)
        assert logits.shape == (1, 4, 64)
        assert np.all(np.isfinite(logits.numpy()))

    def test_backward_all_grads(self):
        """GPT-2 backward produces gradients for all parameters."""
        from polygrad.nn.gpt2 import GPT2
        model = GPT2(n_layers=1, n_heads=2, dim=32, vocab_size=64, max_seq_len=16)
        params = get_parameters(model)
        Tensor.manual_seed(42)
        tokens = Tensor(np.random.randint(0, 64, (1, 4)).astype(np.float32))
        logits = model(tokens)
        loss = logits.mean()
        loss.backward()

        n_with_grad = sum(1 for p in params if p._grad is not None)
        n_finite = sum(1 for p in params if p._grad is not None
                       and np.all(np.isfinite(p._grad.numpy())))
        assert n_with_grad == len(params), f"Only {n_with_grad}/{len(params)} params have grads"
        assert n_finite == len(params), f"Only {n_finite}/{len(params)} params have finite grads"

    def test_training_loss_decreases(self):
        """GPT-2 training loop: loss decreases over iterations."""
        from polygrad.nn.gpt2 import GPT2
        Tensor.manual_seed(42)
        model = GPT2(n_layers=1, n_heads=1, dim=32, vocab_size=50, max_seq_len=16)
        params = get_parameters(model)
        opt = Adam(params, lr=0.001)

        tokens = Tensor(np.array([[5, 12, 23, 38]], dtype=np.float32))
        losses = []
        for _ in range(5):
            logits = model(tokens)
            logits.realize()
            loss = (logits * logits).sum()
            loss.realize()
            loss.backward()
            losses.append(loss.item())
            opt.step()
            opt.zero_grad()

        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        assert all(np.isfinite(l) for l in losses), f"NaN/Inf in losses: {losses}"

    def test_training_benchmark_config(self):
        """GPT-2 training with benchmark config: loss decreases by at least 50%."""
        from polygrad.nn.gpt2 import GPT2
        Tensor.manual_seed(42)
        np.random.seed(42)
        model = GPT2(n_layers=1, n_heads=1, dim=32, vocab_size=50, max_seq_len=8)
        params = get_parameters(model)
        opt = Adam(params, lr=0.001)

        tokens = Tensor(np.random.randint(0, 50, (1, 4)).astype(np.float32))
        losses = []
        for _ in range(10):
            logits = model(tokens)
            logits.realize()
            loss = (logits * logits).sum()
            loss.realize()
            loss.backward()
            losses.append(loss.item())
            opt.step()
            opt.zero_grad()

        assert all(np.isfinite(l) for l in losses), f"NaN/Inf in losses: {losses}"
        assert losses[-1] < losses[0] * 0.5, \
            f"Loss didn't decrease enough: {losses[0]:.2f} -> {losses[-1]:.2f} (need 50% reduction)"


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


# ── CompiledStep ──

class TestCompiledStep:
    def test_compiled_step_basic(self):
        """Compile a simple elementwise op, run twice with different data."""
        from polygrad.tensor import CompiledStep
        from polygrad import _ffi

        ctx = _ffi._lib.poly_ctx_new()
        buf_a = _ffi._lib.poly_buffer_f32(ctx, 4)
        buf_out = _ffi._lib.poly_buffer_f32(ctx, 4)
        one = _ffi._lib.poly_const_float(ctx, 1.0)
        add = _ffi._lib.poly_alu2(ctx, _ffi.OPS['ADD'], buf_a, one)
        store = _ffi._lib.poly_store_val(ctx, buf_out, add)
        sink = _ffi._lib.poly_sink1(ctx, store)

        step_ptr = _ffi._lib.poly_compile_step(ctx, sink)
        assert step_ptr is not None

        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        out = np.zeros(4, dtype=np.float32)
        bindings = [
            (buf_a, a),
            (buf_out, out),
        ]
        step = CompiledStep(step_ptr, bindings)
        assert step.n_kernels >= 1

        # Run 1
        step.run()
        assert approx(out, [2.0, 3.0, 4.0, 5.0])

        # Run 2 with different data
        a[:] = [10.0, 20.0, 30.0, 40.0]
        out[:] = 0.0
        step.run()
        assert approx(out, [11.0, 21.0, 31.0, 41.0])

        _ffi._lib.poly_ctx_destroy(ctx)

    def test_compiled_step_training_sgd(self):
        """Compile a training step with SGD, verify loss decreases."""
        from polygrad.nn import compile_step

        model = Linear(4, 1)
        opt = SGD(get_parameters(model), lr=0.01)

        x = Tensor(np.random.randn(8, 4).astype(np.float32))
        y = Tensor(np.random.randn(8, 1).astype(np.float32))

        def train_step(model, opt, x, y):
            pred = model(x)
            loss = ((pred - y) * (pred - y)).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()
            return loss

        step = compile_step(train_step, model, opt, x, y)
        assert step.n_kernels > 0

        losses = []
        for _ in range(10):
            step.run()
            losses.append(step.loss_value())

        # Loss should decrease
        assert losses[-1] < losses[0], f'Loss did not decrease: {losses[0]} -> {losses[-1]}'

    def test_compiled_step_training_adam(self):
        """Compile a training step with Adam, verify loss decreases."""
        from polygrad.nn import compile_step

        model = Linear(4, 1)
        opt = Adam(get_parameters(model), lr=0.01)

        x = Tensor(np.random.randn(8, 4).astype(np.float32))
        y = Tensor(np.random.randn(8, 1).astype(np.float32))

        def train_step(model, opt, x, y):
            pred = model(x)
            loss = ((pred - y) * (pred - y)).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()
            return loss

        step = compile_step(train_step, model, opt, x, y)
        assert step.n_kernels > 0

        losses = []
        for _ in range(10):
            step.run()
            losses.append(step.loss_value())

        assert losses[-1] < losses[0], f'Loss did not decrease: {losses[0]} -> {losses[-1]}'
