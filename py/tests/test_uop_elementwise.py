"""Raw UOp wrappers must match the C elementwise helpers, not raw ALU ops."""
import numpy as np

from polygrad import Tensor, UOp, Variable, TinyJit, _ffi, dtypes

def test_symbolic_reshape_infers_cancelled_dimension():
    extent = Variable('reshape_tokens', 1, 4)
    for n in (3, 1, 4):
        size = extent.bind(n)
        x = Tensor.full((1, size, 2, 4), 1.0, buffer=False)
        y = x.reshape(1, size, -1)
        assert y.shape[-1] == 8
        assert y.sum().item() == 8 * n

def test_symbolic_arange_preserves_requested_dtype():
    p = Variable('arange_float_extent', 1, 4)
    for n in (3, 1, 4):
        x = Tensor.arange(p.bind(n), dtype=dtypes.float32)
        assert x.dtype == dtypes.float32
        assert x.sum().item() == n * (n - 1) // 2


def test_uop_sub_div_neg_match_c_helpers():
    a = UOp.variable('sp', 0, 12)
    b = UOp.variable('toks', 1, 4)
    end = a + b
    composed = end - a
    via_c = UOp(a.ctx, _ffi._lib.poly_uop_sub(a.ctx, end.raw, a.raw))
    assert composed == via_c
    assert composed.op_name != 'SUB'
    raw = UOp(a.ctx, _ffi._lib.poly_uop_binop(a.ctx, _ffi.OPS['SUB'], end.raw, a.raw))
    assert raw.op_name == 'SUB'
    assert composed != raw

    x = UOp.const(4.0, dtypes.float32)
    y = UOp.const(2.0, dtypes.float32)
    quot = x / y
    via_div = UOp(x.ctx, _ffi._lib.poly_uop_div(x.ctx, x.raw, y.raw))
    assert quot == via_div
    assert quot.op_name != 'FDIV'

    neg = -x
    via_neg = UOp(x.ctx, _ffi._lib.poly_uop_elementwise_neg(x.ctx, x.raw))
    assert neg == via_neg
    assert neg.op_name != 'NEG'


def test_variable_start_const_width_slice():
    x = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    start = Variable('a', 0, 3)
    np.testing.assert_array_equal(x[start.bind(0):start.bind(0) + 3].numpy(), [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(x[start.bind(2):start.bind(2) + 3].numpy(), [3.0, 4.0, 5.0])


def test_repeat_interleave_keeps_symbolic_sequence_axis():
    live = Variable('live', 1, 13)
    k = Tensor.ones(1, 2, 13, 4)[:, :, :live.bind(5), :]
    repeated = k.repeat_interleave(2, -3)
    assert repeated.shape[1] == 4
    q = Tensor.ones(1, 4, 1, 4)
    out = q.scaled_dot_product_attention(k, k, enable_gqa=True)
    arr = out.numpy()
    assert arr.shape == (1, 4, 1, 4)
    assert np.isfinite(arr).all()


def test_symbolic_prefix_sum_and_max_match_constant():
    rng = np.random.RandomState(0)
    x_np = rng.randn(1, 2, 13, 4).astype(np.float32)
    x = Tensor(x_np)
    live = Variable('live', 1, 13)
    a = 5
    np.testing.assert_allclose(x[:, :, :live.bind(a), :].sum(-2).numpy(), x_np[:, :, :a].sum(-2), atol=1e-5)
    np.testing.assert_allclose(x[:, :, :live.bind(a), :].max(-2).numpy(), x_np[:, :, :a].max(-2), atol=1e-5)


def test_symbolic_keepdim_sub_then_sum_matches_constant():
    x_np = np.array(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]],
        np.float32,
    )
    x = Tensor(x_np)
    live = Variable('live', 1, 8)
    a = 5
    xc, xs = x[:, :a], x[:, :live.bind(a)]
    got = (xs - xs.max(-1, True)).sum(-1).numpy()
    want = (xc - xc.max(-1, True)).sum(-1).numpy()
    np.testing.assert_allclose(got, want, atol=1e-5)


def test_symbolic_sdpa_matches_constant_prefix():
    rng = np.random.RandomState(0)
    q_np = rng.randn(1, 2, 1, 4).astype(np.float32)
    k_np = rng.randn(1, 2, 13, 4).astype(np.float32)
    v_np = rng.randn(1, 2, 13, 4).astype(np.float32)
    q, k, v = Tensor(q_np), Tensor(k_np), Tensor(v_np)
    live = Variable('live', 1, 13)
    a = 5
    const = q.scaled_dot_product_attention(k[:, :, :a, :], v[:, :, :a, :], enable_gqa=False).numpy()
    sym = q.scaled_dot_product_attention(
        k[:, :, :live.bind(a), :], v[:, :, :live.bind(a), :], enable_gqa=False
    ).numpy()
    np.testing.assert_allclose(sym, const, atol=1e-5)


def test_symbolic_full_and_arange_extent():
    live = Variable('live', 1, 8)
    np.testing.assert_allclose(Tensor.full((live.bind(5),), 1.0, buffer=False).sum().numpy(), 5.0)
    np.testing.assert_allclose(Tensor.arange(live.bind(5)).sum().numpy(), 10.0)


def test_offset_mask_matches_j_le_p_plus_i():
    for p, n in ((0, 1), (0, 3), (3, 1), (3, 2), (5, 4)):
        s = p + n
        allowed = np.isfinite(
            Tensor.full((1, 1, n, s), float('-inf'), buffer=False).triu(p + 1).numpy()[0, 0]
        )
        i = np.arange(n)[:, None] + p
        j = np.arange(s)[None, :]
        np.testing.assert_array_equal(allowed, j <= i)


def test_symbolic_arange_rebinds_during_jit_replay():
    position = Variable('arange_position', 0, 11)

    @TinyJit
    def run(x, p):
        return (Tensor.arange(p + 1).sum() + x).realize()

    for p in (0, 1, 3, 7, 11):
        assert run(Tensor([0.0]).realize(), position.bind(p)).item() == p * (p + 1) // 2
    assert run.schedule_count == 1


def test_symbolic_offset_mask_rebinds_in_attention():
    position = Variable('mask_position', 0, 11)
    k = Tensor.zeros(1, 1, 13, 2).contiguous().realize()
    v = Tensor(np.arange(26, dtype=np.float32).reshape(1, 1, 13, 2)).realize()

    @TinyJit
    def run(q, p):
        mask = Tensor.full((1, 1, 2, p + 2), float('-inf'), buffer=False).triu(p + 1)
        assert mask.ndim == 4
        return q.scaled_dot_product_attention(k[:, :, :p + 2], v[:, :, :p + 2], attn_mask=mask).realize()

    for p in (0, 1, 3, 7, 11):
        out = run(Tensor.zeros(1, 1, 2, 2).contiguous().realize(), position.bind(p)).numpy()
        np.testing.assert_allclose(out, np.array([p, p + 1, p + 1, p + 2]).reshape(1, 1, 2, 2), atol=1e-5)
    assert run.schedule_count == 1
