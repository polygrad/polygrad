"""Raw UOp wrappers must match the C elementwise helpers, not raw ALU ops."""
import numpy as np

from polygrad import Tensor, UOp, Variable, _ffi, dtypes


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
