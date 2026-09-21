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
