"""Regressions for approved fixes newer than the v0.14 compiler reference."""
import numpy as np
import pytest

from polygrad import Runtime, Tensor, Variable
from polygrad.dtype import dtypes


@pytest.mark.parametrize('name,program', [
    ('elu', lambda x: (x > 0).where(x, 1.0 * ((x - x.relu()).exp() - 1))),
    ('celu', lambda x: 1.0 * (x / 1.0).elu()),
    ('selu', lambda x: 1.0507009873554805 * x.elu(1.6732632423543772)),
    ('relu6', lambda x: (x.relu() < 6).where(x.relu(), 6)),
    ('hardsigmoid', lambda x: (((1 / 6) * x + .5).relu() < 1).where(((1 / 6) * x + .5).relu(), 1)),
    ('asinh', lambda x: (x < 0).where(-1.0, 1.0) *
     (x * (x < 0).where(-1.0, 1.0) + (x.square() + 1).sqrt()).log()),
    ('logsumexp', lambda x: (x - (m := x.max(keepdim=True).detach()).isfinite().where(m, 0)).exp().sum().log() +
     m.isfinite().where(m, 0).squeeze()),
])
def test_backported_composition_graphs(name, program):
    x = Tensor([-2., 0., 3.])
    expected = program(x)
    actual = getattr(x, name)()
    assert actual.uop.raw == expected.uop.raw
    assert actual._core_uop_logical_raw(actual._tensor) == expected._core_uop_logical_raw(expected._tensor)


def test_half_roundtrip_preserves_rounding():
    x = Tensor([70000., 1.0001, -70000.])
    np.testing.assert_array_equal(x.half().float().numpy(), [np.inf, 1., -np.inf])


def test_weak_constant_promotion_preserves_padding():
    x = Tensor(5).reshape(1, 1).expand(1, 2).pad(((0, 2), (0, 0)))
    np.testing.assert_array_equal((x + Tensor([[1], [2], [3]])).numpy(), [[6, 6], [2, 2], [3, 3]])
    np.testing.assert_array_equal((Tensor(5).reshape(1).pad((1, 1)) == 5).numpy(), [False, True, False])
    np.testing.assert_array_equal(Tensor(1).expand(1).cat(Tensor(2).expand(2), Tensor(3).expand(3)).numpy(), [1, 2, 2, 3, 3, 3])


def test_weak_source_keeps_full_width():
    v = Variable('wide_backport', 0, 2**40).bind(2**35 + 7)
    for t in (Tensor(v) // 2**31, (Tensor(v) - 1) // 2**31, Tensor(v).reshape(1) // 2**31):
        assert t.item() == 16


@pytest.mark.parametrize('name', ['fp8e4m3', 'fp8e4m3fnuz', 'fp8e5m2fnuz'])
def test_fp8_pool_and_sort_padding(name):
    dt = getattr(dtypes, name)
    x = Tensor([-3., -1., -2.], dtype=dt)
    for descending in (False, True):
        values, _ = x.sort(descending=descending)
        np.testing.assert_array_equal(values.float().numpy(),
                                      [-1., -2., -3.] if descending else [-3., -2., -1.])
    pooled = x.reshape(1, 1, 1, 3).max_pool2d((1, 2), stride=1, padding=(1, 1, 0, 0))
    np.testing.assert_array_equal(pooled.float().numpy(), [[[[-3., -1., -1., -2.]]]])


def test_infinite_log_reductions():
    x = Tensor([-np.inf, -np.inf, 0.])
    np.testing.assert_array_equal(x[:2].logsumexp().numpy(), -np.inf)
    np.testing.assert_array_equal(x.logcumsumexp(0).numpy(), [-np.inf, -np.inf, 0.])
    a, b = Tensor([-np.inf, np.inf, 1., -np.inf]), Tensor([-np.inf, np.inf, -np.inf, 1.])
    np.testing.assert_array_equal(a.logaddexp(b).numpy(), [-np.inf, np.inf, 1., 1.])


@pytest.mark.parametrize('op,expected', [('elu', 1.), ('celu', 1.), ('selu', 1.0507)])
def test_exponential_activation_large_gradient(op, expected):
    x = Tensor([100., 300.])
    grad = getattr(x, op)().sum().gradient(x)[0]
    np.testing.assert_allclose(grad.numpy(), [expected, expected], rtol=1e-6)


@pytest.mark.parametrize('op,expected', [('erf', 2/np.sqrt(np.pi)), ('asin', 1.), ('asinh', 1.)])
def test_zero_activation_gradient(op, expected):
    x = Tensor([0.])
    np.testing.assert_allclose(getattr(x, op)().sum().gradient(x)[0].numpy(), [expected], rtol=2e-5)


def test_large_activation_values():
    x = Tensor([-1.e10, 0., 1.e10])
    np.testing.assert_allclose(x.asinh().numpy(), np.arcsinh([-1.e10, 0., 1.e10]), rtol=1e-6)
    np.testing.assert_array_equal(x.relu6().numpy(), [0., 0., 6.])
    np.testing.assert_array_equal(x.hardsigmoid().numpy(), [0., .5, 1.])
    boundary = Tensor([0., 6., 7.])
    np.testing.assert_array_equal(boundary.relu6().sum().gradient(boundary)[0].numpy(), [0., 0., 0.])


def test_integer_variance_accumulates_float_squares():
    np.testing.assert_allclose(Tensor([1, 2, 3, 4]).var().numpy(), 5/3, rtol=1e-6)


def test_half_max_gradient_many_ties():
    x = Tensor.ones(70000, dtype=dtypes.float16).contiguous()
    # Division is accumulated in float32, then rounded to the gradient dtype.
    expected = np.full(70000, np.float16(1 / 70000), dtype=np.float16)
    np.testing.assert_array_equal(x.max().gradient(x)[0].numpy(), expected)


@pytest.mark.parametrize('shape,index,value', [
    ((6, 6), (slice(2, 4), slice(3, 5)), [[1., 1.], [1., 1.]]),
    ((6, 6), (slice(2, 4), slice(3, 5)), [1., 2.]),
    ((6, 6), (slice(2, 4), slice(3, 5)), 1.),
    ((6, 6), (3, 4), 1.),
    ((6, 6), (3, None, 4, None), 1.),
    ((4, 4, 4, 4), (Ellipsis, slice(1, 3), slice(None)), 4.),
    ((4, 4, 4, 4), (Ellipsis, slice(1, 3)), 4),
    ((4, 4, 4, 4), (2, slice(1, 3), None, 1), 4),
    ((4, 4, 4, 4), (slice(1, 3), slice(None), slice(0, 4, 2)), 4),
    ((4, 4, 4, 4), (slice(1, 3), slice(None), slice(None), slice(0, 3)), 4),
    ((6, 6), (slice(1, 5, 2), slice(0, 5, 3)), 1.),
    ((6, 6), (slice(5, 1, -2), slice(5, 0, -3)), 1.),
    ((6, 6), (slice(None), slice(0, 6, 2)), 1.),
])
@pytest.mark.parametrize('realize', [False, True])
def test_strided_assignment_updates_base(realize, shape, index, value):
    x = Tensor.zeros(shape).contiguous()
    if realize:
        x.realize()
    x[index] = Tensor(value) if isinstance(value, list) else value
    expected = np.zeros(shape)
    expected[index] = value
    np.testing.assert_array_equal(x.numpy(), expected)


def test_lazy_reshape_assignment_updates_base():
    x = Tensor([1., 2., 3., 4.]).realize()
    x.reshape(2, 2).assign(Tensor([[5., 6.], [7., 8.]]))
    np.testing.assert_array_equal(x.numpy(), [5., 6., 7., 8.])
    y = Tensor([[1., 2., 3., 4.]]).realize()
    y[0] = 7
    np.testing.assert_array_equal(y.numpy(), [[7., 7., 7., 7.]])


@pytest.mark.parametrize('shape,formula', [((3, 3, 4), 'iij->ij'), ((3, 4, 3), 'iji->ij'), ((3, 3, 2), 'iij->j')])
def test_einsum_repeated_nonfinal_axis(shape, formula):
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    np.testing.assert_array_equal(Tensor.einsum(formula, Tensor(data)).numpy(), np.einsum(formula, data))


def test_interpolate_single_aligned_output():
    x = Tensor([[[1., 2., 3.]]])
    np.testing.assert_array_equal(x.interpolate((1,), mode='linear', align_corners=True).numpy(), [[[1.]]])


@pytest.mark.parametrize('count', [1, 4, 17])
def test_x86_nan_and_mixed_width_selections(count):
    with Runtime(device='X86') as rt:
        values = np.resize(np.array([np.nan, 1., 2., -1.], dtype=np.float32), count)
        x = rt.Tensor(values)
        a, b = rt.Tensor(np.arange(count, dtype=np.int32)+10), rt.Tensor(-np.arange(count, dtype=np.int32)-1)
        for op in ['__lt__', '__gt__', '__ne__', '__eq__']:
            np.testing.assert_array_equal(getattr(x, op)(1).where(a, b).numpy(), np.where(getattr(values, op)(1), a.numpy(), b.numpy()))
        p, q = a.cast(dtypes.float64), b.cast(dtypes.float64)
        np.testing.assert_array_equal((x < 1).where(p, q).numpy(), np.where(values < 1, p.numpy(), q.numpy()))


def test_x86_uint8_index_address():
    # Previously faulted in generated machine code; kept isolated by the test runner when reproducing.
    with Runtime(device='X86') as rt:
        x = rt.Tensor([10., 20., 30., 40.])
        i = rt.Tensor([1, 2, 3, 0], dtype=dtypes.uint8)
        np.testing.assert_array_equal(x[i].numpy(), [20., 30., 40., 10.])


def test_x86_half_compare_to_const():
    with Runtime(device='X86') as rt:
        x = rt.Tensor([-1., 1.], dtype=dtypes.float16)
        yes, no = rt.Tensor.ones(2, dtype=dtypes.float16), rt.Tensor.zeros(2, dtype=dtypes.float16)
        np.testing.assert_array_equal((x > 0).where(yes, no).float().numpy(), [0., 1.])


@pytest.mark.parametrize('count', [1, 4, 17])
def test_x86_padded_bool_float_where(count):
    with Runtime(device='X86') as rt:
        data = np.resize([True, False], count)
        condition = rt.Tensor(data).pad(((1, 2),))
        yes, no = rt.Tensor.ones(count + 3), rt.Tensor.full((count + 3,), -2.)
        np.testing.assert_array_equal(condition.where(yes, no).numpy(), np.where(np.pad(data, (1, 2)), 1., -2.))
        for index in (0, 1):
            assert condition[index].where(yes[0], no[0]).item() == (1. if index else -2.)


@pytest.mark.parametrize('name,maximum', [('fp8e4m3', 448.), ('fp8e5m2', np.inf),
                                        ('fp8e4m3fnuz', 240.), ('fp8e5m2fnuz', 57344.)])
def test_fp8_extrema_and_signed_zero(name, maximum):
    dt = getattr(dtypes, name)
    zero = Tensor([0., -0.]).cast(dt).bitcast(dtypes.uint8)
    np.testing.assert_array_equal(zero.numpy(), [0, 0 if 'fnuz' in name else 128])
    x = Tensor([[-1., 0., 1.], [0., 2., 4.]]).cast(dt).contiguous()
    np.testing.assert_array_equal(x.max(axis=1).float().numpy(), [1., 4.])
    np.testing.assert_array_equal(x.min(axis=1).float().numpy(), [-1., 0.])
    # FP8 casts saturate, including e5m2; test infinity by its stored bits.
    if name == 'fp8e5m2':
        np.testing.assert_array_equal(Tensor([252, 124], dtype=dtypes.uint8).bitcast(dt).float().numpy(), [-np.inf, np.inf])
    else:
        np.testing.assert_array_equal(Tensor([-maximum, maximum]).cast(dt).float().numpy(), [-maximum, maximum])


@pytest.mark.parametrize('name,mantissa,bias', [('fp8e4m3', 3, 7), ('fp8e5m2', 2, 15),
                                            ('fp8e4m3fnuz', 3, 8), ('fp8e5m2fnuz', 2, 16)])
def test_fp8_smallest_normals(name, mantissa, bias):
    bits = list(range(1 << mantissa, 2 << mantissa))
    expected = [(1 + i / (1 << mantissa)) * 2**(1 - bias) for i in range(1 << mantissa)]
    np.testing.assert_array_equal(Tensor(bits, dtype=dtypes.uint8).bitcast(getattr(dtypes, name)).float().numpy(), expected)


def test_conv2d_same_padding_with_tuple_dilation():
    from polygrad.nn import Conv2d
    conv = Conv2d(1, 1, (3, 5), padding='same', dilation=(2, 3), bias=False)
    assert conv.padding == (6, 6, 2, 2)
    conv.weight.assign(Tensor.ones(1, 1, 3, 5)).realize()
    x = Tensor.arange(28 * 31).float().reshape(1, 1, 28, 31)
    expected = x.conv2d(conv.weight, padding=(6, 6, 2, 2), dilation=(2, 3))
    actual = conv(x)
    assert actual.shape == x.shape
    np.testing.assert_array_equal(actual.numpy(), expected.numpy())


@pytest.mark.parametrize('name', ['float16', 'bfloat16', 'fp8e4m3', 'fp8e5m2', 'fp8e4m3fnuz', 'fp8e5m2fnuz'])
@pytest.mark.parametrize('args', [(0, 10, 3), (0, 200, 1), (5.5, 175.5, 2.5),
                                  (0., 10., .1), (-3., 3., .03), (0., 200., .3)])
def test_arange_low_precision_accumulates_before_cast(name, args):
    dt = getattr(dtypes, name)
    expected = Tensor.arange(*args).cast(dt).float().numpy()
    np.testing.assert_array_equal(Tensor.arange(*args, dtype=dt).float().numpy(), expected)


def test_repeat_rejects_negative_counts():
    with pytest.raises((ValueError, RuntimeError), match='repeat|negative'):
        Tensor.ones(2, 3).repeat(-1, 2)


def test_selu_default_constants_match_reference():
    x = Tensor([-1., 1.], dtype=dtypes.float64)
    alpha, gamma = 1.6732632423543772, 1.0507009873554805
    np.testing.assert_allclose(x.selu().numpy(), [gamma * alpha * np.expm1(-1), gamma], rtol=1e-12)
