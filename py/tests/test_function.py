"""Pinned tinygrad-compatible value-producing ``function`` tests."""

import numpy as np
import pytest

from polygrad import Tensor, function


def _op_counts(*roots):
    seen, counts, stack = set(), {}, [root.uop for root in roots]
    while stack:
        node = stack.pop()
        if node.raw in seen:
            continue
        seen.add(node.raw)
        counts[node.op_name] = counts.get(node.op_name, 0) + 1
        stack.extend(node.src)
    return counts


def test_function_explicit_forward_has_pinned_value_topology():
    # Pinned tinygrad/function.py:39-94 and uop/ops.py:1083-1092 build one
    # TUPLE/FUNCTION and one GETTUPLE selector around ordered PARAM inputs.
    @function
    def explicit(a, b):
        return (a + b).relu()

    a = Tensor([[-1.0, 2.0], [3.0, -4.0]]).realize()
    b = Tensor([[0.5, 1.0], [-2.0, 6.0]]).realize()
    out = explicit(a, b)
    np.testing.assert_array_equal(out.numpy(), [[0.0, 3.0], [1.0, 2.0]])
    counts = _op_counts(out)
    assert counts['FUNCTION'] == counts['TUPLE'] == counts['GETTUPLE'] == 1
    assert counts['PARAM'] == 2


def test_function_tuple_outputs_share_one_function_and_backward():
    # Pinned mixin/gradient.py:18-47 accumulates selector gradients into one
    # TUPLE and emits one backward FUNCTION for the needed input PARAMs.
    @function
    def pair(a):
        return a + 1.0, a * 2.0

    x = Tensor([3.0, 4.0]).realize().requires_grad_(True)
    first, second = pair(x)
    assert first.uop.src[0].raw == second.uop.src[0].raw
    counts = _op_counts(first, second)
    assert counts['FUNCTION'] == 1
    assert counts['TUPLE'] == 1
    assert counts['GETTUPLE'] == 2
    loss = (first.square().sum() + second.square().sum()).backward()
    assert loss.item() == 141.0
    np.testing.assert_array_equal(x.grad.numpy(), [32.0, 42.0])
    assert _op_counts(x.grad)['FUNCTION'] == 2


def test_function_bound_state_backward_matches_pinned_values():
    class Affine:
        def __init__(self):
            self.weight = Tensor([[2.0, -1.0], [0.5, 3.0]]).realize()
            self.bias = Tensor([0.25, -0.5]).realize()
            self.weight.requires_grad_(True)
            self.bias.requires_grad_(True)

        @function
        def __call__(self, x):
            return x @ self.weight + self.bias

    model = Affine()
    x = Tensor([[1.0, 2.0], [-3.0, 0.5]]).realize().requires_grad_(True)
    out = model(x)
    np.testing.assert_array_equal(out.numpy(), [[3.25, 4.5], [-5.5, 4.0]])
    loss = out.square().mean().backward()
    assert loss.item() == 19.265625
    np.testing.assert_array_equal(x.grad.numpy(), [[1.0, 7.5625], [-7.5, 4.625]])
    np.testing.assert_array_equal(
        model.weight.grad.numpy(), [[9.875, -3.75], [1.875, 5.5]])
    np.testing.assert_array_equal(model.bias.grad.numpy(), [-1.125, 4.25])


def test_function_implicit_capture_is_explicitly_gated():
    closure = Tensor([10.0, 20.0]).realize()

    @function
    def rejected(a):
        return a + closure

    with pytest.raises(RuntimeError, match='implicit buffer'):
        rejected(Tensor([1.0, 2.0]).realize())

    @function(allow_implicit=True)
    def accepted(a):
        return a + closure

    np.testing.assert_array_equal(
        accepted(Tensor([1.0, 2.0]).realize()).numpy(), [11.0, 22.0])


@pytest.mark.parametrize('option', ['precompile', 'precompile_backward'])
def test_function_rejects_unimplemented_precompile_modes(option):
    with pytest.raises(NotImplementedError, match='precompile execution'):
        function(lambda x: x, **{option: True})


def test_function_rejects_unimplemented_custom_gradient():
    with pytest.raises(NotImplementedError, match='grad_fxn callbacks'):
        function(lambda x: x, grad_fxn=lambda *_: None)
