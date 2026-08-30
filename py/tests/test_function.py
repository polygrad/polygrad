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

    x = Tensor([3.0, 4.0]).realize()
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

        @function
        def __call__(self, x):
            return x @ self.weight + self.bias

    model = Affine()
    x = Tensor([[1.0, 2.0], [-3.0, 0.5]]).realize()
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


def test_function_precompile_executes_through_opaque_output_call():
    # Pinned callify.py:101-142 allocates explicit output buffers, rewrites the
    # TUPLE body to STORE into output PARAMs, and turns FUNCTION into an opaque
    # CALL. The public pre-call graph remains the pinned FUNCTION surface.
    weight = Tensor([3.0, 4.0])

    @function(precompile=True, allow_implicit=True)
    def add_weight(x):
        return (x + weight).relu()

    out = add_weight(Tensor([1.0, -3.0]))
    counts = _op_counts(out)
    assert counts['FUNCTION'] == counts['TUPLE'] == counts['GETTUPLE'] == 1
    assert counts['PARAM'] == 2
    np.testing.assert_array_equal(out.numpy(), [4.0, 1.0])


def test_function_precompile_executes_multiple_outputs_in_dependency_order():
    # Pinned schedule/__init__.py:21-68 emits one shared creation COPY before
    # both kernels, and callify.py:101-142 binds the two explicit outputs.
    @function(precompile=True)
    def pair(x):
        return x + 1.0, x * 2.0

    first, second = pair(Tensor([2.0, 3.0]))
    counts = _op_counts(first, second)
    assert counts['FUNCTION'] == counts['TUPLE'] == 1
    assert counts['GETTUPLE'] == 2
    np.testing.assert_array_equal(first.numpy(), [3.0, 4.0])
    np.testing.assert_array_equal(second.numpy(), [4.0, 6.0])


def test_function_nested_precompile_flattens_recursive_linear_calls():
    # Current tinygrad schedule/__init__.py:100-114,181-188 recursively
    # resolves CALL(LINEAR, ...) and flattens the resulting schedule.
    @function(precompile=True)
    def twice(x):
        return x * 2.0

    out = twice(twice(Tensor([1.0, 2.0, 3.0]))) + 1.0
    np.testing.assert_array_equal(out.numpy(), [5.0, 9.0, 13.0])


def test_function_precompile_backward_flag_does_not_change_forward():
    # Pinned function.py:31-79 carries precompile_backward independently of
    # the forward precompile flag. Backward compilation remains a later gate.
    @function(precompile_backward=True)
    def doubled(x):
        return x * 2.0

    np.testing.assert_array_equal(doubled(Tensor([2.0, 3.0])).numpy(), [4.0, 6.0])


def test_function_keeps_distinct_implicit_occurrences_over_one_buffer():
    # Pinned function.py:9-17 appends each matched CONTIGUOUS occurrence and
    # uses buf_uop only to exclude fresh invalid outputs. Two views over one
    # storage must therefore occupy two ordered implicit PARAM slots.
    base = Tensor([1.0, 2.0, 3.0, 4.0]).realize()
    left, right = base[:2].contiguous(), base[2:].contiguous()

    @function(allow_implicit=True)
    def add_views(x):
        return x + left + right

    out = add_views(Tensor([10.0, 20.0]).realize())
    fn = out.uop.src[0]
    assert fn.op_name == 'FUNCTION'
    assert [u.op_name for u in fn.src[1:]] == [
        'BUFFER', 'CONTIGUOUS', 'CONTIGUOUS',
    ]
    np.testing.assert_array_equal(out.numpy(), [14.0, 26.0])


def test_function_rejects_unimplemented_custom_gradient():
    with pytest.raises(NotImplementedError, match='grad_fxn callbacks'):
        function(lambda x: x, grad_fxn=lambda *_: None)
