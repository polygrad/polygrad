"""Tests for the polygrad Python Tensor class."""

import numpy as np
import pytest

from polygrad import Device, Jit, JitError, Runtime, Tensor, Variable, _ffi, can_run, compile as pg_compile, jit, stats as pg_stats
from polygrad.helpers import Context
from polygrad.uop.ops import AxisType, KernelInfo, UOp


class TestCreation:
    def test_from_list(self):
        t = Tensor([1.0, 2.0, 3.0])
        assert t.shape == (3,)
        np.testing.assert_allclose(t.numpy(), [1, 2, 3])

    def test_from_scalar(self):
        cases = [
            (Tensor(True), (), "bool", True),
            (Tensor(42), (), "int32", 42),
            (Tensor(42.0), (), "float32", 42.0),
            (Tensor(7, device="CUDA"), (), "int32", 7),
            (Tensor(1.5, device="CUDA"), (), "float32", 1.5),
        ]
        for tensor, shape, dtype, value in cases:
            assert tensor.shape == shape
            assert tensor.dtype == dtype
            assert tensor.uop.op_name == "CONST"
            assert tensor.uop_logical.raw == tensor.uop.raw
            if tensor.device == "CPU":
                np.testing.assert_allclose(tensor.numpy(), value)

    def test_internal_scalar_stores_typed_current_root(self):
        cases = [
            (Tensor.empty(2, dtype="bool"), True, "bool"),
            (Tensor.empty(2, dtype="int32"), 7, "int32"),
            (Tensor.empty(2, dtype="float32"), 1, "float32"),
        ]
        for source, value, dtype in cases:
            scalar = source._ensure_tensor(value)
            assert scalar.dtype == dtype
            assert scalar.uop_logical.op_name == "CONST"
            assert scalar.uop_physical.op_name == "CONST"
            assert scalar.uop_logical.raw == scalar.uop_physical.raw

    def test_from_2d(self):
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        assert t.shape == (2, 3)
        np.testing.assert_allclose(t.numpy(), [[1, 2, 3], [4, 5, 6]])

    def test_triu_tril_batched_last_two_dims(self):
        arr = np.arange(1, 19, dtype=np.float32).reshape(2, 3, 3)
        np.testing.assert_allclose(
            Tensor(arr).triu().numpy(),
            np.triu(arr),
        )
        np.testing.assert_allclose(
            Tensor(arr).tril(1).numpy(),
            np.tril(arr, k=1),
        )

        z = Tensor.zeros(5, 0, 3)
        assert z.triu().shape == (5, 0, 3)
        assert z.tril().shape == (5, 0, 3)
        np.testing.assert_allclose(z.triu().numpy(), np.zeros((5, 0, 3), dtype=np.float32))

    def test_empty_creates_unrealized_buffer_placeholder(self):
        t = Tensor.empty((2, 3))
        assert t.shape == (2, 3)
        assert t.uop.has_buffer_identity()
        assert not t.uop.is_realized
        assert t.uop_physical is not None
        assert t.uop_logical.buffer.src[0].raw == t.uop_physical.buffer.src[0].raw
        assert t.uop_logical.buffer.src[1:] == ()
        assert t.uop_physical.buffer.src[1].op_name == 'DEVICE'

    def test_movement_is_realized_through_recursive_base(self):
        source = Tensor([1.0, 2.0, 3.0, 4.0]).realize()
        reshaped = source.reshape(2, 2)
        view = reshaped.flatten().shrink(((1, 3),))

        assert view.uop.op_name == 'SHRINK'
        assert view.uop.base == source.uop.base
        assert reshaped.uop.realized is None
        assert reshaped.uop.is_realized
        assert view.uop.realized is None
        assert view.uop.is_realized

    def test_clone_is_lazy_separate_and_preserves_state(self):
        source = Tensor.empty((4,), dtype='float32', requires_grad=True).is_param_(False)
        source.copy_from(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        source.sum().backward()

        cloned = source.clone()
        assert cloned.uop_logical.op_name == 'AFTER'
        assert [u.op_name for u in cloned.uop_logical.src] == ['BUFFER', 'STORE']
        assert cloned.uop_logical.src[0].buffer.raw != source.uop.buffer.raw
        assert cloned.requires_grad is True
        assert cloned.is_param is False
        assert cloned.grad is not None
        assert cloned.grad.uop_logical.op_name == 'AFTER'
        assert cloned.grad.uop_logical.src[0].buffer.raw != source.grad.uop_logical.src[0].buffer.raw
        np.testing.assert_allclose(cloned.numpy(), [1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(cloned.grad.numpy(), np.ones(4, dtype=np.float32))

    def test_detach_is_a_lazy_graph_boundary(self):
        source = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        detached = source.detach()

        assert detached.shape == source.shape
        assert detached.dtype == source.dtype
        assert detached.device == source.device
        assert detached.requires_grad is False
        assert detached.uop_logical.op_name == 'DETACH'
        assert detached.uop_logical.src[0].raw == source.uop_logical.raw
        if detached.uop_physical is not None:
            assert detached.uop_physical.op_name == 'DETACH'
            assert detached.uop_physical.src[0].raw == source.uop.raw

        detached.sum().backward()
        np.testing.assert_allclose(source.grad.numpy(), np.zeros((2, 2), dtype=np.float32))

    def test_clone_preserves_scalar_shape_and_accepts_device(self):
        source = Tensor.full((), 3.0)
        cloned = source.clone(device='INTERP')
        assert cloned.shape == ()
        assert cloned.device == 'INTERP'
        assert cloned.uop_logical.op_name == 'AFTER'
        assert cloned.item() == pytest.approx(3.0)

    def test_backward_clones_deviceless_grad_and_accumulates_in_place(self):
        x = Tensor.empty((4,), dtype='float32', requires_grad=True)
        loss = x.sum()
        loss.backward()
        first_grad = x.grad
        first_root = first_grad.uop_logical
        first_buffer = first_root.src[0].buffer.raw
        assert first_root.op_name == 'AFTER'
        assert int(_ffi._lib.poly_uop_device(first_grad.uop_logical.raw)) == int(
            _ffi._lib.poly_device_by_name(b'auto')
        )
        assert first_grad.uop_physical is not None
        assert int(_ffi._lib.poly_uop_device(first_grad.uop_physical.raw)) == int(
            _ffi._lib.poly_device_by_name(first_grad.device.lower().encode('utf-8'))
        )

        loss.backward()
        assert x.grad is first_grad
        second_root = x.grad.uop_logical
        assert second_root.src[0].raw == first_root.raw
        assert second_root.src[0].src[0].buffer.raw == first_buffer
        np.testing.assert_allclose(x.grad.numpy(), np.full(4, 2.0, dtype=np.float32))

    def test_backward_through_clone_reaches_source(self):
        source = Tensor.empty((4,), dtype='float32', requires_grad=True)
        source.copy_from(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        cloned = source.clone()
        cloned.sum().backward()
        np.testing.assert_allclose(source.grad.numpy(), np.ones(4, dtype=np.float32))
        np.testing.assert_allclose(cloned.grad.numpy(), np.ones(4, dtype=np.float32))

    def test_backward_retains_distinct_wrappers_sharing_one_uop(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        # Pinned Tensor.__init__ wraps an existing current Tensor.uop directly
        # (tensor.py:92-121); retained logical provenance is not executable.
        y = Tensor(x.uop, requires_grad=True)
        assert x is not y
        assert x.uop.raw == y.uop.raw

        x.sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), np.ones(4, dtype=np.float32))
        np.testing.assert_allclose(y.grad.numpy(), np.ones(4, dtype=np.float32))

    def test_default_context_stats_track_core_work(self):
        before = pg_stats()
        x = Tensor.empty((3,), dtype='float32')
        x.copy_from(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        y = (x + 1).realize()
        np.testing.assert_allclose(y.numpy(), [2.0, 3.0, 4.0])
        after = pg_stats()
        assert after['buffer_write_bytes'] >= before['buffer_write_bytes'] + 12
        assert after['buffer_read_bytes'] >= before['buffer_read_bytes'] + 12
        assert after['launch_count'] >= before['launch_count'] + 1
        assert after['runtime_cache_misses'] >= before['runtime_cache_misses']

    def test_can_run_probes_core_op_shape_support(self):
        assert can_run('add', dtype='float32', shape=(4,))
        assert can_run('matmul', dtype='float32', shapes=((2, 3), (3, 4)))
        assert can_run('gather', dtype='float32', shape=(2, 3))
        assert can_run('sort', dtype='float32', shape=(2, 3))
        assert can_run('argsort', dtype='float32', shape=(2, 3))
        assert can_run('topk', dtype='float32', shape=(2, 3))
        with pytest.raises(ValueError, match='shape is required'):
            can_run('add', dtype='float32')
        with pytest.raises(ValueError, match='shape queries require an op'):
            can_run(shape=(4,), dtype='float32')

    def test_custom_kernel_uop_call(self):
        def add_kernel(c, a, b):
            c, a, b = c.flatten(), a.flatten(), b.flatten()
            i = UOp.range(c.ctx, c.numel(), 0)
            return c[i].store(a[i] + b[i]).end(i).sink()

        a = Tensor([1.0, 2.0, 3.0, 4.0])
        b = Tensor([10.0, 20.0, 30.0, 40.0])
        c = Tensor.empty((4,), dtype='float32')
        out = c.custom_kernel(a, b, fxn=add_kernel)[0]
        np.testing.assert_allclose(out.numpy(), [11.0, 22.0, 33.0, 44.0])

    def test_custom_kernel_multi_output_backward_like_tinygrad(self):
        callback = {}

        def addmul_kernel(c, d, a, b):
            c, d, a, b = c.flatten(), d.flatten(), a.flatten(), b.flatten()
            i = UOp.range(c.ctx, c.numel(), 0)
            store_c = c[i].store(a[i] + b[i])
            store_d = d[i].store(a[i] * b[i])
            return store_c.group(store_d).end(i).sink(arg=KernelInfo(name='addmul'))

        def backward_addmul(grad_c, grad_d, call):
            callback['call'] = call
            _c, _d, a, b = call.src[1:]
            grad_a = (Tensor(grad_c) + Tensor(grad_d) * Tensor(b)).uop
            grad_b = (Tensor(grad_c) + Tensor(grad_d) * Tensor(a)).uop
            return (None, None, grad_a, grad_b)

        rng = np.random.default_rng(7)
        a_np = rng.standard_normal((4, 4), dtype=np.float32)
        b_np = rng.standard_normal((4, 4), dtype=np.float32)

        a_ref = Tensor(a_np, requires_grad=True)
        b_ref = Tensor(b_np, requires_grad=True)
        ((a_ref + b_ref).sum() + (a_ref * b_ref).sum()).backward()

        a = Tensor(a_np, requires_grad=True)
        b = Tensor(b_np, requires_grad=True)
        a.realize(b)
        a_physical, b_physical = a.uop_physical, b.uop_physical
        c, d, _, _ = Tensor.empty((4, 4)).custom_kernel(
            Tensor.empty((4, 4)), a, b, fxn=addmul_kernel, grad_fxn=backward_addmul
        )
        (c.sum() + d.sum()).backward()
        assert callback['call'].src[3] == a_physical
        assert callback['call'].src[4] == b_physical
        np.testing.assert_allclose(a.grad.numpy(), a_ref.grad.numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(b.grad.numpy(), b_ref.grad.numpy(), rtol=1e-5, atol=1e-6)

    def test_custom_kernel_physical_after_preserves_data_gradient(self):
        def identity_kernel(x):
            x = x.flatten()
            i = UOp.range(x.ctx, x.numel(), 0)
            return x[i].store(x[i]).end(i).sink(arg=KernelInfo(name='identity'))

        def backward_identity(grad, call):
            assert call.op_name == 'CALL'
            return (None,)

        x = Tensor.empty((4,), dtype='float32', requires_grad=True)
        x.copy_from(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        y = x.custom_kernel(fxn=identity_kernel, grad_fxn=backward_identity)[0]
        assert y.uop_logical.op_name == 'AFTER'
        assert y.uop_physical.op_name == 'AFTER'
        y.sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), np.ones(4, dtype=np.float32))
        np.testing.assert_allclose(y.grad.numpy(), np.ones(4, dtype=np.float32))

    def test_custom_kernel_separates_output_and_input_gradient_edges(self):
        def identity_kernel(out, x):
            out, x = out.flatten(), x.flatten()
            i = UOp.range(out.ctx, out.numel(), 0)
            return out[i].store(x[i]).end(i).sink(arg=KernelInfo(name='identity_grad_edges'))

        def backward_identity(grad, call):
            assert call.op_name == 'CALL'
            return (None, grad)

        out = Tensor.empty((4,), dtype='float32', requires_grad=True)
        x = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        y = out.custom_kernel(x, fxn=identity_kernel, grad_fxn=backward_identity)[0]
        y.sum().backward()

        np.testing.assert_allclose(out.grad.numpy(), np.ones(4, dtype=np.float32))
        np.testing.assert_allclose(x.grad.numpy(), np.ones(4, dtype=np.float32))
        np.testing.assert_allclose(y.grad.numpy(), np.ones(4, dtype=np.float32))

    def test_custom_kernel_duplicate_output_alias_passes_one_accumulated_upstream(self):
        def identity_kernel(out0, out1, x):
            out0, out1, x = out0.flatten(), out1.flatten(), x.flatten()
            i = UOp.range(out0.ctx, out0.numel(), 0)
            return out0[i].store(x[i]).end(i).sink(arg=KernelInfo(name='duplicate_output_grad'))

        callback_counts = []

        def backward_identity(*args, **kwargs):
            call = kwargs.get('call')
            grads = args
            if call is None:
                *grads, call = args
            callback_counts.append(len(grads))
            return (None, None, grads[0])

        out = Tensor.empty((4,), dtype='float32')
        x = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        y0, y1, _ = out.custom_kernel(
            out, x, fxn=identity_kernel, grad_fxn=backward_identity
        )
        assert y0.uop.raw == y1.uop.raw
        (y0.sum() + y1.sum()).backward()

        assert callback_counts == [1]
        np.testing.assert_allclose(x.grad.numpy(), np.full(4, 2.0, dtype=np.float32))

    def test_custom_kernel_without_grad_fxn_rejects_needed_input_gradient(self):
        def identity_kernel(out, x):
            out, x = out.flatten(), x.flatten()
            i = UOp.range(out.ctx, out.numel(), 0)
            return out[i].store(x[i]).end(i).sink(arg=KernelInfo(name='missing_grad_fxn'))

        out = Tensor.empty((4,), dtype='float32')
        x = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        y = out.custom_kernel(x, fxn=identity_kernel)[0]
        with pytest.raises(
            AssertionError, match='expected TUPLE body for gradient, got Ops.SINK'
        ):
            y.sum().backward()
        assert x.grad is None
        assert y.grad is None

    def test_custom_kernel_callback_is_inactive_behind_stop_gradient_ops(self):
        def identity_kernel(out, x):
            out, x = out.flatten(), x.flatten()
            i = UOp.range(out.ctx, out.numel(), 0)
            return out[i].store(x[i]).end(i).sink(arg=KernelInfo(name='stopped_custom_grad'))

        out = Tensor.empty((4,), dtype='float32', requires_grad=True)
        x = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        y = out.custom_kernel(x, fxn=identity_kernel)[0]
        y.detach().sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), np.zeros(4, dtype=np.float32))
        np.testing.assert_allclose(y.grad.numpy(), np.zeros(4, dtype=np.float32))

        calls = []

        def backward_identity(grad, call):
            calls.append(grad.op_name)
            return (None, (Tensor(grad) + 7).uop)

        out = Tensor.empty((4,), dtype='float32', requires_grad=True)
        x = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        y = out.custom_kernel(x, fxn=identity_kernel, grad_fxn=backward_identity)[0]
        (y < 0).float().sum().backward()
        assert calls == []
        np.testing.assert_allclose(x.grad.numpy(), np.zeros(4, dtype=np.float32))
        np.testing.assert_allclose(y.grad.numpy(), np.zeros(4, dtype=np.float32))

    def test_custom_kernel_reuses_buffers_after_input_update(self):
        def add_kernel(c, a, b):
            c, a, b = c.flatten(), a.flatten(), b.flatten()
            i = UOp.range(c.ctx, c.numel(), 0)
            return c[i].store(a[i] + b[i]).end(i).sink()

        a = Tensor.empty((4,), dtype='float32')
        b = Tensor([10.0, 20.0, 30.0, 40.0])
        c = Tensor.empty((4,), dtype='float32')
        for vals, expected in (
            ([1.0, 2.0, 3.0, 4.0], [11.0, 22.0, 33.0, 44.0]),
            ([5.0, 6.0, 7.0, 8.0], [15.0, 26.0, 37.0, 48.0]),
        ):
            a.copy_from(np.array(vals, dtype=np.float32))
            out = c.custom_kernel(a, b, fxn=add_kernel)[0]
            assert out.uop_logical.op_name == 'AFTER'
            out.realize()
            assert out.uop_physical.has_buffer_identity()
            np.testing.assert_allclose(out.numpy(), expected)

    def test_empty_rejects_name_like_tinygrad(self):
        with pytest.raises(TypeError, match='Tensor.empty does not accept name'):
            Tensor.empty((2, 3), name='z')

    def test_empty_symbolic_shape_survives_realize_and_add(self):
        n = Variable('N', 1, 8)
        x = Tensor.empty(n.bind(4))
        assert not isinstance(x.shape[0], int)

        x.realize()
        assert not isinstance(x.shape[0], int)

        y = x + 1
        assert not isinstance(y.shape[0], int)
        y.realize()
        assert not isinstance(y.shape[0], int)

        with pytest.raises(AssertionError, match='no data if shape is symbolic'):
            y.numpy()

    def test_bound_variable_slice_matches_tinygrad_hlb_batcher(self):
        x = Tensor.arange(20).reshape(10, 2)
        vi = Variable('i', 0, 8)

        for start in (0, 4, 8):
            vib = vi.bind(start)
            y = x[vib:vib + 2]
            assert y.shape == (2, 2)
            np.testing.assert_allclose(
                y.numpy(),
                np.arange(20).reshape(10, 2)[start:start + 2],
            )

        vib = vi.bind(4)
        y = x[vib:vib + 2].realize()
        assert y.shape == (2, 2)
        np.testing.assert_allclose((y + 1).numpy(), [[9, 10], [11, 12]])

    def test_symbolic_prefix_slice_keeps_shape_and_rejects_static_broadcast(self):
        x = Tensor.empty(10, 2)
        n = Variable('n', 1, 4).bind(3)

        y = x[:n]
        assert not isinstance(y.shape[0], int)
        assert y.shape[0] != 3
        assert x[:n, :].shape == y.shape
        with pytest.raises(IndexError, match='shape mismatch'):
            y * Tensor.empty(3, 2)

        i = Variable('i', 0, 6).bind(4)
        assert x[i:i + 2].shape == (2, 2)

    def test_symbolic_expand_broadcast_into_slice_matches_tinygrad(self):
        x = Tensor.empty(10, 2)
        n = Variable('n', 1, 4).bind(3)
        y = x[:n]

        prod = y * Tensor.empty(1, 2)
        assert prod.uop.op_name == 'MUL'
        assert prod.shape == y.shape
        assert not isinstance(prod.shape[0], int)

        summed = y + Tensor.empty(2)
        assert summed.uop.op_name == 'ADD'
        assert summed.shape == y.shape
        assert not isinstance(summed.shape[0], int)

        expanded = Tensor.empty(1, 2).expand(y.shape)
        assert expanded.uop.op_name == 'EXPAND'
        assert expanded.shape == y.shape
        assert not isinstance(expanded.shape[0], int)

    def test_symbolic_empty_broadcast_alu_has_valid_c_shape_and_realizes(self):
        n = Variable('n', 1, 4).bind(3)
        empty = Tensor.empty(n, 10)
        assert empty.uop.op_name == 'SHRINK'
        assert empty.uop.src[0].op_name == 'RESHAPE'
        physical = UOp(
            empty._ctx, _ffi._lib.poly_tensor_uop_physical(empty._tensor)
        )
        assert physical.op_name == 'SHRINK'
        assert physical.src[0].op_name == 'RESHAPE'
        cases = (
            (empty * Tensor.empty(1, 10), 'MUL'),
            (Tensor.empty(n, 10) + Tensor.empty(10), 'ADD'),
        )
        for z, op_name in cases:
            assert z.uop.op_name == op_name
            assert z.shape[1] == 10
            assert not isinstance(z.shape[0], int)
            assert _ffi._lib.poly_uop_ndim(z._ctx, z.uop.raw) == 2
            z.realize()

    def test_symbolic_slicing_static_input_preserves_uop_dims(self):
        x = Tensor.empty(10, 8)
        n = Variable('n', 1, 4).bind(3)
        m = Variable('m', 1, 5).bind(4)
        i = Variable('i', 0, 6).bind(2)

        prefix_prefix = x[:n, :m]
        assert not isinstance(prefix_prefix.shape[0], int)
        assert not isinstance(prefix_prefix.shape[1], int)

        prefix_fixed = x[:n, i:i + 2]
        assert not isinstance(prefix_fixed.shape[0], int)
        assert prefix_fixed.shape[1] == 2

        first = Variable('first', 0, 1).bind(0)
        assert x[:n][first:first + 1].shape == (1, 8)

    def test_static_slice_after_symbolic_prefix_uses_vmax_extent(self):
        x = Tensor.empty(10, 8)
        n = Variable('n', 1, 4).bind(3)
        prefix = x[:n]

        assert not isinstance(prefix.shape[0], int)
        assert prefix[:3].shape == (3, 8)
        assert prefix[:4].shape == (4, 8)
        assert prefix[:5].shape == (4, 8)
        assert prefix[:5].uop.op_name == 'SHRINK'

        later = prefix[:, :4]
        assert not isinstance(later.shape[0], int)
        assert later.shape[1] == 4
        assert later.uop.op_name == 'SHRINK'

        reverse_later = prefix[:, 5:1:-1]
        assert not isinstance(reverse_later.shape[0], int)
        assert reverse_later.shape[1] == 4
        assert reverse_later.uop.op_name == 'FLIP'

        tail = prefix[1:]
        assert not isinstance(tail.shape[0], int)
        assert tail.shape[1] == 8
        assert tail.uop.op_name == 'SHRINK'

        with pytest.raises(RuntimeError, match='symbolic shape not supported'):
            prefix[:, 1:5:2]
        with pytest.raises(TypeError, match='not supported for symbolic shape'):
            prefix[::2]
        with pytest.raises(TypeError, match='not supported for symbolic shape'):
            prefix[::-1]
        with pytest.raises(IndexError, match='out of bounds'):
            prefix[-2:]

        assert prefix[0:4:2].shape == (2, 8)
        assert prefix[3:0:-1].shape == (3, 8)

    def test_instance_cat_and_stack_match_tinygrad_binding(self):
        x = Tensor.arange(6).reshape(2, 3)
        a = Tensor.arange(6, 12).reshape(2, 3)
        b = Tensor.arange(12, 18).reshape(2, 3)

        c = x.cat(a, b, dim=0)
        assert c.shape == (6, 3)
        np.testing.assert_allclose(c.numpy(), np.arange(18).reshape(6, 3))

        s = Tensor([1, 2]).stack(Tensor([3, 4]), dim=0)
        assert s.shape == (2, 2)
        np.testing.assert_allclose(s.numpy(), [[1, 2], [3, 4]])

    def test_hlb_pad_reflect_slice_cat_pattern(self):
        def pad_reflect(x, size=1):
            x = x[..., :, 1:size + 1].flip(-1).cat(
                x,
                x[..., :, -(size + 1):-1].flip(-1),
                dim=-1,
            )
            x = x[..., 1:size + 1, :].flip(-2).cat(
                x,
                x[..., -(size + 1):-1, :].flip(-2),
                dim=-2,
            )
            return x

        x = Tensor.arange(2 * 3 * 4 * 4).reshape(2, 3, 4, 4)
        y = pad_reflect(x, 1)
        expected = np.array([
            [5, 4, 5, 6, 7, 6],
            [1, 0, 1, 2, 3, 2],
            [5, 4, 5, 6, 7, 6],
            [9, 8, 9, 10, 11, 10],
            [13, 12, 13, 14, 15, 14],
            [9, 8, 9, 10, 11, 10],
        ], dtype=np.float32)
        assert y.shape == (2, 3, 6, 6)
        np.testing.assert_allclose(y.numpy().reshape(2, 3, 6, 6)[0, 0], expected)

    def test_int_tensor_python_float_scalar_promotes_like_tinygrad(self):
        x = Tensor([0, 1, 0], dtype='int32')
        assert (x * 0.25).dtype == 'float32'
        assert (0.25 * x).dtype == 'float32'
        assert (x + 0.25).dtype == 'float32'
        np.testing.assert_allclose((x * 0.25).numpy(), [0.0, 0.25, 0.0])
        np.testing.assert_allclose((x + 0.25).numpy(), [0.25, 1.25, 0.25])

    def test_cast_and_bitcast_store_exact_physical_roots(self):
        source = Tensor.arange(4, dtype='uint32')
        casted = source.cast('uint64')
        bitcasted = source.bitcast('float32')

        assert casted.uop_logical.op_name == 'CAST'
        assert casted.uop_physical.op_name == 'CAST'
        assert casted.uop_logical.src[0] == source.uop_logical
        assert casted.uop_physical.src[0] == source.uop_physical
        assert bitcasted.uop_logical.op_name == 'BITCAST'
        assert bitcasted.uop_physical.op_name == 'BITCAST'
        assert bitcasted.uop_logical.src[0] == source.uop_logical
        assert bitcasted.uop_physical.src[0] == source.uop_physical

    def test_zeros(self):
        t = Tensor.zeros(3, 4)
        assert t.shape == (3, 4)
        np.testing.assert_allclose(t.numpy(), np.zeros((3, 4)))

    def test_full_defaults_to_writable_buffer_and_supports_buffer_false(self):
        t = Tensor.full((2,), 3.0).realize()
        assert t.uop.op_name == 'BUFFER'
        t.assign(Tensor([4.0, 5.0], dtype=t.dtype)).realize()
        np.testing.assert_allclose(t.numpy(), [4.0, 5.0])

        broadcast = Tensor.full((2,), 3.0, buffer=False)
        assert broadcast.uop_logical.op_name == 'EXPAND'
        assert broadcast.uop_physical == broadcast.uop_logical
        np.testing.assert_allclose(broadcast.numpy(), [3.0, 3.0])

    def test_ones(self):
        t = Tensor.ones(2, 3)
        assert t.shape == (2, 3)
        np.testing.assert_allclose(t.numpy(), np.ones((2, 3)))

    def test_full(self):
        t = Tensor.full((2, 2), 7.0)
        np.testing.assert_allclose(t.numpy(), np.full((2, 2), 7.0))

    def test_bfloat16_numpy_widens_to_float32(self):
        t = Tensor([1.0, 2.0, 3.0, 4.0], dtype='bfloat16')
        y = (t + t).realize()
        out = y.numpy()
        assert out.dtype == np.float32
        np.testing.assert_allclose(out, [2.0, 4.0, 6.0, 8.0])

    def test_numpy_zero_dim_constructs_scalar_const(self):
        cases = [
            (np.array(7, dtype=np.int32), None, 'int32', 7.0),
            (np.array(1.5, dtype=np.float64), None, 'float64', 1.5),
            (np.array(2.0, dtype=np.float64), 'bfloat16', 'bfloat16', 2.0),
        ]
        for data, dtype, expected_dtype, expected in cases:
            t = Tensor(data, dtype=dtype)
            assert t.shape == ()
            assert t.dtype == expected_dtype
            assert t.uop.op_name == 'CONST'
            np.testing.assert_allclose(t.cast('float32').numpy(), expected)

    def test_numpy_many_batches_realize_and_readback(self):
        t = Tensor([1.0, 2.0, 3.0])
        add, mul = Tensor.numpy_many(t + 1, t * 2)
        assert isinstance(add, np.ndarray)
        assert isinstance(mul, np.ndarray)
        np.testing.assert_allclose(add, [2.0, 3.0, 4.0])
        np.testing.assert_allclose(mul, [2.0, 4.0, 6.0])

        i32 = Tensor(np.array([1, 2, 3], dtype=np.int32)) + Tensor(np.array([10, 20, 30], dtype=np.int32))
        f64 = Tensor(np.array([1.0, 2.0], dtype=np.float64)) + 0.5
        empty = Tensor.zeros(0)
        out_i32, out_f64, out_empty = Tensor.numpy_many([i32, f64, empty])
        assert out_i32.dtype == np.int32
        assert out_f64.dtype == np.float64
        assert out_empty.shape == (0,)
        np.testing.assert_array_equal(out_i32, [11, 22, 33])
        np.testing.assert_allclose(out_f64, [1.5, 2.5])

    def test_arange(self):
        t = Tensor.arange(5)
        assert t.uop_physical == t.uop_logical
        assert t.to('cuda').uop.raw == t.uop.raw
        np.testing.assert_allclose(t.numpy(), np.arange(5, dtype=np.float32))

    def test_rand_manual_seed_deterministic(self):
        Tensor.manual_seed(42)
        a = Tensor.rand(5).numpy()
        b = Tensor.rand(5).numpy()
        Tensor.manual_seed(42)
        c = Tensor.rand(5).numpy()
        d = Tensor.rand(5).numpy()
        np.testing.assert_allclose(a, c)
        np.testing.assert_allclose(b, d)
        assert not np.allclose(a, b)

    def test_randn_manual_seed_deterministic(self):
        Tensor.manual_seed(42)
        a = Tensor.randn(5).numpy()
        b = Tensor.randn(5).numpy()
        Tensor.manual_seed(42)
        c = Tensor.randn(5).numpy()
        d = Tensor.randn(5).numpy()
        np.testing.assert_allclose(a, c)
        np.testing.assert_allclose(b, d)
        assert not np.allclose(a, b)

    def test_randint_range_and_manual_seed(self):
        Tensor.manual_seed(42)
        a = Tensor.randint(5, 10, shape=(64,), dtype='int32').numpy()
        Tensor.manual_seed(42)
        b = Tensor.randint(5, 10, shape=(64,), dtype='int32').numpy()
        np.testing.assert_array_equal(a, b)
        assert np.all(a >= 5)
        assert np.all(a < 10)

    def test_rand_rejects_int_dtype(self):
        with pytest.raises(ValueError, match='rand only supports float dtypes'):
            Tensor.rand(4, dtype='int32')

    def test_item(self):
        t = Tensor([42.0])
        assert t.item() == pytest.approx(42.0)

    def test_copy_from_preserves_buffer_identity_and_updates_jit_input(self):
        x = Tensor.empty((3,), dtype='float32')
        buf = x.uop.buffer
        x.copy_from(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        assert x.uop.buffer == buf
        np.testing.assert_allclose(x.numpy(), [1.0, 2.0, 3.0])

        @Jit
        def f(a):
            return (a + 1).realize()

        np.testing.assert_allclose(f(x).numpy(), [2.0, 3.0, 4.0])
        np.testing.assert_allclose(f(x).numpy(), [2.0, 3.0, 4.0])
        assert f.captured

        x.update_from(np.array([10.0, 20.0, 30.0], dtype=np.float32))
        np.testing.assert_allclose(f(x).numpy(), [11.0, 21.0, 31.0])
        assert x.uop.buffer == buf


class TestJit:
    def test_jit_replays_raw_tensor_realize(self):
        @Jit
        def f(x):
            return (x + 1).realize()

        x0 = Tensor([1.0, 2.0, 3.0]).realize()
        y0 = f(x0)
        np.testing.assert_allclose(y0.numpy(), [2.0, 3.0, 4.0])
        assert not f.captured

        x1 = Tensor([10.0, 20.0, 30.0]).realize()
        y1 = f(x1)
        np.testing.assert_allclose(y1.numpy(), [11.0, 21.0, 31.0])
        assert f.captured
        assert f.schedule_count == 1
        stats = f.stats()
        assert stats['captured']
        assert stats['call_count'] == 2
        assert stats['replay_count'] == 0
        assert stats['schedule_count'] == 1
        assert stats['last_call_ms'] >= 0

        x2 = Tensor([100.0, 200.0, 300.0]).realize()
        y2 = f(x2)
        assert y2 is y1
        np.testing.assert_allclose(y2.numpy(), [101.0, 201.0, 301.0])
        np.testing.assert_allclose(x1.numpy(), [10.0, 20.0, 30.0])
        np.testing.assert_allclose(x2.numpy(), [100.0, 200.0, 300.0])
        assert f.stats()['replay_count'] == 1

    def test_jit_replays_assign_with_current_input(self):
        @Jit
        def f(x):
            return x.assign(x + 1).realize()

        x0 = Tensor([1.0, 2.0, 3.0]).realize()
        y0 = f(x0)
        np.testing.assert_allclose(y0.numpy(), [2.0, 3.0, 4.0])
        np.testing.assert_allclose(x0.numpy(), [2.0, 3.0, 4.0])
        assert not f.captured

        x1 = Tensor([10.0, 20.0, 30.0]).realize()
        y1 = f(x1)
        np.testing.assert_allclose(y1.numpy(), [11.0, 21.0, 31.0])
        np.testing.assert_allclose(x1.numpy(), [11.0, 21.0, 31.0])
        assert f.captured

        x2 = Tensor([100.0, 200.0, 300.0]).realize()
        y2 = f(x2)
        assert y2 is y1
        np.testing.assert_allclose(y2.numpy(), [11.0, 21.0, 31.0])
        np.testing.assert_allclose(x1.numpy(), [11.0, 21.0, 31.0])
        np.testing.assert_allclose(x2.numpy(), [101.0, 201.0, 301.0])

    def test_jit_replays_write_only_assign_with_current_input(self):
        @Jit
        def f(x):
            return x.assign(Tensor([7.0, 8.0, 9.0]).realize()).realize()

        x0 = Tensor([1.0, 2.0, 3.0]).realize()
        y0 = f(x0)
        np.testing.assert_allclose(y0.numpy(), [7.0, 8.0, 9.0])
        np.testing.assert_allclose(x0.numpy(), [7.0, 8.0, 9.0])
        assert not f.captured

        x1 = Tensor([10.0, 20.0, 30.0]).realize()
        y1 = f(x1)
        np.testing.assert_allclose(y1.numpy(), [7.0, 8.0, 9.0])
        np.testing.assert_allclose(x1.numpy(), [7.0, 8.0, 9.0])
        assert f.captured

        x2 = Tensor([100.0, 200.0, 300.0]).realize()
        y2 = f(x2)
        assert y2 is y1
        np.testing.assert_allclose(y2.numpy(), [7.0, 8.0, 9.0])
        np.testing.assert_allclose(x1.numpy(), [7.0, 8.0, 9.0])
        np.testing.assert_allclose(x2.numpy(), [7.0, 8.0, 9.0])

    def test_jit_replays_multiple_realizes(self):
        @Jit
        def f(x):
            y = (x + 1).realize()
            z = (x * 2).realize()
            return y, z

        x0 = Tensor([1.0, 2.0, 3.0]).realize()
        y0, z0 = f(x0)
        np.testing.assert_allclose(y0.numpy(), [2.0, 3.0, 4.0])
        np.testing.assert_allclose(z0.numpy(), [2.0, 4.0, 6.0])
        assert not f.captured

        x1 = Tensor([10.0, 20.0, 30.0]).realize()
        ret1 = f(x1)
        y1, z1 = ret1
        np.testing.assert_allclose(y1.numpy(), [11.0, 21.0, 31.0])
        np.testing.assert_allclose(z1.numpy(), [20.0, 40.0, 60.0])
        assert f.captured
        assert f.schedule_count == 2

        x2 = Tensor([100.0, 200.0, 300.0]).realize()
        ret2 = f(x2)
        assert ret2 is ret1
        np.testing.assert_allclose(y1.numpy(), [101.0, 201.0, 301.0])
        np.testing.assert_allclose(z1.numpy(), [200.0, 400.0, 600.0])
        np.testing.assert_allclose(x2.numpy(), [100.0, 200.0, 300.0])

    def test_jit_preserves_list_and_dict_returns_like_tinygrad(self):
        @Jit
        def as_list(x):
            return [(x + 1).realize()]

        @Jit
        def as_dict(x):
            return {'out': (x * 2).realize()}

        np.testing.assert_allclose(as_list(Tensor([1.0, 2.0]).realize())[0].numpy(), [2.0, 3.0])
        list_ret = as_list(Tensor([10.0, 20.0]).realize())
        assert as_list.captured
        np.testing.assert_allclose(list_ret[0].numpy(), [11.0, 21.0])
        replayed_list = as_list(Tensor([100.0, 200.0]).realize())
        assert replayed_list is list_ret
        np.testing.assert_allclose(list_ret[0].numpy(), [101.0, 201.0])

        np.testing.assert_allclose(as_dict(Tensor([1.0, 2.0]).realize())['out'].numpy(), [2.0, 4.0])
        dict_ret = as_dict(Tensor([10.0, 20.0]).realize())
        assert as_dict.captured
        np.testing.assert_allclose(dict_ret['out'].numpy(), [20.0, 40.0])
        replayed_dict = as_dict(Tensor([100.0, 200.0]).realize())
        assert replayed_dict is dict_ret
        np.testing.assert_allclose(dict_ret['out'].numpy(), [200.0, 400.0])

    def test_jit_rejects_duplicate_input_buffers(self):
        @Jit
        def f(x, y):
            return (x + y).realize()

        x = Tensor([1.0, 2.0, 3.0]).realize()
        with pytest.raises(JitError, match='duplicate inputs'):
            f(x, x)

    def test_jit_rejects_shape_mismatch_after_capture(self):
        @Jit
        def f(x):
            return (x + 1).realize()

        f(Tensor([1.0, 2.0, 3.0]).realize())
        f(Tensor([10.0, 20.0, 30.0]).realize())
        assert f.captured

        with pytest.raises(JitError, match='args mismatch'):
            f(Tensor([100.0, 200.0, 300.0, 400.0]).realize())

    def test_jit_rejects_dtype_mismatch_after_capture(self):
        @Jit
        def f(x):
            return (x + 1).realize()

        f(Tensor([1.0, 2.0, 3.0], dtype='float32').realize())
        f(Tensor([10.0, 20.0, 30.0], dtype='float32').realize())
        assert f.captured

        with pytest.raises(JitError, match='args mismatch'):
            f(Tensor([100, 200, 300], dtype='int32').realize())

    def test_compile_realizes_lazy_return_like_tinygrad(self):
        compiled = pg_compile(lambda x: x * 3 - 1, Tensor([1.0, 2.0, 3.0]))
        out = compiled(Tensor([4.0, 5.0, 6.0]))
        np.testing.assert_allclose(out.numpy(), [11.0, 14.0, 17.0])
        assert compiled.schedule_count == 1

    def test_jit_prune_skips_onetime_side_realize_on_replay(self):
        side = Tensor([-1.0, -1.0, -1.0]).realize()
        seed = Tensor([7.0, 8.0, 9.0]).realize()

        def raw(x):
            side.assign(seed).realize()
            return (x + 1).realize()

        f = Jit(raw, prune=True)
        np.testing.assert_allclose(f(Tensor([1.0, 2.0, 3.0]).realize()).numpy(), [2.0, 3.0, 4.0])
        np.testing.assert_allclose(side.numpy(), [7.0, 8.0, 9.0])
        np.testing.assert_allclose(f(Tensor([10.0, 20.0, 30.0]).realize()).numpy(), [11.0, 21.0, 31.0])
        assert f.captured
        assert f.schedule_count == 2

        side.assign(Tensor([-9.0, -9.0, -9.0])).realize()
        y = f(Tensor([100.0, 200.0, 300.0]).realize())
        np.testing.assert_allclose(y.numpy(), [101.0, 201.0, 301.0])
        np.testing.assert_allclose(side.numpy(), [-9.0, -9.0, -9.0])

    def test_jit_prune_decorator_form(self):
        @jit(prune=True)
        def f(x):
            return (x * 2).realize()

        np.testing.assert_allclose(f(Tensor([1.0, 2.0]).realize()).numpy(), [2.0, 4.0])
        np.testing.assert_allclose(f(Tensor([3.0, 4.0]).realize()).numpy(), [6.0, 8.0])
        assert f.captured

    def test_jit_symbolic_empty_shape_replays_with_runtime_var(self):
        n = Variable('N', 1, 8)

        @Jit
        def f(x):
            return (x + 1).realize()

        y0 = f(Tensor.empty(n.bind(4)))
        assert not f.captured
        assert not isinstance(y0.shape[0], int)

        y1 = f(Tensor.empty(n.bind(4)))
        assert f.captured
        # Pinned TinyJit captures two linears for this exact expression:
        # the explicit realize and its returned symbolic SHRINK view
        # (engine/jit.py:267-293; temp/path_b_probe/tinygrad_symbolic_realize.json).
        assert f.schedule_count == 2
        assert not isinstance(y1.shape[0], int)

        y2 = f(Tensor.empty(n.bind(6)))
        assert y2 is y1
        assert not isinstance(y2.shape[0], int)

    def test_compile_warms_capture_and_replays(self):
        def f(x):
            return (x + 1).realize()

        sample = Tensor([1.0, 2.0, 3.0]).realize()
        compiled = pg_compile(f, [sample])
        assert compiled.schedule_count == 1
        stats = compiled.stats()
        assert stats['capture_runs'] == 2
        assert stats['compile_ms'] >= 0
        assert stats['input_count'] == 1

        out = compiled.run([Tensor([10.0, 20.0, 30.0]).realize()])
        np.testing.assert_allclose(out.numpy(), [11.0, 21.0, 31.0])
        stats = compiled.stats()
        assert stats['run_count'] == 1
        assert stats['last_run_ms'] >= 0
        assert stats['schedule_count'] == 1

        with pytest.raises(JitError, match='args mismatch'):
            compiled.run([Tensor([1.0, 2.0, 3.0, 4.0]).realize()])

        compiled.dispose()
        with pytest.raises(JitError, match='disposed'):
            compiled.run([sample])

    def test_jit_captures_custom_kernel_and_replays_after_input_update(self):
        def add_kernel(c, a, b):
            c, a, b = c.flatten(), a.flatten(), b.flatten()
            i = UOp.range(c.ctx, c.numel(), 0)
            return c[i].store(a[i] + b[i]).end(i).sink()

        @Jit
        def f(a, b):
            c = Tensor.empty((4,), dtype='float32')
            return c.custom_kernel(a, b, fxn=add_kernel)[0]

        a = Tensor.empty((4,), dtype='float32')
        b = Tensor([10.0, 20.0, 30.0, 40.0])
        a.copy_from(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        np.testing.assert_allclose(f(a, b).numpy(), [11.0, 22.0, 33.0, 44.0])
        np.testing.assert_allclose(f(a, b).numpy(), [11.0, 22.0, 33.0, 44.0])
        assert f.captured
        assert f.schedule_count == 1

        a.copy_from(np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32))
        np.testing.assert_allclose(f(a, b).numpy(), [15.0, 26.0, 37.0, 48.0])
        assert f.replay_count == 1

    def test_compile_captures_custom_kernel_and_replays_after_input_update(self):
        def add_kernel(c, a, b):
            c, a, b = c.flatten(), a.flatten(), b.flatten()
            i = UOp.range(c.ctx, c.numel(), 0)
            return c[i].store(a[i] + b[i]).end(i).sink()

        def f(a, b):
            c = Tensor.empty((4,), dtype='float32')
            return c.custom_kernel(a, b, fxn=add_kernel)[0]

        a = Tensor.empty((4,), dtype='float32')
        b = Tensor([10.0, 20.0, 30.0, 40.0])
        a.copy_from(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        compiled = pg_compile(f, [a, b])
        assert compiled.schedule_count == 1

        a.copy_from(np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32))
        out = compiled.run([a, b])
        np.testing.assert_allclose(out.numpy(), [15.0, 26.0, 37.0, 48.0])
        assert compiled.stats()['run_count'] == 1

    def test_compile_captures_custom_kernel_fused_reduction_and_replays_after_input_update(self):
        def summary_kernel(out, a, b):
            out, a, b = out.flatten(), a.flatten(), b.flatten()
            c = UOp.range(out.ctx, 2, 0)
            r = UOp.range(out.ctx, 4, 1, AxisType.REDUCE)
            offset = c * 4 + r
            term = a[offset] * b[r]
            return out[c].store(term.sum(r)).end(c).sink()

        def f(a, b):
            out = Tensor.empty((2,), dtype='float32')
            return out.custom_kernel(a, b, fxn=summary_kernel)[0]

        a = Tensor.empty((8,), dtype='float32')
        b = Tensor([1.0, 2.0, 3.0, 4.0])
        a.copy_from(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32))
        compiled = pg_compile(f, [a, b])
        assert compiled.schedule_count == 1
        np.testing.assert_allclose(compiled.run([a, b]).numpy(), [30.0, 70.0])

        a.copy_from(np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float32))
        np.testing.assert_allclose(compiled.run([a, b]).numpy(), [40.0, 80.0])
        assert compiled.stats()['run_count'] == 2

    def test_compile_captures_custom_kernel_multi_output_fused_reductions(self):
        def summary_kernel(out0, out1, a, b):
            out0, out1, a, b = out0.flatten(), out1.flatten(), a.flatten(), b.flatten()
            c = UOp.range(out0.ctx, 2, 0)
            r = UOp.range(out0.ctx, 4, 1, AxisType.REDUCE)
            term = a[c * 4 + r]
            s0 = term.sum(r)
            s1 = (term * b[r]).sum(r)
            st0 = out0[c].store(s0)
            st1 = out1[c].store(s1)
            return st0.end(c).sink(st1.end(c))

        def f(a, b):
            out0 = Tensor.empty((2,), dtype='float32')
            out1 = Tensor.empty((2,), dtype='float32')
            outs = out0.custom_kernel(out1, a, b, fxn=summary_kernel)
            return [outs[0], outs[1]]

        a = Tensor.empty((8,), dtype='float32')
        b = Tensor([1.0, 2.0, 3.0, 4.0])
        a.copy_from(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32))
        compiled = pg_compile(f, [a, b])
        out0, out1 = compiled.run([a, b])
        np.testing.assert_allclose(out0.numpy(), [10.0, 26.0])
        np.testing.assert_allclose(out1.numpy(), [30.0, 70.0])

        a.copy_from(np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float32))
        out0, out1 = compiled.run([a, b])
        np.testing.assert_allclose(out0.numpy(), [14.0, 30.0])
        np.testing.assert_allclose(out1.numpy(), [40.0, 80.0])
        assert compiled.stats()['run_count'] == 2

    def test_compile_captures_grouped_custom_kernel_compact_summary_with_intercept_reductions(self):
        def summary_kernel(out, x, y):
            out, x, y = out.flatten(), x.flatten(), y.flatten()
            candidates = 2
            rows = 128
            c = UOp.range(out.ctx, candidates, 0)
            r = UOp.range(out.ctx, rows, 1, AxisType.REDUCE)
            one = UOp.const(out.ctx, 1.0)
            xv = x[c * rows + r]
            yv = y[r]
            stats = [
                one.sum(r),
                xv.sum(r),
                (xv * xv).sum(r),
                yv.sum(r),
                (xv * yv).sum(r),
            ]
            stores = [out[c + stat * candidates].store(s) for stat, s in enumerate(stats)]
            return stores[0].group(*stores[1:]).end(c).sink()

        def f(x, y):
            out = Tensor.empty((10,), dtype='float32')
            return out.custom_kernel(x, y, fxn=summary_kernel)[0]

        def expected(xv, yv):
            out = np.zeros((10,), dtype=np.float32)
            out[0:2] = 128.0
            for c in range(2):
                xs = xv[c * 128:(c + 1) * 128]
                out[c + 2] = xs.sum()
                out[c + 4] = (xs * xs).sum()
                out[c + 6] = yv.sum()
                out[c + 8] = (xs * yv).sum()
            return out

        x = Tensor.empty((256,), dtype='float32')
        y_data = np.arange(1, 129, dtype=np.float32)
        y = Tensor(y_data)
        x0 = np.arange(1, 257, dtype=np.float32)
        x.copy_from(x0)
        compiled = pg_compile(f, [x, y])
        np.testing.assert_allclose(compiled.run([x, y]).numpy(), expected(x0, y_data))

        x1 = np.arange(2, 258, dtype=np.float32)
        x.copy_from(x1)
        np.testing.assert_allclose(compiled.run([x, y]).numpy(), expected(x1, y_data))

    def test_compile_captures_sym_style_custom_kernel_fused_summary_reductions(self):
        candidates = 4
        rows = 32
        terms = 5
        stats_per_candidate = 2 + 2 * terms + (terms * (terms + 1)) // 2

        def summary_kernel(out, x, y):
            out, x, y = out.flatten(), x.flatten(), y.flatten()
            c = UOp.range(out.ctx, candidates, 0)
            r = UOp.range(out.ctx, rows, 1, AxisType.REDUCE)
            one = UOp.const(out.ctx, 1.0)
            yv = y[r]

            def term_at(t):
                return x[(c * terms + t) * rows + r]

            stats = [one.sum(r), yv.sum(r)]
            for t in range(terms):
                tv = term_at(t)
                stats.append(tv.sum(r))
                stats.append((tv * yv).sum(r))
            for i in range(terms):
                ti = term_at(i)
                for j in range(i, terms):
                    stats.append((ti * term_at(j)).sum(r))
            stores = [out[c + stat * candidates].store(s) for stat, s in enumerate(stats)]
            return stores[0].group(*stores[1:]).end(c).sink()

        def f(x, y):
            out = Tensor.empty((candidates * stats_per_candidate,), dtype='float32')
            return out.custom_kernel(x, y, fxn=summary_kernel)[0]

        def make_x(shift=0):
            i = np.arange(candidates * terms * rows, dtype=np.float32)
            return np.sin((i + shift) * 0.013).astype(np.float32) + np.cos((np.mod(i, 17)) * 0.07).astype(np.float32) + 0.001 * i

        y_data = np.array([np.cos(i * 0.05) - 0.25 for i in range(rows)], dtype=np.float32)

        def expected(xv):
            out = np.zeros((candidates * stats_per_candidate,), dtype=np.float32)
            for c in range(candidates):
                out[c] = rows
                out[c + candidates] = y_data.sum()
                stat = 2
                for t in range(terms):
                    tv = xv[(c * terms + t) * rows:(c * terms + t + 1) * rows]
                    out[c + stat * candidates] = tv.sum()
                    out[c + (stat + 1) * candidates] = (tv * y_data).sum()
                    stat += 2
                for i in range(terms):
                    ti = xv[(c * terms + i) * rows:(c * terms + i + 1) * rows]
                    for j in range(i, terms):
                        tj = xv[(c * terms + j) * rows:(c * terms + j + 1) * rows]
                        out[c + stat * candidates] = (ti * tj).sum()
                        stat += 1
            return out

        x = Tensor.empty((candidates * terms * rows,), dtype='float32')
        y = Tensor(y_data)
        x0 = make_x(0)
        x.copy_from(x0)
        compiled = pg_compile(f, [x, y])
        np.testing.assert_allclose(compiled.run([x, y]).numpy(), expected(x0), rtol=2e-5, atol=2e-3)

        x1 = make_x(3)
        x.copy_from(x1)
        np.testing.assert_allclose(compiled.run([x, y]).numpy(), expected(x1), rtol=2e-5, atol=2e-3)

    def test_compile_captures_custom_kernel_tinygrad_style_set_accumulator_reduction(self):
        def sum_kernel(out, x):
            out, x = out.flatten(), x.flatten()
            candidates = 4
            rows = 64
            c = UOp.range(out.ctx, candidates, 0)
            r = UOp.range(out.ctx, rows, 1, AxisType.REDUCE)
            acc = out[c].set(0.0)
            acc = acc[c].set(acc.after(r)[c] + x[c * rows + r], end=r)
            return acc.end(c).sink(arg=KernelInfo(name='custom_sum_4_64', opts_to_apply=()))

        def f(x):
            out = Tensor.empty((4,), dtype='float32')
            return out.custom_kernel(x, fxn=sum_kernel)[0]

        x_data = np.arange(1, 257, dtype=np.float32)
        x = Tensor(x_data)
        compiled = pg_compile(f, [x])
        expected = x_data.reshape(4, 64).sum(axis=1)
        np.testing.assert_allclose(compiled.run([x]).numpy(), expected, rtol=1e-5, atol=1e-5)

    def test_compile_captures_staged_custom_kernel_outputs_feeding_another_custom_kernel(self):
        candidates = 4
        rows = 64

        def term_kernel(t0, t1, x, y):
            t0, t1, x, y = t0.flatten(), t1.flatten(), x.flatten(), y.flatten()
            c = UOp.range(t0.ctx, candidates, 0)
            r = UOp.range(t0.ctx, rows, 1)
            idx = c * rows + r
            xv = x[idx]
            yv = y[r]
            st0 = t0[idx].store(xv + yv)
            st1 = t1[idx].store(xv * yv)
            return st0.group(st1).end(c, r).sink(
                arg=KernelInfo(name='custom_stage_terms_4_64', opts_to_apply=())
            )

        def summary_kernel(out, t0, t1):
            out, t0, t1 = out.flatten(), t0.flatten(), t1.flatten()
            c = UOp.range(out.ctx, candidates, 0)
            r = UOp.range(out.ctx, rows, 1, AxisType.REDUCE)
            idx = c * rows + r
            s0 = t0[idx].sum(r)
            s1 = t1[idx].sum(r)
            st0 = out[c].store(s0)
            st1 = out[c + candidates].store(s1)
            return st0.group(st1).end(c).sink(
                arg=KernelInfo(name='custom_stage_summary_4_64', opts_to_apply=())
            )

        def f(x, y):
            t0 = Tensor.empty((candidates * rows,), dtype='float32')
            t1 = Tensor.empty((candidates * rows,), dtype='float32')
            staged = t0.custom_kernel(t1, x, y, fxn=term_kernel)
            out = Tensor.empty((candidates * 2,), dtype='float32')
            return out.custom_kernel(staged[0], staged[1], fxn=summary_kernel)[0]

        x_data = np.array([np.sin(i * 0.01) + i * 0.001 for i in range(candidates * rows)], dtype=np.float32)
        y_data = np.array([np.cos(i * 0.02) - 0.25 for i in range(rows)], dtype=np.float32)
        expected = np.zeros((candidates * 2,), dtype=np.float32)
        for c in range(candidates):
            xs = x_data[c * rows:(c + 1) * rows]
            expected[c] = (xs + y_data).sum()
            expected[c + candidates] = (xs * y_data).sum()

        compiled = pg_compile(f, [Tensor(x_data), Tensor(y_data)])
        np.testing.assert_allclose(compiled.run([Tensor(x_data), Tensor(y_data)]).numpy(), expected, rtol=1e-5, atol=1e-5)

    def test_compiled_custom_kernel_consumer_reads_producer_output_after_readback(self):
        n = 1024

        def producer_kernel(out, x):
            out, x = out.flatten(), x.flatten()
            i = UOp.range(out.ctx, out.numel(), 0)
            return out[i].store(x[i] * 2 + 1).end(i).sink(
                arg=KernelInfo(name='custom_producer_readback_rebind', opts_to_apply=())
            )

        def consumer_kernel(out, y):
            out, y = out.flatten(), y.flatten()
            r = UOp.range(out.ctx, n, 0, AxisType.REDUCE)
            return out[0].store(y[r].sum(r)).sink(
                arg=KernelInfo(name='custom_consumer_readback_rebind', opts_to_apply=())
            )

        x0 = np.arange(n, dtype=np.float32) / 17
        x = Tensor(x0)
        producer = pg_compile(
            lambda tx: Tensor.empty((n,), dtype='float32').custom_kernel(tx, fxn=producer_kernel)[0],
            [x],
        )

        first_y = producer.run([x])
        first_y.realize()
        consumer = pg_compile(
            lambda ty: Tensor.empty((1,), dtype='float32').custom_kernel(ty, fxn=consumer_kernel)[0],
            [first_y],
        )

        first = consumer.run([first_y])
        np.testing.assert_allclose(first.numpy(), [(x0 * 2 + 1).sum()], rtol=1e-5, atol=1e-2)

        x1 = 10 + np.arange(n, dtype=np.float32) / 11
        x.copy_from(x1)
        second_y = producer.run([x])
        second = consumer.run([second_y])
        np.testing.assert_allclose(second.numpy(), [(x1 * 2 + 1).sum()], rtol=1e-5, atol=1e-2)

    def test_custom_kernel_exposes_uop_compare_where_and_unary_methods(self):
        def select_kernel(out, a, b):
            out, a, b = out.flatten(), a.flatten(), b.flatten()
            i = UOp.range(out.ctx, out.numel(), 0)
            av = a[i]
            bv = b[i]
            selected = av.lt(0).where(-av, av.max(bv))
            return out[i].store(selected).end(i).sink()

        out = Tensor.empty((4,), dtype='float32')
        a = Tensor([-3.0, 2.0, 5.0, -1.0])
        b = Tensor([1.0, 4.0, 3.0, 9.0])
        np.testing.assert_allclose(
            out.custom_kernel(a, b, fxn=select_kernel)[0].numpy(),
            [3.0, 4.0, 5.0, 1.0],
        )

    def test_custom_kernel_rejects_bool_index_coordinate_before_codegen(self):
        # Pinned codegen/__init__.py verifies spec_tensor before preprocess:
        # UOp.index(bool) constructs, but executing that body is invalid.
        def invalid_index_kernel(out):
            out = out.flatten()
            zero = UOp.const(out.ctx, 0)
            gate = zero.lt(1)
            bad = out.index(gate)
            assert bad is not None
            return bad.store(out.index(zero)).sink()

        out = Tensor.empty((1,), dtype='float32')
        invalid = out.custom_kernel(fxn=invalid_index_kernel)[0]
        with pytest.raises(RuntimeError, match='poly_realize_tensors'):
            invalid.numpy()

    def test_custom_kernel_exposes_uop_floor_div_and_mod(self):
        from polygrad.dtype import dtypes

        def index_kernel(out):
            out = out.flatten()
            i = UOp.range(out.ctx, out.numel(), 0)
            row = i.floormod(4)
            col = i.floordiv(4)
            return out[i].store((col * 10 + row).cast(dtypes.float32)).end(i).sink()

        out = Tensor.empty((12,), dtype='float32')
        np.testing.assert_allclose(
            out.custom_kernel(fxn=index_kernel)[0].numpy(),
            [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23],
        )

        def signed_kernel(out, x, y):
            out, x, y = out.flatten(), x.flatten(), y.flatten()
            i = UOp.range(out.ctx, x.numel(), 0)
            q = x[i].floordiv(y[i])
            r = x[i].floormod(y[i])
            return out[i].store(q).end(i).sink(out[i + x.numel()].store(r).end(i))

        x = Tensor(np.array([-7, -7, 7, 7, -1, 1, 0], dtype=np.int32))
        y = Tensor(np.array([3, -3, -3, 3, 4, -4, 3], dtype=np.int32))
        out = Tensor.empty((14,), dtype='int32')
        np.testing.assert_array_equal(
            out.custom_kernel(x, y, fxn=signed_kernel)[0].numpy(),
            np.array([-3, 2, -3, 2, -1, -1, 0, 2, -1, -2, 1, 3, -3, 0], dtype=np.int32),
        )

    def test_custom_kernel_numeric_literals_follow_float_operand_dtype(self):
        def literal_kernel(out, x):
            out, x = out.flatten(), x.flatten()
            i = UOp.range(out.ctx, x.numel(), 0)
            same_add = (x[i] - 1) / (x[i] + 1)
            same_mul = (x[i] * 2) / (x[i] * 3)
            s0 = out[i].store(same_add)
            s1 = out[i + x.numel()].store(same_mul)
            return s0.end(i).sink(s1.end(i))

        x_np = (np.arange(16, dtype=np.float32) / 10.0) + 1.0
        out = Tensor.empty((32,), dtype='float32')
        got = out.custom_kernel(Tensor(x_np), fxn=literal_kernel)[0].numpy()
        expected = np.concatenate([
            (x_np - 1.0) / (x_np + 1.0),
            (x_np * 2.0) / (x_np * 3.0),
        ])
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


class TestElementwise:
    def test_add(self):
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])
        c = a + b
        np.testing.assert_allclose(c.numpy(), [5, 7, 9])

    def test_sub(self):
        a = Tensor([10, 20, 30])
        b = Tensor([1, 2, 3])
        np.testing.assert_allclose((a - b).numpy(), [9, 18, 27])

    def test_mul(self):
        a = Tensor([2, 3, 4])
        b = Tensor([5, 6, 7])
        np.testing.assert_allclose((a * b).numpy(), [10, 18, 28])

    def test_div(self):
        a = Tensor([10, 20, 30])
        b = Tensor([2, 4, 5])
        np.testing.assert_allclose((a / b).numpy(), [5, 5, 6])

    def test_neg(self):
        a = Tensor([1, -2, 3])
        np.testing.assert_allclose((-a).numpy(), [-1, 2, -3])

    def test_named_elementwise_methods_match_operator_forms(self):
        a = Tensor([2.0, 4.0, 8.0])
        b = Tensor([1.0, 2.0, 3.0])
        cases = [
            (a.add(b), a + b),
            (a.sub(b), a - b),
            (a.mul(b), a * b),
            (a.div(b), a / b),
            (a.pow(b), a ** b),
            (a.neg(), -a),
        ]
        for got, expected in cases:
            np.testing.assert_allclose(got.numpy(), expected.numpy(), rtol=1e-6, atol=1e-6)

    def test_named_elementwise_reverse_matches_tinygrad(self):
        x = Tensor([2.0, 4.0])
        np.testing.assert_allclose(x.add(10.0, reverse=True).numpy(), (10.0 + x).numpy())
        np.testing.assert_allclose(x.sub(10.0, reverse=True).numpy(), [8.0, 6.0])
        np.testing.assert_allclose(x.mul(10.0, reverse=True).numpy(), (10.0 * x).numpy())
        np.testing.assert_allclose(x.div(10.0, reverse=True).numpy(), [5.0, 2.5])
        np.testing.assert_allclose(x.div(2.0, rounding_mode=None).numpy(), [1.0, 2.0])
        np.testing.assert_allclose(x.pow(3.0, reverse=True).numpy(), [9.0, 81.0])

        moved = Tensor.empty(2, 2, device='cpu').realize().to('cuda').to('cpu')
        for out in (
            3.0 + moved,
            moved.add(3.0, reverse=True),
            3.0 * moved,
            moved.mul(3.0, reverse=True),
        ):
            assert out.uop.src[0].op_name == 'EXPAND'
            assert out.uop.src[1].raw == moved.uop.raw

        class Override(Tensor):
            def add(self, other, reverse=False):
                return ('add', reverse)

            def mul(self, other, reverse=False):
                return ('mul', reverse)

        override = Override([1.0])
        assert 3.0 + override == ('add', True)
        assert 3.0 * override == ('mul', True)

    def test_named_elementwise_bool_scalar_matches_tinygrad(self):
        x = Tensor([True, False], dtype='bool')

        add_int = x.add(2)
        assert add_int.dtype == 'int32'
        np.testing.assert_allclose(add_int.numpy(), [3, 2])

        sub_bool = x.sub(True)
        assert sub_bool.dtype == 'bool'
        assert sub_bool.uop.op_name == 'ADD'
        assert any(src.op_name == 'CMPNE' for src in sub_bool.uop.src)
        np.testing.assert_array_equal(sub_bool.numpy(), [True, False])

        mul_int = x.mul(2)
        assert mul_int.dtype == 'int32'
        np.testing.assert_allclose(mul_int.numpy(), [2, 0])

        pow_int = x.pow(2)
        assert pow_int.dtype == 'int32'
        np.testing.assert_allclose(pow_int.numpy(), [1, 0])

        neg = x.neg()
        assert neg.dtype == 'bool'
        assert neg.uop.op_name == 'CMPNE'
        assert [src.op_name for src in neg.uop.src] == ['BUFFER', 'EXPAND']
        np.testing.assert_array_equal(neg.numpy(), [False, True])

        logical_not = x.logical_not()
        assert logical_not.dtype == 'bool'
        assert logical_not.uop.op_name == 'CMPNE'
        assert [src.op_name for src in logical_not.uop.src] == ['BUFFER', 'EXPAND']
        np.testing.assert_array_equal(logical_not.numpy(), [False, True])

    def test_mixed_dtype_comparison_where_promotes_like_tinygrad(self):
        x = Tensor(np.arange(16, dtype=np.float32).reshape(4, 4))
        out = (Tensor.full((4, 4), 7, dtype='int32') > x).where(
            x, Tensor.full((4, 4), -2, dtype='int32')
        ).sum(axis=0)
        np.testing.assert_allclose(out.numpy(), [0, 2, 4, -3])

    def test_named_integer_true_division_matches_tinygrad(self):
        x = Tensor([3, 4], dtype='int32')

        named = x.div(2)
        assert named.dtype == 'float32'
        np.testing.assert_allclose(named.numpy(), [1.5, 2.0], rtol=1e-6, atol=1e-6)

        operator = x / 2
        assert operator.dtype == 'float32'
        np.testing.assert_allclose(operator.numpy(), [1.5, 2.0], rtol=1e-6, atol=1e-6)

        reverse = x.div(2, reverse=True)
        assert reverse.dtype == 'float32'
        np.testing.assert_allclose(reverse.numpy(), [0.6666667, 0.5], rtol=1e-6, atol=1e-6)

        reverse_operator = 2 / x
        assert reverse_operator.dtype == 'float32'
        np.testing.assert_allclose(reverse_operator.numpy(), [0.6666667, 0.5], rtol=1e-6, atol=1e-6)

        tensor_divisor = x.div(Tensor([2, 2], dtype='int32'))
        assert tensor_divisor.dtype == 'float32'
        np.testing.assert_allclose(tensor_divisor.numpy(), [1.5, 2.0], rtol=1e-6, atol=1e-6)

    def test_named_pow_int_base_float_exponent_matches_tinygrad(self):
        x = Tensor([2, 3], dtype='int32')

        named = x.pow(2.0)
        assert named.dtype == 'int32'
        np.testing.assert_allclose(named.numpy(), [4, 9])

        operator = x ** 2.0
        assert operator.dtype == 'int32'
        np.testing.assert_allclose(operator.numpy(), [4, 9])

        reverse = x.pow(2.0, reverse=True)
        assert reverse.dtype == 'float32'
        np.testing.assert_allclose(reverse.numpy(), [4.0, 8.0])

    def test_named_pow_negative_scalar_int_validation_matches_tinygrad(self):
        x = Tensor([2, 3], dtype='int32')

        with pytest.raises(RuntimeError, match='base needs to be float'):
            x.pow(-1)
        with pytest.raises(RuntimeError, match='base needs to be float'):
            x ** -1

        tensor_exponent = x.pow(Tensor([-1, -2], dtype='int32'))
        assert tensor_exponent.dtype == 'int32'
        np.testing.assert_allclose(tensor_exponent.numpy(), [0, 0])

    def test_scalar_add(self):
        a = Tensor([1, 2, 3])
        c = a + 2.0
        np.testing.assert_allclose(c.numpy(), [3, 4, 5])

    def test_scalar_mul(self):
        a = Tensor([1, 2, 3])
        c = a * 3.0
        np.testing.assert_allclose(c.numpy(), [3, 6, 9])

    def test_exp2(self):
        a = Tensor([0, 1, 2, 3])
        np.testing.assert_allclose(a.exp2().numpy(), [1, 2, 4, 8])

    def test_sqrt(self):
        a = Tensor([1, 4, 9, 16])
        np.testing.assert_allclose(a.sqrt().numpy(), [1, 2, 3, 4])

    def test_chain(self):
        a = Tensor([1, 2, 3, 4])
        b = Tensor([0.5, 0.5, 0.5, 0.5])
        c = (a + Tensor([2, 2, 2, 2])) * b
        np.testing.assert_allclose(c.numpy(), [1.5, 2, 2.5, 3])


class TestMovement:
    def test_reshape(self):
        a = Tensor([1, 2, 3, 4, 5, 6])
        b = a.reshape(2, 3)
        assert b.shape == (2, 3)
        np.testing.assert_allclose(b.numpy(), [[1, 2, 3], [4, 5, 6]])

    def test_reshape_keyword_and_argfix_match_tinygrad(self):
        a = Tensor([1, 2])
        b = a.reshape(shape=[2, 1, -1])
        assert b.shape == (2, 1, 1)
        np.testing.assert_allclose(b.numpy(), [[[1]], [[2]]])
        assert Tensor.empty(2, 1, 1).reshape((2, 1, 1)).shape == (2, 1, 1)
        assert Tensor.empty(2, 1, 1).reshape(None, 1, 1).shape == (2, 1, 1)
        with pytest.raises(ValueError, match="bad arg"):
            Tensor.empty(2, 1, 1).reshape([2, 1], 1)

    def test_reshape_inference_validation_matches_tinygrad(self):
        assert Tensor.empty(6).reshape(2, -1).shape == (2, 3)
        assert Tensor.empty(0).reshape(-1, 3).shape == (0, 3)
        assert Tensor.empty(0).reshape(1, 0).shape == (1, 0)
        assert Tensor.empty(2, 3).reshape(None, 3).shape == (2, 3)

        with pytest.raises(ValueError, match="size mismatch"):
            Tensor.empty(3072, dtype='uint8').reshape(-1, 3073)
        with pytest.raises(ValueError, match="size mismatch"):
            Tensor.empty(5).reshape(2, -1)
        with pytest.raises(RuntimeError, match="only one dimension can be inferred"):
            Tensor.empty(6).reshape(-1, -1)
        with pytest.raises(ZeroDivisionError):
            Tensor.empty(0).reshape(0, -1)

    def test_squeeze_and_integer_index_preserve_scalar_rank(self):
        scalar = Tensor(7)
        assert scalar.squeeze() is scalar
        assert Tensor.empty(1).squeeze(0).shape == ()
        assert Tensor.empty(1, 1).squeeze().shape == ()
        assert Tensor.empty(2, 1).squeeze(1).shape == (2,)

        indexed = Tensor.arange(2, dtype="int32")[0]
        assert indexed.shape == ()
        assert indexed.uop.op_name == "RESHAPE"
        np.testing.assert_allclose(indexed.numpy(), 0)

    def test_permute(self):
        a = Tensor(np.arange(12, dtype=np.float32).reshape(3, 4).tolist())
        b = a.permute(1, 0)
        assert b.shape == (4, 3)
        expected = np.arange(12, dtype=np.float32).reshape(3, 4).T
        np.testing.assert_allclose(b.numpy(), expected)

    def test_expand_negative_and_none_keep_original_dim_like_tinygrad(self):
        x = Tensor.arange(2, dtype='int32').reshape(2, 1, 1, 1)
        y = x.expand(-1, 3, 4, None)
        assert y.shape == (2, 3, 4, 1)
        np.testing.assert_allclose(
            y.numpy().reshape(2, -1)[:, :4],
            np.array([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.int32),
        )

        img = Tensor.arange(2 * 3 * 34 * 34).reshape(2, 3, 34, 34).float()
        low_x = Tensor.randint(2, low=0, high=2).reshape(2, 1, 1, 1)
        idx_x = Tensor.arange(32, dtype='int32').reshape((1, 1, 1, 32))
        crop_idx = (low_x + idx_x).expand(-1, 3, img.shape[2], -1)
        assert crop_idx.shape == (2, 3, 34, 32)
        assert img.gather(-1, crop_idx).shape == (2, 3, 34, 32)

    def test_flip(self):
        a = Tensor([1, 2, 3, 4, 5])
        b = a.flip(0)
        np.testing.assert_allclose(b.numpy(), [5, 4, 3, 2, 1])

    def test_pad(self):
        a = Tensor([1, 2, 3])
        b = a.pad(((1, 1),))
        assert b.shape == (5,)
        np.testing.assert_allclose(b.numpy(), [0, 1, 2, 3, 0])

        x = Tensor(np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3))
        y = x.pad((1, 0, 0, 1))
        assert y.shape == (1, 1, 4, 4)
        np.testing.assert_allclose(
            y.numpy(),
            [[[[0, 0, 1, 2], [0, 3, 4, 5], [0, 6, 7, 8], [0, 0, 0, 0]]]],
        )

        moved = Tensor.arange(12).reshape(3, 4).pad(((-1, 2), (1, -1)))
        assert moved.shape == (4, 4)
        np.testing.assert_allclose(
            moved.numpy(),
            [[0, 4, 5, 6], [0, 8, 9, 10], [0, 0, 0, 0], [0, 0, 0, 0]],
        )

    def test_roll_1d(self):
        a = Tensor.arange(5)
        np.testing.assert_allclose(a.roll(2, 0).numpy(), np.roll(np.arange(5, dtype=np.float32), 2))
        np.testing.assert_allclose(a.roll(-1, 0).numpy(), np.roll(np.arange(5, dtype=np.float32), -1))
        np.testing.assert_allclose(a.roll(7, 0).numpy(), np.roll(np.arange(5, dtype=np.float32), 7))

    def test_roll_dims_none_flattens(self):
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        a = Tensor(arr.tolist())
        b = a.roll(2)
        assert b.shape == (3, 4)
        np.testing.assert_allclose(b.numpy(), np.roll(arr.reshape(-1), 2).reshape(3, 4))

    def test_roll_2d_single_and_tuple_dims(self):
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        a = Tensor(arr.tolist())
        np.testing.assert_allclose(a.roll(1, 0).numpy(), np.roll(arr, 1, axis=0))
        np.testing.assert_allclose(a.roll(-2, -1).numpy(), np.roll(arr, -2, axis=-1))
        np.testing.assert_allclose(a.roll((1, -2), (0, 1)).numpy(), np.roll(arr, (1, -2), axis=(0, 1)))

    def test_roll_rejects_shift_dim_length_mismatch(self):
        with pytest.raises(RuntimeError, match=r"len\(dims\)=2 != len\(shifts\)=1"):
            Tensor.arange(12).reshape(3, 4).roll(1, (0, 1))

    def test_gather_matches_tinygrad_probe(self):
        t = Tensor([[1.0, 2.0], [3.0, 4.0]])
        idx = Tensor(np.array([[0, 0], [1, 0]], dtype=np.int32), dtype='int32')
        out = t.gather(1, idx)
        assert out.shape == (2, 2)
        np.testing.assert_allclose(out.numpy(), [[1.0, 1.0], [4.0, 3.0]])
        x3 = Tensor.arange(24).reshape(2, 3, 4)
        idx3 = Tensor(np.array([[[0, 2], [1, 0]], [[2, 1], [0, 2]]], dtype=np.int32), dtype='int32')
        out3 = x3.gather(1, idx3)
        assert out3.shape == (2, 2, 2)
        np.testing.assert_allclose(out3.numpy(), [[[0, 9], [4, 1]], [[20, 17], [12, 21]]])

    def test_one_hot_matches_tinygrad_probe(self):
        out = Tensor(np.array([0, 2, 1], dtype=np.int32), dtype='int32').one_hot(4)
        assert out.shape == (3, 4)
        assert out.dtype == 'int32'
        np.testing.assert_array_equal(
            out.numpy(),
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]],
        )

    def test_tensor_index_rows_matches_tinygrad_probe(self):
        idx = Tensor(np.array([-1, 0, 2], dtype=np.int32), dtype='int32')
        out = Tensor.arange(12).reshape(3, 4)[idx]
        assert out.shape == (3, 4)
        np.testing.assert_allclose(out.numpy(), [[8, 9, 10, 11], [0, 1, 2, 3], [8, 9, 10, 11]])

    def test_scatter_matches_tinygrad_probe(self):
        idx0 = Tensor(np.array([[0, 1, 2, 0]], dtype=np.int32), dtype='int32')
        src = Tensor.arange(1, 11).reshape(2, 5)
        base0 = Tensor.zeros(3, 5, dtype=src.dtype)
        np.testing.assert_allclose(base0.scatter(0, idx0, src).numpy(), [[1, 0, 0, 4, 0], [0, 2, 0, 0, 0], [0, 0, 3, 0, 0]])

        base = Tensor(np.zeros((3, 5), dtype=np.float32))
        idx1 = Tensor(np.array([[0, 1, 2], [0, 1, 4], [2, 3, 4]], dtype=np.int32), dtype='int32')
        src1 = Tensor(np.array([[1, 2, 3], [6, 7, 8], [9, 10, 11]], dtype=np.float32))
        np.testing.assert_allclose(base.scatter(1, idx1, src1).numpy(), [[1, 2, 3, 0, 0], [6, 7, 0, 0, 8], [0, 0, 9, 10, 11]])

        dup_idx = Tensor(np.array([[1, 1, 2]], dtype=np.int32), dtype='int32')
        dup_src = Tensor(np.array([[7, 9, 8]], dtype=np.float32))
        np.testing.assert_allclose(Tensor([[0.0, 0.0, 0.0, 0.0]]).scatter(1, dup_idx, dup_src).numpy(), [[0, 9, 8, 0]])

        scalar_idx = Tensor(np.array([[2], [3]], dtype=np.int32), dtype='int32')
        np.testing.assert_allclose(Tensor.full((2, 4), 2.0).scatter(1, scalar_idx, 1.23, reduce='add').numpy(), [[2, 2, 3.23, 2], [2, 2, 2, 3.23]], rtol=1e-6)
        np.testing.assert_allclose(Tensor.full((2, 4), 2.0).scatter(1, scalar_idx, 1.23, reduce='multiply').numpy(), [[2, 2, 2.46, 2], [2, 2, 2, 2.46]], rtol=1e-6)

        with pytest.raises(TypeError, match="must be one of"):
            base.scatter(1, idx1, src1, reduce='sum')
        with pytest.raises(TypeError, match="non-scalar src"):
            base.scatter(1, idx1, src1, reduce='add')

    def test_scatter_reduce_matches_tinygrad_probe(self):
        base = Tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        idx = Tensor(np.array([[0, 0, 1, 1, 2, 2, 3, 3, 4, 4]], dtype=np.int32), dtype='int32')
        src = Tensor([[1.0, 6.0, 2.0, 7.0, 3.0, 8.0, 4.0, 9.0, 5.0, 10.0]])
        np.testing.assert_allclose(base.scatter_reduce(1, idx, src, 'sum').numpy(), [[8, 11, 14, 17, 20]])
        np.testing.assert_allclose(base.scatter_reduce(1, idx, src, 'prod').numpy(), [[6, 28, 72, 144, 250]])
        np.testing.assert_allclose(base.scatter_reduce(1, idx, src, 'mean', include_self=False).numpy(), [[3.5, 4.5, 5.5, 6.5, 7.5]])
        extreme_base = Tensor([[-10.0, 20.0, 0.0, 5.0, 10.0]])
        np.testing.assert_allclose(extreme_base.scatter_reduce(1, idx, src, 'amax').numpy(), [[6, 20, 8, 9, 10]])
        np.testing.assert_allclose(extreme_base.scatter_reduce(1, idx, src, 'amin').numpy(), [[-10, 2, 0, 4, 5]])
        with pytest.raises(RuntimeError, match="must be one of"):
            base.scatter_reduce(1, idx, src, 'max')


class TestStepSlicing:
    def test_step2_1d(self):
        a = Tensor([1, 2, 3, 4, 5, 6, 7, 8])
        b = a[::2]
        assert b.shape == (4,)
        np.testing.assert_allclose(b.numpy(), [1, 3, 5, 7])

    def test_step3_1d(self):
        a = Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
        b = a[::3]
        assert b.shape == (3,)
        np.testing.assert_allclose(b.numpy(), [1, 4, 7])

    def test_step2_with_start_stop(self):
        a = Tensor([0, 1, 2, 3, 4, 5, 6, 7])
        b = a[1:7:2]
        assert b.shape == (3,)
        np.testing.assert_allclose(b.numpy(), [1, 3, 5])

    def test_step_non_divisible(self):
        # 7 elements, step 3 -> ceildiv(7,3)=3 elements
        a = Tensor([0, 1, 2, 3, 4, 5, 6])
        b = a[::3]
        assert b.shape == (3,)
        np.testing.assert_allclose(b.numpy(), [0, 3, 6])

    def test_negative_step(self):
        a = Tensor([1, 2, 3, 4, 5, 6])
        b = a[::-1]
        np.testing.assert_allclose(b.numpy(), [6, 5, 4, 3, 2, 1])

    def test_negative_step2(self):
        a = Tensor([1, 2, 3, 4, 5, 6])
        b = a[::-2]
        assert b.shape == (3,)
        np.testing.assert_allclose(b.numpy(), [6, 4, 2])

    def test_step_2d_axis0(self):
        a = Tensor(np.arange(12, dtype=np.float32).reshape(4, 3).tolist())
        b = a[::2]
        assert b.shape == (2, 3)
        np.testing.assert_allclose(b.numpy(), [[0, 1, 2], [6, 7, 8]])

    def test_step_2d_axis1(self):
        a = Tensor(np.arange(12, dtype=np.float32).reshape(3, 4).tolist())
        b = a[:, ::2]
        assert b.shape == (3, 2)
        np.testing.assert_allclose(b.numpy(), [[0, 2], [4, 6], [8, 10]])

    def test_step_2d_both(self):
        a = Tensor(np.arange(16, dtype=np.float32).reshape(4, 4).tolist())
        b = a[::2, ::2]
        assert b.shape == (2, 2)
        np.testing.assert_allclose(b.numpy(), [[0, 2], [8, 10]])

    def test_step_backward(self):
        a = Tensor([1, 2, 3, 4, 5, 6, 7, 8], requires_grad=True)
        b = a[::2]  # [1, 3, 5, 7]
        c = b.sum()
        c.backward()
        # grad should be [1,0,1,0,1,0,1,0]
        np.testing.assert_allclose(a.grad.numpy(), [1, 0, 1, 0, 1, 0, 1, 0])


class TestReduce:
    def test_sum_all(self):
        a = Tensor([1, 2, 3, 4])
        s = a.sum()
        assert s.item() == pytest.approx(10.0)

    def test_sum_axis(self):
        a = Tensor([[1, 2, 3], [4, 5, 6]])
        s = a.reshape(2, 3).sum(axis=1)
        np.testing.assert_allclose(s.numpy(), [6, 15])

    def test_argmax_matches_tinygrad_probe(self):
        a = Tensor([[1.2, 0.5, 1.2], [2.2, 1.9, 0.0]])
        np.testing.assert_allclose(a.argmax(axis=1).numpy(), [0, 0])
        np.testing.assert_allclose(a.argmax(axis=1, keepdim=True).numpy(), [[0], [0]])
        assert a.argmax().item() == 3
        singleton = Tensor(np.arange(6, dtype=np.float32).reshape(2, 1, 3))
        np.testing.assert_array_equal(singleton.argmax(axis=1).numpy(), np.zeros((2, 3)))
        empty = Tensor.empty(2, 0, 3, device="CPU")
        np.testing.assert_array_equal(
            empty.argmax(axis=1).numpy(),
            np.full((2, 3), np.iinfo(np.int32).min, dtype=np.int32),
        )

    def test_sort_argsort_topk_match_tinygrad_probe(self):
        x = Tensor([[0.1, 0.5, 1.2, 3.4, 2.1], [2.2, 1.9, 0.3, 4.5, 0.8]])
        vals, idx = x.sort(1, False)
        np.testing.assert_allclose(vals.numpy(), [[0.1, 0.5, 1.2, 2.1, 3.4],
                                                  [0.3, 0.8, 1.9, 2.2, 4.5]], rtol=1e-6)
        np.testing.assert_array_equal(idx.numpy(), [[0, 1, 2, 4, 3], [2, 4, 1, 0, 3]])

        vals, idx = x.topk(2, dim=1)
        np.testing.assert_allclose(vals.numpy(), [[3.4, 2.1], [4.5, 2.2]], rtol=1e-6)
        np.testing.assert_array_equal(idx.numpy(), [[3, 4], [3, 0]])

        vals, idx = x.topk(2, dim=1, largest=False)
        np.testing.assert_allclose(vals.numpy(), [[0.1, 0.5], [0.3, 0.8]], rtol=1e-6)
        np.testing.assert_array_equal(idx.numpy(), [[0, 1], [2, 4]])

        t = Tensor([[2, 3, 4, 1], [1, 4, 3, 2]])
        np.testing.assert_array_equal(t.argsort().numpy(), [[3, 0, 1, 2], [0, 3, 2, 1]])

    def test_topk_tie_order_and_errors_match_tinygrad_probe(self):
        tie = Tensor([[1.0, 1.0, 0.0, 1.0]])
        vals, idx = tie.topk(3, dim=1)
        np.testing.assert_allclose(vals.numpy(), [[1.0, 1.0, 1.0]])
        np.testing.assert_array_equal(idx.numpy(), [[0, 1, 3]])
        with pytest.raises(ValueError, match='selected index k=6 is out of range'):
            Tensor([[0.1, 0.2]]).topk(6, dim=1)
        with pytest.raises(NotImplementedError, match='sorted_=False'):
            Tensor([[0.1, 0.2]]).topk(1, dim=1, sorted_=False)


class TestMatmulAndLoss:
    def test_conv2d_stride_padding_matches_reference(self):
        x = np.arange(1 * 2 * 4 * 5, dtype=np.float32).reshape(1, 2, 4, 5) / 7
        w = (np.arange(3 * 2 * 2 * 3, dtype=np.float32).reshape(3, 2, 2, 3) - 5) / 11
        b = np.array([0.5, -1.0, 2.0], dtype=np.float32)

        out = np.zeros((1, 3, 6, 2), dtype=np.float32)
        for oc in range(3):
            for oy in range(6):
                for ox in range(2):
                    acc = b[oc]
                    for ic in range(2):
                        for ky in range(2):
                            for kx in range(3):
                                iy = oy + ky - 2
                                ix = ox * 2 + kx - 1
                                if 0 <= iy < 4 and 0 <= ix < 5:
                                    acc += x[0, ic, iy, ix] * w[oc, ic, ky, kx]
                    out[0, oc, oy, ox] = acc
        np.testing.assert_allclose(
            Tensor(x).conv2d(Tensor(w), Tensor(b), stride=(1, 2), padding=(1, 0, 2, 1)).numpy(),
            out,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_conv2d_padded_3x3_4x4_devectorize_regression(self):
        # tinygrad: arange(16).reshape(1,1,4,4).conv2d(ones(1,1,3,3), padding=1)
        # This crosses the 128-lane late-devectorize threshold.
        padded = Tensor(np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)).conv2d(
            Tensor(np.ones((1, 1, 3, 3), dtype=np.float32)),
            padding=1,
        )
        np.testing.assert_allclose(
            padded.numpy(),
            [[[[10, 18, 24, 18], [27, 45, 54, 39], [51, 81, 90, 63], [42, 66, 72, 50]]]],
            rtol=1e-5,
            atol=1e-5,
        )

    def test_max_pool2d_padding_matches_reference(self):
        pool_x = np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3)
        np.testing.assert_allclose(
            Tensor(pool_x).max_pool2d(2, stride=1, padding=1).numpy(),
            [[[[0, 1, 2, 2], [3, 4, 5, 5], [6, 7, 8, 8], [6, 7, 8, 8]]]],
        )

    def test_batchnorm_multi_axis_matches_reference(self):
        bn_x = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5) / 10
        mean = np.arange(8, dtype=np.float32).reshape(2, 4) / 20
        inv = np.ones((2, 4), dtype=np.float32) * 0.25
        weight = np.linspace(0.5, 1.2, 8, dtype=np.float32).reshape(2, 4)
        bias = np.linspace(-0.3, 0.4, 8, dtype=np.float32).reshape(2, 4)
        expected = (((bn_x - mean[:, None, :, None]) * weight[:, None, :, None]) *
                    inv[:, None, :, None] + bias[:, None, :, None])
        np.testing.assert_allclose(
            Tensor(bn_x).batchnorm(Tensor(weight), Tensor(bias), Tensor(mean), Tensor(inv), axis=(0, 2)).numpy(),
            expected,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_einsum_c_api_wrapper(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([[5.0, 6.0], [7.0, 8.0]])
        out = Tensor.einsum('ij,jk->ik', a, b)
        assert out.shape == (2, 2)
        np.testing.assert_allclose(out.numpy(), np.array([[19.0, 22.0], [43.0, 50.0]], dtype=np.float32))

        with pytest.raises(ValueError, match='poly_einsum failed'):
            Tensor.einsum('a->' + ('a' * 80), Tensor([1.0]))

        with Runtime(device='cpu') as runtime_a, Runtime(device='cpu') as runtime_b:
            with pytest.raises(ValueError, match='same Polygrad context'):
                runtime_a.Tensor.einsum(
                    'i,i->',
                    runtime_a.Tensor([1.0, 2.0, 3.0]),
                    runtime_b.Tensor([4.0, 5.0, 6.0]),
                )

    def test_rearrange_rejects_malformed_formula(self):
        x = Tensor([1.0, 2.0, 3.0])
        for formula in ('invalid', 'a' * 300 + '->a', 'a->a->a', '((a))->a'):
            with pytest.raises(ValueError, match='poly_rearrange failed'):
                x.rearrange(formula)

    def test_matmul_shape_mismatch_raises(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([[1.0, 2.0, 3.0]])
        with pytest.raises(ValueError, match='cannot dot'):
            a @ b

    def test_matmul_broadcast_batch_values(self):
        a_np = np.array([
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ], dtype=np.float32)
        b_np = np.array([
            [[1.0, 10.0], [100.0, 1000.0]],
        ], dtype=np.float32)
        out = Tensor(a_np) @ Tensor(b_np)
        assert out.shape == (2, 2, 2)
        np.testing.assert_allclose(out.numpy(), np.matmul(a_np, b_np), rtol=1e-6)

    def test_matmul_broadcast_mismatch_raises(self):
        a = Tensor(np.zeros((2, 3, 4), dtype=np.float32))
        b = Tensor(np.zeros((5, 4, 6), dtype=np.float32))
        with pytest.raises(ValueError, match='cannot dot'):
            a @ b

    def test_qr_matches_tinygrad_probe(self):
        cases = [
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
            np.array([
                [[1.0, 2.0], [3.0, 4.0]],
                [[2.0, 0.0], [0.0, 2.0]],
            ], dtype=np.float32),
            np.array([
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[2.0, 1.0], [0.0, 3.0], [4.0, 5.0]],
            ], dtype=np.float32),
            np.array([
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[2.0, 1.0, 0.0], [0.0, 3.0, 4.0]],
            ], dtype=np.float32),
        ]
        for arr in cases:
            q, r = Tensor(arr).qr()
            assert q.shape == arr.shape[:-2] + (arr.shape[-2], arr.shape[-2])
            assert r.shape == arr.shape
            q_np, r_np = q.numpy(), r.numpy()
            assert not np.isnan(q_np).any()
            assert not np.isnan(r_np).any()
            np.testing.assert_allclose(np.matmul(q_np, r_np), arr, rtol=1e-4, atol=1e-4)

    def test_qr_zero_column_and_int_promote_match_tinygrad_probe(self):
        arr = np.array([[0.0, 1.0], [0.0, 2.0]], dtype=np.float32)
        q, r = Tensor(arr).qr()
        np.testing.assert_allclose(q.numpy() @ r.numpy(), arr, rtol=1e-4, atol=1e-4)
        assert not np.isnan(q.numpy()).any()
        assert not np.isnan(r.numpy()).any()

        qi, ri = Tensor(np.array([[1, 2], [3, 4]], dtype=np.int32)).qr()
        assert qi.dtype == 'float32'
        assert ri.dtype == 'float32'
        np.testing.assert_allclose(qi.numpy() @ ri.numpy(), [[1, 2], [3, 4]], rtol=1e-4, atol=1e-4)

    def test_qr_reduced_and_r_modes_match_numpy_shapes(self):
        cases = [
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
            np.array([
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[2.0, 1.0], [0.0, 3.0], [4.0, 5.0]],
            ], dtype=np.float32),
        ]
        for arr in cases:
            q, r = Tensor(arr).qr(mode='reduced')
            nq, nr = np.linalg.qr(arr, mode='reduced')
            assert q.shape == nq.shape
            assert r.shape == nr.shape
            np.testing.assert_allclose(np.matmul(q.numpy(), r.numpy()), arr, rtol=1e-4, atol=1e-4)

            r_only = Tensor(arr).qr(mode='r')
            assert r_only.shape == nr.shape

        with pytest.raises(ValueError, match='qr mode'):
            Tensor(cases[0]).qr(mode='raw')

    def test_triangular_solve_matches_numpy_torch_probe(self):
        torch = pytest.importorskip('torch')
        lower = np.array([[2.0, 0.0, 0.0], [1.0, 3.0, 0.0], [-2.0, 0.5, 4.0]], dtype=np.float32)
        upper = np.array([[2.0, -1.0, 0.5], [0.0, 3.0, 2.0], [0.0, 0.0, 4.0]], dtype=np.float32)
        b_vec = np.array([2.0, 7.0, 9.0], dtype=np.float32)
        b_mat = np.array([[2.0, 1.0], [7.0, 2.0], [9.0, 3.0]], dtype=np.float32)
        lower_batched = np.stack([lower, lower + np.eye(3, dtype=np.float32)], axis=0)
        b_batched = np.stack([b_mat, b_mat + 1.0], axis=0)
        b_broadcast_vec = np.array([2.0, 7.0, 9.0], dtype=np.float32)
        cases = [
            (lower, b_vec, False, False, False),
            (lower, b_mat, False, False, False),
            (upper, b_mat, True, False, False),
            (lower, b_mat, False, True, False),
            (upper, b_mat, True, True, False),
            (lower + np.diag([3.0, 4.0, 5.0]).astype(np.float32), b_mat, False, False, True),
            (lower_batched, b_batched, False, False, False),
            (lower_batched, b_broadcast_vec, False, False, False),
        ]
        for a, b, upper_flag, transpose_a, unit_diagonal in cases:
            eff_a = np.swapaxes(a, -1, -2) if transpose_a else a
            eff_upper = (not upper_flag) if transpose_a else upper_flag
            np_a = eff_a.copy()
            if unit_diagonal:
                diag = np.arange(np_a.shape[-1])
                np_a[..., diag, diag] = 1.0
            expected = np.linalg.solve(np_a, b)

            got = Tensor(a).triangular_solve(
                Tensor(b),
                upper=upper_flag,
                transpose_a=transpose_a,
                unit_diagonal=unit_diagonal,
            )
            assert got.shape == expected.shape
            np.testing.assert_allclose(got.numpy(), expected, rtol=1e-5, atol=1e-5)

            if b.ndim >= 2:
                torch_expected = torch.linalg.solve_triangular(
                    torch.tensor(eff_a),
                    torch.tensor(b),
                    upper=eff_upper,
                    left=True,
                    unitriangular=unit_diagonal,
                ).numpy()
                np.testing.assert_allclose(got.numpy(), torch_expected, rtol=1e-5, atol=1e-5)

    def test_triangular_solve_edge_cases(self):
        a1 = np.array([[4.0]], dtype=np.float32)
        b1 = np.array([8.0], dtype=np.float32)
        np.testing.assert_allclose(Tensor(a1).triangular_solve(Tensor(b1)).numpy(), [2.0], rtol=1e-6)

        a_int = np.array([[2, 0], [4, 2]], dtype=np.int32)
        b_int = np.array([2, 8], dtype=np.int32)
        x_int = Tensor(a_int).triangular_solve(Tensor(b_int))
        assert x_int.dtype == 'float32'
        np.testing.assert_allclose(x_int.numpy(), np.linalg.solve(a_int.astype(np.float32), b_int), rtol=1e-6)

        a64 = np.array([[2.0, 0.0], [1.0, 4.0]], dtype=np.float64)
        b64 = np.array([[2.0], [9.0]], dtype=np.float64)
        x64 = Tensor(a64, dtype='float64').triangular_solve(Tensor(b64, dtype='float64'))
        assert x64.dtype == 'float64'
        np.testing.assert_allclose(x64.numpy(), np.linalg.solve(a64, b64), rtol=1e-12, atol=1e-12)

        with pytest.raises(ValueError, match='cannot triangular_solve'):
            Tensor(np.eye(2, dtype=np.float32)).triangular_solve(Tensor(np.ones((3,), dtype=np.float32)))

        singular = Tensor(np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32))
        out = singular.triangular_solve(Tensor(np.array([1.0, 2.0], dtype=np.float32))).numpy()
        assert np.isinf(out).any() or np.isnan(out).any()

    def test_cholesky_matches_numpy_torch_probe(self):
        torch = pytest.importorskip('torch')
        cases = [
            np.array([[4.0]], dtype=np.float32),
            np.array([[4.0, 2.0], [2.0, 5.0]], dtype=np.float32),
            np.array([[6.0, 2.0, 1.0], [2.0, 5.0, 2.0], [1.0, 2.0, 4.0]], dtype=np.float32),
            np.eye(4, dtype=np.float32) * 4.0,
            np.array([
                [[4.0, 2.0], [2.0, 5.0]],
                [[9.0, 3.0], [3.0, 2.0]],
            ], dtype=np.float32),
        ]
        for a in cases:
            l = Tensor(a).cholesky()
            expected = np.linalg.cholesky(a)
            torch_expected = torch.linalg.cholesky(torch.tensor(a)).numpy()
            np.testing.assert_allclose(l.numpy(), expected, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(l.numpy(), torch_expected, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(l.numpy() @ np.swapaxes(l.numpy(), -1, -2), a, rtol=1e-5, atol=1e-5)

        a = np.array([[4.0, 2.0], [2.0, 5.0]], dtype=np.float32)
        u = Tensor(a).cholesky(upper=True)
        np.testing.assert_allclose(u.numpy(), np.swapaxes(np.linalg.cholesky(a), -1, -2), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(np.swapaxes(u.numpy(), -1, -2) @ u.numpy(), a, rtol=1e-5, atol=1e-5)

    def test_cholesky_edge_cases(self):
        a_int = np.array([[4, 2], [2, 5]], dtype=np.int32)
        l_int = Tensor(a_int).cholesky()
        assert l_int.dtype == 'float32'
        np.testing.assert_allclose(l_int.numpy(), np.linalg.cholesky(a_int.astype(np.float32)), rtol=1e-5)

        a64 = np.array([[4.0, 2.0], [2.0, 5.0]], dtype=np.float64)
        l64 = Tensor(a64, dtype='float64').cholesky()
        assert l64.dtype == 'float64'
        np.testing.assert_allclose(l64.numpy(), np.linalg.cholesky(a64), rtol=1e-12, atol=1e-12)

        with pytest.raises(ValueError, match='cannot cholesky'):
            Tensor(np.ones((2, 3), dtype=np.float32)).cholesky()

        bad = Tensor(np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float32)).cholesky().numpy()
        assert np.isnan(bad).any()

    def test_cholesky_solve_matches_torch_probe(self):
        torch = pytest.importorskip('torch')
        a = np.array([[4.0, 2.0], [2.0, 5.0]], dtype=np.float32)
        b = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b_vec = np.array([1.0, 4.0], dtype=np.float32)
        for upper in [False, True]:
            factor = Tensor(a).cholesky(upper=upper)
            got = factor.cholesky_solve(Tensor(b), upper=upper)
            torch_expected = torch.cholesky_solve(
                torch.tensor(b),
                torch.linalg.cholesky(torch.tensor(a), upper=upper),
                upper=upper,
            ).numpy()
            np.testing.assert_allclose(got.numpy(), torch_expected, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(a @ got.numpy(), b, rtol=1e-5, atol=1e-5)

        ab = np.stack([a, np.array([[9.0, 3.0], [3.0, 2.0]], dtype=np.float32)], axis=0)
        bb_vec = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
        for upper in [False, True]:
            factor = Tensor(ab).cholesky(upper=upper)
            got = factor.cholesky_solve(Tensor(bb_vec), upper=upper)
            torch_expected = torch.cholesky_solve(
                torch.tensor(bb_vec).unsqueeze(-1),
                torch.linalg.cholesky(torch.tensor(ab), upper=upper),
                upper=upper,
            ).squeeze(-1).numpy()
            np.testing.assert_allclose(got.numpy(), torch_expected, rtol=1e-5, atol=1e-5)

            got_broadcast = factor.cholesky_solve(Tensor(b_vec), upper=upper)
            torch_broadcast = torch.cholesky_solve(
                torch.tensor(b_vec).reshape(1, 2, 1).expand(2, 2, 1),
                torch.linalg.cholesky(torch.tensor(ab), upper=upper),
                upper=upper,
            ).squeeze(-1).numpy()
            np.testing.assert_allclose(got_broadcast.numpy(), torch_broadcast, rtol=1e-5, atol=1e-5)

    def test_solve_matches_numpy_torch_probe(self):
        torch = pytest.importorskip('torch')
        a = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float32)
        b_vec = np.array([1.0, 4.0], dtype=np.float32)
        b_mat = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        got_vec = Tensor(a).solve(Tensor(b_vec))
        np.testing.assert_allclose(got_vec.numpy(), np.linalg.solve(a, b_vec), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            got_vec.numpy(),
            torch.linalg.solve(torch.tensor(a), torch.tensor(b_vec)).numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

        got_mat = Tensor(a).solve(Tensor(b_mat))
        np.testing.assert_allclose(got_mat.numpy(), np.linalg.solve(a, b_mat), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            got_mat.numpy(),
            torch.linalg.solve(torch.tensor(a), torch.tensor(b_mat)).numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

        pivot_a = np.array([[0.0, 2.0], [1.0, 3.0]], dtype=np.float32)
        pivot_b_vec = np.array([4.0, 5.0], dtype=np.float32)
        pivot_b_mat = np.array([[4.0, 1.0], [5.0, 2.0]], dtype=np.float32)
        got_pivot_vec = Tensor(pivot_a).solve(Tensor(pivot_b_vec))
        np.testing.assert_allclose(got_pivot_vec.numpy(), np.linalg.solve(pivot_a, pivot_b_vec), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            got_pivot_vec.numpy(),
            torch.linalg.solve(torch.tensor(pivot_a), torch.tensor(pivot_b_vec)).numpy(),
            rtol=1e-5,
            atol=1e-5,
        )
        got_pivot_mat = Tensor(pivot_a).solve(Tensor(pivot_b_mat))
        np.testing.assert_allclose(got_pivot_mat.numpy(), np.linalg.solve(pivot_a, pivot_b_mat), rtol=1e-5, atol=1e-5)

        pivot3 = np.array([[0.0, 2.0, 1.0], [1.0, 0.0, 3.0], [4.0, 1.0, 8.0]], dtype=np.float32)
        pivot3_b = np.array([3.0, 4.0, 13.0], dtype=np.float32)
        got_pivot3 = Tensor(pivot3).solve(Tensor(pivot3_b))
        np.testing.assert_allclose(got_pivot3.numpy(), np.linalg.solve(pivot3, pivot3_b), rtol=1e-5, atol=1e-5)

        ab = np.stack([a, a + np.eye(2, dtype=np.float32)], axis=0)
        bb = np.stack([b_mat, b_mat + 1.0], axis=0)
        got_batch = Tensor(ab).solve(Tensor(bb))
        expected_batch = np.stack([np.linalg.solve(ab[i], bb[i]) for i in range(2)])
        np.testing.assert_allclose(got_batch.numpy(), expected_batch, rtol=1e-5, atol=1e-5)

        bb_vec = np.stack([b_vec, np.array([2.0, 5.0], dtype=np.float32)], axis=0)
        got_batch_vec = Tensor(ab).solve(Tensor(bb_vec))
        expected_batch_vec = np.stack([np.linalg.solve(ab[i], bb_vec[i]) for i in range(2)])
        np.testing.assert_allclose(got_batch_vec.numpy(), expected_batch_vec, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            got_batch_vec.numpy(),
            torch.linalg.solve(torch.tensor(ab), torch.tensor(bb_vec)).numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

        pivot_ab = np.stack([pivot_a, np.array([[3.0, 1.0], [0.0, 2.0]], dtype=np.float32)], axis=0)
        pivot_bb_vec = np.stack([pivot_b_vec, np.array([7.0, 4.0], dtype=np.float32)], axis=0)
        got_pivot_batch = Tensor(pivot_ab).solve(Tensor(pivot_bb_vec))
        expected_pivot_batch = np.stack([np.linalg.solve(pivot_ab[i], pivot_bb_vec[i]) for i in range(2)])
        np.testing.assert_allclose(got_pivot_batch.numpy(), expected_pivot_batch, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            got_pivot_batch.numpy(),
            torch.linalg.solve(torch.tensor(pivot_ab), torch.tensor(pivot_bb_vec)).numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

        got_broadcast_vec = Tensor(ab).solve(Tensor(b_vec))
        expected_broadcast_vec = np.stack([np.linalg.solve(ab[i], b_vec) for i in range(2)])
        np.testing.assert_allclose(got_broadcast_vec.numpy(), expected_broadcast_vec, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            got_broadcast_vec.numpy(),
            torch.linalg.solve(torch.tensor(ab), torch.tensor(b_vec)).numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

        got_singleton_matrix = Tensor(ab).solve(Tensor(b_mat.reshape(1, 2, 2)))
        expected_singleton_matrix = np.stack([np.linalg.solve(ab[i], b_mat) for i in range(2)])
        np.testing.assert_allclose(got_singleton_matrix.numpy(), expected_singleton_matrix, rtol=1e-5, atol=1e-5)

        a64 = np.array([[3.0, 1.0], [1.0, 2.0]], dtype=np.float64)
        b64 = np.array([[4.0], [5.0]], dtype=np.float64)
        got64 = Tensor(a64).solve(Tensor(b64))
        assert got64.dtype == 'float64'
        np.testing.assert_allclose(got64.numpy(), np.linalg.solve(a64, b64), rtol=1e-10, atol=1e-10)

        with pytest.raises(ValueError, match='cannot solve'):
            Tensor(np.ones((2, 3), dtype=np.float32)).solve(Tensor(np.ones((2,), dtype=np.float32)))

    def test_lstsq_matches_numpy_torch_probe(self):
        torch = pytest.importorskip('torch')
        def torch_lstsq(a, b):
            # Use the SVD-based driver so rank-deficient cases check the same
            # minimum-norm policy as numpy.linalg.lstsq instead of a QR-driver
            # default that can vary across LAPACK builds.
            return torch.linalg.lstsq(torch.tensor(a), torch.tensor(b), driver='gelsd').solution.numpy()

        a = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]], dtype=np.float32)
        b_vec = np.array([1.0, 2.0, 2.5], dtype=np.float32)
        b_mat = np.array([[1.0, 0.5], [2.0, 1.0], [2.5, 1.5]], dtype=np.float32)

        got_vec = Tensor(a).lstsq(Tensor(b_vec))
        np.testing.assert_allclose(got_vec.numpy(), np.linalg.lstsq(a, b_vec, rcond=None)[0], rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(
            got_vec.numpy(),
            torch_lstsq(a, b_vec),
            rtol=2e-5,
            atol=2e-5,
        )

        got_mat = Tensor(a).lstsq(Tensor(b_mat))
        np.testing.assert_allclose(got_mat.numpy(), np.linalg.lstsq(a, b_mat, rcond=None)[0], rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(
            got_mat.numpy(),
            torch_lstsq(a, b_mat),
            rtol=2e-5,
            atol=2e-5,
        )

        ab = np.stack([a, np.array([[1.0, 0.0], [1.0, 1.5], [1.0, 3.0]], dtype=np.float32)], axis=0)
        bb = np.stack([b_mat, b_mat + 0.25], axis=0)
        got_batch = Tensor(ab).lstsq(Tensor(bb))
        expected_batch = np.stack([np.linalg.lstsq(ab[i], bb[i], rcond=None)[0] for i in range(2)])
        np.testing.assert_allclose(got_batch.numpy(), expected_batch, rtol=2e-5, atol=2e-5)

        bb_vec = np.stack([b_vec, np.array([1.25, 2.25, 2.75], dtype=np.float32)], axis=0)
        got_batch_vec = Tensor(ab).lstsq(Tensor(bb_vec))
        expected_batch_vec = np.stack([np.linalg.lstsq(ab[i], bb_vec[i], rcond=None)[0] for i in range(2)])
        np.testing.assert_allclose(got_batch_vec.numpy(), expected_batch_vec, rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(
            got_batch_vec.numpy(),
            torch_lstsq(ab, bb_vec),
            rtol=2e-5,
            atol=2e-5,
        )

        got_broadcast_vec = Tensor(ab).lstsq(Tensor(b_vec))
        expected_broadcast_vec = np.stack([np.linalg.lstsq(ab[i], b_vec, rcond=None)[0] for i in range(2)])
        np.testing.assert_allclose(got_broadcast_vec.numpy(), expected_broadcast_vec, rtol=2e-5, atol=2e-5)

        a64 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]], dtype=np.float64)
        b64 = np.array([1.0, 2.0, 4.0], dtype=np.float64)
        got64 = Tensor(a64).lstsq(Tensor(b64))
        assert got64.dtype == 'float64'
        np.testing.assert_allclose(got64.numpy(), np.linalg.lstsq(a64, b64, rcond=None)[0], rtol=1e-10, atol=1e-10)

        wide = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 1.0]], dtype=np.float32)
        wide_b_vec = np.array([1.0, 2.0], dtype=np.float32)
        wide_b_mat = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
        got_wide_vec = Tensor(wide).lstsq(Tensor(wide_b_vec))
        np.testing.assert_allclose(
            got_wide_vec.numpy(),
            np.linalg.lstsq(wide, wide_b_vec, rcond=None)[0],
            rtol=3e-5,
            atol=3e-5,
        )
        np.testing.assert_allclose(
            got_wide_vec.numpy(),
            torch_lstsq(wide, wide_b_vec),
            rtol=3e-5,
            atol=3e-5,
        )
        got_wide_mat = Tensor(wide).lstsq(Tensor(wide_b_mat))
        np.testing.assert_allclose(
            got_wide_mat.numpy(),
            np.linalg.lstsq(wide, wide_b_mat, rcond=None)[0],
            rtol=4e-5,
            atol=4e-5,
        )
        np.testing.assert_allclose(
            got_wide_mat.numpy(),
            torch_lstsq(wide, wide_b_mat),
            rtol=4e-5,
            atol=4e-5,
        )

        rankdef_cases = [
            (
                np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32),
                np.array([3.0, 6.0], dtype=np.float32),
            ),
            (
                np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32),
                np.array([1.0, 2.0, 3.0], dtype=np.float32),
            ),
            (
                np.array([[1.0, 1.0, 0.0], [2.0, 2.0, 0.0]], dtype=np.float32),
                np.array([3.0, 6.0], dtype=np.float32),
            ),
            (
                np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32),
                np.array([[3.0, 1.0], [6.0, 2.0]], dtype=np.float32),
            ),
        ]
        for rd_a, rd_b in rankdef_cases:
            got = Tensor(rd_a).lstsq(Tensor(rd_b)).numpy()
            np.testing.assert_allclose(
                got,
                np.linalg.lstsq(rd_a, rd_b, rcond=None)[0],
                rtol=2e-4,
                atol=2e-4,
            )
            np.testing.assert_allclose(
                got,
                torch_lstsq(rd_a, rd_b),
                rtol=2e-4,
                atol=2e-4,
            )

        with pytest.raises(ValueError, match='cannot lstsq'):
            Tensor(np.ones((2, 3), dtype=np.float32)).lstsq(Tensor(np.ones((3,), dtype=np.float32)))

    def test_cross_entropy_sparse_targets(self):
        logits = Tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        target = Tensor([0.0, 2.0])
        loss = logits.cross_entropy(target)
        assert loss.shape == ()
        np.testing.assert_allclose(loss.numpy(), np.log(3.0), rtol=1e-6)

    def test_cross_entropy_dense_targets(self):
        logits = Tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        target = Tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        loss = logits.cross_entropy(target)
        assert loss.shape == ()
        np.testing.assert_allclose(loss.numpy(), np.log(3.0), rtol=1e-6)

    def test_cross_entropy_realized_variable_bound_dense_targets(self):
        logits_all = Tensor((np.arange(80, dtype=np.float32).reshape(8, 10) / 10.0)).realize()
        labels_all = Tensor(np.eye(10, dtype=np.float32)[np.arange(8) % 10]).realize()
        i = Variable('i', 0, 4).bind(0)

        loss = -(labels_all[i:i + 4] * logits_all[i:i + 4].log_softmax(axis=1)).sum(axis=1).mean()

        assert loss.shape == ()
        np.testing.assert_allclose(loss.numpy(), np.array(2.6434937, dtype=np.float32), rtol=1e-6, atol=1e-6)

    def test_hlb_smoothed_bound_slice_multiplies_static_batch_shape(self):
        labels_all = Tensor(np.eye(10, dtype=np.float32)[np.arange(4) % 10]).realize()
        i = Variable('i', 0, 2).bind(0)

        labels = labels_all[i:i + 2]
        smoothed = labels * 0.9 + 0.01
        static = Tensor(np.ones((2, 10), dtype=np.float32))
        loss_rows = (smoothed * static).sum(axis=1).reshape((2,)).realize()

        assert labels.shape == (2, 10)
        assert smoothed.shape == (2, 10)
        assert loss_rows.shape == (2,)
        np.testing.assert_allclose(loss_rows.numpy(), np.ones((2,), dtype=np.float32), rtol=1e-6, atol=1e-6)

    def test_cross_entropy_sparse_targets_non_last_axis(self):
        logits = Tensor(np.zeros((2, 3, 2), dtype=np.float32))
        target = Tensor(np.array([[0.0, 2.0], [1.0, 0.0]], dtype=np.float32))
        loss = logits.cross_entropy(target, axis=-2)
        assert loss.shape == ()
        np.testing.assert_allclose(loss.numpy(), np.log(3.0), rtol=1e-6)

    def test_cross_entropy_default_matches_tinygrad_class_axis(self):
        logits = Tensor(np.zeros((2, 3, 2), dtype=np.float32))
        target = Tensor(np.array([[0.0, 2.0], [1.0, 0.0]], dtype=np.float32))
        loss = logits.cross_entropy(target)
        assert loss.shape == ()
        np.testing.assert_allclose(loss.numpy(), np.log(3.0), rtol=1e-6)

    def test_cross_entropy_dense_targets_non_last_axis(self):
        logits = Tensor(np.zeros((2, 3, 2), dtype=np.float32))
        target = Tensor(np.array([
            [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
        ], dtype=np.float32))
        loss = logits.cross_entropy(target, axis=1)
        assert loss.shape == ()
        np.testing.assert_allclose(loss.numpy(), np.log(3.0), rtol=1e-6)

    def test_cross_entropy_shape_mismatch_raises(self):
        logits = Tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        target = Tensor([[1.0, 0.0], [0.0, 1.0]])
        with pytest.raises(ValueError, match='shape mismatch'):
            logits.cross_entropy(target)


class TestAutograd:
    def test_grad_mul_sum(self):
        x = Tensor([1, 2, 3, 4], requires_grad=True)
        loss = (x * x).sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.uop_physical is not None
        assert x.grad.uop_physical.op_name == 'ADD'
        assert x.grad.uop == x.grad.uop_physical
        np.testing.assert_allclose(x.grad.numpy(), [2, 4, 6, 8])

    def test_grad_neg_sum(self):
        x = Tensor([1, 2, 3], requires_grad=True)
        loss = (-x).sum()
        loss.backward()
        np.testing.assert_allclose(x.grad.numpy(), [-1, -1, -1])

    def test_grad_max_reduce(self):
        x = Tensor([[1.0, 3.0, 2.0], [4.0, 5.0, 6.0]], requires_grad=True)
        loss = x.max(axis=1).sum()
        loss.backward()
        assert x.grad is not None
        np.testing.assert_allclose(x.grad.numpy(), [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


class TestDevice:
    def test_device_lookup(self):
        assert Device['cpu'] == 'CPU'
        assert Device['CUDA'] == 'CUDA'
        assert Device['CUDA:0'] == 'CUDA'
        assert Device['interp'] == 'INTERP'
        assert Device['cpu:x86'] == 'X86'
        assert Device['hip'] == 'HIP'
        assert Device['wasm'] == 'WASM'
        assert Device['webgpu'] == 'WEBGPU'
        with pytest.raises(ValueError, match='Unsupported device'):
            Device['host']
        with pytest.raises(ValueError, match='Unsupported device'):
            Device['not-a-device']

    def test_default_device_tracks_dev_context(self):
        original = Device.DEFAULT
        with Context(DEV='INTERP'):
            assert Device.DEFAULT == 'INTERP'
            assert Device.canonicalize(None) == 'INTERP'
            assert Tensor.empty(1).device == 'INTERP'
        assert Device.DEFAULT == original

    def test_requires_grad_inplace(self):
        a = Tensor([1.0])
        assert a.requires_grad_(True) is a
        assert a.requires_grad is True

    def test_to_device_roundtrip(self):
        a = Tensor([1.0, 2.0, 3.0], dtype='float64')
        b = (a + 1).to('cuda')
        assert b.device == 'CUDA'

        c = b.to('cpu')
        assert c.device == 'CPU'
        if Device.cuda_available():
            np.testing.assert_allclose(c.numpy(), [2.0, 3.0, 4.0])
        else:
            with pytest.raises(RuntimeError, match='poly_realize_tensors'):
                c.numpy()

    def test_to_cuda_runtime_behavior(self):
        a = Tensor([1.0, 2.0, 3.0])
        b = (a * 2).to('cuda')
        if Device.cuda_available():
            np.testing.assert_allclose(b.numpy(), [2.0, 4.0, 6.0])
        else:
            with pytest.raises(RuntimeError, match='CUDA'):
                b.numpy()

    def test_assign_rejects_device_mismatch_like_tinygrad(self):
        a = Tensor([1.0], device='cpu')
        v = Tensor([5.0], device='cpu').to('cuda')
        with pytest.raises(RuntimeError, match='assign device mismatch CPU != CUDA'):
            a.assign(v)

    def test_assign_rejects_dtype_mismatch_like_tinygrad(self):
        a = Tensor([1.0], dtype='float32')
        v = Tensor([5.0], dtype='float64')
        with pytest.raises(RuntimeError, match='assign dtype mismatch float32 != float64'):
            a.assign(v)

    def test_assign_to_same_device_place_keeps_place_target(self):
        a = Tensor([1.0], device='cpu').to('cuda')
        v = Tensor([5.0], device='cpu').to('cuda')
        assert a.assign(v) is a
        assert a.device == 'CUDA'
        assert not a.uop.has_buffer_identity()

    def test_shared_storage_to_copies_and_preserves_source_across_assign(self):
        source = Tensor([1.0, 2.0, 3.0], device='cpu').realize()
        source_buffer = source.uop.buffer
        target = source.to('interp').realize()
        target_buffer = target.uop.buffer

        assert target_buffer != source_buffer
        np.testing.assert_allclose(source.numpy(), [1.0, 2.0, 3.0])
        np.testing.assert_allclose(target.numpy(), [1.0, 2.0, 3.0])

        for values in ([9.0, 8.0, 7.0], [4.0, 5.0, 6.0]):
            target.assign(Tensor(values, device='interp')).realize()
            assert target.uop.buffer == target_buffer
            np.testing.assert_allclose(source.numpy(), [1.0, 2.0, 3.0])
            np.testing.assert_allclose(target.numpy(), values)


class TestRepr:
    def test_repr(self):
        t = Tensor([1, 2, 3])
        assert 'shape=(3,)' in repr(t)
        assert 'float32' in repr(t)

    def test_bool_raises_like_tinygrad(self):
        with pytest.raises(TypeError, match="__bool__ on Tensor is not defined"):
            bool(Tensor([1.0]))

    def test_repr_f64(self):
        t = Tensor([1, 2, 3], dtype='float64')
        assert 'float64' in repr(t)


class TestFloat64:
    """Tests for float64 dtype support."""

    def test_numpy_dtype_is_preserved_like_tinygrad(self):
        f64 = Tensor(np.array([1.0, 2.0], dtype=np.float64))
        assert f64.dtype == 'float64'
        assert f64.numpy().dtype == np.float64

        i64 = Tensor(np.array([1, 2], dtype=np.int64))
        assert i64.dtype == 'int64'
        assert i64.numpy().dtype == np.int64

        default_list = Tensor([1.0, 2.0])
        assert default_list.dtype == 'float32'

    def test_creation_from_list(self):
        t = Tensor([1.0, 2.0, 3.0], dtype='float64')
        assert t.dtype == 'float64'
        assert t.shape == (3,)
        assert t.numpy().dtype == np.float64
        np.testing.assert_allclose(t.numpy(), [1, 2, 3])

    def test_creation_2d(self):
        t = Tensor([[1, 2], [3, 4]], dtype='float64')
        assert t.dtype == 'float64'
        assert t.shape == (2, 2)
        assert t.numpy().dtype == np.float64
        np.testing.assert_allclose(t.numpy(), [[1, 2], [3, 4]])

    def test_zeros_f64(self):
        t = Tensor.zeros(4, dtype='float64')
        assert t.dtype == 'float64'
        assert t.numpy().dtype == np.float64
        np.testing.assert_allclose(t.numpy(), [0, 0, 0, 0])

    def test_ones_f64(self):
        t = Tensor.ones(3, dtype='float64')
        assert t.dtype == 'float64'
        np.testing.assert_allclose(t.numpy(), [1, 1, 1])

    def test_full_f64(self):
        t = Tensor.full((2, 3), 7.0, dtype='float64')
        assert t.dtype == 'float64'
        np.testing.assert_allclose(t.numpy(), np.full((2, 3), 7.0))

    def test_eye_f64(self):
        t = Tensor.eye(3, dtype='float64')
        assert t.dtype == 'float64'
        np.testing.assert_allclose(t.numpy(), np.eye(3))

    def test_arange_f64(self):
        t = Tensor.arange(5, dtype='float64')
        assert t.dtype == 'float64'
        np.testing.assert_allclose(t.numpy(), [0, 1, 2, 3, 4])

    def test_add_f64(self):
        a = Tensor([1.0, 2.0, 3.0], dtype='float64')
        b = Tensor([4.0, 5.0, 6.0], dtype='float64')
        c = (a + b).numpy()
        assert c.dtype == np.float64
        np.testing.assert_allclose(c, [5, 7, 9])

    def test_mul_f64(self):
        a = Tensor([1.0, 2.0, 3.0], dtype='float64')
        b = Tensor([4.0, 5.0, 6.0], dtype='float64')
        c = (a * b).numpy()
        assert c.dtype == np.float64
        np.testing.assert_allclose(c, [4, 10, 18])

    def test_neg_f64(self):
        a = Tensor([1.0, -2.0, 3.0], dtype='float64')
        c = (-a).numpy()
        assert c.dtype == np.float64
        np.testing.assert_allclose(c, [-1, 2, -3])

    def test_scalar_add_f64(self):
        a = Tensor([1.0, 2.0, 3.0], dtype='float64')
        c = (a + 10.0).numpy()
        assert c.dtype == np.float64
        np.testing.assert_allclose(c, [11, 12, 13])

    def test_sum_f64(self):
        a = Tensor([1.0, 2.0, 3.0, 4.0], dtype='float64')
        s = a.sum().item()
        np.testing.assert_allclose(s, 10.0)

    def test_exp_f64(self):
        a = Tensor([0.0, 1.0, 2.0], dtype='float64')
        c = a.exp().numpy()
        assert c.dtype == np.float64
        np.testing.assert_allclose(c, np.exp([0, 1, 2]), rtol=1e-10)

    def test_sqrt_f64(self):
        a = Tensor([1.0, 4.0, 9.0], dtype='float64')
        c = a.sqrt().numpy()
        assert c.dtype == np.float64
        np.testing.assert_allclose(c, [1, 2, 3], rtol=1e-14)

    def test_chain_f64(self):
        a = Tensor([1.0, 2.0, 3.0], dtype='float64')
        b = Tensor([4.0, 5.0, 6.0], dtype='float64')
        c = ((a + b) * a - b).numpy()
        assert c.dtype == np.float64
        np.testing.assert_allclose(c, [(1+4)*1-4, (2+5)*2-5, (3+6)*3-6])

    def test_reshape_f64(self):
        a = Tensor([1, 2, 3, 4, 5, 6], dtype='float64')
        b = a.reshape(2, 3).numpy()
        assert b.dtype == np.float64
        np.testing.assert_allclose(b, [[1, 2, 3], [4, 5, 6]])

    def test_backward_f64(self):
        a = Tensor([1.0, 2.0, 3.0], dtype='float64', requires_grad=True)
        b = Tensor([4.0, 5.0, 6.0], dtype='float64')
        loss = (a * b).sum()
        loss.backward()
        assert a.grad is not None
        np.testing.assert_allclose(a.grad.numpy(), [4, 5, 6], rtol=1e-14)

    def test_dtype_propagation(self):
        """Ensure dtype propagates through ops."""
        a = Tensor([1.0, 2.0], dtype='float64')
        b = a + 1.0
        assert b.dtype == 'float64'
        c = b * 2.0
        assert c.dtype == 'float64'
        d = c.exp()
        assert d.dtype == 'float64'

    def test_default_is_f32(self):
        """Ensure default dtype is still float32."""
        a = Tensor([1.0, 2.0])
        assert a.dtype == 'float32'
        assert a.numpy().dtype == np.float32


class TestMaterializationParity:
    def test_realize_retargets_live_tensors_sharing_lazy_uop(self):
        a = Tensor([1.0]).realize()
        x1 = a + 1
        x2 = a + 1
        assert x1.uop == x2.uop

        x1.realize()

        assert x1.uop == x2.uop
        assert x1.uop.buffer == x2.uop.buffer
        np.testing.assert_allclose(x2.numpy(), [2.0])

    def test_realize_rewrites_downstream_live_graph_to_snapshot(self):
        a = Tensor([1.0]).realize()
        x = a + 1
        y = x + 2

        x.realize()
        a.assign(Tensor([10.0])).realize()

        np.testing.assert_allclose(x.numpy(), [2.0])
        np.testing.assert_allclose(y.numpy(), [4.0])

    def test_separate_realizes_do_not_alias_assign(self):
        a = Tensor([1.0]).realize()
        y1 = (a + 1).realize()
        y2 = (a + 1).realize()

        assert y1.uop.buffer != y2.uop.buffer

        y2.assign(Tensor([5.0])).realize()
        np.testing.assert_allclose(y1.numpy(), [2.0])
        np.testing.assert_allclose(y2.numpy(), [5.0])

    def test_assign_realized_targets_reuse_current_buffer(self):
        a = Tensor([1.0]).realize()
        a_buf = a.uop.buffer

        # assign() follows tinygrad by creating an effect graph first; realizing
        # that effect writes the existing target storage and returns to the same
        # current buffer root.
        a.assign(Tensor([5.0]))
        assert not a.uop.has_buffer_identity()
        a.realize()
        assert a.uop.buffer == a_buf
        np.testing.assert_allclose(a.numpy(), [5.0])

        x = (Tensor([1.0]) + 1).realize()
        x_buf = x.uop.buffer
        x.assign(Tensor([9.0])).realize()
        assert x.uop.buffer == x_buf
        np.testing.assert_allclose(x.numpy(), [9.0])

    def test_to_keeps_separate_realized_same_logical_occurrences(self):
        a = Tensor([1.0]).realize()
        x1 = (a + 1).realize()
        x2 = (a + 1).realize()
        assert x1.uop.buffer != x2.uop.buffer

        # x1/x2 preserve the same exportable logical expression, but .to()
        # must carry the occurrence-specific realized source into placement.
        x1_cuda = x1.to('cuda')
        x2_cuda = x2.to('cuda')
        assert x1_cuda.uop.op_name == 'COPY'
        assert x2_cuda.uop.op_name == 'COPY'
        assert x1_cuda.uop.buffer is None
        assert x2_cuda.uop.buffer is None
        assert x1_cuda.uop.src[0].buffer == x1.uop.buffer
        assert x2_cuda.uop.src[0].buffer == x2.uop.buffer
        assert x1_cuda.uop.src[0].buffer != x2_cuda.uop.src[0].buffer

        y1 = x1_cuda + 1
        y2 = x2_cuda + 1
        assert y1.uop != y2.uop

    def test_nested_to_keeps_realized_current_and_export_logical_separate(self):
        x = (Tensor([1.0]) + 1).realize()
        x_cuda = x.to('cuda')
        x_cpu = x_cuda.to('cpu')

        # Pinned Tensor.to (tensor.py:327-335) keeps both device moves as exact
        # current COPY occurrences. Polygrad additionally preserves the
        # approved portable logical twin.
        assert x_cuda.uop.op_name == 'COPY'
        assert x_cuda.uop.src[0].raw == x.uop.raw
        assert x_cpu.uop.op_name == 'COPY'
        assert x_cpu.uop.src[0].raw == x_cuda.uop.raw
        assert x_cpu.uop != x.uop
        assert x_cpu.uop_physical == x_cpu.uop
        assert x_cpu.uop_logical == x.uop_logical

        y = x_cpu + 1
        assert y.device == 'CPU'
        assert y.uop != x_cpu.uop

        # Pinned Tensor.alu consumes ordered current roots (tensor.py:128-140).
        # Both operands share the same portable logical X, but the second
        # physical occurrence must remain the exact CUDA->CPU COPY chain.
        mixed = x + x_cpu
        assert mixed.uop.op_name == 'ADD'
        assert mixed.uop.src[0].raw == x.uop.raw
        assert mixed.uop.src[1].raw == x_cpu.uop.raw
        assert mixed.uop.src[1].src[0].raw == x_cuda.uop.raw
        assert mixed.uop.src[1].src[0].src[0].raw == x.uop.raw
        assert mixed.uop_logical.op_name == 'ADD'
        assert mixed.uop_logical.src[0].raw == x.uop_logical.raw
        assert mixed.uop_logical.src[1].raw == x.uop_logical.raw

    def test_minimum_keeps_ordered_roundtrip_occurrence(self):
        x = Tensor([1.0], device='cpu').realize()
        x_cuda = x.to('cuda')
        x_cpu = x_cuda.to('cpu')
        out = x.minimum(x_cpu)

        # Pinned minimum consumes ordered current Tensor.uop operands through
        # inverse -> maximum -> inverse (mixin/elementwise.py:366-393).
        assert out.uop.op_name == 'MUL'
        maximum = out.uop.src[0]
        assert maximum.op_name == 'MAX'
        assert maximum.src[0].op_name == 'MUL'
        assert maximum.src[1].op_name == 'MUL'
        assert maximum.src[0].src[0].raw == x.uop.raw
        assert maximum.src[1].src[0].raw == x_cpu.uop.raw
        assert maximum.src[1].src[0].op_name == 'COPY'
        assert maximum.src[1].src[0].src[0].raw == x_cuda.uop.raw

    def test_clamp_matches_pinned_optional_bounds_and_occurrence(self):
        values = Tensor([-float('inf'), -2.0, 2.0, float('inf')], device='cpu')
        np.testing.assert_equal(
            values.clamp(min_=-1.0).numpy(),
            np.array([-1.0, -1.0, 2.0, np.inf], dtype=np.float32),
        )
        np.testing.assert_equal(
            values.clamp(max_=1.0).numpy(),
            np.array([-np.inf, -2.0, 1.0, 1.0], dtype=np.float32),
        )

        x = Tensor([1.0], device='cpu').realize()
        x_cuda = x.to('cuda')
        x_cpu = x_cuda.to('cpu')
        out = x_cpu.clamp(-1.0, 1.0)

        def count_op(root, name):
            seen, stack, count = set(), [root], 0
            while stack:
                node = stack.pop()
                if node.raw in seen:
                    continue
                seen.add(node.raw)
                count += node.op_name == name
                stack.extend(node.src)
            return count

        # Pinned clamp is the conditional comparison/WHERE program over the
        # exact current Tensor.uop (mixin/elementwise.py:569-580).
        assert out.uop.op_name == 'WHERE'
        assert count_op(out.uop, 'WHERE') == 2
        assert count_op(out.uop, 'COPY') == 2

    def test_trig_matches_pinned_promotion_and_occurrence(self):
        integer = Tensor([0, 1, 2], dtype='int32', device='cpu')
        np.testing.assert_allclose(
            integer.sin().numpy(),
            np.sin(np.array([0, 1, 2], dtype=np.float32)),
            rtol=1e-6,
            atol=1e-6,
        )
        assert integer.sin().dtype == 'float32'
        np.testing.assert_allclose(
            Tensor([False, True], dtype='bool').sin().numpy(),
            np.sin(np.array([0, 1], dtype=np.float32)),
            rtol=1e-6,
            atol=1e-6,
        )

        half = Tensor([0.0, 1.0], dtype='float16')
        assert half.cos().dtype == 'float16'
        np.testing.assert_allclose(
            half.cos().float().numpy(),
            np.cos(np.array([0.0, 1.0], dtype=np.float32)),
            rtol=2e-3,
            atol=2e-3,
        )

        angles32 = Tensor([0.0, 0.25, 0.5], dtype='float32')
        angles64 = Tensor([0.0, 0.25, 0.5], dtype='float64')
        for angles, expected_dtype, numpy_dtype, tolerance in (
            (angles32, 'float32', np.float32, 1e-6),
            (angles64, 'float64', np.float64, 1e-12),
        ):
            expected = np.array([0.0, 0.25, 0.5], dtype=numpy_dtype)
            assert angles.cos().dtype == expected_dtype
            assert angles.tan().dtype == expected_dtype
            np.testing.assert_allclose(
                angles.cos().numpy(), np.cos(expected),
                rtol=tolerance, atol=tolerance,
            )
            np.testing.assert_allclose(
                angles.tan().numpy(), np.tan(expected),
                rtol=tolerance, atol=tolerance,
            )

        x = Tensor([0.25], device='cpu').realize()
        moved = x.to('cuda').to('cpu')

        def count_op(root, name):
            seen, stack, count = set(), [root], 0
            while stack:
                node = stack.pop()
                if node.raw in seen:
                    continue
                seen.add(node.raw)
                count += node.op_name == name
                stack.extend(node.src)
            return count

        assert count_op(moved.cos().uop, 'COPY') == 2
        assert count_op(moved.tan().uop, 'COPY') == 2

    def test_literal_elementwise_composites_match_pinned(self):
        values = np.array([-2.5, -1.0, 0.0, 0.5, 2.5], dtype=np.float32)
        x = Tensor(values)
        sigmoid = 1 / (1 + np.exp(-values))
        gelu = 0.5 * values * (
            1
            + np.tanh(
                np.sqrt(2 / np.pi) * (values + 0.044715 * values**3)
            )
        )
        expected = {
            'square': values * values,
            'ceil': np.ceil(values),
            'floor': np.floor(values),
            'sigmoid': sigmoid,
            'tanh': np.tanh(values),
            'gelu': gelu,
            'quick_gelu': values / (1 + np.exp(-(1.702 * values))),
            'relu6': np.minimum(np.maximum(values, 0), 6),
            'leaky_relu': np.where(values < 0, 0.01 * values, values),
            'hardswish': values * np.minimum(np.maximum(values + 3, 0), 6) / 6,
            'hardsigmoid': np.minimum(np.maximum(values / 6 + 0.5, 0), 1),
            'hardtanh': np.minimum(np.maximum(values, -1), 1),
            'silu': values * sigmoid,
            'elu': np.where(values > 0, values, np.exp(values) - 1),
            'sign': np.sign(values),
            'abs': np.abs(values),
            'isnan': np.isnan(values),
        }
        for method, wanted in expected.items():
            got = getattr(x, method)().numpy()
            np.testing.assert_allclose(got, wanted, rtol=2e-6, atol=2e-6)

        custom = x.hardsigmoid(alpha=0.2, beta=0.3)
        np.testing.assert_allclose(
            custom.numpy(),
            np.minimum(np.maximum(0.2 * values + 0.3, 0), 1),
            rtol=2e-6,
            atol=2e-6,
        )

        half = Tensor(values, dtype='float16')
        assert half.sigmoid().dtype == 'float16'
        assert half.tanh().dtype == 'float16'
        assert half.gelu().dtype == 'float16'
        assert half.quick_gelu().dtype == 'float16'
        np.testing.assert_allclose(
            half.tanh().float().numpy(), np.tanh(values), rtol=2e-3, atol=2e-3
        )
        np.testing.assert_allclose(
            half.gelu().float().numpy(), gelu, rtol=2e-3, atol=2e-3
        )
        np.testing.assert_allclose(
            half.quick_gelu().float().numpy(),
            expected['quick_gelu'],
            rtol=2e-3,
            atol=2e-3,
        )

        ints = Tensor([-2, 0, 3], dtype='int32')
        np.testing.assert_array_equal(ints.sign().numpy(), [-1, 0, 1])
        np.testing.assert_array_equal(ints.abs().numpy(), [2, 0, 3])
        assert ints.gelu().dtype == 'float32'
        assert ints.quick_gelu().dtype == 'float32'
        int_values = np.array([-2, 0, 3], dtype=np.float32)
        np.testing.assert_allclose(
            ints.quick_gelu().numpy(),
            int_values / (1 + np.exp(-(1.702 * int_values))),
            rtol=2e-6,
            atol=2e-6,
        )
        boolean = Tensor([False, True], dtype='bool')
        np.testing.assert_array_equal(boolean.sign().numpy(), [False, True])
        np.testing.assert_array_equal(boolean.abs().numpy(), [False, True])
        assert boolean.gelu().dtype == 'float32'
        assert boolean.quick_gelu().dtype == 'float32'
        bool_values = np.array([0, 1], dtype=np.float32)
        np.testing.assert_allclose(
            boolean.quick_gelu().numpy(),
            bool_values / (1 + np.exp(-(1.702 * bool_values))),
            rtol=2e-6,
            atol=2e-6,
        )

        const = x.const_like(1)
        assert const.dtype == x.dtype
        assert const.shape == x.shape
        assert const.uop.op_name == 'EXPAND'
        np.testing.assert_array_equal(const.numpy(), np.ones_like(values))

        for method in ('ceil', 'floor'):
            grad_input = Tensor(values, requires_grad=True)
            getattr(grad_input, method)().sum().backward()
            np.testing.assert_array_equal(grad_input.grad.numpy(), np.zeros_like(values))

    def test_saturated_float16_gelu_family_backward_matches_pinned_cpu(self):
        # Pinned mixin/elementwise.py:751-759 constructs these composites;
        # symbolic.py:478-480 stabilizes the reciprocal products, and
        # cstyle.py:40-43,62-63,194,232-237 defines the exact half rendering.
        inputs = [-10.0, -8.0, -7.0, -6.0, -5.0]
        expected_bits = {
            'gelu': np.array([0x0000, 0x0000, 0x0000, 0x0000, 0x0000], dtype=np.uint16),
            'quick_gelu': np.array(
                [0x0000, 0x0000, 0x0000, 0x8D8A, 0x9636], dtype=np.uint16
            ),
        }
        expected_lists = {
            'gelu': [-0.0, -0.0, -0.0, -0.0, -0.0],
            'quick_gelu': [
                -0.0,
                -0.0,
                -0.0,
                -0.00022019156313035637,
                -0.0010094891767948866,
            ],
        }
        for method, wanted in expected_bits.items():
            x = Tensor(inputs, dtype='float16', requires_grad=True)
            activated = getattr(x, method)()
            activated.sum().backward()
            gradient = np.asarray(x.grad.numpy(), dtype=np.float16)
            assert np.isfinite(gradient).all()
            np.testing.assert_array_equal(gradient.view(np.uint16), wanted)
            assert activated.tolist() == expected_lists[method]

    def test_round_and_isinf_match_pinned_compositions(self):
        # Pinned mixin/elementwise.py:872-880 is round-half-to-even.
        values = np.array(
            [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], dtype=np.float32
        )
        expected_round = np.array([-2, -2, 0, 0, 2, 2], dtype=np.float32)
        for dtype, expected_dtype in (
            ('float16', 'float16'),
            ('float32', 'float32'),
            ('float64', 'float64'),
        ):
            rounded = Tensor(values, dtype=dtype).round()
            assert rounded.dtype == expected_dtype
            np.testing.assert_array_equal(rounded.numpy(), expected_round)

        rounded_int = Tensor([-2, -1, 0, 1, 2], dtype='int32').round()
        assert rounded_int.dtype == 'float32'
        np.testing.assert_array_equal(rounded_int.numpy(), [-2, -1, 0, 1, 2])

        rounded_bool = Tensor([False, True], dtype='bool').round()
        assert rounded_bool.dtype == 'float32'
        np.testing.assert_array_equal(rounded_bool.numpy(), [0, 1])

        # Pinned mixin/elementwise.py:596-604 independently gates each sign.
        infinity_values = Tensor(
            [-float('inf'), -1.0, -0.0, 0.0, 1.0, float('inf'), float('nan')]
        )
        for detect_positive, detect_negative, expected in (
            (False, False, [False, False, False, False, False, False, False]),
            (False, True, [True, False, False, False, False, False, False]),
            (True, False, [False, False, False, False, False, True, False]),
            (True, True, [True, False, False, False, False, True, False]),
        ):
            result = infinity_values.isinf(
                detect_positive=detect_positive,
                detect_negative=detect_negative,
            )
            assert result.dtype == 'bool'
            np.testing.assert_array_equal(result.numpy(), expected)

        for dtype in ('float16', 'float32', 'float64', 'int32', 'bool'):
            result = Tensor([0, 1], dtype=dtype).isinf()
            assert result.dtype == 'bool'
            np.testing.assert_array_equal(result.numpy(), [False, False])

        def count_op(root, name):
            seen, stack, count = set(), [root], 0
            while stack:
                node = stack.pop()
                if node.raw in seen:
                    continue
                seen.add(node.raw)
                count += node.op_name == name
                stack.extend(node.src)
            return count

        moved = Tensor.empty(2, device='cpu').realize().to('cuda').to('cpu')
        assert count_op(moved.round().uop, 'COPY') == 2
        for detect_positive, detect_negative in (
            (False, False), (False, True), (True, False), (True, True)
        ):
            result = moved.isinf(
                detect_positive=detect_positive,
                detect_negative=detect_negative,
            )
            assert count_op(result.uop, 'COPY') == 2
