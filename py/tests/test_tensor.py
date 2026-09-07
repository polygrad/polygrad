"""Tests for the polygrad Python Tensor class."""

import ctypes
import gc
import math
import numpy as np
import os
import pytest
import subprocess
import sys
import weakref

from polygrad import Device, Jit, JitError, Runtime, Tensor, Variable, _ffi, can_run, compile as pg_compile, jit, stats as pg_stats
from polygrad.dtype import dtypes
from polygrad.helpers import Context
from polygrad.uop.ops import AxisType, KernelInfo, UOp, _dispose_uops_for_ctx


class TestCreation:
    def test_invalid_logical_environment_fails_import(self):
        env = os.environ.copy()
        env['POLY_LOGICAL'] = 'sometimes'
        proc = subprocess.run(
            [sys.executable, '-c', 'import polygrad'],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        assert proc.returncode != 0
        assert 'invalid POLY_LOGICAL' in proc.stderr

    def test_logical_policy_scopes_and_tensor_override(self):
        baseline = Tensor([1.0])
        baseline_policy = baseline.logical_policy
        with Context(LOGICAL='never'):
            physical_only = Tensor([2.0])
            assert physical_only.logical_policy == 'never'
            assert physical_only.logical_state == 'never_constructed'
            assert physical_only.uop_logical is None
        assert Tensor([3.0]).logical_policy == baseline_policy

        with Runtime(device='interp', logical='until_realize') as runtime:
            current = runtime.Tensor([1.0, 2.0]) + 1
            assert current.logical_policy == 'until_realize'
            assert current.logical_state == 'available'
            current.realize()
            assert current.logical_state == 'retired'

            with runtime.logical('always'):
                retained = runtime.Tensor([4.0]) + 1
            retained.realize()
            assert retained.logical_policy == 'always'
            assert retained.logical_state == 'available'

            dropped = runtime.Tensor([5.0], logical=False)
            assert dropped.logical_policy == 'never'
            assert dropped.uop_logical is None
            assert dropped.set_logical_policy('always') is False
            descendant = dropped + 1
            assert descendant.logical_policy == 'never'
            assert descendant.logical_state == 'never_constructed'
            assert descendant.tolist() == [6.0]
            cloned = dropped.clone()
            assert cloned.logical_policy == 'never'
            assert cloned.logical_state == 'never_constructed'
            assert cloned.uop_logical is None
            assert cloned.tolist() == [5.0]
            grad_source = runtime.Tensor([2.0], logical=False)
            (grad_source * grad_source).sum().backward()
            assert grad_source.grad.logical_policy == 'never'
            assert grad_source.grad.logical_state == 'never_constructed'
            assert grad_source.grad.uop_logical is None
            assert grad_source.grad.tolist() == [4.0]

            marked = runtime.Tensor([6.0]) + 1
            assert marked.preserve_logical() is marked
            marked.realize()
            assert marked.logical_state == 'available'

    def test_runtime_dispose_invalidates_borrowed_instance_and_bound_wrappers(self):
        # Polygrad's C arena is an approved ownership divergence from Tinygrad's
        # live Python UOps; disposal must invalidate every borrowed wrapper.
        code = r'''
from polygrad import Runtime

rt = Runtime(device='cpu')
Tensor = rt.Tensor
Variable = rt.Variable
x = Tensor([1.0, 2.0]).realize()
uop = x.uop
inst = rt.Model.from_tensors(inputs={'x': x}, outputs={'output': x + 1})
rt.dispose()
assert inst._ptr is None
inst.free()
for call in (lambda: Tensor.empty(2), lambda: Variable('n', 1, 2), lambda: x.shape,
             lambda: uop.op):
    try:
        call()
    except RuntimeError as exc:
        assert 'disposed' in str(exc)
    else:
        raise AssertionError('stale runtime wrapper remained usable')
print('teardown_ok')
'''
        proc = subprocess.run(
            [sys.executable, '-c', code], capture_output=True, text=True, check=False
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert proc.stdout.strip() == 'teardown_ok'
        assert proc.stderr == ''

    def test_default_context_exit_frees_live_borrowed_instance_first(self):
        code = r'''
from polygrad import Model, Tensor

x = Tensor([1.0, 2.0]).realize()
inst = Model.from_tensors(inputs={'x': x}, outputs={'output': x + 1})
assert inst._ptr
print('leaving_live_instance')
'''
        proc = subprocess.run(
            [sys.executable, '-c', code], capture_output=True, text=True, check=False
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert proc.stdout.strip() == 'leaving_live_instance'

    def test_runtime_dispose_retires_every_wrapper_for_the_same_uop(self):
        runtime = Runtime(device='cpu')
        tensor = runtime.Tensor.empty(4)
        first, second = tensor.uop, tensor.uop
        assert first is not second and first.raw == second.raw
        try:
            _dispose_uops_for_ctx(runtime._ctx)
            assert first.raw is None
            assert second.raw is None
        finally:
            # Keep the failing pre-fix regression from releasing through a
            # context that Runtime.dispose() is about to destroy.
            first._dispose()
            second._dispose()
            runtime.dispose()

    def test_runtime_tensor_gc_retires_core_handle_and_residency(self):
        runtime = Runtime(device='cpu')
        try:
            tensor = runtime.Tensor.empty(1024)
            tensor.copy_from(np.zeros(1024, dtype=np.float32))
            assert runtime.stats()['tensor_records'] == 1
            assert runtime.stats()['mem_used'] == 4096
            del tensor
            gc.collect()
            runtime.collect()
            assert runtime.stats()['tensor_records'] == 0
            assert runtime.stats()['mem_used'] == 0
        finally:
            runtime.dispose()

    def test_custom_kernel_gradient_metadata_does_not_own_residency(self):
        runtime = Runtime(device='cpu')
        Tensor = runtime.Tensor

        def copy_kernel(out, src):
            out, src = out.flatten(), src.flatten()
            i = UOp.range(out.ctx, out.numel(), 0)
            return out[i].store(src[i]).end(i).sink(
                arg=KernelInfo(name='gc_custom_copy')
            )

        try:
            before = runtime.stats()['mem_used']
            out = Tensor.empty(1 << 20)
            src = Tensor(np.zeros(1 << 20, dtype=np.float32))
            result = out.custom_kernel(src, fxn=copy_kernel, grad_fxn=lambda grad, call: (None, grad))[0]
            result.realize()
            assert runtime.stats()['mem_used'] >= before + (2 << 22)
            del result, out, src
            gc.collect()
            runtime.collect()
            assert runtime.stats()['mem_used'] == before
        finally:
            runtime.dispose()

    def test_from_list(self):
        t = Tensor([1.0, 2.0, 3.0])
        assert t.shape == (3,)
        np.testing.assert_allclose(t.numpy(), [1, 2, 3])

    def test_list_dtype_inference_matches_tinygrad(self):
        assert Tensor([1, 2, 3]).dtype is dtypes.int32
        assert Tensor([[1, 2], [3, 4]]).dtype is dtypes.int32
        assert Tensor([True, False]).dtype is dtypes.bool
        assert Tensor([1, 2.5]).dtype is dtypes.float32
        assert Tensor([]).dtype is dtypes.float32

    @pytest.mark.parametrize('payload,dtype,raw_dtype,expected', [
        (b'\x00\x7f\x80\xff', None, 'uint8', [0, 127, 128, 255]),
        (b'', None, 'uint8', []),
        (np.array([1, 65000], dtype=np.uint16).tobytes(), 'uint16', 'uint16', [1, 65000]),
        (np.array([1.25, -2.5], dtype=np.float32).tobytes(), 'float32', 'uint32', [0x3fa00000, 0xc0200000]),
        (np.array([0x3f80, 0xc020], dtype=np.uint16).tobytes(), 'bfloat16', 'uint16', [0x3f80, 0xc020]),
        (b'\x38\xc0', 'fp8e4m3', 'uint8', [0x38, 0xc0]),
    ])
    def test_bytes_are_raw_typed_storage_with_exact_upload_graph(self, payload, dtype, raw_dtype, expected):
        # Pinned Tensor.__init__ -> UOp._frompy(bytes): allocate writable raw
        # storage, then COPY. BF16/FP8 bytes must not stage through float32 CAST.
        tensor = Tensor(payload, dtype=dtype)
        root = tensor.uop_physical
        assert root.op_name == 'COPY' and len(root.src) == 1
        source = root.src[0]
        assert source.op_name == 'BUFFER' and len(source.src) == 1
        assert source.src[0].op_name == 'CONST' and len(source.src[0].src) == 0
        assert source.dtype is root.dtype is getattr(dtypes, dtype or 'uint8')
        assert tensor.shape == (len(expected),)
        readout = tensor if (dtype or 'uint8') == raw_dtype else tensor.bitcast(raw_dtype)
        np.testing.assert_array_equal(readout.numpy(), expected)

    def test_bytes_reject_partial_elements_and_weak_storage(self):
        with pytest.raises(ValueError):
            Tensor(b'\x01\x00\xff', dtype=dtypes.uint16)
        for dtype in dtypes.weaks:
            with pytest.raises(RuntimeError, match='cannot create storage for weak dtype'):
                Tensor(b'\x01', dtype=dtype)

    def test_bytes_storage_is_owned_and_can_be_updated(self):
        payload = b'\x01\x02\x03'
        tensor = Tensor(payload)
        tensor.assign(Tensor([4, 5, 6], dtype='uint8'))
        assert payload == b'\x01\x02\x03'
        del payload
        gc.collect()
        np.testing.assert_array_equal(tensor.numpy(), [4, 5, 6])

    @pytest.mark.parametrize('dtype', [None, 'float64', 'int32', 'uint8', 'bool'])
    def test_none_constructs_scalar_zero_with_exact_const_graph(self, dtype):
        # Pinned Tensor(None) is UOp.const(0.0), not empty storage. Comparing
        # scalar values with an empty NumPy array can pass without checking it.
        tensor = Tensor(None, dtype=dtype)
        expected = Tensor(0.0, dtype=dtype)
        assert tensor.shape == expected.shape == ()
        assert tensor.uop_physical == expected.uop_physical
        assert tensor.uop_physical.op_name == 'CONST'
        assert tensor.item() == 0

    @pytest.mark.parametrize('dtype', dtypes.ints)
    def test_list_storage_truncates_python_integers_before_numpy_admission(self, dtype):
        bits = dtype.bitsize
        values = [[-(1 << 100) - 3, -3.5, -1, 0],
                  [1 << (bits - 1), (1 << bits) - 1, 1 << bits, (1 << 100) + 5]]
        modulus = 1 << bits
        def truncate(value):
            value = int(value) % modulus
            return value - modulus if not dtypes.is_unsigned(dtype) and value >= modulus // 2 else value
        expected = [[truncate(value) for value in row] for row in values]
        tensor = Tensor(values, dtype=dtype)
        source = tensor.uop_physical
        assert source.op_name == 'COPY' and len(source.src) == 1
        assert source.dtype is dtype
        assert source.src[0].op_name == 'RESHAPE'
        assert source.src[0].src[0].op_name == 'BUFFER'
        assert source.src[0].src[0].dtype is dtype
        assert tensor.shape == (2, 4)
        np.testing.assert_array_equal(tensor.numpy(), np.asarray(expected, dtype=dtype.fmt))

    @pytest.mark.parametrize('dtype', dtypes.weaks)
    @pytest.mark.parametrize('data', [[1], (1,), np.array([1])])
    def test_non_scalar_storage_rejects_weak_dtype(self, data, dtype):
        with pytest.raises(RuntimeError, match='cannot create storage for weak dtype'):
            Tensor(data, dtype=dtype)

    @pytest.mark.parametrize('data', [[[1], []], [[], [[]]], [[[1, 1, 1], [1, 1]]]])
    def test_integer_list_storage_preserves_ragged_rejection(self, data):
        with pytest.raises(ValueError):
            Tensor(data, dtype='int32')

    def test_from_scalar(self):
        cases = [
            (Tensor(True), (), "bool", True),
            (Tensor(42), (), "weakint", 42),
            (Tensor(42.0), (), "weakfloat", 42.0),
            (Tensor(7, device="CUDA"), (), "weakint", 7),
            (Tensor(1.5, device="CUDA"), (), "weakfloat", 1.5),
        ]
        for tensor, shape, dtype, value in cases:
            assert tensor.shape == shape
            assert tensor.dtype is getattr(dtypes, dtype)
            assert tensor.uop.op_name == "CONST"
            assert tensor.uop_logical.raw == tensor.uop.raw
            if tensor.device == "CPU":
                np.testing.assert_allclose(tensor.numpy(), value)

    def test_internal_scalar_stores_typed_current_root(self):
        cases = [
            (Tensor.empty(2, dtype="bool"), True, "bool"),
            (Tensor.empty(2, dtype="int32"), 7, "weakint"),
            (Tensor.empty(2, dtype="float32"), 1, "weakint"),
        ]
        for source, value, dtype in cases:
            scalar = source._ensure_tensor(value)
            assert scalar.dtype is getattr(dtypes, dtype)
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
        assert t.uop_logical.buffer.src[0].op_name == 'UNIQUE'
        assert t.uop_logical.buffer.src[1:] == ()
        assert t.uop_physical.buffer.src[0].op_name == 'CONST'

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

    def test_flatten_resolves_negative_dimensions_like_tinygrad(self):
        flattened = Tensor.arange(32).reshape(1, 2, 16).flatten(-2)
        assert flattened.shape == (1, 32)
        assert flattened.uop.op_name == "RESHAPE"
        np.testing.assert_array_equal(flattened.numpy(), np.arange(32).reshape(1, 32))
        with pytest.raises(IndexError, match=r"dim=-4 out of range \[-3, 2\]"):
            Tensor.empty(1, 2, 16).flatten(-4)
        with pytest.raises(IndexError, match=r"dim=3 out of range \[-3, 2\]"):
            Tensor.empty(1, 2, 16).flatten(0, 3)

    def test_clone_is_lazy_separate_and_preserves_state(self):
        source = Tensor.empty((4,), dtype='float32').is_param_(False)
        source.copy_from(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        source.sum().backward()

        cloned = source.clone()
        assert cloned.uop_logical.op_name == 'AFTER'
        assert [u.op_name for u in cloned.uop_logical.src] == ['BUFFER', 'STORE']
        assert cloned.uop_logical.src[0].buffer.raw != source.uop.buffer.raw
        assert cloned.is_param is False
        assert cloned.grad is not None
        assert cloned.grad.uop_logical.op_name == 'AFTER'
        assert cloned.grad.uop_logical.src[0].buffer.raw != source.grad.uop_logical.src[0].buffer.raw
        np.testing.assert_allclose(cloned.numpy(), [1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(cloned.grad.numpy(), np.ones(4, dtype=np.float32))

    def test_detach_is_a_lazy_graph_boundary(self):
        source = Tensor([[1.0, 2.0], [3.0, 4.0]])
        detached = source.detach()

        assert detached.shape == source.shape
        assert detached.dtype == source.dtype
        assert detached.device == source.device
        assert detached.uop_logical.op_name == 'DETACH'
        assert detached.uop_logical.src[0].raw == source.uop_logical.raw
        if detached.uop_physical is not None:
            assert detached.uop_physical.op_name == 'DETACH'
            assert detached.uop_physical.src[0].raw == source.uop.raw

        detached.sum().backward()
        np.testing.assert_allclose(source.grad.numpy(), np.zeros((2, 2), dtype=np.float32))

    def test_contiguous_backward_has_exact_gradient_barrier(self):
        source = Tensor([1.0, -2.0, 3.0])
        result = (source * 2.0).contiguous_backward()

        assert result.uop_logical.op_name == 'CONTIGUOUS_BACKWARD'
        assert result.uop.op_name == 'CONTIGUOUS_BACKWARD'
        assert result.uop_logical.src[0].op_name == 'MUL'
        result.square().sum().backward()
        stack, seen, ops = [source.grad.uop], set(), set()
        while stack:
            uop = stack.pop()
            if uop.raw in seen:
                continue
            seen.add(uop.raw)
            ops.add(uop.op_name)
            stack.extend(uop.src)
        assert 'CONTIGUOUS' in ops
        np.testing.assert_allclose(source.grad.numpy(), [8.0, -16.0, 24.0])

    def test_clone_preserves_scalar_shape_and_accepts_device(self):
        source = Tensor.full((), 3.0)
        cloned = source.clone(device='INTERP')
        assert cloned.shape == ()
        assert cloned.device == 'INTERP'
        assert cloned.uop_logical.op_name == 'AFTER'
        assert cloned.item() == pytest.approx(3.0)

    def test_backward_clones_deviceless_grad_and_accumulates_in_place(self):
        x = Tensor.empty((4,), dtype='float32')
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
        source = Tensor.empty((4,), dtype='float32')
        source.copy_from(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        cloned = source.clone()
        cloned.sum().backward()
        np.testing.assert_allclose(source.grad.numpy(), np.ones(4, dtype=np.float32))
        np.testing.assert_allclose(cloned.grad.numpy(), np.ones(4, dtype=np.float32))

    def test_backward_retains_distinct_wrappers_sharing_one_uop(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0])
        # Pinned Tensor.__init__ wraps an existing current Tensor.uop directly
        # (tensor.py:92-121); retained logical provenance is not executable.
        y = Tensor(x.uop)
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
        assert can_run('add', dtype=dtypes.float16, shape=(4,))
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
            return c[i].store(a[i] + b[i]).end(i).sink(
                arg=KernelInfo(name='custom_add_4')
            )

        a = Tensor([1.0, 2.0, 3.0, 4.0])
        b = Tensor([10.0, 20.0, 30.0, 40.0])
        c = Tensor.empty((4,), dtype='float32')
        out = c.custom_kernel(a, b, fxn=add_kernel)[0]
        np.testing.assert_allclose(out.numpy(), [11.0, 22.0, 33.0, 44.0])

    def test_custom_kernel_range_numeric_scalar_preserves_weakint(self):
        ctx = Tensor.empty((1,))._ctx
        index = UOp.range(ctx, 64, 0)
        offset = index * 64

        assert index.dtype is dtypes.weakint
        assert index.src[0].dtype is dtypes.weakint
        assert offset.dtype is dtypes.weakint
        assert tuple(src.dtype for src in offset.src) == (dtypes.weakint, dtypes.weakint)

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

        a_ref = Tensor(a_np)
        b_ref = Tensor(b_np)
        ((a_ref + b_ref).sum() + (a_ref * b_ref).sum()).backward()

        a = Tensor(a_np)
        b = Tensor(b_np)
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

    def test_custom_kernel_keeps_gradient_callback_while_call_graph_is_live(self):
        runtime = Runtime(device='cpu')
        Tensor = runtime.Tensor

        def build():
            def identity_kernel(out, src):
                out, src = out.flatten(), src.flatten()
                i = UOp.range(out.ctx, out.numel(), 0)
                return out[i].store(src[i]).end(i).sink(
                    arg=KernelInfo(name='callback_lifetime')
                )

            def backward_identity(grad, call):
                return (None, grad)

            src = Tensor([1.0, 2.0, 3.0, 4.0])
            out = Tensor.empty((4,), dtype='float32')
            result = out.custom_kernel(
                src, fxn=identity_kernel, grad_fxn=backward_identity
            )[0]
            return src, out, result, weakref.ref(backward_identity)

        try:
            src, out, result, callback_ref = build()
            gc.collect()
            assert callback_ref() is not None
            result.sum().backward()
            np.testing.assert_allclose(src.grad.numpy(), np.ones(4, dtype=np.float32))
        finally:
            runtime.dispose()
        del src, out, result
        gc.collect()
        assert callback_ref() is None

    def test_custom_kernel_physical_after_preserves_data_gradient(self):
        def identity_kernel(x):
            x = x.flatten()
            i = UOp.range(x.ctx, x.numel(), 0)
            return x[i].store(x[i]).end(i).sink(arg=KernelInfo(name='identity'))

        def backward_identity(grad, call):
            assert call.op_name == 'CALL'
            return (None,)

        x = Tensor.empty((4,), dtype='float32')
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

        out = Tensor.empty((4,), dtype='float32')
        x = Tensor([1.0, 2.0, 3.0, 4.0])
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
        x = Tensor([1.0, 2.0, 3.0, 4.0])
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
        x = Tensor([1.0, 2.0, 3.0, 4.0])
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

        out = Tensor.empty((4,), dtype='float32')
        x = Tensor([1.0, 2.0, 3.0, 4.0])
        y = out.custom_kernel(x, fxn=identity_kernel)[0]
        y.detach().sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), np.zeros(4, dtype=np.float32))
        np.testing.assert_allclose(y.grad.numpy(), np.zeros(4, dtype=np.float32))

        calls = []

        def backward_identity(grad, call):
            calls.append(grad.op_name)
            return (None, (Tensor(grad) + 7).uop)

        out = Tensor.empty((4,), dtype='float32')
        x = Tensor([1.0, 2.0, 3.0, 4.0])
        y = out.custom_kernel(x, fxn=identity_kernel, grad_fxn=backward_identity)[0]
        (y < 0).float().sum().backward()
        assert calls == []
        np.testing.assert_allclose(x.grad.numpy(), np.zeros(4, dtype=np.float32))
        np.testing.assert_allclose(y.grad.numpy(), np.zeros(4, dtype=np.float32))

    def test_custom_kernel_reuses_buffers_after_input_update(self):
        def add_kernel(c, a, b):
            c, a, b = c.flatten(), a.flatten(), b.flatten()
            i = UOp.range(c.ctx, c.numel(), 0)
            return c[i].store(a[i] + b[i]).end(i).sink(
                arg=KernelInfo(name='custom_add_reuse_4')
            )

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

    def test_empty_accepts_symbolic_dimensions_and_expressions_in_any_axis(self):
        n = Variable('N_any_axis', 1, 8)
        m = Variable('M_expression', 1, 6)
        x = Tensor.empty(2, n.bind(3), m.bind(2) * 2)

        assert x.shape[0] == 2
        assert not isinstance(x.shape[1], int)
        assert not isinstance(x.shape[2], int)
        assert x.uop_logical.base.op_name == 'BUFFER'
        assert x.uop_logical.base.src[0].op_name == 'UNIQUE'

        physical = UOp(x._ctx, _ffi._lib.poly_tensor_uop_physical(x._tensor))
        assert physical.base.op_name == 'BUFFER'
        assert physical.base.src[0].op_name == 'CONST'
        alloc = ctypes.c_int64()
        assert _ffi._lib.poly_uop_const_i64(physical.base.src[0].raw, ctypes.byref(alloc)) == 0
        assert alloc.value == 2 * 8 * (6 * 2)

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

    def test_symbolic_numel_repeat_and_reshape_match_llama_repeat_kv(self):
        n = Variable('n', 1, 7).bind(3)
        cache = Tensor.arange(32, dtype='float32').reshape(1, 8, 1, 4).contiguous().realize()
        view = cache[:, :n, :, :]

        numel = view.numel()
        assert isinstance(numel, UOp)
        assert numel.op_name == 'MUL'
        with pytest.raises(AssertionError, match='no data if shape is symbolic'):
            view.numpy()

        repeated = view.repeat((1, 1, 1, 2))
        assert repeated.shape[0] == 1
        assert repeated.shape[1] == view.shape[1]
        assert repeated.shape[2:] == (1, 8)
        assert repeated.uop.op_name == 'RESHAPE'
        permuted = repeated.uop.src[0]
        assert permuted.op_name == 'PERMUTE'
        expanded = permuted.src[0]
        assert expanded.op_name == 'EXPAND'
        assert repeated.uop.src[1].op_name == 'STACK'
        assert len(repeated.uop.src[1].src) == 4

        reshaped = repeated.reshape(1, n, 2, 4)
        assert reshaped.shape[1] == view.shape[1]
        assert reshaped.shape[2:] == (2, 4)
        assert reshaped.uop.op_name == 'RESHAPE'
        assert reshaped.uop.src[1].op_name == 'STACK'
        assert len(reshaped.uop.src[1].src) == 4
        with pytest.raises(ValueError, match='size mismatch'):
            view.reshape(1, n, 3, 4)

        query = Tensor.arange(8, dtype='float32').reshape(1, 2, 1, 4)
        weight = reshaped.transpose(1, 2).transpose(-2, -1)
        scores = query.dot(weight, dtype='float32')
        assert scores.shape[:3] == (1, 2, 1)
        assert scores.shape[3] == view.shape[1]
        assert scores.uop.op_name == 'REDUCE'
        assert scores.uop.src[0].op_name == 'PERMUTE'

        maximum = scores.max(axis=-1, keepdim=True)
        exponential = (scores - maximum.detach()).exp()
        summed = exponential.sum(axis=-1, keepdim=True)
        expected_softmax = exponential * summed.reciprocal()
        softmax = scores.softmax(-1)
        assert softmax.uop.op_name == expected_softmax.uop.op_name == 'MUL'
        assert softmax.shape[3] == view.shape[1]
        assert _ffi._lib.poly_uop_shape_dim(
            softmax._ctx, Tensor._core_uop_raw(softmax._tensor), 3
        ) == view.shape[1].raw

        expected_log_softmax = (scores - maximum.detach()) - summed.log()
        log_softmax = scores.log_softmax(-1)
        assert log_softmax.uop.op_name == expected_log_softmax.uop.op_name == 'ADD'
        assert _ffi._lib.poly_uop_shape_dim(
            log_softmax._ctx, Tensor._core_uop_raw(log_softmax._tensor), 3
        ) == view.shape[1].raw

        max_other_axis = scores.max(axis=1)
        assert max_other_axis.shape == (1, 1, view.shape[1])
        assert _ffi._lib.poly_uop_shape_dim(
            max_other_axis._ctx, Tensor._core_uop_raw(max_other_axis._tensor), 2
        ) == view.shape[1].raw
        values = reshaped.transpose(1, 2)
        attended = softmax.dot(values, dtype='float32')
        assert attended.shape == (1, 2, 1, 4)

        static = Tensor.arange(12, dtype='float32').reshape(1, 3, 1, 4)
        actual = static.repeat((1, 1, 1, 2)).reshape(1, 3, 2, 4).numpy()
        expected = np.arange(12, dtype=np.float32).reshape(1, 3, 1, 4)
        expected = np.tile(expected, (1, 1, 1, 2)).reshape(1, 3, 2, 4)
        np.testing.assert_array_equal(actual, expected)

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
        assert (x * 0.25).dtype is dtypes.weakfloat
        assert (0.25 * x).dtype is dtypes.weakfloat
        assert (x + 0.25).dtype is dtypes.weakfloat
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

    def test_unequal_width_bitcast_matches_pinned_lane_order(self):
        wide = Tensor.full((8,), 1, dtype='uint8').bitcast('uint32')
        narrow = Tensor.full((2,), 1, dtype='uint32').bitcast('uint8')
        assert wide.shape == (2,)
        assert narrow.shape == (8,)
        np.testing.assert_array_equal(wide.numpy(), [0x01010101, 0x01010101])
        np.testing.assert_array_equal(narrow.numpy(), [1, 0, 0, 0, 1, 0, 0, 0])

    def test_bitcast_view_assign_matches_current_tinygrad(self):
        a = Tensor([1.0, 2.0, 3.0, 4.0], dtype='float32').realize()
        view = a.bitcast('uint32')
        view.assign(Tensor(
            [0x40800000, 0x40400000, 0x40000000, 0x3f800000],
            dtype='uint32',
        )).realize()
        np.testing.assert_array_equal(a.numpy(), [4.0, 3.0, 2.0, 1.0])

    def test_unequal_width_bitcast_preserves_symbolic_last_axis(self):
        n = Variable('n_bitcast', 4, 8).bind(4)
        wide = Tensor.empty(n, dtype='uint8').bitcast('uint32')
        assert wide.uop_physical.op_name == 'BITCAST'
        assert wide.shape[0].op_name == 'FLOORDIV'
        last_source = wide.shape[0].src[0]
        assert last_source.is_bound_var
        assert last_source == n.uop
        assert wide.shape[0].src[1].op_name == 'CONST'

        with pytest.raises(RuntimeError, match='unsupported size in bitcast'):
            Tensor.empty(3, dtype='uint8').bitcast('uint32')
        with pytest.raises(RuntimeError, match='bitcast requires concrete dtypes'):
            Tensor.full((1,), 1.0, buffer=False).bitcast('uint32')

    def test_zeros(self):
        t = Tensor.zeros(3, 4)
        assert t.shape == (3, 4)
        np.testing.assert_allclose(t.numpy(), np.zeros((3, 4)))

    def test_full_defaults_to_writable_buffer_and_supports_buffer_false(self):
        t = Tensor.full((2,), 3.0)
        assert t.dtype is dtypes.float32
        assert t.uop.op_name == 'AFTER'
        assert t.uop.src[0].op_name == 'BUFFER'
        assert t.uop.src[0].dtype == dtypes.float32
        assert t.uop.src[1].op_name == 'STORE'
        assert t.uop.src[1].src[0] == t.uop.src[0]
        assert t.uop.src[1].src[1].op_name == 'EXPAND'
        assert t.uop.src[1].src[1].dtype == dtypes.weakfloat
        t.realize()
        assert t.uop.op_name == 'BUFFER'
        t.assign(Tensor([4.0, 5.0], dtype=t.dtype)).realize()
        np.testing.assert_allclose(t.numpy(), [4.0, 5.0])

        broadcast = Tensor.full((2,), 3.0, buffer=False)
        assert broadcast.uop_logical.op_name == 'EXPAND'
        assert broadcast.uop_physical == broadcast.uop_logical
        assert broadcast.dtype is dtypes.weakfloat
        np.testing.assert_allclose(broadcast.numpy(), [3.0, 3.0])

        with pytest.raises(RuntimeError, match='poly_tensor_full'):
            Tensor.full((2,), 3.0, dtype=dtypes.weakfloat)
        with pytest.raises(RuntimeError, match='poly_tensor_empty'):
            Tensor.empty(2, dtype=dtypes.weakfloat)

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

    def test_fp8_host_values_match_current_tinygrad(self):
        values = [-np.inf, -1.5, -0.0, 0.0, 0.1, 1.0, 1.5, 448.0, np.inf, np.nan]
        expected = {
            'fp8e4m3': [np.nan, -1.5, -0.0, 0.0, 0.1015625, 1.0, 1.5, 448.0, np.nan, np.nan],
            'fp8e5m2': [-np.inf, -1.5, -0.0, 0.0, 0.09375, 1.0, 1.5, 448.0, np.inf, np.nan],
            'fp8e4m3fnuz': [np.nan, -1.5, 0.0, 0.0, 0.1015625, 1.0, 1.5, 240.0, np.nan, np.nan],
            'fp8e5m2fnuz': [np.nan, -1.5, 0.0, 0.0, 0.09375, 1.0, 1.5, 448.0, np.nan, np.nan],
        }
        for dtype, wanted in expected.items():
            tensor = Tensor(values, dtype=dtype)
            assert tensor.dtype is getattr(dtypes, dtype)
            actual = tensor.cast('float32').numpy()
            np.testing.assert_equal(actual, np.asarray(wanted, dtype=np.float32))

    def test_numpy_zero_dim_constructs_scalar_const(self):
        cases = [
            (np.array(7, dtype=np.int32), None, 'int32', 7.0),
            (np.array(1.5, dtype=np.float64), None, 'float64', 1.5),
            (np.array(2.0, dtype=np.float64), 'bfloat16', 'bfloat16', 2.0),
        ]
        for data, dtype, expected_dtype, expected in cases:
            t = Tensor(data, dtype=dtype)
            assert t.shape == ()
            assert t.dtype is getattr(dtypes, expected_dtype)
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

        Tensor.manual_seed(1337)
        pinned = Tensor.rand(8).numpy()
        expected_bits = np.array([
            0x3EFA31A0, 0x3EB22B7C, 0x3F28C97E, 0x3F22EFFE,
            0x3EF13C94, 0x3E10DD30, 0x3E8E61EC, 0x3D4C9DC0,
        ], dtype=np.uint32)
        np.testing.assert_array_equal(pinned.view(np.uint32), expected_bits)

    def test_runtime_random_creators_and_seed_are_context_local(self):
        with Runtime(device='interp') as runtime:
            runtime.Tensor.manual_seed(123)
            runtime_a = runtime.Tensor.rand(8).numpy()
            runtime.Tensor.manual_seed(123)
            runtime_b = runtime.Tensor.rand(8).numpy()
            np.testing.assert_array_equal(runtime_a, runtime_b)

            creators = (
                ('rand', (4,)),
                ('uniform', (4,)),
                ('scaled_uniform', (2, 2)),
                ('glorot_uniform', (2, 2)),
                ('kaiming_uniform', (2, 2)),
                ('randint', (4,)),
                ('randperm', (4,)),
            )
            for name, shape in creators:
                value = getattr(runtime.Tensor, name)(*shape)
                assert value._ctx == runtime._ctx, name

            with pytest.raises(ValueError, match='same Polygrad context'):
                runtime.Tensor.ones(2) + Tensor.uniform(2)

        Tensor.manual_seed(999)
        expected_first = Tensor.rand(4).numpy()
        expected_second = Tensor.rand(4).numpy()
        Tensor.manual_seed(999)
        actual_first = Tensor.rand(4).numpy()
        with Runtime(device='interp') as runtime:
            runtime.Tensor.manual_seed(123)
        actual_second = Tensor.rand(4).numpy()
        np.testing.assert_array_equal(actual_first, expected_first)
        np.testing.assert_array_equal(actual_second, expected_second)

    def test_runtime_can_run_uses_runtime_preferred_device(self):
        with Runtime(device='host') as runtime:
            assert runtime.can_run('add', dtype='float32', shape=(2,))

    def test_uniform_bounds_determinism_and_validation(self):
        Tensor.manual_seed(42)
        a = Tensor.uniform(64, low=-2, high=3).numpy()
        Tensor.manual_seed(42)
        b = Tensor.uniform(64, low=-2, high=3).numpy()
        np.testing.assert_array_equal(a, b)
        assert np.all(a >= -2)
        assert np.all(a < 3)
        with pytest.raises(ValueError, match='low < high'):
            Tensor.uniform(2, low=1, high=1)

    def test_scaled_uniform_matches_pinned_expression(self):
        Tensor.manual_seed(42)
        actual = Tensor.scaled_uniform(2, 3)
        Tensor.manual_seed(42)
        expected = Tensor.uniform(2, 3, low=-1.0, high=1.0).mul(6 ** -0.5)
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())

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

    @pytest.mark.parametrize('logical', ['never', 'always', 'until_realize'])
    @pytest.mark.parametrize('realized', [False, True])
    def test_copy_from_host_input_uses_current_storage_and_logical_policy(self, logical, realized):
        x = Tensor(np.array([1, 2, 3], dtype=np.float32), logical=logical)
        retained = x.uop_logical
        if realized:
            x.realize()
        before = x.uop_physical
        x.copy_from([4, 5, 6])
        current = x.uop_physical
        assert current.op_name == 'BUFFER'
        assert len(current.src) == 1 and current.src[0].op_name == 'CONST'
        if realized:
            assert current == before
        if logical == 'never':
            assert x.uop_logical is None
        elif logical == 'always':
            assert x.uop_logical == retained
        else:
            assert x.logical_state == 'retired'
        x.copy_from([7, 8, 9])
        assert x.uop_physical == current
        gc.collect()
        np.testing.assert_array_equal(x.numpy(), [7, 8, 9])

    def test_copy_from_validates_before_materialization_and_runs_pending_assign(self):
        x = Tensor(np.array([1, 2, 3], dtype=np.float32), logical='always')
        before = x.uop_physical
        with pytest.raises(ValueError, match='size mismatch'):
            x.copy_from([9])
        assert x.uop_physical == before
        np.testing.assert_array_equal(x.numpy(), [1, 2, 3])
        x.assign(Tensor([10, 20, 30], dtype='float32'))
        x.copy_from([4, 5, 6])
        np.testing.assert_array_equal(x.numpy(), [4, 5, 6])

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
    def test_runtime_dispose_releases_its_retained_jits_first(self):
        runtime = Runtime(device='interp')
        other_runtime = Runtime(device='interp')

        @Jit
        def add_one(value):
            return (value + 1).realize()

        @Jit
        def double(value):
            return (value * 2).realize()

        add_one(runtime.Tensor([1.0, 2.0]))
        add_one(runtime.Tensor([3.0, 4.0]))
        double(other_runtime.Tensor([1.0, 2.0]))
        double(other_runtime.Tensor([3.0, 4.0]))
        assert add_one.captured
        assert double.captured
        assert add_one._jit
        assert add_one._ctx == runtime._ctx
        assert double._jit

        runtime.dispose()
        assert add_one._jit is None
        assert add_one._ctx is None
        assert double._jit
        np.testing.assert_allclose(
            double(other_runtime.Tensor([5.0, 6.0])).numpy(), [10.0, 12.0]
        )
        other_runtime.dispose()
        assert double._jit is None

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
        # (engine/jit.py:267-293; paired tinygrad symbolic-realize probe).
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
            return c[i].store(a[i] + b[i]).end(i).sink(
                arg=KernelInfo(name='jit_custom_add_4')
            )

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
            return c[i].store(a[i] + b[i]).end(i).sink(
                arg=KernelInfo(name='compile_custom_add_4')
            )

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

    def test_compile_captures_current_custom_sum_and_replays_after_input_update(self):
        def sum_kernel(out, a):
            out, a = out.flatten(), a.flatten()
            r = UOp.range(out.ctx, 8, 0, AxisType.REDUCE)
            acc = out[0].set(0.0)
            acc = acc[0].set(acc.after(r)[0] + a[r], end=r)
            return acc.sink(arg=KernelInfo(name='custom_sum_8', opts_to_apply=()))

        def f(a):
            out = Tensor.empty((1,), dtype='float32')
            return out.custom_kernel(a, fxn=sum_kernel)[0]

        a = Tensor.empty((8,), dtype='float32')
        a.copy_from(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32))
        compiled = pg_compile(f, [a])
        assert compiled.schedule_count == 1
        np.testing.assert_allclose(compiled.run([a]).numpy(), [36.0])

        a.copy_from(np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float32))
        np.testing.assert_allclose(compiled.run([a]).numpy(), [44.0])
        assert compiled.stats()['run_count'] == 2

    def test_compile_captures_current_custom_kernel_multi_output_addmul(self):
        def addmul_kernel(out0, out1, a, b):
            out0, out1, a, b = out0.flatten(), out1.flatten(), a.flatten(), b.flatten()
            i = UOp.range(out0.ctx, 4, 0)
            st0 = out0[i].store(a[i] + b[i])
            st1 = out1[i].store(a[i] * b[i])
            return st0.group(st1).end(i).sink(
                arg=KernelInfo(name='custom_addmul_4')
            )

        def f(a, b):
            out0 = Tensor.empty((4,), dtype='float32')
            out1 = Tensor.empty((4,), dtype='float32')
            outs = out0.custom_kernel(out1, a, b, fxn=addmul_kernel)
            return [outs[0], outs[1]]

        a = Tensor.empty((4,), dtype='float32')
        b = Tensor([1.0, 2.0, 3.0, 4.0])
        a.copy_from(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        compiled = pg_compile(f, [a, b])
        out0, out1 = compiled.run([a, b])
        np.testing.assert_allclose(out0.numpy(), [2.0, 4.0, 6.0, 8.0])
        np.testing.assert_allclose(out1.numpy(), [1.0, 4.0, 9.0, 16.0])

        a.copy_from(np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32))
        out0, out1 = compiled.run([a, b])
        np.testing.assert_allclose(out0.numpy(), [3.0, 5.0, 7.0, 9.0])
        np.testing.assert_allclose(out1.numpy(), [2.0, 6.0, 12.0, 20.0])
        assert compiled.stats()['run_count'] == 2

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
            acc = out[0].set(0.0)
            acc = acc[0].set(acc.after(r)[0] + y[r], end=r)
            return acc.sink(arg=KernelInfo(
                name='custom_consumer_readback_rebind', opts_to_apply=()
            ))

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
            return out[i].store(selected).end(i).sink(
                arg=KernelInfo(name='custom_select')
            )

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
            zero = UOp.const(0, ctx=out.ctx)
            gate = zero.lt(1)
            bad = out.index(gate)
            assert bad is not None
            return bad.store(out.index(zero)).sink(
                arg=KernelInfo(name='invalid_bool_index')
            )

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
            return out[i].store((col * 10 + row).cast(dtypes.float32)).end(i).sink(
                arg=KernelInfo(name='custom_floor_div_mod')
            )

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
            return out[i].store(q).group(out[i + x.numel()].store(r)).end(i).sink(
                arg=KernelInfo(name='custom_signed_div_mod'),
            )

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
            return s0.group(s1).end(i).sink(arg=KernelInfo(name='custom_numeric_literals'))

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

    def test_div_rounding_modes_match_tinygrad_topology(self):
        ints = Tensor([-7, -4, 4, 7], dtype=dtypes.int32)
        int_divisors = Tensor([3, -3, 3, -3], dtype=dtypes.int32)
        trunc_int = ints.div(int_divisors, rounding_mode='trunc')
        floor_int = ints.div(int_divisors, rounding_mode='floor')
        assert trunc_int.uop.op_name == 'CDIV'
        assert floor_int.uop.op_name == 'FLOORDIV'
        np.testing.assert_array_equal(trunc_int.numpy(), [-2, 1, 1, -2])
        np.testing.assert_array_equal(floor_int.numpy(), [-3, 1, 1, -3])

        floats = Tensor([-7.5, -4.5, 4.5, 7.5], dtype=dtypes.float32)
        float_divisors = Tensor([2.0, -2.0, 2.0, -2.0], dtype=dtypes.float32)
        trunc_float = floats.div(float_divisors, rounding_mode='trunc')
        floor_float = floats.div(float_divisors, rounding_mode='floor')
        assert trunc_float.uop.op_name == 'TRUNC'
        assert floor_float.uop.op_name == 'WHERE'
        np.testing.assert_array_equal(trunc_float.numpy(), [-3.0, 2.0, 2.0, -3.0])
        np.testing.assert_array_equal(floor_float.numpy(), [-4.0, 2.0, 2.0, -4.0])

        with pytest.raises(RuntimeError, match="rounding_mode='nearest' is not supported"):
            ints.div(int_divisors, rounding_mode='nearest')

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
            assert out.uop.src[0].op_name == 'CONST'
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
        assert add_int.dtype is dtypes.weakint
        np.testing.assert_allclose(add_int.numpy(), [3, 2])

        sub_bool = x.sub(True)
        assert sub_bool.dtype is dtypes.bool
        assert sub_bool.uop.op_name == 'ADD'
        assert any(src.op_name == 'CMPNE' for src in sub_bool.uop.src)
        np.testing.assert_array_equal(sub_bool.numpy(), [True, False])

        mul_int = x.mul(2)
        assert mul_int.dtype is dtypes.weakint
        np.testing.assert_allclose(mul_int.numpy(), [2, 0])

        pow_int = x.pow(2)
        assert pow_int.dtype is dtypes.weakint
        np.testing.assert_allclose(pow_int.numpy(), [1, 0])

        neg = x.neg()
        assert neg.dtype is dtypes.bool
        assert neg.uop.op_name == 'CMPNE'
        assert [src.op_name for src in neg.uop.src] == ['BUFFER', 'CONST']
        np.testing.assert_array_equal(neg.numpy(), [False, True])

        logical_not = x.logical_not()
        assert logical_not.dtype is dtypes.bool
        assert logical_not.uop.op_name == 'CMPNE'
        assert [src.op_name for src in logical_not.uop.src] == ['BUFFER', 'CONST']
        np.testing.assert_array_equal(logical_not.numpy(), [False, True])

    def test_where_scalar_branch_shapes_before_promotion(self):
        cond = Tensor([[True, False], [False, True]], dtype='bool')
        out = cond.where(0, -float('inf'))

        # tinygrad@2026-08-22/a9069c177a9d mixin/elementwise.py:422-435:
        # scalar branches remain weak CONSTs; shape broadcasting is implicit.
        zero_branch = out.uop.src[1]
        assert zero_branch.op_name == 'CONST'
        assert zero_branch.dtype == dtypes.weakfloat
        np.testing.assert_array_equal(
            out.numpy(),
            np.array([[0.0, -np.inf], [-np.inf, 0.0]], dtype=np.float32),
        )

    def test_logaddexp_softplus_mish_match_pinned_graph(self):
        assert not hasattr(Tensor, '_physicalize_result_for')
        x = Tensor([
            [-20.0, -3.0, -0.0, 2.0, 20.0],
            [1.0, -1.0, 4.0, -4.0, 0.5],
        ])
        other = Tensor([[-2.0], [3.0]])

        def pinned_logaddexp(a, b):
            a, b, _ = a._broadcasted(b)
            m = a.maximum(b)
            return ((a - m).exp() + (b - m).exp()).log() + m

        def pinned_softplus(value, beta=1.0):
            return (1 / beta) * pinned_logaddexp(value * beta, 0.0)

        pairs = [
            (x.logaddexp(0.0), pinned_logaddexp(x, 0.0)),
            (x.logaddexp(other), pinned_logaddexp(x, other)),
            (x.softplus(), pinned_softplus(x)),
            (x.softplus(beta=2.0), pinned_softplus(x, 2.0)),
            (x.mish(), x * pinned_softplus(x).tanh()),
        ]
        for actual, expected in pairs:
            assert actual.uop.raw == expected.uop.raw
            assert actual.uop_logical.raw == expected.uop_logical.raw
            np.testing.assert_allclose(
                actual.numpy(), expected.numpy(), rtol=1e-6, atol=1e-6
            )

    def test_log1p_expm1_use_core_tensor_roots(self):
        from polygrad.tensor import _uop_wrap

        assert not hasattr(Tensor, '_physicalize_result_for')
        x = Tensor([-1e-6, 0.0, 1e-6, 0.25], device='cpu')
        for name, expected_values in (
            ('log1p', np.log1p(np.asarray([-1e-6, 0.0, 1e-6, 0.25]))),
            ('expm1', np.expm1(np.asarray([-1e-6, 0.0, 1e-6, 0.25]))),
        ):
            actual = getattr(x, name)()
            raw_fn = getattr(_ffi._lib, f'poly_{name}')
            expected_logical = _uop_wrap(
                x._ctx, raw_fn(x._ctx, x.uop_logical.raw)
            )
            expected_physical = _uop_wrap(x._ctx, raw_fn(x._ctx, x.uop.raw))
            assert actual.uop_logical.raw == expected_logical.raw
            assert actual.uop.raw == expected_physical.raw
            np.testing.assert_allclose(
                actual.numpy(), expected_values, rtol=1e-6, atol=1e-7
            )

        moved = Tensor.empty((4,), device='cpu').realize().to('cuda').to('cpu')
        for name in ('log1p', 'expm1'):
            actual = getattr(moved, name)()
            raw_fn = getattr(_ffi._lib, f'poly_{name}')
            expected_physical = _uop_wrap(
                moved._ctx, raw_fn(moved._ctx, moved.uop.raw)
            )
            assert actual.uop.raw == expected_physical.raw

    def test_mixed_dtype_comparison_where_promotes_like_tinygrad(self):
        x = Tensor(np.arange(16, dtype=np.float32).reshape(4, 4))
        out = (Tensor.full((4, 4), 7, dtype='int32') > x).where(
            x, Tensor.full((4, 4), -2, dtype='int32')
        ).sum(axis=0)
        np.testing.assert_allclose(out.numpy(), [0, 2, 4, -3])

    def test_named_integer_true_division_matches_tinygrad(self):
        x = Tensor([3, 4], dtype='int32')

        named = x.div(2)
        assert named.dtype is dtypes.float32
        np.testing.assert_allclose(named.numpy(), [1.5, 2.0], rtol=1e-6, atol=1e-6)

        operator = x / 2
        assert operator.dtype is dtypes.float32
        np.testing.assert_allclose(operator.numpy(), [1.5, 2.0], rtol=1e-6, atol=1e-6)

        reverse = x.div(2, reverse=True)
        assert reverse.dtype is dtypes.float32
        np.testing.assert_allclose(reverse.numpy(), [0.6666667, 0.5], rtol=1e-6, atol=1e-6)

        reverse_operator = 2 / x
        assert reverse_operator.dtype is dtypes.float32
        np.testing.assert_allclose(reverse_operator.numpy(), [0.6666667, 0.5], rtol=1e-6, atol=1e-6)

        tensor_divisor = x.div(Tensor([2, 2], dtype='int32'))
        assert tensor_divisor.dtype is dtypes.float32
        np.testing.assert_allclose(tensor_divisor.numpy(), [1.5, 2.0], rtol=1e-6, atol=1e-6)

    def test_named_pow_int_base_float_exponent_matches_tinygrad(self):
        x = Tensor([2, 3], dtype='int32')

        named = x.pow(2.0)
        assert named.dtype is dtypes.weakfloat
        np.testing.assert_allclose(named.numpy(), [4, 9])

        operator = x ** 2.0
        assert operator.dtype is dtypes.weakfloat
        np.testing.assert_allclose(operator.numpy(), [4, 9])

        reverse = x.pow(2.0, reverse=True)
        assert reverse.dtype is dtypes.weakfloat
        np.testing.assert_allclose(reverse.numpy(), [4.0, 8.0])

    def test_named_pow_negative_scalar_int_validation_matches_tinygrad(self):
        x = Tensor([2, 3], dtype='int32')

        with pytest.raises(RuntimeError, match='base needs to be float'):
            x.pow(-1)
        with pytest.raises(RuntimeError, match='base needs to be float'):
            x ** -1

        tensor_exponent = x.pow(Tensor([-1, -2], dtype='int32'))
        assert tensor_exponent.dtype is dtypes.int32
        assert tensor_exponent.uop_physical.op_name == 'POW'

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

    def test_movement_max_shape_preserves_symbolic_dimensions(self):
        t = Tensor.empty(2, UOp.variable('max_extent', 1, 8), 3)
        before = t.uop_physical.raw
        assert t.max_shape == (2, 8, 3)
        assert t.max_numel() == 48
        assert not isinstance(t.shape[1], int)
        assert t.uop_physical.raw == before
        assert Tensor(3).max_shape == ()
        assert Tensor(3).max_numel() == 1
        assert Tensor.empty(0, 3).max_numel() == 0

    def test_movement_optional_shrink_and_zero_extent(self):
        t = Tensor([[0, 1, 2], [3, 4, 5]])
        y = t.shrink((None, (1, 3)))
        expected = _ffi._lib.poly_shrink(t._ctx, t.uop_physical, (ctypes.c_int64 * 4)(0, 2, 1, 3), 2)
        assert y.uop_physical.raw == expected
        assert len(y.uop_physical.src) == 3
        assert y.tolist() == [[1, 2], [4, 5]]
        empty = t.shrink(((1, 1), None))
        assert empty.shape == (0, 3)
        assert empty.numpy().shape == (0, 3)

    def test_movement_symbolic_shrink_keeps_exact_sources(self):
        extent = UOp.variable('shrink_extent', 1, 8)
        t = Tensor.empty(2, extent, 3)
        y = t.shrink(((0, 1), None, (1, 3)))
        start_nodes = [UOp.const(v) for v in (0, 0, 1)]
        size_nodes = [UOp.const(1), extent, UOp.const(2)]
        starts = (ctypes.c_void_p * 3)(*(v.raw for v in start_nodes))
        sizes = (ctypes.c_void_p * 3)(*(v.raw for v in size_nodes))
        expected = _ffi._lib.poly_shrink_uop(t._ctx, t.uop_physical, starts, sizes, 3)
        assert y.uop_physical.raw == expected
        assert y.shape[1] == extent
        assert y.shrink((None, None, None)) is y

    @pytest.mark.parametrize('operation', [
        lambda t: t.shrink(((0, 2), (0, 3))),
        lambda t: t.shrink((None, None)),
        lambda t: t.shrink_to(None, 3),
        lambda t: t.pad_to(None, 3, value=5),
        lambda t: t.flip(()),
        lambda t: t[()], lambda t: t[...], lambda t: t[:],
    ])
    def test_movement_noops_preserve_tensor_identity(self, operation):
        t = Tensor([[0, 1, 2], [3, 4, 5]])
        assert operation(t) is t
        assert t.tolist() == [[0, 1, 2], [3, 4, 5]]

    def test_movement_pad_shrink_to_values_and_graphs(self):
        t = Tensor([[0, 1, 2], [3, 4, 5]])
        for value in (0, -1, True, 1.5):
            y = t.pad_to((3, 5), value=value)
            expected = t.pad(((0, 1), (0, 2)), value=value)
            assert y.uop_physical.raw == expected.uop_physical.raw
            np.testing.assert_array_equal(y.numpy(), expected.numpy())
        y = t.shrink_to(None, 2)
        assert y.shape == (2, 2)
        assert y.tolist() == [[0, 1], [3, 4]]
        assert t.shrink_to((1, 2)).tolist() == [[0, 1]]
        scalar = Tensor(1)
        assert scalar.pad_to(()) is scalar
        assert Tensor.empty(0, 3).pad_to(2, 3, value=7).tolist() == [[7]*3]*2

    @pytest.mark.parametrize('operation', [
        lambda t: t.shrink((None,)), lambda t: t.shrink((None, None, None)),
        lambda t: t.shrink_to(1), lambda t: t.shrink_to(1, 2, 3),
        lambda t: t.pad_to(3), lambda t: t.pad_to(3, 4, 5),
        lambda t: t.pad_to(1, 3),
    ])
    def test_movement_helpers_validate_dimensions(self, operation):
        with pytest.raises(ValueError):
            operation(Tensor.empty(2, 3))

    def test_basic_indices_apply_one_shrink_before_dimension_collapse(self):
        base = Tensor.zeros(2, 1, 8, 1, 4).contiguous().realize()
        indexed = base[0, :, 0:3, :, :]
        assert indexed.shape == (1, 3, 1, 4)
        assert indexed.uop.op_name == 'RESHAPE'
        assert indexed.uop.src[0].op_name == 'SHRINK'
        seen, stack, shrink_count = set(), [indexed.uop], 0
        while stack:
            node = stack.pop()
            if node.raw in seen:
                continue
            seen.add(node.raw)
            shrink_count += node.op_name == 'SHRINK'
            stack.extend(node.src)
        assert shrink_count == 1
        np.testing.assert_allclose(indexed.numpy(), np.zeros((1, 3, 1, 4), dtype=np.float32))

        injected = base[0, None, :, 1:4, :, :]
        assert injected.shape == (1, 1, 3, 1, 4)
        assert injected.uop.op_name == 'SHRINK'

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

        # Current Tinygrad keeps the Python scalar kind through
        # _pad_constant; a float fill promotes an integer source.
        promoted = Tensor([1, 2], dtype=dtypes.int32).pad(((1, 1),), value=5.5)
        assert promoted.dtype is dtypes.weakfloat
        np.testing.assert_allclose(promoted.numpy(), [5.5, 1.0, 2.0, 5.5])
        bool_fill = Tensor([1, 2], dtype=dtypes.int32).pad(((1, 1),), value=True)
        assert bool_fill.dtype is dtypes.int32
        np.testing.assert_array_equal(bool_fill.numpy(), [1, 1, 2, 1])

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
        assert out.dtype is dtypes.weakint
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

    def test_scatter_construction_bypasses_frontend_substitution(self):
        assert not hasattr(Tensor, '_physicalize_result_for')
        base = Tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        idx = Tensor(np.array([[0, 1, 1, 3, 4]], dtype=np.int32), dtype='int32')
        src = Tensor([[6.0, 7.0, 8.0, 9.0, 10.0]])
        np.testing.assert_allclose(
            base.scatter(1, idx, src).numpy(), [[6, 8, 3, 9, 10]]
        )
        np.testing.assert_allclose(
            base.scatter_reduce(1, idx, src, 'sum').numpy(),
            [[7, 17, 3, 13, 15]],
        )


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
        a = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        b = a[::2]  # [1, 3, 5, 7]
        c = b.sum()
        c.backward()
        # grad should be [1,0,1,0,1,0,1,0]
        np.testing.assert_allclose(a.grad.numpy(), [1, 0, 1, 0, 1, 0, 1, 0])


class TestReduce:
    @pytest.mark.parametrize('op', ['all', 'any', 'cumsum', 'cumprod', 'cummax', 'cummin'])
    def test_scan_owners_surface_and_values(self, op):
        x = Tensor([-3, -1, -2, -1], dtype='int8')
        out = getattr(x, op)(0)
        expected = {'all': True, 'any': True, 'cumsum': [-3,-4,-6,-7],
                    'cumprod': [-3,3,-6,6], 'cummax': [-3,-1,-1,-1], 'cummin': [-3,-3,-3,-3]}
        if op in ('cummax', 'cummin'):
            values, indices = out
            np.testing.assert_array_equal(indices.numpy(), [0,1,1,1] if op == 'cummax' else [0,0,0,0])
            assert indices.dtype == dtypes.int32
        else:
            values = out
        np.testing.assert_array_equal(values.numpy(), expected[op])
        assert values.dtype == (dtypes.bool if op in ('all','any') else dtypes.int32 if op == 'cumsum' else dtypes.int8)

    @pytest.mark.parametrize('op', ['all', 'any'])
    def test_scan_owners_boolean_axes(self, op):
        data = np.array([[0, np.nan, -2], [1, 0, 3]], dtype=np.float32)
        x = Tensor(data)
        for axis in (None, 0, -1, (0, 1), ()):
            for keepdim in (False, True):
                out = getattr(x, op)(axis, keepdim)
                expected = getattr(np, op)(data, axis=axis, keepdims=keepdim)
                assert out.shape == expected.shape
                np.testing.assert_array_equal(out.numpy(), expected)
        for shape in ((), (0,), (2, 0), (0, 2)):
            data = np.ones(shape, dtype=np.int8)
            x = Tensor(data)
            for axis in (0, -1):
                expected = getattr(np, op)(data, axis=axis)
                np.testing.assert_array_equal(getattr(x, op)(axis).numpy(), expected)

    @pytest.mark.parametrize('op', ['cumsum', 'cumprod', 'cummax', 'cummin'])
    def test_scan_owners_empty_scalar_and_axes(self, op):
        for shape in ((), (0,), (2, 0), (0, 2)):
            x = Tensor(np.ones(shape, dtype=np.int8))
            for axis in (0, -1):
                result = getattr(x, op)(axis)
                pair = isinstance(result, tuple)
                values = result[0] if pair else result
                assert values.shape == shape
                assert values.dtype == (dtypes.int32 if op == 'cumsum' else dtypes.int8)
                np.testing.assert_array_equal(values.numpy(), np.ones(shape))
                if pair:
                    assert result[1].shape == shape
                    np.testing.assert_array_equal(result[1].numpy(), np.zeros(shape))
            for axis in (max(1, len(shape)), -max(1, len(shape))-1):
                with pytest.raises(IndexError): getattr(x, op)(axis)

    @pytest.mark.parametrize('dtype', ['bool', 'uint8', 'int8', 'uint64', 'int64', 'float16', 'float32'])
    def test_scan_owners_dtypes_and_prefix_indices(self, dtype):
        raw = [[1, 0, 1, 1], [0, 1, 0, 1]] if dtype == 'bool' else [[3, 1, 2, 1], [0, 3, 0, 3]]
        if dtype == 'uint64': raw = [[2**64-1, 0, 2**64-2, 0], [5, 7, 5, 7]]
        if dtype == 'int64': raw = [[-2**63, 0, -2**63+1, 0], [5, 7, 5, 7]]
        data = np.array(raw, dtype=dtype)
        x = Tensor(data)
        for axis in (0, -1):
            for op in ('cummax', 'cummin'):
                values, indices = getattr(x, op)(axis)
                expected = (np.maximum if op == 'cummax' else np.minimum).accumulate(data, axis=axis)
                moved = np.moveaxis(data, axis, -1)
                arg = np.argmax if op == 'cummax' else np.argmin
                expected_idx = np.stack([arg(moved[..., :i+1], axis=-1) for i in range(moved.shape[-1])], axis=-1)
                np.testing.assert_array_equal(values.numpy(), expected)
                np.testing.assert_array_equal(indices.numpy(), np.moveaxis(expected_idx, -1, axis))
            # Pinned sum promotes small integers; prod retains storage dtype.
            for op in ('cumsum', 'cumprod'):
                out = getattr(x, op)(axis)
                result_dtype = 'int32' if op == 'cumsum' and dtype in ('bool', 'int8') else 'uint32' if op == 'cumsum' and dtype == 'uint8' else dtype
                assert out.dtype == getattr(dtypes, result_dtype)
                expected = getattr(np, op)(data, axis=axis, dtype=result_dtype)
                np.testing.assert_array_equal(out.numpy(), expected)

    @pytest.mark.parametrize('length', [512, 513, 1025])
    def test_scan_owners_split_boundary_and_gradients(self, length):
        data = np.ones((length, 2), dtype=np.float32)
        data[0] = [2, 3]
        x = Tensor(data)
        for op in ('cumsum', 'cumprod'):
            np.testing.assert_array_equal(getattr(x, op)(0).numpy(), getattr(np, op)(data, axis=0))
        x.cumsum(0).sum().backward()
        np.testing.assert_array_equal(x.grad.numpy(), np.broadcast_to(np.arange(length, 0, -1)[:, None], data.shape))

    @pytest.mark.parametrize('data,expected', [
        ([2.,3.,4.], [16.,10.,6.]), ([2.,0.,4.], [1.,10.,0.]),
        ([0.,3.,0.], [4.,0.,0.]),
    ])
    def test_scan_owners_product_gradients(self, data, expected):
        x = Tensor(data)
        x.cumprod(0).sum().backward()
        np.testing.assert_array_equal(x.grad.numpy(), expected)

    @pytest.mark.parametrize('dtype,values', [
        ('uint8', [[0, 1, 255], [7, 3, 2]]),
        ('int8', [[-128, -1, 127], [7, 3, 2]]),
        ('uint16', [[0, 1, 65535], [7, 3, 2]]),
        ('int16', [[-32768, -1, 32767], [7, 3, 2]]),
        ('uint32', [[0, 1, 2**32-1], [7, 3, 2]]),
        ('int32', [[-2**31, -1, 2**31-1], [7, 3, 2]]),
        ('uint64', [[0, 1, 2**64-1], [7, 3, 2]]),
        ('int64', [[-2**63, -1, 2**63-1], [7, 3, 2]]),
        ('bool', [[False, True, True], [True, True, True]]),
        ('float32', [[-3, -1, 127], [7, 3, 2]]),
    ])
    def test_min_inverse_matches_pinned(self, dtype, values):
        data = np.asarray(values, dtype=dtype)
        x = Tensor(data)
        for axis, keepdim in [(None, False), (0, False), (-1, True), ((0, 1), True), ((), False)]:
            out = x.min(axis, keepdim)
            inverse_op = 'MUL' if dtype == 'float32' else 'CMPNE' if dtype == 'bool' else 'XOR'
            assert out.uop_physical.op_name == inverse_op
            if axis is None:
                reduced = out.uop_physical.src[0]
                assert reduced.op_name == 'REDUCE'
                assert reduced.src[0].op_name == inverse_op
            assert out.dtype == x.dtype
            np.testing.assert_array_equal(out.numpy(), data.min(axis=axis, keepdims=keepdim))

    @pytest.mark.parametrize('power', [0.2, 1.2, -0.2])
    def test_pow_negative_fraction_constant_matches_buffer(self, power):
        for value in [-28.0, [-28.0]]:
            result = Tensor(value, dtype='float32').pow(power)
            assert result.uop_physical.op_name == 'POW'
            assert np.isnan(result.numpy()).all()

    def test_sum_all(self):
        a = Tensor([1, 2, 3, 4])
        s = a.sum()
        assert s.item() == pytest.approx(10.0)

    def test_sum_axis(self):
        a = Tensor([[1, 2, 3], [4, 5, 6]])
        s = a.reshape(2, 3).sum(axis=1)
        np.testing.assert_allclose(s.numpy(), [6, 15])

    def test_sum_explicit_accumulation_dtype_matches_pinned(self):
        x = Tensor(np.asarray([[1.25, -2.0, 0.5], [3.0, 0.25, -1.5]], dtype=np.float16))
        default = x.sum(axis=1)
        explicit = x.sum(axis=1, dtype=dtypes.float32)
        assert default.dtype is dtypes.float16
        assert explicit.dtype is dtypes.float32
        np.testing.assert_array_equal(default.numpy(), [-0.25, 1.75])
        np.testing.assert_array_equal(explicit.numpy(), [-0.25, 1.75])

    def test_var_matches_pinned_expression_without_substitution(self):
        assert not hasattr(Tensor, '_physicalize_result_for')
        x = Tensor([[1.0, 2.0, 4.0], [3.0, 5.0, 9.0]])

        def pinned_expression(axis, keepdim=False, correction=1):
            squares = (x - x.mean(axis=axis, keepdim=True)).square()
            reduced_shape = squares.sum(axis=axis, keepdim=True).shape
            n = np.prod([
                si for si, so in zip(x.shape, reduced_shape) if si != so
            ], dtype=np.int64).item()
            reduced = squares.sum(axis=axis, keepdim=keepdim)
            return reduced.div((reduced.const_like(n) - correction).relu())

        for axis, keepdim, correction in [
            (1, False, 1),
            (0, True, 0),
            (None, False, 1),
            ((0, 1), False, 1),
            (1, False, 3),
        ]:
            actual = x.var(axis=axis, keepdim=keepdim, correction=correction)
            expected = pinned_expression(axis, keepdim, correction)
            assert actual.uop.raw == expected.uop.raw
            assert actual.uop_logical.raw == expected.uop_logical.raw
            np.testing.assert_allclose(actual.numpy(), expected.numpy())

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
    def test_linear_vector_and_matrix_match_pinned_semantics(self):
        x = Tensor(np.arange(16, dtype=np.float32).reshape(2, 2, 4) / 7)
        vector = x.linear(
            Tensor([1.0, 2.0, 3.0, 4.0]),
            Tensor([0.5, 1.0, 1.5, 2.0]),
        )
        matrix_weight = Tensor(np.arange(12, dtype=np.float32).reshape(4, 3) / 11)
        matrix = x.linear(matrix_weight, Tensor([0.25, -0.5, 0.75]))

        # Pinned mixin/__init__.py:1335-1350: vector weights multiply;
        # matrix weights are already (in,out) and pass directly to dot.
        assert vector.shape == (2, 2, 4)
        assert matrix.shape == (2, 2, 3)
        np.testing.assert_allclose(
            vector.numpy(),
            np.arange(16, dtype=np.float32).reshape(2, 2, 4) / 7 *
            np.array([1, 2, 3, 4], dtype=np.float32) +
            np.array([0.5, 1, 1.5, 2], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            matrix.numpy(),
            np.matmul(
                np.arange(16, dtype=np.float32).reshape(2, 2, 4) / 7,
                np.arange(12, dtype=np.float32).reshape(4, 3) / 11,
            ) + np.array([0.25, -0.5, 0.75], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_attention_primitives_match_pinned_compositions(self):
        x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        repeated = x.repeat_interleave(2, dim=1)
        expected_repeat = x.reshape(2, 2, 1).expand(2, 2, 2).reshape(2, 4)
        assert repeated.uop.raw == expected_repeat.uop.raw
        np.testing.assert_array_equal(repeated.numpy(), [[1, 1, 2, 2], [3, 3, 4, 4]])

        assert x.dropout(0.25) is x
        with pytest.raises(ValueError, match='out of range'):
            x.dropout(1.1)
        Tensor.manual_seed(11)
        with Context(TRAINING=1):
            dropped = x.dropout(0.25)
        Tensor.manual_seed(11)
        with Context(TRAINING=1):
            expected_dropout = (
                (Tensor.rand_like(x, dtype=dtypes.default_float, contiguous=False) >= 0.25)
                .contiguous().where(x, 0) / 0.75
            )

        def op_counts(root):
            seen, stack, counts = set(), [root], {}
            while stack:
                node = stack.pop()
                if node.raw in seen:
                    continue
                seen.add(node.raw)
                counts[node.op_name] = counts.get(node.op_name, 0) + 1
                stack.extend(node.src)
            return counts

        assert op_counts(dropped.uop) == op_counts(expected_dropout.uop)
        np.testing.assert_array_equal(dropped.numpy(), expected_dropout.numpy())

        q = Tensor(np.arange(12, dtype=np.float32).reshape(1, 2, 2, 3) / 13)
        k = Tensor((np.arange(12, dtype=np.float32).reshape(1, 2, 2, 3) - 4) / 11)
        v = Tensor((np.arange(16, dtype=np.float32).reshape(1, 2, 2, 4) + 1) / 17)
        causal = q.scaled_dot_product_attention(k, v, is_causal=True)
        qk = q.matmul(k.transpose(-2, -1), dtype=dtypes.float32) / math.sqrt(3)
        mask = qk.const_like(1).cast(dtypes.bool).tril().where(0, -float('inf'))
        expected_causal = (qk + mask).cast(q.dtype).softmax(-1) @ v
        assert causal.uop.raw == expected_causal.uop.raw
        np.testing.assert_allclose(causal.numpy(), expected_causal.numpy(), rtol=1e-6, atol=1e-6)

        qg = Tensor(np.arange(24, dtype=np.float32).reshape(1, 4, 2, 3) / 19)
        gqa = qg.scaled_dot_product_attention(k, v, enable_gqa=True)
        repeated_k = k.repeat_interleave(2, dim=-3)
        repeated_v = v.repeat_interleave(2, dim=-3)
        expected_gqa = (
            qg.matmul(repeated_k.transpose(-2, -1), dtype=dtypes.float32) /
            math.sqrt(3)
        ).cast(qg.dtype).softmax(-1) @ repeated_v
        assert gqa.shape == (1, 4, 2, 4)
        assert gqa.uop.raw == expected_gqa.uop.raw
        np.testing.assert_allclose(gqa.numpy(), expected_gqa.numpy(), rtol=1e-6, atol=1e-6)

    def test_permute_keyword_negative_validation_and_identity(self):
        x = Tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4))
        keyword = x.permute(order=(0, 2, 1))
        negative = x.permute(0, -1, 1)
        assert keyword.uop.raw == negative.uop.raw
        assert keyword.shape == (2, 4, 3)
        assert x.permute(0, 1, 2) is x
        with pytest.raises(RuntimeError, match='not a valid permutation'):
            x.permute(0, 0, 1)

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

    def test_conv2d_dtype_promotion_and_accumulation_match_pinned(self):
        x = Tensor(np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3) / 7).cast(dtypes.float16).realize()
        weights = np.asarray([[[[0.25, -0.5], [0.75, 0.125]]]], dtype=np.float32)
        w_half = Tensor(weights.astype(np.float16)).realize()
        w_float = Tensor(weights).realize()
        b_half = Tensor(np.asarray([0.0625], dtype=np.float16)).realize()
        b_float = Tensor(np.asarray([0.0625], dtype=np.float32)).realize()

        mixed = x.conv2d(w_float, b_float)
        half_default = x.conv2d(w_half, b_half)
        half_explicit = x.conv2d(w_half, b_half, dtype=dtypes.float32)
        half_float_bias = x.conv2d(w_half, b_float)
        assert [mixed.dtype, half_default.dtype, half_explicit.dtype, half_float_bias.dtype] == [
            dtypes.float32, dtypes.float16, dtypes.float32, dtypes.float32,
        ]
        expected_float = [[[[0.38385009765625, 0.47314453125],
                            [0.65167236328125, 0.740966796875]]]]
        expected_half = [[[[0.3837890625, 0.47314453125],
                           [0.65185546875, 0.7412109375]]]]
        np.testing.assert_array_equal(mixed.numpy(), expected_float)
        np.testing.assert_array_equal(half_default.numpy(), expected_half)
        np.testing.assert_array_equal(half_explicit.numpy(), expected_float)
        # Tinygrad 2026-08-22/a9069c177a9d coalesce.py:95-101 removes the
        # float-half-float roundtrip after float-bias promotion at codegen.
        np.testing.assert_array_equal(half_float_bias.numpy(), expected_float)

        for case_index, (result, expected_casts) in enumerate((
            (mixed, 1), (half_default, 2), (half_explicit, 2), (half_float_bias, 3),
        )):
            topo, seen = [], set()
            def visit(uop):
                if uop.raw in seen:
                    return
                seen.add(uop.raw)
                for child in uop.src:
                    visit(child)
                topo.append(uop)
            visit(result.uop)
            assert sum(u.op_name == 'CAST' for u in topo) == expected_casts
            assert sum(u.op_name == 'MUL' for u in topo) == 1
            assert sum(u.op_name == 'REDUCE' for u in topo) == 1
            assert sum(u.op_name == 'ADD' for u in topo) == 1
            assert all(u.dtype == dtypes.float32 for u in topo if u.op_name == 'REDUCE')
            assert all(
                u.dtype == (dtypes.float32 if case_index == 0 else dtypes.float16)
                for u in topo if u.op_name == 'MUL'
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
        assert not hasattr(Tensor, '_physicalize_result_for')
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([[5.0, 6.0], [7.0, 8.0]])
        logical_inputs = (_ffi._ptr * 2)(
            a.uop_logical.raw, b.uop_logical.raw
        )
        physical_inputs = (_ffi._ptr * 2)(a.uop.raw, b.uop.raw)
        expected_logical = _ffi._lib.poly_einsum(
            a._ctx, b'ij,jk->ik', logical_inputs, 2
        )
        expected_physical = _ffi._lib.poly_einsum(
            a._ctx, b'ij,jk->ik', physical_inputs, 2
        )
        out = Tensor.einsum('ij,jk->ik', a, b)
        assert out.shape == (2, 2)
        assert out.uop_logical.raw == expected_logical
        assert out.uop.raw == expected_physical
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

    def test_rearrange_uses_core_tensor_roots(self):
        from polygrad.tensor import _uop_wrap

        assert not hasattr(Tensor, '_physicalize_result_for')
        source = Tensor.arange(6).reshape(2, 3)
        out = source.rearrange('h w -> w h')
        expected_logical = _uop_wrap(
            source._ctx,
            _ffi._lib.poly_rearrange(
                source._ctx, b'h w -> w h', source.uop_logical.raw,
                None, None, 0,
            ),
        )
        expected_physical = _uop_wrap(
            source._ctx,
            _ffi._lib.poly_rearrange(
                source._ctx, b'h w -> w h', source.uop.raw,
                None, None, 0,
            ),
        )
        assert out.uop_logical.raw == expected_logical.raw
        assert out.uop.raw == expected_physical.raw
        np.testing.assert_array_equal(out.numpy(), [[0, 3], [1, 4], [2, 5]])

        moved = Tensor.empty((6,), device='cpu').realize().to('cuda').to('cpu')
        moved = moved.reshape(2, 3)
        moved_out = moved.rearrange('h w -> w h')
        expected_moved = _uop_wrap(
            moved._ctx,
            _ffi._lib.poly_rearrange(
                moved._ctx, b'h w -> w h', moved.uop.raw,
                None, None, 0,
            ),
        )
        assert moved_out.uop.raw == expected_moved.raw

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

    def test_dot_explicit_accumulation_dtype_and_scalar_shape_match_pinned(self):
        a = Tensor(np.asarray([[1.25, -2.0, 0.5], [3.0, 0.25, -1.5]], dtype=np.float16))
        b = Tensor(np.asarray([[0.5, -1.0], [2.0, 0.25], [-0.75, 3.0]], dtype=np.float16))
        default = a.dot(b)
        explicit = a.dot(b, dtype=dtypes.float32)
        assert default.dtype is dtypes.float16
        assert explicit.dtype is dtypes.float32
        np.testing.assert_array_equal(default.numpy(), [[-3.75, -0.25], [3.125, -7.4375]])
        np.testing.assert_array_equal(explicit.numpy(), [[-3.75, -0.25], [3.125, -7.4375]])
        assert Tensor([1.0, 2.0, 3.0]).dot(Tensor([4.0, 5.0, 6.0])).shape == ()

        mixed = a.dot(Tensor(np.asarray(
            [[0.5, -1.0], [2.0, 0.25], [-0.75, 3.0]], dtype=np.float32,
        )))
        assert mixed.dtype is dtypes.float32
        assert mixed.uop.op_name == 'REDUCE'
        assert mixed.uop.src[0].op_name == 'PERMUTE'
        mixed_mul = mixed.uop.src[0].src[0]
        assert mixed_mul.op_name == 'MUL'
        assert mixed_mul.dtype == dtypes.float32
        assert mixed_mul.src[0].op_name == 'CAST'
        np.testing.assert_allclose(
            mixed.numpy(), [[-3.75, -0.25], [3.125, -7.4375]], rtol=1e-6, atol=1e-6,
        )

    def test_matmul_broadcast_mismatch_raises(self):
        a = Tensor(np.zeros((2, 3, 4), dtype=np.float32))
        b = Tensor(np.zeros((5, 4, 6), dtype=np.float32))
        with pytest.raises(ValueError, match='cannot dot'):
            a @ b

    def test_linalg_construction_bypasses_frontend_substitution(self):
        assert not hasattr(Tensor, '_physicalize_result_for')
        a = Tensor([[4.0, 2.0], [2.0, 5.0]])
        b = Tensor([1.0, 3.0])
        lower = Tensor([[2.0, 0.0], [1.0, 3.0]])
        q, r = a.qr()
        assert q.shape == r.shape == (2, 2)
        assert lower.triangular_solve(b).shape == (2,)
        chol = a.cholesky()
        assert chol.shape == (2, 2)
        assert chol.cholesky_solve(b).shape == (2,)
        assert a.solve(b).shape == (2,)
        assert Tensor([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]).lstsq(
            Tensor([1.0, 2.0, 3.0])
        ).shape == (2,)

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
        assert qi.dtype is dtypes.float32
        assert ri.dtype is dtypes.float32
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
            expected = (
                np.stack([np.linalg.solve(batch, b) for batch in np_a])
                if np_a.ndim > 2 and b.ndim == 1
                else np.linalg.solve(np_a, b)
            )

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
        assert x_int.dtype is dtypes.float32
        np.testing.assert_allclose(x_int.numpy(), np.linalg.solve(a_int.astype(np.float32), b_int), rtol=1e-6)

        a64 = np.array([[2.0, 0.0], [1.0, 4.0]], dtype=np.float64)
        b64 = np.array([[2.0], [9.0]], dtype=np.float64)
        x64 = Tensor(a64, dtype='float64').triangular_solve(Tensor(b64, dtype='float64'))
        assert x64.dtype is dtypes.float64
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
        assert l_int.dtype is dtypes.float32
        np.testing.assert_allclose(l_int.numpy(), np.linalg.cholesky(a_int.astype(np.float32)), rtol=1e-5)

        a64 = np.array([[4.0, 2.0], [2.0, 5.0]], dtype=np.float64)
        l64 = Tensor(a64, dtype='float64').cholesky()
        assert l64.dtype is dtypes.float64
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
        assert got64.dtype is dtypes.float64
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
        assert got64.dtype is dtypes.float64
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
        target = Tensor([0, 2], dtype='int32')
        loss = logits.cross_entropy(target)
        assert loss.shape == ()
        np.testing.assert_allclose(loss.numpy(), np.log(3.0), rtol=1e-6)

    def test_cross_entropy_dense_targets(self):
        logits = Tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        target = Tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        loss = logits.cross_entropy(target)
        assert loss.shape == ()
        np.testing.assert_allclose(loss.numpy(), np.log(3.0), rtol=1e-6)

    def test_cross_entropy_matches_pinned_expression_without_substitution(self):
        assert not hasattr(Tensor, '_physicalize_result_for')
        logits = Tensor([[-1.0, 2.0, -3.0], [1.0, -2.0, 3.0]])
        sparse = Tensor([1, 2], dtype='int32')
        dense = Tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        def pinned_expression(target, reduction='mean', label_smoothing=0.0):
            classes_dim = 1
            if logits.shape != target.shape:
                target = target.unsqueeze(classes_dim)._one_hot_along_dim(
                    logits.shape[classes_dim], classes_dim
                )
            target = (
                (1 - label_smoothing) * target
                + label_smoothing / int(target.shape[classes_dim])
            )
            reduced = logits.log_softmax(classes_dim).mul(target).sum(classes_dim)
            if reduction == 'none':
                return -reduced
            if reduction == 'sum':
                return -reduced.sum()
            if reduction == 'mean':
                return -reduced.mean()
            raise ValueError(reduction)

        pairs = []
        for target, reduction, label_smoothing in [
            (sparse, 'mean', 0.0),
            (dense, 'mean', 0.0),
            (dense, 'none', 0.0),
            (dense, 'sum', 0.0),
            (dense, 'mean', 0.2),
        ]:
            actual = logits.cross_entropy(
                target, reduction=reduction, label_smoothing=label_smoothing
            )
            expected = pinned_expression(target, reduction, label_smoothing)
            assert actual.uop.raw == expected.uop.raw
            assert actual.uop_logical.raw == expected.uop_logical.raw
            pairs.append((actual, expected))
        for actual, expected in pairs:
            np.testing.assert_allclose(actual.numpy(), expected.numpy())

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
        target = Tensor(np.array([[0, 2], [1, 0]], dtype=np.int32))
        loss = logits.cross_entropy(target, axis=-2)
        assert loss.shape == ()
        np.testing.assert_allclose(loss.numpy(), np.log(3.0), rtol=1e-6)

    def test_cross_entropy_default_matches_tinygrad_class_axis(self):
        logits = Tensor(np.zeros((2, 3, 2), dtype=np.float32))
        target = Tensor(np.array([[0, 2], [1, 0]], dtype=np.int32))
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
        with pytest.raises(RuntimeError, match='shape mismatch'):
            logits.cross_entropy(target)


class TestAutograd:
    def test_grad_mul_sum(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0])
        loss = (x * x).sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.uop_physical is not None
        assert x.grad.uop_physical.op_name == 'ADD'
        assert x.grad.uop == x.grad.uop_physical
        np.testing.assert_allclose(x.grad.numpy(), [2, 4, 6, 8])

    def test_grad_neg_sum(self):
        x = Tensor([1.0, 2.0, 3.0])
        loss = (-x).sum()
        loss.backward()
        np.testing.assert_allclose(x.grad.numpy(), [-1, -1, -1])

    def test_grad_max_reduce(self):
        x = Tensor([[1.0, 3.0, 2.0], [4.0, 5.0, 6.0]])
        loss = x.max(axis=1).sum()
        loss.backward()
        assert x.grad is not None
        np.testing.assert_allclose(x.grad.numpy(), [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    def test_gradient_is_targeted_and_does_not_mutate_grad_fields(self):
        a = Tensor([2.0, 3.0]).is_param_(False)
        b = Tensor([4.0, 5.0])
        intermediate = a * b
        loss = intermediate.sum()

        (grad_a,) = loss.gradient(a)
        np.testing.assert_array_equal(grad_a.numpy(), [4.0, 5.0])
        assert a.grad is None
        assert b.grad is None
        assert intermediate.grad is None
        assert loss.grad is None

        loss.backward()
        np.testing.assert_array_equal(a.grad.numpy(), [4.0, 5.0])
        np.testing.assert_array_equal(b.grad.numpy(), [2.0, 3.0])
        np.testing.assert_array_equal(intermediate.grad.numpy(), [1.0, 1.0])
        assert loss.grad.item() == 1.0

    def test_gradient_of_unreachable_target_is_zero(self):
        loss = Tensor([1.0, 2.0]).sum()
        unreachable = Tensor([3.0, 4.0])
        np.testing.assert_array_equal(
            loss.gradient(unreachable)[0].numpy(), [0.0, 0.0]
        )

    def test_backward_type_shape_and_explicit_gradient_admission(self):
        with pytest.raises(RuntimeError, match='only float Tensors have gradient'):
            Tensor([1, 2, 3]).sum().backward()
        with pytest.raises(
            AssertionError,
            match='when no gradient is provided, backward must be called on a scalar tensor',
        ):
            Tensor([1.0, 2.0]).backward()
        vector = Tensor([1.0, 2.0])
        vector.backward(Tensor([3.0, 4.0]))
        np.testing.assert_array_equal(vector.grad.numpy(), [3.0, 4.0])


class TestDevice:
    def test_device_lookup(self):
        assert Device['cpu'].device == 'CPU'
        assert Device['CUDA'].device == 'CUDA'
        assert Device['CUDA:0'].device == 'CUDA'
        assert Device['interp'].device == 'INTERP'
        assert Device['cpu:x86'].device == 'X86'
        assert Device['hip'].device == 'HIP'
        assert Device['wasm'].device == 'WASM'
        assert Device['webgpu'].device == 'WEBGPU'
        assert Device['host'].device == 'HOST'
        with pytest.raises(ValueError, match='Unsupported device'):
            Device['not-a-device']

        opened = Device[Device.DEFAULT]
        supported = opened.renderer.supported_dtypes()
        assert opened.device == Device.DEFAULT
        assert dtypes.float16 in supported
        assert dtypes.float32 in supported
        assert dtypes.weakint not in supported

    def test_default_device_tracks_dev_context(self):
        original = Device.DEFAULT
        with Context(DEV='INTERP'):
            assert Device.DEFAULT == 'INTERP'
            assert Device.canonicalize(None) == 'INTERP'
            assert Tensor.empty(1).device == 'INTERP'
        assert Device.DEFAULT == original

    def test_requires_grad_api_is_removed(self):
        with pytest.raises(TypeError, match="unexpected keyword argument 'requires_grad'"):
            Tensor([1.0], requires_grad=True)
        a = Tensor([1.0])
        assert not hasattr(a, 'requires_grad')
        assert not hasattr(a, 'requires_grad_')

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

    def test_assign_broadcasts_rhs_in_core(self):
        target = Tensor.zeros(2, 3)
        target.assign(Tensor([4.0, 5.0, 6.0])).realize()
        np.testing.assert_allclose(target.numpy(), [[4.0, 5.0, 6.0]] * 2)

    def test_assign_realized_contiguous_cache_view_retargets_both_roots(self):
        cache = Tensor.zeros(2, 1, 8, 1, 4).contiguous().preserve_logical().realize()
        logical_value = cache.uop_logical
        physical_identity = cache.uop_physical
        assert logical_value.op_name == 'AFTER'
        assert physical_identity.has_buffer_identity()

        xk = Tensor.arange(12).float().reshape(1, 3, 1, 4)
        xv = (Tensor.arange(12).float() + 100).reshape(1, 3, 1, 4)
        view = cache[:, :, :3, :, :]
        view.assign(Tensor.stack(xk, xv))

        # Pinned tensor.py:246-252 retargets the nearest current buffer
        # identity. Polygrad applies the same physical rewrite while retaining
        # the independent device-free logical effect graph.
        assert cache.uop_logical.op_name == 'AFTER'
        assert cache.uop_logical.src[0] == logical_value
        assert cache.uop_physical.op_name == 'AFTER'
        assert cache.uop_physical.src[0] == physical_identity
        assert view.uop_logical.op_name == 'SHRINK'
        assert view.uop_logical.src[0] == cache.uop_logical
        assert view.uop_physical.op_name == 'SHRINK'
        assert view.uop_physical.src[0] == cache.uop_physical

        view.realize()
        expected = np.zeros((2, 1, 8, 1, 4), dtype=np.float32)
        expected[0, 0, :3, 0, :] = np.arange(12, dtype=np.float32).reshape(3, 4)
        expected[1, 0, :3, 0, :] = 100 + np.arange(12, dtype=np.float32).reshape(3, 4)
        np.testing.assert_array_equal(cache.numpy(), expected)

    def test_assign_rejects_dtype_mismatch_like_tinygrad(self):
        a = Tensor([1.0], dtype='float32')
        v = Tensor([5.0], dtype='float64')
        with pytest.raises(RuntimeError, match=r'assign dtype mismatch dtypes\.float != dtypes\.double'):
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
        assert 'dtypes.int' in repr(t)

    def test_bool_raises_like_tinygrad(self):
        with pytest.raises(TypeError, match="__bool__ on Tensor is not defined"):
            bool(Tensor([1.0]))

    def test_repr_f64(self):
        t = Tensor([1, 2, 3], dtype='float64')
        assert 'dtypes.double' in repr(t)


class TestFloat64:
    """Tests for float64 dtype support."""

    def test_numpy_dtype_is_preserved_like_tinygrad(self):
        f64 = Tensor(np.array([1.0, 2.0], dtype=np.float64))
        assert f64.dtype is dtypes.float64
        assert f64.numpy().dtype == np.float64

        i64 = Tensor(np.array([1, 2], dtype=np.int64))
        assert i64.dtype is dtypes.int64
        assert i64.numpy().dtype == np.int64

        default_list = Tensor([1.0, 2.0])
        assert default_list.dtype is dtypes.float32

    def test_creation_from_list(self):
        t = Tensor([1.0, 2.0, 3.0], dtype='float64')
        assert t.dtype is dtypes.float64
        assert t.shape == (3,)
        assert t.numpy().dtype == np.float64
        np.testing.assert_allclose(t.numpy(), [1, 2, 3])

    def test_creation_2d(self):
        t = Tensor([[1, 2], [3, 4]], dtype='float64')
        assert t.dtype is dtypes.float64
        assert t.shape == (2, 2)
        assert t.numpy().dtype == np.float64
        np.testing.assert_allclose(t.numpy(), [[1, 2], [3, 4]])

    def test_zeros_f64(self):
        t = Tensor.zeros(4, dtype='float64')
        assert t.dtype is dtypes.float64
        assert t.numpy().dtype == np.float64
        np.testing.assert_allclose(t.numpy(), [0, 0, 0, 0])

    def test_ones_f64(self):
        t = Tensor.ones(3, dtype='float64')
        assert t.dtype is dtypes.float64
        np.testing.assert_allclose(t.numpy(), [1, 1, 1])

    def test_full_f64(self):
        t = Tensor.full((2, 3), 7.0, dtype='float64')
        assert t.dtype is dtypes.float64
        np.testing.assert_allclose(t.numpy(), np.full((2, 3), 7.0))

    def test_eye_f64(self):
        t = Tensor.eye(3, dtype='float64')
        assert t.dtype is dtypes.float64
        np.testing.assert_allclose(t.numpy(), np.eye(3))

    def test_arange_f64(self):
        t = Tensor.arange(5, dtype='float64')
        assert t.dtype is dtypes.float64
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
        a = Tensor([1.0, 2.0, 3.0], dtype='float64')
        b = Tensor([4.0, 5.0, 6.0], dtype='float64')
        loss = (a * b).sum()
        loss.backward()
        assert a.grad is not None
        np.testing.assert_allclose(a.grad.numpy(), [4, 5, 6], rtol=1e-14)

    def test_dtype_propagation(self):
        """Ensure dtype propagates through ops."""
        a = Tensor([1.0, 2.0], dtype='float64')
        b = a + 1.0
        assert b.dtype is dtypes.float64
        c = b * 2.0
        assert c.dtype is dtypes.float64
        d = c.exp()
        assert d.dtype is dtypes.float64

    def test_default_is_f32(self):
        """Ensure default dtype is still float32."""
        a = Tensor([1.0, 2.0])
        assert a.dtype is dtypes.float32
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

    def test_movement_bounds_reject_shrink_past_end(self):
        x = Tensor([1., 2., 3., 4.])
        for bounds in [(2, 5), (0, 5), (5, 5), (-1, 3), (2, 1)]:
            with pytest.raises(ValueError, match='invalid shrink'):
                x.shrink((bounds,))
        np.testing.assert_array_equal(x.shrink(((1, 3),)).numpy(), [2., 3.])
        assert x.shrink(((4, 4),)).shape == (0,)
        np.testing.assert_array_equal(x.pad(((2, 1),)).numpy(), [0., 0., 1., 2., 3., 4., 0.])

    def test_minimum_bool_xor_differs_from_unary_min(self):
        x = Tensor([False, False, True, True], dtype='bool')
        y = Tensor([False, True, False, True], dtype='bool')
        out = x.minimum(y)
        for root in [out.uop_logical, out.uop_physical]:
            assert root.op_name == 'XOR'
            maximum = root.src[0]
            assert maximum.op_name == 'MAX'
            assert [s.op_name for s in maximum.src] == ['XOR', 'XOR']
        np.testing.assert_array_equal(out.numpy(), [False, False, False, True])
        assert x.min().uop_physical.op_name == 'CMPNE'
        np.testing.assert_array_equal(x.minimum(True).numpy(), x.numpy())

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

        clipped = Tensor([-3.0, -0.5, 2.0]).clip(-1.0, 1.0)
        assert clipped.uop.op_name == 'WHERE'
        assert [src.op_name for src in clipped.uop.src] == ['CMPLT', 'CONST', 'WHERE']
        np.testing.assert_array_equal(clipped.numpy(), [-1.0, -0.5, 1.0])

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
        expected_integer = np.array([0, 1, 2], dtype=np.float32)
        for actual, expected in (
            (integer.sin(), np.sin(expected_integer)),
            (integer.cos(), np.cos(expected_integer)),
            (integer.tan(), np.tan(expected_integer)),
        ):
            assert actual.dtype is dtypes.float32
            np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-6, atol=1e-6)
        assert integer.sin().uop.src[0].op_name != 'CAST'
        np.testing.assert_allclose(
            Tensor([False, True], dtype='bool').sin().numpy(),
            np.sin(np.array([0, 1], dtype=np.float32)),
            rtol=1e-6,
            atol=1e-6,
        )

        half = Tensor([0.0, 1.0], dtype='float16')
        assert half.cos().dtype is dtypes.float16
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
            assert angles.cos().dtype is getattr(dtypes, expected_dtype)
            assert angles.tan().dtype is getattr(dtypes, expected_dtype)
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
        assert half.sigmoid().dtype is dtypes.float16
        assert half.tanh().dtype is dtypes.float16
        assert half.gelu().dtype is dtypes.float16
        assert half.quick_gelu().dtype is dtypes.float16
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
        assert ints.gelu().dtype is dtypes.weakfloat
        assert ints.quick_gelu().dtype is dtypes.weakfloat
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
        assert boolean.gelu().dtype is dtypes.weakfloat
        assert boolean.quick_gelu().dtype is dtypes.weakfloat
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
            grad_input = Tensor(values)
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
            x = Tensor(inputs, dtype='float16')
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
            assert rounded.dtype is getattr(dtypes, expected_dtype)
            np.testing.assert_array_equal(rounded.numpy(), expected_round)

        rounded_int = Tensor([-2, -1, 0, 1, 2], dtype='int32').round()
        assert rounded_int.dtype is dtypes.weakfloat
        np.testing.assert_array_equal(rounded_int.numpy(), [-2, -1, 0, 1, 2])

        rounded_bool = Tensor([False, True], dtype='bool').round()
        assert rounded_bool.dtype is dtypes.weakfloat
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
            assert result.dtype is dtypes.bool
            np.testing.assert_array_equal(result.numpy(), expected)

        for dtype in ('float16', 'float32', 'float64', 'int32', 'bool'):
            result = Tensor([0, 1], dtype=dtype).isinf()
            assert result.dtype is dtypes.bool
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
