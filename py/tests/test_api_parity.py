import numpy as np
import pytest

from extra.lr_scheduler import OneCycleLR
from polygrad import _ffi
from polygrad import GlobalCounters, Jit, Tensor, TinyJit, Variable, dtypes, nn
from polygrad.nn import SGD


def test_top_level_tinyjit_alias_uses_existing_jit():
    assert TinyJit is Jit

    @TinyJit
    def add_one(x):
        return (x + 1).realize()

    x = Tensor([1.0]).realize()
    assert add_one(x).item() == 2.0
    x.copy_from([2.0])
    assert add_one(x).item() == 3.0
    x.copy_from([3.0])
    assert add_one(x).item() == 4.0
    assert add_one.cnt == 3


def test_tinyjit_rejects_nested_capture_and_recovers_context():
    x = Tensor([1.0, 2.0, 3.0]).realize()
    recurse = False

    @TinyJit
    def nested(value):
        nonlocal recurse
        if recurse:
            recurse = False
            return nested(value)
        return (value + 1).realize()

    nested(x)
    recurse = True
    with pytest.raises(RuntimeError, match='having TinyJit inside another TinyJit'):
        nested(x)

    @TinyJit
    def other(value):
        return (value * 2).realize()

    other(x)
    np.testing.assert_array_equal(other(x).numpy(), [2.0, 4.0, 6.0])


def test_top_level_nn_and_global_counters_match_tinygrad_surface():
    assert nn.SGD is SGD
    x = Tensor.arange(8, dtype='float32').realize(do_update_stats=False)
    before_mem = GlobalCounters.mem_used
    GlobalCounters.reset()
    assert GlobalCounters.global_ops == 0
    assert GlobalCounters.global_mem == 0
    assert GlobalCounters.time_sum_s == 0.0
    assert GlobalCounters.kernel_count == 0
    assert GlobalCounters.mem_used == before_mem

    (x + 1).realize()
    assert GlobalCounters.global_ops == 8
    assert GlobalCounters.global_mem == 64
    assert GlobalCounters.kernel_count == 1
    assert GlobalCounters.time_sum_s >= 0.0
    assert sum(GlobalCounters.mem_used_per_device.values()) == GlobalCounters.mem_used


def test_global_counters_count_tinyjit_capture_and_replay_once():
    x = Tensor.arange(8, dtype='float32').realize(do_update_stats=False)

    @TinyJit
    def add_one(value):
        return (value + 1).realize()

    add_one(x)  # warmup
    GlobalCounters.reset()
    add_one(x)  # capture execution
    assert (GlobalCounters.global_ops, GlobalCounters.global_mem, GlobalCounters.kernel_count) == (
        8, 64, 1,
    )

    GlobalCounters.reset()
    add_one(x)  # compiled replay
    assert (GlobalCounters.global_ops, GlobalCounters.global_mem, GlobalCounters.kernel_count) == (
        8, 64, 1,
    )


def test_tinyjit_symbolic_input_view_replays_current_binding():
    # tinygrad engine/jit.py:_prepare_jit_inputs unbinds the complete input
    # view, not only its shape. HLB fetch_batches changes this slice offset.
    source = Tensor.arange(12, dtype=dtypes.float32).reshape(6, 2).clone().realize()
    batch_index = Variable('i', 0, 4)

    @TinyJit
    def add_hundred(batch):
        return (batch + 100).contiguous()

    values = []
    for start in (0, 2, 4, 2):
        batch = source[batch_index.bind(start):batch_index.bind(start)+2]
        seen, stack, bind_count = set(), [batch.uop_logical], 0
        while stack:
            uop = stack.pop()
            if not uop or uop in seen:
                continue
            seen.add(uop)
            bind_count += uop.op_name == 'BIND'
            stack.extend(uop.src)
        assert batch.uop_logical.op_name == 'SHRINK'
        assert bind_count == 1
        values.append(add_hundred(batch).numpy().copy())

    np.testing.assert_array_equal(values[0], [[100, 101], [102, 103]])
    np.testing.assert_array_equal(values[1], [[104, 105], [106, 107]])
    np.testing.assert_array_equal(values[2], [[108, 109], [110, 111]])
    np.testing.assert_array_equal(values[3], values[1])
    assert add_hundred.cnt == 4
    assert add_hundred.captured


def test_tinyjit_movement_view_replays_against_new_base():
    # tinygrad engine/jit.py:_prepare_jit_inputs keeps the movement UOp in the
    # input signature/function graph while parameterizing its recursive base.
    @TinyJit
    def add_one(value):
        return (value + 1).realize()

    first = Tensor([1, 2, 3, 4], dtype=dtypes.float32).shrink(((1, 3),))
    np.testing.assert_array_equal(add_one(first).numpy(), [3, 4])
    assert first.uop_physical.op_name == 'SHRINK'
    np.testing.assert_array_equal(add_one(first).numpy(), [3, 4])
    assert first.uop_physical.op_name == 'SHRINK'

    second = Tensor([10, 20, 30, 40], dtype=dtypes.float32).shrink(((1, 3),))
    np.testing.assert_array_equal(add_one(second).numpy(), [21, 31])


def test_bound_view_executes_and_counts_runtime_extent():
    x = Tensor.arange(8, dtype='float32').realize(do_update_stats=False)
    n = Variable('n', 1, 8)
    out = x[:n.bind(4)] + 1

    GlobalCounters.reset()
    out.realize()
    assert (GlobalCounters.global_ops, GlobalCounters.global_mem, GlobalCounters.kernel_count) == (
        4, 32, 1,
    )


def test_pure_view_realize_is_zero_call():
    x = Tensor.arange(8, dtype='float32').realize(do_update_stats=False)
    out = x.reshape(2, 4).permute(1, 0)

    GlobalCounters.reset()
    out.realize()
    assert (GlobalCounters.global_ops, GlobalCounters.global_mem, GlobalCounters.kernel_count) == (
        0, 0, 0,
    )
    np.testing.assert_array_equal(
        out.numpy(),
        np.array([[0, 4], [1, 5], [2, 6], [3, 7]], dtype=np.float32),
    )


def test_realized_contiguous_and_readback_reuse_current_buffer_identity():
    source = (Tensor.arange(8, dtype='float32') + 1).realize()
    source_current = source.uop.raw
    source_logical = source.uop_logical.raw
    assert source.uop_logical.op_name == 'ADD'
    assert source.uop.has_buffer_identity()

    out = source.contiguous()
    assert out is not source
    assert out.uop_logical.op_name == 'CONTIGUOUS'
    assert out.uop_physical is not None
    assert out.uop.raw == source_current
    assert out.uop_physical.raw == source_current
    assert source.uop_logical.raw == source_logical

    GlobalCounters.reset()
    out.realize()
    assert (GlobalCounters.global_ops, GlobalCounters.global_mem, GlobalCounters.kernel_count) == (
        0, 0, 0,
    )
    GlobalCounters.reset()
    np.testing.assert_array_equal(out.numpy(), np.arange(1, 9, dtype=np.float32))
    assert (GlobalCounters.global_ops, GlobalCounters.global_mem, GlobalCounters.kernel_count) == (
        0, 0, 0,
    )

    reshaped = source.reshape(2, 4)
    assert reshaped.uop.has_buffer_identity()
    GlobalCounters.reset()
    np.testing.assert_array_equal(
        reshaped.numpy(), np.arange(1, 9, dtype=np.float32).reshape(2, 4),
    )
    assert (GlobalCounters.global_ops, GlobalCounters.global_mem, GlobalCounters.kernel_count) == (
        0, 0, 0,
    )

    sliced = source[2:6].realize()
    assert sliced.uop.has_buffer_identity()
    GlobalCounters.reset()
    np.testing.assert_array_equal(sliced.numpy(), np.arange(3, 7, dtype=np.float32))
    assert (GlobalCounters.global_ops, GlobalCounters.global_mem, GlobalCounters.kernel_count) == (
        0, 0, 0,
    )

    permuted = source.reshape(2, 4).permute(1, 0)
    GlobalCounters.reset()
    np.testing.assert_array_equal(
        permuted.numpy(),
        np.array([[1, 5], [2, 6], [3, 7], [4, 8]], dtype=np.float32),
    )
    assert GlobalCounters.kernel_count == 1

    casted = source.cast('int32')
    GlobalCounters.reset()
    np.testing.assert_array_equal(casted.numpy(), np.arange(1, 9, dtype=np.int32))
    assert GlobalCounters.kernel_count == 1


def test_random_counter_and_tinyjit_replay_match_pinned_tinygrad():
    expected = np.array([
        [0.8125981092453003, 0.6181309223175049, 0.9865618944168091,
         0.998869776725769, 0.8691444396972656, 0.20871984958648682,
         0.7909691333770752, 0.2107449769973755],
        [0.49933385848999023, 0.7912073135375977, 0.8326060771942139,
         0.27198243141174316, 0.6909018754959106, 0.5686440467834473,
         0.09911370277404785, 0.7808284759521484],
        [0.8395031690597534, 0.14842653274536133, 0.9003503322601318,
         0.525137186050415, 0.6625624895095825, 0.6041145324707031,
         0.9730466604232788, 0.7700762748718262],
        [0.5011155605316162, 0.1397721767425537, 0.24960505962371826,
         0.023981213569641113, 0.4844532012939453, 0.8480784893035889,
         0.19490158557891846, 0.5924421548843384],
    ], dtype=np.float32)

    try:
        Tensor.manual_seed(201)
        first, second = Tensor.rand(8), Tensor.rand(8)
        np.testing.assert_array_equal(first.cat(second).numpy(), expected[:2].reshape(-1))
        counter = next(iter(Tensor._device_rng_counters.values()))
        np.testing.assert_array_equal(counter.numpy(), np.array([16, 0], dtype=np.uint32))

        Tensor.manual_seed(201)
        jit_input = Tensor.zeros(8).contiguous().realize()

        @TinyJit
        def random_jit(value):
            return (Tensor.rand(*value.shape) + value).contiguous()

        actual = np.stack([random_jit(jit_input).numpy().copy() for _ in range(4)])
        np.testing.assert_array_equal(actual, expected)
        counter = next(iter(Tensor._device_rng_counters.values()))
        np.testing.assert_array_equal(counter.numpy(), np.array([32, 0], dtype=np.uint32))

        seen, stack, ops = set(), [random_jit.ret.uop_logical], set()
        while stack:
            uop = stack.pop()
            if uop in seen:
                continue
            seen.add(uop)
            ops.add(uop.op_name)
            stack.extend(uop.src)
        assert {'THREEFRY', 'STORE', 'AFTER'} <= ops
    finally:
        Tensor.manual_seed(0)


def test_backward_uses_realized_random_parameter_version():
    """Backward must differentiate the values used by the physical forward."""
    try:
        Tensor.manual_seed(42)
        weight = (Tensor.rand(2, 2) * 0.5 - 0.25).realize()
        weight.requires_grad = True
        x = Tensor([[0.2, -0.4], [0.7, 0.3]], dtype='float32')
        target = Tensor([[0.1, -0.2], [0.3, 0.4]], dtype='float32')

        loss = (x.dot(weight) - target).square().sum()
        loss.backward()

        np.testing.assert_allclose(
            weight.grad.numpy(),
            [
                [-0.40808677673339844, -0.39865055680274963],
                [-0.029885588213801384, -0.394866019487381],
            ],
            rtol=1e-6,
            atol=1e-6,
        )
    finally:
        Tensor.manual_seed(0)


def test_random_crop_indices_remain_consistent_after_readback():
    """Pinned HLB builds the crop before its two RNG dependencies may realize."""
    try:
        batch, channels, height, width, crop_size = 4, 3, 6, 6, 4
        source_np = np.arange(
            batch * channels * height * width, dtype=np.float32,
        ).reshape(batch, channels, height, width)
        source = Tensor(source_np).contiguous().realize()
        Tensor.manual_seed(201)
        low_x = Tensor.randint(batch, low=0, high=width-crop_size).reshape(batch, 1, 1, 1)
        low_y = Tensor.randint(batch, low=0, high=height-crop_size).reshape(batch, 1, 1, 1)
        idx_x = Tensor.arange(crop_size, dtype=dtypes.int32).reshape(1, 1, 1, crop_size)
        idx_y = Tensor.arange(crop_size, dtype=dtypes.int32).reshape(1, 1, crop_size, 1)
        crop = source.gather(
            -1, (low_x + idx_x).expand(-1, channels, height, -1),
        ).gather(
            -2, (low_y + idx_y).expand(-1, channels, crop_size, crop_size),
        )

        starts_x = low_x.numpy().reshape(-1)
        starts_y = low_y.numpy().reshape(-1)
        actual = crop.numpy()
        for i, (start_x, start_y) in enumerate(zip(starts_x, starts_y)):
            np.testing.assert_array_equal(
                actual[i],
                source_np[i, :, start_y:start_y+crop_size, start_x:start_x+crop_size],
            )
    finally:
        Tensor.manual_seed(0)


def test_python_loader_checks_current_abi_before_use():
    assert _ffi.get_lib().poly_abi_version() == _ffi.POLYGRAD_ABI_VERSION == 24


def test_python_loader_rejects_mismatched_abi_before_declaring_signatures(monkeypatch):
    class FakeAbiFunction:
        restype = None
        argtypes = None

        def __call__(self):
            return 20

    class FakeLibrary:
        poly_abi_version = FakeAbiFunction()

    monkeypatch.setattr(_ffi, '_lib', None)
    monkeypatch.setattr(_ffi, '_find_lib', lambda: 'fake-libpolygrad.so')
    monkeypatch.setattr(_ffi.ctypes, 'CDLL', lambda _path: FakeLibrary())
    with pytest.raises(RuntimeError, match='expected version 24, got 20'):
        _ffi.get_lib()
    assert _ffi._lib is None


def test_optimizer_lr_is_assignable_tensor_and_feeds_c_graph():
    p = Tensor([1.0], requires_grad=True).realize()
    opt = SGD([p], lr=0.1)

    assert isinstance(opt.lr, Tensor)
    opt.lr.assign(Tensor([0.2], dtype=opt.lr.dtype)).realize()
    p._grad = Tensor([1.0])
    opt.step()

    np.testing.assert_allclose(p.numpy(), [0.8], rtol=1e-6, atol=1e-6)


def test_sgd_momentum_commits_lazy_backward_gradient_before_view_state_assign():
    p = Tensor([1.0], requires_grad=True).realize()
    x = Tensor([1.0]).realize()
    opt = SGD(
        [p], lr=0.02, momentum=0.85, nesterov=True,
        weight_decay=0.0, fused=False,
    )

    loss = (p * x).sum()
    opt.zero_grad()
    loss.backward()
    opt.step()

    np.testing.assert_allclose(opt.b[0].numpy(), [1.0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(p.numpy(), [0.963], rtol=1e-6, atol=1e-6)


def test_sgd_vector_momentum_state_is_writable_and_elementwise():
    p = Tensor([1.0, 2.0], requires_grad=True).realize()
    p._grad = Tensor([0.25, -0.5], dtype=p.dtype)
    opt = SGD(
        [p], lr=0.1, momentum=0.9, nesterov=True,
        weight_decay=0.1, fused=False,
    )

    with Tensor.train(True):
        scheduled = opt.schedule_step()
        assert len(scheduled) == 2
        assert scheduled[0] is opt.b[0]
        assert scheduled[1] is p
        scheduled[0].realize(*scheduled[1:])

    np.testing.assert_allclose(opt.b[0].numpy(), [0.35, -0.3], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(p.numpy(), [0.9335, 2.057], rtol=1e-6, atol=1e-6)


def test_optimizer_mixed_device_failure_does_not_mutate_earlier_param():
    p0 = Tensor([1.0], requires_grad=True, device='CPU').realize()
    p1 = Tensor([2.0], requires_grad=True, device='CUDA')
    p0._grad = Tensor([1.0], device='CPU')
    p1._grad = Tensor([1.0], device='CUDA')
    opt = SGD([p0, p1], lr=0.1)
    p0_root = p0.uop.raw
    p1_root = p1.uop.raw

    with pytest.raises(RuntimeError, match='optimizer step graph build failed'):
        opt.schedule_step()

    assert p0.uop.raw == p0_root
    assert p1.uop.raw == p1_root
    np.testing.assert_allclose(p0.numpy(), [1.0], rtol=0, atol=0)


def test_onecyclelr_initial_and_step_values_match_tinygrad_probe():
    p = Tensor([1.0], requires_grad=True).realize()
    opt = SGD([p], lr=0.001)
    scheduler = OneCycleLR(
        opt,
        max_lr=0.1,
        div_factor=10.0,
        final_div_factor=100.0,
        total_steps=4,
        pct_start=0.5,
    )

    values = [opt.lr.item()]
    for _ in range(5):
        scheduler.step()
        values.append(opt.lr.item())

    np.testing.assert_allclose(
        values,
        [0.01, 0.055, 0.1, 0.05005, 0.0001, -0.04985],
        rtol=1e-5,
        atol=1e-6,
    )


def test_tinyjit_optimizer_replay_uses_current_onecyclelr_tensor():
    p = Tensor([1.0], requires_grad=True).realize()
    p._grad = Tensor([1.0]).realize()
    opt = SGD([p], lr=0.01, momentum=0.0, fused=False)
    scheduler = OneCycleLR(
        opt,
        max_lr=0.4,
        div_factor=4.0,
        final_div_factor=100.0,
        total_steps=4,
        pct_start=1.0,
    )
    x = Tensor([0.0]).realize()

    @TinyJit
    def step(inp):
        opt.step()
        scheduler.step()
        return (inp + 1).realize()

    params, lrs = [], []
    for _ in range(3):
        step(x)
        params.append(p.item())
        lrs.append(opt.lr.item())

    assert step.cnt == 3
    np.testing.assert_allclose(params, [0.9, 0.725, 0.475], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(lrs, [0.175, 0.25, 0.325], rtol=1e-6, atol=1e-6)
