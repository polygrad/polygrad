from pathlib import Path
import re

import numpy as np
import pytest

from extra.lr_scheduler import OneCycleLR
from polygrad import _ffi
from polygrad import Context, GlobalCounters, Jit, Tensor, TinyJit, UOp, Variable, dtypes, fetch, getenv, nn
from polygrad.helpers import Context as HelperContext, TRAINING, fetch as helper_fetch, getenv as helper_getenv
from polygrad.nn.optim import SGD
from polygrad.uop.ops import UOp as OpsUOp, resolve


def test_dtype_has_no_pointer_or_image_subclasses():
    # Tinygrad 2026-07-07 removed PtrDType and 2026-07-08 removed ImageDType;
    # ParamArg and UOp shape carry storage metadata.
    assert not hasattr(dtypes.float, "ptr")
    assert not hasattr(dtypes, "imagef")


@pytest.mark.parametrize('name', ['bool', 'short', 'int', 'long', 'half', 'bfloat16', 'float', 'double'])
def test_tensor_dtype_object_matches_current_uop(name):
    tensor = getattr(Tensor([1, 2]), name)()
    assert tensor.dtype is getattr(dtypes, name)
    assert tensor.dtype is tensor.uop_physical.dtype
    assert tensor.dtype.itemsize == getattr(dtypes, name).itemsize
    assert {tensor.dtype: 'typed'}[getattr(dtypes, name)] == 'typed'
    tensor.realize()
    assert tensor.dtype is tensor.uop_physical.dtype
    np.testing.assert_array_equal(tensor.numpy(), [1, 1] if name == 'bool' else [1, 2])


def test_uop_default_context_factories_match_tensor_owner():
    value = UOp.const(2.0)
    typed = UOp.const(2.0, dtypes.float32)
    variable = UOp.variable('api_extent', 1, 8)
    bound = variable.bind(3)
    assert value.ctx == Tensor(2.0)._ctx
    assert value.dtype is dtypes.weakfloat
    assert typed.dtype is dtypes.float32
    assert value.op_name == typed.op_name == 'CONST'
    assert variable.op_name == 'BUFFER' and len(variable.src) == 1
    assert bound.op_name == 'AFTER'
    assert tuple(s.op_name for s in bound.src) == ('BUFFER', 'STORE')
    np.testing.assert_equal(Tensor(typed).numpy(), 2.0)


def test_uop_explicit_context_factories_keep_runtime_ownership():
    from polygrad import Runtime
    runtime = Runtime(device='cpu')
    try:
        value = UOp.const(2.0, dtypes.float32, ctx=runtime._ctx)
        variable = UOp.variable('api_runtime_extent', 1, 8, ctx=runtime._ctx)
        tensor = runtime.Tensor(value)
        assert value.ctx == variable.ctx == tensor._ctx == runtime._ctx
        np.testing.assert_equal(tensor.numpy(), 2.0)
    finally:
        runtime.dispose()
    assert value.raw is None and variable.raw is None


def test_uop_const_rejects_a_foreign_explicit_owner():
    from polygrad import Runtime
    runtime = Runtime(device='cpu')
    try:
        value = UOp.const(2.0, dtypes.float32)
        with pytest.raises(ValueError, match='context mismatch'):
            UOp.const(value, dtypes.float32, ctx=runtime._ctx)
        assert UOp.const(value, dtypes.float32, ctx=value.ctx) == value
    finally:
        runtime.dispose()


@pytest.mark.parametrize('dtype', dtypes.all + dtypes.weaks)
def test_tensor_dtype_queries_match_pinned_metadata(dtype):
    tensor = Tensor(0, dtype=dtype)
    assert tensor.is_floating_point() is dtypes.is_float(dtype)
    if dtype in dtypes.weaks:
        with pytest.raises(RuntimeError, match='element_size requires a concrete dtype'):
            tensor.element_size()
    else:
        assert tensor.element_size() == dtype.itemsize


def test_dtype_is_scalar_only_like_current_tinygrad():
    # Tinygrad 2026-07-12 removed dtype.vec; UOp shape carries lane width.
    assert not hasattr(dtypes.float, "vec")


def test_top_level_public_exports_are_defining_module_objects():
    # Pinned tinygrad/__init__.py re-exports these exact objects rather than
    # defining wrappers or alternate compatibility implementations.
    assert Context is HelperContext
    assert UOp is OpsUOp
    assert fetch is helper_fetch
    assert getenv is helper_getenv
    assert getenv("POLYGRAD_MISSING_EXPORT_TEST", 17) == 17


def test_temp_path_matches_pinned_helper_without_creating_files(tmp_path, monkeypatch):
    import getpass
    import tempfile
    from polygrad.helpers import temp

    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))
    assert temp('nested/artifact.bin') == (tmp_path / 'nested/artifact.bin').as_posix()
    assert temp('artifact.bin', append_user=True) == (tmp_path / f'artifact.bin.{getpass.getuser()}').as_posix()
    assert temp('/absolute/artifact.bin') == '/absolute/artifact.bin'
    assert list(tmp_path.iterdir()) == []


def test_mv_address_matches_writable_view_offsets_and_rejections():
    import ctypes
    from polygrad.helpers import mv_address, to_mv

    storage = bytearray(range(16))
    view = memoryview(storage)
    address = mv_address(view)
    assert address == ctypes.addressof(ctypes.c_char.from_buffer(storage))
    assert mv_address(view[4:]) == address + 4
    assert mv_address(view.cast('I')[1:]) == address + 4
    to_mv(address + 4, 1)[0] = 99
    assert storage[4] == 99
    for invalid in (memoryview(b'abc'), view[::2]):
        with pytest.raises(TypeError):
            mv_address(invalid)
    with pytest.raises(ValueError):
        mv_address(view[:0])


def test_ffi_dtype_layout_is_the_current_scalar_c_abi():
    # src/polygrad.h:PolyDType has no vector/pointer fields. Passing the old
    # larger structure by value would use a different libffi calling layout.
    import ctypes
    assert _ffi.PolyDType._fields_ == [
        ('priority', ctypes.c_int8), ('bitsize', ctypes.c_uint16),
        ('name', ctypes.c_char_p), ('fmt', ctypes.c_char),
    ]


def test_can_lossless_cast_exposes_existing_core_dtype_rules():
    from polygrad.dtype import can_lossless_cast

    for source, dest, expected in (
        (dtypes.int8, dtypes.uint64, False), (dtypes.int32, dtypes.uint32, False),
        (dtypes.uint8, dtypes.int16, True), (dtypes.uint32, dtypes.int64, True),
        (dtypes.int32, dtypes.float, False), (dtypes.int64, dtypes.double, False),
        (dtypes.int8, dtypes.half, True), (dtypes.int8, dtypes.bfloat16, False),
        (dtypes.int64, dtypes.weakint, True), (dtypes.weakfloat, dtypes.float, False),
        (dtypes.bool, dtypes.int8, True), (dtypes.float, dtypes.float, True),
    ):
        assert can_lossless_cast(source, dest) is expected


def test_uop_resolve_simplifies_before_using_bounds():
    # Pinned tinygrad/uop/ops.py:50-54 rewrites first, then returns a proven
    # boolean only when simplified vmin/vmax agree.
    v = Variable('resolve_v', 0, 5).uop
    self_equal = v.eq(v)
    assert self_equal.op_name == 'CMPNE'
    assert tuple(src.op_name for src in self_equal.src) == ('CMPNE', 'CONST')
    assert resolve(True, False) is True
    assert resolve(v.lt(10), False) is True
    assert resolve(v.lt(0), True) is False
    assert resolve(v.lt(3), True) is True
    assert resolve(v.lt(3), False) is False
    assert resolve(self_equal, False) is True
    assert resolve((v * 0).eq(0), False) is True
    with pytest.raises(AssertionError, match='must be bool'):
        resolve(v)


def test_uop_literals_and_binary_promotion_match_current_tinygrad():
    ctx = Variable('literal_ctx', 0, 4)._ctx
    integer = UOp.const(1, ctx=ctx)
    floating = UOp.const(1.0, ctx=ctx)
    value = UOp.variable('literal_float', 0, 4, dtype=dtypes.float32, param=True, ctx=ctx)
    out = value + 1

    assert integer.dtype is dtypes.weakint
    assert floating.dtype is dtypes.weakfloat
    assert out.dtype is dtypes.float32
    assert out.src[1].dtype is dtypes.weakfloat
    assert UOp.const(True, ctx=ctx).dtype is dtypes.bool
    assert UOp.const(1, dtypes.float32, ctx=ctx).dtype is dtypes.float32
    assert UOp.const(1.75, dtypes.int32, ctx=ctx).dtype is dtypes.int32
    assert UOp.const(2, dtypes.bool, ctx=ctx).dtype is dtypes.bool


def test_uop_contiguous_folds_device_free_value_like_current_tinygrad():
    # Tinygrad 2026-08-22/a9069c177a9d mixin/elementwise.py:55-61 returns a
    # device-free UOp unchanged because it has no storage to materialize.
    ctx = Variable('contiguous_uop_ctx', 0, 1)._ctx
    value = UOp.const(1.0, dtypes.float32, ctx=ctx)
    assert value.contiguous().raw == value.raw


def test_tensor_module_cast_is_typing_cast_identity():
    # Pinned tinygrad/tensor.py:5 imports this public name from typing; it is
    # not a Tensor CAST operation.
    from polygrad.tensor import cast

    value = [1, 2]
    assert cast(list[int], value) is value


def test_training_is_shared_context_var_and_restores_state():
    events = []

    @Context(TRAINING=1)
    def decorated(value):
        events.append(("decorated", TRAINING.value))
        return value + 1

    assert decorated(4) == 5
    assert TRAINING.value == 0
    with Context(TRAINING=0) as entered:
        assert entered is None
        events.append(("outer", TRAINING.value))
        with Context(TRAINING=1):
            events.append(("inner", TRAINING.value))
        events.append(("restored_outer", TRAINING.value))
    assert TRAINING.value == 0
    with pytest.raises(RuntimeError, match="probe"):
        with Context(TRAINING=1):
            raise RuntimeError("probe")
    assert TRAINING.value == 0
    assert events == [
        ("decorated", 1),
        ("outer", 0),
        ("inner", 1),
        ("restored_outer", 0),
    ]


def test_top_level_tinyjit_alias_uses_existing_jit():
    assert TinyJit is Jit
    assert TinyJit.__name__ == "TinyJit"

    from polygrad.engine.jit import JitError as EngineJitError
    from polygrad.engine.jit import TinyJit as EngineTinyJit

    assert EngineTinyJit is TinyJit
    assert EngineJitError.__name__ == "JitError"

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
    assert not hasattr(nn, "SGD")
    assert nn.optim.SGD is SGD
    # Pinned Tensor.arange is device-free and realize() is a no-op. Use the
    # host-backed path to exercise one real CPU call and its counters.
    x = Tensor(np.arange(8, dtype=np.float32)).realize(do_update_stats=False)
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
    # TinyJit rejects pinned device-free arange inputs as non-buffer values.
    x = Tensor(np.arange(8, dtype=np.float32)).realize(do_update_stats=False)

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
            bind_count += uop.is_bound_var
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
    # A host-backed input is deviceful; pinned pure arange remains lazy and
    # executes no bound-view kernel.
    x = Tensor(np.arange(8, dtype=np.float32)).realize(do_update_stats=False)
    n = Variable('n', 1, 8)
    out = x[:n.bind(4)] + 1

    GlobalCounters.reset()
    out.realize()
    assert (GlobalCounters.global_ops, GlobalCounters.global_mem, GlobalCounters.kernel_count) == (
        4, 32, 1,
    )


def test_pure_view_realize_is_zero_call():
    x = Tensor.arange(8, dtype='float32').realize(do_update_stats=False)
    source_root = x.uop.raw
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
    assert x.uop.raw == source_root


def test_realized_contiguous_and_readback_reuse_current_buffer_identity():
    # Host-backed input makes the ADD deviceful in pinned tinygrad. A pure
    # arange+1 root is device-free and realize() is intentionally a no-op.
    source = (Tensor(np.arange(8, dtype=np.float32)) + 1).preserve_logical().realize()
    source_current = source.uop.raw
    source_logical = source.uop_logical.raw
    assert source.uop_logical.op_name == 'ADD'
    assert source.uop.has_buffer_identity()

    out = source.contiguous()
    assert out is not source
    assert out.uop_logical.raw == source_logical
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

    sliced = source[2:6]
    sliced_current = sliced.uop.raw
    assert sliced.uop.op_name == 'SHRINK'
    assert not sliced.uop.has_buffer_identity()
    sliced.realize()
    # Pinned Tensor.realize keeps a pure SHRINK(BUFFER) view unchanged:
    # tensor.py:214-219 filters only roots without buffer identity, while
    # callify.py:169-181 maps a realized movement view back to that view.
    assert sliced.uop.raw == sliced_current
    assert sliced.uop.op_name == 'SHRINK'
    assert not sliced.uop.has_buffer_identity()
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
        pair = first.cat(second)
        pair_ops, pair_stack = set(), [pair.uop]
        while pair_stack:
            uop = pair_stack.pop()
            if uop in pair_ops:
                continue
            pair_ops.add(uop)
            pair_stack.extend(uop.src)
        assert sum(uop.op_name == 'AFTER' for uop in pair_ops) == 2
        assert sum(uop.op_name == 'STORE' for uop in pair_ops) == 2
        np.testing.assert_array_equal(pair.numpy(), expected[:2].reshape(-1))

        Tensor.manual_seed(201)
        jit_input = Tensor.zeros(8).contiguous().realize()

        @TinyJit
        def random_jit(value):
            return (Tensor.rand(*value.shape) + value).contiguous().preserve_logical()

        actual = np.stack([random_jit(jit_input).numpy().copy() for _ in range(4)])
        np.testing.assert_array_equal(actual, expected)

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
    assert _ffi.get_lib().poly_abi_version() == _ffi.POLYGRAD_ABI_VERSION == 73


def test_python_source_manifest_contains_quoted_dependencies():
    import runpy
    root = Path(__file__).resolve().parents[2]
    manifest = runpy.run_path(str(root / 'py/scripts/sync-csrc.py'))
    shipped = {(root / rel).resolve() for rel in manifest['SOURCES'] + manifest['HEADERS']}
    missing = []
    # Match setup.py's include paths as well as each source's local directory.
    # Mirror equality alone cannot detect a dependency omitted from the manifest.
    for source in sorted(shipped):
        for include in re.findall(r'^\s*#\s*include\s*"([^"]+)"', source.read_text(encoding='utf-8'), re.M):
            candidates = [source.parent / include, root / 'src' / include,
                          root / 'vendor/cjson' / include]
            resolved = next((p.resolve() for p in candidates if p.is_file()), None)
            if resolved not in shipped:
                missing.append(f'{source.relative_to(root)}: {include}')
    assert not missing, '\n'.join(missing)


@pytest.mark.parametrize('relative', [
    'js/src/core/native.js',
    'js/src/core/wasm_common.js',
])
def test_javascript_loaders_expect_the_current_frontend_abi(relative):
    source = (Path(__file__).resolve().parents[2] / relative).read_text()
    match = re.search(r'const EXPECTED_ABI = (\d+)', source)
    assert match is not None
    assert int(match.group(1)) == _ffi.POLYGRAD_ABI_VERSION


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
    with pytest.raises(
        RuntimeError,
        match=rf'expected version {_ffi.POLYGRAD_ABI_VERSION}, got 20',
    ):
        _ffi.get_lib()
    assert _ffi._lib is None


@Context(TRAINING=1)
def test_optimizer_lr_is_assignable_tensor_and_feeds_c_graph():
    p = Tensor([1.0]).realize()
    opt = SGD([p], lr=0.1)

    assert isinstance(opt.lr, Tensor)
    opt.lr.assign(Tensor([0.2], dtype=opt.lr.dtype)).realize()
    p._grad = Tensor([1.0])
    opt.step()

    np.testing.assert_allclose(p.numpy(), [0.8], rtol=1e-6, atol=1e-6)


def test_optimizer_schedule_requires_shared_training_context():
    p = Tensor([1.0]).realize()
    p._grad = Tensor([1.0])
    opt = SGD([p], lr=0.1, fused=False)

    with pytest.raises(RuntimeError, match="TRAINING=0, TRAINING must be enabled"):
        opt.schedule_step()
    with Context(TRAINING=1):
        scheduled = opt.schedule_step()
    assert [tensor.uop.op_name for tensor in scheduled] == ["AFTER"]


@Context(TRAINING=1)
def test_sgd_momentum_commits_lazy_backward_gradient_before_view_state_assign():
    p = Tensor([1.0]).realize()
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
    p = Tensor([1.0, 2.0]).realize()
    p._grad = Tensor([0.25, -0.5], dtype=p.dtype)
    opt = SGD(
        [p], lr=0.1, momentum=0.9, nesterov=True,
        weight_decay=0.1, fused=False,
    )

    with Context(TRAINING=1):
        scheduled = opt.schedule_step()
        assert len(scheduled) == 2
        assert scheduled[0] is opt.b[0]
        assert scheduled[1] is p
        scheduled[0].realize(*scheduled[1:])

    np.testing.assert_allclose(opt.b[0].numpy(), [0.35, -0.3], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(p.numpy(), [0.9335, 2.057], rtol=1e-6, atol=1e-6)


@Context(TRAINING=1)
def test_optimizer_mixed_device_failure_does_not_mutate_earlier_param():
    p0 = Tensor([1.0], device='CPU').realize()
    p1 = Tensor([2.0], device='CUDA')
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
    p = Tensor([1.0]).realize()
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


@Context(TRAINING=1)
def test_tinyjit_optimizer_replay_uses_current_onecyclelr_tensor():
    p = Tensor([1.0]).realize()
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
