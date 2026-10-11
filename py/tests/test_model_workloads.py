"""Small architectural workloads: values alone must not hide dense gathers.

MODEL_WORKLOAD_ENGINE=tinygrad runs the same Tensor programs against the pin.
Counters exclude construction, input upload and output readback. Byte counts
are compiler traffic estimates, not allocations or measured DRAM traffic.
"""

import contextlib
import gc
import importlib
import os

import numpy as np
import pytest


ENGINE = os.environ.get('MODEL_WORKLOAD_ENGINE', 'polygrad')


@pytest.fixture
def runtime():
    module = importlib.import_module(ENGINE)
    with module.create(device='CPU') if ENGINE == 'polygrad' else contextlib.nullcontext(module) as rt:
        yield rt


def resident(rt, values):
    return rt.Tensor(np.asarray(values), device='CPU').realize()


def measure(rt, build, budget):
    counters = rt.GlobalCounters
    counters.reset()
    before = rt.stats() if ENGINE == 'polygrad' else None
    result = build().realize()
    actual = {key: getattr(counters, key) for key in ('kernel_count', 'global_ops', 'global_mem')}
    if before is not None:
        actual['copies'] = rt.stats()['buffer_copy_count'] - before['buffer_copy_count']
    print(actual)
    for name, limit in budget.items():
        if name in actual:
            assert actual[name] <= limit, (name, actual, budget)
    return result.numpy()


@pytest.mark.parametrize('batch', [1, 3])
def test_moe_routing_work_budget(runtime, batch):
    """OLMoE-style selected-expert execution, extended to multiple tokens.

    The NumPy oracle is independent of the Tensor one-hot gather lowering.
    All expert tables are runtime storage, preventing constant-folded lookups.
    """
    rt = runtime
    rng = np.random.default_rng(23)
    experts, d, hidden, k = 16, 8, 12, 2
    xv = rng.normal(size=(batch, d)).astype(np.float32) * .2
    gv = rng.normal(size=(d, experts)).astype(np.float32) * .1
    uv = rng.normal(size=(experts, d, hidden)).astype(np.float32) * .1
    dv = rng.normal(size=(experts, hidden, d)).astype(np.float32) * .1
    x, gate, up, down = [resident(rt, v) for v in (xv, gv, uv, dv)]

    def forward():
        probabilities, indices = x.dot(gate).softmax(-1).topk(k)
        selected_up, selected_down = up[indices], down[indices]
        values = x.reshape(batch, 1, 1, d).matmul(selected_up).silu().matmul(selected_down)
        return (values.reshape(batch, k, d) * probabilities.unsqueeze(-1)).sum(1)

    # Pin CPU, batch3:21 kernels/23268 ops/12932 bytes. Bounded allowance
    # covers batch1's different schedule without admitting dense expert work.
    got = measure(rt, forward, dict(kernel_count=24, global_ops=24000, global_mem=16000, copies=0))
    logits = xv @ gv
    weights = np.exp(logits - logits.max(-1, keepdims=True))
    weights /= weights.sum(-1, keepdims=True)
    ids = np.argsort(-weights, axis=-1)[:, :k]
    h = xv[:, None, None, :] @ uv[ids]
    h = h / (1 + np.exp(-h))
    expected = ((h @ dv[ids]).reshape(batch, k, d)
                * np.take_along_axis(weights, ids, -1)[..., None]).sum(1)
    np.testing.assert_allclose(got, expected, atol=1e-7, rtol=2e-5)


@pytest.mark.parametrize('rows', [32767, 32768, 65536])
def test_expert_gather_across_split_threshold(runtime, rows):
    rt = runtime
    table = np.arange(rows * 4, dtype=np.float32).reshape(rows, 4) / 64
    ids = np.array([[0, rows - 1], [rows // 2, 0], [3, rows - 2]], np.int32)
    weight, indices = resident(rt, table), resident(rt, ids)
    # The pin splits large gathers into partial indexed reads plus a reduction.
    # This cost is independent of table size: both 32768 and 65536 measure the
    # same 27392 operations. A dense scan would grow with rows and exceed it.
    budget = (dict(kernel_count=1, global_ops=18, global_mem=216, copies=0) if rows < 32768
              else dict(kernel_count=2, global_ops=27392, global_mem=73848, copies=0))
    got = measure(rt, lambda: weight[indices], budget)
    np.testing.assert_array_equal(got, table[ids])


def test_unet_odd_spatial_work_budget(runtime):
    rt = runtime
    rng = np.random.default_rng(4)
    values = [rng.normal(size=s).astype(np.float32) * .1 for s in
              [(1, 2, 7, 9), (3, 2, 3, 3), (3, 2, 3, 3), (2, 4, 3, 3)]]
    x, down, up, merge = [resident(rt, v) for v in values]

    def forward():
        low = x.conv2d(down, stride=2, padding=1).elu()
        high = low.conv_transpose2d(up, stride=2, padding=1)
        return x.cat(high, dim=1).conv2d(merge, padding=1).relu()

    got = measure(rt, forward, dict(kernel_count=3, global_ops=19000, global_mem=5000, copies=0))
    torch = pytest.importorskip('torch')
    f = torch.nn.functional
    tx, td, tu, tm = [torch.from_numpy(v) for v in values]
    high = f.conv_transpose2d(f.elu(f.conv2d(tx, td, stride=2, padding=1)), tu, stride=2, padding=1)
    expected = f.relu(f.conv2d(torch.cat((tx, high), dim=1), tm, padding=1))
    np.testing.assert_allclose(got, expected.numpy(), atol=1e-7, rtol=1e-4)


def test_residual_training_state_stays_bounded(runtime):
    rt = runtime
    rng = np.random.default_rng(11)
    conv = rt.nn.Conv2d(2, 2, 3, padding=1, bias=False)
    bn = rt.nn.BatchNorm2d(2)
    conv.weight = resident(rt, rng.normal(size=(2, 2, 3, 3)).astype(np.float32) * .05)
    # Optimizer.step also realizes non-parameter state, as get_parameters(model)
    # does in pinned ResNet training. Omitting these leaves pending BN updates.
    params = [conv.weight, bn.weight, bn.bias, bn.running_mean, bn.running_var, bn.num_batches_tracked]
    opt = rt.nn.optim.SGD(params, lr=.002, momentum=.9)
    xs = [resident(rt, rng.normal(size=(b, 2, 5, 5)).astype(np.float32)) for b in (3, 1)]
    ys = [resident(rt, np.zeros(x.shape, np.float32)) for x in xs]
    snapshots = []
    with importlib.import_module(ENGINE).Context(TRAINING=1):
        for step in range(80):
            x, y = xs[step % 2], ys[step % 2]
            # Both engines accumulate gradients on floating inputs too.
            for v in xs + ys:
                v.grad = None
            opt.zero_grad()
            prediction = (bn(conv(x)) + x).relu()
            loss = (prediction - y).square().mean()
            loss.backward()
            opt.step()
            assert np.isfinite(loss.item())
            del loss, prediction
            if step in (39, 79) and ENGINE == 'polygrad':
                gc.collect()
                rt.collect()
                snapshots.append(rt.stats())
    assert bn.num_batches_tracked.item() == 80
    if snapshots:
        for key in ('arena_bytes', 'cse_entries', 'buffer_entries', 'buffer_owned_bytes'):
            assert snapshots[1][key] == snapshots[0][key], (key, snapshots)
