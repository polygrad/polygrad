"""Shared causal delta recurrence; no checkpoint or optional reference required."""
import numpy as np
import pytest
import polygrad as pg
from polygrad.nn import gated_delta_rule


def reference(q, k, v, g, beta, state):
    q = q / np.sqrt((q*q).sum(-1, keepdims=True) + 1e-6) / np.sqrt(q.shape[-1])
    k = k / np.sqrt((k*k).sum(-1, keepdims=True) + 1e-6)
    state = state.copy()
    output = []
    for t in range(q.shape[1]):
        state *= np.exp(g[:, t, :, None, None])
        delta = (v[:, t] - (state * k[:, t, :, :, None]).sum(-2)) * beta[:, t, :, None]
        state += k[:, t, :, :, None] * delta[..., None, :]
        output.append((state * q[:, t, :, :, None]).sum(-2))
    return np.stack(output, axis=1), state


@pytest.mark.parametrize('length,chunk', [(1,1), (3,4), (8,8), (11,4), (65,64)])
def test_gated_delta_chunk_and_state(length, chunk):
    rng = np.random.default_rng(19)
    shape = (1,length,2,4)
    q, k = [rng.normal(size=shape).astype('float32') for _ in range(2)]
    v = rng.normal(size=(1,length,2,3)).astype('float32')
    g = -rng.random((1,length,2), dtype=np.float32)
    beta = rng.random((1,length,2), dtype=np.float32)
    state = rng.normal(size=(1,2,4,3)).astype('float32')
    expected = reference(q,k,v,g,beta,state)
    with pg.create(device='CPU') as rt:
        tensors = [rt.Tensor(x).realize() for x in (q,k,v,g,beta,state)]
        out, final = rt.nn.gated_delta_rule(*tensors, chunk_size=chunk)
        np.testing.assert_allclose(out.numpy(),expected[0],atol=2e-5,rtol=2e-4)
        np.testing.assert_allclose(final.numpy(),expected[1],atol=2e-5,rtol=2e-4)
        np.testing.assert_array_equal(tensors[-1].numpy(),state)
        out.dispose(); final.dispose()
        for t in tensors: t.dispose()


def test_gated_delta_correlated_keys_stay_stable():
    q = np.ones((1,64,1,4),np.float32)
    k = q.copy(); k[:,::2] *= -1
    v = np.linspace(-1,1,64,dtype=np.float32).reshape(1,64,1,1)
    g = np.zeros((1,64,1),np.float32)
    beta = np.ones_like(g)
    state = np.zeros((1,1,4,1),np.float32)
    with pg.create(device='CPU') as rt:
        tensors = [rt.Tensor(x).realize() for x in (q,k,v,g,beta,state)]
        out, final = gated_delta_rule(*tensors)
        expected = reference(q,k,v,g,beta,state)
        np.testing.assert_allclose(out.numpy(),expected[0],atol=2e-5)
        np.testing.assert_allclose(final.numpy(),expected[1],atol=2e-5)


def test_gated_delta_invalid_shapes_and_chunks():
    with pg.create(device='CPU') as rt:
        q = rt.Tensor.ones(1,2,1,2)
        g = rt.Tensor.zeros(1,2,1)
        for chunk in (0,3,128,2**32+64):
            with pytest.raises(ValueError,match='chunk_size'):
                gated_delta_rule(q,q,q,g,g,chunk_size=chunk)
        with pytest.raises(ValueError,match='shapes'):
            gated_delta_rule(q,q,q,g,g,rt.Tensor.zeros(1,1,3,2))
        with pytest.raises(ValueError,match='floating'):
            gated_delta_rule(q.cast('int32'),q,q,g,g)


def test_gated_delta_prefill_work_budget():
    # 64 rows must not turn into 64 sequential triangular row solves.
    rng = np.random.default_rng(33)
    with pg.create(device='CPU') as rt:
        inputs = [rt.Tensor(rng.normal(size=(1,64,2,8)).astype('float32')).realize() for _ in range(3)]
        inputs += [rt.Tensor(-rng.random((1,64,2),dtype=np.float32)).realize(),
                   rt.Tensor(rng.random((1,64,2),dtype=np.float32)).realize()]
        out, state = gated_delta_rule(*inputs)
        rt.GlobalCounters.reset()
        out.realize(); state.realize()
        counters = rt.GlobalCounters
        counts = {name:getattr(counters,name) for name in ('kernel_count','global_ops','global_mem')}
        print(counts)
        assert counts['kernel_count'] <= 32, counts
        assert counts['global_ops'] <= 3_000_000, counts
        assert counts['global_mem'] <= 1_000_000, counts


def test_gated_delta_batches_are_independent_and_future_is_causal():
    rng = np.random.default_rng(47)
    data = [rng.normal(size=(2,9,2,3)).astype('float32') for _ in range(2)]
    data += [rng.normal(size=(2,9,2,5)).astype('float32'),
             -rng.random((2,9,2),dtype=np.float32),rng.random((2,9,2),dtype=np.float32)]
    initial = rng.normal(size=(2,2,3,5)).astype('float32')
    expected = reference(*data,initial)
    with pg.create(device='CPU') as rt:
        inputs = [rt.Tensor(a).realize() for a in [*data,initial]]
        actual = rt.nn.gated_delta_rule(*inputs,chunk_size=8)
        for got,want in zip(actual,expected):
            np.testing.assert_allclose(got.numpy(),want,atol=2e-5,rtol=2e-4)
        changed = data[2].copy(); changed[:,5:] += 100
        inputs[2] = rt.Tensor(changed).realize()
        output,_ = rt.nn.gated_delta_rule(*inputs,chunk_size=8)
        np.testing.assert_allclose(output.numpy()[:,:5],expected[0][:,:5],atol=2e-5,rtol=2e-4)


def test_gated_delta_portable_model_and_chunk_carry():
    rng = np.random.default_rng(7)
    data = [rng.normal(size=(1,6,2,4)).astype('float32') for _ in range(3)]
    data += [-rng.random((1,6,2),dtype=np.float32),rng.random((1,6,2),dtype=np.float32)]
    initial = rng.normal(size=(1,2,4,4)).astype('float32')
    expected = reference(*data,initial)
    names = ['q','k','v','g','beta','state']
    with pg.create(device='CPU',logical='always') as rt:
        inputs = [rt.Tensor.empty(*(a[:,:3].shape)) for a in data]
        inputs += [rt.Tensor.empty(*initial.shape)]
        out, state = rt.nn.gated_delta_rule(*inputs,chunk_size=4)
        model = rt.Model.from_tensors(inputs=dict(zip(names,inputs)),outputs={'out':out,'next_state':state})
        restored = rt.Model.load(model.save())
        model.dispose()
        for t in [*inputs,out,state]: t.dispose()
        result, carried = [], initial
        for start in (0,3):
            args = dict(zip(names,[a[:,start:start+3] for a in data]+[carried]))
            outputs = restored.forward(**args)
            result.append(outputs['out']); carried=outputs['next_state']
        np.testing.assert_allclose(np.concatenate(result,axis=1),expected[0],atol=2e-5,rtol=2e-4)
        np.testing.assert_allclose(carried,expected[1],atol=2e-5,rtol=2e-4)
        restored.dispose()
