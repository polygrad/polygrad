"""Imported and Tensor-authored children compose through ordinary Model calls."""
import numpy as np
import pytest

from polygrad import create


def test_late_embedding_composition_preserves_query_and_document_masks():
    from pathlib import Path
    import runpy
    compose = runpy.run_path(str(Path(__file__).parents[1] / 'examples/late_embeddings.py'))['compose_token_embeddings']
    with create(device='CPU') as rt:
        T, M = rt.Tensor, rt.Model
        inputs = {k:T.empty(1,4,dtype='int32') for k in ('input_ids','attention_mask','position_ids')}
        table = T(np.array([[0,0],[1,2],[3,4],[5,6]],'float32')).realize()
        child = M(lambda input_ids,attention_mask,position_ids: {'last_hidden_state': table[input_ids]},
                  inputs=inputs,params={'table':table})
        matrix = np.array([[1,0],[0,1],[1,1]],'float32')
        projection = T(matrix).realize()
        parent = compose(rt,child,projection,inputs,[2])
        child.dispose()
        data = dict(input_ids=np.array([[1,2,3,0]],'int32'),attention_mask=np.array([[1,1,1,0]],'int32'),
                    position_ids=np.arange(4,dtype='int32')[None,:])
        try:
            query = parent.forward(**data,exclude_tokens=np.array([0],'int32'))
            document = parent.forward(**data,exclude_tokens=np.array([1],'int32'))
            np.testing.assert_array_equal(query['token_mask'],[[True,True,True,False]])
            np.testing.assert_array_equal(document['token_mask'],[[True,False,True,False]])
            np.testing.assert_array_equal(query['token_embeddings'],document['token_embeddings'])
            expected = np.array([[[1,2],[3,4],[5,6],[0,0]]],'float32')
            expected = expected @ matrix.T
            expected /= np.maximum(np.linalg.norm(expected,axis=-1,keepdims=True),1e-12)
            np.testing.assert_allclose(query['token_embeddings'],expected,atol=1e-6)
        finally: parent.dispose()


@pytest.fixture(params=['until_realize', 'always'])
def rt(request):
    with create(device='CPU', logical=request.param) as runtime:
        yield runtime


@pytest.mark.parametrize('computed', [False, True])
@pytest.mark.parametrize('saved', [False, True])
def test_composed_matrix_state_preserves_shapes_and_gradients(tmp_path, rt, computed, saved):
    Tensor, Model = rt.Tensor, rt.Model
    x = Tensor.empty(3)
    state = Tensor(np.arange(1, 7, dtype=np.float32).reshape(2, 3))
    if computed:
        state = state * 2
    child = Model.from_tensors(inputs={'x': x}, params={'state': state}, outputs={'y': state @ x})
    data = np.array([1, 2, 3], np.float32)
    expected = np.array([14, 32], np.float32) * (2 if computed else 1)
    np.testing.assert_array_equal(child.forward(x=data)['y'], expected)
    if saved:
        path = tmp_path / 'child.bundle'
        child.save(path)
        child.dispose()
        child = Model.load(path)
    # Compose the current stored state, not the initializer expression.
    scale = 3 * (2 if computed else 1)
    child.write_buffer('state', np.arange(1, 7, dtype=np.float32) * scale)
    expected *= 3

    def forward(x):
        a = child.forward(x=x)['y']
        b = child.forward(x=x * 2)['y']
        assert a.shape == b.shape == (2,)
        return {'y': a + b, 'grad': (a + b).sum().gradient(x)[0]}

    parent = Model(forward, inputs={'x': x})
    assert parent.param_count == 1
    child.write_buffer('state', np.zeros(6, np.float32))
    child.dispose()
    state.dispose()
    rt.collect()
    path = tmp_path / 'parent.bundle'
    parent.save(path)
    restored = Model.load(path)
    try:
        for model in (parent, restored):
            result = model.forward(x=data)
            np.testing.assert_array_equal(result['y'], expected * 3)
            np.testing.assert_array_equal(result['grad'], np.array([15, 21, 27]) * scale)
    finally:
        restored.dispose()
        parent.dispose()
        x.dispose()


def test_composed_model_reuses_weights_and_survives_child_disposal(tmp_path, rt):
    Tensor, Model = rt.Tensor, rt.Model
    x = Tensor.empty(1, 4)
    w = Tensor(np.eye(4, dtype=np.float32) * 2).realize()
    child = Model(lambda x: x @ w, inputs={'x': x}, params={'w': w})
    # The same path must work for a loaded model, not just a retained callable.
    path = tmp_path / 'child.bundle'
    child.save(path)
    child.dispose()
    child = Model.load(path)

    def forward(x):
        for _ in range(3):
            x = child.forward(x=x)['output']
        return x[:, :2]

    parent = Model(forward, inputs={'x': x})
    assert parent.param_count == 1
    child.dispose()
    np.testing.assert_array_equal(parent.forward(x=np.ones((1, 4), np.float32))['output'], [[8, 8]])
    parent.dispose()


def test_composed_onnx_explicit_state_and_sampling(tmp_path, rt):
    Tensor, Model = rt.Tensor, rt.Model
    onnx = pytest.importorskip('onnx')
    from onnx import helper as h, TensorProto as T
    graph = h.make_graph([h.make_node('Add', ['x', 'past'], ['present'])], 'step',
                        [h.make_tensor_value_info(n, T.FLOAT, [1, 4]) for n in ('x', 'past')],
                        [h.make_tensor_value_info('present', T.FLOAT, [1, 4])])
    source = tmp_path / 'step.onnx'
    source.write_bytes(h.make_model(graph, opset_imports=[h.make_opsetid('', 18)]).SerializeToString())
    child = Model.from_onnx(source)
    x, past = Tensor.empty(1, 4), Tensor.empty(1, 4)

    def forward(x, past):
        for _ in range(3):
            past = child.forward(x=x, past=past)['present']
        return {'present': past, 'token': past.sample_logits(top_k=1)}

    parent = Model(forward, inputs={'x': x, 'past': past})
    child.dispose()
    result = parent.forward(x=np.array([[0, 1, 2, 3]], np.float32), past=np.zeros((1, 4), np.float32))
    np.testing.assert_array_equal(result['present'], [[0, 3, 6, 9]])
    np.testing.assert_array_equal(result['token'], [3])
    parent.dispose()


def test_composed_model_rejects_internal_effects_and_invalid_inputs(rt):
    Tensor, Model = rt.Tensor, rt.Model
    x = Tensor.empty(2)
    state = Tensor.zeros(2).realize().is_param_(False)

    def update(x):
        state.assign(state + x)
        return state

    child = Model(update, inputs={'x': x}, params={'state': state})
    with pytest.raises(RuntimeError, match='internal writes'):
        Model(lambda x: child.forward(x=x), inputs={'x': x})
    child.dispose()
    child = Model(lambda x: x + 1, inputs={'x': x})
    with pytest.raises(RuntimeError, match='shape'):
        Model(lambda x: child.forward(x=x), inputs={'x': Tensor.empty(3)})
    with pytest.raises(RuntimeError, match='every declared input'):
        Model(lambda x: child.forward(x=x, extra=x), inputs={'x': x})
    n = rt.Variable('composed_extent', 1, 2)
    with pytest.raises(RuntimeError, match='symbolic shape'):
        Model(lambda x: child.forward(x=x), inputs={'x': Tensor.empty(n.bind(2))})
    child.dispose()


def test_composed_model_failure_does_not_execute_child(rt):
    Tensor, Model = rt.Tensor, rt.Model
    x = Tensor.empty(2)
    child = Model(lambda x: x + 1, inputs={'x': x})

    def bad(x):
        child.forward(x=x)
        raise ValueError('abort construction')

    with pytest.raises(ValueError, match='abort construction'):
        Model(bad, inputs={'x': x})
    parent = Model(lambda x: child.forward(x=x), inputs={'x': x})
    np.testing.assert_array_equal(parent.forward(x=np.array([2, 3], np.float32))['output'], [3, 4])
    parent.dispose()
    child.dispose()


def test_composed_model_rejects_unplaced_device_inputs(rt):
    Tensor, Model = rt.Tensor, rt.Model
    x = Tensor.empty(2, device='CPU')
    child = Model(lambda x: x + 1, inputs={'x': x})
    foreign = Tensor.empty(2, device='INTERP')
    with pytest.raises(RuntimeError, match='device'):
        Model(lambda x: child.forward(x=x), inputs={'x': foreign})
    child.dispose()


def test_composed_weights_are_independent_after_sealing(rt):
    Tensor, Model = rt.Tensor, rt.Model
    x = Tensor.empty(2)
    w = Tensor([2., 3.]).realize()
    child = Model(lambda x: x * w, inputs={'x': x}, params={'w': w})
    data = np.ones(2, np.float32)
    # A previously executed child must still contribute its graph, not its last output.
    child.forward(x=data * 7)
    parent = Model(lambda x: child.forward(x=x), inputs={'x': x})
    child.write_buffer('w', np.array([4., 5.], np.float32))
    np.testing.assert_array_equal(parent.forward(x=data)['output'], [2, 3])
    parent.write_buffer('__model.0', np.array([6., 7.], np.float32))
    np.testing.assert_array_equal(child.forward(x=data)['output'], [4, 5])
    np.testing.assert_array_equal(parent.forward(x=data)['output'], [6, 7])
    parent.dispose()
    child.dispose()


def test_composed_model_accepts_device_free_constants(rt):
    Tensor, Model = rt.Tensor, rt.Model
    x = Tensor.empty(2)
    child = Model(lambda x: x + 1, inputs={'x': x})
    parent = Model(lambda x: child.forward(x=Tensor.zeros(2, dtype='float32', buffer=False))['output'] + x,
                   inputs={'x': x})
    np.testing.assert_array_equal(parent.forward(x=np.array([2, 3], np.float32))['output'], [3, 4])
    parent.dispose()
    child.dispose()


def test_composed_child_uses_placed_device_not_runtime_default(rt):
    Tensor, Model = rt.Tensor, rt.Model
    x = Tensor.empty(2)
    w = Tensor([2., 3.]).realize()
    child = Model(lambda x: x * w, inputs={'x': x}, params={'w': w}).place('INTERP')
    # Placing a sibling changes the context preference without moving the child.
    sibling = Model(lambda x: x + 1, inputs={'x': x}).place('CPU')
    placed_x = Tensor.empty(2, device='INTERP')

    def forward(x):
        y = child.forward(x=x)['output']
        return {'value': y, 'token': y.sample_logits(top_k=2)}

    parent = Model(forward, inputs={'x': placed_x})
    assert '__rng.interp.seed' in [binding['name'] for binding in parent.bindings()]
    child.dispose()
    sibling.dispose()
    result = parent.forward(x=np.ones(2, np.float32))
    np.testing.assert_array_equal(result['value'], [2, 3])
    assert np.asarray(result['token']).item() in (0, 1)
    parent.dispose()


def test_composed_slice_does_not_materialize_full_head(rt):
    Tensor, Model, GlobalCounters = rt.Tensor, rt.Model, rt.GlobalCounters
    x = Tensor.empty(1, 32)
    w = Tensor(np.ones((32, 1024), np.float32)).realize()
    child = Model(lambda x: x @ w, inputs={'x': x}, params={'w': w})
    parent = Model(lambda x: child.forward(x=x)['output'][:, :16], inputs={'x': x})
    data = np.ones((1, 32), np.float32)
    GlobalCounters.reset()
    child.forward(x=data)
    full_ops = GlobalCounters.global_ops
    GlobalCounters.reset()
    result = parent.forward(x=data)['output']
    assert GlobalCounters.global_ops < full_ops / 8, 'composition materialized the unused head'
    np.testing.assert_array_equal(result, np.full((1, 16), 32, np.float32))
    parent.dispose()
    child.dispose()
