"""Tests for the PolyModel Python wrapper."""

import numpy as np
import pytest
from polygrad.model import Model, OPTIM_SGD, OPTIM_ADAM, OPTIM_ADAMW
from polygrad.models import MLP, Graph, Sequential
from polygrad.tensor import Tensor


@pytest.mark.parametrize('device', ['cpu', 'interp'])
def test_definition_typed_bounded_components(device):
    import json
    from pathlib import Path
    import polygrad as pg
    spec = json.loads((Path(__file__).parents[2] / 'test/fixtures/model_components.json').read_text())
    expected = json.loads((Path(__file__).parents[2] / 'test/fixtures/model_components_expected.json').read_text())
    with pg.create(device=device) as rt:
        model = rt.Model(spec)
        try:
            model.write_buffer('nodes.embedding.weight', np.array(expected['table'], np.float32))
            model.write_buffer('nodes.head.weight', np.array(expected['linear'], np.float32))
            for batch in (3, 1):
                outputs = model.forward(tokens=np.arange(batch*3, dtype=np.int32).reshape(batch,3)%4)
                assert outputs['prediction'].shape == (batch,3,2)
                oracle = next(case for case in expected['cases'] if case['batch'] == batch)
                np.testing.assert_allclose(outputs['prediction'], oracle['prediction'], atol=2e-5)
                np.testing.assert_allclose(outputs['mean'], outputs['prediction'].mean(), atol=1e-6)
            loaded = rt.Model.load(model.save())
            try:
                assert loaded.save() == model.save()
                for name, value in loaded.forward(tokens=np.array([[0,1,2]], np.int32)).items():
                    np.testing.assert_allclose(value, outputs[name], atol=1e-6)
            finally:
                loaded.dispose()
        finally:
            model.dispose()


@pytest.mark.parametrize('dtype', ['int32', 'int64', 'uint8', 'bool', 'float16', 'float64'])
def test_definition_preserves_input_dtype(dtype):
    model = Sequential({'input':{'name':'x','shape':[2,2],'dtype':dtype},
                        'layers':[{'name':'copy','type':'identity'}], 'output':'y'})
    try:
        value = np.array([[0,1],[1,0]], dtype=dtype)
        result = model.forward(x=value)['y']
        assert result.dtype == value.dtype
        np.testing.assert_array_equal(result, value)
    finally:
        model.dispose()


def test_definition_bounded_mean_uses_invocation_extent():
    model = Sequential({'input':{'name':'x','shape':[{'name':'n','min':1,'max':4},2],'dtype':'int32'},
                        'layers':[{'name':'average','type':'mean'}], 'output':'y'})
    try:
        for rows in (4,1,3):
            x = np.arange(rows*2, dtype=np.int32).reshape(rows,2)
            np.testing.assert_allclose(model.forward(x=x)['y'], x.mean())
        with pytest.raises((ValueError, RuntimeError), match='bound|extent'):
            model.forward(x=np.zeros((5,2), np.int32))
    finally:
        model.dispose()


@pytest.mark.parametrize('shape,dtype,message', [
    ([{'min':1,'max':3},2], 'float32', 'name'),
    ([{'name':'n','min':3,'max':1},2], 'float32', 'integer'),
    ([2,{'name':'n','min':1,'max':3}], 'float32', 'leading'),
    ([2], 'weakint', 'dtype'),
])
def test_definition_rejects_invalid_typed_bounds(shape, dtype, message):
    with pytest.raises(ValueError, match=message):
        Sequential({'input':{'name':'x','shape':shape,'dtype':dtype},
                    'layers':[{'name':'id','type':'identity'}], 'output':'y'})


def test_definition_shared_embedding_and_norm_state():
    model = Graph({
        'inputs':{'x':{'shape':[2],'dtype':'int32'}},
        'modules':{'emb':{'type':'embedding','vocab_size':4,'embed_dim':2},
                   'norm':{'type':'layernorm'}},
        'nodes':[{'name':'a','call':'emb','inputs':['x']},
                 {'name':'b','call':'emb','inputs':['x']},
                 {'name':'c','call':'norm','inputs':['a']},
                 {'name':'d','call':'norm','inputs':['b']}],
        'outputs':{'a':'c','b':'d'}})
    try:
        assert model.param_count == 3
        np.testing.assert_array_equal(model.read_buffer('modules.norm.weight'), [1,1])
        np.testing.assert_array_equal(model.read_buffer('modules.norm.bias'), [0,0])
        model.write_buffer('modules.emb.weight', np.arange(8, dtype=np.float32).reshape(4,2))
        outputs = model.forward(x=np.array([0,3], np.int32))
        np.testing.assert_array_equal(outputs['a'], outputs['b'])
        np.testing.assert_allclose(outputs['a'], [[-0.99998,0.99998]]*2, atol=1e-5)
    finally:
        model.dispose()


def test_definition_cast_permute_and_nonaffine_norm():
    model = Sequential({'input':{'name':'x','shape':[2,3],'dtype':'int32'}, 'output':'y',
        'layers':[{'name':'to_float','type':'cast','dtype':'float32'},
                  {'name':'transpose','type':'permute','axes':[1,0]},
                  {'name':'norm','type':'layernorm','affine':False}]})
    try:
        assert model.param_count == 0
        y = model.forward(x=np.arange(6, dtype=np.int32).reshape(2,3))['y']
        np.testing.assert_allclose(y, [[-0.9999978,0.9999978]]*3, atol=1e-6)
    finally:
        model.dispose()


def test_definition_bounded_target_and_training():
    bound = {'name':'batch','min':1,'max':4}
    spec = {'inputs':{'x':{'shape':[bound,1],'dtype':'float32'},
                      'y':{'shape':[bound,1],'dtype':'float32','role':'target'}},
        'nodes':[{'name':'head','type':'linear','out_features':1,'bias':False,'inputs':['x']},
                 {'name':'error','type':'sub','inputs':['head','y']},
                 {'name':'square','type':'square','inputs':['error']},
                 {'name':'average','type':'mean','inputs':['square']}],
        'outputs':{'prediction':'head','cost':'average'},
        'entrypoints':[{'name':'forward','inputs':['x'],'outputs':['prediction']},
                       {'name':'loss','inputs':['x','y'],'outputs':['cost'],'objective':'cost'}]}
    model = Graph(spec)
    try:
        model.write_buffer('nodes.head.weight', np.array([2], np.float32))
        model.set_optimizer('sgd', lr=.1)
        weight = 2.
        for rows in (4,1,3):
            x = np.arange(1, rows+1, dtype=np.float32).reshape(rows,1)
            loss = (weight**2 * (x*x)).mean()
            np.testing.assert_allclose(model.train_step(x=x,y=np.zeros_like(x)), loss, atol=1e-5)
            weight -= .2 * weight * (x*x).mean()
            np.testing.assert_allclose(model.read_buffer('nodes.head.weight'), [weight], atol=1e-5)
    finally:
        model.dispose()


@pytest.mark.parametrize('kind,shape,options', [
    ('embedding',[1024],{'vocab_size':1024,'embed_dim':32}),
    ('attention',[4097,1],{}),
])
def test_definition_component_intermediate_budget(kind, shape, options):
    inputs = ['x']*3 if kind == 'attention' else ['x']
    dtype = 'float32' if kind == 'attention' else 'int32'
    with pytest.raises(ValueError, match='element budget'):
        Graph({'inputs':{'x':{'dtype':dtype,'shape':shape}},
               'nodes':[{'name':'huge','type':kind,'inputs':inputs,**options}],
               'outputs':{'y':'huge'}})


@pytest.mark.parametrize('masked,gqa', [(False,False), (True,False), (False,True), (True,True)])
def test_definition_attention_options_match_shared_tensor(masked, gqa):
    q = np.arange(8, dtype=np.float32).reshape(1,2,2,2) / 10
    k = np.ones((1,1 if gqa else 2,2,2), np.float32)
    v = np.arange(k.size, dtype=np.float32).reshape(k.shape)
    data = dict(q=q,k=k,v=v)
    if masked: data['mask'] = np.array([[True,False],[True,True]])
    model = Graph({'inputs':{name:{'shape':list(value.shape),'dtype':str(value.dtype)}
                                     for name,value in data.items()},
                   'nodes':[{'name':'attn','type':'attention','inputs':list(data),'enable_gqa':gqa}],
                   'outputs':{'y':'attn'}})
    try:
        expected = Tensor(q).scaled_dot_product_attention(
            Tensor(k), Tensor(v), attn_mask=Tensor(data['mask']) if masked else None, enable_gqa=gqa)
        np.testing.assert_allclose(model.forward(**data)['y'], expected.numpy(), atol=1e-6)
    finally:
        model.dispose()


def test_definition_rejects_conflicting_batch_declarations():
    with pytest.raises(ValueError, match='same batch name and bounds'):
        Graph({'inputs':{'x':{'shape':[{'name':'n','min':1,'max':3},2],'dtype':'float32'},
                         'y':{'shape':[{'name':'n','min':1,'max':4},2],'dtype':'float32'}},
               'nodes':[], 'outputs':{'out':'x'}})


@pytest.mark.parametrize('family,config', [
    ('mlp', {'layers':[2, 1]}), ('tabm', {'layers':[2, 1], 'n_ensemble':2}),
    ('nam', {'n_features':2, 'hidden_sizes':[2]}),
    ('gpt2', {'vocab_size':8, 'n_embd':4, 'n_head':2, 'n_layer':1, 'n_positions':2}),
])
def test_registered_family_config_dispatch(family, config):
    import polygrad as pg
    with pg.create(device='cpu', logical='never') as rt:
        model = rt.Model({'format':'poly.modeldef@1', 'type':family, **config})
        try:
            assert model._ctx == rt._ctx
            assert model.bindings()
            assert model.entrypoints()
        finally:
            model.dispose()


@pytest.mark.parametrize('family,config,field', [
    ('MLP', {'layers':[2]}, 'layers'),
    ('MLP', {'layers':[2, 1], 'activation':7}, 'activation'),
    ('TabM', {'layers':[2, 1], 'n_ensemble':0}, 'n_ensemble'),
    ('NAM', {'n_features':2, 'hidden_sizes':[0]}, 'hidden_sizes'),
])
def test_family_config_errors_are_structured(family, config, field, capfd):
    import polygrad as pg
    with pytest.raises(ValueError, match=field):
        getattr(pg.models, family)(config)
    assert capfd.readouterr().err == ''


def test_tabm_factory_does_not_inherit_mlp_fixed_layer_limit():
    from polygrad.models import TabM
    model = TabM({'layers': [1] * 34, 'n_ensemble': 1})
    try:
        assert model.param_count == 33 * 4
    finally:
        model.dispose()


def test_bundle_resave_is_canonical_in_shared_runtime():
    import polygrad as pg
    with pg.create(device='cpu', logical='always') as rt:
        x = rt.Tensor.empty(2)
        w = rt.Tensor([2., 3.])
        model = rt.Model(lambda x: {'prediction': x * w}, inputs={'x': x}, params={'w': w})
        try:
            original = model.save(include_optimizer=False)
            copies = []
            try:
                for _ in range(3):
                    copies.append(rt.Model.load(original))
                    assert copies[-1].save(include_optimizer=False) == original
                copies[0].write_buffer('w', np.array([7, 8], np.float32))
                for independent in (model, *copies[1:]):
                    np.testing.assert_array_equal(independent.read_buffer('w'), [2, 3])
            finally:
                for copy in copies: copy.dispose()
        finally:
            model.dispose()


def test_model_input_and_training_errors_describe_contract(capfd):
    model = MLP(layers=[2, 1])
    try:
        with pytest.raises((ValueError, RuntimeError), match='x.*expected float32.*int64'):
            model.forward(x=np.array([[1, 2]], dtype=np.int64))
        with pytest.raises((ValueError, RuntimeError), match='objective'):
            model.fit(x=np.array([[1, 2]], dtype=np.float32))
        assert capfd.readouterr().err == ''
    finally:
        model.dispose()


def test_uniform_placement_accepts_exact_cpu_identity():
    import polygrad as pg
    with pg.create(device='cpu', logical='always') as rt:
        x = rt.Tensor.empty(2)
        model = rt.Model(lambda x: {'y': x + 1}, inputs={'x': x})
        try:
            portable = model.export_ir()
            for device in ('CPU:1', 'cpu:2', 'CPU', 'INTERP'):
                model.place(device)
                np.testing.assert_array_equal(
                    model.forward(x=np.array([2, 3], dtype=np.float32))['y'], [3, 4])
                assert model.export_ir() == portable
            for device in ('CUDA:1', 'CPU:bad', 'AUTO'):
                with pytest.raises((ValueError, RuntimeError)):
                    model.place(device)
                np.testing.assert_array_equal(
                    model.forward(x=np.array([2, 3], dtype=np.float32))['y'], [3, 4])
        finally:
            model.dispose()


@pytest.mark.parametrize('shape', [(2,), (2, 2)])
@pytest.mark.parametrize('weighted', [False, True, 'linear'])
def test_module_device_map_preserves_shaped_cuts(shape, weighted):
    import polygrad as pg
    with pg.create(device='cpu', logical='always') as rt:
        x = rt.Tensor.empty(*shape)
        w = rt.Tensor([[3., 1.], [2., 4.]]) if weighted == 'linear' else rt.Tensor([3., 4.])
        h = x.matmul(w) if weighted == 'linear' else x * w if weighted else x + 3
        y = h * 2
        model = rt.Model.from_tensors(
            inputs={'x': x}, outputs={'y': y}, params={'w': w} if weighted else {},
            modules=[{'name': 'stem', 'inputs': [x], 'output': h},
                     {'name': 'head', 'inputs': [h], 'output': y}])
        try:
            values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
            expected = (values @ [[3, 1], [2, 4]] if weighted == 'linear'
                        else values * [3, 4] if weighted else values + 3) * 2
            for device in ('CPU:1', 'INTERP'):
                model.set_device_map({'stem': 'CPU', 'head': device})
                np.testing.assert_array_equal(model.forward(x=values)['y'], expected)
                with pytest.raises(ValueError):
                    model.set_device_map({'stem': 'CPU'})
                np.testing.assert_array_equal(model.forward(x=values)['y'], expected)
        finally:
            model.dispose()


@pytest.mark.parametrize('device', ['cpu', 'interp', 'cuda'])
def test_disposed_model_storage_reclaimed_by_unrelated_readback(device):
    import polygrad as pg
    from polygrad.device import Device
    if device == 'cuda' and not Device.cuda_available():
        pytest.skip('poly_cuda_available() is false in the selected library')
    with pg.create(device=device) as rt:
        live = rt.Tensor([7.0]).realize()
        rt.collect()
        baseline = rt.stats()['mem_used']
        weight = rt.Tensor.zeros(1 << 20).contiguous().realize()
        model = rt.Model.from_tensors(outputs={'out': weight}, params={'weight': weight})
        try:
            weight.dispose()
            rt.collect()
            assert rt.stats()['mem_used'] >= baseline + (1 << 22)
        finally:
            model.dispose()
        assert live.item() == 7
        assert rt.stats()['mem_used'] <= baseline


@pytest.mark.parametrize('device', ['cpu', 'interp'])
def test_bound_runtime_model_construction_and_import(device):
    import polygrad as pg
    with pg.create(device=device) as rt:
        x = rt.Tensor.empty(2)
        model = rt.Model(lambda x: {'prediction': x + 1}, inputs={'x': x})
        try:
            data = model.save(include_optimizer=False)
            for constructor in (rt.Model.load, rt.Model.from_bundle):
                loaded = constructor(data)
                try:
                    assert loaded._ctx == rt._ctx
                    np.testing.assert_array_equal(loaded.forward(x=np.array([2., 3.], np.float32))['prediction'], [3, 4])
                finally:
                    loaded.dispose()
            family = rt.models.MLP(layers=[2, 1], batch_size=1)
            try:
                assert family._ctx == rt._ctx
            finally:
                family.dispose()
        finally:
            model.dispose()


@pytest.mark.parametrize('route', ['constructor', 'from_callable', 'from_tensors', 'from_bindings'])
def test_bound_runtime_model_rejects_foreign_authoring_before_callback(route):
    import polygrad as pg
    calls = []
    def author(x):
        calls.append(True)
        return x + 1
    with pg.create(device='interp') as rt, pg.create(device='interp') as foreign:
        x = foreign.Tensor.empty(2)
        with pytest.raises(ValueError, match='another (Runtime|PolyCtx)'):
            if route == 'constructor':
                rt.Model(author, inputs={'x': x})
            elif route == 'from_callable':
                rt.Model.from_callable(author, inputs={'x': x})
            elif route == 'from_tensors':
                rt.Model.from_tensors(inputs={'x': x}, outputs={'prediction': x + 1})
            else:
                rt.Model.from_bindings([('x', 'input', x), ('y', 'output', x + 1)],
                                       [('forward', ['x'], ['y'])])
        assert not calls


@pytest.mark.parametrize('device', ['cpu', 'interp'])
def test_bound_runtime_nn_model_training_and_import(device):
    import polygrad as pg
    with pg.create(device=device) as rt:
        net = rt.nn.Linear(1, 1)
        initial = net.weight.numpy().copy()
        model = rt.Model(net, inputs={'x': rt.Tensor.empty(2, 1)},
                         targets={'y': rt.Tensor.empty(2, 1)},
                         loss=lambda prediction, y: (prediction - y).square().mean())
        restored = None
        try:
            inputs = {'x': np.array([[1.], [2.]], np.float32),
                      'y': np.array([[3.], [5.]], np.float32)}
            losses = model.fit(inputs, epochs=3, optimizer='sgd', lr=0.01)
            assert np.isfinite(losses).all() and losses[-1] < losses[0]
            # Capture owns a state snapshot, not the author's Tensor storage.
            np.testing.assert_array_equal(net.weight.numpy(), initial)
            restored = rt.Model.load(model.save(include_optimizer=False))
            np.testing.assert_array_equal(restored.forward(x=inputs['x'])['output'],
                                          model.forward(x=inputs['x'])['output'])
            restored.write_buffer('weight', np.zeros_like(initial))
            assert not np.array_equal(restored.read_buffer('weight'), model.read_buffer('weight'))
        finally:
            if restored is not None:
                restored.dispose()
            model.dispose()


def test_bound_runtime_model_conflicting_owner_and_disposal():
    import polygrad as pg
    with pg.create(device='interp') as rt, pg.create(device='interp') as other:
        with pytest.raises(ValueError, match='another Runtime'):
            rt.Model.load(b'invalid', runtime=other)
        with pytest.raises(ValueError, match='another Runtime'):
            rt.models.MLP(layers=[2, 1], runtime=other)
        x = rt.Tensor.empty(2)
        model = rt.Model(inputs={'x': x}, outputs={'y': x + 1})
        model.dispose()
    with pytest.raises(RuntimeError, match='disposed'):
        rt.Model.load(b'invalid')
    with pytest.raises(RuntimeError, match='disposed'):
        rt.models.MLP(layers=[2, 1])


def test_bound_runtime_initialization_failure_releases_context(monkeypatch):
    import polygrad as pg
    from polygrad import _ffi
    allocated, destroyed = [], []
    new, destroy = _ffi._lib.poly_ctx_new, _ffi._lib.poly_ctx_destroy
    def allocate():
        ctx = new()
        allocated.append(ctx)
        return ctx
    def release(ctx):
        destroyed.append(ctx)
        destroy(ctx)
    def fail(runtime):
        raise MemoryError('namespace allocation')
    monkeypatch.setattr(_ffi._lib, 'poly_ctx_new', allocate)
    monkeypatch.setattr(_ffi._lib, 'poly_ctx_destroy', release)
    monkeypatch.setattr(pg.nn, '_bind_runtime', fail)
    try:
        with pytest.raises(MemoryError, match='namespace allocation'):
            pg.create(device='interp')
        assert destroyed == allocated
    finally:
        # Keep the observed pre-fix failure from leaking the test's C owner.
        for ctx in allocated:
            if ctx not in destroyed:
                destroy(ctx)


def test_callable_model_captures_training_batchnorm_without_mutating_author():
    from polygrad import nn
    from polygrad.helpers import TRAINING
    class Net:
        def __init__(self): self.bn = nn.BatchNorm(2)
        def __call__(self, x): return {'prediction':self.bn(x)}
    net = Net()
    previous_mode = TRAINING.value
    model = Model(net, inputs={'x':Tensor.empty(2,2)}, targets={'y':Tensor.empty(2,2)},
                  loss=lambda out,y:(out['prediction']-y).square().mean())
    restored = None
    try:
        assert TRAINING.value == previous_mode
        np.testing.assert_array_equal(model.read_buffer('bn.running_mean'), [0,0])
        np.testing.assert_array_equal(model.read_buffer('bn.num_batches_tracked'), [0])
        model.set_optimizer('sgd',lr=0.01)
        x = np.array([[1,2],[3,6]],dtype=np.float32)
        model.train_step(x=x,y=np.zeros_like(x))
        np.testing.assert_allclose(model.read_buffer('bn.running_mean'),[0.2,0.4],rtol=1e-6)
        np.testing.assert_allclose(model.read_buffer('bn.running_var'),[1.1,1.7],rtol=1e-6)
        np.testing.assert_array_equal(model.read_buffer('bn.num_batches_tracked'),[1])
        np.testing.assert_array_equal(net.bn.running_mean.numpy(), [0,0])
        np.testing.assert_array_equal(net.bn.num_batches_tracked.numpy(), 0)
        prediction = model.forward(x=x)['prediction']
        expected = ((x-[0.2,0.4])/np.sqrt(np.array([1.1,1.7])+1e-5)*
                    model.read_buffer('bn.weight')+model.read_buffer('bn.bias'))
        np.testing.assert_allclose(prediction,expected,rtol=1e-5)
        np.testing.assert_array_equal(model.read_buffer('bn.num_batches_tracked'),[1])
        restored = Model.load(model.save())
        restored.set_optimizer('sgd',lr=0.01)
        np.testing.assert_allclose(model.train_step(x=x,y=x),restored.train_step(x=x,y=x),rtol=1e-5)
        np.testing.assert_array_equal(restored.read_buffer('bn.num_batches_tracked'),[2])
    finally:
        if restored: restored.dispose()
        model.dispose()


def test_callable_model_rng_capture_is_private_and_resumable():
    from polygrad.helpers import TRAINING
    Tensor.manual_seed(123)
    control = Tensor.rand(16).numpy()
    Tensor.manual_seed(123)
    weight = Tensor([1.0])
    model = Model(lambda x: (x*weight).dropout(0.5), inputs={'x':Tensor.empty(16)},
                  targets={'y':Tensor.empty(16)}, params={'weight':weight},
                  loss=lambda out,y:(out-y).square().mean())
    restored = None
    try:
        np.testing.assert_array_equal(Tensor.rand(16).numpy(),control)
        assert not TRAINING.value
        names = [model.buf_name(i) for i in range(model.buf_count)]
        counters = [name for name in names if name.startswith('__rng.') and name.endswith('.counter')]
        assert len(counters) == 1
        counter = counters[0]
        np.testing.assert_array_equal(model.read_buffer(counter),[0,0])
        model.set_optimizer('sgd',lr=0.01)
        x = np.ones(16,dtype=np.float32)
        y = np.zeros(16,dtype=np.float32)
        model.train_step(x=x,y=y)
        after = model.read_buffer(counter)
        assert np.any(after != 0)
        model.forward(x=x)
        np.testing.assert_array_equal(model.read_buffer(counter),after)
        restored = Model.load(model.save())
        restored.set_optimizer('sgd',lr=0.01)
        for _ in range(2):
            np.testing.assert_allclose(model.train_step(x=x,y=y),restored.train_step(x=x,y=y),rtol=1e-6)
            np.testing.assert_array_equal(model.read_buffer(counter),restored.read_buffer(counter))
            np.testing.assert_array_equal(model.read_buffer('weight'),restored.read_buffer('weight'))
    finally:
        if restored: restored.dispose()
        model.dispose()


def test_callable_model_capture_rejects_effectful_reads_and_restores_modes():
    from polygrad.helpers import TRAINING
    from polygrad import create
    rt = create(device='interp')
    state = rt.Tensor([0.0]).is_param_(False)
    x = rt.Tensor.empty(1)
    before = TRAINING.value
    try:
        for read in (False, True):
            def author(x):
                state.assign(state+1)
                if read: state.numpy()
                else: raise RuntimeError('author failed')
                return x
            with pytest.raises((RuntimeError, ValueError)):
                Model(author,inputs={'x':x},params={'state':state})
            assert TRAINING.value == before
            np.testing.assert_array_equal(state.numpy(),[0])
        def pure(x):
            rt.Tensor([3.0]).numpy()
            return x+1
        model = Model(pure,inputs={'x':x})
        try: np.testing.assert_array_equal(model.forward(x=np.array([2],dtype=np.float32))['output'],[3])
        finally: model.dispose()
    finally:
        rt.dispose()


def test_callable_model_capture_keeps_sequential_aux_updates():
    from polygrad.helpers import TRAINING
    state = Tensor([0.0]).is_param_(False)
    weight = Tensor([1.0])
    def author(x):
        if TRAINING.value:
            state.assign(state+1)
            state.assign(state+2)
        return x*weight
    model = Model(author,inputs={'x':Tensor.empty(1)}, targets={'y':Tensor.empty(1)},
                  params={'state':state,'weight':weight},loss=lambda out,y:(out-y).square().mean())
    try:
        model.set_optimizer('sgd',lr=0.01)
        for i in (1,2):
            model.train_step(x=np.ones(1,dtype=np.float32),y=np.zeros(1,dtype=np.float32))
            np.testing.assert_array_equal(model.read_buffer('state'),[3*i])
        np.testing.assert_array_equal(state.numpy(),[0])
    finally: model.dispose()


def test_callable_model_lazy_parameter_rng_is_not_an_inference_effect():
    class Net:
        def __init__(self): self.weight = None
        def __call__(self,x):
            if self.weight is None: self.weight = Tensor.rand(2)
            return x*self.weight
    model = Model(Net(),inputs={'x':Tensor.empty(2)},targets={'y':Tensor.empty(2)},
                  loss=lambda out,y:(out-y).square().mean())
    try:
        counters = [model.buf_name(i) for i in range(model.buf_count)
                    if model.buf_name(i).startswith('__rng.') and model.buf_name(i).endswith('.counter')]
        before = {name:model.read_buffer(name) for name in counters}
        for _ in range(2):
            model.forward(x=np.ones(2,dtype=np.float32))
            for name in counters: np.testing.assert_array_equal(model.read_buffer(name),before[name])
    finally: model.dispose()


@pytest.mark.parametrize('device', ['cpu', 'interp', 'cuda'])
def test_dynamic_model_results_and_portable_signature(device):
    from polygrad import create
    from polygrad import _ffi
    from polygrad.tensor import Variable
    if device == 'cuda' and (not hasattr(_ffi._lib, 'poly_cuda_available') or not _ffi._lib.poly_cuda_available()):
        pytest.skip('poly_cuda_available() is false in the selected library')
    rt = create(device=device, logical='always')
    n = Variable('model_batch', 1, 32, _ctx=rt._ctx)
    model = Model(lambda x,y: {'prediction':x*2+y},
                  inputs={'x':rt.Tensor.empty(n.bind(17),2), 'y':rt.Tensor.empty(n.bind(17),2)})
    restored = None
    try:
        a = np.arange(34,dtype=np.float32).reshape(17,2)
        first = model.forward(x=rt.Tensor(a), y=a)['prediction']
        assert first.shape == (17,2)
        np.testing.assert_array_equal(first.numpy(), a*3)
        small = np.arange(6,dtype=np.float32).reshape(3,2)
        np.testing.assert_array_equal(model.forward(x=small,y=small)['prediction'], small*3)
        before = model.read_buffer('x')
        with pytest.raises(RuntimeError): model.forward(x=a,y=small)
        np.testing.assert_array_equal(model.read_buffer('x'), before)
        for bad in (np.zeros((33,2),dtype=np.float32), np.zeros((2,3),dtype=np.float32)):
            with pytest.raises(RuntimeError): model.forward(x=bad,y=bad)
            np.testing.assert_array_equal(model.read_buffer('x'), before)
        assert model.buf_shape_bounds(model.find_buf('x')) == ((1,32),(2,2))
        restored = Model.load(model.save(include_optimizer=False), runtime=rt)
        np.testing.assert_array_equal(restored.forward(x=small.ravel(),y=small)['prediction'], small*3)
        model.dispose()
        rt.clear_schedule_cache()
        rt.collect()
        assert first.shape == (17, 2)
        np.testing.assert_array_equal(first.numpy(), a*3)
        rt.dispose()
        with pytest.raises(RuntimeError, match='disposed'):
            first.numpy()
    finally:
        if restored is not None: restored.dispose()
        model.dispose()
        rt.dispose()


def test_empty_model_binding_rejects_before_any_input_write():
    from polygrad import create
    rt = create(device='interp', logical='always')
    n = rt.Variable('empty_model_batch', 0, 4)
    model = Model(lambda offset, x: x + offset,
                  inputs={'offset': rt.Tensor.empty(1), 'x': rt.Tensor.empty(n.bind(3), 2)})
    try:
        model.forward(offset=np.array([7], np.float32), x=np.ones((3, 2), np.float32))
        before = model.read_buffer('offset')
        with pytest.raises(RuntimeError):
            model.forward(offset=np.array([99], np.float32), x=np.empty((0, 2), np.float32))
        np.testing.assert_array_equal(model.read_buffer('offset'), before)
    finally:
        model.dispose()
        rt.dispose()


def test_dynamic_model_import_matches_direct_cpu_construction():
    from polygrad import create
    from polygrad.tensor import Variable
    author = create(device='interp', logical='always')
    runtime = create(device='cpu', logical='always')
    imported_runtime = create(device='cpu', logical='always')
    models = []
    try:
        for rt in (author, runtime):
            n = Variable('import_batch', 1, 32, _ctx=rt._ctx)
            models.append(Model(lambda x: {'prediction': x*2},
                                inputs={'x': rt.Tensor.empty(n.bind(17), 2)}))
        models.append(Model.load(models[0].save(include_optimizer=False), runtime=imported_runtime))
        for size in (17, 3, 11):
            x = np.arange(size*2, dtype=np.float32).reshape(size, 2)
            direct = models[1].forward(x=x)['prediction']
            restored = models[2].forward(x=x)['prediction']
            np.testing.assert_array_equal(direct, x*2)
            np.testing.assert_array_equal(restored, direct)
    finally:
        for model in reversed(models): model.dispose()
        imported_runtime.dispose()
        runtime.dispose()
        author.dispose()


def _canonical_model_ir(ir):
    """Normalize allocation numbering in a fresh namespace, not semantic slots.

    This deliberately reuses the codec: it is a round-trip oracle, not an
    independent proof of relocation. Direct execution and mutation isolation
    below, plus test_ir.c's exact slot/alias assertions, supply that evidence.
    """
    import ctypes
    from polygrad import _ffi
    data = (ctypes.c_uint8 * len(ir)).from_buffer_copy(ir)
    model = Model._from_handle(_ffi._lib.poly_model_from_ir(data, len(ir), None, 0))
    try:
        return model.export_ir()
    finally:
        model.dispose()


def _import_equivalence_model(rt, kind):
    if kind == 'variable':
        n = rt.Variable('import_equivalence_batch', 1, 32)
        return Model(lambda x: {'prediction': x*2},
                     inputs={'x': rt.Tensor.empty(n.bind(17), 2)})
    weight = rt.Tensor([2.0])
    aux = rt.Tensor([1.0]).is_param_(False)
    if kind == 'stateful':
        from polygrad.helpers import TRAINING
        def author(x):
            if TRAINING.value:
                aux.assign(aux+1)
            return {'prediction': (x*weight).dropout(0.5)}
        return Model(author, inputs={'x': rt.Tensor.empty(16)},
                     targets={'y': rt.Tensor.empty(16)},
                     params={'weight': weight, 'alias': weight, 'counter': aux},
                     loss=lambda prediction, y: (prediction['prediction']-y).square().mean())
    def author(x):
        pred = x*weight+aux
        return {'prediction': pred, 'twice': pred*2}
    return Model(author, inputs={'x': rt.Tensor.empty(1)},
                 params={'weight': weight, 'alias': weight, 'offset': aux}, entrypoints=[
                     {'name': 'forward', 'inputs': ['x'], 'outputs': ['prediction']},
                     {'name': 'double', 'inputs': ['x'], 'outputs': ['twice']}])


@pytest.mark.parametrize('kind', ['variable', 'aliases_entrypoints', 'stateful'])
def test_import_equivalence_isolation_and_roundtrip(kind, capfd, monkeypatch):
    import ctypes
    import re
    from polygrad import create, _ffi
    from polygrad.model import ROLE_PARAM, ROLE_AUX
    monkeypatch.setenv('POLY_DUMP_KERNELS', '1')
    # RNG resource names include the authoring device. Compare direct captures
    # on that same device; other cases also exercise INTERP -> CPU placement.
    author = create(device='cpu' if kind == 'stateful' else 'interp', logical='always')
    direct_rt = create(device='cpu', logical='always')
    fresh_rt = create(device='cpu', logical='always')
    shared_rt = create(device='cpu', logical='always')
    models = []
    try:
        source = _import_equivalence_model(author, kind)
        models.append(source)
        ir, weights, bundle = source.export_ir(), source.export_weights(), source.save()
        canonical = _canonical_model_ir(ir)
        assert canonical == _canonical_model_ir(canonical)
        direct = _import_equivalence_model(direct_rt, kind)
        models.append(direct)
        # Initialization (including the complete RNG state) must match; fresh
        # namespace allocation is intentionally allowed to differ.
        if weights is not None:
            direct.import_weights(weights)
        live = shared_rt.Tensor([19.0])
        primer = Model(lambda x: x+37, inputs={'x': shared_rt.Tensor.empty(7)})
        models.append(primer)
        np.testing.assert_array_equal(primer.forward(x=np.arange(7, dtype=np.float32))['output'],
                                      np.arange(7, dtype=np.float32)+37)
        data = (ctypes.c_uint8 * len(bundle)).from_buffer_copy(bundle)
        private = Model._from_handle(_ffi._lib.poly_model_from_bundle(data, len(bundle)))
        models.append(private)
        private.place('cpu')
        fresh = Model.load(bundle, runtime=fresh_rt)
        models.append(fresh)
        shared = Model.load(bundle, runtime=shared_rt)
        models.append(shared)
        sibling = Model.load(bundle, runtime=shared_rt)
        models.append(sibling)
        lanes = [direct, private, fresh, shared]
        for model in lanes + [sibling]:
            assert model.entrypoints() == source.entrypoints()
            assert _canonical_model_ir(model.export_ir()) == canonical
            for i in range(source.buf_count):
                name = source.buf_name(i)
                j = model.find_buf(name)
                assert j >= 0
                assert model.buf_shape_bounds(j) == source.buf_shape_bounds(i)
                assert model.buf_role(j) == source.buf_role(i)
            if kind == 'stateful':
                model.set_optimizer('adam', lr=0.01)

        def state(model):
            return {model.buf_name(i): model.read_buffer(model.buf_name(i))
                    for i in range(model.buf_count) if model.buf_role(i) in (ROLE_PARAM, ROLE_AUX)}

        untouched = state(sibling)

        def execute(model):
            if kind == 'variable':
                return [model.forward(x=np.arange(n*2, dtype=np.float32).reshape(n, 2))['prediction']
                        for n in (17, 3, 11)]
            if kind == 'aliases_entrypoints':
                return [model.call(ep, x=[3])[out] for ep, out in
                        [('forward', 'prediction'), ('double', 'twice'), ('forward', 'prediction')]]
            io = {'x': np.ones(16, np.float32), 'y': np.zeros(16, np.float32)}
            return [model.forward(x=io['x'])['prediction'],
                    np.asarray(model.train_step(**io)), np.asarray(model.train_step(**io)),
                    model.forward(x=io['x'])['prediction']]

        results, kernels, states = [], [], []
        for model in lanes:
            capfd.readouterr()
            results.append(execute(model))
            # Compare full source, including function-name hash and argument
            # signature. A multiset would confuse cache reuse with topology;
            # these cold lanes must each emit the same nonempty kernel set.
            stderr = capfd.readouterr().err
            blocks = re.findall(r'^=== KERNEL (\S+) ===\n(.*?)\n=== END ===$', stderr, re.M | re.S)
            assert blocks, f'{kind}: no compiled CPU kernel evidence'
            kernels.append(set(blocks))
            states.append(state(model))
        for i in range(1, len(lanes)):
            assert kernels[i] == kernels[0], f'{kind}: kernel signature/hash/source changed on import'
            for expected, actual in zip(results[0], results[i]):
                np.testing.assert_array_equal(actual, expected)
            assert states[i].keys() == states[0].keys()
            for name, expected in states[0].items():
                np.testing.assert_array_equal(states[i][name], expected)
        for name, expected in untouched.items():
            np.testing.assert_array_equal(sibling.read_buffer(name), expected)
        if kind != 'variable':
            shared.write_buffer('alias', np.array([71], np.float32))
            np.testing.assert_array_equal(shared.read_buffer('weight'), [71])
            np.testing.assert_array_equal(sibling.read_buffer('weight'), untouched['weight'])
        shared.dispose()
        shared_rt.clear_schedule_cache()
        shared_rt.collect()
        np.testing.assert_array_equal(live.numpy(), [19])
        for name, expected in untouched.items():
            np.testing.assert_array_equal(sibling.read_buffer(name), expected)
        execute(sibling)
    finally:
        for model in reversed(models):
            model.dispose()
        for rt in (shared_rt, fresh_rt, direct_rt, author):
            rt.dispose()


@pytest.mark.parametrize('device', ['cpu', 'interp', 'cuda'])
def test_dynamic_model_training_rebinds_cached_loss(device):
    from polygrad import create
    from polygrad import _ffi
    from polygrad.tensor import Variable
    if device == 'cuda' and (not hasattr(_ffi._lib, 'poly_cuda_available') or not _ffi._lib.poly_cuda_available()):
        pytest.skip('poly_cuda_available() is false in the selected library')
    rt = create(device=device, logical='always')
    n = Variable('train_batch',1,32,_ctx=rt._ctx)
    w = rt.Tensor([0.0])
    model = Model(lambda x: x*w, inputs={'x':rt.Tensor.empty(n.bind(17),1)},
                  targets={'y':rt.Tensor.empty(n.bind(17),1)}, params={'w':w},
                  loss=lambda out,y:(out-y).square().mean())
    try:
        model.set_optimizer('sgd',lr=0.01)
        expected = 0.0
        for size in (17,3,11):
            x = np.arange(1,size+1,dtype=np.float32).reshape(size,1)
            expected_loss = np.mean(((expected-2)*x)**2)
            loss = model.train_step(x=x,y=x*2)
            np.testing.assert_allclose(loss, expected_loss, rtol=1e-5)
            expected -= 0.02*(expected-2)*np.mean(x*x)
            np.testing.assert_allclose(model.read_buffer('w'),[expected],rtol=1e-5)
    finally:
        model.dispose(); rt.dispose()


@pytest.mark.parametrize('device', ['cpu', 'interp', 'cuda'])
@pytest.mark.parametrize('dataset', ['host', 'tensor', 'mixed'])
def test_fit_bounded_minibatches_match_steps(device, dataset, monkeypatch):
    from polygrad import create, _ffi
    if device == 'cuda' and (not hasattr(_ffi._lib, 'poly_cuda_available') or not _ffi._lib.poly_cuda_available()):
        pytest.skip('poly_cuda_available() is false in the selected library')
    rt = create(device=device, logical='always')
    n = rt.Variable('fit_batch', 1, 8)
    def build():
        w = rt.Tensor([0.0])
        return Model(lambda x: x*w, inputs={'x':rt.Tensor.empty(n.bind(4), 2)},
                     targets={'y':rt.Tensor.empty(n.bind(4), 2)}, params={'w':w},
                     loss=lambda out,y:(out-y).square().mean())
    model, control = build(), build()
    x = np.arange(14, dtype=np.float32).reshape(2, 7).T / 10
    tx, ty = rt.Tensor(x.T).transpose(), rt.Tensor(x*2)
    try:
        control.set_optimizer('sgd', lr=0.01)
        expected = [control.train_step(x=x[i:i+3], y=x[i:i+3]*2)
                    for _ in range(2) for i in range(0, 7, 3)]
        data = {'x':x if dataset == 'host' else tx, 'y':ty if dataset == 'tensor' else x*2}
        # Dataset slicing must not materialize through frontend host reads.
        with monkeypatch.context() as patch:
            patch.setattr(Tensor, 'numpy', lambda *a, **k: pytest.fail('Tensor dataset read back to host'))
            losses = model.fit(data, batch_size=3, remainder='keep', epochs=2, optimizer='sgd', lr=0.01)
        np.testing.assert_allclose(losses, expected, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(model.read_buffer('w'), control.read_buffer('w'), rtol=1e-5)
        np.testing.assert_allclose(tx.numpy(), x)
        # A complete one-batch shrink may return its caller: fit must not dispose it.
        small = rt.Tensor(x[:2])
        try:
            assert len(model.fit({'x':small, 'y':x[:2]*2}, batch_size=3, remainder='keep')) == 1
            np.testing.assert_array_equal(small.numpy(), x[:2])
        finally: small.dispose()
        before = model.read_buffer('w')
        for size, remainder in ((9, 'keep'), (3, 'error')):
            with pytest.raises(ValueError): model.fit(data, batch_size=size, remainder=remainder, optimizer='adam')
            np.testing.assert_array_equal(model.read_buffer('w'), before)
    finally:
        tx.dispose(); ty.dispose(); model.dispose(); control.dispose(); rt.dispose()


@pytest.mark.parametrize('dynamic', [False, True])
def test_fit_dataset_preflight_and_slice_cleanup(dynamic, monkeypatch):
    from polygrad import create
    rt, other = create(device='interp', logical='always'), create(device='interp')
    batch = rt.Variable('fit_minimum', 2, 4).bind(3) if dynamic else 3
    w = rt.Tensor([0.0])
    model = Model(lambda x: x*w, inputs={'x':rt.Tensor.empty(batch, 2)},
                  targets={'y':rt.Tensor.empty(batch, 2)}, params={'w':w},
                  loss=lambda out,y:(out-y).square().mean())
    x = np.arange(12,dtype=np.float32).reshape(6,2)/10
    tx, ty, wrong = rt.Tensor(x), rt.Tensor(x*2), other.Tensor(x)
    disposed = rt.Tensor(x)
    disposed.dispose()
    try:
        with monkeypatch.context() as patch:
            patch.setattr(model, 'set_optimizer', lambda *a, **k: pytest.fail('invalid dataset configured optimizer'))
            for data, opts in [({'x':x[:4], 'y':x[:4]}, {'remainder':'keep'}),
                               ({'x':x[:0], 'y':x[:0]}, {'remainder':'keep'}),
                               ({'x':tx, 'y':x.astype(np.float64)}, {}),
                               ({'x':wrong, 'y':ty}, {}), ({'x':disposed, 'y':ty}, {}),
                               ({'x':x, 'y':x.reshape(3,4)}, {})]:
                with pytest.raises((TypeError, ValueError)):
                    model.fit(data, batch_size=3, optimizer='adam', **opts)
        model.set_optimizer('sgd', lr=0.01)
        # Partial batch construction must release already-created views.
        shrink, created = Tensor.shrink, []
        def fail_second(self, arg):
            if created: raise RuntimeError('slice sentinel')
            result = shrink(self, arg)
            created.append(result)
            return result
        with monkeypatch.context() as patch:
            patch.setattr(Tensor, 'shrink', fail_second)
            with pytest.raises(RuntimeError, match='slice sentinel'):
                model.fit({'x':tx,'y':ty}, batch_size=3)
        assert created and created[0]._tensor is None
        def fail_callback(*args): raise RuntimeError('callback sentinel')
        with pytest.raises(RuntimeError, match='callback sentinel'):
            model.fit({'x':tx,'y':ty}, batch_size=3, on_step=fail_callback)
        assert len(model.fit({'x':tx,'y':ty}, batch_size=3)) == 2
        np.testing.assert_array_equal(tx.numpy(), x)
    finally:
        tx.dispose(); ty.dispose(); wrong.dispose(); model.dispose(); other.dispose(); rt.dispose()


def test_host_io_shape_admission_precedes_writes():
    from polygrad import create
    rt = create(device='interp')
    model = Model(lambda x, y: x+y,
                  inputs={'x': rt.Tensor.empty(2, 3), 'y': rt.Tensor.empty(2, 3)})
    try:
        flat = np.arange(6, dtype=np.float32)
        np.testing.assert_array_equal(model.forward(x=flat, y=flat)['output'], (flat*2).reshape(2, 3))
        before = model.read_buffer('x')
        with pytest.raises(RuntimeError):
            model.forward(x=flat+10, y=flat.reshape(3, 2))
        np.testing.assert_array_equal(model.read_buffer('x'), before)
        with pytest.raises(TypeError, match='native byte order'):
            model.forward(x=flat+10, y=flat.astype('>f4'))
        np.testing.assert_array_equal(model.read_buffer('x'), before)
    finally:
        model.dispose()
        rt.dispose()


def test_fit_minibatches_match_explicit_steps_and_reject_remainder():
    from polygrad import create
    rt = create(device='interp')
    def build():
        w = rt.Tensor([0.0])
        return Model(lambda x: x*w, inputs={'x': rt.Tensor.empty(2, 1)},
                     targets={'y': rt.Tensor.empty(2, 1)}, params={'w': w},
                     loss=lambda out, y: (out-y).square().mean())
    model, control = build(), build()
    try:
        x = np.arange(1, 7, dtype=np.float32).reshape(6, 1)
        data = {'x': x, 'y': x*2}
        observed = []
        losses = model.fit(data, batch_size=2, epochs=2, optimizer='sgd', lr=0.01,
                           on_step=lambda step, loss: observed.append((step, loss)))
        control.set_optimizer('sgd', lr=0.01)
        expected = [control.train_step({name: values[i:i+2] for name, values in data.items()})
                    for _ in range(2) for i in range(0, 6, 2)]
        np.testing.assert_allclose(losses, expected)
        np.testing.assert_array_equal(model.read_buffer('w'), control.read_buffer('w'))
        assert [step for step, _ in observed] == list(range(6))
        before = model.read_buffer('w')
        with pytest.raises(ValueError, match='remainder'):
            model.fit({'x': x[:5], 'y': x[:5]*2}, batch_size=2, optimizer='adam')
        np.testing.assert_array_equal(model.read_buffer('w'), before)
        with pytest.raises(ValueError, match='epochs'):
            model.fit(data, epochs=1.5, batch_size=2)
        assert len(model.fit({'x': x[:5], 'y': x[:5]*2}, batch_size=2, remainder='drop')) == 2
        for bad in ({'x': x, 'y': x[:4]}, {'x': x, 'y': x.astype(np.float64)},
                    {'x': x, 'y': x.reshape(3, 2)}):
            before = model.read_buffer('w')
            with pytest.raises((ValueError, TypeError)):
                model.fit(bad, batch_size=2)
            np.testing.assert_array_equal(model.read_buffer('w'), before)
    finally:
        model.dispose()
        control.dispose()
        rt.dispose()


@pytest.mark.parametrize('device', ['cpu', 'interp', 'cuda'])
def test_tensor_io_owns_results_and_validates_before_writes(device):
    from polygrad import create
    from polygrad import _ffi
    if device == 'cuda' and (not hasattr(_ffi._lib, 'poly_cuda_available') or not _ffi._lib.poly_cuda_available()):
        pytest.skip('poly_cuda_available() is false in the selected library')
    rt = create(device=device)
    other = create(device=device)
    model = None
    try:
        model = Model(lambda x, y: {'prediction': x*2+y},
                      inputs={'x': rt.Tensor.empty(2, 2), 'y': rt.Tensor.empty(2, 2)})
        x = rt.Tensor([[1., 2.], [3., 4.]]).transpose()
        first = model.forward(x=x, y=np.ones((2, 2), dtype=np.float32))['prediction']
        assert isinstance(first, Tensor)
        second = model.forward(x=first, y=rt.Tensor.zeros(2, 2))['prediction']
        np.testing.assert_array_equal(first.numpy(), [[3, 7], [5, 9]])
        np.testing.assert_array_equal(second.numpy(), [[6, 14], [10, 18]])
        before = model.read_buffer('x')
        for bad in (other.Tensor.zeros(2, 2), rt.Tensor.zeros(4),
                    rt.Tensor.zeros(2, 2, dtype='int32')):
            with pytest.raises((ValueError, RuntimeError)):
                model.forward(x=rt.Tensor.ones(2, 2), y=bad)
            np.testing.assert_array_equal(model.read_buffer('x'), before)
        # Evaluating a pending assign is ordinary Tensor realization, not an
        # effect to replay on each Model invocation or later input readback.
        effect = rt.Tensor.ones(2, 2).contiguous().realize()
        effect.assign(effect+1)
        once = model.forward(x=effect, y=np.zeros((2, 2), dtype=np.float32))['prediction']
        twice = model.forward(x=effect, y=np.zeros((2, 2), dtype=np.float32))['prediction']
        np.testing.assert_array_equal(once.numpy(), [[4, 4], [4, 4]])
        np.testing.assert_array_equal(twice.numpy(), [[4, 4], [4, 4]])
        np.testing.assert_array_equal(effect.numpy(), [[2, 2], [2, 2]])
        model.dispose()
        rt.clear_schedule_cache()
        rt.collect()
        np.testing.assert_array_equal(first.numpy(), [[3, 7], [5, 9]])
        np.testing.assert_array_equal((second+1).numpy(), [[7, 15], [11, 19]])
    finally:
        if model is not None: model.dispose()
        other.dispose()
        rt.dispose()


@pytest.mark.parametrize('dtype', ['float32', 'float16', 'bfloat16', 'int32', 'int64', 'bool'])
def test_tensor_io_preserves_storage_dtype(dtype):
    from polygrad import create
    rt = create(device='interp')
    model = None
    try:
        x = rt.Tensor([0, 1, 1, 0], dtype=dtype)
        model = Model(lambda x: x.reshape(2, 2), inputs={'x': rt.Tensor.empty(4, dtype=dtype)})
        output = model.forward(x=x)['output']
        assert output.dtype == x.dtype
        assert output.shape == (2, 2)
        assert output.is_param is False
        model.dispose()
        np.testing.assert_array_equal(output.float().numpy(), [[0, 1], [1, 0]])
    finally:
        if model is not None: model.dispose()
        rt.dispose()


def test_bundle_load_uses_default_runtime_context():
    import polygrad as pg
    source = Model(sequential_definition())
    loaded = None
    try:
        loaded = Model.load(source.save(include_optimizer=False))
        assert loaded._ctx == pg._default_ctx
    finally:
        if loaded is not None: loaded.dispose()
        source.dispose()


@pytest.mark.parametrize('device', ['cpu', 'interp'])
def test_runtime_import_isolation_aliases_and_failed_load(device):
    from polygrad import create
    rt = create(device=device, logical='always')
    models = []
    try:
        x, w = rt.Tensor.empty(1), rt.Tensor([2.0])
        live = rt.Tensor([19.0])
        source = Model(lambda x: x*w, inputs={'x': x}, params={'w': w, 'alias': w})
        models.append(source)
        blob = source.save(include_optimizer=False)
        a, b = Model.load(blob, runtime=rt), Model.from_bundle(blob, runtime=rt)
        models.extend([a, b])
        assert a._ctx == b._ctx == rt._ctx
        a.write_buffer('alias', np.array([7], dtype=np.float32))
        np.testing.assert_array_equal(a.read_buffer('w'), [7])
        np.testing.assert_array_equal(b.forward(x=[3])['output'], [6])
        np.testing.assert_array_equal(source.forward(x=[3])['output'], [6])
        # A different artifact, also numbered in its own original namespace.
        other = MLP(layers=[1, 1], bias=False, batch_size=1, seed=1)
        try: other_blob = other.save(include_optimizer=False)
        finally: other.dispose()
        c = Model.load(other_blob, runtime=rt)
        models.append(c)
        assert c._ctx == rt._ctx
        for n in (0, 31, len(blob)//2, len(blob)-1):
            with pytest.raises(RuntimeError): Model.load(blob[:n], runtime=rt)
        with pytest.raises(RuntimeError):
            Model.from_ir(source.export_ir(), b'bad weights', runtime=rt)
        a.dispose()
        rt.collect()
        np.testing.assert_array_equal(b.forward(x=[3])['output'], [6])
        np.testing.assert_array_equal(live.numpy(), [19])
        rt.dispose()
        assert all(m._ptr is None for m in models)
        with pytest.raises(RuntimeError, match='disposed'): Model.load(blob, runtime=rt)
    finally:
        for model in models: model.dispose()
        rt.dispose()


@pytest.mark.parametrize('device', ['cpu', 'interp'])
def test_runtime_import_disposal_reclaims_storage(device):
    from polygrad import create
    with create(device=device) as rt:
        weight = rt.Tensor([2.0])
        source = rt.Model(lambda x: x*weight, inputs={'x': rt.Tensor.empty(2)},
                          params={'weight': weight, 'alias': weight})
        try:
            artifact = source.save(include_optimizer=False)
            live = rt.Tensor([19.0]).realize()
            retained = []
            for _ in range(12):
                loaded = rt.Model.load(artifact)
                try:
                    np.testing.assert_array_equal(loaded.forward(x=[3, 4])['output'], [6, 8])
                    loaded.write_buffer('alias', np.array([7.0], np.float32))
                    np.testing.assert_array_equal(loaded.read_buffer('weight'), [7])
                finally:
                    loaded.dispose()
                # Schedule-cache retention is documented and explicitly released.
                # Count live storage/owners, not arena high-water or cumulative IO.
                rt.clear_schedule_cache()
                rt.collect()
                stats = rt.stats()
                retained.append(tuple(stats[key] for key in
                                      ('buffer_owned_bytes', 'buffer_entries', 'tensor_records')))
            assert len(set(retained)) == 1, retained
            np.testing.assert_array_equal(live.numpy(), [19])
            np.testing.assert_array_equal(source.forward(x=[3, 4])['output'], [6, 8])
        finally:
            source.dispose()


def safetensor_names(data):
    header_len = int.from_bytes(data[:8], 'little')
    header = data[8:8 + header_len].decode('utf-8')
    import json
    return set(json.loads(header).keys()) - {'__metadata__'}


def definition_fixture():
    import json
    from pathlib import Path
    return json.loads((Path(__file__).resolve().parents[2] /
                       'test/fixtures/model_definition.json').read_text())


def sequential_definition():
    return {'format': 'poly.modeldef@1', 'type': 'sequential', 'seed': 42,
            'input': {'name': 'x', 'shape': [1, 2], 'dtype': 'float32'},
            'layers': [{'name': 'stack', 'type': 'repeat', 'count': 2,
                        'body': {'type': 'linear', 'out_features': 2,
                                 'activation': 'relu'}}],
            'output': 'prediction'}


class TestModelDefinition:
    def test_usability_summary_and_path_roundtrip(self, tmp_path, monkeypatch):
        weight = Tensor([2.0])
        model = Model.from_tensors(params={'weight': weight}, outputs={'prediction': weight + 1})
        restored = None
        try:
            def forbid(*args, **kwargs): raise AssertionError('metadata executed or read state')
            with monkeypatch.context() as m:
                m.setattr(model, 'read_buffer', forbid)
                m.setattr(model, 'call', forbid)
                text = model.summary()
                assert 'weight' in text and 'PARAM' in text and 'float32[1]' in text
                assert 'forward' in text and 'prediction' in text
            path = tmp_path / 'model.pgb'
            blob = model.save(path, include_optimizer=False)
            assert path.read_bytes() == blob
            restored = Model.load(path)
            assert restored.summary() == model.summary()
            model.dispose()
            with pytest.raises(RuntimeError, match='disposed'):
                model.save()
            with pytest.raises(RuntimeError, match='disposed'):
                model.summary()
        finally:
            model.dispose()
            if restored is not None: restored.dispose()

    def test_usability_rejects_async_author_and_loss_before_call(self):
        calls = []
        x = Tensor.empty(1)
        async def author(x):
            calls.append('author')
            return x
        async def loss(out): return out.mean()
        def forward(x):
            calls.append('forward')
            return x
        for fn, objective in ((author, None), (forward, loss)):
            with pytest.raises(TypeError, match='synchronous'):
                Model.from_callable(fn, inputs={'x': x}, loss=objective)
        assert calls == []

    def test_usability_mixed_runtime_and_role_conflicts_do_not_invoke_author(self):
        from polygrad import create
        rt = create(device='interp')
        calls = []
        x = Tensor.empty(1)
        try:
            with pytest.raises(ValueError, match='PolyCtx'):
                Model.from_callable(lambda x: calls.append(x), inputs={'x': x}, params={'w': rt.Tensor([2.0])})
            with pytest.raises(ValueError, match='input/target'):
                Model.from_callable(lambda x: calls.append(x), inputs={'x': x}, params={'w': x})
            assert calls == []
        finally:
            rt.dispose()

    @pytest.mark.parametrize('factory,spec', [(Sequential, sequential_definition), (Graph, definition_fixture)])
    def test_constructor_dispatches_tagged_config_to_existing_factory(self, factory, spec):
        automatic = explicit = None
        try:
            automatic, explicit = Model(spec()), factory(spec())
            assert automatic.bindings() == explicit.bindings()
            assert automatic.entrypoints() == explicit.entrypoints()
            actual = automatic.forward(x=np.array([[1, 2]], np.float32))
            expected = explicit.forward(x=np.array([[1, 2]], np.float32))
            for name in actual:
                np.testing.assert_array_equal(actual[name], expected[name])
        finally:
            if automatic is not None: automatic.dispose()
            if explicit is not None: explicit.dispose()

    def test_explicit_callable_factory_and_constructor_share_capture(self):
        x, weight = Tensor.empty(1), Tensor([2.0])
        class Net:
            def __call__(self, x): return x + 99
            def forward(self, x): return x * weight
        net = Net()
        automatic = explicit = None
        try:
            automatic = Model(net, inputs={'x': x}, params={})
            explicit = Model.from_callable(net.forward, inputs={'x': x}, params={'weight': weight})
            np.testing.assert_array_equal(automatic.forward(x=[3])['output'], [102])
            np.testing.assert_array_equal(explicit.forward(x=[3])['output'], [6])
        finally:
            if automatic is not None: automatic.dispose()
            if explicit is not None: explicit.dispose()

    @pytest.mark.parametrize('spec', [{'layers': [1, 2]}, {'nodes': []},
                                    {'format': 'poly.modeldef@2', 'type': 'graph'},
                                    {'format': 'poly.modeldef@1', 'type': 'unknown'}])
    def test_constructor_does_not_guess_configuration(self, spec):
        with pytest.raises((TypeError, ValueError), match='format|configuration|unknown model type'):
            Model(spec)

    def test_constructor_rejects_mixed_configuration_and_tensor_bindings(self):
        with pytest.raises(TypeError, match='combined'):
            Model(sequential_definition(), outputs=Tensor([1.0]))

    def test_configuration_constructor_respects_runtime_ownership(self):
        from polygrad import create
        rt = create(device='interp', logical='always')
        model = Model(sequential_definition(), runtime=rt)
        assert model._ctx == rt._ctx
        rt.dispose()
        assert model._ptr is None
        model.dispose()

    def test_callable_rejects_missing_logical_source_before_invocation(self):
        from polygrad import create
        rt = create(device='interp', logical='never')
        calls = []
        try:
            with pytest.raises(ValueError, match='logical source'):
                Model.from_callable(lambda x: calls.append(x), inputs={'x': rt.Tensor.empty(1)})
            assert calls == []
        finally:
            rt.dispose()

    def test_callable_collects_lazily_initialized_attributes(self):
        class Net:
            def __call__(self, x):
                self.weight = Tensor([2.0])
                return x * self.weight
        net = Net()
        model = Model(net, inputs={'x': Tensor.empty(1)})
        try:
            assert {b['name']: b['role'] for b in model.bindings()}['weight'] == 0
            np.testing.assert_array_equal(model.read_buffer('weight'), [2])
        finally:
            model.dispose()

    def test_callable_constructor_collects_roles_and_owns_state(self):
        from polygrad import create
        rt = create(device='interp', logical='until_realize')
        class Net:
            def __init__(self):
                self.weight = rt.Tensor([2.0])
                self.alias = self.weight
                self.offset = rt.Tensor([1.0]).is_param_(False)
            def __call__(self, x):
                return x * self.weight + self.offset
        model = restored = None
        try:
            net = Net()
            model = Model(net, inputs={'x': rt.Tensor.empty(1)},
                          targets={'y': rt.Tensor.empty(1)},
                          loss=lambda out, y: (out-y).square().mean())
            roles = {b['name']: b['role'] for b in model.bindings()}
            assert roles['weight'] == roles['alias'] == 0
            assert roles['offset'] == 4
            model.set_optimizer('sgd', lr=.1)
            assert model.train_step({'x': [1.0], 'y': [0.0]}) == 9
            np.testing.assert_allclose(model.read_buffer('weight'), [1.4])
            np.testing.assert_array_equal(net.weight.numpy(), [2])
            restored = Model.load(model.save(include_optimizer=False))
            model.dispose()
            model.dispose()
            np.testing.assert_allclose(restored.forward(x=[1.0])['output'], [2.4])
        finally:
            if model is not None: model.free()
            if restored is not None: restored.free()
            rt.dispose()

    def test_callable_constructor_override_and_failure_scope(self):
        from polygrad import create, _ffi
        rt = create(device='interp', logical='until_realize')
        class Net:
            def __init__(self): self.weight = rt.Tensor([2.0])
            def __call__(self, x): return x + 1
        model = None
        try:
            net, x = Net(), rt.Tensor.empty(1)
            _ffi.get_lib().poly_ctx_set_logical_policy(x._ctx, 0)
            before = _ffi.get_lib().poly_ctx_get_logical_policy(x._ctx)
            model = Model(net, inputs={'x': x}, params={})
            assert not any(b['role'] in (0, 4) for b in model.bindings())
            assert _ffi.get_lib().poly_ctx_get_logical_policy(x._ctx) == before
            with pytest.raises(TypeError, match='class'):
                Model(Net, inputs={'x': x})
            with pytest.raises(TypeError, match='outputs'):
                Model(net, inputs={'x': x}, outputs=x)
            def broken(x):
                assert _ffi.get_lib().poly_ctx_get_logical_policy(x._ctx) == 1
                raise ValueError('author failed')
            with pytest.raises(ValueError, match='author failed'):
                Model(broken, inputs={'x': x})
            assert _ffi.get_lib().poly_ctx_get_logical_policy(x._ctx) == before
        finally:
            if model is not None: model.free()
            rt.dispose()

    def test_params_object_and_mapping_have_the_same_roles(self):
        from types import SimpleNamespace
        weight, aux = Tensor([2.0]), Tensor([1.0]).is_param_(False)
        for params in (SimpleNamespace(weight=weight, aux=aux), {'weight': weight, 'aux': aux}):
            model = Model(params=params, outputs={'value': weight+aux})
            try:
                roles = {b['name']: b['role'] for b in model.bindings()}
                assert roles['weight'] == 0 and roles['aux'] == 4
                model.set_trainable('weight', False)
                assert {b['name']: b['role'] for b in model.bindings()}['weight'] == 0
            finally:
                model.free()

    def test_target_and_objective_use_existing_training_path(self):
        spec = {'inputs': {'x': {'shape': [2, 1], 'dtype': 'float32'},
                           'y': {'shape': [2, 1], 'dtype': 'float32', 'role': 'target'}},
                'nodes': [{'name': 'pred', 'type': 'linear', 'out_features': 1, 'bias': False, 'inputs': ['x']},
                          {'name': 'error', 'type': 'sub', 'inputs': ['pred', 'y']},
                          {'name': 'sq', 'type': 'square', 'inputs': ['error']},
                          {'name': 'avg', 'type': 'mean', 'inputs': ['sq']}],
                'outputs': {'prediction': 'pred', 'cost': 'avg'},
                'entrypoints': [{'name': 'forward', 'inputs': ['x'], 'outputs': ['prediction']},
                                {'name': 'loss', 'inputs': ['x', 'y'], 'outputs': ['cost'], 'objective': 'cost'}]}
        model = Graph(spec)
        try:
            model.write_buffer('nodes.pred.weight', np.array([2], np.float32))
            x = np.array([[1], [2]], np.float32)
            np.testing.assert_array_equal(model.forward(x=x)['prediction'], [[2], [4]])
            model.set_optimizer(OPTIM_SGD, lr=.1)
            assert model.train_step(x=x, y=x) == 2.5
            np.testing.assert_allclose(model.read_buffer('nodes.pred.weight'), [1.5], atol=1e-6)
        finally:
            model.free()

    def test_aggregate_named_storage_is_bounded_before_allocation(self):
        spec = {'inputs': {'x': {'shape': [16777216], 'dtype': 'float32'},
                           'y': {'shape': [1], 'dtype': 'float32'}},
                'nodes': [], 'outputs': {'prediction': 'y'}}
        with pytest.raises(ValueError, match='named storage'):
            Graph(spec)

    @pytest.mark.parametrize('key', ['', 'format||type', 'inputs||nodes'])
    def test_unknown_fields_cannot_escape_field_validation(self, key):
        spec = definition_fixture()
        spec[key] = 1
        with pytest.raises(ValueError, match='unknown field'):
            Graph(spec)

    @pytest.mark.parametrize('number', ['01', '1.', '1.e2', '-.1'])
    def test_non_json_number_spellings_are_rejected(self, number):
        import json
        spec = json.dumps(definition_fixture()).replace('"seed": 42', '"seed": '+number)
        with pytest.raises(ValueError):
            Graph(spec)

    def test_factories_select_family_without_redundant_tags(self):
        for factory, spec in ((Sequential, sequential_definition()), (Graph, definition_fixture())):
            spec.pop('type')
            spec.pop('format')
            model = factory(spec)
            model.free()
        with pytest.raises(ValueError, match="type: expected 'sequential'"):
            Sequential(definition_fixture())
        assert not hasattr(Model, 'from_definition')

    def test_factory_scopes_logical_capture_without_mutating_existing_tensors(self):
        import polygrad as pg
        rt = pg.Runtime(device='interp', logical='never')
        try:
            existing = rt.Tensor.empty(1)
            model = Graph(definition_fixture(), runtime=rt)
            model.dispose()
            assert not pg._ffi.get_lib().poly_tensor_uop_logical(existing._tensor)
            assert int(pg._ffi.get_lib().poly_ctx_get_logical_policy(rt._ctx)) == 0
        finally:
            rt.dispose()

    @pytest.mark.parametrize('op,data', [
        ('relu', [-1, 2]), ('sigmoid', [-1, 2]), ('tanh', [-1, 2]),
        ('silu', [-1, 2]), ('gelu', [-1, 2]), ('identity', [-1, 2]),
        ('square', [-1, 2]), ('exp', [-1, 2]), ('log', [1, 2]),
        ('sum', [-1, 2]), ('mean', [-1, 2]), ('reshape', [-1, 2]),
    ])
    def test_catalogue_matches_tensor_operations(self, op, data):
        config = {'input': {'name': 'x', 'shape': [1, 2], 'dtype': 'float32'},
                  'layers': [{'name': 'op', 'type': op}], 'output': 'prediction'}
        if op == 'reshape': config['layers'][0]['shape'] = [2, 1]
        model = Sequential(config)
        tensor = Tensor(np.array([data], np.float32))
        expected = tensor.reshape(2, 1) if op == 'reshape' else tensor if op == 'identity' else getattr(tensor, op)()
        try:
            np.testing.assert_allclose(model.forward(x=np.array([data], np.float32))['prediction'], expected.numpy(), rtol=2e-5, atol=1e-6)
        finally:
            model.free()

    @pytest.mark.parametrize('op', ['add', 'sub', 'mul', 'div'])
    def test_graph_broadcast_arithmetic(self, op):
        config = {'inputs': {'x': {'shape': [2, 2], 'dtype': 'float32'},
                             'y': {'shape': [2], 'dtype': 'float32'}},
                  'nodes': [{'name': 'z', 'type': op, 'inputs': ['x', 'y']}],
                  'outputs': {'prediction': 'z'}}
        model = Graph(config)
        x, y = np.array([[1, 2], [3, 4]], np.float32), np.array([1, 2], np.float32)
        reference = {'add': np.add, 'sub': np.subtract, 'mul': np.multiply, 'div': np.divide}[op](x, y)
        try:
            np.testing.assert_array_equal(model.forward(x=x, y=y)['prediction'], reference)
        finally:
            model.free()

    def test_repeat_layer_array_and_explicit_shared_calls(self):
        spec = sequential_definition()
        spec['modules'] = {'shared': {'type': 'linear', 'out_features': 2, 'bias': False}}
        spec['layers'][0]['body'] = [{'name': 'projection', 'call': 'shared'},
                                    {'name': 'activation', 'type': 'relu'}]
        model = Sequential(spec)
        try:
            assert model.param_count == 1
            model.write_buffer('modules.shared.weight', np.array([1, 2, 3, 4], np.float32))
            np.testing.assert_array_equal(model.forward(x=np.array([[1, 2]], np.float32))['prediction'], [[27, 59]])
        finally:
            model.free()

    def test_shared_math_gradients_and_portable_export(self):
        import polygrad as pg
        rt = pg.Runtime(device='interp', logical='until_realize')
        model = Graph(definition_fixture(), runtime=rt)
        restored = None
        try:
            assert model.param_count == 1
            weights = np.array([[1, 2], [3, 4]], np.float32)
            model.write_buffer('modules.shared.weight', weights)
            x = np.array([[1, 2]], np.float32)
            np.testing.assert_array_equal(model.forward(x=x)['prediction'], [[28, 61]])
            model.set_optimizer(OPTIM_SGD, lr=.1)
            assert model.train_step(x=x) == 89
            np.testing.assert_allclose(model.read_buffer('modules.shared.weight'),
                                       (weights - .1*np.array([[9, 19], [11, 23]], np.float32)).ravel(), atol=1e-6)
            blob = model.save_bundle(include_optimizer=False)
            expected = model.forward(x=x)['prediction'].copy()
            rt.collect()
            assert int(pg._ffi.get_lib().poly_ctx_get_logical_policy(rt._ctx)) == 2  # unchanged
            rt.dispose()
            restored = Model.from_bundle(blob)
            np.testing.assert_allclose(restored.forward(x=x)['prediction'], expected, atol=1e-6)
            with pytest.raises(RuntimeError):
                Graph(definition_fixture(), runtime=rt)
        finally:
            if restored:
                restored.free()
            model.free()
            rt.dispose()

    def test_repeat_is_fresh_and_deterministic_and_matches_tensor_authoring(self):
        spec = sequential_definition()
        model = Sequential(spec)
        twin = Sequential(spec)
        direct = None
        try:
            assert model.param_count == 4
            names = [b['name'] for b in model.bindings() if b['trainable']]
            assert names == ['layers.stack.0.weight', 'layers.stack.0.bias',
                             'layers.stack.1.weight', 'layers.stack.1.bias']
            for name in names:
                np.testing.assert_array_equal(model.read_buffer(name), twin.read_buffer(name))
            x = Tensor.empty(1, 2)
            y = x
            params = {}
            for i in range(2):
                prefix = f'layers.stack.{i}'
                w = Tensor(model.read_buffer(prefix+'.weight').reshape(2, 2))
                b = Tensor(model.read_buffer(prefix+'.bias'))
                params.update({prefix+'.weight': w, prefix+'.bias': b})
                y = (y @ w.T + b).relu()
            direct = Model.from_tensors(inputs={'x': x}, outputs={'prediction': y}, params=params)
            data = np.array([[1, -2]], np.float32)
            np.testing.assert_allclose(model.forward(x=data)['prediction'], direct.forward(x=data)['prediction'], atol=1e-6)
            # Equal zero biases must not acquire shared storage identity.
            model.write_buffer(names[1], np.array([7, 8], np.float32))
            np.testing.assert_array_equal(model.read_buffer(names[3]), [0, 0])
            np.testing.assert_array_equal(twin.read_buffer(names[1]), [0, 0])
        finally:
            if direct:
                direct.free()
            model.free()
            twin.free()

    @pytest.mark.parametrize('field,value,error', [
        ('format', 'poly.modeldef@2', 'format'),
        ('seed', 1.5, 'integer'), ('seed', -1, 'integer'),
        ('seed', True, 'integer'), ('unknown', 1, 'unknown field'),
        ('entrypoints', [], 'entrypoints'),
    ])
    def test_rejects_invalid_root(self, field, value, error):
        spec = definition_fixture()
        spec[field] = value
        with pytest.raises(ValueError, match=error):
            Graph(spec)

    @pytest.mark.parametrize('field,value,error', [
        ('count', 0, 'integer'), ('count', 1.5, 'integer'),
        ('count', 1025, 'integer'), ('count', {'config': 'depth'}, 'integer'),
        ('inputs', ['x'], 'cannot specify inputs'),
        ('unknown', 1, 'unknown field'),
        ('body', {'type': 'missing'}, 'unknown activation'),
    ])
    def test_rejects_invalid_repeat(self, field, value, error):
        spec = sequential_definition()
        spec['layers'][0][field] = value
        with pytest.raises(ValueError, match=error):
            Sequential(spec)

    @pytest.mark.parametrize('field,value,error', [
        ('shape', [1, 0], 'integer'), ('shape', [1, 2.5], 'integer'),
        ('shape', [1]*9, 'rank'), ('shape', [16777216, 2], 'budget'),
        ('dtype', 'weakfloat', 'dtype'), ('dtype', 'typo', 'dtype'),
        ('role', 'parameter', 'role'),
    ])
    def test_rejects_invalid_input(self, field, value, error):
        spec = definition_fixture()
        spec['inputs']['x'][field] = value
        with pytest.raises(ValueError, match=error):
            Graph(spec)

    @pytest.mark.parametrize('json,error', [
        ('{} {}', 'trailing'), ('{"format":1,"format":2}', 'duplicate'),
        ('{"x":{"a":1,"a":2}}', 'duplicate'),
        ('{"a":"\\u0000"}', 'NUL'), ('{}\0', 'NUL'),
        ('['*33 + '0' + ']'*33, 'nesting'), ('{', 'invalid JSON'),
    ])
    def test_rejects_ambiguous_json(self, json, error):
        with pytest.raises(ValueError, match=error):
            Graph(json)

    @pytest.mark.parametrize('change,error', [
        ('forward', 'forward value'), ('cycle', 'forward value'),
        ('duplicate', 'duplicate value'), ('shared_width', 'expected 3 input features, received 2'),
        ('broadcast', 'broadcast dimensions'), ('objective', 'objective'),
        ('unknown_module', 'unknown shared'), ('unused_module', 'unused shared'),
    ])
    def test_failed_construction_leaves_runtime_usable(self, change, error):
        import copy
        import polygrad as pg
        rt = pg.Runtime(device='interp')
        spec = definition_fixture()
        if change == 'forward': spec['nodes'][0]['inputs'] = ['b']
        elif change == 'cycle': spec['nodes'][0]['inputs'] = ['a']
        elif change == 'duplicate': spec['nodes'][1]['name'] = 'a'
        elif change == 'shared_width':
            spec['inputs']['x']['shape'] = [1, 3]
        elif change == 'broadcast':
            spec['nodes'][1] = {'name': 'b', 'type': 'linear', 'out_features': 3, 'inputs': ['a']}
        elif change == 'objective': spec['entrypoints'][1]['objective'] = 'missing'
        elif change == 'unknown_module': spec['nodes'][0]['call'] = 'missing'
        elif change == 'unused_module': spec['modules']['unused'] = copy.deepcopy(spec['modules']['shared'])
        try:
            for _ in range(3):
                with pytest.raises(ValueError, match=error):
                    Graph(spec, runtime=rt)
                rt.collect()
            good = Graph(definition_fixture(), runtime=rt)
            good.free()
            rt.collect()
        finally:
            rt.dispose()


class TestStorageCopies:
    @pytest.mark.parametrize('dtype', ['float16', 'float64'])
    def test_default_dtype_capture_owns_concrete_state(self, dtype):
        from polygrad.helpers import Context
        with Context(DEFAULT_FLOAT=dtype, LOGICAL=1):
            weight = Tensor(1.25)
            before = weight.uop.raw
            model = Model.from_tensors(params={'weight': weight}, outputs={'value': weight + 1})
            try:
                assert weight.uop.raw == before
                data = model.read_buffer('weight')
                assert data.dtype == np.dtype(dtype)
                np.testing.assert_equal(data, [1.25])
            finally:
                model.free()
        np.testing.assert_equal(data, [1.25])

    @pytest.mark.parametrize('dtype', ['bool', 'int8', 'uint8', 'int16', 'uint16',
                                     'int32', 'uint32', 'int64', 'uint64',
                                     'float16', 'bfloat16', 'float32', 'float64'])
    def test_explicit_write_validates_exact_storage_before_mutation(self, dtype):
        storage_dtype = np.dtype('uint16' if dtype == 'bfloat16' else dtype)
        x = Tensor.empty(2, 2, dtype=dtype)
        inst = Model.from_tensors(inputs={'x': x}, outputs={'copy': x})
        try:
            expected = np.array([0, 1, 0, 1], dtype=storage_dtype)
            inst.write_buffer('x', expected.reshape(2, 2))
            np.testing.assert_array_equal(inst.read_buffer('x'), expected)
            with pytest.raises(TypeError):
                inst.write_buffer('x', expected.astype('float64' if dtype != 'float64' else 'float32'))
            with pytest.raises(ValueError):
                inst.write_buffer('x', expected[:2])
            with pytest.raises(ValueError):
                inst.write_buffer('x', expected.reshape(1, 4))
            np.testing.assert_array_equal(inst.read_buffer('x'), expected)
        finally:
            inst.free()

    def test_parameter_and_buffer_reads_are_independent_copies(self):
        inst = MLP(layers=[2, 1], bias=False, loss='none', batch_size=1, seed=42)
        try:
            original = inst.param_data(0).copy()
            param = inst.param_data(0)
            buf = inst.buf_data(inst.find_buf(inst.param_name(0)))
            param[:] = 101
            np.testing.assert_array_equal(inst.param_data(0), original)
            buf[:] = 202
            np.testing.assert_array_equal(inst.param_data(0), original)
        finally:
            inst.free()
        # Only independent arrays may be read after the C owner is destroyed.
        np.testing.assert_array_equal(param, [101, 101])
        np.testing.assert_array_equal(buf, [202, 202])


class TestModelConstructors:
    def test_tied_adam_survives_placement_freeze_and_checkpoint(self):
        w = Tensor([1.0])
        model = Model.from_tensors(params={'w': w, 'tied': w},
                                   losses={'cost': (w+w).square().sum()})
        restored = None
        try:
            model.place('interp')
            model.set_trainable('tied', False)
            assert not model.buf_trainable(model.find_buf('w'))
            model.set_trainable('w', True)
            assert model.buf_trainable(model.find_buf('tied'))
            model.set_optimizer(OPTIM_ADAM, lr=.1)
            assert model.train_step() == 4.0
            np.testing.assert_allclose(model.read_buffer('w'), [.9], atol=1e-6)
            names = {'optim.adam.b1_t', 'optim.adam.b2_t',
                     'optim.adam.m.w', 'optim.adam.v.w'}
            weights = model.export_weights()
            assert {n for n in safetensor_names(weights) if n.startswith('optim.')} == names
            restored = Model.from_ir(model.export_ir(), weights)
            restored.place('interp')
            restored.set_trainable('w', False)
            assert not restored.buf_trainable(restored.find_buf('tied'))
            restored.set_trainable('tied', True)
            restored.set_optimizer(OPTIM_ADAM, lr=.1)
            assert restored.train_step() == model.train_step()
            for name in names | {'w', 'tied'}:
                np.testing.assert_array_equal(restored.read_buffer(name), model.read_buffer(name))
            assert {n for n in safetensor_names(restored.export_weights()) if n.startswith('optim.')} == names
        finally:
            if restored: restored.free()
            model.free()

    @pytest.mark.parametrize('dtype', ['float16', 'float64'])
    def test_non_f32_training_preserves_named_objective_storage(self, dtype):
        from polygrad import create
        rt = create(device='interp', logical='always')
        w = rt.Tensor([2.0], dtype=dtype)
        model = Model.from_tensors(params={'w': w}, losses={'objective': w.square().sum()})
        try:
            model.set_optimizer(OPTIM_SGD, lr=.1)
            assert model.train_step() == 4.0
            named = model.read_buffer('objective')
            assert named.dtype == np.dtype(dtype)
            np.testing.assert_array_equal(named, np.array([4], dtype=dtype))
            np.testing.assert_allclose(model.read_buffer('w'), [1.6], atol=1e-3 if dtype == 'float16' else 1e-6)
        finally:
            model.free()
            rt.dispose()

    def test_trace_trains_c_owned_state_and_exports_without_author(self):
        import gc
        from polygrad import create
        from polygrad.nn.state import get_state_dict
        rt = create(device='interp', logical='always')
        class Net:
            def __init__(self):
                self.weight = rt.Tensor([2.0])
                self.offset = rt.Tensor([1.0]).is_param_(False)
            def __call__(self, x):
                return x * self.weight + self.offset
        net = Net()
        x, y = rt.Tensor.empty(1), rt.Tensor.empty(1)
        model = Model.from_callable(net, inputs={'x': x}, targets={'y': y},
                            loss=lambda out, y: {'mse': (out-y).square().mean()},
                            params=get_state_dict(net))
        restored = None
        try:
            assert {row['name']: row['role'] for row in model.bindings()}['offset'] == 4
            assert next(row for row in model.entrypoints() if row['name'] == 'loss') == {
                'name': 'loss', 'inputs': ['x', 'y'], 'outputs': ['mse'], 'objective': 'mse'}
            model.set_optimizer(OPTIM_SGD, lr=.1)
            assert model.train_step(x=np.array([1], np.float32), y=np.array([0], np.float32)) == 9
            np.testing.assert_allclose(model.read_buffer('weight'), [1.4])
            np.testing.assert_array_equal(net.weight.numpy(), [2])
            blob = model.save_bundle(include_optimizer=False)
            copied = model.read_buffer('weight')
            model.place('interp')
            rt.collect()
            del net, x, y
            model.free()
            rt.dispose()
            gc.collect()
            np.testing.assert_allclose(copied, [1.4])
            restored = Model.from_bundle(blob)
            np.testing.assert_allclose(restored.forward(x=np.array([1], np.float32))['output'], [2.4])
        finally:
            if restored:
                restored.free()
            model.free()
            rt.dispose()

    def test_objective_selection_is_explicit_when_ambiguous(self):
        w = Tensor([2.0])
        a, b = (w*w).sum(), (w*w*w).sum()
        model = Model.from_tensors(params={'w': w}, losses={'a': a, 'b': b}, entrypoints=[
            {'name': 'a_ep', 'outputs': ['a'], 'objective': 'a'},
            {'name': 'b_ep', 'outputs': ['a', 'b'], 'objective': 'b'}])
        try:
            model.set_optimizer(OPTIM_SGD, lr=.1)
            with pytest.raises(RuntimeError):
                model.train_step()
            with pytest.raises(RuntimeError):
                model.train_step(entrypoint='missing')
            np.testing.assert_array_equal(model.read_buffer('w'), [2])
            assert model.train_step(entrypoint='b_ep') == 8
            np.testing.assert_allclose(model.read_buffer('w'), [.8], atol=1e-6)
            np.testing.assert_array_equal(model.read_buffer('b'), [8])
        finally:
            model.free()

    def test_family_constructors_are_not_instance_methods(self):
        assert not hasattr(Model, 'mlp')
        assert callable(MLP)


class TestCompiledProgramExport:
    def test_bound_program_and_separate_weights_roundtrip(self):
        source = MLP({
            'layers': [2, 3, 1], 'activation': 'relu',
            'bias': True, 'loss': 'mse', 'batch_size': 1, 'seed': 42,
        })
        restored = None
        try:
            program = source.export_program()
            weights = source.export_weights()
            assert program[:4] == b'PGPM'
            assert program != source.export_ir()
            with pytest.raises(RuntimeError):
                Model.from_program(program)
            restored = Model.from_program(program, weights)
            x = np.array([1.25, -0.5], dtype=np.float32)
            y = np.array([0.75], dtype=np.float32)
            np.testing.assert_array_equal(
                restored.forward(x=x)['output'], source.forward(x=x)['output'])
            np.testing.assert_array_equal(
                restored.call('loss', x=x, y=y)['loss'],
                source.call('loss', x=x, y=y)['loss'])
            assert restored.export_ir() is None
            assert restored.export_program() == program
        finally:
            if restored is not None:
                restored.free()
            source.free()


class TestMLPCreate:
    @pytest.mark.parametrize('device', ['CPU', 'INTERP'])
    def test_distilgpt2_registry_extension(self, device):
        import polygrad as pg
        spec = dict(vocab_size=8, n_embd=4, n_head=2, n_positions=2)
        with pg.create(device=device) as rt:
            # A new C preset must appear without a Python-specific factory.
            preset = rt.models.DistilGPT2(spec)
            explicit = rt.models.GPT2({**spec, 'n_layer': 6})
            try:
                assert preset.param_count == explicit.param_count == 76
                for i in range(preset.param_count):
                    name = preset.param_name(i)
                    assert name == explicit.param_name(i)
                    count = int(np.prod(preset.param_shape(i)))
                    values = (np.arange(count, dtype=np.float32) % 11 - 5) / 16
                    preset.write_buffer(name, values)
                    explicit.write_buffer(name, values)
                # Same graph, state, signatures and canonical identities.
                assert preset.save() == explicit.save()
                inputs = dict(x=np.array([[1, 2]], dtype=np.int32),
                              positions=np.array([[0, 1]], dtype=np.int32))
                np.testing.assert_array_equal(preset.forward(**inputs)['output'],
                                              explicit.forward(**inputs)['output'])
            finally:
                preset.dispose()
                explicit.dispose()

    def test_model_type_capabilities(self):
        import polygrad as pg
        available = {entry['name']: entry for entry in pg.models.list()}
        assert available['GPT2'] == dict(name='GPT2', constructible=True, hf=True, gguf=True)
        assert available['Llama'] == dict(name='Llama', constructible=True, hf=True, gguf=False)
        assert available['Qwen3']['gguf'] and not available['Qwen3']['hf']
        with pg.create(device='INTERP') as rt:
            assert rt.models.list() == pg.models.list()
        for name, entry in available.items():
            assert callable(getattr(pg.models, name, None)) == entry['constructible']

    @pytest.mark.parametrize('family,spec', [
        ('MLP', {'layers': [2, 3, 2]}),
        ('TabM', {'layers': [2, 3, 2], 'n_ensemble': 2}),
        ('NAM', {'n_features': 2, 'hidden_sizes': [3], 'n_outputs': 2}),
    ])
    @pytest.mark.parametrize('loss', ['mse', 'cross_entropy'])
    @pytest.mark.parametrize('device', ['CPU', 'INTERP'])
    def test_shared_loss_matches_tensor_expression(self, family, spec, loss, device):
        import polygrad as pg
        with pg.create(device=device) as rt:
            model = getattr(pg.models, family)({**spec, 'loss': loss}, runtime=rt)
            try:
                x = np.array([[.25, -.5]], dtype=np.float32)
                y = np.array([[0., 1.]], dtype=np.float32)
                prediction = model.forward(x=x)['output']
                p, t = rt.Tensor(prediction.reshape(1, 2)), rt.Tensor(y)
                expected = (p-t).square().mean() if loss == 'mse' else p.cross_entropy(t)
                np.testing.assert_allclose(model.call('loss', x=x, y=y)['loss'],
                                           expected.numpy(), rtol=1e-5, atol=1e-6)
            finally:
                model.dispose()

    @pytest.mark.parametrize('family,spec', [
        ('MLP', {'layers': [2, 3, 1]}),
        ('TabM', {'layers': [2, 3, 1], 'n_ensemble': 2}),
        ('NAM', {'n_features': 2, 'hidden_sizes': [3], 'n_outputs': 1}),
    ])
    def test_family_uses_default_runtime(self, family, spec):
        import polygrad as pg
        model = getattr(pg.models, family)(spec)
        try:
            assert model._ctx == pg._default_ctx
        finally:
            model.dispose()

    @pytest.mark.parametrize('family,spec', [
        ('MLP', {'layers': [2, 3, 1]}),
        ('TabM', {'layers': [2, 3, 1], 'n_ensemble': 2}),
        ('NAM', {'n_features': 2, 'hidden_sizes': [3], 'n_outputs': 1}),
    ])
    def test_family_runtime_isolation_and_disposal(self, family, spec):
        import polygrad as pg
        rt = pg.create(device='interp', logical='never')
        models = []
        try:
            live = rt.Tensor([19.0])
            caller_records = rt.stats()['tensor_records']
            factory = getattr(pg.models, family)
            a, b = factory(spec, runtime=rt), factory(spec, runtime=rt)
            models.extend([a, b])
            assert a._ctx == b._ctx == rt._ctx
            name = a.param_name(0)
            before = b.read_buffer(name)
            a.write_buffer(name, np.full_like(before, 7))
            np.testing.assert_array_equal(b.read_buffer(name), before)
            expected = b.forward(x=np.array([1, 2], dtype=np.float32))['output']
            a.dispose()
            pg._ffi.get_lib().poly_ctx_collect(rt._ctx)
            assert rt.stats()['tensor_records'] == caller_records
            np.testing.assert_array_equal(b.forward(x=[1, 2])['output'], expected)
            np.testing.assert_array_equal(live.numpy(), [19])
            assert pg._ffi.get_lib().poly_ctx_get_logical_policy(rt._ctx) == 0
            rt.dispose()
            with pytest.raises(RuntimeError):
                b.forward(x=[1, 2])
            with pytest.raises(RuntimeError):
                factory(spec, runtime=rt)
        finally:
            for model in models: model.dispose()
            rt.dispose()

    def test_create_simple(self):
        inst = MLP(
            layers=[2, 4, 1], activation='relu',
            bias=True, loss='mse', batch_size=1, seed=42,
        )
        assert inst.param_count == 4
        assert inst.param_name(0) == 'layers.0.weight'
        assert inst.param_shape(0) == [4, 2]
        inst.free()

    def test_create_no_bias(self):
        inst = MLP({
            'layers': [3, 2], 'activation': 'none',
            'bias': False, 'loss': 'none', 'batch_size': 1, 'seed': 42
        })
        assert inst.param_count == 1
        assert inst.param_name(0) == 'layers.0.weight'
        assert inst.param_shape(0) == [2, 3]
        inst.free()

    def test_deterministic_init(self):
        spec = {
            'layers': [2, 4, 1], 'activation': 'relu',
            'bias': True, 'loss': 'mse', 'batch_size': 1, 'seed': 42
        }
        i1 = MLP(spec)
        i2 = MLP(spec)
        np.testing.assert_array_equal(i1.param_data(0), i2.param_data(0))
        i1.free()
        i2.free()

    def test_null_spec(self):
        with pytest.raises(ValueError, match='layers'):
            MLP('{}')


class TestForward:
    def test_factory_constructs_on_requested_device(self):
        inst = MLP({
            'layers': [2, 4, 1], 'activation': 'relu',
            'bias': True, 'loss': 'mse', 'batch_size': 1, 'seed': 42,
        }, device='INTERP')
        outputs = inst.forward(x=np.array([1.0, 2.0], dtype=np.float32))
        assert np.isfinite(outputs['output'][0])
        inst.free()

    def test_forward_produces_output(self):
        inst = MLP({
            'layers': [2, 4, 1], 'activation': 'relu',
            'bias': True, 'loss': 'mse', 'batch_size': 1, 'seed': 42
        })
        outputs = inst.forward(x=np.array([1.0, 2.0], dtype=np.float32))
        assert 'output' in outputs
        assert np.isfinite(outputs['output'][0])
        inst.free()

    def test_forward_deterministic(self):
        inst = MLP({
            'layers': [2, 4, 1], 'activation': 'relu',
            'bias': True, 'loss': 'mse', 'batch_size': 1, 'seed': 42
        })
        out1 = inst.forward(x=np.array([1.0, 2.0], dtype=np.float32))
        out2 = inst.forward(x=np.array([1.0, 2.0], dtype=np.float32))
        np.testing.assert_array_equal(out1['output'], out2['output'])
        inst.free()

    def test_missing_required_input_rejects_instead_of_reusing_stale_bytes(self):
        x, y = Tensor.empty(2), Tensor.empty(2)
        inst = Model.from_tensors(
            inputs={'x': x, 'y': y}, outputs={'output': x + y},
        )
        try:
            inst.forward(
                x=np.array([1.0, 2.0], dtype=np.float32),
                y=np.array([3.0, 4.0], dtype=np.float32),
            )
            with pytest.raises(RuntimeError):
                inst.forward(x=np.array([9.0, 9.0], dtype=np.float32))
        finally:
            inst.free()

    def test_generic_call_returns_only_selected_entrypoint_outputs(self):
        x = Tensor.empty(2)
        plus, minus = x + 1.0, x - 1.0
        inst = Model.from_tensors(
            inputs={'x': x},
            outputs={'plus': plus, 'minus': minus},
            entrypoints=[
                {'name': 'plus_ep', 'inputs': ['x'], 'outputs': ['plus']},
                {'name': 'minus_ep', 'inputs': ['x'], 'outputs': ['minus']},
            ],
        )
        try:
            data = np.array([3.0, 5.0], dtype=np.float32)
            plus_out = inst.call('plus_ep', x=data)
            minus_out = inst.call('minus_ep', {'x': data})
            assert set(plus_out) == {'plus'}
            assert set(minus_out) == {'minus'}
            np.testing.assert_array_equal(plus_out['plus'], [4.0, 6.0])
            np.testing.assert_array_equal(minus_out['minus'], [2.0, 4.0])
        finally:
            inst.free()

    def test_invalid_parameter_is_rejected_not_silently_filtered(self):
        x = Tensor.empty(2)
        with pytest.raises(TypeError, match='not a Tensor'):
            Model.from_tensors(
                inputs={'x': x}, outputs={'output': x + 1.0}, params={'bad': object()},
            )


class TestTrain:
    def test_train_sgd(self):
        inst = MLP({
            'layers': [2, 1], 'activation': 'none',
            'bias': True, 'loss': 'mse', 'batch_size': 1, 'seed': 42
        })
        inst.set_optimizer(OPTIM_SGD, lr=0.05)
        x = np.array([1.0, 2.0], dtype=np.float32)
        y = np.array([5.0], dtype=np.float32)

        losses = []
        for _ in range(50):
            loss = inst.train_step(x=x, y=y)
            assert np.isfinite(loss)
            losses.append(loss)

        assert losses[-1] < losses[0]
        inst.free()

    def test_train_multi_layer(self):
        inst = MLP({
            'layers': [1, 4, 1], 'activation': 'relu',
            'bias': True, 'loss': 'mse', 'batch_size': 1, 'seed': 42
        })
        inst.set_optimizer(OPTIM_SGD, lr=0.01)
        x = np.array([1.0], dtype=np.float32)
        y = np.array([2.0], dtype=np.float32)

        losses = []
        for _ in range(100):
            loss = inst.train_step(x=x, y=y)
            assert np.isfinite(loss)
            losses.append(loss)

        assert losses[-1] < losses[0]
        inst.free()


class TestWeightIO:
    def test_export_import_round_trip(self):
        spec = {
            'layers': [2, 4, 1], 'activation': 'relu',
            'bias': True, 'loss': 'mse', 'batch_size': 1, 'seed': 42
        }
        inst = MLP(spec)
        original_w = inst.param_data(0).copy()

        # Export
        st_bytes = inst.export_weights()
        assert st_bytes is not None
        assert len(st_bytes) > 0

        # Create fresh instance with different seed
        spec2 = dict(spec, seed=99)
        inst2 = MLP(spec2)
        different_w = inst2.param_data(0).copy()
        assert not np.array_equal(original_w, different_w)

        # Import weights from first instance
        inst2.import_weights(st_bytes)
        np.testing.assert_array_equal(inst2.param_data(0), original_w)

        inst.free()
        inst2.free()

    def test_adam_checkpoint_resume_matches_uninterrupted_training(self):
        spec = {
            'layers': [2, 1], 'activation': 'none',
            'bias': False, 'loss': 'mse', 'batch_size': 1, 'seed': 7,
        }
        source = MLP(spec)
        restored = None
        try:
            source.set_optimizer(OPTIM_ADAM, lr=0.05)
            io = {
                'x': np.array([1.0, 2.0], dtype=np.float32),
                'y': np.array([3.0], dtype=np.float32),
            }
            for _ in range(3):
                source.train_step(**io)
            restored = Model.from_ir(source.export_ir(), source.export_weights())
            restored.set_optimizer(OPTIM_ADAM, lr=0.05)

            source_loss = source.train_step(**io)
            restored_loss = restored.train_step(**io)
            assert source_loss == restored_loss
            np.testing.assert_array_equal(source.param_data(0), restored.param_data(0))
            for name in (
                'optim.adam.b1_t', 'optim.adam.b2_t',
                'optim.adam.m.layers.0.weight', 'optim.adam.v.layers.0.weight',
            ):
                source_i, restored_i = source.find_buf(name), restored.find_buf(name)
                assert source_i >= 0 and restored_i >= 0
                np.testing.assert_array_equal(
                    source.buf_data(source_i), restored.buf_data(restored_i)
                )
        finally:
            if restored is not None:
                restored.free()
            source.free()

    def test_stochastic_named_state_requires_checkpoint_for_portable_activation(self):
        Tensor.manual_seed(11)
        w = Tensor.rand(2)
        x = Tensor.empty(2)
        source = Model.from_tensors(
            inputs={'x': x}, outputs={'output': x * w}, params={'w': w},
        )
        restored = None
        try:
            ir, weights = source.export_ir(), source.export_weights()
            with pytest.raises(RuntimeError):
                Model.from_ir(ir)
            restored = Model.from_ir(ir, weights)
            data = np.array([2.0, 3.0], dtype=np.float32)
            np.testing.assert_array_equal(
                source.forward(x=data)['output'], restored.forward(x=data)['output']
            )
        finally:
            if restored is not None:
                restored.free()
            source.free()

    @pytest.mark.parametrize('dtype,values,numpy_dtype', [
        ('float64', [1.25, -2.5], np.float64),
        ('int32', [1, -2], np.int32),
        ('uint8', [1, 255], np.uint8),
        ('bool', [True, False], np.bool_),
        ('bfloat16', [1.5, -2.0], np.uint16),
    ])
    def test_typed_named_state_round_trips_exact_storage_bytes(
        self, dtype, values, numpy_dtype
    ):
        w = Tensor(np.asarray(values), dtype=dtype)
        x = Tensor.empty(2, dtype=dtype)
        source = Model.from_tensors(
            inputs={'x': x}, outputs={'output': x}, params={'w': w},
        )
        restored = None
        try:
            restored = Model.from_ir(source.export_ir(), source.export_weights())
            before, after = source.param_data(0), restored.param_data(0)
            assert source.param_dtype(0) == restored.param_dtype(0) == dtype
            assert before.dtype == after.dtype == np.dtype(numpy_dtype)
            np.testing.assert_array_equal(
                before.view(np.uint8), after.view(np.uint8)
            )
        finally:
            if restored is not None:
                restored.free()
            source.free()


class TestBufferEnumeration:
    def test_buf_roles(self):
        inst = MLP({
            'layers': [2, 1], 'activation': 'none',
            'bias': True, 'loss': 'mse', 'batch_size': 1, 'seed': 42
        })
        roles = {}
        for i in range(inst.buf_count):
            name = inst.buf_name(i)
            role = inst.buf_role(i)
            roles[name] = role

        assert 'layers.0.weight' in roles
        assert 'x' in roles
        inst.free()

    def test_find_buf(self):
        inst = MLP({
            'layers': [2, 1], 'activation': 'none',
            'bias': True, 'loss': 'mse', 'batch_size': 1, 'seed': 42
        })
        idx = inst.find_buf('output')
        assert idx >= 0
        assert inst.buf_name(idx) == 'output'
        assert inst.find_buf('nonexistent') == -1
        inst.free()


class TestParams:
    def test_param_iteration(self):
        inst = MLP({
            'layers': [2, 4, 1], 'activation': 'relu',
            'bias': True, 'loss': 'mse', 'batch_size': 1, 'seed': 42
        })
        params = list(inst.params())
        assert len(params) == 4
        assert params[0][0] == 'layers.0.weight'
        assert params[0][1] == [4, 2]
        assert params[0][2] is not None
        assert len(params[0][2]) == 8  # 4*2
        inst.free()

    def test_param_trainability_freezes_optimizer_updates(self):
        inst = MLP({
            'layers': [2, 1], 'activation': 'none',
            'bias': True, 'loss': 'mse', 'batch_size': 1, 'seed': 42
        })
        try:
            assert inst.param_trainable(0) is True
            assert inst.param_trainable(1) is True
            inst.set_param_trainable(0, False)
            assert inst.param_trainable(0) is False

            weight_before = inst.param_data(0).copy()
            bias_before = inst.param_data(1).copy()
            inst.set_optimizer(OPTIM_SGD, lr=0.05)
            x = np.array([1.0, 2.0], dtype=np.float32)
            y = np.array([5.0], dtype=np.float32)
            for _ in range(10):
                loss = inst.train_step(x=x, y=y)
                assert np.isfinite(loss)

            np.testing.assert_array_equal(inst.param_data(0), weight_before)
            assert not np.array_equal(inst.param_data(1), bias_before)
        finally:
            inst.free()

    def test_trainability_survives_ir_roundtrip(self):
        inst = MLP({
            'layers': [2, 1], 'activation': 'none',
            'bias': True, 'loss': 'mse', 'batch_size': 1, 'seed': 42
        })
        try:
            inst.set_param_trainable(0, False)
            ir = inst.export_ir()
            weights = inst.export_weights()
            inst2 = Model.from_ir(ir, weights)
            try:
                assert inst2.param_trainable(0) is False
                assert inst2.param_trainable(1) is True
            finally:
                inst2.free()
        finally:
            inst.free()


class TestTrainBatch:
    """batch_size>1 training (P0 regression coverage)."""

    def test_train_batch2_mse(self):
        inst = MLP({
            'layers': [2, 3, 2], 'activation': 'relu',
            'bias': False, 'loss': 'mse', 'batch_size': 2, 'seed': 42
        })
        inst.set_optimizer(OPTIM_SGD, lr=0.01)
        x = np.ones((2, 2), dtype=np.float32) * 0.5
        y = np.ones((2, 2), dtype=np.float32) * 0.3
        first = inst.train_step(x=x, y=y)
        for _ in range(49):
            last = inst.train_step(x=x, y=y)
        assert np.isfinite(first) and np.isfinite(last)
        assert last < first
        inst.free()

    def test_train_batch32_cross_entropy(self):
        inst = MLP({
            'layers': [4, 8, 3], 'activation': 'relu',
            'bias': True, 'loss': 'cross_entropy', 'batch_size': 32, 'seed': 42
        })
        inst.set_optimizer(OPTIM_SGD, lr=0.01)
        x = np.array([[(i % 7) * 0.1] * 4 for i in range(32)], dtype=np.float32)
        y = np.zeros((32, 3), dtype=np.float32)
        for i in range(32):
            y[i, i % 3] = 1.0
        first = inst.train_step(x=x, y=y)
        for _ in range(29):
            last = inst.train_step(x=x, y=y)
        assert np.isfinite(first) and np.isfinite(last)
        assert last < first
        inst.free()


class TestTrainOptimizers:
    """Adam/AdamW optimizer coverage."""

    def test_train_adam(self):
        inst = MLP({
            'layers': [2, 4, 1], 'activation': 'relu',
            'bias': True, 'loss': 'mse', 'batch_size': 1, 'seed': 42
        })
        inst.set_optimizer(OPTIM_ADAM, lr=0.01)
        x = np.array([1.0, 2.0], dtype=np.float32)
        y = np.array([3.0], dtype=np.float32)
        first = inst.train_step(x=x, y=y)
        for _ in range(49):
            last = inst.train_step(x=x, y=y)
        assert np.isfinite(first) and np.isfinite(last)
        assert last < first
        inst.free()

    def test_train_adamw(self):
        inst = MLP({
            'layers': [2, 4, 1], 'activation': 'relu',
            'bias': True, 'loss': 'mse', 'batch_size': 1, 'seed': 42
        })
        inst.set_optimizer(OPTIM_ADAMW, lr=0.01, weight_decay=0.01)
        x = np.array([1.0, 2.0], dtype=np.float32)
        y = np.array([3.0], dtype=np.float32)
        first = inst.train_step(x=x, y=y)
        for _ in range(49):
            last = inst.train_step(x=x, y=y)
        assert np.isfinite(first) and np.isfinite(last)
        assert last < first
        inst.free()

    def test_train_sgd_momentum_creates_named_state(self):
        inst = MLP({
            'layers': [2, 4, 1], 'activation': 'relu',
            'bias': True, 'loss': 'mse', 'batch_size': 1, 'seed': 42
        })
        inst.set_optimizer(OPTIM_SGD, lr=0.01, momentum=0.9)
        x = np.array([1.0, 2.0], dtype=np.float32)
        y = np.array([3.0], dtype=np.float32)
        loss = inst.train_step(x=x, y=y)
        assert np.isfinite(loss)
        bi = inst.find_buf('optim.sgd.b.layers.0.weight')
        assert bi >= 0
        buf = inst.buf_data(bi)
        assert buf is not None
        assert np.any(np.abs(buf) > 0)
        default_names = safetensor_names(inst.export_weights())
        assert 'optim.sgd.b.layers.0.weight' in default_names
        model_only_names = safetensor_names(inst.export_weights(include_optimizer=False))
        assert 'layers.0.weight' in model_only_names
        assert 'optim.sgd.b.layers.0.weight' not in model_only_names
        inst.free()


class TestSetDevice:
    """set_device + training (CUDA train_step crash regression)."""

    def test_train_with_set_device(self):
        from polygrad import _ffi

        inst = MLP({
            'layers': [2, 4, 1], 'activation': 'relu',
            'bias': True, 'loss': 'mse', 'batch_size': 1, 'seed': 42
        })
        # Exercise AUTO through the already-selected ABI, including pip installs.
        assert _ffi.get_lib().poly_model_set_device(inst._ptr, 0) == 0

        inst.set_optimizer(OPTIM_SGD, lr=0.05)
        x = np.array([1.0, 2.0], dtype=np.float32)
        y = np.array([5.0], dtype=np.float32)
        first = inst.train_step(x=x, y=y)
        for _ in range(9):
            last = inst.train_step(x=x, y=y)
        assert np.isfinite(first) and np.isfinite(last)
        assert last < first
        inst.free()


class TestModelTensorParity:
    """Verify Model and Tensor APIs produce same results."""

    def test_linear_sgd_parity(self):
        from polygrad.tensor import Tensor

        # Model path
        inst = MLP({
            'layers': [2, 1], 'activation': 'none',
            'bias': False, 'loss': 'mse', 'batch_size': 1, 'seed': 42
        })
        W_init = inst.param_data(0).copy()  # shape (1, 2)
        inst.set_optimizer(OPTIM_SGD, lr=0.01)
        x_data = np.array([1.0, 2.0], dtype=np.float32)
        y_data = np.array([3.0], dtype=np.float32)
        inst_losses = []
        for _ in range(5):
            loss = inst.train_step(x=x_data, y=y_data)
            inst_losses.append(loss)
        inst.free()

        # Tensor path (manual SGD)
        W = Tensor(W_init.reshape(1, 2))
        tensor_losses = []
        for _ in range(5):
            x = Tensor(x_data.reshape(1, 2))
            y = Tensor(y_data.reshape(1, 1))
            pred = x.matmul(W.transpose())
            diff = pred - y
            loss = (diff * diff).sum()
            tensor_losses.append(loss.item())
            loss.backward()
            # Manual SGD: W = W - lr * grad
            W = Tensor((W.numpy() - 0.01 * W.grad.numpy()))

        for i in range(5):
            np.testing.assert_allclose(inst_losses[i], tensor_losses[i], rtol=1e-4,
                                       err_msg=f'step {i} loss mismatch')
