"""Tests for the PolyModel Python wrapper."""

import numpy as np
import pytest
from polygrad.model import Model, OPTIM_SGD, OPTIM_ADAM, OPTIM_ADAMW
from polygrad.models import MLP, Graph, Sequential
from polygrad.tensor import Tensor


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
        np.testing.assert_array_equal(first.numpy(), a*3)
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
        with pytest.raises((TypeError, ValueError), match='format|configuration'):
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
        with pytest.raises(ValueError, match='type must match'):
            Sequential(definition_fixture())
        assert not hasattr(Model, 'from_definition')

    def test_physical_only_context_is_rejected_without_policy_mutation(self):
        import polygrad as pg
        rt = pg.Runtime(device='interp', logical='never')
        try:
            with pytest.raises(ValueError, match='requires logical construction'):
                Graph(definition_fixture(), runtime=rt)
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
        ('dtype', 'float64', 'float32'), ('dtype', 'typo', 'float32'),
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
        with pytest.raises(RuntimeError):
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
