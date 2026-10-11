"""Tabular templates: member-wise training, bounded batches and quantiles."""
import numpy as np
import pytest
import polygrad as pg


@pytest.mark.parametrize('batch', [1, 2, 32])
@pytest.mark.parametrize('device', ['CPU', 'INTERP', 'CUDA'])
def test_tabm_member_objective_and_independent_head(batch, device):
    if device == 'CUDA' and not pg._ffi.get_lib().poly_cuda_available():
        pytest.skip('poly_cuda_available() is false in the selected library')
    with pg.create(device=device) as rt:
        model = rt.models.TabM(dict(layers=[1, 1], n_ensemble=2,
                                   batch_size=batch, loss='mse'))
        try:
            model.write_buffer('output.weight', np.array([[[1]], [[3]]], np.float32))
            model.write_buffer('output.bias', np.zeros((2, 1), np.float32))
            x, y = np.ones((batch, 1), np.float32), np.zeros((batch, 1), np.float32)
            np.testing.assert_allclose(model.forward(x=x)['output'], 2)
            members = model.call('members', x=x)['members']
            np.testing.assert_array_equal(members.shape, (batch, 2, 1))
            np.testing.assert_allclose(members, np.tile([[[1], [3]]], (batch, 1, 1)))
            np.testing.assert_allclose(model.call('loss', x=x, y=y)['loss'], 5)
            model.set_optimizer('sgd', lr=.1)
            np.testing.assert_allclose(model.train_step(x=x, y=y), 5)
            np.testing.assert_allclose(model.read_buffer('output.weight'), [.9, 2.7])
            np.testing.assert_allclose(model.read_buffer('output.bias'), [-.1, -.3])
        finally:
            model.dispose()


def test_tabm_initialization_is_seeded_and_members_are_distinct():
    with pg.create(device='CPU') as rt:
        models = [rt.models.TabM(dict(layers=[8, 12, 4, 1], n_ensemble=4, seed=s))
                  for s in (42, 42, 43)]
        try:
            for name in (models[0].param_name(i) for i in range(models[0].param_count)):
                np.testing.assert_array_equal(models[0].read_buffer(name), models[1].read_buffer(name))
            r = models[0].read_buffer('layers.0.r').reshape(4, 8)
            assert set(np.unique(r)) == {-1, 1}
            assert np.any(np.ptp(r, axis=0))
            assert not np.array_equal(r.ravel(), models[2].read_buffer('layers.0.r'))
            np.testing.assert_array_equal(models[0].read_buffer('layers.1.r'), 1)
            np.testing.assert_array_equal(models[0].read_buffer('layers.0.s'), 1)
            bias = models[0].read_buffer('layers.0.b').reshape(4, 12)
            np.testing.assert_array_equal(bias, np.tile(bias[:1], (4, 1)))
            assert np.any(np.ptp(models[0].read_buffer('output.weight').reshape(4, -1), axis=0))
        finally:
            for model in models:
                model.dispose()


def test_tabm_classification_averages_probabilities_not_logits():
    with pg.create(device='CPU') as rt:
        model = rt.models.TabM(dict(layers=[1, 2], n_ensemble=2, loss='cross_entropy'))
        try:
            logits = np.array([[4., 0.], [0., 0.]], np.float32)
            model.write_buffer('output.weight', logits.reshape(2, 1, 2))
            model.write_buffer('output.bias', np.zeros((2, 2), np.float32))
            x, y = np.ones((1, 1), np.float32), np.array([[0., 1.]], np.float32)
            probs = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs /= probs.sum(axis=1, keepdims=True)
            # Keep the existing classification output contract usable by softmax consumers.
            np.testing.assert_allclose(model.forward(x=x)['output'], np.log(probs.mean(axis=0))[None], rtol=1e-5)
            np.testing.assert_allclose(model.call('loss', x=x, y=y)['loss'], -np.log(probs[:, 1]).mean(), rtol=1e-5)
        finally:
            model.dispose()


@pytest.mark.parametrize('family,spec', [
    ('MLP', {'layers': [2, 3]}),
    ('NAM', {'n_features': 2, 'hidden_sizes': [], 'n_outputs': 3}),
    ('NAM', {'n_features': 2, 'hidden_sizes': [3, 2], 'n_outputs': 3}),
    ('TabM', {'layers': [2, 3], 'n_ensemble': 2}),
])
def test_tabular_partial_batches_pinball_and_checkpoint(family, spec):
    with pg.create(device='CPU') as rt:
        model = getattr(rt.models, family)({**spec, 'batch_size': 4, 'loss': 'pinball',
                                           'quantiles': [.1, .5, .9]})
        restored = None
        try:
            x = np.arange(10, dtype=np.float32).reshape(5, 2) / 10
            y = np.arange(5, dtype=np.float32).reshape(5, 1) - .7
            for size in (4, 1, 3):
                pred = (model.call('members', x=x[:size])['members'] if family == 'TabM'
                        else model.forward(x=x[:size])['output'])
                target = y[:size, None] if family == 'TabM' else y[:size]
                error = target - pred
                expected = np.maximum(error * [.1, .5, .9], error * [-.9, -.5, -.1]).mean()
                np.testing.assert_allclose(model.call('loss', x=x[:size], y=y[:size])['loss'], expected, rtol=2e-5)
            history = model.fit(x=x, y=y, batch_size=4, remainder='keep', optimizer='adam', lr=.01)
            assert len(history) == 2 and np.isfinite(history).all()
            restored = rt.Model.load(model.save())
            np.testing.assert_array_equal(model.forward(x=x[:3])['output'], restored.forward(x=x[:3])['output'])
        finally:
            if restored is not None:
                restored.dispose()
            model.dispose()


@pytest.mark.parametrize('quantiles', [None, [.5], [.9, .1], [0, .5], [.5, 1], [.5, .5]])
def test_pinball_rejects_invalid_levels(quantiles):
    with pg.create(device='CPU') as rt:
        with pytest.raises(ValueError, match='quantiles'):
            rt.models.TabM(dict(layers=[1, 2], loss='pinball', quantiles=quantiles))


@pytest.mark.parametrize('family,members', [('MLP', 1), ('TabM', 2)])
def test_pinball_parameter_gradients(family, members):
    with pg.create(device='CPU') as rt:
        spec = dict(layers=[1, 2], loss='pinball', quantiles=[.25, .75])
        if family == 'TabM':
            spec['n_ensemble'] = members
        model = getattr(rt.models, family)(spec)
        try:
            name = 'output.weight' if family == 'TabM' else 'layers.0.weight'
            values = np.tile(np.array([1., 3.], np.float32), members)
            model.write_buffer(name, values)
            if family == 'TabM':
                model.write_buffer('output.bias', np.zeros(4, np.float32))
            model.set_optimizer('sgd', lr=.1)
            loss = model.train_step(x=np.ones((1, 1), np.float32), y=np.zeros((1, 1), np.float32))
            np.testing.assert_allclose(loss, .75, atol=1e-6)
            expected = values - .1*np.tile([.75, .25], members)/(2*members)
            np.testing.assert_allclose(model.read_buffer(name), expected, atol=1e-6)
        finally:
            model.dispose()


def test_learning_rate_update_preserves_adam_state_and_resume():
    with pg.create(device='CPU') as rt:
        m = rt.models.MLP(dict(layers=[1, 1], bias=False, loss='mse'))
        restored = None
        try:
            m.write_buffer('layers.0.weight', np.array([1], np.float32))
            m.set_optimizer('adam', lr=.1)
            x, y = np.ones((1, 1), np.float32), np.zeros((1, 1), np.float32)
            m.train_step(x=x, y=y)
            weight, moment, variance = .9, .2, .004
            before = m.read_buffer('optim.adam.m.layers.0.weight').copy()
            m.set_learning_rate(.01)
            np.testing.assert_array_equal(before, m.read_buffer('optim.adam.m.layers.0.weight'))
            restored = rt.Model.load(m.save())
            restored.set_optimizer('adam', lr=.01)
            for step, lr in enumerate((.01, .003, 0.), 2):
                m.set_learning_rate(lr)
                restored.set_learning_rate(lr)
                np.testing.assert_array_equal(m.train_step(x=x, y=y), restored.train_step(x=x, y=y))
                np.testing.assert_array_equal(m.read_buffer('layers.0.weight'), restored.read_buffer('layers.0.weight'))
                grad = 2*weight
                moment = .9*moment + .1*grad
                variance = .999*variance + .001*grad*grad
                weight -= lr*(moment/(1-.9**step))/(np.sqrt(variance/(1-.999**step))+1e-8)
                np.testing.assert_allclose(m.read_buffer('layers.0.weight'), [weight], atol=2e-6)
                np.testing.assert_allclose(m.read_buffer('optim.adam.m.layers.0.weight'), [moment], atol=2e-6)
                np.testing.assert_allclose(m.read_buffer('optim.adam.v.layers.0.weight'), [variance], atol=2e-6)
            for lr in (-1, float('nan'), float('inf')):
                with pytest.raises((ValueError, RuntimeError)):
                    m.set_learning_rate(lr)
        finally:
            if restored is not None:
                restored.dispose()
            m.dispose()


@pytest.mark.parametrize('loss', ['mse', 'cross_entropy'])
@pytest.mark.parametrize('optimizer', ['sgd', 'adam'])
@pytest.mark.parametrize('device', ['CPU', 'CUDA'])
def test_tabm_predictions_gradients_and_updates_match_torch(loss, optimizer, device):
    if device == 'CUDA' and not pg._ffi.get_lib().poly_cuda_available():
        pytest.skip('poly_cuda_available() is false in the selected library')
    torch = pytest.importorskip('torch')
    torch.set_num_threads(1)
    with pg.create(device=device) as rt:
        model = rt.models.TabM(dict(layers=[2, 3, 3, 2], n_ensemble=3, batch_size=4, loss=loss))
        try:
            params = {model.param_name(i): torch.tensor(model.param_data(i).reshape(model.param_shape(i)), requires_grad=True)
                      for i in range(model.param_count)}
            opt = (torch.optim.SGD(params.values(), lr=.01) if optimizer == 'sgd'
                   else torch.optim.Adam(params.values(), lr=.01, eps=1e-8))
            model.set_optimizer(optimizer, lr=.01)
            for step, batch in enumerate((4, 1, 3)):
                x = np.arange(batch*2, dtype=np.float32).reshape(batch, 2)*.2 - .4
                y = np.tile(np.array([[0., 1.]], np.float32), (batch, 1))
                h = torch.tensor(x)[:, None, :].expand(-1, 3, -1)
                for layer in range(2):
                    p = f'layers.{layer}.'
                    h = torch.relu(((h*params[p+'r']) @ params[p+'weight'].T)*params[p+'s'] + params[p+'b'])
                members = (h.transpose(0, 1) @ params['output.weight']).transpose(0, 1) + params['output.bias']
                np.testing.assert_allclose(model.call('members', x=x)['members'], members.detach(), rtol=2e-5, atol=2e-6)
                target = torch.tensor(y)[:, None, :].expand_as(members)
                objective = ((members-target).square().mean() if loss == 'mse' else
                             -(target*members.log_softmax(-1)).sum(-1).mean())
                opt.zero_grad(); objective.backward()
                # SGD with a small LR exposes each parameter's gradient as its update;
                # Adam additionally exercises moments and bias powers across batches.
                expected_loss = objective.item()
                opt.step()
                actual_loss = model.train_step(x=x, y=y)
                np.testing.assert_allclose(actual_loss, expected_loss, rtol=2e-5, atol=2e-6)
                for name, value in params.items():
                    np.testing.assert_allclose(model.read_buffer(name).reshape(value.shape), value.detach(),
                                               rtol=3e-5, atol=3e-6, err_msg=f'{step}: {name}')
        finally:
            model.dispose()


@pytest.mark.parametrize('option', ['dropout', 'num_emb_type', 'n_ensembles'])
def test_tabm_rejects_unknown_options(option):
    with pg.create(device='CPU') as rt:
        with pytest.raises(ValueError, match=option):
            rt.models.TabM(dict(layers=[2, 3, 1], **{option: .5}))
