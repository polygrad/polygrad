import numpy as np
import pytest

from polygrad import Tensor, Model


def test_sample_logits_threshold_ties_and_rng():
    logits = Tensor(np.tile([2.0, 2.0, 2.0, -20.0], (1024, 1)).astype(np.float32))
    Tensor.manual_seed(321)
    first = logits.sample_logits(temperature=1.0, top_k=2).numpy()
    Tensor.manual_seed(321)
    np.testing.assert_array_equal(first, logits.sample_logits(top_k=2).numpy())
    second = logits.sample_logits(top_k=2).numpy()
    assert first.shape == (1024,)
    assert np.any(first != second)
    counts = np.bincount(first, minlength=4)
    assert counts[3] == 0
    assert np.all((counts[:3] > 260) & (counts[:3] < 420))


def test_sample_logits_matches_existing_transformer_formula():
    logits = Tensor(np.arange(64, dtype=np.float32).reshape(8, 8) / 16)
    Tensor.manual_seed(42)
    actual = logits.sample_logits(temperature=0.7, keepdim=True).numpy()
    Tensor.manual_seed(42)
    u = Tensor.rand(8, 8).maximum(1e-12)
    expected = (logits / 0.7 - (-u.log()).log()).argmax(-1, keepdim=True).numpy()
    np.testing.assert_array_equal(actual, expected)


def test_sample_logits_categorical_distribution():
    # Independent categorical probabilities, not a second implementation of
    # Threefry/Gumbel. Fixed seed makes the broad statistical bound repeatable.
    values = np.array([-1.0, 0.1, 0.4, 0.9, 0.9], np.float32)
    temperature = 0.7
    eligible = values >= np.sort(values)[-3]
    probability = np.exp((values - values.max()) / temperature) * eligible
    probability /= probability.sum()
    Tensor.manual_seed(119)
    sampled = Tensor(np.tile(values, (8192, 1))).sample_logits(
        temperature=temperature, top_k=3).numpy()
    observed = np.bincount(sampled, minlength=len(values))
    assert not observed[~eligible].any()
    expected = probability[eligible] * len(sampled)
    chi_square = np.sum((observed[eligible] - expected) ** 2 / expected)
    assert chi_square < 25


def test_sample_logits_model_rng_checkpoint():
    Tensor.manual_seed(42)
    x = Tensor.empty(32, 8)
    model = Model(lambda x: x.sample_logits(top_k=3), inputs={'x': x})
    restored = None
    try:
        data = np.zeros((32, 8), np.float32)
        first = model.forward(x=data)['output']
        restored = Model.load(model.save())
        second = model.forward(x=data)['output']
        np.testing.assert_array_equal(second, restored.forward(x=data)['output'])
        assert np.any(first != second)
    finally:
        if restored:
            restored.dispose()
        model.dispose()


@pytest.mark.parametrize('top_k', [-1, 5, 1.5])
def test_sample_logits_rejects_invalid_k(top_k):
    with pytest.raises((ValueError, TypeError)):
        Tensor([1.0, 2.0, 3.0, 4.0]).sample_logits(top_k=top_k)
