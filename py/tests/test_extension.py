"""Real separately built C author, portable gradients and ownership."""

from pathlib import Path
import numpy as np
import pytest
from polygrad import create

LIB = Path(__file__).resolve().parents[2] / "build/extension/author.so"


def test_extension_model_gradient_and_lifetime():
    assert LIB.is_file(), "run make extension-fixture"
    with create(device="CPU", logical="always") as rt, create(device="CPU") as other:
        extension = rt.load_extension(LIB)
        theta = rt.Tensor.empty(3).copy_from(np.array([1, 2, 3], np.float32))
        X = rt.Tensor.empty(6, 3).copy_from(np.zeros((6, 3), np.float32))
        y = rt.Tensor.empty(6).copy_from(np.zeros(6, np.float32))
        foreign = other.Tensor([1.0])
        try:
            with pytest.raises(ValueError, match="runtime"):
                extension.build([foreign, X, y], [0])
            with pytest.raises(ValueError, match="scalar"):
                extension.build([theta, X, y], [float("nan")])
            with pytest.raises(RuntimeError, match="construction failed"):
                extension.build([theta, X, y], [2])
            np.testing.assert_array_equal(theta.numpy(), [1, 2, 3])
            unsorted = rt.Tensor(np.array([2, -1, 2, 0], np.float32))
            values, indices = extension.build([unsorted, X, y], [3])
            try:
                np.testing.assert_array_equal(values.numpy(), [-1, 0, 2, 2])
                np.testing.assert_array_equal(indices.numpy(), [1, 3, 0, 2])
            finally:
                values.dispose(); indices.dispose(); unsorted.dispose()
            logp, gradient = extension.build([theta, X, y], [0])
            model = rt.Model.from_tensors(
                inputs={"theta": theta}, outputs={"logp": logp, "gradient": gradient}
            )
            extension.dispose()
            with pytest.raises(RuntimeError, match="disposed"):
                extension.build([theta, X, y], [0])
            logp.dispose()
            gradient.dispose()
            restored = rt.Model.load(model.save())
            try:
                for m in (model, restored):
                    got = m.call("forward", {"theta": np.array([-4, 0, 7], np.float32)})
                    np.testing.assert_array_equal(got["gradient"], [4, 0, -7])
                    assert float(got["logp"]) == -32.5
            finally:
                restored.dispose()
                model.dispose()
        finally:
            for t in (theta, X, y, foreign):
                t.dispose()
            extension.dispose()
        rt.clear_schedule_cache()
        rt.collect()
        assert rt.stats()["buffer_owned_bytes"] == 0
        assert rt.stats()["tensor_records"] == 0
    with pytest.raises(RuntimeError, match="disposed"):
        extension.build([], [])
