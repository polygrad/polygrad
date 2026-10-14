"""Real separately built C author, portable gradients and ownership."""

from pathlib import Path
import numpy as np
import pytest
from polygrad import create

LIB = Path(__file__).resolve().parents[2] / "build/extension/author.so"


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('kind,x,value,gradient', [
    (5, 8., 1.1224297172982928e-29, -1.8097067968559843e-28),
    (6, 0., 0., .886226925452758),
    (7, .5, 0., 2.5066282746310005),
    (8, 6., 1.7061176684318005, .1813229557371153),
    (9, 3.2, .8854048271549091, .9988388912865996),
    (10, -12., -75.4106730015688, 12.0822141752545),
])
def test_extension_probability_model(kind, x, value, gradient, dtype):
    with create(device='CPU', logical='always') as rt:
        author = rt.load_extension(LIB)
        input = rt.Tensor.empty(1, dtype=dtype).copy_from(np.array([x], dtype=dtype))
        outputs = []
        model = restored = None
        try:
            outputs = author.build([input, input, input], [kind])
            names = {'value': outputs[0], 'gradient': outputs[1]}
            if kind == 9:
                second, = outputs[1].sum().gradient(input)
                outputs.append(second)
                names['second'] = second
            model = rt.Model.from_tensors(inputs={'x': input}, outputs=names)
            restored = rt.Model.load(model.save())
            author.dispose()
            for m in (model, restored):
                result = m.call('forward', {'x': np.array([x], dtype=dtype)})
                tolerance = 4e-5 if dtype == 'float32' else 2e-10
                for name, expected in [('value', value), ('gradient', gradient)]:
                    np.testing.assert_allclose(result[name], [expected], rtol=tolerance,
                                               atol=0 if expected else tolerance)
                if kind == 9:
                    np.testing.assert_allclose(result['second'], [.3663211907314008],
                                               rtol=tolerance, atol=0)
        finally:
            if restored: restored.dispose()
            if model: model.dispose()
            for output in outputs: output.dispose()
            input.dispose()
            author.dispose()


def test_extension_bind_checks_contract_and_every_import():
    import ctypes as c
    from polygrad import _ffi
    author = c.CDLL(str(LIB))
    author.poly_extension_bind.argtypes = [c.c_void_p]
    author.poly_extension_bind.restype = c.c_int
    core = _ffi.get_lib()
    version = c.CFUNCTYPE(c.c_int)(lambda: 2)
    resolver_type = c.CFUNCTYPE(c.c_void_p, c.c_char_p)
    core_resolve = resolver_type(c.cast(core.poly_get_proc_address, c.c_void_p).value)
    seen = []
    def resolve(name):
        seen.append(name)
        if name == b'poly_extension_abi_version':
            return c.cast(version, c.c_void_p).value
        return core_resolve(name)
    resolver = resolver_type(resolve)
    assert author.poly_extension_bind(resolver) == 0
    assert seen == [b'poly_extension_abi_version']
    missing = resolver_type(lambda name: None if name == b'poly_tensor_release'
                            else core_resolve(name))
    assert author.poly_extension_bind(missing) == 0
    compatible = resolver_type(lambda name: None if name == b'poly_abi_version'
                               else core_resolve(name))
    assert author.poly_extension_bind(compatible) == 1


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
            cropped, pooled = extension.build([theta, X, y], [4])
            try:
                np.testing.assert_array_equal(cropped.numpy(), [2, 3])
                np.testing.assert_array_equal(pooled.numpy(), [[1], [2], [3]])
            finally:
                cropped.dispose(); pooled.dispose()
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
