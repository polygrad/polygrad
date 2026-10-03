"""Live PyTorch/HF oracle; run with make test-vision-interpolate."""
import numpy as np
import pytest
import torch
import polygrad as pg


@pytest.mark.parametrize('shape,size,align', [
    ((1, 2, 3, 5), (7, 9), False), ((1, 1, 7, 9), (3, 4), False),
    ((1, 1, 3, 5), (1, 1), True), ((1, 1, 1, 3), (4, 7), True),
])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_bicubic_values_and_gradient(shape, size, align, dtype):
    values = np.linspace(-1, 1, np.prod(shape), dtype=dtype).reshape(shape)
    ref = torch.tensor(values, requires_grad=True)
    expected = torch.nn.functional.interpolate(ref, size=size, mode='bicubic', align_corners=align)
    (expected * expected).sum().backward()
    with pg.create(device='CPU') as rt:
        x = rt.Tensor(values)
        y = x.interpolate(size, mode='bicubic', align_corners=align)
        tol = 2e-12 if dtype is np.float64 else 3e-6
        np.testing.assert_allclose(y.numpy(), expected.detach().numpy(), atol=tol, rtol=tol)
        (y * y).sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), ref.grad.numpy(), atol=tol*10, rtol=tol*10)
