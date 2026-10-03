"""Live PyTorch/HF oracle; run with make test-vision-interpolate."""
import json
import numpy as np
import pytest
import torch
from safetensors.torch import save
from transformers import Dinov2Config, Dinov2Model
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


def test_dinov2_input_resolution_preserves_checkpoint_and_bundle():
    torch.manual_seed(3)
    config = Dinov2Config(hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
                         image_size=12, patch_size=2)
    config._attn_implementation = 'eager'
    model = Dinov2Model(config).eval()
    pixels = np.linspace(-1, 1, 3*8*8, dtype=np.float32).reshape(1, 3, 8, 8)
    with torch.no_grad():
        expected = model(torch.tensor(pixels))
    weights = save({k: v.contiguous() for k, v in model.state_dict().items()})
    cfg = config.to_dict()
    cfg['input_image_size'] = 8
    with pg.create(device='CPU') as rt:
        m = pg.Model.from_hf(config_json=json.dumps(cfg), weight_bytes_list=[weights], runtime=rt)
        restored = None
        try:
            outputs = m.forward(pixel_values=pixels)
            for name in ('last_hidden_state', 'pooler_output'):
                np.testing.assert_allclose(outputs[name], getattr(expected, name).numpy(), atol=5e-5, rtol=5e-4)
            restored = rt.Model.load(m.save())
            for name, value in restored.forward(pixel_values=pixels).items():
                np.testing.assert_array_equal(value, outputs[name])
        finally:
            if restored is not None:
                restored.dispose()
            m.dispose()
