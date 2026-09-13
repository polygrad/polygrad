"""Executed with -I inside a fresh installed-package environment."""

from pathlib import Path
import sys

import numpy as np
import polygrad
from polygrad import Model, Tensor, _ffi
from polygrad.models import Sequential
from polygrad.helpers import Context
from polygrad.nn import LSTMCell, optim


prefix = Path(sys.prefix).resolve()
assert Path(polygrad.__file__).resolve().is_relative_to(prefix), polygrad.__file__
assert Path(_ffi.get_lib()._name).resolve().is_relative_to(prefix), _ffi.get_lib()._name
assert Path(np.__file__).resolve().is_relative_to(prefix), np.__file__
np.testing.assert_array_equal(Tensor([1, 2, 3]).mul(2).numpy(), [2, 4, 6])
model = Sequential({'input': {'name': 'x', 'shape': [1], 'dtype': 'float32'},
                    'layers': [{'name': 'copy', 'type': 'identity'}], 'output': 'prediction'})
data = model.save_bundle(include_optimizer=False)
model.free()
restored = Model.from_bundle(data)
np.testing.assert_array_equal(restored.forward(x=np.array([7], np.float32))['prediction'], [7])
restored.free()
cell = LSTMCell(2, 2, bias=False)
cell.weight_ih = Tensor.zeros(8, 2)
cell.weight_hh = Tensor.zeros(8, 2)
h, c = cell(Tensor.ones(1, 2), (Tensor.zeros(1, 2), Tensor.ones(1, 2)))
np.testing.assert_allclose(c.numpy(), [[.5, .5]])
np.testing.assert_allclose(h.numpy(), [[.23105858, .23105858]], atol=1e-6)
p = Tensor([[1., 2.], [3., 4.]])
optimizer = optim.Muon([p], lr=.1, ns_steps=2)
with Context(TRAINING=1):
    for _ in range(2):
        p.grad = Tensor([[.1, -.2], [.3, -.4]])
        optimizer.step()
np.testing.assert_allclose(p.numpy(), [[1.03743243, 2.11009645], [2.77558279, 4.05183554]],
                           atol=2e-5, rtol=2e-5)
print({'package': polygrad.__file__, 'library': _ffi.get_lib()._name,
       'tensor': True, 'model_bundle': True, 'lstm': True, 'muon': True})
