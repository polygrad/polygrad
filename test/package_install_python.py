"""Executed with -I inside a fresh installed-package environment."""

from pathlib import Path
import sys

import numpy as np
import polygrad
from polygrad import Model, Tensor, _ffi
from polygrad.models import Sequential


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
print({'package': polygrad.__file__, 'library': _ffi.get_lib()._name,
       'tensor': True, 'model_bundle': True})
