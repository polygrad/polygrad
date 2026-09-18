"""Executed with -I inside a fresh installed-package environment."""

from pathlib import Path
import hashlib
import sys

import numpy as np
import polygrad
from polygrad import Model, Tensor, _ffi
from polygrad.models import Sequential
from polygrad.helpers import Context
from polygrad.nn import LSTMCell, optim
from polygrad.uop.ops import KernelInfo, UOp
from polygrad.dtype import dtypes
from polygrad.llm.gguf import ggml_data_to_tensor


prefix = Path(sys.prefix).resolve()
assert Path(polygrad.__file__).resolve().is_relative_to(prefix), polygrad.__file__
assert Path(_ffi.get_lib()._name).resolve().is_relative_to(prefix), _ffi.get_lib()._name
assert Path(np.__file__).resolve().is_relative_to(prefix), np.__file__
print({'python': sys.version, 'numpy': np.__version__}, flush=True)
# Same IQ3_XXS block and pinned-output digest as test_gguf.py. Execute it on
# the minimum interpreter: parsing/importing alone misses newer int methods.
block = ((np.arange(98, dtype=np.uint16) * 37 + 11) & 0xFF).astype(np.uint8)
block[:2] = np.asarray([1.0], dtype=np.float16).view(np.uint8)
decoded = ggml_data_to_tensor(Tensor(block, dtype=dtypes.uint8, device='INTERP').realize(), 256, 18)
assert hashlib.sha256(decoded.numpy().tobytes()).hexdigest() == (
    '35aa99c61a8a85f22c5fc116e83dcc80299c6f20d820774a00f0cda44d08cdbb')
np.testing.assert_array_equal(Tensor([1, 2, 3]).mul(2).numpy(), [2, 4, 6])
import platform
if platform.machine().lower() in ('x86_64', 'amd64'):
    with polygrad.create(device='X86') as runtime:
        np.testing.assert_array_equal((runtime.Tensor([1., 2., 3.]) * 2).numpy(), [2, 4, 6])

def store_kernel(out, value, *, convert):
    i = UOp.range(out.ctx, 4, 0)
    value = value[i]
    if convert:
        value = value.cast('float32')
    return out[i].store(value).end(i).sink(arg=KernelInfo(name='package_store'))

for convert in (False, True):
    result = Tensor.empty(4).custom_kernel(
        Tensor([11, 22, 33, 44]), fxn=lambda out, value: store_kernel(out, value, convert=convert)
    )[0]
    try:
        values = result.numpy()
    except RuntimeError:
        assert not convert, 'explicit store cast must execute'
    else:
        assert convert, 'CPU vector STORE silently accepted mismatched dtypes'
        np.testing.assert_array_equal(values, [11, 22, 33, 44])

model = Sequential({'input': {'name': 'x', 'shape': [1], 'dtype': 'float32'},
                    'layers': [{'name': 'copy', 'type': 'identity'}], 'output': 'prediction'})
data = model.save_bundle(include_optimizer=False)
model.free()
restored = Model.from_bundle(data)
polygrad.clear_schedule_cache()
polygrad.collect()
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
       'tensor': True, 'model_bundle': True, 'lstm': True, 'muon': True, 'cache_clear': True,
       'gguf_iq3_xxs': True})
