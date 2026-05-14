"""Config-based MLP instance training.

This is the family-template path used by built-in/imported models.

Run from repo root:
  PYTHONPATH=py POLYGRAD_LIB=build/libpolygrad.so python py/examples/config_mlp_fit.py
"""

import numpy as np

from polygrad.instance import Instance


inst = Instance.mlp({
    "layers": [2, 4, 1],
    "activation": "relu",
    "bias": True,
    "loss": "mse",
    "batch_size": 1,
    "seed": 42,
})

history = inst.fit(
    {"x": np.array([[1.0, 2.0]], dtype=np.float32),
     "y": np.array([[4.0]], dtype=np.float32)},
    epochs=12,
    optimizer="sgd",
    lr=0.03,
)

print("loss", history[0], "->", history[-1])
print("output", inst.forward(x=np.array([[1.0, 2.0]], dtype=np.float32))["output"])
