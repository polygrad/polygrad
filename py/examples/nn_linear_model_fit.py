"""Plain Python object + nn.Linear exported with Instance.from_tensors.

Run from repo root:
  PYTHONPATH=py POLYGRAD_LIB=build/libpolygrad.so python py/examples/nn_linear_model_fit.py
"""

import numpy as np

from polygrad import Instance, Tensor, nn


class Net:
    def __init__(self):
        self.fc = nn.Linear(2, 1)
        self.fc.weight = Tensor([[1.0, 1.0]]).realize()
        self.fc.bias = Tensor([0.0]).realize()

    def __call__(self, x):
        return self.fc(x)


net = Net()
x = Tensor.empty((1, 2))
y = Tensor.empty((1, 1))
pred = net(x)
loss = (pred - y).square().mean()
inst = Instance.from_tensors(
    inputs={"x": x},
    targets={"y": y},
    outputs={"output": pred},
    losses={"loss": loss},
    params={"fc.weight": net.fc.weight, "fc.bias": net.fc.bias},
)

history = inst.fit(
    {"x": np.array([[1.0, 2.0]], dtype=np.float32),
     "y": np.array([[4.0]], dtype=np.float32)},
    epochs=8,
    optimizer="sgd",
    lr=0.03,
)

print("loss", history[0], "->", history[-1])
print("forward", inst.forward(x=np.array([[1.0, 2.0]], dtype=np.float32))["output"])
