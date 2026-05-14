"""Plain Python model object + nn.Linear + nn.Model.fit convenience.

Run from repo root:
  PYTHONPATH=py POLYGRAD_LIB=build/libpolygrad.so python py/examples/nn_linear_model_fit.py
"""

import numpy as np

from polygrad import Tensor, nn


class Net:
    def __init__(self):
        self.fc = nn.Linear(2, 1)
        self.fc.weight = Tensor([[1.0, 1.0]], requires_grad=True).realize()
        self.fc.bias = Tensor([0.0], requires_grad=True).realize()

    def __call__(self, x):
        return self.fc(x)


net = Net()
x = nn.Input("x", shape=(1, 2))
y = nn.Target("y", shape=(1, 1))
model = nn.Model.trace(
    net,
    inputs={"x": x},
    targets={"y": y},
    loss=lambda pred, target: (pred - target).square().mean(),
)

history = model.fit(
    {"x": np.array([[1.0, 2.0]], dtype=np.float32),
     "y": np.array([[4.0]], dtype=np.float32)},
    epochs=8,
    optimizer="sgd",
    lr=0.03,
)

print("loss", history[0], "->", history[-1])
print("forward", model.instance.forward(x=np.array([[1.0, 2.0]], dtype=np.float32))["output"])
