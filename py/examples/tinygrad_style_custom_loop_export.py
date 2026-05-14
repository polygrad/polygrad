"""Tinygrad-style custom training loop, then export named entrypoints.

Run from repo root:
  PYTHONPATH=py POLYGRAD_LIB=build/libpolygrad.so python py/examples/tinygrad_style_custom_loop_export.py
"""

import numpy as np

from polygrad import Tensor, nn


class LinearNet:
    def __init__(self):
        self.weight = Tensor([[1.0], [1.0]], requires_grad=True).realize()

    def __call__(self, x):
        return x.dot(self.weight)


model = LinearNet()
opt = nn.optim.Adam([model.weight], lr=0.05)
x = Tensor([[1.0, 2.0]])
y = Tensor([[4.0]])

with Tensor.train():
    losses = []
    for _ in range(6):
        opt.zero_grad()
        loss = (model(x) - y).square().mean()
        loss.backward()
        losses.append(loss.item())
        opt.step()

inp = nn.Input("x", shape=(1, 2))
inst = nn.Model.trace(model, inputs={"x": inp}).export()
out = inst.forward(x=np.array([[1.0, 2.0]], dtype=np.float32))
bundle = inst.save_bundle()

print("forward", out["output"])
print("loss", losses[0], "->", losses[-1])
print("bundle bytes", len(bundle))
