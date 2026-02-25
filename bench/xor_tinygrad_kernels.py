"""Dump tinygrad kernel structure for XOR training step — symbolic parity reference.
Uses CLANG backend for C code comparison with polygrad."""
import os
os.environ["CLANG"] = "1"  # Force CPU/C backend

from tinygrad import Tensor, nn, dtypes, Device
from tinygrad.nn.optim import SGD
from tinygrad.engine.schedule import create_schedule
import numpy as np

Tensor.training = True

# Fixed weights for reproducible comparison
x_data = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
y_data = np.array([[0],[1],[1],[0]], dtype=np.float32)

# Model with fixed weights (skip random init to simplify comparison)
l1_w = Tensor(np.ones((8,2), dtype=np.float32) * 0.1)
l1_b = Tensor(np.zeros(8, dtype=np.float32))
l2_w = Tensor(np.ones((1,8), dtype=np.float32) * 0.1)
l2_b = Tensor(np.zeros(1, dtype=np.float32))

l1_w.requires_grad = True
l1_b.requires_grad = True
l2_w.requires_grad = True
l2_b.requires_grad = True

x = Tensor(x_data)
y = Tensor(y_data)

# Forward: x @ w1.T + b1 → relu → @ w2.T + b2 → sigmoid → MSE
h = (x @ l1_w.T + l1_b).relu()
out = (h @ l2_w.T + l2_b).sigmoid()
loss = ((out - y) ** 2).mean()

# Compute gradients
loss.backward()

# Show gradient shapes
print(f"l1_w.grad shape: {l1_w.grad.shape}")
print(f"l1_b.grad shape: {l1_b.grad.shape}")
print(f"l2_w.grad shape: {l2_w.grad.shape}")
print(f"l2_b.grad shape: {l2_b.grad.shape}")

# SGD update: param = param - lr * grad
lr = 0.5
params_updated = [
    l1_w - lr * l1_w.grad,
    l1_b - lr * l1_b.grad,
    l2_w - lr * l2_w.grad,
    l2_b - lr * l2_b.grad,
]

# Schedule all outputs together (like tinygrad's optimizer.step())
print("\n=== Scheduling realize for loss + updated params ===")
all_tensors = [loss] + params_updated

# Use DEBUG to see scheduled kernels
os.environ["DEBUG"] = "4"
Tensor.realize(*all_tensors)

print(f"\nLoss: {loss.item():.6f}")
for i, (name, p) in enumerate(zip(["l1_w","l1_b","l2_w","l2_b"], params_updated)):
    print(f"{name} updated: {p.numpy().flatten()[:4]}...")
