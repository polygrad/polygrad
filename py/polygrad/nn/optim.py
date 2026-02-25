"""nn.optim — Optimizers for polygrad (tinygrad-compatible)."""

import math
from ..tensor import Tensor


class Optimizer:
    """Base optimizer class."""
    def __init__(self, params, lr=0.001):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p._grad = None

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent with optional momentum."""
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = [None] * len(self.params) if momentum else None

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay:
                g = g + p * self.weight_decay

            if self.momentum:
                if self.velocities[i] is None:
                    self.velocities[i] = Tensor(g.numpy())
                else:
                    self.velocities[i].assign(self.velocities[i] * self.momentum + g).realize()
                g = self.velocities[i]

            p.assign(p - g * self.lr).realize()


class Adam(Optimizer):
    """Adam optimizer."""
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr)
        self.b1, self.b2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [Tensor.zeros(*p.shape) for p in self.params]
        self.v = [Tensor.zeros(*p.shape) for p in self.params]
        self.t = 0
        # Scalar buffer tensors for bias corrections — keeps the UOp graph
        # structure step-invariant so compiled kernels cache across steps.
        self._bc1 = Tensor([1.0]).realize()
        self._bc2 = Tensor([1.0]).realize()

    def step(self):
        self.t += 1
        # Update bias correction buffers (data only, same UOp structure)
        self._bc1._data[0] = 1.0 / (1 - self.b1 ** self.t)
        self._bc2._data[0] = 1.0 / (1 - self.b2 ** self.t)

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay:
                g = g + p * self.weight_decay

            # Update biased first and second moment estimates
            self.m[i].assign(self.m[i] * self.b1 + g * (1 - self.b1)).realize()
            self.v[i].assign(self.v[i] * self.b2 + (g * g) * (1 - self.b2)).realize()

            # Bias correction via buffer tensors (graph structure is step-invariant)
            m_hat = self.m[i] * self._bc1
            v_hat = self.v[i] * self._bc2

            # Update
            p.assign(p - m_hat * self.lr / (v_hat.sqrt() + self.eps)).realize()


class AdamW(Adam):
    """AdamW optimizer (decoupled weight decay)."""
    def step(self):
        self.t += 1
        # Update bias correction buffers (data only, same UOp structure)
        self._bc1._data[0] = 1.0 / (1 - self.b1 ** self.t)
        self._bc2._data[0] = 1.0 / (1 - self.b2 ** self.t)

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad

            # Decoupled weight decay
            if self.weight_decay:
                p.assign(p * (1 - self.lr * self.weight_decay)).realize()

            self.m[i].assign(self.m[i] * self.b1 + g * (1 - self.b1)).realize()
            self.v[i].assign(self.v[i] * self.b2 + (g * g) * (1 - self.b2)).realize()

            # Bias correction via buffer tensors (graph structure is step-invariant)
            m_hat = self.m[i] * self._bc1
            v_hat = self.v[i] * self._bc2

            p.assign(p - m_hat * self.lr / (v_hat.sqrt() + self.eps)).realize()
