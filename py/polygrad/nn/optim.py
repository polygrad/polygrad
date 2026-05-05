"""nn.optim -- thin frontend wrappers over the C optimizer graph builders."""

from .. import _ffi
from ..tensor import Tensor, _ptr_value


OPTIM_SGD = 1
OPTIM_ADAM = 2
OPTIM_ADAMW = 3


def _dedup(items):
    out = []
    seen = set()
    for x in items:
        k = id(x)
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out


def _ptr_array(items):
    if not items:
        return None
    return (_ffi._ptr * len(items))(*[x._tensor for x in items])


def _realize_all(items):
    if items:
        items[0].realize(*items[1:])


class Optimizer:
    """Base optimizer class.

    Python owns the parameter/state Tensor handles. The update math is built by
    src/optim.c so Instance training and standalone frontend optimizers share
    one optimizer implementation.
    """

    def __init__(self, params, lr=0.001, device=None):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        params = list(params)
        for p in params:
            if p.requires_grad is None:
                p.requires_grad_(True)
        self.params = _dedup([p for p in params if p.requires_grad])
        if not self.params:
            raise AssertionError("optimizer must have at least one param")
        self.buffers = _dedup([p for p in params if not p.requires_grad])
        self.lr = float(lr)
        self.device = device or self.params[0].device
        self._ctx = self.params[0]._ctx

    def zero_grad(self):
        for p in self.params:
            p._grad = None

    def _config(self):
        raise NotImplementedError

    def _state_args(self):
        return None, None, None, None, []

    def schedule_step(self):
        grads = []
        for p in self.params:
            if p.grad is None:
                raise RuntimeError("optimizer parameter has no gradient")
            grads.append(p.grad)

        m_state, v_state, bc1, bc2, state_tensors = self._state_args()
        cfg = self._config()
        params_arr = _ptr_array(self.params)
        grads_arr = _ptr_array(grads)
        m_arr = _ptr_array(m_state)
        v_arr = _ptr_array(v_state)

        needed = _ffi._lib.poly_optim_build_step(
            self._ctx, cfg, params_arr, grads_arr, len(self.params),
            m_arr, v_arr, bc1._tensor if bc1 is not None else None,
            bc2._tensor if bc2 is not None else None, None, 0,
        )
        if needed < 0:
            raise RuntimeError("optimizer step graph build failed")

        out_arr = (_ffi._ptr * needed)()
        rc = _ffi._lib.poly_optim_build_step(
            self._ctx, cfg, params_arr, grads_arr, len(self.params),
            m_arr, v_arr, bc1._tensor if bc1 is not None else None,
            bc2._tensor if bc2 is not None else None, out_arr, needed,
        )
        if rc < 0:
            raise RuntimeError("optimizer step graph build failed")

        by_ptr = {_ptr_value(t._tensor): t for t in self.params + state_tensors}
        scheduled = []
        for i in range(rc):
            t = by_ptr.get(_ptr_value(out_arr[i]))
            if t is None:
                raise RuntimeError("optimizer step graph returned an unknown tensor")
            scheduled.append(t)
        return scheduled + self.buffers

    def step(self):
        _realize_all(self.schedule_step())


class OptimizerGroup:
    """Combine multiple optimizers behind tinygrad's OptimizerGroup API."""

    def __init__(self, *optimizers):
        self.optimizers = optimizers
        self.params = [p for o in optimizers for p in o.params]
        self.buffers = [b for o in optimizers for b in o.buffers]

    def __getitem__(self, i):
        return self.optimizers[i]

    def zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def schedule_step(self):
        return [x for opt in self.optimizers for x in opt.schedule_step()]

    def step(self):
        _realize_all(self.schedule_step())


class SGD(Optimizer):
    """Stochastic gradient descent with tinygrad-compatible momentum options."""

    def __init__(
        self,
        params,
        lr=0.001,
        momentum=0.0,
        weight_decay=0.0,
        nesterov=False,
        classic=False,
        device=None,
        fused=False,
    ):
        if fused:
            raise NotImplementedError("fused optimizers are not implemented in Polygrad yet")
        if momentum < 0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        super().__init__(params, lr, device=device)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.nesterov = bool(nesterov)
        self.classic = bool(classic)
        self.b = [
            Tensor.zeros(p.numel(), dtype="float32", device=self.device, _ctx=self._ctx, requires_grad=False).realize()
            for p in self.params
        ] if self.momentum else []
        self.velocities = self.b

    def _config(self):
        return _ffi.PolyOptimConfig(
            OPTIM_SGD, self.lr, 0.0, 0.0, 0.0, self.weight_decay,
            self.momentum, self.nesterov, self.classic,
        )

    def _state_args(self):
        return (self.b if self.momentum else None), None, None, None, self.b


class Adam(Optimizer):
    """Adam optimizer using core-built beta-power and moment update graphs."""

    def __init__(
        self,
        params,
        lr=0.001,
        betas=None,
        eps=1e-8,
        weight_decay=0.0,
        b1=0.9,
        b2=0.999,
        device=None,
        fused=False,
    ):
        if fused:
            raise NotImplementedError("fused optimizers are not implemented in Polygrad yet")
        if weight_decay:
            raise ValueError("Adam weight_decay is not tinygrad-compatible; use AdamW")
        if betas is not None:
            b1, b2 = betas
        super().__init__(params, lr, device=device)
        self.b1, self.b2 = float(b1), float(b2)
        self.eps = float(eps)
        self.weight_decay = 0.0
        self.m = [
            Tensor.zeros(p.numel(), dtype="float32", device=self.device, _ctx=self._ctx, requires_grad=False).realize()
            for p in self.params
        ]
        self.v = [
            Tensor.zeros(p.numel(), dtype="float32", device=self.device, _ctx=self._ctx, requires_grad=False).realize()
            for p in self.params
        ]
        self.b1_t = Tensor.ones(1, dtype="float32", device=self.device, _ctx=self._ctx, requires_grad=False).realize()
        self.b2_t = Tensor.ones(1, dtype="float32", device=self.device, _ctx=self._ctx, requires_grad=False).realize()
        self._bc1 = self.b1_t
        self._bc2 = self.b2_t

    def _kind(self):
        return OPTIM_ADAM

    def _config(self):
        return _ffi.PolyOptimConfig(
            self._kind(), self.lr, self.b1, self.b2, self.eps,
            self.weight_decay, 0.0, False, False,
        )

    def _state_args(self):
        return self.m, self.v, self.b1_t, self.b2_t, self.m + self.v + [self.b1_t, self.b2_t]


class AdamW(Adam):
    """AdamW optimizer with decoupled weight decay, built in the C core."""

    def __init__(
        self,
        params,
        lr=0.001,
        betas=None,
        eps=1e-8,
        weight_decay=0.01,
        b1=0.9,
        b2=0.999,
        device=None,
        fused=False,
    ):
        super().__init__(
            params, lr=lr, betas=betas, eps=eps, weight_decay=0.0,
            b1=b1, b2=b2, device=device, fused=fused,
        )
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        self.weight_decay = float(weight_decay)

    def _kind(self):
        return OPTIM_ADAMW
