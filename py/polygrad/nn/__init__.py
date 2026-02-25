"""nn — Neural network modules for polygrad (tinygrad-compatible)."""

import ctypes
import numpy as np

from .modules import (
    Linear, LayerNorm, GroupNorm, RMSNorm, Embedding, Dropout,
    Conv2d, BatchNorm,
)
from .optim import Optimizer, SGD, Adam, AdamW
from .state import get_state_dict, load_state_dict, get_parameters
from .gpt2 import GPT2, Attention, FeedForward, TransformerBlock, GPT2_CONFIGS

__all__ = [
    'Linear', 'LayerNorm', 'GroupNorm', 'RMSNorm', 'Embedding', 'Dropout',
    'Conv2d', 'BatchNorm',
    'Optimizer', 'SGD', 'Adam', 'AdamW',
    'get_state_dict', 'load_state_dict', 'get_parameters',
    'GPT2', 'Attention', 'FeedForward', 'TransformerBlock', 'GPT2_CONFIGS',
    'compile_step',
]


def _collect_all_tensors(model, opt):
    """Walk model parameters + optimizer state, return all Tensor objects."""
    from ..tensor import Tensor
    result = []
    seen = set()

    def _add(t):
        if isinstance(t, Tensor) and id(t) not in seen:
            seen.add(id(t))
            result.append(t)

    for p in get_parameters(model):
        _add(p)
    for p in opt.params:
        _add(p)
    if hasattr(opt, 'velocities') and opt.velocities:
        for v in opt.velocities:
            if v is not None:
                _add(v)
    if hasattr(opt, 'm'):
        for m in opt.m:
            _add(m)
    if hasattr(opt, 'v'):
        for v in opt.v:
            _add(v)
    if hasattr(opt, '_bc1'):
        _add(opt._bc1)
    if hasattr(opt, '_bc2'):
        _add(opt._bc2)
    return result


def _pre_init_optimizer_state(opt, model):
    """Pre-initialize lazily-created optimizer state for compile_step tracing.

    SGD with momentum creates velocities on first use via Tensor(g.numpy()),
    which fails in compile mode since gradients are lazy. Pre-initialize
    velocities as zero tensors matching parameter shapes.
    """
    from ..tensor import Tensor
    if isinstance(opt, SGD) and opt.momentum and opt.velocities:
        for i, p in enumerate(opt.params):
            if opt.velocities[i] is None:
                opt.velocities[i] = Tensor.zeros(*p.shape)


def _snapshot_tensor(t):
    """Capture full tensor state for later restoration."""
    snap = {
        'uop': t._uop,
        'data': t._data,
        'buffer': t._buffer,
        'inputs': t._inputs[:],
        'grad': t._grad,
        'shape': t._shape,
        'requires_grad': t._requires_grad,
    }
    if hasattr(t, '_assign_data'):
        snap['assign_data'] = t._assign_data
        snap['assign_buffer'] = t._assign_buffer
    if hasattr(t, '_saved_uop'):
        snap['saved_uop'] = t._saved_uop
        snap['saved_inputs'] = t._saved_inputs
    return snap


def _restore_tensor(t, snap):
    """Restore tensor state from snapshot."""
    t._uop = snap['uop']
    t._data = snap['data']
    t._buffer = snap['buffer']
    t._inputs = snap['inputs']
    t._grad = snap['grad']
    t._shape = snap['shape']
    t._requires_grad = snap['requires_grad']
    if 'assign_data' in snap:
        t._assign_data = snap['assign_data']
        t._assign_buffer = snap['assign_buffer']
    else:
        if hasattr(t, '_assign_data'):
            del t._assign_data
            del t._assign_buffer
    if 'saved_uop' in snap:
        t._saved_uop = snap['saved_uop']
        t._saved_inputs = snap['saved_inputs']


class CompiledTrainingStep:
    """Compiled training step with loss output.

    Wraps a C PolyStep* compiled from the full forward+backward+optimizer
    graph. Call run() to execute one training step. Read loss_value() for
    the current loss after run().
    """
    def __init__(self, step_ptr, buf_bindings, loss_data):
        from ..tensor import CompiledStep
        self._inner = CompiledStep(step_ptr, buf_bindings)
        self._loss_data = loss_data

    @property
    def n_kernels(self):
        return self._inner.n_kernels

    @property
    def n_intermediates(self):
        return self._inner.n_intermediates

    def run(self):
        """Execute one training step (forward + backward + optimizer)."""
        self._inner.run()

    def loss_value(self):
        """Return the loss value from the most recent run()."""
        return float(self._loss_data[0])


def compile_step(step_fn, model, opt, *sample_inputs):
    """Compile a training step into a reusable PolyStep.

    step_fn(model, opt, *inputs) should run forward + backward + optimizer
    step and return the loss tensor. Example::

        def train_step(model, opt, x, y):
            loss = model(x).cross_entropy(y)
            loss.backward()
            opt.step()
            opt.zero_grad()
            return loss

        step = compile_step(train_step, model, opt, sample_x, sample_y)
        for x_batch, y_batch in data:
            x_tensor._data[:] = x_batch
            y_tensor._data[:] = y_batch
            step.run()
            print(step.loss_value())

    Args:
        step_fn: Training step function returning the loss tensor.
        model: The model object (parameters are collected from it).
        opt: The optimizer.
        *sample_inputs: Sample Tensor inputs for tracing the computation.

    Returns:
        CompiledTrainingStep with run() and loss_value() methods.
    """
    from .. import _ffi
    from ..tensor import Tensor, CompiledStep

    # Pre-initialize optimizer state that would be lazily created
    _pre_init_optimizer_state(opt, model)

    # Collect all tensors involved
    all_tensors = _collect_all_tensors(model, opt)
    all_ids = set(id(t) for t in all_tensors)
    for inp in sample_inputs:
        if isinstance(inp, Tensor) and id(inp) not in all_ids:
            all_ids.add(id(inp))
            all_tensors.append(inp)

    # Snapshot all tensor states (for restoration after tracing)
    snapshots = {}
    for t in all_tensors:
        snapshots[id(t)] = _snapshot_tensor(t)

    # Trace the step in compile mode.
    # assign().realize() in compile mode does pseudo-realize: saves ASSIGN UOps
    # in program order to _compile_assigns_ordered, restores tensor to buffer
    # state so subsequent ops (m_hat = m * bc1) use buffer UOps not ASSIGN UOps.
    Tensor._compile_assigns_ordered = []
    Tensor._compile_mode = True
    try:
        loss = step_fn(model, opt, *sample_inputs)
    except Exception:
        Tensor._compile_mode = False
        Tensor._compile_assigns_ordered = []
        for t in all_tensors:
            if id(t) in snapshots:
                _restore_tensor(t, snapshots[id(t)])
        raise
    Tensor._compile_mode = False

    ctx = loss._ctx

    # Collect ASSIGNs in program order (the order assign().realize() was called)
    sink_srcs = list(Tensor._compile_assigns_ordered)
    Tensor._compile_assigns_ordered = []

    if not sink_srcs:
        for t in all_tensors:
            if id(t) in snapshots:
                _restore_tensor(t, snapshots[id(t)])
        raise RuntimeError('compile_step: no ASSIGN ops found (optimizer must use assign)')

    # Add loss output: STORE(loss_buf, loss_uop)
    numel = int(np.prod(loss._shape)) if loss._shape else 1
    loss_buf = _ffi._lib.poly_buffer_f32(ctx, numel)
    loss_data = np.zeros(numel, dtype=np.float32)
    loss_store = _ffi._lib.poly_store_val(ctx, loss_buf, loss._uop)
    sink_srcs.append(loss_store)

    # Build SINK
    n_srcs = len(sink_srcs)
    src_arr = (ctypes.c_void_p * n_srcs)(*sink_srcs)
    sink = _ffi._lib.poly_sink_n(ctx, src_arr, n_srcs)

    # Compile
    step_ptr = _ffi._lib.poly_compile_step(ctx, sink)
    if not step_ptr:
        for t in all_tensors:
            if id(t) in snapshots:
                _restore_tensor(t, snapshots[id(t)])
        raise RuntimeError('poly_compile_step failed')

    # Build buffer bindings from tracked tensors (model params, optimizer
    # state, inputs). All buffer-backed tensors should be in all_tensors.
    # Optimizer scalar constants (lr, betas, eps) are CONST UOps, not buffers.
    buf_bindings = []
    seen_bufs = set()

    def _add_binding(buf, holder):
        if buf and id(buf) not in seen_bufs:
            seen_bufs.add(id(buf))
            buf_bindings.append((buf, holder))

    for t in all_tensors:
        snap = snapshots[id(t)]
        if snap['data'] is not None:
            _add_binding(snap['buffer'], t)

    # Loss output buffer
    _add_binding(loss_buf, loss_data)

    # Restore all tensor states
    for t in all_tensors:
        if id(t) in snapshots:
            _restore_tensor(t, snapshots[id(t)])

    return CompiledTrainingStep(step_ptr, buf_bindings, loss_data)
