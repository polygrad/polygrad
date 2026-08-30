#!/usr/bin/env python3
"""Executable, version-locked Tinygrad/Polygrad model compatibility cases."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

from tensor_graph_cases import canonical_graph


ENGINE = os.environ["ENGINE"]
if ENGINE == "tinygrad":
    import tinygrad as tinygrad
elif ENGINE == "polygrad":
    import polygrad as tinygrad
else:
    raise RuntimeError(f"unknown ENGINE={ENGINE!r}")

Tensor, TinyJit, nn = tinygrad.Tensor, tinygrad.TinyJit, tinygrad.nn
Context = tinygrad.Context
function = tinygrad.function
ROOT = Path(__file__).resolve().parents[1]
BEAUTIFUL_MNIST = ROOT / "references" / "tinygrad_latest" / "examples" / "beautiful_mnist.py"
CONVNEXT = ROOT / "references" / "tinygrad_latest" / "extra" / "models" / "convnext.py"


class MLP:
    def __init__(self):
        self.l0, self.l1 = nn.Linear(3, 4), nn.Linear(4, 2)

    def __call__(self, x):
        return self.l1(self.l0(x).relu())


class FunctionMLP(MLP):
    # Pinned examples/beautiful_mnist.py uses this value-producing FUNCTION
    # boundary around the model call. It must compose with backward and JIT.
    @function
    def __call__(self, x):
        return self.l1(self.l0(x).relu())


def set_value(tensor, value):
    tensor.assign(Tensor(np.asarray(value, dtype=np.float32), device="CPU")).realize()


def lists(values):
    return {name: value.tolist() for name, value in values.items()}


def set_mlp_values(model):
    set_value(model.l0.weight, [[0.2, -0.1, 0.3], [0.5, 0.4, -0.2],
                                [-0.3, 0.8, 0.1], [0.7, -0.6, 0.2]])
    set_value(model.l0.bias, [0.1, -0.2, 0.3, -0.4])
    set_value(model.l1.weight, [[0.6, -0.4, 0.2, 0.5], [-0.1, 0.7, -0.3, 0.4]])
    set_value(model.l1.bias, [0.05, -0.15])


def load_beautiful_mnist():
    # Execute the pinned example unchanged. Polygrad supplies the package names
    # the source imports; no translated model body is maintained here.
    if ENGINE == "polygrad":
        import polygrad.helpers
        import polygrad.nn
        import polygrad.nn.datasets
        sys.modules["tinygrad"] = tinygrad
        sys.modules["tinygrad.helpers"] = polygrad.helpers
        sys.modules["tinygrad.nn"] = polygrad.nn
        sys.modules["tinygrad.nn.datasets"] = polygrad.nn.datasets
    spec = importlib.util.spec_from_file_location(
        f"beautiful_mnist_compat_{ENGINE}", BEAUTIFUL_MNIST
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_convnext():
    # Execute the pinned model source unchanged. The alias table changes only
    # the selected package provider; the model body remains hash-bound.
    if ENGINE == "polygrad":
        import polygrad.helpers
        import polygrad.nn
        import polygrad.nn.state
        import polygrad.tensor
        sys.modules["tinygrad"] = tinygrad
        sys.modules["tinygrad.helpers"] = polygrad.helpers
        sys.modules["tinygrad.nn"] = polygrad.nn
        sys.modules["tinygrad.nn.state"] = polygrad.nn.state
        sys.modules["tinygrad.tensor"] = polygrad.tensor
    spec = importlib.util.spec_from_file_location(
        f"convnext_compat_{ENGINE}", CONVNEXT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def beautiful_mnist_state_value(name, shape):
    if name.endswith("num_batches_tracked"):
        return np.asarray(0, dtype=np.int64)
    lane = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    if name.endswith("running_var"):
        return 1.0 + (lane % 17) * np.float32(0.001)
    if name.endswith(".weight") and len(shape) == 1:
        return 0.9 + (lane % 11) * np.float32(0.002)
    name_bias = (sum(name.encode("utf-8")) % 13) - 6
    return ((lane % 31) - 15 + name_bias) * np.float32(0.001)


def sampled_state(value, count=64):
    flat = np.asarray(value).reshape(-1)
    if flat.size <= count:
        return flat.tolist()
    indices = np.linspace(0, flat.size - 1, count, dtype=np.int64)
    return flat[indices].tolist()


def case_mlp_mnist():
    Tensor.manual_seed(0)
    model = MLP()
    set_mlp_values(model)

    state = nn.state.get_state_dict(model)
    x_data = np.asarray([[1.0, -2.0, 0.5], [-0.5, 0.25, 2.0]], dtype=np.float32)
    target_data = np.asarray([[0.2, -0.4], [0.7, 0.1]], dtype=np.float32)
    x = Tensor(x_data, device="CPU").realize()
    target = Tensor(target_data, device="CPU").realize()
    forward = model(x)
    forward_graph = canonical_graph(forward.uop)
    forward_value = forward.numpy().copy()

    optimizer = nn.optim.SGD(nn.state.get_parameters(model), lr=0.05, fused=False)
    with Context(TRAINING=1):
        optimizer.zero_grad()
        loss = ((model(x) - target).square()).mean().backward()
        loss_value = float(loss.item())
        grads = {
            "l0.weight": model.l0.weight.grad.numpy().copy(),
            "l0.bias": model.l0.bias.grad.numpy().copy(),
            "l1.weight": model.l1.weight.grad.numpy().copy(),
            "l1.bias": model.l1.bias.grad.numpy().copy(),
        }
        optimizer.step()

    updated = {name: value.numpy().copy() for name, value in state.items()}

    @TinyJit
    def infer(inp):
        return model(inp).realize()

    jit_values = []
    for delta in (0.0, 0.25, -0.5):
        inp = Tensor(x_data + delta, device="CPU").realize()
        jit_values.append(infer(inp).numpy().copy())

    return {
        "surface": {
            "Tensor": Tensor is tinygrad.Tensor,
            "TinyJit": TinyJit is tinygrad.TinyJit,
            "Linear": hasattr(nn, "Linear"),
            "SGD": hasattr(nn.optim, "SGD"),
            "mnist": hasattr(nn.datasets, "mnist"),
        },
        "state_names": sorted(state),
        "state_shapes": {name: list(value.shape) for name, value in state.items()},
        "forward_graph": forward_graph,
        "forward": forward_value.tolist(),
        "loss": loss_value,
        "grads": lists(grads),
        "updated": lists(updated),
        "jit": [value.tolist() for value in jit_values],
        "jit_count": int(infer.cnt),
    }


def case_function_mlp():
    Tensor.manual_seed(0)
    model = FunctionMLP()
    set_mlp_values(model)
    state = nn.state.get_state_dict(model)
    x_data = np.asarray([[1.0, -2.0, 0.5], [-0.5, 0.25, 2.0]], dtype=np.float32)
    target_data = np.asarray([[0.2, -0.4], [0.7, 0.1]], dtype=np.float32)
    x = Tensor(x_data, device="CPU").realize()
    target = Tensor(target_data, device="CPU").realize()
    forward = model(x)
    forward_graph = canonical_graph(forward.uop)
    forward_value = forward.numpy().copy()

    optimizer = nn.optim.SGD(nn.state.get_parameters(model), lr=0.05, fused=False)
    with Context(TRAINING=1):
        optimizer.zero_grad()
        loss = ((model(x) - target).square()).mean().backward()
        loss_value = float(loss.item())
        grads = {
            "l0.weight": model.l0.weight.grad.numpy().copy(),
            "l0.bias": model.l0.bias.grad.numpy().copy(),
            "l1.weight": model.l1.weight.grad.numpy().copy(),
            "l1.bias": model.l1.bias.grad.numpy().copy(),
        }
        optimizer.step()
    updated = {name: value.numpy().copy() for name, value in state.items()}

    @TinyJit
    def infer(inp):
        return model(inp).realize()

    jit_values = []
    for delta in (0.0, 0.25, -0.5):
        inp = Tensor(x_data + delta, device="CPU").realize()
        jit_values.append(infer(inp).numpy().copy())

    return {
        "surface": {
            "Tensor": Tensor is tinygrad.Tensor,
            "TinyJit": TinyJit is tinygrad.TinyJit,
            "Linear": hasattr(nn, "Linear"),
            "SGD": hasattr(nn.optim, "SGD"),
            "function": hasattr(tinygrad, "function"),
        },
        "state_names": sorted(state),
        "state_shapes": {name: list(value.shape) for name, value in state.items()},
        "forward_graph": forward_graph,
        "forward": forward_value.tolist(),
        "loss": loss_value,
        "grads": lists(grads),
        "updated": lists(updated),
        "jit": [value.tolist() for value in jit_values],
        "jit_count": int(infer.cnt),
    }


def case_beautiful_mnist():
    """Unchanged pinned model: construct, forward, backward/update, JIT."""
    os.environ["BS"] = "1"
    Tensor.manual_seed(0)
    module = load_beautiful_mnist()
    model = module.Model()
    state = nn.state.get_state_dict(model)
    for name in sorted(state):
        value = beautiful_mnist_state_value(name, tuple(int(x) for x in state[name].shape))
        state[name].assign(Tensor(value, device="CPU")).realize()

    x_data = (((np.arange(2 * 28 * 28, dtype=np.float32) % 37) - 18) / 19).reshape(2, 1, 28, 28)
    y_data = np.asarray([3, 7], dtype=np.int32)
    x = Tensor(x_data, device="CPU").realize()
    y = Tensor(y_data, device="CPU").realize()
    forward = model(x)
    forward_graph = canonical_graph(forward.uop)
    forward_value = forward.numpy().copy()

    params = nn.state.get_parameters(model)
    module.opt = nn.optim.SGD(params, lr=0.001)
    Tensor.manual_seed(0)
    losses = [float(model.train_step(x, y).item()) for _ in range(3)]
    names_by_id = {id(value): name for name, value in state.items()}
    grad_surface = {
        name: {
            "present": param.grad is not None,
            "finite": param.grad is None or bool(np.isfinite(param.grad.numpy()).all()),
            "shape": [] if param.grad is None else list(param.grad.shape),
        }
        for name, param in zip(
            [names_by_id[id(param)] for param in params], params
        )
    }
    updated = {
        name: sampled_state(value.numpy()) for name, value in sorted(state.items())
    }

    return {
        "surface": {
            "source_sha256": hashlib.sha256(BEAUTIFUL_MNIST.read_bytes()).hexdigest(),
            "model": model.__class__.__name__ == "Model",
            "all_present_grads_finite": all(row["finite"] for row in grad_surface.values()),
        },
        "state_names": sorted(state),
        "state_shapes": {name: list(value.shape) for name, value in state.items()},
        "forward_graph": forward_graph,
        "forward": forward_value.tolist(),
        "loss": losses[0],
        "grads": grad_surface,
        "updated": updated,
        "jit": losses[1:],
        "jit_count": int(type(model).__dict__["train_step"].cnt),
    }


def case_convnext():
    """Unchanged pinned ConvNeXt: construct, forward, backward/update, JIT."""
    Tensor.manual_seed(0)
    module = load_convnext()
    model = module.ConvNeXt(
        in_chans=3, num_classes=5, depths=[1, 1, 1, 1], dims=[4, 8, 16, 32]
    )
    state = nn.state.get_state_dict(model)
    for name in sorted(state):
        shape = tuple(int(x) for x in state[name].shape)
        lane = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
        value = ((lane % 29) - 14 + (sum(name.encode()) % 7)) * np.float32(0.002)
        state[name].assign(Tensor(value.tolist(), device="CPU")).realize()

    x_data = (
        ((np.arange(3 * 32 * 32, dtype=np.float32) % 43) - 21)
        .reshape(1, 3, 32, 32) / 23
    )
    x = Tensor(x_data.tolist(), device="CPU")
    forward = model(x)
    forward_graph = canonical_graph(forward.uop)
    forward_value = forward.numpy().copy()

    params = nn.state.get_parameters(model)
    names_by_id = {id(value): name for name, value in state.items()}
    optimizer = nn.optim.SGD(params, lr=0.001, fused=False)
    target = Tensor([[0.1, -0.2, 0.3, -0.4, 0.5]], device="CPU")
    with Context(TRAINING=1):
        optimizer.zero_grad()
        loss = (model(Tensor(x_data.tolist(), device="CPU")) - target).square().mean().backward()
        loss_value = float(loss.item())
        grads = {
            names_by_id[id(param)]: sampled_state(param.grad.numpy(), 16)
            for param in params if param.grad is not None
        }
        optimizer.step()
    updated = {
        name: sampled_state(value.numpy(), 16)
        for name, value in sorted(state.items())
    }

    @TinyJit
    def infer(inp):
        return model(inp).realize()

    jit_values = []
    for delta in (0.0, 0.01, -0.02):
        inp = Tensor((x_data + delta).tolist(), device="CPU").realize()
        jit_values.append(infer(inp).numpy().copy())

    return {
        "surface": {
            "source_sha256": hashlib.sha256(CONVNEXT.read_bytes()).hexdigest(),
            "model": model.__class__.__name__ == "ConvNeXt",
            "LayerNorm2d": hasattr(nn, "LayerNorm2d"),
        },
        "state_names": sorted(state),
        "state_shapes": {name: list(value.shape) for name, value in state.items()},
        "forward_graph": forward_graph,
        "forward": forward_value.tolist(),
        "loss": loss_value,
        "grads": grads,
        "updated": updated,
        "jit": [value.tolist() for value in jit_values],
        "jit_count": int(infer.cnt),
    }


case_builders = {
    "beautiful_mnist": case_beautiful_mnist,
    "mlp_mnist": case_mlp_mnist,
    "function_mlp": case_function_mlp,
    "convnext": case_convnext,
}
selected_cases = os.environ.get(
    "COMPAT_CASES", "beautiful_mnist,mlp_mnist,function_mlp"
).split(",")
unknown_cases = [name for name in selected_cases if name not in case_builders]
if unknown_cases:
    raise RuntimeError(f"unknown compatibility cases: {unknown_cases}")

print(json.dumps({
    "schema_version": 1,
    "engine": ENGINE,
    "reference_commit": "a9069c177a9da9cca18593edf55acd2e6073cca6",
    "cases": {name: case_builders[name]() for name in selected_cases},
}, sort_keys=True))
