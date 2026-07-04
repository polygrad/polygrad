# Polygrad Python

Python bindings for Polygrad, a C11 port of tinygrad's compiler core.

The Python frontend exposes a lazy Tensor API with autograd, JIT
capture/replay, neural network layers, structured linalg helpers, and model
loading utilities.

## Install

```bash
pip install polygrad
```

Requirements:

- Linux
- Python 3.9 or newer
- NumPy
- A C compiler and Python development headers for source builds
- `clang` on `PATH` for the CPU runtime

Optional:

```bash
pip install huggingface_hub
```

From this repository:

```bash
make
POLYGRAD_LIB=$PWD/build/libpolygrad.so PYTHONPATH=py python - <<'PY'
from polygrad import Tensor
print((Tensor([1, 2, 3]) * 2 + 1).numpy())
PY
```

## Quick Start

```python
from polygrad import Tensor

a = Tensor.rand(3, 4)
b = Tensor.rand(4, 5)
c = (a @ b).softmax(-1)

print(c.numpy())
```

Autograd:

```python
from polygrad import Tensor

x = Tensor([1.0, 2.0, 3.0])
x.requires_grad = True

loss = (x * x).sum()
loss.backward()

print(x.grad.numpy())  # [2.0, 4.0, 6.0]
```

Training:

```python
from polygrad import Tensor
from polygrad.nn import Linear, SGD, get_parameters

Tensor.manual_seed(42)
model = Linear(2, 1)
opt = SGD(get_parameters(model), lr=0.01)

for _ in range(100):
    opt.zero_grad()
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])
    y = Tensor([[5.0], [11.0]])
    loss = (model(x) - y).square().mean()
    loss.backward()
    opt.step()

print(loss.item())
```

## Devices

```python
from polygrad import Device, Tensor

x = Tensor.rand(4)

if Device.cuda_available():
    y = (x * 2).to("cuda")
else:
    y = (x * 2).to("cpu")

print(y.numpy())
```

Environment variables:

```bash
POLY_DEVICE=cpu|cuda|hip|x86|interp
POLY_DUMP_KERNELS=1
POLY_BEAM=4
```

## JIT

`@jit` follows tinygrad's raw Tensor JIT behavior. The first call runs normally,
the second call captures realized schedules, and later calls replay those
schedules with current input buffers.

```python
from polygrad import Tensor, jit

@jit
def step(x):
    return (x + 1).realize()

print(step(Tensor([1, 2, 3])).numpy())  # normal run
print(step(Tensor([4, 5, 6])).numpy())  # capture
print(step(Tensor([7, 8, 9])).numpy())  # replay
```

For embedding loops, `compile(...)` performs the same warmup and capture up
front and exposes an explicit callable:

```python
from polygrad import Tensor, compile

def step(x):
    return (x + 1).realize()

compiled = compile(step, [Tensor([1, 2, 3]).realize()])
out = compiled.run([Tensor([7, 8, 9]).realize()])

print(out.numpy())
print(compiled.stats())
compiled.dispose()
```

`polygrad.stats()` exposes shared C runtime counters. `Jit.stats()` and
`CompiledCallable.stats()` expose wrapper-level capture and replay counters.

`polygrad.can_run(op, dtype="float32", shape=..., shapes=..., device="auto")`
is an advisory backend capability probe.

## Custom Kernels

`Tensor.custom_kernel(...)` mirrors tinygrad's alpha custom-kernel shape. The
kernel function receives placeholder UOps and returns a `SINK` body. Polygrad
wraps the body in `CALL`, returns `AFTER(...)` tensors, and keeps execution in
the normal schedule and runtime caches.

```python
from polygrad import Tensor
from polygrad.uop.ops import UOp

def add_kernel(out, a, b):
    out, a, b = out.flatten(), a.flatten(), b.flatten()
    i = UOp.range(out.ctx, out.numel(), 0)
    return out[i].store(a[i] + b[i]).end(i).sink()

out = Tensor.empty((4,), dtype="float32")
y = out.custom_kernel(
    Tensor([1, 2, 3, 4]),
    Tensor([10, 20, 30, 40]),
    fxn=add_kernel,
)[0]
print(y.numpy())
```

This is a UOp `CALL` extension point, not a raw program-launch API. Custom
backward functions are not implemented yet.

## Tensor API Summary

Construction:

| Method | Description |
|---|---|
| `Tensor(data)` | From list, NumPy array, or scalar |
| `Tensor.zeros(*shape)` | Tensor of zeros |
| `Tensor.ones(*shape)` | Tensor of ones |
| `Tensor.full(shape, val)` | Tensor filled with value |
| `Tensor.rand(*shape)` | Uniform random values in `[0, 1)` |
| `Tensor.randn(*shape)` | Standard normal values |
| `Tensor.randint(low, high, shape)` | Random integers in `[low, high)` |
| `Tensor.arange(...)` | Arithmetic progression |
| `Tensor.linspace(start, stop, steps)` | Evenly spaced values |
| `Tensor.eye(n)` | Identity matrix |
| `Tensor.empty(*shape)` | Uninitialized tensor |

Core methods:

| Category | Methods |
|---|---|
| Arithmetic | `+`, `-`, `*`, `/`, `**`, `neg`, broadcasting with scalars |
| Elementwise | `exp`, `log`, `sqrt`, `square`, `abs`, `sin`, `cos`, `tanh`, `sigmoid`, `relu`, `gelu`, `silu` |
| Reductions | `sum`, `mean`, `max`, `min`, `argmax`, `sort`, `argsort`, `topk`, `var`, `std` |
| Movement | `reshape`, `view`, `permute`, `transpose`, `expand`, `squeeze`, `unsqueeze`, `flatten`, `shrink`, `pad`, `flip`, `repeat` |
| Indexing | `__getitem__`, `gather`, `take_along_axis`, `cat`, `stack`, `split`, `chunk` |
| Linalg | `matmul`, `dot`, `linear`, `qr`, `triangular_solve`, `solve_triangular`, `cholesky`, `cholesky_solve`, `solve`, `lstsq` |
| Data | `realize`, `numpy`, `item`, `tolist`, `copy_from`, `update_from`, `to`, `cpu`, `cuda`, `detach`, `clone`, `custom_kernel` |

Structured linalg methods are portable tensor-composed fallbacks tested against
NumPy and Torch. They do not add LAPACK or runtime library dependencies.
Current `lstsq` is solution-only for full-rank tall or square systems.

## nn Module

```python
from polygrad.nn import Linear, LayerNorm, RMSNorm, Embedding
from polygrad.nn import SGD, Adam, AdamW, get_parameters
```

Layers:

| Class | Description |
|---|---|
| `Linear(in_features, out_features, bias=True)` | Fully connected layer |
| `LayerNorm(shape, eps=1e-5)` | Layer normalization |
| `RMSNorm(dim, eps=1e-5)` | Root mean square normalization |
| `Embedding(vocab, dim)` | Token embedding |
| `Dropout(p=0.5)` | Training-time dropout |
| `GroupNorm(groups, channels)` | Group normalization |
| `Conv2d(...)` | Tensor-op 2D convolution |
| `BatchNorm(...)` | Batch normalization |

Optimizers:

| Class | Description |
|---|---|
| `SGD(params, lr=0.01, momentum=0.0, weight_decay=0.0)` | SGD with optional momentum |
| `Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8)` | Adam |
| `AdamW(params, lr=0.001, weight_decay=0.01)` | AdamW |

All optimizers provide `step()` and `zero_grad()`.

## HuggingFace Loading

```python
from polygrad.hf import download_hf, load_hf, generate
import numpy as np

model_path = download_hf("hf-internal-testing/tiny-random-gpt2")
inst = load_hf(model_path, max_batch=1, max_seq_len=16)

tokens = np.array([[1, 2, 3, 4]], dtype=np.float32)
result = generate(inst, tokens, max_new_tokens=2, temperature=1.0, top_k=10)
```

Supported path today: GPT-2 style configs and F32/F16/BF16 safetensors. Qwen
family loading is available through the shared C/GGUF paths where configured.

## Tests

```bash
POLYGRAD_LIB=$PWD/build/libpolygrad.so PYTHONPATH=py python -m pytest py/tests -q
```

## License

MIT
