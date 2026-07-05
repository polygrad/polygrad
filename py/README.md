# Polygrad Python

Python bindings for Polygrad, a C11 port of tinygrad's compiler core.

The Python frontend exposes lazy tensors, autograd, JIT capture/replay, neural
network layers, structured linalg helpers, model loading utilities, and access
to the same C runtime used by Node, WASM, WebGPU, CUDA, HIP, x86, and the
interpreter.

Use this package when you want a familiar Python Tensor API but still want the
compiler/runtime to be embeddable through the shared Polygrad C core.

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

Optional model-loading dependency:

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

Linear algebra:

```python
from polygrad import Tensor

A = Tensor([[4.0, 2.0], [2.0, 5.0]])
b = Tensor([1.0, 3.0])
x = A.solve(b)

print(x.numpy())
```

Structured linalg methods are portable tensor-composed fallbacks tested against
NumPy and Torch. They do not add LAPACK or runtime library dependencies.
Current `lstsq` is solution-only for full-rank tall or square systems.

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

Polygrad keeps the exportable logical graph separate from the current physical
placement. Calling `realize()`, `.to("cuda")`, or `.cpu()` changes where values
live, not what logical graph is exported.

The Python package uses a module-level default C context. Caller-created tensors
share that context, so package functions should accept and return `Tensor`
objects rather than copying through NumPy unless readback is required.

## Data Flow

Polygrad tensors are lazy. Use `realize()` to execute and `numpy()` when host
readback is needed.

```python
import numpy as np
from polygrad import Tensor

x = Tensor.empty((4,), dtype="float32")
x.copy_from(np.array([1, 2, 3, 4], dtype=np.float32))

y = (x * 3 - 1).realize()
print(y.numpy())

x.update_from(np.array([5, 6, 7, 8], dtype=np.float32))
print((x + 1).realize().numpy())
```

Use `copy_from` or `update_from` for repeated loops that should preserve input
buffer identity for compiled replay.

## Training

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

## JIT And Compile

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

## Model Loading

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

## API Summary

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

## Package Integration

Python packages should accept caller-created Polygrad tensors and return
Polygrad tensors:

```python
from polygrad import Tensor

def normalize(x: Tensor) -> Tensor:
    mean = x.mean(axis=-1, keepdim=True)
    scale = (x - mean).square().mean(axis=-1, keepdim=True).sqrt()
    return (x - mean) / scale
```

This keeps execution in the caller's Polygrad context and avoids unnecessary
NumPy readback.

`nn` helpers:

```python
from polygrad.nn import Linear, LayerNorm, RMSNorm, Embedding
from polygrad.nn import SGD, Adam, AdamW, get_parameters
```

Layers include `Linear`, `LayerNorm`, `RMSNorm`, `Embedding`, `Dropout`,
`GroupNorm`, `Conv2d`, and `BatchNorm`. Optimizers include `SGD`, `Adam`, and
`AdamW`; each provides `step()` and `zero_grad()`.

## Tests

```bash
POLYGRAD_LIB=$PWD/build/libpolygrad.so PYTHONPATH=py python -m pytest py/tests -q
```

## License

MIT
