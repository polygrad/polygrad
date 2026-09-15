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
POLY_LIB=$PWD/build/libpolygrad.so PYTHONPATH=py python - <<'PY'
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
POLY_DEV=cpu|cuda|hip|x86|interp
POLY_DUMP_KERNELS=1
POLY_BEAM=4
```

Polygrad keeps the exportable logical graph separate from the current physical
placement. Calling `realize()`, `.to("cuda")`, or `.cpu()` changes where values
live, not what logical graph is exported.

Explicit runtimes are optional. Tinygrad-style programs use `from polygrad import
Tensor, nn` and `from polygrad.nn.optim import Adam` without `create()` or a Model
wrapper. The Python package uses a module-level default C context. Caller-created tensors
share that context, so package functions should accept and return `Tensor`
objects rather than copying through NumPy unless readback is required.

For explicit context/device ownership, create a runtime:

```python
import polygrad

rt = polygrad.create(device="cpu")
try:
    layer = rt.nn.Linear(2, 1)
    x = rt.Tensor([[1.0, 2.0]])
    print(layer(x).numpy())
finally:
    rt.dispose()
```

Use explicit runtimes for isolation, device-specific package wiring, or tests
that need independent compiler caches.

Use `rt.nn` for layers, optimizers and state loaders, `rt.models` for C-backed
families, and `rt.Model` for capture/import on that runtime. For example,
`rt.Model.load("model.pgb")` is equivalent to
`polygrad.Model.load("model.pgb", runtime=rt)`. Package-level NN constructors
still use the default runtime. Mixing Tensor owners is rejected even when their
devices match; runtime-bound constructors do not switch a global default or
silently copy foreign state.

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
from polygrad import Context, Tensor
from polygrad.nn import Linear, get_parameters
from polygrad.nn.optim import SGD

Tensor.manual_seed(42)
model = Linear(2, 1)
opt = SGD(get_parameters(model), lr=0.01)

with Context(TRAINING=1):
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

`polygrad.stats()` exposes shared C runtime counters. The tinygrad-compatible
top-level `GlobalCounters` provides `global_ops`, `global_mem`, `time_sum_s`,
`kernel_count`, `mem_used`, `mem_used_per_device`, and `reset()`; reset keeps
live allocation bytes. `Jit.stats()` and `CompiledCallable.stats()` expose
wrapper-level capture and replay counters.

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

### Fit in Python, load in JavaScript

Model owns a captured graph and its state; JS does not need the Python class.

```python
from polygrad import Model, Tensor

class Linear:
    def __init__(self):
        self.a = Tensor([0.0])
        self.b = Tensor([0.0])
    def __call__(self, x):
        return {"prediction": self.a * x + self.b}

model = Model(
    Linear(), inputs={"x": Tensor.empty(5)}, targets={"y": Tensor.empty(5)},
    loss=lambda outputs, y: (outputs["prediction"] - y).square().mean(),
)
try:
    model.fit({"x": [-2, -1, 0, 1, 2], "y": [-4, -1, 2, 5, 8]},
              epochs=100, optimizer="sgd", lr=0.1)
    model.save("linear.pgb", include_optimizer=False)
finally:
    model.dispose()
```

In Node, using a matching Polygrad package:

```javascript
const { Model } = require('polygrad')
const model = Model.load('linear.pgb')
try {
  const { prediction } = model.forward({x: new Float32Array([3, 4, 5, 6, 7])})
  console.log(Array.from(prediction)) // approximately [11, 14, 17, 20, 23]
} finally {
  model.dispose()
}
```

This example fixes the input shape at five elements. `params` defaults to named
Tensor attributes of the supplied object; functions require explicit closure state.
`model.summary()` returns interface metadata without executing or reading weights.
Bundle bytes work across frontends; paths are Python/Node conveniences, not browser
filesystem access. Excluding optimizer state does not remove training entrypoints.

Supplying `loss` captures evaluation and training forwards separately against the
same state. The callable runs twice during construction, never during execution.
BatchNorm/RNG updates affect Model-owned state; the authoring object's Tensor
roots and the training mode are restored even on failure. Set the seed before
construction. Capture rejects parameter/input assignments and effectful reads;
arbitrary Python side effects are not transactional. Saved bundles retain RNG
and auxiliary state. Reapply the same optimizer configuration after loading when
resuming training; optimizer state and configuration are distinct.

For variable-size calls, capture a bounded variable leading dimension with fixed
trailing dimensions. Calls and explicit training steps bind concrete extents;
returned Tensors retain their invocation's shape and values. Portable save/load
preserves this signature; storage currently reserves its maximum capacity.

For host arrays or device Tensor datasets, use
`model.fit(data, epochs=2, batch_size=32)`, where each declared input/target's
leading-axis bounds admit 32. Epochs traverse samples in input order; the return
value contains one loss per step. An incomplete final batch is rejected before
training unless `remainder='drop'` discards it or `remainder='keep'` processes
the smaller extent permitted by every input's bounds. Tensor datasets are sliced
on-device without frontend host readback, and may mix with host inputs. Omit
`batch_size` to repeat one full batch as before. No padding or shuffling is implicit.

Flat arrays use the Model signature. Multidimensional ndarrays carry concrete
shape and must match it, not merely its element count. JavaScript can express
the same binding as `{data: typedArray, shape: [rows, columns]}`.

### Pretrained models

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

## Common API Recipes

Create tensors:

```python
from polygrad import Tensor

x = Tensor([1, 2, 3])
a = Tensor.zeros(2, 3)
b = Tensor.ones(2, 3)
c = Tensor.randn(2, 3)
d = Tensor.arange(0, 6).reshape(2, 3)
```

Use NumPy buffers:

```python
import numpy as np
from polygrad import Tensor

arr = np.array([1, 2, 3, 4], dtype=np.float32)
x = Tensor(arr).reshape(2, 2)
print((x * 2 + 1).numpy())
```

Math, movement, indexing:

```python
x = Tensor.arange(0, 12).reshape(3, 4)
y = x.permute(1, 0).reshape(2, 6)
z = y.relu().sum(axis=1)
picked = x.gather(1, Tensor([[0, 2], [1, 3], [0, 1]], dtype="int32"))
```

Autograd:

```python
x = Tensor([1.0, 2.0, 3.0])

loss = (x * x).sum()
loss.backward()
print(x.grad.numpy())
```

Training loop:

```python
from polygrad import Context, Tensor
from polygrad.nn import Linear, get_parameters
from polygrad.nn.optim import SGD

model = Linear(4, 1)
opt = SGD(get_parameters(model), lr=0.01)

x = Tensor.randn(8, 4)
target = Tensor.randn(8, 1)

with Context(TRAINING=1):
    opt.zero_grad()
    loss = (model(x) - target).square().mean()
    loss.backward()
    opt.step()
```

Devices and explicit runtime ownership:

```python
import polygrad
from polygrad import Tensor

x = Tensor([1, 2, 3]).to("cpu")
print((x * 2).numpy())

pg = polygrad.create(device="cpu")
y = pg.Tensor([1, 2, 3])
print((y * 2 + 1).numpy())
pg.dispose()
```

Linear algebra:

```python
A = Tensor([[4.0, 2.0], [2.0, 5.0]])
b = Tensor([1.0, 3.0])

print(A.solve(b).numpy())
print(A.cholesky().numpy())
print(A.lstsq(b).numpy())
```

JIT and compile:

```python
from polygrad import Tensor, jit, compile

@jit
def step(x):
    return (x + 1).realize()

print(step(Tensor([1, 2, 3])).numpy())  # run
print(step(Tensor([4, 5, 6])).numpy())  # capture
print(step(Tensor([7, 8, 9])).numpy())  # replay

compiled = compile(lambda x: (x * 2).realize(), [Tensor.empty(3)])
print(compiled.run([Tensor([1, 2, 3])]).numpy())
compiled.dispose()
```

Repeated input updates:

```python
import numpy as np
from polygrad import Tensor, compile

x = Tensor(np.array([1, 2, 3], dtype=np.float32)).realize()
f = compile(lambda x: x.square().sum().realize(), [x])

print(f.run([x]).item())
x.copy_from(np.array([4, 5, 6], dtype=np.float32))
print(f.run([x]).item())
f.dispose()
```

Runtime inspection:

```python
import polygrad

print(polygrad.stats())
print(polygrad.can_run("add", shape=[1024]))
```

`can_run(...)` is conservative. For some compound op/shape queries it raises
when support cannot be proven statically.

API reference at a glance:

| Area | Main APIs |
|---|---|
| Runtime | `polygrad.create`, `polygrad.stats`, `polygrad.can_run`, `Device` |
| Tensor creation | `Tensor(data)`, `zeros`, `ones`, `full`, `rand`, `randn`, `randint`, `arange`, `linspace`, `eye`, `empty` |
| Tensor math | `+`, `-`, `*`, `/`, `**`, `exp`, `log`, `sqrt`, `abs`, `sin`, `cos`, `tanh`, `sigmoid`, `relu`, `gelu`, `silu`, `softmax` |
| Reductions | `sum`, `mean`, `max`, `min`, `argmax`, `sort`, `argsort`, `topk`, `var`, `std` |
| Movement/indexing | `reshape`, `view`, `permute`, `transpose`, `expand`, `squeeze`, `unsqueeze`, `flatten`, `shrink`, `pad`, `flip`, `repeat`, `gather`, `take_along_axis`, `cat`, `stack`, `split`, `chunk` |
| Linalg | `matmul`, `dot`, `linear`, `qr`, `triangular_solve`, `solve_triangular`, `cholesky`, `cholesky_solve`, `solve`, `lstsq` |
| Data/readback | `realize`, `numpy`, `item`, `tolist`, `copy_from`, `update_from`, `to`, `cpu`, `cuda`, `detach`, `clone` |
| Compilation | `jit`, `compile`, `Tensor.custom_kernel` |
| Neural nets | `polygrad.nn` layers, `SGD`, `Adam`, `AdamW`, `get_parameters`, `get_state_dict` |

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
from polygrad.nn import get_parameters
from polygrad.nn.optim import SGD, Adam, AdamW
```

Layers include `Linear`, `LayerNorm`, `RMSNorm`, `Embedding`, `Dropout`,
`GroupNorm`, `Conv2d`, and `BatchNorm`. Optimizers include `SGD`, `Adam`, and
`AdamW`; each provides `step()` and `zero_grad()`.

## Tests

```bash
POLY_LIB=$PWD/build/libpolygrad.so PYTHONPATH=py python -m pytest py/tests -q
```

## License

MIT
