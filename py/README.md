# Polygrad Python

Build tensor computations and train neural networks in Python, then export
Polygrad models for Node.js or the browser without rewriting them in JavaScript.
Exported models carry their graph and weights; loading them does not require
the Python class that created them.

The Python API provides automatic differentiation, neural-network layers,
optimizers, NumPy data exchange and JIT compilation, with CPU and CUDA execution.

[Project overview and shared C/runtime reference](https://github.com/polygrad/polygrad#readme) | [JavaScript guide](https://github.com/polygrad/polygrad/blob/main/js/README.md)

## Contents

- [Install](#install)
- [Quick Start](#quick-start)
- [Devices And Runtimes](#devices-and-runtimes)
- [Data Flow](#data-flow)
- [Training](#training)
- [Models](#models)
- [JIT And Compile](#jit-and-compile)
- [Custom Kernels](#custom-kernels)
- [Common API Recipes](#common-api-recipes)
- [Package Integration](#package-integration)
- [Troubleshooting](#troubleshooting)

## Install

```bash
pip install polygrad
```

Requirements:

- Linux
- Python 3.9 or newer (0.5.0 post-publication examples checked on CPython 3.11)
- NumPy
- A C compiler and Python development headers
- A C compiler on `PATH` for CPU execution: clang recommended, GCC fallback;
  generated float16 kernels require clang's `__fp16` support

The PyPI package is distributed as source; pip builds the native extension
during installation unless it can reuse a cached wheel.

Optional model-loading dependency:

```bash
pip install huggingface_hub
```

For a source checkout, see [building from source](https://github.com/polygrad/polygrad#building-from-source).

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

print(x.grad.numpy())  # [2. 4. 6.]
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

## Devices And Runtimes

Never use one runtime concurrently from multiple Python threads: native calls
release the GIL, and overlapping context mutations can crash the process.
The default runtime is process-global. For parallel work, create one runtime
per thread, construct through `rt.Tensor` and `rt.Model`, and keep its objects
in that thread, including cleanup.
Separate-runtime CPU, INTERP and X86 execution is tested; concurrent GPU
initialization is not validated. Alternatively, serialize all use and cleanup
of a shared runtime yourself. Global configuration changes still need coordination.
Call `collect()` and `dispose()` only when no other thread is using the runtime.

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
DEV=CPU python app.py
POLY_DUMP_KERNELS=1 python app.py
BEAM=4 python app.py
```

Device selection uses an explicit `device=`, then `POLY_DEV`, then `DEV`.
Names are case-insensitive; available native backends include CPU, CUDA, X86
and INTERP. Unsupported device ordinals and multi-device targets are rejected.

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
`polygrad.Model.load("model.pgb", runtime=rt)`.
Create tensors and layers on the same runtime: mixing owners is rejected even
when their devices match; package-level constructors use the default runtime.

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

## Models

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

### Capture and input rules

- This example fixes the input shape at five elements. `params` defaults to
  named Tensor attributes of the supplied object; functions require explicit
  closure state. `model.summary()` inspects metadata without executing or
  reading weights.
- With `loss`, construction captures evaluation and training forwards against
  the same state. The callable runs twice during construction, never during
  execution. Without a loss, capture uses the current `TRAINING` value;
  changing it later does not recapture the graph.
- Set the seed before construction. BatchNorm/RNG updates affect Model-owned
  state. Authoring Tensor roots and training mode are restored even on failure;
  arbitrary Python side effects are not. Capture rejects parameter/input
  assignments and effectful reads.
- Bundle bytes work across frontends; paths are Python/Node conveniences, not
  browser filesystem access. Bundles preserve RNG and auxiliary state.
  Excluding optimizer state does not remove training entrypoints. To resume
  training, reapply the optimizer configuration after loading.
- Variable-size calls support one bounded leading dimension with fixed trailing
  dimensions. Save/load preserves the signature; storage currently reserves
  maximum capacity. Tensor results retain their invocation's values and shape
  across later calls and Model disposal, but require a live runtime.
- Flat arrays use the signature. Multidimensional NumPy arrays must match the
  declared shape, not merely its element count. The equivalent JS binding is
  `{data: typedArray, shape: [rows, columns]}`.

### Minibatches

- Use `model.fit(data, epochs=2, batch_size=32)` when every input/target's
  leading-axis bounds admit 32. Epochs visit samples in input order and return
  one loss per step. Omit `batch_size` to repeat one full batch.
- An incomplete final batch rejects before training unless `remainder='drop'`
  discards it or `remainder='keep'` processes the smaller extent permitted by
  every input's bounds. No padding or shuffling is implicit.
- Tensor datasets are sliced on-device without frontend host readback and may
  mix with host inputs.

### Pretrained models

```python
from polygrad.hf import download_hf, load_hf, generate
import numpy as np

model_path = download_hf("hf-internal-testing/tiny-random-gpt2")
model = load_hf(model_path, max_batch=1, max_seq_len=16)
try:
    tokens = np.array([[1, 2, 3, 4]], dtype=np.int32)
    result = generate(model, tokens, max_new_tokens=2, temperature=1.0, top_k=10)
    print(result)
finally:
    model.dispose()
```

`load_hf` supports GPT-2 and Llama configurations with F32/F16/BF16
safetensors. The `generate` helper above expects GPT-2 input/output names;
it is not a Llama generation API. Qwen loading uses the shared C/GGUF path.

For C-built families and JSON-based `models.Sequential` / `models.Graph`, see
[model configuration](https://github.com/polygrad/polygrad#configuration-driven-model-families).
These return the same Model type and use the same training and export APIs.

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
from polygrad.uop.ops import KernelInfo, UOp

def add_kernel(out, a, b):
    out, a, b = out.flatten(), a.flatten(), b.flatten()
    i = UOp.range(out.ctx, out.numel(), 0)
    return out[i].store(a[i] + b[i]).end(i).sink(
        arg=KernelInfo(name="custom_add_4")
    )

out = Tensor.empty((4,), dtype="float32")
y = out.custom_kernel(
    Tensor([1.0, 2.0, 3.0, 4.0]),
    Tensor([10.0, 20.0, 30.0, 40.0]),
    fxn=add_kernel,
)[0]
print(y.numpy())  # [11. 22. 33. 44.]
```

`KernelInfo` is required to mark the executable kernel boundary. Without it,
the body can leave the output unchanged, as in pinned tinygrad. Match each
stored value's dtype to its destination, using an explicit UOp `cast` when
needed. In 0.5.1, the CPU renderer rejects mismatched vector stores with
`vector STORE dtype mismatch; cast the value to the destination dtype`.
Scalar C stores can convert numerically, but that is not a portable kernel
contract. Wasm/INTERP mismatched stores remain a known limitation: cast
explicitly on every backend. Version 0.5.0 can silently return incorrect values
for mismatched vector stores.

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
from polygrad import Tensor

x = Tensor.arange(0, 12).reshape(3, 4)
y = x.permute(1, 0).reshape(2, 6)
z = y.relu().sum(axis=1)
picked = x.gather(1, Tensor([[0, 2], [1, 3], [0, 1]], dtype="int32"))
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
| Models | `Model`, `from_callable`, `from_tensors`, `fit`, `forward`, `save`, `load`, `summary`, `dispose`; `polygrad.models` factories |
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

Layers include `Linear`, `LayerNorm`, `LayerNorm2d`, `RMSNorm`, `Embedding`,
`Dropout`, `GroupNorm`, `Conv1d`, `Conv2d`, `ConvTranspose1d`, `ConvTranspose2d`,
`InstanceNorm`, `LSTMCell`, and `BatchNorm`. Optimizers include `SGD`, `Adam`, and
`AdamW`; each provides `step()` and `zero_grad()`.

## Troubleshooting

- **CPU compilation cannot find a compiler:** install clang (recommended) or GCC,
  or try `DEV=X86`
  on a supported x86 machine or `DEV=INTERP` for interpreted execution.
- **"TRAINING must be enabled":** wrap optimizer steps in
  `with Context(TRAINING=1):`, as in [Training](#training).
- **Operands belong to different runtimes:** create layers and Tensors through
  the same runtime (`rt.nn`, `rt.Tensor`), or use the default API consistently.
- **Memory grows across many different graphs:** at an idle boundary, use
  `rt.clear_schedule_cache()` and `rt.collect()`. Live owners still retain their
  resources; later execution rebuilds cleared schedules. This is a memory-pressure
  tool, not a per-step requirement.

For contributor checks, see [tests](https://github.com/polygrad/polygrad#tests).

## License

MIT
