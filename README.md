# Polygrad

Polygrad is a C11 port of tinygrad's compiler core. It keeps the same broad
direction as tinygrad: Tensor operations build UOps, rewrites lower the graph,
the scheduler emits LINEAR/CALL/PROGRAM work, and backends compile or interpret
the result.

The difference is packaging. The compiler and runtime live in a C library, so
the same core can be used from Python, Node.js, browsers, and other languages
with a C FFI.

## Install

Python:

```bash
pip install polygrad
```

JavaScript:

```bash
npm install polygrad
```

From this source checkout:

```bash
make

POLYGRAD_LIB=$PWD/build/libpolygrad.so PYTHONPATH=py python - <<'PY'
from polygrad import Tensor
print((Tensor([1, 2, 3]) * 2 + 1).numpy())
PY

cd js
npm install
node - <<'JS'
const polygrad = require('.')
;(async () => {
  const pg = await polygrad.create()
  const y = new pg.Tensor([1, 2, 3]).mul(2).add(1)
  console.log(await y.toArray())
  await pg.dispose()
})()
JS
```

## Quick Start

Python:

```python
from polygrad import Tensor

x = Tensor([[1.0, 2.0], [3.0, 4.0]])
w = Tensor([[2.0], [-1.0]])
y = (x @ w).relu()

print(y.numpy())
```

JavaScript:

```js
const polygrad = require('polygrad')

;(async () => {
  const pg = await polygrad.create()
  const { Tensor } = pg

  const x = new Tensor([[1, 2], [3, 4]])
  const w = new Tensor([[2], [-1]])
  const y = x.dot(w).relu()

  console.log(await y.toArray())
  await pg.dispose()
})()
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

## What Works

- Lazy Tensor API in Python and JavaScript.
- Elementwise ops, broadcasting, reductions, movement ops, matmul, softmax,
  normalization, sorting, gather, QR, Cholesky, triangular solve, solve, and
  least squares.
- Reverse-mode autograd for first-order training.
- tinygrad-style raw Tensor JIT capture/replay.
- Python `nn` layers and optimizers.
- Node native addon, Node/browser WASM, browser WebGPU path, CUDA, HIP, x86 ISA
  backend, CPU C backend, and interpreter backend.
- Portable bundles for saving IR and weights together.

See [py/README.md](py/README.md) and [js/README.md](js/README.md) for frontend APIs.

## Device Selection

Python uses the shared C runtime directly:

```python
from polygrad import Tensor, Device

x = Tensor.rand(1024)
y = (x * 2).to("cuda") if Device.cuda_available() else (x * 2).to("cpu")
print(y.numpy())
```

JavaScript selects a core first, then a device:

```js
const pg = await polygrad.create({ core: 'wasm', device: 'webgpu' })
```

Common environment variables:

```bash
POLY_DEVICE=cpu|cuda|hip|x86|interp
POLY_CORE=native|wasm
POLY_DUMP_KERNELS=1
POLY_BEAM=4
```

## JIT

`jit` follows tinygrad's three-call shape: first call runs normally, second call
captures, later calls replay.

Python:

```python
from polygrad import Tensor, jit

@jit
def f(x):
    return (x + 1).realize()

print(f(Tensor([1, 2, 3])).numpy())  # run
print(f(Tensor([4, 5, 6])).numpy())  # capture
print(f(Tensor([7, 8, 9])).numpy())  # replay
```

JavaScript:

```js
const f = pg.jit((x) => x.add(1).realize())
await f(new pg.Tensor([1, 2, 3]))
await f(new pg.Tensor([4, 5, 6]))
console.log(await (await f(new pg.Tensor([7, 8, 9]))).toArray())
```

For embedding loops, Python and JS also expose `compile(...)`, which warms and
captures the same JIT path up front and returns an explicit callable with
`run(...)`, `stats()`, and `dispose()`.

## Custom Kernels

Polygrad exposes a tinygrad-shaped custom kernel path for cases where Tensor
composition is too indirect. A kernel function receives placeholder UOps and
returns a `SINK` body. Polygrad wraps that body in `CALL` and returns
`AFTER(...)` tensors that still run through normal scheduling, placement,
caches, and device residency.

Python:

```python
from polygrad import Tensor
from polygrad.uop.ops import UOp

def add_kernel(out, a, b):
    out, a, b = out.flatten(), a.flatten(), b.flatten()
    i = UOp.range(out.ctx, out.numel(), 0)
    return out[i].store(a[i] + b[i]).end(i).sink()

out = Tensor.empty((4,), dtype="float32")
y = out.custom_kernel(Tensor([1, 2, 3, 4]), Tensor([10, 20, 30, 40]), fxn=add_kernel)[0]
print(y.numpy())
```

JavaScript:

```js
function addKernel(out, a, b) {
  out = out.flatten(); a = a.flatten(); b = b.flatten()
  const i = pg.uop.range(out.numel(), 0)
  return out.index(i).store(a.index(i).add(b.index(i))).end(i).sink()
}

const out = pg.Tensor.empty([4], { dtype: 'float32' })
const y = out.customKernel(new pg.Tensor([1, 2, 3, 4]), new pg.Tensor([10, 20, 30, 40]), addKernel)[0]
console.log(await y.toArray())
```

This API is for UOp `CALL` bodies. It is not a raw program-launch API, and
custom backward functions are not implemented yet.

## Architecture

The shared execution path is:

```text
logical tensor graph
  -> placed tensor graph
  -> LINEAR schedule graph with CALLs
  -> PROGRAM/SOURCE/BINARY plus runtime runner
```

Backends share the same compiler pipeline where possible. Target-specific
details stay in backend renderers and runtimes:

| Backend | Role |
|---|---|
| CPU C | Portable compiled CPU path |
| x86 | tinygrad-style direct x86 ISA lowering |
| CUDA | NVIDIA GPU backend |
| HIP | AMD GPU backend |
| WASM | Node/browser WebAssembly backend |
| WebGPU | Browser GPU backend through the WASM runtime |
| interp | Reference interpreter for differential testing |

For implementation details, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Parity And Tests

Polygrad tracks the vendored `references/tinygrad_latest` checkout.

Current local parity snapshot:

| Target | Result |
|---|---|
| Value parity | 51/51 |
| Optimized value parity | 51/51 |
| Strict IR parity | 51/51 |
| Optimized strict IR parity | 51/51 |

Run the main gates:

```bash
make test
make test-parity
make test-py
make test-js-native
TMPDIR=$PWD/temp/cc_tmp EM_CACHE=$PWD/temp/emscripten-cache make test-js-wasm
DISPLAY=:1 make test-browser
```

Local performance checks:

```bash
make bench-local-baseline
make bench-smoke-regression
make bench-ratios
```

Absolute benchmark baselines are machine-specific and are stored under ignored
local paths.

## Current Limits

- Python currently targets Linux.
- CUDA, HIP, and WebGPU require matching local runtimes.
- Structured linalg is implemented as portable tensor-composed fallbacks. It is
  useful for small and medium problems, but backend-specific blocked kernels are
  still planned.
- RNG is deterministic, but not bit-compatible with tinygrad's current RNG
  stream.
- Intentional tinygrad divergences are documented in [ARCHITECTURE.md](ARCHITECTURE.md).

## Repository

| Path | Purpose |
|---|---|
| `src/` | C core, compiler, schedulers, runtimes, backends |
| `py/` | Python frontend |
| `js/` | Node, WASM, and browser frontend |
| `test/` | C tests |
| `references/tinygrad_latest` | Vendored tinygrad parity target |

## License

MIT
