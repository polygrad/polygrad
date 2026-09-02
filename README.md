# Polygrad

Polygrad is a portable tensor engine.

Write tensor code once, then run the same lazy graph from Python, Node.js, the browser, or native C/C++ applications. The shared C core handles scheduling, compilation, device placement, and execution across CPU, CUDA, HIP, x86, WASM, WebGPU, and interpreter backends.

- [Python guide](py/)
- [JavaScript guide](js/)

## 30-Second Demo

Python:

```python
from polygrad import Tensor

x = Tensor.rand(3, 4)
w = Tensor.rand(4, 5)
y = (x @ w).softmax(-1)

print(y.numpy())
```

Node.js:

```js
const { Tensor } = require('polygrad')

const x = Tensor.rand(3, 4)
const w = Tensor.rand(4, 5)
const y = x.dot(w).softmax(-1)

console.log(y.toArray())
```

Browser:

```js
import { Tensor } from 'polygrad'

const y = new Tensor([1, 2, 3]).mul(2).add(1)
console.log(y.toArray())
```

## Where Polygrad Fits

Polygrad is not trying to replace every tensor framework. It is aimed at tools
that need a small, embeddable compiler/runtime.

Good fits:

- browser-first ML tools;
- Node and Python packages that should share one runtime;
- native libraries that need tensor kernels without a Python dependency;
- model packages that need both WASM/WebGPU and native execution;
- compiler experiments that want tinygrad-like UOps in a C core.

Probably not the right fit yet:

- large distributed training;
- depending on the largest existing model ecosystem;
- vendor BLAS/LAPACK as the primary linalg implementation;
- production workloads that require mature backend-specific kernels for every
  dense linalg path.

## Tinygrad Relationship

Polygrad is a C11 port of tinygrad's compiler direction, not a fork of
tinygrad's Python runtime. Shared concepts keep tinygrad naming where possible:
UOps, rewrites, `LINEAR`, `CALL`, `PROGRAM`, JIT capture/replay, renderer
capabilities, and backend-specific lowering.

The main intentional differences are:

| Area | Polygrad difference |
|---|---|
| Core runtime | Compiler state, buffers, caches, and backend runners live in `PolyCtx` inside a C library |
| Frontends | Python and JavaScript are wrappers over the same C core rather than separate runtimes |
| WASM/browser | Browser execution uses the unified C/WASM runtime path, with WebGPU orchestrated from the C backend |
| Logical vs physical roots | Tensors keep exportable logical graph roots separate from realized/placed physical roots |
| Model tooling | `PolyInstance` stores ABI names, logical buffer bindings, entrypoints, objectives, fit/train helpers, and model bundle metadata |
| Custom kernels | Public custom kernels lower into UOp `CALL` bodies and still run through normal scheduling and runtime caches |
| WebGPU int64 | WGSL has no native 64-bit integers, so renderer lowering uses two 32-bit lanes while C, CUDA, HIP, WASM, and x86 retain native int64; unlike pinned tinygrad, valid dynamic/uint32 shift counts and signed right shift are handled rather than crashing or changing sign semantics |

These differences exist to make Polygrad useful as an embeddable runtime for
tools and model packages, while preserving tinygrad-style compiler semantics
where tinygrad has an equivalent.

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
const { Tensor, disposeDefault } = require('.')
const y = new Tensor([1, 2, 3]).mul(2).add(1)
console.log(y.toArray())
disposeDefault()
JS
```

## Runtime Choices

```text
Python Tensor API       JavaScript Tensor API       C / native package
       |                        |                         |
       +------------------------+-------------------------+
                                |
                       Polygrad C11 runtime
                                |
       +----------------+-------+-------+----------------+
       |                |               |                |
    CPU/x86          CUDA/HIP          WASM            WebGPU
```

| Situation | Use |
|---|---|
| Python research or scripts | `polygrad` from PyPI |
| Node.js product code | `polygrad` from npm with the default `Tensor` API or `polygrad.create()` |
| Browser CPU/WASM | `polygrad` with `core: "wasm"` |
| Browser GPU | `polygrad` with `core: "wasm", device: "webgpu"` |
| Native embedding | C ABI and `libpolygrad` |
| Package integration | create one runtime and pass it into the package |

Backend table:

| Backend | Role |
|---|---|
| CPU C | Portable compiled CPU path |
| interp | Reference interpreter for differential testing |
| x86 | tinygrad-style direct x86 ISA lowering |
| CUDA | NVIDIA GPU backend |
| HIP | AMD GPU backend |
| WASM | Node/browser WebAssembly backend |
| WebGPU | Browser GPU backend through the WASM runtime |

Environment variables:

```bash
POLY_DEVICE=cpu|cuda|hip|x86|interp
POLY_CORE=native|wasm
POLY_DUMP_KERNELS=1
POLY_BEAM=4
```

## Runtime Ownership

A Polygrad runtime is not just a namespace. It owns a `PolyCtx`, compiled
program caches, JIT state, buffer residency, and backend handles such as a WASM
module, WebGPU device state, CUDA runners, or native CPU runners.

Raw C `PolyUOp *` results are borrowed. Retain a root stored across an execution
or collection boundary with `poly_uop_retain()`, release it with
`poly_uop_release()`, and call `poly_ctx_collect()` in allocation-only loops
that never execute. `poly_ctx_stats()` does not collect.

The application should usually create one runtime and pass it to libraries that
need tensor work.

Benefits:

- compiled kernels and JIT captures are reused instead of rebuilt per package;
- tensors from different packages share one buffer residency table;
- browser code gets one WASM/WebGPU runtime instead of several hidden ones;
- Node code gets one native/WASM runtime selection instead of conflicting
  package defaults;
- package APIs can accept and return Polygrad tensors without copying through
  JavaScript arrays or host buffers.

JavaScript package pattern:

```js
const polygrad = require('polygrad')

const pg = polygrad.create({ core: 'wasm' })
const model = SomePackage.create({ polygrad: pg })

const x = new pg.Tensor([[1, 2, 3, 4]])
const y = model.predict(x)
console.log(y.toArray())

pg.dispose()
```

Inside `SomePackage`, use the supplied runtime to allocate tensors, compile
kernels, and dispose package-owned compiled callables. Do not call
`polygrad.create()` internally unless the package explicitly needs isolation:

```js
function create({ polygrad: pg }) {
  const w = pg.Tensor.randn([4, 2]).realize()
  const predict = pg.compile((x) => x.dot(w).realize(), [
    pg.Tensor.empty([1, 4])
  ])

  return {
    predict: (x) => predict.run([x]),
    dispose: () => predict.dispose()
  }
}
```

Python has the same default-context shape for ordinary use and an explicit
runtime API for package isolation or device-specific wiring. Python packages
should accept caller-created `Tensor` or `Instance` objects and keep outputs in
the same context:

```python
from polygrad import Tensor

def normalize(x: Tensor) -> Tensor:
    mean = x.mean(axis=-1, keepdim=True)
    scale = (x - mean).square().mean(axis=-1, keepdim=True).sqrt()
    return (x - mean) / scale
```

Use separate runtimes only when isolation is the point: independent caches,
independent devices, or a package boundary that must outlive/dispose separately.

## The C Core

The C core owns graph construction, scheduling, placement, runtime caches, and
backend dispatch. Frontends are thin wrappers over the same concepts.

```text
default Tensor construction
  -> eager tinygrad-shaped physical tensor graph
  -> LINEAR schedule graph with CALLs
  -> PROGRAM/SOURCE/BINARY plus runtime runner

portable logical program + named state + explicit policy
  -> place from scratch
  -> replacement complete physical tensor graph
```

The key invariant is that logical roots never drive default execution or alter
the Tinygrad-shaped physical graph. Logical lifetime is configurable:

- `until_realize` (default) keeps a producer until that Tensor is materialized,
  then retains only an exact device-free BUFFER/movement resource;
- `always` keeps the full producer graph for portable export or later placement;
- `never` builds only the physical graph and cannot provide portable export.

If any operand has no logical root, a new pure result remains physical-only
even after the ambient context returns to another policy. This never changes
the eager physical graph.

Set the context default with `POLY_LOGICAL=2|1|0`. Python also supports
`Context(LOGICAL=...)`, `Runtime(logical=...)`, `runtime.logical(...)`,
`Tensor(..., logical=...)`, and `tensor.preserve_logical()`. JavaScript supports
`createRuntime({logical: ...})`, `runtime.withLogical(...)`,
`new Tensor(data, {logical: ...})`, and `tensor.preserveLogical()`.
Instance capture owns its logical entrypoints immediately, so later realization
or retirement of the source Tensor cannot invalidate the portable program.

Default Tensor/JIT execution builds and schedules the physical graph eagerly;
it does not invoke placement. Retained `Instance` graphs may instead request an
explicit policy. The first non-uniform policy uses exact named module cuts:

```python
from polygrad import Instance, Tensor

x = Tensor([1.0, 2.0])
h = x + 3
y = h * 2
model = Instance.from_tensors(
    inputs={"x": x}, outputs={"y": y},
    modules=[
        {"name": "stem", "inputs": [x], "output": h},
        {"name": "head", "inputs": [h], "output": y},
    ],
)
model.set_device_map({"stem": "CPU", "head": "CPU:1"})
```

The policy places the aggregate retained graph once, keeps module state with
its module, and inserts explicit `COPY` nodes at cross-device cuts. CPU sibling
identities are executable today; nonzero CUDA/HIP identities, automatic
sharding, pipeline schedules, offload, and VRAM planning remain future work.
PGIR preserves the exact named module boundaries but not their device
assignments, so an imported program can be placed under a new map.

## Export Products

Polygrad keeps portable graphs, bound programs, and weights separate:

- `export_ir()` / `exportIR()` returns portable logical PGIR. Import it with a
  new placement policy when the target device layout may change.
- `export_program()` / `exportProgram()` returns the currently compiled,
  device-bound PROGRAM/LINEAR artifact. It starts without rebuilding the model
  graph, but requires the same Polygrad ABI and a compatible backend/device.
- `export_weights()` / `exportWeights()` returns named safetensors state. Pass
  it separately to either import path when the model has parameters/state.

Python:

```python
program = model.export_program()
weights = model.export_weights()
fast_model = Instance.from_program(program, weights)
result = fast_model.call("forward", {"x": input_array})
```

JavaScript (native/Wasm synchronous runtimes):

```js
const program = model.exportProgram()
const weights = model.exportWeights()
const fastModel = Instance.fromProgram(program, weights)
const result = fastModel.call('forward', { x: inputArray })
```

WebGPU startup is asynchronous, so use `await model.exportProgramAsync()` and
`await Instance.fromProgramAsync(program, weights)`. A bound-program Instance
is inference/call-only: it has no portable logical graph and cannot be
re-placed, trained, differentiated, or exported as PGIR. The existing bundle
format remains the portable PGIR-plus-weights product.

## High-Level APIs

Polygrad includes the usual tensor building blocks:

- elementwise ops, broadcasting, reductions, movement ops, indexing, gather,
  sort, argsort, topk, matmul, softmax, normalization, and loss helpers;
- reverse-mode autograd for first-order training;
- `nn` layers and optimizers in Python, with JavaScript optimizer helpers;
- structured linalg: QR, triangular solve, Cholesky, Cholesky solve, solve, and
  least squares;
- tinygrad-style raw Tensor JIT capture/replay;
- portable bundles for saving IR and weights together, plus separate bound
  compiled-program export for compatible runtimes;
- model loading paths for supported safetensors/GGUF workflows.

Structured linalg is implemented as portable tensor-composed fallback code. It
does not add LAPACK or vendor-runtime dependencies. Backend-specific blocked
kernels are planned for larger matrices.

## JIT And Compile

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
const { Tensor, jit } = require('polygrad')

const f = jit((x) => x.add(1).realize())
f(new Tensor([1, 2, 3]))
f(new Tensor([4, 5, 6]))
console.log(f(new Tensor([7, 8, 9])).toArray())
```

Python and JS also expose `compile(...)`, which warms and captures the same path
up front and returns an explicit callable with `run(...)`, `stats()`, and
`dispose()`.

## Custom Kernels

Polygrad exposes a tinygrad-shaped custom kernel path for cases where Tensor
composition is too indirect. A kernel function receives placeholder UOps and
returns a compiler-ready `SINK(..., arg=KernelInfo(...))` body. As in tinygrad,
`KernelInfo` marks the opaque kernel boundary. Polygrad wraps that body in `CALL` and returns
`AFTER(...)` tensors that still run through normal scheduling, placement,
caches, and device residency.

Python:

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
y = out.custom_kernel(Tensor([1, 2, 3, 4]), Tensor([10, 20, 30, 40]), fxn=add_kernel)[0]
print(y.numpy())
```

JavaScript:

```js
const { Tensor, uop } = require('polygrad')

function addKernel(out, a, b) {
  out = out.flatten(); a = a.flatten(); b = b.flatten()
  const i = uop.range(out.numel(), 0)
  return out.index(i).store(a.index(i).add(b.index(i))).end(i).sink(
    new uop.KernelInfo('custom_add_4')
  )
}

const out = Tensor.empty([4], { dtype: 'float32' })
const y = out.customKernel(new Tensor([1, 2, 3, 4]), new Tensor([10, 20, 30, 40]), addKernel)[0]
console.log(y.toArray())
```

This API is a UOp `CALL` extension point. It is not a raw program-launch API,
and custom backward functions are not implemented yet.

## Tests

Run the main gates:

```bash
make test
make test-parity
make test-py
make test-js-native
TMPDIR=$PWD/temp/cc_tmp EM_CACHE=$PWD/temp/emscripten-cache make test-js-wasm
DISPLAY=:1 make test-browser
make test-browser-matrix
```

`test-browser-matrix` adds non-WebGPU Playwright coverage across Chromium,
Firefox, and installed Chrome/Chromium executables where available.

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
- Browser WebGPU requires a compatible browser and GPU adapter.
- Structured linalg is portable first. Large blocked linalg kernels are planned.
- The tested eager and TinyJit RNG stream is bit-compatible with the pinned
  tinygrad reference; backend-specific lowering still requires each backend's
  normal correctness gate.
- Intentional tinygrad divergences are kept in local architecture notes and
  should be reflected in public docs when they affect users.

## Repository

| Path | Purpose |
|---|---|
| `src/` | C core, compiler, schedulers, runtimes, backends |
| `py/` | Python frontend |
| `js/` | Node, WASM, and browser frontend |
| `r/` | Limited R frontend and `.Call` bridge |
| `test/` | C tests |

## License

MIT
