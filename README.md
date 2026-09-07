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
| Model tooling | `PolyModel` stores ABI names, logical buffer bindings, entrypoints, objectives, fit/train helpers, and model bundle metadata |
| Custom kernels | Public custom kernels lower into UOp `CALL` bodies and still run through normal scheduling and runtime caches |
| WebGPU int64 | WGSL has no native 64-bit integers, so renderer lowering uses two 32-bit lanes while C, CUDA, HIP, WASM, and x86 retain native int64; unlike pinned tinygrad, valid dynamic/uint32 shift counts and signed right shift are handled rather than crashing or changing sign semantics |
| WebGPU narrow integers | `PG-DIV-008`: truncate 8/16-bit integer casts and arithmetic results before widening, correcting the pinned WGSL renderer's lost narrowing; Tensor graphs remain unchanged |

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

Python `Tensor.dtype` returns a `DType` object, for example `dtypes.float32`;
JavaScript retains dtype names such as `'float32'`. Constructors accept names
in both frontends. `element_size()` / `elementSize()` report storage bytes and
reject weak dtypes; `is_floating_point()` / `isFloatingPoint()` classify the
dtype without materializing the Tensor.

Movement helpers accept `None` (Python) or `null` (JS) for unchanged axes in
`shrink`, `shrink_to` / `shrinkTo`, and `pad_to` / `padTo`. Target shapes may
be a sequence or positional dimensions. For nonzero fill, use
`x.pad_to(3, 5, value=-1)` or `x.padTo(3, 5, {value: -1})`.
`pad_to` enlarges only; general symbolic padding is not implemented.
`max_shape` / `maxShape` and `max_numel()` / `maxNumel()` expose allocation
bounds without changing the graph. Python retains symbolic `shape` values;
JS `shape` already exposes maximum extents. Full slices and no-op shrink,
`pad_to`, and empty-axis flip return the original Tensor.

Both frontends expose `all`, `any`, `cumsum`, `cumprod`, `cummax`, and `cummin`
through shared C Tensor operations. `all`/`any` accept axes and `keepdim`;
scans currently require concrete shapes. Cumulative extrema return
`(values, indices)` in Python and `[values, indices]` in JS, with int32 indices
selecting the first equal extremum. Use floating-point tensors for gradients.
WebGPU floating comparisons do not guarantee NaN truthiness under WGSL's
finite-math rules; this also affects the pinned Tinygrad renderer.

Shared C pointwise operations include inverse trig/hyperbolic functions,
`erf`, `celu`/`selu`, `isfinite`, `isclose`, `copysign`, and `lerp`.
Logits BCE and NLL accept `none`, `sum`, or `mean` reduction and optional
weights: Python `binary_crossentropy_logits`/`nll_loss`, JS
`binaryCrossEntropyLogits`/`nllLoss` with an options object. NLL also accepts
`ignore_index` / `ignoreIndex`; its gather path currently requires concrete
shapes. Both frontends use the same C construction and autograd, not host
array implementations.

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
should accept caller-created `Tensor` or `Model` objects and keep outputs in
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
Model capture owns its logical entrypoints immediately, so later realization
or retirement of the source Tensor cannot invalidate the portable program.

Default Tensor/JIT execution builds and schedules the physical graph eagerly;
it does not invoke placement. Retained `Model` graphs may instead request an
explicit policy. The first non-uniform policy uses exact named module cuts:

```python
from polygrad import Model, Tensor

x = Tensor([1.0, 2.0])
h = x + 3
y = h * 2
model = Model.from_tensors(
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
fast_model = Model.from_program(program, weights)
result = fast_model.call("forward", {"x": input_array})
```

JavaScript (native/Wasm synchronous runtimes):

```js
const program = model.exportProgram()
const weights = model.exportWeights()
const fastModel = Model.fromProgram(program, weights)
const result = fastModel.call('forward', { x: inputArray })
```

WebGPU startup is asynchronous, so use `await model.exportProgramAsync()` and
`await Model.fromProgramAsync(program, weights)`. A bound-program Model
is inference/call-only: it has no portable logical graph and cannot be
re-placed, trained, differentiated, or exported as PGIR. The existing bundle
format remains the portable PGIR-plus-weights product.

## High-Level APIs

`Model` is a C-owned runtime with sealed graph topology and mutable named state.
Use `Model.from_tensors(...)` / `Model.fromTensors(...)`, or capture an ordinary
callable once with `Model.trace`. Curated `models.MLP`, `TabM`, and `NAM` use the
same runtime. Training stays on Model; no separate Trainer is required.

### Configuration-driven model families

`models.Sequential(config)` and `models.Graph(config)` build ordinary Models in
C, alongside MLP/TabM/NAM. One JSON configuration can be shared by Python, Node,
and browsers without model-specific source compilation or an authoring callback.
They are factories, not subclasses or a second execution graph.

For example, save this as `network.json`:

```json
{
  "input": {"name": "x", "shape": [1, 4], "dtype": "float32"},
  "layers": [
    {"name": "first", "type": "linear", "out_features": 2, "activation": "relu"},
    {"name": "second", "type": "linear", "out_features": 3, "activation": "relu"},
    {"name": "head", "type": "linear", "out_features": 4}
  ],
  "output": "prediction",
  "seed": 42
}
```

```python
from pathlib import Path
from polygrad import models
model = models.Sequential(Path("network.json").read_text())
# Optional runtime=rt uses an explicit Runtime instead of the default context.
```

```javascript
const model = pg.models.Sequential(config) // Node/native or synchronous Wasm
// WebGPU: await pg.models.SequentialAsync(config)
// Graph and GraphAsync accept the connected form described below.
```

C exposes `poly_sequential_from_json(ctx, json, len, &error)` and
`poly_graph_from_json(...)` in `models/compose.h`. The context is borrowed and
must outlive the returned Model; normal Model disposal/training/export apply.
The context must allow logical construction (`always` or `until_realize`).

The initial component catalogue is deliberately bounded:

| Component | Configuration |
| --- | --- |
| `linear` | `out_features`; optional `bias` (default true), `activation` (default `none`) |
| `relu`, `sigmoid`, `tanh`, `silu`, `gelu` | One input |
| `identity`, `square`, `exp`, `log` | One input |
| `add`, `sub`, `mul`, `div` | Two inputs, existing Tensor broadcasting |
| `sum`, `mean` | Reduce all axes to a scalar |
| `reshape` | Concrete positive `shape`, unchanged element count |
| `repeat` | Positive integer `count`; `body` is one unnamed component or a named layer list |

Graph configurations replace `input/layers/output` with:

- `inputs`: name → `{shape, dtype, role?}`; role is `input` or `target`.
- `nodes`: ordered `{name, type, inputs: [earlier_value_names], ...}` records.
- `outputs`: output name → input or node name.
- Optional `entrypoints`: `{name, inputs, outputs, objective?}` records. Without
  them, `forward` exposes all declared inputs and outputs. An explicit scalar
  objective enables the existing Model training path.

Both families accept `modules`, a table of named leaf-component configurations.
Use `{name, call: "shared", inputs: [...]}` instead of `type` in a Graph node
(omit `inputs` in Sequential). Repeated calls reuse the declared component's
parameters; ordinary Repeat bodies create fresh parameters. Module declarations
are construction components, not device-placement cuts or runtime submodels.
See [the shared-layer Graph configuration](test/fixtures/model_definition.json).

Parameter names are `layers.<name>.weight/bias`, `nodes.<name>.weight/bias`, or
`modules.<name>.weight/bias`; Repeat inserts zero-based indices. Linear weights
use the existing seed/name-keyed C-family Kaiming uniform initializer and biases
are zero. This does not promise Keras/Tinygrad initial-weight equivalence.

Inputs currently require explicit float32, concrete rank ≤8 and positive
dimensions. Names are ASCII identifiers of 1–63 characters. Configuration limits
are 1 MiB JSON, nesting 32, 16,384 JSON values, 1,024 expanded component calls,
construction depth 16, and 64 inputs/outputs/entrypoints. Expanded paths are at
most 191 bytes. Named storage totals at most 16,777,216 float32 elements; shapes,
broadcasts and each linear contraction are bounded by that element count too.
These are construction limits, not a bound on compiler/backend peak memory.

Optional `format: "poly.modeldef@1"` and `type: "sequential"`/`"graph"` tags are
checked when present. The selected factory already identifies the family.
Unknown fields, duplicate keys/names, forward/cyclic references, unused shared
components and incompatible shapes fail. There are no config expressions,
recursive modules, runtime loops, dynamic shapes or Keras JSON compatibility.
Configuration describes construction, not checkpoint state: save/load the
result through existing Model bundle or graph/weights APIs.

### Tensor-authored models

```python
from polygrad import Model, create
from polygrad.nn.state import get_state_dict

rt = create(device="cpu", logical="always")
class Net:
    def __init__(self):
        self.weight = rt.Tensor([2.0])
    def __call__(self, x):
        return x * self.weight

net = Net()
model = Model.trace(net, inputs={"x": rt.Tensor.empty(1)},
                    state=get_state_dict(net))
blob = model.save_bundle(include_optimizer=False)
model.free()
rt.dispose()
```

Captured state is independent of `net`'s Tensor attributes. Reads return copies;
use `read_buffer`/`write_buffer` (JS `readBuffer`/`writeBuffer`) for explicit
state access. `bindings()` and `entrypoints()` describe the sealed interface.
For training, supply targets and named losses, configure the optimizer, and
call `train_step`/`trainStep`; select an entrypoint when objectives are ambiguous.
`fit` repeats the supplied batch, not a Keras-style dataset workflow. WebGPU
uses explicit `trainStepAsync`, `readBufferAsync`, and `writeBufferAsync`.
For WebGPU capture, use `Model.traceAsync` or `await Model.fromTensors(...)`;
authoring still runs once, while C state initialization can suspend.

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

Python `UOp.const(value, dtype=None)` and
`UOp.variable(name, min_val, max_val, ...)` use the default Tensor context.
Advanced embedded callers pass their owning context as `ctx=...`; the old
leading-context signatures are not retained. Other low-level UOp factories
still use their existing explicit-context APIs.

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

For host-fed replay, `Tensor.copy_from(data)` (JS `copyFrom`) materializes
pending work and writes current storage without replacing an existing BUFFER
identity. Use JS `await tensor.copyFromAsync(data)` when WebGPU materialization
is needed; it snapshots the supplied bytes while awaiting execution. Writes
to an already-current WebGPU buffer can remain synchronous. Use `assign` for
mutation expressed in the graph.

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

Unchanged upstream Python tests can run in isolated Tinygrad and Polygrad
processes (CPU). Install pytest, NumPy and each selected file's dependencies in
`PARITY_PY`'s environment. The default selection is upstream `backend/test_ops.py`:

```bash
make test-upstream-runner
make test-compat-tinygrad-upstream UPSTREAM_COMPAT_DIR=temp/upstream-run-001
# Select other full files or exact pytest nodeids explicitly:
make test-compat-tinygrad-upstream UPSTREAM_COMPAT_DIR=temp/upstream-run-002 \
  UPSTREAM_COMPAT_TESTS='test/null/test_dtype.py test/unit/test_conv.py'
# Reproduce the reviewed frontier (known failures remain visible):
make test-compat-tinygrad-upstream-ratchet UPSTREAM_COMPAT_DIR=temp/upstream-ratchet-001
# Separate CPU ops lane with the explicit, source-locked NIR-only adaptation:
make test-compat-tinygrad-ops UPSTREAM_COMPAT_DIR=temp/upstream-ops-001
# Ratchet the separately reviewed CPU ops frontier:
make test-compat-tinygrad-ops UPSTREAM_COMPAT_DIR=temp/upstream-ops-002 \
  UPSTREAM_COMPAT_ARGS='--baseline test/fixtures/tinygrad_upstream_ops_cpu_baseline.json'
```

Each output directory must be new. `report.json` records source/library hashes,
provider module identities, per-test outcomes and collection errors; per-file
logs and flushed progress survive worker crashes. In the default lane only module imports are
redirected: missing APIs/backends are not emulated, assertions and upstream
skips are unchanged, and no Tinygrad implementation fills a Polygrad gap.
Upstream test helpers stay on the reference path; loading Polygrad does not
expose sibling packages such as `py/extra` to the reference process.
Missing dependencies, collection failures and incomplete execution fail closed.
Passing these tests does not replace exact `test-parity-graph` checks.

The separate `cpu-ops` adapter removes exactly the pinned NIR import and its
inactive CPU skip decorator. It rejects changed source, non-CPU execution,
renderer/interface overrides and image mode; the Tinygrad control also checks
that its actual renderer is not NIR. All test bodies, tolerances, gradients and
other skips remain unchanged. Adapted-file hashes are recorded in the report.
Keep its reviewed baseline separate from the unchanged lane; this adapter is
test infrastructure, not a Polygrad renderer implementation or source-audit closure.

`UPSTREAM_COMPAT_ARGS='--write-baseline temp/candidate.json'` writes a new
candidate only after complete execution. Review every nonpass's `reason` before
using `--baseline PATH`; candidates are never accepted or overwritten silently.
The ratchet rejects lost/new tests, pass-to-skip/fail transitions, changed
failure signatures and changed pin/suite/environment/adapter contracts. Leading
exception-location line numbers are excluded; raw diagnostics retain them.
Exception messages and values remain part of the signature. Improvements
also require promotion, so an old expected failure cannot return unnoticed.
Diagnostic runs return failure for nonpasses even when writing a candidate.

For a future upstream revision, `--reference PATH --compare-with OLD_REPORT`
attaches added/removed/changed outcomes without changing the accepted checkout
or baseline. This is migration triage, not source-audit closure. Frontend-private
and compiler-private import gaps may prevent whole upstream files from collecting;
the runner reports that boundary rather than counting their tests as skipped.

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
