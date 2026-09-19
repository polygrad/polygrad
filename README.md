# Polygrad

Polygrad is a tensor library and JIT compiler built around a C11 port of
tinygrad. It provides Python and JavaScript APIs for automatic
differentiation, neural networks and model export. Native applications can use
the C API directly.

- [Python guide](py/README.md): installation, Tensor/NumPy usage, training, Models and JIT.
- [JavaScript guide](js/README.md): Node, browser/WebGPU, typed arrays, training and Models.

## Install

Python:

```bash
pip install polygrad
```

JavaScript:

```bash
npm install polygrad
```

Python requires Linux and Python 3.9+; Node.js requires Node 18+.
See the [Python installation guide](py/README.md#install) and
[JavaScript installation guide](js/README.md#install) for compiler requirements
and native/Wasm setup.

## Quickstart: Tensors

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

Browser (npm package with a browser bundler):

<!-- readme-test: browser -->
```js
import { Tensor } from 'polygrad'

const y = new Tensor([1, 2, 3]).mul(2).add(1)
console.log(y.toArray())
```

Polygrad follows tinygrad's graph-based execution model:

- **Build a graph:** Tensor operations create UOps describing the computation,
  rather than immediately running a kernel for each operation.
- **Lower and render:** when results are needed, the scheduler forms kernels;
  the compiler lowers their operations and renders backend-specific code.
- **Execute:** run on native CPU or CUDA, or use WebAssembly and WebGPU in the
  browser. Direct x86 and interpreter backends are also available.

## Quickstart: Models

Fit `y = 3x + 2` in Python, save its graph and weights, then load it in
JavaScript without redefining the model:

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

Load in Node without the Python class:

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

Polygrad adds three pieces around tinygrad's compiler core:

- **Model and helpers:** capture Tensor computations or construct supported
  architectures, with named inputs, outputs and state.
- **Logical and physical graphs:** retain a portable logical graph separately
  from the device-specific graph used for execution.
- **Portable bundles:** save the logical graph and weights together. Another
  frontend can load the bundle and compile it for its backend without the
  original model class.

## Devices And Runtimes

Explicit runtimes and Models are optional for ordinary Tensor programs.

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

Browser WebGPU execution and readback require async methods. HIP is implemented
but excluded from the 0.5.2 release-validation matrix.

Choose a device through Tensor/Runtime options or, in native applications,
`POLY_DEV` / `DEV`. Use the default runtime for ordinary programs; create an
explicit runtime for isolation and construct related objects through it.
See [Python runtimes](py/README.md#devices-and-runtimes),
[JS runtimes](js/README.md#devices-and-runtimes), and the
[shared settings](#settings) and [ownership reference](#runtime-ownership).

## Current Limits

- Python installation currently targets Linux.
- Polygrad tracks tinygrad 0.14.0, but is not a drop-in replacement or a
  distributed-training framework. Genuine multi-GPU execution is unsupported.
- Do not use one Python runtime concurrently from multiple threads, including
  the default runtime. Overlapping calls can crash the process; use a runtime
  per thread or serialize all use and cleanup.
- Integer division by zero at valid coordinates can terminate native Python
  or Node. Validate divisors supplied by users.
- Structured linear algebra uses portable Tensor compositions, not vendor
  BLAS/LAPACK kernels.

See [compatibility](#tinygrad-relationship) for backend and API gaps, and the
[Python](py/README.md#troubleshooting) or [JS](js/README.md#troubleshooting)
troubleshooting sections for installation and runtime errors.

## Reference

### Guides By Topic

- JIT: [Python](py/README.md#jit-and-compile), [JavaScript](js/README.md#jit-and-compile).
- Custom kernels: [Python](py/README.md#custom-kernels), [JavaScript](js/README.md#custom-kernels).
- Package integration: [Python](py/README.md#package-integration), [JavaScript](js/README.md#package-integration).

### Supported Models

| Model type | JSON construction | HF import | GGUF import | Python generation helper |
| --- | --- | --- | --- | --- |
| MLP, TabM, NAM, Sequential, Graph | Yes | No | No | No |
| GPT2, DistilGPT2 | Yes | Yes | Yes | Yes |
| Llama | Yes | Yes | No | No |
| Qwen3 | No | No | Yes | No |
| CLIP, ViT, DINOv2, DINOv3 | Yes | Yes | No | No |

Checkpoint-required types must have all weights loaded before execution or
export. `models.list()` reports construction/import capabilities; generation
is a separate Python helper, not implied by checkpoint support.

### Configuration-driven Models

Sequential and Graph use the same JSON definitions in Python and JavaScript.
Named factories accept untagged configs; `Model(config)` / `new Model(config)`
requires `format: "poly.modeldef@1"` and a registered `type`.
`models.list()` reports which types support construction or checkpoint import.
These factories return ordinary Models, not architecture-specific subclasses.
The JavaScript snippets below use an existing runtime `pg` and parsed JSON `config`.

<details markdown="1">
<summary>JSON examples, components and construction limits (Python and JavaScript)</summary>

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

<!-- readme-test: config -->
```python
from pathlib import Path
from polygrad import models
model = models.Sequential(Path("network.json").read_text())
# Optional runtime=rt uses an explicit Runtime instead of the default context.
```

<!-- readme-test: config -->
```javascript
const model = pg.models.Sequential(config) // Node/native or synchronous Wasm
// WebGPU: await pg.models.SequentialAsync(config)
// Graph and GraphAsync accept the connected form described below.
```

`models.DistilGPT2(config)` (JS: `pg.models.DistilGPT2(config)`) is a six-block
GPT-2 preset using the same builder and checkpoint names. Config fields can
override defaults. Like GPT-2, it requires weights before execution or export;
HF DistilGPT-2 checkpoints still declare `model_type: "gpt2"` and load normally.

The initial component catalogue is deliberately bounded:

| Component | Configuration |
| --- | --- |
| `linear` | `out_features`; optional `bias` (default true), `activation` (default `none`) |
| `embedding` | Integer indices; `vocab_size`, `embed_dim` |
| `layernorm`, `rmsnorm` | Normalize the fixed last dimension; `eps` (defaults `1e-5`/`1e-6`), `affine` (default true) |
| `rope` | Split-half rotation of `[batch,heads,sequence,head_dim]`; fixed sequence and even head width; `theta` (default 10000) |
| `attention` | Query, key, value, optional mask; `is_causal`, `enable_gqa` (default false); no dropout |
| `relu`, `sigmoid`, `tanh`, `silu`, `gelu` | One input |
| `identity`, `square`, `exp`, `log` | One input |
| `add`, `sub`, `mul`, `div` | Two inputs, existing Tensor broadcasting |
| `sum`, `mean` | Reduce all axes to a scalar |
| `reshape` | Positive `shape`, optionally the same bounded leading dimension; unchanged symbolic element count |
| `permute` | `axes`: a permutation of all input axes |
| `cast` | Explicit destination `dtype` |
| `repeat` | Positive integer `count`; `body` is one unnamed component or a named layer list |

Graph configurations replace `input/layers/output` with:

- `inputs`: name maps to `{shape, dtype, role?}`; role is `input` or `target`.
- `nodes`: ordered `{name, type, inputs: [earlier_value_names], ...}` records.
- `outputs`: output name maps to input or node name.
- Optional `entrypoints`: `{name, inputs, outputs, objective?}` records. Without
  them, `forward` exposes all declared inputs and outputs. An explicit scalar
  objective enables the existing Model training path.

Both model types accept `modules`, a table of named leaf-component configurations.
Use `{name, call: "shared", inputs: [...]}` instead of `type` in a Graph node
(omit `inputs` in Sequential). Here `shared` is the key of a declaration in
`modules`, not a reserved keyword. Repeated calls reuse the declared component's
parameters; ordinary Repeat bodies create fresh parameters. Module declarations
are construction components, not device-placement cuts or runtime submodels.
See [the shared-layer Graph configuration](https://github.com/polygrad/polygrad/blob/main/test/fixtures/model_definition.json).

Parameter names are `layers.<name>.weight/bias`, `nodes.<name>.weight/bias`, or
`modules.<name>.weight/bias`; Repeat inserts zero-based indices. Linear weights
and embedding weights use the existing seed/name-keyed C model Kaiming uniform
initializer; biases are zero. Normalization weights start at one. Parameters are
float32; Tensor promotion rules apply to other input dtypes. RoPE owns deterministic
`freqs_cos`/`freqs_sin` AUX tables under its component name. This does not promise
Keras/Tinygrad initial-weight equivalence or automatic conversion between split-half
and interleaved checkpoint layouts.

Inputs require an explicit concrete scalar dtype supported by the selected backend,
rank <=8 and positive dimensions.
A leading dimension may be `{"name":"batch","min":1,"max":32}`. All bounded
declarations in one definition must use that same name and bounds; trailing
dimensions remain fixed. Calls resolve concrete extents through the normal Model
input contract, and reductions use the invocation extent, not the declared maximum.
See [the typed, bounded transformer-component fixture](https://github.com/polygrad/polygrad/blob/main/test/fixtures/model_components.json).
Names are ASCII identifiers of 1-63 characters. Configuration limits
are 1 MiB JSON, nesting 32, 16,384 JSON values, 1,024 expanded component calls,
construction depth 16, and 64 inputs/outputs/entrypoints. Expanded paths are at
most 191 bytes. Named storage totals at most 16,777,216 elements; node shapes,
linear contractions, embedding selectors and attention scores have the same limit,
evaluated at maximum extents.
These are construction limits, not a bound on compiler/backend peak memory.

Optional `format: "poly.modeldef@1"` and `type: "sequential"`/`"graph"` tags are
checked when present. The selected factory already identifies the model type.
Unknown fields, duplicate keys/names, forward/cyclic references, unused shared
components and incompatible shapes fail. There are no config expressions,
recursive modules, runtime loops, data-dependent shapes or Keras JSON compatibility.
Configuration describes construction, not checkpoint state: save/load the
result through existing Model bundle or graph/weights APIs.

</details>

### Export Products

Use `save()` / `load()` for a portable graph-and-weights bundle.
Polygrad 0.5.2 requires C ABI101 and graph formats PGIR19/PGPM10; incompatible
artifacts are rejected.

For separate artifacts:

- `export_ir()` / `exportIR()` returns portable logical PGIR. Import it with a
  new placement policy when the target device layout may change.
- `export_program()` / `exportProgram()` returns the currently compiled,
  device-bound PROGRAM/LINEAR artifact. It starts without rebuilding the model
  graph, but requires the same Polygrad ABI and a compatible backend/device.
- `export_weights()` / `exportWeights()` returns named safetensors state. Pass
  it separately to either import path when the model has parameters/state.

Python:

<!-- readme-test: export -->
```python
program = model.export_program()
weights = model.export_weights()
fast_model = Model.from_program(program, weights)
result = fast_model.call("forward", {"x": input_array})
```

JavaScript (native/Wasm synchronous runtimes):

<!-- readme-test: export -->
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


### Settings

<details markdown="1">
<summary>Compiler tuning</summary>

Python `Context(NOOPT=1)` and JS `runtime.noopt = 1` disable automatic
kernel scheduling heuristics, not the compiler's required lowering passes or
explicit BEAM search. The setting participates in program-cache keys and does
not change already compiled JIT captures. It is shared by native contexts;
each Wasm module has its own setting. Restore it after a scoped experiment.

Python `Context(BEAM=2)` and JS `runtime.beam = 2` select kernel beam-search
width through the same C policy. Zero disables search. Search compiles and
times candidates on the selected backend; WebGPU timing requires the adapter's
optional `timestamp-query` feature. Without it, candidates are not timed and
the original kernel is kept. Change JS settings while idle and restore them
after use; existing compiled JIT captures are unchanged.

Python `Context(IGNORE_BEAM_CACHE=1)` or JS `runtime.ignoreBeamCache = 1`
bypasses saved search results. `BEAM_TIMEOUT_SEC` bounds candidate compilation
(default 10 seconds; 0 disables the budget). Native compiler subprocesses are
stopped on expiry; in-process driver calls are checked when they return.
`BEAM_STRICT_MODE=1` also propagates native compiler process failures instead
of rejecting those candidates. Ordinary candidate-limit and device compiler
rejections remain recoverable, matching the pinned search policy.

For native search diagnostics, `BEAM_DEBUG=1` prints the input graph and final
options; `BEAM_DEBUG=2` also prints candidate compilation and execution times.
These diagnostics are disabled by default and use the existing C graph printer.

</details>

<details markdown="1">
<summary>Default dtypes</summary>

Default storage dtypes use the same C-owned policy: Python
`Context(DEFAULT_FLOAT='float64', DEFAULT_INT='int64')`, or JS
`runtime.defaultFloat` / `runtime.defaultInt`. Change JS settings while idle
and restore them after use. Explicit dtypes and typed-array inputs are preserved;
backend dtype capabilities still apply. These settings participate in compiler
cache keys. They do not change fixed accumulation/RNG compute widths or the
bounds-based int32/int64 lowering of weak integer indices.

</details>

<details markdown="1">
<summary>DISK storage</summary>

Native file-backed tensors use C-owned `DISK:<path>` storage. Python
`Tensor(pathlib.Path(...))` passes the path to C for memory mapping;
`.to('DISK:<path>')` copies to that exact file, preserving filename case.
`Tensor.empty(..., device='DISK:<path>')` constructs lazy storage with that
identity; it does not open or initialize the file. JS uses
`Tensor.empty(shape, {device: 'disk:<path>'})`.
DISK is not a compute backend. Wasm does not implement native file mapping:
load bytes in JavaScript and pass typed arrays through the host-buffer API.
JS does not currently provide Python's Tensor path constructor.

</details>

<details markdown="1">
<summary>Environment variables</summary>

```bash
DEV=CPU                 # also CUDA, HIP, X86, INTERP, CPU:X86
POLY_DEV=CPU            # Polygrad-specific override of DEV
POLY_LIB=/path/to/libpolygrad.so
POLY_CORE=native        # or wasm
POLY_DUMP_KERNELS=1
BEAM=4
```

Explicit device choices override `POLY_DEV`, which overrides Tinygrad's `DEV`.
Both environment names accept case-insensitive single backend names and
`CPU:X86`. Unsupported renderer, architecture, interface, ordinal-like and
multi-target forms fail rather than silently selecting another backend.
Full Target-selection vocabulary remains open debt PG-PARITY-037, separate
from multi-GPU execution support.
`DEV=CUDA:1` is a renderer request in Tinygrad's target grammar, not GPU1.
Device strings passed directly to Tensor/Runtime retain their existing rules.

`POLY_DEBUG` overrides `DEBUG`, including `POLY_DEBUG=0` to suppress inherited
verbosity. `BEAM` and `NOOPT` retain their Tinygrad names; explicit setters
override their initial environment values. Native runtimes share compiler
policy; each Wasm module owns its policy. Node initializes Wasm compiler
settings once per module. Browsers use runtime properties, not process env.

`POLY_LIB` is Python's explicit native-library path; an invalid Python path
fails without falling back to another installation. Node loads its packaged
addon or Wasm via `POLY_CORE`. The obsolete `POLY_DEVICE` and `POLYGRAD_LIB` names are not recognized.

</details>

## Logical Graphs And Placement

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
explicit policy. Device maps use exact named module cuts:

```python
from polygrad import Model, Tensor

x = Tensor.empty(2)
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
assert model.forward(x=[1.0, 2.0])["y"].tolist() == [8.0, 10.0]
model.dispose()
```

The policy places the aggregate retained graph once, keeps module state with
its module, and inserts explicit `COPY` nodes at cross-device cuts. CPU sibling
identities are executable today; nonzero CUDA/HIP identities, automatic
sharding, pipeline schedules, offload, and VRAM planning remain future work.
PGIR preserves the exact named module boundaries but not their device
assignments, so an imported program can be placed under a new map.

## Runtime Ownership

<details markdown="1">
<summary>Runtime owners, schedule-cache growth and reclamation</summary>

A Runtime owns compiler caches, buffer residency and backend handles. Ordinary
imports use the default runtime; explicit runtimes provide isolation. Use one
owner for related Tensors, layers and Models. See the
[Python](py/README.md#devices-and-runtimes) and
[JavaScript](js/README.md#devices-and-runtimes) guides.

The schedule cache has no automatic eviction or size cap. Every distinct
cached graph keeps its schedule and, unlike Tinygrad's byte keys, its source
graph (`PG-PARITY-032`). Long-lived runtimes producing many distinct graphs can
therefore accumulate memory; `collect()` alone does not evict this cache.
Symbolic bindings that reuse a schedule do not necessarily add cache entries.

At an idle boundary, use Python `pg.clear_schedule_cache()` or
`runtime.clear_schedule_cache()`, JavaScript `runtime.clearScheduleCache()`,
or C `poly_schedule_cache_clear(ctx)` from `polygrad.h`. Then call
`collect()` / `poly_ctx_collect(ctx)` to reclaim resources without other owners.
Storage reclamation is deferred to safe points, including the next Tensor
readback after a storage-owning root is retired. Ordinary JIT/readback does not
scan retained graphs merely because a temporary wrapper was released.
Await pending browser operations before clearing; raw C callers must serialize
access and finish queued device work. Independently owned Model/JIT executions
remain usable. Use clearing at workload boundaries or under memory pressure,
not after every operation. New schedules may need rebuilding, but compiled
kernels can still be reused from other caches; recompilation is not inevitable.
Other caches are untouched, so clearing does not promise that all runtime
memory is returned.

</details>

## Tinygrad Relationship

Polygrad ports tinygrad's compiler concepts to C: UOps, rewrites, LINEAR schedules,
PROGRAMs, JIT capture/replay and backend lowering. The pinned reference is
[tinygrad v0.14.0](https://github.com/tinygrad/tinygrad/tree/v0.14.0), commit
`6f87158d77f66a36d5f8bbe915170b24e2acabe8`.

<details markdown="1">
<summary>Compatibility limits and intentional differences</summary>

The public Tensor APIs do not yet provide SVD/Newton-Schulz,
`nonzero`/`masked_select` (including fixed-size variants), or borrowed-pointer `from_blob`
construction. Tinygrad's `PYTHON` backend name is not an alias for Polygrad's
`INTERP` backend. These are compatibility limits, not passing upstream tests;
the reviewed [Tensor](test/fixtures/tinygrad_upstream_014_baseline.json),
[operation](test/fixtures/tinygrad_upstream_ops_cpu_014_baseline.json) and
[NN/optimizer](test/fixtures/tinygrad_upstream_nn_cpu_014_baseline.json) baselines
keep nonpassing cases explicit, including unsupported Python compiler-private
helpers.

One remaining core limit is shape rank: Tensor/intermediate STAGE shapes
support at most 16 axes (`PG-PARITY-031`). Active loop dependencies are not
rank-limited: materialization preserves every RANGE, while buffer-limit
splitting rejects an intermediate shape it cannot represent. This is an open
parity limitation, not a claim of complete Tinygrad compatibility.

C-style and WGSL parameter names omit Tinygrad's shape suffix
(`PG-PARITY-038`). This source-text parity debt does not permit differences in
argument slots, graph topology or computed values.

The padded-coordinate guard is an intentional divergence (PG-DIV-010);
valid-coordinate division by zero remains subject to [Current Limits](#current-limits).
The strict fixed-width symbolic oracle still fails `uint8_add_wrap_cmp` in both
Polygrad and pinned Tinygrad; see [Tests](#tests).

WebGPU does not guarantee NaN truthiness or preservation through clipping under
WGSL finite-math rules. Unsupported features and reviewed divergences are listed
in the [register](test/fixtures/parity_divergences.json).

The main intentional differences are:

| Area | Polygrad difference |
|---|---|
| Core runtime | Compiler state, buffers, caches, and backend runners live in `PolyCtx` inside a C library |
| Frontends | Python and JavaScript are wrappers over the same C core rather than separate runtimes |
| WASM/browser | Browser execution uses the unified C/WASM runtime path, with WebGPU orchestrated from the C backend |
| Model tooling | `PolyModel` stores ABI names, logical buffer bindings, entrypoints, objectives, fit/train helpers, and model bundle metadata |
| Custom kernels | Public custom kernels lower into UOp `CALL` bodies and still run through normal scheduling and runtime caches |
| WebGPU int64 | WGSL has no native 64-bit integers, so renderer lowering uses two 32-bit lanes while C, CUDA, HIP, WASM, and x86 retain native int64; unlike pinned tinygrad, valid dynamic/uint32 shift counts and signed right shift are handled rather than crashing or changing sign semantics |
| WebGPU narrow integers | `PG-DIV-008`: truncate 8/16-bit integer casts and arithmetic results before widening, correcting the pinned WGSL renderer's lost narrowing; Tensor graphs remain unchanged |

</details>

## Embedding The C Core

See the [C examples](examples/) for Model construction, training and export.

<details markdown="1">
<summary>Headers, context ownership and retained roots</summary>

The C core owns graph construction, scheduling, placement, runtime caches, and
backend dispatch. Frontends are thin wrappers over the same concepts.

For C embedding, `polygrad.h` is the convenience umbrella. Its domain headers
also work independently: `core.h` owns shared types and context controls,
`tensor.h` owns `poly_tensor_*` operations, and `uop/` and `mixin/` headers own
`poly_uop_*` graph construction and composition. Include `model.h` or `nn/nn.h`
when using those APIs. `frontend.h` is separate: it adapts dtype IDs, flattened
arguments and opaque handles for language bindings. Core implementations do
not call these adapters. Use Model-owned declarations or
`poly_model_from_bindings` for named state, and `poly_model_*` helpers for layers.

C Model construction uses `poly_model_from_config(ctx, type, json, len, device,
&error)` from `models/models.h`. A non-NULL context is borrowed and must outlive
the Model; explicit NULL requests an owned context. Typed constructors include
`poly_mlp_into`, `poly_gpt2_into` and `poly_qwen3_into`. Checkpoint import uses
`poly_hf_load` / `poly_gguf_load`, or `_into` variants for a caller-owned context.

Low-level `UOp.variable` bounds retain integer, floating-point and boolean
endpoints independently of the variable dtype. C takes scalar `PolyArg` values;
Python accepts `int`/`float`/`bool`; JavaScript uses `pg.uop.variable(...)`, with
`BigInt` for exact wide integers. NaN, reversed and nonnumeric bounds are rejected.
Typed endpoints can exceed the runtime's signed64
variable-binding domain; metadata support does not imply executable bindings.

Raw C `PolyUOp *` results are borrowed. Retain roots crossing execution or
collection boundaries with `poly_uop_retain()`, and release them with
`poly_uop_release()`. Allocation-only loops that never execute must call
`poly_ctx_collect()`; `poly_ctx_stats()` does not collect.

</details>

## Building From Source

```bash
make

POLY_LIB=$PWD/build/libpolygrad.so PYTHONPATH=py python - <<'PY'
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

Build the standalone browser bundles from the repository root:

```bash
make wasm-pkg
cd js
npm run build:browser
```

## Tests

`make test-readme` checks links and runs the local Python/Node examples,
including setup for multi-block workflows. Shell blocks are syntax-checked;
JSON is parsed. Browser blocks are explicitly excluded, and the HF download
example requires `README_NETWORK=1`. These exclusions are reported as skips.

<details markdown="1">
<summary>Developer checks, release acceptance and benchmarks</summary>

### Functional checks

```bash
make test
make test-parity
make test-py
make test-js-native
TMPDIR=$PWD/temp/cc_tmp EM_CACHE=$PWD/temp/emscripten-cache make test-js-wasm
DISPLAY=:1 make test-browser
make test-browser-matrix
```

`test-all` runs the functional backend/frontend matrix. `test-browser` covers
real browser WebGPU; `test-browser-matrix` adds other browser executables:

```bash
BROWSER_MATRIX="chromium,firefox,old-chrome=chromium@/path/to/chrome" make test-browser-matrix
BROWSER_MATRIX_DEVICES="auto,interp" make test-browser-matrix
```

Browser specs are `chromium`, `firefox`, `webkit`, `chrome`, or
`label=engine@/absolute/path`.

### Release acceptance

```bash
make test-release-list  # inspect the gate list without running it
make test-release PYTHON=/path/to/test/python PARITY_PY=/path/to/parity/python \
  PYTHON_MIN=/path/to/python3.9 HF_PYTHON=/path/to/hf/python
```

Release acceptance runs maintained targets serially: sanitizers, backend and
frontend suites, browser/Qwen, HF fixtures, isolated package installs, interchange,
parity, reviewed analysis, fuzzing and performance checks. CUDA and browser
WebGPU are required. HIP, HCQ2/GETADDR, genuine multi-GPU and PYLITERAL are
[excluded](test/fixtures/release_050_scope.json), not certified. Optional
MSan/TSan/Fil-C and additional browser executables have separate targets.

Required setup:

- `PYTHON` and `PARITY_PY`: CPython 3.11 with the selected test dependencies;
  source-audit AST hashes require that interpreter. Release `PYTHON` defaults
  to `PARITY_PY`.
- `PYTHON_MIN`: Python 3.9 for the isolated minimum-version package check.
- `HF_PYTHON`: the Torch/Transformers/Hugging Face reference environment.
- `CC`: defaults to Clang for release runs, with CPU-renderer `__fp16` support.
  Formatting uses `CLANG_FORMAT=clang-format-14`; analysis uses
  `ANALYZER_CC=clang-14` at the exact reviewed version.
- `build` in `PYTHON`, `z3` in `PARITY_PY`, writable caches, a browser display
  and the required model fixtures. Override `QWEN3_GGUF` for its local path.

Preflight stops on missing prerequisites. The runner does not download model
fixtures, update baselines or approve debts. Individual package checks can use
the network.

Results go to a fresh `temp/release-*` directory or a new `RELEASE_DIR`.
`summary.json` records gate commands, exits, times and source/log/artifact hashes;
individual test counts and skips remain in gate logs. Source changes fail the
run. Non-preflight failures do not stop later gates; any failed gate makes the
final exit nonzero. Interruptions terminate child processes and mark remaining
gates unrun. This target never publishes packages.

`make test-release-runner` tests orchestration without running acceptance.
`make analyze` is the raw zero-warning check; release acceptance uses
`test-analyze-reviewed` with [source-bound reviews](test/fixtures/analyzer_reviews.json).
New warnings, stale reviews, compiler errors and incomplete scans fail.

`test-symbolic-z3-supported` checks general integers and division.
The separate `test-symbolic-z3-fixed` still fails `uint8_add_wrap_cmp` in both
Polygrad and pinned Tinygrad: the interval fold returns false while uint8
`((250 + 10) % 256) < 5` is true. `test-symbolic-z3` runs all modes.

### Model and package checks

```bash
make test-release-gates                    # negative controls: missing HF dependencies / CUDA
make test-qwen3                            # requires POLY_QWEN3_GGUF or the default temp/ fixture
make test-vision                           # small CLIP/ViT/DINO reference, import and native/Wasm checks
POLY_TEST_FILTER=Vision make test-browser   # the same models on browser auto/INTERP/WebGPU
make HAS_CUDA=1 test-qwen3-cuda              # requires a working CUDA GPU, not a skipped test
make test-hf-e2e HF_PYTHON=/path/to/python  # requires huggingface_hub, transformers and torch
make test-release-packages                  # fresh sdist and npm native/fallback installations
```

HF tests can skip missing dependencies in ordinary runs; the strict target fails.
For offline fixtures, set `HF_HUB_CACHE`, `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`. Package checks verify installed library origins and
native-build failure with Wasm fallback. Node browser-export checks do not
replace real-browser execution.

`make fetch-llama-pretrained` downloads the pinned TinyStories Llama 2-style
fixture; `make test-llama-pretrained` compares logits, greedy choices and bundle
reload against Transformers. Set `LLAMA_TEST_DEVICES='cpu cuda'` to require both.
For Llama 3.2 1B, use `make test-llama32-pretrained` with
`LLAMA32_CHECKPOINT=/path/to/Llama-3.2-1B` and
`LLAMA32_TEST_DEVICES='cuda cpu'`. That target does not download weights or test
KV caching/bundle export. Float32 weights alone exceed 4 GB; allow additional
host RAM/swap and device memory.

### Tinygrad compatibility

The default upstream selection is `backend/test_ops.py`. Each output directory
must be new:

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
  UPSTREAM_COMPAT_ARGS='--baseline test/fixtures/tinygrad_upstream_ops_cpu_014_baseline.json'
# NN and optimizer files, with source-locked CPU helper adaptation:
make test-compat-tinygrad-nn UPSTREAM_COMPAT_DIR=temp/upstream-nn-001
# Diagnostic refresh of all674 selected cases per engine, in three serial lanes:
make test-compat-tinygrad-suite UPSTREAM_COMPAT_DIR=temp/upstream-suite-001
```

Reports record source/library hashes, provider identities, outcomes and
collection errors. Missing dependencies and incomplete execution fail. CPU ops
and NN lanes use source-locked helper adaptations, not replacement implementations.
Ratchets reject regressions and changed test/environment contracts; a passing
ratchet does not mean every upstream test passes. Review baseline changes;
see the [runner](scripts/tinygrad_upstream.py) for options.

Complete migration certification is separate: `reference-migration-check`
requires source-wave audits and source-bound execution evidence. Open debts are
not approvals. Durable reviews live in `test/fixtures/migration/`.

### Performance

```bash
make bench-local-baseline
make bench-smoke-regression
make bench-ratios
```

Absolute smoke baselines are machine-specific and stored under ignored paths.
For paired Python eager, training and JIT/readback checks:

```bash
make bench-py-eager PY_PERF_BASELINE=/path/to/baseline/venv/bin/python
```

Use an idle machine and matching Python/NumPy versions. Nine alternating
candidate/baseline pairs retain all samples without retries. Every workload must
meet its 1.02 median ratio limit; numerical results and library origins are checked.
The report is `temp/python-eager-performance.json`.

Release acceptance creates its own isolated, hash-pinned 0.5.1 baseline install,
requiring PyPI access for dependencies. `PY_PERF_BASELINE_SDIST` can supply that
exact source archive locally. Evidence is under `python-performance/`.
The C gate separately builds the pinned 0.5.1 Git checkpoint and current source
with matching flags and nine alternating pairs. That checkpoint must exist locally.
Load averages and affinity are recorded; high load is flagged, not used to discard
samples. These guards do not certify large-model or Model API performance.

</details>

## Repository

| Path | Purpose |
|---|---|
| `src/` | C core, compiler, schedulers, runtimes, backends |
| `py/` | Python frontend |
| `js/` | Node, WASM, and browser frontend |
| `r/` | Unmaintained R prototype; incompatible with the current C API and excluded from release validation |
| `test/` | C tests |

## License

MIT
