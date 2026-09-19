# Polygrad JavaScript

The JavaScript API for Polygrad, a tensor library and JIT compiler built around
a C11 port of tinygrad. It provides tensors, automatic differentiation and
neural-network layers in Node.js and browsers. Load models trained in Python,
or build and train them in JavaScript; browser execution needs no Python server.

[Project overview and shared C/runtime reference](https://github.com/polygrad/polygrad#readme) | [Python guide](https://github.com/polygrad/polygrad/blob/main/py/README.md)

## Contents

- [Install](#install)
- [Quickstart: Tensors](#quickstart-tensors)
- [Quickstart: Models](#quickstart-models)
- [Working With Tensors](#working-with-tensors)
- [Training](#training)
- [Models](#models)
- [Devices And Runtimes](#devices-and-runtimes)
- [Browser](#browser)
- [JIT And Compile](#jit-and-compile)
- [Custom Kernels](#custom-kernels)
- [Package Integration](#package-integration)
- [API Overview](#api-overview)
- [Troubleshooting](#troubleshooting)

## Install

```bash
npm install polygrad
```

Requires Node 18 or newer. The CPU backend needs a C compiler at runtime:
clang is recommended, with GCC as the fallback when clang is absent.
Generated float16 kernels require clang's `__fp16` support. Wasm needs neither.
Browser/WebGPU setup is covered in [Browser](#browser).

### Native installation

Node tries to build the native addon during install, using a C/C++ compiler,
Python and the platform's `node-gyp` build tools. If that fails, runtime
creation can still use the WASM core. To skip native compilation:

```bash
POLYGRAD_SKIP_NATIVE=1 npm install polygrad
```

For a source checkout, see [building from source](https://github.com/polygrad/polygrad#building-from-source).

## Quickstart: Tensors

```js
const { Tensor } = require('polygrad')

const a = Tensor.rand(3, 4)
const b = Tensor.rand(4, 5)
const c = a.dot(b).softmax(-1)

console.log(c.toArray())  // 15 values: three rows of five probabilities
```

Autograd:

```js
const { Tensor } = require('polygrad')

const x = new Tensor([1, 2, 3], { dtype: 'float32' }) // gradients need floats
const loss = x.mul(x).sum()

loss.backward()
console.log(x.grad.toArray())  // [2, 4, 6]
```

For browser setup and WebGPU execution, see [Browser](#browser).

## Quickstart: Models

Fit `y = 3x + 2` in Node with a captured Model:

```js
const { Model, Tensor } = require('polygrad')

const net = {
  a: new Tensor([0], { dtype: 'float32' }),
  b: new Tensor([0], { dtype: 'float32' }),
  forward({ x }) { return { prediction: x.mul(this.a).add(this.b) } }
}
const model = new Model(net, {
  inputs: { x: Tensor.empty(5) },
  targets: { y: Tensor.empty(5) },
  loss: (outputs, { y }) => outputs.prediction.sub(y).square().mean()
})
try {
  model.fit({ x: new Float32Array([-2, -1, 0, 1, 2]), y: new Float32Array([-4, -1, 2, 5, 8]) },
    { epochs: 100, optimizer: 'sgd', lr: 0.1 })
  console.log(Array.from(model.forward({ x: new Float32Array([3, 4, 5, 6, 7]) }).prediction))
} finally {
  model.dispose()
}
```

### Load a Model trained in Python

First run the [Python export example](https://github.com/polygrad/polygrad/blob/main/py/README.md#quickstart-models)
to create `linear.pgb`, then run this in the same working directory:

```js
const { Model } = require('polygrad')

const model = Model.load('linear.pgb')
try {
  const { prediction } = model.forward({ x: new Float32Array([3, 4, 5, 6, 7]) })
  console.log(Array.from(prediction))  // approximately [11, 14, 17, 20, 23]
} finally {
  model.dispose()
}
```

See [Models](#models) for capture, minibatches, state and checkpoint loading.

## Working With Tensors

### Creation And Dtypes

Shape constructors accept either `Tensor.zeros(2, 3)` or `Tensor.zeros([2, 3])`:

```js
const { Tensor } = require('polygrad')

const a = Tensor.zeros([2, 3])
const b = Tensor.ones([2, 3])
const c = Tensor.randn([2, 3])
const d = Tensor.arange(6).reshape(2, 3)
const e = new Tensor(new Float32Array([1, 2, 3, 4])).reshape(2, 2)
```

JavaScript arrays containing only integer-valued numbers infer `int32`, even
when written as `1.0`. Use `{ dtype: 'float32' }` or a `Float32Array` for
floating-point work such as gradients. Typed arrays preserve their dtype.
Set shape with `.reshape(...)`, not a constructor option. The constructor's
public options are `dtype`, `device`, `logical`, and `isParam` (`is_param`).
Unknown constructor options are rejected.

### Math And Indexing

```js
const { Tensor } = require('polygrad')

const x = Tensor.arange(12).reshape(3, 4)
const y = x.permute(1, 0).reshape(2, 6)
const z = y.relu().sum(1)
const picked = x.gather(1, new Tensor([[0, 2], [1, 3], [0, 1]], { dtype: 'int32' }))
```

### Reading And Updating Data

Polygrad tensors are lazy. Build expressions freely, then call `realize()` or
read data back.

```js
const { Tensor } = require('polygrad')

const x = new Tensor(new Float32Array([1, 2, 3, 4]))
const y = x.mul(3).sub(1).realize()

console.log(y.toTypedArray())  // Float32Array
```

For repeated loops, reuse tensor buffers instead of constructing new source
tensors:

```js
const { Tensor } = require('polygrad')

const x = new Tensor(new Float32Array([1, 2, 3, 4]))
x.realize()
x.copyFrom(new Float32Array([5, 6, 7, 8]))
```

Use `toTypedArrays()` when reading several outputs together.

```js
const { Tensor } = require('polygrad')

const x = new Tensor([1, 2, 3])
const a = x.add(1)
const b = x.mul(2)
const [aData, bData] = Tensor.toTypedArrays(a, b)
```

Readback forms:

```js
const { Tensor } = require('polygrad')

const y = new Tensor([1, 2, 3]).mul(2)
console.log(y.toArray())              // typed array on sync runtimes
console.log(y.tolist())               // nested JS arrays
console.log(y.sum().item())          // item() requires a scalar

const [a, b] = Tensor.toTypedArrays(y, y.add(1))
```

### Linear Algebra

```js
const { Tensor } = require('polygrad')

const A = new Tensor([[4.0, 2.0], [2.0, 5.0]], { dtype: 'float32' })
const b = new Tensor([1.0, 3.0], { dtype: 'float32' })
const x = A.solve(b)

console.log(x.toArray())
```

Structured linalg methods are portable tensor-composed fallbacks. Current
`lstsq` is solution-only for full-rank tall or square systems.

## Training

Use Tensor autograd when you want to control the training loop:

```js
const { Tensor, nn } = require('polygrad')

const model = new nn.Linear(4, 1)
const opt = new nn.SGD(nn.getParameters(model), { lr: 0.01 })

const x = Tensor.randn(8, 4)
const target = Tensor.randn(8, 1)

const wasTraining = Tensor.training
Tensor.training = true
try {
  opt.zeroGrad()
  const loss = model.call(x).sub(target).square().mean()
  loss.backward()
  opt.step()
} finally {
  Tensor.training = wasTraining
}
```

## Models

Start with [Quickstart: Models](#quickstart-models) for fitting and cross-language loading.
More runnable scripts are in [JavaScript examples](https://github.com/polygrad/polygrad/tree/main/js/examples).

### Capture and input rules

- Object authors expose `forward(inputs)`; their Tensor attributes supply state
  unless `params` overrides it. With a loss, construction captures evaluation
  and training forwards against the same state. The author runs twice during
  construction, never during `forward` or `fit`.
- Set the seed before construction. BatchNorm/RNG updates belong to Model-owned
  state. Authoring Tensor roots and `Tensor.training` are restored even on
  failure, but arbitrary JavaScript side effects are not rolled back.
- Authored assignments require auxiliary state (`is_param_(false)`); the
  optimizer updates parameters. Without a loss, capture uses the current
  `Tensor.training`; changing it later does not recapture the graph.
- On WebGPU, construct with `await pg.Model.fromCallableAsync(author, options)`.
  The author must remain synchronous and must not start async reads or execution.
- Inputs may have one bounded variable leading dimension and fixed trailing
  dimensions. Storage currently reserves maximum capacity; empty calls reject.
  Flat typed arrays use the signature, or provide
  `{data: typedArray, shape: [rows, columns]}` as a Model input binding.
- Tensor inputs must share the Model's runtime and device. Any Tensor input
  makes outputs owned device Tensors; array-only calls return host arrays.
  Results retain their values and shapes across later calls and Model disposal,
  but not runtime disposal. Dispose results when finished. This uses device
  copies; it is not zero-copy or differentiable Model composition.

Saved bundles preserve auxiliary/RNG and optional optimizer state. To replace
weights, use `model.importWeights(bytes)` on synchronous backends or
`await model.importWeightsAsync(bytes)` on WebGPU. The latter snapshots the
supplied bytes and reads current device state for rollback.

### Minibatches

`fit` accepts `{ batchSize, epochs, optimizer, lr }`. Minibatches visit samples
in order; there is no implicit shuffling or padding. An incomplete last batch
is rejected unless `remainder: 'drop'` skips it or `remainder: 'keep'` processes
a smaller extent allowed by the Model's input bounds. On WebGPU, use `fitAsync`.

### Saving And Loading

`Model.load` uses the default runtime; use `pg.Model.load(...)` for an explicit
runtime. Browser callers pass bundle bytes to `Model.fromBundle`, not filesystem
paths. Each load owns independent state; tied aliases inside a Model remain tied.
Disposing the Model does not dispose its runtime.

`save()` returns bundle bytes; Node also supports `save(path)`. Optimizer state
is included unless `{ includeOptimizer: false }` is supplied. To resume training,
reapply the optimizer configuration after loading.

See [export products](https://github.com/polygrad/polygrad#export-products)
for portable bundles, bound programs and weights, including JavaScript methods.

### Pretrained And Configured Models

For C-built model types and `models.Sequential` / `models.Graph`, see
[shared JSON reference](https://github.com/polygrad/polygrad#configuration-driven-models).
Their configurations support typed inputs, one bounded leading batch dimension,
and shared embedding, normalization, RoPE and attention components. For example,
`{dtype:'int32',shape:[{name:'batch',min:1,max:32},16]}` declares token batches of
1 to 32 rows. Pass `Int32Array` token data; the signature determines its batch size.

`models.GPT2` and `models.Llama` construct checkpoint-required models. Load
weights or explicitly write every parameter before calling or exporting them.
Partial writes do not initialize a parameter; explicitly written zeros do.

`pg.models.list()` reports construction and checkpoint capabilities; see
[supported models](https://github.com/polygrad/polygrad#supported-models).
Calling `pg.models.Qwen3(...)` reports that it is import-only and points to
`pg.Model.fromGGUF(...)`; it does not construct an uninitialized model.

New Qwen3 GGUF imports take only `x` (`Int32Array`) and return `output`:
`(await model.forwardAsync({ x: tokenIds })).output`. Rotary tables are
Model-owned state, included in saved bundles. Older bundles retain their
original signatures; inspect them with `model.entrypoints()`.

### Vision models

`pg.models.CLIP`, `ViT`, `DINOv2` and `DINOv3` use the same C builders as Python.
Load HF config/safetensors bytes with `pg.Model.fromHF(configBytes, weightFiles)`,
or a Python-saved bundle with `pg.Model.load(bytes)`. Configuration-only models
require weights before execution or saving; WebGPU construction uses the `Async`
factory variants.

| Model | Inputs | Forward outputs |
| --- | --- | --- |
| CLIP | `pixel_values`, `input_ids` | `image_embeds`, `text_embeds`, `logits_per_image`, `logits_per_text` |
| ViT | `pixel_values` | `last_hidden_state`, `pooler_output` (tanh pooler) |
| DINOv2 / DINOv3 | `pixel_values` | `last_hidden_state`, `pooler_output` (CLS token) |

Use `await model.forwardAsync(inputs)` on WebGPU. Images are preprocessed
`Float32Array` values in NCHW order at the checkpoint's fixed square resolution;
decoding, resizing and normalization are not part of the Model. CLIP tokens are
right-padded `Int32Array` values containing EOS. Its `encode_image`/`encode_text`
entrypoints return normalized embeddings from one modality; both use the same
configured batch size.

Supported checkpoint classes: HF `CLIPModel`, `ViTModel`, `Dinov2Model` and
`DINOv3ViTModel`. This is unmasked inference, not classification heads,
DINOv2-with-registers, DINOv3 ConvNeXt, training augmentations or variable-resolution
position interpolation. DINOv3 includes register tokens and patch-only 2D RoPE;
the full hidden-state output includes prefix tokens. DINOv2 SwiGLU and DINOv3
gated MLP are supported. JSON tags: `clip`, `vit`, `dinov2`, `dinov3_vit`.

## Devices And Runtimes

Explicit runtime:

```js
const polygrad = require('polygrad')

const pg = polygrad.create({ core: 'wasm' })
const y = new pg.Tensor([1, 2, 3]).mul(2).add(1)
console.log(y.toArray())
pg.dispose()
```

Node workers can each create their own CPU/INTERP runtime (or Wasm instance);
dispose it before the worker exits, and do not transfer native handles between
workers. Concurrent GPU startup has not been validated. The Linux native
addon remains loaded until process exit so
worker teardown cannot unload code still needed by compiler thread-local cleanup.

The default JavaScript API is sync-first. `polygrad.create(...)` returns a
`PolyRuntime` immediately or throws; tensor construction and graph construction
are synchronous too.

Use explicit async startup when the host cannot or should not instantiate the
WASM core synchronously, for example older browser limits or a fetch/streaming
loader path. The async entry is not a drop-in default `Tensor` import because
startup itself must be awaited:

```js
const { createAsync } = require('polygrad/async')

async function main() {
  const pg = await createAsync({ core: 'wasm' })
  try {
    console.log(new pg.Tensor([1, 2, 3]).mul(2).toArray())
  } finally {
    await pg.dispose()
  }
}
main().catch(error => { console.error(error); process.exitCode = 1 })
```

Runtime matrix:

| Work | Sync API | Async API |
|---|---|---|
| Create normal Node/browser runtime | `polygrad.create(opts)` | not needed |
| Create through async WASM loader | throws with guidance | `createAsync(opts)` from `polygrad/async` |
| Construct tensors and graph ops | `new Tensor(...)`, `x.mul(2)` | not needed |
| CPU/native/WASM realize/readback | `x.realize()`, `x.toArray()` | `realizeAsync()` / `toArrayAsync()` also work |
| WebGPU realize/readback | throws `PolyAsyncRequired` | `await x.realizeAsync()`, `await x.toArrayAsync()` |

```js
const polygrad = require('polygrad')

const pg = polygrad.create({ core: 'wasm' })
console.log(pg.core, pg.device)
pg.dispose()
```

Runtime selection:

| Environment | `core: "auto"` | `core: "native"` | `core: "wasm"` |
|---|---|---|---|
| Node with native addon | native | native | wasm |
| Node without native addon | wasm | error | wasm |
| Browser | wasm | error | wasm |

Options:

- `core`: `auto`, `native`, or `wasm`.
- `device`: `auto`, `cpu`, `cuda`, `hip`, `x86`, `interp`, `wasm`, or
  `webgpu`, depending on the selected core and local runtime support.

Environment variables:

```bash
POLY_CORE=wasm node app.js
POLY_DEV=cuda node app.js
```

Device selection uses the explicit `device` option, then `POLY_DEV`, then `DEV`,
then the platform default. Target names are case-insensitive. Native choices
include `cpu`, `cuda`, `hip`, `x86`, and `interp`; Wasm choices include `wasm`,
`interp`, and browser `webgpu` when available. Unsupported ordinals/multi-target
specifications reject rather than selecting another device. `POLY_DEVICE` is
no longer read. Browsers use runtime options, not process environment variables.

Call `pg.dispose()` when an application or long-running script is done with an
explicit runtime.

Native `model.place('CPU:1')` and `setDeviceMap(...)` accept the same exact CPU
identities. Wasm maps plain `CPU` to `WASM`, but rejects CPU
ordinals; unsupported accelerator ordinals also reject. On WebGPU use
`await model.placeAsync(device)`.

### Runtime Inspection

```js
const polygrad = require('polygrad')

console.log(polygrad.stats())
polygrad.getDefaultRuntime().resetCounters()
console.log(polygrad.canRun({ op: 'add', shape: [1024] }))
```

`stats().coreStats` includes `globalOps`, `globalMem`, `timeSumS`,
`kernelCount`, and live `memUsed`. `resetCounters()` clears execution totals
without clearing live allocation accounting.

`canRun(...)` is conservative. For some compound op/shape queries it throws
when support cannot be proven statically.

## Browser

Browser-only, with an npm-installed package and a browser bundler:

<!-- readme-test: browser -->
```js
import { Tensor } from 'polygrad'

const y = new Tensor([1, 2, 3]).mul(2).add(1)
console.log(y.toArray())
```

Browser-only WebGPU: create an explicit runtime and use async execution/readback:

<!-- readme-test: browser -->
```js
import { create } from 'polygrad'

const pg = create({ core: 'wasm', device: 'webgpu' })
const y = new pg.Tensor([1, 2, 3]).mul(2)
console.log(await y.toArrayAsync())
pg.dispose()
```

Bundlers should resolve the bare `polygrad` import to the browser bundle
through the package `browser` export condition. Node `require('polygrad')`
still resolves to the Node entry.

If a bundler or CDN resolver picks the Node entry by mistake, force the browser
entry explicitly (browser-only):

<!-- readme-test: browser -->
```js
import { create } from 'polygrad/browser'
```

For a local checkout or manual browser bundle, see
[building from source](https://github.com/polygrad/polygrad#building-from-source).

Outputs:

- `dist/polygrad.sync.js` for a sync browser global.
- `dist/polygrad.sync.mjs` for sync browser ESM.
- `dist/polygrad.async.js` for an explicit async-startup browser global.
- `dist/polygrad.async.mjs` for explicit async-startup browser ESM.

Browser global:

<!-- readme-test: browser -->
```html
<script src="./dist/polygrad.sync.js"></script>
<script>
  const y = new polygrad.Tensor([1, 2, 3]).mul(2)
  console.log(y.toArray())
  polygrad.disposeDefault()
</script>
```

Local browser ESM with WebGPU:

<!-- readme-test: browser -->
```html
<script type="module">
  import { create } from './dist/polygrad.sync.mjs'

  const pg = create({ core: 'wasm', device: 'webgpu' })
  const y = new pg.Tensor([1, 2, 3]).mul(2)
  console.log(await y.toArrayAsync())
  pg.dispose()
</script>
```

Explicit async startup bundle:

<!-- readme-test: browser -->
```html
<script type="module">
  import { createAsync } from './dist/polygrad.async.mjs'

  const pg = await createAsync({ core: 'wasm' })
  const y = new pg.Tensor([1, 2, 3]).mul(2)
  console.log(y.toArray())
  pg.dispose()
</script>
```

## JIT And Compile

`jit(fn)` follows tinygrad raw Tensor JIT behavior: first call runs normally,
second call captures realized schedules, later calls replay.

```js
const { Tensor, jit, compile } = require('polygrad')

const f = jit((x) => x.add(1).realize())

f(new Tensor([1, 2, 3]))  // normal run
f(new Tensor([4, 5, 6]))  // capture
console.log(f(new Tensor([7, 8, 9])).toArray())  // replay
```

`compile(fn, sampleInputs)` warms and captures immediately:

```js
const { Tensor, compile } = require('polygrad')

const compiled = compile(
  (x) => x.add(1).realize(),
  [new Tensor([1, 2, 3])]
)

const out = compiled.run([new Tensor([7, 8, 9])])
console.log(out.toArray())
console.log(compiled.stats())
compiled.dispose()
```

### Reusing Input Buffers

```js
const { Tensor, compile } = require('polygrad')

const x = new Tensor(new Float32Array([1, 2, 3])).realize()
const f = compile((x) => x.square().sum().realize(), [x])

console.log(f.run([x]).item())
x.copyFrom(new Float32Array([4, 5, 6]))
console.log(f.run([x]).item())
f.dispose()
```

## Custom Kernels

`Tensor.customKernel(...)` mirrors tinygrad's alpha custom-kernel shape. The
kernel function receives placeholder UOps and returns a `SINK` body. Polygrad
wraps the body in `CALL`, returns `AFTER(...)` tensors, and keeps execution in
the normal schedule and runtime caches.

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
const y = out.customKernel(
  new Tensor([1, 2, 3, 4], { dtype: 'float32' }),
  new Tensor([10, 20, 30, 40], { dtype: 'float32' }),
  addKernel
)[0]
console.log(y.toArray()) // [11, 22, 33, 44]
```

`KernelInfo` is required to mark the executable kernel boundary. Without it,
the body can leave the output unchanged, as in pinned tinygrad. Match each
stored value's dtype to its destination, using an explicit UOp `cast` when
needed. Current C and Wasm renderers reject mismatched vector stores with
`vector STORE dtype mismatch; cast the value to the destination dtype`.
Scalar C stores can convert numerically, but that is not a portable kernel
contract. INTERP follows Tinygrad's Python memoryview conversion rules. Cast
explicitly on every backend.

## Package Integration

JavaScript package pattern:

<!-- readme-test: package -->
```js
const polygrad = require('polygrad')

const pg = polygrad.create({ core: 'wasm' })
const model = SomePackage.create({ polygrad: pg })

const x = new pg.Tensor([[1, 2, 3, 4]], { dtype: 'float32' })
const y = model.predict(x)
console.log(y.toArray())

model.dispose()
pg.dispose()
```

Inside `SomePackage`, use the supplied runtime to allocate tensors, compile
kernels, and dispose package-owned compiled callables. Do not call
`polygrad.create()` internally unless the package explicitly needs isolation:

<!-- readme-test: package -->
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

## API Overview

| Area | Main APIs |
|---|---|
| Runtime | `create`, `createAsync`, `disposeDefault`, `stats`, `canRun` |
| Models | `Model`, `fromCallable`, `fromTensors`, `fit`, `forward`, `save`, `load`, `summary`, `dispose`; `models` factories |
| Tensor creation | `new Tensor(data)`, `zeros`, `ones`, `full`, `rand`, `randn`, `eye`, `arange`, `empty` |
| Tensor math | `add`, `sub`, `mul`, `div`, `exp`, `log`, `sqrt`, `relu`, `gelu`, `silu`, `softmax` |
| Reductions | `sum`, `mean`, `max`, `argmax`, `sort`, `argsort`, `topk`, `var`, `std` |
| Movement/indexing | `reshape`, `expand`, `permute`, `shrink`, `flip`, `pad`, `cat`, `stack`, `gather`, `takeAlongAxis`, `scatter`, `scatterReduce` |
| Linalg | `dot`, `qr`, `triangularSolve`, `solveTriangular`, `cholesky`, `choleskySolve`, `solve`, `lstsq` |
| Data/readback | `realize`, `realizeAsync`, `toArray`, `toArrayAsync`, `toTypedArray`, `toTypedArrayAsync`, `toTypedArrays`, `toTypedArraysAsync`, `copyFrom`, `updateFrom` |
| Compilation | `jit`, `jitAsync`, `compile`, `compileAsync`, `Tensor.customKernel` |
| Neural nets | `nn.Linear`, `nn.SGD`, `nn.Adam`, `nn.AdamW`, `nn.getParameters`, `nn.getStateDict` |

## Troubleshooting

- **`unknown type name '__fp16'`:** float16 CPU kernels require Clang.
  Install it and select `CC=clang`; check that `CC` is not forcing GCC.
- **`args mismatch in jit`:** the call must match the traced input shapes,
  dtypes and devices. An `int32` input cannot replace a `float32` sample.
- **Bundle ABI/format mismatch:** use matching producer/consumer Polygrad versions.
  See [bundle compatibility](https://github.com/polygrad/polygrad#export-products);
  do not edit artifact version fields to bypass validation.
- **"No leaf tensors require grad":** check input dtypes first. Integer-valued
  JavaScript arrays infer `int32`, including `[1.0, 2.0]`. For gradients, use
  `new Tensor([1, 2], { dtype: 'float32' })` or a `Float32Array`, and keep the
  floating-point input Tensors alive until `backward()`.
- **CPU compilation cannot find a compiler:** install clang (recommended) or GCC,
  or use `DEV=X86`
  with the native addon on a supported x86 machine, or `DEV=INTERP`.
- **Native installation fell back to Wasm:** inspect `pg.core`. Install the
  native build prerequisites and rebuild if native execution is needed;
  `create({ core: 'native' })` reports failure instead of falling back.
- **`PolyAsyncRequired`:** WebGPU execution/readback must use the async methods,
  such as `await tensor.toArrayAsync()` and `await model.forwardAsync(inputs)`.
- **"TRAINING must be enabled":** enable `Tensor.training` around optimizer
  steps and restore it in `finally`, as in [Training](#training).
- **Memory grows across many different graphs:** at an idle boundary, use
  `pg.clearScheduleCache()` and `pg.collect()`. Live owners still retain their
  resources; later execution rebuilds cleared schedules. Do not clear per step.

For contributor checks, see [tests](https://github.com/polygrad/polygrad#tests).

## License

MIT
