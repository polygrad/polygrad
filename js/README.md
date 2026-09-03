# Polygrad JavaScript

JavaScript bindings for Polygrad, a C11 tensor compiler with Node, WASM, and
browser runtimes.

The package uses the same C compiler core as Polygrad Python. In Node it tries
the native addon first and falls back to packaged WASM. In browsers it runs the
C runtime through WASM, with optional WebGPU execution.

Use this package when JavaScript owns the application flow but tensor execution
should still come from the shared Polygrad compiler/runtime: Node services,
browser ML tools, WebGPU demos, or packages that accept a caller-provided
Polygrad runtime.

## Install

```bash
npm install polygrad
```

Node tries to build the native addon during install. If that fails, runtime
creation can still use the WASM core. To skip native compilation:

```bash
POLYGRAD_SKIP_NATIVE=1 npm install polygrad
```

From this repository:

```bash
cd js
npm install
node test/test_wasm.js
```

## Quick Start

```js
const { Tensor } = require('polygrad')

const a = Tensor.rand(3, 4)
const b = Tensor.rand(4, 5)
const c = a.dot(b).softmax(-1)

console.log(c.toArray())
```

Autograd:

```js
const { Tensor } = require('polygrad')

const x = new Tensor([1.0, 2.0, 3.0])
const loss = x.mul(x).sum()

loss.backward()
console.log(x.grad.toArray())  // [2, 4, 6]
```

Explicit runtime:

```js
const polygrad = require('polygrad')

const pg = polygrad.create({ core: 'wasm' })
const y = new pg.Tensor([1, 2, 3]).mul(2).add(1)
console.log(y.toArray())
pg.dispose()
```

Linear algebra:

```js
const { Tensor } = require('polygrad')

const A = new Tensor([[4.0, 2.0], [2.0, 5.0]])
const b = new Tensor([1.0, 3.0])
const x = A.solve(b)

console.log(x.toArray())
```

Structured linalg methods are portable tensor-composed fallbacks. Current
`lstsq` is solution-only for full-rank tall or square systems.

## Choose A Runtime

The default JavaScript API is sync-first. `polygrad.create(...)` returns a
`PolyRuntime` immediately or throws; tensor construction and graph construction
are synchronous too.

Use explicit async startup when the host cannot or should not instantiate the
WASM core synchronously, for example older browser limits or a fetch/streaming
loader path. The async entry is not a drop-in default `Tensor` import because
startup itself must be awaited:

```js
const { createAsync } = require('polygrad/async')
const pg = await createAsync({ core: 'wasm' })
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
POLY_DEVICE=cuda node app.js
```

For the native core, `POLY_DEVICE` is handled by the C runtime and may select
`cpu`, `cuda`, `hip`, `x86`, or `interp`. For the WASM core, device choices are
`wasm`, `interp`, and browser `webgpu` when available.

Call `pg.dispose()` when an application or long-running script is done with an
explicit runtime.

## Browser

With an npm-installed package and a browser bundler:

```js
import { Tensor } from 'polygrad'

const y = new Tensor([1, 2, 3]).mul(2).add(1)
console.log(y.toArray())
```

For WebGPU, create an explicit runtime and use async execution/readback:

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
entry explicitly:

```js
import { create } from 'polygrad/browser'
```

For a local checkout or manual browser bundle, build browser artifacts from the
repository root:

```bash
make wasm-pkg
cd js
npm run build:browser
```

Outputs:

- `dist/polygrad.sync.js` for a sync browser global.
- `dist/polygrad.sync.mjs` for sync browser ESM.
- `dist/polygrad.async.js` for an explicit async-startup browser global.
- `dist/polygrad.async.mjs` for explicit async-startup browser ESM.

Browser global:

```html
<script src="./dist/polygrad.sync.js"></script>
<script>
  const y = new polygrad.Tensor([1, 2, 3]).mul(2)
  console.log(y.toArray())
  polygrad.disposeDefault()
</script>
```

Local browser ESM with WebGPU:

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

```html
<script type="module">
  import { createAsync } from './dist/polygrad.async.mjs'

  const pg = await createAsync({ core: 'wasm' })
  const y = new pg.Tensor([1, 2, 3]).mul(2)
  console.log(y.toArray())
  pg.dispose()
</script>
```

## Data Flow

Polygrad tensors are lazy. Build expressions freely, then call `realize()` or
read data back.

```js
const { Tensor } = require('polygrad')

const x = new Tensor(new Float32Array([1, 2, 3, 4]), { shape: [4] })
const y = x.mul(3).sub(1).realize()

console.log(y.toTypedArray())  // Float32Array
```

For repeated loops, reuse tensor buffers instead of constructing new source
tensors:

```js
const { Tensor } = require('polygrad')

const x = new Tensor(new Float32Array([1, 2, 3, 4]), { shape: [4] })
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
const compiled = compile(
  (x) => x.add(1).realize(),
  [new Tensor([1, 2, 3])]
)

const out = compiled.run([new Tensor([7, 8, 9])])
console.log(out.toArray())
console.log(compiled.stats())
compiled.dispose()
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
  return out.index(i).store(a.index(i).add(b.index(i))).end(i).sink()
}

const out = Tensor.empty([4], { dtype: 'float32' })
const y = out.customKernel(
  new Tensor([1, 2, 3, 4]),
  new Tensor([10, 20, 30, 40]),
  addKernel
)[0]
console.log(y.toArray())
```

This is a UOp `CALL` extension point, not a raw program-launch API. Custom
backward functions are not implemented yet.

## Common API Recipes

Default runtime:

```js
const { Tensor, jit, compile, nn } = require('polygrad')

const x = new Tensor([1, 2, 3])
console.log(x.mul(2).add(1).toArray())
```

Explicit runtime and device selection:

```js
const polygrad = require('polygrad')
const pg = polygrad.create({ core: 'wasm' })

const x = new pg.Tensor([1, 2, 3])
console.log(x.mul(2).toArray())
pg.dispose()
```

Create tensors. Shape constructors accept either `Tensor.zeros(2, 3)` or `Tensor.zeros([2, 3])`:

```js
const a = Tensor.zeros([2, 3])
const b = Tensor.ones([2, 3])
const c = Tensor.randn([2, 3])
const d = Tensor.arange(6).reshape(2, 3)
const e = new Tensor(new Float32Array([1, 2, 3, 4]), { shape: [2, 2] })
```

Math, movement, indexing:

```js
const x = Tensor.arange(12).reshape(3, 4)
const y = x.permute(1, 0).reshape(2, 6)
const z = y.relu().sum(1)
const picked = x.gather(1, new Tensor([[0, 2], [1, 3], [0, 1]], { dtype: 'int32' }))
```

Autograd and optimization:

```js
const model = new nn.Linear(4, 1)
const opt = new nn.SGD(nn.getParameters(model), { lr: 0.01 })

const x = Tensor.randn(8, 4)
const target = Tensor.randn(8, 1)

opt.zeroGrad()
const loss = model.call(x).sub(target).square().mean()
loss.backward()
opt.step()
```

Linear algebra:

```js
const A = new Tensor([[4, 2], [2, 5]])
const b = new Tensor([1, 3])

console.log(A.solve(b).toArray())
console.log(A.cholesky().toArray())
console.log(A.lstsq(b).toArray())
```

JIT and compile:

```js
const f = jit((x) => x.add(1).realize())
f(new Tensor([1, 2, 3]))  // run
f(new Tensor([4, 5, 6]))  // capture
console.log(f(new Tensor([7, 8, 9])).toArray())  // replay

const compiled = compile((x) => x.mul(2).realize(), [Tensor.empty(3)])
console.log(compiled.run([new Tensor([1, 2, 3])]).toArray())
compiled.dispose()
```

Repeated input updates:

```js
const x = new Tensor(new Float32Array([1, 2, 3]), { shape: [3] }).realize()
const f = compile((x) => x.square().sum().realize(), [x])

console.log(f.run([x]).item())
x.copyFrom(new Float32Array([4, 5, 6]))
console.log(f.run([x]).item())
f.dispose()
```

Readback:

```js
const y = new Tensor([1, 2, 3]).mul(2)
console.log(y.toArray())              // typed array on sync runtimes
console.log(y.tolist())               // nested JS arrays
console.log(y.item())                 // scalar tensors only

const [a, b] = Tensor.toTypedArrays(y, y.add(1))
```

WebGPU readback is explicit async:

```js
const polygrad = require('polygrad')
const pg = polygrad.create({ core: 'wasm', device: 'webgpu' })
const y = new pg.Tensor([1, 2, 3]).mul(2)
console.log(await y.toArrayAsync())
pg.dispose()
```

Runtime inspection:

```js
const polygrad = require('polygrad')

console.log(polygrad.stats())
polygrad.resetCounters()
console.log(polygrad.canRun({ op: 'add', shape: [1024] }))
```

`stats().coreStats` includes `globalOps`, `globalMem`, `timeSumS`,
`kernelCount`, and live `memUsed`. `resetCounters()` clears execution totals
without clearing live allocation accounting.

`canRun(...)` is conservative. For some compound op/shape queries it throws
when support cannot be proven statically.

API reference at a glance:

| Area | Main APIs |
|---|---|
| Runtime | `create`, `createAsync`, `disposeDefault`, `stats`, `canRun` |
| Tensor creation | `new Tensor(data)`, `zeros`, `ones`, `full`, `rand`, `randn`, `eye`, `arange`, `empty` |
| Tensor math | `add`, `sub`, `mul`, `div`, `exp`, `log`, `sqrt`, `relu`, `gelu`, `silu`, `softmax` |
| Reductions | `sum`, `mean`, `max`, `argmax`, `sort`, `argsort`, `topk`, `var`, `std` |
| Movement/indexing | `reshape`, `expand`, `permute`, `shrink`, `flip`, `pad`, `cat`, `stack`, `gather`, `takeAlongAxis`, `scatter`, `scatterReduce` |
| Linalg | `dot`, `qr`, `triangularSolve`, `solveTriangular`, `cholesky`, `choleskySolve`, `solve`, `lstsq` |
| Data/readback | `realize`, `realizeAsync`, `toArray`, `toArrayAsync`, `toTypedArray`, `toTypedArrayAsync`, `toTypedArrays`, `toTypedArraysAsync`, `copyFrom`, `updateFrom` |
| Compilation | `jit`, `jitAsync`, `compile`, `compileAsync`, `Tensor.customKernel` |
| Neural nets | `nn.Linear`, `nn.SGD`, `nn.Adam`, `nn.AdamW`, `nn.getParameters`, `nn.getStateDict` |

## Package Integration

If your package is built on Polygrad, accept a `PolyRuntime` from the caller
instead of creating a hidden runtime:

```js
function createModel({ polygrad: pg }) {
  const weight = pg.Tensor.randn([4, 2]).realize()
  const predict = pg.compile((x) => x.dot(weight).realize(), [
    pg.Tensor.empty([1, 4])
  ])

  return {
    predict: (x) => predict.run([x]),
    dispose: () => predict.dispose()
  }
}
```

This lets applications share one set of runtime caches, device handles, and
buffer residency across packages.

## Tests

From the repository root:

```bash
make test-js-native
TMPDIR=$PWD/temp/cc_tmp EM_CACHE=$PWD/temp/emscripten-cache make test-js-wasm
DISPLAY=:1 make test-browser
make test-browser-matrix
```

`test-browser` keeps the default Chromium/WebGPU path. `test-browser-matrix`
runs the browser sync bundle against a non-WebGPU smoke matrix and skips local
browser executables that are not installed. Override the matrix when you need a
specific browser or older Chrome binary:

```bash
BROWSER_MATRIX="chromium,firefox,old-chrome=chromium@/path/to/chrome" make test-browser-matrix
BROWSER_MATRIX_DEVICES="auto,interp" make test-browser-matrix
```

Browser specs are `chromium`, `firefox`, `webkit`, `chrome`, or
`label=engine@/absolute/path`. WebGPU coverage remains Chromium-only and lives
in `make test-browser`.

## License

MIT
