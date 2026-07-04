# Polygrad JavaScript

JavaScript bindings for Polygrad. The package exposes the shared C11 tensor
compiler through a Node native addon when available, with a packaged WASM
fallback for Node and browser use.

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
const polygrad = require('polygrad')

;(async () => {
  const pg = await polygrad.create()
  const { Tensor } = pg

  const x = new Tensor([1, 2, 3])
  const y = x.mul(2).add(1)
  console.log(await y.toArray())  // [3, 5, 7]

  await pg.dispose()
})().catch(console.error)
```

Autograd:

```js
const a = new pg.Tensor([2, 3], { requiresGrad: true })
const loss = a.mul(a).sum()

await loss.backward()
console.log(await a.grad.toArray())  // [4, 6]
```

Linear algebra:

```js
const A = new pg.Tensor([[4, 2], [2, 5]])
const b = new pg.Tensor([1, 3])
const x = A.solve(b)

console.log(await x.toArray())
```

## Runtime Selection

```js
const pg = await polygrad.create()
const pgNative = await polygrad.create({ core: 'native' })
const pgWasm = await polygrad.create({ core: 'wasm' })
const pgWebgpu = await polygrad.create({ core: 'wasm', device: 'webgpu' })
```

| Environment | `core: "auto"` | `core: "native"` | `core: "wasm"` |
|---|---|---|---|
| Node with native addon | native | native | wasm |
| Node without native addon | wasm | error | wasm |
| Browser | wasm | error | wasm |

Environment variables:

```bash
POLY_CORE=wasm node app.js
POLY_DEVICE=interp node app.js
```

For the native core, `POLY_DEVICE` is handled by the C runtime and may select
`cpu`, `cuda`, `hip`, `x86`, or `interp`. For the WASM core, device choices are
`wasm`, `interp`, and browser `webgpu` when available.

## Browser Bundles

Build browser bundles from the repository root:

```bash
make wasm-pkg
cd js
npm run build:browser
```

Outputs:

- `dist/polygrad.js` for a browser global.
- `dist/polygrad.mjs` for browser ESM.

Browser global:

```html
<script src="./dist/polygrad.js"></script>
<script>
  polygrad.create().then(async (pg) => {
    const y = new pg.Tensor([1, 2, 3]).mul(2)
    console.log(await y.toArray())
    await pg.dispose()
  })
</script>
```

Browser ESM:

```html
<script type="module">
  import { create } from './dist/polygrad.mjs'

  const pg = await create({ device: 'webgpu' })
  const y = new pg.Tensor([1, 2, 3]).mul(2)
  console.log(await y.toArray())
  await pg.dispose()
</script>
```

## API Overview

`polygrad.create(opts?)` returns a `Promise<PolyRuntime>`.

Options:

- `core`: `auto`, `native`, or `wasm`.
- `device`: `auto`, `cpu`, `cuda`, `hip`, `x86`, `interp`, `wasm`, or `webgpu`,
  depending on the selected core and local runtime support.

Runtime fields:

- `pg.Tensor`: runtime-bound Tensor class.
- `pg.nn`: `Linear`, `SGD`, `Adam`, `AdamW`, and parameter helpers.
- `pg.jit(fn)`: tinygrad-style first-run, capture, replay wrapper.
- `pg.compile(fn, sampleInputs)`: explicit wrapper over the same JIT path.
- `pg.stats()`: wrapper and C runtime counters.
- `pg.canRun(query)`: advisory backend capability probe.
- `pg.uop`: UOp helpers for inspection and custom kernels.
- `pg.dispose()`: release runtime resources.

Tensor methods include:

| Category | Methods |
|---|---|
| Creation | `new Tensor(data)`, `zeros`, `ones`, `full`, `rand`, `randn`, `eye`, `arange` |
| Elementwise | `add`, `sub`, `mul`, `div`, `neg`, `exp`, `log`, `sqrt`, `square`, `relu`, `gelu`, `silu` |
| Reductions | `sum`, `mean`, `max`, `argmax`, `sort`, `argsort`, `topk`, `var`, `std`, `softmax` |
| Movement | `reshape`, `expand`, `permute`, `shrink`, `flip`, `pad`, `cat`, `gather`, `takeAlongAxis` |
| Linalg | `dot`, `qr`, `triangularSolve`, `solveTriangular`, `cholesky`, `choleskySolve`, `solve`, `lstsq` |
| Data | `realize`, `toArray`, `toTypedArray`, `toTypedArrays`, `copyFrom`, `updateFrom`, `repr`, `customKernel` |

Structured linalg methods are portable tensor-composed fallbacks. Current
`lstsq` is solution-only for full-rank tall or square systems.

## JIT And Compile

`pg.jit(fn)` follows tinygrad raw Tensor JIT behavior:

```js
const f = pg.jit((x) => x.add(1).realize())

await f(new pg.Tensor([1, 2, 3]))  // normal run
await f(new pg.Tensor([4, 5, 6]))  // capture
console.log(await (await f(new pg.Tensor([7, 8, 9]))).toArray())  // replay
```

`pg.compile(fn, sampleInputs)` warms and captures immediately:

```js
const compiled = await pg.compile(
  (x) => x.add(1).realize(),
  [new pg.Tensor([1, 2, 3])]
)

const out = await compiled.run([new pg.Tensor([7, 8, 9])])
console.log(await out.toArray())
console.log(compiled.stats())
compiled.dispose()
```

## Custom Kernels

`Tensor.customKernel(...)` mirrors tinygrad's alpha custom-kernel shape. The
kernel function receives placeholder UOps and returns a `SINK` body. Polygrad
wraps the body in `CALL`, returns `AFTER(...)` tensors, and keeps execution in
the normal schedule and runtime caches.

```js
function addKernel(out, a, b) {
  out = out.flatten(); a = a.flatten(); b = b.flatten()
  const i = pg.uop.range(out.numel(), 0)
  return out.index(i).store(a.index(i).add(b.index(i))).end(i).sink()
}

const out = pg.Tensor.empty([4], { dtype: 'float32' })
const y = out.customKernel(
  new pg.Tensor([1, 2, 3, 4]),
  new pg.Tensor([10, 20, 30, 40]),
  addKernel
)[0]
console.log(await y.toArray())
```

This is a UOp `CALL` extension point, not a raw program-launch API. Custom
backward functions are not implemented yet.

## Testing

From the repository root:

```bash
make test-js-native
TMPDIR=$PWD/temp/cc_tmp EM_CACHE=$PWD/temp/emscripten-cache make test-js-wasm
DISPLAY=:1 make test-browser
```

## License

MIT
