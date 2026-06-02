# polygrad

Tensor computation library with automatic differentiation. Native (N-API) and WASM backends.

## Install

```bash
npm install polygrad
```

On Node.js, the native addon builds automatically during install. If the build fails (missing compiler, etc.), the package falls back to the WASM backend. Set `POLYGRAD_SKIP_NATIVE=1` to skip the native build entirely.

## Usage

```js
const polygrad = require('polygrad')

;(async () => {
  const pg = await polygrad.create()  // auto core: native > wasm
  const { Tensor } = pg

  const x = new Tensor([1, 2, 3])
  const y = x.mul(2).add(1)
  console.log(await y.toArray())  // [3, 5, 7]

  // Autograd
  const a = new Tensor([2, 3], { requiresGrad: true })
  const loss = a.mul(a).sum()
  await loss.backward()
  console.log(await a.grad.toArray())  // [4, 6]

  await pg.dispose()
})().catch(console.error)
```

## Core and device selection

```js
// Force a specific core
const pg = await polygrad.create({ core: "native" })
const pg = await polygrad.create({ core: "wasm" })

// Environment variables
// POLY_CORE=wasm node app.js
// POLY_DEVICE=wasm node app.js
```

| Environment | `core: "auto"` | `core: "native"` | `core: "wasm"` | default device |
|---|---|---|---|---|
| Node.js with addon | Native | Native | WASM | `cpu` for native, `wasm` for WASM |
| Node.js without addon | WASM | Error | WASM | `wasm` |
| Browser | WASM | Error | WASM | `wasm` |


## Browser bundles

Build the one-file browser bundles after generating the packaged WASM module:

```bash
make -C .. wasm-pkg
npm run build:browser
```

This produces:

- `dist/polygrad.js` -- browser global bundle for `<script src="...">`
- `dist/polygrad.mjs` -- browser ESM bundle for `<script type="module">`

Example:

```html
<script src="./dist/polygrad.js"></script>
<script>
  polygrad.create().then(async (pg) => {
    const x = new pg.Tensor([1, 2, 3])
    console.log(await x.mul(2).toArray())
    await pg.dispose()
  })
</script>
```

## API

### `polygrad.create(opts?)` -> `Promise<PolyRuntime>`

Creates a runtime with the selected core. Options:

- `core`: `'auto'` (default), `'native'`, or `'wasm'`
- `device`: `'auto'` (default), `'cpu'`, `'wasm'`, `'interp'`, `'webgpu'` for the WASM core; native currently reports `'cpu'`

### `PolyRuntime`

- `pg.Tensor` -- runtime-bound Tensor class
- `pg.core` -- `'native'` or `'wasm'`
- `pg.device` -- resolved device, e.g. `'cpu'` for native or `'wasm'` for the WASM core
- `pg.dispose()` -- release resources

### `Tensor`

Creation: `new Tensor(data, opts?)`, `Tensor.zeros(...)`, `Tensor.ones(...)`, `Tensor.rand(...)`, `Tensor.randn(...)`, `Tensor.eye(n)`, `Tensor.arange(start, stop, step?)`

Elementwise: `add`, `sub`, `mul`, `div`, `neg`, `exp`, `log`, `sqrt`, `square`, `abs`, `sigmoid`, `relu`, `tanh`, `gelu`, `silu`

Reduce: `sum`, `mean`, `max`, `var`, `std`, `softmax`

Movement: `reshape`, `expand`, `permute`, `shrink`, `flip`, `pad`, `cat`

Comparison: `eq`, `gt`, `where`, `clamp`, `maximum`

Other: `dot` (matmul), `backward`, `realize`, `toArray`, `tolist`, `repr`

## License

MIT
