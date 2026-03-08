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

const pg = await polygrad.create()  // auto: native > wasm
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
```

## Backend selection

```js
// Force a specific backend
const pg = await polygrad.create({ backend: 'native' })
const pg = await polygrad.create({ backend: 'wasm' })

// Or via environment variable
// POLY_BACKEND=wasm node app.js
```

| Environment | `'auto'` (default) | `'native'` | `'wasm'` |
|---|---|---|---|
| Node.js with addon | Native | Native | WASM |
| Node.js without addon | WASM | Error | WASM |
| Browser | WASM | Error | WASM |

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

Creates a runtime with the selected backend. Options:

- `backend`: `'auto'` (default), `'native'`, or `'wasm'`

### `PolyRuntime`

- `pg.Tensor` -- runtime-bound Tensor class
- `pg.backend` -- `'native'` or `'wasm'`
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
