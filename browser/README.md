# Polygrad Browser (WASM)

Run polygrad tensors in the browser via WebAssembly. Same API as the Node.js package, compiled with Emscripten.

## How It Works

Two-tier WASM architecture:

1. **Core WASM module** (`polygrad.js` + `polygrad.wasm`): The C core compiled to WASM via Emscripten. Handles graph construction, scheduling, linearization, and WASM kernel rendering.
2. **Per-kernel WASM**: Each `realize()` call renders a WASM binary kernel from the core, then instantiates and executes it via `WebAssembly.instantiate()`.

This means polygrad generates WASM from WASM at runtime — the core compiler runs in WASM, and it produces WASM kernels for the actual tensor computations.

## Build

```bash
# From repo root — requires Emscripten (emcc)
make wasm
# Produces: build/polygrad.js + build/polygrad.wasm
```

## Usage (Node.js)

```javascript
const { Tensor } = require('./browser')

// Initialize WASM module (async, required once)
await Tensor._init()

// Then use the same API as the Node.js package
const a = Tensor.rand(3, 4)
const b = Tensor.rand(4, 5)
const c = a.matmul(b).softmax(-1)
console.log(c.toArray())
```

## Usage (Browser)

```html
<script src="polygrad.js"></script>
<script>
  PolygradModule().then(Module => {
    // Module is the raw Emscripten module
    // Use the Tensor API from browser/src/index.js
  })
</script>
```

## API

Same Tensor API as the [Node.js package](../js/). All methods: `add`, `sub`, `mul`, `div`, `matmul`, `softmax`, `relu`, `sigmoid`, `reshape`, `permute`, `sum`, `mean`, `backward`, `einsum`, `rearrange`, etc.

Key differences from Node.js:
- Requires async `_init()` before use (WASM module loading)
- Uses Emscripten FFI instead of koffi
- `realize()` instantiates a WebAssembly kernel per operation

## Visual Benchmark

Compare polygrad vs tf.js in the browser:

- Correctness histograms (add, mul, neg, exp2, sqrt, log2)
- Operation speed (add, mul, sum across sizes)
- Gradient-descent convergence
- Autograd spot checks
- Library/kernel size comparison

```bash
# Build WASM first
make wasm

# Start benchmark server
cd browser && npm run benchmark:visual

# Open the printed URL and click "Run All"
# Optional: PORT=9090 npm run benchmark:visual
```

## Tests

```bash
make test-wasm   # 47 tests via Node.js
```
