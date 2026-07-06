# Changelog

## 0.4.2 (2026-07-06)

### Added
- Sync-first default JavaScript API: `require('polygrad').Tensor`, synchronous `create(...)`, synchronous CPU/native/WASM readback, and explicit `createAsync(...)` / `toArrayAsync()` paths.
- Split browser bundles: `polygrad.sync.*` for normal sync startup and `polygrad.async.*` for explicit async WASM startup.
- Package export tests covering Node CJS, Node ESM, browser-condition imports, sync browser subpaths, and async subpaths.

### Fixed
- WebGPU initialization is lazy and happens at async realization/readback rather than runtime construction.
- WebGPU `caps.f16` now reflects `shader-f16` support after lazy device initialization.
- C WASM tests probe Node relaxed-SIMD execution instead of hardcoding a V8 flag, and WASM matmul emits relaxed-madd with the correct `(lhs * rhs) + acc` operand order on the ABT path.

## 0.4.0 (2026-07-05)

### Added
- `Tensor.customKernel(...)` with public UOp helpers, grouped stores, reductions, comparisons, unary ops, compile/JIT replay, and repeated input mutation coverage.
- Structured tensor APIs for gather/scatter, sort/argsort/topk, linalg solve helpers, typed readback, and stable input updates.
- Browser/WebGPU, JS WASM, native CPU/x86, and native CUDA coverage for common tensor and custom-kernel paths.
- Browser and Node benchmarks for jax-js WASM comparisons, matmul, model-shaped kernels, and Qwen3 smoke tests.

### Changed
- `create({ core: 'native', device })` now honors the requested device in runtime construction and instance placement.
- Browser/WebGPU continues through the unified C WASM runtime path rather than separate JavaScript executors.

### Fixed
- CUDA compiled replay now rebinds custom-kernel producer outputs for downstream compiled consumers.
- WebGPU WGSL rendering now handles `COPY`/`UNROLL`-wrapped multi-output custom-kernel stores and selects the correct lane for unrolled/vector values.
- WASM and x86 lowering fixes for larger custom-kernel reductions and vector/lane rendering.
- `canRun()` now reports sort, argsort, and topk support consistently with direct execution.

## 0.3.0

### Changed
- Aligns the npm package with the current shared C core generation and Python frontend release.
- Ships the unified realization/runtime path, Instance/model/optimizer APIs, and updated browser/WASM artifacts.

## 0.2.0

Merged `polygrad` (WASM) and `polygrad-node` (koffi FFI) into a single package.

### Breaking changes

- Package entry point is now `await polygrad.create()` returning a `PolyRuntime`.
  The old `init()` + global `Tensor` export is removed.
- Native backend uses Node-API (N-API) instead of koffi. No runtime dependency on koffi.
- `Instance` API (model loading, training, weight I/O) is deferred to 0.3.0.

### New

- `polygrad.create({ target, device })` -- async factory returning `PolyRuntime` with runtime-bound `Tensor`.
- Backend auto-detection: native (N-API addon) preferred, WASM fallback.
- `POLY_TARGET` and `POLY_DEVICE` env vars to force runtime selection (`POLY_BACKEND` still works as a compatibility alias for `POLY_TARGET`).
- Best-effort native build on install (never breaks `npm install`).
- Browser support via dedicated browser bundles in `dist/polygrad.js` and `dist/polygrad.mjs`.

### Target selection

| Environment | `target: 'auto'` | `target: 'native'` | `target: 'wasm'` | `device` |
|---|---|---|---|---|
| Node.js with addon | Native | Native | WASM | `cpu` today |
| Node.js without addon | WASM | Error | WASM | `cpu` today |
| Browser | WASM | Error | WASM | `cpu` today |

## 0.1.0

Initial release as two separate packages (`polygrad` + `polygrad-node`).
