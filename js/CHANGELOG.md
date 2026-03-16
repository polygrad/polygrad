# Changelog

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
