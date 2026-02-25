---
title: Home
nav_order: 1
---

# Polygrad

A C11 port of tinygrad's compiler core. Same IR, same pattern-matcher-driven rewrites, same codegen pipeline — but in plain C11.

**Why C11?** A single library that every language can call natively:

```
              ┌── Python (ctypes / cffi)
              ├── JavaScript (WASM)
polygrad (C11) ──┼── Rust (FFI)
              ├── Go (cgo)
              ├── Julia (ccall)
              └── Any language with a C FFI
```

tinygrad is Python-only. To use it from Rust, JS, or a compiled training recipe (llm.c-style), you'd need to either rewrite it or bridge through Python. Polygrad eliminates that: one C11 library, all platforms, no runtime dependency.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                       libpolygrad (C11)                          │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐  ┌────────────────┐ │
│  │  UOp IR  │  │ Pattern  │  │ Scheduler  │  │   Codegen      │ │
│  │          │  │ Matcher  │  │            │  │                │ │
│  │ ops.c    │  │ pat.c    │  │ rangeify.c │  │ linearize      │ │
│  │ dtype.c  │  │ sym.c    │  │ indexing.c │  │ render_c.c     │ │
│  │ uop.c    │  │ alu.c    │  │ sched.c    │  │ render_wasm.c  │ │
│  │ arena.c  │  │          │  │ shape.c    │  │ wasm_builder.c │ │
│  │ hashmap.c│  │          │  │            │  │ codegen.c      │ │
│  └──────────┘  └──────────┘  └────────────┘  │ runtime_cpu.c  │ │
│                                               └────────────────┘ │
│  ┌──────────┐  ┌──────────────────────────────────────────────┐  │
│  │ Autograd │  │ Frontend (FFI surface + ~35 composed ops)    │  │
│  │autograd.c│  │ frontend.c                                   │  │
│  └──────────┘  └──────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
        │                    │                    │
  ┌─────┴─────┐     ┌───────┴──────┐    ┌───────┴──────────┐
  │  Python   │     │  JS (Node)   │    │ Browser (WASM)   │
  │  ctypes   │     │  koffi FFI   │    │ Emscripten FFI   │
  │  tensor.py│     │  tensor.js   │    │ index.js         │
  │  nn/      │     │              │    │                  │
  └───────────┘     └──────────────┘    └──────────────────┘
```

**What works today:** Full tinygrad-compatible Tensor API from Python, JS, and Browser. C core handles: UOp IR → schedule → codegen → linearize → render (C or WASM) → execute. Elementwise ops (~20), reductions (sum, max, mean, var, std), matmul, softmax, movement ops (reshape, expand, permute, shrink, flip, pad), reverse-mode autograd, multi-kernel scheduling, in-place buffer writes (ASSIGN + WAR/WAW ordering). Python `nn` module: Linear, LayerNorm, RMSNorm, Embedding, Dropout + SGD/Adam/AdamW optimizers (all using `assign()` for in-place parameter updates). GPT-2 model training (segment-wise backward through realize boundaries). 31/31 IR parity with tinygrad.

**What's next:** GPU backends (WGSL/WebGPU, CUDA, Metal).

## Documentation

| Frontend | API Docs | Install |
|----------|----------|---------|
| [Python](py/) | Tensor, nn module, optimizers | `pip install -e py/` |
| [JavaScript (Node)](js/) | Tensor (koffi FFI) | `cd js && npm install` |
| [Browser (WASM)](browser/) | Tensor (Emscripten) | `make wasm` |
| [R](r/) | Tensor (.Call FFI) | `R CMD INSTALL r/` |

## Parity

Current parity snapshot against tinygrad `ClangRenderer` (CPU, `DEVECTORIZE=0`, `optimize=False`):
- Value parity: **31/31**
- Full IR parity (kernel count + structure + op sequence): **31/31**

The parity test (`make test-parity`) schedules each case through both polygrad and tinygrad, linearizes, and compares:
- kernel count
- per-kernel structural signature (`RANGE/END/REDUCE/INDEX/LOAD/STORE` counts and loop-depth balance)
- full per-kernel op sequence

The parity suite covers 31 cases:

| Category | Cases |
|----------|-------|
| Elementwise | vecadd, chain, broadcast_scalar, neg_1d, mul_1d, sqrt_1d, where_1d |
| Transcendental | exp2_1d (73-op polynomial decomposition) |
| Reductions | reduce_sum_all, reduce_max_1d, reduce_sum_axis1, reshape_reduce, expand_alu_reduce |
| Multi-kernel | reduce_scalar_chain, reduce_vector_chain, shared_scalar_reduce_branches |
| Movement | permute_2d, shrink_2d, pad_2d, chain_pad_flip, multi_movement |
| Autograd | grad_mul_sum, grad_exp2_sum, grad_fdiv_sum_x, grad_fdiv_sum_y, grad_chain_movement, grad_log2_sum, grad_sqrt_sum, grad_where_sum, grad_multi_use |
| NN | matmul_small |

```bash
# Run parity tests (requires conda env 'tiny' with tinygrad)
make test-parity

# Dump IR for a specific case
ASAN_OPTIONS=detect_leaks=0 CACHELEVEL=0 \
  conda run -n tiny python test/test_tinygrad_parity.py \
  --runner build/polygrad_parity_runner --mode full --no-opt --dump vecadd
```

## Core/API parity contract

This project tracks parity against tinygrad commit `c2be31e75b366638965337b96f2c66c2ba8c4068`.

`Core parity` gates:
- Differential parity (`make test-parity`) must pass in full mode (`--mode full --no-opt`).
- No silent scheduler/indexing fallbacks for invalid mappings; failures must be explicit.
- Any intentional divergence from tinygrad must be documented in `Whats different` with reason.

`API parity` gates:
- Tensor semantics must match tinygrad for forward and first-order backward on shared coverage.
- Python/JS frontend tests and C core tests must all pass before parity claims are updated.

## Whats different
| polygrad | tinygrad | reason |
|----------|----------|--------|
| Segment-wise backward through `realize()` boundaries | Single-pass backward with lazy scheduling | polygrad's rangeify scheduler can't fuse reduce→expand→alu patterns (softmax, layernorm, variance) in a single kernel, so `realize()` boundaries split the graph into segments that are differentiated independently via VJP chains |
| Kahn's algorithm for segment processing order | N/A (single pass) | Shared intermediates (e.g., qkv split into q/k/v in attention) must receive ALL upstream gradient contributions before processing; Kahn's topological sort ensures correct ordering |
| Numpy fallback for reduce-max backward | Autograd handles all ops uniformly | The MAX gradient rule creates reduce→expand→alu patterns that the scheduler can't realize correctly; the max backward is computed in numpy instead |
| Eager gradient realization (no double-backward) | Lazy gradient tensors (double-backward possible) | Gradients are realized eagerly per segment into numpy arrays, so `grad(grad(loss))` isn't possible. First-order optimization (SGD/Adam/AdamW) is unaffected |


## Building

```bash
make               # build libpolygrad.a + libpolygrad.so
make test          # build + run 277 C tests (ASan/UBSan)
make test-wasm     # build + run 47 WASM tests
make test-parity   # 1-to-1 differential parity tests vs tinygrad reference
make bench         # build + run benchmark

# Frontend tests
python -m pytest py/tests/ -v   # 87 Python tests
node js/test/test_tensor.js          # 62 JS tests
```

## Status

### C Core
- [x] UOp IR (ops enum, dtype, arena, hashmap, CSE, toposort)
- [x] Pattern matcher + symbolic simplification (~25 rules)
- [x] Linearizer + C renderer + CPU runtime (end-to-end)
- [x] Shape inference + tensor-to-kernel scheduler (elementwise, reshape, expand, broadcast)
- [x] WASM binary renderer (scalar f32 + f32x4 SIMD) + binary builder
- [x] Reduce ops (REDUCE_AXIS → accumulation kernel: sum, max, product)
- [x] Movement ops (PERMUTE, SHRINK, FLIP, PAD index transforms)
- [x] Reverse-mode autograd core (`poly_grad`) for ALU/movement/reduce-sum paths
- [x] Rangeify + indexing pipeline (consumer map, realize detection, range propagation, multi-kernel BUFFERIZE)
- [x] Multi-kernel execution (`poly_schedule_v2` → `poly_realize` with intermediate buffer allocation)
- [x] Codegen pipeline (`full_rewrite_to_sink`): pm_reduce, pm_decomp, pm_transcendental, pm_add_control_flow
- [x] Frontend composed ops (~35): elementwise math, comparisons, matmul, softmax, layernorm, cross_entropy
- [x] **31/31 IR parity** with tinygrad ClangRenderer (CPU, no vectorization)
- [x] ASSIGN + WAR ordering (in-place buffer writes, WAR/WAW dependency edges, `Tensor.assign()`)

### Language Frontends
- [x] Python Tensor API (ctypes FFI, lazy eval, autograd, tinygrad-compatible)
- [x] Python nn module (Linear, LayerNorm, RMSNorm, Embedding, Dropout, SGD, Adam, AdamW)
- [x] GPT-2 model training (segment-wise backward through realize boundaries)
- [x] JS Node Tensor API (koffi FFI, full op coverage)
- [x] Browser WASM Tensor API (Emscripten FFI, same API as Node)

### Planned
- [ ] GPU backends (WGSL/WebGPU, Metal)
- [ ] Expander, devectorizer

473 tests (277 C + 87 Python + 62 JS + 47 WASM), ASan/UBSan clean.

## Reference

Port of [tinygrad](https://github.com/tinygrad/tinygrad) commit `c2be31e75b366638965337b96f2c66c2ba8c4068`.
