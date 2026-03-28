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
│                                               │ render_x64.c   │ │
│                                               └────────────────┘ │
│  ┌──────────┐  ┌──────────────────────────────────────────────┐  │
│  │ Autograd │  │ Frontend (FFI surface + ~35 composed ops)    │  │
│  │autograd.c│  │ frontend.c                                   │  │
│  └──────────┘  └──────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
        │                    │                    │
  ┌─────┴─────┐     ┌───────┴──────────────┐    ┌────────┴──────────┐
  │  Python   │     │  JS package (`js/`)  │    │ Browser dist       │
  │  ctypes   │     │  Node-API + WASM     │    │ dist/polygrad.js   │
  │  tensor.py│     │  src/index.js        │    │ dist/polygrad.mjs  │
  │  nn/      │     │  src/runtime.js      │    │ src/browser.js     │
  └───────────┘     └──────────────────────┘    └────────────────────┘
```

**What works today:** Full tinygrad-compatible Tensor API from Python and the unified JS package. C core handles: UOp IR -> schedule -> unified codegen pipeline -> render (C, x86-64 JIT, CUDA, HIP, WASM, interpreter) -> execute. Elementwise ops (~20), reductions (sum, max, mean, var, std), matmul, softmax, movement ops (reshape, expand, permute, shrink, flip, pad), step slicing (`t[::2]`, `t[::-1]`), reverse-mode autograd, multi-kernel scheduling, in-place buffer writes (ASSIGN + WAR/WAW ordering). Full float64/float16/bfloat16 support across all backends. The JS package uses `await polygrad.create({ target, device })`, prefers a native Node-API binding in Node, falls back to packaged WASM, and also ships prebuilt browser bundles. Python `nn` module: Linear, LayerNorm, RMSNorm, Embedding, Dropout + SGD/Adam/AdamW optimizers. HuggingFace model loading: load GPT-2 directly from config.json + safetensors, verified logit-exact match with HF Transformers. Value parity with tinygrad is 33/33; full IR parity is 31/33 with two remaining structural divergences. 607 C tests, 164 Python, 101 JS native, 95 JS WASM/browser. All five native backends (CPU, x64, CUDA, HIP, interpreter) pass the full test suite.

**Cross-platform execution:** `poly_realize()` dispatches through a backend vtable (CPU, x64 JIT, CUDA, HIP, interpreter, WASM JIT). All backends share one unified linearizer pipeline (`poly_full_rewrite_to_sink_ex`), with backend differences expressed via `PolyRewriteOpts`. The x64 JIT (`render_x64.c`) emits x86-64 machine code directly -- no C compiler dependency, zero compile latency, SSE2 packed vectorization. CUDA uses native `half`/`nv_bfloat16` types with h* intrinsics. HIP supports AMD MI250X with MFMA tensor core codegen. The interpreter supports vector operations via a lane-array value model with pre-allocated arena. Backend selection via `POLY_DEVICE=cpu|cuda|hip|x64|interp`. `PolyInstance` uses cached slot tables and calls `poly_compiled_plan_run()` directly -- zero per-step allocations in the training loop. The `poly.bundle@1` format packages IR + weights into a single portable file. Save in Python, load in JS (WASM or native) -- predictions match exactly.

**Codegen optimization:** Late pipeline matches tinygrad's architecture (codegen/__init__.py). Devectorizer scatters vectorized ALU ops to scalar, load/store folding regroups contiguous accesses into vector loads. `POLY_OPTIMIZE=1 POLY_DEVECTORIZE=1` enables UPCAST + devectorize for CPU SIMD. BEAM search optimizer (`POLY_BEAM=N`) explores the optimization space by compiling and timing candidates, with disk cache for results. All 607 tests pass in both default and optimized modes across all backends.

**What's next:** WASM build fix, WebGPU backend, more model families (LLaMA).

## Documentation

| Frontend | API Docs | Install |
|----------|----------|---------|
| [Python](py/) | Tensor, nn module, optimizers | `pip install polygrad` |
| [JavaScript + Browser](js/) | Unified npm package (`create({ target, device })`), Node-API + WASM, browser dist bundles | `npm install polygrad` |
| [R](r/) | Tensor (.Call FFI) | `R CMD INSTALL r/` |

## Versioning

Package versions use semver, but `major.minor` tracks the shared C core line across frontends. The `patch` version is frontend-specific, so Python and JS can ship wrapper-only fixes independently while still advertising the same underlying core generation. ABI compatibility is tracked separately by `POLYGRAD_ABI_VERSION` in `src/frontend.h`.

## Parity

Current parity snapshot against tinygrad `ClangRenderer` (CPU, `DEVECTORIZE=0`, `optimize=False`):
- Value parity: **33/33**
- Full IR parity (kernel count + structure + op sequence): **31/33**
- Remaining structural divergences: `matmul_broadcast`, `cross_entropy_nonlast_axis`

The parity test (`make test-parity`) schedules each case through both polygrad and tinygrad, linearizes, and compares:
- kernel count
- per-kernel structural signature (`RANGE/END/REDUCE/INDEX/LOAD/STORE` counts and loop-depth balance)
- full per-kernel op sequence

The parity suite covers 33 cases:

| Category | Cases |
|----------|-------|
| Elementwise | vecadd, chain, broadcast_scalar, neg_1d, mul_1d, sqrt_1d, where_1d |
| Transcendental | exp2_1d (73-op polynomial decomposition) |
| Reductions | reduce_sum_all, reduce_max_1d, reduce_sum_axis1, reshape_reduce, expand_alu_reduce |
| Multi-kernel | reduce_scalar_chain, reduce_vector_chain, shared_scalar_reduce_branches |
| Movement | permute_2d, shrink_2d, pad_2d, chain_pad_flip, multi_movement |
| Autograd | grad_mul_sum, grad_exp2_sum, grad_fdiv_sum_x, grad_fdiv_sum_y, grad_chain_movement, grad_log2_sum, grad_sqrt_sum, grad_where_sum, grad_multi_use |
| NN | matmul_small, matmul_broadcast, cross_entropy_nonlast_axis |

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
| Frontend creation helpers (`rand`, `randn`, `arange`, `full`, `eye`, `linspace`, `tril`, `triu`) use ctx-scoped constant-buffer auto-binding as a convenience path | Device-native RNG/creation flow in Tensor runtime | Track C kept this additive to avoid breaking `PolyStep`/FFI contracts; for large tensors or hot loops, prefer explicit buffer bindings/device-resident generation |
| `poly_rand` uses top-24-bit extraction (SHR 8, CAST f32, MUL 2^-24) producing 2^24 distinct uniform [0,1) values | Mantissa-bit randomization (set exponent=1, bitcast, subtract 1.0) | Simpler codegen path; both produce uniform [0,1) but float bit patterns differ for same THREEFRY output |

## RNG contract

`poly_rand` is deterministic given (seed, shape). No hidden stream state. The pipeline is:

1. Counter tensor: `[0, 1, ..., numel-1]` as uint32
2. Seed mixing: `mixed_key = key_lo ^ ((key_hi << 16) | (key_hi >> 16))` where `key_lo = seed & 0xffffffff`, `key_hi = seed >> 32`
3. `THREEFRY(counter, mixed_key)` -- threefry2x32 with 5 rounds, fully decomposed to integer ALU by codegen
4. Extract top 24 bits: `SHR(bits, 8)`
5. Convert: `CAST(uint32 -> float32) * (1.0 / 16777216.0)` producing uniform [0,1)

`poly_randn` applies Box-Muller to two `poly_rand` calls (second uses `seed ^ 0x9E3779B97F4A7C15`).

Tier 1 (bitwise): same platform + binary + seed -> identical floats. Tested by `c2c_rand_bitpattern_8` and `c2c_rand_determinism`.

Tier 2 (statistical): mean/variance within expected bounds. Tested by `c2c_rand_range_and_stats` and `c2c_randn_tails`.

## Performance

Polygrad compiles tensor operations into fused C kernels at runtime. A disk cache (`~/.cache/polygrad/`) persists compiled kernels across process restarts.

| Metric | Value |
|--------|-------|
| Cold compilation (first run) | ~170ms per kernel |
| Warm cache hit | ~0.5ms per kernel (300x+ speedup) |
| Tensor creation from numpy | Zero-copy (no memcpy for contiguous arrays) |
| numpy() readback | Zero-copy (returns view) |

### Kernel fusion

Polygrad fuses chains of element-wise operations into a single kernel that makes one pass over memory. Numpy executes each operation separately, allocating a temporary array for each intermediate result.

For a chain of N ops on an array of M elements:
- Numpy: N passes over memory, N temporary allocations (each M * 4 bytes for float32)
- Polygrad: 1 pass, 0 temporary allocations (intermediates stay in CPU registers)

**Crossover point:** polygrad beats numpy when the fused kernel's single-pass advantage outweighs the per-call scheduling overhead (~200us). This happens at large arrays with many ops:

```
N=10M float32 elements, fused chain r = r * a + b (repeated):

  ops    numpy       polygrad    speedup
  5      160ms       224ms       0.7x  (numpy faster -- scheduling overhead dominates)
  10     311ms       246ms       1.3x  (polygrad faster)
  20     860ms       267ms       3.2x  (polygrad faster)
```

Speedup = numpy_time / polygrad_time. Numpy time scales linearly with op count (one memory pass per op). Polygrad time stays nearly flat (one fused kernel regardless of op count).

**Memory:** For the 20-op chain on 10M elements (40MB per array), numpy allocates and frees 20 temporary arrays (800MB total transient allocation). Polygrad allocates only the input and output buffers (120MB total), with all intermediates in registers.

This is relevant for workloads like Monte Carlo simulation, where millions of samples pass through a chain of numeric transforms (pricing models, risk calculations). The compute phase fuses into a single kernel with bounded memory, regardless of model complexity.

### Relative benchmarks

`make bench-ratios` runs polygrad against numpy, tinygrad, and PyTorch on a standard workload suite. Results are reported as speedup ratios (polygrad_time / baseline_time) which are stable across hardware:

```
make bench-ratios       # run benchmarks, output JSON
make bench-compare      # compare to stored baseline, flag regressions >10%
make bench-regression   # run + compare (exit 1 on regression)
```

**JS (WASM vs native, instance API -- compile once, execute many):**

```
workload                 native    wasm    wasm/native
mlp_forward (in=64)        3us      5us    1.5x
mlp_forward (in=1024)     52us     55us    1.1x
mlp_forward (in=16384)   880us    881us    1.0x
mlp_train (in=64)          6us     17us    2.7x
mlp_train (in=1024)      104us    225us    2.2x
```

WASM JIT matches native CPU at larger model sizes. At small sizes, WASM dispatch overhead adds 1.5-2.7x.

Environment variables:
- `POLY_DEVICE=cpu|cuda|hip|x64|interp` -- backend selector (default: cpu)
- `POLY_OPT=0|1|2` -- optimization level for kernel compilation (default: 2)
- `POLY_OPTIMIZE=1` -- enable codegen optimizer (UPCAST, devectorize, BEAM)
- `POLY_BEAM=N` -- BEAM search width for optimization space exploration
- `POLY_CACHE=0` -- disable disk cache (always recompile)
- `POLY_DUMP_KERNELS=1` -- print generated kernel source (C, CUDA, HIP, or interp UOps)

## Building

```bash
make               # build libpolygrad.a + libpolygrad.so
make test           # build + run 607 C tests (ASan/UBSan) on CPU
make test-interp   # full suite on interpreter backend
make test-cuda     # full suite on CUDA (requires GPU)
make test-hip      # full suite on HIP/AMD (requires ROCm)
make test-x64      # full suite on x64 JIT backend
make test-wasm     # build + run WASM tests (Emscripten)
make test-parity   # 1-to-1 differential parity tests vs tinygrad reference
make bench         # build + run benchmark

# Frontend tests
python -m pytest py/tests/ -v        # 164 Python tests (tensor, nn, hf, instance, perf)
node js/test/test_tensor.js          # 109 JS tests
node js/test/test_instance.js        # 192 JS tests (MLP Instance)
node js/test/test_hf.js              # 36 JS tests (HF model loading)
```

## Status

### C Core
- [x] UOp IR (ops enum, dtype, arena, hashmap, CSE, toposort)
- [x] Pattern matcher + symbolic simplification (~25 rules)
- [x] Linearizer + C renderer + CPU runtime (end-to-end)
- [x] Shape inference + tensor-to-kernel scheduler (elementwise, reshape, expand, broadcast)
- [x] WASM binary renderer (scalar f32/f64 + f32x4/f64x2 SIMD) + binary builder
- [x] Reduce ops (REDUCE_AXIS → accumulation kernel: sum, max, product)
- [x] Movement ops (PERMUTE, SHRINK, FLIP, PAD index transforms)
- [x] Reverse-mode autograd core (`poly_grad`) for ALU/movement/reduce-sum paths
- [x] Rangeify + indexing pipeline (consumer map, realize detection, range propagation, multi-kernel BUFFERIZE)
- [x] Multi-kernel execution (`poly_schedule_v2` → `poly_realize` with intermediate buffer allocation)
- [x] Codegen pipeline (`full_rewrite_to_sink`): pm_reduce, pm_decomp, pm_transcendental, pm_add_control_flow
- [x] Frontend composed ops (~35): elementwise math, comparisons, matmul, softmax, layernorm, cross_entropy
- [x] Dtype-correct special math: lgamma, digamma, erf/erfc/erfinv, ndtri, log1p, expm1 propagate input dtype through all internal constants (f64 inputs get f64 kernels)
- [~] **33/33 value parity, 31/33 full IR parity** with tinygrad ClangRenderer (CPU, no vectorization)
- [x] ASSIGN + WAR ordering (in-place buffer writes, WAR/WAW dependency edges, `Tensor.assign()`)
- [x] Float16 (`__fp16`) and BFloat16 (`__bf16`) end-to-end on all backends (cast, arithmetic, mixed precision)
- [x] Disk cache for compiled kernels (`~/.cache/polygrad/<hash>.so`, 300x+ speedup on cache hit)
- [x] Unified linearizer (`poly_full_rewrite_to_sink_ex`) -- one pipeline for all backends, config via `PolyRewriteOpts`
- [x] CUDA backend with native f16/bf16 (`half`, `nv_bfloat16`, h* intrinsics) -- 607/607
- [x] HIP/AMD backend with MFMA tensor core codegen, comgr compilation -- 609/609 on MI250X
- [x] Interpreter with vector value model (lane-array, pre-allocated arena) -- 607/607
- [x] `POLY_DEVICE=cpu|cuda|hip|x64|interp` backend selector

### Model Zoo
- [x] PolyInstance runtime (forward, train_step, optimizer, weight import/export)
- [x] MLP builder (`poly_mlp_instance`) with configurable layers, activations, loss
- [x] HuggingFace model loading (`poly_hf_load`): config.json + safetensors -> PolyInstance
- [x] GPT-2 builder: full transformer (attention, FFN, layernorm, causal mask, weight tying)
- [x] Safetensors decoder with multi-dtype support (F32, F16, BF16, F64, integers)
- [x] GPT-2 training (forward + backward + optimizer via PolyInstance)
- [x] Autoregressive text generation (Python)

### Language Frontends
- [x] Float64 support (transcendentals, WASM renderer, all frontends)
- [x] Python Tensor API (ctypes FFI, lazy eval, autograd, f64, zero-copy numpy I/O, tinygrad-compatible)
- [x] Python nn module (Linear, LayerNorm, RMSNorm, Embedding, Dropout, SGD, Adam, AdamW)
- [x] Python HF loader (`load_hf`, `download_hf`, `generate`)
- [x] Unified JS package (`js/`): CommonJS Node entry, Node-API native path, WASM fallback, browser dist bundles, f64
- [ ] JS model/Instance parity with Python (builders, model runtime, HF loading, weight I/O) in the unified package

### Planned
- [ ] WASM build fix (sched_copy symbol portability)
- [ ] WebGPU backend (WGSL renderer exists, needs execution backend)
- [ ] More model families (LLaMA, BERT)
- [ ] Flash attention (depends on WMMA + wave shuffles + LDS)

1038+ tests (607 C + 164 Python + 101 JS native + 95 JS WASM/browser + 71 WASM C + 31 parity), ASan/UBSan clean. All 5 native backends at 607/607+.

## Reference

Port of [tinygrad](https://github.com/tinygrad/tinygrad) commit `c2be31e75b366638965337b96f2c66c2ba8c4068`.
