# Changelog

## 0.4.2 (2026-07-06)

### Added
- Explicit Python runtime creation via `polygrad.create(...)` for parity with JavaScript multi-runtime workflows, while keeping the default `Tensor` API unchanged.

### Fixed
- Python packaging now tracks the shared 0.4.2 core line and the full test suite against the rebuilt shared library.

## 0.4.1 (2026-07-05)

### Fixed
- Include the pure Python `polygrad` package files in wheels built from the
  PyPI source distribution. The 0.4.0 sdist uploaded correctly, but wheels built
  from it installed only `polygrad._native` and metadata.

## 0.4.0 (2026-07-05)

### Added
- Python custom-kernel UOp helpers with compile/JIT replay, grouped stores, reductions, comparisons, unary ops, and repeated input mutation tests.
- Structured linalg APIs for QR modes, triangular solve, Cholesky, Cholesky solve, solve, and least squares, implemented as portable tensor-composed fallbacks and checked against NumPy and Torch references.
- Expanded backend coverage for CUDA, x86, WASM, WebGPU, interpreter, and native CPU common tests.
- Batched typed readback and stable tensor input update helpers for compiled Python loops.

### Changed
- Runtime placement keeps `uop_logical` export roots independent from realized `uop_physical` roots after device execution.
- Python source synchronization now includes the engine JIT files needed by sdist and wheel builds.

### Fixed
- CUDA compiled replay now rebinds custom-kernel producer outputs for downstream compiled consumers.
- WebGPU, WASM, CUDA, and x86 fixes for custom-kernel reductions, typed constants, floor division/modulo, and vector/lane rendering.
- Capability checks now report sort, argsort, and topk support consistently with execution.

## 0.3.0 (2026-05-25)

### Fixed
- CUDA group_for_reduce: removed premature IF/ENDIF creation from `poly_group_for_reduce` that caused the IF guard to float above the accumulation RANGE during linearization. Only thread 0 executed the inner loop, checking vocab indices at stride 256 and missing all non-aligned indices. Qwen3 0.6B embedding lookups for non-zero tokens produced all-zero output. Fix defers the single-writer guard to `poly_add_gpudims` as a gated 3-source INDEX on global stores missing local dims, matching tinygrad's gpudims.py approach.
- CUDA gated STORE rendering: taught `render_cuda.c` to emit `if (gate) { *ptr = val; }` when a STORE's INDEX has a 3rd boolean source. Previously only gated LOADs were supported.
- CUDA integer MULACC: added `(%s*%s+%s)` fallback for non-float MULACC in `render_cuda.c`. Previously rendered as `__fmaf_rn(a,b,c)` which is wrong for integer index math.
- `poly_eq` type mismatch: changed from `CMPNE(ne, INT32(1))` to `CMPNE(ne, BOOL(true))`. The bool/int32 operand mismatch produced incorrect comparison results on CUDA.
- Instance const_registry CUDA migration: anonymous constant buffers (arange in gather, causal mask) are now uploaded to device memory in `slot_cache_build`. Previously passed as host pointers to CUDA kernels.

### Added
- BEAM search optimizer (`POLY_BEAM=N` env var). Explores UPCAST/UNROLL action space by compiling and timing candidates, keeps top-N per iteration (up to 5 iterations). Disk cache in `~/.cache/polygrad/beam/`. Integrated into `poly_full_rewrite_to_sink_ex` and `poly_linearize_env`. 5 new C tests (beam suite).
- Cross-platform execution plan types (`exec_plan.h`): `PolyDeviceId`, `PolyCompileMode`, `PolyPreparedStep`, `PolyExecutableStep`, `PolyRunner`, `PolyBackendDesc`, `PolyAllocator`, `PolyBufferHandle`. Foundation for multi-backend PolyInstance.
- `poly_prepare_step()`: backend-neutral scheduling that produces `PolyPreparedStep` from a tensor SINK. Shared by all backends.
- `poly_lower_step()`: lowers a prepared step into a backend-specific `PolyExecutableStep`. Supports `POLY_DEVICE_CPU` (fork+clang+dlopen) and `POLY_DEVICE_INTERP` (linearize-then-interpret).
- `poly_executable_step_run()`: executes a lowered step with slot-indexed buffer data. Works for both CPU compiled and interpreter runners.
- Interpreter backend (`interp.c`): walks linearized UOps directly in C without external compiler. Handles scalar types, BITCAST, RANGE/END loops, BUFFER(REG) accumulators, and codegen decomposition.
- CPU allocator (`POLY_CPU_ALLOCATOR`): trivial malloc/free/memcpy implementation for host memory.
- 18 new C tests: 5 prepared step, 3 CPU executable step, 4 interpreter, 6 CPU-vs-INTERP parity (chain, neg+sqrt, reduce_sum, where, exp2+log2, multi-kernel reduce chain).
- Backend-aware PolyInstance: `poly_instance_set_device()` for runtime device selection (CPU, INTERP). `poly_instance_call()` for generic entrypoint execution. `poly_instance_value_and_grad()` for forward+backward without optimizer. Prepared step cache survives device changes; executable step cache retains entries for all previously-used devices.
- `poly_instance_forward()` and `poly_instance_train_step()` rewritten as thin wrappers over `call()` and `value_and_grad()` respectively.
- 6 new instance tests: call_basic, set_device_interp, cpu_vs_interp_forward, cpu_vs_interp_train, set_device_roundtrip, set_device_unsupported.
- WASM executable step: `poly_lower_step(..., POLY_DEVICE_WASM_JIT)` renders WASM kernel bytes and compiles them via EM_JS bridge. Two-phase design: compile once during lowering (cached by kernel_id), execute many times. PolyInstance works on WASM with full forward and training support. Instance functions exported from Emscripten build. MLP, TabM, NAM model builders available in WASM.
- Graph-compiled optimizer: SGD/Adam/AdamW emit UOp graphs with ASSIGN, executed via `poly_realize()`. Moment buffers are proper `PolyBufferHandle`s. Training uses one execution path (no host-side optimizer loop).
- DEFINE_VAR support in exec_plan: `PolyExecItem.var_uops[]` stores per-kernel DEFINE_VAR UOps. `poly_compiled_plan_run()` resolves var values from bindings at runtime.

### Changed
- `poly_compile_step()` is now a thin wrapper over `poly_schedule_for()` + `poly_compile_schedule()`. PolyStep holds references to `PolySchedule*` and `PolyCompiledPlan*` instead of its own compiled kernels.
- `poly_step_run()` delegates to `poly_compiled_plan_run()`. No more parallel execution infrastructure.
- `poly_realize_ex()` passes var_bindings through the exec_plan path. No legacy fallback.

### Removed
- `realize_impl()`: legacy monolithic realize path with inline scheduling, compilation, and execution. All execution now routes through exec_plan.
- `graph_has_vars()`: fixed-size stack traversal (4096/8192 arrays with silent truncation) that gated the realize_impl fallback. No longer needed.
- Old schedule cache (`SchedCacheEntry`, `sched_cache_get/put`, `realize_from_sched_cache`, `compile_and_run`): 300+ lines of duplicate caching infrastructure replaced by per-context exec_plan caches.
- `PolyStepKernel`, `ParamMapping`: internal types for PolyStep's own compilation. Replaced by `PolyRunner` in exec_plan.
- Deleted from PolyInstance: `device`, `allocator`, `prep_cache`, `exec_cache`, `apply_optimizer_update()`, `build_slot_data()`, `resolve_vag_slots()`.

### Fixed
- Einsum trace: repeated indices in a single input (e.g. `'ii->'`) now correctly extract the diagonal instead of producing wrong results. Ported tinygrad's diagonal extraction algorithm (permute + flatten + pad + reshape + shrink).
- Float16 C renderer: changed dtype name from `"half"` to `"__fp16"` (matching tinygrad ClangRenderer `type_map`). `half()` and `bfloat16()` now work end-to-end on CPU. 14 new C tests in `test_f16.c`.

### Added
- Step slicing: `t[::2]`, `t[1:7:3]`, `t[::-1]`, `t[::-2]` now work in both Python and JS. Decomposed into movement ops (shrink + flip + pad + reshape + shrink + reshape), matching tinygrad's `_getitem` stride logic. Supports positive and negative steps, multi-dimensional slicing, and backward (autograd). 10 new Python tests, 8 new JS tests (native + WASM + browser).
- JS dtype casting: `cast()`, `half()`, `double()` methods on JS Tensor. Wired `poly_cast_by_id` in both Node-API and WASM backends.
- JS `triu()` and `tril()` methods on Tensor with diagonal offset support. Fixed WASM int64 shape marshalling for triu/tril.
- JS parity: added 16 new shared tests covering pow, reciprocal, exp2, log2, trunc, swish, view, matmul aliases, layernorm, binaryCrossEntropy, cat (1D/2D), stack, repeat (1D/2D). All pass on native + WASM + browser.

### Fixed
- `repeat()` interleave order in both Python and JS: was `(s, 1)` expand `(s, r)` (repeats each element), now `(1, s)` expand `(r, s)` (repeats entire tensor). Matches PyTorch `Tensor.repeat()` semantics.
- Dtype casting: `cast()`, `half()`, `float()`, `double()`, `int()`, `long()`, `short()`, `bool()`, `bfloat16()` methods on Tensor, matching tinygrad's DTypeMixin. C core `poly_cast()` and `poly_cast_by_id()`.
- Disk cache for compiled kernels: `~/.cache/polygrad/<hash>.so` persists compiled kernels across process restarts. Source-keyed (FNV-1a 64-bit hash of C source + optimization flags). Cold compile ~170ms, warm cache hit ~0.5ms (300x+ speedup). Control via `POLY_CACHE=0` (disable) and `POLY_OPT=0/1/2` (optimization level, default `-O2`).
- Zero-copy numpy input: `Tensor(numpy_array)` no longer copies data when the array is already contiguous with the correct dtype. ~10x faster tensor creation for large arrays.
- Zero-copy numpy output: `tensor.numpy()` returns a view instead of a copy.
- Performance tests (`py/tests/test_perf.py`): disk cache verification, overhead-vs-numpy benchmarks, zero-copy correctness tests.
- Kernel fusion advantage: fused 5-op chains on 5M+ elements run 7x faster than numpy (1 memory pass vs 5).

## 0.2.2 (2026-03-14)

### Fixed
- Cross-entropy loss: single-sample returns correct value, batch mode works
- Matmul shape validation: raises error for mismatched inner dimensions
- 0/0 returns NaN

## 0.2.1 (2026-03-14)

### Added
- Initial PyPI release
- Tensor API: creation, arithmetic, reductions, math, comparisons, reshape/view, slicing, matmul, broadcasting, autograd
- Activation functions: relu, sigmoid, tanh, gelu, silu, softmax, elu, leaky_relu, mish, hardswish, hardsigmoid, hardtanh, softplus, relu6, quick_gelu
- Factory methods: zeros, ones, full, eye, arange, linspace, empty, rand, randn, randint
- Advanced ops: einsum, rearrange, cat, stack, where, pad, shrink, flip, chunk, split, repeat, triu, tril, var, std
- nn modules: Linear, Conv2d, Embedding, LayerNorm, GroupNorm, RMSNorm, BatchNorm, Dropout
- Optimizers: SGD, Adam, AdamW
- State dict: get_state_dict, load_state_dict, get_parameters
- GPT-2: model builder, configs, HuggingFace loading, autoregressive generation
- Compiled training steps, Instance API (MLP builder)
- Variable/BoundVariable, Device abstraction
