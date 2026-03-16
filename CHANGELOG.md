# Changelog

## Unreleased

### Added
- Cross-platform execution plan types (`exec_plan.h`): `PolyDeviceId`, `PolyCompileMode`, `PolyPreparedStep`, `PolyExecutableStep`, `PolyRunner`, `PolyBackendDesc`, `PolyAllocator`, `PolyBufferHandle`. Foundation for multi-backend PolyInstance.
- `poly_prepare_step()`: backend-neutral scheduling that produces `PolyPreparedStep` from a tensor SINK. Shared by all backends.
- `poly_lower_step()`: lowers a prepared step into a backend-specific `PolyExecutableStep`. Supports `POLY_DEVICE_CPU` (fork+clang+dlopen) and `POLY_DEVICE_INTERP` (linearize-then-interpret).
- `poly_executable_step_run()`: executes a lowered step with slot-indexed buffer data. Works for both CPU compiled and interpreter runners.
- Interpreter backend (`interp.c`): walks linearized UOps directly in C without external compiler. Handles all scalar types, BITCAST (int32/float32 bit reinterpretation), RANGE/END loops, DEFINE_REG accumulators, and the full codegen decomposition pipeline (EXP2 polynomial, LOG2, SIN). Serves as correctness oracle.
- CPU allocator (`POLY_CPU_ALLOCATOR`): trivial malloc/free/memcpy implementation for host memory.
- 18 new C tests: 5 prepared step, 3 CPU executable step, 4 interpreter, 6 CPU-vs-INTERP parity (chain, neg+sqrt, reduce_sum, where, exp2+log2, multi-kernel reduce chain).
- Backend-aware PolyInstance: `poly_instance_set_device()` for runtime device selection (CPU, INTERP). `poly_instance_call()` for generic entrypoint execution. `poly_instance_value_and_grad()` for forward+backward without optimizer. Prepared step cache survives device changes; executable step cache retains entries for all previously-used devices.
- `poly_instance_forward()` and `poly_instance_train_step()` rewritten as thin wrappers over `call()` and `value_and_grad()` respectively.
- 6 new instance tests: call_basic, set_device_interp, cpu_vs_interp_forward, cpu_vs_interp_train, set_device_roundtrip, set_device_unsupported.

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
