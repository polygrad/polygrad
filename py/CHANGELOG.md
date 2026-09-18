# Changelog

## 0.5.2 (release candidate)

The candidate requires C ABI101; PGIR19 and PGPM10 are unchanged. Intermediate
ABI numbers below describe development checkpoints, not package requirements.
Full release acceptance is pending.

- Cache shared-core CUDA graph estimates while preserving changing symbolic
  bindings, reducing JIT replay overhead without changing Tensor or Model APIs.
- Preserve Python >=3.9 for quantized GGUF decoding and require a Python3.9
  isolated-package execution lane during release acceptance.
- Add shared-core CLIP, ViT, DINOv2 and DINOv3 ViT builders and HF imports.
  Qwen3 owns its rotary tables; newly imported models need only token input x.

- Require ABI101 for shared Model AUX helpers and removal of unused C layer
  constructors. Fix GGUF imports on Python3.9 with deferred annotations.
  Calling an import-only model type now names its supported loader.

- Bind C ABI100's scoped UOp names and ship the separated core/domain headers.
  Python method names are unchanged; dtype-ID adapters delegate to typed C APIs.

- Consolidate C construction at ABI97: use poly_model_from_config for JSON or
  the existing typed poly_mlp_into/poly_gpt2_into/poly_qwen3_into entries. Explicit
  NULL requests an owned context. Remove per-type JSON/no-context aliases and
  their empty headers. Add the registry-discovered DistilGPT2 six-block preset,
  reusing GPT-2 topology and checkpoint mapping with no frontend factory code.

- Share Tensor cross-entropy and Model losses in C; consolidate activation and
  initialization helpers. Replace the separate import registry with model-type
  capabilities exposed by models.list(). Reject mismatched checkpoint shapes;
  only GPT-2 position tables opt into leading-axis cropping. ABI96 replaces
  poly_model_family_name with poly_model_type_name and adds capability queries.

Concurrent calls on a shared Python runtime remain unsupported; use independent
runtimes and thread-local cleanup.

- Sequential/Graph accept typed and bounded-leading-dimension inputs plus shared
  embedding, normalization, RoPE, attention, cast and permute components (ABI95).
  Calls and training preserve invocation extents through save/load.

- Shared C family dispatch (ABI94), GPT2 exposure and tagged Model configurations.
  Checkpoint-required models reject execution/export before complete weight
  initialization. Canonical bundle re-save and named input/objective diagnostics.

- `Model.place('CPU:1')` preserves the exact native CPU identity like module maps.
  Uses the name-taking core placement API (ABI93); unsupported ordinals reject.

- Fix interpreter custom-kernel STORE conversion and batched Model module-map
  inputs; preserve exact cut shapes and atomic failed-map behavior.

- Reclaim unowned storage at the next Tensor readback, including a small final
  view of a large allocation and disposed Model state. Preserve scan-free
  steady JIT/readback through shared-core ownership tracking.
- Avoid redundant collection during already-realized Tensor readback. Add a
  JIT/Adam performance guard with `.item()` inside timing, plus CPU/INTERP/CUDA
  replay/readback regressions; logical defaults and ownership are unchanged.
- Avoid full ancestry walks for cached shapes during Tensor training. Preserve
  logical-policy behavior; add an independently budgeted training performance gate.
- Remove the unreleased per-call locking/handle wrapper; cache immutable device
  mappings. Shared-runtime concurrency remains unsupported: use independent
  runtimes per thread or serialize all shared-runtime use and cleanup.
- Enable X86 in x86-64 source-package builds and test it after isolated installation.
- Fix clip(NaN), RNG policy transitions and invalid padded integer division/remainder.
- Correct the device-map example and compiler requirements.

## 0.5.1 (2026-09-17)

C ABI92 and artifact formats are unchanged. Wasm/INTERP mixed-dtype raw stores
remain a known limitation; portable custom kernels must cast explicitly.

- Fix CPU custom-kernel vector stores that could reinterpret integer bits as
  floats. Explicit UOp casts and scalar C numeric assignment remain supported.
- Correct custom-kernel and pretrained examples; use BEAM rather than POLY_BEAM.

## 0.5.0 (2026-09-16)

Python package version 0.5.0 requires C ABI92 and uses PGIR19/PGPM10 artifacts.
Registered parity debts remain open. Intermediate ABI/format versions below
describe individual development changes, not the final package requirements.

- Route Tensor attention and dropout through shared C NN owners (ABI92),
  retaining training/GQA/mask options and context RNG semantics. Fix the C
  half-precision SDPA accumulation graph used by models. Llama's omitted HF
  epsilon now defaults to1e-6 and rejected bias settings report INVALID.

- CUDA Model/HF construction and typed checkpoint I/O no longer retain implicit CPU
  shadows. Real Llama3.2-1B CUDA peak falls by about4.6GiB in the recorded local run;
  values and device residency are unchanged. Whole-shard loading still uses host memory.

- Borrow immutable HF shard bytes during synchronous import instead of making
  a second whole-file ctypes copy; keep snapshots for mutable inputs.

- Llama HF loading accepts head-only tied checkpoints, including TinyStories
  15M, through the corrected shared C importer. ABI91 unchanged.

- Add models.Llama(config), including optional runtime ownership and existing
  HF/bundle imports. Preserve C checkpoint error details. Requires ABI91;
  fixed-window float32 inference, not cached generation.

- Reject truncated or allocation-failed IR metadata imports while preserving
  existing Runtime owners. Verify compiled/shared imports and result lifetime.

- Fix chained view assignment through pending writes without buffer identity.
  Preserve Tinygrad's read-order semantics across logical policies and frontends.
  Reject malformed PARAM slots during metadata collection without dropping the
  pre-lowering or cached-CALL safety checks. ABI/artifact versions unchanged.

- Add optional rt.nn, rt.Model and rt.models bindings without changing default
  Tensor/NN usage. Preserve runtime ownership in helper/ZIP constructors, keep
  bound filename loaders lazy on DISK, and release failed Runtime initialization.
  Reject foreign mutation/realization before entering C.

- Preserve the source Runtime in rand_like, including derived Tensor wrappers;
  dropout capture no longer constructs its random mask in the default context.

- Capture evaluation/training Model forwards with shared state, restore authoring
  roots/modes on failure, and preserve private RNG/auxiliary state in bundles.
  Model capture requires C ABI90; optimizer configuration remains explicit on load.

- Model.fit supports bounded batch sizes, explicit remainder='keep', and device
  Tensor datasets without frontend host readback. Validate tails before training
  and release temporary views on success or failure without disposing callers.

- Reject unsupported empty Model input bindings before modifying earlier inputs.

- Support bounded leading Model dimensions and concrete owned result shapes
  across calls, training steps and portable imports. Route mean through the C
  symbolic reduction; preserve symbolic slots during import. Require ABI89/PGIR19.

- Preserve multidimensional host input shapes for C validation; equal byte
  counts no longer admit an incompatible shape. Model.fit accepts batch_size
  for ordered host datasets and explicit remainder='drop'; reject incomplete
  batches by default and reject noninteger epochs. Require C ABI88.

- Model call/forward with Tensor inputs returns owned device Tensor snapshots;
  array-only calls retain NumPy outputs. Require exact fixed shape/dtype and the
  Model Runtime/device. Training accepts Tensor bindings too. Require C ABI87.

- MLP/TabM/NAM and HF/GGUF loaders accept runtime= and otherwise use the
  default Runtime. Dispose factory-local Tensor handles after capture; reject
  malformed zero-head transformer configurations without terminating Python.
  Require C ABI 86. Include new factory/loader headers in source packages.

- Model.load/from_bundle/from_ir accept runtime= and otherwise borrow the default
  context. Repeated imports have independent state while retaining tied aliases;
  explicit Runtime disposal invalidates its loaded Models. Require C ABI 85.

- Add metadata-only Model.summary and reject declared async author/loss callbacks
  before capture. Reject save after disposal. Document named-output Python-to-JS
  linear regression and correct POLY_LIB/POLY_DEV package instructions.

- Model(net, ...) and Model(tagged_config) use existing capture/family builders;
  from_callable replaces trace as the explicit callable factory. Callable object
  attributes supply named state unless params overrides them. Remove state=;
  params tensors with is_param=False are AUX, not initially frozen PARAMs.
  Add dispose, bundle save/load, string set_optimizer and mapping train_step.

- Use POLY_DEV over DEV, and POLY_LIB for explicit native loading; old names
  removed. Invalid paths/targets fail closed. Explicit Runtime device defaults
  reach its Tensor factories; Context DEV scopes restore the initial override.
  POLY_DEBUG initializes DEBUG before the unprefixed environment value.

- Add Tensor.const, retain Variable/BoundVariable UOps in Tensor construction,
  and cast explicit UOp dtypes in the graph. Empty DISK tensors retain the path
  without eager file I/O. Reject failed UOp casts before default construction.
  Require C ABI84; package0.5.0 unchanged.

- Restore `+=`, `-=`, `*=` and `/=` assignment semantics; reject capture-time
  Tensor host reads. Shared C JIT now protects replay inputs aliasing outputs.

- Require C ABI83 after the core/FFI ownership cleanup. `can_run` raises for
  operation-specific probe budgets rather than reporting unsupported execution.

- Fix bitcast-view assignment through shared C call preparation, including
  contiguous slices, reshaped/detached views and repeated writes from fresh views.

- Add `clear_schedule_cache()` for the default context and Runtime. Clearing
  releases cache ownership; explicit `collect()` reclaims only unowned graphs.
  Live Model/JIT executions remain owned. Require an idle, serialized runtime.

- Add Conv1d/ConvTranspose1d/2d, InstanceNorm, LSTMCell and LARS/LAMB/Muon.
  Linear, RMSNorm, InstanceNorm and LSTMCell invoke shared C layer programs;
  optimizers retain shared C update math. Fix multidimensional LayerNorm,
  Dropout endpoint/training behavior and explicit 1D convolution tuples.
- Route LayerNorm/GroupNorm/BatchNorm through C, preserving symbolic batch
  shapes and bound reduction denominators. Accept BatchNorm's pinned `sz`
  keyword and dtype objects in Context.
  Allocate empty Tensor storage before readback; initial bytes are unspecified.
  Reject unsupported Model optimizer kinds without replacing existing state.

- Fix X86 non-finite isclose results through the shared instruction encoder.
  Keep exact pinned X86 numerical baselines separate from C-renderer rounding;
  correct device-selection fixtures without relaxing assertions.

- Fix the shared X86 renderer's gated vector-address selection, restoring
  GPT-2 forward/training on X86 without changing the Python API or C ABI.

- Correct the X86 verification target to select Python construction and C
  import/placement together; preserve exact checkpoint-continuation assertions.

- Remove the unused ctypes PolyBuffer layout; buffer handles remain opaque and use C accessors. Verify Model access/lifetime, JIT and cross-language checkpoint contracts against the current core.

- Core: complete half LOG2/SIN and integer late rewrites, preserve pinned transcendental construction, and fail closed on final kernel extraction allocation failure. No Python API, ABI or package-version change.

- Core: preserve complete ordered loop dependencies through materialization/extraction and correct int64→float64 decomposition precision. Failed buffer-limit substitution and extraction-map allocation no longer publish incomplete replacements. No Python API or package-version change.

- Core: preserve bound-prefix sums and flipped coordinates at runtime sizes below allocation maxima. Correct reshape range restoration and movement coordinate arity. No Python API or package-version change.

- Core: correct ordinary FUNCTION gradient and symbolic-shape failure handling, movement admission, symbolic flatten and UNSHARD dtype forwarding. Document that FUNCTION precompile flags currently support forward execution only. ABI/package versions unchanged.

- Preserve explicit gradients of detached targets themselves and return COPY gradients to the source device. Core also preserves symbolic reshape dimensions and tuple gradient slots. ABI and package versions unchanged.

- Preserve signed64 runtime bindings through scheduling and JIT replay; reject larger integers before ctypes conversion. Require core ABI80. Typed bound metadata remains broader than executable binding values; package version unchanged.

- Core: add cached postcompile local-workgroup selection without modifying user buffers; preserve PROGRAM replay/export metadata. No Python API or package-version change.

- Core: correct launch admission, CALL access metadata and waited CPU/CUDA execution accounting; reject overflowing lane-binding storage. No Python API or package-version change.

- Preserve long scalar bounds in UOp diagnostics without a fixed-buffer overflow in the C formatter.

- UOp.variable accepts fractional, infinite and wide-integer bounds without ctypes truncation. Require core ABI79 and graph formats PGIR18/PGPM10, including complete PARAM metadata. Package version and runtime integer-binding limits are unchanged.

- Core: correct derived uint64 bounds and exact-interval allocation-failure cleanup. No Python API or package-version change.

- Fix shared INTERP scalar reads, CUDA scalar argument packing, and JIT input-preparation failure handling. Symbolic/JIT checks pass; package version unchanged.

- Fix shared runtime allocation of eliminated arguments, empty-storage COPY and arena alignment/overflow. Copy/realization/JIT coverage passes; package version unchanged.

- Correct shared compiler launch-limit and reduction-matcher admission; discard incomplete range-shrink decisions on inspection failure. Package version unchanged.

- Follow Tinygrad v0.14.0 core contracts and fix seeded-random concatenation rejected by late gated-load construction. Package version unchanged.

- Require C ABI78 for exact LOCAL stage integer metadata. Portable/bound graph formats are PGIR17/PGPM9; older versions are rejected. Package version remains0.4.2.

- Correct shared schedule slot bounds, graph count admission, buffer-limit failure propagation, and precompiled STORE-value dependencies. ABI/package versions unchanged.

- Stop retaining CPU shadows during buffer readback; returned arrays remain independent copies. Reject invalid transfer extents and missing allocator callbacks. Range-cache allocation failure now follows the existing fatal-OOM policy instead of overflowing or reporting an empty range set. ABI/package versions unchanged.

- Reject nonzero WINO requests with NotImplementedError; ordinary convolution is unchanged. Required HF acceptance fails on missing dependencies, while optional runs retain eight skip records. Installed-sdist checks use a fresh environment and verify package/library origins.

- Fix shared reduction lowering: preserve complete loop tuples and outer dependencies, discover mergeable ENDs from the graph, preserve cloned range metadata, and fail cleanly on local scratch allocation failure. Grouped reductions no longer bypass LOCAL staging when non-group loop counts exceed Tensor rank. LOCAL stage integer identity remains registered parity debt; ABI and package versions are unchanged.

- Shared compiler fixes preserve exact ordering and failure propagation, literal LOCAL admission and SHRINK reindex topology. Native X86 hexadecimal compilation matches byte/whitespace and strict error semantics. ABI and package version unchanged.

- Add C-backed `Context(IGNORE_BEAM_CACHE=...)` with scoped restoration (ABI77; package version unchanged).

- Context(BEAM=...) now controls the C compiler and restores its policy (ABI76; package version unchanged). Shared scheduler fixes honor explicit options and preserve symbolic bounds during divisible splits.

- Require ABI75 and consume the C buffer-set status. Core attachment failures preserve existing storage; host/file constructors propagate failure without leaking mappings. Package version unchanged.

- UOp.device and Tensor.device now expose exact physical metadata, including tuples and None for deviceless expressions (ABI74). Fix deviceless assign/index/gather/scatter admission and remove duplicate frontend result-device inference. Package versions unchanged.

- Shared C lowering avoids undefined float64/FP8-FNUZ arithmetic and rejects range scratch allocation failure without changing PCONTIG or retaining partial compiler state.

- Preserve full DISK paths in C-backed transfer, chained copies, slice readback and `Tensor(Path, device=...)`. Match configured dtype composite/raw-factory construction and handle shape allocation failures safely (ABI73; package versions unchanged).

- C-backed DEFAULT_FLOAT/DEFAULT_INT contexts now control storage, promotion and compiler keys. Fix malformed factory argument rejection, range overflow admission and integer/bool randn output (ABI72; package version unchanged).

- Support Context(NOOPT=...) through C compiler policy and cache keys. Fix scalar reduction axes, virtual weak realization, einsum scalar/ellipsis/uppercase construction and dot/einsum error admission. Python copies own separate C Tensor handles rather than duplicating an owning pointer.

- Shared C spatial Tensor operations and anonymous storage; Runtime-bound invalids preserves context ownership. CHECK_OOB=0 is accepted, nonzero verifier configuration fails explicitly.

- Add indexed assignment through the shared C owner, writable Tensor.grad references, and ordinary binary_crossentropy reduction selection. Broadcast advanced indices together; preserve no-op identity, detached write aliases, mixed bool/int list-index admission, and negative integer slice bounds against symbolic sizes.

- Add C-backed log10/inverse trig and hyperbolic functions, celu/selu, logsigmoid/sinh/cosh/erf/softsign, isfinite/isclose/copysign/lerp, weighted binary_crossentropy_logits and nll_loss with none/sum/mean reductions. Preserve pinned scalar/Tensor weight provenance and autograd; NLL currently requires concrete shapes.

- Add C-backed all/any/cumsum/cumprod/cummax/cummin with paired extrema indices. Correct scan dtypes, product gradients and nested-scan gradient execution; symbolic scans remain unsupported.

- Reject out-of-bounds shrink through shared C validation and match pinned boolean minimum graphs; preserve unary boolean min's logical-not behavior.

- Use the shared C min reduction instead of integer negation. Preserve unsigned zero/signed minima and NaN for scalar negative fractional powers; retain exact weak integer masks in min/minimum graphs.

- Add parsed DEV/Target configuration and fail closed on nonzero IMAGE requests; roll back failed Context entry, including the C logical policy.

- Add max_shape/max_numel, optional shrink axes, shrink_to/pad_to, and identity-preserving full slices and empty-axis flip. Symbolic shrink keeps its C shape sources; pad_to rejects cropping.

- Return DType from Tensor.dtype; align UOp.const/variable signatures with the pin, retaining explicit owning contexts as ctx keywords. Add element_size/is_floating_point and reject conflicting UOp owners.

- Match pinned Tensor construction: None is scalar zero; integer lists truncate to storage width even when NumPy rejects overflowing Python integers; non-scalar weak storage rejects before import.

- Fix copy_from on fresh host inputs, logical='never' and pending assignments; write current physical storage after required materialization. Add the pinned writable-memoryview mv_address helper.

- Interpret Tensor bytes as owned raw storage, defaulting to uint8 and preserving BF16/FP8 encodings. Add the pinned temp helper and expose the core's lossless-cast predicate with the current scalar dtype ABI.

- Core: preserve weak-lowering resource/address tags and discard tags on fresh committed literals, matching the pinned graph rules.

- Core: correct requested UNSHARD boundaries, virtual/ALU output admission, and training-marker/shared-parent call construction; preserve fresh Model initializer import through explicit storage.

- Core: preserve invalid markers during promotion/lowering and correct UNSHARD/MSELECT storage-identity queries.

- Include the core's transitive headers in source distributions; installed packages now compile without the repository include tree. Add a manifest dependency regression.

- Add `models.Sequential(config, runtime=...)` and `models.Graph(config, runtime=...)`; both construct ordinary Models in C from JSON configurations, with no frontend authoring callback.

- Rename Instance to Model; retain fit/train_step/set_optimizer (introduced in ABI68).
- State reads return copies, not mutable views into C-owned storage. Use read_buffer/write_buffer for exact typed state access.
- Add one-time callable capture with Model.trace and binding/entrypoint metadata.
- Fix tied-parameter freeze/unfreeze, objective cache selection, typed loss diagnostics, uniform placement and persistent AUX export.

## 0.4.2 (2026-07-06)

### Added
- Like/normal factories, product/log reductions, normalization, exact GELU, shape helpers, padding modes and sparse-loss options across Python/JS with shared C owners.
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
- C ABI71 includes the Tensor extensions and compiler-policy accessors; package versions remain unchanged.
- Runtime placement keeps `uop_logical` export roots independent from realized `uop_physical` roots after device execution.
- Python source synchronization now includes the engine JIT files needed by sdist and wheel builds.

### Fixed
- Select rounded division after promotion; preserve uint64 scalar/full literals, STACK alias gradients, and strong-typed storage when cloning weak computations. Reject invalid factory shapes and avoid narrowing movement sizes/offsets to int32.
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
