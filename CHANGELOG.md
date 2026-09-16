# Changelog

## 0.5.0 (2026-09-16)

Package version 0.5.0 targets Tinygrad v0.14.0 with C ABI92, PGIR19 and PGPM10.
Registered parity debts remain open; this is not a claim of exhaustive parity.
The development entries below include intermediate ABI and format versions;
the versions above describe the final release.

- Initialize Model storage on its actual host-addressable backend without a
  retained staging copy. Fix Wasm symbolic CPU-alias construction and unsigned
  heap addressing above2GB. ABI92 and artifact formats are unchanged.
- Make release tooling explicit: Clang Format14, source/version-bound analyzer,
  UTF-8 subprocesses, package build dependencies and verified native-failure
  fallback. Keep the stricter shared Tinygrad fixed-width Z3 failure separate
  from release's required general/division proofs.

- Share SDPA and dropout construction in C across models, Python and JS. Match
  Tinygrad's half-precision multiplication with widened reduction; use explicit
  GQA and the existing context RNG. C SDPA signatures changed: ABI92, unchanged
  PGIR19/PGPM10 and package0.5.0. No new attention executor.
- Llama HF configuration defaults omitted rms_norm_eps to1e-6; rejected bias
  settings now return INVALID rather than a success status with no Model.

- Keep ordinary GPU Model state device-resident; allocate host shadows only for
  explicit raw C access. Preserve aliases, initialization and rollback, and avoid
  cached Model host pointers. WebGPU checkpoint replacement uses importWeightsAsync;
  bound-program async imports now await readback. No ABI/artifact-format change.
  Raw C host views must be reacquired after Model operations, including writes;
  their contents are no longer implicitly refreshed. Typed reads return copies.

- Avoid copying immutable Python HF shards at the ctypes boundary; preserve
  mutable-input snapshots. Release JS readback temporaries after single/batched
  reads, including asynchronous and failure paths. No ABI/format change.

- Allocate portable Model state on its initial physical bindings instead of
  retaining duplicate logical-buffer storage. Preserve capture, placement and
  import semantics; no ABI or artifact-format change.

- Accept either retained embedding/head name in tied Llama HF checkpoints;
  reject conflicting duplicates and incomplete untied state. Add a strict,
  pinned pretrained CPU/CUDA logits/replay/bundle gate. ABI91 unchanged.

- Add the shared C Llama family and Python/JS factories: dense Llama 2/3.x,
  fixed-window logits, MHA/GQA, llama3 RoPE scaling and tied weights. HF imports
  validate complete state; Python propagates C import diagnostics. ABI91,
  unchanged PGIR19/PGPM10. KV-cache generation is not included.

- Fix C causal attention's mask dtype to match Tinygrad's boolean masking;
  preserve Model authored-state restrictions in registry tests. Release preflight
  rejects incompatible CPU compilers and source-audit Python versions before
  starting the matrix. No ABI or artifact-format change.

- Reject truncated or allocation-failed IR tuple/RANGE metadata imports without
  destroying the caller's Runtime. Expand private/shared import, alias isolation,
  compiled interchange and output-lifetime coverage across frontends.

- Give BEAM a tag-independent core UOp content key instead of PGIR export.
  Compare complete cache keys before replay; verify core-only linking without
  Model/codecs. No ABI or model artifact version change.

- Fix chained view assignment through pending writes without buffer identity.
  Preserve Tinygrad's read-order semantics across logical policies and frontends.
  Reject malformed PARAM slots during metadata collection without dropping the
  pre-lowering or cached-CALL safety checks. ABI/artifact versions unchanged.

- Preserve explicit Runtime ownership across Python Tensor helpers and NN/state
  constructors; add bound rt.nn, rt.Model and rt.models using existing implementations.
  Reject foreign assignment/realization before mutation. Fix no-kernel typed views
  of realized storage, including batched ZIP headers. ABI/artifact versions unchanged.

- Python rand_like preserves the source Runtime for derived Tensors, fixing
  dropout Model capture outside the default context. Import equivalence checks
  cover canonical round trips, independent state and compiled CPU signatures.

- Callable Models with a loss capture shared-state evaluation/training graphs.
  Preserve authoring roots/modes, isolate RNG, retain AUX effects during training,
  and reject effectful reads/async capture. C ABI90; PGIR19/PGPM10 unchanged.

- Model.fit accepts bounded batch sizes and host/Tensor/mixed datasets. Explicit
  remainder='keep' admits supported smaller tails; default error and drop remain.
  Slice views stay on-device and release after each step or failure. No ABI change.

- Reject unsupported empty Model input bindings before modifying earlier rows.
  Check and jointly release Wasm Model entrypoint/loss/output allocations on
  failure; do not enter C or write address zero. ABI/artifact versions unchanged.

- Preserve bounded leading Model dimensions through capture, portable export and
  invocation. Keep owned result shapes/values stable across calls; bind training
  reductions through core symbolic mean. Fix import remapping of ALU BUFFER -1
  sentinels, which could generate invalid kernel arguments after loading. C ABI89,
  PGIR19; dynamic bound-program export rejects and PGPM10 remains unchanged.

- Model host bindings may include concrete shapes; reject mismatched shapes
  before input writes while retaining flat-storage calls. C ABI88; wire formats
  unchanged. Add ordered fixed-size host minibatches to frontend fit, explicit
  remainder rejection/drop, integer epoch validation, and JS fitAsync for WebGPU.

- Model calls accept fixed-shape, same-context/device Tensor inputs and return
  independently owned device Tensor outputs. Array-only calls are unchanged.
  Realize pending input effects once before snapshotting; preserve results across
  calls and borrowed Model disposal. Share C admission with training and keep
  WebGPU invocation/release on its async queue. C ABI87; wire formats unchanged.

- C model families and HF/GGUF loaders can borrow caller contexts. Frontends
  use their selected/default Runtime; standalone C factories still own contexts.
  Release construction Tensor wrappers after publishing Model roots, restore
  caller defaults on failure, and reject zero attention heads before division.
  C ABI 86; package 0.5.0 and PGIR18/PGPM10 unchanged.

- Portable Model imports can borrow a Runtime context, with fresh storage
  identities per import and retained parser roots through publication. Fix
  partial entrypoint cleanup after allocation failure. C ABI 85; portable
  artifact versions unchanged. Standalone C and bound-program ownership remain.

- Complete Model usability: Node path save/load and async save stay in its host
  adapter; browsers accept bytes and reject filesystem paths. Add metadata-only
  summary in both frontends, reject declared async capture/loss callbacks before
  invocation, and reject Python save after disposal. Add named-output linear
  export/prediction examples to maintained cross-language acceptance.

- Model constructors accept callable objects, Tensor bindings, or explicitly
  tagged Sequential/Graph configurations through existing C builders. Replace
  trace with from_callable/fromCallable (and fromCallableAsync). Collect object
  state by default; explicit params overrides it. Remove state=; is_param=False
  now selects AUX in params, while set_trainable preserves a frozen PARAM role.
  Add bundle save/load conveniences, Python dispose and mapping train_step,
  and optimizer string names in both frontends. C ABI84 and formats unchanged.

- Rename device/library environment overrides to POLY_DEV and POLY_LIB;
  remove POLY_DEVICE/POLYGRAD_LIB. Use DEV as the shared fallback and preserve
  explicit runtime choices. Reject unsupported target syntax and invalid
  Python library paths without fallback. Initialize Node-Wasm BEAM/NOOPT once
  per module; preserve POLY_DEBUG=0 precedence. ABI84/package0.5.0 unchanged.

- Preserve complete DISK identities in lazy empty Tensor construction through
  the shared C factory. Add scalar/UOp `Tensor.const` in Python/JS and preserve
  Python bound-variable graphs during construction. Explicit UOp dtypes cast
  the graph; failed casts reject construction instead of using a default value.
  C ABI84; package0.5.0 and PGIR18/PGPM10 unchanged.

- Count successful cold-WebGPU HOST snapshot writes in C, without changing
  storage or counting failed publication. Guard test registration capacity and
  make buffer-ownership fixtures runnable on native and Wasm.

- Fix JIT output/input alias corruption using Tinygrad's write-only input
  snapshot protection in the shared C core. Restore Python in-place arithmetic
  assignment and reject capture-time Tensor host reads in Python/JS. ABI83 and
  package/artifact versions unchanged.

- Separate typed C creation and capability probing from FFI adapters; expose
  ABI lookup through the core header. Remove no-op CPU cache flushing and unused
  debug exports. Report operation-specific probe budgets as not evaluated.
  C ABI83; package0.5.0 and PGIR18/PGPM10 unchanged.

- Fix shared C call preparation for bitcast-view assignment, preserving original
  storage and typed slice arguments. WebGPU's unaligned-offset limit remains open.

- Add explicit idle-runtime schedule-cache clearing in C, Python and JS.
  Release cache-owned keys and LINEARs without automatic collection or eviction
  of other caches. Reject active execution/capture and suspended WebGPU work.
  Pointer-key lifetime parity remains debt032; package/format versions unchanged.

- Organize C layer/optimizer programs under `src/nn/` and named Model layer
  construction under `src/models/layers.*`. Add shared LSTMCell/InstanceNorm
  programs and C Model LSTM construction; frontends call shared layer graphs.
- Add shared LARS/LAMB/Muon optimizer graphs and Newton–Schulz composition.
  ABI81 extends optimizer configuration; package/artifact versions do not change.
- Share LayerNorm/GroupNorm/BatchNorm programs in C; preserve symbolic batch
  dimensions in GroupNorm and use actual reduced extents, not allocation
  maxima, in raw C mean. Match BatchNorm's `sz` keyword and dtype-object
  Context settings. Allocate empty storage before frontend readback without
  promising initial values. Reject unsupported Model optimizer kinds before
  replacing existing configuration.
- Correct RMSNorm accumulation and SDPA boolean masks/accumulation; preserve
  Tensor view-assignment publication in optimizers and strip interleaved
  assignment effects from call finalization's returned storage views.

- Match Tinygrad's X86 REX encoding for byte-demoted mixed-width register
  operands, fixing non-finite isclose wrong results under register pressure.
  Add exact instruction-byte and execution regressions; retain exact
  backend-specific Python rounding expectations. ABI and formats unchanged.

- Match Tinygrad's X86 conditional-address lowering for gated vector loads.
  Fix GPT-2 embedding forward/training rejection; preserve scalar CMOV for
  uint64 addresses and normalize compound boolean gates. ABI unchanged.

- Select both Python DEV and standalone C POLY_DEVICE in the X86 test lane;
  add a recipe regression preventing mixed CPU/X86 checkpoint comparisons.

- Honor the existing CPU→Wasm alias in Model placement and module device maps,
  matching Tensor placement. Unknown/default explicit targets remain rejected;
  physical device metadata, C ABI and graph formats are unchanged.

- Point HLB verification at the pinned Tinygrad 0.14.0 checkout without changing the workload. Cover WebGPU queued failure/disposal paths, remove a stale unused Python buffer layout, and reject obsolete migration inventory paths. Document schedule-cache key retention as open debt PG-PARITY-032; runtime behavior, ABI and package versions are unchanged.

- Complete half-precision LOG2/SIN decomposition and retain pinned polynomial/reciprocal construction. Support integer MULACC and exact 2^63 power-of-two rewrites; remove the legacy non-uint64 THREEFRY decomposition. Reject final kernel CALL-publication failures instead of treating them as no-match. ABI/package versions unchanged.

- Preserve complete ordered active ranges and full materialization END keys; retain split paths and dependencies during kernel extraction. Make extraction-map allocation and buffer-limit substitution fail explicitly. Use float64 intermediates for int64→float64 decomposition. Record the existing 16-axis STAGE limit separately from range count. ABI/package versions unchanged.

- Correct symbolic flip/reduction extents, singleton ranges, scalar staging and consumer deduplication in indexing. Preserve RANGE dependencies through reshape; fail closed on substitution errors. Match full movement coordinate mapping without weakening partial-index suffix checks. ABI and package versions unchanged.

- Fix FUNCTION gradient input classification, failed parameter-compaction publication and retry after failed symbolic shape resolution. Validate raw permutation/flip shapes, preserve symbolic flatten extents and forward UNSHARD dtypes on rewrite. Document unsupported FUNCTION backward flags/callbacks and diagnostic provenance. ABI/package versions unchanged.

- Fix autograd root seed preservation, tuple gradient accumulation, symbolic reshape dimensions and reverse COPY device selection to match Tinygrad v0.14. Propagate target-walk allocation failure instead of returning zero gradients. ABI and package versions unchanged.

- Widen runtime scalar bindings to signed64 across scheduling, JIT, timing and supported backends (C ABI80). Add JS UOp.bind with BigInt admission; reject out-of-domain frontend values. Preserve wide integer diagnostics on wasm32. Larger Python-integer bindings remain a documented gap; package versions and graph schemas are unchanged.

- Port postcompile local-workgroup selection using independent scratch storage and cached PROGRAM dimensions. Share timing buffers with BEAM; reuse existing runtimes without inserting tuning-only handles. Preserve selected dimensions across JIT replay and executable export/import. No ABI or package-version change.

- Validate launch dimensions without truncation or substitution; preserve unit metadata for interpreter execution. Correct executable CALL access classification and waited kernel/CUDA-graph elapsed accounting. Reject overflowing lane-binding counts before copying inputs. ABI and package versions unchanged.

- Size UOp diagnostic strings from their contents; long PARAM bounds no longer write beyond the former 256-byte allocation.

- Preserve typed PARAM bounds across C, Python and Node/Wasm: fractional/infinite endpoints and arbitrary integers no longer narrow to int64. Keep numeric CSE/ordering and symbolic folding exact; preserve multiple_of and volatility in graph codecs. C ABI79, PGIR18 and PGPM10 require matching bindings/artifacts; package versions and int32 runtime bindings are unchanged.
- Key ordinary compiled execution by the CALL's exact device identity rather than its PROGRAM's backend target; CPU and CPU:1 no longer share a runtime-cache entry.

- Correct derived uint64 bounds before int64 projection and prevent a cleanup crash when exact-interval allocation fails. Add native/Wasm bounds, reduction-clamp and compiler/runtime controls. ABI/package versions unchanged.

- Fix INTERP scalar-binding overreads and negative int64 CUDA scalar arguments in ordinary and graph execution. Reject failed JIT input-signature rewrites before capture. Scalar-binding ABI and package versions unchanged.

- Reject out-of-range runtime scalar bindings instead of silently narrowing them. Abort scalar binding, memory planning and JIT lowering on substitution failure; preserve the original graph for retry. The existing int32 binding domain is tracked as PG-PARITY-028, not a parity allowance. ABI/package versions unchanged.

- Allocate only active PROGRAM globals; allow copying empty storage in ordinary and CUDA graph execution. Correct C arena address alignment (including wasm32) and reject overflowing allocations. Wasm runtime sanitizer failures now stop the gate. ABI/package versions unchanged.

- Match Tinygrad 0.14 symbolic launch limits and reduction matcher axes/arity/boolean-zero contracts. Failed guard inspection no longer publishes partial range shrinking. ABI/package versions unchanged.

- Retarget shared contracts to Tinygrad v0.14.0. Preserve vector gated-load alternatives after movement lowering and correct Wasm scalar predicates in SIMD selection. Keep ABI78 and package versions unchanged.

- Match pinned default late integer decomposition and comparison graphs, including division/multiplication by one, signed division corrections, negative comparisons and singleton intervals.
- Consolidate shared elementwise/UOp construction and move placement, call ordering and cache utilities out of frontend adaptation. Remove the private frontend catch-all and duplicate declarations; verify C/C++ headers. C dtype-ID cast users include frontend.h. Existing frontend ABI/package versions unchanged.

- Preserve INDEX lane metadata and strong local-storage dtypes. Resume partial PROGRAMs without repeating lowering or X86 register allocation; restore missing program metadata and estimates while retaining supplied stages. ABI/package versions unchanged.

- Preserve reduction-axis and loop-barrier dependency order, distinguish ALU from absent address space during load insertion, reject graph IF/ENDIF before linear cleanup, and key compiled programs on supported compiler settings without narrowing TC options to one byte. ABI/package versions unchanged.

- Preserve LOCAL stage integer identities through grouped reduction, CSE and graph serialization. C ABI78, PGIR17 and PGPM9 require matching frontends/artifacts; package version remains0.4.2.

- Preserve full-width schedule argument slots, reject overflowing graph-source counts, and fail buffer-limit lowering on allocation/count failure. Match pinned precompiled-output dependencies on the STORE value. Source-arity debt remains explicit; ABI/package versions unchanged.

- Stop retaining CPU shadows during buffer readback; returned arrays remain independent copies. Reject invalid transfer extents and missing allocator callbacks. Range-cache allocation failure now follows the existing fatal-OOM policy instead of overflowing or reporting an empty range set. ABI/package versions unchanged.

- Make required Qwen CUDA and HF fixture gates fail on unavailable execution/dependencies; preserve individual optional HF skips. Add isolated installed sdist/npm native-and-Wasm acceptance with actual native-build failure. Reject nonzero Python WINO requests instead of silently ignoring them; Winograd remains unimplemented. ABI/package versions unchanged.

- Fix shared reduction lowering: preserve complete loop tuples and outer dependencies, discover mergeable ENDs from the graph, preserve cloned range metadata, and fail cleanly on local scratch allocation failure. Grouped reductions no longer bypass LOCAL staging when non-group loop counts exceed Tensor rank. LOCAL stage integer identity remains registered parity debt; ABI and package versions are unchanged.

### Safety
- Preserve exact large linearizer priorities and TUPLE_ORDER, reject failed temporary ordering allocations, honor NO_MEMORY_PLANNER, propagate mandatory heuristic failures, match literal LOCAL admission and SHRINK reindex topology, and preserve X86 hexadecimal compiler errors and whitespace semantics.
- Match pinned scheduler negative-axis and proved full-axis options; reject oversized symbolic BEAM candidates and preserve strict NVRTC compiler errors. Do not specialize symbolic matvec dimensions through the static shape cache.
- Correct TC swizzle lane mapping and keep warp arithmetic weak until index lowering; nonuniform CUDA FP16/F32 matmul now matches pinned Tinygrad with BEAM off/on. Fix symbolic optimizer threshold, shared-memory and divisibility decisions without treating dynamic extents as zero.
- Remove fixed scheduler metadata/addend limits, preserve copied scheduler ownership on failure, and cancel timed-out BEAM compilation without leaking compiler subprocesses. Align TC/matvec admission and full-axis option history with pinned Tinygrad.
- Size and order native beam scratch using PARAM storage metadata and the C scalar ABI; normalize optimized END ranges before candidate verification. Rejected candidates no longer hide this missing normalization behind baseline execution.
- Release CUDA renderer scratch on rejection, remove fixed64-parameter storage in C/CUDA renderers and C beam timing, and preserve Wasm's optional size output. Beam scratch allocation checks size and allocation failure; ABI75 and package versions unchanged.
- Make C buffer set/attach/adopt transactional and status-returning (ABI75); preserve old storage on metadata allocation failure and reject unsafe alias replacement. Propagate host/file construction failures and clean up mappings. Wasm rejects failed host-write allocation before touching address zero and preserves cold WebGPU bindings on registration failure. Package versions unchanged.

### Added
- C-backed IGNORE_BEAM_CACHE through Python Context and JS runtime.ignoreBeamCache (ABI77; package versions unchanged).
- C-backed BEAM policy through Python Context and JS runtime.beam (ABI76; package versions unchanged), with selected-backend candidate compilation/timing and validated search-cache replay. WebGPU search timing requires optional timestamp queries.
- Exact scalar/tuple/deviceless UOp and Tensor device metadata across Python/JS through C (ABI74). Deviceless operands compose with concrete devices without implicit transfer; C owns result backend selection. Package versions remain unchanged.
- Full-name Tensor device transfer across C/Python/JS (ABI73; package versions unchanged), including native DISK destinations. Opt-in source/dependency/toolchain-locked reuse of original Tinygrad controls; Polygrad always executes.
- Shared C DEFAULT_FLOAT/DEFAULT_INT policy across dtype commitment, promotion, compiler keys and Python/JS controls (ABI72; package version unchanged).
- NOOPT compilation policy in C/Python/JS, including program-cache separation and context restoration; preserve explicit BEAM and required lowering.
- C einsum scalar operands, ellipses and uppercase labels, with Tensor sum accumulation semantics.
- Like/normal factories, product/log reductions, normalization, exact GELU, shape helpers, padding modes and sparse-loss options across Python/JS with shared C owners.
- Shared C spatial operations across Python/JS: average pooling, max-pooling ceil/indices, unpooling, interpolation, transposed convolution, and Invalid-backed anonymous storage.
- Shared C indexed reads/writes with Python/JS syntax adapters, writable frontend grad references, and ordinary BCE reduction selection.
- Shared C/Python/JS pointwise math, finite/closeness checks, copysign/lerp, weighted logits BCE and NLL reductions with autograd.
- Shared C/Python/JS all/any and cumulative sum/product/extrema, including split scans and first-occurrence extrema indices.
- Add a source-locked CPU upstream ops test lane, separate from unchanged-test coverage; retain forward/gradient assertions and explicit known-failure baselines.
- C-backed `models.Sequential` and `models.Graph` factories, with explicit WebGPU async variants. Shared JSON configurations construct ordinary Models using a bounded component catalogue, fresh Repeat expansion, explicit parameter sharing and existing training/export paths.
- One-time Model.trace callable capture, JS traceAsync for WebGPU, and custom-model interchange/lifetime regressions.

### Changed
- C ABI71 includes the Tensor extensions and compiler-policy accessors; package versions remain unchanged.
- Extend the max-pooling C Tensor signature for ceil mode and optional indices (introduced in ABI69).
- Python Tensor.dtype now returns DType objects. UOp.const/variable use value/name-first signatures with an optional explicit ctx keyword; old context-first calls are removed.
- Rename Instance/PolyInstance and their public symbols to Model/PolyModel across C, Python and JavaScript, without compatibility aliases (introduced in ABI68).
- Keep training on Model, with private training/root-publication boundaries and one shared sealing path for explicit bindings, callable capture and C families.
- Python state reads now return independent copies; named typed reads/writes and interface metadata are available in both frontends.

### Fixed
- Read scalar values, not their pointer addresses, in Wasm runner argument packing; add the wasm32 sanitizer runtime regression to the full matrix.
- Consume explicit kernel options before automatic search, reject invalid options, and split structurally divisible symbolic scheduler bounds without substituting maximum dimensions.
- Preserve END predicate/effect dependencies and full RANGE tuple ordering; consume horizontal axes only once in grouped reductions. Reject invalid PROGRAM global slots before runtime argument indexing.
- Avoid undefined signed arithmetic in float64 and FP8-FNUZ dtype decomposition. Reject failed range-propagation scratch allocation without changing PCONTIG or leaking/using a partial range map.
- Correct C GGUF Q4/Q6 bit-plane ordering and byte-aligned reads; reject incomplete conversion and failed Model weight writes. Reclaim failed matcher/binding allocations and reject failed IR root traversal.
- Model import rejects malformed/partial HF shards, GGUF spans and incomplete quantization blocks. Correct GGUF I8/I16/I32 IDs to24/25/26; reclaim partial IR metadata, Model names and safetensors metadata, and prevent F32 byte-count overflow.
- INTERP rejects failed local/index-map allocations and invalid active image stores; checked sizing avoids signed overflow, and loop-local storage is reclaimed on success and failure.
- Shape allocation failures no longer poison cached rank or pass NULL to a shape copy. Preserve initialized-dimension invariants in Tensor consumers.
- Match configured dtype construction in indexed writes, embedding, raw cross-entropy/mean, QR and seeded factories. Raw C sparse cross-entropy now rejects floating class indices like Tinygrad rather than silently casting them.
- Preserve native DISK COPY effects, exact paths and mapped-storage validity through chained copies and readback. Python Path-to-Path construction copies to the requested file. Wasm file mapping remains unsupported.
- Reject empty storage-view shortcuts safely and missing allocator transfer callbacks before staging or writes.
- Reject float shifts and integer arange overflow; infer wider integer ranges when required. Permit randn integer/bool output casts and preserve configured dtypes in derived range graphs.
- Select rounded division after promotion; preserve uint64 scalar/full literals, STACK alias gradients, and strong-typed storage when cloning weak computations. Reject invalid factory shapes and avoid narrowing movement sizes/offsets to int32.
- Match paired/broadcast index construction, last-duplicate writes, detached storage aliases, weak RHS admission and mixed bool/int list indices. Preserve all int64 words in Wasm dimension/stride marshalling and shape reads.
- Match pinned sign dtype/NaN behavior and ordered erf/softplus/isinf construction. Preserve 8/16-bit cast and intermediate arithmetic values in Wasm and WGSL; fix negative float-to-narrow conversion in those backends and INTERP. WGSL correctness extension is registered as PG-DIV-008; no Tensor graph allowance.
- Preserve scan accumulation dtypes and exact identities; add zero-aware product gradients, restore pinned triangular construction, correct linearizer STORE range accounting, and execute nested interpreter IF store masks.
- Validate PAD/SHRINK offsets and extents in shared C shape inference, rejecting definite invalid slices while preserving pinned unresolved-symbolic admission. Match boolean minimum's XOR graph and bypass redundant symbolic rewriting of constant shape lanes.
- Route Python/JS `min` through dtype-aware C inverse/MAX/inverse, preserving unsigned zero, signed minima and exact integer masks. Preserve NaN when folding negative fractional powers; match pinned signed-zero negative-power constants.
- Expose parsed Python DEV target requests and reject unsupported nonzero IMAGE execution; restore earlier Context settings when entry fails. Correct the upstream runner's numeric disabled-backend setting.
- Add Python/JS maximum-shape and pad/shrink-to helpers, optional shrink axes, and identity-preserving full slices/empty flips; preserve symbolic shrink source identities through existing C constructors.
- Add matching Python/JS storage-size and floating-point dtype queries; reject weak storage widths. Reject conflicting explicit owners for existing UOp constants and read source manifests as UTF-8 independently of locale.
- Align JS null/scalar-zero, integer-list truncation and weak/ragged/nonfinite admission with the pinned constructor contract; preserve BigInt precision for integer storage on native, Wasm and browser paths.
- Match pinned Python Tensor(None) scalar-zero construction, integer-list storage truncation and non-scalar weak-dtype rejection; preserve normal NumPy conversion and ragged-input validation.
- Host Tensor writes materialize pending COPY/assign effects before updating current physical storage; preserve logical policies and buffer identity without replacing graph roots. JS adds copyFromAsync for writes requiring WebGPU materialization.
- Interpret Python Tensor bytes as owned raw storage instead of numeric input; preserve typed BF16/FP8 encodings. Expose the existing C lossless-cast predicate through the corrected scalar dtype FFI layout.
- Canonicalize symbolic movement shape operands before graph construction and identity checks; reject mismatched PAD/SHRINK ranks in C composition helpers.
- Normalize and validate C permutation axes, preserve identity permutations, and flatten nested INDEX operations with empty coordinate tuples to match pinned Tinygrad.
- Match pinned weak-lowering tag semantics for fresh constants, ALU resources, and gated addresses.
- Align call construction with pinned requested-base and virtual/ALU rules; remove training markers and reuse shared parent storage. Model import explicitly materializes closed initializers into owned storage.
- Preserve invalid-value markers through scalar promotion and weak lowering; match pinned UNSHARD boundaries and MSELECT storage-identity checks without merging device lanes.
- Include missing Python source-distribution headers and verify manifest dependency closure. Repair the native coverage target's compiler flags and per-source profile paths.
- Deduplicate tied trainable storage after freeze/unfreeze, preventing duplicate Adam updates and assign cycles.
- Select the declared objective and invalidate objective-dependent caches; switching losses no longer silently reuses the previous gradient graph or publishes a hardcoded loss buffer.
- Preserve typed named objective storage while converting C loss diagnostics to float32.
- Include persistent non-optimizer AUX in model-only exports.
- Route uniform placement to the uniform C API. WebGPU capture initializes the device and retains construction ownership through queued snapshot work.

### Removed
- Empty ResNet, ViT and Llama model placeholders and obsolete mirror exclusions; no implemented model family was removed.

## 0.4.2 (2026-07-06)

### Added
- Sync-first JavaScript default API with lazy default runtime exports, explicit runtime creation, and explicit async startup/readback variants.
- Split browser distribution into sync and async bundles with package-export coverage for Node, browser, and async entrypoints.

### Fixed
- WASM specialized matmul relaxed-SIMD validation now probes Node execution semantics instead of hardcoding one V8 flag.
- WASM relaxed-madd operand order is corrected, with relaxed-madd enabled for the ABT specialized matmul path and strict SIMD retained for AB until benchmarks justify it.
- Browser WebGPU lazy-init capability reporting now updates `caps.f16` after device initialization.

## 0.4.1 (2026-07-05)

### Fixed
- Python package repair release: wheels built from the PyPI source distribution
  now include the pure Python `polygrad` package files in addition to
  `polygrad._native`. The npm 0.4.0 package was not affected.

## 0.4.0 (2026-07-05)

### Added
- Tinygrad-style custom kernels across C, Python, and JavaScript frontends, including UOp `CALL` execution, compile/JIT replay, grouped stores, reductions, comparisons, unary ops, and repeated input mutation coverage.
- Structured linalg APIs for QR modes, triangular solve, Cholesky, Cholesky solve, solve, and least squares, implemented as portable tensor-composed fallbacks with NumPy and Torch reference coverage.
- Expanded native x86 backend work aligned with tinygrad X86 instruction selection patterns, plus common and x86-specific C, Python, and JavaScript coverage.
- Batched typed readback helpers and stable tensor input update APIs for compiled loops.
- Qwen3 GGUF smoke coverage on native CUDA/CPU and browser WebGPU, plus jax-js WASM comparison benchmarks for generic kernels, matmul, and model-shaped workloads.

### Changed
- Node and browser runtimes now use the unified C realization path for WASM/WebGPU execution and ship refreshed package artifacts.
- Runtime placement preserves the split between exportable logical tensor roots and realized physical roots after device placement.
- Python source synchronization now includes the engine JIT sources required by the shared runtime in sdist and wheel builds.

### Fixed
- CUDA compiled replay now rebinds custom-kernel producer outputs for downstream compiled consumers.
- WebGPU WGSL rendering now handles `COPY`/`UNROLL`-wrapped multi-output custom-kernel stores and selects the correct lane for unrolled/vector store values.
- WASM, CUDA, and x86 lowering fixes for custom-kernel reductions, floor division/modulo, typed constants, and vector/lane rendering.
- Advisory capability checks now cover sort, argsort, and topk queries that direct execution already supported.

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
- Interpreter backend (`interp.c`): walks linearized UOps directly in C without external compiler. Handles all scalar types, BITCAST (int32/float32 bit reinterpretation), RANGE/END loops, DEFINE_REG accumulators, and the full codegen decomposition pipeline (EXP2 polynomial, LOG2, SIN). Serves as correctness oracle.
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
