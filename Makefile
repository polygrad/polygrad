CC ?= gcc
AR ?= ar
CFLAGS_COMMON = -std=c11 -D_POSIX_C_SOURCE=200809L -Wall -Wextra -Wpedantic -Wno-unused-parameter -pipe -Isrc
CFLAGS_RELEASE = $(CFLAGS_COMMON) -O2
CFLAGS_DEBUG = $(CFLAGS_COMMON) -g -O0 -fsanitize=address,undefined -fno-omit-frame-pointer
LDFLAGS = -lm
LDFLAGS_DEBUG = -lm -ldl -fsanitize=address,undefined
TSAN_CC ?= clang
TSAN_OPTIONS ?= halt_on_error=1:second_deadlock_stack=1
TSAN_RUNNER ?= setarch $$(uname -m) -R
# Keep LeakSanitizer enabled by default for the native debug test binary.
# Driver/runtime targets can still opt out when investigating external runtime
# leaks:
#   make test-cuda ASAN_OPTIONS=detect_leaks=0:protect_shadow_gap=0
ASAN_OPTIONS ?= detect_leaks=1:protect_shadow_gap=0
UBSAN_OPTIONS ?= print_stacktrace=1:halt_on_error=1
SAN_RUN = ASAN_OPTIONS=$(ASAN_OPTIONS) UBSAN_OPTIONS=$(UBSAN_OPTIONS)
EMCC ?= emcc
EMSDK_PYTHON ?= /usr/bin/python3
PYTHON ?= python
NPM ?= npm
TWINE ?= twine
# Polygrad's scheduler/codegen path uses deeper C call chains than
# Emscripten's default stack reliably supports; keep WASM realization away
# from stack OOB traps on current SDKs.
EMCC_CFLAGS_COMMON = -O2 -std=c11 -Wall -Wextra -Wpedantic -Wno-unused-parameter -Isrc -s STACK_SIZE=8388608
FILC ?= $(HOME)/tools/filc-0.678-linux-x86_64/build/bin/clang
FILC_CFLAGS_DEBUG = -std=c11 -D_POSIX_C_SOURCE=200809L -Isrc -g -O0 -w

# Detect CUDA availability
HAS_CUDA := $(shell test -f /usr/include/cuda.h && echo 1 || echo 0)

SRC = src/ops.c src/dtype.c src/arena.c src/hashmap.c src/utils.c src/bigint.c src/selftest.c src/ctx.c src/device.c src/placer.c src/engine/realize.c src/engine/jit.c src/uop/ops.c src/uop/spec.c src/uop/weak.c src/uop/movement.c src/uop/symbolic.c src/mixin/elementwise.c src/mixin/movement.c src/uop/upat.c src/alu.c src/shape.c src/autograd.c src/codegen/codegen.c src/codegen/opt/tc.c src/codegen/decomp/dtype.c src/codegen/simplify.c src/codegen/gpudims.c src/codegen/late/coalesce.c src/codegen/late/gater.c src/codegen/late/linearizer.c src/renderer/cstyle.c src/renderer/wgsl.c src/runtime/support/memory.c src/runtime_cpu.c src/runtime_wasm.c src/runtime_webgpu.c src/wasm_builder.c src/renderer/wasm.c src/frontend.c src/tensor.c src/optim.c src/schedule/rangeify.c src/schedule/multi.c src/schedule/allreduce.c src/schedule/schedule.c src/schedule/memory.c src/schedule/indexing.c src/nn.c src/engine/schedule.c src/interp.c
FILC_SRC = src/ops.c src/dtype.c src/arena.c src/hashmap.c src/utils.c src/bigint.c src/selftest.c src/ctx.c src/device.c src/placer.c src/engine/realize.c src/engine/jit.c src/uop/ops.c src/uop/spec.c src/uop/weak.c src/uop/movement.c src/uop/symbolic.c src/mixin/elementwise.c src/mixin/movement.c src/uop/upat.c src/alu.c src/shape.c src/autograd.c src/codegen/codegen.c src/codegen/opt/tc.c src/codegen/decomp/dtype.c src/codegen/simplify.c src/codegen/gpudims.c src/codegen/late/coalesce.c src/codegen/late/gater.c src/codegen/late/linearizer.c src/renderer/cstyle.c src/renderer/wgsl.c src/runtime/support/memory.c src/runtime_wasm.c src/runtime_webgpu.c src/wasm_builder.c src/renderer/wasm.c src/frontend.c src/tensor.c src/optim.c src/schedule/rangeify.c src/schedule/multi.c src/schedule/allreduce.c src/schedule/schedule.c src/schedule/memory.c src/schedule/indexing.c src/nn.c src/engine/schedule.c src/interp.c
LOADER_SRC = src/loaders/decoded.c src/loaders/import_error.c src/loaders/bind.c src/loaders/hf_decode.c src/loaders/gguf_decode.c src/loaders/gguf_loader.c src/loaders/import_desc.c
CODEC_SRC = vendor/cjson/cJSON.c src/safetensors.c src/wlrn.c src/ir.c src/bundle.c src/model.c src/tokenizer.c src/models/compose.c src/models/mlp.c src/models/tabm.c src/models/nam.c src/models/registry.c src/models/gpt2.c src/models/qwen3.c src/models/hf_loader.c $(LOADER_SRC)
TEST_SRC = test/test_main.c test/test_uop.c test/test_utils.c test/test_dtype.c test/test_bigint.c test/test_pat.c test/test_sym.c test/test_shape.c test/test_schedule_engine.c test/test_autograd.c test/test_codegen.c test/test_wasm.c test/test_rangeify.c test/test_reduce_simplify.c test/test_nn.c test/test_tensor.c test/test_fusion_fuzzer.c test/test_future_passes.c test/test_safetensors.c test/test_wlrn.c test/test_ir.c test/test_model.c test/test_program.c test/test_mlp.c test/test_tabm.c test/test_nam.c test/test_hf.c test/test_qwen3.c test/test_f16.c test/test_schedule_runtime.c test/test_bundle.c test/test_registry.c test/test_placement.c test/test_realize.c test/test_threading.c
PROJECT_HEADERS := $(shell find src test bench vendor -type f -name '*.h' -print | sort)
ANALYZE_SRC = $(filter-out vendor/%,$(sort $(SRC) $(CODEC_SRC)))
ANALYZE_FLAGS ?=
FORMAT_SRC := $(shell find src test bench -type f \( -name '*.c' -o -name '*.h' \) -print | sort)

ifeq ($(HAS_CUDA), 1)
  SRC += src/renderer/cuda.c src/runtime_cuda.c
  TEST_SRC += test/test_cuda.c
  CFLAGS_COMMON += -DPOLY_HAS_CUDA=1
endif

# Detect HIP/ROCm availability (check ROCM_PATH, /opt/rocm, and versioned paths)
ROCM_PATH ?= $(shell ls -d /opt/rocm-* 2>/dev/null | sort -V | tail -1)
ifeq ($(ROCM_PATH),)
  ROCM_PATH := /opt/rocm
endif
HAS_HIP := $(shell test -f $(ROCM_PATH)/include/hip/hip_runtime.h && echo 1 || echo 0)
ifeq ($(HAS_HIP), 1)
  SRC += src/renderer/hip.c src/runtime_hip.c
  TEST_SRC += test/test_hip.c
  CFLAGS_COMMON += -DPOLY_HAS_HIP=1
endif

# Detect x86-64 for the tinygrad-style x86 ISA backend.
HAS_X86 := $(shell uname -m | grep -c x86_64)
ifeq ($(HAS_X86), 1)
  SRC += src/renderer/isa/x86.c
  TEST_SRC += test/test_x86.c
  CFLAGS_COMMON += -DPOLY_HAS_X86=1
endif
STATIC_OBJS = $(patsubst %.c,build/obj/static/%.o,$(SRC) $(CODEC_SRC))
STATIC_DEPS = $(STATIC_OBJS:.o=.d)
PARITY_RUNNER_SRC = test/test_tinygrad_runner.c
PARITY_SCRIPT = test/test_tinygrad_parity.py
PARITY_PY ?= $(if $(wildcard references/.venv-tinygrad-py311/bin/python),references/.venv-tinygrad-py311/bin/python,conda run -n tiny python)

# Emscripten uses the complete core/codec set minus native backend owners.
WASM_SRC = $(filter-out src/runtime_cpu.c src/renderer/cuda.c src/runtime_cuda.c src/renderer/hip.c src/runtime_hip.c src/renderer/isa/x86.c,$(SRC)) $(CODEC_SRC)
WASM_EXPORTS := $(shell $(PYTHON) scripts/wasm_exports.py js/src)

WASM_ASYNCIFY_IMPORTS = ['js_webgpu_dispatch','js_webgpu_read_buffer_to_wasm','js_webgpu_read_buffer_to_hostkey']
WASM_ASYNCIFY_ONLY = ['poly_sequential_from_json','poly_graph_from_json','compose_from_json','poly_model_write_buf_named','poly_model_from_binding_arrays','poly_model_from_bindings','poly_model_build','prepare_build_named_value_snapshots','snapshot_build_named_value','copy_initial_buffer_data','poly_realize_sink','poly_model_call','poly_model_train_step','poly_model_set_device_map_arrays','poly_model_set_device','model_place_uniform_device','model_publish_placement','run_model_sink','poly_model_param_data_raw','poly_model_buf_data','poly_model_buf_data_raw','poly_model_export_weights_ex','poly_model_save_bundle_ex','poly_model_read_buf','poly_model_write_buf','sync_buf_to_host','poly_model_readback_param','poly_model_readback_buf','poly_realize_uops','poly_realize_tensors','poly_jit_end_capture','poly_jit_run','poly_jit_run_captured_linear','poly_run_linear','poly_webgpu_execute','poly_buffer_copy','poly_buffer_ensure_device_current','poly_buffer_ensure_host_current','poly_buffer_read','poly_buffer_write','host_copy_in','webgpu_copy_out']
WASM_ASYNCIFY_FLAGS = -s ASYNCIFY=1 \
	-s "ASYNCIFY_IMPORTS=$(WASM_ASYNCIFY_IMPORTS)" \
	-s "ASYNCIFY_ONLY=$(WASM_ASYNCIFY_ONLY)"

# These targets compile and link source files directly rather than through the
# dependency-emitting static-object rule. Keep them sensitive to every project
# header and filter non-C prerequisites out of their compiler argument lists.
DIRECT_C_BUILDS = build/libpolygrad.so build/polygrad_test build/polygrad_test_filc \
	build/polygrad_parity_runner build/bench_polygrad build/bench_smoke \
	build/bench_cuda build/polygrad_parity_runner_cuda \
	build/polygrad_parity_runner_hip build/bench_hip build/fuzz_sym \
	build/fuzz_sym_div build/test_p2p build/polygrad_test_cov \
	build/polygrad_test_msan build/polygrad_test_tsan build/polygrad.js \
	build/polygrad.wasm build/core.async.js build/core.sync.js
$(DIRECT_C_BUILDS): $(PROJECT_HEADERS) Makefile

# Run the same integer/constant-fold tests with wasm32 size_t and sanitizers.
# Native-only execution cannot detect host-width narrowing in the C core.
.PHONY: test-bigint-wasm
test-bigint-wasm: build/test_bigint.js
	$(SAN_RUN) $(NODE) build/test_bigint.js --require-no-skips bigint

build/test_bigint.js: $(WASM_SRC) test/test_main.c test/test_bigint.c test/test_harness.h $(PROJECT_HEADERS) Makefile
	@mkdir -p build
	EMSDK_PYTHON=$(EMSDK_PYTHON) $(EMCC) $(EMCC_CFLAGS_COMMON) -O1 -g \
		-fsanitize=address,undefined -s ASSERTIONS=1 -s ALLOW_MEMORY_GROWTH=1 \
		-s WASM_ASYNC_COMPILATION=0 -s ENVIRONMENT=node -s EXIT_RUNTIME=1 \
		-o $@ $(filter %.c,$^)

QWEN3_GGUF ?= $(if $(POLY_QWEN3_GGUF),$(POLY_QWEN3_GGUF),$(CURDIR)/temp/Qwen3-0.6B-Q8_0.gguf)
# Broader Playwright browser matrix. The default target stays Chromium-only;
# test-browser-matrix uses this optional smoke matrix and skips unavailable
# local executables so developer machines do not need every browser installed.
BROWSER_MATRIX ?= chromium,firefox,chrome-system=chromium@/usr/bin/google-chrome,chromium-snap=chromium@/snap/bin/chromium
BROWSER_MATRIX_DEVICES ?= auto

.PHONY: all test test-fast test-common test-common-cpu test-common-cuda test-common-hip test-common-interp test-common-x86 test-specific-cuda test-specific-hip test-specific-x86 test-cuda test-hip test-interp test-x86 test-harness-skip-accounting test-parity test-parity-opt test-parity-ir test-parity-ir-opt test-parity-graph parity-graph-report reference-migration-report test-parity-op-census parity-op-census-report test-compat-tinygrad-tier1 test-compat-tinygrad-convnext test-parity-cuda test-parity-hip test-symbolic-z3 test-qwen3 test-browser-qwen3 require-qwen3-gguf test-wasm test-wasm-new test-native test-browser test-browser-matrix test-js-browser-matrix test-model-interchange test-p2p test-p2p-browser bench bench-cuda bench-model-cuda bench-hlb-cuda-semantic bench-hlb-cuda-timing bench-hlb-cuda-manifest bench-hip bench-train-py bench-smoke bench-local-baseline bench-update-local-baseline bench-smoke-regression bench-ci-regression bench-ratios bench-ratio-local-baseline bench-update-local-ratio-baseline bench-parity bench-jax-js-wasm bench-jax-js-matmul-wasm bench-jax-js-model-wasm bench-jax-js-browser-wasm bench-jax-js-browser-matmul-wasm bench-jax-js-browser-model-wasm bench-compare bench-compare-global bench-regression bench-update-baseline fuzz fuzz-smoke fuzz-nightly fuzz-symbolic fuzz-symbolic-div wasm wasm-pkg build-py build-py-sdist build-py-wheel build-python test-py-sdist-install publish-py publish-python build-js publish-js clean analyze cppcheck format format-check test-msan test-tsan verify coverage test-full test-js-native-cpu test-js-native-x86 test-js-native-interp test-js-native-cuda test-js-native-hip test-js-package test-filc-interp-fast sync-source-mirrors verify-source-mirrors test-py-x86

all: build/libpolygrad.a build/libpolygrad.so

build/libpolygrad.a: $(STATIC_OBJS)
	@mkdir -p build
	@tmp="$@.$$$$"; \
		$(AR) rcs "$$tmp" $^ && mv "$$tmp" "$@"

build/obj/static/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS_RELEASE) -MMD -MP -c -o $@ $<

-include $(STATIC_DEPS)

build/libpolygrad.so: $(SRC) $(CODEC_SRC)
	@mkdir -p build
	$(CC) $(CFLAGS_RELEASE) -fPIC -shared -o $@ $(filter %.c,$^) -lm -ldl

test: build/polygrad_test
	$(SAN_RUN) ./build/polygrad_test

test-fast: build/polygrad_test
	$(SAN_RUN) ./build/polygrad_test --fast

# Full C suite routed through the selected backend. Backend portability failures
# must remain visible; do not replace this with a small positive allowlist.
test-common: build/polygrad_test
	$(SAN_RUN) ./build/polygrad_test --common

test-common-cpu: build/polygrad_test
	$(SAN_RUN) POLY_DEVICE=cpu ./build/polygrad_test --common

test-common-cuda: build/polygrad_test
	$(SAN_RUN) POLY_DEVICE=cuda ./build/polygrad_test --common

test-common-hip: build/polygrad_test
	$(SAN_RUN) POLY_DEVICE=hip ./build/polygrad_test --common

test-common-interp: build/polygrad_test
	$(SAN_RUN) POLY_DEVICE=interp ./build/polygrad_test --common

test-common-x86: build/polygrad_test
	$(SAN_RUN) POLY_DEVICE=x86 ./build/polygrad_test --common

test-cuda: test-common-cuda test-specific-cuda

test-hip: test-common-hip test-specific-hip

test-interp: test-common-interp

test-x86: test-common-x86 test-specific-x86

test-specific-x86: build/polygrad_test
	$(SAN_RUN) POLY_DEVICE=x86 ./build/polygrad_test --specific x86

require-qwen3-gguf:
	@if [ ! -f "$(QWEN3_GGUF)" ]; then \
		echo "Qwen3 GGUF fixture not found: $(QWEN3_GGUF)"; \
		echo "Set POLY_QWEN3_GGUF=/path/to/Qwen3-0.6B-Q8_0.gguf or place it under temp/."; \
		exit 2; \
	fi

test-qwen3: build/polygrad_test require-qwen3-gguf
	$(SAN_RUN) POLY_QWEN3_GGUF="$(QWEN3_GGUF)" ./build/polygrad_test qwen3

# Backend-specific tests only. --specific requires TEST_BACKEND and an exact
# suite match, so portable tests with backend names remain in test-common-*.
test-specific-cuda: build/polygrad_test
	$(SAN_RUN) POLY_DEVICE=cuda ./build/polygrad_test --require-no-skips --specific cuda

test-specific-hip: build/polygrad_test
	$(SAN_RUN) POLY_DEVICE=hip ./build/polygrad_test --require-no-skips --specific hip

test-harness-skip-accounting: build/polygrad_test
	@mkdir -p build
	@set +e; \
	  POLY_TEST_FORCE_RUNTIME_SKIP=1 $(SAN_RUN) \
	    ./build/polygrad_test --require-no-skips --specific harness \
	    > build/test-harness-skip.log 2>&1; \
	  rc=$$?; set -e; cat build/test-harness-skip.log; \
	  test $$rc -eq 2

test-parity: build/polygrad_parity_runner
	$(SAN_RUN) CACHELEVEL=0 $(PARITY_PY) $(PARITY_SCRIPT) --runner build/polygrad_parity_runner --mode values

test-parity-opt: build/polygrad_parity_runner
	$(SAN_RUN) CACHELEVEL=0 $(PARITY_PY) $(PARITY_SCRIPT) --runner build/polygrad_parity_runner --mode values

test-parity-ir: build/polygrad_parity_runner
	$(SAN_RUN) CACHELEVEL=0 $(PARITY_PY) $(PARITY_SCRIPT) --runner build/polygrad_parity_runner --mode full --no-opt

test-parity-ir-opt: build/polygrad_parity_runner
	$(SAN_RUN) CACHELEVEL=0 $(PARITY_PY) $(PARITY_SCRIPT) --runner build/polygrad_parity_runner --mode full

GRAPH_PARITY_DIR ?= temp/parity_graph
GRAPH_PARITY_CASE_ARGS ?=
OP_PARITY_DIR ?= temp/parity_ops
COMPAT_TIER1_DIR ?= temp/tinygrad_compat_tier1
COMPAT_CONVNEXT_DIR ?= temp/tinygrad_compat_convnext
UPSTREAM_COMPAT_DIR ?= temp/tinygrad_upstream
UPSTREAM_COMPAT_TESTS ?=
UPSTREAM_COMPAT_ARGS ?=
UPSTREAM_COMPAT_BASELINE ?= test/fixtures/tinygrad_upstream_baseline.json

.PHONY: test-upstream-runner test-compat-tinygrad-upstream test-compat-tinygrad-upstream-ratchet test-compat-tinygrad-ops
test-upstream-runner:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. $(PYTHON) -m pytest -q py/tests/test_tinygrad_upstream.py

# Use a fresh directory for each run; no source edits, import stubs or auto-accept.
test-compat-tinygrad-upstream: build/libpolygrad.so
	$(PARITY_PY) scripts/tinygrad_upstream.py --output $(UPSTREAM_COMPAT_DIR) \
		$(foreach t,$(UPSTREAM_COMPAT_TESTS),--test $(t)) $(UPSTREAM_COMPAT_ARGS)

# Explicit adapted CPU lane: no numerical/gradient test bodies are changed.
test-compat-tinygrad-ops: build/libpolygrad.so
	$(PARITY_PY) scripts/tinygrad_upstream.py --output $(UPSTREAM_COMPAT_DIR) \
		--adapter cpu-ops --test test/backend/test_ops.py $(UPSTREAM_COMPAT_ARGS)

test-compat-tinygrad-upstream-ratchet: build/libpolygrad.so
	$(PARITY_PY) scripts/tinygrad_upstream.py --output $(UPSTREAM_COMPAT_DIR) \
		--baseline $(UPSTREAM_COMPAT_BASELINE) $(UPSTREAM_COMPAT_ARGS)

test-parity-graph parity-graph-report: build/libpolygrad.so
	@mkdir -p $(GRAPH_PARITY_DIR) temp/cc_tmp
	ENGINE=tinygrad DEV=CPU PYTHONPATH=references/tinygrad_latest $(PARITY_PY) \
		test/tensor_graph_cases.py $(GRAPH_PARITY_CASE_ARGS) > $(GRAPH_PARITY_DIR)/tinygrad.json
	ENGINE=polygrad POLY_DEVICE=cpu POLY_TMPDIR=$(abspath temp/cc_tmp) TMPDIR=$(abspath temp/cc_tmp) \
		POLYGRAD_LIB=$(abspath build/libpolygrad.so) PYTHONPATH=py $(PARITY_PY) \
		test/tensor_graph_cases.py $(GRAPH_PARITY_CASE_ARGS) > $(GRAPH_PARITY_DIR)/polygrad.json
	$(PARITY_PY) test/compare_tensor_graphs.py \
		$(GRAPH_PARITY_DIR)/tinygrad.json $(GRAPH_PARITY_DIR)/polygrad.json \
		$(if $(filter parity-graph-report,$@),--report-only,) \
		--output $(GRAPH_PARITY_DIR)/report.json

reference-migration-report: parity-graph-report
	@mkdir -p temp/xdg_cache
	@ARCHBIRD="$$(command -v archbird || true)"; \
	  test -n "$$ARCHBIRD" || ARCHBIRD=/home/anton/tools/miniconda3/envs/agents/bin/archbird; \
	  XDG_CACHE_HOME=$(abspath temp/xdg_cache) "$$ARCHBIRD" map . \
	    --format json --output temp/archbird-polygrad.json --check
	$(PYTHON) scripts/reference_migration.py \
		--graph-report $(GRAPH_PARITY_DIR)/report.json \
		--archbird-map temp/archbird-polygrad.json

MIGRATION_EVIDENCE ?= temp/reference_migration/evidence.json
.PHONY: test-reference-migration reference-migration-check
test-reference-migration:
	PYTHONPATH=. $(PYTHON) -m pytest -q py/tests/test_reference_migration.py

# Validate frozen evidence without silently rebuilding or replacing its inputs.
reference-migration-check: test-reference-migration
	$(PYTHON) scripts/reference_migration.py --strict \
		--graph-report $(GRAPH_PARITY_DIR)/report.json \
		--archbird-map '' --evidence $(MIGRATION_EVIDENCE)

test-parity-op-census parity-op-census-report: build/libpolygrad.so
	@mkdir -p $(OP_PARITY_DIR)
	POLYGRAD_LIB=$(abspath build/libpolygrad.so) \
		PYTHONPATH=py:references/tinygrad_latest $(PARITY_PY) \
		test/op_vocabulary_census.py \
		$(if $(filter parity-op-census-report,$@),--report-only,) \
		--output $(OP_PARITY_DIR)/report.json

test-compat-tinygrad-tier1: build/libpolygrad.so
	@mkdir -p $(COMPAT_TIER1_DIR) temp/cc_tmp
	ENGINE=tinygrad DEV=CPU PYTHONPATH=test:references/tinygrad_latest $(PARITY_PY) \
		test/tinygrad_compat_cases.py > $(COMPAT_TIER1_DIR)/tinygrad.json
	ENGINE=polygrad DEV=CPU POLY_DEVICE=cpu \
		POLY_TMPDIR=$(abspath temp/cc_tmp) TMPDIR=$(abspath temp/cc_tmp) \
		POLYGRAD_LIB=$(abspath build/libpolygrad.so) PYTHONPATH=test:py $(PARITY_PY) \
		test/tinygrad_compat_cases.py > $(COMPAT_TIER1_DIR)/polygrad.json
	$(PARITY_PY) test/compare_tinygrad_compat.py \
		$(COMPAT_TIER1_DIR)/tinygrad.json $(COMPAT_TIER1_DIR)/polygrad.json \
		--output $(COMPAT_TIER1_DIR)/report.json

test-compat-tinygrad-convnext: build/libpolygrad.so
	@mkdir -p $(COMPAT_CONVNEXT_DIR) temp/cc_tmp
	ENGINE=tinygrad COMPAT_CASES=convnext DEV=CPU PYTHONPATH=test:references/tinygrad_latest $(PARITY_PY) \
		test/tinygrad_compat_cases.py > $(COMPAT_CONVNEXT_DIR)/tinygrad.json
	ENGINE=polygrad COMPAT_CASES=convnext DEV=CPU POLY_DEVICE=cpu \
		POLY_TMPDIR=$(abspath temp/cc_tmp) TMPDIR=$(abspath temp/cc_tmp) \
		POLYGRAD_LIB=$(abspath build/libpolygrad.so) PYTHONPATH=test:py $(PARITY_PY) \
		test/tinygrad_compat_cases.py > $(COMPAT_CONVNEXT_DIR)/polygrad.json
	$(PARITY_PY) test/compare_tinygrad_compat.py \
		$(COMPAT_CONVNEXT_DIR)/tinygrad.json $(COMPAT_CONVNEXT_DIR)/polygrad.json \
		--output $(COMPAT_CONVNEXT_DIR)/report.json

Z3_FUZZ_ITERS ?= 128
Z3_FUZZ_SEED ?= 0
test-symbolic-z3: build/libpolygrad.so
	POLYGRAD_LIB=$(abspath build/libpolygrad.so) $(PARITY_PY) \
		test/external/fuzz_symbolic_z3.py --mode general --seed $(Z3_FUZZ_SEED) --iters $(Z3_FUZZ_ITERS)
	POLYGRAD_LIB=$(abspath build/libpolygrad.so) $(PARITY_PY) \
		test/external/fuzz_symbolic_z3.py --mode div --seed $(Z3_FUZZ_SEED) --iters $(Z3_FUZZ_ITERS)
	POLYGRAD_LIB=$(abspath build/libpolygrad.so) $(PARITY_PY) \
		test/external/fuzz_symbolic_z3.py --mode fixed --seed $(Z3_FUZZ_SEED) --iters $(Z3_FUZZ_ITERS)

build/polygrad_test: $(SRC) $(CODEC_SRC) $(TEST_SRC)
	@mkdir -p build
	$(CC) $(CFLAGS_DEBUG) -DPOLY_TESTING -o $@ $(filter %.c,$^) $(LDFLAGS_DEBUG)

build/polygrad_test_filc: $(FILC_SRC) $(CODEC_SRC) $(TEST_SRC)
	@mkdir -p build
	$(FILC) $(FILC_CFLAGS_DEBUG) -o $@ $(filter %.c,$^) -lm -ldl

test-filc-interp-fast: build/polygrad_test_filc
	POLY_DEVICE=interp CC=$(FILC) ./build/polygrad_test_filc --fast

build/polygrad_parity_runner: $(SRC) $(CODEC_SRC) $(PARITY_RUNNER_SRC)
	@mkdir -p build
	$(CC) $(CFLAGS_RELEASE) -o $@ $(filter %.c,$^) $(LDFLAGS)

bench: build/bench_polygrad
	./build/bench_polygrad

build/bench_polygrad: $(SRC) $(CODEC_SRC) bench/bench_polygrad.c
	@mkdir -p build
	$(CC) $(CFLAGS_RELEASE) -o $@ $(filter %.c,$^) -lm -ldl

BENCH_SMOKE_JSON ?= bench/results/smoke-latest.json
BENCH_LOCAL_BASELINE ?= bench/baselines/local/$(shell hostname -s)-cpu.json
BENCH_BASELINE ?= $(BENCH_LOCAL_BASELINE)
BENCH_CI_BASELINE ?=
BENCH_SMOKE_ARGS ?=
BENCH_COMPARE_ARGS ?=

build/bench_smoke: $(SRC) $(CODEC_SRC) bench/bench_smoke.c
	@mkdir -p build
	$(CC) $(CFLAGS_RELEASE) -o $@ $(filter %.c,$^) -lm -ldl

bench-smoke: build/bench_smoke
	$(PYTHON) bench/bench_smoke.py --runner $< --output $(BENCH_SMOKE_JSON) $(BENCH_SMOKE_ARGS)

bench-local-baseline: build/bench_smoke
	$(PYTHON) bench/bench_smoke.py --runner $< --output $(BENCH_LOCAL_BASELINE) $(BENCH_SMOKE_ARGS)
	@echo "Updated local benchmark baseline: $(BENCH_LOCAL_BASELINE)"

bench-update-local-baseline: bench-local-baseline

bench-smoke-regression: bench-smoke
	$(PYTHON) bench/bench_compare_abs.py $(BENCH_BASELINE) $(BENCH_SMOKE_JSON) $(BENCH_COMPARE_ARGS)

bench-ci-regression: bench-smoke
	@if [ -z "$(BENCH_CI_BASELINE)" ]; then \
		echo "Set BENCH_CI_BASELINE=path/to/runner-specific-baseline.json"; \
		exit 2; \
	fi
	$(PYTHON) bench/bench_compare_abs.py $(BENCH_CI_BASELINE) $(BENCH_SMOKE_JSON) $(BENCH_COMPARE_ARGS)

ifeq ($(HAS_CUDA), 1)
bench-cuda: build/bench_cuda
	./build/bench_cuda

bench-model-cuda: build/libpolygrad.so js/build/Release/polygrad_napi.node
	POLYGRAD_LIB=$(abspath build/libpolygrad.so) PYTHONPATH=references/tinygrad_latest $(PARITY_PY) bench/bench_model_cuda_vs_tinygrad.py

# Bounded semantic gate for the exact pinned HLB source. The Python driver
# owns engine-specific caches, model bytes, provenance, and comparison output.
bench-hlb-cuda-semantic: build/libpolygrad.so
	@mkdir -p temp/hlb_benchmark
	@set -eu; \
		run_root=$$(mktemp -d -p temp/hlb_benchmark .semantic.XXXXXX); \
		rmdir "$$run_root"; \
		$(PARITY_PY) bench/bench_hlb_cifar.py \
			$(HLB_BENCH_ARGS) \
			--mode semantic \
			--python $(PARITY_PY) --polygrad-lib $(abspath build/libpolygrad.so) \
			--output-dir "$$run_root"; \
		test -s "$$run_root/comparison.json"

# No-live-Tensor bounded timing gate. The driver first requires a fresh
# one-step semantic canary on the same source/library/state, then records
# deterministic eager/capture/replay windows in both process orders.
bench-hlb-cuda-timing: build/libpolygrad.so
	@mkdir -p temp/hlb_benchmark
	@set -eu; \
		run_root=$$(mktemp -d -p temp/hlb_benchmark .timing.XXXXXX); \
		rmdir "$$run_root"; \
		$(PARITY_PY) bench/bench_hlb_cifar.py \
			$(HLB_BENCH_ARGS) \
			--mode timing \
			--python $(PARITY_PY) --polygrad-lib $(abspath build/libpolygrad.so) \
			--output-dir "$$run_root"; \
		test -s "$$run_root/comparison.json"

bench-hlb-cuda-manifest: build/libpolygrad.so
	$(PARITY_PY) bench/bench_hlb_cifar.py --manifest-only \
		--python $(PARITY_PY) --polygrad-lib $(abspath build/libpolygrad.so)

build/bench_cuda: $(SRC) $(CODEC_SRC) bench/bench_cuda.c
	@mkdir -p build
	$(CC) $(CFLAGS_RELEASE) -o $@ $(filter %.c,$^) -lm -ldl

test-parity-cuda: build/polygrad_parity_runner_cuda
	CACHELEVEL=0 $(PARITY_PY) $(PARITY_SCRIPT) \
		--runner build/polygrad_parity_runner_cuda --cuda --mode values --atol 1e-4

build/polygrad_parity_runner_cuda: $(SRC) $(CODEC_SRC) $(PARITY_RUNNER_SRC)
	@mkdir -p build
	$(CC) $(CFLAGS_RELEASE) -o $@ $(filter %.c,$^) -lm -ldl
else
bench-hlb-cuda-semantic bench-hlb-cuda-timing bench-hlb-cuda-manifest:
	@echo "This target requires HAS_CUDA=1" >&2
	@false
endif

ifeq ($(HAS_HIP), 1)
test-parity-hip: build/polygrad_parity_runner_hip
	CACHELEVEL=0 HIP_ARCH=gfx90a TINYGRAD_ROOT=tinygrad_latest \
		$(PARITY_PY) $(PARITY_SCRIPT) \
		--runner build/polygrad_parity_runner_hip --hip --mode values --atol 1e-4

build/polygrad_parity_runner_hip: $(SRC) $(CODEC_SRC) $(PARITY_RUNNER_SRC)
	@mkdir -p build
	$(CC) $(CFLAGS_RELEASE) -o $@ $(filter %.c,$^) -lm -ldl

bench-hip: build/bench_hip
	./build/bench_hip

build/bench_hip: bench/bench_hip.c $(SRC) $(CODEC_SRC)
	@mkdir -p build
	$(CC) $(CFLAGS_RELEASE) -o $@ $(filter %.c,$^) -lm -ldl
endif

BENCH_JSON = $(shell ls -t bench/results/2*.json 2>/dev/null | head -1)
BENCH_JAX_JS_ITERS ?= 40
BENCH_JAX_JS_WARMUP ?= 10
BENCH_JAX_JS_LARGE_ITERS ?= 3
BENCH_JAX_JS_MATMUL_SIZES ?= 64,128,256,512,1024,2048
BENCH_JAX_JS_MODEL_CASES ?= mlp_small,mlp_token,mlp_batch,qwen_ffn_token,qwen_ffn_batch
BENCH_JAX_JS_EXTRA ?=
BENCH_RATIO_LOCAL_BASELINE ?= bench/baselines/local/$(shell hostname -s)-ratios.json
BENCH_RATIO_BASELINE ?= $(BENCH_RATIO_LOCAL_BASELINE)

bench-ratios: build/libpolygrad.so wasm-pkg
	POLYGRAD_LIB=$(abspath build/libpolygrad.so) PYTHONPATH=py $(PARITY_PY) bench/bench_ratios.py
	$(NODE) bench/bench_ratios.js --json-file $$(ls -t bench/results/2*.json | head -1)

bench-parity: build/libpolygrad.so
	POLYGRAD_LIB=$(abspath build/libpolygrad.so) $(PARITY_PY) bench/bench_tinygrad_parity.py

.PHONY: test-bench-wasm
test-bench-wasm:
	$(NODE) --test bench/test_wasm_checks.mjs

bench-jax-js-wasm: wasm-pkg
	$(NODE) bench/bench_jax_js_wasm.mjs --iters $(BENCH_JAX_JS_ITERS) --warmup $(BENCH_JAX_JS_WARMUP) $(BENCH_JAX_JS_EXTRA)

bench-jax-js-matmul-wasm: wasm-pkg
	$(NODE) bench/bench_jax_js_matmul_wasm.mjs --sizes $(BENCH_JAX_JS_MATMUL_SIZES) --no-openblas --iters $(BENCH_JAX_JS_ITERS) --warmup $(BENCH_JAX_JS_WARMUP) --large-iters $(BENCH_JAX_JS_LARGE_ITERS) $(BENCH_JAX_JS_EXTRA)

bench-jax-js-model-wasm: wasm-pkg
	$(NODE) bench/bench_jax_js_model_wasm.mjs --cases $(BENCH_JAX_JS_MODEL_CASES) --iters $(BENCH_JAX_JS_ITERS) --warmup $(BENCH_JAX_JS_WARMUP) $(BENCH_JAX_JS_EXTRA)

bench-jax-js-browser-wasm: wasm-pkg
	cd js && bash scripts/build-browser.sh
	$(NODE) bench/bench_jax_js_browser_wasm.mjs

bench-jax-js-browser-matmul-wasm: wasm-pkg
	cd js && bash scripts/build-browser.sh
	$(NODE) bench/bench_jax_js_browser_matmul_wasm.mjs --sizes $(BENCH_JAX_JS_MATMUL_SIZES) --iters $(BENCH_JAX_JS_ITERS) --warmup $(BENCH_JAX_JS_WARMUP) --large-iters $(BENCH_JAX_JS_LARGE_ITERS) $(BENCH_JAX_JS_EXTRA)

bench-jax-js-browser-model-wasm: wasm-pkg
	cd js && bash scripts/build-browser.sh
	$(NODE) bench/bench_jax_js_browser_model_wasm.mjs --cases $(BENCH_JAX_JS_MODEL_CASES) --iters $(BENCH_JAX_JS_ITERS) --warmup $(BENCH_JAX_JS_WARMUP) $(BENCH_JAX_JS_EXTRA)

bench-compare:
	$(PYTHON) bench/bench_compare.py $(BENCH_RATIO_BASELINE) $$(ls -t bench/results/2*.json | head -1)

bench-compare-global:
	$(PYTHON) bench/bench_compare.py bench/results/baseline.json $$(ls -t bench/results/2*.json | head -1)

bench-regression: bench-ratios bench-compare

bench-ratio-local-baseline: bench-ratios
	@mkdir -p $$(dirname $(BENCH_RATIO_LOCAL_BASELINE))
	cp $$(ls -t bench/results/2*.json | head -1) $(BENCH_RATIO_LOCAL_BASELINE)
	@echo "Updated local ratio benchmark baseline: $(BENCH_RATIO_LOCAL_BASELINE)"

bench-update-local-ratio-baseline: bench-ratio-local-baseline

bench-update-baseline:
	cp $$(ls -t bench/results/2*.json | head -1) bench/results/baseline.json
	@echo "Updated global ratio benchmark baseline. Review before committing bench/results/baseline.json"

FUZZ_CC ?= clang
FUZZ_ARGS ?= -runs=256 -max_len=512 -timeout=5
FUZZ_SMOKE_ARGS ?= -runs=256 -max_len=512 -timeout=5
FUZZ_NIGHTLY_ARGS ?= -runs=8192 -max_len=2048 -timeout=10
FUZZ_WORK_DIR ?= temp/fuzz-corpus
FUZZ_SEED_DIR ?= test/corpus
FUZZ_ASAN_OPTIONS ?= symbolize=0:detect_leaks=1:protect_shadow_gap=0
FUZZ_RUN_ENV ?= ASAN_OPTIONS=$(FUZZ_ASAN_OPTIONS) UBSAN_OPTIONS=$(UBSAN_OPTIONS)
FUZZ_CFLAGS = -std=c11 -D_POSIX_C_SOURCE=200809L -g -O1 \
	-fsanitize=fuzzer,address,undefined -fno-omit-frame-pointer \
	-fno-pie -no-pie \
	-Wall -Wextra -Wpedantic -Wno-unused-parameter -Isrc

fuzz: fuzz-symbolic fuzz-symbolic-div

fuzz-smoke:
	$(MAKE) fuzz FUZZ_ARGS="$(FUZZ_SMOKE_ARGS)"

fuzz-nightly:
	$(MAKE) fuzz FUZZ_ARGS="$(FUZZ_NIGHTLY_ARGS)"

fuzz-symbolic: build/fuzz_sym
	@mkdir -p $(FUZZ_WORK_DIR)/symbolic
	$(FUZZ_RUN_ENV) ./build/fuzz_sym $(FUZZ_WORK_DIR)/symbolic $(FUZZ_SEED_DIR)/symbolic $(FUZZ_ARGS)

build/fuzz_sym: test/fuzz_sym.c $(SRC) $(CODEC_SRC)
	@mkdir -p build
	$(FUZZ_CC) $(FUZZ_CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS_DEBUG)

fuzz-symbolic-div: build/fuzz_sym_div
	@mkdir -p $(FUZZ_WORK_DIR)/symbolic-div
	$(FUZZ_RUN_ENV) ./build/fuzz_sym_div $(FUZZ_WORK_DIR)/symbolic-div $(FUZZ_SEED_DIR)/symbolic-div $(FUZZ_ARGS)

build/fuzz_sym_div: test/fuzz_sym_div.c $(SRC) $(CODEC_SRC)
	@mkdir -p build
	$(FUZZ_CC) $(FUZZ_CFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS_DEBUG)

NODE ?= $(shell which node 2>/dev/null || echo node)
test-wasm: test-js-browser

sync-source-mirrors:
	$(PYTHON) py/scripts/sync-csrc.py
	$(NODE) js/scripts/sync-csrc.js

verify-source-mirrors:
	$(PYTHON) scripts/verify-source-mirrors.py

test-py: verify-source-mirrors build/libpolygrad.so
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 POLYGRAD_LIB=build/libpolygrad.so PYTHONPATH=py python -m pytest py/tests/ -v

test-py-x86: verify-source-mirrors build/libpolygrad.so
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 POLY_DEVICE=x86 POLYGRAD_LIB=build/libpolygrad.so PYTHONPATH=py python -m pytest py/tests/test_tensor.py py/tests/test_nn.py py/tests/test_model.py py/tests/test_hf.py py/tests/test_hf_e2e.py -v

test-js: test-js-wasm test-js-native test-js-package

test-js-wasm: verify-source-mirrors wasm-pkg
	$(NODE) js/test/test_wasm.js
	$(NODE) --expose-gc js/test/test_gc.js wasm

test-js-native: verify-source-mirrors js/build/Release/polygrad_napi.node
	$(NODE) js/test/test_native.js
	$(NODE) --expose-gc js/test/test_gc.js native

test-js-package: verify-source-mirrors wasm-pkg
	cd js && bash scripts/build-browser.sh && $(NODE) test/test_package_exports.js

test-model-interchange: verify-source-mirrors build/libpolygrad.so js/build/Release/polygrad_napi.node wasm-pkg
	PYTHONPATH=py POLYGRAD_LIB=$(abspath build/libpolygrad.so) \
		python test/test_model_interchange.py --cores native,wasm

js/build/Release/polygrad_napi.node: build/libpolygrad.a js/binding.gyp js/napi_api.c
	cd js && npm run build:native

test-js-native-cpu: verify-source-mirrors js/build/Release/polygrad_napi.node
	POLY_DEVICE=cpu $(NODE) js/test/test_native.js

test-js-native-x86: verify-source-mirrors js/build/Release/polygrad_napi.node
	POLY_DEVICE=x86 $(NODE) js/test/test_native.js

test-js-native-interp: verify-source-mirrors js/build/Release/polygrad_napi.node
	POLY_DEVICE=interp $(NODE) js/test/test_native.js

ifeq ($(HAS_CUDA), 1)
test-js-native-cuda: verify-source-mirrors js/build/Release/polygrad_napi.node
	POLY_DEVICE=cuda $(NODE) js/test/test_native.js
endif

ifeq ($(HAS_HIP), 1)
test-js-native-hip: verify-source-mirrors js/build/Release/polygrad_napi.node
	POLY_DEVICE=hip $(NODE) js/test/test_native.js
endif

# Browser/Playwright coverage should be invoked through Make so the wasm
# package and browser bundle are rebuilt in the right order.
test-browser: test-js-browser

.PHONY: test-browser-runner
test-browser-runner:
	$(NODE) js/test/test_browser_runner.js

test-js-browser: test-browser-runner verify-source-mirrors wasm-pkg
	cd js && bash scripts/build-browser.sh && $(NODE) test/browser/run.js

test-browser-matrix: test-js-browser-matrix

test-js-browser-matrix: verify-source-mirrors wasm-pkg
	cd js && bash scripts/build-browser.sh && \
		POLY_BROWSER_BROWSERS="$(BROWSER_MATRIX)" \
		POLY_BROWSER_DEVICES="$(BROWSER_MATRIX_DEVICES)" \
		POLY_BROWSER_SKIP_UNAVAILABLE=1 \
		$(NODE) test/browser/run.js

test-browser-qwen3: verify-source-mirrors wasm-pkg require-qwen3-gguf
	@mkdir -p temp/chrome_tmp
	cd js && bash scripts/build-browser.sh
	TMPDIR=$(abspath temp/chrome_tmp) POLY_QWEN3_GGUF="$(abspath $(QWEN3_GGUF))" \
		$(NODE) js/test/browser/qwen_webgpu.js

test-js-legacy: build/libpolygrad.so
	$(NODE) js_legacy/test/test_tensor.js

test-wasm-legacy: build/polygrad.js build/polygrad.wasm
	$(NODE) js_legacy/polygrad/test/test_smoke.js

test-native-legacy: build/libpolygrad.so
	$(NODE) js_legacy/polygrad-node/test/test_native_smoke.js

test-browser-legacy: wasm-pkg
	$(NODE) js_legacy/polygrad/test/browser/test_browser.js

# Full cross-backend test suite:
#   C:      cpu (default), x86, interp, cuda*, hip*
#   JS:     wasm core + wasm backend
#           native core + cpu/x86/interp/cuda*/hip* backends
#   Python: py/tests/
#   * only when hardware is available
TEST_ALL_DEPS = test test-x86 test-interp test-js-wasm test-js-package test-js-native-cpu test-js-native-x86 test-js-native-interp test-py test-model-interchange
ifeq ($(HAS_CUDA), 1)
  TEST_ALL_DEPS += test-cuda test-js-native-cuda
endif
ifeq ($(HAS_HIP), 1)
  TEST_ALL_DEPS += test-hip test-js-native-hip
endif
test-all: $(TEST_ALL_DEPS)
	@echo ""
	@echo "=== test-all: all backends passed ==="

wasm: build/polygrad.js build/polygrad.wasm

build/polygrad.js build/polygrad.wasm: $(WASM_SRC) Makefile
	@mkdir -p build
		EMSDK_PYTHON=$(EMSDK_PYTHON) $(EMCC) $(EMCC_CFLAGS_COMMON) \
		-s WASM=1 -s MODULARIZE=1 -s EXPORT_NAME=PolygradModule \
		$(WASM_ASYNCIFY_FLAGS) \
		-s EXPORTED_FUNCTIONS='[$(WASM_EXPORTS)]' \
		-s "EXPORTED_RUNTIME_METHODS=['ccall','cwrap','getValue','setValue','HEAPU8','HEAP32','HEAPF32','HEAPF64','UTF8ToString','addFunction']" \
		-s ALLOW_TABLE_GROWTH=1 \
		-s ALLOW_MEMORY_GROWTH=1 -s MAXIMUM_MEMORY=4GB \
		-s WASM_BIGINT \
		-o build/polygrad.js $(WASM_SRC)

wasm-pkg: build/core.async.js build/core.sync.js
	@mkdir -p js/wasm
	cp build/core.async.js js/wasm/core.async.js
	cp build/core.sync.js js/wasm/core.sync.js

build/core.async.js: $(WASM_SRC) Makefile
	@mkdir -p build
		EMSDK_PYTHON=$(EMSDK_PYTHON) $(EMCC) $(EMCC_CFLAGS_COMMON) \
		-s WASM=1 -s MODULARIZE=1 -s EXPORT_NAME=createPolygrad \
		$(WASM_ASYNCIFY_FLAGS) \
		-s EXPORTED_FUNCTIONS='[$(WASM_EXPORTS)]' \
		-s "EXPORTED_RUNTIME_METHODS=['ccall','cwrap','getValue','setValue','HEAPU8','HEAP32','HEAPF32','HEAPF64','UTF8ToString','addFunction']" \
		-s ALLOW_TABLE_GROWTH=1 \
		-s ALLOW_MEMORY_GROWTH=1 -s MAXIMUM_MEMORY=4GB \
		-s WASM_BIGINT \
		-s SINGLE_FILE=1 \
		-s SINGLE_FILE_BINARY_ENCODE=0 \
		-s ENVIRONMENT='web,node' \
		-o build/core.async.js $(WASM_SRC)

build/core.sync.js: $(WASM_SRC) Makefile js/scripts/wasm-sync-post.js
	@mkdir -p build
		EMSDK_PYTHON=$(EMSDK_PYTHON) $(EMCC) $(EMCC_CFLAGS_COMMON) \
		-s WASM=1 \
		-s WASM_ASYNC_COMPILATION=0 \
		$(WASM_ASYNCIFY_FLAGS) \
		-s EXPORTED_FUNCTIONS='[$(WASM_EXPORTS)]' \
		-s "EXPORTED_RUNTIME_METHODS=['ccall','cwrap','getValue','setValue','HEAPU8','HEAP32','HEAPF32','HEAPF64','UTF8ToString','addFunction']" \
		-s ALLOW_TABLE_GROWTH=1 \
		-s ALLOW_MEMORY_GROWTH=1 -s MAXIMUM_MEMORY=4GB \
		-s WASM_BIGINT \
		-s SINGLE_FILE=1 \
		-s SINGLE_FILE_BINARY_ENCODE=0 \
		-s ENVIRONMENT='web,node' \
		--post-js js/scripts/wasm-sync-post.js \
		-o build/core.sync.js $(WASM_SRC)

VENDOR_SRC = vendor/dht/dht.c vendor/stun/STUNExternalIP.c
P2P_SRC = src/p2p.c

bench-train-py:
	POLYGRAD_LIB=$(abspath build/libpolygrad.so) PYTHONPATH=py python bench/train_mlp_python.py

# ── Package builds / publishing ─────────────────────────────────────

build-py: build-py-sdist

build-py-sdist: verify-source-mirrors
	cd py && rm -rf csrc dist build *.egg-info && \
		$(PYTHON) scripts/sync-csrc.py && \
		$(PYTHON) -m build --sdist

build-py-wheel: verify-source-mirrors
	cd py && rm -rf csrc dist build *.egg-info && \
		$(PYTHON) scripts/sync-csrc.py && \
		$(PYTHON) -m build --wheel

build-python: build-py

test-py-sdist-install: build-py-sdist
	rm -rf temp/py-sdist-smoke && mkdir -p temp/py-sdist-smoke/run temp/py-sdist-smoke/tmp && \
		$(PYTHON) -m venv --system-site-packages temp/py-sdist-smoke/venv && \
		. temp/py-sdist-smoke/venv/bin/activate && \
		TMPDIR=$(abspath temp/py-sdist-smoke/tmp) python -m pip install --no-cache-dir --no-deps py/dist/polygrad-*.tar.gz && \
		cd temp/py-sdist-smoke/run && \
		PYTHONPATH= TMPDIR=$(abspath temp/py-sdist-smoke/tmp) python -c "from extra.bench_log import BenchEvent; from polygrad import Tensor; print(BenchEvent.STEP.value, ((Tensor([1,2,3])*2+1).numpy()).tolist())"

publish-py: build-py-sdist
	cd py && $(TWINE) upload dist/*.tar.gz

publish-python: publish-py

build-js: verify-source-mirrors wasm-pkg
	cd js && $(NODE) scripts/sync-csrc.js && $(NPM) run build:browser

publish-js: build-js
	cd js && $(NPM) publish

# ── P2P distributed training ───────────────────────────────────────

test-p2p: build/test_p2p
	./build/test_p2p

build/test_p2p: $(SRC) $(P2P_SRC) $(VENDOR_SRC) test/test_p2p.c
	@mkdir -p build
	$(CC) $(CFLAGS_DEBUG) -Ivendor/dht -Ivendor/stun -o $@ $(filter %.c,$^) $(LDFLAGS_DEBUG)

test-p2p-browser:
	$(NODE) browser/test/test_p2p.js

# ── Coverage ──────────────────────────────────────────────────────

GCOV ?= gcov
COVERAGE_FILTER ?= --fast
COVERAGE_OBJS = $(patsubst %.c,build/obj/coverage/%.o,$(SRC) $(CODEC_SRC) $(TEST_SRC))

coverage: build/polygrad_test_cov
	./build/polygrad_test_cov $(COVERAGE_FILTER)
	@set -e; for f in $(SRC) $(CODEC_SRC); do \
		$(GCOV) -n -b -c -o "build/obj/coverage/$${f%.c}.gcno" "$$f"; \
	done

# Preserve source directories: src/ops.c and src/uop/ops.c must not share
# profile counters. Use the same feature/include flags as the native tests.
build/obj/coverage/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS_COMMON) -DPOLY_TESTING -g -O0 --coverage -MMD -MP -c $< -o $@

build/polygrad_test_cov: $(COVERAGE_OBJS)
	@mkdir -p build
	$(CC) --coverage -o $@ $(filter %.o,$^) -lm -ldl

-include $(COVERAGE_OBJS:.o=.d)

clean:
	rm -rf build/ *.o

# ── Safety tooling ──────────────────────────────────────────────────

# Clang Static Analyzer (requires clang)
analyze:
	@mkdir -p build
	@rm -f build/analyze.log; status=0; \
	  for src in $(ANALYZE_SRC); do \
	    echo "==> $$src" >> build/analyze.log; \
	    clang --analyze $(filter-out -pipe,$(CFLAGS_COMMON)) $(ANALYZE_FLAGS) "$$src" \
	      >> build/analyze.log 2>&1 || status=1; \
	  done; \
	  cat build/analyze.log; \
	  if grep -Eq '(^|: )(warning|error):' build/analyze.log; then status=1; fi; \
	  rm -f *.plist; \
	  exit $$status
	@echo "Analysis complete. See build/analyze.log"

# Cppcheck (install: apt install cppcheck)
cppcheck:
	cppcheck --enable=warning,performance,portability --std=c11 \
		--suppress=missingIncludeSystem --error-exitcode=1 $(SRC)

# clang-format (install: apt install clang-format)
format:
	clang-format -i $(FORMAT_SRC)

format-check:
	clang-format --dry-run --Werror $(FORMAT_SRC)

# MemorySanitizer (requires clang, incompatible with ASan)
test-msan: build/polygrad_test_msan
	./build/polygrad_test_msan

build/polygrad_test_msan: $(SRC) $(CODEC_SRC) $(TEST_SRC)
	@mkdir -p build
	clang -std=c11 -g -O1 -fsanitize=memory -fno-omit-frame-pointer \
		-o $@ $(filter %.c,$^) -lm -ldl -fsanitize=memory

# ThreadSanitizer focused smoke. The current threading contract permits
# independent contexts on separate threads; one PolyCtx remains thread-confined.
test-tsan: build/polygrad_test_tsan
	TSAN_OPTIONS=$(TSAN_OPTIONS) $(TSAN_RUNNER) ./build/polygrad_test_tsan threading

build/polygrad_test_tsan: $(SRC) $(CODEC_SRC) $(TEST_SRC)
	@mkdir -p build
	$(TSAN_CC) $(CFLAGS_COMMON) -g -O1 -fsanitize=thread -fno-omit-frame-pointer \
		-o $@ $(filter %.c,$^) -lm -ldl -pthread -fsanitize=thread

# ── Full verification ──────────────────────────────────────────────────

verify: test test-harness-skip-accounting test-parity format-check analyze fuzz-smoke
	@echo "All verification checks passed."
