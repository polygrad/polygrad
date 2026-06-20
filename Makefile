CC ?= gcc
CFLAGS_COMMON = -std=c11 -D_POSIX_C_SOURCE=200809L -Wall -Wextra -Wpedantic -Wno-unused-parameter -Isrc
CFLAGS_RELEASE = $(CFLAGS_COMMON) -O2
CFLAGS_DEBUG = $(CFLAGS_COMMON) -g -O0 -fsanitize=address,undefined -fno-omit-frame-pointer
LDFLAGS = -lm
LDFLAGS_DEBUG = -lm -ldl -fsanitize=address,undefined
# Keep LeakSanitizer enabled by default for the native debug test binary.
# Driver/runtime targets can still opt out when investigating external runtime
# leaks:
#   make test-cuda ASAN_OPTIONS=detect_leaks=0,protect_shadow_gap=0
ASAN_OPTIONS ?= detect_leaks=1,protect_shadow_gap=0
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

SRC = src/ops.c src/dtype.c src/arena.c src/hashmap.c src/utils.c src/ctx.c src/device.c src/placer.c src/engine/realize.c src/uop.c src/pat.c src/alu.c src/sym.c src/shape.c src/autograd.c src/codegen.c src/render_c.c src/render_wgsl.c src/runtime_cpu.c src/runtime_wasm.c src/runtime_webgpu.c src/wasm_builder.c src/render_wasm.c src/frontend.c src/tensor.c src/optim.c src/schedule/rangeify.c src/simplify.c src/schedule/indexing.c src/nn.c src/engine/schedule.c src/interp.c
FILC_SRC = src/ops.c src/dtype.c src/arena.c src/hashmap.c src/utils.c src/ctx.c src/device.c src/placer.c src/engine/realize.c src/uop.c src/pat.c src/alu.c src/sym.c src/shape.c src/autograd.c src/codegen.c src/render_c.c src/render_wgsl.c src/runtime_cpu.c src/runtime_wasm.c src/runtime_webgpu.c src/wasm_builder.c src/render_wasm.c src/frontend.c src/tensor.c src/optim.c src/schedule/rangeify.c src/simplify.c src/schedule/indexing.c src/nn.c src/engine/schedule.c src/interp.c
LOADER_SRC = src/loaders/decoded.c src/loaders/import_error.c src/loaders/bind.c src/loaders/hf_decode.c src/loaders/gguf_decode.c src/loaders/gguf_loader.c src/loaders/import_desc.c
CODEC_SRC = vendor/cjson/cJSON.c src/safetensors.c src/wlrn.c src/ir.c src/bundle.c src/instance.c src/tokenizer.c src/models/mlp.c src/models/tabm.c src/models/nam.c src/models/registry.c src/models/gpt2.c src/models/qwen3.c src/models/hf_loader.c $(LOADER_SRC)
TEST_SRC = test/test_main.c test/test_uop.c test/test_utils.c test/test_dtype.c test/test_pat.c test/test_sym.c test/test_shape.c test/test_schedule_engine.c test/test_autograd.c test/test_codegen.c test/test_wasm.c test/test_rangeify.c test/test_reduce_simplify.c test/test_nn.c test/test_tensor.c test/test_future_passes.c test/test_safetensors.c test/test_wlrn.c test/test_ir.c test/test_instance.c test/test_mlp.c test/test_tabm.c test/test_nam.c test/test_hf.c test/test_f16.c test/test_schedule_runtime.c test/test_bundle.c test/test_registry.c test/test_realize.c

ifeq ($(HAS_CUDA), 1)
  SRC += src/render_cuda.c src/runtime_cuda.c
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
  SRC += src/render_hip.c src/runtime_hip.c
  TEST_SRC += test/test_hip.c
  CFLAGS_COMMON += -DPOLY_HAS_HIP=1
endif

# Detect x86-64 (not available in Emscripten WASM builds)
HAS_X64 := $(shell uname -m | grep -c x86_64)
ifeq ($(HAS_X64), 1)
  SRC += src/render_x64.c
  TEST_SRC += test/test_x64.c
  CFLAGS_COMMON += -DPOLY_HAS_X64=1
endif
PARITY_RUNNER_SRC = test/test_tinygrad_runner.c
PARITY_SCRIPT = test/test_tinygrad_parity.py
PARITY_PY ?= conda run -n tiny python

# Emscripten WASM build (excludes runtime_cpu.c — no fork/dlopen in WASM)
WASM_SRC = src/ops.c src/dtype.c src/arena.c src/hashmap.c src/utils.c src/ctx.c src/device.c src/placer.c src/engine/realize.c src/uop.c src/pat.c src/alu.c src/sym.c src/shape.c src/autograd.c src/codegen.c src/render_c.c src/render_wgsl.c src/runtime_wasm.c src/runtime_webgpu.c src/wasm_builder.c src/render_wasm.c src/frontend.c src/tensor.c src/optim.c src/schedule/rangeify.c src/simplify.c src/schedule/indexing.c src/nn.c src/engine/schedule.c src/interp.c vendor/cjson/cJSON.c src/safetensors.c src/ir.c src/bundle.c src/instance.c src/tokenizer.c src/models/mlp.c src/models/tabm.c src/models/nam.c src/models/registry.c src/models/gpt2.c src/models/qwen3.c src/models/hf_loader.c $(LOADER_SRC)
WASM_EXPORTS = _poly_ctx_new,_poly_ctx_destroy,_poly_ctx_set_preferred_device,_poly_ctx_set_frontend_buffer_release,_poly_ctx_named_count,_poly_op_count,_poly_op_name,_poly_const_float,_poly_const_double,_poly_const_int,_poly_contiguous,_poly_add,_poly_sub,_poly_mul,_poly_div,_poly_alu1,_poly_alu2,_poly_alu3,_poly_store_val,_poly_sink1,_poly_sink_n,_poly_buffer_by_id,_poly_buffer_f32,_poly_buffer_f64,_poly_buffer_from_host,_poly_buffer_get_ptr,_poly_buffer_get_key,_poly_buffer_read,_poly_set_frontend_buffer_release,_poly_realize_uops,_poly_tensor_create,_poly_tensor_create_with_roots,_poly_tensor_update,_poly_tensor_to_device,_poly_tensor_assign,_poly_tensor_uop,_poly_tensor_uop_logical,_poly_tensor_uop_physical,_poly_tensor_device,_poly_realize_tensors,_poly_uop_has_buffer_identity,_poly_uop_get_buffer_identity,_poly_uop_reachable,_poly_uop_substitute,_poly_reshape,_poly_expand,_poly_reduce_axis,_poly_permute,_poly_shrink,_poly_flip,_poly_pad,_poly_grad,_poly_grad_many,_poly_abi_version,_poly_device_by_name,_poly_device_name,_poly_dtype_id_by_name,_poly_uop_dtype_id,_poly_exp,_poly_log,_poly_log1p,_poly_expm1,_poly_sin,_poly_cos,_poly_tan,_poly_erf,_poly_erfc,_poly_erfinv,_poly_ndtri,_poly_digamma,_poly_lgamma,_poly_sigmoid,_poly_tanh_act,_poly_relu,_poly_relu6,_poly_leaky_relu,_poly_gelu,_poly_quick_gelu,_poly_silu,_poly_elu,_poly_softplus,_poly_mish,_poly_hardtanh,_poly_hardswish,_poly_hardsigmoid,_poly_abs,_poly_sign,_poly_square,_poly_rsqrt,_poly_ceil,_poly_floor,_poly_round_f,_poly_isinf,_poly_isnan,_poly_eq,_poly_ne,_poly_gt,_poly_ge,_poly_le,_poly_where_op,_poly_maximum,_poly_minimum,_poly_clamp,_poly_detach,_poly_cast_by_id,_poly_rand,_poly_randn,_poly_arange,_poly_eye,_poly_linspace,_poly_full,_poly_tril,_poly_triu,_poly_cholesky,_poly_triangular_solve,_poly_sum_reduce,_poly_max_reduce,_poly_mean_reduce,_poly_logsumexp,_poly_dot,_poly_cross_entropy,_poly_einsum,_poly_rearrange,_exp2f,_log2f,_sinf,_powf,_malloc,_free,_poly_instance_from_ir,_poly_instance_free,_poly_instance_set_device,_poly_instance_call,_poly_instance_value_and_grad,_poly_instance_forward,_poly_instance_train_step,_poly_instance_set_optimizer,_poly_instance_param_count,_poly_instance_param_name,_poly_instance_param_data,_poly_instance_param_shape,_poly_instance_buf_count,_poly_instance_buf_name,_poly_instance_buf_role,_poly_instance_buf_data,_poly_instance_buf_shape,_poly_instance_export_weights,_poly_instance_import_weights,_poly_instance_export_ir,_poly_mlp_from_json,_poly_tabm_instance,_poly_nam_instance,_poly_instance_save_bundle,_poly_instance_from_bundle,_poly_uop_ndim,_poly_uop_dims,_poly_ctx_arena,_poly_ctx_shape_cache,_poly_softmax,_poly_log_softmax,_poly_dot,_poly_cross_entropy,_poly_gather,_poly_sum_reduce,_poly_max_reduce,_poly_mean_reduce,_poly_var_reduce,_poly_tril,_poly_triu,_poly_rmsnorm_apply,_poly_sdpa,_poly_rope,_poly_repeat_interleave,_poly_argmax,_poly_mse_loss,_poly_mae_loss,_poly_hf_load,_poly_gguf_load,_poly_gguf_decode,_poly_gguf_decoded_free,_poly_gguf_kv_int,_poly_gguf_kv_float,_poly_gguf_kv_string,_poly_import_last_error_code,_poly_import_last_error_message,_poly_tokenizer_from_gguf,_poly_tokenizer_from_json,_poly_tokenize,_poly_detokenize,_poly_tokenizer_free,_poly_tokenizer_vocab_size,_poly_tokenizer_bos_id,_poly_tokenizer_eos_id,_poly_gpt2,_poly_qwen3

WASM_EXPORTS := $(WASM_EXPORTS),_poly_tensor_requires_grad,_poly_tensor_set_requires_grad,_poly_instance_param_trainable,_poly_instance_set_param_trainable,_poly_instance_buf_trainable,_poly_instance_set_buf_trainable,_poly_instance_readback_param,_poly_instance_readback_buf,_poly_optim_build_step,_poly_register_buffer_by_id,_poly_register_existing_buffer,_poly_instance_from_sinks,_poly_instance_from_binding_arrays

WASM_ASYNCIFY_IMPORTS = ['js_webgpu_dispatch','js_webgpu_read_buffer_to_wasm','js_webgpu_read_buffer_to_hostkey']
WASM_ASYNCIFY_ONLY = ['poly_instance_call','poly_instance_forward','poly_instance_value_and_grad','poly_instance_train_step','run_instance_sink','poly_instance_param_data','poly_instance_buf_data','poly_instance_export_weights','poly_instance_save_bundle','poly_instance_readback_param','poly_instance_readback_buf','poly_realize_sink','poly_realize_uops','poly_realize_tensors','poly_run_schedule','poly_webgpu_execute','copy_execute_fn','poly_buffer_copy','poly_buffer_read','host_copy_in','webgpu_copy_out','sync_buf_to_host','readback_handle']
WASM_ASYNCIFY_FLAGS = -s ASYNCIFY=1 \
	-s "ASYNCIFY_IMPORTS=$(WASM_ASYNCIFY_IMPORTS)" \
	-s "ASYNCIFY_ONLY=$(WASM_ASYNCIFY_ONLY)"

.PHONY: all test test-fast test-cuda test-hip test-interp test-x64 test-cuda-only test-hip-only test-parity test-parity-opt test-parity-ir test-parity-ir-opt test-parity-cuda test-parity-hip test-wasm test-wasm-new test-native test-browser test-p2p test-p2p-browser bench bench-cuda bench-hip bench-train-py bench-ratios bench-parity bench-compare bench-regression bench-update-baseline fuzz fuzz-smoke fuzz-nightly fuzz-symbolic fuzz-symbolic-div wasm wasm-pkg build-py build-py-sdist build-py-wheel build-python publish-py publish-python build-js publish-js clean analyze cppcheck format format-check test-msan verify coverage test-full test-js-native-cpu test-js-native-x64 test-js-native-interp test-js-native-cuda test-js-native-hip test-filc-interp-fast

all: build/libpolygrad.a build/libpolygrad.so

build/libpolygrad.a: $(SRC) $(CODEC_SRC)
	@mkdir -p build
	$(CC) $(CFLAGS_RELEASE) -c $(SRC) $(CODEC_SRC)
	ar rcs $@ *.o
	@rm -f *.o

build/libpolygrad.so: $(SRC) $(CODEC_SRC)
	@mkdir -p build
	$(CC) $(CFLAGS_RELEASE) -fPIC -shared -o $@ $^ -lm -ldl

test: build/polygrad_test
	$(SAN_RUN) ./build/polygrad_test

test-fast: build/polygrad_test
	$(SAN_RUN) ./build/polygrad_test --fast

# Full suite routed through specific backend (POLY_DEVICE selector)
test-cuda: build/polygrad_test
	$(SAN_RUN) POLY_DEVICE=cuda ./build/polygrad_test

test-hip: build/polygrad_test
	$(SAN_RUN) POLY_DEVICE=hip ./build/polygrad_test

test-interp: build/polygrad_test
	$(SAN_RUN) POLY_DEVICE=interp ./build/polygrad_test

test-x64: build/polygrad_test
	$(SAN_RUN) POLY_DEVICE=x64 ./build/polygrad_test

# Backend-specific tests only (uses substring filter)
test-cuda-only: build/polygrad_test
	$(SAN_RUN) ./build/polygrad_test cuda

test-hip-only: build/polygrad_test
	$(SAN_RUN) ./build/polygrad_test hip

test-parity: build/polygrad_parity_runner
	$(SAN_RUN) CACHELEVEL=0 $(PARITY_PY) $(PARITY_SCRIPT) --runner build/polygrad_parity_runner --mode values

test-parity-opt: build/polygrad_parity_runner
	$(SAN_RUN) CACHELEVEL=0 POLY_OPTIMIZE=1 $(PARITY_PY) $(PARITY_SCRIPT) --runner build/polygrad_parity_runner --mode values

test-parity-ir: build/polygrad_parity_runner
	$(SAN_RUN) CACHELEVEL=0 $(PARITY_PY) $(PARITY_SCRIPT) --runner build/polygrad_parity_runner --mode full --no-opt

test-parity-ir-opt: build/polygrad_parity_runner
	$(SAN_RUN) CACHELEVEL=0 POLY_OPTIMIZE=1 POLY_DEVECTORIZE=0 $(PARITY_PY) $(PARITY_SCRIPT) --runner build/polygrad_parity_runner --mode full

build/polygrad_test: $(SRC) $(CODEC_SRC) $(TEST_SRC)
	@mkdir -p build
	$(CC) $(CFLAGS_DEBUG) -o $@ $^ $(LDFLAGS_DEBUG)

build/polygrad_test_filc: $(FILC_SRC) $(CODEC_SRC) $(TEST_SRC)
	@mkdir -p build
	$(FILC) $(FILC_CFLAGS_DEBUG) -o $@ $^ -lm -ldl

test-filc-interp-fast: build/polygrad_test_filc
	POLY_DEVICE=interp CC=$(FILC) ./build/polygrad_test_filc --fast

build/polygrad_parity_runner: $(SRC) $(CODEC_SRC) $(PARITY_RUNNER_SRC)
	@mkdir -p build
	$(CC) $(CFLAGS_RELEASE) -o $@ $^ $(LDFLAGS)

bench: build/bench_polygrad
	./build/bench_polygrad

build/bench_polygrad: $(SRC) $(CODEC_SRC) bench/bench_polygrad.c
	@mkdir -p build
	$(CC) $(CFLAGS_RELEASE) -o $@ $^ -lm -ldl

ifeq ($(HAS_CUDA), 1)
bench-cuda: build/bench_cuda
	./build/bench_cuda

build/bench_cuda: $(SRC) $(CODEC_SRC) bench/bench_cuda.c
	@mkdir -p build
	$(CC) $(CFLAGS_RELEASE) -o $@ $^ -lm -ldl

test-parity-cuda: build/polygrad_parity_runner_cuda
	CACHELEVEL=0 $(PARITY_PY) $(PARITY_SCRIPT) \
		--runner build/polygrad_parity_runner_cuda --cuda --mode values --atol 1e-4

build/polygrad_parity_runner_cuda: $(SRC) $(CODEC_SRC) $(PARITY_RUNNER_SRC)
	@mkdir -p build
	$(CC) $(CFLAGS_RELEASE) -o $@ $^ -lm -ldl
endif

ifeq ($(HAS_HIP), 1)
test-parity-hip: build/polygrad_parity_runner_hip
	CACHELEVEL=0 HIP_ARCH=gfx90a TINYGRAD_ROOT=tinygrad_latest \
		$(PARITY_PY) $(PARITY_SCRIPT) \
		--runner build/polygrad_parity_runner_hip --hip --mode values --atol 1e-4

build/polygrad_parity_runner_hip: $(SRC) $(CODEC_SRC) $(PARITY_RUNNER_SRC)
	@mkdir -p build
	$(CC) $(CFLAGS_RELEASE) -o $@ $^ -lm -ldl

bench-hip: build/bench_hip
	./build/bench_hip

build/bench_hip: bench/bench_hip.c $(SRC) $(CODEC_SRC)
	@mkdir -p build
	$(CC) $(CFLAGS_RELEASE) -o $@ $^ -lm -ldl
endif

BENCH_JSON = $(shell ls -t bench/results/2*.json 2>/dev/null | head -1)

bench-ratios: build/libpolygrad.so wasm-pkg
	POLYGRAD_LIB=$(abspath build/libpolygrad.so) PYTHONPATH=py python bench/bench_ratios.py
	$(NODE) bench/bench_ratios.js --json-file $$(ls -t bench/results/2*.json | head -1)

bench-parity: build/libpolygrad.so
	POLYGRAD_LIB=$(abspath build/libpolygrad.so) $(PARITY_PY) bench/bench_tinygrad_parity.py

bench-compare:
	python bench/bench_compare.py bench/results/baseline.json $$(ls -t bench/results/2*.json | head -1)

bench-regression: bench-ratios bench-compare

bench-update-baseline:
	cp $$(ls -t bench/results/2*.json | head -1) bench/results/baseline.json
	@echo "Updated. Review and commit bench/results/baseline.json"

FUZZ_CC ?= clang
FUZZ_ARGS ?= -runs=256 -max_len=512 -timeout=5
FUZZ_SMOKE_ARGS ?= -runs=256 -max_len=512 -timeout=5
FUZZ_NIGHTLY_ARGS ?= -runs=8192 -max_len=2048 -timeout=10
FUZZ_WORK_DIR ?= temp/fuzz-corpus
FUZZ_SEED_DIR ?= test/corpus
FUZZ_ASAN_OPTIONS ?= symbolize=0
FUZZ_RUN_ENV ?= ASAN_OPTIONS=$(FUZZ_ASAN_OPTIONS) UBSAN_OPTIONS=$(UBSAN_OPTIONS)
FUZZ_CFLAGS = -std=c11 -D_POSIX_C_SOURCE=200809L -g -O1 \
	-fsanitize=fuzzer,address,undefined -fno-omit-frame-pointer \
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
	$(FUZZ_CC) $(FUZZ_CFLAGS) -o $@ $^ $(LDFLAGS_DEBUG)

fuzz-symbolic-div: build/fuzz_sym_div
	@mkdir -p $(FUZZ_WORK_DIR)/symbolic-div
	$(FUZZ_RUN_ENV) ./build/fuzz_sym_div $(FUZZ_WORK_DIR)/symbolic-div $(FUZZ_SEED_DIR)/symbolic-div $(FUZZ_ARGS)

build/fuzz_sym_div: test/fuzz_sym_div.c $(SRC) $(CODEC_SRC)
	@mkdir -p build
	$(FUZZ_CC) $(FUZZ_CFLAGS) -o $@ $^ $(LDFLAGS_DEBUG)

NODE ?= $(shell which node 2>/dev/null || echo node)
test-wasm: test-js-browser

test-py: build/libpolygrad.so
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 POLYGRAD_LIB=build/libpolygrad.so PYTHONPATH=py python -m pytest py/tests/ -v

test-js: test-js-wasm test-js-native

test-js-wasm: wasm-pkg
	$(NODE) js/test/test_wasm.js

test-js-native: js/build/Release/polygrad_napi.node
	$(NODE) js/test/test_native.js

js/build/Release/polygrad_napi.node: build/libpolygrad.a
	cd js && npm run build:native

test-js-native-cpu: js/build/Release/polygrad_napi.node
	POLY_DEVICE=cpu $(NODE) js/test/test_native.js

test-js-native-x64: js/build/Release/polygrad_napi.node
	POLY_DEVICE=x64 $(NODE) js/test/test_native.js

test-js-native-interp: js/build/Release/polygrad_napi.node
	POLY_DEVICE=interp $(NODE) js/test/test_native.js

ifeq ($(HAS_CUDA), 1)
test-js-native-cuda: js/build/Release/polygrad_napi.node
	POLY_DEVICE=cuda $(NODE) js/test/test_native.js
endif

ifeq ($(HAS_HIP), 1)
test-js-native-hip: js/build/Release/polygrad_napi.node
	POLY_DEVICE=hip $(NODE) js/test/test_native.js
endif

# Browser/Playwright coverage should be invoked through Make so the wasm
# package and browser bundle are rebuilt in the right order.
test-browser: test-js-browser

test-js-browser: wasm-pkg
	cd js && bash scripts/build-browser.sh && $(NODE) test/browser/run.js

test-js-legacy: build/libpolygrad.so
	$(NODE) js_legacy/test/test_tensor.js

test-wasm-legacy: build/polygrad.js build/polygrad.wasm
	$(NODE) js_legacy/polygrad/test/test_smoke.js

test-native-legacy: build/libpolygrad.so
	$(NODE) js_legacy/polygrad-node/test/test_native_smoke.js

test-browser-legacy: wasm-pkg
	$(NODE) js_legacy/polygrad/test/browser/test_browser.js

# Full cross-backend test suite:
#   C:      cpu (default), x64, interp, cuda*, hip*
#   JS:     wasm core + wasm backend
#           native core + cpu/x64/interp/cuda*/hip* backends
#   Python: py/tests/
#   * only when hardware is available
TEST_ALL_DEPS = test test-x64 test-interp test-js-wasm test-js-native-cpu test-js-native-x64 test-js-native-interp test-py
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

wasm-pkg: build/polygrad-pkg.js
	@mkdir -p js/wasm
	cp build/polygrad-pkg.js js/wasm/polygrad.js

build/polygrad-pkg.js: $(WASM_SRC) Makefile
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
		-o build/polygrad-pkg.js $(WASM_SRC)

VENDOR_SRC = vendor/dht/dht.c vendor/stun/STUNExternalIP.c
P2P_SRC = src/p2p.c

bench-train-py:
	POLYGRAD_LIB=$(abspath build/libpolygrad.so) PYTHONPATH=py python bench/train_mlp_python.py

# ── Package builds / publishing ─────────────────────────────────────

build-py: build-py-sdist

build-py-sdist:
	cd py && rm -rf csrc dist build *.egg-info && \
		$(PYTHON) scripts/sync-csrc.py && \
		$(PYTHON) -m build --sdist

build-py-wheel:
	cd py && rm -rf csrc dist build *.egg-info && \
		$(PYTHON) scripts/sync-csrc.py && \
		$(PYTHON) -m build --wheel

build-python: build-py

publish-py: build-py-sdist
	cd py && $(TWINE) upload dist/*.tar.gz

publish-python: publish-py

build-js: wasm-pkg
	cd js && $(NODE) scripts/sync-csrc.js && $(NPM) run build:browser

publish-js: build-js
	cd js && $(NPM) publish

# ── P2P distributed training ───────────────────────────────────────

test-p2p: build/test_p2p
	./build/test_p2p

build/test_p2p: $(SRC) $(P2P_SRC) $(VENDOR_SRC) test/test_p2p.c
	@mkdir -p build
	$(CC) $(CFLAGS_DEBUG) -Ivendor/dht -Ivendor/stun -o $@ $^ $(LDFLAGS_DEBUG)

test-p2p-browser:
	$(NODE) browser/test/test_p2p.js

# ── Coverage ──────────────────────────────────────────────────────

coverage: build/polygrad_test_cov
	./build/polygrad_test_cov --fast
	@gcov -o build $(SRC) > /dev/null 2>&1
	@echo ""
	@echo "Coverage summary:"
	@for f in $(SRC); do \
		pct=$$(gcov -n "$$f" 2>/dev/null | grep -oP '\d+\.\d+%' | head -1); \
		[ -n "$$pct" ] && printf "  %-30s %s\n" "$$(basename $$f)" "$$pct"; \
	done
	@rm -f *.gcov

build/polygrad_test_cov: $(SRC) $(CODEC_SRC) $(TEST_SRC)
	@mkdir -p build
	$(CC) -std=c11 -g -O0 -fprofile-arcs -ftest-coverage \
		-o $@ $^ -lm -ldl -lgcov

clean:
	rm -rf build/ *.o

# ── Safety tooling ──────────────────────────────────────────────────

# Clang Static Analyzer (requires clang)
analyze:
	@mkdir -p build
	clang --analyze -std=c11 -Wno-unused-parameter $(SRC) 2>&1 | tee build/analyze.log
	@rm -f *.plist
	@echo "Analysis complete. See build/analyze.log"

# Cppcheck (install: apt install cppcheck)
cppcheck:
	cppcheck --enable=warning,performance,portability --std=c11 \
		--suppress=missingIncludeSystem --error-exitcode=1 $(SRC)

# clang-format (install: apt install clang-format)
format:
	clang-format -i src/*.c src/*.h test/*.c

format-check:
	clang-format --dry-run --Werror src/*.c src/*.h test/*.c

# MemorySanitizer (requires clang, incompatible with ASan)
test-msan: build/polygrad_test_msan
	./build/polygrad_test_msan

build/polygrad_test_msan: $(SRC) $(CODEC_SRC) $(TEST_SRC)
	@mkdir -p build
	clang -std=c11 -g -O1 -fsanitize=memory -fno-omit-frame-pointer \
		-o $@ $^ -lm -ldl -fsanitize=memory

# ── Full verification ──────────────────────────────────────────────────

verify: test test-parity format-check analyze fuzz-smoke
	@echo "All verification checks passed."
