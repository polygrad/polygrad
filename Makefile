CC ?= gcc
CFLAGS_COMMON = -std=c11 -D_POSIX_C_SOURCE=200809L -Wall -Wextra -Wpedantic -Wno-unused-parameter -Isrc
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

SRC = src/ops.c src/dtype.c src/arena.c src/hashmap.c src/utils.c src/selftest.c src/ctx.c src/device.c src/placer.c src/engine/realize.c src/engine/jit.c src/uop.c src/pat.c src/alu.c src/sym.c src/shape.c src/autograd.c src/codegen.c src/render_c.c src/render_wgsl.c src/runtime_cpu.c src/runtime_wasm.c src/runtime_webgpu.c src/wasm_builder.c src/render_wasm.c src/frontend.c src/tensor.c src/optim.c src/schedule/rangeify.c src/simplify.c src/schedule/indexing.c src/nn.c src/engine/schedule.c src/interp.c
FILC_SRC = src/ops.c src/dtype.c src/arena.c src/hashmap.c src/utils.c src/selftest.c src/ctx.c src/device.c src/placer.c src/engine/realize.c src/engine/jit.c src/uop.c src/pat.c src/alu.c src/sym.c src/shape.c src/autograd.c src/codegen.c src/render_c.c src/render_wgsl.c src/runtime_cpu.c src/runtime_wasm.c src/runtime_webgpu.c src/wasm_builder.c src/render_wasm.c src/frontend.c src/tensor.c src/optim.c src/schedule/rangeify.c src/simplify.c src/schedule/indexing.c src/nn.c src/engine/schedule.c src/interp.c
LOADER_SRC = src/loaders/decoded.c src/loaders/import_error.c src/loaders/bind.c src/loaders/hf_decode.c src/loaders/gguf_decode.c src/loaders/gguf_loader.c src/loaders/import_desc.c
CODEC_SRC = vendor/cjson/cJSON.c src/safetensors.c src/wlrn.c src/ir.c src/bundle.c src/instance.c src/tokenizer.c src/models/mlp.c src/models/tabm.c src/models/nam.c src/models/registry.c src/models/gpt2.c src/models/qwen3.c src/models/hf_loader.c $(LOADER_SRC)
TEST_SRC = test/test_main.c test/test_uop.c test/test_utils.c test/test_dtype.c test/test_pat.c test/test_sym.c test/test_shape.c test/test_schedule_engine.c test/test_autograd.c test/test_codegen.c test/test_wasm.c test/test_rangeify.c test/test_reduce_simplify.c test/test_nn.c test/test_tensor.c test/test_future_passes.c test/test_safetensors.c test/test_wlrn.c test/test_ir.c test/test_instance.c test/test_mlp.c test/test_tabm.c test/test_nam.c test/test_hf.c test/test_qwen3.c test/test_f16.c test/test_schedule_runtime.c test/test_bundle.c test/test_registry.c test/test_realize.c test/test_threading.c

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

# Detect x86-64 for the tinygrad-style x86 ISA backend.
HAS_X86 := $(shell uname -m | grep -c x86_64)
ifeq ($(HAS_X86), 1)
  SRC += src/render_x86.c
  TEST_SRC += test/test_x86.c
  CFLAGS_COMMON += -DPOLY_HAS_X86=1
endif
PARITY_RUNNER_SRC = test/test_tinygrad_runner.c
PARITY_SCRIPT = test/test_tinygrad_parity.py
PARITY_PY ?= $(if $(wildcard references/.venv-tinygrad-py311/bin/python),references/.venv-tinygrad-py311/bin/python,conda run -n tiny python)

# Emscripten WASM build (excludes runtime_cpu.c — no fork/dlopen in WASM)
WASM_SRC = src/ops.c src/dtype.c src/arena.c src/hashmap.c src/utils.c src/selftest.c src/ctx.c src/device.c src/placer.c src/engine/realize.c src/engine/jit.c src/uop.c src/pat.c src/alu.c src/sym.c src/shape.c src/autograd.c src/codegen.c src/render_c.c src/render_wgsl.c src/runtime_wasm.c src/runtime_webgpu.c src/wasm_builder.c src/render_wasm.c src/frontend.c src/tensor.c src/optim.c src/schedule/rangeify.c src/simplify.c src/schedule/indexing.c src/nn.c src/engine/schedule.c src/interp.c vendor/cjson/cJSON.c src/safetensors.c src/ir.c src/bundle.c src/instance.c src/tokenizer.c src/models/mlp.c src/models/tabm.c src/models/nam.c src/models/registry.c src/models/gpt2.c src/models/qwen3.c src/models/hf_loader.c $(LOADER_SRC)
WASM_EXPORTS = _poly_ctx_new,_poly_ctx_destroy,_poly_ctx_set_preferred_device,_poly_ctx_set_frontend_buffer_release,_poly_ctx_named_count,_poly_ctx_stats,_poly_can_run_op,_poly_selftest,_poly_selftest_device,_poly_op_count,_poly_op_name,_poly_const_float,_poly_const_double,_poly_const_int,_poly_contiguous,_poly_add,_poly_sub,_poly_mul,_poly_div,_poly_alu1,_poly_alu2,_poly_alu3,_poly_store_val,_poly_sink1,_poly_sink_n,_poly_buffer_by_id,_poly_buffer_var_by_id,_poly_buffer_f32,_poly_buffer_f64,_poly_buffer_from_host,_poly_buffer_get_ptr,_poly_buffer_get_key,_poly_buffer_read,_poly_buffer_write,_poly_set_frontend_buffer_release,_poly_realize_uops,_poly_tensor_create,_poly_tensor_create_with_roots,_poly_tensor_update,_poly_tensor_to_device,_poly_tensor_assign,_poly_tensor_uop,_poly_tensor_uop_logical,_poly_tensor_uop_physical,_poly_tensor_device,_poly_realize_tensors,_poly_jit_new,_poly_jit_free,_poly_jit_set_prune,_poly_jit_begin_capture,_poly_jit_end_capture,_poly_jit_cancel_capture,_poly_jit_is_captured,_poly_jit_schedule_count,_poly_jit_run,_poly_jit_run_with_vars,_poly_uop_has_buffer_identity,_poly_uop_get_buffer_identity,_poly_uop_reachable,_poly_uop_substitute,_poly_reshape,_poly_expand,_poly_reduce_axis,_poly_permute,_poly_shrink,_poly_flip,_poly_pad,_poly_grad,_poly_grad_many,_poly_abi_version,_poly_device_by_name,_poly_device_name,_poly_dtype_id_by_name,_poly_uop_dtype_id,_poly_exp,_poly_log,_poly_log1p,_poly_expm1,_poly_sin,_poly_cos,_poly_tan,_poly_erf,_poly_erfc,_poly_erfinv,_poly_ndtri,_poly_digamma,_poly_lgamma,_poly_sigmoid,_poly_tanh_act,_poly_relu,_poly_relu6,_poly_leaky_relu,_poly_gelu,_poly_quick_gelu,_poly_silu,_poly_elu,_poly_softplus,_poly_mish,_poly_hardtanh,_poly_hardswish,_poly_hardsigmoid,_poly_abs,_poly_sign,_poly_square,_poly_rsqrt,_poly_ceil,_poly_floor,_poly_round_f,_poly_isinf,_poly_isnan,_poly_eq,_poly_ne,_poly_gt,_poly_ge,_poly_le,_poly_where_op,_poly_maximum,_poly_minimum,_poly_clamp,_poly_detach,_poly_cast_by_id,_poly_rand,_poly_randn,_poly_arange,_poly_eye,_poly_linspace,_poly_full,_poly_tril,_poly_triu,_poly_cholesky,_poly_cholesky_solve,_poly_triangular_solve,_poly_solve,_poly_lstsq,_poly_sum_reduce,_poly_max_reduce,_poly_mean_reduce,_poly_logsumexp,_poly_dot,_poly_qr,_poly_qr_ex,_poly_cross_entropy,_poly_gather_dim,_poly_scatter,_poly_scatter_reduce,_poly_sort,_poly_argsort,_poly_topk,_poly_einsum,_poly_rearrange,_exp2f,_log2f,_sinf,_powf,_malloc,_free,_poly_instance_from_ir,_poly_instance_free,_poly_instance_set_device,_poly_instance_call,_poly_instance_value_and_grad,_poly_instance_forward,_poly_instance_train_step,_poly_instance_set_optimizer,_poly_instance_set_optimizer_ex,_poly_instance_param_count,_poly_instance_param_name,_poly_instance_param_data,_poly_instance_param_shape,_poly_instance_buf_count,_poly_instance_buf_name,_poly_instance_buf_role,_poly_instance_buf_data,_poly_instance_buf_shape,_poly_instance_export_weights,_poly_instance_export_weights_ex,_poly_instance_import_weights,_poly_instance_export_ir,_poly_mlp_from_json,_poly_tabm_instance,_poly_nam_instance,_poly_instance_save_bundle,_poly_instance_save_bundle_ex,_poly_instance_from_bundle,_poly_uop_ndim,_poly_uop_max_shape_dims,_poly_uop_shape_dim,_poly_uop_const_i64,_poly_uop_unbind_var,_poly_uop_bind_value,_poly_ctx_arena,_poly_ctx_shape_cache,_poly_softmax,_poly_log_softmax,_poly_dot,_poly_qr,_poly_qr_ex,_poly_cross_entropy,_poly_gather,_poly_sum_reduce,_poly_max_reduce,_poly_mean_reduce,_poly_var_reduce,_poly_tril,_poly_triu,_poly_rmsnorm_apply,_poly_sdpa,_poly_rope,_poly_repeat_interleave,_poly_argmax,_poly_mse_loss,_poly_mae_loss,_poly_hf_load,_poly_gguf_load,_poly_gguf_decode,_poly_gguf_decoded_free,_poly_gguf_kv_int,_poly_gguf_kv_float,_poly_gguf_kv_string,_poly_import_last_error_code,_poly_import_last_error_message,_poly_tokenizer_from_gguf,_poly_tokenizer_from_json,_poly_tokenize,_poly_detokenize,_poly_tokenizer_free,_poly_tokenizer_vocab_size,_poly_tokenizer_bos_id,_poly_tokenizer_eos_id,_poly_gpt2,_poly_qwen3

WASM_EXPORTS := $(WASM_EXPORTS),_poly_tensor_requires_grad,_poly_tensor_set_requires_grad,_poly_instance_param_trainable,_poly_instance_set_param_trainable,_poly_instance_buf_trainable,_poly_instance_set_buf_trainable,_poly_instance_readback_param,_poly_instance_readback_buf,_poly_optim_build_step,_poly_register_buffer_by_id,_poly_register_existing_buffer,_poly_instance_from_sinks,_poly_instance_from_binding_arrays

WASM_ASYNCIFY_IMPORTS = ['js_webgpu_dispatch','js_webgpu_read_buffer_to_wasm','js_webgpu_read_buffer_to_hostkey']
WASM_ASYNCIFY_ONLY = ['poly_instance_call','poly_instance_forward','poly_instance_value_and_grad','poly_instance_train_step','run_instance_sink','poly_instance_param_data','poly_instance_buf_data','poly_instance_export_weights','poly_instance_export_weights_ex','poly_instance_save_bundle','poly_instance_save_bundle_ex','poly_instance_readback_param','poly_instance_readback_buf','poly_realize_uops','poly_realize_tensors','poly_jit_run','poly_jit_run_with_vars','poly_run_schedule','poly_schedule_execute_runner_call','poly_webgpu_execute','copy_execute_fn','poly_buffer_copy','poly_buffer_ensure_host_current','poly_buffer_read','poly_buffer_write','host_copy_in','webgpu_copy_out']
WASM_ASYNCIFY_FLAGS = -s ASYNCIFY=1 \
	-s "ASYNCIFY_IMPORTS=$(WASM_ASYNCIFY_IMPORTS)" \
	-s "ASYNCIFY_ONLY=$(WASM_ASYNCIFY_ONLY)"

QWEN3_GGUF ?= $(if $(POLY_QWEN3_GGUF),$(POLY_QWEN3_GGUF),$(CURDIR)/temp/Qwen3-0.6B-Q8_0.gguf)

.PHONY: all test test-fast test-common test-common-cpu test-common-cuda test-common-hip test-common-interp test-common-x86 test-specific-cuda test-specific-hip test-specific-x86 test-cuda test-hip test-interp test-x86 test-parity test-parity-opt test-parity-ir test-parity-ir-opt test-parity-cuda test-parity-hip test-symbolic-z3 test-qwen3 test-browser-qwen3 require-qwen3-gguf test-wasm test-wasm-new test-native test-browser test-p2p test-p2p-browser bench bench-cuda bench-model-cuda bench-hip bench-train-py bench-smoke bench-local-baseline bench-update-local-baseline bench-smoke-regression bench-ci-regression bench-ratios bench-ratio-local-baseline bench-update-local-ratio-baseline bench-parity bench-jax-js-wasm bench-jax-js-matmul-wasm bench-jax-js-model-wasm bench-jax-js-browser-wasm bench-jax-js-browser-matmul-wasm bench-jax-js-browser-model-wasm bench-compare bench-compare-global bench-regression bench-update-baseline fuzz fuzz-smoke fuzz-nightly fuzz-symbolic fuzz-symbolic-div wasm wasm-pkg build-py build-py-sdist build-py-wheel build-python publish-py publish-python build-js publish-js clean analyze cppcheck format format-check test-msan test-tsan verify coverage test-full test-js-native-cpu test-js-native-x86 test-js-native-interp test-js-native-cuda test-js-native-hip test-filc-interp-fast verify-source-mirrors test-py-x86

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

# Portable C backend suite routed through specific backend (POLY_DEVICE selector).
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
	$(SAN_RUN) POLY_DEVICE=x86 ./build/polygrad_test x86

require-qwen3-gguf:
	@if [ ! -f "$(QWEN3_GGUF)" ]; then \
		echo "Qwen3 GGUF fixture not found: $(QWEN3_GGUF)"; \
		echo "Set POLY_QWEN3_GGUF=/path/to/Qwen3-0.6B-Q8_0.gguf or place it under temp/."; \
		exit 2; \
	fi

test-qwen3: build/polygrad_test require-qwen3-gguf
	$(SAN_RUN) POLY_QWEN3_GGUF="$(QWEN3_GGUF)" ./build/polygrad_test qwen3

# Backend-specific tests only (uses substring filter)
test-specific-cuda: build/polygrad_test
	$(SAN_RUN) ./build/polygrad_test cuda

test-specific-hip: build/polygrad_test
	$(SAN_RUN) ./build/polygrad_test hip

test-parity: build/polygrad_parity_runner
	$(SAN_RUN) CACHELEVEL=0 $(PARITY_PY) $(PARITY_SCRIPT) --runner build/polygrad_parity_runner --mode values

test-parity-opt: build/polygrad_parity_runner
	$(SAN_RUN) CACHELEVEL=0 POLY_OPTIMIZE=1 $(PARITY_PY) $(PARITY_SCRIPT) --runner build/polygrad_parity_runner --mode values

test-parity-ir: build/polygrad_parity_runner
	$(SAN_RUN) CACHELEVEL=0 $(PARITY_PY) $(PARITY_SCRIPT) --runner build/polygrad_parity_runner --mode full --no-opt

test-parity-ir-opt: build/polygrad_parity_runner
	$(SAN_RUN) CACHELEVEL=0 POLY_OPTIMIZE=1 $(PARITY_PY) $(PARITY_SCRIPT) --runner build/polygrad_parity_runner --mode full

Z3_FUZZ_ITERS ?= 128
Z3_FUZZ_SEED ?= 0
test-symbolic-z3: build/libpolygrad.so
	POLYGRAD_LIB=$(abspath build/libpolygrad.so) $(PARITY_PY) \
		test/external/fuzz_symbolic_z3.py --mode general --seed $(Z3_FUZZ_SEED) --iters $(Z3_FUZZ_ITERS)
	POLYGRAD_LIB=$(abspath build/libpolygrad.so) $(PARITY_PY) \
		test/external/fuzz_symbolic_z3.py --mode div --seed $(Z3_FUZZ_SEED) --iters $(Z3_FUZZ_ITERS)

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

BENCH_SMOKE_JSON ?= bench/results/smoke-latest.json
BENCH_LOCAL_BASELINE ?= bench/baselines/local/$(shell hostname -s)-cpu.json
BENCH_BASELINE ?= $(BENCH_LOCAL_BASELINE)
BENCH_CI_BASELINE ?=
BENCH_SMOKE_ARGS ?=
BENCH_COMPARE_ARGS ?=

build/bench_smoke: $(SRC) $(CODEC_SRC) bench/bench_smoke.c
	@mkdir -p build
	$(CC) $(CFLAGS_RELEASE) -o $@ $^ -lm -ldl

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

verify-source-mirrors:
	$(PYTHON) scripts/verify-source-mirrors.py

test-py: verify-source-mirrors build/libpolygrad.so
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 POLYGRAD_LIB=build/libpolygrad.so PYTHONPATH=py python -m pytest py/tests/ -v

test-py-x86: verify-source-mirrors build/libpolygrad.so
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 POLY_DEVICE=x86 POLYGRAD_LIB=build/libpolygrad.so PYTHONPATH=py python -m pytest py/tests/test_tensor.py py/tests/test_nn.py py/tests/test_instance.py py/tests/test_hf.py py/tests/test_hf_e2e.py -v

test-js: test-js-wasm test-js-native

test-js-wasm: verify-source-mirrors wasm-pkg
	$(NODE) js/test/test_wasm.js

test-js-native: verify-source-mirrors js/build/Release/polygrad_napi.node
	$(NODE) js/test/test_native.js

js/build/Release/polygrad_napi.node: build/libpolygrad.a
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

test-js-browser: verify-source-mirrors wasm-pkg
	cd js && bash scripts/build-browser.sh && $(NODE) test/browser/run.js

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
TEST_ALL_DEPS = test test-x86 test-interp test-js-wasm test-js-native-cpu test-js-native-x86 test-js-native-interp test-py
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

build-py-sdist: verify-source-mirrors
	cd py && rm -rf csrc dist build *.egg-info && \
		$(PYTHON) scripts/sync-csrc.py && \
		$(PYTHON) -m build --sdist

build-py-wheel: verify-source-mirrors
	cd py && rm -rf csrc dist build *.egg-info && \
		$(PYTHON) scripts/sync-csrc.py && \
		$(PYTHON) -m build --wheel

build-python: build-py

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

# ThreadSanitizer focused smoke. The current threading contract permits
# independent contexts on separate threads; one PolyCtx remains thread-confined.
test-tsan: build/polygrad_test_tsan
	TSAN_OPTIONS=$(TSAN_OPTIONS) $(TSAN_RUNNER) ./build/polygrad_test_tsan threading

build/polygrad_test_tsan: $(SRC) $(CODEC_SRC) $(TEST_SRC)
	@mkdir -p build
	$(TSAN_CC) $(CFLAGS_COMMON) -g -O1 -fsanitize=thread -fno-omit-frame-pointer \
		-o $@ $^ -lm -ldl -pthread -fsanitize=thread

# ── Full verification ──────────────────────────────────────────────────

verify: test test-parity format-check analyze fuzz-smoke
	@echo "All verification checks passed."
