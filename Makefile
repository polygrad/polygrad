CC ?= gcc
CFLAGS_COMMON = -std=c11 -Wall -Wextra -Wpedantic -Wno-unused-parameter
CFLAGS_RELEASE = $(CFLAGS_COMMON) -O2
CFLAGS_DEBUG = $(CFLAGS_COMMON) -g -O0 -fsanitize=address,undefined -fno-omit-frame-pointer
LDFLAGS = -lm
LDFLAGS_DEBUG = -lm -ldl -fsanitize=address,undefined

# Detect CUDA availability
HAS_CUDA := $(shell test -f /usr/include/cuda.h && echo 1 || echo 0)

SRC = src/ops.c src/dtype.c src/arena.c src/hashmap.c src/uop.c src/pat.c src/alu.c src/sym.c src/shape.c src/sched.c src/autograd.c src/codegen.c src/render_c.c src/render_wgsl.c src/runtime_cpu.c src/wasm_builder.c src/render_wasm.c src/frontend.c src/rangeify.c src/indexing.c src/nn.c src/exec_plan.c
CODEC_SRC = vendor/cjson/cJSON.c src/safetensors.c src/wlrn.c src/ir.c src/instance.c src/model_mlp.c src/model_tabm.c src/model_nam.c src/modelzoo/modelzoo.c src/modelzoo/models/gpt2.c src/modelzoo/hf_loader.c
TEST_SRC = test/test_main.c test/test_uop.c test/test_dtype.c test/test_pat.c test/test_sym.c test/test_shape.c test/test_sched.c test/test_autograd.c test/test_codegen.c test/test_wasm.c test/test_rangeify.c test/test_nn.c test/test_future_passes.c test/test_safetensors.c test/test_wlrn.c test/test_ir.c test/test_instance.c test/test_mlp.c test/test_tabm.c test/test_nam.c test/test_hf.c test/test_f16.c

ifeq ($(HAS_CUDA), 1)
  SRC += src/render_cuda.c src/runtime_cuda.c
  TEST_SRC += test/test_cuda.c
  CFLAGS_COMMON += -DPOLY_HAS_CUDA=1
endif
PARITY_RUNNER_SRC = test/test_tinygrad_runner.c
PARITY_SCRIPT = test/test_tinygrad_parity.py
PARITY_PY ?= conda run -n tiny python

# Emscripten WASM build (excludes runtime_cpu.c — no fork/dlopen in WASM)
WASM_SRC = src/ops.c src/dtype.c src/arena.c src/hashmap.c src/uop.c src/pat.c src/alu.c src/sym.c src/shape.c src/sched.c src/autograd.c src/codegen.c src/render_c.c src/render_wgsl.c src/wasm_builder.c src/render_wasm.c src/frontend.c src/rangeify.c src/indexing.c src/nn.c src/exec_plan.c
WASM_EXPORTS = _poly_ctx_new,_poly_ctx_destroy,_poly_op_count,_poly_op_name,_poly_const_float,_poly_const_double,_poly_const_int,_poly_alu1,_poly_alu2,_poly_alu3,_poly_store_val,_poly_sink1,_poly_sink_n,_poly_buffer_f32,_poly_buffer_f64,_poly_reshape,_poly_expand,_poly_reduce_axis,_poly_permute,_poly_shrink,_poly_flip,_poly_pad,_poly_grad,_poly_render_kernel_wasm,_poly_kernel_buf,_poly_render_step_wasm_plan,_poly_wasm_stepplan_n_kernels,_poly_wasm_stepplan_kernel_bytes,_poly_wasm_stepplan_kernel_n_params,_poly_wasm_stepplan_n_buffers,_poly_wasm_stepplan_n_bindable_buffers,_poly_wasm_stepplan_kernel_param_buf_index,_poly_wasm_stepplan_exec_order,_poly_wasm_stepplan_destroy,_poly_wasm_stepplan_buf_size,_poly_wasm_stepplan_buf_nbytes,_poly_wasm_stepplan_bindable_buf_index,_poly_const_buffer_data,_poly_abi_version,_poly_exp,_poly_log,_poly_log1p,_poly_expm1,_poly_sin,_poly_cos,_poly_tan,_poly_erf,_poly_erfc,_poly_erfinv,_poly_ndtri,_poly_digamma,_poly_lgamma,_poly_sigmoid,_poly_tanh_act,_poly_relu,_poly_relu6,_poly_leaky_relu,_poly_gelu,_poly_quick_gelu,_poly_silu,_poly_elu,_poly_softplus,_poly_mish,_poly_hardtanh,_poly_hardswish,_poly_hardsigmoid,_poly_abs,_poly_sign,_poly_square,_poly_rsqrt,_poly_ceil,_poly_floor,_poly_round_f,_poly_isinf,_poly_isnan,_poly_eq,_poly_ne,_poly_gt,_poly_ge,_poly_le,_poly_where_op,_poly_maximum,_poly_minimum,_poly_clamp,_poly_detach,_poly_cast_by_id,_poly_rand,_poly_randn,_poly_arange,_poly_eye,_poly_linspace,_poly_full,_poly_tril,_poly_triu,_poly_cholesky,_poly_triangular_solve,_poly_sum_reduce,_poly_max_reduce,_poly_mean_reduce,_poly_logsumexp,_poly_dot,_poly_cross_entropy,_poly_einsum,_poly_rearrange,_exp2f,_log2f,_sinf,_powf,_malloc,_free

.PHONY: all test test-fast test-parity test-parity-opt test-parity-cuda test-wasm test-wasm-new test-native test-browser test-p2p test-p2p-browser bench bench-cuda bench-train-c bench-train-py wasm wasm-pkg clean analyze cppcheck format format-check test-msan verify coverage

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
	./build/polygrad_test

test-fast: build/polygrad_test
	./build/polygrad_test --fast

test-parity: build/polygrad_parity_runner
	ASAN_OPTIONS=detect_leaks=0 CACHELEVEL=0 $(PARITY_PY) $(PARITY_SCRIPT) --runner build/polygrad_parity_runner --mode full --no-opt

test-parity-opt: build/polygrad_parity_runner
	ASAN_OPTIONS=detect_leaks=0 CACHELEVEL=0 POLY_OPTIMIZE=1 POLY_EXPERIMENTAL_LATE=1 $(PARITY_PY) $(PARITY_SCRIPT) --runner build/polygrad_parity_runner --mode full

build/polygrad_test: $(SRC) $(CODEC_SRC) $(TEST_SRC)
	@mkdir -p build
	$(CC) $(CFLAGS_DEBUG) -o $@ $^ $(LDFLAGS_DEBUG)

build/polygrad_parity_runner: $(SRC) $(PARITY_RUNNER_SRC)
	@mkdir -p build
	$(CC) $(CFLAGS_DEBUG) -o $@ $^ $(LDFLAGS_DEBUG)

bench: build/bench_polygrad
	./build/bench_polygrad

build/bench_polygrad: $(SRC) bench/bench_polygrad.c
	@mkdir -p build
	$(CC) $(CFLAGS_RELEASE) -o $@ $^ -lm -ldl

ifeq ($(HAS_CUDA), 1)
bench-cuda: build/bench_cuda
	./build/bench_cuda

build/bench_cuda: $(SRC) bench/bench_cuda.c
	@mkdir -p build
	$(CC) $(CFLAGS_RELEASE) -o $@ $^ -lm -ldl

test-parity-cuda: build/polygrad_parity_runner_cuda
	CACHELEVEL=0 $(PARITY_PY) $(PARITY_SCRIPT) \
		--runner build/polygrad_parity_runner_cuda --cuda --mode values --atol 1e-4

build/polygrad_parity_runner_cuda: $(SRC) $(PARITY_RUNNER_SRC)
	@mkdir -p build
	$(CC) $(CFLAGS_RELEASE) -o $@ $^ -lm -ldl
endif

NODE ?= $(shell which node 2>/dev/null || echo node)
test-wasm: test-js-browser

test-py: build/libpolygrad.so
	python -m pytest py/tests/ -v

test-js: test-js-wasm test-js-native

test-js-wasm: wasm-pkg
	$(NODE) js/test/test_wasm.js

test-js-native:
	cd js && npm run build:native && cd .. && $(NODE) js/test/test_native.js

test-js-browser: wasm-pkg
	$(NODE) js/test/test_browser.js

test-js-legacy: build/libpolygrad.so
	$(NODE) js_legacy/test/test_tensor.js

test-wasm-legacy: build/polygrad.js build/polygrad.wasm
	$(NODE) js_legacy/polygrad/test/test_smoke.js

test-native-legacy: build/libpolygrad.so
	$(NODE) js_legacy/polygrad-node/test/test_native_smoke.js

test-browser-legacy: wasm-pkg
	$(NODE) js_legacy/polygrad/test/browser/test_browser.js

test-all: test test-parity test-js test-js-browser test-py

wasm: build/polygrad.js build/polygrad.wasm

build/polygrad.js build/polygrad.wasm: $(WASM_SRC) Makefile
	@mkdir -p build
	emcc -O2 -std=c11 -Wall -Wextra -Wpedantic -Wno-unused-parameter \
		-s WASM=1 -s MODULARIZE=1 -s EXPORT_NAME=PolygradModule \
		-s EXPORTED_FUNCTIONS='[$(WASM_EXPORTS)]' \
		-s "EXPORTED_RUNTIME_METHODS=['cwrap','getValue','setValue','HEAPU8','HEAP32','HEAPF32','HEAPF64']" \
		-s ALLOW_MEMORY_GROWTH=1 \
		-s WASM_BIGINT \
		-o build/polygrad.js $(WASM_SRC)

wasm-pkg: build/polygrad-pkg.js
	@mkdir -p js/wasm
	cp build/polygrad-pkg.js js/wasm/polygrad.js

build/polygrad-pkg.js: $(WASM_SRC) Makefile
	@mkdir -p build
	emcc -O2 -std=c11 -Wall -Wextra -Wpedantic -Wno-unused-parameter \
		-s WASM=1 -s MODULARIZE=1 -s EXPORT_NAME=createPolygrad \
		-s EXPORTED_FUNCTIONS='[$(WASM_EXPORTS)]' \
		-s "EXPORTED_RUNTIME_METHODS=['cwrap','getValue','setValue','HEAPU8','HEAP32','HEAPF32','HEAPF64']" \
		-s ALLOW_MEMORY_GROWTH=1 \
		-s WASM_BIGINT \
		-s SINGLE_FILE=1 \
		-s SINGLE_FILE_BINARY_ENCODE=0 \
		-s ENVIRONMENT='web,node' \
		-o build/polygrad-pkg.js $(WASM_SRC)

RECIPE_SRC = src/recipe.c
VENDOR_SRC = vendor/dht/dht.c vendor/stun/STUNExternalIP.c
P2P_SRC = src/p2p.c

bench-train-c: build/bench_train_mlp_c
	./build/bench_train_mlp_c

build/bench_train_mlp_c: $(SRC) $(RECIPE_SRC) bench/train_mlp_c.c
	@mkdir -p build
	$(CC) $(CFLAGS_RELEASE) -o $@ $^ -lm -ldl

bench-train-py:
	python bench/train_mlp_python.py

# ── P2P distributed training ───────────────────────────────────────

test-p2p: build/test_p2p
	./build/test_p2p

build/test_p2p: $(SRC) $(RECIPE_SRC) $(P2P_SRC) $(VENDOR_SRC) test/test_p2p.c
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

publish-python:
	cd py && rm -rf csrc dist build *.egg-info && \
		python scripts/sync-csrc.py && \
		python -m build --sdist && twine upload dist/*.tar.gz

verify: test test-parity format-check analyze
	@echo "All verification checks passed."
