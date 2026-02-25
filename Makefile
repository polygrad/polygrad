CC ?= gcc
CFLAGS_COMMON = -std=c11 -Wall -Wextra -Wpedantic -Wno-unused-parameter
CFLAGS_RELEASE = $(CFLAGS_COMMON) -O2
CFLAGS_DEBUG = $(CFLAGS_COMMON) -g -O0 -fsanitize=address,undefined -fno-omit-frame-pointer
LDFLAGS = -lm
LDFLAGS_DEBUG = -lm -ldl -fsanitize=address,undefined

# Detect CUDA availability
HAS_CUDA := $(shell test -f /usr/include/cuda.h && echo 1 || echo 0)

SRC = src/ops.c src/dtype.c src/arena.c src/hashmap.c src/uop.c src/pat.c src/alu.c src/sym.c src/shape.c src/sched.c src/autograd.c src/codegen.c src/render_c.c src/render_wgsl.c src/runtime_cpu.c src/wasm_builder.c src/render_wasm.c src/frontend.c src/rangeify.c src/indexing.c src/nn.c
TEST_SRC = test/test_main.c test/test_uop.c test/test_dtype.c test/test_pat.c test/test_sym.c test/test_shape.c test/test_sched.c test/test_autograd.c test/test_codegen.c test/test_wasm.c test/test_rangeify.c test/test_nn.c test/test_future_passes.c

ifeq ($(HAS_CUDA), 1)
  SRC += src/render_cuda.c src/runtime_cuda.c
  TEST_SRC += test/test_cuda.c
  CFLAGS_COMMON += -DPOLY_HAS_CUDA=1
endif
PARITY_RUNNER_SRC = test/test_tinygrad_runner.c
PARITY_SCRIPT = test/test_tinygrad_parity.py
PARITY_PY ?= conda run -n tiny python

# Emscripten WASM build (excludes runtime_cpu.c — no fork/dlopen in WASM)
WASM_SRC = src/ops.c src/dtype.c src/arena.c src/hashmap.c src/uop.c src/pat.c src/alu.c src/sym.c src/shape.c src/sched.c src/autograd.c src/codegen.c src/render_c.c src/render_wgsl.c src/wasm_builder.c src/render_wasm.c src/frontend.c src/rangeify.c src/indexing.c src/nn.c
WASM_EXPORTS = _poly_ctx_new,_poly_ctx_destroy,_poly_op_count,_poly_op_name,_poly_const_float,_poly_const_int,_poly_alu1,_poly_alu2,_poly_alu3,_poly_store_val,_poly_sink1,_poly_sink_n,_poly_buffer_f32,_poly_reshape,_poly_expand,_poly_reduce_axis,_poly_permute,_poly_shrink,_poly_flip,_poly_pad,_poly_grad,_poly_render_kernel_wasm,_poly_kernel_buf,_poly_exp,_poly_log,_poly_sin,_poly_cos,_poly_tan,_poly_sigmoid,_poly_tanh_act,_poly_relu,_poly_relu6,_poly_leaky_relu,_poly_gelu,_poly_quick_gelu,_poly_silu,_poly_elu,_poly_softplus,_poly_mish,_poly_hardtanh,_poly_hardswish,_poly_hardsigmoid,_poly_abs,_poly_sign,_poly_square,_poly_rsqrt,_poly_ceil,_poly_floor,_poly_round_f,_poly_isinf,_poly_isnan,_poly_eq,_poly_ne,_poly_gt,_poly_ge,_poly_le,_poly_where_op,_poly_maximum,_poly_minimum,_poly_clamp,_poly_sum_reduce,_poly_max_reduce,_poly_mean_reduce,_poly_dot,_poly_einsum,_poly_rearrange,_malloc,_free

.PHONY: all test test-fast test-parity test-parity-opt test-parity-cuda test-wasm test-p2p test-p2p-browser bench bench-cuda bench-train-c bench-train-py wasm clean analyze cppcheck format format-check test-msan

all: build/libpolygrad.a build/libpolygrad.so

build/libpolygrad.a: $(SRC)
	@mkdir -p build
	$(CC) $(CFLAGS_RELEASE) -c $(SRC)
	ar rcs $@ *.o
	@rm -f *.o

build/libpolygrad.so: $(SRC)
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

build/polygrad_test: $(SRC) $(TEST_SRC)
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
test-wasm: build/polygrad.js build/polygrad.wasm
	$(NODE) browser/test/test_tensor.js

wasm: build/polygrad.js build/polygrad.wasm

build/polygrad.js build/polygrad.wasm: $(WASM_SRC)
	@mkdir -p build
	emcc -O2 -std=c11 -Wall -Wextra -Wpedantic -Wno-unused-parameter \
		-s WASM=1 -s MODULARIZE=1 -s EXPORT_NAME=PolygradModule \
		-s EXPORTED_FUNCTIONS='[$(WASM_EXPORTS)]' \
		-s EXPORTED_RUNTIME_METHODS='[cwrap,getValue,setValue]' \
		-s ALLOW_MEMORY_GROWTH=1 \
		-s WASM_BIGINT \
		-o build/polygrad.js $(WASM_SRC)

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

build/polygrad_test_msan: $(SRC) $(TEST_SRC)
	@mkdir -p build
	clang -std=c11 -g -O1 -fsanitize=memory -fno-omit-frame-pointer \
		-o $@ $^ -lm -ldl -fsanitize=memory
