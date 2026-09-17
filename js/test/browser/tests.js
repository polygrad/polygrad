"use strict";
(() => {
  var __getOwnPropNames = Object.getOwnPropertyNames;
  var __commonJS = (cb, mod) => function __require() {
    return mod || (0, cb[__getOwnPropNames(cb)[0]])((mod = { exports: {} }).exports, mod), mod.exports;
  };

  // test/test_tensor.js
  var require_test_tensor = __commonJS({
    "test/test_tensor.js"(exports, module) {
      "use strict";
      function assertClose(arr, expected, tol) {
        if (tol === void 0) tol = 1e-4;
        if (arr.length !== expected.length) {
          throw new Error(`Length mismatch: ${arr.length} vs ${expected.length}`);
        }
        for (let i = 0; i < arr.length; i++) {
          if (Number.isNaN(expected[i])) {
            if (!Number.isNaN(arr[i])) {
              throw new Error(`Mismatch at [${i}]: ${arr[i]} vs ${expected[i]}`);
            }
            continue;
          }
          const diff = Math.abs(arr[i] - expected[i]);
          if (!Number.isFinite(diff) || diff > tol) {
            throw new Error(`Mismatch at [${i}]: ${arr[i]} vs ${expected[i]}`);
          }
        }
      }
      function assert(cond, msg) {
        if (!cond) throw new Error(msg || "assertion failed");
      }
      function assertShape(actual, expected) {
        if (JSON.stringify(actual) !== JSON.stringify(expected)) {
          throw new Error(`shape mismatch: got [${actual}], expected [${expected}]`);
        }
      }
      function countGraphOp(root, op) {
        const seen = /* @__PURE__ */ new Set();
        const stack = [root];
        let count = 0;
        while (stack.length) {
          const node = stack.pop();
          if (!node || seen.has(node.key)) continue;
          seen.add(node.key);
          if (node.op === op) count++;
          for (const src of node.src) stack.push(src);
        }
        return count;
      }
      async function runTensorTests(pg, createRuntime) {
        const Tensor = pg.Tensor;
        const caps = pg.caps || {};
        const testFilter = (() => {
          if (pg && pg.testFilter) return String(pg.testFilter);
          if (typeof globalThis !== "undefined" && globalThis.__POLY_TEST_FILTER) {
            return String(globalThis.__POLY_TEST_FILTER);
          }
          if (typeof process !== "undefined" && process.env && process.env.POLY_TEST_FILTER) {
            return String(process.env.POLY_TEST_FILTER);
          }
          return "";
        })();
        const supportsF16 = caps.f16 !== false;
        const supportsF64 = caps.f64 !== false;
        let passed = 0, failed = 0, skipped = 0;
        const isolatedRuntime = (fn) => async () => {
          const runtime = await createRuntime();
          try {
            await fn(runtime);
          } finally {
            await runtime.dispose();
          }
        };
        async function test(name, fn) {
          if (testFilter && !name.includes(testFilter)) return;
          try {
            await fn();
            console.log(`  [PASS] ${name}`);
            passed++;
          } catch (e) {
            console.log(`  [FAIL] ${name}: ${e.message}`);
            if (e && e.stack) console.log(e.stack);
            failed++;
          }
        }
        async function testIf(cond, name, fn) {
          if (!cond) {
            console.log(`  [SKIP] ${name}`);
            skipped++;
            return;
          }
          await test(name, fn);
        }
        console.log(`Core: ${pg.core}, device: ${pg.device}
`);
        await test("device identity rejects unknown targets without AUTO fallback", async () => {
          const source = new Tensor([1, 2, 3]);
          const scalar = new Tensor(1);
          const sourceDevice = source.uop.device;
          try {
            for (const device of ["bogus", "CUDA:1", "HIP:1"]) {
              const requests = [
                ["array", () => new Tensor([1, 2], { device })],
                ["scalar", () => new Tensor(1, { device })],
                ["empty", () => Tensor.empty([2], { device })],
                ["clone", () => source.clone(device)],
                ["to", () => source.to(device)],
                ["scalar to", () => scalar.to(device)]
              ];
              for (const [name, request] of requests) {
                let rejected = false;
                let result;
                try {
                  result = request();
                } catch (e) {
                  rejected = /unsupported device/i.test(e.message);
                }
                if (result && result !== source && result !== scalar) result.dispose();
                assert(rejected, `${name} accepted unsupported device ${device}`);
              }
            }
            assertShape(source.shape, [3]);
            assert(source.uop.device === sourceDevice, "failed request changed source placement");
          } finally {
            scalar.dispose();
            source.dispose();
          }
        });
        await test("device identity preserves DISK path case without executing it", async () => {
          const source = new Tensor([1, 2, 3], { dtype: "float32" });
          const first = source.to("DISK:temp/CaseSensitive.bin");
          const second = first.to("DISK:temp/Different.bin");
          assert(first.device === "DISK:temp/CaseSensitive.bin", `lost path: ${first.device}`);
          assert(second.device === "DISK:temp/Different.bin", `lost path: ${second.device}`);
          assert(first !== second, "distinct paths collapsed to one backend identity");
          assert(first.to(first.device) === first, "same exact device should return self");
          first.dispose();
          second.dispose();
          source.dispose();
        });
        await test("construction const preserves UOp dtype and bound value", async () => {
          const value = new Tensor(1.5, { dtype: "float32" });
          const out = Tensor.const(value.uop, "int8");
          assert(out.dtype === "int8", "const did not cast its UOp");
          assert(await out.item() === 1, "const returned the wrong cast value");
          const variable = pg.uop.variable("construction_bound", 1, 10);
          const bound = variable.bind(5);
          const scalar = Tensor.const(bound, "int32");
          assert(await scalar.item() === 5, "const lost its bound value");
          scalar.dispose();
          bound.dispose();
          variable.dispose();
          out.dispose();
          value.dispose();
        });
        await test("construction empty preserves named disk storage without host I/O", async () => {
          const out = Tensor.empty([4], { dtype: "float32", device: "disk:CaseSensitive.bin" });
          assert(out.device === "DISK:CaseSensitive.bin", "empty lost the disk path");
          assert(!out.uop.is_realized, "empty allocated storage during construction");
          let rejected = false;
          try {
            pg._core.ffi.poly_tensor_empty_uop_name_by_id(pg._core.ctx, 0, [], 1, "disk:bad");
          } catch (e) {
            rejected = /shape length/.test(e.message);
          }
          assert(rejected, "FFI accepted a shape length mismatch");
          out.dispose();
        });
        await test("construction empty preserves symbolic dimensions", async () => {
          const n = pg.uop.variable("empty_batch", 1, 32), bound = n.bind(17);
          const matrix = Tensor.empty([bound, 2]), vector = Tensor.empty(bound);
          try {
            assert(JSON.stringify(matrix.shape) === "[32,2]", "empty lost bounded capacity");
            assert(JSON.stringify(vector.shape) === "[32]", "trailing UOp was parsed as options");
            assert(matrix.uop.op === pg._core.ops.SHRINK, "empty must retain its symbolic view");
            const weight = new Tensor([2, 3], { dtype: "float32" });
            let mixed;
            try {
              mixed = matrix.mul(weight);
              assertShape(mixed.shape, [32, 2]);
            } finally {
              if (mixed) mixed.dispose();
              weight.dispose();
            }
            for (const bad of [NaN, Infinity, 1.5, -1]) {
              let rejected = false;
              try {
                Tensor.empty([bad]);
              } catch {
                rejected = true;
              }
              assert(rejected, "invalid dimension was coerced into storage");
            }
          } finally {
            matrix.dispose();
            vector.dispose();
            bound.dispose();
            n.dispose();
          }
        });
        await test("construction rejects a failed UOp cast instead of creating zero", async () => {
          const source = pg.uop.constant(1.5, "float32");
          source.cast = () => null;
          let rejected = false, output;
          try {
            output = new Tensor(source, { dtype: "int8" });
          } catch (e) {
            rejected = /cast failed/.test(e.message);
          } finally {
            if (output) output.dispose();
            source.dispose();
          }
          assert(rejected, "failed cast became a zero Tensor");
        });
        await test("execution scalar reductions and virtual oneHot", async () => {
          for (const axis of [0, -1]) {
            const x2 = new Tensor(2, { dtype: "float32" });
            assertClose(await x2.sum(axis).toArray(), [2]);
            assertClose(await x2.softmax(axis).toArray(), [1]);
            assertClose(await x2.logSoftmax(axis).toArray(), [0]);
          }
          const x = new Tensor([1, 2, 4]).oneHot(6);
          const before = x.uop.key;
          await x.realizeAsync();
          assert(x.uop.key === before, "virtual realization changed its graph");
          assertClose(await x.toArray(), [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]);
        });
        await test("execution NOOPT policy restores and computes product", async () => {
          const before = pg.noopt;
          try {
            for (const mode of [0, 1, 0]) {
              pg.noopt = mode;
              assert(pg.noopt === mode, "NOOPT core value disagrees");
              assertClose(await new Tensor([1, 2, 3], { dtype: "float32" }).prod().toArray(), [6]);
            }
          } finally {
            pg.noopt = before;
          }
        });
        await test("execution IGNORE_BEAM_CACHE policy reaches core", async () => {
          const old = pg.ignoreBeamCache;
          try {
            for (const mode of [1, 0, 1]) {
              pg.ignoreBeamCache = mode;
              assert(pg.ignoreBeamCache === mode, "IGNORE_BEAM_CACHE core value disagrees");
              assert(pg._core.ffi.poly_get_ignore_beam_cache() === mode, "policy did not reach C");
            }
            let rejected = false;
            try {
              pg.ignoreBeamCache = 0.5;
            } catch (e) {
              rejected = /int32/.test(e.message);
            }
            assert(rejected && pg.ignoreBeamCache === 1, "invalid policy mutated C state");
          } finally {
            pg.ignoreBeamCache = old;
          }
        });
        await test("execution BEAM policy reaches core and computes values", async () => {
          const before = pg.beam;
          const beforeCache = pg.ignoreBeamCache;
          const module2 = pg._core.Module;
          const state = module2 && module2.__polygradWebGpuState;
          const device = state && state.device;
          const createQuerySet = device && device.createQuerySet;
          let timestamps = 0;
          if (device && device.features.has("timestamp-query")) {
            device.createQuerySet = function(descriptor) {
              if (descriptor.type === "timestamp") timestamps++;
              return createQuerySet.call(this, descriptor);
            };
          }
          const x = new Tensor(new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]));
          try {
            pg.ignoreBeamCache = 1;
            for (const width of [1, 2, 0]) {
              pg.beam = width;
              assert(pg.beam === width, "BEAM core value disagrees");
              assertClose(
                await x.mul(2).add(width).toArray(),
                [2, 4, 6, 8, 10, 12, 14, 16].map((v) => v + width)
              );
            }
            let rejected = false;
            try {
              pg.beam = 0.5;
            } catch (e) {
              rejected = /int32/.test(e.message);
            }
            assert(rejected, "fractional BEAM width accepted");
            if (device && device.features.has("timestamp-query")) {
              assert(timestamps > 0, "BEAM returned values without timing a WebGPU candidate");
            }
          } finally {
            if (device) device.createQuerySet = createQuerySet;
            pg.beam = before;
            pg.ignoreBeamCache = beforeCache;
            x.dispose();
          }
        });
        await test("execution einsum scalar ellipsis trace and accumulation", async () => {
          const scalar = new Tensor(2, { dtype: "float32" });
          assertClose(await Tensor.einsum("->", scalar).toArray(), [2]);
          const a = Tensor.arange(12).cast("float32").reshape(2, 2, 3);
          const b = Tensor.ones([2, 3, 2]);
          assertClose(await Tensor.einsum("...ij,...jk->...ik", a, b).toArray(), [3, 3, 12, 12, 21, 21, 30, 30]);
          const x = Tensor.arange(18).cast("float32").reshape(2, 3, 3);
          assertClose(await Tensor.einsum("...ii->...", x).toArray(), [12, 39]);
          const small = new Tensor(new Int8Array([100, 100, 100])).reshape(1, 3);
          const sum = Tensor.einsum("ij->i", small);
          assert(sum.dtype === "int32", `einsum accumulation dtype ${sum.dtype}`);
          assertClose(await sum.toArray(), [300]);
        });
        console.log("-- Creation --");
        await test("default dtype policy reaches C and restores", async () => {
          const before = [pg.defaultFloat, pg.defaultInt];
          try {
            pg.defaultFloat = "float16";
            pg.defaultInt = "int16";
            assert(new Tensor([1.25]).dtype === "float16", "floating list ignored default");
            assert(new Tensor([1]).dtype === "int16", "integer list ignored default");
            assert(new Tensor(new Float32Array([1.25])).dtype === "float32");
            assert(Tensor.arange(3).dtype === "int16");
            for (const x2 of [
              Tensor.ones(2),
              Tensor.empty(2),
              Tensor.rand(2),
              Tensor.randn(2),
              Tensor.eye(2),
              Tensor.linspace(0, 1, 3)
            ]) assert(x2.dtype === "float16", `factory dtype ${x2.dtype}`);
            const x = new Tensor([1], { dtype: "int32" }).exp();
            assert(x.dtype === "float16", `C transcendental dtype ${x.dtype}`);
            if (pg.canRun({ dtype: "float16" })) assertClose(await x.toArray(), [Math.E], 0.01);
            const data = new Tensor([3, 1, 2], { dtype: "float32" });
            assertClose(await data.argmax().toArray(), [0]);
            assertClose(await data.sort()[1].toArray(), [1, 2, 0]);
            let error;
            try {
              pg.defaultFloat = "not_a_dtype";
            } catch (e) {
              error = e;
            }
            assert(error && pg.defaultFloat === "float16", "failed dtype update changed policy");
          } finally {
            pg.defaultFloat = before[0];
            pg.defaultInt = before[1];
          }
        });
        await test("dtype admission rejects floating shifts and overflowing arange", async () => {
          for (const fn of [
            () => new Tensor([1.5]).lshift(1),
            () => new Tensor([1.5]).rshift(1),
            () => Tensor.arange(129, { dtype: "int8" })
          ]) {
            let error;
            try {
              fn();
            } catch (e) {
              error = e;
            }
            assert(error, "invalid operation accepted");
          }
          assert(Tensor.arange(2 ** 31, 2 ** 31 + 3).dtype === "int64");
        });
        await test("dtype admission random output casting and unknown options", async () => {
          let error;
          try {
            Tensor.rand(2, { generator: "x" });
          } catch (e) {
            error = e;
          }
          assert(error instanceof TypeError, "rand silently ignored an unknown option");
          for (const dtype of ["bool", "int8", "int32"]) {
            Tensor.manual_seed(42);
            const expected = await Tensor.randn(3).cast(dtype).toArray();
            Tensor.manual_seed(42);
            const out = Tensor.randn(3, { dtype });
            assert(out.dtype === dtype);
            assertClose(await out.toArray(), expected);
          }
        });
        await test("indexing owner bound prefix reduction and flip", async () => {
          const n = pg.uop.variable("indexing_extent", 2, 8);
          const zero = pg.uop.constant(0);
          for (const size of [2, 4, 7]) {
            const source = new Tensor([1, 2, 3, 4, 5, 6, 7, 8], { dtype: "float32" });
            const bound = n.bind(size);
            const root = source.uopPhysical;
            const raw = pg._core.ffi.poly_shrink_uop(root.ctx, root.raw, [zero.raw], [bound.raw], 1);
            assert(raw, "symbolic prefix construction failed");
            const view = pg.uop.wrap(raw);
            const prefix = new Tensor(view);
            const total = prefix.sum();
            const flipped = prefix.flip(0);
            const first = flipped.shrink([[0, 1]]).sum();
            assertClose(await total.toArray(), [size * (size + 1) / 2], 0);
            assertClose(await first.toArray(), [size], 0);
            for (const value of [first, flipped, total, prefix, view, root, bound, source]) await value.dispose();
          }
          await zero.dispose();
          await n.dispose();
        });
        await test("typed PARAM bounds preserve scalar values and ownership", async () => {
          const cases = [
            [0.25, 0.75, "float32"],
            [-Infinity, Infinity, "float32"],
            [2n ** 63n, 2n ** 64n - 1n, "uint64"],
            [2n ** 130n, 2n ** 131n, "weakint"],
            [0n, 10n ** 600n - 1n, "weakint"]
          ];
          for (const [lo, hi, dtype] of cases) {
            const value = pg.uop.variable("typed_bound", lo, hi, dtype, 1, true);
            assert(value, "variable construction failed");
            const text = value.toString();
            for (const endpoint of [lo, hi]) {
              const expected = String(endpoint).replace("Infinity", "inf");
              assert(text.includes(expected), `${expected} not preserved: ${text}`);
            }
            value.dispose();
          }
          const a = pg.uop.variable("key", 0, 1, "float32", 1, true);
          const b = pg.uop.variable("key", 0n, 1n, "float32", 1, true);
          assert(pg.uop.key(a) === pg.uop.key(b), "equal numeric bounds must CSE");
          a.dispose();
          b.dispose();
          assert(pg.uop.variable("bad", NaN, 1, "float32", 1, true) === null);
        });
        await test("wide scalar bindings preserve BigInt and execute supported values", async () => {
          const variable = pg.uop.variable("wide_binding", -(1n << 63n), (1n << 63n) - 1n, "int64");
          for (const value of [
            -(1n << 63n),
            -(1n << 40n) - 1n,
            (1n << 32n) + 7n,
            (1n << 53n) + 1n,
            (1n << 63n) - 1n
          ]) {
            const bound = variable.bind(value);
            assert(bound && bound.op === pg._core.ops.AFTER);
            const sources = bound.src;
            const stored = sources[1].src;
            assert(sources[1].op === pg._core.ops.STORE);
            assert(stored[1].toString().includes(String(value)), `lost binding value ${value}: ${stored[1]}`);
            if (pg.device !== "webgpu") {
              const input = new Tensor(bound);
              const output = input.contiguous();
              const data = await output.toArray();
              assert(data.length === 1 && BigInt(data[0]) === value, `binding execution changed ${value}`);
              output.dispose();
              input.dispose();
            }
            for (const item of [...sources, ...stored]) item.dispose();
            bound.dispose();
          }
          for (const value of [-(1n << 63n) - 1n, 1n << 63n, 1n << 100n, 9007199254740992, 1.25]) {
            let error;
            try {
              variable.bind(value);
            } catch (e) {
              error = e;
            }
            assert(error instanceof RangeError || error instanceof TypeError);
          }
          variable.dispose();
          const small = pg.uop.variable("binding_exec", -8, 8, "int32");
          for (const value of [-3n, 7n]) {
            const bound = small.bind(value);
            const input = new Tensor(bound);
            const output = input.contiguous();
            assertClose(await output.toArray(), [Number(value)]);
            output.dispose();
            input.dispose();
            bound.dispose();
          }
          small.dispose();
        });
        await test("dtype API queries match pinned metadata", async () => {
          const bytes = {
            bool: 1,
            int8: 1,
            uint8: 1,
            int16: 2,
            uint16: 2,
            int32: 4,
            uint32: 4,
            int64: 8,
            uint64: 8,
            float16: 2,
            bfloat16: 2,
            float32: 4,
            float64: 8,
            fp8e4m3: 1,
            fp8e5m2: 1,
            fp8e4m3fnuz: 1,
            fp8e5m2fnuz: 1,
            weakint: null,
            weakfloat: null
          };
          for (const [dtype, size] of Object.entries(bytes)) {
            const tensor = new Tensor(0, { dtype });
            assert(tensor.isFloatingPoint() === /^(float|bfloat|fp8|weakfloat)/.test(dtype));
            assert(pg.uop.dtype(tensor.uopPhysical) === dtype);
            if (size === null) {
              let error;
              try {
                tensor.elementSize();
              } catch (e) {
                error = e;
              }
              assert(error && /elementSize requires a concrete dtype/.test(error.message));
            } else assert(tensor.elementSize() === size);
          }
        });
        await test("device metadata preserves deviceless and placed roots", async () => {
          const t = new Tensor(pg.uop.constant(2, "float32"));
          assert(t.uop.device === null && t.device === null, "constant has no storage device");
          const x = Tensor.empty([4]);
          assert(typeof x.uop.device === "string", "BUFFER carries its exact device");
          assert(x.add(1).uop.device === x.uop.device, "ALU inherits the physical source device");
          const moved = x.to("interp");
          assert(moved.uop.device === "INTERP" && moved.device === "INTERP");
        });
        await test("Runtime ownership: reject foreign realization before FFI", async () => {
          const other = await createRuntime();
          const x = new Tensor([1, 2], { dtype: "float32" });
          const y = new other.Tensor([3, 4], { dtype: "float32" });
          try {
            let failure;
            try {
              await x.realizeAsync(y);
            } catch (e) {
              failure = e;
            }
            assert(failure && /another Runtime/.test(failure.message), "foreign Tensor reached realization");
            assertClose(await x.toArrayAsync(), [1, 2]);
            assertClose(await y.toArrayAsync(), [3, 4]);
          } finally {
            await x.dispose();
            await y.dispose();
            await other.dispose();
          }
        });
        await testIf(pg.device !== "webgpu", "Runtime ownership: realized typed storage views", async () => {
          const source = new Tensor(Array.from({ length: 100 }, (_, i) => i), { dtype: "uint8" });
          await source.realizeAsync();
          const views = [26, 65].map((n) => source.shrink([[n, n + 4]]).bitcast("uint16"));
          try {
            await views[0].realizeAsync(views[1]);
            assertClose(await views[0].toArrayAsync(), [6938, 7452], 0);
            assertClose(await views[1].toArrayAsync(), [16961, 17475], 0);
          } finally {
            for (const view of views) await view.dispose();
            await source.dispose();
          }
        });
        await test("device metadata admits deviceless assign and indexing", async () => {
          const value = new Tensor(pg.uop.constant(1.25, "float32"), { device: "interp" });
          const target = Tensor.empty([4]);
          assert(value.device === null);
          target.assign(value);
          assertClose(await target.toArrayAsync(), [1.25, 1.25, 1.25, 1.25]);
          const index = new Tensor([1, 0], { dtype: "int32" });
          const source = Tensor.arange(2, { device: "interp" });
          assert(source.device === null);
          assertClose(await source.add(index).toArrayAsync(), [1, 1]);
          assertClose(await source.gather(0, index).toArrayAsync(), [1, 0]);
          assertClose(await source.getitem(index).toArrayAsync(), [1, 0]);
          assertClose(await Tensor.zeros([2]).scatter(0, index, source.cast("float32")).toArrayAsync(), [1, 0]);
        });
        await test("UOp accessors preserve live wrapper identity", async () => {
          const source = pg.uop.constant(2, "float32");
          const t = new Tensor(source);
          const root = t.uop;
          assert(root === source, "constructor must preserve its supplied live UOp");
          assert(t.uop === root, "unchanged root must reuse its live wrapper");
          await t.realizeAsync();
          assert(t.uop === root && t.uopPhysical === root);
        });
        await test("UOp accessors refresh disposed and replaced roots", async () => {
          const t = Tensor.empty([4], { logical: "always" });
          const root = t.uop;
          await root.dispose();
          const replacement = t.uop;
          assert(replacement !== root && replacement.raw);
          t.copyFrom(new Float32Array([1, 1, 1, 1]));
          const result = t.add(1).contiguous();
          result.preserveLogical();
          const before = result.uop;
          const logical = result.uopLogical;
          await result.realizeAsync();
          const after = result.uop;
          assert(after !== before && after.key !== before.key);
          assert(result.uop === after && result.uopLogical === logical);
          assert(before.raw, "caller still owns the old graph");
        });
        await test("dtype API bound UOp constants retain their owner and type", async () => {
          for (const [value, dtype] of [[true, "bool"], [1, "int32"], [1.75, "float32"]]) {
            const uop = pg.uop.constant(value, dtype);
            assert(uop.ctx === pg._core.ctx);
            assert(uop.op === pg._core.ops.CONST && uop.src.length === 0);
            assert(pg.uop.dtype(uop) === dtype);
            const tensor = new Tensor(uop);
            assert(tensor.dtype === dtype);
            assertClose(await tensor.toArrayAsync(), [Number(value)]);
          }
        });
        await test("constructor parity null is scalar zero", async () => {
          for (const dtype of [void 0, "float32", "int32", "bool"]) {
            const tensor = new Tensor(null, { dtype });
            assertShape(tensor.shape, []);
            assert(tensor.uopPhysical.op === pg._core.ops.CONST);
            assert(tensor.uopPhysical.src.length === 0);
            if (!dtype) assert(tensor.dtype === "weakfloat");
            assertClose(await tensor.toArrayAsync(), [0]);
          }
        });
        for (const dtype of ["int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"]) {
          await test(`constructor parity integer storage ${dtype}`, async () => {
            const bits = Number(dtype.match(/\d+/)[0]);
            const values = [-(1n << 100n) - 3n, -3.5, -1, 0, 129, 256, (1n << 100n) + 5n];
            const narrow = dtype.startsWith("uint") ? BigInt.asUintN : BigInt.asIntN;
            const expected = values.map((v) => narrow(bits, typeof v === "bigint" ? v : BigInt(Math.trunc(v))));
            const tensor = new Tensor(values, { dtype });
            assertShape(tensor.shape, [7]);
            const root = tensor.uopPhysical;
            assert(root.op === pg._core.ops.COPY && root.src.length === 1);
            assert(root.src[0].op === pg._core.ops.BUFFER && root.src[0].src.length === 1);
            assert(tensor.dtype === dtype);
            const got = Array.from(await tensor.toArrayAsync());
            assert(got.length === expected.length);
            assert(got.every((v, i) => BigInt(v) === expected[i]), `wrong ${dtype} storage values`);
          });
        }
        await test("constructor parity rejects weak storage and ragged shapes before import", async () => {
          for (const dtype of ["weakint", "weakfloat"]) {
            for (const values of [[1], new Float32Array([1])]) {
              let error;
              try {
                new Tensor(values, { dtype });
              } catch (e) {
                error = e;
              }
              assert(error && /cannot create storage for weak dtype/.test(error.message));
            }
          }
          for (const data of [[[1], []], [[], [[]]], [[1, 2], [3], [4, 5, 6]]]) {
            let error;
            try {
              new Tensor(data, { dtype: "int32" });
            } catch (e) {
              error = e;
            }
            assert(error && /inhomogeneous shape/.test(error.message));
          }
        });
        await test("constructor parity integer lists reject nonfinite values", async () => {
          for (const dtype of ["int32", "uint32", "int64", "uint64"]) {
            for (const value of [NaN, Infinity, -Infinity]) {
              let error;
              try {
                new Tensor([value], { dtype });
              } catch (e) {
                error = e;
              }
              assert(error instanceof RangeError, `${dtype} admitted ${value}`);
            }
          }
        });
        await test("logical policy scope and tensor override", async () => {
          const current = new Tensor([0]).add(1);
          assert(current.logicalPolicy === "until_realize");
          await current.realizeAsync();
          assert(current.logicalState === "retired");
          const scoped = pg.withLogical("never", () => new Tensor([1, 2]));
          assert(scoped.logicalPolicy === "never");
          assert(scoped.logicalState === "never_constructed");
          assert(scoped.uopLogical === null);
          const retained = pg.withLogical("always", () => new Tensor([3, 4]).add(1));
          await retained.realizeAsync();
          assert(retained.logicalPolicy === "always");
          assert(retained.logicalState === "available");
          const dropped = new Tensor([5, 6], { logical: false });
          assert(dropped.logicalPolicy === "never");
          assert(dropped.uopLogical === null);
          assert(dropped.setLogicalPolicy("always") === false);
          const descendant = dropped.add(1);
          assert(descendant.logicalPolicy === "never");
          assert(descendant.logicalState === "never_constructed");
          assertClose(await descendant.toArray(), [6, 7]);
          const cloned = dropped.clone();
          assert(cloned.logicalPolicy === "never");
          assert(cloned.logicalState === "never_constructed");
          assert(cloned.uopLogical === null);
          assertClose(await cloned.toArray(), [5, 6]);
          const gradSource = new Tensor([2], { dtype: "float32", logical: false });
          gradSource.mul(gradSource).sum().backward();
          assert(gradSource.grad.logicalPolicy === "never");
          assert(gradSource.grad.logicalState === "never_constructed");
          assert(gradSource.grad.uopLogical === null);
          assertClose(await gradSource.grad.toArray(), [4]);
        });
        for (const logical of ["never", "always", "until_realize"]) {
          for (const realized of [false, true]) {
            await test(`copyFrom host input ${logical} realized=${realized}`, async () => {
              const x = new Tensor(new Float32Array([1, 2, 3]), { logical });
              const retained = x.uopLogical;
              if (realized) await x.realizeAsync();
              const before = x.uopPhysical.key;
              if (pg.device === "webgpu" && !realized) {
                let error;
                try {
                  x.copyFrom([4, 5, 6]);
                } catch (e) {
                  error = e;
                }
                assert(error && String(error.message).includes("copyFromAsync"));
                assert(x.uopPhysical.key === before);
                await x.copyFromAsync([4, 5, 6]);
              } else {
                x.copyFrom([4, 5, 6]);
              }
              const current = x.uopPhysical;
              assert(current.op === pg._core.ops.BUFFER && current.src.length === 1 && current.src[0].op === pg._core.ops.CONST);
              if (realized) assert(current.key === before);
              if (logical === "never") assert(x.uopLogical === null);
              else if (logical === "always") assert(x.uopLogical.key === retained.key);
              else assert(x.logicalState === "retired");
              x.copyFrom([7, 8, 9]);
              assert(x.uopPhysical.key === current.key);
              assertClose(await x.toArrayAsync(), [7, 8, 9]);
            });
          }
        }
        await test("copyFrom validates before materialization and orders pending assign", async () => {
          const x = new Tensor(new Float32Array([1, 2, 3]), { logical: "always" });
          const before = x.uopPhysical.key;
          for (const bad of [new Float32Array([9]), new Int32Array([9, 9, 9])]) {
            let error;
            try {
              x.copyFrom(bad);
            } catch (e) {
              error = e;
            }
            assert(error && /size mismatch|dtype mismatch/.test(error.message));
            assert(x.uopPhysical.key === before);
          }
          assertClose(await x.toArrayAsync(), [1, 2, 3]);
          x.assign(new Tensor([10, 20, 30], { dtype: "float32" }));
          if (pg.device === "webgpu") await x.copyFromAsync([4, 5, 6]);
          else x.copyFrom([4, 5, 6]);
          assertClose(await x.toArrayAsync(), [4, 5, 6]);
        });
        await test("copyFromAsync snapshots caller bytes before materialization", async () => {
          const x = new Tensor(new Float32Array([1, 2, 3]));
          const values = new Float32Array([4, 5, 6]);
          const pending = x.copyFromAsync(values);
          values.fill(99);
          assert(await pending === x);
          assertClose(await x.toArrayAsync(), [4, 5, 6]);
          const current = x.uopPhysical.key;
          let error;
          try {
            await x.copyFromAsync([0]);
          } catch (e) {
            error = e;
          }
          assert(error && String(error.message).includes("size mismatch"));
          assert(x.uopPhysical.key === current);
          assertClose(await x.toArrayAsync(), [4, 5, 6]);
        });
        await test("from vector", async () => {
          const t = new Tensor([1, 2, 3]);
          assertShape(t.shape, [3]);
          assert(t.dtype === "int32", `expected int32, got ${t.dtype}`);
          assertClose(await t.toArray(), [1, 2, 3]);
        });
        for (const mode of ["single", "batch", "singleAsync", "batchAsync"]) {
          await testIf(
            pg.device !== "webgpu" || mode.endsWith("Async"),
            `readback temporaries retire before return ${mode}`,
            isolatedRuntime(async (pg2) => {
              const halfSource = new pg2.Tensor(new Float32Array([3, 4]));
              const inputs = [
                new pg2.Tensor(new Float32Array([1, 2])),
                halfSource.cast(pg2.device === "webgpu" ? "int32" : "float16")
              ];
              const read = async () => mode === "single" ? inputs.map((t) => t.toArray()) : mode === "batch" ? pg2.Tensor.toTypedArrays(inputs) : mode === "singleAsync" ? [await inputs[0].toArrayAsync(), await inputs[1].toArrayAsync()] : pg2.Tensor.toTypedArraysAsync(inputs);
              await halfSource.realizeAsync();
              for (const t of inputs) await t.realizeAsync();
              pg2.collect();
              const baseline = pg2.stats().coreStats.tensorRecords;
              const checkOwners = () => assert(
                pg2.stats().coreStats.tensorRecords === baseline,
                `readback retained temporary owners: ${baseline} -> ${pg2.stats().coreStats.tensorRecords}`
              );
              const proto = pg2.Tensor.prototype;
              const method = mode.endsWith("Async") ? "_readBufferBytesAsync" : "_readBufferBytes";
              const original = proto[method];
              let outputs;
              try {
                for (let i = 0; i < 3; i++) {
                  outputs = await read();
                  assertClose(outputs[0], [1, 2]);
                  assertClose(outputs[1], [3, 4]);
                  checkOwners();
                }
                proto[method] = () => {
                  throw new Error("injected readback failure");
                };
                let error;
                try {
                  await read();
                } catch (e) {
                  error = e;
                }
                assert(error && /injected readback failure/.test(error.message));
                checkOwners();
                proto[method] = original;
                if (mode.startsWith("batch")) {
                  const prepare = inputs[1]._prepareReadback;
                  inputs[1]._prepareReadback = () => {
                    throw new Error("injected preparation failure");
                  };
                  try {
                    error = null;
                    try {
                      await read();
                    } catch (e) {
                      error = e;
                    }
                    assert(error && /injected preparation failure/.test(error.message));
                    checkOwners();
                  } finally {
                    inputs[1]._prepareReadback = prepare;
                  }
                }
              } finally {
                proto[method] = original;
                for (const t of inputs) await t.dispose();
                await halfSource.dispose();
              }
              pg2.collect();
              assertClose(outputs[0], [1, 2]);
              assertClose(outputs[1], [3, 4]);
            })
          );
        }
        await test("readback copy has no retained host shadow", isolatedRuntime(async (pg2) => {
          const value = pg2.Tensor.arange(4).cast("float32").add(1).contiguous();
          await value.realize();
          const before = pg2.stats().coreStats.bufferOwnedSourceBytes;
          for (let i = 0; i < 3; i++) {
            const output2 = await value.toArrayAsync();
            assertClose(output2, [1, 2, 3, 4]);
            output2[0] = 99;
            assert(
              pg2.stats().coreStats.bufferOwnedSourceBytes === before,
              "readback attached a persistent host mirror"
            );
          }
          const output = await value.toArrayAsync();
          await value.dispose();
          pg2.collect();
          await pg2.dispose();
          assertClose(output, [1, 2, 3, 4]);
        }));
        await test("schedule cache clear preserves live Model and JIT", isolatedRuntime(async (rt) => {
          const x = rt.Tensor.empty([3], { dtype: "float32" });
          const model = await rt.Model.fromTensors({ inputs: { x }, outputs: { prediction: x.add(2) } });
          const data = new Float32Array([1, 2, 3]);
          const f = rt.jit((a) => a.add(1).realize());
          try {
            assertClose((await model.forward({ x: data })).prediction, [3, 4, 5]);
            await x.copyFromAsync(data);
            for (let i = 0; i < 3; i++) assertClose(await (await f(x)).toArrayAsync(), [2, 3, 4]);
            const schedules = f.scheduleCount;
            for (let i = 0; i < 3; i++) {
              rt.clearScheduleCache();
              rt.collect();
              assertClose(await (await f(x)).toArrayAsync(), [2, 3, 4]);
              assert(f.scheduleCount === schedules, "cache clear must not recapture live JIT");
              assertClose((await model.forward({ x: data })).prediction, [3, 4, 5]);
            }
          } finally {
            f.dispose();
            await model.dispose();
          }
          await rt.dispose();
          let rejected = false;
          try {
            rt.clearScheduleCache();
          } catch (e) {
            rejected = /disposed/.test(e.message);
          }
          assert(rejected, "disposed runtime accepted cache clear");
        }));
        await test("schedule cache clear rejects active async readback", isolatedRuntime(async (rt) => {
          const x = await rt.Tensor.arange(4, { dtype: "float32" }).realize();
          const pending = x.toArrayAsync();
          let error;
          try {
            rt.clearScheduleCache();
          } catch (e) {
            error = e;
          }
          assertClose(await pending, [0, 1, 2, 3]);
          if (rt.caps.core === "wasm" && rt.caps.device === "webgpu") {
            assert(error && /active async work/.test(error.message), "suspended WebGPU accepted cache clear");
          } else {
            assert(!error, "synchronous backend rejected idle cache clear");
          }
          rt.clearScheduleCache();
          rt.collect();
          assertClose(await x.toArrayAsync(), [0, 1, 2, 3]);
        }));
        await test("Tensor dispose retires its exact core owner", isolatedRuntime(async (pg2) => {
          const Tensor2 = pg2.Tensor;
          const before = pg2.stats().coreStats.tensorRecords;
          const t = Tensor2.empty([8], { dtype: "float32" });
          assert(
            pg2.stats().coreStats.tensorRecords === before + 1,
            "Tensor construction should add one core owner"
          );
          await t.dispose();
          assert(
            pg2.stats().coreStats.tensorRecords === before,
            `Tensor dispose should retire its exact core owner: before=${before}, after=${pg2.stats().coreStats.tensorRecords}`
          );
          await t.dispose();
        }));
        await test("raw UOp owns residency after Tensor dispose", isolatedRuntime(async (pg2) => {
          const Tensor2 = pg2.Tensor;
          const before = pg2.stats().coreStats;
          const t = Tensor2.empty([1024], { dtype: "float32" });
          t.copyFrom(new Float32Array(1024));
          const uop = t.uop;
          await t.dispose();
          let stats = pg2.stats().coreStats;
          assert(
            stats.tensorRecords === before.tensorRecords,
            "raw UOp ownership should not retain the Tensor record"
          );
          assert(
            stats.memUsed === before.memUsed + 4096,
            `raw physical UOp should retain its storage: before=${before.memUsed}, after=${stats.memUsed}`
          );
          await uop.dispose();
          pg2.collect();
          stats = pg2.stats().coreStats;
          assert(
            stats.memUsed === before.memUsed,
            "raw UOp dispose should retire its storage"
          );
        }));
        await test("downstream core graph owns disposed input residency", isolatedRuntime(async (pg2) => {
          const Tensor2 = pg2.Tensor;
          const before = pg2.stats().coreStats;
          const input = Tensor2.empty([1024], { dtype: "float32" });
          input.copyFrom(new Float32Array(1024));
          const one = new Tensor2(1, { dtype: "float32" });
          const out = input.add(one);
          await input.dispose();
          await one.dispose();
          let stats = pg2.stats().coreStats;
          assert(
            stats.memUsed === before.memUsed + 4096,
            "downstream physical UOp graph should retain input storage"
          );
          await out.realize();
          stats = pg2.stats().coreStats;
          assert(
            stats.memUsed === before.memUsed + 4096,
            "realized output should replace obsolete input storage"
          );
          await out.dispose();
          pg2.collect();
          stats = pg2.stats().coreStats;
          assert(
            stats.memUsed === before.memUsed,
            "last downstream owner should retire output storage"
          );
        }));
        await test("array and TypedArray dtype inference matches tinygrad", async () => {
          assert(new Tensor([[1, 2], [3, 4]]).dtype === "int32", "nested integers should infer int32");
          assert(new Tensor([true, false]).dtype === "bool", "booleans should infer bool");
          assert(new Tensor([1, 2.5]).dtype === "float32", "mixed numeric values should infer float32");
          assert(new Tensor([]).dtype === "float32", "empty arrays should infer float32");
          assert(new Tensor(new Int16Array([1, 2])).dtype === "int16", "Int16Array should preserve int16");
          assert(new Tensor(new Uint32Array([1, 2])).dtype === "uint32", "Uint32Array should preserve uint32");
        });
        await test("bfloat16 host values stage through float32", async () => {
          const t = new Tensor([1, 2, 3, 4], { dtype: "bfloat16" });
          const y = await t.add(t).realize();
          assert(y.dtype === "bfloat16", `expected bfloat16, got ${y.dtype}`);
          assertClose(await y.toArray(), [2, 4, 6, 8]);
        });
        await test("explicit bfloat16 overrides Float64Array source dtype", async () => {
          const t = new Tensor(new Float64Array([1, 2]), { dtype: "bfloat16" });
          assert(t.dtype === "bfloat16", `expected bfloat16, got ${t.dtype}`);
          assertShape(t.shape, [2]);
          assertClose(await t.toArray(), [1, 2]);
        });
        await test("fp8 host values match current tinygrad", async () => {
          const values = [-Infinity, -1.5, -0, 0, 0.1, 1, 1.5, 448, Infinity, NaN];
          const cases = {
            fp8e4m3: [NaN, -1.5, -0, 0, 0.1015625, 1, 1.5, 448, NaN, NaN],
            fp8e5m2: [-Infinity, -1.5, -0, 0, 0.09375, 1, 1.5, 448, Infinity, NaN],
            fp8e4m3fnuz: [NaN, -1.5, 0, 0, 0.1015625, 1, 1.5, 240, NaN, NaN],
            fp8e5m2fnuz: [NaN, -1.5, 0, 0, 0.09375, 1, 1.5, 448, NaN, NaN]
          };
          for (const [dtype, expected] of Object.entries(cases)) {
            const tensor = new Tensor(values, { dtype });
            assert(tensor.dtype === dtype, `${dtype}: got ${tensor.dtype}`);
            assert(countGraphOp(tensor.uop, pg._core.ops.COPY) === 1, `${dtype}: missing creation COPY`);
            const actual = await tensor.cast("float32").toArray();
            assert(actual.length === expected.length, `${dtype}: length mismatch`);
            for (let i = 0; i < actual.length; i++) {
              if (Number.isNaN(expected[i])) assert(Number.isNaN(actual[i]), `${dtype}[${i}] expected NaN`);
              else assert(Object.is(actual[i], expected[i]), `${dtype}[${i}] ${actual[i]} != ${expected[i]}`);
            }
          }
        });
        await testIf(supportsF16, "numeric float16 host values preserve pinned bits and direct topology", async () => {
          const values = [1.5, -2.25, 0.5, NaN, Infinity, -Infinity, 65504];
          const inputs = [
            ["array", values, [7]],
            ["nested", [[1.5, -2.25], [0.5, 65504]], [2, 2]],
            ["float64array", new Float64Array(values), [7]],
            ["uint16array-numeric", new Uint16Array([1, 2, 3]), [3]]
          ];
          for (const [name, input, shape] of inputs) {
            const direct = new Tensor(input, { dtype: "float16" });
            const control = new Tensor(input, { dtype: "float32" }).cast("float16");
            assertShape(direct.shape, shape);
            assert(countGraphOp(direct.uop, pg._core.ops.CAST) === 0, `${name} direct graph gained CAST`);
            assert(countGraphOp(control.uop, pg._core.ops.CAST) === 1, `${name} control graph lost CAST`);
            const actual = await direct.toArray();
            const expected = await control.toArray();
            assert(actual.length === expected.length, `${name} length mismatch`);
            for (let i = 0; i < actual.length; i++) {
              if (Number.isNaN(expected[i])) assert(Number.isNaN(actual[i]), `${name}[${i}] expected NaN`);
              else assert(
                Object.is(actual[i], expected[i]) || actual[i] === expected[i],
                `${name}[${i}] ${actual[i]} != ${expected[i]}`
              );
            }
          }
          const directSubnormal = new Tensor([2 ** -24], { dtype: "float16" });
          const controlSubnormal = new Tensor([2 ** -24], { dtype: "float32" }).cast("float16");
          if (pg.core === "native" && pg.device === "cpu") {
            assert(
              await directSubnormal.item() === 2 ** -24,
              "native CPU direct float16 lost minimum subnormal storage"
            );
            assert(
              await controlSubnormal.item() === 2 ** -24,
              "native CPU float16 control lost minimum subnormal"
            );
          } else if (pg.device === "wasm" || pg.device === "interp") {
            assert(
              await directSubnormal.item() === 0,
              `${pg.device} direct float16 must match pinned non-native-half flush`
            );
            assert(
              await controlSubnormal.item() === 2 ** -24,
              `${pg.device} float16 control must match pinned non-native-half graph`
            );
          }
          const scalar = new Tensor(1.5, { dtype: "float16" });
          assertShape(scalar.shape, []);
          assert(await scalar.item() === 1.5, "scalar float16 construction mismatch");
        });
        await test("from scalar", async () => {
          const cases = [
            [new Tensor(true), "bool", true],
            [new Tensor(42), "weakint", 42],
            [new Tensor(42, { dtype: "float32" }), "float32", 42],
            [new Tensor(7, { device: "cuda" }), "weakint", 7],
            [new Tensor(1.5, { device: "cuda" }), "weakfloat", 1.5]
          ];
          for (const [tensor, dtype, value] of cases) {
            assertShape(tensor.shape, []);
            assert(tensor.dtype === dtype, `expected ${dtype}, got ${tensor.dtype}`);
            assert(tensor.uop.op === pg._core.ops.CONST, "expected scalar CONST root");
            assert(tensor.uop.key === tensor.uopLogical.key, "expected shared scalar roots");
            if (tensor.device === "CPU") {
              const actual = await tensor.item();
              assert(Math.abs(Number(actual) - Number(value)) < 1e-4, `Expected ${value}, got ${actual}`);
            }
          }
        });
        await test("from 2D", async () => {
          const t = new Tensor([[1, 2], [3, 4]]);
          assertShape(t.shape, [2, 2]);
          assertClose(await t.toArray(), [1, 2, 3, 4]);
        });
        await test("empty creates unrealized buffer placeholder", async () => {
          const t = Tensor.empty([2, 3]);
          assertShape(t.shape, [2, 3]);
          assert(t.uop.hasBufferIdentity(), "empty should be backed by a BUFFER UOp");
          assert(t.uopPhysical, "empty should have a physical root at construction");
          assert(
            t.uopLogical.buffer.src[0].op === pg._core.ops.UNIQUE,
            "logical BUFFER should retain the portable resource identity"
          );
          assert(t.uopLogical.buffer.src.length === 1, "logical BUFFER should stay device-free");
          assert(
            t.uopPhysical.buffer.src[0].op === pg._core.ops.CONST,
            "physical BUFFER should encode the same resource slot in ParamArg"
          );
        });
        await test("movement is realized through recursive base", async () => {
          const source = await new Tensor([1, 2, 3, 4]).realize();
          const reshaped = source.reshape(2, 2);
          const view = reshaped.flatten().shrink([[1, 3]]);
          assert(view.uop.op === pg._core.ops.SHRINK, "expected a SHRINK view");
          assert(view.uop.base.key === source.uop.base.key, "movement base should be recursive");
          assert(reshaped.uop.realized === null, "RESHAPE is not directly realized");
          assert(reshaped.uop.isRealized, "RESHAPE should be realized through its base");
          assert(view.uop.realized === null, "movement UOp is not directly realized");
          assert(view.uop.isRealized, "allocated recursive base should realize the movement view");
          assert(view.uop.is_realized, "snake-case realization alias should match");
        });
        await test("flatten resolves negative dimensions like tinygrad", async () => {
          const flattened = Tensor.arange(32).reshape(1, 2, 16).flatten(-2);
          assertShape(flattened.shape, [1, 32]);
          assert(flattened.uop.op === pg._core.ops.RESHAPE, "flatten should be one RESHAPE");
          assertClose(await flattened.toArray(), Array.from({ length: 32 }, (_, i) => i));
        });
        await test("clone is lazy separate and preserves state", async () => {
          const source = Tensor.empty([4], { dtype: "float32" }).is_param_(false);
          source.copyFrom(new Float32Array([1, 2, 3, 4]));
          await source.sum().backward();
          const cloned = source.clone(pg.device);
          assert(cloned.uopLogical && cloned.uopLogical.src.length === 2, "clone should be AFTER");
          assert(cloned.uopLogical.src[1].src.length === 2, "clone effect should be STORE");
          assert(cloned.uopLogical.src[0].buffer.key !== source.uop.buffer.key, "clone needs a separate buffer");
          assert(cloned.isParam === false, "clone should preserve isParam");
          assert(cloned.grad && cloned.grad.uopLogical.src.length === 2, "clone should recursively clone grad");
          assert(
            cloned.grad.uopLogical.src[0].buffer.key !== source.grad.uopLogical.src[0].buffer.key,
            "cloned grad needs a separate buffer"
          );
          assertClose(await cloned.toArray(), [1, 2, 3, 4]);
          assertClose(await cloned.grad.toArray(), [1, 1, 1, 1]);
        });
        await test("clone preserves scalar shape across devices", async () => {
          const source = Tensor.full([], 3, { device: "cpu" });
          const cloned = source.clone("interp");
          assertShape(cloned.shape, []);
          assert(cloned.device === "INTERP", `expected INTERP, got ${cloned.device}`);
          assertClose(await cloned.toArray(), [3]);
        });
        await test("static constructors preserve scalar shape", async () => {
          const tensors = [Tensor.zeros([]), Tensor.ones([]), Tensor.full([], 3)];
          for (const tensor of tensors) assertShape(tensor.shape, []);
          assertClose(await tensors[0].toArray(), [0]);
          assertClose(await tensors[1].toArray(), [1]);
          assertClose(await tensors[2].toArray(), [3]);
        });
        await test("detach is a lazy graph boundary", async () => {
          const source = new Tensor([[1, 2], [3, 4]], { dtype: "float32" });
          const detached = source.detach();
          assertShape(detached.shape, source.shape);
          assert(detached.dtype === source.dtype, "detach should preserve dtype");
          assert(detached.device === source.device, "detach should preserve device");
          assert(detached.isParam === true, "ordinary Tensor operations should default to isParam=true");
          assert(detached.uopLogical.op === pg._core.ops.DETACH, "detach should create a DETACH UOp");
          assert(detached.uopLogical.src.length === 1, "detach should retain a unary graph node");
          assert(
            detached.uopLogical.src[0].key === source.uopLogical.key,
            "logical detach should retain the logical source UOp"
          );
          if (detached.uopPhysical) {
            assert(detached.uopPhysical.op === pg._core.ops.DETACH, "physical detach should retain DETACH");
            assert(
              detached.uopPhysical.src[0].key === source.uop.key,
              "physical detach should retain the current source UOp"
            );
          }
          await detached.sum().backward();
          assertClose(await source.grad.toArray(), [0, 0, 0, 0]);
        });
        await test("gradient owner detached self retains seed", async () => {
          const source = Tensor.full([], 2, { dtype: "float32" });
          const detached = source.detach();
          await detached.backward();
          assertClose(await detached.grad.toArray(), [1]);
          assertClose(await source.grad.toArray(), [0]);
        });
        await test("contiguousBackward has exact gradient barrier", async () => {
          const source = new Tensor([1, -2, 3], { dtype: "float32" });
          const result = source.mul(2).contiguousBackward();
          assert(
            result.uopLogical.op === pg._core.ops.CONTIGUOUS_BACKWARD,
            "logical root should be CONTIGUOUS_BACKWARD"
          );
          assert(
            result.uop.op === pg._core.ops.CONTIGUOUS_BACKWARD,
            "physical root should be CONTIGUOUS_BACKWARD"
          );
          assert(
            result.uopLogical.src[0].op === pg._core.ops.MUL,
            "CONTIGUOUS_BACKWARD should wrap the exact MUL input"
          );
          assert(
            result.contiguous_backward().uop.op === pg._core.ops.CONTIGUOUS_BACKWARD,
            "snake-case alias should retain CONTIGUOUS_BACKWARD"
          );
          await result.square().sum().backward();
          assertClose(await source.grad.toArray(), [8, -16, 24]);
        });
        await test("backward clones deviceless grad and accumulates in place", async () => {
          const x = Tensor.empty([4], { dtype: "float32" });
          const loss = x.sum();
          await loss.backward();
          const firstGrad = x.grad;
          const firstRoot = firstGrad.uopLogical;
          const firstBuffer = firstRoot.src[0].buffer.key;
          assert(firstRoot.src.length === 2, "first grad should be AFTER");
          await loss.backward();
          assert(x.grad === firstGrad, "gradient accumulation should preserve Tensor identity");
          const secondRoot = x.grad.uopLogical;
          assert(secondRoot.src[0].key === firstRoot.key, "gradient effect root changed");
          assert(secondRoot.src[0].src[0].buffer.key === firstBuffer, "gradient buffer identity changed");
          assertClose(await x.grad.toArray(), [2, 2, 2, 2]);
        });
        await test("backward through clone reaches source", async () => {
          const source = Tensor.empty([4], { dtype: "float32" });
          source.copyFrom(new Float32Array([1, 2, 3, 4]));
          const cloned = source.clone();
          await cloned.sum().backward();
          assertClose(await source.grad.toArray(), [1, 1, 1, 1]);
          assertClose(await cloned.grad.toArray(), [1, 1, 1, 1]);
        });
        await test("backward retains distinct wrappers sharing one UOp", async () => {
          const x = new Tensor([1, 2, 3, 4], { dtype: "float32" });
          const y = new Tensor(x.uop, {});
          assert(x !== y, "expected distinct Tensor wrappers");
          assert(x.uop.key === y.uop.key, "expected one shared current UOp");
          await x.sum().backward();
          assertClose(await x.grad.toArray(), [1, 1, 1, 1]);
          assertClose(await y.grad.toArray(), [1, 1, 1, 1]);
        });
        await test("static constructors accept tinygrad-style shape arrays", async () => {
          const z = Tensor.zeros([2, 3]);
          const o = Tensor.ones([2, 3]);
          const e = Tensor.empty([2, 3]);
          assertShape(z.shape, [2, 3]);
          assertShape(o.shape, [2, 3]);
          assertShape(e.shape, [2, 3]);
          assertClose(await z.toArray(), [0, 0, 0, 0, 0, 0]);
          assertClose(await o.toArray(), [1, 1, 1, 1, 1, 1]);
        });
        await test("arange follows tinygrad start stop order", async () => {
          const deviceFree = Tensor.arange(6);
          const deviceFreeRoot = deviceFree.uop.key;
          assert(
            Number(pg._core.ffi.poly_uop_device(deviceFree.uop.raw)) === 0,
            "pure arange should remain device-free before readback"
          );
          assertClose(await deviceFree.toArray(), [0, 1, 2, 3, 4, 5]);
          assert(
            deviceFree.uop.key === deviceFreeRoot,
            "device-free readback must realize a temporary without rewriting the source root"
          );
          assertClose(await Tensor.arange(0, 6).toArray(), [0, 1, 2, 3, 4, 5]);
          assertClose(await Tensor.arange(2, 8, 2).toArray(), [2, 4, 6]);
        });
        await test("runtime exposes uop namespace", async () => {
          const t = new Tensor([[1, 2], [3, 4]]);
          assert(pg.uop, "runtime should expose pg.uop");
          assertShape(pg.uop.shape(t.uop), [2, 2]);
          assert(pg.uop.dtype(t.uop) === "int32", `expected int32, got ${pg.uop.dtype(t.uop)}`);
          assert(!pg.uop.hasBufferIdentity(t.uop), "host import should remain a lazy COPY");
          await t.realize();
          assert(pg.uop.hasBufferIdentity(t.uop), "realized host tensor should have buffer identity");
          assert(pg.uop.buffer(t.uop), "pg.uop.buffer should return a UOp");
        });
        await test("constructor rejects unsupported options", async () => {
          for (const key of ["shape", "dytpe", "_unknown"]) {
            let error = null;
            try {
              new Tensor(new Float32Array([1, 2, 3, 4]), { [key]: [2, 2] });
            } catch (e) {
              error = e;
            }
            assert(error instanceof TypeError && error.message.includes(key), `expected option error for ${key}`);
          }
          const tensor = new Tensor(new Float32Array([1, 2, 3, 4])).reshape(2, 2);
          assertShape(tensor.shape, [2, 2]);
          assertClose(await tensor.toArray(), [1, 2, 3, 4]);
          tensor.dispose();
        });
        await test("customKernel explicit store cast supports mixed dtypes", async () => {
          const input = new Tensor([11, 22, 33, 44], { dtype: "int32" });
          const output = Tensor.empty(4, { dtype: "float32" });
          const result = output.customKernel(input, (out, value) => {
            const i = pg.uop.range(4, 0);
            return out.index(i).store(value.index(i).cast("float32")).end(i).sink(
              new pg.uop.KernelInfo("mixed_store_cast")
            );
          })[0];
          assertClose(await result.toArray(), [11, 22, 33, 44]);
          result.dispose();
          output.dispose();
          input.dispose();
        });
        if (pg.core === "native" && pg.device === "cpu") {
          await test("customKernel CPU rejects vector store dtype mismatch", async () => {
            const input = new Tensor([11, 22, 33, 44], { dtype: "int32" });
            const output = Tensor.empty(4, { dtype: "float32" });
            let result = null, error = null;
            try {
              result = output.customKernel(input, (out, value) => {
                const i = pg.uop.range(4, 0);
                return out.index(i).store(value.index(i)).end(i).sink(
                  new pg.uop.KernelInfo("invalid_vector_store")
                );
              })[0];
              await result.toArray();
            } catch (e) {
              error = e;
            } finally {
              if (result) result.dispose();
              output.dispose();
              input.dispose();
            }
            assert(error instanceof Error, "mismatched vector STORE must not return reinterpreted bits");
          });
        }
        await test("customKernel executes UOp CALL body", async () => {
          function addKernel(c2, a2, b2) {
            c2 = c2.flatten();
            a2 = a2.flatten();
            b2 = b2.flatten();
            const i = pg.uop.range(c2.numel(), 0);
            return c2.index(i).store(a2.index(i).add(b2.index(i))).end(i).sink(
              new pg.uop.KernelInfo("custom_add_4")
            );
          }
          const a = new Tensor([1, 2, 3, 4], { dtype: "float32" });
          const b = new Tensor([10, 20, 30, 40], { dtype: "float32" });
          const c = Tensor.empty([4], { dtype: "float32" });
          const out = c.customKernel(a, b, addKernel)[0];
          assertClose(await out.toArray(), [11, 22, 33, 44]);
        });
        await test("customKernel RANGE numeric scalar preserves weakint", async () => {
          const index = pg.uop.range(64, 0);
          const offset = index.mul(64);
          assert(pg.uop.dtype(index) === "weakint", "RANGE should expose weakint dtype");
          assert(pg.uop.dtype(index.src[0]) === "weakint", "RANGE bound should be weakint");
          assert(pg.uop.dtype(offset) === "weakint", "index expression should remain weakint");
          assert(
            offset.src.every((src) => pg.uop.dtype(src) === "weakint"),
            "numeric scalar should be coerced to the index weakint dtype"
          );
        });
        await test("customKernel multi-output backward matches tinygrad pattern", async () => {
          let callbackCall = null;
          function addmulKernel(c2, d2, a2, b2) {
            c2 = c2.flatten();
            d2 = d2.flatten();
            a2 = a2.flatten();
            b2 = b2.flatten();
            const i = pg.uop.range(c2.numel(), 0);
            const storeC = c2.index(i).store(a2.index(i).add(b2.index(i)));
            const storeD = d2.index(i).store(a2.index(i).mul(b2.index(i)));
            return storeC.group(storeD).end(i).sink({ arg: new pg.uop.KernelInfo("addmul") });
          }
          function backwardAddmul(gradC, gradD, call) {
            callbackCall = call;
            const [, , , a2, b2] = call.src;
            const gradA = new Tensor(gradC).add(new Tensor(gradD).mul(new Tensor(b2))).uop;
            const gradB = new Tensor(gradC).add(new Tensor(gradD).mul(new Tensor(a2))).uop;
            return [null, null, gradA, gradB];
          }
          const aVals = [
            0.3,
            -1.2,
            0.7,
            2.1,
            -0.5,
            1.4,
            -2.2,
            0.9,
            1.1,
            -0.8,
            2.4,
            -1.7,
            0.2,
            0.6,
            -0.4,
            1.8
          ];
          const bVals = [
            1.2,
            0.5,
            -0.3,
            0.8,
            2,
            -1.1,
            0.4,
            -0.7,
            0.9,
            1.5,
            -2.5,
            0.1,
            -1.3,
            0.2,
            1.7,
            -0.6
          ];
          const aRef = new Tensor(aVals, {}).reshape(4, 4);
          const bRef = new Tensor(bVals, {}).reshape(4, 4);
          await aRef.add(bRef).sum().add(aRef.mul(bRef).sum()).backward();
          const a = new Tensor(aVals, {}).reshape(4, 4);
          const b = new Tensor(bVals, {}).reshape(4, 4);
          await a.realize(b);
          const aPhysical = a.uopPhysical.key;
          const bPhysical = b.uopPhysical.key;
          const [c, d] = Tensor.empty([4, 4]).customKernel(
            Tensor.empty([4, 4]),
            a,
            b,
            { fxn: addmulKernel, gradFxn: backwardAddmul }
          );
          await c.sum().add(d.sum()).backward();
          assert(callbackCall.src[3].key === aPhysical, "callback must receive physical a CALL slot");
          assert(callbackCall.src[4].key === bPhysical, "callback must receive physical b CALL slot");
          assertClose(await a.grad.toArray(), await aRef.grad.toArray(), 1e-4);
          assertClose(await b.grad.toArray(), await bRef.grad.toArray(), 1e-4);
        });
        await test("customKernel physical AFTER preserves data gradient", async () => {
          function identityKernel(x2) {
            x2 = x2.flatten();
            const i = pg.uop.range(x2.numel(), 0);
            return x2.index(i).store(x2.index(i)).end(i).sink({ arg: new pg.uop.KernelInfo("identity") });
          }
          function backwardIdentity(grad, call) {
            assert(call.src.length === 2, "expected one-argument custom CALL in backward");
            return [null];
          }
          const x = Tensor.empty([4], { dtype: "float32" });
          x.copyFrom(new Float32Array([1, 2, 3, 4]));
          const y = x.customKernel({ fxn: identityKernel, gradFxn: backwardIdentity })[0];
          assert(y.uopLogical && y.uopLogical.src.length === 2, "expected logical AFTER");
          assert(y.uopPhysical && y.uopPhysical.src.length === 2, "expected physical AFTER");
          assert(y.uopLogical.op === y.uopPhysical.op, "logical and physical aliases must both be AFTER");
          await y.sum().backward();
          assertClose(await x.grad.toArray(), [1, 1, 1, 1]);
          assertClose(await y.grad.toArray(), [1, 1, 1, 1]);
        });
        await test("customKernel separates output and input gradient edges", async () => {
          function identityKernel(out2, x2) {
            out2 = out2.flatten();
            x2 = x2.flatten();
            const i = pg.uop.range(out2.numel(), 0);
            return out2.index(i).store(x2.index(i)).end(i).sink({ arg: new pg.uop.KernelInfo("identity_grad_edges") });
          }
          function backwardIdentity(grad, call) {
            assert(call.src.length === 3, "expected output and input custom CALL arguments");
            return [null, grad];
          }
          const out = Tensor.empty([4], { dtype: "float32" });
          const x = new Tensor([1, 2, 3, 4], { dtype: "float32" });
          const y = out.customKernel(x, { fxn: identityKernel, gradFxn: backwardIdentity })[0];
          await y.sum().backward();
          assertClose(await out.grad.toArray(), [1, 1, 1, 1]);
          assertClose(await x.grad.toArray(), [1, 1, 1, 1]);
          assertClose(await y.grad.toArray(), [1, 1, 1, 1]);
        });
        await test("customKernel duplicate output alias passes one accumulated upstream", async () => {
          function identityKernel(out0, out1, x2) {
            out0 = out0.flatten();
            out1 = out1.flatten();
            x2 = x2.flatten();
            const i = pg.uop.range(out0.numel(), 0);
            return out0.index(i).store(x2.index(i)).end(i).sink({ arg: new pg.uop.KernelInfo("duplicate_output_grad") });
          }
          const callbackCounts = [];
          function backwardIdentity(...args) {
            const call = args.pop();
            callbackCounts.push(args.length);
            return [null, null, args[0]];
          }
          const out = Tensor.empty([4], { dtype: "float32" });
          const x = new Tensor([1, 2, 3, 4], { dtype: "float32" });
          const [y0, y1] = out.customKernel(out, x, { fxn: identityKernel, gradFxn: backwardIdentity });
          assert(y0.uop.key === y1.uop.key, "duplicate output aliases should share one AFTER");
          await y0.sum().add(y1.sum()).backward();
          assert(callbackCounts.length === 1 && callbackCounts[0] === 1, "expected one accumulated callback upstream");
          assertClose(await x.grad.toArray(), [2, 2, 2, 2]);
        });
        await test("customKernel without gradFxn rejects a needed input gradient", async () => {
          function identityKernel(out2, x2) {
            out2 = out2.flatten();
            x2 = x2.flatten();
            const i = pg.uop.range(out2.numel(), 0);
            return out2.index(i).store(x2.index(i)).end(i).sink({ arg: new pg.uop.KernelInfo("missing_grad_fxn") });
          }
          const out = Tensor.empty([4], { dtype: "float32" });
          const x = new Tensor([1, 2, 3, 4], { dtype: "float32" });
          const y = out.customKernel(x, identityKernel)[0];
          let threw = false;
          try {
            await y.sum().backward();
          } catch (e) {
            threw = String(e.message || e).includes("expected TUPLE body for gradient, got Ops.SINK");
          }
          assert(threw, "missing gradFxn should reject an opaque CALL input gradient");
          assert(x.grad === null, "failed backward must not assign x.grad");
          assert(y.grad === null, "failed backward must not assign y.grad");
        });
        await test("customKernel callback is inactive behind stop-gradient ops", async () => {
          function identityKernel(out2, x2) {
            out2 = out2.flatten();
            x2 = x2.flatten();
            const i = pg.uop.range(out2.numel(), 0);
            return out2.index(i).store(x2.index(i)).end(i).sink({ arg: new pg.uop.KernelInfo("stopped_custom_grad") });
          }
          let out = Tensor.empty([4], { dtype: "float32" });
          let x = new Tensor([1, 2, 3, 4], { dtype: "float32" });
          let y = out.customKernel(x, identityKernel)[0];
          await y.detach().sum().backward();
          assertClose(await x.grad.toArray(), [0, 0, 0, 0]);
          assertClose(await y.grad.toArray(), [0, 0, 0, 0]);
          const calls = [];
          function backwardIdentity(grad, call) {
            calls.push(grad.op);
            return [null, new Tensor(grad).add(7).uop];
          }
          out = Tensor.empty([4], { dtype: "float32" });
          x = new Tensor([1, 2, 3, 4], { dtype: "float32" });
          y = out.customKernel(x, { fxn: identityKernel, gradFxn: backwardIdentity })[0];
          await y.lt(0).cast("float32").sum().backward();
          assert(calls.length === 0, "stop-gradient ops must not invoke custom callbacks");
          assertClose(await x.grad.toArray(), [0, 0, 0, 0]);
          assertClose(await y.grad.toArray(), [0, 0, 0, 0]);
        });
        await test("customKernel reuses buffers after input update", async () => {
          function addKernel(c2, a2, b2) {
            c2 = c2.flatten();
            a2 = a2.flatten();
            b2 = b2.flatten();
            const i = pg.uop.range(c2.numel(), 0);
            return c2.index(i).store(a2.index(i).add(b2.index(i))).end(i).sink(
              new pg.uop.KernelInfo("custom_add_reuse_4")
            );
          }
          const a = Tensor.empty([4], { dtype: "float32" });
          const b = new Tensor([10, 20, 30, 40], { dtype: "float32" });
          const c = Tensor.empty([4], { dtype: "float32" });
          const runs = [
            [new Float32Array([1, 2, 3, 4]), [11, 22, 33, 44]],
            [new Float32Array([5, 6, 7, 8]), [15, 26, 37, 48]]
          ];
          for (const [vals, expected] of runs) {
            a.copyFrom(vals);
            const out = c.customKernel(a, b, addKernel)[0];
            assert(out.uopLogical, "custom output should keep a logical root");
            await out.realize();
            assert(out.uopPhysical && out.uopPhysical.hasBufferIdentity(), "custom output should realize to a buffer-backed root");
            assertClose(await out.toArray(), expected);
          }
        });
        await test("customKernel exposes UOp compare where and unary methods", async () => {
          function selectKernel(out2, a2, b2) {
            out2 = out2.flatten();
            a2 = a2.flatten();
            b2 = b2.flatten();
            const i = pg.uop.range(out2.numel(), 0);
            const av = a2.index(i);
            const bv = b2.index(i);
            const selected = av.lt(0).where(av.neg(), av.max(bv));
            return out2.index(i).store(selected).end(i).sink(
              new pg.uop.KernelInfo("custom_select")
            );
          }
          const out = Tensor.empty([4], { dtype: "float32" });
          const a = new Tensor([-3, 2, 5, -1], { dtype: "float32" });
          const b = new Tensor([1, 4, 3, 9], { dtype: "float32" });
          assertClose(await out.customKernel(a, b, selectKernel)[0].toArray(), [3, 4, 5, 1]);
        });
        await test("customKernel rejects bool INDEX coordinate before codegen", async () => {
          function invalidIndexKernel(out2) {
            out2 = out2.flatten();
            const zero = pg.uop.constant(0);
            const gate = zero.lt(1);
            const bad = out2.index(gate);
            assert(bad !== null, "UOp.index(bool) construction should match tinygrad");
            return bad.store(out2.index(zero)).sink(
              new pg.uop.KernelInfo("invalid_bool_index")
            );
          }
          const out = Tensor.empty([1], { dtype: "float32" });
          let threw = false;
          try {
            await out.customKernel(invalidIndexKernel)[0].toArray();
          } catch (e) {
            threw = true;
          }
          assert(threw, "invalid bool INDEX coordinate must fail before codegen");
        });
        await test("customKernel exposes tinygrad-style floor div and mod", async () => {
          function divKernel(out2, x2, y2) {
            out2 = out2.flatten();
            x2 = x2.flatten();
            y2 = y2.flatten();
            const i = pg.uop.range(x2.numel(), 0);
            const q = x2.index(i).floordiv(y2.index(i));
            const r = x2.index(i).floormod(y2.index(i));
            return out2.index(i).store(q).group(out2.index(i.add(x2.numel())).store(r)).end(i).sink(new pg.uop.KernelInfo("custom_signed_div_mod"));
          }
          const x = new Tensor(new Int32Array([-7, -7, 7, 7, -1, 1, 0]), { dtype: "int32" });
          const y = new Tensor(new Int32Array([3, -3, -3, 3, 4, -4, 3]), { dtype: "int32" });
          const out = Tensor.empty([14], { dtype: "int32" });
          assertClose(
            await out.customKernel(x, y, divKernel)[0].toArray(),
            [-3, 2, -3, 2, -1, -1, 0, 2, -1, -2, 1, 3, -3, 0]
          );
        });
        await test("customKernel numeric literals follow float operand dtype", async () => {
          function literalKernel(out2, x) {
            out2 = out2.flatten();
            x = x.flatten();
            const i = pg.uop.range(x.numel(), 0);
            const xv = x.index(i);
            const sameAdd = xv.sub(1).div(xv.add(1));
            const sameMul = xv.mul(2).div(xv.mul(3));
            assert(pg.uop.dtype(xv.add(1).src[1]) === "weakfloat", "float scalar promotion should retain weakfloat");
            assert(pg.uop.dtype(pg.uop.constant(true)) === "bool", "boolean literals should retain bool");
            assert(pg.uop.dtype(pg.uop.constant(1, "float32")) === "float32", "typed int should convert to float");
            assert(pg.uop.dtype(pg.uop.constant(1.75, "int32")) === "int32", "typed float should convert to int");
            const s0 = out2.index(i).store(sameAdd);
            const s1 = out2.index(i.add(x.numel())).store(sameMul);
            return s0.group(s1).end(i).sink(new pg.uop.KernelInfo("custom_numeric_literals"));
          }
          const xData = new Float32Array(16);
          for (let i = 0; i < xData.length; i++) xData[i] = i / 10 + 1;
          const out = Tensor.empty([32], { dtype: "float32" });
          const got = await out.customKernel(new Tensor(xData, { dtype: "float32" }), literalKernel)[0].toArray();
          const expected = [];
          for (const x of xData) expected.push((x - 1) / (x + 1));
          for (const x of xData) expected.push(x * 2 / (x * 3));
          assertClose(got, expected, 1e-6);
        });
        await test("customKernel descriptor branches keep indexed placeholders typed", async () => {
          function termKernel(out2, x, fa2, fb2, op2, p02, p12) {
            out2 = out2.flatten();
            x = x.flatten();
            fa2 = fa2.flatten();
            fb2 = fb2.flatten();
            op2 = op2.flatten();
            p02 = p02.flatten();
            p12 = p12.flatten();
            const rows = 4;
            const cols = 3;
            const idx = pg.uop.range(out2.numel(), 0);
            const c = idx.floordiv(rows);
            const r = idx.mod(rows);
            const a = x.index(r.mul(cols).add(fa2.index(c)));
            const b = x.index(r.mul(cols).add(fb2.index(c)));
            assert(pg.uop.dtype(a) === "float32", `indexed placeholder should be float32, got ${pg.uop.dtype(a)}`);
            const ab = b.lt(0).where(b.neg(), b);
            const safeDen = ab.lt(1e-6).where(b.lt(0).where(-1e-6, 1e-6), b);
            const z = a.mul(p02.index(c)).add(p12.index(c));
            const opv = op2.index(c);
            let v = a.add(b);
            v = opv.eq(1).where(a.sub(b), v);
            v = opv.eq(2).where(a.mul(b), v);
            v = opv.eq(3).where(a.div(safeDen), v);
            v = opv.eq(4).where(z.sin(), v);
            return out2.index(idx).store(v).end(idx).sink(
              new pg.uop.KernelInfo("custom_descriptor_branches")
            );
          }
          const xData = new Float32Array([
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12
          ]);
          const fa = new Int32Array([0, 1]);
          const fb = new Int32Array([1, 2]);
          const op = new Int32Array([0, 1]);
          const p0 = new Float32Array([1.25, -0.5]);
          const p1 = new Float32Array([0.1, 0.2]);
          const expected = [];
          for (let c = 0; c < 2; c++) {
            for (let r = 0; r < 4; r++) {
              const a = xData[r * 3 + fa[c]];
              const b = xData[r * 3 + fb[c]];
              expected.push(c === 0 ? a + b : a - b);
            }
          }
          const out = Tensor.empty([8], { dtype: "float32" });
          const got = out.customKernel(
            new Tensor(xData),
            new Tensor(fa, { dtype: "int32" }),
            new Tensor(fb, { dtype: "int32" }),
            new Tensor(op, { dtype: "int32" }),
            new Tensor(p0),
            new Tensor(p1),
            termKernel
          )[0];
          assertClose(await got.toArray(), expected, 1e-4);
        });
        await test("jit correctness protects output fed back as input", async () => {
          const f = pg.jitAsync((buf, frame) => {
            const joined = buf.shrink([[1, 3]]).cat(frame);
            return [joined.contiguous(), joined.shrink([[0, 1]]).contiguous()];
          });
          try {
            let buf = new Tensor(new Float32Array([0, 1, 2])).contiguous();
            for (let i = 0; i < 6; i++) {
              const expected = Array.from(await buf.toArrayAsync()).slice(1).concat(10 + i);
              const outputs = await f(buf, new Tensor(new Float32Array([10 + i])).contiguous());
              buf = outputs[0];
              assertClose(await buf.toArrayAsync(), expected);
              assertClose(await outputs[1].toArrayAsync(), expected.slice(0, 1));
            }
          } finally {
            await f.dispose();
          }
        });
        await test("jit correctness protects overlapping single-output replay", async () => {
          const n = 8192;
          const f = pg.jitAsync((buf, frame) => buf.shrink([[n / 2, n]]).cat(frame).contiguous());
          try {
            let buf = new Tensor(new Float32Array(n)).contiguous();
            for (let i = 0; i < 5; i++) {
              buf = await f(buf, new Tensor(new Float32Array(n / 2).fill(i + 1)).contiguous());
              const expected = new Float32Array(n).fill(i);
              expected.fill(i + 1, n / 2);
              assertClose(await buf.toArrayAsync(), expected);
            }
          } finally {
            await f.dispose();
          }
        });
        for (const read of ["array", "batch"]) await test(`jit correctness rejects capture ${read} reads and recovers`, async () => {
          const f = pg.jitAsync(async (x2) => {
            if (read === "batch") await Tensor.toTypedArraysAsync(x2);
            else await x2.toArrayAsync();
            return x2.add(1);
          });
          const x = new Tensor(new Float32Array([1, 2]));
          try {
            assertClose(await (await f(x)).toArrayAsync(), [2, 3]);
            let error;
            try {
              await f(x);
            } catch (e) {
              error = e;
            }
            assert(error && /cannot access tensor data during JIT capture/.test(error.message), "capture read must fail explicitly");
          } finally {
            await f.dispose();
          }
          const other = pg.jitAsync((x2) => x2.add(2));
          try {
            for (let i = 0; i < 3; i++) assertClose(await (await other(x)).toArrayAsync(), [3, 4]);
          } finally {
            await other.dispose();
          }
        });
        await test("jit captures customKernel and replays after input update", async () => {
          function addKernel(c, a2, b2) {
            c = c.flatten();
            a2 = a2.flatten();
            b2 = b2.flatten();
            const i = pg.uop.range(c.numel(), 0);
            return c.index(i).store(a2.index(i).add(b2.index(i))).end(i).sink(
              new pg.uop.KernelInfo("jit_custom_add_4")
            );
          }
          const f = pg.jit((a2, b2) => {
            const c = Tensor.empty([4], { dtype: "float32" });
            return c.customKernel(a2, b2, addKernel)[0];
          });
          const a = Tensor.empty([4], { dtype: "float32" });
          const b = new Tensor([10, 20, 30, 40], { dtype: "float32" });
          a.copyFrom(new Float32Array([1, 2, 3, 4]));
          assertClose(await (await f(a, b)).toArray(), [11, 22, 33, 44]);
          assertClose(await (await f(a, b)).toArray(), [11, 22, 33, 44]);
          assert(f.scheduleCount === 1, `expected one captured schedule, got ${f.scheduleCount}`);
          a.copyFrom(new Float32Array([5, 6, 7, 8]));
          assertClose(await (await f(a, b)).toArray(), [15, 26, 37, 48]);
          assert(f.stats().replayCount === 1, "third customKernel call should replay");
        });
        await test("compile captures customKernel and replays after input update", async () => {
          function addKernel(c, a2, b2) {
            c = c.flatten();
            a2 = a2.flatten();
            b2 = b2.flatten();
            const i = pg.uop.range(c.numel(), 0);
            return c.index(i).store(a2.index(i).add(b2.index(i))).end(i).sink(
              new pg.uop.KernelInfo("compile_custom_add_4")
            );
          }
          const a = Tensor.empty([4], { dtype: "float32" });
          const b = new Tensor([10, 20, 30, 40], { dtype: "float32" });
          a.copyFrom(new Float32Array([1, 2, 3, 4]));
          const compiled = await pg.compile((x, y) => {
            const c = Tensor.empty([4], { dtype: "float32" });
            return c.customKernel(x, y, addKernel)[0];
          }, [a, b]);
          assert(compiled.scheduleCount === 1, `expected one captured schedule, got ${compiled.scheduleCount}`);
          a.copyFrom(new Float32Array([5, 6, 7, 8]));
          const out = await compiled.run([a, b]);
          assertClose(await out.toArray(), [15, 26, 37, 48]);
          assert(compiled.stats().runCount === 1, "compiled customKernel run should update runCount");
          compiled.dispose();
        });
        await test("compile captures current custom sum and replays after input update", async () => {
          function sumKernel(out, a2) {
            out = out.flatten();
            a2 = a2.flatten();
            const r = pg.uop.range(8, 0, pg.uop.AxisType.REDUCE);
            let acc = out.index(0).set(0);
            acc = acc.index(0).set(acc.after(r).index(0).add(a2.index(r)), r);
            return acc.sink(new pg.uop.KernelInfo({ name: "custom_sum_8", opts_to_apply: [] }));
          }
          const a = Tensor.empty([8], { dtype: "float32" });
          a.copyFrom(new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]));
          const compiled = await pg.compile((x) => {
            const out = Tensor.empty([1], { dtype: "float32" });
            return out.customKernel(x, sumKernel)[0];
          }, [a]);
          assert(compiled.scheduleCount === 1, `expected one captured schedule, got ${compiled.scheduleCount}`);
          assertClose(await (await compiled.run([a])).toArray(), [36]);
          a.copyFrom(new Float32Array([2, 3, 4, 5, 6, 7, 8, 9]));
          assertClose(await (await compiled.run([a])).toArray(), [44]);
          assert(compiled.stats().runCount === 2, "compiled custom reduction should replay twice");
          compiled.dispose();
        });
        await test("compile captures current customKernel multi-output addmul", async () => {
          function addmulKernel(out0, out1, a2, b2) {
            out0 = out0.flatten();
            out1 = out1.flatten();
            a2 = a2.flatten();
            b2 = b2.flatten();
            const i = pg.uop.range(4, 0);
            const st0 = out0.index(i).store(a2.index(i).add(b2.index(i)));
            const st1 = out1.index(i).store(a2.index(i).mul(b2.index(i)));
            return st0.group(st1).end(i).sink(
              new pg.uop.KernelInfo("custom_addmul_4")
            );
          }
          const a = Tensor.empty([4], { dtype: "float32" });
          const b = new Tensor([1, 2, 3, 4], { dtype: "float32" });
          a.copyFrom(new Float32Array([1, 2, 3, 4]));
          const compiled = await pg.compile((x, y) => {
            const out0 = Tensor.empty([4], { dtype: "float32" });
            const out1 = Tensor.empty([4], { dtype: "float32" });
            const outs = out0.customKernel(out1, x, y, addmulKernel);
            return [outs[0], outs[1]];
          }, [a, b]);
          const got0 = await compiled.run([a, b]);
          assertClose(await got0[0].toArray(), [2, 4, 6, 8]);
          assertClose(await got0[1].toArray(), [1, 4, 9, 16]);
          a.copyFrom(new Float32Array([2, 3, 4, 5]));
          const got1 = await compiled.run([a, b]);
          assertClose(await got1[0].toArray(), [3, 5, 7, 9]);
          assertClose(await got1[1].toArray(), [2, 6, 12, 20]);
          assert(compiled.stats().runCount === 2, "compiled custom multi-output reduction should replay twice");
          compiled.dispose();
        });
        await test("compile captures customKernel tinygrad-style set accumulator reduction", async () => {
          function sumKernel(out, x2) {
            out = out.flatten();
            x2 = x2.flatten();
            const candidates = 4;
            const rows = 64;
            const c = pg.uop.range(candidates, 0);
            const r = pg.uop.range(rows, 1, pg.uop.AxisType.REDUCE);
            let acc = out.index(c).set(0);
            acc = acc.index(c).set(acc.after(r).index(c).add(x2.index(c.mul(rows).add(r))), r);
            return acc.end(c).sink(new pg.uop.KernelInfo({ name: "custom_sum_4_64", opts_to_apply: [] }));
          }
          const xData = Float32Array.from({ length: 256 }, (_, i) => i + 1);
          const expected = [];
          for (let c = 0; c < 4; c++) {
            let s = 0;
            for (let r = 0; r < 64; r++) s += xData[c * 64 + r];
            expected.push(s);
          }
          const x = new Tensor(xData);
          const compiled = await pg.compile((tx) => {
            const out = Tensor.empty([4], { dtype: "float32" });
            return out.customKernel(tx, sumKernel)[0];
          }, [x]);
          assertClose(await (await compiled.run([x])).toArray(), expected);
          compiled.dispose();
        });
        await test("compiled customKernel consumer reads producer output after readback", async () => {
          const n = 1024;
          function producerKernel(out, x2) {
            out = out.flatten();
            x2 = x2.flatten();
            const i = pg.uop.range(out.numel(), 0);
            return out.index(i).store(x2.index(i).mul(2).add(1)).end(i).sink(
              new pg.uop.KernelInfo({ name: "custom_producer_readback_rebind", opts_to_apply: [] })
            );
          }
          function consumerKernel(out, y) {
            out = out.flatten();
            y = y.flatten();
            const r = pg.uop.range(n, 0, pg.uop.AxisType.REDUCE);
            let acc = out.index(0).set(0);
            acc = acc.index(0).set(acc.after(r).index(0).add(y.index(r)), r);
            return acc.sink(new pg.uop.KernelInfo({
              name: "custom_consumer_readback_rebind",
              opts_to_apply: []
            }));
          }
          const x0 = Float32Array.from({ length: n }, (_, i) => i / 17);
          const x1 = Float32Array.from({ length: n }, (_, i) => 10 + i / 11);
          const x = new Tensor(x0, { dtype: "float32" });
          const producer = await pg.compile((tx) => {
            const out = Tensor.empty([n], { dtype: "float32" });
            return out.customKernel(tx, producerKernel)[0];
          }, [x]);
          const firstY = await producer.run([x]);
          await firstY.realize();
          const consumer = await pg.compile((ty) => {
            const out = Tensor.empty([1], { dtype: "float32" });
            return out.customKernel(ty, consumerKernel)[0];
          }, [firstY]);
          const first = await consumer.run([firstY]);
          assertClose(await first.toArray(), [Array.from(x0).reduce((s, v) => s + v * 2 + 1, 0)], 0.01);
          x.copyFrom(x1);
          const secondY = await producer.run([x]);
          const second = await consumer.run([secondY]);
          assertClose(await second.toArray(), [Array.from(x1).reduce((s, v) => s + v * 2 + 1, 0)], 0.01);
          producer.dispose();
          consumer.dispose();
        });
        await test("runtime exposes conservative stats and capability checks", async () => {
          assert(typeof pg.stats === "function", "runtime should expose stats()");
          assert(typeof pg.canRun === "function", "runtime should expose canRun()");
          const stats = pg.stats();
          assert(stats.core === pg.core, "runtime stats should include core");
          assert(stats.device === pg.device, "runtime stats should include device");
          assert(stats.coreStats && typeof stats.coreStats.launchCount === "number", "runtime stats should include core counters");
          assert(typeof stats.coreStats.globalOps === "number", "runtime stats should include globalOps");
          assert(typeof stats.coreStats.globalMem === "number", "runtime stats should include globalMem");
          assert(typeof stats.coreStats.timeSumS === "number", "runtime stats should include timeSumS");
          assert(typeof stats.coreStats.kernelCount === "number", "runtime stats should include kernelCount");
          assert(typeof stats.coreStats.memUsed === "number", "runtime stats should include memUsed");
          assert(stats.jit && typeof stats.jit.liveCount === "number", "runtime stats should include jit live count");
          assert(typeof pg.resetCounters === "function", "runtime should expose resetCounters()");
          const liveMem = stats.coreStats.memUsed;
          pg.resetCounters();
          const before = pg.stats().coreStats;
          assert(
            before.globalOps === 0 && before.globalMem === 0 && before.kernelCount === 0,
            "resetCounters should clear execution counters"
          );
          assert(before.memUsed === liveMem, "resetCounters should preserve live memory");
          const x = Tensor.empty([3], { dtype: "float32" });
          x.copyFrom(new Float32Array([1, 2, 3]));
          const written = pg.stats().coreStats;
          assert(written.bufferWriteCount === before.bufferWriteCount + 1, "host write should count exactly once");
          assert(written.bufferWriteBytes === before.bufferWriteBytes + 12, "host write should count exactly 12 bytes");
          const t = await x.add(1).realize();
          assertClose(await t.toArray(), [2, 3, 4]);
          const after = pg.stats().coreStats;
          assert(after.bufferWriteBytes >= before.bufferWriteBytes + 12, "stats should count host writes");
          assert(after.bufferReadBytes >= before.bufferReadBytes + 12, "stats should count host reads");
          assert(after.launchCount >= before.launchCount + 1, "stats should count backend launches");
          const expectedOps = pg.device === "x86" ? 0 : 3;
          const expectedMem = pg.device === "x86" ? 0 : 24;
          assert(after.globalOps === expectedOps, `expected ${expectedOps} global ops, got ${after.globalOps}`);
          assert(after.globalMem === expectedMem, `expected ${expectedMem} global memory bytes, got ${after.globalMem}`);
          assert(after.kernelCount === 1, `expected one tracked call, got ${after.kernelCount}`);
          x.copyFrom(new Float32Array([4, 5, 6]));
          const rewritten = pg.stats().coreStats;
          assert(rewritten.bufferWriteCount === after.bufferWriteCount + 1, "warm write should count exactly once");
          assert(rewritten.bufferWriteBytes === after.bufferWriteBytes + 12, "warm write should count exactly 12 bytes");
          assertClose(await x.toArray(), [4, 5, 6]);
          assert(pg.canRun({ dtype: "float32" }), "float32 should be supported by every current runtime");
          if (pg.caps.f64 === false) {
            assert(!pg.canRun({ dtype: "float64" }), "canRun should reject f64 when caps.f64 is false");
          }
          assert(pg.canRun({ op: "add", dtype: "float32", shape: [4] }), "canRun should probe add");
          assert(
            pg.canRun({ op: "matmul", dtype: "float32", shapes: [[2, 3], [3, 4]] }),
            "canRun should probe matmul shapes"
          );
          assert(pg.canRun({ op: "gather", dtype: "float32", shape: [2, 3] }), "canRun should probe gather");
          assert(pg.canRun({ op: "sort", dtype: "float32", shape: [2, 3] }), "canRun should probe sort");
          assert(pg.canRun({ op: "argsort", dtype: "float32", shape: [2, 3] }), "canRun should probe argsort");
          assert(pg.canRun({ op: "topk", dtype: "float32", shape: [2, 3] }), "canRun should probe topk");
          let threw = false;
          try {
            pg.canRun({ shape: [4], dtype: "float32" });
          } catch (e) {
            threw = String(e.message || e).includes("require an op");
          }
          assert(threw, "shape-only canRun queries should fail explicitly");
        });
        await test("runtime capability probe reports construction budgets", async () => {
          const x = new Tensor([1, 2]);
          assertClose(await x.add(1).toArray(), [2, 3]);
          assert(pg.canRun({ op: "add", dtype: "float32", shape: [2] }), "add should compile");
          let budgetError;
          try {
            pg.canRun({ op: "qr", dtype: "float32", shape: [17, 17] });
          } catch (e) {
            budgetError = e;
          }
          assert(budgetError && /cannot prove/.test(budgetError.message), "probe budget is not unsupported");
        });
        await test("runtime compile wrapper warms capture and replays", async () => {
          assert(typeof pg.compile === "function", "runtime should expose pg.compile");
          const sample = new Tensor(new Float32Array([1, 2, 3]));
          const compiled = await pg.compile((x) => x.add(1), [sample]);
          assert(compiled.scheduleCount === 1, `expected one captured schedule, got ${compiled.scheduleCount}`);
          const out = await compiled.run([new Tensor(new Float32Array([10, 20, 30]))]);
          assertClose(await out.toArray(), [11, 21, 31]);
          const stats = compiled.stats();
          assert(stats.captureRuns === 2, "compile should perform two setup runs");
          assert(stats.runCount === 1, "compiled run should update runCount");
          compiled.dispose();
        });
        await test("toTypedArray aliases flat typed readback", async () => {
          const t = new Tensor([1, 2, 3]);
          const arr = await t.toTypedArray();
          assert(arr instanceof Int32Array, `expected Int32Array, got ${arr.constructor.name}`);
          assertClose(arr, [1, 2, 3]);
        });
        await test("toTypedArrays batches flat typed readback", async () => {
          const t = new Tensor([1, 2, 3]);
          const outs = await Tensor.toTypedArrays(t.add(1), t.mul(2));
          assert(Array.isArray(outs) && outs.length === 2, "expected two output arrays");
          assert(outs[0] instanceof Int32Array, `expected Int32Array, got ${outs[0].constructor.name}`);
          assertClose(outs[0], [2, 3, 4]);
          assertClose(outs[1], [2, 4, 6]);
          const i32 = new Tensor(new Int32Array([1, 2, 3]), { dtype: "int32" });
          const empty = Tensor.zeros([0]);
          assertShape(empty.shape, [0]);
          const more = await Tensor.toTypedArrays([i32, empty]);
          assert(more[0] instanceof Int32Array, `expected Int32Array, got ${more[0].constructor.name}`);
          assert(more[1] instanceof Float32Array && more[1].length === 0, "expected empty Float32Array");
          assertClose(more[0], [1, 2, 3]);
        });
        await test("empty rejects named tensor keyword like tinygrad", async () => {
          let threw = false;
          try {
            Tensor.empty([2, 3], { name: "z" });
          } catch (_) {
            threw = true;
          }
          assert(threw, "expected Tensor.empty(..., {name}) to reject");
        });
        console.log("\n-- Elementwise --");
        await test("add", async () => {
          const a = new Tensor([1, 2, 3]);
          const b = new Tensor([4, 5, 6]);
          assertClose(await a.add(b).toArray(), [5, 7, 9]);
        });
        await test("sub", async () => {
          const a = new Tensor([10, 20, 30]);
          const b = new Tensor([1, 2, 3]);
          const out = a.sub(b);
          assert(out.uop.op === pg._core.ops.ADD, "subtraction root must be ADD");
          assert(out.uop.src[1].op === pg._core.ops.MUL, "subtraction rhs must be negating MUL");
          assertClose(await out.toArray(), [9, 18, 27]);
        });
        await test("mul", async () => {
          const a = new Tensor([2, 3, 4]);
          const b = new Tensor([5, 6, 7]);
          assertClose(await a.mul(b).toArray(), [10, 18, 28]);
        });
        await test("div", async () => {
          const a = new Tensor([10, 20, 30]);
          const b = new Tensor([2, 4, 5]);
          assertClose(await a.div(b).toArray(), [5, 5, 6]);
        });
        await test("div rounding modes match tinygrad topology", async () => {
          const ints = new Tensor([-7, -4, 4, 7], { dtype: "int32" });
          const intDivisors = new Tensor([3, -3, 3, -3], { dtype: "int32" });
          const truncInt = ints.div(intDivisors, "trunc");
          const floorInt = ints.div(intDivisors, "floor");
          assert(truncInt.uop.op === pg._core.ops.CDIV, "integer trunc division must be CDIV");
          assert(floorInt.uop.op === pg._core.ops.FLOORDIV, "integer floor division must be FLOORDIV");
          assertClose(await truncInt.toArray(), [-2, 1, 1, -2]);
          assertClose(await floorInt.toArray(), [-3, 1, 1, -3]);
          const floats = new Tensor([-7.5, -4.5, 4.5, 7.5], { dtype: "float32" });
          const floatDivisors = new Tensor([2, -2, 2, -2], { dtype: "float32" });
          const truncFloat = floats.div(floatDivisors, "trunc");
          const floorFloat = floats.div(floatDivisors, "floor");
          assert(truncFloat.uop.op === pg._core.ops.TRUNC, "float trunc division must end in TRUNC");
          assert(floorFloat.uop.op === pg._core.ops.WHERE, "float floor division must end in WHERE");
          assertClose(await truncFloat.toArray(), [-3, 2, 2, -3]);
          assertClose(await floorFloat.toArray(), [-4, 2, 2, -4]);
          let threw = false;
          try {
            ints.div(intDivisors, "nearest");
          } catch (err) {
            threw = /rounding_mode='nearest' is not supported/.test(String(err.message));
          }
          assert(threw, "unsupported division rounding mode must throw");
        });
        await test("neg", async () => {
          const a = new Tensor([1, -2, 3]);
          const out = a.neg();
          assert(out.uop.op === pg._core.ops.MUL, "negation root must be MUL");
          assertClose(await out.toArray(), [-1, 2, -3]);
        });
        await test("scalar add", async () => {
          const a = new Tensor([1, 2, 3]);
          assertClose(await a.add(10).toArray(), [11, 12, 13]);
        });
        await test("scalar mul", async () => {
          const a = new Tensor([1, 2, 3]);
          assertClose(await a.mul(3).toArray(), [3, 6, 9]);
        });
        await test("chain: (a + 2) * b", async () => {
          const a = new Tensor([1, 2, 3]);
          const b = new Tensor([4, 5, 6]);
          assertClose(await a.add(2).mul(b).toArray(), [12, 20, 30]);
        });
        console.log("\n-- Composed math --");
        await test("exp", async () => {
          const a = new Tensor([0, 1]);
          const arr = await a.exp().toArray();
          assertClose(arr, [1, Math.E], 1e-3);
        });
        await test("log", async () => {
          const a = new Tensor([1, Math.E]);
          const arr = await a.log().toArray();
          assertClose(arr, [0, 1], 1e-3);
        });
        await test("logaddexp softplus mish match pinned graph", async () => {
          assert(
            Tensor.prototype._physicalizeResult === void 0,
            "legacy frontend substitution helper must stay deleted"
          );
          const x = new Tensor([
            [-20, -3, -0, 2, 20],
            [1, -1, 4, -4, 0.5]
          ]);
          const other = new Tensor([[-2], [3]]);
          const pinnedLogaddexp = (lhs, rhs) => {
            let b = lhs._ensureTensor(rhs);
            const shape = lhs._broadcastShape(b.shape);
            const a = lhs._broadcastTensor(shape);
            b = b._broadcastTensor(shape);
            const m = a.maximum(b);
            return a.sub(m).exp().add(b.sub(m).exp()).log().add(m);
          };
          const pinnedSoftplus = (value, beta = 1) => pinnedLogaddexp(value.mul(beta), 0).mul(1 / beta, true);
          const pairs = [
            [x.logaddexp(0), pinnedLogaddexp(x, 0)],
            [x.logaddexp(other), pinnedLogaddexp(x, other)],
            [x.softplus(), pinnedSoftplus(x)],
            [x.softplus(2), pinnedSoftplus(x, 2)],
            [x.mish(), x.mul(pinnedSoftplus(x).tanh())]
          ];
          for (const [actual, expected] of pairs) {
            assert(actual.uop.key === expected.uop.key, "physical graph differs");
            assert(actual.uopLogical.key === expected.uopLogical.key, "logical graph differs");
            assertClose(await actual.toArray(), await expected.toArray(), 1e-6);
          }
        });
        await test("where scalar branch shapes before promotion", async () => {
          const cond = new Tensor([[true, false], [false, true]], { dtype: "bool" });
          const out = cond.where(0, -Infinity);
          const zeroBranch = out.uop.src[1];
          assert(zeroBranch.op === pg._core.ops.CONST, "expected scalar CONST branch");
          assert(pg.uop.dtype(zeroBranch) === "weakfloat", "expected promoted weakfloat branch");
          const values = await out.toArray();
          assert(
            values[0] === 0 && values[1] === -Infinity && values[2] === -Infinity && values[3] === 0,
            `unexpected where values ${Array.from(values)}`
          );
        });
        await test("log1p expm1 use core tensor roots", async () => {
          assert(
            Tensor.prototype._physicalizeResult === void 0,
            "legacy frontend substitution helper must stay deleted"
          );
          const x = new Tensor([-1e-6, 0, 1e-6, 0.25], { device: "cpu" });
          const rows = [
            ["log1p", x.log1p(), [-1e-6, 0, 1e-6, 0.25].map(Math.log1p)],
            ["expm1", x.expm1(), [-1e-6, 0, 1e-6, 0.25].map(Math.expm1)]
          ];
          for (const [name, actual, expectedValues] of rows) {
            const rawFn = pg._core.ffi[`poly_${name}`];
            const expectedLogical = rawFn(x._ctx, x.uopLogical.raw);
            const expectedPhysical = rawFn(x._ctx, x.uop.raw);
            assert(
              actual.uopLogical.key === String(pg._core.ffi.poly_uop_key(expectedLogical)),
              `${name} logical graph differs`
            );
            assert(
              actual.uop.key === String(pg._core.ffi.poly_uop_key(expectedPhysical)),
              `${name} physical graph differs`
            );
            assertClose(await actual.toArray(), expectedValues, 1e-6);
          }
          const moved = (await Tensor.empty([4], {
            device: "cpu"
          }).realize()).to("cuda").to("cpu");
          for (const [name, actual] of [
            ["log1p", moved.log1p()],
            ["expm1", moved.expm1()]
          ]) {
            const expected = pg._core.ffi[`poly_${name}`](moved._ctx, moved.uop.raw);
            assert(
              actual.uop.key === String(pg._core.ffi.poly_uop_key(expected)),
              `${name} lost the nested physical occurrence`
            );
          }
        });
        await test("sin cos tan match pinned promotion", async () => {
          const integer = new Tensor([0, 1, 2], { dtype: "int32" });
          for (const [name, actual, expected] of [
            ["sin", integer.sin(), [0, Math.sin(1), Math.sin(2)]],
            ["cos", integer.cos(), [1, Math.cos(1), Math.cos(2)]],
            ["tan", integer.tan(), [0, Math.tan(1), Math.tan(2)]]
          ]) {
            assert(actual.dtype === "float32", `integer ${name} should promote to float32`);
            assertClose(await actual.toArray(), expected, 1e-6);
          }
          assert(
            integer.sin().uop.src[0].op !== pg._core.ops.CAST,
            "integer SIN should retain its original source without a frontend CAST"
          );
          const bool = new Tensor([false, true], { dtype: "bool" });
          assertClose(await bool.sin().toArray(), [0, Math.sin(1)], 1e-6);
          const angles = new Tensor([0, 0.25, 0.5]);
          assertClose(await angles.cos().toArray(), [Math.cos(0), Math.cos(0.25), Math.cos(0.5)], 1e-6);
          assertClose(await angles.tan().toArray(), [Math.tan(0), Math.tan(0.25), Math.tan(0.5)], 1e-6);
        });
        await testIf(supportsF16, "sin cos tan preserve float16 promotion", async () => {
          const half = new Tensor([0, 1]).cast("float16");
          const halfCos = half.cos();
          assert(halfCos.dtype === "float16", "float16 cos should cast back to float16");
          assertClose(await halfCos.cast("float32").toArray(), [1, Math.cos(1)], 2e-3);
        });
        await testIf(supportsF64, "sin cos tan preserve float64 promotion", async () => {
          const angles = new Tensor([0, 0.25, 0.5], { dtype: "float64" });
          const cos = angles.cos();
          const tan = angles.tan();
          assert(cos.dtype === "float64", "float64 cos should retain dtype");
          assert(tan.dtype === "float64", "float64 tan should retain dtype");
          assertClose(await cos.toArray(), [Math.cos(0), Math.cos(0.25), Math.cos(0.5)], 1e-12);
          assertClose(await tan.toArray(), [Math.tan(0), Math.tan(0.25), Math.tan(0.5)], 1e-12);
        });
        await test("sin cos tan preserve nested current occurrence", async () => {
          const x = await new Tensor([0.25], { device: "cpu" }).realize();
          const moved = x.to("cuda").to("cpu");
          assert(
            countGraphOp(moved.sin().uop, pg._core.ops.COPY) === 2,
            "sin should retain the nested COPY source occurrence"
          );
          assert(
            countGraphOp(moved.cos().uop, pg._core.ops.COPY) === 2,
            "cos should retain the nested COPY source occurrence"
          );
          assert(
            countGraphOp(moved.tan().uop, pg._core.ops.COPY) === 2,
            "tan should retain the nested COPY source occurrence"
          );
        });
        await test("sqrt", async () => {
          const a = new Tensor([1, 4, 9, 16]);
          assertClose(await a.sqrt().toArray(), [1, 2, 3, 4]);
        });
        await test("abs", async () => {
          const a = new Tensor([-1, 2, -3]);
          assertClose(await a.abs().toArray(), [1, 2, 3]);
        });
        await test("square", async () => {
          const a = new Tensor([2, 3, 4]);
          assertClose(await a.square().toArray(), [4, 9, 16]);
        });
        await test("named reverse add and mul preserve scalar-first ordering", async () => {
          const moved = (await new Tensor([1, 2], { device: "cpu" }).realize()).to("cuda").to("cpu");
          for (const out of [moved.add(3, true), moved.mul(3, true)]) {
            assert(
              out.uop.src[0].op === pg._core.ops.CONST,
              "reverse scalar should be the first implicit-broadcast operand"
            );
            assert(
              out.uop.src[1].key === moved.uop.key,
              "nested current occurrence should be the second operand"
            );
          }
          const values = new Tensor([1, 2], { device: "cpu" });
          assertClose(await values.add(3, true).toArray(), [4, 5]);
          assertClose(await values.mul(3, true).toArray(), [3, 6]);
        });
        await test("literal elementwise composites match pinned", async () => {
          const values = [-2.5, -1, 0, 0.5, 2.5];
          const x = new Tensor(values);
          const sigmoid = (v) => 1 / (1 + Math.exp(-v));
          const expected = {
            square: values.map((v) => v * v),
            ceil: values.map(Math.ceil),
            floor: values.map(Math.floor),
            sigmoid: values.map(sigmoid),
            tanh: values.map(Math.tanh),
            relu6: values.map((v) => Math.min(Math.max(v, 0), 6)),
            leakyRelu: values.map((v) => v < 0 ? 0.01 * v : v),
            hardswish: values.map((v) => v * Math.min(Math.max(v + 3, 0), 6) / 6),
            hardsigmoid: values.map((v) => Math.min(Math.max(v / 6 + 0.5, 0), 1)),
            hardtanh: values.map((v) => Math.min(Math.max(v, -1), 1)),
            silu: values.map((v) => v * sigmoid(v)),
            elu: values.map((v) => v > 0 ? v : Math.exp(v) - 1),
            sign: values.map(Math.sign),
            abs: values.map(Math.abs),
            isnan: values.map(Number.isNaN)
          };
          for (const [method, wanted] of Object.entries(expected)) {
            assertClose(await x[method]().toArray(), wanted, 2e-6);
          }
          assertClose(
            await x.hardsigmoid(0.2, 0.3).toArray(),
            values.map((v) => Math.min(Math.max(0.2 * v + 0.3, 0), 1)),
            2e-6
          );
          const ints = new Tensor(new Int32Array([-2, 0, 3]), { dtype: "int32" });
          assertClose(await ints.sign().toArray(), [-1, 0, 1]);
          assertClose(await ints.abs().toArray(), [2, 0, 3]);
          const bools = new Tensor([false, true], { dtype: "bool" });
          assertClose(await bools.sign().toArray(), [0, 1]);
          assertClose(await bools.abs().toArray(), [0, 1]);
          const constant = x.constLike(1);
          assert(constant.dtype === x.dtype, "constLike should retain dtype");
          assertShape(constant.shape, x.shape);
          assert(constant.uop.op === pg._core.ops.EXPAND, "constLike should remain a CONST graph");
          assertClose(await constant.toArray(), values.map(() => 1));
          for (const method of ["ceil", "floor"]) {
            const gradInput = new Tensor(values, {});
            await gradInput[method]().sum().backward();
            assertClose(await gradInput.grad.toArray(), values.map(() => 0));
          }
          const moved = (await new Tensor([0.5], { device: "cpu" }).realize()).to("cuda").to("cpu");
          for (const method of ["square", "sigmoid", "tanh", "relu6", "sign", "abs"]) {
            assert(
              countGraphOp(moved[method]().uop, pg._core.ops.COPY) === 2,
              `${method} should retain the nested current occurrence`
            );
          }
        });
        await testIf(supportsF16, "literal composites preserve float16 promotion", async () => {
          const values = [-2.5, -1, 0, 0.5, 2.5];
          const half = new Tensor(values).cast("float16");
          assert(half.sigmoid().dtype === "float16", "float16 sigmoid should retain dtype");
          assert(half.tanh().dtype === "float16", "float16 tanh should retain dtype");
          assertClose(await half.tanh().cast("float32").toArray(), values.map(Math.tanh), 2e-3);
        });
        await test("round and isinf match pinned compositions", async () => {
          const values = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5];
          const expectedRound = [-2, -2, 0, 0, 2, 2];
          const rounded32 = new Tensor(values).round();
          assert(rounded32.dtype === "float32", `expected float32, got ${rounded32.dtype}`);
          assertClose(await rounded32.toArray(), expectedRound);
          if (supportsF16) {
            const rounded16 = new Tensor(values).cast("float16").round();
            assert(rounded16.dtype === "float16", `expected float16, got ${rounded16.dtype}`);
            assertClose(await rounded16.cast("float32").toArray(), expectedRound);
          }
          if (supportsF64) {
            const rounded64 = new Tensor(values).cast("float64").round();
            assert(rounded64.dtype === "float64", `expected float64, got ${rounded64.dtype}`);
            assertClose(await rounded64.toArray(), expectedRound);
          }
          const roundedInt = new Tensor(
            new Int32Array([-2, -1, 0, 1, 2]),
            { dtype: "int32" }
          ).round();
          assert(roundedInt.dtype === "weakfloat", `expected weakfloat, got ${roundedInt.dtype}`);
          assertClose(await roundedInt.toArray(), [-2, -1, 0, 1, 2]);
          const roundedBool = new Tensor([false, true], { dtype: "bool" }).round();
          assert(roundedBool.dtype === "weakfloat", `expected weakfloat, got ${roundedBool.dtype}`);
          assertClose(await roundedBool.toArray(), [0, 1]);
          const infinityValues = new Tensor(
            [-Infinity, -1, -0, 0, 1, Infinity, Number.NaN]
          );
          for (const [detectPositive, detectNegative, expected] of [
            [false, false, [false, false, false, false, false, false, false]],
            [false, true, [true, false, false, false, false, false, false]],
            [true, false, [false, false, false, false, false, true, false]],
            [true, true, [true, false, false, false, false, true, false]]
          ]) {
            const result = infinityValues.isinf(detectPositive, detectNegative);
            assert(result.dtype === "bool", `expected bool, got ${result.dtype}`);
            assertClose(await result.toArray(), expected);
          }
          for (const tensor of [
            new Tensor([0, 1]),
            new Tensor(new Int32Array([0, 1]), { dtype: "int32" }),
            new Tensor([false, true], { dtype: "bool" })
          ]) {
            const result = tensor.isinf();
            assert(result.dtype === "bool", `expected bool, got ${result.dtype}`);
            assertClose(await result.toArray(), [false, false]);
          }
          const moved = (await new Tensor([0.5], { device: "cpu" }).realize()).to("cuda").to("cpu");
          assert(
            countGraphOp(moved.round().uop, pg._core.ops.COPY) === 2,
            "round should retain the nested current occurrence"
          );
          for (const [detectPositive, detectNegative] of [
            [false, false],
            [false, true],
            [true, false],
            [true, true]
          ]) {
            assert(
              countGraphOp(
                moved.isinf(detectPositive, detectNegative).uop,
                pg._core.ops.COPY
              ) === 2,
              "isinf should retain the nested current occurrence"
            );
          }
        });
        await test("sigmoid", async () => {
          const a = new Tensor([0]);
          const arr = await a.sigmoid().toArray();
          assertClose(arr, [0.5], 1e-3);
        });
        console.log("\n-- Activations --");
        await test("relu", async () => {
          const a = new Tensor([-1, 0, 1, 2]);
          assertClose(await a.relu().toArray(), [0, 0, 1, 2]);
        });
        await test("gelu", async () => {
          const values = [-2.5, -1, 0, 0.5, 2.5];
          const expected = values.map(
            (x) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)))
          );
          assertClose(await new Tensor(values).gelu().toArray(), expected, 2e-6);
          if (supportsF16) {
            const half = new Tensor(values).cast("float16").gelu();
            assert(half.dtype === "float16", `expected float16, got ${half.dtype}`);
            assertClose(await half.cast("float32").toArray(), expected, 2e-3);
          }
          const moved = (await new Tensor(values, { device: "cpu" }).realize()).to("cuda").to("cpu").gelu();
          assert(
            countGraphOp(moved.uop, pg._core.ops.COPY) === 2,
            "gelu should retain the nested current occurrence"
          );
        });
        await test("quick gelu", async () => {
          const values = [-2.5, -1, 0, 0.5, 2.5];
          const expected = values.map((x) => x / (1 + Math.exp(-(1.702 * x))));
          assertClose(await new Tensor(values).quickGelu().toArray(), expected, 2e-6);
          if (supportsF16) {
            const half = new Tensor(values).cast("float16").quickGelu();
            assert(half.dtype === "float16", `expected float16, got ${half.dtype}`);
            assertClose(await half.cast("float32").toArray(), expected, 25e-4);
          }
          const moved = (await new Tensor(values, { device: "cpu" }).realize()).to("cuda").to("cpu").quickGelu();
          assert(
            countGraphOp(moved.uop, pg._core.ops.COPY) === 2,
            "quickGelu should retain the nested current occurrence"
          );
        });
        await test("silu", async () => {
          const a = new Tensor([0]);
          const arr = await a.silu().toArray();
          assert(Math.abs(arr[0]) < 0.01, `silu(0) should be ~0, got ${arr[0]}`);
        });
        console.log("\n-- Comparisons --");
        await test("eq", async () => {
          const a = new Tensor([1, 2, 3]);
          const b = new Tensor([1, 5, 3]);
          assertClose(await a.eq(b).toArray(), [1, 0, 1]);
        });
        await test("gt", async () => {
          const a = new Tensor([1, 5, 3]);
          const b = new Tensor([2, 3, 3]);
          assertClose(await a.gt(b).toArray(), [0, 1, 0]);
        });
        await test("mixed dtype comparison and where promote like tinygrad", async () => {
          const x = new Tensor([
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15]
          ]);
          const out = Tensor.full([4, 4], 7).gt(x).where(x, Tensor.full([4, 4], -2)).sum(0);
          assertClose(await out.toArray(), [0, 2, 4, -3]);
        });
        await test("where with optimized-away middle input keeps param slots", async () => {
          const idx = Tensor.arange(2);
          const out = idx.ge(0).where(new Tensor([1, 3]), new Tensor([7, 8]));
          assertClose(await out.toArray(), [1, 3]);
        });
        await test("maximum", async () => {
          const a = new Tensor([1, 5, 3]);
          const b = new Tensor([2, 3, 4]);
          assertClose(await a.maximum(b).toArray(), [2, 5, 4]);
        });
        await test("clamp", async () => {
          const a = new Tensor([1, 5, 3]);
          assertClose(await a.clamp(2, 4).toArray(), [2, 4, 3]);
          const values = new Tensor([-Infinity, -2, 2, Infinity]);
          const minOnly = await values.clamp(-1, void 0).toArray();
          const maxOnly = await values.clamp(void 0, 1).toArray();
          assert(
            minOnly[0] === -1 && minOnly[1] === -1 && minOnly[2] === 2 && minOnly[3] === Infinity,
            `clamp min-only mismatch: ${minOnly}`
          );
          assert(
            maxOnly[0] === -Infinity && maxOnly[1] === -2 && maxOnly[2] === 1 && maxOnly[3] === 1,
            `clamp max-only mismatch: ${maxOnly}`
          );
          const clipped = new Tensor([-3, -0.5, 2]).clip(-1, 1);
          assert(clipped.uop.op === pg._core.ops.WHERE, "clip alias must end in WHERE");
          assertClose(await clipped.toArray(), [-1, -0.5, 1]);
        });
        await test("clamp preserves nested current occurrence", async () => {
          const x = await new Tensor([1], { device: "cpu" }).realize();
          const clamped = x.to("cuda").to("cpu").clamp(-1, 1);
          assert(
            countGraphOp(clamped.uop, pg._core.ops.COPY) === 2,
            "clamp should retain the nested COPY source occurrence"
          );
          assert(
            countGraphOp(clamped.uop, pg._core.ops.WHERE) === 2,
            "two-bound clamp should contain two conditional WHERE nodes"
          );
        });
        console.log("\n-- Movement --");
        await test("movement helpers maximum shape", async () => {
          const t = Tensor.empty(2, 3);
          const before = t.uop.key;
          assertShape(t.maxShape, [2, 3]);
          assert(t.maxNumel() === 6 && t.uop.key === before);
          assertShape(new Tensor(3).maxShape, []);
          assert(new Tensor(3).maxNumel() === 1);
          assert(Tensor.empty(0, 3).maxNumel() === 0);
        });
        await test("movement helpers optional shrink and exact graph", async () => {
          const t = new Tensor([[0, 1, 2], [3, 4, 5]]);
          const y = t.shrink([null, [1, 3]]);
          assert(y.uop.key === t.shrink([[0, 2], [1, 3]]).uop.key);
          assert(y.uop.src.length === 3);
          assertShape(y.shape, [2, 2]);
          assertClose(await y.toArrayAsync(), [1, 2, 4, 5]);
          const empty = t.shrink([[1, 1], null]);
          assertShape(empty.shape, [0, 3]);
          assertClose(await empty.toArrayAsync(), []);
        });
        await test("movement helpers noops preserve tensor identity", async () => {
          const t = new Tensor([[0, 1, 2], [3, 4, 5]]);
          for (const run of [
            (t2) => t2.shrink([[0, 2], [0, 3]]),
            (t2) => t2.shrink([null, null]),
            (t2) => t2.shrinkTo(null, 3),
            (t2) => t2.padTo(null, 3, { value: 5 }),
            (t2) => t2.flip([]),
            (t2) => t2.getitem(),
            (t2) => t2.getitem({ step: 1 })
          ]) assert(run(t) === t);
          assertClose(await t.toArrayAsync(), [0, 1, 2, 3, 4, 5]);
        });
        await test("movement helpers pad shrink to values and graphs", async () => {
          const t = new Tensor([[0, 1, 2], [3, 4, 5]]);
          for (const value of [0, -1, true, 1.5]) {
            const y = t.padTo([3, 5], { value });
            const expected = t.pad([[0, 1], [0, 2]], "constant", value);
            assert(y.uop.key === expected.uop.key);
            assertClose(await y.toArrayAsync(), await expected.toArrayAsync());
          }
          assertClose(await t.shrinkTo(null, 2).toArrayAsync(), [0, 1, 3, 4]);
          assertClose(await t.shrinkTo([1, 2]).toArrayAsync(), [0, 1]);
          assertClose(await Tensor.empty(0, 3).padTo(2, 3, { value: 7 }).toArrayAsync(), [7, 7, 7, 7, 7, 7]);
        });
        await test("movement helpers reject invalid dimensions", async () => {
          const t = Tensor.empty(2, 3);
          for (const run of [
            (t2) => t2.shrink([null]),
            (t2) => t2.shrink([null, null, null]),
            (t2) => t2.shrinkTo(1),
            (t2) => t2.shrinkTo(1, 2, 3),
            (t2) => t2.padTo(3),
            (t2) => t2.padTo(3, 4, 5),
            (t2) => t2.padTo(1, 3)
          ]) {
            let error;
            try {
              run(t);
            } catch (e) {
              error = e;
            }
            assert(error instanceof Error, "invalid shape must reject");
          }
        });
        await test("reshape", async () => {
          const t = new Tensor([1, 2, 3, 4, 5, 6]).reshape(2, 3);
          assertShape(t.shape, [2, 3]);
          assertClose(await t.toArray(), [1, 2, 3, 4, 5, 6]);
        });
        await test("reshape inference validates cardinality", async () => {
          assertShape(Tensor.empty(6).reshape(2, -1).shape, [2, 3]);
          assertShape(Tensor.empty(0).reshape(-1, 3).shape, [0, 3]);
          assertShape(Tensor.empty(0).reshape(1, 0).shape, [1, 0]);
          assertShape(Tensor.empty(2, 3).reshape(null, 3).shape, [2, 3]);
          for (const [shape, target, message] of [
            [[3072], [-1, 3073], "size mismatch"],
            [[5], [2, -1], "size mismatch"],
            [[6], [-1, -1], "only one dimension can be inferred"],
            [[0], [0, -1], "division by zero"]
          ]) {
            let error = null;
            try {
              Tensor.empty(...shape).reshape(...target);
            } catch (e) {
              error = e;
            }
            assert(error && error.message.includes(message), `expected ${message}, got ${error && error.message}`);
          }
        });
        await test("flip", async () => {
          const t = new Tensor([1, 2, 3]);
          assertClose(await t.flip(0).toArray(), [3, 2, 1]);
        });
        await test("permute", async () => {
          const t = new Tensor([[1, 2, 3], [4, 5, 6]]);
          const p = t.permute(1, 0);
          assertShape(p.shape, [3, 2]);
          assertClose(await p.toArray(), [1, 4, 2, 5, 3, 6]);
        });
        await test("expand negative and null keep original dim like tinygrad", async () => {
          const x = Tensor.arange(2, { dtype: "int32" }).reshape(2, 1, 1, 1);
          const y = x.expand(-1, 3, 4, null);
          assertShape(y.shape, [2, 3, 4, 1]);
          assertClose(Array.from(await y.toArray()).slice(0, 4), [0, 0, 0, 0]);
          assertClose(Array.from(await y.toArray()).slice(12, 16), [1, 1, 1, 1]);
          const img = Tensor.arange(2 * 3 * 34 * 34).reshape(2, 3, 34, 34);
          const lowX = Tensor.randint(2, { low: 0, high: 2 }).reshape(2, 1, 1, 1);
          const idxX = Tensor.arange(32, { dtype: "int32" }).reshape(1, 1, 1, 32);
          const cropIdx = lowX.add(idxX).expand(-1, 3, img.shape[2], -1);
          assertShape(cropIdx.shape, [2, 3, 34, 32]);
          assertShape(img.gather(-1, cropIdx).shape, [2, 3, 34, 32]);
        });
        await test("pad", async () => {
          const t = new Tensor([1, 2, 3]);
          const p = t.pad([[1, 1]]);
          assertShape(p.shape, [5]);
          assertClose(await p.toArray(), [0, 1, 2, 3, 0]);
          const x = Tensor.arange(9, { dtype: "float32" }).reshape(1, 1, 3, 3);
          const flat = x.pad([1, 0, 0, 1]);
          assertShape(flat.shape, [1, 1, 4, 4]);
          assertClose(await flat.toArray(), [0, 0, 1, 2, 0, 3, 4, 5, 0, 6, 7, 8, 0, 0, 0, 0]);
          const moved = Tensor.arange(12).reshape(3, 4).pad([[-1, 2], [1, -1]]);
          assertShape(moved.shape, [4, 4]);
          assertClose(await moved.toArray(), [0, 4, 5, 6, 0, 8, 9, 10, 0, 0, 0, 0, 0, 0, 0, 0]);
          const promoted = new Tensor([1, 2], { dtype: "int32" }).pad([[1, 1]], "constant", 5.5);
          assert(promoted.dtype === "weakfloat", `expected weakfloat, got ${promoted.dtype}`);
          assertClose(await promoted.toArray(), [5.5, 1, 2, 5.5]);
          const boolFill = new Tensor([1, 2], { dtype: "int32" }).pad([[1, 1]], "constant", true);
          assert(boolFill.dtype === "int32", `expected int32, got ${boolFill.dtype}`);
          assertClose(await boolFill.toArray(), [1, 1, 2, 1]);
        });
        await test("pad readback preserves non-float dtype", async () => {
          const p = new Tensor([1, 2, 3], { dtype: "int32" }).pad([[1, 1]]);
          assert(p.dtype === "int32", `expected int32, got ${p.dtype}`);
          const arr = await p.toArray();
          assert(arr.constructor.name === "Int32Array", `expected Int32Array, got ${arr.constructor.name}`);
          assertClose(arr, [0, 1, 2, 3, 0]);
        });
        await test("pad readback preserves byte dtype", async () => {
          const p = new Tensor([1, 0, 1], { dtype: "uint8" }).pad([[1, 1]]);
          assert(p.dtype === "uint8", `expected uint8, got ${p.dtype}`);
          const arr = await p.toArray();
          assert(arr instanceof Uint8Array, `expected Uint8Array-compatible view, got ${arr.constructor.name}`);
          assertClose(arr, [0, 1, 0, 1, 0]);
        });
        await test("gather matches tinygrad probe", async () => {
          const t = new Tensor([[1, 2], [3, 4]]);
          const idx = new Tensor(new Int32Array([0, 0, 1, 0]), { dtype: "int32" }).reshape(2, 2);
          const out = t.gather(1, idx);
          assertShape(out.shape, [2, 2]);
          assertClose(await out.toArray(), [1, 1, 4, 3]);
          const x3 = Tensor.arange(24).reshape(2, 3, 4);
          const idx3 = new Tensor(new Int32Array([0, 2, 1, 0, 2, 1, 0, 2]), { dtype: "int32" }).reshape(2, 2, 2);
          const out3 = x3.gather(1, idx3);
          assertShape(out3.shape, [2, 2, 2]);
          assertClose(await out3.toArray(), [0, 9, 4, 1, 20, 17, 12, 21]);
        });
        await test("oneHot matches tinygrad probe", async () => {
          const out = new Tensor(new Int32Array([0, 2, 1]), { dtype: "int32" }).oneHot(4);
          assertShape(out.shape, [3, 4]);
          assert(out.dtype === "weakint", `expected weakint, got ${out.dtype}`);
          assertClose(await out.toArray(), [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0]);
        });
        await test("tensor row indexing matches tinygrad probe", async () => {
          const idx = new Tensor(new Int32Array([-1, 0, 2]), { dtype: "int32" });
          const out = Tensor.arange(12).reshape(3, 4).getitem(idx);
          assertShape(out.shape, [3, 4]);
          assertClose(await out.toArray(), [8, 9, 10, 11, 0, 1, 2, 3, 8, 9, 10, 11]);
        });
        await test("squeeze and integer indexing preserve scalar rank", async () => {
          const scalar = new Tensor(7);
          assert(scalar.squeeze() === scalar, "scalar squeeze must be a no-op");
          assertShape(Tensor.empty([1]).squeeze(0).shape, []);
          assertShape(Tensor.empty([1, 1]).squeeze().shape, []);
          assertShape(Tensor.empty([2, 1]).squeeze(1).shape, [2]);
          const indexed = Tensor.arange(2, { dtype: "int32" }).getitem(0);
          assertShape(indexed.shape, []);
          assert(indexed.uop.op === pg._core.ops.RESHAPE, "integer index must collapse to RESHAPE");
          assertClose(await indexed.toArray(), [0]);
        });
        await test("basic indices use one shrink before dimension collapse", async () => {
          const base = Tensor.zeros(2, 1, 8, 1, 4).contiguous();
          await base.realize();
          const indexed = base.getitem(0, [0, 1], [0, 3], [0, 1], [0, 4]);
          assertShape(indexed.shape, [1, 3, 1, 4]);
          assert(indexed.uop.op === pg._core.ops.RESHAPE, "basic index root must be RESHAPE");
          assert(indexed.uop.src[0].op === pg._core.ops.SHRINK, "aggregate SHRINK must precede collapse");
          assert(countGraphOp(indexed.uop, pg._core.ops.SHRINK) === 1, "basic indexing must emit one SHRINK");
          assertClose(await indexed.toArray(), new Array(12).fill(0));
          const injected = base.getitem(0, null, [0, 1], [1, 4], [0, 1], [0, 4]);
          assertShape(injected.shape, [1, 1, 3, 1, 4]);
          assert(injected.uop.op === pg._core.ops.SHRINK, "identity final reshape must be elided");
        });
        await test("scatter matches tinygrad probe", async () => {
          const base = Tensor.zeros(3, 5);
          const idx0 = new Tensor(new Int32Array([0, 1, 2, 0]), { dtype: "int32" }).reshape(1, 4);
          const src0 = new Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], { dtype: "float32" }).reshape(2, 5);
          assertClose(await base.scatter(0, idx0, src0).toArray(), [1, 0, 0, 4, 0, 0, 2, 0, 0, 0, 0, 0, 3, 0, 0]);
          const idx1 = new Tensor(new Int32Array([0, 1, 2, 0, 1, 4, 2, 3, 4]), { dtype: "int32" }).reshape(3, 3);
          const src1 = new Tensor([1, 2, 3, 6, 7, 8, 9, 10, 11], { dtype: "float32" }).reshape(3, 3);
          assertClose(await base.scatter(1, idx1, src1).toArray(), [1, 2, 3, 0, 0, 6, 7, 0, 0, 8, 0, 0, 9, 10, 11]);
          const dupIdx = new Tensor(new Int32Array([1, 1, 2]), { dtype: "int32" }).reshape(1, 3);
          const dupSrc = new Tensor([7, 9, 8]).reshape(1, 3);
          assertClose(await new Tensor([[0, 0, 0, 0]]).scatter(1, dupIdx, dupSrc).toArray(), [0, 9, 8, 0]);
          const scalarIdx = new Tensor(new Int32Array([2, 3]), { dtype: "int32" }).reshape(2, 1);
          const floatBase = Tensor.full([2, 4], 2, { dtype: "float32" });
          assertClose(await floatBase.scatter(1, scalarIdx, 1.23, "add").toArray(), [2, 2, 3.23, 2, 2, 2, 2, 3.23]);
          assertClose(await floatBase.scatter(1, scalarIdx, 1.23, "multiply").toArray(), [2, 2, 2.46, 2, 2, 2, 2, 2.46]);
          let threw = false;
          try {
            base.scatter(1, idx1, src1, "sum");
          } catch (e) {
            threw = true;
          }
          assert(threw, "expected invalid scatter reduce string to throw");
          threw = false;
          try {
            base.scatter(1, idx1, src1, "add");
          } catch (e) {
            threw = true;
          }
          assert(threw, "expected tensor src with scatter reduce arg to throw");
        });
        await test("scatterReduce matches tinygrad probe", async () => {
          const base = new Tensor([[1, 2, 3, 4, 5]], { dtype: "float32" });
          const idx = new Tensor(new Int32Array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]), { dtype: "int32" }).reshape(1, 10);
          const src = new Tensor([[1, 6, 2, 7, 3, 8, 4, 9, 5, 10]], { dtype: "float32" });
          assertClose(await base.scatterReduce(1, idx, src, "sum").toArray(), [8, 11, 14, 17, 20]);
          assertClose(await base.scatter_reduce(1, idx, src, "prod").toArray(), [6, 28, 72, 144, 250]);
          assertClose(await base.scatterReduce(1, idx, src, "mean", false).toArray(), [3.5, 4.5, 5.5, 6.5, 7.5]);
          const extremeBase = new Tensor([[-10, 20, 0, 5, 10]], { dtype: "float32" });
          assertClose(await extremeBase.scatterReduce(1, idx, src, "amax").toArray(), [6, 20, 8, 9, 10]);
          assertClose(await extremeBase.scatterReduce(1, idx, src, "amin").toArray(), [-10, 2, 0, 4, 5]);
          let threw = false;
          try {
            base.scatterReduce(1, idx, src, "max");
          } catch (e) {
            threw = true;
          }
          assert(threw, "expected invalid scatterReduce reduction to throw");
        });
        await test("scatterReduce preserves missing lanes and duplicate indices on both axes", async () => {
          for (const n of [3, 5, 9]) {
            const m = 2 * (n - 1);
            for (const axis of [0, 1]) {
              const at = (c, i, width) => axis === 0 ? i * 2 + c : c * width + i;
              const baseData = new Array(2 * n);
              const indexData = new Int32Array(2 * m);
              const sourceData = new Array(2 * m);
              for (let c = 0; c < 2; c++) {
                for (let i = 0; i < n; i++) baseData[at(c, i, n)] = c === 0 ? i + 1 : -i - 1;
                for (let j = 0; j < m; j++) {
                  indexData[at(c, j, m)] = Math.floor(j / 2);
                  sourceData[at(c, j, m)] = (j % 2 === 0 ? -2 : 3) + c;
                }
              }
              const base = new Tensor(baseData, { dtype: "float32" }).reshape(axis === 0 ? [n, 2] : [2, n]);
              const idx = new Tensor(indexData, { dtype: "int32" }).reshape(axis === 0 ? [m, 2] : [2, m]);
              const src = new Tensor(sourceData, { dtype: "float32" }).reshape(axis === 0 ? [m, 2] : [2, m]);
              for (const includeSelf of [false, true]) {
                for (const reduction of ["sum", "prod", "mean", "amax", "amin"]) {
                  const expected = baseData.slice();
                  for (let c = 0; c < 2; c++) {
                    for (let i = 0; i < n - 1; i++) {
                      const values = [-2 + c, 3 + c];
                      if (includeSelf) values.unshift(baseData[at(c, i, n)]);
                      expected[at(c, i, n)] = reduction === "prod" ? values.reduce((a, b) => a * b, 1) : reduction === "amax" ? Math.max(...values) : reduction === "amin" ? Math.min(...values) : values.reduce((a, b) => a + b, 0) / (reduction === "mean" ? values.length : 1);
                    }
                  }
                  const result = base.scatterReduce(axis, idx, src, reduction, includeSelf);
                  try {
                    assertClose(await result.toArray(), expected);
                  } catch (err) {
                    throw new Error(`n=${n} axis=${axis} ${reduction} includeSelf=${includeSelf}: ${err.message}`);
                  } finally {
                    result.dispose();
                  }
                }
              }
              base.dispose();
              idx.dispose();
              src.dispose();
            }
          }
        });
        await test("packed storage reductions preserve small integer and bool values", async () => {
          for (const n of [257, 1024]) {
            for (const dtype of ["bool", "int8", "uint8", "int16", "uint16", "float32"]) {
              const values = Array.from({ length: n }, (_, i) => dtype === "bool" ? i % 2 : dtype === "int8" ? i % 121 - 60 : dtype === "uint8" ? i % 251 : dtype === "int16" ? i * 173 % 60001 - 3e4 : dtype === "uint16" ? i * 173 % 65536 : i % 121 - 60);
              const input = new Tensor(values, { dtype });
              const hi = input.max();
              assertClose(await hi.toArray(), [Math.max(...values)]);
              hi.dispose();
              input.dispose();
            }
          }
        });
        await test("scatter construction bypasses frontend substitution", async () => {
          assert(
            Tensor.prototype._physicalizeResult === void 0,
            "legacy frontend substitution helper must stay deleted"
          );
          const base = new Tensor([[1, 2, 3, 4, 5]]);
          const idx = new Tensor(new Int32Array([0, 1, 1, 3, 4]), { dtype: "int32" }).reshape(1, 5);
          const src = new Tensor([[6, 7, 8, 9, 10]]);
          assertClose(await base.scatter(1, idx, src).toArray(), [6, 8, 3, 9, 10]);
          assertClose(await base.scatterReduce(1, idx, src, "sum").toArray(), [7, 17, 3, 13, 15]);
        });
        console.log("\n-- Step slicing --");
        await test("step2 1d", async () => {
          const t = new Tensor([1, 2, 3, 4, 5, 6, 7, 8]);
          const r = t.getitem({ start: 0, stop: 8, step: 2 });
          assertShape(r.shape, [4]);
          assertClose(await r.toArray(), [1, 3, 5, 7]);
        });
        await test("step3 1d", async () => {
          const t = new Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]);
          const r = t.getitem({ start: 0, stop: 9, step: 3 });
          assertShape(r.shape, [3]);
          assertClose(await r.toArray(), [1, 4, 7]);
        });
        await test("step2 with start stop", async () => {
          const t = new Tensor([0, 1, 2, 3, 4, 5, 6, 7]);
          const r = t.getitem({ start: 1, stop: 7, step: 2 });
          assertShape(r.shape, [3]);
          assertClose(await r.toArray(), [1, 3, 5]);
        });
        await test("step non-divisible", async () => {
          const t = new Tensor([0, 1, 2, 3, 4, 5, 6]);
          const r = t.getitem({ start: 0, stop: 7, step: 3 });
          assertShape(r.shape, [3]);
          assertClose(await r.toArray(), [0, 3, 6]);
        });
        await test("negative step (reverse)", async () => {
          const t = new Tensor([1, 2, 3, 4, 5, 6]);
          const r = t.getitem({ step: -1 });
          assertClose(await r.toArray(), [6, 5, 4, 3, 2, 1]);
        });
        await test("negative step2", async () => {
          const t = new Tensor([1, 2, 3, 4, 5, 6]);
          const r = t.getitem({ step: -2 });
          assertShape(r.shape, [3]);
          assertClose(await r.toArray(), [6, 4, 2]);
        });
        await test("step 2d axis0", async () => {
          const t = new Tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]);
          const r = t.getitem({ start: 0, stop: 4, step: 2 });
          assertShape(r.shape, [2, 3]);
          assertClose(await r.toArray(), [0, 1, 2, 6, 7, 8]);
        });
        await test("step 2d both axes", async () => {
          const t = new Tensor([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]);
          const r = t.getitem({ start: 0, stop: 4, step: 2 }, { start: 0, stop: 4, step: 2 });
          assertShape(r.shape, [2, 2]);
          assertClose(await r.toArray(), [0, 2, 8, 10]);
        });
        console.log("\n-- Cast --");
        await testIf(supportsF64, "cast float32 to float64", async () => {
          const t = new Tensor([1, 2, 3]);
          const r = t.cast("float64");
          assert(r.dtype === "float64", `expected float64, got ${r.dtype}`);
          const arr = await r.toArray();
          console.log("DEBUG cast f32->f64", Array.from(arr));
          assertClose(arr, [1, 2, 3]);
        });
        await testIf(supportsF64, "cast float64 to float32", async () => {
          const t = new Tensor([1.5, 2.5, 3.5], { dtype: "float64" });
          const r = t.cast("float32");
          assert(r.dtype === "float32", `expected float32, got ${r.dtype}`);
          const arr = await r.toArray();
          console.log("DEBUG cast f64->f32", Array.from(arr));
          assertClose(arr, [1.5, 2.5, 3.5]);
        });
        await test("cast no-op same dtype", async () => {
          const t = new Tensor([1, 2, 3], { dtype: "float32" });
          const r = t.cast("float32");
          assert(r === t, "same dtype should return self");
          assertClose(await r.toArray(), [1, 2, 3]);
        });
        await test("owned identity C return survives JS adoption", async () => {
          const t = new Tensor([1, 2], { dtype: "float32", device: "cpu" });
          t._adoptCoreTensor(t._coreToDevice("cpu"));
          assertClose(await t.toArray(), [1, 2]);
        });
        await test("cast and bitcast store exact physical roots", async () => {
          const source = Tensor.arange(4, { dtype: "uint32" });
          const casted = source.cast("uint64");
          const bitcasted = source.bitcast("float32");
          assert(casted.uopLogical.op === pg._core.ops.CAST, "cast logical root must be CAST");
          assert(casted.uopPhysical.op === pg._core.ops.CAST, "cast physical root must be CAST");
          assert(casted.uopLogical.src[0].key === source.uopLogical.key, "cast logical source mismatch");
          assert(casted.uopPhysical.src[0].key === source.uopPhysical.key, "cast physical source mismatch");
          assert(bitcasted.uopLogical.op === pg._core.ops.BITCAST, "bitcast logical root must be BITCAST");
          assert(bitcasted.uopPhysical.op === pg._core.ops.BITCAST, "bitcast physical root must be BITCAST");
          assert(bitcasted.uopLogical.src[0].key === source.uopLogical.key, "bitcast logical source mismatch");
          assert(bitcasted.uopPhysical.src[0].key === source.uopPhysical.key, "bitcast physical source mismatch");
        });
        await test("unequal-width bitcast matches pinned lane order", async () => {
          const wide = Tensor.full([8], 1, { dtype: "uint8" }).bitcast("uint32");
          const narrow = Tensor.full([2], 1, { dtype: "uint32" }).bitcast("uint8");
          assert(wide.shape.length === 1 && wide.shape[0] === 2, "wide bitcast shape mismatch");
          assert(narrow.shape.length === 1 && narrow.shape[0] === 8, "narrow bitcast shape mismatch");
          assertClose(await wide.toArray(), [16843009, 16843009], 0);
          assertClose(await narrow.toArray(), [1, 0, 0, 0, 1, 0, 0, 0], 0);
          let invalid = null;
          try {
            Tensor.empty([3], { dtype: "uint8" }).bitcast("uint32");
          } catch (err) {
            invalid = err;
          }
          assert(
            invalid && /unsupported size in bitcast/.test(invalid.message),
            "statically non-divisible bitcast must fail in the C Tensor boundary"
          );
          let weak = null;
          try {
            Tensor.full([1], 1.5, { buffer: false }).bitcast("uint32");
          } catch (err) {
            weak = err;
          }
          assert(
            weak && /bitcast requires concrete dtypes/.test(weak.message),
            "weak bitcast must fail at the Tensor API boundary"
          );
        });
        for (const mode of ["plain", "reshape", "shrink", "aligned-shrink", "detach", "nested", "repeated"]) {
          const unsupportedOffset = mode === "shrink" && pg.device === "webgpu";
          const name = unsupportedOffset ? "bitcast view assign rejects unsupported WebGPU offset" : `bitcast view assign matches current tinygrad: ${mode}`;
          await test(name, async () => {
            const initial = mode === "aligned-shrink" ? Array(64).fill(1).concat([2, 3, 4]) : [1, 2, 3, 4];
            const a = new Tensor(initial, { dtype: "float32" });
            await a.realize();
            let view = a.bitcast("uint32");
            let values = new Uint32Array([1082130432, 1077936128, 1073741824, 1065353216]);
            let expected = [4, 3, 2, 1];
            if (mode === "reshape") view = view.reshape(2, 2);
            if (mode === "shrink") {
              view = a.shrink([[1, 3]]).bitcast("uint32");
              values = values.slice(1, 3);
              expected = [1, 3, 2, 4];
            }
            if (mode === "aligned-shrink") {
              view = a.shrink([[64, 66]]).bitcast("uint32");
              values = values.slice(1, 3);
              expected = Array(64).fill(1).concat([3, 2, 4]);
            }
            if (mode === "detach") view = view.detach();
            if (mode === "nested") {
              view = view.bitcast("int32");
              values = new Int32Array(values.buffer);
            }
            const originalStorage = a.uop;
            view.assign(new Tensor(values).reshape(view.shape));
            if (unsupportedOffset) {
              let error;
              try {
                await view.realize();
              } catch (e) {
                error = e;
              }
              assert(error && /poly_realize_tensors failed/.test(error.message), "unaligned view must reject");
              assertClose(await new Tensor(originalStorage).toArray(), initial, 0);
              return;
            }
            await view.realize();
            assertClose(await a.toArray(), expected, 0);
            if (mode === "repeated") {
              view = a.bitcast("uint32");
              view.assign(new Tensor(new Uint32Array(4).fill(1073741824)));
              await view.realize();
              assertClose(await a.toArray(), [2, 2, 2, 2], 0);
            }
          });
        }
        await testIf(supportsF16, "half and double convenience", async () => {
          const t = new Tensor([1, 2, 3]);
          const h = t.half();
          assert(h.dtype === "float16", `expected float16, got ${h.dtype}`);
          assertClose(await h.toArray(), [1, 2, 3]);
          if (supportsF64) {
            const d = t.double();
            assert(d.dtype === "float64", `expected float64, got ${d.dtype}`);
            assertClose(await d.toArray(), [1, 2, 3]);
          }
        });
        await testIf(supportsF64, "cast then compute", async () => {
          const t = new Tensor([1, 2, 3]).cast("float64");
          const r = t.add(new Tensor([10, 20, 30], { dtype: "float64" }));
          assertClose(await r.toArray(), [11, 22, 33]);
        });
        console.log("\n-- Triu/Tril --");
        await test("triu 2d", async () => {
          const t = new Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
          const r = t.triu();
          const arr = await r.toArray();
          console.log("DEBUG triu", Array.from(arr));
          assertClose(arr, [1, 2, 3, 0, 5, 6, 0, 0, 9]);
        });
        await test("tril 2d", async () => {
          const t = new Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
          const r = t.tril();
          const arr = await r.toArray();
          console.log("DEBUG tril", Array.from(arr));
          assertClose(arr, [1, 0, 0, 4, 5, 0, 7, 8, 9]);
        });
        await test("triu with diagonal", async () => {
          const t = new Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
          const r = t.triu(1);
          const arr = await r.toArray();
          console.log("DEBUG triu diag", Array.from(arr));
          assertClose(arr, [0, 2, 3, 0, 0, 6, 0, 0, 0]);
        });
        await test("triu/tril batched last two dims", async () => {
          const data = [
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[10, 11, 12], [13, 14, 15], [16, 17, 18]]
          ];
          const t = new Tensor(data);
          const upper = t.triu();
          const lower = t.tril(1);
          assertShape(upper.shape, [2, 3, 3]);
          assertShape(lower.shape, [2, 3, 3]);
          assertClose(await upper.toArray(), [1, 2, 3, 0, 5, 6, 0, 0, 9, 10, 11, 12, 0, 14, 15, 0, 0, 18]);
          assertClose(await lower.toArray(), [1, 2, 0, 4, 5, 6, 7, 8, 9, 10, 11, 0, 13, 14, 15, 16, 17, 18]);
          const z = Tensor.zeros(5, 0, 3);
          assertShape(z.triu().shape, [5, 0, 3]);
          assertShape(z.tril().shape, [5, 0, 3]);
          assertClose(await z.triu().toArray(), []);
        });
        await test("triu/tril use pinned composition without substitution", async () => {
          assert(
            Tensor.prototype._physicalizeResult === void 0,
            "legacy frontend substitution helper must stay deleted"
          );
          const x = new Tensor([
            [1, 2, 3, 4],
            [5, 6, 7, 8]
          ]);
          const upper = x.triu(-1);
          const lower = x.tril(1);
          const expectedUpper = Tensor._tri(2, 4, -1, x.device).where(x, x.constLike(0));
          const expectedLower = Tensor._tri(2, 4, 2, x.device).where(x.constLike(0), x);
          assert(upper.uop.key === expectedUpper.uop.key, "triu physical graph differs");
          assert(lower.uop.key === expectedLower.uop.key, "tril physical graph differs");
          assert(upper.uopLogical.key === expectedUpper.uopLogical.key, "triu logical graph differs");
          assert(lower.uopLogical.key === expectedLower.uopLogical.key, "tril logical graph differs");
          assertClose(await upper.toArray(), [1, 2, 3, 4, 5, 6, 7, 8]);
          assertClose(await lower.toArray(), [1, 2, 0, 0, 5, 6, 7, 0]);
          const moved = (await new Tensor([
            [1, 2, 3, 4],
            [5, 6, 7, 8]
          ], { device: "cpu" }).realize()).to("cuda").to("cpu");
          assert(
            countGraphOp(moved.triu().uop, pg._core.ops.COPY) === 2,
            "triu lost moved occurrence"
          );
          assert(
            countGraphOp(moved.tril().uop, pg._core.ops.COPY) === 2,
            "tril lost moved occurrence"
          );
        });
        console.log("\n-- Reduction --");
        await test("linear vector and matrix match pinned semantics", async () => {
          const x = new Tensor(Float32Array.from({ length: 16 }, (_, i) => i / 7)).reshape(2, 2, 4);
          const vector = x.linear(
            new Tensor(new Float32Array([1, 2, 3, 4])),
            new Tensor(new Float32Array([0.5, 1, 1.5, 2]))
          );
          const matrixWeight = new Tensor(Float32Array.from({ length: 12 }, (_, i) => i / 11)).reshape(4, 3);
          const matrix = x.linear(matrixWeight, new Tensor(new Float32Array([0.25, -0.5, 0.75])));
          assertShape(vector.shape, [2, 2, 4]);
          assertShape(matrix.shape, [2, 2, 3]);
          assertClose(await vector.toArray(), [
            0.5,
            1 + 2 / 7,
            1.5 + 6 / 7,
            2 + 12 / 7,
            4 / 7 + 0.5,
            1 + 10 / 7,
            1.5 + 18 / 7,
            2 + 28 / 7,
            8 / 7 + 0.5,
            1 + 18 / 7,
            1.5 + 30 / 7,
            2 + 44 / 7,
            12 / 7 + 0.5,
            1 + 26 / 7,
            1.5 + 42 / 7,
            2 + 60 / 7
          ]);
          assertClose(await matrix.toArray(), [
            42 / 77 + 0.25,
            48 / 77 - 0.5,
            54 / 77 + 0.75,
            114 / 77 + 0.25,
            136 / 77 - 0.5,
            158 / 77 + 0.75,
            186 / 77 + 0.25,
            224 / 77 - 0.5,
            262 / 77 + 0.75,
            258 / 77 + 0.25,
            312 / 77 - 0.5,
            366 / 77 + 0.75
          ]);
        });
        await test("attention shared C dropout and rank-five GQA", async () => {
          const q = Tensor.arange(48).cast("float32").reshape(1, 2, 4, 2, 3).div(37);
          const k = Tensor.arange(36).cast("float32").reshape(1, 2, 2, 3, 3).div(29);
          const v = Tensor.arange(12).cast("float32").reshape(1, 2, 1, 3, 2).div(17);
          const oldTraining = Tensor.training;
          try {
            let invalid = false;
            try {
              q.scaledDotProductAttention(k, v, { dropout_p: NaN });
            } catch (err) {
              invalid = /out of range/.test(String(err));
            }
            assert(invalid, "snake-case dropout probability must reject NaN");
            const empty = Tensor.zeros(2, 0).scaledDotProductAttention(Tensor.zeros(3, 0), Tensor.ones(3, 4));
            assertShape(empty.shape, [2, 4]);
            assertClose(await empty.toArray(), Array(8).fill(NaN));
            empty.dispose();
            const active = q._rt._activeAsync;
            q._rt._activeAsync++;
            Tensor.training = true;
            try {
              for (const call of [() => q.dropout(0.25), () => q.scaledDotProductAttention(k, v)]) {
                let rejected = false;
                try {
                  call();
                } catch (err) {
                  rejected = /active async work/.test(String(err));
                }
                assert(rejected, "attention/dropout must not enter a suspended core");
              }
            } finally {
              q._rt._activeAsync = active;
            }
            for (const [training, p] of [[false, 0.25], [true, 0.25], [true, 1]]) {
              Tensor.training = training;
              Tensor.manual_seed(11);
              const actual = q.scaledDotProductAttention(k, v, { enableGqa: true, dropoutP: p });
              const nextActual = Tensor.rand(4);
              Tensor.manual_seed(11);
              const scores = q.matmul(k.repeatInterleave(2, -3).transpose(-2, -1), false, "float32").div(Math.sqrt(3));
              let weights = scores.cast(q.dtype).softmax(-1);
              if (training) {
                weights = p === 1 ? weights.constLike(0) : Tensor.randLike(weights, { dtype: "float32", contiguous: false }).ge(p).contiguous().where(weights, 0).div(1 - p);
              }
              const expected = weights.matmul(v.repeatInterleave(4, -3));
              const nextExpected = Tensor.rand(4);
              assertShape(actual.shape, [1, 2, 4, 2, 2]);
              assertClose(await actual.toArray(), await expected.toArray());
              assertClose(await nextActual.toArray(), await nextExpected.toArray(), 0);
              for (const t of [actual, expected, nextActual, nextExpected]) t.dispose();
            }
          } finally {
            Tensor.training = oldTraining;
            for (const t of [q, k, v]) t.dispose();
          }
        });
        await test("attention primitives match pinned compositions", async () => {
          const x = new Tensor([[1, 2], [3, 4]], { dtype: "float32" });
          const repeated = x.repeatInterleave(2, 1);
          const expectedRepeat = x.reshape(2, 2, 1).expand(2, 2, 2).reshape(2, 4);
          assert(repeated.uop.key === expectedRepeat.uop.key, "repeatInterleave graph differs");
          assertClose(await repeated.toArray(), [1, 1, 2, 2, 3, 3, 4, 4]);
          assert(x.dropout(0.25) === x, "eval dropout must return self");
          let rangeError = false;
          try {
            x.dropout(1.1);
          } catch (err) {
            rangeError = /out of range/.test(String(err));
          }
          assert(rangeError, "dropout must reject p outside [0,1]");
          Tensor.manual_seed(11);
          Tensor.training = true;
          const dropped = x.dropout(0.25);
          Tensor.manual_seed(11);
          const expectedDropout = Tensor.randLike(x, { dtype: "float32", contiguous: false }).ge(0.25).contiguous().where(x, 0).div(0.75);
          Tensor.training = false;
          assert(dropped.uop.key !== expectedDropout.uop.key, "reset RNG occurrences must remain distinct");
          assertClose(await dropped.toArray(), await expectedDropout.toArray());
          const q = new Tensor(Float32Array.from({ length: 12 }, (_, i) => i / 13)).reshape(1, 2, 2, 3);
          const k = new Tensor(Float32Array.from({ length: 12 }, (_, i) => (i - 4) / 11)).reshape(1, 2, 2, 3);
          const v = new Tensor(Float32Array.from({ length: 16 }, (_, i) => (i + 1) / 17)).reshape(1, 2, 2, 4);
          const causal = q.scaledDotProductAttention(k, v, { isCausal: true });
          const qk = q.matmul(k.transpose(-2, -1), false, "float32").div(Math.sqrt(3));
          const mask = qk.cast("bool").constLike(true).tril().where(0, -Infinity);
          const expectedCausal = qk.add(mask).cast(q.dtype).softmax(-1).matmul(v);
          assert(causal.uop.key === expectedCausal.uop.key, "causal attention graph differs");
          assertClose(await causal.toArray(), await expectedCausal.toArray());
          const qg = new Tensor(Float32Array.from({ length: 24 }, (_, i) => i / 19)).reshape(1, 4, 2, 3);
          const gqa = qg.scaledDotProductAttention(k, v, { enableGqa: true });
          assertShape(gqa.shape, [1, 4, 2, 4]);
          assertClose(await gqa.toArray(), [
            0.17793298,
            0.23675652,
            0.29558003,
            0.35440356,
            0.18231565,
            0.24113917,
            0.2999627,
            0.35878623,
            0.18668212,
            0.24550565,
            0.3043292,
            0.3631527,
            0.1910204,
            0.24984394,
            0.30866745,
            0.36749098,
            0.66590714,
            0.72473067,
            0.7835542,
            0.8423777,
            0.6701545,
            0.7289781,
            0.78780156,
            0.84662515,
            0.67434025,
            0.7331638,
            0.7919873,
            0.8508109,
            0.67845434,
            0.73727787,
            0.79610145,
            0.8549249
          ], 1e-5);
        });
        await test("permute resolves negative axes validates and preserves identity", async () => {
          const x = new Tensor(Float32Array.from({ length: 24 }, (_, i) => i)).reshape(2, 3, 4);
          const direct = x.permute(0, 2, 1);
          const negative = x.permute(0, -1, 1);
          assert(direct.uop.key === negative.uop.key, "negative permute graph differs");
          assertShape(direct.shape, [2, 4, 3]);
          assert(x.permute(0, 1, 2) === x, "identity permute must return self");
          let invalid = false;
          try {
            x.permute(0, 0, 1);
          } catch (err) {
            invalid = /not a valid permutation/.test(String(err));
          }
          assert(invalid, "duplicate permute axes must fail");
        });
        await test("sum all", async () => {
          const t = new Tensor([1, 2, 3]);
          const v = await t.sum().item();
          assert(Math.abs(v - 6) < 1e-4, `Expected 6, got ${v}`);
        });
        await testIf(supportsF64, "sum supports an explicit accumulation dtype", async () => {
          const x = new Tensor(new Float32Array([1.25, -2, 0.5, 3, 0.25, -1.5])).reshape(2, 3);
          const out = x.sum(1, false, "float64");
          assert(out.dtype === "float64", `expected float64, got ${out.dtype}`);
          assertClose(await out.toArray(), [-0.25, 1.75]);
        });
        await test("sum axis", async () => {
          const t = new Tensor([[1, 2], [3, 4]]);
          const s = t.sum(1);
          assertShape(s.shape, [2]);
          assertClose(await s.toArray(), [3, 7]);
        });
        await test("mean", async () => {
          const t = new Tensor([2, 4, 6]);
          const v = await t.mean().item();
          assert(Math.abs(v - 4) < 1e-4, `Expected 4, got ${v}`);
        });
        await test("var matches pinned expression without substitution", async () => {
          assert(
            Tensor.prototype._physicalizeResult === void 0,
            "legacy frontend substitution helper must stay deleted"
          );
          const x = new Tensor([[1, 2, 4], [3, 5, 9]]);
          const pinnedExpression = (axis, keepdim = false, correction = 1) => {
            const squares = x.sub(x.mean(axis, true)).square();
            const reducedShape = squares.sum(axis, true).shape;
            const n = x.shape.filter((si, i) => si !== reducedShape[i]).reduce((a, b) => a * b, 1);
            const reduced = squares.sum(axis, keepdim);
            return reduced.div(reduced.constLike(n).sub(correction).relu());
          };
          for (const [axis, keepdim, correction] of [
            [1, false, 1],
            [0, true, 0],
            [null, false, 1],
            [[0, 1], false, 1],
            [1, false, 3]
          ]) {
            const actual = x.var(axis, keepdim, correction);
            const expected = pinnedExpression(axis, keepdim, correction);
            assert(actual.uop.key === expected.uop.key, "variance physical graph differs");
            assert(actual.uopLogical.key === expected.uopLogical.key, "variance logical graph differs");
            const actualValues = await actual.toArray();
            const expectedValues = await expected.toArray();
            if (correction === 3) {
              assert(
                actualValues.length === expectedValues.length && actualValues.every((value, i) => value === expectedValues[i]),
                "variance non-finite values differ"
              );
            } else {
              assertClose(actualValues, expectedValues);
            }
          }
        });
        await test("max", async () => {
          const t = new Tensor([[1, 5], [3, 2]]);
          const m = t.max(1);
          assertClose(await m.toArray(), [5, 3]);
        });
        await test("argmax matches tinygrad probe", async () => {
          const t = new Tensor([[1.2, 0.5, 1.2], [2.2, 1.9, 0]]);
          assertClose(await t.argmax(1).toArray(), [0, 0]);
          assertShape(t.argmax(1, true).shape, [2, 1]);
          assertClose(await t.argmax(1, true).toArray(), [0, 0]);
          const flat = await t.argmax().item();
          assert(flat === 3, `expected flattened argmax 3, got ${flat}`);
          const singleton = Tensor.arange(6, { dtype: "float32" }).reshape(2, 1, 3);
          assertClose(await singleton.argmax(1).toArray(), [0, 0, 0, 0, 0, 0]);
          const empty = Tensor.empty(2, 0, 3, { device: "cpu" });
          assertClose(
            await empty.argmax(1).toArray(),
            [-2147483648, -2147483648, -2147483648, -2147483648, -2147483648, -2147483648]
          );
        });
        await test("sort argsort topk match tinygrad probe", async () => {
          const x = new Tensor([[0.1, 0.5, 1.2, 3.4, 2.1], [2.2, 1.9, 0.3, 4.5, 0.8]]);
          let pair = x.sort(1, false);
          assertShape(pair[0].shape, [2, 5]);
          assertShape(pair[1].shape, [2, 5]);
          assertClose(await pair[0].toArray(), [0.1, 0.5, 1.2, 2.1, 3.4, 0.3, 0.8, 1.9, 2.2, 4.5]);
          assertClose(await pair[1].toArray(), [0, 1, 2, 4, 3, 2, 4, 1, 0, 3]);
          pair = x.sort(1, true);
          assertClose(await pair[0].toArray(), [3.4, 2.1, 1.2, 0.5, 0.1, 4.5, 2.2, 1.9, 0.8, 0.3]);
          assertClose(await pair[1].toArray(), [3, 4, 2, 1, 0, 3, 0, 1, 4, 2]);
          pair = x.topk(2, 1);
          assertShape(pair[0].shape, [2, 2]);
          assertShape(pair[1].shape, [2, 2]);
          assertClose(await pair[0].toArray(), [3.4, 2.1, 4.5, 2.2]);
          assertClose(await pair[1].toArray(), [3, 4, 3, 0]);
          pair = x.topk(2, 1, false);
          assertClose(await pair[0].toArray(), [0.1, 0.5, 0.3, 0.8]);
          assertClose(await pair[1].toArray(), [0, 1, 2, 4]);
          const t = new Tensor([[2, 3, 4, 1], [1, 4, 3, 2]]);
          assertClose(await t.argsort().toArray(), [3, 0, 1, 2, 0, 3, 2, 1]);
        });
        await test("topk tie order and errors match tinygrad probe", async () => {
          const tie = new Tensor([[1, 1, 0, 1]]);
          const pair = tie.topk(3, 1);
          assertClose(await pair[0].toArray(), [1, 1, 1]);
          assertClose(await pair[1].toArray(), [0, 1, 3]);
          let threw = false;
          try {
            new Tensor([[0.1, 0.2]]).topk(6, 1);
          } catch (_) {
            threw = true;
          }
          assert(threw, "expected topk k out of range to throw");
          threw = false;
          try {
            new Tensor([[0.1, 0.2]]).topk(1, 1, true, false);
          } catch (_) {
            threw = true;
          }
          assert(threw, "expected topk sorted_=false to throw");
        });
        await test("softmax", async () => {
          const t = new Tensor([1, 2, 3]);
          const arr = await (await t.softmax()).toArray();
          const sum = arr[0] + arr[1] + arr[2];
          assert(Math.abs(sum - 1) < 1e-4, `Softmax sum should be 1, got ${sum}`);
        });
        await test("fusion fuzzer smoke patterns", async () => {
          const movement = new Tensor([
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11]
          ]).pad([[1, 0], [0, 1]]).shrink([[1, 4], [1, 5]]).mul(0.25).add(1).sum(1);
          assertShape(movement.shape, [3]);
          assertClose(await movement.toArray(), [5.5, 8.5, 11.5], 2e-4);
          const x = new Tensor([
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15]
          ]);
          const whereReduce = Tensor.full([4, 4], 7).gt(x).where(x, Tensor.full([4, 4], -2)).sum(0);
          assertShape(whereReduce.shape, [4]);
          assertClose(await whereReduce.toArray(), [0, 2, 4, -3], 2e-4);
          const base = new Tensor([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
          ]);
          const noBarrier = base.add(1);
          assertClose(await noBarrier.square().add(noBarrier).sum(0).toArray(), [108, 144, 186], 2e-4);
          const barrier = base.add(1);
          await barrier.realize();
          assertClose(await barrier.square().add(barrier).sum(0).toArray(), [108, 144, 186], 2e-4);
          const [q, r] = new Tensor([[1, 2], [-1, 3], [0.5, 4]]).qr();
          assertClose(await q.dot(r).toArray(), [1, 2, -1, 3, 0.5, 4], 2e-3);
          const [qb, rb] = new Tensor([[1, 2], [-1, 3], [0.5, 4]]).qr();
          await qb.realize(rb);
          assertClose(await qb.dot(rb).toArray(), [1, 2, -1, 3, 0.5, 4], 2e-3);
          const chol = new Tensor([[4, 2], [2, 5]]).cholesky();
          assertClose(
            await chol.choleskySolve(new Tensor([[1, 2], [3, 4]])).toArray(),
            [-0.0625, 0.125, 0.625, 0.75],
            3e-4
          );
          const cholBarrier = new Tensor([[4, 2], [2, 5]]).cholesky();
          await cholBarrier.realize();
          assertClose(
            await cholBarrier.choleskySolve(new Tensor([[1, 2], [3, 4]])).toArray(),
            [-0.0625, 0.125, 0.625, 0.75],
            3e-4
          );
          assertClose(
            await new Tensor([[0, 2], [1, 3]]).solve(new Tensor([4, 5])).toArray(),
            [-1, 2],
            4e-4
          );
          assertClose(
            await new Tensor([[1, 0], [1, 1], [1, 2]]).lstsq(new Tensor([1, 2, 3])).toArray(),
            [1, 1],
            6e-4
          );
        });
        await test("matmul", async () => {
          const a = new Tensor([[1, 2], [3, 4]]);
          const b = new Tensor([[5, 6], [7, 8]]);
          assertClose(await a.dot(b).toArray(), [19, 22, 43, 50]);
        });
        await testIf(supportsF64, "dot supports an explicit accumulation dtype", async () => {
          const a = new Tensor(new Float32Array([1.25, -2, 0.5, 3, 0.25, -1.5])).reshape(2, 3);
          const b = new Tensor(new Float32Array([0.5, -1, 2, 0.25, -0.75, 3])).reshape(3, 2);
          const out = a.dot(b, "float64");
          assert(out.dtype === "float64", `expected float64, got ${out.dtype}`);
          assertClose(await out.toArray(), [-3.75, -0.25, 3.125, -7.4375]);
        });
        await test("vector dot is scalar", async () => {
          const scalar = new Tensor(new Float32Array([1, 2, 3])).dot(new Tensor(new Float32Array([4, 5, 6])));
          assert(scalar.shape.length === 0, `expected scalar shape, got ${JSON.stringify(scalar.shape)}`);
        });
        await testIf(supportsF16, "mixed float16 float32 dot promotes to float32", async () => {
          const mixedA = new Tensor([1.25, -2, 0.5, 3, 0.25, -1.5]).reshape(2, 3).cast("float16");
          const mixedB = new Tensor([0.5, -1, 2, 0.25, -0.75, 3]).reshape(3, 2);
          const mixed = mixedA.dot(mixedB);
          assert(mixed.dtype === "float32", `expected mixed dot float32, got ${mixed.dtype}`);
          assert(mixed.uop.op === mixed.uop.ffi.__polygradOps.REDUCE, "expected mixed dot REDUCE");
          assertClose(await mixed.toArray(), [-3.75, -0.25, 3.125, -7.4375], 1e-5);
        });
        await test("matmul shape mismatch throws", async () => {
          const a = new Tensor([[1, 2], [3, 4]]);
          const b = new Tensor([[1, 2, 3]]);
          let ok = false;
          try {
            const c = a.dot(b);
            ok = c.shape.length === 0;
          } catch (e) {
            ok = true;
          }
          assert(ok, "expected dot to fail on shape mismatch");
        });
        await test("matmul broadcast batch", async () => {
          const a = new Tensor([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]]
          ]);
          const b = new Tensor([
            [[1, 10], [100, 1e3]]
          ]);
          const out = a.dot(b);
          assertShape(out.shape, [2, 2, 2]);
          assertClose(await out.toArray(), [201, 2010, 403, 4030, 605, 6050, 807, 8070]);
        });
        await test("matmul broadcast mismatch throws", async () => {
          const a = new Tensor([
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
          ]);
          const b = new Tensor(new Array(5).fill(0).map(() => [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
          ]));
          let ok = false;
          try {
            const c = a.dot(b);
            ok = c.shape.length === 0;
          } catch (e) {
            ok = true;
          }
          assert(ok, "expected dot to fail on broadcast-mismatched shapes");
        });
        await test("einsum uses the shared native/WASM adapter contract", async () => {
          assert(
            Tensor.prototype._physicalizeResult === void 0,
            "legacy frontend substitution helper must stay deleted"
          );
          const a = new Tensor([[1, 2], [3, 4]]);
          const b = new Tensor([[5, 6], [7, 8]]);
          const formula = "ij,jk->ik";
          const logical = pg._core.ffi.poly_einsum(
            a._ctx,
            formula,
            [{ _uop: a._logicalUopRaw() }, { _uop: b._logicalUopRaw() }]
          ).uop;
          const physical = pg._core.ffi.poly_einsum(
            a._ctx,
            formula,
            [{ _uop: a._currentUopRaw() }, { _uop: b._currentUopRaw() }]
          ).uop;
          const out = Tensor.einsum(formula, a, b);
          assertShape(out.shape, [2, 2]);
          assert(
            out.uopLogical.key === String(pg._core.ffi.poly_uop_key(logical)),
            "einsum logical graph differs"
          );
          assert(
            out.uop.key === String(pg._core.ffi.poly_uop_key(physical)),
            "einsum physical graph differs"
          );
          assertClose(await out.toArray(), [19, 22, 43, 50]);
        });
        await test("rearrange forwards named axis sizes on native and WASM", async () => {
          assert(
            Tensor.prototype._physicalizeResult === void 0,
            "legacy frontend substitution helper must stay deleted"
          );
          const source = Tensor.arange(6);
          const out = source.rearrange("(h w) -> h w", { h: 2, w: 3 });
          const expectedLogical = pg._core.ffi.poly_rearrange(
            source._ctx,
            "(h w) -> h w",
            source.uopLogical.raw,
            source.shape,
            { h: 2, w: 3 }
          );
          const expectedPhysical = pg._core.ffi.poly_rearrange(
            source._ctx,
            "(h w) -> h w",
            source.uop.raw,
            source.shape,
            { h: 2, w: 3 }
          );
          assert(
            out.uopLogical.key === String(pg._core.ffi.poly_uop_key(expectedLogical.uop)),
            "rearrange logical graph differs"
          );
          assert(
            out.uop.key === String(pg._core.ffi.poly_uop_key(expectedPhysical.uop)),
            "rearrange physical graph differs"
          );
          assertShape(out.shape, [2, 3]);
          assertClose(await out.toArray(), [0, 1, 2, 3, 4, 5]);
          const moved = (await Tensor.empty([6], { device: "cpu" }).realize()).to("cuda").to("cpu").reshape([2, 3]);
          const movedOut = moved.rearrange("h w -> w h");
          const expectedMoved = pg._core.ffi.poly_rearrange(
            moved._ctx,
            "h w -> w h",
            moved.uop.raw,
            moved.shape,
            {}
          );
          assert(
            movedOut.uop.key === String(pg._core.ffi.poly_uop_key(expectedMoved.uop)),
            "rearrange lost the nested physical occurrence"
          );
        });
        await test("einsum and rearrange reject malformed core inputs", async () => {
          const x = new Tensor([1, 2, 3]);
          let message = "";
          try {
            Tensor.einsum("i->z", x);
          } catch (error) {
            message = String(error && error.message ? error.message : error);
          }
          assert(/poly_einsum failed/.test(message), `unexpected einsum error: ${message}`);
          for (const formula of ["invalid", `${"a".repeat(300)}->a`, "a->a->a", "((a))->a"]) {
            message = "";
            try {
              x.rearrange(formula);
            } catch (error) {
              message = String(error && error.message ? error.message : error);
            }
            assert(
              /poly_rearrange failed/.test(message),
              `unexpected rearrange error for ${formula.slice(0, 24)}: ${message}`
            );
          }
        });
        await test("linalg construction bypasses frontend substitution", async () => {
          assert(
            Tensor.prototype._physicalizeResult === void 0,
            "legacy frontend substitution helper must stay deleted"
          );
          const a = new Tensor([[4, 2], [2, 5]]);
          const b = new Tensor([1, 3]);
          const lower = new Tensor([[2, 0], [1, 3]]);
          const [q, r] = a.qr();
          assertShape(q.shape, [2, 2]);
          assertShape(r.shape, [2, 2]);
          assert(Number.isFinite(await q.sum().item()), "qr result must execute");
          assertShape(lower.triangularSolve(b).shape, [2]);
          const chol = a.cholesky();
          assertShape(chol.shape, [2, 2]);
          assertShape(chol.choleskySolve(b).shape, [2]);
          assertShape(a.solve(b).shape, [2]);
          assertShape(
            new Tensor([[1, 0], [1, 1], [1, 2]]).lstsq(new Tensor([1, 2, 3])).shape,
            [2]
          );
        });
        await test("qr matches tinygrad probe", async () => {
          const cases = [
            { arr: [[1, 2], [3, 4]], q: [2, 2], r: [2, 2], flat: [1, 2, 3, 4] },
            { arr: [[1, 2], [3, 4], [5, 6]], q: [3, 3], r: [3, 2], flat: [1, 2, 3, 4, 5, 6] },
            { arr: [[1, 2, 3], [4, 5, 6]], q: [2, 2], r: [2, 3], flat: [1, 2, 3, 4, 5, 6] },
            { arr: [[0, 1], [0, 2]], q: [2, 2], r: [2, 2], flat: [0, 1, 0, 2] },
            {
              arr: [[[1, 2], [3, 4]], [[2, 0], [0, 2]]],
              q: [2, 2, 2],
              r: [2, 2, 2],
              flat: [1, 2, 3, 4, 2, 0, 0, 2]
            },
            {
              arr: [[[1, 2], [3, 4], [5, 6]], [[2, 1], [0, 3], [4, 5]]],
              q: [2, 3, 3],
              r: [2, 3, 2],
              flat: [1, 2, 3, 4, 5, 6, 2, 1, 0, 3, 4, 5]
            },
            {
              arr: [[[1, 2, 3], [4, 5, 6]], [[2, 1, 0], [0, 3, 4]]],
              q: [2, 2, 2],
              r: [2, 2, 3],
              flat: [1, 2, 3, 4, 5, 6, 2, 1, 0, 0, 3, 4]
            }
          ];
          for (const c of cases) {
            const pair = new Tensor(c.arr).qr();
            assertShape(pair[0].shape, c.q);
            assertShape(pair[1].shape, c.r);
            assertClose(await pair[0].dot(pair[1]).toArray(), c.flat, 2e-3);
            const qVals = Array.from(await pair[0].toArray());
            const rVals = Array.from(await pair[1].toArray());
            const vals = qVals.concat(rVals);
            for (const v of vals) assert(Number.isFinite(v), `expected finite QR value, got ${v}`);
          }
        });
        await test("qr reduced and r modes match reference shapes", async () => {
          const cases = [
            { arr: [[1, 2], [3, 4], [5, 6]], q: [3, 2], r: [2, 2], flat: [1, 2, 3, 4, 5, 6] },
            { arr: [[1, 2, 3], [4, 5, 6]], q: [2, 2], r: [2, 3], flat: [1, 2, 3, 4, 5, 6] },
            {
              arr: [[[1, 2], [3, 4], [5, 6]], [[2, 1], [0, 3], [4, 5]]],
              q: [2, 3, 2],
              r: [2, 2, 2],
              flat: [1, 2, 3, 4, 5, 6, 2, 1, 0, 3, 4, 5]
            }
          ];
          for (const c of cases) {
            const pair = new Tensor(c.arr).qr("reduced");
            assertShape(pair[0].shape, c.q);
            assertShape(pair[1].shape, c.r);
            assertClose(await pair[0].dot(pair[1]).toArray(), c.flat, 2e-3);
            const rOnly = new Tensor(c.arr).qr("r");
            assertShape(rOnly.shape, c.r);
          }
          let ok = false;
          try {
            new Tensor([[1, 2], [3, 4]]).qr("raw");
          } catch (e) {
            ok = true;
          }
          assert(ok, "expected invalid QR mode to fail");
        });
        await test("triangularSolve matches numpy torch probe", async () => {
          const lower = [[2, 0, 0], [1, 3, 0], [-2, 0.5, 4]];
          const upper = [[2, -1, 0.5], [0, 3, 2], [0, 0, 4]];
          const lowerUnit = [[5, 0, 0], [1, 7, 0], [-2, 0.5, 9]];
          const bVec = [2, 7, 9];
          const bMat = [[2, 1], [7, 2], [9, 3]];
          const lowerBatch = [
            [[2, 0, 0], [1, 3, 0], [-2, 0.5, 4]],
            [[3, 0, 0], [1, 4, 0], [-2, 0.5, 5]]
          ];
          const bBatch = [
            [[2, 1], [7, 2], [9, 3]],
            [[3, 2], [8, 3], [10, 4]]
          ];
          const cases = [
            { a: lower, b: bVec, opts: {}, shape: [3], out: [1, 2, 2.5] },
            { a: lower, b: bMat, opts: {}, shape: [3, 2], out: [1, 0.5, 2, 0.5, 2.5, 0.9375] },
            {
              a: upper,
              b: bMat,
              opts: { upper: true },
              shape: [3, 2],
              out: [0.8541666865, 0.3958333433, 0.8333333135, 0.1666666716, 2.25, 0.75]
            },
            {
              a: lower,
              b: bMat,
              opts: { transposeA: true },
              shape: [3, 2],
              out: [2.2708332539, 0.9791666865, 1.9583333731, 0.5416666865, 2.25, 0.75]
            },
            {
              a: upper,
              b: bMat,
              opts: { upper: true, transposeA: true },
              shape: [3, 2],
              out: [1, 0.5, 2.6666667461, 0.8333333135, 0.7916666865, 0.2708333433]
            },
            {
              a: lowerUnit,
              b: bMat,
              opts: { unitDiagonal: true },
              shape: [3, 2],
              out: [2, 1, 5, 1, 10.5, 4.5]
            },
            {
              a: lowerBatch,
              b: bBatch,
              opts: {},
              shape: [2, 3, 2],
              out: [1, 0.5, 2, 0.5, 2.5, 0.9375, 1, 0.6666666865, 1.75, 0.5833333135, 2.2249999046, 1.0083333254]
            },
            {
              a: lowerBatch,
              b: bVec,
              opts: {},
              shape: [2, 3],
              out: [1, 2, 2.5, 0.6666666865, 1.5833333731, 1.9083333015]
            }
          ];
          for (const c of cases) {
            const x = new Tensor(c.a).triangularSolve(new Tensor(c.b), c.opts);
            assertShape(x.shape, c.shape);
            assertClose(await x.toArray(), c.out, 2e-4);
          }
        });
        await test("cholesky matches numpy torch probe", async () => {
          const cases = [
            { a: [[4]], shape: [1, 1], out: [2] },
            { a: [[4, 2], [2, 5]], shape: [2, 2], out: [2, 0, 1, 2] },
            {
              a: [[6, 2, 1], [2, 5, 2], [1, 2, 4]],
              shape: [3, 3],
              out: [2.4494898319, 0, 0, 0.8164966106, 2.0816659927, 0, 0.4082483053, 0.8006407619, 1.7867029905]
            },
            {
              a: [[4, 0, 0, 0], [0, 4, 0, 0], [0, 0, 4, 0], [0, 0, 0, 4]],
              shape: [4, 4],
              out: [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2]
            },
            {
              a: [[[4, 2], [2, 5]], [[9, 3], [3, 2]]],
              shape: [2, 2, 2],
              out: [2, 0, 1, 2, 3, 0, 1, 1]
            }
          ];
          for (const c of cases) {
            const l = new Tensor(c.a).cholesky();
            assertShape(l.shape, c.shape);
            assertClose(await l.toArray(), c.out, 2e-4);
          }
          const u = new Tensor([[4, 2], [2, 5]]).cholesky({ upper: true });
          assertShape(u.shape, [2, 2]);
          assertClose(await u.toArray(), [2, 1, 0, 2], 2e-4);
        });
        await test("choleskySolve matches torch probe", async () => {
          const a = [[4, 2], [2, 5]];
          const b = [[1, 2], [3, 4]];
          for (const upper of [false, true]) {
            const f = new Tensor(a).cholesky({ upper });
            const x = f.choleskySolve(new Tensor(b), { upper });
            assertShape(x.shape, [2, 2]);
            assertClose(await x.toArray(), [-0.0625, 0.125, 0.625, 0.75], 2e-4);
          }
          const ab = [a, [[9, 3], [3, 2]]];
          const bbVec = [[1, 3], [2, 4]];
          for (const upper of [false, true]) {
            const f = new Tensor(ab).cholesky({ upper });
            const x = f.choleskySolve(new Tensor(bbVec), { upper });
            assertShape(x.shape, [2, 2]);
            assertClose(await x.toArray(), [-0.0625, 0.625, -0.8888889, 3.3333333], 5e-4);
            const xb = f.choleskySolve(new Tensor([1, 4]), { upper });
            assertShape(xb.shape, [2, 2]);
            assertClose(await xb.toArray(), [-0.1875, 0.875, -1.1111112, 3.6666667], 6e-4);
          }
        });
        await test("solve matches numpy torch probe", async () => {
          const a = [[2, 1], [1, 3]];
          const bVec = [1, 4];
          const bMat = [[1, 2], [3, 4]];
          const xVec = new Tensor(a).solve(new Tensor(bVec));
          assertShape(xVec.shape, [2]);
          assertClose(await xVec.toArray(), [-0.2, 1.4], 3e-4);
          const xMat = new Tensor(a).solve(new Tensor(bMat));
          assertShape(xMat.shape, [2, 2]);
          assertClose(await xMat.toArray(), [0, 0.4, 1, 1.2], 3e-4);
          const pivotA = [[0, 2], [1, 3]];
          const pivotBVec = [4, 5];
          const pivotBMat = [[4, 1], [5, 2]];
          const xpVec = new Tensor(pivotA).solve(new Tensor(pivotBVec));
          assertShape(xpVec.shape, [2]);
          assertClose(await xpVec.toArray(), [-1, 2], 3e-4);
          const xpMat = new Tensor(pivotA).solve(new Tensor(pivotBMat));
          assertShape(xpMat.shape, [2, 2]);
          assertClose(await xpMat.toArray(), [-1, 0.5, 2, 0.5], 3e-4);
          const ab = [a, [[3, 1], [1, 4]]];
          const bb = [bMat, [[2, 3], [4, 5]]];
          const xb = new Tensor(ab).solve(new Tensor(bb));
          assertShape(xb.shape, [2, 2, 2]);
          assertClose(await xb.toArray(), [0, 0.4, 1, 1.2, 0.3636363745, 0.6363636255, 0.9090909362, 1.0909091234], 3e-4);
          const bbVec = [bVec, [2, 5]];
          const xbVec = new Tensor(ab).solve(new Tensor(bbVec));
          assertShape(xbVec.shape, [2, 2]);
          assertClose(await xbVec.toArray(), [-0.2, 1.4, 0.27272728, 1.1818182], 4e-4);
          const pivotAB = [pivotA, [[3, 1], [0, 2]]];
          const pivotBBVec = [pivotBVec, [7, 4]];
          const xpbVec = new Tensor(pivotAB).solve(new Tensor(pivotBBVec));
          assertShape(xpbVec.shape, [2, 2]);
          assertClose(await xpbVec.toArray(), [-1, 2, 1.6666667, 2], 4e-4);
          const xbBroadcastVec = new Tensor(ab).solve(new Tensor(bVec));
          assertShape(xbBroadcastVec.shape, [2, 2]);
          assertClose(await xbBroadcastVec.toArray(), [-0.2, 1.4, 0, 1], 4e-4);
          const xbSingletonMatrix = new Tensor(ab).solve(new Tensor([bMat]));
          assertShape(xbSingletonMatrix.shape, [2, 2, 2]);
          assertClose(await xbSingletonMatrix.toArray(), [0, 0.4, 1, 1.2, 0.09090909, 0.36363637, 0.7272727, 0.9090909], 5e-4);
          let ok = false;
          try {
            new Tensor([[1, 1, 1], [1, 1, 1]]).solve(new Tensor([1, 1]));
          } catch (e) {
            ok = true;
          }
          assert(ok, "expected solve to reject non-square A");
        });
        await test("lstsq matches numpy torch probe", async () => {
          const a = [[1, 0], [1, 1], [1, 2]];
          const bVec = [1, 2, 2.5];
          const bMat = [[1, 0.5], [2, 1], [2.5, 1.5]];
          const xVec = new Tensor(a).lstsq(new Tensor(bVec));
          assertShape(xVec.shape, [2]);
          assertClose(await xVec.toArray(), [1.0833334, 0.75], 5e-4);
          const xMat = new Tensor(a).lstsq(new Tensor(bMat));
          assertShape(xMat.shape, [2, 2]);
          assertClose(await xMat.toArray(), [1.0833334, 0.5, 0.75, 0.5], 5e-4);
          const ab = [a, [[1, 0], [1, 1.5], [1, 3]]];
          const bb = [bMat, [[1.25, 0.75], [2.25, 1.25], [2.75, 1.75]]];
          const xb = new Tensor(ab).lstsq(new Tensor(bb));
          assertShape(xb.shape, [2, 2, 2]);
          assertClose(await xb.toArray(), [1.0833334, 0.5, 0.75, 0.5, 1.3333334, 0.75, 0.5, 0.33333334], 6e-4);
          const bbVec = [bVec, [1.25, 2.25, 2.75]];
          const xbVec = new Tensor(ab).lstsq(new Tensor(bbVec));
          assertShape(xbVec.shape, [2, 2]);
          assertClose(await xbVec.toArray(), [1.0833334, 0.75, 1.3333334, 0.5], 6e-4);
          const xbBroadcastVec = new Tensor(ab).lstsq(new Tensor(bVec));
          assertShape(xbBroadcastVec.shape, [2, 2]);
          assertClose(await xbBroadcastVec.toArray(), [1.0833334, 0.75, 1.0833334, 0.5], 6e-4);
          const wide = [[1, 2, 0], [0, 1, 1]];
          const wideVec = new Tensor(wide).lstsq(new Tensor([1, 2]));
          assertShape(wideVec.shape, [3]);
          assertClose(await wideVec.toArray(), [-0.33333334, 0.6666667, 1.3333334], 8e-4);
          const wideMat = new Tensor(wide).lstsq(new Tensor([[1, 3], [2, 4]]));
          assertShape(wideMat.shape, [3, 2]);
          assertClose(
            await wideMat.toArray(),
            [-0.33333334, -0.33333334, 0.6666667, 1.6666666, 1.3333334, 2.3333333],
            1e-3
          );
          const rankSquareVec = new Tensor([[1, 1], [2, 2]]).lstsq(new Tensor([3, 6]));
          assertShape(rankSquareVec.shape, [2]);
          assertClose(await rankSquareVec.toArray(), [1.5, 1.5], 2e-3);
          const rankSquareMat = new Tensor([[1, 1], [2, 2]]).lstsq(new Tensor([[3, 1], [6, 2]]));
          assertShape(rankSquareMat.shape, [2, 2]);
          assertClose(await rankSquareMat.toArray(), [1.5, 0.5, 1.5, 0.5], 2e-3);
          const rankTall = new Tensor([[1, 1], [2, 2], [3, 3]]).lstsq(new Tensor([1, 2, 3]));
          assertShape(rankTall.shape, [2]);
          assertClose(await rankTall.toArray(), [0.5, 0.5], 2e-3);
          const rankWide = new Tensor([[1, 1, 0], [2, 2, 0]]).lstsq(new Tensor([3, 6]));
          assertShape(rankWide.shape, [3]);
          assertClose(await rankWide.toArray(), [1.5, 1.5, 0], 2e-3);
          let ok = false;
          try {
            new Tensor([[1, 1, 1], [1, 1, 1]]).lstsq(new Tensor([1, 1, 1]));
          } catch (e) {
            ok = true;
          }
          assert(ok, "expected lstsq to reject incompatible RHS rows");
        });
        await test("crossEntropy with sparse targets", async () => {
          const logits = new Tensor([[0, 0, 0], [0, 0, 0]]);
          const target = new Tensor([0, 2], { dtype: "int32" });
          const loss = await logits.crossEntropy(target);
          assertShape(loss.shape, []);
          assertClose(await loss.toArray(), [Math.log(3)]);
        });
        await test("crossEntropy with dense targets", async () => {
          const logits = new Tensor([[0, 0, 0], [0, 0, 0]]);
          const target = new Tensor([[1, 0, 0], [0, 0, 1]]);
          const loss = await logits.crossEntropy(target);
          assertShape(loss.shape, []);
          assertClose(await loss.toArray(), [Math.log(3)]);
        });
        await test("crossEntropy matches pinned expression without substitution", async () => {
          assert(
            Tensor.prototype._physicalizeResult === void 0,
            "legacy frontend substitution helper must stay deleted"
          );
          const logits = new Tensor([[-1, 2, -3], [1, -2, 3]]);
          const sparse = new Tensor([1, 2], { dtype: "int32" });
          const dense = new Tensor([[0, 1, 0], [0, 0, 1]]);
          const pinnedExpression = (target, reduction = "mean", labelSmoothing = 0) => {
            const classesDim = 1;
            if (JSON.stringify(logits.shape) !== JSON.stringify(target.shape)) {
              target = target.unsqueeze(classesDim)._oneHotAlongDim(
                logits.shape[classesDim],
                classesDim
              );
            }
            target = target.mul(1 - labelSmoothing).add(
              labelSmoothing / target.shape[classesDim]
            );
            const reduced = logits.logSoftmax(classesDim).mul(target).sum(classesDim);
            if (reduction === "none") return reduced.neg();
            if (reduction === "sum") return reduced.sum().neg();
            if (reduction === "mean") return reduced.mean().neg();
            throw new Error(`invalid reduction ${reduction}`);
          };
          const pairs = [];
          for (const [target, reduction, labelSmoothing] of [
            [sparse, "mean", 0],
            [dense, "mean", 0],
            [dense, "none", 0],
            [dense, "sum", 0],
            [dense, "mean", 0.2]
          ]) {
            const actual = logits.crossEntropy(target, reduction, labelSmoothing);
            const expected = pinnedExpression(target, reduction, labelSmoothing);
            assert(actual.uop.key === expected.uop.key, "crossEntropy physical graph differs");
            assert(actual.uopLogical.key === expected.uopLogical.key, "crossEntropy logical graph differs");
            pairs.push([actual, expected]);
          }
          for (const [actual, expected] of pairs) {
            assertClose(await actual.toArray(), await expected.toArray());
          }
        });
        await test("crossEntropy with sparse targets on non-last axis", async () => {
          const logits = new Tensor([
            [[0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 0]]
          ]);
          const target = new Tensor([
            [0, 2],
            [1, 0]
          ], { dtype: "int32" });
          const loss = await logits.crossEntropy(target, "mean", 0, -2);
          assertShape(loss.shape, []);
          assertClose(await loss.toArray(), [Math.log(3)]);
        });
        await test("crossEntropy default matches tinygrad class axis", async () => {
          const logits = new Tensor([
            [[0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 0]]
          ]);
          const target = new Tensor([
            [0, 2],
            [1, 0]
          ], { dtype: "int32" });
          const loss = await logits.crossEntropy(target);
          assertShape(loss.shape, []);
          assertClose(await loss.toArray(), [Math.log(3)]);
        });
        await test("crossEntropy with dense targets on non-last axis", async () => {
          const logits = new Tensor([
            [[0, 0], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 0]]
          ]);
          const target = new Tensor([
            [[1, 0], [0, 0], [0, 1]],
            [[0, 1], [1, 0], [0, 0]]
          ]);
          const loss = await logits.crossEntropy(target, "mean", 0, 1);
          assertShape(loss.shape, []);
          assertClose(await loss.toArray(), [Math.log(3)]);
        });
        await test("crossEntropy shape mismatch throws", async () => {
          const logits = new Tensor([[0, 0, 0], [0, 0, 0]]);
          const target = new Tensor([[1, 0], [0, 1]]);
          let ok = false;
          try {
            const loss = await logits.crossEntropy(target);
            ok = loss.shape.length === 0;
          } catch (e) {
            ok = true;
          }
          assert(ok, "expected crossEntropy to fail on shape mismatch");
        });
        console.log("\n-- Autograd --");
        await test("current autograd API has no requiresGrad constructor flag", async () => {
          let rejected = false;
          try {
            new Tensor([1], { dtype: "float32", requiresGrad: true });
          } catch (e) {
            rejected = e instanceof TypeError && e.message.includes("requiresGrad");
          }
          assert(rejected, "requiresGrad must be rejected like current tinygrad requires_grad");
        });
        await test("backward targets every live reachable floating Tensor", async () => {
          const x = new Tensor([2], { dtype: "float32" });
          const y = x.square();
          const loss = y.sum();
          await loss.backward();
          assertClose(await x.grad.toArray(), [4]);
          assertClose(await y.grad.toArray(), [1]);
          assertClose(await loss.grad.toArray(), [1]);
        });
        await test("grad: mul sum", async () => {
          const a = new Tensor([1, 2, 3], { dtype: "float32" });
          const b = new Tensor([4, 5, 6]);
          const loss = a.mul(b).sum();
          await loss.backward();
          assert(a.grad, "grad is null");
          assertClose(await a.grad.toArray(), [4, 5, 6]);
        });
        await test("grad: neg sum", async () => {
          const a = new Tensor([1, 2, 3], { dtype: "float32" });
          const loss = a.neg().sum();
          await loss.backward();
          assert(a.grad, "grad is null");
          assertClose(await a.grad.toArray(), [-1, -1, -1]);
        });
        await test("grad: matmul backward", async () => {
          const W = new Tensor([[1, 2], [3, 4]], { dtype: "float32" });
          const x = new Tensor([[1, 0]]);
          const loss = x.dot(W).sum();
          await loss.backward();
          assert(W.grad, "W.grad is null");
          assertClose(await W.grad.toArray(), [1, 1, 0, 0]);
        });
        await test("grad: relu backward", async () => {
          const a = new Tensor([-1, 2, -3, 4], { dtype: "float32" });
          const loss = a.relu().sum();
          await loss.backward();
          assert(a.grad, "grad is null");
          assertClose(await a.grad.toArray(), [0, 1, 0, 1]);
        });
        await test("grad: chain backward", async () => {
          const a = new Tensor([1, 2, 3], { dtype: "float32" });
          const loss = a.mul(a).sum();
          await loss.backward();
          assert(a.grad, "grad is null");
          assert(a.grad.uopPhysical, "gradient must store its exact physical root");
          assert(a.grad.uopPhysical.op === pg._core.ops.ADD, "square gradient physical root must be ADD");
          assert(a.grad.uopPhysical.key === a.grad.uop.key, "gradient current root must be physical");
          assertClose(await a.grad.toArray(), [2, 4, 6]);
        });
        await test("grad: backward uses current physical value after copyFrom", async () => {
          const weight = new Tensor([1], { dtype: "float32" }).mul(2);
          await weight.realize();
          const physicalBuffer = weight.uopPhysical.buffer.raw;
          pg._core.ffi.poly_buffer_ensure_device_allocated(
            weight._ctx,
            physicalBuffer,
            pg._core.deviceIds[weight.device.toLowerCase()]
          );
          pg._core.ffi.poly_buffer_write(weight._ctx, physicalBuffer, new Float32Array([3]));
          const loss = weight.square().sum();
          await loss.backward();
          assert(weight.grad, "weight.grad is null");
          assertClose(await weight.grad.toArray(), [6]);
        });
        console.log("\n-- Assign --");
        await test("assign basic", async () => {
          const a = new Tensor([1, 2, 3]);
          a.assign(a.add(10));
          await a.realize();
          assertClose(await a.toArray(), [11, 12, 13]);
        });
        await test("assign broadcasts rhs in core", async () => {
          const target = Tensor.zeros([2, 3]);
          target.assign(new Tensor([4, 5, 6], { dtype: "float32" }));
          await target.realize();
          assertClose(await target.toArray(), [4, 5, 6, 4, 5, 6]);
        });
        await test("assign rejects device mismatch", async () => {
          const a = new Tensor([1], { device: "cpu" });
          const v = new Tensor([5], { device: "cpu" }).to("cuda");
          let ok = false;
          try {
            a.assign(v);
          } catch (e) {
            ok = /assign device mismatch CPU != CUDA/.test(String(e && e.message ? e.message : e));
          }
          assert(ok, "expected assign device mismatch CPU != CUDA");
        });
        await test("assign rejects dtype mismatch", async () => {
          const a = new Tensor([1], { dtype: "float32" });
          const mismatchDtype = supportsF64 ? "float64" : "int32";
          const v = new Tensor([5], { dtype: mismatchDtype });
          let ok = false;
          try {
            a.assign(v);
          } catch (e) {
            ok = new RegExp(`assign dtype mismatch float32 != ${mismatchDtype}`).test(
              String(e && e.message ? e.message : e)
            );
          }
          assert(ok, `expected assign dtype mismatch float32 != ${mismatchDtype}`);
        });
        await test("assign to same-device place keeps place target", async () => {
          const a = new Tensor([1], { device: "cpu" }).to("cuda");
          const v = new Tensor([5], { device: "cpu" }).to("cuda");
          assert(a.assign(v) === a, "assign should return self");
          assert(a.device === "CUDA", "assign should keep CUDA placement");
          assert(!a.uop.hasBufferIdentity(), "assign before realize should be an effect graph");
        });
        await test("assign realized targets reuse current buffer", async () => {
          const a = new Tensor([1]);
          await a.realize();
          const aBuffer = a.uop.buffer.key;
          a.assign(new Tensor([5]));
          assert(!a.uop.hasBufferIdentity(), "assign before realize should be an effect graph");
          await a.realize();
          assert(a.uop.buffer.key === aBuffer, "realized assign should reuse target buffer");
          assertClose(await a.toArray(), [5]);
          const x = await new Tensor([1]).add(1).realize();
          const xBuffer = x.uop.buffer.key;
          x.assign(new Tensor([9]));
          await x.realize();
          assert(x.uop.buffer.key === xBuffer, "realized expression assign should reuse target buffer");
          assertClose(await x.toArray(), [9]);
        });
        await test("shared-storage to copies and preserves source across assign", async () => {
          const source = await new Tensor([1, 2, 3], { device: "cpu" }).realize();
          const sourceBuffer = source.uop.buffer.key;
          const target = await source.to("interp").realize();
          const targetBuffer = target.uop.buffer.key;
          assert(targetBuffer !== sourceBuffer, "cross-device to should allocate a distinct target buffer");
          assertClose(await source.toArray(), [1, 2, 3]);
          assertClose(await target.toArray(), [1, 2, 3]);
          for (const values of [[9, 8, 7], [4, 5, 6]]) {
            target.assign(new Tensor(values, { device: "interp" }));
            await target.realize();
            assert(target.uop.buffer.key === targetBuffer, "assign should retain the copied target buffer");
            assertClose(await source.toArray(), [1, 2, 3]);
            assertClose(await target.toArray(), values);
          }
        });
        await test("copyFrom preserves buffer identity and updates JIT replay input", async () => {
          const x = Tensor.empty([3], { dtype: "float32" });
          const bufferKey = x.uop.buffer.key;
          x.copyFrom(new Float32Array([1, 2, 3]));
          assert(x.uop.buffer.key === bufferKey, "copyFrom should preserve input buffer identity");
          assertClose(await x.toArray(), [1, 2, 3]);
          const f = pg.jit((a) => a.add(1).realize());
          assertClose(await (await f(x)).toArray(), [2, 3, 4]);
          pg.resetCounters();
          const captured = await f(x);
          let counterStats = pg.stats().coreStats;
          const expectedOps = pg.device === "x86" ? 0 : 3;
          const expectedMem = pg.device === "x86" ? 0 : 24;
          assert(
            counterStats.globalOps === expectedOps && counterStats.globalMem === expectedMem && counterStats.kernelCount === 1,
            "JIT capture execution should update counters exactly once"
          );
          assertClose(await captured.toArray(), [2, 3, 4]);
          assert(f.scheduleCount === 1, `expected one captured schedule, got ${f.scheduleCount}`);
          const replayBufferKey = x.uop.buffer.key;
          x.updateFrom(new Float32Array([10, 20, 30]));
          assert(x.uop.buffer.key === replayBufferKey, "updateFrom should preserve captured input buffer identity");
          pg.resetCounters();
          const replayed = await f(x);
          counterStats = pg.stats().coreStats;
          assert(
            counterStats.globalOps === expectedOps && counterStats.globalMem === expectedMem && counterStats.kernelCount === 1,
            "JIT replay should update counters exactly once"
          );
          assertClose(await replayed.toArray(), [11, 21, 31]);
          f.dispose();
        });
        await test("pure movement view realize is zero call", async () => {
          const x = await Tensor.arange(8, { dtype: "float32" }).realize();
          const out = x.reshape(2, 4).permute(1, 0);
          const memBefore = pg.stats().coreStats.memUsed;
          pg.resetCounters();
          await out.realize();
          const stats = pg.stats().coreStats;
          assert(
            stats.globalOps === 0 && stats.globalMem === 0 && stats.kernelCount === 0,
            `pure view realize should execute zero calls, got ${stats.globalOps}/${stats.globalMem}/${stats.kernelCount}`
          );
          assert(stats.memUsed === memBefore, "pure view realize should not allocate storage");
          assertClose(await out.toArray(), [0, 4, 1, 5, 2, 6, 3, 7]);
        });
        await test("realized contiguous and readback reuse current buffer identity", async () => {
          const source = await new Tensor([0, 1, 2, 3, 4, 5, 6, 7], { dtype: "float32" }).add(1).preserveLogical().realize();
          const sourceCurrent = source.uop.key;
          const sourceLogical = source.uopLogical.key;
          assert(source.uopLogical.op === pg._core.ops.ADD, "source logical root should retain ADD provenance");
          assert(source.uop.hasBufferIdentity(), "realized source should have buffer identity");
          const out = source.contiguous();
          assert(out !== source, "contiguous should return a new Tensor object");
          assert(out.uopLogical.key === sourceLogical, "device-free logical result should fold contiguous");
          assert(
            out.uopPhysical && out.uopPhysical.key === sourceCurrent,
            "physical result should reuse the exact current buffer"
          );
          assert(out.uop.key === sourceCurrent, "current result should reuse the exact current buffer");
          assert(source.uopLogical.key === sourceLogical, "source logical provenance should be unchanged");
          pg.resetCounters();
          await out.realize();
          let stats = pg.stats().coreStats;
          assert(
            stats.globalOps === 0 && stats.globalMem === 0 && stats.kernelCount === 0,
            `realized contiguous should execute zero calls, got ${stats.globalOps}/${stats.globalMem}/${stats.kernelCount}`
          );
          pg.resetCounters();
          assertClose(await out.toArray(), [1, 2, 3, 4, 5, 6, 7, 8]);
          stats = pg.stats().coreStats;
          assert(
            stats.globalOps === 0 && stats.globalMem === 0 && stats.kernelCount === 0,
            `realized readback should execute zero calls, got ${stats.globalOps}/${stats.globalMem}/${stats.kernelCount}`
          );
          const reshaped = source.reshape(2, 4);
          assert(reshaped.uop.hasBufferIdentity(), "reshape of current buffer should retain identity");
          pg.resetCounters();
          assertClose(await reshaped.toArray(), [1, 2, 3, 4, 5, 6, 7, 8]);
          stats = pg.stats().coreStats;
          assert(
            stats.globalOps === 0 && stats.globalMem === 0 && stats.kernelCount === 0,
            "reshape readback should execute zero calls"
          );
          const permuted = source.reshape(2, 4).permute(1, 0);
          pg.resetCounters();
          assertClose(await permuted.toArray(), [1, 5, 2, 6, 3, 7, 4, 8]);
          assert(pg.stats().coreStats.kernelCount === 1, "noncontiguous permute should materialize once");
          const casted = source.cast("int32");
          pg.resetCounters();
          assertClose(await casted.toArray(), [1, 2, 3, 4, 5, 6, 7, 8]);
          assert(pg.stats().coreStats.kernelCount === 1, "lazy cast should materialize once");
        });
        console.log("\n-- Static constructors --");
        await test("zeros", async () => {
          const t = Tensor.zeros(3);
          assertClose(await t.toArray(), [0, 0, 0]);
        });
        await test("ones", async () => {
          const t = Tensor.ones(2, 2);
          assertClose(await t.toArray(), [1, 1, 1, 1]);
        });
        await test("full", async () => {
          const t = Tensor.full([3], 7);
          assert(t.dtype === "int32", `buffered integer full should commit int32 storage, got ${t.dtype}`);
          assert(t.uop.op === pg._core.ops.AFTER, "buffered full should produce AFTER");
          assert(t.uop.src[0].op === pg._core.ops.BUFFER, "buffered full should own BUFFER storage");
          assert(t.uop.src[1].op === pg._core.ops.STORE, "buffered full should contain STORE");
          assert(t.uop.src[1].src[0].key === t.uop.src[0].key, "STORE must target the same BUFFER");
          assert(t.uop.src[1].src[1].op === pg._core.ops.EXPAND, "STORE value should stay EXPAND");
          assertClose(await t.toArray(), [7, 7, 7]);
          const raw = Tensor.full([3], 7, { buffer: false });
          assert(raw.dtype === "weakint", `unbuffered integer full should stay weakint, got ${raw.dtype}`);
          assert(raw.uop.op === pg._core.ops.EXPAND, "unbuffered full should stay EXPAND");
          const zeroBroadcast = Tensor.full([0, 3], 1.5, { buffer: false }).add(Tensor.full([1, 3], 2.5, { buffer: false }));
          assertShape(zeroBroadcast.shape, [0, 3]);
          assert(zeroBroadcast.uop.op === pg._core.ops.ADD, "weak full add should stay ADD");
          for (const source of zeroBroadcast.uop.src) {
            assert(source.op === pg._core.ops.EXPAND, "promoted weak full should be direct EXPAND");
            assert(source.src[0].op === pg._core.ops.CONST, "promoted weak full should drop old movement chain");
          }
          assertClose(await zeroBroadcast.toArray(), []);
          const nonempty = Tensor.full([2, 3], 1.5, { buffer: false }).add(Tensor.full([1, 3], 2.5, { buffer: false }));
          assertClose(await nonempty.toArray(), [4, 4, 4, 4, 4, 4]);
          let weakFullRejected = false;
          try {
            Tensor.full([2], 1, { dtype: "weakfloat" });
          } catch (_) {
            weakFullRejected = true;
          }
          assert(weakFullRejected, "explicit weak full storage must be rejected");
          let weakEmptyRejected = false;
          try {
            Tensor.empty([2], { dtype: "weakfloat" });
          } catch (_) {
            weakEmptyRejected = true;
          }
          assert(weakEmptyRejected, "explicit weak empty storage must be rejected");
        });
        await test("arange", async () => {
          const t = Tensor.arange(4);
          assertClose(await t.toArray(), [0, 1, 2, 3]);
        });
        await test("eye", async () => {
          const t = Tensor.eye(2);
          assertClose(await t.toArray(), [1, 0, 0, 1]);
        });
        await test("pure constructors store the pinned device-free root", async () => {
          const values = [
            Tensor.full([2, 3], 2, { buffer: false }),
            Tensor.arange(4),
            Tensor.linspace(0, 1, 4),
            Tensor.eye(3)
          ];
          for (const value of values) {
            assert(value.uopPhysical, "pure constructor must store a physical root");
            assert(value.uopLogical, "pure constructor must store a logical root");
            assert(
              value.uopPhysical.key === value.uopLogical.key,
              "pure constructor roots must be the same UOp"
            );
          }
          const arange = values[1];
          const moved = arange.to("cuda");
          assert(moved.uop.key === arange.uop.key, "device-free Tensor.to must preserve the root");
        });
        await test("internal scalars store typed current roots", async () => {
          const cases = [
            [Tensor.empty([2], { dtype: "bool" }), true, "bool"],
            [Tensor.empty([2], { dtype: "int32" }), 7, "weakint"],
            [Tensor.empty([2], { dtype: "float32" }), 1, "weakint"]
          ];
          for (const [source, value, dtype] of cases) {
            const scalar = source._ensureTensor(value);
            assert(scalar.dtype === dtype, `expected ${dtype}, got ${scalar.dtype}`);
            assert(
              scalar.uopLogical.op === pg._core.ops.CONST,
              "internal scalar logical root must be CONST"
            );
            assert(
              scalar.uopPhysical.op === pg._core.ops.CONST,
              "internal scalar physical root must be CONST"
            );
            assert(
              scalar.uopLogical.key === scalar.uopPhysical.key,
              "internal scalar roots must be the same UOp"
            );
          }
        });
        console.log("\n-- Lazy RNG --");
        await test("rand uniform [0,1)", async () => {
          Tensor.manual_seed(42);
          const t = Tensor.rand(100);
          const arr = await t.toArray();
          assert(arr.length === 100, `Expected 100 elements, got ${arr.length}`);
          let min = arr[0], max = arr[0];
          for (let i = 1; i < arr.length; i++) {
            if (arr[i] < min) min = arr[i];
            if (arr[i] > max) max = arr[i];
          }
          assert(min >= 0 && max < 1, `Out of range: min=${min}, max=${max}`);
        });
        await test("uniform supports bounds and rejects empty intervals", async () => {
          Tensor.manual_seed(42);
          const arr = await Tensor.uniform(100, { low: -2, high: 3 }).toArray();
          assert(arr.every((value) => value >= -2 && value < 3), "uniform value outside requested range");
          let rejected = false;
          try {
            Tensor.uniform(2, { low: 1, high: 1 });
          } catch (error) {
            rejected = String(error && error.message).includes("low < high");
          }
          assert(rejected, "uniform should reject an empty interval");
        });
        await test("scaled_uniform matches the pinned initializer expression", async () => {
          Tensor.manual_seed(42);
          const actual = Tensor.scaled_uniform(2, 3);
          Tensor.manual_seed(42);
          const expected = Tensor.uniform(2, 3, { low: -1, high: 1 }).mul(6 ** -0.5);
          const [a, b] = await Promise.all([actual.toArray(), expected.toArray()]);
          assert(
            a.length === b.length && a.every((value, i) => value === b[i]),
            "scaled_uniform values must match the pinned expression"
          );
        });
        await test("glorot_uniform matches the pinned initializer expression", async () => {
          const bound = Math.sqrt(6 / (2 + 3));
          Tensor.manual_seed(42);
          const actual = Tensor.glorot_uniform(2, 3);
          Tensor.manual_seed(42);
          const expected = Tensor.uniform(2, 3, { low: -bound, high: bound });
          const [a, b] = await Promise.all([actual.toArray(), expected.toArray()]);
          assert(
            a.length === b.length && a.every((value, i) => value === b[i]),
            "glorot_uniform values must match the pinned expression"
          );
        });
        await test("randn gaussian", async () => {
          Tensor.manual_seed(99);
          const t = Tensor.randn(1e3);
          const arr = await t.toArray();
          let sum = 0;
          for (let i = 0; i < arr.length; i++) sum += arr[i];
          const mean = sum / arr.length;
          assert(Math.abs(mean) < 0.3, `Mean too far from 0: ${mean}`);
        });
        await test("rand deterministic with seed", async () => {
          Tensor.manual_seed(1337);
          const a = await Tensor.rand(8).toArray();
          Tensor.manual_seed(1337);
          const b = await Tensor.rand(8).toArray();
          for (let i = 0; i < 8; i++) {
            assert(a[i] === b[i], `Not deterministic at [${i}]: ${a[i]} vs ${b[i]}`);
          }
          const expected = [
            0.4886598587036133,
            0.3479880094528198,
            0.6593245267868042,
            0.6364744901657104,
            0.4711652994155884,
            0.14146876335144043,
            0.27809083461761475,
            0.049955129623413086
          ];
          for (let i = 0; i < expected.length; i++) {
            assert(a[i] === expected[i], `Pinned RNG mismatch at [${i}]: ${a[i]} vs ${expected[i]}`);
          }
        });
        await test("rand concatenation keeps vector gated-load alternatives executable", async () => {
          Tensor.manual_seed(201);
          const first = Tensor.rand(8), second = Tensor.rand(8);
          const joined = await first.cat(second).toArrayAsync();
          const expected = [...await first.toArrayAsync(), ...await second.toArrayAsync()];
          assertClose(joined, expected, 0);
        });
        console.log("\n-- Float64 --");
        await testIf(supportsF64, "f64: creation", async () => {
          const t = new Tensor([1.5, 2.5, 3.5], { dtype: "float64" });
          assertClose(await t.toArray(), [1.5, 2.5, 3.5]);
        });
        await testIf(supportsF64, "f64: zeros", async () => {
          const t = Tensor.zeros(3, { dtype: "float64" });
          assertClose(await t.toArray(), [0, 0, 0]);
        });
        await testIf(supportsF64, "f64: ones", async () => {
          const t = Tensor.ones(2, { dtype: "float64" });
          assertClose(await t.toArray(), [1, 1]);
        });
        await testIf(supportsF64, "f64: add", async () => {
          const a = new Tensor([1, 2, 3], { dtype: "float64" });
          const b = new Tensor([4, 5, 6], { dtype: "float64" });
          const arr = await a.add(b).toArray();
          console.log("DEBUG f64 add", Array.from(arr));
          assertClose(arr, [5, 7, 9]);
        });
        await testIf(supportsF64, "f64: mul", async () => {
          const a = new Tensor([2, 3], { dtype: "float64" });
          const b = new Tensor([4, 5], { dtype: "float64" });
          const arr = await a.mul(b).toArray();
          console.log("DEBUG f64 mul", Array.from(arr));
          assertClose(arr, [8, 15]);
        });
        await testIf(supportsF64, "f64: sum", async () => {
          const t = new Tensor([1, 2, 3], { dtype: "float64" });
          const v = await t.sum().item();
          console.log("DEBUG f64 sum", v);
          assert(Math.abs(v - 6) < 1e-10, `Expected 6, got ${v}`);
        });
        await testIf(supportsF64, "f64: backward", async () => {
          const a = new Tensor([1, 2, 3], { dtype: "float64" });
          const b = new Tensor([4, 5, 6], { dtype: "float64" });
          const loss = a.mul(b).sum();
          await loss.backward();
          assert(a.grad, "grad is null");
          assertClose(await a.grad.toArray(), [4, 5, 6]);
        });
        await test("f64: integer list still infers int32", async () => {
          const a = new Tensor([1, 2, 3]);
          assert(a.dtype === "int32", `expected int32, got ${a.dtype}`);
        });
        console.log("\n-- Cache consistency --");
        await test("repeated realize produces identical results", async () => {
          const a = new Tensor([1, 2, 3, 4]);
          const b = new Tensor([10, 20, 30, 40]);
          const r1 = await a.add(b).toArray();
          const r2 = await new Tensor([1, 2, 3, 4]).add(new Tensor([10, 20, 30, 40])).toArray();
          const r3 = await new Tensor([1, 2, 3, 4]).add(new Tensor([10, 20, 30, 40])).toArray();
          assertClose(r1, [11, 22, 33, 44]);
          assertClose(r2, [11, 22, 33, 44]);
          assertClose(r3, [11, 22, 33, 44]);
        });
        await test("realize retargets live tensors sharing lazy root", async () => {
          const a = new Tensor([1]);
          await a.realize();
          const x1 = a.add(1);
          const x2 = a.add(1);
          assert(x1.uop.key === x2.uop.key, "lazy roots should be hash-consed together");
          await x1.realize();
          assert(x1.uop.key === x2.uop.key, "shared lazy root should retarget to one realized root");
          assert(x1.uop.buffer.key === x2.uop.buffer.key, "shared lazy root should share realized buffer");
          assertClose(await x2.toArray(), [2]);
        });
        await test("realize rewrites downstream live graph", async () => {
          const a = new Tensor([1]);
          await a.realize();
          const x = a.add(1);
          const y = x.add(2);
          const oldYKey = y.uop.key;
          await x.realize();
          assert(y.uop.key !== oldYKey, "downstream graph should be rewritten through realized x");
          assertClose(await y.toArray(), [4]);
        });
        await test("separate realizes do not alias buffers", async () => {
          const a = new Tensor([1]);
          await a.realize();
          const y1 = await a.add(1).realize();
          const y2 = await a.add(1).realize();
          assert(y1.uop.buffer.key !== y2.uop.buffer.key, "separate materializations should get separate buffers");
          assertClose(await y1.toArray(), [2]);
          assertClose(await y2.toArray(), [2]);
        });
        await test("to keeps separate realized same-logical occurrences", async () => {
          const a = new Tensor([1], { device: "cpu" });
          await a.realize();
          const x1 = await a.add(1).realize();
          const x2 = await a.add(1).realize();
          assert(x1.uop.buffer.key !== x2.uop.buffer.key, "separate materializations should stay distinct");
          const x1Cuda = x1.to("cuda");
          const x2Cuda = x2.to("cuda");
          assert(x1Cuda.uop.op === pg._core.ops.COPY, "x1.to(cuda) should be an eager COPY");
          assert(x2Cuda.uop.op === pg._core.ops.COPY, "x2.to(cuda) should be an eager COPY");
          assert(x1Cuda.uop.buffer === null, "an unrealized COPY should not claim buffer identity");
          assert(x2Cuda.uop.buffer === null, "an unrealized COPY should not claim buffer identity");
          assert(
            x1Cuda.uop.src[0].buffer.key === x1.uop.buffer.key,
            "x1.to(cuda) COPY should point at x1 buffer"
          );
          assert(
            x2Cuda.uop.src[0].buffer.key === x2.uop.buffer.key,
            "x2.to(cuda) COPY should point at x2 buffer"
          );
          assert(
            x1Cuda.uop.src[0].buffer.key !== x2Cuda.uop.src[0].buffer.key,
            "to(cuda) COPY sources should not alias"
          );
          const y1 = x1Cuda.add(1);
          const y2 = x2Cuda.add(1);
          assert(y1.uop.key !== y2.uop.key, "downstream graphs should use distinct occurrence sources");
        });
        await test("nested to keeps realized current and export logical separate", async () => {
          const x = await new Tensor([1]).add(1).realize();
          const xBase = x.to("cpu");
          const xCuda = xBase.to("cuda");
          const xCpu = xCuda.to("cpu");
          assert(xCuda.uop.op === pg._core.ops.COPY, "CUDA move should be an eager COPY");
          assert(xCuda.uop.src[0].key === xBase.uop.key, "CUDA COPY should use the CPU occurrence");
          assert(xCpu.uop.op === pg._core.ops.COPY, "CPU roundtrip should be an eager COPY");
          assert(xCpu.uop.src[0].key === xCuda.uop.key, "CPU COPY should use the CUDA occurrence");
          assert(xCpu.uop.key !== x.uop.key, "roundtrip COPY should stay occurrence-distinct");
          assert(xCpu.uopPhysical.key === xCpu.uop.key, "physical root should be the current COPY");
          assert(xCpu.uopLogical.key === x.uopLogical.key, "nested to should preserve export logical root");
          const y = xCpu.add(1);
          assert(y.device === "CPU", "downstream value should keep selected CPU placement");
          assert(y.uop.key !== xCpu.uop.key, "downstream value should build a new current graph");
          const mixed = xBase.add(xCpu);
          assert(mixed.uop.op === pg._core.ops.ADD, "mixed result should be ADD");
          assert(mixed.uop.src[0].key === xBase.uop.key, "first ADD input should be CPU occurrence");
          assert(mixed.uop.src[1].key === xCpu.uop.key, "second ADD input should be CPU COPY");
          assert(mixed.uop.src[1].src[0].key === xCuda.uop.key, "CPU COPY should contain CUDA COPY");
          assert(
            mixed.uop.src[1].src[0].src[0].key === xBase.uop.key,
            "CUDA COPY should contain CPU occurrence"
          );
          assert(mixed.uopLogical.src[0].key === x.uopLogical.key, "logical lhs should be X");
          assert(mixed.uopLogical.src[1].key === x.uopLogical.key, "logical rhs should be X");
        });
        await test("composite ops preserve repeated logical physical occurrences", async () => {
          const x = await new Tensor([0, 0, 0, 0], { device: "cpu", dtype: "float32" }).reshape(1, 1, 2, 2).realize();
          const movedWeight = x.to("cuda").to("cpu");
          const conv = x.conv2d(movedWeight);
          assert(
            countGraphOp(conv.uop, pg._core.ops.COPY) === 2,
            "conv2d should retain the nested COPY weight occurrence"
          );
          assert(
            countGraphOp(conv.uop, pg._core.ops.EXPAND) === 1,
            "single-output conv2d should elide the no-op channel EXPAND"
          );
          const bnX = await new Tensor([0, 0, 0], { device: "cpu", dtype: "float32" }).reshape(1, 3, 1, 1).realize();
          const stat = await new Tensor([0, 0, 0], { device: "cpu", dtype: "float32" }).realize();
          const movedInvstd = stat.to("cuda").to("cpu");
          const bn = bnX.batchnorm(null, null, stat, movedInvstd, 1);
          assert(
            countGraphOp(bn.uop, pg._core.ops.COPY) === 2,
            "batchnorm should retain the nested COPY invstd occurrence"
          );
          const minX = await new Tensor([1], { device: "cpu", dtype: "float32" }).realize();
          const minMoved = minX.to("cuda").to("cpu");
          const minimum = minX.minimum(minMoved);
          assert(
            countGraphOp(minimum.uop, pg._core.ops.COPY) === 2,
            "minimum should retain the nested COPY right occurrence"
          );
          assert(minimum.uop.op === pg._core.ops.MUL, "minimum should end with inverse MUL");
          const maximum = minimum.uop.src[0];
          assert(maximum.op === pg._core.ops.MAX, "minimum should contain ordered MAX");
          assert(maximum.src[0].src[0].key === minX.uop.key, "minimum lhs occurrence mismatch");
          assert(maximum.src[1].src[0].key === minMoved.uop.key, "minimum rhs occurrence mismatch");
        });
        await test("repeated fused chain produces identical results", async () => {
          const a = [2, 3, 4];
          const b = [1, 1, 1];
          const expected = [(2 + 1) * 2 - 1, (3 + 1) * 3 - 1, (4 + 1) * 4 - 1];
          for (let i = 0; i < 3; i++) {
            const ta = new Tensor(a), tb = new Tensor(b);
            const r = await ta.add(tb).mul(ta).sub(tb).toArray();
            assertClose(r, expected);
          }
        });
        console.log("\n-- Missing math ops --");
        await test("pow integer", async () => {
          const t = new Tensor([2, 3, 4]);
          const r = t.pow(2);
          assertClose(await r.toArray(), [4, 9, 16]);
        });
        await test("pow float", async () => {
          const t = new Tensor([4, 9, 16]);
          const r = t.pow(0.5);
          assertClose(await r.toArray(), [2, 3, 4]);
        });
        await test("movement minimum owners: shrink bounds", async () => {
          const x = new Tensor([1, 2, 3, 4]);
          for (const bounds of [[2, 5], [0, 5], [5, 5], [-1, 3], [2, 1]]) {
            let rejected = false;
            try {
              x.shrink([bounds]);
            } catch (_) {
              rejected = true;
            }
            assert(rejected, `shrink must reject ${bounds}`);
          }
          assertClose(await x.shrink([[1, 3]]).toArray(), [2, 3], 0);
          assertShape(x.shrink([[4, 4]]).shape, [0]);
          assertClose(await x.pad([[2, 1]]).toArray(), [0, 0, 1, 2, 3, 4, 0], 0);
        });
        await test("movement minimum owners: boolean binary XOR and unary CMPNE", async () => {
          const x = new Tensor([false, false, true, true], { dtype: "bool" });
          const y = new Tensor([false, true, false, true], { dtype: "bool" });
          const out = x.minimum(y);
          for (const root of [out.uopLogical, out.uopPhysical]) {
            assert(root.op === pg._core.ops.XOR, "minimum inverse must use XOR");
            const maximum = root.src[0];
            assert(maximum.op === pg._core.ops.MAX);
            assert(maximum.src.every((s) => s.op === pg._core.ops.XOR));
          }
          assertClose(await out.toArray(), [0, 0, 0, 1], 0);
          assert(x.min().uopPhysical.op === pg._core.ops.CMPNE);
          assertClose(await x.minimum(true).toArray(), [0, 0, 1, 1], 0);
        });
        for (const op of ["all", "any", "cumsum", "cumprod", "cummax", "cummin"]) {
          await test(`scan owners: ${op} values and dtype`, async () => {
            const x = new Tensor([-3, -1, -2, -1], { dtype: "int8" });
            const result = x[op](0);
            const expected = {
              all: [1],
              any: [1],
              cumsum: [-3, -4, -6, -7],
              cumprod: [-3, 3, -6, 6],
              cummax: [-3, -1, -1, -1],
              cummin: [-3, -3, -3, -3]
            };
            const pair = op === "cummax" || op === "cummin";
            const values = pair ? result[0] : result;
            if (pair) {
              assertClose(await result[1].toArray(), op === "cummax" ? [0, 1, 1, 1] : [0, 0, 0, 0], 0);
              assert(result[1].dtype === "int32");
            }
            assertClose(await values.toArray(), expected[op], 0);
            assert(values.dtype === (op === "all" || op === "any" ? "bool" : op === "cumsum" ? "int32" : "int8"));
          });
        }
        await test("scan owners: boolean axes and empty identities", async () => {
          const x = new Tensor([0, 2.5, -2, 1, 0, 3]).reshape(2, 3);
          for (const op of ["all", "any"]) {
            assertClose(await x[op](0).toArray(), op === "all" ? [0, 0, 1] : [1, 1, 1], 0);
            assertClose(await x[op](-1, true).toArray(), op === "all" ? [0, 0] : [1, 1], 0);
            assertShape(x[op]({ axis: [0, 1], keepdim: true }).shape, [1, 1]);
            assertClose(await x[op]([]).toArray(), [0, 1, 1, 1, 0, 1], 0);
            for (const shape of [[], [0], [2, 0], [0, 2]]) {
              const t = Tensor.ones(shape, { dtype: "int8" });
              for (const axis of [0, -1]) {
                const out = t[op](axis);
                const expected = new Array(out.shape.reduce((a, b) => a * b, 1)).fill(shape.length && shape[axis < 0 ? shape.length - 1 : axis] === 0 ? Number(op === "all") : 1);
                assertClose(await out.toArray(), expected, 0);
              }
            }
          }
        });
        await testIf(pg.device !== "webgpu", "scan owners: NaN truthiness (outside WGSL finite math)", async () => {
          const x = new Tensor([0, NaN, -2, 1, 0, 3], { dtype: "float32" }).reshape(2, 3);
          assertClose(await x.any(0).toArray(), [1, 1, 1], 0);
          assertClose(await x.all([]).toArray(), [0, 1, 1, 1, 0, 1], 0);
        });
        await test("scan owners: empty scalar and invalid axes", async () => {
          for (const op of ["cumsum", "cumprod", "cummax", "cummin"]) {
            for (const shape of [[], [0], [2, 0], [0, 2]]) {
              const x = Tensor.ones(shape, { dtype: "int8" });
              for (const axis of [0, -1]) {
                const result = x[op](axis);
                const pair = Array.isArray(result);
                const values = pair ? result[0] : result;
                assertShape(values.shape, shape);
                assert(values.dtype === (op === "cumsum" ? "int32" : "int8"));
                assertClose(await values.toArray(), shape.length ? [] : [1], 0);
                if (pair) {
                  assertShape(result[1].shape, shape);
                  assertClose(await result[1].toArray(), shape.length ? [] : [0], 0);
                }
              }
              for (const axis of [Math.max(1, shape.length), -Math.max(1, shape.length) - 1]) {
                let threw = false;
                try {
                  x[op](axis);
                } catch {
                  threw = true;
                }
                assert(threw, `${op} admitted axis ${axis}`);
              }
            }
          }
        });
        await test("scan owners: uint8 promotion and boolean indices", async () => {
          const x = new Tensor([250, 10, 3, 2], { dtype: "uint8" });
          assert(x.cumsum().dtype === "uint32");
          assertClose(await x.cumsum().toArray(), [250, 260, 263, 265], 0);
          assert(x.cumprod(0).dtype === "uint8");
          assertClose(await x.cumprod(0).toArray(), [250, 196, 76, 152], 0);
          const y = new Tensor([true, false, true, true], { dtype: "bool" });
          const [max, maxidx] = y.cummax(), [min, minidx] = y.cummin();
          assertClose(await max.toArray(), [1, 1, 1, 1], 0);
          assertClose(await maxidx.toArray(), [0, 0, 0, 0], 0);
          assertClose(await min.toArray(), [1, 0, 0, 0], 0);
          assertClose(await minidx.toArray(), [0, 1, 1, 1], 0);
        });
        for (const dtype of ["uint64", "int64", "float16"]) {
          await testIf(pg.canRun({ dtype }), `scan owners: ${dtype} exact extrema`, async () => {
            const data = dtype === "uint64" ? [18446744073709551615n, 0n, 18446744073709551614n, 0n] : dtype === "int64" ? [-9223372036854775808n, 0n, -9223372036854775807n, 0n] : [3, 1, 2, 1];
            const x = new Tensor(data, { dtype });
            for (const op of ["cummax", "cummin"]) {
              const [values, indices] = x[op]();
              const expected = [], positions = [];
              let at = 0;
              for (let i = 0; i < data.length; i++) {
                if (op === "cummax" ? data[i] > data[at] : data[i] < data[at]) at = i;
                expected.push(data[at]);
                positions.push(at);
              }
              const actual = await values.toArray();
              assert(Array.from(actual).every((v, i) => v === expected[i]), `${dtype} ${op} lost exact values`);
              assertClose(await indices.toArray(), positions, 0);
            }
          });
        }
        await test("scan owners: split boundary values and gradient", async () => {
          for (const n of [512, 513, 1025]) {
            const data = new Float32Array(n * 2).fill(1);
            data[0] = 2;
            data[1] = 3;
            const x = new Tensor(data).reshape(n, 2);
            const sum = x.cumsum(0), prod = x.cumprod(0);
            const expected = Array.from({ length: n * 2 }, (_, i) => Math.floor(i / 2) + 2 + i % 2);
            assertClose(await sum.toArray(), expected, 0);
            assertClose(await prod.toArray(), Array.from({ length: n * 2 }, (_, i) => 2 + i % 2), 0);
            await x.cumsum(0).sum().backward();
            assertClose(await x.grad.toArray(), Array.from({ length: n * 2 }, (_, i) => n - Math.floor(i / 2)), 0);
          }
        });
        await test("scan owners: product gradients with zeros", async () => {
          for (const [data, expected] of [[[2, 3, 4], [16, 10, 6]], [[2, 0, 4], [1, 10, 0]], [[0, 3, 0], [4, 0, 0]]]) {
            const x = new Tensor(data, { dtype: "float32" });
            await x.cumprod(0).sum().backward();
            assertClose(await x.grad.toArray(), expected, 0);
          }
        });
        await test("numeric owners: min inverse matches pinned", async () => {
          const cases = [
            ["uint8", [0, 1, 255, 7, 3, 2]],
            ["int8", [-128, -1, 127, 7, 3, 2]],
            ["uint16", [0, 1, 65535, 7, 3, 2]],
            ["int16", [-32768, -1, 32767, 7, 3, 2]],
            ["uint32", [0, 1, 2 ** 32 - 1, 7, 3, 2]],
            ["int32", [-(2 ** 31), -1, 2 ** 31 - 1, 7, 3, 2]],
            ["bool", [false, true, true, true, true, true]],
            ["float32", [-3, -1, 127, 7, 3, 2]]
          ];
          for (const [dtype, values] of cases) {
            const x = new Tensor(values, { dtype }).reshape(2, 3);
            for (const [axis, keepdim] of [[null, false], [0, false], [-1, true], [[0, 1], true], [[], false]]) {
              const out = x.min({ axis, keepdim });
              const inverseOp = pg._core.ops[dtype === "float32" ? "MUL" : dtype === "bool" ? "CMPNE" : "XOR"];
              assert(out.uopPhysical.op === inverseOp, `${dtype} min inverse`);
              if (axis === null) {
                const reduced = out.uopPhysical.src[0];
                assert(reduced.op === pg._core.ops.REDUCE, `${dtype} min reduction`);
                assert(reduced.src[0].op === inverseOp, `${dtype} min input inverse`);
              }
              assert(out.dtype === dtype, `${dtype} min dtype`);
              const numbers = values.map(Number);
              const expected = axis === 0 ? numbers.slice(0, 3).map((v, i) => Math.min(v, numbers[i + 3])) : axis === -1 ? [Math.min(...numbers.slice(0, 3)), Math.min(...numbers.slice(3))] : Array.isArray(axis) && !axis.length ? numbers : [Math.min(...numbers)];
              assertClose(await out.toArray(), expected, 0);
            }
            assertClose(await x.min().toArray(), [Number(values[0])], 0);
            assertShape(x.min(-1, true).shape, [2, 1]);
            assertClose(await x.min(0).toArray(), values.slice(0, 3).map((v, i) => Math.min(Number(v), Number(values[i + 3]))), 0);
          }
        });
        await test("numeric owners: pow negative fraction constant matches buffer", async () => {
          for (const power of [0.2, 1.2, -0.2]) {
            for (const value of [-28, [-28]]) {
              const result = new Tensor(value, { dtype: "float32" }).pow(power);
              assert(result.uopPhysical.op === pg._core.ops.POW, "pow must start with raw POW");
              assertClose(await result.toArray(), [NaN]);
            }
          }
        });
        for (const dtype of ["int64", "uint64"]) {
          await testIf(pg.canRun({ dtype }), `numeric owners: min ${dtype} exact storage`, async () => {
            const values = dtype === "int64" ? [-(1n << 63n), -1n, (1n << 63n) - 1n] : [0n, (1n << 63n) + 1n, (1n << 64n) - 1n];
            const x = new Tensor(values, { dtype });
            const result = x.min();
            assert(result.dtype === dtype);
            const actual = await result.toArray();
            assert(actual.length === 1 && BigInt(actual[0]) === values[0]);
          });
        }
        await test("pow scalar promotion and validation match tinygrad", async () => {
          const t = new Tensor([2, 3], { dtype: "int32" });
          const exponent = new Tensor(2, { dtype: "weakfloat" });
          const promoted = t.pow(exponent);
          assert(promoted.dtype === "weakfloat", `expected weakfloat, got ${promoted.dtype}`);
          assert(promoted.uop.op === pg._core.ops.POW, "promoted pow should remain a raw POW");
          assertClose(await promoted.toArray(), [4, 9]);
          const reverse = t.pow(exponent, true);
          assert(reverse.dtype === "weakfloat", `expected reverse weakfloat, got ${reverse.dtype}`);
          assertClose(await reverse.toArray(), [4, 8]);
          let rejected = false;
          try {
            t.pow(-1);
          } catch (e) {
            rejected = e.message.includes("base needs to be float");
          }
          assert(rejected, "negative integer scalar exponent should reject an integer common dtype");
          const tensorExponent = t.pow(new Tensor([-1, -2], { dtype: "int32" }));
          assert(tensorExponent.dtype === "int32", `expected int32, got ${tensorExponent.dtype}`);
          assert(tensorExponent.uop.op === pg._core.ops.POW, "integer Tensor pow should remain raw POW");
          const strong = t.cast("float32").pow(new Tensor([-1, -2], { dtype: "float32" }));
          assertClose(await strong.toArray(), [0.5, 1 / 9]);
        });
        await test("reciprocal", async () => {
          const t = new Tensor([2, 4, 5]);
          const r = t.reciprocal();
          assertClose(await r.toArray(), [0.5, 0.25, 0.2]);
        });
        await test("exp2", async () => {
          const t = new Tensor([0, 1, 2, 3]);
          const r = t.exp2();
          assertClose(await r.toArray(), [1, 2, 4, 8]);
        });
        await test("log2", async () => {
          const t = new Tensor([1, 2, 4, 8]);
          const r = t.log2();
          assertClose(await r.toArray(), [0, 1, 2, 3]);
        });
        await test("trunc", async () => {
          const t = new Tensor([1.7, -2.3, 3.9]);
          const r = t.trunc();
          assertClose(await r.toArray(), [1, -2, 3]);
        });
        console.log("\n-- Aliases --");
        await test("swish is silu", async () => {
          const t = new Tensor([1, 2, -1]);
          assertClose(await t.swish().toArray(), await t.silu().toArray());
        });
        await test("view is reshape", async () => {
          const t = new Tensor([1, 2, 3, 4, 5, 6]);
          assertShape(t.view(2, 3).shape, [2, 3]);
          assertClose(await t.view(2, 3).toArray(), [1, 2, 3, 4, 5, 6]);
        });
        await test("matmul is dot", async () => {
          const a = new Tensor([[1, 2], [3, 4]]);
          const b = new Tensor([[5, 6], [7, 8]]);
          assertClose(await a.matmul(b).toArray(), await a.dot(b).toArray());
        });
        console.log("\n-- Composed ops --");
        await test("conv2d stride and padding match reference", async () => {
          const xData = Float32Array.from({ length: 1 * 2 * 4 * 5 }, (_, i) => i / 7);
          const wData = Float32Array.from({ length: 3 * 2 * 2 * 3 }, (_, i) => (i - 5) / 11);
          const bData = Float32Array.from([0.5, -1, 2]);
          const expected = new Float32Array(1 * 3 * 6 * 2);
          for (let oc = 0; oc < 3; oc++) {
            for (let oy = 0; oy < 6; oy++) {
              for (let ox = 0; ox < 2; ox++) {
                let acc = bData[oc];
                for (let ic = 0; ic < 2; ic++) {
                  for (let ky = 0; ky < 2; ky++) {
                    for (let kx = 0; kx < 3; kx++) {
                      const iy = oy + ky - 2;
                      const ix = ox * 2 + kx - 1;
                      if (iy >= 0 && iy < 4 && ix >= 0 && ix < 5) {
                        acc += xData[(ic * 4 + iy) * 5 + ix] * wData[((oc * 2 + ic) * 2 + ky) * 3 + kx];
                      }
                    }
                  }
                }
                expected[(oc * 6 + oy) * 2 + ox] = acc;
              }
            }
          }
          const conv = new Tensor(xData).reshape(1, 2, 4, 5).conv2d(new Tensor(wData).reshape(3, 2, 2, 3), new Tensor(bData), 1, [1, 2], 1, [1, 0, 2, 1]);
          assertShape(conv.shape, [1, 3, 6, 2]);
          assertClose(await conv.toArray(), expected, 1e-4);
        });
        await testIf(supportsF16, "conv2d dtype promotion and accumulation match pinned", async () => {
          const x = new Tensor(Float32Array.from({ length: 9 }, (_, i) => i / 8)).reshape(1, 1, 3, 3).cast("float16");
          const weightData = Float32Array.from([0.25, -0.5, 0.75, 0.125]);
          const wHalf = new Tensor(weightData).reshape(1, 1, 2, 2).cast("float16");
          const wFloat = new Tensor(weightData).reshape(1, 1, 2, 2);
          const bHalf = new Tensor(new Float32Array([0.0625])).cast("float16");
          const bFloat = new Tensor(new Float32Array([0.0625]));
          const mixed = x.conv2d(wFloat, bFloat);
          const halfDefault = x.conv2d(wHalf, bHalf);
          const halfExplicit = x.conv2d(wHalf, bHalf, { dtype: "float32" });
          const halfFloatBias = x.conv2d(wHalf, bFloat);
          assert(mixed.dtype === "float32", `expected float32, got ${mixed.dtype}`);
          assert(halfDefault.dtype === "float16", `expected float16, got ${halfDefault.dtype}`);
          assert(halfExplicit.dtype === "float32", `expected float32, got ${halfExplicit.dtype}`);
          assert(halfFloatBias.dtype === "float32", `expected float32, got ${halfFloatBias.dtype}`);
          const expected = [0.34375, 0.421875, 0.578125, 0.65625];
          assertClose(await mixed.toArray(), expected, 0);
          assertClose(await halfDefault.toArray(), expected, 0);
          assertClose(await halfExplicit.toArray(), expected, 0);
          assertClose(await halfFloatBias.toArray(), expected, 0);
        });
        await test("conv2d padded 3x3 4x4 devectorize regression", async () => {
          const padded = Tensor.arange(16, { dtype: "float32" }).reshape(1, 1, 4, 4).conv2d(Tensor.ones(1, 1, 3, 3), null, 1, 1, 1, 1);
          assertShape(padded.shape, [1, 1, 4, 4]);
          assertClose(await padded.toArray(), [10, 18, 24, 18, 27, 45, 54, 39, 51, 81, 90, 63, 42, 66, 72, 50]);
        });
        await test("max_pool2d padding matches reference", async () => {
          const pool = Tensor.arange(9, { dtype: "float32" }).reshape(1, 1, 3, 3).max_pool2d(2, 1, 1, 1);
          assertShape(pool.shape, [1, 1, 4, 4]);
          assertClose(await pool.toArray(), [0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 8, 6, 7, 8, 8]);
        });
        await test("batchnorm multi-axis matches reference", async () => {
          const bnX = Float32Array.from({ length: 2 * 3 * 4 * 5 }, (_, i) => i / 10);
          const mean = Float32Array.from({ length: 8 }, (_, i) => i / 20);
          const inv = Float32Array.from({ length: 8 }, () => 0.25);
          const weight = Float32Array.from({ length: 8 }, (_, i) => 0.5 + 0.7 * i / 7);
          const bias = Float32Array.from({ length: 8 }, (_, i) => -0.3 + 0.7 * i / 7);
          const bnExpected = new Float32Array(bnX.length);
          for (let n = 0; n < 2; n++) for (let c = 0; c < 3; c++) for (let h = 0; h < 4; h++) for (let w = 0; w < 5; w++) {
            const ki = n * 4 + h;
            const oi = ((n * 3 + c) * 4 + h) * 5 + w;
            bnExpected[oi] = (bnX[oi] - mean[ki]) * weight[ki] * inv[ki] + bias[ki];
          }
          const bn = new Tensor(bnX).reshape(2, 3, 4, 5).batchnorm(
            new Tensor(weight).reshape(2, 4),
            new Tensor(bias).reshape(2, 4),
            new Tensor(mean).reshape(2, 4),
            new Tensor(inv).reshape(2, 4),
            [0, 2]
          );
          assertClose(await bn.toArray(), bnExpected, 1e-5);
        });
        await test("nn empty input readback after normalization", async () => {
          const x = Tensor.empty(2, 4);
          await new pg.nn.LayerNorm(4).call(x).realize();
          assert((await x.toArray()).length === 8, "uninitialized input remains readable");
        });
        await test("nn multidimensional LayerNorm matches all trailing axes", async () => {
          const layer = new pg.nn.LayerNorm([2, 2], { elementwiseAffine: false });
          const x = new Tensor([1, 2, 3, 4]).reshape(1, 2, 2);
          assertClose(await layer.call(x).toArray(), [-1.34163547, -0.44721183, 0.44721183, 1.34163547], 1e-5);
          let rejected = false;
          try {
            layer.call(Tensor.ones(1, 3, 2));
          } catch (e) {
            rejected = /must match/.test(e.message);
          }
          assert(rejected, "LayerNorm must validate the entire normalized shape");
        });
        await test("nn Conv2d accepts one spatial dimension", async () => {
          const layer = new pg.nn.Conv2d(2, 3, [3], { bias: false });
          assertShape(layer.weight.shape, [3, 2, 3]);
          assertShape(layer.call(Tensor.ones(1, 2, 5)).shape, [1, 3, 3]);
        });
        await test("nn convolution factories InstanceNorm and LSTMCell", async () => {
          const conv = new pg.nn.Conv1d(1, 1, 2, { bias: false });
          conv.weight = Tensor.ones(1, 1, 2);
          const x = new Tensor([1, 2, 3], { dtype: "float32" }).reshape(1, 1, 3);
          assertClose(await conv.call(x).toArray(), [3, 5]);
          const transposed = new pg.nn.ConvTranspose1d(1, 1, 2, { stride: 2, bias: false });
          transposed.weight = Tensor.ones(1, 1, 2);
          assertClose(await transposed.call(x).toArray(), [1, 1, 2, 2, 3, 3]);
          const norm = new pg.nn.InstanceNorm(2);
          norm.weight = new Tensor([2, 3], { dtype: "float32" });
          norm.bias = new Tensor([4, 5], { dtype: "float32" });
          assertClose(await norm.call(new Tensor([1, 3, 10, 14], { dtype: "float32" }).reshape(1, 2, 2)).toArray(), [2, 6, 2, 8], 2e-5);
          const cell = new pg.nn.LSTMCell(2, 2, { bias: false });
          cell.weightIh = Tensor.zeros(8, 2);
          cell.weightHh = Tensor.zeros(8, 2);
          const [zeroH, zeroC] = cell.call(Tensor.ones(1, 2));
          assertClose(await zeroH.toArray(), [0, 0]);
          assertClose(await zeroC.toArray(), [0, 0]);
          const [h, c] = cell.call(Tensor.ones(1, 2), [Tensor.zeros(1, 2), Tensor.ones(1, 2)]);
          assertClose(await c.toArray(), [0.5, 0.5]);
          assertClose(await h.toArray(), [0.23105858, 0.23105858], 1e-6);
          h.sum().backward();
          assert(cell.weightIh.grad !== null);
          assert((await cell.weightIh.grad.toArray()).every(Number.isFinite));
        });
        await test("nn RMSNorm and Embedding match Python module contracts", async () => {
          const norm = new pg.nn.RMSNorm(2, { elementwiseAffine: false });
          const x = new Tensor([1, 2, 3, 4], { dtype: "float32" }).reshape(2, 2);
          assertClose(await norm.call(x).toArray(), [0.6324554, 1.2649108, 0.8485281, 1.1313708], 1e-5);
          const embedding = new pg.nn.Embedding(3, 2);
          embedding.weight = new Tensor([1, 2, 3, 4, 5, 6], { dtype: "float32" }).reshape(3, 2);
          const out = embedding.call(new Tensor([2, 0], { dtype: "int32" }));
          assertClose(await out.toArray(), [5, 6, 1, 2]);
          let rejected = false;
          try {
            embedding.call(new Tensor([1.5]));
          } catch (e) {
            rejected = /integer/.test(e.message);
          }
          assert(rejected, "embedding must reject floating indices");
          out.sum().backward();
          assert(embedding.weight.grad !== null, "embedding weight gradient missing");
        });
        await test("nn Dropout delegates Tensor training and endpoint behavior", async () => {
          const previous = Tensor.training;
          const x = new Tensor([1, 2, 3]);
          try {
            Tensor.training = false;
            assert(new pg.nn.Dropout(0.5).call(x) === x, "eval dropout must be identity");
            Tensor.training = true;
            assertClose(await new pg.nn.Dropout(1).call(x).toArray(), [0, 0, 0]);
            let rejected = false;
            try {
              new pg.nn.Dropout(1.1).call(x);
            } catch (e) {
              rejected = /out of range/.test(e.message);
            }
            assert(rejected, "dropout must validate probability");
          } finally {
            Tensor.training = previous;
          }
        });
        await test("nn BatchNorm training state and eval share one module", async () => {
          const previous = Tensor.training;
          try {
            assert(pg.nn.BatchNorm === pg.nn.BatchNorm2d && pg.nn.BatchNorm === pg.nn.BatchNorm3d, "BatchNorm aliases");
            const layer = new pg.nn.BatchNorm(2, { affine: false, momentum: 0.5 });
            const x = new Tensor([1, 2, 3, 4]).reshape(2, 2);
            Tensor.training = true;
            assertClose(await layer.call(x).toArray(), [-0.999995, -0.999995, 0.999995, 0.999995], 1e-5);
            assertClose(await layer.runningMean.toArray(), [1, 1.5]);
            assertClose(await layer.runningVar.toArray(), [1.5, 1.5]);
            assertClose(Array.from(await layer.numBatchesTracked.toArray(), Number), [1]);
            Tensor.training = false;
            assertClose(await layer.call(x).toArray(), [0, 0.4082469, 1.6329877, 2.0412346], 1e-5);
          } finally {
            Tensor.training = previous;
          }
        });
        await test("nn LSTM Model recurrent state and training checkpoint", isolatedRuntime(async (rt) => {
          const T = rt.Tensor;
          const x = T.empty(1, 2), h = T.empty(1, 2), c = T.empty(1, 2);
          const cell = new rt.nn.LSTMCell(2, 2, { bias: false });
          cell.weightIh = T.full([8, 2], 0.25);
          cell.weightHh = T.full([8, 2], 0.125);
          const [hidden, state] = cell.call(x, [h, c]);
          const authored = await rt.Model.fromTensors({
            inputs: { x, h, c },
            params: { wi: cell.weightIh, wh: cell.weightHh },
            outputs: { hidden, cell: state },
            losses: { loss: hidden.square().mean() }
          });
          const bundle = await authored.saveBundleAsync({ includeOptimizer: false });
          await authored.dispose();
          const model = await rt.Model.fromBundle(bundle);
          let restored = null;
          try {
            const inputs = { x: new Float32Array([1, 2]), h: new Float32Array(2), c: new Float32Array(2) };
            const first = await model.forward(inputs);
            const second = await model.forward({ x: inputs.x, h: first.hidden, c: first.cell });
            assert(second.cell.every((v, i) => v > first.cell[i]), "explicit recurrent state must advance");
            model.setOptimizer(rt.OPTIM_ADAM, 0.01);
            await model.trainStepAsync(inputs);
            restored = await rt.Model.fromBundle(await model.saveBundleAsync());
            restored.setOptimizer(rt.OPTIM_ADAM, 0.01);
            assertClose([await model.trainStepAsync(inputs)], [await restored.trainStepAsync(inputs)], 1e-5);
            assertClose(await model.readBufferAsync("wi"), await restored.readBufferAsync("wi"), 1e-5);
            assert((await model.readBufferAsync("wi")).some((v) => Math.abs(v - 0.25) > 1e-5), "model weights must update");
          } finally {
            if (restored) await restored.dispose();
            await model.dispose();
          }
        }));
        await test("nn state discovery includes underscore and root tensor paths", async () => {
          const x = new Tensor([1]);
          assert(pg.nn.getStateDict({ _weight: x })._weight === x, "underscore is a valid state name");
          assert(pg.nn.getStateDict(x)[""] === x, "root tensor path is empty");
          assert(pg.nn.getStateDict({ weight: x }, "net.")["net.weight"] === x, "state prefix lost");
          const shared = { weight: x };
          const obj = { a: shared, b: shared };
          obj.self = obj;
          assert(Object.keys(pg.nn.getStateDict(obj)).length === 2, "aliases must survive cycle protection");
        });
        await test("nn state loading preserves target handles and consumes named sources", async () => {
          const model = { weight: new Tensor([1, 2], { dtype: "float32" }), scalar: new Tensor(0) };
          const target = model.weight;
          const state = { weight: new Tensor([3, 4], { dtype: "float32" }), scalar: new Tensor([5], { dtype: "float32" }) };
          const loaded = await pg.nn.loadStateDictAsync(model, state, { consume: true });
          assert(loaded.length === 2 && model.weight === target, "load must preserve target Tensor handles");
          assert(Object.keys(state).length === 0, "consumed keys remain");
          assertClose(await target.toArray(), [3, 4]);
          assertClose(await model.scalar.toArray(), [5]);
          let rejected = false;
          try {
            pg.nn.loadStateDict(model, { weight: new Tensor([1]) }, { realize: false });
          } catch (e) {
            rejected = /Shape mismatch/.test(e.message);
          }
          assert(rejected, "state shape mismatch must reject");
          assert(pg.nn.loadStateDict(model, {}, { strict: false }).length === 0, "nonstrict missing keys must skip");
        });
        await test("nn LARS LAMB and Muon use shared optimizer graphs", async () => {
          const previous = Tensor.training;
          try {
            Tensor.training = true;
            for (const [name, opts, expected] of [
              ["LAMB", { weightDecay: 0.01 }, [0.438561946, 2.54476404, 2.4274404, 4.53364801]],
              ["LARS", { momentum: 0.9 }, [0.999709964, 2.00057888, 2.99913001, 4.00115776]],
              ["Muon", { nsSteps: 2 }, [1.03743243, 2.11009645, 2.77558279, 4.05183554]]
            ]) {
              const p = new Tensor([1, 2, 3, 4], { dtype: "float32" }).reshape(2, 2);
              const opt = new pg.nn.optim[name]([p], { lr: 0.1, ...opts });
              for (let i = 0; i < 2; i++) {
                p._grad = new Tensor([0.1, -0.2, 0.3, -0.4]).reshape(2, 2);
                await opt.stepAsync();
              }
              assertClose(await p.toArray(), expected, 2e-5);
            }
          } finally {
            Tensor.training = previous;
          }
        });
        await test("nn optimizer rejects eval and preserves supplied learning rate Tensor", async () => {
          const previous = Tensor.training;
          const p = new Tensor([1], { dtype: "float32" });
          p._grad = new Tensor([2], { dtype: "float32" });
          const lr = new Tensor([0.1]);
          try {
            const opt = new pg.nn.optim.SGD([p], { lr });
            assert(opt.lr === lr, "learning rate Tensor must not be coerced to a number");
            Tensor.training = false;
            let rejected = false;
            try {
              opt.scheduleStep();
            } catch (e) {
              rejected = /TRAINING/.test(e.message);
            }
            assert(rejected, "optimizer must reject eval before publishing effects");
            Tensor.training = true;
            const scheduled = opt.scheduleStep();
            lr.assign([0.2]);
            await lr.realizeAsync();
            await scheduled[0].realizeAsync(...scheduled.slice(1));
            assertClose(await p.toArray(), [0.6]);
          } finally {
            Tensor.training = previous;
          }
        });
        await test("nn Conv2d backward populates parameters", async () => {
          const mod = new pg.nn.Conv2d(3, 2, 3, { padding: 1 });
          const loss = mod.call(Tensor.randn(1, 3, 4, 4)).relu().mean();
          loss.backward();
          assert(mod.weight.grad !== null, "Conv2d weight grad missing");
          assert(mod.bias.grad !== null, "Conv2d bias grad missing");
        });
        await test("nn GroupNorm matches pinned layernorm composition", async () => {
          const mod = new pg.nn.GroupNorm(2, 4);
          const x = Tensor.arange(32, { dtype: "float32" }).reshape(1, 4, 2, 4).div(11);
          const out = mod.call(x);
          assertShape(out.shape, [1, 4, 2, 4]);
          const values = await out.toArray();
          for (let group = 0; group < 2; group++) {
            const start = group * 16;
            const mean = values.slice(start, start + 16).reduce((a, b) => a + b, 0) / 16;
            assert(Math.abs(mean) < 1e-5, `GroupNorm group ${group} mean ${mean}`);
          }
          out.sum().backward();
          assert(mod.weight.grad !== null, "GroupNorm weight grad missing");
          assert(mod.bias.grad !== null, "GroupNorm bias grad missing");
        });
        await test("nn LayerNorm2d matches pinned NHWC composition", async () => {
          const mod = new pg.nn.LayerNorm2d(3);
          const input = Array.from({ length: 24 }, (_, i) => (i - 7) / 5);
          const out = mod.call(new Tensor(input).reshape(2, 3, 2, 2));
          assertShape(out.shape, [2, 3, 2, 2]);
          assert(out.uop.op === pg._core.ops.PERMUTE, "LayerNorm2d root must be PERMUTE");
          assert(countGraphOp(out.uop, pg._core.ops.PERMUTE) === 4, "expected four PERMUTEs");
          assert(countGraphOp(out.uop, pg._core.ops.REDUCE) === 2, "expected two REDUCE nodes");
          const values = await out.toArray();
          for (let n = 0; n < 2; n++) for (let h = 0; h < 2; h++) for (let w = 0; w < 2; w++) {
            const lanes = Array.from({ length: 3 }, (_, c) => values[((n * 3 + c) * 2 + h) * 2 + w]);
            const mean = lanes.reduce((a, b) => a + b, 0) / lanes.length;
            const variance = lanes.reduce((a, b) => a + b * b, 0) / lanes.length;
            assert(Math.abs(mean) < 1e-5, `LayerNorm2d mean ${mean}`);
            assert(Math.abs(variance - 1) < 1e-4, `LayerNorm2d variance ${variance}`);
          }
          out.sum().backward();
          assert(mod.weight.grad !== null, "LayerNorm2d weight grad missing");
          assert(mod.bias.grad !== null, "LayerNorm2d bias grad missing");
        });
        await test("layernorm", async () => {
          const t = new Tensor([[1, 2, 3], [4, 5, 6]]);
          const r = await t.layernorm();
          const arr = await r.toArray();
          const row0 = arr.slice(0, 3);
          const mean0 = row0.reduce((a, b) => a + b) / 3;
          assert(Math.abs(mean0) < 1e-4, `Row 0 mean should be ~0, got ${mean0}`);
        });
        const pointwiseReferences = {
          log10: Math.log10,
          atanh: Math.atanh,
          asinh: Math.asinh,
          acosh: Math.acosh,
          asin: Math.asin,
          acos: Math.acos,
          atan: Math.atan,
          celu: (x) => x,
          selu: (x) => 1.0507 * x,
          logsigmoid: (x) => -Math.log1p(Math.exp(-x)),
          sinh: Math.sinh,
          cosh: Math.cosh,
          erf: null,
          softsign: (x) => x / (1 + Math.abs(x))
        };
        for (const [op, fn] of Object.entries(pointwiseReferences)) {
          await test(`pointwise owners: ${op} values and gradient`, async () => {
            const data = op === "acosh" ? [1.2, 1.5, 2] : [0.15, 0.4, 0.7];
            const x = new Tensor(data);
            const out = x[op]();
            assert(out.dtype === "float32");
            assertClose(await out.toArray(), fn ? data.map(fn) : [0.1679959, 0.42839235, 0.67780113], 5e-6);
            await out.sum().backward();
            const expected = data.map((v) => {
              if (op === "erf") return 2 / Math.sqrt(Math.PI) * Math.exp(-v * v);
              const h = 1e-5;
              return (fn(v + h) - fn(v - h)) / (2 * h);
            });
            assertClose(await x.grad.toArray(), expected, 4e-5);
          });
        }
        await test("pointwise owners: parameterized activations and finite comparisons", async () => {
          const x = new Tensor([-2, -0.5, 0, 2]);
          assertClose(await x.celu(2).toArray(), [-1.2642411, -0.4423984, 0, 2], 3e-6);
          assertClose(await x.selu(2, 3).toArray(), [-5.1879883, -2.360816, 0, 6], 3e-6);
          assertClose(await x.isfinite().toArray(), [1, 1, 1, 1]);
          assertClose(await x.isclose(0, { atol: 0.5, rtol: 0 }).toArray(), [0, 1, 1, 0]);
        });
        await testIf(pg.device !== "webgpu", "pointwise owners: IEEE NaN infinity and signed zero", async () => {
          const a = new Tensor([1, 2, Infinity, -Infinity, NaN]);
          const b = new Tensor([1.000005, 2.1, Infinity, Infinity, NaN]);
          assertClose(await a.isfinite().toArray(), [1, 1, 0, 0, 0]);
          for (const equalNan of [false, true]) {
            assertClose(await a.isclose(b, { equalNan }).toArray(), [1, 0, 1, 0, Number(equalNan)]);
          }
          assertClose(await new Tensor([2, -3]).copysign(new Tensor(new Float32Array([-0, 0]))).toArray(), [-2, 3]);
        });
        await test("pointwise owners: copysign lerp and uint8 weight provenance", async () => {
          assertClose(await new Tensor([2, -3]).copysign(new Tensor([-1, 1])).toArray(), [-2, 3]);
          assertClose(await new Tensor([2, -3]).lerp(new Tensor([0, 1]), 0.25).toArray(), [1.5, -2]);
          const a = new Tensor([250, 10], { dtype: "uint8" }), b = new Tensor([10, 250], { dtype: "uint8" });
          assertClose(await a.lerp(b, new Tensor([0.5, 0.5])).toArray(), [2, 2]);
          assertClose(await a.lerp(b, 0.5).toArray(), pg.core === "wasm" || ["interp", "webgpu", "x86"].includes(pg.device) ? [258, 130] : [130, 130]);
        });
        await test("pointwise owners: narrow integer casts survive widening", async () => {
          const input = new Tensor(new Int32Array([128, -240, 65535, -65537]));
          for (const [dtype, expected] of [
            ["int8", [-128, 16, -1, -1]],
            ["uint8", [128, 16, 255, 255]],
            ["int16", [128, -240, -1, -1]],
            ["uint16", [128, 65296, 65535, 65535]]
          ]) {
            assertClose(await input.cast(dtype).cast("int32").toArray(), expected);
            if (["interp", "wasm", "webgpu"].includes(pg.device)) {
              const floats = new Tensor(new Float32Array([128.75, -240.5, 65535, -65537]));
              assertClose(await floats.cast(dtype).cast("int32").toArray(), expected);
            }
          }
          if (["interp", "wasm", "webgpu"].includes(pg.device)) {
            for (const [dtype, input2, expected] of [
              ["int8", 127, -128],
              ["uint8", 255, 0],
              ["int16", 32767, -32768],
              ["uint16", 65535, 0]
            ]) assertClose(await new Tensor([input2], { dtype }).add(1).cast("int32").toArray(), [expected]);
          }
        });
        for (const reduction of ["none", "sum", "mean"]) {
          await test(`pointwise owners: weighted losses ${reduction}`, async () => {
            const x = new Tensor([[-2, 0, 2], [1, -1, 0.5]]);
            const y = new Tensor([[0, 1, 1], [1, 0, 1]]), w = new Tensor([2, 3, 4]);
            const loss = x.binaryCrossEntropyLogits(y, { reduction, posWeight: w });
            const values = [
              Math.log1p(Math.exp(-2)),
              3 * Math.log(2),
              4 * Math.log1p(Math.exp(-2)),
              2 * Math.log1p(Math.exp(-1)),
              Math.log1p(Math.exp(-1)),
              4 * Math.log1p(Math.exp(-0.5))
            ];
            const sum = values.reduce((a, b) => a + b, 0);
            assertClose(await loss.toArray(), reduction === "none" ? values : [sum / (reduction === "mean" ? 6 : 1)], 4e-6);
            await loss.sum().backward();
            assertClose(await x.grad.toArray(), [
              1 / (1 + Math.exp(2)),
              -1.5,
              -4 / (1 + Math.exp(2)),
              -2 / (1 + Math.exp(1)),
              1 / (1 + Math.exp(1)),
              -4 / (1 + Math.exp(0.5))
            ].map((v) => v / (reduction === "mean" ? 6 : 1)), 4e-6);
            const scores = new Tensor([[-2, 0, 2], [1, -1, 0.5]]);
            const nll = scores.nllLoss(new Tensor([2, 1], { dtype: "int32" }), {
              weight: new Tensor([1, 2, 3]),
              ignoreIndex: 1,
              reduction
            });
            assertClose(await nll.toArray(), reduction === "none" ? [-6, 0] : [reduction === "mean" ? -2 : -6]);
            await nll.sum().backward();
            assertClose(await scores.grad.toArray(), [0, 0, reduction === "mean" ? -1 : -3, 0, 0, 0]);
          });
        }
        await test("pointwise owners: loss admission and spatial targets", async () => {
          const x = new Tensor([[[1, 2], [3, 4], [5, 6]]]);
          const y = new Tensor([[0, 2]], { dtype: "int32" });
          assertClose(await x.nllLoss(y, { reduction: "none" }).toArray(), [-1, -6]);
          assertClose(await x.nllLoss(y).toArray(), [-3.5]);
          for (const method of ["nllLoss", "binaryCrossEntropyLogits"]) {
            let rejected = false;
            try {
              x[method](y, { reduction: "typo" });
            } catch (e) {
              rejected = /reduction/.test(e.message);
            }
            assert(rejected, "invalid reduction must fail before graph publication");
          }
        });
        await test("indexed owners: paired and separated advanced reads", async () => {
          const x = Tensor.arange(12).reshape(3, 4);
          assertClose(await x.getitem(new Tensor([0, 2]), new Tensor([1, 3])).toArray(), [1, 11]);
          const out = Tensor.arange(24).reshape(2, 3, 4).getitem(
            new Tensor([[0], [1]]),
            { step: 1 },
            new Tensor([[0, 2]])
          );
          assertShape(out.shape, [2, 2, 3]);
          assertClose(await out.toArray(), [0, 4, 8, 2, 6, 10, 12, 16, 20, 14, 18, 22]);
        });
        await test("indexed owners: scalar Tensor index", async () => {
          assertClose(await Tensor.arange(6).reshape(2, 3).getitem(new Tensor(1)).toArray(), [3, 4, 5]);
        });
        await test("indexed inplace: unrealized assignment and realized forward", async () => {
          const expected = {
            add: [11, 22, 3, 4],
            sub: [-9, -18, 3, 4],
            mul: [10, 40, 3, 4],
            div: [0.1, 0.1, 3, 4]
          };
          for (const logical of ["always", "until_realize", "never"]) {
            for (const realized of [false, true]) for (const op of ["add", "sub", "mul", "div"]) {
              const z = new Tensor([1, 2, 3, 4], { dtype: "float32", logical });
              const x = new Tensor([10, 20], { dtype: "float32", logical });
              if (realized) await z.realize();
              const view = z.getitem({ start: 0, stop: 2 });
              const computed = view[op](x);
              view.assign(computed);
              await computed.dispose();
              z.setitem({ start: 0, stop: 2 }, view);
              assertClose(await z.toArray(), expected[op]);
              await view.dispose();
              await x.dispose();
              await z.dispose();
            }
          }
        });
        await test("chained assignment: Tinygrad read-order semantics", async () => {
          const cases = [
            ["wvz", { w: [16], v: [16, 22], z: [16, 22, 3, 4] }],
            ["wzv", { w: [16], z: [16, 22, 3, 4], v: [16, 22] }],
            ["vwz", { v: [11, 22], w: [16], z: [16, 22, 3, 4] }],
            ["vzw", { v: [11, 22], z: [11, 22, 3, 4], w: [16] }],
            ["zwv", { z: [1, 2, 3, 4], w: [16], v: [16, 22] }],
            ["zvw", { z: [1, 2, 3, 4], v: [11, 22], w: [16] }]
          ];
          for (const logical of ["always", "until_realize", "never"]) {
            for (const dtype of ["float32", "int32"]) for (const [order, expected] of cases) {
              const z = new Tensor([1, 2, 3, 4], { dtype, logical });
              const x = new Tensor([10, 20], { dtype, logical });
              const v = z.getitem({ start: 0, stop: 2 }), vx = v.add(x);
              v.assign(vx);
              await vx.dispose();
              const w = v.getitem({ start: 0, stop: 1 }), w5 = w.add(5);
              w.assign(w5);
              await w5.dispose();
              const tensors = { z, v, w };
              for (const name of order) assertClose(Array.from(await tensors[name].toArray()), expected[name]);
              await w.dispose();
              await v.dispose();
              await x.dispose();
              await z.dispose();
            }
          }
        });
        await test("indexed owners: detached write and identity read", async () => {
          const x = Tensor.zeros(4);
          await x.realize();
          x.detach().setitem(1, 5);
          assertClose(await x.toArray(), [0, 5, 0, 0]);
          assert(x.getitem() === x && x.getitem({ step: 1 }) === x, "no-op reads retain object identity");
        });
        await test("indexed owners: weak RHS admission", async () => {
          const x = Tensor.zeros(4);
          x.setitem(1, new Tensor(5));
          assertClose(await x.toArray(), [0, 5, 0, 0]);
          const y = Tensor.zeros(4, { dtype: "int32" });
          let rejected = false;
          try {
            y.setitem(1, new Tensor(1.5));
          } catch (e) {
            rejected = /dtype mismatch/.test(e.message);
          }
          assert(rejected, "weak float must not silently truncate to integer storage");
        });
        await test("indexed owners: int64 dimensions retain high word", async () => {
          for (const dim of [2147483649, 4294967297, 6442450945]) {
            assertShape(Tensor.empty(0, dim).shape, [0, dim]);
          }
        });
        await test("indexed owners: int64 positive and negative strides", async () => {
          const x = Tensor.arange(3);
          assertClose(await x.getitem({ step: 4294967297 }).toArray(), [0]);
          assertClose(await x.getitem({ step: -4294967297 }).toArray(), [2]);
        });
        for (const realized of [false, true]) {
          await test(`indexed owners: strided write realized=${realized}`, async () => {
            const x = Tensor.arange(8).cast("float32");
            if (realized) await x.realize();
            x.setitem({ start: 1, stop: 7, step: 2 }, new Tensor([10, 20, 30], { dtype: "float32" }));
            assertClose(await x.toArray(), [0, 10, 2, 20, 4, 30, 6, 7]);
          });
        }
        await test("indexed owners: last duplicate write and live use rejection", async () => {
          const x = new Tensor([0, 1, 2, 3]);
          x.setitem(new Tensor([1, 1, 3]), new Tensor([7, 8, 9]));
          assertClose(await x.toArray(), [0, 8, 2, 9]);
          const y = new Tensor([1, 2]), other = y.add(1);
          let rejected = false;
          try {
            y.setitem(0, 4);
          } catch (e) {
            rejected = /other uses/.test(e.message);
          }
          assert(rejected, "setitem must reject a live dependent computation");
          assertClose(await other.toArray(), [2, 3]);
        });
        await test("indexed owners: gradient assignment and reset", async () => {
          const x = new Tensor([1, 2], { dtype: "float32" }), g = new Tensor([10, 20], { dtype: "float32" });
          x.grad = g;
          assert(x.grad === g, "grad assignment must retain the same Tensor object");
          await x.mul(x).sum().backward();
          assertClose(await x.grad.toArray(), [12, 24]);
          x.grad = null;
          await x.mul(3).sum().backward();
          assertClose(await x.grad.toArray(), [3, 3]);
        });
        for (const reduction of ["none", "sum", "mean"]) {
          await test(`indexed owners: ordinary BCE ${reduction}`, async () => {
            const x = new Tensor([0.2, 0.7]), y = new Tensor([0, 1]);
            const out = x.binaryCrossEntropy(y, reduction);
            const terms = [-Math.log(0.8), -Math.log(0.7)];
            assertClose(await out.toArray(), reduction === "none" ? terms : [terms.reduce((a, b) => a + b) / (reduction === "mean" ? 2 : 1)]);
            await out.sum().backward();
            assertClose(await x.grad.toArray(), [1 / 0.8, -1 / 0.7].map((v) => v / (reduction === "mean" ? 2 : 1)));
          });
        }
        await test("binaryCrossEntropy", async () => {
          const pred = new Tensor([0.9, 0.1, 0.8]);
          const target = new Tensor([1, 0, 1]);
          const loss = await pred.binaryCrossEntropy(target).item();
          const expected = -(Math.log(0.9) + Math.log(0.9) + Math.log(0.8)) / 3;
          assert(Math.abs(loss - expected) < 1e-3, `Expected ~${expected}, got ${loss}`);
        });
        await test("cat 1d", async () => {
          const a = new Tensor([1, 2, 3]);
          const b = new Tensor([4, 5, 6]);
          const r = Tensor.cat(a, b);
          assertShape(r.shape, [6]);
          assertClose(await r.toArray(), [1, 2, 3, 4, 5, 6]);
        });
        await test("cat 2d axis0", async () => {
          const a = new Tensor([[1, 2], [3, 4]]);
          const b = new Tensor([[5, 6]]);
          const r = Tensor.cat(a, b, { dim: 0 });
          assertShape(r.shape, [3, 2]);
          const arr = await r.toArray();
          console.log("DEBUG cat axis0", Array.from(arr));
          assertClose(arr, [1, 2, 3, 4, 5, 6]);
        });
        await test("instance cat matches tinygrad binding", async () => {
          const a = new Tensor([[1, 2], [3, 4]]);
          const b = new Tensor([[5, 6]]);
          const c = new Tensor([[7, 8]]);
          const r = a.cat(b, c, { dim: 0 });
          assertShape(r.shape, [4, 2]);
          assertClose(await r.toArray(), [1, 2, 3, 4, 5, 6, 7, 8]);
        });
        await test("stack", async () => {
          const a = new Tensor([1, 2, 3]);
          const b = new Tensor([4, 5, 6]);
          const r = Tensor.stack(a, b);
          assertShape(r.shape, [2, 3]);
          const arr = await r.toArray();
          console.log("DEBUG stack", Array.from(arr));
          assertClose(arr, [1, 2, 3, 4, 5, 6]);
        });
        await test("instance stack matches tinygrad binding", async () => {
          const a = new Tensor([1, 2]);
          const b = new Tensor([3, 4]);
          const r = a.stack(b, { dim: 0 });
          assertShape(r.shape, [2, 2]);
          assertClose(await r.toArray(), [1, 2, 3, 4]);
        });
        await test("repeat", async () => {
          const t = new Tensor([1, 2, 3]);
          const r = t.repeat(3);
          assertShape(r.shape, [9]);
          assertClose(await r.toArray(), [1, 2, 3, 1, 2, 3, 1, 2, 3]);
        });
        await test("repeat readback preserves int32 dtype", async () => {
          const t = new Tensor([1, 2, 3], { dtype: "int32" });
          const r = t.repeat(3);
          assert(r.dtype === "int32", `expected int32, got ${r.dtype}`);
          const arr = await r.toArray();
          assert(arr.constructor.name === "Int32Array", `expected Int32Array, got ${arr.constructor.name}`);
          assertClose(arr, [1, 2, 3, 1, 2, 3, 1, 2, 3]);
        });
        await test("repeat 2d", async () => {
          const t = new Tensor([[1, 2], [3, 4]]);
          const r = t.repeat(2, 3);
          assertShape(r.shape, [4, 6]);
          assertClose(await r.toArray(), [
            1,
            2,
            1,
            2,
            1,
            2,
            3,
            4,
            3,
            4,
            3,
            4,
            1,
            2,
            1,
            2,
            1,
            2,
            3,
            4,
            3,
            4,
            3,
            4
          ]);
        });
        await test("spatial owners average pooling padding and gradient", async () => {
          for (const ceilMode of [false, true]) for (const countIncludePad of [false, true]) {
            const x = new Tensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], { dtype: "float32" });
            const y = x.avgPool2d([2, 2], { stride: 2, padding: [1, 0, 0, 1], ceilMode, countIncludePad });
            y.sum().backward();
            assertClose(await y.toArray(), countIncludePad ? [1.25, 4, 1.75, 4.25] : [2.5, 4, 7, 8.5]);
            assertClose(await x.grad.toArray(), countIncludePad ? Array(9).fill(0.25) : [0.5, 0.25, 0.25, 0.5, 0.25, 0.25, 1, 0.5, 0.5]);
          }
        });
        await test("spatial owners max pooling ceil and indices", async () => {
          const x = new Tensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], { dtype: "float32" });
          const [values, indices] = x.maxPool2d([2, 2], { ceilMode: true, returnIndices: true });
          assertClose(await values.toArray(), [5, 6, 8, 9]);
          assertClose(await indices.toArray(), [4, 5, 7, 8]);
          assertClose(await x.avgPool2d([2, 2], { ceilMode: true }).toArray(), [3, 4.5, 7.5, 9]);
        });
        await test("spatial owners interpolation values and gradients", async () => {
          for (const [mode, alignCorners, expected, grad] of [
            ["linear", false, [1, 1.8, 3, 5.4, 7], [1.6, 1.8, 1.6]],
            ["linear", true, [1, 2, 3, 5, 7], [1.5, 2, 1.5]],
            ["nearest", false, [1, 1, 3, 3, 7], [2, 2, 1]],
            ["nearest-exact", false, [1, 1, 3, 7, 7], [2, 1, 2]]
          ]) {
            const x = new Tensor([[[1, 3, 7]]], { dtype: "float32" });
            const y = x.interpolate([5], { mode, alignCorners });
            y.sum().backward();
            assertClose(await y.toArray(), expected);
            assertClose(await x.grad.toArray(), grad);
          }
        });
        await test("spatial owners transpose convolution groups and gradient", async () => {
          const x = new Tensor([[[1, 2, 3], [4, 5, 6]]], { dtype: "float32" });
          const w = new Tensor([[[1, 2]], [[3, 4]]], { dtype: "float32" });
          const y = x.convTranspose2d(w, null, { groups: 2, stride: 2, dilation: 2, padding: 1, outputPadding: 1 });
          y.sum().backward();
          assertClose(await y.toArray(), [0, 4, 0, 7, 0, 6, 0, 31, 0, 38, 0, 24]);
          assertClose(await x.grad.toArray(), [2, 3, 3, 4, 7, 7]);
        });
        await test("spatial owners invalid storage disjoint writes", async () => {
          const x = Tensor.invalids([6], { dtype: "int32" });
          await x.realize();
          x.setitem([{ start: 1, stop: 3 }], 7);
          x.setitem([{ start: 4, stop: 5 }], 9);
          assertClose(await x.getitem({ start: 1, stop: 3 }).toArray(), [7, 7]);
          assertClose(await x.getitem({ start: 4, stop: 5 }).toArray(), [9]);
        });
        await test("spatial owners interpolation empty and singleton", async () => {
          const x = new Tensor([1, 3, 7], { dtype: "float32" });
          const singleton = await x.interpolate([1], { alignCorners: true }).toArray();
          assert(Number.isNaN(singleton[0]));
          for (const alignCorners of [false, true]) {
            const out = x.interpolate([0], { alignCorners });
            assertShape(out.shape, [0]);
            assert((await out.toArray()).length === 0);
          }
        });
        await test("spatial owners unpool negative infinity and output size", async () => {
          const x = new Tensor([[[[-Infinity, 2], [3, 4]]]], { dtype: "float32" });
          const idx = new Tensor([[[[0, 1], [2, 3]]]], { dtype: "int32" });
          const got = await x.maxUnpool2d(idx, [1, 1]).toArray();
          assert(got[0] === -Infinity);
          assertClose(got.slice(1), [2, 3, 4]);
          const y = new Tensor([[[[1, 2], [3, 4]]]], { dtype: "float32" });
          const [value, index] = y.maxPool2d([2, 2], { returnIndices: true });
          assertClose(await value.maxUnpool2d(index).toArray(), [0, 0, 0, 4]);
          assertShape(value.maxUnpool2d(index, [2, 2], { outputSize: [1, 1, 3, 3] }).shape, [1, 1, 3, 3]);
        });
        await test("spatial owners reject malformed dimension arrays", async () => {
          const x = Tensor.ones([1, 1, 3, 3]);
          for (const make of [
            () => x.avgPool2d([2, 2], { stride: [1] }),
            () => x.maxPool2d([2, 2], { dilation: [1] })
          ]) {
            let error;
            try {
              make();
            } catch (e) {
              error = e;
            }
            assert(error && /stride\/dilation mismatch/.test(error.message));
          }
        });
        await test("spatial owners transpose dimension array admission", async () => {
          const x = Tensor.ones([1, 1, 3, 3]), w = Tensor.ones([1, 1, 2, 2]);
          for (const [opts, pattern] of [
            [{ dilation: [1] }, /stride\/dilation mismatch/],
            [{ stride: [2] }, /stride/],
            [{ outputPadding: [] }, /output_padding/]
          ]) {
            let error;
            try {
              x.convTranspose2d(w, null, opts);
            } catch (e) {
              error = e;
            }
            assert(error && pattern.test(error.message));
          }
          assertShape(x.convTranspose2d(w, null, { stride: [] }).shape, [1, 1, 4, 4]);
          assertShape(x.convTranspose2d(w, null, { outputPadding: [1] }).shape, [1, 1, 4, 6]);
          assertShape(x.convTranspose2d(w, null, { outputPadding: [1, 1, 1] }).shape, [1, 1, 5, 5]);
        });
        await test("surface owners factories and integer operations", async () => {
          const x = new Tensor([-7, -1, 1, 7], { dtype: "int16" });
          assertClose(await x.fullLike(7).toArray(), [7, 7, 7, 7]);
          assert(x.onesLike().dtype === x.dtype);
          assertClose(await x.zerosLike().toArray(), [0, 0, 0, 0]);
          assertClose(await x.bitwiseNot().toArray(), [6, 0, -2, -8]);
          assertClose(await x.mod(3).toArray(), [2, 2, 1, 1]);
          assertClose(await x.fmod(3).toArray(), [-1, -1, 1, 1]);
          assertClose(await x.floorDiv(3).toArray(), [-3, -1, 0, 2]);
          assertClose(await Tensor.normal([2, 3], { mean: 2, std: 0 }).toArray(), Array(6).fill(2));
          assertShape(Tensor.kaimingNormal([2, 3], { a: 0.2 }).shape, [2, 3]);
        });
        await test("surface owners reductions and stable scans", async () => {
          const x = new Tensor([[1, 2, 3], [4, 5, 6]], { dtype: "float32" });
          assertClose(await x.prod(1).toArray(), [6, 120]);
          assertClose(await x.argmin(1).toArray(), [0, 0]);
          assertClose(await x.logsumexp(1).toArray(), [3.407606, 6.407606]);
          assertClose(await x.logcumsumexp(1).toArray(), [1, 2.313262, 3.407606, 4, 5.313262, 6.407606]);
          assertClose(await x.normalize({ dim: 1 }).toArray(), [1 / Math.sqrt(14), 2 / Math.sqrt(14), 3 / Math.sqrt(14), 4 / Math.sqrt(77), 5 / Math.sqrt(77), 6 / Math.sqrt(77)]);
          assertClose(await x.softmin(1).toArray(), await x.neg().softmax(1).toArray());
          const [std, mean] = x.stdMean(1, false, 0);
          assertClose(await std.toArray(), [Math.sqrt(2 / 3), Math.sqrt(2 / 3)]);
          assertClose(await mean.toArray(), [2, 5]);
          assertClose(await new Tensor([1e3, 1001, 1002], { dtype: "float32" }).logcumsumexp().toArray(), [1e3, 1001.313262, 1002.407606], 1e-3);
        });
        await test("surface owners promoted division and stack gradients", async () => {
          for (const value of [3, 3.5]) {
            const copy = new Tensor(value).add(value).clone();
            assert(copy.dtype === (Number.isInteger(value) ? "int32" : "float32"));
            assertClose(await copy.toArray(), [2 * value]);
          }
          const x = new Tensor([-4, 7, 5], { dtype: "int32" }), y = new Tensor([2, -3, 8], { dtype: "float32" });
          assertClose(await x.div(y, "trunc").toArray(), [-2, -2, 0]);
          assertClose(await x.fmod(y).toArray(), [0, 1, 5]);
          assertClose(await x.mod(y).toArray(), [0, -2, 5]);
          const a = new Tensor([1, 2], { dtype: "float32" }), b = new Tensor([3, 4], { dtype: "float32" });
          Tensor.stack(a, b, a).mul(new Tensor([[1, 1], [2, 2], [3, 3]], { dtype: "float32" })).sum().backward();
          assertClose(await a.grad.toArray(), [4, 4]);
          assertClose(await b.grad.toArray(), [2, 2]);
        });
        await test("surface owners uint64 scalar and full literals", async () => {
          const value = (1n << 64n) - 1n;
          const x = new Tensor(value, { dtype: "uint64" });
          const got = await x.div(1n, "trunc").toArray();
          assert(BigInt(got[0]) === value);
          const full = await x.fullLike(value).toArray();
          assert(BigInt(full[0]) === value);
        });
        await test("surface owners shape helpers and padding", async () => {
          const x = new Tensor([1, 2, 3], { dtype: "float32" });
          assertClose(await x.diag().toArray(), [1, 0, 0, 0, 2, 0, 0, 0, 3]);
          assertClose(await x.unfold(0, 2, 1).toArray(), [1, 2, 2, 3]);
          assertClose(await x.unfold(0, 2, 2 ** 32).toArray(), [1, 2]);
          for (const offset of [2 ** 32, -(2 ** 32)]) {
            let error;
            try {
              x.reshape([1, 3]).diagonal(offset);
            } catch (e) {
              error = e;
            }
            assert(error, "wide offset must be rejected, not truncated to zero");
          }
          assertClose(await x.maskedFill(x.gt(1), -4).toArray(), [1, -4, -4]);
          assertClose(await new Tensor([[1, 2, 3], [4, 5, 6]]).diagonal(1).toArray(), [2, 6]);
          const [a, b] = Tensor.meshgrid(new Tensor([1, 2]), new Tensor([3, 4, 5]), { indexing: "xy" });
          assertClose(await a.toArray(), [1, 2, 1, 2, 1, 2]);
          assertClose(await b.toArray(), [3, 3, 4, 4, 5, 5]);
          for (const [mode, expected] of [["circular", [3, 1, 2, 3, 1]], ["reflect", [2, 1, 2, 3, 2]], ["replicate", [1, 1, 2, 3, 3]]]) {
            assertClose(await x.pad([1, 1], mode).toArray(), expected);
          }
          const empty = Tensor.empty([0, 3]);
          assert(empty.roll(1, 0) === empty);
        });
        await test("surface owners GELU and sparse loss options", async () => {
          const x = new Tensor([-1, 0, 1], { dtype: "float32" });
          assertClose(await x.gelu("none").toArray(), [-0.158655, 0, 0.841345], 1e-5);
          const logits = new Tensor([[1, 2, 3], [3, 2, 1]], { dtype: "float32" });
          const labels = new Tensor([2, 99], { dtype: "int32" });
          assertClose(await logits.sparseCategoricalCrossentropy(labels, { ignoreIndex: 99, labelSmoothing: 0.2, reduction: "none" }).toArray(), [0.60760596, 0]);
        });
        console.log(`
Results: ${passed} passed, ${failed} failed, ${skipped} skipped, ${passed + failed + skipped} total`);
        return { passed, failed, skipped };
      }
      async function checkLogicalRuntimeOption(polygrad, core) {
        const runtime = await polygrad.create({ core, logical: "never" });
        try {
          const tensor = new runtime.Tensor([1, 2]);
          assert(tensor.logicalPolicy === "never");
          assert(tensor.uopLogical === null);
        } finally {
          await runtime.dispose();
        }
      }
      module.exports = { checkLogicalRuntimeOption, runTensorTests };
    }
  });

  // ../test/fixtures/llama.json
  var require_llama = __commonJS({
    "../test/fixtures/llama.json"(exports, module) {
      module.exports = {
        reference: "Transformers 5.3.0 LlamaForCausalLM, Torch 2.10.0; unscaled cases cross-checked against Tinygrad v0.14.0 6f87158d77f66a36d5f8bbe915170b24e2acabe8; test/generate_llama_fixture.py",
        cases: [
          {
            version: "2",
            config: {
              model_type: "llama",
              hidden_size: 8,
              intermediate_size: 12,
              num_attention_heads: 2,
              num_key_value_heads: 1,
              num_hidden_layers: 2,
              vocab_size: 11,
              rms_norm_eps: 1e-5,
              rope_theta: 1e4,
              max_position_embeddings: 16,
              batch_size: 1,
              max_seq_len: 3,
              tie_word_embeddings: false
            },
            tokens: [
              [
                1,
                4,
                2
              ]
            ],
            weights: {
              "model.embed_tokens.weight": [
                11,
                8
              ],
              "model.layers.0.input_layernorm.weight": [
                8
              ],
              "model.layers.0.post_attention_layernorm.weight": [
                8
              ],
              "model.layers.0.self_attn.q_proj.weight": [
                8,
                8
              ],
              "model.layers.0.self_attn.k_proj.weight": [
                4,
                8
              ],
              "model.layers.0.self_attn.v_proj.weight": [
                4,
                8
              ],
              "model.layers.0.self_attn.o_proj.weight": [
                8,
                8
              ],
              "model.layers.0.mlp.gate_proj.weight": [
                12,
                8
              ],
              "model.layers.0.mlp.up_proj.weight": [
                12,
                8
              ],
              "model.layers.0.mlp.down_proj.weight": [
                8,
                12
              ],
              "model.layers.1.input_layernorm.weight": [
                8
              ],
              "model.layers.1.post_attention_layernorm.weight": [
                8
              ],
              "model.layers.1.self_attn.q_proj.weight": [
                8,
                8
              ],
              "model.layers.1.self_attn.k_proj.weight": [
                4,
                8
              ],
              "model.layers.1.self_attn.v_proj.weight": [
                4,
                8
              ],
              "model.layers.1.self_attn.o_proj.weight": [
                8,
                8
              ],
              "model.layers.1.mlp.gate_proj.weight": [
                12,
                8
              ],
              "model.layers.1.mlp.up_proj.weight": [
                12,
                8
              ],
              "model.layers.1.mlp.down_proj.weight": [
                8,
                12
              ],
              "model.norm.weight": [
                8
              ],
              "lm_head.weight": [
                11,
                8
              ]
            },
            logits: [
              -0.08373266458511353,
              0.2123628556728363,
              0.40279674530029297,
              0.2580987513065338,
              -0.1411241590976715,
              -0.07445239275693893,
              -0.5254123210906982,
              -0.0989779531955719,
              -0.37538909912109375,
              0.3875514566898346,
              0.24285343289375305,
              -0.2758362293243408,
              -0.399817556142807,
              0.06484386324882507,
              -0.33821916580200195,
              0.8110876679420471,
              -0.3126642405986786,
              0.09696711599826813,
              -0.2963690161705017,
              -0.17400763928890228,
              0.04431106895208359,
              -0.35875195264816284,
              -0.13071969151496887,
              -0.48286890983581543,
              0.19611120223999023,
              -0.4180883765220642,
              0.5790441632270813,
              -0.24215082824230194,
              -0.004503313452005386,
              -0.1523132026195526,
              -0.24975328147411346,
              0.1745176911354065,
              -0.43968188762664795
            ],
            freqs_cos: [
              1,
              1,
              0.5403023362159729,
              0.9999499917030334,
              -0.416146844625473,
              0.9998000264167786
            ],
            freqs_sin: [
              0,
              0,
              0.8414709568023682,
              0.009999833069741726,
              0.9092974066734314,
              0.019998665899038315
            ]
          },
          {
            version: "2",
            config: {
              model_type: "llama",
              hidden_size: 8,
              intermediate_size: 12,
              num_attention_heads: 2,
              num_key_value_heads: 2,
              num_hidden_layers: 2,
              vocab_size: 11,
              rms_norm_eps: 1e-5,
              rope_theta: 1e4,
              max_position_embeddings: 16,
              batch_size: 1,
              max_seq_len: 3,
              tie_word_embeddings: false
            },
            tokens: [
              [
                1,
                4,
                2
              ]
            ],
            weights: {
              "model.embed_tokens.weight": [
                11,
                8
              ],
              "model.layers.0.input_layernorm.weight": [
                8
              ],
              "model.layers.0.post_attention_layernorm.weight": [
                8
              ],
              "model.layers.0.self_attn.q_proj.weight": [
                8,
                8
              ],
              "model.layers.0.self_attn.k_proj.weight": [
                8,
                8
              ],
              "model.layers.0.self_attn.v_proj.weight": [
                8,
                8
              ],
              "model.layers.0.self_attn.o_proj.weight": [
                8,
                8
              ],
              "model.layers.0.mlp.gate_proj.weight": [
                12,
                8
              ],
              "model.layers.0.mlp.up_proj.weight": [
                12,
                8
              ],
              "model.layers.0.mlp.down_proj.weight": [
                8,
                12
              ],
              "model.layers.1.input_layernorm.weight": [
                8
              ],
              "model.layers.1.post_attention_layernorm.weight": [
                8
              ],
              "model.layers.1.self_attn.q_proj.weight": [
                8,
                8
              ],
              "model.layers.1.self_attn.k_proj.weight": [
                8,
                8
              ],
              "model.layers.1.self_attn.v_proj.weight": [
                8,
                8
              ],
              "model.layers.1.self_attn.o_proj.weight": [
                8,
                8
              ],
              "model.layers.1.mlp.gate_proj.weight": [
                12,
                8
              ],
              "model.layers.1.mlp.up_proj.weight": [
                12,
                8
              ],
              "model.layers.1.mlp.down_proj.weight": [
                8,
                12
              ],
              "model.norm.weight": [
                8
              ],
              "lm_head.weight": [
                11,
                8
              ]
            },
            logits: [
              -0.07643559575080872,
              0.20543313026428223,
              -0.19305849075317383,
              0.22550560534000397,
              -0.0670381486415863,
              -0.058988794684410095,
              -0.4904012382030487,
              -0.08312639594078064,
              0.09775981307029724,
              -0.19974926114082336,
              0.21881479024887085,
              -0.32808613777160645,
              -0.24772416055202484,
              -0.04583235830068588,
              -0.2104089856147766,
              0.8113012909889221,
              -0.3779626786708832,
              0.06057564914226532,
              -0.3405245244503021,
              -0.06800717115402222,
              -0.05827075242996216,
              -0.2228473722934723,
              -0.3212721645832062,
              -0.1643604338169098,
              0.23591576516628265,
              -0.1650802493095398,
              0.4286355674266815,
              -0.37274041771888733,
              0.03325618803501129,
              -0.32103222608566284,
              -0.188359797000885,
              0.23615577816963196,
              -0.16484034061431885
            ],
            freqs_cos: [
              1,
              1,
              0.5403023362159729,
              0.9999499917030334,
              -0.416146844625473,
              0.9998000264167786
            ],
            freqs_sin: [
              0,
              0,
              0.8414709568023682,
              0.009999833069741726,
              0.9092974066734314,
              0.019998665899038315
            ]
          },
          {
            version: "3",
            config: {
              model_type: "llama",
              hidden_size: 8,
              intermediate_size: 12,
              num_attention_heads: 2,
              num_key_value_heads: 1,
              num_hidden_layers: 2,
              vocab_size: 11,
              rms_norm_eps: 1e-5,
              rope_theta: 5e5,
              max_position_embeddings: 16,
              batch_size: 1,
              max_seq_len: 3,
              tie_word_embeddings: false
            },
            tokens: [
              [
                1,
                4,
                2
              ]
            ],
            weights: {
              "model.embed_tokens.weight": [
                11,
                8
              ],
              "model.layers.0.input_layernorm.weight": [
                8
              ],
              "model.layers.0.post_attention_layernorm.weight": [
                8
              ],
              "model.layers.0.self_attn.q_proj.weight": [
                8,
                8
              ],
              "model.layers.0.self_attn.k_proj.weight": [
                4,
                8
              ],
              "model.layers.0.self_attn.v_proj.weight": [
                4,
                8
              ],
              "model.layers.0.self_attn.o_proj.weight": [
                8,
                8
              ],
              "model.layers.0.mlp.gate_proj.weight": [
                12,
                8
              ],
              "model.layers.0.mlp.up_proj.weight": [
                12,
                8
              ],
              "model.layers.0.mlp.down_proj.weight": [
                8,
                12
              ],
              "model.layers.1.input_layernorm.weight": [
                8
              ],
              "model.layers.1.post_attention_layernorm.weight": [
                8
              ],
              "model.layers.1.self_attn.q_proj.weight": [
                8,
                8
              ],
              "model.layers.1.self_attn.k_proj.weight": [
                4,
                8
              ],
              "model.layers.1.self_attn.v_proj.weight": [
                4,
                8
              ],
              "model.layers.1.self_attn.o_proj.weight": [
                8,
                8
              ],
              "model.layers.1.mlp.gate_proj.weight": [
                12,
                8
              ],
              "model.layers.1.mlp.up_proj.weight": [
                12,
                8
              ],
              "model.layers.1.mlp.down_proj.weight": [
                8,
                12
              ],
              "model.norm.weight": [
                8
              ],
              "lm_head.weight": [
                11,
                8
              ]
            },
            logits: [
              -0.08373266458511353,
              0.2123628556728363,
              0.40279674530029297,
              0.2580987513065338,
              -0.1411241590976715,
              -0.07445239275693893,
              -0.5254123210906982,
              -0.0989779531955719,
              -0.37538909912109375,
              0.3875514566898346,
              0.24285343289375305,
              -0.2758150100708008,
              -0.39993587136268616,
              0.06468798965215683,
              -0.338351845741272,
              0.8112592697143555,
              -0.3126344084739685,
              0.09720116853713989,
              -0.29634302854537964,
              -0.17398260533809662,
              0.04415999352931976,
              -0.35887986421585083,
              -0.1312912106513977,
              -0.4825049042701721,
              0.19624802470207214,
              -0.4178074300289154,
              0.5791915059089661,
              -0.24188901484012604,
              -0.004148326814174652,
              -0.1528570055961609,
              -0.2501845955848694,
              0.17468221485614777,
              -0.439373254776001
            ],
            freqs_cos: [
              1,
              1,
              0.5403023362159729,
              0.9999989867210388,
              -0.416146844625473,
              0.9999960064888
            ],
            freqs_sin: [
              0,
              0,
              0.8414709568023682,
              0.0014142129803076386,
              0.9092974066734314,
              0.0028284231666475534
            ]
          },
          {
            version: "3.2",
            config: {
              model_type: "llama",
              hidden_size: 16,
              intermediate_size: 24,
              num_attention_heads: 2,
              num_key_value_heads: 1,
              num_hidden_layers: 2,
              vocab_size: 11,
              rms_norm_eps: 1e-5,
              rope_theta: 5e5,
              max_position_embeddings: 16,
              batch_size: 1,
              max_seq_len: 3,
              tie_word_embeddings: true,
              rope_scaling: {
                rope_type: "llama3",
                factor: 32,
                low_freq_factor: 1,
                high_freq_factor: 4,
                original_max_position_embeddings: 8192,
                rope_theta: 5e5
              }
            },
            tokens: [
              [
                1,
                4,
                2
              ]
            ],
            weights: {
              "model.embed_tokens.weight": [
                11,
                16
              ],
              "model.layers.0.input_layernorm.weight": [
                16
              ],
              "model.layers.0.post_attention_layernorm.weight": [
                16
              ],
              "model.layers.0.self_attn.q_proj.weight": [
                16,
                16
              ],
              "model.layers.0.self_attn.k_proj.weight": [
                8,
                16
              ],
              "model.layers.0.self_attn.v_proj.weight": [
                8,
                16
              ],
              "model.layers.0.self_attn.o_proj.weight": [
                16,
                16
              ],
              "model.layers.0.mlp.gate_proj.weight": [
                24,
                16
              ],
              "model.layers.0.mlp.up_proj.weight": [
                24,
                16
              ],
              "model.layers.0.mlp.down_proj.weight": [
                16,
                24
              ],
              "model.layers.1.input_layernorm.weight": [
                16
              ],
              "model.layers.1.post_attention_layernorm.weight": [
                16
              ],
              "model.layers.1.self_attn.q_proj.weight": [
                16,
                16
              ],
              "model.layers.1.self_attn.k_proj.weight": [
                8,
                16
              ],
              "model.layers.1.self_attn.v_proj.weight": [
                8,
                16
              ],
              "model.layers.1.self_attn.o_proj.weight": [
                16,
                16
              ],
              "model.layers.1.mlp.gate_proj.weight": [
                24,
                16
              ],
              "model.layers.1.mlp.up_proj.weight": [
                24,
                16
              ],
              "model.layers.1.mlp.down_proj.weight": [
                16,
                24
              ],
              "model.norm.weight": [
                16
              ]
            },
            logits: [
              0.2713891267776489,
              0.9291383624076843,
              6756186485290527e-19,
              0.20870482921600342,
              -0.09061503410339355,
              -0.919259786605835,
              -0.18096023797988892,
              0.28555113077163696,
              0.28547197580337524,
              0.13032814860343933,
              0.08401966094970703,
              -0.1375708281993866,
              -0.7450186014175415,
              -0.6900337934494019,
              0.19840417802333832,
              1.279048204421997,
              0.43855059146881104,
              -0.2694587707519531,
              0.11852874606847763,
              -0.1626301407814026,
              -0.797093391418457,
              -0.4763565957546234,
              0.1928788125514984,
              0.5480930209159851,
              0.997273325920105,
              0.25748005509376526,
              -0.10457956790924072,
              -0.1515563428401947,
              -0.8919546604156494,
              -0.0967339277267456,
              0.21408119797706604,
              0.6117810606956482,
              0.07914119958877563
            ],
            freqs_cos: [
              1,
              1,
              1,
              1,
              0.5403023362159729,
              0.9992929697036743,
              0.9999999403953552,
              1,
              -0.416146844625473,
              0.9971728920936584,
              0.9999996423721313,
              1
            ],
            freqs_sin: [
              0,
              0,
              0,
              0,
              0.8414709568023682,
              0.0375971682369709,
              429556705057621e-18,
              16619674170215148e-22,
              0.9092974066734314,
              0.07514116913080215,
              8591132936999202e-19,
              33239348340430297e-22
            ]
          }
        ]
      };
    }
  });

  // ../test/fixtures/model_definition.json
  var require_model_definition = __commonJS({
    "../test/fixtures/model_definition.json"(exports, module) {
      module.exports = {
        format: "poly.modeldef@1",
        type: "graph",
        seed: 42,
        inputs: { x: { shape: [1, 2], dtype: "float32" } },
        modules: { shared: { type: "linear", out_features: 2, bias: false } },
        nodes: [
          { name: "a", call: "shared", inputs: ["x"] },
          { name: "b", call: "shared", inputs: ["a"] },
          { name: "residual", type: "add", inputs: ["x", "b"] },
          { name: "total", type: "sum", inputs: ["residual"] }
        ],
        outputs: { prediction: "residual", cost: "total" },
        entrypoints: [
          { name: "forward", inputs: ["x"], outputs: ["prediction"] },
          { name: "loss", inputs: ["x"], outputs: ["cost"], objective: "cost" }
        ]
      };
    }
  });

  // ../test/fixtures/gguf_quantized_blocks.json
  var require_gguf_quantized_blocks = __commonJS({
    "../test/fixtures/gguf_quantized_blocks.json"(exports, module) {
      module.exports = {
        reference: "a9069c177a9da9cca18593edf55acd2e6073cca6",
        cases: [
          { type: 2, bytes: [0, 60, 240, 225, 210, 195, 180, 165, 150, 135, 120, 105, 90, 75, 60, 45, 30, 15], values: [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8] },
          { type: 3, bytes: [0, 60, 0, 64, 240, 225, 210, 195, 180, 165, 150, 135, 120, 105, 90, 75, 60, 45, 30, 15], values: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2] },
          { type: 14, bytes: [7, 20, 33, 46, 59, 72, 85, 98, 111, 124, 137, 150, 163, 176, 189, 202, 215, 228, 241, 254, 11, 24, 37, 50, 63, 76, 89, 102, 115, 128, 141, 154, 167, 180, 193, 206, 219, 232, 245, 2, 15, 28, 41, 54, 67, 80, 93, 106, 119, 132, 145, 158, 171, 184, 197, 210, 223, 236, 249, 6, 19, 32, 45, 58, 71, 84, 97, 110, 123, 136, 149, 162, 175, 188, 201, 214, 227, 240, 253, 10, 23, 36, 49, 62, 75, 88, 101, 114, 127, 140, 153, 166, 179, 192, 205, 218, 231, 244, 1, 14, 27, 40, 53, 66, 79, 92, 105, 118, 131, 144, 157, 170, 183, 196, 209, 222, 235, 248, 5, 18, 31, 44, 57, 70, 83, 96, 109, 122, 3, 10, 17, 24, 31, 38, 45, 52, 59, 66, 73, 80, 87, 94, 101, 108, 115, 122, 129, 136, 143, 150, 157, 164, 171, 178, 185, 192, 199, 206, 213, 220, 227, 234, 241, 248, 255, 6, 13, 20, 27, 34, 41, 48, 55, 62, 69, 76, 83, 90, 97, 104, 111, 118, 125, 132, 139, 146, 153, 160, 167, 174, 181, 188, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0, 60], values: [23, 4, -15, -18, 27, 8, -11, -30, 31, 12, -7, -26, 19, 0, -3, -22, 46, 8, -30, -36, 54, 16, -22, -60, 62, 24, -14, -52, 38, 0, -6, -44, -75, 12, -93, 42, 81, -24, 63, -42, 45, -60, 27, -78, -39, 48, -9, 78, -100, 16, -124, 56, 108, -32, 84, -56, 60, -80, 36, -104, -52, 64, -12, 104, -160, -155, -70, -70, -65, 20, 25, 110, 110, -125, -120, -35, -30, -25, 55, 60, 174, 180, -102, -102, -192, -90, -84, 18, 18, 120, 126, -156, -150, -144, -48, -42, -154, -147, -140, -140, -133, -126, -119, -224, -224, -105, -98, -91, -84, -77, -77, -70, -72, -64, 72, 72, 80, 88, 96, 104, 104, 112, 120, 128, 136, 144, 144, 152, 207, 36, -135, -162, 243, 72, -99, -270, 279, 108, -63, -234, 171, 0, -27, -198, 230, 40, -150, -180, 270, 80, -110, -300, 310, 120, -70, -260, 190, 0, -30, -220, -275, 44, -341, 154, 297, -88, 231, -154, 165, -220, 99, -286, -143, 176, -33, 286, -300, 48, -372, 168, 324, -96, 252, -168, 180, -240, 108, -312, -156, 192, -36, 312, 52, 65, 286, 286, 299, -312, -299, -78, -78, 143, 156, 377, 390, 403, -221, -416, -210, -196, 42, 42, 56, 294, 308, -350, -350, -112, -98, 140, 154, 168, 392, 406, 450, 465, 240, 240, 255, -450, -435, -420, -420, -405, -390, -375, -360, -345, -105, -90, -80, -64, -48, -48, -32, -16, -256, 16, 16, 32, 48, 64, 80, 96, 96, 112] }
        ]
      };
    }
  });

  // test/test_model_runtime.js
  var require_test_model_runtime = __commonJS({
    "test/test_model_runtime.js"(exports, module) {
      "use strict";
      var llamaFixture = require_llama();
      async function checkLlamaFamily(pg) {
        const gpu = pg.device === "webgpu";
        for (const item of llamaFixture.cases) {
          const model = gpu ? await pg.models.LlamaAsync(item.config) : pg.models.Llama(item.config);
          let restored, imported;
          try {
            const header = {}, parts = [];
            let offsetBytes = 0;
            for (const [name2, shape] of Object.entries(item.weights)) {
              const offset = Array.from(name2).reduce((s, c) => s + c.charCodeAt(0), 0);
              const values = Float32Array.from({ length: shape.reduce((a, b) => a * b, 1) }, (_, i) => (shape.length === 1 ? 1 : 0) + ((i * 7 + offset) % 23 - 11) * 0.017);
              await model.writeBufferAsync(name2, values);
              header[name2] = { dtype: "F32", shape, data_offsets: [offsetBytes, offsetBytes + values.byteLength] };
              parts.push(new Uint8Array(values.buffer));
              offsetBytes += values.byteLength;
            }
            if (item.config.tie_word_embeddings) {
              header["lm_head.weight"] = header["model.embed_tokens.weight"];
              delete header["model.embed_tokens.weight"];
            }
            const headerBytes = new TextEncoder().encode(JSON.stringify(header));
            const checkpoint = new Uint8Array(8 + headerBytes.length + offsetBytes);
            new DataView(checkpoint.buffer).setBigUint64(0, BigInt(headerBytes.length), true);
            checkpoint.set(headerBytes, 8);
            offsetBytes = 8 + headerBytes.length;
            for (const part of parts) {
              checkpoint.set(part, offsetBytes);
              offsetBytes += part.length;
            }
            imported = pg.Model.fromHF(new TextEncoder().encode(JSON.stringify(item.config)), [checkpoint], { maxSeqLen: 3 });
            assertClose(await model.readBufferAsync("freqs_cos"), item.freqs_cos, 2e-6);
            assertClose(await model.readBufferAsync("freqs_sin"), item.freqs_sin, 2e-6);
            const tokens = new Int32Array([1, 4, 2]);
            const first = (await model.forwardAsync({ tokens })).logits;
            assertClose(first, item.logits, 2e-5);
            assertClose((await imported.forwardAsync({ tokens })).logits, item.logits, 2e-5);
            const changed = (await model.forwardAsync({ tokens: new Int32Array([1, 4, 7]) })).logits;
            assertClose(changed.slice(0, 22), first.slice(0, 22), 2e-6);
            restored = pg.Model.load(await model.saveAsync({ includeOptimizer: false }));
            assertClose((await restored.forwardAsync({ tokens })).logits, item.logits, 2e-5);
            const name = "model.embed_tokens.weight", before = await model.readBufferAsync(name);
            await restored.writeBufferAsync(name, new Float32Array(before.length));
            assertClose(await model.readBufferAsync(name), before, 0);
            if (item.config.tie_word_embeddings)
              assertClose(await restored.readBufferAsync("lm_head.weight"), new Float32Array(before.length), 0);
          } finally {
            if (restored) await restored.dispose();
            if (imported) await imported.dispose();
            await model.dispose();
          }
        }
      }
      var compositionFixture = require_model_definition();
      var quantizedFixture = require_gguf_quantized_blocks();
      async function checkModelCheckpointReplacement(pg) {
        const model = pg.models.MLP({ layers: [2, 1], loss: "none", seed: 3 });
        try {
          const name = "layers.0.weight";
          const original = await model.readBufferAsync(name);
          const weights = await model.exportWeightsAsync();
          await model.writeBufferAsync(name, new Float32Array(original.length));
          if (pg.device === "webgpu") {
            let rejected2 = false;
            try {
              model.importWeights(weights);
            } catch (e) {
              rejected2 = /Async/.test(e.message);
            }
            assert(rejected2, "synchronous GPU replacement must reject before entering Wasm");
          }
          const pending = model.importWeightsAsync(weights);
          weights.fill(0);
          await pending;
          assertClose(await model.readBufferAsync(name), original, 0);
          let rejected = false;
          try {
            await model.importWeightsAsync(new Uint8Array([1, 2, 3]));
          } catch (_) {
            rejected = true;
          }
          assert(rejected, "malformed checkpoint must reject");
          assertClose(await model.readBufferAsync(name), original, 0);
        } finally {
          await model.dispose();
        }
      }
      async function checkModelStatefulCapture(pg) {
        const gpu = String(pg.device).toLowerCase() === "webgpu";
        const net = { bn: new pg.nn.BatchNorm(2), forward({ x }) {
          return { prediction: this.bn.call(x) };
        } };
        const mode = pg.Tensor.training;
        const opts = {
          inputs: { x: pg.Tensor.empty([2, 2]) },
          targets: { y: pg.Tensor.empty([2, 2]) },
          loss: (out, { y }) => out.prediction.sub(y).square().mean()
        };
        const model = gpu ? await pg.Model.fromCallableAsync(net, opts) : new pg.Model(net, opts);
        let restored;
        try {
          assert(pg.Tensor.training === mode, "capture changed authoring mode");
          assertClose(await model.readBufferAsync("bn.runningMean"), [0, 0], 0);
          assert(Number((await model.readBufferAsync("bn.numBatchesTracked"))[0]) === 0);
          model.setOptimizer("sgd", 0.01);
          const x = new Float32Array([1, 2, 3, 6]), y = new Float32Array(4);
          await model.trainStepAsync({ x, y });
          assertClose(await model.readBufferAsync("bn.runningMean"), [0.2, 0.4], 1e-6);
          assertClose(await model.readBufferAsync("bn.runningVar"), [1.1, 1.7], 1e-6);
          assert(Number((await model.readBufferAsync("bn.numBatchesTracked"))[0]) === 1);
          assertClose(await net.bn.runningMean.toArrayAsync(), [0, 0], 0);
          const weight = await model.readBufferAsync("bn.weight"), bias = await model.readBufferAsync("bn.bias");
          const expected = Array.from(x, (v, i) => (v - [0.2, 0.4][i % 2]) / Math.sqrt([1.1, 1.7][i % 2] + 1e-5) * weight[i % 2] + bias[i % 2]);
          assertClose((await model.forwardAsync({ x })).prediction, expected, 1e-5);
          assert(Number((await model.readBufferAsync("bn.numBatchesTracked"))[0]) === 1);
          restored = pg.Model.load(await model.saveAsync());
          restored.setOptimizer("sgd", 0.01);
          const a = await model.trainStepAsync({ x, y: x }), b = await restored.trainStepAsync({ x, y: x });
          assert(Math.abs(a - b) < 1e-5, "restored training diverged");
          assert(Number((await restored.readBufferAsync("bn.numBatchesTracked"))[0]) === 2);
        } finally {
          if (restored) await restored.dispose();
          await model.dispose();
          for (const tensor of Object.values(net.bn)) if (tensor && tensor.dispose) await tensor.dispose();
          for (const tensor of [...Object.values(opts.inputs), ...Object.values(opts.targets)]) await tensor.dispose();
        }
      }
      async function checkModelCaptureRng(pg) {
        const gpu = String(pg.device).toLowerCase() === "webgpu";
        pg.Tensor.manual_seed(123);
        const controlTensor = pg.Tensor.rand(16);
        const control = await controlTensor.toArrayAsync();
        await controlTensor.dispose();
        pg.Tensor.manual_seed(123);
        const weight = new pg.Tensor([1], { dtype: "float32" }), input = pg.Tensor.empty(16), target = pg.Tensor.empty(16);
        const opts = { inputs: { x: input }, targets: { y: target }, params: { weight, alias: weight }, loss: (out, { y }) => out.sub(y).square().mean() };
        const author = ({ x }) => x.mul(weight).dropout(0.5);
        const model = gpu ? await pg.Model.fromCallableAsync(author, opts) : new pg.Model(author, opts);
        let restored, sibling, roundtripped;
        try {
          const after = pg.Tensor.rand(16);
          try {
            assertClose(await after.toArrayAsync(), control, 0);
          } finally {
            await after.dispose();
          }
          const counters = Array.from({ length: model.bufCount }, (_, i) => model.bufName(i)).filter((name) => name.startsWith("__rng.") && name.endsWith(".counter"));
          assert(counters.length === 1, "capture must own one RNG counter");
          const counter = counters[0];
          assertClose(await model.readBufferAsync(counter), [0, 0], 0);
          model.setOptimizer("sgd", 0.01);
          const x = new Float32Array(16).fill(1), y = new Float32Array(16);
          await model.trainStepAsync({ x, y });
          const state = await model.readBufferAsync(counter);
          assert(state.some((v) => v !== 0), "training did not advance RNG");
          await model.forwardAsync({ x });
          assertClose(await model.readBufferAsync(counter), state, 0);
          const bytes = await model.saveAsync();
          restored = pg.Model.load(bytes);
          sibling = pg.Model.load(bytes);
          roundtripped = pg.Model.load(await restored.saveAsync());
          const siblingWeight = await sibling.readBufferAsync("weight");
          for (const copy of [restored, roundtripped]) copy.setOptimizer("sgd", 0.01);
          for (let i = 0; i < 2; i++) {
            const a = await model.trainStepAsync({ x, y });
            for (const copy of [restored, roundtripped]) {
              const b = await copy.trainStepAsync({ x, y });
              assert(Math.abs(a - b) < 1e-5, "checkpoint changed RNG continuation");
              assertClose(await model.readBufferAsync(counter), await copy.readBufferAsync(counter), 0);
              assertClose(await model.readBufferAsync("weight"), await copy.readBufferAsync("weight"), 0);
              assertClose(await copy.readBufferAsync("alias"), await copy.readBufferAsync("weight"), 0);
            }
            assertClose(await sibling.readBufferAsync(counter), state, 0);
            assertClose(await sibling.readBufferAsync("weight"), siblingWeight, 0);
          }
          await restored.writeBufferAsync("alias", new Float32Array([71]));
          assertClose(await restored.readBufferAsync("weight"), [71], 0);
          assertClose(await sibling.readBufferAsync("weight"), siblingWeight, 0);
          await restored.dispose();
          pg.clearScheduleCache();
          pg.collect();
          assertClose(await sibling.readBufferAsync(counter), state, 0);
          assertClose((await sibling.forwardAsync({ x })).output, Array(16).fill(siblingWeight[0]), 0);
        } finally {
          if (roundtripped) await roundtripped.dispose();
          if (sibling) await sibling.dispose();
          if (restored) await restored.dispose();
          await model.dispose();
          for (const tensor of [weight, input, target]) await tensor.dispose();
        }
      }
      async function checkModelCaptureFailure(pg) {
        const gpu = String(pg.device).toLowerCase() === "webgpu";
        const x = pg.Tensor.empty(1), state = new pg.Tensor([0]).is_param_(false);
        const options = { inputs: { x }, params: { state } };
        const mode = pg.Tensor.training;
        const attempt = async (author) => {
          let model;
          try {
            model = gpu ? await pg.Model.fromCallableAsync(author, options) : new pg.Model(author, options);
            return false;
          } catch {
            return true;
          } finally {
            if (model) await model.dispose();
          }
        };
        try {
          assert(await attempt(({ x: x2 }) => {
            state.assign(state.add(1));
            throw new Error("author failed");
          }));
          assert(pg.Tensor.training === mode);
          assertClose(await state.toArrayAsync(), [0], 0);
          state.is_param_(true);
          assert(await attempt(({ x: x2 }) => {
            state.assign(state.add(1));
            return x2;
          }), "parameter assignment escaped capture");
          assertClose(await state.toArrayAsync(), [0], 0);
          state.is_param_(false);
          let pending;
          const rejected = await attempt(({ x: x2 }) => {
            pending = state.toArrayAsync().catch(() => {
            });
            return x2;
          });
          await pending;
          assert(rejected, "capture allowed asynchronous execution to escape its scope");
        } finally {
          await x.dispose();
          await state.dispose();
        }
      }
      async function checkModelVariableShapes(pg) {
        const gpu = String(pg.device).toLowerCase() === "webgpu";
        const n = pg.uop.variable("model_batch", 1, 32), bound = n.bind(17);
        const x = pg.Tensor.empty([bound, 2]), y = pg.Tensor.empty([bound, 2]);
        const options = { inputs: { x, y } };
        const author = ({ x: x2, y: y2 }) => ({ prediction: x2.mul(2).add(y2) });
        const model = gpu ? await pg.Model.fromCallableAsync(author, options) : new pg.Model(author, options);
        let restored, first;
        const a = Float32Array.from({ length: 34 }, (_, i) => i);
        const input = new pg.Tensor(a).reshape(17, 2);
        try {
          const pending = model.forwardAsync({ x: input, y: { data: a, shape: [17, 2] } });
          if (gpu) {
            for (const query of [() => model.bufCurrentShape(0), () => model.bufShapeBounds(0)]) {
              let rejected2 = false;
              try {
                query();
              } catch (e) {
                rejected2 = /active async work/.test(e.message);
              }
              assert(rejected2, "metadata query entered suspended Wasm");
            }
          }
          first = (await pending).prediction;
          assert(JSON.stringify(first.shape) === "[17,2]", "result shape must be concrete");
          assertClose(await first.toArrayAsync(), Array.from(a, (v) => v * 3), 0);
          const small = a.slice(0, 6);
          const result = await model.forwardAsync({ x: small, y: { data: small, shape: [3, 2] } });
          assertClose(result.prediction, Array.from(small, (v) => v * 3), 0);
          assert(JSON.stringify(model.bufCurrentShape(model.findBuf("prediction"))) === "[3,2]");
          assert(JSON.stringify(model.bufShapeBounds(model.findBuf("x"))) === "[[1,32],[2,2]]");
          const before = await model.readBufferAsync("x");
          let rejected = false;
          try {
            await model.forwardAsync({ x: a, y: small });
          } catch {
            rejected = true;
          }
          assert(rejected, "shared shape variable must reject inconsistent inputs");
          assertClose(await model.readBufferAsync("x"), before, 0);
          const bytes = await model.saveAsync({ includeOptimizer: false });
          restored = pg.Model.load(bytes);
          const roundtripped = pg.Model.load(await restored.saveAsync({ includeOptimizer: false }));
          try {
            for (const copy of [restored, roundtripped]) {
              assert(JSON.stringify(copy.bufShapeBounds(copy.findBuf("x"))) === "[[1,32],[2,2]]");
              for (const size of [17, 3, 11]) {
                const values = Float32Array.from({ length: size * 2 }, (_, i) => i);
                assertClose((await copy.forwardAsync({ x: values, y: values })).prediction, Array.from(values, (v) => v * 3), 0);
              }
            }
          } finally {
            await roundtripped.dispose();
          }
          await model.dispose();
          pg.clearScheduleCache();
          pg.collect();
          assertClose(await first.toArrayAsync(), Array.from(a, (v) => v * 3), 0);
          const w = new pg.Tensor([0], { dtype: "float32" });
          const trainingOptions = {
            inputs: { x: pg.Tensor.empty([bound, 1]) },
            targets: { y: pg.Tensor.empty([bound, 1]) },
            params: { w },
            loss: (out, { y: y2 }) => out.sub(y2).square().mean()
          };
          const trainAuthor = ({ x: x2 }) => x2.mul(w);
          const training = gpu ? await pg.Model.fromCallableAsync(trainAuthor, trainingOptions) : new pg.Model(trainAuthor, trainingOptions);
          try {
            training.setOptimizer("sgd", 0.01);
            let expected = 0;
            for (const size of [17, 3, 11]) {
              const values = Float32Array.from({ length: size }, (_, i) => i + 1);
              const meanSquare = values.reduce((sum, v) => sum + v * v, 0) / size;
              const loss = await training.trainStepAsync({ x: values, y: values.map((v) => v * 2) });
              const wanted = (expected - 2) ** 2 * meanSquare;
              assert(Math.abs(loss - wanted) <= 1e-5 * Math.max(1, wanted), "loss used a stale batch extent");
              expected -= 0.02 * (expected - 2) * meanSquare;
              assertClose(await training.readBufferAsync("w"), [expected], 1e-4);
            }
          } finally {
            await training.dispose();
            await w.dispose();
          }
        } finally {
          if (restored) await restored.dispose();
          if (first) await first.dispose();
          await input.dispose();
          await model.dispose();
          await x.dispose();
          await y.dispose();
          bound.dispose();
          n.dispose();
        }
      }
      async function checkModelEmptyInputAdmission(pg) {
        const n = pg.uop.variable("empty_model_batch", 0, 4), bound = n.bind(3);
        const offset = pg.Tensor.empty([1]), x = pg.Tensor.empty([bound, 2]);
        const model = await pg.Model.fromCallableAsync(({ offset: offset2, x: x2 }) => x2.add(offset2), { inputs: { offset, x } });
        try {
          await model.forwardAsync({ offset: new Float32Array([7]), x: new Float32Array(6).fill(1) });
          const before = await model.readBufferAsync("offset");
          let rejected = false;
          try {
            await model.forwardAsync({ offset: new Float32Array([99]), x: new Float32Array(0) });
          } catch {
            rejected = true;
          }
          assert(rejected, "unsupported empty Model binding must reject");
          assertClose(await model.readBufferAsync("offset"), before, 0);
        } finally {
          await model.dispose();
          await offset.dispose();
          await x.dispose();
          bound.dispose();
          n.dispose();
        }
      }
      async function checkModelTensorIO(pg) {
        const gpu = String(pg.device).toLowerCase() === "webgpu";
        const author = ({ x, y }) => ({ prediction: x.mul(2).add(y) });
        const options = { inputs: { x: pg.Tensor.empty([2, 2]), y: pg.Tensor.empty([2, 2]) } };
        const model = gpu ? await pg.Model.fromCallableAsync(author, options) : new pg.Model(author, options);
        const owned = [];
        try {
          const input = new pg.Tensor([[1, 2], [3, 4]], { dtype: "float32" }).transpose();
          owned.push(input);
          const pending = model.forwardAsync({ x: input, y: new Float32Array([1, 1, 1, 1]) });
          const disposing = input.dispose();
          const first = (await pending).prediction;
          await disposing;
          owned.push(first);
          assert(first instanceof pg.Tensor, "Tensor input must select Tensor outputs");
          assert(JSON.stringify(first.shape) === "[2,2]", "result lost concrete shape");
          const zero = pg.Tensor.zeros([2, 2]);
          owned.push(zero);
          const second = (await model.forwardAsync({ x: first, y: zero })).prediction;
          owned.push(second);
          assertClose(await first.toArrayAsync(), [3, 7, 5, 9], 0);
          assertClose(await second.toArrayAsync(), [6, 14, 10, 18], 0);
          const before = await model.readBufferAsync("x");
          let shapeRejected = false;
          try {
            await model.forwardAsync({
              x: new Float32Array([9, 9, 9, 9]),
              y: { data: new Float32Array([0, 0, 0, 0]), shape: [1, 4] }
            });
          } catch {
            shapeRejected = true;
          }
          assert(shapeRejected, "equal bytes must not admit a wrong explicit host shape");
          assertClose(await model.readBufferAsync("x"), before, 0);
          const shaped = await model.forwardAsync({
            x: { data: new Float32Array([1, 2, 3, 4]), shape: [2, 2] },
            y: new Float32Array([0, 0, 0, 0])
          });
          assertClose(shaped.prediction, [2, 4, 6, 8], 0);
          const afterShape = await model.readBufferAsync("x");
          for (const bad of [pg.Tensor.zeros([4]), pg.Tensor.zeros([2, 2], { dtype: "int32" })]) {
            owned.push(bad);
            let rejected = false;
            try {
              await model.forwardAsync({ x: zero, y: bad });
            } catch {
              rejected = true;
            }
            assert(rejected, "Tensor signature mismatch must reject");
            assertClose(await model.readBufferAsync("x"), afterShape, 0);
          }
          const effect = pg.Tensor.ones([2, 2]).contiguous();
          owned.push(effect);
          await effect.realizeAsync();
          effect.assign(effect.add(1));
          for (let i = 0; i < 2; i++) {
            const result = (await model.forwardAsync({ x: effect, y: zero })).prediction;
            owned.push(result);
            assertClose(await result.toArrayAsync(), [4, 4, 4, 4], 0);
          }
          assertClose(await effect.toArrayAsync(), [2, 2, 2, 2], 0);
          const finishing = model.forwardAsync({ x: second, y: zero });
          const closing = model.dispose();
          const last = (await finishing).prediction;
          owned.push(last);
          await closing;
          pg.clearScheduleCache();
          pg.collect();
          assertClose(await first.toArrayAsync(), [3, 7, 5, 9], 0);
          assertClose(await last.toArrayAsync(), [12, 28, 20, 36], 0);
          const chained = second.add(1);
          owned.push(chained);
          assertClose(await chained.toArrayAsync(), [7, 15, 11, 19], 0);
        } finally {
          await model.dispose();
          for (const tensor of owned) await tensor.dispose();
        }
      }
      async function checkModelBoundedMinibatches(pg) {
        const n = pg.uop.variable("fit_batch", 1, 8), bound = n.bind(4);
        const owned = [], models = [];
        const build = async () => {
          const w = new pg.Tensor([0], { dtype: "float32" }), x = pg.Tensor.empty([bound, 2]), y = pg.Tensor.empty([bound, 2]);
          owned.push(w, x, y);
          const model = await pg.Model.fromCallableAsync(({ x: x2 }) => x2.mul(w), {
            inputs: { x },
            targets: { y },
            params: { w },
            loss: (out, { y: y2 }) => out.sub(y2).square().mean()
          });
          models.push(model);
          return model;
        };
        try {
          const x = Float32Array.from({ length: 14 }, (_, i) => i / 10), y = x.map((v) => v * 2);
          const tx = new pg.Tensor(
            [Array.from(x.filter((_, i) => i % 2 === 0)), Array.from(x.filter((_, i) => i % 2 === 1))],
            { dtype: "float32" }
          ).transpose(), ty = new pg.Tensor(y).reshape(7, 2);
          owned.push(tx, ty);
          const control = await build();
          control.setOptimizer("sgd", 0.01);
          const expected = [];
          for (let epoch = 0; epoch < 2; epoch++) for (let i = 0; i < 7; i += 3)
            expected.push(await control.trainStepAsync({ x: x.subarray(i * 2, (i + 3) * 2), y: y.subarray(i * 2, (i + 3) * 2) }));
          for (const data of [{ x, y }, { x: tx, y: ty }, { x: tx, y }]) {
            const model2 = await build();
            const losses = await model2.fitAsync(data, { batchSize: 3, remainder: "keep", epochs: 2, optimizer: "sgd", lr: 0.01 });
            assertClose(losses, expected, 1e-5);
            assertClose(await model2.readBufferAsync("w"), await control.readBufferAsync("w"), 1e-5);
            const before = await model2.readBufferAsync("w");
            for (const opts of [{ batchSize: 9, remainder: "keep" }, { batchSize: 3, remainder: "error" }]) {
              let rejected = false;
              try {
                await model2.fitAsync(data, { ...opts, optimizer: "adam" });
              } catch {
                rejected = true;
              }
              assert(rejected, "invalid batch must reject before any update");
              assertClose(await model2.readBufferAsync("w"), before, 0);
            }
          }
          assertClose(await tx.toArrayAsync(), x, 0);
          const model = await build(), small = new pg.Tensor(x.subarray(0, 4)).reshape(2, 2);
          owned.push(small);
          assert((await model.fitAsync({ x: small, y: y.subarray(0, 4) }, { batchSize: 3, remainder: "keep", optimizer: "sgd" })).length === 1);
          assertClose(await small.toArrayAsync(), x.subarray(0, 4), 0);
          const ffi = pg._core.ffi, shrink = ffi.poly_tensor_shrink, release = ffi.poly_tensor_release;
          const train = pg._core.model.trainStep;
          for (const failure of ["shrink", "train"]) {
            let allocated = [], calls = 0, rejected = false;
            const injectedShrink = (...args) => {
              if (++calls === 2 && failure === "shrink") return null;
              const handle = shrink(...args);
              allocated.push(handle);
              return handle;
            };
            const injectedRelease = (handle) => {
              const index = allocated.indexOf(handle);
              if (index >= 0) allocated.splice(index, 1);
              return release(handle);
            };
            pg._core.ffi = Object.create(ffi, {
              poly_tensor_shrink: { value: injectedShrink },
              poly_tensor_release: { value: injectedRelease }
            });
            pg._core.model.trainStep = (...args) => {
              if (failure === "train") throw new Error("train sentinel");
              return train(...args);
            };
            try {
              await model.fitAsync({ x: tx, y: ty }, { batchSize: 3, remainder: "keep" });
            } catch (e) {
              rejected = /shrink failed|train sentinel/.test(e.message);
            } finally {
              pg._core.ffi = ffi;
              pg._core.model.trainStep = train;
            }
            assert(rejected, "injected batch failure must propagate");
            assert(allocated.length === 0, "batch failure leaked Tensor slice owners");
          }
          let failed = false;
          try {
            await model.fitAsync({ x: tx, y: ty }, { batchSize: 3, remainder: "keep", onStep: () => {
              throw new Error("batch callback");
            } });
          } catch (e) {
            failed = /batch callback/.test(e.message);
          }
          assert(failed, "callback exception must clean up Tensor slices");
          const pending = model.fitAsync({ x: tx, y: ty }, { batchSize: 3, remainder: "keep" });
          const releases = [tx.dispose(), ty.dispose(), model.dispose()];
          assert((await pending).length === 3, "queued dataset must survive caller disposal");
          await Promise.all(releases);
        } finally {
          for (const model of models) await model.dispose();
          for (const tensor of owned) await tensor.dispose();
          bound.dispose();
          n.dispose();
        }
      }
      async function checkModelMinibatches(pg) {
        const gpu = String(pg.device).toLowerCase() === "webgpu";
        const build = async () => {
          const w = new pg.Tensor([0], { dtype: "float32" });
          const opts = {
            inputs: { x: pg.Tensor.empty([2, 1]) },
            targets: { y: pg.Tensor.empty([2, 1]) },
            params: { w },
            loss: (out, { y }) => out.sub(y).square().mean()
          };
          return gpu ? pg.Model.fromCallableAsync(({ x }) => x.mul(w), opts) : new pg.Model(({ x }) => x.mul(w), opts);
        };
        const model = await build(), control = await build();
        try {
          const x = new Float32Array([1, 2, 3, 4, 5, 6]), y = new Float32Array([2, 4, 6, 8, 10, 12]);
          const seen = [];
          const losses = await model.fitAsync({ x: { data: x, shape: [6, 1] }, y }, {
            batchSize: 2,
            epochs: 2,
            optimizer: "sgd",
            lr: 0.01,
            onStep: (step) => seen.push(step)
          });
          control.setOptimizer("sgd", 0.01);
          const expected = [];
          for (let epoch = 0; epoch < 2; epoch++) for (let i = 0; i < 6; i += 2)
            expected.push(await control.trainStepAsync({ x: x.subarray(i, i + 2), y: y.subarray(i, i + 2) }));
          assertClose(losses, expected, 1e-5);
          assertClose(await model.readBufferAsync("w"), await control.readBufferAsync("w"), 1e-5);
          assert(JSON.stringify(seen) === "[0,1,2,3,4,5]", "callback step numbers must span epochs");
          const before = await model.readBufferAsync("w");
          let rejected = false;
          try {
            await model.fitAsync({ x: x.subarray(0, 5), y: y.subarray(0, 5) }, { batchSize: 2, optimizer: "adam" });
          } catch (e) {
            rejected = /remainder/.test(e.message);
          }
          assert(rejected, "incomplete batch must reject before any update");
          assertClose(await model.readBufferAsync("w"), before, 0);
          rejected = false;
          try {
            await model.fitAsync({ x: x.subarray(0, 5), y: y.subarray(0, 5) }, { batchSize: 2, remainder: "keep", optimizer: "adam" });
          } catch (e) {
            rejected = /remainder/.test(e.message);
          }
          assert(rejected, "keep cannot admit an extent outside the fixed signature");
          assertClose(await model.readBufferAsync("w"), before, 0);
          for (const bad of [
            { x, y: y.subarray(0, 4) },
            { x, y: new Int32Array(y) },
            { x: { data: x, shape: [3, 2] }, y }
          ]) {
            let invalid = false;
            try {
              await model.fitAsync(bad, { batchSize: 2, optimizer: "adam" });
            } catch {
              invalid = true;
            }
            assert(invalid, "invalid dataset must fail preflight");
            assertClose(await model.readBufferAsync("w"), before, 0);
          }
          const dropped = await model.fitAsync({ x: x.subarray(0, 5), y: y.subarray(0, 5) }, { batchSize: 2, remainder: "drop" });
          assert(dropped.length === 2, "drop must process only explicitly selected complete batches");
          let callbackFailed = false;
          try {
            await model.fitAsync({ x, y }, { batchSize: 2, onStep: () => {
              throw new Error("callback sentinel");
            } });
          } catch (e) {
            callbackFailed = /callback sentinel/.test(e.message);
          }
          assert(callbackFailed, "callback failure must propagate without poisoning the queue");
          const finishing = model.fitAsync({ x, y }, { batchSize: 2 });
          const closing = model.dispose();
          assert((await finishing).length === 3, "admitted fit must survive immediate Model disposal");
          await closing;
        } finally {
          await model.dispose();
          await control.dispose();
        }
      }
      async function checkQuantizedModelWeights(pg) {
        for (const c of quantizedFixture.cases) {
          const bytes = [];
          const u32 = (n) => {
            for (let i = 0; i < 4; i++) bytes.push(n >>> 8 * i & 255);
          };
          const u64 = (n) => {
            u32(n);
            u32(0);
          };
          const str = (s) => {
            const b = new TextEncoder().encode(s);
            u64(b.length);
            bytes.push(...b);
          };
          bytes.push(71, 71, 85, 70);
          u32(3);
          u64(1);
          u64(5);
          str("general.architecture");
          u32(8);
          str("gpt2");
          for (const [key, value] of [["embedding_length", 32], ["attention.head_count", 2], ["block_count", 1], ["context_length", 2]]) {
            str(`gpt2.${key}`);
            u32(4);
            u32(value);
          }
          str("token_embd.weight");
          u32(2);
          u64(32);
          u64(8);
          u32(c.type);
          u64(0);
          while (bytes.length % 32) bytes.push(0);
          const expected = [];
          for (let i = 0; i < 256 / c.values.length; i++) {
            bytes.push(...c.bytes);
            expected.push(...c.values);
          }
          const model = pg.Model.fromGGUF(new Uint8Array(bytes), { maxBatch: 1, maxSeqLen: 2 });
          try {
            assertClose(await model.readBufferAsync("wte.weight"), expected, 0);
          } finally {
            await model.dispose();
          }
        }
      }
      function checkModelCodecRejection(pg) {
        const invalidHeads = new TextEncoder().encode(JSON.stringify({
          model_type: "gpt2",
          n_embd: 4,
          n_head: 0,
          n_layer: 1,
          vocab_size: 8,
          n_positions: 2
        }));
        let invalidRejected = false;
        try {
          pg.Model.fromHF(invalidHeads, []).dispose();
        } catch (e) {
          invalidRejected = /fromHF failed/.test(e.message);
        }
        assert(invalidRejected, "zero attention heads must fail before division");
        const config = new TextEncoder().encode(JSON.stringify({
          model_type: "gpt2",
          n_embd: 8,
          n_head: 2,
          n_layer: 1,
          vocab_size: 4,
          n_positions: 4
        }));
        const badShard = new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 123]);
        let error;
        try {
          pg.Model.fromHF(config, [badShard]).dispose();
        } catch (e) {
          error = e;
        }
        assert(
          error && /failed to decode weight file/.test(error.message),
          "Model import must reject a malformed shard before construction"
        );
        const header = new TextEncoder().encode(JSON.stringify({ "transformer.wte.weight": {
          dtype: "F32",
          shape: [1],
          data_offsets: [0, 4]
        } }));
        const wrongShape = new Uint8Array(8 + header.length + 4);
        new DataView(wrongShape.buffer).setUint32(0, header.length, true);
        wrongShape.set(header, 8);
        error = null;
        try {
          pg.Model.fromHF(config, [wrongShape]).dispose();
        } catch (e) {
          error = e;
        }
        assert(error && /numel mismatch/.test(error.message), "Model import must propagate weight-copy failure");
        const badGguf = new Uint8Array(64);
        badGguf.set([71, 71, 85, 70, 3]);
        badGguf[16] = 1;
        badGguf.fill(255, 24, 32);
        error = null;
        try {
          pg.Model.fromGGUF(badGguf).dispose();
        } catch (e) {
          error = e;
        }
        assert(error && /GGUF/.test(error.message), "Model import must reject an invalid GGUF length");
        for (const [type, numel, nbytes, valid] of [
          [24, 1, 1, true],
          [25, 1, 2, true],
          [26, 1, 4, true],
          [18, 1, 4, false],
          [8, 1, 34, false],
          [8, 32, 34, true]
        ]) {
          const bytes = new Uint8Array(64 + nbytes);
          bytes.set([71, 71, 85, 70, 3]);
          const view = new DataView(bytes.buffer);
          view.setUint32(8, 1, true);
          view.setUint32(24, 1, true);
          bytes[32] = 120;
          view.setUint32(33, 1, true);
          view.setUint32(37, numel, true);
          view.setUint32(45, type, true);
          error = null;
          try {
            pg.Model.fromGGUF(bytes).dispose();
          } catch (e) {
            error = e;
          }
          assert(
            error && (valid ? /unsupported GGUF architecture/ : /invalid or unallocatable GGUF/).test(error.message),
            `GGUF type ${type}, numel ${numel}: wrong decoder admission`
          );
        }
      }
      async function checkCompositionFactories(pg) {
        const { Model, models } = pg;
        const webgpu = String(pg.device).toLowerCase() === "webgpu";
        const spec = JSON.parse(JSON.stringify(compositionFixture));
        delete spec.type;
        delete spec.format;
        assert(!Model.fromDefinition, "construction families must not be Model methods");
        if (webgpu) {
          let rejected = false;
          try {
            models.Graph(spec);
          } catch (err) {
            rejected = err.name === "PolyAsyncRequired";
          }
          assert(rejected, "WebGPU construction requires explicit async admission");
        }
        const model = await models.GraphAsync(spec);
        let restored = null;
        try {
          assert(model.paramCount === 1, "shared calls must have one parameter");
          await model.writeBufferAsync("modules.shared.weight", new Float32Array([1, 2, 3, 4]));
          const io = { x: new Float32Array([1, 2]) };
          assertClose((await model.forward(io)).prediction, [28, 61]);
          model.setOptimizer(pg.OPTIM_SGD, 0.1);
          const loss = await model.trainStepAsync(io);
          assert(Math.abs(loss - 89) < 1e-4, `wrong objective: ${loss}`);
          assertClose(await model.readBufferAsync("modules.shared.weight"), [0.1, 0.1, 1.9, 1.7]);
          const blob = await model.saveBundleAsync({ includeOptimizer: false });
          restored = Model.fromBundle(blob);
          assertClose((await restored.forward(io)).prediction, (await model.forward(io)).prediction);
        } finally {
          if (restored) await restored.dispose();
          await model.dispose();
        }
        const sequential = {
          input: { name: "x", shape: [1, 2], dtype: "float32" },
          layers: [{
            name: "stack",
            type: "repeat",
            count: 2,
            body: { type: "linear", out_features: 2, activation: "relu" }
          }],
          output: "prediction",
          seed: 42
        };
        const a = await models.SequentialAsync(sequential);
        const b = await models.SequentialAsync(JSON.stringify(sequential));
        try {
          assert(a.paramCount === 4, "Repeat must create fresh weights and biases");
          for (const binding of a.bindings().filter((v) => v.trainable)) {
            assertClose(await a.readBufferAsync(binding.name), await b.readBufferAsync(binding.name), 0);
          }
          await a.writeBufferAsync("layers.stack.0.bias", new Float32Array([7, 8]));
          assertClose(await a.readBufferAsync("layers.stack.1.bias"), [0, 0], 0);
          assertClose(await b.readBufferAsync("layers.stack.0.bias"), [0, 0], 0);
          const io = { x: new Float32Array([1, 2]) };
          const result = (await a.forward(io)).prediction;
          assert(result.length === 2 && Array.from(result).every(Number.isFinite), "Repeat output");
        } finally {
          await a.dispose();
          await b.dispose();
        }
        for (const [bad, error] of [
          ['{"type":"graph","type":"sequential"}', /duplicate/],
          [{ ...spec, type: "sequential" }, /type must match/],
          [{ ...spec, format: "poly.modeldef@99" }, /format/],
          [{ ...spec, nodes: [{ name: "bad", type: "add", inputs: ["later", "x"] }] }, /forward value/]
        ]) {
          let caught = null;
          try {
            await models.GraphAsync(bad);
          } catch (err) {
            caught = err;
          }
          assert(caught && error.test(caught.message), `missing validation: ${caught && caught.message}`);
        }
      }
      async function checkCompositionCatalogue(pg) {
        const unary = {
          relu: (x) => Math.max(0, x),
          sigmoid: (x) => 1 / (1 + Math.exp(-x)),
          tanh: Math.tanh,
          silu: (x) => x / (1 + Math.exp(-x)),
          gelu: (x) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3))),
          identity: (x) => x,
          square: (x) => x * x,
          exp: Math.exp,
          log: Math.log
        };
        for (const type of [...Object.keys(unary), "sum", "mean", "reshape"]) {
          const data = type === "log" ? [1, 2] : [-1, 2];
          const layer = { name: "op", type };
          if (type === "reshape") layer.shape = [2, 1];
          const model2 = await pg.models.SequentialAsync({
            input: { name: "x", shape: [1, 2], dtype: "float32" },
            layers: [layer],
            output: "prediction"
          });
          try {
            const expected = type === "sum" ? [1] : type === "mean" ? [0.5] : type === "reshape" ? data : data.map(unary[type]);
            assertClose((await model2.forward({ x: new Float32Array(data) })).prediction, expected, 2e-5);
          } finally {
            await model2.dispose();
          }
        }
        for (const [type, expected] of [
          ["add", [2, 4, 4, 6]],
          ["sub", [0, 0, 2, 2]],
          ["mul", [1, 4, 3, 8]],
          ["div", [1, 1, 3, 2]]
        ]) {
          const model2 = await pg.models.GraphAsync({
            inputs: { x: { shape: [2, 2], dtype: "float32" }, y: { shape: [2], dtype: "float32" } },
            nodes: [{ name: "op", type, inputs: ["x", "y"] }],
            outputs: { prediction: "op" }
          });
          try {
            assertClose((await model2.forward({ x: new Float32Array([1, 2, 3, 4]), y: new Float32Array([1, 2]) })).prediction, expected);
          } finally {
            await model2.dispose();
          }
        }
        const model = await pg.models.GraphAsync({
          inputs: { x: { shape: [2, 1], dtype: "float32" }, y: { shape: [2, 1], dtype: "float32", role: "target" } },
          nodes: [
            { name: "pred", type: "linear", out_features: 1, bias: false, inputs: ["x"] },
            { name: "error", type: "sub", inputs: ["pred", "y"] },
            { name: "sq", type: "square", inputs: ["error"] },
            { name: "avg", type: "mean", inputs: ["sq"] }
          ],
          outputs: { prediction: "pred", cost: "avg" },
          entrypoints: [
            { name: "forward", inputs: ["x"], outputs: ["prediction"] },
            { name: "loss", inputs: ["x", "y"], outputs: ["cost"], objective: "cost" }
          ]
        });
        try {
          const x = new Float32Array([1, 2]);
          await model.writeBufferAsync("nodes.pred.weight", new Float32Array([2]));
          assertClose((await model.forward({ x })).prediction, [2, 4], 0);
          model.setOptimizer(pg.OPTIM_SGD, 0.1);
          assertClose([await model.trainStepAsync({ x, y: x })], [2.5], 1e-6);
          assertClose(await model.readBufferAsync("nodes.pred.weight"), [1.5], 1e-6);
        } finally {
          await model.dispose();
        }
      }
      async function checkTiedAdamCheckpoint(pg) {
        const w = new pg.Tensor([1], { dtype: "float32" });
        const model = await pg.Model.fromTensors({
          params: { w, tied: w },
          losses: { cost: w.add(w).square().sum() }
        });
        let restored = null;
        try {
          await model.placeAsync(pg.device);
          model.setTrainable("tied", false);
          assert(!model.bufTrainable(model.findBuf("w")), "placed alias did not freeze");
          model.setTrainable("w", true);
          assert(model.bufTrainable(model.findBuf("tied")), "placed alias did not unfreeze");
          model.setOptimizer(pg.OPTIM_ADAM, 0.1);
          assertClose([await model.trainStepAsync({})], [4], 0);
          assertClose(await model.readBufferAsync("w"), [0.9], 1e-6);
          const names = ["optim.adam.b1_t", "optim.adam.b2_t", "optim.adam.m.w", "optim.adam.v.w"];
          const weights = await model.exportWeightsAsync();
          const optimizerNames = (bytes) => [...safetensorNames(bytes)].filter((n) => n.startsWith("optim.")).sort().join(",");
          assert(optimizerNames(weights) === [...names].sort().join(","), "duplicated tied optimizer state");
          restored = pg.Model.fromIR(model.exportIR(), weights);
          await restored.placeAsync(pg.device);
          restored.setTrainable("w", false);
          assert(!restored.bufTrainable(restored.findBuf("tied")), "restored alias did not freeze");
          restored.setTrainable("tied", true);
          restored.setOptimizer(pg.OPTIM_ADAM, 0.1);
          const expectedLoss = await model.trainStepAsync({});
          assertClose([await restored.trainStepAsync({})], [expectedLoss], 0);
          for (const name of [...names, "w", "tied"]) {
            assertClose(await restored.readBufferAsync(name), await model.readBufferAsync(name), 0);
          }
          assert(optimizerNames(await restored.exportWeightsAsync()) === [...names].sort().join(","), "restored alias state duplicated");
        } finally {
          if (restored) await restored.dispose();
          await model.dispose();
        }
      }
      function assert(cond, msg) {
        if (!cond) throw new Error(msg || "assertion failed");
      }
      function assertClose(actual, expected, tol = 1e-4) {
        if (actual.length !== expected.length) {
          throw new Error(`length mismatch: ${actual.length} vs ${expected.length}`);
        }
        for (let i = 0; i < actual.length; i++) {
          if (Number.isNaN(expected[i])) {
            if (!Number.isNaN(actual[i])) {
              throw new Error(`mismatch at [${i}]: ${actual[i]} vs ${expected[i]}`);
            }
            continue;
          }
          const diff = Math.abs(actual[i] - expected[i]);
          if (!Number.isFinite(diff) || diff > tol) {
            throw new Error(`mismatch at [${i}]: ${actual[i]} vs ${expected[i]}`);
          }
        }
      }
      function safetensorNames(bytes) {
        const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        const headerLen = Number(view.getBigUint64(0, true));
        const headerBytes = bytes.subarray(8, 8 + headerLen);
        const header = JSON.parse(new TextDecoder().decode(headerBytes));
        return new Set(Object.keys(header).filter((k) => k !== "__metadata__"));
      }
      function testFilterFor(pg) {
        if (pg && pg.testFilter) return String(pg.testFilter);
        if (typeof globalThis !== "undefined" && globalThis.__POLY_TEST_FILTER) {
          return String(globalThis.__POLY_TEST_FILTER);
        }
        if (typeof process !== "undefined" && process.env && process.env.POLY_TEST_FILTER) {
          return String(process.env.POLY_TEST_FILTER);
        }
        return "";
      }
      async function checkTypedIntegerInput(pg, Model) {
        const x = pg.Tensor.empty([3], { dtype: "int32" });
        const outTensor = x.cast("float32");
        const inst = await Model.fromTensors({
          inputs: { typed_x: x },
          outputs: { typed_out: outTensor }
        });
        try {
          const output = await inst.forward({ typed_x: new Int32Array([0, 1, 2]) });
          assertClose(output.typed_out, [0, 1, 2], 0);
          let rejected = false;
          try {
            await inst.forward({ typed_x: new Float32Array([0, 1, 2]) });
          } catch (e) {
            rejected = /call\('forward'\) failed/.test(String(e && e.message));
          }
          assert(rejected, "float32 bytes must not bind to an int32 Model input");
        } finally {
          inst.dispose();
        }
      }
      async function checkCallSignatureAndSelectedOutputs(pg, Model) {
        const x = pg.Tensor.empty([2]);
        const y = pg.Tensor.empty([2]);
        const inst = await Model.fromTensors({
          inputs: { x, y },
          outputs: { plus: x.add(y), minus: x.sub(y) },
          entrypoints: [
            { name: "plus_ep", inputs: ["x", "y"], outputs: ["plus"] },
            { name: "minus_ep", inputs: ["x", "y"], outputs: ["minus"] }
          ]
        });
        const webgpu = String(pg.device).toLowerCase() === "webgpu";
        const call = (entrypoint, io) => webgpu ? inst.callAsync(entrypoint, io) : inst.call(entrypoint, io);
        try {
          const io = {
            x: new Float32Array([5, 7]),
            y: new Float32Array([2, 3])
          };
          const plus = await call("plus_ep", io);
          const minus = await call("minus_ep", io);
          assert(Object.keys(plus).join(",") === "plus", "plus_ep returned undeclared outputs");
          assert(Object.keys(minus).join(",") === "minus", "minus_ep returned undeclared outputs");
          assertClose(plus.plus, [7, 10], 0);
          assertClose(minus.minus, [3, 4], 0);
          let rejected = false;
          try {
            await call("plus_ep", { x: new Float32Array([9, 9]) });
          } catch (err) {
            rejected = /call\('plus_ep'\) failed/.test(String(err && err.message));
          }
          assert(rejected, "missing required input must not reuse stale Model bytes");
        } finally {
          if (webgpu) await inst.dispose();
          else inst.dispose();
        }
        let invalidParamRejected = false;
        try {
          const unexpected = await Model.fromTensors({
            inputs: { x },
            outputs: { output: x.add(1) },
            params: { bad: {} }
          });
          unexpected.dispose();
        } catch (err) {
          invalidParamRejected = /not a Tensor/.test(String(err && err.message));
        }
        assert(invalidParamRejected, "invalid parameter must not be silently filtered");
      }
      async function checkModuleDeviceMap(pg, Model) {
        const Tensor = pg.Tensor;
        const x = Tensor.empty([2]);
        const webgpu = String(pg.device).toLowerCase() === "webgpu";
        const w0 = webgpu ? null : new Tensor([3, 4], { dtype: "float32" });
        const w1 = webgpu ? null : new Tensor([2, 3], { dtype: "float32" });
        const hidden = webgpu ? x.add(3) : x.add(w0);
        const output = webgpu ? hidden.mul(2) : hidden.mul(w1);
        const inst = await Model.fromTensors({
          inputs: { x },
          outputs: { output },
          params: webgpu ? null : { "layers.0.weight": w0, "layers.1.weight": w1 },
          modules: [
            { name: "layers.0", inputs: [x], output: hidden },
            { name: "layers.1", inputs: [hidden], output }
          ]
        });
        const first = String(pg.device).toUpperCase();
        const second = first === "INTERP" ? pg.core === "native" ? "CPU" : "WASM" : "INTERP";
        const place = async (map) => {
          if (webgpu) await inst.setDeviceMapAsync(map);
          else inst.setDeviceMap(map);
        };
        const forward = (input) => webgpu ? inst.forwardAsync(input) : inst.forward(input);
        const expected = webgpu ? [8, 10] : [8, 18];
        const present = (value, label) => {
          assert(value != null, `${label} returned null`);
          return value;
        };
        const assertOptionalBytesEqual = (actual, expectedBytes, label) => {
          assert(actual == null === (expectedBytes == null), `${label} presence changed`);
          if (actual != null) assertClose(actual, expectedBytes, 0);
        };
        try {
          const irBefore = present(inst.exportIR(), "initial IR export");
          const weightsBefore = await inst.exportWeights();
          await place({ "layers.0": first, "layers.1": second });
          let result = await forward({ x: new Float32Array([1, 2]) });
          assertClose(present(result.output, "first placed output"), expected);
          assertClose(present(inst.exportIR(), "first placed IR export"), irBefore, 0);
          assertOptionalBytesEqual(await inst.exportWeights(), weightsBefore, "first placed weight export");
          let rejected = false;
          try {
            await place({ "layers.0": first });
          } catch (err) {
            rejected = /incomplete|device map/.test(String(err.message || err));
          }
          assert(rejected, "expected incomplete device map to fail");
          await place({ "layers.0": second, "layers.1": first });
          result = await forward({ x: new Float32Array([1, 2]) });
          assertClose(present(result.output, "replacement placed output"), expected);
          const irAfter = present(inst.exportIR(), "replacement IR export");
          const weightsAfter = await inst.exportWeights();
          assertClose(irAfter, irBefore, 0);
          assertOptionalBytesEqual(weightsAfter, weightsBefore, "replacement weight export");
          const restored = Model.fromIR(irAfter, weightsAfter);
          try {
            const restoredResult = webgpu ? await restored.forwardAsync({ x: new Float32Array([1, 2]) }) : await restored.forward({ x: new Float32Array([1, 2]) });
            assertClose(present(restoredResult.output, "restored output"), expected);
          } finally {
            if (webgpu) await restored.dispose();
            else restored.dispose();
          }
        } finally {
          if (webgpu) await inst.dispose();
          else inst.dispose();
        }
      }
      async function checkModelStorageObjectives(pg, Model) {
        const w = new pg.Tensor([2], { dtype: "float32" });
        const a = w.mul(w).sum(), b = w.mul(w).mul(w).sum();
        const model = await Model.fromTensors({ params: { w, tied: w }, losses: { a, b }, entrypoints: [
          { name: "a_ep", outputs: ["a"], objective: "a" },
          { name: "b_ep", outputs: ["a", "b"], objective: "b" }
        ] });
        try {
          const copied = await model.readBufferAsync("w");
          copied[0] = 100;
          assertClose(await model.readBufferAsync("w"), [2], 0);
          let rejected = false;
          try {
            await model.writeBufferAsync("w", new Int32Array([3]));
          } catch (_) {
            rejected = true;
          }
          assert(rejected, "write accepted a different dtype");
          const written = new Float32Array([2]);
          const pendingWrite = model.writeBufferAsync("w", written);
          written[0] = 9;
          await pendingWrite;
          assertClose(await model.readBufferAsync("w"), [2], 0);
          model.setTrainable("tied", false);
          model.setTrainable("w", true);
          model.setOptimizer(pg.OPTIM_SGD, 0.1, 0, 0, 0, 0);
          rejected = false;
          try {
            await model.trainStepAsync({});
          } catch (_) {
            rejected = true;
          }
          assert(rejected, "ambiguous objective silently selected");
          assertClose([await model.trainStepAsync({}, "b_ep")], [8], 1e-6);
          assertClose(await model.readBufferAsync("w"), [0.8], 1e-6);
          assertClose(await model.readBufferAsync("b"), [8], 0);
          await model.placeAsync(pg.device);
          assertClose(await model.readBufferAsync("w"), [0.8], 1e-6);
          await model.dispose();
          assertClose(copied, [100], 0);
          rejected = false;
          try {
            await model.readBufferAsync("w");
          } catch (_) {
            rejected = true;
          }
          assert(rejected, "disposed Model accepted a state read");
        } finally {
          await model.dispose();
        }
      }
      async function checkModelTrace(pg, Model) {
        const x = pg.Tensor.empty([1]), y = pg.Tensor.empty([1]);
        const w = new pg.Tensor([1], { dtype: "float32" }).add(1);
        const offset = new pg.Tensor([1], { dtype: "float32" }).is_param_(false);
        let calls = 0;
        const trace = String(pg.device).toLowerCase() === "webgpu" ? Model.fromCallableAsync.bind(Model) : Model.fromCallable.bind(Model);
        const model = await trace(({ x: x2 }) => {
          calls++;
          return x2.mul(w).add(offset);
        }, {
          inputs: { x },
          targets: { y },
          params: { w, offset },
          loss: (out, { y: y2 }) => ({ mse: out.sub(y2).square().mean() })
        });
        try {
          assert(calls === 2, "loss capture must construct evaluation and training forwards");
          model.setOptimizer(pg.OPTIM_SGD, 0.1, 0, 0, 0, 0);
          assertClose([await model.trainStepAsync({
            x: new pg.Tensor([1], { dtype: "float32" }),
            y: new pg.Tensor([0], { dtype: "float32" })
          })], [9], 1e-6);
          assertClose(await model.readBufferAsync("w"), [1.4], 1e-6);
          assertClose(await model.readBufferAsync("offset"), [1], 0);
          assert(model.bindings().some((row) => row.name === "offset" && row.role === 4), "AUX role lost");
          const saved = await model.exportWeightsAsync({ includeOptimizer: false });
          assert(safetensorNames(saved).has("offset"), "model-only export lost persistent AUX");
          assert(calls === 2, "training re-invoked authoring code after train/eval capture");
          const objective = model.entrypoints().find((row) => row.name === "loss");
          assert(objective.objective === "mse" && objective.inputs.join(",") === "x,y", "objective signature lost");
        } finally {
          await model.dispose();
        }
      }
      async function checkModelConstructor(pg, Model) {
        class Net {
          constructor() {
            this.weight = new pg.Tensor([2], { dtype: "float32" });
            this.alias = this.weight;
            this.offset = new pg.Tensor([1], { dtype: "float32" }).is_param_(false);
          }
          forward({ x }) {
            return x.mul(this.weight).add(this.offset);
          }
        }
        const net = new Net();
        const options = {
          inputs: { x: pg.Tensor.empty([1]) },
          targets: { y: pg.Tensor.empty([1]) },
          loss: (out, { y }) => out.sub(y).square().mean()
        };
        const gpu = String(pg.device).toLowerCase() === "webgpu";
        const model = gpu ? await Model.fromCallableAsync(net, options) : new Model(net, options);
        try {
          const roles = Object.fromEntries(model.bindings().map((b) => [b.name, b.role]));
          assert(roles.weight === 0 && roles.alias === 0 && roles.offset === 4, "constructor state roles");
          model.setOptimizer("sgd", 0.1);
          assertClose([await model.trainStepAsync({ x: [1], y: [0] })], [9], 1e-6);
          assertClose(await model.readBufferAsync("weight"), [1.4], 1e-6);
          assertClose(await net.weight.toArrayAsync(), [2], 0);
          const bytes = gpu ? await model.saveAsync({ includeOptimizer: false }) : model.save({ includeOptimizer: false });
          assert(bytes instanceof Uint8Array && bytes.length > 0, "bundle save");
        } finally {
          await model.dispose();
        }
      }
      async function checkModelDispatch(pg, Model) {
        const gpu = String(pg.device).toLowerCase() === "webgpu";
        for (const config of [compositionFixture, {
          format: "poly.modeldef@1",
          type: "sequential",
          seed: 42,
          input: { name: "x", shape: [1, 2], dtype: "float32" },
          layers: [{ name: "dense", type: "linear", out_features: 2 }],
          output: "prediction"
        }]) {
          const factory = config.type === "graph" ? "Graph" : "Sequential";
          let automatic, explicit;
          try {
            if (gpu) {
              let rejected2 = false;
              try {
                new Model(config);
              } catch (e) {
                rejected2 = /Async/.test(e.message);
              }
              assert(rejected2, "WebGPU constructor must identify the explicit async factory");
              automatic = await pg.models[factory + "Async"](config);
            } else automatic = new Model(config);
            explicit = await pg.models[factory + (gpu ? "Async" : "")](config);
            assert(JSON.stringify(automatic.bindings()) === JSON.stringify(explicit.bindings()), "configuration bindings differ");
            const a = await automatic.forwardAsync({ x: new Float32Array([1, 2]) });
            const b = await explicit.forwardAsync({ x: new Float32Array([1, 2]) });
            for (const key of Object.keys(a)) assertClose(a[key], b[key], 0);
          } finally {
            if (automatic) await automatic.dispose();
            if (explicit) await explicit.dispose();
          }
        }
        const x = pg.Tensor.empty([1]);
        const source = { forward: ({ x: x2 }) => x2.add(99), selected: ({ x: x2 }) => x2.mul(2) };
        const model = gpu ? await Model.fromCallableAsync(source.selected, { inputs: { x }, params: {} }) : Model.fromCallable(source.selected, { inputs: { x }, params: {} });
        try {
          assertClose((await model.forwardAsync({ x: [3] })).output, [6], 0);
        } finally {
          await model.dispose();
        }
        for (const invalid of [
          { layers: [1, 2] },
          { nodes: [] },
          { format: "poly.modeldef@2", type: "graph" },
          { format: "poly.modeldef@1", type: "unknown" }
        ]) {
          let rejected2 = false;
          try {
            new Model(invalid);
          } catch (_) {
            rejected2 = true;
          }
          assert(rejected2, "constructor guessed an untagged/unknown configuration");
        }
        let rejected = false;
        try {
          new Model(compositionFixture, { outputs: x });
        } catch (e) {
          rejected = /combined/.test(e.message);
        }
        assert(rejected, "configuration accepted conflicting Tensor bindings");
        class LazyNet {
          forward({ x: x2 }) {
            this.weight = new pg.Tensor([2], { dtype: "float32" });
            return x2.mul(this.weight);
          }
        }
        const lazy = gpu ? await Model.fromCallableAsync(new LazyNet(), { inputs: { x } }) : new Model(new LazyNet(), { inputs: { x } });
        try {
          assert(lazy.bindings().some((b) => b.name === "weight" && b.role === 0), "lazy parameter not collected");
        } finally {
          await lazy.dispose();
        }
      }
      async function checkModelUsability(pg, Model) {
        const x = pg.Tensor.empty([1]);
        const gpu = String(pg.device).toLowerCase() === "webgpu";
        const create = (fn, options) => gpu ? Model.fromCallableAsync(fn, options) : Model.fromCallable(fn, options);
        const model = await create(({ x: x2 }) => ({ prediction: x2.mul(2) }), { inputs: { x } });
        try {
          const text = model.summary();
          assert(text.includes("prediction") && text.includes("float32[1]") && text.includes("forward"), "missing model metadata");
          if (gpu) {
            const bytes = await model.saveAsync();
            const running = model.forwardAsync({ x: [3] });
            let rejected2 = false;
            try {
              Model.load(bytes);
            } catch (e) {
              rejected2 = /async/.test(e.message);
            }
            await running;
            assert(rejected2, "bundle load entered suspended Wasm");
          }
          let called = false;
          for (const [fn, loss] of [
            [async ({ x: x2 }) => {
              called = true;
              return x2;
            }, void 0],
            [({ x: x2 }) => {
              called = true;
              return x2;
            }, async (out) => out.mean()]
          ]) {
            let rejected2 = false;
            try {
              await create(fn, { inputs: { x }, loss });
            } catch (e) {
              rejected2 = /synchronous/.test(e.message);
            }
            assert(rejected2 && !called, "async author/loss was invoked");
          }
          const policy = x.logicalPolicy;
          let rejected = false;
          try {
            await create(() => {
              throw new Error("author failed");
            }, { inputs: { x } });
          } catch (e) {
            rejected = /author failed/.test(e.message);
          }
          assert(rejected && pg.Tensor.empty([1]).logicalPolicy === policy, "failed capture changed logical policy");
          if (typeof process === "undefined" || !process.versions || !process.versions.node) {
            rejected = false;
            try {
              model.save("model.pgb");
            } catch (e) {
              rejected = /Node/.test(e.message);
            }
            assert(rejected, "browser save accepted a filesystem path");
          }
        } finally {
          await model.dispose();
        }
      }
      async function checkRuntimeImports(pg, Model, createRuntime) {
        const gpu = String(pg.device).toLowerCase() === "webgpu";
        const x = pg.Tensor.empty([1]), w = new pg.Tensor([2], { dtype: "float32" });
        const live = new pg.Tensor([19], { dtype: "float32" });
        const options = { inputs: { x }, params: { w, alias: w }, entrypoints: [
          { name: "forward", inputs: ["x"], outputs: ["output"] },
          { name: "double", inputs: ["x"], outputs: ["twice"] }
        ] };
        const author = ({ x: x2 }) => ({ output: x2.mul(w), twice: x2.mul(w).mul(2) });
        const source = gpu ? await Model.fromCallableAsync(author, options) : Model.fromCallable(author, options);
        const models = [source];
        const read = (m) => gpu ? m.readBufferAsync("w") : m.readBuffer("w");
        const forward = (m) => gpu ? m.forwardAsync({ x: [3] }) : m.forward({ x: [3] });
        try {
          const bytes = gpu ? await source.saveAsync() : source.save();
          const a = Model.load(bytes), b = Model.fromBundle(bytes);
          models.push(a, b);
          if (gpu) await a.writeBufferAsync("alias", new Float32Array([7]));
          else a.writeBuffer("alias", new Float32Array([7]));
          assertClose(await read(a), [7], 0);
          assertClose((await forward(b)).output, [6], 0);
          assertClose((await forward(source)).output, [6], 0);
          const c = Model.load(await b.saveAsync());
          models.push(c);
          assertClose((await c.callAsync("double", { x: [3] })).twice, [12], 0);
          await b.writeBufferAsync("alias", new Float32Array([11]));
          assertClose((await c.callAsync("double", { x: [3] })).twice, [12], 0);
          assertClose((await c.forwardAsync({ x: [3] })).output, [6], 0);
          for (const n of [0, 31, Math.floor(bytes.length / 2), bytes.length - 1]) {
            let rejected = false;
            try {
              Model.load(bytes.slice(0, n));
            } catch {
              rejected = true;
            }
            assert(rejected, "truncated import accepted");
          }
          await a.dispose();
          pg.collect();
          assertClose((await forward(b)).output, [33], 0);
          assertClose((await c.callAsync("double", { x: [3] })).twice, [12], 0);
          assertClose(gpu ? await live.toArrayAsync() : live.toArray(), [19], 0);
          const runtime = await createRuntime();
          const held = new runtime.Tensor([19], { dtype: "float32" });
          try {
            await held.realizeAsync();
            runtime.clearScheduleCache();
            runtime.collect();
            const fields = ["bufferOwnedBytes", "bufferEntries", "tensorRecords"];
            const baseline = runtime.stats().coreStats;
            assert(baseline.tensorRecords === 1, "baseline must contain only the held Tensor");
            for (let i = 0; i < 8; i++) {
              const loaded = runtime.Model.load(bytes);
              try {
                assertClose((await loaded.forwardAsync({ x: [3] })).output, [6], 0);
                await loaded.writeBufferAsync("alias", new Float32Array([7]));
                assertClose(await loaded.readBufferAsync("w"), [7], 0);
              } finally {
                await loaded.dispose();
              }
              runtime.clearScheduleCache();
              runtime.collect();
              const current = runtime.stats().coreStats;
              for (const field of fields)
                assert(
                  current[field] === baseline[field],
                  `import/dispose retained ${field}: ${baseline[field]} -> ${current[field]}`
                );
            }
            assertClose(await held.toArrayAsync(), [19], 0);
          } finally {
            held.dispose();
            await runtime.dispose();
          }
        } finally {
          for (const model of models) await model.dispose();
          x.dispose();
          w.dispose();
          live.dispose();
        }
      }
      async function checkFamilyRuntimeOwnership(pg) {
        const gpu = String(pg.device).toLowerCase() === "webgpu";
        const live = new pg.Tensor([19], { dtype: "float32" });
        const specs = [
          ["MLP", { layers: [2, 3, 1] }],
          ["TabM", { layers: [2, 3, 1], n_ensemble: 2 }],
          ["NAM", { n_features: 2, hidden_sizes: [3], n_outputs: 1 }]
        ];
        try {
          for (const [name, spec] of specs) {
            const beforeStats = pg.stats().coreStats;
            const a = pg.models[name](spec), b = pg.models[name](spec);
            try {
              assert(
                pg.stats().coreStats.tensorRecords === beforeStats.tensorRecords,
                "family retained construction Tensor wrappers"
              );
              const parameter = a.paramName(0);
              const before = await b.readBufferAsync(parameter);
              await a.writeBufferAsync(parameter, new Float32Array(before.length).fill(7));
              assertClose(await b.readBufferAsync(parameter), before, 0);
              const expected = await b.forwardAsync({ x: [1, 2] });
              const input = new pg.Tensor([[1, 2]], { dtype: "float32" });
              let tensorOutputs;
              try {
                tensorOutputs = await b.forwardAsync({ x: input });
                assertClose(await tensorOutputs.output.toArrayAsync(), expected.output, 0);
              } finally {
                if (tensorOutputs) for (const output of Object.values(tensorOutputs)) output.dispose();
                input.dispose();
              }
              await a.dispose();
              pg.collect();
              assertClose((await b.forwardAsync({ x: [1, 2] })).output, expected.output, 0);
              assertClose(gpu ? await live.toArrayAsync() : live.toArray(), [19], 0);
              pg._activeAsync++;
              try {
                let rejected = false;
                try {
                  pg.models[name](spec);
                } catch (e) {
                  rejected = /idle/.test(e.message);
                }
                assert(rejected, "factory entered a busy Runtime");
              } finally {
                pg._activeAsync--;
              }
            } finally {
              await a.dispose();
              await b.dispose();
            }
          }
        } finally {
          live.dispose();
        }
      }
      async function runModelRuntimeTests(pg, createRuntime) {
        const Model = pg.Model;
        const { MLP, TabM, NAM } = pg.models;
        const testFilter = testFilterFor(pg);
        let passed = 0;
        let failed = 0;
        if (!pg.supportsModel) {
          console.log("\n== Model ==");
          console.log("  [SKIP] core does not expose PolyModel runtime yet");
          return { passed: 0, failed: 0 };
        }
        async function test(name, fn) {
          if (testFilter && !name.includes(testFilter)) return;
          try {
            await fn();
            console.log(`  [PASS] ${name}`);
            passed++;
          } catch (e) {
            console.log(`  [FAIL] ${name}: ${e.message}`);
            failed++;
          }
        }
        console.log("\n== Model ==");
        await test("Model checkpoint replacement uses queued readback", () => checkModelCheckpointReplacement(pg));
        await test("Model stateful capture shares train eval state", () => checkModelStatefulCapture(pg));
        await test("Model stateful capture owns resumable RNG", () => checkModelCaptureRng(pg));
        await test("Model stateful capture restores failures and rejects async execution", () => checkModelCaptureFailure(pg));
        await test("Model copied storage exact writes and objective selection", () => checkModelStorageObjectives(pg, Model));
        await test("Model codecs reject malformed bytes before publication", () => checkModelCodecRejection(pg));
        await test("Model quantized weights match pinned GGUF bit planes", () => checkQuantizedModelWeights(pg));
        await test("Model composition factories share C construction", () => checkCompositionFactories(pg));
        await test("Model composition catalogue and named target objective", () => checkCompositionCatalogue(pg));
        await test("Model tied Adam placement freeze and checkpoint", () => checkTiedAdamCheckpoint(pg));
        await test("Model constructor collects object state", () => checkModelConstructor(pg, Model));
        await test("Model constructor dispatch and explicit factories", () => checkModelDispatch(pg, Model));
        await test("Model usability summary and capture failures", () => checkModelUsability(pg, Model));
        await test("Model runtime imports isolation and failure", () => checkRuntimeImports(pg, Model, createRuntime));
        await test("Model family runtime ownership", () => checkFamilyRuntimeOwnership(pg));
        await test("Llama family reference and shared import", () => checkLlamaFamily(pg));
        await test("Model Tensor I/O owns device results", () => checkModelTensorIO(pg));
        await test("Model variable shapes preserve results and portable signatures", () => checkModelVariableShapes(pg));
        await test("Model empty bindings reject before input writes", () => checkModelEmptyInputAdmission(pg));
        await test("Model minibatches match explicit training steps", () => checkModelMinibatches(pg));
        await test("Model bounded minibatches and Tensor datasets", () => checkModelBoundedMinibatches(pg));
        await test("Model trace seals independent state with named loss", () => checkModelTrace(pg, Model));
        await test("generic call validates signature and returns selected outputs", async () => {
          await checkCallSignatureAndSelectedOutputs(pg, Model);
        });
        await test("scalar rank8 and shared multi-output round trip", async () => {
          const scalarX = pg.Tensor.empty([]);
          const scalarW = pg.Tensor.full([], 3, { dtype: "float32" });
          const scalar = await Model.fromTensors({
            inputs: { x: scalarX },
            outputs: { output: scalarX.mul(scalarW) },
            params: { w: scalarW }
          });
          let scalarRestored = null;
          try {
            scalarRestored = Model.fromIR(scalar.exportIR(), await scalar.exportWeights());
            const result = await scalarRestored.forward({ x: new Float32Array([2]) });
            assertClose(result.output, [6], 0);
            assert(
              JSON.stringify(scalarRestored.bufShape(scalarRestored.findBuf("output"))) === "[]",
              "scalar output shape must remain []"
            );
          } finally {
            if (scalarRestored) scalarRestored.dispose();
            scalar.dispose();
          }
          const shape = [1, 1, 1, 1, 1, 1, 1, 1];
          const x = pg.Tensor.empty(shape);
          const w = pg.Tensor.ones(shape, {});
          const shared = x.add(w);
          const source = await Model.fromTensors({
            inputs: { x },
            outputs: { plus: shared.add(1), minus: shared.sub(1) },
            params: { w }
          });
          let restored = null;
          try {
            restored = Model.fromIR(source.exportIR(), await source.exportWeights());
            const result = await restored.forward({ x: new Float32Array([2]) });
            assertClose(result.plus, [4], 0);
            assertClose(result.minus, [2], 0);
            assert(
              JSON.stringify(restored.bufShape(restored.findBuf("plus"))) === JSON.stringify(shape),
              "rank-8 output shape was not preserved"
            );
          } finally {
            if (restored) restored.dispose();
            source.dispose();
          }
        });
        await test("duplicate ABI storage alias fails closed like TinyJit", async () => {
          const x = pg.Tensor.empty([2]);
          let error = null;
          try {
            const unexpected = await Model.fromTensors({
              inputs: { a: x, b: x },
              outputs: { output: x.add(x) }
            });
            unexpected.dispose();
          } catch (err) {
            error = err;
          }
          assert(error, "duplicate ABI storage must be rejected");
        });
        await test("dynamic input alias with persistent state fails closed", async () => {
          const x = pg.Tensor.empty([2]);
          let error = null;
          try {
            const unexpected = await Model.fromTensors({
              inputs: { x },
              outputs: { output: x.add(x) },
              params: { w: x }
            });
            unexpected.dispose();
          } catch (err) {
            error = err;
          }
          assert(error, "dynamic input must not alias persistent state");
        });
        await test("output alias of dynamic input round trips", async () => {
          const x = pg.Tensor.empty([2]);
          const source = await Model.fromTensors({ inputs: { x }, outputs: { output: x } });
          let restored = null;
          try {
            restored = Model.fromIR(source.exportIR());
            const value = new Float32Array([3, 4]);
            assertClose((await source.forward({ x: value })).output, value, 0);
            assertClose((await restored.forward({ x: value })).output, value, 0);
          } finally {
            if (restored) restored.dispose();
            source.dispose();
          }
        });
        await test("named partial view state fails closed", async () => {
          const base = new pg.Tensor([1, 2, 3, 4], { dtype: "float32" });
          const view = base.shrink([[1, 3]]);
          const x = pg.Tensor.empty([2]);
          let error = null;
          try {
            const unexpected = await Model.fromTensors({
              inputs: { x },
              outputs: { output: x.add(view) },
              params: { base, view }
            });
            unexpected.dispose();
          } catch (err) {
            error = err;
          }
          assert(error, "named partial view storage must be rejected");
        });
        await test("input-dependent named state effect fails closed", async () => {
          const x = pg.Tensor.empty([1]);
          const w = new pg.Tensor([1], { dtype: "float32" });
          const output = w.assign(w.add(x));
          let error = null;
          try {
            const unexpected = await Model.fromTensors({
              inputs: { x },
              outputs: { output },
              params: { w }
            });
            unexpected.dispose();
          } catch (err) {
            error = err;
          }
          assert(error, "input-dependent named state effect must be rejected");
        });
        await test("stochastic output requires named RNG state", async () => {
          pg.Tensor.manual_seed(123);
          const x = pg.Tensor.empty([2]);
          let error = null;
          try {
            const unexpected = await Model.fromTensors({
              inputs: { x },
              outputs: { output: x.add(pg.Tensor.rand(2)) }
            });
            unexpected.dispose();
          } catch (err) {
            error = err;
          }
          assert(error, "stochastic output without named RNG state must be rejected");
        });
        await test("state traversal preserves diamond aliases and stops cycles", async () => {
          const shared = new pg.Tensor([1, 2], { dtype: "float32" });
          const root = { left: { weight: shared }, right: { weight: shared } };
          root.self = root;
          const state = pg.nn.getStateDict(root);
          assert(
            JSON.stringify(Object.keys(state)) === JSON.stringify(["left.weight", "right.weight"]),
            `unexpected state paths: ${Object.keys(state)}`
          );
          assert(state["left.weight"] === state["right.weight"], "alias paths must retain one Tensor");
          const params = pg.nn.getParameters(root);
          assert(params.length === 2, `unexpected parameter count: ${params.length}`);
          assert(params[0] === shared && params[1] === shared, "parameter aliases must match state paths");
        });
        await test("float16 Model state preserves exact storage bits", async () => {
          const w = new pg.Tensor([1.5, -2], { dtype: "float16" });
          const x = pg.Tensor.empty([2], { dtype: "float16" });
          const inst = await Model.fromTensors({
            inputs: { x },
            outputs: { output: x.add(w) },
            params: { w }
          });
          try {
            assert(inst.paramDtype(0) === "float16", `unexpected dtype ${inst.paramDtype(0)}`);
            const raw = await inst.paramData(0);
            assert(raw instanceof Uint16Array, `expected Uint16Array, got ${raw.constructor.name}`);
            assert(
              raw.length === 2 && raw[0] === 15872 && raw[1] === 49152,
              `unexpected float16 bits: ${Array.from(raw)}`
            );
            const restored = Model.fromIR(inst.exportIR(), await inst.exportWeights());
            try {
              assert(restored.paramDtype(0) === "float16", "restored dtype must remain float16");
              const restoredRaw = await restored.paramData(0);
              assert(
                restoredRaw instanceof Uint16Array,
                "restored float16 state must remain raw Uint16Array"
              );
              assert(
                restoredRaw[0] === 15872 && restoredRaw[1] === 49152,
                `unexpected restored bits: ${Array.from(restoredRaw)}`
              );
            } finally {
              restored.dispose();
            }
          } finally {
            inst.dispose();
          }
        });
        await test("typed Model state round trips exact storage bytes", async () => {
          const cases = [
            ["float64", [1.25, -2.5], Float64Array],
            ["int32", [1, -2], Int32Array],
            ["uint8", [1, 255], Uint8Array],
            ["bool", [true, false], Uint8Array],
            ["bfloat16", [1.5, -2], Uint16Array]
          ];
          for (const [dtype, values, ArrayType] of cases) {
            const w = new pg.Tensor(values, { dtype });
            const x = pg.Tensor.empty([2], { dtype });
            const source = await Model.fromTensors({
              inputs: { x },
              outputs: { output: x },
              params: { w }
            });
            try {
              const restored = Model.fromIR(source.exportIR(), await source.exportWeights());
              try {
                const before = await source.paramData(0);
                const after = await restored.paramData(0);
                assert(
                  before instanceof ArrayType,
                  `${dtype}: expected ${ArrayType.name}, got ${before.constructor.name}`
                );
                assert(
                  after instanceof ArrayType,
                  `${dtype}: restored ${after.constructor.name}`
                );
                const beforeBytes = new Uint8Array(before.buffer, before.byteOffset, before.byteLength);
                const afterBytes = new Uint8Array(after.buffer, after.byteOffset, after.byteLength);
                assertClose(afterBytes, beforeBytes, 0);
              } finally {
                restored.dispose();
              }
            } finally {
              source.dispose();
            }
          }
        });
        await test("typed integer input preserves bytes and rejects float binding", async () => {
          await checkTypedIntegerInput(pg, Model);
        });
        await test("module device map places exact Tensor cuts atomically", async () => {
          await checkModuleDeviceMap(pg, Model);
        });
        await test("model-family constructors are not Model methods", async () => {
          assert(typeof Model.mlp === "undefined", "Model.mlp should not exist");
          assert(typeof MLP === "function", "pg.models.MLP should exist");
        });
        await test("mlp create + param enumeration", async () => {
          const inst = MLP({
            layers: [2, 4, 1],
            activation: "relu",
            bias: true,
            loss: "mse",
            batch_size: 1,
            seed: 42
          });
          try {
            assert(inst.paramCount === 4, `expected 4 params, got ${inst.paramCount}`);
            assert(inst.paramName(0) === "layers.0.weight", "unexpected first param name");
            assert(JSON.stringify(inst.paramShape(0)) === JSON.stringify([4, 2]), "unexpected first param shape");
          } finally {
            inst.dispose();
          }
        });
        await test("param trainability freezes optimizer updates", async () => {
          const inst = MLP({
            layers: [2, 1],
            activation: "none",
            bias: true,
            loss: "mse",
            batch_size: 1,
            seed: 42
          });
          try {
            assert(inst.paramTrainable(0) === true, "weight should default trainable");
            assert(inst.paramTrainable(1) === true, "bias should default trainable");
            inst.setParamTrainable(0, false);
            assert(inst.paramTrainable(0) === false, "weight should be frozen");
            const weightBefore = await inst.paramData(0);
            const biasBefore = await inst.paramData(1);
            inst.setOptimizer(pg.OPTIM_SGD, 0.05);
            const x = new Float32Array([1, 2]);
            const y = new Float32Array([5]);
            for (let step = 0; step < 10; step++) {
              const loss = await inst.trainStep({ x, y });
              assert(Number.isFinite(loss), `loss should be finite, got ${loss}`);
            }
            assertClose(await inst.paramData(0), weightBefore);
            let biasChanged = false;
            const biasAfter = await inst.paramData(1);
            for (let i = 0; i < biasAfter.length; i++) {
              if (biasAfter[i] !== biasBefore[i]) biasChanged = true;
            }
            assert(biasChanged, "unfrozen bias should update");
          } finally {
            inst.dispose();
          }
        });
        await test("param trainability survives IR round trip", async () => {
          const inst1 = MLP({
            layers: [2, 1],
            activation: "none",
            bias: true,
            loss: "mse",
            batch_size: 1,
            seed: 42
          });
          try {
            inst1.setParamTrainable(0, false);
            const inst2 = Model.fromIR(inst1.exportIR(), await inst1.exportWeights());
            try {
              assert(inst2.paramTrainable(0) === false, "frozen flag should round trip");
              assert(inst2.paramTrainable(1) === true, "unfrozen flag should round trip");
            } finally {
              inst2.dispose();
            }
          } finally {
            inst1.dispose();
          }
        });
        await test("mlp forward produces output", async () => {
          const inst = MLP({
            layers: [2, 4, 1],
            activation: "relu",
            bias: true,
            loss: "mse",
            batch_size: 1,
            seed: 42
          });
          try {
            const outputs = await inst.forward({ x: new Float32Array([1, 2]) });
            assert(outputs.output instanceof Float32Array, "output should be Float32Array");
            assert(outputs.output.length === 1, `expected output length 1, got ${outputs.output.length}`);
            assert(Number.isFinite(outputs.output[0]), "output should be finite");
          } finally {
            inst.dispose();
          }
        });
        await test("mlp train step decreases loss", async () => {
          const inst = MLP({
            layers: [2, 1],
            activation: "none",
            bias: true,
            loss: "mse",
            batch_size: 1,
            seed: 42
          });
          try {
            inst.setOptimizer(pg.OPTIM_SGD, 0.05);
            const x = new Float32Array([1, 2]);
            const y = new Float32Array([5]);
            let first = null;
            let last = null;
            for (let step = 0; step < 50; step++) {
              last = await inst.trainStep({ x, y });
              if (first == null) first = last;
            }
            assert(last < first, `expected loss to decrease (${first} -> ${last})`);
          } finally {
            inst.dispose();
          }
        });
        await test("weights export/import round trip", async () => {
          const spec = {
            layers: [2, 4, 1],
            activation: "relu",
            bias: true,
            loss: "mse",
            batch_size: 1,
            seed: 42
          };
          const inst1 = MLP(spec);
          const inst2 = MLP({ ...spec, seed: 99 });
          try {
            const original = await inst1.paramData(0);
            const different = await inst2.paramData(0);
            let anyDiff = false;
            for (let i = 0; i < original.length; i++) {
              if (original[i] !== different[i]) {
                anyDiff = true;
                break;
              }
            }
            assert(anyDiff, "different seed should change weights");
            const weights = await inst1.exportWeights();
            assert(weights instanceof Uint8Array && weights.length > 0, "expected non-empty weights export");
            inst2.importWeights(weights);
            assertClose(await inst2.paramData(0), original);
          } finally {
            inst1.dispose();
            inst2.dispose();
          }
        });
        await test("Adam checkpoint resumes the uninterrupted training trajectory", async () => {
          const spec = {
            layers: [2, 1],
            activation: "none",
            bias: false,
            loss: "mse",
            batch_size: 1,
            seed: 7
          };
          const source = MLP(spec);
          let restored = null;
          try {
            source.setOptimizer(pg.OPTIM_ADAM, 0.05);
            const io = { x: new Float32Array([1, 2]), y: new Float32Array([3]) };
            for (let i = 0; i < 3; i++) await source.trainStep(io);
            restored = Model.fromIR(source.exportIR(), await source.exportWeights());
            restored.setOptimizer(pg.OPTIM_ADAM, 0.05);
            const sourceLoss = await source.trainStep(io);
            const restoredLoss = await restored.trainStep(io);
            assert(
              sourceLoss === restoredLoss,
              `restored loss ${restoredLoss} != uninterrupted ${sourceLoss}`
            );
            assertClose(await restored.paramData(0), await source.paramData(0), 0);
            for (const name of [
              "optim.adam.b1_t",
              "optim.adam.b2_t",
              "optim.adam.m.layers.0.weight",
              "optim.adam.v.layers.0.weight"
            ]) {
              const sourceIndex = source.findBuf(name);
              const restoredIndex = restored.findBuf(name);
              assert(sourceIndex >= 0 && restoredIndex >= 0, `missing ${name}`);
              assertClose(
                await restored.bufData(restoredIndex),
                await source.bufData(sourceIndex),
                0
              );
            }
          } finally {
            if (restored) restored.dispose();
            source.dispose();
          }
        });
        await test("stochastic named state requires checkpoint for portable activation", async () => {
          pg.Tensor.manual_seed(11);
          const w = pg.Tensor.rand(2, {});
          const x = pg.Tensor.empty([2]);
          const source = await Model.fromTensors({
            inputs: { x },
            outputs: { output: x.mul(w) },
            params: { w }
          });
          let restored = null;
          try {
            const ir = source.exportIR();
            const weights = await source.exportWeights();
            let rejected = false;
            try {
              const unexpected = Model.fromIR(ir);
              unexpected.dispose();
            } catch (err) {
              rejected = true;
            }
            assert(rejected, "fresh stochastic activation must fail without checkpoint bytes");
            restored = Model.fromIR(ir, weights);
            const input = new Float32Array([2, 3]);
            const sourceOut = await source.forward({ x: input });
            const restoredOut = await restored.forward({ x: input });
            assertClose(restoredOut.output, sourceOut.output, 0);
          } finally {
            if (restored) restored.dispose();
            source.dispose();
          }
        });
        await test("ir export/fromIR round trip", async () => {
          const inst1 = MLP({
            layers: [2, 4, 1],
            activation: "relu",
            bias: true,
            loss: "mse",
            batch_size: 1,
            seed: 42
          });
          try {
            const ir = inst1.exportIR();
            const weights = await inst1.exportWeights();
            const inst2 = Model.fromIR(ir, weights);
            try {
              const out1 = (await inst1.forward({ x: new Float32Array([1, 2]) })).output;
              const out2 = (await inst2.forward({ x: new Float32Array([1, 2]) })).output;
              assertClose(out2, out1);
            } finally {
              inst2.dispose();
            }
          } finally {
            inst1.dispose();
          }
        });
        await test("bound program export uses separate weights and has no portable IR", async () => {
          const inst1 = MLP({
            layers: [2, 4, 1],
            activation: "relu",
            bias: true,
            loss: "none",
            batch_size: 1,
            seed: 42
          });
          let inst2 = null;
          try {
            const program = await inst1.exportProgramAsync();
            const weights = await inst1.exportWeights();
            assert(
              program && new TextDecoder().decode(program.subarray(0, 4)) === "PGPM",
              "compiled program magic mismatch"
            );
            let missingWeightsRejected = false;
            try {
              const unexpected = Model.fromProgram(program);
              await unexpected.dispose();
            } catch (err) {
              missingWeightsRejected = true;
            }
            assert(missingWeightsRejected, "bound state must require separate weights");
            inst2 = Model.fromProgram(program, weights);
            const input = new Float32Array([1.25, -0.5]);
            const out1 = (await inst1.forward({ x: input })).output;
            const out2 = (await inst2.forward({ x: input })).output;
            assertClose(out2, out1, 0);
            const portable = inst2.exportIR();
            assert(!portable || portable.length === 0, "program-only Model exposed portable IR");
            const program2 = await inst2.exportProgramAsync();
            assertClose(program2, program, 0);
          } finally {
            if (inst2) await inst2.dispose();
            await inst1.dispose();
          }
        });
        await test("mlp batch_size=32 forward produces correct shape", async () => {
          const inst = MLP({
            layers: [4, 8, 3],
            activation: "relu",
            bias: true,
            loss: "cross_entropy",
            batch_size: 32,
            seed: 42
          });
          try {
            const x = new Float32Array(32 * 4);
            for (let i = 0; i < x.length; i++) x[i] = Math.random();
            const outputs = await inst.forward({ x });
            assert(outputs.output instanceof Float32Array, "output should be Float32Array");
            assert(
              outputs.output.length === 32 * 3,
              `expected output length ${32 * 3}, got ${outputs.output.length}`
            );
            for (let i = 0; i < outputs.output.length; i++) {
              assert(
                Number.isFinite(outputs.output[i]),
                `output[${i}] should be finite, got ${outputs.output[i]}`
              );
            }
          } finally {
            inst.dispose();
          }
        });
        await test("mlp batch_size=32 train step decreases loss (P0)", async () => {
          const inst = MLP({
            layers: [4, 8, 3],
            activation: "relu",
            bias: true,
            loss: "cross_entropy",
            batch_size: 32,
            seed: 42
          });
          try {
            inst.setOptimizer(pg.OPTIM_SGD, 0.01);
            const x = new Float32Array(32 * 4);
            const y = new Float32Array(32 * 3);
            for (let i = 0; i < x.length; i++) x[i] = i % 7 * 0.1;
            for (let i = 0; i < 32; i++) y[i * 3 + i % 3] = 1;
            let first = null;
            let last = null;
            for (let step = 0; step < 30; step++) {
              last = await inst.trainStep({ x, y });
              if (first == null) first = last;
            }
            assert(Number.isFinite(first), `first loss should be finite, got ${first}`);
            assert(Number.isFinite(last), `last loss should be finite, got ${last}`);
            assert(last < first, `expected loss to decrease (${first} -> ${last})`);
          } finally {
            inst.dispose();
          }
        });
        await test("tabm and nam builders are available", async () => {
          const tabm = TabM({
            layers: [2, 4, 1],
            activation: "relu",
            loss: "mse",
            batch_size: 1,
            seed: 42,
            n_ensemble: 4
          });
          const nam = NAM({
            n_features: 2,
            hidden_sizes: [4],
            activation: "relu",
            n_outputs: 1,
            loss: "mse",
            batch_size: 1,
            seed: 42
          });
          try {
            const tabmOut = await tabm.forward({ x: new Float32Array([1, 2]) });
            const namOut = await nam.forward({ x: new Float32Array([1, 2]) });
            assert(tabmOut.output instanceof Float32Array, "tabm output missing");
            assert(namOut.output instanceof Float32Array, "nam output missing");
          } finally {
            tabm.dispose();
            nam.dispose();
          }
        });
        await test("mlp train step with Adam", async () => {
          const inst = MLP({
            layers: [2, 4, 1],
            activation: "relu",
            bias: true,
            loss: "mse",
            batch_size: 1,
            seed: 42
          });
          try {
            inst.setOptimizer(pg.OPTIM_ADAM, 0.01);
            const x = new Float32Array([1, 2]);
            const y = new Float32Array([3]);
            let first = null;
            let last = null;
            for (let step = 0; step < 50; step++) {
              last = await inst.trainStep({ x, y });
              if (first == null) first = last;
            }
            assert(last < first, `expected loss to decrease (${first} -> ${last})`);
          } finally {
            inst.dispose();
          }
        });
        await test("mlp train step with SGD momentum creates named state", async () => {
          const inst = MLP({
            layers: [2, 4, 1],
            activation: "relu",
            bias: true,
            loss: "mse",
            batch_size: 1,
            seed: 42
          });
          try {
            inst.setOptimizer(pg.OPTIM_SGD, 0.01, 0.9, 0.999, 1e-8, 0, 0.9);
            const x = new Float32Array([1, 2]);
            const y = new Float32Array([3]);
            const loss = await inst.trainStep({ x, y });
            assert(Number.isFinite(loss), `loss should be finite, got ${loss}`);
            const bi = inst.findBuf("optim.sgd.b.layers.0.weight");
            assert(bi >= 0, "missing SGD momentum state buffer");
            const b = await inst.bufData(bi);
            assert(Array.from(b).some((v) => Math.abs(v) > 0), "momentum state should update");
            const defaultNames = safetensorNames(await inst.exportWeights());
            assert(defaultNames.has("optim.sgd.b.layers.0.weight"), "default export should include optimizer state");
            const modelOnlyNames = safetensorNames(await inst.exportWeights({ includeOptimizer: false }));
            assert(modelOnlyNames.has("layers.0.weight"), "model-only export should include params");
            assert(!modelOnlyNames.has("optim.sgd.b.layers.0.weight"), "model-only export should exclude optimizer state");
          } finally {
            inst.dispose();
          }
        });
        await test("mlp batch_size=4 mse train", async () => {
          const inst = MLP({
            layers: [2, 4, 2],
            activation: "relu",
            bias: true,
            loss: "mse",
            batch_size: 4,
            seed: 42
          });
          try {
            inst.setOptimizer(pg.OPTIM_SGD, 0.01);
            const x = new Float32Array(4 * 2).fill(0.5);
            const y = new Float32Array(4 * 2).fill(0.3);
            let first = null;
            let last = null;
            for (let step = 0; step < 50; step++) {
              last = await inst.trainStep({ x, y });
              if (first == null) first = last;
            }
            assert(Number.isFinite(first), `first loss should be finite, got ${first}`);
            assert(Number.isFinite(last), `last loss should be finite, got ${last}`);
            assert(last < first, `expected loss to decrease (${first} -> ${last})`);
          } finally {
            inst.dispose();
          }
        });
        await test("mlp 100-step convergence", async () => {
          const inst = MLP({
            layers: [2, 8, 1],
            activation: "relu",
            bias: true,
            loss: "mse",
            batch_size: 1,
            seed: 42
          });
          try {
            inst.setOptimizer(pg.OPTIM_SGD, 0.01);
            const x = new Float32Array([1, 2]);
            const y = new Float32Array([5]);
            let first = null;
            let last = null;
            for (let step = 0; step < 100; step++) {
              last = await inst.trainStep({ x, y });
              if (first == null) first = last;
            }
            assert(last < first * 0.1, `expected >90% loss reduction (${first} -> ${last})`);
          } finally {
            inst.dispose();
          }
        });
        console.log(`
Model tests: ${passed} passed, ${failed} failed`);
        return { passed, failed };
      }
      async function runModelSmokeTests(pg, createRuntime) {
        const Model = pg.Model;
        const { MLP, TabM, NAM } = pg.models;
        const testFilter = testFilterFor(pg);
        let passed = 0;
        let failed = 0;
        if (!pg.supportsModel) {
          console.log("\n== Model ==");
          console.log("  [SKIP] core does not expose PolyModel runtime yet");
          return { passed: 0, failed: 0 };
        }
        async function test(name, fn) {
          if (testFilter && !name.includes(testFilter)) return;
          try {
            await fn();
            console.log(`  [PASS] ${name}`);
            passed++;
          } catch (e) {
            console.log(`  [FAIL] ${name}: ${e.message}`);
            failed++;
          }
        }
        console.log("\n== Model ==");
        await test("Model stateful capture shares train eval state", () => checkModelStatefulCapture(pg));
        await test("Model stateful capture owns resumable RNG", () => checkModelCaptureRng(pg));
        await test("Model stateful capture restores failures and rejects async execution", () => checkModelCaptureFailure(pg));
        await test("Model composition factories share C construction", () => checkCompositionFactories(pg));
        await test("Model checkpoint replacement uses queued readback", () => checkModelCheckpointReplacement(pg));
        await test("Model composition catalogue and named target objective", () => checkCompositionCatalogue(pg));
        await test("Model tied Adam placement freeze and checkpoint", () => checkTiedAdamCheckpoint(pg));
        await test("Model constructor collects object state", () => checkModelConstructor(pg, Model));
        await test("Model constructor dispatch and explicit factories", () => checkModelDispatch(pg, Model));
        await test("Model usability summary and capture failures", () => checkModelUsability(pg, Model));
        await test("Model runtime imports isolation and failure", () => checkRuntimeImports(pg, Model, createRuntime));
        await test("Model family runtime ownership", () => checkFamilyRuntimeOwnership(pg));
        await test("Llama family reference and shared import", () => checkLlamaFamily(pg));
        await test("Model Tensor I/O owns device results", () => checkModelTensorIO(pg));
        await test("Model variable shapes preserve results and portable signatures", () => checkModelVariableShapes(pg));
        await test("Model empty bindings reject before input writes", () => checkModelEmptyInputAdmission(pg));
        await test("Model minibatches match explicit training steps", () => checkModelMinibatches(pg));
        await test("Model bounded minibatches and Tensor datasets", () => checkModelBoundedMinibatches(pg));
        await test("Model trace seals independent state with named loss", () => checkModelTrace(pg, Model));
        await test("Model copied storage exact writes and objective selection", () => checkModelStorageObjectives(pg, Model));
        await test("Model codecs reject malformed bytes before publication", () => checkModelCodecRejection(pg));
        await test("Model quantized weights match pinned GGUF bit planes", () => checkQuantizedModelWeights(pg));
        await test("typed integer input preserves bytes and rejects float binding", async () => {
          await checkTypedIntegerInput(pg, Model);
        });
        await test("generic call validates signature and returns selected outputs", async () => {
          await checkCallSignatureAndSelectedOutputs(pg, Model);
        });
        await test("webgpu module device map places exact Tensor cuts atomically", async () => {
          await checkModuleDeviceMap(pg, Model);
        });
        await test("webgpu mlp forward smoke", async () => {
          const inst = MLP({
            layers: [2, 4, 1],
            activation: "relu",
            bias: true,
            loss: "mse",
            batch_size: 1,
            seed: 42
          });
          try {
            const outputs = await inst.forward({ x: new Float32Array([1, 2]) });
            assert(outputs.output instanceof Float32Array, "output should be Float32Array");
            assert(outputs.output.length === 1, `expected output length 1, got ${outputs.output.length}`);
            assert(Number.isFinite(outputs.output[0]), "output should be finite");
          } finally {
            inst.dispose();
          }
        });
        console.log(`
Model smoke tests: ${passed} passed, ${failed} failed`);
        return { passed, failed };
      }
      module.exports = { runModelRuntimeTests, runModelSmokeTests };
    }
  });

  // test/browser/test_browser_entry.js
  var require_test_browser_entry = __commonJS({
    "test/browser/test_browser_entry.js"() {
      var { runTensorTests } = require_test_tensor();
      var { runModelRuntimeTests, runModelSmokeTests } = require_test_model_runtime();
      window.__runTensorTests = runTensorTests;
      window.__runModelRuntimeTests = runModelRuntimeTests;
      window.__runModelSmokeTests = runModelSmokeTests;
    }
  });
  require_test_browser_entry();
})();
