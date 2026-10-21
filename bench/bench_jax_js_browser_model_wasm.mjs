import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
import { chromium } from "../js/node_modules/playwright/index.mjs";
import { assessABBA, assessPagePairs, assessKernelPages, pagePairOrder, decodedWatOpcodes, structuralSnapshot } from "./wasm_checks.mjs";
const { launchOptionsFor } = createRequire(import.meta.url)("../js/test/browser/run.js");

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const root = path.resolve(__dirname, "..");

const CASES = {
  mlp_small: { kind: "mlp", tokens: 64, d: 256, hidden: 1024, out: 256, activation: "relu" },
  mlp_token: { kind: "mlp", tokens: 1, d: 768, hidden: 3072, out: 768, activation: "relu" },
  mlp_batch: { kind: "mlp", tokens: 16, d: 768, hidden: 3072, out: 768, activation: "relu" },
  qwen_ffn_token: { kind: "qwen_ffn", tokens: 1, d: 1024, hidden: 2816, out: 1024, activation: "silu" },
  qwen_ffn_batch: { kind: "qwen_ffn", tokens: 16, d: 1024, hidden: 2816, out: 1024, activation: "silu" },
};

function parseArgs(argv) {
  const args = {
    cases: ["mlp_small", "mlp_token", "mlp_batch", "qwen_ffn_token", "qwen_ffn_batch"],
    iters: 10,
    warmup: 2,
    paired: false,
    rounds: 1,
    bundle: "js/dist/polygrad.sync.js",
    baseline: null,
    async: false,
    kernels: null,
    stageBreakdown: false,
    output: null,
    maxSlowdown: null,
    device: "wasm",
    pipeline: "staged",
    baselinePipeline: null,
    profileTimestamps: true,
    pagePairs: null,
    polygradOnly: false,
    structure: false,
    snapshot: null,
    updateSnapshot: false,
    qualify: false,
    kernelQualify: false,
  };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    const next = () => {
      if (i + 1 >= argv.length) throw new Error("missing value for " + a);
      return argv[++i];
    };
    if (a === "--cases") args.cases = next().split(",").map((x) => x.trim()).filter(Boolean);
    else if (a === "--device") args.device = next();
    else if (a === "--pipeline") args.pipeline = next();
    else if (a === "--baseline-pipeline") args.baselinePipeline = next();
    else if (a === "--no-profile-timestamps") args.profileTimestamps = false;
    else if (a === "--page-pairs") args.pagePairs = Number(next());
    else if (a === "--polygrad-only") args.polygradOnly = true;
    else if (a === "--structure") args.structure = true;
    else if (a === "--snapshot") { args.snapshot = path.resolve(root, next()); args.structure = true; }
    else if (a === "--update-snapshot") args.updateSnapshot = true;
    else if (a === "--qualify") args.qualify = true;
    else if (a === "--kernel-qualify") args.kernelQualify = true;
    else if (a === "--iters") args.iters = Number(next());
    else if (a === "--warmup") args.warmup = Number(next());
    else if (a === "--bundle") args.bundle = next();
    else if (a === "--baseline") args.baseline = next();
    else if (a === "--output") args.output = next();
    else if (a === "--max-slowdown") args.maxSlowdown = Number(next());
    else if (a === "--async") args.async = true;
    else if (a === "--kernels") args.kernels = true;
    else if (a === "--no-kernels") args.kernels = false;
    else if (a === "--stage-breakdown") args.stageBreakdown = true;
    else if (a === "--help" || a === "-h") {
      console.log('Kernel controls: --kernel-qualify --pipeline whole (twenty fresh pages, no baseline; not a whole-call approval)');
      console.log("Usage: node bench/bench_jax_js_browser_model_wasm.mjs [--device wasm|webgpu] [--pipeline staged|whole|captured-stages] [--baseline-pipeline staged|whole|captured-stages] [--no-profile-timestamps] [--cases a,b] [--iters N] [--warmup N] [--paired] [--rounds N] [--bundle repo/path.js] [--baseline repo/old.js] [--async] [--kernels|--no-kernels] [--stage-breakdown] [--output report.json] [--max-slowdown 0.10] [--page-pairs 10] [--polygrad-only] [--qualify] [--structure] [--snapshot file.json [--update-snapshot]]");
      process.exit(0);
    } else if (a === "--paired") {
      args.paired = true;
    } else if (a === "--quiet") {
      // Console output is already limited to results.
    } else if (a === "--rounds") {
      args.rounds = Number(next());
    } else {
      throw new Error("unknown option " + a);
    }
  }
  for (const name of args.cases) {
    if (!CASES[name]) throw new Error("unknown case '" + name + "'");
  }
  if (!Number.isFinite(args.iters) || args.iters <= 0) throw new Error("--iters must be positive");
  if (!Number.isFinite(args.warmup) || args.warmup < 0) throw new Error("--warmup must be nonnegative");
  if (!Number.isInteger(args.rounds) || args.rounds <= 0) throw new Error("--rounds must be a positive integer");
  if (args.pagePairs !== null) {
    pagePairOrder(args.pagePairs);
    if (args.pagePairs < 10 || !args.baseline) throw new Error('--page-pairs requires a baseline and at least ten pairs');
  }
  if (args.structure && (args.device !== 'wasm' || args.baseline || args.pagePairs))
    throw new Error('structure requires one Wasm candidate; never mix instrumentation with comparative timing');
  if (args.structure) args.polygradOnly = true;
  if (args.kernelQualify) {
    if (args.device !== 'wasm' || args.pipeline !== 'whole' || args.baseline || args.baselinePipeline || args.pagePairs
      || args.qualify || args.structure || args.stageBreakdown || args.maxSlowdown !== null || args.rounds !== 1)
      throw new Error('--kernel-qualify requires one Wasm whole pipeline, no baseline or other measurement mode');
    args.polygradOnly = true;
  }
  if (args.qualify) {
    if (args.device !== 'wasm' || args.structure || args.pagePairs === null || args.iters < 20 || args.iters % 10)
      throw new Error('qualification requires Wasm, at least ten page pairs, and iterations divisible by ten (at least twenty)');
    args.polygradOnly = true;
    if (args.baselinePipeline) throw new Error('qualification requires one pipeline');
  }
  if (args.updateSnapshot && !args.snapshot) throw new Error('--update-snapshot requires --snapshot');
  if (args.maxSlowdown !== null && (!(args.baseline || args.baselinePipeline) || !Number.isFinite(args.maxSlowdown) || args.maxSlowdown < 0)) {
    throw new Error("--max-slowdown requires --baseline or --baseline-pipeline and a nonnegative fraction");
  }
  if (args.maxSlowdown !== null && args.pagePairs === null)
    throw new Error('--max-slowdown requires --page-pairs; four-page ABBA is diagnostic only');
  if (!["wasm", "webgpu"].includes(args.device)) throw new Error("--device must be wasm or webgpu");
  for (const mode of [args.pipeline, args.baselinePipeline].filter(Boolean)) {
    if (!["staged", "whole", "captured-stages"].includes(mode)) throw new Error("pipeline must be staged, whole or captured-stages");
  }
  if ([args.pipeline, args.baselinePipeline].includes("captured-stages") && args.device !== "webgpu")
    throw new Error("captured-stages requires WebGPU async JIT");
  if (args.device === "webgpu") {
    args.async = true;
    if (args.bundle === "js/dist/polygrad.sync.js") args.bundle = "js/dist/polygrad.async.js";
  }
  args.output = path.resolve(root, args.output ?? `temp/performance/${args.device}-model-${Date.now()}-${process.pid}.json`);
  if (fs.existsSync(args.output)) throw new Error("refusing to overwrite report: " + args.output);
  for (const key of ["bundle", "baseline"]) {
    if (args[key] === null) continue;
    const bundlePath = path.resolve(root, args[key]);
    if (!bundlePath.startsWith(root + path.sep) || !fs.statSync(bundlePath).isFile()) throw new Error("--" + key + " must be a file inside the repository");
    args[key] = path.relative(root, bundlePath).split(path.sep).map(encodeURIComponent).join("/");
  }
  args.iters = Math.trunc(args.iters);
  args.warmup = Math.trunc(args.warmup);
  return args;
}

const args = parseArgs(process.argv);

const MIME = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".mjs": "text/javascript",
  ".wasm": "application/wasm",
  ".json": "application/json",
};

function makePage() {
  return `<!doctype html>
<meta charset="utf-8">
<title>polygrad vs jax-js browser wasm model</title>
<pre id="log"></pre>
<script src="/${args.bundle}"></script>
${args.structure || args.qualify || args.kernelQualify ? '<script src="/js/test/test_jit_boundaries.js"></script>' : ''}
<script type="module">
import { checkWorkload, modelReference, timeWorkload, measureKernelBlocks, assertClose } from "/bench/wasm_checks.mjs";
import { observeWebGPU } from "/bench/webgpu_profile.mjs";
const args = ${JSON.stringify(args)};
const CASES = ${JSON.stringify(CASES)};
let blockUntilReady, defaultDevice, init, jit, nn, np;
if (!args.polygradOnly) ({ blockUntilReady, defaultDevice, init, jit, nn, numpy: np } = await import("/references/jax-js/dist/index.js"));
const gpu = args.device === "webgpu" ? observeWebGPU() : null;
const boundaries = args.structure || args.kernelQualify || (args.qualify && args.control === 'kernel') ? observeBoundaries() : null;
const kernelHashes = new WeakMap();
const polyJit = (pg, fn) => gpu ? pg.jit.async(fn) : pg.jit(fn);
const realizeTensor = (tensor, ...others) => gpu ? tensor.realizeAsync(...others) : tensor.realize(...others);

async function completeOutput(out, engine) {
  if (!gpu) return engine === "polygrad" ? out.value.realize() : out.value.blockUntilReady();
  // The comparison includes completed output readback, not just submission.
  return engine === "polygrad" ? (out.value.toArrayAsync ? out.value.toArrayAsync() : out.value.toArray()) : out.value.ref.data();
}

let workerCreates = 0;
const NativeWorker = globalThis.Worker;
if (typeof NativeWorker !== "undefined") {
  function CountingWorker(...workerArgs) {
    workerCreates++;
    return new NativeWorker(...workerArgs);
  }
  CountingWorker.prototype = NativeWorker.prototype;
  globalThis.Worker = CountingWorker;
}

function makeData(n, seed, scale = 0.02) {
  const data = new Float32Array(n);
  let x = seed >>> 0;
  for (let i = 0; i < n; i++) {
    x = (Math.imul(x, 1664525) + 1013904223) >>> 0;
    data[i] = (((x >>> 8) % 1009) - 504) * scale / 504;
  }
  return data;
}

function median(xs) {
  const ys = [...xs].sort((a, b) => a - b);
  return ys[Math.floor(ys.length / 2)];
}

function modelFlops(c) {
  if (c.kind === "mlp") return 2 * c.tokens * c.d * c.hidden + 2 * c.tokens * c.hidden * c.out;
  if (c.kind === "qwen_ffn") return 2 * c.tokens * c.d * c.hidden * 2 + 2 * c.tokens * c.hidden * c.d;
  return NaN;
}

function caseLabel(c) {
  if (c.kind === "mlp") return c.kind + " tokens=" + c.tokens + " d=" + c.d + " hidden=" + c.hidden + " out=" + c.out;
  return c.kind + " tokens=" + c.tokens + " d=" + c.d + " hidden=" + c.hidden;
}

function caseInputs(c) {
  const x = makeData(c.tokens * c.d, 1);
  if (c.kind === "mlp") {
    return {
      x,
      w1: makeData(c.hidden * c.d, 2),
      b1: makeData(c.hidden, 3, 0.01),
      w2: makeData(c.out * c.hidden, 4),
      b2: makeData(c.out, 5, 0.01),
    };
  }
  return {
    x,
    wg: makeData(c.hidden * c.d, 6),
    wu: makeData(c.hidden * c.d, 7),
    wd: makeData(c.d * c.hidden, 8),
  };
}

async function timeAsync(call, ready, dispose) {
  return (await timeWorkload(call, ready, dispose, args)).samples;
}

function createJaxInputs(c, inputData) {
  if (c.kind === "mlp") {
    return [
      np.array(inputData.x, { shape: [c.tokens, c.d], device: args.device }),
      np.array(inputData.w1, { shape: [c.hidden, c.d], device: args.device }),
      np.array(inputData.b1, { shape: [1, c.hidden], device: args.device }),
      np.array(inputData.w2, { shape: [c.out, c.hidden], device: args.device }),
      np.array(inputData.b2, { shape: [1, c.out], device: args.device }),
    ];
  }
  return [
    np.array(inputData.x, { shape: [c.tokens, c.d], device: args.device }),
    np.array(inputData.wg, { shape: [c.hidden, c.d], device: args.device }),
    np.array(inputData.wu, { shape: [c.hidden, c.d], device: args.device }),
    np.array(inputData.wd, { shape: [c.d, c.hidden], device: args.device }),
  ];
}

function createJaxPipeline(c) {
  if (args.pipeline === "whole") {
    const f = jit((...inputs) => {
      if (c.kind === "mlp") {
        const [x, w1, b1, w2, b2] = inputs;
        const pre = np.matmul(x.ref, w1.ref.transpose()).ref.add(b1.ref);
        const h = c.activation === "silu" ? nn.silu(pre) : np.maximum(pre, 0);
        return np.matmul(h, w2.ref.transpose()).ref.add(b2.ref);
      }
      const [x, wg, wu, wd] = inputs;
      const gate = nn.silu(np.matmul(x.ref, wg.ref.transpose()));
      const up = np.matmul(x.ref, wu.ref.transpose());
      return np.matmul(gate.mul(up), wd.ref.transpose());
    }, { device: args.device });
    return {
      call(inputs) { return { value: f(...inputs.map(x => x.ref)) }; },
      async ready(out) { await completeOutput(out, "jax-js"); },
      disposeOutput(out) { out.value.dispose(); },
      dispose() { f.dispose?.(); },
    };
  }
  if (c.kind === "mlp") {
    const f1 = jit((x, w1, b1) => {
      const pre = np.matmul(x.ref, w1.ref.transpose()).ref.add(b1.ref);
      return c.activation === "silu" ? nn.silu(pre) : np.maximum(pre, 0);
    }, { device: args.device });
    const f2 = jit((h, w2, b2) => np.matmul(h.ref, w2.ref.transpose()).ref.add(b2.ref), { device: args.device });
    return {
      call(inputs) {
        const [x, w1, b1, w2, b2] = inputs;
        const h = f1(x.ref, w1.ref, b1.ref);
        const y = f2(h.ref, w2.ref, b2.ref);
        return { value: y, temps: [h] };
      },
      async ready(out) { await completeOutput(out, "jax-js"); },
      disposeOutput(out) {
        out.value.dispose();
        for (const t of out.temps) t.dispose();
      },
      dispose() {
        f1.dispose?.();
        f2.dispose?.();
      },
    };
  }
  const fGate = jit((x, wg) => nn.silu(np.matmul(x.ref, wg.ref.transpose())), { device: args.device });
  const fUp = jit((x, wu) => np.matmul(x.ref, wu.ref.transpose()), { device: args.device });
  const fMul = jit((gate, up) => gate.ref.mul(up.ref), { device: args.device });
  const fDown = jit((h, wd) => np.matmul(h.ref, wd.ref.transpose()), { device: args.device });
  return {
    call(inputs) {
      const [x, wg, wu, wd] = inputs;
      const gate = fGate(x.ref, wg.ref);
      const up = fUp(x.ref, wu.ref);
      const h = fMul(gate.ref, up.ref);
      const y = fDown(h.ref, wd.ref);
      return { value: y, temps: [gate, up, h] };
    },
    async ready(out) { await completeOutput(out, "jax-js"); },
    disposeOutput(out) {
      out.value.dispose();
      for (const t of out.temps) t.dispose();
    },
    dispose() {
      fGate.dispose?.();
      fUp.dispose?.();
      fMul.dispose?.();
      fDown.dispose?.();
    },
  };
}

async function createPolyInputs(pg, c, inputData) {
  const { Tensor } = pg;
  if (c.kind === "mlp") {
    const inputs = [
      new Tensor(inputData.x).reshape(c.tokens, c.d),
      new Tensor(inputData.w1).reshape(c.hidden, c.d),
      new Tensor(inputData.b1).reshape(1, c.hidden),
      new Tensor(inputData.w2).reshape(c.out, c.hidden),
      new Tensor(inputData.b2).reshape(1, c.out),
    ];
    await realizeTensor(inputs[0], ...inputs.slice(1));
    return inputs;
  }
  const inputs = [
    new Tensor(inputData.x).reshape(c.tokens, c.d),
    new Tensor(inputData.wg).reshape(c.hidden, c.d),
    new Tensor(inputData.wu).reshape(c.hidden, c.d),
    new Tensor(inputData.wd).reshape(c.d, c.hidden),
  ];
  await realizeTensor(inputs[0], ...inputs.slice(1));
  return inputs;
}

function createPolyPipeline(pg, c) {
  if (args.pipeline === "captured-stages") {
    // Same materialization boundaries as staged, captured into one LINEAR.
    // JAX stays staged in this diagnostic control; whole changes both engines.
    const f = pg.jit.async(async (...inputs) => {
      if (c.kind === "mlp") {
        const [x, w1, b1, w2, b2] = inputs;
        const pre = x.matmul(w1.permute(1, 0)).add(b1);
        const h = c.activation === "silu" ? pre.silu() : pre.relu();
        await h.realizeAsync();
        return h.matmul(w2.permute(1, 0)).add(b2);
      }
      const [x, wg, wu, wd] = inputs;
      const gate = x.matmul(wg.permute(1, 0)).silu();
      await gate.realizeAsync();
      const up = x.matmul(wu.permute(1, 0));
      await up.realizeAsync();
      const h = gate.mul(up);
      await h.realizeAsync();
      return h.matmul(wd.permute(1, 0));
    });
    return {
      async call(inputs) { return { value: await f(...inputs) }; },
      async ready(out) { await completeOutput(out, "polygrad"); },
      disposeOutput(_) {},
      dispose() { f.dispose?.(); },
      get scheduleCount() { return f.scheduleCount || 0; },
    };
  }
  if (args.pipeline === "whole") {
    const f = polyJit(pg, (...inputs) => {
      if (c.kind === "mlp") {
        const [x, w1, b1, w2, b2] = inputs;
        const pre = x.matmul(w1.permute(1, 0)).add(b1);
        const h = c.activation === "silu" ? pre.silu() : pre.relu();
        return h.matmul(w2.permute(1, 0)).add(b2);
      }
      const [x, wg, wu, wd] = inputs;
      const gate = x.matmul(wg.permute(1, 0)).silu();
      const up = x.matmul(wu.permute(1, 0));
      return gate.mul(up).matmul(wd.permute(1, 0));
    });
    return {
      async call(inputs) { return { value: await f(...inputs) }; },
      async ready(out) { await completeOutput(out, "polygrad"); },
      disposeOutput(_) {},
      dispose() { f.dispose?.(); },
      get scheduleCount() { return f.scheduleCount || 0; },
    };
  }
  if (c.kind === "mlp") {
    const f1 = polyJit(pg, (x, w1, b1) => {
      const pre = x.matmul(w1.permute(1, 0)).add(b1);
      return c.activation === "silu" ? pre.silu() : pre.relu();
    });
    const f2 = polyJit(pg, (h, w2, b2) => h.matmul(w2.permute(1, 0)).add(b2));
    return {
      async call(inputs) {
        const [x, w1, b1, w2, b2] = inputs;
        const h = await f1(x, w1, b1);
        await realizeTensor(h);
        const y = await f2(h, w2, b2);
        return { value: y };
      },
      async ready(out) { await completeOutput(out, "polygrad"); },
      disposeOutput(_) {},
      dispose() {
        f1.dispose?.();
        f2.dispose?.();
      },
      get scheduleCount() { return (f1.scheduleCount || 0) + (f2.scheduleCount || 0); },
    };
  }
  const fGate = polyJit(pg, (x, wg) => x.matmul(wg.permute(1, 0)).silu());
  const fUp = polyJit(pg, (x, wu) => x.matmul(wu.permute(1, 0)));
  const fMul = polyJit(pg, (gate, up) => gate.mul(up));
  const fDown = polyJit(pg, (h, wd) => h.matmul(wd.permute(1, 0)));
  return {
    async call(inputs) {
      const [x, wg, wu, wd] = inputs;
      const gate = await fGate(x, wg);
      const up = await fUp(x, wu);
      await realizeTensor(gate, up);
      const h = await fMul(gate, up);
      await realizeTensor(h);
      const y = await fDown(h, wd);
      return { value: y };
    },
    async ready(out) { await completeOutput(out, "polygrad"); },
    disposeOutput(_) {},
    dispose() {
      fGate.dispose?.();
      fUp.dispose?.();
      fMul.dispose?.();
      fDown.dispose?.();
    },
    get scheduleCount() {
      return (fGate.scheduleCount || 0) + (fUp.scheduleCount || 0) +
        (fMul.scheduleCount || 0) + (fDown.scheduleCount || 0);
    },
  };
}

async function primePipeline(pipeline, inputs) {
  for (let i = 0; i < 2; i++) {
    const y = await pipeline.call(inputs);
    await pipeline.ready(y);
    pipeline.disposeOutput(y);
  }
}

async function fingerprintCompilation(call) {
  if (gpu) {
    const start = gpu.shaders.length;
    await call();
    return Promise.all(gpu.shaders.slice(start).map(async source => {
      const hash = new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(source)));
      return Array.from(hash, x => x.toString(16).padStart(2, "0")).join("");
    }));
  }
  const Original = WebAssembly.Module;
  const kernels = [];
  const modules = [];
  // Observe only Polygrad compilation, outside timing; jax-js also exports "kernel".
  WebAssembly.Module = new Proxy(Original, { construct(target, args) {
    const module = Reflect.construct(target, args);
    if (Original.exports(module).some(e => e.name === "kernel")) {
      modules.push(module);
      const bytes = args[0];
      kernels.push(new Uint8Array(bytes instanceof ArrayBuffer ? bytes :
        bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)).slice());
    }
    return module;
  } });
  try { await call(); }
  finally { WebAssembly.Module = Original; }
  const hashes = await Promise.all(kernels.map(async (bytes, i) => {
    const hash = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
    const hex = Array.from(hash, x => x.toString(16).padStart(2, "0")).join("");
    kernelHashes.set(modules[i], hex);
    return hex;
  }));
  return args.structure ? { hashes, modules: kernels.map(bytes => Array.from(bytes)) } : hashes;
}

async function runPolyCase(pg, name, c, inputData) {
  const inputs = await createPolyInputs(pg, c, inputData);
  const pipeline = createPolyPipeline(pg, c);
  let validation;
  const compiled = await fingerprintCompilation(async () => {
    await primePipeline(pipeline, inputs);
    validation = await checkWorkload({ name, call: () => pipeline.call(inputs), ready: y => pipeline.ready(y), dispose: y => pipeline.disposeOutput(y) }, modelReference(c,inputData), {atol:c.kind==='mlp'?1e-7:1e-10,rtol:2e-4});
  });
  let structure;
  if (args.structure) {
    const samples = [];
    for (let i = 0; i < 3; i++) {
      boundaries.exports = boundaries.imports = boundaries.kernels = boundaries.allocationExports = 0;
      boundaries.active = true;
      let y;
      try { y = await pipeline.call(inputs); await pipeline.ready(y); }
      finally { boundaries.active = false; if (y) pipeline.disposeOutput(y); }
      samples.push({ coreExports: boundaries.exports, coreImports: boundaries.imports,
        dispatches: boundaries.kernels, allocationExports: boundaries.allocationExports });
    }
    if (samples.some(s => JSON.stringify(s) !== JSON.stringify(samples[0])))
      throw new Error(name + ': warm structural counts are not stable');
    structure = { cold: { modules: compiled.modules }, warm: samples[0] };
  }
  // Qualification injects executed work, never scaled reported durations.
  // The five FFN fixtures are stateless: repeating their captured kernels with
  // the current call's arguments leaves outputs unchanged. Do not use this
  // injection for training/RNG/aliasing workloads.
  let burnIterations = 0, checksum = 1;
  const burn = n => { let x = checksum; for (let i = 0; i < n; i++) x = Math.imul(x ^ i, 16777619); checksum = x; };
  if (args.qualify && args.control === 'host') {
    for (let i = 0; i < 5; i++) burn(100000);
    const start = performance.now();
    for (let i = 0; i < 10; i++) burn(100000);
    const perIteration = (performance.now() - start) / 1000000;
    const pilot = await timeAsync(() => pipeline.call(inputs), y => pipeline.ready(y), y => pipeline.disposeOutput(y));
    burnIterations = Math.max(1, Math.round(median(pilot) * 1000 * .1 / perIteration));
    if (!Number.isFinite(burnIterations) || burnIterations > 100000000) throw new Error('invalid host injection calibration');
  }
  const qualification = args.qualify ? { burnIterations, targetFraction: .1, kernelExtraEvery: 10,
    extraKernelCalls: 0, hostBurnCalls: 0 } : undefined;
  return {
    measure: async () => {
      if (!args.qualify) return timeAsync(() => pipeline.call(inputs), y => pipeline.ready(y), y => pipeline.disposeOutput(y));
      // Both lanes collect the same trace; only the candidate injects work.
      const measured = await timeWorkload(() => pipeline.call(inputs), y => pipeline.ready(y), y => pipeline.disposeOutput(y), {
        warmup: args.warmup, iters: args.iters, observer: boundaries,
        inject: args.lane === 'candidate' && args.control !== 'identity' ? args.control : null,
        burn: () => burn(burnIterations),
      });
      qualification.extraKernelCalls += measured.extraKernelCalls;
      qualification.hostBurnCalls += measured.hostBurnCalls;
      // Check that injected work preserved the public output contract.
      await checkWorkload({ name, call: () => pipeline.call(inputs), ready: y => pipeline.ready(y), dispose: y => pipeline.disposeOutput(y) }, modelReference(c,inputData), {atol:c.kind==='mlp'?1e-7:1e-10,rtol:2e-4});
      qualification.checksum = checksum;
      return measured.samples;
    },
    schedules: pipeline.scheduleCount,
    validation,
    compiledKernelHashes: compiled.hashes ?? compiled,
    structure,
    qualification,
    kernelProfile: async () => {
      const referenceOutput = await pipeline.call(inputs);
      let before;
      try { await pipeline.ready(referenceOutput); before = Float32Array.from(await referenceOutput.value.toArray()); }
      finally { pipeline.disposeOutput(referenceOutput); }
      let output;
      boundaries.trace = [];
      try {
        output = await pipeline.call(inputs);
        await pipeline.ready(output);
        const trace = boundaries.trace;
        boundaries.trace = null;
        for (const t of trace) {
          t.hash = kernelHashes.get(t.module);
          if (!t.hash) throw new Error('captured kernel has no compilation fingerprint');
        }
        // No runtime call or disposal while raw pointers are replayed. These
        // fixed FFNs have no state/RNG and overwrite outputs on every dispatch.
        const measurements = measureKernelBlocks(trace, { page: args.kernelPage });
        const actual = await output.value.toArray();
        const beforeBits = new Uint32Array(before.buffer), afterBits = new Uint32Array(Float32Array.from(actual).buffer);
        if (beforeBits.length !== afterBits.length || beforeBits.some((v, i) => v !== afterBits[i]))
          throw new Error(name + ': raw replay changed output bits');
        assertClose(actual, modelReference(c, inputData), name + ' raw replay',
          {atol:c.kind==='mlp'?1e-7:1e-10, rtol:2e-4});
        return measurements;
      } finally {
        boundaries.trace = null;
        if (output) pipeline.disposeOutput(output);
      }
    },
    profile: async () => gpu.profile(async () => {
      const out = await pipeline.call(inputs);
      try { await pipeline.ready(out); } finally { pipeline.disposeOutput(out); }
    }, { timestamps: args.profileTimestamps }),
    dispose() { pipeline.dispose(); for (const x of inputs) x.dispose?.(); },
  };
}

async function runJaxCase(name, c, inputData) {
  const inputs = createJaxInputs(c, inputData);
  await blockUntilReady(inputs);
  const pipeline = createJaxPipeline(c);
  await primePipeline(pipeline, inputs);
  const validation = await checkWorkload({ name, call: () => pipeline.call(inputs), ready: y => pipeline.ready(y), dispose: y => pipeline.disposeOutput(y) }, modelReference(c,inputData), {atol:c.kind==='mlp'?1e-7:1e-10,rtol:2e-4});
  return {
    measure: () => timeAsync(() => pipeline.call(inputs), y => pipeline.ready(y), y => pipeline.disposeOutput(y)),
    validation,
    profile: async () => gpu.profile(async () => {
      const out = await pipeline.call(inputs);
      try { await pipeline.ready(out); } finally { pipeline.disposeOutput(out); }
    }, { timestamps: args.profileTimestamps }),
    dispose() { pipeline.dispose(); for (const x of inputs) x.dispose(); },
  };
}

async function runPairedCase(pg, name, c, inputData) {
  const lanes = [];
  const samples = [[], []];
  const order = [];
  try {
    lanes.push(await runPolyCase(pg, name, c, inputData));
    if (!args.polygradOnly) lanes.push(await runJaxCase(name, c, inputData));
    for (let round = 0; round < args.rounds; round++) {
      // Keep compiled pipelines alive; alternate timing order, not compilation.
      const indices = args.polygradOnly ? [0] : args.paired && round % 2 ? [1, 0] : [0, 1];
      for (const i of indices) {
        order.push(i === 0 ? "polygrad" : "jax-js");
        samples[i].push(...await lanes[i].measure());
      }
    }
    console.log(JSON.stringify({ case: name, timingOrder: order }));
    // Separate diagnostics so query work never enters the latency samples.
    const profiles = [[], []];
    if (gpu && !args.polygradOnly) for (let round = 0; round < 4; round++) {
      for (const i of round % 2 ? [1, 0] : [0, 1]) profiles[i].push(await lanes[i].profile());
    }
    const kernelMeasurements = args.kernelQualify ? await lanes[0].kernelProfile() : undefined;
    return lanes.map((lane, i) => {
      // The injected extra kernel runs every tenth call; a per-call median
      // would discard that real cost. A page contributes exactly one mean.
      const s = samples[i].reduce((a, b) => a + b, 0) / samples[i].length;
      return { name, label: caseLabel(c), seconds: s, us: s * 1e6,
        tokens_per_s: c.tokens / s, gflops: modelFlops(c) / s / 1e9,
        schedules: lane.schedules, samples: samples[i], validation: lane.validation,
        compiledKernelHashes: lane.compiledKernelHashes, structure: lane.structure,
        qualification: lane.qualification, profiles: profiles[i], kernelMeasurements };
    });
  } finally {
    for (const lane of lanes) lane.dispose();
  }
}

async function runPolyQwenStages(pg, name, c, inputData) {
  if (c.kind !== "qwen_ffn") return [];
  const [x, wg, wu, wd] = await createPolyInputs(pg, c, inputData);
  const fGate = polyJit(pg, (xx, ww) => xx.matmul(ww.permute(1, 0)).silu());
  const fUp = polyJit(pg, (xx, ww) => xx.matmul(ww.permute(1, 0)));
  const fMul = polyJit(pg, (gate, up) => gate.mul(up));
  const fDown = polyJit(pg, (h, ww) => h.matmul(ww.permute(1, 0)));

  const gate0 = await fGate(x, wg);
  const up0 = await fUp(x, wu);
  await realizeTensor(gate0, up0);
  const h0 = await fMul(gate0, up0);
  await realizeTensor(h0);
  await realizeTensor(await fDown(h0, wd));

  const timeStage = async (stage, call, getSchedules) => {
    const samples = await timeAsync(call, (y) => completeOutput({ value: y }, "polygrad"), null);
    const s = median(samples);
    return {
      name, stage, seconds: s, us: s * 1e6,
      tokens_per_s: c.tokens / s, schedules: getSchedules(),
    };
  };

  const rows = [
    await timeStage("gate", () => fGate(x, wg), () => fGate.scheduleCount || 0),
    await timeStage("up", () => fUp(x, wu), () => fUp.scheduleCount || 0),
    await timeStage("mul", () => fMul(gate0, up0), () => fMul.scheduleCount || 0),
    await timeStage("down", () => fDown(h0, wd), () => fDown.scheduleCount || 0),
  ];

  fGate.dispose?.();
  fUp.dispose?.();
  fMul.dispose?.();
  fDown.dispose?.();
  return rows;
}

async function runJaxQwenStages(name, c, inputData) {
  if (c.kind !== "qwen_ffn") return [];
  const [x, wg, wu, wd] = createJaxInputs(c, inputData);
  await blockUntilReady([x, wg, wu, wd]);
  const fGate = jit((xx, ww) => nn.silu(np.matmul(xx.ref, ww.ref.transpose())), { device: args.device });
  const fUp = jit((xx, ww) => np.matmul(xx.ref, ww.ref.transpose()), { device: args.device });
  const fMul = jit((gate, up) => gate.ref.mul(up.ref), { device: args.device });
  const fDown = jit((h, ww) => np.matmul(h.ref, ww.ref.transpose()), { device: args.device });

  const gate0 = fGate(x.ref, wg.ref);
  const up0 = fUp(x.ref, wu.ref);
  await blockUntilReady([gate0, up0]);
  const h0 = fMul(gate0.ref, up0.ref);
  await h0.blockUntilReady();
  await fDown(h0.ref, wd.ref).blockUntilReady();

  const timeStage = async (stage, call) => {
    const samples = await timeAsync(call, (y) => completeOutput({ value: y }, "jax-js"), (y) => y.dispose());
    const s = median(samples);
    return { name, stage, seconds: s, us: s * 1e6, tokens_per_s: c.tokens / s };
  };

  const rows = [
    await timeStage("gate", () => fGate(x.ref, wg.ref)),
    await timeStage("up", () => fUp(x.ref, wu.ref)),
    await timeStage("mul", () => fMul(gate0.ref, up0.ref)),
    await timeStage("down", () => fDown(h0.ref, wd.ref)),
  ];

  fGate.dispose?.();
  fUp.dispose?.();
  fMul.dispose?.();
  fDown.dispose?.();
  gate0.dispose();
  up0.dispose();
  h0.dispose();
  for (const arr of [x, wg, wu, wd]) arr.dispose();
  return rows;
}

function printSummary(pgResults, jaxResults) {
  const brief = rows => rows.map(({ profiles, ...row }) => ({ ...row,
    profiles: profiles?.map(({ dispatches, ...profile }) => ({ ...profile, kernels: dispatches.length })) }));
  console.log(JSON.stringify({ backend: "polygrad-browser-" + args.device + "-model-jit", results: brief(pgResults) }));
  console.log(JSON.stringify({ backend: "jax-js-browser-" + args.device + "-model-jit", results: brief(jaxResults) }));
  console.log("\\ncase,polygrad_us,jax_js_us,ratio_pg_over_jax,polygrad_tok_s,jax_js_tok_s,schedules");
  for (const pg of pgResults) {
    const jx = jaxResults.find((x) => x.name === pg.name);
    if (!jx) continue;
    console.log([
      pg.name,
      pg.us.toFixed(3),
      jx.us.toFixed(3),
      (pg.us / jx.us).toFixed(3),
      pg.tokens_per_s.toFixed(3),
      jx.tokens_per_s.toFixed(3),
      pg.schedules,
    ].join(","));
  }
}

function printStageSummary(pgRows, jaxRows) {
  if (!pgRows.length && !jaxRows.length) return;
  console.log(JSON.stringify({ backend: "polygrad-browser-" + args.device + "-qwen-ffn-stages", results: pgRows }));
  console.log(JSON.stringify({ backend: "jax-js-browser-" + args.device + "-qwen-ffn-stages", results: jaxRows }));
  console.log("\\ncase,stage,polygrad_us,jax_js_us,ratio_pg_over_jax,polygrad_tok_s,jax_js_tok_s,schedules");
  for (const pg of pgRows) {
    const jx = jaxRows.find((x) => x.name === pg.name && x.stage === pg.stage);
    if (!jx) continue;
    console.log([
      pg.name,
      pg.stage,
      pg.us.toFixed(3),
      jx.us.toFixed(3),
      (pg.us / jx.us).toFixed(3),
      pg.tokens_per_s.toFixed(3),
      jx.tokens_per_s.toFixed(3),
      pg.schedules,
    ].join(","));
  }
}

async function main() {
  const env = {
    crossOriginIsolated,
    hasSharedArrayBuffer: typeof SharedArrayBuffer !== "undefined",
    hasWorker: typeof NativeWorker !== "undefined",
    hardwareConcurrency: navigator.hardwareConcurrency || 0,
  };
  console.log(JSON.stringify({ browser_env: env, args }));

  const pg = await polygrad[args.async ? "createAsync" : "create"]({
    core: "wasm", device: gpu ? "webgpu" : "auto", kernels: args.kernels,
    env: args.kernels ? { POLY_DEBUG_KERNELS: "1" } : {},
  });
  if (!args.polygradOnly) {
    const devices = await init(args.device);
    if (!devices.includes(args.device)) throw new Error("jax-js requested device unavailable: " + args.device);
    defaultDevice(args.device);
  }

  const inputsByCase = new Map();
  for (const name of args.cases) inputsByCase.set(name, caseInputs(CASES[name]));

  const pgResults = [];
  const workerCreatesBeforeJax = workerCreates;
  const jaxResults = [];
  for (const name of args.cases) {
    const [p, j] = await runPairedCase(pg, name, CASES[name], inputsByCase.get(name));
    pgResults.push(p);
    if (j) jaxResults.push(j);
  }
  const workerCreatesDuringJax = workerCreates - workerCreatesBeforeJax;

  if (!args.polygradOnly) printSummary(pgResults, jaxResults);
  console.log(JSON.stringify({ jaxWorkerCreates: workerCreatesDuringJax }));

  let pgStageResults = [];
  let jaxStageResults = [];
  let stageWorkerCreatesDuringJax = 0;
  if (args.stageBreakdown) {
    for (const name of args.cases) {
      pgStageResults.push(...await runPolyQwenStages(pg, name, CASES[name], inputsByCase.get(name)));
    }
    const stageWorkerCreatesBeforeJax = workerCreates;
    if (!args.polygradOnly) for (const name of args.cases) {
      jaxStageResults.push(...await runJaxQwenStages(name, CASES[name], inputsByCase.get(name)));
    }
    stageWorkerCreatesDuringJax = workerCreates - stageWorkerCreatesBeforeJax;
    printStageSummary(pgStageResults, jaxStageResults);
    console.log(JSON.stringify({ jaxStageWorkerCreates: stageWorkerCreatesDuringJax }));
  }

  const results = {
    args, env, gpu: gpu?.metadata(), workerCreatesDuringJax, stageWorkerCreatesDuringJax,
    pgResults, jaxResults, pgStageResults, jaxStageResults,
  };
  await pg.dispose?.();
  boundaries?.restore();
  window.__benchResults = results;
}

main().catch((e) => {
  console.error(e.stack || e.message || String(e));
  window.__benchResults = { error: e.message || String(e), stack: e.stack || "" };
});
</script>`;
}

function safePath(urlPath) {
  const clean = decodeURIComponent(urlPath.split("?")[0]);
  if (clean === "/") return null;
  const filePath = path.join(root, clean);
  if (!filePath.startsWith(root + path.sep)) return false;
  return filePath;
}

const server = http.createServer((req, res) => {
  const headers = {
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp",
    "Cross-Origin-Resource-Policy": "same-origin",
  };
  const filePath = safePath(req.url || "/");
  if (filePath === false) {
    res.writeHead(403, headers);
    res.end("forbidden");
    return;
  }
  if (filePath === null) {
    res.writeHead(200, { ...headers, "Content-Type": "text/html" });
    res.end(makePage());
    return;
  }
  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404, headers);
      res.end("not found");
      return;
    }
    const ext = path.extname(filePath);
    res.writeHead(200, { ...headers, "Content-Type": MIME[ext] || "application/octet-stream" });
    res.end(data);
  });
});

async function main() {
  const fingerprint = file => ({ path: file, sha256: createHash("sha256").update(fs.readFileSync(file)).digest("hex") });
  const report = {
    schema: 2, createdAt: new Date().toISOString(), args: { ...args },
    statistic: args.kernelQualify
      ? 'kernel controls aggregate paired blocks per fresh page; whole-call means are secondary observations'
      : 'mean timed call duration per fresh page; confidence intervals use page pairs only',
    machine: { platform: process.platform, arch: process.arch, cpu: os.cpus()[0]?.model, node: process.version, loadBefore: os.loadavg() },
    artifacts: [...new Set([args.bundle, args.baseline].filter(Boolean))].map(p => fingerprint(path.resolve(root, decodeURIComponent(p)))),
    reference: args.polygradOnly ? null : fingerprint(path.join(root, "references/jax-js/dist/index.js")),
    harness: [__filename, path.join(__dirname, "wasm_checks.mjs"), path.join(__dirname, "webgpu_profile.mjs"),
      path.join(root, "js/test/test_jit_boundaries.js")].map(fingerprint),
    pages: [],
  };
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const port = server.address().port;
  let browser;
  try {
    browser = await chromium.launch(args.device === "webgpu" ?
      launchOptionsFor({ launcherName: "chromium" }, "webgpu").opts : { args: ["--no-sandbox"] });
    report.browser = browser.version();
    // ABBA pages in one browser isolate bundle identity from browser restarts.
    const candidate = { bundle: args.bundle, pipeline: args.pipeline };
    const baseline = { bundle: args.baseline ?? args.bundle, pipeline: args.baselinePipeline ?? args.pipeline };
    const lanes = args.kernelQualify ? Array.from({ length: 20 }, (_, kernelPage) => ({ ...candidate, kernelPage }))
      : args.qualify ? Array.from({ length: args.pagePairs }, (_, pair) =>
      ['identity', 'kernel', 'host'].map((_, i, controls) => controls[(i + pair) % controls.length]).flatMap(control =>
        pagePairOrder(args.pagePairs).slice(pair * 2, pair * 2 + 2).map(p => ({ ...baseline, ...p, control })))).flat()
      : args.pagePairs !== null ? pagePairOrder(args.pagePairs).map(p => ({
      ...(p.lane === 'baseline' ? baseline : candidate), ...p,
    })) : args.baseline || args.baselinePipeline ? [baseline, candidate, candidate, baseline] : [candidate];
    for (const lane of lanes) {
      Object.assign(args, lane);
      const page = await browser.newPage();
      page.on("console", msg => console.log(msg.text()));
      page.on("pageerror", err => console.error("PAGE ERROR:", err.message));
      try {
        await page.goto("http://127.0.0.1:" + port + "/", { waitUntil: "load" });
        const timeout = Math.max(180000, (args.iters + args.warmup) * args.rounds * args.cases.length * 30000);
        const results = await page.waitForFunction(() => window.__benchResults, undefined, { timeout });
        const value = await results.jsonValue();
        if (args.kernelQualify) value.kernelPage = lane.kernelPage;
        if (args.pagePairs !== null) { value.lane = lane.lane; value.pair = lane.pair; value.control = lane.control; }
        report.pages.push(value);
        if (value?.error) throw new Error(value.error + "\\n" + (value.stack || ""));
      } finally {
        await page.close();
      }
    }
    if (args.kernelQualify) {
      report.kernelQualification = assessKernelPages(report.pages);
      report.kernelQualification.protocol = { pages: 20, blocks: 8, targetBlockMs: 5,
        identityMargin: .05, injectionFraction: .1, detectionMargin: .03,
        unitScope: 'dispatches repeated hot, plus complete trace in original order; actual retained buffers',
        wholeCallTiming: 'secondary observations only; not qualified by kernel controls' };
      process.exitCode = report.kernelQualification.qualified ? 0 : 2;
      console.log(JSON.stringify({ kernelQualified: report.kernelQualification.qualified,
        results: report.kernelQualification.results }));
    } else if (args.qualify) {
      report.qualification = Object.fromEntries(['identity', 'kernel', 'host'].map(control => [control,
        assessPagePairs(report.pages.filter(p => p.control === control), control === 'identity' ? .05 : .03, .05 / 3)]));
      report.qualified = Object.entries(report.qualification).every(([control, results]) =>
        results.every(r => r.status === (control === 'identity' ? 'equivalent' : 'regression')));
      process.exitCode = report.qualified ? 0 : 2;
      console.log(JSON.stringify({ qualified: report.qualified, qualification: report.qualification }));
    } else if (args.baseline || args.baselinePipeline) {
      report.comparison = args.pagePairs !== null
        ? assessPagePairs(report.pages, args.maxSlowdown ?? 0.10) : assessABBA(report.pages, args.maxSlowdown ?? 0.10);
      console.log(JSON.stringify({ comparison: report.comparison }));
      if (args.maxSlowdown !== null) {
        process.exitCode = report.comparison.some(r => r.status === "regression") ? 1
          : report.comparison.some(r => r.status === "inconclusive") ? 2 : 0;
      }
    }
    if (args.structure) {
      const disassembler = process.env.WASM_DIS ?? 'wasm-dis';
      report.disassembler = execFileSync(disassembler, ['--version'], { encoding: 'utf8' }).trim();
      for (const p of report.pages) for (const r of p.pgResults) {
        r.structure.cold.modules = r.structure.cold.modules.map(data => {
          const bytes = Buffer.from(data);
          const wat = execFileSync(disassembler, ['-', '--all-features'], { input: bytes, maxBuffer: 32 * 1024 * 1024 }).toString();
          return { sha256: createHash('sha256').update(bytes).digest('hex'), bytes: bytes.length,
            opcodes: decodedWatOpcodes(wat) };
        });
      }
      report.snapshot = structuralSnapshot(report);
      if (args.snapshot) {
        if (args.updateSnapshot) {
          fs.writeFileSync(args.snapshot, JSON.stringify(report.snapshot, null, 2) + '\n');
          report.snapshotStatus = 'updated-unreviewed';
          console.log('Snapshot updated; review its diff and run timing qualification before accepting it.');
        } else {
          const expected = JSON.parse(fs.readFileSync(args.snapshot, 'utf8'));
          report.snapshotStatus = JSON.stringify(expected) === JSON.stringify(report.snapshot) ? 'match' : 'changed-review-required';
          if (report.snapshotStatus !== 'match') process.exitCode = 1;
          console.log('Structural snapshot: ' + report.snapshotStatus);
        }
      }
    }
  } catch (error) {
    report.error = error.stack || String(error);
    throw error;
  } finally {
    await browser?.close();
    server.close();
    report.machine.loadAfter = os.loadavg();
    fs.mkdirSync(path.dirname(args.output), { recursive: true });
    fs.writeFileSync(args.output, JSON.stringify(report, null, 2) + "\n", { flag: "wx" });
    console.log("Performance report: " + args.output);
  }
}

await main();
