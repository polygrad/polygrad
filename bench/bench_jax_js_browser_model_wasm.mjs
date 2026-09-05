import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "../js/node_modules/playwright/index.mjs";

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
    stageBreakdown: false,
  };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    const next = () => {
      if (i + 1 >= argv.length) throw new Error("missing value for " + a);
      return argv[++i];
    };
    if (a === "--cases") args.cases = next().split(",").map((x) => x.trim()).filter(Boolean);
    else if (a === "--iters") args.iters = Number(next());
    else if (a === "--warmup") args.warmup = Number(next());
    else if (a === "--stage-breakdown") args.stageBreakdown = true;
    else if (a === "--help" || a === "-h") {
      console.log("Usage: node bench/bench_jax_js_browser_model_wasm.mjs [--cases a,b] [--iters N] [--warmup N] [--stage-breakdown]");
      process.exit(0);
    } else if (a === "--paired" || a === "--quiet") {
      // Accepted for Makefile BENCH_JAX_JS_EXTRA parity with the Node model bench.
    } else if (a === "--rounds") {
      next();
    } else {
      throw new Error("unknown option " + a);
    }
  }
  for (const name of args.cases) {
    if (!CASES[name]) throw new Error("unknown case '" + name + "'");
  }
  if (!Number.isFinite(args.iters) || args.iters <= 0) throw new Error("--iters must be positive");
  if (!Number.isFinite(args.warmup) || args.warmup < 0) throw new Error("--warmup must be nonnegative");
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
<script src="/js/dist/polygrad.sync.js"></script>
<script type="module">
import { checkWorkload, modelReference } from "/bench/wasm_checks.mjs";
import {
  blockUntilReady,
  defaultDevice,
  init,
  jit,
  nn,
  numpy as np,
} from "/references/jax-js/dist/index.js";

const args = ${JSON.stringify(args)};
const CASES = ${JSON.stringify(CASES)};

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
  let y = await call();
  await ready(y);
  dispose?.(y);
  for (let i = 0; i < args.warmup; i++) {
    y = await call();
    await ready(y);
    dispose?.(y);
  }
  const times = [];
  for (let i = 0; i < args.iters; i++) {
    const t0 = performance.now();
    y = await call();
    await ready(y);
    times.push((performance.now() - t0) / 1000);
    dispose?.(y);
  }
  return times;
}

function createJaxInputs(c, inputData) {
  if (c.kind === "mlp") {
    return [
      np.array(inputData.x, { shape: [c.tokens, c.d], device: "wasm" }),
      np.array(inputData.w1, { shape: [c.hidden, c.d], device: "wasm" }),
      np.array(inputData.b1, { shape: [1, c.hidden], device: "wasm" }),
      np.array(inputData.w2, { shape: [c.out, c.hidden], device: "wasm" }),
      np.array(inputData.b2, { shape: [1, c.out], device: "wasm" }),
    ];
  }
  return [
    np.array(inputData.x, { shape: [c.tokens, c.d], device: "wasm" }),
    np.array(inputData.wg, { shape: [c.hidden, c.d], device: "wasm" }),
    np.array(inputData.wu, { shape: [c.hidden, c.d], device: "wasm" }),
    np.array(inputData.wd, { shape: [c.d, c.hidden], device: "wasm" }),
  ];
}

function createJaxPipeline(c) {
  if (c.kind === "mlp") {
    const f1 = jit((x, w1, b1) => {
      const pre = np.matmul(x.ref, w1.ref.transpose()).ref.add(b1.ref);
      return c.activation === "silu" ? nn.silu(pre) : np.maximum(pre, 0);
    }, { device: "wasm" });
    const f2 = jit((h, w2, b2) => np.matmul(h.ref, w2.ref.transpose()).ref.add(b2.ref), { device: "wasm" });
    return {
      call(inputs) {
        const [x, w1, b1, w2, b2] = inputs;
        const h = f1(x.ref, w1.ref, b1.ref);
        const y = f2(h.ref, w2.ref, b2.ref);
        return { value: y, temps: [h] };
      },
      async ready(out) { await out.value.blockUntilReady(); },
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
  const fGate = jit((x, wg) => nn.silu(np.matmul(x.ref, wg.ref.transpose())), { device: "wasm" });
  const fUp = jit((x, wu) => np.matmul(x.ref, wu.ref.transpose()), { device: "wasm" });
  const fMul = jit((gate, up) => gate.ref.mul(up.ref), { device: "wasm" });
  const fDown = jit((h, wd) => np.matmul(h.ref, wd.ref.transpose()), { device: "wasm" });
  return {
    call(inputs) {
      const [x, wg, wu, wd] = inputs;
      const gate = fGate(x.ref, wg.ref);
      const up = fUp(x.ref, wu.ref);
      const h = fMul(gate.ref, up.ref);
      const y = fDown(h.ref, wd.ref);
      return { value: y, temps: [gate, up, h] };
    },
    async ready(out) { await out.value.blockUntilReady(); },
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
    await inputs[0].realize(...inputs.slice(1));
    return inputs;
  }
  const inputs = [
    new Tensor(inputData.x).reshape(c.tokens, c.d),
    new Tensor(inputData.wg).reshape(c.hidden, c.d),
    new Tensor(inputData.wu).reshape(c.hidden, c.d),
    new Tensor(inputData.wd).reshape(c.d, c.hidden),
  ];
  await inputs[0].realize(...inputs.slice(1));
  return inputs;
}

function createPolyPipeline(pg, c) {
  if (c.kind === "mlp") {
    const f1 = pg.jit((x, w1, b1) => {
      const pre = x.matmul(w1.permute(1, 0)).add(b1);
      return c.activation === "silu" ? pre.silu() : pre.relu();
    });
    const f2 = pg.jit((h, w2, b2) => h.matmul(w2.permute(1, 0)).add(b2));
    return {
      async call(inputs) {
        const [x, w1, b1, w2, b2] = inputs;
        const h = await f1(x, w1, b1);
        await h.realize();
        const y = await f2(h, w2, b2);
        return { value: y };
      },
      async ready(out) { await out.value.realize(); },
      disposeOutput(_) {},
      dispose() {
        f1.dispose?.();
        f2.dispose?.();
      },
      get scheduleCount() { return (f1.scheduleCount || 0) + (f2.scheduleCount || 0); },
    };
  }
  const fGate = pg.jit((x, wg) => x.matmul(wg.permute(1, 0)).silu());
  const fUp = pg.jit((x, wu) => x.matmul(wu.permute(1, 0)));
  const fMul = pg.jit((gate, up) => gate.mul(up));
  const fDown = pg.jit((h, wd) => h.matmul(wd.permute(1, 0)));
  return {
    async call(inputs) {
      const [x, wg, wu, wd] = inputs;
      const gate = await fGate(x, wg);
      const up = await fUp(x, wu);
      await gate.realize(up);
      const h = await fMul(gate, up);
      await h.realize();
      const y = await fDown(h, wd);
      return { value: y };
    },
    async ready(out) { await out.value.realize(); },
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

async function runPolyCase(pg, name, c, inputData) {
  const inputs = await createPolyInputs(pg, c, inputData);
  const pipeline = createPolyPipeline(pg, c);
  await primePipeline(pipeline, inputs);
  await checkWorkload({ name, call: () => pipeline.call(inputs), ready: y => pipeline.ready(y), dispose: y => pipeline.disposeOutput(y) }, modelReference(c,inputData), {atol:c.kind==='mlp'?1e-7:1e-10,rtol:2e-4});
  const samples = await timeAsync(() => pipeline.call(inputs), (y) => pipeline.ready(y), (y) => pipeline.disposeOutput(y));
  const s = median(samples);
  const schedules = pipeline.scheduleCount;
  pipeline.dispose();
  return {
    name, label: caseLabel(c), seconds: s, us: s * 1e6,
    tokens_per_s: c.tokens / s, gflops: modelFlops(c) / s / 1e9, schedules,
  };
}

async function runJaxCase(name, c, inputData) {
  const inputs = createJaxInputs(c, inputData);
  await blockUntilReady(inputs);
  const pipeline = createJaxPipeline(c);
  await primePipeline(pipeline, inputs);
  await checkWorkload({ name, call: () => pipeline.call(inputs), ready: y => pipeline.ready(y), dispose: y => pipeline.disposeOutput(y) }, modelReference(c,inputData), {atol:c.kind==='mlp'?1e-7:1e-10,rtol:2e-4});
  const samples = await timeAsync(() => pipeline.call(inputs), (y) => pipeline.ready(y), (y) => pipeline.disposeOutput(y));
  const s = median(samples);
  pipeline.dispose();
  for (const x of inputs) x.dispose();
  return {
    name, label: caseLabel(c), seconds: s, us: s * 1e6,
    tokens_per_s: c.tokens / s, gflops: modelFlops(c) / s / 1e9,
  };
}

async function runPolyQwenStages(pg, name, c, inputData) {
  if (c.kind !== "qwen_ffn") return [];
  const [x, wg, wu, wd] = await createPolyInputs(pg, c, inputData);
  const fGate = pg.jit((xx, ww) => xx.matmul(ww.permute(1, 0)).silu());
  const fUp = pg.jit((xx, ww) => xx.matmul(ww.permute(1, 0)));
  const fMul = pg.jit((gate, up) => gate.mul(up));
  const fDown = pg.jit((h, ww) => h.matmul(ww.permute(1, 0)));

  const gate0 = await fGate(x, wg);
  const up0 = await fUp(x, wu);
  await gate0.realize(up0);
  const h0 = await fMul(gate0, up0);
  await h0.realize();
  await (await fDown(h0, wd)).realize();

  const timeStage = async (stage, call, getSchedules) => {
    const samples = await timeAsync(call, (y) => y.realize(), null);
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
  const fGate = jit((xx, ww) => nn.silu(np.matmul(xx.ref, ww.ref.transpose())), { device: "wasm" });
  const fUp = jit((xx, ww) => np.matmul(xx.ref, ww.ref.transpose()), { device: "wasm" });
  const fMul = jit((gate, up) => gate.ref.mul(up.ref), { device: "wasm" });
  const fDown = jit((h, ww) => np.matmul(h.ref, ww.ref.transpose()), { device: "wasm" });

  const gate0 = fGate(x.ref, wg.ref);
  const up0 = fUp(x.ref, wu.ref);
  await blockUntilReady([gate0, up0]);
  const h0 = fMul(gate0.ref, up0.ref);
  await h0.blockUntilReady();
  await fDown(h0.ref, wd.ref).blockUntilReady();

  const timeStage = async (stage, call) => {
    const samples = await timeAsync(call, (y) => y.blockUntilReady(), (y) => y.dispose());
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
  console.log(JSON.stringify({ backend: "polygrad-browser-wasm-model-jit", results: pgResults }));
  console.log(JSON.stringify({ backend: "jax-js-browser-wasm-model-jit", results: jaxResults }));
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
  console.log(JSON.stringify({ backend: "polygrad-browser-wasm-qwen-ffn-stages", results: pgRows }));
  console.log(JSON.stringify({ backend: "jax-js-browser-wasm-qwen-ffn-stages", results: jaxRows }));
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
  console.log(JSON.stringify({ browser_env: env }));

  const pg = await polygrad.create({ core: "wasm", device: "auto" });
  await init("wasm");
  defaultDevice("wasm");

  const inputsByCase = new Map();
  for (const name of args.cases) inputsByCase.set(name, caseInputs(CASES[name]));

  const pgResults = [];
  for (const name of args.cases) pgResults.push(await runPolyCase(pg, name, CASES[name], inputsByCase.get(name)));

  const workerCreatesBeforeJax = workerCreates;
  const jaxResults = [];
  for (const name of args.cases) jaxResults.push(await runJaxCase(name, CASES[name], inputsByCase.get(name)));
  const workerCreatesDuringJax = workerCreates - workerCreatesBeforeJax;

  printSummary(pgResults, jaxResults);
  console.log(JSON.stringify({ jaxWorkerCreates: workerCreatesDuringJax }));

  let pgStageResults = [];
  let jaxStageResults = [];
  let stageWorkerCreatesDuringJax = 0;
  if (args.stageBreakdown) {
    for (const name of args.cases) {
      pgStageResults.push(...await runPolyQwenStages(pg, name, CASES[name], inputsByCase.get(name)));
    }
    const stageWorkerCreatesBeforeJax = workerCreates;
    for (const name of args.cases) {
      jaxStageResults.push(...await runJaxQwenStages(name, CASES[name], inputsByCase.get(name)));
    }
    stageWorkerCreatesDuringJax = workerCreates - stageWorkerCreatesBeforeJax;
    printStageSummary(pgStageResults, jaxStageResults);
    console.log(JSON.stringify({ jaxStageWorkerCreates: stageWorkerCreatesDuringJax }));
  }

  window.__benchResults = {
    env, workerCreatesDuringJax, stageWorkerCreatesDuringJax,
    pgResults, jaxResults, pgStageResults, jaxStageResults,
  };
  await pg.dispose?.();
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
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const port = server.address().port;
  const browser = await chromium.launch({ args: ["--no-sandbox"] });
  const page = await browser.newPage();
  page.on("console", (msg) => console.log(msg.text()));
  page.on("pageerror", (err) => console.error("PAGE ERROR:", err.message));
  try {
    await page.goto("http://127.0.0.1:" + port + "/", { waitUntil: "load" });
    const timeout = Math.max(180000, (args.iters + args.warmup) * args.cases.length * 30000);
    const results = await page.waitForFunction(() => window.__benchResults, undefined, { timeout });
    const value = await results.jsonValue();
    if (value?.error) throw new Error(value.error + "\\n" + (value.stack || ""));
  } finally {
    await browser.close();
    server.close();
  }
}

await main();
