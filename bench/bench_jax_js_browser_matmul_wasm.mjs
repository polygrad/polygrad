import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "../js/node_modules/playwright/index.mjs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const root = path.resolve(__dirname, "..");

function parseArgs(argv) {
  const args = {
    sizes: [64, 128, 256, 512, 1024, 2048],
    iters: 5,
    warmup: 2,
    largeIters: 3,
  };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    const next = () => {
      if (i + 1 >= argv.length) throw new Error("missing value for " + a);
      return argv[++i];
    };
    if (a === "--sizes") args.sizes = next().split(",").map((x) => Number(x.trim())).filter(Boolean);
    else if (a === "--iters") args.iters = Number(next());
    else if (a === "--warmup") args.warmup = Number(next());
    else if (a === "--large-iters") args.largeIters = Number(next());
    else if (a === "--paired" || a === "--quiet" || a === "--no-openblas") {
      // Accepted for Makefile BENCH_JAX_JS_EXTRA parity with the Node matmul bench.
    } else if (a === "--rounds" || a === "--polygrad-max-size" || a === "--blas-threads") {
      next();
    } else if (a === "--help" || a === "-h") {
      console.log("Usage: node bench/bench_jax_js_browser_matmul_wasm.mjs [--sizes 64,128] [--iters N] [--warmup N] [--large-iters N]");
      process.exit(0);
    } else {
      throw new Error("unknown option " + a);
    }
  }
  if (!args.sizes.length) throw new Error("--sizes must include at least one size");
  for (const n of args.sizes) {
    if (!Number.isFinite(n) || n <= 0) throw new Error("invalid size " + n);
  }
  if (!Number.isFinite(args.iters) || args.iters <= 0) throw new Error("--iters must be positive");
  if (!Number.isFinite(args.largeIters) || args.largeIters <= 0) throw new Error("--large-iters must be positive");
  if (!Number.isFinite(args.warmup) || args.warmup < 0) throw new Error("--warmup must be nonnegative");
  args.iters = Math.trunc(args.iters);
  args.warmup = Math.trunc(args.warmup);
  args.largeIters = Math.trunc(args.largeIters);
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
<title>polygrad vs jax-js browser wasm matmul</title>
<pre id="log"></pre>
<script src="/js/dist/polygrad.js"></script>
<script type="module">
import {
  blockUntilReady,
  defaultDevice,
  init,
  jit,
  numpy as np,
} from "/references/jax-js/dist/index.js";

const args = ${JSON.stringify(args)};

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

function makeMatrixData(n) {
  const data = new Float32Array(n * n);
  for (let i = 0; i < data.length; i++) data[i] = (i % 7) - 3;
  return data;
}

function median(xs) {
  const ys = [...xs].sort((a, b) => a - b);
  return ys[Math.floor(ys.length / 2)];
}

function gflops(n, seconds) {
  return (2 * n * n * n) / seconds / 1e9;
}

function samplesFor(n) {
  return n <= 1024 ? args.iters : args.largeIters;
}

async function timeAsync(n, call, ready, dispose) {
  let y = await call();
  await ready(y);
  dispose?.(y);
  for (let i = 0; i < args.warmup; i++) {
    y = await call();
    await ready(y);
    dispose?.(y);
  }
  const times = [];
  const iters = samplesFor(n);
  for (let i = 0; i < iters; i++) {
    const t0 = performance.now();
    y = await call();
    await ready(y);
    times.push((performance.now() - t0) / 1000);
    dispose?.(y);
  }
  return median(times);
}

async function primePolygradJit(fn, a, b) {
  await (await fn(a, b)).realize();
  await (await fn(a, b)).realize();
}

async function runPolygrad() {
  const pg = await polygrad.create({ core: "wasm", device: "auto" });
  const { Tensor } = pg;
  const out = [];
  for (const n of args.sizes) {
    const a = new Tensor(makeMatrixData(n)).reshape(n, n);
    const b = new Tensor(makeMatrixData(n)).reshape(n, n);
    await a.realize(b);
    const fAB = pg.jit((aa, bb) => aa.matmul(bb));
    const fBT = pg.jit((aa, bb) => aa.matmul(bb.permute(1, 0)));
    await primePolygradJit(fAB, a, b);
    await primePolygradJit(fBT, a, b);
    const tAB = await timeAsync(n, () => fAB(a, b), (y) => y.realize(), null);
    const tBT = await timeAsync(n, () => fBT(a, b), (y) => y.realize(), null);
    out.push({
      n,
      poly_ab: gflops(n, tAB),
      poly_bt: gflops(n, tBT),
      poly_ab_us: tAB * 1e6,
      poly_bt_us: tBT * 1e6,
      poly_ab_schedules: fAB.scheduleCount,
      poly_bt_schedules: fBT.scheduleCount,
    });
    fAB.dispose?.();
    fBT.dispose?.();
  }
  await pg.dispose?.();
  return out;
}

async function runJax() {
  await init("wasm");
  defaultDevice("wasm");
  const out = [];
  for (const n of args.sizes) {
    const a = np.array(makeMatrixData(n), { shape: [n, n], device: "wasm" });
    const b = np.array(makeMatrixData(n), { shape: [n, n], device: "wasm" });
    await blockUntilReady([a, b]);
    const tAB = await timeAsync(n, () => np.matmul(a.ref, b.ref), (y) => y.blockUntilReady(), (y) => y.dispose());
    const tBT = await timeAsync(n, () => np.matmul(a.ref, b.ref.transpose()), (y) => y.blockUntilReady(), (y) => y.dispose());
    out.push({
      n,
      jax_ab: gflops(n, tAB),
      jax_bt: gflops(n, tBT),
      jax_ab_us: tAB * 1e6,
      jax_bt_us: tBT * 1e6,
    });
    a.dispose();
    b.dispose();
  }
  return out;
}

function mergeRows(pgRows, jaxRows) {
  return args.sizes.map((n) => ({
    n,
    ...(jaxRows.find((r) => r.n === n) || {}),
    ...(pgRows.find((r) => r.n === n) || {}),
  }));
}

function printSummary(rows, pgRows, jaxRows) {
  console.log(JSON.stringify({ backend: "polygrad-browser-wasm-matmul-jit", results: pgRows }));
  console.log(JSON.stringify({ backend: "jax-js-browser-wasm-matmul", results: jaxRows }));
  console.log("\\nn,jax_ab_gfs,jax_abt_gfs,poly_ab_gfs,poly_abt_gfs,poly_ab_over_jax,poly_abt_over_jax");
  for (const r of rows) {
    console.log([
      r.n,
      r.jax_ab?.toFixed(3) || "",
      r.jax_bt?.toFixed(3) || "",
      r.poly_ab?.toFixed(3) || "",
      r.poly_bt?.toFixed(3) || "",
      r.jax_ab && r.poly_ab ? (r.poly_ab / r.jax_ab).toFixed(3) : "",
      r.jax_bt && r.poly_bt ? (r.poly_bt / r.jax_bt).toFixed(3) : "",
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
  const pgRows = await runPolygrad();
  const workerCreatesBeforeJax = workerCreates;
  const jaxRows = await runJax();
  const workerCreatesDuringJax = workerCreates - workerCreatesBeforeJax;
  const rows = mergeRows(pgRows, jaxRows);
  printSummary(rows, pgRows, jaxRows);
  console.log(JSON.stringify({ jaxWorkerCreates: workerCreatesDuringJax }));
  window.__benchResults = { env, workerCreatesDuringJax, rows, pgRows, jaxRows };
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
    const timeout = Math.max(180000, (args.iters + args.warmup + args.largeIters) * args.sizes.length * 20000);
    const results = await page.waitForFunction(() => window.__benchResults, undefined, { timeout });
    const value = await results.jsonValue();
    if (value?.error) throw new Error(value.error + "\\n" + (value.stack || ""));
  } finally {
    await browser.close();
    server.close();
  }
}

await main();
