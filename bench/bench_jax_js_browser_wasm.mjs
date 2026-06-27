import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "../js/node_modules/playwright/index.mjs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const root = path.resolve(__dirname, "..");
const iters = Number(process.argv[2] || 30);
const warmup = Number(process.argv[3] || 8);

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
<title>polygrad vs jax-js browser wasm</title>
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

const iters = ${iters};
const warmup = ${warmup};
let workerCreates = 0;
const NativeWorker = globalThis.Worker;
if (typeof NativeWorker !== "undefined") {
  function CountingWorker(...args) {
    workerCreates++;
    return new NativeWorker(...args);
  }
  CountingWorker.prototype = NativeWorker.prototype;
  globalThis.Worker = CountingWorker;
}

function make1(n, mod, div, shift) {
  const data = new Float32Array(n);
  for (let i = 0; i < n; i++) data[i] = (i % mod) / div - shift;
  return data;
}

function makeLin(n, lo, hi) {
  const data = new Float32Array(n);
  const step = (hi - lo) / (n - 1);
  for (let i = 0; i < n; i++) data[i] = lo + i * step;
  return data;
}

function median(xs) {
  const ys = [...xs].sort((a, b) => a - b);
  return ys[Math.floor(ys.length / 2)];
}

async function bench(name, call, ready, dispose) {
  let y = await call();
  await ready(y);
  if (dispose) dispose(y);

  for (let i = 0; i < warmup; i++) {
    y = await call();
    await ready(y);
    if (dispose) dispose(y);
  }

  const times = [];
  for (let i = 0; i < iters; i++) {
    const t0 = performance.now();
    y = await call();
    await ready(y);
    const t1 = performance.now();
    if (dispose) dispose(y);
    times.push((t1 - t0) * 1000);
  }
  return { name, median_us: median(times), min_us: Math.min(...times), iters };
}

async function runPolygrad(inputs) {
  const pg = await polygrad.create({ core: "wasm", device: "auto" });
  const { Tensor } = pg;
  const [a0, b0, c0, d0, x20, row0, col0] = inputs;
  const a = new Tensor(a0);
  const b = new Tensor(b0);
  const c = new Tensor(c0);
  const d = new Tensor(d0);
  const x2 = new Tensor(x20).reshape(1024, 1024);
  const row = new Tensor(row0).reshape(1024, 1);
  const col = new Tensor(col0).reshape(1, 1024);
  await a.realize(b, c, d, x2, row, col);

  const workloads = [
    ["pointwise_1m", pg.jit((aa, bb) => aa.add(bb).mul(aa.sub(bb)).add(aa.mul(bb)).relu()), [a, b]],
    [
      "where_1m",
      pg.jit((aa, bb, cc, dd) =>
        aa.gt(bb).where(aa.add(cc).mul(bb.sub(dd)), aa.sub(cc).mul(bb.add(dd)))
      ),
      [a, b, c, d],
    ],
    [
      "broadcast_reduce_1024",
      pg.jit((xx, rr, cc) => xx.add(rr).mul(cc).sub(0.25).relu().sum(1)),
      [x2, row, col],
    ],
    [
      "transpose_copy_1024",
      pg.jit((xx) => xx.reshape(512, 2048).permute(1, 0).contiguous()),
      [x2],
    ],
  ];

  const results = [];
  for (const [name, fn, args] of workloads) {
    results.push(await bench(name, () => fn(...args), async () => {}, null));
  }
  for (const [, fn] of workloads) fn.dispose?.();
  await pg.dispose?.();
  return results;
}

async function runJax(inputs) {
  await init("wasm");
  defaultDevice("wasm");
  const [a0, b0, c0, d0, x20, row0, col0] = inputs;
  const a = np.array(a0, { shape: [a0.length], device: "wasm" });
  const b = np.array(b0, { shape: [b0.length], device: "wasm" });
  const c = np.array(c0, { shape: [c0.length], device: "wasm" });
  const d = np.array(d0, { shape: [d0.length], device: "wasm" });
  const x2 = np.array(x20, { shape: [1024, 1024], device: "wasm" });
  const row = np.array(row0, { shape: [1024, 1], device: "wasm" });
  const col = np.array(col0, { shape: [1, 1024], device: "wasm" });
  await blockUntilReady({ a, b, c, d, x2, row, col });

  const workloads = [
    [
      "pointwise_1m",
      jit((aa, bb) => np.maximum(aa.ref.add(bb.ref).mul(aa.ref.sub(bb.ref)).add(aa.ref.mul(bb.ref)), 0), { device: "wasm" }),
      [a, b],
    ],
    [
      "where_1m",
      jit((aa, bb, cc, dd) =>
        np.where(
          np.greater(aa.ref, bb.ref),
          aa.ref.add(cc.ref).mul(bb.ref.sub(dd.ref)),
          aa.ref.sub(cc.ref).mul(bb.ref.add(dd.ref)),
        ),
        { device: "wasm" },
      ),
      [a, b, c, d],
    ],
    [
      "broadcast_reduce_1024",
      jit((xx, rr, cc) => np.sum(np.maximum(xx.ref.add(rr.ref).mul(cc.ref).sub(0.25), 0), 1), { device: "wasm" }),
      [x2, row, col],
    ],
    [
      "transpose_copy_1024",
      jit((xx) => np.reshape(xx.ref, [512, 2048]).transpose().add(0), { device: "wasm" }),
      [x2],
    ],
  ];

  const results = [];
  for (const [name, fn, args] of workloads) {
    results.push(await bench(name, () => fn(...args.map(x => x.ref)), y => y.blockUntilReady(), y => y.dispose()));
  }
  for (const [, fn] of workloads) fn.dispose?.();
  for (const x of [a, b, c, d, x2, row, col]) x.dispose();
  return results;
}

function printSummary(pgResults, jaxResults) {
  console.log(JSON.stringify({ backend: "polygrad-browser-wasm-jit", results: pgResults }));
  console.log(JSON.stringify({ backend: "jax-js-browser-wasm-jit", results: jaxResults }));
  console.log("\\ncase,polygrad_us,jax_js_us,ratio_pg_over_jax");
  for (const pg of pgResults) {
    const jx = jaxResults.find(x => x.name === pg.name);
    if (!jx) continue;
    console.log(\`\${pg.name},\${pg.median_us.toFixed(3)},\${jx.median_us.toFixed(3)},\${(pg.median_us / jx.median_us).toFixed(3)}\`);
  }
}

async function main() {
  const n = 1 << 20;
  const inputs = [
    make1(n, 1021, 257, 2),
    make1(n, 509, 127, 1.5),
    make1(n, 251, 63, -0.25),
    make1(n, 127, 31, 0.75),
    make1(1024 * 1024, 997, 211, 2),
    makeLin(1024, -0.5, 0.5),
    makeLin(1024, 0.75, 1.25),
  ];

  const env = {
    crossOriginIsolated,
    hasSharedArrayBuffer: typeof SharedArrayBuffer !== "undefined",
    hasWorker: typeof NativeWorker !== "undefined",
    hardwareConcurrency: navigator.hardwareConcurrency || 0,
  };
  console.log(JSON.stringify({ browser_env: env }));
  const pgResults = await runPolygrad(inputs);
  const workerCreatesBeforeJax = workerCreates;
  const jaxResults = await runJax(inputs);
  const workerCreatesDuringJax = workerCreates - workerCreatesBeforeJax;
  printSummary(pgResults, jaxResults);
  window.__benchResults = { env, workerCreatesDuringJax, pgResults, jaxResults };
  console.log(JSON.stringify({ jaxWorkerCreates: workerCreatesDuringJax }));
}

main().catch(e => {
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
  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  const port = server.address().port;
  const browser = await chromium.launch({ args: ["--no-sandbox"] });
  const page = await browser.newPage();
  page.on("console", msg => console.log(msg.text()));
  page.on("pageerror", err => console.error("PAGE ERROR:", err.message));
  try {
    await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: "load" });
    const timeout = Math.max(120000, (iters + warmup) * 8000);
    const results = await page.waitForFunction(
      () => window.__benchResults,
      { timeout },
    ).then(h => h.jsonValue());
    if (results.error) {
      console.error(results.stack || results.error);
      process.exitCode = 1;
    }
  } finally {
    await browser.close();
    server.close();
  }
}

main().catch(e => {
  console.error(e.stack || e.message || String(e));
  server.close();
  process.exit(1);
});
