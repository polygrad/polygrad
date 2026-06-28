import { performance } from "node:perf_hooks";
import polygrad from "../js/src/index.js";
import {
  blockUntilReady,
  defaultDevice,
  init,
  jit,
  numpy as np,
} from "../references/jax-js/dist/index.js";

function parseArgs(argv) {
  const out = {
    iters: 40,
    warmup: 10,
    rounds: 1,
    paired: false,
    json: false,
    progress: true,
  };
  const positional = [];
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--iters") {
      out.iters = Number(argv[++i]);
    } else if (a === "--warmup") {
      out.warmup = Number(argv[++i]);
    } else if (a === "--rounds") {
      out.rounds = Number(argv[++i]);
    } else if (a === "--paired") {
      out.paired = true;
    } else if (a === "--quiet") {
      out.progress = false;
    } else if (a === "--json") {
      out.json = true;
    } else if (a === "--help" || a === "-h") {
      console.log(`Usage: node bench/bench_jax_js_wasm.mjs [iters warmup]
       node bench/bench_jax_js_wasm.mjs --iters N --warmup N [--rounds N] [--paired] [--json]`);
      process.exit(0);
    } else {
      positional.push(a);
    }
  }
  if (positional[0] !== undefined) out.iters = Number(positional[0]);
  if (positional[1] !== undefined) out.warmup = Number(positional[1]);
  if (!Number.isFinite(out.iters) || out.iters <= 0) throw new Error("invalid iters");
  if (!Number.isFinite(out.warmup) || out.warmup < 0) throw new Error("invalid warmup");
  if (!Number.isFinite(out.rounds) || out.rounds <= 0) throw new Error("invalid rounds");
  out.iters = Math.trunc(out.iters);
  out.warmup = Math.trunc(out.warmup);
  out.rounds = Math.trunc(out.rounds);
  return out;
}

const args = parseArgs(process.argv.slice(2));

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

async function collectTimes(call, ready, dispose) {
  let y = await call();
  await ready(y);
  if (dispose) dispose(y);

  for (let i = 0; i < args.warmup; i++) {
    y = await call();
    await ready(y);
    if (dispose) dispose(y);
  }

  const times = [];
  for (let i = 0; i < args.iters; i++) {
    const t0 = performance.now();
    y = await call();
    await ready(y);
    const t1 = performance.now();
    if (dispose) dispose(y);
    times.push((t1 - t0) * 1000);
  }
  return times;
}

function resultFromTimes(name, times) {
  return { name, median_us: median(times), min_us: Math.min(...times), iters: args.iters, rounds: args.rounds };
}

async function bench(name, call, ready, dispose) {
  const times = [];
  for (let r = 0; r < args.rounds; r++) {
    times.push(...await collectTimes(call, ready, dispose));
  }
  return resultFromTimes(name, times);
}

function printSummary(pgResults, jaxResults) {
  const rows = pgResults.map((pg) => {
    const jx = jaxResults.find(x => x.name === pg.name);
    return {
      name: pg.name,
      polygrad_us: pg.median_us,
      jax_js_us: jx?.median_us ?? NaN,
      ratio_pg_over_jax: jx ? pg.median_us / jx.median_us : NaN,
      polygrad_min_us: pg.min_us,
      jax_js_min_us: jx?.min_us ?? NaN,
    };
  });
  if (args.json) {
    console.log(JSON.stringify({
      backend: "polygrad-js-wasm-node-vs-jax-js-wasm-node",
      threads: 1,
      worker: false,
      compile_timed: false,
      readback_timed: false,
      iters: args.iters,
      warmup: args.warmup,
      rounds: args.rounds,
      paired_requested: args.paired,
      paired_actual: args.paired,
      results: rows,
    }, null, 2));
    return;
  }
  console.log(JSON.stringify({ backend: "polygrad-js-wasm-jit", results: pgResults }));
  console.log(JSON.stringify({ backend: "jax-js-wasm-node-jit", results: jaxResults }));
  console.log("\ncase,polygrad_us,jax_js_us,ratio_pg_over_jax");
  for (const row of rows) {
    if (!Number.isFinite(row.jax_js_us)) continue;
    console.log(`${row.name},${row.polygrad_us.toFixed(3)},${row.jax_js_us.toFixed(3)},${row.ratio_pg_over_jax.toFixed(3)}`);
  }
}

async function setupPolygrad(inputs) {
  const pg = await polygrad.create({ core: "wasm" });
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
    { name: "pointwise_1m", fn: pg.jit((aa, bb) => aa.add(bb).mul(aa.sub(bb)).add(aa.mul(bb)).relu()), args: [a, b] },
    {
      name: "where_1m",
      fn: pg.jit((aa, bb, cc, dd) =>
        aa.gt(bb).where(aa.add(cc).mul(bb.sub(dd)), aa.sub(cc).mul(bb.add(dd)))
      ),
      args: [a, b, c, d],
    },
    {
      name: "broadcast_reduce_1024",
      fn: pg.jit((xx, rr, cc) => xx.add(rr).mul(cc).sub(0.25).relu().sum(1)),
      args: [x2, row, col],
    },
    {
      name: "transpose_copy_1024",
      fn: pg.jit((xx) => xx.reshape(512, 2048).permute(1, 0).contiguous()),
      args: [x2],
    },
  ];

  return {
    workloads: workloads.map((w) => ({
      name: w.name,
      call: () => w.fn(...w.args),
      ready: async () => {},
      dispose: null,
      fn: w.fn,
    })),
    cleanup() {
      for (const w of workloads) w.fn.dispose?.();
      pg.destroy?.();
    },
  };
}

async function runPolygrad(inputs) {
  const setup = await setupPolygrad(inputs);
  const results = [];
  for (const w of setup.workloads) {
    results.push(await bench(w.name, w.call, w.ready, w.dispose));
  }
  setup.cleanup();
  return results;
}

async function setupJax(inputs) {
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
    {
      name: "pointwise_1m",
      fn: jit((aa, bb) => np.maximum(aa.ref.add(bb.ref).mul(aa.ref.sub(bb.ref)).add(aa.ref.mul(bb.ref)), 0), { device: "wasm" }),
      args: [a, b],
    },
    {
      name: "where_1m",
      fn: jit((aa, bb, cc, dd) =>
        np.where(
          np.greater(aa.ref, bb.ref),
          aa.ref.add(cc.ref).mul(bb.ref.sub(dd.ref)),
          aa.ref.sub(cc.ref).mul(bb.ref.add(dd.ref)),
        ),
        { device: "wasm" },
      ),
      args: [a, b, c, d],
    },
    {
      name: "broadcast_reduce_1024",
      fn: jit((xx, rr, cc) => np.sum(np.maximum(xx.ref.add(rr.ref).mul(cc.ref).sub(0.25), 0), 1), { device: "wasm" }),
      args: [x2, row, col],
    },
    {
      name: "transpose_copy_1024",
      fn: jit((xx) => np.reshape(xx.ref, [512, 2048]).transpose().add(0), { device: "wasm" }),
      args: [x2],
    },
  ];

  return {
    workloads: workloads.map((w) => ({
      name: w.name,
      call: () => w.fn(...w.args.map(x => x.ref)),
      ready: y => y.blockUntilReady(),
      dispose: y => y.dispose(),
      fn: w.fn,
    })),
    cleanup() {
      for (const w of workloads) w.fn.dispose?.();
      for (const x of [a, b, c, d, x2, row, col]) x.dispose();
    },
  };
}

async function runJax(inputs) {
  const setup = await setupJax(inputs);
  const results = [];
  for (const w of setup.workloads) {
    results.push(await bench(w.name, w.call, w.ready, w.dispose));
  }
  setup.cleanup();
  return results;
}

async function runPaired(inputs) {
  const pg = await setupPolygrad(inputs);
  const jax = await setupJax(inputs);
  const pgResults = [];
  const jaxResults = [];

  for (let i = 0; i < pg.workloads.length; i++) {
    const pw = pg.workloads[i];
    const jw = jax.workloads[i];
    const pgTimes = [];
    const jaxTimes = [];
    for (let r = 0; r < args.rounds; r++) {
      if (args.progress) console.error(`[paired] ${pw.name} round=${r + 1}/${args.rounds}`);
      if (r % 2 === 0) {
        pgTimes.push(...await collectTimes(pw.call, pw.ready, pw.dispose));
        jaxTimes.push(...await collectTimes(jw.call, jw.ready, jw.dispose));
      } else {
        jaxTimes.push(...await collectTimes(jw.call, jw.ready, jw.dispose));
        pgTimes.push(...await collectTimes(pw.call, pw.ready, pw.dispose));
      }
    }
    pgResults.push(resultFromTimes(pw.name, pgTimes));
    jaxResults.push(resultFromTimes(jw.name, jaxTimes));
  }

  pg.cleanup();
  jax.cleanup();
  return { pgResults, jaxResults };
}

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

const { pgResults, jaxResults } = args.paired
  ? await runPaired(inputs)
  : { pgResults: await runPolygrad(inputs), jaxResults: await runJax(inputs) };
printSummary(pgResults, jaxResults);
