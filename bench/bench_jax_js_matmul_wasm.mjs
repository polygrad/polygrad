#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import { performance } from "node:perf_hooks";

import polygrad from "../js/src/index.js";
import { checkWorkload, matmulReference } from "./wasm_checks.mjs";
import {
  blockUntilReady,
  defaultDevice,
  init,
  jit,
  numpy as np,
} from "../references/jax-js/dist/index.js";

const PR_TABLE = [
  [64, 80.35, 77.23, 3.50, 1.74],
  [128, 37.72, 38.19, 21.58, 13.98],
  [256, 112.66, 121.13, 100.56, 90.99],
  [512, 253.03, 254.27, 205.42, 234.19],
  [1024, 345.04, 340.20, 297.89, 326.12],
  [2048, 447.52, 475.36, 361.21, 447.46],
  [4096, 530.52, 531.89, 362.51, 468.80],
];

function parseArgs(argv) {
  const args = {
    sizes: PR_TABLE.map((r) => r[0]),
    iters: 5,
    warmup: 2,
    largeIters: 3,
    includeOpenblas: true,
    includeJax: true,
    includePolygrad: true,
    polygradMaxSize: Infinity,
    blasThreads: 8,
    paired: false,
    rounds: 1,
    json: false,
    progress: true,
  };

  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    const next = () => {
      if (i + 1 >= argv.length) throw new Error(`missing value for ${a}`);
      return argv[++i];
    };
    if (a === "--sizes") args.sizes = next().split(",").map((x) => Number(x.trim())).filter(Boolean);
    else if (a === "--iters") args.iters = Number(next());
    else if (a === "--warmup") args.warmup = Number(next());
    else if (a === "--large-iters") args.largeIters = Number(next());
    else if (a === "--rounds") args.rounds = Number(next());
    else if (a === "--blas-threads") args.blasThreads = Number(next());
    else if (a === "--polygrad-max-size") args.polygradMaxSize = Number(next());
    else if (a === "--no-openblas") args.includeOpenblas = false;
    else if (a === "--no-jax") args.includeJax = false;
    else if (a === "--no-polygrad") args.includePolygrad = false;
    else if (a === "--paired") args.paired = true;
    else if (a === "--quiet") args.progress = false;
    else if (a === "--json") args.json = true;
    else if (a === "--help" || a === "-h") {
      console.log(`Usage: node bench/bench_jax_js_matmul_wasm.mjs [options]

Reproduce the jax-js PR #116 WASM matmul table locally.
All performance columns are GFLOP/s using 2*n^3 FLOPs.

Options:
  --sizes 64,128,...     sizes to run (default: 64,128,256,512,1024,2048,4096)
  --iters N              median samples for n <= 1024 (default: 5)
  --large-iters N        median samples for n > 1024 (default: 3)
  --warmup N             warmup iterations per case (default: 2)
  --paired               alternate jax-js/polygrad rounds per size
  --rounds N             paired rounds per size (default: 1)
  --blas-threads N       OPENBLAS_NUM_THREADS for NumPy/OpenBLAS (default: 8)
  --polygrad-max-size N   skip Polygrad columns for n > N
  --no-openblas          skip local NumPy/OpenBLAS columns
  --no-jax               skip jax-js WASM columns
  --no-polygrad          skip Polygrad WASM JIT columns
  --quiet                suppress progress logs on stderr
  --json                 print JSON instead of markdown table
`);
      process.exit(0);
    } else {
      throw new Error(`unknown option ${a}`);
    }
  }
  if (!Number.isFinite(args.iters) || args.iters <= 0) throw new Error("--iters must be positive");
  if (!Number.isFinite(args.largeIters) || args.largeIters <= 0) throw new Error("--large-iters must be positive");
  if (!Number.isFinite(args.warmup) || args.warmup < 0) throw new Error("--warmup must be nonnegative");
  if (!Number.isFinite(args.rounds) || args.rounds <= 0) throw new Error("--rounds must be positive");
  args.rounds = Math.floor(args.rounds);
  return args;
}

function makeMatrixData(n) {
  const data = new Float32Array(n * n);
  for (let i = 0; i < data.length; i++) data[i] = (i % 7) - 3;
  return data;
}

async function checkMatmul(n, callAB, callBT, ready, dispose) {
  // Integer products/sums for these inputs are exactly representable in f32.
  const tolerance = { atol: 0, rtol: 0 };
  const ab = await checkWorkload({ name: `matmul AB ${n}`, call: callAB, ready, dispose }, matmulReference(n, false), tolerance);
  const bt = await checkWorkload({ name: `matmul ABT ${n}`, call: callBT, ready, dispose }, matmulReference(n, true), tolerance);
  return { ab, bt };
}

function median(xs) {
  const ys = [...xs].sort((a, b) => a - b);
  return ys[Math.floor(ys.length / 2)];
}

function gflops(n, seconds) {
  return (2 * n * n * n) / seconds / 1e9;
}

function samplesFor(args, n) {
  return n <= 1024 ? args.iters : args.largeIters;
}

async function timeAsync(args, n, call, ready, dispose) {
  let y = await call();
  await ready(y);
  dispose?.(y);

  for (let i = 0; i < args.warmup; i++) {
    y = await call();
    await ready(y);
    dispose?.(y);
  }

  const times = [];
  const iters = samplesFor(args, n);
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

function runOpenBlas(args) {
  if (!args.includeOpenblas) return new Map();
  if (args.progress) console.error(`[openblas] running sizes=${args.sizes.join(",")} threads=${args.blasThreads}`);
  const py = `
import json, os, time
import numpy as np
sizes = ${JSON.stringify(args.sizes)}
warmup = ${args.warmup}
iters_small = ${args.iters}
iters_large = ${args.largeIters}
def median(xs):
  xs = sorted(xs)
  return xs[len(xs)//2]
def bench(fn, n):
  y = fn(); float(y.ravel()[0])
  for _ in range(warmup):
    y = fn(); float(y.ravel()[0])
  times = []
  iters = iters_small if n <= 1024 else iters_large
  for _ in range(iters):
    t0 = time.perf_counter()
    y = fn(); float(y.ravel()[0])
    times.append(time.perf_counter() - t0)
  return median(times)
for n in sizes:
  idx = np.arange(n*n, dtype=np.float32)
  a = ((idx % 7) - 3).reshape(n, n).astype(np.float32, copy=False)
  b = ((idx % 7) - 3).reshape(n, n).astype(np.float32, copy=False)
  t_ab = bench(lambda: a @ b, n)
  t_bt = bench(lambda: a @ b.T, n)
  flops = 2*n*n*n/1e9
  print(json.dumps({"n": n, "openblas_ab": flops/t_ab, "openblas_bt": flops/t_bt, "openblas_ab_s": t_ab, "openblas_bt_s": t_bt}), flush=True)
`;

  const child = spawnSync("python", ["-c", py], {
    encoding: "utf8",
    env: { ...process.env, OPENBLAS_NUM_THREADS: String(args.blasThreads) },
    maxBuffer: 1024 * 1024,
  });
  if (child.status !== 0) {
    throw new Error(`OpenBLAS benchmark failed:\n${child.stderr || child.stdout}`);
  }

  const out = new Map();
  for (const line of child.stdout.trim().split(/\n+/).filter(Boolean)) {
    const row = JSON.parse(line);
    if (args.progress)
      console.error(`[openblas] n=${row.n} A@B=${fmt(row.openblas_ab)} GF/s A@B.T=${fmt(row.openblas_bt)} GF/s`);
    out.set(row.n, row);
  }
  return out;
}

async function runJax(args) {
  const out = new Map();
  if (!args.includeJax) return out;

  const devices = await init("wasm");
  if (!devices.includes("wasm")) throw new Error("jax-js wasm device unavailable");
  defaultDevice("wasm");

  for (const n of args.sizes) {
    if (args.progress) console.error(`[jax-js] n=${n}`);
    const a = np.array(makeMatrixData(n), { shape: [n, n], device: "wasm" });
    const b = np.array(makeMatrixData(n), { shape: [n, n], device: "wasm" });
    await blockUntilReady([a, b]);

    await checkMatmul(n, () => np.matmul(a.ref, b.ref), () => np.matmul(a.ref, b.ref.transpose()), y => y.blockUntilReady(), y => y.dispose());

    const tAB = await timeAsync(args, n, () => np.matmul(a.ref, b.ref), (y) => y.blockUntilReady(), (y) => y.dispose());
    const tBT = await timeAsync(args, n, () => np.matmul(a.ref, b.ref.transpose()), (y) => y.blockUntilReady(), (y) => y.dispose());
    out.set(n, { jax_ab: gflops(n, tAB), jax_bt: gflops(n, tBT), jax_ab_s: tAB, jax_bt_s: tBT });
    if (args.progress)
      console.error(`[jax-js] n=${n} A@B=${fmt(gflops(n, tAB))} GF/s A@B.T=${fmt(gflops(n, tBT))} GF/s`);

    a.dispose();
    b.dispose();
  }
  return out;
}

async function runPolygrad(args) {
  const out = new Map();
  if (!args.includePolygrad) return out;

  const pg = await polygrad.create({ core: "wasm" });
  const { Tensor } = pg;

  for (const n of args.sizes) {
    if (n > args.polygradMaxSize) {
      if (args.progress) console.error(`[polygrad] n=${n} skipped by --polygrad-max-size=${args.polygradMaxSize}`);
      continue;
    }
    if (args.progress) console.error(`[polygrad] n=${n}`);
    const a = new Tensor(makeMatrixData(n)).reshape(n, n);
    const b = new Tensor(makeMatrixData(n)).reshape(n, n);
    await a.realize(b);

    const fAB = pg.jit((aa, bb) => aa.matmul(bb));
    const fBT = pg.jit((aa, bb) => aa.matmul(bb.permute(1, 0)));
    await primePolygradJit(fAB, a, b);
    await primePolygradJit(fBT, a, b);
    await checkMatmul(n, () => fAB(a, b), () => fBT(a, b), y => y.realize(), null);
    const tAB = await timeAsync(args, n, () => fAB(a, b), (y) => y.realize(), null);
    const tBT = await timeAsync(args, n, () => fBT(a, b), (y) => y.realize(), null);
    out.set(n, {
      poly_ab: gflops(n, tAB),
      poly_bt: gflops(n, tBT),
      poly_ab_s: tAB,
      poly_bt_s: tBT,
      poly_ab_schedules: fAB.scheduleCount,
      poly_bt_schedules: fBT.scheduleCount,
    });
    if (args.progress)
      console.error(`[polygrad] n=${n} A@B=${fmt(gflops(n, tAB))} GF/s A@B.T=${fmt(gflops(n, tBT))} GF/s`);

    fAB.dispose?.();
    fBT.dispose?.();
  }

  pg.dispose?.();
  return out;
}

async function runPaired(args) {
  const jaxOut = new Map();
  const polyOut = new Map();

  let pg = null;
  let Tensor = null;
  if (args.includePolygrad) {
    pg = await polygrad.create({ core: "wasm" });
    Tensor = pg.Tensor;
  }

  if (args.includeJax) {
    const devices = await init("wasm");
    if (!devices.includes("wasm")) throw new Error("jax-js wasm device unavailable");
    defaultDevice("wasm");
  }

  const rounds = args.paired ? args.rounds : 1;

  for (const n of args.sizes) {
    if (args.progress) console.error(`[paired] n=${n} rounds=${rounds}`);
    const dataA = makeMatrixData(n);
    const dataB = makeMatrixData(n);

    let ja = null, jb = null;
    if (args.includeJax) {
      ja = np.array(dataA, { shape: [n, n], device: "wasm" });
      jb = np.array(dataB, { shape: [n, n], device: "wasm" });
      await blockUntilReady([ja, jb]);
    }

    let pa = null, pb = null, fAB = null, fBT = null;
    if (args.includePolygrad && n <= args.polygradMaxSize) {
      pa = new Tensor(dataA).reshape(n, n);
      pb = new Tensor(dataB).reshape(n, n);
      await pa.realize(pb);
      fAB = pg.jit((aa, bb) => aa.matmul(bb));
      fBT = pg.jit((aa, bb) => aa.matmul(bb.permute(1, 0)));
      await primePolygradJit(fAB, pa, pb);
      await primePolygradJit(fBT, pa, pb);
    } else if (args.includePolygrad && args.progress) {
      console.error(`[polygrad] n=${n} skipped by --polygrad-max-size=${args.polygradMaxSize}`);
    }

    const jaxAB = [], jaxBT = [], polyAB = [], polyBT = [];
    if (ja) await checkMatmul(n, () => np.matmul(ja.ref, jb.ref), () => np.matmul(ja.ref, jb.ref.transpose()), y => y.blockUntilReady(), y => y.dispose());
    if (fAB) await checkMatmul(n, () => fAB(pa, pb), () => fBT(pa, pb), y => y.realize(), null);
    const runJaxRound = async () => {
      if (!args.includeJax) return;
      jaxAB.push(await timeAsync(args, n, () => np.matmul(ja.ref, jb.ref), (y) => y.blockUntilReady(), (y) => y.dispose()));
      jaxBT.push(await timeAsync(args, n, () => np.matmul(ja.ref, jb.ref.transpose()), (y) => y.blockUntilReady(), (y) => y.dispose()));
    };
    const runPolyRound = async () => {
      if (!fAB || !fBT) return;
      polyAB.push(await timeAsync(args, n, () => fAB(pa, pb), (y) => y.realize(), null));
      polyBT.push(await timeAsync(args, n, () => fBT(pa, pb), (y) => y.realize(), null));
    };

    for (let r = 0; r < rounds; r++) {
      if (args.progress) console.error(`[paired] n=${n} round=${r + 1}/${rounds}`);
      if (r % 2 === 0) {
        await runJaxRound();
        await runPolyRound();
      } else {
        await runPolyRound();
        await runJaxRound();
      }
    }

    if (jaxAB.length) {
      const tAB = median(jaxAB);
      const tBT = median(jaxBT);
      jaxOut.set(n, {
        jax_ab: gflops(n, tAB),
        jax_bt: gflops(n, tBT),
        jax_ab_s: tAB,
        jax_bt_s: tBT,
        jax_ab_samples_s: jaxAB,
        jax_bt_samples_s: jaxBT,
      });
      if (args.progress)
        console.error(`[jax-js] n=${n} A@B=${fmt(gflops(n, tAB))} GF/s A@B.T=${fmt(gflops(n, tBT))} GF/s`);
    }

    if (polyAB.length) {
      const tAB = median(polyAB);
      const tBT = median(polyBT);
      polyOut.set(n, {
        poly_ab: gflops(n, tAB),
        poly_bt: gflops(n, tBT),
        poly_ab_s: tAB,
        poly_bt_s: tBT,
        poly_ab_schedules: fAB.scheduleCount,
        poly_bt_schedules: fBT.scheduleCount,
        poly_ab_samples_s: polyAB,
        poly_bt_samples_s: polyBT,
      });
      if (args.progress)
        console.error(`[polygrad] n=${n} A@B=${fmt(gflops(n, tAB))} GF/s A@B.T=${fmt(gflops(n, tBT))} GF/s`);
    }

    fAB?.dispose?.();
    fBT?.dispose?.();
    ja?.dispose?.();
    jb?.dispose?.();
  }

  pg?.dispose?.();
  return { jax: jaxOut, poly: polyOut };
}

function fmt(x) {
  return Number.isFinite(x) ? x.toFixed(2) : "";
}

function printMarkdown(rows, args) {
  console.log(`# jax-js PR #116 WASM Matmul Table Reproduction`);
  console.log("");
  console.log(`Command: \`${process.argv.map((x) => x.includes(" ") ? JSON.stringify(x) : x).join(" ")}\``);
  console.log(`Metric: GFLOP/s, computed as \`2*n^3 / median_seconds / 1e9\`.`);
  console.log(`Samples: warmup=${args.warmup}, iters=${args.iters}, large_iters=${args.largeIters}, rounds=${args.rounds}, paired=${args.paired}, openblas_threads=${args.blasThreads}.`);
  console.log("");
  console.log("## PR Table");
  console.log("");
  console.log("| n | PR OpenBLAS A @ B | PR OpenBLAS A @ B.T | PR wasm A @ B | PR wasm A @ B.T |");
  console.log("| --: | --: | --: | --: | --: |");
  for (const [n, ob, obt, wasm, wasmt] of PR_TABLE) {
    if (!args.sizes.includes(n)) continue;
    console.log(`| ${n} | ${fmt(ob)} | ${fmt(obt)} | ${fmt(wasm)} | ${fmt(wasmt)} |`);
  }
  console.log("");
  console.log("## Local Reproduction");
  console.log("");
  console.log("| n | OpenBLAS A @ B | OpenBLAS A @ B.T | jax-js wasm A @ B | jax-js wasm A @ B.T | Polygrad wasm JIT A @ B | Polygrad wasm JIT A @ B.T |");
  console.log("| --: | --: | --: | --: | --: | --: | --: |");
  for (const row of rows) {
    console.log(`| ${row.n} | ${fmt(row.openblas_ab)} | ${fmt(row.openblas_bt)} | ${fmt(row.jax_ab)} | ${fmt(row.jax_bt)} | ${fmt(row.poly_ab)} | ${fmt(row.poly_bt)} |`);
  }
}

const args = parseArgs(process.argv);

const openblas = runOpenBlas(args);
const measured = (args.paired || args.rounds > 1)
  ? await runPaired(args)
  : { jax: await runJax(args), poly: await runPolygrad(args) };

const rows = args.sizes.map((n) => ({
  n,
  ...(openblas.get(n) || {}),
  ...(measured.jax.get(n) || {}),
  ...(measured.poly.get(n) || {}),
}));

if (args.json) {
  console.log(JSON.stringify({ args, correctness_checked: true, pr_table: PR_TABLE, rows }, null, 2));
} else {
  printMarkdown(rows, args);
}
