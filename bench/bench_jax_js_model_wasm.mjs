#!/usr/bin/env node
import { performance } from "node:perf_hooks";

import polygrad from "../js/src/index.js";
import {
  blockUntilReady,
  defaultDevice,
  init,
  jit,
  nn,
  numpy as np,
} from "../references/jax-js/dist/index.js";

const CASES = {
  mlp_small: {
    kind: "mlp",
    tokens: 64,
    d: 256,
    hidden: 1024,
    out: 256,
    activation: "relu",
  },
  mlp_token: {
    kind: "mlp",
    tokens: 1,
    d: 768,
    hidden: 3072,
    out: 768,
    activation: "relu",
  },
  mlp_batch: {
    kind: "mlp",
    tokens: 16,
    d: 768,
    hidden: 3072,
    out: 768,
    activation: "relu",
  },
  qwen_ffn_token: {
    kind: "qwen_ffn",
    tokens: 1,
    d: 1024,
    hidden: 2816,
    out: 1024,
    activation: "silu",
  },
  qwen_ffn_batch: {
    kind: "qwen_ffn",
    tokens: 16,
    d: 1024,
    hidden: 2816,
    out: 1024,
    activation: "silu",
  },
};

function parseArgs(argv) {
  const args = {
    cases: ["mlp_small", "mlp_token", "mlp_batch", "qwen_ffn_token", "qwen_ffn_batch"],
    iters: 10,
    warmup: 2,
    rounds: 1,
    paired: false,
    json: false,
    progress: true,
  };

  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    const next = () => {
      if (i + 1 >= argv.length) throw new Error(`missing value for ${a}`);
      return argv[++i];
    };
    if (a === "--cases") args.cases = next().split(",").map((x) => x.trim()).filter(Boolean);
    else if (a === "--iters") args.iters = Number(next());
    else if (a === "--warmup") args.warmup = Number(next());
    else if (a === "--rounds") args.rounds = Number(next());
    else if (a === "--paired") args.paired = true;
    else if (a === "--quiet") args.progress = false;
    else if (a === "--json") args.json = true;
    else if (a === "--help" || a === "-h") {
      console.log(`Usage: node bench/bench_jax_js_model_wasm.mjs [options]

Compare model-shaped Polygrad WASM JIT and jax-js WASM JIT throughput.
This is a shared staged MLP / Qwen-like FFN graph benchmark, not a full Qwen
tokenizer or GGUF end-to-end benchmark.

Options:
  --cases a,b       cases to run (default: ${args.cases.join(",")})
                   available: ${Object.keys(CASES).join(",")}
  --iters N         median samples per round (default: ${args.iters})
  --warmup N        warmup iterations before each sample group (default: ${args.warmup})
  --paired          alternate jax-js/polygrad rounds per case
  --rounds N        paired rounds per case (default: ${args.rounds})
  --quiet           suppress progress logs on stderr
  --json            print JSON instead of markdown table
`);
      process.exit(0);
    } else {
      throw new Error(`unknown option ${a}`);
    }
  }

  for (const name of args.cases) {
    if (!CASES[name]) throw new Error(`unknown case '${name}'`);
  }
  if (!Number.isFinite(args.iters) || args.iters <= 0) throw new Error("--iters must be positive");
  if (!Number.isFinite(args.warmup) || args.warmup < 0) throw new Error("--warmup must be nonnegative");
  if (!Number.isFinite(args.rounds) || args.rounds <= 0) throw new Error("--rounds must be positive");
  args.iters = Math.trunc(args.iters);
  args.warmup = Math.trunc(args.warmup);
  args.rounds = Math.trunc(args.rounds);
  return args;
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

function fmt(x) {
  return Number.isFinite(x) ? x.toFixed(3) : "";
}

function modelFlops(c) {
  if (c.kind === "mlp") {
    return 2 * c.tokens * c.d * c.hidden + 2 * c.tokens * c.hidden * c.out;
  }
  if (c.kind === "qwen_ffn") {
    return 2 * c.tokens * c.d * c.hidden * 2 + 2 * c.tokens * c.hidden * c.d;
  }
  return NaN;
}

function caseLabel(c) {
  if (c.kind === "mlp") {
    return `${c.kind} tokens=${c.tokens} d=${c.d} hidden=${c.hidden} out=${c.out}`;
  }
  return `${c.kind} tokens=${c.tokens} d=${c.d} hidden=${c.hidden}`;
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

async function timeAsync(args, call, ready, dispose) {
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
    const f2 = jit((h, w2, b2) =>
      np.matmul(h.ref, w2.ref.transpose()).ref.add(b2.ref),
      { device: "wasm" },
    );
    return {
      async call(inputs) {
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
    async call(inputs) {
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
      get scheduleCount() {
        return (f1.scheduleCount || 0) + (f2.scheduleCount || 0);
      },
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
      return (fGate.scheduleCount || 0) +
        (fUp.scheduleCount || 0) +
        (fMul.scheduleCount || 0) +
        (fDown.scheduleCount || 0);
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

async function runCase(args, pg, name, c) {
  if (args.progress) console.error(`[model] ${name}: ${caseLabel(c)}`);
  const inputData = caseInputs(c);

  const jaxInputs = createJaxInputs(c, inputData);
  await blockUntilReady(jaxInputs);
  const jaxPipeline = createJaxPipeline(c);
  await primePipeline(jaxPipeline, jaxInputs);

  const polyInputs = await createPolyInputs(pg, c, inputData);
  const polyPipeline = createPolyPipeline(pg, c);
  await primePipeline(polyPipeline, polyInputs);

  const jaxSamples = [];
  const polySamples = [];
  const rounds = args.paired ? args.rounds : 1;
  const runJaxRound = async () => {
    jaxSamples.push(...await timeAsync(
      args,
      () => jaxPipeline.call(jaxInputs),
      (y) => jaxPipeline.ready(y),
      (y) => jaxPipeline.disposeOutput(y),
    ));
  };
  const runPolyRound = async () => {
    polySamples.push(...await timeAsync(
      args,
      () => polyPipeline.call(polyInputs),
      (y) => polyPipeline.ready(y),
      (y) => polyPipeline.disposeOutput(y),
    ));
  };

  for (let r = 0; r < rounds; r++) {
    if (args.progress && rounds > 1) console.error(`[model] ${name} round ${r + 1}/${rounds}`);
    if (args.paired && r % 2 === 1) {
      await runPolyRound();
      await runJaxRound();
    } else {
      await runJaxRound();
      await runPolyRound();
    }
  }

  const jaxS = median(jaxSamples);
  const polyS = median(polySamples);
  const flops = modelFlops(c);
  const polySchedules = polyPipeline.scheduleCount;

  jaxPipeline.dispose();
  polyPipeline.dispose();
  for (const x of jaxInputs) x.dispose();

  return {
    name,
    ...c,
    label: caseLabel(c),
    flops,
    jax_s: jaxS,
    poly_s: polyS,
    jax_us: jaxS * 1e6,
    poly_us: polyS * 1e6,
    jax_tokens_per_s: c.tokens / jaxS,
    poly_tokens_per_s: c.tokens / polyS,
    jax_gflops: flops / jaxS / 1e9,
    poly_gflops: flops / polyS / 1e9,
    ratio_poly_over_jax_time: polyS / jaxS,
    poly_schedules: polySchedules,
    jax_samples_s: jaxSamples,
    poly_samples_s: polySamples,
  };
}

function printMarkdown(rows, args) {
  console.log("# jax-js vs Polygrad WASM Model Throughput");
  console.log("");
  console.log(`Command: \`${process.argv.map((x) => x.includes(" ") ? JSON.stringify(x) : x).join(" ")}\``);
  console.log(`Samples: warmup=${args.warmup}, iters=${args.iters}, rounds=${args.rounds}, paired=${args.paired}.`);
  console.log("Metric: median JIT replay time; compile and input upload excluded.");
  console.log("");
  console.log("| case | shape | pg us | jax us | pg tok/s | jax tok/s | pg GF/s | jax GF/s | pg/jax time | schedules |");
  console.log("| --- | --- | --: | --: | --: | --: | --: | --: | --: | --: |");
  for (const row of rows) {
    console.log(`| ${row.name} | ${row.label} | ${fmt(row.poly_us)} | ${fmt(row.jax_us)} | ${fmt(row.poly_tokens_per_s)} | ${fmt(row.jax_tokens_per_s)} | ${fmt(row.poly_gflops)} | ${fmt(row.jax_gflops)} | ${fmt(row.ratio_poly_over_jax_time)} | ${row.poly_schedules} |`);
  }
}

const args = parseArgs(process.argv);

const devices = await init("wasm");
if (!devices.includes("wasm")) throw new Error("jax-js wasm device unavailable");
defaultDevice("wasm");

const pg = await polygrad.create({ core: "wasm" });
const rows = [];
try {
  for (const name of args.cases) {
    rows.push(await runCase(args, pg, name, CASES[name]));
  }
} finally {
  await pg.dispose?.();
}

if (args.json) {
  console.log(JSON.stringify({ args, rows }, null, 2));
} else {
  printMarkdown(rows, args);
}
