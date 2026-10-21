// Measurement contracts and independent references for the Wasm workloads.
export async function timeWorkload(call, ready, dispose, { warmup, iters, observer = null, inject = null, burn = null }) {
  if (!Number.isInteger(warmup) || warmup < 0 || !Number.isInteger(iters) || iters <= 0)
    throw new Error('invalid timing iteration count');
  if (inject && !['kernel', 'host'].includes(inject)) throw new Error('unknown timing control');
  if (inject === 'kernel' && (!observer || iters % 10)) throw new Error('kernel control requires trace and multiples of ten');
  if (inject === 'host' && typeof burn !== 'function') throw new Error('host control requires executed work');
  for (let i = 0; i <= warmup; i++) {
    const y = await call();
    try { await ready(y); } finally { dispose?.(y); }
  }
  const result = { samples: [], extraKernelCalls: 0, hostBurnCalls: 0 };
  for (let i = 0; i < iters; i++) {
    if (observer) observer.trace = [];
    const start = performance.now();
    let y;
    try {
      y = await call(); await ready(y);
      if (inject === 'kernel' && i % 10 === 9) {
        if (!observer.trace.length) throw new Error('kernel control captured no dispatches');
        for (const { fn, args } of observer.trace) { fn(...args); result.extraKernelCalls++; }
      }
      if (inject === 'host') { burn(); result.hostBurnCalls++; }
      result.samples.push((performance.now() - start) / 1000);
    } finally { if (observer) observer.trace = null; if (y !== undefined) dispose?.(y); }
  }
  return result;
}

export function decodedWatOpcodes(wat) {
  // Consume disassembler output, never scan binary immediates as instructions.
  const text = wat.replace(/"(?:\\.|[^"\\])*"/g, '""').replace(/;;[^\n]*/g, '');
  const counts = {};
  for (const [, op] of text.matchAll(/\(([\w.]+)(?=[\s)])/g)) {
    if (/^(?:(?:i32|i64|f32|f64|v128|i8x16|i16x8|i32x4|i64x2|f32x4|f64x2|local|global|memory|table)\.|(?:block|loop|if|br|br_if|br_table|return|call|call_indirect|drop|select|nop|unreachable)$)/.test(op))
      counts[op] = (counts[op] ?? 0) + 1;
  }
  return Object.fromEntries(Object.entries(counts).sort(([a], [b]) => a.localeCompare(b)));
}

export function structuralSnapshot(report) {
  const { device, pipeline, kernels } = report.args;
  if (device !== 'wasm' || report.pages.length !== 1) throw new Error('snapshot requires one Wasm page');
  return { schema: 1, device, pipeline, kernels, phases: {
    cold: 'generated modules, in compilation order', warm: 'one validated replay, excluding output disposal',
    allocationScope: 'exported malloc/calloc/realloc calls only; internal C allocations covered by C tests',
  }, workloads: report.pages[0].pgResults.map(r => {
    if (!r.structure || !r.structure.cold.modules.length || !r.structure.warm.dispatches)
      throw new Error(`${r.name}: missing structural observations`);
    return { name: r.name, schedules: r.schedules, cold: r.structure.cold, warm: r.structure.warm };
  }) };
}

export function pagePairOrder(pairs) {
  if (!Number.isInteger(pairs) || pairs < 2 || pairs % 2) throw new Error('page pairs must be a positive even count');
  return Array.from({ length: pairs }, (_, pair) => (pair % 2
    ? ['candidate', 'baseline'] : ['baseline', 'candidate']).map(lane => ({ pair, lane }))).flat();
}

// Only for the stateless FFN fixtures. The caller must retain the captured
// buffers and perform no allocating runtime calls until raw replay completes.
export function measureKernelBlocks(trace, { page = 0, blocks = 8, targetMs = 5, now = () => performance.now() } = {}) {
  if (!trace.length || trace.some(t => typeof t.fn !== 'function' || !Array.isArray(t.args)))
    throw new Error('empty or invalid kernel trace');
  if (!Number.isInteger(blocks) || blocks < 2 || blocks % 2 || !(targetMs > 0))
    throw new Error('kernel blocks must be positive and even');
  const units = trace.map((t, i) => ({ unit: `dispatch${i}`, trace: [t] }));
  units.push({ unit: 'trace', trace });
  return units.map(({ unit, trace: calls }) => {
    const run = (n, extra) => {
      const start = now();
      for (let i = 0; i < n; i++) {
        for (const { fn, args } of calls) fn(...args);
        if (extra && i % 10 === 9) for (const { fn, args } of calls) fn(...args);
      }
      return now() - start;
    };
    run(20, false); // same function/arguments in both controls; warm tiering
    const pilotMs = run(20, false);
    const repetitions = Math.min(100000, Math.max(10, Math.ceil(targetMs * 20 / Math.max(pilotMs, .001) / 10) * 10));
    const samples = [];
    for (let block = 0; block < blocks; block++) {
      for (const control of (block + page) % 2 ? ['kernel', 'identity'] : ['identity', 'kernel']) {
        const order = (block + page) % 2 ? 'BA' : 'AB';
        let baselineMs, candidateMs;
        for (const lane of order) {
          const elapsed = run(repetitions, lane === 'B' && control === 'kernel');
          if (lane === 'A') baselineMs = elapsed; else candidateMs = elapsed;
        }
        if (!(baselineMs > 0 && candidateMs > 0)) throw new Error('kernel timing below clock resolution');
        samples.push({ control, order, baselineMs, candidateMs, ratio: candidateMs / baselineMs,
          baselineCalls: repetitions * calls.length,
          candidateCalls: (repetitions + (control === 'kernel' ? repetitions / 10 : 0)) * calls.length });
      }
    }
    return { unit, hashes: calls.map(t => t.hash), repetitions, samples };
  });
}

function assessRatios(ratios, margin, alpha, comparisons) {
  const n = ratios.length;
  let mass = 2 ** -n, tail = 0, trim = -1;
  for (let k = 0; k < n / 2; k++) {
    tail += mass;
    if (2 * tail <= alpha / comparisons) trim = k;
    else break;
    mass *= (n - k) / (k + 1);
  }
  const sorted = [...ratios].sort((a, b) => a - b);
  const interval = trim < 0 ? [0, Infinity] : [sorted[trim], sorted[n - 1 - trim]];
  const [lo, hi] = interval;
  const status = lo > 1 + margin ? 'regression' : hi < 1 - margin ? 'improvement'
    : lo >= 1 - margin && hi <= 1 + margin ? 'equivalent'
      : hi <= 1 + margin ? 'nonregression' : 'inconclusive';
  return { pairs: n, ratios, ratio: (sorted[n / 2 - 1] + sorted[n / 2]) / 2, interval,
    intervalOrder: { method: 'binomial-order-statistic', comparisons, trimmedPairsPerTail: trim < 0 ? null : trim },
    familyAlpha: alpha, margin, status };
}

export function assessKernelPages(pages) {
  if (pages.length < 20 || pages.length % 2 || pages.some((p, i) => p.kernelPage !== i))
    throw new Error('kernel qualification requires at least twenty ordered fresh pages');
  const flatten = p => p.pgResults.flatMap(r => r.kernelMeasurements.map(k => ({ ...k, name: `${r.name}/${k.unit}` })));
  const rows = pages.map(flatten);
  const identity = row => JSON.stringify(row.map(r => [r.name, r.hashes]));
  if (!rows[0].length || rows.some(r => identity(r) !== identity(rows[0])))
    throw new Error('kernel identity changed between pages');
  const results = rows[0].flatMap((unit, i) => ['identity', 'kernel'].map(control => {
    const observations = rows.map(row => {
      const samples = row[i].samples.filter(s => s.control === control);
      if (!samples.length || samples.some(s => !Number.isFinite(s.baselineMs) || !Number.isFinite(s.candidateMs)
        || s.baselineMs <= 0 || s.candidateMs <= 0 || !(s.baselineCalls > 0)
        || s.candidateCalls * 10 !== s.baselineCalls * (control === 'kernel' ? 11 : 10)))
        throw new Error('invalid timing or planted work count');
      return { baseline: samples.reduce((a, s) => a + s.baselineMs, 0),
        candidate: samples.reduce((a, s) => a + s.candidateMs, 0) };
    });
    return { name: unit.name, hashes: unit.hashes, control, observations,
      ...assessRatios(observations.map(o => o.candidate / o.baseline), control === 'identity' ? .05 : .03, .05, rows[0].length * 2) };
  }));
  return { sampleUnit: 'one aggregate of paired blocks per fresh page; inner blocks are not independent samples',
    qualified: results.every(r => r.status === (r.control === 'identity' ? 'equivalent' : 'regression')), results };
}

export function assessPagePairs(pages, margin, alpha = .05) {
  if (!Number.isFinite(margin) || margin < 0 || margin >= 1 || !(alpha > 0 && alpha < 1))
    throw new Error('invalid comparison margin or alpha');
  const n = pages.length / 2;
  if (n < 10) throw new Error('comparison requires at least ten page pairs');
  const order = pagePairOrder(n);
  if (pages.some((p, i) => p.pair !== order[i].pair || p.lane !== order[i].lane))
    throw new Error('page order must be adjacent balanced AB/BA pairs');
  const names = pages[0].pgResults.map(r => r.name);
  if (!names.length || new Set(names).size !== names.length || pages.some(p =>
    JSON.stringify(p.pgResults.map(r => r.name)) !== JSON.stringify(names))) throw new Error('page cases must match');
  // Order-statistic interval for independent page-pair ratios. Bonferroni
  // covers all workloads in this comparison. Inner timing iterations are NOT
  // independent samples. Serial environmental bias still requires an A/A control.
  return names.map((name, i) => {
    const ratios = [], observations = [];
    for (let pair = 0; pair < n; pair++) {
      const ab = pages.slice(pair * 2, pair * 2 + 2);
      const baseline = ab.find(p => p.lane === 'baseline').pgResults[i].seconds;
      const candidate = ab.find(p => p.lane === 'candidate').pgResults[i].seconds;
      if (![baseline, candidate].every(t => Number.isFinite(t) && t > 0)) throw new Error(`${name}: invalid timing`);
      ratios.push(candidate / baseline);
      observations.push({ baseline, candidate });
    }
    return { name, observations, ...assessRatios(ratios, margin, alpha, names.length) };
  });
}

export function assessABBA(pages, threshold) {
  if (pages.length !== 4) throw new Error('ABBA comparison requires four pages');
  if (!Number.isFinite(threshold) || threshold < 0) throw new Error('invalid slowdown threshold');
  const names = pages[0].pgResults.map(r => r.name);
  if (!names.length || new Set(names).size !== names.length || pages.some(p =>
    JSON.stringify(p.pgResults.map(r => r.name)) !== JSON.stringify(names))) {
    throw new Error('ABBA cases must match');
  }
  return names.map((name, i) => {
    const ms = pages.map(p => p.pgResults[i].seconds * 1000);
    if (ms.some(x => !Number.isFinite(x) || x <= 0)) throw new Error(`${name}: invalid timing`);
    const baselineMs = [ms[0], ms[3]], candidateMs = [ms[1], ms[2]];
    // Both candidate pages must exceed both baseline pages to call a regression.
    // Overlapping the threshold is inconclusive, not a passing gate.
    const status = Math.min(...candidateMs) > Math.max(...baselineMs) * (1 + threshold)
      ? 'regression' : Math.max(...candidateMs) <= Math.min(...baselineMs) * (1 + threshold)
        ? 'pass' : 'inconclusive';
    return { name, baselineMs, candidateMs, ratio: (ms[1] + ms[2]) / (ms[0] + ms[3]), status };
  });
}

export function assertClose(actual, expected, label, { atol = 1e-5, rtol = 1e-5 } = {}) {
  if (actual.length !== expected.length) throw new Error(`${label}: length ${actual.length} != ${expected.length}`);
  let maxAbsError = 0;
  for (let i = 0; i < expected.length; i++) {
    const error = Math.abs(actual[i] - expected[i]);
    if (!Number.isFinite(actual[i]) || !Number.isFinite(expected[i]) || error > atol + rtol * Math.abs(expected[i])) {
      throw new Error(`${label}[${i}]: ${actual[i]} != ${expected[i]} (atol=${atol}, rtol=${rtol})`);
    }
    maxAbsError = Math.max(maxAbsError, error);
  }
  return { elements: expected.length, max_abs_error: maxAbsError, atol, rtol };
}

export async function checkWorkload(workload, expected, tolerance) {
  // Include eager construction, capture and replay, before taking samples.
  let validation;
  for (let run = 0; run < 3; run++) {
    const output = await workload.call();
    try {
      await workload.ready(output);
      const value = output.value ?? output;
      // JAX-JS data() consumes its reference; disposal below owns the original.
      const actual = value.toArrayAsync ? await value.toArrayAsync()
        : value.toArray ? await value.toArray() : await value.ref.data();
      validation = assertClose(actual, expected, workload.name, tolerance);
    } finally {
      workload.dispose?.(output);
    }
  }
  return validation;
}

export function genericReference(name, inputs) {
  const [a, b, c, d, x, row, col] = inputs;
  const f = Math.fround;
  const result = new Float32Array(name === 'broadcast_reduce_1024' ? 1024 : a.length);
  if (name === 'pointwise_1m') {
    for (let i = 0; i < result.length; i++) result[i] = Math.max(f(f(f(a[i]+b[i])*f(a[i]-b[i]))+f(a[i]*b[i])), 0);
  } else if (name === 'where_1m') {
    for (let i = 0; i < result.length; i++) result[i] = a[i] > b[i]
      ? f(f(a[i]+c[i])*f(b[i]-d[i])) : f(f(a[i]-c[i])*f(b[i]+d[i]));
  } else if (name === 'broadcast_reduce_1024') {
    for (let i = 0; i < 1024; i++) {
      let sum = 0;
      for (let j = 0; j < 1024; j++) sum += Math.max(f(f(f(x[i*1024+j]+row[i])*col[j])-0.25), 0);
      result[i] = sum;
    }
  } else if (name === 'transpose_copy_1024') {
    for (let i = 0; i < 512; i++) for (let j = 0; j < 2048; j++) result[j*512+i] = x[i*2048+j];
  } else throw new Error(`unknown generic reference ${name}`);
  return result;
}

export function matmulReference(n, transpose) {
  // makeMatrixData is period seven. Sum complete periods plus the tail so
  // checking the large default sizes does not require a cubic JS matmul.
  const result = new Float32Array(n*n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    let period = 0, tail = 0;
    for (let k = 0; k < 7; k++) {
      const product = ((i*n+k)%7-3) * ((transpose ? j*n+k : k*n+j)%7-3);
      period += product;
      if (k < n%7) tail += product;
    }
    result[i*n+j] = period*Math.floor(n/7)+tail;
  }
  return result;
}

export function modelReference(c, data) {
  const linear = (input, weights, bias, rows, cols, outputCols) => {
    const result = new Float32Array(rows*outputCols);
    for (let i = 0; i < rows; i++) for (let j = 0; j < outputCols; j++) {
      let sum = 0;
      for (let k = 0; k < cols; k++) sum += input[i*cols+k]*weights[j*cols+k];
      result[i*outputCols+j] = Math.fround(sum) + (bias?.[j] ?? 0);
    }
    return result;
  };
  const activate = x => c.activation === 'silu' ? x/(1+Math.exp(-x)) : Math.max(x,0);
  if (c.kind === 'mlp') {
    const hidden = linear(data.x, data.w1, data.b1, c.tokens, c.d, c.hidden);
    for (let i = 0; i < hidden.length; i++) hidden[i] = activate(hidden[i]);
    return linear(hidden, data.w2, data.b2, c.tokens, c.hidden, c.out);
  }
  const gate = linear(data.x, data.wg, null, c.tokens, c.d, c.hidden);
  const up = linear(data.x, data.wu, null, c.tokens, c.d, c.hidden);
  for (let i = 0; i < gate.length; i++) gate[i] = Math.fround(activate(gate[i]))*up[i];
  return linear(gate, data.wd, null, c.tokens, c.hidden, c.d);
}
