import test from 'node:test';
import assert from 'node:assert/strict';
import boundaryTools from '../js/test/test_jit_boundaries.js';
import { assessABBA, assessPagePairs, pagePairOrder, timeWorkload, measureKernelBlocks, assessKernelPages, decodedWatOpcodes, structuralSnapshot, assertClose, checkWorkload, genericReference, matmulReference, modelReference } from './wasm_checks.mjs';

test('kernel qualification executes paired real work and retains full-trace order', () => {
  let clock = 0;
  const order = [];
  const trace = [1, 2].map(arg => ({ fn: x => { assert.equal(x, arg); clock++; order.push(x); }, args: [arg] }));
  const rows = measureKernelBlocks(trace, { page: 0, blocks: 2, targetMs: 10, now: () => clock });
  assert.equal(rows.length, 3);
  for (const row of rows) {
    assert.equal(row.samples.length, 4);
    for (const sample of row.samples) {
      assert.equal(sample.ratio, sample.control === 'identity' ? 1 : 1.1);
      assert.equal(sample.candidateCalls * 10, sample.baselineCalls * (sample.control === 'kernel' ? 11 : 10));
    }
    assert.deepEqual(row.samples.filter(s => s.control === 'identity').map(s => s.order), ['AB', 'BA']);
  }
  const tail = order.slice(-22);
  assert.deepEqual(tail, Array.from({length: 22}, (_, i) => i % 2 + 1));
  assert.throws(() => measureKernelBlocks([], {}), /empty/);
  assert.throws(() => measureKernelBlocks(trace, { blocks: 3 }), /even/);
});

test('kernel qualification uses fresh pages and checks identities, counts and planted work', () => {
  const pages = Array.from({length: 20}, (_, page) => ({ kernelPage: page, pgResults: [{ name: 'mlp',
    kernelMeasurements: [{ unit: 'trace', hashes: ['abc'], repetitions: 10, samples: [
      { control: 'identity', baselineMs: 10, candidateMs: 10, baselineCalls: 10, candidateCalls: 10 },
      { control: 'kernel', baselineMs: 10, candidateMs: 11, baselineCalls: 10, candidateCalls: 11 },
    ] }],
  }] }));
  assert.equal(assessKernelPages(pages).qualified, true);
  assert.equal(assessKernelPages(pages).results[0].pairs, 20);
  assert.throws(() => assessKernelPages(pages.slice(0, 10)), /twenty/);
  pages[0].pgResults[0].kernelMeasurements[0].hashes = ['different'];
  assert.throws(() => assessKernelPages(pages), /identity/);
  pages[0].pgResults[0].kernelMeasurements[0].hashes = ['abc'];
  pages[0].pgResults[0].kernelMeasurements[0].samples[1].candidateCalls = 10;
  assert.throws(() => assessKernelPages(pages), /work count/);
});

test('qualification and normal timing use identical warmup; planted work is actually executed', async () => {
  for (const inject of [null, 'kernel', 'host']) {
    const observer = { trace: null };
    let calls = 0, bodies = 0, burns = 0, disposed = 0;
    const body = () => { bodies++; };
    const result = await timeWorkload(async () => {
      calls++; body(); observer.trace?.push({ fn: body, args: [] }); return {};
    }, async () => {}, () => { disposed++; }, { iters: 20, warmup: 5, observer, inject, burn: () => { burns++; } });
    assert.equal(calls, 26); assert.equal(disposed, 26);
    assert.equal(bodies, inject === 'kernel' ? 28 : 26);
    assert.equal(result.extraKernelCalls, bodies - calls);
    assert.equal(burns, inject === 'host' ? 20 : 0);
    assert.equal(result.hostBurnCalls, burns);
    assert.equal(result.samples.length, 20);
    assert.equal(observer.trace, null);
  }
});

test('timing fails closed on an absent kernel trace and disposes failed reads', async () => {
  let disposed = 0;
  const observer = { trace: null };
  await assert.rejects(timeWorkload(() => ({}), async () => {}, () => { disposed++; },
    { iters: 10, warmup: 0, observer, inject: 'kernel' }), /no dispatches/);
  assert.equal(disposed, 11); assert.equal(observer.trace, null);
  await assert.rejects(timeWorkload(() => ({}), async () => { throw new Error('read failed'); }, () => { disposed++; },
    { iters: 10, warmup: 0 }), /read failed/);
  assert.equal(disposed, 12);
});

test('opcode snapshots count decoded instructions, not data or export names', () => {
  assert.deepEqual(decodedWatOpcodes('(module (export "(f32.mul)" (func $0)) (func $0 (param i32) (result i32) (i32.add (i32.const 253) (local.get 0))))'),
    { 'i32.add': 1, 'i32.const': 1, 'local.get': 1 });
  assert.throws(() => structuralSnapshot({ args: { device: 'wasm' }, pages: [{ pgResults: [{ name: 'missing' }] }] }), /missing structural/);
});

test('Wasm observer captures real dispatches without changing values or function arity', () => {
  // () -> i32, exported as kernel; returns 7. No mocked dispatch counter.
  const bytes = Uint8Array.from([0,97,115,109,1,0,0,0,1,5,1,96,0,1,127,
    3,2,1,0,7,10,1,6,107,101,114,110,101,108,0,0,10,6,1,4,0,65,7,11]);
  const original = WebAssembly.Instance;
  const state = boundaryTools.observeBoundaries();
  try {
    const { kernel } = new WebAssembly.Instance(new WebAssembly.Module(bytes)).exports;
    assert.equal(kernel.length, 0);
    assert.equal(kernel(), 7);
    assert.equal(state.kernels, 0);
    state.active = true; state.trace = [];
    assert.equal(kernel(), 7);
    assert.equal(state.kernels, 1);
    assert.equal(state.trace.length, 1);
    assert.ok(state.trace[0].module instanceof WebAssembly.Module);
    assert.equal(state.trace[0].fn(...state.trace[0].args), 7);
    assert.equal(state.kernels, 1); // direct body replay excludes the wrapper
    state.executeKernels = false;
    assert.equal(kernel(), undefined);
    assert.equal(state.kernels, 2);
  } finally { state.restore(); }
  assert.equal(WebAssembly.Instance, original);
});

test('page-pair gate balances order and uses pages, not inner iterations, as samples', () => {
  const pages = (ratio, n = 10) => pagePairOrder(n).map(({ pair, lane }) => ({
    pair, lane, pgResults: [{ name: 'mlp', seconds: (1 + pair / 10) * (lane === 'candidate' ? ratio : 1),
      samples: Array(100).fill(1) }],
  }));
  assert.deepEqual(pagePairOrder(2), [
    { pair: 0, lane: 'baseline' }, { pair: 0, lane: 'candidate' },
    { pair: 1, lane: 'candidate' }, { pair: 1, lane: 'baseline' },
  ]);
  assert.throws(() => pagePairOrder(3), /even/);
  const identity = assessPagePairs(pages(1), 0.05)[0];
  assert.equal(identity.status, 'equivalent');
  assert.deepEqual(identity.interval, [1, 1]);
  assert.equal(identity.pairs, 10);
  assert.equal(assessPagePairs(pages(1.1), 0.05)[0].status, 'regression');
  assert.equal(assessPagePairs(pages(.9), 0.05)[0].status, 'improvement');
  assert.equal(assessPagePairs(pages(1.04), 0.05)[0].status, 'equivalent');
  const noisy = pages(1);
  noisy.filter(p => p.lane === 'candidate').forEach((p, i) => { p.pgResults[0].seconds *= i % 2 ? 1.2 : .8; });
  assert.equal(assessPagePairs(noisy, 0.05)[0].status, 'inconclusive');
  assert.throws(() => assessPagePairs(pages(1, 2), .05), /ten/);
  const missing = pages(1); missing[2].pgResults = [];
  assert.throws(() => assessPagePairs(missing, .05), /cases/);
  const bad = pages(1); bad[3].pgResults[0].seconds = NaN;
  assert.throws(() => assessPagePairs(bad, .05), /timing/);
  assert.throws(() => assessPagePairs(pages(1).reverse(), .05), /order/);
});

test('qualification reports its extreme-based interval without hiding an outlier', () => {
  const pages = pagePairOrder(10).map(({ pair, lane }) => ({ pair, lane,
    pgResults: Array.from({ length: 5 }, (_, i) => ({ name: `case${i}`,
      seconds: lane === 'candidate' && pair === 0 ? 1.2 : 1 })),
  }));
  for (const row of assessPagePairs(pages, .05, .05 / 3)) {
    assert.equal(row.intervalOrder.trimmedPairsPerTail, 0);
    assert.equal(row.intervalOrder.comparisons, 5);
    assert.deepEqual(row.interval, [1, 1.2]);
    assert.equal(row.status, 'inconclusive');
  }
  for (const p of pages) for (const r of p.pgResults) r.seconds = 1;
  assert.ok(assessPagePairs(pages, .05, .05 / 3).every(r => r.status === 'equivalent'));
});

test('ABBA gate distinguishes regressions, passes and noisy comparisons', () => {
  const page = ms => ({ pgResults: [{ name: 'mlp', seconds: ms / 1000 }] });
  const assess = times => assessABBA(times.map(page), 0.1)[0];
  assert.equal(assess([2, 7, 7.1, 2.1]).status, 'regression');
  assert.equal(assess([2, 2.05, 2.1, 2.1]).status, 'pass');
  assert.equal(assess([2, 2.05, 3, 2.1]).status, 'inconclusive');
  assert.equal(assess([2, 1, 1.1, 2.1]).status, 'pass');
  assert.throws(() => assessABBA([page(1)], 0.1), /four/);
  assert.throws(() => assessABBA([page(1), page(NaN), page(1), page(1)], 0.1), /timing/);
  assert.throws(() => assessABBA([page(1), { pgResults: [] }, page(1), page(1)], 0.1), /cases/);
  assert.throws(() => assessABBA(Array(4).fill(page(1)), -0.1), /threshold/);
});

test('historical Polygrad bundles are checked through their public async toArray', async () => {
  let reads = 0;
  await checkWorkload({ name: 'historical', call: () => ({
    async toArray() { reads++; return [2]; },
  }), ready: async () => {} }, [2]);
  assert.equal(reads, 3);
});

test('comparison rejects wrong lengths, nonfinite and wrong values', () => {
  assert.throws(() => assertClose([], [1], 'length'), /length/);
  for (const value of [NaN, Infinity, 0]) assert.throws(() => assertClose([value], [1], 'wrong'), /wrong/);
  assert.equal(assertClose([1], [1], 'equal').max_abs_error, 0);
});

test('periodic matmul reference matches independent cubic products, including tails', () => {
  for (const n of [1, 2, 3, 7, 8, 13]) for (const transpose of [false, true]) {
    const actual = matmulReference(n, transpose);
    const data = Float32Array.from({length: n*n}, (_,i) => i%7-3);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      let expected = 0;
      for (let k = 0; k < n; k++) expected += data[i*n+k]*data[transpose ? j*n+k : k*n+j];
      assert.equal(actual[i*n+j], expected);
    }
  }
});

test('generic reference evaluates both WHERE branches', () => {
  const inputs = [[3,-2],[1,4],[2,3],[1,2]];
  assert.deepEqual([...genericReference('pointwise_1m', inputs)], [11,0]);
  assert.deepEqual([...genericReference('where_1m', inputs)], [0,-30]);
});

test('model reference applies bias and activation between layers', () => {
  const c = { kind:'mlp', tokens:1, d:2, hidden:2, out:1, activation:'relu' };
  const data = { x:[1,2], w1:[1,0,0,1], b1:[-2,1], w2:[5,7], b2:[1] };
  assert.deepEqual([...modelReference(c,data)], [22]);
});

test('validation checks replay and balances consuming JAX-JS reads', async () => {
  let live=0, calls=0;
  const workload = {
    name:'owned',
    call() {
      calls++; live++;
      return { get ref() { live++; return { async data() { live--; return [2]; } }; } };
    },
    async ready() {}, dispose() { live--; },
  };
  await checkWorkload(workload, [2]);
  assert.equal(calls,3); assert.equal(live,0);
  await assert.rejects(checkWorkload(workload,[3]), /owned/);
  assert.equal(live,0);
});
