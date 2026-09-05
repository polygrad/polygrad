import test from 'node:test';
import assert from 'node:assert/strict';
import { assertClose, checkWorkload, genericReference, matmulReference, modelReference } from './wasm_checks.mjs';

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
