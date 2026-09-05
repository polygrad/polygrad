// Untimed, backend-independent references for the Wasm comparison workloads.
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
      const actual = value.toArrayAsync ? await value.toArrayAsync() : await value.ref.data();
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
