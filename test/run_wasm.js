#!/usr/bin/env node
/*
 * run_wasm.js — Node.js test runner for polygrad WASM kernels
 *
 * Usage: node run_wasm.js <wasm_file> <op> <n>
 *
 * Loads a WASM module, instantiates with shared memory,
 * sets up input buffers, calls the kernel, and verifies output.
 */

const fs = require('fs');
const path = require('path');

async function main() {
  const [wasmFile, op, nStr] = process.argv.slice(2);
  if (!wasmFile || !op || !nStr) {
    console.error('Usage: node run_wasm.js <wasm_file> <op> <n>');
    process.exit(1);
  }

  const N = parseInt(nStr);
  const bytes = fs.readFileSync(wasmFile);

  // Create shared memory (1 page = 64KB, enough for test data)
  const memory = new WebAssembly.Memory({ initial: 1 });

  // Math imports for transcendental functions
  const imports = {
    env: { memory },
    math: {
      exp2f: x => Math.pow(2, x),
      log2f: x => Math.log2(x),
      sinf: x => Math.sin(x),
      exp2: x => Math.pow(2, x),
      log2: x => Math.log2(x),
      sin: x => Math.sin(x),
      powf: (a, b) => Math.pow(a, b),
      pow: (a, b) => Math.pow(a, b),
    },
  };

  let module;
  try {
    module = await WebAssembly.instantiate(bytes, imports);
  } catch (e) {
    console.error('WASM instantiation failed:', e.message);
    process.exit(1);
  }

  const kernel = module.instance.exports.kernel;

  // Detect f64 ops (op name ends with _f64)
  const isF64 = op.endsWith('_f64');
  const baseOp = isF64 ? op.slice(0, -4) : op;
  const elemSize = isF64 ? 8 : 4;
  const view = isF64 ? new Float64Array(memory.buffer) : new Float32Array(memory.buffer);

  // Compute byte offsets for buffers
  // For binary ops: a (offset 0), b (offset N*elemSize), c (offset N*2*elemSize)
  // For unary ops:  a (offset 0), b (offset N*elemSize)
  const offA = 0;
  const offB = N * elemSize;
  const offC = N * 2 * elemSize;

  // Initialize input data
  for (let i = 0; i < N; i++) {
    const a = baseOp === 'pow' && isF64 ? i + 1.25 : i + 1;
    const b = baseOp === 'pow' && isF64 ? i * 0.375 + 0.625 : (i + 1) * 2;
    view[offA / elemSize + i] = a;
    view[offB / elemSize + i] = b;
    view[offC / elemSize + i] = 0;           // c = [0, ...]
  }

  // Call kernel
  if (baseOp === 'add' || baseOp === 'mul' || baseOp === 'sub' || baseOp === 'pow') {
    kernel(offA, offB, offC);
  } else if (baseOp === 'neg' || baseOp === 'sqrt' || baseOp === 'exp2' ||
             baseOp === 'log2' || baseOp === 'sin') {
    kernel(offA, offB); // unary: b = op(a)
  } else {
    console.error('Unknown op:', baseOp);
    process.exit(1);
  }

  // Verify output
  let pass = true;
  const tol = baseOp === 'pow' ? (isF64 ? 1e-13 : 2e-3) : (isF64 ? 1e-14 : 1e-5);

  for (let i = 0; i < N; i++) {
    const a = baseOp === 'pow' && isF64 ? i + 1.25 : i + 1;
    const b = baseOp === 'pow' && isF64 ? i * 0.375 + 0.625 : (i + 1) * 2;
    let expected;

    switch (baseOp) {
      case 'add': expected = a + b; break;
      case 'mul': expected = a * b; break;
      case 'sub': expected = a - b; break;
      case 'pow': expected = Math.pow(a, b); break;
      case 'neg': expected = -a; break;
      case 'sqrt': expected = Math.sqrt(a); break;
      case 'exp2': expected = Math.pow(2, a); break;
      case 'log2': expected = Math.log2(a); break;
      case 'sin': expected = Math.sin(a); break;
      default: expected = 0;
    }

    // Read result from appropriate buffer
    let result;
    if (baseOp === 'neg' || baseOp === 'sqrt' || baseOp === 'exp2' ||
        baseOp === 'log2' || baseOp === 'sin') {
      result = view[offB / elemSize + i]; // unary: result in b
    } else {
      result = view[offC / elemSize + i]; // binary: result in c
    }

    if (Math.abs(result - expected) > tol) {
      console.error(`FAIL: ${op}[${i}] = ${result}, expected ${expected}`);
      pass = false;
    }
  }

  if (pass) {
    console.log(`PASS: ${op} N=${N}`);
    process.exit(0);
  } else {
    process.exit(1);
  }
}

main().catch(e => { console.error(e); process.exit(1); });
