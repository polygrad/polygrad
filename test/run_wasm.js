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
      powf: (a, b) => Math.pow(a, b),
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
  const view = new Float32Array(memory.buffer);

  // Compute byte offsets for buffers
  // For binary ops: a (offset 0), b (offset N*4), c (offset N*8)
  // For unary ops:  a (offset 0), b (offset N*4)
  const offA = 0;
  const offB = N * 4;
  const offC = N * 8;

  // Initialize input data
  for (let i = 0; i < N; i++) {
    view[offA / 4 + i] = i + 1;       // a = [1, 2, 3, ...]
    view[offB / 4 + i] = (i + 1) * 2; // b = [2, 4, 6, ...]
    view[offC / 4 + i] = 0;           // c = [0, ...]
  }

  // Call kernel
  if (op === 'add' || op === 'mul' || op === 'sub' || op === 'pow') {
    kernel(offA, offB, offC);
  } else if (op === 'neg' || op === 'sqrt') {
    kernel(offA, offB); // unary: b = op(a)
  } else {
    console.error('Unknown op:', op);
    process.exit(1);
  }

  // Verify output
  let pass = true;
  const tol = 1e-5;

  for (let i = 0; i < N; i++) {
    const a = i + 1;
    const b = (i + 1) * 2;
    let expected;

    switch (op) {
      case 'add': expected = a + b; break;
      case 'mul': expected = a * b; break;
      case 'sub': expected = a - b; break;
      case 'pow': expected = Math.pow(a, b); break;
      case 'neg': expected = -a; break;
      case 'sqrt': expected = Math.sqrt(a); break;
      default: expected = 0;
    }

    // Read result from appropriate buffer
    let result;
    if (op === 'neg' || op === 'sqrt') {
      result = view[offB / 4 + i]; // unary: result in b
    } else {
      result = view[offC / 4 + i]; // binary: result in c
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
