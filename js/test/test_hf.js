/**
 * Tests for HuggingFace model loading (JS Node).
 * Mirrors py/tests/test_hf.py using tape.
 */

'use strict'

const test = require('tape')
const { Instance, ROLE_PARAM, ROLE_OUTPUT } = require('../src/instance')

// --- Safetensors builder ---

/**
 * Build a minimal safetensors file from tensor descriptors.
 * @param {Object} tensors - {name: {dtype, shape, data: Float32Array|Uint16Array}}
 * @returns {Buffer}
 */
function makeSafetensors(tensors) {
  const header = {}
  let offset = 0
  const dataParts = []

  for (const [name, { dtype, shape, data }] of Object.entries(tensors)) {
    const raw = Buffer.from(data.buffer, data.byteOffset, data.byteLength)
    header[name] = {
      dtype,
      shape,
      data_offsets: [offset, offset + raw.length]
    }
    dataParts.push(raw)
    offset += raw.length
  }

  const headerJson = Buffer.from(JSON.stringify(header), 'utf-8')
  const headerSize = Buffer.alloc(8)
  headerSize.writeBigUInt64LE(BigInt(headerJson.length))

  return Buffer.concat([headerSize, headerJson, ...dataParts])
}

// --- Config ---

const GPT2_TINY_CONFIG = {
  model_type: 'gpt2',
  vocab_size: 32,
  n_embd: 16,
  n_head: 2,
  n_layer: 1,
  n_positions: 8,
  layer_norm_epsilon: 1e-5
}

// --- HF Loading: Basic ---

test('hf: load from config + weights', (t) => {
  const wte = new Float32Array(32 * 16)
  const st = makeSafetensors({
    'transformer.wte.weight': { dtype: 'F32', shape: [32, 16], data: wte }
  })
  const inst = Instance.fromHF(GPT2_TINY_CONFIG, [st])
  t.equal(inst.paramCount, 16, '16 params: wte+wpe+1layer(12)+ln_f(2)')
  inst.free()
  t.end()
})

test('hf: param names', (t) => {
  const wte = new Float32Array(32 * 16)
  const st = makeSafetensors({
    'transformer.wte.weight': { dtype: 'F32', shape: [32, 16], data: wte }
  })
  const inst = Instance.fromHF(GPT2_TINY_CONFIG, [st])

  const names = new Set()
  for (let i = 0; i < inst.paramCount; i++) names.add(inst.paramName(i))

  t.ok(names.has('wte.weight'), 'has wte.weight')
  t.ok(names.has('wpe.weight'), 'has wpe.weight')
  t.ok(names.has('h.0.ln_1.weight'), 'has h.0.ln_1.weight')
  t.ok(names.has('h.0.attn.c_attn.weight'), 'has h.0.attn.c_attn.weight')
  t.ok(names.has('h.0.mlp.c_fc.weight'), 'has h.0.mlp.c_fc.weight')
  t.ok(names.has('ln_f.weight'), 'has ln_f.weight')
  inst.free()
  t.end()
})

test('hf: weight data loaded', (t) => {
  const wte = new Float32Array(32 * 16)
  for (let i = 0; i < wte.length; i++) wte[i] = i * 0.001
  const st = makeSafetensors({
    'transformer.wte.weight': { dtype: 'F32', shape: [32, 16], data: wte }
  })
  const inst = Instance.fromHF(GPT2_TINY_CONFIG, [st])

  for (let i = 0; i < inst.paramCount; i++) {
    if (inst.paramName(i) === 'wte.weight') {
      const data = inst.paramData(i)
      for (let j = 0; j < 5; j++) {
        t.ok(Math.abs(data[j] - wte[j]) < 1e-6, `wte[${j}] matches`)
      }
      break
    }
  }
  inst.free()
  t.end()
})

test('hf: model prefix stripping', (t) => {
  const wte = new Float32Array(32 * 16).fill(1.0)
  const st = makeSafetensors({
    'model.wte.weight': { dtype: 'F32', shape: [32, 16], data: wte }
  })
  const inst = Instance.fromHF(GPT2_TINY_CONFIG, [st])
  t.ok(inst, 'loaded with model. prefix')
  inst.free()
  t.end()
})

// --- HF Loading: Edge Cases ---

test('hf: unsupported model type throws', (t) => {
  const config = { model_type: 'llama', vocab_size: 100 }
  t.throws(() => Instance.fromHF(config, []), /NULL/, 'throws on unsupported model')
  t.end()
})

test('hf: attn.bias ignored', (t) => {
  const bias = new Float32Array(1 * 1 * 8 * 8)
  const st = makeSafetensors({
    'transformer.h.0.attn.bias': { dtype: 'F32', shape: [1, 1, 8, 8], data: bias }
  })
  const inst = Instance.fromHF(GPT2_TINY_CONFIG, [st])
  t.ok(inst, 'loaded despite attn.bias')
  inst.free()
  t.end()
})

test('hf: lm_head.weight skipped (weight tying)', (t) => {
  const wte = new Float32Array(32 * 16).fill(0.5)
  const lmHead = new Float32Array(32 * 16).fill(0.9)
  const st = makeSafetensors({
    'transformer.wte.weight': { dtype: 'F32', shape: [32, 16], data: wte },
    'lm_head.weight': { dtype: 'F32', shape: [32, 16], data: lmHead }
  })
  const inst = Instance.fromHF(GPT2_TINY_CONFIG, [st])

  for (let i = 0; i < inst.paramCount; i++) {
    if (inst.paramName(i) === 'wte.weight') {
      const data = inst.paramData(i)
      t.ok(Math.abs(data[0] - 0.5) < 1e-6, 'wte has wte data, not lm_head')
      break
    }
  }
  inst.free()
  t.end()
})

test('hf: multiple weight files (sharded)', (t) => {
  const wte = new Float32Array(32 * 16).fill(0.1)
  const wpe = new Float32Array(8 * 16).fill(0.2)
  const st1 = makeSafetensors({
    'transformer.wte.weight': { dtype: 'F32', shape: [32, 16], data: wte }
  })
  const st2 = makeSafetensors({
    'transformer.wpe.weight': { dtype: 'F32', shape: [8, 16], data: wpe }
  })
  const inst = Instance.fromHF(GPT2_TINY_CONFIG, [st1, st2])

  let foundWte = false, foundWpe = false
  for (let i = 0; i < inst.paramCount; i++) {
    const name = inst.paramName(i)
    const data = inst.paramData(i)
    if (name === 'wte.weight') {
      t.ok(Math.abs(data[0] - 0.1) < 1e-6, 'wte loaded')
      foundWte = true
    } else if (name === 'wpe.weight') {
      t.ok(Math.abs(data[0] - 0.2) < 1e-6, 'wpe loaded')
      foundWpe = true
    }
  }
  t.ok(foundWte, 'found wte')
  t.ok(foundWpe, 'found wpe')
  inst.free()
  t.end()
})

// --- Multi-layer ---

test('hf: 3-layer model param count', (t) => {
  const config = {
    model_type: 'gpt2',
    vocab_size: 64,
    n_embd: 32,
    n_head: 4,
    n_layer: 3,
    n_positions: 16,
    layer_norm_epsilon: 1e-5
  }
  const st = makeSafetensors({
    'transformer.wte.weight': { dtype: 'F32', shape: [64, 32], data: new Float32Array(64 * 32) }
  })
  const inst = Instance.fromHF(config, [st])
  // 2 (wte+wpe) + 3*12 (layers) + 2 (ln_f) = 40
  t.equal(inst.paramCount, 40, '40 params for 3-layer')
  inst.free()
  t.end()
})

// --- Forward Pass ---

test('hf: gpt2 forward e2e', (t) => {
  const inst = Instance.fromHF(GPT2_TINY_CONFIG, [
    makeSafetensors({
      'transformer.wte.weight': { dtype: 'F32', shape: [32, 16], data: new Float32Array(32 * 16) }
    })
  ])

  // Init weights: LN weights=1, biases=0, others=small values
  for (let i = 0; i < inst.paramCount; i++) {
    const name = inst.paramName(i)
    const data = inst.paramData(i)
    if (!data) continue
    const arr = new Float32Array(data.length)
    if (name.includes('ln_') && name.includes('weight')) {
      arr.fill(1.0)
    } else if (name.includes('bias')) {
      arr.fill(0.0)
    } else {
      for (let j = 0; j < arr.length; j++) arr[j] = 0.02 * (j % 100 / 100 - 0.5)
    }
    inst.setParamData(i, arr)
  }

  // Set input/aux buffers (skip params and outputs)
  for (let i = 0; i < inst.bufCount; i++) {
    const name = inst.bufName(i)
    const role = inst.bufRole(i)
    if (role === ROLE_PARAM || role === ROLE_OUTPUT) continue
    const data = inst.bufData(i)
    if (!data) continue
    const arr = new Float32Array(data.length)
    if (name === 'x') {
      for (let j = 0; j < arr.length; j++) arr[j] = j % 4
    } else if (name === 'positions' || name === 'arange') {
      for (let j = 0; j < arr.length; j++) arr[j] = j
    } else if (name === 'vocab_arange') {
      for (let j = 0; j < arr.length; j++) arr[j] = j
    }
    inst.setBufData(i, arr)
  }

  // Run forward
  const outputs = inst.forward({})
  t.ok(outputs.output, 'has output')
  // Output should be (1, 8, 32) = 256 floats
  t.equal(outputs.output.length, 1 * 8 * 32, 'output size')

  // Verify finite and not all zero
  let allZero = true
  for (let j = 0; j < outputs.output.length; j++) {
    t.ok(isFinite(outputs.output[j]), `output[${j}] finite`)
    if (Math.abs(outputs.output[j]) > 1e-10) allZero = false
    if (j > 10) break // don't test all 256 individually
  }
  t.ok(!allZero, 'output not all zero')

  inst.free()
  t.end()
})
