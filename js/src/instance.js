/**
 * instance.js -- PolyInstance wrapper for Node.js (koffi FFI)
 *
 * High-level API for running models created from IR bytes or family
 * builders (MLP). Independent of the Tensor class.
 */

'use strict'

const ffi = require('./ffi')
const { koffi } = ffi

// PolyIOBinding struct for forward/train bindings
const PolyIOBinding = koffi.struct('PolyIOBinding', {
  name: 'const char *',
  data: 'float *'
})
const BINDING_SIZE = koffi.sizeof(PolyIOBinding)

// Role constants
const ROLE_PARAM = 0
const ROLE_INPUT = 1
const ROLE_TARGET = 2
const ROLE_OUTPUT = 3
const ROLE_AUX = 4

// Optimizer constants
const OPTIM_NONE = 0
const OPTIM_SGD = 1
const OPTIM_ADAM = 2
const OPTIM_ADAMW = 3

class Instance {
  constructor(ptr) {
    if (!ptr) throw new Error('Failed to create PolyInstance (NULL pointer)')
    this._ptr = ptr
  }

  free() {
    if (this._ptr) {
      ffi.poly_instance_free(this._ptr)
      this._ptr = null
    }
  }

  // ── Lifecycle ──────────────────────────────────────────────────

  static fromIR(irBytes, weightsBytes) {
    const irBuf = Buffer.from(irBytes)
    let wBuf = null, wLen = 0
    if (weightsBytes) {
      wBuf = Buffer.from(weightsBytes)
      wLen = wBuf.length
    }
    return new Instance(ffi.poly_instance_from_ir(irBuf, irBuf.length, wBuf, wLen))
  }

  static mlp(spec) {
    if (typeof spec === 'object') spec = JSON.stringify(spec)
    const buf = Buffer.from(spec, 'utf-8')
    return new Instance(ffi.poly_mlp_instance(buf, buf.length))
  }

  /**
   * Load a HuggingFace model from config JSON + safetensors buffers.
   * @param {string|object} configJson - config.json content or parsed object.
   * @param {Buffer[]} weightBuffers - Array of safetensors file Buffers.
   * @param {number} [maxBatch=1] - Maximum batch size.
   * @param {number} [maxSeqLen=0] - Maximum sequence length (0 = config default).
   */
  static fromHF(configJson, weightBuffers, maxBatch = 1, maxSeqLen = 0) {
    if (typeof configJson === 'object') configJson = JSON.stringify(configJson)
    const configBuf = Buffer.from(configJson, 'utf-8')

    const n = weightBuffers.length
    // Build parallel arrays of pointers and lengths for koffi
    const filePtrs = weightBuffers.map(b => Buffer.from(b))
    const fileLens = new BigInt64Array(n)
    for (let i = 0; i < n; i++) fileLens[i] = BigInt(filePtrs[i].length)

    const ptr = ffi.poly_hf_load(
      configBuf, configBuf.length,
      filePtrs, fileLens,
      n, maxBatch, maxSeqLen
    )
    return new Instance(ptr)
  }

  // ── Param Enumeration ─────────────────────────────────────────

  get paramCount() {
    return ffi.poly_instance_param_count(this._ptr)
  }

  paramName(i) {
    return ffi.poly_instance_param_name(this._ptr, i)
  }

  paramShape(i) {
    const buf = new BigInt64Array(8)
    const ndim = ffi.poly_instance_param_shape(this._ptr, i, buf, 8)
    return Array.from(buf.slice(0, ndim), Number)
  }

  paramData(i) {
    const numel = new BigInt64Array(1)
    const ptr = ffi.poly_instance_param_data(this._ptr, i, numel)
    if (!ptr) return null
    const n = Number(numel[0])
    return koffi.decode(ptr, 'float', n)
  }

  setParamData(i, arr) {
    const numel = new BigInt64Array(1)
    const ptr = ffi.poly_instance_param_data(this._ptr, i, numel)
    if (!ptr) return
    const n = Number(numel[0])
    const data = arr instanceof Float32Array ? arr : new Float32Array(arr)
    const type = koffi.array('float', n)
    koffi.encode(ptr, 0, type, Array.from(data.subarray(0, n)))
  }

  // ── Buffer Enumeration ────────────────────────────────────────

  get bufCount() {
    return ffi.poly_instance_buf_count(this._ptr)
  }

  bufName(i) {
    return ffi.poly_instance_buf_name(this._ptr, i)
  }

  bufRole(i) {
    return ffi.poly_instance_buf_role(this._ptr, i)
  }

  bufShape(i) {
    const buf = new BigInt64Array(8)
    const ndim = ffi.poly_instance_buf_shape(this._ptr, i, buf, 8)
    return Array.from(buf.slice(0, ndim), Number)
  }

  bufData(i) {
    const numel = new BigInt64Array(1)
    const ptr = ffi.poly_instance_buf_data(this._ptr, i, numel)
    if (!ptr) return null
    const n = Number(numel[0])
    return koffi.decode(ptr, 'float', n)
  }

  setBufData(i, arr) {
    const numel = new BigInt64Array(1)
    const ptr = ffi.poly_instance_buf_data(this._ptr, i, numel)
    if (!ptr) return
    const n = Number(numel[0])
    const data = arr instanceof Float32Array ? arr : new Float32Array(arr)
    const type = koffi.array('float', n)
    koffi.encode(ptr, 0, type, Array.from(data.subarray(0, n)))
  }

  findBuf(name) {
    for (let i = 0; i < this.bufCount; i++) {
      if (this.bufName(i) === name) return i
    }
    return -1
  }

  // ── Weight I/O ────────────────────────────────────────────────

  exportWeights() {
    const outLen = new Int32Array(1)
    const ptr = ffi.poly_instance_export_weights(this._ptr, outLen)
    if (!ptr) return null
    const len = outLen[0]
    return Buffer.from(koffi.decode(ptr, 'uint8_t', len))
  }

  importWeights(data) {
    const buf = Buffer.from(data)
    const ret = ffi.poly_instance_import_weights(this._ptr, buf, buf.length)
    if (ret !== 0) throw new Error(`importWeights failed (ret=${ret})`)
  }

  exportIR() {
    const outLen = new Int32Array(1)
    const ptr = ffi.poly_instance_export_ir(this._ptr, outLen)
    if (!ptr) return null
    const len = outLen[0]
    return Buffer.from(koffi.decode(ptr, 'uint8_t', len))
  }

  // ── Execution ─────────────────────────────────────────────────

  setOptimizer(kind, lr = 0.01, beta1 = 0.9, beta2 = 0.999,
               eps = 1e-8, weightDecay = 0.0) {
    const ret = ffi.poly_instance_set_optimizer(
      this._ptr, kind, lr, beta1, beta2, eps, weightDecay)
    if (ret !== 0) throw new Error(`setOptimizer failed (ret=${ret})`)
  }

  forward(inputs) {
    const { buf, n } = this._makeBindings(inputs)
    const ret = ffi.poly_instance_forward(this._ptr, buf, n)
    if (ret !== 0) throw new Error(`forward failed (ret=${ret})`)
    return this._collectOutputs()
  }

  trainStep(io) {
    const { buf, n } = this._makeBindings(io)
    const loss = new Float32Array(1)
    const ret = ffi.poly_instance_train_step(this._ptr, buf, n, loss)
    if (ret !== 0) throw new Error(`trainStep failed (ret=${ret})`)
    return loss[0]
  }

  // ── Internals ─────────────────────────────────────────────────

  _makeBindings(ioObj) {
    const entries = Object.entries(ioObj)
    const n = entries.length
    if (n === 0) return { buf: null, n: 0 }
    const ptr = koffi.alloc(PolyIOBinding, n)

    // Keep references alive during the call
    this._bindingRefs = []

    for (let i = 0; i < n; i++) {
      const [name, data] = entries[i]
      const dataArr = data instanceof Float32Array ? data : new Float32Array(data)
      this._bindingRefs.push(dataArr)
      koffi.encode(ptr, i * BINDING_SIZE, PolyIOBinding, {
        name: name,
        data: dataArr
      })
    }

    return { buf: ptr, n }
  }

  _collectOutputs() {
    const result = {}
    for (let i = 0; i < this.bufCount; i++) {
      if (this.bufRole(i) === ROLE_OUTPUT) {
        const name = this.bufName(i)
        const data = this.bufData(i)
        if (data) result[name] = new Float32Array(data)
      }
    }
    return result
  }
}

module.exports = {
  Instance,
  ROLE_PARAM, ROLE_INPUT, ROLE_TARGET, ROLE_OUTPUT, ROLE_AUX,
  OPTIM_NONE, OPTIM_SGD, OPTIM_ADAM, OPTIM_ADAMW
}
