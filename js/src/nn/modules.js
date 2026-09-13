'use strict'

const { isIntegerDtype } = require('../tensor')

function createBoundModules(runtime) {
  const Tensor = runtime.Tensor

  class Linear {
    constructor(inFeatures, outFeatures, opts = {}) {
      this.inFeatures = Number(inFeatures)
      this.outFeatures = Number(outFeatures)
      const bound = 1.0 / Math.sqrt(this.inFeatures)
      this.weight = Tensor.uniform(this.outFeatures, this.inFeatures, {
        low: -bound, high: bound
      })
      this.bias = opts.bias === false
        ? null
        : Tensor.uniform(this.outFeatures, { low: -bound, high: bound })
    }

    call(x) {
      const core = runtime._core.ffi.poly_tensor_linear_apply(x._ctx, x._tensor,
        this.weight._tensor, this.bias === null ? null : this.bias._tensor)
      return x._makeResultFromCore(core)
    }
  }

  class LayerNorm {
    constructor(normalizedShape, opts = {}) {
      this.normalizedShape = Array.isArray(normalizedShape)
        ? Array.from(normalizedShape, Number)
        : [Number(normalizedShape)]
      this.axis = this.normalizedShape.map((_, i) => -1 - i)
      this.eps = opts.eps == null ? 1e-5 : Number(opts.eps)
      const affine = opts.elementwiseAffine !== false
      this.weight = affine ? Tensor.ones(...this.normalizedShape) : null
      this.bias = affine ? Tensor.zeros(...this.normalizedShape) : null
    }

    call(x) {
      const tail = x.shape.slice(-this.normalizedShape.length)
      if (tail.length !== this.normalizedShape.length ||
          tail.some((dim, i) => dim !== this.normalizedShape[i])) {
        throw new Error(`last dimensions of ${JSON.stringify(x.shape)} must match ${JSON.stringify(this.normalizedShape)}`)
      }
      return x._makeResultFromCore(runtime._core.ffi.poly_tensor_layernorm_axes_apply(x._ctx, x._tensor,
        this.weight === null ? null : this.weight._tensor, this.bias === null ? null : this.bias._tensor, this.axis, this.eps))
    }
  }

  class LayerNorm2d extends LayerNorm {
    call(x) {
      // Pinned tinygrad/nn/__init__.py:263-278.
      return super.call(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    }
  }

  class Conv2d {
    constructor(inChannels, outChannels, kernelSize, opts = {}) {
      this.inChannels = Number(inChannels)
      this.outChannels = Number(outChannels)
      const k = typeof kernelSize === 'number' ? [kernelSize, kernelSize] : Array.from(kernelSize)
      this.kernelSize = k
      this.stride = opts.stride == null
        ? Array(k.length).fill(1)
        : (typeof opts.stride === 'number' ? Array(k.length).fill(opts.stride) : Array.from(opts.stride))
      this.dilation = opts.dilation == null
        ? Array(k.length).fill(1)
        : (typeof opts.dilation === 'number' ? Array(k.length).fill(opts.dilation) : Array.from(opts.dilation))
      this.groups = opts.groups == null ? 1 : Number(opts.groups)
      if (typeof opts.padding === 'string') {
        if (opts.padding.toLowerCase() !== 'same') {
          throw new Error(`Invalid padding string ${opts.padding}, only 'same' is supported`)
        }
        if (!this.stride.every(s => s === 1)) {
          throw new Error("padding='same' is not supported for strided convolutions")
        }
        this.padding = []
        for (let i = 0; i < k.length; i++) {
          const d = this.dilation[i]
          const kk = k[k.length - 1 - i]
          this.padding.push(Math.floor(d * (kk - 1) / 2), d * (kk - 1) - Math.floor(d * (kk - 1) / 2))
        }
      } else {
        this.padding = opts.padding == null
          ? Array(k.length).fill(0)
          : (typeof opts.padding === 'number' ? Array(k.length).fill(opts.padding) : Array.from(opts.padding))
      }
      const bound = 1.0 / Math.sqrt(this.inChannels * k.reduce((a, b) => a * b, 1))
      this.weight = Tensor.uniform(
        this.outChannels, Math.floor(this.inChannels / this.groups), ...k,
        { low: -bound, high: bound }
      )
      this.bias = opts.bias === false
        ? null
        : Tensor.uniform(this.outChannels, { low: -bound, high: bound })
    }

    call(x) {
      return x.conv2d(this.weight, this.bias, {
        groups: this.groups,
        stride: this.stride,
        dilation: this.dilation,
        padding: this.padding
      })
    }
  }

  function Conv1d(inChannels, outChannels, kernelSize, opts = {}) {
    return new Conv2d(inChannels, outChannels, [kernelSize], opts)
  }

  class ConvTranspose2d extends Conv2d {
    constructor(inChannels, outChannels, kernelSize, opts = {}) {
      super(inChannels, outChannels, kernelSize, opts)
      const bound = 1 / Math.sqrt(this.inChannels * this.kernelSize.reduce((a, b) => a * b, 1))
      this.weight = Tensor.uniform(this.inChannels, Math.floor(this.outChannels / this.groups), ...this.kernelSize,
        {low: -bound, high: bound})
      this.outputPadding = opts.outputPadding == null ? 0 : opts.outputPadding
    }

    call(x) {
      return x.convTranspose2d(this.weight, this.bias, {groups: this.groups, stride: this.stride,
        dilation: this.dilation, padding: this.padding, outputPadding: this.outputPadding})
    }
  }

  function ConvTranspose1d(inChannels, outChannels, kernelSize, opts = {}) {
    return new ConvTranspose2d(inChannels, outChannels, [kernelSize], opts)
  }

  class InstanceNorm {
    constructor(numFeatures, opts = {}) {
      this.numFeatures = Number(numFeatures)
      this.eps = opts.eps == null ? 1e-5 : Number(opts.eps)
      this.weight = opts.affine === false ? null : Tensor.ones(this.numFeatures)
      this.bias = opts.affine === false ? null : Tensor.zeros(this.numFeatures)
    }

    call(x) {
      const core = runtime._core.ffi.poly_tensor_instancenorm_apply(x._ctx, x._tensor,
        this.weight === null ? null : this.weight._tensor, this.bias === null ? null : this.bias._tensor,
        this.numFeatures, this.eps)
      return x._makeResultFromCore(core)
    }
  }

  class LSTMCell {
    constructor(inputSize, hiddenSize, opts = {}) {
      const bound = 1 / Math.sqrt(hiddenSize)
      this.weightIh = Tensor.uniform(hiddenSize * 4, inputSize, {low: -bound, high: bound})
      this.weightHh = Tensor.uniform(hiddenSize * 4, hiddenSize, {low: -bound, high: bound})
      this.biasIh = opts.bias === false ? null : Tensor.zeros(hiddenSize * 4)
      this.biasHh = opts.bias === false ? null : Tensor.zeros(hiddenSize * 4)
    }

    call(x, hc = null) {
      const [h, c] = hc === null ? [null, null] : hc
      const inputs = [x, h, c, this.weightIh, this.weightHh, this.biasIh, this.biasHh]
      if (inputs.some(t => t !== null && t._ctx !== x._ctx)) throw new Error('LSTMCell inputs must share the same context')
      const pair = runtime._core.ffi.poly_tensor_lstm_cell(x._ctx, ...inputs.map(t => t === null ? null : t._tensor))
      if (!pair) throw new Error('poly_tensor_lstm_cell failed')
      return pair.map(core => x._makeResultFromCore(core))
    }
  }

  class GroupNorm {
    constructor(numGroups, numChannels, opts = {}) {
      this.numGroups = Number(numGroups)
      this.numChannels = Number(numChannels)
      this.eps = opts.eps == null ? 1e-5 : Number(opts.eps)
      if (this.numChannels % this.numGroups !== 0) {
        throw new Error('numChannels must be divisible by numGroups')
      }
      const affine = opts.affine !== false
      this.weight = affine ? Tensor.ones(this.numChannels) : null
      this.bias = affine ? Tensor.zeros(this.numChannels) : null
    }

    call(x) {
      const shape = x.shape
      if (shape.length < 2) throw new Error('GroupNorm expects input with at least 2 dimensions')
      if (shape[1] !== this.numChannels) {
        throw new Error(`GroupNorm expected C=${this.numChannels}, got C=${shape[1]}`)
      }
      return x._makeResultFromCore(runtime._core.ffi.poly_tensor_groupnorm_apply(x._ctx, x._tensor,
        this.weight === null ? null : this.weight._tensor, this.bias === null ? null : this.bias._tensor, this.numGroups, this.eps))
    }
  }

  class RMSNorm {
    constructor(dim, opts = {}) {
      this.eps = opts.eps == null ? 1e-6 : Number(opts.eps)
      this.weight = opts.elementwiseAffine === false ? null : Tensor.ones(dim)
    }

    call(x) {
      const core = runtime._core.ffi.poly_tensor_rmsnorm_apply(x._ctx, x._tensor,
        this.weight === null ? null : this.weight._tensor, this.eps)
      return x._makeResultFromCore(core)
    }
  }

  class Embedding {
    constructor(vocabSize, embedDim) {
      this.vocabSize = Number(vocabSize)
      this.weight = Tensor.glorotUniform(this.vocabSize, Number(embedDim))
    }

    call(idx) {
      if (!isIntegerDtype(idx.dtype)) throw new TypeError(`Expected integer dtype for index in embedding, got ${idx.dtype}`)
      // nn._embedding_fwd: ordered selector and explicit weight-dtype reduction.
      const mask = Tensor.arange(this.vocabSize).eq(idx.unsqueeze(-1))
      return mask.unsqueeze(-1).where(this.weight, 0).sum(-2, false, this.weight.dtype)
    }
  }

  class Dropout {
    constructor(p = 0.5) { this.p = p }
    call(x) { return x.dropout(this.p) }
  }

  class BatchNorm {
    constructor(numFeatures, opts = {}) {
      this.eps = opts.eps == null ? 1e-5 : Number(opts.eps)
      this.momentum = opts.momentum == null ? 0.1 : Number(opts.momentum)
      this.trackRunningStats = opts.trackRunningStats !== false
      this.weight = opts.affine === false ? null : Tensor.ones(numFeatures)
      this.bias = opts.affine === false ? null : Tensor.zeros(numFeatures)
      this.numBatchesTracked = Tensor.zeros({dtype: 'int64'}).is_param_(false)
      if (this.trackRunningStats) {
        this.runningMean = Tensor.zeros(numFeatures).is_param_(false)
        this.runningVar = Tensor.ones(numFeatures).is_param_(false)
      }
    }

    calcStats(x) {
      return runtime._core.ffi.poly_tensor_batchnorm_stats(x._ctx, x._tensor,
        this.trackRunningStats ? this.runningMean._tensor : null, this.trackRunningStats ? this.runningVar._tensor : null,
        !!Tensor.training).map(core => x._makeResultFromCore(core))
    }

    call(x) {
      return x._makeResultFromCore(runtime._core.ffi.poly_tensor_batchnorm_apply(x._ctx, x._tensor,
        this.weight === null ? null : this.weight._tensor, this.bias === null ? null : this.bias._tensor,
        this.trackRunningStats ? this.runningMean._tensor : null, this.trackRunningStats ? this.runningVar._tensor : null,
        this.numBatchesTracked._tensor, !!Tensor.training, this.eps, this.momentum))
    }
  }

  return { Linear, LayerNorm, LayerNorm2d, Conv1d, Conv2d, ConvTranspose1d, ConvTranspose2d,
    InstanceNorm, LSTMCell, GroupNorm, RMSNorm, Embedding, Dropout,
    BatchNorm, BatchNorm2d: BatchNorm, BatchNorm3d: BatchNorm }
}

module.exports = { createBoundModules }
