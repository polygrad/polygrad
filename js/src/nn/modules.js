'use strict'

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
      this.weight.requiresGrad = true
      this.bias = opts.bias === false
        ? null
        : Tensor.uniform(this.outFeatures, { low: -bound, high: bound })
      if (this.bias) this.bias.requiresGrad = true
    }

    call(x) {
      // Pinned nn/__init__.py:156-174 stores (out,in); Tensor.linear consumes
      // the transposed (in,out) matrix.
      return x.linear(this.weight.transpose(), this.bias)
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
      if (this.weight) this.weight.requiresGrad = true
      if (this.bias) this.bias.requiresGrad = true
    }

    call(x) {
      const tail = x.shape.slice(-this.normalizedShape.length)
      if (tail.length !== this.normalizedShape.length ||
          tail.some((dim, i) => dim !== this.normalizedShape[i])) {
        throw new Error(`last dimensions of ${JSON.stringify(x.shape)} must match ${JSON.stringify(this.normalizedShape)}`)
      }
      let result = x.layernorm(this.axis, this.eps)
      if (this.weight !== null && this.bias !== null) result = result.mul(this.weight).add(this.bias)
      return result
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
        ? [1, 1]
        : (typeof opts.stride === 'number' ? [opts.stride, opts.stride] : Array.from(opts.stride))
      this.dilation = opts.dilation == null
        ? [1, 1]
        : (typeof opts.dilation === 'number' ? [opts.dilation, opts.dilation] : Array.from(opts.dilation))
      this.groups = opts.groups == null ? 1 : Number(opts.groups)
      if (typeof opts.padding === 'string') {
        if (opts.padding.toLowerCase() !== 'same') {
          throw new Error(`Invalid padding string ${opts.padding}, only 'same' is supported`)
        }
        if (!(this.stride.length === 2 && this.stride[0] === 1 && this.stride[1] === 1)) {
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
          ? [0, 0]
          : (typeof opts.padding === 'number' ? [opts.padding, opts.padding] : Array.from(opts.padding))
      }
      const bound = 1.0 / Math.sqrt(Math.max(1, this.inChannels * k[0] * k[1]))
      this.weight = Tensor.uniform(
        this.outChannels, Math.floor(this.inChannels / this.groups), k[0], k[1],
        { low: -bound, high: bound }
      )
      this.weight.requiresGrad = true
      this.bias = opts.bias === false
        ? null
        : Tensor.uniform(this.outChannels, { low: -bound, high: bound })
      if (this.bias) this.bias.requiresGrad = true
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
      if (this.weight) this.weight.requiresGrad = true
      if (this.bias) this.bias.requiresGrad = true
    }

    call(x) {
      // Literal pinned tinygrad/nn/__init__.py:200-207 composition.
      const shape = x.shape
      if (shape.length < 2) throw new Error('GroupNorm expects input with at least 2 dimensions')
      if (shape[1] !== this.numChannels) {
        throw new Error(`GroupNorm expected C=${this.numChannels}, got C=${shape[1]}`)
      }
      let result = x.reshape(shape[0], this.numGroups, -1).layernorm(-1, this.eps).reshape(...shape)
      if (this.weight === null || this.bias === null) return result
      const affineShape = [1, -1, ...Array(Math.max(0, shape.length - 2)).fill(1)]
      return result.mul(this.weight.reshape(...affineShape)).add(this.bias.reshape(...affineShape))
    }
  }

  return { Linear, LayerNorm, LayerNorm2d, Conv2d, GroupNorm }
}

module.exports = { createBoundModules }
