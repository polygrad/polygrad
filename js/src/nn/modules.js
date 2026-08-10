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
      return x.linear(this.weight, this.bias)
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

  return { Linear, Conv2d }
}

module.exports = { createBoundModules }
