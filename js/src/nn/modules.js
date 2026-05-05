'use strict'

function createBoundModules(runtime) {
  const Tensor = runtime.Tensor

  class Linear {
    constructor(inFeatures, outFeatures, opts = {}) {
      this.inFeatures = Number(inFeatures)
      this.outFeatures = Number(outFeatures)
      const bound = Math.sqrt(6.0 / Math.max(1, this.inFeatures))
      const w = new Float32Array(this.outFeatures * this.inFeatures)
      for (let i = 0; i < w.length; i++) w[i] = (Math.random() * 2.0 - 1.0) * bound
      this.weight = new Tensor(w, { requiresGrad: true }).reshape(this.outFeatures, this.inFeatures)
      this.bias = opts.bias === false
        ? null
        : new Tensor(new Float32Array(this.outFeatures), { requiresGrad: true })
    }

    call(x) {
      return x.linear(this.weight, this.bias)
    }
  }

  return { Linear }
}

module.exports = { createBoundModules }
