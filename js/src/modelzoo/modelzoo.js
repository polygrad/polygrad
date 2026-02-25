'use strict'

const { Tensor } = require('../tensor')
const ffi = require('../ffi')

/**
 * JS Adapter for Polygrad's C Parameterized Models
 * Exposes a Tensor-compatible API over the C model zoo registry.
 */
class PolyCModel {
    /**
     * @param {string} kind - The registered model kind in C (e.g. 'gpt2')
     * @param {object} config - Architecture configuration dictionary
     */
    constructor(ctx, kind, config) {
        this._ctx = ctx
        this.kind = kind
        this.config = config

        // 1. Convert JS config into C PolyModelConfig*
        // const c_config = ffi.poly_model_config_new()
        // ... map JS fields securely ...

        // 2. Instantiate C model registry entry
        // this._c_model = ffi.poly_model_create(this._ctx, kind, c_config)
    }

    /**
     * Performs a symbolic/lazy forward pass.
     * Returns generic Tensor classes making it compatible with JS train loops.
     */
    forward(...inputs) {
        // 1. Unwrap inputs to extract underlying PolyUOp*
        const cInputs = inputs.map(t => t._uop)

        // 2. Execute FFI call
        // const outUOps = ffi.poly_model_forward(this._c_model, cInputs, cInputs.length)

        // 3. Wrap back into JS Tensors for downstream composition
        // return outUOps.map(uop => new Tensor(null, { _ctx: this._ctx, _uop: uop }))
    }

    /**
     * Retrieves a named parameter as a JS Tensor.
     */
    getParameter(name) {
        // const paramUOp = ffi.poly_model_parameter(this._c_model, name)
        // return new Tensor(null, { _ctx: this._ctx, _uop: paramUOp, requiresGrad: true })
    }
}

module.exports = { PolyCModel }
