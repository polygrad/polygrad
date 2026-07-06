import cjs from './index.js'

export const create = cjs.create
export const createAsync = cjs.createAsync
export const getDefaultRuntime = cjs.getDefaultRuntime
export const disposeDefault = cjs.disposeDefault
export const PolyAsyncRequired = cjs.PolyAsyncRequired
export const PolyWasmSyncUnsupported = cjs.PolyWasmSyncUnsupported

export const Tensor = cjs.Tensor
export const uop = cjs.uop
export const jit = cjs.jit
export const jitAsync = cjs.jitAsync
export const compile = cjs.compile
export const compileAsync = cjs.compileAsync
export const Instance = cjs.Instance
export const models = cjs.models
export const Tokenizer = cjs.Tokenizer
export const nn = cjs.nn

export const ROLE_PARAM = cjs.ROLE_PARAM
export const ROLE_INPUT = cjs.ROLE_INPUT
export const ROLE_TARGET = cjs.ROLE_TARGET
export const ROLE_OUTPUT = cjs.ROLE_OUTPUT
export const ROLE_AUX = cjs.ROLE_AUX
export const OPTIM_NONE = cjs.OPTIM_NONE
export const OPTIM_SGD = cjs.OPTIM_SGD
export const OPTIM_ADAM = cjs.OPTIM_ADAM
export const OPTIM_ADAMW = cjs.OPTIM_ADAMW

export const stats = cjs.stats
export const canRun = cjs.canRun

export default cjs
