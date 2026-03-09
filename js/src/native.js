'use strict'

function loadNativeBinding() {
  try {
    return require('../build/Release/polygrad_napi.node')
  } catch {
    try {
      return require('../build/Debug/polygrad_napi.node')
    } catch {
      return null
    }
  }
}

function createNativeBackend() {
  const binding = loadNativeBinding()
  if (!binding) {
    throw new Error('polygrad: target=\'native\' unavailable (N-API addon not built)')
  }

  const ctx = binding.poly_ctx_new()

  const ops = {}
  const opCount = binding.poly_op_count()
  for (let i = 0; i < opCount; i++) {
    const name = binding.poly_op_name(i)
    if (name) ops[name] = i
  }

  const EXPECTED_ABI = 1
  const abi = binding.poly_abi_version()
  if (abi !== EXPECTED_ABI) {
    throw new Error(
      `polygrad native ABI mismatch: expected version ${EXPECTED_ABI}, got ${abi}. ` +
      'Rebuild the native addon with: npm run build:native'
    )
  }

  async function realize(ctx, sink, numel, leafMap, isF64) {
    binding.poly_realize_begin(ctx)
    for (const [buf, data] of leafMap) {
      binding.poly_realize_bind(ctx, buf, data)
    }
    const rc = binding.poly_realize_exec(ctx, sink)
    if (rc !== 0) {
      throw new Error('poly_realize_exec failed')
    }
    let outData = null
    for (const [, data] of leafMap) {
      outData = data
    }
    const AT = isF64 ? Float64Array : Float32Array
    return new AT(outData)
  }

  return {
    ffi: binding,
    ctx,
    ops,
    realize,
    caps: { simd: false, f64: true, target: 'native', device: 'cpu' },
    destroy() {
      binding.poly_ctx_destroy(ctx)
      binding.poly_cpu_cache_flush()
      binding.poly_sched_cache_flush()
    }
  }
}

module.exports = { createNativeBackend }
