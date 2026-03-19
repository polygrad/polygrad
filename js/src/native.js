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

  const instance = {
    fromIR(irBytes, weightsBytes) {
      const inst = binding.poly_instance_from_ir(irBytes, weightsBytes ?? null)
      if (inst) binding.poly_instance_set_device(inst, 0)
      return inst
    },
    mlp(specJson) {
      const inst = binding.poly_mlp_instance(specJson)
      if (inst) binding.poly_instance_set_device(inst, 0)
      return inst
    },
    tabm(specJson) {
      const inst = binding.poly_tabm_instance(specJson)
      if (inst) binding.poly_instance_set_device(inst, 0)
      return inst
    },
    nam(specJson) {
      const inst = binding.poly_nam_instance(specJson)
      if (inst) binding.poly_instance_set_device(inst, 0)
      return inst
    },
    free(inst) {
      binding.poly_instance_free(inst)
    },
    paramCount(inst) {
      return binding.poly_instance_param_count(inst)
    },
    paramName(inst, i) {
      return binding.poly_instance_param_name(inst, i)
    },
    paramShape(inst, i) {
      return binding.poly_instance_param_shape(inst, i)
    },
    paramData(inst, i) {
      return binding.poly_instance_param_data(inst, i)
    },
    bufCount(inst) {
      return binding.poly_instance_buf_count(inst)
    },
    bufName(inst, i) {
      return binding.poly_instance_buf_name(inst, i)
    },
    bufRole(inst, i) {
      return binding.poly_instance_buf_role(inst, i)
    },
    bufShape(inst, i) {
      return binding.poly_instance_buf_shape(inst, i)
    },
    bufData(inst, i) {
      return binding.poly_instance_buf_data(inst, i)
    },
    exportWeights(inst) {
      return binding.poly_instance_export_weights(inst)
    },
    importWeights(inst, bytes) {
      return binding.poly_instance_import_weights(inst, bytes)
    },
    exportIR(inst) {
      return binding.poly_instance_export_ir(inst)
    },
    saveBundle(inst) {
      return binding.poly_instance_save_bundle(inst)
    },
    fromBundle(bytes) {
      const inst = binding.poly_instance_from_bundle(bytes)
      if (inst) binding.poly_instance_set_device(inst, 0)
      return inst
    },
    setOptimizer(inst, kind, lr, beta1, beta2, eps, weightDecay) {
      return binding.poly_instance_set_optimizer(inst, kind, lr, beta1, beta2, eps, weightDecay)
    },
    forward(inst, names, arrays) {
      return binding.poly_instance_forward(inst, names, arrays)
    },
    trainStep(inst, names, arrays) {
      return binding.poly_instance_train_step(inst, names, arrays)
    }
  }

  return {
    ffi: binding,
    instance,
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
