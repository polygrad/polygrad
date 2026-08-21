'use strict'

function loadNativeBinding() {
  try {
    return require('../../build/Release/polygrad_napi.node')
  } catch {
    try {
      return require('../../build/Debug/polygrad_napi.node')
    } catch {
      return null
    }
  }
}

function createNativeCore(device) {
  const binding = loadNativeBinding()
  if (!binding) {
    throw new Error('polygrad: core=\'native\' unavailable (N-API addon not built)')
  }

  const dtypeIds = {
    bool: binding.poly_dtype_id_by_name('bool'),
    int8: binding.poly_dtype_id_by_name('int8'),
    uint8: binding.poly_dtype_id_by_name('uint8'),
    int16: binding.poly_dtype_id_by_name('int16'),
    uint16: binding.poly_dtype_id_by_name('uint16'),
    int32: binding.poly_dtype_id_by_name('int32'),
    uint32: binding.poly_dtype_id_by_name('uint32'),
    int64: binding.poly_dtype_id_by_name('int64'),
    uint64: binding.poly_dtype_id_by_name('uint64'),
    float16: binding.poly_dtype_id_by_name('float16'),
    bfloat16: binding.poly_dtype_id_by_name('bfloat16'),
    float32: binding.poly_dtype_id_by_name('float32'),
    float64: binding.poly_dtype_id_by_name('float64'),
    weakint: binding.poly_dtype_id_by_name('weakint')
  }

  const ctx = binding.poly_ctx_new()
  const nativeDevice = resolveNativeDevice(binding, device)
  const deviceId = nativeDevice.id
  binding.poly_ctx_set_preferred_device(ctx, deviceId)

  function setInstanceDevice(inst) {
    if (!inst) return inst
    if (binding.poly_instance_set_device(inst, deviceId) !== 0) {
      binding.poly_instance_free(inst)
      throw new Error(`polygrad: set_device failed for native device '${nativeDevice.name}'`)
    }
    return inst
  }

  const ops = {}
  const opCount = binding.poly_op_count()
  for (let i = 0; i < opCount; i++) {
    const name = binding.poly_op_name(i)
    if (name) ops[name] = i
  }

  const EXPECTED_ABI = 55
  const abi = binding.poly_abi_version()
  if (abi !== EXPECTED_ABI) {
    throw new Error(
      `polygrad native ABI mismatch: expected version ${EXPECTED_ABI}, got ${abi}. ` +
      'Rebuild the native addon with: npm run build:native'
    )
  }

  const instance = {
    fromIR(irBytes, weightsBytes) {
      const inst = binding.poly_instance_from_ir(irBytes, weightsBytes ?? null)
      return setInstanceDevice(inst)
    },
    mlp(specJson) {
      const inst = binding.poly_mlp_from_json(specJson, deviceId)
      return setInstanceDevice(inst)
    },
    tabm(specJson) {
      const inst = binding.poly_tabm_instance(specJson, deviceId)
      return setInstanceDevice(inst)
    },
    nam(specJson) {
      const inst = binding.poly_nam_instance(specJson, deviceId)
      return setInstanceDevice(inst)
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
    paramDtypeId(inst, i) {
      return binding.poly_instance_param_dtype_id(inst, i)
    },
    paramTrainable(inst, i) {
      return binding.poly_instance_param_trainable(inst, i)
    },
    setParamTrainable(inst, i, trainable) {
      return binding.poly_instance_set_param_trainable(inst, i, trainable)
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
    bufTrainable(inst, i) {
      return binding.poly_instance_buf_trainable(inst, i)
    },
    setBufTrainable(inst, i, trainable) {
      return binding.poly_instance_set_buf_trainable(inst, i, trainable)
    },
    bufShape(inst, i) {
      return binding.poly_instance_buf_shape(inst, i)
    },
    bufData(inst, i) {
      return binding.poly_instance_buf_data(inst, i)
    },
    bufDtypeId(inst, i) {
      return binding.poly_instance_buf_dtype_id(inst, i)
    },
    exportWeights(inst, flags) {
      return binding.poly_instance_export_weights(inst, flags)
    },
    importWeights(inst, bytes) {
      return binding.poly_instance_import_weights(inst, bytes)
    },
    exportIR(inst) {
      return binding.poly_instance_export_ir(inst)
    },
    saveBundle(inst, flags) {
      return binding.poly_instance_save_bundle(inst, flags)
    },
    fromBundle(bytes) {
      const inst = binding.poly_instance_from_bundle(bytes)
      return setInstanceDevice(inst)
    },
    fromSinks(ctxPtr, names, sinks) {
      const inst = binding.poly_instance_from_sinks(ctxPtr, names, sinks)
      return setInstanceDevice(inst)
    },
    fromBindings(ctxPtr, bindings, entries) {
      const bindingNames = bindings.map(b => b.name)
      const bindingRoles = bindings.map(b => b.role)
      const bindingTensors = bindings.map(b => b.tensor)
      const bindingFlags = bindings.map(b => b.flags || 0)
      const entryNames = entries.map(e => e.name)
      const entryInputs = entries.flatMap(e => e.inputs || [])
      const entryInputCounts = entries.map(e => (e.inputs || []).length)
      const entryOutputs = entries.flatMap(e => e.outputs || [])
      const entryOutputCounts = entries.map(e => (e.outputs || []).length)
      const entryObjectives = entries.map(e => e.objective || null)
      const entryFlags = entries.map(e => e.flags || 0)
      const inst = binding.poly_instance_from_binding_arrays(
        ctxPtr,
        bindingNames, bindingRoles, bindingTensors, bindingFlags,
        entryNames, entryInputs, entryInputCounts, entryOutputs, entryOutputCounts,
        entryObjectives, entryFlags
      )
      return setInstanceDevice(inst)
    },
    defineModules(inst, modules) {
      return binding.poly_instance_define_module_arrays(
        inst,
        modules.map(m => m.name),
        modules.flatMap(m => m.inputs || []),
        modules.map(m => (m.inputs || []).length),
        modules.map(m => m.output)
      )
    },
    setDeviceMap(inst, entries) {
      return binding.poly_instance_set_device_map_arrays(
        inst, entries.map(e => e.module), entries.map(e => e.device)
      )
    },
    loadHF(configBytes, weightFilesBytes, maxBatch, maxSeqLen) {
      const inst = binding.poly_hf_load(configBytes, weightFilesBytes,
        maxBatch || 1, maxSeqLen || 0, deviceId)
      return setInstanceDevice(inst)
    },
    loadGGUF(ggufBytes, maxBatch, maxSeqLen) {
      const inst = binding.poly_gguf_load(
        ggufBytes, maxBatch || 1, maxSeqLen || 0, deviceId)
      return setInstanceDevice(inst)
    },
    importLastError() {
      const code = binding.poly_import_last_error_code()
      if (code === 0) return null
      return { code, message: binding.poly_import_last_error_message() || 'unknown' }
    },
    tokenizerFromGGUF(ggufBytes) {
      return binding.poly_tokenizer_from_gguf(ggufBytes)
    },
    tokenizerFromJSON(jsonBytes) {
      return binding.poly_tokenizer_from_json(jsonBytes, jsonBytes.length)
    },
    tokenize(tokPtr, text) { return binding.poly_tokenize(tokPtr, text) },
    detokenize(tokPtr, ids) {
      return binding.poly_detokenize(tokPtr, Array.from(ids))
    },
    tokenizerFree(tokPtr) { binding.poly_tokenizer_free(tokPtr) },
    tokenizerVocabSize(tokPtr) { return binding.poly_tokenizer_vocab_size(tokPtr) },
    tokenizerBosId(tokPtr) { return binding.poly_tokenizer_bos_id(tokPtr) },
    tokenizerEosId(tokPtr) { return binding.poly_tokenizer_eos_id(tokPtr) },
    setOptimizer(inst, kind, lr, beta1, beta2, eps, weightDecay, momentum, nesterov, classic) {
      return binding.poly_instance_set_optimizer(
        inst, kind, lr, beta1, beta2, eps, weightDecay, momentum || 0, !!nesterov, !!classic
      )
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
    dtypeIds,
    instance,
    ctx,
    ops,
    deviceIds: makeDeviceIds(binding),
    caps: { simd: false, f64: true, core: 'native', device: nativeDevice.name },
    canRunOp(device, op, dtypeId, shape) {
      return binding.poly_can_run_op(ctx, device, op, dtypeId, Array.from(shape || []))
    },
    destroy() {
      binding.poly_ctx_destroy(ctx)
    }
  }
}

const NATIVE_DEVICE_NAMES = new Set(['cpu', 'interp', 'x86', 'cuda', 'hip'])

function normalizeDeviceName(device) {
  let name = null
  if (device && device !== 'auto') name = String(device)
  else if (process.env.POLY_DEVICE && process.env.POLY_DEVICE !== 'auto') name = String(process.env.POLY_DEVICE)
  else name = 'cpu'
  name = name.toLowerCase()
  if (name.startsWith('cpu:')) name = name.slice(4)
  return name
}

function resolveNativeDevice(binding, device) {
  const name = normalizeDeviceName(device)
  if (!NATIVE_DEVICE_NAMES.has(name)) {
    throw new Error(`polygrad: unsupported native device '${name}'`)
  }
  const id = binding.poly_device_by_name(name)
  if (!id) throw new Error(`polygrad: unknown native device '${name}'`)
  return { id, name: binding.poly_device_name(id) || name }
}

function makeDeviceIds(binding) {
  const names = ['auto', 'cpu', 'interp', 'x86', 'cuda', 'hip']
  const ids = {}
  for (const name of names) ids[name] = binding.poly_device_by_name(name)
  return ids
}

module.exports = { createNativeCore }
