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

function createNativeCore() {
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
    float64: binding.poly_dtype_id_by_name('float64')
  }

  const ctx = binding.poly_ctx_new()

  const ops = {}
  const opCount = binding.poly_op_count()
  for (let i = 0; i < opCount; i++) {
    const name = binding.poly_op_name(i)
    if (name) ops[name] = i
  }

  const EXPECTED_ABI = 4
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
      if (inst) binding.poly_instance_set_device(inst, 0)
      return inst
    },
    mlp(specJson) {
      const inst = binding.poly_mlp_from_json(specJson)
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
    loadHF(configBytes, weightFilesBytes, maxBatch, maxSeqLen) {
      const inst = binding.poly_hf_load(configBytes, weightFilesBytes,
        maxBatch || 1, maxSeqLen || 0)
      if (inst) binding.poly_instance_set_device(inst, 0)
      return inst
    },
    loadGGUF(ggufBytes, maxBatch, maxSeqLen) {
      const inst = binding.poly_gguf_load(ggufBytes, maxBatch || 1, maxSeqLen || 0)
      if (inst) binding.poly_instance_set_device(inst, 0)
      return inst
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
    dtypeIds,
    instance,
    ctx,
    ops,
    caps: { simd: false, f64: true, core: 'native', device: 'cpu' },
    destroy() {
      binding.poly_ctx_destroy(ctx)
    }
  }
}

module.exports = { createNativeCore }
