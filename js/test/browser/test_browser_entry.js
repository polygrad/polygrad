/**
 * Browser test entry point.
 *
 * Bundles test_tensor.js and test_model_runtime.js into a single
 * browser-loadable script. Built by esbuild into test/browser/tests.js.
 */
'use strict'

const { runTensorTests } = require('../test_tensor')
const { runModelRuntimeTests, runModelSmokeTests } = require('../test_model_runtime')
const { checkOnnxEncoder } = require('../check_onnx_encoder')

// Expose globally for index.html
window.__runTensorTests = runTensorTests
window.__runModelRuntimeTests = runModelRuntimeTests
window.__runModelSmokeTests = runModelSmokeTests
window.__checkOnnxEncoder = checkOnnxEncoder
window.__checkExtension = require('../check_extension')
window.__runJitLifetimeTests = pg => require('../test_jit').runJitTests(pg, 'lifetime:')
window.__runOptimLifetimeTests = pg => require('../test_optim').runOptimTests(pg, 'lifetime:')
