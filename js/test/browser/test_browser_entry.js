/**
 * Browser test entry point.
 *
 * Bundles test_tensor.js and test_instance.js into a single
 * browser-loadable script. Built by esbuild into test/browser/tests.js.
 */
'use strict'

const { runTensorTests } = require('../test_tensor')
const { runInstanceTests, runInstanceSmokeTests } = require('../test_instance')

// Expose globally for index.html
window.__runTensorTests = runTensorTests
window.__runInstanceTests = runInstanceTests
window.__runInstanceSmokeTests = runInstanceSmokeTests
