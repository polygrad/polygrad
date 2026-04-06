/**
 * Browser test entry point.
 *
 * Bundles test_shared.js and test_instance_shared.js into a single
 * browser-loadable script. Built by esbuild into test/browser/tests.js.
 */
'use strict'

const { runTests } = require('../test_shared')
const { runInstanceTests } = require('../test_instance_shared')

// Expose globally for index.html
window.__runTests = runTests
window.__runInstanceTests = runInstanceTests
