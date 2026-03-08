'use strict'

// Best-effort native addon build. Always exits 0.
// Set POLYGRAD_SKIP_NATIVE=1 to skip entirely.

if (process.env.POLYGRAD_SKIP_NATIVE === '1') {
  console.log('polygrad: skipping native build (POLYGRAD_SKIP_NATIVE=1)')
  process.exit(0)
}

const { execSync } = require('child_process')

try {
  // Sync C sources if building from source (not published tarball)
  try {
    execSync('node scripts/sync-csrc.js', { stdio: 'inherit', cwd: __dirname + '/..' })
  } catch (e) {
    // csrc/ may already exist in published package
  }

  execSync('node-gyp rebuild', {
    stdio: 'inherit',
    cwd: __dirname + '/..',
    timeout: 120000
  })
  console.log('polygrad: native addon built successfully')
} catch (e) {
  console.log('polygrad: native addon build failed (will use WASM fallback)')
}

process.exit(0)
