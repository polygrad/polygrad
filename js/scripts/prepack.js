'use strict'

const fs = require('fs')
const path = require('path')
const { spawnSync } = require('child_process')

const scriptsDir = __dirname
const pkgDir = path.resolve(scriptsDir, '..')
const repoDir = path.resolve(pkgDir, '..')

function run(cmd, args, extraEnv) {
  const result = spawnSync(cmd, args, {
    cwd: pkgDir,
    stdio: 'inherit',
    env: { ...process.env, ...extraEnv }
  })
  if (result.status !== 0) {
    process.exit(result.status || 1)
  }
}

run(process.execPath, [path.join(scriptsDir, 'sync-csrc.js')])

// npm tarballs publish js/wasm/polygrad.js and browser dist artifacts.
// Build them here so `npm pack` from a clean checkout produces a complete tarball.
const makeEnv = {
  EM_CACHE: process.env.EM_CACHE || path.join(repoDir, 'build', '.emcache')
}

if (!process.env.EMSDK_PYTHON && fs.existsSync('/usr/bin/python3')) {
  makeEnv.EMSDK_PYTHON = '/usr/bin/python3'
}

run('make', ['-C', repoDir, 'wasm-pkg'], makeEnv)
run('bash', [path.join(scriptsDir, 'build-browser.sh')])

const required = [
  path.join(pkgDir, 'csrc'),
  path.join(pkgDir, 'wasm', 'polygrad.js'),
  path.join(pkgDir, 'dist', 'polygrad.js'),
  path.join(pkgDir, 'dist', 'polygrad.mjs')
]

for (const target of required) {
  if (!fs.existsSync(target)) {
    console.error(`prepack: missing required publish artifact: ${target}`)
    process.exit(1)
  }
}
