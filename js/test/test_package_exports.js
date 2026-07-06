'use strict'

const assert = require('assert')
const fs = require('fs')
const path = require('path')
const { spawnSync } = require('child_process')

const pkgDir = path.resolve(__dirname, '..')
const node = process.execPath

function run(cmd, args, opts = {}) {
  const result = spawnSync(cmd, args, {
    cwd: opts.cwd || pkgDir,
    env: { ...process.env, ...(opts.env || {}) },
    encoding: 'utf8'
  })
  if (result.status !== 0) {
    throw new Error([
      `${cmd} ${args.join(' ')} failed with status ${result.status}`,
      result.stdout,
      result.stderr
    ].filter(Boolean).join('\n'))
  }
  return result.stdout.trim()
}

function assertIncludes(haystack, needle, label) {
  assert(haystack.includes(needle), `${label} missing ${needle}`)
}

function assertExcludes(haystack, needle, label) {
  assert(!haystack.includes(needle), `${label} unexpectedly contains ${needle}`)
}

function nodeEval(code, extraArgs = [], env = {}) {
  return run(node, [...extraArgs, '-e', code], { env: { POLY_CORE: 'wasm', ...env } })
}

function nodeModuleEval(code, extraArgs = [], env = {}) {
  return run(node, [...extraArgs, '--input-type=module', '-e', code], { env: { POLY_CORE: 'wasm', ...env } })
}

function testNodeCjsRoot() {
  const out = nodeEval(`
    const pg = require('polygrad')
    const rt = pg.create({core:'wasm'})
    if (rt && typeof rt.then === 'function') throw new Error('create returned Promise')
    const y = new pg.Tensor([1,2,3]).mul(2)
    console.log(Array.from(y.toArray()).join(','))
    pg.disposeDefault(); rt.dispose()
  `)
  assert.strictEqual(out, '2,4,6')
}

function testNodeEsmRoot() {
  const out = nodeModuleEval(`
    import { Tensor, create, disposeDefault } from 'polygrad'
    const rt = create({core:'wasm'})
    if (rt && typeof rt.then === 'function') throw new Error('create returned Promise')
    const y = new Tensor([1,2,3]).mul(3)
    console.log(Array.from(y.toArray()).join(','))
    disposeDefault(); rt.dispose()
  `)
  assert.strictEqual(out, '3,6,9')
}

function testBrowserConditionSync() {
  const out = nodeModuleEval(`
    import { Tensor, create, disposeDefault } from 'polygrad'
    const rt = create({core:'wasm'})
    if (rt && typeof rt.then === 'function') throw new Error('create returned Promise')
    const y = new Tensor([1,2,3]).mul(4)
    console.log(Array.from(y.toArray()).join(','))
    disposeDefault(); rt.dispose()
  `, ['--conditions=browser'])
  assert.strictEqual(out, '4,8,12')
}

function testBrowserSubpathSync() {
  const out = nodeModuleEval(`
    import { Tensor, create, disposeDefault } from 'polygrad/browser'
    const rt = create({core:'wasm'})
    if (rt && typeof rt.then === 'function') throw new Error('create returned Promise')
    const y = new Tensor([1,2,3]).mul(5)
    console.log(Array.from(y.toArray()).join(','))
    disposeDefault(); rt.dispose()
  `, ['--conditions=browser'])
  assert.strictEqual(out, '5,10,15')
}

function testBrowserConditionAsync() {
  const out = nodeModuleEval(`
    import { create, createAsync } from 'polygrad/async'
    let threw = false
    try { create({core:'wasm'}) } catch (e) { threw = e && e.name === 'PolyWasmSyncUnsupported' }
    if (!threw) throw new Error('async browser create() did not throw PolyWasmSyncUnsupported')
    const rt = await createAsync({core:'wasm'})
    const y = new rt.Tensor([1,2,3]).mul(6)
    console.log(Array.from(y.toArray()).join(','))
    rt.dispose()
  `, ['--conditions=browser'])
  assert.strictEqual(out, '6,12,18')
}

function testBrowserAsyncSubpath() {
  const out = nodeModuleEval(`
    import { createAsync } from 'polygrad/browser/async'
    const rt = await createAsync({core:'wasm'})
    const y = new rt.Tensor([1,2,3]).mul(7)
    console.log(Array.from(y.toArray()).join(','))
    rt.dispose()
  `, ['--conditions=browser'])
  assert.strictEqual(out, '7,14,21')
}

function testEsbuildBundleSelection() {
  const esbuild = path.join(pkgDir, 'node_modules', '.bin', process.platform === 'win32' ? 'esbuild.cmd' : 'esbuild')
  if (!fs.existsSync(esbuild)) throw new Error('esbuild binary missing; run npm install in js/')
  const tmpRoot = path.join(pkgDir, 'temp')
  fs.mkdirSync(tmpRoot, { recursive: true })
  const tmp = fs.mkdtempSync(path.join(tmpRoot, 'package-exports-'))
  try {
    const syncEntry = path.join(tmp, 'sync.mjs')
    const asyncEntry = path.join(tmp, 'async.mjs')
    const syncBundle = path.join(tmp, 'sync.bundle.mjs')
    const asyncBundle = path.join(tmp, 'async.bundle.mjs')
    fs.writeFileSync(syncEntry, `import { Tensor } from 'polygrad'; console.log(new Tensor([1,2,3]).mul(2).toArray()[0]);\n`)
    fs.writeFileSync(asyncEntry, `import { createAsync } from 'polygrad/async'; const pg = await createAsync({core:'wasm'}); console.log(new pg.Tensor([1,2,3]).mul(2).toArray()[0]); pg.dispose();\n`)
    run(esbuild, [syncEntry, '--bundle', '--platform=browser', '--format=esm', `--outfile=${syncBundle}`])
    run(esbuild, [asyncEntry, '--bundle', '--platform=browser', '--format=esm', `--outfile=${asyncBundle}`])
    const syncText = fs.readFileSync(syncBundle, 'utf8')
    const asyncText = fs.readFileSync(asyncBundle, 'utf8')
    assertIncludes(syncText, 'core.sync.js', 'sync browser bundle')
    assertIncludes(syncText, 'sync-only', 'sync browser bundle')
    assertExcludes(syncText, 'core.async.js', 'sync browser bundle')
    assertIncludes(asyncText, 'core.async.js', 'async browser bundle')
    assertIncludes(asyncText, 'async-only', 'async browser bundle')
    assertExcludes(asyncText, 'core.sync.js', 'async browser bundle')
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true })
  }
}

function main() {
  const tests = [
    ['node cjs root', testNodeCjsRoot],
    ['node esm root', testNodeEsmRoot],
    ['browser condition sync root', testBrowserConditionSync],
    ['browser condition sync subpath', testBrowserSubpathSync],
    ['browser condition async root', testBrowserConditionAsync],
    ['browser async subpath', testBrowserAsyncSubpath],
    ['esbuild bundle selection', testEsbuildBundleSelection]
  ]
  let failed = 0
  console.log('== Package exports ==')
  for (const [name, fn] of tests) {
    try {
      fn()
      console.log(`  [PASS] ${name}`)
    } catch (e) {
      failed++
      console.log(`  [FAIL] ${name}: ${e.message}`)
    }
  }
  if (failed) process.exit(1)
}

main()
