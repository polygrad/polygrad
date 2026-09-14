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
    if (typeof pg.Model !== 'function' || 'Instance' in pg) throw new Error('incorrect Model export')
    const rt = pg.create({core:'wasm'})
    if (rt && typeof rt.then === 'function') throw new Error('create returned Promise')
    if (pg.Model.fromDefinition || typeof pg.models.Graph !== 'function') throw new Error('incorrect factory namespace')
    const model = rt.models.Sequential({input:{name:'x',shape:[1],dtype:'float32'},layers:[{name:'copy',type:'identity'}],output:'prediction'})
    if (model.forward({x:new Float32Array([7])}).prediction[0] !== 7) throw new Error('packaged Sequential failed')
    model.dispose()
    const y = new pg.Tensor([1,2,3]).mul(2)
    console.log(Array.from(y.toArray()).join(','))
    pg.disposeDefault(); rt.dispose()
  `)
  assert.strictEqual(out, '2,4,6')
}

function testNodeEsmRoot() {
  const out = nodeModuleEval(`
    import { Tensor, Model, create, disposeDefault } from 'polygrad'
    for (const name of ['fromCallable', 'fromCallableAsync', 'fromTensors', 'load']) {
      if (typeof Model[name] !== 'function') throw new Error('missing Model.' + name + ' export')
    }
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
    import { Tensor, Model, create, disposeDefault } from 'polygrad'
    if (typeof Model.fromTensors !== 'function') throw new Error('missing browser Model export')
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
    const model = await rt.models.GraphAsync({inputs:{x:{shape:[1],dtype:'float32'}},nodes:[],outputs:{prediction:'x'}})
    if (model.forward({x:new Float32Array([7])}).prediction[0] !== 7) throw new Error('packaged Graph failed')
    model.dispose()
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

function testEnvironmentPrecedence() {
  for (const core of ['native', 'wasm']) {
    for (const [env, device, expected] of [
      [{DEV: 'iNtErP'}, undefined, 'interp'],
      [{DEV: 'INTERP', POLY_DEV: 'CPU'}, undefined, 'cpu'],
      [{DEV: 'CPU', POLY_DEV: 'INTERP'}, undefined, 'interp'],
      [{POLY_DEV: 'CPU;CUDA'}, 'cpu', 'cpu'],
      [{POLY_DEV: 'INTERP'}, 'cpu', 'cpu'],
    ]) {
      const out = nodeEval(`
        const rt = require('polygrad').create(${JSON.stringify({core, device})});
        console.log(rt.device); rt.dispose();
      `, [], {DEV: '', POLY_DEV: '', DEBUG: '0', POLY_DEBUG: '0', ...env})
      // Public selection keeps the CPU alias; execution remains C/Wasm.
      assert.strictEqual(out, expected)
    }
    for (const target of ['CUDA:0', 'CUDA:1', 'CPU;CUDA', 'CPU:CUDA', 'CPU:X86:arch', 'NV+CUDA']) {
      const out = nodeEval(`
        const assert = require('assert');
        assert.throws(() => require('polygrad').create({core:${JSON.stringify(core)}}), /Unsupported.*target/);
        console.log('rejected');
      `, [], {DEV: '', POLY_DEV: target, DEBUG: '0', POLY_DEBUG: '0'})
      assert.strictEqual(out, 'rejected')
    }
  }
}

function testEnvironmentCompilerControls() {
  for (const core of ['native', 'wasm']) {
    const rejected = nodeEval(`
      const assert = require('assert');
      assert.throws(() => require('polygrad').create({core:${JSON.stringify(core)}}), /BEAM must be a decimal int32/);
      console.log('rejected');
    `, [], {BEAM:'0x10', POLY_DEBUG:'0'})
    assert.strictEqual(rejected, 'rejected')
    for (const async of [false, true]) {
      const create = async ? 'await pg.createAsync' : 'pg.create'
      const out = nodeEval(`
      (async () => {
      const pg = require('polygrad');
      const rt = ${create}({core:${JSON.stringify(core)}, device:'interp'});
      console.log(rt.beam, rt.noopt);
      rt.beam = 0; rt.noopt = 0;
      const other = ${create}({core:${JSON.stringify(core)}, device:'interp'});
      console.log(rt.beam, other.beam, rt.noopt, other.noopt);
      await other.dispose(); await rt.dispose();
      })().catch(e => {console.error(e); process.exitCode = 1});
    `, [], {BEAM:'2', NOOPT:'1', POLY_DEBUG:'0'})
      // Async Wasm creates independent modules; native and sync Wasm share policy.
      assert.strictEqual(out, core === 'wasm' && async ? '2 1\n0 2 0 1' : '2 1\n0 0 0 0')
    }
  }
}

function testModelFileIO() {
  for (const core of ['native', 'wasm']) {
    const out = nodeEval(`
      (async () => {
        const fs = require('node:fs'), path = require('node:path'), os = require('node:os')
        const pg = require('polygrad')
        const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'polygrad-model-files-'))
        const file = path.join(dir, 'linear.pgb')
        const rt = pg.create({core: ${JSON.stringify(core)}, device:'interp'})
        let model, restored, asyncRuntime
        try {
          model = new rt.Model(({x}) => ({prediction:x.mul(2)}), {inputs:{x:rt.Tensor.empty([1])}})
          const bytes = model.save(file, {includeOptimizer:false})
          if (!(bytes instanceof Uint8Array) || !fs.readFileSync(file).equals(Buffer.from(bytes))) throw new Error('save did not write bundle')
          restored = rt.Model.load(file)
          if (restored.forward({x:[3]}).prediction[0] !== 6) throw new Error('path load failed')
          restored.dispose(); restored = null
          await model.saveAsync(file, {includeOptimizer:false})
          asyncRuntime = await pg.createAsync({core:${JSON.stringify(core)}, device:'interp'})
          restored = asyncRuntime.Model.load(file)
          if (restored.forward({x:[4]}).prediction[0] !== 8) throw new Error('async runtime lacks file adapter')
          let rejected = false
          try { model.save(path.join(dir,'absent','model.pgb')) } catch (e) { rejected = e.code === 'ENOENT' }
          if (!rejected || model.forward({x:[5]}).prediction[0] !== 10) throw new Error('file failure destroyed Model')
          console.log('model files pass')
        } finally {
          if (restored) await restored.dispose()
          if (model) await model.dispose()
          if (asyncRuntime) await asyncRuntime.dispose()
          await rt.dispose()
          fs.rmSync(dir, {recursive:true, force:true})
        }
      })().catch(e => { console.error(e); process.exitCode=1 })
    `)
    assert.strictEqual(out, 'model files pass')
  }
}

function main() {
  const tests = [
    ['Model filesystem roundtrip', testModelFileIO],
    ['environment precedence', testEnvironmentPrecedence],
    ['environment compiler controls', testEnvironmentCompilerControls],
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
