'use strict'

// Count real C-core transitions, not FFI wrappers (some wrappers make zero or
// several transitions). Generated kernels and callback adapters are separate
// modules; callback exports must remain Wasm functions for Table.set.
function observeBoundaries() {
  const Module = WebAssembly.Module, Instance = WebAssembly.Instance
  const instantiate = WebAssembly.instantiate
  const state = { active: false, cores: 0, exports: 0, imports: 0, kernels: 0 }
  const isCore = m => Module.exports(m).filter(x => x.kind === 'function').length > 100
  function importsFor(input) {
    return Object.fromEntries(Object.entries(input || {}).map(([ns, entries]) => [ns,
      Object.fromEntries(Object.entries(entries).map(([key, fn]) => [key, typeof fn !== 'function' ? fn :
        function (...args) { if (state.active) state.imports++; return fn(...args) }]))]))
  }
  function wrapped(instance, core) {
    const exports = Object.fromEntries(Object.entries(instance.exports).map(([key, fn]) => {
      if (typeof fn !== 'function' || (!core && key !== 'kernel')) return [key, fn]
      const wrapper = function (...args) {
        if (state.active) state[core ? 'exports' : 'kernels']++
        return fn(...args)
      }
      Object.defineProperty(wrapper, 'length', { value: fn.length })
      return [key, wrapper]
    }))
    return new Proxy(instance, { get: (target, key) => key === 'exports' ? exports : Reflect.get(target, key) })
  }
  WebAssembly.Instance = new Proxy(Instance, { construct(target, args) {
    const core = isCore(args[0])
    if (core) state.cores++
    const instance = Reflect.construct(target, core ? [args[0], importsFor(args[1])] : args)
    return core || typeof instance.exports.kernel === 'function' ? wrapped(instance, core) : instance
  } })
  WebAssembly.instantiate = async function (input, imports) {
    const isModule = input instanceof Module, module = isModule ? input : new Module(input)
    if (!isCore(module)) return instantiate.call(WebAssembly, input, imports)
    state.cores++
    const instance = wrapped(await instantiate.call(WebAssembly, module, importsFor(imports)), true)
    return isModule ? instance : { module, instance }
  }
  state.restore = () => { WebAssembly.Instance = Instance; WebAssembly.instantiate = instantiate }
  return state
}

async function checkReplayBoundaries(polygrad, state, policy) {
  // Check the default and explicitly disabled provider policy separately.
  if (!policy) return [...await checkReplayBoundaries(polygrad, state, 'disabled'),
    ...await checkReplayBoundaries(polygrad, state, 'default')]
  const assert = (ok, message) => { if (!ok) throw new Error(message) }
  const disabled = policy === 'disabled'
  const pg = await polygrad.create({ core: 'wasm', device: 'cpu',
    ...(disabled ? { kernels: false } : {}), env: { BEAM: '0', NOOPT: '0' } })
  // Fixed token shapes/stage boundaries from the one-thread performance gate.
  // Ordinary row-major weights, default compiler, no private provider or workers.
  // Expected tuple: JS->core exports, core->JS imports, generated-kernel calls.
  const cases = [
    { name: 'mlp_token', d: 768, h: 3072, expected: [94, 14, 2] },
    { name: 'qwen_ffn_token', d: 1024, h: 2816, expected: [131, 28, 4] }
  ]
  function data(n, seed) {
    return Float32Array.from({ length: n }, () => {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0
      return (((seed >>> 8) % 1009) - 504) * 0.02 / 504
    })
  }
  function dense(x, w, rows, bias) {
    return Float64Array.from({ length: rows }, (_, j) => {
      let sum = bias ? bias[j] : 0
      for (let k = 0; k < x.length; k++) sum += x[k] * w[j * x.length + k]
      return sum
    })
  }
  const rows = []
  try {
    assert(state.cores === 1, `observer missed or duplicated shared core instantiation: ${state.cores}`)
    for (const c of cases) {
      const mlp = c.name === 'mlp_token', arrays = [data(c.d, 1), data(c.h * c.d, 2)]
      const shapes = [[1, c.d], [c.h, c.d]], tensors = [], stages = []
      if (mlp) {
        arrays.push(data(c.h, 3), data(c.d * c.h, 4), data(c.d, 5))
        shapes.push([1, c.h], [c.d, c.h], [1, c.d])
        stages.push(pg.jit((x, w, b) => x.matmul(w.permute(1, 0)).add(b).relu()),
          pg.jit((x, w, b) => x.matmul(w.permute(1, 0)).add(b)))
      } else {
        arrays.push(data(c.h * c.d, 3), data(c.d * c.h, 4))
        shapes.push([c.h, c.d], [c.d, c.h])
        stages.push(pg.jit((x, w) => x.matmul(w.permute(1, 0)).silu()),
          pg.jit((x, w) => x.matmul(w.permute(1, 0))), pg.jit((x, y) => x.mul(y)),
          pg.jit((x, w) => x.matmul(w.permute(1, 0))))
      }
      async function call() {
        if (mlp) {
          const h = await stages[0](tensors[0], tensors[1], tensors[2])
          await h.realize()
          return stages[1](h, tensors[3], tensors[4])
        }
        const gate = await stages[0](tensors[0], tensors[1]), up = await stages[1](tensors[0], tensors[2])
        await gate.realize(up)
        const h = await stages[2](gate, up)
        await h.realize()
        return stages[3](h, tensors[3])
      }
      try {
        for (let i = 0; i < arrays.length; i++) {
          const flat = new pg.Tensor(arrays[i])
          try { tensors.push(flat.reshape(...shapes[i])) } finally { flat.dispose() }
        }
        await tensors[0].realize(...tensors.slice(1))
        for (let i = 0; i < 5; i++) await (await call()).realize()
        let hidden = dense(arrays[0], arrays[1], c.h, mlp ? arrays[2] : null)
        if (mlp) hidden = hidden.map(x => Math.max(x, 0))
        else {
          const up = dense(arrays[0], arrays[2], c.h)
          hidden = hidden.map((x, i) => x / (1 + Math.exp(-x)) * up[i])
        }
        const expected = dense(hidden, arrays[3], c.d, mlp ? arrays[4] : null)
        const actual = await (await call()).toArray()
        for (let i = 0; i < actual.length; i++) assert(Math.abs(actual[i] - expected[i]) < 1e-7 + 2e-4 * Math.abs(expected[i]), `${c.name}: value ${i}`)
        assert(actual.length === c.d, `${c.name}: output shape`)
        assert(stages.every(f => f.scheduleCount === 1), `${c.name}: capture changed`)
        const samples = []
        for (let i = 0; i < 5; i++) {
          state.exports = state.imports = state.kernels = 0
          state.active = true
          try { await (await call()).realize() } finally { state.active = false }
          samples.push([state.exports, state.imports, state.kernels])
        }
        rows.push({ name: c.name, policy, expected: c.expected, samples })
      } finally {
        state.active = false
        for (const stage of stages) await stage.dispose()
        for (const tensor of tensors) await tensor.dispose()
      }
    }
    // Exact, reviewed counts: changes need explanation, not a wider budget.
    assert(rows.every(r => r.samples.every(s => s.every((n, i) => n === r.expected[i]))), JSON.stringify(rows))
    return rows
  } finally { await pg.dispose() }
}

async function main() {
  if (!process.argv.includes('--browser')) {
    delete process.env.DEBUG
    delete process.env.POLY_DEBUG
    delete process.env.POLY_KERNELS
    const state = observeBoundaries()
    try { console.log(JSON.stringify(await checkReplayBoundaries(require('..'), state))) }
    finally { state.restore() }
    return
  }
  const http = require('node:http'), fs = require('node:fs'), path = require('node:path')
  const { chromium } = require('playwright'), { launchOptionsFor } = require('./browser/run')
  const server = http.createServer((req, res) => {
    if (req.url === '/polygrad.js') {
      res.setHeader('Content-Type', 'text/javascript')
      fs.createReadStream(path.resolve(__dirname, '../dist/polygrad.sync.js')).pipe(res)
    } else { res.setHeader('Content-Type', 'text/html'); res.end('<script src="/polygrad.js"></script>') }
  })
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
  let browser
  try {
    browser = await chromium.launch(launchOptionsFor({ launcherName: 'chromium' }, 'webgpu').opts)
    const page = await browser.newPage()
    await page.addInitScript(`globalThis.boundaryState = (${observeBoundaries.toString()})()`)
    await page.goto(`http://127.0.0.1:${server.address().port}/`)
    console.log(JSON.stringify(await page.evaluate(`(${checkReplayBoundaries.toString()})(polygrad, boundaryState)`)))
  } finally { await browser?.close(); server.close() }
}
if (require.main === module) main().catch(e => { console.error(e); process.exitCode = 1 })
