'use strict'
module.exports = async function checkExtensionMethods(pg, source, create) {
  const assert = (ok, message) => { if (!ok) throw Error(message) }
  const reject = (fn, pattern) => { try { fn() } catch (e) { if(pattern.test(e.message))return;throw e } throw Error('expected rejection') }
  const author = await pg.loadExtension(source), second = await pg.loadExtension(source)
  const m = author.methods
  let x, y, model, restored, a, b
  try {
    m.example_reset()
    x = pg.Tensor.empty(1); await x.copyFromAsync(new Float32Array([1]))
    const stale = m.example_new(x, 'old generation')
    await stale.dispose()
    const next = m.example_new(x, 'new generation')
    reject(()=>m.example_name(stale), /live object/)
    await next.dispose(); m.example_reset()
    a = m.example_new(x, 'μ'); b = m.example_new(x, 'second')
    assert(m.example_name(a) === 'μ' && m.example_name_copy(a) === 'μ', 'string ownership')
    reject(()=>second.methods.example_finish(a), /extension instance/)
    reject(()=>m.example_finish(1), /extension instance/)
    reject(()=>m.example_new(x, 'bad\0name'), /NUL/)
    reject(()=>m.example_add(a, x, 'fail'), /failed.*\(-1\)/)
    if (pg.device === 'webgpu') {
      const pending = x.toArrayAsync()
      let closing
      try { reject(()=>m.example_finish(a), /idle runtime/); closing = b.dispose() }
      finally { await pending; await closing }
      reject(()=>m.example_name(b), /live object/)
    }
    m.example_add(a, x, 'ok')
    await x.dispose(); x = null; await pg.collect()
    y = m.example_finish(a)
    const parameter = m.example_input(a)
    try { model = await pg.Model.fromTensors({params:{x:parameter}, outputs:{y}}) }
    finally { await parameter.dispose() }
    model.setMetadata('example', {name:'μ', schema:1})
    await y.dispose(); y = null
    await author.dispose()
    reject(()=>m.example_finish(a), /disposed/)
    await a.dispose(); await b.dispose()
    // Native addons share module state; Wasm instances intentionally do not.
    if(pg.core === 'native') {
      assert(second.methods.example_order() === 21, 'not reverse creation order')
      assert(second.methods.example_live() === 0, 'objects leaked')
    }
    restored = await pg.Model.load(await model.saveAsync())
    const result = await restored.callAsync('forward', {})
    assert(Math.abs(result.y[0] - Math.exp(2)) < 2e-6, 'retained graph changed')
    assert(restored.getMetadata('example').name === 'μ', 'application metadata lost')
    if (create) {
      const rt = await create({core:pg.core, device:pg.device==='wasm'?'cpu':pg.device, logical:'always'})
      const ext = await rt.loadExtension(source), t = rt.Tensor.empty(1)
      await t.copyFromAsync(new Float32Array([0]))
      const o = ext.methods.example_new(t, 'runtime owned')
      reject(()=>second.methods.example_new(t,'foreign'), /runtime/)
      await rt.dispose()
      reject(()=>ext.methods.example_finish(o), /disposed/)
      await o.dispose()
      if(pg.core === 'native') assert(second.methods.example_live() === 0, 'runtime did not close objects')
    }
    // Separate Wasm instances cannot observe each other's C static counters.
    // Core Tensor ownership is shared, so use it to check author teardown.
    await pg.collect()
    const baseline = pg.stats().coreStats.tensorRecords
    const leakAuthor = await pg.loadExtension(source), term = pg.Tensor.empty(1)
    try {
      const held = leakAuthor.methods.example_new(term, 'retained until close')
      await term.dispose(); await pg.collect()
      assert(pg.stats().coreStats.tensorRecords > baseline, 'fixture did not retain its Tensor')
      await leakAuthor.dispose(); await pg.collect()
      reject(()=>leakAuthor.methods.example_name(held), /disposed/)
      assert(pg.stats().coreStats.tensorRecords === baseline, 'author leaked core Tensors')
    } finally { await term.dispose(); await leakAuthor.dispose() }
    if (create) {
      const rt = await create({core:pg.core, device:pg.device==='wasm'?'cpu':pg.device})
      const ext = await rt.loadExtension(source), term = rt.Tensor.empty(1)
      ext.methods.example_new(term, 'close despite earlier error')
      const failing = {dispose() { throw Error('fixture close failure') }}
      rt._extensions.add(failing)
      let error
      try {
        try { await rt.dispose() } catch(e) { error=e }
        assert(error && /fixture close failure/.test(error.message), 'close error lost')
        assert(rt._core === null && !rt._lifetime.alive, 'close error stranded runtime')
      } finally {
        rt._extensions.delete(failing)
        if (rt._core) { rt._disposePromise=null; await rt.dispose() }
      }
    }
    return {methods:9, ownership:true, metadata:true}
  } finally {
    await author.dispose(); await second.dispose()
    if(restored)await restored.dispose();if(model)await model.dispose()
    if(y)await y.dispose();if(x)await x.dispose()
  }
}
