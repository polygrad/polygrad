'use strict'
module.exports = async function checkExtensionMethods(pg, source, create) {
  const assert = (ok, message) => { if (!ok) throw Error(message) }
  const reject = (fn, pattern) => { try { fn() } catch (e) { if(pattern.test(e.message))return;throw e } throw Error('expected rejection') }
  const author = await pg.loadExtension(source), second = await pg.loadExtension(source)
  const m = author.methods
  let x, y, model, restored, a, b
  try {
    assert(m.example_dtype('int32')===pg._core.dtypeIds.int32,'imported dtype string')
    assert(m.example_dtype('int64')===pg._core.dtypeIds.int64,'imported int64 string')
    assert(m.example_dtype('')===-1&&m.example_dtype('invalid')===-1,'unknown dtype string')
    assert(m.example_dtype_pointer(0)===-1,'nullable dtype string')
    if(pg.core==='wasm') {
      reject(()=>m.example_dtype_pointer(1),/invalid string argument/)
      reject(()=>m.example_dtype_pointer(2),/unterminated string argument/)
      assert(m.example_dtype('float32')===pg._core.dtypeIds.float32,'string failure recovery')
      const M=pg._core.Module,original=M._poly_dtype_id_by_name
      let failing
      try {
        M._poly_dtype_id_by_name=()=>{throw Error('dtype provider failure')}
        failing=await pg.loadExtension(source)
        reject(()=>failing.methods.example_dtype('int32'),/dtype provider failure/)
      }finally{M._poly_dtype_id_by_name=original;if(failing)await failing.dispose()}
      const malloc=M._malloc,free=M._free,owned=new Set()
      try {
        M._malloc=n=>{const p=malloc(n);if(p)owned.add(p);return p}
        M._free=p=>{owned.delete(p);return free(p)}
        assert(m.example_dtype('x'.repeat(20*1024*1024))===-1,'string after author heap growth')
        assert(m.example_dtype('int32')===pg._core.dtypeIds.int32,'string after heap growth')
        assert(owned.size===0,'temporary string copies released')
        M._malloc=()=>0
        reject(()=>m.example_dtype('int32'),/marshalling allocation failed/)
      }finally{M._malloc=malloc;M._free=free}
    }
    const shaped = pg.Tensor.empty([2, 3]), scalar = new pg.Tensor(1)
    assert(Number(m.example_static_dim(shaped, 0)) === 2, 'static first extent')
    assert(Number(m.example_static_dim(shaped, 1)) === 3, 'static second extent')
    assert(Number(m.example_static_dim(shaped, 2)) === -1, 'invalid axis')
    assert(Number(m.example_static_dim(shaped, -1)) === -1, 'negative axis')
    assert(Number(m.example_static_dim(scalar, 0)) === -1, 'scalar has no axis')
    const large = pg.Tensor.empty(2**34), empty = pg.Tensor.empty([0, 3])
    const dynamic = pg.Tensor.empty([pg.uop.variable('author_shape', 1, 8).bind(3), 2])
    assert(m.example_static_dim(large, 0) === 2**34, 'int64 extent lost across SDK')
    assert(m.example_static_dim(empty, 0) === 0, 'empty extent')
    assert(m.example_static_dim(dynamic, 0) === -1, 'variable extent accepted as static')
    assert(m.example_static_dim(dynamic, 1) === 2, 'static axis of symbolic shape')
    if (pg.core === 'wasm') {
      const M=pg._core.Module, original=M._poly_tensor_static_dim
      let failing
      try {
        M._poly_tensor_static_dim=()=>{throw Error('shape probe failure')}
        failing=await pg.loadExtension(source)
        reject(()=>failing.methods.example_static_dim(shaped,0), /shape probe failure/)
      } finally {M._poly_tensor_static_dim=original;if(failing)await failing.dispose()}
    }
    shaped.dispose(); scalar.dispose(); large.dispose(); empty.dispose(); dynamic.dispose()
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
    assert(m.example_model_buffers(model) === model.bufCount, 'borrowed Model admission')
    reject(()=>m.example_model_buffers(x), /live Model/)
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
      reject(()=>ext.methods.example_model_buffers(model), /live Model/)
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
    return {methods:13, ownership:true, metadata:true}
  } finally {
    await author.dispose(); await second.dispose()
    if(restored)await restored.dispose();if(model)await model.dispose()
    if(y)await y.dispose();if(x)await x.dispose()
  }
}
