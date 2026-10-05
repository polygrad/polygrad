/* Runs in Node or in the existing browser harness. No consumer private glue. */
async function checkExtension(pg,source,create) {
  const author=await pg.loadExtension(source),owned=[]
  const own=t=>{owned.push(t);return t}
  const theta=own(pg.Tensor.empty(3)),X=own(pg.Tensor.empty(6,3)),y=own(pg.Tensor.empty(6))
  await theta.copyFromAsync(new Float32Array([1,2,3]))
  await X.copyFromAsync(new Float32Array(18));await y.copyFromAsync(new Float32Array(6))
  const equal=(a,b)=>{if(a.length!==b.length||a.some((v,i)=>v!==b[i]))throw Error(`${a} != ${b}`)}
  const reject=(fn,pattern)=>{try{fn()}catch(e){if(pattern.test(e.message))return;throw e}throw Error('expected rejection')}
  let model,restored
  try {
    if(pg.device==='webgpu') {
      const pending=theta.toArrayAsync()
      try {reject(()=>author.build([theta,X,y],[0]),/idle runtime/)}
      finally {await pending}
    }
    if(pg.core==='native') {
      let rejected=false
      try {await pg.loadExtension({bind:source.bind,build:source.build,abi:()=>-1})}
      catch(e) {if(!/ABI mismatch/.test(e.message))throw e;rejected=true}
      if(!rejected)throw Error('extension ABI mismatch was accepted')
    }
    if(create) {
      const other=await create({core:pg.core,device:pg.device==='wasm'?'cpu':pg.device}),foreign=other.Tensor.empty(3)
      try {reject(()=>author.build([foreign,X,y],[0]),/runtime/)}
      finally {foreign.dispose();await other.dispose()}
    }
    const disposed=pg.Tensor.empty(3);disposed.dispose()
    reject(()=>author.build([disposed,X,y],[0]),/live Tensors/)
    reject(()=>author.build([theta,X,y],[NaN]),/scalar/)
    if(pg.core==='wasm') {
      const M=pg._core.Module,original=M._poly_tensor_assign
      let failed
      try {
        M._poly_tensor_assign=()=>{throw Error('injected bridge error')}
        failed=await pg.loadExtension(source)
      } finally {M._poly_tensor_assign=original}
      try {reject(()=>failed.build([theta,X,y],[2]),/injected bridge error/)}
      finally {failed.dispose()}
    }
    reject(()=>author.build([theta,X,y],[2]),/construction failed/)
    equal(await theta.toArrayAsync(),[1,2,3])
    const unsorted=own(pg.Tensor.empty(4))
    await unsorted.copyFromAsync(new Float32Array([2,-1,2,0]))
    const sorted=author.build([unsorted,X,y],[3]);sorted.forEach(own)
    equal(await sorted[0].toArrayAsync(),[-1,0,2,2])
    equal(await sorted[1].toArrayAsync(),[1,3,0,2])
    // Internal views may exceed the eight-axis named-binding limit.
    const flat=own(pg.Tensor.empty(512)),cube=own(flat.reshape(Array(9).fill(2)))
    const window=own(cube.shrink([...Array(8).fill([0,2]),[0,1]])),out=own(window.reshape(256))
    const highRank=await pg.Model.fromTensors({inputs:{flat},outputs:{out}})
    try {
      const result=await highRank.callAsync('forward',{flat:Float32Array.from({length:512},(_,i)=>i)})
      equal(result.out,Float32Array.from({length:256},(_,i)=>i*2))
    } finally {await highRank.dispose()}
    const sortInput=own(pg.Tensor.empty(448,{dtype:'int32'}))
    const [sortValues,sortIndices]=sortInput.sort();own(sortValues);own(sortIndices)
    const sortModel=await pg.Model.fromTensors({inputs:{x:sortInput},outputs:{sorted:sortValues}})
    let sortRestored
    try {
      sortRestored=pg.Model.load(await sortModel.saveAsync())
      const input=Int32Array.from({length:448},(_,i)=>447-i)
      for(const m of [sortModel,sortRestored]) {
        const result=await m.callAsync('forward',{x:input})
        equal(result.sorted,Int32Array.from({length:448},(_,i)=>i))
      }
    } finally {if(sortRestored)await sortRestored.dispose();await sortModel.dispose()}
    const [logp,gradient]=author.build([theta,X,y],[0]);own(logp);own(gradient)
    model=await pg.Model.fromTensors({inputs:{theta},outputs:{logp,gradient}})
    author.dispose()
    reject(()=>author.build([theta,X,y],[0]),/disposed/)
    restored=pg.Model.load(await model.saveAsync())
    for(const m of [model,restored]) {
      const got=await m.callAsync('forward',{theta:new Float32Array([-4,0,7])})
      equal(got.gradient,[4,0,-7]);equal(got.logp,[-32.5])
    }
  } finally {
    if(restored)await restored.dispose();if(model)await model.dispose()
    for(const t of owned)await t.dispose();author.dispose()
  }
  pg.clearScheduleCache();pg.collect()
  const s=pg.stats().coreStats
  if(s.tensorRecords||s.bufferOwnedBytes)throw Error(`extension leak: ${JSON.stringify(s)}`)
  return {gradient:'passed',checkpoint:'passed',rollback:'passed',bufferBytes:s.bufferOwnedBytes}
}
if(typeof module!=='undefined')module.exports=checkExtension
