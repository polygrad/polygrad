'use strict'

const api = require('./extension_api.json')

function manifest(text) {
  const m=JSON.parse(text)
  if(!Number.isInteger(m.inputs)||m.inputs<0||m.inputs>4096||
     !Number.isInteger(m.outputs)||m.outputs<1||m.outputs>4096||
     !Array.isArray(m.scalars)||m.scalars.length>4096||m.scalars.some(t=>!['int','double'].includes(t))||
     !Array.isArray(m.imports)||m.imports.some(n=>!Object.hasOwn(api,n)))throw Error('unsupported extension manifest')
  return m
}

async function loadExtension(rt,source) {
  const core=rt._core
  const idle=()=>{
    if(!rt._lifetime.alive||rt._closing||rt._core!==core)throw Error('extension runtime is disposed')
    if(rt._activeAsync)throw Error('extension construction requires an idle runtime')
  }
  idle()
  let build,spec,closed=false
  if(rt.core==='native') {
    if(!source||typeof source.bind!=='function'||typeof source.build!=='function'||typeof source.abi!=='function')
      throw TypeError('native loadExtension expects the generated extension addon')
    if(source.abi()!==core.ffi.poly_abi_version())throw Error('extension ABI mismatch')
    spec=manifest(source.manifest())
    if(!source.bind(core.ffi.poly_extension_resolver()))throw Error('extension symbol/ABI mismatch')
    build=(inputs,values)=>source.build(core.ctx,inputs,values)
  } else {
    const M=core.Module
    const module=source instanceof WebAssembly.Module?source:await WebAssembly.compile(source)
    idle()
    let instance,error
    const env={}
    for(const imp of WebAssembly.Module.imports(module)) {
      // No consumer heap views are cached; each copy reacquires memory.buffer.
      if(imp.module==='env'&&imp.name==='emscripten_notify_memory_growth'&&imp.kind==='function') {
        env[imp.name]=()=>{};continue
      }
      const name=imp.name,d=api[name],fn=M['_'+name]
      if(imp.module!=='env'||imp.kind!=='function'||!d||typeof fn!=='function')throw Error(`unsupported extension import: ${name}`)
      env[name]=(...args)=>{
        if(error&&!d.cleanup)return d.failure
        const allocations=[]
        try {
          for(const [i,ci,width,direction] of d.arrays) {
            const count=ci<0?-ci:args[ci],ptr=args[i]>>>0,len=count*width,heap=instance.exports.memory.buffer
            if(!Number.isSafeInteger(count)||count<0||!Number.isSafeInteger(len)||ptr+len>heap.byteLength||(!ptr&&len))
              throw Error(`invalid array argument: ${name}`)
            if(!len){args[i]=0;continue}
            const target=M._malloc(len)>>>0
            if(!target)throw Error('extension marshalling allocation failed')
            allocations.push({ptr,target,len,direction})
            if(direction==='in')M.HEAPU8.set(new Uint8Array(heap,ptr,len),target)
            else M.HEAPU8.fill(0,target,target+len)
            args[i]=target
          }
          const result=fn(...args)
          for(const {ptr,target,len,direction} of allocations)if(direction==='out')
            new Uint8Array(instance.exports.memory.buffer,ptr,len).set(M.HEAPU8.subarray(target,target+len))
          return result
        } catch(e) {error ||= e;return d.failure}
        finally {for(const {target} of allocations)M._free(target)}
      }
    }
    instance=await WebAssembly.instantiate(module,{env})
    idle()
    const e=instance.exports
    for(const name of ['poly_extension_abi','poly_extension_manifest','poly_extension_build','malloc','free'])
      if(typeof e[name]!=='function')throw Error(`missing extension export: ${name}`)
    if(e.poly_extension_abi()!==core.ffi.poly_abi_version())throw Error('extension ABI mismatch')
    const ptr=e.poly_extension_manifest()>>>0,heap=new Uint8Array(e.memory.buffer)
    let end=ptr
    while(end<heap.length&&end-ptr<65536&&heap[end])end++
    if(!ptr||end===heap.length||end-ptr===65536)throw Error('invalid extension manifest')
    spec=manifest(new TextDecoder().decode(heap.subarray(ptr,end)))
    build=(inputs,values)=>{
      error=null
      const blocks=[];let out=0,handles=[]
      const alloc=n=>{const p=e.malloc(Math.max(n,8))>>>0;if(!p)throw Error('extension allocation failed');blocks.push(p);return p}
      try {
        const ip=alloc(inputs.length*4),sp=alloc(values.length*8)
        out=alloc(spec.outputs*4)
        new Uint32Array(e.memory.buffer,ip,inputs.length).set(inputs)
        new Float64Array(e.memory.buffer,sp,values.length).set(values)
        new Uint32Array(e.memory.buffer,out,spec.outputs).fill(0)
        const ok=e.poly_extension_build(core.ctx,ip,sp,out)
        handles=Array.from(new Uint32Array(e.memory.buffer,out,spec.outputs))
        if(error||!ok||handles.some(h=>!h))throw error||Error('extension construction failed')
        const result=handles;handles=[];return result
      } finally {
        for(const h of handles)if(h)core.ffi.poly_tensor_release(h)
        for(const p of blocks)e.free(p)
      }
    }
  }
  return {
    build(tensors=[],values=[]) {
      idle()
      if(closed)throw Error('extension is disposed')
      if(!Array.isArray(tensors)||tensors.length!==spec.inputs||!Array.isArray(values)||values.length!==spec.scalars.length)
        throw TypeError('extension argument count mismatch')
      for(const t of tensors)if(!t||t._rt!==rt||!t._tensor)throw Error('extension inputs must be live Tensors from this runtime')
      values.forEach((v,i)=>{if(typeof v!=='number'||!Number.isFinite(v)||
        (spec.scalars[i]==='int'&&(!Number.isInteger(v)||v< -2147483648||v>2147483647)))throw TypeError('invalid extension scalar')})
      const handles=build(tensors.map(t=>t._tensor),values),results=[]
      try {
        for(let i=0;i<handles.length;i++) {
          let device=core.ffi.poly_device_name(core.ffi.poly_tensor_device(handles[i]))
          if(device==='auto'||device==='disk'||(device==='wasm'&&rt.device==='cpu'))device=rt.device
          const t=new rt.Tensor(null,{_tensor:handles[i],_ctx:core.ctx,_device:device})
          handles[i]=null;results.push(t)
        }
        return results
      } catch(e) {for(const t of results)t.dispose();throw e}
      finally {for(const h of handles)if(h)core.ffi.poly_tensor_release(h)}
    },
    // Models/Tensors own their results independently. No captured C callback is
    // registered in the graph; disposing the author does not invalidate them.
    dispose() {closed=true;build=null}
  }
}
module.exports={loadExtension}
