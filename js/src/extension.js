'use strict'

const api = require('./extension_api.json')
const {validateMethods, attachMethods} = require('./extension_methods')

function manifest(text) {
  const m=JSON.parse(text)
  if(!Number.isInteger(m.inputs)||m.inputs<0||m.inputs>4096||
     !Number.isInteger(m.outputs)||m.outputs<1||m.outputs>4096||
     !Array.isArray(m.scalars)||m.scalars.length>4096||m.scalars.some(t=>!['int','double'].includes(t))||
     !Array.isArray(m.imports)||m.imports.some(n=>!Object.hasOwn(api,n)))throw Error('unsupported extension manifest')
  validateMethods(m)
  return m
}

function checkAbi(spec, legacyAbi, core) {
  if (Object.hasOwn(spec, 'construction_abi')) {
    const version = spec.construction_abi
    if (!Number.isInteger(version) || ![1,2].includes(version) ||
        !(version === 1 ? core.ffi.poly_extension_abi_version() === 1 : core.ffi.poly_extension_supports_abi?.(version)))
      throw Error('extension construction ABI mismatch')
  } else if (legacyAbi !== core.ffi.poly_abi_version()) {
    throw Error('extension ABI mismatch')
  }
}

async function loadExtension(rt,source) {
  const core=rt._core
  const idle=()=>{
    if(!rt._lifetime.alive||rt._closing||rt._core!==core)throw Error('extension runtime is disposed')
    if(rt._activeAsync)throw Error('extension construction requires an idle runtime')
  }
  idle()
  let build,spec,invoke,release,closed=false,methodApi
  if(rt.core==='native') {
    if(!source||typeof source.bind!=='function'||typeof source.build!=='function'||typeof source.abi!=='function')
      throw TypeError('native loadExtension expects the generated extension addon')
    spec=manifest(source.manifest())
    checkAbi(spec, source.abi(), core)
    if(!source.bind(core.ffi.poly_extension_resolver()))throw Error('extension symbol/ABI mismatch')
    build=(inputs,values)=>source.build(core.ctx,inputs,values)
    if (spec.methods) {
      if (typeof source.invoke !== 'function' || typeof source.release !== 'function') throw Error('missing extension method adapters')
      invoke=(method,handles,strings,values)=>source.invoke(core.ctx,method,handles,strings,values)
      release=(method,handle)=>source.release(method,handle)
    }
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
      if(d?.native)throw Error(`extension import requires native core: ${name}`)
      if(imp.module!=='env'||imp.kind!=='function'||!d||typeof fn!=='function')throw Error(`unsupported extension import: ${name}`)
      // Wasm i64 imports require BigInt even on the error path. Preserve the
      // original provider error instead of masking it with a coercion failure.
      const failure = d.return === 'int64_t' || d.return === 'uint64_t' ? BigInt(d.failure) : d.failure
      env[name]=(...args)=>{
        if(error&&!d.cleanup)return failure
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
        } catch(e) {error ||= e;return failure}
        finally {for(const {target} of allocations)M._free(target)}
      }
    }
    instance=await WebAssembly.instantiate(module,{env})
    idle()
    const e=instance.exports
    for(const name of ['poly_extension_abi','poly_extension_manifest','poly_extension_build','malloc','free'])
      if(typeof e[name]!=='function')throw Error(`missing extension export: ${name}`)
    const ptr=e.poly_extension_manifest()>>>0,heap=new Uint8Array(e.memory.buffer)
    let end=ptr
    while(end<heap.length&&end-ptr<65536&&heap[end])end++
    if(!ptr||end===heap.length||end-ptr===65536)throw Error('invalid extension manifest')
    spec=manifest(new TextDecoder().decode(heap.subarray(ptr,end)))
    checkAbi(spec, e.poly_extension_abi(), core)
    if (spec.methods) {
      for (const name of ['poly_extension_invoke', 'poly_extension_release'])
        if (typeof e[name] !== 'function') throw Error(`missing extension export: ${name}`)
      release=(method,handle)=>{ error=null; e.poly_extension_release(method,handle); if(error)throw error }
      invoke=(method,handles,strings,values)=>{
        error=null
        const blocks=[]; let output=0,handle=0,text=0
        const alloc=n=>{const p=e.malloc(Math.max(n,8))>>>0;if(!p)throw Error('extension allocation failed');blocks.push(p);return p}
        try {
          const hp=alloc(handles.length*4),sp=alloc(strings.length*4),vp=alloc(values.length*8)
          output=alloc(16)
          new Uint8Array(e.memory.buffer,output,16).fill(0)
          new Uint32Array(e.memory.buffer,hp,handles.length).set(handles.map(h=>h||0))
          new Float64Array(e.memory.buffer,vp,values.length).set(values)
          const pointers=strings.map(s=>{
            if(s===null)return 0
            const bytes=new TextEncoder().encode(s+'\0'),p=alloc(bytes.length)
            new Uint8Array(e.memory.buffer,p,bytes.length).set(bytes);return p
          })
          new Uint32Array(e.memory.buffer,sp,strings.length).set(pointers)
          const rc=e.poly_extension_invoke(method,core.ctx,hp,sp,vp,output,output+4,output+8)
          const dv=new DataView(e.memory.buffer)
          handle=dv.getUint32(output,true);text=dv.getUint32(output+4,true)
          if(error||rc)throw error||Error(`extension method ${spec.methods[method].name} failed (${rc})`)
          let string
          if(text){const heap=new Uint8Array(e.memory.buffer);let end=text;while(end<heap.length&&heap[end])end++
            if(end===heap.length)throw Error('invalid extension returned string')
            string=new TextDecoder('utf-8',{fatal:true}).decode(heap.subarray(text,end))}
          const result={handle,text:string,value:dv.getFloat64(output+8,true)}
          handle=0;return result
        } finally {
          if(handle)release(method,handle)
          if(text)e.free(text)
          for(const p of blocks)e.free(p)
        }
      }
    }
    // The module's actual imports must be declared, not just a compatible
    // subset in its manifest. The bridge already checked availability above.
    for (const imp of WebAssembly.Module.imports(module))
      if (imp.name.startsWith('poly_') && !spec.imports.includes(imp.name))
        throw Error(`undeclared extension import: ${imp.name}`)
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
  const wrap=handle=>{
    let device=core.ffi.poly_device_name(core.ffi.poly_tensor_device(handle))
    if(device==='auto'||device==='disk'||(device==='wasm'&&rt.device==='cpu'))device=rt.device
    return new rt.Tensor(null,{_tensor:handle,_ctx:core.ctx,_device:device})
  }
  if(spec.methods)methodApi=attachMethods(rt,spec,invoke,release,wrap,idle)
  return {
    ...(methodApi ? {methods:methodApi.methods} : {}),
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
    dispose() {closed=true;build=null;return methodApi?.dispose()}
  }
}
module.exports={loadExtension}
