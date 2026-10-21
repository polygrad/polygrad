'use strict'
const assert=require('assert/strict')
const {loadExtension}=require('../src/extension')
// No instantiation or core entry is permitted for a native pointer/record ABI.
const string=s=>[s.length,...Buffer.from(s)]
const section=(id,b)=>[id,b.length,...b]
const bytes=new Uint8Array([0,97,115,109,1,0,0,0,
  ...section(1,[1,96,0,0]),
  ...section(2,[1,...string('env'),...string('poly_model_call'),0,0])])
const core={Module:{}}
const runtime={core:'wasm',_core:core,_lifetime:{alive:true}}
loadExtension(runtime,bytes).then(()=>{throw Error('native import admitted')},error=>{
  assert.match(error.message,/requires native core: poly_model_call/)
  console.log('native-only import rejected before instantiation')
}).catch(error=>{console.error(error);process.exitCode=1})
