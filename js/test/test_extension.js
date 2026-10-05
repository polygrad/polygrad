'use strict'
const fs=require('fs'),path=require('path'),{create}=require('..')
const checkExtension=require('./check_extension')
const root=path.resolve(__dirname,'../../build/extension')
;(async()=>{
  const core=process.env.POLY_CORE||'native',pg=await create({core,device:'cpu',logical:'always'})
  try{console.log(JSON.stringify({core,...await checkExtension(pg,core==='native'?require(path.join(root,'native.node')):fs.readFileSync(path.join(root,'author.wasm')),create)}))}
  finally{await pg.dispose()}
})().catch(e=>{console.error(e);process.exitCode=1})
