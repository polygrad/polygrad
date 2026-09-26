// Matched optional-kernel comparison, including Model input upload/readback.
import { createRequire } from 'node:module';
import http from 'node:http';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
for (const arg of process.argv.slice(2))
  if (!['--browser', '--webgpu', '--profile', '--plain'].includes(arg)) throw new Error(`unknown option ${arg}`);
const browserMode = process.argv.includes('--browser');
const options = {device: process.argv.includes('--webgpu') ? 'webgpu' : 'cpu',
  profile: process.argv.includes('--profile'), epilogue: !process.argv.includes('--plain'),
  pairs: Number(process.env.KERNEL_PAIRS || 3), warmup: Number(process.env.KERNEL_WARMUP || 30),
  samples: Number(process.env.KERNEL_SAMPLES || 15)};
options.shapes = process.env.KERNEL_SHAPES ? JSON.parse(process.env.KERNEL_SHAPES) : [[8,64,48], [128,384,1536]];
if (!Array.isArray(options.shapes) || !options.shapes.length || options.shapes.some(s =>
    !Array.isArray(s) || s.length !== 3 || s.some(n => !Number.isSafeInteger(n) || n < 1)))
  throw new Error('KERNEL_SHAPES must contain positive integer [M,K,N] triples');
for (const key of ['pairs', 'warmup', 'samples'])
  if (!Number.isInteger(options[key]) || options[key] < 1) throw new Error(`invalid ${key}`);
if (options.device === 'webgpu' && !browserMode) throw new Error('WebGPU requires --browser');

// Diagnostic-only instrumentation; no runtime or scheduling changes. GPU
// timestamps exclude transfers. Wasm direct replays exclude FFI and packing
// when reporting the matmul kernel, while retaining each kernel's own row.
function installKernelProbe() {
  const records = [];
  const Instance = WebAssembly.Instance;
  WebAssembly.Instance = function(mod, imports) {
    const inst = new Instance(mod, imports);
    if (!inst.exports.kernel) return inst;
    const fn = inst.exports.kernel, record = {fn, args:null};
    records.push(record);
    const kernel = (...args) => {record.args=args; return fn(...args);};
    Object.defineProperty(kernel, 'length', {value:fn.length});
    return {exports:{...inst.exports,kernel}};
  };
  WebAssembly.Instance.prototype = Instance.prototype;
  let gpu = null, armed = false;
  const shaderCodes = new WeakMap(), pipelineCodes = new WeakMap();
  if (typeof GPUDevice !== 'undefined') {
    const shader = GPUDevice.prototype.createShaderModule;
    GPUDevice.prototype.createShaderModule = function(desc) {
      const s=shader.call(this,desc); shaderCodes.set(s,desc.code); return s;
    };
    const pipeline=GPUDevice.prototype.createComputePipeline;
    GPUDevice.prototype.createComputePipeline = function(desc) {
      const p=pipeline.call(this,desc);pipelineCodes.set(p,shaderCodes.get(desc.compute.module));return p;
    };
    const encoder=GPUDevice.prototype.createCommandEncoder;
    GPUDevice.prototype.createCommandEncoder = function(...args) {
      const enc=encoder.apply(this,args);
      if (!armed) return enc;
      if (!this.features.has('timestamp-query')) throw new Error('timestamp-query required for profiling');
      if (!gpu) gpu={device:this,query:this.createQuerySet({type:'timestamp',count:2048}),rows:[]};
      const begin=enc.beginComputePass.bind(enc);
      enc.beginComputePass = (desc={}) => {
        const index=gpu.rows.length*2;
        if(index>=2048) throw new Error('too many profiled passes');
        const row={index};gpu.rows.push(row);
        const pass=begin({...desc,timestampWrites:{querySet:gpu.query,beginningOfPassWriteIndex:index,endOfPassWriteIndex:index+1}});
        const set=pass.setPipeline.bind(pass);
        pass.setPipeline=p=>{row.code=pipelineCodes.get(p);set(p);};
        return pass;
      };
      return enc;
    };
  }
  globalThis.kernelProbe={
    mark:()=>records.length,
    forget:start=>{records.splice(start);},
    wasm:(start)=>records.slice(start).filter(r=>r.args).map((r,id)=>{
      for(let i=0;i<50;i++) r.fn(...r.args);
      const samples=[];
      for(let round=0;round<5;round++) {
        const t=performance.now();for(let i=0;i<20;i++) r.fn(...r.args);
        samples.push((performance.now()-t)/20);
      }
      return {id,medianMs:samples.sort((a,b)=>a-b)[2]};
    }),
    arm:()=>{armed=true;},
    finish:async()=>{
      armed=false;
      if(!gpu) throw new Error('no GPU passes recorded');
      const {device,query,rows}=gpu, size=rows.length*16;
      const resolve=device.createBuffer({size,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});
      const read=device.createBuffer({size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
      try {
        const enc=device.createCommandEncoder();enc.resolveQuerySet(query,0,rows.length*2,resolve,0);
        enc.copyBufferToBuffer(resolve,0,read,0,size);device.queue.submit([enc.finish()]);
        await read.mapAsync(GPUMapMode.READ);
        const stamps=new BigUint64Array(read.getMappedRange());
        return rows.map(r=>({code:r.code,ms:Number(stamps[r.index+1]-stamps[r.index])/1e6}));
      } finally {read.unmap();read.destroy();resolve.destroy();query.destroy();gpu=null;}
    }
  };
}

async function benchmark(opts) {
  const rows = [], median = xs => [...xs].sort((a,b)=>a-b)[Math.floor(xs.length/2)];
  // Small projection and MiniLM FFN up-projection shapes, followed by bias/ReLU.
  for (const [m,k,n] of opts.shapes) {
    let portable;
    const a = Float32Array.from({length:m*k}, (_,i)=>(i%7-3)/17);
    const b = Float32Array.from({length:k*n}, (_,i)=>(i%11-5)/19);
    const bias = Float32Array.from({length:n}, (_,i)=>(i%3-1)/23);
    const expected = new Float32Array(m*n);
    for (let i=0;i<m;i++) for (let j=0;j<n;j++) {
      let v=0;
      for (let r=0;r<k;r++) v += a[i*k+r]*b[r*n+j];
      expected[i*n+j] = opts.epilogue ? Math.max(v+bias[j],0) : v;
    }
    for (let pair=0;pair<opts.pairs;pair++) for (const kernels of pair%2 ? [true,false] : [false,true]) {
      const firstKernel=globalThis.kernelProbe?.mark();
      const pg = await globalThis.polygrad.createAsync({core:'wasm',device:opts.device,kernels});
      let model;
      const x=pg.Tensor.empty([m,k]), w=new pg.Tensor(b).reshape(k,n), z=new pg.Tensor(bias);
      try {
        await w.realizeAsync(); await z.realizeAsync();
        const cold=performance.now();
        model = await pg.Model.fromCallableAsync(({x})=>opts.epilogue ? x.dot(w).add(z).relu() : x.dot(w), {inputs:{x},params:{w,z}});
        const call = async () => (await model.forwardAsync({x:a})).output;
        let maxAbsError=0;
        const check = y => {
          if (y.length !== expected.length) throw new Error('wrong output size');
          for (let i=0;i<y.length;i++) {
            const error=Math.abs(y[i]-expected[i]);
            if (!Number.isFinite(error) || error > 2e-5+2e-5*Math.abs(expected[i]))
              throw new Error(`wrong value ${opts.device}/${kernels}/${m},${k},${n} at ${i}: ${y[i]} != ${expected[i]}`);
            maxAbsError=Math.max(maxAbsError,error);
          }
        };
        check(await call());
        const coldMs=performance.now()-cold;
        const bytes=await model.saveAsync();
        if (portable && (bytes.length !== portable.length || bytes.some((v,i)=>v!==portable[i])))
          throw new Error('kernel selection changed portable bundle bytes');
        portable=bytes;
        for (let i=0;i<opts.warmup;i++) check(await call());
        const times=[];
        for (let i=0;i<opts.samples;i++) {
          const start=performance.now(), y=await call();
          times.push(performance.now()-start);
          check(y);
        }
        const row={core:pg.core,device:pg.device,physicalDevice:x.uop.device,shape:[m,k,n],pair,kernels,
          coldMs,medianMs:median(times),samplesMs:times,maxAbsError};
        rows.push(row); console.log(JSON.stringify(row));
        if(opts.profile) {
          if(opts.device==='webgpu') {
            kernelProbe.arm();
            for(let i=0;i<5;i++) check(await call());
            console.log(JSON.stringify({profile:'gpu-timestamps',shape:[m,k,n],kernels,pair,passes:await kernelProbe.finish()}));
          } else {
            console.log(JSON.stringify({profile:'wasm-direct',shape:[m,k,n],kernels,pair,kernelTimes:kernelProbe.wasm(firstKernel)}));
          }
        }
      } finally {
        if(model) await model.dispose();
        x.dispose(); w.dispose(); z.dispose(); await pg.dispose();
        if(opts.profile) kernelProbe.forget(firstKernel);
      }
    }
  }
  return rows;
}

console.log(JSON.stringify({node:process.version,load:os.loadavg(),options,
  revision:execFileSync('git',['rev-parse','HEAD'],{cwd:root,encoding:'utf8'}).trim(),
  asyncCoreSHA256:createHash('sha256').update(fs.readFileSync(path.join(root,'js/wasm/core.async.js'))).digest('hex')}));
if (!browserMode) {
  if(options.profile) installKernelProbe();
  globalThis.polygrad = createRequire(import.meta.url)('../js');
  await benchmark(options);
} else {
  const {chromium} = await import('../js/node_modules/playwright/index.mjs');
  const bundle = path.join(root,'js/dist/polygrad.async.js');
  const server = http.createServer((req,res)=>{
    res.setHeader('Cross-Origin-Opener-Policy','same-origin');
    res.setHeader('Cross-Origin-Embedder-Policy','require-corp');
    if(req.url === '/bundle.js') {
      res.setHeader('Content-Type','text/javascript'); fs.createReadStream(bundle).pipe(res);
    } else if(req.url === '/') {
      res.setHeader('Content-Type','text/html'); res.end('<script src="/bundle.js"></script>');
    } else {res.writeHead(req.url === '/favicon.ico' ? 204 : 404);res.end();}
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  let browser;
  try {
    browser=await chromium.launch({headless:false, executablePath:process.env.POLY_BROWSER_EXECUTABLE || '/usr/bin/google-chrome',
      args:['--no-sandbox','--enable-unsafe-webgpu','--enable-features=Vulkan,DefaultANGLEVulkan,VulkanFromANGLE',
        '--disable-gpu-sandbox','--ignore-gpu-blocklist','--disable-software-rasterizer','--use-angle=vulkan']});
    const page=await browser.newPage();
    page.on('console',msg=>console.log(msg.text()));
    page.on('pageerror',err=>console.error(err));
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    if(options.profile) await page.evaluate(installKernelProbe);
    console.log(JSON.stringify({browser:browser.version(),adapter:await page.evaluate(async()=>{
      if(!navigator.gpu) return null;
      const adapter=await navigator.gpu.requestAdapter();
      return adapter ? {vendor:adapter.info.vendor,architecture:adapter.info.architecture,
        device:adapter.info.device,description:adapter.info.description} : null;
    })}));
    await page.evaluate(benchmark, options);
  } finally {if(browser) await browser.close();await new Promise(resolve=>server.close(resolve));}
}
console.log(JSON.stringify({loadEnd:os.loadavg()}));
