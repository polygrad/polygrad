'use strict'

// Each synchronous scenario starts in a fresh process/page: its Wasm module
// snapshots the environment once. Native runtimes still use process.env.
async function checkEnvironment(pg, asyncMode, device, scenario) {
  const assert = (ok, message) => { if (!ok) throw new Error(message) }
  const create = options => asyncMode ? pg.createAsync(options) : pg.create(options)
  const options = { core: 'wasm', device }
  if (scenario !== 'default' && scenario !== 'inherit') options.env = {
    REDUCEOP_SPLIT_THRESHOLD: '1024', BEAM_COMPARE: '0',
    ...(scenario === 'disabled' ? { SPLIT_REDUCEOP: '0' } : {}),
    ...(scenario === 'budget' ? { REDUCEOP_SPLIT_SIZE: '0' } : {}),
    ...(scenario === 'invalid' ? { REDUCEOP_SPLIT_THRESHOLD: '1e3' } : {}),
    ...(scenario === 'overflow' ? { REDUCEOP_SPLIT_THRESHOLD: '2147483648' } : {})
  }
  const rt = await create(options)
  const Module = rt._core.Module
  assert(!Object.hasOwn(Module, '__polygradStartupEnv'), 'module retains inherited environment snapshot')
  assert(!Object.hasOwn(Module, 'polygradEnv'), 'module retains startup environment argument')
  assert(JSON.stringify(Module.__polygradExplicitEnv) === JSON.stringify(options.env || {}),
    'conflict bookkeeping must contain only explicit environment keys')
  async function count(runtime) {
    const x = new runtime.Tensor(Float32Array.from({ length: 8192 }, (_, i) => i % 7))
    await x.realizeAsync()
    const before = runtime.stats().coreStats.kernelCount, y = x.sum()
    try {
      const value = await y.toArrayAsync()
      assert(value[0] === 24571, 'sum value changed')
      return runtime.stats().coreStats.kernelCount - before
    } finally { await y.dispose(); await x.dispose() }
  }
  try {
    const kernels = await count(rt), split = scenario === 'env' || scenario === 'inherit'
    assert(kernels === (split ? 2 : 1), `${scenario}: expected ${split ? 2 : 1} kernels, got ${kernels}`)
    if (scenario !== 'default') assert(rt.beamCompare === 0, 'BEAM_COMPARE environment ignored')
    if (!asyncMode && (scenario === 'default' || scenario === 'inherit')) {
      let error
      try {
        const other = await create({ ...options, env: { REDUCEOP_SPLIT_THRESHOLD: scenario === 'inherit' ? '1024' : '32768' } })
        await other.dispose()
      } catch (e) { error = e }
      assert(error && /environment.*initialized/i.test(error.message), 'late explicit key was not rejected')
    }
    if (scenario === 'env') {
      const same = await create(options)
      try { assert(await count(same) === 2, 'matching environment rejected or changed') }
      finally { await same.dispose() }
      let other, error
      try { other = await create({ ...options, env: { REDUCEOP_SPLIT_THRESHOLD: '32768' } }) }
      catch (e) { error = e }
      try {
        if (asyncMode) {
          assert(!error && other, 'async modules were not isolated')
          assert(await count(other) === 1 && await count(rt) === 2, 'async environment leaked')
        } else assert(error && /environment.*initialized/i.test(error.message), 'sync environment conflict ignored')
      } finally { if (other) await other.dispose() }
    }
    return { scenario, asyncMode, device, kernels }
  } finally { await rt.dispose() }
}

async function main() {
  const assert = require('node:assert/strict'), path = require('node:path')
  if (process.argv[2] === '--child') {
    const pg = require('..'), asyncMode = process.argv[3] === 'async', scenario = process.argv[4]
    for (const env of [[], null, false, { A: 1 }, { 'A=B': 'x' }, { '': 'x' }, { A: 'x\0y' }]) {
      await assert.rejects(async () => {
        const rt = await (asyncMode ? pg.createAsync({core:'wasm', env}) : pg.create({core:'wasm', env}))
        await rt.dispose()
      }, /env/)
    }
    assert.throws(() => pg.create({core:'native',env:{A:'1'}}), /env/)
    console.log(JSON.stringify(await checkEnvironment(pg, asyncMode, 'cpu', scenario)))
    return
  }
  if (process.argv[2] !== '--browser') {
    const { spawnSync } = require('node:child_process')
    for (const mode of ['sync', 'async']) for (const scenario of ['default','env','inherit','disabled','budget','invalid','overflow']) {
      const env = { ...process.env }
      env.POLY_TEST_ENV_SECRET = 'environment-retention-regression-sentinel'
      for (const k of ['SPLIT_REDUCEOP','REDUCEOP_SPLIT_THRESHOLD','REDUCEOP_SPLIT_SIZE','BEAM','NOOPT','BEAM_COMPARE']) delete env[k]
      if (scenario === 'inherit') Object.assign(env,{REDUCEOP_SPLIT_THRESHOLD:'1024',BEAM_COMPARE:'0'})
      if (scenario === 'env') Object.assign(env,{REDUCEOP_SPLIT_THRESHOLD:'32768',BEAM_COMPARE:'1'})
      const result=spawnSync(process.execPath,[__filename,'--child',mode,scenario],{env,encoding:'utf8'})
      process.stdout.write(result.stdout);process.stderr.write(result.stderr)
      assert.equal(result.status,0,`${mode}/${scenario} failed`)
      if (scenario === 'invalid' || scenario === 'overflow')
        assert.match(result.stderr, /REDUCEOP_SPLIT_THRESHOLD requires a decimal integer.*using default 32768/)
    }
    return
  }
  const fs=require('node:fs'),http=require('node:http'),{chromium}=require('playwright')
  const {launchOptionsFor}=require('./browser/run')
  const server=http.createServer((req,res)=>{
    if(req.url==='/sync.js'||req.url==='/async.js') {
      res.setHeader('Content-Type','text/javascript')
      fs.createReadStream(path.join(__dirname,'../dist',`polygrad.${req.url.slice(1)}`)).pipe(res)
    } else { res.setHeader('Content-Type','text/html');res.end(`<!doctype html><script src="/${req.url.includes('sync-only')?'sync':'async'}.js"></script>`) }
  })
  await new Promise(r=>server.listen(0,'127.0.0.1',r));let browser
  try {
    browser=await chromium.launch(launchOptionsFor({launcherName:'chromium',executablePath:process.env.POLY_BROWSER_PATH||''},'webgpu').opts)
    for(const [asyncMode,device] of [[false,'cpu'],[true,'cpu'],[true,'webgpu']])for(const scenario of ['default','env','disabled','budget','invalid','overflow']) {
      const page=await browser.newPage()
      const messages=[]
      page.on('console', message => messages.push(message.text()))
      try {
        await page.goto(`http://127.0.0.1:${server.address().port}/${asyncMode?'async':'sync-only'}`)
        const result=await page.evaluate(({source,asyncMode,device,scenario})=>new Function(`return (${source})`)()(polygrad,asyncMode,device,scenario),{source:checkEnvironment.toString(),asyncMode,device,scenario})
        console.log(JSON.stringify(result))
        if (scenario === 'invalid' || scenario === 'overflow')
          assert.match(messages.join('\n'), /REDUCEOP_SPLIT_THRESHOLD requires a decimal integer.*using default 32768/)
      } finally { await page.close() }
    }
  } finally { await browser?.close();server.close() }
}
if(require.main===module)main().catch(e=>{console.error(e);process.exitCode=1})
module.exports={checkEnvironment}
