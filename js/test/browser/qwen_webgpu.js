'use strict'

const http = require('http')
const fs = require('fs')
const path = require('path')

const repoRoot = path.resolve(__dirname, '..', '..', '..')
const jsDir = path.join(repoRoot, 'js')
const ggufPath = process.env.POLY_QWEN3_GGUF

const EXPECTED_LEN = 639447744
const EXPECTED_TOKENS = [785, 6722, 315, 9625, 374]
const EXPECTED_ARGMAX = 12095

const MIME = {
  '.html': 'text/html',
  '.js': 'text/javascript',
  '.mjs': 'text/javascript',
  '.gguf': 'application/octet-stream'
}

const WEBGPU_ARGS = [
  '--no-sandbox',
  '--enable-unsafe-webgpu',
  '--enable-features=Vulkan,DefaultANGLEVulkan,VulkanFromANGLE',
  '--disable-gpu-sandbox',
  '--ignore-gpu-blocklist',
  '--disable-software-rasterizer',
  '--use-angle=vulkan'
]

function serveStatic(filePath, res) {
  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404)
      res.end('not found')
      return
    }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(filePath)] || 'application/octet-stream' })
    res.end(data)
  })
}

const pageHtml = `<!doctype html>
<html>
<head><meta charset="utf-8"><title>polygrad qwen3 webgpu</title></head>
<body>
<pre id="log"></pre>
<script src="/dist/polygrad.sync.js"></script>
<script>
globalThis.__polygradDebugLevel = Number(new URLSearchParams(location.search).get('debug') || 0) || 0
const log = document.getElementById('log')
const origLog = console.log
console.log = (...args) => {
  const line = args.join(' ')
  log.textContent += line + '\\n'
  origLog.apply(console, args)
}

const EXPECTED_TOKENS = ${JSON.stringify(EXPECTED_TOKENS)}
const EXPECTED_ARGMAX = ${EXPECTED_ARGMAX}

async function main() {
  console.log('creating webgpu runtime')
  const pg = polygrad.create({ core: 'wasm', device: 'webgpu' })
  console.log('runtime', pg.core, pg.device)

  console.log('fetching gguf')
  const res = await fetch('/model.gguf')
  if (!res.ok) throw new Error('GGUF fetch failed: ' + res.status)
  const gguf = new Uint8Array(await res.arrayBuffer())
  console.log('gguf bytes', gguf.length)

  console.log('loading qwen3')
  const loadStart = performance.now()
  const inst = pg.Instance.fromGGUF(gguf, { maxBatch: 1, maxSeqLen: 25 })
  console.log('loaded params', inst.paramCount, 'seconds', ((performance.now() - loadStart) / 1000).toFixed(2))

  const tok = pg.Tokenizer.fromGGUF(gguf)
  const ids = tok.encode('The capital of France is')
  console.log('tokens', Array.from(ids).join(','))
  if (ids.length !== EXPECTED_TOKENS.length ||
      EXPECTED_TOKENS.some((v, i) => ids[i] !== v)) {
    throw new Error('tokenizer ids mismatch: ' + Array.from(ids).join(','))
  }

  const x = new Float32Array(25)
  for (let i = 0; i < ids.length; i++) x[i] = ids[i]

  console.log('forward start')
  const forwardStart = performance.now()
  const outputs = await inst.forward({ x })
  const logits = outputs.output
  if (!(logits instanceof Float32Array)) throw new Error('missing Float32Array output')
  console.log('forward seconds', ((performance.now() - forwardStart) / 1000).toFixed(2), 'output', logits.length)

  const V = 151936
  const pos = ids.length - 1
  const base = pos * V
  let best = 0
  let bestVal = -Infinity
  let nNan = 0
  let nInf = 0
  for (let i = 0; i < V; i++) {
    const v = logits[base + i]
    if (Number.isNaN(v)) nNan++
    if (!Number.isFinite(v) && !Number.isNaN(v)) nInf++
    if (v > bestVal) { bestVal = v; best = i }
  }
  console.log('finite nan', nNan, 'inf', nInf)
  console.log('argmax', best, 'value', bestVal)
  if (nNan !== 0 || nInf !== 0) throw new Error('non-finite logits: nan=' + nNan + ' inf=' + nInf)
  if (best !== EXPECTED_ARGMAX) throw new Error('unexpected argmax ' + best + ', expected ' + EXPECTED_ARGMAX)

  tok.free()
  inst.free()
  await pg.dispose()
  window.__qwenResults = { ok: true, argmax: best, nNan, nInf }
}

main().catch(e => {
  console.log('FATAL', e.message, e.stack || '')
  window.__qwenResults = { ok: false, error: e.message }
})
</script>
</body>
</html>`

async function launchBrowser(chromium) {
  const executablePath = (() => {
    try { fs.accessSync('/usr/bin/google-chrome'); return '/usr/bin/google-chrome' } catch (e) { return undefined }
  })()
  return chromium.launch({
    executablePath,
    headless: false,
    ignoreDefaultArgs: ['--enable-unsafe-swiftshader'],
    args: WEBGPU_ARGS
  })
}

async function main() {
  if (!ggufPath) throw new Error('Set POLY_QWEN3_GGUF=/path/to/Qwen3-0.6B-Q8_0.gguf')
  if (!fs.existsSync(ggufPath)) throw new Error('GGUF not found: ' + ggufPath)
  const st = fs.statSync(ggufPath)
  if (st.size !== EXPECTED_LEN) {
    throw new Error('unexpected GGUF size ' + st.size + ', expected ' + EXPECTED_LEN)
  }

  const server = http.createServer((req, res) => {
    const url = (req.url || '/').split('?')[0]
    if (url === '/' || url === '/index.html') {
      res.writeHead(200, { 'Content-Type': 'text/html' })
      res.end(pageHtml)
      return
    }
    if (url === '/model.gguf') {
      res.writeHead(200, {
        'Content-Type': 'application/octet-stream',
        'Content-Length': st.size
      })
      fs.createReadStream(ggufPath).pipe(res)
      return
    }
    const filePath = path.join(jsDir, url)
    if (!filePath.startsWith(jsDir)) {
      res.writeHead(403)
      res.end('forbidden')
      return
    }
    serveStatic(filePath, res)
  })

  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
  const port = server.address().port
  console.log('server', 'http://127.0.0.1:' + port + '/')
  console.log('gguf', ggufPath, st.size)

  const { chromium } = require('playwright')
  const browser = await launchBrowser(chromium)

  try {
    const page = await browser.newPage()
    page.on('console', msg => {
      const text = msg.text()
      if (msg.type() === 'error') console.error(text)
      else console.log(text)
    })
    page.on('pageerror', err => console.error('PAGE ERROR:', err.message))

    const debug = process.env.POLY_DEBUG || process.env.DEBUG || ''
    await page.goto('http://127.0.0.1:' + port + '/' + (debug ? '?debug=' + encodeURIComponent(debug) : ''))

    const probe = await page.evaluate(async () => {
      if (!navigator.gpu) return { ok: false, reason: 'navigator.gpu missing' }
      const adapter = await navigator.gpu.requestAdapter()
      if (!adapter) return { ok: false, reason: 'requestAdapter returned null' }
      return { ok: true, features: [...adapter.features] }
    })
    console.log('webgpu probe', JSON.stringify(probe))
    if (!probe.ok) throw new Error('WebGPU unavailable: ' + probe.reason)

    const result = await page.waitForFunction(() => window.__qwenResults, { timeout: 900000 })
      .then(h => h.jsonValue())
    console.log('result', JSON.stringify(result))
    if (!result.ok) throw new Error(result.error || 'qwen probe failed')
  } finally {
    await browser.close().catch(() => {})
    server.close()
  }
}

main().catch(e => {
  console.error(e)
  process.exit(1)
})
