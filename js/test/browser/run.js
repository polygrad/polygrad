'use strict'

const http = require('http')
const fs = require('fs')
const path = require('path')

const MIME = {
  '.html': 'text/html',
  '.js': 'text/javascript',
  '.mjs': 'text/javascript',
  '.wasm': 'application/wasm',
}

const jsDir = path.resolve(__dirname, '..', '..')

const server = http.createServer((req, res) => {
  const url = (req.url || '/').split('?')[0]
  const filePath = path.join(jsDir, url === '/' ? '/test/browser/index.html' : url)

  if (!filePath.startsWith(jsDir)) {
    res.writeHead(403)
    res.end('forbidden')
    return
  }

  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404)
      res.end('not found: ' + url)
      return
    }
    const ext = path.extname(filePath)
    res.writeHead(200, { 'Content-Type': MIME[ext] || 'application/octet-stream' })
    res.end(data)
  })
})

// WebGPU requires: secure context (localhost), real GPU, --enable-unsafe-webgpu.
// Headed mode under a real display (DISPLAY=:1) or xvfb is needed because
// Playwright's headless-shell binary doesn't expose navigator.gpu.
const WEBGPU_ARGS = [
  '--no-sandbox',
  '--enable-unsafe-webgpu',
  '--enable-features=Vulkan',
  '--disable-gpu-sandbox',
]

async function launchForDevice(chromium, device) {
  if (device !== 'webgpu') return chromium.launch()
  // WebGPU needs the full Chrome binary (not headless-shell) with GPU access.
  // Use system Chrome if available, fall back to Playwright Chromium.
  const executablePath = (() => {
    try { require('fs').accessSync('/usr/bin/google-chrome'); return '/usr/bin/google-chrome' }
    catch (e) { return undefined }
  })()
  return chromium.launch({
    executablePath,
    headless: false,
    args: WEBGPU_ARGS
  })
}

async function runForDevice(browser, port, device) {
  const debugLevel = process.env.POLY_DEBUG || process.env.DEBUG || ''
  const testFilter = process.env.POLY_TEST_FILTER || ''
  const url = `http://127.0.0.1:${port}/?device=${device}` +
    (debugLevel ? `&debug=${encodeURIComponent(debugLevel)}` : '') +
    (testFilter ? `&filter=${encodeURIComponent(testFilter)}` : '')
  const page = await browser.newPage()

  page.on('console', msg => {
    if (msg.type() === 'error') console.error(msg.text())
    else if (msg.type() === 'log' || msg.type() === 'debug' || msg.type() === 'warning')
      console.log(msg.text())
  })
  page.on('pageerror', err => console.error('PAGE ERROR:', err.message))

  await page.goto(url)

  // WebGPU probe: log adapter info before running tests
  if (device === 'webgpu') {
    const probe = await page.evaluate(async () => {
      if (!navigator.gpu) return { ok: false, reason: 'navigator.gpu missing' }
      const adapter = await navigator.gpu.requestAdapter()
      if (!adapter) return { ok: false, reason: 'requestAdapter returned null' }
      return { ok: true, features: [...adapter.features] }
    })
    if (probe.ok) {
      console.log(`[webgpu] adapter OK, features: ${probe.features.join(', ')}`)
    } else {
      console.error(`[webgpu] probe: ${probe.reason}`)
    }
  }

  const timeout = device === 'webgpu' ? 120000 : 60000
  const results = await page.waitForFunction(
    () => window.__testResults,
    { timeout }
  ).then(h => h.jsonValue())

  await page.close()
  return results
}

async function main() {
  // Build browser test bundle from test_tensor.js + test_instance.js
  const { execSync } = require('child_process')
  execSync('npx esbuild test/browser/test_browser_entry.js --bundle --format=iife --platform=browser --outfile=test/browser/tests.js', {
    cwd: path.resolve(__dirname, '..', '..'),
    stdio: 'inherit'
  })

  const { chromium } = require('playwright')

  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
  const port = server.address().port
  console.log(`Server listening on http://127.0.0.1:${port}/`)

  const devices = (process.env.POLY_BROWSER_DEVICES || 'auto,interp,webgpu')
    .split(',')
    .map(s => s.trim())
    .filter(Boolean)
  let totalFailed = 0

  for (const device of devices) {
    // Launch separate browser per device (WebGPU needs special flags)
    const browser = await launchForDevice(chromium, device)
    const r = await runForDevice(browser, port, device)
    await browser.close()

    if (r.error) {
      console.error(`\n[${device}] ERROR: ${r.error}`)
      totalFailed++
    } else {
      console.log(
        `\n[${device}] ${r.passed} passed, ${r.failed} failed` +
        (r.skipped ? `, ${r.skipped} skipped` : '')
      )
      totalFailed += r.failed
    }
  }

  server.close()

  console.log(`\nBrowser total: ${totalFailed === 0 ? 'all passed' : totalFailed + ' failed'}`)
  process.exit(totalFailed > 0 ? 1 : 0)
}

main().catch(e => {
  console.error(e)
  server.close()
  process.exit(1)
})
