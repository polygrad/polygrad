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
  if (url === '/favicon.ico') {
    res.writeHead(204)
    res.end()
    return
  }
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

function executableExists(file) {
  try { fs.accessSync(file, fs.constants.X_OK); return true }
  catch (e) { return false }
}

function parseBrowserSpec(raw) {
  let label = raw
  let body = raw
  const eq = raw.indexOf('=')
  if (eq >= 0) {
    label = raw.slice(0, eq)
    body = raw.slice(eq + 1)
  }

  let name = body
  let target = ''
  const at = body.indexOf('@')
  if (at >= 0) {
    name = body.slice(0, at)
    target = body.slice(at + 1)
  }

  let launcherName = name
  let channel = ''
  let executablePath = ''
  if (name === 'chrome' || name === 'chrome-stable') {
    launcherName = 'chromium'
    channel = 'chrome'
  } else if (name.startsWith('chrome-')) {
    launcherName = 'chromium'
    channel = name.slice('chrome-'.length)
  }

  if (target) {
    if (target.startsWith('/')) executablePath = target
    else channel = target
  }

  if (!['chromium', 'firefox', 'webkit'].includes(launcherName)) {
    throw new Error(`unknown browser spec '${raw}'`)
  }
  return { raw, label, launcherName, channel, executablePath }
}

function parseBrowserSpecs() {
  return (process.env.POLY_BROWSER_BROWSERS || process.env.POLY_BROWSER_ENGINES || 'chromium')
    .split(',')
    .map(s => s.trim())
    .filter(Boolean)
    .map(parseBrowserSpec)
}

function launchOptionsFor(spec, device) {
  const opts = {}
  if (spec.channel) opts.channel = spec.channel
  if (spec.executablePath) opts.executablePath = spec.executablePath

  if (device === 'webgpu') {
    if (spec.launcherName !== 'chromium') {
      return { skip: `WebGPU tests currently require a Chromium-family browser` }
    }
    // WebGPU needs the full Chrome binary (not headless-shell) with GPU access.
    // Use an explicitly requested executable/channel first, then system Chrome.
    if (!opts.executablePath && !opts.channel && executableExists('/usr/bin/google-chrome')) {
      opts.executablePath = '/usr/bin/google-chrome'
    }
    opts.headless = false
    opts.args = WEBGPU_ARGS
  }
  return { opts }
}

async function launchForDevice(playwright, spec, device) {
  if (spec.executablePath && !executableExists(spec.executablePath)) {
    return { skip: `executable not found: ${spec.executablePath}` }
  }
  const { skip, opts } = launchOptionsFor(spec, device)
  if (skip) return { skip }
  const launcher = playwright[spec.launcherName]
  const browser = await launcher.launch(opts)
  return { browser }
}

async function runForDevice(browser, port, device, spec) {
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

  console.log(`=== browser: ${spec.label}, engine: ${spec.launcherName}, device: ${device} ===`)
  console.log(`[${spec.label}] version: ${browser.version()}`)

  await page.goto(url)

  const timeout = device === 'webgpu' ? 120000 : 60000
  const results = await page.waitForFunction(
    () => window.__testResults,
    { timeout }
  ).then(h => h.jsonValue())

  // Keep the probe out of the test hot path. With lazy WebGPU initialization,
  // concurrently requesting an adapter while Asyncify-backed readback is
  // starting can invalidate Chrome's external WebGPU object handles.
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

  const playwright = require('playwright')

  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
  const port = server.address().port
  console.log(`Server listening on http://127.0.0.1:${port}/`)

  const devices = (process.env.POLY_BROWSER_DEVICES || 'auto,interp,webgpu')
    .split(',')
    .map(s => s.trim())
    .filter(Boolean)
  const browserSpecs = parseBrowserSpecs()
  const skipUnavailable = process.env.POLY_BROWSER_SKIP_UNAVAILABLE === '1'
  let totalFailed = 0

  for (const spec of browserSpecs) {
    for (const device of devices) {
      let launched
      try {
        launched = await launchForDevice(playwright, spec, device)
      } catch (e) {
        if (skipUnavailable) {
          console.log(`\n[${spec.label}/${device}] skipped: ${e.message}`)
          continue
        }
        throw e
      }
      if (launched.skip) {
        console.log(`\n[${spec.label}/${device}] skipped: ${launched.skip}`)
        continue
      }

      const browser = launched.browser
      const r = await runForDevice(browser, port, device, spec)
      await browser.close()

      if (r.error) {
        console.error(`\n[${spec.label}/${device}] ERROR: ${r.error}`)
        totalFailed++
      } else {
        console.log(
          `\n[${spec.label}/${device}] ${r.passed} passed, ${r.failed} failed` +
          (r.skipped ? `, ${r.skipped} skipped` : '')
        )
        totalFailed += r.failed
      }
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
