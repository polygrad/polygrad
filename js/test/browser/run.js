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

async function runForDevice(browser, port, device) {
  const url = `http://127.0.0.1:${port}/?device=${device}`
  const page = await browser.newPage()

  page.on('console', msg => {
    if (msg.type() === 'log') console.log(msg.text())
  })
  page.on('pageerror', err => console.error('PAGE ERROR:', err.message))

  await page.goto(url)

  const results = await page.waitForFunction(
    () => window.__testResults,
    { timeout: 60000 }
  ).then(h => h.jsonValue())

  await page.close()
  return results
}

async function main() {
  const { chromium } = require('playwright')

  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
  const port = server.address().port
  console.log(`Server listening on http://127.0.0.1:${port}/`)

  const browser = await chromium.launch()
  const devices = ['auto', 'interp']
  let totalFailed = 0

  for (const device of devices) {
    const r = await runForDevice(browser, port, device)
    if (r.error) {
      console.error(`\n[${device}] ERROR: ${r.error}`)
      totalFailed++
    } else {
      console.log(`\n[${device}] ${r.passed} passed, ${r.failed} failed`)
      totalFailed += r.failed
    }
  }

  await browser.close()
  server.close()

  console.log(`\nBrowser total: ${totalFailed === 0 ? 'all passed' : totalFailed + ' failed'}`)
  process.exit(totalFailed > 0 ? 1 : 0)
}

main().catch(e => {
  console.error(e)
  server.close()
  process.exit(1)
})
