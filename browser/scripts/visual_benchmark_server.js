'use strict'

const fs = require('fs')
const http = require('http')
const path = require('path')

const projectRoot = path.resolve(__dirname, '..', '..')
const benchmarkPath = '/browser/examples/visual_comparison.html'
const host = process.env.HOST || '127.0.0.1'
const port = Number(process.env.PORT || 8080)

function ensureArtifacts() {
  const required = [
    path.join(projectRoot, 'build', 'polygrad.js'),
    path.join(projectRoot, 'build', 'polygrad.wasm')
  ]
  for (const artifact of required) {
    if (!fs.existsSync(artifact)) {
      const rel = path.relative(projectRoot, artifact)
      console.error(`Missing ${rel}.`)
      console.error('Run `make wasm` from the polygrad project root, then retry.')
      process.exit(1)
    }
  }
}

function contentType(filePath) {
  const ext = path.extname(filePath).toLowerCase()
  if (ext === '.html') return 'text/html; charset=utf-8'
  if (ext === '.js') return 'application/javascript; charset=utf-8'
  if (ext === '.css') return 'text/css; charset=utf-8'
  if (ext === '.json') return 'application/json; charset=utf-8'
  if (ext === '.wasm') return 'application/wasm'
  if (ext === '.map') return 'application/json; charset=utf-8'
  return 'application/octet-stream'
}

function resolveRequestPath(reqUrl) {
  let pathname
  try {
    pathname = decodeURIComponent((reqUrl || '/').split('?')[0])
  } catch (e) {
    return null
  }
  if (pathname === '/') pathname = benchmarkPath

  const absolutePath = path.resolve(projectRoot, `.${pathname}`)
  const rel = path.relative(projectRoot, absolutePath)
  if (rel.startsWith('..') || path.isAbsolute(rel)) return null
  return absolutePath
}

function sendNotFound(res) {
  res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' })
  res.end('Not found')
}

ensureArtifacts()

const server = http.createServer((req, res) => {
  const requestedPath = resolveRequestPath(req.url)
  if (!requestedPath) {
    res.writeHead(403, { 'Content-Type': 'text/plain; charset=utf-8' })
    res.end('Forbidden')
    return
  }

  fs.stat(requestedPath, (statErr, stats) => {
    if (statErr) {
      sendNotFound(res)
      return
    }

    const filePath = stats.isDirectory()
      ? path.join(requestedPath, 'index.html')
      : requestedPath

    fs.readFile(filePath, (readErr, data) => {
      if (readErr) {
        sendNotFound(res)
        return
      }
      res.writeHead(200, {
        'Content-Type': contentType(filePath),
        'Cache-Control': 'no-cache'
      })
      res.end(data)
    })
  })
})

server.listen(port, host, () => {
  console.log(`Serving ${projectRoot}`)
  console.log(`Visual benchmark: http://${host}:${port}${benchmarkPath}`)
  console.log('Open the URL in your browser, then click "Run All".')
  console.log('Press Ctrl+C to stop the server.')
})

server.on('error', (err) => {
  console.error('Server error:', err.message)
  process.exit(1)
})
