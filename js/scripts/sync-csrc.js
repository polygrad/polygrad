'use strict'

const fs = require('fs')
const path = require('path')

const repoRoot = path.resolve(__dirname, '..', '..')
const dstDir = path.resolve(__dirname, '..', 'csrc')

// JS native builds do not consume the browser-only render/WASM sources.
const EXCLUDE = new Set([
  'wasm.c',
  'wasm_builder.c',
  'llama3.c',
  'resnet.c',
  'vit.c'
])

if (!fs.existsSync(path.join(repoRoot, 'src'))) {
  console.log('sync-csrc: src/ not found (published package), skipping')
  process.exit(0)
}

if (fs.existsSync(dstDir)) fs.rmSync(dstDir, { recursive: true })
fs.mkdirSync(dstDir, { recursive: true })

let copied = 0
function copyTree(relDir) {
  const srcDir = path.join(repoRoot, relDir)
  if (!fs.existsSync(srcDir)) return
  for (const entry of fs.readdirSync(srcDir, { withFileTypes: true })) {
    const relPath = path.join(relDir, entry.name)
    const srcPath = path.join(repoRoot, relPath)
    const dstPath = path.join(dstDir, relPath)
    if (entry.isDirectory()) {
      fs.mkdirSync(dstPath, { recursive: true })
      copyTree(relPath)
      continue
    }
    if (!(entry.name.endsWith('.c') || entry.name.endsWith('.h'))) continue
    if (EXCLUDE.has(entry.name)) continue
    fs.mkdirSync(path.dirname(dstPath), { recursive: true })
    fs.copyFileSync(srcPath, dstPath)
    copied++
  }
}

copyTree('src')
copyTree(path.join('vendor', 'cjson'))

console.log(`sync-csrc: copied ${copied} files to csrc/`)
