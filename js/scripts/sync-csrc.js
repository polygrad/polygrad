'use strict'

const fs = require('fs')
const path = require('path')

const srcDir = path.resolve(__dirname, '..', '..', 'src')
const dstDir = path.resolve(__dirname, '..', 'csrc')

// Exclude renderers/runtimes not needed for N-API addon
const EXCLUDE = new Set([
  'render_wasm.c', 'wasm_builder.c',
  'render_cuda.c', 'runtime_cuda.c',
  'render_wgsl.c'
])

if (!fs.existsSync(srcDir)) {
  console.log('sync-csrc: src/ not found (published package), skipping')
  process.exit(0)
}

if (fs.existsSync(dstDir)) {
  fs.rmSync(dstDir, { recursive: true })
}
fs.mkdirSync(dstDir, { recursive: true })

// Also copy modelzoo subdirectories
const modelzooSrc = path.join(srcDir, 'modelzoo')
const modelzooModels = path.join(modelzooSrc, 'models')

let copied = 0
for (const f of fs.readdirSync(srcDir)) {
  if (EXCLUDE.has(f)) continue
  if (f.endsWith('.c') || f.endsWith('.h')) {
    fs.copyFileSync(path.join(srcDir, f), path.join(dstDir, f))
    copied++
  }
}

// Copy vendor cJSON
const cjsonDir = path.resolve(__dirname, '..', '..', 'vendor', 'cjson')
if (fs.existsSync(cjsonDir)) {
  for (const f of fs.readdirSync(cjsonDir)) {
    if (f.endsWith('.c') || f.endsWith('.h')) {
      fs.copyFileSync(path.join(cjsonDir, f), path.join(dstDir, f))
      copied++
    }
  }
}

// Copy modelzoo files
if (fs.existsSync(modelzooSrc)) {
  const mzDst = path.join(dstDir, 'modelzoo')
  fs.mkdirSync(mzDst, { recursive: true })
  for (const f of fs.readdirSync(modelzooSrc)) {
    if (f.endsWith('.c') || f.endsWith('.h')) {
      fs.copyFileSync(path.join(modelzooSrc, f), path.join(mzDst, f))
      copied++
    }
  }
  if (fs.existsSync(modelzooModels)) {
    const modelsDst = path.join(mzDst, 'models')
    fs.mkdirSync(modelsDst, { recursive: true })
    for (const f of fs.readdirSync(modelzooModels)) {
      if (f.endsWith('.c') || f.endsWith('.h')) {
        fs.copyFileSync(path.join(modelzooModels, f), path.join(modelsDst, f))
        copied++
      }
    }
  }
}

// Rewrite vendor includes to flat includes (cJSON.h is now in csrc/)
function rewriteIncludes(dir) {
  for (const f of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, f.name)
    if (f.isDirectory()) {
      rewriteIncludes(full)
    } else if (f.name.endsWith('.c') || f.name.endsWith('.h')) {
      let content = fs.readFileSync(full, 'utf8')
      const re = /#include\s+"(?:\.\.\/)*vendor\/cjson\/cJSON\.h"/g
      if (re.test(content)) {
        // Compute relative path from this file to csrc/cJSON.h
        const rel = path.relative(path.dirname(full), dstDir)
        const replacement = rel ? `#include "${rel}/cJSON.h"` : '#include "cJSON.h"'
        content = content.replace(re, replacement)
        fs.writeFileSync(full, content)
      }
    }
  }
}
rewriteIncludes(dstDir)

console.log(`sync-csrc: copied ${copied} files to csrc/`)
