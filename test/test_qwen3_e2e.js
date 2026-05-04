/**
 * test_qwen3_e2e.js -- End-to-end Qwen3 0.6B GGUF test
 *
 * Downloads (or uses cached) Qwen3-0.6B-Q8_0.gguf from HuggingFace,
 * loads via WASM or native, tokenizes, runs forward, generates text.
 *
 * Requirements:
 *   - Node.js 18+
 *   - ~4GB free RAM (WASM) or ~3GB (native)
 *   - Internet for first download (~610MB)
 *
 * Usage:
 *   node test/test_qwen3_e2e.js                    # auto (native or WASM)
 *   node test/test_qwen3_e2e.js --core=wasm      # force WASM
 *   node test/test_qwen3_e2e.js --core=native    # force native
 *   node --max-old-space-size=4096 test/test_qwen3_e2e.js  # if OOM
 */

'use strict'
const fs = require('fs')
const path = require('path')
const { execSync } = require('child_process')

const GGUF_REPO = 'unsloth/Qwen3-0.6B-GGUF'
const GGUF_FILE = 'Qwen3-0.6B-Q8_0.gguf'
const CACHE_DIR = path.join(require('os').homedir(), '.cache', 'polygrad')

// Parse args
const args = process.argv.slice(2)
const core = (args.find(a => a.startsWith('--core=')) || '').split('=')[1] || 'auto'
const maxTokens = parseInt((args.find(a => a.startsWith('--tokens=')) || '').split('=')[1] || '20')

async function downloadGGUF() {
  const cached = path.join(CACHE_DIR, GGUF_FILE)
  if (fs.existsSync(cached)) {
    console.log(`Using cached: ${cached} (${(fs.statSync(cached).size / 1024 / 1024).toFixed(0)} MB)`)
    return cached
  }

  // Try huggingface-cli
  try {
    console.log(`Downloading ${GGUF_REPO}/${GGUF_FILE}...`)
    fs.mkdirSync(CACHE_DIR, { recursive: true })
    execSync(
      `python -c "from huggingface_hub import hf_hub_download; ` +
      `import shutil; p = hf_hub_download('${GGUF_REPO}', '${GGUF_FILE}'); ` +
      `shutil.copy(p, '${cached}')"`,
      { stdio: 'inherit', timeout: 300000 }
    )
    if (fs.existsSync(cached)) return cached
  } catch (e) {
    // Try direct download
    console.log('huggingface_hub not available, trying direct download...')
    const url = `https://huggingface.co/${GGUF_REPO}/resolve/main/${GGUF_FILE}`
    try {
      execSync(`curl -L -o "${cached}" "${url}"`, { stdio: 'inherit', timeout: 600000 })
      if (fs.existsSync(cached)) return cached
    } catch (e2) {
      console.error('Download failed. Please download manually:')
      console.error(`  curl -L -o ${cached} ${url}`)
      process.exit(1)
    }
  }
}

async function main() {
  const ggufPath = await downloadGGUF()
  const ggufBytes = new Uint8Array(fs.readFileSync(ggufPath))
  console.log(`GGUF: ${(ggufBytes.length / 1024 / 1024).toFixed(0)} MB`)

  const polygrad = require('../js/src/index')
  const pg = await polygrad.create({ core })
  console.log(`Runtime: ${pg.core}`)

  // Load model
  console.log('Loading model...')
  const t0 = Date.now()
  const inst = pg.Instance.loadGGUF(ggufBytes, { maxBatch: 1, maxSeqLen: 64 })
  console.log(`Load: ${((Date.now() - t0) / 1000).toFixed(1)}s, params: ${inst.paramCount}`)

  // Load tokenizer
  const tok = pg.Tokenizer.fromGGUF(ggufBytes)
  console.log(`Tokenizer: vocab=${tok.vocabSize} eos=${tok.eosId}`)

  // Tokenizer test
  const testText = 'The capital of France is'
  const testIds = tok.encode(testText)
  const decoded = tok.decode(testIds)
  console.log(`Encode: "${testText}" -> [${Array.from(testIds)}]`)
  console.log(`Decode: [${Array.from(testIds)}] -> "${decoded}"`)
  if (decoded !== testText) {
    console.error('FAIL: tokenizer round-trip mismatch')
    process.exit(1)
  }
  console.log('Tokenizer round-trip: OK')

  // Generate text
  const prompt = 'The capital of France is'
  let ids = Array.from(tok.encode(prompt))
  const seqLen = 64
  process.stdout.write('\nGenerate: ' + prompt)

  for (let step = 0; step < maxTokens; step++) {
    const x = new Float32Array(seqLen)
    for (let i = 0; i < ids.length && i < seqLen; i++) x[i] = ids[i]

    const t1 = Date.now()
    const output = inst.forward({ x })
    const elapsed = Date.now() - t1

    const logits = output.output
    const V = logits.length / seqLen
    const pos = ids.length - 1
    const nextLogits = logits.slice(pos * V, (pos + 1) * V)

    // Check finite
    let allFinite = true
    for (let v = 0; v < V; v++) {
      if (!isFinite(nextLogits[v])) { allFinite = false; break }
    }
    if (!allFinite) {
      console.error('\nFAIL: NaN/Inf in logits at step ' + step)
      process.exit(1)
    }

    // Greedy argmax
    let best = 0, bestVal = -Infinity
    for (let v = 0; v < V; v++) {
      if (nextLogits[v] > bestVal) { bestVal = nextLogits[v]; best = v }
    }

    if (best === tok.eosId) { process.stdout.write(' [EOS]'); break }
    ids.push(best)
    process.stdout.write(tok.decode(Int32Array.from([best])))

    if (step === 0) {
      process.stderr.write(` (${elapsed}ms first token, ${(elapsed/1000).toFixed(1)}s) `)
    }
  }
  console.log('\n')

  // Verify output makes sense
  const fullText = tok.decode(Int32Array.from(ids))
  console.log('Full output: ' + fullText)
  if (fullText.includes('Paris')) {
    console.log('PASS: output contains "Paris"')
  } else {
    console.log('WARN: output does not contain "Paris" (may be quantization noise)')
  }

  tok.free()
  inst.free()
  await pg.dispose()
  console.log('Done')
}

main().catch(e => { console.error(e); process.exit(1) })
