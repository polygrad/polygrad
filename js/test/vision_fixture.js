'use strict'

async function verify(bytes, expected) {
  const digest = new Uint8Array(await globalThis.crypto.subtle.digest('SHA-256', bytes))
  const actual = Array.from(digest, x => x.toString(16).padStart(2, '0')).join('')
  if (actual !== expected) throw new Error('vision payload formula drift')
}

async function expandVisionCase(item) {
  const header = new TextEncoder().encode(item.weights.header)
  const entries = Object.entries(JSON.parse(item.weights.header)).filter(([k]) => k !== '__metadata__')
  const size = Math.max(...entries.map(([,v]) => v.data_offsets[1]))
  const bytes = new Uint8Array(8 + header.length + size), view = new DataView(bytes.buffer)
  view.setBigUint64(0, BigInt(header.length), true)
  bytes.set(header, 8)
  for (const [name, entry] of entries) {
    if (entry.dtype !== 'F32') throw new Error('vision fixture requires float32')
    const [start, end] = entry.data_offsets
    const seed = Array.from(new TextEncoder().encode(name)).reduce((a,b) => a+b, 0)
    for (let i = 0; i < (end - start) / 4; i++) {
      let value = Math.fround(((i * 7 + seed) % 29 - 14) * Math.fround(.013))
      if ((name.includes('norm') && name.endsWith('weight')) || name.endsWith('lambda1')) value = Math.fround(value + 1)
      if (name === 'logit_scale') value = Math.fround(1.3)
      view.setFloat32(8 + header.length + start + i * 4, value, true)
    }
  }
  await verify(bytes, item.weights.sha256)
  const spec = item.inputs.pixel_values, count = spec.shape.reduce((a,b) => a*b, 1)
  let pixels = Array.from({length: count}, (_,i) => Math.fround(Math.fround((i * 11 % 101) / 50) - 1))
  const raw = new Uint8Array(count * 4), pixelView = new DataView(raw.buffer)
  pixels.forEach((v,i) => pixelView.setFloat32(i*4, v, true))
  await verify(raw, spec.sha256)
  for (const width of spec.shape.slice(1).reverse()) {
    const rows = []
    for (let i = 0; i < pixels.length; i += width) rows.push(pixels.slice(i,i+width))
    pixels = rows
  }
  return {...item, weights:bytes, inputs:{...item.inputs, pixel_values:pixels}}
}

module.exports = {expandVisionCase}
