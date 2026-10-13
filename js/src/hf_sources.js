'use strict'

// I/O only: C selects tensors and owns names, layout, dtype and tied weights.
function bytes(value) {
  if (value instanceof Uint8Array) return value
  if (value instanceof ArrayBuffer) return new Uint8Array(value)
  throw new TypeError('checkpoint range must be a Uint8Array or ArrayBuffer')
}

async function sourceReader(source, signal) {
  if (typeof source === 'string' || (typeof URL !== 'undefined' && source instanceof URL)) {
    let size, tag
    const read = async (offset, length) => {
      if (!length) return new Uint8Array(0)
      const headers = { Range: `bytes=${offset}-${offset + length - 1}` }
      if (tag) headers['If-Range'] = tag
      const response = await fetch(source, { headers, signal })
      const range = /^bytes (\d+)-(\d+)\/(\d+)$/.exec(response.headers.get('Content-Range') || '')
      const currentTag = response.headers.get('ETag')
      if (response.status !== 206 || !range || Number(range[1]) !== offset ||
          Number(range[2]) !== offset + length - 1 || !Number.isSafeInteger(Number(range[3])) ||
          (size != null && size !== Number(range[3])) || (tag && currentTag !== tag)) {
        if (response.body) await response.body.cancel()
        throw new Error('checkpoint server must return a consistent 206 Content-Range (CORS-exposed for cross-origin URLs)')
      }
      size = Number(range[3])
      if (currentTag && !currentTag.startsWith('W/')) tag = currentTag
      // Bound a dishonest or malformed response before it becomes a whole-file allocation.
      const out = new Uint8Array(length), reader = response.body.getReader()
      let used = 0
      try {
        for (;;) {
          const { value, done } = await reader.read()
          if (done) break
          if (value.length > length - used) throw new Error('oversized checkpoint range')
          out.set(value, used); used += value.length
        }
        if (used !== length) throw new Error('short checkpoint range')
        return out
      } finally { await reader.cancel(); reader.releaseLock() }
    }
    const prefix = await read(0, 8)
    return { size, read: (offset, length) => offset === 0 && length === 8 ? prefix : read(offset, length) }
  }
  if (typeof Blob !== 'undefined' && source instanceof Blob)
    return { size: source.size, read: async (offset, length) => new Uint8Array(await source.slice(offset, offset + length).arrayBuffer()) }
  if (source instanceof Uint8Array || source instanceof ArrayBuffer) {
    const data = bytes(source)
    return { size: data.length, read: (offset, length) => data.subarray(offset, offset + length) }
  }
  if (source && typeof source.read === 'function') return source
  throw new TypeError('checkpoint source must be a URL, Blob, bytes, or {size, read(offset, length)}')
}

async function importSources(api, config, sources, opts) {
  const check = ok => {
    if (!ok) throw new Error('polygrad: HF import failed: ' + (api.importLastError()?.message || 'unknown error'))
  }
  const read = async (source, offset, length) => {
    if (opts.signal) opts.signal.throwIfAborted()
    if (!Number.isSafeInteger(offset) || !Number.isSafeInteger(length) || offset < 0 || length < 0 || length > source.size - offset)
      throw new Error('invalid checkpoint range')
    const data = bytes(await source.read(offset, length))
    if (opts.signal) opts.signal.throwIfAborted()
    if (data.length !== length) throw new Error('short or oversized checkpoint range')
    return data
  }
  const session = await api.hfImportNew(config, opts.maxBatch ?? 1, opts.maxSeqLen ?? 0)
  check(session)
  try {
    const readers = []
    for (const source of sources) {
      const reader = await sourceReader(source, opts.signal)
      if (!Number.isSafeInteger(reader.size) || reader.size < 8) throw new Error('invalid checkpoint file size')
      readers.push(reader)
      const prefix = await read(reader, 0, 8)
      const length = Number(new DataView(prefix.buffer, prefix.byteOffset, 8).getBigUint64(0, true))
      if (length > 100000000 || length > reader.size - 8) throw new Error('invalid safetensors header size')
      const header = new Uint8Array(8 + length)
      header.set(prefix); header.set(await read(reader, 8, length), 8)
      check(api.hfImportAddFile(session, header, reader.size) === 0)
    }
    check(await api.hfImportBuild(session) === 0)
    for (;;) {
      const request = api.hfImportRequest(session)
      if (!request) break
      const { file, offset, length } = JSON.parse(request)
      check(await api.hfImportFeed(session, await read(readers[file], offset, length)) === 0)
    }
    const model = api.hfImportFinish(session)
    check(model)
    return model
  } finally { api.hfImportFree(session) }
}

module.exports = { importSources }
