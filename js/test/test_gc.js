'use strict'

const assert = require('assert')
const polygrad = require('..')

async function collectUntil(pg, predicate) {
  for (let i = 0; i < 100; i++) {
    global.gc()
    await new Promise(resolve => setImmediate(resolve))
    const stats = pg.stats().coreStats
    if (predicate(stats)) return stats
  }
  return pg.stats().coreStats
}

async function main() {
  if (typeof global.gc !== 'function') throw new Error('test_gc.js requires --expose-gc')
  const core = process.argv[2] || 'native'
  const pg = await polygrad.create({ core })
  try {
    const before = pg.stats().coreStats
    ;(() => { pg.Tensor.empty([8], { dtype: 'float32' }) })()
    let stats = await collectUntil(pg, s => s.tensorRecords === before.tensorRecords)
    assert.strictEqual(stats.tensorRecords, before.tensorRecords,
      'Tensor finalizer did not retire its core owner')

    ;(() => {
      const tensor = pg.Tensor.empty([1024], { dtype: 'float32' })
      tensor.copyFrom(new Float32Array(1024))
      void tensor.uop
    })()
    stats = await collectUntil(pg, s =>
      s.tensorRecords === before.tensorRecords && s.memUsed === before.memUsed)
    assert.strictEqual(stats.tensorRecords, before.tensorRecords,
      'Tensor finalizer left a core owner')
    assert.strictEqual(stats.memUsed, before.memUsed,
      'raw UOp finalizer left physical residency')

    await (async () => {
      const copyKernel = (out, src) => {
        out = out.flatten(); src = src.flatten()
        const i = pg.uop.range(out.numel(), 0)
        return out.index(i).store(src.index(i)).end(i).sink({
          arg: new pg.uop.KernelInfo('gc_custom_copy')
        })
      }
      const out = pg.Tensor.empty([1 << 20], { dtype: 'float32' })
      const src = new pg.Tensor(new Float32Array(1 << 20))
      const result = out.customKernel(src, {
        fxn: copyKernel,
        gradFxn: grad => [null, grad]
      })[0]
      await result.realize()
      assert(pg.stats().coreStats.memUsed >= before.memUsed + (2 << 22),
        'customKernel residency probe did not allocate both buffers')
    })()
    stats = await collectUntil(pg, s => s.memUsed === before.memUsed)
    assert.strictEqual(stats.memUsed, before.memUsed,
      'customKernel gradient metadata retained physical residency')

    const x = pg.Tensor.empty([2], { dtype: 'float32' })
    const uop = x.uop
    const inst = await pg.Instance.fromTensors({
      inputs: { x }, outputs: { output: x.add(1) }
    })
    await pg.dispose()
    assert.strictEqual(inst._handle, null,
      'Runtime disposal left a borrowed Instance handle alive')
    assert.throws(() => uop.op, /disposed/,
      'UOp read through a destroyed runtime arena')
    assert.throws(() => pg.Tensor.empty([1]), /disposed/,
      'bound Tensor factory used a destroyed runtime context')
    console.log(`[PASS] ${core} Tensor/UOp finalizers retire core owners`)
  } finally {
    await pg.dispose()
  }
}

main().catch(err => { console.error(err); process.exit(1) })
