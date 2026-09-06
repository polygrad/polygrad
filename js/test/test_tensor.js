/**
 * Shared test suite for the polygrad Tensor class.
 * Target-agnostic: runs against WASM or native bindings.
 *
 * Usage: require this module and call runTensorTests(pg).
 */

'use strict'

function assertClose(arr, expected, tol) {
  if (tol === undefined) tol = 1e-4
  if (arr.length !== expected.length) {
    throw new Error(`Length mismatch: ${arr.length} vs ${expected.length}`)
  }
  for (let i = 0; i < arr.length; i++) {
    if (Number.isNaN(expected[i])) {
      if (!Number.isNaN(arr[i])) {
        throw new Error(`Mismatch at [${i}]: ${arr[i]} vs ${expected[i]}`)
      }
      continue
    }
    const diff = Math.abs(arr[i] - expected[i])
    if (!Number.isFinite(diff) || diff > tol) {
      throw new Error(`Mismatch at [${i}]: ${arr[i]} vs ${expected[i]}`)
    }
  }
}

function assert(cond, msg) {
  if (!cond) throw new Error(msg || 'assertion failed')
}

function assertShape(actual, expected) {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    throw new Error(`shape mismatch: got [${actual}], expected [${expected}]`)
  }
}

function countGraphOp(root, op) {
  const seen = new Set()
  const stack = [root]
  let count = 0
  while (stack.length) {
    const node = stack.pop()
    if (!node || seen.has(node.key)) continue
    seen.add(node.key)
    if (node.op === op) count++
    for (const src of node.src) stack.push(src)
  }
  return count
}

function countGraphNodes(root) {
  const seen = new Set()
  const stack = [root]
  while (stack.length) {
    const node = stack.pop()
    if (!node || seen.has(node.key)) continue
    seen.add(node.key)
    for (const src of node.src) stack.push(src)
  }
  return seen.size
}

async function runTensorTests(pg) {
  const Tensor = pg.Tensor
  const caps = pg.caps || {}
  const testFilter = (() => {
    if (pg && pg.testFilter) return String(pg.testFilter)
    if (typeof globalThis !== 'undefined' && globalThis.__POLY_TEST_FILTER) {
      return String(globalThis.__POLY_TEST_FILTER)
    }
    if (typeof process !== 'undefined' && process.env && process.env.POLY_TEST_FILTER) {
      return String(process.env.POLY_TEST_FILTER)
    }
    return ''
  })()
  const supportsF16 = caps.f16 !== false
  const supportsF64 = caps.f64 !== false
  let passed = 0, failed = 0, skipped = 0

  async function test(name, fn) {
    if (testFilter && !name.includes(testFilter)) return
    try {
      await fn()
      console.log(`  [PASS] ${name}`)
      passed++
    } catch (e) {
      console.log(`  [FAIL] ${name}: ${e.message}`)
      if (e && e.stack) console.log(e.stack)
      failed++
    }
  }

  async function testIf(cond, name, fn) {
    if (!cond) {
      console.log(`  [SKIP] ${name}`)
      skipped++
      return
    }
    await test(name, fn)
  }

  console.log(`Core: ${pg.core}, device: ${pg.device}\n`)

  // -- Creation --
  console.log('-- Creation --')

  await test('dtype API queries match pinned metadata', async () => {
    const bytes = {bool:1, int8:1, uint8:1, int16:2, uint16:2, int32:4, uint32:4,
      int64:8, uint64:8, float16:2, bfloat16:2, float32:4, float64:8,
      fp8e4m3:1, fp8e5m2:1, fp8e4m3fnuz:1, fp8e5m2fnuz:1, weakint:null, weakfloat:null}
    for (const [dtype, size] of Object.entries(bytes)) {
      const tensor = new Tensor(0, {dtype})
      assert(tensor.isFloatingPoint() === /^(float|bfloat|fp8|weakfloat)/.test(dtype))
      assert(pg.uop.dtype(tensor.uopPhysical) === dtype)
      if (size === null) {
        let error
        try { tensor.elementSize() } catch (e) { error=e }
        assert(error && /elementSize requires a concrete dtype/.test(error.message))
      } else assert(tensor.elementSize() === size)
    }
  })

  await test('dtype API bound UOp constants retain their owner and type', async () => {
    for (const [value, dtype] of [[true,'bool'], [1,'int32'], [1.75,'float32']]) {
      const uop = pg.uop.constant(value, dtype)
      assert(uop.ctx === pg._core.ctx)
      assert(uop.op === pg._core.ops.CONST && uop.src.length === 0)
      assert(pg.uop.dtype(uop) === dtype)
      const tensor = new Tensor(uop)
      assert(tensor.dtype === dtype)
      assertClose(await tensor.toArrayAsync(), [Number(value)])
    }
  })

  await test('constructor parity null is scalar zero', async () => {
    for (const dtype of [undefined, 'float32', 'int32', 'bool']) {
      const tensor = new Tensor(null, { dtype })
      assertShape(tensor.shape, [])
      assert(tensor.uopPhysical.op === pg._core.ops.CONST)
      assert(tensor.uopPhysical.src.length === 0)
      if (!dtype) assert(tensor.dtype === 'weakfloat')
      assertClose(await tensor.toArrayAsync(), [0])
    }
  })

  for (const dtype of ['int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']) {
    await test(`constructor parity integer storage ${dtype}`, async () => {
      const bits = Number(dtype.match(/\d+/)[0])
      const values = [-(1n << 100n) - 3n, -3.5, -1, 0, 129, 256, (1n << 100n) + 5n]
      const narrow = dtype.startsWith('uint') ? BigInt.asUintN : BigInt.asIntN
      const expected = values.map(v => narrow(bits, typeof v === 'bigint' ? v : BigInt(Math.trunc(v))))
      const tensor = new Tensor(values, { dtype })
      assertShape(tensor.shape, [7])
      const root = tensor.uopPhysical
      assert(root.op === pg._core.ops.COPY && root.src.length === 1)
      assert(root.src[0].op === pg._core.ops.BUFFER && root.src[0].src.length === 1)
      assert(tensor.dtype === dtype)
      const got = Array.from(await tensor.toArrayAsync())
      assert(got.length === expected.length)
      assert(got.every((v,i) => BigInt(v) === expected[i]), `wrong ${dtype} storage values`)
    })
  }

  await test('constructor parity rejects weak storage and ragged shapes before import', async () => {
    for (const dtype of ['weakint', 'weakfloat']) {
      for (const values of [[1], new Float32Array([1])]) {
        let error
        try { new Tensor(values, { dtype }) } catch (e) { error = e }
        assert(error && /cannot create storage for weak dtype/.test(error.message))
      }
    }
    for (const data of [[[1], []], [[], [[]]], [[1, 2], [3], [4, 5, 6]]]) {
      let error
      try { new Tensor(data, { dtype: 'int32' }) } catch (e) { error = e }
      assert(error && /inhomogeneous shape/.test(error.message))
    }
  })

  await test('constructor parity integer lists reject nonfinite values', async () => {
    for (const dtype of ['int32', 'uint32', 'int64', 'uint64']) {
      for (const value of [NaN, Infinity, -Infinity]) {
        let error
        try { new Tensor([value], { dtype }) } catch (e) { error = e }
        assert(error instanceof RangeError, `${dtype} admitted ${value}`)
      }
    }
  })

  await test('logical policy scope and tensor override', async () => {
    const current = new Tensor([0]).add(1)
    assert(current.logicalPolicy === 'until_realize')
    await current.realizeAsync()
    assert(current.logicalState === 'retired')

    const scoped = pg.withLogical('never', () => new Tensor([1, 2]))
    assert(scoped.logicalPolicy === 'never')
    assert(scoped.logicalState === 'never_constructed')
    assert(scoped.uopLogical === null)

    const retained = pg.withLogical('always', () => new Tensor([3, 4]).add(1))
    await retained.realizeAsync()
    assert(retained.logicalPolicy === 'always')
    assert(retained.logicalState === 'available')

    const dropped = new Tensor([5, 6], { logical: false })
    assert(dropped.logicalPolicy === 'never')
    assert(dropped.uopLogical === null)
    assert(dropped.setLogicalPolicy('always') === false)
    const descendant = dropped.add(1)
    assert(descendant.logicalPolicy === 'never')
    assert(descendant.logicalState === 'never_constructed')
    assertClose(await descendant.toArray(), [6, 7])
    const cloned = dropped.clone()
    assert(cloned.logicalPolicy === 'never')
    assert(cloned.logicalState === 'never_constructed')
    assert(cloned.uopLogical === null)
    assertClose(await cloned.toArray(), [5, 6])
    const gradSource = new Tensor([2], { dtype: 'float32', logical: false })
    gradSource.mul(gradSource).sum().backward()
    assert(gradSource.grad.logicalPolicy === 'never')
    assert(gradSource.grad.logicalState === 'never_constructed')
    assert(gradSource.grad.uopLogical === null)
    assertClose(await gradSource.grad.toArray(), [4])
  })

  for (const logical of ['never', 'always', 'until_realize']) {
    for (const realized of [false, true]) {
      await test(`copyFrom host input ${logical} realized=${realized}`, async () => {
        const x = new Tensor(new Float32Array([1, 2, 3]), { logical })
        const retained = x.uopLogical
        if (realized) await x.realizeAsync()
        const before = x.uopPhysical.key
        if (pg.device === 'webgpu' && !realized) {
          let error
          try { x.copyFrom([4, 5, 6]) } catch (e) { error = e }
          assert(error && String(error.message).includes('copyFromAsync'))
          assert(x.uopPhysical.key === before)
          await x.copyFromAsync([4, 5, 6])
        } else {
          x.copyFrom([4, 5, 6])
        }
        const current = x.uopPhysical
        assert(current.op === pg._core.ops.BUFFER && current.src.length === 1 &&
          current.src[0].op === pg._core.ops.CONST)
        if (realized) assert(current.key === before)
        if (logical === 'never') assert(x.uopLogical === null)
        else if (logical === 'always') assert(x.uopLogical.key === retained.key)
        else assert(x.logicalState === 'retired')
        x.copyFrom([7, 8, 9])
        assert(x.uopPhysical.key === current.key)
        assertClose(await x.toArrayAsync(), [7, 8, 9])
      })
    }
  }

  await test('copyFrom validates before materialization and orders pending assign', async () => {
    const x = new Tensor(new Float32Array([1, 2, 3]), { logical: 'always' })
    const before = x.uopPhysical.key
    for (const bad of [new Float32Array([9]), new Int32Array([9, 9, 9])]) {
      let error
      try { x.copyFrom(bad) } catch (e) { error = e }
      assert(error && /size mismatch|dtype mismatch/.test(error.message))
      assert(x.uopPhysical.key === before)
    }
    assertClose(await x.toArrayAsync(), [1, 2, 3])
    x.assign(new Tensor([10, 20, 30], { dtype: 'float32' }))
    if (pg.device === 'webgpu') await x.copyFromAsync([4, 5, 6])
    else x.copyFrom([4, 5, 6])
    assertClose(await x.toArrayAsync(), [4, 5, 6])
  })

  await test('copyFromAsync snapshots caller bytes before materialization', async () => {
    const x = new Tensor(new Float32Array([1, 2, 3]))
    const values = new Float32Array([4, 5, 6])
    const pending = x.copyFromAsync(values)
    values.fill(99)
    assert(await pending === x)
    assertClose(await x.toArrayAsync(), [4, 5, 6])
    const current = x.uopPhysical.key
    let error
    try { await x.copyFromAsync([0]) } catch (e) { error = e }
    assert(error && String(error.message).includes('size mismatch'))
    assert(x.uopPhysical.key === current)
    assertClose(await x.toArrayAsync(), [4, 5, 6])
  })

  await test('from vector', async () => {
    const t = new Tensor([1, 2, 3])
    assertShape(t.shape, [3])
    assert(t.dtype === 'int32', `expected int32, got ${t.dtype}`)
    assertClose(await t.toArray(), [1, 2, 3])
  })

  await test('Tensor dispose retires its exact core owner', async () => {
    const before = pg.stats().coreStats.tensorRecords
    const t = Tensor.empty([8], { dtype: 'float32' })
    assert(pg.stats().coreStats.tensorRecords === before + 1,
      'Tensor construction should add one core owner')
    await t.dispose()
    assert(pg.stats().coreStats.tensorRecords === before,
      'Tensor dispose should retire its exact core owner')
    await t.dispose()
  })

  await test('raw UOp owns residency after Tensor dispose', async () => {
    const before = pg.stats().coreStats
    const t = Tensor.empty([1024], { dtype: 'float32' })
    t.copyFrom(new Float32Array(1024))
    const uop = t.uop
    await t.dispose()
    let stats = pg.stats().coreStats
    assert(stats.tensorRecords === before.tensorRecords,
      'raw UOp ownership should not retain the Tensor record')
    assert(stats.memUsed === before.memUsed + 4096,
      'raw physical UOp should retain its storage')
    await uop.dispose()
    pg.collect()
    stats = pg.stats().coreStats
    assert(stats.memUsed === before.memUsed,
      'raw UOp dispose should retire its storage')
  })

  await test('downstream core graph owns disposed input residency', async () => {
    const before = pg.stats().coreStats
    const input = Tensor.empty([1024], { dtype: 'float32' })
    input.copyFrom(new Float32Array(1024))
    const one = new Tensor(1, { dtype: 'float32' })
    const out = input.add(one)
    await input.dispose()
    await one.dispose()
    let stats = pg.stats().coreStats
    assert(stats.memUsed === before.memUsed + 4096,
      'downstream physical UOp graph should retain input storage')
    await out.realize()
    stats = pg.stats().coreStats
    assert(stats.memUsed === before.memUsed + 4096,
      'realized output should replace obsolete input storage')
    await out.dispose()
    pg.collect()
    stats = pg.stats().coreStats
    assert(stats.memUsed === before.memUsed,
      'last downstream owner should retire output storage')
  })

  await test('array and TypedArray dtype inference matches tinygrad', async () => {
    assert(new Tensor([[1, 2], [3, 4]]).dtype === 'int32', 'nested integers should infer int32')
    assert(new Tensor([true, false]).dtype === 'bool', 'booleans should infer bool')
    assert(new Tensor([1, 2.5]).dtype === 'float32', 'mixed numeric values should infer float32')
    assert(new Tensor([]).dtype === 'float32', 'empty arrays should infer float32')
    assert(new Tensor(new Int16Array([1, 2])).dtype === 'int16', 'Int16Array should preserve int16')
    assert(new Tensor(new Uint32Array([1, 2])).dtype === 'uint32', 'Uint32Array should preserve uint32')
  })

  await test('bfloat16 host values stage through float32', async () => {
    const t = new Tensor([1.0, 2.0, 3.0, 4.0], { dtype: 'bfloat16' })
    const y = await t.add(t).realize()
    assert(y.dtype === 'bfloat16', `expected bfloat16, got ${y.dtype}`)
    assertClose(await y.toArray(), [2.0, 4.0, 6.0, 8.0])
  })

  await test('explicit bfloat16 overrides Float64Array source dtype', async () => {
    const t = new Tensor(new Float64Array([1.0, 2.0]), { dtype: 'bfloat16' })
    assert(t.dtype === 'bfloat16', `expected bfloat16, got ${t.dtype}`)
    assertShape(t.shape, [2])
    assertClose(await t.toArray(), [1.0, 2.0])
  })

  await test('fp8 host values match current tinygrad', async () => {
    const values = [-Infinity, -1.5, -0, 0, 0.1, 1, 1.5, 448, Infinity, NaN]
    const cases = {
      fp8e4m3: [NaN, -1.5, -0, 0, 0.1015625, 1, 1.5, 448, NaN, NaN],
      fp8e5m2: [-Infinity, -1.5, -0, 0, 0.09375, 1, 1.5, 448, Infinity, NaN],
      fp8e4m3fnuz: [NaN, -1.5, 0, 0, 0.1015625, 1, 1.5, 240, NaN, NaN],
      fp8e5m2fnuz: [NaN, -1.5, 0, 0, 0.09375, 1, 1.5, 448, NaN, NaN]
    }
    for (const [dtype, expected] of Object.entries(cases)) {
      const tensor = new Tensor(values, { dtype })
      assert(tensor.dtype === dtype, `${dtype}: got ${tensor.dtype}`)
      assert(countGraphOp(tensor.uop, pg._core.ops.COPY) === 1, `${dtype}: missing creation COPY`)
      const actual = await tensor.cast('float32').toArray()
      assert(actual.length === expected.length, `${dtype}: length mismatch`)
      for (let i = 0; i < actual.length; i++) {
        if (Number.isNaN(expected[i])) assert(Number.isNaN(actual[i]), `${dtype}[${i}] expected NaN`)
        else assert(Object.is(actual[i], expected[i]), `${dtype}[${i}] ${actual[i]} != ${expected[i]}`)
      }
    }
  })

  await testIf(supportsF16, 'numeric float16 host values preserve pinned bits and direct topology', async () => {
    const values = [1.5, -2.25, 0.5, NaN, Infinity, -Infinity, 65504]
    const inputs = [
      ['array', values, [7]],
      ['nested', [[1.5, -2.25], [0.5, 65504]], [2, 2]],
      ['float64array', new Float64Array(values), [7]],
      ['uint16array-numeric', new Uint16Array([1, 2, 3]), [3]]
    ]
    for (const [name, input, shape] of inputs) {
      const direct = new Tensor(input, { dtype: 'float16' })
      const control = new Tensor(input, { dtype: 'float32' }).cast('float16')
      assertShape(direct.shape, shape)
      assert(countGraphOp(direct.uop, pg._core.ops.CAST) === 0, `${name} direct graph gained CAST`)
      assert(countGraphOp(control.uop, pg._core.ops.CAST) === 1, `${name} control graph lost CAST`)
      const actual = await direct.toArray()
      const expected = await control.toArray()
      assert(actual.length === expected.length, `${name} length mismatch`)
      for (let i = 0; i < actual.length; i++) {
        if (Number.isNaN(expected[i])) assert(Number.isNaN(actual[i]), `${name}[${i}] expected NaN`)
        else assert(Object.is(actual[i], expected[i]) || actual[i] === expected[i],
          `${name}[${i}] ${actual[i]} != ${expected[i]}`)
      }
    }
    const directSubnormal = new Tensor([2 ** -24], { dtype: 'float16' })
    const controlSubnormal = new Tensor([2 ** -24], { dtype: 'float32' }).cast('float16')
    if (pg.core === 'native' && pg.device === 'cpu') {
      assert((await directSubnormal.item()) === 2 ** -24,
        'native CPU direct float16 lost minimum subnormal storage')
      assert((await controlSubnormal.item()) === 2 ** -24,
        'native CPU float16 control lost minimum subnormal')
    } else if (pg.device === 'wasm' || pg.device === 'interp') {
      // Pinned PythonRenderer on Python 3.11 decomposes unsupported half LOAD
      // through f2f: a direct half subnormal flushes, while the unmaterialized
      // f32->half->f32 control remains exact (codegen/__init__.py:116-140).
      assert((await directSubnormal.item()) === 0,
        `${pg.device} direct float16 must match pinned non-native-half flush`)
      assert((await controlSubnormal.item()) === 2 ** -24,
        `${pg.device} float16 control must match pinned non-native-half graph`)
    }
    const scalar = new Tensor(1.5, { dtype: 'float16' })
    assertShape(scalar.shape, [])
    assert((await scalar.item()) === 1.5, 'scalar float16 construction mismatch')
  })

  await test('from scalar', async () => {
    const cases = [
      [new Tensor(true), 'bool', true],
      [new Tensor(42), 'weakint', 42],
      [new Tensor(42.0, { dtype: 'float32' }), 'float32', 42],
      [new Tensor(7, { device: 'cuda' }), 'weakint', 7],
      [new Tensor(1.5, { device: 'cuda' }), 'weakfloat', 1.5]
    ]
    for (const [tensor, dtype, value] of cases) {
      assertShape(tensor.shape, [])
      assert(tensor.dtype === dtype, `expected ${dtype}, got ${tensor.dtype}`)
      assert(tensor.uop.op === pg._core.ops.CONST, 'expected scalar CONST root')
      assert(tensor.uop.key === tensor.uopLogical.key, 'expected shared scalar roots')
      if (tensor.device === 'CPU') {
        const actual = await tensor.item()
        assert(Math.abs(Number(actual) - Number(value)) < 1e-4, `Expected ${value}, got ${actual}`)
      }
    }
  })

  await test('from 2D', async () => {
    const t = new Tensor([[1, 2], [3, 4]])
    assertShape(t.shape, [2, 2])
    assertClose(await t.toArray(), [1, 2, 3, 4])
  })

  await test('empty creates unrealized buffer placeholder', async () => {
    const t = Tensor.empty([2, 3])
    assertShape(t.shape, [2, 3])
    assert(t.uop.hasBufferIdentity(), 'empty should be backed by a BUFFER UOp')
    assert(t.uopPhysical, 'empty should have a physical root at construction')
    assert(
      t.uopLogical.buffer.src[0].op === pg._core.ops.UNIQUE,
      'logical BUFFER should retain the portable resource identity'
    )
    assert(t.uopLogical.buffer.src.length === 1, 'logical BUFFER should stay device-free')
    assert(
      t.uopPhysical.buffer.src[0].op === pg._core.ops.CONST,
      'physical BUFFER should encode the same resource slot in ParamArg'
    )
  })

  await test('movement is realized through recursive base', async () => {
    const source = await new Tensor([1, 2, 3, 4]).realize()
    const reshaped = source.reshape(2, 2)
    const view = reshaped.flatten().shrink([[1, 3]])

    assert(view.uop.op === pg._core.ops.SHRINK, 'expected a SHRINK view')
    assert(view.uop.base.key === source.uop.base.key, 'movement base should be recursive')
    assert(reshaped.uop.realized === null, 'RESHAPE is not directly realized')
    assert(reshaped.uop.isRealized, 'RESHAPE should be realized through its base')
    assert(view.uop.realized === null, 'movement UOp is not directly realized')
    assert(view.uop.isRealized, 'allocated recursive base should realize the movement view')
    assert(view.uop.is_realized, 'snake-case realization alias should match')
  })

  await test('flatten resolves negative dimensions like tinygrad', async () => {
    const flattened = Tensor.arange(32).reshape(1, 2, 16).flatten(-2)
    assertShape(flattened.shape, [1, 32])
    assert(flattened.uop.op === pg._core.ops.RESHAPE, 'flatten should be one RESHAPE')
    assertClose(await flattened.toArray(), Array.from({ length: 32 }, (_, i) => i))
  })

  await test('clone is lazy separate and preserves state', async () => {
    const source = Tensor.empty([4], { dtype: 'float32' }).is_param_(false)
    source.copyFrom(new Float32Array([1, 2, 3, 4]))
    await source.sum().backward()

    const cloned = source.clone(pg.device)
    assert(cloned.uopLogical && cloned.uopLogical.src.length === 2, 'clone should be AFTER')
    assert(cloned.uopLogical.src[1].src.length === 2, 'clone effect should be STORE')
    assert(cloned.uopLogical.src[0].buffer.key !== source.uop.buffer.key, 'clone needs a separate buffer')
    assert(cloned.isParam === false, 'clone should preserve isParam')
    assert(cloned.grad && cloned.grad.uopLogical.src.length === 2, 'clone should recursively clone grad')
    assert(
      cloned.grad.uopLogical.src[0].buffer.key !== source.grad.uopLogical.src[0].buffer.key,
      'cloned grad needs a separate buffer'
    )
    assertClose(await cloned.toArray(), [1, 2, 3, 4])
    assertClose(await cloned.grad.toArray(), [1, 1, 1, 1])
  })

  await test('clone preserves scalar shape across devices', async () => {
    const source = Tensor.full([], 3, { device: 'cpu' })
    const cloned = source.clone('interp')
    assertShape(cloned.shape, [])
    assert(cloned.device === 'INTERP', `expected INTERP, got ${cloned.device}`)
    assertClose(await cloned.toArray(), [3])
  })

  await test('static constructors preserve scalar shape', async () => {
    const tensors = [Tensor.zeros([]), Tensor.ones([]), Tensor.full([], 3)]
    for (const tensor of tensors) assertShape(tensor.shape, [])
    assertClose(await tensors[0].toArray(), [0])
    assertClose(await tensors[1].toArray(), [1])
    assertClose(await tensors[2].toArray(), [3])
  })

  await test('detach is a lazy graph boundary', async () => {
    const source = new Tensor([[1, 2], [3, 4]], { dtype: 'float32' })
    const detached = source.detach()

    assertShape(detached.shape, source.shape)
    assert(detached.dtype === source.dtype, 'detach should preserve dtype')
    assert(detached.device === source.device, 'detach should preserve device')
    assert(detached.isParam === true, 'ordinary Tensor operations should default to isParam=true')
    assert(detached.uopLogical.op === pg._core.ops.DETACH, 'detach should create a DETACH UOp')
    assert(detached.uopLogical.src.length === 1, 'detach should retain a unary graph node')
    assert(
      detached.uopLogical.src[0].key === source.uopLogical.key,
      'logical detach should retain the logical source UOp'
    )
    if (detached.uopPhysical) {
      assert(detached.uopPhysical.op === pg._core.ops.DETACH, 'physical detach should retain DETACH')
      assert(
        detached.uopPhysical.src[0].key === source.uop.key,
        'physical detach should retain the current source UOp'
      )
    }

    await detached.sum().backward()
    assertClose(await source.grad.toArray(), [0, 0, 0, 0])
  })

  await test('contiguousBackward has exact gradient barrier', async () => {
    const source = new Tensor([1, -2, 3], { dtype: 'float32' })
    const result = source.mul(2).contiguousBackward()
    assert(
      result.uopLogical.op === pg._core.ops.CONTIGUOUS_BACKWARD,
      'logical root should be CONTIGUOUS_BACKWARD'
    )
    assert(
      result.uop.op === pg._core.ops.CONTIGUOUS_BACKWARD,
      'physical root should be CONTIGUOUS_BACKWARD'
    )
    assert(
      result.uopLogical.src[0].op === pg._core.ops.MUL,
      'CONTIGUOUS_BACKWARD should wrap the exact MUL input'
    )
    assert(
      result.contiguous_backward().uop.op === pg._core.ops.CONTIGUOUS_BACKWARD,
      'snake-case alias should retain CONTIGUOUS_BACKWARD'
    )
    await result.square().sum().backward()
    assertClose(await source.grad.toArray(), [8, -16, 24])
  })

  await test('backward clones deviceless grad and accumulates in place', async () => {
    const x = Tensor.empty([4], { dtype: 'float32' })
    const loss = x.sum()
    await loss.backward()
    const firstGrad = x.grad
    const firstRoot = firstGrad.uopLogical
    const firstBuffer = firstRoot.src[0].buffer.key
    assert(firstRoot.src.length === 2, 'first grad should be AFTER')

    await loss.backward()
    assert(x.grad === firstGrad, 'gradient accumulation should preserve Tensor identity')
    const secondRoot = x.grad.uopLogical
    assert(secondRoot.src[0].key === firstRoot.key, 'gradient effect root changed')
    assert(secondRoot.src[0].src[0].buffer.key === firstBuffer, 'gradient buffer identity changed')
    assertClose(await x.grad.toArray(), [2, 2, 2, 2])
  })

  await test('backward through clone reaches source', async () => {
    const source = Tensor.empty([4], { dtype: 'float32' })
    source.copyFrom(new Float32Array([1, 2, 3, 4]))
    const cloned = source.clone()
    await cloned.sum().backward()
    assertClose(await source.grad.toArray(), [1, 1, 1, 1])
    assertClose(await cloned.grad.toArray(), [1, 1, 1, 1])
  })

  await test('backward retains distinct wrappers sharing one UOp', async () => {
    const x = new Tensor([1, 2, 3, 4], { dtype: 'float32' })
    // Pinned Tensor.__init__ wraps an existing current Tensor.uop directly
    // (tensor.py:92-121); retained logical provenance is not executable state.
    const y = new Tensor(x.uop, {})
    assert(x !== y, 'expected distinct Tensor wrappers')
    assert(x.uop.key === y.uop.key, 'expected one shared current UOp')

    await x.sum().backward()
    assertClose(await x.grad.toArray(), [1, 1, 1, 1])
    assertClose(await y.grad.toArray(), [1, 1, 1, 1])
  })

  await test('static constructors accept tinygrad-style shape arrays', async () => {
    const z = Tensor.zeros([2, 3])
    const o = Tensor.ones([2, 3])
    const e = Tensor.empty([2, 3])
    assertShape(z.shape, [2, 3])
    assertShape(o.shape, [2, 3])
    assertShape(e.shape, [2, 3])
    assertClose(await z.toArray(), [0, 0, 0, 0, 0, 0])
    assertClose(await o.toArray(), [1, 1, 1, 1, 1, 1])
  })

  await test('arange follows tinygrad start stop order', async () => {
    const deviceFree = Tensor.arange(6)
    const deviceFreeRoot = deviceFree.uop.key
    assert(Number(pg._core.ffi.poly_uop_device(deviceFree.uop.raw)) === 0,
      'pure arange should remain device-free before readback')
    assertClose(await deviceFree.toArray(), [0, 1, 2, 3, 4, 5])
    assert(deviceFree.uop.key === deviceFreeRoot,
      'device-free readback must realize a temporary without rewriting the source root')
    assertClose(await Tensor.arange(0, 6).toArray(), [0, 1, 2, 3, 4, 5])
    assertClose(await Tensor.arange(2, 8, 2).toArray(), [2, 4, 6])
  })

  await test('runtime exposes uop namespace', async () => {
    const t = new Tensor([[1, 2], [3, 4]])
    assert(pg.uop, 'runtime should expose pg.uop')
    assertShape(pg.uop.shape(t.uop), [2, 2])
    assert(pg.uop.dtype(t.uop) === 'int32', `expected int32, got ${pg.uop.dtype(t.uop)}`)
    // Tinygrad 2026-08-22/a9069c177a9d uop/ops.py:828-839 keeps the host
    // BUFFER behind a lazy COPY on every target, including Wasm.
    assert(!pg.uop.hasBufferIdentity(t.uop), 'host import should remain a lazy COPY')
    await t.realize()
    assert(pg.uop.hasBufferIdentity(t.uop), 'realized host tensor should have buffer identity')
    assert(pg.uop.buffer(t.uop), 'pg.uop.buffer should return a UOp')
  })

  await test('customKernel executes UOp CALL body', async () => {
    function addKernel(c, a, b) {
      c = c.flatten(); a = a.flatten(); b = b.flatten()
      const i = pg.uop.range(c.numel(), 0)
      return c.index(i).store(a.index(i).add(b.index(i))).end(i).sink(
        new pg.uop.KernelInfo('custom_add_4')
      )
    }
    // Pinned UOp.store preserves its value dtype (uop/ops.py:531-533); raw
    // custom kernels must cast explicitly or use matching storage/value types.
    const a = new Tensor([1, 2, 3, 4], { dtype: 'float32' })
    const b = new Tensor([10, 20, 30, 40], { dtype: 'float32' })
    const c = Tensor.empty([4], { dtype: 'float32' })
    const out = c.customKernel(a, b, addKernel)[0]
    assertClose(await out.toArray(), [11, 22, 33, 44])
  })

  await test('customKernel RANGE numeric scalar preserves weakint', async () => {
    const index = pg.uop.range(64, 0)
    const offset = index.mul(64)
    assert(pg.uop.dtype(index) === 'weakint', 'RANGE should expose weakint dtype')
    assert(pg.uop.dtype(index.src[0]) === 'weakint', 'RANGE bound should be weakint')
    assert(pg.uop.dtype(offset) === 'weakint', 'index expression should remain weakint')
    assert(
      offset.src.every(src => pg.uop.dtype(src) === 'weakint'),
      'numeric scalar should be coerced to the index weakint dtype'
    )
  })

  await test('customKernel multi-output backward matches tinygrad pattern', async () => {
    let callbackCall = null
    function addmulKernel(c, d, a, b) {
      c = c.flatten(); d = d.flatten(); a = a.flatten(); b = b.flatten()
      const i = pg.uop.range(c.numel(), 0)
      const storeC = c.index(i).store(a.index(i).add(b.index(i)))
      const storeD = d.index(i).store(a.index(i).mul(b.index(i)))
      return storeC.group(storeD).end(i).sink({ arg: new pg.uop.KernelInfo('addmul') })
    }
    function backwardAddmul(gradC, gradD, call) {
      callbackCall = call
      const [, , , a, b] = call.src
      const gradA = new Tensor(gradC).add(new Tensor(gradD).mul(new Tensor(b))).uop
      const gradB = new Tensor(gradC).add(new Tensor(gradD).mul(new Tensor(a))).uop
      return [null, null, gradA, gradB]
    }
    const aVals = [
      0.3, -1.2, 0.7, 2.1,
      -0.5, 1.4, -2.2, 0.9,
      1.1, -0.8, 2.4, -1.7,
      0.2, 0.6, -0.4, 1.8,
    ]
    const bVals = [
      1.2, 0.5, -0.3, 0.8,
      2.0, -1.1, 0.4, -0.7,
      0.9, 1.5, -2.5, 0.1,
      -1.3, 0.2, 1.7, -0.6,
    ]
    const aRef = new Tensor(aVals, {}).reshape(4, 4)
    const bRef = new Tensor(bVals, {}).reshape(4, 4)
    await aRef.add(bRef).sum().add(aRef.mul(bRef).sum()).backward()

    const a = new Tensor(aVals, {}).reshape(4, 4)
    const b = new Tensor(bVals, {}).reshape(4, 4)
    await a.realize(b)
    const aPhysical = a.uopPhysical.key
    const bPhysical = b.uopPhysical.key
    const [c, d] = Tensor.empty([4, 4]).customKernel(
      Tensor.empty([4, 4]), a, b, { fxn: addmulKernel, gradFxn: backwardAddmul }
    )
    await c.sum().add(d.sum()).backward()
    assert(callbackCall.src[3].key === aPhysical, 'callback must receive physical a CALL slot')
    assert(callbackCall.src[4].key === bPhysical, 'callback must receive physical b CALL slot')
    assertClose(await a.grad.toArray(), await aRef.grad.toArray(), 1e-4)
    assertClose(await b.grad.toArray(), await bRef.grad.toArray(), 1e-4)
  })

  await test('customKernel physical AFTER preserves data gradient', async () => {
    function identityKernel(x) {
      x = x.flatten()
      const i = pg.uop.range(x.numel(), 0)
      return x.index(i).store(x.index(i)).end(i).sink({ arg: new pg.uop.KernelInfo('identity') })
    }
    function backwardIdentity(grad, call) {
      assert(call.src.length === 2, 'expected one-argument custom CALL in backward')
      return [null]
    }
    const x = Tensor.empty([4], { dtype: 'float32' })
    x.copyFrom(new Float32Array([1, 2, 3, 4]))
    const y = x.customKernel({ fxn: identityKernel, gradFxn: backwardIdentity })[0]
    assert(y.uopLogical && y.uopLogical.src.length === 2, 'expected logical AFTER')
    assert(y.uopPhysical && y.uopPhysical.src.length === 2, 'expected physical AFTER')
    assert(y.uopLogical.op === y.uopPhysical.op, 'logical and physical aliases must both be AFTER')
    await y.sum().backward()
    assertClose(await x.grad.toArray(), [1, 1, 1, 1])
    assertClose(await y.grad.toArray(), [1, 1, 1, 1])
  })

  await test('customKernel separates output and input gradient edges', async () => {
    function identityKernel(out, x) {
      out = out.flatten(); x = x.flatten()
      const i = pg.uop.range(out.numel(), 0)
      return out.index(i).store(x.index(i)).end(i).sink({ arg: new pg.uop.KernelInfo('identity_grad_edges') })
    }
    function backwardIdentity(grad, call) {
      assert(call.src.length === 3, 'expected output and input custom CALL arguments')
      return [null, grad]
    }
    const out = Tensor.empty([4], { dtype: 'float32' })
    const x = new Tensor([1, 2, 3, 4], { dtype: 'float32' })
    const y = out.customKernel(x, { fxn: identityKernel, gradFxn: backwardIdentity })[0]
    await y.sum().backward()

    assertClose(await out.grad.toArray(), [1, 1, 1, 1])
    assertClose(await x.grad.toArray(), [1, 1, 1, 1])
    assertClose(await y.grad.toArray(), [1, 1, 1, 1])
  })

  await test('customKernel duplicate output alias passes one accumulated upstream', async () => {
    function identityKernel(out0, out1, x) {
      out0 = out0.flatten(); out1 = out1.flatten(); x = x.flatten()
      const i = pg.uop.range(out0.numel(), 0)
      return out0.index(i).store(x.index(i)).end(i).sink({ arg: new pg.uop.KernelInfo('duplicate_output_grad') })
    }
    const callbackCounts = []
    function backwardIdentity(...args) {
      const call = args.pop()
      callbackCounts.push(args.length)
      return [null, null, args[0]]
    }
    const out = Tensor.empty([4], { dtype: 'float32' })
    const x = new Tensor([1, 2, 3, 4], { dtype: 'float32' })
    const [y0, y1] = out.customKernel(out, x, { fxn: identityKernel, gradFxn: backwardIdentity })
    assert(y0.uop.key === y1.uop.key, 'duplicate output aliases should share one AFTER')
    await y0.sum().add(y1.sum()).backward()

    assert(callbackCounts.length === 1 && callbackCounts[0] === 1, 'expected one accumulated callback upstream')
    assertClose(await x.grad.toArray(), [2, 2, 2, 2])
  })

  await test('customKernel without gradFxn rejects a needed input gradient', async () => {
    function identityKernel(out, x) {
      out = out.flatten(); x = x.flatten()
      const i = pg.uop.range(out.numel(), 0)
      return out.index(i).store(x.index(i)).end(i).sink({ arg: new pg.uop.KernelInfo('missing_grad_fxn') })
    }
    const out = Tensor.empty([4], { dtype: 'float32' })
    const x = new Tensor([1, 2, 3, 4], { dtype: 'float32' })
    const y = out.customKernel(x, identityKernel)[0]
    let threw = false
    try {
      await y.sum().backward()
    } catch (e) {
      threw = String(e.message || e).includes('expected TUPLE body for gradient, got Ops.SINK')
    }
    assert(threw, 'missing gradFxn should reject an opaque CALL input gradient')
    assert(x.grad === null, 'failed backward must not assign x.grad')
    assert(y.grad === null, 'failed backward must not assign y.grad')
  })

  await test('customKernel callback is inactive behind stop-gradient ops', async () => {
    function identityKernel(out, x) {
      out = out.flatten(); x = x.flatten()
      const i = pg.uop.range(out.numel(), 0)
      return out.index(i).store(x.index(i)).end(i).sink({ arg: new pg.uop.KernelInfo('stopped_custom_grad') })
    }

    let out = Tensor.empty([4], { dtype: 'float32' })
    let x = new Tensor([1, 2, 3, 4], { dtype: 'float32' })
    let y = out.customKernel(x, identityKernel)[0]
    await y.detach().sum().backward()
    assertClose(await x.grad.toArray(), [0, 0, 0, 0])
    assertClose(await y.grad.toArray(), [0, 0, 0, 0])

    const calls = []
    function backwardIdentity(grad, call) {
      calls.push(grad.op)
      return [null, new Tensor(grad).add(7).uop]
    }
    out = Tensor.empty([4], { dtype: 'float32' })
    x = new Tensor([1, 2, 3, 4], { dtype: 'float32' })
    y = out.customKernel(x, { fxn: identityKernel, gradFxn: backwardIdentity })[0]
    await y.lt(0).cast('float32').sum().backward()
    assert(calls.length === 0, 'stop-gradient ops must not invoke custom callbacks')
    assertClose(await x.grad.toArray(), [0, 0, 0, 0])
    assertClose(await y.grad.toArray(), [0, 0, 0, 0])
  })

  await test('customKernel reuses buffers after input update', async () => {
    function addKernel(c, a, b) {
      c = c.flatten(); a = a.flatten(); b = b.flatten()
      const i = pg.uop.range(c.numel(), 0)
      return c.index(i).store(a.index(i).add(b.index(i))).end(i).sink(
        new pg.uop.KernelInfo('custom_add_reuse_4')
      )
    }
    const a = Tensor.empty([4], { dtype: 'float32' })
    const b = new Tensor([10, 20, 30, 40], { dtype: 'float32' })
    const c = Tensor.empty([4], { dtype: 'float32' })
    const runs = [
      [new Float32Array([1, 2, 3, 4]), [11, 22, 33, 44]],
      [new Float32Array([5, 6, 7, 8]), [15, 26, 37, 48]],
    ]
    for (const [vals, expected] of runs) {
      a.copyFrom(vals)
      const out = c.customKernel(a, b, addKernel)[0]
      assert(out.uopLogical, 'custom output should keep a logical root')
      await out.realize()
      assert(out.uopPhysical && out.uopPhysical.hasBufferIdentity(), 'custom output should realize to a buffer-backed root')
      assertClose(await out.toArray(), expected)
    }
  })

  await test('customKernel exposes UOp compare where and unary methods', async () => {
    function selectKernel(out, a, b) {
      out = out.flatten(); a = a.flatten(); b = b.flatten()
      const i = pg.uop.range(out.numel(), 0)
      const av = a.index(i)
      const bv = b.index(i)
      const selected = av.lt(0).where(av.neg(), av.max(bv))
      return out.index(i).store(selected).end(i).sink(
        new pg.uop.KernelInfo('custom_select')
      )
    }
    const out = Tensor.empty([4], { dtype: 'float32' })
    const a = new Tensor([-3, 2, 5, -1], { dtype: 'float32' })
    const b = new Tensor([1, 4, 3, 9], { dtype: 'float32' })
    assertClose(await out.customKernel(a, b, selectKernel)[0].toArray(), [3, 4, 5, 1])
  })

  await test('customKernel rejects bool INDEX coordinate before codegen', async () => {
    function invalidIndexKernel(out) {
      out = out.flatten()
      const zero = pg.uop.constant(0)
      const gate = zero.lt(1)
      const bad = out.index(gate)
      assert(bad !== null, 'UOp.index(bool) construction should match tinygrad')
      return bad.store(out.index(zero)).sink(
        new pg.uop.KernelInfo('invalid_bool_index')
      )
    }
    const out = Tensor.empty([1], { dtype: 'float32' })
    let threw = false
    try {
      await out.customKernel(invalidIndexKernel)[0].toArray()
    } catch (e) {
      threw = true
    }
    assert(threw, 'invalid bool INDEX coordinate must fail before codegen')
  })

  await test('customKernel exposes tinygrad-style floor div and mod', async () => {
    function divKernel(out, x, y) {
      out = out.flatten(); x = x.flatten(); y = y.flatten()
      const i = pg.uop.range(x.numel(), 0)
      const q = x.index(i).floordiv(y.index(i))
      const r = x.index(i).floormod(y.index(i))
      return out.index(i).store(q)
        .group(out.index(i.add(x.numel())).store(r))
        .end(i).sink(new pg.uop.KernelInfo('custom_signed_div_mod'))
    }
    const x = new Tensor(new Int32Array([-7, -7, 7, 7, -1, 1, 0]), { dtype: 'int32' })
    const y = new Tensor(new Int32Array([3, -3, -3, 3, 4, -4, 3]), { dtype: 'int32' })
    const out = Tensor.empty([14], { dtype: 'int32' })
    assertClose(
      await out.customKernel(x, y, divKernel)[0].toArray(),
      [-3, 2, -3, 2, -1, -1, 0, 2, -1, -2, 1, 3, -3, 0]
    )
  })


  await test('customKernel numeric literals follow float operand dtype', async () => {
    function literalKernel(out, x) {
      out = out.flatten(); x = x.flatten()
      const i = pg.uop.range(x.numel(), 0)
      const xv = x.index(i)
      const sameAdd = xv.sub(1).div(xv.add(1))
      const sameMul = xv.mul(2).div(xv.mul(3))
      assert(pg.uop.dtype(xv.add(1).src[1]) === 'weakfloat', 'float scalar promotion should retain weakfloat')
      assert(pg.uop.dtype(pg.uop.constant(true)) === 'bool', 'boolean literals should retain bool')
      assert(pg.uop.dtype(pg.uop.constant(1, 'float32')) === 'float32', 'typed int should convert to float')
      assert(pg.uop.dtype(pg.uop.constant(1.75, 'int32')) === 'int32', 'typed float should convert to int')
      const s0 = out.index(i).store(sameAdd)
      const s1 = out.index(i.add(x.numel())).store(sameMul)
      return s0.group(s1).end(i).sink(new pg.uop.KernelInfo('custom_numeric_literals'))
    }
    const xData = new Float32Array(16)
    for (let i = 0; i < xData.length; i++) xData[i] = i / 10 + 1
    const out = Tensor.empty([32], { dtype: 'float32' })
    const got = await out.customKernel(new Tensor(xData, { dtype: 'float32' }), literalKernel)[0].toArray()
    const expected = []
    for (const x of xData) expected.push((x - 1) / (x + 1))
    for (const x of xData) expected.push((x * 2) / (x * 3))
    assertClose(got, expected, 1e-6)
  })

  await test('customKernel descriptor branches keep indexed placeholders typed', async () => {
    function termKernel(out, x, fa, fb, op, p0, p1) {
      out = out.flatten(); x = x.flatten(); fa = fa.flatten(); fb = fb.flatten()
      op = op.flatten(); p0 = p0.flatten(); p1 = p1.flatten()
      const rows = 4
      const cols = 3
      const idx = pg.uop.range(out.numel(), 0)
      const c = idx.floordiv(rows)
      const r = idx.mod(rows)
      const a = x.index(r.mul(cols).add(fa.index(c)))
      const b = x.index(r.mul(cols).add(fb.index(c)))
      assert(pg.uop.dtype(a) === 'float32', `indexed placeholder should be float32, got ${pg.uop.dtype(a)}`)
      const ab = b.lt(0).where(b.neg(), b)
      const safeDen = ab.lt(1e-6).where(b.lt(0).where(-1e-6, 1e-6), b)
      const z = a.mul(p0.index(c)).add(p1.index(c))
      const opv = op.index(c)
      let v = a.add(b)
      v = opv.eq(1).where(a.sub(b), v)
      v = opv.eq(2).where(a.mul(b), v)
      v = opv.eq(3).where(a.div(safeDen), v)
      v = opv.eq(4).where(z.sin(), v)
      return out.index(idx).store(v).end(idx).sink(
        new pg.uop.KernelInfo('custom_descriptor_branches')
      )
    }
    const xData = new Float32Array([
      1, 2, 3,
      4, 5, 6,
      7, 8, 9,
      10, 11, 12,
    ])
    const fa = new Int32Array([0, 1])
    const fb = new Int32Array([1, 2])
    const op = new Int32Array([0, 1])
    const p0 = new Float32Array([1.25, -0.5])
    const p1 = new Float32Array([0.1, 0.2])
    const expected = []
    for (let c = 0; c < 2; c++) {
      for (let r = 0; r < 4; r++) {
        const a = xData[r * 3 + fa[c]]
        const b = xData[r * 3 + fb[c]]
        expected.push(c === 0 ? a + b : a - b)
      }
    }
    const out = Tensor.empty([8], { dtype: 'float32' })
    const got = out.customKernel(
      new Tensor(xData),
      new Tensor(fa, { dtype: 'int32' }),
      new Tensor(fb, { dtype: 'int32' }),
      new Tensor(op, { dtype: 'int32' }),
      new Tensor(p0),
      new Tensor(p1),
      termKernel
    )[0]
    assertClose(await got.toArray(), expected, 1e-4)
  })

  await test('jit captures customKernel and replays after input update', async () => {
    function addKernel(c, a, b) {
      c = c.flatten(); a = a.flatten(); b = b.flatten()
      const i = pg.uop.range(c.numel(), 0)
      return c.index(i).store(a.index(i).add(b.index(i))).end(i).sink(
        new pg.uop.KernelInfo('jit_custom_add_4')
      )
    }
    const f = pg.jit((a, b) => {
      const c = Tensor.empty([4], { dtype: 'float32' })
      return c.customKernel(a, b, addKernel)[0]
    })
    const a = Tensor.empty([4], { dtype: 'float32' })
    const b = new Tensor([10, 20, 30, 40], { dtype: 'float32' })
    a.copyFrom(new Float32Array([1, 2, 3, 4]))
    assertClose(await (await f(a, b)).toArray(), [11, 22, 33, 44])
    assertClose(await (await f(a, b)).toArray(), [11, 22, 33, 44])
    assert(f.scheduleCount === 1, `expected one captured schedule, got ${f.scheduleCount}`)
    a.copyFrom(new Float32Array([5, 6, 7, 8]))
    assertClose(await (await f(a, b)).toArray(), [15, 26, 37, 48])
    assert(f.stats().replayCount === 1, 'third customKernel call should replay')
  })

  await test('compile captures customKernel and replays after input update', async () => {
    function addKernel(c, a, b) {
      c = c.flatten(); a = a.flatten(); b = b.flatten()
      const i = pg.uop.range(c.numel(), 0)
      return c.index(i).store(a.index(i).add(b.index(i))).end(i).sink(
        new pg.uop.KernelInfo('compile_custom_add_4')
      )
    }
    const a = Tensor.empty([4], { dtype: 'float32' })
    const b = new Tensor([10, 20, 30, 40], { dtype: 'float32' })
    a.copyFrom(new Float32Array([1, 2, 3, 4]))
    const compiled = await pg.compile((x, y) => {
      const c = Tensor.empty([4], { dtype: 'float32' })
      return c.customKernel(x, y, addKernel)[0]
    }, [a, b])
    assert(compiled.scheduleCount === 1, `expected one captured schedule, got ${compiled.scheduleCount}`)
    a.copyFrom(new Float32Array([5, 6, 7, 8]))
    const out = await compiled.run([a, b])
    assertClose(await out.toArray(), [15, 26, 37, 48])
    assert(compiled.stats().runCount === 1, 'compiled customKernel run should update runCount')
    compiled.dispose()
  })

  await test('compile captures current custom sum and replays after input update', async () => {
    function sumKernel(out, a) {
      out = out.flatten(); a = a.flatten()
      const r = pg.uop.range(8, 0, pg.uop.AxisType.REDUCE)
      let acc = out.index(0).set(0.0)
      acc = acc.index(0).set(acc.after(r).index(0).add(a.index(r)), r)
      return acc.sink(new pg.uop.KernelInfo({ name: 'custom_sum_8', opts_to_apply: [] }))
    }
    const a = Tensor.empty([8], { dtype: 'float32' })
    a.copyFrom(new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]))
    const compiled = await pg.compile((x) => {
      const out = Tensor.empty([1], { dtype: 'float32' })
      return out.customKernel(x, sumKernel)[0]
    }, [a])
    assert(compiled.scheduleCount === 1, `expected one captured schedule, got ${compiled.scheduleCount}`)
    assertClose(await (await compiled.run([a])).toArray(), [36])
    a.copyFrom(new Float32Array([2, 3, 4, 5, 6, 7, 8, 9]))
    assertClose(await (await compiled.run([a])).toArray(), [44])
    assert(compiled.stats().runCount === 2, 'compiled custom reduction should replay twice')
    compiled.dispose()
  })

  await test('compile captures current customKernel multi-output addmul', async () => {
    function addmulKernel(out0, out1, a, b) {
      out0 = out0.flatten(); out1 = out1.flatten(); a = a.flatten(); b = b.flatten()
      const i = pg.uop.range(4, 0)
      const st0 = out0.index(i).store(a.index(i).add(b.index(i)))
      const st1 = out1.index(i).store(a.index(i).mul(b.index(i)))
      return st0.group(st1).end(i).sink(
        new pg.uop.KernelInfo('custom_addmul_4')
      )
    }
    const a = Tensor.empty([4], { dtype: 'float32' })
    const b = new Tensor([1, 2, 3, 4], { dtype: 'float32' })
    a.copyFrom(new Float32Array([1, 2, 3, 4]))
    const compiled = await pg.compile((x, y) => {
      const out0 = Tensor.empty([4], { dtype: 'float32' })
      const out1 = Tensor.empty([4], { dtype: 'float32' })
      const outs = out0.customKernel(out1, x, y, addmulKernel)
      return [outs[0], outs[1]]
    }, [a, b])
    const got0 = await compiled.run([a, b])
    assertClose(await got0[0].toArray(), [2, 4, 6, 8])
    assertClose(await got0[1].toArray(), [1, 4, 9, 16])
    a.copyFrom(new Float32Array([2, 3, 4, 5]))
    const got1 = await compiled.run([a, b])
    assertClose(await got1[0].toArray(), [3, 5, 7, 9])
    assertClose(await got1[1].toArray(), [2, 6, 12, 20])
    assert(compiled.stats().runCount === 2, 'compiled custom multi-output reduction should replay twice')
    compiled.dispose()
  })

  await test('compile captures customKernel tinygrad-style set accumulator reduction', async () => {
    function sumKernel(out, x) {
      out = out.flatten(); x = x.flatten()
      const candidates = 4
      const rows = 64
      const c = pg.uop.range(candidates, 0)
      const r = pg.uop.range(rows, 1, pg.uop.AxisType.REDUCE)
      let acc = out.index(c).set(0.0)
      acc = acc.index(c).set(acc.after(r).index(c).add(x.index(c.mul(rows).add(r))), r)
      return acc.end(c).sink(new pg.uop.KernelInfo({ name: 'custom_sum_4_64', opts_to_apply: [] }))
    }
    const xData = Float32Array.from({ length: 256 }, (_, i) => i + 1)
    const expected = []
    for (let c = 0; c < 4; c++) {
      let s = 0
      for (let r = 0; r < 64; r++) s += xData[c * 64 + r]
      expected.push(s)
    }
    const x = new Tensor(xData)
    const compiled = await pg.compile((tx) => {
      const out = Tensor.empty([4], { dtype: 'float32' })
      return out.customKernel(tx, sumKernel)[0]
    }, [x])
    assertClose(await (await compiled.run([x])).toArray(), expected)
    compiled.dispose()
  })

  await test('compiled customKernel consumer reads producer output after readback', async () => {
    const n = 1024
    function producerKernel(out, x) {
      out = out.flatten(); x = x.flatten()
      const i = pg.uop.range(out.numel(), 0)
      return out.index(i).store(x.index(i).mul(2).add(1)).end(i).sink(
        new pg.uop.KernelInfo({ name: 'custom_producer_readback_rebind', opts_to_apply: [] })
      )
    }
    function consumerKernel(out, y) {
      out = out.flatten(); y = y.flatten()
      const r = pg.uop.range(n, 0, pg.uop.AxisType.REDUCE)
      let acc = out.index(0).set(0.0)
      acc = acc.index(0).set(acc.after(r).index(0).add(y.index(r)), r)
      return acc.sink(new pg.uop.KernelInfo({
        name: 'custom_consumer_readback_rebind', opts_to_apply: []
      }))
    }
    const x0 = Float32Array.from({ length: n }, (_, i) => i / 17)
    const x1 = Float32Array.from({ length: n }, (_, i) => 10 + i / 11)
    const x = new Tensor(x0, { dtype: 'float32' })
    const producer = await pg.compile((tx) => {
      const out = Tensor.empty([n], { dtype: 'float32' })
      return out.customKernel(tx, producerKernel)[0]
    }, [x])

    const firstY = await producer.run([x])
    await firstY.realize()
    const consumer = await pg.compile((ty) => {
      const out = Tensor.empty([1], { dtype: 'float32' })
      return out.customKernel(ty, consumerKernel)[0]
    }, [firstY])

    const first = await consumer.run([firstY])
    assertClose(await first.toArray(), [Array.from(x0).reduce((s, v) => s + v * 2 + 1, 0)], 1e-2)

    x.copyFrom(x1)
    const secondY = await producer.run([x])
    const second = await consumer.run([secondY])
    assertClose(await second.toArray(), [Array.from(x1).reduce((s, v) => s + v * 2 + 1, 0)], 1e-2)

    producer.dispose()
    consumer.dispose()
  })

  await test('runtime exposes conservative stats and capability checks', async () => {
    assert(typeof pg.stats === 'function', 'runtime should expose stats()')
    assert(typeof pg.canRun === 'function', 'runtime should expose canRun()')
    const stats = pg.stats()
    assert(stats.core === pg.core, 'runtime stats should include core')
    assert(stats.device === pg.device, 'runtime stats should include device')
    assert(stats.coreStats && typeof stats.coreStats.launchCount === 'number', 'runtime stats should include core counters')
    assert(typeof stats.coreStats.globalOps === 'number', 'runtime stats should include globalOps')
    assert(typeof stats.coreStats.globalMem === 'number', 'runtime stats should include globalMem')
    assert(typeof stats.coreStats.timeSumS === 'number', 'runtime stats should include timeSumS')
    assert(typeof stats.coreStats.kernelCount === 'number', 'runtime stats should include kernelCount')
    assert(typeof stats.coreStats.memUsed === 'number', 'runtime stats should include memUsed')
    assert(stats.jit && typeof stats.jit.liveCount === 'number', 'runtime stats should include jit live count')
    assert(typeof pg.resetCounters === 'function', 'runtime should expose resetCounters()')
    const liveMem = stats.coreStats.memUsed
    pg.resetCounters()
    const before = pg.stats().coreStats
    assert(before.globalOps === 0 && before.globalMem === 0 && before.kernelCount === 0,
      'resetCounters should clear execution counters')
    assert(before.memUsed === liveMem, 'resetCounters should preserve live memory')
    const x = Tensor.empty([3], { dtype: 'float32' })
    x.copyFrom(new Float32Array([1, 2, 3]))
    const t = await x.add(1).realize()
    assertClose(await t.toArray(), [2, 3, 4])
    const after = pg.stats().coreStats
    assert(after.bufferWriteBytes >= before.bufferWriteBytes + 12, 'stats should count host writes')
    assert(after.bufferReadBytes >= before.bufferReadBytes + 12, 'stats should count host reads')
    assert(after.launchCount >= before.launchCount + 1, 'stats should count backend launches')
    // Pinned tinygrad CPU:X86 computes estimates after ISA register allocation;
    // this exact add is therefore 0/0/1 there while value backends are 3/24/1.
    const expectedOps = pg.device === 'x86' ? 0 : 3
    const expectedMem = pg.device === 'x86' ? 0 : 24
    assert(after.globalOps === expectedOps, `expected ${expectedOps} global ops, got ${after.globalOps}`)
    assert(after.globalMem === expectedMem, `expected ${expectedMem} global memory bytes, got ${after.globalMem}`)
    assert(after.kernelCount === 1, `expected one tracked call, got ${after.kernelCount}`)
    assert(pg.canRun({ dtype: 'float32' }), 'float32 should be supported by every current runtime')
    if (pg.caps.f64 === false) {
      assert(!pg.canRun({ dtype: 'float64' }), 'canRun should reject f64 when caps.f64 is false')
    }
    assert(pg.canRun({ op: 'add', dtype: 'float32', shape: [4] }), 'canRun should probe add')
    assert(
      pg.canRun({ op: 'matmul', dtype: 'float32', shapes: [[2, 3], [3, 4]] }),
      'canRun should probe matmul shapes'
    )
    assert(pg.canRun({ op: 'gather', dtype: 'float32', shape: [2, 3] }), 'canRun should probe gather')
    assert(pg.canRun({ op: 'sort', dtype: 'float32', shape: [2, 3] }), 'canRun should probe sort')
    assert(pg.canRun({ op: 'argsort', dtype: 'float32', shape: [2, 3] }), 'canRun should probe argsort')
    assert(pg.canRun({ op: 'topk', dtype: 'float32', shape: [2, 3] }), 'canRun should probe topk')
    let threw = false
    try {
      pg.canRun({ shape: [4], dtype: 'float32' })
    } catch (e) {
      threw = String(e.message || e).includes('require an op')
    }
    assert(threw, 'shape-only canRun queries should fail explicitly')
  })

  await test('runtime compile wrapper warms capture and replays', async () => {
    assert(typeof pg.compile === 'function', 'runtime should expose pg.compile')
    const sample = new Tensor(new Float32Array([1, 2, 3]))
    const compiled = await pg.compile((x) => x.add(1), [sample])
    assert(compiled.scheduleCount === 1, `expected one captured schedule, got ${compiled.scheduleCount}`)
    const out = await compiled.run([new Tensor(new Float32Array([10, 20, 30]))])
    assertClose(await out.toArray(), [11, 21, 31])
    const stats = compiled.stats()
    assert(stats.captureRuns === 2, 'compile should perform two setup runs')
    assert(stats.runCount === 1, 'compiled run should update runCount')
    compiled.dispose()
  })

  await test('toTypedArray aliases flat typed readback', async () => {
    const t = new Tensor([1, 2, 3])
    const arr = await t.toTypedArray()
    assert(arr instanceof Int32Array, `expected Int32Array, got ${arr.constructor.name}`)
    assertClose(arr, [1, 2, 3])
  })

  await test('toTypedArrays batches flat typed readback', async () => {
    const t = new Tensor([1, 2, 3])
    const outs = await Tensor.toTypedArrays(t.add(1), t.mul(2))
    assert(Array.isArray(outs) && outs.length === 2, 'expected two output arrays')
    assert(outs[0] instanceof Int32Array, `expected Int32Array, got ${outs[0].constructor.name}`)
    assertClose(outs[0], [2, 3, 4])
    assertClose(outs[1], [2, 4, 6])

    const i32 = new Tensor(new Int32Array([1, 2, 3]), { dtype: 'int32' })
    const empty = Tensor.zeros([0])
    assertShape(empty.shape, [0])
    const more = await Tensor.toTypedArrays([i32, empty])
    assert(more[0] instanceof Int32Array, `expected Int32Array, got ${more[0].constructor.name}`)
    assert(more[1] instanceof Float32Array && more[1].length === 0, 'expected empty Float32Array')
    assertClose(more[0], [1, 2, 3])
  })

  await test('empty rejects named tensor keyword like tinygrad', async () => {
    let threw = false
    try {
      Tensor.empty([2, 3], { name: 'z' })
    } catch (_) {
      threw = true
    }
    assert(threw, 'expected Tensor.empty(..., {name}) to reject')
  })

  // -- Elementwise --
  console.log('\n-- Elementwise --')

  await test('add', async () => {
    const a = new Tensor([1, 2, 3])
    const b = new Tensor([4, 5, 6])
    assertClose(await a.add(b).toArray(), [5, 7, 9])
  })

  await test('sub', async () => {
    const a = new Tensor([10, 20, 30])
    const b = new Tensor([1, 2, 3])
    const out = a.sub(b)
    assert(out.uop.op === pg._core.ops.ADD, 'subtraction root must be ADD')
    assert(out.uop.src[1].op === pg._core.ops.MUL, 'subtraction rhs must be negating MUL')
    assertClose(await out.toArray(), [9, 18, 27])
  })

  await test('mul', async () => {
    const a = new Tensor([2, 3, 4])
    const b = new Tensor([5, 6, 7])
    assertClose(await a.mul(b).toArray(), [10, 18, 28])
  })

  await test('div', async () => {
    const a = new Tensor([10, 20, 30])
    const b = new Tensor([2, 4, 5])
    assertClose(await a.div(b).toArray(), [5, 5, 6])
  })

  await test('div rounding modes match tinygrad topology', async () => {
    const ints = new Tensor([-7, -4, 4, 7], { dtype: 'int32' })
    const intDivisors = new Tensor([3, -3, 3, -3], { dtype: 'int32' })
    const truncInt = ints.div(intDivisors, 'trunc')
    const floorInt = ints.div(intDivisors, 'floor')
    assert(truncInt.uop.op === pg._core.ops.CDIV, 'integer trunc division must be CDIV')
    assert(floorInt.uop.op === pg._core.ops.FLOORDIV, 'integer floor division must be FLOORDIV')
    assertClose(await truncInt.toArray(), [-2, 1, 1, -2])
    assertClose(await floorInt.toArray(), [-3, 1, 1, -3])

    const floats = new Tensor([-7.5, -4.5, 4.5, 7.5], { dtype: 'float32' })
    const floatDivisors = new Tensor([2, -2, 2, -2], { dtype: 'float32' })
    const truncFloat = floats.div(floatDivisors, 'trunc')
    const floorFloat = floats.div(floatDivisors, 'floor')
    assert(truncFloat.uop.op === pg._core.ops.TRUNC, 'float trunc division must end in TRUNC')
    assert(floorFloat.uop.op === pg._core.ops.WHERE, 'float floor division must end in WHERE')
    assertClose(await truncFloat.toArray(), [-3, 2, 2, -3])
    assertClose(await floorFloat.toArray(), [-4, 2, 2, -4])

    let threw = false
    try { ints.div(intDivisors, 'nearest') } catch (err) {
      threw = /rounding_mode='nearest' is not supported/.test(String(err.message))
    }
    assert(threw, 'unsupported division rounding mode must throw')
  })

  await test('neg', async () => {
    const a = new Tensor([1, -2, 3])
    const out = a.neg()
    assert(out.uop.op === pg._core.ops.MUL, 'negation root must be MUL')
    assertClose(await out.toArray(), [-1, 2, -3])
  })

  await test('scalar add', async () => {
    const a = new Tensor([1, 2, 3])
    assertClose(await a.add(10).toArray(), [11, 12, 13])
  })

  await test('scalar mul', async () => {
    const a = new Tensor([1, 2, 3])
    assertClose(await a.mul(3).toArray(), [3, 6, 9])
  })

  await test('chain: (a + 2) * b', async () => {
    const a = new Tensor([1, 2, 3])
    const b = new Tensor([4, 5, 6])
    assertClose(await a.add(2).mul(b).toArray(), [12, 20, 30])
  })

  // -- Composed math --
  console.log('\n-- Composed math --')

  await test('exp', async () => {
    const a = new Tensor([0, 1])
    const arr = await a.exp().toArray()
    assertClose(arr, [1, Math.E], 1e-3)
  })

  await test('log', async () => {
    const a = new Tensor([1, Math.E])
    const arr = await a.log().toArray()
    assertClose(arr, [0, 1], 1e-3)
  })

  await test('logaddexp softplus mish match pinned graph', async () => {
    assert(Tensor.prototype._physicalizeResult === undefined,
      'legacy frontend substitution helper must stay deleted')
      const x = new Tensor([
        [-20, -3, -0, 2, 20],
        [1, -1, 4, -4, 0.5]
      ])
      const other = new Tensor([[-2], [3]])
      const pinnedLogaddexp = (lhs, rhs) => {
        let b = lhs._ensureTensor(rhs)
        const shape = lhs._broadcastShape(b.shape)
        const a = lhs._broadcastTensor(shape)
        b = b._broadcastTensor(shape)
        const m = a.maximum(b)
        return a.sub(m).exp().add(b.sub(m).exp()).log().add(m)
      }
      const pinnedSoftplus = (value, beta = 1) =>
        pinnedLogaddexp(value.mul(beta), 0).mul(1 / beta, true)
      const pairs = [
        [x.logaddexp(0), pinnedLogaddexp(x, 0)],
        [x.logaddexp(other), pinnedLogaddexp(x, other)],
        [x.softplus(), pinnedSoftplus(x)],
        [x.softplus(2), pinnedSoftplus(x, 2)],
        [x.mish(), x.mul(pinnedSoftplus(x).tanh())]
      ]
      for (const [actual, expected] of pairs) {
        assert(actual.uop.key === expected.uop.key, 'physical graph differs')
        assert(actual.uopLogical.key === expected.uopLogical.key, 'logical graph differs')
        assertClose(await actual.toArray(), await expected.toArray(), 1e-6)
      }
  })

  await test('where scalar branch shapes before promotion', async () => {
    const cond = new Tensor([[true, false], [false, true]], { dtype: 'bool' })
    const out = cond.where(0, -Infinity)

    // Tinygrad 2026-08-22/a9069c177a9d mixin/elementwise.py:422-435 keeps
    // scalar branches as weak CONSTs; broadcasting is implicit.
    const zeroBranch = out.uop.src[1]
    assert(zeroBranch.op === pg._core.ops.CONST, 'expected scalar CONST branch')
    assert(pg.uop.dtype(zeroBranch) === 'weakfloat', 'expected promoted weakfloat branch')
    const values = await out.toArray()
    assert(values[0] === 0 && values[1] === -Infinity && values[2] === -Infinity && values[3] === 0,
      `unexpected where values ${Array.from(values)}`)
  })

  await test('log1p expm1 use core tensor roots', async () => {
    assert(Tensor.prototype._physicalizeResult === undefined,
      'legacy frontend substitution helper must stay deleted')
      const x = new Tensor([-1e-6, 0, 1e-6, 0.25], { device: 'cpu' })
      const rows = [
        ['log1p', x.log1p(), [-1e-6, 0, 1e-6, 0.25].map(Math.log1p)],
        ['expm1', x.expm1(), [-1e-6, 0, 1e-6, 0.25].map(Math.expm1)]
      ]
      for (const [name, actual, expectedValues] of rows) {
        const rawFn = pg._core.ffi[`poly_${name}`]
        const expectedLogical = rawFn(x._ctx, x.uopLogical.raw)
        const expectedPhysical = rawFn(x._ctx, x.uop.raw)
        assert(
          actual.uopLogical.key === String(pg._core.ffi.poly_uop_key(expectedLogical)),
          `${name} logical graph differs`
        )
        assert(
          actual.uop.key === String(pg._core.ffi.poly_uop_key(expectedPhysical)),
          `${name} physical graph differs`
        )
        assertClose(await actual.toArray(), expectedValues, 1e-6)
      }

      const moved = (await Tensor.empty([4], {
        device: 'cpu'
      }).realize()).to('cuda').to('cpu')
      for (const [name, actual] of [
        ['log1p', moved.log1p()],
        ['expm1', moved.expm1()]
      ]) {
        const expected = pg._core.ffi[`poly_${name}`](moved._ctx, moved.uop.raw)
        assert(
          actual.uop.key === String(pg._core.ffi.poly_uop_key(expected)),
          `${name} lost the nested physical occurrence`
        )
      }
  })

  await test('sin cos tan match pinned promotion', async () => {
    const integer = new Tensor([0, 1, 2], { dtype: 'int32' })
    for (const [name, actual, expected] of [
      ['sin', integer.sin(), [0, Math.sin(1), Math.sin(2)]],
      ['cos', integer.cos(), [1, Math.cos(1), Math.cos(2)]],
      ['tan', integer.tan(), [0, Math.tan(1), Math.tan(2)]]
    ]) {
      assert(actual.dtype === 'float32', `integer ${name} should promote to float32`)
      assertClose(await actual.toArray(), expected, 1e-6)
    }
    assert(integer.sin().uop.src[0].op !== pg._core.ops.CAST,
      'integer SIN should retain its original source without a frontend CAST')

    const bool = new Tensor([false, true], { dtype: 'bool' })
    assertClose(await bool.sin().toArray(), [0, Math.sin(1)], 1e-6)

    const angles = new Tensor([0, 0.25, 0.5])
    assertClose(await angles.cos().toArray(), [Math.cos(0), Math.cos(0.25), Math.cos(0.5)], 1e-6)
    assertClose(await angles.tan().toArray(), [Math.tan(0), Math.tan(0.25), Math.tan(0.5)], 1e-6)
  })

  await testIf(supportsF16, 'sin cos tan preserve float16 promotion', async () => {
    const half = new Tensor([0, 1]).cast('float16')
    const halfCos = half.cos()
    assert(halfCos.dtype === 'float16', 'float16 cos should cast back to float16')
    assertClose(await halfCos.cast('float32').toArray(), [1, Math.cos(1)], 2e-3)
  })

  await testIf(supportsF64, 'sin cos tan preserve float64 promotion', async () => {
    const angles = new Tensor([0, 0.25, 0.5], { dtype: 'float64' })
    const cos = angles.cos()
    const tan = angles.tan()
    assert(cos.dtype === 'float64', 'float64 cos should retain dtype')
    assert(tan.dtype === 'float64', 'float64 tan should retain dtype')
    assertClose(await cos.toArray(), [Math.cos(0), Math.cos(0.25), Math.cos(0.5)], 1e-12)
    assertClose(await tan.toArray(), [Math.tan(0), Math.tan(0.25), Math.tan(0.5)], 1e-12)
  })

  await test('sin cos tan preserve nested current occurrence', async () => {
    const x = await new Tensor([0.25], { device: 'cpu' }).realize()
    const moved = x.to('cuda').to('cpu')
    assert(
      countGraphOp(moved.sin().uop, pg._core.ops.COPY) === 2,
      'sin should retain the nested COPY source occurrence'
    )
    assert(
      countGraphOp(moved.cos().uop, pg._core.ops.COPY) === 2,
      'cos should retain the nested COPY source occurrence'
    )
    assert(
      countGraphOp(moved.tan().uop, pg._core.ops.COPY) === 2,
      'tan should retain the nested COPY source occurrence'
    )
  })

  await test('sqrt', async () => {
    const a = new Tensor([1, 4, 9, 16])
    assertClose(await a.sqrt().toArray(), [1, 2, 3, 4])
  })

  await test('abs', async () => {
    const a = new Tensor([-1, 2, -3])
    assertClose(await a.abs().toArray(), [1, 2, 3])
  })

  await test('square', async () => {
    const a = new Tensor([2, 3, 4])
    assertClose(await a.square().toArray(), [4, 9, 16])
  })

  await test('named reverse add and mul preserve scalar-first ordering', async () => {
    const moved = (await new Tensor([1, 2], { device: 'cpu' }).realize())
      .to('cuda').to('cpu')
    for (const out of [moved.add(3, true), moved.mul(3, true)]) {
      assert(out.uop.src[0].op === pg._core.ops.CONST,
        'reverse scalar should be the first implicit-broadcast operand')
      assert(out.uop.src[1].key === moved.uop.key,
        'nested current occurrence should be the second operand')
    }
    const values = new Tensor([1, 2], { device: 'cpu' })
    assertClose(await values.add(3, true).toArray(), [4, 5])
    assertClose(await values.mul(3, true).toArray(), [3, 6])
  })

  await test('literal elementwise composites match pinned', async () => {
    const values = [-2.5, -1, 0, 0.5, 2.5]
    const x = new Tensor(values)
    const sigmoid = v => 1 / (1 + Math.exp(-v))
    const expected = {
      square: values.map(v => v * v),
      ceil: values.map(Math.ceil),
      floor: values.map(Math.floor),
      sigmoid: values.map(sigmoid),
      tanh: values.map(Math.tanh),
      relu6: values.map(v => Math.min(Math.max(v, 0), 6)),
      leakyRelu: values.map(v => v < 0 ? 0.01 * v : v),
      hardswish: values.map(v => v * Math.min(Math.max(v + 3, 0), 6) / 6),
      hardsigmoid: values.map(v => Math.min(Math.max(v / 6 + 0.5, 0), 1)),
      hardtanh: values.map(v => Math.min(Math.max(v, -1), 1)),
      silu: values.map(v => v * sigmoid(v)),
      elu: values.map(v => v > 0 ? v : Math.exp(v) - 1),
      sign: values.map(Math.sign),
      abs: values.map(Math.abs),
      isnan: values.map(Number.isNaN)
    }
    for (const [method, wanted] of Object.entries(expected)) {
      assertClose(await x[method]().toArray(), wanted, 2e-6)
    }

    assertClose(
      await x.hardsigmoid(0.2, 0.3).toArray(),
      values.map(v => Math.min(Math.max(0.2 * v + 0.3, 0), 1)),
      2e-6
    )

    const ints = new Tensor(new Int32Array([-2, 0, 3]), { dtype: 'int32' })
    assertClose(await ints.sign().toArray(), [-1, 0, 1])
    assertClose(await ints.abs().toArray(), [2, 0, 3])
    const bools = new Tensor([false, true], { dtype: 'bool' })
    assertClose(await bools.sign().toArray(), [0, 1])
    assertClose(await bools.abs().toArray(), [0, 1])

    const constant = x.constLike(1)
    assert(constant.dtype === x.dtype, 'constLike should retain dtype')
    assertShape(constant.shape, x.shape)
    assert(constant.uop.op === pg._core.ops.EXPAND, 'constLike should remain a CONST graph')
    assertClose(await constant.toArray(), values.map(() => 1))

    for (const method of ['ceil', 'floor']) {
      const gradInput = new Tensor(values, {})
      await gradInput[method]().sum().backward()
      assertClose(await gradInput.grad.toArray(), values.map(() => 0))
    }

    const moved = (await new Tensor([0.5], { device: 'cpu' }).realize())
      .to('cuda').to('cpu')
    for (const method of ['square', 'sigmoid', 'tanh', 'relu6', 'sign', 'abs']) {
      assert(countGraphOp(moved[method]().uop, pg._core.ops.COPY) === 2,
        `${method} should retain the nested current occurrence`)
    }
  })

  await testIf(supportsF16, 'literal composites preserve float16 promotion', async () => {
    const values = [-2.5, -1, 0, 0.5, 2.5]
    const half = new Tensor(values).cast('float16')
    assert(half.sigmoid().dtype === 'float16', 'float16 sigmoid should retain dtype')
    assert(half.tanh().dtype === 'float16', 'float16 tanh should retain dtype')
    assertClose(await half.tanh().cast('float32').toArray(), values.map(Math.tanh), 2e-3)
  })

  await test('round and isinf match pinned compositions', async () => {
    // Pinned mixin/elementwise.py:872-880 is round-half-to-even.
    const values = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    const expectedRound = [-2, -2, 0, 0, 2, 2]

    const rounded32 = new Tensor(values).round()
    assert(rounded32.dtype === 'float32', `expected float32, got ${rounded32.dtype}`)
    assertClose(await rounded32.toArray(), expectedRound)

    if (supportsF16) {
      // Keep this composition gate independent of direct host-bit packing.
      const rounded16 = new Tensor(values).cast('float16').round()
      assert(rounded16.dtype === 'float16', `expected float16, got ${rounded16.dtype}`)
      assertClose(await rounded16.cast('float32').toArray(), expectedRound)
    }
    if (supportsF64) {
      const rounded64 = new Tensor(values).cast('float64').round()
      assert(rounded64.dtype === 'float64', `expected float64, got ${rounded64.dtype}`)
      assertClose(await rounded64.toArray(), expectedRound)
    }

    const roundedInt = new Tensor(
      new Int32Array([-2, -1, 0, 1, 2]), { dtype: 'int32' }
    ).round()
    assert(roundedInt.dtype === 'weakfloat', `expected weakfloat, got ${roundedInt.dtype}`)
    assertClose(await roundedInt.toArray(), [-2, -1, 0, 1, 2])

    const roundedBool = new Tensor([false, true], { dtype: 'bool' }).round()
    assert(roundedBool.dtype === 'weakfloat', `expected weakfloat, got ${roundedBool.dtype}`)
    assertClose(await roundedBool.toArray(), [0, 1])

    // Pinned mixin/elementwise.py:596-604 independently gates each sign.
    const infinityValues = new Tensor(
      [-Infinity, -1, -0, 0, 1, Infinity, Number.NaN]
    )
    for (const [detectPositive, detectNegative, expected] of [
      [false, false, [false, false, false, false, false, false, false]],
      [false, true, [true, false, false, false, false, false, false]],
      [true, false, [false, false, false, false, false, true, false]],
      [true, true, [true, false, false, false, false, true, false]]
    ]) {
      const result = infinityValues.isinf(detectPositive, detectNegative)
      assert(result.dtype === 'bool', `expected bool, got ${result.dtype}`)
      assertClose(await result.toArray(), expected)
    }

    for (const tensor of [
      new Tensor([0, 1]),
      new Tensor(new Int32Array([0, 1]), { dtype: 'int32' }),
      new Tensor([false, true], { dtype: 'bool' })
    ]) {
      const result = tensor.isinf()
      assert(result.dtype === 'bool', `expected bool, got ${result.dtype}`)
      assertClose(await result.toArray(), [false, false])
    }

    const moved = (await new Tensor([0.5], { device: 'cpu' }).realize())
      .to('cuda').to('cpu')
    assert(countGraphOp(moved.round().uop, pg._core.ops.COPY) === 2,
      'round should retain the nested current occurrence')
    for (const [detectPositive, detectNegative] of [
      [false, false], [false, true], [true, false], [true, true]
    ]) {
      assert(
        countGraphOp(
          moved.isinf(detectPositive, detectNegative).uop,
          pg._core.ops.COPY
        ) === 2,
        'isinf should retain the nested current occurrence'
      )
    }
  })

  await test('sigmoid', async () => {
    const a = new Tensor([0])
    const arr = await a.sigmoid().toArray()
    assertClose(arr, [0.5], 1e-3)
  })

  // -- Activations --
  console.log('\n-- Activations --')

  await test('relu', async () => {
    const a = new Tensor([-1, 0, 1, 2])
    assertClose(await a.relu().toArray(), [0, 0, 1, 2])
  })

  await test('gelu', async () => {
    const values = [-2.5, -1, 0, 0.5, 2.5]
    const expected = values.map(
      x => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)))
    )
    assertClose(await new Tensor(values).gelu().toArray(), expected, 2e-6)

    if (supportsF16) {
      // Isolate GELU from the separately tested direct host-bit packing path.
      const half = new Tensor(values).cast('float16').gelu()
      assert(half.dtype === 'float16', `expected float16, got ${half.dtype}`)
      assertClose(await half.cast('float32').toArray(), expected, 2e-3)
    }

    const moved = (await new Tensor(values, { device: 'cpu' }).realize())
      .to('cuda').to('cpu').gelu()
    assert(countGraphOp(moved.uop, pg._core.ops.COPY) === 2,
      'gelu should retain the nested current occurrence')
  })

  await test('quick gelu', async () => {
    // Pinned tinygrad/mixin/elementwise.py:752-753:
    // quick_gelu(x) = x * sigmoid(1.702*x).
    const values = [-2.5, -1, 0, 0.5, 2.5]
    const expected = values.map(x => x / (1 + Math.exp(-(1.702 * x))))
    assertClose(await new Tensor(values).quickGelu().toArray(), expected, 2e-6)

    if (supportsF16) {
      // Keep numeric host staging out of this API boundary exactly as GELU
      // does above: the graph-level float32 -> float16 cast is already exact.
      const half = new Tensor(values).cast('float16').quickGelu()
      assert(half.dtype === 'float16', `expected float16, got ${half.dtype}`)
      // Pinned tinygrad CUDA returns 2.462890625 for x=2.5 (0x40ed), exactly
      // matching Polygrad; the full-precision formula differs by 0.0021232.
      assertClose(await half.cast('float32').toArray(), expected, 2.5e-3)
    }

    const moved = (await new Tensor(values, { device: 'cpu' }).realize())
      .to('cuda').to('cpu').quickGelu()
    assert(countGraphOp(moved.uop, pg._core.ops.COPY) === 2,
      'quickGelu should retain the nested current occurrence')
  })

  await test('silu', async () => {
    const a = new Tensor([0])
    const arr = await a.silu().toArray()
    assert(Math.abs(arr[0]) < 0.01, `silu(0) should be ~0, got ${arr[0]}`)
  })

  // -- Comparisons --
  console.log('\n-- Comparisons --')

  await test('eq', async () => {
    const a = new Tensor([1, 2, 3])
    const b = new Tensor([1, 5, 3])
    assertClose(await a.eq(b).toArray(), [1, 0, 1])
  })

  await test('gt', async () => {
    const a = new Tensor([1, 5, 3])
    const b = new Tensor([2, 3, 3])
    assertClose(await a.gt(b).toArray(), [0, 1, 0])
  })

  await test('mixed dtype comparison and where promote like tinygrad', async () => {
    const x = new Tensor([
      [0, 1, 2, 3],
      [4, 5, 6, 7],
      [8, 9, 10, 11],
      [12, 13, 14, 15]
    ])
    const out = Tensor.full([4, 4], 7).gt(x)
      .where(x, Tensor.full([4, 4], -2)).sum(0)
    assertClose(await out.toArray(), [0, 2, 4, -3])
  })

  await test('where with optimized-away middle input keeps param slots', async () => {
    const idx = Tensor.arange(2)
    const out = idx.ge(0).where(new Tensor([1, 3]), new Tensor([7, 8]))
    assertClose(await out.toArray(), [1, 3])
  })

  await test('maximum', async () => {
    const a = new Tensor([1, 5, 3])
    const b = new Tensor([2, 3, 4])
    assertClose(await a.maximum(b).toArray(), [2, 5, 4])
  })

  await test('clamp', async () => {
    const a = new Tensor([1, 5, 3])
    assertClose(await a.clamp(2, 4).toArray(), [2, 4, 3])

    const values = new Tensor([-Infinity, -2, 2, Infinity])
    const minOnly = await values.clamp(-1, undefined).toArray()
    const maxOnly = await values.clamp(undefined, 1).toArray()
    assert(
      minOnly[0] === -1 && minOnly[1] === -1 && minOnly[2] === 2 &&
        minOnly[3] === Infinity,
      `clamp min-only mismatch: ${minOnly}`
    )
    assert(
      maxOnly[0] === -Infinity && maxOnly[1] === -2 && maxOnly[2] === 1 &&
        maxOnly[3] === 1,
      `clamp max-only mismatch: ${maxOnly}`
    )

    const clipped = new Tensor([-3, -0.5, 2]).clip(-1, 1)
    assert(clipped.uop.op === pg._core.ops.WHERE, 'clip alias must end in WHERE')
    assertClose(await clipped.toArray(), [-1, -0.5, 1])
  })

  await test('clamp preserves nested current occurrence', async () => {
    const x = await new Tensor([1], { device: 'cpu' }).realize()
    const clamped = x.to('cuda').to('cpu').clamp(-1, 1)
    assert(
      countGraphOp(clamped.uop, pg._core.ops.COPY) === 2,
      'clamp should retain the nested COPY source occurrence'
    )
    assert(
      countGraphOp(clamped.uop, pg._core.ops.WHERE) === 2,
      'two-bound clamp should contain two conditional WHERE nodes'
    )
  })

  // -- Movement --
  console.log('\n-- Movement --')

  await test('reshape', async () => {
    const t = new Tensor([1, 2, 3, 4, 5, 6]).reshape(2, 3)
    assertShape(t.shape, [2, 3])
    assertClose(await t.toArray(), [1, 2, 3, 4, 5, 6])
  })

  await test('reshape inference validates cardinality', async () => {
    assertShape(Tensor.empty(6).reshape(2, -1).shape, [2, 3])
    assertShape(Tensor.empty(0).reshape(-1, 3).shape, [0, 3])
    assertShape(Tensor.empty(0).reshape(1, 0).shape, [1, 0])
    assertShape(Tensor.empty(2, 3).reshape(null, 3).shape, [2, 3])

    for (const [shape, target, message] of [
      [[3072], [-1, 3073], 'size mismatch'],
      [[5], [2, -1], 'size mismatch'],
      [[6], [-1, -1], 'only one dimension can be inferred'],
      [[0], [0, -1], 'division by zero']
    ]) {
      let error = null
      try { Tensor.empty(...shape).reshape(...target) } catch (e) { error = e }
      assert(error && error.message.includes(message), `expected ${message}, got ${error && error.message}`)
    }
  })

  await test('flip', async () => {
    const t = new Tensor([1, 2, 3])
    assertClose(await t.flip(0).toArray(), [3, 2, 1])
  })

  await test('permute', async () => {
    const t = new Tensor([[1, 2, 3], [4, 5, 6]])  // [2,3]
    const p = t.permute(1, 0)  // [3,2]
    assertShape(p.shape, [3, 2])
    assertClose(await p.toArray(), [1, 4, 2, 5, 3, 6])
  })

  await test('expand negative and null keep original dim like tinygrad', async () => {
    const x = Tensor.arange(2, { dtype: 'int32' }).reshape(2, 1, 1, 1)
    const y = x.expand(-1, 3, 4, null)
    assertShape(y.shape, [2, 3, 4, 1])
    assertClose(Array.from(await y.toArray()).slice(0, 4), [0, 0, 0, 0])
    assertClose(Array.from(await y.toArray()).slice(12, 16), [1, 1, 1, 1])

    const img = Tensor.arange(2 * 3 * 34 * 34).reshape(2, 3, 34, 34)
    const lowX = Tensor.randint(2, { low: 0, high: 2 }).reshape(2, 1, 1, 1)
    const idxX = Tensor.arange(32, { dtype: 'int32' }).reshape(1, 1, 1, 32)
    const cropIdx = lowX.add(idxX).expand(-1, 3, img.shape[2], -1)
    assertShape(cropIdx.shape, [2, 3, 34, 32])
    assertShape(img.gather(-1, cropIdx).shape, [2, 3, 34, 32])
  })

  await test('pad', async () => {
    const t = new Tensor([1, 2, 3])
    const p = t.pad([[1, 1]])
    assertShape(p.shape, [5])
    assertClose(await p.toArray(), [0, 1, 2, 3, 0])

    const x = Tensor.arange(9, { dtype: 'float32' }).reshape(1, 1, 3, 3)
    const flat = x.pad([1, 0, 0, 1])
    assertShape(flat.shape, [1, 1, 4, 4])
    assertClose(await flat.toArray(), [0, 0, 1, 2, 0, 3, 4, 5, 0, 6, 7, 8, 0, 0, 0, 0])

    const moved = Tensor.arange(12).reshape(3, 4).pad([[-1, 2], [1, -1]])
    assertShape(moved.shape, [4, 4])
    assertClose(await moved.toArray(), [0, 4, 5, 6, 0, 8, 9, 10, 0, 0, 0, 0, 0, 0, 0, 0])

    const promoted = new Tensor([1, 2], { dtype: 'int32' }).pad([[1, 1]], 'constant', 5.5)
    assert(promoted.dtype === 'weakfloat', `expected weakfloat, got ${promoted.dtype}`)
    assertClose(await promoted.toArray(), [5.5, 1, 2, 5.5])
    const boolFill = new Tensor([1, 2], { dtype: 'int32' }).pad([[1, 1]], 'constant', true)
    assert(boolFill.dtype === 'int32', `expected int32, got ${boolFill.dtype}`)
    assertClose(await boolFill.toArray(), [1, 1, 2, 1])
  })

  await test('pad readback preserves non-float dtype', async () => {
    const p = new Tensor([1, 2, 3], { dtype: 'int32' }).pad([[1, 1]])
    assert(p.dtype === 'int32', `expected int32, got ${p.dtype}`)
    const arr = await p.toArray()
    assert(arr.constructor.name === 'Int32Array', `expected Int32Array, got ${arr.constructor.name}`)
    assertClose(arr, [0, 1, 2, 3, 0])
  })

  await test('pad readback preserves byte dtype', async () => {
    const p = new Tensor([1, 0, 1], { dtype: 'uint8' }).pad([[1, 1]])
    assert(p.dtype === 'uint8', `expected uint8, got ${p.dtype}`)
    const arr = await p.toArray()
    assert(arr instanceof Uint8Array, `expected Uint8Array-compatible view, got ${arr.constructor.name}`)
    assertClose(arr, [0, 1, 0, 1, 0])
  })

  await test('gather matches tinygrad probe', async () => {
    const t = new Tensor([[1, 2], [3, 4]])
    const idx = new Tensor(new Int32Array([0, 0, 1, 0]), { dtype: 'int32' }).reshape(2, 2)
    const out = t.gather(1, idx)
    assertShape(out.shape, [2, 2])
    assertClose(await out.toArray(), [1, 1, 4, 3])
    const x3 = Tensor.arange(24).reshape(2, 3, 4)
    const idx3 = new Tensor(new Int32Array([0, 2, 1, 0, 2, 1, 0, 2]), { dtype: 'int32' }).reshape(2, 2, 2)
    const out3 = x3.gather(1, idx3)
    assertShape(out3.shape, [2, 2, 2])
    assertClose(await out3.toArray(), [0, 9, 4, 1, 20, 17, 12, 21])
  })

  await test('oneHot matches tinygrad probe', async () => {
    const out = new Tensor(new Int32Array([0, 2, 1]), { dtype: 'int32' }).oneHot(4)
    assertShape(out.shape, [3, 4])
    assert(out.dtype === 'weakint', `expected weakint, got ${out.dtype}`)
    assertClose(await out.toArray(), [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0])
  })

  await test('tensor row indexing matches tinygrad probe', async () => {
    const idx = new Tensor(new Int32Array([-1, 0, 2]), { dtype: 'int32' })
    const out = Tensor.arange(12).reshape(3, 4).getitem(idx)
    assertShape(out.shape, [3, 4])
    assertClose(await out.toArray(), [8, 9, 10, 11, 0, 1, 2, 3, 8, 9, 10, 11])
  })

  await test('squeeze and integer indexing preserve scalar rank', async () => {
    const scalar = new Tensor(7)
    assert(scalar.squeeze() === scalar, 'scalar squeeze must be a no-op')
    assertShape(Tensor.empty([1]).squeeze(0).shape, [])
    assertShape(Tensor.empty([1, 1]).squeeze().shape, [])
    assertShape(Tensor.empty([2, 1]).squeeze(1).shape, [2])

    const indexed = Tensor.arange(2, { dtype: 'int32' }).getitem(0)
    assertShape(indexed.shape, [])
    assert(indexed.uop.op === pg._core.ops.RESHAPE, 'integer index must collapse to RESHAPE')
    assertClose(await indexed.toArray(), [0])
  })

  await test('basic indices use one shrink before dimension collapse', async () => {
    const base = Tensor.zeros(2, 1, 8, 1, 4).contiguous()
    await base.realize()
    const indexed = base.getitem(0, [0, 1], [0, 3], [0, 1], [0, 4])
    assertShape(indexed.shape, [1, 3, 1, 4])
    assert(indexed.uop.op === pg._core.ops.RESHAPE, 'basic index root must be RESHAPE')
    assert(indexed.uop.src[0].op === pg._core.ops.SHRINK, 'aggregate SHRINK must precede collapse')
    assert(countGraphOp(indexed.uop, pg._core.ops.SHRINK) === 1, 'basic indexing must emit one SHRINK')
    assertClose(await indexed.toArray(), new Array(12).fill(0))

    const injected = base.getitem(0, null, [0, 1], [1, 4], [0, 1], [0, 4])
    assertShape(injected.shape, [1, 1, 3, 1, 4])
    assert(injected.uop.op === pg._core.ops.SHRINK, 'identity final reshape must be elided')
  })

  await test('scatter matches tinygrad probe', async () => {
    const base = Tensor.zeros(3, 5)
    const idx0 = new Tensor(new Int32Array([0, 1, 2, 0]), { dtype: 'int32' }).reshape(1, 4)
    const src0 = new Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], { dtype: 'float32' }).reshape(2, 5)
    assertClose(await base.scatter(0, idx0, src0).toArray(), [1, 0, 0, 4, 0, 0, 2, 0, 0, 0, 0, 0, 3, 0, 0])

    const idx1 = new Tensor(new Int32Array([0, 1, 2, 0, 1, 4, 2, 3, 4]), { dtype: 'int32' }).reshape(3, 3)
    const src1 = new Tensor([1, 2, 3, 6, 7, 8, 9, 10, 11], { dtype: 'float32' }).reshape(3, 3)
    assertClose(await base.scatter(1, idx1, src1).toArray(), [1, 2, 3, 0, 0, 6, 7, 0, 0, 8, 0, 0, 9, 10, 11])

    const dupIdx = new Tensor(new Int32Array([1, 1, 2]), { dtype: 'int32' }).reshape(1, 3)
    const dupSrc = new Tensor([7, 9, 8]).reshape(1, 3)
    assertClose(await new Tensor([[0, 0, 0, 0]]).scatter(1, dupIdx, dupSrc).toArray(), [0, 9, 8, 0])

    const scalarIdx = new Tensor(new Int32Array([2, 3]), { dtype: 'int32' }).reshape(2, 1)
    const floatBase = Tensor.full([2, 4], 2, { dtype: 'float32' })
    assertClose(await floatBase.scatter(1, scalarIdx, 1.23, 'add').toArray(), [2, 2, 3.23, 2, 2, 2, 2, 3.23])
    assertClose(await floatBase.scatter(1, scalarIdx, 1.23, 'multiply').toArray(), [2, 2, 2.46, 2, 2, 2, 2, 2.46])

    let threw = false
    try { base.scatter(1, idx1, src1, 'sum') } catch (e) { threw = true }
    assert(threw, 'expected invalid scatter reduce string to throw')
    threw = false
    try { base.scatter(1, idx1, src1, 'add') } catch (e) { threw = true }
    assert(threw, 'expected tensor src with scatter reduce arg to throw')
  })

  await test('scatterReduce matches tinygrad probe', async () => {
    const base = new Tensor([[1, 2, 3, 4, 5]], { dtype: 'float32' })
    const idx = new Tensor(new Int32Array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]), { dtype: 'int32' }).reshape(1, 10)
    const src = new Tensor([[1, 6, 2, 7, 3, 8, 4, 9, 5, 10]], { dtype: 'float32' })
    assertClose(await base.scatterReduce(1, idx, src, 'sum').toArray(), [8, 11, 14, 17, 20])
    assertClose(await base.scatter_reduce(1, idx, src, 'prod').toArray(), [6, 28, 72, 144, 250])
    assertClose(await base.scatterReduce(1, idx, src, 'mean', false).toArray(), [3.5, 4.5, 5.5, 6.5, 7.5])
    const extremeBase = new Tensor([[-10, 20, 0, 5, 10]], { dtype: 'float32' })
    assertClose(await extremeBase.scatterReduce(1, idx, src, 'amax').toArray(), [6, 20, 8, 9, 10])
    assertClose(await extremeBase.scatterReduce(1, idx, src, 'amin').toArray(), [-10, 2, 0, 4, 5])
    let threw = false
    try { base.scatterReduce(1, idx, src, 'max') } catch (e) { threw = true }
    assert(threw, 'expected invalid scatterReduce reduction to throw')
  })

  await test('scatterReduce preserves missing lanes and duplicate indices on both axes', async () => {
    // Pinned mixin/op.py:scatter_reduce; the no-hit mask uses bool storage.
    for (const n of [3, 5, 9]) {
      const m = 2 * (n - 1)
      for (const axis of [0, 1]) {
        const at = (c, i, width) => axis === 0 ? i * 2 + c : c * width + i
        const baseData = new Array(2 * n)
        const indexData = new Int32Array(2 * m)
        const sourceData = new Array(2 * m)
        for (let c = 0; c < 2; c++) {
          for (let i = 0; i < n; i++) baseData[at(c, i, n)] = c === 0 ? i + 1 : -i - 1
          for (let j = 0; j < m; j++) {
            indexData[at(c, j, m)] = Math.floor(j / 2)
            sourceData[at(c, j, m)] = (j % 2 === 0 ? -2 : 3) + c
          }
        }
        const base = new Tensor(baseData, { dtype: 'float32' }).reshape(axis === 0 ? [n, 2] : [2, n])
        const idx = new Tensor(indexData, { dtype: 'int32' }).reshape(axis === 0 ? [m, 2] : [2, m])
        const src = new Tensor(sourceData, { dtype: 'float32' }).reshape(axis === 0 ? [m, 2] : [2, m])
        for (const includeSelf of [false, true]) {
          for (const reduction of ['sum', 'prod', 'mean', 'amax', 'amin']) {
            const expected = baseData.slice()
            for (let c = 0; c < 2; c++) {
              for (let i = 0; i < n - 1; i++) {
                const values = [-2 + c, 3 + c]
                if (includeSelf) values.unshift(baseData[at(c, i, n)])
                expected[at(c, i, n)] = reduction === 'prod' ? values.reduce((a, b) => a * b, 1)
                  : reduction === 'amax' ? Math.max(...values)
                    : reduction === 'amin' ? Math.min(...values)
                      : values.reduce((a, b) => a + b, 0) / (reduction === 'mean' ? values.length : 1)
              }
            }
            const result = base.scatterReduce(axis, idx, src, reduction, includeSelf)
            try {
              assertClose(await result.toArray(), expected)
            } catch (err) {
              throw new Error(`n=${n} axis=${axis} ${reduction} includeSelf=${includeSelf}: ${err.message}`)
            } finally {
              result.dispose()
            }
          }
        }
        base.dispose()
        idx.dispose()
        src.dispose()
      }
    }
  })

  await test('packed storage reductions preserve small integer and bool values', async () => {
    // Wide reductions cover packed storage and odd-length tails.
    for (const n of [257, 1024]) {
      for (const dtype of ['bool', 'int8', 'uint8', 'int16', 'uint16', 'float32']) {
        const values = Array.from({ length: n }, (_, i) => dtype === 'bool' ? i % 2
          : dtype === 'int8' ? i % 121 - 60 : dtype === 'uint8' ? i % 251
            : dtype === 'int16' ? (i * 173) % 60001 - 30000
              : dtype === 'uint16' ? (i * 173) % 65536 : i % 121 - 60)
        const input = new Tensor(values, { dtype })
        const hi = input.max()
        assertClose(await hi.toArray(), [Math.max(...values)])
        hi.dispose()
        input.dispose()
      }
    }
  })

  await test('scatter construction bypasses frontend substitution', async () => {
    assert(Tensor.prototype._physicalizeResult === undefined,
      'legacy frontend substitution helper must stay deleted')
      const base = new Tensor([[1, 2, 3, 4, 5]])
      const idx = new Tensor(new Int32Array([0, 1, 1, 3, 4]), { dtype: 'int32' }).reshape(1, 5)
      const src = new Tensor([[6, 7, 8, 9, 10]])
      assertClose(await base.scatter(1, idx, src).toArray(), [6, 8, 3, 9, 10])
      assertClose(await base.scatterReduce(1, idx, src, 'sum').toArray(), [7, 17, 3, 13, 15])
  })

  // -- Step slicing --
  console.log('\n-- Step slicing --')

  await test('step2 1d', async () => {
    const t = new Tensor([1, 2, 3, 4, 5, 6, 7, 8])
    const r = t.getitem({ start: 0, stop: 8, step: 2 })
    assertShape(r.shape, [4])
    assertClose(await r.toArray(), [1, 3, 5, 7])
  })

  await test('step3 1d', async () => {
    const t = new Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
    const r = t.getitem({ start: 0, stop: 9, step: 3 })
    assertShape(r.shape, [3])
    assertClose(await r.toArray(), [1, 4, 7])
  })

  await test('step2 with start stop', async () => {
    const t = new Tensor([0, 1, 2, 3, 4, 5, 6, 7])
    const r = t.getitem({ start: 1, stop: 7, step: 2 })
    assertShape(r.shape, [3])
    assertClose(await r.toArray(), [1, 3, 5])
  })

  await test('step non-divisible', async () => {
    const t = new Tensor([0, 1, 2, 3, 4, 5, 6])
    const r = t.getitem({ start: 0, stop: 7, step: 3 })
    assertShape(r.shape, [3])
    assertClose(await r.toArray(), [0, 3, 6])
  })

  await test('negative step (reverse)', async () => {
    const t = new Tensor([1, 2, 3, 4, 5, 6])
    const r = t.getitem({ step: -1 })
    assertClose(await r.toArray(), [6, 5, 4, 3, 2, 1])
  })

  await test('negative step2', async () => {
    const t = new Tensor([1, 2, 3, 4, 5, 6])
    const r = t.getitem({ step: -2 })
    assertShape(r.shape, [3])
    assertClose(await r.toArray(), [6, 4, 2])
  })

  await test('step 2d axis0', async () => {
    const t = new Tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    const r = t.getitem({ start: 0, stop: 4, step: 2 })
    assertShape(r.shape, [2, 3])
    assertClose(await r.toArray(), [0, 1, 2, 6, 7, 8])
  })

  await test('step 2d both axes', async () => {
    const t = new Tensor([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]])
    const r = t.getitem({ start: 0, stop: 4, step: 2 }, { start: 0, stop: 4, step: 2 })
    assertShape(r.shape, [2, 2])
    assertClose(await r.toArray(), [0, 2, 8, 10])
  })

  // -- Cast --
  console.log('\n-- Cast --')

  await testIf(supportsF64, 'cast float32 to float64', async () => {
    const t = new Tensor([1, 2, 3])
    const r = t.cast('float64')
    assert(r.dtype === 'float64', `expected float64, got ${r.dtype}`)
    const arr = await r.toArray()
    console.log('DEBUG cast f32->f64', Array.from(arr))
    assertClose(arr, [1, 2, 3])
  })

  await testIf(supportsF64, 'cast float64 to float32', async () => {
    const t = new Tensor([1.5, 2.5, 3.5], { dtype: 'float64' })
    const r = t.cast('float32')
    assert(r.dtype === 'float32', `expected float32, got ${r.dtype}`)
    const arr = await r.toArray()
    console.log('DEBUG cast f64->f32', Array.from(arr))
    assertClose(arr, [1.5, 2.5, 3.5])
  })

  await test('cast no-op same dtype', async () => {
    const t = new Tensor([1, 2, 3], { dtype: 'float32' })
    const r = t.cast('float32')
    assert(r === t, 'same dtype should return self')
    assertClose(await r.toArray(), [1, 2, 3])
  })

  await test('owned identity C return survives JS adoption', async () => {
    const t = new Tensor([1, 2], { dtype: 'float32', device: 'cpu' })
    t._adoptCoreTensor(t._coreToDevice('cpu'))
    assertClose(await t.toArray(), [1, 2])
  })

  await test('cast and bitcast store exact physical roots', async () => {
    const source = Tensor.arange(4, { dtype: 'uint32' })
    const casted = source.cast('uint64')
    const bitcasted = source.bitcast('float32')

    assert(casted.uopLogical.op === pg._core.ops.CAST, 'cast logical root must be CAST')
    assert(casted.uopPhysical.op === pg._core.ops.CAST, 'cast physical root must be CAST')
    assert(casted.uopLogical.src[0].key === source.uopLogical.key, 'cast logical source mismatch')
    assert(casted.uopPhysical.src[0].key === source.uopPhysical.key, 'cast physical source mismatch')
    assert(bitcasted.uopLogical.op === pg._core.ops.BITCAST, 'bitcast logical root must be BITCAST')
    assert(bitcasted.uopPhysical.op === pg._core.ops.BITCAST, 'bitcast physical root must be BITCAST')
    assert(bitcasted.uopLogical.src[0].key === source.uopLogical.key, 'bitcast logical source mismatch')
    assert(bitcasted.uopPhysical.src[0].key === source.uopPhysical.key, 'bitcast physical source mismatch')
  })

  await test('unequal-width bitcast matches pinned lane order', async () => {
    const wide = Tensor.full([8], 1, { dtype: 'uint8' }).bitcast('uint32')
    const narrow = Tensor.full([2], 1, { dtype: 'uint32' }).bitcast('uint8')
    assert(wide.shape.length === 1 && wide.shape[0] === 2, 'wide bitcast shape mismatch')
    assert(narrow.shape.length === 1 && narrow.shape[0] === 8, 'narrow bitcast shape mismatch')
    assertClose(await wide.toArray(), [0x01010101, 0x01010101], 0)
    assertClose(await narrow.toArray(), [1, 0, 0, 0, 1, 0, 0, 0], 0)
    let invalid = null
    try { Tensor.empty([3], { dtype: 'uint8' }).bitcast('uint32') } catch (err) { invalid = err }
    assert(invalid && /unsupported size in bitcast/.test(invalid.message),
      'statically non-divisible bitcast must fail in the C Tensor boundary')
    let weak = null
    try { Tensor.full([1], 1.5, { buffer: false }).bitcast('uint32') } catch (err) { weak = err }
    assert(weak && /bitcast requires concrete dtypes/.test(weak.message),
      'weak bitcast must fail at the Tensor API boundary')
  })

  await test('bitcast view assign matches current tinygrad', async () => {
    const a = new Tensor([1, 2, 3, 4], { dtype: 'float32' })
    await a.realize()
    const view = a.bitcast('uint32')
    view.assign(new Tensor(
      [0x40800000, 0x40400000, 0x40000000, 0x3f800000],
      { dtype: 'uint32' }
    ))
    await view.realize()
    assertClose(await a.toArray(), [4, 3, 2, 1], 0)
  })

  await testIf(supportsF16, 'half and double convenience', async () => {
    const t = new Tensor([1, 2, 3])
    const h = t.half()
    assert(h.dtype === 'float16', `expected float16, got ${h.dtype}`)
    assertClose(await h.toArray(), [1, 2, 3])
    if (supportsF64) {
      const d = t.double()
      assert(d.dtype === 'float64', `expected float64, got ${d.dtype}`)
      assertClose(await d.toArray(), [1, 2, 3])
    }
  })

  await testIf(supportsF64, 'cast then compute', async () => {
    const t = new Tensor([1, 2, 3]).cast('float64')
    const r = t.add(new Tensor([10, 20, 30], { dtype: 'float64' }))
    assertClose(await r.toArray(), [11, 22, 33])
  })

  // -- Triu/Tril --
  console.log('\n-- Triu/Tril --')

  await test('triu 2d', async () => {
    const t = new Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    const r = t.triu()
    const arr = await r.toArray()
    console.log('DEBUG triu', Array.from(arr))
    assertClose(arr, [1, 2, 3, 0, 5, 6, 0, 0, 9])
  })

  await test('tril 2d', async () => {
    const t = new Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    const r = t.tril()
    const arr = await r.toArray()
    console.log('DEBUG tril', Array.from(arr))
    assertClose(arr, [1, 0, 0, 4, 5, 0, 7, 8, 9])
  })

  await test('triu with diagonal', async () => {
    const t = new Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    const r = t.triu(1)
    const arr = await r.toArray()
    console.log('DEBUG triu diag', Array.from(arr))
    assertClose(arr, [0, 2, 3, 0, 0, 6, 0, 0, 0])
  })

  await test('triu/tril batched last two dims', async () => {
    const data = [
      [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
      [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
    ]
    const t = new Tensor(data)
    const upper = t.triu()
    const lower = t.tril(1)
    assertShape(upper.shape, [2, 3, 3])
    assertShape(lower.shape, [2, 3, 3])
    assertClose(await upper.toArray(), [1, 2, 3, 0, 5, 6, 0, 0, 9, 10, 11, 12, 0, 14, 15, 0, 0, 18])
    assertClose(await lower.toArray(), [1, 2, 0, 4, 5, 6, 7, 8, 9, 10, 11, 0, 13, 14, 15, 16, 17, 18])

    const z = Tensor.zeros(5, 0, 3)
    assertShape(z.triu().shape, [5, 0, 3])
    assertShape(z.tril().shape, [5, 0, 3])
    assertClose(await z.triu().toArray(), [])
  })

  await test('triu/tril use pinned composition without substitution', async () => {
    assert(Tensor.prototype._physicalizeResult === undefined,
      'legacy frontend substitution helper must stay deleted')
      const x = new Tensor([
        [1, 2, 3, 4],
        [5, 6, 7, 8]
      ])
      const upper = x.triu(-1)
      const lower = x.tril(1)
      const expectedUpper = Tensor._tri(2, 4, -1, x.device).where(x, x.constLike(0))
      const expectedLower = Tensor._tri(2, 4, 2, x.device).where(x.constLike(0), x)
      assert(upper.uop.key === expectedUpper.uop.key, 'triu physical graph differs')
      assert(lower.uop.key === expectedLower.uop.key, 'tril physical graph differs')
      assert(upper.uopLogical.key === expectedUpper.uopLogical.key, 'triu logical graph differs')
      assert(lower.uopLogical.key === expectedLower.uopLogical.key, 'tril logical graph differs')
      assertClose(await upper.toArray(), [1, 2, 3, 4, 5, 6, 7, 8])
      assertClose(await lower.toArray(), [1, 2, 0, 0, 5, 6, 7, 0])

      const moved = (await new Tensor([
        [1, 2, 3, 4],
        [5, 6, 7, 8]
      ], { device: 'cpu' }).realize()).to('cuda').to('cpu')
      assert(countGraphOp(moved.triu().uop, pg._core.ops.COPY) === 2,
        'triu lost moved occurrence')
      assert(countGraphOp(moved.tril().uop, pg._core.ops.COPY) === 2,
        'tril lost moved occurrence')
  })

  // -- Reduction --
  console.log('\n-- Reduction --')

  await test('linear vector and matrix match pinned semantics', async () => {
    const x = new Tensor(Float32Array.from({ length: 16 }, (_, i) => i / 7)).reshape(2, 2, 4)
    const vector = x.linear(
      new Tensor(new Float32Array([1, 2, 3, 4])),
      new Tensor(new Float32Array([0.5, 1, 1.5, 2]))
    )
    const matrixWeight = new Tensor(Float32Array.from({ length: 12 }, (_, i) => i / 11)).reshape(4, 3)
    const matrix = x.linear(matrixWeight, new Tensor(new Float32Array([0.25, -0.5, 0.75])))
    assertShape(vector.shape, [2, 2, 4])
    assertShape(matrix.shape, [2, 2, 3])
    assertClose(await vector.toArray(), [
      0.5, 1 + 2 / 7, 1.5 + 6 / 7, 2 + 12 / 7,
      4 / 7 + 0.5, 1 + 10 / 7, 1.5 + 18 / 7, 2 + 28 / 7,
      8 / 7 + 0.5, 1 + 18 / 7, 1.5 + 30 / 7, 2 + 44 / 7,
      12 / 7 + 0.5, 1 + 26 / 7, 1.5 + 42 / 7, 2 + 60 / 7,
    ])
    assertClose(await matrix.toArray(), [
      42 / 77 + 0.25, 48 / 77 - 0.5, 54 / 77 + 0.75,
      114 / 77 + 0.25, 136 / 77 - 0.5, 158 / 77 + 0.75,
      186 / 77 + 0.25, 224 / 77 - 0.5, 262 / 77 + 0.75,
      258 / 77 + 0.25, 312 / 77 - 0.5, 366 / 77 + 0.75,
    ])
  })

  await test('attention primitives match pinned compositions', async () => {
    const x = new Tensor([[1, 2], [3, 4]], { dtype: 'float32' })
    const repeated = x.repeatInterleave(2, 1)
    const expectedRepeat = x.reshape(2, 2, 1).expand(2, 2, 2).reshape(2, 4)
    assert(repeated.uop.key === expectedRepeat.uop.key, 'repeatInterleave graph differs')
    assertClose(await repeated.toArray(), [1, 1, 2, 2, 3, 3, 4, 4])

    assert(x.dropout(0.25) === x, 'eval dropout must return self')
    let rangeError = false
    try { x.dropout(1.1) } catch (err) { rangeError = /out of range/.test(String(err)) }
    assert(rangeError, 'dropout must reject p outside [0,1]')
    Tensor.manual_seed(11); Tensor.training = true
    const dropped = x.dropout(0.25)
    Tensor.manual_seed(11)
    const expectedDropout = Tensor.randLike(x, { dtype: 'float32', contiguous: false })
      .ge(0.25).contiguous().where(x, 0).div(0.75)
    Tensor.training = false
    // Pinned manual_seed clears RNG state but allocates a fresh seed/counter
    // occurrence, so the reset graph is structurally equal but not the same
    // UOp identity. Exact topology is gated by dropout_stateful_rng in the
    // cross-engine canonical graph corpus.
    assert(dropped.uop.key !== expectedDropout.uop.key, 'reset RNG occurrences must remain distinct')
    assertClose(await dropped.toArray(), await expectedDropout.toArray())

    const q = new Tensor(Float32Array.from({ length: 12 }, (_, i) => i / 13)).reshape(1, 2, 2, 3)
    const k = new Tensor(Float32Array.from({ length: 12 }, (_, i) => (i - 4) / 11)).reshape(1, 2, 2, 3)
    const v = new Tensor(Float32Array.from({ length: 16 }, (_, i) => (i + 1) / 17)).reshape(1, 2, 2, 4)
    const causal = q.scaledDotProductAttention(k, v, { isCausal: true })
    const qk = q.matmul(k.transpose(-2, -1), false, 'float32').div(Math.sqrt(3))
    const mask = qk.constLike(1).cast('bool').tril().where(0, -Infinity)
    const expectedCausal = qk.add(mask).cast(q.dtype).softmax(-1).matmul(v)
    assert(causal.uop.key === expectedCausal.uop.key, 'causal attention graph differs')
    assertClose(await causal.toArray(), await expectedCausal.toArray())

    const qg = new Tensor(Float32Array.from({ length: 24 }, (_, i) => i / 19)).reshape(1, 4, 2, 3)
    const gqa = qg.scaledDotProductAttention(k, v, { enableGqa: true })
    assertShape(gqa.shape, [1, 4, 2, 4])
    assertClose(await gqa.toArray(), [
      0.17793298, 0.23675652, 0.29558003, 0.35440356,
      0.18231565, 0.24113917, 0.2999627, 0.35878623,
      0.18668212, 0.24550565, 0.3043292, 0.3631527,
      0.1910204, 0.24984394, 0.30866745, 0.36749098,
      0.66590714, 0.72473067, 0.7835542, 0.8423777,
      0.6701545, 0.7289781, 0.78780156, 0.84662515,
      0.67434025, 0.7331638, 0.7919873, 0.8508109,
      0.67845434, 0.73727787, 0.79610145, 0.8549249,
    ], 1e-5)
  })

  await test('permute resolves negative axes validates and preserves identity', async () => {
    const x = new Tensor(Float32Array.from({ length: 24 }, (_, i) => i)).reshape(2, 3, 4)
    const direct = x.permute(0, 2, 1)
    const negative = x.permute(0, -1, 1)
    assert(direct.uop.key === negative.uop.key, 'negative permute graph differs')
    assertShape(direct.shape, [2, 4, 3])
    assert(x.permute(0, 1, 2) === x, 'identity permute must return self')
    let invalid = false
    try { x.permute(0, 0, 1) } catch (err) { invalid = /not a valid permutation/.test(String(err)) }
    assert(invalid, 'duplicate permute axes must fail')
  })

  await test('sum all', async () => {
    const t = new Tensor([1, 2, 3])
    const v = await t.sum().item()
    assert(Math.abs(v - 6) < 1e-4, `Expected 6, got ${v}`)
  })

  await testIf(supportsF64, 'sum supports an explicit accumulation dtype', async () => {
    const x = new Tensor(new Float32Array([1.25, -2, 0.5, 3, 0.25, -1.5])).reshape(2, 3)
    const out = x.sum(1, false, 'float64')
    assert(out.dtype === 'float64', `expected float64, got ${out.dtype}`)
    assertClose(await out.toArray(), [-0.25, 1.75])
  })

  await test('sum axis', async () => {
    const t = new Tensor([[1, 2], [3, 4]])
    const s = t.sum(1)
    assertShape(s.shape, [2])
    assertClose(await s.toArray(), [3, 7])
  })

  await test('mean', async () => {
    const t = new Tensor([2, 4, 6])
    const v = await t.mean().item()
    assert(Math.abs(v - 4) < 1e-4, `Expected 4, got ${v}`)
  })

  await test('var matches pinned expression without substitution', async () => {
    assert(Tensor.prototype._physicalizeResult === undefined,
      'legacy frontend substitution helper must stay deleted')
      const x = new Tensor([[1, 2, 4], [3, 5, 9]])
      const pinnedExpression = (axis, keepdim = false, correction = 1) => {
        const squares = x.sub(x.mean(axis, true)).square()
        const reducedShape = squares.sum(axis, true).shape
        const n = x.shape.filter((si, i) => si !== reducedShape[i]).reduce((a, b) => a * b, 1)
        const reduced = squares.sum(axis, keepdim)
        return reduced.div(reduced.constLike(n).sub(correction).relu())
      }
      for (const [axis, keepdim, correction] of [
        [1, false, 1], [0, true, 0], [null, false, 1], [[0, 1], false, 1], [1, false, 3]
      ]) {
        const actual = x.var(axis, keepdim, correction)
        const expected = pinnedExpression(axis, keepdim, correction)
        assert(actual.uop.key === expected.uop.key, 'variance physical graph differs')
        assert(actual.uopLogical.key === expected.uopLogical.key, 'variance logical graph differs')
        const actualValues = await actual.toArray()
        const expectedValues = await expected.toArray()
        if (correction === 3) {
          assert(
            actualValues.length === expectedValues.length &&
            actualValues.every((value, i) => value === expectedValues[i]),
            'variance non-finite values differ'
          )
        } else {
          assertClose(actualValues, expectedValues)
        }
      }
  })

  await test('max', async () => {
    const t = new Tensor([[1, 5], [3, 2]])
    const m = t.max(1)
    assertClose(await m.toArray(), [5, 3])
  })

  await test('argmax matches tinygrad probe', async () => {
    const t = new Tensor([[1.2, 0.5, 1.2], [2.2, 1.9, 0.0]])
    assertClose(await t.argmax(1).toArray(), [0, 0])
    assertShape(t.argmax(1, true).shape, [2, 1])
    assertClose(await t.argmax(1, true).toArray(), [0, 0])
    const flat = await t.argmax().item()
    assert(flat === 3, `expected flattened argmax 3, got ${flat}`)
    const singleton = Tensor.arange(6, { dtype: 'float32' }).reshape(2, 1, 3)
    assertClose(await singleton.argmax(1).toArray(), [0, 0, 0, 0, 0, 0])
    const empty = Tensor.empty(2, 0, 3, { device: 'cpu' })
    assertClose(
      await empty.argmax(1).toArray(),
      [-2147483648, -2147483648, -2147483648, -2147483648, -2147483648, -2147483648]
    )
  })

  await test('sort argsort topk match tinygrad probe', async () => {
    const x = new Tensor([[0.1, 0.5, 1.2, 3.4, 2.1], [2.2, 1.9, 0.3, 4.5, 0.8]])
    let pair = x.sort(1, false)
    assertShape(pair[0].shape, [2, 5])
    assertShape(pair[1].shape, [2, 5])
    assertClose(await pair[0].toArray(), [0.1, 0.5, 1.2, 2.1, 3.4, 0.3, 0.8, 1.9, 2.2, 4.5])
    assertClose(await pair[1].toArray(), [0, 1, 2, 4, 3, 2, 4, 1, 0, 3])

    pair = x.sort(1, true)
    assertClose(await pair[0].toArray(), [3.4, 2.1, 1.2, 0.5, 0.1, 4.5, 2.2, 1.9, 0.8, 0.3])
    assertClose(await pair[1].toArray(), [3, 4, 2, 1, 0, 3, 0, 1, 4, 2])

    pair = x.topk(2, 1)
    assertShape(pair[0].shape, [2, 2])
    assertShape(pair[1].shape, [2, 2])
    assertClose(await pair[0].toArray(), [3.4, 2.1, 4.5, 2.2])
    assertClose(await pair[1].toArray(), [3, 4, 3, 0])

    pair = x.topk(2, 1, false)
    assertClose(await pair[0].toArray(), [0.1, 0.5, 0.3, 0.8])
    assertClose(await pair[1].toArray(), [0, 1, 2, 4])

    const t = new Tensor([[2, 3, 4, 1], [1, 4, 3, 2]])
    assertClose(await t.argsort().toArray(), [3, 0, 1, 2, 0, 3, 2, 1])
  })

  await test('topk tie order and errors match tinygrad probe', async () => {
    const tie = new Tensor([[1.0, 1.0, 0.0, 1.0]])
    const pair = tie.topk(3, 1)
    assertClose(await pair[0].toArray(), [1.0, 1.0, 1.0])
    assertClose(await pair[1].toArray(), [0, 1, 3])

    let threw = false
    try {
      new Tensor([[0.1, 0.2]]).topk(6, 1)
    } catch (_) {
      threw = true
    }
    assert(threw, 'expected topk k out of range to throw')

    threw = false
    try {
      new Tensor([[0.1, 0.2]]).topk(1, 1, true, false)
    } catch (_) {
      threw = true
    }
    assert(threw, 'expected topk sorted_=false to throw')
  })

  await test('softmax', async () => {
    const t = new Tensor([1, 2, 3])
    const arr = await (await t.softmax()).toArray()
    const sum = arr[0] + arr[1] + arr[2]
    assert(Math.abs(sum - 1.0) < 1e-4, `Softmax sum should be 1, got ${sum}`)
  })

  await test('fusion fuzzer smoke patterns', async () => {
    const movement = new Tensor([
      [0, 1, 2, 3],
      [4, 5, 6, 7],
      [8, 9, 10, 11]
    ]).pad([[1, 0], [0, 1]]).shrink([[1, 4], [1, 5]])
      .mul(0.25).add(1).sum(1)
    assertShape(movement.shape, [3])
    assertClose(await movement.toArray(), [5.5, 8.5, 11.5], 2e-4)

    const x = new Tensor([
      [0, 1, 2, 3],
      [4, 5, 6, 7],
      [8, 9, 10, 11],
      [12, 13, 14, 15]
    ])
    const whereReduce = Tensor.full([4, 4], 7).gt(x).where(x, Tensor.full([4, 4], -2)).sum(0)
    assertShape(whereReduce.shape, [4])
    assertClose(await whereReduce.toArray(), [0, 2, 4, -3], 2e-4)

    const base = new Tensor([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9]
    ])
    const noBarrier = base.add(1)
    assertClose(await noBarrier.square().add(noBarrier).sum(0).toArray(), [108, 144, 186], 2e-4)

    const barrier = base.add(1)
    await barrier.realize()
    assertClose(await barrier.square().add(barrier).sum(0).toArray(), [108, 144, 186], 2e-4)

    const [q, r] = new Tensor([[1, 2], [-1, 3], [0.5, 4]]).qr()
    assertClose(await q.dot(r).toArray(), [1, 2, -1, 3, 0.5, 4], 2e-3)

    const [qb, rb] = new Tensor([[1, 2], [-1, 3], [0.5, 4]]).qr()
    await qb.realize(rb)
    assertClose(await qb.dot(rb).toArray(), [1, 2, -1, 3, 0.5, 4], 2e-3)

    const chol = new Tensor([[4, 2], [2, 5]]).cholesky()
    assertClose(
      await chol.choleskySolve(new Tensor([[1, 2], [3, 4]])).toArray(),
      [-0.0625, 0.125, 0.625, 0.75],
      3e-4
    )

    const cholBarrier = new Tensor([[4, 2], [2, 5]]).cholesky()
    await cholBarrier.realize()
    assertClose(
      await cholBarrier.choleskySolve(new Tensor([[1, 2], [3, 4]])).toArray(),
      [-0.0625, 0.125, 0.625, 0.75],
      3e-4
    )

    assertClose(
      await new Tensor([[0, 2], [1, 3]]).solve(new Tensor([4, 5])).toArray(),
      [-1, 2],
      4e-4
    )

    assertClose(
      await new Tensor([[1, 0], [1, 1], [1, 2]]).lstsq(new Tensor([1, 2, 3])).toArray(),
      [1, 1],
      6e-4
    )
  })

  await test('matmul', async () => {
    const a = new Tensor([[1, 2], [3, 4]])
    const b = new Tensor([[5, 6], [7, 8]])
    assertClose(await a.dot(b).toArray(), [19, 22, 43, 50])
  })

  await testIf(supportsF64, 'dot supports an explicit accumulation dtype', async () => {
    const a = new Tensor(new Float32Array([1.25, -2, 0.5, 3, 0.25, -1.5])).reshape(2, 3)
    const b = new Tensor(new Float32Array([0.5, -1, 2, 0.25, -0.75, 3])).reshape(3, 2)
    const out = a.dot(b, 'float64')
    assert(out.dtype === 'float64', `expected float64, got ${out.dtype}`)
    assertClose(await out.toArray(), [-3.75, -0.25, 3.125, -7.4375])
  })

  await test('vector dot is scalar', async () => {
    const scalar = new Tensor(new Float32Array([1, 2, 3])).dot(new Tensor(new Float32Array([4, 5, 6])))
    assert(scalar.shape.length === 0, `expected scalar shape, got ${JSON.stringify(scalar.shape)}`)
  })

  await testIf(supportsF16, 'mixed float16 float32 dot promotes to float32', async () => {
    const mixedA = new Tensor([1.25, -2, 0.5, 3, 0.25, -1.5]).reshape(2, 3).cast('float16')
    const mixedB = new Tensor([0.5, -1, 2, 0.25, -0.75, 3]).reshape(3, 2)
    const mixed = mixedA.dot(mixedB)
    assert(mixed.dtype === 'float32', `expected mixed dot float32, got ${mixed.dtype}`)
    assert(mixed.uop.op === mixed.uop.ffi.__polygradOps.REDUCE, 'expected mixed dot REDUCE')
    assertClose(await mixed.toArray(), [-3.75, -0.25, 3.125, -7.4375], 1e-5)
  })

  await test('matmul shape mismatch throws', async () => {
    const a = new Tensor([[1, 2], [3, 4]])
    const b = new Tensor([[1, 2, 3]])
    let ok = false
    try {
      const c = a.dot(b)
      ok = c.shape.length === 0  // native: returns empty-shape result
    } catch (e) { ok = true }    // wasm: throws immediately
    assert(ok, 'expected dot to fail on shape mismatch')
  })

  await test('matmul broadcast batch', async () => {
    const a = new Tensor([
      [[1, 2], [3, 4]],
      [[5, 6], [7, 8]]
    ])
    const b = new Tensor([
      [[1, 10], [100, 1000]]
    ])
    const out = a.dot(b)
    assertShape(out.shape, [2, 2, 2])
    assertClose(await out.toArray(), [201, 2010, 403, 4030, 605, 6050, 807, 8070])
  })

  await test('matmul broadcast mismatch throws', async () => {
    const a = new Tensor([
      [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
      [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    ])
    const b = new Tensor(new Array(5).fill(0).map(() => [
      [0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0]
    ]))
    let ok = false
    try {
      const c = a.dot(b)
      ok = c.shape.length === 0
    } catch (e) { ok = true }
    assert(ok, 'expected dot to fail on broadcast-mismatched shapes')
  })

  await test('einsum uses the shared native/WASM adapter contract', async () => {
    assert(Tensor.prototype._physicalizeResult === undefined,
      'legacy frontend substitution helper must stay deleted')
    const a = new Tensor([[1, 2], [3, 4]])
    const b = new Tensor([[5, 6], [7, 8]])
      const formula = 'ij,jk->ik'
      const logical = pg._core.ffi.poly_einsum(
        a._ctx, formula, [{ _uop: a._logicalUopRaw() }, { _uop: b._logicalUopRaw() }]
      ).uop
      const physical = pg._core.ffi.poly_einsum(
        a._ctx, formula, [{ _uop: a._currentUopRaw() }, { _uop: b._currentUopRaw() }]
      ).uop
      const out = Tensor.einsum(formula, a, b)
      assertShape(out.shape, [2, 2])
      assert(
        out.uopLogical.key === String(pg._core.ffi.poly_uop_key(logical)),
        'einsum logical graph differs'
      )
      assert(
        out.uop.key === String(pg._core.ffi.poly_uop_key(physical)),
        'einsum physical graph differs'
      )
      assertClose(await out.toArray(), [19, 22, 43, 50])
  })

  await test('rearrange forwards named axis sizes on native and WASM', async () => {
    assert(Tensor.prototype._physicalizeResult === undefined,
      'legacy frontend substitution helper must stay deleted')
      const source = Tensor.arange(6)
      const out = source.rearrange('(h w) -> h w', { h: 2, w: 3 })
      const expectedLogical = pg._core.ffi.poly_rearrange(
        source._ctx, '(h w) -> h w', source.uopLogical.raw, source.shape,
        { h: 2, w: 3 }
      )
      const expectedPhysical = pg._core.ffi.poly_rearrange(
        source._ctx, '(h w) -> h w', source.uop.raw, source.shape,
        { h: 2, w: 3 }
      )
      assert(
        out.uopLogical.key === String(pg._core.ffi.poly_uop_key(expectedLogical.uop)),
        'rearrange logical graph differs'
      )
      assert(
        out.uop.key === String(pg._core.ffi.poly_uop_key(expectedPhysical.uop)),
        'rearrange physical graph differs'
      )
      assertShape(out.shape, [2, 3])
      assertClose(await out.toArray(), [0, 1, 2, 3, 4, 5])

      const moved = (await Tensor.empty([6], { device: 'cpu' }).realize())
        .to('cuda').to('cpu').reshape([2, 3])
      const movedOut = moved.rearrange('h w -> w h')
      const expectedMoved = pg._core.ffi.poly_rearrange(
        moved._ctx, 'h w -> w h', moved.uop.raw, moved.shape, {}
      )
      assert(
        movedOut.uop.key === String(pg._core.ffi.poly_uop_key(expectedMoved.uop)),
        'rearrange lost the nested physical occurrence'
      )
  })

  await test('einsum and rearrange reject malformed core inputs', async () => {
    const x = new Tensor([1, 2, 3])
    let message = ''
    try {
      Tensor.einsum('i->z', x)
    } catch (error) {
      message = String(error && error.message ? error.message : error)
    }
    assert(/poly_einsum failed/.test(message), `unexpected einsum error: ${message}`)

    for (const formula of ['invalid', `${'a'.repeat(300)}->a`, 'a->a->a', '((a))->a']) {
      message = ''
      try {
        x.rearrange(formula)
      } catch (error) {
        message = String(error && error.message ? error.message : error)
      }
      assert(/poly_rearrange failed/.test(message),
        `unexpected rearrange error for ${formula.slice(0, 24)}: ${message}`)
    }
  })

  await test('linalg construction bypasses frontend substitution', async () => {
    assert(Tensor.prototype._physicalizeResult === undefined,
      'legacy frontend substitution helper must stay deleted')
      const a = new Tensor([[4, 2], [2, 5]])
      const b = new Tensor([1, 3])
      const lower = new Tensor([[2, 0], [1, 3]])
      const [q, r] = a.qr()
      assertShape(q.shape, [2, 2])
      assertShape(r.shape, [2, 2])
      assert(Number.isFinite(await q.sum().item()), 'qr result must execute')
      assertShape(lower.triangularSolve(b).shape, [2])
      const chol = a.cholesky()
      assertShape(chol.shape, [2, 2])
      assertShape(chol.choleskySolve(b).shape, [2])
      assertShape(a.solve(b).shape, [2])
      assertShape(
        new Tensor([[1, 0], [1, 1], [1, 2]]).lstsq(new Tensor([1, 2, 3])).shape,
        [2]
      )
  })

  await test('qr matches tinygrad probe', async () => {
    const cases = [
      { arr: [[1, 2], [3, 4]], q: [2, 2], r: [2, 2], flat: [1, 2, 3, 4] },
      { arr: [[1, 2], [3, 4], [5, 6]], q: [3, 3], r: [3, 2], flat: [1, 2, 3, 4, 5, 6] },
      { arr: [[1, 2, 3], [4, 5, 6]], q: [2, 2], r: [2, 3], flat: [1, 2, 3, 4, 5, 6] },
      { arr: [[0, 1], [0, 2]], q: [2, 2], r: [2, 2], flat: [0, 1, 0, 2] },
      {
        arr: [[[1, 2], [3, 4]], [[2, 0], [0, 2]]],
        q: [2, 2, 2], r: [2, 2, 2], flat: [1, 2, 3, 4, 2, 0, 0, 2]
      },
      {
        arr: [[[1, 2], [3, 4], [5, 6]], [[2, 1], [0, 3], [4, 5]]],
        q: [2, 3, 3], r: [2, 3, 2], flat: [1, 2, 3, 4, 5, 6, 2, 1, 0, 3, 4, 5]
      },
      {
        arr: [[[1, 2, 3], [4, 5, 6]], [[2, 1, 0], [0, 3, 4]]],
        q: [2, 2, 2], r: [2, 2, 3], flat: [1, 2, 3, 4, 5, 6, 2, 1, 0, 0, 3, 4]
      },
    ]
    for (const c of cases) {
      const pair = new Tensor(c.arr).qr()
      assertShape(pair[0].shape, c.q)
      assertShape(pair[1].shape, c.r)
      assertClose(await pair[0].dot(pair[1]).toArray(), c.flat, 2e-3)
      const qVals = Array.from(await pair[0].toArray())
      const rVals = Array.from(await pair[1].toArray())
      const vals = qVals.concat(rVals)
      for (const v of vals) assert(Number.isFinite(v), `expected finite QR value, got ${v}`)
    }
  })

  await test('qr reduced and r modes match reference shapes', async () => {
    const cases = [
      { arr: [[1, 2], [3, 4], [5, 6]], q: [3, 2], r: [2, 2], flat: [1, 2, 3, 4, 5, 6] },
      { arr: [[1, 2, 3], [4, 5, 6]], q: [2, 2], r: [2, 3], flat: [1, 2, 3, 4, 5, 6] },
      {
        arr: [[[1, 2], [3, 4], [5, 6]], [[2, 1], [0, 3], [4, 5]]],
        q: [2, 3, 2], r: [2, 2, 2], flat: [1, 2, 3, 4, 5, 6, 2, 1, 0, 3, 4, 5]
      },
    ]
    for (const c of cases) {
      const pair = new Tensor(c.arr).qr('reduced')
      assertShape(pair[0].shape, c.q)
      assertShape(pair[1].shape, c.r)
      assertClose(await pair[0].dot(pair[1]).toArray(), c.flat, 2e-3)
      const rOnly = new Tensor(c.arr).qr('r')
      assertShape(rOnly.shape, c.r)
    }
    let ok = false
    try { new Tensor([[1, 2], [3, 4]]).qr('raw') } catch (e) { ok = true }
    assert(ok, 'expected invalid QR mode to fail')
  })

  await test('triangularSolve matches numpy torch probe', async () => {
    const lower = [[2, 0, 0], [1, 3, 0], [-2, 0.5, 4]]
    const upper = [[2, -1, 0.5], [0, 3, 2], [0, 0, 4]]
    const lowerUnit = [[5, 0, 0], [1, 7, 0], [-2, 0.5, 9]]
    const bVec = [2, 7, 9]
    const bMat = [[2, 1], [7, 2], [9, 3]]
    const lowerBatch = [
      [[2, 0, 0], [1, 3, 0], [-2, 0.5, 4]],
      [[3, 0, 0], [1, 4, 0], [-2, 0.5, 5]]
    ]
    const bBatch = [
      [[2, 1], [7, 2], [9, 3]],
      [[3, 2], [8, 3], [10, 4]]
    ]
    const cases = [
      { a: lower, b: bVec, opts: {}, shape: [3], out: [1, 2, 2.5] },
      { a: lower, b: bMat, opts: {}, shape: [3, 2], out: [1, 0.5, 2, 0.5, 2.5, 0.9375] },
      {
        a: upper, b: bMat, opts: { upper: true }, shape: [3, 2],
        out: [0.8541666865, 0.3958333433, 0.8333333135, 0.1666666716, 2.25, 0.75]
      },
      {
        a: lower, b: bMat, opts: { transposeA: true }, shape: [3, 2],
        out: [2.2708332539, 0.9791666865, 1.9583333731, 0.5416666865, 2.25, 0.75]
      },
      {
        a: upper, b: bMat, opts: { upper: true, transposeA: true }, shape: [3, 2],
        out: [1, 0.5, 2.6666667461, 0.8333333135, 0.7916666865, 0.2708333433]
      },
      {
        a: lowerUnit, b: bMat, opts: { unitDiagonal: true }, shape: [3, 2],
        out: [2, 1, 5, 1, 10.5, 4.5]
      },
      {
        a: lowerBatch, b: bBatch, opts: {}, shape: [2, 3, 2],
        out: [1, 0.5, 2, 0.5, 2.5, 0.9375, 1, 0.6666666865, 1.75, 0.5833333135, 2.2249999046, 1.0083333254]
      },
      {
        a: lowerBatch, b: bVec, opts: {}, shape: [2, 3],
        out: [1, 2, 2.5, 0.6666666865, 1.5833333731, 1.9083333015]
      },
    ]
    for (const c of cases) {
      const x = new Tensor(c.a).triangularSolve(new Tensor(c.b), c.opts)
      assertShape(x.shape, c.shape)
      assertClose(await x.toArray(), c.out, 2e-4)
    }
  })

  await test('cholesky matches numpy torch probe', async () => {
    const cases = [
      { a: [[4]], shape: [1, 1], out: [2] },
      { a: [[4, 2], [2, 5]], shape: [2, 2], out: [2, 0, 1, 2] },
      {
        a: [[6, 2, 1], [2, 5, 2], [1, 2, 4]],
        shape: [3, 3],
        out: [2.4494898319, 0, 0, 0.8164966106, 2.0816659927, 0, 0.4082483053, 0.8006407619, 1.7867029905]
      },
      {
        a: [[4, 0, 0, 0], [0, 4, 0, 0], [0, 0, 4, 0], [0, 0, 0, 4]],
        shape: [4, 4],
        out: [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2]
      },
      {
        a: [[[4, 2], [2, 5]], [[9, 3], [3, 2]]],
        shape: [2, 2, 2],
        out: [2, 0, 1, 2, 3, 0, 1, 1]
      },
    ]
    for (const c of cases) {
      const l = new Tensor(c.a).cholesky()
      assertShape(l.shape, c.shape)
      assertClose(await l.toArray(), c.out, 2e-4)
    }
    const u = new Tensor([[4, 2], [2, 5]]).cholesky({ upper: true })
    assertShape(u.shape, [2, 2])
    assertClose(await u.toArray(), [2, 1, 0, 2], 2e-4)
  })

  await test('choleskySolve matches torch probe', async () => {
    const a = [[4, 2], [2, 5]]
    const b = [[1, 2], [3, 4]]
    for (const upper of [false, true]) {
      const f = new Tensor(a).cholesky({ upper })
      const x = f.choleskySolve(new Tensor(b), { upper })
      assertShape(x.shape, [2, 2])
      assertClose(await x.toArray(), [-0.0625, 0.125, 0.625, 0.75], 2e-4)
    }
    const ab = [a, [[9, 3], [3, 2]]]
    const bbVec = [[1, 3], [2, 4]]
    for (const upper of [false, true]) {
      const f = new Tensor(ab).cholesky({ upper })
      const x = f.choleskySolve(new Tensor(bbVec), { upper })
      assertShape(x.shape, [2, 2])
      assertClose(await x.toArray(), [-0.0625, 0.625, -0.8888889, 3.3333333], 5e-4)

      const xb = f.choleskySolve(new Tensor([1, 4]), { upper })
      assertShape(xb.shape, [2, 2])
      assertClose(await xb.toArray(), [-0.1875, 0.875, -1.1111112, 3.6666667], 6e-4)
    }
  })

  await test('solve matches numpy torch probe', async () => {
    const a = [[2, 1], [1, 3]]
    const bVec = [1, 4]
    const bMat = [[1, 2], [3, 4]]
    const xVec = new Tensor(a).solve(new Tensor(bVec))
    assertShape(xVec.shape, [2])
    assertClose(await xVec.toArray(), [-0.2, 1.4], 3e-4)

    const xMat = new Tensor(a).solve(new Tensor(bMat))
    assertShape(xMat.shape, [2, 2])
    assertClose(await xMat.toArray(), [0, 0.4, 1, 1.2], 3e-4)

    const pivotA = [[0, 2], [1, 3]]
    const pivotBVec = [4, 5]
    const pivotBMat = [[4, 1], [5, 2]]
    const xpVec = new Tensor(pivotA).solve(new Tensor(pivotBVec))
    assertShape(xpVec.shape, [2])
    assertClose(await xpVec.toArray(), [-1, 2], 3e-4)

    const xpMat = new Tensor(pivotA).solve(new Tensor(pivotBMat))
    assertShape(xpMat.shape, [2, 2])
    assertClose(await xpMat.toArray(), [-1, 0.5, 2, 0.5], 3e-4)

    const ab = [a, [[3, 1], [1, 4]]]
    const bb = [bMat, [[2, 3], [4, 5]]]
    const xb = new Tensor(ab).solve(new Tensor(bb))
    assertShape(xb.shape, [2, 2, 2])
    assertClose(await xb.toArray(), [0, 0.4, 1, 1.2, 0.3636363745, 0.6363636255, 0.9090909362, 1.0909091234], 3e-4)

    const bbVec = [bVec, [2, 5]]
    const xbVec = new Tensor(ab).solve(new Tensor(bbVec))
    assertShape(xbVec.shape, [2, 2])
    assertClose(await xbVec.toArray(), [-0.2, 1.4, 0.27272728, 1.1818182], 4e-4)

    const pivotAB = [pivotA, [[3, 1], [0, 2]]]
    const pivotBBVec = [pivotBVec, [7, 4]]
    const xpbVec = new Tensor(pivotAB).solve(new Tensor(pivotBBVec))
    assertShape(xpbVec.shape, [2, 2])
    assertClose(await xpbVec.toArray(), [-1, 2, 1.6666667, 2], 4e-4)

    const xbBroadcastVec = new Tensor(ab).solve(new Tensor(bVec))
    assertShape(xbBroadcastVec.shape, [2, 2])
    assertClose(await xbBroadcastVec.toArray(), [-0.2, 1.4, 0, 1], 4e-4)

    const xbSingletonMatrix = new Tensor(ab).solve(new Tensor([bMat]))
    assertShape(xbSingletonMatrix.shape, [2, 2, 2])
    assertClose(await xbSingletonMatrix.toArray(), [0, 0.4, 1, 1.2, 0.09090909, 0.36363637, 0.7272727, 0.9090909], 5e-4)

    let ok = false
    try {
      new Tensor([[1, 1, 1], [1, 1, 1]]).solve(new Tensor([1, 1]))
    } catch (e) { ok = true }
    assert(ok, 'expected solve to reject non-square A')
  })

  await test('lstsq matches numpy torch probe', async () => {
    const a = [[1, 0], [1, 1], [1, 2]]
    const bVec = [1, 2, 2.5]
    const bMat = [[1, 0.5], [2, 1], [2.5, 1.5]]
    const xVec = new Tensor(a).lstsq(new Tensor(bVec))
    assertShape(xVec.shape, [2])
    assertClose(await xVec.toArray(), [1.0833334, 0.75], 5e-4)

    const xMat = new Tensor(a).lstsq(new Tensor(bMat))
    assertShape(xMat.shape, [2, 2])
    assertClose(await xMat.toArray(), [1.0833334, 0.5, 0.75, 0.5], 5e-4)

    const ab = [a, [[1, 0], [1, 1.5], [1, 3]]]
    const bb = [bMat, [[1.25, 0.75], [2.25, 1.25], [2.75, 1.75]]]
    const xb = new Tensor(ab).lstsq(new Tensor(bb))
    assertShape(xb.shape, [2, 2, 2])
    assertClose(await xb.toArray(), [1.0833334, 0.5, 0.75, 0.5, 1.3333334, 0.75, 0.5, 0.33333334], 6e-4)

    const bbVec = [bVec, [1.25, 2.25, 2.75]]
    const xbVec = new Tensor(ab).lstsq(new Tensor(bbVec))
    assertShape(xbVec.shape, [2, 2])
    assertClose(await xbVec.toArray(), [1.0833334, 0.75, 1.3333334, 0.5], 6e-4)

    const xbBroadcastVec = new Tensor(ab).lstsq(new Tensor(bVec))
    assertShape(xbBroadcastVec.shape, [2, 2])
    assertClose(await xbBroadcastVec.toArray(), [1.0833334, 0.75, 1.0833334, 0.5], 6e-4)

    const wide = [[1, 2, 0], [0, 1, 1]]
    const wideVec = new Tensor(wide).lstsq(new Tensor([1, 2]))
    assertShape(wideVec.shape, [3])
    assertClose(await wideVec.toArray(), [-0.33333334, 0.6666667, 1.3333334], 8e-4)

    const wideMat = new Tensor(wide).lstsq(new Tensor([[1, 3], [2, 4]]))
    assertShape(wideMat.shape, [3, 2])
    assertClose(
      await wideMat.toArray(),
      [-0.33333334, -0.33333334, 0.6666667, 1.6666666, 1.3333334, 2.3333333],
      1e-3
    )

    const rankSquareVec = new Tensor([[1, 1], [2, 2]]).lstsq(new Tensor([3, 6]))
    assertShape(rankSquareVec.shape, [2])
    assertClose(await rankSquareVec.toArray(), [1.5, 1.5], 2e-3)

    const rankSquareMat = new Tensor([[1, 1], [2, 2]]).lstsq(new Tensor([[3, 1], [6, 2]]))
    assertShape(rankSquareMat.shape, [2, 2])
    assertClose(await rankSquareMat.toArray(), [1.5, 0.5, 1.5, 0.5], 2e-3)

    const rankTall = new Tensor([[1, 1], [2, 2], [3, 3]]).lstsq(new Tensor([1, 2, 3]))
    assertShape(rankTall.shape, [2])
    assertClose(await rankTall.toArray(), [0.5, 0.5], 2e-3)

    const rankWide = new Tensor([[1, 1, 0], [2, 2, 0]]).lstsq(new Tensor([3, 6]))
    assertShape(rankWide.shape, [3])
    assertClose(await rankWide.toArray(), [1.5, 1.5, 0], 2e-3)

    let ok = false
    try {
      new Tensor([[1, 1, 1], [1, 1, 1]]).lstsq(new Tensor([1, 1, 1]))
    } catch (e) { ok = true }
    assert(ok, 'expected lstsq to reject incompatible RHS rows')
  })

  await test('crossEntropy with sparse targets', async () => {
    const logits = new Tensor([[0, 0, 0], [0, 0, 0]])
    const target = new Tensor([0, 2], { dtype: 'int32' })
    const loss = await logits.crossEntropy(target)
    assertShape(loss.shape, [])
    assertClose(await loss.toArray(), [Math.log(3)])
  })

  await test('crossEntropy with dense targets', async () => {
    const logits = new Tensor([[0, 0, 0], [0, 0, 0]])
    const target = new Tensor([[1, 0, 0], [0, 0, 1]])
    const loss = await logits.crossEntropy(target)
    assertShape(loss.shape, [])
    assertClose(await loss.toArray(), [Math.log(3)])
  })

  await test('crossEntropy matches pinned expression without substitution', async () => {
    assert(Tensor.prototype._physicalizeResult === undefined,
      'legacy frontend substitution helper must stay deleted')
      const logits = new Tensor([[-1, 2, -3], [1, -2, 3]])
      const sparse = new Tensor([1, 2], { dtype: 'int32' })
      const dense = new Tensor([[0, 1, 0], [0, 0, 1]])
      const pinnedExpression = (target, reduction = 'mean', labelSmoothing = 0) => {
        const classesDim = 1
        if (JSON.stringify(logits.shape) !== JSON.stringify(target.shape)) {
          target = target.unsqueeze(classesDim)._oneHotAlongDim(
            logits.shape[classesDim], classesDim
          )
        }
        target = target.mul(1 - labelSmoothing).add(
          labelSmoothing / target.shape[classesDim]
        )
        const reduced = logits.logSoftmax(classesDim).mul(target).sum(classesDim)
        if (reduction === 'none') return reduced.neg()
        if (reduction === 'sum') return reduced.sum().neg()
        if (reduction === 'mean') return reduced.mean().neg()
        throw new Error(`invalid reduction ${reduction}`)
      }
      const pairs = []
      for (const [target, reduction, labelSmoothing] of [
        [sparse, 'mean', 0],
        [dense, 'mean', 0],
        [dense, 'none', 0],
        [dense, 'sum', 0],
        [dense, 'mean', 0.2]
      ]) {
        const actual = logits.crossEntropy(target, reduction, labelSmoothing)
        const expected = pinnedExpression(target, reduction, labelSmoothing)
        assert(actual.uop.key === expected.uop.key, 'crossEntropy physical graph differs')
        assert(actual.uopLogical.key === expected.uopLogical.key, 'crossEntropy logical graph differs')
        pairs.push([actual, expected])
      }
      for (const [actual, expected] of pairs) {
        assertClose(await actual.toArray(), await expected.toArray())
      }
  })

  await test('crossEntropy with sparse targets on non-last axis', async () => {
    const logits = new Tensor([
      [[0, 0], [0, 0], [0, 0]],
      [[0, 0], [0, 0], [0, 0]]
    ])
    const target = new Tensor([
      [0, 2],
      [1, 0]
    ], { dtype: 'int32' })
    const loss = await logits.crossEntropy(target, 'mean', 0, -2)
    assertShape(loss.shape, [])
    assertClose(await loss.toArray(), [Math.log(3)])
  })

  await test('crossEntropy default matches tinygrad class axis', async () => {
    const logits = new Tensor([
      [[0, 0], [0, 0], [0, 0]],
      [[0, 0], [0, 0], [0, 0]]
    ])
    const target = new Tensor([
      [0, 2],
      [1, 0]
    ], { dtype: 'int32' })
    const loss = await logits.crossEntropy(target)
    assertShape(loss.shape, [])
    assertClose(await loss.toArray(), [Math.log(3)])
  })

  await test('crossEntropy with dense targets on non-last axis', async () => {
    const logits = new Tensor([
      [[0, 0], [0, 0], [0, 0]],
      [[0, 0], [0, 0], [0, 0]]
    ])
    const target = new Tensor([
      [[1, 0], [0, 0], [0, 1]],
      [[0, 1], [1, 0], [0, 0]]
    ])
    const loss = await logits.crossEntropy(target, 'mean', 0, 1)
    assertShape(loss.shape, [])
    assertClose(await loss.toArray(), [Math.log(3)])
  })

  await test('crossEntropy shape mismatch throws', async () => {
    const logits = new Tensor([[0, 0, 0], [0, 0, 0]])
    const target = new Tensor([[1, 0], [0, 1]])
    let ok = false
    try {
      const loss = await logits.crossEntropy(target)
      ok = loss.shape.length === 0
    } catch (e) { ok = true }
    assert(ok, 'expected crossEntropy to fail on shape mismatch')
  })

  // -- Autograd --
  console.log('\n-- Autograd --')

  await test('current autograd API has no requiresGrad constructor flag', async () => {
    let rejected = false
    try {
      new Tensor([1], { dtype: 'float32', requiresGrad: true })
    } catch (e) {
      rejected = e instanceof TypeError && e.message.includes('requiresGrad')
    }
    assert(rejected, 'requiresGrad must be rejected like current tinygrad requires_grad')
  })

  await test('backward targets every live reachable floating Tensor', async () => {
    const x = new Tensor([2], { dtype: 'float32' })
    const y = x.square()
    const loss = y.sum()
    await loss.backward()
    assertClose(await x.grad.toArray(), [4])
    assertClose(await y.grad.toArray(), [1])
    assertClose(await loss.grad.toArray(), [1])
  })

  await test('grad: mul sum', async () => {
    const a = new Tensor([1, 2, 3], { dtype: 'float32' })
    const b = new Tensor([4, 5, 6])
    const loss = a.mul(b).sum()
    await loss.backward()
    assert(a.grad, 'grad is null')
    assertClose(await a.grad.toArray(), [4, 5, 6])
  })

  await test('grad: neg sum', async () => {
    const a = new Tensor([1, 2, 3], { dtype: 'float32' })
    const loss = a.neg().sum()
    await loss.backward()
    assert(a.grad, 'grad is null')
    assertClose(await a.grad.toArray(), [-1, -1, -1])
  })

  await test('grad: matmul backward', async () => {
    const W = new Tensor([[1, 2], [3, 4]], { dtype: 'float32' })
    const x = new Tensor([[1, 0]])
    const loss = x.dot(W).sum()
    await loss.backward()
    assert(W.grad, 'W.grad is null')
    // dL/dW = x^T @ ones = [[1,1],[0,0]]
    assertClose(await W.grad.toArray(), [1, 1, 0, 0])
  })

  await test('grad: relu backward', async () => {
    const a = new Tensor([-1, 2, -3, 4], { dtype: 'float32' })
    const loss = a.relu().sum()
    await loss.backward()
    assert(a.grad, 'grad is null')
    // relu grad: 0 where input<=0, 1 where input>0
    assertClose(await a.grad.toArray(), [0, 1, 0, 1])
  })

  await test('grad: chain backward', async () => {
    const a = new Tensor([1, 2, 3], { dtype: 'float32' })
    const loss = a.mul(a).sum()  // d/da(a^2) = 2a
    await loss.backward()
    assert(a.grad, 'grad is null')
    assert(a.grad.uopPhysical, 'gradient must store its exact physical root')
    assert(a.grad.uopPhysical.op === pg._core.ops.ADD, 'square gradient physical root must be ADD')
    assert(a.grad.uopPhysical.key === a.grad.uop.key, 'gradient current root must be physical')
    assertClose(await a.grad.toArray(), [2, 4, 6])
  })

  await test('grad: backward uses current physical value after copyFrom', async () => {
    const weight = new Tensor([1], { dtype: 'float32' }).mul(2)
    await weight.realize()
    const physicalBuffer = weight.uopPhysical.buffer.raw
    pg._core.ffi.poly_buffer_ensure_device_allocated(
      weight._ctx, physicalBuffer, pg._core.deviceIds[weight.device.toLowerCase()]
    )
    pg._core.ffi.poly_buffer_write(weight._ctx, physicalBuffer, new Float32Array([3]))
    const loss = weight.square().sum()
    await loss.backward()
    assert(weight.grad, 'weight.grad is null')
    assertClose(await weight.grad.toArray(), [6])
  })

  // -- Assign --
  console.log('\n-- Assign --')

  await test('assign basic', async () => {
    const a = new Tensor([1, 2, 3])
    a.assign(a.add(10))
    await a.realize()
    assertClose(await a.toArray(), [11, 12, 13])
  })

  await test('assign broadcasts rhs in core', async () => {
    const target = Tensor.zeros([2, 3])
    target.assign(new Tensor([4, 5, 6], { dtype: 'float32' }))
    await target.realize()
    assertClose(await target.toArray(), [4, 5, 6, 4, 5, 6])
  })

  await test('assign rejects device mismatch', async () => {
    const a = new Tensor([1], { device: 'cpu' })
    const v = new Tensor([5], { device: 'cpu' }).to('cuda')
    let ok = false
    try {
      a.assign(v)
    } catch (e) {
      ok = /assign device mismatch CPU != CUDA/.test(String(e && e.message ? e.message : e))
    }
    assert(ok, 'expected assign device mismatch CPU != CUDA')
  })

  await test('assign rejects dtype mismatch', async () => {
    const a = new Tensor([1], { dtype: 'float32' })
    const mismatchDtype = supportsF64 ? 'float64' : 'int32'
    const v = new Tensor([5], { dtype: mismatchDtype })
    let ok = false
    try {
      a.assign(v)
    } catch (e) {
      ok = new RegExp(`assign dtype mismatch float32 != ${mismatchDtype}`).test(
        String(e && e.message ? e.message : e)
      )
    }
    assert(ok, `expected assign dtype mismatch float32 != ${mismatchDtype}`)
  })

  await test('assign to same-device place keeps place target', async () => {
    const a = new Tensor([1], { device: 'cpu' }).to('cuda')
    const v = new Tensor([5], { device: 'cpu' }).to('cuda')
    assert(a.assign(v) === a, 'assign should return self')
    assert(a.device === 'CUDA', 'assign should keep CUDA placement')
    assert(!a.uop.hasBufferIdentity(), 'assign before realize should be an effect graph')
  })

  await test('assign realized targets reuse current buffer', async () => {
    const a = new Tensor([1])
    await a.realize()
    const aBuffer = a.uop.buffer.key

    // assign() creates an effect graph first. Realizing that effect writes the
    // existing target storage and returns to the same current buffer root.
    a.assign(new Tensor([5]))
    assert(!a.uop.hasBufferIdentity(), 'assign before realize should be an effect graph')
    await a.realize()
    assert(a.uop.buffer.key === aBuffer, 'realized assign should reuse target buffer')
    assertClose(await a.toArray(), [5])

    const x = await new Tensor([1]).add(1).realize()
    const xBuffer = x.uop.buffer.key
    x.assign(new Tensor([9]))
    await x.realize()
    assert(x.uop.buffer.key === xBuffer, 'realized expression assign should reuse target buffer')
    assertClose(await x.toArray(), [9])
  })

  await test('shared-storage to copies and preserves source across assign', async () => {
    const source = await new Tensor([1, 2, 3], { device: 'cpu' }).realize()
    const sourceBuffer = source.uop.buffer.key
    const target = await source.to('interp').realize()
    const targetBuffer = target.uop.buffer.key

    assert(targetBuffer !== sourceBuffer, 'cross-device to should allocate a distinct target buffer')
    assertClose(await source.toArray(), [1, 2, 3])
    assertClose(await target.toArray(), [1, 2, 3])

    for (const values of [[9, 8, 7], [4, 5, 6]]) {
      target.assign(new Tensor(values, { device: 'interp' }))
      await target.realize()
      assert(target.uop.buffer.key === targetBuffer, 'assign should retain the copied target buffer')
      assertClose(await source.toArray(), [1, 2, 3])
      assertClose(await target.toArray(), values)
    }
  })

  await test('copyFrom preserves buffer identity and updates JIT replay input', async () => {
    const x = Tensor.empty([3], { dtype: 'float32' })
    const bufferKey = x.uop.buffer.key
    x.copyFrom(new Float32Array([1, 2, 3]))
    assert(x.uop.buffer.key === bufferKey, 'copyFrom should preserve input buffer identity')
    assertClose(await x.toArray(), [1, 2, 3])

    const f = pg.jit((a) => a.add(1).realize())
    assertClose(await (await f(x)).toArray(), [2, 3, 4])
    pg.resetCounters()
    const captured = await f(x)
    let counterStats = pg.stats().coreStats
    const expectedOps = pg.device === 'x86' ? 0 : 3
    const expectedMem = pg.device === 'x86' ? 0 : 24
    assert(counterStats.globalOps === expectedOps && counterStats.globalMem === expectedMem && counterStats.kernelCount === 1,
      'JIT capture execution should update counters exactly once')
    assertClose(await captured.toArray(), [2, 3, 4])
    assert(f.scheduleCount === 1, `expected one captured schedule, got ${f.scheduleCount}`)

    const replayBufferKey = x.uop.buffer.key
    x.updateFrom(new Float32Array([10, 20, 30]))
    assert(x.uop.buffer.key === replayBufferKey, 'updateFrom should preserve captured input buffer identity')
    pg.resetCounters()
    const replayed = await f(x)
    counterStats = pg.stats().coreStats
    assert(counterStats.globalOps === expectedOps && counterStats.globalMem === expectedMem && counterStats.kernelCount === 1,
      'JIT replay should update counters exactly once')
    assertClose(await replayed.toArray(), [11, 21, 31])
    f.dispose()
  })

  await test('pure movement view realize is zero call', async () => {
    const x = await Tensor.arange(8, { dtype: 'float32' }).realize()
    const out = x.reshape(2, 4).permute(1, 0)
    const memBefore = pg.stats().coreStats.memUsed

    pg.resetCounters()
    await out.realize()
    const stats = pg.stats().coreStats
    assert(
      stats.globalOps === 0 && stats.globalMem === 0 && stats.kernelCount === 0,
      `pure view realize should execute zero calls, got ${stats.globalOps}/${stats.globalMem}/${stats.kernelCount}`
    )
    assert(stats.memUsed === memBefore, 'pure view realize should not allocate storage')
    assertClose(await out.toArray(), [0, 4, 1, 5, 2, 6, 3, 7])
  })

  await test('realized contiguous and readback reuse current buffer identity', async () => {
    // Pinned arange is device-free and realize() is a no-op. Use host-backed
    // input so this case exercises a realized ADD and its zero-copy views.
    const source = await new Tensor([0, 1, 2, 3, 4, 5, 6, 7], { dtype: 'float32' })
      .add(1).preserveLogical().realize()
    const sourceCurrent = source.uop.key
    const sourceLogical = source.uopLogical.key
    assert(source.uopLogical.op === pg._core.ops.ADD, 'source logical root should retain ADD provenance')
    assert(source.uop.hasBufferIdentity(), 'realized source should have buffer identity')

    const out = source.contiguous()
    assert(out !== source, 'contiguous should return a new Tensor object')
    assert(out.uopLogical.key === sourceLogical, 'device-free logical result should fold contiguous')
    assert(out.uopPhysical && out.uopPhysical.key === sourceCurrent,
      'physical result should reuse the exact current buffer')
    assert(out.uop.key === sourceCurrent, 'current result should reuse the exact current buffer')
    assert(source.uopLogical.key === sourceLogical, 'source logical provenance should be unchanged')

    pg.resetCounters()
    await out.realize()
    let stats = pg.stats().coreStats
    assert(stats.globalOps === 0 && stats.globalMem === 0 && stats.kernelCount === 0,
      `realized contiguous should execute zero calls, got ${stats.globalOps}/${stats.globalMem}/${stats.kernelCount}`)
    pg.resetCounters()
    assertClose(await out.toArray(), [1, 2, 3, 4, 5, 6, 7, 8])
    stats = pg.stats().coreStats
    assert(stats.globalOps === 0 && stats.globalMem === 0 && stats.kernelCount === 0,
      `realized readback should execute zero calls, got ${stats.globalOps}/${stats.globalMem}/${stats.kernelCount}`)

    const reshaped = source.reshape(2, 4)
    assert(reshaped.uop.hasBufferIdentity(), 'reshape of current buffer should retain identity')
    pg.resetCounters()
    assertClose(await reshaped.toArray(), [1, 2, 3, 4, 5, 6, 7, 8])
    stats = pg.stats().coreStats
    assert(stats.globalOps === 0 && stats.globalMem === 0 && stats.kernelCount === 0,
      'reshape readback should execute zero calls')

    const permuted = source.reshape(2, 4).permute(1, 0)
    pg.resetCounters()
    assertClose(await permuted.toArray(), [1, 5, 2, 6, 3, 7, 4, 8])
    assert(pg.stats().coreStats.kernelCount === 1, 'noncontiguous permute should materialize once')

    const casted = source.cast('int32')
    pg.resetCounters()
    assertClose(await casted.toArray(), [1, 2, 3, 4, 5, 6, 7, 8])
    assert(pg.stats().coreStats.kernelCount === 1, 'lazy cast should materialize once')
  })

  // -- Static constructors --
  console.log('\n-- Static constructors --')

  await test('zeros', async () => {
    const t = Tensor.zeros(3)
    assertClose(await t.toArray(), [0, 0, 0])
  })

  await test('ones', async () => {
    const t = Tensor.ones(2, 2)
    assertClose(await t.toArray(), [1, 1, 1, 1])
  })

  await test('full', async () => {
    const t = Tensor.full([3], 7)
    assert(t.dtype === 'int32', `buffered integer full should commit int32 storage, got ${t.dtype}`)
    assert(t.uop.op === pg._core.ops.AFTER, 'buffered full should produce AFTER')
    assert(t.uop.src[0].op === pg._core.ops.BUFFER, 'buffered full should own BUFFER storage')
    assert(t.uop.src[1].op === pg._core.ops.STORE, 'buffered full should contain STORE')
    assert(t.uop.src[1].src[0].key === t.uop.src[0].key, 'STORE must target the same BUFFER')
    assert(t.uop.src[1].src[1].op === pg._core.ops.EXPAND, 'STORE value should stay EXPAND')
    assertClose(await t.toArray(), [7, 7, 7])

    const raw = Tensor.full([3], 7, { buffer: false })
    assert(raw.dtype === 'weakint', `unbuffered integer full should stay weakint, got ${raw.dtype}`)
    assert(raw.uop.op === pg._core.ops.EXPAND, 'unbuffered full should stay EXPAND')

    // Current ElementwiseMixin._broadcasted promotes shaped weak constants
    // through const_like(root), rebuilding each as direct CONST -> EXPAND
    // (mixin/elementwise.py:19-29, uop/ops.py:581-583).
    const zeroBroadcast = Tensor.full([0, 3], 1.5, { buffer: false })
      .add(Tensor.full([1, 3], 2.5, { buffer: false }))
    assertShape(zeroBroadcast.shape, [0, 3])
    assert(zeroBroadcast.uop.op === pg._core.ops.ADD, 'weak full add should stay ADD')
    for (const source of zeroBroadcast.uop.src) {
      assert(source.op === pg._core.ops.EXPAND, 'promoted weak full should be direct EXPAND')
      assert(source.src[0].op === pg._core.ops.CONST, 'promoted weak full should drop old movement chain')
    }
    assertClose(await zeroBroadcast.toArray(), [])
    const nonempty = Tensor.full([2, 3], 1.5, { buffer: false })
      .add(Tensor.full([1, 3], 2.5, { buffer: false }))
    assertClose(await nonempty.toArray(), [4, 4, 4, 4, 4, 4])

    let weakFullRejected = false
    try { Tensor.full([2], 1.0, { dtype: 'weakfloat' }) } catch (_) { weakFullRejected = true }
    assert(weakFullRejected, 'explicit weak full storage must be rejected')
    let weakEmptyRejected = false
    try { Tensor.empty([2], { dtype: 'weakfloat' }) } catch (_) { weakEmptyRejected = true }
    assert(weakEmptyRejected, 'explicit weak empty storage must be rejected')
  })

  await test('arange', async () => {
    const t = Tensor.arange(4)
    assertClose(await t.toArray(), [0, 1, 2, 3])
  })

  await test('eye', async () => {
    const t = Tensor.eye(2)
    assertClose(await t.toArray(), [1, 0, 0, 1])
  })

  await test('pure constructors store the pinned device-free root', async () => {
    const values = [
      Tensor.full([2, 3], 2, { buffer: false }),
      Tensor.arange(4),
      Tensor.linspace(0, 1, 4),
      Tensor.eye(3)
    ]
    for (const value of values) {
      assert(value.uopPhysical, 'pure constructor must store a physical root')
      assert(value.uopLogical, 'pure constructor must store a logical root')
      assert(
        value.uopPhysical.key === value.uopLogical.key,
        'pure constructor roots must be the same UOp'
      )
    }
    const arange = values[1]
    const moved = arange.to('cuda')
    assert(moved.uop.key === arange.uop.key, 'device-free Tensor.to must preserve the root')
  })

  await test('internal scalars store typed current roots', async () => {
    const cases = [
      [Tensor.empty([2], { dtype: 'bool' }), true, 'bool'],
      [Tensor.empty([2], { dtype: 'int32' }), 7, 'weakint'],
      [Tensor.empty([2], { dtype: 'float32' }), 1, 'weakint']
    ]
    for (const [source, value, dtype] of cases) {
      const scalar = source._ensureTensor(value)
      assert(scalar.dtype === dtype, `expected ${dtype}, got ${scalar.dtype}`)
      assert(
        scalar.uopLogical.op === pg._core.ops.CONST,
        'internal scalar logical root must be CONST'
      )
      assert(
        scalar.uopPhysical.op === pg._core.ops.CONST,
        'internal scalar physical root must be CONST'
      )
      assert(
        scalar.uopLogical.key === scalar.uopPhysical.key,
        'internal scalar roots must be the same UOp'
      )
    }
  })

  // -- Lazy RNG --
  console.log('\n-- Lazy RNG --')

  await test('rand uniform [0,1)', async () => {
    Tensor.manual_seed(42)
    const t = Tensor.rand(100)
    const arr = await t.toArray()
    assert(arr.length === 100, `Expected 100 elements, got ${arr.length}`)
    let min = arr[0], max = arr[0]
    for (let i = 1; i < arr.length; i++) {
      if (arr[i] < min) min = arr[i]
      if (arr[i] > max) max = arr[i]
    }
    assert(min >= 0 && max < 1, `Out of range: min=${min}, max=${max}`)
  })

  await test('uniform supports bounds and rejects empty intervals', async () => {
    Tensor.manual_seed(42)
    const arr = await Tensor.uniform(100, { low: -2, high: 3 }).toArray()
    assert(arr.every(value => value >= -2 && value < 3), 'uniform value outside requested range')
    let rejected = false
    try {
      Tensor.uniform(2, { low: 1, high: 1 })
    } catch (error) {
      rejected = String(error && error.message).includes('low < high')
    }
    assert(rejected, 'uniform should reject an empty interval')
  })

  await test('scaled_uniform matches the pinned initializer expression', async () => {
    Tensor.manual_seed(42)
    const actual = Tensor.scaled_uniform(2, 3)
    Tensor.manual_seed(42)
    const expected = Tensor.uniform(2, 3, { low: -1, high: 1 }).mul(6 ** -0.5)
    const [a, b] = await Promise.all([actual.toArray(), expected.toArray()])
    assert(a.length === b.length && a.every((value, i) => value === b[i]),
      'scaled_uniform values must match the pinned expression')
  })

  await test('glorot_uniform matches the pinned initializer expression', async () => {
    const bound = Math.sqrt(6 / (2 + 3))
    Tensor.manual_seed(42)
    const actual = Tensor.glorot_uniform(2, 3)
    Tensor.manual_seed(42)
    const expected = Tensor.uniform(2, 3, { low: -bound, high: bound })
    const [a, b] = await Promise.all([actual.toArray(), expected.toArray()])
    assert(a.length === b.length && a.every((value, i) => value === b[i]),
      'glorot_uniform values must match the pinned expression')
  })

  await test('randn gaussian', async () => {
    Tensor.manual_seed(99)
    const t = Tensor.randn(1000)
    const arr = await t.toArray()
    let sum = 0
    for (let i = 0; i < arr.length; i++) sum += arr[i]
    const mean = sum / arr.length
    assert(Math.abs(mean) < 0.3, `Mean too far from 0: ${mean}`)
  })

  await test('rand deterministic with seed', async () => {
    Tensor.manual_seed(1337)
    const a = await Tensor.rand(8).toArray()
    Tensor.manual_seed(1337)
    const b = await Tensor.rand(8).toArray()
    for (let i = 0; i < 8; i++) {
      assert(a[i] === b[i], `Not deterministic at [${i}]: ${a[i]} vs ${b[i]}`)
    }
    const expected = [
      0.4886598587036133, 0.3479880094528198, 0.6593245267868042,
      0.6364744901657104, 0.4711652994155884, 0.14146876335144043,
      0.27809083461761475, 0.049955129623413086
    ]
    for (let i = 0; i < expected.length; i++) {
      assert(a[i] === expected[i], `Pinned RNG mismatch at [${i}]: ${a[i]} vs ${expected[i]}`)
    }
  })

  // -- Float64 --
  console.log('\n-- Float64 --')

  // tinygrad gates backend dtype tests through is_dtype_supported. Polygrad's
  // WebGPU caps likewise advertise no f64 because WGSL has no f64 shader type.
  await testIf(supportsF64, 'f64: creation', async () => {
    const t = new Tensor([1.5, 2.5, 3.5], { dtype: 'float64' })
    assertClose(await t.toArray(), [1.5, 2.5, 3.5])
  })

  await testIf(supportsF64, 'f64: zeros', async () => {
    const t = Tensor.zeros(3, { dtype: 'float64' })
    assertClose(await t.toArray(), [0, 0, 0])
  })

  await testIf(supportsF64, 'f64: ones', async () => {
    const t = Tensor.ones(2, { dtype: 'float64' })
    assertClose(await t.toArray(), [1, 1])
  })

  await testIf(supportsF64, 'f64: add', async () => {
    const a = new Tensor([1, 2, 3], { dtype: 'float64' })
    const b = new Tensor([4, 5, 6], { dtype: 'float64' })
    const arr = await a.add(b).toArray()
    console.log('DEBUG f64 add', Array.from(arr))
    assertClose(arr, [5, 7, 9])
  })

  await testIf(supportsF64, 'f64: mul', async () => {
    const a = new Tensor([2, 3], { dtype: 'float64' })
    const b = new Tensor([4, 5], { dtype: 'float64' })
    const arr = await a.mul(b).toArray()
    console.log('DEBUG f64 mul', Array.from(arr))
    assertClose(arr, [8, 15])
  })

  await testIf(supportsF64, 'f64: sum', async () => {
    const t = new Tensor([1, 2, 3], { dtype: 'float64' })
    const v = await t.sum().item()
    console.log('DEBUG f64 sum', v)
    assert(Math.abs(v - 6) < 1e-10, `Expected 6, got ${v}`)
  })

  await testIf(supportsF64, 'f64: backward', async () => {
    const a = new Tensor([1, 2, 3], { dtype: 'float64' })
    const b = new Tensor([4, 5, 6], { dtype: 'float64' })
    const loss = a.mul(b).sum()
    await loss.backward()
    assert(a.grad, 'grad is null')
    assertClose(await a.grad.toArray(), [4, 5, 6])
  })

  await test('f64: integer list still infers int32', async () => {
    const a = new Tensor([1, 2, 3])
    assert(a.dtype === 'int32', `expected int32, got ${a.dtype}`)
  })

  // -- Kernel cache consistency --
  console.log('\n-- Cache consistency --')

  await test('repeated realize produces identical results', async () => {
    // Exercises the in-memory schedule cache + disk cache (native) or
    // WASM step plan cache. Same graph structure, same data = same result.
    const a = new Tensor([1, 2, 3, 4])
    const b = new Tensor([10, 20, 30, 40])
    const r1 = await a.add(b).toArray()
    const r2 = await (new Tensor([1, 2, 3, 4])).add(new Tensor([10, 20, 30, 40])).toArray()
    const r3 = await (new Tensor([1, 2, 3, 4])).add(new Tensor([10, 20, 30, 40])).toArray()
    assertClose(r1, [11, 22, 33, 44])
    assertClose(r2, [11, 22, 33, 44])
    assertClose(r3, [11, 22, 33, 44])
  })

  await test('realize retargets live tensors sharing lazy root', async () => {
    const a = new Tensor([1])
    await a.realize()
    const x1 = a.add(1)
    const x2 = a.add(1)
    assert(x1.uop.key === x2.uop.key, 'lazy roots should be hash-consed together')

    await x1.realize()

    assert(x1.uop.key === x2.uop.key, 'shared lazy root should retarget to one realized root')
    assert(x1.uop.buffer.key === x2.uop.buffer.key, 'shared lazy root should share realized buffer')
    assertClose(await x2.toArray(), [2])
  })

  await test('realize rewrites downstream live graph', async () => {
    const a = new Tensor([1])
    await a.realize()
    const x = a.add(1)
    const y = x.add(2)
    const oldYKey = y.uop.key

    await x.realize()

    assert(y.uop.key !== oldYKey, 'downstream graph should be rewritten through realized x')
    assertClose(await y.toArray(), [4])
  })

  await test('separate realizes do not alias buffers', async () => {
    const a = new Tensor([1])
    await a.realize()
    const y1 = await a.add(1).realize()
    const y2 = await a.add(1).realize()

    assert(y1.uop.buffer.key !== y2.uop.buffer.key, 'separate materializations should get separate buffers')
    assertClose(await y1.toArray(), [2])
    assertClose(await y2.toArray(), [2])
  })

  await test('to keeps separate realized same-logical occurrences', async () => {
    // Keep this cross-device under every suite default. Pinned Tensor.to
    // returns self for a same-device request (tensor.py:327-335).
    const a = new Tensor([1], { device: 'cpu' })
    await a.realize()
    const x1 = await a.add(1).realize()
    const x2 = await a.add(1).realize()
    assert(x1.uop.buffer.key !== x2.uop.buffer.key, 'separate materializations should stay distinct')

    // The preserved logical expression can be shared, but placement must see
    // the occurrence-specific realized source when .to(device) creates a fact.
    const x1Cuda = x1.to('cuda')
    const x2Cuda = x2.to('cuda')
    assert(x1Cuda.uop.op === pg._core.ops.COPY, 'x1.to(cuda) should be an eager COPY')
    assert(x2Cuda.uop.op === pg._core.ops.COPY, 'x2.to(cuda) should be an eager COPY')
    assert(x1Cuda.uop.buffer === null, 'an unrealized COPY should not claim buffer identity')
    assert(x2Cuda.uop.buffer === null, 'an unrealized COPY should not claim buffer identity')
    assert(
      x1Cuda.uop.src[0].buffer.key === x1.uop.buffer.key,
      'x1.to(cuda) COPY should point at x1 buffer'
    )
    assert(
      x2Cuda.uop.src[0].buffer.key === x2.uop.buffer.key,
      'x2.to(cuda) COPY should point at x2 buffer'
    )
    assert(
      x1Cuda.uop.src[0].buffer.key !== x2Cuda.uop.src[0].buffer.key,
      'to(cuda) COPY sources should not alias'
    )

    const y1 = x1Cuda.add(1)
    const y2 = x2Cuda.add(1)
    assert(y1.uop.key !== y2.uop.key, 'downstream graphs should use distinct occurrence sources')
  })

  await test('nested to keeps realized current and export logical separate', async () => {
    const x = await new Tensor([1]).add(1).realize()
    const xBase = x.to('cpu')
    const xCuda = xBase.to('cuda')
    const xCpu = xCuda.to('cpu')

    // Pinned Tensor.to (tensor.py:327-335) keeps both device moves as exact
    // current COPY occurrences. Polygrad additionally preserves the approved
    // portable logical twin.
    assert(xCuda.uop.op === pg._core.ops.COPY, 'CUDA move should be an eager COPY')
    assert(xCuda.uop.src[0].key === xBase.uop.key, 'CUDA COPY should use the CPU occurrence')
    assert(xCpu.uop.op === pg._core.ops.COPY, 'CPU roundtrip should be an eager COPY')
    assert(xCpu.uop.src[0].key === xCuda.uop.key, 'CPU COPY should use the CUDA occurrence')
    assert(xCpu.uop.key !== x.uop.key, 'roundtrip COPY should stay occurrence-distinct')
    assert(xCpu.uopPhysical.key === xCpu.uop.key, 'physical root should be the current COPY')
    assert(xCpu.uopLogical.key === x.uopLogical.key, 'nested to should preserve export logical root')

    const y = xCpu.add(1)
    assert(y.device === 'CPU', 'downstream value should keep selected CPU placement')
    assert(y.uop.key !== xCpu.uop.key, 'downstream value should build a new current graph')

    // Pinned Tensor.alu consumes ordered current roots (tensor.py:128-140).
    // Sharing one portable logical X must not collapse the second occurrence.
    const mixed = xBase.add(xCpu)
    assert(mixed.uop.op === pg._core.ops.ADD, 'mixed result should be ADD')
    assert(mixed.uop.src[0].key === xBase.uop.key, 'first ADD input should be CPU occurrence')
    assert(mixed.uop.src[1].key === xCpu.uop.key, 'second ADD input should be CPU COPY')
    assert(mixed.uop.src[1].src[0].key === xCuda.uop.key, 'CPU COPY should contain CUDA COPY')
    assert(
      mixed.uop.src[1].src[0].src[0].key === xBase.uop.key,
      'CUDA COPY should contain CPU occurrence'
    )
    assert(mixed.uopLogical.src[0].key === x.uopLogical.key, 'logical lhs should be X')
    assert(mixed.uopLogical.src[1].key === x.uopLogical.key, 'logical rhs should be X')
  })

  await test('composite ops preserve repeated logical physical occurrences', async () => {
    // Pinned composite methods consume ordered Tensor.uop occurrences. Two
    // logical aliases must not become one substitution key.
    const x = await new Tensor([0, 0, 0, 0], { device: 'cpu', dtype: 'float32' })
      .reshape(1, 1, 2, 2).realize()
    const movedWeight = x.to('cuda').to('cpu')
    const conv = x.conv2d(movedWeight)
    assert(
      countGraphOp(conv.uop, pg._core.ops.COPY) === 2,
      'conv2d should retain the nested COPY weight occurrence'
    )
    assert(
      countGraphOp(conv.uop, pg._core.ops.EXPAND) === 1,
      'single-output conv2d should elide the no-op channel EXPAND'
    )

    const bnX = await new Tensor([0, 0, 0], { device: 'cpu', dtype: 'float32' })
      .reshape(1, 3, 1, 1).realize()
    const stat = await new Tensor([0, 0, 0], { device: 'cpu', dtype: 'float32' }).realize()
    const movedInvstd = stat.to('cuda').to('cpu')
    const bn = bnX.batchnorm(null, null, stat, movedInvstd, 1)
    assert(
      countGraphOp(bn.uop, pg._core.ops.COPY) === 2,
      'batchnorm should retain the nested COPY invstd occurrence'
    )

    const minX = await new Tensor([1], { device: 'cpu', dtype: 'float32' }).realize()
    const minMoved = minX.to('cuda').to('cpu')
    const minimum = minX.minimum(minMoved)
    assert(
      countGraphOp(minimum.uop, pg._core.ops.COPY) === 2,
      'minimum should retain the nested COPY right occurrence'
    )
    assert(minimum.uop.op === pg._core.ops.MUL, 'minimum should end with inverse MUL')
    const maximum = minimum.uop.src[0]
    assert(maximum.op === pg._core.ops.MAX, 'minimum should contain ordered MAX')
    assert(maximum.src[0].src[0].key === minX.uop.key, 'minimum lhs occurrence mismatch')
    assert(maximum.src[1].src[0].key === minMoved.uop.key, 'minimum rhs occurrence mismatch')

  })

  await test('repeated fused chain produces identical results', async () => {
    // Fused multi-op chain through cache: (a+b)*a-b
    const a = [2, 3, 4]
    const b = [1, 1, 1]
    const expected = [(2+1)*2-1, (3+1)*3-1, (4+1)*4-1]
    for (let i = 0; i < 3; i++) {
      const ta = new Tensor(a), tb = new Tensor(b)
      const r = await ta.add(tb).mul(ta).sub(tb).toArray()
      assertClose(r, expected)
    }
  })

  // -- Missing math ops --
  console.log('\n-- Missing math ops --')

  await test('pow integer', async () => {
    const t = new Tensor([2, 3, 4])
    const r = t.pow(2)
    assertClose(await r.toArray(), [4, 9, 16])
  })

  await test('pow float', async () => {
    const t = new Tensor([4, 9, 16])
    const r = t.pow(0.5)
    assertClose(await r.toArray(), [2, 3, 4])
  })

  await test('pow scalar promotion and validation match tinygrad', async () => {
    const t = new Tensor([2, 3], { dtype: 'int32' })
    // JavaScript erases the lexical distinction between 2 and 2.0.
    const exponent = new Tensor(2.0, { dtype: 'weakfloat' })
    const promoted = t.pow(exponent)
    assert(promoted.dtype === 'weakfloat', `expected weakfloat, got ${promoted.dtype}`)
    assert(promoted.uop.op === pg._core.ops.POW, 'promoted pow should remain a raw POW')
    assertClose(await promoted.toArray(), [4, 9])

    const reverse = t.pow(exponent, true)
    assert(reverse.dtype === 'weakfloat', `expected reverse weakfloat, got ${reverse.dtype}`)
    assertClose(await reverse.toArray(), [4, 8])

    let rejected = false
    try { t.pow(-1) } catch (e) { rejected = e.message.includes('base needs to be float') }
    assert(rejected, 'negative integer scalar exponent should reject an integer common dtype')

    const tensorExponent = t.pow(new Tensor([-1, -2], { dtype: 'int32' }))
    assert(tensorExponent.dtype === 'int32', `expected int32, got ${tensorExponent.dtype}`)
    assert(tensorExponent.uop.op === pg._core.ops.POW, 'integer Tensor pow should remain raw POW')
    // Tinygrad 2026-08-22/a9069c177a9d accepts the raw integer graph, while
    // its xpow lowering cannot STORE the float result into integer storage.
    // Backend compilation failure is not a portable Tensor API contract.
    const strong = t.cast('float32').pow(new Tensor([-1, -2], { dtype: 'float32' }))
    assertClose(await strong.toArray(), [0.5, 1 / 9])
  })

  await test('reciprocal', async () => {
    const t = new Tensor([2, 4, 5])
    const r = t.reciprocal()
    assertClose(await r.toArray(), [0.5, 0.25, 0.2])
  })

  await test('exp2', async () => {
    const t = new Tensor([0, 1, 2, 3])
    const r = t.exp2()
    assertClose(await r.toArray(), [1, 2, 4, 8])
  })

  await test('log2', async () => {
    const t = new Tensor([1, 2, 4, 8])
    const r = t.log2()
    assertClose(await r.toArray(), [0, 1, 2, 3])
  })

  await test('trunc', async () => {
    const t = new Tensor([1.7, -2.3, 3.9])
    const r = t.trunc()
    assertClose(await r.toArray(), [1, -2, 3])
  })

  // -- Aliases --
  console.log('\n-- Aliases --')

  await test('swish is silu', async () => {
    const t = new Tensor([1, 2, -1])
    assertClose(await t.swish().toArray(), await t.silu().toArray())
  })

  await test('view is reshape', async () => {
    const t = new Tensor([1, 2, 3, 4, 5, 6])
    assertShape(t.view(2, 3).shape, [2, 3])
    assertClose(await t.view(2, 3).toArray(), [1, 2, 3, 4, 5, 6])
  })

  await test('matmul is dot', async () => {
    const a = new Tensor([[1, 2], [3, 4]])
    const b = new Tensor([[5, 6], [7, 8]])
    assertClose(await a.matmul(b).toArray(), await a.dot(b).toArray())
  })

  // -- Composed ops --
  console.log('\n-- Composed ops --')

  await test('conv2d stride and padding match reference', async () => {
    const xData = Float32Array.from({ length: 1 * 2 * 4 * 5 }, (_, i) => i / 7)
    const wData = Float32Array.from({ length: 3 * 2 * 2 * 3 }, (_, i) => (i - 5) / 11)
    const bData = Float32Array.from([0.5, -1.0, 2.0])
    const expected = new Float32Array(1 * 3 * 6 * 2)
    for (let oc = 0; oc < 3; oc++) {
      for (let oy = 0; oy < 6; oy++) {
        for (let ox = 0; ox < 2; ox++) {
          let acc = bData[oc]
          for (let ic = 0; ic < 2; ic++) {
            for (let ky = 0; ky < 2; ky++) {
              for (let kx = 0; kx < 3; kx++) {
                const iy = oy + ky - 2
                const ix = ox * 2 + kx - 1
                if (iy >= 0 && iy < 4 && ix >= 0 && ix < 5) {
                  acc += xData[((ic * 4 + iy) * 5) + ix] * wData[((oc * 2 + ic) * 2 + ky) * 3 + kx]
                }
              }
            }
          }
          expected[(oc * 6 + oy) * 2 + ox] = acc
        }
      }
    }
    const conv = new Tensor(xData).reshape(1, 2, 4, 5)
      .conv2d(new Tensor(wData).reshape(3, 2, 2, 3), new Tensor(bData), 1, [1, 2], 1, [1, 0, 2, 1])
    assertShape(conv.shape, [1, 3, 6, 2])
    assertClose(await conv.toArray(), expected, 1e-4)
  })

  await testIf(supportsF16, 'conv2d dtype promotion and accumulation match pinned', async () => {
    // Binary-exact values keep this gate on conv composition. Standalone
    // realized float16 storage on core-WASM is a separate backend debt.
    const x = new Tensor(Float32Array.from({ length: 9 }, (_, i) => i / 8))
      .reshape(1, 1, 3, 3).cast('float16')
    const weightData = Float32Array.from([0.25, -0.5, 0.75, 0.125])
    const wHalf = new Tensor(weightData).reshape(1, 1, 2, 2).cast('float16')
    const wFloat = new Tensor(weightData).reshape(1, 1, 2, 2)
    const bHalf = new Tensor(new Float32Array([0.0625])).cast('float16')
    const bFloat = new Tensor(new Float32Array([0.0625]))
    const mixed = x.conv2d(wFloat, bFloat)
    const halfDefault = x.conv2d(wHalf, bHalf)
    const halfExplicit = x.conv2d(wHalf, bHalf, { dtype: 'float32' })
    const halfFloatBias = x.conv2d(wHalf, bFloat)
    assert(mixed.dtype === 'float32', `expected float32, got ${mixed.dtype}`)
    assert(halfDefault.dtype === 'float16', `expected float16, got ${halfDefault.dtype}`)
    assert(halfExplicit.dtype === 'float32', `expected float32, got ${halfExplicit.dtype}`)
    assert(halfFloatBias.dtype === 'float32', `expected float32, got ${halfFloatBias.dtype}`)
    const expected = [0.34375, 0.421875, 0.578125, 0.65625]
    assertClose(await mixed.toArray(), expected, 0)
    assertClose(await halfDefault.toArray(), expected, 0)
    assertClose(await halfExplicit.toArray(), expected, 0)
    assertClose(await halfFloatBias.toArray(), expected, 0)
  })

  await test('conv2d padded 3x3 4x4 devectorize regression', async () => {
    // tinygrad: arange(16).reshape(1,1,4,4).conv2d(ones(1,1,3,3), padding=1)
    // This crosses the 128-lane late-devectorize threshold.
    const padded = Tensor.arange(16, { dtype: 'float32' }).reshape(1, 1, 4, 4)
      .conv2d(Tensor.ones(1, 1, 3, 3), null, 1, 1, 1, 1)
    assertShape(padded.shape, [1, 1, 4, 4])
    assertClose(await padded.toArray(), [10, 18, 24, 18, 27, 45, 54, 39, 51, 81, 90, 63, 42, 66, 72, 50])
  })

  await test('max_pool2d padding matches reference', async () => {
    const pool = Tensor.arange(9, { dtype: 'float32' }).reshape(1, 1, 3, 3).max_pool2d(2, 1, 1, 1)
    assertShape(pool.shape, [1, 1, 4, 4])
    assertClose(await pool.toArray(), [0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 8, 6, 7, 8, 8])
  })

  await test('batchnorm multi-axis matches reference', async () => {
    const bnX = Float32Array.from({ length: 2 * 3 * 4 * 5 }, (_, i) => i / 10)
    const mean = Float32Array.from({ length: 8 }, (_, i) => i / 20)
    const inv = Float32Array.from({ length: 8 }, () => 0.25)
    const weight = Float32Array.from({ length: 8 }, (_, i) => 0.5 + (0.7 * i) / 7)
    const bias = Float32Array.from({ length: 8 }, (_, i) => -0.3 + (0.7 * i) / 7)
    const bnExpected = new Float32Array(bnX.length)
    for (let n = 0; n < 2; n++) for (let c = 0; c < 3; c++) for (let h = 0; h < 4; h++) for (let w = 0; w < 5; w++) {
      const ki = n * 4 + h
      const oi = ((n * 3 + c) * 4 + h) * 5 + w
      bnExpected[oi] = ((bnX[oi] - mean[ki]) * weight[ki]) * inv[ki] + bias[ki]
    }
    const bn = new Tensor(bnX).reshape(2, 3, 4, 5).batchnorm(
      new Tensor(weight).reshape(2, 4), new Tensor(bias).reshape(2, 4),
      new Tensor(mean).reshape(2, 4), new Tensor(inv).reshape(2, 4), [0, 2]
    )
    assertClose(await bn.toArray(), bnExpected, 1e-5)
  })

  await test('nn Conv2d backward populates parameters', async () => {
    const mod = new pg.nn.Conv2d(3, 2, 3, { padding: 1 })
    const loss = mod.call(Tensor.randn(1, 3, 4, 4)).relu().mean()
    loss.backward()
    assert(mod.weight.grad !== null, 'Conv2d weight grad missing')
    assert(mod.bias.grad !== null, 'Conv2d bias grad missing')
  })

  await test('nn GroupNorm matches pinned layernorm composition', async () => {
    const mod = new pg.nn.GroupNorm(2, 4)
    const x = Tensor.arange(32, { dtype: 'float32' }).reshape(1, 4, 2, 4).div(11)
    const out = mod.call(x)
    assertShape(out.shape, [1, 4, 2, 4])
    const values = await out.toArray()
    for (let group = 0; group < 2; group++) {
      const start = group * 16
      const mean = values.slice(start, start + 16).reduce((a, b) => a + b, 0) / 16
      assert(Math.abs(mean) < 1e-5, `GroupNorm group ${group} mean ${mean}`)
    }
    out.sum().backward()
    assert(mod.weight.grad !== null, 'GroupNorm weight grad missing')
    assert(mod.bias.grad !== null, 'GroupNorm bias grad missing')
  })

  await test('nn LayerNorm2d matches pinned NHWC composition', async () => {
    const mod = new pg.nn.LayerNorm2d(3)
    const input = Array.from({ length: 24 }, (_, i) => (i - 7) / 5)
    const out = mod.call(new Tensor(input).reshape(2, 3, 2, 2))
    assertShape(out.shape, [2, 3, 2, 2])
    assert(out.uop.op === pg._core.ops.PERMUTE, 'LayerNorm2d root must be PERMUTE')
    assert(countGraphOp(out.uop, pg._core.ops.PERMUTE) === 4, 'expected four PERMUTEs')
    assert(countGraphOp(out.uop, pg._core.ops.REDUCE) === 2, 'expected two REDUCE nodes')
    const values = await out.toArray()
    for (let n = 0; n < 2; n++) for (let h = 0; h < 2; h++) for (let w = 0; w < 2; w++) {
      const lanes = Array.from({ length: 3 }, (_, c) => values[((n * 3 + c) * 2 + h) * 2 + w])
      const mean = lanes.reduce((a, b) => a + b, 0) / lanes.length
      const variance = lanes.reduce((a, b) => a + b * b, 0) / lanes.length
      assert(Math.abs(mean) < 1e-5, `LayerNorm2d mean ${mean}`)
      assert(Math.abs(variance - 1) < 1e-4, `LayerNorm2d variance ${variance}`)
    }
    out.sum().backward()
    assert(mod.weight.grad !== null, 'LayerNorm2d weight grad missing')
    assert(mod.bias.grad !== null, 'LayerNorm2d bias grad missing')
  })

  await test('layernorm', async () => {
    const t = new Tensor([[1, 2, 3], [4, 5, 6]])
    const r = await t.layernorm()
    const arr = await r.toArray()
    // Each row should have mean ~0 and std ~1
    const row0 = arr.slice(0, 3)
    const mean0 = row0.reduce((a, b) => a + b) / 3
    assert(Math.abs(mean0) < 1e-4, `Row 0 mean should be ~0, got ${mean0}`)
  })

  await test('binaryCrossEntropy', async () => {
    const pred = new Tensor([0.9, 0.1, 0.8])
    const target = new Tensor([1, 0, 1])
    const loss = await pred.binaryCrossEntropy(target).item()
    // -mean(t*log(p) + (1-t)*log(1-p))
    const expected = -(Math.log(0.9) + Math.log(0.9) + Math.log(0.8)) / 3
    assert(Math.abs(loss - expected) < 1e-3, `Expected ~${expected}, got ${loss}`)
  })

  await test('cat 1d', async () => {
    const a = new Tensor([1, 2, 3])
    const b = new Tensor([4, 5, 6])
    const r = Tensor.cat(a, b)
    assertShape(r.shape, [6])
    assertClose(await r.toArray(), [1, 2, 3, 4, 5, 6])
  })

  await test('cat 2d axis0', async () => {
    const a = new Tensor([[1, 2], [3, 4]])
    const b = new Tensor([[5, 6]])
    const r = Tensor.cat(a, b, { dim: 0 })
    assertShape(r.shape, [3, 2])
    const arr = await r.toArray()
    console.log('DEBUG cat axis0', Array.from(arr))
    assertClose(arr, [1, 2, 3, 4, 5, 6])
  })

  await test('instance cat matches tinygrad binding', async () => {
    const a = new Tensor([[1, 2], [3, 4]])
    const b = new Tensor([[5, 6]])
    const c = new Tensor([[7, 8]])
    const r = a.cat(b, c, { dim: 0 })
    assertShape(r.shape, [4, 2])
    assertClose(await r.toArray(), [1, 2, 3, 4, 5, 6, 7, 8])
  })

  await test('stack', async () => {
    const a = new Tensor([1, 2, 3])
    const b = new Tensor([4, 5, 6])
    const r = Tensor.stack(a, b)
    assertShape(r.shape, [2, 3])
    const arr = await r.toArray()
    console.log('DEBUG stack', Array.from(arr))
    assertClose(arr, [1, 2, 3, 4, 5, 6])
  })

  await test('instance stack matches tinygrad binding', async () => {
    const a = new Tensor([1, 2])
    const b = new Tensor([3, 4])
    const r = a.stack(b, { dim: 0 })
    assertShape(r.shape, [2, 2])
    assertClose(await r.toArray(), [1, 2, 3, 4])
  })

  await test('repeat', async () => {
    const t = new Tensor([1, 2, 3])
    const r = t.repeat(3)
    assertShape(r.shape, [9])
    assertClose(await r.toArray(), [1, 2, 3, 1, 2, 3, 1, 2, 3])
  })

  await test('repeat readback preserves int32 dtype', async () => {
    const t = new Tensor([1, 2, 3], { dtype: 'int32' })
    const r = t.repeat(3)
    assert(r.dtype === 'int32', `expected int32, got ${r.dtype}`)
    const arr = await r.toArray()
    assert(arr.constructor.name === 'Int32Array', `expected Int32Array, got ${arr.constructor.name}`)
    assertClose(arr, [1, 2, 3, 1, 2, 3, 1, 2, 3])
  })

  await test('repeat 2d', async () => {
    const t = new Tensor([[1, 2], [3, 4]])
    const r = t.repeat(2, 3)
    assertShape(r.shape, [4, 6])
    assertClose(await r.toArray(), [
      1, 2, 1, 2, 1, 2,
      3, 4, 3, 4, 3, 4,
      1, 2, 1, 2, 1, 2,
      3, 4, 3, 4, 3, 4
    ])
  })

  console.log(`\nResults: ${passed} passed, ${failed} failed, ${skipped} skipped, ${passed + failed + skipped} total`)
  return { passed, failed, skipped }
}

async function checkLogicalRuntimeOption(polygrad, core) {
  const runtime = await polygrad.create({ core, logical: 'never' })
  try {
    const tensor = new runtime.Tensor([1, 2])
    assert(tensor.logicalPolicy === 'never')
    assert(tensor.uopLogical === null)
  } finally {
    await runtime.dispose()
  }
}

module.exports = { checkLogicalRuntimeOption, runTensorTests }
