'use strict'

const onnxFixture = require('../../test/fixtures/onnx.json')
const embeddingGemma2Fixture = require('../../test/fixtures/embeddinggemma2.json')
const embeddingGemma2ImageFixture = require('../../test/fixtures/embeddinggemma2_image.json')
const embeddingGemma2AudioFixture = require('../../test/fixtures/embeddinggemma2_audio.json')

async function checkEmbeddingGemma2(pg) {
  for (const f of [embeddingGemma2Fixture, embeddingGemma2ImageFixture, embeddingGemma2AudioFixture]) {
    const weights = Uint8Array.from(atob(f.weights), c => c.charCodeAt(0))
    const m = pg.Model.fromHF(new TextEncoder().encode(JSON.stringify(f.config)), [weights], {maxBatch: 2, maxSeqLen: f.config.max_seq_len || 6})
    let restored
    try {
      restored = pg.Model.load(await m.saveAsync())
      for (const c of f.cases) {
        const raw = c.inputs || {input_ids:c.input_ids, attention_mask:c.attention_mask}
        const inputs = Object.fromEntries(Object.entries(raw).map(([k,v]) =>
          [k, ['pixel_values','input_features'].includes(k) ? Float32Array.from(v.flat(Infinity)) : Int32Array.from(v.flat(Infinity))]))
        for (const model of [m, restored]) {
          const result = await model.forwardAsync(inputs)
          assertClose(result.last_hidden_state, c.last_hidden_state.flat(2), 2e-5)
          assertClose(result.sentence_embedding, c.sentence_embedding.flat(), 2e-5)
          if (c.image_hidden_states) {
            const width=f.config.text_config.hidden_size, selected=[]
            for (let i=0;i<result.image_attention_mask.length;i++) {
              if(result.image_attention_mask[i]) selected.push(...result.image_hidden_states.slice(i*width,(i+1)*width))
            }
            assertClose(selected,c.image_hidden_states.flat(),2e-5)
          }
          if (c.audio_hidden_states) {
            const width=f.config.text_config.hidden_size, selected=[]
            for (let i=0;i<result.audio_attention_mask.length;i++) {
              if(result.audio_attention_mask[i]) selected.push(...result.audio_hidden_states.slice(i*width,(i+1)*width))
            }
            assertClose(selected,c.audio_hidden_states.flat(),2e-5)
          }
          for (const [label, token] of [['image', f.config.image_token_id], ['audio', f.config.audio_token_id]]) {
            if (!Number.isInteger(token) || !f.config[label === 'image' ? 'vision_config' : 'audio_config']) continue;
            for (const delta of [-1, 1]) {
              const bad = {...inputs, input_ids: inputs.input_ids.slice()}
              const at = bad.input_ids.findIndex(v => delta < 0 ? v === token : v !== token)
              bad.input_ids[at] = delta < 0 ? 1 : token
              let error
              try { await model.forwardAsync(bad) } catch (e) { error = e }
              assert(error && error.message.includes(label + ' features and token slots do not match'),
                     'invalid ' + label + ' slot count must reject: ' + error)
              assertClose(await model.readBufferAsync('input_ids'), inputs.input_ids, 0)
              assertClose((await model.forwardAsync(inputs)).sentence_embedding, result.sentence_embedding, 0)
            }
          }
        }
      }
    } finally {
      if (restored) await restored.dispose()
      await m.dispose()
    }
  }
  const config = {...embeddingGemma2AudioFixture.config, modalities: ['text']}
  delete config.audio_seq_len
  const weights = Uint8Array.from(atob(embeddingGemma2AudioFixture.weights), c => c.charCodeAt(0))
  const text = pg.Model.fromHF(new TextEncoder().encode(JSON.stringify(config)), [weights],
                             {maxBatch: 2, maxSeqLen: config.max_seq_len})
  let restored
  try {
    restored = pg.Model.load(await text.saveAsync())
    const ones = new Int32Array(2 * config.max_seq_len).fill(1)
    const inputs = {input_ids: ones, attention_mask: ones}
    const expected = await text.forwardAsync(inputs)
    assert(!('audio_hidden_states' in expected), 'text selection must omit the audio tower')
    assertClose((await restored.forwardAsync(inputs)).sentence_embedding, expected.sentence_embedding, 0)
  } finally {
    if (restored) await restored.dispose()
    await text.dispose()
  }
}

async function checkIntervalSum(createRuntime) {
  for (const kernels of [false,true]) {
    const pg=await createRuntime({kernels,logical:'always'})
    let model,restored
    try {
      const n=17,width=3,starts=[0,0,3,8,-5,17,2147483647,-2147483648],stops=[0,3,8,3,26,17,-2147483648,2147483647]
      const x=pg.Tensor.empty(n,width),lo=pg.Tensor.empty(8,1,1,{dtype:'int32'}),hi=pg.Tensor.empty(8,1,1,{dtype:'int32'})
      const i=pg.Tensor.arange(n,{dtype:'int32'}).reshape(1,n,1)
      // Existing C AND constructor, also used by separately compiled authors.
      const y=i.ge(lo)._binop(i.lt(hi),'AND').where(x.reshape(1,n,width),0).sum(1)
      const [dx]=y.sum().gradient(x)
      model=await pg.Model.fromTensors({inputs:{x,lo,hi},outputs:{y,dx}})
      const saved=await model.saveAsync()
      const values=Float32Array.from({length:n*width},(_,j)=>(j%13-6)/8)
      const expected=new Float32Array(8*width),gradient=new Float32Array(n*width)
      for(let s=0;s<8;s++)for(let row=0;row<n;row++)if(row>=starts[s]&&row<stops[s])for(let c=0;c<width;c++) {
        expected[s*width+c]+=values[row*width+c];gradient[row*width+c]++
      }
      const inputs={x:values,lo:Int32Array.from(starts),hi:Int32Array.from(stops)}
      for(let repeat=0;repeat<2;repeat++) {
        const got=await model.forwardAsync(inputs)
        assertClose(got.y,expected,0);assertClose(got.dx,gradient,0)
      }
      const program=new TextDecoder().decode(await model.exportProgramAsync())
      assert(program.includes('segment_sum')===kernels,'interval reduction provider selection')
      restored=pg.Model.load(saved)
      const got=await restored.forwardAsync(inputs)
      assertClose(got.y,expected,0);assertClose(got.dx,gradient,0)
    } finally {
      if(restored)await restored.dispose()
      if(model)await model.dispose()
      await pg.dispose()
    }
  }
}

async function checkPortableGradient(pg) {
  const x = new pg.Tensor(new Float32Array([1,2,3])).is_param_(false)
  const square = x.square(), loss = square.sum()
  const [grad] = loss.gradient(x), sum = grad.sum(), [second] = sum.gradient(x)
  let model, restored
  try {
    assert(x.grad == null, 'targeted gradient must not change .grad')
    model = await pg.Model.fromTensors({params:{x}, outputs:{gradient:grad, second}})
    restored = pg.Model.load(await model.saveAsync())
    for (const current of [model, restored]) for (const values of [[1,2,3], [-4,0,7]]) {
      await current.writeBufferAsync('x', new Float32Array(values))
      const result = await current.forwardAsync({})
      assertClose(result.gradient, values.map(x => 2*x), 0)
      assertClose(result.second, [2,2,2], 0)
    }
    const hidden = new pg.Tensor(new Float32Array([3,4,5]))
    const invalid = x.add(hidden)
    try {
      let error
      try { await pg.Model.fromTensors({params:{x}, outputs:{invalid}}) }
      catch (e) { error = e }
      assert(error && /unbound storage BUFFER/.test(error.message), `lost Model error: ${error}`)
    } finally { invalid.dispose(); hidden.dispose() }
  } finally {
    if (restored) await restored.dispose()
    if (model) await model.dispose()
    for (const t of [second,sum,grad,loss,square,x]) t.dispose()
  }
}

async function checkPackedGemmModel(pg, gpuTile = false, provider = null) {
  const rows = gpuTile ? 32 : 2, kdim = gpuTile ? 32 : 7, ncols = gpuTile ? 192 : 48
  const a = Float32Array.from({length:2*rows*kdim}, (_, i) => (i % 9 - 4) / 11)
  const b = Float32Array.from({length:kdim*ncols}, (_, i) => (i % 13 - 6) / 17)
  const x = pg.Tensor.empty([2, rows, kdim]), w = new pg.Tensor(b).reshape(kdim, ncols)
  await w.realize()
  const model = await pg.Model.fromCallableAsync(({x}) => x.add(1).permute(1, 0, 2).reshape(2*rows, kdim).dot(w), {inputs:{x}, params:{w}})
  let restored
  try {
    for (const scale of [1, -0.75]) {
      const weight = Float32Array.from(b, v => v * scale), expected = new Float32Array(2*rows*ncols)
      for (let m = 0; m < 2*rows; m++) for (let n = 0; n < ncols; n++) {
        let value = 0
        const row = (m % 2) * rows + Math.floor(m / 2)
        for (let k = 0; k < kdim; k++) value += Math.fround(a[row*kdim+k] + 1) * weight[k*ncols+n]
        expected[m*ncols+n] = value
      }
      await model.writeBufferAsync('w', weight)
      assertClose((await model.forwardAsync({x:a})).output, expected, 2e-5)
    }
    restored = pg.Model.load(await model.saveAsync())
    assertClose((await restored.forwardAsync({x:a})).output, (await model.forwardAsync({x:a})).output, 2e-5)
    if (provider) {
      // Bound exports contain the programs that actually ran, unlike portable
      // bundles, which intentionally preserve the generic authoring graph.
      const program = new TextDecoder().decode(await model.exportProgramAsync())
      assert(program.includes(provider), `expected ${provider} in executed program; generic fallback used`)
    }
  } finally {
    if (restored) await restored.dispose()
    await model.dispose()
    x.dispose(); w.dispose()
  }
}

async function checkPortableGemmModel(createRuntime) {
  const rt = await createRuntime({kernels:true})
  try {
    await checkPackedGemmModel(rt)
    const strict = typeof process !== 'undefined' && process.env.POLY_REQUIRE_KERNELS === '1'
    const provider = rt.device === 'interp' ? null : rt.device === 'webgpu' ? 'webgpu_gemm' : rt.core === 'wasm' ? 'wasm_gemm' :
      strict ? 'avx2_gemm' : null
    await checkPackedGemmModel(rt, true, provider)
    assert(rt._core.ffi.poly_ctx_set_kernel_policy(rt._core.ctx, 0) === -1,
      'kernel policy changed after compilation')
    // More than one workgroup, transposed weights, then an incomplete tile
    // that must keep the generic path. Gradients use the same generic authoring graph.
    for (const [m,k,n] of [[32,32,64], [64,96,128], [32,48,64], [64,17,64], [8,31,48], [3,7,17]]) {
      const a=Float32Array.from({length:m*k},(_,i)=>(i%13-6)/32)
      const b=Float32Array.from({length:n*k},(_,i)=>(i%17-8)/32)
      const x=new rt.Tensor(a).reshape(m,k), w=new rt.Tensor(b).reshape(n,k).transpose()
      const y=x.dot(w), loss=y.sum(), expected=new Float32Array(m*n)
      for(let i=0;i<m;i++) for(let j=0;j<n;j++) for(let r=0;r<k;r++)
        expected[i*n+j]+=a[i*k+r]*b[j*k+r]
      try {
        assertClose(await y.toArrayAsync(),expected,2e-5)
        await loss.backward()
        const dx=x.grad, dw=w.grad
        try {
          const gx=new Float32Array(m*k), gw=new Float32Array(k*n)
          for(let i=0;i<m;i++) for(let r=0;r<k;r++) for(let j=0;j<n;j++) gx[i*k+r]+=b[j*k+r]
          for(let r=0;r<k;r++) for(let j=0;j<n;j++) for(let i=0;i<m;i++) gw[r*n+j]+=a[i*k+r]
          assertClose(await dx.toArrayAsync(),gx,2e-5)
          assertClose(await dw.toArrayAsync(),gw,2e-5)
        } finally {dx.dispose();dw.dispose()}
      } finally {loss.dispose();y.dispose();x.dispose();w.dispose()}
    }
  } finally { await rt.dispose() }
}

async function checkCpuAttentionModel(pg) {
  const scores = Float32Array.from({length:56}, (_, i) => i % 7 === 6 ? -Infinity : (i % 13 - 6) / 11)
  const values = Float32Array.from({length:112}, (_, i) => (i % 17 - 8) / 19)
  const x = pg.Tensor.empty([2, 4, 7]), v = new pg.Tensor(values).reshape(2, 7, 8)
  await v.realize()
  const model = await pg.Model.fromCallableAsync(({x}) => x.softmax(-1).dot(v), {inputs:{x}, params:{v}})
  let restored
  try {
    const expected = new Float32Array(64)
    for (let h = 0; h < 2; h++) for (let m = 0; m < 4; m++) {
      const weights = Array.from(scores.slice((h*4+m)*7, (h*4+m+1)*7), Math.exp)
      const sum = weights.reduce((a,b) => a+b, 0)
      for (let n = 0; n < 8; n++) for (let k = 0; k < 7; k++) expected[(h*4+m)*8+n] += weights[k] / sum * values[(h*7+k)*8+n]
    }
    for (let i = 0; i < 2; i++) assertClose((await model.forwardAsync({x:scores})).output, expected, 2e-5)
    restored = pg.Model.load(await model.saveAsync())
    assertClose((await restored.forwardAsync({x:scores})).output, expected, 2e-5)
  } finally {
    if (restored) await restored.dispose()
    await model.dispose(); x.dispose(); v.dispose()
  }
}

async function checkOnnxImport(pg) {
  const bytes = str => Uint8Array.from(typeof atob === 'function' ? atob(str) : Buffer.from(str, 'base64').toString('binary'), c => c.charCodeAt(0))
  async function forward(model, inputs, name) {
    const state = pg._core.Module && pg._core.Module.__polygradWebGpuState
    if (!state || name !== 'encoder') return model.forwardAsync(inputs)
    const queue = state.device.queue, submit = queue.submit
    let submissions = 0
    queue.submit = function(...args) { submissions++; return submit.apply(this, args) }
    try {
      const result = await model.forwardAsync(inputs)
      assert(submissions > 0, 'imported ONNX encoder submitted no WebGPU work')
      return result
    } finally { queue.submit = submit }
  }
  for (const item of onnxFixture.cases) {
    const source = bytes(item.onnx)
    const external = Object.fromEntries(Object.entries(item.external).map(([k,v]) => [k, bytes(v)]))
    const model = pg.Model.fromONNX(source, { dimensions: item.dimensions, externalData: external })
    let restored
    try {
      source.fill(0)
      for (const data of Object.values(external)) data.fill(0)
      const inputs = Object.fromEntries(Object.entries(item.inputs).map(([k,v]) => [k, new Float32Array(v.values)]))
      const first = await forward(model, inputs, item.name)
      for (const [name, value] of Object.entries(item.outputs)) assertClose(first[name], value.values, 2e-5)
      restored = pg.Model.load(await model.saveAsync())
      const result = await forward(restored, inputs, item.name)
      for (const [name, value] of Object.entries(item.outputs)) assertClose(result[name], value.values, 2e-5)
      assert(!(model instanceof pg.models.Transformer), 'ONNX imported as Transformer')
    } finally {
      if (restored) await restored.dispose()
      await model.dispose()
    }
  }
  let rejected = false
  try { pg.Model.fromONNX(new Uint8Array([0x80])) } catch (e) { rejected = /ONNX/.test(e.message) }
  assert(rejected, 'truncated ONNX accepted')
  rejected = false
  const external = onnxFixture.cases.find(item => item.name === 'external')
  try { pg.Model.fromONNX(bytes(external.onnx)) } catch (e) { rejected = /was not supplied/.test(e.message) }
  assert(rejected, 'missing external weights accepted')
}

const llamaFixture = require('../../test/fixtures/llama.json')
const componentFixture = require('../../test/fixtures/model_components.json')
const componentOracle = require('../../test/fixtures/model_components_expected.json')

async function checkCachedLlama(pg) {
  async function appendRestored(model, tokens) {
    const state = pg._core.Module && pg._core.Module.__polygradWebGpuState
    if (!state) return model.appendTokensAsync(tokens)
    const queue = state.device.queue, submit = queue.submit, wait = queue.onSubmittedWorkDone
    let submissions = 0, waits = 0
    queue.submit = function(...args) { submissions++; return submit.apply(this, args) }
    queue.onSubmittedWorkDone = function(...args) { waits++; return wait.apply(this, args) }
    try {
      const result = await model.appendTokensAsync(tokens)
      assert(submissions > 0, 'restored WebGPU append submitted no GPU work')
      assert(waits === 0, 'ordinary WebGPU dispatch waited for the entire GPU queue')
      return result
    } finally { queue.submit = submit; queue.onSubmittedWorkDone = wait }
  }
  for (const item of llamaFixture.cases) {
    const config = {...item.config, max_seq_len:5}
    const plain = await pg.models.LlamaAsync(config)
    pg.Tensor.manual_seed(0)
    const cached = await pg.models.LlamaAsync({...config, cache_capacity:5, prefill_chunk_size:4})
    let restored
    try {
      for (const [name, shape] of Object.entries(item.weights)) {
        const offset = Array.from(name).reduce((s,c)=>s+c.charCodeAt(0),0)
        const values = Float32Array.from({length:shape.reduce((a,b)=>a*b,1)},(_,i)=>
          (shape.length===1?1:0)+((i*7+offset)%23-11)*0.017)
        await plain.writeBufferAsync(name, values)
        await cached.writeBufferAsync(name, values)
      }
      assert(cached instanceof pg.Model && cached instanceof pg.models.Transformer, 'cached factory did not specialize')
      assert(typeof plain.appendTokens === 'undefined', 'generic Model exposes generation')
      const tokens = new Int32Array([1,4,7,2,9]), vocab = config.vocab_size
      const expected = (await plain.forwardAsync({tokens})).logits
      const initial = await cached.saveAsync()
      for (const n of [1,3,2,4]) {
        await cached.resetTransientAsync()
        const out = await cached.callAsync('prefill', {tokens_prefill: tokens.slice(0,n)},
          {controls: {start_pos:0}})
        assertClose(out.logits_prefill, expected.slice((n-1)*vocab,n*vocab), 3e-5)
      }
      for (const chunks of [[5], [1,3,1], [2,2,1], [1,1,1,1,1]]) {
        await cached.resetTransientAsync()
        let pos = 0
        for (const n of chunks) {
          const result = await cached.appendTokensAsync(tokens.slice(pos,pos+n))
          pos += n
          assert(cached.decodePosition === pos, 'incorrect committed decoder position')
          assertClose(result, expected.slice((pos-1)*vocab,pos*vocab), 3e-5)
        }
        let error
        try { await cached.appendTokensAsync(tokens.slice(0,1)) } catch (e) { error=e }
        assert(error && /capacity/.test(error.message), 'capacity overflow accepted')
      }
      const saved = await cached.saveAsync()
      assert(saved.length === initial.length && saved.every((x,i)=>x===initial[i]), 'cache history was exported')
      restored = pg.Model.load(saved)
      assert(typeof restored.appendTokens === 'undefined', 'generic load specialized implicitly')
      const resaved = await restored.saveAsync()
      assert(saved.length === resaved.length && saved.every((x,i)=>x===resaved[i]), 'generic load lost metadata')
      assert(pg.models.Transformer.fromModel(restored) === restored, 'specialization created a second owner')
      assert(restored.decodePosition === 0, 'imported decoder has stale position')
      assertClose(await appendRestored(restored, tokens.slice(0,2)), expected.slice(vocab,2*vocab), 3e-5)
      const before = await restored.readBufferAsync('model.layers.0.cache_kv')
      let overflow
      try {
        await restored.callAsync('prefill', {tokens_prefill:tokens.slice(0,3)},
          {controls:{start_pos:3}})
      } catch (e) { overflow = e }
      assert(overflow && /view bounds.*start_pos=3.*tokens_prefill\[1\]=3/.test(overflow.message),
        'combined position/width overflow must report its bound values')
      assertClose(await restored.readBufferAsync('model.layers.0.cache_kv'), before, 0)
      // A read must not perform deferred placement and invalidate a live decoder.
      await restored.readBufferAsync('model.layers.0.cache_kv')
      assert(restored.decodePosition === 2, 'readback changed restored decoder placement')
      assertClose(await restored.appendTokensAsync(tokens.slice(2)), expected.slice(-vocab), 3e-5)
      await restored.dispose()
      restored = pg.models.Transformer.load(saved)
      assert(typeof restored.reset === 'function' && typeof restored.resetAsync === 'function',
        'Transformer must expose reset in both frontend styles')
      await restored.resetAsync()
      assertClose(await appendRestored(restored, tokens.slice(0,2)), expected.slice(vocab,2*vocab), 3e-5)
      await restored.readBufferAsync('model.layers.0.cache_kv')
      assert(restored.decodePosition === 2, 'reset failed to establish restored placement')
      await cached.resetTransientAsync()
      await cached.appendTokensAsync(tokens)
      for (const prompt of [tokens, tokens.slice(0,3), new Int32Array([1,4,3]), new Int32Array([9])]) {
        await restored.resetTransientAsync()
        const reference = await restored.appendTokensAsync(prompt)
        assertClose(await cached.prefillTokensAsync(prompt), reference, 3e-5)
        assert(cached.decodePosition === prompt.length, 'prefix reuse cursor mismatch')
      }
      const cache = await cached.readBufferAsync('model.layers.0.cache_kv')
      cached.rewind(0)
      assertClose(await cached.readBufferAsync('model.layers.0.cache_kv'), cache, 0)
      let badRewind
      try { cached.rewind(1) } catch (error) { badRewind = error }
      assert(badRewind && /prefix/.test(badRewind.message), 'invalid rewind accepted')
      const prompt = new Int32Array([1,4]), generated = [], teacher = [1,4]
      for (let step = 0; step < 3; step++) {
        const padded = new Int32Array(5)
        padded.set(teacher)
        const values = (await plain.forwardAsync({tokens:padded})).logits.slice((teacher.length-1)*vocab,teacher.length*vocab)
        teacher.push(values.indexOf(Math.max(...values)))
      }
      await restored.resetTransientAsync()
      let retiredBaseline
      for await (const token of restored.generate(prompt, {temperature:0, maxTokens:99})) {
        if (generated.length === 2)
          assert(pg.stats().coreStats.memUsed <= retiredBaseline, 'decode retained dropped storage')
        generated.push(token)
        if (generated.length === 2) {
          retiredBaseline = pg.stats().coreStats.memUsed
          const scratch = pg.Tensor.empty([1 << 20])
          try {
            await scratch.toArrayAsync()
            assert(pg.stats().coreStats.memUsed >= (4 << 20), 'scratch storage was not allocated')
          } finally { await scratch.dispose() }
        }
      }
      assert(JSON.stringify(generated) === JSON.stringify(teacher.slice(2)), 'generated tokens differ from uncached teacher')
      assert(restored.decodePosition === 4, 'last generated token should not be consumed yet')
      await restored.dispose()
      restored = pg.models.Transformer.load(saved)
      // No read/save/reset before generation: those would hide missing deferred placement.
      const gpu = pg._core.Module && pg._core.Module.__polygradWebGpuState
      const queue = gpu && gpu.device.queue, submit = queue && queue.submit
      let submissions = 0
      if (queue) queue.submit = function(...args) { submissions++; return submit.apply(this,args) }
      try {
        const cold = []
        for await (const token of restored.generate(prompt, {maxTokens:1})) cold.push(token)
        assert(cold.length === 1 && cold[0] === teacher[2], 'cold restored generation mismatch')
        if (queue) assert(submissions > 0, 'cold restored generation submitted no WebGPU work')
      } finally { if (queue) queue.submit = submit }
      if (vocab === 11) {
        // Freshly loaded RNG starts at the checkpoint, not at a conversation reset.
        await restored.dispose()
        restored = pg.models.Transformer.load(initial)
        await restored.resetAsync()
        const draws = []
        for (let i = 0; i < 16; i++) {
          if (i === 8) await restored.resetAsync()
          const out = await restored.callAsync('sample', {
            'sampling.logits':Float32Array.from({length:11},(_,j)=>-1+j/5),
            'sampling.temperature':new Float32Array([0.7])})
          draws.push(out['sampling.token'][0])
        }
        assert(JSON.stringify(draws) === '[10,10,10,10,9,10,7,5,6,6,2,0,0,10,10,10]',
          'conversation reset must not rewind pinned Gumbel draws')
      }
    } finally {
      if (restored) await restored.dispose()
      await cached.dispose()
      await plain.dispose()
    }
  }
}

async function checkIntegerControls(pg) {
  const invalidPosition = pg.uop.variable('invalid_position', 0, 3)
  const invalidInput = pg.Tensor.empty([1]), target = pg.Tensor.empty([1])
  try {
    for (const usesControl of [false, true]) {
      let authorCalls = 0, model, error
      try {
        model = await pg.Model.fromCallableAsync(({x, pos}) => {
          authorCalls++
          if (!usesControl) return x.add(1)
          const scalar = new pg.Tensor(pos)
          try { return x.add(scalar) } finally { scalar.dispose() }
        }, {inputs:{x:invalidInput}, targets:{y:target}, controls:{pos:invalidPosition},
            loss:(out,{y})=>out.sub(y).square().mean()})
      } catch (e) { error=e }
      finally { if (model) await model.dispose() }
      assert(error && /training.*controls/.test(error.message), 'controlled training accepted')
      assert(authorCalls === 0, 'unsupported training invoked the author')
    }
    let model, error
    try {
      model = await pg.Model.fromCallableAsync(({x})=>x.add(1),
        {inputs:{x:invalidInput}, controls:{pos:invalidPosition}})
    } catch (e) { error=e }
    finally { if (model) await model.dispose() }
    assert(error && /control.*pos.*not used/.test(error.message), 'unused control accepted')
  } finally {
    await target.dispose(); await invalidInput.dispose(); invalidPosition.dispose()
  }
  const position = pg.uop.variable('model_position', 0, 11)
  const x = pg.Tensor.empty([1])
  let model, restored
  try {
    model = await pg.Model.fromCallableAsync(({x, position}) => {
      const scalar = new pg.Tensor(position)
      try { return x.add(scalar) } finally { scalar.dispose() }
    },
      {inputs:{x}, controls:{position}})
    restored = pg.Model.load(await model.saveAsync())
    for (const m of [model, restored]) {
      for (const p of [0, 5, 11, 2]) {
        const out = await m.callAsync('forward', {x:new Float32Array([3])}, {controls:{position:p}})
        assertClose(out.output, [3+p], 0)
      }
      const input = new pg.Tensor(new Float32Array([3]))
      let result
      try {
        result = (await m.callAsync('forward', {x:input}, {controls:{position:5}})).output
        await m.callAsync('forward', {x:new Float32Array([9])}, {controls:{position:2}})
        assertClose(await result.toArrayAsync(), [8], 0)
      } finally {
        if (result) await result.dispose()
        await input.dispose()
      }
      for (const controls of [{}, {unknown:1}, {position:-1}, {position:12}, {position:1.5}]) {
        let error
        try { await m.callAsync('forward', {x:new Float32Array([9])}, {controls}) } catch (e) { error=e }
        assert(error && /control/.test(error.message), 'invalid control accepted')
      }
    }
  } finally {
    if (restored) await restored.dispose()
    if (model) await model.dispose()
    await x.dispose()
    position.dispose()
  }
  const offset = pg.uop.variable('read_position', 0, 7), width = pg.uop.constant(2)
  const values = new pg.Tensor(new Float32Array([0,1,2,3,4,5,6,7]))
  const readInput = pg.Tensor.empty([2])
  let reader, readerCopy
  try {
    await values.realizeAsync()
    reader = await pg.Model.fromCallableAsync(({x, p}) => {
      // Tensor.shrink currently takes static bounds in JS. Use its existing
      // symbolic UOp interface to exercise Model admission on both JS cores.
      const roots = [values.uopLogical, values.uopPhysical]
      const views = roots.map(root => pg._core.ffi.poly_uop_shrink_symbolic(
        root.ctx, root.raw, [p.raw], [width.raw], 1))
      const view = values._makeResultFromCore(values._coreCreateWithRoots(views[0], views[1], 0))
      try { return x.mul(view) } finally { view.dispose(); roots.forEach(root => root.dispose()) }
    }, {inputs:{x:readInput}, params:{buf:values}, controls:{p:offset}})
    readerCopy = pg.Model.load(await reader.saveAsync())
    for (const m of [reader, readerCopy]) {
      let error
      try {
        await m.callAsync('forward', {x:new Float32Array([1,1])}, {controls:{p:7}})
      } catch (e) { error=e }
      assert(error && /view bounds.*p=7/.test(error.message), 'out-of-bounds read accepted')
      assertClose((await m.callAsync('forward', {x:new Float32Array([1,1])}, {controls:{p:6}})).output, [6,7], 0)
    }
  } finally {
    if (readerCopy) await readerCopy.dispose()
    if (reader) await reader.dispose()
    await readInput.dispose(); await values.dispose(); width.dispose(); offset.dispose()
  }
  const p = pg.uop.variable('local_position', 0, 4)
  const input = pg.Tensor.empty([1]), scalar = new pg.Tensor(p), plain = input.add(1), shifted = input.add(scalar)
  let local, imported
  try {
    local = await pg.Model.fromTensors({inputs:{x:input}, outputs:{plain,shifted}, controls:{position:p},
      entrypoints:[{name:'plain', inputs:['x'], outputs:['plain'], controls:{}},
                   {name:'shift', inputs:['x'], outputs:['shifted']}]})
    imported = pg.Model.load(await local.saveAsync())
    for (const m of [local, imported]) {
      assertClose((await m.callAsync('plain', {x:new Float32Array([2])})).plain, [3], 0)
      assertClose((await m.callAsync('shift', {x:new Float32Array([2])}, {controls:{position:3}})).shifted, [5], 0)
    }
  } finally {
    if (imported) await imported.dispose()
    if (local) await local.dispose()
    await shifted.dispose(); await plain.dispose(); await scalar.dispose(); await input.dispose(); p.dispose()
  }
}

async function checkBoundedComponents(pg) {
  const model = await pg.models.GraphAsync(componentFixture)
  let restored
  try {
    await model.writeBufferAsync('nodes.embedding.weight', new Float32Array(componentOracle.table.flat()))
    await model.writeBufferAsync('nodes.head.weight', new Float32Array(componentOracle.linear.flat()))
    for (const batch of [3, 1]) {
      const tokens = Int32Array.from({length:batch*3}, (_,i)=>i%4)
      const result = await model.forwardAsync({tokens})
      const expected = componentOracle.cases.find(c=>c.batch===batch)
      assertClose(result.prediction, expected.prediction.flat(2), 2e-5)
      assertClose(result.mean, [expected.mean], 2e-5)
    }
    const bytes = await model.saveAsync()
    restored = pg.Model.load(bytes)
    const saved = await restored.saveAsync()
    assert(saved.length===bytes.length && saved.every((b,i)=>b===bytes[i]), 'component round trip is not canonical')
    assertClose((await restored.forwardAsync({tokens:new Int32Array([0,1,2])})).prediction,
      componentOracle.cases[0].prediction.flat(2), 2e-5)
    let error
    try { await model.forwardAsync({tokens:new Int32Array(12)}) } catch (e) { error=e }
    assert(error && /bound|extent/.test(error.message), 'out-of-bound component input was accepted')
  } finally {
    if (restored) await restored.dispose()
    await model.dispose()
  }
  const mean = await pg.models.SequentialAsync({
    input:{name:'x',dtype:'int32',shape:[{name:'n',min:1,max:4},2]},
    layers:[{name:'avg',type:'mean'}],output:'y'
  })
  try {
    for (const rows of [4,1,3]) {
      const x = Int32Array.from({length:rows*2},(_,i)=>i)
      assertClose((await mean.forwardAsync({x})).y, [(rows*2-1)/2], 1e-6)
    }
  } finally { await mean.dispose() }
}

async function checkModelContractErrors(pg) {
  const model = pg.models.MLP({layers:[2,1]})
  try {
    let message = ''
    try { await model.forwardAsync({x:new Int32Array([1,2])}) } catch (e) { message = e.message }
    assert(/x.*expected float32.*int32/.test(message), `dtype diagnostic: ${message}`)
    message = ''
    try { await model.trainStepAsync({x:new Float32Array([1,2])}) } catch (e) { message = e.message }
    assert(/objective/.test(message), `objective diagnostic: ${message}`)
    const bytes = await model.saveAsync({includeOptimizer:false})
    const imported = pg.Model.load(bytes)
    try {
      const saved = await imported.saveAsync({includeOptimizer:false})
      assert(saved.length === bytes.length && saved.every((b,i)=>b===bytes[i]), 'noncanonical round trip')
    } finally { await imported.dispose() }
  } finally { await model.dispose() }
}

async function checkFamilyRegistry(pg) {
  const types = Object.fromEntries(pg.models.list().map(type => [type.name, type]))
  assert(types.GPT2.constructible && types.GPT2.hf && types.GPT2.gguf, 'GPT2 capabilities')
  assert(types.Llama.hf && !types.Llama.gguf, 'Llama capabilities')
  assert(types.Qwen3.gguf && !types.Qwen3.hf && !types.Qwen3.constructible, 'Qwen3 capabilities')
  let importMessage = ''
  try { pg.models.Qwen3({}) } catch (e) { importMessage = e.message }
  assert(/Qwen3 is import-only; use Model.fromGGUF/.test(importMessage), importMessage)
  importMessage = ''
  try { await pg.models.Qwen3Async({}) } catch (e) { importMessage = e.message }
  assert(/Qwen3 is import-only; use Model.fromGGUF/.test(importMessage), importMessage)
  const configs = {MLP:{layers:[2,1]}, TabM:{layers:[2,1],n_ensemble:2},
    NAM:{n_features:2,hidden_sizes:[2]}, GPT2:{vocab_size:8,n_embd:4,n_head:2,n_layer:1,n_positions:2},
    DistilGPT2:{vocab_size:8,n_embd:4,n_head:2,n_positions:2}}
  for (const [family, config] of Object.entries(configs)) {
    assert(typeof pg.models[family] === 'function', `missing ${family}`)
    const tagged = {format:'poly.modeldef@1',type:family.toLowerCase(),...config}
    const model = pg.device === 'webgpu' ? await pg.models[family+'Async'](tagged) : new pg.Model(tagged)
    try {
      assert(model.bindings().length > 0, `empty ${family}`)
      if (family === 'DistilGPT2') {
        const explicit = await pg.models.GPT2Async({...config,n_layer:6})
        try {
          assert(model.paramCount === 76 && explicit.paramCount === 76, 'six-block preset')
          for (let i = 0; i < model.paramCount; i++) {
            const name = model.paramName(i)
            const count = model.paramShape(i).reduce((a,b)=>a*b,1)
            const values = Float32Array.from({length:count},(_,j)=>(j%11-5)/16)
            await model.writeBufferAsync(name,values)
            await explicit.writeBufferAsync(name,values)
          }
          const a = await model.saveAsync(), b = await explicit.saveAsync()
          assert(a.length === b.length && a.every((v,i)=>v===b[i]), 'preset changed bundle bytes')
          const inputs = {x:new Int32Array([1,2]),positions:new Int32Array([0,1])}
          assertClose((await model.forwardAsync(inputs)).output,
            (await explicit.forwardAsync(inputs)).output,0)
        } finally { await explicit.dispose() }
      }
    }
    finally { await model.dispose() }
  }
  let message = ''
  try { await pg.models.TabMAsync({layers:[2,1],n_ensemble:0}) } catch(e) { message=e.message }
  assert(/n_ensemble/.test(message), `missing factory diagnostic: ${message}`)
}

async function checkQwenRotaryState(pg) {
  const fixture = require('../../test/fixtures/qwen3.json')
  const bytes = Uint8Array.from(atob(fixture.gguf), c => c.charCodeAt(0))
  const model = pg.Model.fromGGUF(bytes, { maxSeqLen: 4 })
  let restored
  try {
    assert(JSON.stringify(model.entrypoints()[0].inputs) === '["x"]', 'Qwen accepts only tokens')
    for (const name of ['rope_cos', 'rope_sin']) {
      const binding = model.bindings().find(b => b.name === name)
      assert(binding && binding.role === pg.Model.ROLE_AUX && !binding.trainable, `${name} must be AUX`)
      assert(JSON.stringify(binding.shape) === '[1,1,4,4]', `${name} broadcast shape`)
      assertClose(await model.readBufferAsync(name), fixture[name].flat(), 2e-7)
    }
    const bundle = await model.saveAsync()
    let loadError = ''
    try { pg.models.Transformer.load(bundle).dispose() }
    catch (error) { loadError = error.message }
    assert(/Transformer/.test(loadError), 'uncached bundle accepted as Transformer')
    assert(pg._activeAsync === 0, 'failed bundle adoption left asynchronous cleanup behind')
    restored = pg.Model.load(bundle)
    const saved = await restored.saveAsync()
    assert(bundle.length === saved.length && bundle.every((v, i) => v === saved[i]), 'Qwen canonical round trip')
    const x = new Int32Array(fixture.tokens.flat())
    for (const current of [model, restored]) {
      assert(JSON.stringify(current.entrypoints()[0].inputs) === '["x"]', 'round trip preserves signature')
      assertClose((await current.forwardAsync({ x })).output, fixture.logits.flat(2), 3e-5)
    }
    const before = await model.readBufferAsync('rope_cos')
    await restored.writeBufferAsync('rope_cos', new Float32Array(before.length))
    assertClose(await model.readBufferAsync('rope_cos'), before, 0)
  } finally {
    if (restored) await restored.dispose()
    await model.dispose()
  }
  const cached = pg.models.Transformer.fromGGUF(bytes,
    { maxSeqLen: 4, cacheCapacity: 4, prefillChunkSize: 2 })
  assert(cached instanceof pg.models.Transformer, 'GGUF loader did not specialize')
  let loaded
  try {
    const tokens = new Int32Array(fixture.tokens.flat())
    for (let i = 0; i < tokens.length; i++)
      assertClose(await cached.appendTokensAsync(tokens.slice(i, i + 1)), fixture.logits[0][i], 3e-5)
    let message = ''
    try { await cached.appendTokensAsync(tokens.slice(0, 1)) } catch (error) { message = error.message }
    assert(/capacity/.test(message), message)
    loaded = pg.models.Transformer.load(await cached.saveAsync())
    assert(loaded.decodePosition === 0, 'import starts with an empty cache')
    assertClose(await loaded.prefillTokensAsync(tokens), fixture.logits[0][3], 3e-5)
  } finally {
    if (loaded) await loaded.dispose()
    await cached.dispose()
  }
  let adoptionError = ''
  try { pg.models.Transformer.fromGGUF(bytes).dispose() }
  catch (error) { adoptionError = error.message }
  assert(/Transformer/.test(adoptionError), 'uncached GGUF accepted as Transformer')
  assert(pg._activeAsync === 0, 'failed adoption left asynchronous cleanup behind')
  for (const opts of [{cacheCapacity: -1}, {cacheCapacity: 4, prefillChunkSize: 5},
                      {prefillChunkSize: 2}, {cacheCapacity: 4, maxBatch: 2}]) {
    let message = ''
    try { pg.Model.fromGGUF(bytes, opts).dispose() } catch (error) { message = error.message }
    assert(/cache|prefill/.test(message), message)
  }
}

async function checkVisionModels(pg) {
  for (const packed of require('../../test/fixtures/vision.json').cases) {
    const item = await require('./vision_fixture').expandVisionCase(packed)
    const weights = item.weights
    const model = pg.Model.fromHF(new TextEncoder().encode(JSON.stringify(item.config)), [weights], {maxBatch: 2})
    let restored
    try {
      const inputs = Object.fromEntries(Object.entries(item.inputs).map(([k,v]) =>
        [k, k === 'input_ids' ? new Int32Array(v.flat(Infinity)) : new Float32Array(v.flat(Infinity))]))
      const outputs = await model.forwardAsync(inputs)
      for (const [k,v] of Object.entries(item.outputs)) assertClose(outputs[k], v.flat(Infinity), 1e-4)
      const bundle = await model.saveAsync()
      restored = pg.Model.load(bundle)
      const saved = await restored.saveAsync()
      assert(bundle.length === saved.length && bundle.every((v,i) => v === saved[i]), `${item.name} canonical bundle`)
      const roundtrip = await restored.forwardAsync(inputs)
      for (const [k,v] of Object.entries(item.outputs)) assertClose(roundtrip[k], v.flat(Infinity), 1e-4)
      if (item.name === 'CLIP') {
        assertClose((await model.callAsync('encode_image', {pixel_values:inputs.pixel_values})).image_embeds, outputs.image_embeds, 1e-5)
        assertClose((await model.callAsync('encode_text', {input_ids:inputs.input_ids})).text_embeds, outputs.text_embeds, 1e-5)
        for (const name of ['visual_projection.weight', 'text_projection.weight']) {
          const values = await model.readBufferAsync(name)
          await model.writeBufferAsync(name, new Float32Array(values.length))
        }
        for (const values of Object.values(await model.forwardAsync(inputs))) {
          assert(Array.from(values).every(v => v === 0), 'CLIP zero projection must normalize to finite zeros')
        }
      }
    } finally {
      if (restored) await restored.dispose()
      await model.dispose()
    }
  }
}

async function checkLlamaFamily(pg) {
  const gpu = pg.device === 'webgpu'
  for (const item of llamaFixture.cases) {
    const model = gpu ? await pg.models.LlamaAsync(item.config) : pg.models.Llama(item.config)
    let restored, imported
    try {
      for (const invoke of [() => model.forwardAsync({tokens:new Int32Array(item.tokens)}),
        () => model.saveAsync(), () => model.exportWeightsAsync(), () => model.exportIR()]) {
        let message = ''
        try { await invoke() } catch (e) { message = e.message }
        assert(/weight.*not initialized/.test(message), `unloaded Llama: ${message}`)
      }
      const header = {}, parts = []
      let offsetBytes = 0
      for (const [name, shape] of Object.entries(item.weights)) {
        const offset = Array.from(name).reduce((s,c)=>s+c.charCodeAt(0),0)
        const values = Float32Array.from({length:shape.reduce((a,b)=>a*b,1)},(_,i)=>
          (shape.length===1?1:0)+((i*7+offset)%23-11)*0.017)
        await model.writeBufferAsync(name, values)
        header[name] = {dtype:'F32',shape,data_offsets:[offsetBytes,offsetBytes+values.byteLength]}
        parts.push(new Uint8Array(values.buffer))
        offsetBytes += values.byteLength
      }
      // The real TinyStories safetensors retains the head, not the embedding.
      if (item.config.tie_word_embeddings) {
        header['lm_head.weight'] = header['model.embed_tokens.weight']
        delete header['model.embed_tokens.weight']
      }
      const headerBytes = new TextEncoder().encode(JSON.stringify(header))
      const checkpoint = new Uint8Array(8+headerBytes.length+offsetBytes)
      new DataView(checkpoint.buffer).setBigUint64(0,BigInt(headerBytes.length),true)
      checkpoint.set(headerBytes,8)
      offsetBytes = 8+headerBytes.length
      for (const part of parts) { checkpoint.set(part,offsetBytes); offsetBytes += part.length }
      imported = pg.Model.fromHF(new TextEncoder().encode(JSON.stringify(item.config)),[checkpoint],{maxSeqLen:3})
      assertClose(await model.readBufferAsync('freqs_cos'),item.freqs_cos,2e-6)
      assertClose(await model.readBufferAsync('freqs_sin'),item.freqs_sin,2e-6)
      const tokens = new Int32Array([1,4,2])
      const first = (await model.forwardAsync({tokens})).logits
      assertClose(first,item.logits,2e-5)
      assertClose((await imported.forwardAsync({tokens})).logits,item.logits,2e-5)
      const changed = (await model.forwardAsync({tokens:new Int32Array([1,4,7])})).logits
      assertClose(changed.slice(0,22),first.slice(0,22),2e-6)
      restored = pg.Model.load(await model.saveAsync({includeOptimizer:false}))
      assertClose((await restored.forwardAsync({tokens})).logits,item.logits,2e-5)
      const name = 'model.embed_tokens.weight', before = await model.readBufferAsync(name)
      await restored.writeBufferAsync(name,new Float32Array(before.length))
      assertClose(await model.readBufferAsync(name),before,0)
      if (item.config.tie_word_embeddings)
        assertClose(await restored.readBufferAsync('lm_head.weight'),new Float32Array(before.length),0)
    } finally {
      if (restored) await restored.dispose()
      if (imported) await imported.dispose()
      await model.dispose()
    }
  }
}

const compositionFixture = require('../../test/fixtures/model_definition.json')
const quantizedFixture = require('../../test/fixtures/gguf_quantized_blocks.json')

async function checkModelCheckpointReplacement(pg) {
  const model = pg.models.MLP({layers:[2,1],loss:'none',seed:3})
  try {
    const name = 'layers.0.weight'
    const original = await model.readBufferAsync(name)
    const weights = await model.exportWeightsAsync()
    await model.writeBufferAsync(name, new Float32Array(original.length))
    if (pg.device === 'webgpu') {
      let rejected = false
      try { model.importWeights(weights) } catch (e) { rejected = /Async/.test(e.message) }
      assert(rejected, 'synchronous GPU replacement must reject before entering Wasm')
    }
    const pending = model.importWeightsAsync(weights)
    weights.fill(0) // Queued imports own their input bytes.
    await pending
    assertClose(await model.readBufferAsync(name), original, 0)
    let rejected = false
    try { await model.importWeightsAsync(new Uint8Array([1,2,3])) } catch (_) { rejected = true }
    assert(rejected, 'malformed checkpoint must reject')
    assertClose(await model.readBufferAsync(name), original, 0)
  } finally { await model.dispose() }
}

async function checkModelStatefulCapture(pg) {
  const gpu = String(pg.device).toLowerCase() === 'webgpu'
  const net = {bn:new pg.nn.BatchNorm(2), forward({x}) { return {prediction:this.bn.call(x)} }}
  const mode = pg.Tensor.training
  const opts = {inputs:{x:pg.Tensor.empty([2,2])},targets:{y:pg.Tensor.empty([2,2])},
    loss:(out,{y})=>out.prediction.sub(y).square().mean()}
  const model = gpu ? await pg.Model.fromCallableAsync(net,opts) : new pg.Model(net,opts)
  let restored
  try {
    assert(pg.Tensor.training === mode,'capture changed authoring mode')
    assertClose(await model.readBufferAsync('bn.runningMean'),[0,0],0)
    assert(Number((await model.readBufferAsync('bn.numBatchesTracked'))[0])===0)
    model.setOptimizer('sgd',0.01)
    const x = new Float32Array([1,2,3,6]), y = new Float32Array(4)
    await model.trainStepAsync({x,y})
    assertClose(await model.readBufferAsync('bn.runningMean'),[0.2,0.4],1e-6)
    assertClose(await model.readBufferAsync('bn.runningVar'),[1.1,1.7],1e-6)
    assert(Number((await model.readBufferAsync('bn.numBatchesTracked'))[0])===1)
    assertClose(await net.bn.runningMean.toArrayAsync(),[0,0],0)
    const weight = await model.readBufferAsync('bn.weight'), bias = await model.readBufferAsync('bn.bias')
    const expected = Array.from(x,(v,i)=>(v-[0.2,0.4][i%2])/Math.sqrt([1.1,1.7][i%2]+1e-5)*weight[i%2]+bias[i%2])
    assertClose((await model.forwardAsync({x})).prediction,expected,1e-5)
    assert(Number((await model.readBufferAsync('bn.numBatchesTracked'))[0])===1)
    restored = pg.Model.load(await model.saveAsync())
    restored.setOptimizer('sgd',0.01)
    const a = await model.trainStepAsync({x,y:x}), b = await restored.trainStepAsync({x,y:x})
    assert(Math.abs(a-b)<1e-5,'restored training diverged')
    assert(Number((await restored.readBufferAsync('bn.numBatchesTracked'))[0])===2)
  } finally {
    if(restored) await restored.dispose()
    await model.dispose()
    for(const tensor of Object.values(net.bn)) if(tensor && tensor.dispose) await tensor.dispose()
    for(const tensor of [...Object.values(opts.inputs),...Object.values(opts.targets)]) await tensor.dispose()
  }
}

async function checkModelCaptureRng(pg) {
  const gpu = String(pg.device).toLowerCase() === 'webgpu'
  pg.Tensor.manual_seed(123)
  const controlTensor = pg.Tensor.rand(16)
  const control = await controlTensor.toArrayAsync()
  await controlTensor.dispose()
  pg.Tensor.manual_seed(123)
  const weight = new pg.Tensor([1.0], {dtype:'float32'}), input = pg.Tensor.empty(16), target = pg.Tensor.empty(16)
  const opts = {inputs:{x:input},targets:{y:target},params:{weight,alias:weight},loss:(out,{y})=>out.sub(y).square().mean()}
  const author = ({x})=>x.mul(weight).dropout(0.5)
  const model = gpu ? await pg.Model.fromCallableAsync(author,opts) : new pg.Model(author,opts)
  let restored, sibling, roundtripped
  try {
    const after = pg.Tensor.rand(16)
    try { assertClose(await after.toArrayAsync(),control,0) } finally { await after.dispose() }
    const counters = Array.from({length:model.bufCount},(_,i)=>model.bufName(i))
      .filter(name=>name.startsWith('__rng.') && name.endsWith('.counter'))
    assert(counters.length===1,'capture must own one RNG counter')
    const counter = counters[0]
    assertClose(await model.readBufferAsync(counter),[0,0],0)
    model.setOptimizer('sgd',0.01)
    const x = new Float32Array(16).fill(1), y = new Float32Array(16)
    await model.trainStepAsync({x,y})
    const state = await model.readBufferAsync(counter)
    assert(state.some(v=>v!==0),'training did not advance RNG')
    await model.forwardAsync({x})
    assertClose(await model.readBufferAsync(counter),state,0)
    const bytes = await model.saveAsync()
    restored = pg.Model.load(bytes)
    sibling = pg.Model.load(bytes)
    roundtripped = pg.Model.load(await restored.saveAsync())
    const siblingWeight = await sibling.readBufferAsync('weight')
    for (const copy of [restored,roundtripped]) copy.setOptimizer('sgd',0.01)
    for(let i=0;i<2;i++) {
      const a=await model.trainStepAsync({x,y})
      for (const copy of [restored,roundtripped]) {
        const b=await copy.trainStepAsync({x,y})
        assert(Math.abs(a-b)<1e-5,'checkpoint changed RNG continuation')
        assertClose(await model.readBufferAsync(counter),await copy.readBufferAsync(counter),0)
        assertClose(await model.readBufferAsync('weight'),await copy.readBufferAsync('weight'),0)
        assertClose(await copy.readBufferAsync('alias'),await copy.readBufferAsync('weight'),0)
      }
      assertClose(await sibling.readBufferAsync(counter),state,0)
      assertClose(await sibling.readBufferAsync('weight'),siblingWeight,0)
    }
    await restored.writeBufferAsync('alias',new Float32Array([71]))
    assertClose(await restored.readBufferAsync('weight'),[71],0)
    assertClose(await sibling.readBufferAsync('weight'),siblingWeight,0)
    await restored.dispose()
    pg.clearScheduleCache(); pg.collect()
    assertClose(await sibling.readBufferAsync(counter),state,0)
    assertClose((await sibling.forwardAsync({x})).output,Array(16).fill(siblingWeight[0]),0)
  } finally {
    if(roundtripped) await roundtripped.dispose()
    if(sibling) await sibling.dispose()
    if(restored) await restored.dispose()
    await model.dispose()
    for(const tensor of [weight,input,target]) await tensor.dispose()
  }
}

async function checkModelCaptureFailure(pg) {
  const gpu = String(pg.device).toLowerCase() === 'webgpu'
  const x = pg.Tensor.empty(1), state = new pg.Tensor([0.0]).is_param_(false)
  const options = {inputs:{x},params:{state}}
  const mode = pg.Tensor.training
  const attempt = async author => {
    let model
    try {
      model = gpu ? await pg.Model.fromCallableAsync(author,options) : new pg.Model(author,options)
      return false
    } catch { return true }
    finally { if(model) await model.dispose() }
  }
  try {
    assert(await attempt(({x})=>{ state.assign(state.add(1)); throw new Error('author failed') }))
    assert(pg.Tensor.training===mode)
    assertClose(await state.toArrayAsync(),[0],0)
    state.is_param_(true)
    assert(await attempt(({x})=>{state.assign(state.add(1)); return x}), 'parameter assignment escaped capture')
    assertClose(await state.toArrayAsync(),[0],0)
    state.is_param_(false)
    let pending
    const rejected = await attempt(({x})=>{
      pending = state.toArrayAsync().catch(()=>{})
      return x
    })
    await pending
    assert(rejected,'capture allowed asynchronous execution to escape its scope')
  } finally { await x.dispose(); await state.dispose() }
}

async function checkModelVariableShapes(pg) {
  await checkModelMultipleAxes(pg)
  const gpu = String(pg.device).toLowerCase() === 'webgpu'
  const n = pg.uop.variable('model_batch', 1, 32), bound = n.bind(17)
  const x = pg.Tensor.empty([bound, 2]), y = pg.Tensor.empty([bound, 2])
  const options = {inputs:{x,y}}
  const author = ({x,y}) => ({prediction:x.mul(2).add(y)})
  const model = gpu ? await pg.Model.fromCallableAsync(author, options) : new pg.Model(author, options)
  let restored, first
  const a = Float32Array.from({length:34}, (_,i)=>i)
  const input = new pg.Tensor(a).reshape(17,2)
  try {
    const pending = model.forwardAsync({x:input,y:{data:a,shape:[17,2]}})
    if (gpu) {
      for (const query of [() => model.bufCurrentShape(0), () => model.bufShapeBounds(0)]) {
        let rejected = false
        try { query() } catch (e) { rejected = /active async work/.test(e.message) }
        assert(rejected, 'metadata query entered suspended Wasm')
      }
    }
    first = (await pending).prediction
    assert(JSON.stringify(first.shape)==='[17,2]', 'result shape must be concrete')
    assertClose(await first.toArrayAsync(), Array.from(a,v=>v*3), 0)
    const small = a.slice(0,6)
    const result = await model.forwardAsync({x:small,y:{data:small,shape:[3,2]}})
    assertClose(result.prediction, Array.from(small,v=>v*3),0)
    assert(JSON.stringify(model.bufCurrentShape(model.findBuf('prediction')))==='[3,2]')
    assert(JSON.stringify(model.bufShapeBounds(model.findBuf('x')))==='[[1,32],[2,2]]')
    const before = await model.readBufferAsync('x')
    let rejected = false
    try { await model.forwardAsync({x:a,y:small}) } catch { rejected=true }
    assert(rejected, 'shared shape variable must reject inconsistent inputs')
    assertClose(await model.readBufferAsync('x'),before,0)
    const bytes = await model.saveAsync({includeOptimizer:false})
    restored = pg.Model.load(bytes)
    const roundtripped = pg.Model.load(await restored.saveAsync({includeOptimizer:false}))
    try {
      for (const copy of [restored,roundtripped]) {
        assert(JSON.stringify(copy.bufShapeBounds(copy.findBuf('x')))==='[[1,32],[2,2]]')
        for (const size of [17,3,11]) {
          const values=Float32Array.from({length:size*2},(_,i)=>i)
          assertClose((await copy.forwardAsync({x:values,y:values})).prediction,Array.from(values,v=>v*3),0)
        }
      }
    } finally { await roundtripped.dispose() }
    await model.dispose()
    pg.clearScheduleCache(); pg.collect()
    assertClose(await first.toArrayAsync(),Array.from(a,v=>v*3),0)
    const w = new pg.Tensor([0.0], {dtype:'float32'})
    const trainingOptions = {inputs:{x:pg.Tensor.empty([bound,1])},
      targets:{y:pg.Tensor.empty([bound,1])}, params:{w},
      loss:(out,{y})=>out.sub(y).square().mean()}
    const trainAuthor=({x})=>x.mul(w)
    const training=gpu ? await pg.Model.fromCallableAsync(trainAuthor,trainingOptions)
      : new pg.Model(trainAuthor,trainingOptions)
    try {
      training.setOptimizer('sgd',0.01)
      let expected=0
      for (const size of [17,3,11]) {
        const values=Float32Array.from({length:size},(_,i)=>i+1)
        const meanSquare=values.reduce((sum,v)=>sum+v*v,0)/size
        const loss=await training.trainStepAsync({x:values,y:values.map(v=>v*2)})
        const wanted=(expected-2)**2*meanSquare
        assert(Math.abs(loss-wanted)<=1e-5*Math.max(1,wanted), 'loss used a stale batch extent')
        expected-=0.02*(expected-2)*meanSquare
        assertClose(await training.readBufferAsync('w'),[expected],1e-4)
      }
    } finally { await training.dispose(); await w.dispose() }
  } finally {
    if(restored) await restored.dispose()
    if(first) await first.dispose()
    await input.dispose(); await model.dispose()
    await x.dispose(); await y.dispose(); bound.dispose(); n.dispose()
  }
}

async function checkModelMultipleAxes(pg) {
  const n = pg.uop.variable('matrix_rows', 1, 4), m = pg.uop.variable('matrix_cols', 1, 5)
  const nb = n.bind(2), mb = m.bind(3)
  const x = pg.Tensor.empty([nb, mb]), y = x.mul(2).add(1)
  let model, restored
  try {
    model = await pg.Model.fromTensors({inputs:{x}, outputs:{out:y}})
    restored = pg.Model.load(await model.saveAsync())
    for (const target of [model, restored]) {
      for (const [rows, cols] of [[2,3],[3,2],[1,4]]) {
        const data = Array.from({length:rows}, (_,i)=>Array.from({length:cols}, (_,j)=>i*cols+j))
        const value = new pg.Tensor(data, {dtype:'float32'})
        let output
        try {
          output = (await target.callAsync('forward', {x:value})).out
          assertClose(await output.toArrayAsync(), data.flat().map(v=>2*v+1), 0)
          assertClose(await target.readBufferAsync('out'), data.flat().map(v=>2*v+1), 0)
          const host = await target.callAsync('forward', {x:{data:Float32Array.from(data.flat()), shape:[rows,cols]}})
          assertClose(host.out, data.flat().map(v=>2*v+1), 0)
        } finally {if(output) await output.dispose(); await value.dispose()}
      }
      let error
      try {await target.callAsync('forward', {x:new Float32Array(6)})} catch(e) {error=e}
      assert(error && /shape/.test(error.message), 'ambiguous flat input needs explicit shape')
    }
  } finally {
    if(restored) await restored.dispose()
    if(model) await model.dispose()
    await y.dispose(); await x.dispose(); nb.dispose(); mb.dispose(); n.dispose(); m.dispose()
  }
}

function checkTokenizerJSON(pg) {
  for (const config of [
    {model:{type:'BPE',vocab:{a:0}}},
    {model:{type:'WordPiece',vocab:{a:0}}},
    {model:{type:'Unigram',vocab:[['a',0]]}}, {}
  ]) {
    let error, unexpected
    try {unexpected = pg.Tokenizer.fromJSON(JSON.stringify(config))} catch(e) {error=e}
    finally {if (unexpected) unexpected.free()}
    assert(error && /BPE|pre-tokenizer/.test(error.message), 'JSON tokenizer must explain the unsupported pipeline')
  }

  const config = JSON.parse(JSON.stringify(require('../../test/fixtures/tokenizer_byte_bpe.json')))
  const jsonTok = pg.Tokenizer.fromJSON(JSON.stringify(config))
  try {
    assertClose(jsonTok.encode('abc ab'), [97,257,32,256], 0)
    assert(jsonTok.decode(jsonTok.encode('é e\u0301')) === 'é e\u0301', 'JSON byte roundtrip')
    assert(jsonTok.bosId === -1 && jsonTok.eosId === -1, 'JSON must not guess special-token roles')
  } finally {jsonTok.free()}
  config.normalizer = {type:'NFC'}
  let error
  try { pg.Tokenizer.fromJSON(JSON.stringify(config)) } catch(e) { error = e }
  assert(error && /NFC/.test(error.message), 'strict JSON import must reject NFC')
  const warn = console.warn, warnings = []
  console.warn = message => warnings.push(message)
  try {
    const approximate = pg.Tokenizer.fromJSON(JSON.stringify(config), {strict:false})
    try {
      assert(approximate.decode(approximate.encode('e\u0301')) === 'e\u0301', 'only NFC is skipped')
      assert(warnings.length === 1 && /NFC normalization skipped/.test(warnings[0]), 'best effort must warn once')
    } finally {approximate.free()}
    config.model.type = 'WordPiece'
    error = null
    try { pg.Tokenizer.fromJSON(JSON.stringify(config), {strict:false}) } catch(e) { error = e }
    assert(error && /BPE/.test(error.message), 'best effort must still reject other algorithms')
    config.model.type = 'BPE'
    config.added_tokens = ['ab', 'abc'].map((content, i) => ({
      id:i ? 258 : 256, content, special:true, normalized:!!i,
      single_word:false, lstrip:false, rstrip:false
    }))
    error = null
    try { pg.Tokenizer.fromJSON(JSON.stringify(config), {strict:false}) } catch(e) { error = e }
    assert(error && /matching phases/.test(error.message), 'mixed added-token matching phases must reject')
  } finally {console.warn = warn}

  // Minimal metadata-only GGUF exercises the public loader without a checkpoint.
  const bytes = []
  const u32 = n => { for (let i=0;i<4;i++) bytes.push((n >>> (8*i)) & 255) }
  const u64 = n => { u32(n); u32(0) }
  const str = s => { const b=new TextEncoder().encode(s); u64(b.length); bytes.push(...b) }
  bytes.push(71,71,85,70); u32(3); u64(0); u64(4)
  str('tokenizer.ggml.tokens'); u32(9); u32(8); u64(4)
  for (const token of ['a','b','ab','<end>']) str(token)
  str('tokenizer.ggml.token_type'); u32(9); u32(5); u64(4)
  for (const type of [1,1,1,3]) u32(type)
  str('tokenizer.ggml.pre'); u32(8); str('llama3')
  str('tokenizer.ggml.eos_token_id'); u32(4); u32(3)
  while(bytes.length % 32) bytes.push(0)
  const tok = pg.Tokenizer.fromGGUF(Uint8Array.from(bytes))
  try {
    assert(tok.vocabSize === 4 && tok.eosId === 3 && tok.bosId === -1, 'GGUF metadata mismatch')
    assertClose(tok.encode('ab<end>'),[2,3],0)
    const text='ab<end>'.repeat(5000), ids=tok.encode(text)
    assert(ids.length === 10000, 'long encode was silently truncated')
    assert(tok.decode(ids) === text, 'long decode was silently truncated')
    assert(tok.encode('').length === 0 && tok.decode([]) === '', 'empty tokenization')
  } finally {tok.free()}
}

async function checkModelEmptyInputAdmission(pg) {
  const n = pg.uop.variable('empty_model_batch', 0, 4), bound = n.bind(3)
  const offset = pg.Tensor.empty([1]), x = pg.Tensor.empty([bound, 2])
  const model = await pg.Model.fromCallableAsync(({offset, x}) => x.add(offset), {inputs:{offset, x}})
  try {
    await model.forwardAsync({offset:new Float32Array([7]), x:new Float32Array(6).fill(1)})
    const before = await model.readBufferAsync('offset')
    let rejected = false
    try { await model.forwardAsync({offset:new Float32Array([99]), x:new Float32Array(0)}) }
    catch { rejected = true }
    assert(rejected, 'unsupported empty Model binding must reject')
    assertClose(await model.readBufferAsync('offset'), before, 0)
  } finally {
    await model.dispose(); await offset.dispose(); await x.dispose(); bound.dispose(); n.dispose()
  }
}

async function checkModelTensorIO(pg) {
  const gpu = String(pg.device).toLowerCase() === 'webgpu'
  const author = ({x, y}) => ({prediction: x.mul(2).add(y)})
  const options = {inputs:{x:pg.Tensor.empty([2,2]), y:pg.Tensor.empty([2,2])}}
  const model = gpu ? await pg.Model.fromCallableAsync(author, options) : new pg.Model(author, options)
  const owned = []
  try {
    const input = new pg.Tensor([[1,2],[3,4]], {dtype:'float32'}).transpose()
    owned.push(input)
    const pending = model.forwardAsync({x:input, y:new Float32Array([1,1,1,1])})
    // The queued call owns access before explicit input release, also on Asyncify.
    const disposing = input.dispose()
    const first = (await pending).prediction
    await disposing
    owned.push(first)
    assert(first instanceof pg.Tensor, 'Tensor input must select Tensor outputs')
    assert(JSON.stringify(first.shape) === '[2,2]', 'result lost concrete shape')
    const zero = pg.Tensor.zeros([2,2])
    owned.push(zero)
    const second = (await model.forwardAsync({x:first, y:zero})).prediction
    owned.push(second)
    assertClose(await first.toArrayAsync(), [3,7,5,9], 0)
    assertClose(await second.toArrayAsync(), [6,14,10,18], 0)
    const before = await model.readBufferAsync('x')
    let shapeRejected = false
    try {
      await model.forwardAsync({x:new Float32Array([9,9,9,9]),
        y:{data:new Float32Array([0,0,0,0]), shape:[1,4]}})
    } catch { shapeRejected = true }
    assert(shapeRejected, 'equal bytes must not admit a wrong explicit host shape')
    assertClose(await model.readBufferAsync('x'), before, 0)
    const shaped = await model.forwardAsync({x:{data:new Float32Array([1,2,3,4]), shape:[2,2]},
      y:new Float32Array([0,0,0,0])})
    assertClose(shaped.prediction, [2,4,6,8], 0)
    const afterShape = await model.readBufferAsync('x')
    for (const bad of [pg.Tensor.zeros([4]), pg.Tensor.zeros([2,2], {dtype:'int32'})]) {
      owned.push(bad)
      let rejected = false
      try { await model.forwardAsync({x:zero, y:bad}) } catch { rejected = true }
      assert(rejected, 'Tensor signature mismatch must reject')
      assertClose(await model.readBufferAsync('x'), afterShape, 0)
    }
    const effect = pg.Tensor.ones([2,2]).contiguous()
    owned.push(effect)
    await effect.realizeAsync()
    effect.assign(effect.add(1))
    for (let i = 0; i < 2; i++) {
      const result = (await model.forwardAsync({x:effect, y:zero})).prediction
      owned.push(result)
      assertClose(await result.toArrayAsync(), [4,4,4,4], 0)
    }
    assertClose(await effect.toArrayAsync(), [2,2,2,2], 0)
    const finishing = model.forwardAsync({x:second, y:zero})
    const closing = model.dispose()
    const last = (await finishing).prediction
    owned.push(last)
    await closing
    pg.clearScheduleCache()
    pg.collect()
    assertClose(await first.toArrayAsync(), [3,7,5,9], 0)
    assertClose(await last.toArrayAsync(), [12,28,20,36], 0)
    const chained = second.add(1)
    owned.push(chained)
    assertClose(await chained.toArrayAsync(), [7,15,11,19], 0)
  } finally {
    await model.dispose()
    for (const tensor of owned) await tensor.dispose()
  }
}

async function checkModelBoundedMinibatches(pg) {
  const n = pg.uop.variable('fit_batch', 1, 8), bound = n.bind(4)
  const owned = [], models = []
  const build = async () => {
    const w = new pg.Tensor([0], {dtype:'float32'}), x = pg.Tensor.empty([bound,2]), y = pg.Tensor.empty([bound,2])
    owned.push(w,x,y)
    const model = await pg.Model.fromCallableAsync(({x}) => x.mul(w), {inputs:{x},targets:{y},params:{w},
      loss:(out,{y}) => out.sub(y).square().mean()})
    models.push(model)
    return model
  }
  try {
    const x = Float32Array.from({length:14},(_,i)=>i/10), y = x.map(v=>v*2)
    const tx = new pg.Tensor([Array.from(x.filter((_,i)=>i%2===0)),Array.from(x.filter((_,i)=>i%2===1))],
      {dtype:'float32'}).transpose(), ty = new pg.Tensor(y).reshape(7,2)
    owned.push(tx,ty)
    const control = await build()
    control.setOptimizer('sgd',0.01)
    const expected = []
    for (let epoch=0; epoch<2; epoch++) for (let i=0; i<7; i+=3)
      expected.push(await control.trainStepAsync({x:x.subarray(i*2,(i+3)*2),y:y.subarray(i*2,(i+3)*2)}))
    for (const data of [{x,y}, {x:tx,y:ty}, {x:tx,y}]) {
      const model = await build()
      const losses = await model.fitAsync(data,{batchSize:3,remainder:'keep',epochs:2,optimizer:'sgd',lr:0.01})
      assertClose(losses,expected,1e-5)
      assertClose(await model.readBufferAsync('w'),await control.readBufferAsync('w'),1e-5)
      const before = await model.readBufferAsync('w')
      for (const opts of [{batchSize:9,remainder:'keep'}, {batchSize:3,remainder:'error'}]) {
        let rejected=false
        try { await model.fitAsync(data,{...opts,optimizer:'adam'}) } catch { rejected=true }
        assert(rejected, 'invalid batch must reject before any update')
        assertClose(await model.readBufferAsync('w'),before,0)
      }
    }
    assertClose(await tx.toArrayAsync(), x, 0)
    const model = await build(), small = new pg.Tensor(x.subarray(0,4)).reshape(2,2)
    owned.push(small)
    assert((await model.fitAsync({x:small,y:y.subarray(0,4)}, {batchSize:3,remainder:'keep',optimizer:'sgd'})).length===1)
    assertClose(await small.toArrayAsync(),x.subarray(0,4),0)
    const ffi = pg._core.ffi, shrink = ffi.poly_tensor_shrink, release = ffi.poly_tensor_release
    const train = pg._core.model.trainStep
    for (const failure of ['shrink', 'train']) {
      let allocated = [], calls = 0, rejected = false
      const injectedShrink = (...args) => {
        if (++calls === 2 && failure === 'shrink') return null
        const handle = shrink(...args)
        allocated.push(handle)
        return handle
      }
      const injectedRelease = handle => {
        const index = allocated.indexOf(handle)
        if (index >= 0) allocated.splice(index,1)
        return release(handle)
      }
      // N-API exports are read-only. Shadow them on a test-local facade rather
      // than modifying the addon object or weakening native failure coverage.
      pg._core.ffi = Object.create(ffi, {
        poly_tensor_shrink:{value:injectedShrink}, poly_tensor_release:{value:injectedRelease}
      })
      pg._core.model.trainStep = (...args) => {
        if (failure === 'train') throw new Error('train sentinel')
        return train(...args)
      }
      try {
        await model.fitAsync({x:tx,y:ty},{batchSize:3,remainder:'keep'})
      } catch(e) { rejected = /shrink failed|train sentinel/.test(e.message) }
      finally {
        pg._core.ffi = ffi
        pg._core.model.trainStep = train
      }
      assert(rejected, 'injected batch failure must propagate')
      assert(allocated.length === 0, 'batch failure leaked Tensor slice owners')
    }
    let failed=false
    try { await model.fitAsync({x:tx,y:ty},{batchSize:3,remainder:'keep',onStep:()=>{throw new Error('batch callback')}}) }
    catch(e) { failed=/batch callback/.test(e.message) }
    assert(failed, 'callback exception must clean up Tensor slices')
    const pending = model.fitAsync({x:tx,y:ty},{batchSize:3,remainder:'keep'})
    const releases = [tx.dispose(),ty.dispose(),model.dispose()]
    assert((await pending).length===3, 'queued dataset must survive caller disposal')
    await Promise.all(releases)
  } finally {
    for (const model of models) await model.dispose()
    for (const tensor of owned) await tensor.dispose()
    bound.dispose(); n.dispose()
  }
}

async function checkModelMinibatches(pg) {
  const gpu = String(pg.device).toLowerCase() === 'webgpu'
  const build = async () => {
    const w = new pg.Tensor([0], {dtype:'float32'})
    const opts = {inputs:{x:pg.Tensor.empty([2,1])}, targets:{y:pg.Tensor.empty([2,1])}, params:{w},
      loss:(out, {y}) => out.sub(y).square().mean()}
    return gpu ? pg.Model.fromCallableAsync(({x}) => x.mul(w), opts) : new pg.Model(({x}) => x.mul(w), opts)
  }
  const model = await build(), control = await build()
  try {
    const x = new Float32Array([1,2,3,4,5,6]), y = new Float32Array([2,4,6,8,10,12])
    const seen = []
    const losses = await model.fitAsync({x:{data:x, shape:[6,1]}, y}, {
      batchSize:2, epochs:2, optimizer:'sgd', lr:0.01, onStep:step => seen.push(step)})
    control.setOptimizer('sgd', 0.01)
    const expected = []
    for (let epoch = 0; epoch < 2; epoch++) for (let i = 0; i < 6; i += 2)
      expected.push(await control.trainStepAsync({x:x.subarray(i,i+2), y:y.subarray(i,i+2)}))
    assertClose(losses, expected, 1e-5)
    assertClose(await model.readBufferAsync('w'), await control.readBufferAsync('w'), 1e-5)
    assert(JSON.stringify(seen) === '[0,1,2,3,4,5]', 'callback step numbers must span epochs')
    const before = await model.readBufferAsync('w')
    let rejected = false
    try { await model.fitAsync({x:x.subarray(0,5), y:y.subarray(0,5)}, {batchSize:2, optimizer:'adam'}) }
    catch (e) { rejected = /remainder/.test(e.message) }
    assert(rejected, 'incomplete batch must reject before any update')
    assertClose(await model.readBufferAsync('w'), before, 0)
    rejected = false
    try { await model.fitAsync({x:x.subarray(0,5),y:y.subarray(0,5)}, {batchSize:2,remainder:'keep',optimizer:'adam'}) }
    catch(e) { rejected = /remainder/.test(e.message) }
    assert(rejected, 'keep cannot admit an extent outside the fixed signature')
    assertClose(await model.readBufferAsync('w'),before,0)
    for (const bad of [{x, y:y.subarray(0,4)}, {x, y:new Int32Array(y)},
      {x:{data:x, shape:[3,2]}, y}]) {
      let invalid = false
      try { await model.fitAsync(bad, {batchSize:2, optimizer:'adam'}) } catch { invalid = true }
      assert(invalid, 'invalid dataset must fail preflight')
      assertClose(await model.readBufferAsync('w'), before, 0)
    }
    const dropped = await model.fitAsync({x:x.subarray(0,5), y:y.subarray(0,5)}, {batchSize:2, remainder:'drop'})
    assert(dropped.length === 2, 'drop must process only explicitly selected complete batches')
    let callbackFailed = false
    try { await model.fitAsync({x,y}, {batchSize:2, onStep:() => { throw new Error('callback sentinel') }}) }
    catch (e) { callbackFailed = /callback sentinel/.test(e.message) }
    assert(callbackFailed, 'callback failure must propagate without poisoning the queue')
    const finishing = model.fitAsync({x,y}, {batchSize:2})
    const closing = model.dispose()
    assert((await finishing).length === 3, 'admitted fit must survive immediate Model disposal')
    await closing
  } finally {
    await model.dispose()
    await control.dispose()
  }
}

async function checkQuantizedModelWeights(pg) {
  for (const c of quantizedFixture.cases) {
    const bytes = []
    const u32 = n => { for (let i = 0; i < 4; i++) bytes.push((n >>> (8 * i)) & 255) }
    const u64 = n => { u32(n); u32(0) }
    const str = s => { const b = new TextEncoder().encode(s); u64(b.length); bytes.push(...b) }
    bytes.push(71, 71, 85, 70); u32(3); u64(1); u64(5)
    str('general.architecture'); u32(8); str('gpt2')
    for (const [key, value] of [['embedding_length', 32], ['attention.head_count', 2], ['block_count', 1], ['context_length', 2]]) {
      str(`gpt2.${key}`); u32(4); u32(value)
    }
    str('token_embd.weight'); u32(2); u64(32); u64(8); u32(c.type); u64(0)
    while (bytes.length % 32) bytes.push(0)
    const expected = []
    for (let i = 0; i < 256 / c.values.length; i++) { bytes.push(...c.bytes); expected.push(...c.values) }
    const model = pg.Model.fromGGUF(new Uint8Array(bytes), { maxBatch: 1, maxSeqLen: 2 })
    try { assertClose(await model.readBufferAsync('wte.weight'), expected, 0) } finally { await model.dispose() }
  }
}

async function checkModelCodecRejection(pg) {
  const invalidHeads = new TextEncoder().encode(JSON.stringify({model_type:'gpt2',
    n_embd:4, n_head:0, n_layer:1, vocab_size:8, n_positions:2}))
  let invalidRejected = false
  try { pg.Model.fromHF(invalidHeads, []).dispose() } catch (e) { invalidRejected = /fromHF failed/.test(e.message) }
  assert(invalidRejected, 'zero attention heads must fail before division')
  const zeroHeads = [71, 71, 85, 70]
  const u32 = n => zeroHeads.push(n & 255, (n >>> 8) & 255, (n >>> 16) & 255, (n >>> 24) & 255)
  const u64 = n => { u32(n); u32(0) }
  const str = s => { const bytes = new TextEncoder().encode(s); u64(bytes.length); zeroHeads.push(...bytes) }
  u32(3); u64(0); u64(2)
  str('general.architecture'); u32(8); str('qwen3')
  str('qwen3.attention.head_count'); u32(4); u32(0)
  while (zeroHeads.length % 32) zeroHeads.push(0)
  invalidRejected = false
  try { pg.Model.fromGGUF(new Uint8Array(zeroHeads)).dispose() }
  catch (e) { invalidRejected = /attention.head_count must be positive/.test(e.message) }
  assert(invalidRejected, 'GGUF zero heads must return an import error, not trap')
  const config = new TextEncoder().encode(JSON.stringify({ model_type: 'gpt2',
    n_embd: 8, n_head: 2, n_layer: 1, vocab_size: 4, n_positions: 4 }))
  const checkpoint = name => {
    const header = new TextEncoder().encode(JSON.stringify({ [name]: {
      dtype: 'F32', shape: [4, 8], data_offsets: [0, 128]
    } }))
    const bytes = new Uint8Array(8 + header.length + 128)
    new DataView(bytes.buffer).setUint32(0, header.length, true)
    bytes.set(header, 8)
    return bytes
  }
  const weight = checkpoint('transformer.wte.weight')
  for (const name of ['transformer.wte.weight', 'unused.weight', 'transformer.h.0.attn.bias']) {
    const shard = checkpoint(name)
    // WebGPU disposal holds an async runtime lease; await it before the next import.
    await pg.Model.fromHF(config, [shard]).dispose()
    let duplicate
    try { await pg.Model.fromHF(config, [shard, shard]).dispose() } catch (e) { duplicate = e }
    assert(duplicate && /duplicate weight/.test(duplicate.message), `duplicate source name must reject: ${duplicate}`)
  }
  let collision
  try { await pg.Model.fromHF(config, [weight, checkpoint('wte.weight')]).dispose() }
  catch (e) { collision = e }
  assert(collision && /duplicate weight binding/.test(collision.message), 'normalized names must not overwrite')
  await pg.Model.fromHF(config, [weight]).dispose() // Rejection must leave the runtime usable.
  const badShard = new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0, 123])
  let error
  try { pg.Model.fromHF(config, [badShard]).dispose() } catch (e) { error = e }
  assert(error && /failed to decode weight file/.test(error.message),
    'Model import must reject a malformed shard before construction')
  const header = new TextEncoder().encode(JSON.stringify({ 'transformer.wte.weight': {
    dtype: 'F32', shape: [1], data_offsets: [0, 4]
  } }))
  const wrongShape = new Uint8Array(8 + header.length + 4)
  new DataView(wrongShape.buffer).setUint32(0, header.length, true)
  wrongShape.set(header, 8)
  error = null
  try { pg.Model.fromHF(config, [wrongShape]).dispose() } catch (e) { error = e }
  assert(error && /shape mismatch for 'wte.weight'/.test(error.message),
    'Model import must propagate the named checkpoint shape mismatch')
  const badGguf = new Uint8Array(64)
  badGguf.set([71, 71, 85, 70, 3])
  badGguf[16] = 1
  badGguf.fill(255, 24, 32)
  error = null
  try { pg.Model.fromGGUF(badGguf).dispose() } catch (e) { error = e }
  assert(error && /GGUF/.test(error.message), 'Model import must reject an invalid GGUF length')
  // Missing architecture is deliberate: distinguish successful byte decoding
  // followed by family rejection from rejecting malformed tensor metadata.
  for (const [type, numel, nbytes, valid] of [
    [24, 1, 1, true], [25, 1, 2, true], [26, 1, 4, true],
    [18, 1, 4, false], [8, 1, 34, false], [8, 32, 34, true]
  ]) {
    const bytes = new Uint8Array(64 + nbytes)
    bytes.set([71, 71, 85, 70, 3])
    const view = new DataView(bytes.buffer)
    view.setUint32(8, 1, true) // one tensor, no KV metadata
    view.setUint32(24, 1, true)
    bytes[32] = 120 // name x
    view.setUint32(33, 1, true)
    view.setUint32(37, numel, true)
    view.setUint32(45, type, true)
    error = null
    try { pg.Model.fromGGUF(bytes).dispose() } catch (e) { error = e }
    assert(error && (valid ? /unsupported GGUF architecture/ : /invalid or unallocatable GGUF/).test(error.message),
      `GGUF type ${type}, numel ${numel}: wrong decoder admission`)
  }
}

async function checkCompositionFactories(pg) {
  const { Model, models } = pg
  const webgpu = String(pg.device).toLowerCase() === 'webgpu'
  const spec = JSON.parse(JSON.stringify(compositionFixture))
  delete spec.type
  delete spec.format
  assert(!Model.fromDefinition, 'construction families must not be Model methods')
  if (webgpu) {
    let rejected = false
    try { models.Graph(spec) } catch (err) { rejected = err.name === 'PolyAsyncRequired' }
    assert(rejected, 'WebGPU construction requires explicit async admission')
  }
  const model = await models.GraphAsync(spec)
  let restored = null
  try {
    assert(model.paramCount === 1, 'shared calls must have one parameter')
    await model.writeBufferAsync('modules.shared.weight', new Float32Array([1, 2, 3, 4]))
    const io = { x: new Float32Array([1, 2]) }
    assertClose((await model.forward(io)).prediction, [28, 61])
    model.setOptimizer(pg.OPTIM_SGD, .1)
    const loss = await model.trainStepAsync(io)
    assert(Math.abs(loss - 89) < 1e-4, `wrong objective: ${loss}`)
    assertClose(await model.readBufferAsync('modules.shared.weight'), [.1, .1, 1.9, 1.7])
    const blob = await model.saveBundleAsync({ includeOptimizer: false })
    restored = Model.fromBundle(blob)
    assertClose((await restored.forward(io)).prediction, (await model.forward(io)).prediction)
  } finally {
    if (restored) await restored.dispose()
    await model.dispose()
  }
  const sequential = {
    input: { name: 'x', shape: [1, 2], dtype: 'float32' },
    layers: [{ name: 'stack', type: 'repeat', count: 2,
      body: { type: 'linear', out_features: 2, activation: 'relu' } }],
    output: 'prediction', seed: 42
  }
  const a = await models.SequentialAsync(sequential)
  const b = await models.SequentialAsync(JSON.stringify(sequential))
  try {
    assert(a.paramCount === 4, 'Repeat must create fresh weights and biases')
    for (const binding of a.bindings().filter(v => v.trainable)) {
      assertClose(await a.readBufferAsync(binding.name), await b.readBufferAsync(binding.name), 0)
    }
    await a.writeBufferAsync('layers.stack.0.bias', new Float32Array([7, 8]))
    assertClose(await a.readBufferAsync('layers.stack.1.bias'), [0, 0], 0)
    assertClose(await b.readBufferAsync('layers.stack.0.bias'), [0, 0], 0)
    const io = { x: new Float32Array([1, 2]) }
    const result = (await a.forward(io)).prediction
    assert(result.length === 2 && Array.from(result).every(Number.isFinite), 'Repeat output')
  } finally { await a.dispose(); await b.dispose() }

  for (const [bad, error] of [
    ['{"type":"graph","type":"sequential"}', /duplicate/],
    [{ ...spec, type: 'sequential' }, /type: expected 'graph'/],
    [{ ...spec, format: 'poly.modeldef@99' }, /format/],
    [{ ...spec, nodes: [{ name: 'bad', type: 'add', inputs: ['later', 'x'] }] }, /forward value/]
  ]) {
    let caught = null
    try { await models.GraphAsync(bad) } catch (err) { caught = err }
    assert(caught && error.test(caught.message), `missing validation: ${caught && caught.message}`)
  }
}

async function checkCompositionCatalogue(pg) {
  const unary = {
    relu: x => Math.max(0, x), sigmoid: x => 1 / (1 + Math.exp(-x)),
    tanh: Math.tanh, silu: x => x / (1 + Math.exp(-x)),
    gelu: x => .5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + .044715 * x ** 3))),
    identity: x => x, square: x => x * x, exp: Math.exp, log: Math.log
  }
  for (const type of [...Object.keys(unary), 'sum', 'mean', 'reshape']) {
    const data = type === 'log' ? [1, 2] : [-1, 2]
    const layer = { name: 'op', type }
    if (type === 'reshape') layer.shape = [2, 1]
    const model = await pg.models.SequentialAsync({
      input: { name: 'x', shape: [1, 2], dtype: 'float32' },
      layers: [layer], output: 'prediction'
    })
    try {
      const expected = type === 'sum' ? [1] : type === 'mean' ? [.5]
        : type === 'reshape' ? data : data.map(unary[type])
      assertClose((await model.forward({ x: new Float32Array(data) })).prediction, expected, 2e-5)
    } finally { await model.dispose() }
  }
  for (const [type, expected] of [
    ['add', [2, 4, 4, 6]], ['sub', [0, 0, 2, 2]],
    ['mul', [1, 4, 3, 8]], ['div', [1, 1, 3, 2]]
  ]) {
    const model = await pg.models.GraphAsync({
      inputs: { x: { shape: [2, 2], dtype: 'float32' }, y: { shape: [2], dtype: 'float32' } },
      nodes: [{ name: 'op', type, inputs: ['x', 'y'] }], outputs: { prediction: 'op' }
    })
    try {
      assertClose((await model.forward({ x: new Float32Array([1, 2, 3, 4]), y: new Float32Array([1, 2]) })).prediction, expected)
    } finally { await model.dispose() }
  }
  // A target is required by the objective, not by the prediction entrypoint.
  const model = await pg.models.GraphAsync({
    inputs: { x: { shape: [2, 1], dtype: 'float32' }, y: { shape: [2, 1], dtype: 'float32', role: 'target' } },
    nodes: [
      { name: 'pred', type: 'linear', out_features: 1, bias: false, inputs: ['x'] },
      { name: 'error', type: 'sub', inputs: ['pred', 'y'] },
      { name: 'sq', type: 'square', inputs: ['error'] },
      { name: 'avg', type: 'mean', inputs: ['sq'] }
    ],
    outputs: { prediction: 'pred', cost: 'avg' },
    entrypoints: [
      { name: 'forward', inputs: ['x'], outputs: ['prediction'] },
      { name: 'loss', inputs: ['x', 'y'], outputs: ['cost'], objective: 'cost' }
    ]
  })
  try {
    const x = new Float32Array([1, 2])
    await model.writeBufferAsync('nodes.pred.weight', new Float32Array([2]))
    assertClose((await model.forward({ x })).prediction, [2, 4], 0)
    model.setOptimizer(pg.OPTIM_SGD, .1)
    assertClose([await model.trainStepAsync({ x, y: x })], [2.5], 1e-6)
    assertClose(await model.readBufferAsync('nodes.pred.weight'), [1.5], 1e-6)
  } finally { await model.dispose() }
}

async function checkDescriptionGradient(pg) {
  const model = await pg.models.GraphAsync({
    inputs: { x: { shape: [2], dtype: 'float32' } },
    nodes: [
      { name: 'sq', type: 'square', inputs: ['x'] },
      { name: 'loss', type: 'sum', inputs: ['sq'] },
      { name: 'g', type: 'gradient', inputs: ['loss', 'x'] },
      { name: 'gs', type: 'sum', inputs: ['g'] },
      { name: 'gg', type: 'gradient', inputs: ['gs', 'x'] }
    ], outputs: { loss: 'loss', gradient: 'g', second: 'gg' }
  })
  try {
    for (const x of [[.25, -.5], [1, 2]]) {
      const out = await model.callAsync('forward', { x: Float32Array.from(x) })
      assertClose(out.gradient, x.map(v => 2*v), 0)
      assertClose(out.second, [2, 2], 0)
    }
    const input = new pg.Tensor(new Float32Array([3,4]))
    try {
      await model.callAsync('forward', {x:input}, {readback:false})
      assertClose(await model.readBufferAsync('gradient'), [6,8], 0)
    } finally { input.dispose() }
  } finally { await model.dispose() }
}

async function checkDescriptionState(pg) {
  const model = await pg.models.GraphAsync(require('../../test/fixtures/model_state.json'))
  try {
    await model.writeBufferAsync('a', new Float32Array([1,2]))
    await model.writeBufferAsync('b', new Float32Array([3,4]))
    // Skipping readback must not call either frontend output collector.
    const sync = model._collectOutputsRaw, async = model._collectOutputsRawAsync
    model._collectOutputsRaw = model._collectOutputsRawAsync = () => { throw Error('unexpected readback') }
    try {
      const out = await model.callAsync('step', {}, {controls:{iteration:1},readback:false})
      assert(Object.keys(out).length === 0)
    } finally { model._collectOutputsRaw = sync; model._collectOutputsRawAsync = async }
    assertClose(await model.readBufferAsync('a'), [3,4], 0)
    assertClose(await model.readBufferAsync('b'), [2,3], 0)
    let rejected = false
    try { await model.callAsync('step', {}, {controls:{iteration:11},readback:false}) }
    catch (_) { rejected = true }
    assert(rejected, 'out-of-bounds control accepted')
    assertClose(await model.readBufferAsync('a'), [3,4], 0)
    const restored = pg.Model.load(await model.saveAsync())
    try {
      await restored.callAsync('step', {}, {controls:{iteration:2},readback:false})
      assertClose(await restored.readBufferAsync('a'), [2,3], 0)
      assertClose(await restored.readBufferAsync('b'), [5,6], 0)
    } finally { await restored.dispose() }
  } finally { await model.dispose() }
}

async function checkTiedAdamCheckpoint(pg) {
  const w = new pg.Tensor([1], { dtype: 'float32' })
  const model = await pg.Model.fromTensors({
    params: { w, tied: w }, losses: { cost: w.add(w).square().sum() }
  })
  let restored = null
  try {
    await model.placeAsync(pg.device)
    model.setTrainable('tied', false)
    assert(!model.bufTrainable(model.findBuf('w')), 'placed alias did not freeze')
    model.setTrainable('w', true)
    assert(model.bufTrainable(model.findBuf('tied')), 'placed alias did not unfreeze')
    model.setOptimizer(pg.OPTIM_ADAM, .1)
    assertClose([await model.trainStepAsync({})], [4], 0)
    assertClose(await model.readBufferAsync('w'), [.9], 1e-6)
    const names = ['optim.adam.b1_t', 'optim.adam.b2_t', 'optim.adam.m.w', 'optim.adam.v.w']
    const weights = await model.exportWeightsAsync()
    const optimizerNames = bytes => [...safetensorNames(bytes)].filter(n => n.startsWith('optim.')).sort().join(',')
    assert(optimizerNames(weights) === [...names].sort().join(','), 'duplicated tied optimizer state')
    restored = pg.Model.fromIR(model.exportIR(), weights)
    await restored.placeAsync(pg.device)
    restored.setTrainable('w', false)
    assert(!restored.bufTrainable(restored.findBuf('tied')), 'restored alias did not freeze')
    restored.setTrainable('tied', true)
    restored.setOptimizer(pg.OPTIM_ADAM, .1)
    const expectedLoss = await model.trainStepAsync({})
    assertClose([await restored.trainStepAsync({})], [expectedLoss], 0)
    for (const name of [...names, 'w', 'tied']) {
      assertClose(await restored.readBufferAsync(name), await model.readBufferAsync(name), 0)
    }
    assert(optimizerNames(await restored.exportWeightsAsync()) === [...names].sort().join(','), 'restored alias state duplicated')
  } finally {
    if (restored) await restored.dispose()
    await model.dispose()
  }
}

function assert(cond, msg) {
  if (!cond) throw new Error(msg || 'assertion failed')
}

function assertClose(actual, expected, tol = 1e-4) {
  if (actual.length !== expected.length) {
    throw new Error(`length mismatch: ${actual.length} vs ${expected.length}`)
  }
  for (let i = 0; i < actual.length; i++) {
    if (Number.isNaN(expected[i])) {
      if (!Number.isNaN(actual[i])) {
        throw new Error(`mismatch at [${i}]: ${actual[i]} vs ${expected[i]}`)
      }
      continue
    }
    const diff = Math.abs(actual[i] - expected[i])
    if (!Number.isFinite(diff) || diff > tol) {
      throw new Error(`mismatch at [${i}]: ${actual[i]} vs ${expected[i]}`)
    }
  }
}

function safetensorNames(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  const headerLen = Number(view.getBigUint64(0, true))
  const headerBytes = bytes.subarray(8, 8 + headerLen)
  const header = JSON.parse(new TextDecoder().decode(headerBytes))
  return new Set(Object.keys(header).filter(k => k !== '__metadata__'))
}

function testFilterFor(pg) {
  if (pg && pg.testFilter) return String(pg.testFilter)
  if (typeof globalThis !== 'undefined' && globalThis.__POLY_TEST_FILTER) {
    return String(globalThis.__POLY_TEST_FILTER)
  }
  if (typeof process !== 'undefined' && process.env && process.env.POLY_TEST_FILTER) {
    return String(process.env.POLY_TEST_FILTER)
  }
  return ''
}

async function checkTypedIntegerInput(pg, Model) {
  const x = pg.Tensor.empty([3], { dtype: 'int32' })
  const outTensor = x.cast('float32')
  const inst = await Model.fromTensors({
    inputs: { typed_x: x },
    outputs: { typed_out: outTensor }
  })
  try {
    const output = await inst.forward({ typed_x: new Int32Array([0, 1, 2]) })
    assertClose(output.typed_out, [0, 1, 2], 0)

    let rejection = ''
    try {
      await inst.forward({ typed_x: new Float32Array([0, 1, 2]) })
    } catch (e) {
      rejection = String(e && e.message)
    }
    assert(/input 'typed_x': expected int32, received float32/.test(rejection),
      `float32 bytes must not bind to an int32 Model input; received: ${rejection}`)
  } finally {
    inst.dispose()
  }
}

async function checkCallSignatureAndSelectedOutputs(pg, Model) {
  const x = pg.Tensor.empty([2])
  const y = pg.Tensor.empty([2])
  const inst = await Model.fromTensors({
    inputs: { x, y },
    outputs: { plus: x.add(y), minus: x.sub(y) },
    entrypoints: [
      { name: 'plus_ep', inputs: ['x', 'y'], outputs: ['plus'] },
      { name: 'minus_ep', inputs: ['x', 'y'], outputs: ['minus'] }
    ]
  })
  const webgpu = String(pg.device).toLowerCase() === 'webgpu'
  const call = (entrypoint, io) => webgpu
    ? inst.callAsync(entrypoint, io) : inst.call(entrypoint, io)
  try {
    const io = {
      x: new Float32Array([5, 7]),
      y: new Float32Array([2, 3])
    }
    const plus = await call('plus_ep', io)
    const minus = await call('minus_ep', io)
    assert(Object.keys(plus).join(',') === 'plus', 'plus_ep returned undeclared outputs')
    assert(Object.keys(minus).join(',') === 'minus', 'minus_ep returned undeclared outputs')
    assertClose(plus.plus, [7, 10], 0)
    assertClose(minus.minus, [3, 4], 0)

    let rejection = ''
    try {
      await call('plus_ep', { x: new Float32Array([9, 9]) })
    } catch (err) {
      rejection = String(err && err.message)
    }
    assert(/entrypoint 'plus_ep' requires input 'y' exactly once/.test(rejection),
      `missing required input must not reuse stale Model bytes; received: ${rejection}`)
  } finally {
    if (webgpu) await inst.dispose()
    else inst.dispose()
  }

  let invalidParamRejected = false
  try {
    const unexpected = await Model.fromTensors({
      inputs: { x }, outputs: { output: x.add(1) }, params: { bad: {} }
    })
    unexpected.dispose()
  } catch (err) {
    invalidParamRejected = /not a Tensor/.test(String(err && err.message))
  }
  assert(invalidParamRejected, 'invalid parameter must not be silently filtered')
}

async function checkModuleDeviceMap(pg, Model, batched = false) {
  const Tensor = pg.Tensor
  const x = Tensor.empty(batched ? [2, 2] : [2])
  const input = new Float32Array(batched ? [1, 2, 3, 4] : [1, 2])
  const webgpu = String(pg.device).toLowerCase() === 'webgpu'
  const w0 = webgpu ? null : new Tensor([3, 4], { dtype: 'float32' })
  const w1 = webgpu ? null : new Tensor([2, 3], { dtype: 'float32' })
  const hidden = webgpu ? x.add(3) : x.add(w0)
  const output = webgpu ? hidden.mul(2) : hidden.mul(w1)
  const inst = await Model.fromTensors({
    inputs: { x },
    outputs: { output },
    params: webgpu ? null : { 'layers.0.weight': w0, 'layers.1.weight': w1 },
    modules: [
      { name: 'layers.0', inputs: [x], output: hidden },
      { name: 'layers.1', inputs: [hidden], output }
    ]
  })
  const first = String(pg.device).toUpperCase()
  // Pinned Device opens only runtime/ops_* implementations (device.py:15-35).
  // Native Polygrad likewise has no WASM backend; Emscripten does. Keep the
  // test cross-device on every core without treating a graph spelling as an
  // executable runtime merely because it is a valid DEVICE name.
  const second = first === 'INTERP' ? (pg.core === 'native' ? 'CPU' : 'WASM') : 'INTERP'
  const place = async map => {
    if (webgpu) await inst.setDeviceMapAsync(map)
    else inst.setDeviceMap(map)
  }
  const forward = input => webgpu
    ? inst.forwardAsync(input) : inst.forward(input)
  const expected = webgpu ? (batched ? [8, 10, 12, 14] : [8, 10])
    : (batched ? [8, 18, 12, 24] : [8, 18])
  const present = (value, label) => {
    assert(value != null, `${label} returned null`)
    return value
  }
  const assertOptionalBytesEqual = (actual, expectedBytes, label) => {
    assert((actual == null) === (expectedBytes == null), `${label} presence changed`)
    if (actual != null) assertClose(actual, expectedBytes, 0)
  }

  try {
    const irBefore = present(inst.exportIR(), 'initial IR export')
    const weightsBefore = await inst.exportWeights()
    // Uniform placement accepts the same exact identities as module maps.
    // CPU ordinals are native-only; never silently alias them to WASM.
    const uniform = device => webgpu ? inst.placeAsync(device) : inst.place(device)
    const targets = pg.core === 'native' ? ['CPU:1', 'cpu:2', first] : [first]
    for (const device of targets) {
      await uniform(device)
      assertClose((await forward({ x: input })).output, expected)
      assertClose(inst.exportIR(), irBefore, 0)
      assertOptionalBytesEqual(await inst.exportWeights(), weightsBefore, 'uniform weights')
    }
    const invalid = ['CUDA:1', 'CPU:bad', 'AUTO']
    if (pg.core !== 'native') invalid.push('CPU:1')
    for (const device of invalid) {
      let rejected = false
      try { await uniform(device) } catch (_) { rejected = true }
      assert(rejected, `uniform placement must reject ${device}`)
      assertClose((await forward({ x: input })).output, expected)
    }
    await place({ 'layers.0': first, 'layers.1': second })
    let result = await forward({ x: input })
    assertClose(present(result.output, 'first placed output'), expected)
    assertClose(present(inst.exportIR(), 'first placed IR export'), irBefore, 0)
    assertOptionalBytesEqual(await inst.exportWeights(), weightsBefore, 'first placed weight export')

    let rejected = false
    try {
      await place({ 'layers.0': first })
    } catch (err) {
      rejected = /incomplete|device map/.test(String(err.message || err))
    }
    assert(rejected, 'expected incomplete device map to fail')

    await place({ 'layers.0': second, 'layers.1': first })
    result = await forward({ x: input })
    assertClose(present(result.output, 'replacement placed output'), expected)
    const irAfter = present(inst.exportIR(), 'replacement IR export')
    const weightsAfter = await inst.exportWeights()
    assertClose(irAfter, irBefore, 0)
    assertOptionalBytesEqual(weightsAfter, weightsBefore, 'replacement weight export')

    const restored = Model.fromIR(irAfter, weightsAfter)
    try {
      const restoredResult = webgpu
        ? await restored.forwardAsync({ x: input })
        : await restored.forward({ x: input })
      assertClose(present(restoredResult.output, 'restored output'), expected)
    } finally {
      if (webgpu) await restored.dispose()
      else restored.dispose()
    }
  } finally {
    if (webgpu) await inst.dispose()
    else inst.dispose()
  }
}

async function checkModelStorageObjectives(pg, Model) {
  const w = new pg.Tensor([2], { dtype: 'float32' })
  const a = w.mul(w).sum(), b = w.mul(w).mul(w).sum()
  const model = await Model.fromTensors({ params: { w, tied: w }, losses: { a, b }, entrypoints: [
    { name: 'a_ep', outputs: ['a'], objective: 'a' },
    { name: 'b_ep', outputs: ['a', 'b'], objective: 'b' }
  ] })
  try {
    const copied = await model.readBufferAsync('w')
    copied[0] = 100
    assertClose(await model.readBufferAsync('w'), [2], 0)
    let rejected = false
    try { await model.writeBufferAsync('w', new Int32Array([3])) } catch (_) { rejected = true }
    assert(rejected, 'write accepted a different dtype')
    const written = new Float32Array([2])
    const pendingWrite = model.writeBufferAsync('w', written)
    written[0] = 9
    await pendingWrite
    assertClose(await model.readBufferAsync('w'), [2], 0)
    model.setTrainable('tied', false)
    model.setTrainable('w', true)
    model.setOptimizer(pg.OPTIM_SGD, .1, 0, 0, 0, 0)
    rejected = false
    try { await model.trainStepAsync({}) } catch (_) { rejected = true }
    assert(rejected, 'ambiguous objective silently selected')
    assertClose([await model.trainStepAsync({}, 'b_ep')], [8], 1e-6)
    assertClose(await model.readBufferAsync('w'), [.8], 1e-6)
    assertClose(await model.readBufferAsync('b'), [8], 0)
    await model.placeAsync(pg.device)
    assertClose(await model.readBufferAsync('w'), [.8], 1e-6)
    await model.dispose()
    assertClose(copied, [100], 0)
    rejected = false
    try { await model.readBufferAsync('w') } catch (_) { rejected = true }
    assert(rejected, 'disposed Model accepted a state read')
  } finally { await model.dispose() }
}

async function checkModelTrace(pg, Model) {
  const x = pg.Tensor.empty([1]), y = pg.Tensor.empty([1])
  // Computed state exercises async snapshot execution, not only host copying.
  const w = new pg.Tensor([1], { dtype: 'float32' }).add(1)
  const offset = new pg.Tensor([1], { dtype: 'float32' }).is_param_(false)
  let calls = 0
  const trace = String(pg.device).toLowerCase() === 'webgpu' ? Model.fromCallableAsync.bind(Model) : Model.fromCallable.bind(Model)
  const model = await trace(({ x }) => { calls++; return x.mul(w).add(offset) }, {
    inputs: { x }, targets: { y }, params: { w, offset },
    loss: (out, { y }) => ({ mse: out.sub(y).square().mean() })
  })
  try {
    assert(calls === 2, 'loss capture must construct evaluation and training forwards')
    model.setOptimizer(pg.OPTIM_SGD, .1, 0, 0, 0, 0)
    assertClose([await model.trainStepAsync({ x: new pg.Tensor([1], {dtype:'float32'}),
      y: new pg.Tensor([0], {dtype:'float32'}) })], [9], 1e-6)
    assertClose(await model.readBufferAsync('w'), [1.4], 1e-6)
    assertClose(await model.readBufferAsync('offset'), [1], 0)
    assert(model.bindings().some(row => row.name === 'offset' && row.role === 4), 'AUX role lost')
    const saved = await model.exportWeightsAsync({ includeOptimizer: false })
    assert(safetensorNames(saved).has('offset'), 'model-only export lost persistent AUX')
    assert(calls === 2, 'training re-invoked authoring code after train/eval capture')
    const objective = model.entrypoints().find(row => row.name === 'loss')
    assert(objective.objective === 'mse' && objective.inputs.join(',') === 'x,y', 'objective signature lost')
  } finally { await model.dispose() }
}

async function checkModelConstructor(pg, Model) {
  class Net {
    constructor() {
      this.weight = new pg.Tensor([2], { dtype: 'float32' })
      this.alias = this.weight
      this.offset = new pg.Tensor([1], { dtype: 'float32' }).is_param_(false)
    }
    forward({ x }) { return x.mul(this.weight).add(this.offset) }
  }
  const net = new Net()
  const options = { inputs: { x: pg.Tensor.empty([1]) }, targets: { y: pg.Tensor.empty([1]) },
    loss: (out, { y }) => out.sub(y).square().mean() }
  const gpu = String(pg.device).toLowerCase() === 'webgpu'
  const model = gpu ? await Model.fromCallableAsync(net, options) : new Model(net, options)
  try {
    const roles = Object.fromEntries(model.bindings().map(b => [b.name, b.role]))
    assert(roles.weight === 0 && roles.alias === 0 && roles.offset === 4, 'constructor state roles')
    model.setOptimizer('sgd', .1)
    assertClose([await model.trainStepAsync({ x: [1], y: [0] })], [9], 1e-6)
    assertClose(await model.readBufferAsync('weight'), [1.4], 1e-6)
    assertClose(await net.weight.toArrayAsync(), [2], 0)
    const bytes = gpu ? await model.saveAsync({ includeOptimizer: false }) : model.save({ includeOptimizer: false })
    assert(bytes instanceof Uint8Array && bytes.length > 0, 'bundle save')
  } finally { await model.dispose() }
}

async function checkModelDispatch(pg, Model) {
  const gpu = String(pg.device).toLowerCase() === 'webgpu'
  for (const config of [compositionFixture, {
    format: 'poly.modeldef@1', type: 'sequential', seed: 42,
    input: { name: 'x', shape: [1, 2], dtype: 'float32' },
    layers: [{ name: 'dense', type: 'linear', out_features: 2 }], output: 'prediction'
  }]) {
    const factory = config.type === 'graph' ? 'Graph' : 'Sequential'
    let automatic, explicit
    try {
      if (gpu) {
        let rejected = false
        try { new Model(config) } catch (e) { rejected = /Async/.test(e.message) }
        assert(rejected, 'WebGPU constructor must identify the explicit async factory')
        automatic = await pg.models[factory + 'Async'](config)
      } else automatic = new Model(config)
      explicit = await pg.models[factory + (gpu ? 'Async' : '')](config)
      assert(JSON.stringify(automatic.bindings()) === JSON.stringify(explicit.bindings()), 'configuration bindings differ')
      const a = await automatic.forwardAsync({ x: new Float32Array([1, 2]) })
      const b = await explicit.forwardAsync({ x: new Float32Array([1, 2]) })
      for (const key of Object.keys(a)) assertClose(a[key], b[key], 0)
    } finally {
      if (automatic) await automatic.dispose()
      if (explicit) await explicit.dispose()
    }
  }
  const x = pg.Tensor.empty([1])
  const source = { forward: ({ x }) => x.add(99), selected: ({ x }) => x.mul(2) }
  const model = gpu
    ? await Model.fromCallableAsync(source.selected, { inputs: { x }, params: {} })
    : Model.fromCallable(source.selected, { inputs: { x }, params: {} })
  try { assertClose((await model.forwardAsync({ x: [3] })).output, [6], 0) }
  finally { await model.dispose() }
  for (const invalid of [{ layers: [1, 2] }, { nodes: [] },
    { format: 'poly.modeldef@2', type: 'graph' }, { format: 'poly.modeldef@1', type: 'unknown' }]) {
    let rejected = false
    try { new Model(invalid) } catch (_) { rejected = true }
    assert(rejected, 'constructor guessed an untagged/unknown configuration')
  }
  let rejected = false
  try { new Model(compositionFixture, { outputs: x }) } catch (e) { rejected = /combined/.test(e.message) }
  assert(rejected, 'configuration accepted conflicting Tensor bindings')
  class LazyNet {
    forward({ x }) {
      this.weight = new pg.Tensor([2], { dtype: 'float32' })
      return x.mul(this.weight)
    }
  }
  const lazy = gpu ? await Model.fromCallableAsync(new LazyNet(), { inputs: { x } })
    : new Model(new LazyNet(), { inputs: { x } })
  try { assert(lazy.bindings().some(b => b.name === 'weight' && b.role === 0), 'lazy parameter not collected') }
  finally { await lazy.dispose() }
}

async function checkModelUsability(pg, Model) {
  const x = pg.Tensor.empty([1])
  const gpu = String(pg.device).toLowerCase() === 'webgpu'
  const create = (fn, options) => gpu ? Model.fromCallableAsync(fn, options) : Model.fromCallable(fn, options)
  const model = await create(({x}) => ({prediction:x.mul(2)}), {inputs:{x}})
  try {
    const text = model.summary()
    assert(text.includes('prediction') && text.includes('float32[1]') && text.includes('forward'), 'missing model metadata')
    if (gpu) {
      const bytes = await model.saveAsync()
      const running = model.forwardAsync({x:[3]})
      let rejected = false
      try { Model.load(bytes) } catch (e) { rejected = /async/.test(e.message) }
      await running
      assert(rejected, 'bundle load entered suspended Wasm')
    }
    let called = false
    for (const [fn, loss] of [
      [async ({x}) => { called = true; return x }, undefined],
      [({x}) => { called = true; return x }, async out => out.mean()]
    ]) {
      let rejected = false
      try { await create(fn, {inputs:{x}, loss}) } catch (e) { rejected = /synchronous/.test(e.message) }
      assert(rejected && !called, 'async author/loss was invoked')
    }
    const policy = x.logicalPolicy
    let rejected = false
    try { await create(() => { throw new Error('author failed') }, {inputs:{x}}) }
    catch (e) { rejected = /author failed/.test(e.message) }
    assert(rejected && pg.Tensor.empty([1]).logicalPolicy === policy, 'failed capture changed logical policy')
    // Browser entrypoints must reject paths before attempting filesystem access.
    if (typeof process === 'undefined' || !process.versions || !process.versions.node) {
      rejected = false
      try { model.save('model.pgb') } catch (e) { rejected = /Node/.test(e.message) }
      assert(rejected, 'browser save accepted a filesystem path')
    }
  } finally { await model.dispose() }
}

async function checkRuntimeImports(pg, Model, createRuntime) {
  const gpu = String(pg.device).toLowerCase() === 'webgpu'
  const x = pg.Tensor.empty([1]), w = new pg.Tensor([2], {dtype:'float32'})
  const live = new pg.Tensor([19], {dtype:'float32'})
  const options = {inputs:{x}, params:{w, alias:w}, entrypoints:[
    {name:'forward',inputs:['x'],outputs:['output']},
    {name:'double',inputs:['x'],outputs:['twice']}
  ]}
  const author = ({x}) => ({output:x.mul(w),twice:x.mul(w).mul(2)})
  const source = gpu ? await Model.fromCallableAsync(author, options)
    : Model.fromCallable(author, options)
  const models = [source]
  const read = m => gpu ? m.readBufferAsync('w') : m.readBuffer('w')
  const forward = m => gpu ? m.forwardAsync({x:[3]}) : m.forward({x:[3]})
  try {
    const bytes = gpu ? await source.saveAsync() : source.save()
    const a = Model.load(bytes), b = Model.fromBundle(bytes)
    models.push(a, b)
    if (gpu) await a.writeBufferAsync('alias', new Float32Array([7]))
    else a.writeBuffer('alias', new Float32Array([7]))
    assertClose(await read(a), [7], 0)
    assertClose((await forward(b)).output, [6], 0)
    assertClose((await forward(source)).output, [6], 0)
    const c = Model.load(await b.saveAsync())
    models.push(c)
    assertClose((await c.callAsync('double',{x:[3]})).twice,[12],0)
    await b.writeBufferAsync('alias',new Float32Array([11]))
    assertClose((await c.callAsync('double',{x:[3]})).twice,[12],0)
    assertClose((await c.forwardAsync({x:[3]})).output,[6],0)
    for (const n of [0, 31, Math.floor(bytes.length/2), bytes.length-1]) {
      let rejected = false
      try { Model.load(bytes.slice(0, n)) } catch { rejected = true }
      assert(rejected, 'truncated import accepted')
    }
    await a.dispose()
    pg.collect()
    assertClose((await forward(b)).output, [33], 0)
    assertClose((await c.callAsync('double',{x:[3]})).twice,[12],0)
    assertClose(gpu ? await live.toArrayAsync() : live.toArray(), [19], 0)
    // Other tests' deferred finalizers must not change this measurement's
    // baseline. The isolated runtime contains only explicitly owned objects.
    const runtime = await createRuntime()
    const held = new runtime.Tensor([19], {dtype:'float32'})
    try {
      // Readback creates a GC-owned contiguous wrapper. Do not include that
      // temporary in this import-lifetime baseline; check held's value below.
      await held.realizeAsync()
      runtime.clearScheduleCache(); runtime.collect()
      const fields = ['bufferOwnedBytes', 'bufferEntries', 'tensorRecords']
      const baseline = runtime.stats().coreStats
      assert(baseline.tensorRecords === 1, 'baseline must contain only the held Tensor')
      for (let i = 0; i < 8; i++) {
        const loaded = runtime.Model.load(bytes)
        try {
          assertClose((await loaded.forwardAsync({x:[3]})).output, [6], 0)
          await loaded.writeBufferAsync('alias', new Float32Array([7]))
          assertClose(await loaded.readBufferAsync('w'), [7], 0)
        } finally { await loaded.dispose() }
        runtime.clearScheduleCache(); runtime.collect()
        const current = runtime.stats().coreStats
        for (const field of fields)
          assert(current[field] === baseline[field],
            `import/dispose retained ${field}: ${baseline[field]} -> ${current[field]}`)
      }
      assertClose(await held.toArrayAsync(), [19], 0)
    } finally { held.dispose(); await runtime.dispose() }
  } finally {
    for (const model of models) await model.dispose()
    x.dispose(); w.dispose(); live.dispose()
  }
}

async function checkFamilyRuntimeOwnership(pg) {
  const gpu = String(pg.device).toLowerCase() === 'webgpu'
  const live = new pg.Tensor([19], {dtype:'float32'})
  const specs = [
    ['MLP', {layers:[2, 3, 1]}],
    ['TabM', {layers:[2, 3, 1], n_ensemble:2}],
    ['NAM', {n_features:2, hidden_sizes:[3], n_outputs:1}],
  ]
  try {
    for (const [name, spec] of specs) {
      const beforeStats = pg.stats().coreStats
      const a = pg.models[name](spec), b = pg.models[name](spec)
      try {
        assert(pg.stats().coreStats.tensorRecords === beforeStats.tensorRecords,
          'family retained construction Tensor wrappers')
        const parameter = a.paramName(0)
        const before = await b.readBufferAsync(parameter)
        await a.writeBufferAsync(parameter, new Float32Array(before.length).fill(7))
        assertClose(await b.readBufferAsync(parameter), before, 0)
        const expected = await b.forwardAsync({x:[1, 2]})
        // Tensor I/O checks actual C context ownership. Net buffer counts can
        // fall when a factory allocation collects previously retired storage.
        const input = new pg.Tensor([[1, 2]], {dtype:'float32'})
        let tensorOutputs
        try {
          tensorOutputs = await b.forwardAsync({x:input})
          assertClose(await tensorOutputs.output.toArrayAsync(), expected.output, 0)
        } finally {
          if (tensorOutputs) for (const output of Object.values(tensorOutputs)) output.dispose()
          input.dispose()
        }
        await a.dispose()
        pg.collect()
        assertClose((await b.forwardAsync({x:[1, 2]})).output, expected.output, 0)
        assertClose(gpu ? await live.toArrayAsync() : live.toArray(), [19], 0)
        // Reject before entering a Wasm module suspended on the host bridge.
        pg._activeAsync++
        try {
          let rejected = false
          try { pg.models[name](spec) } catch (e) { rejected = /idle/.test(e.message) }
          assert(rejected, 'factory entered a busy Runtime')
        } finally { pg._activeAsync-- }
      } finally { await a.dispose(); await b.dispose() }
    }
  } finally { live.dispose() }
}

async function runModelRuntimeTests(pg, createRuntime) {
  const Model = pg.Model
  const { MLP, TabM, NAM } = pg.models
  const testFilter = testFilterFor(pg)
  let passed = 0
  let failed = 0

  if (!pg.supportsModel) {
    console.log('\n== Model ==')
    console.log('  [SKIP] core does not expose PolyModel runtime yet')
    return { passed: 0, failed: 0 }
  }

  async function test(name, fn) {
    if (testFilter && !name.includes(testFilter)) return
    try {
      await fn()
      console.log(`  [PASS] ${name}`)
      passed++
    } catch (e) {
      console.log(`  [FAIL] ${name}: ${e.message}`)
      failed++
    }
  }

  console.log('\n== Model ==')

  await test('Model packed GEMM mutation and portable bundle', () => checkPackedGemmModel(pg))
  await test('Model portable GEMM selection', () => checkPortableGemmModel(createRuntime))
  await test('Model interval sum selection and portable gradients', () => checkIntervalSum(createRuntime))
  await test('Model CPU attention probabilities', () => checkCpuAttentionModel(pg))
  await test('Model contract errors and canonical round trip', () => checkModelContractErrors(pg))
  await test('Model ONNX reference graphs and bundle round trips', () => checkOnnxImport(pg))
  await test('Model registered family dispatch', () => checkFamilyRegistry(pg))

  await test('Model checkpoint replacement uses queued readback', () => checkModelCheckpointReplacement(pg))
  await test('Model stateful capture shares train eval state', () => checkModelStatefulCapture(pg))
  await test('Model stateful capture owns resumable RNG', () => checkModelCaptureRng(pg))
  await test('Model stateful capture restores failures and rejects async execution', () => checkModelCaptureFailure(pg))
  await test('Model copied storage exact writes and objective selection', () => checkModelStorageObjectives(pg, Model))
  await test('Model codecs reject malformed bytes before publication', () => checkModelCodecRejection(pg))
  await test('Model quantized weights match pinned GGUF bit planes', () => checkQuantizedModelWeights(pg))

  await test('Model composition factories share C construction', () => checkCompositionFactories(pg))
  await test('Model bounded components match pinned Tensor programs', () => checkBoundedComponents(pg))
  await test('Model integer controls survive portable import', () => checkIntegerControls(pg))
  await test('Model cached Llama partition reset and import', () => checkCachedLlama(pg))
  await test('Model composition catalogue and named target objective', () => checkCompositionCatalogue(pg))
  await test('Model description explicit gradients', () => checkDescriptionGradient(pg))
  await test('Model description state writes and deferred readback', () => checkDescriptionState(pg))
  await test('Model tied Adam placement freeze and checkpoint', () => checkTiedAdamCheckpoint(pg))
  await test('Model constructor collects object state', () => checkModelConstructor(pg, Model))
  await test('Model constructor dispatch and explicit factories', () => checkModelDispatch(pg, Model))
  await test('Model portable Tensor gradients and construction errors', () => checkPortableGradient(pg))
  await test('Model usability summary and capture failures', () => checkModelUsability(pg, Model))
  await test('Model runtime imports isolation and failure', () => checkRuntimeImports(pg, Model, createRuntime))
  await test('Model family runtime ownership', () => checkFamilyRuntimeOwnership(pg))
  await test('Llama family reference and shared import', () => checkLlamaFamily(pg))
  await test('Qwen rotary state and shared import', () => checkQwenRotaryState(pg))
  await test('Vision models reference and portable state', () => checkVisionModels(pg))
  await test('EmbeddingGemma2 text/image/audio reference and portable state', () => checkEmbeddingGemma2(pg))
  await test('Model Tensor I/O owns device results', () => checkModelTensorIO(pg))
  await test('Model variable shapes preserve results and portable signatures', () => checkModelVariableShapes(pg))
  await test('Tokenizer GGUF and JSON byte BPE', () => checkTokenizerJSON(pg))
  await test('Model empty bindings reject before input writes', () => checkModelEmptyInputAdmission(pg))
  await test('Model minibatches match explicit training steps', () => checkModelMinibatches(pg))
  await test('Model bounded minibatches and Tensor datasets', () => checkModelBoundedMinibatches(pg))
  await test('Model trace seals independent state with named loss', () => checkModelTrace(pg, Model))

  await test('generic call validates signature and returns selected outputs', async () => {
    await checkCallSignatureAndSelectedOutputs(pg, Model)
  })

  await test('scalar rank8 and shared multi-output round trip', async () => {
    const scalarX = pg.Tensor.empty([])
    const scalarW = pg.Tensor.full([], 3, { dtype: 'float32' })
    const scalar = await Model.fromTensors({
      inputs: { x: scalarX }, outputs: { output: scalarX.mul(scalarW) },
      params: { w: scalarW }
    })
    let scalarRestored = null
    try {
      scalarRestored = Model.fromIR(scalar.exportIR(), await scalar.exportWeights())
      const result = await scalarRestored.forward({ x: new Float32Array([2]) })
      assertClose(result.output, [6], 0)
      assert(
        JSON.stringify(scalarRestored.bufShape(scalarRestored.findBuf('output'))) === '[]',
        'scalar output shape must remain []'
      )
    } finally {
      if (scalarRestored) scalarRestored.dispose()
      scalar.dispose()
    }

    const shape = [1, 1, 1, 1, 1, 1, 1, 1]
    const x = pg.Tensor.empty(shape)
    const w = pg.Tensor.ones(shape, {})
    const shared = x.add(w)
    const source = await Model.fromTensors({
      inputs: { x },
      outputs: { plus: shared.add(1), minus: shared.sub(1) },
      params: { w }
    })
    let restored = null
    try {
      restored = Model.fromIR(source.exportIR(), await source.exportWeights())
      const result = await restored.forward({ x: new Float32Array([2]) })
      assertClose(result.plus, [4], 0)
      assertClose(result.minus, [2], 0)
      assert(
        JSON.stringify(restored.bufShape(restored.findBuf('plus'))) === JSON.stringify(shape),
        'rank-8 output shape was not preserved'
      )
    } finally {
      if (restored) restored.dispose()
      source.dispose()
    }
  })

  await test('duplicate ABI storage alias fails closed like TinyJit', async () => {
    const x = pg.Tensor.empty([2])
    let error = null
    try {
      const unexpected = await Model.fromTensors({
        inputs: { a: x, b: x }, outputs: { output: x.add(x) }
      })
      unexpected.dispose()
    } catch (err) {
      error = err
    }
    assert(error, 'duplicate ABI storage must be rejected')
  })

  await test('dynamic input alias with persistent state fails closed', async () => {
    const x = pg.Tensor.empty([2])
    let error = null
    try {
      const unexpected = await Model.fromTensors({
        inputs: { x }, outputs: { output: x.add(x) }, params: { w: x }
      })
      unexpected.dispose()
    } catch (err) {
      error = err
    }
    assert(error, 'dynamic input must not alias persistent state')
  })

  await test('output alias of dynamic input round trips', async () => {
    const x = pg.Tensor.empty([2])
    const source = await Model.fromTensors({ inputs: { x }, outputs: { output: x } })
    let restored = null
    try {
      restored = Model.fromIR(source.exportIR())
      const value = new Float32Array([3, 4])
      assertClose((await source.forward({ x: value })).output, value, 0)
      assertClose((await restored.forward({ x: value })).output, value, 0)
    } finally {
      if (restored) restored.dispose()
      source.dispose()
    }
  })

  await test('named partial view state fails closed', async () => {
    const base = new pg.Tensor([1, 2, 3, 4], { dtype: 'float32' })
    const view = base.shrink([[1, 3]])
    const x = pg.Tensor.empty([2])
    let error = null
    try {
      const unexpected = await Model.fromTensors({
        inputs: { x }, outputs: { output: x.add(view) }, params: { base, view }
      })
      unexpected.dispose()
    } catch (err) {
      error = err
    }
    assert(error, 'named partial view storage must be rejected')
  })

  await test('input-dependent named state effect fails closed', async () => {
    const x = pg.Tensor.empty([1])
    const w = new pg.Tensor([1], { dtype: 'float32' })
    const output = w.assign(w.add(x))
    let error = null
    try {
      const unexpected = await Model.fromTensors({
        inputs: { x }, outputs: { output }, params: { w }
      })
      unexpected.dispose()
    } catch (err) {
      error = err
    }
    assert(error, 'input-dependent named state effect must be rejected')
  })

  await test('stochastic output requires named RNG state', async () => {
    pg.Tensor.manual_seed(123)
    const x = pg.Tensor.empty([2])
    let error = null
    try {
      const unexpected = await Model.fromTensors({
        inputs: { x }, outputs: { output: x.add(pg.Tensor.rand(2)) }
      })
      unexpected.dispose()
    } catch (err) {
      error = err
    }
    assert(error, 'stochastic output without named RNG state must be rejected')
  })

  await test('state traversal preserves diamond aliases and stops cycles', async () => {
    const shared = new pg.Tensor([1, 2], { dtype: 'float32' })
    const root = { left: { weight: shared }, right: { weight: shared } }
    root.self = root
    const state = pg.nn.getStateDict(root)
    assert(
      JSON.stringify(Object.keys(state)) === JSON.stringify(['left.weight', 'right.weight']),
      `unexpected state paths: ${Object.keys(state)}`
    )
    assert(state['left.weight'] === state['right.weight'], 'alias paths must retain one Tensor')
    const params = pg.nn.getParameters(root)
    assert(params.length === 2, `unexpected parameter count: ${params.length}`)
    assert(params[0] === shared && params[1] === shared, 'parameter aliases must match state paths')
  })

  await test('float16 Model state preserves exact storage bits', async () => {
    const w = new pg.Tensor([1.5, -2], { dtype: 'float16' })
    const x = pg.Tensor.empty([2], { dtype: 'float16' })
    const inst = await Model.fromTensors({
      inputs: { x },
      outputs: { output: x.add(w) },
      params: { w }
    })
    try {
      assert(inst.paramDtype(0) === 'float16', `unexpected dtype ${inst.paramDtype(0)}`)
      const raw = await inst.paramData(0)
      assert(raw instanceof Uint16Array, `expected Uint16Array, got ${raw.constructor.name}`)
      assert(raw.length === 2 && raw[0] === 0x3e00 && raw[1] === 0xc000,
        `unexpected float16 bits: ${Array.from(raw)}`)
      const restored = Model.fromIR(inst.exportIR(), await inst.exportWeights())
      try {
        assert(restored.paramDtype(0) === 'float16', 'restored dtype must remain float16')
        const restoredRaw = await restored.paramData(0)
        assert(restoredRaw instanceof Uint16Array,
          'restored float16 state must remain raw Uint16Array')
        assert(restoredRaw[0] === 0x3e00 && restoredRaw[1] === 0xc000,
          `unexpected restored bits: ${Array.from(restoredRaw)}`)
      } finally {
        restored.dispose()
      }
    } finally {
      inst.dispose()
    }
  })

  await test('typed Model state round trips exact storage bytes', async () => {
    const cases = [
      ['float64', [1.25, -2.5], Float64Array],
      ['int32', [1, -2], Int32Array],
      ['uint8', [1, 255], Uint8Array],
      ['bool', [true, false], Uint8Array],
      ['bfloat16', [1.5, -2], Uint16Array]
    ]
    for (const [dtype, values, ArrayType] of cases) {
      const w = new pg.Tensor(values, { dtype })
      const x = pg.Tensor.empty([2], { dtype })
      const source = await Model.fromTensors({
        inputs: { x }, outputs: { output: x }, params: { w }
      })
      try {
        const restored = Model.fromIR(source.exportIR(), await source.exportWeights())
        try {
          const before = await source.paramData(0)
          const after = await restored.paramData(0)
          assert(before instanceof ArrayType,
            `${dtype}: expected ${ArrayType.name}, got ${before.constructor.name}`)
          assert(after instanceof ArrayType,
            `${dtype}: restored ${after.constructor.name}`)
          const beforeBytes = new Uint8Array(before.buffer, before.byteOffset, before.byteLength)
          const afterBytes = new Uint8Array(after.buffer, after.byteOffset, after.byteLength)
          assertClose(afterBytes, beforeBytes, 0)
        } finally {
          restored.dispose()
        }
      } finally {
        source.dispose()
      }
    }
  })

  await test('typed integer input preserves bytes and rejects float binding', async () => {
    await checkTypedIntegerInput(pg, Model)
  })

  await test('module device map places exact Tensor cuts atomically', async () => {
    await checkModuleDeviceMap(pg, Model)
  })

  await test('module device map preserves batched Tensor cuts', async () => {
    await checkModuleDeviceMap(pg, Model, true)
  })

  await test('model-family constructors are not Model methods', async () => {
    assert(typeof Model.mlp === 'undefined', 'Model.mlp should not exist')
    assert(typeof MLP === 'function', 'pg.models.MLP should exist')
  })

  await test('mlp create + param enumeration', async () => {
    const inst = MLP({
      layers: [2, 4, 1],
      activation: 'relu',
      bias: true,
      loss: 'mse',
      batch_size: 1,
      seed: 42
    })
    try {
      assert(inst.paramCount === 4, `expected 4 params, got ${inst.paramCount}`)
      assert(inst.paramName(0) === 'layers.0.weight', 'unexpected first param name')
      assert(JSON.stringify(inst.paramShape(0)) === JSON.stringify([4, 2]), 'unexpected first param shape')
    } finally {
      inst.dispose()
    }
  })

  await test('param trainability freezes optimizer updates', async () => {
    const inst = MLP({
      layers: [2, 1],
      activation: 'none',
      bias: true,
      loss: 'mse',
      batch_size: 1,
      seed: 42
    })
    try {
      assert(inst.paramTrainable(0) === true, 'weight should default trainable')
      assert(inst.paramTrainable(1) === true, 'bias should default trainable')
      inst.setParamTrainable(0, false)
      assert(inst.paramTrainable(0) === false, 'weight should be frozen')

      const weightBefore = await inst.paramData(0)
      const biasBefore = await inst.paramData(1)
      inst.setOptimizer(pg.OPTIM_SGD, 0.05)
      const x = new Float32Array([1, 2])
      const y = new Float32Array([5])
      for (let step = 0; step < 10; step++) {
        const loss = await inst.trainStep({ x, y })
        assert(Number.isFinite(loss), `loss should be finite, got ${loss}`)
      }

      assertClose(await inst.paramData(0), weightBefore)
      let biasChanged = false
      const biasAfter = await inst.paramData(1)
      for (let i = 0; i < biasAfter.length; i++) {
        if (biasAfter[i] !== biasBefore[i]) biasChanged = true
      }
      assert(biasChanged, 'unfrozen bias should update')
    } finally {
      inst.dispose()
    }
  })

  await test('param trainability survives IR round trip', async () => {
    const inst1 = MLP({
      layers: [2, 1],
      activation: 'none',
      bias: true,
      loss: 'mse',
      batch_size: 1,
      seed: 42
    })
    try {
      inst1.setParamTrainable(0, false)
      const inst2 = Model.fromIR(inst1.exportIR(), await inst1.exportWeights())
      try {
        assert(inst2.paramTrainable(0) === false, 'frozen flag should round trip')
        assert(inst2.paramTrainable(1) === true, 'unfrozen flag should round trip')
      } finally {
        inst2.dispose()
      }
    } finally {
      inst1.dispose()
    }
  })

  await test('mlp forward produces output', async () => {
    const inst = MLP({
      layers: [2, 4, 1],
      activation: 'relu',
      bias: true,
      loss: 'mse',
      batch_size: 1,
      seed: 42
    })
    try {
      const outputs = await inst.forward({ x: new Float32Array([1, 2]) })
      assert(outputs.output instanceof Float32Array, 'output should be Float32Array')
      assert(outputs.output.length === 1, `expected output length 1, got ${outputs.output.length}`)
      assert(Number.isFinite(outputs.output[0]), 'output should be finite')
    } finally {
      inst.dispose()
    }
  })

  await test('mlp train step decreases loss', async () => {
    const inst = MLP({
      layers: [2, 1],
      activation: 'none',
      bias: true,
      loss: 'mse',
      batch_size: 1,
      seed: 42
    })
    try {
      inst.setOptimizer(pg.OPTIM_SGD, 0.05)
      const x = new Float32Array([1, 2])
      const y = new Float32Array([5])
      let first = null
      let last = null
      for (let step = 0; step < 50; step++) {
        last = await inst.trainStep({ x, y })
        if (first == null) first = last
      }
      assert(last < first, `expected loss to decrease (${first} -> ${last})`)
    } finally {
      inst.dispose()
    }
  })

  await test('weights export/import round trip', async () => {
    const spec = {
      layers: [2, 4, 1],
      activation: 'relu',
      bias: true,
      loss: 'mse',
      batch_size: 1,
      seed: 42
    }
    const inst1 = MLP(spec)
    const inst2 = MLP({ ...spec, seed: 99 })
    try {
      const original = await inst1.paramData(0)
      const different = await inst2.paramData(0)
      let anyDiff = false
      for (let i = 0; i < original.length; i++) {
        if (original[i] !== different[i]) {
          anyDiff = true
          break
        }
      }
      assert(anyDiff, 'different seed should change weights')

      const weights = await inst1.exportWeights()
      assert(weights instanceof Uint8Array && weights.length > 0, 'expected non-empty weights export')
      inst2.importWeights(weights)
      assertClose(await inst2.paramData(0), original)
    } finally {
      inst1.dispose()
      inst2.dispose()
    }
  })

  await test('Adam checkpoint resumes the uninterrupted training trajectory', async () => {
    const spec = {
      layers: [2, 1], activation: 'none', bias: false,
      loss: 'mse', batch_size: 1, seed: 7
    }
    const source = MLP(spec)
    let restored = null
    try {
      source.setOptimizer(pg.OPTIM_ADAM, 0.05)
      const io = { x: new Float32Array([1, 2]), y: new Float32Array([3]) }
      for (let i = 0; i < 3; i++) await source.trainStep(io)
      restored = Model.fromIR(source.exportIR(), await source.exportWeights())
      restored.setOptimizer(pg.OPTIM_ADAM, 0.05)

      const sourceLoss = await source.trainStep(io)
      const restoredLoss = await restored.trainStep(io)
      assert(sourceLoss === restoredLoss,
        `restored loss ${restoredLoss} != uninterrupted ${sourceLoss}`)
      assertClose(await restored.paramData(0), await source.paramData(0), 0)
      for (const name of [
        'optim.adam.b1_t', 'optim.adam.b2_t',
        'optim.adam.m.layers.0.weight', 'optim.adam.v.layers.0.weight'
      ]) {
        const sourceIndex = source.findBuf(name)
        const restoredIndex = restored.findBuf(name)
        assert(sourceIndex >= 0 && restoredIndex >= 0, `missing ${name}`)
        assertClose(
          await restored.bufData(restoredIndex), await source.bufData(sourceIndex), 0
        )
      }
    } finally {
      if (restored) restored.dispose()
      source.dispose()
    }
  })

  await test('stochastic named state requires checkpoint for portable activation', async () => {
    pg.Tensor.manual_seed(11)
    const w = pg.Tensor.rand(2, {})
    const x = pg.Tensor.empty([2])
    const source = await Model.fromTensors({
      inputs: { x }, outputs: { output: x.mul(w) }, params: { w }
    })
    let restored = null
    try {
      const ir = source.exportIR()
      const weights = await source.exportWeights()
      let rejected = false
      try {
        const unexpected = Model.fromIR(ir)
        unexpected.dispose()
      } catch (err) {
        rejected = true
      }
      assert(rejected, 'fresh stochastic activation must fail without checkpoint bytes')
      restored = Model.fromIR(ir, weights)
      const input = new Float32Array([2, 3])
      const sourceOut = await source.forward({ x: input })
      const restoredOut = await restored.forward({ x: input })
      assertClose(restoredOut.output, sourceOut.output, 0)
    } finally {
      if (restored) restored.dispose()
      source.dispose()
    }
  })

  await test('ir export/fromIR round trip', async () => {
    const inst1 = MLP({
      layers: [2, 4, 1],
      activation: 'relu',
      bias: true,
      loss: 'mse',
      batch_size: 1,
      seed: 42
    })
    try {
      const ir = inst1.exportIR()
      const weights = await inst1.exportWeights()
      const inst2 = Model.fromIR(ir, weights)
      try {
        const out1 = (await inst1.forward({ x: new Float32Array([1, 2]) })).output
        const out2 = (await inst2.forward({ x: new Float32Array([1, 2]) })).output
        assertClose(out2, out1)
      } finally {
        inst2.dispose()
      }
    } finally {
      inst1.dispose()
    }
  })

  await test('bound program export uses separate weights and has no portable IR', async () => {
    const inst1 = MLP({
      layers: [2, 4, 1],
      activation: 'relu',
      bias: true,
      loss: 'none',
      batch_size: 1,
      seed: 42
    })
    let inst2 = null
    try {
      const program = await inst1.exportProgramAsync()
      const weights = await inst1.exportWeights()
      assert(program && new TextDecoder().decode(program.subarray(0, 4)) === 'PGPM',
        'compiled program magic mismatch')
      let missingWeightsRejected = false
      try {
        const unexpected = Model.fromProgram(program)
        await unexpected.dispose()
      } catch (err) {
        missingWeightsRejected = true
      }
      assert(missingWeightsRejected, 'bound state must require separate weights')
      inst2 = Model.fromProgram(program, weights)
      const input = new Float32Array([1.25, -0.5])
      const out1 = (await inst1.forward({ x: input })).output
      const out2 = (await inst2.forward({ x: input })).output
      assertClose(out2, out1, 0)
      const portable = inst2.exportIR()
      assert(!portable || portable.length === 0, 'program-only Model exposed portable IR')
      const program2 = await inst2.exportProgramAsync()
      assertClose(program2, program, 0)
    } finally {
      if (inst2) await inst2.dispose()
      await inst1.dispose()
    }
  })

  await test('mlp batch_size=32 forward produces correct shape', async () => {
    const inst = MLP({
      layers: [4, 8, 3],
      activation: 'relu',
      bias: true,
      loss: 'cross_entropy',
      batch_size: 32,
      seed: 42
    })
    try {
      const x = new Float32Array(32 * 4)
      for (let i = 0; i < x.length; i++) x[i] = Math.random()
      const outputs = await inst.forward({ x })
      assert(outputs.output instanceof Float32Array, 'output should be Float32Array')
      assert(outputs.output.length === 32 * 3,
        `expected output length ${32 * 3}, got ${outputs.output.length}`)
      for (let i = 0; i < outputs.output.length; i++) {
        assert(Number.isFinite(outputs.output[i]),
          `output[${i}] should be finite, got ${outputs.output[i]}`)
      }
    } finally {
      inst.dispose()
    }
  })

  // Known bug: batch_size>1 cross_entropy backward has shape mismatch in codegen
  // Reproduces on CPU too (not WASM-specific). See PLAN.md P0.
  await test('mlp batch_size=32 train step decreases loss (P0)', async () => {
    const inst = MLP({
      layers: [4, 8, 3],
      activation: 'relu',
      bias: true,
      loss: 'cross_entropy',
      batch_size: 32,
      seed: 42
    })
    try {
      inst.setOptimizer(pg.OPTIM_SGD, 0.01)
      const x = new Float32Array(32 * 4)
      const y = new Float32Array(32 * 3)
      for (let i = 0; i < x.length; i++) x[i] = (i % 7) * 0.1
      for (let i = 0; i < 32; i++) y[i * 3 + (i % 3)] = 1.0
      let first = null
      let last = null
      for (let step = 0; step < 30; step++) {
        last = await inst.trainStep({ x, y })
        if (first == null) first = last
      }
      assert(Number.isFinite(first), `first loss should be finite, got ${first}`)
      assert(Number.isFinite(last), `last loss should be finite, got ${last}`)
      assert(last < first, `expected loss to decrease (${first} -> ${last})`)
    } finally {
      inst.dispose()
    }
  })

  await test('tabm and nam builders are available', async () => {
    const tabm = TabM({
      layers: [2, 4, 1],
      activation: 'relu',
      loss: 'mse',
      batch_size: 1,
      seed: 42,
      n_ensemble: 4
    })
    const nam = NAM({
      n_features: 2,
      hidden_sizes: [4],
      activation: 'relu',
      n_outputs: 1,
      loss: 'mse',
      batch_size: 1,
      seed: 42
    })
    try {
      const tabmOut = await tabm.forward({ x: new Float32Array([1, 2]) })
      const namOut = await nam.forward({ x: new Float32Array([1, 2]) })
      assert(tabmOut.output instanceof Float32Array, 'tabm output missing')
      assert(namOut.output instanceof Float32Array, 'nam output missing')
    } finally {
      tabm.dispose()
      nam.dispose()
    }
  })

  await test('mlp train step with Adam', async () => {
    const inst = MLP({
      layers: [2, 4, 1],
      activation: 'relu',
      bias: true,
      loss: 'mse',
      batch_size: 1,
      seed: 42
    })
    try {
      inst.setOptimizer(pg.OPTIM_ADAM, 0.01)
      const x = new Float32Array([1, 2])
      const y = new Float32Array([3])
      let first = null
      let last = null
      for (let step = 0; step < 50; step++) {
        last = await inst.trainStep({ x, y })
        if (first == null) first = last
      }
      assert(last < first, `expected loss to decrease (${first} -> ${last})`)
    } finally {
      inst.dispose()
    }
  })

  await test('mlp train step with SGD momentum creates named state', async () => {
    const inst = MLP({
      layers: [2, 4, 1],
      activation: 'relu',
      bias: true,
      loss: 'mse',
      batch_size: 1,
      seed: 42
    })
    try {
      inst.setOptimizer(pg.OPTIM_SGD, 0.01, 0.9, 0.999, 1e-8, 0.0, 0.9)
      const x = new Float32Array([1, 2])
      const y = new Float32Array([3])
      const loss = await inst.trainStep({ x, y })
      assert(Number.isFinite(loss), `loss should be finite, got ${loss}`)
      const bi = inst.findBuf('optim.sgd.b.layers.0.weight')
      assert(bi >= 0, 'missing SGD momentum state buffer')
      const b = await inst.bufData(bi)
      assert(Array.from(b).some(v => Math.abs(v) > 0), 'momentum state should update')
      const defaultNames = safetensorNames(await inst.exportWeights())
      assert(defaultNames.has('optim.sgd.b.layers.0.weight'), 'default export should include optimizer state')
      const modelOnlyNames = safetensorNames(await inst.exportWeights({ includeOptimizer: false }))
      assert(modelOnlyNames.has('layers.0.weight'), 'model-only export should include params')
      assert(!modelOnlyNames.has('optim.sgd.b.layers.0.weight'), 'model-only export should exclude optimizer state')
    } finally {
      inst.dispose()
    }
  })

  await test('mlp batch_size=4 mse train', async () => {
    const inst = MLP({
      layers: [2, 4, 2],
      activation: 'relu',
      bias: true,
      loss: 'mse',
      batch_size: 4,
      seed: 42
    })
    try {
      inst.setOptimizer(pg.OPTIM_SGD, 0.01)
      const x = new Float32Array(4 * 2).fill(0.5)
      const y = new Float32Array(4 * 2).fill(0.3)
      let first = null
      let last = null
      for (let step = 0; step < 50; step++) {
        last = await inst.trainStep({ x, y })
        if (first == null) first = last
      }
      assert(Number.isFinite(first), `first loss should be finite, got ${first}`)
      assert(Number.isFinite(last), `last loss should be finite, got ${last}`)
      assert(last < first, `expected loss to decrease (${first} -> ${last})`)
    } finally {
      inst.dispose()
    }
  })

  await test('mlp 100-step convergence', async () => {
    const inst = MLP({
      layers: [2, 8, 1],
      activation: 'relu',
      bias: true,
      loss: 'mse',
      batch_size: 1,
      seed: 42
    })
    try {
      inst.setOptimizer(pg.OPTIM_SGD, 0.01)
      const x = new Float32Array([1, 2])
      const y = new Float32Array([5])
      let first = null
      let last = null
      for (let step = 0; step < 100; step++) {
        last = await inst.trainStep({ x, y })
        if (first == null) first = last
      }
      assert(last < first * 0.1, `expected >90% loss reduction (${first} -> ${last})`)
    } finally {
      inst.dispose()
    }
  })

  console.log(`\nModel tests: ${passed} passed, ${failed} failed`)
  return { passed, failed }
}

async function runModelSmokeTests(pg, createRuntime) {
  const Model = pg.Model
  const { MLP, TabM, NAM } = pg.models
  const testFilter = testFilterFor(pg)
  let passed = 0
  let failed = 0

  if (!pg.supportsModel) {
    console.log('\n== Model ==')
    console.log('  [SKIP] core does not expose PolyModel runtime yet')
    return { passed: 0, failed: 0 }
  }

  async function test(name, fn) {
    if (testFilter && !name.includes(testFilter)) return
    try {
      await fn()
      console.log(`  [PASS] ${name}`)
      passed++
    } catch (e) {
      console.log(`  [FAIL] ${name}: ${e.message}`)
      failed++
    }
  }

  console.log('\n== Model ==')

  await test('Model contract errors and canonical round trip', () => checkModelContractErrors(pg))
  await test('Model ONNX reference graphs and bundle round trips', () => checkOnnxImport(pg))
  await test('Model packed GEMM mutation and portable bundle', () => checkPackedGemmModel(pg))
  await test('Model portable GEMM selection', () => checkPortableGemmModel(createRuntime))
  await test('Model interval sum selection and portable gradients', () => checkIntervalSum(createRuntime))
  await test('Model CPU attention probabilities', () => checkCpuAttentionModel(pg))
  await test('Model stateful capture shares train eval state', () => checkModelStatefulCapture(pg))
  await test('Model registered family dispatch', () => checkFamilyRegistry(pg))
  await test('Model stateful capture owns resumable RNG', () => checkModelCaptureRng(pg))
  await test('Model stateful capture restores failures and rejects async execution', () => checkModelCaptureFailure(pg))
  await test('Model composition factories share C construction', () => checkCompositionFactories(pg))
  await test('Model bounded components match pinned Tensor programs', () => checkBoundedComponents(pg))
  await test('Model integer controls survive portable import', () => checkIntegerControls(pg))
  await test('Model cached Llama partition reset and import', () => checkCachedLlama(pg))
  await test('Model checkpoint replacement uses queued readback', () => checkModelCheckpointReplacement(pg))
  await test('Model composition catalogue and named target objective', () => checkCompositionCatalogue(pg))
  await test('Model description explicit gradients', () => checkDescriptionGradient(pg))
  await test('Model description state writes and deferred readback', () => checkDescriptionState(pg))
  await test('Model tied Adam placement freeze and checkpoint', () => checkTiedAdamCheckpoint(pg))
  await test('Model constructor collects object state', () => checkModelConstructor(pg, Model))
  await test('Model constructor dispatch and explicit factories', () => checkModelDispatch(pg, Model))
  await test('Model usability summary and capture failures', () => checkModelUsability(pg, Model))
  await test('Model runtime imports isolation and failure', () => checkRuntimeImports(pg, Model, createRuntime))
  await test('Model family runtime ownership', () => checkFamilyRuntimeOwnership(pg))
  await test('Llama family reference and shared import', () => checkLlamaFamily(pg))
  await test('Model Tensor I/O owns device results', () => checkModelTensorIO(pg))
  await test('Qwen rotary state and shared import', () => checkQwenRotaryState(pg))
  await test('Vision models reference and portable state', () => checkVisionModels(pg))
  await test('EmbeddingGemma2 text/image/audio reference and portable state', () => checkEmbeddingGemma2(pg))
  await test('Model variable shapes preserve results and portable signatures', () => checkModelVariableShapes(pg))
  await test('Tokenizer GGUF and JSON byte BPE', () => checkTokenizerJSON(pg))
  await test('Model empty bindings reject before input writes', () => checkModelEmptyInputAdmission(pg))
  await test('Model minibatches match explicit training steps', () => checkModelMinibatches(pg))
  await test('Model bounded minibatches and Tensor datasets', () => checkModelBoundedMinibatches(pg))
  await test('Model trace seals independent state with named loss', () => checkModelTrace(pg, Model))

  await test('Model copied storage exact writes and objective selection', () => checkModelStorageObjectives(pg, Model))
  await test('Model codecs reject malformed bytes before publication', () => checkModelCodecRejection(pg))
  await test('Model quantized weights match pinned GGUF bit planes', () => checkQuantizedModelWeights(pg))

  await test('typed integer input preserves bytes and rejects float binding', async () => {
    await checkTypedIntegerInput(pg, Model)
  })

  await test('generic call validates signature and returns selected outputs', async () => {
    await checkCallSignatureAndSelectedOutputs(pg, Model)
  })

  await test('webgpu module device map places exact Tensor cuts atomically', async () => {
    await checkModuleDeviceMap(pg, Model)
  })

  await test('webgpu module device map preserves batched Tensor cuts', async () => {
    await checkModuleDeviceMap(pg, Model, true)
  })

  await test('webgpu mlp forward smoke', async () => {
    const inst = MLP({
      layers: [2, 4, 1],
      activation: 'relu',
      bias: true,
      loss: 'mse',
      batch_size: 1,
      seed: 42
    })
    try {
      const outputs = await inst.forward({ x: new Float32Array([1, 2]) })
      assert(outputs.output instanceof Float32Array, 'output should be Float32Array')
      assert(outputs.output.length === 1, `expected output length 1, got ${outputs.output.length}`)
      assert(Number.isFinite(outputs.output[0]), 'output should be finite')
    } finally {
      inst.dispose()
    }
  })

  console.log(`\nModel smoke tests: ${passed} passed, ${failed} failed`)
  return { passed, failed }
}

module.exports = { runModelRuntimeTests, runModelSmokeTests }
