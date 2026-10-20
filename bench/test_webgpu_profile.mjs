import test from 'node:test';
import assert from 'node:assert/strict';
import { observeWebGPU } from './webgpu_profile.mjs';

test('GPU observer counts calls without inventing unsupported timestamps and restores hooks', async t => {
  const saved = { GPUAdapter: globalThis.GPUAdapter, GPUBuffer: globalThis.GPUBuffer };
  t.after(() => {
    for (const [key, value] of Object.entries(saved)) {
      if (value === undefined) delete globalThis[key]; else globalThis[key] = value;
    }
  });
  class Buffer { async mapAsync() {} }
  const device = {
    features: new Set(),
    queue: { submit() {} },
    createShaderModule: desc => ({ desc }),
    createComputePipeline: desc => ({ desc }),
    createComputePipelineAsync: async desc => ({ desc }),
    popErrorScope: async () => null,
    createCommandEncoder: () => ({
      copyBufferToBuffer() {},
      beginComputePass: () => ({ setPipeline() {}, dispatchWorkgroups() {} }),
    }),
  };
  globalThis.GPUAdapter = class { async requestDevice() { return device; } };
  globalThis.GPUBuffer = Buffer;
  const observer = observeWebGPU();
  const actual = await new GPUAdapter().requestDevice();
  const module = actual.createShaderModule({ code: 'shader' });
  const pipeline = await actual.createComputePipelineAsync({ compute: { module } });
  const submit = device.queue.submit, encoder = device.createCommandEncoder;
  const map = Buffer.prototype.mapAsync, pop = device.popErrorScope;
  const result = await observer.profile(async () => {
    const enc = actual.createCommandEncoder(), pass = enc.beginComputePass();
    pass.setPipeline(pipeline); pass.dispatchWorkgroups(3, 2, 1);
    enc.copyBufferToBuffer({}, 0, {}, 0, 16);
    actual.queue.submit([]);
    await actual.popErrorScope(); await new Buffer().mapAsync();
  });
  assert.equal(result.gpuMs, null);
  assert.deepEqual(result.dispatches, [{ grid: [3, 2, 1], wgsl: 'shader' }]);
  assert.equal(result.passes[0].dispatches, 1);
  assert.equal(result.submissions, 1); assert.equal(result.maps, 1);
  assert.equal(result.copyBytes, 16);
  assert.deepEqual(result.events.map(e => e.kind), ['submit', 'validation', 'map']);
  assert.ok(result.events.every(e => e.at >= 0));
  device.features.add('timestamp-query');
  // No query creation API exists in this mock: disabling timestamps must
  // leave the command stream untouched even on a capable adapter.
  const untimed = await observer.profile(async () => {
    actual.createCommandEncoder().beginComputePass().dispatchWorkgroups(1);
  }, { timestamps: false });
  assert.equal(untimed.gpuMs, null);
  assert.equal(untimed.dispatches.length, 1);
  device.features.clear();
  await assert.rejects(observer.profile(async () => { throw new Error('call failed'); }), /call failed/);
  assert.equal(device.queue.submit, submit); assert.equal(device.createCommandEncoder, encoder);
  assert.equal(Buffer.prototype.mapAsync, map); assert.equal(device.popErrorScope, pop);
});
