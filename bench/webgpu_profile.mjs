// Benchmark-only observation. Timed calls keep their normal submission path;
// timestamp queries and method instrumentation are installed only for profiles.
export function observeWebGPU() {
  if (!globalThis.GPUAdapter) throw new Error('WebGPU adapter API missing');
  const devices = [], shaders = [];
  const request = GPUAdapter.prototype.requestDevice;
  GPUAdapter.prototype.requestDevice = async function (...args) {
    const device = await request.apply(this, args);
    const info = this.info ?? {};
    const pipelines = new WeakMap(), modules = new WeakMap();
    const record = { device, pipelines, info: {
      vendor: info.vendor, architecture: info.architecture, device: info.device,
      description: info.description, fallback: this.isFallbackAdapter,
      features: [...device.features],
    } };
    devices.push(record);
    const shader = device.createShaderModule.bind(device);
    device.createShaderModule = desc => {
      const module = shader(desc);
      modules.set(module, desc.code);
      shaders.push(desc.code);
      return module;
    };
    for (const method of ['createComputePipeline', 'createComputePipelineAsync']) {
      const original = device[method].bind(device);
      device[method] = desc => {
        const result = original(desc);
        const remember = pipeline => { pipelines.set(pipeline, modules.get(desc.compute.module)); return pipeline; };
        return method.endsWith('Async') ? result.then(remember) : remember(result);
      };
    }
    return device;
  };
  return {
    shaders,
    metadata: () => devices.map(r => r.info),
    async profile(call, { timestamps = true } = {}) {
      const result = { passes: [], dispatches: [], submissions: 0, maps: 0, copyBytes: 0, events: [] };
      let start;
      const at = () => performance.now() - start;
      const restores = [], resources = [];
      const replace = (object, key, fn) => {
        const original = object[key];
        object[key] = fn(original);
        restores.push(() => { object[key] = original; });
      };
      const capacity = 2048;
      try {
        for (const { device, pipelines } of devices) {
          // Allocation and query readback are outside the measured call.
          const timed = timestamps && device.features.has('timestamp-query');
          const query = timed ? device.createQuerySet({ type: 'timestamp', count: capacity }) : null;
          const resolve = timed ? device.createBuffer({ size: capacity * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }) : null;
          const read = timed ? device.createBuffer({ size: capacity * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }) : null;
          const resource = { device, query, resolve, read, count: 0, passes: [] };
          resources.push(resource);
          replace(device.queue, 'submit', original => function (...args) {
            result.submissions++; result.events.push({ kind: 'submit', at: at() });
            return original.apply(this, args);
          });
          replace(device, 'popErrorScope', original => function (...args) {
            const event = { kind: 'validation', at: at() };
            result.events.push(event);
            return original.apply(this, args).then(value => { event.end = at(); return value; });
          });
          replace(device, 'createCommandEncoder', original => function (...args) {
            const encoder = original.apply(this, args);
            const copy = encoder.copyBufferToBuffer.bind(encoder);
            encoder.copyBufferToBuffer = (...args) => { result.copyBytes += args[4]; return copy(...args); };
            const begin = encoder.beginComputePass.bind(encoder);
            encoder.beginComputePass = (desc = {}) => {
              if (desc.timestampWrites) throw new Error('benchmark cannot replace existing timestamps');
              const index = resource.count;
              if (timed && index + 2 > capacity) throw new Error('profile timestamp capacity exceeded');
              resource.count += 2;
              const entry = { device: devices.findIndex(r => r.device === device), dispatches: 0 };
              result.passes.push(entry); resource.passes.push(entry);
              const pass = begin(timed ? { ...desc, timestampWrites: { querySet: query,
                beginningOfPassWriteIndex: index, endOfPassWriteIndex: index + 1 } } : desc);
              let pipeline;
              const set = pass.setPipeline.bind(pass), dispatch = pass.dispatchWorkgroups.bind(pass);
              pass.setPipeline = value => { pipeline = value; return set(value); };
              pass.dispatchWorkgroups = (...grid) => {
                entry.dispatches++;
                result.dispatches.push({ grid, wgsl: pipelines.get(pipeline) ?? null });
                return dispatch(...grid);
              };
              return pass;
            };
            return encoder;
          });
        }
        replace(GPUBuffer.prototype, 'mapAsync', original => function (...args) {
          result.maps++;
          const event = { kind: 'map', at: at() };
          result.events.push(event);
          return original.apply(this, args).then(value => { event.end = at(); return value; });
        });
        start = performance.now();
        await call();
        result.hostCallMs = at();
      } finally {
        for (const restore of restores.reverse()) restore();
        try {
          for (const r of resources) {
            if (!r.query || !r.count) continue;
            const encoder = r.device.createCommandEncoder();
            encoder.resolveQuerySet(r.query, 0, r.count, r.resolve, 0);
            encoder.copyBufferToBuffer(r.resolve, 0, r.read, 0, r.count * 8);
            r.device.queue.submit([encoder.finish()]);
            await r.read.mapAsync(GPUMapMode.READ);
            const stamps = new BigUint64Array(r.read.getMappedRange());
            for (let i = 0; i < r.passes.length; i++) {
              if (stamps[2*i+1] < stamps[2*i]) throw new Error('negative GPU timestamp interval');
              r.passes[i].milliseconds = Number(stamps[2*i+1] - stamps[2*i]) / 1e6;
            }
            r.read.unmap();
          }
        } finally {
          for (const r of resources) { r.query?.destroy(); r.resolve?.destroy(); r.read?.destroy(); }
        }
      }
      result.gpuMs = result.passes.length && result.passes.every(p => p.milliseconds !== undefined)
        ? result.passes.reduce((sum, p) => sum + p.milliseconds, 0) : null;
      return result;
    },
  };
}
