// WebGPU adapter/device setup and buffer helpers. Works on the main thread and in workers.
import heatWgsl from './heat.wgsl?raw';

export function gpuAvailable(): boolean {
  return typeof navigator !== 'undefined' && 'gpu' in navigator && navigator.gpu != null;
}

export async function requestAdapter(): Promise<GPUAdapter | null> {
  if (!gpuAvailable()) return null;
  return navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
}

/** Optional features the harness uses when the adapter offers them. */
const OPTIONAL_FEATURES: GPUFeatureName[] = ['timestamp-query', 'subgroups' as GPUFeatureName];

/** Limits raised to the adapter's maximum, since large sweeps exceed the WebGPU defaults. */
const RAISED_LIMITS = [
  'maxBufferSize',
  'maxStorageBufferBindingSize',
  'maxComputeWorkgroupStorageSize',
  'maxComputeInvocationsPerWorkgroup',
  'maxComputeWorkgroupSizeX',
  'maxComputeWorkgroupSizeY',
  'maxComputeWorkgroupsPerDimension',
] as const;

export async function requestDevice(adapter: GPUAdapter): Promise<GPUDevice> {
  const requiredFeatures = OPTIONAL_FEATURES.filter((f) => adapter.features.has(f));
  const requiredLimits: Record<string, number> = {};
  for (const name of RAISED_LIMITS) requiredLimits[name] = adapter.limits[name];
  return adapter.requestDevice({ requiredFeatures, requiredLimits });
}

let shared: Promise<GPUDevice> | null = null;

/**
 * One device shared by every GPU implementation in this worker. The runner creates it
 * before measuring, so device creation isn't charged to whichever implementation runs first.
 */
export function sharedDevice(): Promise<GPUDevice> {
  shared ??= (async () => {
    const adapter = await requestAdapter();
    if (!adapter) throw new Error('no WebGPU adapter in this context');
    const device = await requestDevice(adapter);
    device.lost.then(() => (shared = null));
    return device;
  })().catch((e) => {
    shared = null;
    throw e;
  });
  return shared;
}

/**
 * Copies `size` bytes of `src` (needs COPY_SRC usage), from byte `srcOffset` on, into a
 * staging buffer and maps it. Pass a reusable `staging` buffer (COPY_DST | MAP_READ,
 * ≥ size bytes) to skip allocating one per call; it is left unmapped. Otherwise a
 * temporary one is created and destroyed.
 */
export async function readBuffer(device: GPUDevice, src: GPUBuffer, size: number, staging?: GPUBuffer, srcOffset = 0): Promise<ArrayBuffer> {
  const buffer = staging ?? device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(src, srcOffset, buffer, 0, size);
  device.queue.submit([encoder.finish()]);
  await buffer.mapAsync(GPUMapMode.READ, 0, size);
  const copy = buffer.getMappedRange(0, size).slice(0);
  if (staging) buffer.unmap();
  else buffer.destroy();
  return copy;
}

/** A staging buffer for readBuffer(), reusable across reads of up to `size` bytes. */
export function createStagingBuffer(device: GPUDevice, size: number): GPUBuffer {
  return device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
}

/** vec4s per heater dispatch: 32 MB in and 32 MB out, ~0.5–2 ms per dispatch depending on the clock. */
const HEAT_VEC4S = 2 ** 21;
/** Heater submissions kept in flight, so the GPU never waits on the CPU. */
const HEAT_IN_FLIGHT = 3;

interface Heater {
  pipeline: GPUComputePipeline;
  bindGroup: GPUBindGroup;
}

const heaters = new WeakMap<GPUDevice, Promise<Heater>>();

function heater(device: GPUDevice): Promise<Heater> {
  let h = heaters.get(device);
  if (!h) {
    h = (async () => {
      const pipeline = await device.createComputePipelineAsync({
        layout: 'auto',
        compute: { module: device.createShaderModule({ code: heatWgsl }), entryPoint: 'main' },
      });
      const size = HEAT_VEC4S * 16;
      const src = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE });
      const dst = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE });
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: src } },
          { binding: 1, resource: { buffer: dst } },
        ],
      });
      return { pipeline, bindGroup };
    })();
    heaters.set(device, h);
  }
  return h;
}

/**
 * Keeps the GPU saturated for `ms` with back-to-back heater dispatches, so the OS's
 * frequency governor raises the GPU clock before an implementation is measured. A
 * GPU that has been mostly idle (as it is between short, CPU-synchronized iterations)
 * otherwise runs at its lowest clock for a while, and kernel times jump by up to ~4.6×
 * (M4 Pro) when the governor changes its mind mid-record. Returns the dispatch count.
 */
export async function heatGpu(device: GPUDevice, ms: number): Promise<number> {
  const { pipeline, bindGroup } = await heater(device);
  const groups = HEAT_VEC4S / 256;
  const x = Math.min(groups, device.limits.maxComputeWorkgroupsPerDimension);
  const inFlight: Promise<undefined>[] = [];
  let dispatches = 0;
  const t0 = performance.now();
  while (performance.now() - t0 < ms) {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(x, Math.ceil(groups / x));
    pass.end();
    device.queue.submit([encoder.finish()]);
    dispatches++;
    inFlight.push(device.queue.onSubmittedWorkDone());
    if (inFlight.length >= HEAT_IN_FLIGHT) await inFlight.shift();
  }
  await Promise.all(inFlight);
  return dispatches;
}
