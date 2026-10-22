// WebGPU adapter/device setup and buffer helpers. Works on the main thread and in workers.

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
 * Copies `size` bytes of `src` (needs COPY_SRC usage) into a staging buffer and maps it.
 * Pass a reusable `staging` buffer (COPY_DST | MAP_READ, ≥ size bytes) to skip allocating
 * one per call; it is left unmapped. Otherwise a temporary one is created and destroyed.
 */
export async function readBuffer(device: GPUDevice, src: GPUBuffer, size: number, staging?: GPUBuffer): Promise<ArrayBuffer> {
  const buffer = staging ?? device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(src, 0, buffer, 0, size);
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
