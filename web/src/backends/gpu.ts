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

/** Copies `size` bytes of `src` (needs COPY_SRC usage) into a staging buffer and maps it. */
export async function readBuffer(device: GPUDevice, src: GPUBuffer, size: number): Promise<ArrayBuffer> {
  const staging = device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(src, 0, staging, 0, size);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const copy = staging.getMappedRange().slice(0);
  staging.destroy();
  return copy;
}
