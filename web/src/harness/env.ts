// Environment metadata recorded with every run (PLAN.md §8). Runs unchanged on the
// main thread and in the benchmark worker, so the two can be compared.
import { gpuAvailable, requestAdapter } from '../backends/gpu';
import { detectWasmFeatures, type WasmFeatures } from '../backends/wasm';

export interface GpuAdapterEnv {
  info: {
    vendor: string;
    architecture: string;
    device: string;
    description: string;
    subgroupMinSize?: number;
    subgroupMaxSize?: number;
    isFallbackAdapter?: boolean;
  };
  features: string[];
  limits: Record<string, number>;
  timestampQuery: boolean;
}

export interface GpuEnv {
  /** `navigator.gpu` exists in this context. */
  api: boolean;
  adapter: GpuAdapterEnv | null;
  error?: string;
}

export interface EnvReport {
  context: 'window' | 'worker';
  collectedAt: string;
  harnessCommit: string;
  userAgent: string;
  /** Chromium only: high-entropy client hints. */
  userAgentData?: Record<string, unknown>;
  hardwareConcurrency: number;
  crossOriginIsolated: boolean;
  isSecureContext: boolean;
  sharedArrayBuffer: boolean;
  /** Smallest observed non-zero step of performance.now(), in ms. */
  timerResolutionMs: number;
  wasm: WasmFeatures;
  gpu: GpuEnv;
}

const ADAPTER_LIMITS = [
  'maxBufferSize',
  'maxStorageBufferBindingSize',
  'maxComputeWorkgroupStorageSize',
  'maxComputeInvocationsPerWorkgroup',
  'maxComputeWorkgroupSizeX',
  'maxComputeWorkgroupSizeY',
  'maxComputeWorkgroupSizeZ',
  'maxComputeWorkgroupsPerDimension',
  'maxStorageBuffersPerShaderStage',
] as const;

export async function collectEnv(): Promise<EnvReport> {
  const [wasm, gpu, userAgentData] = await Promise.all([detectWasmFeatures(), collectGpu(), collectUserAgentData()]);
  return {
    context: typeof window === 'undefined' ? 'worker' : 'window',
    collectedAt: new Date().toISOString(),
    harnessCommit: __GIT_COMMIT__,
    userAgent: navigator.userAgent,
    userAgentData,
    hardwareConcurrency: navigator.hardwareConcurrency,
    crossOriginIsolated: self.crossOriginIsolated === true,
    isSecureContext: self.isSecureContext,
    sharedArrayBuffer: typeof SharedArrayBuffer !== 'undefined',
    timerResolutionMs: measureTimerResolution(),
    wasm,
    gpu,
  };
}

async function collectGpu(): Promise<GpuEnv> {
  if (!gpuAvailable()) return { api: false, adapter: null };
  try {
    const adapter = await requestAdapter();
    if (!adapter) return { api: true, adapter: null, error: 'requestAdapter() returned null' };
    const info = adapter.info as GPUAdapterInfo & { subgroupMinSize?: number; subgroupMaxSize?: number; isFallbackAdapter?: boolean };
    const limits: Record<string, number> = {};
    for (const name of ADAPTER_LIMITS) limits[name] = adapter.limits[name];
    return {
      api: true,
      adapter: {
        info: {
          vendor: info.vendor,
          architecture: info.architecture,
          device: info.device,
          description: info.description,
          subgroupMinSize: info.subgroupMinSize,
          subgroupMaxSize: info.subgroupMaxSize,
          isFallbackAdapter: info.isFallbackAdapter,
        },
        features: [...adapter.features].sort(),
        limits,
        timestampQuery: adapter.features.has('timestamp-query'),
      },
    };
  } catch (e) {
    return { api: true, adapter: null, error: String(e) };
  }
}

async function collectUserAgentData(): Promise<Record<string, unknown> | undefined> {
  const uaData = (navigator as { userAgentData?: { getHighEntropyValues(hints: string[]): Promise<Record<string, unknown>> } })
    .userAgentData;
  if (!uaData) return undefined;
  try {
    return await uaData.getHighEntropyValues(['architecture', 'bitness', 'model', 'platformVersion', 'fullVersionList']);
  } catch {
    return undefined;
  }
}

/** Busy-waits ~20 ms and returns the smallest non-zero step seen. Browsers coarsen this unless cross-origin isolated. */
function measureTimerResolution(): number {
  let min = Infinity;
  let last = performance.now();
  const end = last + 20;
  while (last < end) {
    const t = performance.now();
    if (t > last) {
      min = Math.min(min, t - last);
      last = t;
    }
  }
  return min;
}
