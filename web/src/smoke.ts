// Milestone 0 smoke tests, run inside the benchmark worker: load every "hello"
// WASM variant and check it got the features its flags promise, and run one tiny
// WebGPU compute dispatch end to end.
import { readBuffer, requestAdapter, requestDevice } from './backends/gpu';
import {
  loadWasm,
  variantSupported,
  wasmThreadCount,
  WASM_VARIANTS,
  type EmscriptenModule,
  type WasmFeatures,
  type WasmVariant,
} from './backends/wasm';

export interface SmokeResult {
  id: string;
  status: 'pass' | 'fail' | 'unsupported';
  detail: string;
  /** Instantiation / device+pipeline creation time, ms. */
  initMs?: number;
}

interface HelloModule extends EmscriptenModule {
  _hello_variant(): number;
  _hello_simd_enabled(): number;
  _hello_threads_enabled(): number;
  _hello_sum(ptr: number, n: number): number;
  _hello_parallel(n: number): number;
}

export async function runSmokeTests(features: WasmFeatures): Promise<SmokeResult[]> {
  const results: SmokeResult[] = [];
  for (const variant of WASM_VARIANTS) results.push(await smokeWasm(variant, features));
  results.push(await smokeGpu());
  return results;
}

/** Same 4-accumulator order as hello_sum in wasm/src/hello.cpp, in f32, so results match bit for bit. */
function sumReference(x: Float32Array): number {
  const f = Math.fround;
  let a0 = 0, a1 = 0, a2 = 0, a3 = 0, i = 0;
  for (; i + 4 <= x.length; i += 4) {
    a0 = f(a0 + x[i]);
    a1 = f(a1 + x[i + 1]);
    a2 = f(a2 + x[i + 2]);
    a3 = f(a3 + x[i + 3]);
  }
  let s = f(f(a0 + a1) + f(a2 + a3));
  for (; i < x.length; i++) s = f(s + x[i]);
  return s;
}

async function smokeWasm(variant: WasmVariant, features: WasmFeatures): Promise<SmokeResult> {
  const id = `wasm-${variant}`;
  if (!variantSupported(variant, features)) {
    return { id, status: 'unsupported', detail: `browser lacks ${variant === 'simd' ? 'SIMD' : 'SIMD and/or threads'}` };
  }
  try {
    const t0 = performance.now();
    const m = await loadWasm<HelloModule>('hello', variant);
    const initMs = performance.now() - t0;

    const problems: string[] = [];
    const reported = m.UTF8ToString(m._hello_variant());
    if (reported !== variant) problems.push(`module reports variant "${reported}"`);
    if (m._hello_simd_enabled() !== (variant === 'scalar' ? 0 : 1)) problems.push('unexpected __wasm_simd128__');
    if (m._hello_threads_enabled() !== (variant === 'threads' ? 1 : 0)) problems.push('unexpected pthreads');

    const x = Float32Array.from({ length: 1003 }, (_, i) => Math.sin(i)); // odd length exercises the tail loop
    const ptr = m._malloc(x.byteLength);
    m.HEAPF32.set(x, ptr >> 2);
    const got = m._hello_sum(ptr, x.length);
    m._free(ptr);
    const want = sumReference(x);
    if (got !== want) problems.push(`hello_sum = ${got}, expected ${want}`);

    const threads = variant === 'threads' ? wasmThreadCount() : 1;
    const ran = m._hello_parallel(threads);
    if (ran !== threads) problems.push(`hello_parallel ran ${ran}/${threads} threads`);

    return problems.length
      ? { id, status: 'fail', detail: problems.join('; '), initMs }
      : { id, status: 'pass', detail: `sum matches JS bit-exactly; ${threads} thread(s) ran`, initMs };
  } catch (e) {
    return { id, status: 'fail', detail: `load/run threw: ${String(e)}` };
  }
}

const DOUBLE_PLUS_ONE_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> data: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x < arrayLength(&data)) {
    data[id.x] = data[id.x] * 2u + 1u;
  }
}`;

async function smokeGpu(): Promise<SmokeResult> {
  const id = 'webgpu-compute';
  const adapter = await requestAdapter().catch(() => null);
  if (!adapter) return { id, status: 'unsupported', detail: 'no WebGPU adapter in this context' };
  let device: GPUDevice | undefined;
  try {
    const t0 = performance.now();
    device = await requestDevice(adapter);
    device.pushErrorScope('validation');
    const module = device.createShaderModule({ code: DOUBLE_PLUS_ONE_WGSL });
    const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    const initMs = performance.now() - t0;

    const input = Uint32Array.from({ length: 1000 }, (_, i) => i);
    const buffer = device.createBuffer({
      size: input.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buffer, 0, input);
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer } }],
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(input.length / 64));
    pass.end();
    device.queue.submit([encoder.finish()]);

    const out = new Uint32Array(await readBuffer(device, buffer, input.byteLength));
    const error = await device.popErrorScope();
    if (error) return { id, status: 'fail', detail: `validation error: ${error.message}`, initMs };
    const bad = out.findIndex((v, i) => v !== input[i] * 2 + 1);
    if (bad >= 0) return { id, status: 'fail', detail: `out[${bad}] = ${out[bad]}, expected ${input[bad] * 2 + 1}`, initMs };

    const ts = device.features.has('timestamp-query') ? 'enabled' : 'unavailable';
    return { id, status: 'pass', detail: `${input.length}-element dispatch verified; timestamp-query ${ts}`, initMs };
  } catch (e) {
    return { id, status: 'fail', detail: `threw: ${String(e)}` };
  } finally {
    device?.destroy();
  }
}
