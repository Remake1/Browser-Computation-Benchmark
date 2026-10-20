// CPU and GPU timing (PLAN.md §5).
//
// CPU side: performance.now() around each phase. Under cross-origin isolation its
// resolution is ~5–20 µs depending on the browser, which is why the runner batches
// short iterations.
//
// GPU side: timestamp-query, when the device has it. Implementations put
// `timer.timestampWrites()` on their compute pass(es) and call `timer.resolve(encoder)`
// before submitting; results are mapped later, outside the timed region.

export async function timed<T>(fn: () => Promise<T>): Promise<[T, number]> {
  const t0 = performance.now();
  const value = await fn();
  return [value, performance.now() - t0];
}

const QUERY_BYTES = 16; // two u64 timestamps

export class GpuTimer {
  readonly enabled: boolean;
  private readonly querySet?: GPUQuerySet;
  private readonly resolveBuffer?: GPUBuffer;
  private readonly free: GPUBuffer[] = [];
  private pending: GPUBuffer[] = [];

  constructor(private readonly device: GPUDevice) {
    this.enabled = device.features.has('timestamp-query');
    if (!this.enabled) return;
    this.querySet = device.createQuerySet({ type: 'timestamp', count: 2 });
    this.resolveBuffer = device.createBuffer({ size: QUERY_BYTES, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  }

  /**
   * For a single pass use the default; for multi-pass kernels put `{ begin: true }` on
   * the first pass and `{ end: true }` on the last, so the span covers all of them.
   */
  timestampWrites(which: { begin?: boolean; end?: boolean } = { begin: true, end: true }): GPUComputePassTimestampWrites | undefined {
    if (!this.querySet) return undefined;
    return {
      querySet: this.querySet,
      beginningOfPassWriteIndex: which.begin ? 0 : undefined,
      endOfPassWriteIndex: which.end ? 1 : undefined,
    };
  }

  /** Records the copy of this submission's timestamps into a mappable buffer. */
  resolve(encoder: GPUCommandEncoder): void {
    if (!this.querySet || !this.resolveBuffer) return;
    encoder.resolveQuerySet(this.querySet, 0, 2, this.resolveBuffer, 0);
    const staging =
      this.free.pop() ?? this.device.createBuffer({ size: QUERY_BYTES, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    encoder.copyBufferToBuffer(this.resolveBuffer, 0, staging, 0, QUERY_BYTES);
    this.pending.push(staging);
  }

  /** Total GPU time of every resolve() since the last call, in ms; undefined if none or if any span is invalid. */
  async collectMs(): Promise<number | undefined> {
    const buffers = this.pending;
    this.pending = [];
    if (buffers.length === 0) return undefined;
    await Promise.all(buffers.map((b) => b.mapAsync(GPUMapMode.READ)));
    let totalNs = 0n;
    let valid = true;
    for (const b of buffers) {
      const [begin, end] = new BigUint64Array(b.getMappedRange());
      if (end < begin) valid = false;
      else totalNs += end - begin;
      b.unmap();
      this.free.push(b);
    }
    return valid ? Number(totalNs) / 1e6 : undefined;
  }

  destroy(): void {
    this.querySet?.destroy();
    this.resolveBuffer?.destroy();
    for (const b of [...this.free, ...this.pending]) b.destroy();
  }
}

const NOOP_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> sink: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) { sink[id.x] = id.x; }`;

/**
 * Browsers may quantize timestamp-query values. Writes timestamps around a few tiny
 * dispatches and returns the GCD of all their offsets from the first one, in ns
 * (1 means no visible quantization). Undefined without timestamp-query.
 */
export async function measureTimestampResolutionNs(device: GPUDevice): Promise<number | undefined> {
  if (!device.features.has('timestamp-query')) return undefined;
  const passes = 16;
  const querySet = device.createQuerySet({ type: 'timestamp', count: passes * 2 });
  const resolve = device.createBuffer({ size: passes * QUERY_BYTES, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const staging = device.createBuffer({ size: passes * QUERY_BYTES, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const sink = device.createBuffer({ size: 64 * 4, usage: GPUBufferUsage.STORAGE });
  try {
    const module = device.createShaderModule({ code: NOOP_WGSL });
    const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: sink } }] });
    const encoder = device.createCommandEncoder();
    for (let i = 0; i < passes; i++) {
      const pass = encoder.beginComputePass({ timestampWrites: { querySet, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 } });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(1);
      pass.end();
    }
    encoder.resolveQuerySet(querySet, 0, passes * 2, resolve, 0);
    encoder.copyBufferToBuffer(resolve, 0, staging, 0, passes * QUERY_BYTES);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const ts = [...new BigUint64Array(staging.getMappedRange())];
    staging.unmap();
    let g = 0n;
    for (const t of ts) g = gcd(g, t > ts[0] ? t - ts[0] : ts[0] - t);
    return g === 0n ? undefined : Number(g);
  } finally {
    querySet.destroy();
    resolve.destroy();
    staging.destroy();
    sink.destroy();
  }
}

function gcd(a: bigint, b: bigint): bigint {
  while (b) [a, b] = [b, a % b];
  return a;
}
