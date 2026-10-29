import { createStagingBuffer, readBuffer, sharedDevice } from '../../backends/gpu';
import { GpuTimer } from '../../harness/timer';
import { Unsupported, type Implementation } from '../../harness/types';
import type { EditInput } from '.';
import naiveWgsl from './naive.wgsl?raw';
import tiledWgsl from './tiled.wgsl?raw';

/** Writes characters packed four per u32 (as bytes); a partial last word is zero-padded. */
function writeChars(device: GPUDevice, buffer: GPUBuffer, s: Uint8Array): void {
  const whole = s.length & ~3;
  if (whole) device.queue.writeBuffer(buffer, 0, s, 0, whole);
  if (whole < s.length) {
    const tail = new Uint8Array(4);
    tail.set(s.subarray(whole));
    device.queue.writeBuffer(buffer, whole, tail);
  }
}

interface Dispatch {
  /** Dynamic offset of this dispatch's Params slot. */
  offset: number;
  groups: number;
}

/**
 * Shared by both GPU implementations. Both shaders bind the packed strings, two
 * storage buffers and a uniform Params slot selected with a dynamic offset, one slot
 * per dispatch, all dispatches in one compute pass. Buffers are kept across
 * iterations at the same lengths; setup() only uploads the strings, and readback
 * copies the last DP row out of `result`.
 */
abstract class GpuEditDistance implements Implementation<EditInput, Uint32Array> {
  abstract readonly id: string;
  readonly tech = 'webgpu';
  private device?: GPUDevice;
  private pipeline?: GPUComputePipeline;
  private layout?: GPUBindGroupLayout;
  private timer?: GpuTimer;
  private buffers: GPUBuffer[] = [];
  private a?: GPUBuffer;
  private b?: GPUBuffer;
  private staging?: GPUBuffer;
  private bindGroup?: GPUBindGroup;
  private dispatches: Dispatch[] = [];
  /** Buffer and byte offset of the last row, D[la][0..lb]. */
  private result?: GPUBuffer;
  private resultOffset = 0;
  /** Lengths the buffers were allocated for. */
  private la = -1;
  private lb = -1;

  protected abstract readonly wgsl: string;
  /** The two storage buffers' sizes in u32s and initial contents (undefined: zeros). */
  protected abstract storage(la: number, lb: number): [number, Uint32Array | undefined][];
  /** Params (4 × u32) and workgroup count of every dispatch, in order. */
  protected abstract plan(la: number, lb: number): { params: number[]; groups: number }[];
  /** Which storage buffer (0 or 1) holds the last row, and at which u32 index. */
  protected abstract resultAt(la: number, lb: number): [number, number];

  async init(): Promise<void> {
    try {
      this.device = await sharedDevice();
    } catch (e) {
      throw new Unsupported(String(e));
    }
    const storage = (type: GPUBufferBindingType) => ({ visibility: GPUShaderStage.COMPUTE, buffer: { type } });
    this.layout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, ...storage('read-only-storage') },
        { binding: 1, ...storage('read-only-storage') },
        { binding: 2, ...storage('storage') },
        { binding: 3, ...storage('storage') },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true } },
      ],
    });
    this.pipeline = await this.device.createComputePipelineAsync({
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.layout] }),
      compute: { module: this.device.createShaderModule({ code: this.wgsl }), entryPoint: 'main' },
    });
    this.timer = new GpuTimer(this.device);
  }

  async setup(input: EditInput): Promise<void> {
    if (input.a.length !== this.la || input.b.length !== this.lb) {
      this.release();
      this.allocate(input.a.length, input.b.length);
    }
    const device = this.device!;
    writeChars(device, this.a!, input.a);
    writeChars(device, this.b!, input.b);
    await device.queue.onSubmittedWorkDone();
  }

  private allocate(la: number, lb: number): void {
    const device = this.device!;
    const create = (bytes: number, usage: GPUBufferUsageFlags) => {
      const buf = device.createBuffer({ size: Math.max(4, Math.ceil(bytes / 4) * 4), usage });
      this.buffers.push(buf);
      return buf;
    };
    this.a = create(la, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.b = create(lb, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const work = this.storage(la, lb).map(([elements, initial]) => {
      const buf = create(elements * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
      if (initial) device.queue.writeBuffer(buf, 0, initial);
      return buf;
    });
    const [which, index] = this.resultAt(la, lb);
    this.result = work[which];
    this.resultOffset = index * 4;
    this.staging = createStagingBuffer(device, (lb + 1) * 4);
    this.buffers.push(this.staging);

    const stride = device.limits.minUniformBufferOffsetAlignment;
    const plan = this.plan(la, lb);
    const params = create(plan.length * stride, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const data = new Uint32Array((plan.length * stride) / 4);
    plan.forEach((p, s) => data.set(p.params, (s * stride) / 4));
    device.queue.writeBuffer(params, 0, data);
    this.dispatches = plan.map((p, s) => ({ offset: s * stride, groups: p.groups }));
    this.bindGroup = device.createBindGroup({
      layout: this.layout!,
      entries: [
        { binding: 0, resource: { buffer: this.a } },
        { binding: 1, resource: { buffer: this.b } },
        { binding: 2, resource: { buffer: work[0] } },
        { binding: 3, resource: { buffer: work[1] } },
        { binding: 4, resource: { buffer: params, size: 16 } },
      ],
    });
    this.la = la;
    this.lb = lb;
  }

  async run(): Promise<void> {
    const device = this.device!;
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass({ timestampWrites: this.timer!.timestampWrites() });
    pass.setPipeline(this.pipeline!);
    for (const d of this.dispatches) {
      pass.setBindGroup(0, this.bindGroup!, [d.offset]);
      pass.dispatchWorkgroups(d.groups);
    }
    pass.end();
    this.timer!.resolve(encoder);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
  }

  async readback(): Promise<Uint32Array> {
    return new Uint32Array(await readBuffer(this.device!, this.result!, (this.lb + 1) * 4, this.staging, this.resultOffset));
  }

  release(): void {
    for (const buf of this.buffers) buf.destroy();
    this.buffers = [];
    this.a = this.b = this.staging = this.result = this.bindGroup = undefined;
    this.dispatches = [];
    this.la = this.lb = -1;
  }

  gpuTimeMs(): Promise<number | undefined> {
    return this.timer!.collectMs();
  }

  dispose(): void {
    this.release();
    this.timer?.destroy();
  }
}

/** Workgroup size of naive.wgsl; one invocation per cell of an anti-diagonal. */
const NAIVE_WG = 256;

/**
 * naive.wgsl: one dispatch per anti-diagonal of cells (la + lb + 1 of them). Storage:
 * three rotating anti-diagonals of la + 1 cells, and the last row.
 */
export class GpuEditDistanceNaive extends GpuEditDistance {
  readonly id = 'gpu-naive';
  protected readonly wgsl = naiveWgsl;

  protected storage(la: number, lb: number): [number, Uint32Array | undefined][] {
    return [
      [3 * (la + 1), undefined],
      [lb + 1, undefined],
    ];
  }

  protected plan(la: number, lb: number): { params: number[]; groups: number }[] {
    return Array.from({ length: la + lb + 1 }, (_, d) => {
      const lo = Math.max(0, d - lb);
      return { params: [d, lo, la, lb], groups: Math.ceil((Math.min(la, d) - lo + 1) / NAIVE_WG) };
    });
  }

  protected resultAt(): [number, number] {
    return [1, 0];
  }
}

/** Tile side of tiled.wgsl (T), also its workgroup size: one invocation per tile row. */
const TILE = 256;

/**
 * tiled.wgsl: one workgroup per 256 × 256 tile, one dispatch per anti-diagonal of
 * tiles. Storage: the tile-row boundaries `rows` and tile-column boundaries `cols`,
 * whose matrix edges are written here once and never overwritten by the kernel.
 */
export class GpuEditDistanceTiled extends GpuEditDistance {
  readonly id = 'gpu-opt';
  protected readonly wgsl = tiledWgsl;

  protected storage(la: number, lb: number): [number, Uint32Array | undefined][] {
    const nr = Math.ceil(la / TILE);
    const nc = Math.ceil(lb / TILE);
    const rows = new Uint32Array((nr + 1) * (lb + 1));
    for (let j = 0; j <= lb; j++) rows[j] = j;
    for (let r = 1; r <= nr; r++) rows[r * (lb + 1)] = Math.min(r * TILE, la);
    const cols = new Uint32Array((nc + 1) * (la + 1));
    for (let i = 0; i <= la; i++) cols[i] = i;
    for (let c = 1; c <= nc; c++) cols[c * (la + 1)] = Math.min(c * TILE, lb);
    return [
      [rows.length, rows],
      [cols.length, cols],
    ];
  }

  protected plan(la: number, lb: number): { params: number[]; groups: number }[] {
    const nr = Math.ceil(la / TILE);
    const nc = Math.ceil(lb / TILE);
    return Array.from({ length: nr + nc - 1 }, (_, k) => {
      const r0 = Math.max(0, k - nc + 1);
      const r1 = Math.min(k, nr - 1);
      return { params: [la, lb, k, r0], groups: r1 - r0 + 1 };
    });
  }

  protected resultAt(la: number, lb: number): [number, number] {
    return [0, Math.ceil(la / TILE) * (lb + 1)]; // rows[nr]
  }
}
