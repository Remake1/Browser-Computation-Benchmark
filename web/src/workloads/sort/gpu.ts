import { createStagingBuffer, readBuffer, sharedDevice } from '../../backends/gpu';
import { GpuTimer } from '../../harness/timer';
import { Unsupported, type Implementation } from '../../harness/types';
import bitonicWgsl from './bitonic.wgsl?raw';
import countWgsl from './radix-count.wgsl?raw';
import scanWgsl from './radix-scan.wgsl?raw';
import scatterWgsl from './radix-scatter.wgsl?raw';

/** Pad value: sorts to the end, so the first n sorted keys are the result. */
const PAD = 0xffffffff;

/** 1D grid of `groups` workgroups, or 2D above the per-dimension limit (shaders skip the spare ones). */
function grid(device: GPUDevice, groups: number): [number, number] {
  const x = Math.min(groups, device.limits.maxComputeWorkgroupsPerDimension);
  return [x, Math.ceil(groups / x)];
}

/**
 * Shared by both GPU sorts. The kernel sorts a padded key buffer in place; buffers are
 * kept across iterations at the same n, setup() only uploads the n keys (the padding,
 * written once at allocation, stays at the end after every sort), and readback copies
 * the first n keys. All dispatches of a sort go in one compute pass.
 */
abstract class GpuSort implements Implementation<Uint32Array, Uint32Array> {
  abstract readonly id: string;
  readonly tech = 'webgpu';
  protected device?: GPUDevice;
  private timer?: GpuTimer;
  protected buffers: GPUBuffer[] = [];
  /** The buffer that holds the keys before and after run(). */
  private keys?: GPUBuffer;
  private staging?: GPUBuffer;
  /** Input length the buffers were allocated for. */
  private n = -1;

  protected abstract createPipelines(device: GPUDevice): Promise<void>;
  /** Keys padded to this length. */
  protected abstract paddedLength(n: number): number;
  /** Allocates everything besides the keys and staging buffers (push them onto `buffers`). */
  protected abstract allocateKernel(keys: GPUBuffer, padded: number): void;
  protected abstract encode(pass: GPUComputePassEncoder): void;
  protected abstract releaseKernel(): void;

  async init(): Promise<void> {
    try {
      this.device = await sharedDevice();
    } catch (e) {
      throw new Unsupported(String(e));
    }
    await this.createPipelines(this.device);
    this.timer = new GpuTimer(this.device);
  }

  async setup(input: Uint32Array): Promise<void> {
    if (input.length !== this.n) {
      this.release();
      this.allocate(input.length);
    }
    const device = this.device!;
    device.queue.writeBuffer(this.keys!, 0, input);
    await device.queue.onSubmittedWorkDone();
  }

  private allocate(n: number): void {
    const device = this.device!;
    const padded = this.paddedLength(n);
    const keys = device.createBuffer({ size: padded * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    if (padded > n) device.queue.writeBuffer(keys, n * 4, new Uint32Array(padded - n).fill(PAD));
    const staging = createStagingBuffer(device, n * 4);
    this.buffers = [keys, staging];
    this.keys = keys;
    this.staging = staging;
    this.n = n;
    this.allocateKernel(keys, padded);
  }

  async run(): Promise<void> {
    const device = this.device!;
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass({ timestampWrites: this.timer!.timestampWrites() });
    this.encode(pass);
    pass.end();
    this.timer!.resolve(encoder);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
  }

  async readback(): Promise<Uint32Array> {
    return new Uint32Array(await readBuffer(this.device!, this.keys!, this.n * 4, this.staging));
  }

  release(): void {
    for (const buf of this.buffers) buf.destroy();
    this.buffers = [];
    this.keys = this.staging = undefined;
    this.n = -1;
    this.releaseKernel();
  }

  gpuTimeMs(): Promise<number | undefined> {
    return this.timer!.collectMs();
  }

  dispose(): void {
    this.release();
    this.timer?.destroy();
  }
}

/** Workgroup size of bitonic.wgsl; each invocation handles one pair. */
const BITONIC_WG = 256;

/**
 * bitonic.wgsl: global-memory bitonic sort of the keys padded to a power of two,
 * one dispatch per (k, j) step. Each step's (k, j) is a 256-byte slot of one uniform
 * buffer, selected with a dynamic offset.
 */
export class GpuSortBitonic extends GpuSort {
  readonly id = 'gpu-naive';
  private pipeline?: GPUComputePipeline;
  private layout?: GPUBindGroupLayout;
  private bindGroup?: GPUBindGroup;
  private steps = 0;
  private stride = 0;
  private groups: [number, number] = [0, 0];

  protected async createPipelines(device: GPUDevice): Promise<void> {
    this.layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true } },
      ],
    });
    this.pipeline = await device.createComputePipelineAsync({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.layout] }),
      compute: { module: device.createShaderModule({ code: bitonicWgsl }), entryPoint: 'main' },
    });
    this.stride = device.limits.minUniformBufferOffsetAlignment;
  }

  protected paddedLength(n: number): number {
    return 2 ** Math.ceil(Math.log2(Math.max(n, 1)));
  }

  protected allocateKernel(keys: GPUBuffer, padded: number): void {
    const device = this.device!;
    const slots = this.stride / 4;
    const steps: number[] = [];
    for (let k = 2; k <= padded; k *= 2) for (let j = k / 2; j >= 1; j /= 2) steps.push(k, j);
    this.steps = steps.length / 2;
    if (this.steps === 0) return; // n = 1
    const params = device.createBuffer({ size: this.steps * this.stride, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const data = new Uint32Array(this.steps * slots);
    for (let s = 0; s < this.steps; s++) data.set([steps[2 * s], steps[2 * s + 1]], s * slots);
    device.queue.writeBuffer(params, 0, data);
    this.buffers.push(params);
    this.bindGroup = device.createBindGroup({
      layout: this.layout!,
      entries: [
        { binding: 0, resource: { buffer: keys } },
        { binding: 1, resource: { buffer: params, size: 16 } },
      ],
    });
    this.groups = grid(device, Math.ceil(padded / 2 / BITONIC_WG));
  }

  protected encode(pass: GPUComputePassEncoder): void {
    if (this.steps === 0) return;
    pass.setPipeline(this.pipeline!);
    for (let s = 0; s < this.steps; s++) {
      pass.setBindGroup(0, this.bindGroup!, [s * this.stride]);
      pass.dispatchWorkgroups(...this.groups);
    }
  }

  protected releaseKernel(): void {
    this.bindGroup = undefined;
    this.steps = 0;
  }
}

/** Keys per workgroup of the radix shaders: 256 invocations × one vec4u. */
const BLOCK = 1024;
/** Elements per workgroup of radix-scan.wgsl. */
const SCAN_BLOCK = 1024;
const RADIX = 16;
const PASSES = 8;

interface Dispatch {
  pipeline: GPUComputePipeline;
  bindGroup: GPUBindGroup;
  groups: [number, number];
}

/**
 * radix-count/scan/scatter.wgsl: LSD radix sort, 4-bit digits, 8 passes of
 * count → scan → scatter ping-ponging between the keys buffer and a scratch buffer
 * (radix-scatter.wgsl describes the algorithm). Keys are padded to a whole block.
 */
export class GpuSortRadix extends GpuSort {
  readonly id = 'gpu-opt';
  private count?: GPUComputePipeline;
  private scan?: GPUComputePipeline;
  private add?: GPUComputePipeline;
  private scatter?: GPUComputePipeline;
  private dispatches: Dispatch[] = [];

  protected async createPipelines(device: GPUDevice): Promise<void> {
    const scanModule = device.createShaderModule({ code: scanWgsl });
    const compile = (module: GPUShaderModule, entryPoint: string) => device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint } });
    [this.count, this.scan, this.add, this.scatter] = await Promise.all([
      compile(device.createShaderModule({ code: countWgsl }), 'main'),
      compile(scanModule, 'scan'),
      compile(scanModule, 'add'),
      compile(device.createShaderModule({ code: scatterWgsl }), 'main'),
    ]);
  }

  protected paddedLength(n: number): number {
    return Math.ceil(Math.max(n, 1) / BLOCK) * BLOCK;
  }

  protected allocateKernel(keys: GPUBuffer, padded: number): void {
    const device = this.device!;
    const blocks = padded / BLOCK;
    const storage = (elements: number) => {
      const buf = device.createBuffer({ size: elements * 4, usage: GPUBufferUsage.STORAGE });
      this.buffers.push(buf);
      return buf;
    };
    const scratch = storage(padded);

    // Scan levels: level 0 is the histogram, level i + 1 holds level i's block totals.
    const lengths = [RADIX * blocks];
    while (lengths[lengths.length - 1] > SCAN_BLOCK) lengths.push(Math.ceil(lengths[lengths.length - 1] / SCAN_BLOCK));
    const levels = lengths.map(storage);
    const dummy = storage(1);
    const hist = levels[0];

    const stride = device.limits.minUniformBufferOffsetAlignment;
    const params = device.createBuffer({ size: PASSES * stride, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.buffers.push(params);
    for (let p = 0; p < PASSES; p++) device.queue.writeBuffer(params, p * stride, new Uint32Array([4 * p, blocks, 0, 0]));

    const bind = (pipeline: GPUComputePipeline, buffers: GPUBufferBinding[]) =>
      device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: buffers.map((resource, binding) => ({ binding, resource })) });
    const scanDispatches: Dispatch[] = [];
    levels.forEach((data, i) =>
      scanDispatches.push({
        pipeline: this.scan!,
        bindGroup: bind(this.scan!, [{ buffer: data }, { buffer: levels[i + 1] ?? dummy }]),
        groups: grid(device, Math.ceil(lengths[i] / SCAN_BLOCK)),
      }),
    );
    for (let i = levels.length - 2; i >= 0; i--) {
      scanDispatches.push({
        pipeline: this.add!,
        bindGroup: bind(this.add!, [{ buffer: levels[i] }, { buffer: levels[i + 1] }]),
        groups: grid(device, Math.ceil(lengths[i] / SCAN_BLOCK)),
      });
    }

    const groups = grid(device, blocks);
    this.dispatches = [];
    for (let p = 0; p < PASSES; p++) {
      const [src, dst] = p % 2 === 0 ? [keys, scratch] : [scratch, keys];
      const param = { buffer: params, offset: p * stride, size: 16 };
      this.dispatches.push({ pipeline: this.count!, bindGroup: bind(this.count!, [{ buffer: src }, { buffer: hist }, param]), groups });
      this.dispatches.push(...scanDispatches);
      this.dispatches.push({
        pipeline: this.scatter!,
        bindGroup: bind(this.scatter!, [{ buffer: src }, { buffer: dst }, { buffer: hist }, param]),
        groups,
      });
    }
  }

  protected encode(pass: GPUComputePassEncoder): void {
    for (const d of this.dispatches) {
      pass.setPipeline(d.pipeline);
      pass.setBindGroup(0, d.bindGroup);
      pass.dispatchWorkgroups(...d.groups);
    }
  }

  protected releaseKernel(): void {
    this.dispatches = [];
  }
}
