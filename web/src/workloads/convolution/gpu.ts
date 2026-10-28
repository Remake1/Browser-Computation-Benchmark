import { createStagingBuffer, readBuffer, sharedDevice } from '../../backends/gpu';
import { GpuTimer } from '../../harness/timer';
import { Unsupported, type Implementation } from '../../harness/types';
import type { ConvolutionInput } from '.';

/** Workgroup edge of both shaders (@workgroup_size(TILE, TILE)); one invocation per output pixel. */
const TILE = 16;

/**
 * Runs one of the convolution shaders (naive.wgsl, tiled.wgsl); they share bindings and dispatch shape.
 * Buffers and the bind group are kept across iterations at the same W; setup() only uploads.
 */
export class GpuConvolution implements Implementation<ConvolutionInput, Float32Array> {
  readonly tech = 'webgpu';
  private device?: GPUDevice;
  private pipeline?: GPUComputePipeline;
  private timer?: GpuTimer;
  private params?: GPUBuffer;
  private buffers: GPUBuffer[] = [];
  private src?: GPUBuffer;
  private dst?: GPUBuffer;
  private staging?: GPUBuffer;
  private bindGroup?: GPUBindGroup;
  /** W the buffers were allocated for. */
  private w = -1;

  constructor(
    readonly id: string,
    private readonly wgsl: string,
  ) {}

  async init(): Promise<void> {
    try {
      this.device = await sharedDevice();
    } catch (e) {
      throw new Unsupported(String(e));
    }
    const module = this.device.createShaderModule({ code: this.wgsl });
    this.pipeline = await this.device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    this.timer = new GpuTimer(this.device);
    this.params = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  async setup(input: ConvolutionInput): Promise<void> {
    if (input.w !== this.w) {
      this.release();
      this.allocate(input.w);
    }
    const device = this.device!;
    device.queue.writeBuffer(this.src!, 0, input.image);
    await device.queue.onSubmittedWorkDone();
  }

  private allocate(w: number): void {
    const device = this.device!;
    const bytes = w * w * 4;
    const src = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const dst = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const staging = createStagingBuffer(device, bytes);
    this.buffers = [src, dst, staging];
    this.src = src;
    this.dst = dst;
    this.staging = staging;
    this.w = w;
    device.queue.writeBuffer(this.params!, 0, new Uint32Array([w, 0, 0, 0]));
    this.bindGroup = device.createBindGroup({
      layout: this.pipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: src } },
        { binding: 1, resource: { buffer: dst } },
        { binding: 2, resource: { buffer: this.params! } },
      ],
    });
  }

  async run(): Promise<void> {
    const device = this.device!;
    const groups = Math.ceil(this.w / TILE);
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass({ timestampWrites: this.timer!.timestampWrites() });
    pass.setPipeline(this.pipeline!);
    pass.setBindGroup(0, this.bindGroup!);
    pass.dispatchWorkgroups(groups, groups);
    pass.end();
    this.timer!.resolve(encoder);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
  }

  async readback(): Promise<Float32Array> {
    return new Float32Array(await readBuffer(this.device!, this.dst!, this.w * this.w * 4, this.staging));
  }

  release(): void {
    for (const buf of this.buffers) buf.destroy();
    this.buffers = [];
    this.src = this.dst = this.staging = this.bindGroup = undefined;
    this.w = -1;
  }

  gpuTimeMs(): Promise<number | undefined> {
    return this.timer!.collectMs();
  }

  dispose(): void {
    this.release();
    this.timer?.destroy();
    this.params?.destroy();
  }
}
