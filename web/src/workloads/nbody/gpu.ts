import { createStagingBuffer, readBuffer, sharedDevice } from '../../backends/gpu';
import { GpuTimer } from '../../harness/timer';
import { Unsupported, type Implementation } from '../../harness/types';

/** Workgroup size of both shaders (@workgroup_size(WG)); one invocation per body. */
const WG = 128;

/**
 * Runs one of the N-body shaders (naive.wgsl, tiled.wgsl); they share bindings and dispatch shape.
 * Buffers and the bind group are kept across iterations at the same n; setup() only uploads.
 */
export class GpuNbody implements Implementation<Float32Array, Float32Array> {
  readonly tech = 'webgpu';
  private device?: GPUDevice;
  private pipeline?: GPUComputePipeline;
  private timer?: GpuTimer;
  private params?: GPUBuffer;
  private buffers: GPUBuffer[] = [];
  private bodies?: GPUBuffer;
  private acc?: GPUBuffer;
  private staging?: GPUBuffer;
  private bindGroup?: GPUBindGroup;
  /** Body count the buffers were allocated for. */
  private n = -1;

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

  async setup(input: Float32Array): Promise<void> {
    const n = input.length >> 2;
    if (n !== this.n) {
      this.release();
      this.allocate(n);
    }
    const device = this.device!;
    device.queue.writeBuffer(this.bodies!, 0, input);
    await device.queue.onSubmittedWorkDone();
  }

  private allocate(n: number): void {
    const device = this.device!;
    const bytes = n * 16;
    const bodies = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const acc = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const staging = createStagingBuffer(device, bytes);
    this.buffers = [bodies, acc, staging];
    this.bodies = bodies;
    this.acc = acc;
    this.staging = staging;
    this.n = n;
    device.queue.writeBuffer(this.params!, 0, new Uint32Array([n, 0, 0, 0]));
    this.bindGroup = device.createBindGroup({
      layout: this.pipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bodies } },
        { binding: 1, resource: { buffer: acc } },
        { binding: 2, resource: { buffer: this.params! } },
      ],
    });
  }

  async run(): Promise<void> {
    const device = this.device!;
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass({ timestampWrites: this.timer!.timestampWrites() });
    pass.setPipeline(this.pipeline!);
    pass.setBindGroup(0, this.bindGroup!);
    pass.dispatchWorkgroups(Math.ceil(this.n / WG));
    pass.end();
    this.timer!.resolve(encoder);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
  }

  async readback(): Promise<Float32Array> {
    return new Float32Array(await readBuffer(this.device!, this.acc!, this.n * 16, this.staging));
  }

  release(): void {
    for (const buf of this.buffers) buf.destroy();
    this.buffers = [];
    this.bodies = this.acc = this.staging = this.bindGroup = undefined;
    this.n = -1;
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
