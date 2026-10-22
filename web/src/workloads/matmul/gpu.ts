import { createStagingBuffer, readBuffer, sharedDevice } from '../../backends/gpu';
import { GpuTimer } from '../../harness/timer';
import { Unsupported, type Implementation } from '../../harness/types';
import type { MatmulInput } from '.';

/** Workgroup edge of both shaders (@workgroup_size(TILE, TILE)); one invocation per element of C. */
const TILE = 16;

/**
 * Runs one of the matmul shaders (naive.wgsl, tiled.wgsl); they share bindings and dispatch shape.
 * Buffers and the bind group are kept across iterations at the same N; setup() only uploads.
 */
export class GpuMatmul implements Implementation<MatmulInput, Float32Array> {
  readonly tech = 'webgpu';
  private device?: GPUDevice;
  private pipeline?: GPUComputePipeline;
  private timer?: GpuTimer;
  private params?: GPUBuffer;
  private buffers: GPUBuffer[] = [];
  private a?: GPUBuffer;
  private b?: GPUBuffer;
  private c?: GPUBuffer;
  private staging?: GPUBuffer;
  private bindGroup?: GPUBindGroup;
  /** N the buffers were allocated for. */
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

  async setup(input: MatmulInput): Promise<void> {
    if (input.n !== this.n) {
      this.release();
      this.allocate(input.n);
    }
    const device = this.device!;
    device.queue.writeBuffer(this.a!, 0, input.a);
    device.queue.writeBuffer(this.b!, 0, input.b);
    await device.queue.onSubmittedWorkDone();
  }

  private allocate(n: number): void {
    const device = this.device!;
    const bytes = n * n * 4;
    const a = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const b = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const c = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const staging = createStagingBuffer(device, bytes);
    this.buffers = [a, b, c, staging];
    this.a = a;
    this.b = b;
    this.c = c;
    this.staging = staging;
    this.n = n;
    device.queue.writeBuffer(this.params!, 0, new Uint32Array([n, 0, 0, 0]));
    this.bindGroup = device.createBindGroup({
      layout: this.pipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: a } },
        { binding: 1, resource: { buffer: b } },
        { binding: 2, resource: { buffer: c } },
        { binding: 3, resource: { buffer: this.params! } },
      ],
    });
  }

  async run(): Promise<void> {
    const device = this.device!;
    const groups = Math.ceil(this.n / TILE);
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
    return new Float32Array(await readBuffer(this.device!, this.c!, this.n * this.n * 4, this.staging));
  }

  release(): void {
    for (const buf of this.buffers) buf.destroy();
    this.buffers = [];
    this.a = this.b = this.c = this.staging = this.bindGroup = undefined;
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
