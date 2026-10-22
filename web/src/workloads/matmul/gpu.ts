import { readBuffer, sharedDevice } from '../../backends/gpu';
import { GpuTimer } from '../../harness/timer';
import { Unsupported, type Implementation } from '../../harness/types';
import type { MatmulInput } from '.';

/** Workgroup edge of both shaders (@workgroup_size(TILE, TILE)); one invocation per element of C. */
const TILE = 16;

/** Runs one of the matmul shaders (naive.wgsl, tiled.wgsl); they share bindings and dispatch shape. */
export class GpuMatmul implements Implementation<MatmulInput, Float32Array> {
  readonly tech = 'webgpu';
  private device?: GPUDevice;
  private pipeline?: GPUComputePipeline;
  private timer?: GpuTimer;
  private params?: GPUBuffer;
  private buffers: GPUBuffer[] = [];
  private c?: GPUBuffer;
  private bindGroup?: GPUBindGroup;
  private n = 0;

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
    const device = this.device!;
    const bytes = input.a.byteLength;
    this.n = input.n;
    const a = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const b = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const c = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.buffers = [a, b, c];
    this.c = c;
    device.queue.writeBuffer(a, 0, input.a);
    device.queue.writeBuffer(b, 0, input.b);
    device.queue.writeBuffer(this.params!, 0, new Uint32Array([input.n, 0, 0, 0]));
    this.bindGroup = device.createBindGroup({
      layout: this.pipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: a } },
        { binding: 1, resource: { buffer: b } },
        { binding: 2, resource: { buffer: c } },
        { binding: 3, resource: { buffer: this.params! } },
      ],
    });
    await device.queue.onSubmittedWorkDone();
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
    return new Float32Array(await readBuffer(this.device!, this.c!, this.n * this.n * 4));
  }

  teardown(): void {
    for (const buf of this.buffers) buf.destroy();
    this.buffers = [];
    this.c = this.bindGroup = undefined;
  }

  gpuTimeMs(): Promise<number | undefined> {
    return this.timer!.collectMs();
  }

  dispose(): void {
    this.timer?.destroy();
    this.params?.destroy();
  }
}
