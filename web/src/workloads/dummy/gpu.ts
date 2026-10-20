import { readBuffer, sharedDevice } from '../../backends/gpu';
import { GpuTimer } from '../../harness/timer';
import { Unsupported, type Implementation } from '../../harness/types';
import sumWgsl from './sum.wgsl?raw';

export class GpuSum implements Implementation<Float32Array, number> {
  readonly id = 'gpu-naive';
  readonly tech = 'webgpu';
  private device?: GPUDevice;
  private pipeline?: GPUComputePipeline;
  private timer?: GpuTimer;
  private data?: GPUBuffer;
  private result?: GPUBuffer;
  private bindGroup?: GPUBindGroup;

  async init(): Promise<void> {
    try {
      this.device = await sharedDevice();
    } catch (e) {
      throw new Unsupported(String(e));
    }
    const module = this.device.createShaderModule({ code: sumWgsl });
    this.pipeline = await this.device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    this.timer = new GpuTimer(this.device);
  }

  async setup(input: Float32Array): Promise<void> {
    const device = this.device!;
    this.data = device.createBuffer({ size: input.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.result = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    device.queue.writeBuffer(this.data, 0, input);
    this.bindGroup = device.createBindGroup({
      layout: this.pipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.data } },
        { binding: 1, resource: { buffer: this.result } },
      ],
    });
    await device.queue.onSubmittedWorkDone();
  }

  async run(): Promise<void> {
    const device = this.device!;
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass({ timestampWrites: this.timer!.timestampWrites() });
    pass.setPipeline(this.pipeline!);
    pass.setBindGroup(0, this.bindGroup!);
    pass.dispatchWorkgroups(1);
    pass.end();
    this.timer!.resolve(encoder);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
  }

  async readback(): Promise<number> {
    return new Float32Array(await readBuffer(this.device!, this.result!, 4))[0];
  }

  teardown(): void {
    this.data?.destroy();
    this.result?.destroy();
    this.data = this.result = this.bindGroup = undefined;
  }

  gpuTimeMs(): Promise<number | undefined> {
    return this.timer!.collectMs();
  }

  dispose(): void {
    this.timer?.destroy();
  }
}
