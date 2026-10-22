import { createStagingBuffer, readBuffer, sharedDevice } from '../../backends/gpu';
import { GpuTimer } from '../../harness/timer';
import { Unsupported, type Implementation } from '../../harness/types';
import naiveWgsl from './naive.wgsl?raw';
import optSubgroupsWgsl from './opt-subgroups.wgsl?raw';
import optTreeWgsl from './opt-tree.wgsl?raw';
import optWgsl from './opt.wgsl?raw';

/** Workgroup size of every reduction shader. */
const WG = 256;

/** One dispatch: it writes `out` values using `groups` workgroups. */
interface Pass {
  out: number;
  groups: number;
}

/**
 * Runs a reduction as a sequence of dispatches in one compute pass (WebGPU makes each
 * dispatch's writes visible to the next). Pass k reads pass k−1's output; intermediate
 * outputs ping-pong between two scratch buffers and the last pass writes the 4-byte result.
 * Buffers and bind groups are kept across iterations at the same size; setup() only uploads.
 */
abstract class GpuReduction implements Implementation<Float32Array, number> {
  abstract readonly id: string;
  readonly tech = 'webgpu';
  private device?: GPUDevice;
  private pipeline?: GPUComputePipeline;
  private timer?: GpuTimer;
  private buffers: GPUBuffer[] = [];
  private data?: GPUBuffer;
  private result?: GPUBuffer;
  private staging?: GPUBuffer;
  /** Input length the buffers were allocated for. */
  private n = -1;
  private dispatches: { bindGroup: GPUBindGroup; x: number; y: number }[] = [];

  /** Floats per element of the shader's `src` array: 1 for array<f32>, 4 for array<vec4f>. */
  protected abstract readonly srcWidth: number;
  protected abstract shader(device: GPUDevice): string;
  /** The passes for n inputs; the last one writes a single value. */
  protected abstract passes(n: number): Pass[];

  async init(): Promise<void> {
    try {
      this.device = await sharedDevice();
    } catch (e) {
      throw new Unsupported(String(e));
    }
    const module = this.device.createShaderModule({ code: this.shader(this.device) });
    this.pipeline = await this.device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    this.timer = new GpuTimer(this.device);
  }

  async setup(input: Float32Array): Promise<void> {
    if (input.length !== this.n) {
      this.release();
      this.allocate(input.length);
    }
    const device = this.device!;
    device.queue.writeBuffer(this.data!, 0, input);
    await device.queue.onSubmittedWorkDone();
  }

  private allocate(n: number): void {
    const device = this.device!;
    const passes = this.passes(n);
    /** Binding size of `len` floats as `src`, rounded up to whole elements. */
    const srcBytes = (len: number) => Math.ceil(len / this.srcWidth) * this.srcWidth * 4;

    // Padded to a multiple of 4 floats for vec4 loads. Buffers start zeroed and nothing
    // writes past n, so the pad reads as 0; so do scratch elements past a pass's output
    // that the next pass's vec4 binding covers (only the two-pass opt reads those, and
    // only pass 0 writes its scratch buffer).
    const data = device.createBuffer({ size: Math.ceil(n / 4) * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const result = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const scratchBytes = [0, 0];
    passes.slice(0, -1).forEach((p, k) => (scratchBytes[k % 2] = Math.max(scratchBytes[k % 2], srcBytes(p.out))));
    const scratch = scratchBytes.map((size) => (size ? device.createBuffer({ size, usage: GPUBufferUsage.STORAGE }) : undefined));
    const staging = createStagingBuffer(device, 4);
    this.buffers = [data, result, staging, ...scratch.filter((b) => b !== undefined)];
    this.data = data;
    this.result = result;
    this.staging = staging;
    this.n = n;

    const maxDim = device.limits.maxComputeWorkgroupsPerDimension;
    let src = data;
    let srcLen = n;
    this.dispatches = passes.map((p, k) => {
      const dst = k === passes.length - 1 ? result : scratch[k % 2]!;
      const bindGroup = device.createBindGroup({
        layout: this.pipeline!.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: src, size: srcBytes(srcLen) } },
          { binding: 1, resource: { buffer: dst, size: p.out * 4 } },
        ],
      });
      src = dst;
      srcLen = p.out;
      const x = Math.min(p.groups, maxDim);
      return { bindGroup, x, y: Math.ceil(p.groups / x) };
    });
  }

  async run(): Promise<void> {
    const device = this.device!;
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass({ timestampWrites: this.timer!.timestampWrites() });
    pass.setPipeline(this.pipeline!);
    for (const d of this.dispatches) {
      pass.setBindGroup(0, d.bindGroup);
      pass.dispatchWorkgroups(d.x, d.y);
    }
    pass.end();
    this.timer!.resolve(encoder);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
  }

  async readback(): Promise<number> {
    return new Float32Array(await readBuffer(this.device!, this.result!, 4, this.staging))[0];
  }

  release(): void {
    for (const buf of this.buffers) buf.destroy();
    this.buffers = [];
    this.dispatches = [];
    this.data = this.result = this.staging = undefined;
    this.n = -1;
  }

  gpuTimeMs(): Promise<number | undefined> {
    return this.timer!.collectMs();
  }

  dispose(): void {
    this.release();
    this.timer?.destroy();
  }
}

/** naive.wgsl: one element per invocation, 256 → 1 per pass, ⌈log₂₅₆ n⌉ passes. */
export class GpuReductionNaive extends GpuReduction {
  readonly id = 'gpu-naive';
  protected readonly srcWidth = 1;

  protected shader(): string {
    return naiveWgsl;
  }

  protected passes(n: number): Pass[] {
    const passes: Pass[] = [];
    let len = n;
    do {
      len = Math.ceil(len / WG);
      passes.push({ out: len, groups: len });
    } while (len > 1);
    return passes;
  }
}

/** Upper bound on the first pass's grid; the second pass then reduces ≤ 1024 partials in one workgroup. */
const MAX_GROUPS = 1024;
/** Each invocation sums at least this many vec4s before another workgroup is added. */
const MIN_VEC4_PER_INVOCATION = 4;

/** opt.wgsl: grid-stride vec4 loads into registers, then one workgroup-level sum; at most two passes. */
export class GpuReductionOpt extends GpuReduction {
  readonly id = 'gpu-opt';
  protected readonly srcWidth = 4;

  protected shader(device: GPUDevice): string {
    return device.features.has('subgroups') ? `enable subgroups;\n${optWgsl}\n${optSubgroupsWgsl}` : `${optWgsl}\n${optTreeWgsl}`;
  }

  protected passes(n: number): Pass[] {
    const groups = Math.min(MAX_GROUPS, Math.ceil(n / 4 / (WG * MIN_VEC4_PER_INVOCATION)));
    return groups === 1 ? [{ out: 1, groups: 1 }] : [{ out: groups, groups }, { out: 1, groups: 1 }];
  }
}
