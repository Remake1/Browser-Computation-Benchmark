import { detectWasmFeatures, loadWasm, variantSupported, wasmThreadCount, type EmscriptenModule, type WasmVariant } from '../../backends/wasm';
import { Unsupported, type Implementation } from '../../harness/types';
import type { ConvolutionInput } from '.';

interface ConvolutionModule extends EmscriptenModule {
  _convolution_init(threads: number): number;
  _convolution_shutdown(): void;
  _convolution(src: number, dst: number, w: number): void;
}

export class WasmConvolution implements Implementation<ConvolutionInput, Float32Array> {
  readonly id: string;
  readonly tech = 'wasm';
  private m?: ConvolutionModule;
  private w = 0;
  private src = 0;
  private dst = 0;

  constructor(private readonly variant: WasmVariant) {
    this.id = `wasm-${variant}`;
  }

  async init(): Promise<void> {
    if (!variantSupported(this.variant, await detectWasmFeatures())) throw new Unsupported(`browser lacks features for ${this.id}`);
    this.m = await loadWasm<ConvolutionModule>('convolution', this.variant);
    const want = this.variant === 'threads' ? wasmThreadCount() : 1;
    const got = this.m._convolution_init(want);
    if (got !== want) throw new Error(`convolution_init(${want}) returned ${got}`);
  }

  async setup(input: ConvolutionInput): Promise<void> {
    const m = this.m!;
    const bytes = input.image.byteLength;
    this.w = input.w;
    this.src = m._malloc(bytes);
    this.dst = m._malloc(bytes);
    if (!this.src || !this.dst) throw new Error(`malloc(${bytes}) failed`);
    m.HEAPF32.set(input.image, this.src >> 2); // read HEAPF32 after malloc: growth replaces the view
  }

  async run(): Promise<void> {
    this.m!._convolution(this.src, this.dst, this.w);
  }

  async readback(): Promise<Float32Array> {
    const start = this.dst >> 2;
    return this.m!.HEAPF32.slice(start, start + this.w * this.w);
  }

  teardown(): void {
    const m = this.m!;
    for (const ptr of [this.src, this.dst]) if (ptr) m._free(ptr);
    this.src = this.dst = 0;
  }

  dispose(): void {
    this.m?._convolution_shutdown();
    this.m = undefined;
  }
}
