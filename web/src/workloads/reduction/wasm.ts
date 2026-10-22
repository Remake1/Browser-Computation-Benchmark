import { detectWasmFeatures, loadWasm, variantSupported, wasmThreadCount, type EmscriptenModule, type WasmVariant } from '../../backends/wasm';
import { Unsupported, type Implementation } from '../../harness/types';

interface ReductionModule extends EmscriptenModule {
  _reduction_init(threads: number): number;
  _reduction_shutdown(): void;
  _reduction(x: number, n: number): number;
}

export class WasmReduction implements Implementation<Float32Array, number> {
  readonly id: string;
  readonly tech = 'wasm';
  private m?: ReductionModule;
  private n = 0;
  private x = 0;
  private result = 0;

  constructor(private readonly variant: WasmVariant) {
    this.id = `wasm-${variant}`;
  }

  async init(): Promise<void> {
    if (!variantSupported(this.variant, await detectWasmFeatures())) throw new Unsupported(`browser lacks features for ${this.id}`);
    this.m = await loadWasm<ReductionModule>('reduction', this.variant);
    const want = this.variant === 'threads' ? wasmThreadCount() : 1;
    const got = this.m._reduction_init(want);
    if (got !== want) throw new Error(`reduction_init(${want}) returned ${got}`);
  }

  async setup(input: Float32Array): Promise<void> {
    const m = this.m!;
    this.n = input.length;
    this.x = m._malloc(input.byteLength);
    if (!this.x) throw new Error(`malloc(${input.byteLength}) failed`);
    m.HEAPF32.set(input, this.x >> 2); // read HEAPF32 after malloc: growth replaces the view
  }

  async run(): Promise<void> {
    this.result = this.m!._reduction(this.x, this.n);
  }

  async readback(): Promise<number> {
    return this.result;
  }

  teardown(): void {
    if (this.x) this.m!._free(this.x);
    this.x = 0;
  }

  dispose(): void {
    this.m?._reduction_shutdown();
    this.m = undefined;
  }
}
