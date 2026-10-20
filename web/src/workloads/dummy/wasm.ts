import { detectWasmFeatures, loadWasm, variantSupported, type EmscriptenModule, type WasmVariant } from '../../backends/wasm';
import { Unsupported, type Implementation } from '../../harness/types';

interface HelloModule extends EmscriptenModule {
  _hello_sum(ptr: number, n: number): number;
}

export class WasmSum implements Implementation<Float32Array, number> {
  readonly id: string;
  readonly tech = 'wasm';
  private m?: HelloModule;
  private ptr = 0;
  private n = 0;
  private result = 0;

  constructor(private readonly variant: WasmVariant) {
    this.id = `wasm-${variant}`;
  }

  async init(): Promise<void> {
    if (!variantSupported(this.variant, await detectWasmFeatures())) throw new Unsupported(`browser lacks features for ${this.id}`);
    this.m = await loadWasm<HelloModule>('hello', this.variant);
  }

  async setup(input: Float32Array): Promise<void> {
    const m = this.m!;
    this.n = input.length;
    this.ptr = m._malloc(input.byteLength);
    if (!this.ptr) throw new Error(`malloc(${input.byteLength}) failed`);
    m.HEAPF32.set(input, this.ptr >> 2); // read HEAPF32 after malloc: growth replaces the view
  }

  async run(): Promise<void> {
    this.result = this.m!._hello_sum(this.ptr, this.n);
  }

  async readback(): Promise<number> {
    return this.result;
  }

  teardown(): void {
    if (this.ptr) this.m!._free(this.ptr);
    this.ptr = 0;
  }

  dispose(): void {
    this.m = undefined;
  }
}
