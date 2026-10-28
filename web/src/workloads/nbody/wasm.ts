import { detectWasmFeatures, loadWasm, variantSupported, wasmThreadCount, type EmscriptenModule, type WasmVariant } from '../../backends/wasm';
import { Unsupported, type Implementation } from '../../harness/types';

interface NbodyModule extends EmscriptenModule {
  _nbody_init(threads: number): number;
  _nbody_shutdown(): void;
  _nbody(b: number, acc: number, n: number): void;
}

export class WasmNbody implements Implementation<Float32Array, Float32Array> {
  readonly id: string;
  readonly tech = 'wasm';
  private m?: NbodyModule;
  private n = 0;
  private b = 0;
  private acc = 0;

  constructor(private readonly variant: WasmVariant) {
    this.id = `wasm-${variant}`;
  }

  async init(): Promise<void> {
    if (!variantSupported(this.variant, await detectWasmFeatures())) throw new Unsupported(`browser lacks features for ${this.id}`);
    this.m = await loadWasm<NbodyModule>('nbody', this.variant);
    const want = this.variant === 'threads' ? wasmThreadCount() : 1;
    const got = this.m._nbody_init(want);
    if (got !== want) throw new Error(`nbody_init(${want}) returned ${got}`);
  }

  async setup(input: Float32Array): Promise<void> {
    const m = this.m!;
    const bytes = input.byteLength;
    this.n = input.length >> 2;
    this.b = m._malloc(bytes);
    this.acc = m._malloc(bytes);
    if (!this.b || !this.acc) throw new Error(`malloc(${bytes}) failed`);
    m.HEAPF32.set(input, this.b >> 2); // read HEAPF32 after malloc: growth replaces the view
  }

  async run(): Promise<void> {
    this.m!._nbody(this.b, this.acc, this.n);
  }

  async readback(): Promise<Float32Array> {
    const start = this.acc >> 2;
    return this.m!.HEAPF32.slice(start, start + 4 * this.n);
  }

  teardown(): void {
    const m = this.m!;
    for (const ptr of [this.b, this.acc]) if (ptr) m._free(ptr);
    this.b = this.acc = 0;
  }

  dispose(): void {
    this.m?._nbody_shutdown();
    this.m = undefined;
  }
}
