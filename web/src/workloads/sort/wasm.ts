import { detectWasmFeatures, loadWasm, variantSupported, wasmThreadCount, type EmscriptenModule, type WasmVariant } from '../../backends/wasm';
import { Unsupported, type Implementation } from '../../harness/types';

interface SortModule extends EmscriptenModule {
  _sort_init(threads: number): number;
  _sort_shutdown(): void;
  _sort_radix(a: number, tmp: number, n: number): void;
  _sort_std(a: number, n: number): void;
}

/**
 * wasm/src/sort.cpp. `radix` runs sort_radix in the given build (parallel in the
 * threads build); `std` runs std::sort, in the scalar build only (id 'wasm-std').
 */
export class WasmSort implements Implementation<Uint32Array, Uint32Array> {
  readonly id: string;
  readonly tech = 'wasm';
  private m?: SortModule;
  private n = 0;
  private a = 0;
  private tmp = 0;

  constructor(
    private readonly variant: WasmVariant,
    private readonly algorithm: 'radix' | 'std' = 'radix',
  ) {
    this.id = algorithm === 'std' ? 'wasm-std' : `wasm-${variant}`;
  }

  async init(): Promise<void> {
    if (!variantSupported(this.variant, await detectWasmFeatures())) throw new Unsupported(`browser lacks features for ${this.id}`);
    this.m = await loadWasm<SortModule>('sort', this.variant);
    const want = this.variant === 'threads' ? wasmThreadCount() : 1;
    const got = this.m._sort_init(want);
    if (got !== want) throw new Error(`sort_init(${want}) returned ${got}`);
  }

  async setup(input: Uint32Array): Promise<void> {
    const m = this.m!;
    this.n = input.length;
    this.a = m._malloc(input.byteLength);
    if (this.algorithm === 'radix') this.tmp = m._malloc(input.byteLength);
    if (!this.a || (this.algorithm === 'radix' && !this.tmp)) throw new Error(`malloc(${input.byteLength}) failed`);
    m.HEAPU32.set(input, this.a >> 2); // read HEAPU32 after malloc: growth replaces the view
  }

  async run(): Promise<void> {
    if (this.algorithm === 'radix') this.m!._sort_radix(this.a, this.tmp, this.n);
    else this.m!._sort_std(this.a, this.n);
  }

  async readback(): Promise<Uint32Array> {
    const start = this.a >> 2;
    return this.m!.HEAPU32.slice(start, start + this.n);
  }

  teardown(): void {
    const m = this.m!;
    for (const ptr of [this.a, this.tmp]) if (ptr) m._free(ptr);
    this.a = this.tmp = 0;
  }

  dispose(): void {
    this.m?._sort_shutdown();
    this.m = undefined;
  }
}
