import { detectWasmFeatures, loadWasm, variantSupported, wasmThreadCount, type EmscriptenModule, type WasmVariant } from '../../backends/wasm';
import { Unsupported, type Implementation } from '../../harness/types';
import type { EditInput } from '.';

type EdFn = (a: number, la: number, b: number, lb: number, scratch: number, out: number) => void;

interface EditDistanceModule extends EmscriptenModule {
  _ed_init(threads: number): number;
  _ed_shutdown(): void;
  _ed_scratch_elements(la: number, lb: number): number;
  _ed_rows: EdFn;
  _ed_tiled: EdFn;
}

/**
 * wasm/src/editdistance.cpp. The scalar build runs ed_rows (the two-row DP of the JS
 * baseline); the simd and threads builds run ed_tiled (blocked anti-diagonal
 * wavefront, vectorized within a tile, tiles across the pool in the threads build).
 */
export class WasmEditDistance implements Implementation<EditInput, Uint16Array> {
  readonly id: string;
  readonly tech = 'wasm';
  private m?: EditDistanceModule;
  private la = 0;
  private lb = 0;
  private a = 0;
  private b = 0;
  private scratch = 0;
  private out = 0;

  constructor(private readonly variant: WasmVariant) {
    this.id = `wasm-${variant}`;
  }

  async init(): Promise<void> {
    if (!variantSupported(this.variant, await detectWasmFeatures())) throw new Unsupported(`browser lacks features for ${this.id}`);
    this.m = await loadWasm<EditDistanceModule>('editdistance', this.variant);
    const want = this.variant === 'threads' ? wasmThreadCount() : 1;
    const got = this.m._ed_init(want);
    if (got !== want) throw new Error(`ed_init(${want}) returned ${got}`);
  }

  async setup(input: EditInput): Promise<void> {
    const m = this.m!;
    this.la = input.a.length;
    this.lb = input.b.length;
    this.a = m._malloc(this.la);
    this.b = m._malloc(this.lb);
    this.scratch = m._malloc(2 * m._ed_scratch_elements(this.la, this.lb));
    this.out = m._malloc(2 * (this.lb + 1));
    if (!this.a || !this.b || !this.scratch || !this.out) throw new Error('malloc failed');
    m.HEAPU8.set(input.a, this.a); // read HEAPU8 after malloc: growth replaces the view
    m.HEAPU8.set(input.b, this.b);
  }

  async run(): Promise<void> {
    const m = this.m!;
    const fn = this.variant === 'scalar' ? m._ed_rows : m._ed_tiled;
    fn(this.a, this.la, this.b, this.lb, this.scratch, this.out);
  }

  async readback(): Promise<Uint16Array> {
    return new Uint16Array(this.m!.HEAPU8.buffer, this.out, this.lb + 1).slice();
  }

  teardown(): void {
    const m = this.m!;
    for (const ptr of [this.a, this.b, this.scratch, this.out]) if (ptr) m._free(ptr);
    this.a = this.b = this.scratch = this.out = 0;
  }

  dispose(): void {
    this.m?._ed_shutdown();
    this.m = undefined;
  }
}
