import { detectWasmFeatures, loadWasm, variantSupported, wasmThreadCount, type EmscriptenModule, type WasmVariant } from '../../backends/wasm';
import { Unsupported, type Implementation } from '../../harness/types';
import type { MatmulInput } from '.';

interface MatmulModule extends EmscriptenModule {
  _matmul_init(threads: number): number;
  _matmul_shutdown(): void;
  _matmul(a: number, b: number, c: number, n: number): void;
}

export class WasmMatmul implements Implementation<MatmulInput, Float32Array> {
  readonly id: string;
  readonly tech = 'wasm';
  private m?: MatmulModule;
  private n = 0;
  private a = 0;
  private b = 0;
  private c = 0;

  constructor(private readonly variant: WasmVariant) {
    this.id = `wasm-${variant}`;
  }

  async init(): Promise<void> {
    if (!variantSupported(this.variant, await detectWasmFeatures())) throw new Unsupported(`browser lacks features for ${this.id}`);
    this.m = await loadWasm<MatmulModule>('matmul', this.variant);
    const want = this.variant === 'threads' ? wasmThreadCount() : 1;
    const got = this.m._matmul_init(want);
    if (got !== want) throw new Error(`matmul_init(${want}) returned ${got}`);
  }

  async setup(input: MatmulInput): Promise<void> {
    const m = this.m!;
    const bytes = input.a.byteLength;
    this.n = input.n;
    this.a = m._malloc(bytes);
    this.b = m._malloc(bytes);
    this.c = m._malloc(bytes);
    if (!this.a || !this.b || !this.c) throw new Error(`malloc(${bytes}) failed`);
    // Read HEAPF32 after malloc: memory growth replaces the view.
    m.HEAPF32.set(input.a, this.a >> 2);
    m.HEAPF32.set(input.b, this.b >> 2);
  }

  async run(): Promise<void> {
    this.m!._matmul(this.a, this.b, this.c, this.n);
  }

  async readback(): Promise<Float32Array> {
    const start = this.c >> 2;
    return this.m!.HEAPF32.slice(start, start + this.n * this.n);
  }

  teardown(): void {
    const m = this.m!;
    for (const ptr of [this.a, this.b, this.c]) if (ptr) m._free(ptr);
    this.a = this.b = this.c = 0;
  }

  dispose(): void {
    this.m?._matmul_shutdown();
    this.m = undefined;
  }
}
