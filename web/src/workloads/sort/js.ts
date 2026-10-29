import type { Implementation } from '../../harness/types';

/**
 * Shared by both JS sorts: the sort is in place and the input is shared by every
 * implementation and iteration, so setup() copies it (the 'upload' phase), and
 * readback() returns the sorted copy.
 */
abstract class JsSortBase implements Implementation<Uint32Array, Uint32Array> {
  abstract readonly id: string;
  readonly tech = 'js';
  protected a = new Uint32Array(0);

  async init(): Promise<void> {}

  async setup(input: Uint32Array): Promise<void> {
    this.a = input.slice();
  }

  abstract run(): Promise<void>;

  async readback(): Promise<Uint32Array> {
    return this.a;
  }

  teardown(): void {
    this.a = new Uint32Array(0);
  }

  dispose(): void {}
}

/** Uint32Array.prototype.sort(): engine code (C++ or self-hosted), not JS written here. */
export class JsSortBuiltin extends JsSortBase {
  readonly id = 'js-builtin';

  async run(): Promise<void> {
    this.a.sort();
  }
}

/**
 * The LSD radix sort of wasm/src/sort.cpp (scalar build): 8-bit digits, one counting
 * pass for all four histograms, then four stable scatter passes ping-ponging between
 * the array and a scratch buffer, so the result ends up back in the array. The
 * scratch buffer is allocated in setup(), like WASM's.
 */
export class JsSortRadix extends JsSortBase {
  readonly id = 'js-radix';
  private tmp = new Uint32Array(0);
  private readonly count = new Uint32Array(4 * 256);

  override async setup(input: Uint32Array): Promise<void> {
    await super.setup(input);
    this.tmp = new Uint32Array(input.length);
  }

  async run(): Promise<void> {
    const n = this.a.length;
    const count = this.count;
    count.fill(0);
    let src = this.a;
    let dst = this.tmp;
    for (let i = 0; i < n; i++) {
      const x = src[i];
      count[x & 0xff]++;
      count[256 + ((x >>> 8) & 0xff)]++;
      count[512 + ((x >>> 16) & 0xff)]++;
      count[768 + (x >>> 24)]++;
    }
    for (let pass = 0; pass < 4; pass++) {
      const shift = 8 * pass;
      const base = 256 * pass;
      let sum = 0;
      for (let d = base; d < base + 256; d++) {
        const c = count[d];
        count[d] = sum;
        sum += c;
      }
      for (let i = 0; i < n; i++) {
        const x = src[i];
        dst[count[base + ((x >>> shift) & 0xff)]++] = x;
      }
      [src, dst] = [dst, src];
    }
  }

  override teardown(): void {
    super.teardown();
    this.tmp = new Uint32Array(0);
  }
}
