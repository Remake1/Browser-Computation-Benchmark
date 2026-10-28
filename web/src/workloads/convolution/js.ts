import type { Implementation } from '../../harness/types';
import type { ConvolutionInput } from '.';
import { KERNEL, KSIZE, RADIUS } from './kernel';

const clamp = (v: number, hi: number) => (v < 0 ? 0 : v > hi ? hi : v);

/**
 * The direct 2D loops of convolve_rows in wasm/src/convolution.cpp: per output row the
 * five source rows are clamped once, interior columns sum the 25 taps unclamped, and
 * only the border columns clamp per tap. The accumulator is a JS number (f64); the
 * inputs make every product and sum exact, so the result equals the f32 builds'.
 */
export class JsConvolution implements Implementation<ConvolutionInput, Float32Array> {
  readonly id = 'js';
  readonly tech = 'js';
  private input?: ConvolutionInput;
  private out = new Float32Array(0);
  /** Start offsets of the five clamped source rows of the current output row. */
  private readonly rows = new Int32Array(KSIZE);

  async init(): Promise<void> {}

  async setup(input: ConvolutionInput): Promise<void> {
    this.input = input;
    this.out = new Float32Array(input.w * input.w);
  }

  async run(): Promise<void> {
    const { w, image: src } = this.input!;
    const out = this.out;
    const rows = this.rows;
    // [0, x0) and [x1, w) are border columns; both are empty-safe for w < 2R.
    const x0 = Math.min(RADIUS, w);
    const x1 = Math.max(x0, w - RADIUS);
    for (let y = 0; y < w; y++) {
      for (let ky = 0; ky < KSIZE; ky++) rows[ky] = clamp(y + ky - RADIUS, w - 1) * w;
      const o = y * w;
      this.border(o, 0, x0);
      for (let x = x0; x < x1; x++) {
        let s = 0;
        for (let ky = 0; ky < KSIZE; ky++) {
          const r = rows[ky] + x - RADIUS;
          for (let kx = 0; kx < KSIZE; kx++) s += KERNEL[ky * KSIZE + kx] * src[r + kx];
        }
        out[o + x] = s;
      }
      this.border(o, x1, w);
    }
  }

  /** Columns [x0, x1) of the output row at offset o, with each tap's column clamped. */
  private border(o: number, x0: number, x1: number): void {
    const { w, image: src } = this.input!;
    const rows = this.rows;
    for (let x = x0; x < x1; x++) {
      let s = 0;
      for (let ky = 0; ky < KSIZE; ky++) {
        for (let kx = 0; kx < KSIZE; kx++) s += KERNEL[ky * KSIZE + kx] * src[rows[ky] + clamp(x + kx - RADIUS, w - 1)];
      }
      this.out[o + x] = s;
    }
  }

  async readback(): Promise<Float32Array> {
    return this.out;
  }

  teardown(): void {
    this.input = undefined;
    this.out = new Float32Array(0);
  }

  dispose(): void {}
}
