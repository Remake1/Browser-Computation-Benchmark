import type { Implementation } from '../../harness/types';

/** Softening ε², the f32 value of wasm/src/nbody.cpp's EPS2 and the shaders'. */
export const EPS2 = Math.fround(0.01);

/**
 * The double loop of `body` in wasm/src/nbody.cpp: for each body i, sum over every
 * j (including i, whose term is 0) in order. Arithmetic is in JS numbers (f64) on
 * f32 data, like the other JS baselines.
 */
export class JsNbody implements Implementation<Float32Array, Float32Array> {
  readonly id = 'js';
  readonly tech = 'js';
  private b: Float32Array = new Float32Array(0);
  private acc = new Float32Array(0);

  async init(): Promise<void> {}

  async setup(input: Float32Array): Promise<void> {
    this.b = input;
    this.acc = new Float32Array(input.length);
  }

  async run(): Promise<void> {
    const b = this.b;
    const acc = this.acc;
    const n = b.length >> 2;
    for (let i = 0; i < n; i++) {
      const xi = b[4 * i];
      const yi = b[4 * i + 1];
      const zi = b[4 * i + 2];
      let ax = 0;
      let ay = 0;
      let az = 0;
      for (let j = 0; j < n; j++) {
        const dx = b[4 * j] - xi;
        const dy = b[4 * j + 1] - yi;
        const dz = b[4 * j + 2] - zi;
        const d2 = dx * dx + dy * dy + dz * dz + EPS2;
        const inv = 1 / Math.sqrt(d2);
        const s = b[4 * j + 3] * inv * inv * inv;
        ax += dx * s;
        ay += dy * s;
        az += dz * s;
      }
      acc[4 * i] = ax;
      acc[4 * i + 1] = ay;
      acc[4 * i + 2] = az;
      acc[4 * i + 3] = 0;
    }
  }

  async readback(): Promise<Float32Array> {
    return this.acc;
  }

  teardown(): void {
    this.b = new Float32Array(0);
    this.acc = new Float32Array(0);
  }

  dispose(): void {}
}
