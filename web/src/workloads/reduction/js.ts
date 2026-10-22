import type { Implementation } from '../../harness/types';

/**
 * The simple loop, in the same order as the scalar build of wasm/src/reduction.cpp.
 * The accumulator is a plain JS number (f64): forcing f32 with Math.fround puts
 * f64→f32→f64 conversions on the loop-carried chain and runs ~3× slower in V8,
 * while an f64 add has the same latency as an f32 add. The inputs make every order
 * and precision exact, so the result is the same as wasm-scalar's.
 */
export class JsReduction implements Implementation<Float32Array, number> {
  readonly id = 'js';
  readonly tech = 'js';
  private x: Float32Array = new Float32Array(0);
  private result = 0;

  async init(): Promise<void> {}

  async setup(input: Float32Array): Promise<void> {
    this.x = input;
  }

  async run(): Promise<void> {
    const x = this.x;
    let s = 0;
    for (let i = 0; i < x.length; i++) s += x[i];
    this.result = s;
  }

  async readback(): Promise<number> {
    return this.result;
  }

  teardown(): void {
    this.x = new Float32Array(0);
  }

  dispose(): void {}
}
