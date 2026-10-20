import type { Implementation } from '../../harness/types';

/** Four f32 accumulators, the same order as hello_sum in wasm/src/hello.cpp. */
export class JsSum implements Implementation<Float32Array, number> {
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
    const f = Math.fround;
    let a0 = 0, a1 = 0, a2 = 0, a3 = 0, i = 0;
    for (; i + 4 <= x.length; i += 4) {
      a0 = f(a0 + x[i]);
      a1 = f(a1 + x[i + 1]);
      a2 = f(a2 + x[i + 2]);
      a3 = f(a3 + x[i + 3]);
    }
    let s = f(f(a0 + a1) + f(a2 + a3));
    for (; i < x.length; i++) s = f(s + x[i]);
    this.result = s;
  }

  async readback(): Promise<number> {
    return this.result;
  }

  dispose(): void {}
}
