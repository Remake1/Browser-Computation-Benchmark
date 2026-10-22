import type { Implementation } from '../../harness/types';
import type { MatmulInput } from '.';

/** i-k-j loop order, the same as matmul_rows in wasm/src/matmul.cpp. */
export class JsMatmul implements Implementation<MatmulInput, Float32Array> {
  readonly id = 'js';
  readonly tech = 'js';
  private input?: MatmulInput;
  private c = new Float32Array(0);

  async init(): Promise<void> {}

  async setup(input: MatmulInput): Promise<void> {
    this.input = input;
    this.c = new Float32Array(input.n * input.n);
  }

  async run(): Promise<void> {
    const { n, a, b } = this.input!;
    const c = this.c;
    for (let i = 0; i < n; i++) {
      const ci = i * n;
      c.fill(0, ci, ci + n);
      for (let k = 0; k < n; k++) {
        const aik = a[ci + k];
        const bk = k * n;
        for (let j = 0; j < n; j++) c[ci + j] += aik * b[bk + j];
      }
    }
  }

  async readback(): Promise<Float32Array> {
    return this.c;
  }

  teardown(): void {
    this.input = undefined;
    this.c = new Float32Array(0);
  }

  dispose(): void {}
}
