import type { Implementation } from '../../harness/types';
import type { EditInput } from '.';

/**
 * The two-row DP of `ed_rows` in wasm/src/editdistance.cpp: row by row, the cell to
 * the left and the diagonal one carried in locals. Rows are Uint16Array like the
 * C++ cells; arithmetic is on JS numbers (small integers).
 */
export class JsEditDistance implements Implementation<EditInput, Uint16Array> {
  readonly id = 'js';
  readonly tech = 'js';
  private input?: EditInput;
  private rows: [Uint16Array, Uint16Array] = [new Uint16Array(0), new Uint16Array(0)];
  private out: Uint16Array = new Uint16Array(0);

  async init(): Promise<void> {}

  async setup(input: EditInput): Promise<void> {
    this.input = input;
    this.rows = [new Uint16Array(input.b.length + 1), new Uint16Array(input.b.length + 1)];
  }

  async run(): Promise<void> {
    const { a, b } = this.input!;
    const lb = b.length;
    let [prev, cur] = this.rows;
    for (let j = 0; j <= lb; j++) prev[j] = j;
    for (let i = 1; i <= a.length; i++) {
      const ai = a[i - 1];
      let diag = prev[0];
      let left = i;
      cur[0] = i;
      for (let j = 1; j <= lb; j++) {
        const up = prev[j];
        left = Math.min(Math.min(diag + +(ai !== b[j - 1]), up + 1), left + 1);
        cur[j] = left;
        diag = up;
      }
      [prev, cur] = [cur, prev];
    }
    this.out = prev;
  }

  async readback(): Promise<Uint16Array> {
    return this.out;
  }

  teardown(): void {
    this.input = undefined;
    this.rows = [new Uint16Array(0), new Uint16Array(0)];
    this.out = new Uint16Array(0);
  }

  dispose(): void {}
}
