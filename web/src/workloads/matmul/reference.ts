// f64 reference for verification. Checks a componentwise bound,
//   |C − R|ᵢⱼ ≤ tol · (|A|·|B|)ᵢⱼ,
// the standard error bound for a dot product, so it stays meaningful for entries
// near zero (inputs are in [−1, 1)). f32 summation in any order is within
// ~N·2⁻²⁴ ≈ 1.2e-4 of this at N = 2048, well under the 1e-3 tolerance.
import type { MatmulInput } from '.';

const REL_TOLERANCE = 1e-3;

/** Above this many multiply-adds the reference checks a subset of rows, so verification stays under ~0.5 s. */
const REFERENCE_BUDGET = 2 ** 28;
/** Rows always checked at each edge, so partial tiles / last thread's block are always covered. */
const EDGE_ROWS = 16;

/** Which rows of C get checked: all of them, or the edges plus an even spread. */
export function checkedRows(n: number): number[] {
  const count = Math.min(n, Math.max(2 * EDGE_ROWS, Math.floor(REFERENCE_BUDGET / (n * n))));
  if (count === n) return Array.from({ length: n }, (_, i) => i);
  const rows = new Set<number>();
  for (let i = 0; i < EDGE_ROWS; i++) rows.add(i).add(n - 1 - i);
  const middle = count - rows.size;
  for (let r = 0; r < middle; r++) rows.add(EDGE_ROWS + Math.floor(((r + 0.5) * (n - 2 * EDGE_ROWS)) / middle));
  return [...rows].sort((x, y) => x - y);
}

interface Reference {
  rows: number[];
  /** f64 A·B and |A|·|B| for each checked row, row-major. */
  value: Float64Array;
  bound: Float64Array;
}

/** Computed once per input and shared by every implementation's verification. */
const cache = new WeakMap<MatmulInput, Reference>();

function reference(input: MatmulInput): Reference {
  const hit = cache.get(input);
  if (hit) return hit;
  const { n, a, b } = input;
  const rows = checkedRows(n);
  const absB = Float64Array.from(b, Math.abs);
  const value = new Float64Array(rows.length * n);
  const bound = new Float64Array(rows.length * n);
  rows.forEach((i, r) => {
    const o = r * n;
    for (let k = 0; k < n; k++) {
      const aik = a[i * n + k];
      const abs = Math.abs(aik);
      const bk = k * n;
      for (let j = 0; j < n; j++) {
        value[o + j] += aik * b[bk + j];
        bound[o + j] += abs * absB[bk + j];
      }
    }
  });
  const ref = { rows, value, bound };
  cache.set(input, ref);
  return ref;
}

export function verifyMatmul(c: Float32Array, input: MatmulInput): boolean {
  const { n } = input;
  if (c.length !== n * n) return false;
  const { rows, value, bound } = reference(input);
  for (let r = 0; r < rows.length; r++) {
    const ci = rows[r] * n;
    for (let j = 0; j < n; j++) {
      const err = Math.abs(c[ci + j] - value[r * n + j]);
      // `!(a <= b)` so NaN fails.
      if (!(err <= REL_TOLERANCE * bound[r * n + j])) return false;
    }
  }
  return true;
}
