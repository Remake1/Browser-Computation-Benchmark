// f64 reference for verification. Checks a componentwise bound,
//   |a − R|ᵢ ≤ tol · Σ_j |m_j d_ij / (|d_ij|² + ε²)^{3/2}|,
// the standard error bound for a sum, so it stays meaningful for components that
// nearly cancel. f32 errors measured natively are below 1e-5 of this at n = 16384,
// far under the 1e-3 tolerance of PLAN.md §7.
import { EPS2 } from './js';

const REL_TOLERANCE = 1e-3;

/** Above this many interactions the reference checks a subset of bodies, so verification stays under ~0.5 s. */
const REFERENCE_BUDGET = 2 ** 27;
/** Bodies always checked at each end, so the first and last thread's blocks and partial workgroups are covered. */
const EDGE_BODIES = 16;

/** Which bodies get checked: all of them, or both ends plus an even spread. */
export function checkedBodies(n: number): number[] {
  const count = Math.min(n, Math.max(2 * EDGE_BODIES, Math.floor(REFERENCE_BUDGET / n)));
  if (count === n) return Array.from({ length: n }, (_, i) => i);
  const set = new Set<number>();
  for (let i = 0; i < EDGE_BODIES; i++) set.add(i).add(n - 1 - i);
  const middle = count - set.size;
  for (let r = 0; r < middle; r++) set.add(EDGE_BODIES + Math.floor(((r + 0.5) * (n - 2 * EDGE_BODIES)) / middle));
  return [...set].sort((x, y) => x - y);
}

interface Reference {
  bodies: number[];
  /** f64 acceleration and its bound, 3 components per checked body. */
  value: Float64Array;
  bound: Float64Array;
}

/** Computed once per input and shared by every implementation's verification. */
const cache = new WeakMap<Float32Array, Reference>();

function reference(b: Float32Array): Reference {
  const hit = cache.get(b);
  if (hit) return hit;
  const n = b.length >> 2;
  const bodies = checkedBodies(n);
  const value = new Float64Array(3 * bodies.length);
  const bound = new Float64Array(3 * bodies.length);
  bodies.forEach((i, r) => {
    for (let j = 0; j < n; j++) {
      const dx = b[4 * j] - b[4 * i];
      const dy = b[4 * j + 1] - b[4 * i + 1];
      const dz = b[4 * j + 2] - b[4 * i + 2];
      const d2 = dx * dx + dy * dy + dz * dz + EPS2;
      const s = b[4 * j + 3] / (d2 * Math.sqrt(d2));
      value[3 * r] += dx * s;
      value[3 * r + 1] += dy * s;
      value[3 * r + 2] += dz * s;
      bound[3 * r] += Math.abs(dx * s);
      bound[3 * r + 1] += Math.abs(dy * s);
      bound[3 * r + 2] += Math.abs(dz * s);
    }
  });
  const ref = { bodies, value, bound };
  cache.set(b, ref);
  return ref;
}

export function verifyNbody(acc: Float32Array, input: Float32Array): boolean {
  if (acc.length !== input.length) return false;
  const { bodies, value, bound } = reference(input);
  for (let r = 0; r < bodies.length; r++) {
    const i = bodies[r];
    if (acc[4 * i + 3] !== 0) return false;
    for (let c = 0; c < 3; c++) {
      const err = Math.abs(acc[4 * i + c] - value[3 * r + c]);
      // `!(a <= b)` so NaN fails.
      if (!(err <= REL_TOLERANCE * bound[3 * r + c])) return false;
    }
  }
  return true;
}
