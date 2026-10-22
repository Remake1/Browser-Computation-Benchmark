// f64 Kahan sum, the reference every result is compared against. generate() makes
// every f32 summation order exact, so the check is equality: a dropped, duplicated
// or out-of-range element shows up at any n, which a relative tolerance would hide
// at large n (one element is ~1e-8 of Σ|x| at 2²⁶).

/** Computed once per input and shared by every implementation's verification. */
const cache = new WeakMap<Float32Array, number>();

export function referenceSum(x: Float32Array): number {
  const hit = cache.get(x);
  if (hit !== undefined) return hit;
  let sum = 0;
  let c = 0;
  for (let i = 0; i < x.length; i++) {
    const y = x[i] - c;
    const t = sum + y;
    c = t - sum - y;
    sum = t;
  }
  cache.set(x, sum);
  return sum;
}

export function verifyReduction(output: number, input: Float32Array): boolean {
  return output === referenceSum(input);
}
