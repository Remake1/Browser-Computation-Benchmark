// Exact verification: the output must equal the input sorted ascending, element for
// element, which checks both order and multiset (a dropped, duplicated or padded key
// shows up as a mismatch). The reference is the engine's own Uint32Array sort, so
// js-builtin is checked against itself; every other implementation is independent of it.

/** Computed once per input and shared by every implementation's verification. */
const cache = new WeakMap<Uint32Array, Uint32Array>();

export function referenceSort(input: Uint32Array): Uint32Array {
  let ref = cache.get(input);
  if (!ref) {
    ref = input.slice().sort();
    cache.set(input, ref);
  }
  return ref;
}

export function verifySort(output: Uint32Array, input: Uint32Array): boolean {
  const ref = referenceSort(input);
  if (output.length !== ref.length) return false;
  for (let i = 0; i < ref.length; i++) if (output[i] !== ref[i]) return false;
  return true;
}
