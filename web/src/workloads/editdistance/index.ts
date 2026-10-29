// Edit distance: Levenshtein distance between two strings of length L over
// {A, C, G, T}, L = 64…32768 (PLAN.md §7). Every implementation returns the last row
// of the DP matrix, D[L][0..L], whose last entry is the distance. JS and the scalar
// WASM build run the two-row DP; the simd and threads builds a blocked anti-diagonal
// wavefront (vectorized inside a tile, tiles across threads); on the GPU one dispatch
// per anti-diagonal of cells (naive) or of 256 × 256 tiles (opt). The negative
// control of PLAN.md §7: every cell depends on its left, upper and diagonal neighbours.
import { Rng } from '../../harness/rng';
import type { Workload } from '../../harness/types';
import { GpuEditDistanceNaive, GpuEditDistanceTiled } from './gpu';
import { JsEditDistance } from './js';
import { verifyEditDistance } from './reference';
import { WasmEditDistance } from './wasm';

/** Characters as codes 0–3 (A, C, G, T), one per byte. */
export interface EditInput {
  a: Uint8Array;
  b: Uint8Array;
}

export const editdistance: Workload<EditInput, Uint16Array | Uint32Array> = {
  id: 'editdistance',
  // Powers of two with 1.5× steps between them: 64, 96, 128, …, 32768.
  sizes: Array.from({ length: 19 }, (_, i) => (i % 2 ? 3 : 2) * 2 ** (5 + (i >> 1))),
  /** Two independent uniform strings. */
  generate(n, seed) {
    const rng = new Rng(seed);
    const dna = () => Uint8Array.from({ length: n }, () => rng.u32() >>> 30);
    const a = dna();
    return { a, b: dna() };
  },
  verify: verifyEditDistance,
  /** DP cells: M items/s is Mcells/s (CUPS). */
  items: (n) => n * n,
  implementations: [
    new JsEditDistance(),
    new WasmEditDistance('scalar'),
    new WasmEditDistance('simd'),
    new WasmEditDistance('threads'),
    new GpuEditDistanceNaive(),
    new GpuEditDistanceTiled(),
  ],
};
