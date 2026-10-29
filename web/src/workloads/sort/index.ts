// Sort: u32[n] ascending, n = 2¹⁰…2²⁴ (PLAN.md §7). The engine's Uint32Array sort
// and an LSD radix sort in JS; the same radix sort in every WASM build (parallel in
// the threads build) plus std::sort; on the GPU a global-memory bitonic sort (naive)
// and a 4-bit LSD radix sort (opt).
import { Rng } from '../../harness/rng';
import type { Workload } from '../../harness/types';
import { GpuSortBitonic, GpuSortRadix } from './gpu';
import { JsSortBuiltin, JsSortRadix } from './js';
import { verifySort } from './reference';
import { WasmSort } from './wasm';

export const sort: Workload<Uint32Array, Uint32Array> = {
  id: 'sort',
  // Powers of two with 1.5× steps between them: 1024, 1536, 2048, …, 2²⁴.
  sizes: Array.from({ length: 29 }, (_, i) => (i % 2 ? 3 : 2) * 2 ** (9 + (i >> 1))),
  /** Uniform u32 keys: every radix digit is uniform, the common benchmark case. */
  generate(n, seed) {
    return new Rng(seed).fillU32(new Uint32Array(n));
  },
  verify: verifySort,
  items: (n) => n,
  implementations: [
    new JsSortBuiltin(),
    new JsSortRadix(),
    new WasmSort('scalar', 'std'),
    new WasmSort('scalar'),
    new WasmSort('simd'),
    new WasmSort('threads'),
    new GpuSortBitonic(),
    new GpuSortRadix(),
  ],
};
