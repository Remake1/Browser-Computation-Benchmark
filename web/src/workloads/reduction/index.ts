// Reduction: sum of f32[n], n = 2¹⁰…2²⁶ (PLAN.md §7). JS and wasm-scalar run the
// simple loop; wasm-simd uses f32x4 intrinsics with four accumulators; wasm-threads
// adds per-thread partial sums; the GPU versions are a multi-pass tree reduction
// (naive) and a two-pass grid-stride reduction with vec4 loads (opt).
import { Rng } from '../../harness/rng';
import type { Workload } from '../../harness/types';
import { GpuReductionNaive, GpuReductionOpt } from './gpu';
import { JsReduction } from './js';
import { verifyReduction } from './reference';
import { WasmReduction } from './wasm';

export const reduction: Workload<Float32Array, number> = {
  id: 'reduction',
  // Powers of two with 1.5× steps between them: 1024, 1536, 2048, …, 2²⁶.
  sizes: Array.from({ length: 33 }, (_, i) => (i % 2 ? 3 : 2) * 2 ** (9 + (i >> 1))),
  /**
   * Multiples of 1/256 in (−1, 1), symmetric around 0. Every partial sum, in any
   * order, stays far below 2¹⁶ (2²⁴ steps of 1/256), so every f32 addition is exact
   * and all implementations must return exactly the reference sum. The values have no
   * effect on speed (no subnormals), only on what verification can catch.
   */
  generate(n, seed) {
    const rng = new Rng(seed);
    const x = new Float32Array(n);
    for (let i = 0; i < n; i++) x[i] = (rng.int(511) - 255) / 256;
    return x;
  },
  verify: verifyReduction,
  flops: (n) => n - 1,
  bytes: (n) => 4 * n,
  implementations: [
    new JsReduction(),
    new WasmReduction('scalar'),
    new WasmReduction('simd'),
    new WasmReduction('threads'),
    new GpuReductionNaive(),
    new GpuReductionOpt(),
  ],
};
