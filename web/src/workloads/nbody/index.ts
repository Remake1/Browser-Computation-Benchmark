// N-body, one step: softened gravitational acceleration of n bodies, O(n²),
// n = 64…65536 (PLAN.md §7). Bodies are vec4 (x, y, z, mass); the output is
// vec4 (ax, ay, az, 0) per body. CPU implementations share the double loop of
// wasm/src/nbody.cpp; the GPU ones use one invocation per body, reading the other
// bodies straight from storage (naive) or in tiles staged in workgroup memory (opt).
import { Rng } from '../../harness/rng';
import type { Workload } from '../../harness/types';
import { GpuNbody } from './gpu';
import { JsNbody } from './js';
import naiveWgsl from './naive.wgsl?raw';
import { verifyNbody } from './reference';
import tiledWgsl from './tiled.wgsl?raw';
import { WasmNbody } from './wasm';

export const nbody: Workload<Float32Array, Float32Array> = {
  id: 'nbody',
  // Powers of two with 1.5× steps between them: 64, 96, 128, …, 65536.
  sizes: Array.from({ length: 21 }, (_, i) => (i % 2 ? 3 : 2) * 2 ** (5 + (i >> 1))),
  /** Positions uniform in [−1, 1)³, masses uniform in [0.5, 1). */
  generate(n, seed) {
    const rng = new Rng(seed);
    const bodies = new Float32Array(4 * n);
    for (let i = 0; i < n; i++) {
      bodies[4 * i] = rng.float() * 2 - 1;
      bodies[4 * i + 1] = rng.float() * 2 - 1;
      bodies[4 * i + 2] = rng.float() * 2 - 1;
      bodies[4 * i + 3] = 0.5 + rng.float() * 0.5;
    }
    return bodies;
  },
  verify: verifyNbody,
  /** The conventional 20 flops per interaction (sqrt and division counted as one each). */
  flops: (n) => 20 * n * n,
  implementations: [new JsNbody(), new WasmNbody('scalar'), new WasmNbody('simd'), new WasmNbody('threads'), new GpuNbody('gpu-naive', naiveWgsl), new GpuNbody('gpu-opt', tiledWgsl)],
};
