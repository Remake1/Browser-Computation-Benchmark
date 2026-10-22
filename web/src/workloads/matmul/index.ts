// Matrix multiply C = A·B: two N×N f32 matrices, row-major (PLAN.md §7).
// CPU implementations share the i-k-j loop order; the GPU ones use one invocation
// per output element, naive or with 16×16 tiles in workgroup memory.
import { Rng } from '../../harness/rng';
import type { Workload } from '../../harness/types';
import { GpuMatmul } from './gpu';
import { JsMatmul } from './js';
import naiveWgsl from './naive.wgsl?raw';
import { verifyMatmul } from './reference';
import tiledWgsl from './tiled.wgsl?raw';
import { WasmMatmul } from './wasm';

export interface MatmulInput {
  n: number;
  a: Float32Array;
  b: Float32Array;
}

export const matmul: Workload<MatmulInput, Float32Array> = {
  id: 'matmul',
  // Powers of two with 1.5× steps between them, for finer crossover resolution.
  // 24 is not a multiple of the 16×16 GPU tile, so partial tiles are exercised.
  sizes: [16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048],
  generate(n, seed) {
    const rng = new Rng(seed);
    return { n, a: rng.fillF32(new Float32Array(n * n), -1, 1), b: rng.fillF32(new Float32Array(n * n), -1, 1) };
  },
  verify: verifyMatmul,
  flops: (n) => 2 * n ** 3,
  implementations: [
    new JsMatmul(),
    new WasmMatmul('scalar'),
    new WasmMatmul('simd'),
    new WasmMatmul('threads'),
    new GpuMatmul('gpu-naive', naiveWgsl),
    new GpuMatmul('gpu-opt', tiledWgsl),
  ],
};
