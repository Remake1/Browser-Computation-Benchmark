// Convolution: W×W grayscale f32 image, fixed 5×5 Gaussian, clamped edges,
// W = 64…4096 (PLAN.md §7). CPU implementations share the direct 2D loops of
// wasm/src/convolution.cpp; the GPU ones use one invocation per pixel, reading
// the image straight from storage (naive) or from a tile + halo staged in
// workgroup memory (opt).
import { Rng } from '../../harness/rng';
import type { Workload } from '../../harness/types';
import { GpuConvolution } from './gpu';
import { JsConvolution } from './js';
import naiveWgsl from './naive.wgsl?raw';
import { verifyConvolution } from './reference';
import tiledWgsl from './tiled.wgsl?raw';
import { WasmConvolution } from './wasm';

export interface ConvolutionInput {
  /** Image width and height. */
  w: number;
  /** w×w pixels, row-major. */
  image: Float32Array;
}

export const convolution: Workload<ConvolutionInput, Float32Array> = {
  id: 'convolution',
  // Powers of two with 1.5× steps between them: 64, 96, 128, …, 4096.
  sizes: Array.from({ length: 13 }, (_, i) => (i % 2 ? 3 : 2) * 2 ** (5 + (i >> 1))),
  /**
   * 8-bit grayscale: multiples of 1/256 in [0, 1). The kernel weights are multiples of
   * 1/256 too, so every product and partial sum is a multiple of 2⁻¹⁶ below 1 and every
   * f32 operation is exact, in any tap order and with or without FMA: all
   * implementations must return exactly the reference image.
   */
  generate(w, seed) {
    const rng = new Rng(seed);
    const image = new Float32Array(w * w);
    for (let i = 0; i < image.length; i++) image[i] = rng.int(256) / 256;
    return { w, image };
  },
  verify: verifyConvolution,
  /** 25 multiply-adds per pixel. */
  flops: (w) => 50 * w * w,
  /** Each pixel read once and written once. */
  bytes: (w) => 8 * w * w,
  implementations: [
    new JsConvolution(),
    new WasmConvolution('scalar'),
    new WasmConvolution('simd'),
    new WasmConvolution('threads'),
    new GpuConvolution('gpu-naive', naiveWgsl),
    new GpuConvolution('gpu-opt', tiledWgsl),
  ],
};
