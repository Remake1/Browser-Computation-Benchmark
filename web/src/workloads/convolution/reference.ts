// f64 reference: a direct per-tap clamped convolution, independent of the
// implementations' interior/border split. generate() makes every product and partial
// sum exact in f32, so the check is equality: an off-by-one clamp, a skipped row at
// a thread or tile boundary, or a wrong halo shows up as a mismatch at any W. Equality
// is stricter than the 1e-4 relative tolerance of PLAN.md §7, which these inputs make
// unnecessary.
import type { ConvolutionInput } from '.';
import { KERNEL, KSIZE, RADIUS } from './kernel';

/** Computed once per input and shared by every implementation's verification. */
const cache = new WeakMap<ConvolutionInput, Float32Array>();

export function referenceConvolution(input: ConvolutionInput): Float32Array {
  const hit = cache.get(input);
  if (hit) return hit;
  const { w, image } = input;
  const clamp = (v: number) => Math.min(Math.max(v, 0), w - 1);
  // The exact f64 sums are representable in f32, so storing them as f32 loses nothing.
  const ref = new Float32Array(w * w);
  for (let y = 0; y < w; y++) {
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let ky = 0; ky < KSIZE; ky++) {
        const row = clamp(y + ky - RADIUS) * w;
        for (let kx = 0; kx < KSIZE; kx++) s += KERNEL[ky * KSIZE + kx] * image[row + clamp(x + kx - RADIUS)];
      }
      ref[y * w + x] = s;
    }
  }
  cache.set(input, ref);
  return ref;
}

export function verifyConvolution(output: Float32Array, input: ConvolutionInput): boolean {
  const ref = referenceConvolution(input);
  if (output.length !== ref.length) return false;
  for (let i = 0; i < ref.length; i++) if (output[i] !== ref[i]) return false;
  return true;
}
