// Dummy workload for milestone 1: sum of f32[n]. It exists to exercise the harness
// with all three technologies (WASM reuses hello_sum from the milestone 0 module),
// not as a measurement; the real reduction workload comes in milestone 3.
import { Rng } from '../../harness/rng';
import type { Workload } from '../../harness/types';
import { GpuSum } from './gpu';
import { JsSum } from './js';
import { WasmSum } from './wasm';

/** f64 Kahan sum, the reference every f32 result is compared against. */
export function referenceSum(x: Float32Array): number {
  let sum = 0;
  let c = 0;
  for (let i = 0; i < x.length; i++) {
    const y = x[i] - c;
    const t = sum + y;
    c = t - sum - y;
    sum = t;
  }
  return sum;
}

const REL_TOLERANCE = 1e-4;

export const dummy: Workload<Float32Array, number> = {
  id: 'dummy',
  sizes: [2 ** 10, 2 ** 12, 2 ** 14, 2 ** 16, 2 ** 18, 2 ** 20, 2 ** 22],
  generate: (size, seed) => new Rng(seed).fillF32(new Float32Array(size)),
  verify(output, input) {
    const want = referenceSum(input);
    return Number.isFinite(output) && Math.abs(output - want) <= REL_TOLERANCE * Math.abs(want);
  },
  implementations: [new JsSum(), new WasmSum('scalar'), new WasmSum('simd'), new WasmSum('threads'), new GpuSum()],
};
