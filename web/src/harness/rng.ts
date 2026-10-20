// Seeded PRNG so every implementation, browser and run sees identical inputs.
// xoshiro128** seeded through splitmix32; 32-bit integer ops only, so it is exact
// in JS and easy to mirror in C++ if the native build ever needs the same data.

function splitmix32(state: number): () => number {
  return () => {
    state = (state + 0x9e3779b9) | 0;
    let z = state;
    z = Math.imul(z ^ (z >>> 16), 0x85ebca6b);
    z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35);
    return (z ^ (z >>> 16)) >>> 0;
  };
}

export class Rng {
  private s0: number;
  private s1: number;
  private s2: number;
  private s3: number;

  constructor(seed: number) {
    const sm = splitmix32(seed >>> 0);
    this.s0 = sm();
    this.s1 = sm();
    this.s2 = sm();
    this.s3 = sm();
  }

  /** Uniform u32. */
  u32(): number {
    const result = Math.imul(rotl(Math.imul(this.s1, 5), 7), 9) >>> 0;
    const t = this.s1 << 9;
    this.s2 ^= this.s0;
    this.s3 ^= this.s1;
    this.s1 ^= this.s2;
    this.s0 ^= this.s3;
    this.s2 ^= t;
    this.s3 = rotl(this.s3, 11);
    return result;
  }

  /** Uniform in [0, 1) with 24 bits, so the value is exactly representable as f32. */
  float(): number {
    return (this.u32() >>> 8) * 2 ** -24;
  }

  /** Uniform integer in [0, n), n ≤ 2^32 (multiply-shift; bias is negligible for harness use). */
  int(n: number): number {
    return Math.floor((this.u32() / 2 ** 32) * n);
  }

  fillF32(out: Float32Array, lo = 0, hi = 1): Float32Array {
    const span = hi - lo;
    for (let i = 0; i < out.length; i++) out[i] = lo + this.float() * span;
    return out;
  }

  fillU32(out: Uint32Array): Uint32Array {
    for (let i = 0; i < out.length; i++) out[i] = this.u32();
    return out;
  }

  /** In-place Fisher–Yates. */
  shuffle<T>(items: T[]): T[] {
    for (let i = items.length - 1; i > 0; i--) {
      const j = this.int(i + 1);
      [items[i], items[j]] = [items[j], items[i]];
    }
    return items;
  }
}

function rotl(x: number, k: number): number {
  return (x << k) | (x >>> (32 - k));
}

/** Mixes a base seed with labels (FNV-1a), so each (workload, size) gets its own stream independent of run order. */
export function deriveSeed(seed: number, ...labels: (string | number)[]): number {
  let h = 0x811c9dc5 ^ (seed >>> 0);
  for (const ch of labels.join('/')) h = Math.imul(h ^ ch.charCodeAt(0), 0x01000193);
  return splitmix32(h)();
}
