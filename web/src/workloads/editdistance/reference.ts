// Exact reference for verification: the last DP row D[la][0..lb], computed with
// Myers' bit-parallel algorithm (J. ACM 46(3), 1999) in its multi-word form for
// global distance, a different algorithm from every implementation's DP. a is the
// "pattern": its DP column is held as bit vectors of vertical +1/−1 deltas in 32-bit
// words, and each character of b advances all of them, O(la·lb/32) word operations.
// The score, D[la][j], changes by the bottom word's horizontal delta per column.
import type { EditInput } from '.';

/** Computed once per input and shared by every implementation's verification. */
const cache = new WeakMap<EditInput, Int32Array>();

function reference(input: EditInput): Int32Array {
  const hit = cache.get(input);
  if (hit) return hit;
  const { a, b } = input;
  const m = a.length;
  const words = Math.ceil(m / 32);
  // peq[c * words + w]: bit k set where a[32w + k] == c.
  const peq = new Int32Array(4 * words);
  for (let i = 0; i < m; i++) peq[a[i] * words + (i >> 5)] |= 1 << (i & 31);
  const pv = new Int32Array(words).fill(-1); // D[i][0] = i: every vertical delta +1
  const mv = new Int32Array(words);
  const last = new Int32Array(b.length + 1);
  let score = m;
  last[0] = score;
  const lastHigh = 1 << ((m - 1) & 31);
  for (let j = 0; j < b.length; j++) {
    const base = b[j] * words;
    let hin = 1; // D[0][j] = j: the top row's horizontal delta is +1
    for (let w = 0; w < words; w++) {
      let eq = peq[base + w];
      const p = pv[w];
      const n = mv[w];
      const xv = eq | n;
      if (hin < 0) eq |= 1;
      const xh = ((((eq & p) + p) | 0) ^ p) | eq;
      let ph = n | ~(xh | p);
      let mh = p & xh;
      const high = w === words - 1 ? lastHigh : 1 << 31;
      const hout = ph & high ? 1 : mh & high ? -1 : 0;
      ph <<= 1;
      mh <<= 1;
      if (hin < 0) mh |= 1;
      else if (hin > 0) ph |= 1;
      pv[w] = mh | ~(xv | ph);
      mv[w] = ph & xv;
      hin = hout;
    }
    score += hin;
    last[j + 1] = score;
  }
  cache.set(input, last);
  return last;
}

export function verifyEditDistance(output: ArrayLike<number>, input: EditInput): boolean {
  const ref = reference(input);
  if (output.length !== ref.length) return false;
  for (let j = 0; j < ref.length; j++) if (output[j] !== ref[j]) return false;
  return true;
}
