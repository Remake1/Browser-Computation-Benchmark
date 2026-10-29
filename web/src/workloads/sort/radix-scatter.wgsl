// GPU LSD radix sort: 4-bit digits, 8 passes ping-ponging between two key buffers,
// so the result ends up back in the first. The host pads the keys to a multiple of
// the 1024-key block with 0xFFFFFFFF, which a stable sort keeps at the end. Each
// pass is three steps, all dispatches in one compute pass:
//   1. radix-count.wgsl  per-block digit counts → hist[digit · blocks + block]
//   2. radix-scan.wgsl   exclusive scan of hist: the first output position of every
//                        (digit, block)
//   3. this shader       stable scatter of each block.
// The algorithm is reduce-then-scan; it needs no inter-workgroup communication
// within a dispatch, so it doesn't depend on forward-progress guarantees that
// WebGPU doesn't give (as single-pass "decoupled look-back" sorts do).
//
// Scatter: invocation lid owns keys 4·lid … 4·lid + 3 of its block, in order. A
// key's output position is offsets[digit · blocks + block] + (keys of the same digit
// owned by earlier invocations) + (same-digit keys earlier in its own vec4). The
// middle term comes from a workgroup scan of per-invocation digit counts, packed
// 16 bits per digit (at most 1024 per block) into two vec4u.

struct Params {
  shift: u32,
  blocks: u32,
}

/** Digit counts, 16 bits each: lo holds digits 0–7, hi digits 8–15, two per u32. */
struct Packed {
  lo: vec4u,
  hi: vec4u,
}

@group(0) @binding(0) var<storage, read> src: array<vec4u>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<storage, read> offsets: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

const WG: u32 = 256u;
const RADIX: u32 = 16u;

var<workgroup> counts: array<Packed, WG>;
var<workgroup> base: array<u32, RADIX>;

/** A count of 1 for digit d. */
fn unit(d: u32) -> Packed {
  let w = d >> 1u;
  let lane = select(vec4u(0u), vec4u(1u << ((d & 1u) * 16u)), vec4u(0u, 1u, 2u, 3u) == vec4u(w & 3u));
  return Packed(select(vec4u(0u), lane, w < 4u), select(lane, vec4u(0u), w < 4u));
}

fn plus(a: Packed, b: Packed) -> Packed {
  return Packed(a.lo + b.lo, a.hi + b.hi);
}

/** Count of digit d. */
fn field(p: Packed, d: u32) -> u32 {
  let w = d >> 1u;
  let word = select(p.hi[w & 3u], p.lo[w & 3u], w < 4u);
  return (word >> ((d & 1u) * 16u)) & 0xffffu;
}

@compute @workgroup_size(WG)
fn main(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(num_workgroups) nwg: vec3u,
  @builtin(local_invocation_index) lid: u32,
) {
  let block = wid.y * nwg.x + wid.x;
  // Spare workgroups of a 2D grid.
  if (block >= params.blocks) {
    return;
  }
  let k = src[block * WG + lid];
  let d = (k >> vec4u(params.shift)) & vec4u(RADIX - 1u);

  // Rank of each key among this invocation's earlier keys of the same digit.
  var mine = Packed(vec4u(0u), vec4u(0u));
  var rank = vec4u(0u);
  mine = plus(mine, unit(d.x));
  rank.y = field(mine, d.y);
  mine = plus(mine, unit(d.y));
  rank.z = field(mine, d.z);
  mine = plus(mine, unit(d.z));
  rank.w = field(mine, d.w);
  mine = plus(mine, unit(d.w));

  // Inclusive Hillis–Steele scan of the packed counts over the workgroup.
  counts[lid] = mine;
  workgroupBarrier();
  for (var off = 1u; off < WG; off <<= 1u) {
    var x = Packed(vec4u(0u), vec4u(0u));
    if (lid >= off) {
      x = counts[lid - off];
    }
    workgroupBarrier();
    counts[lid] = plus(counts[lid], x);
    workgroupBarrier();
  }
  // Exclusive: every field of the inclusive sum is ≥ the same field of `mine`, so no borrows.
  let inclusive = counts[lid];
  let before = Packed(inclusive.lo - mine.lo, inclusive.hi - mine.hi);

  if (lid < RADIX) {
    base[lid] = offsets[lid * params.blocks + block];
  }
  workgroupBarrier();
  dst[base[d.x] + field(before, d.x) + rank.x] = k.x;
  dst[base[d.y] + field(before, d.y) + rank.y] = k.y;
  dst[base[d.z] + field(before, d.z) + rank.z] = k.z;
  dst[base[d.w] + field(before, d.w) + rank.w] = k.w;
}
