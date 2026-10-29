// Pass step 1 of the GPU radix sort (radix-scatter.wgsl has the overview): each
// workgroup counts the 4-bit digit at `shift` of its block of 1024 keys and writes
// the 16 counts to hist[digit · blocks + block], digit-major, so that one exclusive
// scan of hist gives every (digit, block) its first output position.

struct Params {
  shift: u32,
  blocks: u32,
}

@group(0) @binding(0) var<storage, read> src: array<vec4u>;
@group(0) @binding(1) var<storage, read_write> hist: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

const WG: u32 = 256u;
const RADIX: u32 = 16u;

// Workgroup memory starts zeroed.
var<workgroup> counts: array<atomic<u32>, RADIX>;

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
  let d = (src[block * WG + lid] >> vec4u(params.shift)) & vec4u(RADIX - 1u);
  atomicAdd(&counts[d.x], 1u);
  atomicAdd(&counts[d.y], 1u);
  atomicAdd(&counts[d.z], 1u);
  atomicAdd(&counts[d.w], 1u);
  workgroupBarrier();
  if (lid < RADIX) {
    hist[lid * params.blocks + block] = atomicLoad(&counts[lid]);
  }
}
