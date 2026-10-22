// workgroup_sum() for opt.wgsl without subgroups: tree in workgroup memory.

var<workgroup> partial: array<f32, WG>;

/** Sum of `s` over the workgroup; every invocation must call it. */
fn workgroup_sum(s: f32, lid: u32) -> f32 {
  partial[lid] = s;
  workgroupBarrier();
  for (var stride = WG / 2u; stride > 0u; stride >>= 1u) {
    if (lid < stride) {
      partial[lid] += partial[lid + stride];
    }
    workgroupBarrier();
  }
  return partial[0];
}
