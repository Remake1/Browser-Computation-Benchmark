// workgroup_sum() for opt.wgsl with subgroups: subgroupAdd within each subgroup,
// then invocation 0 adds the per-subgroup sums (8 for 32-wide subgroups).
// WebGPU doesn't specify how invocations map to subgroups, so each subgroup takes
// a slot from an atomic counter instead of computing one from its index. The slot
// order can vary between runs; with this workload's exact inputs it can't change
// the result.

var<workgroup> partial: array<f32, WG>;
var<workgroup> slots: atomic<u32>;

/** Sum of `s` over the workgroup, valid in invocation 0; every invocation must call it. */
fn workgroup_sum(s: f32, lid: u32) -> f32 {
  let t = subgroupAdd(s);
  if (subgroupElect()) {
    partial[atomicAdd(&slots, 1u)] = t;
  }
  workgroupBarrier();
  var total = 0.0;
  if (lid == 0u) {
    let count = atomicLoad(&slots);
    for (var j = 0u; j < count; j++) {
      total += partial[j];
    }
  }
  return total;
}
