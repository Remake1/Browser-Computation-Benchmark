// Single-workgroup sum: each invocation sums a strided slice, then a tree
// reduction in workgroup memory. Deliberately simple; harness test only.

@group(0) @binding(0) var<storage, read> data: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

const WG: u32 = 256u;
var<workgroup> partial: array<f32, WG>;

@compute @workgroup_size(WG)
fn main(@builtin(local_invocation_index) lid: u32) {
  let n = arrayLength(&data);
  var s = 0.0;
  for (var i = lid; i < n; i += WG) {
    s += data[i];
  }
  partial[lid] = s;
  workgroupBarrier();
  for (var stride = WG / 2u; stride > 0u; stride >>= 1u) {
    if (lid < stride) {
      partial[lid] += partial[lid + stride];
    }
    workgroupBarrier();
  }
  if (lid == 0u) {
    result[0] = partial[0];
  }
}
