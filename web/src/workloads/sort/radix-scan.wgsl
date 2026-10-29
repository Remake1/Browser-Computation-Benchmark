// Exclusive prefix sum of `data` in place, for the GPU radix sort's histogram.
// `scan` scans each block of 1024 elements (4 per invocation) and writes the block's
// total to sums[block]; when there is more than one block the host scans `sums` the
// same way and then runs `add`, which adds each block's scanned total back. The
// host recurses until one block covers a level (two levels at n = 2²⁴), and binds a
// 1-element dummy as the top level's `sums`.

@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<storage, read_write> sums: array<u32>;

const WG: u32 = 256u;
const PER: u32 = 4u;

var<workgroup> totals: array<u32, WG>;

fn block_index(wid: vec3u, nwg: vec3u) -> u32 {
  return wid.y * nwg.x + wid.x;
}

@compute @workgroup_size(WG)
fn scan(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(num_workgroups) nwg: vec3u,
  @builtin(local_invocation_index) lid: u32,
) {
  let n = arrayLength(&data);
  let block = block_index(wid, nwg);
  // Spare workgroups of a 2D grid.
  if (block * WG * PER >= n) {
    return;
  }
  let first = (block * WG + lid) * PER;
  var v: array<u32, PER>;
  var total = 0u;
  for (var i = 0u; i < PER; i++) {
    var x = 0u;
    if (first + i < n) {
      x = data[first + i];
    }
    v[i] = total; // exclusive within this invocation
    total += x;
  }

  // Inclusive Hillis–Steele scan of the per-invocation totals.
  totals[lid] = total;
  workgroupBarrier();
  for (var off = 1u; off < WG; off <<= 1u) {
    var x = 0u;
    if (lid >= off) {
      x = totals[lid - off];
    }
    workgroupBarrier();
    totals[lid] += x;
    workgroupBarrier();
  }
  let before = totals[lid] - total;

  for (var i = 0u; i < PER; i++) {
    if (first + i < n) {
      data[first + i] = before + v[i];
    }
  }
  if (lid == WG - 1u && block < arrayLength(&sums)) {
    sums[block] = totals[lid];
  }
}

@compute @workgroup_size(WG)
fn add(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(num_workgroups) nwg: vec3u,
  @builtin(local_invocation_index) lid: u32,
) {
  let n = arrayLength(&data);
  let block = block_index(wid, nwg);
  if (block >= arrayLength(&sums)) {
    return;
  }
  let s = sums[block];
  let first = (block * WG + lid) * PER;
  for (var i = 0u; i < PER; i++) {
    if (first + i < n) {
      data[first + i] += s;
    }
  }
}
