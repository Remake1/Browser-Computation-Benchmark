// GPU heater: keeps every core busy with both ALU work and memory traffic so the
// OS raises the GPU (and memory) clock before measuring (backends/gpu.ts heatGpu).
// The result is written so the compiler can't drop the loop; nobody reads it.

@group(0) @binding(0) var<storage, read> src: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4f>;

const ITERATIONS = 64u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let i = gid.y * nwg.x * 256u + gid.x;
  if (i >= arrayLength(&src)) {
    return;
  }
  var x = src[i];
  for (var k = 0u; k < ITERATIONS; k++) {
    x = x * 0.999 + vec4f(0.5);
  }
  dst[i] = x;
}
