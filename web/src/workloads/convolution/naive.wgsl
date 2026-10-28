// Naive convolution: one invocation per output pixel, reading its 25 taps straight
// from storage with each row and column clamped to the image. x is the column, so
// neighbouring invocations read neighbouring pixels.

struct Params { w: u32 }

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const R = 2;
const K = 5;
/** [1 4 6 4 1]ᵀ[1 4 6 4 1] / 256, row-major (web/src/workloads/convolution/kernel.ts). */
const KERNEL = array<f32, 25>(
  1.0 / 256.0, 4.0 / 256.0, 6.0 / 256.0, 4.0 / 256.0, 1.0 / 256.0,
  4.0 / 256.0, 16.0 / 256.0, 24.0 / 256.0, 16.0 / 256.0, 4.0 / 256.0,
  6.0 / 256.0, 24.0 / 256.0, 36.0 / 256.0, 24.0 / 256.0, 6.0 / 256.0,
  4.0 / 256.0, 16.0 / 256.0, 24.0 / 256.0, 16.0 / 256.0, 4.0 / 256.0,
  1.0 / 256.0, 4.0 / 256.0, 6.0 / 256.0, 4.0 / 256.0, 1.0 / 256.0,
);

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let w = i32(params.w);
  let x = i32(gid.x);
  let y = i32(gid.y);
  if (x >= w || y >= w) {
    return;
  }
  var sum = 0.0;
  for (var ky = 0; ky < K; ky++) {
    let row = clamp(y + ky - R, 0, w - 1) * w;
    for (var kx = 0; kx < K; kx++) {
      sum += KERNEL[ky * K + kx] * src[row + clamp(x + kx - R, 0, w - 1)];
    }
  }
  dst[y * w + x] = sum;
}
