// Tiled convolution: each 16×16 workgroup computes a 16×16 block of the output.
// It first stages the block's 20×20 input tile (the block plus a 2-pixel halo, with
// rows and columns clamped to the image) in workgroup memory, two loads per
// invocation, so each input pixel is read from storage ~1.6 times instead of up to
// 25. The 25 taps then read workgroup memory with no clamping.

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

const TILE = 16u;
const SIDE = 20u; // TILE + 2R
var<workgroup> tile: array<f32, 400>; // SIDE × SIDE

@compute @workgroup_size(TILE, TILE)
fn main(
  @builtin(workgroup_id) wid: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
  @builtin(local_invocation_index) lidx: u32,
) {
  let w = i32(params.w);
  let ox = i32(wid.x * TILE) - R;
  let oy = i32(wid.y * TILE) - R;
  for (var i = lidx; i < SIDE * SIDE; i += TILE * TILE) {
    let sy = clamp(oy + i32(i / SIDE), 0, w - 1);
    let sx = clamp(ox + i32(i % SIDE), 0, w - 1);
    tile[i] = src[sy * w + sx];
  }
  // No early return before this: every invocation must reach the barrier.
  workgroupBarrier();

  let x = wid.x * TILE + lid.x;
  let y = wid.y * TILE + lid.y;
  if (x >= u32(w) || y >= u32(w)) {
    return;
  }
  var sum = 0.0;
  for (var ky = 0u; ky < u32(K); ky++) {
    let row = (lid.y + ky) * SIDE + lid.x;
    for (var kx = 0u; kx < u32(K); kx++) {
      sum += KERNEL[ky * u32(K) + kx] * tile[row + kx];
    }
  }
  dst[y * u32(w) + x] = sum;
}
