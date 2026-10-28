// The fixed 5×5 kernel shared by the JS implementation and the reference. The same
// weights are written out in wasm/src/convolution.cpp and in the WGSL shaders.

export const RADIUS = 2;
export const KSIZE = 2 * RADIUS + 1;

/** Binomial approximation of a Gaussian: [1 4 6 4 1]ᵀ[1 4 6 4 1] / 256, row-major, exact in f32. */
const BINOMIAL = [1, 4, 6, 4, 1];
export const KERNEL = Float32Array.from({ length: KSIZE * KSIZE }, (_, i) => (BINOMIAL[Math.floor(i / KSIZE)] * BINOMIAL[i % KSIZE]) / 256);
