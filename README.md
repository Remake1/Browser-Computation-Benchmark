# BCB-research — browser compute benchmark harness

Measures where WebGPU and WebAssembly overtake JavaScript, per workload and input size. See [PLAN.md](PLAN.md).

## Pinned toolchain

| Tool | Version | Notes |
|---|---|---|
| Emscripten SDK | 6.0.13 | installed repo-locally by `scripts/setup-emsdk.sh` into `.tools/emsdk` |
| Node.js | ≥ 20.19 (tested 20.20.0) | required by Vite 8 |
| pnpm | 11.17.0 | pinned via `packageManager` in `web/package.json` |
| Vite | 8.3.4 | dev server and production build |
| TypeScript | 7.0.2 | type-check only; Vite does the transpiling |
| Nginx | 1.30.6 (`nginx:1.30.6-alpine`) | production server, run with Docker Compose |

Exact JS dependency versions are locked in `web/pnpm-lock.yaml`.

## Setup

```sh
scripts/setup-emsdk.sh          # once: pinned emsdk → .tools/emsdk
cd web
pnpm install --frozen-lockfile
pnpm build:wasm                 # wasm/src/*.cpp → web/public/wasm/<name>-{scalar,simd,threads}.{mjs,wasm}
pnpm check:simd                 # confirms v128 instructions in the simd/threads builds and none in scalar
```

Optional native ceiling (same C++, `clang -O3 -ffp-contract=off`):

```sh
make -C native run              # bin/<workload>-{scalar,simd,threads}: median ms and GFLOP/s or GB/s per size
```

`wasm/Makefile` uses `.tools/emsdk` when it exists and otherwise uses `em++` from `PATH`. To use the SDK in a shell, run `source .tools/emsdk/emsdk_env.sh`.

## Running

Every server sends `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`. Without them, `SharedArrayBuffer` (needed for WASM threads) is unavailable and `performance.now()` is coarsened.

| Purpose | Command | URL |
|---|---|---|
| Development | `pnpm dev` | http://localhost:5173 |
| **Benchmark runs (production)** | `pnpm build && pnpm serve:nginx` | http://localhost:8080 |
| Quick production check without Docker | `pnpm build && pnpm preview` | http://localhost:4173 |

Benchmark numbers must come from the production build served by Nginx (`deploy/nginx/`). The dev server is only for development.

WebGPU needs a secure context. `localhost` counts as one. To test from another device, serve over HTTPS.

## Milestone 0: environment report

Open `env.html` (linked from the runner page). It collects the environment on the main thread and inside the benchmark worker (`web/src/worker.ts`). In the worker it also:

- loads `hello-scalar`, `hello-simd` and `hello-threads`. For each build it checks that the build's features match its flags, that `hello_sum` matches JS bit for bit, and that the threads build actually runs `min(hardwareConcurrency, 8)` pthreads;
- runs one small WebGPU compute dispatch and reads the result back.

The page shows a pass/fail verdict. Add device notes (model, plugged in or on battery), then use **Download env.json** and commit the file under `results/`.

Notes:

- Firefox returns empty `GPUAdapterInfo` strings (vendor, architecture) by default.
- Headless Firefox has no WebGPU adapter, so run it headed.

## Milestone 1: harness core

The runner page (`index.html`) runs the selected workloads in a fresh benchmark worker. It shows a summary for each record as results come in, and offers **Download JSON** (run metadata plus every raw sample) and **Download CSV** (one row per sample). **Stop** terminates the worker and keeps the records that already finished (`complete: false`).

The protocol is implemented in `web/src/harness/runner.ts` (PLAN.md §4):

- Inputs come from `deriveSeed(seed, workload, 'input', size)`, so they don't depend on run order. The implementation order is shuffled per size with its own derived seed and recorded as `order`.
- `init()` runs once per implementation per run, the first time it is used, and is recorded as `initMs`. Throwing `Unsupported` marks the implementation `unsupported` for every size.
- One iteration is `setup → run → readback` (timed as upload / compute / readback), followed by an untimed `teardown()`. Every iteration starts from freshly uploaded input, so in-place kernels such as sort don't need a `reset()`. After the last iteration at a size the runner calls `release()` (untimed).
- WebGPU implementations allocate their buffers (including the readback staging buffer) on the first `setup()` at a size and reuse them until `release()`; later `setup()` calls only `writeBuffer` the input. This matches WASM, whose `malloc`/`free` per iteration reuses heap pages that are already touched. Verification runs on the first iteration, which always uses freshly allocated buffers, so a stale result from an earlier iteration can't pass.
- The first iteration is verified (`invalid` if wrong) and stored as `firstRun`. If it takes longer than `timeoutMs`, the record is `timeout`. After a timeout or an error, larger sizes are recorded as `skipped`.
- Warmup runs for `warmupRuns` iterations or `warmupBudgetMs`, whichever comes first. If an iteration is shorter than `minSampleMs`, each sample averages `k` iterations (`repeats`, capped at 1000). `k` is calibrated by timing back-to-back iterations as one span until it reaches `minSampleMs`. Individually timed phases are each rounded to the 5 µs timer resolution, so estimating from them made `k` jump between 1000, 200 and 100, or fall to 40 when the warmup median was JIT-cold.
- Sampling collects at least `minSamples` and at most `maxSamples`, and stops between the two when `sampleBudgetMs` runs out.
- WebGPU implementations share one device, created before measuring (`gpu.deviceInitMs`). They report GPU-side time through `gpuTimeMs()` using `GpuTimer` (timestamp-query). The observed quantization of the timestamps is recorded as `gpu.timestampResolutionNs`; Chrome rounds to 65,536 ns by default (1 ns with `--enable-webgpu-developer-features`).
- **GPU clock warmup.** After the verified first iteration, each WebGPU implementation runs `heatGpu` (`web/src/backends/gpu.ts`, `heat.wgsl`) for `gpuWarmupMs` (default 200 ms) before its warmup iterations. It keeps the GPU saturated with back-to-back ALU + memory dispatches (three submissions in flight) so the OS raises the GPU clock. This time doesn't count against `warmupBudgetMs`, and `firstRun` stays a cold measurement. Without it, the GPU sits mostly idle between short, CPU-synchronized iterations and macOS keeps it at a low clock, then switches mid-record: in a plugged-in run (Chrome 156, M4 Pro, 1 ns timestamps), the kernel's own timestamp jumped between two levels in 38 of 134 GPU records, by exactly 4.6× in N-body (e.g. 0.60 → 0.13 ms at n = 1536 after 16 samples) and 3.5–4.5× in the memory-bound kernels. Medians then depended on which clock state happened to dominate, and compute IQRs reached 145 %. In an A/B (Chrome 155 headless, GPU implementations only, two alternating runs per setting; reduction ≥ 2¹⁷, convolution ≥ 1024, N-body ≥ 384), records whose GPU timestamps were bimodal (p90/p10 > 1.8) fell from 30 and 20 of 80 to 0 and 2 of 80 (both borderline at 1.8), records with compute IQR > 25 % fell from 21 and 13 to 8 and 7, and medians became reproducible across runs (reduction 2²⁶ `gpu-opt` kernel: 1.61 / 1.67 ms without, 1.205 / 1.206 ms with). The remaining spread is in the CPU-side part of the compute phase (submit to `onSubmittedWorkDone`), not in the kernel. So GPU numbers are steady-state, high-clock numbers; a page that issues occasional short GPU bursts may see its kernels run up to ~4.6× slower.

The `dummy` workload (an f32 sum through JS, the three `hello` WASM builds, and a single-workgroup WGSL kernel) exists only to exercise the harness.

## Milestone 2: matrix multiply

`web/src/workloads/matmul/` computes C = A·B for N×N row-major `f32` matrices, N = 16…2048, with 1.5× steps between powers of two. Inputs are uniform in [−1, 1).

| Implementation | What it is |
|---|---|
| `js` | i-k-j loops over `Float32Array` |
| `wasm-scalar`, `wasm-simd` | `wasm/src/matmul.cpp`, the same i-k-j loops. In the SIMD build the inner loop auto-vectorizes to `f32x4` (`pnpm check:simd`) |
| `wasm-threads` | the same loops, with contiguous blocks of rows split across a persistent pool (`wasm/include/threadpool.hpp`) of `min(hardwareConcurrency, 8)` threads, created in `init()` |
| `gpu-naive` | `naive.wgsl`: one invocation per element of C, 16×16 workgroups |
| `gpu-opt` | `tiled.wgsl`: 16×16 tiles of A and B staged in workgroup memory |

Verification (`reference.ts`) compares against an f64 reference using the componentwise bound `|C − R| ≤ 1e-3 · (|A|·|B|)`, which stays meaningful for entries near zero. The reference is computed once per input. Above about 2²⁸ multiply-adds it checks a subset of rows: the first and last 16, plus an even spread of the rest (64 rows at N = 2048). The UI shows GFLOP/s (2N³ / median compute time) for workloads that define `flops()`.

`native/` builds the same `matmul.cpp` natively for comparison. Apple M4 Pro, median GFLOP/s:

| N | scalar | simd (NEON) | threads (8) |
|---|---|---|---|
| 256 | 7.7 | 33 | 202 |
| 1024 | 8.3 | 33 | 251 |
| 2048 | 8.3 | 33 | 249 |

**Caveat: Chrome on macOS can run workers on the efficiency cores.** On battery, Chrome 151 ran dedicated workers, including the pthread workers of `wasm-threads`, on the M4 Pro's 4 E-cores. The main thread stayed on the P-cores. The same JS kernel ran at 2.2 GFLOP/s on the page's main thread and 0.3–0.4 in a worker. Combined throughput of 1–8 workers capped at about 4 cores. So CPU numbers measured this way are E-core numbers. This affects CPU numbers only (GPU work is not run on worker threads), and the env notes should say whether the machine was plugged in. On mains power the worker reaches native-like speeds: JS 2.8 GFLOP/s (Node: 2.75), wasm-scalar 8.0 (native: 8.3), wasm-simd 24.5 (native: 33). **Collect benchmark data plugged in.**

## Milestone 3: reduction

`web/src/workloads/reduction/` sums an `f32[n]`, n = 2¹⁰…2²⁶, with 1.5× steps between powers of two. The UI shows GB/s (4n bytes / median compute time) for workloads that define `bytes()`, which is the natural unit for this memory-bound kernel.

| Implementation | What it is |
|---|---|
| `js` | the simple loop over `Float32Array` |
| `wasm-scalar` | `wasm/src/reduction.cpp`, the same simple loop with one `f32` accumulator. Without `-ffast-math` the compiler can't reassociate, so it is bound by add latency |
| `wasm-simd` | hand-written `wasm_simd128.h` intrinsics: four `f32x4` accumulators (16 lanes), then a fixed-order horizontal sum and scalar tail |
| `wasm-threads` | the SIMD kernel on contiguous chunks across the persistent pool; the caller adds the per-thread partial sums in thread order |
| `gpu-naive` | `naive.wgsl`: multi-pass tree reduction, one element per invocation, 256 → 1 per pass in workgroup memory (4 passes at 2²⁶). Above 65,535 workgroups the dispatch becomes 2D |
| `gpu-opt` | `opt.wgsl`: at most 1024 workgroups stride over the input with `vec4` loads, so each invocation sums many elements in registers, then one workgroup-level sum: two passes, or one when a single workgroup covers the input. The workgroup sum uses `subgroupAdd` when the device has `subgroups` (`opt-subgroups.wgsl`, recorded in the env adapter features), otherwise a workgroup-memory tree (`opt-tree.wgsl`) |

All passes of a GPU reduction are dispatches in one compute pass, so `timestamp-query` covers them all.

Buffer reuse (above) was measured on this workload (Chrome 155, M4 Pro, GPU only, n = 2²⁴…2²⁶, two alternating runs each). Allocating per iteration made the medians slower and noisier: end-to-end 7.4–8.7 ms with reuse vs. 9.5–25.9 ms without at 2²⁴, and 29.8–34.0 vs. 37.5–75.9 ms at 2²⁶. At 2²⁴–2²⁵ the timestamp-measured kernel time also rose (`gpu-naive` up to 2.2× and `gpu-opt` up to 2.4× in the worst run). The compute phase still runs about 0.3 ms longer than the GPU timestamps at small n, and about 1.0–1.4 ms at 2²⁶ with or without reuse. That part is not explained by allocation. A per-submit cost that grows with the size of the bound buffers (for example Metal residency) would fit, but this is untested.

Methodology decisions:

- **Inputs are multiples of 1/256 in (−1, 1), symmetric around 0.** Every partial sum, in any order, stays far below 2¹⁶, so every `f32` addition is exact and every implementation must return **exactly** the f64 Kahan reference. A relative tolerance couldn't catch a dropped or duplicated element at large n: one element is ~1e-8 of Σ|x| at 2²⁶, below the legitimate rounding error of a sequential `f32` sum. The values don't affect speed (no subnormals). Uniform [0, 1) inputs would not work at all: a sequential `f32` sum stops growing at 2²⁴.
- **The JS accumulator is a JS number (f64).** `s = Math.fround(s + x[i])` puts f64→f32→f64 conversions on the loop-carried chain and ran 3.2× slower in V8 (Node 20: 2.0 vs 0.62 ns/element), while a CPU's f64 add has the same latency as its f32 add. The data is still `f32`, and with these inputs the result is identical.
- **The scalar → SIMD gap includes ILP**, not just vector width: the simple loop runs one add per add latency, and the SIMD kernel's four accumulators remove that chain. Without reassociation the scalar loop can't be split, so this is inherent to "simple loop vs. intrinsics" as specified.
- Odd lengths (1, 3, 5, 17, 255, 257, 1023, 4097, 65537, 262147, 1048577, 2²⁴+3, 2²⁶+3) were verified for all six implementations, with both `gpu-opt` workgroup-sum variants, in Chrome 155. The sweep's sizes are all multiples of 512 and don't exercise those tails.

`native/` builds the same `reduction.cpp` (`NATIVE_SIMD` swaps the wasm intrinsics for clang vector extensions, i.e. NEON). Apple M4 Pro, median GB/s:

| n | scalar | simd (NEON) | threads (8) |
|---|---|---|---|
| 2¹⁰ | 8.1 | 104 | 0.18 (≈ 23 µs pool wake-up) |
| 2¹⁸ | 7.2 | 106 | 40 |
| 2²² | 7.2 | 106 | 308 |
| 2²⁶ | 7.2 | 95 | 244 (DRAM; 273 GB/s peak) |

## Milestone 3: convolution

`web/src/workloads/convolution/` convolves a W×W grayscale `f32` image with a fixed 5×5 Gaussian (the binomial kernel [1 4 6 4 1]ᵀ[1 4 6 4 1] / 256), clamping at the edges, for W = 64…4096 with 1.5× steps between powers of two. The UI shows GFLOP/s (50 W², i.e. 25 multiply-adds per pixel) and GB/s (8 W²: each pixel read once and written once).

| Implementation | What it is |
|---|---|
| `js` | direct 2D loops over `Float32Array`: per output row the five source rows are clamped once, interior columns sum the 25 taps (nested `ky`/`kx` loops over the kernel) without clamping, and only the two border columns on each side clamp per tap |
| `wasm-scalar`, `wasm-simd` | `wasm/src/convolution.cpp`, the same loops. In the SIMD build the 25 taps unroll and the interior loop over x auto-vectorizes to `f32x4` (25 `f32x4.mul` + 25 `f32x4.add`; `pnpm check:simd`). Each output is still summed in tap order, so no reassociation is needed |
| `wasm-threads` | the same loops, with contiguous blocks of output rows split across the persistent pool |
| `gpu-naive` | `naive.wgsl`: one invocation per pixel, 16×16 workgroups, 25 clamped reads from storage |
| `gpu-opt` | `tiled.wgsl`: each 16×16 workgroup stages its 20×20 input tile (block + 2-pixel halo, clamped) in workgroup memory, then reads the 25 taps from there |

Methodology decisions:

- **Pixels are multiples of 1/256 in [0, 1) (8-bit grayscale).** The weights are multiples of 1/256 too, so every product and partial sum is a multiple of 2⁻¹⁶ below 1 and every `f32` operation is exact, in any tap order and with or without FMA. Verification (`reference.ts`) therefore requires **exact** equality with an independent f64 per-tap clamped reference, over every pixel, which is stricter than PLAN.md's 1e-4 relative tolerance.
- **JS keeps the nested `ky`/`kx` loops of the C++ source.** Clang unrolls the 25 taps; V8 doesn't. Unrolling the `kx` loop by hand in JS was 1.9× faster in Node 20 (W = 2048: 110 → 59 ms, 1.9 → 3.5 GFLOP/s), still about 4× short of `wasm-scalar`. The baseline stays the same source as the C++, so the JIT's lack of unrolling is part of what is measured.
- Odd sizes (1, 2, 3, 4, 5, 7, 17, 33, 100, 257, 1000, 4099: smaller than the kernel, partial GPU tiles, uneven thread splits) were verified for all six implementations in Chrome 155. The sweep's sizes are all multiples of 32 and don't exercise those cases.
- In a same-session A/B (Chrome 155, M4 Pro, timestamp-query), `gpu-opt` beat `gpu-naive` by 10–20 % at W ≥ 3072 (0.46 vs. 0.58 ms at 3072; 0.96–1.01 vs. 1.14–1.16 ms at 4096) and by 5–8 % at 2048 (0.20 vs. 0.21–0.22 ms). Apple GPUs cache the naive kernel's overlapping reads well, so staging in workgroup memory gains little. On battery, the GPU sometimes stayed in a low-clock state for a whole run (3.5 ms at W = 4096 for either kernel), which is another reason to collect data plugged in.

`native/` builds the same `convolution.cpp`. Every size, plus the odd sizes above, matches the reference exactly. Apple M4 Pro, median GFLOP/s:

| W | scalar | simd (NEON) | threads (8) |
|---|---|---|---|
| 64 | 17.9 | 43 | 8.3 (≈ 25 µs, mostly pool wake-up) |
| 512 | 17.9 | 51 | 203 |
| 2048 | 17.8 | 50 | 367 |
| 4096 | 17.8 | 45 | 323 |

## Milestone 3: N-body

`web/src/workloads/nbody/` computes one step of an n-body simulation, n = 64…65536 with 1.5× steps between powers of two. Bodies are `vec4` (x, y, z, mass), with positions uniform in [−1, 1)³ and masses in [0.5, 1). The output is each body's softened gravitational acceleration, a_i = Σ_j m_j (p_j − p_i) / (|p_j − p_i|² + ε²)^{3/2} with G = 1 and ε² = 0.01, as `vec4` (ax, ay, az, 0). The sum runs over every j including i, whose term is exactly 0 thanks to the softening, so no implementation branches in the inner loop. The UI shows GFLOP/s at the conventional 20 flops per interaction (20 n²).

| Implementation | What it is |
|---|---|
| `js` | the double loop over `Float32Array`, `1 / Math.sqrt(d2)` |
| `wasm-scalar` | `wasm/src/nbody.cpp`, the same double loop in `f32` |
| `wasm-simd` | hand-written `wasm_simd128.h` intrinsics over four bodies i at a time (one per lane), broadcasting body j (`v128.load32_splat`), with `f32x4.sqrt` and `f32x4.div`; scalar tail for n mod 4 |
| `wasm-threads` | the SIMD kernel on contiguous blocks of bodies across the persistent pool |
| `gpu-naive` | `naive.wgsl`: one invocation per body, 128-wide workgroups, every body j read from storage |
| `gpu-opt` | `tiled.wgsl`: the workgroup stages bodies j in tiles of 128 in workgroup memory (one load per invocation), so each body is read from storage once per workgroup. The last tile is padded with zero-mass bodies, whose terms are exactly 0 |

Methodology decisions:

- **"One step" is one force evaluation; the O(n) integration (v += a·dt, p += v·dt) is left out.** It costs nothing next to the O(n²) loop, and verifying positions after one small step would test the force loop only weakly (the change is tiny relative to the position).
- **SIMD vectorizes over bodies i, not over the inner j loop.** The j loop is a sum, which doesn't vectorize without `-ffast-math`. Across i, each lane sums over j in the same order as the scalar loop with the same correctly rounded `sqrt` and division, so `wasm-simd` and `wasm-scalar` produce identical bits (natively too). The auto-vectorizer doesn't do this outer-loop vectorization, hence the intrinsics.
- Every implementation computes `1 / sqrt(d2)`, including the shaders (no `inverseSqrt`), so all of them run the same formula.
- **Verification** (`reference.ts`) uses a componentwise bound like matmul's: |a − R| ≤ 1e-3 · Σ_j |term_j| per component, and w must be 0. Natively, the measured error is at most 8.5e-6 of this bound at n = 16384. The f64 reference is O(n²), so above 2²⁷ interactions it checks a subset of bodies: the first and last 16 plus an even spread (2,048 bodies at n = 65536).
- Odd sizes (1, 2, 3, 5, 7, 33, 100, 129, 1001, 4097, 13001: SIMD tails, partial workgroups and tiles, uneven thread splits, and the subset verification) were verified for all six implementations in Chrome 155.

First sweep (Chrome 155 headless, M4 Pro on battery, one run; median compute GFLOP/s, end-to-end ms in parentheses where the GPU is concerned):

| n | js | wasm-scalar | wasm-simd | wasm-threads | gpu-naive | gpu-opt |
|---|---|---|---|---|---|---|
| 256 | 12.7 | 14.8 | 47 | 31 | 4.1 (0.53 ms) | 3.7 (0.58 ms) |
| 1024 | 12.1 | 13.1 | 47 | 174 | 32 (0.86 ms) | 50 (0.65 ms) |
| 4096 | 12.5 | 13.8 | 46 | 330 | 520 (0.89 ms) | 780 (0.68 ms) |
| 16384 | 12.3 | 13.9 | 46 | 348 | 1,980 | 3,020 |
| 65536 | timeout (6.9 s) | timeout (6.3 s) | 45 | 348 | 2,580 | 3,720 |

N-body is compute-bound with O(n) data. Upload and readback are tiny, so the GPU's fixed cost of ~0.5 ms per iteration (submit + `onSubmittedWorkDone` + map) is the whole story at small n. JS is within 15 % of `wasm-scalar` here: the loop is bound by `sqrt` and division latency, which V8's f64 code pays just like the f32 WASM.

`native/` builds the same `nbody.cpp` (`NATIVE_SIMD` swaps the wasm intrinsics for clang vector extensions). Every size up to 16384 and the odd sizes pass the same bound. Apple M4 Pro, median GFLOP/s:

| n | scalar | simd (NEON) | threads (8) |
|---|---|---|---|
| 64 | 14.8 | 51 | 3.4 (≈ 24 µs, mostly pool wake-up) |
| 1024 | 16.0 | 52 | 256 |
| 16384 | 15.9 | 52 | 395 |
| 65536 | 15.5 | 51 | 384 |
