# BCB-research — browser compute benchmark harness

Measures where WebGPU and WebAssembly overtake JavaScript, per workload and input size. See [PLAN.md](PLAN.md).

## Pinned toolchain

| Tool | Version | Notes |
|---|---|---|
| Emscripten SDK | 6.0.13 | installed repo-locally by `scripts/setup-emsdk.sh` into `.tools/emsdk` |
| Node.js | ≥ 22.18 (tested 24.16.0) | Vite 8 needs ≥ 20.19; `pnpm bench` runs TypeScript through Node's built-in type stripping (≥ 22.18) |
| pnpm | 11.17.0 | pinned via `packageManager` in `web/package.json` |
| Vite | 8.3.4 | dev server and production build |
| TypeScript | 7.0.2 | type-check only; Vite does the transpiling |
| Playwright | 1.64.1 | `pnpm bench`: drives the installed Google Chrome and Playwright's Firefox 157 build (`playwright firefox v1555`) |
| Nginx | 1.30.6 (`nginx:1.30.6-alpine`) | production server, run with Docker Compose |

Exact JS dependency versions are locked in `web/pnpm-lock.yaml`.

## Setup

```sh
scripts/setup-emsdk.sh          # once: pinned emsdk → .tools/emsdk
cd web
pnpm install --frozen-lockfile
pnpm build:wasm                 # wasm/src/*.cpp → web/public/wasm/<name>-{scalar,simd,threads}.{mjs,wasm}
pnpm check:simd                 # confirms v128 instructions in the simd/threads builds and none in scalar
pnpm exec playwright install firefox   # once, for pnpm bench (Chrome is the installed Google Chrome)
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
| **Automated benchmark runs (Chrome, Firefox)** | `pnpm bench` (milestone 5) | |
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

Sweep (Chrome 156, M4 Pro plugged in, 8 threads, 1 ns timestamps, one run; `results/results-chrome-2026-10-29T17-52-40.json`). Median compute GFLOP/s, with end-to-end ms and GPU timestamp ms in parentheses for the GPU:

| N | js | wasm-scalar | wasm-simd | wasm-threads | gpu-naive | gpu-opt |
|---|---|---|---|---|---|---|
| 64 | 2.8 | 7.5 | 20.7 | 10.4 | 1.7 (0.53 / 0.0095 ms) | 1.7 (0.52 / 0.0046 ms) |
| 256 | 2.7 | 7.9 | 24.9 | 138 | 82.6 (0.77 / 0.11 ms) | 96.6 (0.71 / 0.046 ms) |
| 512 | 2.7 | 7.9 | 24.2 | 179 | 273 (1.8 / 0.72 ms) | 459 (1.4 / 0.33 ms) |
| 1024 | 2.7 | 7.6 | 24.2 | 185 | 350 (7.2 / 5.6 ms) | 667 (4.3 / 2.6 ms) |
| 2048 | timeout (6.4 s) | 7.4 | 23.0 | 167 | 385 (50 / 44 ms) | 816 (26 / 21 ms) |

- **The WASM builds reach 70–96 % of native:** `wasm-scalar` 7.4–8.0 GFLOP/s (native 8.3), `wasm-simd` 23–27 (native 33), `wasm-threads` up to 185 at N = 768–1024 (native 251). JS runs at a third of `wasm-scalar`.
- **`gpu-opt` beats `wasm-threads` from N = 384 (compute) or 512 (end-to-end)** and reaches 816 GFLOP/s at 2048, 4.9× `wasm-threads`. Its kernel is 2.1–2.2× faster than `gpu-naive`'s from N = 512 on (20.7 vs. 44.2 ms at 2048).
- **Up to N = 64 the GPU is all fixed cost:** about 0.5 ms end-to-end, of which ~0.29 ms is the compute phase (submit to `onSubmittedWorkDone`) around a kernel of under 10 µs.

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

Sweep (Chrome 156, M4 Pro plugged in, 8 threads, 1 ns timestamps, one run; `results/results-chrome-2026-10-29T17-52-40.json`). Median compute GB/s, with end-to-end ms and GPU timestamp ms in parentheses for the GPU:

| n | js | wasm-scalar | wasm-simd | wasm-threads | gpu-naive | gpu-opt |
|---|---|---|---|---|---|---|
| 2¹⁰ | 4.8 | 6.3 | 24.3 | 0.19 | 0.015 (0.47 / 0.0065 ms) | 0.015 (0.47 / 0.0037 ms) |
| 2¹⁴ | 5.8 | 7.8 | 80.8 | 2.6 | 0.24 (0.49 / 0.0074 ms) | 0.24 (0.49 / 0.0065 ms) |
| 2¹⁸ | 5.6 | 7.5 | 90.4 | 32.0 | 3.4 (0.77 / 0.036 ms) | 3.7 (0.73 / 0.0074 ms) |
| 2²⁰ | 5.7 | 7.3 | 90.2 | 102 | 7.8 (0.99 / 0.11 ms) | 10.0 (0.83 / 0.012 ms) |
| 2²² | 5.7 | 7.3 | 93.2 | 183 | 24.1 (2.6 / 0.41 ms) | 52.4 (2.2 / 0.045 ms) |
| 2²⁴ | 5.9 | 7.3 | 81.3 | 210 | 29.8 (9.0 / 1.9 ms) | 122 (7.8 / 0.25 ms) |
| 2²⁶ | 6.0 | 7.4 | 80.4 | 166 | 37.5 (37 / 5.8 ms) | 90.6 (35 / 1.7 ms) |

- **The CPU side matches native:** `wasm-scalar` (7.3–7.8 GB/s) is bound by add latency like the native build (7.2), and JS (5.7–6.0) is not far behind. `wasm-simd` reaches 80–93 GB/s (native 95–106), and `wasm-threads` passes it at 2²⁰ and peaks at 213 GB/s at 2²³ (native 308 at 2²²).
- **The compute phase hides the GPU kernel.** It never drops below ~0.27 ms, so the GB/s above lag far behind the timestamps: `gpu-opt`'s kernel sums 2²⁴ floats in 0.25 ms (274 GB/s, the M4 Pro's DRAM peak) and 2²² in 0.045 ms (370 GB/s, presumably from the system-level cache), while the compute phase takes 0.55 and 0.32 ms. Measured by the compute phase, `gpu-opt` never beats `wasm-threads` and beats `wasm-simd` only at some sizes from 3·2²² on, where the compute phase becomes erratic (0.55 ms at 2²⁴, 2.6 ms at 2²⁵). `gpu-naive`'s kernel is 3–9× slower than `gpu-opt`'s from 2²² on.
- **End-to-end the GPU never beats `wasm-simd`.** Uploading the input runs at ~9 GB/s (30 ms for 256 MB at 2²⁶), ten times longer than `wasm-simd` takes to sum it.

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

Sweep (Chrome 156, M4 Pro plugged in, 8 threads, 1 ns timestamps, one run; `results/results-chrome-2026-10-29T17-52-40.json`). Median compute GFLOP/s, with end-to-end ms and GPU timestamp ms in parentheses for the GPU:

| W | js | wasm-scalar | wasm-simd | wasm-threads | gpu-naive | gpu-opt |
|---|---|---|---|---|---|---|
| 64 | 2.7 | 14.8 | 38.4 | 6.9 | 0.77 (0.47 / 0.0029 ms) | 0.76 (0.47 / 0.0027 ms) |
| 256 | 2.9 | 15.3 | 43.1 | 58.2 | 12.1 (0.55 / 0.0055 ms) | 12.2 (0.55 / 0.0056 ms) |
| 1024 | 2.9 | 15.4 | 44.4 | 244 | 113 (1.2 / 0.052 ms) | 115 (1.2 / 0.051 ms) |
| 2048 | 2.9 | 15.4 | 44.1 | 294 | 378 (4.2 / 0.21 ms) | 392 (4.1 / 0.20 ms) |
| 3072 | 2.9 | 15.4 | 44.8 | 322 | 486 (8.2 / 0.58 ms) | 581 (8.1 / 0.46 ms) |
| 4096 | 2.9 | 14.3 | 42.3 | 314 | 556 (14.8 / 1.0 ms) | 491 (13.9 / 0.86 ms) |

- **The WASM builds reach about 85 % of native** (`wasm-scalar` 14.3–15.5 GFLOP/s vs. 17.8, `wasm-simd` 36–45 vs. 43–51, `wasm-threads` up to 322 vs. 367). JS stays at 2.7–2.9 (no unrolling, see above).
- **The GPU kernels confirm the A/B above:** `gpu-opt` is 20 % faster than `gpu-naive` at W = 3072 and 17 % at 4096 by timestamps, and within 5 % at W = 2048 and below (except at 1536, where both kernels' timestamps are noisy). At 4096 its kernel runs at 970 GFLOP/s. The compute-phase medians at W = 1024 and 4096 are noisy (IQR ~40 %), which reverses the order of the two kernels at 4096 in the table; the timestamps are the reliable comparison.
- **Transfers decide end-to-end.** At W = 4096 the GPU spends 6.2 ms uploading and 6.0 ms reading back 64 MB each, against a 0.86 ms kernel. By the compute phase the GPU beats `wasm-threads` from W = 2048; end-to-end it never does (13.9 vs. 2.7 ms at 4096). It beats `wasm-simd` from 512 (compute) or 1024 (end-to-end).

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

Sweep (Chrome 156, M4 Pro plugged in, 8 threads, 1 ns timestamps, one run; `results/results-chrome-2026-10-29T17-52-40.json`). Median compute GFLOP/s, with end-to-end ms and GPU timestamp ms in parentheses for the GPU:

| n | js | wasm-scalar | wasm-simd | wasm-threads | gpu-naive | gpu-opt |
|---|---|---|---|---|---|---|
| 256 | 12.4 | 13.8 | 44.8 | 30.5 | 4.7 (0.48 / 0.022 ms) | 4.7 (0.48 / 0.0096 ms) |
| 1024 | 12.4 | 14.1 | 45.3 | 174 | 60.1 (0.55 / 0.086 ms) | 71.1 (0.50 / 0.031 ms) |
| 4096 | 12.5 | 14.2 | 45.4 | 331 | 527 (0.88 / 0.34 ms) | 847 (0.62 / 0.13 ms) |
| 16384 | 12.3 | 14.1 | 45.5 | 350 | 1,980 (3.0 / 2.4 ms) | 3,020 (2.1 / 1.5 ms) |
| 65536 | timeout (6.9 s) | timeout (6.0 s) | 45.7 | 350 | 2,640 (33 / 32 ms) | 3,890 (23 / 22 ms) |

N-body is compute-bound with O(n) data. Upload and readback are tiny, so the GPU's fixed cost of ~0.5 ms per iteration (submit + `onSubmittedWorkDone` + map) is the whole story at small n. JS is within 15 % of `wasm-scalar` here: the loop is bound by `sqrt` and division latency, which V8's f64 code pays just like the f32 WASM. The WASM builds reach about 90 % of native (`wasm-simd` 45.5 vs. 52, `wasm-threads` 350 vs. 384–395). `gpu-opt` beats `wasm-simd` from n = 1024 (compute) or 1536 (end-to-end) and `wasm-threads` from 3072, and runs 11× faster than `wasm-threads` at 65536. Its kernel is 1.5–2.8× faster than `gpu-naive`'s.

`native/` builds the same `nbody.cpp` (`NATIVE_SIMD` swaps the wasm intrinsics for clang vector extensions). Every size up to 16384 and the odd sizes pass the same bound. Apple M4 Pro, median GFLOP/s:

| n | scalar | simd (NEON) | threads (8) |
|---|---|---|---|
| 64 | 14.8 | 51 | 3.4 (≈ 24 µs, mostly pool wake-up) |
| 1024 | 16.0 | 52 | 256 |
| 16384 | 15.9 | 52 | 395 |
| 65536 | 15.5 | 51 | 384 |

## Milestone 4: sort

`web/src/workloads/sort/` sorts a `u32[n]` ascending, n = 2¹⁰…2²⁴, with 1.5× steps between powers of two. Keys are uniform u32. The UI shows M items/s (here Mkeys/s: n / median compute time) for workloads that define `items()`.

| Implementation | What it is |
|---|---|
| `js-builtin` | `Uint32Array.prototype.sort()`: engine code, not JS written here |
| `js-radix` | hand-written LSD radix sort: 8-bit digits, one counting pass for all four histograms, then four stable scatter passes ping-ponging between the array and a scratch buffer |
| `wasm-std` | `std::sort` (libc++ introsort) from `wasm/src/sort.cpp`, scalar build |
| `wasm-scalar`, `wasm-simd` | `wasm/src/sort.cpp`, the same radix sort as `js-radix`. It has no vector form (see below), so the simd build is the same code with `-msimd128` |
| `wasm-threads` | parallel LSD radix sort on contiguous chunks across the persistent pool: per pass, every thread counts its chunk, the caller turns the counts into offsets in (digit, thread) order, every thread scatters its chunk. Stable, so the output is identical |
| `gpu-naive` | `bitonic.wgsl`: global-memory bitonic sort, keys padded to a power of two with `0xFFFFFFFF`, one dispatch per (k, j) step: log₂N·(log₂N + 1)/2 dispatches (300 at 2²⁴), each through the whole array |
| `gpu-opt` | `radix-{count,scan,scatter}.wgsl`: LSD radix sort, 4-bit digits, 8 passes of count → scan → scatter over 1024-key blocks, keys padded to a whole block with `0xFFFFFFFF`. The scatter ranks keys stably with a workgroup scan of per-invocation digit counts packed 16 bits per digit into two `vec4u` |

All dispatches of a GPU sort are in one compute pass, so `timestamp-query` covers them all (40 dispatches per `gpu-opt` sort at 2²⁴: 8 × count, scan, scan, add, scatter).

Methodology decisions:

- **The CPU implementations share one algorithm, LSD radix sort, plus two library sorts as reference points.** PLAN.md §7 lists `std::sort` for the C++ scalar/SIMD builds and parallel radix for threads. Following that literally would make `wasm-threads` vs. `wasm-simd` an algorithm comparison as well as a threading one, and leave `js-radix` with no WASM counterpart. So `wasm-scalar`/`wasm-simd`/`wasm-threads` all run the radix sort of `js-radix`, and `std::sort` is measured separately as `wasm-std`. `js-builtin` vs. `wasm-std` compares the engine's own sort with a C++ library sort compiled to WASM.
- **`wasm-simd` is not vectorized.** Radix sort's work is a histogram increment and a scatter per key, and simd128 has neither gather nor scatter (the increments of one vector's lanes can also hit the same counter). The compiler emits no `v128` instruction for this source, so `check-simd` exempts sort (`NO_SIMD` in `wasm/Makefile`). The variant is kept because `wasm-threads` uses the same flags and it is its single-thread baseline.
- **Serial radix counts once, parallel radix counts every pass.** The serial versions build all four histograms in one read of the input. Per-thread chunk histograms change after every pass, so the threads build counts each pass again: 4 reads for counting instead of 1.
- **Sorting is in place, so every implementation copies the input in `setup()`** (the upload phase), including the JS ones, and the radix scratch buffer is allocated there too. The compute phase is the sort alone.
- **The GPU radix sort is reduce-then-scan, not single-pass.** Single-pass "decoupled look-back" sorts (Onesweep) need forward-progress guarantees between workgroups that WebGPU doesn't give. Reduce-then-scan needs no communication between the workgroups of a dispatch. 4-bit digits keep the histogram (16 × n/1024 entries) to a two-level scan at 2²⁴ and fit 16 counters into the scatter's packed scan.
- **Padding.** Both GPU sorts write the `0xFFFFFFFF` padding once, when the buffers are allocated. A sort leaves the largest values at the end, which are all `0xFFFFFFFF` because the padding alone provides that many, so the padding is intact for the next iteration's `setup()`.
- **Verification** (`reference.ts`) requires the output to equal the input sorted by the engine's `Uint32Array` sort, element for element. That checks order and multiset at once. `js-builtin` is checked against itself.
- Odd sizes (1, 2, 3, 4, 5, 7, 17, 255, 256, 257, 1000, 1023, 1024, 1025, 1536, 4097, 65537) with uniform, all-zero, all-`0xFFFFFFFF`, sorted, reversed, few-distinct (`x & 0x0300000f`) and half-`0xFFFFFFFF` keys, plus 2²⁰+1, 2²⁴ and 2²⁴+3 with uniform and half-`0xFFFFFFFF` keys, were verified for all eight implementations in Chrome 156, two iterations each so the second runs on reused GPU buffers. The input was checked to be unchanged afterwards.

Sweep (Chrome 156, M4 Pro plugged in, 8 threads, 1 ns timestamps, one run; `results/results-chrome-2026-10-29T17-52-40.json`). Median compute Mkeys/s, with end-to-end ms and GPU timestamp ms in parentheses for the GPU:

| n | js-builtin | js-radix | wasm-std | wasm-scalar | wasm-simd | wasm-threads | gpu-naive | gpu-opt |
|---|---|---|---|---|---|---|---|---|
| 2¹⁰ | 182 | 192 | 118 | 327 | 324 | 5.1 | 2.6 (0.59 / 0.099 ms) | 2.8 (0.56 / 0.075 ms) |
| 2¹⁴ | 124 | 222 | 80.7 | 359 | 360 | 69.1 | 30.3 (0.76 / 0.22 ms) | 45.2 (0.59 / 0.076 ms) |
| 2¹⁸ | 67.6 | 219 | 44.8 | 393 | 393 | 530 | 79.0 (3.9 / 2.8 ms) | 425 (1.2 / 0.31 ms) |
| 2²⁰ | 62.2 | 220 | 40.3 | 370 | 370 | 1,170 | 90.2 (12.5 / 11.0 ms) | 763 (2.2 / 0.87 ms) |
| 2²² | 59.8 | 217 | 37.5 | 342 | 342 | 1,710 | 83.4 (54 / 50 ms) | 1,160 (7.4 / 3.2 ms) |
| 2²⁴ | 56.7 | 210 | 34.9 | 316 | 317 | 1,740 | 66.7 (265 / 251 ms) | 1,300 (26 / 12.4 ms) |

The first sweep (headless, plugged in, 65,536 ns timestamps; `results/results-chrome-2026-10-29T03-28-10.json`) gave the same picture and the same GPU and `wasm-threads` crossovers.

- **The radix sorts are flat in n, the comparison sorts fall off** as log n grows: `js-builtin` from 182 to 57 Mkeys/s, `wasm-std` from 118 to 35. `js-radix` beats `js-builtin` at every size, by 3.7× at 2²⁴, so the engine sort alone would make JS look much slower than JS code can be.
- **`wasm-scalar` is 1.5–1.8× `js-radix`** at every size, with the same algorithm. As expected, `wasm-simd` is indistinguishable from it.
- **`wasm-threads` passes `wasm-scalar` at 196,608** (≈ 0.2 ms of pool wake-ups: 8 `pool->run` calls per sort) and plateaus at 1.6–1.8 Gkeys/s, the same as native.
- **`gpu-opt` beats every single-threaded CPU sort from 2¹⁸ (compute) or 2²⁰ (end-to-end), but never `wasm-threads`** on this machine: 12.4 ms of GPU time at 2²⁴ against 9.6 ms for `wasm-threads`, and end-to-end it also pays 7.2 ms upload + 6.2 ms readback of 64 MB. Crossovers against JS (median only, one run): compute from 49,152 vs `js-builtin` and 131,072 vs `js-radix`; end-to-end from 65,536 and 262,144.
- **`gpu-naive` loses to everything but the two library sorts at 2²⁴.** Bitonic sort does O(n log² n) work, and every one of its up to 300 steps is a full pass through global memory. Sizes of the form 3·2ᵏ run at the speed of 4·2ᵏ because they are padded to it, hence the sawtooth in its Mkeys/s.

`native/` builds the same `sort.cpp` (Apple clang 17 libc++ `std::sort`). Every size, plus odd and few-distinct inputs, matches `std::sort` exactly. Apple M4 Pro, median Mkeys/s:

| n | radix scalar | radix simd | radix threads (8) | std::sort |
|---|---|---|---|---|
| 2¹⁰ | 289 | 289 | 5.1 (≈ 200 µs, pool wake-ups) | 129 |
| 2¹⁴ | 529 | 483 | 79 | 54 |
| 2¹⁸ | 461 | 456 | 802 | 29 |
| 2²⁰ | 360 | 352 | 1,485 | 25 |
| 2²⁴ | 339 | 330 | 1,794 | 21 |

The browser's `wasm-scalar` reaches 65–85 % of native radix scalar below 2²⁰, where the data is in cache, and 92–100 % from 2²⁰ on. The library sorts don't line up: Chrome's built-in sort (57 Mkeys/s at 2²⁴) and `wasm-std` (35, Emscripten's libc++) are both faster than this native `std::sort` (21, the system libc++). The comparison between library sorts reflects the library versions, not WASM overhead.

## Milestone 4: edit distance

`web/src/workloads/editdistance/` computes the Levenshtein distance (unit costs) between two strings of length L over {A, C, G, T}, L = 64…32768, with 1.5× steps between powers of two. The strings are independent and uniform, stored one code (0–3) per byte. Every implementation returns the **last row of the DP matrix**, D[L][0..L], the distance from a to every prefix of b; its last entry is the edit distance. The UI shows M items/s, here Mcells/s (L² / median compute time). This is the negative control of PLAN.md §7: every cell depends on its left, upper and diagonal neighbours.

| Implementation | What it is |
|---|---|
| `js` | the two-row DP over `Uint16Array` rows, carrying the left and diagonal cells in locals |
| `wasm-scalar` | `ed_rows` in `wasm/src/editdistance.cpp`, the same loop |
| `wasm-simd` | `ed_tiled`: blocked anti-diagonal wavefront over 512×512 tiles, one tile after another. Inside a tile the cells of one anti-diagonal are independent; that loop auto-vectorizes to `i16x8` (8 u16 cells per instruction: `i8x16.ne` on the characters, `i16x8.min_u`; `pnpm check:simd`) |
| `wasm-threads` | `ed_tiled` with the tiles of each tile anti-diagonal split across the persistent pool; `pool->run()` returning is the barrier between tile anti-diagonals (2L/512 − 1 of them) |
| `gpu-naive` | `naive.wgsl`: one dispatch per anti-diagonal of cells, one invocation per cell, three rotating anti-diagonals in a storage buffer: 2L + 1 dispatches (65,537 at L = 32768) |
| `gpu-opt` | `tiled.wgsl`: one workgroup of 256 invocations per 256×256 tile, one dispatch per anti-diagonal of tiles (255 at L = 32768). Invocation t owns tile row t + 1 and computes its cells left to right, one per step, reading the cell above from invocation t − 1's previous step in double-buffered workgroup memory: one `workgroupBarrier` per step, 511 per full tile |

Each dispatch selects its parameters (anti-diagonal index) from a 256-byte slot of one uniform buffer with a dynamic offset, like the bitonic sort. All dispatches are in one compute pass, so `timestamp-query` covers them all.

Methodology decisions:

- **`wasm-simd` runs a different algorithm from `wasm-scalar`.** PLAN.md §7 lists the two-row DP for the C++ scalar/SIMD builds, but each cell of a row depends on the one to its left, so the row loop has no vector form; with it, the simd build would be the scalar code again (as in sort). Cells of one anti-diagonal are independent, so the simd and threads builds run the tiled anti-diagonal wavefront and `wasm-threads` vs. `wasm-simd` stays a pure threading comparison. The scalar → SIMD gap therefore includes the change of algorithm. `native/` measures it: the tiled kernel compiled without vectorization runs at 1.88 Gcells/s against 1.36 for the rows (both scalar), so ~1.4× of the native 6.6× gap comes from the anti-diagonal layout (no loop-carried `min` chain) and the rest from the 8 lanes.
- **Tiles pass their boundaries, not the whole matrix.** `rows[r]` holds D[min(512r, L)][0..L] and `cols[c]` holds D[0..L][min(512c, L)]. Tile (r, c) reads `rows[r][j0..j1]` and `cols[c][i0+1..i1]` and writes `rows[r+1][j0+1..j1]` and `cols[c+1][i0+1..i1]`, so tiles on the same anti-diagonal never touch the same entry and `rows[last]` is the output. The GPU kernel uses the same layout with 256-wide tiles. The matrix edges are written once (when the GPU buffers are allocated), never overwritten, so reused buffers start every iteration correctly.
- **Tile size 512 on the CPU** (native, M4 Pro, 8 threads): 128-wide tiles reached 27 Gcells/s at L = 32768, 256-wide 32, 512-wide 45 and 1024-wide 46; at L = 2048, 512 and 1024 gave 11.4 and 10.8. Single-threaded, 512 is 6 % faster than 256. The cost is parallelism at small L: up to L = 512 there is one tile and `wasm-threads` runs on one thread plus a pool wake-up.
- **Cells are u16** in the CPU implementations (8 per `v128`), which limits L to 65535; the GPU stores u32.
- **The JS and C++ loops use the same expression,** `min(min(diag + (aᵢ ≠ bⱼ), up + 1), left + 1)`, with the mismatch as `+(ai !== b[j - 1])` in JS. Writing it as a ternary `? 0 : 1` made the loop 15 % slower on random DNA in Node 24 (presumably a mispredicted branch). `min(diag + cost, min(up, left) + 1)` was 13 % faster than the chosen form in Node 24 but 24 % slower natively, because it puts three operations on the `left` dependency chain instead of two. The baseline stays the same source as the C++, as in convolution.
- **Verification** (`reference.ts`) requires the output row to equal, entry for entry, the last row computed by Myers' bit-parallel algorithm (multi-word, global distance): a different algorithm from every implementation's DP, O(L²/32) word operations, ~0.1 s in Node at L = 32768. Checking the whole row catches errors that leave the last cell right. The reference was itself checked against a full-matrix DP for 768 pairs (1–300 characters, unequal lengths, similar strings) and rejects a row with one entry changed.
- Odd and unequal lengths (all 196 pairs of 1, 2, 3, 4, 5, 7, 17, 255, 256, 257, 511, 512, 513, 1000), similar (3000, 150 substitutions), identical (4097), all-`A` vs. all-`A` (1500 × 700), all-`A` vs. all-`T` (900 × 1300), 4097 × 4099, 32768 × 300, 300 × 32768 and 32768 × 32768 were verified for all six implementations in Chrome 156, two iterations each so the second runs on reused GPU buffers.

Sweep (Chrome 156, M4 Pro plugged in, 8 threads, 1 ns timestamps, one run; `results/results-chrome-2026-10-29T17-52-40.json`). Median compute Mcells/s, with end-to-end ms and GPU timestamp ms in parentheses for the GPU:

| L | js | wasm-scalar | wasm-simd | wasm-threads | gpu-naive | gpu-opt |
|---|---|---|---|---|---|---|
| 64 | 876 | 1,500 | 3,350 | 172 | 7.6 (0.74 / 0.19 ms) | 13.8 (0.50 / 0.025 ms) |
| 256 | 1,030 | 1,590 | 5,630 | 2,310 | 48.2 (1.6 / 0.77 ms) | 182 (0.56 / 0.094 ms) |
| 1024 | 1,060 | 1,590 | 5,800 | 5,780 | 221 (5.0 / 3.2 ms) | 1,070 (1.2 / 0.66 ms) |
| 4096 | 1,090 | 1,590 | 5,820 | 12,800 | 897 (19 / 13 ms) | 5,100 (3.5 / 2.9 ms) |
| 16384 | 1,090 | 1,580 | 5,830 | 26,000 | 3,130 (86 / 64 ms) | 19,000 (14 / 14 ms) |
| 32768 | 1,090 | 1,580 | 5,820 | 28,200 | 4,870 (221 / 178 ms) | 26,500 (41 / 40 ms) |

An earlier sweep on battery (headless, 65,536 ns timestamps; `results/results-chrome-2026-10-29T14-57-38.json`) agreed within 4 % from L = 6144 on and within 12 % (in both directions) from 256 on, so the worker got P-cores there too.

- **Every implementation is flat or nearly flat in L except the parallel ones,** which need L large enough to fill their wavefront: `wasm-threads` passes `wasm-simd` at L = 1536 (3×3 tiles) and `gpu-opt` keeps gaining up to 32768, where the widest tile anti-diagonal has 128 workgroups.
- **`gpu-naive` is dispatch-bound.** 178 ms of GPU time for 65,537 dispatches at L = 32768 is 2.7 µs per dispatch for at most 32,768 cells each, and encoding and submitting them adds another ~40 ms of CPU time to the compute phase. It still beats `js` from L = 6144 and `wasm-scalar` from 8192, but never `wasm-simd`.
- **`gpu-opt` is barrier-bound.** 255 dispatches take 40 ms at L = 32768, ~156 µs per tile anti-diagonal, i.e. ~0.3 µs per `workgroupBarrier` step. It beats `js` from L = 1024 (compute, a tie there) or 1536 (end-to-end), `wasm-scalar` from 1536 (compute) or 2048 (end-to-end), `wasm-simd` from 6144, but **never `wasm-threads`** (40 vs. 38 ms at 32768). Upload and readback are tiny (≤ 0.3 ms: L bytes in, 4L bytes out), so unlike sort the GPU loses on the kernel, not on transfers.
- So the GPU does not lose outright: with the dependency structure exploited (tiles, one barrier per step) it overtakes every single-threaded CPU implementation, but the serial chain of 2L/256 dispatches × 511 barriers keeps it level with 8 CPU threads at best.

`native/` builds the same `editdistance.cpp`; every size up to 8192 and the odd and unequal lengths match a full-matrix DP exactly (larger sizes: rows and tiled agree). Apple M4 Pro, median Mcells/s:

| L | rows scalar | tiled scalar | tiled simd (NEON) | tiled threads (8) |
|---|---|---|---|---|
| 64 | 1,690 | 1,700 | 3,930 | 170 (≈ 24 µs, pool wake-up) |
| 1024 | 1,390 | 1,890 | 8,910 | 7,860 |
| 4096 | 1,350 | 1,870 | 8,910 | 19,100 |
| 32768 | 1,350 | 1,880 | 8,900 | 43,800 |

The browser's `wasm-scalar` runs the rows loop faster than this native build (1,580 vs. 1,350 Mcells/s; Emscripten's LLVM and Apple clang 17 schedule the `min` chain differently), while `wasm-simd` and `wasm-threads` reach 65 % of native tiled simd and threads. JS ran at 0.9–1.1 Gcells/s in Chrome 156 but only 0.45 in Node 24, so V8 versions differ widely on this loop.

## Milestone 5: automation

`pnpm bench` (in `web/`; `web/scripts/bench.ts`) produces one results file per browser with one command:

1. runs `pnpm build:wasm` and `pnpm build` (skip with `--no-build`);
2. uses the server already answering at the URL, or starts one and stops it at the end: Nginx through Docker Compose (default, http://localhost:8080) or `vite preview` with `--server preview` (http://localhost:4173). A server that answers without COOP/COEP headers is rejected;
3. for each browser in turn (Chrome, then Firefox), launches it **headed** with Playwright, opens `env.html` and requires the milestone 0 verdict to pass, then opens the runner page, selects the workloads, fills in the seed and device notes, clicks **Run**, prints progress, and saves the page's own **Download JSON** file to `results/` under the page's file name (`--csv` also saves the CSV);
4. prints a summary per browser (user agent, records per status, thread count, timestamp resolution, commit) and warns about an incomplete run, a worker without cross-origin isolation or WebGPU adapter, and invalid or errored records.

```sh
cd web
pnpm bench                                          # everything, both browsers
pnpm bench --browsers firefox --workloads matmul,sort --seed 2
pnpm bench --server preview                         # without Docker
pnpm bench --help
```

- **Chrome** is the installed Google Chrome (`channel: 'chrome'`), the same browser as manual runs, started with `--enable-webgpu-developer-features` for 1 ns timestamps. Playwright already passes `--disable-background-timer-throttling`, `--disable-backgrounding-occluded-windows` and `--disable-renderer-backgrounding`, so a covered window keeps its timers and priority.
- **Firefox** is Playwright's patched build, because Playwright can't drive stable Firefox. `dom.webgpu.enabled` and `dom.webgpu.workers.enabled` are set explicitly. Its user agent (`Firefox/157.0`) is recorded like any other.
- **Headed, because** headless Firefox has no WebGPU adapter (milestone 0) and headless Chrome's GPU numbers were not used for the sweeps.
- **Device notes** are the `--notes` text (default: the CPU and hardware model from `sysctl`), the power source from `pmset` and `automated: Playwright <version>, headed`, e.g. `Apple M4 Pro (Mac16,8); plugged in; automated: Playwright 1.64.1, headed`. **On battery the script refuses to run** (workers land on the E-cores, milestone 2) unless given `--force`, which also overrides a failed environment check.
- **Ctrl-C** clicks **Stop**, saves the records finished so far (`complete: false`) and skips the remaining browsers; a second Ctrl-C quits at once.
- The exit status is 0 only if every browser produced a complete, cross-origin-isolated run.

**Automated Chrome measures the same as manual Chrome.** A matmul run through `pnpm bench` (Chrome 156, M4 Pro plugged in) matched the manual sweep (`results/results-chrome-2026-10-29T17-52-40.json`) within 3 % in median compute time for every CPU implementation from N = 128 on (e.g. `wasm-threads` 183.8 vs. 185.4 GFLOP/s at 1024), so the automated worker gets the P-cores. The GPU implementations matched within 2 % from N = 128 on, except at 512 (`gpu-opt` 416 vs. 459 GFLOP/s, `gpu-naive` 246 vs. 273), a size whose compute phase is noisy. Below N = 128, where a run takes microseconds, the two runs differ by up to ±21 % in both directions (`wasm-threads`, `gpu-naive` at N = 48), which is run-to-run noise rather than a bias of the automation.

**Firefox reports GPU completion only every 100 ms.** Its GPU process polls devices on a repeating 100 ms timer (`POLL_TIME_MS` in `dom/webgpu/ipc/WebGPUParent.cpp`), and neither `submit` nor `mapAsync` triggers a poll. So every `onSubmittedWorkDone` and `mapAsync` resolves at the next tick. In a sort run (Firefox 157, Playwright's build) every GPU phase took 102–105 ms from n = 1024 to 196,608: upload, compute and readback each, around kernels of 0.4–4 ms by timestamps (Chrome: compute phase 0.4–1.6 ms). Each iteration is one tick per phase, so the waits stay phase-locked to the timer. A run in stable Firefox showed the same ~300 ms per iteration on `dummy` at every size, so this is Firefox itself, not Playwright's build. The CPU implementations are unaffected. Consequences:

- Firefox's GPU compute and end-to-end medians measure the timer, not the GPU. Its kernel times come only from `timestamp-query`. The ~300 ms end-to-end floor is real for any page that awaits GPU results in Firefox.
- Each GPU record takes ~8 s at every size (first run, 5 warmups and at least 15 samples of ~310 ms), so a Firefox run spends nearly all its time waiting: in sort, 259 of the first 267 s went to the two GPU implementations.

### Safari (manual)

Playwright's WebKit is not Safari, so Safari runs from the UI:

1. `pnpm build && pnpm serve:nginx`, with the Mac plugged in;
2. open http://localhost:8080/env.html in Safari and check that the verdict passes;
3. open http://localhost:8080/, leave the default workloads (all but `dummy`), enter device notes in the same form (`Apple M4 Pro (Mac16,8); plugged in; manual`), click **Run** and keep the window in front until the status says `Done`;
4. **Download JSON** and move the file (`results-safari-<timestamp>.json`) into `results/`.

## Milestone 6: analysis

`analysis/analyze.py` turns results files into the paper's figures and crossover tables. It reads the runner page's **JSON** export or its **CSV** export (`--csv` in `pnpm bench`); if both exports of a run are given, the JSON one is used.

```sh
python3 -m venv analysis/.venv
analysis/.venv/bin/pip install -r analysis/requirements.txt     # pinned numpy, pandas, matplotlib, pytest
analysis/.venv/bin/python analysis/analyze.py                   # every file in results/ → analysis/output/
analysis/.venv/bin/python analysis/analyze.py results/results-chrome-2026-10-30T03-08-28.json --width 5.5
analysis/.venv/bin/python analysis/analyze.py --style clean --width 7 --titles   # sans-serif look, with figure titles
analysis/.venv/bin/python analysis/analyze.py --help
analysis/.venv/bin/python -m pytest analysis/tests              # crossover rules, JSON/CSV agreement
```

Output, per browser in `analysis/output/<browser>/` (gitignored; it regenerates from `results/`):

| File | What it shows |
|---|---|
| `figures/time-<workload>` | median time per call vs. size (log-log) with the IQR as a band, one panel each for compute, end-to-end and kernel; timed-out sizes as hollow markers at their first iteration's time |
| `figures/speedup-<metric>` | speedup over the JS baseline per workload (small multiples). The line crosses 1× at the crossover; the band is the conservative quartile ratio [q1(JS)/q3(impl), q3(JS)/q1(impl)], so the band clearing 1× is the IQR rule below. Large filled markers: crossovers; hollow: median-only crossovers |
| `figures/winners-<metric>` | the fastest implementation at every size; hatched where its IQR overlaps the runner-up's |
| `figures/crossover-map-<metric>` | for each pair in the crossover table, the size range where the challenger stays ahead |
| `crossovers.tex`, `crossovers.md` | the crossover table (booktabs, fits a two-column `table*`) |
| `report.md` | which run fed each workload, warnings (incomplete runs, battery, coarse timestamps, Firefox's 100 ms poll, noisy records), all tables |

and, for all browsers together, `analysis/output/summary.csv` (status, sample count, median / q1 / q3 / IQR % and throughput per metric for every record) and `analysis/output/crossovers.csv` (every ordered pair of implementations, metric and rule, with the interpolated equal-time size).

Methodology:

- **Metrics.** `compute` is the compute phase; `e2e` is upload + compute + readback, what a page pays per call; `kernel` is the timestamp-query time for WebGPU and the compute phase for CPU implementations. Statistics use the measured samples only (the verified first iteration is excluded), each already averaged over `repeats` iterations.
- **Crossover (PLAN.md §9):** the smallest size N at which the challenger's median is lower than the reference's **and the IQRs don't overlap** (q3 < q1), at N and at every larger size of the sweep. A size where only the medians are ordered counts as a loss under this rule; the median-only crossover is reported alongside it (in parentheses in the table, where it is earlier). `all`: ahead at every size; `—`: not ahead at the largest size, even if it is ahead over some smaller range (listed as `transient_wins` in `crossovers.csv`).
- **Timeouts.** A size where an implementation timed out (or was skipped after a timeout) is a loss for it against one that finished; the speedup there is a lower bound from the first iteration. Sizes where both timed out, where the reference failed, or where the metric is missing (no GPU timestamps) are left out; a challenger that fails verification loses that size.
- **Baselines.** JS is `js`, except for sort, where it is `js-radix` (the same algorithm as the WASM radix sorts; `--baseline sort=js-builtin` switches to the engine sort). The library sorts are drawn in gray as reference points.
- **Several runs** of one browser and workload: the latest complete run is used (`--combine latest`, the default), or the samples of all runs are pooled (`--combine pool`). Runs are grouped by browser only, so pass the files of one machine at a time.
- **Figure style.** The default `--style arxiv` matches a LaTeX paper. Text and math are set in Computer Modern (bundled with matplotlib, so no TeX installation is needed). Size symbols are italic math (*N* × *N*). Axes are boxed, with inward ticks on all four sides and minor ticks on log axes. The grid is light and dotted, and panels are labelled like subfigures: (a) Matrix multiply, (b) …. Figures have no titles, because the paper's captions carry them (`--titles` adds titles). `--style clean` is a sans-serif alternative with gray chrome.
- **Figure size.** Figures are 6.5 in wide by default, the text width of `article` with 1 in margins. Set `--width` to your `\textwidth` in inches (NeurIPS/ICML 5.5, IEEE/ACM two-column `figure*` 7.0) and include the PDF at `width=\textwidth`. At that scale the figure text is 7–8.5 pt next to a 10 pt body, i.e. `\footnotesize` to `\small`.
- **Figures** are vector PDFs with embedded TrueType fonts (no Type 3, which arXiv and most venues flag), plus 300 dpi PNGs. Each implementation keeps one color and one marker in every figure, so the figures stay readable in grayscale and for color-blind readers. The palette was checked with a color-vision-deficiency simulator. The JS baseline is black, and the three colors that stay distinguishable in every pair belong to `gpu-opt`, `wasm-threads` and `wasm-simd`.

Crossovers in Chrome 156 (M4 Pro plugged in, `results/results-chrome-2026-10-30T03-08-28.json`, IQR rule; median-only crossover in parentheses where earlier):

| Workload | Metric | wasm-scalar › JS | wasm-simd › JS | wasm-threads › JS | gpu-opt › JS | wasm-threads › wasm-simd | gpu-opt › wasm-simd | gpu-opt › wasm-threads |
|---|---|---|---|---|---|---|---|---|
| Matrix multiply (N) | compute | all | all | 48 | 96 | 96 | 192 | 384 |
| Matrix multiply (N) | end-to-end | all | all | 48 | 96 | 96 | 192 | 512 |
| Reduction (n) | compute | all | all | 3·2¹⁴ | 2¹⁹ | 3·2¹⁹ (2²⁰) | 3·2²² | — |
| Reduction (n) | end-to-end | 2¹⁹ (3·2¹⁷) | all | 2¹⁶ | 3·2¹⁹ | 3·2²² (2²³) | — | — |
| Sort (n), JS = js-radix | compute | all | all | 3·2¹⁵ | 2¹⁷ | 3·2¹⁶ | 3·2¹⁷ (2¹⁸) | — |
| Sort (n), JS = js-radix | end-to-end | all | all | 3·2¹⁵ | 3·2¹⁷ (2¹⁸) | 3·2¹⁶ | 2²⁰ | — |
| Convolution 5×5 (W) | compute | all | all | all | 192 | 256 | 768 | 2,048 |
| Convolution 5×5 (W) | end-to-end | all | all | all | 192 | 256 | 1,024 | — |
| N-body (n) | compute | all | all | 192 | 512 | 384 | 1,024 | 3,072 |
| N-body (n) | end-to-end | all | all | 192 | 768 | 384 | 1,536 | 4,096 |
| Edit distance (L) | compute | all | all | 192 | 1,024 | 1,024 | 6,144 | — |
| Edit distance (L) | end-to-end | all | all | 192 | 1,536 | 1,024 | 6,144 | — |

These agree with the median crossovers read off the sweeps by hand in the milestone sections above (e.g. matmul `gpu-opt` over `wasm-threads` from 384 / 512, N-body `gpu-opt` over `wasm-simd` from 1024 / 1536, sort `gpu-opt` over `js-radix` from 131,072 = 2¹⁷).

- **WASM beats JS at every size, with one exception:** end-to-end on reduction, `wasm-scalar` overtakes JS only from 2¹⁹ (median: 3·2¹⁷). Its compute-phase lead there is only ~1.3× (7.3 vs. 5.7 GB/s), and end-to-end time also includes copying the input into the heap.
- **`gpu-opt` overtakes JS early in every workload:** from N = 96 (matmul), W = 192 (convolution), n = 512–768 (N-body), L = 1024–1536 (edit distance), n = 2¹⁷–3·2¹⁷ (sort) and 2¹⁹–3·2¹⁹ (reduction). Below that, the GPU's ~0.5 ms fixed cost per call decides the result. Against `wasm-simd` the crossovers come 2–12× later in size, and end-to-end on reduction never.
- **`gpu-opt` overtakes `wasm-threads` only where the work grows faster than the data:** matmul (O(N³)), N-body (O(n²)) and, by compute phase only, convolution. It never overtakes it on reduction (memory-bound), sort (the GPU radix kernel alone is slower than 8 threads; transfers add to that) or edit distance (bound by dependencies and barriers). End-to-end, convolution joins those three, because 64 MB of upload and readback outweigh the kernel.
