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
- WebGPU implementations share one device, created before measuring (`gpu.deviceInitMs`). They report GPU-side time through `gpuTimeMs()` using `GpuTimer` (timestamp-query). The observed quantization of the timestamps is recorded as `gpu.timestampResolutionNs`; Chrome rounds to 65,536 ns by default.

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
