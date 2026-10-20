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
- One iteration is `setup → run → readback` (timed as upload / compute / readback), followed by an untimed `teardown()`. Every iteration starts from freshly uploaded input, so in-place kernels such as sort don't need a `reset()`.
- The first iteration is verified (`invalid` if wrong) and stored as `firstRun`. If it takes longer than `timeoutMs`, the record is `timeout`. After a timeout or an error, larger sizes are recorded as `skipped`.
- Warmup runs for `warmupRuns` iterations or `warmupBudgetMs`, whichever comes first. If an iteration is shorter than `minSampleMs`, each sample averages `k` iterations (`repeats`, capped at 1000).
- Sampling collects at least `minSamples` and at most `maxSamples`, and stops between the two when `sampleBudgetMs` runs out.
- WebGPU implementations share one device, created before measuring (`gpu.deviceInitMs`). They report GPU-side time through `gpuTimeMs()` using `GpuTimer` (timestamp-query). The observed quantization of the timestamps is recorded as `gpu.timestampResolutionNs`; Chrome rounds to 65,536 ns by default.

The `dummy` workload (an f32 sum through JS, the three `hello` WASM builds, and a single-workgroup WGSL kernel) exists only to exercise the harness.
