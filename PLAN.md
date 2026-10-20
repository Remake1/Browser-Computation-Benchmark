# Benchmark Harness — Implementation Plan

## Research question

> For each kind of workload, at what input size does WebGPU or WASM beat JavaScript, and what explains where that crossover happens?

**Workloads:** matrix multiply, reduction, sort, convolution, N-body, edit distance.
**Technologies:** JavaScript (TypeScript), WebAssembly (C++ via Emscripten), WebGPU (WGSL compute shaders).

## 1. Architecture

```
┌──────────────── Browser tab (cross-origin isolated) ────────────────┐
│  UI page: pick workloads/sizes, show progress, download results     │
│         │ postMessage                                               │
│         ▼                                                           │
│  Benchmark Worker  ── runner ── timer/stats ── verify               │
│      ├─ JS implementations (TypeScript, typed arrays)               │
│      ├─ WASM modules (C++ → Emscripten: scalar / SIMD / threads)    │
│      └─ WebGPU implementations (WGSL compute shaders)               │
└─────────────────────────────────────────────────────────────────────┘
          │ results.json / results.csv
          ▼
   analysis/ (Python: plots + crossover detection)
```

Key decisions:

- **Vite + TypeScript** as the dev/build server. It can send the `COOP: same-origin` and `COEP: require-corp` headers, which are needed for `SharedArrayBuffer` (required by WASM threads) and for finer `performance.now()` resolution.
- **Benchmarks run in a dedicated Web Worker**, not on the main thread, so UI rendering and garbage collection on the page don't add noise. Milestone 0 checks that WebGPU is available in workers in each browser.
- **One C++ source per workload, compiled into three WASM builds.** Each build is its own module, so the same code is measured under different compiler flags.
- **Measure production builds** (`vite build`, served by Nginx with the same COOP/COEP headers; see `deploy/`), not the dev server.

## 2. Repository layout

```
BCB-research/
├── web/
│   ├── index.html
│   ├── vite.config.ts            # COOP/COEP headers (dev + preview)
│   └── src/
│       ├── main.ts               # UI
│       ├── worker.ts             # benchmark entry point
│       ├── harness/
│       │   ├── types.ts          # Workload / Implementation interfaces
│       │   ├── runner.ts         # warmup, sampling, time budget, ordering
│       │   ├── timer.ts          # CPU + GPU timing
│       │   ├── stats.ts          # median, IQR, min
│       │   ├── env.ts            # browser/GPU/CPU metadata
│       │   ├── rng.ts            # seeded PRNG (identical inputs everywhere)
│       │   └── results.ts        # JSON + CSV export
│       ├── backends/
│       │   ├── gpu.ts            # adapter/device setup, buffer helpers, readback
│       │   └── wasm.ts           # module loading, heap views, feature detection
│       └── workloads/
│           ├── matmul/   { index.ts, js.ts, wasm.ts, gpu.ts, naive.wgsl, tiled.wgsl, reference.ts }
│           ├── reduction/ …
│           ├── sort/ …
│           ├── convolution/ …
│           ├── nbody/ …
│           └── editdistance/ …
├── wasm/
│   ├── CMakeLists.txt            # or Makefile; builds 3 variants per workload
│   ├── include/threadpool.hpp    # persistent pool for the threads variant
│   └── src/ matmul.cpp, reduction.cpp, sort.cpp, convolution.cpp, nbody.cpp, editdistance.cpp
├── deploy/                       # Nginx config + docker-compose for production runs (COOP/COEP)
├── scripts/setup-emsdk.sh        # installs the pinned emsdk into .tools/
├── native/                       # (optional) same C++ compiled natively as a ceiling
├── analysis/ { analyze.py, requirements.txt }
├── results/                      # raw run outputs, committed
└── README.md                     # pinned toolchain versions, how to run
```

## 3. Core interfaces

```ts
type Phase = 'upload' | 'compute' | 'readback';

interface Implementation<I, O> {
  id: string;                      // 'js', 'wasm-scalar', 'wasm-simd', 'wasm-threads', 'gpu-naive', 'gpu-opt'
  tech: 'js' | 'wasm' | 'webgpu';
  init(): Promise<void>;           // once: load module, compile pipeline (timed as "cold" cost)
  setup(input: I): Promise<void>;  // allocate + upload input   → 'upload' phase
  run(): Promise<void>;            // the kernel                 → 'compute' phase
  readback(): Promise<O>;          // copy result back to JS     → 'readback' phase
  reset?(): Promise<void>;         // untimed: restore input for in-place algorithms (sort)
  dispose(): void;
}

interface Workload<I, O> {
  id: 'matmul' | 'reduction' | 'sort' | 'convolution' | 'nbody' | 'editdistance';
  sizes: number[];                 // log-spaced sweep
  generate(size: number, seed: number): I;
  verify(output: O, input: I): boolean;   // tolerance-aware, vs f64/exact reference
  implementations: Implementation<I, O>[];
}
```

Each sample records the three phases separately. **End-to-end time** is their sum, and **kernel time** is `compute` alone. For WASM, "upload" means copying the data into the WASM heap. For JS, upload and readback are close to zero, which is the honest result.

## 4. Measurement protocol (`runner.ts`)

For each (workload, size):

1. **Generate input** once with a fixed seed, so every implementation gets identical data.
2. **Shuffle the order of implementations** for each size. If the machine heats up partway through, the slowdown doesn't always land on the same technology.
3. **Run each implementation:**
   - Run `init()` and record it separately as the cold cost (WASM instantiation, shader compilation).
   - Run once, then **verify correctness**. If verification fails, mark the result invalid and skip it.
   - **Warm up** with a fixed number of runs (e.g., 5) that aren't recorded.
   - **Sample**: at least 15–30 samples, or until a time budget runs out. If a single run is shorter than about 1 ms, repeat it k times per sample and divide, so the timer resolution doesn't swamp the result.
   - **Time budget**: if one run takes longer than about 5 s, record `timeout` and skip larger sizes for that implementation.
4. **Store every raw sample**, not just summaries. Medians and IQRs are computed during analysis.

## 5. Timing per technology (`timer.ts`)

| Tech | Upload | Compute | Readback |
|---|---|---|---|
| JS | ~0 (input already in typed array) | `performance.now()` around call | ~0 |
| WASM | copy into `HEAPF32`/`HEAPU32` | `performance.now()` around exported call | copy out of heap |
| WebGPU | `writeBuffer` + `onSubmittedWorkDone()` | encode + submit + `onSubmittedWorkDone()`; **plus** `timestamp-query` when available, for GPU-only time | `copyBufferToBuffer` to a staging buffer + `mapAsync` |

Notes:

- Browsers may round `timestamp-query` values to a coarser resolution; log the observed resolution in the environment metadata.
- WebGPU has no `f64` type, so **`f32` is used in every implementation** to keep the comparison fair. The C++ is compiled **without `-ffast-math`** so it keeps JS-like floating-point semantics. Both are recorded as methodology decisions.

## 6. WASM build (C++ / Emscripten)

- **Toolchain:** `emsdk`, one version pinned in the README. C++20, `-O3`.
- **Three builds** from the same source:
  - `scalar`: `-O3`
  - `simd`: `-O3 -msimd128`
  - `threads`: `-O3 -msimd128 -pthread -sPTHREAD_POOL_SIZE=N`, with fixed `INITIAL_MEMORY` and no memory growth (memory growth combined with pthreads costs performance).
- **Exports:** `extern "C" EMSCRIPTEN_KEEPALIVE` functions, built with `-sMODULARIZE -sEXPORT_ES6`. Buffers are `malloc`'d once in `setup()`, and JS writes into heap views of them.
- **Threads:** a persistent thread pool (`threadpool.hpp`) created in `init()`, so thread startup is never timed. Use `min(hardwareConcurrency, 8)` threads and record the count.
- **Check that SIMD actually happens:** run `wasm-objdump -d` (or `wasm2wat`) and confirm `v128` instructions appear. Float reductions **won't auto-vectorize** without fast-math, so the reduction SIMD variant is written by hand with `wasm_simd128.h`.
- **Optional:** compile the same `.cpp` natively with `clang -O3` as a "native ceiling" that helps explain the results.

## 7. Variant matrix per workload

| Workload | Input / size param | JS & C++ scalar/SIMD | C++ threads | WebGPU naive | WebGPU optimized | Verification |
|---|---|---|---|---|---|---|
| **Matrix multiply** | two N×N `f32`, N = 16…2048 | i-k-j loop order (cache-friendly), same in both | rows split across threads | 1 thread per output element | 16×16 tiles in workgroup shared memory | rel. error ≤ 1e-3 vs f64 reference |
| **Reduction (sum)** | `f32[n]`, n = 2¹⁰…2²⁶ | simple loop; SIMD = intrinsics, 4 accumulators | per-thread partial sums | multi-pass tree reduction in shared memory | fewer passes / more elements per thread (subgroups if available) | rel. error vs f64 Kahan sum |
| **Sort** | `u32[n]`, n = 2¹⁰…2²⁴ | JS: `Uint32Array.sort()` **and** a hand-written JS radix sort; C++: `std::sort` | parallel radix or merge sort | bitonic sort (pad to power of 2) | radix sort (stretch) | exact: sorted + same multiset |
| **Convolution** | grayscale `f32` W×W image, fixed 5×5 Gaussian, clamped edges, W = 64…4096 | direct 2D loops | rows split across threads | 1 thread per pixel | tile + halo in shared memory | rel. error ≤ 1e-4 |
| **N-body (1 step)** | n bodies (`vec4` pos+mass), n = 64…65536 | O(n²) force loop with softening | bodies split across threads | 1 thread per body | shared-memory tiling of bodies | rel. error ≤ 1e-3 |
| **Edit distance** | two strings of length L over {A,C,G,T}, L = 64…32768 | two-row DP | anti-diagonal wavefront with barriers | one dispatch per anti-diagonal | blocked wavefront, fewer dispatches | exact integer |

Fairness rules:

- All CPU implementations use the **same algorithm and loop order**.
- The JS baseline is idiomatic, well-written TypeScript on typed arrays, not deliberately slow code.
- `Uint32Array.sort()` runs native engine code, not JS, so sort also gets a hand-written JS radix sort.
- Edit distance is expected to lose on the GPU and is kept as the **negative control**: the explanation (dependencies, dispatch overhead) should predict losses as well as wins.

## 8. Environment metadata (`env.ts`)

Recorded with every run:

- `userAgent` (and `userAgentData` on Chromium)
- `hardwareConcurrency`
- `crossOriginIsolated`
- WASM SIMD and threads support (feature detection)
- `GPUAdapter.info` (vendor, architecture, device, description)
- relevant adapter limits
- `timestamp-query` availability and measured timer resolution
- harness git commit hash (injected at build time)
- free-text field for device model and "plugged in / on battery"

## 9. Milestones

| # | Milestone | Done when |
|---|---|---|
| **0** | **Toolchain & skeleton:** install emsdk, set up Vite+TS with COOP/COEP, create the worker, build a "hello" C++ module in all 3 variants | An environment report page shows `crossOriginIsolated: true`, WASM SIMD/threads support, and a WebGPU adapter **inside the worker** in Chrome, Firefox and Safari |
| **1** | **Harness core:** interfaces, runner, timer, stats, PRNG, verification, time budget, JSON/CSV download, minimal UI | A dummy workload produces a valid results file |
| **2** | **Matrix multiply end-to-end, all 6 variants** | Results are verified and GFLOP/s numbers are plausible (sanity-checked against the native build). The harness is debugged on this workload before more are written |
| **3** | Reduction, convolution, N-body | All variants verified and measured |
| **4** | Sort, edit distance (the hardest GPU kernels) | All variants verified and measured |
| **5** | **Automation:** Playwright drives Chrome and Firefox with real GPU (headed mode); Safari runs manually from the UI | One command produces a complete results file per browser |
| **6** | **Analysis:** log-log plots of time vs. size per workload, plus a crossover table | The crossover is reported as the smallest N where A's median beats B's and stays ahead for all larger N, with non-overlapping IQRs |

Milestone 2 is the critical one: if matrix multiply produces believable numbers across all three technologies, the harness is probably sound, and the remaining workloads are mostly kernel writing.

## 10. Open decisions

1. **Multithreaded JS variant (`js-workers`).** `wasm-threads` uses many cores while JS uses one. Either add a Workers + `SharedArrayBuffer` JS variant or state the limitation explicitly in the write-up. Currently a stretch goal.
2. **Native C++ ceiling.** Optional, but useful for sanity-checking WASM numbers and explaining results.
