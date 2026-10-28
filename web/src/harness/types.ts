// Workload / Implementation interfaces (PLAN.md §3) and the shapes of recorded results.

export type Phase = 'upload' | 'compute' | 'readback';
export type Tech = 'js' | 'wasm' | 'webgpu';

/** 'dummy' exists only to exercise the harness (milestone 1). */
export type WorkloadId = 'dummy' | 'matmul' | 'reduction' | 'sort' | 'convolution' | 'nbody' | 'editdistance';

/**
 * One way of computing a workload. Per sample the runner calls
 * setup → run → readback (each timed as its phase) → teardown (untimed),
 * so every iteration starts from freshly uploaded input; in-place algorithms
 * (sort) need no separate reset step. After the last iteration at a size the
 * runner calls release (untimed).
 */
export interface Implementation<I, O> {
  /** 'js', 'wasm-scalar', 'wasm-simd', 'wasm-threads', 'gpu-naive', 'gpu-opt', … */
  id: string;
  tech: Tech;
  /** Once per run: load module, compile pipeline. Timed separately as the cold cost. Throw `Unsupported` to skip. */
  init(): Promise<void>;
  /** Allocate + upload input → 'upload' phase. */
  setup(input: I): Promise<void>;
  /** The kernel → 'compute' phase. */
  run(): Promise<void>;
  /** Copy the result back to JS → 'readback' phase. */
  readback(): Promise<O>;
  /**
   * Untimed, after every iteration: release per-iteration state. Size-dependent resources
   * (GPU buffers) may instead be kept for the next setup() at the same size, so samples
   * don't pay fresh-allocation costs that WASM's reused heap pages don't pay either.
   */
  teardown?(): void;
  /** Untimed, after the last iteration at a size: release what setup() kept for reuse. */
  release?(): void;
  /** Untimed: GPU-side time (timestamp-query) of the run() calls since the last call, in ms; undefined if unavailable. */
  gpuTimeMs?(): Promise<number | undefined>;
  dispose(): void;
}

export interface Workload<I, O> {
  id: WorkloadId;
  /** Log-spaced sweep, ascending. */
  sizes: number[];
  generate(size: number, seed: number): I;
  /** Tolerance-aware check against an f64 / exact reference. */
  verify(output: O, input: I): boolean;
  /** Floating-point operations of one run at this size, for GFLOP/s; omitted where FLOPs aren't the natural unit. */
  flops?(size: number): number;
  /** Bytes one run must read from and write to memory at this size, for GB/s; omitted where bandwidth isn't the natural unit. */
  bytes?(size: number): number;
  implementations: Implementation<I, O>[];
}

/** Element type of the workload registry, whose workloads have different I/O types. */
export type AnyWorkload = Workload<any, any>;

/** Thrown from init() when the browser lacks what the implementation needs. */
export class Unsupported extends Error {
  override name = 'Unsupported';
}

export interface RunConfig {
  workloads: WorkloadId[];
  /** Base seed; per-(workload, size) input and ordering seeds derive from it. */
  seed: number;
  /** Unrecorded runs after verification. */
  warmupRuns: number;
  /** Warmup stops early once this much time has been spent (slow sizes are already JIT-hot). */
  warmupBudgetMs: number;
  minSamples: number;
  maxSamples: number;
  /** Past minSamples, keep sampling until maxSamples or this much time has been spent. */
  sampleBudgetMs: number;
  /** If one iteration is shorter than this, each sample averages k iterations so timer resolution doesn't dominate. */
  minSampleMs: number;
  /** An iteration slower than this is recorded as 'timeout' and larger sizes are skipped. */
  timeoutMs: number;
}

export const DEFAULT_CONFIG: RunConfig = {
  workloads: [],
  seed: 1,
  warmupRuns: 5,
  warmupBudgetMs: 2000,
  minSamples: 15,
  maxSamples: 30,
  sampleBudgetMs: 3000,
  minSampleMs: 1,
  timeoutMs: 5000,
};

/** Per-phase times of one sample, in ms per iteration (averaged over `repeats`). */
export interface PhaseTimes {
  upload: number;
  compute: number;
  readback: number;
  /** timestamp-query GPU time of the compute phase, when available. */
  gpuCompute?: number;
}

export type RecordStatus = 'ok' | 'invalid' | 'timeout' | 'error' | 'unsupported' | 'skipped';

/** Everything measured for one (workload, size, implementation). */
export interface SampleRecord {
  workload: WorkloadId;
  size: number;
  implementation: string;
  tech: Tech;
  /** Position in this size's shuffled implementation order (0 = ran first). */
  order: number;
  status: RecordStatus;
  message?: string;
  /** Cold cost of init(), on the record of the size where init() happened. */
  initMs?: number;
  /** The first, verified iteration (not part of `samples`). */
  firstRun?: PhaseTimes;
  /** Iterations averaged per sample. */
  repeats: number;
  samples: PhaseTimes[];
}
