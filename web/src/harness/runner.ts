// Measurement protocol (PLAN.md §4). For each (workload, size): generate the input
// once from a derived seed, shuffle the implementation order, and for each
// implementation: init once (cold cost), one verified first iteration, warmup,
// then samples until the sample count / time budget is met.
import { deriveSeed, Rng } from './rng';
import { median } from './stats';
import { timed } from './timer';
import { Unsupported, type AnyWorkload, type Implementation, type PhaseTimes, type RunConfig, type SampleRecord } from './types';

/** Cap on iterations per sample when a single iteration is near the timer resolution. */
const MAX_REPEATS = 1000;

export interface RunnerEvents {
  status(text: string): void;
  record(record: SampleRecord): void;
}

type Impl = Implementation<unknown, unknown>;

/** Why an implementation produces no more measurements in this workload. */
interface Stopped {
  status: 'unsupported' | 'skipped';
  message: string;
}

const total = (t: PhaseTimes) => t.upload + t.compute + t.readback;

export async function runBenchmarks(workloads: AnyWorkload[], config: RunConfig, events: RunnerEvents): Promise<void> {
  for (const workload of workloads) await runWorkload(workload, config, events);
}

async function runWorkload(workload: AnyWorkload, config: RunConfig, events: RunnerEvents): Promise<void> {
  const initialized = new Set<Impl>();
  const stopped = new Map<Impl, Stopped>();
  try {
    for (const size of workload.sizes) {
      events.status(`${workload.id} n=${size}: generating input`);
      const input = workload.generate(size, deriveSeed(config.seed, workload.id, 'input', size));
      const order = new Rng(deriveSeed(config.seed, workload.id, 'order', size)).shuffle([...workload.implementations] as Impl[]);

      for (const [position, impl] of order.entries()) {
        const record: SampleRecord = {
          workload: workload.id,
          size,
          implementation: impl.id,
          tech: impl.tech,
          order: position,
          status: 'ok',
          repeats: 1,
          samples: [],
        };
        const stop = stopped.get(impl);
        if (stop) {
          events.record({ ...record, ...stop });
          continue;
        }
        events.status(`${workload.id} n=${size}: ${impl.id}`);

        if (!initialized.has(impl)) {
          try {
            [, record.initMs] = await timed(() => impl.init());
            initialized.add(impl);
          } catch (e) {
            if (e instanceof Unsupported) {
              stopped.set(impl, { status: 'unsupported', message: e.message });
              events.record({ ...record, status: 'unsupported', message: e.message });
            } else {
              stopped.set(impl, { status: 'skipped', message: `init failed at n=${size}` });
              events.record({ ...record, status: 'error', message: `init: ${String(e)}` });
            }
            continue;
          }
        }

        try {
          await measure(workload, impl, input, config, record);
        } catch (e) {
          record.status = 'error';
          record.message = String(e);
        } finally {
          impl.release?.();
        }
        if (record.status === 'timeout' || record.status === 'error') {
          stopped.set(impl, { status: 'skipped', message: `${record.status} at n=${size}` });
        }
        events.record(record);
      }
    }
  } finally {
    for (const impl of initialized) impl.dispose();
  }
}

/** One full setup → run → readback cycle, each phase timed; teardown is untimed. */
async function iteration(impl: Impl, input: unknown): Promise<{ output: unknown; times: PhaseTimes }> {
  try {
    const [, upload] = await timed(() => impl.setup(input));
    const [, compute] = await timed(() => impl.run());
    const [output, readback] = await timed(() => impl.readback());
    return { output, times: { upload, compute, readback } };
  } finally {
    impl.teardown?.();
  }
}

/**
 * Iterations per sample so a sample spans at least minSampleMs. Short iterations are
 * timed as one span of back-to-back runs: their per-phase times are each rounded to
 * the timer resolution (5 µs when cross-origin isolated), which made the old estimate
 * jump between 1000, 200 and 100 repeats, or fall to 40 on a JIT-cold warmup median.
 */
async function calibrateRepeats(impl: Impl, input: unknown, minSampleMs: number): Promise<number> {
  let count = 0;
  let elapsed = 0;
  const t0 = performance.now();
  while (elapsed < minSampleMs && count < MAX_REPEATS) {
    await iteration(impl, input);
    count++;
    elapsed = performance.now() - t0;
  }
  await impl.gpuTimeMs?.(); // discard calibration GPU times
  return elapsed > 0 ? Math.min(MAX_REPEATS, Math.ceil((minSampleMs * count) / elapsed)) : MAX_REPEATS;
}

/** Fills `record` in place: status, firstRun, repeats, samples. */
async function measure(workload: AnyWorkload, impl: Impl, input: unknown, config: RunConfig, record: SampleRecord): Promise<void> {
  // First iteration: verify correctness and check the time budget.
  const warmupStart = performance.now();
  const first = await iteration(impl, input);
  record.firstRun = { ...first.times, gpuCompute: await impl.gpuTimeMs?.() };
  if (!workload.verify(first.output, input)) {
    record.status = 'invalid';
    record.message = 'output failed verification';
    return;
  }
  if (total(first.times) > config.timeoutMs) {
    record.status = 'timeout';
    record.message = `first iteration took ${total(first.times).toFixed(0)} ms`;
    return;
  }

  // Warmup (unrecorded); the clock includes the first iteration, so a slow size warms up less.
  const warm: number[] = [];
  while (warm.length < config.warmupRuns && performance.now() - warmupStart < config.warmupBudgetMs) {
    warm.push(total((await iteration(impl, input)).times));
  }
  await impl.gpuTimeMs?.(); // discard warmup GPU times

  // Batch short iterations so each sample spans at least minSampleMs.
  const perIteration = warm.length ? median(warm) : total(first.times);
  const repeats = perIteration >= config.minSampleMs ? 1 : await calibrateRepeats(impl, input, config.minSampleMs);
  record.repeats = repeats;

  const samplingStart = performance.now();
  while (
    record.samples.length < config.maxSamples &&
    (record.samples.length < config.minSamples || performance.now() - samplingStart < config.sampleBudgetMs)
  ) {
    const sum: PhaseTimes = { upload: 0, compute: 0, readback: 0 };
    for (let r = 0; r < repeats; r++) {
      const { times } = await iteration(impl, input);
      sum.upload += times.upload;
      sum.compute += times.compute;
      sum.readback += times.readback;
    }
    const gpu = await impl.gpuTimeMs?.();
    record.samples.push({
      upload: sum.upload / repeats,
      compute: sum.compute / repeats,
      readback: sum.readback / repeats,
      gpuCompute: gpu === undefined ? undefined : gpu / repeats,
    });
  }
}
