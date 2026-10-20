// Messages between the UI pages (main.ts, env-page.ts) and the benchmark worker (worker.ts).
import type { EnvReport } from "./harness/env";
import type { RunMeta } from "./harness/results";
import type { RunConfig, SampleRecord } from "./harness/types";
import type { SmokeResult } from "./smoke";

export type WorkerRequest =
  { type: "env" } | { type: "run"; config: RunConfig };

/**
 * A run streams 'start', then 'status'/'record' as it goes, then 'done', so the page
 * keeps everything finished so far even if it stops the run by terminating the worker.
 */
export type WorkerResponse =
  | { type: "env"; env: EnvReport; smoke: SmokeResult[] }
  | { type: "start"; meta: RunMeta }
  | { type: "status"; text: string }
  | { type: "record"; record: SampleRecord }
  | { type: "done" }
  | { type: "error"; message: string };
