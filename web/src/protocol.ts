// Messages between the UI page (main.ts) and the benchmark worker (worker.ts).
import type { EnvReport } from './harness/env';
import type { SmokeResult } from './smoke';

export type WorkerRequest = { type: 'env' };

export type WorkerResponse =
  | { type: 'env'; env: EnvReport; smoke: SmokeResult[] }
  | { type: 'error'; message: string };
