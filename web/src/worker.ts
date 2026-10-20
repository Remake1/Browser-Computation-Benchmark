// Benchmark worker entry point. Benchmarks run here, off the main thread, so page
// rendering and main-thread GC don't add noise. Milestone 0 only answers env requests.
import { collectEnv } from './harness/env';
import type { WorkerRequest, WorkerResponse } from './protocol';
import { runSmokeTests } from './smoke';

function reply(msg: WorkerResponse): void {
  self.postMessage(msg);
}

self.addEventListener('message', async (ev: MessageEvent<WorkerRequest>) => {
  if (ev.data.type !== 'env') return;
  try {
    const env = await collectEnv();
    const smoke = await runSmokeTests(env.wasm);
    reply({ type: 'env', env, smoke });
  } catch (e) {
    reply({ type: 'error', message: String(e) });
  }
});
