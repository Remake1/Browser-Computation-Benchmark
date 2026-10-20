// Benchmark worker entry point. Benchmarks run here, off the main thread, so page
// rendering and main-thread GC don't add noise.
import { sharedDevice } from "./backends/gpu";
import { wasmThreadCount } from "./backends/wasm";
import { collectEnv } from "./harness/env";
import type { GpuRunInfo } from "./harness/results";
import { runBenchmarks } from "./harness/runner";
import { measureTimestampResolutionNs, timed } from "./harness/timer";
import type { RunConfig } from "./harness/types";
import type { WorkerRequest, WorkerResponse } from "./protocol";
import { runSmokeTests } from "./smoke";
import { WORKLOADS } from "./workloads";

function reply(msg: WorkerResponse): void {
  self.postMessage(msg);
}

async function gpuInfo(): Promise<GpuRunInfo> {
  try {
    const [device, deviceInitMs] = await timed(() => sharedDevice());
    return {
      deviceInitMs,
      timestampResolutionNs: await measureTimestampResolutionNs(device),
    };
  } catch (e) {
    return { error: String(e) };
  }
}

async function run(config: RunConfig): Promise<void> {
  const workloads = WORKLOADS.filter((w) => config.workloads.includes(w.id));
  const env = await collectEnv();
  const usesGpu = workloads.some((w) =>
    w.implementations.some((i) => i.tech === "webgpu"),
  );
  const gpu = usesGpu ? await gpuInfo() : {};
  reply({
    type: "start",
    meta: {
      startedAt: new Date().toISOString(),
      config,
      env,
      gpu,
      wasmThreads: wasmThreadCount(),
    },
  });
  await runBenchmarks(workloads, config, {
    status: (text) => reply({ type: "status", text }),
    record: (record) => reply({ type: "record", record }),
  });
  reply({ type: "done" });
}

self.addEventListener("message", async (ev: MessageEvent<WorkerRequest>) => {
  try {
    if (ev.data.type === "env") {
      const env = await collectEnv();
      const smoke = await runSmokeTests(env.wasm);
      reply({ type: "env", env, smoke });
    } else if (ev.data.type === "run") {
      await run(ev.data.config);
    }
  } catch (e) {
    reply({ type: "error", message: String(e) });
  }
});
