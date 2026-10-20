// UI page. Milestone 0: collect the environment on the main thread and inside the
// benchmark worker, show the milestone checks, and offer the report as JSON.
import { collectEnv, type EnvReport } from './harness/env';
import type { WorkerRequest, WorkerResponse } from './protocol';
import type { SmokeResult } from './smoke';
import './style.css';

const WORKER_TIMEOUT_MS = 30_000;

interface FullReport {
  kind: 'env-report';
  notes: string;
  main: EnvReport;
  worker: EnvReport;
  smoke: SmokeResult[];
}

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const checksEl = $<HTMLElement>('checks');
const rawEl = $<HTMLPreElement>('raw');
const notesEl = $<HTMLTextAreaElement>('notes');
const rerunBtn = $<HTMLButtonElement>('rerun');
const downloadBtn = $<HTMLButtonElement>('download');

let report: FullReport | null = null;

/** Fresh worker per run, so a re-run never reuses an already-instantiated module or pthread pool. */
function runInWorker(): Promise<{ env: EnvReport; smoke: SmokeResult[] }> {
  const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
  return new Promise<{ env: EnvReport; smoke: SmokeResult[] }>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`worker did not answer within ${WORKER_TIMEOUT_MS / 1000} s`)), WORKER_TIMEOUT_MS);
    worker.onmessage = (ev: MessageEvent<WorkerResponse>) => {
      clearTimeout(timer);
      if (ev.data.type === 'env') resolve(ev.data);
      else reject(new Error(ev.data.message));
    };
    worker.onerror = (ev) => {
      clearTimeout(timer);
      reject(new Error(ev.message || 'worker failed to load'));
    };
    worker.postMessage({ type: 'env' } satisfies WorkerRequest);
  }).finally(() => worker.terminate());
}

function el<K extends keyof HTMLElementTagNameMap>(tag: K, className?: string, text?: string): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function table(head: string[], rows: (string | HTMLElement)[][]): HTMLTableElement {
  const t = el('table');
  const tr = t.createTHead().insertRow();
  for (const h of head) tr.append(el('th', undefined, h));
  const body = t.createTBody();
  for (const row of rows) {
    const r = body.insertRow();
    for (const cell of row) r.insertCell().append(cell);
  }
  return t;
}

type Status = 'pass' | 'fail' | 'unsupported';
const badge = (status: Status) => el('span', `badge ${status}`, status);
const yesNo = (v: boolean) => badge(v ? 'pass' : 'fail');

function adapterLabel(env: EnvReport): string {
  const a = env.gpu.adapter;
  if (a) return [a.info.vendor, a.info.architecture, a.info.description || a.info.device].filter(Boolean).join(' · ') || '(adapter, no info)';
  return env.gpu.error ?? (env.gpu.api ? 'no adapter' : 'navigator.gpu missing');
}

function render(r: FullReport): void {
  const w = r.worker;
  const criteria: [string, boolean, string][] = [
    ['crossOriginIsolated (worker)', w.crossOriginIsolated, w.crossOriginIsolated ? '' : 'check COOP/COEP response headers'],
    ['WASM SIMD (worker)', w.wasm.simd, ''],
    ['WASM threads (worker)', w.wasm.threads, w.wasm.threads ? '' : 'needs cross-origin isolation + shared memory'],
    ['WebGPU adapter (worker)', w.gpu.adapter !== null, adapterLabel(w)],
  ];
  const allPass = criteria.every(([, ok]) => ok) && r.smoke.every((s) => s.status === 'pass');

  const verdict = el('p', `verdict ${allPass ? 'pass' : 'fail'}`, allPass ? 'Milestone 0 checks pass in this browser.' : 'Some milestone 0 checks fail in this browser.');

  const criteriaTable = table(
    ['Criterion', 'Result', 'Detail'],
    criteria.map(([label, ok, detail]) => [label, yesNo(ok), detail]),
  );

  const smokeTable = table(
    ['Smoke test (in worker)', 'Result', 'Init', 'Detail'],
    r.smoke.map((s) => [s.id, badge(s.status), s.initMs === undefined ? '' : `${s.initMs.toFixed(1)} ms`, s.detail]),
  );

  const m = r.main;
  const fmtRes = (ms: number) => (Number.isFinite(ms) ? `${(ms * 1000).toFixed(1)} µs` : 'n/a');
  const compareTable = table(
    ['', 'Main thread', 'Worker'],
    [
      ['crossOriginIsolated', String(m.crossOriginIsolated), String(w.crossOriginIsolated)],
      ['SharedArrayBuffer', String(m.sharedArrayBuffer), String(w.sharedArrayBuffer)],
      ['performance.now() step', fmtRes(m.timerResolutionMs), fmtRes(w.timerResolutionMs)],
      ['WASM SIMD / threads', `${m.wasm.simd} / ${m.wasm.threads}`, `${w.wasm.simd} / ${w.wasm.threads}`],
      ['WebGPU adapter', adapterLabel(m), adapterLabel(w)],
      ['timestamp-query', String(m.gpu.adapter?.timestampQuery ?? false), String(w.gpu.adapter?.timestampQuery ?? false)],
    ],
  );

  const meta = el('p', 'meta', `${w.userAgent} · ${w.hardwareConcurrency} logical cores · harness ${w.harnessCommit}`);

  checksEl.replaceChildren(verdict, criteriaTable, smokeTable, el('h2', undefined, 'Main thread vs worker'), compareTable, meta);
}

function currentReport(): FullReport | null {
  return report && { ...report, notes: notesEl.value };
}

async function run(): Promise<void> {
  rerunBtn.disabled = true;
  downloadBtn.disabled = true;
  checksEl.replaceChildren(el('p', undefined, 'Running checks…'));
  try {
    const [main, fromWorker] = await Promise.all([collectEnv(), runInWorker()]);
    report = { kind: 'env-report', notes: notesEl.value, main, worker: fromWorker.env, smoke: fromWorker.smoke };
    render(report);
    rawEl.textContent = JSON.stringify(report, null, 2);
    downloadBtn.disabled = false;
  } catch (e) {
    checksEl.replaceChildren(el('p', 'verdict fail', `Worker failed: ${String(e)}`));
  } finally {
    rerunBtn.disabled = false;
  }
}

function browserTag(ua: string): string {
  if (/Edg\//.test(ua)) return 'edge';
  if (/Firefox\//.test(ua)) return 'firefox';
  if (/Chrome\//.test(ua)) return 'chrome';
  if (/Safari\//.test(ua)) return 'safari';
  return 'browser';
}

downloadBtn.addEventListener('click', () => {
  const r = currentReport();
  if (!r) return;
  const blob = new Blob([JSON.stringify(r, null, 2)], { type: 'application/json' });
  const a = el('a');
  a.href = URL.createObjectURL(blob);
  a.download = `env-${browserTag(r.worker.userAgent)}-${r.worker.collectedAt.slice(0, 10)}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
});
rerunBtn.addEventListener('click', run);

run();
