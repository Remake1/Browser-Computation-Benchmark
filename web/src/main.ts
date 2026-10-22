// Benchmark runner page: pick workloads, run them in the benchmark worker, show a
// per-record summary as results stream in, download the raw results.
import { download, resultsFileName, toCsv, toJson, type Results } from './harness/results';
import { summarize } from './harness/stats';
import { DEFAULT_CONFIG, type PhaseTimes, type RunConfig, type SampleRecord, type WorkloadId } from './harness/types';
import type { WorkerRequest, WorkerResponse } from './protocol';
import { WORKLOADS } from './workloads';
import './style.css';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const workloadsEl = $<HTMLFieldSetElement>('workloads');
const seedEl = $<HTMLInputElement>('seed');
const notesEl = $<HTMLTextAreaElement>('notes');
const startBtn = $<HTMLButtonElement>('start');
const stopBtn = $<HTMLButtonElement>('stop');
const jsonBtn = $<HTMLButtonElement>('json');
const csvBtn = $<HTMLButtonElement>('csv');
const statusEl = $<HTMLParagraphElement>('status');
const rowsEl = $<HTMLTableSectionElement>('rows');

let worker: Worker | null = null;
let results: Results | null = null;

for (const w of WORKLOADS) {
  const label = document.createElement('label');
  const box = document.createElement('input');
  box.type = 'checkbox';
  box.value = w.id;
  box.checked = w.id !== 'dummy'; // harness test only
  label.append(box, ` ${w.id} (n = ${w.sizes[0]}…${w.sizes[w.sizes.length - 1]}; ${w.implementations.map((i) => i.id).join(', ')})`);
  workloadsEl.append(label);
}

const fmt = (ms: number | undefined) => (ms === undefined || !Number.isFinite(ms) ? '' : ms.toPrecision(3));
const flopsOf = new Map(WORKLOADS.map((w) => [w.id, w.flops]));

function addRow(r: SampleRecord): void {
  const totals = summarize(r.samples.map((t: PhaseTimes) => t.upload + t.compute + t.readback));
  const compute = summarize(r.samples.map((t) => t.compute));
  const gpu = r.samples.flatMap((t) => (t.gpuCompute === undefined ? [] : [t.gpuCompute]));
  const flops = flopsOf.get(r.workload)?.(r.size);
  const cells = [
    r.workload,
    String(r.size),
    r.implementation,
    r.message ? `${r.status}: ${r.message}` : r.status,
    r.samples.length ? `${r.repeats} × ${r.samples.length}` : '',
    fmt(totals.median),
    compute.n ? `${fmt(compute.median)} ± ${fmt(compute.iqr)}` : '',
    gpu.length ? fmt(summarize(gpu).median) : '',
    flops && compute.n ? fmt(flops / (compute.median * 1e6)) : '', // from the compute median
    fmt(r.initMs),
  ];
  const tr = rowsEl.insertRow();
  for (const c of cells) tr.insertCell().textContent = c;
}

function finish(text: string, complete: boolean): void {
  worker?.terminate();
  worker = null;
  if (results) {
    results.complete = complete;
    results.finishedAt = new Date().toISOString();
  }
  statusEl.textContent = text;
  startBtn.disabled = false;
  stopBtn.disabled = true;
  jsonBtn.disabled = csvBtn.disabled = results === null;
}

function start(): void {
  const config: RunConfig = {
    ...DEFAULT_CONFIG,
    workloads: [...workloadsEl.querySelectorAll<HTMLInputElement>('input:checked')].map((b) => b.value as WorkloadId),
    seed: Number(seedEl.value) >>> 0,
  };
  if (config.workloads.length === 0) {
    statusEl.textContent = 'Select at least one workload.';
    return;
  }
  results = null;
  rowsEl.replaceChildren();
  startBtn.disabled = jsonBtn.disabled = csvBtn.disabled = true;
  stopBtn.disabled = false;
  statusEl.textContent = 'Starting worker…';

  worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
  worker.onmessage = (ev: MessageEvent<WorkerResponse>) => {
    const msg = ev.data;
    switch (msg.type) {
      case 'start':
        results = { kind: 'bcb-results', schemaVersion: 1, notes: notesEl.value, complete: false, records: [], ...msg.meta };
        break;
      case 'status':
        statusEl.textContent = msg.text;
        break;
      case 'record':
        results?.records.push(msg.record);
        addRow(msg.record);
        break;
      case 'done':
        finish(`Done: ${results?.records.length ?? 0} records.`, true);
        break;
      case 'error':
        finish(`Run failed: ${msg.message}`, false);
        break;
    }
  };
  worker.onerror = (ev) => finish(`Worker crashed: ${ev.message || 'unknown error'}`, false);
  worker.postMessage({ type: 'run', config } satisfies WorkerRequest);
}

function save(ext: 'json' | 'csv'): void {
  if (!results) return;
  const r = { ...results, notes: notesEl.value };
  if (ext === 'json') download(toJson(r), resultsFileName(r, 'json'), 'application/json');
  else download(toCsv(r), resultsFileName(r, 'csv'), 'text/csv');
}

startBtn.addEventListener('click', start);
stopBtn.addEventListener('click', () => finish(`Stopped: ${results?.records.length ?? 0} records kept.`, false));
jsonBtn.addEventListener('click', () => save('json'));
csvBtn.addEventListener('click', () => save('csv'));
