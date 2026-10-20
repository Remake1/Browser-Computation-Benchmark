// Results file: run metadata + every raw sample, exported as JSON (complete) and
// CSV (one row per sample, long format, for quick loading in pandas/R).
import type { EnvReport } from './env';
import type { PhaseTimes, RunConfig, SampleRecord } from './types';

export interface GpuRunInfo {
  /** Shared device creation time (not charged to any implementation). */
  deviceInitMs?: number;
  /** Observed timestamp-query quantization, ns (1 = none visible). */
  timestampResolutionNs?: number;
  error?: string;
}

/** Collected by the worker when a run starts. */
export interface RunMeta {
  startedAt: string;
  config: RunConfig;
  env: EnvReport;
  gpu: GpuRunInfo;
  /** Thread count of the wasm-threads builds. */
  wasmThreads: number;
}

export interface Results extends RunMeta {
  kind: 'bcb-results';
  schemaVersion: 1;
  notes: string;
  finishedAt?: string;
  /** False if the run was stopped or crashed; records then cover only what finished. */
  complete: boolean;
  records: SampleRecord[];
}

export function browserTag(userAgent: string): string {
  if (/Edg\//.test(userAgent)) return 'edge';
  if (/Firefox\//.test(userAgent)) return 'firefox';
  if (/Chrome\//.test(userAgent)) return 'chrome';
  if (/Safari\//.test(userAgent)) return 'safari';
  return 'browser';
}

export function resultsFileName(r: Results, ext: 'json' | 'csv'): string {
  const stamp = r.startedAt.replace(/[:.]/g, '-').slice(0, 19);
  return `results-${browserTag(r.env.userAgent)}-${stamp}.${ext}`;
}

export function toJson(r: Results): string {
  return JSON.stringify(r, null, 2);
}

const CSV_COLUMNS = [
  'browser',
  'commit',
  'workload',
  'size',
  'implementation',
  'tech',
  'order',
  'status',
  'repeats',
  'sample',
  'upload_ms',
  'compute_ms',
  'readback_ms',
  'total_ms',
  'gpu_compute_ms',
  'init_ms',
  'message',
] as const;

/** One row per sample; the verified first iteration has sample = "first"; records without samples get one row with empty times. */
export function toCsv(r: Results): string {
  const browser = browserTag(r.env.userAgent);
  const lines = [CSV_COLUMNS.join(',')];
  for (const rec of r.records) {
    const row = (sample: string, t?: PhaseTimes) =>
      [
        browser,
        r.env.harnessCommit,
        rec.workload,
        rec.size,
        rec.implementation,
        rec.tech,
        rec.order,
        rec.status,
        rec.repeats,
        sample,
        t?.upload,
        t?.compute,
        t?.readback,
        t && t.upload + t.compute + t.readback,
        t?.gpuCompute,
        rec.initMs,
        rec.message,
      ]
        .map(csvCell)
        .join(',');
    if (rec.firstRun) lines.push(row('first', rec.firstRun));
    rec.samples.forEach((t, i) => lines.push(row(String(i), t)));
    if (!rec.firstRun && rec.samples.length === 0) lines.push(row(''));
  }
  return lines.join('\n') + '\n';
}

function csvCell(v: string | number | undefined): string {
  if (v === undefined) return '';
  const s = String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

/** Main thread only: saves `text` as a file via a temporary link. */
export function download(text: string, fileName: string, type: string): void {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = fileName;
  a.click();
  URL.revokeObjectURL(a.href);
}
