// Milestone 5: one command produces a complete results file per browser.
//
//   pnpm bench                                 # build, serve, Chrome then Firefox → results/
//   pnpm bench --browsers chrome --workloads matmul,sort
//
// Playwright drives the runner page (index.html) exactly as a person would: it checks the
// environment report (env.html) first, fills in the run, clicks Run, waits, and saves the
// page's own "Download JSON" file. Browsers run headed (headless Firefox has no WebGPU
// adapter, and headless Chrome falls back to a software one) and one at a time.
// Safari is run manually from the same UI (README, milestone 5).
//
// Runs with Node's built-in type stripping (Node ≥ 22.18), so this file uses only erasable
// TypeScript.
import { execFileSync, spawn, type ChildProcess } from 'node:child_process';
import { mkdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { chromium, firefox, type Browser, type Page } from 'playwright';

const WEB_DIR = fileURLToPath(new URL('..', import.meta.url));
const COMPOSE_FILE = join(WEB_DIR, '..', 'deploy', 'docker-compose.yml');
const BROWSERS = ['chrome', 'firefox'] as const;
type BrowserName = (typeof BROWSERS)[number];
const SERVER_URLS = { nginx: 'http://localhost:8080', preview: 'http://localhost:4173' } as const;
type ServerKind = keyof typeof SERVER_URLS;

const USAGE = `Usage: pnpm bench [options]

  --browsers <list>   comma-separated, from: ${BROWSERS.join(', ')} (default: all)
  --workloads <list>  comma-separated workload ids (default: the page's default, all but dummy)
  --seed <n>          base seed (default: 1)
  --notes <text>      device notes (default: CPU model and hardware model); power source is appended
  --server <kind>     nginx (production, default) or preview; started only if --url isn't already serving
  --url <url>         where the build is served (default: ${SERVER_URLS.nginx} for nginx, ${SERVER_URLS.preview} for preview)
  --out <dir>         where results files go (default: ../results)
  --csv               also save the page's CSV export
  --no-build          skip pnpm build:wasm and pnpm build
  --force             run even on battery power or when the environment checks fail
  -h, --help`;

const { values: opts } = parseArgs({
  options: {
    browsers: { type: 'string', default: BROWSERS.join(',') },
    workloads: { type: 'string' },
    seed: { type: 'string', default: '1' },
    notes: { type: 'string' },
    server: { type: 'string', default: 'nginx' },
    url: { type: 'string' },
    out: { type: 'string', default: join(WEB_DIR, '..', 'results') },
    csv: { type: 'boolean', default: false },
    'no-build': { type: 'boolean', default: false },
    force: { type: 'boolean', default: false },
    help: { type: 'boolean', short: 'h', default: false },
  },
});

if (opts.help) {
  console.log(USAGE);
  process.exit(0);
}

function fail(message: string): never {
  console.error(`bench: ${message}`);
  process.exit(2);
}

const browserNames = list(opts.browsers);
for (const b of browserNames) if (!(BROWSERS as readonly string[]).includes(b)) fail(`unknown browser '${b}'\n\n${USAGE}`);
if (!(opts.server in SERVER_URLS)) fail(`unknown server '${opts.server}'\n\n${USAGE}`);
const serverKind = opts.server as ServerKind;
const baseUrl = (opts.url ?? SERVER_URLS[serverKind]).replace(/\/$/, '');
const seed = Number(opts.seed);
if (!Number.isInteger(seed) || seed < 0) fail(`--seed must be a non-negative integer`);

function list(s: string): string[] {
  return s
    .split(',')
    .map((x) => x.trim())
    .filter(Boolean);
}

// ── Device notes ─────────────────────────────────────────────────────────────

function sh(cmd: string, args: string[]): string | undefined {
  try {
    return execFileSync(cmd, args, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim();
  } catch {
    return undefined;
  }
}

/** 'plugged in', 'on battery', or undefined where it can't be read (macOS only for now). */
function powerSource(): string | undefined {
  if (process.platform !== 'darwin') return undefined;
  const ps = sh('pmset', ['-g', 'ps']);
  if (ps?.includes("'AC Power'")) return 'plugged in';
  if (ps?.includes("'Battery Power'")) return 'on battery';
  return undefined;
}

function defaultDevice(): string {
  if (process.platform === 'darwin') {
    const cpu = sh('sysctl', ['-n', 'machdep.cpu.brand_string']);
    const model = sh('sysctl', ['-n', 'hw.model']);
    if (cpu || model) return [cpu, model && `(${model})`].filter(Boolean).join(' ');
  }
  return `${process.platform} ${process.arch}`;
}

function notesFor(power: string | undefined): string {
  const device = opts.notes ?? defaultDevice();
  return [device, power ?? 'power source unknown', `automated: Playwright ${playwrightVersion()}, headed`].join('; ');
}

function playwrightVersion(): string {
  const pkg = JSON.parse(readFileSync(join(WEB_DIR, 'node_modules', 'playwright', 'package.json'), 'utf8')) as { version: string };
  return pkg.version;
}

// ── Build and server ─────────────────────────────────────────────────────────

function pnpm(...args: string[]): void {
  console.log(`$ pnpm ${args.join(' ')}`);
  execFileSync('pnpm', args, { cwd: WEB_DIR, stdio: 'inherit' });
}

/** True if `url` answers with the cross-origin isolation headers; throws if it answers without them. */
async function serving(url: string): Promise<boolean> {
  let res: Response;
  try {
    res = await fetch(url, { signal: AbortSignal.timeout(2000) });
  } catch {
    return false;
  }
  if (res.headers.get('cross-origin-embedder-policy') !== 'require-corp') {
    throw new Error(`${url} is served without COOP/COEP headers; it isn't the harness's server`);
  }
  return true;
}

async function waitUntilServing(url: string, timeoutMs: number): Promise<void> {
  const end = Date.now() + timeoutMs;
  while (!(await serving(url))) {
    if (Date.now() > end) throw new Error(`server at ${url} didn't come up within ${timeoutMs / 1000} s`);
    await new Promise((r) => setTimeout(r, 250));
  }
}

/** Starts the server unless `baseUrl` already answers; returns how to stop what was started. */
async function ensureServer(): Promise<() => void> {
  if (await serving(baseUrl)) {
    console.log(`Using the server already running at ${baseUrl}`);
    return () => {};
  }
  if (opts.url) throw new Error(`nothing is serving ${opts.url}`);
  if (serverKind === 'nginx') {
    console.log('$ docker compose up -d');
    try {
      execFileSync('docker', ['compose', '-f', COMPOSE_FILE, 'up', '-d'], { stdio: 'inherit' });
    } catch {
      throw new Error('could not start Nginx with Docker Compose (is Docker running?); --server preview uses vite preview instead');
    }
    await waitUntilServing(baseUrl, 30_000);
    return () => execFileSync('docker', ['compose', '-f', COMPOSE_FILE, 'down'], { stdio: 'inherit' });
  }
  const port = new URL(baseUrl).port || '80';
  console.log(`$ vite preview --port ${port}`);
  // Own process group, so stopping it also stops vite under pnpm, and Ctrl-C doesn't reach it.
  const child: ChildProcess = spawn('pnpm', ['exec', 'vite', 'preview', '--port', port, '--strictPort'], {
    cwd: WEB_DIR,
    stdio: 'ignore',
    detached: true,
  });
  const stop = () => process.kill(-child.pid!, 'SIGTERM');
  try {
    await waitUntilServing(baseUrl, 30_000);
  } catch (e) {
    stop();
    throw e;
  }
  return stop;
}

// ── Browsers ─────────────────────────────────────────────────────────────────

function launch(name: BrowserName): Promise<Browser> {
  // handleSIGINT: false, so Ctrl-C stops the run (see main) instead of closing the browser.
  switch (name) {
    case 'chrome':
      // Installed Google Chrome, as in manual runs. Developer features turn off the 64 µs
      // quantization of timestamp-query (1 ns instead). Playwright already passes the
      // --disable-background*/--disable-renderer-backgrounding switches, so an occluded
      // window keeps its timers and priority.
      return chromium.launch({ channel: 'chrome', headless: false, handleSIGINT: false, args: ['--enable-webgpu-developer-features'] });
    case 'firefox':
      // Playwright's patched Firefox build (stable Firefox can't be driven); WebGPU and
      // WebGPU in workers are switched on explicitly in case the build defaults them off.
      return firefox.launch({
        headless: false,
        handleSIGINT: false,
        firefoxUserPrefs: { 'dom.webgpu.enabled': true, 'dom.webgpu.workers.enabled': true },
      });
  }
}

/** Milestone 0 checks in env.html; returns the failure details, or undefined if they pass. */
async function envChecks(page: Page): Promise<string | undefined> {
  await page.goto(`${baseUrl}/env.html`);
  const verdict = page.locator('#checks .verdict');
  await verdict.waitFor({ timeout: 60_000 });
  if ((await verdict.getAttribute('class'))?.includes('pass')) return undefined;
  return (await page.locator('#checks').innerText()).trim();
}

/** Mirrors the fields of Results (src/harness/results.ts) that are reported here. */
interface ResultsSummary {
  complete: boolean;
  startedAt: string;
  finishedAt?: string;
  env: { userAgent: string; crossOriginIsolated: boolean; gpu: { adapter: unknown }; harnessCommit: string };
  wasmThreads: number;
  gpu: { timestampResolutionNs?: number; error?: string };
  records: { status: string }[];
}

let interrupted = false;

async function runOne(name: BrowserName, outDir: string): Promise<boolean> {
  const power = powerSource();
  if (power === 'on battery') {
    const msg = 'on battery power: Chrome on macOS then runs workers on the efficiency cores (README, milestone 2)';
    if (!opts.force) {
      console.error(`${name}: ${msg}; plug in, or pass --force`);
      return false;
    }
    console.warn(`${name}: warning: ${msg}`);
  }

  console.log(`\n── ${name} ──`);
  const browser = await launch(name);
  try {
    const page = await browser.newPage({ acceptDownloads: true });
    let crashed = false;
    page.on('crash', () => (crashed = true));
    await page.bringToFront();

    const envFailure = await envChecks(page);
    if (envFailure) {
      console.error(`${name}: environment checks fail:\n${envFailure}`);
      if (!opts.force) return false;
      console.warn(`${name}: continuing anyway (--force)`);
    } else {
      console.log(`${name}: environment checks pass`);
    }

    await page.goto(`${baseUrl}/`);
    const boxes = page.locator('#workloads input[type=checkbox]');
    await boxes.first().waitFor();
    if (opts.workloads) {
      const wanted = list(opts.workloads);
      const available: string[] = [];
      for (let i = 0; i < (await boxes.count()); i++) available.push((await boxes.nth(i).getAttribute('value')) ?? '');
      const unknown = wanted.filter((w) => !available.includes(w));
      if (unknown.length) throw new Error(`unknown workload(s) ${unknown.join(', ')}; available: ${available.join(', ')}`);
      for (const id of available) await page.locator(`#workloads input[value="${id}"]`).setChecked(wanted.includes(id));
    }
    await page.fill('#seed', String(seed));
    await page.fill('#notes', notesFor(power));
    await page.click('#start');

    // The page re-enables Run when the run ends (done, failed, crashed or stopped).
    const started = Date.now();
    let last = '';
    while (await page.locator('#start').isDisabled()) {
      if (crashed) throw new Error('page crashed');
      if (interrupted) {
        await page.click('#stop');
        break;
      }
      const status = await page.locator('#status').innerText();
      const rows = await page.locator('#rows tr').count();
      const current = `${rows} records · ${status}`;
      if (current !== last || process.stdout.isTTY) progress(`${name} ${clock(Date.now() - started)} · ${current}`);
      last = current;
      await page.waitForTimeout(1000);
    }
    progress('');
    console.log(`${name}: ${await page.locator('#status').innerText()} (${clock(Date.now() - started)})`);
    if (await page.locator('#json').isDisabled()) throw new Error('the run ended before the worker reported any metadata');

    const json = await save(page, '#json', outDir);
    if (opts.csv) await save(page, '#csv', outDir);
    return report(name, JSON.parse(readFileSync(json, 'utf8')) as ResultsSummary);
  } finally {
    await browser.close();
  }
}

/** Clicks a download button and saves the file under the page's own file name. */
async function save(page: Page, button: string, outDir: string): Promise<string> {
  const [download] = await Promise.all([page.waitForEvent('download'), page.click(button)]);
  const path = join(outDir, download.suggestedFilename());
  await download.saveAs(path);
  console.log(`saved ${path}`);
  return path;
}

function report(name: BrowserName, r: ResultsSummary): boolean {
  const counts = new Map<string, number>();
  for (const rec of r.records) counts.set(rec.status, (counts.get(rec.status) ?? 0) + 1);
  const statuses = [...counts].map(([s, n]) => `${n} ${s}`).join(', ');
  console.log(`${name}: ${r.env.userAgent}`);
  console.log(
    `${name}: ${r.records.length} records (${statuses}); wasm threads ${r.wasmThreads}; ` +
      `timestamp resolution ${r.gpu.timestampResolutionNs ?? 'n/a'} ns; commit ${r.env.harnessCommit}`,
  );
  const problems = [
    !r.complete && 'the run is incomplete (complete: false)',
    !r.env.crossOriginIsolated && 'the worker was not cross-origin isolated',
    !r.env.gpu.adapter && 'the worker had no WebGPU adapter',
    r.gpu.error && `GPU device: ${r.gpu.error}`,
    counts.get('invalid') && `${counts.get('invalid')} records failed verification`,
    counts.get('error') && `${counts.get('error')} records errored`,
  ].filter((p): p is string => Boolean(p));
  for (const p of problems) console.warn(`${name}: warning: ${p}`);
  return r.complete && r.env.crossOriginIsolated;
}

function clock(ms: number): string {
  const s = Math.round(ms / 1000);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

/** One rewritten status line on a terminal; one line per change otherwise. */
function progress(line: string): void {
  if (process.stdout.isTTY) process.stdout.write(`\r\x1b[2K${line.slice(0, process.stdout.columns - 1)}`);
  else if (line) console.log(line);
}

// ── Main ─────────────────────────────────────────────────────────────────────

async function main(): Promise<number> {
  if (!opts['no-build']) {
    pnpm('build:wasm');
    pnpm('build');
  }
  mkdirSync(opts.out, { recursive: true });
  const stopServer = await ensureServer();

  // First Ctrl-C stops the run and keeps what finished (complete: false); a second one exits.
  process.on('SIGINT', () => {
    if (interrupted) process.exit(130);
    interrupted = true;
    console.warn('\nStopping: saving the records finished so far (Ctrl-C again to quit now)');
  });

  const outcomes: [BrowserName, boolean][] = [];
  try {
    for (const name of browserNames as BrowserName[]) {
      if (interrupted) break;
      try {
        outcomes.push([name, await runOne(name, opts.out)]);
      } catch (e) {
        progress('');
        console.error(`${name}: ${e instanceof Error ? e.message : String(e)}`);
        outcomes.push([name, false]);
      }
    }
  } finally {
    stopServer();
  }
  console.log(`\n${outcomes.map(([n, ok]) => `${n}: ${ok ? 'complete' : 'FAILED'}`).join(' · ')}`);
  return outcomes.length === browserNames.length && outcomes.every(([, ok]) => ok) ? 0 : 1;
}

main().then(
  (code) => process.exit(code),
  (e) => fail(e instanceof Error ? e.message : String(e)),
);
