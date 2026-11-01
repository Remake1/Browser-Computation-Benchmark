"""Load results files exported by the runner page (web/src/harness/results.ts) into long tables.

Both export formats are accepted:

* JSON (``kind: 'bcb-results'``): run metadata plus every record and raw sample.
* CSV: one row per sample (``sample`` = ``first`` for the verified first iteration, an index for
  measured samples, empty for records without samples). It carries no run metadata except the
  browser tag and commit, so the start time is taken from the file name when it has the runner's
  ``results-<browser>-<stamp>.csv`` form.

If the JSON and CSV export of the same run are both given, the JSON one is used.
"""
from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PHASES = ["upload", "compute", "readback", "gpu_compute"]
_STAMP = re.compile(r"results-(?P<browser>[a-z]+)-(?P<stamp>\d{4}-\d\d-\d\dT\d\d-\d\d-\d\d)")


@dataclass
class Dataset:
    #: One row per input run.
    runs: pd.DataFrame
    #: One row per (run, workload, size, implementation): status, repeats, message.
    records: pd.DataFrame
    #: One row per timed iteration; ``sample`` = -1 is the verified first iteration.
    samples: pd.DataFrame


def browser_tag(user_agent: str) -> str:
    """Same rules as browserTag() in results.ts."""
    if "Edg/" in user_agent:
        return "edge"
    if "Firefox/" in user_agent:
        return "firefox"
    if "Chrome/" in user_agent:
        return "chrome"
    if "Safari/" in user_agent:
        return "safari"
    return "browser"


def browser_label(user_agent: str, tag: str) -> str:
    """Human-readable browser name with major version, e.g. 'Chrome 156 (headless)'."""
    patterns = {
        "edge": r"Edg/(\d+)",
        "firefox": r"Firefox/(\d+)",
        "chrome": r"Chrome/(\d+)",
        "safari": r"Version/(\d+(?:\.\d+)?)",
    }
    name = {"edge": "Edge", "firefox": "Firefox", "chrome": "Chrome", "safari": "Safari"}.get(tag, tag.title())
    m = re.search(patterns.get(tag, r"$^"), user_agent or "")
    label = f"{name} {m.group(1)}" if m else name
    if "HeadlessChrome" in (user_agent or ""):
        label += " (headless)"
    return label


def _phase_row(t: dict | None) -> dict:
    t = t or {}
    return {
        "upload": t.get("upload", np.nan),
        "compute": t.get("compute", np.nan),
        "readback": t.get("readback", np.nan),
        "gpu_compute": t.get("gpuCompute", np.nan),
    }


def _load_json(path: Path) -> tuple[dict, list[dict], list[dict]]:
    data = json.loads(path.read_text())
    if data.get("kind") != "bcb-results":
        raise ValueError(f"{path}: not a BCB results file (kind = {data.get('kind')!r})")
    if data.get("schemaVersion") != 1:
        warnings.warn(f"{path}: schemaVersion {data.get('schemaVersion')} (expected 1)")
    env = data.get("env", {})
    ua = env.get("userAgent", "")
    tag = browser_tag(ua)
    run_id = path.stem
    run = {
        "run_id": run_id,
        "file": str(path),
        "format": "json",
        "browser": tag,
        "browser_label": browser_label(ua, tag),
        "started_at": pd.Timestamp(data["startedAt"]),
        "complete": bool(data.get("complete", True)),
        "notes": data.get("notes", ""),
        "commit": env.get("harnessCommit", ""),
        "wasm_threads": data.get("wasmThreads"),
        "timestamp_resolution_ns": (data.get("gpu") or {}).get("timestampResolutionNs"),
        "workloads": ",".join(data.get("config", {}).get("workloads", [])),
    }
    records, samples = [], []
    for r in data["records"]:
        key = {
            "run_id": run_id,
            "browser": tag,
            "workload": r["workload"],
            "size": int(r["size"]),
            "implementation": r["implementation"],
            "tech": r["tech"],
        }
        records.append(
            key
            | {
                "status": r["status"],
                "repeats": r.get("repeats", 1),
                "order": r.get("order"),
                "init_ms": r.get("initMs", np.nan),
                "message": r.get("message", ""),
            }
        )
        if r.get("firstRun"):
            samples.append(key | {"sample": -1} | _phase_row(r["firstRun"]))
        for i, t in enumerate(r.get("samples", [])):
            samples.append(key | {"sample": i} | _phase_row(t))
    return run, records, samples


def _load_csv(path: Path) -> tuple[dict, list[dict], list[dict]]:
    df = pd.read_csv(path, dtype={"sample": "string", "message": "string", "commit": "string"}, keep_default_na=False)
    required = {"browser", "workload", "size", "implementation", "tech", "status", "sample", "compute_ms"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: not a BCB results CSV (missing columns {sorted(missing)})")
    tag = str(df["browser"].iloc[0]) if len(df) else "browser"
    m = _STAMP.search(path.name)
    if m:
        started = pd.Timestamp(m.group("stamp")[:10] + "T" + m.group("stamp")[11:].replace("-", ":"), tz="UTC")
    else:
        started = pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC")
    run_id = path.stem
    run = {
        "run_id": run_id,
        "file": str(path),
        "format": "csv",
        "browser": tag,
        "browser_label": tag.title(),
        "started_at": started,
        "complete": True,  # not recorded in the CSV
        "notes": "",
        "commit": str(df["commit"].iloc[0]) if "commit" in df and len(df) else "",
        "wasm_threads": None,
        "timestamp_resolution_ns": None,
        "workloads": ",".join(dict.fromkeys(df["workload"].astype(str))),
    }

    def num(v):
        return float(v) if v not in ("", None) else np.nan

    records, samples = [], []
    for (wl, size, impl), g in df.groupby(["workload", "size", "implementation"], sort=False):
        first = g.iloc[0]
        key = {
            "run_id": run_id,
            "browser": tag,
            "workload": str(wl),
            "size": int(size),
            "implementation": str(impl),
            "tech": str(first["tech"]),
        }
        records.append(
            key
            | {
                "status": str(first["status"]),
                "repeats": int(first["repeats"]) if str(first.get("repeats", "")) != "" else 1,
                "order": int(first["order"]) if str(first.get("order", "")) != "" else None,
                "init_ms": num(first.get("init_ms", "")),
                "message": str(first.get("message", "")),
            }
        )
        for _, row in g.iterrows():
            s = str(row["sample"])
            if s == "":
                continue
            samples.append(
                key
                | {
                    "sample": -1 if s == "first" else int(s),
                    "upload": num(row.get("upload_ms", "")),
                    "compute": num(row.get("compute_ms", "")),
                    "readback": num(row.get("readback_ms", "")),
                    "gpu_compute": num(row.get("gpu_compute_ms", "")),
                }
            )
    return run, records, samples


def expand_inputs(paths: list[str | Path]) -> list[Path]:
    """Files as given; directories expand to their results-*.json / *.csv files."""
    out: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(q for q in p.iterdir() if q.suffix in (".json", ".csv") and q.name.startswith("results-"))
        else:
            out.append(p)
    return out


def load(paths: list[str | Path]) -> Dataset:
    files = expand_inputs(paths)
    if not files:
        raise SystemExit("no results files given")
    # The same run exported as JSON and CSV: keep the JSON (it has the run metadata).
    by_stem: dict[str, Path] = {}
    for f in files:
        if f.stem in by_stem and by_stem[f.stem].suffix == ".json":
            continue
        by_stem[f.stem] = f
    runs, records, samples = [], [], []
    for f in by_stem.values():
        if f.suffix == ".json":
            run, rec, smp = _load_json(f)
        elif f.suffix == ".csv":
            run, rec, smp = _load_csv(f)
        else:
            raise SystemExit(f"{f}: expected a .json or .csv results file")
        runs.append(run)
        records += rec
        samples += smp
    sample_cols = ["run_id", "browser", "workload", "size", "implementation", "tech", "sample", *PHASES]
    return Dataset(
        runs=pd.DataFrame(runs).sort_values("started_at").reset_index(drop=True),
        records=pd.DataFrame(records),
        samples=pd.DataFrame(samples, columns=sample_cols),
    )
