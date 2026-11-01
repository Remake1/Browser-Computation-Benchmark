"""Run selection and per-record summary statistics (median, quartiles) for each metric."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .load import Dataset
from .workloads import workload_info

#: Metrics the analysis can compare.
#:  compute: the compute phase (run() until the result is ready; for WebGPU submit → onSubmittedWorkDone).
#:  e2e:     upload + compute + readback, what a page pays per call.
#:  kernel:  WebGPU timestamp-query time for GPU implementations, the compute phase for CPU ones.
METRICS = {
    "compute": "Compute phase",
    "e2e": "End-to-end (upload + compute + readback)",
    "kernel": "Kernel (GPU timestamps; CPU compute phase)",
}
METRIC_SHORT = {"compute": "compute", "e2e": "end-to-end", "kernel": "kernel"}

#: Statuses where the implementation was too slow: it loses to any implementation that finished.
SLOW = {"timeout", "skipped"}
STATUS_PRIORITY = ["ok", "timeout", "skipped", "invalid", "error", "unsupported"]


def select_runs(ds: Dataset, combine: str = "latest") -> pd.DataFrame:
    """Which runs feed each (browser, workload).

    latest: the most recent complete run containing the workload (an incomplete one only if
            no complete run has it).
    pool:   every run; the samples of all runs are pooled per (size, implementation).
    """
    rows = []
    present = ds.records[["run_id", "browser", "workload"]].drop_duplicates()
    runs = ds.runs.set_index("run_id")
    for (browser, wl), g in present.groupby(["browser", "workload"], sort=False):
        cand = runs.loc[g["run_id"]].sort_values("started_at")
        if combine == "pool":
            chosen = list(cand.index)
        elif combine == "latest":
            complete = cand[cand["complete"]]
            chosen = [(complete if len(complete) else cand).index[-1]]
        else:
            raise ValueError(f"unknown combine mode {combine!r}")
        for rid in cand.index:
            rows.append({"browser": browser, "workload": wl, "run_id": rid, "used": rid in chosen})
    return pd.DataFrame(rows)


def metric_values(samples: pd.DataFrame) -> pd.DataFrame:
    s = samples.copy()
    s["e2e"] = s["upload"] + s["compute"] + s["readback"]
    s["kernel"] = np.where(s["tech"] == "webgpu", s["gpu_compute"], s["compute"])
    return s


def _q(x: np.ndarray, p: float) -> float:
    x = x[~np.isnan(x)]
    return float(np.percentile(x, p)) if len(x) else np.nan


def summarize(ds: Dataset, selection: pd.DataFrame) -> pd.DataFrame:
    """One row per (browser, workload, size, implementation) with the status and, per metric,
    median / q1 / q3 over the measured samples and the first iteration's value."""
    used = selection.loc[selection["used"], ["browser", "workload", "run_id"]]
    keys = ["browser", "workload", "size", "implementation"]
    rec = ds.records.merge(used, on=["browser", "workload", "run_id"])
    smp = metric_values(ds.samples.merge(used, on=["browser", "workload", "run_id"]))
    labels = ds.runs.set_index("run_id")["browser_label"]

    smp_groups = {k: g for k, g in smp.groupby(keys, sort=False)}
    out = []
    for k, g in rec.groupby(keys, sort=False):
        statuses = list(g["status"])
        status = min(statuses, key=lambda s: STATUS_PRIORITY.index(s) if s in STATUS_PRIORITY else 99)
        ok_runs = set(g.loc[g["status"] == "ok", "run_id"])
        sg = smp_groups.get(k)
        row = dict(zip(keys, k))
        row |= {
            "browser_label": labels[g["run_id"].iloc[-1]],
            "tech": g["tech"].iloc[0],
            "status": status,
            "runs": ",".join(g["run_id"]),
            "repeats": int(g["repeats"].max()),
            "message": "; ".join(m for m in g["message"].fillna("") if m),
        }
        measured = sg[(sg["sample"] >= 0) & sg["run_id"].isin(ok_runs)] if sg is not None else None
        first = sg[sg["sample"] == -1] if sg is not None else None
        row["n_samples"] = 0 if measured is None else len(measured)
        for m in METRICS:
            x = measured[m].to_numpy(float) if measured is not None else np.array([])
            row[f"{m}_median"] = _q(x, 50)
            row[f"{m}_q1"] = _q(x, 25)
            row[f"{m}_q3"] = _q(x, 75)
            row[f"{m}_first"] = float(first[m].median()) if first is not None and len(first) else np.nan
        info = workload_info(k[1])
        work = info.work(k[2])
        for m in METRICS:
            med = row[f"{m}_median"]
            row[f"{m}_throughput"] = work / (med / 1e3) if med and med > 0 else np.nan
            row[f"{m}_iqr_pct"] = 100 * (row[f"{m}_q3"] - row[f"{m}_q1"]) / med if med and med > 0 else np.nan
        row["throughput_unit"] = info.throughput_unit
        out.append(row)
    df = pd.DataFrame(out)
    return df.sort_values(keys).reset_index(drop=True)
