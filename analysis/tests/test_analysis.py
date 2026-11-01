"""Run: analysis/.venv/bin/python -m pytest analysis/tests"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bcbanalysis import fmt  # noqa: E402
from bcbanalysis.crossover import crossover_table, find_crossover, outcomes  # noqa: E402
from bcbanalysis.load import load  # noqa: E402
from bcbanalysis.stats import select_runs, summarize  # noqa: E402


# ── synthetic summaries ─────────────────────────────────────────────────────


def rec(impl, size, med, spread=0.05, status="ok", first=np.nan):
    row = {"browser": "chrome", "workload": "matmul", "size": size, "implementation": impl, "status": status}
    for m in ("compute", "e2e", "kernel"):
        if status == "ok":
            row |= {f"{m}_median": med, f"{m}_q1": med * (1 - spread), f"{m}_q3": med * (1 + spread)}
        else:
            row |= {f"{m}_median": np.nan, f"{m}_q1": np.nan, f"{m}_q3": np.nan}
        row[f"{m}_first"] = first
    return row


def summary_of(rows):
    return pd.DataFrame(rows)


def xo(rows, rule="robust", a="a", b="b"):
    return find_crossover(outcomes(summary_of(rows), a, b, "compute"), rule)


def test_crossover_basic_and_interpolation():
    # A: 10 ms everywhere; B: 2.5, 5, 20, 40 → A loses, loses, wins, wins.
    rows = [rec("a", s, 10) for s in (1, 2, 4, 8)] + [rec("b", s, t) for s, t in zip((1, 2, 4, 8), (2.5, 5, 20, 40))]
    c = xo(rows)
    assert (c.status, c.size, c.prev_size) == ("crossover", 4, 2)
    m = xo(rows, "median")
    # log-ratio goes from log(0.5) at 2 to log(2) at 4: equal at the geometric midpoint.
    assert m.interpolated == pytest.approx(2 * math.sqrt(2))


def test_must_stay_ahead():
    # A wins at 2, loses at 4, wins from 8: the crossover is 8 and 2 is a transient win.
    rows = [rec("a", s, 10) for s in (1, 2, 4, 8, 16)]
    rows += [rec("b", s, t) for s, t in zip((1, 2, 4, 8, 16), (5, 20, 5, 20, 30))]
    c = xo(rows)
    assert (c.status, c.size, c.transient_wins) == ("crossover", 8, [2])


def test_always_and_never():
    always = [rec("a", s, 1) for s in (1, 2)] + [rec("b", s, 2) for s in (1, 2)]
    assert xo(always).status == "always"
    never = [rec("a", s, 3) for s in (1, 2)] + [rec("b", s, 2) for s in (1, 2)]
    assert xo(never).status == "never"
    # Wins in a band but loses at the largest size: never.
    band = [rec("a", s, 10) for s in (1, 2, 4)] + [rec("b", s, t) for s, t in zip((1, 2, 4), (5, 20, 5))]
    c = xo(band)
    assert c.status == "never" and c.transient_wins == [2]


def test_robust_requires_separated_iqrs():
    # Medians 10 vs 10.5 with ±5 % quartiles overlap: a median win, not a robust one.
    rows = [rec("a", 1, 20), rec("a", 2, 10), rec("b", 1, 10), rec("b", 2, 10.5)]
    assert xo(rows, "median").status == "crossover"
    assert xo(rows, "robust").status == "never"


def test_timeouts():
    # B times out at 4: A (which finished) wins there; the speedup is a lower bound from B's first run.
    rows = [rec("a", s, 10) for s in (1, 2, 4)]
    rows += [rec("b", 1, 5), rec("b", 2, 20), rec("b", 4, None, status="timeout", first=6000)]
    outs = outcomes(summary_of(rows), "a", "b", "compute")
    assert outs[-1].win_robust is True and outs[-1].speedup == pytest.approx(600) and not outs[-1].measured
    assert xo(rows).size == 2
    # A times out: it loses there, so it never stays ahead.
    rows2 = [rec("a", 1, 1), rec("a", 2, None, status="timeout"), rec("b", 1, 5), rec("b", 2, 5)]
    assert xo(rows2).status == "never"
    # Both too slow at the largest size: that size is left out.
    rows3 = [rec("a", 1, 1), rec("a", 2, None, status="timeout"), rec("b", 1, 5), rec("b", 2, None, status="skipped")]
    assert xo(rows3).status == "always"


def test_failures_and_missing_metric():
    # A invalid → loss; B invalid → size left out.
    rows = [rec("a", 1, 1), rec("a", 2, 1, status="invalid"), rec("b", 1, 5), rec("b", 2, 5)]
    assert xo(rows).status == "never"
    rows = [rec("a", 1, 1), rec("a", 2, 1), rec("b", 1, 5), rec("b", 2, 5, status="error")]
    c = xo(rows)
    assert c.status == "always"
    # No GPU timestamps → kernel metric has no data.
    rows = [rec("a", 1, np.nan), rec("b", 1, 5)]
    assert xo(rows).status == "no data"


def test_crossover_table_all_pairs():
    rows = [rec("a", s, 10) for s in (1, 2)] + [rec("b", s, t) for s, t in zip((1, 2), (5, 20))]
    t = crossover_table(summary_of(rows))
    assert len(t) == 2 * 3  # (a,b), (b,a) × 3 metrics
    ab = t[(t.challenger == "a") & (t.metric == "compute")].iloc[0]
    assert ab.robust_status == "crossover" and ab.robust_size == 2 and ab.speedup_at_largest == pytest.approx(2)


def test_size_format():
    assert fmt.size(1024, True) == "2¹⁰"
    assert fmt.size(3 * 2**15, True) == "3·2¹⁵"
    assert fmt.size(3 * 2**15, True, "tex") == r"$3\cdot 2^{15}$"
    assert fmt.size(1536, False) == "1,536"
    assert fmt.size(65536, False, "tex") == "65{,}536"
    assert fmt.approx_size(1234, False) == "≈1,200"
    assert fmt.approx_size(2**17.26, True) == "≈2¹⁷·³"


# ── loading: JSON and CSV exports of one run agree ──────────────────────────


def _results_json():
    def phases(c, gpu=None):
        t = {"upload": 0.1, "compute": c, "readback": 0.05}
        if gpu is not None:
            t["gpuCompute"] = gpu
        return t

    return {
        "kind": "bcb-results",
        "schemaVersion": 1,
        "startedAt": "2026-10-30T03:08:28.218Z",
        "finishedAt": "2026-10-30T03:20:00.000Z",
        "complete": True,
        "notes": "test machine; plugged in",
        "wasmThreads": 8,
        "gpu": {"timestampResolutionNs": 1},
        "config": {"workloads": ["matmul"], "seed": 1},
        "env": {"userAgent": "Mozilla/5.0 (Macintosh) AppleWebKit/537.36 Chrome/156.0.0.0 Safari/537.36",
                "harnessCommit": "abc123"},
        "records": [
            {"workload": "matmul", "size": 64, "implementation": "js", "tech": "js", "order": 0, "status": "ok",
             "repeats": 4, "initMs": 0.2, "firstRun": phases(2.0), "samples": [phases(x) for x in (1.0, 1.1, 0.9, 1.2)]},
            {"workload": "matmul", "size": 64, "implementation": "gpu-opt", "tech": "webgpu", "order": 1,
             "status": "ok", "repeats": 1, "firstRun": phases(0.9, 0.01),
             "samples": [phases(x, x / 10) for x in (0.3, 0.31, 0.29)]},
            {"workload": "matmul", "size": 128, "implementation": "js", "tech": "js", "order": 1,
             "status": "timeout", "repeats": 1, "firstRun": phases(6000.0), "samples": [],
             "message": "first iteration took 6000 ms"},
            {"workload": "matmul", "size": 128, "implementation": "gpu-opt", "tech": "webgpu", "order": 0,
             "status": "ok", "repeats": 1, "firstRun": phases(1.0, 0.1),
             "samples": [phases(x, x / 5) for x in (0.5, 0.52, 0.48)]},
            {"workload": "matmul", "size": 256, "implementation": "js", "tech": "js", "order": 0,
             "status": "skipped", "repeats": 1, "samples": []},
        ],
    }


def _to_csv(r: dict) -> str:
    """Python port of toCsv() in web/src/harness/results.ts."""
    cols = ["browser", "commit", "workload", "size", "implementation", "tech", "order", "status", "repeats",
            "sample", "upload_ms", "compute_ms", "readback_ms", "total_ms", "gpu_compute_ms", "init_ms", "message"]

    def cell(v):
        if v is None:
            return ""
        s = str(v)
        return '"' + s.replace('"', '""') + '"' if any(c in s for c in '",\n') else s

    lines = [",".join(cols)]
    for rec in r["records"]:
        def row(sample, t=None):
            vals = ["chrome", r["env"]["harnessCommit"], rec["workload"], rec["size"], rec["implementation"],
                    rec["tech"], rec["order"], rec["status"], rec["repeats"], sample,
                    t and t["upload"], t and t["compute"], t and t["readback"],
                    t and t["upload"] + t["compute"] + t["readback"], t and t.get("gpuCompute"),
                    rec.get("initMs"), rec.get("message")]
            return ",".join(cell(v) for v in vals)
        if rec.get("firstRun"):
            lines.append(row("first", rec["firstRun"]))
        lines += [row(str(i), t) for i, t in enumerate(rec["samples"])]
        if not rec.get("firstRun") and not rec["samples"]:
            lines.append(row(""))
    return "\n".join(lines) + "\n"


def test_json_and_csv_agree(tmp_path):
    data = _results_json()
    jdir, cdir = tmp_path / "j", tmp_path / "c"
    jdir.mkdir(), cdir.mkdir()
    name = "results-chrome-2026-10-30T03-08-28"
    (jdir / f"{name}.json").write_text(json.dumps(data))
    (cdir / f"{name}.csv").write_text(_to_csv(data))
    sj = summarize(dj := load([jdir]), select_runs(dj))
    sc = summarize(dc := load([cdir]), select_runs(dc))
    assert dc.runs.started_at.iloc[0] == dj.runs.started_at.iloc[0].floor("s")
    cols = [c for c in sj.columns if c.split("_")[0] in ("compute", "e2e", "kernel")] + ["status", "n_samples"]
    pd.testing.assert_frame_equal(sj[cols], sc[cols])
    js64 = sj[(sj.implementation == "js") & (sj["size"] == 64)].iloc[0]
    assert js64.compute_median == pytest.approx(1.05) and js64.n_samples == 4  # the first run is excluded
    gpu = sj[(sj.implementation == "gpu-opt") & (sj["size"] == 64)].iloc[0]
    assert gpu.kernel_median == pytest.approx(0.03) and gpu.e2e_median == pytest.approx(0.45)
    to = sj[(sj.implementation == "js") & (sj["size"] == 128)].iloc[0]
    assert to.status == "timeout" and to.compute_first == 6000 and math.isnan(to.compute_median)
    # Both exports of the same run given together: counted once (the JSON).
    both = load([jdir, cdir])
    assert len(both.runs) == 1 and both.runs.format.iloc[0] == "json"


def test_latest_complete_run_is_selected(tmp_path):
    a, b = _results_json(), _results_json()
    b["startedAt"] = "2026-10-31T00:00:00.000Z"
    b["complete"] = False
    for rec in b["records"]:
        rec["samples"] = [{"upload": 0, "compute": 99.0, "readback": 0}] * 3
    (tmp_path / "results-chrome-2026-10-30T03-08-28.json").write_text(json.dumps(a))
    (tmp_path / "results-chrome-2026-10-31T00-00-00.json").write_text(json.dumps(b))
    ds = load([tmp_path])
    sel = select_runs(ds, "latest")
    assert list(sel.loc[sel.used, "run_id"]) == ["results-chrome-2026-10-30T03-08-28"]
    pooled = summarize(ds, select_runs(ds, "pool"))
    js64 = pooled[(pooled.implementation == "js") & (pooled["size"] == 64)].iloc[0]
    assert js64.n_samples == 7
