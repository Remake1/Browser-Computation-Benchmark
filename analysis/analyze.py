#!/usr/bin/env python3
"""Milestone 6: figures and crossover tables from BCB results files (JSON or CSV exports).

    analysis/.venv/bin/python analysis/analyze.py                      # every file in results/
    analysis/.venv/bin/python analysis/analyze.py results/results-chrome-2026-10-30T03-08-28.json
    analysis/.venv/bin/python analysis/analyze.py results/ --combine pool --formats pdf,png,svg
    analysis/.venv/bin/python analysis/analyze.py --width 5.5          # NeurIPS/ICML text width
    analysis/.venv/bin/python analysis/analyze.py --style clean --width 7 --titles

Writes, per browser, to <out>/<browser>/:
    figures/time-<workload>.{pdf,png}      median time vs size (log-log), IQR bands, one panel per metric
    figures/speedup-<metric>.{pdf,png}     speedup over the JS baseline per workload, crossovers marked
    figures/winners-<metric>.{pdf,png}     fastest implementation at each size
    figures/crossover-map-<metric>.{pdf,png} where each challenger stays ahead of its reference
    crossovers.tex / crossovers.md         the paper's crossover table
    report.md                              runs used, warnings, crossover tables
and, for all browsers together, <out>/summary.csv (per record statistics) and
<out>/crossovers.csv (every ordered pair, metric and rule).
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from bcbanalysis import plots, report, style  # noqa: E402
from bcbanalysis.crossover import crossover_table  # noqa: E402
from bcbanalysis.load import load  # noqa: E402
from bcbanalysis.stats import METRICS, select_runs, summarize  # noqa: E402
from bcbanalysis.workloads import WORKLOAD_ORDER, impl_sort_key  # noqa: E402


def _csv_list(s: str, allowed=None) -> list[str]:
    items = [x.strip() for x in s.split(",") if x.strip()]
    if allowed is not None:
        bad = [x for x in items if x not in allowed]
        if bad:
            raise argparse.ArgumentTypeError(f"unknown value(s) {bad}; choose from {sorted(allowed)}")
    return items


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="*", default=[str(HERE.parent / "results")],
                   help="results .json/.csv files or directories (default: results/)")
    p.add_argument("-o", "--out", default=str(HERE / "output"), help="output directory (default: analysis/output)")
    p.add_argument("--combine", choices=["latest", "pool"], default="latest",
                   help="several runs of one browser and workload: use the latest complete one (default) "
                        "or pool their samples")
    p.add_argument("--metrics", type=lambda s: _csv_list(s, METRICS), default=list(METRICS),
                   help="metrics to plot (default: compute,e2e,kernel)")
    p.add_argument("--table-metrics", type=lambda s: _csv_list(s, METRICS), default=["compute", "e2e"],
                   help="metrics in the crossover table (default: compute,e2e)")
    p.add_argument("--formats", type=_csv_list, default=["pdf", "png"], help="figure formats (default: pdf,png)")
    p.add_argument("--baseline", action="append", default=[], metavar="WORKLOAD=IMPL",
                   help="baseline for a workload, e.g. sort=js-builtin (default: js; sort: js-radix)")
    p.add_argument("--workloads", type=_csv_list, help="only these workloads")
    p.add_argument("--browsers", type=_csv_list, help="only these browsers (chrome, firefox, safari, …)")
    p.add_argument("--style", choices=style.STYLES, default="arxiv",
                   help="arxiv: LaTeX look (Computer Modern, boxed axes, inward ticks, (a)/(b) panel labels; "
                        "default); clean: sans-serif with gray chrome")
    p.add_argument("--width", type=float, metavar="INCHES",
                   help="figure width; set it to your paper's \\textwidth so text keeps its size when the figure is "
                        "included at \\textwidth (default: 6.5 for arxiv, i.e. article with 1 in margins; 7.0 for clean)")
    p.add_argument("--titles", action=argparse.BooleanOptionalAction, default=False,
                   help="add a title above each figure (default: off; the paper's captions carry them)")
    args = p.parse_args(argv)

    baseline_override = {}
    for b in args.baseline:
        if "=" not in b:
            p.error(f"--baseline expects WORKLOAD=IMPL, got {b!r}")
        k, v = b.split("=", 1)
        baseline_override[k.strip()] = v.strip()

    style.apply(args.style, args.width)
    warnings.filterwarnings("ignore", message=".*tight_layout.*")
    ds = load(args.inputs)
    if args.workloads:
        ds.records = ds.records[ds.records["workload"].isin(args.workloads)]
        ds.samples = ds.samples[ds.samples["workload"].isin(args.workloads)]
    if args.browsers:
        ds.records = ds.records[ds.records["browser"].isin(args.browsers)]
        ds.samples = ds.samples[ds.samples["browser"].isin(args.browsers)]
    if ds.records.empty:
        print("nothing to analyse after filtering", file=sys.stderr)
        return 1

    selection = select_runs(ds, args.combine)
    summary = summarize(ds, selection)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "summary.csv", index=False, float_format="%.6g")
    xo = crossover_table(summary)
    xo.to_csv(out / "crossovers.csv", index=False, float_format="%.6g")
    print(f"{len(ds.runs)} run(s), {len(summary)} records → {out}/summary.csv, {out}/crossovers.csv")

    for browser, sb in summary.groupby("browser", sort=False):
        label = sb["browser_label"].mode().iloc[0]
        title = label if args.titles else None
        bdir = out / browser
        fdir = bdir / "figures"
        sel_b = selection[selection["browser"] == browser]
        print(f"\n{label}:")
        for r in sel_b[sel_b["used"]].itertuples():
            print(f"  {r.workload:<13} ← {ds.runs.set_index('run_id').loc[r.run_id, 'file']}")
        written = []
        wls = [w for w in WORKLOAD_ORDER if w in set(sb["workload"])] + sorted(set(sb["workload"]) - set(WORKLOAD_ORDER))
        for wid in wls:
            g = sb[sb["workload"] == wid]
            t = f"{plots.workload_info(wid).title} ({label})" if args.titles else None
            written += plots.save(plots.time_figure(g, wid, args.metrics, t), fdir, f"time-{wid}", args.formats)
        for metric in args.metrics:
            written += plots.save(plots.speedup_figure(sb, metric, baseline_override, title), fdir,
                                  f"speedup-{metric}", args.formats)
            written += plots.save(plots.winners_figure(sb, metric, title), fdir, f"winners-{metric}", args.formats)
            written += plots.save(plots.crossover_map_figure(sb, metric, baseline_override, title), fdir,
                                  f"crossover-map-{metric}", args.formats)

        (bdir / "crossovers.tex").write_text(report.latex_table(sb, args.table_metrics, baseline_override, label))
        md_table = report.markdown_table(sb, args.table_metrics, baseline_override)
        (bdir / "crossovers.md").write_text(md_table + "\n")

        warn = report.warnings_for(ds.runs, sel_b, sb)
        lines = [
            f"# Crossover analysis — {label}",
            "",
            "Generated by `analysis/analyze.py` "
            f"(`--combine {args.combine}`). A crossover is the smallest size from which the challenger's median "
            "is lower than the reference's at every larger size, with non-overlapping IQRs (PLAN.md §9); "
            "in parentheses, the crossover by median alone where it is earlier. `all`: ahead at every size; "
            "`—`: not ahead at the largest size. Sizes where an implementation timed out count as a loss for it.",
            "",
            "## Runs",
            "",
            report.runs_table(ds.runs, sel_b),
            "",
        ]
        if warn:
            lines += ["## Warnings", ""] + [f"- {w}" for w in warn] + [""]
        lines += ["## Crossovers", "", md_table, ""]
        for metric in args.metrics:
            if metric not in args.table_metrics:
                lines += [f"### {metric}", "", report.markdown_table(sb, [metric], baseline_override), ""]
        lines += ["## Figures", ""] + [f"- `{p.relative_to(bdir)}`" for p in written if p.suffix == ".pdf"
                                       or len(args.formats) == 1 or "pdf" not in args.formats]
        (bdir / "report.md").write_text("\n".join(lines) + "\n")
        for w in warn:
            print(f"  ! {w}")
        print(f"  {len(written)} figure files → {fdir}")
        print(f"  tables → {bdir}/crossovers.tex, crossovers.md, report.md")
        print()
        print(md_table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
