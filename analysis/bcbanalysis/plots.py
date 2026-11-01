"""Paper figures. Every function takes the summary of one browser and returns a matplotlib Figure."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter, NullLocator

from . import fmt
from .crossover import find_crossover, outcomes
from .stats import METRIC_SHORT, METRICS, SLOW
from . import style
from .style import GRID, INK, INK_2, LOSS_BAND, MUTED, SURFACE, impl_style, mt, panel_title, strip_axes
from .workloads import PAPER_PAIRS, WORKLOAD_ORDER, impl_sort_key, workload_info



def _width() -> float:
    """Figure width in inches (style preset or --width)."""
    return style.CURRENT["width"]


def _suptitle(fig, text: str) -> None:
    fig.suptitle(mt(text), x=0.01, ha="left", fontsize=9,
                 fontweight="normal" if style.CURRENT["name"] == "arxiv" else "bold")


# ── axes helpers ────────────────────────────────────────────────────────────


def size_axis(ax, sizes, wid: str, label: bool = True) -> None:
    info = workload_info(wid)
    lo, hi = min(sizes), max(sizes)
    ax.set_xscale("log", base=2)
    k0, k1 = math.ceil(math.log2(lo)), math.floor(math.log2(hi))
    step = max(1, math.ceil((k1 - k0 + 1) / 7))
    ticks = [2**k for k in range(k0, k1 + 1) if (k - k0) % step == 0]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt.size(round(v), info.pow2_labels, "mpl")))
    pad = 2**0.25
    ax.set_xlim(lo / pad, hi * pad)
    if label:
        sym = f"${info.symbol}$" if style.CURRENT["name"] == "arxiv" else info.symbol
        ax.set_xlabel(f"{sym} ({mt(info.size_desc)})")


def _log_y(ax, suffix: str = "") -> None:
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    if style.CURRENT["name"] == "arxiv":
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=tuple(range(2, 10)), numticks=12))
    ax.yaxis.set_minor_formatter(NullFormatter())
    suffix = mt(suffix)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}{suffix}"))


def _impls(g: pd.DataFrame) -> list[str]:
    return sorted(g["implementation"].unique(), key=impl_sort_key)


def _legend_handle(impl: str, label: str | None = None) -> Line2D:
    st = impl_style(impl)
    return Line2D([], [], color=st["color"], marker=st["marker"], ls=st["ls"], label=label or impl,
                  markeredgecolor=st["color"])


def _baseline(g: pd.DataFrame, wid: str, override: dict[str, str]) -> str | None:
    impls = set(g["implementation"])
    for b in ([override[wid]] if wid in override else []) + list(workload_info(wid).baselines):
        if b in impls:
            return b
    return None


def save(fig, outdir: Path, name: str, formats: list[str]) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in formats:
        p = outdir / f"{name}.{f}"
        fig.savefig(p, metadata={"Creator": "BCB-research analysis"} if f == "pdf" else None)
        paths.append(p)
    plt.close(fig)
    return paths


# ── 1. time vs size, one figure per workload ────────────────────────────────


def time_figure(g: pd.DataFrame, wid: str, metrics: list[str], title: str | None) -> plt.Figure:
    """Median time per iteration vs size (log-log), IQR as a band, one panel per metric."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(_width(), 2.5), sharey=True, squeeze=False)
    axes = axes[0]
    impls = _impls(g)
    for k, (ax, metric) in enumerate(zip(axes, metrics)):
        for impl in impls:
            st = impl_style(impl)
            d = g[g["implementation"] == impl].sort_values("size")
            ok = d[(d["status"] == "ok") & d[f"{metric}_median"].notna()]
            if len(ok):
                ax.fill_between(ok["size"], ok[f"{metric}_q1"], ok[f"{metric}_q3"], color=st["color"], alpha=0.15, lw=0)
                ax.plot(ok["size"], ok[f"{metric}_median"], color=st["color"], ls=st["ls"], marker=st["marker"],
                        markeredgecolor=st["color"], markersize=2.8, lw=1.1, zorder=3)
            # Timed-out sizes: the first (only) iteration, hollow and joined by a dotted line.
            to = d[d["status"].isin(SLOW) & d[f"{metric}_first"].notna()]
            if len(to):
                xs = list(ok["size"].tail(1)) + list(to["size"])
                ys = list(ok[f"{metric}_median"].tail(1)) + list(to[f"{metric}_first"])
                ax.plot(xs, ys, color=st["color"], ls=":", lw=1)
                ax.plot(to["size"], to[f"{metric}_first"], ls="none", marker=st["marker"], markersize=5,
                        markerfacecolor=SURFACE, markeredgecolor=st["color"], zorder=4)
        size_axis(ax, g["size"], wid)
        _log_y(ax)
        panel_title(ax, k, METRIC_SHORT[metric].capitalize())
    axes[0].set_ylabel("time per call (ms)")
    handles = [_legend_handle(i) for i in impls]
    if (g["status"].isin(SLOW)).any():
        handles.append(Line2D([], [], color=INK_2, ls=":", marker="o", markerfacecolor=SURFACE, label="timeout (first run)"))
    handles.append(Patch(facecolor=INK_2, alpha=0.15, label="IQR"))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 9), bbox_to_anchor=(0.5, -0.02),
               handlelength=1.8, columnspacing=1.2)
    if title:
        _suptitle(fig, title)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


# ── 2. speedup over the JS baseline, small multiples ────────────────────────


def _mark_crossover(ax, outs, impl, rule: str, filled: bool) -> int | None:
    c = find_crossover(outs, rule)
    if c.status != "crossover":
        return None
    o = next(o for o in outs if o.size == c.size)
    if not math.isfinite(o.speedup) or o.speedup <= 0:
        return None
    st = impl_style(impl)
    ax.plot([c.size], [o.speedup], ls="none", marker=st["marker"], markersize=8,
            markerfacecolor=st["color"] if filled else SURFACE, markeredgecolor=st["color"] if not filled else SURFACE,
            markeredgewidth=1.2, zorder=6)
    return c.size


def speedup_figure(summary: pd.DataFrame, metric: str, baseline_override: dict[str, str], title: str | None,
                   ncols: int = 3) -> plt.Figure:
    wls = [w for w in WORKLOAD_ORDER if w in set(summary["workload"])] + sorted(
        set(summary["workload"]) - set(WORKLOAD_ORDER))
    nrows = math.ceil(len(wls) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(_width(), 2.05 * nrows + 0.5), squeeze=False)
    seen: dict[str, None] = {}
    any_lower_bound = False
    for k, (ax, wid) in enumerate(zip(axes.flat, wls)):
        g = summary[summary["workload"] == wid]
        base = _baseline(g, wid, baseline_override)
        if base is None:
            ax.set_visible(False)
            continue
        ys_all = []
        for impl in _impls(g):
            if impl == base:
                continue
            st = impl_style(impl)
            outs = [o for o in outcomes(g, impl, base, metric) if o.win_median is not None]
            pts = [(o.size, o.speedup, o.speedup_lo, o.speedup_hi) for o in outs
                   if o.measured and o.speedup > 0]
            if not pts:
                continue
            seen[impl] = None
            x, y, lo, hi = map(np.array, zip(*pts))
            ys_all += list(y)
            ax.fill_between(x, lo, np.minimum(hi, y * 1e3), color=st["color"], alpha=0.12, lw=0)
            ax.plot(x, y, color=st["color"], ls=st["ls"], marker=st["marker"], markeredgecolor=st["color"],
                    markersize=3, zorder=3)
            # Sizes where the baseline timed out: the speedup is a lower bound.
            lb = [(o.size, o.speedup) for o in outs if not o.measured and math.isfinite(o.speedup) and o.speedup > 0]
            if lb:
                any_lower_bound = True
                lx, ly = zip(*lb)
                ys_all += list(ly)
                ax.plot([x[-1], *lx], [y[-1], *ly], color=st["color"], ls=":", lw=1)
                ax.plot(lx, ly, ls="none", marker=6, markersize=5, color=st["color"], zorder=4)  # caret up: "≥"
            r = _mark_crossover(ax, outs, impl, "robust", filled=True)
            m = find_crossover(outs, "median")
            if m.status == "crossover" and (r is None or m.size < r):
                _mark_crossover(ax, outs, impl, "median", filled=False)
        ax.axhline(1, color=INK, lw=0.8, zorder=2)
        _log_y(ax, "×")
        if ys_all:
            lo_y, hi_y = min(min(ys_all), 1), max(max(ys_all), 1)
            ax.set_ylim(lo_y / 1.8, hi_y * 1.8)
        ax.axhspan(ax.get_ylim()[0], 1, color=LOSS_BAND, zorder=0, lw=0)
        size_axis(ax, g["size"], wid)
        info = workload_info(wid)
        panel_title(ax, k, info.title)
        # Curves run bottom-left → top-right, so these corners stay clear.
        ax.text(0.96, 0.04, f"slower than {base}", transform=ax.transAxes, fontsize=6, color=MUTED, va="bottom",
                ha="right")
        ax.text(0.04, 0.96, f"faster than {base}", transform=ax.transAxes, fontsize=6, color=MUTED, va="top")
    for ax in list(axes.flat)[len(wls):]:
        ax.set_visible(False)
    for row in axes:
        row[0].set_ylabel("speedup (baseline / impl.)")
    handles = [_legend_handle(i) for i in sorted(seen, key=impl_sort_key)]
    handles += [
        Line2D([], [], ls="none", marker="o", markersize=7, markerfacecolor=INK_2, markeredgecolor=SURFACE,
               label="crossover (IQRs separate)"),
        Line2D([], [], ls="none", marker="o", markersize=7, markerfacecolor=SURFACE, markeredgecolor=INK_2,
               label="crossover (median only)"),
    ]
    if any_lower_bound:
        handles.append(Line2D([], [], ls="none", marker=6, markersize=5, color=INK_2, label=mt("baseline timed out (≥)")))
    fig.legend(handles=handles, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.01), handlelength=1.8,
               columnspacing=1.0)
    if title:
        _suptitle(fig, f"Speedup over the JS baseline, {METRIC_SHORT[metric]} time ({title})")
    fig.tight_layout(rect=(0, 0.075 if nrows > 1 else 0.13, 1, 1))
    return fig


# ── 3. fastest implementation at each size ──────────────────────────────────


def winners(g: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per size: the implementation with the lowest median, and whether its IQR clears the runner-up's."""
    rows = []
    for size, d in g.groupby("size"):
        ok = d[(d["status"] == "ok") & d[f"{metric}_median"].notna()].sort_values(f"{metric}_median")
        if ok.empty:
            continue
        w = ok.iloc[0]
        r = ok.iloc[1] if len(ok) > 1 else None
        clear = r is None or w[f"{metric}_q3"] < r[f"{metric}_q1"]
        rows.append({"size": size, "winner": w["implementation"], "runner_up": None if r is None else r["implementation"],
                     "margin": np.nan if r is None else r[f"{metric}_median"] / w[f"{metric}_median"], "clear": clear})
    return pd.DataFrame(rows)


def winners_figure(summary: pd.DataFrame, metric: str, title: str | None) -> plt.Figure:
    wls = [w for w in WORKLOAD_ORDER if w in set(summary["workload"])] + sorted(
        set(summary["workload"]) - set(WORKLOAD_ORDER))
    fig, axes = plt.subplots(len(wls), 1, figsize=(_width(), 0.62 * len(wls) + 0.75), squeeze=False)
    seen: dict[str, None] = {}
    for ax, wid in zip(axes[:, 0], wls):
        g = summary[summary["workload"] == wid]
        w = winners(g, metric)
        if w.empty:
            ax.set_visible(False)
            continue
        sizes = list(w["size"])
        # Cells span the geometric midpoints between neighbouring sizes.
        logs = np.log2(sizes)
        edges = np.concatenate([[logs[0] - (logs[1] - logs[0]) / 2 if len(logs) > 1 else logs[0] - 0.5],
                                (logs[1:] + logs[:-1]) / 2,
                                [logs[-1] + (logs[-1] - logs[-2]) / 2 if len(logs) > 1 else logs[-1] + 0.5]])
        for i, row in enumerate(w.itertuples()):
            st = impl_style(row.winner)
            seen[row.winner] = None
            x0, x1 = 2 ** edges[i], 2 ** edges[i + 1]
            ax.add_patch(Rectangle((x0, 0), x1 - x0, 1, facecolor=st["color"], alpha=1.0 if row.clear else 0.4,
                                   edgecolor=SURFACE, lw=1.2, hatch=None if row.clear else "////"))
            ax.plot([math.sqrt(x0 * x1)], [0.5], marker=st["marker"], ls="none", markersize=3.6,
                    color=SURFACE if row.clear else INK, markeredgecolor=SURFACE if row.clear else INK, zorder=4)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.grid(False)
        strip_axes(ax)
        info = workload_info(wid)
        size_axis(ax, sizes, wid, label=False)
        ax.set_xlim(2 ** edges[0], 2 ** edges[-1])
        sym = f"${info.symbol}$" if style.CURRENT["name"] == "arxiv" else info.symbol
        ax.set_ylabel(f"{mt(info.title)}\n({sym})", rotation=0, ha="right", va="center", fontsize=7.5, color=INK)
        ax.tick_params(axis="x", length=2, pad=1.5)
    handles = [Patch(facecolor=impl_style(i)["color"], label=i) for i in sorted(seen, key=impl_sort_key)]
    handles.append(Patch(facecolor="#d9d8d2", hatch="////", edgecolor=INK_2, lw=0,
                         label="within runner-up's IQR"))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 8), bbox_to_anchor=(0.5, -0.01),
               handlelength=1.4, columnspacing=1.2)
    if title:
        _suptitle(fig, f"Fastest implementation by median {METRIC_SHORT[metric]} time ({title})")
    fig.tight_layout(rect=(0, 0.07, 1, 1), h_pad=0.6)
    return fig


# ── 4. crossover map: where each challenger stays ahead ─────────────────────


def resolve_pairs(g: pd.DataFrame, wid: str, baseline_override: dict[str, str],
                  pairs: list[tuple[str, str]] = PAPER_PAIRS) -> list[tuple[str, str]]:
    base = _baseline(g, wid, baseline_override)
    impls = set(g["implementation"])
    out = []
    for a, b in pairs:
        b = base if b == "BASE" else b
        if a in impls and b in impls and (a, b) not in out:
            out.append((a, b))
    return out


def crossover_map_figure(summary: pd.DataFrame, metric: str, baseline_override: dict[str, str],
                         title: str | None, ncols: int = 2) -> plt.Figure:
    wls = [w for w in WORKLOAD_ORDER if w in set(summary["workload"])] + sorted(
        set(summary["workload"]) - set(WORKLOAD_ORDER))
    nrows = math.ceil(len(wls) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(_width(), 1.75 * nrows + 0.55), squeeze=False)
    for k, (ax, wid) in enumerate(zip(axes.flat, wls)):
        g = summary[summary["workload"] == wid]
        info = workload_info(wid)
        pairs = resolve_pairs(g, wid, baseline_override)
        sizes = sorted(g["size"].unique())
        lo, hi = sizes[0], sizes[-1]
        for row, (a, b) in enumerate(pairs):
            y = len(pairs) - 1 - row
            st = impl_style(a)
            outs = outcomes(g, a, b, metric)
            ax.plot([lo, hi], [y, y], color=GRID, lw=4, solid_capstyle="round", zorder=1)
            rob, med = find_crossover(outs, "robust"), find_crossover(outs, "median")
            start_r = {"crossover": rob.size, "always": lo}.get(rob.status)
            start_m = {"crossover": med.size, "always": lo}.get(med.status)
            if start_m is not None and (start_r is None or start_m < start_r):
                ax.plot([start_m, start_r or hi], [y, y], color=st["color"], lw=4, alpha=0.35,
                        solid_capstyle="butt", zorder=2)
            if start_r is not None:
                ax.plot([start_r, hi], [y, y], color=st["color"], lw=4, solid_capstyle="butt", zorder=3)
                ax.plot([start_r], [y], marker=st["marker"], markersize=5.5, color=st["color"],
                        markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=4)
            if rob.status == "crossover":
                label = f"≥ {fmt.size(rob.size, info.pow2_labels, 'mpl')}"
            elif rob.status == "always":
                label = "all sizes"
            elif med.status in ("crossover", "always"):
                label = "median only"
            else:
                label = "never" if rob.status == "never" else "n/a"
            # Label to the right of the start marker when there's room, else to its left.
            x_lab = start_r if start_r is not None else hi
            right = start_r is not None and math.log2(hi / start_r) > 0.42 * math.log2(hi / lo)
            ax.text(x_lab * (2**0.45 if right else 2**-0.45), y + 0.3, mt(label), fontsize=6.5,
                    ha="left" if right else "right", va="center", color=INK_2 if start_r else MUTED, zorder=5)
        ax.set_yticks(range(len(pairs)))
        ax.set_yticklabels([mt(f"{a} › {b}") for a, b in reversed(pairs)], fontsize=7, color=INK)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.6, len(pairs) - 0.4)
        ax.grid(axis="y", visible=False)
        strip_axes(ax)
        size_axis(ax, sizes, wid)
        panel_title(ax, k, info.title)
    for ax in list(axes.flat)[len(wls):]:
        ax.set_visible(False)
    handles = [
        Line2D([], [], color=INK_2, lw=4, label=mt("A › B: A stays ahead (IQRs separate)")),
        Line2D([], [], color=INK_2, lw=4, alpha=0.35, label="ahead by median only"),
        Line2D([], [], color=GRID, lw=4, label="sweep range"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.005), handlelength=2.2)
    if title:
        _suptitle(fig, f"Crossovers (A › B), {METRIC_SHORT[metric]} time ({title})")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig
