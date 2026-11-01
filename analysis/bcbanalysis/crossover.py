"""Crossover detection (PLAN.md §9, milestone 6).

The crossover of a challenger A over a reference B is the smallest size N such that A beats B at N
**and at every larger size** of the sweep. Two rules decide whether A beats B at one size:

* ``robust`` (the PLAN's definition): A's median is lower **and the IQRs don't overlap**,
  i.e. q3(A) < q1(B);
* ``median``: A's median is lower.

Sizes where an implementation was too slow (``timeout``, or ``skipped`` after a timeout) count as
a loss for it: an implementation that finished beats one that hit the time limit. Sizes where both
were too slow, where B failed (invalid / error / unsupported) or where the metric is unavailable
(no GPU timestamps) are left out. A result A failed to verify counts as a loss for A.

Outcomes:
  ``crossover`` – A loses at some size and wins from N on;
  ``always``    – A wins at every compared size (the crossover is at or below the smallest size);
  ``never``     – A does not win at the largest compared size (it may win in a band before that);
  ``no data``   – no size could be compared.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .stats import METRICS, SLOW

RULES = ("robust", "median")


@dataclass
class SizeOutcome:
    size: int
    #: True / False, or None when the size is left out.
    win_median: bool | None
    win_robust: bool | None
    #: B's median / A's median (> 1: A faster); a lower bound when B timed out; NaN if unknown.
    speedup: float = math.nan
    #: Conservative speedup interval from the quartiles: [q1(B)/q3(A), q3(B)/q1(A)].
    speedup_lo: float = math.nan
    speedup_hi: float = math.nan
    #: Both medians are measured, so the crossover may be interpolated across this size.
    measured: bool = False
    note: str = ""


@dataclass
class Crossover:
    status: str  # 'crossover' | 'always' | 'never' | 'no data'
    size: int | None = None
    #: The compared size just below the crossover (where A did not win).
    prev_size: int | None = None
    #: Log-log interpolated size where the medians are equal (median rule only).
    interpolated: float | None = None
    #: Sizes where A wins before its final lead (wins that don't last).
    transient_wins: list[int] = field(default_factory=list)


def _get(row: pd.Series | None, metric: str, stat: str) -> float:
    if row is None:
        return math.nan
    v = row.get(f"{metric}_{stat}", math.nan)
    return math.nan if v is None else float(v)


def compare(a: pd.Series | None, b: pd.Series | None, metric: str, size: int) -> SizeOutcome:
    sa = a["status"] if a is not None else "missing"
    sb = b["status"] if b is not None else "missing"
    if sa == "missing" or sb == "missing":
        return SizeOutcome(size, None, None, note=f"not measured ({'A' if sa == 'missing' else 'B'})")
    if sa in SLOW and sb in SLOW:
        return SizeOutcome(size, None, None, note="both too slow")
    if sa in SLOW:
        return SizeOutcome(size, False, False, speedup=0.0, note=f"A {sa}")
    if sa != "ok":
        return SizeOutcome(size, False, False, note=f"A {sa}")
    a_med, a_q1, a_q3 = (_get(a, metric, s) for s in ("median", "q1", "q3"))
    if math.isnan(a_med):
        return SizeOutcome(size, None, None, note="no metric for A")
    if sb in SLOW:
        # B didn't finish within the time limit; its first iteration (if recorded) bounds it from below.
        b_first = _get(b, metric, "first")
        lb = b_first / a_med if not math.isnan(b_first) else math.inf
        return SizeOutcome(size, True, True, speedup=lb, speedup_lo=lb, speedup_hi=math.inf, note=f"B {sb}")
    if sb != "ok":
        return SizeOutcome(size, None, None, note=f"B {sb}")
    b_med, b_q1, b_q3 = (_get(b, metric, s) for s in ("median", "q1", "q3"))
    if math.isnan(b_med):
        return SizeOutcome(size, None, None, note="no metric for B")
    return SizeOutcome(
        size,
        bool(a_med < b_med),
        bool(a_q3 < b_q1),
        speedup=b_med / a_med,
        speedup_lo=b_q1 / a_q3,
        speedup_hi=b_q3 / a_q1 if a_q1 > 0 else math.inf,
        measured=True,
    )


def outcomes(summary: pd.DataFrame, a: str, b: str, metric: str) -> list[SizeOutcome]:
    """Per-size comparison of A against B; ``summary`` is one (browser, workload) slice."""
    by = {impl: g.set_index("size") for impl, g in summary.groupby("implementation")}
    ga, gb = by.get(a), by.get(b)
    sizes = sorted(set(ga.index if ga is not None else []) | set(gb.index if gb is not None else []))
    return [
        compare(
            ga.loc[s] if ga is not None and s in ga.index else None,
            gb.loc[s] if gb is not None and s in gb.index else None,
            metric,
            int(s),
        )
        for s in sizes
    ]


def find_crossover(outs: list[SizeOutcome], rule: str = "robust") -> Crossover:
    attr = f"win_{rule}"
    cmp = [o for o in outs if getattr(o, attr) is not None]
    if not cmp:
        return Crossover("no data")
    wins = [getattr(o, attr) for o in cmp]
    if not wins[-1]:
        return Crossover("never", transient_wins=[o.size for o, w in zip(cmp, wins) if w])
    i = len(wins) - 1
    while i > 0 and wins[i - 1]:
        i -= 1
    transient = [o.size for o, w in zip(cmp[:i], wins[:i]) if w]
    if i == 0:
        return Crossover("always", size=cmp[0].size)
    prev, at = cmp[i - 1], cmp[i]
    interp = None
    if rule == "median" and prev.measured and at.measured and prev.speedup > 0 and at.speedup > 0:
        r0, r1 = math.log(prev.speedup), math.log(at.speedup)  # r0 ≤ 0 < r1
        if r1 > r0:
            t = -r0 / (r1 - r0)
            interp = 2 ** (math.log2(prev.size) + t * (math.log2(at.size) - math.log2(prev.size)))
    return Crossover("crossover", size=at.size, prev_size=prev.size, interpolated=interp, transient_wins=transient)


def crossover_table(summary: pd.DataFrame, pairs: list[tuple[str, str]] | None = None) -> pd.DataFrame:
    """Every ordered pair (or the given pairs) × metric × browser × workload, both rules."""
    rows = []
    for (browser, wl), g in summary.groupby(["browser", "workload"], sort=False):
        impls = list(dict.fromkeys(g["implementation"]))
        want = pairs if pairs is not None else [(x, y) for x in impls for y in impls if x != y]
        for a, b in want:
            if a not in impls or b not in impls:
                continue
            for metric in METRICS:
                outs = outcomes(g, a, b, metric)
                row = {"browser": browser, "workload": wl, "challenger": a, "reference": b, "metric": metric}
                for rule in RULES:
                    c = find_crossover(outs, rule)
                    row[f"{rule}_status"] = c.status
                    row[f"{rule}_size"] = c.size
                    row[f"{rule}_prev_size"] = c.prev_size
                    row[f"{rule}_transient_wins"] = " ".join(map(str, c.transient_wins))
                    if rule == "median":
                        row["median_interpolated"] = c.interpolated
                last = next((o for o in reversed(outs) if o.win_median is not None), None)
                row["largest_size"] = last.size if last else None
                row["speedup_at_largest"] = last.speedup if last else np.nan
                row["compared_sizes"] = sum(o.win_median is not None for o in outs)
                row["notes"] = "; ".join(sorted({o.note for o in outs if o.note}))
                rows.append(row)
    return pd.DataFrame(rows)
