"""Static knowledge about workloads and implementations, mirrored from web/src/workloads/*/index.ts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class WorkloadInfo:
    id: str
    title: str
    #: Symbol of the size parameter, as in the README tables.
    symbol: str
    #: What the size parameter counts, for axis labels.
    size_desc: str
    #: Ticks and table cells as powers of two (2¹⁰, 3·2¹⁰) instead of plain integers.
    pow2_labels: bool
    #: Throughput unit and work per run (same formulas as the UI).
    throughput_unit: str
    work: Callable[[float], float]
    #: Default baseline implementation(s) for speedup plots and the crossover table.
    baselines: tuple[str, ...] = ("js",)


WORKLOADS: dict[str, WorkloadInfo] = {
    w.id: w
    for w in [
        WorkloadInfo("matmul", "Matrix multiply", "N", "N×N matrices", False, "GFLOP/s", lambda n: 2 * n**3 / 1e9),
        WorkloadInfo("reduction", "Reduction", "n", "f32 elements", True, "GB/s", lambda n: 4 * n / 1e9),
        WorkloadInfo("sort", "Sort", "n", "u32 keys", True, "Mkeys/s", lambda n: n / 1e6, ("js-radix", "js-builtin")),
        WorkloadInfo("convolution", "Convolution 5×5", "W", "W×W image", False, "GFLOP/s", lambda w: 50 * w * w / 1e9),
        WorkloadInfo("nbody", "N-body", "n", "bodies", False, "GFLOP/s", lambda n: 20 * n * n / 1e9),
        WorkloadInfo("editdistance", "Edit distance", "L", "string length", False, "Mcells/s", lambda n: n * n / 1e6),
    ]
}

WORKLOAD_ORDER = list(WORKLOADS)


def workload_info(wid: str) -> WorkloadInfo:
    if wid in WORKLOADS:
        return WORKLOADS[wid]
    # Unknown workloads (e.g. 'dummy') still get analysed, with neutral labels.
    return WorkloadInfo(wid, wid, "n", "size", False, "Mitems/s", lambda n: n / 1e6)


#: Legend / table order. Unknown implementations are appended alphabetically.
IMPL_ORDER = [
    "js",
    "js-builtin",
    "js-radix",
    "wasm-std",
    "wasm-scalar",
    "wasm-simd",
    "wasm-threads",
    "gpu-naive",
    "gpu-opt",
]


def impl_sort_key(impl: str) -> tuple[int, str]:
    return (IMPL_ORDER.index(impl) if impl in IMPL_ORDER else len(IMPL_ORDER), impl)


#: Ordered pairs (challenger, reference) for the paper's crossover table. "BASE" is the workload's
#: first baseline (js, or js-radix for sort, which runs the same algorithm as the WASM radix sorts).
PAPER_PAIRS: list[tuple[str, str]] = [
    ("wasm-scalar", "BASE"),
    ("wasm-simd", "BASE"),
    ("wasm-threads", "BASE"),
    ("gpu-opt", "BASE"),
    ("wasm-threads", "wasm-simd"),
    ("gpu-opt", "wasm-simd"),
    ("gpu-opt", "wasm-threads"),
]
