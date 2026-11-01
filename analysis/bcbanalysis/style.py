"""Figure styles for print: `arxiv` (LaTeX look, the default) and `clean` (sans-serif, gray chrome).

Both keep one fixed color + marker per implementation and embed TrueType fonts.

Colors: the JS baseline is ink and the library sorts are muted gray (reference lines). The five
other implementations take the first five slots of a categorical palette that was checked with a
color-vision-deficiency validator (adjacent pairs ΔE ≥ 9.1 under protan/deutan/tritan simulation,
≥ 19.6 for normal vision). The three slots that stay distinguishable on *every* pair go to the
series behind the key crossovers (gpu-opt, wasm-threads, wasm-simd). Each implementation also has
its own marker, so identity never rests on color alone (grayscale print, CVD).
"""
from __future__ import annotations

import logging
import re

import matplotlib as mpl

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"
LOSS_BAND = "#f4f3ef"

IMPL_STYLE: dict[str, dict] = {
    # baselines and reference points
    "js": dict(color=INK, marker="o", ls="-"),
    "js-radix": dict(color=INK, marker="o", ls="-"),
    "js-builtin": dict(color=MUTED, marker="x", ls="--"),
    "wasm-std": dict(color=MUTED, marker="+", ls=":"),
    # categorical slots (1 blue, 2 orange, 3 aqua, 4 yellow, 5 magenta)
    "gpu-opt": dict(color="#2a78d6", marker="D", ls="-"),
    "wasm-threads": dict(color="#eb6834", marker="^", ls="-"),
    "wasm-simd": dict(color="#1baf7a", marker="s", ls="-"),
    "wasm-scalar": dict(color="#eda100", marker="v", ls="-"),
    "gpu-naive": dict(color="#e87ba4", marker="P", ls="-"),
}
_FALLBACK_MARKERS = "*h<>p8"


def impl_style(impl: str) -> dict:
    if impl in IMPL_STYLE:
        return IMPL_STYLE[impl]
    # Not one of the known implementations: gray, distinguished by marker only (never a new hue).
    i = sum(map(ord, impl)) % len(_FALLBACK_MARKERS)
    return dict(color=INK_2, marker=_FALLBACK_MARKERS[i], ls="-.")


#: Active preset, set by apply(): name, figure width (in), panel labels "(a) …".
CURRENT: dict = {"name": "arxiv", "width": 6.5, "panel_labels": True}

STYLES = ("arxiv", "clean")
DEFAULT_WIDTH = {
    # LaTeX `article` with 1 in margins (the most common arXiv layout). Set --width to your
    # \textwidth (in inches) so the figure text keeps its size when included at \textwidth.
    "arxiv": 6.5,
    # A two-column paper's text width (IEEE 7.16 in, ACM 7.0 in).
    "clean": 7.0,
}

_MATH = [
    (re.compile(r"(\w)×(\w)"), r"$\1\\times \2$"),  # N×N, 5×5 → math, as LaTeX would set them
    (re.compile("×"), r"$\\times$"),
    # Relations get no text spaces around them: mathtext already pads them. (A plain '>' in
    # Computer Modern text renders as '¿', so it must be math.)
    (re.compile(r"≥\s*"), r"${\\geq}$"),
    (re.compile(r"\s*›\s*"), r"$>$"),
    (re.compile("·"), r"$\\cdot$"),
    (re.compile("≈"), r"$\\approx$"),
    (re.compile("—"), "--"),
]


def mt(text: str) -> str:
    """Text for a figure. Computer Modern (arxiv) has no glyphs for ×, ≥, ›, · …: typeset them as math."""
    if CURRENT["name"] != "arxiv":
        return text
    for pat, rep in _MATH:
        text = pat.sub(rep, text)
    return text


def panel_title(ax, index: int, text: str) -> None:
    """arxiv: centred '(a) Text' like a subfigure caption; clean: bold, left-aligned."""
    if CURRENT["panel_labels"]:
        ax.set_title(f"({chr(ord('a') + index)}) {mt(text)}")
    else:
        ax.set_title(text, loc="left")


def strip_axes(ax) -> None:
    """For band charts (winners, crossover map): only the bottom axis line, no box."""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(top=False, right=False, left=False, which="both")
    ax.tick_params(axis="x", direction="out")


_COMMON = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    # Embed TrueType (Type 42) fonts: many venues and arXiv's checks reject Type 3 fonts in PDFs.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}

_PRESETS = {
    # The look of a LaTeX paper: Computer Modern text and math (bundled with matplotlib, so no TeX
    # install is needed), boxed axes with inward ticks on all four sides, minor ticks on log axes,
    # a light dotted grid, black text, framed legend, normal-weight titles.
    "arxiv": {
        "font.family": "serif",
        "font.serif": ["cmr10"],
        "mathtext.fontset": "cm",
        "mathtext.rm": "serif",
        "axes.formatter.use_mathtext": True,  # cmr10 has no Unicode minus
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.titleweight": "normal",
        "axes.titlepad": 4,
        "axes.labelsize": 8,
        "axes.labelcolor": "black",
        "axes.edgecolor": "black",
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "axes.grid.which": "major",
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.axisbelow": True,
        "grid.color": "#cfcfcf",
        "grid.linestyle": ":",
        "grid.linewidth": 0.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.labelcolor": "black",
        "ytick.labelcolor": "black",
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "legend.fontsize": 7.5,
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.framealpha": 1.0,
        "legend.edgecolor": "black",
        "patch.linewidth": 0.5,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.2,
        "text.color": "black",
    },
    # Sans-serif, open axes, recessive gray chrome (the original look).
    "clean": {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.titleweight": "bold",
        "axes.labelsize": 7.5,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "axes.grid.which": "major",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.size": 0,
        "ytick.minor.size": 0,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "lines.linewidth": 1.3,
        "lines.markersize": 3.6,
        "text.color": INK,
    },
}


def apply(name: str = "arxiv", width: float | None = None) -> None:
    if name not in _PRESETS:
        raise ValueError(f"unknown style {name!r}; choose from {STYLES}")
    # fontTools logs harmless notes while subsetting fonts into the PDF.
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    mpl.rcdefaults()
    mpl.rcParams.update(_COMMON | _PRESETS[name])
    CURRENT.update(name=name, width=width or DEFAULT_WIDTH[name], panel_labels=name == "arxiv")
