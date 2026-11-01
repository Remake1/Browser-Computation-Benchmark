"""Size formatting for tables and axes (2¹⁰ / 3·2¹⁰ for power-of-two sweeps, plain integers otherwise)."""
from __future__ import annotations

import math

_SUP = str.maketrans("0123456789.-", "⁰¹²³⁴⁵⁶⁷⁸⁹·⁻")


def _pow2_parts(n: int) -> tuple[int, int] | None:
    """n = m·2^k with m in (1, 3); None otherwise."""
    for m in (1, 3):
        if n % m == 0:
            q = n // m
            if q > 0 and q & (q - 1) == 0:
                return m, q.bit_length() - 1
    return None


def size(n: int | float | None, pow2: bool, style: str = "text") -> str:
    """style: 'text' (Unicode, Markdown), 'tex' (LaTeX math), 'mpl' (matplotlib mathtext)."""
    if n is None or (isinstance(n, float) and math.isnan(n)):
        return ""
    n = int(n)
    parts = _pow2_parts(n) if pow2 else None
    if parts and n >= 1024:
        m, k = parts
        if style == "text":
            return ("3·" if m == 3 else "") + "2" + str(k).translate(_SUP)
        pre = r"3\cdot " if m == 3 else ""
        return f"${pre}2^{{{k}}}$"
    if style == "tex":
        return f"{n:,}".replace(",", "{,}") if n >= 10000 else str(n)
    return f"{n:,}" if style == "text" else str(n)


def approx_size(x: float | None, pow2: bool, style: str = "text") -> str:
    """An interpolated (non-integer) crossover size."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    if pow2:
        k = math.log2(x)
        if style == "text":
            return "≈2" + f"{k:.1f}".translate(_SUP)
        return f"$\\approx 2^{{{k:.1f}}}$"
    digits = max(0, int(math.floor(math.log10(x))) - 1)
    r = int(round(x, -digits))
    s = f"{r:,}"
    if style == "tex":
        return f"$\\approx${s.replace(',', '{,}')}"
    return f"≈{s}"
