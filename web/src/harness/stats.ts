// Summary statistics. Raw samples are always stored; these are for the UI and
// for calibration. The analysis scripts recompute everything from raw data.

export function sorted(values: readonly number[]): number[] {
  return [...values].sort((a, b) => a - b);
}

/** Linear-interpolated quantile (R type 7, numpy default) of an ascending array. */
export function quantileSorted(asc: readonly number[], q: number): number {
  if (asc.length === 0) return NaN;
  const pos = (asc.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return asc[lo] + (asc[hi] - asc[lo]) * (pos - lo);
}

export function median(values: readonly number[]): number {
  return quantileSorted(sorted(values), 0.5);
}

export interface Summary {
  n: number;
  min: number;
  q1: number;
  median: number;
  q3: number;
  max: number;
  iqr: number;
}

export function summarize(values: readonly number[]): Summary {
  const asc = sorted(values);
  const q1 = quantileSorted(asc, 0.25);
  const q3 = quantileSorted(asc, 0.75);
  return {
    n: asc.length,
    min: asc.length ? asc[0] : NaN,
    q1,
    median: quantileSorted(asc, 0.5),
    q3,
    max: asc.length ? asc[asc.length - 1] : NaN,
    iqr: q3 - q1,
  };
}
