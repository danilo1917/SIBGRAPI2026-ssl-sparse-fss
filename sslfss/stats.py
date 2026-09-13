from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats


def mean_ci(
    values: list[float] | np.ndarray,
    confidence: float = 0.99,
) -> tuple[float, float, float]:
    """Return ``(mean, ci_low, ci_high)`` using the Z-distribution."""
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be in (0, 1)")

    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)

    if n == 0:
        return float("nan"), float("nan"), float("nan")

    m = float(arr.mean())

    if n < 2:
        return m, m, m

    se = arr.std(ddof=1) / np.sqrt(n)

    if se == 0.0:
        return m, m, m

    lo, hi = _scipy_stats.norm.interval(confidence, loc=m, scale=se)
    return m, float(lo), float(hi)


def query_mean_ci(
    dice: list[float] | np.ndarray,
    query_idx: list[int] | np.ndarray,
    confidence: float = 0.99,
) -> tuple[float, float, float]:
    """Return ``(mean, ci_low, ci_high)`` treating each query image as the independent unit of observation."""
    dice = np.asarray(dice, dtype=float)
    query_idx = np.asarray(query_idx)

    query_means = (
        pd.Series(dice)
          .groupby(query_idx)
          .mean()
          .values
    )

    return mean_ci(query_means, confidence)


def ci_bounds(
    values: list[float] | np.ndarray,
    confidence: float = 0.99,
) -> tuple[float, float]:
    """Return ``(ci_low, ci_high)`` — suitable for ``ax.fill_between``."""
    _, lo, hi = mean_ci(values, confidence)
    return lo, hi


def ci_half_width(
    values: list[float] | np.ndarray,
    confidence: float = 0.99,
) -> float:
    """Return the half-width of the confidence interval."""
    m, _, hi = mean_ci(values, confidence)
    return hi - m