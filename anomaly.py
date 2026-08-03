"""Exact-Poisson anomaly statistics for the muon rate.

Pure numpy + math (no scipy). The Poisson CDF is the regularized upper
incomplete gamma, so it is O(iterations) rather than O(k) even at the
~1000s-of-counts scale of an hour-binned run.
"""
from __future__ import annotations

import math
import numpy as np

_ITMAX = 300
_EPS = 3.0e-14
_FPMIN = 1.0e-300


def _gser(a: float, x: float) -> float:
    """Regularized lower incomplete gamma P(a, x) by series (x < a+1)."""
    if x <= 0.0:
        return 0.0
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(_ITMAX):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * _EPS:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gcf(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x) by continued fraction."""
    b = x + 1.0 - a
    c = 1.0 / _FPMIN
    d = 1.0 / b
    h = d
    for i in range(1, _ITMAX):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = b + an / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def _gammq(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x) = 1 - P(a, x)."""
    if x < 0.0 or a <= 0.0:
        raise ValueError("invalid arguments to _gammq")
    if x < a + 1.0:
        return 1.0 - _gser(a, x)
    return _gcf(a, x)


def poisson_cdf(k: int, lam: float) -> float:
    """P(N <= k) for N ~ Poisson(lam), via Q(k+1, lam)."""
    if k < 0:
        return 0.0
    if lam <= 0.0:
        return 1.0
    return _gammq(k + 1.0, lam)


def baseline_mean(corrected_rate_hz: np.ndarray) -> float:
    r = np.asarray(corrected_rate_hz, dtype=float)
    if not np.any(np.isfinite(r)):
        return float("nan")
    return float(np.nanmean(r))


def lambda_per_bin(mu: float, livetime_s: np.ndarray, adj: np.ndarray) -> np.ndarray:
    return mu * np.asarray(livetime_s, dtype=float) / np.asarray(adj, dtype=float)


def threshold_counts(lam: float, p: float) -> tuple:
    """Two-sided exact-Poisson critical counts (k_lo, k_hi) at total tail p."""
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0, 1)")
    half = p / 2.0
    spread = int(10.0 * math.sqrt(lam)) + 10
    hi_bound = int(lam) + spread
    lo_bound = max(0, int(lam) - spread)

    # k_lo: largest k with cdf(k) <= half (or -1 if even cdf(0) > half)
    if poisson_cdf(0, lam) > half:
        k_lo = -1
    else:
        lo, hi = 0, hi_bound
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if poisson_cdf(mid, lam) <= half:
                lo = mid
            else:
                hi = mid - 1
        k_lo = lo

    # k_hi: smallest k with 1 - cdf(k-1) <= half
    lo, hi = lo_bound, hi_bound
    while lo < hi:
        mid = (lo + hi) // 2
        if 1.0 - poisson_cdf(mid - 1, lam) <= half:
            hi = mid
        else:
            lo = mid + 1
    k_hi = lo
    return k_lo, k_hi


def threshold_rates(k_counts: np.ndarray, livetime_s: np.ndarray,
                    adj: np.ndarray) -> np.ndarray:
    return (np.asarray(adj, dtype=float) * np.asarray(k_counts, dtype=float)
            / np.asarray(livetime_s, dtype=float))


def flag_bins(counts: np.ndarray, k_lo: np.ndarray, k_hi: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts)
    return (counts <= np.asarray(k_lo)) | (counts >= np.asarray(k_hi))
