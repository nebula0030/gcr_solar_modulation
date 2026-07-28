"""Bin coincident muon events into a rate time series.

Only ``Flag == 1`` (coincident) events count toward the rate. The raw
all-event rate of a v3X detector is dominated by low-ADC noise; the
coincident subset is the muon-like sample.

Livetime per bin is the bin duration minus the *increase* in the detector's
cumulative deadtime counter across that bin.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from cosmicwatch_io import Events


class RateError(ValueError):
    """Raised when a run cannot produce a usable rate series."""


@dataclass
class RateSeries:
    """Binned coincident-muon rate with per-bin meteorology."""

    bin_start_utc: np.ndarray
    bin_mid_utc: np.ndarray
    counts: np.ndarray
    livetime_s: np.ndarray
    rate_hz: np.ndarray
    rate_err_hz: np.ndarray
    press_hpa: np.ndarray
    temp_c: np.ndarray
    bin_length_s: float

    @property
    def mean_fractional_error(self) -> float:
        """Mean Poisson error as a fraction of the rate.

        Bins with a non-finite rate (dead bins where livetime was not
        strictly positive) are excluded rather than poisoning the mean.
        """
        good = np.isfinite(self.rate_hz) & (self.rate_hz > 0)
        if not np.any(good):
            return float("nan")
        return float(np.mean(self.rate_err_hz[good] / self.rate_hz[good]))


def compute_rate(events: Events, bin_length_s: float) -> RateSeries:
    """Bin coincident events into fixed-width bins.

    Bins are anchored at the first event's timestamp. Any trailing partial
    bin is dropped so every reported bin covers a full interval.
    """
    if bin_length_s <= 0:
        raise RateError("bin length must be positive, got {0}".format(bin_length_s))

    t = events.timestamp_s
    if len(t) == 0:
        raise RateError("no events in this run; cannot compute a rate")
    run_span = float(t[-1] - t[0])
    if bin_length_s > run_span:
        raise RateError(
            "bin length {0:g} s exceeds the run duration {1:g} s".format(
                bin_length_s, run_span
            )
        )

    n_bins = int(np.floor(run_span / bin_length_s))
    if n_bins < 1:
        # Defensive/unreachable: the bin_length_s > run_span guard above
        # already rejects any bin_length_s that would floor to zero bins.
        raise RateError(
            "bin length {0:g} s yields no complete bins over {1:g} s".format(
                bin_length_s, run_span
            )
        )

    t0 = float(t[0])
    edges = t0 + bin_length_s * np.arange(n_bins + 1, dtype=np.float64)

    coincident = events.flag == 1
    if not np.any(coincident):
        raise RateError(
            "no coincident (Flag == 1) events in this run; nothing to bin"
        )

    counts, _ = np.histogram(t[coincident], bins=edges)
    counts = counts.astype(np.int64)

    # Cumulative deadtime interpolated onto the bin edges, then differenced.
    deadtime_at_edges = np.interp(edges, t, events.deadtime_s)
    deadtime_per_bin = np.diff(deadtime_at_edges)
    covered_per_bin = _covered_per_bin(edges, events.coverage())
    livetime_s = covered_per_bin - deadtime_per_bin

    # A bin whose livetime is not strictly positive (deadtime increase >=
    # the bin width, i.e. the detector was effectively dead the whole bin)
    # is not a real measurement. Mark its rate as NaN rather than clamping
    # livetime to something tiny, which would fabricate an absurd rate and
    # silently poison mean_fractional_error. counts/livetime_s are left as
    # the actual computed values for diagnostics.
    dead_bin = livetime_s <= 0
    with np.errstate(divide="ignore", invalid="ignore"):
        rate_hz = counts / livetime_s
        rate_err_hz = np.sqrt(counts) / livetime_s
    rate_hz = np.where(dead_bin, np.nan, rate_hz)
    rate_err_hz = np.where(dead_bin, np.nan, rate_err_hz)

    press_hpa = _bin_mean(t, events.press_pa / 100.0, edges)
    temp_c = _bin_mean(t, events.temp_c, edges)

    offsets_ns = ((edges[:-1] - t0) * 1e9).astype("timedelta64[ns]")
    bin_start_utc = events.start_utc + offsets_ns
    bin_mid_utc = bin_start_utc + np.timedelta64(
        int(bin_length_s * 1e9 / 2), "ns"
    )

    return RateSeries(
        bin_start_utc=bin_start_utc,
        bin_mid_utc=bin_mid_utc,
        counts=counts,
        livetime_s=livetime_s,
        rate_hz=rate_hz,
        rate_err_hz=rate_err_hz,
        press_hpa=press_hpa,
        temp_c=temp_c,
        bin_length_s=float(bin_length_s),
    )


def _covered_per_bin(
    edges: np.ndarray, coverage: List[Tuple[float, float]]
) -> np.ndarray:
    """Covered seconds within each bin: overlap of the bin with coverage.

    Coverage intervals are disjoint (spliced files never overlap), so summing
    per-interval overlaps is exact.
    """
    covered = np.zeros(len(edges) - 1, dtype=np.float64)
    for c0, c1 in coverage:
        lo = np.maximum(edges[:-1], c0)
        hi = np.minimum(edges[1:], c1)
        covered += np.maximum(0.0, hi - lo)
    return covered


def _bin_mean(t: np.ndarray, values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Mean of ``values`` per bin, using all events (not just coincident)."""
    totals, _ = np.histogram(t, bins=edges, weights=values)
    counts, _ = np.histogram(t, bins=edges)
    return totals / np.maximum(counts, 1)
