"""Resample external series onto the muon rate's bins.

External sources arrive at their own cadences -- NMDB hourly, Kp 3-hourly,
sunspot daily, GOES per minute. Each is mapped onto the rate's bin edges.

Where a bin contains no native sample the value is interpolated, and every
interpolated point is flagged so plots can say so. That distinction matters:
a daily sunspot number spread across 24 hourly bins is one measurement, not
24, and the hover text must not imply otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from external_sources import ExternalSeries
from rate import RateSeries


@dataclass
class AlignedSeries:
    """An external series resampled onto the rate bins."""

    name: str
    units: str
    source: str
    values: np.ndarray
    interpolated: np.ndarray

    @property
    def percent_deviation(self) -> np.ndarray:
        """Values as a percent deviation from their own mean."""
        finite = np.isfinite(self.values)
        if not np.any(finite):
            return np.full_like(self.values, np.nan)
        mean = float(np.mean(self.values[finite]))
        if mean == 0:
            return np.full_like(self.values, np.nan)
        return 100.0 * (self.values - mean) / mean


def align_to_bins(series: ExternalSeries, rs: RateSeries) -> AlignedSeries:
    """Map ``series`` onto the bins of ``rs``.

    Bins containing native samples get their mean and are marked measured.
    Bins between samples are linearly interpolated and marked interpolated.
    Bins outside the source's coverage are NaN.
    """
    n_bins = len(rs.bin_start_utc)
    values = np.full(n_bins, np.nan, dtype=np.float64)
    interpolated = np.zeros(n_bins, dtype=bool)

    if len(series.utc) == 0:
        return AlignedSeries(
            name=series.name,
            units=series.units,
            source=series.source,
            values=values,
            interpolated=interpolated,
        )

    bin_length_ns = np.timedelta64(int(rs.bin_length_s * 1e9), "ns")
    edges_ns = np.concatenate(
        [
            rs.bin_start_utc.astype("datetime64[ns]").astype(np.int64),
            [(rs.bin_start_utc[-1] + bin_length_ns).astype(np.int64)],
        ]
    )
    sample_ns = series.utc.astype("datetime64[ns]").astype(np.int64)

    # Mean of the samples that fall inside each bin.
    totals, _ = np.histogram(sample_ns, bins=edges_ns, weights=series.values)
    counts, _ = np.histogram(sample_ns, bins=edges_ns)
    measured = counts > 0
    values[measured] = totals[measured] / counts[measured]

    # Fill the remaining bins by interpolating at the bin midpoints, but only
    # inside the source's own coverage.
    missing = ~measured
    if np.any(missing):
        mid_ns = rs.bin_mid_utc.astype("datetime64[ns]").astype(np.int64)
        order = np.argsort(sample_ns)
        sorted_ns = sample_ns[order]
        sorted_values = series.values[order]

        inside = (mid_ns >= sorted_ns[0]) & (mid_ns <= sorted_ns[-1])
        fill = missing & inside
        if np.any(fill):
            values[fill] = np.interp(mid_ns[fill], sorted_ns, sorted_values)
            interpolated[fill] = True

    return AlignedSeries(
        name=series.name,
        units=series.units,
        source=series.source,
        values=values,
        interpolated=interpolated,
    )
