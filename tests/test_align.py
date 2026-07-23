from __future__ import annotations

import numpy as np
import pytest

from align import align_to_bins
from external_sources import ExternalSeries
from rate import RateSeries


def make_rate_series(n=4, bin_length_s=3600.0):
    start = np.datetime64("2026-07-10T00:00:00.000000000")
    offsets = (np.arange(n) * bin_length_s * 1e9).astype("timedelta64[ns]")
    return RateSeries(
        bin_start_utc=start + offsets,
        bin_mid_utc=start + offsets + np.timedelta64(int(bin_length_s * 5e8), "ns"),
        counts=np.full(n, 100, dtype=np.int64),
        livetime_s=np.full(n, bin_length_s),
        rate_hz=np.full(n, 0.5),
        rate_err_hz=np.full(n, 0.05),
        press_hpa=np.full(n, 1008.0),
        temp_c=np.full(n, 23.0),
        bin_length_s=bin_length_s,
    )


def make_external(times, values, name="test"):
    return ExternalSeries(
        name=name,
        units="u",
        source="src",
        utc=np.asarray(times, dtype="datetime64[ns]"),
        values=np.asarray(values, dtype=np.float64),
    )


def test_samples_inside_a_bin_are_averaged_and_marked_measured():
    rs = make_rate_series(n=2)
    ext = make_external(
        ["2026-07-10T00:10:00", "2026-07-10T00:50:00", "2026-07-10T01:30:00"],
        [10.0, 20.0, 40.0],
    )
    aligned = align_to_bins(ext, rs)
    assert aligned.values[0] == pytest.approx(15.0)
    assert aligned.values[1] == pytest.approx(40.0)
    assert not aligned.interpolated[0]
    assert not aligned.interpolated[1]


def test_bins_without_samples_are_interpolated_and_flagged():
    """Kp is 3-hourly; with 1-hour bins most bins have no native sample."""
    rs = make_rate_series(n=4)
    ext = make_external(
        ["2026-07-10T00:00:00", "2026-07-10T03:00:00"], [1.0, 4.0]
    )
    aligned = align_to_bins(ext, rs)
    assert not aligned.interpolated[0]
    assert aligned.interpolated[1]
    # Bin 1's midpoint is 01:30, halfway between the 00:00 and 03:00 samples
    # at the 50% mark of a 1.0 -> 4.0 ramp.
    assert aligned.values[1] == pytest.approx(2.5)


def test_bins_outside_the_data_range_are_nan():
    rs = make_rate_series(n=4)
    ext = make_external(["2026-07-10T00:30:00"], [7.0])
    aligned = align_to_bins(ext, rs)
    assert aligned.values[0] == pytest.approx(7.0)
    assert np.isnan(aligned.values[3])


def test_metadata_is_carried_through():
    rs = make_rate_series(n=2)
    ext = make_external(["2026-07-10T00:30:00"], [1.0], name="Kp index")
    aligned = align_to_bins(ext, rs)
    assert aligned.name == "Kp index"
    assert aligned.units == "u"
    assert aligned.source == "src"


def test_percent_deviation_is_relative_to_the_mean():
    rs = make_rate_series(n=2)
    ext = make_external(
        ["2026-07-10T00:30:00", "2026-07-10T01:30:00"], [90.0, 110.0]
    )
    aligned = align_to_bins(ext, rs)
    assert aligned.percent_deviation[0] == pytest.approx(-10.0)
    assert aligned.percent_deviation[1] == pytest.approx(10.0)


def test_empty_external_series_yields_all_nan():
    rs = make_rate_series(n=3)
    ext = make_external([], [])
    aligned = align_to_bins(ext, rs)
    assert np.all(np.isnan(aligned.values))
