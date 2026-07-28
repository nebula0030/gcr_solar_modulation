from __future__ import annotations

import numpy as np
import pytest

from cosmicwatch_io import Events
from rate import RateError, compute_rate


def make_events(timestamps, flags, deadtimes, temps=None, presses=None):
    n = len(timestamps)
    return Events(
        timestamp_s=np.asarray(timestamps, dtype=np.float64),
        flag=np.asarray(flags, dtype=np.int8),
        adc=np.full(n, 500, dtype=np.int32),
        deadtime_s=np.asarray(deadtimes, dtype=np.float64),
        temp_c=np.asarray(temps if temps is not None else [23.0] * n, dtype=np.float64),
        press_pa=np.asarray(
            presses if presses is not None else [100800.0] * n, dtype=np.float64
        ),
        start_utc=np.datetime64("2026-07-10T00:00:00.000000000"),
        clock_drift_s=0.0,
    )


def test_counts_only_coincident_events():
    # Run spans 0..21 s, so 10 s bins give exactly 2 complete bins.
    # Bin 0 covers [0, 10): coincident at t=0 and t=2 -> 2.
    # Bin 1 covers [10, 20]: coincident at t=10 only -> 1.
    ev = make_events(
        timestamps=[0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 20.0, 21.0],
        flags=[1, 0, 1, 0, 1, 0, 0, 0],
        deadtimes=[0.0] * 8,
    )
    rs = compute_rate(ev, bin_length_s=10.0)
    assert rs.counts.tolist() == [2, 1]


def test_livetime_uses_cumulative_deadtime_difference():
    # Cumulative deadtime rises by 1.0 s during bin 0 and 2.0 s during bin 1.
    ev = make_events(
        timestamps=[0.0, 5.0, 10.0, 15.0, 20.0],
        flags=[1, 1, 1, 1, 1],
        deadtimes=[0.0, 1.0, 1.0, 3.0, 3.0],
    )
    rs = compute_rate(ev, bin_length_s=10.0)
    assert rs.livetime_s[0] == pytest.approx(9.0)
    assert rs.livetime_s[1] == pytest.approx(8.0)


def test_rate_and_poisson_error():
    # 101 events at 1 s spacing spans exactly 100 s -> one complete 100 s bin.
    ev = make_events(
        timestamps=[float(i) for i in range(101)],
        flags=[1] * 101,
        deadtimes=[0.0] * 101,
    )
    rs = compute_rate(ev, bin_length_s=100.0)
    assert len(rs.counts) == 1
    assert rs.livetime_s[0] == pytest.approx(100.0)
    assert rs.rate_hz[0] == pytest.approx(rs.counts[0] / 100.0)
    assert rs.rate_err_hz[0] == pytest.approx(np.sqrt(rs.counts[0]) / 100.0)


def test_partial_trailing_bin_is_dropped():
    # Run spans 0..14 s; with 10 s bins the second bin is incomplete.
    ev = make_events(
        timestamps=[0.0, 5.0, 12.0, 14.0],
        flags=[1, 1, 1, 1],
        deadtimes=[0.0] * 4,
    )
    rs = compute_rate(ev, bin_length_s=10.0)
    assert len(rs.counts) == 1


def test_per_bin_meteorology_is_averaged():
    ev = make_events(
        timestamps=[0.0, 5.0, 10.0, 15.0, 20.0],
        flags=[1, 1, 1, 1, 1],
        deadtimes=[0.0] * 5,
        temps=[20.0, 22.0, 30.0, 32.0, 40.0],
        presses=[100000.0, 100200.0, 101000.0, 101200.0, 102000.0],
    )
    rs = compute_rate(ev, bin_length_s=10.0)
    assert rs.temp_c[0] == pytest.approx(21.0)
    assert rs.press_hpa[0] == pytest.approx(1001.0)


def test_bin_longer_than_run_raises():
    ev = make_events([0.0, 1.0], [1, 1], [0.0, 0.0])
    with pytest.raises(RateError) as exc:
        compute_rate(ev, bin_length_s=1000.0)
    assert "bin length" in str(exc.value).lower()


def test_no_coincident_events_raises():
    ev = make_events([0.0, 1.0, 2.0], [0, 0, 0], [0.0, 0.0, 0.0])
    with pytest.raises(RateError) as exc:
        compute_rate(ev, bin_length_s=1.0)
    assert "coincident" in str(exc.value).lower()


def test_mean_fractional_error():
    # Two complete 2 s bins with *different* known counts (100 and 300), so
    # the test actually exercises averaging across bins rather than passing
    # trivially on a single bin.
    bin0 = [i * (2.0 / 100) for i in range(100)]  # 100 events in [0, 2)
    bin1 = [2.0 + i * (2.0 / 300) for i in range(300)]  # 300 events in [2, 4)
    ev = make_events(
        timestamps=bin0 + bin1 + [4.5],  # trailing point pushes span past 4 s
        flags=[1] * 400 + [0],
        deadtimes=[0.0] * 401,
    )
    rs = compute_rate(ev, bin_length_s=2.0)
    assert rs.counts.tolist() == [100, 300]
    assert rs.livetime_s.tolist() == pytest.approx([2.0, 2.0])

    # Expected value computed explicitly from the known per-bin counts,
    # independent of the rate_hz/rate_err_hz arrays under test.
    frac0 = (np.sqrt(100) / 2.0) / (100 / 2.0)
    frac1 = (np.sqrt(300) / 2.0) / (300 / 2.0)
    expected = (frac0 + frac1) / 2.0
    assert rs.mean_fractional_error == pytest.approx(expected)


def test_dead_bin_livetime_is_nan_and_excluded_from_mean():
    # 3 bins of 10 s. Bin 1's cumulative deadtime jumps by 15 s (more than
    # the 10 s bin width), so that bin is unusable. Bins 0 and 2 have normal
    # small deadtime increases and should be unaffected.
    ev = make_events(
        timestamps=[0.0, 2.0, 5.0, 8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 22.0, 25.0, 28.0, 30.0],
        flags=[1] * 13,
        deadtimes=[0.0, 0.2, 0.5, 0.8, 1.0, 5.0, 10.0, 14.0, 16.0, 16.2, 16.5, 16.8, 17.0],
    )
    rs = compute_rate(ev, bin_length_s=10.0)
    assert rs.counts.tolist() == [4, 4, 5]
    assert rs.livetime_s[0] == pytest.approx(9.0)
    assert rs.livetime_s[1] == pytest.approx(-5.0)  # diagnostic value preserved
    assert rs.livetime_s[2] == pytest.approx(9.0)

    assert np.isnan(rs.rate_hz[1])
    assert np.isnan(rs.rate_err_hz[1])

    assert np.isfinite(rs.rate_hz[0])
    assert np.isfinite(rs.rate_hz[2])
    assert rs.rate_hz[0] == pytest.approx(4 / 9.0)
    assert rs.rate_hz[2] == pytest.approx(5 / 9.0)

    frac0 = (np.sqrt(4) / 9.0) / (4 / 9.0)
    frac2 = (np.sqrt(5) / 9.0) / (5 / 9.0)
    expected = (frac0 + frac2) / 2.0
    assert np.isfinite(rs.mean_fractional_error)
    assert rs.mean_fractional_error == pytest.approx(expected)


def test_empty_events_raises_rate_error():
    ev = make_events([], [], [])
    with pytest.raises(RateError):
        compute_rate(ev, bin_length_s=10.0)


def test_gap_bins_are_nan_when_coverage_has_a_hole():
    # Events cover [0,10] and [30,40]; the 10 s bins over [10,30) are a gap.
    ev = make_events(
        timestamps=[0.0, 5.0, 10.0, 30.0, 35.0, 40.0],
        flags=[1, 1, 1, 1, 1, 1],
        deadtimes=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    ev.coverage_s = [(0.0, 10.0), (30.0, 40.0)]
    rs = compute_rate(ev, bin_length_s=10.0)
    # bins: [0,10],[10,20],[20,30],[30,40] -> gap bins 1 and 2 are NaN.
    assert not np.isnan(rs.rate_hz[0])
    assert np.isnan(rs.rate_hz[1])
    assert np.isnan(rs.rate_hz[2])
    assert not np.isnan(rs.rate_hz[3])


def test_partial_coverage_bin_uses_covered_duration_for_livetime():
    # One bin [0,10] covered only for its first 4 s.
    ev = make_events(
        timestamps=[0.0, 1.0, 2.0, 3.0, 10.0, 11.0],
        flags=[1, 1, 1, 1, 1, 1],
        deadtimes=[0.0] * 6,
    )
    ev.coverage_s = [(0.0, 4.0), (10.0, 11.0)]
    rs = compute_rate(ev, bin_length_s=10.0)
    # bin 0 covered duration is 4 s, not 10 s.
    assert rs.livetime_s[0] == pytest.approx(4.0)


def test_default_coverage_reproduces_full_span_behaviour():
    # No coverage_s set -> identical to a continuous run.
    ev = make_events(
        timestamps=[float(i) for i in range(101)],
        flags=[1] * 101,
        deadtimes=[0.0] * 101,
    )
    rs = compute_rate(ev, bin_length_s=100.0)
    assert rs.livetime_s[0] == pytest.approx(100.0)
    assert not np.any(np.isnan(rs.rate_hz))
