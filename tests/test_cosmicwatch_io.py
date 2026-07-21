from __future__ import annotations

import os

import numpy as np
import pytest

from cosmicwatch_io import DataFormatError, read_events

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def test_reads_all_events():
    ev = read_events(os.path.join(FIXTURES, "sample_13col.txt"))
    assert len(ev.timestamp_s) == 6
    assert ev.flag.tolist() == [1, 0, 1, 1, 0, 1]
    assert ev.adc.tolist() == [808, 77, 450, 211, 202, 68]


def test_deadtime_is_cumulative_and_monotonic():
    ev = read_events(os.path.join(FIXTURES, "sample_13col.txt"))
    assert np.all(np.diff(ev.deadtime_s) > 0)
    assert ev.deadtime_s[-1] == pytest.approx(0.006)


def test_pressure_converted_to_hpa():
    ev = read_events(os.path.join(FIXTURES, "sample_13col.txt"))
    assert ev.press_hpa[0] == pytest.approx(1008.0)


def test_start_utc_parsed_as_day_month_year():
    ev = read_events(os.path.join(FIXTURES, "sample_13col.txt"))
    # 10/07/2026 is 10 July 2026, not 7 October.
    assert str(ev.start_utc).startswith("2026-07-10T00:00:00")


def test_utc_derived_from_start_plus_timestamp_offset():
    ev = read_events(os.path.join(FIXTURES, "sample_13col.txt"))
    deltas = (ev.utc - ev.start_utc) / np.timedelta64(1, "s")
    assert deltas.tolist() == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])


def test_clock_drift_is_reported_and_small_for_consistent_file():
    ev = read_events(os.path.join(FIXTURES, "sample_13col.txt"))
    assert abs(ev.clock_drift_s) < 0.01


def test_ten_column_file_is_rejected():
    with pytest.raises(DataFormatError) as exc:
        read_events(os.path.join(FIXTURES, "sample_10col.txt"))
    assert "13" in str(exc.value)


def test_missing_file_raises():
    with pytest.raises((FileNotFoundError, OSError)):
        read_events(os.path.join(FIXTURES, "does_not_exist.txt"))


def test_truncated_final_line_is_skipped(tmp_path):
    src = os.path.join(FIXTURES, "sample_13col.txt")
    text = open(src).read()
    truncated = tmp_path / "truncated.txt"
    truncated.write_text(text + "8\t6.000000\t1\t99\n")
    ev = read_events(str(truncated))
    assert len(ev.timestamp_s) == 6
