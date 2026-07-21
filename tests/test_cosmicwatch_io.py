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


def test_wrong_width_final_line_is_skipped(tmp_path):
    # This row has only 4 tab-separated fields, so it is rejected by the
    # column-count guard before the per-field try/except is ever reached.
    src = os.path.join(FIXTURES, "sample_13col.txt")
    text = open(src).read()
    truncated = tmp_path / "truncated.txt"
    truncated.write_text(text + "8\t6.000000\t1\t99\n")
    ev = read_events(str(truncated))
    assert len(ev.timestamp_s) == 6


def _row(
    event="99",
    timestamp="6.000000",
    flag="1",
    adc="12345",
    sipm="4.0",
    deadtime="0.007000",
    temp="23.4",
    press="100820",
    accel="0.005:-0.003:-0.998",
    gyro="0.1:0.0:-0.1",
    name="TestDet",
    time="00:00:06.000000",
    date="10/07/2026",
):
    """Build one well-formed (13-field) data row, with fields overridable."""
    return "\t".join(
        [event, timestamp, flag, adc, sipm, deadtime, temp, press, accel, gyro, name, time, date]
    )


def test_row_with_corrupt_time_field_is_fully_excluded(tmp_path):
    src = os.path.join(FIXTURES, "sample_13col.txt")
    text = open(src).read()
    bad_row = _row(time="BADTIME_CORRUPT", adc="777777")
    path = tmp_path / "bad_time.txt"
    path.write_text(text + bad_row + "\n")

    ev = read_events(str(path))

    assert len(ev.timestamp_s) == 6
    assert len(ev.flag) == 6
    assert len(ev.adc) == 6
    assert len(ev.deadtime_s) == 6
    assert len(ev.temp_c) == 6
    assert len(ev.press_pa) == 6
    assert 777777 not in ev.adc.tolist()


def test_row_with_non_numeric_adc_is_fully_excluded(tmp_path):
    src = os.path.join(FIXTURES, "sample_13col.txt")
    text = open(src).read()
    bad_row = _row(adc="notanumber", deadtime="0.999999")
    path = tmp_path / "bad_adc.txt"
    path.write_text(text + bad_row + "\n")

    ev = read_events(str(path))

    assert len(ev.timestamp_s) == 6
    assert len(ev.flag) == 6
    assert len(ev.adc) == 6
    assert len(ev.deadtime_s) == 6
    assert len(ev.temp_c) == 6
    assert len(ev.press_pa) == 6
    assert 0.999999 not in ev.deadtime_s.tolist()


def test_corrupt_first_row_does_not_anchor_start_utc(tmp_path):
    # Header, then a bad first data row (corrupt Time), then two good rows.
    header_lines = [
        "#" * 20,
        "# CosmicWatch: The Desktop Muon Detector v3X",
        "# Questions? saxani@udel.edu",
        "# Event  Timestamp[s]  Flag  ADC[12b]  SiPM[mV]  Deadtime[s]  Temp[C]"
        "  Press[Pa]  Accel(X:Y:Z)[g]  Gyro(X:Y:Z)[deg/sec]  Name  Time  Date",
        "#" * 20,
    ]
    bad_first_row = _row(
        event="1", timestamp="0.000000", adc="999999", time="BADTIME_CORRUPT",
        date="10/07/2026",
    )
    good_row_1 = _row(
        event="2", timestamp="1.000000", adc="111",
        time="00:00:01.000000", date="10/07/2026",
    )
    good_row_2 = _row(
        event="3", timestamp="2.000000", adc="222",
        time="00:00:02.000000", date="10/07/2026",
    )
    path = tmp_path / "bad_first.txt"
    path.write_text(
        "\n".join(header_lines + [bad_first_row, good_row_1, good_row_2]) + "\n"
    )

    ev = read_events(str(path))

    assert len(ev.timestamp_s) == 2
    assert 999999 not in ev.adc.tolist()
    # start_utc must anchor on the first VALID row (event 2, 00:00:01), not
    # the skipped corrupt row (event 1, 00:00:00).
    assert str(ev.start_utc).startswith("2026-07-10T00:00:01")
    deltas = (ev.utc - ev.start_utc) / np.timedelta64(1, "s")
    assert deltas.tolist() == pytest.approx([0.0, 1.0])


def test_chunking_does_not_change_result(tmp_path):
    # A tiny chunk_size forces multiple chunk boundaries within the fixture's
    # 6 data rows; the result must be byte-identical to a single large chunk.
    src = os.path.join(FIXTURES, "sample_13col.txt")
    ev_small_chunks = read_events(src, chunk_size=2)
    ev_one_chunk = read_events(src, chunk_size=1_000_000)

    assert ev_small_chunks.timestamp_s.tobytes() == ev_one_chunk.timestamp_s.tobytes()
    assert ev_small_chunks.flag.tobytes() == ev_one_chunk.flag.tobytes()
    assert ev_small_chunks.adc.tobytes() == ev_one_chunk.adc.tobytes()
    assert ev_small_chunks.deadtime_s.tobytes() == ev_one_chunk.deadtime_s.tobytes()
    assert ev_small_chunks.temp_c.tobytes() == ev_one_chunk.temp_c.tobytes()
    assert ev_small_chunks.press_pa.tobytes() == ev_one_chunk.press_pa.tobytes()
    assert ev_small_chunks.start_utc == ev_one_chunk.start_utc
    assert ev_small_chunks.clock_drift_s == ev_one_chunk.clock_drift_s

    # dtypes must be preserved regardless of chunk size.
    assert ev_small_chunks.timestamp_s.dtype == np.float64
    assert ev_small_chunks.flag.dtype == np.int8
    assert ev_small_chunks.adc.dtype == np.int32
    assert ev_small_chunks.deadtime_s.dtype == np.float64
    assert ev_small_chunks.temp_c.dtype == np.float64
    assert ev_small_chunks.press_pa.dtype == np.float64
