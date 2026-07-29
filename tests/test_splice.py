from __future__ import annotations

import os

import numpy as np
import pytest

from splice import (
    SpliceError,
    group_by_detector,
    parse_end_overrides,
    parse_start_overrides,
    read_events_multi,
)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def fx(name):
    return os.path.join(FIXTURES, name)


def test_group_by_detector_does_not_full_parse(monkeypatch):
    """Grouping uses the light first-row read, not a full parse, so a file is
    not parsed twice (once to group, once to splice)."""
    import splice as splice_mod

    def boom(*args, **kwargs):
        raise AssertionError("group_by_detector must not call read_events")

    monkeypatch.setattr(splice_mod, "read_events", boom)
    groups = group_by_detector([fx("sample_13col.txt"), fx("det_b.txt")])
    assert list(groups.keys()) == ["TestDet", "DetB"]


def test_parse_start_overrides_basename_and_datetime():
    m = parse_start_overrides(["run2.txt=2026-07-11T00:00:00Z"])
    assert "run2.txt" in m
    assert m["run2.txt"] == np.datetime64("2026-07-11T00:00:00")


def test_parse_start_overrides_strips_directory_to_basename():
    m = parse_start_overrides(["/data/runs/run2.txt=2026-07-11T00:00:00Z"])
    assert "run2.txt" in m


def test_parse_start_overrides_rejects_malformed_item():
    with pytest.raises(SpliceError):
        parse_start_overrides(["no-equals-sign"])


def test_parse_start_overrides_rejects_bad_datetime():
    with pytest.raises(SpliceError):
        parse_start_overrides(["run.txt=not-a-date"])


def test_group_by_detector_splits_two_detectors_preserving_order():
    groups = group_by_detector([fx("sample_13col.txt"), fx("det_b.txt")])
    assert list(groups.keys()) == ["TestDet", "DetB"]
    assert groups["TestDet"] == [fx("sample_13col.txt")]
    assert groups["DetB"] == [fx("det_b.txt")]


def test_group_by_detector_groups_same_detector():
    groups = group_by_detector([fx("sample_13col.txt"), fx("sample_13col.txt")])
    assert list(groups.keys()) == ["TestDet"]
    assert len(groups["TestDet"]) == 2


def test_read_events_multi_single_file_matches_read_events():
    from cosmicwatch_io import read_events
    combined = read_events_multi([fx("sample_13col.txt")], {})
    single = read_events(fx("sample_13col.txt"))
    assert len(combined.timestamp_s) == len(single.timestamp_s)
    assert combined.detector_name == single.detector_name
    assert combined.coverage_s is not None
    assert len(combined.coverage_s) == 1
    # Pin the zero-basing formula: the single-file combined timeline is just
    # the original timestamps shifted so the first event is at t=0.
    assert np.allclose(
        combined.timestamp_s, single.timestamp_s - single.timestamp_s[0]
    )


def test_read_events_multi_two_files_with_a_gap(tmp_path):
    # Copy the fixture under a second basename and start it a day later, so the
    # same detector is spliced with a real gap between the two spans.
    import shutil
    src = fx("sample_13col.txt")
    second = tmp_path / "sample_13col_b.txt"
    shutil.copy(src, str(second))
    overrides = parse_start_overrides(
        ["sample_13col_b.txt=2026-07-11T00:00:00Z"])
    combined = read_events_multi([src, str(second)], overrides)
    # Two coverage intervals, and the second starts ~1 day (86400 s) after the
    # first, i.e. a large gap.
    assert len(combined.coverage_s) == 2
    gap_start = combined.coverage_s[0][1]
    gap_end = combined.coverage_s[1][0]
    assert gap_end - gap_start > 80000.0  # ~1 day minus the first file's span


def test_read_events_multi_same_detector_overlap_raises():
    # The same file twice overlaps itself in absolute time -> SpliceError.
    with pytest.raises(SpliceError) as exc:
        read_events_multi([fx("sample_13col.txt"), fx("sample_13col.txt")], {})
    assert "overlap" in str(exc.value).lower()


def test_read_events_multi_deadtime_is_globally_monotonic():
    combined = read_events_multi([fx("sample_13col.txt")], {})
    assert np.all(np.diff(combined.deadtime_s) >= 0)


def test_read_events_multi_deadtime_cumulates_across_files(tmp_path):
    # Splice two copies of the same fixture, with the second file starting
    # well after the first ends, so read_events_multi actually has to walk
    # two files and offset the second file's deadtime by the running total
    # from the first. A single-file test can't exercise this: with only one
    # iteration, deadtime_offset is always 0.
    import shutil
    from cosmicwatch_io import read_events

    src = fx("sample_13col.txt")
    second = tmp_path / "sample_13col_b.txt"
    shutil.copy(src, str(second))
    overrides = parse_start_overrides(
        ["sample_13col_b.txt=2026-07-11T00:00:00Z"])
    combined = read_events_multi([src, str(second)], overrides)

    # 1) Globally monotonic non-decreasing, as before.
    assert np.all(np.diff(combined.deadtime_s) >= 0)

    # 2) The cross-file cumulation actually happened: the second file's
    # events must be offset by the first file's final deadtime. Both files
    # are byte-identical copies of the same fixture, so the second file's
    # own (un-offset) deadtime values are known from read_events on the
    # source directly.
    first = read_events(src)
    n1 = len(first.timestamp_s)
    d1_last = float(first.deadtime_s[-1])
    second_first_deadtime = float(first.deadtime_s[0])

    assert len(combined.deadtime_s) == 2 * n1
    # The first event contributed by the second file sits right after the
    # first file's n1 events in the concatenated array.
    assert combined.deadtime_s[n1] == pytest.approx(
        d1_last + second_first_deadtime
    )
    # Equivalently, and more simply: it must exceed the first file's final
    # deadtime. If `+ deadtime_offset` were removed from read_events_multi,
    # this would be exactly d1_last's own small increment (i.e. it would
    # NOT include d1_last), so this assertion would fail.
    assert combined.deadtime_s[n1] > d1_last


def test_parse_end_overrides_basename_and_datetime():
    m = parse_end_overrides(["run.txt=2026-07-11T00:00:00Z"])
    assert m["run.txt"] == np.datetime64("2026-07-11T00:00:00")


def test_parse_end_overrides_rejects_malformed_item():
    with pytest.raises(SpliceError):
        parse_end_overrides(["no-equals-sign"])


def test_parse_end_overrides_rejects_bad_datetime():
    with pytest.raises(SpliceError):
        parse_end_overrides(["run.txt=not-a-date"])


def test_end_time_truncates_events_after_cutoff():
    # sample_13col.txt: 6 events at t=0..5 s from 2026-07-10T00:00:00 (its own
    # clock). Cut off at +2 s -> keep events at absolute 00:00:00..00:00:02.
    ends = parse_end_overrides(["sample_13col.txt=2026-07-10T00:00:02Z"])
    ev = read_events_multi([fx("sample_13col.txt")], {}, ends)
    # events at 0,1,2 s kept (inclusive); 3,4,5 dropped.
    assert len(ev.timestamp_s) == 3
    assert ev.coverage_s[0][1] == pytest.approx(2.0)


def test_end_time_boundary_event_is_kept():
    ends = parse_end_overrides(["sample_13col.txt=2026-07-10T00:00:03Z"])
    ev = read_events_multi([fx("sample_13col.txt")], {}, ends)
    assert len(ev.timestamp_s) == 4  # 0,1,2,3 s


def test_end_before_start_raises():
    ends = parse_end_overrides(["sample_13col.txt=2026-07-09T00:00:00Z"])
    with pytest.raises(SpliceError) as exc:
        read_events_multi([fx("sample_13col.txt")], {}, ends)
    assert "sample_13col.txt" in str(exc.value)


def test_end_time_can_resolve_an_overlap(tmp_path):
    # First file (own clock) spans 00:00:00..00:00:05. A second copy started at
    # +3s spans 00:00:03..00:00:08 -> the two overlap.
    import shutil
    src = fx("sample_13col.txt")
    second = tmp_path / "sample_13col_b.txt"
    shutil.copy(src, str(second))
    starts = parse_start_overrides(["sample_13col_b.txt=2026-07-10T00:00:03Z"])

    # Without any end-time, the overlap is a SpliceError.
    with pytest.raises(SpliceError):
        read_events_multi([src, str(second)], starts)

    # Truncating the first file to end at +2s removes the overlap.
    ends = parse_end_overrides(["sample_13col.txt=2026-07-10T00:00:02Z"])
    ev = read_events_multi([src, str(second)], starts, ends)  # must not raise
    assert len(ev.coverage_s) == 2
    assert ev.coverage_s[0][1] == pytest.approx(2.0)  # first file truncated


def test_no_end_overrides_matches_two_arg_call():
    a = read_events_multi([fx("sample_13col.txt")], {})
    b = read_events_multi([fx("sample_13col.txt")], {}, {})
    assert len(a.timestamp_s) == len(b.timestamp_s)
    assert a.coverage_s == b.coverage_s
