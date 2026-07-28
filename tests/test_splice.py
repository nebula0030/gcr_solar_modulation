from __future__ import annotations

import os

import numpy as np
import pytest

from splice import SpliceError, group_by_detector, parse_start_overrides

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def fx(name):
    return os.path.join(FIXTURES, name)


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
