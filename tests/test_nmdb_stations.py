from __future__ import annotations

import pytest

from nmdb_stations import (
    STATIONS,
    StationError,
    rank_by_rigidity,
    select_station,
    vertical_cutoff_rigidity,
)

SUNNYVALE = (37.3688, -122.0363)


def test_station_table_is_populated():
    assert len(STATIONS) >= 60
    codes = [s.code for s in STATIONS]
    assert "OULU" in codes
    assert "JUNG" in codes
    assert "UFSZ" in codes
    assert len(codes) == len(set(codes))


def test_known_station_values():
    by_code = {s.code: s for s in STATIONS}
    assert by_code["OULU"].rigidity_gv == pytest.approx(0.81)
    assert by_code["JUNG"].rigidity_gv == pytest.approx(4.49)
    assert by_code["JUNG"].altitude_m == 3570


def test_cutoff_rigidity_at_sunnyvale():
    """Stormer approximation should land near the accepted ~4 GV."""
    rc = vertical_cutoff_rigidity(*SUNNYVALE)
    assert 3.5 < rc < 4.8


def test_cutoff_rigidity_is_high_at_the_equator_and_low_at_the_pole():
    equatorial = vertical_cutoff_rigidity(0.0, 100.0)
    polar = vertical_cutoff_rigidity(89.0, 0.0)
    assert equatorial > 10.0
    assert polar < 1.0


def test_rank_by_rigidity_orders_by_absolute_difference():
    ranked = rank_by_rigidity(4.13)
    diffs = [abs(s.rigidity_gv - 4.13) for s in ranked]
    assert diffs == sorted(diffs)
    assert ranked[0].code == "UFSZ"


def test_select_station_picks_closest_rigidity():
    chosen, detector_rc, candidates = select_station(*SUNNYVALE)
    assert chosen.code == "UFSZ"
    assert 3.5 < detector_rc < 4.8
    assert len(candidates) == 5
    assert candidates[0] is chosen


def test_select_station_honours_override():
    chosen, _, _ = select_station(*SUNNYVALE, override_code="OULU")
    assert chosen.code == "OULU"


def test_override_is_case_insensitive():
    chosen, _, _ = select_station(*SUNNYVALE, override_code="oulu")
    assert chosen.code == "OULU"


def test_unknown_override_raises_with_helpful_message():
    with pytest.raises(StationError) as exc:
        select_station(*SUNNYVALE, override_code="ZZZZ")
    assert "ZZZZ" in str(exc.value)
