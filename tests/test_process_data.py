from __future__ import annotations

import os

import numpy as np
import pytest

import process_data
from external_sources import ExternalSeries, FetchError
from nmdb_stations import select_station
from process_data import build_parser, fetch_nmdb_with_fallback, main

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
SUNNYVALE = (37.3688, -122.0363)


def test_parser_defaults_to_sunnyvale():
    args = build_parser().parse_args(["f.txt", "--bin-length", "3600"])
    assert args.lat == pytest.approx(37.3688)
    assert args.lon == pytest.approx(-122.0363)


def test_parser_defaults_to_fit_and_onboard():
    args = build_parser().parse_args(["f.txt", "--bin-length", "3600"])
    assert args.correction_method == "fit"
    assert args.met_source == "onboard"


def test_parser_accepts_all_documented_flags():
    args = build_parser().parse_args([
        "f.txt", "--bin-length", "600",
        "--lat", "10.0", "--lon", "20.0",
        "--met-source", "external",
        "--correction-method", "literature",
        "--beta-p", "-0.13", "--beta-t", "-0.05",
        "--beta-p-source", "Some paper 2020",
        "--nmdb-station", "OULU",
        "--sources", "nmdb,kp",
        "--cache-dir", "/tmp/c",
        "--refresh-cache",
        "--output-dir", "/tmp/o",
        "--export-csv",
    ])
    assert args.beta_p == pytest.approx(-0.13)
    assert args.nmdb_station == "OULU"
    assert args.sources == "nmdb,kp"
    assert args.refresh_cache is True
    assert args.export_csv is True


def test_literature_without_beta_p_exits_nonzero(capsys):
    path = os.path.join(FIXTURES, "sample_13col.txt")
    code = main([
        path, "--bin-length", "2",
        "--correction-method", "literature",
        "--sources", "",
    ])
    assert code != 0
    assert "--beta-p" in capsys.readouterr().err


def test_missing_input_file_exits_nonzero(capsys):
    code = main(["/nonexistent/file.txt", "--bin-length", "3600",
                 "--sources", ""])
    assert code != 0


def test_bin_longer_than_run_exits_nonzero(capsys):
    path = os.path.join(FIXTURES, "sample_13col.txt")
    code = main([path, "--bin-length", "100000", "--sources", ""])
    assert code != 0
    assert "bin length" in capsys.readouterr().err.lower()


def test_unknown_source_returns_nonzero_int_not_systemexit(tmp_path, capsys):
    """Finding 2: main(argv) -> int must hold even for a bad --sources.

    gather_external used to ``raise SystemExit`` directly, which breaks the
    contract for programmatic callers (SystemExit propagates rather than
    returning). It must come back as an ordinary int exit code instead.
    """
    path = os.path.join(FIXTURES, "sample_13col.txt")
    code = main([
        path, "--bin-length", "2",
        "--correction-method", "literature", "--beta-p", "-0.13",
        "--sources", "bogus",
        "--cache-dir", str(tmp_path / "cache"),
    ])
    assert isinstance(code, int)
    assert code != 0
    assert "unknown source" in capsys.readouterr().err.lower()


def test_offline_run_still_produces_both_plots(tmp_path):
    """With no external sources requested, the tool must still work."""
    path = os.path.join(FIXTURES, "sample_13col.txt")
    code = main([
        path, "--bin-length", "2",
        "--correction-method", "literature", "--beta-p", "-0.13",
        "--sources", "",
        "--output-dir", str(tmp_path),
        "--cache-dir", str(tmp_path / "cache"),
    ])
    assert code == 0
    produced = sorted(p.name for p in tmp_path.glob("*.html"))
    assert any(n.endswith("_overlay.html") for n in produced)
    assert any(n.endswith("_sidebyside.html") for n in produced)


def test_csv_export_writes_a_file(tmp_path):
    path = os.path.join(FIXTURES, "sample_13col.txt")
    code = main([
        path, "--bin-length", "2",
        "--correction-method", "literature", "--beta-p", "-0.13",
        "--sources", "", "--export-csv",
        "--output-dir", str(tmp_path),
        "--cache-dir", str(tmp_path / "cache"),
    ])
    assert code == 0
    assert list(tmp_path.glob("*.csv"))


def test_summary_states_the_utc_assumption(tmp_path, capsys):
    path = os.path.join(FIXTURES, "sample_13col.txt")
    main([
        path, "--bin-length", "2",
        "--correction-method", "literature", "--beta-p", "-0.13",
        "--sources", "",
        "--output-dir", str(tmp_path),
        "--cache-dir", str(tmp_path / "cache"),
    ])
    assert "UTC" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# NMDB station fallback
#
# Task 6 found that for real run windows the rigidity-best station for
# Sunnyvale (UFSZ) can return an HTTP 200 with no usable rows, while a
# lower-ranked station (e.g. OULU) does have data. The design spec promises
# that the tool "validates that the chosen station has data for the run
# window and falls back to the next-best match with a warning if not." These
# tests exercise that behaviour with a stubbed fetcher -- no network.
# ---------------------------------------------------------------------------


def _series(code: str) -> ExternalSeries:
    return ExternalSeries(
        name="Neutron monitor ({0})".format(code),
        units="counts/s",
        source="NMDB",
        utc=np.array(
            [np.datetime64("2026-07-10T00:00:00", "ns"),
             np.datetime64("2026-07-10T00:00:10", "ns")]
        ),
        values=np.array([99.0, 98.5]),
    )


def test_fallback_uses_next_best_station_when_first_has_no_data():
    station, rigidity, _ = select_station(*SUNNYVALE)
    assert station.code == "UFSZ"  # rigidity-best for Sunnyvale

    calls = []

    def fetch_fn(code):
        calls.append(code)
        if code == "UFSZ":
            raise FetchError("no usable data rows in the NMDB response")
        return _series(code)

    series, used, warnings, notes = fetch_nmdb_with_fallback(
        station, rigidity, None, fetch_fn
    )

    assert series is not None
    assert used.code != "UFSZ"
    assert calls[0] == "UFSZ"
    assert len(calls) == 2  # first station, then the very next-best match
    # The successful fallback is a NOTE, not a warning/failure: the run
    # worked, just not with the first-choice station.
    assert any("falling back from UFSZ" in n for n in notes)
    assert any(used.code in n for n in notes)
    # A successful fallback's whole trail (including the first-choice
    # station having had no data) is informational, not a warning.
    assert warnings == []


def test_fallback_reports_unavailable_when_every_candidate_fails():
    station, rigidity, _ = select_station(*SUNNYVALE)

    def fetch_fn(code):
        raise FetchError("no data for {0}".format(code))

    series, used, warnings, notes = fetch_nmdb_with_fallback(
        station, rigidity, None, fetch_fn
    )

    assert series is None
    assert any("unavailable" in w for w in warnings)
    assert notes == []


def test_explicit_override_does_not_fall_back():
    station, rigidity, _ = select_station(*SUNNYVALE, override_code="UFSZ")

    calls = []

    def fetch_fn(code):
        calls.append(code)
        raise FetchError("no usable data rows in the NMDB response")

    series, used, warnings, notes = fetch_nmdb_with_fallback(
        station, rigidity, "UFSZ", fetch_fn
    )

    assert series is None
    assert used.code == "UFSZ"
    assert calls == ["UFSZ"]  # no fallback attempted
    assert any(
        "not falling back" in w or "explicitly requested" in w
        for w in warnings
    )
    assert notes == []


def test_fallback_reaches_a_deeply_ranked_station_when_all_closer_ones_fail():
    """The fallback search is unbounded -- for the real dataset the only
    station with data (OULU) ranks 29th by rigidity for a Sunnyvale
    detector, so a capped search would report NMDB as unavailable on every
    real run. This stubs every candidate ranked ahead of OULU as failing and
    asserts the loop still reaches OULU rather than giving up early.
    """
    station, rigidity, _ = select_station(*SUNNYVALE)
    ranked = process_data.rank_by_rigidity(rigidity)
    oulu_rank = next(i for i, s in enumerate(ranked) if s.code == "OULU")
    assert oulu_rank >= 20  # sanity check: genuinely deep in the ranking

    calls = []

    def fetch_fn(code):
        calls.append(code)
        if code == "OULU":
            return _series("OULU")
        raise FetchError("no data for {0}".format(code))

    series, used, warnings, notes = fetch_nmdb_with_fallback(
        station, rigidity, None, fetch_fn
    )

    assert series is not None
    assert used.code == "OULU"
    assert calls[-1] == "OULU"
    assert len(calls) == oulu_rank + 1  # every closer candidate was tried
    # A successful (if deep) fallback is still a NOTE, not a warning.
    assert any("OULU" in n for n in notes)
    assert warnings == []


def test_main_falls_back_and_reports_the_station_actually_used(
    tmp_path, capsys, monkeypatch
):
    """Mirrors the live Task 6 finding: UFSZ has no data, a next-best
    rigidity match does. The stub below succeeds on ZUGS -- the very
    next-best match after UFSZ for a Sunnyvale detector (the live OULU case
    is much further down the rigidity ranking and is covered separately by
    the deep-fallback test).

    Finding 1: a successful fallback must be reported as a NOTE, not a
    FAIL/Unavailable -- both in the printed summary and in the plot footer.
    """

    def fake_fetch_nmdb(start, end, station_code, bin_length_s, cache):
        if station_code == "ZUGS":
            return _series("ZUGS")
        raise FetchError("no usable data rows in the NMDB response")

    monkeypatch.setattr(process_data, "fetch_nmdb", fake_fetch_nmdb)

    captured_meta = {}
    real_build_overlay = process_data.build_overlay

    def spy_build_overlay(rs, correction, aligned, meta):
        captured_meta["meta"] = meta
        return real_build_overlay(rs, correction, aligned, meta)

    monkeypatch.setattr(process_data, "build_overlay", spy_build_overlay)

    path = os.path.join(FIXTURES, "sample_13col.txt")
    code = main([
        path, "--bin-length", "2",
        "--correction-method", "literature", "--beta-p", "-0.13",
        "--sources", "nmdb",
        "--output-dir", str(tmp_path),
        "--cache-dir", str(tmp_path / "cache"),
    ])
    out = capsys.readouterr().out

    assert code == 0
    assert "falling back from UFSZ to ZUGS" in out
    assert "Selected:       ZUGS" in out

    # The fallback succeeded -- it must not be rendered as a failure.
    fallback_lines = [
        line for line in out.splitlines() if "falling back from UFSZ" in line
    ]
    assert fallback_lines
    assert all("FAIL" not in line for line in fallback_lines)
    assert any(line.strip().startswith("NOTE") for line in fallback_lines)
    assert "OK   Neutron monitor (ZUGS)" in out

    footer = "<br>".join(captured_meta["meta"].footer_notes)
    assert "Unavailable" not in footer
    assert "falling back from UFSZ to ZUGS" in footer

    produced = sorted(p.name for p in tmp_path.glob("*.html"))
    assert any(n.endswith("_overlay.html") for n in produced)
    assert any(n.endswith("_sidebyside.html") for n in produced)


def test_main_continues_offline_when_all_nmdb_candidates_fail(
    tmp_path, capsys, monkeypatch
):
    """A total NMDB outage must still let the run complete offline: every
    rigidity-ranked candidate is tried (there is no cap), and once they are
    all exhausted the run continues with NMDB reported unavailable.
    """
    calls = []

    def fake_fetch_nmdb(start, end, station_code, bin_length_s, cache):
        calls.append(station_code)
        raise FetchError("no usable data rows in the NMDB response")

    monkeypatch.setattr(process_data, "fetch_nmdb", fake_fetch_nmdb)

    path = os.path.join(FIXTURES, "sample_13col.txt")
    code = main([
        path, "--bin-length", "2",
        "--correction-method", "literature", "--beta-p", "-0.13",
        "--sources", "nmdb",
        "--output-dir", str(tmp_path),
        "--cache-dir", str(tmp_path / "cache"),
    ])
    out = capsys.readouterr().out

    assert code == 0
    assert "NMDB comparison unavailable" in out
    # No cap: every station in the ranking was tried before giving up.
    all_codes = {s.code for s in process_data.rank_by_rigidity(0.0)}
    assert set(calls) == all_codes
    produced = sorted(p.name for p in tmp_path.glob("*.html"))
    assert any(n.endswith("_overlay.html") for n in produced)
    assert any(n.endswith("_sidebyside.html") for n in produced)


def test_main_explicit_override_with_no_data_is_reported_not_replaced(
    tmp_path, capsys, monkeypatch
):
    calls = []

    def fake_fetch_nmdb(start, end, station_code, bin_length_s, cache):
        calls.append(station_code)
        raise FetchError("no usable data rows in the NMDB response")

    monkeypatch.setattr(process_data, "fetch_nmdb", fake_fetch_nmdb)

    path = os.path.join(FIXTURES, "sample_13col.txt")
    code = main([
        path, "--bin-length", "2",
        "--correction-method", "literature", "--beta-p", "-0.13",
        "--sources", "nmdb",
        "--nmdb-station", "UFSZ",
        "--output-dir", str(tmp_path),
        "--cache-dir", str(tmp_path / "cache"),
    ])
    out = capsys.readouterr().out

    assert code == 0
    assert calls == ["UFSZ"]  # never tried a fallback station
    assert "Selected:       UFSZ" in out
    assert "not falling back" in out or "explicitly requested" in out
