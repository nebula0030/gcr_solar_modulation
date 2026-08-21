from __future__ import annotations

import os

import numpy as np
import pytest

import process_data
from correction import CorrectionResult
from external_sources import ExternalSeries, FetchError
from nmdb_stations import Station, select_station
from process_data import build_parser, fetch_nmdb_with_fallback, format_summary, main
from rate import RateSeries

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
    assert produced == ["sample_13col.html"]
    assert not any("_overlay" in n or "_sidebyside" in n for n in produced)


def test_writes_single_combined_html_not_separate_files(tmp_path):
    path = os.path.join(FIXTURES, "sample_13col.txt")
    code = main([path, "--bin-length", "2", "--sources", "",
                 "--correction-method", "literature", "--beta-p", "-0.13",
                 "--output-dir", str(tmp_path),
                 "--cache-dir", str(tmp_path / "cache")])
    assert code == 0
    htmls = sorted(p.name for p in tmp_path.glob("*.html"))
    # Exactly one combined file; no _overlay/_sidebyside split.
    assert htmls == ["sample_13col.html"]
    assert not any("_overlay" in n or "_sidebyside" in n for n in htmls)
    content = (tmp_path / "sample_13col.html").read_text()
    assert 'id="tab-overlay"' in content and 'id="tab-side"' in content


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

    def spy_build_overlay(detectors, aligned, meta, master_utc, gaps=None,
                          events=None):
        captured_meta["meta"] = meta
        return real_build_overlay(detectors, aligned, meta, master_utc,
                                  gaps=gaps, events=events)

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
    assert produced == ["sample_13col.html"]
    assert not any("_overlay" in n or "_sidebyside" in n for n in produced)


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
    assert produced == ["sample_13col.html"]
    assert not any("_overlay" in n or "_sidebyside" in n for n in produced)


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


# ---------------------------------------------------------------------------
# Finding 1: mean raw rate must not print "nan" when a bin is dead
#
# rate_hz carries np.nan for dead bins (livetime <= 0, see rate.py). Every
# other reduction over rate_hz in the codebase guards NaN (mean_fractional_
# error masks with isfinite, plotting.build_overlay uses np.nanmean); the
# summary's "Mean raw rate" line used a plain .mean(), which goes nan the
# moment any single bin is dead. This exercises format_summary directly with
# a RateSeries that has one dead (NaN) bin among otherwise-good bins.
# ---------------------------------------------------------------------------


def _rate_series_with_one_dead_bin() -> RateSeries:
    n = 4
    bin_length_s = 60.0
    start = np.datetime64("2026-07-10T00:00:00", "ns")
    offsets = (np.arange(n) * bin_length_s * 1e9).astype("timedelta64[ns]")
    bin_start_utc = start + offsets
    bin_mid_utc = bin_start_utc + np.timedelta64(int(bin_length_s * 1e9 / 2), "ns")

    rate_hz = np.array([1.0, np.nan, 1.2, 0.9])
    rate_err_hz = np.array([0.1, np.nan, 0.11, 0.09])

    return RateSeries(
        bin_start_utc=bin_start_utc,
        bin_mid_utc=bin_mid_utc,
        counts=np.array([60, 0, 72, 54], dtype=np.int64),
        livetime_s=np.array([60.0, 0.0, 60.0, 60.0]),
        rate_hz=rate_hz,
        rate_err_hz=rate_err_hz,
        press_hpa=np.full(n, 1013.0),
        temp_c=np.full(n, 20.0),
        bin_length_s=bin_length_s,
    )


def _minimal_correction_result() -> CorrectionResult:
    return CorrectionResult(
        corrected_rate_hz=np.array([1.0, np.nan, 1.2, 0.9]),
        corrected_err_hz=np.array([0.1, np.nan, 0.11, 0.09]),
        beta_p=-0.0018,
        beta_t=0.0,
        beta_p_err=0.0002,
        beta_t_err=0.0,
        r_squared=0.5,
        p0_hpa=1013.0,
        t0_c=20.0,
        method="fit",
    )


# ---------------------------------------------------------------------------
# Task 7: multi-file CLI orchestration -- group by detector, stack on one page
# ---------------------------------------------------------------------------


def test_parser_accepts_multiple_files_and_start_time():
    args = build_parser().parse_args(
        ["a.txt", "b.txt", "--bin-length", "3600",
         "--start-time", "b.txt=2026-07-11T00:00:00Z"])
    assert args.input_files == ["a.txt", "b.txt"]
    assert args.start_time == ["b.txt=2026-07-11T00:00:00Z"]


def test_parser_accepts_end_time():
    args = build_parser().parse_args(
        ["a.txt", "--bin-length", "3600",
         "--end-time", "a.txt=2026-07-11T00:00:00Z"])
    assert args.end_time == ["a.txt=2026-07-11T00:00:00Z"]


def test_bad_end_time_exits_nonzero(capsys):
    path = os.path.join(FIXTURES, "sample_13col.txt")
    code = main([path, "--bin-length", "2", "--sources", "",
                 "--correction-method", "literature", "--beta-p", "-0.13",
                 "--end-time", "sample_13col.txt=not-a-date",
                 "--output-dir", "/tmp/endtime_bad"])
    assert code != 0
    assert "end-time" in capsys.readouterr().err.lower()


def test_end_time_truncates_and_still_writes_plots(tmp_path):
    path = os.path.join(FIXTURES, "sample_13col.txt")

    def run_and_count_bins(out_sub, extra_args):
        out = tmp_path / out_sub
        code = main([path, "--bin-length", "1", "--sources", "",
                     "--correction-method", "literature", "--beta-p", "-0.13",
                     "--export-csv",
                     "--output-dir", str(out),
                     "--cache-dir", str(tmp_path / "cache")] + extra_args)
        assert code == 0
        produced = sorted(p.name for p in out.glob("*.html"))
        assert produced == ["sample_13col.html"]
        assert not any("_overlay" in n or "_sidebyside" in n for n in produced)
        csv_path = next(out.glob("*_binned.csv"))
        rows = csv_path.read_text().strip().splitlines()
        return len(rows) - 1  # minus the header row

    full_bins = run_and_count_bins("full", [])
    truncated_bins = run_and_count_bins(
        "trunc", ["--end-time", "sample_13col.txt=2026-07-10T00:00:03Z"])
    # The flag must actually take effect: truncating the file's tail yields
    # strictly fewer bins than the untruncated run (this fails if --end-time
    # were silently ignored).
    assert 0 < truncated_bins < full_bins


def test_two_same_detector_files_overlap_exits_nonzero(capsys):
    path = os.path.join(FIXTURES, "sample_13col.txt")
    code = main([path, path, "--bin-length", "2", "--sources", "",
                 "--correction-method", "literature", "--beta-p", "-0.13",
                 "--output-dir", "/tmp/splice_overlap"])
    assert code != 0
    assert "overlap" in capsys.readouterr().err.lower()


def test_single_file_still_produces_both_plots(tmp_path):
    path = os.path.join(FIXTURES, "sample_13col.txt")
    code = main([path, "--bin-length", "2", "--sources", "",
                 "--correction-method", "literature", "--beta-p", "-0.13",
                 "--output-dir", str(tmp_path),
                 "--cache-dir", str(tmp_path / "cache")])
    assert code == 0
    produced = sorted(p.name for p in tmp_path.glob("*.html"))
    assert produced == ["sample_13col.html"]
    assert not any("_overlay" in n or "_sidebyside" in n for n in produced)


def test_two_different_detector_files_stack_on_one_page(tmp_path):
    """Two different detectors must run through the pipeline independently
    and land on ONE combined tabbed (overlay/side-by-side) page (not one per
    detector).
    """
    path_a = os.path.join(FIXTURES, "sample_13col.txt")
    path_b = os.path.join(FIXTURES, "det_b.txt")
    code = main([path_a, path_b, "--bin-length", "2", "--sources", "",
                 "--correction-method", "literature", "--beta-p", "-0.13",
                 "--output-dir", str(tmp_path),
                 "--cache-dir", str(tmp_path / "cache")])
    assert code == 0
    produced = sorted(p.name for p in tmp_path.glob("*.html"))
    assert len(produced) == 1
    assert not any("_overlay" in n or "_sidebyside" in n for n in produced)
    # run_name derives from the first file's stem, with a "+Nmore" suffix.
    assert produced[0].startswith("sample_13col_+1more")


# ---------------------------------------------------------------------------
# Cross-correlation line in the header
# ---------------------------------------------------------------------------


def test_header_shows_na_cross_correlation_when_nmdb_not_loaded(tmp_path):
    """With no external sources requested at all, the NMDB aligned series
    never exists, so the header must say so explicitly rather than silently
    omitting the line.
    """
    path = os.path.join(FIXTURES, "sample_13col.txt")
    code = main([
        path, "--bin-length", "2",
        "--correction-method", "literature", "--beta-p", "-0.13",
        "--sources", "",
        "--output-dir", str(tmp_path),
        "--cache-dir", str(tmp_path / "cache"),
    ])
    assert code == 0
    content = (tmp_path / "sample_13col.html").read_text()
    assert "Cross-correlation with NMDB: n/a (NMDB not loaded)" in content


def test_header_shows_cross_correlation_r_when_nmdb_loaded(tmp_path, monkeypatch):
    """With a real (stubbed, offline) NMDB series in play, the header must
    show a computed r rather than the n/a fallback.
    """

    def fake_fetch_nmdb(start, end, station_code, bin_length_s, cache):
        # A dense, varying synthetic series spanning the run window, so
        # every master bin gets a native (non-interpolated) sample and a
        # real Pearson r can be computed rather than "insufficient overlap".
        n = 200
        total_ns = (end - start).astype("timedelta64[ns]").astype(np.int64)
        step_ns = max(total_ns // (n - 1), 1)
        utc = start + (np.arange(n) * step_ns).astype("timedelta64[ns]")
        values = np.linspace(80.0, 120.0, n)
        return ExternalSeries(
            name="Neutron monitor ({0})".format(station_code),
            units="counts/s",
            source="NMDB",
            utc=utc,
            values=values,
        )

    monkeypatch.setattr(process_data, "fetch_nmdb", fake_fetch_nmdb)

    path = os.path.join(FIXTURES, "sample_13col.txt")
    code = main([
        path, "--bin-length", "0.5",
        "--correction-method", "literature", "--beta-p", "-0.13",
        "--sources", "nmdb",
        "--output-dir", str(tmp_path),
        "--cache-dir", str(tmp_path / "cache"),
    ])
    assert code == 0
    content = (tmp_path / "sample_13col.html").read_text()
    assert "Cross-correlation with" in content
    assert "r = " in content


def test_summary_mean_raw_rate_is_finite_when_a_bin_is_dead():
    rs = _rate_series_with_one_dead_bin()
    assert np.isnan(rs.rate_hz).any()  # sanity: the scenario actually has a dead bin

    station = Station("OULU", "Oulu", 0.81, 15)
    args = build_parser().parse_args(["f.txt", "--bin-length", "60"])

    summary = format_summary(
        input_file="f.txt",
        rs=rs,
        correction=_minimal_correction_result(),
        station=station,
        detector_rigidity=5.0,
        candidates=[station],
        aligned=[],
        warnings=[],
        notes=[],
        args=args,
        clock_drift_s=0.0,
    )

    mean_line = next(
        line for line in summary.splitlines() if "Mean raw rate:" in line
    )
    assert "nan" not in mean_line.lower()
    # np.nanmean([1.0, 1.2, 0.9]) == 1.0333...
    assert "1.0333" in mean_line


# --- --events CLI wiring (Task 6) -------------------------------------------


def test_no_events_flag_makes_no_donki_call(tmp_path, monkeypatch):
    """--no-events skips the fetch entirely (no network, no checkbox)."""
    import external_sources

    def boom(*a, **k):
        raise AssertionError("fetch_solar_events must not run with --no-events")

    monkeypatch.setattr(external_sources, "fetch_solar_events", boom)

    path = os.path.join(FIXTURES, "sample_13col.txt")
    code = main([
        path, "--bin-length", "2",
        "--correction-method", "literature", "--beta-p", "-0.13",
        "--sources", "",
        "--no-events",
        "--output-dir", str(tmp_path),
        "--cache-dir", str(tmp_path / "cache"),
    ])
    assert code == 0
    content = (tmp_path / "sample_13col.html").read_text()
    assert 'id="events-toggle"' not in content


def test_events_fetched_by_default(tmp_path, monkeypatch):
    """Events are on by default: a run with no events flag fetches DONKI and
    shows the toggle when events are returned."""
    import external_sources
    import numpy as np

    called = {"n": 0}

    def fake(*a, **k):
        called["n"] += 1
        return [external_sources.SolarEvent(
            "flare", np.datetime64("2026-07-10T00:00:02"), "X1.0")]

    monkeypatch.setattr(external_sources, "fetch_solar_events", fake)

    path = os.path.join(FIXTURES, "sample_13col.txt")
    code = main([
        path, "--bin-length", "2",
        "--correction-method", "literature", "--beta-p", "-0.13",
        "--sources", "",
        "--output-dir", str(tmp_path),
        "--cache-dir", str(tmp_path / "cache"),
    ])
    assert code == 0
    assert called["n"] == 1  # fetched without any events flag
    content = (tmp_path / "sample_13col.html").read_text()
    assert 'id="events-toggle"' in content


def test_events_flag_on_fetches_and_passes(tmp_path, monkeypatch):
    import external_sources
    import numpy as np

    monkeypatch.setattr(
        external_sources, "fetch_solar_events",
        lambda *a, **k: [external_sources.SolarEvent(
            "flare", np.datetime64("2026-07-10T00:00:02"), "X1.0")])

    path = os.path.join(FIXTURES, "sample_13col.txt")
    code = main([
        path, "--bin-length", "2",
        "--correction-method", "literature", "--beta-p", "-0.13",
        "--sources", "",
        "--events",
        "--output-dir", str(tmp_path),
        "--cache-dir", str(tmp_path / "cache"),
    ])
    assert code == 0
    content = (tmp_path / "sample_13col.html").read_text()
    assert 'id="events-toggle"' in content


def test_events_flag_fetch_failure_is_fail_soft(tmp_path, monkeypatch):
    import external_sources

    def raise_fetch_error(*a, **k):
        raise external_sources.FetchError("offline")

    monkeypatch.setattr(external_sources, "fetch_solar_events", raise_fetch_error)

    path = os.path.join(FIXTURES, "sample_13col.txt")
    code = main([
        path, "--bin-length", "2",
        "--correction-method", "literature", "--beta-p", "-0.13",
        "--sources", "",
        "--events",
        "--output-dir", str(tmp_path),
        "--cache-dir", str(tmp_path / "cache"),
    ])
    assert code == 0
    content = (tmp_path / "sample_13col.html").read_text()
    assert "Unavailable" in content
    assert "solar events" in content
    assert 'id="events-toggle"' not in content


def test_event_lead_days_default_and_zero():
    p = build_parser()
    assert p.parse_args(["f.txt", "--bin-length", "3600"]).event_lead_days == 5
    assert p.parse_args(["f.txt", "--bin-length", "3600",
                         "--event-lead-days", "0"]).event_lead_days == 0


def test_event_fetch_window_is_widened_by_lead_days(tmp_path, monkeypatch):
    import external_sources
    seen = {}

    def capture(start_utc, end_utc, *a, **k):
        seen["start"] = np.datetime64(start_utc)
        return []

    monkeypatch.setattr(external_sources, "fetch_solar_events", capture)

    path = os.path.join(FIXTURES, "sample_13col.txt")
    code = main([
        path, "--bin-length", "2",
        "--correction-method", "literature", "--beta-p", "-0.13",
        "--sources", "", "--event-lead-days", "3",
        "--output-dir", str(tmp_path),
        "--cache-dir", str(tmp_path / "cache"),
    ])
    assert code == 0
    # sample_13col.txt starts 2026-07-10T00:00:00; 3-day lead-in -> 2026-07-07.
    assert seen["start"] == np.datetime64("2026-07-07T00:00:00")
