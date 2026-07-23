#!/usr/bin/env python3
"""Process CosmicWatch muon detector data and compare it to solar activity.

Pipeline: parse -> bin coincident events -> correct for pressure and
temperature -> fetch external solar-activity data -> align -> plot.

Data problems are fatal. Network problems are not: any external source that
fails is reported as a warning and the run continues with whatever was
retrieved, so the tool still works offline.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Callable, List, Optional, Tuple

import numpy as np

from align import AlignedSeries, align_to_bins
from correction import CorrectionError, CorrectionResult, correct
from cosmicwatch_io import DataFormatError, read_events
from external_sources import (
    NMDB_ACKNOWLEDGEMENT,
    Cache,
    ExternalSeries,
    FetchError,
    fetch_goes_xray,
    fetch_kp,
    fetch_nmdb,
    fetch_sunspot,
)
from nmdb_stations import Station, StationError, rank_by_rigidity, select_station
from plotting import PlotMetadata, build_overlay, build_side_by_side, write_html
from rate import RateError, RateSeries, compute_rate

ALL_SOURCES = ("nmdb", "goes", "kp", "sunspot")

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


class SourcesError(ValueError):
    """Raised when --sources names a source that isn't recognized."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="process_data.py",
        description=(
            "Estimate a pressure/temperature-corrected cosmic-ray muon rate "
            "from CosmicWatch v3X data and compare it against public "
            "solar-activity databases."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_file", help="CosmicWatch v3X 13-column data file")
    parser.add_argument(
        "--bin-length", type=float, required=True, metavar="SECONDS",
        help="rate bin width in seconds",
    )

    location = parser.add_argument_group("detector location")
    location.add_argument("--lat", type=float, default=37.3688,
                          help="latitude (default: %(default)s, Sunnyvale CA)")
    location.add_argument("--lon", type=float, default=-122.0363,
                          help="longitude (default: %(default)s)")

    met = parser.add_argument_group("meteorological data")
    met.add_argument("--met-source", choices=("onboard", "external"),
                     default="onboard",
                     help="pressure/temperature source (default: %(default)s)")

    corr = parser.add_argument_group("correction")
    corr.add_argument("--correction-method", choices=("fit", "literature"),
                      default="fit",
                      help="coefficient source (default: %(default)s)")
    corr.add_argument("--beta-p", type=float, default=None, metavar="PCT_PER_HPA",
                      help="barometric coefficient in %%/hPa; required for "
                           "--correction-method literature")
    corr.add_argument("--beta-t", type=float, default=None, metavar="PCT_PER_C",
                      help="temperature coefficient in %%/degC (default: 0)")
    corr.add_argument("--beta-p-source", default=None, metavar="TEXT",
                      help="provenance for --beta-p, echoed in the output")

    station = parser.add_argument_group("neutron monitor station")
    station.add_argument("--nmdb-station", default=None, metavar="CODE",
                         help="override automatic rigidity-based selection")

    ext = parser.add_argument_group("external sources")
    ext.add_argument("--sources", default=",".join(ALL_SOURCES),
                     help="comma-separated subset of %s (default: all)"
                          % ",".join(ALL_SOURCES))

    cache_group = parser.add_argument_group("cache")
    cache_group.add_argument("--cache-dir", default="cache",
                             help="cache directory (default: %(default)s)")
    cache_group.add_argument("--refresh-cache", action="store_true",
                             help="force re-download of external data")

    out = parser.add_argument_group("output")
    out.add_argument("--output-dir", default=None,
                     help="where to write plots (default: alongside the input)")
    out.add_argument("--export-csv", action="store_true",
                     help="also write the aligned binned table as CSV")

    return parser


def fetch_external_meteorology(
    rs: RateSeries, lat: float, lon: float, cache: Cache
) -> Tuple[np.ndarray, np.ndarray]:
    """Surface pressure (hPa) and 2 m temperature (C) from Open-Meteo ERA5.

    Verified: the ERA5 archive endpoint serves surface variables only, which
    is why the atmospheric effective-temperature treatment is deferred.
    """
    from external_sources import http_get

    start = rs.bin_start_utc[0].astype("datetime64[D]")
    end = rs.bin_start_utc[-1].astype("datetime64[D]")
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": str(start), "end_date": str(end),
        "hourly": "surface_pressure,temperature_2m",
        "timezone": "UTC",
    }
    key = "openmeteo|{0}|{1}|{2}|{3}".format(lat, lon, start, end)
    payload = cache.get_or_fetch(
        key, lambda: http_get(OPEN_METEO_ARCHIVE_URL, params)
    )

    import json

    try:
        data = json.loads(payload.decode("utf-8", errors="replace"))
        hourly = data["hourly"]
        times = np.asarray(
            [np.datetime64(t, "ns") for t in hourly["time"]],
            dtype="datetime64[ns]",
        )
        pressure = np.asarray(hourly["surface_pressure"], dtype=np.float64)
        temperature = np.asarray(hourly["temperature_2m"], dtype=np.float64)
    except (ValueError, KeyError, TypeError) as exc:
        raise FetchError("Open-Meteo response was unusable: {0}".format(exc))

    press_series = ExternalSeries("Surface pressure", "hPa", "Open-Meteo",
                                  times, pressure)
    temp_series = ExternalSeries("Air temperature", "C", "Open-Meteo",
                                 times, temperature)
    return (
        align_to_bins(press_series, rs).values,
        align_to_bins(temp_series, rs).values,
    )


def fetch_nmdb_with_fallback(
    station: Station,
    detector_rigidity_gv: float,
    override_code: Optional[str],
    fetch_fn: Callable[[str], ExternalSeries],
) -> Tuple[Optional[ExternalSeries], Station, List[str], List[str]]:
    """Fetch NMDB data for ``station``, falling back through the next-best
    rigidity-ranked stations if it has no data for the run window.

    A live finding from Task 6: for some run windows the rigidity-best
    station (e.g. UFSZ for a Sunnyvale detector) returns an HTTP 200 with no
    usable rows, while the next-best candidate (e.g. OULU) has data. NMDB
    "no data" is a 200 that DOES get cached, but the cache key includes the
    station code, so trying the next station is a genuine new lookup rather
    than a repeat of the same cached miss.

    The fallback loop tries every rigidity-ranked candidate station in turn
    (skipping the originally-chosen one, since it was already tried) until
    one returns data or the ranking is exhausted. There is no cap: for the
    real dataset the only station with data for a given run can rank far
    down the list (e.g. OULU at rank ~29), so bounding the search would
    routinely report NMDB as unavailable on real runs. An explicit
    ``--nmdb-station`` override is unaffected -- it tries only that one
    station, with no iteration.

    ``fetch_fn`` takes a station code and returns an ``ExternalSeries``,
    raising ``FetchError`` when that station has no usable data. It is
    injected so this loop can be exercised in tests without any network
    access (see ``tests/test_process_data.py``).

    If ``override_code`` is not ``None`` the caller explicitly asked for one
    station via ``--nmdb-station``; in that case a failure is reported as
    that station being unavailable and no fallback is attempted, since
    substituting a different station would silently ignore the user's
    request.

    Returns ``(series_or_None, station_used, warnings, notes)``.
    ``station_used`` is the station whose data (if any) is in
    ``series_or_None``, so callers can make the summary and plot metadata
    reflect what was actually used rather than what was first chosen.

    Whether a given run's fallback trail is reported as ``warnings`` or as
    ``notes`` is decided by the *outcome*, not by each individual step: if
    the fallback ultimately finds data, the whole trail (the first-choice
    station having no data, any stations tried and skipped in between, and
    the final station that worked) is purely informational -- the run
    succeeded, it just wasn't the first-choice station, and for the real
    dataset that happens on every run, so none of it should read as a
    failure. Only when every candidate is exhausted without finding data
    does that same trail become ``warnings``, because at that point NMDB
    genuinely has nothing to offer for this run. The explicit-override path
    is simpler: a failure there is unconditionally a warning, since no
    fallback is attempted and the requested station is definitely
    unavailable.
    """
    trail: List[str] = []
    try:
        series = fetch_fn(station.code)
        return series, station, [], []
    except FetchError as exc:
        if override_code is not None:
            return None, station, [
                "nmdb: station {0} has no data for this run window ({1}); "
                "not falling back because it was explicitly requested via "
                "--nmdb-station".format(station.code, exc)
            ], []
        trail.append(
            "nmdb: station {0} has no data for this run window ({1}); "
            "trying next-best rigidity match".format(station.code, exc)
        )

    for candidate in rank_by_rigidity(detector_rigidity_gv):
        if candidate.code == station.code:
            continue
        try:
            series = fetch_fn(candidate.code)
        except FetchError as exc:
            trail.append(
                "nmdb: station {0} also has no data ({1})".format(
                    candidate.code, exc
                )
            )
            continue
        trail.append(
            "nmdb: falling back from {0} to {1} ({2}), which has data for "
            "this run window".format(station.code, candidate.code, candidate.name)
        )
        return series, candidate, [], trail

    trail.append(
        "nmdb: no data at any rigidity-matched station for this run window; "
        "NMDB comparison unavailable"
    )
    return None, station, trail, []


def gather_external(
    args: argparse.Namespace,
    rs: RateSeries,
    station: Station,
    detector_rigidity: float,
    cache: Cache,
) -> Tuple[List[AlignedSeries], List[str], List[str], Station]:
    """Fetch and align every requested external source.

    Failures are collected as warnings rather than raised, so one dead
    endpoint cannot stop the run. NMDB additionally falls back through
    rigidity-ranked candidate stations (see ``fetch_nmdb_with_fallback``);
    the station actually used is returned as the fourth element so the
    caller's summary and plot metadata can reflect it. A successful
    fallback is reported via the third element, ``notes`` -- it succeeded,
    it just isn't the first-choice station, so it must not be rendered
    alongside genuine failures.

    Raises ``SourcesError`` if ``args.sources`` names anything outside
    ``ALL_SOURCES``; the caller (``main``) turns that into a clean nonzero
    exit rather than a raw ``SystemExit``.
    """
    requested = [s.strip().lower() for s in args.sources.split(",") if s.strip()]
    unknown = [s for s in requested if s not in ALL_SOURCES]
    if unknown:
        raise SourcesError(
            "unknown source(s): {0}; valid choices are {1}".format(
                ", ".join(unknown), ", ".join(ALL_SOURCES)
            )
        )

    start = rs.bin_start_utc[0]
    end = rs.bin_start_utc[-1] + np.timedelta64(
        int(rs.bin_length_s * 1e9), "ns"
    )

    aligned: List[AlignedSeries] = []
    warnings: List[str] = []
    notes: List[str] = []
    station_used = station

    fetchers = {
        "goes": lambda: fetch_goes_xray(start, end, cache),
        "kp": lambda: fetch_kp(start, end, cache),
        "sunspot": lambda: fetch_sunspot(start, end, cache),
    }

    for name in ALL_SOURCES:
        if name not in requested:
            continue
        if name == "nmdb":
            def nmdb_fetch_fn(code: str) -> ExternalSeries:
                return fetch_nmdb(start, end, code, rs.bin_length_s, cache)

            series, station_used, fb_warnings, fb_notes = fetch_nmdb_with_fallback(
                station, detector_rigidity, args.nmdb_station, nmdb_fetch_fn
            )
            warnings.extend(fb_warnings)
            notes.extend(fb_notes)
            if series is not None:
                aligned.append(align_to_bins(series, rs))
            continue

        try:
            series = fetchers[name]()
        except FetchError as exc:
            warnings.append("{0}: unavailable ({1})".format(name, exc))
            continue
        aligned.append(align_to_bins(series, rs))

    return aligned, warnings, notes, station_used


def format_summary(
    input_file: str,
    rs: RateSeries,
    correction: CorrectionResult,
    station: Station,
    detector_rigidity: float,
    candidates: List[Station],
    aligned: List[AlignedSeries],
    warnings: List[str],
    notes: List[str],
    args: argparse.Namespace,
    clock_drift_s: float,
) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append("CosmicWatch muon rate vs. solar activity")
    lines.append("=" * 72)
    lines.append("Input:            {0}".format(input_file))
    lines.append("Timestamps:       assumed UTC (no timezone conversion applied)")
    lines.append("Detector clock:   drift vs wall clock {0:+.2f} s over the run"
                 .format(clock_drift_s))
    lines.append("")
    lines.append("Rate")
    lines.append("  Bin length:     {0:g} s".format(rs.bin_length_s))
    lines.append("  Complete bins:  {0}".format(len(rs.counts)))
    lines.append("  Events used:    Flag == 1 (coincident) only, {0} total"
                 .format(int(rs.counts.sum())))
    # nanmean warns and returns nan if every bin is dead (all-NaN slice);
    # that degenerate case is acceptable, just skip the call to keep the
    # summary output quiet (format_summary's own ``warnings`` parameter
    # shadows the stdlib ``warnings`` module, so silencing via
    # catch_warnings isn't available here).
    if np.any(np.isfinite(rs.rate_hz)):
        mean_raw_rate = float(np.nanmean(rs.rate_hz))
    else:
        mean_raw_rate = float("nan")
    lines.append("  Mean raw rate:  {0:.4f} Hz".format(mean_raw_rate))
    lines.append("  Poisson error:  {0:.2f}% per bin (mean)"
                 .format(rs.mean_fractional_error * 100.0))
    if rs.mean_fractional_error > 0.03:
        lines.append("  NOTE: per-bin error exceeds 3%. Solar modulation in "
                     "muon rate is typically 0.5-3%,")
        lines.append("        so consider a longer --bin-length.")
    lines.append("")
    lines.append("Correction")
    lines.append("  Method:         {0}".format(correction.method))
    lines.append("  Met source:     {0}".format(args.met_source))
    lines.append("  Reference P0:   {0:.2f} hPa".format(correction.p0_hpa))
    lines.append("  Reference T0:   {0:.2f} C".format(correction.t0_c))
    if correction.method == "fit":
        lines.append("  beta_P:         {0:+.4f} +/- {1:.4f} %/hPa".format(
            correction.beta_p_percent, correction.beta_p_err * 100.0))
        lines.append("  beta_T:         {0:+.4f} +/- {1:.4f} %/C".format(
            correction.beta_t_percent, correction.beta_t_err * 100.0))
        lines.append("  R^2:            {0:.4f}".format(correction.r_squared))
    else:
        lines.append("  beta_P:         {0:+.4f} %/hPa (supplied)".format(
            correction.beta_p_percent))
        lines.append("  beta_T:         {0:+.4f} %/C (supplied)".format(
            correction.beta_t_percent))
        lines.append("  Provenance:     {0}".format(
            args.beta_p_source or "not stated"))
    lines.append("  NOTE: the temperature term corrects a DETECTOR systematic")
    lines.append("        (BMP280 enclosure temperature -> SiPM gain drift ->")
    lines.append("        threshold shift). It is NOT the atmospheric")
    lines.append("        temperature effect, which needs upper-air profiles.")
    for warning in correction.warnings:
        lines.append("  WARNING: {0}".format(warning))
    lines.append("")
    lines.append("Neutron monitor station")
    lines.append("  Detector Rc:    {0:.2f} GV (Stormer approximation, ~10-20% "
                 "accurate)".format(detector_rigidity))
    lines.append("  Selected:       {0} ({1})".format(station.code, station.name))
    lines.append("                  Rc={0:.2f} GV, altitude={1} m".format(
        station.rigidity_gv, station.altitude_m))
    lines.append("  Nearest by Rc:")
    for candidate in candidates:
        lines.append("      {0:5s} Rc={1:5.2f} GV  alt={2:5d} m  {3}".format(
            candidate.code, candidate.rigidity_gv, candidate.altitude_m,
            candidate.name))
    lines.append("  NOTE: rigidity matching favours high-altitude stations;")
    lines.append("        override with --nmdb-station if you want a")
    lines.append("        sea-level comparison.")
    lines.append("")
    lines.append("External sources")
    if aligned:
        for series in aligned:
            n_interp = int(np.count_nonzero(series.interpolated))
            n_valid = int(np.count_nonzero(np.isfinite(series.values)))
            lines.append("  OK   {0} [{1}] via {2}: {3} bins, {4} interpolated"
                         .format(series.name, series.units, series.source,
                                 n_valid, n_interp))
    else:
        lines.append("  (none retrieved)")
    for note in notes:
        lines.append("  NOTE {0}".format(note))
    for warning in warnings:
        lines.append("  FAIL {0}".format(warning))
    lines.append("")
    lines.append(NMDB_ACKNOWLEDGEMENT)
    lines.append("=" * 72)
    return "\n".join(lines)


def write_csv(path: str, rs: RateSeries, correction: CorrectionResult,
              aligned: List[AlignedSeries]) -> None:
    header = [
        "bin_start_utc", "counts", "livetime_s", "raw_rate_hz",
        "raw_rate_err_hz", "corrected_rate_hz", "corrected_err_hz",
        "pressure_hpa", "temperature_c",
    ]
    for series in aligned:
        header.append(series.name)
        header.append(series.name + "_interpolated")

    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for i in range(len(rs.counts)):
            row = [
                str(rs.bin_start_utc[i]), int(rs.counts[i]),
                "{0:.3f}".format(rs.livetime_s[i]),
                "{0:.6f}".format(rs.rate_hz[i]),
                "{0:.6f}".format(rs.rate_err_hz[i]),
                "{0:.6f}".format(correction.corrected_rate_hz[i]),
                "{0:.6f}".format(correction.corrected_err_hz[i]),
                "{0:.2f}".format(rs.press_hpa[i]),
                "{0:.2f}".format(rs.temp_c[i]),
            ]
            for series in aligned:
                value = series.values[i]
                row.append("" if not np.isfinite(value)
                           else "{0:.6g}".format(value))
                row.append("1" if series.interpolated[i] else "0")
            writer.writerow(row)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        events = read_events(args.input_file)
    except (DataFormatError, OSError) as exc:
        print("error: {0}".format(exc), file=sys.stderr)
        return 2

    try:
        rs = compute_rate(events, args.bin_length)
    except RateError as exc:
        print("error: {0}".format(exc), file=sys.stderr)
        return 2

    cache = Cache(args.cache_dir, refresh=args.refresh_cache)

    external_press = external_temp = None
    met_warnings: List[str] = []
    if args.met_source == "external":
        try:
            external_press, external_temp = fetch_external_meteorology(
                rs, args.lat, args.lon, cache
            )
        except FetchError as exc:
            met_warnings.append(
                "external meteorology unavailable ({0}); "
                "falling back to onboard sensors".format(exc)
            )
            external_press = external_temp = None

    try:
        correction = correct(
            rs,
            method=args.correction_method,
            beta_p_percent=args.beta_p,
            beta_t_percent=args.beta_t,
            press_hpa=external_press,
            temp_c=external_temp,
        )
    except CorrectionError as exc:
        print("error: {0}".format(exc), file=sys.stderr)
        return 2

    try:
        station, detector_rigidity, candidates = select_station(
            args.lat, args.lon, override_code=args.nmdb_station
        )
    except StationError as exc:
        print("error: {0}".format(exc), file=sys.stderr)
        return 2

    try:
        aligned, warnings, notes, station = gather_external(
            args, rs, station, detector_rigidity, cache
        )
    except SourcesError as exc:
        print("error: {0}".format(exc), file=sys.stderr)
        return 2
    warnings = met_warnings + warnings

    summary = format_summary(
        args.input_file, rs, correction, station, detector_rigidity,
        candidates, aligned, warnings, notes, args, events.clock_drift_s,
    )
    print(summary)

    run_name = os.path.splitext(os.path.basename(args.input_file))[0]
    output_dir = args.output_dir or os.path.dirname(
        os.path.abspath(args.input_file)
    )
    os.makedirs(output_dir, exist_ok=True)

    if correction.method == "fit":
        correction_label = "fit: beta_P={0:+.4f} %/hPa, beta_T={1:+.4f} %/C".format(
            correction.beta_p_percent, correction.beta_t_percent)
    else:
        correction_label = "literature: beta_P={0:+.4f} %/hPa ({1})".format(
            correction.beta_p_percent, args.beta_p_source or "source not stated")

    footer_notes = [
        "Temperature term corrects a detector systematic (SiPM gain drift), "
        "not the atmospheric temperature effect.",
        NMDB_ACKNOWLEDGEMENT,
    ]
    footer_notes.extend("Note: " + n for n in notes)
    footer_notes.extend("Unavailable - " + w for w in warnings)

    meta = PlotMetadata(
        run_name=run_name,
        station_label="{0} ({1}), Rc={2:.2f} GV, alt={3} m".format(
            station.code, station.name, station.rigidity_gv, station.altitude_m),
        correction_label=correction_label,
        footer_notes=footer_notes,
    )

    overlay_path = os.path.join(output_dir, run_name + "_overlay.html")
    side_path = os.path.join(output_dir, run_name + "_sidebyside.html")
    write_html(build_overlay(rs, correction, aligned, meta), overlay_path)
    write_html(build_side_by_side(rs, correction, aligned, meta), side_path)
    print("Wrote {0}".format(overlay_path))
    print("Wrote {0}".format(side_path))

    if args.export_csv:
        csv_path = os.path.join(output_dir, run_name + "_binned.csv")
        write_csv(csv_path, rs, correction, aligned)
        print("Wrote {0}".format(csv_path))

    return 0


if __name__ == "__main__":
    sys.exit(main())
