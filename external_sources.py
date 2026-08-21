"""Fetch public solar-activity and space-weather data.

Every fetcher shares one shape -- given a UTC window it returns an
``ExternalSeries`` -- and every network read goes through ``Cache`` so repeat
runs and offline work do not hit the network. Exception: ``fetch_solar_events``
only caches its DONKI reads when a ``cache`` is supplied, since ``cache`` is
optional there.

All endpoints below were verified live on 2026-07-21. Two carry traps worth
stating explicitly:

* NMDB's documented ``formchk=1`` parameter returns an HTML page. Use
  ``wget=1``, which returns plain ASCII.
* The Kp host ``kp.gfz-potsdam.de`` 301-redirects to ``kp.gfz.de``.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import requests

HTTP_TIMEOUT_S = 60

NMDB_URL = "https://www.nmdb.eu/nest/draw_graph.php"

#: Averaging intervals NMDB accepts, in minutes.
NMDB_RESOLUTIONS_MIN = (2, 5, 10, 30, 60, 120, 360, 720, 1440)

NMDB_ACKNOWLEDGEMENT = (
    "We acknowledge the NMDB database (www.nmdb.eu), founded under the "
    "European Union's FP7 programme (contract no. 213007), and the PIs of "
    "the individual neutron monitor stations."
)


class FetchError(Exception):
    """Raised when an external source cannot be retrieved or parsed."""


@dataclass
class ExternalSeries:
    """A time series from one external source."""

    name: str
    units: str
    source: str
    utc: np.ndarray
    values: np.ndarray


class Cache:
    """Filesystem cache of raw responses, keyed by a caller-supplied string."""

    def __init__(self, directory: str, refresh: bool = False) -> None:
        self.directory = directory
        self.refresh = refresh
        os.makedirs(directory, exist_ok=True)

    def _path(self, key: str) -> str:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
        return os.path.join(self.directory, digest + ".cache")

    def get_or_fetch(self, key: str, fetcher: Callable[[], bytes]) -> bytes:
        """Return cached bytes for ``key``, calling ``fetcher`` on a miss.

        A failing fetcher is never cached, so a transient network error does
        not poison later runs.
        """
        path = self._path(key)
        if not self.refresh and os.path.exists(path):
            with open(path, "rb") as handle:
                return handle.read()
        payload = fetcher()
        tmp = path + ".tmp"
        with open(tmp, "wb") as handle:
            handle.write(payload)
        os.replace(tmp, path)
        return payload


def http_get(url: str, params=None) -> bytes:
    """GET a URL, following redirects, raising ``FetchError`` on any failure."""
    try:
        response = requests.get(
            url, params=params, timeout=HTTP_TIMEOUT_S, allow_redirects=True
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise FetchError("request to {0} failed: {1}".format(url, exc))
    return response.content


def nmdb_resolution_minutes(bin_length_s: float) -> int:
    """Largest NMDB averaging interval that is no coarser than the bin."""
    wanted = bin_length_s / 60.0
    usable = [r for r in NMDB_RESOLUTIONS_MIN if r <= wanted]
    if not usable:
        return NMDB_RESOLUTIONS_MIN[0]
    return max(usable)


_NMDB_ROW = re.compile(
    r"^(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})\s*;\s*(\S+)\s*$"
)


def parse_nmdb_ascii(text: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parse NMDB's ASCII output into (times, values), dropping null rows."""
    if "<html" in text.lower() or "<!doctype" in text.lower():
        raise FetchError(
            "NMDB returned an HTML page rather than ASCII data. The request "
            "must use wget=1, not formchk=1."
        )

    times: List[np.datetime64] = []
    values: List[float] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _NMDB_ROW.match(line)
        if match is None:
            continue
        raw_value = match.group(2)
        if raw_value.lower() in ("null", "nan", ""):
            continue
        try:
            value = float(raw_value)
        except ValueError:
            continue
        times.append(np.datetime64(match.group(1).replace(" ", "T"), "ns"))
        values.append(value)

    if not times:
        raise FetchError("no usable data rows in the NMDB response")

    return np.asarray(times, dtype="datetime64[ns]"), np.asarray(
        values, dtype=np.float64
    )


def fetch_nmdb(
    start_utc: np.datetime64,
    end_utc: np.datetime64,
    station_code: str,
    bin_length_s: float,
    cache: Cache,
) -> ExternalSeries:
    """Fetch neutron monitor counts, corrected for efficiency."""
    resolution = nmdb_resolution_minutes(bin_length_s)
    start = _to_datetime(start_utc)
    end = _to_datetime(end_utc)

    params = {
        "wget": "1",
        "stations[]": station_code,
        "output": "ascii",
        "dtype": "corr_for_efficiency",
        "date_choice": "bydate",
        "start_year": start.year, "start_month": start.month,
        "start_day": start.day, "start_hour": start.hour,
        "start_min": start.minute,
        "end_year": end.year, "end_month": end.month,
        "end_day": end.day, "end_hour": end.hour, "end_min": end.minute,
        "yunits": "0",
        "tresolution": resolution,
    }
    key = "nmdb|{0}|{1}|{2}|{3}".format(station_code, start, end, resolution)
    payload = cache.get_or_fetch(key, lambda: http_get(NMDB_URL, params))
    times, values = parse_nmdb_ascii(payload.decode("utf-8", errors="replace"))

    return ExternalSeries(
        name="Neutron monitor ({0})".format(station_code),
        units="counts/s",
        source="NMDB",
        utc=times,
        values=values,
    )


def _to_datetime(value: np.datetime64):
    """numpy datetime64 -> stdlib datetime, truncated to seconds."""
    import datetime as _dt

    seconds = int(value.astype("datetime64[s]").astype(np.int64))
    return _dt.datetime(1970, 1, 1) + _dt.timedelta(seconds=seconds)


# --------------------------------------------------------------------------
# Geomagnetic Kp index (GFZ Potsdam)
# --------------------------------------------------------------------------

#: The kp.gfz-potsdam.de host 301-redirects here; use the new host directly.
KP_URL = "https://kp.gfz.de/app/json/"


def parse_kp_json(payload: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """Parse the GFZ Kp JSON payload into (times, values)."""
    import json

    try:
        data = json.loads(payload.decode("utf-8", errors="replace"))
    except ValueError as exc:
        raise FetchError("Kp response was not valid JSON: {0}".format(exc))

    if not isinstance(data, dict) or "Kp" not in data or "datetime" not in data:
        raise FetchError(
            "Kp response missing 'Kp' or 'datetime' keys; got "
            "{0}".format(sorted(data) if isinstance(data, dict) else type(data))
        )

    raw_values = data["Kp"]
    raw_times = data["datetime"]
    if len(raw_values) != len(raw_times):
        raise FetchError(
            "Kp response has {0} values but {1} timestamps".format(
                len(raw_values), len(raw_times)
            )
        )
    if not raw_values:
        raise FetchError("Kp response contained no samples")

    times = np.asarray(
        [np.datetime64(t.replace("Z", ""), "ns") for t in raw_times],
        dtype="datetime64[ns]",
    )
    values = np.asarray(raw_values, dtype=np.float64)
    return times, values


def fetch_kp(
    start_utc: np.datetime64, end_utc: np.datetime64, cache: Cache
) -> ExternalSeries:
    """Fetch the 3-hourly planetary Kp index."""
    start = _to_datetime(start_utc)
    end = _to_datetime(end_utc)
    params = {
        "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "index": "Kp",
    }
    key = "kp|{0}|{1}".format(start, end)
    payload = cache.get_or_fetch(key, lambda: http_get(KP_URL, params))
    times, values = parse_kp_json(payload)
    return ExternalSeries(
        name="Kp index",
        units="Kp (0-9)",
        source="GFZ Potsdam",
        utc=times,
        values=values,
    )


# --------------------------------------------------------------------------
# Sunspot number (SILSO, Royal Observatory of Belgium)
# --------------------------------------------------------------------------

SILSO_URL = "https://www.sidc.be/SILSO/DATA/EISN/EISN_current.csv"

#: SILSO writes -1 where no value is available.
_SILSO_MISSING = -1.0


def parse_silso_csv(text: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parse SILSO's daily estimated sunspot number CSV.

    Columns are: year, month, day, decimal year, value, stddev, ...
    """
    times: List[np.datetime64] = []
    values: List[float] = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
            value = float(parts[4])
        except ValueError:
            continue
        if value <= _SILSO_MISSING:
            continue
        times.append(
            np.datetime64("{0:04d}-{1:02d}-{2:02d}".format(year, month, day), "ns")
        )
        values.append(value)

    if not times:
        raise FetchError("no usable rows in the SILSO response")

    return np.asarray(times, dtype="datetime64[ns]"), np.asarray(
        values, dtype=np.float64
    )


def fetch_sunspot(
    start_utc: np.datetime64, end_utc: np.datetime64, cache: Cache
) -> ExternalSeries:
    """Fetch daily estimated international sunspot number.

    SILSO serves one rolling file rather than a date-range query, so the
    whole file is cached and then trimmed to the requested window.
    """
    key = "silso|current"
    payload = cache.get_or_fetch(key, lambda: http_get(SILSO_URL))
    times, values = parse_silso_csv(payload.decode("utf-8", errors="replace"))

    # Widen by a day each side so alignment has neighbours to interpolate from.
    lo = start_utc - np.timedelta64(1, "D")
    hi = end_utc + np.timedelta64(1, "D")
    inside = (times >= lo) & (times <= hi)
    if not np.any(inside):
        raise FetchError(
            "SILSO has no sunspot data covering {0} to {1}".format(
                start_utc, end_utc
            )
        )

    return ExternalSeries(
        name="Sunspot number",
        units="SSN",
        source="SILSO / Royal Observatory of Belgium",
        utc=times[inside],
        values=values[inside],
    )


# --------------------------------------------------------------------------
# GOES X-ray flux (NOAA SWPC live feed, NCEI archive fallback)
# --------------------------------------------------------------------------

GOES_LIVE_URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"

GOES_ARCHIVE_DIR = (
    "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/"
    "goes/goes19/l2/data/xrsf-l2-avg1m_science/{year:04d}/{month:02d}/"
)

#: SWPC's rolling JSON feed covers only the last 7 days.
GOES_LIVE_WINDOW_DAYS = 7

#: The long-wavelength band, the one conventionally quoted for flare class.
GOES_LONG_BAND = "0.1-0.8nm"


def parse_goes_live_json(payload: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """Parse SWPC's X-ray JSON, keeping only the long band."""
    import json

    try:
        records = json.loads(payload.decode("utf-8", errors="replace"))
    except ValueError as exc:
        raise FetchError("GOES response was not valid JSON: {0}".format(exc))
    if not isinstance(records, list):
        raise FetchError("GOES response was not a JSON list")

    pairs = []
    for record in records:
        if not isinstance(record, dict):
            continue
        if record.get("energy") != GOES_LONG_BAND:
            continue
        try:
            when = np.datetime64(str(record["time_tag"]).replace("Z", ""), "ns")
            flux = float(record["flux"])
        except (KeyError, ValueError, TypeError):
            continue
        pairs.append((when, flux))

    if not pairs:
        raise FetchError(
            "GOES response contained no {0} samples".format(GOES_LONG_BAND)
        )

    pairs.sort(key=lambda item: item[0])
    times = np.asarray([p[0] for p in pairs], dtype="datetime64[ns]")
    values = np.asarray([p[1] for p in pairs], dtype=np.float64)
    return times, values


#: Matches the version suffix of a GOES archive filename, e.g. "2-10-0" out
#: of "..._v2-10-0.nc".
_GOES_VERSION_RE = re.compile(r"_v(\d+(?:-\d+)*)\.nc$")


def _goes_version_key(filename: str) -> Tuple[int, ...]:
    """Numeric sort key for a GOES archive filename's version suffix.

    Comparing filenames as strings is wrong: "v2-2-1" sorts after
    "v2-10-0" lexicographically even though 2-10-0 is the newer version.
    A filename whose version cannot be parsed sorts lowest, so it never
    wins over a well-formed name and discovery never crashes on it.
    If every candidate is unparseable, they all tie at this same lowest
    key, and ``max()`` returns an arbitrary one of them rather than any
    guaranteed order.
    """
    match = _GOES_VERSION_RE.search(filename)
    if match is None:
        return (-1,)
    return tuple(int(part) for part in match.group(1).split("-"))


def discover_goes_archive_url(
    year: int, month: int, day: int, cache: Cache
) -> str:
    """Find the archive filename for one day by listing the month directory.

    The filename embeds both a satellite number and a processing version
    (for example ``sci_xrsf-l2-avg1m_g19_d20260710_v2-2-1.nc``). Both change
    over time -- the primary GOES satellite rotates and NCEI reprocessing
    bumps the version -- so the name is discovered rather than constructed.
    """
    directory = GOES_ARCHIVE_DIR.format(year=year, month=month)
    key = "goes-archive-listing|{0:04d}-{1:02d}".format(year, month)
    listing = cache.get_or_fetch(key, lambda: http_get(directory)).decode(
        "utf-8", errors="replace"
    )

    stamp = "d{0:04d}{1:02d}{2:02d}".format(year, month, day)
    pattern = re.compile(r'(sci_xrsf-l2-avg1m_g\d+_' + stamp + r'_v[\d-]+\.nc)')
    matches = pattern.findall(listing)
    if not matches:
        raise FetchError(
            "no GOES archive file for {0:04d}-{1:02d}-{2:02d} in {3}".format(
                year, month, day, directory
            )
        )
    winner = max(set(matches), key=_goes_version_key)
    return directory + winner


def _decode_goes_netcdf(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Decode one archived netCDF file into (times, long-band flux).

    Pure function of a filesystem path so it can be exercised offline with
    a synthetic file. netCDF4 returns a masked array wherever a sample is
    the fill-value sentinel; ``np.ma.filled`` turns those into NaN (instead
    of leaking the raw sentinel, which ``np.asarray`` would do) so the
    existing finiteness filter drops them regardless of the sentinel's
    sign. The same treatment applies to ``time`` so a masked timestamp
    cannot silently map to epoch+NaN.
    """
    import netCDF4

    dataset = netCDF4.Dataset(path)
    try:
        raw_flux = dataset.variables["xrsb_flux"][:]
        flux = np.ma.filled(raw_flux.astype("float64"), np.nan)
        raw_seconds = dataset.variables["time"][:]
        seconds = np.ma.filled(raw_seconds.astype("float64"), np.nan)
    finally:
        dataset.close()

    good = np.isfinite(flux) & np.isfinite(seconds) & (flux > 0)
    seconds = seconds[good]
    flux = flux[good]

    # GOES-R archive time is seconds since 2000-01-01 12:00:00 UTC.
    epoch = np.datetime64("2000-01-01T12:00:00", "ns")
    times = epoch + (seconds * 1e9).astype("timedelta64[ns]")
    return times, flux


def _read_goes_archive_day(
    year: int, month: int, day: int, cache: Cache
) -> Tuple[np.ndarray, np.ndarray]:
    """Download and read one archived netCDF day of long-band flux."""
    import tempfile

    url = discover_goes_archive_url(year, month, day, cache)
    key = "goes-archive|{0}".format(url)
    payload = cache.get_or_fetch(key, lambda: http_get(url))

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as handle:
        handle.write(payload)
        temp_path = handle.name
    try:
        return _decode_goes_netcdf(temp_path)
    finally:
        os.unlink(temp_path)


def fetch_goes_xray(
    start_utc: np.datetime64, end_utc: np.datetime64, cache: Cache
) -> ExternalSeries:
    """Fetch GOES long-band X-ray flux, live or from the NCEI archive."""
    now = np.datetime64("now", "ns")
    age_days = float((now - start_utc) / np.timedelta64(1, "D"))

    if age_days <= GOES_LIVE_WINDOW_DAYS:
        key = "goes-live|{0}".format(
            _to_datetime(start_utc).strftime("%Y-%m-%d")
        )
        payload = cache.get_or_fetch(key, lambda: http_get(GOES_LIVE_URL))
        times, values = parse_goes_live_json(payload)
    else:
        all_times: List[np.ndarray] = []
        all_values: List[np.ndarray] = []
        day = start_utc.astype("datetime64[D]")
        last = end_utc.astype("datetime64[D]")
        while day <= last:
            stamp = _to_datetime(day)
            day_times, day_values = _read_goes_archive_day(
                stamp.year, stamp.month, stamp.day, cache
            )
            all_times.append(day_times)
            all_values.append(day_values)
            day = day + np.timedelta64(1, "D")
        if not all_times:
            raise FetchError("no GOES archive days retrieved")
        times = np.concatenate(all_times)
        values = np.concatenate(all_values)

    inside = (times >= start_utc) & (times <= end_utc)
    if not np.any(inside):
        raise FetchError(
            "GOES data does not cover {0} to {1}".format(start_utc, end_utc)
        )

    return ExternalSeries(
        name="GOES X-ray flux (0.1-0.8 nm)",
        units="W/m^2",
        source="NOAA SWPC / NCEI",
        utc=times[inside],
        values=values[inside],
    )


# --------------------------------------------------------------------------
# Solar events (NASA DONKI): CME arrivals and X-class flares
# --------------------------------------------------------------------------

DONKI_ACKNOWLEDGEMENT = (
    "Solar events (CME/flare) courtesy of the NASA CCMC DONKI database "
    "(https://kauai.ccmc.gsfc.nasa.gov/DONKI/).")
_DONKI = "https://api.nasa.gov/DONKI"


@dataclass
class SolarEvent:
    """A single solar-activity event to mark on a plot."""

    kind: str            # "cme" | "flare"
    utc: np.datetime64
    label: str


def _donki_time(s):
    # "2026-08-11T14:24Z" -> np.datetime64("2026-08-11T14:24")
    return np.datetime64(str(s).replace("Z", ""))


def parse_donki_flares(payload: bytes, start, end) -> List[SolarEvent]:
    """Parse DONKI FLR JSON, keeping only X-class flares within the window."""
    out = []
    for f in json.loads(payload):
        cls = f.get("classType") or ""
        peak = f.get("peakTime") or f.get("beginTime")
        if not cls.startswith("X") or not peak:
            continue
        t = _donki_time(peak)
        if start <= t <= end:
            out.append(SolarEvent("flare", t, cls))
    return out


def _direct_hit_arrival(cme) -> Optional[str]:
    """Earth shock-arrival time if this CME's best model run is a DIRECT hit
    (reaches Earth, not a glancing blow, not a minor impact); else None.

    Best run: most-accurate analysis, then highest levelOfData, then latest
    submissionTime; judged on its first enlil entry that has an arrival time.
    """
    analyses = cme.get("cmeAnalyses") or []
    # latest submissionTime first (stable base), then most-accurate + highest level.
    analyses = sorted(analyses, key=lambda a: a.get("submissionTime") or "",
                      reverse=True)
    analyses = sorted(analyses, key=lambda a: (not a.get("isMostAccurate"),
                                               -(a.get("levelOfData") or 0)))
    for a in analyses:
        for e in (a.get("enlilList") or []):
            if e.get("estimatedShockArrivalTime"):
                if not e.get("isEarthGB") and not e.get("isEarthMinorImpact"):
                    return e["estimatedShockArrivalTime"]
                return None  # best arrival-bearing run is glancing/minor -> drop
    return None


def parse_donki_cme(payload: bytes, start, end) -> List[SolarEvent]:
    """Parse DONKI CME JSON, keeping only DIRECT Earth hits (not glancing blows
    or minor impacts), marked at their predicted Earth shock arrival."""
    out = []
    for c in json.loads(payload):
        arrival = _direct_hit_arrival(c)
        if not arrival:
            continue
        t = _donki_time(arrival)
        if start <= t <= end:
            out.append(SolarEvent("cme", t, "CME"))
    return out


def fetch_solar_events(
    start_utc: np.datetime64,
    end_utc: np.datetime64,
    api_key: str | None = None,
    cache: Cache | None = None,
) -> List[SolarEvent]:
    """Fetch X-class flares and Earth-directed CME arrivals from DONKI.

    Any ``FetchError`` from ``http_get`` propagates to the caller so it can
    be handled (or failed soft) by the caller.
    """
    key = api_key or os.environ.get("NASA_API_KEY") or "DEMO_KEY"
    s = str(np.datetime64(start_utc, "D"))
    e = str(np.datetime64(end_utc, "D"))

    def _get(url, params, cache_key):
        if cache is not None:
            return cache.get_or_fetch(cache_key, lambda: http_get(url, params=params))
        return http_get(url, params=params)

    flr = _get(_DONKI + "/FLR",
               {"startDate": s, "endDate": e, "api_key": key},
               "donki-flr|{0}|{1}".format(s, e))
    cme = _get(_DONKI + "/CME",
               {"startDate": s, "endDate": e, "api_key": key},
               "donki-cme|{0}|{1}".format(s, e))
    events = (parse_donki_flares(flr, np.datetime64(start_utc),
                                  np.datetime64(end_utc))
              + parse_donki_cme(cme, np.datetime64(start_utc),
                                 np.datetime64(end_utc)))
    events.sort(key=lambda ev: ev.utc)
    return events
