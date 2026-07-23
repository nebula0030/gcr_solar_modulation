"""Fetch public solar-activity and space-weather data.

Every fetcher shares one shape -- given a UTC window it returns an
``ExternalSeries`` -- and every network read goes through ``Cache`` so repeat
runs and offline work do not hit the network.

All endpoints below were verified live on 2026-07-21. Two carry traps worth
stating explicitly:

* NMDB's documented ``formchk=1`` parameter returns an HTML page. Use
  ``wget=1``, which returns plain ASCII.
* The Kp host ``kp.gfz-potsdam.de`` 301-redirects to ``kp.gfz.de``.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Callable, List, Tuple

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
