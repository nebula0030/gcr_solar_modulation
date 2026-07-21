"""Parse CosmicWatch v3X detector data files.

The v3X format is 5 header lines followed by 13 tab-delimited columns:

    Event  Timestamp[s]  Flag  ADC[12b]  SiPM[mV]  Deadtime[s]  Temp[C]
    Press[Pa]  Accel(X:Y:Z)[g]  Gyro(X:Y:Z)[deg/sec]  Name  Time  Date

Two properties of this format drive the implementation:

* ``Deadtime[s]`` is *cumulative* over the run, not per-event.
* ``Time``/``Date`` are UTC, and ``Date`` is DD/MM/YYYY.

Files run to ~100 MB, so rows are accumulated in chunks and only the six
columns actually needed are retained.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import numpy as np

N_COLUMNS = 13
N_HEADER_LINES = 5

_COL_TIMESTAMP = 1
_COL_FLAG = 2
_COL_ADC = 3
_COL_DEADTIME = 5
_COL_TEMP = 6
_COL_PRESS = 7
_COL_TIME = 11
_COL_DATE = 12


class DataFormatError(ValueError):
    """Raised when a file is not a readable 13-column v3X data file."""


@dataclass
class Events:
    """All events from one run, in file order."""

    timestamp_s: np.ndarray
    flag: np.ndarray
    adc: np.ndarray
    deadtime_s: np.ndarray
    temp_c: np.ndarray
    press_pa: np.ndarray
    start_utc: np.datetime64
    clock_drift_s: float

    @property
    def press_hpa(self) -> np.ndarray:
        return self.press_pa / 100.0

    @property
    def utc(self) -> np.ndarray:
        """Absolute UTC per event.

        Derived from the run start plus the detector's own timestamp rather
        than by parsing 750k datetime strings. ``clock_drift_s`` records how
        far the two disagree across the whole run.
        """
        offsets = self.timestamp_s - self.timestamp_s[0]
        return self.start_utc + (offsets * 1e9).astype("timedelta64[ns]")


def _parse_wall_clock(time_field: str, date_field: str) -> np.datetime64:
    """Turn 'HH:MM:SS.ffffff' + 'DD/MM/YYYY' into a numpy UTC datetime."""
    dt = datetime.strptime(
        date_field.strip() + " " + time_field.strip(), "%d/%m/%Y %H:%M:%S.%f"
    )
    return np.datetime64(dt, "ns")


def read_events(path: str, chunk_size: int = 500_000) -> Events:
    """Read a v3X data file.

    Raises ``DataFormatError`` if the file is not 13-column v3X data.
    Malformed rows (including a truncated final line) are skipped.
    """
    timestamps: List[float] = []
    flags: List[int] = []
    adcs: List[int] = []
    deadtimes: List[float] = []
    temps: List[float] = []
    presses: List[float] = []

    first_wall: Optional[np.datetime64] = None
    last_wall: Optional[np.datetime64] = None
    saw_any_row = False
    saw_wrong_width = False

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for lineno, line in enumerate(handle):
            if lineno < N_HEADER_LINES or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != N_COLUMNS:
                saw_wrong_width = True
                continue
            try:
                timestamps.append(float(fields[_COL_TIMESTAMP]))
                flags.append(int(fields[_COL_FLAG]))
                adcs.append(int(fields[_COL_ADC]))
                deadtimes.append(float(fields[_COL_DEADTIME]))
                temps.append(float(fields[_COL_TEMP]))
                presses.append(float(fields[_COL_PRESS]))
                wall = _parse_wall_clock(fields[_COL_TIME], fields[_COL_DATE])
            except ValueError:
                # Truncated or corrupt row. It may have appended to some
                # buffers before failing, so trim every buffer back to the
                # shortest length and keep the columns aligned.
                buffers = [timestamps, flags, adcs, deadtimes, temps, presses]
                _truncate_all(buffers, min(len(b) for b in buffers))
                continue
            saw_any_row = True
            if first_wall is None:
                first_wall = wall
            last_wall = wall

    if not saw_any_row:
        if saw_wrong_width:
            raise DataFormatError(
                "No 13-column rows found in {0!r}. This tool supports only the "
                "13-column CosmicWatch v3X format; the 10-column v3 format is "
                "not supported.".format(path)
            )
        raise DataFormatError("No data rows found in {0!r}.".format(path))

    timestamp_arr = np.asarray(timestamps, dtype=np.float64)
    span_by_timestamp = float(timestamp_arr[-1] - timestamp_arr[0])
    span_by_wall = float((last_wall - first_wall) / np.timedelta64(1, "s"))
    clock_drift_s = span_by_wall - span_by_timestamp

    return Events(
        timestamp_s=timestamp_arr,
        flag=np.asarray(flags, dtype=np.int8),
        adc=np.asarray(adcs, dtype=np.int32),
        deadtime_s=np.asarray(deadtimes, dtype=np.float64),
        temp_c=np.asarray(temps, dtype=np.float64),
        press_pa=np.asarray(presses, dtype=np.float64),
        start_utc=first_wall,
        clock_drift_s=clock_drift_s,
    )


def _truncate_all(buffers: List[List], length: int) -> None:
    for buf in buffers:
        del buf[length:]
