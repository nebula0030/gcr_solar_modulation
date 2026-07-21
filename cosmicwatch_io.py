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

    Rows are accumulated into in-memory blocks of at most ``chunk_size``
    rows; each completed block is converted to numpy arrays immediately and
    the underlying Python lists are discarded, so at most one chunk's worth
    of rows exist as Python objects at a time.
    """
    timestamps: List[float] = []
    flags: List[int] = []
    adcs: List[int] = []
    deadtimes: List[float] = []
    temps: List[float] = []
    presses: List[float] = []

    timestamp_chunks: List[np.ndarray] = []
    flag_chunks: List[np.ndarray] = []
    adc_chunks: List[np.ndarray] = []
    deadtime_chunks: List[np.ndarray] = []
    temp_chunks: List[np.ndarray] = []
    press_chunks: List[np.ndarray] = []

    def _flush_chunk() -> None:
        if not timestamps:
            return
        timestamp_chunks.append(np.asarray(timestamps, dtype=np.float64))
        flag_chunks.append(np.asarray(flags, dtype=np.int8))
        adc_chunks.append(np.asarray(adcs, dtype=np.int32))
        deadtime_chunks.append(np.asarray(deadtimes, dtype=np.float64))
        temp_chunks.append(np.asarray(temps, dtype=np.float64))
        press_chunks.append(np.asarray(presses, dtype=np.float64))
        timestamps.clear()
        flags.clear()
        adcs.clear()
        deadtimes.clear()
        temps.clear()
        presses.clear()

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
                # Parse every field into a local first. Nothing is appended
                # to a buffer until the whole row -- including the wall
                # clock -- has parsed successfully, so a mid-row failure
                # can never leave the buffers misaligned.
                timestamp = float(fields[_COL_TIMESTAMP])
                flag = int(fields[_COL_FLAG])
                adc = int(fields[_COL_ADC])
                deadtime = float(fields[_COL_DEADTIME])
                temp = float(fields[_COL_TEMP])
                press = float(fields[_COL_PRESS])
                wall = _parse_wall_clock(fields[_COL_TIME], fields[_COL_DATE])
            except ValueError:
                # Truncated or corrupt row. Nothing has been committed yet,
                # so skipping it is just a `continue`.
                continue

            timestamps.append(timestamp)
            flags.append(flag)
            adcs.append(adc)
            deadtimes.append(deadtime)
            temps.append(temp)
            presses.append(press)

            saw_any_row = True
            if first_wall is None:
                first_wall = wall
            last_wall = wall

            if len(timestamps) >= chunk_size:
                _flush_chunk()

    _flush_chunk()

    if not saw_any_row:
        if saw_wrong_width:
            raise DataFormatError(
                "No 13-column rows found in {0!r}. This tool supports only the "
                "13-column CosmicWatch v3X format; the 10-column v3 format is "
                "not supported.".format(path)
            )
        raise DataFormatError("No data rows found in {0!r}.".format(path))

    timestamp_arr = np.concatenate(timestamp_chunks)
    span_by_timestamp = float(timestamp_arr[-1] - timestamp_arr[0])
    span_by_wall = float((last_wall - first_wall) / np.timedelta64(1, "s"))
    clock_drift_s = span_by_wall - span_by_timestamp

    return Events(
        timestamp_s=timestamp_arr,
        flag=np.concatenate(flag_chunks),
        adc=np.concatenate(adc_chunks),
        deadtime_s=np.concatenate(deadtime_chunks),
        temp_c=np.concatenate(temp_chunks),
        press_pa=np.concatenate(press_chunks),
        start_utc=first_wall,
        clock_drift_s=clock_drift_s,
    )
