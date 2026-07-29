"""Combine multiple CosmicWatch files: group by detector, splice per group.

Files are grouped by detector name (the ``Name`` column). Files from the same
detector are spliced onto one absolute-UTC timeline; files from different
detectors stay separate. Only same-detector files may be spliced -- different
detectors are combined at the plotting layer, not here.
"""
from __future__ import annotations

import os
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np

from cosmicwatch_io import Events, read_detector_name, read_events


class SpliceError(ValueError):
    """Raised when files cannot be spliced (overlap, bad override, etc.)."""


def _parse_datetime_overrides(
    items: List[str], flag_name: str
) -> Dict[str, np.datetime64]:
    """Parse ``["FILE=ISO8601", ...]`` into a basename -> datetime64 map.

    ``FILE`` is reduced to its basename. A malformed item or an unparseable
    datetime raises ``SpliceError`` (with ``flag_name`` in the message).
    """
    overrides: Dict[str, np.datetime64] = {}
    for item in items:
        if "=" not in item:
            raise SpliceError(
                "bad {0} {1!r}; expected FILE=ISO8601".format(flag_name, item)
            )
        raw_path, raw_dt = item.split("=", 1)
        key = os.path.basename(raw_path.strip())
        text = raw_dt.strip().rstrip("Z")  # numpy datetime64 rejects trailing Z
        try:
            when = np.datetime64(text, "ns")
        except ValueError:
            raise SpliceError(
                "bad {0} datetime {1!r} for {2!r}".format(flag_name, raw_dt, key)
            )
        overrides[key] = when
    return overrides


def parse_start_overrides(items: List[str]) -> Dict[str, np.datetime64]:
    """Parse ``--start-time`` items (see ``_parse_datetime_overrides``)."""
    return _parse_datetime_overrides(items, "--start-time")


def parse_end_overrides(items: List[str]) -> Dict[str, np.datetime64]:
    """Parse ``--end-time`` items (see ``_parse_datetime_overrides``)."""
    return _parse_datetime_overrides(items, "--end-time")


def group_by_detector(paths: List[str]) -> "OrderedDict[str, List[str]]":
    """Group file paths by detector name, preserving first-seen order.

    Uses a light first-row read (``read_detector_name``) to obtain each file's
    detector name rather than a full parse -- the full parse happens once, in
    ``read_events_multi``. The grouping key is the file's first-row detector
    name (identical to ``read_events``'s majority name for well-formed files).
    A file with no valid data row falls back to its basename as the key; the
    full parse in ``read_events_multi`` then surfaces the ``DataFormatError``.
    """
    groups: "OrderedDict[str, List[str]]" = OrderedDict()
    for path in paths:
        name = read_detector_name(path) or os.path.basename(path)
        groups.setdefault(name, []).append(path)
    return groups


def _absolute_start(events: Events, basename: str,
                    overrides: Dict[str, np.datetime64]) -> np.datetime64:
    """The file's absolute UTC start: an override if given, else its clock."""
    if basename in overrides:
        return overrides[basename]
    return events.start_utc


def _truncate_to_end(
    events: Events, abs_start: np.datetime64, end: np.datetime64, basename: str
) -> Events:
    """Return ``events`` with only those whose absolute time is <= ``end``.

    Absolute time of each event is ``abs_start + (timestamp_s - timestamp_s[0])``.
    The boundary is inclusive. Raises ``SpliceError`` if nothing remains.
    """
    import dataclasses

    local = events.timestamp_s - events.timestamp_s[0]
    abs_ns = abs_start + (local * 1e9).astype("timedelta64[ns]")
    keep = abs_ns <= end
    if not np.any(keep):
        raise SpliceError(
            "--end-time {0} for {1!r} precedes the file's start; it removes "
            "every event".format(str(end), basename)
        )
    return dataclasses.replace(
        events,
        timestamp_s=events.timestamp_s[keep],
        flag=events.flag[keep],
        adc=events.adc[keep],
        deadtime_s=events.deadtime_s[keep],
        temp_c=events.temp_c[keep],
        press_pa=events.press_pa[keep],
    )


def read_events_multi(
    paths: List[str],
    start_overrides: Dict[str, np.datetime64],
    end_overrides: Optional[Dict[str, np.datetime64]] = None,
) -> Events:
    """Splice one detector's files onto one absolute-UTC axis.

    All ``paths`` are expected to share a detector name (the caller groups by
    detector). Returns a single ``Events`` whose ``timestamp_s`` is seconds
    from the earliest file's start, whose ``deadtime_s`` is cumulative across
    the whole record, and whose ``coverage_s`` lists each file's span.

    Raises ``SpliceError`` if two files overlap in absolute time.
    """
    if not paths:
        raise SpliceError("no files to splice")

    if end_overrides is None:
        end_overrides = {}

    parsed = []  # (basename, events, abs_start_datetime64, span_seconds)
    for path in paths:
        events = read_events(path)
        basename = os.path.basename(path)
        abs_start = _absolute_start(events, basename, start_overrides)
        if basename in end_overrides:
            events = _truncate_to_end(
                events, abs_start, end_overrides[basename], basename
            )
        span = float(events.timestamp_s[-1] - events.timestamp_s[0])
        parsed.append((basename, events, abs_start, span))

    # Chronological order by absolute start.
    parsed.sort(key=lambda item: item[2])

    # Absolute end of each file, for overlap detection.
    def abs_end(abs_start, span):
        return abs_start + np.timedelta64(int(round(span * 1e9)), "ns")

    for i in range(1, len(parsed)):
        prev_name, _, prev_start, prev_span = parsed[i - 1]
        name, _, start, _ = parsed[i]
        prev_end = abs_end(prev_start, prev_span)
        if start < prev_end:
            overlap_s = float((prev_end - start) / np.timedelta64(1, "s"))
            raise SpliceError(
                "files {0!r} and {1!r} overlap in time by {2:.1f} s; fix "
                "their clocks or drop one".format(prev_name, name, overlap_s)
            )

    t0 = parsed[0][2]  # earliest absolute start (numpy datetime64)

    ts_parts: List[np.ndarray] = []
    flag_parts: List[np.ndarray] = []
    adc_parts: List[np.ndarray] = []
    dead_parts: List[np.ndarray] = []
    temp_parts: List[np.ndarray] = []
    press_parts: List[np.ndarray] = []
    coverage: List[Tuple[float, float]] = []

    deadtime_offset = 0.0
    for basename, events, abs_start, span in parsed:
        # Seconds of this file's start from the global origin.
        start_s = float((abs_start - t0) / np.timedelta64(1, "s"))
        local = events.timestamp_s - events.timestamp_s[0]  # 0-based within file
        ts_parts.append(start_s + local)
        flag_parts.append(events.flag)
        adc_parts.append(events.adc)
        dead_parts.append(events.deadtime_s + deadtime_offset)
        temp_parts.append(events.temp_c)
        press_parts.append(events.press_pa)
        coverage.append((start_s, start_s + span))
        deadtime_offset += float(events.deadtime_s[-1])

    detector_name = parsed[0][1].detector_name

    return Events(
        timestamp_s=np.concatenate(ts_parts),
        flag=np.concatenate(flag_parts),
        adc=np.concatenate(adc_parts),
        deadtime_s=np.concatenate(dead_parts),
        temp_c=np.concatenate(temp_parts),
        press_pa=np.concatenate(press_parts),
        start_utc=np.datetime64(t0, "ns"),
        clock_drift_s=0.0,
        detector_name=detector_name,
        coverage_s=coverage,
    )
