"""Combine multiple CosmicWatch files: group by detector, splice per group.

Files are grouped by detector name (the ``Name`` column). Files from the same
detector are spliced onto one absolute-UTC timeline; files from different
detectors stay separate. Only same-detector files may be spliced -- different
detectors are combined at the plotting layer, not here.
"""
from __future__ import annotations

import os
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np

from cosmicwatch_io import Events, read_events


class SpliceError(ValueError):
    """Raised when files cannot be spliced (overlap, bad override, etc.)."""


def parse_start_overrides(items: List[str]) -> Dict[str, np.datetime64]:
    """Parse ``["FILE=ISO8601", ...]`` into a basename -> datetime64 map.

    ``FILE`` is reduced to its basename so it matches how files are keyed
    elsewhere. A malformed item or an unparseable datetime raises
    ``SpliceError``.
    """
    overrides: Dict[str, np.datetime64] = {}
    for item in items:
        if "=" not in item:
            raise SpliceError(
                "bad --start-time {0!r}; expected FILE=ISO8601".format(item)
            )
        raw_path, raw_dt = item.split("=", 1)
        key = os.path.basename(raw_path.strip())
        text = raw_dt.strip().rstrip("Z")  # numpy datetime64 rejects trailing Z
        try:
            when = np.datetime64(text, "ns")
        except ValueError:
            raise SpliceError(
                "bad --start-time datetime {0!r} for {1!r}".format(raw_dt, key)
            )
        overrides[key] = when
    return overrides


def group_by_detector(paths: List[str]) -> "OrderedDict[str, List[str]]":
    """Group file paths by detector name, preserving first-seen order.

    Reads each file (to obtain its detector name via the parser). The
    grouping key is ``Events.detector_name``.
    """
    groups: "OrderedDict[str, List[str]]" = OrderedDict()
    for path in paths:
        name = read_events(path).detector_name or os.path.basename(path)
        groups.setdefault(name, []).append(path)
    return groups
