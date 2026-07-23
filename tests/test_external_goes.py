from __future__ import annotations

import os

import numpy as np
import pytest

from external_sources import (
    GOES_LIVE_WINDOW_DAYS,
    FetchError,
    parse_goes_live_json,
)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def read_bytes(name):
    with open(os.path.join(FIXTURES, name), "rb") as handle:
        return handle.read()


def test_parse_goes_selects_the_long_band_only():
    times, values = parse_goes_live_json(read_bytes("goes_live_sample.json"))
    assert len(values) == 3
    assert values[0] == pytest.approx(2.1e-06)


def test_parse_goes_times_are_sorted():
    times, _ = parse_goes_live_json(read_bytes("goes_live_sample.json"))
    assert list(times) == sorted(times)


def test_parse_goes_rejects_payload_without_the_long_band():
    payload = (
        b'[{"time_tag": "2026-07-20T00:00:00Z", "flux": 1.0e-07, '
        b'"energy": "0.05-0.4nm"}]'
    )
    with pytest.raises(FetchError):
        parse_goes_live_json(payload)


def test_parse_goes_rejects_invalid_json():
    with pytest.raises(FetchError):
        parse_goes_live_json(b"not json at all")


def test_live_window_is_seven_days():
    assert GOES_LIVE_WINDOW_DAYS == 7
