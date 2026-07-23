from __future__ import annotations

import os

import numpy as np
import pytest

from external_sources import FetchError, parse_kp_json, parse_silso_csv

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def read_bytes(name):
    with open(os.path.join(FIXTURES, name), "rb") as handle:
        return handle.read()


def test_parse_kp_reads_values_and_times():
    times, values = parse_kp_json(read_bytes("kp_sample.json"))
    assert len(values) == 8
    assert values[0] == pytest.approx(1.667)
    assert str(times[0]).startswith("2026-07-10T00:00:00")


def test_parse_kp_rejects_malformed_payload():
    with pytest.raises(FetchError):
        parse_kp_json(b"{\"unexpected\": true}")


def test_parse_kp_rejects_mismatched_lengths():
    with pytest.raises(FetchError):
        parse_kp_json(b'{"Kp": [1.0, 2.0], "datetime": ["2026-07-10T00:00:00Z"]}')


def test_parse_silso_reads_daily_values():
    text = open(os.path.join(FIXTURES, "silso_sample.csv")).read()
    times, values = parse_silso_csv(text)
    assert len(values) == 4  # the -1 row is dropped
    assert values[0] == pytest.approx(41.0)
    assert str(times[0]).startswith("2026-07-09")


def test_parse_silso_drops_missing_markers():
    text = open(os.path.join(FIXTURES, "silso_sample.csv")).read()
    _, values = parse_silso_csv(text)
    assert np.all(values >= 0)


def test_parse_silso_rejects_empty_input():
    with pytest.raises(FetchError):
        parse_silso_csv("\n\n")
