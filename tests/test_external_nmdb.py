from __future__ import annotations

import os

import numpy as np
import pytest

from external_sources import (
    Cache,
    FetchError,
    nmdb_resolution_minutes,
    parse_nmdb_ascii,
)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def read_fixture(name):
    with open(os.path.join(FIXTURES, name)) as handle:
        return handle.read()


def test_parse_nmdb_skips_header_and_reads_values():
    times, values = parse_nmdb_ascii(read_fixture("nmdb_oulu_sample.txt"))
    assert len(times) == 6  # 7 rows minus the null
    assert values[0] == pytest.approx(99.106)
    assert str(times[0]).startswith("2026-07-10T00:00:00")


def test_parse_nmdb_drops_null_rows():
    _, values = parse_nmdb_ascii(read_fixture("nmdb_oulu_sample.txt"))
    assert np.all(np.isfinite(values))


def test_parse_nmdb_rejects_a_response_with_no_data():
    with pytest.raises(FetchError):
        parse_nmdb_ascii("#  only a header\n#  and nothing else\n")


def test_parse_nmdb_rejects_an_html_error_page():
    """Guards the formchk=1 vs wget=1 trap documented in the spec."""
    with pytest.raises(FetchError):
        parse_nmdb_ascii("<!DOCTYPE HTML><html><body>oops</body></html>")


@pytest.mark.parametrize(
    "bin_length_s,expected",
    [(60.0, 2), (600.0, 10), (3600.0, 60), (21600.0, 360), (86400.0, 1440),
     (999999.0, 1440), (30.0, 2)],
)
def test_nmdb_resolution_snaps_to_allowed_values(bin_length_s, expected):
    assert nmdb_resolution_minutes(bin_length_s) == expected


def test_cache_stores_and_reuses(tmp_path):
    cache = Cache(str(tmp_path))
    calls = []

    def fetcher():
        calls.append(1)
        return b"payload"

    assert cache.get_or_fetch("k1", fetcher) == b"payload"
    assert cache.get_or_fetch("k1", fetcher) == b"payload"
    assert len(calls) == 1


def test_cache_refresh_forces_refetch(tmp_path):
    calls = []

    def fetcher():
        calls.append(1)
        return b"payload"

    Cache(str(tmp_path)).get_or_fetch("k1", fetcher)
    Cache(str(tmp_path), refresh=True).get_or_fetch("k1", fetcher)
    assert len(calls) == 2


def test_cache_does_not_store_failed_fetches(tmp_path):
    cache = Cache(str(tmp_path))

    def failing():
        raise FetchError("network down")

    with pytest.raises(FetchError):
        cache.get_or_fetch("k1", failing)
    assert cache.get_or_fetch("k1", lambda: b"later") == b"later"
