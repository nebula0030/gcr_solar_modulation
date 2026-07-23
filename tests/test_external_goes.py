from __future__ import annotations

import os

import netCDF4
import numpy as np
import pytest

import external_sources
from external_sources import (
    GOES_LIVE_WINDOW_DAYS,
    Cache,
    FetchError,
    _decode_goes_netcdf,
    discover_goes_archive_url,
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


def _write_synthetic_goes_netcdf(path):
    """Write a tiny netCDF file shaped like a GOES-R archive day.

    One sample is valid; the other is written through the variable's
    fill-value so netCDF4 reports it as masked on read back -- exactly the
    "missing sample" case Finding 1 is about.
    """
    dataset = netCDF4.Dataset(path, "w")
    try:
        dataset.createDimension("time", None)
        flux_var = dataset.createVariable(
            "xrsb_flux", "f8", ("time",), fill_value=-9999.0
        )
        time_var = dataset.createVariable("time", "f8", ("time",))
        time_var.units = "seconds since 2000-01-01 12:00:00 UTC"
        # index 0: valid sample, exactly at the epoch (time=0).
        # index 1: left unset so it reads back through the fill value.
        flux_var[0] = 4.2e-07
        time_var[:] = [0.0, 60.0]
    finally:
        dataset.close()


def test_decode_goes_netcdf_drops_fill_values_and_decodes_noon_epoch(tmp_path):
    """Pins Finding 1: masked flux is dropped, and t=0 decodes to noon."""
    path = str(tmp_path / "synthetic.nc")
    _write_synthetic_goes_netcdf(path)

    times, values = _decode_goes_netcdf(path)

    assert len(values) == 1
    assert values[0] == pytest.approx(4.2e-07)
    # The GOES-R epoch is NOON, not midnight -- a decode that used midnight
    # would put this 12 hours off.
    assert times[0] == np.datetime64("2000-01-01T12:00:00", "ns")


def test_discover_goes_archive_url_finds_the_matching_file(tmp_path, monkeypatch):
    """Cache.get_or_fetch never touches the network: the fake listing is
    returned by a monkeypatched http_get, so this test is fully offline."""
    listing = (
        "sci_xrsf-l2-avg1m_g19_d20260709_v2-2-1.nc\n"
        "sci_xrsf-l2-avg1m_g19_d20260710_v2-2-1.nc\n"
    )
    monkeypatch.setattr(
        external_sources,
        "http_get",
        lambda url, params=None: listing.encode("utf-8"),
    )
    cache = Cache(str(tmp_path))

    url = discover_goes_archive_url(2026, 7, 10, cache)

    assert url.endswith("sci_xrsf-l2-avg1m_g19_d20260710_v2-2-1.nc")


def test_discover_goes_archive_url_raises_when_no_file_matches(tmp_path, monkeypatch):
    monkeypatch.setattr(
        external_sources,
        "http_get",
        lambda url, params=None: b"<html>nothing here</html>",
    )
    cache = Cache(str(tmp_path))

    with pytest.raises(FetchError):
        discover_goes_archive_url(2026, 7, 10, cache)


def test_discover_goes_archive_url_picks_numerically_newest_version(
    tmp_path, monkeypatch
):
    """Pins Finding 2: v2-10-0 must win over v2-2-1. Under the old
    lexicographic ``sorted(set(matches))[-1]`` logic, "v2-2-1" sorts after
    "v2-10-0" as a string, so this would select the older file."""
    listing = (
        "sci_xrsf-l2-avg1m_g19_d20260710_v2-2-1.nc\n"
        "sci_xrsf-l2-avg1m_g19_d20260710_v2-10-0.nc\n"
    )
    monkeypatch.setattr(
        external_sources,
        "http_get",
        lambda url, params=None: listing.encode("utf-8"),
    )
    cache = Cache(str(tmp_path))

    url = discover_goes_archive_url(2026, 7, 10, cache)

    assert url.endswith("v2-10-0.nc")
