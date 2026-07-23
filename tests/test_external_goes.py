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


# netCDF4's real default fill value for an f8 variable. This MUST be
# positive: the production filter is `flux > 0`, so a negative sentinel
# (e.g. -9999.0) would be screened out by that comparison even under the
# old, broken decode logic (`np.asarray` instead of `np.ma.filled`), which
# made a negative-sentinel test pass for the wrong reason -- it never
# exercised the code path it was meant to guard. A positive sentinel this
# large is still a finite float, so `np.isfinite(...) & (flux > 0)` alone
# cannot screen it out; only properly filling masked entries with NaN
# (the fix) does.
POSITIVE_FILL = 9.969209968386869e36


def _write_synthetic_goes_netcdf(path):
    """Write a tiny netCDF file shaped like a GOES-R archive day.

    Three samples, each pinning a different case:
      index 0: fully valid -- flux and time both real, time == 0 (epoch).
      index 1: flux written through the fill-value sentinel, so netCDF4
        reports it as masked on read back -- the "missing sample" case
        Finding 1 is about.
      index 2: time written through the fill-value sentinel instead --
        the masked-*time* case Finding 2 is about, so a masked time can't
        silently map to epoch + garbage.
    """
    dataset = netCDF4.Dataset(path, "w")
    try:
        dataset.createDimension("time", None)
        flux_var = dataset.createVariable(
            "xrsb_flux", "f8", ("time",), fill_value=POSITIVE_FILL
        )
        time_var = dataset.createVariable(
            "time", "f8", ("time",), fill_value=POSITIVE_FILL
        )
        time_var.units = "seconds since 2000-01-01 12:00:00 UTC"
        flux_var[0] = 4.2e-07
        flux_var[2] = 6.6e-07
        time_var[:] = [0.0, 60.0, 120.0]
        # Overwrite the two sentinel slots explicitly (rather than relying
        # on them being left unset) so the intent is unambiguous: these
        # two entries are masked on purpose, each in a different variable.
        flux_var[1] = POSITIVE_FILL
        time_var[2] = POSITIVE_FILL
    finally:
        dataset.close()


def test_decode_goes_netcdf_drops_fill_values_and_decodes_noon_epoch(tmp_path):
    """Pins Finding 1 and Finding 2.

    Finding 1: a sample whose *flux* reads back masked (index 1) is
    dropped, using a POSITIVE fill sentinel so the assertion actually
    depends on the mask being filled with NaN rather than on the `flux >
    0` filter incidentally catching a negative sentinel.

    Finding 2: a sample whose *time* reads back masked (index 2) is also
    dropped, so a masked time cannot silently decode to epoch + garbage.

    Only index 0 should survive, and t=0 there must decode to NOON (the
    GOES-R epoch), not midnight.
    """
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
