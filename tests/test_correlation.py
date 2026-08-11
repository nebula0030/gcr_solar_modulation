import numpy as np
import pytest

import correlation


def _grid(n=40):
    t0 = np.datetime64("2026-07-10T00:00:00")
    return t0 + (np.arange(n) * 3600).astype("timedelta64[s]")


def test_perfect_positive_correlation_is_one():
    t = _grid()
    rate = np.linspace(0.45, 0.50, t.size)
    nmdb = 100.0 * rate  # exact linear -> r = 1
    res = correlation.pearson_muon_nmdb(rate, t, nmdb, np.zeros(t.size, bool),
                                        t, "NM (X)")
    assert res.r == pytest.approx(1.0, abs=1e-9)
    assert res.n == t.size and res.reason == ""


def test_perfect_negative_correlation_is_minus_one():
    t = _grid()
    rate = np.linspace(0.45, 0.50, t.size)
    nmdb = -3.0 * rate + 50.0
    res = correlation.pearson_muon_nmdb(rate, t, nmdb, np.zeros(t.size, bool),
                                        t, "NM (X)")
    assert res.r == pytest.approx(-1.0, abs=1e-9)


def test_dead_muon_bins_and_interpolated_nmdb_excluded():
    t = _grid()
    rate = np.linspace(0.45, 0.50, t.size)
    nmdb = 100.0 * rate
    rate[0] = np.nan                     # dead muon bin -> excluded
    interp = np.zeros(t.size, bool)
    interp[1] = True                     # interpolated NMDB fill -> excluded
    res = correlation.pearson_muon_nmdb(rate, t, nmdb, interp, t, "NM (X)")
    assert res.n == t.size - 2
    assert res.r == pytest.approx(1.0, abs=1e-9)


def test_insufficient_overlap_returns_none():
    t = _grid(4)
    rate = np.linspace(0.45, 0.50, 4)
    res = correlation.pearson_muon_nmdb(rate, t, 100.0 * rate,
                                        np.zeros(4, bool), t, "NM (X)",
                                        min_points=5)
    assert res.r is None and res.reason == "insufficient overlap" and res.n == 4


def test_no_variance_returns_none():
    t = _grid()
    rate = np.full(t.size, 0.47)         # constant -> undefined correlation
    res = correlation.pearson_muon_nmdb(rate, t, np.linspace(1, 2, t.size),
                                        np.zeros(t.size, bool), t, "NM (X)")
    assert res.r is None and res.reason == "no variance"


def test_interpolates_nmdb_onto_detector_grid():
    # NMDB on a coarser master grid, detector on a finer offset grid.
    master = _grid(20)
    nmdb = np.linspace(90.0, 110.0, 20)
    det_t = _grid(40)
    rate = np.interp(det_t.astype("int64"), master.astype("int64"), nmdb) / 200.0
    res = correlation.pearson_muon_nmdb(rate, det_t, nmdb, np.zeros(20, bool),
                                        master, "NM (X)")
    assert res.r == pytest.approx(1.0, abs=1e-6)
    assert res.n >= 5
