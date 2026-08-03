import math
import numpy as np
import pytest
import anomaly


def test_poisson_cdf_matches_direct_sum_small_lambda():
    lam = 3.0
    for k in range(0, 12):
        direct = sum(math.exp(-lam) * lam**j / math.factorial(j)
                     for j in range(0, k + 1))
        assert anomaly.poisson_cdf(k, lam) == pytest.approx(direct, abs=1e-12)


def test_poisson_cdf_monotonic_and_bounded_large_lambda():
    lam = 1700.0
    vals = [anomaly.poisson_cdf(k, lam) for k in range(1500, 1900, 10)]
    assert all(0.0 <= v <= 1.0 for v in vals)
    assert all(b >= a for a, b in zip(vals, vals[1:]))
    assert anomaly.poisson_cdf(1700, lam) == pytest.approx(0.5, abs=0.02)


def test_baseline_mean_ignores_nan():
    r = np.array([0.48, np.nan, 0.50, 0.46])
    assert anomaly.baseline_mean(r) == pytest.approx((0.48 + 0.50 + 0.46) / 3)


def test_baseline_mean_all_nan_is_nan():
    assert math.isnan(anomaly.baseline_mean(np.array([np.nan, np.nan])))


def test_threshold_counts_two_sided_tail_probabilities():
    lam = 1000.0
    p = 0.05
    k_lo, k_hi = anomaly.threshold_counts(lam, p)
    # lower tail at k_lo is <= p/2, and one higher is > p/2 (k_lo is the largest)
    assert anomaly.poisson_cdf(k_lo, lam) <= p / 2
    assert anomaly.poisson_cdf(k_lo + 1, lam) > p / 2
    # upper tail at k_hi is <= p/2, and one lower is > p/2 (k_hi is the smallest)
    assert 1.0 - anomaly.poisson_cdf(k_hi - 1, lam) <= p / 2
    assert 1.0 - anomaly.poisson_cdf(k_hi - 2, lam) > p / 2
    assert k_lo < lam < k_hi


def test_threshold_counts_smaller_p_widens_the_band():
    lam = 1000.0
    lo5, hi5 = anomaly.threshold_counts(lam, 0.05)
    lo1, hi1 = anomaly.threshold_counts(lam, 0.01)
    assert lo1 <= lo5
    assert hi1 >= hi5


def test_threshold_rates_and_flags():
    livetime = np.array([3600.0, 3600.0])
    adj = np.array([1.0, 1.0])
    k_lo = np.array([1000, 1000])
    k_hi = np.array([1100, 1100])
    lower = anomaly.threshold_rates(k_lo, livetime, adj)
    upper = anomaly.threshold_rates(k_hi, livetime, adj)
    assert lower[0] == pytest.approx(1000 / 3600.0)
    assert upper[0] == pytest.approx(1100 / 3600.0)
    counts = np.array([1050, 1200])  # in-band, above-band
    flags = anomaly.flag_bins(counts, k_lo, k_hi)
    assert flags.tolist() == [False, True]
