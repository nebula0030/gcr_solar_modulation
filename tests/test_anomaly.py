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


def test_flag_bins_lower_tail():
    counts = np.array([950, 1050])
    k_lo = np.array([1000, 1000])
    k_hi = np.array([1100, 1100])
    assert anomaly.flag_bins(counts, k_lo, k_hi).tolist() == [True, False]


def test_poisson_cdf_converges_at_large_lambda():
    lam = 43200.0
    assert anomaly.poisson_cdf(int(lam), lam) == pytest.approx(0.5, abs=0.01)
    # and thresholds are a sensible band around the mean
    k_lo, k_hi = anomaly.threshold_counts(lam, 0.05)
    assert k_lo < lam < k_hi


def test_threshold_counts_rejects_nonfinite_or_nonpositive_lambda():
    import math as _m
    for bad in (float("nan"), float("inf"), -5.0, 0.0):
        with pytest.raises(ValueError):
            anomaly.threshold_counts(bad, 0.05)


def _synthetic(n=200, lam=1000.0, T=3600.0, seed=0):
    rng = np.random.default_rng(seed)
    counts = rng.poisson(lam, size=n).astype(float)
    livetime = np.full(n, T)
    adj = np.ones(n)
    rate = counts / livetime
    return rate, livetime, adj


def test_marginal_observed_counts_sum_to_finite_bins():
    rate, livetime, adj = _synthetic()
    mu = anomaly.baseline_mean(rate)
    m = anomaly.marginal_distribution(rate, livetime, adj, mu)
    assert sum(m["observed"]) == len(rate)
    assert len(m["edges"]) == len(m["observed"]) + 1
    assert 10 <= len(m["observed"]) <= 40


def test_marginal_expected_total_matches_bin_count_and_no_scipy():
    import sys
    rate, livetime, adj = _synthetic()
    mu = anomaly.baseline_mean(rate)
    m = anomaly.marginal_distribution(rate, livetime, adj, mu)
    # Expected counts across all buckets ~ number of bins (mass mostly inside range)
    assert sum(m["expected"]) == pytest.approx(len(rate), rel=0.05)
    assert "scipy" not in sys.modules


def test_marginal_degenerate_returns_empty():
    rate = np.array([np.nan, 0.48])
    m = anomaly.marginal_distribution(rate, np.array([3600.0, 3600.0]),
                                      np.array([1.0, 1.0]), 0.48)
    assert m["observed"] == [] and m["edges"] == [] and m["expected"] == []


def test_marginal_expected_includes_top_edge_count():
    # 3 bins, identical livetime/adj; the max rate maps exactly to an integer
    # count (1010). Before the top-edge fix, that count's mass is dropped.
    T = 3600.0
    counts = np.array([1000.0, 1000.0, 1010.0])
    livetime = np.full(3, T)
    adj = np.ones(3)
    rate = counts / livetime
    mu = anomaly.baseline_mean(rate)
    m = anomaly.marginal_distribution(rate, livetime, adj, mu)
    lam = mu * T  # adj = 1, same for all three bins
    # Reference: all three bins' mass over the inclusive count range [1000, 1010].
    ref = 3.0 * (anomaly.poisson_cdf(1010, lam) - anomaly.poisson_cdf(999, lam))
    assert sum(m["expected"]) == pytest.approx(ref, abs=1e-6)


def test_marginal_handles_heterogeneous_livetime_and_adj():
    rng = np.random.default_rng(3)
    n = 150
    livetime = rng.uniform(2400.0, 3600.0, size=n)   # varying dead time
    adj = rng.uniform(0.97, 1.03, size=n)            # varying correction factor
    true_rate = 0.5
    counts = rng.poisson(true_rate * livetime / adj)
    rate = adj * counts / livetime
    mu = anomaly.baseline_mean(rate)
    m = anomaly.marginal_distribution(rate, livetime, adj, mu)
    assert sum(m["observed"]) == n
    assert len(m["edges"]) == len(m["observed"]) + 1
    assert 0.0 < sum(m["expected"]) <= n + 1e-6
    assert all(e2 > e1 for e1, e2 in zip(m["edges"], m["edges"][1:]))
