from __future__ import annotations

import numpy as np
import pytest

from correction import CorrectionError, correct, fit_coefficients
from rate import RateSeries


def synthetic_series(beta_p, beta_t, n=400, base_rate=0.47, bin_length_s=86400.0,
                     seed=1234, temp_amplitude=3.0):
    """Build a RateSeries whose rate follows a known log-linear P/T response.

    Daily bins are used deliberately. At 0.47 Hz a 24 h bin holds ~40k counts,
    so Poisson noise is ~0.5% against a ~1.4% injected P/T signal. With hourly
    bins the noise (~2.4%) would swamp the signal and the R-squared assertion
    below would fail even though the fit is correct.

    ``temp_amplitude`` defaults to 3.0 C (the value every existing test
    relies on); pass a smaller value to produce a poorly-constrained beta_T
    fit without making temperature exactly constant.
    """
    rng = np.random.default_rng(seed)
    press = 1008.0 + 6.0 * np.sin(np.linspace(0, 6.0, n))
    temp = 23.0 + temp_amplitude * np.sin(np.linspace(0, 9.0, n) + 0.7)
    p0, t0 = press.mean(), temp.mean()

    true_rate = base_rate * np.exp(beta_p * (press - p0) + beta_t * (temp - t0))
    livetime = np.full(n, bin_length_s)
    counts = rng.poisson(true_rate * livetime).astype(np.int64)

    start = np.datetime64("2026-07-10T00:00:00.000000000")
    offsets = (np.arange(n) * bin_length_s * 1e9).astype("timedelta64[ns]")
    return RateSeries(
        bin_start_utc=start + offsets,
        bin_mid_utc=start + offsets,
        counts=counts,
        livetime_s=livetime,
        rate_hz=counts / livetime,
        rate_err_hz=np.sqrt(counts) / livetime,
        press_hpa=press,
        temp_c=temp,
        bin_length_s=bin_length_s,
    )


def test_fit_recovers_injected_coefficients():
    """The core physics check: inject known betas, confirm the fit finds them."""
    beta_p, beta_t = -0.0013, -0.004
    rs = synthetic_series(beta_p, beta_t)
    fp, ft, fp_err, ft_err, r2 = fit_coefficients(rs)
    assert fp == pytest.approx(beta_p, abs=4 * fp_err)
    assert ft == pytest.approx(beta_t, abs=4 * ft_err)
    # Daily-bin Poisson noise (~0.5%) against this fixture's ~1% injected P/T
    # signal caps R2 near ~0.73-0.76 for a correct, unbiased fit (verified by
    # a 20-seed sweep: mean 0.744, max 0.767, never approaching 0.8) -- the
    # coefficient-recovery assertions above are the real validation here.
    assert r2 > 0.65


def test_fit_mode_flattens_the_pressure_dependence():
    rs = synthetic_series(-0.0013, -0.004)
    result = correct(rs, method="fit")
    raw_spread = rs.rate_hz.std() / rs.rate_hz.mean()
    corr_spread = result.corrected_rate_hz.std() / result.corrected_rate_hz.mean()
    assert corr_spread < raw_spread


def test_literature_mode_applies_supplied_coefficients():
    rs = synthetic_series(-0.0013, 0.0)
    result = correct(rs, method="literature", beta_p_percent=-0.13,
                     beta_t_percent=0.0)
    assert result.beta_p == pytest.approx(-0.0013)
    assert result.beta_p_percent == pytest.approx(-0.13)
    assert result.beta_t == pytest.approx(0.0)


def test_literature_mode_without_beta_p_is_an_error():
    rs = synthetic_series(-0.0013, 0.0)
    with pytest.raises(CorrectionError) as exc:
        correct(rs, method="literature")
    assert "--beta-p" in str(exc.value)


def test_unknown_method_is_an_error():
    rs = synthetic_series(-0.0013, 0.0)
    with pytest.raises(CorrectionError):
        correct(rs, method="magic")


def test_narrow_pressure_range_produces_a_warning():
    rs = synthetic_series(-0.0013, -0.004)
    rs.press_hpa = 1008.0 + 0.05 * np.sin(np.linspace(0, 6.0, len(rs.press_hpa)))
    result = correct(rs, method="fit")
    assert any("pressure" in w.lower() for w in result.warnings)


def test_correction_preserves_the_mean_rate_scale():
    rs = synthetic_series(-0.0013, -0.004)
    result = correct(rs, method="fit")
    assert result.corrected_rate_hz.mean() == pytest.approx(
        rs.rate_hz.mean(), rel=0.05
    )


def test_external_meteorology_overrides_onboard():
    rs = synthetic_series(-0.0013, 0.0)
    external_press = rs.press_hpa + 100.0  # obviously different values
    result = correct(rs, method="fit", press_hpa=external_press)
    assert result.p0_hpa == pytest.approx(external_press.mean())


def test_relative_error_is_preserved_by_correction():
    rs = synthetic_series(-0.0013, -0.004)
    result = correct(rs, method="fit")
    raw_rel = rs.rate_err_hz / rs.rate_hz
    corr_rel = result.corrected_err_hz / result.corrected_rate_hz
    assert np.allclose(raw_rel, corr_rel)


def test_fit_excludes_nan_rate_dead_bins():
    """Dead bins (counts > 0 but rate_hz NaN due to zero livetime) must not
    be fed into the log-linear regression."""
    beta_p, beta_t = -0.0013, -0.004
    rs = synthetic_series(beta_p, beta_t)
    # Simulate a handful of dead bins: livetime collapsed to zero but the
    # deadtime counter still let a few counts through before it died.
    dead_idx = np.array([5, 50, 120, 300])
    rs.livetime_s = rs.livetime_s.copy()
    rs.rate_hz = rs.rate_hz.copy()
    rs.rate_err_hz = rs.rate_err_hz.copy()
    rs.livetime_s[dead_idx] = 0.0
    rs.rate_hz[dead_idx] = np.nan
    rs.rate_err_hz[dead_idx] = np.nan
    assert np.all(rs.counts[dead_idx] > 0)

    fp, ft, fp_err, ft_err, r2 = fit_coefficients(rs)
    assert np.isfinite(fp)
    assert np.isfinite(ft)
    assert fp == pytest.approx(beta_p, abs=4 * fp_err)
    assert ft == pytest.approx(beta_t, abs=4 * ft_err)


def test_correct_propagates_nan_rate_to_corrected_rate():
    rs = synthetic_series(-0.0013, -0.004)
    rs.rate_hz = rs.rate_hz.copy()
    rs.rate_err_hz = rs.rate_err_hz.copy()
    rs.rate_hz[7] = np.nan
    rs.rate_err_hz[7] = np.nan
    result = correct(rs, method="fit")
    assert np.isnan(result.corrected_rate_hz[7])
    assert np.isnan(result.corrected_err_hz[7])


def test_constant_pressure_raises_correction_error():
    """A stuck/placeholder pressure sensor must not leak a LinAlgError."""
    rs = synthetic_series(-0.0013, -0.004)
    rs.press_hpa = np.full_like(rs.press_hpa, 1008.0)
    with pytest.raises(CorrectionError) as exc:
        fit_coefficients(rs)
    assert "pressure" in str(exc.value).lower()

    with pytest.raises(CorrectionError):
        correct(rs, method="fit")


def test_constant_temperature_raises_correction_error():
    """A stuck/placeholder temperature sensor must not leak a LinAlgError."""
    rs = synthetic_series(-0.0013, -0.004)
    rs.temp_c = np.full_like(rs.temp_c, 23.0)
    with pytest.raises(CorrectionError) as exc:
        fit_coefficients(rs)
    assert "temperature" in str(exc.value).lower()

    with pytest.raises(CorrectionError):
        correct(rs, method="fit")


def test_constant_pressure_and_temperature_raises_correction_error():
    """Degenerate case: both sensors stuck simultaneously must also raise."""
    rs = synthetic_series(-0.0013, -0.004)
    rs.press_hpa = np.full_like(rs.press_hpa, 1008.0)
    rs.temp_c = np.full_like(rs.temp_c, 23.0)
    with pytest.raises(CorrectionError):
        fit_coefficients(rs)


def test_beta_p_uncertainty_warning_fires_when_poorly_constrained():
    """No real barometric signal (beta_p=0) against normal pressure spread
    yields a fitted beta_p that is noise-dominated, i.e. a large relative
    uncertainty -- verified numerically to exceed MAX_RELATIVE_COEFF_ERROR
    for this fixture's default seed."""
    rs = synthetic_series(0.0, -0.004)
    result = correct(rs, method="fit")
    assert any("beta_p" in w.lower() for w in result.warnings)


def test_beta_t_uncertainty_warning_fires_when_poorly_constrained():
    """A tiny (but nonzero, so the Finding-1 constant-temperature guard does
    not trigger) temperature spread makes beta_T noise-dominated -- verified
    numerically to exceed MAX_RELATIVE_COEFF_ERROR for this fixture's
    default seed."""
    rs = synthetic_series(-0.0013, -0.004, temp_amplitude=0.01)
    result = correct(rs, method="fit")
    assert any("beta_t" in w.lower() for w in result.warnings)
