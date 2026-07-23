"""Remove pressure and temperature effects from the muon rate.

Model (log-linear, after Duperier 1949 / Dorman 2004; see the design spec
for full provenance)::

    ln(R_meas / R_0) = beta_P * (P - P_0) + beta_T * (T - T_0)
    R_corr           = R_meas * exp(-beta_P * (P - P_0) - beta_T * (T - T_0))

Coefficients are stored internally as *fractional* change per hPa and per
degree C. The CLI exposes them as percent; conversion happens at the
boundary.

A caveat that the output text must preserve: the onboard temperature sensor
(Bosch BMP280) reads enclosure temperature, not atmospheric temperature. The
temperature term here is therefore a *detector systematic* correction --
principally SiPM gain drift shifting the effective threshold -- and not the
classical atmospheric temperature effect.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from rate import RateSeries

#: Below this total pressure swing (hPa) a fitted beta_P is poorly constrained.
MIN_PRESSURE_RANGE_HPA = 5.0

#: Relative uncertainty above which a fitted coefficient is called unreliable.
MAX_RELATIVE_COEFF_ERROR = 0.5


class CorrectionError(ValueError):
    """Raised when a correction cannot be performed as requested."""


@dataclass
class CorrectionResult:
    """Corrected rate plus the coefficients and diagnostics behind it."""

    corrected_rate_hz: np.ndarray
    corrected_err_hz: np.ndarray
    beta_p: float
    beta_t: float
    beta_p_err: float
    beta_t_err: float
    r_squared: float
    p0_hpa: float
    t0_c: float
    method: str
    warnings: List[str] = field(default_factory=list)

    @property
    def beta_p_percent(self) -> float:
        return self.beta_p * 100.0

    @property
    def beta_t_percent(self) -> float:
        return self.beta_t * 100.0


def fit_coefficients(
    rs: RateSeries,
    press_hpa: Optional[np.ndarray] = None,
    temp_c: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, float, float]:
    """Weighted least-squares fit of ln(rate) on pressure and temperature.

    Returns ``(beta_p, beta_t, beta_p_err, beta_t_err, r_squared)`` in
    fractional units. Bins are weighted by their counts because the variance
    of ``ln(R)`` is approximately ``1 / N``.

    Bins with zero counts, non-finite pressure/temperature, or a non-finite
    ``rate_hz`` are excluded. The last case matters because a "dead" bin
    (deadtime consumed the whole bin) can still have ``counts > 0`` from
    events recorded before the detector went dead, while ``rate_hz`` itself
    is NaN -- such a bin must not enter the regression.
    """
    press = rs.press_hpa if press_hpa is None else press_hpa
    temp = rs.temp_c if temp_c is None else temp_c

    usable = (
        (rs.counts > 0)
        & np.isfinite(press)
        & np.isfinite(temp)
        & np.isfinite(rs.rate_hz)
    )
    if usable.sum() < 3:
        raise CorrectionError(
            "need at least 3 bins with counts to fit coefficients, got "
            "{0}".format(int(usable.sum()))
        )

    p = press[usable]
    t = temp[usable]
    y = np.log(rs.rate_hz[usable])
    w = np.sqrt(rs.counts[usable].astype(np.float64))

    p_constant = np.ptp(p) == 0
    t_constant = np.ptp(t) == 0
    if p_constant and t_constant:
        raise CorrectionError(
            "pressure and temperature are both constant across the usable "
            "bins, so beta_P and beta_T cannot be fit; check the pressure "
            "and temperature sensors or use --correction-method literature."
        )
    if p_constant:
        raise CorrectionError(
            "pressure is constant across the usable bins, so beta_P cannot "
            "be fit; check the pressure sensor or use "
            "--correction-method literature."
        )
    if t_constant:
        raise CorrectionError(
            "temperature is constant across the usable bins, so beta_T "
            "cannot be fit; check the temperature sensor or use "
            "--correction-method literature."
        )

    design = np.column_stack(
        [np.ones(usable.sum()), p - p.mean(), t - t.mean()]
    )
    weighted_design = design * w[:, None]
    weighted_y = y * w

    coeffs, _, _, _ = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)
    beta_p, beta_t = float(coeffs[1]), float(coeffs[2])

    residuals = weighted_y - weighted_design.dot(coeffs)
    dof = max(usable.sum() - 3, 1)
    sigma2 = float(residuals.dot(residuals)) / dof
    covariance = sigma2 * np.linalg.inv(weighted_design.T.dot(weighted_design))
    beta_p_err = float(np.sqrt(covariance[1, 1]))
    beta_t_err = float(np.sqrt(covariance[2, 2]))

    ss_res = float(np.sum((y - design.dot(coeffs)) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return beta_p, beta_t, beta_p_err, beta_t_err, r_squared


def correct(
    rs: RateSeries,
    method: str,
    beta_p_percent: Optional[float] = None,
    beta_t_percent: Optional[float] = None,
    press_hpa: Optional[np.ndarray] = None,
    temp_c: Optional[np.ndarray] = None,
) -> CorrectionResult:
    """Apply the pressure/temperature correction.

    ``method`` is ``"fit"`` (derive coefficients from this run) or
    ``"literature"`` (use the supplied ``beta_p_percent`` /
    ``beta_t_percent``). ``press_hpa`` and ``temp_c`` override the onboard
    values when an external meteorology source is in use.

    ``rate_hz`` bins that are NaN (dead bins; see ``rate.compute_rate``)
    propagate to NaN in ``corrected_rate_hz`` / ``corrected_err_hz`` via the
    multiplication below -- they are not silently dropped or fabricated.
    """
    press = rs.press_hpa if press_hpa is None else np.asarray(press_hpa, dtype=float)
    temp = rs.temp_c if temp_c is None else np.asarray(temp_c, dtype=float)
    warnings: List[str] = []

    if method == "fit":
        beta_p, beta_t, beta_p_err, beta_t_err, r_squared = fit_coefficients(
            rs, press_hpa=press, temp_c=temp
        )
        pressure_range = float(np.nanmax(press) - np.nanmin(press))
        if pressure_range < MIN_PRESSURE_RANGE_HPA:
            warnings.append(
                "Pressure range over this run is only {0:.2f} hPa (below the "
                "{1:.0f} hPa guideline), so the fitted barometric coefficient "
                "is weakly constrained.".format(
                    pressure_range, MIN_PRESSURE_RANGE_HPA
                )
            )
        if beta_p != 0 and abs(beta_p_err / beta_p) > MAX_RELATIVE_COEFF_ERROR:
            warnings.append(
                "Fitted beta_P has {0:.0f}% relative uncertainty; treat the "
                "corrected rate with caution or supply a known coefficient "
                "via --correction-method literature --beta-p.".format(
                    100.0 * abs(beta_p_err / beta_p)
                )
            )
        if beta_t != 0 and abs(beta_t_err / beta_t) > MAX_RELATIVE_COEFF_ERROR:
            warnings.append(
                "Fitted beta_T has {0:.0f}% relative uncertainty; treat the "
                "temperature term of the corrected rate with caution or "
                "supply a known coefficient via --correction-method "
                "literature --beta-t.".format(
                    100.0 * abs(beta_t_err / beta_t)
                )
            )
    elif method == "literature":
        if beta_p_percent is None:
            raise CorrectionError(
                "--correction-method literature requires an explicit --beta-p "
                "(in percent per hPa). No default is provided because no "
                "verifiable CosmicWatch-specific published value was found; "
                "use --correction-method fit to derive one from this run."
            )
        beta_p = float(beta_p_percent) / 100.0
        beta_t = float(beta_t_percent or 0.0) / 100.0
        beta_p_err = beta_t_err = 0.0
        r_squared = float("nan")
    else:
        raise CorrectionError(
            "unknown correction method {0!r}; expected 'fit' or "
            "'literature'".format(method)
        )

    p0 = float(np.nanmean(press))
    t0 = float(np.nanmean(temp))
    adjustment = np.exp(-beta_p * (press - p0) - beta_t * (temp - t0))

    corrected_rate = rs.rate_hz * adjustment
    corrected_err = rs.rate_err_hz * adjustment

    return CorrectionResult(
        corrected_rate_hz=corrected_rate,
        corrected_err_hz=corrected_err,
        beta_p=beta_p,
        beta_t=beta_t,
        beta_p_err=beta_p_err,
        beta_t_err=beta_t_err,
        r_squared=r_squared,
        p0_hpa=p0,
        t0_c=t0,
        method=method,
        warnings=warnings,
    )
