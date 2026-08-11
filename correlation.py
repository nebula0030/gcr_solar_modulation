"""Zero-lag Pearson correlation between a detector's corrected muon rate and
the NMDB neutron-monitor series. Pure numpy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CorrelationResult:
    r: Optional[float]
    n: int
    nmdb_name: str
    reason: str = ""


def _seconds(t) -> np.ndarray:
    return np.asarray(t, dtype="datetime64[ns]").astype("int64").astype(float)


def pearson_muon_nmdb(corrected_rate_hz, det_bin_mid_utc, nmdb_values,
                      nmdb_interpolated, master_mid, nmdb_name,
                      min_points: int = 5) -> CorrelationResult:
    rate = np.asarray(corrected_rate_hz, dtype=float)
    det_t = _seconds(det_bin_mid_utc)
    mst_t = _seconds(master_mid)
    nmdb = np.asarray(nmdb_values, dtype=float)
    interp = np.asarray(nmdb_interpolated, dtype=float)

    # Interpolate from the finite NMDB samples only, in time order.
    order = np.argsort(mst_t)
    mst_t, nmdb, interp = mst_t[order], nmdb[order], interp[order]
    finite = np.isfinite(nmdb)
    mst_t, nmdb, interp = mst_t[finite], nmdb[finite], interp[finite]
    if mst_t.size < 2:
        return CorrelationResult(None, 0, nmdb_name, "insufficient overlap")

    in_range = (det_t >= mst_t[0]) & (det_t <= mst_t[-1])
    nmdb_on = np.interp(det_t, mst_t, nmdb)
    fill_on = np.interp(det_t, mst_t, interp)  # >0 => drew on an interpolated fill

    usable = (in_range & np.isfinite(rate) & np.isfinite(nmdb_on)
              & (fill_on == 0.0))
    n = int(np.count_nonzero(usable))
    if n < min_points:
        return CorrelationResult(None, n, nmdb_name, "insufficient overlap")

    x, y = rate[usable], nmdb_on[usable]
    # Use peak-to-peak rather than std() == 0.0: std() subtracts a computed
    # mean and can leave float noise (~1e-17) even for a truly constant
    # array, so an exact-zero comparison on std() is unreliable.
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return CorrelationResult(None, n, nmdb_name, "no variance")
    return CorrelationResult(float(np.corrcoef(x, y)[0, 1]), n, nmdb_name, "")
