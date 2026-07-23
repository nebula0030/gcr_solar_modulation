"""NMDB neutron monitor stations and detector-to-station matching.

Stations are matched to the detector by *geomagnetic vertical cutoff
rigidity*, not geographic distance. Cutoff rigidity determines which primary
cosmic rays can reach a location, and the amplitude of Forbush decreases and
solar modulation depends strongly on it -- so a rigidity-matched station is
the meaningful comparison, even if it is geographically distant.

Station rigidity and altitude are NMDB's published values, captured from
https://www.nmdb.eu/nest/help.php on 2026-07-21. NMDB does not publish
station coordinates, so distance-based matching is not offered.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

#: Geomagnetic north pole, IGRF centred-dipole approximation (epoch ~2020).
_GEOMAG_POLE_LAT_DEG = 80.65
_GEOMAG_POLE_LON_DEG = -72.68

#: Stormer constant for vertical cutoff at the Earth's surface, in GV.
_STORMER_CONSTANT_GV = 14.5

N_CANDIDATES_REPORTED = 5


class StationError(ValueError):
    """Raised when a requested station cannot be resolved."""


@dataclass(frozen=True)
class Station:
    """One NMDB neutron monitor."""

    code: str
    name: str
    rigidity_gv: float
    altitude_m: int


STATIONS: List[Station] = [
    Station("AATA", "Alma-Ata A", 5.9, 897),
    Station("AATB", "Alma-Ata B", 5.9, 3340),
    Station("AHMD", "Ahmedabad", 15.94, 50),
    Station("APTY", "Apatity", 0.65, 181),
    Station("ARNM", "Aragats", 7.1, 3200),
    Station("ATHN", "Athens", 8.53, 260),
    Station("BKSN", "Baksan", 5.7, 1700),
    Station("CALG", "Calgary", 1.08, 1123),
    Station("CALM", "NM de Castilla la Mancha", 6.95, 708),
    Station("CHAC", "CHACALTAYA", 11.8, 5270),
    Station("CLMX", "Climax", 3.0, 3400),
    Station("DJON", "Daejeon", 11.2, 200),
    Station("DOMB", "Dome C mini NM (bare)", 0.01, 3233),
    Station("DOMC", "Dome C mini NM", 0.01, 3233),
    Station("DRBS", "Dourbes", 3.18, 225),
    Station("DRHM", "Durham", 2.21, 20),
    Station("ESOI", "Emilio Segre Obs. Israel", 10.75, 2055),
    Station("FSMT", "Fort Smith", 0.3, 180),
    Station("HRMS", "Hermanus", 4.58, 26),
    Station("HUAN", "Huancayo", 12.92, 3400),
    Station("ICRB", "Izana Cosmic Ray Observatory (bare)", 11.5, 2390),
    Station("ICRO", "Izana Cosmic Ray Observatory", 11.5, 2390),
    Station("INVK", "Inuvik", 0.3, 21),
    Station("IRK2", "Irkustk 2", 3.64, 2000),
    Station("IRK3", "Irkutsk 3", 3.64, 3000),
    Station("IRKT", "Irkustk", 3.64, 435),
    Station("JBGO", "JangBogo", 0.3, 29),
    Station("JUNG", "IGY Jungfraujoch", 4.49, 3570),
    Station("KERG", "Kerguelen", 1.14, 33),
    Station("KGSN", "Kingston", 1.88, 65),
    Station("KIEL", "Kiel", 2.36, 54),
    Station("LMKS", "Lomnicky Stit", 3.84, 2634),
    Station("MCMU", "Mc Murdo", 0.3, 48),
    Station("MCRL", "Mobile Cosmic Ray Laboratory", 2.46, 200),
    Station("MGDN", "Magadan", 2.1, 220),
    Station("MOSC", "Moscow", 2.43, 200),
    Station("MRNY", "Mirny", 0.03, 30),
    Station("MWSB", "Mawson Bare", 0.22, 30),
    Station("MWSN", "Mawson", 0.22, 30),
    Station("MXCO", "Mexico", 8.28, 2274),
    Station("NAIN", "Nain", 0.3, 46),
    Station("NANM", "Nor-Amberd", 7.1, 2000),
    Station("NEU3", "Neumayer III mini neutron monitor", 0.1, 40),
    Station("NEWK", "Newark", 2.4, 50),
    Station("NRLK", "Norilsk", 0.63, 0),
    Station("NVBK", "Novosibirsk", 2.91, 163),
    Station("OULU", "Oulu", 0.81, 15),
    Station("PSNM", "Doi Inthanon (Princess Sirindhorn NM)", 16.8, 2565),
    Station("PTFM", "Potchefstroom", 6.98, 1351),
    Station("PWNK", "Peawanuck", 0.3, 53),
    Station("ROME", "Rome", 6.27, 0),
    Station("SANB", "Sanae D", 0.73, 52),
    Station("SNAE", "Sanae IV", 0.73, 856),
    Station("SOPB", "South Pole Bare", 0.1, 2820),
    Station("SOPO", "South Pole", 0.1, 2820),
    Station("TERA", "Terre Adelie", 0.01, 32),
    Station("THUL", "Thule", 0.3, 26),
    Station("TSMB", "Tsumeb", 9.15, 1240),
    Station("TXBY", "Tixie Bay", 0.48, 0),
    Station("UFSZ", "Zugspitze", 4.1, 2650),
    Station("YKTK", "Yakutsk", 1.65, 105),
    Station("ZUGS", "Zugspitze", 4.24, 2960),
]


def geomagnetic_latitude(lat_deg: float, lon_deg: float) -> float:
    """Geomagnetic latitude from a centred-dipole model, in degrees."""
    lat = math.radians(lat_deg)
    pole_lat = math.radians(_GEOMAG_POLE_LAT_DEG)
    delta_lon = math.radians(lon_deg - _GEOMAG_POLE_LON_DEG)
    sin_maglat = math.sin(lat) * math.sin(pole_lat) + math.cos(lat) * math.cos(
        pole_lat
    ) * math.cos(delta_lon)
    sin_maglat = max(-1.0, min(1.0, sin_maglat))
    return math.degrees(math.asin(sin_maglat))


def vertical_cutoff_rigidity(lat_deg: float, lon_deg: float) -> float:
    """Estimate vertical cutoff rigidity in GV via the Stormer approximation.

    ``Rc = 14.5 * cos^4(geomagnetic latitude)``. Accurate to roughly 10-20%,
    which is ample for ranking stations but should not be quoted as a precise
    value. A full treatment would trace trajectories through an IGRF field
    model.
    """
    maglat = math.radians(geomagnetic_latitude(lat_deg, lon_deg))
    return _STORMER_CONSTANT_GV * math.cos(maglat) ** 4


def rank_by_rigidity(target_rigidity_gv: float) -> List[Station]:
    """Stations ordered by how close their cutoff rigidity is to the target."""
    return sorted(STATIONS, key=lambda s: abs(s.rigidity_gv - target_rigidity_gv))


def select_station(
    lat_deg: float,
    lon_deg: float,
    override_code: Optional[str] = None,
) -> Tuple[Station, float, List[Station]]:
    """Choose the NMDB station to compare against.

    Returns ``(chosen_station, detector_rigidity_gv, top_candidates)``. The
    candidate list is for transparency in the run summary: rigidity matching
    tends to favour high-altitude mountain stations, and the caller should
    show the user what else was close.
    """
    detector_rigidity = vertical_cutoff_rigidity(lat_deg, lon_deg)
    ranked = rank_by_rigidity(detector_rigidity)
    candidates = ranked[:N_CANDIDATES_REPORTED]

    if override_code is not None:
        wanted = override_code.strip().upper()
        for station in STATIONS:
            if station.code == wanted:
                return station, detector_rigidity, candidates
        raise StationError(
            "unknown NMDB station code {0!r}. Known codes: {1}".format(
                override_code, ", ".join(sorted(s.code for s in STATIONS))
            )
        )

    return ranked[0], detector_rigidity, candidates
