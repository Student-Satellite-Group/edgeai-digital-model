"""
orbital_model.py — Phase 1 orbital mechanics for the edgeai Digital Model.

Provides closed-form orbit parameters for a circular Low-Earth Orbit (LEO):
    - orbital_period(altitude_km)      : Keplerian period T = 2*pi*sqrt(a^3/mu)
    - sun_sync_inclination(altitude_km): J2 nodal-precession sun-synchronous inclination
    - ground_track(altitude_km, ...)   : sub-satellite lat/lon over time (SGP4 propagation)

All values follow the locked-in mission parameters (altitude 500 km, region
28.6 N / 77.2 E, radius 50 km). Units are SI for the internal math; inputs are
kilometres, outputs are seconds / degrees respectively.
"""

import numpy as np

# Locked-in mission altitude (km) — PROVISIONAL until final mission review.
TARGET_ALTITUDE_KM = 500.0

# Locked-in target region — New Delhi, radius 50 km (Task 1.5).
TARGET_REGION_LAT_DEG = 28.6
TARGET_REGION_LON_DEG = 77.2
TARGET_REGION_RADIUS_KM = 50.0

# Earth constants (SI units).
MU_EARTH = 398600e9      # standard gravitational parameter, m^3/s^2
RE_EARTH = 6371e3        # mean radius, m (nominal WGS-84 geoid; see cross_validation_log.md
                          # for why this differs slightly from the 6378.137 km equatorial radius)
J2_EARTH = 1.0826e-3     # J2 zonal harmonic (Earth oblateness)
DEG_PER_DAY = 360.0 / 365.25   # mean orbital Sun motion, deg/day

# Sun-synchronous condition: RAAN precession must match orbital Sun motion.
SSO_RAAN_DOT_DEG_PER_DAY = 0.9856  # deg/day (360 deg / 365.25 days)


def orbital_period(altitude_km):
    """Return the orbital period in seconds for a circular orbit at altitude_km.

    T = 2*pi*sqrt(a^3 / mu), with a = R_e + altitude (semi-major axis).

    >>> orbital_period(500.0)          # ~94.3 minutes = 5658 s
    >>> round(orbital_period(500.0) / 60.0, 2)
    94.33
    """
    a = RE_EARTH + altitude_km * 1000.0          # semi-major axis, m
    return 2.0 * np.pi * np.sqrt(a ** 3 / MU_EARTH)


def sun_sync_inclination(altitude_km):
    """Return the sun-synchronous inclination in degrees (J2 precession model).

    Solves the nodal precession equation for the circular-orbit case:
        RAAN_dot = -(3/2) J2 (R_e / a)^2 n cos(i)
    set equal to the orbital Sun motion (0.9856 deg/day ... converted to rad/s),
    then inverts for i using arccos. np.clip guards the arccos domain.
    """
    a = RE_EARTH + altitude_km * 1000.0          # semi-major axis, m
    n = 2.0 * np.pi / orbital_period(altitude_km)  # mean motion, rad/s

    # Sun motion in rad/s (0.9856 deg/day -> rad/s).
    raan_dot_target = SSO_RAAN_DOT_DEG_PER_DAY * np.pi / 180.0 / 86400.0

    # RAAN_dot = -(3/2) J2 (R_e / a)^2 n cos(i)  =>  cos(i) = ...
    cos_i = -raan_dot_target * (2.0 / 3.0) * (a ** 2) / (J2_EARTH * RE_EARTH ** 2 * n)
    cos_i = np.clip(cos_i, -1.0, 1.0)            # guard floating-point edges
    return float(np.degrees(np.arccos(cos_i)))


# ---------------------------------------------------------------------------
# Ground track (Task 1.4) — propagated via SGP4, not the closed-form formulas
# above. SGP4 traditionally pairs with WGS72 mu/Re, which is why the two
# constants below differ slightly from MU_EARTH/RE_EARTH: mixing conventions
# would reintroduce exactly the kind of mismatch documented and cross-checked
# in cross_validation_log.md. This is the same element-based Satrec
# initialization technique validated (against hand calc + orbital_period /
# sun_sync_inclination) in Task 1.3's cross_validation.py.
# ---------------------------------------------------------------------------
_SGP4_MU_KM3_PER_MIN2 = 398600.8 * 3600.0   # WGS72 mu, s^-2 -> min^-2
_SGP4_RE_KM = 6378.135                       # WGS72 equatorial radius, km


def _make_satrec(altitude_km, inclination_deg, raan_deg=0.0, epoch=(2026, 1, 1, 0, 0, 0)):
    """Build an SGP4 Satrec for a circular orbit directly from Keplerian
    elements (no TLE string / checksum needed).

    raan_deg (right ascension of ascending node at epoch) is what actually
    determines *which* longitudes the ground track passes over -- it is a
    real mission-planning free parameter (set via launch timing), not an
    arbitrary constant, which is why it is exposed here rather than hardcoded
    (see passes_over_region()/generate_overpass_schedule.py, which calibrate
    it so the locked-in New Delhi target is actually covered)."""
    from sgp4.api import Satrec, WGS84, jday

    a_km = _SGP4_RE_KM + altitude_km
    n_rad_per_min = np.sqrt(_SGP4_MU_KM3_PER_MIN2 / a_km ** 3)
    sat = Satrec()
    jd, fr = jday(*epoch)
    sat.sgp4init(
        WGS84, 'i', 99999, jd + fr - 2433281.5,
        0.0, 0.0, 0.0,             # bstar, ndot, nddot -- no drag
        0.0001,                     # eccentricity (near-zero; avoids SGP4 singularity)
        0.0,                         # argument of perigee, rad
        np.deg2rad(inclination_deg),
        0.0,                         # mean anomaly, rad
        n_rad_per_min,
        np.deg2rad(raan_deg),
    )
    return sat, epoch


def ground_track(altitude_km, inclination_deg=None, duration_hours=24.0, step_s=60.0, raan_deg=0.0):
    """Propagate a circular orbit and return its ground track.

    If inclination_deg is omitted, the sun-synchronous inclination for
    altitude_km is used (the locked-in mission case). Propagation is via
    SGP4 (through skyfield's WGS-84 geodetic subpoint calculation, which
    correctly converts the propagated ECI/TEME position through Earth's
    rotation to geodetic lat/lon -- this is the "ECI -> ECEF -> lat/lon"
    step named in the task spec).

    raan_deg shifts which longitudes the ground track covers (see
    _make_satrec) -- defaults to 0.0 for a generic/illustrative track (as
    used by Task 1.4's plot); pass a calibrated value when a specific
    ground-target overpass is required (Task 1.5).

    Returns (timestamps_utc_iso, lat_deg, lon_deg) as three parallel
    numpy arrays, one row per propagation step.
    """
    from skyfield.api import load, EarthSatellite, wgs84

    if inclination_deg is None:
        inclination_deg = sun_sync_inclination(altitude_km)

    sat, epoch = _make_satrec(altitude_km, inclination_deg, raan_deg)
    ts = load.timescale(builtin=True)   # bundled leap-second data; no network needed
    n_steps = int(round(duration_hours * 3600.0 / step_s)) + 1
    seconds = np.arange(n_steps) * step_s
    t = ts.utc(*epoch[:5], seconds)

    earth_sat = EarthSatellite.from_satrec(sat, ts)
    subpoint = wgs84.geographic_position_of(earth_sat.at(t))

    return np.array(t.utc_iso()), subpoint.latitude.degrees, subpoint.longitude.degrees


def haversine_km(lat1_deg, lon1_deg, lat2_deg, lon2_deg):
    """Great-circle distance (km) between two lat/lon points, using the mean
    Earth radius already defined above (RE_EARTH) for consistency with the
    rest of this module -- broadcasts over array lat1/lon1 against scalar
    lat2/lon2. Public: also used by generate_overpass_schedule.py's RAAN
    calibration search."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1_deg, lon1_deg, lat2_deg, lon2_deg))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return (RE_EARTH / 1000.0) * c   # km


def passes_over_region(
    timestamps,
    lat_deg,
    lon_deg,
    target_lat_deg=TARGET_REGION_LAT_DEG,
    target_lon_deg=TARGET_REGION_LON_DEG,
    radius_km=TARGET_REGION_RADIUS_KM,
):
    """Find every contiguous span of a ground track (as returned by
    ground_track()) during which the sub-satellite point lies within
    radius_km of (target_lat_deg, target_lon_deg), via the Haversine
    great-circle distance.

    Consecutive in-range samples are grouped into a single pass. Returns a
    list of dicts, one per pass, in chronological order:
        {start, end, duration_s, n_samples, min_distance_km}

    This is a HARD JOIN: the returned passes (their timestamps -> ground-
    track footprints) are Team B's Task 3.3 resampling input.
    """
    lat_deg = np.asarray(lat_deg)
    lon_deg = np.asarray(lon_deg)
    distances_km = haversine_km(lat_deg, lon_deg, target_lat_deg, target_lon_deg)
    inside = distances_km <= radius_km

    passes = []
    i = 0
    n = len(inside)
    while i < n:
        if inside[i]:
            j = i
            while j + 1 < n and inside[j + 1]:
                j += 1
            start_dt = _parse_iso(timestamps[i])
            end_dt = _parse_iso(timestamps[j])
            passes.append({
                "start": str(timestamps[i]),
                "end": str(timestamps[j]),
                "duration_s": (end_dt - start_dt).total_seconds(),
                "n_samples": j - i + 1,
                "min_distance_km": float(distances_km[i:j + 1].min()),
            })
            i = j + 1
        else:
            i += 1
    return passes


def _parse_iso(timestamp):
    """Parses the 'Z'-suffixed UTC ISO-8601 strings ground_track() returns."""
    from datetime import datetime
    return datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))