"""
orbital_model.py — Phase 1 orbital mechanics for the edgeai Digital Model.

Provides closed-form orbit parameters for a circular Low-Earth Orbit (LEO):
    - orbital_period(altitude_km)      : Keplerian period T = 2*pi*sqrt(a^3/mu)
    - sun_sync_inclination(altitude_km): J2 nodal-precession sun-synchronous inclination

All values follow the locked-in mission parameters (altitude 500 km, region
28.6 N / 77.2 E, radius 50 km). Units are SI for the internal math; inputs are
kilometres, outputs are seconds / degrees respectively.
"""

import numpy as np

# Locked-in mission altitude (km) — PROVISIONAL until final mission review.
TARGET_ALTITUDE_KM = 500.0

# Earth constants (SI units).
MU_EARTH = 398600e9      # standard gravitational parameter, m^3/s^2
RE_EARTH = 6371e3        # mean equatorial radius, m (nominal WGS-84 geoid)
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