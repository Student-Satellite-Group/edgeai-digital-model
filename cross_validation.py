"""
cross_validation.py — Task 1.3: Cross-validate orbital_model.py's orbital
period and sun-synchronous inclination against independent methods.

Three methods are used:
  1. Hand calculation — the same governing equations, re-typed independently
     (catches unit-conversion / typo bugs in the coded version).
  2. orbital_model.py — the actual module under test (Tasks 1.1/1.2).
  3. Independent numerical propagation via SGP4 (the `sgp4` library that
     skyfield itself depends on). SGP4 numerically propagates mean orbital
     elements with J2/J3/J4 secular terms — a genuinely different
     computational path from the closed-form J2-only algebra in
     orbital_model.py, so agreement here is real evidence the physics and
     the code are both correct, not just internally self-consistent.

GMAT is not installed in this environment (it is a desktop GUI application),
so it could not be run as part of this script. A ready-to-run GMAT script
(gmat_validation.script) is provided alongside this file for the team to
execute directly, which was the specification's originally-named third tool.
"""

import numpy as np
from scipy.optimize import brentq
from sgp4.api import Satrec, WGS84, jday

import orbital_model as om

ALTITUDE_KM = 500.0
TARGET_SUN_SYNC_DEG_PER_DAY = 360.0 / 365.25


# ---------------------------------------------------------------------------
# Method 1 — Hand calculation (independent re-derivation)
# ---------------------------------------------------------------------------
def hand_calculation(altitude_km):
    """T and i re-derived from first principles, typed out independently of
    orbital_model.py (separate constants, separate variable names) so a bug
    shared only by copy-paste wouldn't silently agree with itself."""
    mu = 398600.0            # km^3/s^2
    re = 6378.137             # km, WGS-84 equatorial radius
    j2 = 1.08263e-3
    sun_deg_per_day = 360.0 / 365.25

    a = re + altitude_km
    T_sec = 2.0 * np.pi * np.sqrt((a * 1000.0) ** 3 / (mu * 1e9))

    n_rad_s = 2.0 * np.pi / T_sec
    raan_dot_target = np.deg2rad(sun_deg_per_day) / 86400.0
    cos_i = -raan_dot_target / (1.5 * j2 * n_rad_s * (re / a) ** 2)
    cos_i = np.clip(cos_i, -1.0, 1.0)
    i_deg = np.degrees(np.arccos(cos_i))
    return T_sec, i_deg


# ---------------------------------------------------------------------------
# Method 3 — Independent SGP4 numerical propagation
# ---------------------------------------------------------------------------
MU_KM3_PER_MIN2 = 398600.8 * 3600.0   # WGS72 mu, s^2 -> min^2
RE_KM = 6378.135                       # WGS72 equatorial radius


def _make_satellite(altitude_km, inclination_deg):
    a_km = RE_KM + altitude_km
    n_rad_per_min = np.sqrt(MU_KM3_PER_MIN2 / a_km ** 3)
    sat = Satrec()
    jd, fr = jday(2026, 1, 1, 0, 0, 0)
    sat.sgp4init(
        WGS84, 'i', 99999, jd + fr - 2433281.5,
        0.0, 0.0, 0.0,             # bstar, ndot, nddot -- no drag
        0.0001,                     # eccentricity (near-zero; avoids SGP4 singularity)
        0.0,                         # argument of perigee, rad
        np.deg2rad(inclination_deg),
        0.0,                         # mean anomaly, rad
        n_rad_per_min,
        0.0,                         # RAAN, rad
    )
    return sat, jd, fr


def _propagate(sat, jd, fr, minutes):
    fr_arr = fr + np.asarray(minutes) / 1440.0
    jd_arr = np.full_like(fr_arr, jd)
    e, r, v = sat.sgp4_array(jd_arr, fr_arr)
    bad = np.unique(e[e != 0])
    assert bad.size == 0, f"SGP4 propagation error code(s): {bad}"
    return r, v


def sgp4_period_minutes(altitude_km, inclination_deg, guess_minutes):
    """Measures the nodal period directly from propagated positions: the
    satellite starts at an ascending node (z=0, rising) at t=0, so the next
    z=0-rising crossing near the Keplerian guess is the measured period."""
    sat, jd, fr = _make_satellite(altitude_km, inclination_deg)
    t = np.linspace(0.2 * guess_minutes, 1.3 * guess_minutes, 40000)
    r, _ = _propagate(sat, jd, fr, t)
    z = r[:, 2]
    crossings = np.where((z[:-1] < 0) & (z[1:] >= 0))[0]
    idx = crossings[np.argmin(np.abs(t[crossings] - guess_minutes))]
    t0, t1, z0, z1 = t[idx], t[idx + 1], z[idx], z[idx + 1]
    return t0 - z0 * (t1 - t0) / (z1 - z0)   # minutes


def sgp4_raan_rate_deg_per_day(altitude_km, inclination_deg, n_orbits, period_min):
    """Measures d(Omega)/dt directly from the drift of the orbital-plane
    node vector over n_orbits — not from any closed-form J2 formula."""
    sat, jd, fr = _make_satellite(altitude_km, inclination_deg)
    r0, v0 = _propagate(sat, jd, fr, np.array([0.0]))
    t_final = n_orbits * period_min
    r1, v1 = _propagate(sat, jd, fr, np.array([t_final]))
    h0 = np.cross(r0[0], v0[0])
    h1 = np.cross(r1[0], v1[0])
    node0 = np.cross([0.0, 0.0, 1.0], h0)
    node1 = np.cross([0.0, 0.0, 1.0], h1)
    raan0 = np.degrees(np.arctan2(node0[1], node0[0]))
    raan1 = np.degrees(np.arctan2(node1[1], node1[0]))
    d_raan = (raan1 - raan0 + 180.0) % 360.0 - 180.0   # wrap to [-180, 180)
    days = t_final / 1440.0
    return d_raan / days


def sgp4_sun_sync_inclination(altitude_km, period_min, n_orbits=15):
    def f(i_deg):
        return sgp4_raan_rate_deg_per_day(altitude_km, i_deg, n_orbits, period_min) - TARGET_SUN_SYNC_DEG_PER_DAY
    return brentq(f, 90.5, 104.9, xtol=1e-5)


# ---------------------------------------------------------------------------
# Run all three methods and report
# ---------------------------------------------------------------------------
def pct_diff(a, b):
    return abs(a - b) / a * 100.0


if __name__ == "__main__":
    T_hand_s, i_hand_deg = hand_calculation(ALTITUDE_KM)
    T_code_s = om.orbital_period(ALTITUDE_KM)
    i_code_deg = om.sun_sync_inclination(ALTITUDE_KM)

    T_sgp4_min = sgp4_period_minutes(ALTITUDE_KM, i_code_deg, guess_minutes=T_code_s / 60.0)
    T_sgp4_s = T_sgp4_min * 60.0
    i_sgp4_deg = sgp4_sun_sync_inclination(ALTITUDE_KM, T_sgp4_min)

    print(f"{'Method':<28}{'Period (min)':>15}{'Inclination (deg)':>20}")
    print(f"{'1. Hand calculation':<28}{T_hand_s/60:>15.4f}{i_hand_deg:>20.4f}")
    print(f"{'2. orbital_model.py':<28}{T_code_s/60:>15.4f}{i_code_deg:>20.4f}")
    print(f"{'3. SGP4 (independent)':<28}{T_sgp4_s/60:>15.4f}{i_sgp4_deg:>20.4f}")
    print()
    print("Pairwise period differences (%):")
    print(f"  hand vs code : {pct_diff(T_hand_s, T_code_s):.5f}%")
    print(f"  hand vs sgp4 : {pct_diff(T_hand_s, T_sgp4_s):.5f}%")
    print(f"  code vs sgp4 : {pct_diff(T_code_s, T_sgp4_s):.5f}%")
    print()
    print("Pairwise inclination differences (deg):")
    print(f"  hand vs code : {abs(i_hand_deg - i_code_deg):.5f} deg")
    print(f"  hand vs sgp4 : {abs(i_hand_deg - i_sgp4_deg):.5f} deg")
    print(f"  code vs sgp4 : {abs(i_code_deg - i_sgp4_deg):.5f} deg")
