"""
generate_overpass_schedule.py — Task 1.5: compute pass-over-target-region
timestamps for the locked-in mission (New Delhi, 28.6N/77.2E, radius 50 km)
and write the deliverable: data/overpass_schedule.txt

HARD JOIN: this schedule's footprints feed Team B's Task 3.3 resampling.

Note on RAAN: which longitudes a ground track passes over is set by the
orbit's RAAN at epoch (a real mission-planning parameter, normally fixed by
launch timing) -- an arbitrary RAAN will not generally put New Delhi under
the track within a single 24h window. This script calibrates RAAN (coarse
then fine search) so the locked-in target is actually covered, then reports
the pass schedule for that calibrated orbit.
"""

import os

import numpy as np

import orbital_model as om

ALTITUDE_KM = 500.0
DURATION_HOURS = 24.0
STEP_S = 60.0

OUT_PATH = os.path.join("data", "overpass_schedule.txt")


def find_calibrated_raan(altitude_km, inclination_deg, duration_hours, step_s,
                          target_lat, target_lon, radius_km):
    """Coarse-to-fine search for a RAAN producing at least one pass over the
    target region within the propagation window."""
    best_raan, best_dist = 0.0, float("inf")
    for raan in np.arange(0.0, 360.0, 5.0):
        _, lat, lon = om.ground_track(altitude_km, inclination_deg, duration_hours, step_s, raan_deg=raan)
        d = om.haversine_km(lat, lon, target_lat, target_lon).min()
        if d < best_dist:
            best_dist, best_raan = d, raan

    for raan in np.arange(best_raan - 5.0, best_raan + 5.0, 0.2):
        _, lat, lon = om.ground_track(altitude_km, inclination_deg, duration_hours, step_s, raan_deg=raan)
        d = om.haversine_km(lat, lon, target_lat, target_lon).min()
        if d < best_dist:
            best_dist, best_raan = d, raan

    return best_raan, best_dist


def main():
    inclination_deg = om.sun_sync_inclination(ALTITUDE_KM)

    raan_deg, closest_km = find_calibrated_raan(
        ALTITUDE_KM, inclination_deg, DURATION_HOURS, STEP_S,
        om.TARGET_REGION_LAT_DEG, om.TARGET_REGION_LON_DEG, om.TARGET_REGION_RADIUS_KM,
    )
    assert closest_km <= om.TARGET_REGION_RADIUS_KM, (
        f"calibration failed to find any pass within {om.TARGET_REGION_RADIUS_KM} km "
        f"(closest achievable: {closest_km:.1f} km) -- widen the RAAN search or the window"
    )

    timestamps, lat, lon = om.ground_track(
        ALTITUDE_KM, inclination_deg, DURATION_HOURS, STEP_S, raan_deg=raan_deg
    )
    passes = om.passes_over_region(
        timestamps, lat, lon,
        om.TARGET_REGION_LAT_DEG, om.TARGET_REGION_LON_DEG, om.TARGET_REGION_RADIUS_KM,
    )

    lines = [
        "Overpass schedule -- Task 1.5",
        f"Target region: lat={om.TARGET_REGION_LAT_DEG} deg, lon={om.TARGET_REGION_LON_DEG} deg "
        f"(New Delhi), radius={om.TARGET_REGION_RADIUS_KM} km",
        f"Orbit: altitude={ALTITUDE_KM} km, inclination={inclination_deg:.4f} deg, "
        f"RAAN={raan_deg:.2f} deg (calibrated so the target is covered), "
        f"window={DURATION_HOURS:.0f}h @ {STEP_S:.0f}s steps",
        f"Passes found: {len(passes)}",
        "",
    ]
    for k, p in enumerate(passes, start=1):
        lines.append(
            f"Pass {k}: start={p['start']}  end={p['end']}  "
            f"duration_s={p['duration_s']:.0f}  n_samples={p['n_samples']}  "
            f"min_distance_km={p['min_distance_km']:.2f}"
        )
    if any(p["n_samples"] == 1 for p in passes):
        lines += [
            "",
            "Note: a 1-sample (duration_s=0) pass is expected, not a bug -- at "
            f"~7 km/s ground-track speed, crossing a {om.TARGET_REGION_RADIUS_KM:.0f} km-radius "
            f"region takes roughly {2*om.TARGET_REGION_RADIUS_KM/7.0:.0f}s, which is shorter than "
            f"the {STEP_S:.0f}s sampling step, so most such crossings are only caught by a single sample.",
        ]

    os.makedirs("data", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nSaved {OUT_PATH}")

    assert 1 <= len(passes) <= 3, f"expected 1-3 passes/day, found {len(passes)}"
    print("Sanity check passed: 1-3 passes/day.")


if __name__ == "__main__":
    main()
