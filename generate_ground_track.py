"""
generate_ground_track.py — Task 1.4: propagate the locked-in mission orbit
for 24 hours and produce the ground-track deliverables:
    data/ground_track.csv
    data/ground_track_plot.png
"""

import csv
import os

import matplotlib.pyplot as plt

import orbital_model as om

ALTITUDE_KM = 500.0
DURATION_HOURS = 24.0
STEP_S = 60.0

OUT_DIR = "data"
CSV_PATH = os.path.join(OUT_DIR, "ground_track.csv")
PLOT_PATH = os.path.join(OUT_DIR, "ground_track_plot.png")


def main():
    inclination_deg = om.sun_sync_inclination(ALTITUDE_KM)
    timestamps, lat, lon = om.ground_track(
        ALTITUDE_KM, inclination_deg, DURATION_HOURS, STEP_S
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "latitude", "longitude"])
        writer.writerows(zip(timestamps, lat, lon))

    period_min = om.orbital_period(ALTITUDE_KM) / 60.0
    orbits_per_day = 1440.0 / period_min

    plt.figure(figsize=(12, 6))
    plt.scatter(lon, lat, s=1.5, c="tab:blue")
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.xlabel("Longitude (deg)")
    plt.ylabel("Latitude (deg)")
    plt.title(
        f"Ground Track — {ALTITUDE_KM:.0f} km SSO, i={inclination_deg:.2f} deg, "
        f"24 h ({orbits_per_day:.1f} orbits/day)"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)

    print(f"Inclination used: {inclination_deg:.4f} deg")
    print(f"Orbital period: {period_min:.4f} min -> {orbits_per_day:.2f} orbits/day")
    print(f"Rows written: {len(timestamps)}")
    print(f"Saved {CSV_PATH}")
    print(f"Saved {PLOT_PATH}")

    # Sanity checks matching the issue's acceptance criteria.
    assert 14.0 <= orbits_per_day <= 15.5, f"orbits/day out of expected LEO range: {orbits_per_day}"
    span_hours = (len(timestamps) - 1) * STEP_S / 3600.0
    assert abs(span_hours - DURATION_HOURS) < 1e-6, "CSV does not span a full 24h"
    print("Sanity checks passed: ~14-15 orbits/day, full 24h span.")


if __name__ == "__main__":
    main()
