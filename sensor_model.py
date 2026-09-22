# sensor_model.py
import json
import math

#provisional
PROVISIONAL = True  # Set to False once physical payload procurement is locked
PROVISIONAL_TAG = "PROVISIONAL — pending final camera procurement"


def load_sensor_specs(specs_path: str = "sensor_specs.json") -> dict:
    """
    Loads camera physical parameters from the sensor specification file.

    # PROVISIONAL — pending final camera procurement
    """
    with open(specs_path, "r", encoding="utf-8") as f:
        specs = json.load(f)

    # Inject provisional metadata status cleanly for programmatic downstream checks
    specs.setdefault("is_provisional", PROVISIONAL)
    specs.setdefault("provisional_tag", PROVISIONAL_TAG)
    return specs


def gsd(
    pixel_size_um: float,
    altitude_km: float,
    focal_length_mm: float,
) -> float:
    """
    Computes Ground Sampling Distance (GSD) in meters/pixel.

    Formula:
        GSD = (pixel_size_m * altitude_m) / focal_length_m

    # PROVISIONAL — pending final camera procurement
    """
    pixel_size_m = pixel_size_um / 1e6
    altitude_m = altitude_km * 1e3
    focal_length_m = focal_length_mm / 1e3

    return (pixel_size_m * altitude_m) / focal_length_m


def swath(
    pixel_size_um: float,
    altitude_km: float,
    focal_length_mm: float,
    pixels_across_track: int,
) -> float:
    """
    Computes cross-track ground swath width in meters.

    Formula:
        Swath = GSD * pixels_across_track

    # PROVISIONAL — pending final camera procurement
    """
    calculated_gsd = gsd(
        pixel_size_um,
        altitude_km,
        focal_length_mm,
    )

    return calculated_gsd * pixels_across_track


def footprint(
    center_lat: float,
    center_lon: float,
    altitude_km: float,
    specs_path: str = "sensor_specs.json",
) -> dict:
    """
    Calculates ground-footprint bounding box (min/max lat/lon in degrees).

    Approximates bounding geometry from cross-track swath around 
    the sub-satellite point.

    # PROVISIONAL — pending final camera procurement
    """
    specs = load_sensor_specs(specs_path)

    swath_m = swath(
        specs["pixel_size_um"],
        altitude_km,
        specs["focal_length_mm"],
        specs["pixels_across_track"],
    )

    # Half swath extends symmetrically on both sides of sub-satellite point
    half_swath_km = (swath_m / 1000.0) / 2.0

    # 1 degree latitude ≈ 111 km
    lat_deg_offset = half_swath_km / 111.0

    # Longitude degree distance shrinks with cosine of latitude
    cos_lat = math.cos(math.radians(center_lat))

    if abs(cos_lat) > 1e-10:
        lon_deg_offset = half_swath_km / (111.0 * cos_lat)
    else:
        lon_deg_offset = float("inf")

    return {
        "min_lat": center_lat - lat_deg_offset,
        "max_lat": center_lat + lat_deg_offset,
        "min_lon": center_lon - lon_deg_offset,
        "max_lon": center_lon + lon_deg_offset,
        "is_provisional": PROVISIONAL,
        "status": PROVISIONAL_TAG,
    }


# ============================================================
# SANITY CHECK / VERIFICATION EXECUTION
# ============================================================

if __name__ == "__main__":
    specs = load_sensor_specs()

    # EXAMPLE Standard Phase 1 orbital benchmark altitude
    #Replace with Confirmed Phase 1's mission altitutde
    test_alt_km = 500.0

    calc_gsd = gsd(
        specs["pixel_size_um"],
        test_alt_km,
        specs["focal_length_mm"],
    )

    calc_swath = swath(
        specs["pixel_size_um"],
        test_alt_km,
        specs["focal_length_mm"],
        specs["pixels_across_track"],
    )

    print(f"--- Sensor Model Verification ({PROVISIONAL_TAG}) ---")
    print(f"Is Provisional : {PROVISIONAL}")
    print(f"Altitude       : {test_alt_km} km")
    print(f"GSD            : {calc_gsd:.2f} m/pixel")
    print(f"Swath Width    : {calc_swath / 1000.0:.2f} km")

    sample_fp = footprint(
        center_lat=13.53,
        center_lon=80.02,
        altitude_km=test_alt_km,
    )

    print(f"Footprint BBox : {sample_fp}")