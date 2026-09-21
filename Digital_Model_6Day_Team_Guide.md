# Digital Model — 6-Day, 3-Team Implementation Guide

**Source documents:** `Digital_Model_Implementation_Plan.md` + `Digital_Model_Parallelization_Analysis.md`
**Scope:** Complete Digital Model (self-contained simulation, zero hardware dependency). Per Kritzinger et al. (2018) taxonomy — this is a *Digital Model*, not a Digital Twin or Digital Shadow.
**Terminology note:** Every result in this plan is explicitly labeled as proxy/stand-in data until real ground-sensor data becomes available.
**Team structure:** Team A (Foundation & Sensor), Team B (Data & Algorithms), Team C (ML, Assembly & Validation)

---

## Task Dictionary — Every Task Defined

This section defines every numbered task in the plan so that each team member knows exactly what a task means before they start working on it.

---

### Phase 1 — Orbital Mechanics Module

**What this phase produces:** The foundation layer — orbital period, sun-synchronous inclination, ground track, and pass-over-target timestamps. This is 100% hardware-independent (pure physics). Everything downstream depends on this phase.

---

#### Task 1.1 — Implement Orbital Period

**What this task means:** Calculate how long the satellite takes to complete one orbit around Earth using the orbital mechanics formula. This is the most basic orbital parameter — everything else builds on it.

**Why it matters:** Without knowing the orbital period, you cannot determine when the satellite passes over any location. This value feeds into ground track generation, pass scheduling, and ultimately the imagery acquisition timeline.

**Formula:** $T = 2\pi\sqrt{a^3/\mu}$
- $T$ = orbital period (seconds)
- $a$ = semi-major axis = $R_e + \text{altitude}$ (meters)
- $R_e$ = Earth's mean radius = 6,371 km
- $\mu$ = Earth's gravitational parameter = 398,600 km³/s²
- altitude = chosen mission altitude (external shared decision — confirm with team lead before starting)

**Step-by-step sequence:**

1. **Confirm the target altitude** with the project lead. This is an external shared decision, not a calculation. Common choices: 400 km (LEO), 500 km (typical sun-synchronous), 600 km. Write down the chosen altitude value.
2. **Set up the calculation environment.** Open a new Python file `orbital_model.py`. Import NumPy: `import numpy as np`.
3. **Define constants in code:**
   ```python
   MU_EARTH = 398600e9  # m³/s² (converted from km³/s²)
   RE_EARTH = 6371e3    # m
   ```
4. **Write the `orbital_period(altitude)` function:**
   ```python
   def orbital_period(altitude_km):
       a = (RE_EARTH + altitude_km * 1000)  # semi-major axis in meters
       T = 2 * np.pi * np.sqrt(a**3 / MU_EARTH)
       return T  # returns period in seconds
   ```
5. **Convert to minutes for readability:** Divide the result by 60 to get orbital period in minutes.
6. **Test with a known value:** For 500 km altitude, $a$ = 6,871 km, $T$ ≈ 5,657 seconds ≈ **94.3 minutes**. If your output is close to 94 minutes, the implementation is correct.
7. **Test edge cases:** Try 400 km and 600 km altitudes. Period should increase with altitude (higher orbit = longer period).
8. **Document the function** with a docstring explaining the formula, inputs, and units.
9. **Deliverable:** A tested `orbital_period(altitude_km)` function that returns the orbital period in seconds, with verification against the known 500 km → ~94 min result.

**Tools needed:** Python, NumPy, calculator/spreadsheet for hand verification

**Expected output:** `orbital_period()` function returning period in seconds; verified against known values

---

#### Task 1.2 — Implement Sun-Synchronous Inclination Solver

**What this task means:** Calculate the orbital inclination required for a sun-synchronous orbit (SSO). A sun-synchronous orbit precesses at the same rate Earth orbits the Sun, so the satellite always passes over a given location at the same local solar time. This is critical for consistent lighting conditions in satellite imagery.

**Why it matters:** Sun-synchronous orbits are the standard for Earth observation satellites because they provide consistent illumination. Without this calculation, the imagery could have unpredictable lighting that ruins classification accuracy.

**Formula:** $\frac{d\Omega}{dt} = -\frac{3}{2} n J_2 \left(\frac{R_e}{a}\right)^2 \frac{\cos i}{(1-e^2)^2}$

Solving for inclination $i$:
$$i = \arccos\left(-\frac{\dot{\Omega} \cdot (1-e^2)^2 \cdot 2a^2}{3 \cdot n \cdot J_2 \cdot R_e^2}\right)$$

Where:
- $\frac{d\Omega}{dt}$ = nodal precession rate = 360°/365.25 days ≈ 0.9856°/day (required for sun-synchronicity)
- $n$ = mean motion = $2\pi / T$ (from Task 1.1's output)
- $J_2$ = Earth's second zonal harmonic = $1.0826 \times 10^{-3}$
- $R_e$ = Earth's mean radius = 6,371 km
- $a$ = semi-major axis (from Task 1.1)
- $e$ = orbital eccentricity (assume 0 for circular orbit)
- $i$ = inclination (the unknown we solve for)

**Step-by-step sequence:**

1. **Define the constants in `orbital_model.py`:**
   ```python
   J2_EARTH = 1.0826e-3
   DEG_PER_DAY = 360.0 / 365.25  # ≈ 0.9856 degrees/day
   ```
2. **Write the `sun_sync_inclination(altitude_km)` function:**
   ```python
   def sun_sync_inclination(altitude_km):
       a = RE_EARTH + altitude_km * 1000  # meters
       T = orbital_period(altitude_km)      # seconds (reuse Task 1.1 function)
       n = 2 * np.pi / T                     # mean motion (rad/s)
       dOmega_dt = np.radians(DEG_PER_DAY)   # convert to rad/s
       # Solve for inclination
       cos_i = -(dOmega_dt * (1 - 0**2)**2 * 2 * a**2) / (3 * n * J2_EARTH * RE_EARTH**2)
       i_rad = np.arccos(np.clip(cos_i, -1, 1))  # clip to valid range
       i_deg = np.degrees(i_rad)
       return i_deg
   ```
3. **Use `np.clip` to handle floating-point edge cases** where `cos_i` might be slightly outside [-1, 1].
4. **Test with 500 km altitude:** Expected inclination is approximately **97°–99°** for sun-synchronous orbits. Verify your output falls in this range.
5. **Test with multiple altitudes:** 400 km, 500 km, 600 km. Inclination should decrease as altitude increases (higher orbits need less inclination to maintain sun-synchronicity).
6. **Verify using an independent method:** Use the `poliastro` library to compute the sun-synchronous inclination and compare:
   ```python
   from poliastro.twobody import Orbit
   # Compare your result with poliastro's computed inclination
   ```
7. **Document the function** with a docstring explaining the J2 precession equation and inputs.
8. **Deliverable:** A tested `sun_sync_inclination(altitude_km)` function returning inclination in degrees, verified to produce 97°–99° for typical LEO altitudes.

**Tools needed:** Python, NumPy, `poliastro` or `skyfield` library for cross-validation

**Expected output:** `sun_sync_inclination()` function returning inclination in degrees; verified to produce sun-synchronous values

---

#### Task 1.3 — Cross-Validate Orbital Parameters

**What this task means:** Verify that your orbital calculations (from Tasks 1.1 and 1.2) are correct by computing the same values using three completely independent methods: (1) hand calculation with a calculator/spreadsheet, (2) GMAT or similar orbital mechanics tool, (3) a Python library (`skyfield` or `poliastro`).

**Why it matters:** If all three methods agree, you can be confident the orbital model is correct. If they disagree, one of them has a bug or wrong input. This triple-validation is the quality gate before anything downstream uses these numbers.

**Acceptance criteria:** All three methods must agree within **0.5% on orbital period** and **a fraction of a degree on inclination**.

**Step-by-step sequence:**

1. **Hand calculation (Method 1):**
   - Pick a test altitude (e.g., 500 km).
   - Compute $a = R_e + \text{altitude}$ on paper or in a spreadsheet.
   - Compute $T = 2\pi\sqrt{a^3/\mu}$ using a calculator.
   - Compute inclination using the J2 equation on paper.
   - Write down both results with full working shown.

2. **GMV or equivalent (Method 2):**
   - Open GMAT (or STK, Orekit, etc.).
   - Create a spacecraft with the chosen altitude.
   - Propagate for one orbit.
   - Read the orbital period from GMAT's output.
   - Read the inclination from GMAT's output.
   - Write down both results.

3. **Python library (Method 3):**
   - Install `poliastro` or `skyfield`: `pip install poliastro`
   - Create an orbit with the chosen altitude and sun-synchronous inclination:
     ```python
     from poliastro.twobody import Orbit
     from poliastro.bodies import Earth
     # Create orbit and extract period and inclination
     ```
   - Write down both results.

4. **Compare all three results:**
   - Create a comparison table in `cross_validation_log.md`:
     | Method | Period (seconds) | Inclination (degrees) |
     |--------|-----------------|----------------------|
     | Hand calculation | ... | ... |
     | GMAT | ... | ... |
     | poliastro/skyfield | ... | ... |
   - Compute percentage differences between each pair.
   - Verify all differences are < 0.5% for period and < 1° for inclination.

5. **If results disagree:**
   - Check units (are you mixing km and m?).
   - Check altitude values (are all three using the same altitude?).
   - Check time conversions (is GMAT using seconds or days?).
   - Fix discrepancies and re-run.

6. **Deliverable:** `cross_validation_log.md` — a written three-way cross-validation record showing all three method outputs, the computed deltas, and confirmation that agreement is within acceptance criteria.

**Tools needed:** Calculator/spreadsheet, GMAT (or equivalent), `poliastro` or `skyfield` Python library

**Expected output:** `cross_validation_log.md` confirming all three methods agree within 0.5% (period) and sub-degree (inclination)

---

#### Task 1.4 — Generate Ground Track

**What this task means:** Propagate the satellite's orbit over a 24-hour period and compute the ground track — the path of the satellite's sub-point (the point on Earth directly below the satellite) projected onto a map. This produces latitude/longitude coordinates over time.

**Why it matters:** The ground track tells you exactly where the satellite flies over the Earth's surface. This is the foundation for determining which areas the satellite can image and when. Without the ground track, you cannot plan passes over target regions.

**Step-by-step sequence:**

1. **Choose propagation parameters:**
   - Altitude: the mission altitude from Phase 1 (e.g., 500 km).
   - Inclination: the sun-synchronous inclination from Task 1.2.
   - Window: 24 hours (86,400 seconds).
   - Time step: 60 seconds (or finer for smoother tracks).

2. **Propagate the orbit:**
   - Use `poliastro` or `skyfield` to propagate the orbit for 24 hours:
     ```python
     from poliastro.twobody import Orbit
     from poliastro.bodies import Earth
     from astropy.time import Time
     # Create orbit with chosen altitude and inclination
     # Propagate for 24 hours
     # Extract position vectors (ECI frame)
     ```
   - Alternatively, use a simplified numerical propagation:
     ```python
     def propagate_orbit(altitude_km, inclination_deg, duration_hours=24, dt_seconds=60):
         # Generate position vectors over time
         # Convert ECI to ECEF (Earth-Centered, Earth-Fixed)
         # Convert ECEF to lat/lon
         pass
     ```

3. **Convert ECI to geodetic coordinates:**
   - Convert the satellite's position from Earth-Centered Inertial (ECI) frame to Earth-Centered, Earth-Fixed (ECEF) frame.
   - Convert ECEF coordinates to latitude, longitude, and altitude.
   - Only the latitude and longitude matter for the ground track.

4. **Generate the ground track data:**
   - Create arrays of latitude and longitude values at each time step.
   - Save to `data/ground_track.csv` with columns: `timestamp, latitude, longitude, altitude_km`.
   - Generate a visualization plot using matplotlib:
     ```python
     import matplotlib.pyplot as plt
     plt.figure(figsize=(12, 6))
     plt.plot(longitudes, latitudes, 'b-', linewidth=0.5)
     plt.xlabel('Longitude')
     plt.ylabel('Latitude')
     plt.title('Ground Track — 24-Hour Propagation')
     plt.grid(True)
     plt.savefig('data/ground_track_plot.png', dpi=150)
     ```

5. **Verify the ground track:**
   - For a 500 km sun-synchronous orbit, expect approximately 14–15 orbits per day.
   - Ground track should cover most of the Earth's latitudes between approximately ±98°.
   - The track should show the characteristic westward drift per orbit (about 22.5° per orbit for LEO SSO).

6. **Deliverable:** `data/ground_track.csv` (timestamp, latitude, longitude columns) and `data/ground_track_plot.png` (visualization).

**Tools needed:** `poliastro` or `skyfield`, matplotlib, NumPy

**Expected output:** `data/ground_track.csv` and `data/ground_track_plot.png` showing the 24-hour ground track

---

#### Task 1.5 — Implement Pass-Over-Target-Region Query

**What this task means:** Given the ground track from Task 1.4 and a target lat/lon region (the area of interest for the mission), compute the exact timestamps when the satellite passes over that region.

**Why it matters:** This tells you when the satellite will image your target area. These timestamps define the pass schedule, which drives the entire imagery acquisition pipeline (Phase 3). Without this, you cannot know when to expect data over your region of interest.

**Step-by-step sequence:**

1. **Define the target region:**
   - Choose the target lat/lon region (external shared decision — confirm with team lead).
   - Define it as: `target_lat`, `target_lon`, `radius_km` (the radius around the target point).
   - Example: target_lat = 28.6°N, target_lon = 77.2°E, radius_km = 50 km (for a region around New Delhi).

2. **Implement the `passes_over_region()` function:**
   ```python
   def passes_over_region(ground_track_csv, target_lat, target_lon, radius_km):
       import pandas as pd
       from math import radians, cos, sin, asin, sqrt
       
       # Load ground track data
       df = pd.read_csv(ground_track_csv)
       
       # Haversine distance function
       def haversine(lat1, lon1, lat2, lon2):
           R = 6371  # Earth radius in km
           lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
           dlat = lat2 - lat1
           dlon = lon2 - lon1
           a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
           c = 2 * asin(sqrt(a))
           return R * c
       
       # Find all points within the radius
       passes = []
       for _, row in df.iterrows():
           dist = haversine(target_lat, target_lon, row['latitude'], row['longitude'])
           if dist <= radius_km:
               passes.append(row['timestamp'])
       
       return passes
   ```

3. **Group consecutive timestamps into passes:**
   - Consecutive timestamps within a few minutes belong to the same pass.
   - Group them and compute pass start/end times and duration.
   - This prevents listing every single data point as a separate "pass."

4. **Test with the actual target region:**
   - Run the function with the confirmed target coordinates.
   - Verify the number of daily passes is reasonable (typically 1–3 passes per day per region for LEO).
   - Verify the timestamps are spread across the 24-hour window (not clustered).

5. **Verify the output format:**
   - Return a list of dictionaries or a structured format:
     ```python
     [{'date': '2026-01-15', 'start_time': '06:23:12', 'end_time': '06:27:45', 'max_elevation': 42.3}, ...]
     ```

6. **Deliverable:** Working `passes_over_region()` function + a text file `data/overpass_schedule.txt` listing all overpass timestamps for the target region over a representative period (e.g., one week).

**Tools needed:** Python, pandas, NumPy, Haversine formula implementation

**Expected output:** `passes_over_region()` function + `data/overpass_schedule.txt` with all overpass timestamps

---

#### Task 2.1 — Select Candidate Camera & Pull Datasheet

**What this task means:** Choose a realistic RGB camera module that could fly on the satellite, and pull its published specifications (pixel size and focal length) from its actual datasheet. You must use a real, findable spec sheet — never invent camera numbers.

**Why it matters:** The camera specs determine the ground sampling distance (GSD) — how big each pixel represents on the ground. This is the single most important sensor parameter for determining whether the imagery is suitable for the classification task. Using fake camera specs would make all downstream results meaningless.

**Step-by-step sequence:**

1. **Research candidate camera modules:**
   - Search for realistic satellite camera modules (e.g., SX130 from iXblue, SpaceCam from Point Research, or commercial-off-the-shelf variants).
   - Look for modules with published datasheets that include pixel size and focal length.
   - Good sources: manufacturer websites, NASA/USGS procurement catalogs, published satellite constellations.

2. **Download the datasheet:**
   - Find the candidate camera's datasheet PDF or web page.
   - Save it as `data/camera_datasheets/[camera_name]_datasheet.pdf`.
   - Also save a web page screenshot or HTML dump as backup.

3. **Extract the key specifications:**
   - **Pixel size** (in micrometers, µm): the physical size of each photosensor pixel.
   - **Focal length** (in millimeters, mm): the optical focal length of the lens.
   - **Resolution** (pixels across track): the number of pixels in the across-track direction.
   - Record all three values precisely as stated in the datasheet.

4. **Create `sensor_specs.json`:**
   ```json
   {
     "camera_model": "SX-130 (example)",
     "pixel_size_um": 5.0,
     "focal_length_mm": 100.0,
     "pixels_across_track": 4096,
     "datasheet_source_url": "https://example.com/sx130-datasheet.pdf",
     "datasheet_file": "data/camera_datasheets/SX-130_datasheet.pdf",
     "date_retrieved": "2026-09-21"
   }
   ```
   - Replace with actual values from the datasheet you found.
   - Every value must be traceable to the datasheet.

5. **Verify the specs are realistic:**
   - Pixel size should be in the range 1–20 µm for satellite cameras.
   - Focal length should be in the range 50–500 mm for typical LEO imaging.
   - If values fall outside these ranges, double-check the datasheet.

6. **Deliverable:** `sensor_specs.json` with camera model, pixel size, focal length, pixels across track, and source URL. Also `data/camera_datasheets/[camera_name]_datasheet.pdf`.

**Tools needed:** Web browser, PDF reader, text editor

**Expected output:** `sensor_specs.json` with all camera parameters and source citations + downloaded datasheet PDF

---

#### Task 2.2 — Compute GSD and Swath

**What this task means:** Using the camera specifications from Task 2.1 and the mission altitude from Task 1.1, compute two critical parameters:
- **GSD (Ground Sampling Distance):** the physical size on the ground that one pixel represents (e.g., 10 meters per pixel). This determines the spatial resolution of the imagery.
- **Swath:** the width of the imaging footprint on the ground (e.g., 20 km wide). This determines how much area the satellite can image in a single pass.

**Why it matters:** GSD determines whether you can distinguish features of interest (e.g., can you see individual buildings at 10m GSD vs. only large fields at 100m GSD?). Swath determines how much area you cover per orbit. Both are essential for determining mission feasibility.

**Formulas:**
- $\text{GSD} = \dfrac{\text{pixel\_size} \times \text{altitude}}{\text{focal\_length}}$
- $\text{Swath} = \text{GSD} \times \text{pixels\_across\_track}$

**Step-by-step sequence:**

1. **Load the camera specs:**
   ```python
   import json
   with open('sensor_specs.json') as f:
       specs = json.load(f)
   pixel_size_um = specs['pixel_size_um']      # e.g., 5.0 µm
   focal_length_mm = specs['focal_length_mm']   # e.g., 100.0 mm
   pixels_across_track = specs['pixels_across_track']  # e.g., 4096
   ```

2. **Convert units consistently:**
   - Pixel size: convert from micrometers to meters: `pixel_size_m = pixel_size_um / 1e6`
   - Focal length: convert from mm to meters: `focal_length_m = focal_length_mm / 1000`
   - Altitude: in meters from Task 1.1

3. **Compute GSD:**
   ```python
   def gsd(pixel_size_um, altitude_km, focal_length_mm):
       pixel_size_m = pixel_size_um / 1e6
       focal_length_m = focal_length_mm / 1000
       altitude_m = altitude_km * 1000
       return (pixel_size_m * altitude_m) / focal_length_m  # in meters
   ```
   - Example: 5 µm pixel, 500 km altitude, 100 mm focal length → GSD = (5e-6 × 500000) / 0.1 = **25 meters per pixel**

4. **Compute Swath:**
   ```python
   def swath(pixel_size_um, altitude_km, focal_length_mm, pixels_across_track):
       g = gsd(pixel_size_um, altitude_km, focal_length_mm)
       return g * pixels_across_track  # in meters
   ```
   - Example: 25 m GSD × 4096 pixels = **102.4 km swath**

5. **Record the values as PROVISIONAL constants:**
   ```python
   # In sensor_model.py
   PROVISIONAL_GSD_M = gsd(...)  # e.g., 25.0
   PROVISIONAL_SWATH_M = swath(...)  # e.g., 102400.0
   PROVISIONAL_LABEL = "PROVISIONAL — pending final camera procurement"
   ```

6. **Add `PROVISIONAL` tags everywhere these values are used:**
   - In code: add `# PROVISIONAL` comments to every constant and configuration flag.
   - In documentation: add `PROVISIONAL` labels to every mention of GSD/swath.
   - Create a `PROVISIONAL_LABELS.md` listing every instance.

7. **Document the preliminary go/no-go judgment:**
   - Is the GSD fine enough for the classification task?
   - Example: 25 m GSD is adequate for cloud detection but may be marginal for small-object detection.
   - Write the reasoning in `sensor_go_nogo.md`.

8. **Deliverable:** Working `gsd()` and `swath()` functions in `sensor_model.py`, `PROVISIONAL` tags everywhere, `sensor_go_nogo.md`, `PROVISIONAL_LABELS.md`.

**Tools needed:** Python, `sensor_specs.json`, calculator/spreadsheet for verification

**Expected output:** `sensor_model.py` with `gsd()`, `swath()`, `footprint()` functions (all PROVISIONAL-tagged), `sensor_go_nogo.md`, `PROVISIONAL_LABELS.md`

---

#### Task 2.3 — Tag PROVISIONAL Everywhere

**What this task means:** Add an explicit `PROVISIONAL — pending final camera procurement` label to every place in code and documentation where the GSD/swath values from Phase 2 are used. This ensures no one accidentally treats these provisional values as final.

**Why it matters:** Since the camera hasn't been procured yet, all sensor-derived values are provisional. If someone uses these values to make decisions without knowing they're provisional, they could design a system that fails when the real camera arrives. The PROVISIONAL tag is a safety mechanism.

**Step-by-step sequence:**

1. **Scan all code files for GSD/swath references:**
   - `orbital_model.py`
   - `sensor_model.py`
   - `resample_pipeline.py`
   - Any file that uses altitude or GSD values

2. **Add PROVISIONAL comments to constants:**
   ```python
   # PROVISIONAL — pending final camera procurement
   PROVISIONAL_GSD_M = 25.0
   # PROVISIONAL — pending final camera procurement
   PROVISIONAL_SWATH_M = 102400.0
   ```

3. **Add PROVISIONAL config flags:**
   ```python
   PROVISIONAL = True  # Set to False when real camera datasheet is available
   ```

4. **Add PROVISIONAL labels to documentation:**
   - Every mention of GSD, swath, or altitude in any `.md` file should have `(PROVISIONAL)` appended.
   - Example: "GSD = 25 m (PROVISIONAL)"

5. **Create `PROVISIONAL_LABELS.md`:**
   - List every file and line where PROVISIONAL labels appear.
   - Include the current provisional value and what it should be replaced with when hardware arrives.

6. **Deliverable:** All code and documentation files updated with PROVISIONAL tags; `PROVISIONAL_LABELS.md` created.

**Tools needed:** Text editor, grep/search tool

**Expected output:** All GSD/swath references tagged with PROVISIONAL; `PROVISIONAL_LABELS.md` listing every instance

---

#### Task 2.4 — Preliminary Go/No-Go Judgment

**What this task means:** Assess whether the candidate camera's computed GSD is adequate for the intended classification task (e.g., cloud detection, thermal anomaly detection). This is a preliminary judgment — it will be re-confirmed when the real camera is procured.

**Why it matters:** If the GSD is too coarse, the imagery won't have enough spatial detail to distinguish the features you're trying to classify. Catching this early saves months of wasted work on an inadequate sensor configuration.

**Step-by-step sequence:**

1. **Define the classification task requirements:**
   - What is the smallest feature you need to detect?
   - Example: Cloud detection → need to see cloud patterns → GSD < 50 m is adequate. Thermal anomaly detection → need to see small heat sources → GSD < 30 m might be needed.

2. **Compare the computed GSD to task requirements:**
   - If GSD ≤ smallest feature size → **GO** (adequate)
   - If GSD > smallest feature size → **NO-GO** (inadequate, need a different camera or higher altitude)

3. **Document the reasoning:**
   - Write `sensor_go_nogo.md` with:
     - The candidate camera model and its GSD
     - The classification task and its spatial resolution requirement
     - The comparison result (GO or NO-GO)
     - The reasoning behind the judgment
     - A clear caveat: "This judgment is preliminary and based on PROVISIONAL GSD values. It will be re-confirmed when the real camera is procured."

4. **Consider alternatives if NO-GO:**
   - Lower altitude → smaller GSD (but smaller swath)
   - Different camera with smaller pixel size → smaller GSD
   - Longer focal length → smaller GSD (but heavier/bulkier)
   - Document these alternatives in the file.

5. **Deliverable:** `sensor_go_nogo.md` with preliminary go/no-go judgment and reasoning, clearly labeled as PROVISIONAL.

**Tools needed:** Classification task definition, computed GSD from Task 2.2

**Expected output:** `sensor_go_nogo.md` with GO/NO-GO judgment and reasoning (PROVISIONAL-caveated)

---

### Phase 3 — Proxy Imagery Acquisition & Resampling Pipeline

**What this phase produces:** A labeled, orbit-resampled proxy imagery dataset (100+ samples) with a repeatable pipeline. All imagery comes from public satellite data (Sentinel-2, Landsat) — no payload hardware involved.

---

#### Task 3.1 — Register Data-Access Accounts

**What this task means:** Create accounts on the two primary public satellite data platforms needed for this project: Copernicus Open Access Hub (for Sentinel-2 optical data) and USGS EarthExplorer (for Landsat thermal data).

**Step-by-step sequence:**

1. **Register for Copernicus Open Access Hub:**
   - Go to https://scihub.copernicus.eu/dhus/
   - Create an account with your email and a secure password.
   - Verify your email address.
   - Log in and navigate to the search interface.
   - Verify you can search for Sentinel-2 images.

2. **Register for USGS EarthExplorer:**
   - Go to https://earthexplorer.usgs.gov/
   - Create an account.
   - Verify your email address.
   - Log in and navigate to the search interface.
   - Verify you can search for Landsat 8/9 thermal band data.

3. **Save credentials securely:**
   - Create a `.env` file in the project root:
     ```
     COPERNICUS_USERNAME=your_username
     COPERNICUS_PASSWORD=your_password
     EARTHEXPLORER_USERNAME=your_username
     EARTHEXPLORER_PASSWORD=your_password
     ```
   - Add `.env` to `.gitignore` so credentials are never committed to version control.

4. **Document the registration process:**
   - Write `data_access.md` with:
     - Platform name and URL
     - Account credentials (referencing `.env` — never put them in the document directly)
     - Verification steps taken
     - Any issues encountered and how they were resolved

5. **Test data access:**
   - Using the Copernicus API, search for a Sentinel-2 image over the target region with less than 20% cloud cover.
   - Using the EarthExplorer API, search for a Landsat thermal image over the target region.
   - Verify both searches return results.

6. **Deliverable:** Working accounts on both platforms, `.env` file with credentials (in `.gitignore`), `data_access.md` documenting the registration and verification process.

**Tools needed:** Web browser, text editor

**Expected output:** Working Copernicus and USGS EarthExplorer accounts; `.env` file; `data_access.md`

---

#### Task 3.2 — Download Broad Regional Imagery Tiles

**What this task means:** Download Sentinel-2 optical and Landsat thermal-band tiles covering the target region from both platforms. Download a range of scene conditions (varied cloud cover, varied thermal backgrounds) to ensure the dataset is diverse enough for the classification task.

**Step-by-step sequence:**

1. **Define the search parameters:**
   - Target region coordinates (from Task 1.5).
   - Date range: last 1–2 years of available data.
   - Cloud cover filter: vary from 0% to 50% to get diverse scenes.
   - For Landsat: thermal band (Band 10 or 11).

2. **Search and download Sentinel-2 data:**
   - Use the Copernicus SciHub API or the web interface:
     ```python
     import requests
     # Search for Sentinel-2 L2A products over the target region
     # Download the top N scenes (aim for 10+)
     ```
   - Download the SAFE format products.
   - Organize into `data/raw/sentinel2/` directory.

3. **Search and download Landsat data:**
   - Use the EarthExplorer API or web interface:
     ```python
     # Search for Landsat 8/9 Collection 2 Level 2 products
     # Focus on thermal band (Band 10)
     # Download the top N scenes
     ```
   - Organize into `data/raw/landsat/` directory.

4. **Create a raw download manifest:**
   - Create `data/manifest_raw.csv` with columns:
     - `scene_id`: unique identifier for each scene
     - `source`: "sentinel2" or "landsat"
     - `platform`: "Sentinel-2" or "Landsat-8/9"
     - `acquisition_date`: date of image acquisition
     - `cloud_cover`: percentage
     - `path`: local file path to the downloaded file
     - `thermal_band_path`: path to thermal band file (for Landsat)
     - `download_date`: when you downloaded it

5. **Verify downloads:**
   - Check that all downloaded files are complete (not corrupted).
   - Verify file sizes are reasonable (Sentinel-2 SAFE products are typically 100–500 MB each).
   - Open at least one scene to confirm it contains the expected bands.

6. **Deliverable:** Raw tile collection in `data/raw/sentinel2/` and `data/raw/landsat/`; `data/manifest_raw.csv` with complete download metadata.

**Tools needed:** Copernicus SciHub API, USGS EarthExplorer API, Python `requests` library, storage space

**Expected output:** Raw satellite imagery tiles organized in directories; `data/manifest_raw.csv`

---

#### Task 3.3 — Build Crop/Reproject/Resample Script

**What this task means:** Build the core data processing script that takes raw satellite tiles and converts them into orbit-representative image pairs at the correct resolution. This is the **first hard join point** — it needs two inputs from Team A: (1) pass footprints from Task 1.5 and (2) the provisional GSD from Task 2.2.

**Step-by-step sequence:**

1. **Get inputs from Team A:**
   - `passes_over_region()` output from Task 1.5 → gives you the pass footprints (lat/lon polygons).
   - `gsd()` value from Task 2.2 → gives you the target resolution (e.g., 25 m/pixel).

2. **Set up the resampling pipeline:**
   ```python
   import rasterio
   from rasterio.warp import reproject, Resampling
   import geopandas as gp
   
   def resample_to_footprint(raw_tile_path, footprint_geojson, target_gsd_m, output_path):
       """
       Crop raw tile to footprint, reproject to target CRS, resample to target GSD.
       """
       with rasterio.open(raw_tile_path) as src:
           # 1. Crop to footprint
           # 2. Reproject to target CRS (e.g., EPSG:4326 or UTM zone)
           # 3. Resample to target GSD
           # 4. Write output to output_path
           pass
   ```

3. **Implement cropping to footprint:**
   - Use the GeoJSON polygon from the pass footprint.
   - Clip the raster to the polygon bounds using `rasterio.mask.mask()`.

4. **Implement reprojection:**
   - Reproject from the source CRS to the target CRS (usually EPSG:4326 for global coverage or a UTM zone for local coverage).
   - Use `rasterio.warp.reproject()` with the appropriate destination CRS.

5. **Implement resampling to target GSD:**
   - Calculate the target resolution in degrees or meters based on the GSD value.
   - Set the output resolution: `transform = rasterio.transform.from_origin(..., width=..., height=..., resolution=(target_gsd_deg, target_gsd_deg))`
   - Use `Resampling.bilinear` for optical imagery; `Resampling.nearest` for thermal bands.

6. **Process each pass:**
   - For each pass in the overpass schedule:
     - Find the corresponding Sentinel-2 optical tile and Landsat thermal tile.
     - Crop both to the pass footprint.
     - Resample both to the provisional GSD.
     - Save the resampled pair as `data/resampled/[pass_id]_optical.tif` and `[pass_id]_thermal.tif`.

7. **Test the pipeline:**
   - Run it on one pass first.
   - Verify: output dimensions match the expected GSD, both optical and thermal tiles cover the same footprint, no corrupted output.

8. **Deliverable:** Working `resample_pipeline.py` with `resample_to_footprint()` function.

**Tools needed:** `rasterio`, GDAL, GeoPandas, Team A's pass footprints and GSD values

**Expected output:** `resample_pipeline.py` — working crop/reproject/resample script

---

#### Task 3.4 — Package Dataset + Manifest

**What this task means:** Organize all resampled image pairs into a structured dataset directory with consistent naming, and create a manifest CSV that ties every sample to its source pass metadata.

**Step-by-step sequence:**

1. **Define the dataset directory structure:**
   ```
   data/labeled/
   ├── manifest.csv
   ├── pass_001/
   │   ├── optical.tif
   │   └── thermal.tif
   ├── pass_002/
   │   ├── optical.tif
   │   └── thermal.tif
   └── ...
   ```

2. **Create the manifest CSV:**
   ```csv
   sample_id,pass_id,date,optical_path,thermal_path,footprint_geojson,gsd_m,label
   pass_001,2026-01-15T06:23:12Z,data/labeled/pass_001/optical.tif,data/labeled/pass_001/thermal.tif,"{...}",25.0,cloud
   ```
   - Columns: `sample_id`, `pass_id`, `date`, `optical_path`, `thermal_path`, `footprint_geojson`, `gsd_m`, `label`

3. **Organize all resampled pairs into the directory structure:**
   - Each pass gets its own subdirectory.
   - Optical and thermal TIFFs are stored together.
   - File names are consistent and descriptive.

4. **Verify the dataset:**
   - Count total samples: target is 100+.
   - Check that every sample has both optical and thermal components.
   - Verify that every manifest row has valid file paths.

5. **Deliverable:** Complete labeled dataset in `data/labeled/` with `manifest.csv`.

**Tools needed:** Python, `pandas` for manifest creation, `os` for directory management

**Expected output:** `data/labeled/manifest.csv` + organized directory structure with 100+ image pairs

---

#### Task 3.5 — Label the Dataset

**What this task means:** Apply classification labels to every sample in the dataset. The label indicates whether the scene contains the feature of interest (e.g., cloud/no-cloud, thermal-anomaly/no-anomaly).

**Step-by-step sequence:**

1. **Define the labeling scheme:**
   - For cloud detection: label = `cloud` or `no-cloud`
   - For thermal anomaly: label = `anomaly` or `no-anomaly`
   - Use binary labels for simplicity.

2. **Determine labels for each sample:**
   - **Cloud detection:** Use Sentinel-2 cloud masks (available from Copernicus as part of the Scene Classification Layer — SCL band).
     ```python
     # Read the SCL band from Sentinel-2
     # SCL values: 0=dark, 1=cloud, 2=shadow, 3=veil, 4=snow, 5=clear, etc.
     # If any pixel in the footprint has SCL=1 (cloud), label = 'cloud'
     ```
   - **Thermal anomaly:** Use known hotspots from public databases (e.g., Global Volcanism Program, NASA MODIS active fire database) or Landsat thermal band thresholding.

3. **Apply labels to each sample:**
   - For each row in `manifest.csv`, determine the label based on the method above.
   - Update the `label` column in the manifest.

4. **Handle edge cases:**
   - Partial cloud cover: If < 30% of the footprint is covered, label as `no-cloud`. If ≥ 30%, label as `cloud`.
   - Ambiguous cases: Document the reasoning for any borderline cases.

5. **Deliverable:** Updated `manifest.csv` with all labels applied; `labeling_method.md` describing how labels were determined.

**Tools needed:** Sentinel-2 SCL band data, Python, public hotspot databases

**Expected output:** `manifest.csv` with all samples labeled; `labeling_method.md`

---

#### Task 3.6 — Quality Check the Dataset

**What this task means:** Verify the dataset integrity — no blank or corrupted tiles, correct footprint alignment between optical and thermal tiles of each pair, and resolution matching the Phase 2 GSD (not the source imagery's native resolution).

**Step-by-step sequence:**

1. **Check for blank/corrupted tiles:**
   ```python
   import rasterio
   for sample in manifest['sample_id']:
       try:
           with rasterio.open(optical_path) as src:
               data = src.read()
               assert data.size > 0, "Empty tile"
               assert not np.all(np.isnan(data)), "All-NaN tile"
       except Exception as e:
           print(f"FAILED: {sample} — {e}")
   ```

2. **Check footprint alignment:**
   - For each pair, verify that the optical and thermal TIFFs cover the same geographic area.
   - Compare the bounding boxes: `optical.bounds` should approximately equal `thermal.bounds`.
   - Allow a tolerance of a few meters (due to resampling).

3. **Check resolution:**
   - Read the transform of each TIFF: `src.transform`.
   - Verify the pixel size matches the Phase 2 GSD value (e.g., 25 m).
   - If the resolution doesn't match, the resampling pipeline has a bug.

4. **Document findings:**
   - Create `qc_report.md` with:
     - Total samples checked
     - Number passed/failed
     - Any issues found and how they were resolved
     - Confirmation that the dataset is ready for downstream use

5. **Deliverable:** `qc_report.md` confirming dataset integrity.

**Tools needed:** `rasterio`, Python, NumPy

**Expected output:** `qc_report.md` — dataset quality-check report confirming integrity

---

### Phase 4 — Registration & Fusion Algorithm Development

**What this phase produces:** Working cross-modal image registration and fusion algorithms, validated on the proxy dataset. All algorithms are developed and tuned against stand-in data (Sentinel-2 optical + Landsat thermal) with explicit proxy-data caveats.

---

#### Task 4.1 — Implement Registration Pipeline (Code)

**What this task means:** Write the code for the image registration pipeline that aligns optical and thermal images. This is the code implementation only — the tuning against real proxy data happens in Task 4.2. Use placeholder or synthetic image pairs for initial development.

**Step-by-step sequence:**

1. **Set up the registration pipeline structure:**
   ```python
   # registration.py
   import cv2
   import numpy as np
   
   def detect_keypoints(image):
       """Detect ORB keypoints in an image."""
       orb = cv2.ORB_create(nfeatures=500)
       keypoints, descriptors = orb.detectAndCompute(image, None)
       return keypoints, descriptors
   
   def match_features(descriptors1, descriptors2):
       """Match features between two images using BFMatcher."""
       bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
       matches = bf.match(descriptors1, descriptors2)
       matches = sorted(matches, key=lambda x: x.distance)
       return matches
   
   def estimate_homography(keypoints1, keypoints2, matches):
       """Estimate homography matrix using RANSAC."""
       src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
       dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
       M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
       return M, mask
   
   def warp_images(image1, M, shape):
       """Warp image1 to align with image2 using homography."""
       warped = cv2.warpPerspective(image1, M, (shape[1], shape[0]))
       return warped
   ```

2. **Create a test pipeline that chains all steps:**
   ```python
   def register_images(optical_path, thermal_path):
       optical = cv2.imread(optical_path, cv2.IMREAD_GRAYSCALE)
       thermal = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)
       kp1, desc1 = detect_keypoints(optical)
       kp2, desc2 = detect_keypoints(thermal)
       matches = match_features(desc1, desc2)
       M, mask = estimate_homography(kp1, kp2, matches)
       warped = warp_images(thermal, M, optical.shape)
       return warped, M, mask
   ```

3. **Test with placeholder images:**
   - Create two simple test images (e.g., overlapping squares with different patterns).
   - Verify the pipeline runs without errors.
   - Verify the output warped image aligns with the reference.

4. **Test with real placeholder data:**
   - Use two random images from the broad download (Task 3.2).
   - Run the registration pipeline.
   - Verify it produces a homography matrix and warped output.

5. **Handle edge cases:**
   - What if no matches are found? Add a check: `if len(matches) < 10: raise ValueError("Insufficient matches")`
   - What if the homography is singular? Check `cv2.findHomography` return value.

6. **Deliverable:** Working `registration.py` with `detect_keypoints()`, `match_features()`, `estimate_homography()`, `warp_images()` functions.

**Tools needed:** OpenCV, Python, NumPy, placeholder images

**Expected output:** `registration.py` — working registration pipeline code

---

#### Task 4.2 — Tune Registration on Proxy Data

**What this task means:** Using the labeled proxy dataset from Phase 3, tune the registration algorithm parameters (ORB feature count, RANSAC threshold, matching ratio) to achieve the best alignment accuracy. This is where the algorithm is actually measured against real data.

**Step-by-step sequence:**

1. **Load the proxy dataset:**
   ```python
   import pandas as pd
   manifest = pd.read_csv('data/labeled/manifest.csv')
   ```

2. **Set up parameter grids for tuning:**
   ```python
   param_grid = {
       'nfeatures': [200, 500, 1000],
       'scaleFactor': [1.1, 1.2, 1.3],
       'nlevels': [6, 8, 10],
       'ransac_thresh': [3.0, 5.0, 10.0]
   }
   ```

3. **Run registration with each parameter combination:**
   - For each combination, compute RMSE on a subsample of point pairs.
   - Track which parameters produce the lowest RMSE.

4. **Characterize accuracy across scene types:**
   - Group samples by cloud cover percentage (0–10%, 10–30%, 30–50%, >50%).
   - Compute RMSE for each group.
   - Flag any scene type where registration degrades sharply.

5. **Document parameter choices:**
   - Record the best parameters in `registration_tuning_log.md`.
   - Include RMSE values for each parameter combination tested.
   - Include the breakdown by scene type.

6. **Deliverable:** Tuned registration parameters; `registration_tuning_log.md` with results.

**Tools needed:** `registration.py` from Task 4.1, Phase 3 dataset, `sklearn` for analysis

**Expected output:** Optimized registration parameters; `registration_tuning_log.md`

---

#### Task 4.3 — Optional Public Dataset Supplement

**What this task means:** If available, supplement the proxy dataset with a public RGB+thermal paired dataset (e.g., FLIR's ADAS thermal dataset) to get more diverse training signal for registration and fusion development. This is fully optional and must be explicitly labeled as supplementary development data.

**Step-by-step sequence:**

1. **Identify a suitable public dataset:**
   - FLIR ADAS dataset (visible + thermal paired images).
   - Any other public RGB+thermal paired dataset.
   - Verify the dataset is publicly available and has a suitable license.

2. **Download and organize:**
   - Download the dataset.
   - Organize into `data/supplementary/` directory.
   - Create a manifest for supplementary samples.

3. **Integrate into the development workflow:**
   - Use supplementary data alongside Phase 3 proxy data for algorithm development.
   - **Critical:** Never conflate supplementary data with the payload's own sensor characteristics.
   - All results using supplementary data must be labeled: "supplementary development data."

4. **Deliverable:** Supplementary dataset in `data/supplementary/`; `supplementary_data_notes.md`.

**Tools needed:** Web browser, dataset download access

**Expected output:** Supplementary dataset; `supplementary_data_notes.md` (optional)

---

#### Task 4.4 — Build RMSE Measurement Harness

**What this task means:** Build a reusable tool to measure the accuracy of image registration. This requires manually annotating corresponding points on a subsample of images, then computing the Root Mean Square Error (RMSE) between the annotated source points and their projected positions in the warped image.

**Step-by-step sequence:**

1. **Create a point annotation tool:**
   ```python
   # annotate_points.py
   import cv2
   import json
   
   def annotate_corresponding_points(image1_path, image2_path, output_json):
       """Manually click corresponding points on two images."""
       points1 = []
       points2 = []
       
       def click_callback(event, x, y, flags, param):
           if event == cv2.EVENT_LBUTTONDOWN:
               # Add point to the appropriate list
               pass
       
       cv2.namedWindow('Image 1')
       cv2.setMouseCallback('Image 1', click_callback)
       cv2.imshow('Image 1', cv2.imread(image1_path))
       cv2.waitKey(0)
       # ... similar for Image 2
       # Save points to JSON
   ```

2. **Annotate at least 10–20 corresponding point pairs:**
   - Pick distinctive features visible in both optical and thermal images.
   - Click the same feature in both images.
   - Save the point pairs to a JSON file.

3. **Implement the RMSE calculation:**
   ```python
   def compute_rmse(points_source, points_warped):
       """Compute RMSE between source points and their projected positions."""
       errors = []
       for src, warp in zip(points_source, points_warped):
           # Project source point using homography
           projected = cv2.perspectiveTransform(np.array([[src]]), M)[0][0]
           error = np.sqrt((projected[0] - warp[0])**2 + (projected[1] - warp[1])**2)
           errors.append(error)
       return np.sqrt(np.mean(np.array(errors)**2))
   ```

4. **Measure registration accuracy:**
   - Apply the homography from Task 4.2 to the annotated points.
   - Compute RMSE for each image pair.
   - Average RMSE across all pairs.
   - Record results in `rmse_results.md`.

5. **Verify the harness works:**
   - Test on a pair where you know the registration should be good.
   - Test on a pair where registration should be poor (e.g., heavily cloudy images).
   - Verify RMSE values reflect the expected quality.

6. **Deliverable:** Working RMSE harness (`rmse_harness.py`); `rmse_results.md` with accuracy measurements.

**Tools needed:** OpenCV, Python, manually annotated point pairs

**Expected output:** `rmse_harness.py` — working RMSE measurement tool; `rmse_results.md`

---

#### Task 4.5 — Implement Fusion Formula (Code)

**What this task means:** Implement the weighted overlay fusion formula that combines the registered optical and thermal images into a single fused output. The formula is: $F(x,y) = \alpha\cdot\text{RGB}(x,y) + (1-\alpha)\cdot\text{Thermal}_{\text{registered}}(x,y)$

**Why it matters:** Fusion combines the best of both modalities — the structural detail from optical imagery and the thermal contrast from thermal imagery — into a single image that is more informative than either alone.

**Step-by-step sequence:**

1. **Implement the fusion formula:**
   ```python
   # fusion.py
   import numpy as np
   import cv2
   
   def fuse_images(optical, thermal_registered, alpha=0.5):
       """
       Fuse optical and registered thermal images using weighted overlay.
       F(x,y) = alpha * RGB(x,y) + (1-alpha) * Thermal_registered(x,y)
       """
       # Ensure both images have the same dimensions
       if optical.shape != thermal_registered.shape:
           # Resize thermal to match optical
           thermal_registered = cv2.resize(thermal_registered, (optical.shape[1], optical.shape[0]))
       
       # Normalize both to [0, 1]
       optical_norm = optical.astype(np.float64) / 255.0
       thermal_norm = thermal_registered.astype(np.float64) / 255.0
       
       # Apply fusion formula
       fused = alpha * optical_norm + (1 - alpha) * thermal_norm
       
       # Clip to [0, 1] and convert back to uint8
       fused = np.clip(fused, 0, 1)
       fused_uint8 = (fused * 255).astype(np.uint8)
       return fused_uint8
   ```

2. **Implement alpha sweep:**
   ```python
   def alpha_sweep(optical, thermal_registered, alpha_min=0.0, alpha_max=1.0, step=0.1):
       """Sweep alpha from 0 to 1 and return all fused results."""
       results = {}
       alpha = alpha_min
       while alpha <= alpha_max:
           fused = fuse_images(optical, thermal_registered, alpha)
           results[round(alpha, 1)] = fused
           alpha += step
       return results
   ```

3. **Test with placeholder images:**
   - Use two placeholder images (e.g., a grayscale optical image and a grayscale thermal image).
   - Run fusion with alpha = 0.0, 0.5, 1.0.
   - Verify: alpha=0.0 → pure thermal, alpha=1.0 → pure optical, alpha=0.5 → equal blend.

4. **Test with proxy data:**
   - Use one pair from the Phase 3 dataset.
   - Run alpha sweep and compute SSIM/PSNR for each alpha value.
   - Record the results.

5. **Deliverable:** Working `fusion.py` with `fuse_images()` and `alpha_sweep()` functions.

**Tools needed:** Python, NumPy, OpenCV, proxy image pairs

**Expected output:** `fusion.py` — working fusion implementation with alpha sweep

---

#### Task 4.6 — Document Proxy-Data Caveats

**What this task means:** Write a clear caveat for every finding in Phase 4, stating that results were measured on proxy/stand-in data and require re-measurement on real ground sensor data. This is a critical documentation requirement.

**Step-by-step sequence:**

1. **Review all Phase 4 results:**
   - Registration RMSE values (Task 4.4)
   - Fusion SSIM/PSNR values (Task 4.5)
   - Alpha recommendation (Task 4.5)
   - Registration tuning parameters (Task 4.2)

2. **Write a caveat for each result:**
   - For each finding, add: "This measurement was made on proxy/stand-in data (Sentinel-2 optical + Landsat thermal imagery) and requires re-measurement on real ground-captured RGB+thermal frame pairs when hardware becomes available."

3. **Create `proxy_data_caveats.md`:**
   - Organize by finding (registration, fusion, alpha).
   - Include the measured values and the caveat for each.
   - Add a summary statement: "All Phase 4 results are provisional and based on proxy data."

4. **Deliverable:** `proxy_data_caveats.md` — all findings tagged with proxy-data caveats.

**Tools needed:** Phase 4 results

**Expected output:** `proxy_data_caveats.md` — comprehensive proxy-data caveats for all Phase 4 findings

---

### Phase 5 — Inference Model Development & Quantization

**What this phase produces:** A trained float32 classification model and its quantized int8 counterpart, both evaluated on a held-out test set with a fully measured quantization accuracy delta.

---

#### Task 5.1 — Finalize Classification Task Definition

**What this task means:** Confirm the exact classification task the inference model will perform. This determines the model's output format, loss function, and evaluation metrics.

**Step-by-step sequence:**

1. **Choose the classification task:**
   - Option A: Cloud/No-Cloud classification (binary).
   - Option B: Thermal Anomaly/No-Anomaly classification (binary).
   - The choice should match what the Phase 3 dataset was labeled for (Task 3.5).

2. **Define the task specifications:**
   - **Input:** A fused RGB+thermal image pair (from Phase 6 pipeline output).
   - **Output:** Binary classification (e.g., 0 = no-cloud, 1 = cloud).
   - **Classes:** 2 classes.
   - **Evaluation metric:** Accuracy, plus per-class precision/recall and confusion matrix.

3. **Document the task definition:**
   ```markdown
   Task: Cloud/No-Cloud Classification
   Input: Fused optical+thermal image (25m GSD)
   Output: Binary label (0: no-cloud, 1: cloud)
   Loss Function: Binary Cross-Entropy
   Evaluation: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
   ```

4. **Write `task_definition.md`:**
   - Task choice and rationale.
   - Input/output specifications.
   - Label encoding.
   - Expected dataset size and split ratios.

5. **Deliverable:** `task_definition.md` with complete task definition and specs.

**Tools needed:** Phase 3 dataset, classification requirements

**Expected output:** `task_definition.md` — final classification task definition

---

#### Task 5.2 — Design CNN Architecture

**What this task means:** Design a MobileNet-style depthwise-separable CNN within the architecture's hard parameter ceiling. This is the model design only — actual training happens in Task 5.3.

**Step-by-step sequence:**

1. **Determine the parameter ceiling:**
   - From the architecture doc: low hundreds-of-thousands to a few million parameters.
   - Choose a target: e.g., under 2 million parameters.

2. **Design the MobileNet-style architecture:**
   ```python
   # model_architecture.py
   import tensorflow as tf
   from tensorflow.keras import layers, Model
   
   def build_model(input_shape=(256, 256, 3), num_classes=2, max_params=2_000_000):
       """Build a MobileNet-style depthwise-separable CNN."""
       inputs = layers.Input(shape=input_shape)
       
       # Initial convolution
       x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
       x = layers.BatchNormalization()(x)
       x = layers.ReLU()(x)
       
       # Depthwise separable convolution blocks
       for filters in [64, 128, 256, 512]:
           x = layers.DepthwiseConv2D((3, 3), padding='same')(x)
           x = layers.BatchNormalization()(x)
           x = layers.ReLU()(x)
           x = layers.Conv2D(filters, (1, 1), padding='same')(x)
           x = layers.BatchNormalization()(x)
           x = layers.ReLU()(x)
           x = layers.MaxPooling2D((2, 2))(x)
       
       # Global pooling and classification head
       x = layers.GlobalAveragePooling2D()(x)
       x = layers.Dropout(0.3)(x)
       outputs = layers.Dense(num_classes, activation='softmax')(x)
       
       model = Model(inputs, outputs)
       
       # Verify parameter count
       param_count = model.count_params()
       assert param_count <= max_params, f"Model has {param_count} params, exceeds {max_params}"
       
       return model
   ```

3. **Create the architecture diagram:**
   - Draw the layer structure (input → conv blocks → pooling → global avg pool → dropout → dense → output).
   - Include parameter counts for each layer.

4. **Verify the model compiles:**
   ```python
   model = build_model()
   model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
   model.summary()  # Verify parameter count and layer structure
   ```

5. **Deliverable:** `model_architecture.py` — model architecture specification and implementation; architecture diagram.

**Tools needed:** TensorFlow/Keras or PyTorch

**Expected output:** `model_architecture.py` — working CNN model skeleton under the parameter ceiling

---

#### Task 5.3 — Train the Inference Model

**What this task means:** Train the CNN model on the Phase 3 labeled dataset. This is where the model actually learns to classify images.

**Step-by-step sequence:**

1. **Set up the training environment:**
   ```python
   import tensorflow as tf
   from tensorflow.keras import callbacks
   
   # Set random seeds for reproducibility
   tf.random.set_seed(42)
   import numpy as np
   np.random.seed(42)
   ```

2. **Load and preprocess the dataset:**
   ```python
   import pandas as pd
   manifest = pd.read_csv('data/labeled/manifest.csv')
   
   def load_and_preprocess(sample_row):
       optical = tf.io.decode_file(image_path)  # Load TIFF
       thermal = tf.io.decode_file(thermal_path)
       # Resize to model input size
       optical = tf.image.resize(optical, [256, 256])
       thermal = tf.image.resize(thermal, [256, 256])
       # Fuse optical and thermal
       fused = 0.5 * optical + 0.5 * thermal
       return fused, label
   ```

3. **Create train/validation/test split:**
   - 70% training, 15% validation, 15% test.
   - Use `sklearn.model_selection.train_test_split` for the split.
   - Ensure the split is stratified (same proportion of cloud/no-cloud in each split).

4. **Set up data augmentation:**
   ```python
   data_augmentation = tf.keras.Sequential([
       layers.RandomFlip("horizontal"),
       layers.RandomRotation(0.1),
       layers.RandomZoom(0.1),
       layers.RandomBrightness(0.1),
   ])
   ```

5. **Configure training:**
   ```python
   model = build_model()  # From Task 5.2
   model.compile(
       optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
       loss='binary_crossentropy',
       metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
   )
   
   early_stopping = callbacks.EarlyStopping(patience=10, restore_best_weights=True)
   reduce_lr = callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
   ```

6. **Train the model:**
   ```python
   history = model.fit(
       train_dataset,
       validation_data=val_dataset,
       epochs=100,
       callbacks=[early_stopping, reduce_lr],
       batch_size=32
   )
   ```

7. **Monitor training:**
   - Plot training and validation loss curves.
   - Plot training and validation accuracy curves.
   - Watch for overfitting (training loss decreases but validation loss increases).
   - If overfitting occurs, increase dropout or add more augmentation.

8. **Save the trained model:**
   ```python
   model.save('model_float32.h5')
   ```

9. **Deliverable:** `model_float32.h5` — trained float32 model; training history plots.

**Tools needed:** TensorFlow/Keras, Phase 3 labeled dataset, matplotlib for plotting

**Expected output:** `model_float32.h5` — trained float32 model on the held-out test split

---

#### Task 5.4 — Evaluate Float32 Accuracy

**What this task means:** Evaluate the trained float32 model on the held-out test set to measure its classification performance. This gives the baseline accuracy before quantization.

**Step-by-step sequence:**

1. **Load the trained model and test set:**
   ```python
   model = tf.keras.models.load_model('model_float32.h5')
   test_dataset = ...  # Load test split
   ```

2. **Run inference on the test set:**
   ```python
   y_pred = model.predict(test_dataset)
   y_pred_classes = np.argmax(y_pred, axis=1)
   y_true = ...  # True labels from test set
   ```

3. **Compute accuracy:**
   ```python
   accuracy = np.mean(y_pred_classes == y_true)
   print(f"Test Accuracy: {accuracy:.4f}")
   ```

4. **Generate confusion matrix:**
   ```python
   from sklearn.metrics import confusion_matrix, classification_report
   cm = confusion_matrix(y_true, y_pred_classes)
   print(classification_report(y_true, y_pred_classes))
   ```

5. **Compute per-class precision and recall:**
   - Precision (class 1) = TP / (TP + FP)
   - Recall (class 1) = TP / (TP + FN)
   - F1-Score = 2 * (Precision * Recall) / (Precision + Recall)

6. **Save results:**
   ```python
   with open('evaluation_float32.md', 'w') as f:
       f.write(f"## Float32 Model Evaluation\n\n")
       f.write(f"**Test Accuracy:** {accuracy:.4f}\n\n")
       f.write(f"### Confusion Matrix\n```\n{cm}\n```\n\n")
       f.write(f"### Per-Class Metrics\n{classification_report(y_true, y_pred_classes)}\n")
   ```

7. **Deliverable:** `evaluation_float32.md` — accuracy, confusion matrix, per-class precision/recall/F1.

**Tools needed:** Trained model, test dataset, `sklearn.metrics`

**Expected output:** `evaluation_float32.md` — complete float32 model evaluation

---

#### Task 5.5 — Convert to .tflite and Apply Post-Training Int8 Quantization

**What this task means:** Convert the float32 model to TensorFlow Lite format and apply post-training int8 quantization. Quantization reduces the model size and inference speed requirements, making it suitable for deployment on edge devices (e.g., Raspberry Pi Zero 2W).

**Step-by-step sequence:**

1. **Convert the float32 model to TFLite:**
   ```python
   import tensorflow as tf
   
   # Load the float32 model
   model = tf.keras.models.load_model('model_float32.h5')
   
   # Convert to TFLite
   converter = tf.lite.TFLiteConverter.from_keras_model(model)
   tflite_model_float32 = converter.convert()
   
   # Save the float32 TFLite model
   with open('model_float32.tflite', 'wb') as f:
       f.write(tflite_model_float32)
   ```

2. **Apply post-training int8 quantization:**
   ```python
   converter = tf.lite.TFLiteConverter.from_keras_model(model)
   converter.optimizations = [tf.lite.Optimize.DEFAULT]
   # For full integer quantization, provide a representative dataset
   def representative_dataset():
       for batch in train_dataset.take(100):
           yield [batch[0].numpy().astype(np.float32)]
   converter.representative_dataset = representative_dataset
   converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
   converter.inference_input_type = tf.int8
   converter.inference_output_type = tf.int8
   
   tflite_model_int8 = converter.convert()
   
   with open('model_int8.tflite', 'wb') as f:
       f.write(tflite_model_int8)
   ```

3. **Verify the quantized model works:**
   ```python
   # Load the int8 model
   interpreter = tf.lite.Interpreter(model_path='model_int8.tflite')
   interpreter.allocate_tensors()
   # Run a test inference to verify it works
   ```

4. **Deliverable:** `model_int8.tflite` — quantized int8 model.

**Tools needed:** TensorFlow Lite Converter, trained float32 model, representative dataset

**Expected output:** `model_int8.tflite` — quantized int8 model

---

#### Task 5.6 — Compute Quantization Accuracy Delta

**What this task means:** Run both the float32 and int8 models on the identical held-out test set and compute the accuracy drop. This is a fully legitimate, hardware-independent measurement — both models can be run on a development machine's TFLite interpreter, so no physical hardware is required.

**Step-by-step sequence:**

1. **Run both models on the identical test set:**
   ```python
   import tensorflow as tf
   import numpy as np
   
   # Load float32 model
   interpreter_f32 = tf.lite.Interpreter(model_path='model_float32.tflite')
   interpreter_f32.allocate_tensors()
   
   # Load int8 model
   interpreter_i8 = tf.lite.Interpreter(model_path='model_int8.tflite')
   interpreter_i8.allocate_tensors()
   
   # Run inference on each test sample
   def run_inference(interpreter, test_data):
       input_details = interpreter.get_input_details()
       output_details = interpreter.get_output_details()
       predictions = []
       for sample in test_data:
           interpreter.set_tensor(input_details[0]['index'], sample)
           interpreter.invoke()
           output = interpreter.get_tensor(output_details[0]['index'])
           predictions.append(np.argmax(output))
       return np.array(predictions)
   
   y_pred_f32 = run_inference(interpreter_f32, test_data)
   y_pred_i8 = run_inference(interpreter_i8, test_data)
   ```

2. **Compute accuracy for each model:**
   ```python
   accuracy_f32 = np.mean(y_pred_f32 == y_true)
   accuracy_i8 = np.mean(y_pred_i8 == y_true)
   ```

3. **Compute the quantization delta:**
   ```python
   quantization_delta = accuracy_f32 - accuracy_i8
   print(f"Float32 Accuracy: {accuracy_f32:.4f}")
   print(f"Int8 Accuracy: {accuracy_i8:.4f}")
   print(f"Quantization Delta: {quantization_delta:.4f}")
   ```

4. **Determine if the delta is acceptable:**
   - If delta < 2%: Acceptable — quantization is working well.
   - If delta 2–5%: Marginal — consider quantization-aware training.
   - If delta > 5%: Unacceptable — apply quantization-aware training (Task 5.7).

5. **Save results:**
   ```python
   with open('quantization_delta.md', 'w') as f:
       f.write(f"# Quantization Accuracy Delta\n\n")
       f.write(f"| Model | Accuracy |\n")
       f.write(f"|-------|----------|\n")
       f.write(f"| Float32 | {accuracy_f32:.4f} |\n")
       f.write(f"| Int8 | {accuracy_i8:.4f} |\n")
       f.write(f"| **Delta** | **{quantization_delta:.4f}** |\n\n")
       f.write(f"{'Acceptable' if quantization_delta < 0.02 else 'Marginal' if quantization_delta < 0.05 else 'Unacceptable'}\n")
   ```

6. **Deliverable:** `quantization_delta.md` — measured quantization accuracy drop.

**Tools needed:** TFLite interpreter, both float32 and int8 models, held-out test set

**Expected output:** `quantization_delta.md` — measured quantization accuracy delta

---

#### Task 5.7 — Quantization-Aware Training Fallback (if needed)

**What this task means:** If the quantization accuracy delta from Task 5.6 is unacceptable (>5%), apply quantization-aware training (QAT) to retrain the model with simulated quantization during training, so the model learns to be robust to quantization effects.

**Step-by-step sequence:**

1. **Check if QAT is needed:**
   - Read `quantization_delta.md`.
   - If delta ≤ 5%, skip this task.
   - If delta > 5%, proceed with QAT.

2. **Set up quantization-aware training:**
   ```python
   import tensorflow_model_optimization as tfmot
   
   # Apply quantization aware training
   quantify_model = tfmot.quantization.keras.quantize_model
   qat_model = quantify_model(model)
   
   qat_model.compile(
       optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
       loss='binary_crossentropy',
       metrics=['accuracy']
   )
   ```

3. **Retrain with QAT:**
   ```python
   qat_history = qat_model.fit(
       train_dataset,
       validation_data=val_dataset,
       epochs=50,  # Fewer epochs since the model is already trained
       callbacks=[early_stopping],
       batch_size=32
   )
   ```

4. **Re-quantize and re-measure:**
   ```python
   converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
   converter.optimizations = [tf.lite.Optimize.DEFAULT]
   # ... (same representative dataset as before)
   qat_tflite_model = converter.convert()
   
   # Re-run Task 5.6 measurement on the QAT model
   ```

5. **Verify the QAT model's accuracy:**
   - Compare the QAT int8 accuracy against the float32 baseline.
   - If delta is now acceptable, proceed.

6. **Deliverable:** QAT-trained model; updated `quantization_delta.md` showing improved results.

**Tools needed:** TensorFlow Model Optimization toolkit, original model, training data

**Expected output:** QAT-trained model; updated `quantization_delta.md` (if triggered)

---

### Phase 6 — Digital Model Assembly

**What this phase produces:** The fully assembled end-to-end Digital Model pipeline — orbit → footprint → proxy imagery → registration → fusion → inference → classification result. This is the point at which the Digital Model actually comes into existence as a single running system.

---

#### Task 6.1 — Build Orchestration Pipeline

**What this task means:** Wire Phases 1–5 together into one coherent pipeline. This single script takes a target altitude and region, computes the pass schedule, pulls the corresponding resampled imagery, and runs it through registration → fusion → inference, producing a final classification result per simulated pass.

**Step-by-step sequence:**

1. **Define the pipeline structure:**
   ```python
   # digital_model_pipeline.py
   import sys
   sys.path.append('.')
   
   from orbital_model import orbital_period, sun_sync_inclination, passes_over_region
   from sensor_model import gsd, swath, footprint
   from resample_pipeline import resample_to_footprint
   from registration import register_images
   from fusion import fuse_images
   from model_architecture import build_model
   import tensorflow as tf
   
   def run_pipeline(altitude_km, target_lat, target_lon, radius_km):
       """
       End-to-end Digital Model pipeline.
       1. Compute orbital parameters
       2. Compute sensor parameters
       3. Get pass schedule
       4. For each pass: resample → register → fuse → infer
       5. Return classification results
       """
       # Phase 1: Orbital mechanics
       period = orbital_period(altitude_km)
       inclination = sun_sync_inclination(altitude_km)
       overpasses = passes_over_region(target_lat, target_lon, radius_km)
       
       # Phase 2: Sensor model
       target_gsd = gsd(...)  # Use provisional GSD
       
       # Load trained model
       model = tf.keras.models.load_model('model_float32.h5')
       
       results = []
       for pass_info in overpasses:
           # Phase 3: Resample imagery for this pass
           optical_path, thermal_path = get_resampled_paths(pass_info)
           
           # Phase 4: Register and fuse
           warped, M, mask = register_images(optical_path, thermal_path)
           fused = fuse_images(optical_image, warped_thermal, alpha=0.5)
           
           # Phase 5: Inference
           prediction = model.predict(fused)
           result = {
               'pass_id': pass_info['pass_id'],
               'date': pass_info['date'],
               'prediction': prediction,
               'classification': 'cloud' if prediction[1] > 0.5 else 'no-cloud'
           }
           results.append(result)
       
       return results
   ```

2. **Ensure modularity:**
   - Each phase is a separate, independently callable function/module.
   - The pipeline orchestration script calls each module in sequence.
   - No phase's code is embedded inside another phase's code.

3. **Test the pipeline on a single sample:**
   - Run the pipeline for one pass.
   - Verify it produces a classification result without errors.
   - Check that each module is called correctly.

4. **Test the pipeline on the full dataset:**
   - Run the pipeline on all 100+ samples.
   - Verify it completes without unhandled errors on every sample.
   - Record any errors and fix them.

5. **Deliverable:** `digital_model_pipeline.py` — working end-to-end pipeline.

**Tools needed:** All Phase 1–5 modules, trained model, Phase 3 dataset

**Expected output:** `digital_model_pipeline.py` — working end-to-end orchestration pipeline

---

#### Task 6.2 — Enforce Modular Boundaries

**What this task means:** Verify that every stage of the pipeline is a clearly separated, independently callable function/module. This modularity is what will allow the Digital Shadow stage to swap in real ground-sensor inputs without rewriting the registration/fusion/inference logic.

**Step-by-step sequence:**

1. **Audit each module's independence:**
   - Can `orbital_model.py` be imported and used without importing any other phase's code?
   - Can `sensor_model.py` be imported independently?
   - Can `registration.py` run on any pair of images without requiring specific pipeline context?
   - Can `fusion.py` fuse any two images without requiring pipeline state?
   - Can the inference model be loaded and run independently?

2. **Fix any coupling issues:**
   - If any module depends on another module's internal state, refactor it.
   - Use configuration files or environment variables instead of hardcoded inter-module dependencies.

3. **Verify importability:**
   ```python
   # Test each module can be imported independently
   import orbital_model
   import sensor_model
   import resample_pipeline
   import registration
   import fusion
   import model_architecture
   ```

4. **Document the module interfaces:**
   - For each module, list its public functions and their signatures.
   - Note which inputs are provisional and what needs to be swapped in.

5. **Deliverable:** Verified modular pipeline; `module_interfaces.md`.

**Tools needed:** All Phase 1–5 modules

**Expected output:** Verified modular pipeline; `module_interfaces.md`

---

#### Task 6.3 — Full End-to-End Run on Complete Dataset

**What this task means:** Run the assembled pipeline on the full Phase 3 dataset (100+ samples) and confirm it completes without unhandled errors on every sample.

**Step-by-step sequence:**

1. **Run the pipeline on all samples:**
   ```python
   from digital_model_pipeline import run_pipeline
   
   all_results = []
   errors = []
   for sample in manifest.itertuples():
       try:
           result = run_pipeline(altitude_km, target_lat, target_lon, radius_km)
           all_results.append(result)
       except Exception as e:
           errors.append({'sample_id': sample.sample_id, 'error': str(e)})
   
   print(f"Processed: {len(all_results)}/{len(manifest)} samples")
   print(f"Errors: {len(errors)}")
   ```

2. **Fix any errors:**
   - For each error, identify the cause.
   - Fix the pipeline code or the data issue.
   - Re-run until 100% completion.

3. **Verify all results are reasonable:**
   - Check that classification results are in the expected range (0 or 1 for binary).
   - Check that no result is null or undefined.
   - Spot-check a few results manually.

4. **Deliverable:** Confirmed successful end-to-end run on the full dataset; `pipeline_run_results.md`.

**Tools needed:** `digital_model_pipeline.py`, Phase 3 dataset, trained model

**Expected output:** `pipeline_run_results.md` — confirmation of successful full-dataset run

---

#### Task 6.4 — Write README

**What this task means:** Write a single README that describes how to run the full Digital Model pipeline from a fresh checkout. Include all PROVISIONAL labels and state exactly what needs to be swapped in once hardware exists.

**Step-by-step sequence:**

1. **Write the README structure:**
   ```markdown
   # Digital Model — End-to-End Pipeline
   
   ## Overview
   This is a Digital Model — a self-contained simulation with no automatic data connection to physical hardware.
   
   ## Quick Start
   ```bash
   git clone <repo>
   pip install -r requirements.txt
   python digital_model_pipeline.py --altitude 500 --lat 28.6 --lon 77.2 --radius 50
   ```
   
   ## PROVISIONAL Labels
   - Camera specs: PROVISIONAL — pending final camera procurement
   - Orbital altitude: PROVISIONAL — shared decision value of 500 km
   - Classification task: PROVISIONAL — cloud/no-cloud (pending confirmation)
   
   ## What Changes When Hardware Arrives
   - See `hardware_handoff.md` for the complete list of provisional inputs and their replacements.
   ```

2. **Include all PROVISIONAL labels:**
   - Reference `PROVISIONAL_LABELS.md` for the complete list.
   - Add a clear "PROVISIONAL ASSUMPTIONS" section near the top.

3. **State what to swap in:**
   - Real camera datasheet values → replace in `sensor_specs.json`.
   - Real ground-captured RGB+thermal data → replace `data/labeled/`.
   - Real ground baseline → compare against orbit-side results.

4. **Deliverable:** `README.md`.

**Tools needed:** All project files, PROVISIONAL_LABELS.md, hardware_handoff.md

**Expected output:** `README.md` — complete pipeline usage guide

---

### Phase 7 — Internal Validation & Self-Consistency Benchmarking

**What this phase produces:** A written internal-validation report confirming the Digital Model behaves sensibly and consistently on its own terms, independent of any ground comparison.

---

#### Task 7.1 — Registration Stability by Scene Type

**What this task means:** Confirm that the RMSE (from Task 4.4) stays stable and reasonable across the full range of scene types in the Phase 3 dataset. Flag any scene category where registration degrades sharply.

**Step-by-step sequence:**

1. **Stratify the dataset by scene type:**
   - Group samples by cloud cover percentage: 0–10%, 10–30%, 30–50%, >50%.
   - Or by thermal background type: urban, water, vegetation, etc.

2. **Compute RMSE for each stratum:**
   - Use the RMSE harness from Task 4.4.
   - For each scene type group, compute the average RMSE.

3. **Flag problematic scene types:**
   - If any stratum's RMSE is >2× the average RMSE, flag it.
   - Document the affected scene types and the magnitude of degradation.

4. **Write the finding into `internal_validation_report.md`:**
   ```markdown
   ### Registration Stability by Scene Type
   | Scene Type | Average RMSE (pixels) | Status |
   |------------|----------------------|--------|
   | Clear sky | 1.2 | ✅ Stable |
   | Partial cloud | 3.8 | ⚠️ Degraded (>2× average) |
   ...
   ```

5. **Deliverable:** Registration stability analysis in `internal_validation_report.md`.

**Tools needed:** RMSE harness (Task 4.4), Phase 3 dataset

**Expected output:** Registration stability analysis in `internal_validation_report.md`

---

#### Task 7.2 — Altitude Sensitivity Sweep

**What this task means:** Re-run the pipeline at two or three different candidate altitudes and confirm the classification accuracy responds in the expected direction (coarser GSD at higher altitude should NOT improve accuracy — if it does, something in the resampling logic is likely wrong).

**Step-by-step sequence:**

1. **Choose candidate altitudes:**
   - e.g., 400 km, 500 km, 600 km.

2. **For each altitude:**
   - Compute the GSD for that altitude.
   - Resample the dataset to that GSD.
   - Run the full pipeline (registration → fusion → inference).
   - Record the classification accuracy.

3. **Plot altitude vs. accuracy:**
   ```python
   import matplotlib.pyplot as plt
   altitudes = [400, 500, 600]
   accuracies = [0.85, 0.82, 0.78]  # Example values
   plt.plot(altitudes, accuracies, 'bo-')
   plt.xlabel('Altitude (km)')
   plt.ylabel('Classification Accuracy')
   plt.title('Altitude Sensitivity')
   plt.savefig('data/altitude_sensitivity.png')
   ```

4. **Verify the expected direction:**
   - Higher altitude → coarser GSD → lower accuracy (expected).
   - If accuracy increases with altitude, investigate the resampling logic for bugs.

5. **Deliverable:** Altitude sensitivity analysis in `internal_validation_report.md`; `data/altitude_sensitivity.png`.

**Tools needed:** Pipeline code, matplotlib, candidate altitudes

**Expected output:** Altitude sensitivity analysis in `internal_validation_report.md`

---

#### Task 7.3 — Alpha Sensitivity Re-Check

**What this task means:** Confirm that the fusion α value chosen in Phase 4 still produces the best (or near-best) inference accuracy across the full Phase 3 dataset, not just the tuning subsample it was originally chosen on.

**Step-by-step sequence:**

1. **Re-run the alpha sweep on the full dataset:**
   ```python
   alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
   accuracies = []
   for alpha in alphas:
       # Run pipeline with this alpha
       accuracy = run_pipeline_with_alpha(alpha)
       accuracies.append(accuracy)
   ```

2. **Identify the best alpha:**
   - Find the alpha that produces the highest average accuracy.
   - Compare to the alpha chosen in Phase 4.

3. **Verify consistency:**
   - If the best alpha is within ±0.2 of the Phase 4 choice → consistent.
   - If the best alpha differs significantly → document the change and investigate why.

4. **Deliverable:** Alpha sensitivity analysis in `internal_validation_report.md`.

**Tools needed:** Pipeline code, fusion module

**Expected output:** Alpha sensitivity analysis in `internal_validation_report.md`

---

#### Task 7.4 — Provisional-Camera Sensitivity Check

**What this task means:** Re-run the full pipeline with one or two alternative candidate camera specs (different pixel size/focal length) to see how sensitive the final classification accuracy is to the still-unconfirmed camera choice.

**Step-by-step sequence:**

1. **Choose alternative camera specs:**
   - Camera A: Current provisional specs (from `sensor_specs.json`).
   - Camera B: Alternative with larger pixel size (e.g., 7 µm instead of 5 µm).
   - Camera C: Alternative with longer focal length (e.g., 150 mm instead of 100 mm).

2. **For each camera spec:**
   - Recompute GSD with the new camera specs.
   - Resample the dataset to the new GSD.
   - Run the full pipeline.
   - Record the classification accuracy.

3. **Compare results:**
   ```markdown
   | Camera | Pixel Size (µm) | Focal Length (mm) | GSD (m) | Accuracy | Delta vs Current |
   |--------|-----------------|-------------------|---------|----------|-----------------|
   | Current | 5.0 | 100 | 25.0 | 0.82 | — |
   | Alt B | 7.0 | 100 | 35.0 | 0.76 | -0.06 |
   | Alt C | 5.0 | 150 | 16.7 | 0.85 | +0.03 |
   ```

4. **Assess risk:**
   - If accuracy varies by >10% across camera choices → high risk riding on Phase 2's provisional value.
   - Document this risk in `internal_validation_report.md`.

5. **Deliverable:** Camera sensitivity analysis in `internal_validation_report.md`.

**Tools needed:** Pipeline code, alternative camera specs

**Expected output:** Camera sensitivity analysis in `internal_validation_report.md`

---

#### Task 7.5 — Reproducibility Check

**What this task means:** Re-run the full Phase 6 pipeline a second time on the same inputs and confirm identical (or near-identical) results. This verifies the simulation isn't silently non-deterministic in a way that would undermine any result drawn from it later.

**Step-by-step sequence:**

1. **Run the pipeline once and save all results:**
   ```python
   results_run1 = run_pipeline(altitude_km, target_lat, target_lon, radius_km)
   save_results(results_run1, 'run_1_results.json')
   ```

2. **Re-seed any random number generators:**
   ```python
   tf.random.set_seed(42)
   np.random.seed(42)
   ```

3. **Run the pipeline a second time:**
   ```python
   results_run2 = run_pipeline(altitude_km, target_lat, target_lon, radius_km)
   save_results(results_run2, 'run_2_results.json')
   ```

4. **Compare results:**
   ```python
   # Compare classification results
   for r1, r2 in zip(results_run1, results_run2):
       assert r1['classification'] == r2['classification'], f"Mismatch on {r1['pass_id']}"
   ```

5. **Document the comparison:**
   - If all results match → reproducibility confirmed.
   - If results differ within a small tolerance (e.g., due to data augmentation) → document the tolerance.
   - If results differ significantly → investigate non-deterministic code paths.

6. **Deliverable:** Reproducibility confirmation in `internal_validation_report.md`.

**Tools needed:** Pipeline code, Python random seeding

**Expected output:** Reproducibility confirmation in `internal_validation_report.md`

---

### Phase 8 — Completion Package & Handoff to Digital Shadow Stage

**What this phase produces:** A complete, versioned Digital Model package ready to be handed off to the Digital Shadow stage.

---

#### Task 8.1 — Write Final Digital Model Report

**What this task means:** Compile all findings from Phases 1–7 into a comprehensive final report covering orbital parameters, provisional sensor model, proxy dataset composition, registration/fusion/inference results (all explicitly labeled as proxy-data), and Phase 7 self-consistency findings.

**Step-by-step sequence:**

1. **Gather all source documents:**
   - `cross_validation_log.md` (Phase 1)
   - `sensor_model.py` and `sensor_go_nogo.md` (Phase 2)
   - `qc_report.md` (Phase 3)
   - `proxy_data_caveats.md` (Phase 4)
   - `quantization_delta.md` (Phase 5)
   - `internal_validation_report.md` (Phase 7)

2. **Write the report structure:**
   ```markdown
   # Final Digital Model Report
   
   ## 1. Orbital Parameters
   - Altitude: [value] (PROVISIONAL)
   - Orbital Period: [value] seconds
   - Sun-Synchronous Inclination: [value] degrees
   - Cross-validation: Three-way agreement within 0.5% (see cross_validation_log.md)
   
   ## 2. Provisional Sensor Model
   - Camera: [model] (PROVISIONAL)
   - GSD: [value] m (PROVISIONAL)
   - Swath: [value] m (PROVISIONAL)
   - Go/No-Go Judgment: [judgment] (PROVISIONAL)
   
   ## 3. Proxy Dataset
   - Samples: [number] (100+)
   - Sources: Sentinel-2 optical + Landsat thermal
   - Labels: [classification task]
   - Quality: QC passed (see qc_report.md)
   - **All data is proxy/stand-in data.**
   
   ## 4. Registration & Fusion Results
   - RMSE: [value] pixels (on proxy data)
   - Fusion alpha: [value] (SSIM/PSNR optimized)
   - **All results measured on proxy data.**
   
   ## 5. Inference Model Results
   - Float32 accuracy: [value]
   - Int8 accuracy: [value]
   - Quantization delta: [value]
   - **All results measured on proxy data.**
   
   ## 6. Self-Consistency Validation
   - Registration stability: [stable/degraded by scene type]
   - Altitude sensitivity: [expected direction confirmed]
   - Alpha sensitivity: [consistent]
   - Camera sensitivity: [risk level]
   - Reproducibility: [confirmed]
   
   ## 7. Terminology Statement
   This is a Digital Model — a self-contained simulation with no automatic data connection to physical hardware, since none currently exists.
   ```

3. **Deliverable:** `final_digital_model_report.md` — complete final report.

**Tools needed:** All source documents

**Expected output:** `final_digital_model_report.md`

---

#### Task 8.2 — Finalize "What Changes When Hardware Arrives"

**What this task means:** Complete the hardware handoff document listing every provisional input and exactly what replaces it when hardware arrives.

**Step-by-step sequence:**

1. **List every provisional input:**
   - Provisional camera datasheet values → real procured camera's datasheet values (re-run Phase 2).
   - Proxy-data registration/fusion tuning → re-validated against real ground-captured RGB+thermal frame pairs.
   - Proxy-data-trained inference model → re-evaluated (and likely fine-tuned or retrained) on real ground-captured labeled data.
   - Proxy dataset → replaced with real ground-captured data.
   - Orbital altitude → confirmed/updated based on mission selection.

2. **Structure as a clear table:**
   ```markdown
   | Provisional Input | Current Value | What Replaces It | Phase |
   |-------------------|---------------|-----------------|-------|
   | Camera datasheet | SX-130 (5µm, 100mm) | Real procured camera datasheet | Phase 2 |
   | Proxy dataset | Sentinel-2 + Landsat | Real ground RGB+thermal frames | Phase 3 |
   | Inference model | Proxy-trained | Re-trained on real ground data | Phase 5 |
   | Ground baseline | None | Ground-captured comparison metrics | Digital Shadow |
   ```

3. **Deliverable:** Final `hardware_handoff.md`.

**Tools needed:** All provisional input documentation

**Expected output:** Final `hardware_handoff.md`

---

#### Task 8.3 — Finalize Terminology Statement

**What this task means:** Ensure the Digital Model / Digital Shadow / Digital Twin terminology statement is final and consistent throughout all documents.

**Step-by-step sequence:**

1. **Write the final terminology statement:**
   ```markdown
   **Terminology:** Per the Kritzinger et al. (2018) taxonomy:
   - *Digital Model*: No automatic data connection to physical hardware. **This is what this plan produces.**
   - *Digital Shadow*: Automatic one-way connection (physical → digital). Begins when ground-test data feeds into the model.
   - *Digital Twin*: Automatic two-way connection (physical ↔ digital). Requires a flown payload streaming telemetry back.
   
   This plan produces a Digital Model — a self-contained simulation with no automatic data connection to physical hardware, since none currently exists. It will become a Digital Shadow once ground-test data feeds into it, and would only become a full Digital Twin if a flown payload streamed telemetry back into it automatically — which is out of scope for this project.
   ```

2. **Verify consistency across all documents:**
   - Check `README.md`, `final_digital_model_report.md`, `hardware_handoff.md`, and any other docs.
   - Ensure every document uses the correct terminology.

3. **Deliverable:** Final `terminology_statement.md`.

**Tools needed:** All project documents

**Expected output:** Final `terminology_statement.md`

---

#### Task 8.4 — Tag/Archive Final Version

**What this task means:** Tag the exact code, dataset, and trained model versions used to produce the final report, so the Digital Shadow work has a precise, reproducible starting point.

**Step-by-step sequence:**

1. **Create a git tag:**
   ```bash
   git add -A
   git commit -m "Digital Model completion package — version 1.0"
   git tag -a v1.0-digital-model -m "Complete Digital Model: all 8 phases finished, validation passed"
   git push origin v1.0-digital-model
   ```

2. **Archive the specific versions:**
   - The exact code commit hash.
   - The exact dataset manifest version.
   - The exact trained model file (`model_float32.h5` and `model_int8.tflite`).
   - Record all three in a `version_info.json` file.

3. **Create a release on GitHub (if applicable):**
   - Tag the release.
   - Attach the final report and key deliverables.

4. **Deliverable:** Git tag v1.0-digital-model; `version_info.json`.

**Tools needed:** Git, GitHub (if applicable)

**Expected output:** Git tag v1.0-digital-model; `version_info.json`

---

## Critical Path Summary

```
Day 1: Team A starts Phase 1 (1.1, 1.2) + Team B starts Phase 3 (3.1, 3.2) + Tracks D/E/F code
Day 2: Team A completes Phase 1 (1.3–1.5) + Phase 2 (2.1–2.4) → unblocks Phase 3 task 3.3
Day 3: Team B completes Phase 3 (3.3–3.6) → unblocks Phase 4 tuning + Phase 5 training
Day 4: Team C completes Phase 5 training + evaluation → Team B completes Phase 4
Day 5: Team C completes Phase 5 quantization + Phase 6 assembly
Day 6: All teams complete Phase 7 validation + Phase 8 completion package
```

**Hard Join Points:**
1. **End of Day 2** — Team A delivers pass footprints + GSD → unblocks Team B's resampling (3.3)
2. **End of Day 3** — Team B delivers labeled dataset → unblocks Team C's training (5.3)
3. **End of Day 4** — Team B completes Phase 4 → Team C can assemble pipeline (6.1)
4. **End of Day 5** — Team C has assembled pipeline → validation runs on Day 6

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Phase 1/2 deliverables delayed past Day 2 | Medium | High — blocks Phase 3.3 | Team A starts Day 1; parallel validation work begins Day 2 |
| Phase 3 dataset < 100 samples by Day 3 | Medium | High — blocks training | Download extra tiles; use broader region or more scene conditions |
| Registration accuracy poor on proxy data | Medium | Medium — affects Fusion/Inference | Use supplementary public dataset (Track D/4.3); adjust algorithm parameters |
| Quantization delta unacceptable | Low | Medium — triggers QAT fallback | Budget Day 5–6 for QAT if needed; use quantization-aware training |
| Track D/E code developed against placeholder data doesn't hold | Medium | Medium — re-validation needed | Budget short re-validation pass when Phase 3 data lands (Day 3–4) |
| Team handoff failures | Low | High — lost work | Daily standups; handoff checklists; cross-team documentation |

---

## Deliverables Checklist

### Team A Deliverables (Phases 1–2)
- [ ] `orbital_model.py` — `orbital_period()`, `sun_sync_inclination()`, `ground_track()`, `passes_over_region()`
- [ ] `cross_validation_log.md` — three-way cross-validation record (hand calculation, GMAT, Python library)
- [ ] `data/ground_track.csv` — 24-hour ground track propagation
- [ ] `data/ground_track_plot.png` — ground track visualization
- [ ] `data/overpass_schedule.txt` — overpass timestamps for target region
- [ ] `sensor_model.py` — `gsd()`, `swath()`, `footprint()` with PROVISIONAL tags
- [ ] `sensor_specs.json` — candidate camera specs with source citations
- [ ] `sensor_go_nogo.md` — preliminary go/no-go judgment with reasoning
- [ ] `PROVISIONAL_LABELS.md` — list of all provisional references
- [ ] `data/camera_datasheets/` — downloaded camera datasheet PDFs
- [ ] `validation_suite.py` — complete Phase 7 validation scripts
- [ ] `internal_validation_report.md` — Phase 7 self-consistency findings

### Team B Deliverables (Phases 3–4)
- [ ] `data_access.md` — account registration and access documentation
- [ ] `.env` — credentials file (in .gitignore)
- [ ] `data/manifest_raw.csv` — raw download manifest
- [ ] `data/raw/sentinel2/` — optical tile collection
- [ ] `data/raw/landsat/` — thermal-band tile collection
- [ ] `resample_pipeline.py` — crop/reproject/resample script
- [ ] `data/labeled/manifest.csv` — labeled dataset manifest (100+ samples)
- [ ] `data/labeled/` — organized directory with labeled image pairs
- [ ] `labeling_method.md` — description of how labels were determined
- [ ] `qc_report.md` — dataset quality-check report
- [ ] `registration.py` — registration pipeline (ORB → matching → RANSAC → warp)
- [ ] `registration_tuning_log.md` — registration parameter tuning results
- [ ] `rmse_harness.py` — working RMSE measurement tool
- [ ] `rmse_results.md` — RMSE accuracy report
- [ ] `fusion.py` — fusion implementation with alpha sweep
- [ ] `fusion_alpha_recommendation.md` — SSIM/PSNR-informed α recommendation
- [ ] `proxy_data_caveats.md` — all findings tagged with proxy-data caveats
- [ ] `supplementary_data_notes.md` (optional) — supplementary dataset integration
- [ ] Complete handoff package to Team C

### Team C Deliverables (Phases 5–8)
- [ ] `task_definition.md` — final classification task definition
- [ ] `model_architecture.py` — CNN architecture specification + implementation
- [ ] `model_float32.h5` — trained float32 model
- [ ] `model_float32.tflite` — float32 TFLite model
- [ ] `evaluation_float32.md` — float32 accuracy, confusion matrix, precision/recall
- [ ] `model_int8.tflite` — quantized int8 model
- [ ] `quantization_delta.md` — measured quantization accuracy drop
- [ ] `digital_model_pipeline.py` — end-to-end orchestration pipeline
- [ ] `module_interfaces.md` — module interface documentation
- [ ] `pipeline_run_results.md` — confirmation of successful full-dataset run
- [ ] `README.md` — pipeline usage guide with all PROVISIONAL labels
- [ ] `internal_validation_report.md` — Phase 7 self-consistency validation
- [ ] `data/altitude_sensitivity.png` — altitude sensitivity chart
- [ ] `final_digital_model_report.md` — complete final report
- [ ] `hardware_handoff.md` — "what changes when hardware arrives"
- [ ] `terminology_statement.md` — Digital Model / Shadow / Twin terminology
- [ ] `version_info.json` — version tracking for code, data, and models
- [ ] Git tag v1.0-digital-model
- [ ] Versioned package ready for Digital Shadow handoff

---

## Team Communication Protocol

| Event | Who → Whom | When |
|-------|-----------|------|
| Phase 1+2 deliverables | Team A → Team B | End of Day 2 |
| Phase 3 dataset complete | Team B → Team C | End of Day 3 |
| Phase 4 complete | Team B → Team C | End of Day 4 |
| Phase 5+6 assembly | Team C → Team A (for validation) | End of Day 5 |
| Phase 7+8 validation | All teams → Final review | Day 6 |
| Daily standup | All teams | 15 min each morning |
| Handoff checklist | Team B → Team C | End of Day 4 & Day 5 |

---

## Quick-Reference: What Each Team Does Each Day

| Day | Team A | Team B | Team C |
|-----|--------|--------|--------|
| **Day 1** | 1.1 (orbital period), 1.2 (inclination), 2.1 (camera datasheet) | 3.1 (accounts), 3.2 (downloads), 4.1 (registration code), 4.5 (fusion code), 5.1 (task definition), 5.2 (CNN design), 8.2/8.3 (documentation) | 8.2 review, 8.3 review, 7.1–7.5 validation planning |
| **Day 2** | 1.3 (cross-validation), 1.4 (ground track), 1.5 (pass-over-region), 2.2 (GSD/swath), 2.3 (PROVISIONAL tags), 2.4 (go/no-go) | 3.3 (resampling), 3.5 (labeling), 4.2 (registration tuning), 4.4 (RMSE harness) | 5.3 prep (training pipeline), 7.1–7.5 framework |
| **Day 3** | QA review, validation prep | 3.4 (package dataset), 3.6 (QC), 4.2 (continue tuning), 4.4 (complete RMSE) | 5.3 (train model), 5.1/5.2 finalize |
| **Day 4** | Altitude sensitivity, reproducibility scripts | 4.6 (proxy caveats), 4.3 (optional), handoff prep | 5.4 (evaluate float32), 5.5 (quantize) |
| **Day 5** | Complete validation scripts | Final handoff to C | 5.6 (quantization delta), 5.7 (QAT if needed), 6.1 (assembly) |
| **Day 6** | Run Phase 7 validation | — | 6.2 (modularity), 6.3 (full run), 6.4 (README), 7.*, 8.1–8.4 (completion) |

---

*This guide was derived from `Digital_Model_Implementation_Plan.md` (the 8-phase specification) and `Digital_Model_Parallelization_Analysis.md` (the dependency analysis and parallelization tracks). All task descriptions, formulas, tools, and deliverables are drawn directly from the source plan. The 3-team, 6-day structure and day-by-day assignments are original synthesis produced to map the 8 phases onto a compressed timeline using the parallelization analysis.*
