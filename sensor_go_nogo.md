# Preliminary Sensor Geometry Go/No-Go Decision

> **CAVEAT:** This judgment is preliminary and based on PROVISIONAL GSD values derived from candidate datasheets. It must be re-evaluated when physical payload procurement completes.

## 1. Parameters Assessed
* **Candidate Camera Model:** Provisional LEO Optical Payload
* **Assumed Orbital Altitude:** 500.0 km
* **Computed Provisional GSD:** 25.0 meters/pixel
* **Computed Provisional Swath:** 102.4 km

## 2. Classification Task Target
* **Primary Mission Objective:** Cloud vs. No-Cloud binary boundary classification and spatial mapping.
* **Spatial Resolution Requirement:** Target GSD ≤ 50.0 meters per pixel.

## 3. Decision Verdict: [ GO ]
* **Rationale:** The computed provisional GSD of **25.0 m/pixel** comfortably satisfies the target threshold of **50.0 m/pixel** required for cloud boundary identification without introducing excessive image footprint data overhead for onboard edge computing.
* **Trade-off Evaluation:** 
  * If altitude increases to 600 km, GSD becomes 30.0 m (still **GO**).
  * If altitude drops to 400 km, GSD becomes 20.0 m (higher precision, smaller swath).

## 4. Alternate Task Mitigations
* If mission scope shifts to sub-10m point thermal hotspot detection, the current sensor configuration will trigger a **NO-GO**, requiring optics with focal length $f \ge 250\text{ mm}$.