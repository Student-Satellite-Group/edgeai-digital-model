# Provisional Parameters Tracking Register

All parameters below are derived from provisional candidate camera specifications (`sensor_specs.json`) and assumed LEO orbital altitudes (500 km). They are subject to mandatory re-calculation upon final hardware payload procurement.

| File Path | Entity / Constant | Current Value | Status / Dependency |
| :--- | :--- | :--- | :--- |
| `sensor_specs.json` | `pixel_size_um` | 5.0 µm | PROVISIONAL — Pending procurement |
| `sensor_specs.json` | `focal_length_mm` | 100.0 mm | PROVISIONAL — Pending procurement |
| `sensor_model.py` | `PROVISIONAL` | `True` | Global module flag |
| `sensor_model.py` | `PROVISIONAL_TAG` | String Tag | Hardcoded warning string |
| `sensor_model.py` | `gsd()` return | 25.0 m/pixel | Derived at 500 km altitude |
| `sensor_model.py` | `swath()` return | 102.4 km | Derived at 500 km altitude |
| `sensor_go_nogo.md` | Target GSD Threshold | 25.0 m/pixel | Evaluated against 50.0 m target |