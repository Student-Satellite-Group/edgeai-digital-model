# Task 8.2 — What Changes When Hardware Arrives

**Status:** FINAL; Track: Team C, Phase 8.

Every input to this Digital Model is currently **provisional** because no
physical hardware exists yet. When hardware arrives, each provisional input is
replaced as follows:

| Provisional Input | Current Value | What Replaces It | Phase |
|-------------------|---------------|-----------------|-------|
| Camera datasheet | nüSpace CMOS (Nüvü Caméras) specsheet v3.1.3 — 4.6 µm, 320 mm focal, GSD 7.2 m @ 500 km (`sensor_specs.json`) | Real procured camera datasheet (re-run Phase 2 values) | Phase 2 |
| Proxy dataset | Sentinel-2 L2A optical + Landsat-8 B10 thermal (`data/manifest_raw.csv`) | Real ground-captured RGB + thermal frame pairs | Phase 3 |
| Registration / fusion tuning | Tuned on synthetic + proxy pairs (`registration.py`, `fusion.py`) | Re-validated and re-tuned on real ground-captured frame pairs | Phase 4 |
| Inference model | Proxy-trained cloud/no-cloud CNN (`model_architecture.py`) | Re-trained (or fine-tuned) on real ground-captured labeled data | Phase 5 |
| Orbital altitude | 500 km (locked, PROVISIONAL) | Confirmed/updated value from final mission selection | Phase 1 |
| Ground baseline | None — no ground truth exists | Ground-captured comparison metrics | Digital Shadow |

## Timing / owners

- Phase boxes above mark **where each replacement lands** in the 6-day plan.
- The "Digital Shadow" entry means the model's value only becomes fully
  grounded once ground-test frames feed in — that is a post-implementation
  effort, not a Day-1-6 task.

## Trigger

This document is refreshed whenever any provisional input above changes —
specifically after: camera selection confirmation (end of Phase 2), Phase 3
dataset finalization, and any mission-altitude change.