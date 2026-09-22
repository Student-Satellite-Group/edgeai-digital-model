# Task 8.3 — Terminology Statement (Digital Model / Shadow / Twin)

**Status:** FINAL; Track: Team C, Phase 8.

## The statement

Per the **Kritzinger et al. (2018)** taxonomy:

- **Digital Model** — no automatic data connection to physical hardware.
  **This is what this project produces.**
- **Digital Shadow** — automatic one-way connection (physical → digital).
  Begins when ground-test data feeds into the model.
- **Digital Twin** — automatic two-way connection (physical ↔ digital).
  Requires a flown payload streaming telemetry back.

This plan produces a **Digital Model** — a self-contained simulation with no
automatic data connection to physical hardware, since none currently exists.
It will become a **Digital Shadow** once ground-test data feeds into it and
would only become a full **Digital Twin** if a flown payload streamed
telemetry back into it automatically — which is out of scope for this project.

## Reasoning

- The model runs stand-alone on proxy data; nothing is wired to real payloads.
- Training/simulation inputs are provisional (see `hardware_handoff.md`).
- Until hardware exists, "Twin" would be a misnomer — the term is reserved,
  not renamed.

## Consistency check (2026-09)

| Document | Uses correct term? | Notes |
|----------|-------------------|-------|
| `README.md` | ✅ | Title only (`edgeai-digital-model`); no conflicting term |
| `Digital_Model_6Day_Team_Guide.md` | ✅ | Consistently "Digital Model" |
| `task_definition.md` | ✅ | "Digital Model" only |
| `hardware_handoff.md` | ✅ | "Model"/"Digital Shadow" used correctly |
| `final_digital_model_report.md` | ✅ (not yet final) | Re-run on Day 6 before release |

Rule going forward: if any doc says "Digital Twin" before hardware exists,
treat it as an error.