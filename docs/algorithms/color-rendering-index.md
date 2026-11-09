---
type: algorithm
status: stub
tags: [easyrobot, colorscience]
workstream: W1
module: "[[colorscience]]"
source: [x/math/colorscience/temperature.go]
tests: []
test_result: none
complete: no
correct: no
evidence: "probe ComputeCRI(D65) → 0, error 'insufficient test color samples' (getCIETestColorSamples returns empty) (2026-11-09)"
---
# Colour rendering index (CIE 13.3 Ra)

**What it does (intended):** general CRI Ra of a light source. It compares the 8 CIE test colour samples (TCS1–8) under the source and under a reference illuminant of the same CCT.

**Where:** `temperature.go:218-315` `(*ColorScience).ComputeCRI`; `:318-345` `computeXYZReflective`; `:348-405` `generatePlanckianSPD`; `:410-425` `getCIETestColorSamples`.

**Inputs → outputs:** `ComputeCRI(spd SPD) (Ra float32, err)`.

## Completeness review

- Complete: **no (stub).** `getCIETestColorSamples` returns `[]SPD{}` (`:424`, "Placeholder"), so every call fails at `:257-259`. Probe: `ComputeCRI(D65)` gives `0, "insufficient test color samples"`. `cmd/spectrometer/PLAN.md:265,279` marks CRI "✅ already implemented"; that is false.
- Further deviations from CIE 13.3, which apply even once TCS data exist:
  - CCT comes from [[correlated-color-temperature]], which mis-handles emissive SPDs and has a wrong locus.
  - The reference above 5000 K is always D65 (`:247-252`, TODO). CIE uses the D-series SPD at the exact CCT.
  - `computeXYZReflective` builds TCS × illuminant and then calls `ComputeXYZ`, which multiplies by the instance illuminant **again** (`:337-344`).
  - ΔE is CIELAB ΔE76 (`:288-292`). CIE 13.3 uses CIE 1964 U\*V\*W\* after a von Kries adaptation in uv. Lab uses the instance white point, not the test or reference white.
  - Planck SPD (`:348-405`): the formula and constants are correct (h, c, k exact SI), but it is evaluated in float64 and then normalised to peak.
- Tested: none.
- Correct: no.

## TODOs

- [ ] Embed CIE 13.3 TCS1–14 reflectances (5 nm) — `temperature.go:410-425` #p2
- [ ] Reference = D-series SPD at CCT for ≥5000 K (needs D-series S0/S1/S2 basis) — `temperature.go:244-253` #p2
- [ ] Remove double illuminant multiplication; use emissive XYZ — `temperature.go:337-344` #p1 #backlog
- [ ] Use U\*V\*W\* + von Kries per CIE 13.3 (or implement TM-30 / CIE 224 Rf instead) — `temperature.go:283-295` #p2
- [ ] Fix PLAN claim "CRI ✅" — `cmd/spectrometer/PLAN.md:265` #p3

## Links

- Module: [[colorscience]]
