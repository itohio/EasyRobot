---
type: algorithm
status: partial
tags: [easyrobot, colorscience, spectral]
workstream: W1
module: "[[colorscience]]"
summary: "- `Peaks`/`Valleys` find strict 3-point local extrema, filtered by a value threshold and a…"
source: [x/math/colorscience/analysis.go]
tests: []
test_result: none
complete: no
correct: no
evidence: "no tests; probe: D65 slice shifted 7 samples, index-wavelengths → DetectCalibrationPoints returns 0 points (2026-11-09)"
---
# Spectral peak/valley detection and automatic calibration-point matching

**What it does:**
- `Peaks`/`Valleys` find strict 3-point local extrema, filtered by a value threshold and a "prominence" (difference from the mean of the two neighbours).
- `DetectCalibrationPoints` tries to pair features of an uncalibrated measured spectrum with a reference spectrum, producing (index, nm) pairs for [[spectrometer-wavelength-calibration]].

**Where:** `analysis.go:34-75` `SPD.Peaks`, `:87-132` `SPD.Valleys`, `:149-223` `SPD.DetectCalibrationPoints`, `:229` deprecated wrapper, `:234-297` `correlationAroundPeak` (windowed Pearson), `:300-360` `detectCalibrationByCorrelation`.

**Inputs → outputs:** `Peaks(threshold, minProminence) []Peak{Index, Wavelength, Value, Prominence}`; `DetectCalibrationPoints(ref SPD, minConfidence) []CalibrationPoint`.

## Completeness review

- Peaks/Valleys: simple and work as written. Production use: `cmd/spectrometer/internal/render/renderer.go:231`.
  - No smoothing and no plateau handling (strict `>`): flat-topped or noisy peaks are missed or multiplied.
  - "Prominence" is not topographic prominence; it is local curvature only.
  - No sub-pixel centroid (parabolic or Gaussian fit), which calibration needs.
- DetectCalibrationPoints: **broken.**
  - The measured SPD is resampled onto the reference grid using its own, still uncalibrated wavelengths (`:165`), which is circular.
  - Peak indices from the **original** measured SPD (`:177`) are used to index the **resampled** one (`:197`), which mixes index spaces.
  - Every measured peak is matched to its single best reference peak with no ordering or uniqueness constraint (`:189-214`).
  - Fallback `detectCalibrationByCorrelation` compares measured and reference at the **same index** (`:331-332`) and emits `(i, refWl[i])` (`:351-354`). That is an identity mapping that assumes calibration already exists.
- Tested: none.
- Correct: no. Probe: measured = D65 samples 107…306 with wavelength row = pixel index (a pure 7-sample shift); `DetectCalibrationPoints(D65, 0.8)` returned **0 points**.

## TODOs

- [ ] Redesign auto-calibration: cross-correlate in index space (or match peak sequences with RANSAC/DTW against a line list) — `analysis.go:149-223` #p2
- [ ] Remove identity-mapping fallback — `analysis.go:300-360` #p1 #backlog
- [ ] Sub-pixel peak centroid + smoothing; true topographic prominence — `analysis.go:49-71` #p2
- [ ] Tests for Peaks/Valleys and calibration detection with synthetic shifted spectra — (none) #p2

## Links

- Module: [[colorscience]]
