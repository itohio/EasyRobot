---
type: algorithm
status: implemented
tags: [easyrobot, colorscience, spectral]
workstream: W1
module: "[[colorscience]]"
summary: "two element-wise radiometric corrections."
source: [x/math/colorscience/colorscience.go, x/math/colorscience/luminance.go, x/math/colorscience/options.go]
tests: []
test_result: none
complete: yes
correct: unknown
evidence: "no tests; probe (*ColorScience).Calibrate dark=1, light=(11,21,0), meas=(6,11,5) → (0.5,0.5,−4) matches formula (2026-11-09)"
---
# Spectral dark/white calibration and sensor responsivity

**What it does:** two element-wise radiometric corrections.
1. **Reflectance/transmittance normalisation:** `R(λ) = (M − D)/(L − D)` from measurement M, dark D and reference-white/light L.
2. **Sensor responsivity:** `s(λ) = response/reference` (optionally `(response − dark)/reference`), from a known reference source.

**Where:**
- `colorscience.go:290-352` `(*ColorScience).Calibrate(dst *SPD, measurement SPD, optWhitePoint ...WhitePoint)`. D and L are set via `WithDark`/`WithLight` (`options.go:55-77`).
- `luminance.go:60-112` `CalibrateSensorSensitivity`, `:114-155` `CalibrateSensorSensitivityWithDark`, `:158-174` `wavelengthsMatch` (0.1 nm tolerance).

**Inputs → outputs:** all inputs are linearly resampled onto `dst`'s grid (Calibrate) or the reference grid (sensitivity). Division by zero gives 0.

## Completeness review

- Complete: yes for the stated formulas.
  - The `optWhitePoint` parameter is unused (`colorscience.go:346`, "Reserved"). There is no white-tile reflectance correction (a real white standard is about 0.95–0.99, not 1.0).
  - No clamping or validity check when L ≤ D: probe returns −4 for a channel where light < dark.
  - `CalibrateSensorSensitivity` rejects SPDs of different lengths (`luminance.go:99-101`) before it reaches the code that would resample them (`:104-109`), so that resampling branch only runs for equal-length grids with different values.
- Tested: none (`TestCalibrate*` cover `SPD.Calibrate`, the wavelength calibration in [[spectrometer-wavelength-calibration]], not this).
- Correct: unknown; the formulas are standard. Probe values match hand calculation. Not exercised with real spectrometer data offline.

## TODOs

- [ ] Use `optWhitePoint`/white-tile reflectance table for absolute reflectance — `colorscience.go:346` #p2
- [ ] Flag/clamp channels where light−dark ≤ noise floor — `colorscience.go:331-341` #p2
- [ ] Allow differing grid lengths in `CalibrateSensorSensitivity` (drop length check) — `luminance.go:99-101` #p3
- [ ] Unit tests for both calibrations — (none) #p2

## Links

- Module: [[colorscience]]
