---
type: algorithm
status: tested
tags: [easyrobot, colorscience, spectral]
workstream: W1
module: "[[colorscience]]"
summary: "pass (`-run` only)"
source: [x/math/colorscience/spd.go, x/math/colorscience/calibration.go]
tests: [x/math/colorscience/colorscience_test.go]
test_result: pass
complete: yes
correct: unknown
evidence: "go test -count=1 -v -run 'TestCalibrate' ./x/math/colorscience/ → 8 PASS (2026-11-09); masked (not executed) in full-package run by earlier panic"
---
# Spectrometer pixel→wavelength calibration (Catmull-Rom)

**What it does:** maps detector pixel indices to wavelengths from ≥ 2 known (index, nm) reference lines. Between points it uses a Catmull-Rom spline; outside them, linear extrapolation from the end segments.

**Where:** `spd.go:215-374` `CalibrationPoint`, `SPD.Calibrate(pairs ...float32) (SPD, error)`; spline from `x/math/interpolation` `CubicCatmulRomSpline1D` (see [[interpolation]]). Helper `calibration.go:12-55` `FindWavelengthIndex` (binary search for exact/nearest index).

**Inputs → outputs:** SPD (row 0 ignored, row 1 values) and pairs `idx0, nm0, idx1, nm1, …`. Returns a new SPD with row 0 replaced and values unchanged. Errors: nil SPD, empty or odd pairs, < 2 points, index out of range, duplicate index (`:233-280`).

## Completeness review

- Complete: yes for its scope.
- Tested: `TestCalibrate` (3 points: exact at the points, interior values between them, strictly monotone, values preserved), `TestCalibrateTwoPoints`, plus 6 error-path tests → all PASS with `-run`. They never execute in the full-package run because `TestComputeXYZMismatchedLengths` panics first (see [[spectral-xyz-integration]]).
- Correct: unknown for real use.
  - The spline parameter is t = (i − idx0)/(idx1 − idx0) per segment, and the tangents come from neighbouring **wavelengths only** (`:313-348`). They ignore index spacing, so unevenly spaced calibration lines give tangent mismatch and possible non-monotonicity. The end segments duplicate the endpoint (p1 = p2, p4 = p3).
  - The usual spectrometer practice is a least-squares polynomial (2nd–3rd order) fit, which also averages line-position noise. Catmull-Rom interpolates exactly and so propagates centroid noise.
  - Pair indices are passed as float32 and truncated with `int()` (`:252`), so sub-pixel peak centroids are lost.
  - Not checked against a reference lamp (Hg/Ar lines).

## TODOs

- [ ] Accept sub-pixel indices (float) and offer least-squares polynomial fit option — `spd.go:252,313-348` #p2
- [ ] Use non-uniform Catmull-Rom (index-spaced tangents) or monotone cubic — `spd.go:327-347` #p2
- [ ] Test with uneven calibration points / known Hg-Ar line set — `colorscience_test.go:150` #p2

## Links

- Module: [[colorscience]]
