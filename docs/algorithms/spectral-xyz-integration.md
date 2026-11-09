---
type: algorithm
status: partial
tags: [easyrobot, colorscience, spectral]
workstream: W1
module: "[[colorscience]]"
summary: "fail (TestComputeXYZ NaN; 2 test bugs)"
source: [x/math/colorscience/colorscience.go, x/math/colorscience/integration.go, x/math/colorscience/spd.go, x/math/colorscience/data_loader.go, x/math/colorscience/loader.go, x/math/colorscience/data/]
tests: [x/math/colorscience/colorscience_test.go]
test_result: fail
complete: no
correct: no
evidence: "go test -run 'TestComputeXYZ$' → FAIL (Z=NaN); probe: 2° D65/D50/A white XYZ within 0.01 of CIE; 10° Z=NaN (2026-11-09)"
---
# Spectral → CIE XYZ integration

**What it does:** integrates a sample spectrum against CIE colour-matching functions (CMF) and a reference illuminant to get tristimulus XYZ, normalised so a perfect reflector has Y = 100 (reflective colorimetry).

**Where:**
- `colorscience.go:164-265` `(*ColorScience).ComputeXYZ`; `:26-74` `New` (loads CMF+illuminant); `:78-125` `matchCMFAndIlluminantWavelengths`; `:128-161` `interpolateCMF`.
- `integration.go:11-42` `IntegrationKernel` (trapezoid weights), `:45-56` `Integrate`.
- `spd.go:78-134` `SPD.Interpolate` (linear, constant end extrapolation); `spd.go:678-731` `parseCSV`.
- `data_loader.go:47-133` loads embedded CSVs (`//go:embed data/*.csv`): CIE 1931 2°, CIE 1964 10° CMF (360–830 nm, 1 nm), illuminants A, D50, D65 (300–830 nm, 1 nm). `loader.go:13-20` builds them in `init()` (panics on failure).

**Inputs → outputs:** `ComputeXYZ(m matTypes.Matrix) (XYZ, error)`; `m` is 1 row (values on the CMF grid) or 2 rows (wavelengths nm, values; resampled to the CMF grid). Output XYZ on the 0–100 scale: `X = k·Σ w_i S_i E_i x̄_i`, `k = 100/Σ w_i E_i ȳ_i`, `w` = trapezoid kernel. The input is always treated as **reflectance/transmittance** (multiplied by the illuminant).

At construction the CMF and illuminant are put on one grid: whichever has more samples wins (`colorscience.go:110`). Default D65 (531 pts) wins over CMF (471 pts), so the CMF is resampled to 300–830 nm with constant extrapolation 300–359 nm (CMF ≈ 1e-4 there, negligible).

## Completeness review

- Complete: no.
  - **10° observer (the default) gives Z = NaN.** `data/CIE_xyz_1964_10deg.csv` has `NaN` in the z̄ column from 560 nm to 830 nm (271 rows, e.g. line 201 `560,0.705224,0.99734,NaN`). `parseCSV` passes them through because `strconv.ParseFloat("NaN")` succeeds (`spd.go:722`), and NaN spreads through the dot product at `colorscience.go:218-254`. CIE defines z̄₁₀ = 0 there. Probe: default `New()` with a flat reflector gives XYZ = (94.811, 100.000, NaN), and Lab b = NaN.
  - Emissive (light-source) spectra have no path: everything is multiplied by the illuminant (`:208-213`). `ComputeLuminance` (`luminance.go:33`) therefore returns relative Y, never cd/m², whatever its doc says.
  - When the measured range is narrower than the CMF grid, it is extended flat (`spd.go:115-120`). A 400–700 nm measurement is padded with its edge values out to 300–830 nm, which biases X and Z. The usual practice is zero padding or truncating the integration range.
  - 1-row input is checked only by length, not by wavelength values (`:183-186`).
  - Only A, D50, D65 have SPDs. `WithIlluminant("F2")` fails in `New`, although `getWhitePointForIlluminant` lists 27 names (`white_points.go:70-139`).
- Tested:
  - `TestIntegrationKernel` checks kernel values 5/10/5 → PASS. `TestIntegrate` checks ∫1 over [0,4] = 4 → PASS. `TestSPDInterpolate` checks 1 < v(450) < 2 → PASS. All three run only with `-run`; the full-package run never reaches them (see below).
  - `TestComputeXYZ` → **FAIL** "XYZ values should not be NaN" (`colorscience_test.go:123`). The implementation/data are at fault (10° CSV NaN), not the test.
  - `TestComputeXYZMismatchedLengths` → **FAIL by panic**. The test builds its input with `NewSPD(wl[3], vals[2])`, and `NewSPD` panics on the length mismatch (`spd.go:41`) before `ComputeXYZ` is ever called. **Test bug**: it should build a ragged raw `mat.Matrix` literal. The check at `colorscience.go:192-194` is reachable only through such ragged literals; `mat.New` rows are always equal length. The panic aborts the test binary, so **the 20 tests after it in `colorscience_test.go` never run** in a full-package run. The recorded "28 pass / 3 fail" covers only 31 of 51 tests. Run separately (`go test -count=1 -v -run 'TestCalibrate|TestXYZToLAB$|…|TestSPDInterpolate' ./x/math/colorscience/`), all 20 → PASS.
  - `TestNewWithCustomIlluminant` → **FAIL**. It asserts that the stored illuminant matrix equals the 5-point input. By design `New` resamples it onto the 471-point CMF grid (`colorscience.go:110-113`), so the actual matrix is 2×471. **Test is stale/wrong.** Design caveat: values outside 400–600 nm are flat-extrapolated to 1.0, which is physically dubious.
- Correct: partial. Probe (`tmp/easyrobot/filters/colorsci/main.go`) integrates a flat reflector:

| config | got | CIE reference |
|---|---|---|
| D65 / 2° | 95.049, 100.000, 108.892 | 95.047, 100, 108.883 |
| D50 / 2° | 96.422, 100.000, 82.501 | 96.422, 100, 82.521 |
| A / 2° | 109.851, 100.000, 35.586 | 109.850, 100, 35.585 |
| D65 / 10° | 94.811, 100.000, **NaN** | 94.811, 100, 107.304 |

The 2° path is correct to about 0.02 %. The 10° path is broken.

## TODOs

- [ ] Map NaN→0 in CMF loading or clean `CIE_xyz_1964_10deg.csv` (z̄ ≥ 560 nm); default observer currently yields Z=NaN — `spd.go:722`, `data/CIE_xyz_1964_10deg.csv:201-471` #p1 #backlog
- [ ] Fix `TestComputeXYZMismatchedLengths` to build a raw matrix (panic masks 20 tests) — `colorscience_test.go:143` #p1 #backlog
- [ ] Update `TestNewWithCustomIlluminant` to assert resampled grid/values instead of matrix identity — `colorscience_test.go:43` #p2
- [ ] Add emissive mode (k from CMF only, or absolute with K_m=683 lm/W) so light sources and luminance work — `colorscience.go:208-262` #p1 #backlog
- [ ] Zero-pad or clip instead of flat extrapolation when resampling measured spectra — `spd.go:115-120` #p2
- [ ] Value test for ComputeXYZ against CIE white points (2° and 10°) — `colorscience_test.go:110` #p2
- [ ] Ship more illuminant SPDs (D55, D75, F-series, LED) or reject names without SPD consistently — `data_loader.go:100-133` #p3

## Links

- Module: [[colorscience]]
