---
type: module
group: "Math and maps"
status: partial
tags: [easyrobot, colorscience, spectral]
workstream: W1
packages: [x/math/colorscience]
test_result: fail
---
# colorscience

**Purpose:** spectral colorimetry for the spectrometer/colorimeter tools (`cmd/spectrometer`, `cmd/cr30`). It covers SPD containers, CIE CMF/illuminant data, spectrum→XYZ, Lab/sRGB, chromatic adaptation, ΔE, CCT/CRI, pixel→wavelength calibration, dark/white correction, multi-channel (AS734x-style) SPD reconstruction, peak analysis and a spectral library.

**Packages:** `x/math/colorscience`. No build tags. 15 Go source files, 3,761 LOC; 2 test files, 731 LOC (51 tests); `SPEC.md` 1,119 lines.

| File | LOC | Content |
|---|---|---|
| spd.go | 731 | `SPD`, `ObserverCMF`, `Interpolate`, `Calibrate` (wavelength), `Reconstruct`, `ReconstructWeighted`, `parseCSV` |
| temperature.go | 425 | CCT/Duv, CRI (stub), Planck SPD |
| analysis.go | 360 | Peaks/Valleys, DetectCalibrationPoints |
| reconstruction.go | 355 | filter models, channel→response, reconstruct wrappers |
| colorscience.go | 352 | `ColorScience`, `New`, `ComputeXYZ`, dark/light `Calibrate` |
| color_types.go | 288 | `XYZ`/`LAB`/`RGB` types, ΔE76/94/2000 |
| database.go | 253 | spectral library |
| data_loader.go | 182 | `//go:embed data/*.csv` loaders |
| luminance.go | 174 | `Luminance`, `ComputeLuminance`, sensor responsivity |
| conversions.go | 149 | XYZ↔Lab, XYZ↔sRGB |
| white_points.go | 139 | 27 white-point constants |
| adaptation.go | 129 | Bradford / von Kries / CAT02 |
| options.go | 77 | functional options |
| integration.go | 56 | trapezoid kernel |
| calibration.go | 55 | `FindWavelengthIndex` |
| loader.go | 36 | package `init()` data load |

Embedded data (`data/`, ≈57 KB): CIE 1931 2° and CIE 1964 10° CMF, 360–830 nm at 1 nm (**the 10° file has NaN z̄ from 560 nm**). Illuminants A, D50, D65 at 300–830 nm.

## Public API

- **Instance:** `New(opts...) (*ColorScience, error)`. Options: `WithObserver(Observer2Deg|Observer10Deg)` (default 10°), `WithIlluminant("A"|"D50"|"D65")` (default D65, auto white point), `WithIlluminantSPD(m)`, `WithWhitePoint(wp)`, `WithDark(m)`, `WithLight(m)`. Methods: `ComputeXYZ(m) (XYZ, error)`, `ComputeLuminance`, `ComputeColorTemperature(SPD)`, `ComputeCRI(SPD)`, `Calibrate(dst *SPD, meas SPD, wp...)`, `WhitePoint()`.
- **Data:** `LoadCMF(obs)`, `LoadIlluminantSPD(name)`, `AvailableIlluminants()`, `NewDataLoader()`.
- **SPD** (`struct{ matTypes.Matrix }`, row 0 = nm, row 1 = values): `NewSPD(wl, v)` (panics on length mismatch), `Wavelengths`, `Values`, `Len`, `Interpolate`, `Calibrate(pairs...)`, `Reconstruct`, `ReconstructWeighted`, `ReconstructFromChannels`, `Peaks`, `Valleys`, `DetectCalibrationPoints`.
- **Colour values:** `XYZ`, `LAB`, `RGB` (`vec.Vector3D` aliases) with `ToLAB/ToRGB/ToXYZ/Adapt/Luminance`; free functions `XYZToLAB`, `LABToXYZ`, `XYZToRGB`, `RGBToXYZ`, `RGBToLAB`, `LABToRGB`, `AdaptXYZ`, `DeltaE76/94/2000`.
- **Misc:** `IntegrationKernel`, `Integrate`, `FindWavelengthIndex`, `CalibrateSensorSensitivity[WithDark]`, `FilterSPDFromChannel`, `SensorChannelsToResponses`, `ReconstructSPDFromChannels`, `ReconstructSPDWithConstraints`, `NewSpectrumDatabase()` (`SpectrumDatabase` interface).

**Composition:** raw spectrometer counts → `SPD.Calibrate` (pixel→nm) → `ColorScience.Calibrate` (dark/white → reflectance) → `ComputeXYZ` → `ToLAB(cs.WhitePoint())` / `ToRGB` → `DeltaE76` against reference patches. Filter-channel sensors go through `ReconstructSPDFromChannels` → `ComputeXYZ`.

Trivial helpers kept in this note:
- `Luminance` returns Y.
- `FindWavelengthIndex` is a binary search returning nearest/exact index.
- `SPD.Interpolate` is a linear O(N·M) scan with **constant end extrapolation** (`spd.go:78-134`). `TestSPDInterpolate` (weak) passes.
- `IntegrationKernel` is the trapezoid kernel; it is checked in [[spectral-xyz-integration]].

## Algorithms

<!-- vault:algorithms h=1d0e21ff3500 -->

Counts: 2 tested · 3 implemented · 6 partial · 1 stub · 0 unverified (12 algorithms)

| Algorithm | Status | Tests | Summary |
|---|---|---|---|
| [[cie-lab-conversion]] | tested | pass | pass (`-run`) |
| [[spectrometer-wavelength-calibration]] | tested | pass | pass (`-run` only) |
| [[chromatic-adaptation-transforms]] | implemented | pass | pass, weak; probe correct |
| [[spectral-radiometric-calibration]] | implemented | none | two element-wise radiometric corrections. |
| [[spectrum-database-matching]] | implemented | none | an in-memory spectral library with two ranked searches. |
| [[color-difference-delta-e]] | partial | pass | pass, weak; ΔE2000 wrong |
| [[correlated-color-temperature]] | partial | none | estimates CCT (K) and Duv (signed distance from the Planckian locus in CIE 1960 uv) for a spectrum. |
| [[multisensor-spd-reconstruction]] | partial | none | recovers a spectrum X (N wavelengths) from M filtered photodetector readings R, given channel… |
| [[spectral-peak-detection-and-auto-calibration]] | partial | none | - `Peaks`/`Valleys` find strict 3-point local extrema, filtered by a value threshold and a… |
| [[spectral-xyz-integration]] | partial | fail | fail (TestComputeXYZ NaN; 2 test bugs) |
| [[srgb-xyz-conversion]] | partial | pass | pass, weak; XYZ scale bug |
| [[color-rendering-index]] | stub | none |  |

<!-- /vault:algorithms -->

## Build targets and dependencies

- Imports [[mat-vec]] (`x/math/mat`, `x/math/vec` and their `types`), [[interpolation]] (`Lerp`, `CubicCatmulRomSpline1D`), `x/math` (`Clamp`), `github.com/chewxy/math32`. Stdlib: `embed`, `encoding/csv`, `strconv`, `sort`, `fmt`, `math`. Pure Go, no CGO.
- **Embedded/TinyGo suitability: poor as-is.**
  - `init()` parses ≈57 KB of CSV with `encoding/csv` + `strconv` into float32 matrices at program start and panics on failure (`loader.go:13-20`). Every importer pays for this, including firmware that only wants `XYZToLAB`.
  - Each `New` clones and resamples CMF and illuminant (≈2×4×531 float32).
  - `ComputeXYZ` allocates 6 N-length vectors per call.
  - `AdaptXYZ` builds matrices per call.
  - The damped reconstruction inverts an N×N matrix (N = number of wavelengths).
  - Mostly float32; CCT/Planck use float64 `math`.
  - No goroutines. Reflection only via `fmt` (errors). Pervasive `.(vec.Vector)` / `.(mat.Matrix)` assertions panic on other `matTypes` implementations.
  - [INFERENCE] TinyGo supports `//go:embed`; not compiled here (no TinyGo per contract).
  - For MCU use (W5/K230→ESP32 colour features): split pure conversions (Lab/sRGB/adaptation/ΔE) from the data-loading part, and precompute CMF×illuminant weight vectors (3×N) offline as constants.

## Tests and evidence

- **Recorded run (slice lead, 2026-11-09):** `go test -count=1 -json ./x/math/colorscience/...` → FAIL, 28 pass / 3 fail, same as the baseline. **Only 31 of 51 tests ran.** `TestComputeXYZMismatchedLengths` panics in `NewSPD` (`spd.go:41`) and aborts the binary, so the 20 later tests in `colorscience_test.go` never execute.
- `GOTOOLCHAIN=local GOPROXY=off GOFLAGS=-mod=readonly go test -count=1 -v -run 'TestNewWithCustomIlluminant|TestComputeXYZ$|TestComputeXYZMismatchedLengths' ./x/math/colorscience/` → 3 FAIL:
  - `TestNewWithCustomIlluminant`: **test wrong**. It expects the 5-point matrix, but `New` resamples to the 471-point CMF grid by design (`colorscience.go:110-113`).
  - `TestComputeXYZ`: **implementation/data wrong**. Z = NaN from NaN z̄ in `data/CIE_xyz_1964_10deg.csv:201-471`, passed through by `spd.go:722`.
  - `TestComputeXYZMismatchedLengths`: **test wrong**. It panics in its own setup via `NewSPD` (`colorscience_test.go:143`).
- `… go test -count=1 -v -run 'TestCalibrate|TestXYZToLAB$|TestLABToXYZ$|TestXYZToRGB$|TestRGBToXYZ$|TestRGBToXYZ255|TestAdaptXYZ|TestIntegrationKernel|TestIntegrate$|TestSPDInterpolate|TestSPDMatrix|TestObserverCMFMatrix' ./x/math/colorscience/` → the 20 masked tests all PASS.
- Test quality: most color_types tests check only ranges or positivity. Value checks exist only for Lab white (→ 100, 0, 0), ΔE76, trapezoid kernel and wavelength calibration. No tests for CCT, CRI, reconstruction, analysis, database, sensor calibration.
- Probe (`GOTOOLCHAIN=local GOPROXY=off GOFLAGS=-mod=mod go run .`, outside the repo):
  - 2° white XYZ matches CIE to ≈0.02 %; 10° gives Z = NaN.
  - Lab(D65 white) = (100, 0, 0).
  - Bradford/CAT02/von Kries map D65 → D50 white exactly; Lindbloom red ✓.
  - ΔE2000 is off by up to 4.4 on Sharma data.
  - CCT: McCamy branch ✓, fallback search and Duv wrong, emissive input wrong.
  - CRI always errors.
  - Reconstruction: undamped fails for M < N.
  - Auto-calibration returns 0 points.
- Reference constants checked:
  - Lab threshold (6/29)³ and inverse 6/29 ✓.
  - McCamy n = (x − 0.3320)/(0.1858 − y) and polynomial ✓.
  - Bradford, HPE, CAT02 matrices and inverses ✓.
  - sRGB matrices and gamma ✓.
  - White points:
    - D65/2°, D50/2°, A, B, C, E, F1–F12 ✓ (2° values).
    - **D55/10° and D75/10° hold the 2° values** (`white_points.go:13,17`).
    - D65/10° Z 107.32 vs CIE 107.304; D50/10° Z 81.43 vs 81.427.
    - The LED white points are rough guesses (LED_CW_6500K = D65/2°).

**Users:**
- `cmd/cr30/main.go:125,298-302`: `New()` → `ComputeXYZ` → `ToLAB`.
- `cmd/spectrometer/measure/patches.go:130-153`: `New()`, `ComputeXYZ`, `ToLAB`, `ToRGB(true)`, `DeltaE76`.
- `cmd/spectrometer/measure/measure.go:465`: `WhitePointD65_10`.
- `cmd/spectrometer/internal/render/{patch,renderer,colorimetry}.go`: types, `DeltaE76`, `Peaks`, `ToRGB`.
- `cmd/spectrometer/internal/{acquirer,calibrate}`: `SPD`/`NewSPD`.
- Both `New()` callers use the 10° default, so **their Z and Lab b are NaN**, and sRGB swatches saturate (scale bug). `cmd/spectrometer/PLAN.md` claims CRI is ✅; it is not.

## API notes (ease of use, composability)

- **Good:**
  - Functional options with sane defaults.
  - Small value types (`XYZ`, `LAB`, `RGB`) with chainable `ToX` methods.
  - SPD embeds `matTypes.Matrix`, so it composes with the [[mat-vec]] stack.
  - Errors rather than panics on most paths.
- **Unit and scale ambiguity is the main hazard.**
  - `ComputeXYZ` returns XYZ with Y = 100, the white points are on 0–100, but `XYZToRGB` expects 0–1 and `RGBToXYZ` guesses 0–1 vs 0–255.
  - No type distinguishes a reflectance from an emissive SPD. `ComputeXYZ`, `ComputeColorTemperature` and `ComputeCRI` all silently multiply by the illuminant.
  - Proposal: an explicit `Kind` (reflective/emissive), or separate `ReflectiveXYZ` and `EmissiveXYZ`; a normalised `XYZ` convention (Y = 1); 8-bit RGB as a separate type.
- **Shape-by-convention matrices.** SPD = 2 rows, CMF = 4 rows, and a 1-row/2-row overload in `ComputeXYZ`. These are easy to misuse. `NewSPD` panics where everything else returns errors.
- **In-place vs copy is inconsistent.**
  - `SPD.Calibrate` returns a new SPD, while `Reconstruct*` and `ColorScience.Calibrate` write into the receiver or `dst`.
  - `ReconstructWeighted` mutates the caller's weights.
  - `InMemoryDatabase.Add` aliases the caller's matrix.
  - Value-receiver `SPD` methods that mutate the shared matrix are surprising.
- Stringly-typed enums (`AdaptationMethod` string, `"graphic"|"textiles"`, `FilterShape int` next to a typed `FilterShapeModel`). Unused parameters (`optWhitePoint`, `smoothnessWeight`) signal unfinished API.
- Package-global `init()` data load couples cheap conversions to heavy data. This hurts embedded reuse and test isolation.
- Deprecated aliases are still exported (`DetectCalibrationPoints` func, `GaussianSPDFromChannel`).

## TODOs

Algorithm defects are tracked in the algorithm notes (table above) and rolled up in [[todo]]; this list holds module-level items only.

- [ ] Fix Planckian locus branches / real Robertson — `temperature.go:97-205` #p1 #backlog
- [ ] Split pure conversions from embedded-data loading (lazy load, no init panic) — `loader.go:13-20` #p2
- [ ] Value tests across CCT, adaptation, sRGB, reconstruction, database — `colorscience_test.go`, `color_types_test.go` #p2

## Inferred objectives (proposals)

- **proposal:** make `colorscience` the colour backend for a low-cost multi-channel spectral sensor (AS7341/AS7343, implied by `SensorChannel` docs) on ESP32. Sensor readings → reconstructed SPD or direct channel→XYZ matrix → Lab/CCT, with offline-fitted matrices instead of on-device SPD reconstruction.
- **proposal:** feed W5 (K230 camera features → ESP32). Camera RGB → XYZ via a calibrated 3×3 or 3×N matrix plus Bradford adaptation to estimate scene white/CCT for white balance and colour-based object features; reuse `AdaptXYZ`/`XYZToLAB`/ΔE as float32 MCU kernels.
- **proposal:** a reliable lab colorimeter path (CR30 + spectrometer): 2°/10° XYZ, Lab, ΔE00 against reference patches, validated against published CIE/Sharma data in tests.
