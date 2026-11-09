---
type: algorithm
status: partial
tags: [easyrobot, colorscience]
workstream: W1
module: "[[colorscience]]"
summary: "pass, weak; XYZ scale bug"
source: [x/math/colorscience/conversions.go, x/math/colorscience/color_types.go]
tests: [x/math/colorscience/colorscience_test.go, x/math/colorscience/color_types_test.go]
test_result: pass
complete: yes
correct: no
evidence: "tests only check range/round-trip ±0.1; probe XYZToRGB(0.95047,1,1.08883)=(1,1,0.9999) ok, XYZToRGB(20,21,22)=(1,1,1) saturated (0–100 input) (2026-11-09)"
---
# sRGB ↔ XYZ (D65)

**What it does:** IEC 61966-2-1 sRGB. Linear matrix plus the piecewise gamma (encode threshold 0.0031308, decode threshold 0.04045, exponent 2.4).

**Where:** `conversions.go:71-102` `XYZToRGB(X,Y,Z, out255)`, `:105-136` `RGBToXYZ(r,g,b)`; methods `color_types.go:83,103`.

**Inputs → outputs:**
- `XYZToRGB` expects XYZ on the **0–1** scale (Y_white = 1). It returns RGB in 0–1, or 0–255 if `out255`, clamped.
- `RGBToXYZ` accepts 0–1 or 0–255 and decides which by heuristic: values are divided by 255 if any component > 1 (`:107`).

The matrices are the standard 4-decimal sRGB/D65 pair (3.2406, −1.5372, … / 0.4124, 0.3576, …). They are correct.

## Completeness review

- Complete: yes for sRGB. No other RGB spaces (Adobe, P3, linear-sRGB output) and no out-of-gamut handling beyond clamping.
- Tested:
  - `TestXYZToRGB` passes D65 white on the 0–100 scale and expects ≈1. It passes only because everything clamps to 1, so the test does not detect the scale bug.
  - `TestRGBToXYZ` and `TestColorConversionRoundTripRGB`: round trip ±0.1.
  - Remaining tests only check ranges or positivity.
  - All PASS (`TestXYZToRGB`, `TestRGBToXYZ`, `TestRGBToXYZ255` only via `-run`).
- Correct: **no as a package contract.**
  - `ColorScience.ComputeXYZ` returns XYZ on 0–100, while `XYZToRGB` expects 0–1 and no scaling sits between them. Probe: `XYZToRGB(20,21,22)` gives (1, 1, 1); the right answer is ≈0.49 grey after /100. This hits `cmd/spectrometer/measure/patches.go:140` (`measuredXYZ.ToRGB(true)` on `ComputeXYZ` output) and `cmd/spectrometer/internal/render/renderer.go:372` (measured XYZ). `renderer.go:358` does the same for the reference XYZ, whose scale depends on the patch source. [INFERENCE] Measured swatches render saturated (not run).
  - `RGBToXYZ` heuristic: 8-bit (1,0,0) or (1,1,1) is read as full-intensity 0–1 colour (probe `RGBToXYZ(1,0,0)` = (0.4124, 0.2126, 0.0193)).
  - The formulas themselves are correct on the 0–1 scale: `XYZToRGB(0.95047, 1, 1.08883)` = (1.0000, 1.0000, 0.9999).

## TODOs

- [ ] Make XYZ scale explicit (normalise by Y_white/100 in `XYZ.ToRGB` or have `ComputeXYZ` offer 0–1) — `conversions.go:71`, `cmd/spectrometer/measure/patches.go:140` #p1 #backlog
- [ ] Replace the `>1 ⇒ /255` heuristic with explicit 8-bit vs float API — `conversions.go:107` #p2
- [ ] Value tests (e.g. XYZ(0.4124,0.2126,0.0193) → (1,0,0); mid-grey) — `colorscience_test.go:293` #p2

## Links

- Module: [[colorscience]]
