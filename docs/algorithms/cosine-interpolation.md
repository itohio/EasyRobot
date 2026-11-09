---
type: algorithm
status: tested
tags: [easyrobot, interpolation]
workstream: W3
module: "[[interpolation]]"
summary: "Ease-in/ease-out blend: μ₂=(1−cos πt)/2, returns a(1−μ₂)+bμ₂ (Paul Bourke)."
source: [x/math/interpolation/cosine.go]
tests: [x/math/interpolation/interpolation_helpers_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "TestCosine1D pass"
---
# Cosine interpolation

**What it does:** Ease-in/ease-out blend: μ₂=(1−cos πt)/2, returns a(1−μ₂)+bμ₂ (Paul Bourke). Zero slope at both ends — useful for smooth joint moves / gait phase blending.

**Where:** `x/math/interpolation/cosine.go:5` (`Cosine1D`).

**Inputs → outputs:** float32 a, b, t∈[0,1] → float32.

## Completeness review

- Complete: yes.
- Tested: TestCosine1D (`interpolation_helpers_test.go`) checks endpoint/midpoint values. Run: go test -count=1 -json ./x/math/{colorscience,learn,grid,interpolation}/... (2026-11-09, slice lead; raw: tmp/easyrobot/filters/test-2026-11-09.json) → pass.
- Correct: yes for tested values. Note: uses `math.Cos` in **float64** (`cosine.go:6`) while the rest of the package uses `math32` — costs a float64 software path on FPU-less/single-precision MCUs (ESP32 Xtensa LX6 has single-precision FPU only).

## TODOs

- [ ] Use math32.Cos to stay float32 — x/math/interpolation/cosine.go:6 #p3

## Links

- Module: [[interpolation]]
