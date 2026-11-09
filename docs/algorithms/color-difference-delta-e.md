---
type: algorithm
status: partial
tags: [easyrobot, colorscience]
workstream: W1
module: "[[colorscience]]"
summary: "pass, weak; ΔE2000 wrong"
source: [x/math/colorscience/color_types.go]
tests: [x/math/colorscience/color_types_test.go]
test_result: pass
complete: yes
correct: no
evidence: "TestDeltaE76/76WithKnownValues exact → PASS; DeltaE94/2000 tests only check >0; probe vs Sharma 2005 CIEDE2000 data: errors up to 4.39 (pair 31.9030 → 27.5153) (2026-11-09)"
---
# Colour difference ΔE (CIE76, CIE94, CIEDE2000)

**What it does:** perceptual distance between two CIELAB colours.

**Where:** `color_types.go:133-135` `DeltaE76` (Euclidean); `:140-172` `DeltaE94(lab1, lab2, "graphic"|"textiles")`; `:176-251` `DeltaE2000`; helpers `:255-288`.

**Inputs → outputs:** `LAB, LAB → float32`. Production use: `cmd/spectrometer/internal/render/patch.go:20` calls `DeltaE76` only.

## Completeness review

- Complete: yes, three variants. No CMC l:c.
- **ΔE76**: correct. `TestDeltaE76`, `TestDeltaE76WithKnownValues` assert exact Euclidean values → PASS.
- **ΔE94**: formula correct (S_L = 1, S_C = 1 + K1·C1, S_H = 1 + K2·C1, K_C = K_H = 1, textiles K_L = 2). Probe Sharma pair 1 gives 1.3950, which matches the published ΔE94 for that pair. The test only checks > 0 and graphic ≠ textiles.
- **ΔE2000**: **wrong.** Three defects:
  1. G factor: `0.5·(1 − √(C̄/(7C̄² + 625)))` (`:183`). Correct: `0.5·(1 − √(C̄⁷/(C̄⁷ + 25⁷)))`.
  2. R_C: `2·√(C̄'⁷)/(C̄'² + 625)` (`:238`). Correct: `2·√(C̄'⁷/(C̄'⁷ + 25⁷))`.
  3. Hue angles come from `atan2Deg` in (−180, 180] (`:190-191`, `:286-288`); the standard requires h' ∈ [0, 360). The mean hue h̄' feeds Δθ = 30·exp(−((h̄' − 275)/25)²), which is not periodic, so blue hues (h ≈ 270° ≡ −90°) lose the R_T rotation term. The |Δh| = 180 tie case is also misassigned.
  - `TestDeltaE2000` only asserts 0 for identical colours and > 0 for different ones.
  - Probe vs Sharma, Wu & Dalal (2005) test data:

| pair | got | expected |
|---|---|---|
| (50, 2.6772, −79.7751) / (50, 0, −82.7485) | 2.2936 | 2.0425 |
| (50, 0, 0) / (50, −1, 2) | 2.3556 | 2.3669 |
| (50, 2.49, −0.001) / (50, −2.49, 0.0009) | 7.0394 | 7.1792 |
| (50, 2.5, 0) / (56, −27, −3) | 27.5153 | 31.9030 |
| (50, 2.5, 0) / (50, 3.1736, 0.5854) | 0.9877 | 1.0000 |
| (60.2574, −34.0099, 36.2677) / (60.4626, −34.1751, 39.4387) | 1.1839 | 1.2644 |

- Mixed float32/float64 (`hypot` in float64, rest float32); fine for ΔE precision.

## TODOs

- [ ] Fix CIEDE2000 G and R_C formulas — `color_types.go:183,238` #p1 #backlog
- [ ] Normalise h' to [0,360) before ΔH/h̄' — `color_types.go:190-191,286-288` #p1 #backlog
- [ ] Add Sharma 34-pair table test for ΔE2000 and published ΔE94 values — `color_types_test.go:207` #p2

## Links

- Module: [[colorscience]]
