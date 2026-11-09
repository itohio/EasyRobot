---
type: algorithm
status: implemented
tags: [easyrobot, colorscience]
workstream: W1
module: "[[colorscience]]"
summary: "pass, weak; probe correct"
source: [x/math/colorscience/adaptation.go, x/math/colorscience/color_types.go]
tests: [x/math/colorscience/colorscience_test.go, x/math/colorscience/color_types_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "tests check only positivity/error; probe Bradford D65→D50 white = 96.422,100.000,82.521 and sRGB red → 43.607,22.250,1.393 (Lindbloom 43.607,22.249,1.393) (2026-11-09)"
---
# Chromatic adaptation (von Kries-type: Bradford, HPE von Kries, CAT02)

**What it does:** linear von Kries adaptation of an XYZ colour from a source white Ws to a destination white Wd. Steps: transform to cone-like space with M, scale each channel by the ratio of Wd to Ws in that space, transform back with M⁻¹.

**Where:** `adaptation.go:24-63` `AdaptXYZ`; matrices `:66-96` (M) and `:99-129` (M⁻¹, hard-coded); method `color_types.go:121` `XYZ.Adapt`.

**Inputs → outputs:** `AdaptXYZ(X,Y,Z, Ws, Wd WhitePoint, method) (X',Y',Z', error)`; `method ∈ {"bradford","von_kries","cat02"}`. Scale-agnostic, but Ws/Wd must share the XYZ scale.

## Completeness review

- Complete: yes. Bradford is the linearised form (the original Bradford has a non-linear blue channel, which is normal to omit). CAT02 is used without the CIECAM02 degree-of-adaptation D (full adaptation, D = 1). No CAT16 and no "XYZ scaling".
- Constants checked against Lindbloom / CIE 159:
  - Bradford M = [0.8951 0.2664 −0.1614; −0.7502 1.7135 0.0367; 0.0389 −0.0685 1.0296] ✓; inverse ✓.
  - HPE (von Kries) [0.38971 0.68898 −0.07868; −0.22981 1.18340 0.04641; 0 0 1] ✓ (4 dp); inverse ✓.
  - CAT02 [0.7328 0.4296 −0.1624; −0.7036 1.6975 0.0061; 0.0030 0.0136 0.9834] ✓; inverse ✓.
- Tested: `TestAdaptXYZ` (only checks X changed and outputs > 0), `TestAdaptXYZInvalidMethod` (error), `TestXYZAdapt` (positivity) → PASS. No value assertions.
- Correct: yes (probe).

| input | method | got | reference |
|---|---|---|---|
| D65 white (95.047, 100, 108.883) → D50 | Bradford | 96.422, 100.000, 82.521 | D50 white |
| D65 white → D50 | von Kries | 96.421, 100.000, 82.521 | D50 white |
| D65 white → D50 | CAT02 | 96.422, 100.000, 82.521 | D50 white |
| sRGB red (41.246, 21.267, 1.933) → D50 | Bradford | 43.607, 22.250, 1.393 | 43.607, 22.249, 1.393 (Lindbloom) |

- Minor: allocates 3 matrices and 5 vectors per call (`:36-60`). Matrices are rebuilt on every call instead of being package-level constants; a 3×3 combined matrix could be precomputed per (Ws, Wd).

## TODOs

- [ ] Value tests (white→white mapping, Lindbloom sRGB red D65→D50) — `colorscience_test.go:326` #p2
- [ ] Precompute matrices / combined 3×3 per white pair; avoid per-call allocation — `adaptation.go:26-60` #p3

## Links

- Module: [[colorscience]]
