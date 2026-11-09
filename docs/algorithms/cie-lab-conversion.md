---
type: algorithm
status: tested
tags: [easyrobot, colorscience]
workstream: W1
module: "[[colorscience]]"
summary: "pass (`-run`)"
source: [x/math/colorscience/conversions.go, x/math/colorscience/color_types.go, x/math/colorscience/white_points.go]
tests: [x/math/colorscience/colorscience_test.go, x/math/colorscience/color_types_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "TestXYZToLAB (white→100,0,0 ±0.1), TestLABToXYZ round-trip ±0.1 → PASS (via -run; masked in full run); probe Lab(D65 white)=(100,0,0), XYZ=0.8→L=7.2264 linear branch (2026-11-09)"
---
# CIE XYZ ↔ CIELAB (1976 L\*a\*b\*)

**What it does:** converts XYZ to CIELAB relative to a reference white, and back again.

**Where:** `conversions.go:9-40` `XYZToLAB`, `:42-68` `LABToXYZ`, `:139-149` `RGBToLAB`/`LABToRGB` chains. Methods: `color_types.go:75` `XYZ.ToLAB`, `:90` `LAB.ToXYZ`. White points: `white_points.go:9-61`.

**Inputs → outputs:** `XYZToLAB(X,Y,Z float32, wp WhitePoint) (L,a,b)`. XYZ must be on the **same scale as the white point** (the package constants use Y = 100). f(t) = t^{1/3} if t > (6/29)³, else t/(3·(6/29)²) + 4/29. L = 116 f_y − 16, a = 500(f_x − f_y), b = 200(f_y − f_z). The inverse switches at f > 6/29 and uses 3δ²(f − 4/29). Both match CIE 15:2004.

## Completeness review

- Complete: yes for 1976 CIELAB. No LCh, Luv or Oklab.
- Tested:
  - `TestXYZToLAB` (`colorscience_test.go:271`): XYZ = D65/10 white gives (100, 0, 0) ±0.1.
  - `TestLABToXYZ` (`:282`): round trip of (50, 60, 70) within ±0.1.
  - `TestColorConversionRoundTrip` (`color_types_test.go:222`): round trip.
  - The first two pass only when run via `-run` (masked in the full run by the panic described in [[colorscience]]). The third passes in the recorded run.
- Correct: yes. Probe: `XYZToLAB(95.047,100,108.883, D65_2)` = (100.0000, 0.0000, 0.0000). XYZ = 0.8 (linear branch) gives L = 7.2264, matching 116·(0.008/(3δ²) + 4/29) − 16 = 7.226. The equations match the CIE definition exactly.
- Caveats:
  - The white-point constants are partly wrong (see [[colorscience]]): `WhitePointD55_10`/`D75_10` hold the 2° values (`white_points.go:13,17`), and `D65_10` Z = 107.32 vs CIE 107.304 (`:15`).
  - Callers that pass 0–1 XYZ with these 0–100 white points get very low L (seen in tests `color_types_test.go:56`). No scale guard.

## TODOs

- [ ] Fix `WhitePointD55_10` (95.799,100,90.926) and `WhitePointD75_10` (94.416,100,120.641); add `WhitePointD55_2` — `white_points.go:13,17,79` #p1 #backlog
- [ ] Use CIE value Z=107.304 for D65/10 (and 81.427 for D50/10) — `white_points.go:11,15` #p3
- [ ] Document/enforce XYZ scale (0–100) for Lab/RGB conversions — `conversions.go:9` #p3

## Links

- Module: [[colorscience]]
