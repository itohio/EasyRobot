---
type: algorithm
status: tested
tags: [easyrobot, interpolation, bezier, trajectory]
workstream: W3
module: "[[interpolation]]"
summary: "Evaluates a cubic Bézier (Bernstein form) from 4 control points at parameter t…"
source: [x/math/interpolation/bezier.go]
tests: [x/math/interpolation/interpolation_helpers_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "TestBezier1d, TestBezierVector pass"
---
# Cubic Bézier curve

**What it does:** Evaluates a cubic Bézier (Bernstein form) from 4 control points at parameter t: (1−t)³P₁+3t(1−t)²P₂+3t²(1−t)P₃+t³P₄. Candidate for swing-leg foot trajectories (workstream W3).

**Where:** `x/math/interpolation/bezier.go:3` (`Bezier1d`), `:13` (`Bezier2d` [2]float32), `:28` (`Bezier3d` [3]float32), `:44` (`Bezier` []float32, allocates result `:57`, nil on length mismatch).

**Inputs → outputs:** control points float32, t∈[0,1] (unclamped) → point.

## Completeness review

- Complete: yes for evaluation. Missing: derivative/velocity, de Casteljau split, arc-length reparametrisation, `dst` variant of `Bezier` (allocates per call).
- Tested: TestBezier1d, TestBezierVector — endpoint and midpoint values. Run: go test -count=1 -json ./x/math/{colorscience,learn,grid,interpolation}/... (2026-11-09, slice lead; raw: tmp/easyrobot/filters/test-2026-11-09.json) → pass.
- Correct: yes for tested values (Bernstein formula matches).

## TODOs

- [ ] Derivative (velocity/accel) for trajectory timing — x/math/interpolation/bezier.go #p2
- [ ] Non-allocating `Bezier(dst, ...)` — x/math/interpolation/bezier.go:57 #p3

## Links

- Module: [[interpolation]]
