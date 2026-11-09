---
type: algorithm
status: tested
tags: [easyrobot, interpolation, spline]
workstream: W3
module: "[[interpolation]]"
summary: "Local 4-point cubic segment interpolation between p2 and p3 (p1, p4 neighbours), ported from Paul…"
source: [x/math/interpolation/spline.go]
tests: [x/math/interpolation/interpolation_helpers_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "TestCubicSpline1DProperties, TestCubicCatmulRomSpline1D pass"
---
# Cubic 4-point spline (Bourke cubic, Catmull-Rom)

**What it does:** Local 4-point cubic segment interpolation between p2 and p3 (p1, p4 neighbours), ported from Paul Bourke. Not a global (natural/clamped) spline: no tridiagonal solve, no boundary conditions — caller supplies neighbours.

**Where:** `x/math/interpolation/spline.go:6` (`CubicSpline1D` — Bourke "cubic": a0=p4−p3−p1+p2, C¹ not guaranteed), `:17` (`CubicCatmulRomSpline1D` — uniform Catmull-Rom, tangent (p3−p1)/2, C¹). Catmull-Rom is duplicated in `upsample.go:152-154` and `matrix_upsample.go:171` (`cubicInterpolate1D`).

**Inputs → outputs:** float32 p1..p4, t∈[0,1] → value on segment p2→p3.

## Completeness review

- Complete: yes for 1-D segment evaluation. No multi-point spline object, no derivatives, no non-uniform/centripetal Catmull-Rom.
- Tested: TestCubicSpline1DProperties (endpoints = p2/p3, constant input stays constant, ascending/descending), TestCubicCatmulRomSpline1D (t=0 → p2, t=1 → p3, t=0.5 → 0.5 on linear data). Run: go test -count=1 -json ./x/math/{colorscience,learn,grid,interpolation}/... (2026-11-09, slice lead; raw: tmp/easyrobot/filters/test-2026-11-09.json) → pass.
- Correct: yes for tested values; coefficients match Bourke's formulas.

## TODOs

- [ ] Deduplicate the three Catmull-Rom copies — x/math/interpolation/upsample.go:152, x/math/interpolation/matrix_upsample.go:171 #p3
- [ ] Multi-knot spline (natural/clamped, non-uniform time) for gait/joint trajectories — x/math/interpolation/spline.go #p2

## Links

- Module: [[interpolation]]
