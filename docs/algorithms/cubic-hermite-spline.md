---
type: algorithm
status: partial
tags: [easyrobot, interpolation, spline]
workstream: W3
module: "[[interpolation]]"
summary: "partial (wrong basis; test codifies bug)"
source: [x/math/interpolation/spline.go]
tests: [x/math/interpolation/interpolation_helpers_test.go]
test_result: pass
complete: yes
correct: no
evidence: "test passes but asserts the wrong value (t=0 → 3, expected p2=1); probe flat 5,5,5,5 → 10 at t=0"
---
# Cubic Hermite spline (tension/bias, Kochanek–Bartels style)

**What it does:** Intended: Bourke's Hermite interpolation between p2 and p3 with tension and bias controlling tangents.

**Where:** `x/math/interpolation/spline.go:32-53` (`CubicHermiteSpline1D(p1,p2,p3,p4,t,tension,bias)`).

**Inputs → outputs:** float32 points, t∈[0,1], tension (comment: 0 high, 0.5 normal, 1 low), bias → value.

## Completeness review

- Complete: yes (signature/body present).
- Tested: TestCubicHermiteSpline1D — **endpoint subtest expects 3 at t=0 for p=(0,1,2,10)**, i.e. it codifies the bug (correct value is p2=1); tension/bias subtests only compare ordering. Run: go test -count=1 -json ./x/math/{colorscience,learn,grid,interpolation}/... (2026-11-09, slice lead; raw: tmp/easyrobot/filters/test-2026-11-09.json) → pass.
- Correct: **no**. `spline.go:50` `a3 := 1 - a1` should be `a3 := -2*mu3 + 3*mu2` (= 1 − a0; the inline comment even says so). Result: at t=0 returns p2+p3. Probe (probe tmp/easyrobot/filters/gridinterp/main.go, GOTOOLCHAIN=local GOPROXY=off GOFLAGS=-mod=mod go run . (2026-11-09)): (0,1,2,10,t=0) → 3 (want 1); flat (5,5,5,5,t=0) → 10 (want 5); flat t=0.5 → 6.875 (want 5). Tension parameterisation differs from Bourke: tangent factor `(1±bias)·tension` (`:39-40`) vs Bourke `(1±bias)(1−tension)/2`; consistent with the comment only.

## TODOs

- [ ] Fix basis: a3 = -2*mu3 + 3*mu2 — x/math/interpolation/spline.go:50 #p1 #backlog
- [ ] Fix test expecting 3 at t=0 (should be p2) and add flat-input test — x/math/interpolation/interpolation_helpers_test.go:224 #p1 #backlog
- [ ] Document/align tension parameterisation with Bourke/KB — x/math/interpolation/spline.go:28-40 #p3

## Links

- Module: [[interpolation]]
