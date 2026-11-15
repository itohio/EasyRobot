---
type: algorithm
status: tested
tags: [easyrobot, interpolation, lerp]
workstream: W3
module: "[[interpolation]]"
summary: "`Lerp(a,b,t) = a+(b-a)t`; `LerpD(a,d,t) = a+d·t` (precomputed delta)."
source: [x/math/interpolation/lerp.go]
tests: [x/math/interpolation/interpolation_helpers_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "TestLerpD pass"
---
# Linear interpolation (Lerp)

**What it does:** `Lerp(a,b,t) = a+(b-a)t`; `LerpD(a,d,t) = a+d·t` (precomputed delta).

**Where:** `x/math/interpolation/lerp.go:4` (`Lerp`), `:9` (`LerpD`). float32, scalar, no allocation.

**Inputs → outputs:** float32 endpoints, t∈[0,1] (not clamped; extrapolates outside).

## Completeness review

- Complete: yes (scalar only; vector lerp lives in `vec`/elsewhere).
- Tested: TestLerpD (`interpolation_helpers_test.go`) checks values; `Lerp` exercised through `LinearUpsample` tests (`upsample_test.go` TestLinearUpsample expected arrays). Run: go test -count=1 -json ./x/math/{colorscience,learn,grid,interpolation}/... (2026-11-09, slice lead; raw: ) → pass.
- Correct: yes (value-checking tests pass; trivially exact formula).

## TODOs

- [ ] Direct TestLerp value test — x/math/interpolation/lerp.go:4 #p3

## Links

- Module: [[interpolation]]
