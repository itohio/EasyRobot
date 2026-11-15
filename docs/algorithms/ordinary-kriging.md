---
type: algorithm
status: tested
tags: [easyrobot, interpolation, kriging, geostatistics]
workstream: W8
module: "[[interpolation]]"
summary: "Ordinary kriging in semivariogram form: solves [Γ 1; 1ᵀ 0][w; μ] = [γ(x); 1] per query, value =…"
source: [x/math/interpolation/kriging.go]
tests: [x/math/interpolation/kriging_test.go, x/math/interpolation/interpolation_helpers_test.go]
test_result: pass
complete: yes
correct: unknown
evidence: "TestKriging_* + TestVariogramModels pass; probe 3-point spherical → 0.651, no error"
---
# Ordinary kriging (2-D scattered data)

**What it does:** Ordinary kriging in semivariogram form: solves [Γ 1; 1ᵀ 0][w; μ] = [γ(x); 1] per query, value = Σwᵢvᵢ (weights sum to 1). Fixed variogram models with nugget 0, sill 1.

**Where:** `x/math/interpolation/kriging.go` — `Sample :12`, `VariogramModel :17`, `ExponentialVariogram :21` (1−e^(−3d/a)), `GaussianVariogram :31` (1−e^(−3d²/a²)), `SphericalVariogram :41`, `Kriging :54`, `NewKriging :62`, `AddSample :70`, `Fit :83` (builds (n+1)² matrix only), `Interpolate :111`, `solveLinear :185` (Gaussian elimination, partial pivoting, abs pivot threshold 1e-6 `:211`).

**Inputs → outputs:** samples (x,y,v float32), variogram range → `Interpolate(x,y) (float32, error)`.

## Completeness review

- Complete: yes for basic OK. Missing: nugget/sill parameters, variogram fitting from data, kriging variance output, anisotropy.
- Performance: `Fit` doesn't factorise; **every `Interpolate` re-solves the full (n+1)×(n+1) system** (`:139`) — O(n³) per query plus allocations of rhs, weights and augmented matrix (`:123`, `:138`, `solveLinear :192-197`). Precomputing an LU (or Γ⁻¹) once in `Fit` makes queries O(n²)/O(n). Duplicate sample locations → singular → error. float32 elimination; abs pivot threshold not scale-aware.
- Tested: TestKriging_SingleSample (42), TestKriging_LinearData (approx expected value), TestKriging_ExactReproduction (returns sample values at sample points), TestKriging_NoSamples (error), TestVariogramModels (γ(0)=0, monotone), TestKrigingSamples (shared backing slice). Run: go test -count=1 -json ./x/math/{colorscience,learn,grid,interpolation}/... (2026-11-09, slice lead) → pass.
- Correct: unknown beyond exactness at samples (value assertions pass); probe 3-point spherical(10) query → 0.651, no error (2026-11-09) — not cross-checked against a reference.

## TODOs

- [ ] Factorise once in Fit; reuse for all queries; no per-query allocation — x/math/interpolation/kriging.go:83,139 #p2
- [ ] Nugget/sill/range parameters and kriging variance — x/math/interpolation/kriging.go:21-50 #p3
- [ ] Reference-value test (e.g. vs gstools/pykrige numbers) — x/math/interpolation/kriging_test.go #p2

## Links

- Module: [[interpolation]]
