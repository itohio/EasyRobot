---
type: algorithm
status: partial
tags: [easyrobot, interpolation, rbf]
workstream: W8
module: "[[interpolation]]"
summary: "partial (no polynomial term; TPS wrong)"
source: [x/math/interpolation/rbf.go]
tests: [x/math/interpolation/rbf_test.go, x/math/interpolation/interpolation_helpers_test.go]
test_result: pass
complete: no
correct: no
evidence: "tests pass; probe: TPS on linear field returns -1.0 (want 1); Gaussian eps=0.05 6 pts → singular error"
---
# Radial basis function interpolation

**What it does:** Global RBF interpolant s(x)=Σcᵢφ(ε‖x−xᵢ‖); solves A c = v with Aᵢⱼ=φ(ε‖xᵢ−xⱼ‖) by Gaussian elimination.

**Where:** `x/math/interpolation/rbf.go` — `RBF :15`, `NewRBF :24`, `Fit :45-87` (allocs n×n `:60`), `Interpolate :90`, kernels `GaussianKernel :122`, `MultiquadricKernel :129`, `InverseMultiquadricKernel :136`, `ThinPlateSplineKernel :143` (r² log r), `BiharmonicKernel :153` (r), `TriharmonicKernel :163` (r³); shares `solveLinear` with kriging (`kriging.go:185`).

**Inputs → outputs:** samples (x,y,v), kernel, ε → `Interpolate(x,y) (float32, error)`; coefficients cached until `AddSample`.

## Completeness review

- Complete: **no**. Conditionally positive-definite kernels (TPS, biharmonic r, triharmonic r³, multiquadric) need an appended low-degree polynomial term + side conditions; none is implemented, so the system can be singular/indefinite and linear fields are not reproduced. No smoothing (λI) option.
- Tested: TestRBF_SingleSample, TestRBF_ExactReproduction (reproduces sample values), TestRBF_Interpolation (approx mid value), TestRBFKernels (finite values), TestRBF_NoSamples, TestRBF_ImplicitFit. Run: go test -count=1 -json ./x/math/{colorscience,learn,grid,interpolation}/... (2026-11-09, slice lead; raw: tmp/easyrobot/filters/test-2026-11-09.json) → pass.
- Correct: **no** for polyharmonic kernels. Probe (probe tmp/easyrobot/filters/gridinterp/main.go, GOTOOLCHAIN=local GOPROXY=off GOFLAGS=-mod=mod go run . (2026-11-09)): TPS ε=1 on unit square with v=x+y → value at (0.5,0.5) = **−1.0** (want 1). Gaussian ε=0.05 with 6 collinear points spacing 1 → `singular or ill-conditioned matrix` (float32 + abs pivot 1e-6 at `kriging.go:211`; Gaussian A becomes near-rank-1 for small ε). `log` returns −1e10 for x≤0 (`:174-178`).

## TODOs

- [ ] Polynomial augmentation (degree 1) for TPS/polyharmonic/MQ kernels — x/math/interpolation/rbf.go:45-87 #p1 #backlog
- [ ] Conditioning: float64 solve or Cholesky/LU with relative pivot tolerance; optional regularisation λ — x/math/interpolation/kriging.go:185-211 #p2
- [ ] Test that linear fields are reproduced by TPS — x/math/interpolation/rbf_test.go #p2

## Links

- Module: [[interpolation]]
