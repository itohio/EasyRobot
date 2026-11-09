---
type: module
group: "Math and maps"
status: partial
tags: [easyrobot, interpolation, trajectory]
workstream: [W3, W8]
packages: [x/math/interpolation]
test_result: pass
---
# Interpolation (x/math/interpolation)

**Purpose:** Scalar curve interpolation (lerp, cosine, Bézier, 4-point cubic/Catmull-Rom/Hermite), 1-D/2-D resampling, and 2-D scattered-data interpolation (ordinary kriging, RBF). Relevant to gait/joint trajectories (W3) and to sparse-map/field reconstruction (W8).

**Packages:** `x/math/interpolation` — 8 source files, 950 LOC (bezier.go 63, cosine.go 8, lerp.go 11, spline.go 53, kriging.go 247, rbf.go 179, upsample.go 180, matrix_upsample.go 209); tests 871 LOC (interpolation_helpers_test.go, kriging_test.go, rbf_test.go, upsample_test.go, matrix_upsample_test.go). No build tags. Deps: `x/math/mat`, `x/math/vec`, `math32`; `cosine.go` uses stdlib `math` (float64). Only in-repo consumer: `x/math/colorscience/spd.go` ([[colorscience]]).

## Public API

- Scalar: `Lerp`, `LerpD`, `Cosine1D`, `Bezier1d/2d/3d`, `Bezier([]float32)`, `CubicSpline1D`, `CubicCatmulRomSpline1D`, `CubicHermiteSpline1D(…, tension, bias)` — all `(points…, t)` free functions on float32.
- Resampling: `LinearUpsample(src, dst vec.Vector)`, `CubicUpsample`, `LinearMatrixUpsample(src, dst mat.Matrix)`, `BicubicMatrixUpsample` — dst-sized output.
- Scattered 2-D: `NewKriging(VariogramModel)` + `AddSample/Fit/Interpolate`; `NewRBF(RBFKernel, eps)` + same methods; `Sample{X,Y,V}`; variogram constructors `Exponential/Gaussian/SphericalVariogram(range)`; kernels `Gaussian/Multiquadric/InverseMultiquadric/ThinPlateSpline/Biharmonic/TriharmonicKernel()`. Kriging and RBF share `Sample` and `solveLinear` (`kriging.go:185`).

## Algorithms

<!-- vault:algorithms h=868c9bef8692 -->

Counts: 5 tested · 0 implemented · 3 partial · 0 stub · 0 unverified (8 algorithms)

| Algorithm | Status | Tests | Summary |
|---|---|---|---|
| [[bezier-curve]] | tested | pass | Evaluates a cubic Bézier (Bernstein form) from 4 control points at parameter t… |
| [[cosine-interpolation]] | tested | pass | Ease-in/ease-out blend: μ₂=(1−cos πt)/2, returns a(1−μ₂)+bμ₂ (Paul Bourke). |
| [[cubic-spline-interpolation]] | tested | pass | Local 4-point cubic segment interpolation between p2 and p3 (p1, p4 neighbours), ported from Paul… |
| [[linear-interpolation]] | tested | pass | `Lerp(a,b,t) = a+(b-a)t`; `LerpD(a,d,t) = a+d·t` (precomputed delta). |
| [[ordinary-kriging]] | tested | pass | Ordinary kriging in semivariogram form: solves [Γ 1; 1ᵀ 0][w; μ] = [γ(x); 1] per query, value =… |
| [[cubic-hermite-spline]] | partial | pass | partial (wrong basis; test codifies bug) |
| [[rbf-interpolation]] | partial | pass | partial (no polynomial term; TPS wrong) |
| [[vector-matrix-upsampling]] | partial | pass | partial (NaN for target size 1) |

<!-- /vault:algorithms -->

## Build targets and dependencies

- Pure Go, no CGO, no reflection, no goroutines. float32 except `Cosine1D` (float64 `math.Cos`, `cosine.go:6`).
- Scalar curve functions: zero allocation — embedded friendly. `Bezier([]float32)` allocates result. `CubicUpsample` allocates coefficient slice per call; `LinearMatrixUpsample` allocates intermediate matrix + temp vectors per call.
- Kriging: O(n³) solve + 3 allocations **per query**; RBF: O(n³) once in Fit, O(n) per query. Both use float32 Gaussian elimination with an absolute 1e-6 pivot threshold → conditioning failures for smooth kernels (probe: Gaussian ε=0.05 singular). Not suitable on MCU for n ≳ 50 [INFERENCE].

## Tests and evidence

- go test -count=1 -json ./x/math/{colorscience,learn,grid,interpolation}/... (2026-11-09, slice lead; raw: tmp/easyrobot/filters/test-2026-11-09.json) → `x/math/interpolation` 27 pass, 0 fail.
- Probe probe tmp/easyrobot/filters/gridinterp/main.go, GOTOOLCHAIN=local GOPROXY=off GOFLAGS=-mod=mod go run . (2026-11-09): Hermite (0,1,2,10,t=0) → 3 (want 1), flat 5s → 10; Linear/CubicUpsample 3→1 → NaN; TPS on linear field → −1.0 (want 1); Gaussian RBF ε=0.05 → singular error; kriging 3-point → 0.651 (no reference).
- grep TODO/FIXME/panic("not implemented") in non-test files: none.

## API notes (ease of use, composability)

- Curves are 4-point segment primitives only; there is no spline/trajectory object over N knots with time parameterisation, derivatives (velocity/acceleration) or clamped end tangents — the things a gait/joint planner needs. Callers must manage knot windows and boundary duplication themselves.
- Argument order is `(points…, t)` everywhere but Hermite inserts `t` before tension/bias — easy to misuse; names `CubicCatmulRom` (typo) and `Bezier1d` vs `Cosine1D` casing inconsistent.
- No vector variants except Bézier; `vec.Vector` interop requires per-component loops. Catmull-Rom is implemented three times (spline.go, upsample.go, matrix_upsample.go).
- Kriging/RBF share the same `AddSample/Fit/Interpolate` shape — easy to swap — but there's no common interface type, no batch/grid evaluation (`InterpolateGrid(dst mat.Matrix, …)`) for building maps, and `Samples()` returns the internal slice (mutations bypass cache invalidation).
- Resampling functions are dst-sized and composable with `mat`/`vec`; docs mislabel the kernels (natural spline / Mitchell-Netravali vs actual Catmull-Rom).

## TODOs

Algorithm defects are tracked in the algorithm notes (table above) and rolled up in [[todo]]; this list holds module-level items only.

- [ ] Kriging: factorise once in Fit — x/math/interpolation/kriging.go:139 #p2
- [ ] N-knot trajectory spline with derivatives for W3 gaits — x/math/interpolation/spline.go #p2

## Inferred objectives (proposals)

- **proposal:** W3: a `Trajectory` type (piecewise cubic Hermite / quintic with position+velocity constraints, time-indexed, zero-alloc `Eval(t, dst)`), reusing Bézier for swing-foot arcs and Hermite (fixed) for stance blending.
- **proposal:** W8: `InterpolateGrid` for Kriging/RBF to fill sparse lidar/sensor fields into a `mat.Matrix` usable by [[grid-maps]].
- **proposal:** shared float64 (or Cholesky) solver from [[mat-vec]] instead of local `solveLinear`.
