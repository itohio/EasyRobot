---
type: algorithm
status: partial
tags: [easyrobot, interpolation, resampling, grid]
workstream: W3
module: "[[interpolation]]"
summary: "partial (NaN for target size 1)"
source: [x/math/interpolation/upsample.go, x/math/interpolation/matrix_upsample.go]
tests: [x/math/interpolation/upsample_test.go, x/math/interpolation/matrix_upsample_test.go]
test_result: pass
complete: yes
correct: no
evidence: "tests pass; probe: LinearUpsample/CubicUpsample 3→1 return NaN"
---
# Vector and matrix resampling (linear / Catmull-Rom / bicubic)

**What it does:** Resamples a vector N→M or matrix R×C→R'×C' with endpoint-aligned sampling (pos = i·(N−1)/(M−1)). Works for down-sampling too despite the name.

**Where:**
- `x/math/interpolation/upsample.go:9` `LinearUpsample(src,dst)`; `:51` `CubicUpsample` — doc says "natural cubic spline" (`:78`, `:102`) but `computeCubicSpline :105-163` is **Catmull-Rom with duplicated end points** (allocates 3(n−1) coeffs per call `:122`).
- `x/math/interpolation/matrix_upsample.go:11` `LinearMatrixUpsample` (separable: rows then columns via `LinearUpsample`; allocates intermediate matrix + 4 temp vectors `:44-61`); `:84` `BicubicMatrixUpsample` — doc says "Mitchell-Netravali" (`:83`) but kernel is Catmull-Rom (`:171-181`), clamped 4×4 neighbourhood (`:150-151`).

**Inputs → outputs:** caller-allocated `dst` (size defines target); returns dst or nil for empty input.

## Completeness review

- Complete: yes for stated scope (no area-averaging/anti-aliased downsampling).
- Tested: TestLinearUpsample (exact expected arrays + endpoints), TestLinearUpsample_Invalid, TestCubicUpsample (endpoints, constant input, monotone data endpoints), TestLinearMatrixUpsample (corners, constant, centre near max), TestBicubicMatrixUpsample (corners, monotone diagonal), TestMatrixUpsample_Invalid. Run: go test -count=1 -json ./x/math/{colorscience,learn,grid,interpolation}/... (2026-11-09, slice lead; raw: tmp/easyrobot/filters/test-2026-11-09.json) → pass.
- Correct: **no** for target size 1: scale=(N−1)/0 → +Inf, pos=0·Inf=NaN (`upsample.go:32,85`; `matrix_upsample.go:116-117`). Probe (probe tmp/easyrobot/filters/gridinterp/main.go, GOTOOLCHAIN=local GOPROXY=off GOFLAGS=-mod=mod go run . (2026-11-09)): `LinearUpsample([1,2,3], len 1)` → `[NaN]`, `CubicUpsample` → `[NaN]`. Catmull-Rom/bicubic overshoot possible (not monotone-preserving) [INFERENCE from kernel]. Doc mismatches above.

## TODOs

- [ ] Handle len(dst)==1 (and rows/cols==1) — x/math/interpolation/upsample.go:32,85; x/math/interpolation/matrix_upsample.go:116 #p1 #backlog
- [ ] Fix docs: Catmull-Rom, not natural spline / Mitchell-Netravali — x/math/interpolation/upsample.go:78,102; x/math/interpolation/matrix_upsample.go:83 #p3
- [ ] Non-allocating variants (scratch buffers) for embedded/K230 feature maps — x/math/interpolation/upsample.go:122; x/math/interpolation/matrix_upsample.go:44-61 #p3

## Links

- Module: [[interpolation]]
