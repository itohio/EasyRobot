---
type: algorithm
status: partial
tags: [easyrobot, dsp, convolution, correlation, fp32, c64]
workstream: [W9]
module: "[[primitive-kernels]]"
summary: "correlation only non-negative lags; c64 correlation not conjugated…"
source: [x/math/primitive/fp32/dsp_convolution.go, x/math/primitive/fp32/dsp_correlation.go, x/math/primitive/c64/dsp_convolution.go, x/math/primitive/c64/dsp_correlation.go]
tests: [x/math/primitive/fp32/dsp_convolution_test.go, x/math/primitive/fp32/dsp_correlation_test.go, x/math/primitive/c64/dsp_convolution_test.go, x/math/primitive/c64/dsp_correlation_test.go]
test_result: pass
complete: no
correct: no
evidence: "go test -count=1 ./x/... (2026-11-09) → fp32/c64 TestConvolve*/TestCrossCorrelate* pass; probe (2026-11-09): xcorr has only non-negative lags; c64 xcorr unconjugated (i⋆i = -1); Convolve2D truncates columns"
---
# Primitive 1D/2D convolution and cross-correlation (fp32, complex64)

**What it does:** Direct time-domain DSP convolution `y[n]=Σ x[k]h[n−k]` and correlation `R(τ)=Σ x[n]y[n+τ]`, in 1D and 2D, with plain and accumulate (`*Add`) variants. This is not the NN conv; see [[conv2d-kernel]].

**Where:**
- `x/math/primitive/fp32/dsp_convolution.go`: `Convolve1D` 7-49, `Convolve1DAdd` 55-94, `Convolve2D` 102-171, `Convolve2DAdd` 179-246.
- `x/math/primitive/fp32/dsp_correlation.go`: `CrossCorrelate1D` 6-42, `CrossCorrelate2D` 50-112, `CrossCorrelate1DAdd` 118-150, `CrossCorrelate2DAdd` 158-216.
- `x/math/primitive/c64/` has identical complex64 copies at the same line ranges.

**Inputs → outputs:**
- 1D: `dst` with length up to `n1+n2−1`; output is truncated to `len(dst)`.
- 2D: row-major flattened signals with row counts; `cols = len/rows`. Convolve2D's `dst` is `dstRows × max(cols1,cols2)`; correlation's `dst` is `dstRows × (cols1+cols2−1)`. Too small a dst panics.

## Completeness review

- Complete: no.
  - **Correlation computes only lags τ ≥ 0.** Indices `i ≥ n2` are written as 0, so the trailing `n1−1` outputs of a "full"-length dst are always zero (dsp_correlation.go:27-40). The tests encode those zeros (`{5, 8, 11, 4, 0}`, dsp_correlation_test.go:60).
  - `Convolve2D` output width is `max(cols1,cols2)` rather than `cols1+cols2−1` (dsp_convolution.go:113-116), inconsistent with 1D "full" convolution and with 2D correlation.
  - No FFT-based fast path; that lives in `x/math/dsp`, see [[fft-convolution-correlation]].
  - No "same"/"valid" modes.
- Tested: `TestConvolve1D[Add|VsAdd|_Empty|_SmallDst]`, `TestConvolve2D[Add|_Panic]`, `TestCrossCorrelate1D[Add|_Empty|_SmallDst]`, `TestCrossCorrelate2D[Add|_Panic]` check values for fp32 and c64. All pass (2026-11-09).
- Correct: **no** against standard definitions (yes against the in-file formula).
  - Probe `CrossCorrelate1D([1,2,3],[0,1,0])` gave [2 1 0 0 0]; the negative-lag value 3 is never produced.
  - **c64 correlation does not conjugate**: `Σ x[n]·y[n+τ]` instead of `Σ conj(x[n])·y[n+τ]` (c64/dsp_correlation.go:36, 104). Probe: correlating `i` with `i` at lag 0 gave −1; the standard correlation gives 1. This breaks matched filtering and phase estimation on complex baseband.
  - Probe `Convolve2D` of 2×2 ones with 2×2 ones into 3 rows gave [1 2 2 4 1 2 0 0 0]. The full result is [1 2 1 2 4 2 1 2 1]: the last column is lost, and the "row 2" values land in the wrong cells because dst width is 2.

## Embedded suitability

No allocation. float32/complex64 only. Bounds-check-eliminated loops with split index ranges. O(n1·n2) per call, fine for short FIR kernels on MCU. No generics; fp32 and c64 are copy-pasted twins.

## TODOs

- [ ] Compute negative lags (full length n1+n2−1, τ from −(n1−1)) or document the one-sided definition and shrink the output length — x/math/primitive/fp32/dsp_correlation.go:6-42; c64 same #p2
- [ ] Conjugate the first operand in complex correlation — x/math/primitive/c64/dsp_correlation.go:36,104,144,208 #p1 #ER-020
- [ ] Make Convolve2D full-size (cols1+cols2−1) or add an explicit mode parameter — x/math/primitive/fp32/dsp_convolution.go:113-116; c64 same #p2
- [ ] Generate fp32/c64 twins from one generic source to avoid drift — dsp_convolution.go, dsp_correlation.go #p3

## Links

- Module: [[primitive-kernels]]
- Related: [[dsp-core]], [[primitive-fft]], [[fft-convolution-correlation]]
