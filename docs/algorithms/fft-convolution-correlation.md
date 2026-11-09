---
type: algorithm
status: partial
tags: [easyrobot, dsp, fft, convolution, correlation]
workstream: W7
module: "[[dsp-core]]"
summary: "2D real conv multiplies packed halves element-wise; 2D complex and mis-sized 1D complex wrong; lag…"
source: [x/math/dsp/fft1d.go, x/math/dsp/fft1dc.go, x/math/dsp/fft2d.go, x/math/dsp/fft2dc.go]
tests: [x/math/dsp/fft1d_test.go, x/math/dsp/fft1dc_test.go]
test_result: pass
complete: no
correct: no
evidence: "go test -count=1 -v ./x/math/dsp/ → ok, 71 PASS (2026-11-09); probe tmp/easyrobot/audio/dspcore/main.go shows wrong 2D/complex results (2026-11-09)"
---
# FFT fast convolution and cross-correlation (1D/2D, real/complex)

**What it does:** linear convolution `x*h` and cross-correlation via FFT → pointwise product (conjugated for correlation) → IFFT, zero-padded to `NextPow2(len(x)+len(h)−1)` per dimension.

**Where:** methods on each FFT type:

| Method | Location | Buffers |
|---|---|---|
| `FFT1D.Convolve(dst, signal, kernel)` | x/math/dsp/fft1d.go:165-227 | `fp32.Pool` |
| `FFT1D.Correlate(dst, s1, s2)` | x/math/dsp/fft1d.go:232-295 | `fp32.Pool` |
| `FFT1DC.Convolve` | x/math/dsp/fft1dc.go:185-252 | `tensor.New` ×2 + maybe new FFT1DC per call |
| `FFT1DC.Correlate` | x/math/dsp/fft1dc.go:257-311 | 3 new FFT1DC + 2 tensors per call; ignores receiver |
| `FFT2D.Convolve` | x/math/dsp/fft2d.go:263-323 | `fp32.Pool` |
| `FFT2D.Correlate` | x/math/dsp/fft2d.go:328-388 | `fp32.Pool` |
| `FFT2DC.Convolve` | x/math/dsp/fft2dc.go:254-326 | tensors + new FFT2DC |
| `FFT2DC.Correlate` | x/math/dsp/fft2dc.go:331-386 | tensors + 2 new FFT2DC |

**Inputs → outputs:** dst ≥ `len(x)+len(h)−1` (per dim). `FFT1D`/`FFT2D` require the receiver to be exactly `NextPow2(resultLen)` and panic otherwise; complex variants accept any receiver and build helpers. Docs claim "no allocations" for all eight — false for the complex variants.

## Completeness review

- Complete: no; several variants are wrong.
  - **FFT2D.Convolve / Correlate multiply the packed buffers element-wise** (`sigFlat[i] *= kernFlat[i]`, fft2d.go:308-310, 373-375): Re·Re and Im·Im separately, not a complex product; Correlate also skips the conjugate. Probe: 2×2 `[1 2;3 4]` ⊛ `[1 0;0 1]` → `[3 1.75 0 1.75; 2 2.5 0.5 0 …]`, expected `[1 2 0; 3 5 2; 0 3 4]`.
  - **FFT2DC.Convolve** forwards each input with an FFT2DC sized to that input (fft2dc.go:275-284) but writes into an `[f.rows,f.cols]` tensor with the helper's row stride → scrambled layout and circular (not linear) convolution. Probe: same 2×2 case → `[1.25 …]` constant. `FFT2DC.Correlate` same defect (fft2dc.go:346-355).
  - **FFT1DC.Convolve** reuses the receiver when `NextPow2(len(signal)) == f.length` even if `f.length < resultLen` (fft1dc.go:200-211) → circular aliasing, output truncated to `f.length`. Probe: `NewFFT1DC(4).Convolve([1 2 3 4],[1 1 1 1])` → `[10 10 10 10 0 0 0]`, expected `[1 3 6 10 9 7 4]`.
  - **Correlate lag ordering is inconsistent.** Output is the circular correlation in FFT order (lag 0 at index 0, negative lags at the tail), then everything past `resultLen` is zeroed (fft1d.go:288-294). When `fftSize > resultLen` negative lags are discarded; when `fftSize == resultLen` they appear wrapped at indices `len(s1)…`. Probe (len 5 × len 4, fft 8): `[3 13 −8 3 10 1 3 −3]` = lags 0..4 then −3..−1; full linear xcorr lags −3..4 is `[1 3 −3 3 13 −8 3 10]`. The real test `naiveCorrelate` only checks non-negative lags; the complex test pins the wrapped circular result — the two variants define different outputs.
  - 2D zero-fill of the dst tail uses flat indices `≥ resultRows*resultCols` (fft2d.go:315-322; fft2dc.go:318-325) — wrong when dst cols > resultCols.
- Tested (PASS 2026-11-09): `TestFFT1D_ConvolveMatchesNaive` (5⊛3 vs naive, tol 1e-3), `TestFFT1D_CorrelateMatchesNaive` (non-negative lags only), `TestFFT1D_Convolve_LengthMismatch`/`_Correlate_LengthMismatch` (panic expected), `TestFFT1DC_ConvolveMatchesNaive` (3⊛2, receiver sized to result), `TestFFT1DC_CorrelateMatchesNaive` (pins circular wrap). No 2D convolution/correlation tests.
- Correct: no — 1D real convolution right for the tested sizing; 2D real, 2D complex and mis-sized 1D complex are wrong (probe above).

## TODOs

- [ ] FFT2D Convolve/Correlate: complex multiply on packed rows (Re/Im halves), conjugate for correlation — x/math/dsp/fft2d.go:308-310,373-375 #p1 #ER-022
- [ ] FFT2DC Convolve/Correlate: transform both inputs at [f.rows,f.cols] = NextPow2(result) — x/math/dsp/fft2dc.go:275-284,346-355 #p1 #ER-022
- [ ] FFT1DC.Convolve: size FFT from resultLen, never reuse a smaller receiver — x/math/dsp/fft1dc.go:200-211 #p1 #ER-022
- [ ] Define one correlation output convention (full lags −(M−1)..N−1, lag-0 index documented) for all four Correlate methods — x/math/dsp/fft1d.go:288-294; x/math/dsp/fft1dc.go:303-310 #p1 #ER-022
- [ ] Complex variants: no per-call allocation (use receiver scratch / pool) or drop the "no allocations" doc claim — x/math/dsp/fft1dc.go:214-215,271-275 #p2
- [ ] 2D tail zeroing per row — x/math/dsp/fft2d.go:315-322; x/math/dsp/fft2dc.go:318-325 #p2
- [ ] Tests: 2D conv/corr vs naive; correlation negative lags; mis-sized receivers — x/math/dsp/fft2d_test.go #p2

## Links

- Module: [[dsp-core]]
- Transforms: [[fft-radix2-real]], [[fft-complex-1d]], [[fft-2d]]
- Direct (time-domain) correlation used by generators: [[chirp-generators]], [[sine-oscillator-generators]]
- Filtering alternatives: [[filters]]
