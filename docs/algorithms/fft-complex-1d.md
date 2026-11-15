---
type: algorithm
status: tested
tags: [easyrobot, dsp, fft, audio]
workstream: W7
module: "[[dsp-core]]"
summary: "matches naive DFT; same LUT advice"
source: [x/math/dsp/fft1dc.go, x/math/primitive/c64/dsp_fft.go, x/math/primitive/c64/dsp_fft_lut.go, x/math/dsp/fft.go]
tests: [x/math/dsp/fft1dc_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "go test -count=1 -v ./x/math/dsp/ → ok, 71 PASS (2026-11-09); probe vs float64 naive DFT (2026-11-09)"
---
# Complex 1D FFT (radix-2, complex64)

**What it does:** forward/inverse DFT of a complex64 rank-1 tensor, zero-padded to the next power of two. Same iterative radix-2 Cooley–Tukey DIT as [[fft-radix2-real]] but on `[]complex64`.

**Where:**
- `x/math/dsp/fft1dc.go:23-181` — `FFT1DC`, `NewFFT1DC(length, lut...)`, `Forward(dst, src types.Tensor)`, `Backward(dst, src)`.
- Kernel: `x/math/primitive/c64/dsp_fft.go:29` `FFT`, `:97` `IFFT` (conjugate trick, 1/N), `:138-285` LUT variants (`ComputeFFTLUT` = N−1 complex factors, `FFTWithLUT`, `IFFTWithLUT`).
- Tensor façade: `x/math/dsp/fft.go:330-372,397-439` (`FFT` rank-1 C64 path).

**Inputs → outputs:** src/dst must be C64, rank 1 (panics otherwise, fft1dc.go:87-98). Src longer than N truncated, shorter zero-padded; dst receives `min(N, len(dst))` bins (a short dst silently gets a truncated spectrum). Forward unscaled, inverse scaled 1/N (numpy convention).

## Completeness review

- Complete: yes for 1D power-of-2 scope. Unlike `FFT1D`, `Backward` does use the LUT (fft1dc.go:168-170). No float64/complex128 path; no non-power-of-2 algorithm.
- Tested: `TestFFT1DC_ForwardBackward` — 4-point forward vs numpy values and inverse round trip, tol 1e-4 (PASS 2026-11-09). Convolve/Correlate tests → [[fft-convolution-correlation]].
- Correct: yes. Probe vs float64 naive DFT, random complex input: relErr N=8 1.3e-7 / 7.1e-8 (no LUT / LUT); N=256 3.4e-7 / 1.8e-7; N=4096 1.4e-5 / 2.7e-7. Round trip max abs at 4096: 6.9e-5 no LUT, 5.2e-7 LUT. Same twiddle-recurrence drift as the fp32 kernel → prefer LUT for N ≥ 1024.

## TODOs

- [ ] Twiddle recurrence error grows with N (relErr 1.4e-5 at 4096 without LUT) — x/math/primitive/c64/dsp_fft.go:58-90 #p2
- [ ] Short dst silently truncates spectrum; panic or document like FFT1D does — x/math/dsp/fft1dc.go:125-130 #p3
- [ ] Reference-DFT test at N ≥ 64 — x/math/dsp/fft1dc_test.go #p2

## Links

- Module: [[dsp-core]]
- Siblings: [[fft-radix2-real]], [[fft-2d]], [[fft-convolution-correlation]]
- Data container: [[tensor]]
