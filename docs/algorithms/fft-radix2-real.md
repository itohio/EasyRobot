---
type: algorithm
status: tested
tags: [easyrobot, dsp, fft, audio]
workstream: W7
module: "[[dsp-core]]"
summary: "matches naive DFT 6e-8..1.4e-5 relErr; LUT 50× more accurate at 4096"
source: [x/math/dsp/fft1d.go, x/math/primitive/fp32/dsp_fft.go, x/math/primitive/fp32/dsp_fft_lut.go, x/math/dsp/fft.go]
tests: [x/math/dsp/fft1d_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "go test -count=1 -v ./x/math/dsp/ → ok, 71 PASS (2026-11-09, tmp/easyrobot/audio/dsp_v.txt); probe tmp/easyrobot/audio/dspcore/main.go vs float64 naive DFT (2026-11-09)"
---
# Real-input 1D FFT (radix-2, packed layout)

**What it does:** forward/inverse DFT of a real float32 vector, zero-padded to the next power of two. Despite the "real FFT" name it is a full complex radix-2 FFT run on `[re…, im…]` with the imaginary half zeroed; Hermitian symmetry is **not** exploited (2× work and memory vs a true real FFT).

**Where:**
- `x/math/dsp/fft1d.go:21-161` — `FFT1D`, `NewFFT1D(length, lut...)`, `Forward(dst, src)`, `Backward(dst, src)`.
- Kernel: `x/math/primitive/fp32/dsp_fft.go:31-113` `FFT` (iterative Cooley–Tukey DIT, bit-reversal, twiddles by complex recurrence), `:120-154` `IFFT` (conjugate → FFT → conjugate, scale 1/N), `:167-334` `ComputeFFTLUT`/`FillFFTLUT`/`FFTWithLUT`/`IFFTWithLUT`; `dsp_fft_lut.go` `NewFFTLUT`.
- Tensor façade: `x/math/dsp/fft.go:218-327` (`FFT.Forward/Backward` on rank-1 FP32 tensors, lazy LUT via `withLUT`).

**Inputs → outputs:**
- `Forward(dst, src vec.Vector)`: src real, length ≤ N (excess silently truncated, shorter zero-padded); dst ≥ 2N, `dst[0:N]`=Re, `dst[N:2N]`=Im. Panics if `len(dst) < 2N` (fft1d.go:87).
- `Backward(dst, src)`: src packed 2N; dst gets the real part only (imaginary part discarded without check), scaled 1/N (convention: forward unscaled, inverse 1/N, numpy-compatible).
- N = `NextPowerOfTwo(length)`; non-power-of-2 lengths give the DFT of the **zero-padded** signal, not an N-point DFT.
- LUT layout `[cos×M, sin×M]`, M = N−1; a LUT built for a larger N also works (stages are stored in ascending order).

## Completeness review

- Complete: yes for 1D power-of-2 / zero-padded scope. `Backward` ignores a supplied LUT (fft1d.go:153 always `fp32.IFFT`) — performance only. No real-FFT (N/2 complex) optimisation; no float64 variant; no mixed-radix/Bluestein.
- Tested (fft1d_test.go, all PASS 2026-11-09):
  - `TestFFT1D_ForwardImpulse` — impulse → DC=1, Im[0]=0 (values).
  - `TestFFT1D_ForwardBackward` — round trip N=4 and N=6 (padded 8), tol 1e-4.
  - `TestFFT1D_EdgeCases` — N=1 identity; N=3 "has non-zero values" (weak).
  - Convolve/Correlate tests → [[fft-convolution-correlation]].
- Correct: yes (probe, float64 naive DFT reference, uniform random input):

| N | relErr no LUT | relErr LUT | round-trip max abs (no LUT / LUT) |
|---|---|---|---|
| 8 | 7.7e-8 | 6.1e-8 | 6e-8 / 6e-8 |
| 64 | 2.8e-7 | 1.3e-7 | 3.6e-7 / 2.4e-7 |
| 1024 | 2.2e-6 | 2.2e-7 | 6.7e-6 / 3.9e-6 |
| 4096 | 1.4e-5 | 2.7e-7 | 4.7e-5 / 3.3e-5 |
| 65536 | — | — | 6.1e-4 / 4.1e-4 |

  IFFT of all-ones spectrum (N=16) → 1.0 at n=0 (1/N scaling confirmed). N=6 → matches 8-point DFT of zero-padded input (6.5e-8).
  Without LUT the per-stage twiddle recurrence (dsp_fft.go:106-109) accumulates float32 error ∝ N: ~50× worse than LUT at 4096. Use a LUT for N ≥ 1024.

## TODOs

- [ ] Use `IFFTWithLUT` when a LUT is set (Backward ignores it) — x/math/dsp/fft1d.go:153 #p3
- [ ] Twiddle recurrence error grows with N (relErr 1.4e-5 at 4096 vs 2.7e-7 with LUT); compute twiddles directly or default to LUT — x/math/primitive/fp32/dsp_fft.go:106-109 #p2
- [ ] True real-input FFT (N/2 complex + split) and a half-spectrum layout — x/math/dsp/fft1d.go:111-117 #p3
- [ ] `Backward` silently drops the imaginary part; `Forward` silently truncates src > N — document or return error — x/math/dsp/fft1d.go:93-96,155-160 #p3
- [ ] Test values against a reference DFT at N ≥ 64 (current tests use N ≤ 8) — x/math/dsp/fft1d_test.go #p2

## Links

- Module: [[dsp-core]]
- Complex counterpart: [[fft-complex-1d]]; 2D: [[fft-2d]]; fast conv/corr built on it: [[fft-convolution-correlation]]
- Consumers: none outside tests in-repo (grep `NewFFT1D` 2026-11-09); spectral checks in [[modulation]] tests use Goertzel instead ([[goertzel]]).
