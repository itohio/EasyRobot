---
type: algorithm
status: tested
tags: [easyrobot, dsp, fft, fp32, c64]
workstream: [W9]
module: "[[primitive-kernels]]"
summary: "radix-2, power-of-2; non-LUT twiddle recurrence drifts (1.6e-2 vs 2e-4 at n = 16384)"
source: [x/math/primitive/fp32/dsp_fft.go, x/math/primitive/fp32/dsp_fft_lut.go, x/math/primitive/c64/dsp_fft.go, x/math/primitive/c64/dsp_fft_lut.go]
tests: [x/math/primitive/fp32/dsp_fft_test.go, x/math/primitive/c64/dsp_fft_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "go test -count=1 ./x/... (2026-11-09) → fp32/c64 TestFFT_MatchesNumpyReference, *_Roundtrip, *WithLUT pass; probe (2026-11-09): vs float64 DFT n=16384 maxErr 0.016 (recurrence) / 1.96e-4 (LUT), max|X|=414"
---
# Primitive FFT (fp32 planar and complex64)

**What it does:** In-place iterative radix-2 decimation-in-time Cooley–Tukey FFT/IFFT, either with on-the-fly twiddles (recurrence) or with a precomputed twiddle LUT.

**Where:**
- fp32, planar layout `data[0:N]` = real, `data[N:2N]` = imag: `x/math/primitive/fp32/dsp_fft.go`:
  - `NextPowerOfTwo` 13-24, `FFT` 31-113, `IFFT` 120-154
  - `ComputeFFTLUT` 167-181, `FillFFTLUT` 187-215, `FFTWithLUT` 224-291, `IFFTWithLUT` 300-334
  - `NewFFTLUT` in `dsp_fft_lut.go`
- complex64 `x/math/primitive/c64/dsp_fft.go`: `FFT` 29-92, `IFFT` 97-132, LUT variants 138-287.
- Higher-level wrappers in `x/math/dsp` are covered by [[fft-complex-1d]] and [[fft-radix2-real]].

**Inputs → outputs:**
- Sizes must be powers of 2; anything else panics, as does an odd fp32 length.
- The forward transform is unscaled with sign e^{−2πi kn/N}. IFFT is `conj(FFT(conj(x)))/N`.
- LUT layout:
  - fp32: `[cos…(N−1), sin…(N−1)]`; pass `totalFactors=N−1`.
  - c64: `N−1` complex64 values.

## Completeness review

- Complete: yes, for power-of-2 complex FFT.
  - No real-input FFT in the primitives (no rfft half-spectrum packing), no radix-4/split-radix, no Bluestein for other sizes.
  - `FFTLUT` struct (dsp_fft.go:157-161) is declared but never used.
  - fp32 `NextPowerOfTwo` lacks `n >> 32`, so it is wrong for n > 2³²; c64 `nextPowerOfTwo` is unexported.
- Tested, for both packages:
  - Value checks: `TestFFT_Impulse`, `_Constant`, `_SineWave`, `_MatchesNumpyReference`, `_LargeSize`, `_IFFT_Identity`.
  - `TestIFFT_Roundtrip`, `TestFFTWithLUT[_Roundtrip]`, `TestIFFTWithLUT[_Roundtrip]`, `TestComputeFFTLUT`, `TestFillFFTLUT`, `TestNewFFTLUT`, `*_ErrorConditions`.
  - All pass (2026-11-09).
- Correct: yes. Probe vs a float64 direct DFT on Gaussian noise:

  | n | recurrence (fp32 and c64) | LUT | max\|X\| |
  |---|---|---|---|
  | 1024 | 2.7e-4 | 2.1e-5 | 89.5 |
  | 16384 | 1.6e-2 | 1.96e-4 | 414 |

  The non-LUT twiddle recurrence `w *= wlen` (fp32 dsp_fft.go:105-108, c64 85-88) accumulates float32 rounding linearly in stage length, about 80× worse than the LUT at n=16384.

## Embedded suitability

No allocation in FFT/IFFT. `NewFFTLUT` allocates 2(N−1) floats once. float32 `math32` only, no float64. Bit reversal is an O(N log N) loop, not a table. Fine for MCU audio sizes (256–2048). Prefer the LUT variants for accuracy and speed.

## TODOs

- [ ] Compute twiddles directly per stage (cos/sin of j·angle) or use the LUT in the non-LUT FFT to remove recurrence drift — x/math/primitive/fp32/dsp_fft.go:105-108; c64/dsp_fft.go:85-88; probe err 0.016 vs 1.96e-4 #p2
- [ ] Remove the unused FFTLUT struct; add `n>>32` to NextPowerOfTwo — x/math/primitive/fp32/dsp_fft.go:13-24,157-161 #p3
- [ ] Add a real-input FFT (rfft) primitive for audio/vibration on MCU — fp32/dsp_fft.go #p2

## Links

- Module: [[primitive-kernels]]
- Related: [[dsp-core]], [[fft-complex-1d]], [[fft-radix2-real]], [[fft-convolution-correlation]], [[primitive-convolution-correlation]]
