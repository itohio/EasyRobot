---
type: algorithm
status: partial
tags: [easyrobot, dsp, fft]
workstream: W7
module: "[[dsp-core]]"
summary: "`FFT2DC.Backward` no zero-padding; FP32 rank-2 façade panics"
source: [x/math/dsp/fft2d.go, x/math/dsp/fft2dc.go, x/math/dsp/fft.go]
tests: [x/math/dsp/fft2d_test.go, x/math/dsp/fft2dc_test.go, x/math/dsp/fft2d_debug_test.go]
test_result: pass
complete: no
correct: unknown
evidence: "go test -count=1 -v ./x/math/dsp/ → ok, 71 PASS (2026-11-09, tmp/easyrobot/audio/dsp_v.txt); probe tmp/easyrobot/audio/dspcore/main.go (2026-11-09)"
---
# 2D FFT (row–column, real packed and complex64)

**What it does:** separable 2D DFT — 1D radix-2 FFT over every row, then every column (inverse: columns then rows), each dimension zero-padded to its next power of two. `FFT2D` (real float32 `mat.Matrix`, packed per row `[re×cols | im×cols]`) and `FFT2DC` (complex64 rank-2 tensor, row-major) share the algorithm and the 1D kernels of [[fft-radix2-real]] / [[fft-complex-1d]]; merged here.

**Where:**
- `x/math/dsp/fft2d.go:23-56` `FFT2D`/`NewFFT2D(rows, cols, lut...)`; `:91-166` forward; `:170-241` backward; `:255-259` `ifftRow` (unused, dead).
- `x/math/dsp/fft2dc.go:23-53` `FFT2DC`/`NewFFT2DC`; `:87-172` forward; `:176-250` backward.
- `x/math/dsp/fft.go:375-391,442-458` — `FFT` tensor façade, rank-2 **C64 only**; rank-2 FP32 panics (`fft.go:271,324`), rank > 2 panics (`fft.go:273,326,393,460`).

**Inputs → outputs:**
- `FFT2D.Forward(dst, src matTypes.Matrix)`: src `[r,c]` real; dst must have ≥ 2·C columns (C = padded cols) and R rows. `Backward` writes the real part, scaled 1/(R·C).
- `FFT2DC.Forward(dst, src types.Tensor)`: C64 rank 2; dst assumed exactly `[R, C]` (row stride `f.cols`, no shape check). Inverse scaled 1/(R·C).
- LUT: one LUT sized for `max(R, C)` serves both dimensions (stage-ordered layout).

## Completeness review

- Complete: no.
  - `FFT2DC.backward` copies src into dst but **does not zero-pad** when src is smaller than `[R,C]` (fft2dc.go:198-210) → stale dst content enters the transform. Probe: 1×1 src `[4]` into a dirty 2×2 dst → `[75.2 −23.8 −23.8 −23.8]` instead of all 1.
  - `FFT2D.forward` passes the whole dst row to `fp32.FFT` (fft2d.go:140-143); if dst has more than 2·C columns the kernel uses `len/2` as size → wrong transform. Hard `dst.(mat.Matrix)` assertions (fft2d.go:106-107,150-163) panic for other `matTypes.Matrix` implementations.
  - `FFT2D` inverse ignores the LUT (fft2d.go:214,237).
  - `FFT2DC.forward/backward` allocate `tempRow` per call (fft2dc.go:137,215); the façade builds a new `FFT2DC` (rows·cols scratch) on every call (fft.go:383-389, 450-456).
  - Mixed value/pointer receivers on `FFT2DC` (`Backward`, `Convolve` take value receivers, fft2dc.go:81,254) → copies the struct, harmless but inconsistent.
- Tested (PASS 2026-11-09):
  - `TestFFT2D_ForwardMatchesNumpy` — 2×3 (padded 2×4) forward vs `np.fft.fft2`, tol 1e-4.
  - `TestFFT2D_BackwardMatchesNumpy` — inverse of the same spectrum.
  - `TestFFT2D_BackwardDump` — logs only, asserts nothing (debug leftover).
  - `TestFFT2DC_ForwardBackwardMatchesNumpy` — 2×2 forward + inverse vs numpy.
- Correct: unknown overall — matching-shape transforms are right (probe: façade C64 4×8 non-square vs naive 2D DFT relErr 9.0e-8 no LUT, 8.6e-8 LUT); padding path in `FFT2DC.Backward` is wrong (above).

## TODOs

- [ ] Zero-pad dst in `FFT2DC.backward` when src is smaller than [R,C] — x/math/dsp/fft2dc.go:198-210 #p1 #ER-022
- [ ] Slice dst rows to exactly 2·C before `fftRow` — x/math/dsp/fft2d.go:140-143 #p2
- [ ] Validate dst shape in `FFT2DC.forward` (stride assumes [R,C]) — x/math/dsp/fft2dc.go:102-133 #p2
- [ ] Rank-2 FP32 path in `FFT` façade (panics) — x/math/dsp/fft.go:271,324 #p2
- [ ] Hoist `tempRow` into the struct; cache FFT2DC in the façade — x/math/dsp/fft2dc.go:137,215; x/math/dsp/fft.go:383-389 #p3
- [ ] Delete `ifftRow` and `fft2d_debug_test.go` (no assertion) — x/math/dsp/fft2d.go:255-259; x/math/dsp/fft2d_debug_test.go #p3
- [ ] Tests for non-square, padded and LUT paths — x/math/dsp/fft2dc_test.go #p2

## Links

- Module: [[dsp-core]]
- 1D kernels: [[fft-radix2-real]], [[fft-complex-1d]]; 2D convolution: [[fft-convolution-correlation]]
- Containers: [[tensor]]
