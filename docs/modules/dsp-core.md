---
type: module
group: "Signal, codecs, audio"
status: partial
tags: [easyrobot, dsp, fft, audio, generators]
workstream: W7
packages: [x/math/dsp]
test_result: pass
evidence: "go test -count=1 -v ./x/math/dsp/ → ok, 71 PASS (2026-11-09); numeric probe vs float64 references (2026-11-09)"
---
# dsp-core — `x/math/dsp` (package root)

**Purpose:** float32 signal-processing primitives under the audio modem stack: FFTs (real packed, complex, 2D), FFT convolution/correlation, windows, sample generators (oscillators, pulse shapes, noise, chirps) and measurements (RMS stats, Goertzel). Subpackages `coding/` and `modulation/` are separate notes: [[line-coding-and-fec]], [[modulation]]. Consumers: [[modulation]], [[audio-modem-cli]]; data containers from [[tensor]] / `vec` / `mat`. Workstream: [[W7-filters-codecs-voice]].

**Packages:** `x/math/dsp` root — 4 751 LOC non-test, 2 253 LOC tests, `SPEC.md` 833 lines. No build tags, no CGO, no goroutines.

| File | LOC | Contents |
|---|---|---|
| dsp.go | 10 | `Sample` constraint (`types.Number \| types.Unsigned`) used by modulation/adaptive |
| fft.go | 461 | `FFT` tensor façade (FP32 1D, C64 1D/2D), lazy LUT, `Release` |
| fft1d.go | 295 | `FFT1D` real packed FFT + `Convolve`/`Correlate` |
| fft1dc.go | 311 | `FFT1DC` complex64 FFT + `Convolve`/`Correlate` |
| fft2d.go | 388 | `FFT2D` real packed 2D (`mat.Matrix`) + conv/corr |
| fft2dc.go | 386 | `FFT2DC` complex64 2D (tensor) + conv/corr |
| windows.go | 320 | 14 windows, `ApplyWindow*`, `CreateWindow*` |
| generators.go | 669 | `Generator`/`Correlator` interfaces; square, triangle, noise ×3, constant, ramp, Gaussian pulse |
| generators_trig.go | 634 | sine, cosine, sinc, raised cosine |
| generators_chirp.go | 800 | linear, log, pink-linear, pink-log chirps |
| measure.go | 477 | `Measure1D/2D/1DC/2DC`, `Goertzel`/`GoertzelC`, `CalculateRMSStatistics`/`…C` |

FFT kernels live in `x/math/primitive/fp32/dsp_fft.go` and `x/math/primitive/c64/dsp_fft.go` (radix-2 Cooley–Tukey, documented in the FFT notes).

## Public API

- **FFT objects** (value-returning constructors; call methods on an addressable variable): `NewFFT1D(n, lut...)`, `NewFFT1DC(n, lut...)`, `NewFFT2D(r, c, lut...)`, `NewFFT2DC(r, c, lut...)`; methods `Forward(dst, src)`, `Backward(dst, src)`, `Convolve(dst, a, b)`, `Correlate(dst, a, b)`. Sizes round up to powers of two; forward unscaled, inverse 1/N. LUTs from `fp32.NewFFTLUT(n)` / `c64.NewFFTLUT(n)`.
- **Façade** `NewFFT(shape, withLUT) FFT` → `Forward/Backward(dst, src types.Tensor)`; locks to the first dtype seen; rank-2 FP32 and rank > 2 panic.
- **Windows:** `WindowParams{Type, Param}`; `ApplyWindowVector(v, p)`, `ApplyWindowMatrix(m, p)`, `CreateWindowVector(n, p, dst)`, `CreateWindowMatrix(r, c, p, dst)`.
- **Generators:** `Generator{Generate, Add, GenerateWithOp(dst, reverse, op)}`, `Correlator{Correlate(dst, samples)}`; constructors `New{Sine,Cosine,Square,Triangle}Generator(fs, f, A[, φ])`, `NewNoise{White,Pink,Brown}Generator(A)`, `New{Chirp,LogarithmicChirp,PinkChirp,PinkLogarithmicChirp}Generator(fs, f0, f1, A)`, `NewSincGenerator`, `NewRaisedCosineGenerator`, `NewGaussianGenerator`, `NewConstantGenerator`, `NewRampGenerator`.
- **Measurements:** `Measure1D(vec) MeasurementResult`, `Measure2D(mat)`, `Measure1DC/2DC(tensor)`, `Goertzel(vec, f, fs) FrequencyMeasurement`, `GoertzelC`, `CalculateRMSStatistics(vec) RMSStatistics`, `…C`.

Composition: generators write/accumulate into caller `[]float32` (`Add` to mix tone + noise); windows mutate in place before `FFT1D.Forward`; Goertzel/measurements consume plain vectors. FFT types are not used anywhere outside their own tests (grep 2026-11-09); modulators use generators + Goertzel only.

## Algorithms

<!-- vault:algorithms h=5ec651b92e54 -->

Counts: 4 tested · 0 implemented · 7 partial · 0 stub · 0 unverified (11 algorithms)

| Algorithm | Status | Tests | Summary |
|---|---|---|---|
| [[fft-complex-1d]] | tested | pass | matches naive DFT; same LUT advice |
| [[fft-radix2-real]] | tested | pass | matches naive DFT 6e-8..1.4e-5 relErr; LUT 50× more accurate at 4096 |
| [[goertzel]] | tested | pass | equals DFT bin (amp, phase); integer bin only; empty → NaN |
| [[pulse-shape-generators]] | tested | pass | RC matches formula 1e-5; RC (not RRC) only |
| [[chirp-generators]] | partial | pass | `Correlate` not a matched filter (peak 28.6 vs 399.7); chunked sweeps overshoot; "pink" ≠ pink |
| [[fft-2d]] | partial | pass | `FFT2DC.Backward` no zero-padding; FP32 rank-2 façade panics |
| [[fft-convolution-correlation]] | partial | pass | 2D real conv multiplies packed halves element-wise; 2D complex and mis-sized 1D complex wrong; lag… |
| [[noise-generators]] | partial | pass | white = uniform (SNR +4.77 dB vs nominal as AWGN); pink diverges at sample ~13; brown Nyquist peak |
| [[signal-measurements]] | partial | pass | SNR constant 20 dB; THD always 0 |
| [[sine-oscillator-generators]] | partial | pass | float32 time accumulator → sine/cosine antiphase after ~1–60 s, block-size dependent |
| [[window-functions]] | partial | pass | 13 tapers exact; `ApplyWindowVector(Rectangular)` overwrites signal with 1; N=1 → NaN |

<!-- /vault:algorithms -->

Tally: tested 4, partial 7, implemented 0, stub 0, unverified 0. The FFT tensor façade (`FFT`) is a dispatcher, covered in [[fft-radix2-real]], [[fft-complex-1d]], [[fft-2d]]; it has no direct test (probe: FP32 1D equals `FFT1D`, C64 4×8 matches naive 2D DFT 9e-8).

## Build targets and dependencies

- Pure Go; imports `github.com/chewxy/math32`, `x/math/{primitive/fp32, primitive/c64, vec, mat, shape, tensor, tensor/types}`, `x/types`, `math/rand`. Builds on any GOOS/GOARCH Go supports [INFERENCE: no tags/CGO; only darwin/arm64 run here].
- float32 throughout (math32), except `RMSStatistics` (float64 sums). No float64 FFT.
- Allocation: FFT objects allocate scratch at construction; `FFT1D`/`FFT1DC`/`FFT2D` Forward/Backward are allocation-free; `FFT2DC` allocates a row buffer per call; the façade's 2D path builds a new `FFT2DC` per call; complex Convolve/Correlate allocate tensors and FFT objects per call; real Convolve/Correlate use `fp32.Pool`. Generators allocate only in `Correlate` when dst < 2N−1 and N > 64. `CreateWindowMatrix`, `Measure1DC/2DC` allocate.
- Embedded/TinyGo: no goroutines, no reflection beyond type assertions (`View().(vec.Vector)`, `dst.(mat.Matrix)`) and generic `GetTensorData`; float32 friendly for Cortex-M4F. Dependency on `x/math/tensor` (FFT1DC/2DC, façade, measure) may be heavy on MCUs; noise uses global `math/rand`. [INFERENCE: TinyGo not built — out of scope for this run.]

## Tests and evidence

- `GOTOOLCHAIN=local GOPROXY=off GOFLAGS=-mod=readonly go test -count=1 -v ./x/math/dsp/` → ok, 71 PASS (2026-11-09). Prior baseline 2026-11-08 same HEAD: ok, 71 pass.
- Numeric probe (`go run .`, 2026-11-09, ok, 8.7 s): FFT vs float64 naive DFT (1D real/complex, 2D façade), conv/corr vs naive, windows vs formula, noise statistics/PSD, sine drift, chirp reverse/correlate/chunking/octave energy, Goertzel vs DFT, SNR constant. Results in each algorithm note.
- Weak tests: most generator/window tests check ranges or "has variation"; `TestGenerator_StateTracking`, `TestChirpGenerator_StateContinuity` assert only `NotNil`; `TestFFT2D_BackwardDump` asserts nothing. 28 benchmarks in `generators_bench_test.go` not run.
- `SPEC.md` is stale: documents `NewSignalGenerator1D/2D`, `NewMeasurements`, `CrossCorrelate1D`, `EstimateShift2D` (none exist in this package); shows `fft1D.Forward(signal, spectrum)` (actual order is `(dst, src)`) with an N-length spectrum (needs 2N); claims recurrence-based oscillators (sine/cosine call `math32.Sin` per sample).

## API notes (ease of use, composability)

- **Inconsistent containers.** Real 1D takes `vec.Vector`, real 2D `matTypes.Matrix` (but hard-asserts `mat.Matrix`), complex takes `types.Tensor`, windows/measure take `vecTypes.Vector` but assert `vec.Vector`, generators take `[]float32`. Moving between them needs wrapping at each step; pick one (`[]float32` / `[]complex64` slices at the primitive layer, tensors in the façade).
- **Packed real spectrum `[re…|im…]`** differs from the `[]complex64` used by everything else and from the interleaved layout most libraries use; no helpers for magnitude, power, bin→Hz, or half-spectrum. A user doing a power spectrum must index `s[k]`, `s[k+N]` by hand.
- **Argument order is `(dst, src)`** for FFTs but **internally flipped** (`forward(src, dst)`), and SPEC shows the opposite → easy to misuse; the method names `Backward` (not `Inverse`) add friction.
- **Silent truncation/padding.** Inputs longer than N are truncated, short dst gets a truncated result, and non-power-of-2 sizes are padded without telling the caller the actual N (no `Len()` accessor).
- **Constructors return values** while methods have pointer receivers (mixed with value receivers on `FFT2DC`) → `NewFFT1D(n).Forward(...)` does not compile; must assign first.
- **Generators are the most composable part** (`Generate`/`Add`/`GenerateWithOp` + `reverse`), but: state is a float32 clock (block-size dependent), chirp duration = buffer length, no `Reset` except Ramp, no seed injection for noise, `Correlate` semantics differ per type and are O(N²). They do not implement the `chain.Processor` interfaces used by [[pipeline-and-chain]] / [[modulation]], so they are wired by hand in each modulator.
- **Measurements return fake fields** (`SNR`, `THD`) that look authoritative.
- Panics (not errors) for every misuse; acceptable for hot paths but surprising in `Convolve` length mismatches.

## TODOs

Algorithm defects are tracked in the algorithm notes (table above) and rolled up in [[todo]]; this list holds module-level items only.

- [ ] Rewrite SPEC.md against the real API (or delete stale sections) — x/math/dsp/SPEC.md:18-268 #p3
- [ ] Spectrum helpers (magnitude/power/bin frequency, half-spectrum) for the packed layout — x/math/dsp/fft1d.go:9-20 #p2
- [ ] Unify container types across FFT/window/measure/generator APIs — x/math/dsp/fft1d.go:64; x/math/dsp/fft1dc.go:66; x/math/dsp/windows.go:38-41 #p3

## Inferred objectives (proposals)

- **Proposal:** before tuning any modem noise figures (owner: "real audio pipeline not fully satisfy me with noise figures"), fix the measurement substrate first — Gaussian AWGN with exact σ, a real SNR/EVM estimator, block-size-invariant oscillators — then re-baseline BER vs Eb/N0 for each [[modulation]] scheme against theory. Several current figures are biased by +4.77 dB (uniform noise) and by LO phase drift.
- **Proposal:** give generators a sample-counter/phase-accumulator core (float64 or wrapped float32 phase) and a `Reset`/seek API, and expose them as `chain.Processor` sources so [[pipeline-and-chain]] can compose TX chains without bespoke glue.
- **Proposal:** collapse the four FFT types + façade into one slice-based API (`FFT{N}` with `Forward(dst, src []complex64)`, `RealForward(dst []complex64, src []float32)` half-spectrum), with tensor/vec adapters on top; drop or fix the 2D convolution until a consumer needs it.
- **Proposal:** replace O(N²) generator `Correlate` with FFT-based matched filtering for chirp sync in [[audio-modem-cli]] once [[fft-convolution-correlation]] is fixed.
