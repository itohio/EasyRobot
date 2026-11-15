---
type: algorithm
status: partial
tags: [easyrobot, dsp, generator, oscillator, audio]
workstream: W7
module: "[[dsp-core]]"
summary: "float32 time accumulator → sine/cosine antiphase after ~1–60 s, block-size dependent"
source: [x/math/dsp/generators_trig.go, x/math/dsp/generators.go]
tests: [x/math/dsp/generators_test.go, x/math/dsp/generators_bench_test.go]
test_result: pass
complete: yes
correct: no
evidence: "go test -count=1 -v ./x/math/dsp/ → ok (2026-11-09); probe (sine phase drift); DSPAudio probe (cosine block-size dependence) (2026-11-09)"
---
# Periodic oscillators: sine, cosine, square, triangle (+ constant, ramp)

**What it does:** stateful sample generators for `A·sin(2πft+φ)`, `A·cos(…)`, ±A square (50 % duty), triangle, constant and linear ramp. All implement `Generator` (`Generate`, `Add`, `GenerateWithOp(dst, reverse, op)`); sine/cosine also `GenerateReverse`, `AddReverse`, `Correlate`.

**Where:**
- `SineGenerator` x/math/dsp/generators_trig.go:8-173; `CosineGenerator` :176-336 (forward phase `ω·g.time+φ` at :73 / :240, state `g.time += n·dt` at :94 / :258).
- `SquareGenerator` x/math/dsp/generators.go:31-126; `TriangleGenerator` :129-218; `ConstantGenerator` :444-492; `RampGenerator` :495-559 (`Reset`).
- Interfaces `Generator`, `Correlator`, `GeneratorAssignOp` x/math/dsp/generators.go:9-28.

**Inputs → outputs:** constructors take float32 `(sampleRate, frequency, amplitude[, phase rad])`; methods fill `dst []float32`, return `len(dst)`. `reverse=true` emits the *next* n samples time-reversed and still advances state. `Correlate(dst, samples)` = direct O(N·M) convolution of `samples` with the reversed reference of length `len(samples)`; dst ≥ 2N−1 lets it use dst's tail as scratch, otherwise a 64-float stack buffer or `make`.

## Completeness review

- Complete: yes for the stated scope. Per-sample `math32.Sin/Cos` (SPEC.md:189 claims recurrence-based updates — false for these). `SquareGenerator.period`/`TriangleGenerator.period` fields are computed in samples and never used (generators.go:41,139).
- Tested (PASS 2026-11-09): `TestSinusoidGenerator_Generate` (sin(0)=0, sample 7 ≈ −√2/2, range), `TestCosineGenerator_Generate` (cos(0)=1), `TestSquareGenerator_Generate` (dst[0]=1, dst[4]=−1), `TestTriangleGenerator_Generate` (periodicity, dst[7]≈−0.5), `TestConstantGenerator_Generate`, `TestRampGenerator_Generate/_Reset` (exact values), `TestSinusoidGenerator_Correlate` (peak > 10 only), `TestSine/CosineGenerator_PhaseWrapping` (range only), `TestGenerator_StateTracking` (asserts only `NotNil` — checks nothing). 28 benchmarks in generators_bench_test.go (not run). No test spans more than 100 samples or checks continuity across calls.
- Correct: **no for streaming use.**
  - **Phase from a float32 time accumulator.** Sine/cosine recompute `phase = ω·g.time + φ` at each call with `g.time` float32 advanced by `n·dt` (generators_trig.go:73,94,240,258). Rounding of `g.time` grows with run time and depends on block size. Probe (sine, fs 48 k, f 1 kHz, 1024-sample blocks) max abs error vs float64 ideal: 1 s 0.001, 5 s 0.10, 60 s 1.77, 600 s 0.43, 3600 s 0.61 (|err| ≈ 2 = antiphase). DSPAudio cosine probe (fs 96 k, fc 20 k): blocks of 12 vs 96 are antiphase (max diff 2.000) within 1 s; block-96 vs ideal 1.098 after 1 s, 2.000 after 10 s. Consequence (DSPAudio BER probe): TX/RX local oscillators in [[modulation]] (dsss modulator/demodulator, mask coherent detector use `NewCosineGenerator`) lose coherence over long packets — DSSS BER ≈ 0.2–0.29 noiseless on 1000-symbol packets.
  - **Square/triangle accumulate `g.time += dt` per sample** in float32 (generators.go:107,192) → the effective dt quantises as `g.time` grows (ulp(20 s) ≈ 1.9 µs vs dt = 20.8 µs at 48 kHz) → frequency error rising with run time [INFERENCE from float32 ulp; not probed].
  - Short-buffer values are right (tests above).

## TODOs

- [ ] Replace float32 `g.time` with a wrapped phase accumulator (or float64 sample counter) in Sine/Cosine; must be block-size independent — x/math/dsp/generators_trig.go:73,94,240,258 #p1 #ER-022
- [ ] Same for Square/Triangle per-sample `g.time += dt` — x/math/dsp/generators.go:86,107,177,192 #p1 #ER-022
- [ ] Continuity test: N blocks of varying size vs one block vs float64 ideal over ≥10 s — x/math/dsp/generators_test.go:475-490 #p2
- [ ] `Correlate` is O(N·M) time-domain; route long references through [[fft-convolution-correlation]] once fixed — x/math/dsp/generators_trig.go:159-170 #p2
- [ ] Remove unused `period` fields; fix SPEC.md recurrence claim — x/math/dsp/generators.go:41,139; x/math/dsp/SPEC.md:189 #p3

## Links

- Module: [[dsp-core]]
- Siblings: [[chirp-generators]], [[noise-generators]], [[pulse-shape-generators]]
- Consumers: [[modulation]] (mask, dsss LOs), [[audio-modem-cli]] (`gen_sine.go`, `gen_chirp.go`)
