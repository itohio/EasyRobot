---
type: algorithm
status: partial
tags: [easyrobot, dsp, generator, chirp, audio]
workstream: W7
module: "[[dsp-core]]"
summary: "`Correlate` not a matched filter (peak 28.6 vs 399.7); chunked sweeps overshoot; \"pink\" ≠ pink"
source: [x/math/dsp/generators_chirp.go]
tests: [x/math/dsp/generators_test.go]
test_result: pass
complete: no
correct: no
evidence: "go test -count=1 -v ./x/math/dsp/ → ok (chirp tests assert 'has variation' / peak > 10 only, 2026-11-09); probe tmp/easyrobot/audio/dspcore/main.go: Correlate not a matched filter, chunked sweep overshoots, pink variants not pink (2026-11-09)"
---
# Chirp generators: linear, logarithmic, "pink" linear, "pink" logarithmic

**What it does:** frequency sweeps f0 → f1 over the length of the buffer passed in:
- Linear: φ(t) = 2π(f0·t + ½k·t²), k = (f1−f0)/T.
- Logarithmic (exponential): f(t) = f0·(f1/f0)^(t/T), phase accumulated per sample.
- Pink linear / pink log: same sweeps × exponential amplitude envelope `exp(K·t)`, `K = ln(Kend)/T`, `Kend = 10^(−0.3·log2(fMax/fMin))` (−6 dB amplitude per octave of span).
- `Correlate(dst, samples)` (linear, log): direct O(N²) convolution of `samples` with a "reversed" reference, for chirp detection/ranging.

**Where:** x/math/dsp/generators_chirp.go
- `ChirpGenerator` :8-213 (`GenerateWithOp` :36-126, `Correlate` :160-213).
- `LogarithmicChirpGenerator` :216-445 (`Correlate` :397-445).
- `PinkChirpGenerator` :450-608; `PinkLogarithmicChirpGenerator` :613-800 (no `Correlate`).

**Inputs → outputs:** `(sampleRate, f0, f1, amplitude)` float32. **T is `len(dst)/fs` of the current call**, so the sweep duration is defined by the buffer size of each call. Log/pink variants return without writing when f ≤ 0.

## Completeness review

- Complete: no — no duration parameter (cannot stream a long chirp in blocks), no `Correlate` on pink variants (cmd/audio falls back to the log generator, `cmd/audio/recv_chirp.go:229-233`), no FFT correlation.
- Tested (PASS 2026-11-09): `TestChirpGenerator_Generate`, `TestLogarithmicChirpGenerator_Generate`, `TestPink*_Generate` ("some sample differs from the first"), `TestChirpGenerator_Correlate` / `TestLogarithmicChirpGenerator_Correlate` (max |corr| > 10), `TestPink*_GenerateWithOp` (|x| ≤ 1), `_Add` (values changed), `_Reverse` (differs from forward), `_EnvelopeDecreases` (late RMS < 1.5 × early — passes even if it increases up to 50 %), `TestChirpGenerator_StateContinuity` (asserts only `NotNil`). No test checks frequency, phase, or correlation peak position.
- Correct: no (probe, fs 8 kHz, 200 → 2000 Hz, N = 800 unless noted):
  - **`Correlate` is not a matched filter.** It builds `NewChirpGenerator(fs, f1, f0)` and then calls `GenerateReverse`, which swaps again (generators_chirp.go:187-188; log :420-421) → the reference is the *forward* chirp, starting at phase 0. Self-correlation peak 28.6 at index 1523 vs ideal matched-filter peak 399.7 (= signal energy) at 799; log chirp 48.3 at 122 vs 400.1 at 799. `cmd/audio/recv_chirp.go:224-233` relies on this for chirp detection.
  - **`GenerateReverse` ≠ time-reversed `Generate`**: it sweeps f1→f0 from phase 0 (generators_chirp.go:53-87); max |diff| vs reversed forward = 1.82.
  - **Chunked generation overshoots.** Each call recomputes k (linear, :92-94) or the ratio step (log, :331-332) from the current buffer length while `g.time` keeps running → a second block sweeps past f1. 2×400 vs 1×800: max diff 1.999; zero crossings in last 100 samples 89 vs 46 (linear), 54 vs 43 (log).
  - **"Pink" variants are not pink.** Energy per octave 100–6400 Hz (fs 16 kHz, 65 536 samples), dB relative to first octave: plain log `[0 0.08 0.08 0.08 0.08 0.06]` (already pink: dwell time ∝ 1/f); pink-log `[0 −5.9 −11.9 −17.9 −23.9 −29.9]` (1/f³ PSD); pink-linear `[0 2.5 3.8 3.5 0.2 −8.4]` (envelope exponential in t, not a power law in f); plain linear `[0 3.3 6.3 9.3 12.3 15.3]` (white). For a pink linear sweep the amplitude should follow f^−½; for a log sweep no envelope is needed. Downward sweeps (f0 > f1) still start the envelope at 1 at f0 → louder highs.
  - Linear phase increment adds `2π·k·dt²` instead of `π·k·dt²` per step (:70,104) → constant frequency offset k·dt/2 (probe case: k = 18 000 Hz/s, dt = 1/8000 → 1.1 Hz) — small vs sweep span [derived, not probed].
  - Float32 `time`/phase state shares the long-run drift class of [[sine-oscillator-generators]] (log/pink-log keep a wrapped phase, better).

## TODOs

- [ ] `Correlate` must use the true time-reversed (conjugate) reference: drop the f0/f1 swap or use `Generate` + reverse — x/math/dsp/generators_chirp.go:187-188,420-421 #p1 #ER-022
- [ ] `GenerateReverse` should equal time-reversed `Generate` (phase φ(T−t)) — x/math/dsp/generators_chirp.go:49-88,258-309 #p1 #ER-022
- [ ] Add a duration (or total samples) parameter; derive k / ratio from it, not from `len(dst)` — x/math/dsp/generators_chirp.go:46,92-94,252,331-332 #p1 #ER-022
- [ ] Pink envelopes: linear sweep A ∝ f^−½, log sweep no envelope; handle f0 > f1 — x/math/dsp/generators_chirp.go:510-516,663-669 #p2
- [ ] Correlate is O(N²) (recv_chirp correlates whole chirp regions); use FFT correlation — x/math/dsp/generators_chirp.go:196-210 #p2
- [ ] Fix ½ factor in linear phase increment — x/math/dsp/generators_chirp.go:70,104 #p3
- [ ] Tests: instantaneous frequency vs time, block-size invariance, matched-filter peak = energy at lag N−1, octave energy slope — x/math/dsp/generators_test.go:58-130,492-506 #p2

## Links

- Module: [[dsp-core]]
- Consumers: [[audio-modem-cli]] (`gen_chirp.go`, `recv_chirp.go`)
- Siblings: [[sine-oscillator-generators]], [[noise-generators]], [[pulse-shape-generators]]; FFT correlation: [[fft-convolution-correlation]]
