---
type: algorithm
status: tested
tags: [easyrobot, dsp, goertzel, tone-detection, audio]
workstream: W7
module: "[[dsp-core]]"
summary: "equals DFT bin (amp, phase); integer bin only; empty → NaN"
source: [x/math/dsp/measure.go]
tests: [x/math/dsp/measure_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "go test -count=1 -v ./x/math/dsp/ → ok, TestMeasurements_Goertzel/_GoertzelC PASS (2026-11-09); probe: amplitude and phase equal naive DFT bin (2026-11-09)"
---
# Goertzel single-bin DFT (real and complex)

**What it does:** evaluates one DFT bin X[k] with the second-order Goertzel recursion `s[n] = x[n] + 2cos(ω)·s[n−1] − s[n−2]`, then `X[k] = e^{jω}·s[N−1] − s[N−2]`; returns amplitude `2|X|/N` and phase `arg X`.

**Where:** `Goertzel(signal vecTypes.Vector, targetFreq, sampleRate float32) FrequencyMeasurement` x/math/dsp/measure.go:103-144; `GoertzelC(signal types.Tensor, …)` (C64 rank 1) :351-398. Result type `FrequencyMeasurement{Amplitude, Phase, Frequency}` :24-28.

**Inputs → outputs:** float32 samples, target Hz, fs Hz. k = round(N·f/fs), clamped to N−1 (measure.go:108-111) — the frequency is **quantised to the nearest bin**, and `Frequency` just echoes the request. Amplitude `2|X|/N` is the peak amplitude of a real tone exactly on a bin; for complex input a single tone `A·e^{jωn}` yields 2A (normalisation is the real-signal one) [derived, not probed].

## Completeness review

- Complete: yes for integer-bin Goertzel. No fractional-k (generalised) Goertzel, no window option, no streaming/sliding form; `GoertzelC` constructs `coeffComplex` inside the loop (measure.go:375) — trivial cost.
- Tested (PASS 2026-11-09): `TestMeasurements_Goertzel` — 10 Hz sine at fs 1 kHz, asserts amplitude > 0.9 and `Frequency`≈10 (trivially true); `TestMeasurements_GoertzelC` same for a complex tone. Neither checks phase or off-bin behaviour.
- Correct: yes. Probe vs naive DFT bin (N=100, fs 1 kHz, cos with phase 0.3): on-bin 10 Hz → amp 1.0000, phase 0.3000 (DFT identical); off-bin 12.5 Hz → k=1, amp 0.8549, phase 0.9758 (DFT identical — scalloping loss inherent to integer k). Edge: empty input → `{NaN, NaN}` (k=0, ω = 0/0, measure.go:113).
- Use in-repo: `x/math/dsp/modulation/mfsk/demodulator.go` (tone decisions), `cmd/audio/recv_chirp.go` (reference-tone detection).

## TODOs

- [ ] Empty signal returns NaN amplitude/phase; return zero value — x/math/dsp/measure.go:105-113 #p2
- [ ] Offer fractional-bin (generalised) Goertzel so MFSK tones need not sit on bins; report the actual bin frequency k·fs/N — x/math/dsp/measure.go:108-113,142 #p2
- [ ] Test phase and an off-bin tone — x/math/dsp/measure_test.go:104-120 #p3

## Links

- Module: [[dsp-core]]
- Consumers: [[modulation]] (mfsk), [[audio-modem-cli]]
- Full-spectrum alternative: [[fft-radix2-real]]; windowing: [[window-functions]]
