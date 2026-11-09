---
type: algorithm
status: partial
tags: [easyrobot, filter, fir, dsp]
workstream: W7
module: "[[filters]]"
summary: "Wrong band impulse response"
source: [x/math/filter/fir/fir.go]
tests: [x/math/filter/fir/fir_test.go]
test_result: pass
complete: yes
correct: no
evidence: "probe dspprobe: BP 500-1500 Hz gives +20 dB at 50 Hz; BS gives +33 dB at 50 Hz"
---
# FIR band design (general band, band-pass, band-stop)

**What it does:** Builds an FIR from a list of `BandSpec{StartFreq, EndFreq, Gain, Weight}` by summing ideal band impulse responses with a Hamming window; `NewBandPass`/`NewBandStop` are three-band wrappers.

**Where:** `x/math/filter/fir/fir.go`: `BandSpec` :8-13, `NewGeneralBand` :184-227, `NewBandPass` :230-243, `NewBandStop` :246-259.

**Inputs → outputs:** order, bands (Hz), fs → `*FIR`.

## Completeness review

- Complete: no — `Weight` is ignored (no least-squares/Remez), comment admits approximation (:192-194).
- Tested: `TestFIRBandPass`/`TestFIRBandStop` assert only non-NaN first output. Run: go test -count=1 -json ./x/math/filter/fir (2026-11-09, raw tmp/easyrobot/filters/test-2026-11-09.json) → ok.
- Correct: **no**. The ideal band response is `h[n] = G·(2f₂·sinc(2f₂n) − 2f₁·sinc(2f₁n))`; `fir.go:216` uses `G·2·(sinc(2f₂n) − sinc(2f₁n))` (missing the `f₂`, `f₁` factors) while the n=0 term (:213) is right. Probe (`dspprobe`, fs 8 kHz, order 64): band-pass 500–1500 Hz → +20.3 dB at 50 Hz, +14.9 dB at 1 kHz, −12 dB above 2 kHz; band-stop → +33.5 dB at 50 Hz. Neither behaves as specified.

## TODOs

- [ ] Fix band impulse response (multiply sinc terms by 2f) — fir.go:216 #p1 #ER-019
- [ ] Response tests for BP/BS (in-band ≈0 dB, out-of-band < −40 dB) — fir_test.go #p1 #ER-019
- [ ] Use `Weight` (frequency-sampling LS or Parks–McClellan) or drop the field #p3

## Links

- Module: [[filters]]
- Working designs: [[fir-windowed-sinc-lowpass-highpass]]
