---
type: algorithm
status: implemented
tags: [easyrobot, filter, iir, dsp]
workstream: W7
module: "[[filters]]"
summary: "Correct by probe"
source: [x/math/filter/iir/iir.go]
tests: [x/math/filter/iir/iir_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "probe dspprobe: BP 500-1500 Hz @8 kHz → -0.3 dB at 1 kHz, -3.2/-3.8 dB at edges; BS → -11.6 dB at 1 kHz, ~0 dB away"
---
# RBJ biquad band-pass / band-stop (named "Butterworth")

**What it does:** Second-order band-pass (constant 0 dB peak) and notch from Robert Bristow-Johnson's Audio-EQ cookbook, centre = geometric mean of edges, Q = f₀/BW.

**Where:** `x/math/filter/iir/iir.go`: `NewButterworthBandPass` :194-223, `NewButterworthBandStop` :226-255 (run by DF-II `Process` :75-99).

**Inputs → outputs:** order (must be even ≥2, otherwise ignored), low/high cutoff Hz, fs → `*IIR` with 3 b + 2 a coefficients.

## Completeness review

- Complete: yes as a single biquad; misnamed (not Butterworth) and `order` > 2 is accepted but ignored (:195, :202).
- Tested: first-sample NaN checks only (`iir_test.go`). Run: go test -count=1 -json ./x/math/filter/iir (2026-11-09) → ok.
- Correct: yes (probe, fs 8 kHz, 500–1500 Hz): BP −3.2 dB @500 Hz, −0.3 dB @1 kHz, −3.8 dB @1.5 kHz, −24 dB @50 Hz; BS −11.6 dB @1 kHz, ≤0.3 dB at 50/200/3000 Hz. Coefficients match the cookbook formulas (b=[α,0,−α]/a0, a=[1,−2cosω₀,1−α]/a0).

## TODOs

- [ ] Rename (`NewBiquadBandPass`/`NewNotch`) or implement true Butterworth band filters with order support — iir.go:194,226 #p3
- [ ] Response tests (probe values) — iir_test.go #p2

## Links

- Module: [[filters]]
