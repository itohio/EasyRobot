---
type: algorithm
status: implemented
tags: [easyrobot, filter, fir, dsp]
workstream: W7
module: "[[filters]]"
summary: "Correct by probe"
source: [x/math/filter/fir/fir.go]
tests: [x/math/filter/fir/fir_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "fir 8/8 pass (NaN checks only); probe dspprobe: LP/HP 1 kHz @8 kHz order 64 → −6.0 dB at cutoff, ≥54 dB stopband"
---
# FIR filter core + windowed-sinc low/high-pass

**What it does:** Direct-form FIR with circular buffer (`Process`, `ProcessBuffer`), plus Hamming-windowed sinc design for low-pass and spectral-inversion high-pass.

**Where:** `x/math/filter/fir/fir.go`: `FIR` :18-28, `New` :31, `Process` :58-77, `ProcessBuffer` :81-89, `NewLowPass` :120-146, `NewHighPass` :149-180, `sinc`/`hamming` :107-117.

**Inputs → outputs:** `order` (taps = order+1), cutoff Hz, fs Hz → `*FIR`; per-sample float32 in/out. No allocation per sample.

## Completeness review

- Complete: yes for LP/HP. DC gain of the low-pass is not normalised to exactly 1 (:134-140; probe shows 0.0 dB anyway at order 64); odd `order` gives an asymmetric kernel (`halfOrder = order/2`, :132) — a non-linear-phase filter, and an odd-length high-pass is not possible as type II.
- Tested: `TestFIRBasic`, `TestFIRLowPass`, `TestFIRHighPass`… only assert the first output is not NaN/Inf (`fir_test.go:33-59`). Run: go test -count=1 -json ./x/math/filter/fir (2026-11-09, raw tmp/easyrobot/filters/test-2026-11-09.json) → ok, 8 pass.
- Correct: yes (probe). `tmp/easyrobot/filters/dspprobe/main.go`, fs=8 kHz, order 64, cutoff 1 kHz: LP 0.0 dB at 50–500 Hz, −6.0 dB at 1 kHz, −60 dB at 1.5 kHz; HP −65 dB at 50 Hz, −6.0 dB at 1 kHz, 0.0 dB above 1.5 kHz (standard windowed-sinc behaviour: −6 dB at fc).
- Performance: O(taps) per sample with a modulo-free branch; `GetCoeffs` allocates. No SIMD/fixed-point path.

## TODOs

- [ ] Frequency-response tests (gain at pass/stop frequencies) — fir_test.go #p2
- [ ] Normalise LP DC gain; reject/handle odd order for HP — fir.go:132-140,161 #p3
- [ ] Block `ProcessBuffer` without per-sample modulo (double-length delay line) for ESP32 throughput #p3

## Links

- Module: [[filters]]
- Broken band designs: [[fir-band-design]]
