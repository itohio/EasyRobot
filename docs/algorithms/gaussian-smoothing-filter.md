---
type: algorithm
status: partial
tags: [easyrobot, filter, dsp]
workstream: W7
module: "[[filters]]"
summary: "Rotated kernel, passthrough"
source: [x/math/filter/gaussian/gaussian.go]
tests: []
test_result: none
complete: yes
correct: no
evidence: "no tests; probe dspprobe: impulse response 1.000 0.054 0.054 0.244 0 … (peak misplaced)"
---
# Gaussian smoothing filter (streaming FIR)

**What it does:** FIR with normalised Gaussian kernel `exp(−x²/2σ²)` over an odd window, applied to a ring buffer.

**Where:** `x/math/filter/gaussian/gaussian.go`: `New` :24, `Process` :58-90, `ProcessBuffer` :94-107, `computeCoefficients` :118-139.

**Inputs → outputs:** σ (samples), window → float32 per sample.

## Completeness review

- Complete: API yes; coefficient design correct (:118-139).
- Tested: no test file (`[no test files]` in go test -count=1 -json ./x/math/filter/... (2026-11-09)).
- Correct: **no**.
  - Same warm-up gate bug as [[median-filter]]: `gaussian.go:65` uses ring index instead of sample count → periodic raw passthrough.
  - Kernel/buffer misalignment: `:74` starts at `index − halfWindow`, which is *not* the oldest sample (oldest is `index`), so the symmetric kernel is rotated around the ring. Probe (σ=1, w=5, unit impulse): output `1.000, 0.054, 0.054, 0.244, 0, …` (expected 0.054, 0.244, 0.403, 0.244, 0.054 delayed by 2); magnitude response non-monotonic (−3.1 dB @1 kHz, −2.0 dB @2 kHz, −3.8 dB @3 kHz at fs 8 kHz).
  - Renormalises by Σcoeff each sample (:83-86) although coefficients are already normalised.

## TODOs

- [ ] Index from oldest sample (`(index+i) % w`) and gate on sample count — gaussian.go:65,74 #p1 #ER-019
- [ ] Tests: impulse response equals kernel, DC gain 1 #p2
- [ ] Consider replacing with `fir.New(coeffs...)` (shared FIR core) #p3

## Links

- Module: [[filters]]
