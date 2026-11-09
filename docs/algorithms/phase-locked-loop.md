---
type: algorithm
status: partial
tags: [easyrobot, filter, pll, dsp, modem]
workstream: W7
module: "[[filters]]"
summary: "Single-wrap phase error"
source: [x/math/filter/pll/pll.go]
tests: [x/math/filter/pll/pll_test.go]
test_result: fail
complete: yes
correct: no
evidence: "pll 9 pass, 2 fail (TestLoop_FrequencyTracking, TestLoop_Rotate) + TestLoop_NoiseLowSNR/5dB_Medium flaky; probe pllprobe: repo CFO 1010 Hz vs 10 Hz with full modulo wrap"
---
# Phase-locked loop (2nd-order, PI loop filter)

**What it does:** Carrier phase/frequency tracker for coherent demodulators: `freq += ki·e; phase = wrap(phase + kp·e + freq)`; gains from loop bandwidth/damping (`kp=2ζωn`, `ki=ωn²`, ωn normalised by symbol rate); helpers to de-rotate IQ samples and report CFO in Hz.

**Where:** `x/math/filter/pll/pll.go`: `Loop` :7-18, `New` :21, `NewFromLoopBandwidth` :31-43, `SetLoopBandwidth` :46-59, `Update` :62-68, `FrequencyOffset` :76-78, `Rotate`/`RotateInterleaved` :81-90, `wrapPi` :100-108.

**Inputs → outputs:** phase error (rad) per symbol → phase estimate; CFO = freq·Rs/2π. Zero allocations; float32.

## Completeness review

- Complete: yes for a type-2 loop driven by an external phase detector. No built-in phase detectors (Costas/decision-directed are in callers), no lock detector.
- Tested (run go test -count=1 -json ./x/math/filter/pll (2026-11-09, raw tmp/easyrobot/filters/test-2026-11-09.json) → FAIL, 9 pass / 2 fail; baseline 2026-11-08 same):
  - `TestLoop_FrequencyTracking` fails (`pll_test.go:27-42`, "PLL CFO mismatch diff=999.999634"). **Implementation defect:** the test feeds unwrapped `actual − loop.Phase()` (grows to ~63 rad); `wrapPi` (:100-108) subtracts 2π only once, so the error is not reduced to (−π, π], and `freq` locks to an aliased value. Probe (`tmp/easyrobot/filters/pllprobe`): repo loop reports **1010 Hz**; identical loop with `math32.Remainder(x, 2π)` wrapping reports **10.000 Hz** (expected 10). `FrequencyOffset` also never wraps `freq`.
  - `TestLoop_Rotate` fails (:44-52, got (0.985, −0.172)). **Test defect:** with kp=0.1, ki=0.01 one `Update(π/2)` moves phase to 0.173 rad (kp·e+ki·e); the test expects a full π/2 jump.
  - `TestLoop_NoiseLowSNR/5dB_Medium` flaky (:513-595): noise from `dsp.NoiseWhiteGenerator` uses unseeded global `rand.Float32()` (uniform, `x/math/dsp/generators.go:252`); measured std 0.1365 vs limit 0.13. Rough linear-theory jitter σ² ≈ 2·B_L·σn² with B_L≈0.067 (normalised), σn²=a²/3≈0.105 → σ≈0.12, so the 0.13 limit is within noise of the expected value [INFERENCE].
  - Passing tests (step response, noise convergence/sweeps) check convergence thresholds.
- Correct: loop equations are the standard discrete PI type-2 loop (small-ωn approximation of gains); phase wrapping is wrong for |error| > 3π.

## TODOs

- [ ] Full modulo wrap in `wrapPi`; wrap `freq` to (−π, π] — pll.go:100-108,64 #p1 #ER-019
- [ ] Fix `TestLoop_Rotate` expectation (or use kp=1 for the rotation check) — pll_test.go:44-52 #p2
- [ ] Seed the noise source in tests (inject `*rand.Rand` into `NoiseWhiteGenerator`) and derive thresholds from theory — pll_test.go:513-595 #p2
- [ ] Exact gain mapping (Gardner/Rice: kp,ki from ωn·T with denominator 1+2ζθ+θ²) for wide loops #p3

## Links

- Module: [[filters]]; consumers in demodulators: [[modulation]]
