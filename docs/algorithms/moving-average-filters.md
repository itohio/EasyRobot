---
type: algorithm
status: tested
tags: [easyrobot, filter, dsp]
workstream: W7
module: "[[filters]]"
summary: "`Filter` = running-sum SMA over a ring buffer (averages over the samples seen so far until the…"
source: [x/math/filter/ma/ma.go]
tests: [x/math/filter/ma/ma_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "ma 8/8 pass (value checks)"
---
# Moving average (SMA) and exponential moving average (EMA)

**What it does:** `Filter` = running-sum SMA over a ring buffer (averages over the samples seen so far until the window fills); `Exponential` = EMA `y = αx + (1−α)y`, seeded with the first sample; helpers `AlphaFromHalfLife`, `AlphaFromTimeConstant`.

**Where:** `x/math/filter/ma/ma.go`: `New` :19, `Process` :41-59, `NewEMA` :70, `Exponential.Process` :87-98, `AlphaFromHalfLife` :115-120 (`1−0.5^(1/h)`), `AlphaFromTimeConstant` :124-129 (`1−e^(−1/τ)`).

**Inputs → outputs:** window size / α / half-life (samples) → float32 per sample. Zero allocations per sample; implements `filter.Processor[float32]`.

## Completeness review

- Complete: yes.
- Tested: `TestMovingAverage` (partial-window and full-window values), resets, EMA values, alpha helpers, processor interface (`ma_test.go:11-151`). Run: go test -count=1 -json ./x/math/filter/ma (2026-11-09) → ok, 8 pass.
- Correct: yes for tested values; formulas for α helpers are the standard ones. Running float32 sum (:43-47) accumulates rounding drift on long streams (never re-summed) — negligible for short runs, visible after ~10⁷ samples of large-magnitude data [INFERENCE].

## TODOs

- [ ] Periodic re-summation or Kahan sum for long-running SMA — ma.go:43-47 #p3

## Links

- Module: [[filters]]
