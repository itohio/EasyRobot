---
type: algorithm
status: partial
tags: [easyrobot, dsp, modulation, synchronisation]
workstream: W7
module: "[[modulation]]"
summary: "probe: 9277 Hz for a 100 Hz offset"
source: [x/math/dsp/modulation/internal/afc.go]
tests: []
test_result: none
complete: no
correct: no
evidence: "probe on a copy (2026-11-09): pure 100 Hz complex tone, fs 8 kHz → estimated offset 9277.5 Hz"
---
# AFC frequency tracker (internal, unused)

**What it does (intended):** estimates carrier frequency offset from the sample-to-sample phase derivative with a PI update, and derotates with an accumulated phase.

**Where:** `internal/afc.go:40-76` constructors, `:84-134` `Process`, `:147-170` getters/setters.

## Completeness review

- **Correct: no.**
  - The derivative of the raw sample phase includes the data modulation; nothing removes it (no M-th power, no decision direction).
  - Frequency is scaled by `symbolRate` while it is updated per sample (`:104-108`).
  - The correction's `sin/cos` are refreshed only when the phase accumulator wraps (`:122-130`, `updateTrig` at `:125,129`), so between wraps a **constant** rotation is applied.
  - The estimate integrates without a bound: the probe gives 9277.5 Hz for a true 100 Hz offset.
- Tested: none. Unused by any modem.

## TODOs

- [ ] Recompute sin/cos every sample (or use a complex NCO), fix the units, add modulation removal, and test against a known CFO — internal/afc.go:84-134 #p1 #ER-022

## Links

- Module: [[modulation]] · [[complex-signal-primitives]] (CFO corrector, same stale-trig bug) · [[W7-filters-codecs-voice]]
