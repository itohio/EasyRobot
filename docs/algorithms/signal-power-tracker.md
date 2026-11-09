---
type: algorithm
status: implemented
tags: [easyrobot, dsp, modulation]
workstream: W7
module: "[[modulation]]"
summary: "indirect"
source: [x/math/dsp/modulation/internal/power_tracker.go]
tests: [x/math/dsp/modulation/mask/roundtrip_test.go]
test_result: fail
complete: yes
correct: unknown
evidence: "used by mask demodulator; exercised indirectly by mask tests (4 pass, 2 fail on unrelated assertions), 2026-11-09"
---
# Signal power tracker (EMA / peak / min / max)

**What it does:** a generic (real or complex) power tracker. `Update(samples)` computes mean |x|². `UpdatePower(p)` feeds a value directly. It keeps the EMA (α), a peak with decay, min/max and a count. Non-positive powers are **ignored** (`power_tracker.go:49-52`).

**Where:** `internal/power_tracker.go:24-178`; used in `mask/demodulator.go:68,93,206,221`.

## Completeness review

- Complete: yes. Ignoring zero power means digital silence doesn't pull the background estimate down; this is intended, but it biases the noise-floor estimate when the input is gated. Min/max never decay.
- Tested: indirectly through mask; no unit tests. Correct: unknown (no value checks).

## TODOs

- [ ] Unit test the EMA/peak maths — internal/power_tracker.go #p3

## Links

- Module: [[modulation]] · [[mask-modem]] · [[complex-agc]]
