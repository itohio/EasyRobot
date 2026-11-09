---
type: algorithm
status: implemented
tags: [easyrobot, dsp, modulation]
workstream: W7
module: "[[modulation]]"
summary: "—"
source: [x/math/dsp/modulation/internal/iq_imbalance.go]
tests: []
test_result: none
complete: yes
correct: unknown
evidence: "read only; no tests, no callers (2026-11-09)"
---
# I/Q imbalance corrector (internal, unused)

**What it does:** applies a 2×2 correction matrix [a b; c d] to (I, Q). It is built from a known gain imbalance (dB) and phase imbalance (rad), or adapts b and c by gradient descent to decorrelate I and Q (`error = I'·Q'`).

**Where:** `internal/iq_imbalance.go:31-95` constructors, `:97-120` `Process`, `:132-151` accessors.

## Completeness review

- Complete: yes for its scope. The adaptive mode only decorrelates (phase skew); it doesn't equalise gain. Relevance to EasyRobot: none so far, since the I/Q here is computed digitally (no analogue mixer). It would matter only for stereo-cable I/Q (`gen_psk` writes I/Q as L/R) or for SDR front ends.
- Tested: none. Correct: unknown.

## TODOs

- [ ] Keep only if an analogue I/Q path is planned; otherwise delete — internal/iq_imbalance.go #p3

## Links

- Module: [[modulation]] · [[W7-filters-codecs-voice]]
